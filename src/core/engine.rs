use async_stream::stream;
use futures::{Stream, StreamExt};
use std::sync::Arc;

use crate::core::agent::Agent;
use crate::core::persist::PersistTask;
use crate::store::conversation::ConversationStore;
use crate::store::response::ResponseStore;
use crate::store::translator::{normalize_input, resolve_tool_choice, resolve_tools, to_input_items};
use crate::types::responses::{InputItem, ResponsesInput, ResponsesRequest, ResponsesResponse, StreamEvent};
use crate::utils::errors::AgenticApiError;

const DONE_MARKER: &str = "data: [DONE]\n\n";
const TERMINAL_EVENT_TYPES: &[&str] = &["response.completed", "response.failed", "response.incomplete"];

type Result<T> = std::result::Result<T, AgenticApiError>;
type BoxStream = std::pin::Pin<Box<dyn Stream<Item = String> + Send>>;

struct RequestContext {
    request_body: ResponsesRequest,
    new_input_items: Vec<InputItem>,
    our_response_id: String,
    conversation_id: Option<String>,
}

impl RequestContext {
    fn inject_ids(&self, response: &mut ResponsesResponse, prev_resp_id: Option<String>) {
        response.id.clone_from(&self.our_response_id);
        response.conversation_id.clone_from(&self.conversation_id);
        response.previous_response_id = prev_resp_id;
    }

    fn into_persist_task(
        self,
        original_body: ResponsesRequest,
        response_store: ResponseStore,
        conversation_store: Option<ConversationStore>,
    ) -> PersistTask {
        PersistTask::new(
            original_body,
            self.request_body,
            self.new_input_items,
            self.conversation_id,
            response_store,
            conversation_store,
        )
    }
}

pub struct Engine {
    body: ResponsesRequest,
    response_store: ResponseStore,
    conversation_store: Option<ConversationStore>,
    agent: Arc<Agent>,
    client_auth: Option<String>,
}

impl Engine {
    #[must_use]
    pub fn new(
        body: ResponsesRequest,
        response_store: ResponseStore,
        conversation_store: Option<ConversationStore>,
        agent: Arc<Agent>,
        client_auth: Option<String>,
    ) -> Self {
        Self {
            body,
            response_store,
            conversation_store,
            agent,
            client_auth,
        }
    }

    /// # Errors
    ///
    /// Returns an error if context building, upstream API call, or stream processing fails.
    pub async fn run(self) -> Result<either::Either<ResponsesResponse, BoxStream>> {
        let ctx = self.build_context().await?;
        if self.body.stream {
            Ok(either::Either::Right(Box::pin(self.run_stream(ctx))))
        } else {
            Ok(either::Either::Left(self.run_blocking(ctx).await?))
        }
    }

    async fn run_blocking(self, ctx: RequestContext) -> Result<ResponsesResponse> {
        let upstream = self.agent.run_stream(&ctx.request_body, self.client_auth.as_deref());
        futures::pin_mut!(upstream);

        let mut terminal: Option<ResponsesResponse> = None;
        while let Some(result) = upstream.next().await {
            match result {
                Err(e) => return Err(e),
                Ok(StreamEvent::Response(r))
                    if matches!(r.type_.as_str(), "response.completed" | "response.incomplete") =>
                {
                    terminal = Some(r.response);
                }
                _ => {}
            }
        }

        let mut response = terminal.ok_or_else(|| AgenticApiError::bad_input("no response generated"))?;
        ctx.inject_ids(&mut response, self.body.previous_response_id.clone());

        if self.body.store {
            let task = ctx.into_persist_task(self.body, self.response_store, self.conversation_store);
            task.spawn(response.clone());
        }
        Ok(response)
    }

    fn run_stream(self, ctx: RequestContext) -> impl Stream<Item = String> {
        let agent = self.agent.clone();
        let prev_resp_id = self.body.previous_response_id.clone();
        let client_auth = self.client_auth.clone();

        let request_body = ctx.request_body.clone();
        let our_response_id = ctx.our_response_id.clone();
        let conversation_id = ctx.conversation_id.clone();

        let persist_task = self
            .body
            .store
            .then(|| ctx.into_persist_task(self.body.clone(), self.response_store, self.conversation_store));

        stream! {
            let upstream = agent.run_stream(&request_body, client_auth.as_deref());
            futures::pin_mut!(upstream);

            let mut terminal: Option<ResponsesResponse> = None;
            let mut done_emitted = false;

            while let Some(result) = upstream.next().await {
                match result {
                    Err(e) => {
                        yield format!("data: {{\"error\": \"{e}\"}}\n\n");
                        yield DONE_MARKER.to_string();
                        return;
                    }
                    Ok(mut event) => {
                        if let StreamEvent::Response(ref mut r) = event {
                            if matches!(r.type_.as_str(), "response.completed" | "response.incomplete") {
                                r.response.id.clone_from(&our_response_id);
                                r.response.conversation_id.clone_from(&conversation_id);
                                r.response.previous_response_id.clone_from(&prev_resp_id);
                                terminal = Some(r.response.clone());
                            }
                        }
                        yield event.as_responses_chunk();
                        if !done_emitted && TERMINAL_EVENT_TYPES.contains(&event.type_str()) {
                            yield DONE_MARKER.to_string();
                            done_emitted = true;
                            break;
                        }
                    }
                }
            }

            if !done_emitted {
                yield DONE_MARKER.to_string();
            }

            if let (Some(task), Some(response)) = (persist_task, terminal) {
                task.spawn(response);
            }
        }
    }

    fn build_request_context(
        &self,
        new_input_items: Vec<InputItem>,
        our_response_id: String,
        conversation_id: Option<String>,
    ) -> RequestContext {
        let mut request_body = self.body.clone();
        request_body.input = ResponsesInput::Items(new_input_items.clone());
        RequestContext {
            request_body,
            new_input_items,
            our_response_id,
            conversation_id,
        }
    }

    async fn build_context(&self) -> Result<RequestContext> {
        let our_response_id = crate::utils::common::uuid7_str("resp_");
        let new_input_items = normalize_input(&self.body.input);

        if !self.body.store {
            if let Some(ref prev_id) = self.body.previous_response_id {
                if self.response_store.get(prev_id).await?.is_none() {
                    return Err(AgenticApiError::responses_api(
                        format!("No response found with id '{prev_id}'."),
                        400,
                        "invalid_request_error",
                        Some("previous_response_id".into()),
                        Some("previous_response_not_found".into()),
                    ));
                }
            }
            return Ok(self.build_request_context(new_input_items, our_response_id, None));
        }

        // previous_response_id — single DB read covers history + conversation lookup
        if let Some(ref prev_id) = self.body.previous_response_id {
            let stored = self.response_store.get_or_raise(prev_id).await?;
            let conversation = match (&self.conversation_store, &stored.conversation_id) {
                (Some(s), Some(id)) => s.get(id).await?,
                _ => None,
            };

            let mut items = to_input_items(self.response_store.rehydrate(&stored).await?);
            items.extend(new_input_items.clone());

            let mut request_body = self.body.clone();
            request_body.previous_response_id = None;
            request_body.input = ResponsesInput::Items(items);
            request_body.tools = resolve_tools(
                request_body.tools.as_ref(),
                stored.metadata.effective_tools.as_ref(),
                request_body.tools.is_some(),
            );
            request_body.tool_choice =
                resolve_tool_choice(&request_body.tool_choice, &stored.metadata.effective_tool_choice, false);

            let conversation_id = conversation.as_ref().map(|c| c.conversation_id.clone());
            return Ok(RequestContext {
                request_body,
                new_input_items,
                our_response_id,
                conversation_id,
            });
        }

        let Some(conv_store) = self.conversation_store.as_ref() else {
            return Ok(self.build_request_context(new_input_items, our_response_id, None));
        };

        if let Some(ref conv_id) = self.body.conversation_id {
            let (conversation, history) =
                tokio::try_join!(conv_store.get_or_create(conv_id), conv_store.rehydrate(conv_id))?;

            let mut request_body = self.body.clone();
            if history.is_empty() {
                request_body.input = ResponsesInput::Items(new_input_items.clone());
            } else {
                let mut items = to_input_items(history);
                items.extend(new_input_items.clone());
                request_body.input = ResponsesInput::Items(items);
                if let Some(ref meta) = conversation.metadata {
                    request_body.tools = resolve_tools(
                        request_body.tools.as_ref(),
                        meta.effective_tools.as_ref(),
                        request_body.tools.is_some(),
                    );
                    request_body.tool_choice =
                        resolve_tool_choice(&request_body.tool_choice, &meta.effective_tool_choice, false);
                }
            }

            return Ok(RequestContext {
                request_body,
                new_input_items,
                our_response_id,
                conversation_id: Some(conversation.conversation_id),
            });
        }

        let conversation_id_str = crate::utils::common::uuid7_str("conv_");
        conv_store.get_or_create(&conversation_id_str).await?;
        Ok(self.build_request_context(new_input_items, our_response_id, Some(conversation_id_str)))
    }
}
