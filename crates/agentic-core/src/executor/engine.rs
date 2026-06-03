//! Agentic loop executor.
//!
//! Exposes each step of the loop as a public function so consumers can compose
//! them directly (e.g. as Praxis filters). [`execute`] is the convenience entry
//! point that composes all steps with the default control flow.

use std::pin::Pin;
use std::sync::Arc;

use async_stream::stream;
use either::Either;
use futures::{Stream, StreamExt};
use tracing::warn;

use crate::executor::accumulator::ResponseAccumulator;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::modes::{ConversationHandler, ResponseHandler};
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::storage::InOutItem;
use crate::types::event::ResponseStatus;
use crate::types::io::{InputItem, ResponsesInput, resolve_tool_choice, resolve_tools};
use crate::types::request_response::{RequestPayload, ResponsePayload};
use crate::utils::common::serialize_to_string;
use crate::utils::uuid7_str;

/// SSE stream of raw lines sent to the client (`data: …\n\n` per event).
pub type BoxStream = Pin<Box<dyn Stream<Item = String> + Send>>;

/// Wire-format marker signalling end-of-stream to the client.
const DONE_MARKER: &str = "data: [DONE]\n\n";

/// Makes a non-streaming HTTP POST to the LLM backend and returns the full JSON body.
///
/// Used by [`run_blocking`] so it can pass the result to [`ResponseAccumulator::from_json`].
async fn fetch_response_json(
    upstream_json: String,
    url: &str,
    client: &reqwest::Client,
    auth: Option<&str>,
) -> ExecutorResult<String> {
    let mut req = client
        .post(url)
        .header("Content-Type", "application/json")
        .body(upstream_json);
    if let Some(key) = auth {
        req = req.bearer_auth(key);
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) if e.is_timeout() => {
            return Err(ExecutorError::LLMRequest {
                status: http::StatusCode::GATEWAY_TIMEOUT,
                body: "upstream timeout".into(),
            });
        }
        Err(_) => {
            return Err(ExecutorError::LLMRequest {
                status: http::StatusCode::BAD_GATEWAY,
                body: "upstream unavailable".into(),
            });
        }
    };

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(ExecutorError::LLMRequest {
            status: http::StatusCode::from_u16(status).unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR),
            body,
        });
    }

    resp.text()
        .await
        .map_err(|e| ExecutorError::StreamError(format!("failed to read response body: {e}")))
}

/// Step 1 — Build [`RequestContext`] by rehydrating conversation history.
///
/// `request` is moved into the context as `enriched_request`; one clone is taken
/// for `original_request` so the engine retains an unmodified copy for persistence
/// and ID resolution.
///
/// Dispatches to one of four paths based on `store` flag and which ID is present:
/// - `store=false` + `previous_response_id`: validate the prior response exists, no history loaded
/// - `store=true`  + `previous_response_id`: [`rehydrate_from_response`]
/// - `store=true`  + `conversation_id`:      [`rehydrate_from_conversation`]
/// - `store=true`  + no ids:                 create a new conversation
///
/// # Errors
/// Returns [`ExecutorError`] if storage is unavailable or a referenced ID does not exist.
pub async fn rehydrate_conversation(
    request: RequestPayload,
    exec_ctx: &ExecutionContext,
) -> ExecutorResult<RequestContext> {
    let response_id = uuid7_str("resp_");
    let new_input_items: Vec<InputItem> = Vec::from(&request.input);

    // One clone for the unmodified original; `request` is moved as enriched_request.
    let original_request = request.clone();
    let mut ctx = RequestContext {
        enriched_request: request,
        original_request,
        new_input_items,
        response_id,
        conversation_id: None,
    };

    if !ctx.original_request.store {
        // Non-store path: validate previous_response_id only; no history needed.
        if ctx.original_request.previous_response_id.is_some() {
            exec_ctx.resp_handler.validate_exists(&ctx).await?;
        }
        return Ok(ctx);
    }

    if ctx.original_request.previous_response_id.is_some() {
        rehydrate_from_response(&mut ctx, exec_ctx).await?;
        return Ok(ctx);
    }

    if ctx.original_request.conversation_id.is_some() {
        rehydrate_from_conversation(&mut ctx, exec_ctx).await?;
        return Ok(ctx);
    }

    // Store + no ids: create a fresh conversation.
    let conv_data = exec_ctx.conv_handler.create().await?;
    ctx.conversation_id = Some(conv_data.conversation_id);
    ctx.enriched_request.input = ResponsesInput::Items(ctx.new_input_items.clone());
    Ok(ctx)
}

/// Hydrates `ctx` from the previous response chain.
///
/// Loads the stored response, rehydrates its history items, resolves effective
/// tools and tool choice from the stored metadata, and prepends the history to
/// the enriched request input.
async fn rehydrate_from_response(ctx: &mut RequestContext, exec_ctx: &ExecutionContext) -> ExecutorResult<()> {
    let stored = exec_ctx.resp_handler.get(ctx).await?;
    let history = exec_ctx.resp_handler.rehydrate(ctx).await?;

    let mut items = InOutItem::into_input_items(history);
    items.reserve(ctx.new_input_items.len());
    items.extend(ctx.new_input_items.iter().cloned());

    ctx.enriched_request.previous_response_id = None;
    ctx.enriched_request.input = ResponsesInput::Items(items);
    ctx.enriched_request.tools = resolve_tools(
        ctx.original_request.tools.as_deref(),
        stored.metadata.effective_tools.as_deref(),
        ctx.original_request.tools.is_some(),
    );
    ctx.enriched_request.tool_choice = resolve_tool_choice(
        &ctx.original_request.tool_choice,
        &stored.metadata.effective_tool_choice,
        false,
    );
    ctx.conversation_id = stored.conversation_id;
    Ok(())
}

/// Hydrates `ctx` from the conversation store.
///
/// Gets or creates the conversation and rehydrates its history in parallel,
/// then prepends the history items to the enriched request input.
async fn rehydrate_from_conversation(ctx: &mut RequestContext, exec_ctx: &ExecutionContext) -> ExecutorResult<()> {
    let (conv_data, history) = tokio::try_join!(
        exec_ctx.conv_handler.get_or_create(ctx),
        exec_ctx.conv_handler.rehydrate(ctx),
    )?;

    let mut items = InOutItem::into_input_items(history);
    items.reserve(ctx.new_input_items.len());
    items.extend(ctx.new_input_items.iter().cloned());

    ctx.enriched_request.input = ResponsesInput::Items(items);
    ctx.conversation_id = Some(conv_data.conversation_id);
    Ok(())
}

/// Step 2 — Call the LLM inference backend; yields raw SSE lines (`data: …`).
///
/// Always requests `stream=true` upstream. Stops on `[DONE]`.
/// Yields `Err` on connection failure (502), timeout (504), or non-2xx status.
pub fn call_inference(
    upstream_json: String,
    url: String,
    client: Arc<reqwest::Client>,
    auth: Option<String>,
) -> impl Stream<Item = Result<String, ExecutorError>> + Send + 'static {
    stream! {
        let mut req = client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(upstream_json);
        if let Some(ref key) = auth {
            req = req.bearer_auth(key);
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) if e.is_timeout() => {
                yield Err(ExecutorError::LLMRequest {
                    status: http::StatusCode::GATEWAY_TIMEOUT,
                    body: "upstream timeout".into(),
                });
                return;
            }
            Err(_) => {
                yield Err(ExecutorError::LLMRequest {
                    status: http::StatusCode::BAD_GATEWAY,
                    body: "upstream unavailable".into(),
                });
                return;
            }
        };

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            yield Err(ExecutorError::LLMRequest {
                status: http::StatusCode::from_u16(status)
                    .unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR),
                body,
            });
            return;
        }

        let buf_cap = resp
            .content_length()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(8192)
            .min(4 * 1024 * 1024);

        let mut byte_stream = resp.bytes_stream();
        let mut buf = String::with_capacity(buf_cap);

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(ExecutorError::StreamError(format!("stream read error: {e}")));
                    return;
                }
            };

            match std::str::from_utf8(&chunk) {
                Ok(s) => buf.push_str(s),
                Err(_) => buf.push_str(&String::from_utf8_lossy(&chunk)),
            }

            while let Some(pos) = buf.find('\n') {
                let line_end = if pos > 0 && buf.as_bytes()[pos - 1] == b'\r' {
                    pos - 1
                } else {
                    pos
                };
                let line = &buf[..line_end];
                if line.starts_with("data: ") {
                    if line == "data: [DONE]" {
                        return;
                    }
                    yield Ok(line.to_string());
                }
                buf.drain(..=pos);
            }
        }
    }
}

/// Step 3 — Persist the completed response to storage.
///
/// Skipped if [`ResponseStatus`] is not `Completed`/`Incomplete` or `payload.id` is empty.
/// Routes to [`ConversationHandler`] when `ctx.conversation_id` is set,
/// otherwise [`ResponseHandler`].
///
/// # Errors
/// Returns [`ExecutorError`] if the storage operation fails.
pub async fn persist_response(
    payload: ResponsePayload,
    ctx: RequestContext,
    conv_handler: ConversationHandler,
    resp_handler: ResponseHandler,
) -> ExecutorResult<()> {
    // Use typed enum — no hardcoded status strings.
    if !matches!(
        payload.status.parse::<ResponseStatus>().unwrap_or_default(),
        ResponseStatus::Completed | ResponseStatus::Incomplete
    ) || payload.id.is_empty()
    {
        return Ok(());
    }

    // Move output items from payload; handlers build ResponseMetadata from ctx internally.
    let output_items = payload.output;

    if ctx.conversation_id.is_some() {
        conv_handler.execute_turn(ctx, output_items).await
    } else {
        resp_handler.execute_turn(ctx, output_items).await
    }
}

async fn run_blocking(ctx: RequestContext, exec_ctx: &ExecutionContext) -> ExecutorResult<ResponsePayload> {
    let url = exec_ctx.responses_url();
    // Non-streaming request: stream=false → full JSON body → from_json.
    let upstream_json = serialize_to_string(&ctx.enriched_request.to_upstream_request(false))
        .map_err(|e| ExecutorError::ParseError(e.to_string()))?;

    let body = fetch_response_json(upstream_json, &url, &exec_ctx.client, exec_ctx.client_auth.as_deref()).await?;

    let acc = ResponseAccumulator::from_json(&body, ctx.conversation_id.as_deref())?;
    let mut payload = acc.finalize(
        &ctx.enriched_request.model,
        ctx.original_request.previous_response_id.as_deref(),
        ctx.original_request.instructions.as_deref(),
    );
    ctx.inject_ids(&mut payload);

    if ctx.original_request.store {
        let ch = exec_ctx.conv_handler.clone();
        let rh = exec_ctx.resp_handler.clone();
        if let Err(e) = persist_response(payload.clone(), ctx, ch, rh).await {
            warn!("persist failed: {e}");
        }
    }

    Ok(payload)
}

fn run_stream(ctx: RequestContext, exec_ctx: Arc<ExecutionContext>) -> BoxStream {
    let url = exec_ctx.responses_url();
    // Streaming request: stream=true → SSE lines → from_stream.
    let upstream_json = match serialize_to_string(&ctx.enriched_request.to_upstream_request(true)) {
        Ok(s) => s,
        Err(e) => {
            return Box::pin(stream! {
                yield format!("data: {{\"error\": \"serialize error: {e}\"}}\n\n");
                yield DONE_MARKER.to_string();
            });
        }
    };

    let store = ctx.original_request.store;

    Box::pin(stream! {
        let line_stream = Box::pin(call_inference(
            upstream_json,
            url,
            Arc::clone(&exec_ctx.client),
            exec_ctx.client_auth.clone(),
        ));

        // from_stream feeds SSE lines to a spawn_blocking worker via channel.
        // All JSON parsing is CPU-bound and runs off the async executor.
        match ResponseAccumulator::from_stream(line_stream, ctx.conversation_id.as_deref()).await {
            Err(e) => {
                yield format!("data: {{\"error\": \"{e}\"}}\n\n");
                yield DONE_MARKER.to_string();
            }
            Ok(acc) => {
                let mut payload = acc.finalize(
                    &ctx.enriched_request.model,
                    ctx.original_request.previous_response_id.as_deref(),
                    ctx.original_request.instructions.as_deref(),
                );
                ctx.inject_ids(&mut payload);
                yield payload.as_responses_chunk();
                yield DONE_MARKER.to_string();

                if store {
                    let ch = exec_ctx.conv_handler.clone();
                    let rh = exec_ctx.resp_handler.clone();
                    if let Err(e) = persist_response(payload, ctx, ch, rh).await {
                        warn!("persist failed: {e}");
                    }
                }
            }
        }
    })
}

/// Create a new conversation and return its data.
///
/// Exposes the conversation-creation step as a standalone function so callers
/// (e.g. `agentic-server`, Praxis filters, or tests) can pre-create a
/// conversation before submitting response turns.
///
/// # Errors
/// Returns [`ExecutorError`] if the conversation store is unavailable.
pub async fn create_conversation(exec_ctx: &ExecutionContext) -> ExecutorResult<crate::ConversationData> {
    exec_ctx.conv_handler.create().await
}

/// Run the full agentic loop.
///
/// Returns `Either::Left(ResponsePayload)` for non-streaming requests, or
/// `Either::Right(BoxStream)` for streaming, each yielded `String` is an SSE
/// line ready to forward to the client.
///
/// # Errors
/// Returns [`ExecutorError`] if rehydration or (non-streaming) LLM inference fails.
pub async fn execute(
    request: RequestPayload,
    exec_ctx: Arc<ExecutionContext>,
) -> ExecutorResult<Either<ResponsePayload, BoxStream>> {
    let ctx = rehydrate_conversation(request, &exec_ctx).await?;
    if ctx.original_request.stream {
        Ok(Either::Right(run_stream(ctx, exec_ctx)))
    } else {
        Ok(Either::Left(run_blocking(ctx, &exec_ctx).await?))
    }
}
