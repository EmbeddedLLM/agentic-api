//! Stateful conversation executor.
//!
//! Exposes each step of the conversation pipeline as a public function so consumers
//! can compose them directly (e.g. as Praxis filters). [`ExecuteRequest`] is the
//! primary entry point; [`execute`] is a convenience shim for callers that don't
//! need per-request configuration.

use std::sync::Arc;

use async_stream::stream;
use either::Either;
use tracing::warn;

use crate::executor::accumulator::ResponseAccumulator;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::inference::{DONE_MARKER, call_inference, fetch_response_json};
use crate::executor::persist::persist_response;
use crate::executor::rehydrate::rehydrate_conversation;
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::types::request_response::{RequestPayload, ResponsePayload};
use crate::utils::common::serialize_to_string;

pub use crate::executor::inference::BoxStream;

async fn run_blocking(
    ctx: RequestContext,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
) -> ExecutorResult<ResponsePayload> {
    let url = exec_ctx.responses_url();
    // Non-streaming request: stream=false → full JSON body → from_json.
    let upstream_json =
        serialize_to_string(&ctx.enriched_request.to_upstream_request(false)).map_err(ExecutorError::JsonError)?;

    let body = fetch_response_json(upstream_json, &url, &exec_ctx.client, auth).await?;

    let acc = ResponseAccumulator::from_json(&body, ctx.conversation_id.as_deref())?;
    let mut payload = acc.finalize(
        &ctx.enriched_request.model,
        ctx.original_request.previous_response_id.as_deref(),
        ctx.original_request.instructions.as_deref(),
    );
    ctx.inject_ids(&mut payload);

    let should_persist = ctx.original_request.store
        || ctx.original_request.previous_response_id.is_some()
        || ctx.original_request.conversation_id.is_some();
    if should_persist {
        let ch = exec_ctx.conv_handler.clone();
        let rh = exec_ctx.resp_handler.clone();
        if let Err(e) = persist_response(payload.clone(), ctx, ch, rh).await {
            warn!("persist failed: {e}");
        }
    }

    Ok(payload)
}

fn run_stream(ctx: RequestContext, exec_ctx: Arc<ExecutionContext>, auth: Option<String>) -> BoxStream {
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

    // Persist when store=true, or when an ID is passed — context continuity must
    // be preserved even if the caller sets store=false.
    let should_persist = ctx.original_request.store
        || ctx.original_request.previous_response_id.is_some()
        || ctx.original_request.conversation_id.is_some();

    Box::pin(stream! {
        let line_stream = Box::pin(call_inference(
            upstream_json,
            url,
            Arc::clone(&exec_ctx.client),
            auth,
            exec_ctx.streaming_timeout,
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

                if should_persist {
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

/// Builder for a stateful conversation turn.
///
/// ```ignore
/// ExecuteRequest::new(payload, exec_ctx).with_auth(token).run().await
/// ```
pub struct ExecuteRequest {
    payload: RequestPayload,
    exec_ctx: Arc<ExecutionContext>,
    client_auth: Option<String>,
}

impl ExecuteRequest {
    #[must_use]
    pub fn new(payload: RequestPayload, exec_ctx: Arc<ExecutionContext>) -> Self {
        Self {
            payload,
            exec_ctx,
            client_auth: None,
        }
    }

    /// Override the bearer token for this request only; does not touch the shared [`ExecutionContext`].
    #[must_use]
    pub fn with_auth(mut self, token: Option<String>) -> Self {
        self.client_auth = token;
        self
    }

    /// Execute one stateful conversation turn.
    ///
    /// Returns `Either::Left(ResponsePayload)` for non-streaming requests, or
    /// `Either::Right(BoxStream)` for streaming, each yielded `String` is an SSE
    /// line ready to forward to the client.
    ///
    /// # Errors
    /// Returns [`ExecutorError`] if rehydration or (non-streaming) LLM inference fails.
    pub async fn run(self) -> ExecutorResult<Either<ResponsePayload, BoxStream>> {
        let ctx = rehydrate_conversation(self.payload, &self.exec_ctx).await?;
        if ctx.original_request.stream {
            Ok(Either::Right(run_stream(ctx, self.exec_ctx, self.client_auth)))
        } else {
            Ok(Either::Left(
                run_blocking(ctx, &self.exec_ctx, self.client_auth.as_deref()).await?,
            ))
        }
    }
}

/// Execute one stateful conversation turn.
///
/// Thin shim over [`ExecuteRequest`] for callers that don't need per-request auth override.
///
/// # Errors
/// Returns [`ExecutorError`] if rehydration or (non-streaming) LLM inference fails.
pub async fn execute(
    request: RequestPayload,
    exec_ctx: Arc<ExecutionContext>,
) -> ExecutorResult<Either<ResponsePayload, BoxStream>> {
    ExecuteRequest::new(request, exec_ctx).run().await
}
