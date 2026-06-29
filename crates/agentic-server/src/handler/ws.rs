//! WebSocket handler for the `/v1/responses/ws` endpoint.

use std::sync::Arc;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::HeaderMap;
use axum::response::Response;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use http::StatusCode;
use serde_json::{Value, json};
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use agentic_core::executor::accumulator::ResponseAccumulator;
use agentic_core::executor::{
    ExecutionContext, ExecutorError, RequestContext, call_inference, persist_response, rehydrate_conversation,
};
use agentic_core::types::ResponsePayload;
use agentic_core::types::request_response::RequestPayload;
use agentic_core::utils::common::serialize_to_string;

use crate::app::AppState;

use super::common::{MAX_BODY_SIZE, extract_bearer};

type WsSender = SplitSink<WebSocket, Message>;
type WsReceiver = SplitStream<WebSocket>;

#[derive(Debug, Error)]
pub(super) enum WsError {
    #[error(transparent)]
    Executor(#[from] ExecutorError),

    #[error("invalid JSON: {0}")]
    InvalidJson(#[source] serde_json::Error),

    #[error("failed to serialize websocket event: {0}")]
    SerializeJson(#[source] serde_json::Error),

    #[error("websocket message type must be response.create")]
    UnexpectedType,

    #[error("websocket messages must be JSON text frames")]
    BinaryFrame,

    #[error("websocket received a new message while response stream is active")]
    ConcurrentMessage,

    #[error("websocket send failed")]
    SendFailed,

    #[error("websocket client disconnected")]
    ClientDisconnected,

    #[error("websocket shutdown requested")]
    Shutdown,

    #[error("websocket receive failed: {0}")]
    Receive(String),
}

impl WsError {
    fn status(&self) -> StatusCode {
        match self {
            Self::Executor(err) => err.http_status(),
            Self::InvalidJson(_) | Self::UnexpectedType | Self::BinaryFrame | Self::ConcurrentMessage => {
                StatusCode::BAD_REQUEST
            }
            Self::SerializeJson(_)
            | Self::SendFailed
            | Self::ClientDisconnected
            | Self::Shutdown
            | Self::Receive(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Executor(err) => err.error_code(),
            Self::InvalidJson(_) => "invalid_json",
            Self::UnexpectedType | Self::BinaryFrame | Self::ConcurrentMessage => "invalid_request_error",
            Self::SerializeJson(_)
            | Self::SendFailed
            | Self::ClientDisconnected
            | Self::Shutdown
            | Self::Receive(_) => "server_error",
        }
    }

    fn to_ws_frame(&self) -> Option<Value> {
        if matches!(
            self,
            Self::SerializeJson(_) | Self::SendFailed | Self::ClientDisconnected | Self::Shutdown | Self::Receive(_)
        ) {
            return None;
        }

        let code = self.code();
        Some(json!({
            "type": "error",
            "status": self.status().as_u16(),
            "error": {
                "message": self.to_string(),
                "type": code,
                "code": code
            }
        }))
    }
}

pub async fn responses_ws(State(state): State<AppState>, headers: HeaderMap, ws: WebSocketUpgrade) -> Response {
    ws.max_message_size(MAX_BODY_SIZE)
        .max_frame_size(MAX_BODY_SIZE)
        .on_upgrade(move |socket| responses_ws_loop(socket, state, headers))
}

async fn responses_ws_loop(socket: WebSocket, state: AppState, headers: HeaderMap) {
    let shutdown_token = state.shutdown_token.clone();
    let (mut sender, mut receiver) = socket.split();

    loop {
        let message = tokio::select! {
            () = shutdown_token.cancelled() => break,
            message = receiver.next() => message,
        };

        let Some(message) = message else {
            break;
        };

        match message {
            Ok(Message::Text(text)) => {
                match handle_ws_text(
                    &mut sender,
                    &mut receiver,
                    &state,
                    &headers,
                    text.as_str(),
                    &shutdown_token,
                )
                .await
                {
                    Ok(()) => {}
                    Err(err) => {
                        if !handle_ws_error(&mut sender, err).await {
                            break;
                        }
                    }
                }
            }
            Ok(Message::Binary(_)) => {
                if !handle_ws_error(&mut sender, WsError::BinaryFrame).await {
                    break;
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(Message::Ping(payload)) => {
                if sender.send(Message::Pong(payload)).await.is_err() {
                    break;
                }
            }
            Ok(Message::Pong(_)) => {}
            Err(e) => {
                warn!("responses websocket receive error: {e}");
                break;
            }
        }
    }
}

async fn handle_ws_text(
    sender: &mut WsSender,
    receiver: &mut WsReceiver,
    state: &AppState,
    headers: &HeaderMap,
    text: &str,
    shutdown_token: &CancellationToken,
) -> Result<(), WsError> {
    let value = serde_json::from_str::<Value>(text).map_err(WsError::InvalidJson)?;

    if value.get("type").and_then(Value::as_str) != Some("response.create") {
        return Err(WsError::UnexpectedType);
    }

    let mut payload = serde_json::from_value::<RequestPayload>(value).map_err(ExecutorError::from)?;
    payload.stream = true;

    let auth = extract_bearer(headers);
    let exec_ctx = Arc::clone(&state.exec_ctx);
    let ctx = rehydrate_conversation(payload, &exec_ctx).await?;
    let upstream_json =
        serialize_to_string(&ctx.enriched_request.to_upstream_request(true)).map_err(ExecutorError::from)?;

    stream_ws_response(sender, receiver, exec_ctx, ctx, upstream_json, auth, shutdown_token).await
}

async fn stream_ws_response(
    sender: &mut WsSender,
    receiver: &mut WsReceiver,
    exec_ctx: Arc<ExecutionContext>,
    ctx: RequestContext,
    upstream_json: String,
    auth: Option<String>,
    shutdown_token: &CancellationToken,
) -> Result<(), WsError> {
    let should_persist = ctx.original_request.store
        || ctx.original_request.previous_response_id.is_some()
        || ctx.conversation_id.is_some();
    let mut lines = Vec::new();
    let mut stream = Box::pin(call_inference(
        upstream_json,
        exec_ctx.responses_url(),
        Arc::clone(&exec_ctx.client),
        auth,
        exec_ctx.streaming_timeout,
    ));

    'stream: loop {
        let next_line = tokio::select! {
            () = shutdown_token.cancelled() => return Err(WsError::Shutdown),
            message = receiver.next() => {
                match message {
                    None | Some(Ok(Message::Close(_))) => return Err(WsError::ClientDisconnected),
                    Some(Ok(Message::Ping(payload))) => {
                        sender.send(Message::Pong(payload)).await.map_err(|_| WsError::SendFailed)?;
                        continue 'stream;
                    }
                    Some(Ok(Message::Pong(_))) => continue 'stream,
                    Some(Ok(Message::Binary(_))) => return Err(WsError::BinaryFrame),
                    Some(Ok(Message::Text(_))) => return Err(WsError::ConcurrentMessage),
                    Some(Err(e)) => return Err(WsError::Receive(e.to_string())),
                }
            }
            line = stream.next() => line,
        };
        let Some(line) = next_line else {
            break;
        };
        let line = match line {
            Ok(line) => line,
            Err(e) => return Err(WsError::Executor(e)),
        };
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            continue;
        }
        let mut value = match serde_json::from_str::<Value>(data) {
            Ok(value) => value,
            Err(e) => return Err(WsError::Executor(ExecutorError::from(e))),
        };
        apply_gateway_response_ids(&mut value, &ctx);
        send_ws_json(sender, value).await?;
        if should_persist {
            lines.push(line);
        }
    }

    if should_persist && !lines.is_empty() {
        let acc = ResponseAccumulator::from_sse_lines(lines, ctx.conversation_id.as_deref());
        let mut payload = acc.finalize(
            &ctx.enriched_request.model,
            ctx.original_request.previous_response_id.as_deref(),
            ctx.original_request.instructions.as_deref(),
        );
        apply_gateway_payload_ids(&mut payload, &ctx);
        let ch = exec_ctx.conv_handler.clone();
        let rh = exec_ctx.resp_handler.clone();
        if let Err(e) = persist_response(payload, ctx, ch, rh).await {
            warn!("persist failed: {e}");
        }
    }
    Ok(())
}

fn apply_gateway_response_ids(value: &mut Value, ctx: &RequestContext) {
    let Some(response) = value.get_mut("response").and_then(Value::as_object_mut) else {
        return;
    };
    response.insert("id".to_owned(), Value::String(ctx.response_id.clone()));
    if let Some(previous_response_id) = &ctx.original_request.previous_response_id {
        response.insert(
            "previous_response_id".to_owned(),
            Value::String(previous_response_id.clone()),
        );
    }
    if let Some(conversation_id) = &ctx.conversation_id {
        response.insert("conversation_id".to_owned(), Value::String(conversation_id.clone()));
    }
}

fn apply_gateway_payload_ids(payload: &mut ResponsePayload, ctx: &RequestContext) {
    payload.id.clone_from(&ctx.response_id);
    payload.conversation_id.clone_from(&ctx.conversation_id);
    payload
        .previous_response_id
        .clone_from(&ctx.original_request.previous_response_id);
}

async fn handle_ws_error(sender: &mut WsSender, err: WsError) -> bool {
    match err {
        WsError::Shutdown | WsError::ClientDisconnected | WsError::SendFailed => false,
        WsError::Receive(message) => {
            warn!("responses websocket receive error: {message}");
            false
        }
        err => send_ws_error(sender, &err).await.is_ok(),
    }
}

async fn send_ws_error(sender: &mut WsSender, err: &WsError) -> Result<(), WsError> {
    let Some(frame) = err.to_ws_frame() else {
        return Err(WsError::SendFailed);
    };
    send_ws_json(sender, frame).await
}

async fn send_ws_json(sender: &mut WsSender, value: Value) -> Result<(), WsError> {
    let text = serde_json::to_string(&value).map_err(WsError::SerializeJson)?;
    sender
        .send(Message::Text(text.into()))
        .await
        .map_err(|_| WsError::SendFailed)
}
