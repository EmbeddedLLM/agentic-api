use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::{Json, body::Body};
use futures::StreamExt;
use http::StatusCode;

use crate::core::agent::Agent;
use crate::core::engine::Engine;
use crate::store::conversation::ConversationStore;
use crate::store::response::ResponseStore;
use crate::types::responses::ResponsesRequest;
use crate::utils::errors::error_response;

pub async fn create_response(
    State(agent): State<Arc<Agent>>,
    State(response_store): State<ResponseStore>,
    State(conversation_store): State<Option<ConversationStore>>,
    Json(body): Json<ResponsesRequest>,
) -> Response {
    let engine = Engine::new(body, response_store, conversation_store, agent);

    match engine.run().await {
        Err(e) => error_response(e),
        Ok(either::Either::Left(response)) => (StatusCode::OK, Json(response)).into_response(),
        Ok(either::Either::Right(sse_stream)) => {
            let body = Body::from_stream(sse_stream.map(Ok::<_, std::convert::Infallible>));
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .header("x-accel-buffering", "no")
                .body(body)
                .unwrap()
        }
    }
}
