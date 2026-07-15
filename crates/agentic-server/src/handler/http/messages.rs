use axum::extract::{Request, State};
use axum::response::Response;
use bytes::Bytes;

use agentic_core::proxy::{ProxyAuth, ProxyRequest, proxy_request_with_path};

use super::super::common::{convert_response, read_bytes_with_auth};
use crate::app::AppState;

async fn proxy_messages(state: &AppState, request: Request, path: &'static str) -> Response {
    let (parts, body) = request.into_parts();
    let body: Bytes = match read_bytes_with_auth(body, ProxyAuth::Anthropic).await {
        Ok(body) => body,
        Err(response) => return response,
    };

    convert_response(
        proxy_request_with_path(
            ProxyRequest {
                headers: parts.headers,
                body,
                query: parts.uri.query().map(str::to_owned),
            },
            path,
            ProxyAuth::Anthropic,
            &state.proxy_state,
        )
        .await,
    )
}

pub async fn messages(State(state): State<AppState>, request: Request) -> Response {
    proxy_messages(&state, request, "/v1/messages").await
}

pub async fn count_tokens(State(state): State<AppState>, request: Request) -> Response {
    proxy_messages(&state, request, "/v1/messages/count_tokens").await
}
