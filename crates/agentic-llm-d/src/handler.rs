//! The endpoints and their axum glue. The split routes require a shared token;
//! probes do not. Keep the listener cluster-internal regardless — the token
//! authenticates the calling workload, not a tenant.

use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::value::RawValue;
use tracing::warn;

use agentic_core::executor::request::RequestContext;

use agentic_core::executor::{
    ExecutorError, UpstreamBody, commit, decode_upstream, rehydrate_conversation, upstream_request,
};
use agentic_core::types::request_response::RequestPayload;

use crate::BackendState;
use crate::context::{Hydration, ensure_splittable, seal, unseal};

const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;
/// The calling workload's shared secret.
pub const WORKLOAD_TOKEN_HEADER: &str = "x-agentic-workload-token";
/// Readiness means storage answers - llm-d owns the model fleet.
const STORAGE_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Body of `POST /v1alpha/responses/persist`: the context, plus one response form.
#[derive(Debug, Deserialize)]
pub struct PersistRequest {
    context: String,
    response: Option<Box<RawValue>>,
    sse: Option<String>,
}

/// Rejects any split-route call without the shared secret. The probes are
/// layered separately and stay open.
pub async fn require_token(State(state): State<BackendState>, request: Request, next: Next) -> Response {
    // Not `Authorization`: that stays free for the end user's token.
    let presented = request
        .headers()
        .get(WORKLOAD_TOKEN_HEADER)
        .and_then(|value| value.to_str().ok());
    match presented {
        Some(token) if token_matches(token, &state.api_token) => next.run(request).await,
        _ => json(
            StatusCode::UNAUTHORIZED,
            br#"{"error":{"type":"invalid_request_error","message":"missing or invalid bearer token"}}"#.to_vec(),
        ),
    }
}

/// No early return, so a wrong token takes the same time whatever byte differs.
fn token_matches(presented: &str, expected: &str) -> bool {
    presented.len() == expected.len()
        && presented
            .bytes()
            .zip(expected.bytes())
            .fold(0_u8, |differences, (a, b)| differences | (a ^ b))
            == 0
}

pub async fn health() -> StatusCode {
    StatusCode::OK
}

pub async fn ready(State(state): State<BackendState>) -> StatusCode {
    if state.exec_ctx.storage_ready(STORAGE_PROBE_TIMEOUT).await {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

pub async fn hydrate(State(state): State<BackendState>, req: Request) -> Response {
    let payload: RequestPayload = match read_json(req.into_body()).await {
        Ok(payload) => payload,
        Err(response) => return response,
    };
    match build_hydration(payload, &state).await {
        Ok(hydration) => axum::Json(hydration).into_response(),
        Err(error) => error_response(error),
    }
}

/// Rehydrates the turn and builds the request the caller forwards to a model.
#[allow(clippy::result_large_err)] // `ExecutorError` is core's; boxing it is not ours to decide
async fn build_hydration(
    request: RequestPayload,
    state: &BackendState,
) -> agentic_core::executor::ExecutorResult<Hydration> {
    ensure_splittable(&request)?;
    let ctx = rehydrate_conversation(request, state.exec_ctx.as_ref()).await?;
    // Rehydration can restore a gateway-owned tool from the stored turn, so
    // check what will actually run.
    ensure_splittable(&ctx.enriched_request)?;
    let stream = ctx.original_request.stream;
    let request = RawValue::from_string(upstream_request(&ctx, stream)?).map_err(ExecutorError::JsonError)?;
    let context = seal(ctx.into(), &state.signing_key)?;
    Ok(Hydration { request, context })
}

pub async fn persist(State(state): State<BackendState>, req: Request) -> Response {
    let PersistRequest { context, response, sse } = match read_json(req.into_body()).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    // serde rejects `RawValue` in `untagged`, so "exactly one of" is checked here.
    let upstream = match (response.as_deref(), sse.as_deref()) {
        (Some(json), None) => UpstreamBody::Json(json.get()),
        (None, Some(sse)) => UpstreamBody::Sse(sse),
        _ => {
            let message = "exactly one of `response` or `sse` is required".to_owned();
            return error_response(ExecutorError::InvalidRequest(message));
        }
    };
    let context = match unseal(&context, &state.signing_key) {
        Ok(context) => context,
        Err(error) => return error_response(error),
    };
    let ctx = RequestContext::from(context);
    let stored = match decode_upstream(&ctx, upstream) {
        Ok(payload) => commit(ctx, payload, state.exec_ctx.as_ref()).await,
        Err(error) => Err(error),
    };
    match stored {
        Ok(payload) => axum::Json(payload).into_response(),
        Err(error) => error_response(error),
    }
}

/// Renders an error with the status and envelope core defines.
fn error_response(error: ExecutorError) -> Response {
    let status = error.http_status();
    warn!("backend error ({status}): {error}");
    json(status, error.into_response_body())
}

#[allow(clippy::result_large_err)] // an axum `Response` is the idiomatic error here
async fn read_json<T: DeserializeOwned>(body: Body) -> Result<T, Response> {
    let too_large = br#"{"error":{"type":"invalid_request_error","message":"request body too large"}}"#;
    let bytes = axum::body::to_bytes(body, MAX_BODY_SIZE)
        .await
        .map_err(|_| json(StatusCode::PAYLOAD_TOO_LARGE, too_large.to_vec()))?;
    serde_json::from_slice(&bytes).map_err(|error| error_response(ExecutorError::from(error)))
}

fn json(status: StatusCode, body: Vec<u8>) -> Response {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(body))
        .expect("valid response")
}
