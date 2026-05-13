use std::sync::Arc;

use axum::Router;
use axum::extract::{FromRef, State};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, body::Body};
use futures::StreamExt;
use http::{HeaderMap, StatusCode};
use tower_http::cors::CorsLayer;

use crate::config::RuntimeConfig;
use crate::core::agent::Agent;
use crate::core::engine::Engine;
use crate::database::db::create_pool;
use crate::database::schema::SchemaManager;
use crate::store::conversation::ConversationStore;
use crate::store::response::ResponseStore;
use crate::types::responses::ResponsesRequest;
use crate::utils::errors::error_response;

#[derive(Clone, FromRef)]
pub struct AppState {
    pub config: Arc<RuntimeConfig>,
    pub agent: Arc<Agent>,
    pub response_store: ResponseStore,
    pub conversation_store: Option<ConversationStore>,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/responses", post(dispatch_response))
        .route("/v1/conversations", post(dispatch_response))
        .layer(CorsLayer::very_permissive())
        .with_state(state)
}

/// # Errors
///
/// Returns an error if pool creation or schema initialization fails.
pub async fn build_app(config: RuntimeConfig) -> Result<Router, Box<dyn std::error::Error>> {
    let (response_store, conversation_store) = if config.response_store_enabled {
        let pool = create_pool(&config.db_url).await?;
        SchemaManager::new(&pool)
            .ensure_ready()
            .await
            .map_err(|e| format!("schema init failed: {e}"))?;
        (
            ResponseStore::new(Arc::clone(&pool)),
            Some(ConversationStore::new(Arc::clone(&pool))),
        )
    } else {
        (ResponseStore::disabled(), None)
    };

    let state = AppState {
        config: Arc::new(config.clone()),
        agent: Arc::new(Agent::new(&config)),
        response_store,
        conversation_store,
    };

    Ok(build_router(state))
}

fn extract_bearer_auth(headers: &HeaderMap) -> Option<String> {
    headers
        .get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.strip_prefix("Bearer ").unwrap_or(s).to_owned())
}

async fn run_engine(engine: Engine) -> Response {
    match engine.run().await {
        Err(e) => error_response(&e),
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

pub async fn dispatch_response(
    State(agent): State<Arc<Agent>>,
    State(response_store): State<ResponseStore>,
    State(conversation_store): State<Option<ConversationStore>>,
    headers: HeaderMap,
    Json(body): Json<ResponsesRequest>,
) -> Response {
    let client_auth = extract_bearer_auth(&headers);
    let engine = Engine::new(body, response_store, conversation_store, agent, client_auth);
    run_engine(engine).await
}
