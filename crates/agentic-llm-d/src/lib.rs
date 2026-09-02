//! Backend mode — agentic-api as state services for an orchestrator that runs
//! inference itself (the llm-d coordinator). Nothing here proxies or calls a model.

pub mod context;
pub mod handler;
pub mod runner;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::{Router, middleware};

use agentic_core::executor::ExecutionContext;

/// All the endpoints need — far less than the gateway's `AppState`.
#[derive(Clone)]
pub struct BackendState {
    pub exec_ctx: Arc<ExecutionContext>,
    /// Seals the context between hydrate and persist.
    pub signing_key: Arc<Vec<u8>>,
    /// Shared secret every split-route caller must present.
    pub api_token: Arc<String>,
}

/// The whole surface: two split-execution endpoints and two probes.
pub fn build_router(state: BackendState) -> Router {
    // Probes stay open so an orchestrator can check liveness without the secret.
    let probes = Router::new()
        .route("/health", get(handler::health))
        .route("/ready", get(handler::ready));
    let responses = Router::new()
        .route("/v1alpha/responses/hydrate", post(handler::hydrate))
        .route("/v1alpha/responses/persist", post(handler::persist))
        .route_layer(middleware::from_fn_with_state(state.clone(), handler::require_token));
    probes.merge(responses).with_state(state)
}
