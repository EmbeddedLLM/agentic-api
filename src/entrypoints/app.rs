use std::sync::Arc;

use axum::Router;
use axum::extract::FromRef;
use axum::routing::post;
use tower_http::cors::CorsLayer;

use crate::config::RuntimeConfig;
use crate::core::agent::Agent;
use crate::database::db::{configure_pool, create_pool};
use crate::database::schema::SchemaManager;
use crate::routers::conversations::create_conversation_response;
use crate::routers::responses::create_response;
use crate::store::conversation::ConversationStore;
use crate::store::response::ResponseStore;

#[derive(Clone, FromRef)]
pub struct AppState {
    pub config: Arc<RuntimeConfig>,
    pub agent: Arc<Agent>,
    pub response_store: ResponseStore,
    pub conversation_store: Option<ConversationStore>,
}

pub async fn build_app(config: RuntimeConfig) -> Result<Router, Box<dyn std::error::Error>> {
    let (response_store, conversation_store) = if config.response_store_enabled {
        let pool = create_pool(&config.db_url).await?;
        SchemaManager::new(&pool)
            .ensure_ready()
            .await
            .map_err(|e| format!("schema init failed: {e}"))?;
        configure_pool(pool);
        (ResponseStore::new(None), Some(ConversationStore::new(None)))
    } else {
        (ResponseStore::new(None), None)
    };

    let state = AppState {
        config: Arc::new(config.clone()),
        agent: Arc::new(Agent::new(&config)),
        response_store,
        conversation_store,
    };

    Ok(Router::new()
        .route("/v1/responses", post(create_response))
        .route("/v1/conversations", post(create_conversation_response))
        .layer(CorsLayer::very_permissive())
        .with_state(state))
}
