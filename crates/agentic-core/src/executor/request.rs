use std::sync::Arc;

use crate::executor::modes::{ConversationHandler, ResponseHandler};
use crate::types::io::InputItem;
use crate::types::request_response::{RequestPayload, ResponsePayload};

/// Context built by `rehydrate_conversation`, threaded through the execute pipeline.
pub struct RequestContext {
    /// Untouched original request from the client.
    pub original_request: RequestPayload,
    /// Enriched request with rehydrated conversation history injected into `.input`.
    /// This is the request forwarded to the LLM.
    pub enriched_request: RequestPayload,
    /// Only the new input items submitted by the client this turn (used for persistence).
    pub new_input_items: Vec<InputItem>,
    /// Our generated response ID (uuid7 with "resp_" prefix).
    pub response_id: String,
    /// Resolved conversation ID. `None` when `store=false` or non-conversational.
    pub conversation_id: Option<String>,
}

impl RequestContext {
    /// Inject our `response_id` and `conversation_id` into a `ResponsePayload`
    /// received from the LLM (which carries the upstream's own IDs).
    pub(crate) fn inject_ids(&self, payload: &mut ResponsePayload) {
        payload.id.clone_from(&self.response_id);
        payload.conversation_id.clone_from(&self.conversation_id);
        payload
            .previous_response_id
            .clone_from(&self.original_request.previous_response_id);
    }
}

/// Runtime dependencies passed into `execute()`.
///
/// Owns the storage handlers, HTTP client, and LLM endpoint configuration.
pub struct ExecutionContext {
    pub conv_handler: ConversationHandler,
    pub resp_handler: ResponseHandler,
    pub client: Arc<reqwest::Client>,
    /// Base URL for the LLM backend, e.g. `"http://localhost:8000"`.
    pub llm_base_url: String,
    /// Bearer token forwarded from the client, if any.
    pub client_auth: Option<String>,
}

impl ExecutionContext {
    /// Returns the full URL for the `/v1/responses` endpoint.
    #[must_use]
    pub fn responses_url(&self) -> String {
        format!("{}/v1/responses", self.llm_base_url)
    }

    /// Returns the full URL for the `/v1/conversations` endpoint.
    #[must_use]
    pub fn conversations_url(&self) -> String {
        format!("{}/v1/conversations", self.llm_base_url)
    }

    #[must_use]
    pub fn new(
        conv_handler: ConversationHandler,
        resp_handler: ResponseHandler,
        client: Arc<reqwest::Client>,
        llm_base_url: String,
        client_auth: Option<String>,
    ) -> Self {
        Self {
            conv_handler,
            resp_handler,
            client,
            llm_base_url,
            client_auth,
        }
    }
}
