use tracing::warn;

use crate::store::conversation::ConversationStore;
use crate::store::response::{ResponseMetadata, ResponseStore};
use crate::store::translator::InOutItem;
use crate::types::responses::{InputItem, ResponsesRequest, ResponsesResponse};
use crate::utils::errors::AgenticApiError;

type Result<T> = std::result::Result<T, AgenticApiError>;

enum PersistPath {
    Conversation {
        conversation_id: String,
        conversation_store: ConversationStore,
    },
    Response {
        response_store: ResponseStore,
    },
}

/// Owns everything needed to persist one completed response.
/// Designed to be sent into `tokio::spawn` after the SSE stream closes.
///
/// # Persistence Modes
///
/// Encodes two mutually exclusive persistence paths:
/// - **Conversation mode:** Stores turn-by-turn history in the conversation store
/// - **Response mode:** Stores individual response in the response store
///
/// The mode is determined at construction time, eliminating runtime `Option` checks.
pub struct PersistTask {
    original_body: ResponsesRequest,
    request_body: ResponsesRequest,
    new_input_items: Vec<InputItem>,
    path: PersistPath,
}

impl PersistTask {
    /// Creates a new persist task. Determines persistence mode at construction time.
    #[must_use]
    pub fn new(
        original_body: ResponsesRequest,
        request_body: ResponsesRequest,
        new_input_items: Vec<InputItem>,
        conversation_id: Option<String>,
        response_store: ResponseStore,
        conversation_store: Option<ConversationStore>,
    ) -> Self {
        let path = match (conversation_id, conversation_store) {
            (Some(conv_id), Some(conv_store)) => PersistPath::Conversation {
                conversation_id: conv_id,
                conversation_store: conv_store,
            },
            _ => PersistPath::Response { response_store },
        };
        Self {
            original_body,
            request_body,
            new_input_items,
            path,
        }
    }

    /// Fire-and-forget: spawn the task, logging any error.
    pub fn spawn(self, response: ResponsesResponse) {
        tokio::spawn(async move {
            if let Err(e) = self.run(response).await {
                warn!("persist failed: {e}");
            }
        });
    }

    async fn run(self, response: ResponsesResponse) -> Result<()> {
        if !matches!(response.status.as_str(), "completed" | "incomplete") || response.id.is_empty() {
            return Ok(());
        }

        let metadata = ResponseMetadata {
            model: response.model.clone(),
            previous_response_id: response.previous_response_id.clone(),
            effective_tools: self.request_body.tools.clone(),
            effective_tool_choice: self.request_body.tool_choice.clone(),
            effective_instructions: self.request_body.instructions.clone(),
        };

        match self.path {
            PersistPath::Conversation {
                conversation_id,
                conversation_store,
            } => {
                let mut items: Vec<InOutItem> = self.new_input_items.into_iter().map(InOutItem::Input).collect();
                items.extend(response.output.iter().map(|o| InOutItem::Output(o.clone())));
                conversation_store
                    .put_turn(
                        &conversation_id,
                        &response.id,
                        response.previous_response_id.as_deref(),
                        &items,
                        &metadata,
                    )
                    .await?;
            }
            PersistPath::Response { response_store } => {
                response_store
                    .put_completed(&self.original_body, &self.request_body, &response)
                    .await?;
            }
        }
        Ok(())
    }
}
