use tracing::warn;

use crate::store::conversation::ConversationStore;
use crate::store::response::{ResponseMetadata, ResponseStore};
use crate::store::translator::InOutItem;
use crate::types::responses::{InputItem, ResponsesRequest, ResponsesResponse};
use crate::utils::errors::AgenticApiError;

type Result<T> = std::result::Result<T, AgenticApiError>;

/// Owns everything needed to persist one completed response.
/// Designed to be sent into `tokio::spawn` after the SSE stream closes.
pub struct PersistTask {
    pub original_body: ResponsesRequest,
    pub request_body: ResponsesRequest,
    pub new_input_items: Vec<InputItem>,
    pub conversation_id: Option<String>,
    pub response_store: ResponseStore,
    pub conversation_store: Option<ConversationStore>,
}

impl PersistTask {
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

        if let (Some(conv_id), Some(conv_store)) = (self.conversation_id.as_ref(), self.conversation_store.as_ref()) {
            let mut items: Vec<InOutItem> = self.new_input_items.into_iter().map(InOutItem::Input).collect();
            items.extend(response.output.iter().map(|o| InOutItem::Output(o.clone())));
            conv_store
                .put_turn(
                    conv_id,
                    &response.id,
                    response.previous_response_id.as_deref(),
                    &items,
                    &metadata,
                )
                .await?;
        } else {
            self.response_store
                .put_completed(&self.original_body, &self.request_body, &response)
                .await?;
        }
        Ok(())
    }
}
