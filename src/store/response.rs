use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::database::db::{DbPool, get_pool};
use crate::database::{item, response};
use crate::store::translator::{InOutItem, ItemPayload, normalize_input};
use crate::types::responses::{ResponsesRequest, ResponsesResponse, ResponsesTool, ToolChoice};
use crate::utils::common::uuid7_str;
use crate::utils::errors::AgenticApiError;

type Result<T> = std::result::Result<T, AgenticApiError>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseMetadata {
    pub model: String,
    pub previous_response_id: Option<String>,
    pub effective_tools: Option<Vec<ResponsesTool>>,
    pub effective_tool_choice: ToolChoice,
    pub effective_instructions: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub response_id: String,
    pub conversation_id: Option<String>,
    pub previous_response_id: Option<String>,
    pub created_at: String,
    pub history_item_ids: Vec<String>,
    pub metadata: ResponseMetadata,
}

pub struct ResponseStore {
    pool: &'static DbPool,
}

impl ResponseStore {
    pub fn new(pool: Option<&'static DbPool>) -> Self {
        Self {
            pool: pool.unwrap_or_else(get_pool),
        }
    }

    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn get(&self, response_id: &str) -> Result<Option<StoredResponse>> {
        let Some(row) = response::get_response(self.pool, response_id).await? else {
            return Ok(None);
        };

        let metadata: ResponseMetadata = row.metadata_as().unwrap_or_default();

        let history_item_ids = row.history_item_ids_vec();
        Ok(Some(StoredResponse {
            response_id: row.id,
            conversation_id: row.conversation_id,
            previous_response_id: row.previous_response_id,
            created_at: row.created_at,
            history_item_ids,
            metadata,
        }))
    }

    /// # Errors
    ///
    /// Returns an error if the response is not found or the database query fails.
    pub async fn get_or_raise(&self, response_id: &str) -> Result<StoredResponse> {
        self.get(response_id).await?.ok_or_else(|| {
            AgenticApiError::responses_api(
                format!("No response found with id '{response_id}'."),
                400,
                "invalid_request_error",
                Some("previous_response_id".into()),
                Some("previous_response_not_found".into()),
            )
        })
    }

    /// # Errors
    ///
    /// Returns an error if a database operation fails.
    pub async fn put_completed(
        &self,
        request: &ResponsesRequest,
        hydrated_request: &ResponsesRequest,
        response: &ResponsesResponse,
    ) -> Result<()> {
        if !matches!(response.status.as_str(), "completed" | "incomplete")
            || response.id.is_empty()
            || !request.response_store_enabled
        {
            return Ok(());
        }

        let normalized_input = normalize_input(&hydrated_request.input);
        let input_payloads = normalized_input.iter().map(ItemPayload::from_input);
        let output_payloads = response.output.iter().map(ItemPayload::from_output);

        let item_tuples: Vec<(String, String)> = input_payloads
            .chain(output_payloads)
            .map(|payload| (uuid7_str("item_"), payload.to_json_string()))
            .collect();

        let metadata = ResponseMetadata {
            model: response.model.clone(),
            previous_response_id: response.previous_response_id.clone(),
            effective_tools: hydrated_request.tools.clone(),
            effective_tool_choice: hydrated_request.tool_choice.clone(),
            effective_instructions: hydrated_request.instructions.clone(),
        };

        let item_ids: Vec<&str> = item_tuples.iter().map(|(id, _)| id.as_str()).collect();
        let history_str = serde_json::to_string(&item_ids).unwrap_or_default();
        let metadata_str = serde_json::to_string(&metadata).unwrap_or_default();
        let mut tx = self.pool.begin().await?;
        item::create_items_in_tx(&mut tx, &item_tuples).await?;
        response::create_response_in_tx(
            &mut tx,
            &response.id,
            response.conversation_id.as_deref(),
            response.previous_response_id.as_deref(),
            Some(&history_str),
            Some(&metadata_str),
        )
        .await?;
        tx.commit().await?;

        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn rehydrate(&self, stored: &StoredResponse) -> Result<Vec<InOutItem>> {
        if stored.history_item_ids.is_empty() {
            return Ok(vec![]);
        }

        let rows = item::get_items(self.pool, &stored.history_item_ids).await?;
        let by_id: HashMap<_, _> = rows.into_iter().map(|r| (r.id.clone(), r)).collect();

        if by_id.len() < stored.history_item_ids.len() {
            let missing: Vec<_> = stored
                .history_item_ids
                .iter()
                .filter(|id| !by_id.contains_key(*id))
                .collect();
            warn!(
                "rehydrate: {} item(s) missing for response {}: {:?}",
                missing.len(),
                stored.response_id,
                missing
            );
        }

        Ok(stored
            .history_item_ids
            .iter()
            .filter_map(|id| by_id.get(id).and_then(ItemPayload::from_item_row))
            .collect())
    }
}
