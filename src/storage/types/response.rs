//! Domain type for response storage.

use serde::{Deserialize, Serialize};
use serde_json;

use super::super::models::Response as StorageDbResponse;
use crate::types::io::{ResponsesTool, ToolChoice};

/// Response metadata with effective configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseMetadata {
    pub model: String,
    pub previous_response_id: Option<String>,
    pub effective_tools: Option<Vec<ResponsesTool>>,
    pub effective_tool_choice: ToolChoice,
    pub effective_instructions: Option<String>,
}

/// Domain entity for a stored LLM response.
#[derive(Debug, Clone)]
pub struct ResponseData {
    /// Unique response identifier
    pub response_id: String,
    /// Optional conversation this response belongs to
    pub conversation_id: Option<String>,
    /// Optional reference to previous response for chaining
    pub previous_response_id: Option<String>,
    /// Creation timestamp in ISO 8601 format
    pub created_at: String,
    /// Deserialized history item IDs (vec of item IDs)
    pub history_item_ids: Vec<String>,
    /// Response metadata with effective configuration (fully typed)
    pub metadata: ResponseMetadata,
}

impl From<StorageDbResponse> for ResponseData {
    fn from(row: StorageDbResponse) -> Self {
        let history_item_ids = row.history_item_ids_vec();
        let metadata = row.metadata_as::<ResponseMetadata>().unwrap_or_default();

        Self {
            response_id: row.id,
            conversation_id: row.conversation_id,
            previous_response_id: row.previous_response_id,
            created_at: row.created_at,
            history_item_ids,
            metadata,
        }
    }
}

impl From<&ResponseMetadata> for String {
    fn from(metadata: &ResponseMetadata) -> Self {
        serde_json::to_string(metadata).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_data_from_db_response() {
        let db_row = StorageDbResponse {
            id: "resp_123".to_string(),
            conversation_id: Some("conv_456".to_string()),
            previous_response_id: None,
            history_item_ids: Some(r#"["item_1"]"#.to_string()),
            metadata: Some(
                r#"{"model":"gpt-4","previous_response_id":null,"effective_tools":null,"effective_tool_choice":"auto","effective_instructions":null}"#
                    .to_string(),
            ),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let response: ResponseData = db_row.into();
        assert_eq!(response.response_id, "resp_123");
        assert_eq!(response.conversation_id, Some("conv_456".to_string()));
        assert_eq!(response.created_at, "2024-01-01T00:00:00Z");
        assert_eq!(response.history_item_ids, vec!["item_1".to_string()]);
        assert_eq!(response.metadata.model, "gpt-4");
    }

    #[test]
    fn test_response_data_from_db_response_optional_fields() {
        let db_row = StorageDbResponse {
            id: "resp_789".to_string(),
            conversation_id: None,
            previous_response_id: None,
            history_item_ids: None,
            metadata: None,
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let response: ResponseData = db_row.into();
        assert_eq!(response.response_id, "resp_789");
        assert!(response.conversation_id.is_none());
        assert!(response.history_item_ids.is_empty());
        assert_eq!(response.metadata.model, "");
    }
}
