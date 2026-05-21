//! LLM API response stored in the database.

use super::super::pool::{DbPool, DbResult, DbTransaction};
use crate::utils::common::{from_json_string_opt, from_json_string_opt_or_default, utcnow_str};

/// LLM API response stored in the database.
///
/// Maps to the `responses` table and represents a single API response
/// with its metadata and history chain.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Response {
    /// Unique response identifier.
    pub id: String,

    /// Optional conversation this response belongs to.
    pub conversation_id: Option<String>,

    /// Optional reference to previous response for chaining.
    pub previous_response_id: Option<String>,

    /// History item IDs as JSON array string.
    pub history_item_ids: Option<String>,

    /// Response metadata as JSON object string.
    pub metadata: Option<String>,

    /// Creation timestamp in ISO 8601 format.
    pub created_at: String,
}

/// Create a response in a transaction and return it.
///
/// # Errors
/// Returns `DbResult::Err` if the database insertion fails.
pub async fn create_in_tx(
    tx: &mut DbTransaction<'_>,
    id: &str,
    conversation_id: Option<&str>,
    previous_response_id: Option<&str>,
    history_item_ids: Option<&str>,
    metadata: Option<&str>,
) -> DbResult<Response> {
    let now = utcnow_str();
    sqlx::query_as::<_, Response>(
        "INSERT INTO responses \
         (id, conversation_id, previous_response_id, history_item_ids, metadata, created_at) \
         VALUES (?, ?, ?, ?, ?, ?) RETURNING *",
    )
    .bind(id)
    .bind(conversation_id)
    .bind(previous_response_id)
    .bind(history_item_ids)
    .bind(metadata)
    .bind(&now)
    .fetch_one(&mut **tx)
    .await
}

/// Get a response by ID.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn get(pool: &DbPool, id: &str) -> DbResult<Option<Response>> {
    sqlx::query_as::<_, Response>("SELECT * FROM responses WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await
}

impl Response {
    /// Deserialize `history_item_ids` from JSON string to Vec<String>.
    #[must_use]
    pub fn history_item_ids_vec(&self) -> Vec<String> {
        from_json_string_opt_or_default(&self.history_item_ids)
    }

    /// Deserialize metadata from JSON string to the given type.
    #[must_use]
    pub fn metadata_as<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        from_json_string_opt(&self.metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_history_ids_empty() {
        let response = Response {
            id: "test".to_string(),
            conversation_id: None,
            previous_response_id: None,
            history_item_ids: None,
            metadata: None,
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let ids: Vec<String> = response.history_item_ids_vec();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_response_history_ids_valid() {
        let response = Response {
            id: "test".to_string(),
            conversation_id: None,
            previous_response_id: None,
            history_item_ids: Some(r#"["item_1", "item_2"]"#.to_string()),
            metadata: None,
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let ids = response.history_item_ids_vec();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], "item_1");
    }

    #[test]
    fn test_response_metadata_deserialize() {
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct TestMeta {
            model: String,
        }

        let response = Response {
            id: "resp_1".to_string(),
            conversation_id: None,
            previous_response_id: None,
            history_item_ids: None,
            metadata: Some(r#"{"model":"gpt-4"}"#.to_string()),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let meta: Option<TestMeta> = response.metadata_as();
        assert!(meta.is_some());
    }
}
