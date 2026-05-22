//! Conversation history item stored in the database.

use super::super::pool::{DbPool, DbResult, DbTransaction};
use super::super::types::item::InOutItem;
use crate::types::io::{InputItem, OutputItem};
use crate::utils::common::{deserialize_from_str_opt, utcnow_str};

/// Conversation history item stored in the database.
///
/// Maps to the `items` table and represents a single message/event
/// in a conversation timeline.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Item {
    /// Unique identifier for this item.
    pub id: String,

    /// Item data stored as JSON text.
    /// Deserialized based on context (`message`, `tool_call`, etc.)
    pub data: String,

    /// Creation timestamp in ISO 8601 format.
    pub created_at: String,

    /// Optional conversation ID for grouping items.
    pub conversation_id: Option<String>,

    /// Optional sequence number within conversation.
    pub seq: Option<i64>,
}

impl Item {
    /// Deserialize data column as `InputItem`.
    #[must_use]
    pub fn as_input(&self) -> Option<InputItem> {
        deserialize_from_str_opt(&self.data)
    }

    /// Deserialize data column as `OutputItem`.
    #[must_use]
    pub fn as_output(&self) -> Option<OutputItem> {
        deserialize_from_str_opt(&self.data)
    }

    /// Deserialize data column as either `InputItem` or `OutputItem`.
    #[must_use]
    pub fn as_inout(&self) -> Option<InOutItem> {
        self.as_input()
            .map(InOutItem::Input)
            .or_else(|| self.as_output().map(InOutItem::Output))
    }
}

/// Create items in a transaction with optional conversation context.
///
/// If `conversation_id` and `seq_start` are provided, items are created with sequence numbers.
/// Otherwise, items are created without conversation context.
///
/// # Errors
/// Returns `DbResult::Err` if the database insertion fails.
pub async fn create_in_tx(
    tx: &mut DbTransaction<'_>,
    items: Vec<(String, String)>,
    conversation_id: Option<&str>,
    seq_start: Option<i64>,
) -> DbResult<Vec<Item>> {
    let now = utcnow_str();
    let mut created_items = Vec::with_capacity(items.len());

    for (idx, (id, data)) in items.into_iter().enumerate() {
        let item = match (conversation_id, seq_start) {
            (Some(conv_id), Some(start_seq)) => {
                #[allow(clippy::cast_possible_wrap)]
                let seq = start_seq + idx as i64;
                sqlx::query_as::<_, Item>(
                    "INSERT INTO items (id, data, created_at, conversation_id, seq) VALUES (?, ?, ?, ?, ?) RETURNING *",
                )
                .bind(&id)
                .bind(&data)
                .bind(&now)
                .bind(conv_id)
                .bind(seq)
                .fetch_one(&mut **tx)
                .await?
            }
            _ => {
                sqlx::query_as::<_, Item>("INSERT INTO items (id, data, created_at) VALUES (?, ?, ?) RETURNING *")
                    .bind(&id)
                    .bind(&data)
                    .bind(&now)
                    .fetch_one(&mut **tx)
                    .await?
            }
        };
        created_items.push(item);
    }
    Ok(created_items)
}

/// Get items by IDs.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn get_items(pool: &DbPool, ids: &[String]) -> DbResult<Vec<Item>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!("SELECT * FROM items WHERE id IN ({placeholders})");
    let mut q = sqlx::query_as::<_, Item>(&sql);
    for id in ids {
        q = q.bind(id);
    }
    q.fetch_all(pool).await
}

/// Get items by conversation ID ordered by sequence.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn get_items_by_conversation(pool: &DbPool, conversation_id: &str) -> DbResult<Vec<Item>> {
    sqlx::query_as::<_, Item>("SELECT * FROM items WHERE conversation_id = ? ORDER BY seq ASC")
        .bind(conversation_id)
        .fetch_all(pool)
        .await
}

/// Get count of items for a conversation.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn conversation_item_count(pool: &DbPool, conversation_id: &str) -> DbResult<Option<i64>> {
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM items WHERE conversation_id = ?")
        .bind(conversation_id)
        .fetch_optional(pool)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_basic() {
        let item = Item {
            id: "item_123".to_string(),
            data: r#"{"role":"user","content":"hello"}"#.to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            conversation_id: Some("conv_456".to_string()),
            seq: Some(1),
        };

        assert_eq!(item.id, "item_123");
        assert_eq!(item.conversation_id, Some("conv_456".to_string()));
        assert_eq!(item.seq, Some(1));
    }

    #[test]
    fn test_item_optional_fields() {
        let item = Item {
            id: "item_789".to_string(),
            data: r#"{"role":"assistant"}"#.to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            conversation_id: None,
            seq: None,
        };

        assert!(item.conversation_id.is_none());
        assert!(item.seq.is_none());
    }
}
