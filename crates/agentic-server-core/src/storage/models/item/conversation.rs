//! Item queries authorized through the owning conversation.

use super::{DbPool, DbResult, DbTransaction, Item};
use crate::types::conversations::ItemOrder;

/// List a page using a cursor sequence and the same ordering for filtering and sorting.
///
/// # Errors
/// Returns an error if the query fails.
pub async fn list_for_conversation(
    pool: &DbPool,
    tenant_id: &str,
    conversation_id: &str,
    limit: i64,
    after_sequence: Option<i64>,
    order: ItemOrder,
) -> DbResult<Vec<Item>> {
    let (comparison, direction) = match order {
        ItemOrder::Asc => (">", "ASC"),
        ItemOrder::Desc => ("<", "DESC"),
    };
    let sql = format!(
        "SELECT items.* FROM items JOIN conversations ON conversations.id = items.conversation_id \
         WHERE conversations.id = $1 AND conversations.tenant_id = $2 \
         AND (CAST($3 AS BIGINT) IS NULL OR items.seq {comparison} $3) ORDER BY items.seq {direction} LIMIT $4"
    );
    sqlx::query_as(&sql)
        .bind(conversation_id)
        .bind(tenant_id)
        .bind(after_sequence)
        .bind(limit)
        .fetch_all(pool)
        .await
}

/// Get an item through its conversation, including items written by Responses persistence.
///
/// # Errors
/// Returns an error if the query fails.
pub async fn get_for_conversation(
    pool: &DbPool,
    tenant_id: &str,
    conversation_id: &str,
    item_id: &str,
) -> DbResult<Option<Item>> {
    sqlx::query_as(
        "SELECT items.* FROM items JOIN conversations ON conversations.id = items.conversation_id \
         WHERE conversations.id = $1 AND conversations.tenant_id = $2 AND items.id = $3",
    )
    .bind(conversation_id)
    .bind(tenant_id)
    .bind(item_id)
    .fetch_optional(pool)
    .await
}

/// Remove one or all items from a locked conversation while preserving stored response history.
///
/// # Errors
/// Returns an error if the update fails.
pub async fn detach_from_conversation_in_tx(
    tx: &mut DbTransaction<'_>,
    conversation_id: &str,
    item_id: Option<&str>,
) -> DbResult<u64> {
    let result = sqlx::query(
        "UPDATE items SET conversation_id = NULL, seq = NULL \
         WHERE conversation_id = $1 AND (CAST($2 AS TEXT) IS NULL OR id = $2)",
    )
    .bind(conversation_id)
    .bind(item_id)
    .execute(&mut **tx)
    .await?;
    Ok(result.rows_affected())
}
