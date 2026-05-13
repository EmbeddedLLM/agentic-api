use super::db::{DbPool, DbResult, DbTransaction};
use super::models::Item;
use crate::utils::common::utcnow_str;

/// # Errors
///
/// Returns a [`sqlx::Error`] if any query fails.
pub async fn create_items_in_tx(tx: &mut DbTransaction<'_>, items: &[(String, String)]) -> DbResult<()> {
    let now = utcnow_str();
    for (id, data_str) in items {
        sqlx::query("INSERT INTO items (id, data, created_at) VALUES (?, ?, ?)")
            .bind(id)
            .bind(data_str)
            .bind(&now)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

/// # Errors
///
/// Returns a [`sqlx::Error`] if any query fails.
pub async fn create_conversation_items_in_tx(
    tx: &mut DbTransaction<'_>,
    items: &[(String, String)],
    conversation_id: &str,
    seq_start: i64,
) -> DbResult<()> {
    let now = utcnow_str();
    for (i, (id, data_str)) in items.iter().enumerate() {
        #[allow(clippy::cast_possible_wrap)]
        let seq = seq_start + i as i64;
        sqlx::query("INSERT INTO items (id, data, created_at, conversation_id, seq) VALUES (?, ?, ?, ?, ?)")
            .bind(id)
            .bind(data_str)
            .bind(&now)
            .bind(conversation_id)
            .bind(seq)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

/// Returns the item count for a conversation if it exists, or `None` if not found.
///
/// # Errors
///
/// Returns a [`sqlx::Error`] if the query fails.
pub async fn conversation_item_count(pool: &DbPool, conversation_id: &str) -> DbResult<Option<i64>> {
    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT COUNT(i.id) FROM conversations c \
         LEFT JOIN items i ON i.conversation_id = c.id \
         WHERE c.id = ? \
         GROUP BY c.id",
    )
    .bind(conversation_id)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|(n,)| n))
}

/// # Errors
///
/// Returns a [`sqlx::Error`] if the query fails.
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

/// # Errors
///
/// Returns a [`sqlx::Error`] if the query fails.
pub async fn get_items_by_conversation(pool: &DbPool, conversation_id: &str) -> DbResult<Vec<Item>> {
    sqlx::query_as::<_, Item>("SELECT * FROM items WHERE conversation_id = ? ORDER BY seq ASC")
        .bind(conversation_id)
        .fetch_all(pool)
        .await
}
