use super::db::{DbPool, DbResult, DbTransaction};
use super::models::Response;
use crate::utils::common::utcnow_str;

/// # Errors
///
/// Returns a [`sqlx::Error`] if the query fails.
pub async fn create_response_in_tx(
    tx: &mut DbTransaction<'_>,
    id: &str,
    conversation_id: Option<&str>,
    previous_response_id: Option<&str>,
    history_item_ids: Option<&str>,
    metadata: Option<&str>,
) -> DbResult<()> {
    let now = utcnow_str();
    sqlx::query(
        "INSERT INTO responses \
         (id, conversation_id, previous_response_id, history_item_ids, metadata, created_at, updated_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(id)
    .bind(conversation_id)
    .bind(previous_response_id)
    .bind(history_item_ids)
    .bind(metadata)
    .bind(&now)
    .bind(&now)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// # Errors
///
/// Returns a [`sqlx::Error`] if the query fails.
pub async fn get_response(pool: &DbPool, id: &str) -> DbResult<Option<Response>> {
    sqlx::query_as::<_, Response>("SELECT * FROM responses WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await
}
