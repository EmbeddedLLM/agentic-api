use serde_json::Value;

use super::db::{DbPool, DbResult, DbTransaction};
use super::models::Response;
use crate::utils::common::utcnow_str;

pub async fn create_response_in_tx(
    tx: &mut DbTransaction<'_>,
    id: &str,
    conversation_id: Option<&str>,
    previous_response_id: Option<&str>,
    history_item_ids: Option<&Value>,
    metadata: Option<&Value>,
) -> DbResult<()> {
    let now = utcnow_str();
    let history_str = history_item_ids.map(|v| serde_json::to_string(v).unwrap_or_default());
    let metadata_str = metadata.map(|v| serde_json::to_string(v).unwrap_or_default());
    sqlx::query(
        "INSERT INTO responses \
         (id, conversation_id, previous_response_id, history_item_ids, metadata, created_at, updated_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(id)
    .bind(conversation_id)
    .bind(previous_response_id)
    .bind(history_str)
    .bind(metadata_str)
    .bind(&now)
    .bind(&now)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

pub async fn get_response(pool: &DbPool, id: &str) -> DbResult<Option<Response>> {
    sqlx::query_as::<_, Response>("SELECT * FROM responses WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await
}

pub async fn get_responses_by_conversation(pool: &DbPool, conversation_id: &str) -> DbResult<Vec<Response>> {
    sqlx::query_as::<_, Response>("SELECT * FROM responses WHERE conversation_id = ? ORDER BY created_at ASC")
        .bind(conversation_id)
        .fetch_all(pool)
        .await
}

pub async fn delete_response(pool: &DbPool, id: &str) -> DbResult<()> {
    sqlx::query("DELETE FROM responses WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}
