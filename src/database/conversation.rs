use serde_json::Value;

use super::db::{DbPool, DbResult};
use super::models::Conversation;
use crate::utils::common::utcnow_str;

/// # Errors
///
/// Returns a [`sqlx::Error`] if the query fails.
pub async fn create_conversation(pool: &DbPool, id: &str, metadata: Option<&Value>) -> DbResult<Conversation> {
    let now = utcnow_str();
    let metadata_str = metadata.map(|v| serde_json::to_string(v).unwrap_or_default());
    sqlx::query_as::<_, Conversation>(
        "INSERT INTO conversations (id, metadata, created_at, updated_at) \
         VALUES (?, ?, ?, ?) RETURNING *",
    )
    .bind(id)
    .bind(metadata_str)
    .bind(&now)
    .bind(&now)
    .fetch_one(pool)
    .await
}

/// # Errors
///
/// Returns a [`sqlx::Error`] if the query fails.
pub async fn get_or_create_conversation(pool: &DbPool, id: &str, metadata: Option<&Value>) -> DbResult<Conversation> {
    let now = utcnow_str();
    let metadata_str = metadata.map(|v| serde_json::to_string(v).unwrap_or_default());
    sqlx::query_as::<_, Conversation>(
        "INSERT INTO conversations (id, metadata, created_at, updated_at) \
         VALUES (?, ?, ?, ?) \
         ON CONFLICT (id) DO UPDATE SET updated_at = updated_at \
         RETURNING *",
    )
    .bind(id)
    .bind(metadata_str)
    .bind(&now)
    .bind(&now)
    .fetch_one(pool)
    .await
}

/// # Errors
///
/// Returns a [`sqlx::Error`] if the query fails.
pub async fn get_conversation(pool: &DbPool, id: &str) -> DbResult<Option<Conversation>> {
    sqlx::query_as::<_, Conversation>("SELECT * FROM conversations WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await
}
