//! Conversation context and history.

use super::super::pool::{DbPool, DbResult, DbTransaction};
use crate::storage::backend::DatabaseBackend;
use crate::utils::common::utcnow_str;

/// Conversation context and history.
///
/// Maps to the `conversations` table and represents a logical conversation
/// containing multiple responses and items.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Conversation {
    /// Unique conversation identifier.
    pub id: String,

    /// Optional metadata as JSON string.
    pub metadata: Option<String>,

    /// Creation timestamp as Unix timestamp in seconds.
    pub created_at: i64,
}

/// Create a new conversation.
///
/// # Errors
/// Returns `DbResult::Err` if the database insertion fails.
pub async fn create(pool: &DbPool, id: &str) -> DbResult<Conversation> {
    let now = utcnow_str();
    sqlx::query_as::<_, Conversation>(
        "INSERT INTO conversations (id, created_at) \
         VALUES ($1, $2) RETURNING *",
    )
    .bind(id)
    .bind(now)
    .fetch_one(pool)
    .await
}

/// Get or create a conversation.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn get_or_create(pool: &DbPool, id: &str) -> DbResult<Conversation> {
    let now = utcnow_str();
    sqlx::query_as::<_, Conversation>(
        "INSERT INTO conversations (id, created_at) \
         VALUES ($1, $2) \
         ON CONFLICT (id) DO UPDATE SET created_at = created_at \
         RETURNING *",
    )
    .bind(id)
    .bind(now)
    .fetch_one(pool)
    .await
}

/// Get a conversation by ID.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn get(pool: &DbPool, id: &str) -> DbResult<Option<Conversation>> {
    sqlx::query_as::<_, Conversation>("SELECT * FROM conversations WHERE id = $1")
        .bind(id)
        .fetch_optional(pool)
        .await
}

/// Locks an existing conversation for the lifetime of the transaction.
///
/// `PostgreSQL` takes a row lock without writing the row. `SQLite` uses a no-op
/// update to acquire its database-wide write lock, which serializes persistence
/// across all conversations. Both protect sequence allocation when multiple
/// gateway replicas persist turns concurrently, but with different lock granularity.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails or the conversation does not exist.
pub async fn lock_in_tx(tx: &mut DbTransaction<'_>, id: &str) -> DbResult<()> {
    if DatabaseBackend::from_connection(tx.as_mut()) == DatabaseBackend::Postgres {
        let locked_id = sqlx::query_scalar::<_, String>("SELECT id FROM conversations WHERE id = $1 FOR UPDATE")
            .bind(id)
            .fetch_optional(&mut **tx)
            .await?;
        return locked_id.map(|_| ()).ok_or(sqlx::Error::RowNotFound);
    }

    let result = sqlx::query("UPDATE conversations SET created_at = created_at WHERE id = $1")
        .bind(id)
        .execute(&mut **tx)
        .await?;
    if result.rows_affected() == 0 {
        return Err(sqlx::Error::RowNotFound);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_basic() {
        let conversation = Conversation {
            id: "conv_1".to_string(),
            metadata: None,
            created_at: 1_704_067_200,
        };

        assert_eq!(conversation.id, "conv_1");
        assert!(conversation.metadata.is_none());
        assert_eq!(conversation.created_at, 1_704_067_200);
    }
}
