//! Tenant-scoped conversation mutations, coordinated by `ConversationStore`.

use super::{Conversation, DbPool, DbResult, DbTransaction, utcnow_str};

/// Create a conversation in the same transaction as its initial items.
///
/// # Errors
/// Returns an error if insertion fails.
pub async fn create_with_metadata_in_tx(
    tx: &mut DbTransaction<'_>,
    id: &str,
    tenant_id: Option<&str>,
    metadata: Option<&str>,
) -> DbResult<Conversation> {
    sqlx::query_as::<_, Conversation>(
        "INSERT INTO conversations (id, tenant_id, metadata, created_at) VALUES ($1, $2, $3, $4) RETURNING *",
    )
    .bind(id)
    .bind(tenant_id)
    .bind(metadata)
    .bind(utcnow_str())
    .fetch_one(&mut **tx)
    .await
}

/// Get a conversation owned by the supplied tenant.
///
/// # Errors
/// Returns an error if the query fails.
pub async fn get_by_tenant(pool: &DbPool, tenant_id: &str, conversation_id: &str) -> DbResult<Option<Conversation>> {
    sqlx::query_as("SELECT * FROM conversations WHERE id = $1 AND tenant_id = $2")
        .bind(conversation_id)
        .bind(tenant_id)
        .fetch_optional(pool)
        .await
}

/// Replace conversation metadata.
///
/// # Errors
/// Returns an error if the query fails or the conversation is not owned by the tenant.
pub async fn update_metadata(
    pool: &DbPool,
    tenant_id: &str,
    conversation_id: &str,
    metadata: &str,
) -> DbResult<Conversation> {
    sqlx::query_as("UPDATE conversations SET metadata = $1 WHERE id = $2 AND tenant_id = $3 RETURNING *")
        .bind(metadata)
        .bind(conversation_id)
        .bind(tenant_id)
        .fetch_one(pool)
        .await
}

/// Delete a locked conversation after its items have been detached.
///
/// # Errors
/// Returns an error if deletion fails.
pub async fn delete_in_tx(tx: &mut DbTransaction<'_>, conversation_id: &str) -> DbResult<()> {
    sqlx::query("DELETE FROM conversations WHERE id = $1")
        .bind(conversation_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}

/// Advance the item revision while holding the conversation lock.
///
/// # Errors
/// Returns an error if the update fails.
pub async fn bump_revision_in_tx(tx: &mut DbTransaction<'_>, id: &str) -> DbResult<()> {
    sqlx::query("UPDATE conversations SET revision = revision + 1 WHERE id = $1")
        .bind(id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}
