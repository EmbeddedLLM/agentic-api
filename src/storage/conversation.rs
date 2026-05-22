//! Conversation storage operations.

use std::sync::Arc;

use super::models::{conversation, item, response};
use super::pool::DbPool;
use super::types::{ConversationData, InOutItem, ResponseMetadata, Result, StorageError};
use crate::utils::common::{serialize_to_string, uuid7_str};

/// Conversation storage operations.
#[derive(Clone)]
pub struct ConversationStore {
    pool: Option<Arc<DbPool>>,
}

impl ConversationStore {
    /// Creates a disabled conversation store.
    #[must_use]
    pub fn disabled() -> Self {
        Self { pool: None }
    }

    /// Creates a new conversation store with database pool.
    #[must_use]
    pub fn new(pool: Arc<DbPool>) -> Self {
        Self { pool: Some(pool) }
    }

    /// Returns a reference to the database pool.
    ///
    /// # Errors
    ///
    /// Returns error if store is disabled (no pool configured).
    fn pool(&self) -> Result<&DbPool> {
        self.pool.as_deref().ok_or(StorageError::NotConfigured)
    }

    /// Creates a new conversation.
    ///
    /// # Errors
    ///
    /// Returns error if database query fails.
    pub async fn create(&self) -> Result<ConversationData> {
        let pool = self.pool()?;
        let row = conversation::create(pool, &uuid7_str("conv_")).await?;
        Ok(row.into())
    }

    /// Gets a conversation or creates it if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns error if database query fails.
    pub async fn get_or_create(&self, conversation_id: &str) -> Result<ConversationData> {
        let pool = self.pool()?;
        let row = conversation::get_or_create(pool, conversation_id).await?;
        Ok(row.into())
    }

    /// Gets a conversation by ID.
    ///
    /// # Errors
    ///
    /// Returns error if database query fails.
    pub async fn get(&self, conversation_id: &str) -> Result<Option<ConversationData>> {
        let pool = self.pool()?;
        let Some(row) = conversation::get(pool, conversation_id).await? else {
            return Ok(None);
        };
        Ok(Some(row.into()))
    }

    /// Rehydrates a conversation with all its items.
    ///
    /// # Errors
    ///
    /// Returns error if conversation not found or database query fails.
    pub async fn rehydrate(&self, conversation_id: &str) -> Result<Vec<InOutItem>> {
        let pool = self.pool()?;
        let rows = item::get_items_by_conversation(pool, conversation_id)
            .await
            .ok()
            .ok_or_else(|| StorageError::not_found("Conversation", conversation_id))?;

        Ok(rows.into_iter().filter_map(|row| row.as_inout()).collect())
    }

    /// Persists conversation turn with new items and response metadata.
    ///
    /// Creates items in the conversation and stores the associated response record.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if conversation not found or database operation fails.
    pub async fn persist(
        &self,
        conversation_id: &str,
        response_id: &str,
        previous_response_id: Option<&str>,
        new_items: Vec<InOutItem>,
        metadata: &ResponseMetadata,
    ) -> Result<()> {
        let pool = self.pool()?;
        let seq_start = item::conversation_item_count(pool, conversation_id)
            .await?
            .ok_or_else(|| StorageError::not_found("Conversation", conversation_id))?;

        let mut item_ids: Vec<String> = Vec::new();
        let items_: Vec<(String, String)> = new_items
            .into_iter()
            .map(|any_item| {
                let item_id = uuid7_str("item_");
                item_ids.push(item_id.clone());
                let data_str: String = (&any_item).into();
                (item_id, data_str)
            })
            .collect();

        let mut tx = pool.begin().await?;

        item::create_in_tx(&mut tx, items_, Some(conversation_id), Some(seq_start)).await?;

        let history_item_ids_json = serialize_to_string(&item_ids);
        let metadata_json: String = metadata.into();

        response::create_in_tx(
            &mut tx,
            response_id,
            Some(conversation_id),
            previous_response_id,
            Some(&history_item_ids_json),
            Some(&metadata_json),
        )
        .await?;
        tx.commit().await?;

        Ok(())
    }
}
