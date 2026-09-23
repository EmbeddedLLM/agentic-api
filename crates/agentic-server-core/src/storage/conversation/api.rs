//! Conversation CRUD and ordered item mutations.

use super::{
    ConversationData, ConversationStore, InOutItem, StorageError, StoreResult, conversation, item, serialize_to_string,
    uuid7_str,
};
use crate::storage::{DbTransaction, Item};
use crate::types::conversations::{ConversationMetadata, ItemOrder};

impl ConversationStore {
    /// Create a conversation and its initial items atomically.
    ///
    /// # Errors
    /// Returns an error if serialization or persistence fails.
    pub async fn create_with_metadata_and_items(
        &self,
        tenant_id: Option<&str>,
        metadata: Option<ConversationMetadata>,
        initial_items: Vec<InOutItem>,
    ) -> StoreResult<ConversationData> {
        let items = serialize_items(initial_items)?;
        let metadata = metadata.map(|value| serialize_to_string(&value)).transpose()?;
        let id = uuid7_str("conv_");
        let mut tx = self.pool()?.begin().await?;
        let row = conversation::create_with_metadata_in_tx(&mut tx, &id, tenant_id, metadata.as_deref()).await?;
        if !items.is_empty() {
            item::create_in_tx(&mut tx, items, Some(&id)).await?;
            conversation::bump_revision_in_tx(&mut tx, &id).await?;
        }
        tx.commit().await?;
        Ok(row.into())
    }

    /// Retrieve a conversation owned by the tenant.
    ///
    /// # Errors
    /// Returns an error if the resource is missing or the query fails.
    pub async fn retrieve(&self, tenant_id: &str, id: &str) -> StoreResult<ConversationData> {
        conversation::get_by_tenant(self.pool()?, tenant_id, id)
            .await?
            .map(Into::into)
            .ok_or_else(|| StorageError::not_found("Conversation", id))
    }

    /// Replace metadata on a conversation owned by the tenant.
    ///
    /// # Errors
    /// Returns an error if the resource is missing or the update fails.
    pub async fn update_metadata(
        &self,
        tenant_id: &str,
        id: &str,
        metadata: ConversationMetadata,
    ) -> StoreResult<ConversationData> {
        let metadata = serialize_to_string(&metadata)?;
        conversation::update_metadata(self.pool()?, tenant_id, id, &metadata)
            .await
            .map(Into::into)
            .map_err(|error| conversation_error(error, id))
    }

    /// Delete a conversation without deleting items referenced by stored responses.
    ///
    /// # Errors
    /// Returns an error if the resource is missing or the transaction fails.
    pub async fn delete(&self, tenant_id: &str, id: &str) -> StoreResult<()> {
        let mut tx = self.pool()?.begin().await?;
        lock_owned(&mut tx, tenant_id, id).await?;
        item::detach_from_conversation_in_tx(&mut tx, id, None).await?;
        conversation::delete_in_tx(&mut tx, id).await?;
        tx.commit().await?;
        Ok(())
    }

    /// Append items in order, sharing sequence allocation with Responses persistence.
    ///
    /// # Errors
    /// Returns an error if the resource is missing or persistence fails.
    pub async fn create_items(&self, tenant_id: &str, id: &str, items: Vec<InOutItem>) -> StoreResult<Vec<Item>> {
        let items = serialize_items(items)?;
        let mut tx = self.pool()?.begin().await?;
        lock_owned(&mut tx, tenant_id, id).await?;
        let rows = item::create_in_tx(&mut tx, items, Some(id)).await?;
        if !rows.is_empty() {
            conversation::bump_revision_in_tx(&mut tx, id).await?;
        }
        tx.commit().await?;
        Ok(rows)
    }

    /// List a page, resolving the cursor within the authorized conversation.
    ///
    /// # Errors
    /// Returns an error for a missing conversation/cursor or a failed query.
    pub async fn list_items(
        &self,
        tenant_id: &str,
        id: &str,
        limit: i64,
        after: Option<&str>,
        order: ItemOrder,
    ) -> StoreResult<Vec<Item>> {
        self.retrieve(tenant_id, id).await?;
        let after_sequence = match after {
            Some(cursor) => {
                let row = self.retrieve_item(tenant_id, id, cursor).await?;
                Some(row.seq.ok_or_else(|| StorageError::InvalidConversationSequence {
                    conversation_id: id.to_owned(),
                    item_id: cursor.to_owned(),
                })?)
            }
            None => None,
        };
        Ok(item::list_for_conversation(self.pool()?, tenant_id, id, limit, after_sequence, order).await?)
    }

    /// Retrieve one item belonging to the authorized conversation.
    ///
    /// # Errors
    /// Returns an error if the item is missing or the query fails.
    pub async fn retrieve_item(&self, tenant_id: &str, id: &str, item_id: &str) -> StoreResult<Item> {
        item::get_for_conversation(self.pool()?, tenant_id, id, item_id)
            .await?
            .ok_or_else(|| StorageError::not_found("Conversation item", item_id))
    }

    /// Remove an item from conversation history and advance the snapshot revision.
    ///
    /// # Errors
    /// Returns an error if the resource is missing or the transaction fails.
    pub async fn delete_item(&self, tenant_id: &str, id: &str, item_id: &str) -> StoreResult<ConversationData> {
        let mut tx = self.pool()?.begin().await?;
        let row = lock_owned(&mut tx, tenant_id, id).await?;
        if item::detach_from_conversation_in_tx(&mut tx, id, Some(item_id)).await? == 0 {
            return Err(StorageError::not_found("Conversation item", item_id));
        }
        conversation::bump_revision_in_tx(&mut tx, id).await?;
        tx.commit().await?;
        Ok(row.into())
    }
}

fn serialize_items(items: Vec<InOutItem>) -> StoreResult<Vec<(String, String)>> {
    items
        .into_iter()
        .map(|item| Ok((uuid7_str("item_"), String::try_from(&item)?)))
        .collect()
}

async fn lock_owned(tx: &mut DbTransaction<'_>, tenant_id: &str, id: &str) -> StoreResult<conversation::Conversation> {
    let row = conversation::lock_in_tx(tx, id)
        .await
        .map_err(|error| conversation_error(error, id))?;
    if row.tenant_id.as_deref() != Some(tenant_id) {
        return Err(StorageError::not_found("Conversation", id));
    }
    Ok(row)
}

fn conversation_error(error: sqlx::Error, id: &str) -> StorageError {
    match error {
        sqlx::Error::RowNotFound => StorageError::not_found("Conversation", id),
        other => other.into(),
    }
}
