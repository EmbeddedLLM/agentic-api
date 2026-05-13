use crate::database::db::get_pool;
use crate::database::{conversation, item, response};
use crate::store::response::ResponseMetadata;
use crate::store::translator::{InOutItem, ItemPayload};
use crate::utils::common::uuid7_str;
use crate::utils::errors::AgenticApiError;

type Result<T> = std::result::Result<T, AgenticApiError>;

#[derive(Debug, Clone)]
pub struct StoredConversation {
    pub conversation_id: String,
    pub created_at: String,
    pub metadata: Option<ResponseMetadata>,
}

pub struct ConversationStore {
    pool: &'static crate::database::db::DbPool,
}

impl ConversationStore {
    pub fn new(pool: Option<&'static crate::database::db::DbPool>) -> Self {
        Self {
            pool: pool.unwrap_or_else(get_pool),
        }
    }

    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn create(&self) -> Result<StoredConversation> {
        let row = conversation::create_conversation(self.pool, &uuid7_str("conv_"), None).await?;
        Ok(StoredConversation {
            conversation_id: row.id,
            created_at: row.created_at,
            metadata: None,
        })
    }

    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn get_or_create(&self, conversation_id: &str) -> Result<StoredConversation> {
        let row = conversation::get_or_create_conversation(self.pool, conversation_id, None).await?;
        let metadata = row.metadata_as();
        Ok(StoredConversation {
            conversation_id: row.id,
            created_at: row.created_at,
            metadata,
        })
    }

    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn get(&self, conversation_id: &str) -> Result<Option<StoredConversation>> {
        let Some(row) = conversation::get_conversation(self.pool, conversation_id).await? else {
            return Ok(None);
        };
        let metadata = row.metadata_as();
        Ok(Some(StoredConversation {
            conversation_id: row.id,
            created_at: row.created_at,
            metadata,
        }))
    }

    /// # Errors
    ///
    /// Returns an error if the conversation is not found or a database operation fails.
    pub async fn put_turn(
        &self,
        conversation_id: &str,
        response_id: &str,
        previous_response_id: Option<&str>,
        new_items: &[InOutItem],
        metadata: &ResponseMetadata,
    ) -> Result<()> {
        let seq_start = item::conversation_item_count(self.pool, conversation_id)
            .await?
            .ok_or_else(|| AgenticApiError::bad_input(format!("Conversation not found: {conversation_id}")))?;

        let item_tuples: Vec<(String, String)> = new_items
            .iter()
            .map(|any_item| {
                let payload = match any_item {
                    InOutItem::Input(i) => ItemPayload::from_input(i),
                    InOutItem::Output(o) => ItemPayload::from_output(o),
                };
                (uuid7_str("item_"), payload.to_json_string())
            })
            .collect();

        let item_ids: Vec<&str> = item_tuples.iter().map(|(id, _)| id.as_str()).collect();
        let history_str = serde_json::to_string(&item_ids).unwrap_or_default();
        let metadata_str = serde_json::to_string(metadata).unwrap_or_default();

        let mut tx = self.pool.begin().await?;
        item::create_conversation_items_in_tx(&mut tx, &item_tuples, conversation_id, seq_start).await?;
        response::create_response_in_tx(
            &mut tx,
            response_id,
            Some(conversation_id),
            previous_response_id,
            Some(&history_str),
            Some(&metadata_str),
        )
        .await?;
        tx.commit().await?;

        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if the conversation is not found or a database query fails.
    pub async fn rehydrate(&self, conversation_id: &str) -> Result<Vec<InOutItem>> {
        self.get(conversation_id).await?.ok_or_else(|| {
            AgenticApiError::responses_api(
                format!("Conversation '{conversation_id}' not found."),
                400,
                "invalid_request_error",
                Some("conversation_id".into()),
                Some("conversation_not_found".into()),
            )
        })?;

        Ok(item::get_items_by_conversation(self.pool, conversation_id)
            .await?
            .into_iter()
            .filter_map(|row| ItemPayload::from_item_row(&row))
            .collect())
    }
}
