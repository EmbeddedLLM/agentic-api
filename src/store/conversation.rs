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

    pub async fn create(&self) -> Result<StoredConversation> {
        let row = conversation::create_conversation(self.pool, &uuid7_str("conv_"), None).await?;
        Ok(StoredConversation {
            conversation_id: row.id,
            created_at: row.created_at,
            metadata: None,
        })
    }

    pub async fn get_or_create(&self, conversation_id: &str) -> Result<StoredConversation> {
        let row = conversation::get_or_create_conversation(self.pool, conversation_id, None).await?;
        let metadata = row.metadata_json().and_then(|v| serde_json::from_value(v).ok());
        Ok(StoredConversation {
            conversation_id: row.id,
            created_at: row.created_at,
            metadata,
        })
    }

    pub async fn get(&self, conversation_id: &str) -> Result<Option<StoredConversation>> {
        let Some(row) = conversation::get_conversation(self.pool, conversation_id).await? else {
            return Ok(None);
        };
        let metadata = row.metadata_json().and_then(|v| serde_json::from_value(v).ok());
        Ok(Some(StoredConversation {
            conversation_id: row.id,
            created_at: row.created_at,
            metadata,
        }))
    }

    pub async fn put_turn(
        &self,
        conversation_id: &str,
        response_id: &str,
        previous_response_id: Option<&str>,
        new_items: &[InOutItem],
        metadata: &ResponseMetadata,
    ) -> Result<()> {
        let stored = self
            .get(conversation_id)
            .await?
            .ok_or_else(|| AgenticApiError::bad_input(format!("Conversation not found: {conversation_id}")))?;

        let existing = item::get_items_by_conversation(self.pool, &stored.conversation_id).await?;
        let seq_start = existing.len() as i64;

        let item_tuples: Vec<(String, serde_json::Value)> = new_items
            .iter()
            .map(|any_item| {
                let payload = match any_item {
                    InOutItem::Input(i) => ItemPayload::from_input(i.clone()),
                    InOutItem::Output(o) => ItemPayload::from_output(o.clone()),
                };
                (uuid7_str("item_"), payload.to_json_value())
            })
            .collect();

        let item_ids: Vec<String> = item_tuples.iter().map(|(id, _)| id.clone()).collect();
        let history_value = serde_json::to_value(&item_ids).ok();
        let metadata_value = serde_json::to_value(metadata).ok();

        let mut tx = self.pool.begin().await?;
        item::create_conversation_items_in_tx(&mut tx, &item_tuples, conversation_id, seq_start).await?;
        response::create_response_in_tx(
            &mut tx,
            response_id,
            Some(conversation_id),
            previous_response_id,
            history_value.as_ref(),
            metadata_value.as_ref(),
        )
        .await?;
        tx.commit().await?;

        Ok(())
    }

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
            .filter_map(ItemPayload::from_item_row)
            .collect())
    }
}
