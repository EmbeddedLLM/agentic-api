//! Domain type for conversation storage.

use super::super::models::Conversation as StorageDbConversation;

/// Domain entity for a stored conversation.
///
/// Represents a conversation context with metadata and history tracking.
#[derive(Debug, Clone)]
pub struct ConversationData {
    /// Unique conversation identifier
    pub conversation_id: String,
    /// Creation timestamp in ISO 8601 format
    pub created_at: String,
}

impl From<StorageDbConversation> for ConversationData {
    fn from(row: StorageDbConversation) -> Self {
        Self {
            conversation_id: row.id,
            created_at: row.created_at,
        }
    }
}

impl From<ConversationData> for StorageDbConversation {
    fn from(data: ConversationData) -> Self {
        Self {
            id: data.conversation_id,
            created_at: data.created_at.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_from_db_conversation() {
        let db_row = StorageDbConversation {
            id: "conv_123".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let conversation: ConversationData = db_row.into();
        assert_eq!(conversation.conversation_id, "conv_123");
        assert_eq!(conversation.created_at, "2024-01-01T00:00:00Z");
    }

    #[test]
    fn test_conversation_roundtrip() {
        let data = ConversationData {
            conversation_id: "conv_456".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let db_row: StorageDbConversation = data.into();
        assert_eq!(db_row.id, "conv_456");
        assert_eq!(db_row.created_at, "2024-01-01T00:00:00Z");
    }
}
