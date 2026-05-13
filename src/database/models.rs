#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Item {
    pub id: String,
    pub data: String,       // JSON stored as TEXT
    pub created_at: String, // ISO 8601 timestamp stored as TEXT
    pub conversation_id: Option<String>,
    pub seq: Option<i64>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Response {
    pub id: String,
    pub conversation_id: Option<String>,
    pub previous_response_id: Option<String>,
    pub history_item_ids: Option<String>, // JSON array stored as TEXT
    pub metadata: Option<String>,         // JSON object stored as TEXT
    pub created_at: String,
    pub updated_at: String,
}

impl Response {
    #[must_use]
    pub fn history_item_ids_vec(&self) -> Vec<String> {
        self.history_item_ids
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn metadata_as<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        self.metadata.as_deref().and_then(|s| serde_json::from_str(s).ok())
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Conversation {
    pub id: String,
    pub metadata: Option<String>, // JSON object stored as TEXT
    pub created_at: String,
    pub updated_at: String,
}

impl Conversation {
    #[must_use]
    pub fn metadata_as<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        self.metadata.as_deref().and_then(|s| serde_json::from_str(s).ok())
    }
}
