use chrono::Utc;
use uuid::Uuid;

#[must_use]
pub fn uuid7_str(prefix: &str) -> String {
    format!("{}{}", prefix, Uuid::now_v7())
}

#[must_use]
pub fn utcnow_str() -> String {
    Utc::now().to_rfc3339()
}

/// Serialize any type to JSON string, returning empty string on error.
#[must_use]
pub fn serialize_to_string<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// Deserialize JSON string to any type, returning None on error.
#[must_use]
pub fn deserialize_from_str_opt<T: serde::de::DeserializeOwned>(json_str: &str) -> Option<T> {
    serde_json::from_str(json_str).ok()
}

/// Deserialize optional JSON String to any type, returning default on error or if None.
#[must_use]
pub fn deserialize_from_string_opt_or_default<T: serde::de::DeserializeOwned + Default>(
    json_str: &Option<String>,
) -> T {
    json_str
        .as_ref()
        .and_then(|s| deserialize_from_str_opt::<T>(s))
        .unwrap_or_default()
}

/// Deserialize optional JSON String to any type, returning None on error or if None.
#[must_use]
pub fn deserialize_from_string_opt<T: serde::de::DeserializeOwned>(json_str: &Option<String>) -> Option<T> {
    json_str.as_ref().and_then(|s| deserialize_from_str_opt::<T>(s))
}
