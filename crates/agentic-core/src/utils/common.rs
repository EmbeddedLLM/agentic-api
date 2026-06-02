use chrono::Utc;
use tracing::warn;
use uuid::Uuid;

#[must_use]
pub fn uuid7_str(prefix: &str) -> String {
    format!("{}{}", prefix, Uuid::now_v7())
}

#[must_use]
pub fn utcnow_str() -> i64 {
    Utc::now().timestamp()
}

/// Serialize any type to JSON string, returning empty string on error.
///
/// If serialization fails, logs a warning and returns an empty string.
#[must_use]
pub fn serialize_to_string<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .inspect_err(|e| warn!("failed to serialize value to JSON: {e}"))
        .unwrap_or_default()
}

/// Deserialize JSON string to any type, returning None on error.
///
/// If deserialization fails, logs a warning and returns None.
#[must_use]
pub fn deserialize_from_str_opt<T: serde::de::DeserializeOwned>(json_str: &str) -> Option<T> {
    serde_json::from_str(json_str)
        .inspect_err(|e| warn!("failed to deserialize JSON string: {e}"))
        .ok()
}

/// Deserialize optional JSON String to any type, returning default on error or if None.
///
/// If deserialization fails, logs a warning and returns the default value for T.
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
///
/// If deserialization fails, logs a warning and returns None.
#[must_use]
pub fn deserialize_from_string_opt<T: serde::de::DeserializeOwned>(json_str: &Option<String>) -> Option<T> {
    json_str.as_ref().and_then(|s| deserialize_from_str_opt::<T>(s))
}
