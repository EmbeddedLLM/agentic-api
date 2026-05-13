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
