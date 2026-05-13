use chrono::Utc;

#[must_use]
pub fn utcnow_str() -> String {
    Utc::now().to_rfc3339()
}
