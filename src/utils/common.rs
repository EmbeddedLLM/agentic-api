use chrono::Utc;

pub fn utcnow_str() -> String {
    Utc::now().to_rfc3339()
}
