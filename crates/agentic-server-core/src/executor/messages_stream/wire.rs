//! Messages SSE serialization and error forwarding.

use crate::executor::error::ExecutorError;
use crate::utils::common::{deserialize_from_str, serialize_to_string};
use serde_json::{Value, json};

pub(super) fn sse(event: &str, value: &Value) -> String {
    let json = serialize_to_string(value).unwrap_or_default();
    format!("event: {event}\ndata: {json}\n\n")
}

pub(super) fn error_sse(message: &str) -> String {
    let event = json!({"type": "error", "error": {"type": "api_error", "message": message}});
    let json = serialize_to_string(&event).unwrap_or_default();
    format!("event: error\ndata: {json}\n\n")
}

pub(super) fn executor_error_sse(error: &ExecutorError) -> String {
    if let ExecutorError::LLMRequest { body, .. } = error
        && let Ok(value) = deserialize_from_str::<Value>(body)
        && value.get("type").and_then(Value::as_str) == Some("error")
    {
        let data = if body.contains(['\r', '\n']) {
            serialize_to_string(&value).unwrap_or_else(|_| body.clone())
        } else {
            body.clone()
        };
        return format!("event: error\ndata: {data}\n\n");
    }
    error_sse(&error.to_string())
}
