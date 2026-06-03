use http::StatusCode;
use thiserror::Error;

use crate::StorageError;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ExecutorError {
    /// A storage layer operation failed.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// The LLM backend returned a non-2xx status or was unreachable.
    #[error("LLM request failed ({status}): {body}")]
    LLMRequest { status: StatusCode, body: String },

    /// A network error occurred reading from the LLM response stream.
    ///
    /// The original `reqwest::Error` is preserved as the error source so
    /// callers can inspect the underlying network failure.
    #[error("network error: {0}")]
    NetworkError(
        #[from]
        #[source]
        reqwest::Error,
    ),

    /// JSON deserialisation failed.
    ///
    /// The original `serde_json::Error` is preserved as the error source so
    /// callers can inspect the exact parse failure location and kind.
    #[error("json error: {0}")]
    JsonError(
        #[from]
        #[source]
        serde_json::Error,
    ),

    /// A general stream processing error with a human-readable message.
    ///
    /// Used for non-network stream failures (e.g. worker thread panic).
    #[error("stream error: {0}")]
    StreamError(String),

    /// A validation error on the request payload with a human-readable message.
    ///
    /// Used when required fields are missing or structurally invalid.
    #[error("parse error: {0}")]
    ParseError(String),

    #[error("{entity} not found: {id}")]
    NotFound { entity: String, id: String },

    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

pub type ExecutorResult<T> = Result<T, ExecutorError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_error_display() {
        let err = ExecutorError::InvalidRequest("test message".into());
        assert!(err.to_string().contains("invalid request"));
        assert!(err.to_string().contains("test message"));
    }

    #[test]
    fn test_executor_error_stream() {
        let err = ExecutorError::StreamError("connection lost".into());
        assert!(err.to_string().contains("stream error"));
    }

    #[test]
    fn test_executor_error_not_found() {
        let err = ExecutorError::NotFound {
            entity: "Conversation".into(),
            id: "conv_123".into(),
        };
        assert!(err.to_string().contains("Conversation"));
        assert!(err.to_string().contains("conv_123"));
    }

    #[test]
    fn test_executor_error_from_storage() {
        let storage_err = StorageError::NotConfigured;
        let exec_err = ExecutorError::from(storage_err);
        assert!(exec_err.to_string().contains("storage error"));
    }

    #[test]
    fn test_executor_error_json_preserves_source() {
        use std::error::Error;
        let json_err: serde_json::Error = serde_json::from_str::<serde_json::Value>("{bad}").unwrap_err();
        let exec_err = ExecutorError::from(json_err);
        assert!(exec_err.source().is_some(), "source should be chained");
        assert!(exec_err.to_string().contains("json error"));
    }
}
