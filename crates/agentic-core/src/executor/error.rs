use http::StatusCode;
use thiserror::Error;

use crate::StorageError;

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("LLM request failed ({status}): {body}")]
    LLMRequest { status: StatusCode, body: String },

    #[error("stream error: {0}")]
    StreamError(String),

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
}
