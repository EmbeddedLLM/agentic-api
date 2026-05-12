use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgenticApiError {
    #[error("{0}")]
    BadInput(String),

    #[error("{message}")]
    ResponsesApi {
        message: String,
        status_code: u16,
        #[allow(dead_code)]
        error_type: String,
        param: Option<String>,
        code: Option<String>,
    },

    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
}

impl AgenticApiError {
    pub fn bad_input(msg: impl Into<String>) -> Self {
        Self::BadInput(msg.into())
    }

    pub fn responses_api(
        message: impl Into<String>,
        status_code: u16,
        error_type: impl Into<String>,
        param: Option<String>,
        code: Option<String>,
    ) -> Self {
        Self::ResponsesApi {
            message: message.into(),
            status_code,
            error_type: error_type.into(),
            param,
            code,
        }
    }

    pub fn status_code(&self) -> u16 {
        match self {
            Self::ResponsesApi { status_code, .. } => *status_code,
            Self::BadInput(_) => 400,
            Self::Database(_) => 500,
        }
    }
}
