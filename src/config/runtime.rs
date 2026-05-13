use clap::Parser;
use validator::{Validate, ValidationError, ValidationErrors};

// ---------------------------------------------------------------------------
// DbDialect
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DbDialect {
    #[default]
    Sqlite,
    Postgresql,
}

impl DbDialect {
    fn from_url(url: &str) -> Option<Self> {
        if url.starts_with("sqlite") {
            Some(Self::Sqlite)
        } else if url.starts_with("postgresql") || url.starts_with("postgres") {
            Some(Self::Postgresql)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_db_url(url: &str) -> Result<(), ValidationError> {
    if DbDialect::from_url(url).is_none() {
        let mut e = ValidationError::new("unsupported_dialect");
        e.message =
            Some(format!("db_url must start with sqlite://, postgresql://, or postgres://, got: {url:?}").into());
        return Err(e);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// RuntimeConfig — single struct, both CLI-parseable and validated
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Parser, Validate)]
pub struct RuntimeConfig {
    /// Base URL of the upstream vLLM server. Injected programmatically, not a CLI arg.
    #[arg(skip)]
    pub llm_api_base: String,

    #[arg(long, env = "OPENAI_API_KEY")]
    pub openai_api_key: Option<String>,

    #[arg(long, default_value = "0.0.0.0")]
    pub gateway_host: String,

    #[arg(long, default_value_t = 9000)]
    #[validate(range(min = 1, max = 65535))]
    pub gateway_port: u16,

    #[arg(long, default_value_t = 1)]
    #[validate(range(min = 1))]
    pub gateway_workers: usize,

    #[arg(long, default_value_t = 600.0)]
    #[validate(range(min = 0.0))]
    pub upstream_ready_timeout_s: f64,

    #[arg(long, default_value_t = 2.0)]
    #[validate(range(min = 0.0))]
    pub upstream_ready_interval_s: f64,

    #[arg(long, default_value = "sqlite://./agentic_api.db")]
    #[validate(custom(function = "validate_db_url"))]
    pub db_url: String,

    /// Derived from `db_url` at construction time, never set via CLI.
    #[arg(skip)]
    #[validate(skip)]
    pub db_dialect: DbDialect,

    #[arg(long, default_value_t = true)]
    pub response_store_enabled: bool,

    #[arg(long, default_value_t = false)]
    pub log_model_messages: bool,
}

impl RuntimeConfig {
    /// Inject `llm_api_base` into an already-parsed config and validate.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationErrors`] if any field value is invalid.
    pub fn finalize(mut self, llm_api_base: &str) -> Result<Self, ValidationErrors> {
        self.llm_api_base = normalize_base_url(llm_api_base);
        self.db_dialect = DbDialect::from_url(&self.db_url).unwrap_or(DbDialect::Sqlite);
        self.validate()?;
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[must_use]
pub fn normalize_base_url(url: &str) -> String {
    let mut s = url.trim_end_matches('/').to_owned();
    if s.ends_with("/v1") {
        s.truncate(s.len() - 3);
        s = s.trim_end_matches('/').to_owned();
    }
    s
}
