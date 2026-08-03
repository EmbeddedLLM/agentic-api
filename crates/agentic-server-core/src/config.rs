use std::time::Duration;

pub const DEFAULT_POSTGRES_MAX_CONNECTIONS: u32 = 10;
pub const DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS: u64 = 30;
pub const DEFAULT_POSTGRES_IDLE_TIMEOUT_SECONDS: u64 = 600;
pub const DEFAULT_POSTGRES_LOCK_TIMEOUT_SECONDS: u64 = 5;
pub const DEFAULT_POSTGRES_MAX_LIFETIME_SECONDS: u64 = 1_800;
pub const DEFAULT_POSTGRES_MIGRATION_TIMEOUT_SECONDS: u64 = 300;
pub const DEFAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS: u64 = 30;
pub const DEFAULT_SQLITE_MAX_CONNECTIONS: u32 = 4;
pub const DEFAULT_SQLITE_JOURNAL_SIZE_LIMIT_BYTES: u64 = 6_144_000;
pub const DEFAULT_SQLITE_MMAP_SIZE_BYTES: u64 = 268_435_456;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostgresConfig {
    pub max_connections: u32,
    pub acquire_timeout: Duration,
    pub lock_timeout: Duration,
    pub migration_timeout: Duration,
    pub statement_timeout: Duration,
    pub idle_timeout: Option<Duration>,
    pub max_lifetime: Option<Duration>,
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            max_connections: DEFAULT_POSTGRES_MAX_CONNECTIONS,
            acquire_timeout: Duration::from_secs(DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS),
            lock_timeout: Duration::from_secs(DEFAULT_POSTGRES_LOCK_TIMEOUT_SECONDS),
            migration_timeout: Duration::from_secs(DEFAULT_POSTGRES_MIGRATION_TIMEOUT_SECONDS),
            statement_timeout: Duration::from_secs(DEFAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS),
            idle_timeout: Some(Duration::from_secs(DEFAULT_POSTGRES_IDLE_TIMEOUT_SECONDS)),
            max_lifetime: Some(Duration::from_secs(DEFAULT_POSTGRES_MAX_LIFETIME_SECONDS)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SqliteTempStore {
    Default,
    File,
    #[default]
    Memory,
}

impl SqliteTempStore {
    #[must_use]
    pub fn as_pragma_value(self) -> &'static str {
        match self {
            Self::Default => "DEFAULT",
            Self::File => "FILE",
            Self::Memory => "MEMORY",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqliteConfig {
    pub max_connections: u32,
    pub journal_size_limit_bytes: u64,
    pub temp_store: SqliteTempStore,
    pub mmap_size_bytes: u64,
}

impl Default for SqliteConfig {
    fn default() -> Self {
        Self {
            max_connections: DEFAULT_SQLITE_MAX_CONNECTIONS,
            journal_size_limit_bytes: DEFAULT_SQLITE_JOURNAL_SIZE_LIMIT_BYTES,
            temp_store: SqliteTempStore::default(),
            mmap_size_bytes: DEFAULT_SQLITE_MMAP_SIZE_BYTES,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub llm_api_base: String,
    pub openai_api_key: Option<String>,
    pub llm_ready_timeout_s: f64,
    pub llm_ready_interval_s: f64,
    pub skip_llm_ready_check: bool,
    /// Database URL for conversation and response storage.
    /// `None` means stateful features are disabled; all requests are proxied.
    pub db_url: Option<String>,
    pub postgres: PostgresConfig,
    pub sqlite: SqliteConfig,
}

#[must_use]
pub fn normalize_base_url(url: &str) -> String {
    let mut s = url.trim_end_matches('/').to_owned();
    if s.ends_with("/v1") {
        s.truncate(s.len() - 3);
        s = s.trim_end_matches('/').to_owned();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_trailing_v1() {
        assert_eq!(normalize_base_url("http://host:8000/v1"), "http://host:8000");
        assert_eq!(normalize_base_url("http://host:8000/v1/"), "http://host:8000");
    }

    #[test]
    fn no_v1_unchanged() {
        assert_eq!(normalize_base_url("http://host:8000"), "http://host:8000");
        assert_eq!(normalize_base_url("http://host:8000/"), "http://host:8000");
    }
}
