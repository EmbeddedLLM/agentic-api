//! Database URL classification and sanitization.

use std::time::Duration;

/// Database backend selected by a connection URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatabaseBackend {
    /// `PostgreSQL`, using either the `postgres` or `postgresql` URI scheme.
    Postgres,
    /// `SQLite`.
    Sqlite,
    /// Another configured backend.
    Other,
}

impl DatabaseBackend {
    /// Parses a database URL and classifies its normalized URI scheme.
    ///
    /// # Errors
    ///
    /// Returns [`url::ParseError`] when `database_url` is not a valid absolute URL.
    pub fn from_url(database_url: &str) -> Result<Self, url::ParseError> {
        let url = url::Url::parse(database_url)?;
        Ok(match url.scheme() {
            "postgres" | "postgresql" => Self::Postgres,
            "sqlite" => Self::Sqlite,
            _ => Self::Other,
        })
    }

    /// Returns the backend name used in diagnostics.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Postgres => "PostgreSQL",
            Self::Sqlite => "SQLite",
            Self::Other => "configured",
        }
    }

    pub(crate) fn from_connection(connection: &sqlx::AnyConnection) -> Self {
        match connection.backend_name() {
            "PostgreSQL" => Self::Postgres,
            "SQLite" => Self::Sqlite,
            _ => Self::Other,
        }
    }
}

pub(crate) async fn configure_postgres_timeouts(
    connection: &mut sqlx::AnyConnection,
    lock_timeout: Duration,
    statement_timeout: Duration,
) -> Result<(), sqlx::Error> {
    let lock_timeout_ms = format!("{}ms", lock_timeout.as_millis());
    sqlx::query("SELECT set_config('lock_timeout', $1, false)")
        .bind(lock_timeout_ms)
        .execute(&mut *connection)
        .await?;
    let statement_timeout_ms = format!("{}ms", statement_timeout.as_millis());
    sqlx::query("SELECT set_config('statement_timeout', $1, false)")
        .bind(statement_timeout_ms)
        .execute(connection)
        .await?;
    Ok(())
}

pub(crate) fn redact_database_urls(message: &str) -> String {
    const DATABASE_SCHEMES: [&str; 3] = ["postgresql://", "postgres://", "mysql://"];

    let mut redacted = message.to_owned();
    let mut lowercase = message.to_ascii_lowercase();
    for scheme in DATABASE_SCHEMES {
        let mut search_from = 0;
        let replacement = format!("{scheme}[redacted]");
        while let Some(offset) = lowercase[search_from..].find(scheme) {
            let start = search_from + offset;
            let end = redacted[start..]
                .find(char::is_whitespace)
                .map_or(redacted.len(), |length| start + length);
            redacted.replace_range(start..end, &replacement);
            lowercase.replace_range(start..end, &replacement);
            search_from = start + replacement.len();
        }
    }
    redacted
}
