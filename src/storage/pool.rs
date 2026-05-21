//! Database connection pooling and initialization.

use std::sync::Arc;

use sqlx::any::AnyPoolOptions;

/// Generic database pool type supporting `SQLite`, `PostgreSQL`, and `MySQL`.
pub type DbPool = sqlx::Pool<sqlx::Any>;

/// Database transaction type for multi-statement operations.
pub type DbTransaction<'a> = sqlx::Transaction<'a, sqlx::Any>;

/// Convenience type alias for database operation results.
///
/// All database queries return `DbResult<T>` which is `Result<T, sqlx::Error>`.
pub type DbResult<T> = Result<T, sqlx::Error>;

/// Prepares database URL with appropriate parameters.
///
/// For `SQLite` connections, adds `?mode=rwc` if not already present.
/// This enables write mode (`rwc` = read-write-create) for file-based databases.
///
/// For other database types (`PostgreSQL`, `MySQL`), returns URL as-is.
fn prepare_db_url(url: &str) -> String {
    if url.starts_with("sqlite") && !url.contains('?') {
        format!("{url}?mode=rwc")
    } else {
        url.to_string()
    }
}

/// Creates a connection pool for the database.
///
/// Initializes a connection pool with sensible defaults:
/// - Max connections: 10 (configurable via [`AnyPoolOptions`])
/// - Driver auto-detection: supports `SQLite`, `PostgreSQL`, `MySQL`
/// - `SQLite` file mode: read-write-create for file-based databases
///
/// The pool is wrapped in `Arc` for thread-safe sharing across async tasks.
/// See [Rust Cookbook § Database](https://rust-lang-nursery.github.io/rust-cookbook/database.html)
/// for pooling best practices.
///
/// # Arguments
///
/// * `db_url` - Database connection URL (e.g., `sqlite://data.db`, `postgresql://user:pass@host/db`)
///
/// # Errors
///
/// Returns [`sqlx::Error`] if:
/// - Connection URL is invalid
/// - Database server is unreachable
/// - Connection limit is exceeded
/// - Authentication fails
///
/// # Examples
///
/// ```ignore
/// use agentic_api::storage::pool;
///
/// // SQLite (file-based)
/// let pool = pool::create_pool("sqlite://data.db").await?;
///
/// // PostgreSQL
/// let pool = pool::create_pool("postgresql://user:pass@localhost/mydb").await?;
///
/// // Use the pool (shared via Arc)
/// let result = sqlx::query("SELECT * FROM responses")
///     .fetch_one(pool.as_ref())
///     .await?;
/// ```
///
/// # Performance Considerations (From Rust Cookbook)
///
/// - Connection pooling reduces overhead of establishing new connections
/// - Connections are reused from the pool for subsequent queries
/// - Maximum connections should be tuned to database and application capacity
/// - No blocking I/O on connection retrieval - uses async/await
pub async fn create_pool(db_url: &str) -> DbResult<Arc<DbPool>> {
    // Install default drivers for auto-detection
    sqlx::any::install_default_drivers();

    // Prepare URL with database-specific parameters
    let url = prepare_db_url(db_url);

    // Create connection pool with 10 max connections
    // This is a conservative default - tune based on your workload
    let pool = AnyPoolOptions::new().max_connections(10).connect(&url).await?;

    // Wrap in Arc for thread-safe sharing across async tasks
    Ok(Arc::new(pool))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_sqlite_url_without_params() {
        let url = "sqlite://test.db";
        let prepared = prepare_db_url(url);
        assert_eq!(prepared, "sqlite://test.db?mode=rwc");
    }

    #[test]
    fn test_prepare_sqlite_url_with_params() {
        let url = "sqlite://test.db?cache=shared";
        let prepared = prepare_db_url(url);
        assert_eq!(prepared, "sqlite://test.db?cache=shared");
    }

    #[test]
    fn test_prepare_postgres_url() {
        let url = "postgresql://user:pass@localhost/db";
        let prepared = prepare_db_url(url);
        assert_eq!(prepared, "postgresql://user:pass@localhost/db");
    }

    #[test]
    fn test_prepare_mysql_url() {
        let url = "mysql://user:pass@localhost/db";
        let prepared = prepare_db_url(url);
        assert_eq!(prepared, "mysql://user:pass@localhost/db");
    }
}
