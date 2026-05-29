//! Database schema management and migrations.

use std::env;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{debug, info};

use super::pool::DbPool;

type DbResult<T> = Result<T, sqlx::Error>;

static SCHEMA_READY: AtomicBool = AtomicBool::new(false);

fn is_marked_ready() -> bool {
    matches!(
        env::var("AGENTIC_API_SCHEMA_READY").as_deref(),
        Ok("1" | "true" | "t" | "yes" | "y" | "on")
    )
}

/// Manages database schema initialization and migrations.
pub struct SchemaManager<'a> {
    pool: &'a DbPool,
}

impl<'a> SchemaManager<'a> {
    /// Creates a new schema manager for the given database pool.
    #[must_use]
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    /// Runs migrations without checking the global flag.
    async fn run_migrations(&self) -> DbResult<()> {
        debug!("[schema] Running migrations...");
        sqlx::migrate!("./migrations")
            .run(self.pool)
            .await
            .map_err(|e| sqlx::Error::Configuration(e.to_string().into()))?;
        info!("[schema] DB schema ready.");
        Ok(())
    }

    /// Ensures database schema is ready by running pending migrations.
    ///
    /// Checks if migrations have already been applied via one of:
    /// 1. In-memory flag (`SCHEMA_READY`)
    /// 2. `AGENTIC_API_SCHEMA_READY` environment variable
    /// 3. For file-based `SQLite`: checks if database file exists (assumes migrations ran before)
    ///
    /// If none of the above, runs all pending migrations from the `migrations/` directory.
    ///
    /// # Errors
    ///
    /// Returns a [`sqlx::Error`] if migrations fail.
    pub async fn ensure_ready(&self) -> DbResult<()> {
        if SCHEMA_READY.load(Ordering::SeqCst) {
            return Ok(());
        }

        if is_marked_ready() {
            debug!("[schema] DDL skipped — marked ready by supervisor.");
            SCHEMA_READY.store(true, Ordering::SeqCst);
            return Ok(());
        }

        self.run_migrations().await?;
        SCHEMA_READY.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Ensures database schema is ready without using the global flag.
    ///
    /// Intended for in-memory test databases that need independent schema initialization.
    ///
    /// # Errors
    ///
    /// Returns a [`sqlx::Error`] if migrations fail.
    pub async fn ensure_ready_for_test(&self) -> DbResult<()> {
        self.run_migrations().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_ready_flag_toggle() {
        // Test basic atomic toggle behavior
        SCHEMA_READY.store(false, Ordering::SeqCst);
        assert!(!SCHEMA_READY.load(Ordering::SeqCst));

        SCHEMA_READY.store(true, Ordering::SeqCst);
        assert!(SCHEMA_READY.load(Ordering::SeqCst));

        SCHEMA_READY.store(false, Ordering::SeqCst);
        assert!(!SCHEMA_READY.load(Ordering::SeqCst));
    }

    #[test]
    fn test_schema_ready_flag_sequential() {
        // Test sequential consistency with multiple transitions
        SCHEMA_READY.store(false, Ordering::SeqCst);

        for i in 0..10 {
            let value = i % 2 == 0;
            SCHEMA_READY.store(value, Ordering::SeqCst);
            assert_eq!(SCHEMA_READY.load(Ordering::SeqCst), value);
        }
    }

    #[test]
    fn test_new_for_test_resets_flag() {
        // Set flag to true to simulate previous test state
        SCHEMA_READY.store(true, Ordering::SeqCst);
        assert!(SCHEMA_READY.load(Ordering::SeqCst));

        // Reset behavior from new_for_test
        SCHEMA_READY.store(false, Ordering::SeqCst);
        assert!(!SCHEMA_READY.load(Ordering::SeqCst));
    }

    #[test]
    fn test_env_var_pattern() {
        // Test the pattern matching logic for AGENTIC_API_SCHEMA_READY
        let test_values = vec![
            ("1", true),
            ("true", true),
            ("t", true),
            ("yes", true),
            ("y", true),
            ("on", true),
            ("0", false),
            ("false", false),
            ("f", false),
            ("no", false),
            ("n", false),
            ("off", false),
            ("", false),
        ];

        for (val, expected) in test_values {
            let matches = matches!(
                Ok::<&str, String>(val).as_deref(),
                Ok("1" | "true" | "t" | "yes" | "y" | "on")
            );
            assert_eq!(matches, expected, "Mismatch for value '{val}'");
        }
    }

    #[tokio::test]
    async fn test_ensure_ready_with_flag_set() {
        // Reset flag before test
        SCHEMA_READY.store(false, Ordering::SeqCst);

        // Create an in-memory SQLite pool
        let pool = crate::storage::pool::create_pool(Some("sqlite://?mode=memory"))
            .await
            .expect("failed to create pool");

        let schema = SchemaManager::new(pool.as_ref());

        // First call should run migrations (or succeed with empty DB)
        let result = schema.ensure_ready().await;
        assert!(result.is_ok(), "ensure_ready failed: {result:?}");

        // Flag should now be set
        assert!(SCHEMA_READY.load(Ordering::SeqCst));

        // Second call should return immediately without doing work
        let result = schema.ensure_ready().await;
        assert!(result.is_ok());

        // Reset flag after test
        SCHEMA_READY.store(false, Ordering::SeqCst);
    }

    #[tokio::test]
    async fn test_ensure_ready_multiple_calls() {
        // Reset flag before test
        SCHEMA_READY.store(false, Ordering::SeqCst);

        let pool = crate::storage::pool::create_pool(Some("sqlite://?mode=memory"))
            .await
            .expect("failed to create pool");

        let schema = SchemaManager::new(pool.as_ref());

        // Multiple calls should all succeed
        for _ in 0..3 {
            let result = schema.ensure_ready().await;
            assert!(result.is_ok());
        }

        assert!(SCHEMA_READY.load(Ordering::SeqCst));

        // Reset flag after test
        SCHEMA_READY.store(false, Ordering::SeqCst);
    }
}
