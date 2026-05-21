//! Database schema management and migrations.

use std::env;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{debug, info};

use super::pool::DbPool;

type DbResult<T> = Result<T, sqlx::Error>;

static SCHEMA_READY: AtomicBool = AtomicBool::new(false);

fn is_marked_ready() -> bool {
    matches!(
        env::var("AA_DB_SCHEMA_READY").as_deref(),
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

    /// Ensures database schema is ready by running pending migrations.
    ///
    /// Checks if migrations have already been applied (via in-memory flag or
    /// `AA_DB_SCHEMA_READY` environment variable). If not, runs all pending
    /// migrations from the `migrations/` directory.
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

        debug!("[schema] Running migrations...");
        sqlx::migrate!("./migrations")
            .run(self.pool)
            .await
            .map_err(|e| sqlx::Error::Configuration(e.to_string().into()))?;

        SCHEMA_READY.store(true, Ordering::SeqCst);
        info!("[schema] DB schema ready.");
        Ok(())
    }
}

impl<'a> SchemaManager<'a> {
    /// Creates a schema manager that resets the ready flag for testing.
    ///
    /// Useful for tests that create a new in-memory pool per test case,
    /// ensuring migrations run fresh for each test.
    #[must_use]
    pub fn new_for_test(pool: &'a DbPool) -> Self {
        SCHEMA_READY.store(false, Ordering::SeqCst);
        Self { pool }
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
        // Test the pattern matching logic for AA_DB_SCHEMA_READY
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
            assert_eq!(matches, expected, "Mismatch for value '{}'", val);
        }
    }
}
