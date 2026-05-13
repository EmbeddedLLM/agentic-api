use std::env;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{debug, info};

use super::db::{DbPool, DbResult};

static SCHEMA_READY: AtomicBool = AtomicBool::new(false);

fn is_marked_ready() -> bool {
    matches!(
        env::var("AA_DB_SCHEMA_READY").as_deref(),
        Ok("1" | "true" | "t" | "yes" | "y" | "on")
    )
}

pub struct SchemaManager<'a> {
    pool: &'a DbPool,
}

impl<'a> SchemaManager<'a> {
    #[must_use]
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

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
    /// Constructor that resets the schema-ready flag before returning,
    /// so each fresh pool always gets migrations applied.
    /// Intended for tests that create a new in-memory pool per test case.
    #[must_use]
    pub fn new_for_test(pool: &'a DbPool) -> Self {
        SCHEMA_READY.store(false, Ordering::SeqCst);
        Self { pool }
    }
}
