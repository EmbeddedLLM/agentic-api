use std::env;
use std::sync::OnceLock;

use tracing::{debug, info};

use super::db::{DbPool, DbResult};

static SCHEMA_READY: OnceLock<()> = OnceLock::new();

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
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    pub async fn ensure_ready(&self) -> DbResult<()> {
        if SCHEMA_READY.get().is_some() {
            return Ok(());
        }

        if is_marked_ready() {
            debug!("[schema] DDL skipped — marked ready by supervisor.");
            SCHEMA_READY.get_or_init(|| ());
            return Ok(());
        }

        debug!("[schema] Running migrations...");
        sqlx::migrate!("./migrations")
            .run(self.pool)
            .await
            .map_err(|e| sqlx::Error::Configuration(e.to_string().into()))?;

        SCHEMA_READY.get_or_init(|| ());
        info!("[schema] DB schema ready.");
        Ok(())
    }
}
