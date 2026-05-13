// Submodules (config, proxy, store) are declared directly by each test file
// via `#[path]` to avoid dead_code warnings from cross-binary compilation.
// create_test_pool is here because it is shared by both proxy and store tests.

use std::sync::Arc;

use agentic_api::database::db::DbPool;
use agentic_api::database::schema::SchemaManager;

pub async fn create_test_pool() -> Arc<DbPool> {
    sqlx::any::install_default_drivers();
    let pool = sqlx::any::AnyPoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .expect("failed to create test pool");
    let pool = Arc::new(pool);
    SchemaManager::new_for_test(&pool)
        .ensure_ready()
        .await
        .expect("failed to run test schema");
    pool
}
