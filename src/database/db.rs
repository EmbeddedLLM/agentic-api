use sqlx::any::AnyPoolOptions;

pub type DbPool = sqlx::Pool<sqlx::Any>;
pub type DbTransaction<'a> = sqlx::Transaction<'a, sqlx::Any>;
pub type DbResult<T> = Result<T, sqlx::Error>;

fn prepare_db_url(url: &str) -> String {
    if url.starts_with("sqlite") && !url.contains('?') {
        format!("{url}?mode=rwc")
    } else {
        url.to_string()
    }
}

pub async fn create_pool(db_url: &str) -> DbResult<DbPool> {
    sqlx::any::install_default_drivers();
    let url = prepare_db_url(db_url);
    AnyPoolOptions::new().max_connections(10).connect(&url).await
}
