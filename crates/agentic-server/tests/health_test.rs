mod common;

use std::sync::Arc;

use agentic_core::config::Config;
use agentic_core::executor::ExecutionContext;
use common::{spawn_gateway, spawn_mock_llm, test_config, test_state};

fn test_config_no_key(llm_url: &str) -> Config {
    Config {
        openai_api_key: None,
        ..test_config(llm_url)
    }
}

#[tokio::test]
async fn test_health_returns_200() {
    let (llm_url, _h1) = spawn_mock_llm().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    let resp = reqwest::get(format!("{gw_url}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_health_returns_200_even_when_llm_down() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_addr = listener.local_addr().unwrap();
    drop(listener);
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config_no_key(&format!("http://{dead_addr}")))).await;
    let resp = reqwest::get(format!("{gw_url}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_ready_returns_200_when_llm_healthy() {
    let (llm_url, _h1) = spawn_mock_llm().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    let resp = reqwest::get(format!("{gw_url}/ready")).await.unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_ready_returns_503_when_llm_unreachable() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_addr = listener.local_addr().unwrap();
    drop(listener);
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config_no_key(&format!("http://{dead_addr}")))).await;
    let resp = reqwest::get(format!("{gw_url}/ready")).await.unwrap();
    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn test_ready_returns_503_when_database_pool_is_exhausted() {
    let (llm_url, _h1) = spawn_mock_llm().await;
    let mut config = test_config(&llm_url);
    config.db_url = Some("sqlite://?mode=memory".to_owned());
    let exec_ctx = Arc::new(
        ExecutionContext::from_config(&config)
            .await
            .expect("create execution context"),
    );
    let held_connection = exec_ctx
        .storage_pool()
        .expect("configured storage pool")
        .acquire()
        .await
        .expect("hold only database connection");
    let mut state = test_state(&config);
    state.exec_ctx = exec_ctx;
    let (gw_url, _h2) = spawn_gateway(state).await;

    let resp = reqwest::get(format!("{gw_url}/ready")).await.unwrap();

    assert_eq!(resp.status(), 503);
    drop(held_connection);
}
