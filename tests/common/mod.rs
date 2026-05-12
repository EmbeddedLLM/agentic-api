pub mod fixtures;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use bytes::Bytes;
use futures::stream;
use http::StatusCode;
use tokio::net::TcpListener;

use agentic_api::config::RuntimeConfig;
use agentic_api::entrypoints::proxy::ProxyState;

use agentic_api::database::db::DbPool;
use agentic_api::database::schema::SchemaManager;

pub fn test_config(upstream_url: &str) -> RuntimeConfig {
    RuntimeConfig {
        llm_api_base: upstream_url.to_owned(),
        openai_api_key: Some("env-upstream-key".to_owned()),
        gateway_host: "127.0.0.1".to_owned(),
        gateway_port: 0,
        upstream_ready_timeout_s: 5.0,
        upstream_ready_interval_s: 0.1,
        db_url: "sqlite::memory:".to_owned(),
        db_dialect: agentic_api::config::DbDialect::Sqlite,
        gateway_workers: 1,
        response_store_enabled: true,
        log_model_messages: false,
    }
}

pub fn test_config_no_key(upstream_url: &str) -> RuntimeConfig {
    RuntimeConfig {
        openai_api_key: None,
        ..test_config(upstream_url)
    }
}

pub async fn create_test_pool() -> &'static DbPool {
    sqlx::any::install_default_drivers();
    let pool = sqlx::any::AnyPoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .expect("failed to create test pool");
    SchemaManager::new(&pool)
        .ensure_ready()
        .await
        .expect("failed to run test schema");
    Box::leak(Box::new(pool))
}

async fn health_handler() -> impl IntoResponse {
    StatusCode::OK
}

async fn responses_handler(req: Request) -> Response {
    let headers = req.headers().clone();
    let body_bytes = axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024)
        .await
        .unwrap_or_default();

    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();

    if body
        .get("echo_auth")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        let auth = headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("");
        let resp_body = serde_json::json!({"authorization": auth});
        return (
            StatusCode::OK,
            [("content-type", "application/json"), ("x-upstream", "responses")],
            serde_json::to_string(&resp_body).unwrap(),
        )
            .into_response();
    }

    if body.get("force_error").and_then(serde_json::Value::as_u64) == Some(429) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [("content-type", "application/json"), ("x-upstream", "error")],
            r#"{"error":{"message":"rate limited","code":"rate_limit"}}"#,
        )
            .into_response();
    }

    if body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false) {
        let chunks: Vec<Result<Bytes, Infallible>> = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
            )),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ];
        let body = Body::from_stream(stream::iter(chunks));
        return (
            StatusCode::OK,
            [
                ("content-type", "text/event-stream; charset=utf-8"),
                ("x-upstream", "responses-stream"),
            ],
            body,
        )
            .into_response();
    }

    let out = r#"{"id":"resp_test","object":"response","status":"completed"}"#;
    (
        StatusCode::OK,
        [
            ("content-type", "application/json"),
            ("x-upstream", "responses"),
            ("connection", "keep-alive"),
        ],
        out,
    )
        .into_response()
}

pub async fn spawn_upstream() -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/responses", post(responses_handler));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (format!("http://{addr}"), handle)
}

pub async fn spawn_gateway(config: RuntimeConfig) -> (String, SocketAddr, tokio::task::JoinHandle<()>) {
    let state = ProxyState::new(config);
    let router = agentic_api::entrypoints::app::build_router(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (format!("http://{addr}"), addr, handle)
}

pub async fn spawn_mid_stream_failure_upstream() -> (String, tokio::task::JoinHandle<()>) {
    async fn handler(_req: Request) -> Response {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(2);
        tokio::spawn(async move {
            let _ = tx
                .send(Ok(Bytes::from(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
                )))
                .await;
            tokio::time::sleep(Duration::from_millis(10)).await;
            drop(tx);
        });
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let body = Body::from_stream(stream);
        (
            StatusCode::OK,
            [
                ("content-type", "text/event-stream; charset=utf-8"),
                ("x-upstream", "fake-stream"),
            ],
            body,
        )
            .into_response()
    }

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/responses", post(handler));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (format!("http://{addr}"), handle)
}

pub async fn spawn_timeout_upstream() -> (String, tokio::task::JoinHandle<()>) {
    async fn handler(_req: Request) -> Response {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        StatusCode::OK.into_response()
    }

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/responses", post(handler));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (format!("http://{addr}"), handle)
}

pub fn proxy_state_with_short_timeout(config: RuntimeConfig) -> ProxyState {
    let stream_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(100))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();

    let non_stream_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(100))
        .read_timeout(Duration::from_millis(100))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();

    ProxyState {
        config,
        stream_client,
        non_stream_client,
    }
}
