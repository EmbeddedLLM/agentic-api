use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
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
use agentic_api::core::agent::Agent;
use agentic_api::database::db::DbPool;
use agentic_api::entrypoints::app::AppState;
use agentic_api::entrypoints::proxy::ProxyState;
use agentic_api::store::conversation::ConversationStore;
use agentic_api::store::response::ResponseStore;

async fn health_handler() -> impl IntoResponse {
    StatusCode::OK
}

async fn responses_handler(req: Request) -> Response {
    let headers = req.headers().clone();
    let body_bytes = axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024)
        .await
        .unwrap_or_default();

    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();

    if body.get("force_error").and_then(serde_json::Value::as_u64) == Some(429) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [("content-type", "application/json"), ("x-upstream", "error")],
            r#"{"error":{"message":"rate limited","code":"rate_limit"}}"#,
        )
            .into_response();
    }

    let is_stream = body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);
    let x_upstream = if is_stream { "responses-stream" } else { "responses" };

    let auth_header = headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("");
    let completed_event = serde_json::json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": body.get("model").and_then(|m| m.as_str()).unwrap_or("model-a"),
            "status": "completed",
            "output": [],
            "previous_response_id": null,
            "conversation_id": null,
            "instructions": auth_header  // reuse instructions field to echo auth
        }
    });
    let chunks: Vec<Result<Bytes, Infallible>> = vec![
        Ok(Bytes::from(format!(
            "data: {}\n\n",
            serde_json::to_string(&completed_event).unwrap()
        ))),
        Ok(Bytes::from("data: [DONE]\n\n")),
    ];
    let body = Body::from_stream(stream::iter(chunks));
    (
        StatusCode::OK,
        [
            ("content-type", "text/event-stream; charset=utf-8"),
            ("x-upstream", x_upstream),
        ],
        body,
    )
        .into_response()
}

pub async fn spawn_error_upstream(status: u16) -> (String, tokio::task::JoinHandle<()>) {
    let handler = move |_req: Request| async move {
        (
            StatusCode::from_u16(status).unwrap(),
            [("content-type", "application/json")],
            r#"{"error":{"message":"upstream error","code":"rate_limit"}}"#,
        )
            .into_response()
    };

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

pub async fn spawn_gateway(
    config: RuntimeConfig,
    pool: Arc<DbPool>,
) -> (String, SocketAddr, tokio::task::JoinHandle<()>) {
    let state = AppState {
        config: Arc::new(config.clone()),
        agent: Arc::new(Agent::new(&config)),
        response_store: ResponseStore::new(Arc::clone(&pool)),
        conversation_store: Some(ConversationStore::new(Arc::clone(&pool))),
    };
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
        config: Arc::new(config),
        stream_client,
        non_stream_client,
    }
}
