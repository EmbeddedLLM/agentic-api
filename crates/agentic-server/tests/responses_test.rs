mod common;

use axum::Router;
use axum::body::Bytes;
use axum::response::IntoResponse;
use axum::routing::post;
use http::StatusCode;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use common::{spawn_gateway, spawn_mock_llm, test_config, test_state};

/// Spawn a mock vLLM that returns a minimal valid JSON response.
async fn spawn_mock_vllm_json() -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            axum::response::Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"id":"mock_id","object":"response","status":"completed",
                        "model":"test","output":[],"created_at":0}"#,
                ))
                .unwrap()
                .into_response()
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), handle)
}

async fn spawn_mock_vllm_json_capture() -> (String, Arc<Mutex<Vec<serde_json::Value>>>, tokio::task::JoinHandle<()>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_requests = Arc::clone(&requests);
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let route_requests = Arc::clone(&route_requests);
            async move {
                let body = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
                route_requests.lock().await.push(body);
                axum::response::Response::builder()
                    .status(200)
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(
                        r#"{"id":"mock_id","object":"response","status":"completed",
                            "model":"test","output":[],"created_at":0}"#,
                    ))
                    .unwrap()
                    .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), requests, handle)
}

async fn spawn_mock_vllm_namespace_response()
-> (String, Arc<Mutex<Vec<serde_json::Value>>>, tokio::task::JoinHandle<()>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_requests = Arc::clone(&requests);
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let route_requests = Arc::clone(&route_requests);
            async move {
                let body = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
                route_requests.lock().await.push(body);
                axum::response::Response::builder()
                    .status(200)
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(
                        r#"{"id":"mock_id","object":"response","status":"completed","model":"test","tools":[{"type":"function","name":"agentic_ns__mcp__shell__run"}],"output":[{"type":"function_call","call_id":"call_1","name":"agentic_ns__mcp__shell__run","arguments":"{}"}],"created_at":0}"#,
                    ))
                    .unwrap()
                    .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), requests, handle)
}

/// Spawn a mock vLLM that returns an SSE stream.
async fn spawn_mock_vllm_sse() -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            axum::response::Response::builder()
                .status(200)
                .header("Content-Type", "text/event-stream; charset=utf-8")
                .body(axum::body::Body::from(
                    "data: {\"type\":\"response.done\"}\n\ndata: [DONE]\n\n",
                ))
                .unwrap()
                .into_response()
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), handle)
}

#[tokio::test]
async fn test_store_false_proxies_json_to_vllm() {
    // Arrange
    let (llm_url, _h1) = spawn_mock_vllm_json().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({"model":"test","input":[{"type":"message","role":"user","content":"hi"}],"store":false,"stream":false}))
        .send()
        .await
        .unwrap();

    // Assert — proxy forwards vLLM response verbatim; mock_id is not resp_-prefixed
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["id"], "mock_id");
}

#[tokio::test]
async fn test_store_false_proxy_flattens_and_restores_namespace_tools() {
    let (llm_url, requests, _h1) = spawn_mock_vllm_namespace_response().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": "run pwd",
            "tools": [{
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}}]
            }],
            "tool_choice": {"type": "function", "namespace": "mcp__shell", "name": "run"},
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["tools"][0]["type"], "namespace");
    assert_eq!(body["tools"][0]["name"], "mcp__shell");
    assert_eq!(body["output"][0]["namespace"], "mcp__shell");
    assert_eq!(body["output"][0]["name"], "run");

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["tools"][0]["type"], "function");
    assert_eq!(requests[0]["tools"][0]["name"], "agentic_ns__mcp__shell__run");
    assert_eq!(requests[0]["tool_choice"]["name"], "agentic_ns__mcp__shell__run");
    assert!(requests[0]["tool_choice"].get("namespace").is_none());
}

#[tokio::test]
async fn test_store_false_proxy_rejects_namespace_flat_name_collisions() {
    let (llm_url, requests, _h1) = spawn_mock_vllm_json_capture().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let collision_payloads = [
        serde_json::json!({
            "model": "test",
            "input": "hi",
            "store": false,
            "tools": [
                {"type": "function", "name": "agentic_ns__mcp__shell__run"},
                {
                    "type": "namespace",
                    "name": "mcp__shell",
                    "tools": [{"type": "function", "name": "run"}]
                }
            ]
        }),
        serde_json::json!({
            "model": "test",
            "input": "hi",
            "store": false,
            "tools": [
                {
                    "type": "namespace",
                    "name": "a__b",
                    "tools": [{"type": "function", "name": "c"}]
                },
                {
                    "type": "namespace",
                    "name": "a",
                    "tools": [{"type": "function", "name": "b__c"}]
                }
            ]
        }),
    ];

    for payload in collision_payloads {
        let response = reqwest::Client::new()
            .post(format!("{gw_url}/v1/responses"))
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "invalid_request_error");
    }

    assert!(
        requests.lock().await.is_empty(),
        "invalid requests must not reach upstream"
    );
}

#[tokio::test]
async fn test_store_false_with_web_search_reaches_executor() {
    // Arrange
    let (llm_url, requests, _h1) = spawn_mock_vllm_json_capture().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_preview"}],
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Assert — gateway tools need executor normalization even when persistence is disabled.
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["id"].as_str().unwrap_or("").starts_with("resp_"));
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["tools"][0]["type"], "function");
    assert_eq!(requests[0]["tools"][0]["name"], "web_search");
}

#[tokio::test]
async fn test_store_false_proxies_large_json_body_to_vllm() {
    // Arrange
    let (llm_url, requests, _h1) = spawn_mock_vllm_json_capture().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    let prompt = "x".repeat(100 * 1024);

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": prompt}],
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Assert — the gateway keeps this below-limit request on the proxy path.
    assert_eq!(resp.status(), 200);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["store"], false);
    assert_eq!(requests[0]["stream"], false);
    assert_eq!(requests[0]["input"][0]["content"].as_str().unwrap().len(), 100 * 1024);
}

#[tokio::test]
async fn test_store_false_proxies_sse_to_vllm() {
    // Arrange
    let (llm_url, _h1) = spawn_mock_vllm_sse().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({"model":"test","input":[{"type":"message","role":"user","content":"hi"}],"store":false,"stream":true}))
        .send()
        .await
        .unwrap();

    // Assert — SSE content-type forwarded from mock vLLM
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()["content-type"]
            .to_str()
            .unwrap()
            .contains("event-stream")
    );
}

#[tokio::test]
async fn test_store_true_reaches_executor_not_proxy() {
    // Arrange — mock vLLM returns 200, but executor path will fail at storage layer
    let (llm_url, _h1) = spawn_mock_vllm_json().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({"model":"test","input":[{"type":"message","role":"user","content":"hi"}],"store":true,"stream":false}))
        .send()
        .await
        .unwrap();

    // Assert — executor path reached: executor assigns a resp_-prefixed id
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let id = body["id"].as_str().unwrap_or("");
    assert!(
        id.starts_with("resp_"),
        "expected executor-assigned id starting with resp_, got: {id}"
    );
}

#[tokio::test]
async fn test_oversized_body_returns_413() {
    // Arrange — LLM is never reached (gateway rejects the body first)
    let (llm_url, _h1) = spawn_mock_llm().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act — 11 MB body
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .header("Content-Type", "application/json")
        .body("x".repeat(11 * 1024 * 1024))
        .send()
        .await
        .unwrap();

    // Assert
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
