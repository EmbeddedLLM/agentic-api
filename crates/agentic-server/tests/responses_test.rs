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
async fn test_store_false_tool_search_history_without_tools_reaches_executor() {
    let (llm_url, requests, _h1) = spawn_mock_vllm_json_capture().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": [
                {
                    "type": "tool_search_call",
                    "execution": "client",
                    "call_id": "call_search",
                    "status": "completed",
                    "arguments": {"query": "echo"}
                },
                {
                    "type": "tool_search_output",
                    "execution": "client",
                    "call_id": "call_search",
                    "status": "completed",
                    "tools": [{
                        "type": "function",
                        "name": "echo",
                        "defer_loading": true,
                        "parameters": {"type": "object"}
                    }]
                }
            ],
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["id"].as_str().unwrap_or("").starts_with("resp_"));

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["input"][0]["type"], "tool_search_call");
    assert_eq!(requests[0]["input"][1]["type"], "tool_search_output");
    assert_eq!(requests[0]["tools"][0]["type"], "function");
    assert_eq!(requests[0]["tools"][0]["name"], "echo");
    assert!(requests[0]["tools"][0].get("defer_loading").is_none());
    assert!(!requests[0].to_string().contains("_agentic_item_kind"));
}

fn tool_search_history(
    execution: Option<&str>,
    status: Option<&str>,
    call_id: &str,
    output_call_id: &str,
    tools: &serde_json::Value,
) -> serde_json::Value {
    let mut call = serde_json::json!({
        "type": "tool_search_call",
        "call_id": call_id,
        "arguments": {"query": "echo"}
    });
    let mut output = serde_json::json!({
        "type": "tool_search_output",
        "call_id": output_call_id,
        "tools": tools
    });
    for item in [&mut call, &mut output] {
        if let Some(execution) = execution {
            item["execution"] = serde_json::Value::String(execution.to_owned());
        }
        if let Some(status) = status {
            item["status"] = serde_json::Value::String(status.to_owned());
        }
    }
    serde_json::json!([call, output])
}

fn nonpromotable_tool_search_histories() -> [(&'static str, serde_json::Value); 7] {
    let function = || serde_json::json!([{"type": "function", "name": "echo"}]);
    [
        (
            "incomplete",
            tool_search_history(
                Some("client"),
                Some("incomplete"),
                "call_search",
                "call_search",
                &function(),
            ),
        ),
        (
            "optional fields omitted",
            tool_search_history(None, None, "call_search", "call_search", &function()),
        ),
        (
            "server execution",
            tool_search_history(
                Some("server"),
                Some("completed"),
                "call_search",
                "call_search",
                &function(),
            ),
        ),
        (
            "unmatched output",
            tool_search_history(Some("client"), Some("completed"), "call_one", "call_two", &function()),
        ),
        (
            "output only",
            serde_json::json!([{
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "tools": function()
            }]),
        ),
        (
            "no loaded functions",
            tool_search_history(
                Some("client"),
                Some("completed"),
                "call_search",
                "call_search",
                &serde_json::json!([]),
            ),
        ),
        (
            "ambiguous namespace members",
            tool_search_history(
                Some("client"),
                Some("completed"),
                "call_search",
                "call_search",
                &serde_json::json!([
                    {"type": "namespace", "name": "one", "tools": function()},
                    {"type": "namespace", "name": "two", "tools": function()}
                ]),
            ),
        ),
    ]
}

#[tokio::test]
async fn test_store_false_nonpromotable_tool_search_history_stays_on_transparent_proxy() {
    let (llm_url, requests, _h1) = spawn_mock_vllm_json_capture().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    let cases = nonpromotable_tool_search_histories();
    let client = reqwest::Client::new();
    let mut expected_requests = Vec::with_capacity(cases.len());

    for (label, input) in cases {
        let payload = serde_json::json!({
            "model": "test",
            "input": input,
            "store": false,
            "stream": false
        });
        let resp = client
            .post(format!("{gw_url}/v1/responses"))
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), 200, "{label}");
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["id"], "mock_id", "{label} should stay on proxy path");
        expected_requests.push(payload);
    }

    let requests = requests.lock().await;
    assert_eq!(*requests, expected_requests, "proxy must preserve each wire payload");
}

#[tokio::test]
async fn test_gateway_normalization_preserves_parallel_tool_calls() {
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
            "parallel_tool_calls": false,
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Assert
    assert_eq!(resp.status(), 200);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["parallel_tool_calls"], false);
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
