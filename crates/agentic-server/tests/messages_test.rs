#[allow(dead_code)]
mod common;

use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::OriginalUri;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use http::{HeaderMap, StatusCode};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use common::{spawn_gateway, test_config, test_state};

#[derive(Clone, Debug)]
struct RecordedRequest {
    uri: String,
    headers: HeaderMap,
    body: Bytes,
}

async fn spawn_recording_upstream(
    status: StatusCode,
    content_type: &'static str,
    response_body: &'static str,
) -> (String, Arc<Mutex<Vec<RecordedRequest>>>, tokio::task::JoinHandle<()>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_requests = Arc::clone(&requests);
    let count_tokens_requests = Arc::clone(&requests);
    let app = Router::new()
        .route(
            "/v1/messages",
            post(move |OriginalUri(uri), headers: HeaderMap, body: Bytes| {
                let route_requests = Arc::clone(&route_requests);
                async move {
                    route_requests.lock().await.push(RecordedRequest {
                        uri: uri.to_string(),
                        headers,
                        body,
                    });
                    Response::builder()
                        .status(status)
                        .header("content-type", content_type)
                        .body(axum::body::Body::from(response_body))
                        .unwrap()
                        .into_response()
                }
            }),
        )
        .route(
            "/v1/messages/count_tokens",
            post(move |OriginalUri(uri), headers: HeaderMap, body: Bytes| {
                let route_requests = Arc::clone(&count_tokens_requests);
                async move {
                    route_requests.lock().await.push(RecordedRequest {
                        uri: uri.to_string(),
                        headers,
                        body,
                    });
                    Response::builder()
                        .status(status)
                        .header("content-type", content_type)
                        .body(axum::body::Body::from(response_body))
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

#[tokio::test]
async fn messages_forwards_raw_body_query_headers_and_open_beta_list() {
    let (llm_url, requests, _upstream) =
        spawn_recording_upstream(StatusCode::OK, "application/json", r#"{"id":"msg_1"}"#).await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    let body = br#"{"model":"test","tools":[{"name":"WebSearch","description":"Search the web","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},{"name":"WebFetch","description":"Fetch a web page","input_schema":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}],"stream":false,"new_field":{"keep":true}}"#;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/messages?beta=true"))
        .header("anthropic-version", "2023-06-01")
        .header("anthropic-beta", "future-beta-unknown,web-search-2025-03-05")
        .header("x-claude-code-session-id", "session-1")
        .header("x-claude-code-agent-id", "agent-1")
        .header("x-api-key", "anthropic-key")
        .header("connection", "keep-alive")
        .header("host", "gateway.invalid")
        .body(body.to_vec())
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), r#"{"id":"msg_1"}"#);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].uri, "/v1/messages?beta=true");
    assert_eq!(requests[0].body.as_ref(), body);
    assert_eq!(requests[0].headers["anthropic-version"], "2023-06-01");
    assert_eq!(
        requests[0].headers["anthropic-beta"],
        "future-beta-unknown,web-search-2025-03-05"
    );
    assert_eq!(requests[0].headers["x-claude-code-session-id"], "session-1");
    assert_eq!(requests[0].headers["x-claude-code-agent-id"], "agent-1");
    assert_eq!(requests[0].headers["x-api-key"], "anthropic-key");
    assert!(!requests[0].headers.contains_key("connection"));
    assert_ne!(
        requests[0].headers.get("host").and_then(|v| v.to_str().ok()),
        Some("gateway.invalid")
    );
}

#[tokio::test]
async fn messages_forwards_system_attribution_blocks_verbatim() {
    let (llm_url, requests, _upstream) =
        spawn_recording_upstream(StatusCode::OK, "application/json", r#"{"id":"msg_system"}"#).await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    let body = br#"{"model":"test","system":[{"type":"text","text":"<attribution>session-1</attribution>"},{"type":"text","text":"You are helpful."}],"messages":[{"role":"user","content":"hello"}],"stream":false}"#;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/messages"))
        .body(body.to_vec())
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].body.as_ref(), body);
}

#[tokio::test]
async fn messages_forwards_sse_bytes_unchanged() {
    let sse = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let (llm_url, _requests, _upstream) = spawn_recording_upstream(StatusCode::OK, "text/event-stream", sse).await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/messages"))
        .body(r#"{"model":"test","stream":true}"#)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(response.bytes().await.unwrap().as_ref(), sse.as_bytes());
}

#[tokio::test]
async fn messages_count_tokens_uses_matching_upstream_path() {
    let (llm_url, requests, _upstream) =
        spawn_recording_upstream(StatusCode::OK, "application/json", r#"{"input_tokens":3}"#).await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/messages/count_tokens"))
        .body(r#"{"model":"test","messages":[]}"#)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), r#"{"input_tokens":3}"#);
    assert_eq!(requests.lock().await[0].uri, "/v1/messages/count_tokens");
}

#[tokio::test]
async fn messages_preserves_upstream_error_status_and_body() {
    let (llm_url, _requests, _upstream) = spawn_recording_upstream(
        StatusCode::BAD_REQUEST,
        "application/json",
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#,
    )
    .await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/messages"))
        .body("{}")
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response.text().await.unwrap(),
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#
    );
}

#[tokio::test]
async fn messages_returns_anthropic_error_for_unreachable_upstream() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_addr = listener.local_addr().unwrap();
    drop(listener);
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&format!("http://{dead_addr}")))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/messages"))
        .body("{}")
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(
        body,
        serde_json::json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "LLM unavailable",
            },
        })
    );
}
