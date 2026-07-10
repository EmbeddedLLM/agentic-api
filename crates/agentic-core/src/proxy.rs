use std::pin::Pin;
use std::time::Duration;

use async_stream::stream;
use bytes::Bytes;
use futures::{Stream, StreamExt, TryStreamExt};
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use reqwest::Client;
use serde_json::Value;
use tracing::{debug, warn};

use crate::config::Config;
use crate::error::Error;
use crate::tool::codex::RawCodexNamespaceNormalization;
use crate::tool::{CodexNamespaceHandler, ToolError};

const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
];

const REQUEST_DROP_EXTRA: &[&str] = &["host", "content-length"];

fn is_hop_by_hop(name: &str) -> bool {
    HOP_BY_HOP.iter().any(|h| h.eq_ignore_ascii_case(name))
}

fn is_request_drop(name: &str) -> bool {
    is_hop_by_hop(name) || REQUEST_DROP_EXTRA.iter().any(|h| h.eq_ignore_ascii_case(name))
}

pub struct ProxyRequest {
    pub headers: HeaderMap,
    pub body: Bytes,
    pub query: Option<String>,
}

pub enum ProxyBody {
    Full(Bytes),
    Stream(Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>),
}

#[derive(Debug)]
struct NormalizedProxyRequest {
    body: Bytes,
    namespace: RawCodexNamespaceNormalization,
}

pub struct ProxyResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: ProxyBody,
}

#[derive(Clone)]
pub struct ProxyState {
    pub config: Config,
    pub stream_client: Client,
    pub non_stream_client: Client,
}

impl ProxyState {
    /// # Errors
    ///
    /// Returns an error if the HTTP clients cannot be built.
    pub fn new(config: Config) -> Result<Self, Error> {
        let stream_client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(900))
            .pool_max_idle_per_host(0)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(Error::HttpClient)?;

        let non_stream_client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(300))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(Error::HttpClient)?;

        Ok(Self {
            config,
            stream_client,
            non_stream_client,
        })
    }
}

fn filter_request_headers(headers: &HeaderMap, config: &Config) -> reqwest::header::HeaderMap {
    let mut out = reqwest::header::HeaderMap::new();
    for (name, value) in headers {
        if is_request_drop(name.as_str()) {
            continue;
        }
        if let Ok(n) = reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(v) = reqwest::header::HeaderValue::from_bytes(value.as_bytes()) {
                out.append(n, v);
            }
        }
    }

    let has_auth = out.contains_key(reqwest::header::AUTHORIZATION);
    if !has_auth {
        if let Some(key) = config.openai_api_key.as_deref() {
            let trimmed = key.trim();
            if !trimmed.is_empty() {
                if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {trimmed}")) {
                    out.insert(reqwest::header::AUTHORIZATION, v);
                }
            }
        }
    }

    out
}

fn filter_response_headers(headers: &reqwest::header::HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in headers {
        if is_hop_by_hop(name.as_str()) {
            continue;
        }
        if let Ok(n) = HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(v) = HeaderValue::from_bytes(value.as_bytes()) {
                out.append(n, v);
            }
        }
    }
    out
}

fn is_sse_content_type(headers: &reqwest::header::HeaderMap) -> bool {
    headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.to_ascii_lowercase().starts_with("text/event-stream"))
}

fn normalize_proxy_request_body(body: Bytes) -> Result<NormalizedProxyRequest, ToolError> {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return Ok(NormalizedProxyRequest {
            body,
            namespace: RawCodexNamespaceNormalization::default(),
        });
    };

    let mut namespace = RawCodexNamespaceNormalization::default();
    let changed = if let Some(object) = value.as_object_mut() {
        let tools_changed = CodexNamespaceHandler::flatten_raw_tools_for_upstream(object, &mut namespace)?;
        let tool_choice_changed = CodexNamespaceHandler::rewrite_raw_tool_choice_for_upstream(object, &namespace);
        tools_changed || tool_choice_changed
    } else {
        false
    };

    if !changed {
        return Ok(NormalizedProxyRequest { body, namespace });
    }
    Ok(NormalizedProxyRequest {
        body: serde_json::to_vec(&value).map_or(body, Bytes::from),
        namespace,
    })
}

fn normalize_response_body(body: Bytes, namespace: &RawCodexNamespaceNormalization) -> Bytes {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    if !CodexNamespaceHandler::restore_raw_response_value(&mut value, namespace) {
        return body;
    }
    serde_json::to_vec(&value).map_or(body, Bytes::from)
}

fn normalize_sse_stream(
    mut upstream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    namespace: RawCodexNamespaceNormalization,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    Box::pin(stream! {
        let mut buffer = Vec::new();
        while let Some(chunk) = upstream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            buffer.extend_from_slice(&chunk);

            while let Some(pos) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=pos).collect::<Vec<_>>();
                let line_end = if pos > 0 && line.get(pos - 1) == Some(&b'\r') {
                    pos - 1
                } else {
                    pos
                };
                let Ok(raw_line) = std::str::from_utf8(&line[..line_end]) else {
                    yield Ok(Bytes::from(line));
                    continue;
                };
                let normalized = CodexNamespaceHandler::restore_raw_sse_line(raw_line, &namespace);
                yield Ok(Bytes::from(format!("{normalized}\n")));
            }
        }

        if !buffer.is_empty() {
            let Ok(raw_line) = std::str::from_utf8(&buffer) else {
                yield Ok(Bytes::from(buffer));
                return;
            };
            let normalized = CodexNamespaceHandler::restore_raw_sse_line(raw_line, &namespace);
            yield Ok(Bytes::from(normalized));
        }
    })
}

#[must_use]
pub fn error_response(status: StatusCode, code: &str, message: &str) -> ProxyResponse {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "api_error",
            "param": null,
            "code": code,
        }
    });
    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    ProxyResponse {
        status,
        headers,
        body: ProxyBody::Full(Bytes::from(serde_json::to_vec(&body).unwrap_or_default())),
    }
}

/// Proxy a GET request to an arbitrary upstream path.
///
/// Applies the same header filtering and auth injection as [`proxy_request`].
/// Uses the non-streaming client; the response body is returned as a full
/// [`ProxyBody::Full`] payload.
pub async fn proxy_get(path: &str, request_headers: &HeaderMap, state: &ProxyState) -> ProxyResponse {
    let llm_headers = filter_request_headers(request_headers, &state.config);
    let base = state.config.llm_api_base.trim_end_matches('/');
    let url = format!("{base}/{}", path.trim_start_matches('/'));

    let llm_resp = match state.non_stream_client.get(&url).headers(llm_headers).send().await {
        Ok(r) => r,
        Err(e) if e.is_timeout() => {
            warn!("upstream GET {path} timed out: {e}");
            return error_response(StatusCode::GATEWAY_TIMEOUT, "upstream_timeout", "upstream timeout");
        }
        Err(e) => {
            warn!("upstream GET {path} failed: {e}");
            return error_response(StatusCode::BAD_GATEWAY, "upstream_unavailable", "upstream unavailable");
        }
    };

    let status = StatusCode::from_u16(llm_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let response_headers = filter_response_headers(llm_resp.headers());

    match llm_resp.bytes().await {
        Ok(payload) => ProxyResponse {
            status,
            headers: response_headers,
            body: ProxyBody::Full(payload),
        },
        Err(e) => {
            warn!("failed to read upstream GET {path} body: {e}");
            error_response(
                StatusCode::BAD_GATEWAY,
                "upstream_unavailable",
                "failed to read upstream response",
            )
        }
    }
}

pub async fn proxy_request(request: ProxyRequest, state: &ProxyState) -> ProxyResponse {
    let request_body_bytes = request.body.len();
    let is_streaming = serde_json::from_slice::<Value>(&request.body)
        .ok()
        .and_then(|v| v.get("stream")?.as_bool())
        .unwrap_or(false);

    let llm_headers = filter_request_headers(&request.headers, &state.config);
    let normalized_request = match normalize_proxy_request_body(request.body) {
        Ok(request) => request,
        Err(error) => {
            warn!(%error, "rejected invalid namespace tool declaration on proxy path");
            return error_response(StatusCode::BAD_REQUEST, "invalid_request_error", &error.to_string());
        }
    };
    let body = normalized_request.body;
    let namespace = normalized_request.namespace;

    let base = state.config.llm_api_base.trim_end_matches('/');
    let mut url = format!("{base}/v1/responses");
    if let Some(q) = &request.query {
        url.push('?');
        url.push_str(q);
    }
    debug!(
        url = %url,
        stream = is_streaming,
        request_body_bytes,
        normalized_body_bytes = body.len(),
        has_namespace_normalization = !namespace.is_empty(),
        "proxying responses request to upstream"
    );

    let client = if is_streaming {
        &state.stream_client
    } else {
        &state.non_stream_client
    };

    let llm_resp = match client.post(&url).headers(llm_headers).body(body).send().await {
        Ok(r) => r,
        Err(e) if e.is_timeout() => {
            warn!("LLM request timed out: {e}");
            return error_response(StatusCode::GATEWAY_TIMEOUT, "llm_timeout", "LLM timeout");
        }
        Err(e) => {
            warn!("LLM request failed: {e}");
            return error_response(StatusCode::BAD_GATEWAY, "llm_unavailable", "LLM unavailable");
        }
    };

    let status = StatusCode::from_u16(llm_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let response_is_sse = is_sse_content_type(llm_resp.headers());
    debug!(
        status = status.as_u16(),
        response_is_sse,
        has_namespace_normalization = !namespace.is_empty(),
        "received upstream proxy response"
    );
    let mut response_headers = filter_response_headers(llm_resp.headers());
    if !namespace.is_empty() {
        // Restoring flattened names and the original `tools` declaration can
        // change the body length, so the upstream value is no longer valid.
        response_headers.remove("content-length");
    }

    if response_is_sse {
        response_headers.insert("x-accel-buffering", HeaderValue::from_static("no"));

        let byte_stream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> =
            Box::pin(llm_resp.bytes_stream().map_err(std::io::Error::other));
        let byte_stream = if namespace.is_empty() {
            byte_stream
        } else {
            normalize_sse_stream(byte_stream, namespace)
        };

        return ProxyResponse {
            status,
            headers: response_headers,
            body: ProxyBody::Stream(byte_stream),
        };
    }

    let mut payload: Bytes = match llm_resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            warn!("failed to read LLM response body: {e}");
            return error_response(
                StatusCode::BAD_GATEWAY,
                "llm_unavailable",
                "Failed to read LLM response",
            );
        }
    };
    if !namespace.is_empty() {
        payload = normalize_response_body(payload, &namespace);
    }

    ProxyResponse {
        status,
        headers: response_headers,
        body: ProxyBody::Full(payload),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn test_config() -> Config {
        Config {
            llm_api_base: "http://localhost:8000".to_owned(),
            openai_api_key: Some("test-key".to_owned()),
            llm_ready_timeout_s: 5.0,
            llm_ready_interval_s: 0.1,
            skip_llm_ready_check: false,
            db_url: None,
        }
    }

    fn test_config_no_key() -> Config {
        Config {
            openai_api_key: None,
            ..test_config()
        }
    }

    #[test]
    fn hop_by_hop_detected() {
        assert!(is_hop_by_hop("connection"));
        assert!(is_hop_by_hop("Connection"));
        assert!(is_hop_by_hop("keep-alive"));
        assert!(is_hop_by_hop("transfer-encoding"));
        assert!(is_hop_by_hop("proxy-authorization"));
    }

    #[test]
    fn non_hop_by_hop_passes() {
        assert!(!is_hop_by_hop("content-type"));
        assert!(!is_hop_by_hop("x-custom"));
        assert!(!is_hop_by_hop("authorization"));
    }

    #[test]
    fn request_drop_includes_host_and_content_length() {
        assert!(is_request_drop("host"));
        assert!(is_request_drop("content-length"));
        assert!(is_request_drop("connection"));
        assert!(!is_request_drop("content-type"));
    }

    #[test]
    fn filter_request_headers_strips_hop_by_hop() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("connection", "keep-alive".parse().unwrap());
        headers.insert("proxy-authorization", "Basic abc".parse().unwrap());
        headers.insert("x-custom", "value".parse().unwrap());

        let config = test_config_no_key();
        let filtered = filter_request_headers(&headers, &config);

        assert!(filtered.contains_key("content-type"));
        assert!(filtered.contains_key("x-custom"));
        assert!(!filtered.contains_key("connection"));
        assert!(!filtered.contains_key("proxy-authorization"));
    }

    #[test]
    fn filter_request_headers_strips_host_and_content_length() {
        let mut headers = HeaderMap::new();
        headers.insert("host", "example.com".parse().unwrap());
        headers.insert("content-length", "42".parse().unwrap());
        headers.insert("accept", "*/*".parse().unwrap());

        let config = test_config_no_key();
        let filtered = filter_request_headers(&headers, &config);

        assert!(!filtered.contains_key("host"));
        assert!(!filtered.contains_key("content-length"));
        assert!(filtered.contains_key("accept"));
    }

    #[test]
    fn auth_injected_when_no_client_auth() {
        let headers = HeaderMap::new();
        let config = test_config();
        let filtered = filter_request_headers(&headers, &config);

        assert_eq!(
            filtered.get("authorization").unwrap().to_str().unwrap(),
            "Bearer test-key"
        );
    }

    #[test]
    fn client_auth_takes_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer client-token".parse().unwrap());

        let config = test_config();
        let filtered = filter_request_headers(&headers, &config);

        assert_eq!(
            filtered.get("authorization").unwrap().to_str().unwrap(),
            "Bearer client-token"
        );
    }

    #[test]
    fn no_auth_injected_when_key_empty() {
        let headers = HeaderMap::new();
        let config = Config {
            openai_api_key: Some("  ".to_owned()),
            ..test_config()
        };
        let filtered = filter_request_headers(&headers, &config);

        assert!(!filtered.contains_key("authorization"));
    }

    #[test]
    fn no_auth_injected_when_key_none() {
        let headers = HeaderMap::new();
        let config = test_config_no_key();
        let filtered = filter_request_headers(&headers, &config);

        assert!(!filtered.contains_key("authorization"));
    }

    #[test]
    fn raw_proxy_flattens_namespace_tools_and_rewrites_tool_choice() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__shell","tools":[{"type":"function","name":"run"}]}],"tool_choice":{"type":"function","namespace":"mcp__shell","name":"run"}}"#,
        );

        let normalized = normalize_proxy_request_body(body).expect("valid raw namespace tools");
        let value: Value = serde_json::from_slice(&normalized.body).unwrap();

        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["name"], "agentic_ns__mcp__shell__run");
        assert_eq!(value["tool_choice"]["name"], "agentic_ns__mcp__shell__run");
        assert!(value["tool_choice"].get("namespace").is_none());
        assert!(!normalized.namespace.is_empty());
    }

    #[test]
    fn raw_proxy_rejects_ambiguous_flattened_namespace_names() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"a__b","tools":[{"type":"function","name":"c"}]},{"type":"namespace","name":"a","tools":[{"type":"function","name":"b__c"}]}]}"#,
        );

        let error = normalize_proxy_request_body(body).unwrap_err();

        assert!(error.to_string().contains("a.b__c collides with a__b.c"));
    }

    #[test]
    fn raw_proxy_restores_tools_and_function_calls_in_json_response() {
        let request = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__shell","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(request).expect("valid raw namespace tools");
        let upstream = Bytes::from_static(
            br#"{"tools":[{"type":"function","name":"agentic_ns__mcp__shell__run"}],"output":[{"type":"function_call","name":"agentic_ns__mcp__shell__run"}]}"#,
        );

        let restored = normalize_response_body(upstream, &normalized_request.namespace);
        let value: Value = serde_json::from_slice(&restored).unwrap();

        assert_eq!(value["tools"][0]["type"], "namespace");
        assert_eq!(value["tools"][0]["name"], "mcp__shell");
        assert_eq!(value["output"][0]["namespace"], "mcp__shell");
        assert_eq!(value["output"][0]["name"], "run");
    }

    #[test]
    fn raw_proxy_restores_tools_in_nested_sse_response() {
        let request = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__shell","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(request).expect("valid raw namespace tools");
        let line = r#"data: {"type":"response.completed","response":{"tools":[{"type":"function","name":"agentic_ns__mcp__shell__run"}],"output":[{"type":"function_call","name":"agentic_ns__mcp__shell__run"}]}}"#;

        let restored = CodexNamespaceHandler::restore_raw_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(restored.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["response"]["tools"][0]["type"], "namespace");
        assert_eq!(value["response"]["tools"][0]["name"], "mcp__shell");
        assert_eq!(value["response"]["output"][0]["namespace"], "mcp__shell");
        assert_eq!(value["response"]["output"][0]["name"], "run");
    }

    #[tokio::test]
    async fn raw_proxy_sse_restores_utf8_split_across_chunks() {
        let request = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__shell","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(request).expect("valid raw namespace tools");
        let snowman = "\u{2603}";
        let line = format!(
            r#"data: {{"type":"response.output_item.done","item":{{"type":"function_call","name":"agentic_ns__mcp__shell__run","arguments":"{{\"text\":\"{snowman}\"}}"}}}}"#
        ) + "\n";
        let bytes = line.as_bytes();
        let split_at = bytes
            .windows(snowman.len())
            .position(|window| window == snowman.as_bytes())
            .expect("snowman bytes present")
            + 1;
        let chunks = vec![
            Ok(Bytes::copy_from_slice(&bytes[..split_at])),
            Ok(Bytes::copy_from_slice(&bytes[split_at..])),
        ];
        let mut stream = normalize_sse_stream(Box::pin(futures::stream::iter(chunks)), normalized_request.namespace);

        let output = stream.next().await.expect("normalized line").expect("stream ok");
        assert!(stream.next().await.is_none());
        let text = String::from_utf8(output.to_vec()).expect("normalized line is utf8");
        assert!(!text.contains('\u{FFFD}'));

        let value: Value = serde_json::from_str(text.trim().strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(value["item"]["namespace"], "mcp__shell");
        assert_eq!(value["item"]["name"], "run");
    }

    #[test]
    fn filter_response_headers_strips_hop_by_hop() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("connection", "keep-alive".parse().unwrap());
        headers.insert("x-request-id", "abc".parse().unwrap());

        let filtered = filter_response_headers(&headers);

        assert!(filtered.contains_key("content-type"));
        assert!(filtered.contains_key("x-request-id"));
        assert!(!filtered.contains_key("connection"));
    }

    #[test]
    fn sse_content_type_detected() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("content-type", "text/event-stream; charset=utf-8".parse().unwrap());
        assert!(is_sse_content_type(&headers));
    }

    #[test]
    fn sse_content_type_case_insensitive() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("content-type", "Text/Event-Stream".parse().unwrap());
        assert!(is_sse_content_type(&headers));
    }

    #[test]
    fn non_sse_content_type_rejected() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        assert!(!is_sse_content_type(&headers));
    }

    #[test]
    fn missing_content_type_not_sse() {
        let headers = reqwest::header::HeaderMap::new();
        assert!(!is_sse_content_type(&headers));
    }
}
