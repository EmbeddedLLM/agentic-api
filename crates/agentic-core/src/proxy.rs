use std::pin::Pin;
use std::time::Duration;

use async_stream::stream;
use bytes::Bytes;
use futures::{Stream, StreamExt, TryStreamExt};
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use reqwest::Client;
use serde_json::Value;
use tracing::{debug, warn};

use crate::config::{Config, resolve_model_alias};
use crate::error::Error;
use crate::tool::{CodexNamespaceHandler, RawCodexNamespaceNormalization};

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

#[cfg(test)]
fn normalize_request_body(body: Bytes, config: &Config) -> Bytes {
    normalize_proxy_request_body(body, config).body
}

fn normalize_proxy_request_body(body: Bytes, config: &Config) -> NormalizedProxyRequest {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return NormalizedProxyRequest {
            body,
            namespace: RawCodexNamespaceNormalization::default(),
        };
    };

    let mut changed = false;
    let mut namespace = RawCodexNamespaceNormalization::default();
    if let Some(object) = value.as_object_mut() {
        if let Some(model) = object.get("model").and_then(Value::as_str) {
            let resolved = resolve_model_alias(model, &config.model_aliases);
            if resolved != model {
                debug!(
                    model_before = %model,
                    model_after = %resolved,
                    "rewrote proxy request model alias"
                );
                object.insert("model".to_string(), Value::String(resolved));
                changed = true;
            }
        }
        changed |= CodexNamespaceHandler.flatten_raw_tools_for_upstream(object, &mut namespace);
        changed |= CodexNamespaceHandler.rewrite_raw_tool_choice_for_upstream(object, &namespace);
    }

    if !changed {
        return NormalizedProxyRequest { body, namespace };
    }
    NormalizedProxyRequest {
        body: serde_json::to_vec(&value).map_or(body, Bytes::from),
        namespace,
    }
}

fn normalize_response_body(body: Bytes, namespace: &RawCodexNamespaceNormalization) -> Bytes {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    if !CodexNamespaceHandler.restore_raw_response_value(&mut value, namespace) {
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
                let normalized = CodexNamespaceHandler.restore_raw_sse_line(raw_line, &namespace);
                yield Ok(Bytes::from(format!("{normalized}\n")));
            }
        }

        if !buffer.is_empty() {
            let Ok(raw_line) = std::str::from_utf8(&buffer) else {
                yield Ok(Bytes::from(buffer));
                return;
            };
            let normalized = CodexNamespaceHandler.restore_raw_sse_line(raw_line, &namespace);
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
    let normalized_request = normalize_proxy_request_body(request.body, &state.config);
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

    if response_is_sse {
        response_headers.insert("x-accel-buffering", HeaderValue::from_static("no"));

        let byte_stream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> =
            Box::pin(llm_resp.bytes_stream().map_err(std::io::Error::other));
        let byte_stream = if namespace.is_empty() {
            byte_stream
        } else {
            debug!("normalizing namespace calls in upstream SSE proxy response");
            normalize_sse_stream(byte_stream, namespace)
        };

        return ProxyResponse {
            status,
            headers: response_headers,
            body: ProxyBody::Stream(Box::pin(byte_stream)),
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
        debug!("normalizing namespace calls in upstream JSON proxy response");
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
            model_aliases: std::collections::HashMap::new(),
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
    fn model_alias_rewrites_request_body() {
        let mut config = test_config();
        config
            .model_aliases
            .insert("codex-auto-review".to_string(), "real-model".to_string());
        let body = Bytes::from_static(br#"{"model":"codex-auto-review","input":"hi","store":false}"#);

        let rewritten = normalize_request_body(body, &config);
        let value: Value = serde_json::from_slice(&rewritten).unwrap();

        assert_eq!(value["model"], "real-model");
        assert_eq!(value["input"], "hi");
    }

    #[test]
    fn proxy_request_body_carries_instructions_forward() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","instructions":"rules","input":[{"role":"user","content":"hi"}],"store":false}"#,
        );

        let rewritten = normalize_request_body(body, &config);
        let value: Value = serde_json::from_slice(&rewritten).unwrap();

        assert_eq!(value["instructions"], "rules");
        assert_eq!(value["input"][0]["role"], "user");
        assert_eq!(value["input"][0]["content"], "hi");
    }

    #[test]
    fn proxy_request_body_carries_instructions_with_string_input() {
        let config = test_config();
        let body = Bytes::from_static(br#"{"model":"test","instructions":"rules","input":"hi","store":false}"#);

        let rewritten = normalize_request_body(body, &config);
        let value: Value = serde_json::from_slice(&rewritten).unwrap();

        assert_eq!(value["instructions"], "rules");
        assert_eq!(value["input"], "hi");
    }

    #[test]
    fn proxy_request_body_flattens_namespace_tools_for_upstream() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","input":"hi","tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"echo_text","parameters":{"type":"object"}},{"type":"function","name":"add_numbers","parameters":{"type":"object"}}]}],"store":false}"#,
        );

        let normalized = normalize_proxy_request_body(body, &config);
        let value: Value = serde_json::from_slice(&normalized.body).unwrap();

        assert_eq!(value["tools"].as_array().unwrap().len(), 2);
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["name"], "agentic_ns__mcp__agentic_fixture__echo_text");
        assert_eq!(value["tools"][1]["type"], "function");
        assert_eq!(
            value["tools"][1]["name"],
            "agentic_ns__mcp__agentic_fixture__add_numbers"
        );
        assert!(normalized.namespace.has_original_tools());
        assert!(
            normalized
                .namespace
                .contains_call("agentic_ns__mcp__agentic_fixture__echo_text")
        );
    }

    #[test]
    fn proxy_request_body_rewrites_tool_choice_for_flattened_namespace_tool() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","input":"hi","tools":[{"type":"namespace","name":"mcp__shell","tools":[{"type":"function","name":"run"}]}],"tool_choice":{"type":"function","name":"run"},"store":false}"#,
        );

        let normalized = normalize_proxy_request_body(body, &config);
        let value: Value = serde_json::from_slice(&normalized.body).unwrap();

        assert_eq!(value["tool_choice"]["type"], "function");
        assert_eq!(value["tool_choice"]["name"], "agentic_ns__mcp__shell__run");
    }

    #[test]
    fn proxy_request_body_rewrites_namespaced_tool_choice_for_flattened_namespace_tool() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","input":"hi","tools":[{"type":"namespace","name":"mcp__shell","tools":[{"type":"function","name":"run"}]},{"type":"namespace","name":"mcp__git","tools":[{"type":"function","name":"run"}]}],"tool_choice":{"type":"function","namespace":"mcp__git","name":"run"},"store":false}"#,
        );

        let normalized = normalize_proxy_request_body(body, &config);
        let value: Value = serde_json::from_slice(&normalized.body).unwrap();

        assert_eq!(value["tool_choice"]["type"], "function");
        assert!(value["tool_choice"].get("namespace").is_none());
        assert_eq!(value["tool_choice"]["name"], "agentic_ns__mcp__git__run");
    }

    #[test]
    fn proxy_request_body_does_not_flatten_namespace_member_over_top_level_name() {
        let body = Bytes::from_static(
            br#"{"model":"test","input":"hi","tools":[{"type":"function","name":"agentic_ns__mcp__shell__run"},{"type":"namespace","name":"mcp__shell","tools":[{"type":"function","name":"run"}]},{"type":"namespace","name":"mcp__git","tools":[{"type":"function","name":"status"}]}],"tool_choice":{"type":"function","namespace":"mcp__shell","name":"run"},"store":false}"#,
        );

        let normalized = normalize_proxy_request_body(body, &test_config());
        let value: Value = serde_json::from_slice(&normalized.body).unwrap();
        let tools = value["tools"].as_array().unwrap();
        let flat_function_count = tools
            .iter()
            .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
            .filter(|tool| tool.get("name").and_then(Value::as_str) == Some("agentic_ns__mcp__shell__run"))
            .count();

        assert_eq!(flat_function_count, 1);
        assert!(tools.iter().any(|tool| {
            tool.get("type").and_then(Value::as_str) == Some("namespace")
                && tool.get("name").and_then(Value::as_str) == Some("mcp__shell")
        }));
        assert!(tools.iter().any(|tool| {
            tool.get("type").and_then(Value::as_str) == Some("function")
                && tool.get("name").and_then(Value::as_str) == Some("agentic_ns__mcp__git__status")
        }));
        assert_eq!(value["tool_choice"]["namespace"], "mcp__shell");
        assert_eq!(value["tool_choice"]["name"], "run");
        assert!(!normalized.namespace.contains_call("agentic_ns__mcp__shell__run"));
    }

    #[test]
    fn proxy_request_body_keeps_colliding_namespace_whole_while_flattening_others() {
        for shell_choice in ["status", "run"] {
            let body = Bytes::from(
                serde_json::to_vec(&serde_json::json!({
                    "model": "test",
                    "input": "hi",
                    "tools": [
                        {"type": "function", "name": "agentic_ns__mcp__shell__run"},
                        {
                            "type": "namespace",
                            "name": "mcp__shell",
                            "tools": [
                                {"type": "function", "name": "run"},
                                {"type": "function", "name": "status"}
                            ]
                        },
                        {
                            "type": "namespace",
                            "name": "mcp__git",
                            "tools": [{"type": "function", "name": "status"}]
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "namespace": "mcp__shell",
                        "name": shell_choice
                    },
                    "store": false
                }))
                .unwrap(),
            );

            let normalized = normalize_proxy_request_body(body, &test_config());
            let value: Value = serde_json::from_slice(&normalized.body).unwrap();
            let tools = value["tools"].as_array().unwrap();
            let shell_namespace = tools
                .iter()
                .find(|tool| {
                    tool.get("type").and_then(Value::as_str) == Some("namespace")
                        && tool.get("name").and_then(Value::as_str) == Some("mcp__shell")
                })
                .expect("shell namespace");
            let shell_member_names: Vec<&str> = shell_namespace["tools"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|member| member.get("name").and_then(Value::as_str))
                .collect();

            assert_eq!(shell_member_names, vec!["run", "status"]);
            assert!(!tools.iter().any(|tool| {
                tool.get("type").and_then(Value::as_str) == Some("function")
                    && tool.get("name").and_then(Value::as_str) == Some("agentic_ns__mcp__shell__status")
            }));
            assert!(tools.iter().any(|tool| {
                tool.get("type").and_then(Value::as_str) == Some("function")
                    && tool.get("name").and_then(Value::as_str) == Some("agentic_ns__mcp__git__status")
            }));
            assert_eq!(value["tool_choice"]["namespace"], "mcp__shell");
            assert_eq!(value["tool_choice"]["name"], shell_choice);
            assert!(!normalized.namespace.contains_namespace_call("mcp__shell"));
            assert!(normalized.namespace.contains_call("agentic_ns__mcp__git__status"));
        }
    }

    #[test]
    fn proxy_sse_maps_flat_namespace_member_call() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"echo_text"},{"type":"function","name":"add_numbers"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &config);
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"agentic_ns__mcp__agentic_fixture__echo_text","call_id":"call_1","arguments":"{\"text\":\"hi\"}"}}"#;

        let normalized = CodexNamespaceHandler.restore_raw_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "echo_text");
        assert_eq!(value["item"]["arguments"], "{\"text\":\"hi\"}");
    }

    #[test]
    fn proxy_sse_does_not_namespace_non_function_call_item() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"web_search_call","name":"mcp__agentic_fixture.run","status":"completed"}}"#;

        let normalized = CodexNamespaceHandler.restore_raw_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert!(value["item"].get("namespace").is_none());
        assert_eq!(value["item"]["name"], "mcp__agentic_fixture.run");
        assert_eq!(value["item"]["type"], "web_search_call");
    }

    #[test]
    fn proxy_sse_flat_namespace_member_preserves_tools_argument() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &config);
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"agentic_ns__mcp__agentic_fixture__run","call_id":"call_1","arguments":"{\"tools\":\"legitimate\",\"cmd\":\"pwd\"}"}}"#;

        let normalized = CodexNamespaceHandler.restore_raw_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "run");
        assert_eq!(value["item"]["arguments"], "{\"tools\":\"legitimate\",\"cmd\":\"pwd\"}");
    }

    #[test]
    fn proxy_sse_maps_underscore_namespace_member_alias() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"echo_text"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"mcp__agentic_fixture_echo_text","call_id":"call_1","arguments":"{\"text\":\"hi\"}"}}"#;

        let normalized = CodexNamespaceHandler.restore_raw_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "echo_text");
    }

    #[test]
    fn proxy_sse_ambiguous_underscore_namespace_member_alias_is_not_normalized() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__a_b","tools":[{"type":"function","name":"c"}]},{"type":"namespace","name":"mcp__a","tools":[{"type":"function","name":"b_c"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"mcp__a_b_c","call_id":"call_1","arguments":"{}"}}"#;

        let normalized = CodexNamespaceHandler.restore_raw_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert!(value["item"].get("namespace").is_none());
        assert_eq!(value["item"]["name"], "mcp__a_b_c");
    }

    #[test]
    fn proxy_sse_maps_unambiguous_bare_namespace_member() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"run","call_id":"call_1","arguments":"{\"cmd\":\"pwd\"}"}}"#;

        let normalized = CodexNamespaceHandler.restore_raw_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "run");
    }

    #[test]
    fn proxy_json_restores_original_tools_and_maps_flat_call() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"echo_text"},{"type":"function","name":"add_numbers"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &config);
        let upstream = Bytes::from_static(
            br#"{"id":"resp_1","object":"response","tools":[{"type":"function","name":"agentic_ns__mcp__agentic_fixture__echo_text"}],"output":[{"type":"function_call","name":"agentic_ns__mcp__agentic_fixture__echo_text","call_id":"call_1","arguments":"{\"text\":\"hi\"}"}]}"#,
        );

        let normalized = normalize_response_body(upstream, &normalized_request.namespace);
        let value: Value = serde_json::from_slice(&normalized).unwrap();

        assert_eq!(value["tools"][0]["type"], "namespace");
        assert_eq!(value["tools"][0]["name"], "mcp__agentic_fixture");
        assert_eq!(value["output"][0]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["output"][0]["name"], "echo_text");
    }

    #[test]
    fn proxy_sse_normalizes_namespace_container_call() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"mcp__agentic_fixture","call_id":"call_1","arguments":"{\"tools\":\"opaque\",\"cmd\":\"pwd\"}"}}"#;

        let normalized = CodexNamespaceHandler.restore_raw_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "run");
        assert_eq!(value["item"]["arguments"], "{\"cmd\":\"pwd\"}");
    }

    #[tokio::test]
    async fn proxy_sse_stream_preserves_utf8_split_across_chunks() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let snowman = "\u{2603}";
        let line = format!(
            r#"data: {{"type":"response.output_item.done","item":{{"type":"function_call","name":"agentic_ns__mcp__agentic_fixture__run","call_id":"call_1","arguments":"{{\"text\":\"snow {snowman}\"}}"}}}}"#
        );
        let bytes = format!("{line}\n").into_bytes();
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

        let value: Value = serde_json::from_str(text.trim_end().strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "run");
        assert_eq!(value["item"]["arguments"], format!(r#"{{"text":"snow {snowman}"}}"#));
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
