use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::TryStreamExt;
use http::{HeaderMap, HeaderName, StatusCode};
use reqwest::Client;
use serde_json::Value;

use crate::config::RuntimeConfig;

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

const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

fn is_hop_by_hop(name: &str) -> bool {
    HOP_BY_HOP.iter().any(|h| h.eq_ignore_ascii_case(name))
}

fn is_request_drop(name: &str) -> bool {
    is_hop_by_hop(name) || REQUEST_DROP_EXTRA.iter().any(|h| h.eq_ignore_ascii_case(name))
}

#[derive(Clone)]
pub struct ProxyState {
    pub config: RuntimeConfig,
    pub stream_client: Client,
    pub non_stream_client: Client,
}

impl ProxyState {
    /// # Panics
    ///
    /// Panics if the HTTP clients cannot be built (invalid TLS backend).
    #[must_use]
    pub fn new(config: RuntimeConfig) -> Self {
        let stream_client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(0)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("failed to build stream HTTP client");

        let non_stream_client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(300))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("failed to build non-stream HTTP client");

        Self {
            config,
            stream_client,
            non_stream_client,
        }
    }
}

fn filter_request_headers(headers: &HeaderMap, config: &RuntimeConfig) -> reqwest::header::HeaderMap {
    let mut out = reqwest::header::HeaderMap::new();
    for (name, value) in headers {
        if is_request_drop(name.as_str()) {
            continue;
        }
        if let Ok(n) = reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(v) = reqwest::header::HeaderValue::from_bytes(value.as_bytes()) {
                out.insert(n, v);
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
            if let Ok(v) = http::HeaderValue::from_bytes(value.as_bytes()) {
                out.insert(n, v);
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

fn proxy_error(status: StatusCode, code: &str, message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "api_error",
            "param": null,
            "code": code,
        }
    });
    (status, axum::Json(body)).into_response()
}

pub async fn proxy_responses(State(state): State<ProxyState>, req: axum::extract::Request) -> Response {
    let (parts, body) = req.into_parts();
    let Ok(body_bytes) = axum::body::to_bytes(body, MAX_BODY_SIZE).await else {
        return proxy_error(StatusCode::BAD_REQUEST, "body_too_large", "Request body too large");
    };

    let is_streaming = serde_json::from_slice::<Value>(&body_bytes)
        .ok()
        .and_then(|v| v.get("stream")?.as_bool())
        .unwrap_or(false);

    let upstream_headers = filter_request_headers(&parts.headers, &state.config);

    let base = state.config.llm_api_base.trim_end_matches('/');
    let mut url = format!("{base}/v1/responses");
    if let Some(q) = parts.uri.query() {
        url.push('?');
        url.push_str(q);
    }

    let client = if is_streaming {
        &state.stream_client
    } else {
        &state.non_stream_client
    };

    let upstream_resp = match client
        .post(&url)
        .headers(upstream_headers)
        .body(body_bytes)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) if e.is_timeout() => {
            return proxy_error(StatusCode::GATEWAY_TIMEOUT, "upstream_timeout", "Upstream timeout");
        }
        Err(_) => {
            return proxy_error(StatusCode::BAD_GATEWAY, "upstream_unavailable", "Upstream unavailable");
        }
    };

    let status = StatusCode::from_u16(upstream_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response_headers = filter_response_headers(upstream_resp.headers());

    if is_sse_content_type(upstream_resp.headers()) {
        response_headers.insert("x-accel-buffering", http::HeaderValue::from_static("no"));

        let byte_stream = upstream_resp.bytes_stream().map_err(std::io::Error::other);

        let body = Body::from_stream(byte_stream);
        let mut resp = Response::new(body);
        *resp.status_mut() = status;
        *resp.headers_mut() = response_headers;
        return resp;
    }

    let payload: Bytes = match upstream_resp.bytes().await {
        Ok(b) => b,
        Err(_) => {
            return proxy_error(
                StatusCode::BAD_GATEWAY,
                "upstream_unavailable",
                "Failed to read upstream response",
            );
        }
    };

    let mut resp = Response::new(Body::from(payload));
    *resp.status_mut() = status;
    *resp.headers_mut() = response_headers;
    resp
}
