use async_stream::stream;
use futures::{Stream, StreamExt};
use tracing::warn;

use crate::config::RuntimeConfig;
use crate::types::responses::{ResponsesRequest, StreamEvent};
use crate::utils::errors::AgenticApiError;

const DEFAULT_BUFFER_SIZE: usize = 8192;
const MAX_BUFFER_SIZE: usize = 4 * 1024 * 1024;

#[derive(Clone)]
pub struct Agent {
    client: reqwest::Client,
    responses_url: String,
    api_key: Option<String>,
    buffer_capacity: usize,
}

impl Agent {
    #[must_use]
    pub fn new(config: &RuntimeConfig) -> Self {
        Self::with_timeouts(
            config,
            std::time::Duration::from_secs(10),
            std::time::Duration::from_secs(300),
        )
    }

    #[must_use]
    pub fn new_for_test(config: &RuntimeConfig, read_timeout_ms: u64) -> Self {
        Self::with_timeouts(
            config,
            std::time::Duration::from_millis(100),
            std::time::Duration::from_millis(read_timeout_ms),
        )
    }

    fn with_timeouts(
        config: &RuntimeConfig,
        connect_timeout: std::time::Duration,
        read_timeout: std::time::Duration,
    ) -> Self {
        let base = config.llm_api_base.trim_end_matches('/');
        Self {
            client: reqwest::Client::builder()
                .connect_timeout(connect_timeout)
                .read_timeout(read_timeout)
                .build()
                .expect("failed to build HTTP client"),
            responses_url: format!("{base}/v1/responses"),
            api_key: config.openai_api_key.clone(),
            buffer_capacity: DEFAULT_BUFFER_SIZE,
        }
    }

    fn estimate_buffer_size(&self, content_length: Option<u64>) -> usize {
        content_length
            .and_then(|len| {
                usize::try_from(len)
                    .ok()
                    .map(|len| len.min(MAX_BUFFER_SIZE).max(self.buffer_capacity))
            })
            .unwrap_or(self.buffer_capacity)
    }

    pub fn run_stream(
        &self,
        request: &ResponsesRequest,
        client_auth: Option<&str>,
    ) -> impl Stream<Item = Result<StreamEvent, AgenticApiError>> + '_ {
        let upstream_req = request.to_upstream_request(true);
        let mut req = self.client.post(&self.responses_url).json(&upstream_req);
        let effective_auth = client_auth.or(self.api_key.as_deref());
        if let Some(key) = effective_auth {
            req = req.bearer_auth(key);
        }

        stream! {
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) if e.is_timeout() => {
                    yield Err(AgenticApiError::responses_api(
                        "Upstream timeout",
                        504,
                        "api_error",
                        None,
                        Some("upstream_timeout".into()),
                    ));
                    return;
                }
                Err(_) => {
                    yield Err(AgenticApiError::responses_api(
                        "Upstream unavailable",
                        502,
                        "api_error",
                        None,
                        Some("upstream_unavailable".into()),
                    ));
                    return;
                }
            };

            if !resp.status().is_success() {
                let status = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();
                yield Err(AgenticApiError::responses_api(
                    format!("upstream error: {body}"),
                    status,
                    "api_error",
                    None,
                    None,
                ));
                return;
            }

            let buf_capacity = self.estimate_buffer_size(resp.content_length());
            let mut stream = resp.bytes_stream();
            let mut buf = String::with_capacity(buf_capacity);

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(AgenticApiError::bad_input(format!("stream read error: {e}")));
                        return;
                    }
                };

                match std::str::from_utf8(&chunk) {
                    Ok(s) => buf.push_str(s),
                    Err(_) => buf.push_str(&String::from_utf8_lossy(&chunk)),
                }

                while let Some(pos) = buf.find('\n') {
                    let line_end = if pos > 0 && buf.as_bytes()[pos - 1] == b'\r' {
                        pos - 1
                    } else {
                        pos
                    };

                    if let Some(data) = buf[..line_end].strip_prefix("data: ") {
                        if data == "[DONE]" {
                            return;
                        }
                        match serde_json::from_str::<StreamEvent>(data) {
                            Ok(event) => yield Ok(event),
                            Err(e) => {
                                if data.len() > 100 {
                                    warn!("failed to parse SSE event: {e}");
                                } else {
                                    warn!("failed to parse SSE event: {e} — data: {data}");
                                }
                            }
                        }
                    }
                    buf.drain(..=pos);
                }
            }
        }
    }
}
