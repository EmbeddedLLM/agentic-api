use async_stream::stream;
use futures::{Stream, StreamExt};
use tracing::warn;

use crate::config::RuntimeConfig;
use crate::types::responses::{ResponsesRequest, StreamEvent};
use crate::utils::errors::AgenticApiError;

#[derive(Clone)]
pub struct Agent {
    client: reqwest::Client,
    responses_url: String,
    api_key: Option<String>,
}

impl Agent {
    pub fn new(config: &RuntimeConfig) -> Self {
        let base = config.llm_api_base.trim_end_matches('/');
        Self {
            client: reqwest::Client::new(),
            responses_url: format!("{base}/v1/responses"),
            api_key: config.openai_api_key.clone(),
        }
    }

    pub fn run_stream(
        &self,
        request: &ResponsesRequest,
    ) -> impl Stream<Item = Result<StreamEvent, AgenticApiError>> + '_ {
        let body = request.to_upstream_body(true);
        let mut req = self.client.post(&self.responses_url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        stream! {
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(AgenticApiError::bad_input(format!("upstream request failed: {e}")));
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

            let mut stream = resp.bytes_stream();
            let mut buf = String::new();

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(AgenticApiError::bad_input(format!("stream read error: {e}")));
                        return;
                    }
                };

                buf.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].trim_end_matches('\r').to_string();
                    buf = buf[pos + 1..].to_string();

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            return;
                        }
                        match serde_json::from_str::<StreamEvent>(data) {
                            Ok(event) => yield Ok(event),
                            Err(e) => warn!("failed to parse SSE event: {e} — data: {data}"),
                        }
                    }
                }
            }
        }
    }
}
