use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{SinkExt, StreamExt};
use reqwest::Client;
use serde_json::{Map, Value, json};
use thiserror::Error;
use tokio::net::TcpStream;
use tokio::sync::{Barrier, Mutex};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

use crate::prompts::transport_marker;
use crate::types::{
    ExpectedToolCall, ObservedToolCall, PromptSpec, ProviderSpec, SessionResult, ToolDefinition, Transport,
    TurnExpectation, TurnResult, Usage,
};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Clone, Debug)]
pub struct RunnerConfig {
    pub model: String,
    pub output_dir: PathBuf,
    pub timeout: Duration,
    pub live_jsonl: bool,
}

#[derive(Debug, Error)]
enum RunnerError {
    #[error("failed to access {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("WebSocket request failed: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),
    #[error("invalid Responses endpoint {0}")]
    InvalidEndpoint(String),
    #[error("provider returned HTTP {status}: {body}")]
    HttpStatus { status: reqwest::StatusCode, body: String },
    #[error("invalid provider event JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Responses protocol error: {0}")]
    Protocol(String),
}

#[derive(Clone, Debug)]
struct TimedEvent {
    elapsed_ms: u64,
    value: Value,
}

#[derive(Debug)]
struct ModelResponse {
    events: Vec<TimedEvent>,
    response: Value,
    round_latency_ms: u64,
    first_output_ms: Option<u64>,
    first_output_event_type: Option<String>,
    first_tool_call_ms: Option<u64>,
    first_text_delta_ms: Option<u64>,
    request_bytes: u64,
    response_bytes: u64,
    errors: Vec<String>,
}

struct LiveEventContext<'a> {
    enabled: bool,
    provider: &'a ProviderSpec,
    session_index: usize,
    turn_index: usize,
    output_lock: &'a Mutex<()>,
}

enum TransportClient {
    WebSocket(Box<WsStream>),
    HttpSse { client: Client, url: String },
    HttpJson { client: Client, url: String },
}

#[derive(Default)]
struct ConversationState {
    previous_response_id: Option<String>,
    replay_items: Vec<Value>,
}

struct TurnExecution {
    events: Vec<TimedEvent>,
    response_id: Option<String>,
    saw_completed: bool,
    observed_tool_calls: Vec<ObservedToolCall>,
    tool_calls_failed: usize,
    tool_output_marker_found: bool,
    final_answer: Option<String>,
    first_output_ms: Option<u64>,
    first_output_event_type: Option<String>,
    first_tool_call_ms: Option<u64>,
    first_text_delta_ms: Option<u64>,
    round_latencies_ms: Vec<u64>,
    request_bytes: u64,
    response_bytes: u64,
    usage: Usage,
    errors: Vec<String>,
}

pub async fn run_session(
    config: Arc<RunnerConfig>,
    provider: ProviderSpec,
    session_index: usize,
    prompts: Vec<PromptSpec>,
    start_barrier: Arc<Barrier>,
    live_output_lock: Arc<Mutex<()>>,
) -> SessionResult {
    let session_started = Instant::now();
    let mut result = SessionResult {
        provider: provider.id.clone(),
        session_index,
        elapsed_ms: 0,
        fatal_error: None,
        turns: Vec::with_capacity(prompts.len()),
    };

    let mut client = match TransportClient::connect(&provider).await {
        Ok(client) => client,
        Err(error) => {
            result.fatal_error = Some(error.to_string());
            result.elapsed_ms = millis(session_started.elapsed());
            start_barrier.wait().await;
            return result;
        }
    };
    let mut state = ConversationState::default();
    start_barrier.wait().await;
    let workload_started = Instant::now();

    for prompt in prompts {
        let turn = run_turn(
            &config,
            &provider,
            session_index,
            &prompt,
            &mut client,
            &mut state,
            &live_output_lock,
        )
        .await;
        let can_continue = turn.saw_turn_completed && !turn.timed_out && turn.error_events == 0;
        if matches!(
            prompt.workload,
            crate::types::Workload::Transport | crate::types::Workload::ToolCall
        ) {
            state = ConversationState::default();
        }
        eprintln!(
            "[{provider_id} s{session_index:03} t{turn_index:03}] success={success} latency={latency}ms tools={tools}",
            provider_id = provider.id,
            turn_index = prompt.turn_index,
            success = turn.success,
            latency = turn.end_to_end_latency_ms,
            tools = turn.tool_calls_completed,
        );
        result.turns.push(turn);
        if !can_continue {
            result.fatal_error = Some(format!(
                "turn {} did not leave a resumable Responses session",
                prompt.turn_index
            ));
            break;
        }
    }

    result.elapsed_ms = millis(workload_started.elapsed());
    result
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn run_turn(
    config: &RunnerConfig,
    provider: &ProviderSpec,
    session_index: usize,
    prompt: &PromptSpec,
    client: &mut TransportClient,
    state: &mut ConversationState,
    live_output_lock: &Mutex<()>,
) -> TurnResult {
    let started = Instant::now();
    let context = LiveEventContext {
        enabled: config.live_jsonl,
        provider,
        session_index,
        turn_index: prompt.turn_index,
        output_lock: live_output_lock,
    };
    let execution = tokio::time::timeout(
        config.timeout,
        execute_turn(config, provider, prompt, client, state, started, &context),
    )
    .await;

    let (execution, timed_out) = match execution {
        Ok(Ok(execution)) => (execution, false),
        Ok(Err(error)) => (failed_execution(error.to_string()), false),
        Err(_) => (
            failed_execution(format!("turn timed out after {} ms", millis(config.timeout))),
            true,
        ),
    };
    let elapsed_ms = millis(started.elapsed());
    let event_dir = config
        .output_dir
        .join("events")
        .join(&provider.id)
        .join(format!("session-{session_index:03}"));
    let raw_path = event_dir.join(format!("turn-{:03}.responses.jsonl", prompt.turn_index));
    let timestamped_path = event_dir.join(format!("turn-{:03}.timestamped.jsonl", prompt.turn_index));
    let error_path = event_dir.join(format!("turn-{:03}.errors.log", prompt.turn_index));
    if let Err(error) = write_event_files(
        &raw_path,
        &timestamped_path,
        &error_path,
        &execution.events,
        &execution.errors,
    )
    .await
    {
        eprintln!("warning: failed to write turn event files: {error}");
    }

    let expected_marker = prompt.expectation.expected_marker();
    let final_answer_marker_found = expected_marker.is_some_and(|marker| {
        execution
            .final_answer
            .as_deref()
            .is_some_and(|answer| answer.contains(marker))
    });
    let exact_final_answer = expected_marker.is_some_and(|marker| {
        execution
            .final_answer
            .as_deref()
            .is_some_and(|answer| answer.trim().trim_matches('`') == marker)
    });
    let tool_call_correct = match &prompt.expectation {
        TurnExpectation::Marker { .. } => false,
        TurnExpectation::ToolCalls { .. } => {
            tool_calls_match(prompt.expectation.expected_tool_calls(), &execution.observed_tool_calls)
        }
        TurnExpectation::Transport { .. } => {
            tool_calls_match_ordered(prompt.expectation.expected_tool_calls(), &execution.observed_tool_calls)
        }
    };
    let task_correct = match &prompt.expectation {
        TurnExpectation::Marker { .. } => exact_final_answer,
        TurnExpectation::ToolCalls { .. } => tool_call_correct,
        TurnExpectation::Transport { .. } => {
            tool_call_correct && execution.tool_output_marker_found && exact_final_answer
        }
    };
    let success = !timed_out
        && execution.saw_completed
        && execution.errors.is_empty()
        && execution.tool_calls_failed == 0
        && task_correct;
    let tool_durations = execution.observed_tool_calls.iter().filter_map(|call| {
        call.started_at_ms
            .map(|started_at| call.completed_at_ms.saturating_sub(started_at))
    });
    let tool_durations = tool_durations.collect::<Vec<_>>();
    let mean_tool_duration_ms = (!tool_durations.is_empty()).then(|| {
        tool_durations.iter().map(|value| u64_as_f64(*value)).sum::<f64>() / usize_as_f64(tool_durations.len())
    });
    let output_tokens = execution.usage.output_tokens;
    let effective_output_tokens_per_second =
        (elapsed_ms > 0).then_some(i64_as_f64(output_tokens) * 1_000.0 / u64_as_f64(elapsed_ms));
    let initial_model_round_ms = execution.round_latencies_ms.first().copied();
    let continuation_round_latencies_ms = execution.round_latencies_ms.iter().skip(1).copied().collect();

    TurnResult {
        provider: provider.id.clone(),
        transport: provider.transport,
        workload: prompt.workload,
        session_index,
        turn_index: prompt.turn_index,
        prompt_id: prompt.prompt_id.clone(),
        source_id: prompt.source_id.clone(),
        expected_marker: expected_marker.map(str::to_owned),
        response_id: execution.response_id,
        attempted: true,
        success,
        task_correct,
        tool_call_correct,
        timed_out,
        transport_fallback: false,
        saw_turn_completed: execution.saw_completed,
        invalid_json_lines: 0,
        error_events: execution.errors.len(),
        tool_calls_started: execution.observed_tool_calls.len(),
        tool_calls_completed: execution.observed_tool_calls.len(),
        tool_calls_failed: execution.tool_calls_failed,
        observed_tool_calls: execution.observed_tool_calls,
        expected_tool_calls: prompt.expectation.expected_tool_calls().to_vec(),
        tool_output_marker_found: execution.tool_output_marker_found,
        final_answer_marker_found,
        exact_final_answer,
        time_to_first_output_event_ms: execution.first_output_ms,
        first_output_event_type: execution.first_output_event_type,
        time_to_first_tool_call_ms: execution.first_tool_call_ms,
        ttft_ms: execution.first_text_delta_ms,
        mean_tool_duration_ms,
        initial_model_round_ms,
        continuation_round_latencies_ms,
        end_to_end_latency_ms: elapsed_ms,
        request_bytes: execution.request_bytes,
        response_bytes: execution.response_bytes,
        turn_usage: Some(execution.usage),
        effective_output_tokens_per_second,
        raw_jsonl_path: relative_path(&config.output_dir, &raw_path),
        timestamped_jsonl_path: relative_path(&config.output_dir, &timestamped_path),
        error_log_path: relative_path(&config.output_dir, &error_path),
        errors: execution.errors,
    }
}

fn failed_execution(message: String) -> TurnExecution {
    TurnExecution {
        events: Vec::new(),
        response_id: None,
        saw_completed: false,
        observed_tool_calls: Vec::new(),
        tool_calls_failed: 0,
        tool_output_marker_found: false,
        final_answer: None,
        first_output_ms: None,
        first_output_event_type: None,
        first_tool_call_ms: None,
        first_text_delta_ms: None,
        round_latencies_ms: Vec::new(),
        request_bytes: 0,
        response_bytes: 0,
        usage: Usage::default(),
        errors: vec![message],
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn execute_turn(
    config: &RunnerConfig,
    provider: &ProviderSpec,
    prompt: &PromptSpec,
    client: &mut TransportClient,
    state: &mut ConversationState,
    turn_started: Instant,
    live_context: &LiveEventContext<'_>,
) -> Result<TurnExecution, RunnerError> {
    let gateway_managed_history = provider.id.starts_with("agentic-api");
    let mut new_items = vec![json!({
        "type": "message",
        "role": "user",
        "content": prompt.prompt,
    })];
    let max_rounds = match &prompt.expectation {
        TurnExpectation::Transport { calls, .. } => calls.len().saturating_add(1),
        TurnExpectation::Marker { .. } | TurnExpectation::ToolCalls { .. } => 1,
    };
    let mut events = Vec::new();
    let mut response_id = None;
    let mut saw_completed = true;
    let mut observed_tool_calls = Vec::new();
    let mut tool_calls_failed = 0;
    let mut tool_output_marker_found = false;
    let mut final_answer = None;
    let mut first_output_ms = None;
    let mut first_output_event_type = None;
    let mut first_tool_call_ms = None;
    let mut first_text_delta_ms = None;
    let mut round_latencies_ms = Vec::new();
    let mut request_bytes = 0u64;
    let mut response_bytes = 0u64;
    let mut usage = Usage::default();
    let mut errors = Vec::new();

    for _ in 0..max_rounds {
        let input = if gateway_managed_history {
            new_items.clone()
        } else {
            state
                .replay_items
                .iter()
                .cloned()
                .chain(new_items.iter().cloned())
                .collect()
        };
        let body = response_request(
            &config.model,
            input,
            &prompt.tools,
            client.is_streaming(),
            gateway_managed_history,
            gateway_managed_history
                .then(|| state.previous_response_id.clone())
                .flatten(),
        );
        let model_response = client.request(&body, turn_started, live_context).await?;
        first_output_ms = first_output_ms.or(model_response.first_output_ms);
        if first_output_event_type.is_none() {
            first_output_event_type.clone_from(&model_response.first_output_event_type);
        }
        first_tool_call_ms = first_tool_call_ms.or(model_response.first_tool_call_ms);
        first_text_delta_ms = first_text_delta_ms.or(model_response.first_text_delta_ms);
        round_latencies_ms.push(model_response.round_latency_ms);
        request_bytes = request_bytes.saturating_add(model_response.request_bytes);
        response_bytes = response_bytes.saturating_add(model_response.response_bytes);
        errors.extend(model_response.errors.clone());
        events.extend(model_response.events.clone());

        let current_response_id = model_response
            .response
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let status = model_response
            .response
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        saw_completed &= status == "completed";
        let output = model_response
            .response
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        add_usage(&mut usage, &response_usage(&model_response.response));
        final_answer = response_text(&model_response.response).or(final_answer);
        let response_completed_ms = model_response
            .events
            .last()
            .map_or(model_response.round_latency_ms, |event| event.elapsed_ms);
        let calls = response_tool_calls(&output, &model_response.events, response_completed_ms);

        if gateway_managed_history {
            state.previous_response_id.clone_from(&current_response_id);
        } else {
            state.replay_items.extend(new_items);
            state.replay_items.extend(output.clone());
        }
        response_id = current_response_id;
        observed_tool_calls.extend(calls.iter().map(|call| call.observed.clone()));

        if !matches!(prompt.expectation, TurnExpectation::Transport { .. }) || calls.is_empty() {
            break;
        }

        new_items = calls
            .into_iter()
            .map(|call| {
                let output = match execute_transport_call(&call.observed) {
                    Ok(output) => {
                        if prompt
                            .expectation
                            .expected_marker()
                            .is_some_and(|marker| output.contains(marker))
                        {
                            tool_output_marker_found = true;
                        }
                        output
                    }
                    Err(error) => {
                        tool_calls_failed += 1;
                        errors.push(error.clone());
                        format!("BENCHMARK_TOOL_ERROR: {error}")
                    }
                };
                json!({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                })
            })
            .collect();
    }

    Ok(TurnExecution {
        events,
        response_id,
        saw_completed,
        observed_tool_calls,
        tool_calls_failed,
        tool_output_marker_found,
        final_answer,
        first_output_ms,
        first_output_event_type,
        first_tool_call_ms,
        first_text_delta_ms,
        round_latencies_ms,
        request_bytes,
        response_bytes,
        usage,
        errors,
    })
}

fn response_request(
    model: &str,
    input: Vec<Value>,
    tools: &[ToolDefinition],
    stream: bool,
    store: bool,
    previous_response_id: Option<String>,
) -> Value {
    let mut body = Map::from_iter([
        ("model".to_owned(), json!(model)),
        ("input".to_owned(), Value::Array(input)),
        ("stream".to_owned(), json!(stream)),
        ("store".to_owned(), json!(store)),
    ]);
    if let Some(previous_response_id) = previous_response_id {
        body.insert("previous_response_id".to_owned(), json!(previous_response_id));
    }
    if !tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(
                tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        })
                    })
                    .collect(),
            ),
        );
        body.insert("tool_choice".to_owned(), json!("auto"));
        body.insert("parallel_tool_calls".to_owned(), json!(false));
    }
    Value::Object(body)
}

impl TransportClient {
    async fn connect(provider: &ProviderSpec) -> Result<Self, RunnerError> {
        let endpoint = responses_endpoint(&provider.base_url);
        match provider.transport {
            Transport::Websocket => {
                let websocket_url = websocket_url(&endpoint)?;
                let (socket, _) = connect_async(websocket_url.as_str()).await?;
                Ok(Self::WebSocket(Box::new(socket)))
            }
            Transport::HttpSse => Ok(Self::HttpSse {
                client: Client::new(),
                url: endpoint,
            }),
            Transport::HttpJson => Ok(Self::HttpJson {
                client: Client::new(),
                url: endpoint,
            }),
        }
    }

    const fn is_streaming(&self) -> bool {
        matches!(self, Self::WebSocket(_) | Self::HttpSse { .. })
    }

    async fn request(
        &mut self,
        body: &Value,
        turn_started: Instant,
        live_context: &LiveEventContext<'_>,
    ) -> Result<ModelResponse, RunnerError> {
        match self {
            Self::WebSocket(socket) => websocket_request(socket, body, turn_started, live_context).await,
            Self::HttpSse { client, url } => http_sse_request(client, url, body, turn_started, live_context).await,
            Self::HttpJson { client, url } => http_json_request(client, url, body, turn_started, live_context).await,
        }
    }
}

async fn websocket_request(
    socket: &mut WsStream,
    body: &Value,
    turn_started: Instant,
    live_context: &LiveEventContext<'_>,
) -> Result<ModelResponse, RunnerError> {
    let round_started = Instant::now();
    let mut request = body.clone();
    request
        .as_object_mut()
        .ok_or_else(|| RunnerError::Protocol("request body must be an object".to_owned()))?
        .insert("type".to_owned(), json!("response.create"));
    let request_text = request.to_string();
    let request_bytes = byte_len(request_text.len());
    socket.send(Message::Text(request_text.into())).await?;
    let mut events = Vec::new();
    let mut response_bytes = 0u64;
    loop {
        let message = socket
            .next()
            .await
            .ok_or_else(|| RunnerError::Protocol("WebSocket closed before a terminal event".to_owned()))??;
        match message {
            Message::Text(text) => {
                response_bytes = response_bytes.saturating_add(byte_len(text.len()));
                let value = serde_json::from_str::<Value>(&text)?;
                let event = TimedEvent {
                    elapsed_ms: millis(turn_started.elapsed()),
                    value,
                };
                emit_live_event(live_context, &event).await;
                let terminal = is_terminal_event(&event.value);
                events.push(event);
                if terminal {
                    break;
                }
            }
            Message::Ping(payload) => socket.send(Message::Pong(payload)).await?,
            Message::Pong(_) | Message::Frame(_) => {}
            Message::Close(frame) => {
                return Err(RunnerError::Protocol(format!(
                    "WebSocket closed before a terminal event: {frame:?}"
                )));
            }
            Message::Binary(_) => {
                return Err(RunnerError::Protocol(
                    "WebSocket returned a binary frame instead of JSON text".to_owned(),
                ));
            }
        }
    }
    model_response_from_stream(events, millis(round_started.elapsed()), request_bytes, response_bytes)
}

async fn http_sse_request(
    client: &Client,
    url: &str,
    body: &Value,
    turn_started: Instant,
    live_context: &LiveEventContext<'_>,
) -> Result<ModelResponse, RunnerError> {
    let round_started = Instant::now();
    let request_body = serde_json::to_vec(body)?;
    let request_bytes = byte_len(request_body.len());
    let response = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(request_body)
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        return Err(RunnerError::HttpStatus {
            status,
            body: response.text().await.unwrap_or_default(),
        });
    }
    let mut stream = response.bytes_stream();
    let mut pending = Vec::new();
    let mut events = Vec::new();
    let mut terminal = false;
    let mut response_bytes = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        response_bytes = response_bytes.saturating_add(byte_len(chunk.len()));
        pending.extend_from_slice(&chunk);
        while let Some(frame) = take_sse_frame(&mut pending) {
            let Some(value) = parse_sse_frame(&frame)? else {
                continue;
            };
            let event = TimedEvent {
                elapsed_ms: millis(turn_started.elapsed()),
                value,
            };
            emit_live_event(live_context, &event).await;
            terminal |= is_terminal_event(&event.value);
            events.push(event);
        }
        if terminal {
            break;
        }
    }
    if !terminal {
        return Err(RunnerError::Protocol(
            "HTTP/SSE stream ended before a terminal event".to_owned(),
        ));
    }
    model_response_from_stream(events, millis(round_started.elapsed()), request_bytes, response_bytes)
}

async fn http_json_request(
    client: &Client,
    url: &str,
    body: &Value,
    turn_started: Instant,
    live_context: &LiveEventContext<'_>,
) -> Result<ModelResponse, RunnerError> {
    let round_started = Instant::now();
    let request_body = serde_json::to_vec(body)?;
    let request_bytes = byte_len(request_body.len());
    let response = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(request_body)
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        return Err(RunnerError::HttpStatus {
            status,
            body: response.text().await.unwrap_or_default(),
        });
    }
    let response_body = response.bytes().await?;
    let response_bytes = byte_len(response_body.len());
    let response = serde_json::from_slice::<Value>(&response_body)?;
    let event = TimedEvent {
        elapsed_ms: millis(turn_started.elapsed()),
        value: response.clone(),
    };
    emit_live_event(live_context, &event).await;
    let status = response.get("status").and_then(Value::as_str).unwrap_or_default();
    let errors = if status == "completed" {
        Vec::new()
    } else {
        vec![response_error(&response)]
    };
    Ok(ModelResponse {
        events: vec![event],
        response,
        round_latency_ms: millis(round_started.elapsed()),
        first_output_ms: None,
        first_output_event_type: None,
        first_tool_call_ms: None,
        first_text_delta_ms: None,
        request_bytes,
        response_bytes,
        errors,
    })
}

fn model_response_from_stream(
    events: Vec<TimedEvent>,
    round_latency_ms: u64,
    request_bytes: u64,
    response_bytes: u64,
) -> Result<ModelResponse, RunnerError> {
    let terminal = events
        .last()
        .ok_or_else(|| RunnerError::Protocol("provider returned no Responses events".to_owned()))?;
    let event_type = terminal.value.get("type").and_then(Value::as_str).unwrap_or_default();
    let response = terminal
        .value
        .get("response")
        .cloned()
        .ok_or_else(|| RunnerError::Protocol(format!("terminal event {event_type:?} has no response object")))?;
    let first_output_ms = events
        .iter()
        .find(|event| is_output_event(&event.value))
        .map(|event| event.elapsed_ms);
    let first_output_event_type = events
        .iter()
        .find(|event| is_output_event(&event.value))
        .and_then(|event| event.value.get("type"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let first_tool_call_ms = events
        .iter()
        .find(|event| is_tool_event(&event.value))
        .map(|event| event.elapsed_ms);
    let first_text_delta_ms = events
        .iter()
        .find(|event| event.value.get("type").and_then(Value::as_str) == Some("response.output_text.delta"))
        .map(|event| event.elapsed_ms);
    let errors = if event_type == "response.completed" && response["status"] == "completed" {
        Vec::new()
    } else {
        vec![response_error(&terminal.value)]
    };
    Ok(ModelResponse {
        events,
        response,
        round_latency_ms,
        first_output_ms,
        first_output_event_type,
        first_tool_call_ms,
        first_text_delta_ms,
        request_bytes,
        response_bytes,
        errors,
    })
}

fn responses_endpoint(base_url: &str) -> String {
    format!("{}/responses", base_url.trim_end_matches('/'))
}

fn websocket_url(endpoint: &str) -> Result<url::Url, RunnerError> {
    let mut url = url::Url::parse(endpoint).map_err(|_| RunnerError::InvalidEndpoint(endpoint.to_owned()))?;
    let scheme = match url.scheme() {
        "http" => "ws",
        "https" => "wss",
        "ws" | "wss" => return Ok(url),
        _ => return Err(RunnerError::InvalidEndpoint(endpoint.to_owned())),
    };
    url.set_scheme(scheme)
        .map_err(|()| RunnerError::InvalidEndpoint(endpoint.to_owned()))?;
    Ok(url)
}

fn take_sse_frame(pending: &mut Vec<u8>) -> Option<Vec<u8>> {
    let position = pending
        .windows(2)
        .position(|window| window == b"\n\n")
        .or_else(|| pending.windows(4).position(|window| window == b"\r\n\r\n"))?;
    let delimiter_len = if pending.get(position..position + 4) == Some(b"\r\n\r\n") {
        4
    } else {
        2
    };
    let frame = pending.drain(..position).collect();
    pending.drain(..delimiter_len);
    Some(frame)
}

fn parse_sse_frame(frame: &[u8]) -> Result<Option<Value>, RunnerError> {
    let text = std::str::from_utf8(frame)
        .map_err(|error| RunnerError::Protocol(format!("SSE frame is not UTF-8: {error}")))?;
    let data = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&data)?))
}

fn is_terminal_event(value: &Value) -> bool {
    matches!(
        value.get("type").and_then(Value::as_str),
        Some("response.completed" | "response.failed" | "response.incomplete" | "error")
    )
}

fn is_output_event(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str).is_some_and(|kind| {
        kind.starts_with("response.output_")
            || kind.starts_with("response.function_call_arguments.")
            || kind.starts_with("response.reasoning_")
    })
}

fn is_tool_event(value: &Value) -> bool {
    let kind = value.get("type").and_then(Value::as_str).unwrap_or_default();
    kind.starts_with("response.function_call_arguments.")
        || (kind == "response.output_item.added"
            && value.pointer("/item/type").and_then(Value::as_str) == Some("function_call"))
}

#[derive(Clone)]
struct CompletedToolCall {
    call_id: String,
    observed: ObservedToolCall,
}

fn response_tool_calls(output: &[Value], events: &[TimedEvent], fallback_completed_ms: u64) -> Vec<CompletedToolCall> {
    output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
        .map(|item| {
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let item_id = item.get("id").and_then(Value::as_str);
            let started_at_ms = events.iter().find_map(|event| {
                let event_item = event.value.get("item")?;
                let matches_id = item_id.is_some_and(|id| event_item.get("id").and_then(Value::as_str) == Some(id));
                let matches_call =
                    !call_id.is_empty() && event_item.get("call_id").and_then(Value::as_str) == Some(call_id.as_str());
                (event.value.get("type").and_then(Value::as_str) == Some("response.output_item.added")
                    && (matches_id || matches_call))
                    .then_some(event.elapsed_ms)
            });
            let completed_at_ms = events
                .iter()
                .find_map(|event| {
                    let event_item = event.value.get("item")?;
                    let matches_id = item_id.is_some_and(|id| event_item.get("id").and_then(Value::as_str) == Some(id));
                    let matches_call = !call_id.is_empty()
                        && event_item.get("call_id").and_then(Value::as_str) == Some(call_id.as_str());
                    (event.value.get("type").and_then(Value::as_str) == Some("response.output_item.done")
                        && (matches_id || matches_call))
                        .then_some(event.elapsed_ms)
                })
                .unwrap_or(fallback_completed_ms);
            let arguments = item
                .get("arguments")
                .and_then(Value::as_str)
                .and_then(|text| serde_json::from_str(text).ok())
                .unwrap_or_else(|| item.get("arguments").cloned().unwrap_or(Value::Null));
            CompletedToolCall {
                call_id,
                observed: ObservedToolCall {
                    name: item.get("name").and_then(Value::as_str).unwrap_or_default().to_owned(),
                    arguments,
                    started_at_ms,
                    completed_at_ms,
                },
            }
        })
        .collect()
}

fn execute_transport_call(call: &ObservedToolCall) -> Result<String, String> {
    if call.name != "benchmark_step" {
        return Err(format!("unexpected transport tool {:?}", call.name));
    }
    let arguments = call
        .arguments
        .as_object()
        .ok_or_else(|| "benchmark_step arguments must be a JSON object".to_owned())?;
    let run_id = arguments
        .get("run_id")
        .and_then(Value::as_str)
        .ok_or_else(|| "benchmark_step run_id must be a string".to_owned())?;
    let step = positive_usize(arguments, "step")?;
    let total_steps = positive_usize(arguments, "total_steps")?;
    Ok(transport_marker(run_id, step, total_steps))
}

fn positive_usize(arguments: &Map<String, Value>, name: &str) -> Result<usize, String> {
    arguments
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("benchmark_step {name} must be a positive integer"))
}

fn response_text(response: &Value) -> Option<String> {
    let texts = response
        .get("output")?
        .as_array()?
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter_map(|content| content.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>();
    (!texts.is_empty()).then(|| texts.join(""))
}

fn response_usage(response: &Value) -> Usage {
    let usage = response.get("usage").unwrap_or(&Value::Null);
    Usage {
        input_tokens: usage.get("input_tokens").and_then(Value::as_i64).unwrap_or_default(),
        cached_input_tokens: usage
            .pointer("/input_tokens_details/cached_tokens")
            .or_else(|| usage.get("cached_input_tokens"))
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        output_tokens: usage.get("output_tokens").and_then(Value::as_i64).unwrap_or_default(),
        reasoning_output_tokens: usage
            .pointer("/output_tokens_details/reasoning_tokens")
            .or_else(|| usage.get("reasoning_output_tokens"))
            .and_then(Value::as_i64)
            .unwrap_or_default(),
    }
}

fn add_usage(total: &mut Usage, value: &Usage) {
    total.input_tokens = total.input_tokens.saturating_add(value.input_tokens);
    total.cached_input_tokens = total.cached_input_tokens.saturating_add(value.cached_input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(value.output_tokens);
    total.reasoning_output_tokens = total
        .reasoning_output_tokens
        .saturating_add(value.reasoning_output_tokens);
}

fn response_error(value: &Value) -> String {
    value
        .pointer("/error/message")
        .or_else(|| value.pointer("/response/error/message"))
        .or_else(|| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("provider returned a non-completed response")
        .to_owned()
}

async fn emit_live_event(context: &LiveEventContext<'_>, event: &TimedEvent) {
    if !context.enabled {
        return;
    }
    let wrapper = json!({
        "provider": context.provider.id,
        "transport": context.provider.transport,
        "session_index": context.session_index,
        "turn_index": context.turn_index,
        "elapsed_ms": event.elapsed_ms,
        "event": event.value,
    });
    let _guard = context.output_lock.lock().await;
    println!("{wrapper}");
}

async fn write_event_files(
    raw_path: &Path,
    timestamped_path: &Path,
    error_path: &Path,
    events: &[TimedEvent],
    errors: &[String],
) -> Result<(), RunnerError> {
    let parent = raw_path
        .parent()
        .ok_or_else(|| RunnerError::Protocol("event path has no parent".to_owned()))?;
    create_dir_all(parent).await?;
    let mut raw = String::new();
    let mut timestamped = String::new();
    for event in events {
        raw.push_str(&event.value.to_string());
        raw.push('\n');
        timestamped.push_str(&json!({"elapsed_ms": event.elapsed_ms, "event": event.value}).to_string());
        timestamped.push('\n');
    }
    write_file(raw_path, raw.as_bytes()).await?;
    write_file(timestamped_path, timestamped.as_bytes()).await?;
    write_file(error_path, errors.join("\n").as_bytes()).await
}

fn tool_calls_match(expected: &[ExpectedToolCall], observed: &[ObservedToolCall]) -> bool {
    if expected.len() != observed.len() {
        return false;
    }
    let mut matched = vec![false; observed.len()];
    for expected_call in expected {
        let Some((index, _)) = observed.iter().enumerate().find(|(index, observed_call)| {
            !matched[*index]
                && observed_call.name == expected_call.name
                && arguments_match(&expected_call.arguments, &observed_call.arguments)
        }) else {
            return false;
        };
        matched[index] = true;
    }
    true
}

fn tool_calls_match_ordered(expected: &[ExpectedToolCall], observed: &[ObservedToolCall]) -> bool {
    expected.len() == observed.len()
        && expected.iter().zip(observed).all(|(expected_call, observed_call)| {
            observed_call.name == expected_call.name
                && arguments_match(&expected_call.arguments, &observed_call.arguments)
        })
}

fn arguments_match(expected: &BTreeMap<String, Vec<Value>>, observed: &Value) -> bool {
    let parsed;
    let observed = if let Some(object) = observed.as_object() {
        object
    } else if let Some(text) = observed.as_str() {
        parsed = serde_json::from_str::<Value>(text).ok();
        let Some(object) = parsed.as_ref().and_then(Value::as_object) else {
            return expected.is_empty();
        };
        object
    } else {
        return expected.is_empty();
    };

    if observed.keys().any(|name| !expected.contains_key(name)) {
        return false;
    }
    expected.iter().all(|(name, accepted)| match observed.get(name) {
        Some(value) => accepted.iter().any(|candidate| candidate == value),
        None => accepted.iter().any(is_omission_sentinel),
    })
}

fn is_omission_sentinel(value: &Value) -> bool {
    value.as_str() == Some("")
}

async fn create_dir_all(path: &Path) -> Result<(), RunnerError> {
    tokio::fs::create_dir_all(path).await.map_err(|source| RunnerError::Io {
        path: path.to_owned(),
        source,
    })
}

async fn write_file(path: &Path, contents: &[u8]) -> Result<(), RunnerError> {
    tokio::fs::write(path, contents)
        .await
        .map_err(|source| RunnerError::Io {
            path: path.to_owned(),
            source,
        })
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).unwrap_or(path).to_string_lossy().into_owned()
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn byte_len(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

#[allow(clippy::cast_precision_loss)]
fn i64_as_f64(value: i64) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn u64_as_f64(value: u64) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn usize_as_f64(value: usize) -> f64 {
    value as f64
}
