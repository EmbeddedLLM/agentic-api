use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Serialize)]
pub struct ProviderSpec {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub transport: Transport,
    pub supports_websockets: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    Websocket,
    HttpSse,
    HttpJson,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Workload {
    /// Multi-round turns that exercise connection reuse and streaming transport.
    Transport,
    /// BFCL function-selection and argument-accuracy cases sent as Responses tools.
    ToolCall,
    /// Short continuations that require state from the immediately preceding turn.
    HistoryRehydration,
}

impl Workload {
    #[must_use]
    pub const fn default_requests(self) -> usize {
        match self {
            Self::Transport | Self::ToolCall => 1,
            Self::HistoryRehydration => 10,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExpectedToolCall {
    pub name: String,
    /// BFCL represents each accepted argument value as a list of alternatives.
    pub arguments: BTreeMap<String, Vec<Value>>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TurnExpectation {
    Marker {
        marker: String,
    },
    ToolCalls {
        calls: Vec<ExpectedToolCall>,
    },
    Transport {
        calls: Vec<ExpectedToolCall>,
        final_marker: String,
    },
}

impl TurnExpectation {
    #[must_use]
    pub fn expected_marker(&self) -> Option<&str> {
        match self {
            Self::Marker { marker, .. } => Some(marker),
            Self::ToolCalls { .. } => None,
            Self::Transport { final_marker, .. } => Some(final_marker),
        }
    }

    #[must_use]
    pub fn expected_tool_calls(&self) -> &[ExpectedToolCall] {
        match self {
            Self::Marker { .. } => &[],
            Self::ToolCalls { calls } | Self::Transport { calls, .. } => calls,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PromptSpec {
    pub workload: Workload,
    pub session_index: usize,
    pub turn_index: usize,
    pub prompt_id: String,
    pub source_id: Option<String>,
    pub prompt: String,
    pub expectation: TurnExpectation,
    pub tools: Vec<ToolDefinition>,
}

#[derive(Clone, Debug)]
pub struct SessionSpec {
    pub session_index: usize,
    pub prompts: Vec<PromptSpec>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[allow(clippy::struct_field_names)]
pub struct Usage {
    pub input_tokens: i64,
    pub cached_input_tokens: i64,
    pub output_tokens: i64,
    pub reasoning_output_tokens: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ObservedToolCall {
    pub name: String,
    pub arguments: Value,
    pub started_at_ms: Option<u64>,
    pub completed_at_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct TurnResult {
    pub provider: String,
    pub transport: Transport,
    pub workload: Workload,
    pub session_index: usize,
    pub turn_index: usize,
    pub prompt_id: String,
    pub source_id: Option<String>,
    pub expected_marker: Option<String>,
    pub response_id: Option<String>,
    pub attempted: bool,
    pub success: bool,
    pub task_correct: bool,
    pub tool_call_correct: bool,
    pub timed_out: bool,
    pub transport_fallback: bool,
    pub saw_turn_completed: bool,
    pub invalid_json_lines: usize,
    pub error_events: usize,
    pub tool_calls_started: usize,
    pub tool_calls_completed: usize,
    pub tool_calls_failed: usize,
    pub observed_tool_calls: Vec<ObservedToolCall>,
    pub expected_tool_calls: Vec<ExpectedToolCall>,
    pub tool_output_marker_found: bool,
    pub final_answer_marker_found: bool,
    pub exact_final_answer: bool,
    pub time_to_first_output_event_ms: Option<u64>,
    pub first_output_event_type: Option<String>,
    pub time_to_first_tool_call_ms: Option<u64>,
    pub ttft_ms: Option<u64>,
    pub mean_tool_duration_ms: Option<f64>,
    pub initial_model_round_ms: Option<u64>,
    pub continuation_round_latencies_ms: Vec<u64>,
    pub end_to_end_latency_ms: u64,
    /// Bytes of serialized request body sent to the provider, summed across every model round in this turn.
    pub request_bytes: u64,
    /// Bytes of raw response payload (WS text frames / SSE chunks / JSON body) received from the provider.
    pub response_bytes: u64,
    pub turn_usage: Option<Usage>,
    pub effective_output_tokens_per_second: Option<f64>,
    pub raw_jsonl_path: String,
    pub timestamped_jsonl_path: String,
    pub error_log_path: String,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionResult {
    pub provider: String,
    pub session_index: usize,
    pub elapsed_ms: u64,
    pub fatal_error: Option<String>,
    pub turns: Vec<TurnResult>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Distribution {
    pub count: usize,
    pub mean: Option<f64>,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProviderSummary {
    pub provider: String,
    pub transport: Transport,
    pub planned_turns: usize,
    pub attempted_turns: usize,
    pub successful_turns: usize,
    pub timed_out_turns: usize,
    pub transport_fallback_turns: usize,
    pub tool_compliant_turns: usize,
    pub task_correct_turns: usize,
    pub tool_call_correct_turns: usize,
    pub success_rate: f64,
    pub tool_compliance_rate: f64,
    pub task_correctness_rate: f64,
    pub tool_call_accuracy: f64,
    pub provider_wall_clock_ms: u64,
    pub successful_turns_per_second: f64,
    pub end_to_end_latency_ms: Distribution,
    pub time_to_first_output_event_ms: Distribution,
    pub ttft_ms: Distribution,
    pub time_to_first_tool_call_ms: Distribution,
    pub mean_tool_duration_ms: Distribution,
    pub continuation_round_latency_ms: Distribution,
    pub request_bytes: Distribution,
    pub response_bytes: Distribution,
    pub total_turn_input_tokens: i64,
    pub total_turn_cached_input_tokens: i64,
    pub total_turn_output_tokens: i64,
    pub total_turn_reasoning_output_tokens: i64,
    pub aggregate_effective_output_tokens_per_second: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Comparison {
    pub paired_successful_turns: usize,
    pub median_agentic_minus_vllm_latency_ms: Option<f64>,
    pub median_agentic_over_vllm_latency_ratio: Option<f64>,
    pub median_agentic_minus_vllm_first_output_ms: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunConfig {
    pub model: String,
    pub workload: Workload,
    pub dataset_questions: Option<PathBuf>,
    pub dataset_answers: Option<PathBuf>,
    pub sessions_per_provider: usize,
    pub requests_per_session: usize,
    pub seed: u64,
    pub timeout_seconds: u64,
    pub providers: Vec<ProviderSpec>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunReport {
    pub schema_version: u32,
    pub started_at_unix_ms: u64,
    pub elapsed_ms: u64,
    pub output_dir: PathBuf,
    pub config: RunConfig,
    pub prompts: Vec<PromptSpec>,
    pub sessions: Vec<SessionResult>,
    pub summaries: Vec<ProviderSummary>,
    pub comparison: Option<Comparison>,
    pub ttft_note: String,
    pub accuracy_note: String,
}

/// One provider's results at one fixed session depth, produced by running that depth as its own
/// independent batch of sessions rather than bucketing a single long-lived run after the fact.
#[derive(Clone, Debug, Serialize)]
pub struct DepthSummaryRow {
    pub depth: usize,
    pub provider: String,
    pub transport: Transport,
    pub sessions: usize,
    pub success_rate: f64,
    pub task_correctness_rate: f64,
    pub p50_latency_ms: Option<f64>,
    pub p50_request_bytes: Option<f64>,
    pub p50_response_bytes: Option<f64>,
    pub total_request_bytes: u64,
    pub total_response_bytes: u64,
}
