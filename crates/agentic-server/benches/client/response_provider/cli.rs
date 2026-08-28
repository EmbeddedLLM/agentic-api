use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use crate::types::Workload;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ProviderSelection {
    /// Gateway WebSocket, HTTP/SSE, and JSON plus direct vLLM HTTP/SSE and JSON.
    All,
    /// Gateway WebSocket and direct vLLM HTTP/SSE (backwards-compatible pair).
    Both,
    AgenticApi,
    AgenticApiHttp,
    AgenticApiJson,
    Vllm,
    VllmJson,
}

#[derive(Debug, Parser)]
#[command(
    name = "agentic-responses-benchmark",
    about = "Run concurrent Responses workloads against Agentic API and direct vLLM"
)]
pub struct Cli {
    /// Model slug exposed by each selected provider.
    #[arg(long, env = "RESPONSE_PROVIDER_BENCH_MODEL")]
    pub model: String,

    /// Capability measured by this run.
    #[arg(long, value_enum, default_value_t = Workload::HistoryRehydration)]
    pub workload: Workload,

    /// Number of concurrent sessions per selected provider.
    #[arg(long, default_value_t = 5)]
    pub sessions: usize,

    /// Sequential turns in each session. Defaults depend on `--workload`.
    #[arg(long, short = 'n')]
    pub requests_per_session: Option<usize>,

    /// Fixed session depths (history-rehydration only), each run as its own independent batch of
    /// `--sessions` sessions instead of bucketing one long run after the fact. Overrides
    /// `--requests-per-session` when set. Example: --depths 1,5,10,25,50,100
    #[arg(long, value_delimiter = ',')]
    pub depths: Vec<usize>,

    /// Responses created inside each transport workload turn.
    #[arg(long, default_value_t = 4)]
    pub transport_rounds: usize,

    /// BFCL question JSONL for the tool-call workload.
    #[arg(long, requires = "dataset_answers")]
    pub dataset_questions: Option<PathBuf>,

    /// BFCL possible-answer JSONL paired by case ID.
    #[arg(long, requires = "dataset_questions")]
    pub dataset_answers: Option<PathBuf>,

    /// Zero-based BFCL case offset before selecting deterministic cases.
    #[arg(long, default_value_t = 0)]
    pub dataset_offset: usize,

    /// Providers to benchmark. `all` includes both gateway transports.
    #[arg(long, value_enum, default_value_t = ProviderSelection::Both)]
    pub provider: ProviderSelection,

    /// Agentic API OpenAI-compatible base URL.
    #[arg(long, default_value = "http://localhost:9000/v1")]
    pub agentic_url: String,

    /// Direct vLLM OpenAI-compatible base URL.
    #[arg(long, default_value = "http://localhost:5050/v1")]
    pub vllm_url: String,

    /// Timeout for one logical turn, including every model/tool round.
    #[arg(long, default_value_t = 300)]
    pub timeout_seconds: u64,

    /// Seed for deterministic generated prompts and continuation secrets.
    #[arg(long, default_value_t = 2_026_081_7)]
    pub seed: u64,

    /// Result directory. Defaults to target/response-provider-benchmark/<unix-milliseconds>.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// Stream timestamped, provider-tagged Responses events to stdout.
    #[arg(long)]
    pub live_jsonl: bool,

    /// Validate inputs, print generated prompt specifications as JSONL, and exit.
    #[arg(long)]
    pub print_prompts: bool,

    /// Automatically appended by `cargo bench` for custom harnesses.
    #[arg(long, hide = true)]
    pub bench: bool,
}

impl Cli {
    #[must_use]
    pub fn requests_per_session(&self) -> usize {
        self.requests_per_session
            .unwrap_or_else(|| self.workload.default_requests())
    }
}
