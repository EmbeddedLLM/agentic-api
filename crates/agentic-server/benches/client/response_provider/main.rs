mod cli;
mod prompts;
mod report;
mod runner;
mod types;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use thiserror::Error;
use tokio::sync::{Barrier, Mutex};
use tokio::task::JoinSet;

use crate::cli::{Cli, ProviderSelection};
use crate::runner::RunnerConfig;
use crate::types::{ProviderSpec, RunConfig, RunReport, SessionResult, Transport, Workload};

const TTFT_NOTE: &str = "For WebSocket and HTTP/SSE runs, time_to_first_output_event_ms is measured from request send \
to the first output event, and ttft_ms is measured to the first output_text delta (token-level \
TTFT). Non-streaming HTTP/JSON has no token-level TTFT, so those fields are null.";

#[derive(Debug, Error)]
enum Error {
    #[error("{0}")]
    InvalidArgument(String),
    #[error("failed to access {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("benchmark task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error(transparent)]
    Prompt(#[from] prompts::PromptError),
}

#[tokio::main]
#[allow(clippy::too_many_lines)]
async fn main() -> Result<(), Error> {
    let cli = Cli::parse();
    validate(&cli)?;
    let started_at_unix_ms = unix_millis();
    let dataset_questions = match &cli.dataset_questions {
        Some(path) => Some(canonicalize(path).await?),
        None => None,
    };
    let dataset_answers = match &cli.dataset_answers {
        Some(path) => Some(canonicalize(path).await?),
        None => None,
    };
    let base_output_dir = cli
        .output_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("target/response-provider-benchmark/{started_at_unix_ms}")));

    if cli.print_prompts {
        let turn_counts = if cli.depths.is_empty() {
            vec![cli.requests_per_session()]
        } else {
            cli.depths.clone()
        };
        for turns in turn_counts {
            let generated = prompts::generate(&prompts::GenerationConfig {
                workload: cli.workload,
                seed: cli.seed,
                sessions: cli.sessions,
                turns,
                transport_rounds: cli.transport_rounds,
                dataset_questions: dataset_questions.clone(),
                dataset_answers: dataset_answers.clone(),
                dataset_offset: cli.dataset_offset,
            })
            .await?;
            for prompt in generated.iter().flat_map(|session| &session.prompts) {
                println!(
                    "{}",
                    serde_json::to_string(prompt).map_err(|source| Error::Io {
                        path: PathBuf::from("generated prompt JSON"),
                        source: std::io::Error::other(source),
                    })?
                );
            }
        }
        return Ok(());
    }

    if cli.depths.is_empty() {
        let run_report = run_pipeline(
            &cli,
            cli.requests_per_session(),
            base_output_dir,
            dataset_questions,
            dataset_answers,
            started_at_unix_ms,
            "run",
        )
        .await?;
        if run_report.summaries.iter().any(|summary| summary.successful_turns == 0) {
            return Err(Error::InvalidArgument(
                "one or more providers completed zero successful turns; inspect run.json and per-turn error logs"
                    .to_owned(),
            ));
        }
        return Ok(());
    }

    create_dir_all(&base_output_dir).await?;
    let mut depths = cli.depths.clone();
    depths.sort_unstable();
    depths.dedup();
    let mut depth_reports = Vec::with_capacity(depths.len());
    let mut any_zero_success = false;
    for depth in depths {
        let depth_dir = base_output_dir.join(format!("depth-{depth:05}"));
        let label = format!("depth={depth}");
        match run_pipeline(
            &cli,
            depth,
            depth_dir,
            dataset_questions.clone(),
            dataset_answers.clone(),
            started_at_unix_ms,
            &label,
        )
        .await
        {
            Ok(run_report) => {
                if run_report.summaries.iter().any(|summary| summary.successful_turns == 0) {
                    any_zero_success = true;
                    eprintln!("warning: [{label}] one or more providers completed zero successful turns");
                }
                depth_reports.push((depth, run_report));
            }
            Err(error) => {
                any_zero_success = true;
                eprintln!("warning: [{label}] failed: {error}");
            }
        }
    }

    let rows = report::depth_rollup(&depth_reports);
    let rollup_markdown = report::depth_rollup_markdown(&rows);
    let rollup_json = serde_json::to_vec_pretty(&rows).map_err(|source| Error::Io {
        path: base_output_dir.join("depth_summary.json"),
        source: std::io::Error::other(source),
    })?;
    tokio::fs::write(base_output_dir.join("depth_summary.md"), rollup_markdown)
        .await
        .map_err(|source| Error::Io {
            path: base_output_dir.clone(),
            source,
        })?;
    tokio::fs::write(base_output_dir.join("depth_summary.json"), rollup_json)
        .await
        .map_err(|source| Error::Io {
            path: base_output_dir.clone(),
            source,
        })?;
    eprintln!(
        "depth-scaling summary: {}",
        base_output_dir.join("depth_summary.md").display()
    );

    if any_zero_success {
        return Err(Error::InvalidArgument(
            "one or more depth runs completed zero successful turns for some provider; inspect each depth-*/run.json"
                .to_owned(),
        ));
    }
    Ok(())
}

/// Depth-batch mode in `main` calls this once per fixed depth so each depth is an independent,
/// fully-sampled run instead of a bucket sliced out of one long run.
#[allow(clippy::too_many_arguments)]
async fn run_pipeline(
    cli: &Cli,
    requests_per_session: usize,
    output_dir: PathBuf,
    dataset_questions: Option<PathBuf>,
    dataset_answers: Option<PathBuf>,
    started_at_unix_ms: u64,
    label: &str,
) -> Result<RunReport, Error> {
    let run_started = Instant::now();
    create_dir_all(&output_dir).await?;
    let output_dir = canonicalize(&output_dir).await?;
    let providers = selected_providers(cli);
    let generated = prompts::generate(&prompts::GenerationConfig {
        workload: cli.workload,
        seed: cli.seed,
        sessions: cli.sessions,
        turns: requests_per_session,
        transport_rounds: cli.transport_rounds,
        dataset_questions: dataset_questions.clone(),
        dataset_answers: dataset_answers.clone(),
        dataset_offset: cli.dataset_offset,
    })
    .await?;
    let flat_prompts = generated
        .iter()
        .flat_map(|session| session.prompts.iter().cloned())
        .collect::<Vec<_>>();

    let runner_config = Arc::new(RunnerConfig {
        model: cli.model.clone(),
        output_dir: output_dir.clone(),
        timeout: Duration::from_secs(cli.timeout_seconds),
        live_jsonl: cli.live_jsonl,
    });

    let task_count = providers.len().saturating_mul(cli.sessions);
    let start_barrier = Arc::new(Barrier::new(task_count.saturating_add(1)));
    let live_output_lock = Arc::new(Mutex::new(()));
    let mut tasks = JoinSet::new();
    for provider in &providers {
        for session in &generated {
            tasks.spawn(runner::run_session(
                Arc::clone(&runner_config),
                provider.clone(),
                session.session_index,
                session.prompts.clone(),
                Arc::clone(&start_barrier),
                Arc::clone(&live_output_lock),
            ));
        }
    }

    eprintln!(
        "[{label}] starting {task_count} concurrent Responses sessions ({} per provider, {} turns per session)",
        cli.sessions, requests_per_session
    );
    start_barrier.wait().await;
    let mut sessions: Vec<SessionResult> = Vec::with_capacity(task_count);
    while let Some(result) = tasks.join_next().await {
        sessions.push(result?);
    }
    sessions.sort_by(|left, right| {
        left.provider
            .cmp(&right.provider)
            .then(left.session_index.cmp(&right.session_index))
    });

    let summaries = report::summarize(&providers, &sessions, cli.sessions, requests_per_session);
    let comparison = report::compare(&sessions);
    let run_report = RunReport {
        schema_version: 3,
        started_at_unix_ms,
        elapsed_ms: millis(run_started.elapsed()),
        output_dir: output_dir.clone(),
        config: RunConfig {
            model: cli.model.clone(),
            workload: cli.workload,
            dataset_questions,
            dataset_answers,
            sessions_per_provider: cli.sessions,
            requests_per_session,
            seed: cli.seed,
            timeout_seconds: cli.timeout_seconds,
            providers,
        },
        prompts: flat_prompts,
        sessions,
        summaries,
        comparison,
        ttft_note: TTFT_NOTE.to_owned(),
        accuracy_note: report::ACCURACY_NOTE.to_owned(),
    };
    report::write_reports(&output_dir, &run_report)
        .await
        .map_err(|source| Error::Io {
            path: output_dir.clone(),
            source,
        })?;

    let summary = report::console_summary(&run_report.summaries, run_report.comparison.as_ref());
    if cli.live_jsonl {
        eprintln!("[{label}]\n{summary}");
    } else {
        println!("[{label}]\n{summary}");
    }
    eprintln!("[{label}] results: {}", output_dir.display());
    Ok(run_report)
}

fn validate(cli: &Cli) -> Result<(), Error> {
    if cli.model.trim().is_empty() {
        return Err(Error::InvalidArgument("--model must not be empty".to_owned()));
    }
    if cli.sessions == 0 {
        return Err(Error::InvalidArgument(
            "--sessions must be greater than zero".to_owned(),
        ));
    }
    if cli.requests_per_session() == 0 {
        return Err(Error::InvalidArgument(
            "--requests-per-session must be greater than zero".to_owned(),
        ));
    }
    if cli.timeout_seconds == 0 {
        return Err(Error::InvalidArgument(
            "--timeout-seconds must be greater than zero".to_owned(),
        ));
    }
    if cli.transport_rounds == 0 {
        return Err(Error::InvalidArgument(
            "--transport-rounds must be greater than zero".to_owned(),
        ));
    }
    if cli.workload == Workload::ToolCall && (cli.dataset_questions.is_none() || cli.dataset_answers.is_none()) {
        return Err(Error::InvalidArgument(
            "--workload tool-call requires --dataset-questions and --dataset-answers".to_owned(),
        ));
    }
    if !cli.depths.is_empty() {
        if cli.workload != Workload::HistoryRehydration {
            return Err(Error::InvalidArgument(
                "--depths is only supported for --workload history-rehydration".to_owned(),
            ));
        }
        if cli.depths.contains(&0) {
            return Err(Error::InvalidArgument(
                "--depths values must be greater than zero".to_owned(),
            ));
        }
    }
    Ok(())
}

fn selected_providers(cli: &Cli) -> Vec<ProviderSpec> {
    let agentic = ProviderSpec {
        id: "agentic-api".to_owned(),
        name: "agentic-api".to_owned(),
        base_url: cli.agentic_url.clone(),
        transport: Transport::Websocket,
        supports_websockets: true,
    };
    let agentic_http = ProviderSpec {
        id: "agentic-api-http".to_owned(),
        name: "agentic-api-http".to_owned(),
        base_url: cli.agentic_url.clone(),
        transport: Transport::HttpSse,
        supports_websockets: false,
    };
    let agentic_json = ProviderSpec {
        id: "agentic-api-json".to_owned(),
        name: "agentic-api-json".to_owned(),
        base_url: cli.agentic_url.clone(),
        transport: Transport::HttpJson,
        supports_websockets: false,
    };
    let vllm = ProviderSpec {
        id: "vllm".to_owned(),
        name: "vllm".to_owned(),
        base_url: cli.vllm_url.clone(),
        transport: Transport::HttpSse,
        supports_websockets: false,
    };
    let vllm_json = ProviderSpec {
        id: "vllm-json".to_owned(),
        name: "vllm-json".to_owned(),
        base_url: cli.vllm_url.clone(),
        transport: Transport::HttpJson,
        supports_websockets: false,
    };
    match cli.provider {
        ProviderSelection::All => vec![agentic, agentic_http, agentic_json, vllm, vllm_json],
        ProviderSelection::Both => vec![agentic, vllm],
        ProviderSelection::AgenticApi => vec![agentic],
        ProviderSelection::AgenticApiHttp => vec![agentic_http],
        ProviderSelection::AgenticApiJson => vec![agentic_json],
        ProviderSelection::Vllm => vec![vllm],
        ProviderSelection::VllmJson => vec![vllm_json],
    }
}

async fn canonicalize(path: &Path) -> Result<PathBuf, Error> {
    tokio::fs::canonicalize(path).await.map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}

async fn create_dir_all(path: &Path) -> Result<(), Error> {
    tokio::fs::create_dir_all(path).await.map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(millis)
        .unwrap_or_default()
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}
