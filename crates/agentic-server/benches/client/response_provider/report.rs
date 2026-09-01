use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::types::{
    Comparison, DepthSummaryRow, Distribution, ProviderSpec, ProviderSummary, RunReport, SessionResult, Transport,
    TurnResult, Workload,
};

pub const ACCURACY_NOTE: &str = "Task correctness and tool-call accuracy are correctness/regression checks against \
the same underlying model, not a performance comparison: a gateway that owns conversation state is expected to \
present the model with equivalent context, so its accuracy should match direct vLLM. A gap that widens with turn \
depth indicates a rehydration bug, not a capability advantage.";

pub fn summarize(
    providers: &[ProviderSpec],
    sessions: &[SessionResult],
    sessions_per_provider: usize,
    requests_per_session: usize,
) -> Vec<ProviderSummary> {
    providers
        .iter()
        .map(|provider| summarize_provider(provider, sessions, sessions_per_provider, requests_per_session))
        .collect()
}

fn summarize_provider(
    provider: &ProviderSpec,
    sessions: &[SessionResult],
    sessions_per_provider: usize,
    requests_per_session: usize,
) -> ProviderSummary {
    let provider_sessions: Vec<&SessionResult> = sessions
        .iter()
        .filter(|session| session.provider == provider.id)
        .collect();
    let turns: Vec<&TurnResult> = provider_sessions
        .iter()
        .flat_map(|session| session.turns.iter())
        .collect();
    let successful: Vec<&TurnResult> = turns.iter().copied().filter(|turn| turn.success).collect();
    let attempted_turns = turns.iter().filter(|turn| turn.attempted).count();
    let successful_turns = successful.len();
    let timed_out_turns = turns.iter().filter(|turn| turn.timed_out).count();
    let transport_fallback_turns = turns.iter().filter(|turn| turn.transport_fallback).count();
    let tool_compliant_turns = turns
        .iter()
        .filter(|turn| turn.tool_calls_completed > 0 && turn.tool_calls_failed == 0)
        .count();
    let task_correct_turns = turns.iter().filter(|turn| turn.task_correct).count();
    let tool_call_correct_turns = turns.iter().filter(|turn| turn.tool_call_correct).count();
    let planned_turns = sessions_per_provider.saturating_mul(requests_per_session);
    let provider_wall_clock_ms = provider_sessions
        .iter()
        .map(|session| session.elapsed_ms)
        .max()
        .unwrap_or_default();
    let denominator = usize_as_f64(planned_turns.max(1));

    let total_turn_input_tokens = successful
        .iter()
        .filter_map(|turn| turn.turn_usage.as_ref())
        .map(|usage| usage.input_tokens)
        .sum();
    let total_turn_cached_input_tokens = successful
        .iter()
        .filter_map(|turn| turn.turn_usage.as_ref())
        .map(|usage| usage.cached_input_tokens)
        .sum();
    let total_turn_output_tokens: i64 = successful
        .iter()
        .filter_map(|turn| turn.turn_usage.as_ref())
        .map(|usage| usage.output_tokens)
        .sum();
    let total_turn_reasoning_output_tokens = successful
        .iter()
        .filter_map(|turn| turn.turn_usage.as_ref())
        .map(|usage| usage.reasoning_output_tokens)
        .sum();
    let total_latency_ms: u64 = successful.iter().map(|turn| turn.end_to_end_latency_ms).sum();
    let aggregate_effective_output_tokens_per_second =
        (total_latency_ms > 0).then_some(i64_as_f64(total_turn_output_tokens) * 1_000.0 / u64_as_f64(total_latency_ms));

    ProviderSummary {
        provider: provider.id.clone(),
        transport: provider.transport,
        planned_turns,
        attempted_turns,
        successful_turns,
        timed_out_turns,
        transport_fallback_turns,
        tool_compliant_turns,
        task_correct_turns,
        tool_call_correct_turns,
        success_rate: usize_as_f64(successful_turns) / denominator,
        tool_compliance_rate: usize_as_f64(tool_compliant_turns) / denominator,
        task_correctness_rate: usize_as_f64(task_correct_turns) / denominator,
        tool_call_accuracy: usize_as_f64(tool_call_correct_turns) / denominator,
        provider_wall_clock_ms,
        successful_turns_per_second: if provider_wall_clock_ms == 0 {
            0.0
        } else {
            usize_as_f64(successful_turns) * 1_000.0 / u64_as_f64(provider_wall_clock_ms)
        },
        end_to_end_latency_ms: distribution(successful.iter().map(|turn| u64_as_f64(turn.end_to_end_latency_ms))),
        time_to_first_output_event_ms: distribution(
            successful
                .iter()
                .filter_map(|turn| turn.time_to_first_output_event_ms.map(u64_as_f64)),
        ),
        ttft_ms: distribution(successful.iter().filter_map(|turn| turn.ttft_ms.map(u64_as_f64))),
        time_to_first_tool_call_ms: distribution(
            successful
                .iter()
                .filter_map(|turn| turn.time_to_first_tool_call_ms.map(u64_as_f64)),
        ),
        mean_tool_duration_ms: distribution(successful.iter().filter_map(|turn| turn.mean_tool_duration_ms)),
        continuation_round_latency_ms: distribution(
            successful
                .iter()
                .flat_map(|turn| turn.continuation_round_latencies_ms.iter().copied().map(u64_as_f64)),
        ),
        request_bytes: distribution(successful.iter().map(|turn| u64_as_f64(turn.request_bytes))),
        response_bytes: distribution(successful.iter().map(|turn| u64_as_f64(turn.response_bytes))),
        total_turn_input_tokens,
        total_turn_cached_input_tokens,
        total_turn_output_tokens,
        total_turn_reasoning_output_tokens,
        aggregate_effective_output_tokens_per_second,
    }
}

pub fn compare(sessions: &[SessionResult]) -> Option<Comparison> {
    let turns: Vec<&TurnResult> = sessions.iter().flat_map(|session| session.turns.iter()).collect();
    let mut agentic = HashMap::new();
    let mut vllm = HashMap::new();
    for turn in turns.into_iter().filter(|turn| turn.success) {
        let key = (turn.session_index, turn.turn_index);
        match turn.provider.as_str() {
            "agentic-api" => {
                agentic.insert(key, turn);
            }
            "vllm" => {
                vllm.insert(key, turn);
            }
            _ => {}
        }
    }

    let mut latency_deltas = Vec::new();
    let mut latency_ratios = Vec::new();
    let mut first_output_deltas = Vec::new();
    for (key, agentic_turn) in agentic {
        let Some(vllm_turn) = vllm.get(&key) else {
            continue;
        };
        latency_deltas
            .push(u64_as_f64(agentic_turn.end_to_end_latency_ms) - u64_as_f64(vllm_turn.end_to_end_latency_ms));
        if vllm_turn.end_to_end_latency_ms > 0 {
            latency_ratios
                .push(u64_as_f64(agentic_turn.end_to_end_latency_ms) / u64_as_f64(vllm_turn.end_to_end_latency_ms));
        }
        if let (Some(agentic_first), Some(vllm_first)) = (
            agentic_turn.time_to_first_output_event_ms,
            vllm_turn.time_to_first_output_event_ms,
        ) {
            first_output_deltas.push(u64_as_f64(agentic_first) - u64_as_f64(vllm_first));
        }
    }
    if latency_deltas.is_empty() {
        return None;
    }

    Some(Comparison {
        paired_successful_turns: latency_deltas.len(),
        median_agentic_minus_vllm_latency_ms: median(latency_deltas),
        median_agentic_over_vllm_latency_ratio: median(latency_ratios),
        median_agentic_minus_vllm_first_output_ms: median(first_output_deltas),
    })
}

pub async fn write_reports(output_dir: &Path, report: &RunReport) -> Result<(), std::io::Error> {
    let json = serde_json::to_vec_pretty(report).map_err(std::io::Error::other)?;
    tokio::fs::write(output_dir.join("run.json"), json).await?;
    tokio::fs::write(output_dir.join("turns.csv"), turns_csv(&report.sessions)).await?;
    tokio::fs::write(output_dir.join("summary.md"), markdown_summary(report)).await
}

pub fn console_summary(summaries: &[ProviderSummary], comparison: Option<&Comparison>) -> String {
    let mut output = String::new();
    output.push_str("provider          success  task correct  tool accuracy  p50 latency  p50 continuation  turns/s\n");
    for summary in summaries {
        let _ = writeln!(
            output,
            "{:<17} {:>6.1}%  {:>10.1}%  {:>10.1}%  {:>9} ms  {:>16} ms  {:>7.2}",
            summary.provider,
            summary.success_rate * 100.0,
            summary.task_correctness_rate * 100.0,
            summary.tool_call_accuracy * 100.0,
            optional_number(summary.end_to_end_latency_ms.p50),
            optional_number(summary.continuation_round_latency_ms.p50),
            summary.successful_turns_per_second,
        );
    }
    if let Some(comparison) = comparison {
        let _ = writeln!(
            output,
            "paired turns: {}; median Agentic-vLLM latency: {} ms; median ratio: {}",
            comparison.paired_successful_turns,
            optional_number(comparison.median_agentic_minus_vllm_latency_ms),
            optional_decimal(comparison.median_agentic_over_vllm_latency_ratio),
        );
    }
    output
}

fn markdown_summary(report: &RunReport) -> String {
    let mut output = format!(
        "# Responses provider benchmark: {}\n\n\
         Streaming TTFT is measured at the first `response.output_text.delta`; JSON runs report it as `n/a`. \
         Request/response bytes are the client-visible wire payload per turn: flat across turn depth for a provider \
         that rehydrates history server-side, growing for a provider the client must resend full history to.\n\n\
         | Provider | Transport | Success | Task correctness | Tool-call accuracy | p50 latency (ms) | p95 latency (ms) | p50 continuation round (ms) | p50 first output (ms) | p50 TTFT (ms) | Turns/s | p50 request bytes | p50 response bytes |\n\
         | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
        workload_name(report.config.workload),
    );
    for summary in &report.summaries {
        let _ = writeln!(
            output,
            "| {} | {} | {:.1}% | {:.1}% | {:.1}% | {} | {} | {} | {} | {} | {:.2} | {} | {} |",
            summary.provider,
            transport_name(summary.transport),
            summary.success_rate * 100.0,
            summary.task_correctness_rate * 100.0,
            summary.tool_call_accuracy * 100.0,
            optional_number(summary.end_to_end_latency_ms.p50),
            optional_number(summary.end_to_end_latency_ms.p95),
            optional_number(summary.continuation_round_latency_ms.p50),
            optional_number(summary.time_to_first_output_event_ms.p50),
            optional_number(summary.ttft_ms.p50),
            summary.successful_turns_per_second,
            optional_number(summary.request_bytes.p50),
            optional_number(summary.response_bytes.p50),
        );
    }
    if let Some(comparison) = &report.comparison {
        let _ = writeln!(
            output,
            "\nPaired successful turns: {}. Median Agentic API minus vLLM latency: {} ms. Median Agentic API / vLLM latency ratio: {}.",
            comparison.paired_successful_turns,
            optional_number(comparison.median_agentic_minus_vllm_latency_ms),
            optional_decimal(comparison.median_agentic_over_vllm_latency_ratio),
        );
    }
    let _ = writeln!(output, "\n> **Correctness, not competition.** {}", report.accuracy_note);
    if report.config.workload == Workload::ToolCall {
        output.push_str("\n## Tool-call compatibility (pass/fail by case)\n\n");
        output.push_str(&tool_call_matrix(&report.summaries, &report.sessions));
    }
    output
}

/// Per-BFCL-case pass/fail across providers, in place of a single aggregate accuracy percentage.
/// This is the tool-shape compatibility view: whether the gateway preserved each individual
/// tool-call case through translation, not how often it happens to be right on average.
fn tool_call_matrix(summaries: &[ProviderSummary], sessions: &[SessionResult]) -> String {
    let providers: Vec<&str> = summaries.iter().map(|summary| summary.provider.as_str()).collect();
    let mut cases: Vec<(String, HashMap<&str, bool>)> = Vec::new();
    let mut index_by_case: HashMap<String, usize> = HashMap::new();
    for turn in sessions.iter().flat_map(|session| &session.turns) {
        let case_id = turn.source_id.clone().unwrap_or_else(|| turn.prompt_id.clone());
        let index = *index_by_case.entry(case_id.clone()).or_insert_with(|| {
            cases.push((case_id, HashMap::new()));
            cases.len() - 1
        });
        cases[index].1.insert(turn.provider.as_str(), turn.tool_call_correct);
    }
    cases.sort_by(|left, right| left.0.cmp(&right.0));

    let mut output = format!("| Case | {} |\n", providers.join(" | "));
    let _ = writeln!(output, "| --- |{}", " ---: |".repeat(providers.len()));
    for (case_id, results) in &cases {
        let cells = providers
            .iter()
            .map(|provider| match results.get(provider) {
                Some(true) => "✅",
                Some(false) => "❌",
                None => "n/a",
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let _ = writeln!(output, "| {case_id} | {cells} |");
    }
    output
}

/// Roll up independently-run fixed-depth session batches into one table per provider, keyed by
/// turn depth. Each depth is its own complete batch of sessions rather than a bucket sliced out of
/// one long run, so sample counts stay even across depths instead of thinning out at the tail.
#[must_use]
pub fn depth_rollup(depth_reports: &[(usize, RunReport)]) -> Vec<DepthSummaryRow> {
    let mut rows = Vec::new();
    for (depth, report) in depth_reports {
        for summary in &report.summaries {
            rows.push(DepthSummaryRow {
                depth: *depth,
                provider: summary.provider.clone(),
                transport: summary.transport,
                sessions: report.config.sessions_per_provider,
                success_rate: summary.success_rate,
                task_correctness_rate: summary.task_correctness_rate,
                p50_latency_ms: summary.end_to_end_latency_ms.p50,
                p50_request_bytes: summary.request_bytes.p50,
                p50_response_bytes: summary.response_bytes.p50,
                total_request_bytes: sum_u64(&summary.provider, &report.sessions, |turn| turn.request_bytes),
                total_response_bytes: sum_u64(&summary.provider, &report.sessions, |turn| turn.response_bytes),
            });
        }
    }
    rows.sort_by(|left, right| left.provider.cmp(&right.provider).then(left.depth.cmp(&right.depth)));
    rows
}

fn sum_u64(provider: &str, sessions: &[SessionResult], field: impl Fn(&TurnResult) -> u64) -> u64 {
    sessions
        .iter()
        .filter(|session| session.provider == provider)
        .flat_map(|session| session.turns.iter())
        .filter(|turn| turn.success)
        .map(field)
        .sum()
}

#[must_use]
pub fn depth_rollup_markdown(rows: &[DepthSummaryRow]) -> String {
    let mut output = String::from(
        "# State-scaling: request bytes and accuracy versus turn depth\n\n\
         Each depth below ran as its own independent batch of sessions (not a bucket sliced out of one long run), \
         so sample counts are even across depths instead of thinning out at the tail. Accuracy is a \
         correctness/regression check, not a competition: it should not diverge between providers, and a widening \
         gap at deeper turns points to a rehydration bug.\n\n\
         | Provider | Depth | Sessions | Success | Task correctness | p50 latency (ms) | p50 request bytes | p50 response bytes | total request bytes | total response bytes |\n\
         | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for row in rows {
        let _ = writeln!(
            output,
            "| {} | {} | {} | {:.1}% | {:.1}% | {} | {} | {} | {} | {} |",
            row.provider,
            row.depth,
            row.sessions,
            row.success_rate * 100.0,
            row.task_correctness_rate * 100.0,
            optional_number(row.p50_latency_ms),
            optional_number(row.p50_request_bytes),
            optional_number(row.p50_response_bytes),
            row.total_request_bytes,
            row.total_response_bytes,
        );
    }
    output
}

fn turns_csv(sessions: &[SessionResult]) -> String {
    let mut output = String::from(
        "provider,transport,workload,session,turn,prompt_id,source_id,success,task_correct,tool_call_correct,timed_out,transport_fallback,latency_ms,first_output_ms,ttft_ms,first_tool_ms,initial_model_round_ms,continuation_round_latencies_ms,tool_duration_ms,tool_calls_completed,tool_calls_failed,observed_tool_calls,expected_tool_calls,request_bytes,response_bytes,input_tokens,cached_input_tokens,output_tokens,reasoning_output_tokens,output_tokens_per_second,response_id,raw_jsonl_path,error_log_path,errors\n",
    );
    for turn in sessions.iter().flat_map(|session| &session.turns) {
        let usage = turn.turn_usage.as_ref();
        let fields = [
            turn.provider.clone(),
            transport_name(turn.transport).to_owned(),
            workload_name(turn.workload).to_owned(),
            turn.session_index.to_string(),
            turn.turn_index.to_string(),
            turn.prompt_id.clone(),
            turn.source_id.clone().unwrap_or_default(),
            turn.success.to_string(),
            turn.task_correct.to_string(),
            turn.tool_call_correct.to_string(),
            turn.timed_out.to_string(),
            turn.transport_fallback.to_string(),
            turn.end_to_end_latency_ms.to_string(),
            optional_u64(turn.time_to_first_output_event_ms),
            optional_u64(turn.ttft_ms),
            optional_u64(turn.time_to_first_tool_call_ms),
            optional_u64(turn.initial_model_round_ms),
            serde_json::to_string(&turn.continuation_round_latencies_ms).unwrap_or_default(),
            optional_decimal(turn.mean_tool_duration_ms),
            turn.tool_calls_completed.to_string(),
            turn.tool_calls_failed.to_string(),
            serde_json::to_string(&turn.observed_tool_calls).unwrap_or_default(),
            serde_json::to_string(&turn.expected_tool_calls).unwrap_or_default(),
            turn.request_bytes.to_string(),
            turn.response_bytes.to_string(),
            usage.map_or_else(String::new, |value| value.input_tokens.to_string()),
            usage.map_or_else(String::new, |value| value.cached_input_tokens.to_string()),
            usage.map_or_else(String::new, |value| value.output_tokens.to_string()),
            usage.map_or_else(String::new, |value| value.reasoning_output_tokens.to_string()),
            optional_decimal(turn.effective_output_tokens_per_second),
            turn.response_id.clone().unwrap_or_default(),
            turn.raw_jsonl_path.clone(),
            turn.error_log_path.clone(),
            turn.errors.join(" | "),
        ];
        output.push_str(&fields.map(|field| csv_escape(&field)).join(","));
        output.push('\n');
    }
    output
}

fn distribution(values: impl Iterator<Item = f64>) -> Distribution {
    let mut values: Vec<f64> = values.filter(|value| value.is_finite()).collect();
    values.sort_by(f64::total_cmp);
    Distribution {
        count: values.len(),
        mean: (!values.is_empty()).then(|| values.iter().sum::<f64>() / usize_as_f64(values.len())),
        p50: percentile(&values, 0.50),
        p95: percentile(&values, 0.95),
        p99: percentile(&values, 0.99),
        min: values.first().copied(),
        max: values.last().copied(),
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn percentile(sorted: &[f64], quantile: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = quantile * usize_as_f64(sorted.len().saturating_sub(1));
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - usize_as_f64(lower);
    Some(sorted[lower] + (sorted[upper] - sorted[lower]) * fraction)
}

fn median(values: Vec<f64>) -> Option<f64> {
    let mut values: Vec<f64> = values.into_iter().filter(|value| value.is_finite()).collect();
    values.sort_by(f64::total_cmp);
    percentile(&values, 0.5)
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn transport_name(transport: Transport) -> &'static str {
    match transport {
        Transport::Websocket => "responses_websocket",
        Transport::HttpSse => "responses_http_sse",
        Transport::HttpJson => "responses_http_json",
    }
}

fn workload_name(workload: crate::types::Workload) -> &'static str {
    match workload {
        crate::types::Workload::Transport => "transport",
        crate::types::Workload::ToolCall => "tool_call",
        crate::types::Workload::HistoryRehydration => "history_rehydration",
    }
}

fn optional_number(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |number| format!("{number:.0}"))
}

fn optional_decimal(value: Option<f64>) -> String {
    value.map_or_else(String::new, |number| format!("{number:.3}"))
}

fn optional_u64(value: Option<u64>) -> String {
    value.map_or_else(String::new, |number| number.to_string())
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
