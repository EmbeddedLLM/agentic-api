//! Cassette replay tests for NVIDIA Dynamo as the inference upstream.
//!
//! The recordings in `tests/cassettes/dynamo` were captured from a Dynamo
//! frontend (`python -m dynamo.frontend`) fronting a `dynamo.vllm` worker
//! serving `openai/gpt-oss-20b`. Dynamo speaks the same `/v1/responses` wire
//! format as vLLM but is stateless: it rejects `previous_response_id` with
//! `501 Not Implemented`. These tests pin the gateway behavior that makes
//! Dynamo usable as an upstream anyway — the gateway owns the conversation
//! state and sends the fully rehydrated item history on every turn. The
//! function-call cassettes cover a client-executed function tool.
//!
//! Re-record with `tests/cassettes/record_dynamo_cassettes.sh`.

mod support;

use agentic_core::executor::execute;
use agentic_core::types::io::OutputItem;
use agentic_core::types::request_response::ResponsePayload;
use agentic_core::types::tools::ResponsesTool;
use serde_json::Value;
use std::sync::Arc;
use support::{
    TestFixture, Turn, collect_stream, expected_text, load_cassette, make_request, output_text, request_input_texts,
    unwrap_blocking,
};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/dynamo");
const MODEL_SLUG: &str = "openai-gpt-oss-20b";

const TURN1_PROMPT: &str = "Remember the word APPLE. Just say: OK";
const TURN2_PROMPT: &str = "What word did I ask you to remember? Reply with just the word.";
const TOOL_PROMPT: &str = "What is the current NVIDIA stock price? Use the tool.";

fn cassette_path(name: &str, streaming: bool) -> String {
    let suffix = if streaming { "streaming" } else { "nonstreaming" };
    format!("{DIR}/{name}-{MODEL_SLUG}-{suffix}.yaml")
}

fn turn2_prompt_from(turn: &Turn) -> String {
    request_input_texts(&serde_json::json!({ "input": turn.request.body.input }))
        .pop()
        .expect("turn 2 recording ends with the user prompt")
}

fn function_calls(payload: &ResponsePayload) -> Vec<(String, String)> {
    payload
        .output
        .iter()
        .filter_map(|item| match item {
            OutputItem::FunctionCall(fc) => Some((fc.name.clone(), fc.arguments.clone())),
            _ => None,
        })
        .collect()
}

fn first_message_id(payload: &ResponsePayload) -> &str {
    payload
        .output
        .iter()
        .find_map(|item| match item {
            OutputItem::Message(msg) => Some(msg.id.as_str()),
            _ => None,
        })
        .expect("turn 1 output contains an assistant message")
}

/// The upstream request for the second turn must carry the rehydrated item
/// history instead of `previous_response_id`, because Dynamo rejects the latter.
/// The history is compared structurally with the recorded turn-2 request; only
/// the assistant message id differs per run, so it is taken from turn 1's payload.
fn assert_upstream_requests_are_stateless(requests: &[Value], t2: &Turn, p1: &ResponsePayload) {
    assert_eq!(requests.len(), 2, "one upstream call per turn");
    for request in requests {
        assert!(
            request.get("previous_response_id").is_none(),
            "Dynamo returns 501 for previous_response_id, even when null; upstream request was {request}"
        );
    }
    assert_eq!(request_input_texts(&requests[0]), vec![TURN1_PROMPT]);

    let mut expected_history = t2.request.body.input.clone();
    let recorded_assistant = &mut expected_history[1];
    assert_eq!(
        recorded_assistant["role"], "assistant",
        "recorded turn 2 replays the assistant item"
    );
    recorded_assistant["id"] = Value::String(first_message_id(p1).to_owned());
    assert_eq!(
        requests[1]["input"], expected_history,
        "turn 2 must replay the full item history to the stateless upstream"
    );
}

async fn run_stateful_two_turn(streaming: bool) {
    let cassette = load_cassette(&cassette_path("dynamo-stateful", streaming));
    let (t1, t2) = (&cassette.turns[0], &cassette.turns[1]);
    assert_eq!(turn2_prompt_from(t2), TURN2_PROMPT);
    let fixture = TestFixture::new(&[t1, t2]).await;

    let first = execute(
        make_request(TURN1_PROMPT, true, streaming, None, None),
        Arc::clone(&fixture.exec_ctx),
    )
    .await
    .expect("t1");
    let p1 = if streaming {
        collect_stream(first).await
    } else {
        unwrap_blocking(first)
    };
    let second = execute(
        make_request(TURN2_PROMPT, true, streaming, Some(p1.id.clone()), None),
        Arc::clone(&fixture.exec_ctx),
    )
    .await
    .expect("t2");
    let p2 = if streaming {
        collect_stream(second).await
    } else {
        unwrap_blocking(second)
    };

    assert_eq!(p1.status, "completed");
    assert_eq!(output_text(&p1), expected_text(t1));
    assert_eq!(output_text(&p1), "OK");
    assert_ne!(p2.id, p1.id);
    assert_eq!(p2.status, "completed");
    assert_eq!(p2.previous_response_id.as_deref(), Some(p1.id.as_str()));
    assert_eq!(output_text(&p2), expected_text(t2));
    assert_eq!(output_text(&p2), "APPLE");

    assert_upstream_requests_are_stateless(&fixture.request_bodies().await, t2, &p1);
}

#[tokio::test]
async fn dynamo_stateful_two_turn_nonstreaming() {
    run_stateful_two_turn(false).await;
}

#[tokio::test]
async fn dynamo_stateful_two_turn_streaming() {
    run_stateful_two_turn(true).await;
}

async fn run_function_tool_call(streaming: bool) {
    let cassette = load_cassette(&cassette_path("dynamo-tool-call-auto", streaming));
    let t1 = &cassette.turns[0];
    let tools: Vec<ResponsesTool> =
        serde_json::from_value(Value::Array(t1.request.body.tools.clone())).expect("recorded tools parse");
    let fixture = TestFixture::new(&[t1]).await;

    let mut request = make_request(TOOL_PROMPT, true, streaming, None, None);
    request.tools = Some(tools);
    let result = execute(request, Arc::clone(&fixture.exec_ctx)).await.expect("t1");
    let payload = if streaming {
        collect_stream(result).await
    } else {
        unwrap_blocking(result)
    };

    assert_eq!(payload.status, "completed");
    let calls = function_calls(&payload);
    assert_eq!(
        calls.len(),
        1,
        "Dynamo's harmony parser yields one function_call: {calls:?}"
    );
    let (name, arguments) = &calls[0];
    assert_eq!(name, "get_stock_price");
    let arguments: Value = serde_json::from_str(arguments).expect("arguments are JSON");
    assert_eq!(arguments["ticker"], "NVDA");

    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 1, "client-executed function tools take one model call");
    let upstream_tools = requests[0]["tools"].as_array().expect("tools forwarded upstream");
    assert!(
        upstream_tools.iter().any(|tool| tool["name"] == "get_stock_price"),
        "function declarations must reach Dynamo unchanged"
    );
}

#[tokio::test]
async fn dynamo_function_tool_call_nonstreaming() {
    run_function_tool_call(false).await;
}

#[tokio::test]
async fn dynamo_function_tool_call_streaming() {
    run_function_tool_call(true).await;
}
