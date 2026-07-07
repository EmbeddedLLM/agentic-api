use agentic_core::executor::accumulator::ResponseAccumulator;
use agentic_core::types::io::OutputItem;

mod support;

const MCP_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/mcp");
const READ_RESOURCE_URI: &str =
    "repo://crates/agentic-core/tests/cassettes/web_search/gpt_oss_web_search_nonstreaming.yaml";

fn load_mcp_cassette(filename: &str) -> support::Cassette {
    let path = format!("{MCP_DIR}/{filename}");
    support::load_cassette(&path)
}

fn extract_data_lines(sse_entries: &[String]) -> Vec<String> {
    sse_entries
        .iter()
        .flat_map(|entry| entry.lines())
        .filter(|line| line.starts_with("data: "))
        .map(ToString::to_string)
        .collect()
}

fn count_function_calls(output: &[OutputItem]) -> usize {
    output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .count()
}

fn has_reasoning(output: &[OutputItem]) -> bool {
    output.iter().any(|item| matches!(item, OutputItem::Reasoning(_)))
}

fn process_nonstreaming_turn(cassette: &support::Cassette, turn_idx: usize, model: &str) -> Vec<OutputItem> {
    let body = cassette.turns[turn_idx]
        .response
        .body
        .as_ref()
        .unwrap_or_else(|| panic!("turn {} must have response body", turn_idx + 1));
    let body_str = serde_json::to_string(body).unwrap();
    let acc = ResponseAccumulator::from_json(&body_str, None).unwrap();
    let payload = acc.finalize(model, None, None);
    assert_eq!(payload.status, "completed");
    payload.output
}

fn process_streaming_turn(cassette: &support::Cassette, turn_idx: usize, model: &str) -> Vec<OutputItem> {
    let sse = cassette.turns[turn_idx]
        .response
        .sse
        .as_ref()
        .unwrap_or_else(|| panic!("turn {} must have SSE events", turn_idx + 1));
    let data_lines = extract_data_lines(sse);
    assert!(
        !data_lines.is_empty(),
        "streaming turn {} must have SSE data lines",
        turn_idx + 1
    );
    let final_payload = data_lines
        .iter()
        .rev()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .find(|event| event["status"].as_str() == Some("completed") && event["output"].is_array())
        .unwrap_or_else(|| panic!("turn {} must include a completed final response payload", turn_idx + 1));
    let final_payload = serde_json::to_string(&final_payload).unwrap();
    let acc = ResponseAccumulator::from_json(&final_payload, None).unwrap();
    let payload = acc.finalize(model, None, None);
    assert_eq!(payload.status, "completed");
    payload.output
}

fn assert_completed_read_mcp_resource(output: &[OutputItem]) {
    assert_eq!(
        count_function_calls(output),
        0,
        "raw read_mcp_resource function call should not leak"
    );
    let mcp_call = output
        .iter()
        .find_map(|item| match item {
            OutputItem::McpToolCall(call) if call.status.as_str() == "completed" => Some(call),
            _ => None,
        })
        .expect("cassette output should include a completed mcp_tool_call item");
    assert_eq!(mcp_call.status.as_str(), "completed");
    assert_eq!(mcp_call.server, "repo");
    assert_eq!(mcp_call.tool, "read_mcp_resource");
    assert_eq!(mcp_call.arguments["server"], "repo");
    assert_eq!(mcp_call.arguments["uri"], READ_RESOURCE_URI);

    let result = mcp_call.result.as_ref().expect("mcp tool call should include a result");
    let text = result["contents"][0]["text"]
        .as_str()
        .expect("read_mcp_resource result should include text content");
    assert!(text.contains("web_search_preview"));
    assert!(text.contains("Potato - Wikipedia"));
    assert!(
        output.iter().any(|item| matches!(item, OutputItem::Message(_))),
        "cassette output should include a final assistant message"
    );
}

#[test]
fn read_mcp_resource_cassette_nonstreaming() {
    let cassette = load_mcp_cassette("mcp-read-resource-Qwen-Qwen3-30B-A3B-FP8-nonstreaming.yaml");
    assert_eq!(cassette.turns.len(), 1);
    let body = &cassette.turns[0].request.body;
    let tool = &body.tools[0];
    assert_eq!(tool["type"].as_str().unwrap(), "mcp");
    assert_eq!(tool["name"].as_str().unwrap(), "read_mcp_resource");
    assert_eq!(tool["server_label"].as_str().unwrap(), "repo");
    assert_eq!(tool["server_url"].as_str().unwrap(), "http://127.0.0.1:8000/mcp");
    assert_eq!(body.tool_choice.as_ref().unwrap().as_str().unwrap(), "required");

    let output = process_nonstreaming_turn(&cassette, 0, "Qwen/Qwen3-30B-A3B-FP8");
    assert!(
        has_reasoning(&output),
        "mcp read_resource cassette should include reasoning"
    );
    assert_completed_read_mcp_resource(&output);
}

#[test]
fn read_mcp_resource_cassette_streaming_success_events() {
    let cassette = load_mcp_cassette("mcp-read-resource-Qwen-Qwen3-30B-A3B-FP8-streaming.yaml");
    assert_eq!(cassette.turns.len(), 1);
    let body = &cassette.turns[0].request.body;
    assert_eq!(body.tools[0]["type"].as_str().unwrap(), "mcp");

    let sse = cassette.turns[0]
        .response
        .sse
        .as_ref()
        .expect("streaming cassette should include SSE events");
    let data_lines = extract_data_lines(sse);
    let events: Vec<serde_json::Value> = data_lines
        .iter()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect();
    let event_types: Vec<&str> = events.iter().filter_map(|event| event["type"].as_str()).collect();
    assert!(
        !event_types.contains(&"response.error") && !event_types.contains(&"response.failed"),
        "streaming MCP cassette must record a successful tool loop"
    );
    assert!(event_types.contains(&"response.output_item.added"));
    assert!(event_types.contains(&"response.mcp_tool_call.in_progress"));
    assert!(event_types.contains(&"response.mcp_tool_call.completed"));
    assert!(event_types.contains(&"response.output_item.done"));
    assert!(
        events
            .iter()
            .any(|event| event["status"].as_str() == Some("completed") && event["output"].is_array()),
        "streaming MCP cassette should include a completed final response payload"
    );

    let output = process_streaming_turn(&cassette, 0, "Qwen/Qwen3-30B-A3B-FP8");
    assert_completed_read_mcp_resource(&output);
}
