use agentic_core::executor::accumulator::ResponseAccumulator;
use agentic_core::tool::{GatewayExecutors, ToolRegistry, ToolType};
use agentic_core::types::io::{McpCall, OutputItem};
use agentic_core::types::tools::ResponsesTool;
use serde_json::{Value, json};
use std::collections::HashMap;

mod support;

const MODEL: &str = "Qwen/Qwen3.5-35B-A3B-FP8";
const MCP_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/mcp");
const GATEWAY_MODEL_SLUG: &str = "Qwen-Qwen3.5-35B-A3B-FP8";
const OPENAI_MODEL_SLUG: &str = "gpt-4o";

fn load_mcp_cassette(filename: &str) -> support::Cassette {
    support::load_cassette(&format!("{MCP_DIR}/{filename}"))
}

fn load_scenario_pair(scenario: &str, streaming: bool) -> (support::Cassette, support::Cassette) {
    let mode = if streaming { "streaming" } else { "nonstreaming" };
    let openai = load_mcp_cassette(&format!(
        "mcp-openai-reference-counter-{scenario}-{OPENAI_MODEL_SLUG}-{mode}.yaml"
    ));
    let gateway = load_mcp_cassette(&format!(
        "mcp-gateway-counter-{scenario}-{GATEWAY_MODEL_SLUG}-{mode}.yaml"
    ));
    (openai, gateway)
}

fn native_mcp_declaration() -> ResponsesTool {
    serde_json::from_value(serde_json::json!({
        "type": "mcp",
        "server_label": "counter",
        "server_url": "http://127.0.0.1:8000/mcp",
        "allowed_tools": ["increment"],
        "require_approval": "never"
    }))
    .expect("native MCP declaration")
}

#[test]
fn native_mcp_declaration_uses_server_identity_without_a_tool_name() {
    let ResponsesTool::Mcp(param) = native_mcp_declaration() else {
        panic!("expected MCP declaration");
    };

    assert_eq!(param.server_label, "counter");
    assert_eq!(param.server_url.as_deref(), Some("http://127.0.0.1:8000/mcp"));
    assert_eq!(
        param.allowed_tools.as_deref(),
        Some(["increment".to_owned()].as_slice())
    );
    assert_eq!(param.require_approval.as_deref(), Some("never"));
}

#[test]
fn native_mcp_declaration_ignores_a_client_supplied_tool_name() {
    let tool = serde_json::from_value::<ResponsesTool>(serde_json::json!({
        "type": "mcp",
        "name": "increment",
        "server_label": "counter",
        "server_url": "http://127.0.0.1:8000/mcp"
    }))
    .expect("MCP declaration with an unknown field");

    let serialized = serde_json::to_value(tool).expect("serialized MCP declaration");
    assert_eq!(serialized["server_label"], "counter");
    assert!(serialized.get("name").is_none());
}

#[tokio::test]
async fn read_mcp_resource_function_is_client_owned() {
    let mut tools = vec![
        serde_json::from_value::<ResponsesTool>(serde_json::json!({
            "type": "function",
            "name": "read_mcp_resource",
            "description": "A client-owned function with no gateway MCP semantics",
            "parameters": {"type": "object"},
            "metadata": {
                "server_label": "repo",
                "server_url": "http://127.0.0.1:8000/mcp"
            }
        }))
        .expect("function declaration"),
    ];
    let mut executors = GatewayExecutors::default();

    let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
        .await
        .expect("function registry");
    let entry = registry.lookup("read_mcp_resource").expect("function registry entry");

    assert_eq!(entry.tool_type, ToolType::Function);
    assert!(entry.handler.is_none());
}

fn assert_matching_native_mcp_requests(
    openai: &support::Cassette,
    gateway: &support::Cassette,
    streaming: bool,
    allowed_tools: Option<&[&str]>,
) {
    assert_eq!(openai.turns.len(), 1);
    assert_eq!(gateway.turns.len(), 1);

    let openai_request = &openai.turns[0].request;
    let gateway_request = &gateway.turns[0].request;
    let openai_server_url = openai_request.body.tools[0]["server_url"]
        .as_str()
        .expect("OpenAI MCP server_url");
    let gateway_server_url = gateway_request.body.tools[0]["server_url"]
        .as_str()
        .expect("gateway MCP server_url");
    for (provider, server_url) in [("OpenAI", openai_server_url), ("gateway", gateway_server_url)] {
        assert!(
            server_url.starts_with("https://"),
            "{provider} MCP cassette requires a public HTTPS server_url"
        );
    }

    for request in [openai_request, gateway_request] {
        assert_eq!(request.path, "/v1/responses");
        assert_eq!(request.body.stream, streaming);
        assert_eq!(request.body.tools.len(), 1);

        let declaration = &request.body.tools[0];
        assert_eq!(declaration["type"], "mcp");
        assert_eq!(declaration["server_label"], "counter");
        assert_eq!(declaration["require_approval"], "never");
        assert!(declaration.get("name").is_none());

        match allowed_tools {
            Some(expected) => {
                let actual = declaration["allowed_tools"]
                    .as_array()
                    .expect("allowed_tools array")
                    .iter()
                    .map(|name| name.as_str().expect("allowed tool name"))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
            }
            None => assert!(declaration.get("allowed_tools").is_none()),
        }
    }

    assert_eq!(openai_request.body.tool_choice, gateway_request.body.tool_choice);
}

fn streaming_events(turn: &support::Turn) -> Vec<Value> {
    support::recorded_named_sse_events(turn)
}

fn response_output(turn: &support::Turn) -> Vec<OutputItem> {
    let response = if let Some(body) = &turn.response.body {
        body.clone()
    } else {
        streaming_events(turn)
            .into_iter()
            .rev()
            .filter_map(|event| event.get("response").cloned())
            .find(|response| response["status"] == "completed" && response["output"].is_array())
            .expect("completed streaming response payload")
    };
    let accumulator = ResponseAccumulator::from_json(&response.to_string(), None).expect("valid completed response");
    let payload = accumulator.finalize(MODEL, None, None);
    assert_eq!(payload.status, "completed");
    payload.output
}

fn output_text(output: &[OutputItem]) -> String {
    output
        .iter()
        .filter_map(|item| match item {
            OutputItem::Message(message) => Some(
                message
                    .content
                    .iter()
                    .map(|content| content.text.as_str())
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect::<String>()
        .trim()
        .to_owned()
}

fn mcp_calls(output: &[OutputItem]) -> Vec<&McpCall> {
    assert!(
        !output.iter().any(|item| matches!(item, OutputItem::FunctionCall(_))),
        "internal function calls must not leak from the gateway"
    );
    output
        .iter()
        .filter_map(|item| match item {
            OutputItem::McpCall(call) => Some(call),
            _ => None,
        })
        .collect()
}

fn normalized_json_string(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()))
}

fn normalized_optional_output(output: Option<&str>) -> Value {
    output.map_or(Value::Null, normalized_json_string)
}

fn assert_calls_match_openai(openai: &[OutputItem], gateway: &[OutputItem], compare_arguments: bool) {
    let expected = mcp_calls(openai);
    let actual = mcp_calls(gateway);
    assert_eq!(actual.len(), expected.len());

    for (expected, actual) in expected.into_iter().zip(actual) {
        assert_eq!(actual.server_label, expected.server_label);
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.status, expected.status);
        assert_eq!(
            normalized_optional_output(actual.output.as_deref()),
            normalized_optional_output(expected.output.as_deref())
        );
        assert_eq!(
            serde_json::to_value(&actual.error).expect("gateway MCP error JSON"),
            serde_json::to_value(&expected.error).expect("OpenAI MCP error JSON")
        );
        if compare_arguments {
            assert_eq!(
                normalized_json_string(&actual.arguments),
                normalized_json_string(&expected.arguments)
            );
        } else {
            assert!(normalized_json_string(&actual.arguments).is_object());
            assert!(normalized_json_string(&expected.arguments).is_object());
        }
    }
}

fn mcp_call_event_traces(events: &[Value]) -> Vec<(String, Vec<String>)> {
    let mut traces: HashMap<String, (String, Vec<String>)> = HashMap::new();

    for event in events {
        let Some(event_type) = event["type"].as_str() else {
            continue;
        };
        let item = event.get("item");
        let mcp_item = item.filter(|item| item["type"] == "mcp_call");
        let item_id = mcp_item
            .and_then(|item| item["id"].as_str())
            .or_else(|| event["item_id"].as_str());
        let Some(item_id) = item_id else {
            continue;
        };

        if let Some(item) = mcp_item {
            let name = item["name"].as_str().expect("mcp_call name").to_owned();
            traces.entry(item_id.to_owned()).or_insert_with(|| (name, Vec::new()));
        }
        if event_type.starts_with("response.mcp_call")
            || matches!(event_type, "response.output_item.added" | "response.output_item.done") && mcp_item.is_some()
        {
            traces
                .get_mut(item_id)
                .expect("mcp_call added before lifecycle events")
                .1
                .push(event_type.to_owned());
        }
    }

    let mut traces = traces.into_values().collect::<Vec<_>>();
    traces.sort_by(|left, right| left.0.cmp(&right.0));
    traces
}

fn normalized_mcp_item_transitions(events: &[Value]) -> HashMap<String, Vec<Value>> {
    let mut transitions: HashMap<String, Vec<Value>> = HashMap::new();
    for event in events {
        let Some(event_type) = event["type"].as_str() else {
            continue;
        };
        if !matches!(event_type, "response.output_item.added" | "response.output_item.done") {
            continue;
        }
        let Some(item) = event.get("item") else {
            continue;
        };
        if item["type"] != "mcp_call" {
            continue;
        }
        let name = item["name"].as_str().expect("mcp_call name").to_owned();
        transitions.entry(name).or_default().push(json!({
            "event_type": event_type,
            "type": item["type"],
            "server_label": item["server_label"],
            "name": item["name"],
            "status": item["status"],
            "arguments": item["arguments"].as_str().map(normalized_json_string),
            "output": item["output"].as_str().map(normalized_json_string),
            "error": item["error"],
            "approval_request_id": item["approval_request_id"],
        }));
    }
    transitions
}

fn assert_streaming_contract_matches_openai(openai: &support::Turn, gateway: &support::Turn) {
    let expected = streaming_events(openai);
    let actual = streaming_events(gateway);
    assert_eq!(mcp_call_event_traces(&actual), mcp_call_event_traces(&expected));
    assert_eq!(
        normalized_mcp_item_transitions(&actual),
        normalized_mcp_item_transitions(&expected)
    );
    assert!(actual.iter().all(|event| {
        !event["type"]
            .as_str()
            .is_some_and(|kind| kind.contains("mcp_tool_call"))
    }));
}

#[test]
fn mcp_tool_listing_has_no_calls_on_either_provider() {
    let (openai, gateway) = load_scenario_pair("list-tools", true);
    assert_matching_native_mcp_requests(&openai, &gateway, true, None);

    let openai_output = response_output(&openai.turns[0]);
    let gateway_output = response_output(&gateway.turns[0]);
    assert_calls_match_openai(&openai_output, &gateway_output, true);
    assert_streaming_contract_matches_openai(&openai.turns[0], &gateway.turns[0]);

    for text in [output_text(&openai_output), output_text(&gateway_output)] {
        for tool_name in ["increment", "get_value", "sum"] {
            assert!(
                text.contains(tool_name),
                "tool listing should contain {tool_name}: {text}"
            );
        }
    }
}

#[test]
fn successful_streaming_mcp_calls_match_openai() {
    let (openai, gateway) = load_scenario_pair("call-sum-and-echo", true);
    assert_matching_native_mcp_requests(&openai, &gateway, true, Some(&["sum", "echo"]));

    let openai_output = response_output(&openai.turns[0]);
    let gateway_output = response_output(&gateway.turns[0]);
    assert_calls_match_openai(&openai_output, &gateway_output, true);
    assert_streaming_contract_matches_openai(&openai.turns[0], &gateway.turns[0]);
    assert_eq!(output_text(&gateway_output), output_text(&openai_output));
}

#[test]
fn missing_argument_mcp_failure_matches_openai() {
    let (openai, gateway) = load_scenario_pair("sum-missing-argument", true);
    assert_matching_native_mcp_requests(&openai, &gateway, true, Some(&["sum"]));

    let openai_output = response_output(&openai.turns[0]);
    let gateway_output = response_output(&gateway.turns[0]);
    assert_calls_match_openai(&openai_output, &gateway_output, true);
    assert_streaming_contract_matches_openai(&openai.turns[0], &gateway.turns[0]);
}

#[test]
fn invalid_argument_type_mcp_failure_matches_openai() {
    let (openai, gateway) = load_scenario_pair("sum-invalid-argument-type", true);
    assert_matching_native_mcp_requests(&openai, &gateway, true, Some(&["sum"]));

    let openai_output = response_output(&openai.turns[0]);
    let gateway_output = response_output(&gateway.turns[0]);
    assert_calls_match_openai(&openai_output, &gateway_output, true);
    assert_streaming_contract_matches_openai(&openai.turns[0], &gateway.turns[0]);
}

#[test]
fn successful_blocking_mcp_call_matches_openai() {
    let (openai, gateway) = load_scenario_pair("say-hello", false);
    assert_matching_native_mcp_requests(&openai, &gateway, false, Some(&["say_hello"]));

    let openai_output = response_output(&openai.turns[0]);
    let gateway_output = response_output(&gateway.turns[0]);
    // Arguments are model-generated. Qwen adds an ignored placeholder field
    // while GPT-4o sends `{}`; both execute the same zero-argument MCP tool.
    assert_calls_match_openai(&openai_output, &gateway_output, false);
    assert_eq!(output_text(&gateway_output), output_text(&openai_output));
}
