//! Cassette-driven validation of the tool framework wire types and normalization pipeline.
//!
//! Validates that real cassette request bodies — the exact JSON the gateway receives —
//! parse correctly into `Vec<ResponsesTool>` and normalize through the full pipeline.

use serde::Deserialize;

use agentic_core::tool::{ToolRegistry, ToolType, flatten_tools_for_upstream, model_visible_namespace_member_name};
use agentic_core::types::request_response::RequestPayload;
use agentic_core::types::tools::ResponsesTool;

const MULTI_TURN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/tool_calls/multi_turn");
const CODEX_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/codex");

const CODEX_CASSETTES: &[&str] = &[
    "codex-direct-vllm-http-flat-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-direct-vllm-http-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-http-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-http-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-websocket-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-websocket-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-openai-https-function-tool-gpt-4o-streaming.yaml",
    "codex-openai-https-namespace-tool-gpt-4o-streaming.yaml",
    "codex-openai-websocket-function-tool-gpt-4o-streaming.yaml",
    "codex-openai-websocket-namespace-tool-gpt-4o-streaming.yaml",
];

const CODEX_NAMESPACE_CASSETTES: &[&str] = &[
    "codex-gateway-http-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-websocket-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-openai-https-namespace-tool-gpt-4o-streaming.yaml",
    "codex-openai-websocket-namespace-tool-gpt-4o-streaming.yaml",
];

#[derive(Deserialize)]
struct TurnCassette {
    turns: Vec<Turn>,
}

#[derive(Deserialize)]
struct Turn {
    request: serde_yml::Value,
    #[allow(dead_code)]
    response: serde_yml::Value,
}

fn load_cassette_from(dir: &str, filename: &str) -> TurnCassette {
    let path = format!("{dir}/{filename}");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_yml::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn load_cassette(filename: &str) -> TurnCassette {
    load_cassette_from(MULTI_TURN_DIR, filename)
}

fn load_codex_cassette(filename: &str) -> TurnCassette {
    load_cassette_from(CODEX_DIR, filename)
}

fn tools_from_turn(turn: &Turn) -> Option<serde_json::Value> {
    let body = turn.request.get("body")?;
    let json: serde_json::Value = serde_json::to_value(body).ok()?;
    json.get("tools").cloned()
}

fn request_body_from_turn(turn: &Turn) -> serde_json::Value {
    let body = turn.request.get("body").expect("turn has body");
    serde_json::to_value(body).expect("body to json")
}

fn parse_tools_from_turn(cassette_file: &str, turn_idx: usize, turn: &Turn) -> Vec<ResponsesTool> {
    let tools_val =
        tools_from_turn(turn).unwrap_or_else(|| panic!("{cassette_file} turn {turn_idx}: expected tools array"));
    serde_json::from_value(tools_val.clone())
        .unwrap_or_else(|e| panic!("{cassette_file} turn {turn_idx}: tools parse failed: {e}\nJSON: {tools_val}"))
}

/// Parse `request.body.tools` from every turn of a cassette into `Vec<ResponsesTool>`.
fn assert_tools_parse(cassette_file: &str) {
    let cassette = load_cassette(cassette_file);
    for (i, turn) in cassette.turns.iter().enumerate() {
        let Some(tools_val) = tools_from_turn(turn) else {
            continue;
        };
        let tools: Vec<ResponsesTool> = serde_json::from_value(tools_val.clone())
            .unwrap_or_else(|e| panic!("{cassette_file} turn {i}: tools parse failed: {e}\nJSON: {tools_val}"));
        assert!(
            !tools.is_empty(),
            "{cassette_file} turn {i}: expected non-empty tools array"
        );
    }
}

/// For every turn that has tools, verify normalization produces only `FunctionTool` entries.
fn assert_tools_normalize(cassette_file: &str) {
    let cassette = load_cassette(cassette_file);
    for (i, turn) in cassette.turns.iter().enumerate() {
        let Some(tools_val) = tools_from_turn(turn) else {
            continue;
        };
        let tools: Vec<ResponsesTool> = serde_json::from_value(tools_val).expect("tools parse");
        let normalized: Vec<_> = tools.iter().filter_map(ResponsesTool::to_function_tool).collect();
        for ft in &normalized {
            assert_eq!(
                ft.type_, "function",
                "{cassette_file} turn {i}: normalized type must be 'function'"
            );
            assert!(
                !ft.name.is_empty(),
                "{cassette_file} turn {i}: normalized name must not be empty"
            );
        }
        // Every Function variant must normalize — count only Function entries
        // so the assertion holds even if future cassettes include gateway-only types.
        let function_count = tools.iter().filter(|t| matches!(t, ResponsesTool::Function(_))).count();
        assert_eq!(
            normalized.len(),
            function_count,
            "{cassette_file} turn {i}: each Function tool must produce exactly one FunctionTool (got {} of {})",
            normalized.len(),
            function_count
        );
    }
}

/// For every turn that has tools, verify `ToolRegistry::build` produces correct entries.
fn assert_registry_lookup(cassette_file: &str) {
    let cassette = load_cassette(cassette_file);
    for (i, turn) in cassette.turns.iter().enumerate() {
        let Some(tools_val) = tools_from_turn(turn) else {
            continue;
        };
        let tools: Vec<ResponsesTool> = serde_json::from_value(tools_val).expect("tools parse");
        let registry = ToolRegistry::build(&tools);
        for tool in &tools {
            if let ResponsesTool::Function(p) = tool {
                let entry = registry
                    .lookup(p.name.as_str())
                    .unwrap_or_else(|| panic!("{cassette_file} turn {i}: tool '{}' not found in registry", p.name));
                assert_eq!(
                    entry.tool_type,
                    ToolType::Function,
                    "{cassette_file} turn {i}: tool '{}' must be Function type",
                    p.name
                );
            }
        }
    }
}

/// Full round-trip: deserialize `request.body` → `RequestPayload` → `to_upstream_request()`
/// → assert upstream tools only contains `FunctionTool` entries.
fn assert_full_roundtrip(cassette_file: &str) {
    let cassette = load_cassette(cassette_file);
    for (i, turn) in cassette.turns.iter().enumerate() {
        let body = turn.request.get("body").expect("turn has body");
        let json: serde_json::Value = serde_json::to_value(body).expect("body to json");
        let payload: RequestPayload = serde_json::from_value(json.clone())
            .unwrap_or_else(|e| panic!("{cassette_file} turn {i}: RequestPayload parse failed: {e}"));
        let upstream = payload.to_upstream_request(false);
        if let Some(tools) = &upstream.tools {
            for ft in tools {
                assert_eq!(
                    ft.type_, "function",
                    "{cassette_file} turn {i}: upstream tools must only contain FunctionTool"
                );
                assert!(
                    !ft.name.is_empty(),
                    "{cassette_file} turn {i}: upstream tool name must not be empty"
                );
            }
        }
    }
}

#[test]
fn tools_parse_3turn() {
    assert_tools_parse("openai_responses_tool_calls_3turn.yaml");
}

#[test]
fn tools_parse_5turn() {
    assert_tools_parse("openai_responses_tool_calls_5turn.yaml");
}

#[test]
fn tools_parse_parallel() {
    assert_tools_parse("openai_responses_tool_calls_parallel.yaml");
}

#[test]
fn tools_normalize_3turn() {
    assert_tools_normalize("openai_responses_tool_calls_3turn.yaml");
}

#[test]
fn tools_normalize_5turn() {
    assert_tools_normalize("openai_responses_tool_calls_5turn.yaml");
}

#[test]
fn tools_normalize_parallel() {
    assert_tools_normalize("openai_responses_tool_calls_parallel.yaml");
}

#[test]
fn registry_lookup_3turn() {
    assert_registry_lookup("openai_responses_tool_calls_3turn.yaml");
}

#[test]
fn registry_lookup_5turn() {
    assert_registry_lookup("openai_responses_tool_calls_5turn.yaml");
}

#[test]
fn registry_lookup_parallel() {
    assert_registry_lookup("openai_responses_tool_calls_parallel.yaml");
}

#[test]
fn roundtrip_3turn() {
    assert_full_roundtrip("openai_responses_tool_calls_3turn.yaml");
}

#[test]
fn roundtrip_5turn() {
    assert_full_roundtrip("openai_responses_tool_calls_5turn.yaml");
}

#[test]
fn roundtrip_parallel() {
    assert_full_roundtrip("openai_responses_tool_calls_parallel.yaml");
}

#[test]
fn codex_request_payloads_parse_all_recorded_shapes() {
    for filename in CODEX_CASSETTES {
        let cassette = load_codex_cassette(filename);
        assert_eq!(cassette.turns.len(), 2, "{filename} should have two turns");

        for (i, turn) in cassette.turns.iter().enumerate() {
            let json = request_body_from_turn(turn);
            let payload: RequestPayload = serde_json::from_value(json.clone())
                .unwrap_or_else(|e| panic!("{filename} turn {i}: RequestPayload parse failed: {e}\nJSON: {json}"));
            assert!(
                payload.tools.as_ref().is_some_and(|tools| !tools.is_empty()),
                "{filename} turn {i}: expected tools"
            );
            if filename.contains("websocket") {
                assert!(
                    json.get("stream").is_none(),
                    "{filename} turn {i}: websocket cassette should not contain HTTP-only stream field"
                );
            }
        }
    }
}

#[test]
fn codex_namespace_cassettes_flatten_to_safe_upstream_function_name() {
    let expected_flat_name = model_visible_namespace_member_name("mcp__agentic_fixture", "add_numbers");

    for filename in CODEX_NAMESPACE_CASSETTES {
        let cassette = load_codex_cassette(filename);
        for (i, turn) in cassette.turns.iter().enumerate() {
            let tools = parse_tools_from_turn(filename, i, turn);
            assert!(
                tools.iter().any(|tool| {
                    matches!(
                        tool,
                        ResponsesTool::Namespace(namespace)
                            if namespace.name == "mcp__agentic_fixture"
                                && namespace.tools.iter().any(|member| {
                                    matches!(
                                        member,
                                        agentic_core::types::CodexNamespaceMember::Function(function)
                                            if function.name.as_str() == "add_numbers"
                                    )
                                })
                    )
                }),
                "{filename} turn {i}: expected raw namespace tool"
            );

            let flattened = flatten_tools_for_upstream(Some(&tools)).expect("flattened tools");
            assert!(
                flattened
                    .iter()
                    .all(|tool| !matches!(tool, ResponsesTool::Namespace(_))),
                "{filename} turn {i}: namespace tools should be flattened before upstream"
            );
            assert!(
                flattened.iter().any(|tool| {
                    matches!(
                        tool,
                        ResponsesTool::Function(function) if function.name.as_str() == expected_flat_name
                    )
                }),
                "{filename} turn {i}: expected flattened tool name {expected_flat_name}"
            );
            let upstream_tools: Vec<_> = flattened.iter().filter_map(ResponsesTool::to_function_tool).collect();
            assert!(
                upstream_tools.iter().any(|tool| tool.name == expected_flat_name),
                "{filename} turn {i}: expected upstream FunctionTool {expected_flat_name}"
            );
        }
    }
}

#[test]
fn codex_direct_vllm_flat_namespace_cassette_is_plain_function_tool() {
    let filename = "codex-direct-vllm-http-flat-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml";
    let expected_flat_name = model_visible_namespace_member_name("mcp__agentic_fixture", "add_numbers");
    let cassette = load_codex_cassette(filename);

    for (i, turn) in cassette.turns.iter().enumerate() {
        let tools = parse_tools_from_turn(filename, i, turn);
        assert_eq!(
            tools.len(),
            1,
            "{filename} turn {i}: direct vLLM flat namespace fixture should declare one tool"
        );
        assert!(
            matches!(
                &tools[0],
                ResponsesTool::Function(function) if function.name.as_str() == expected_flat_name
            ),
            "{filename} turn {i}: expected direct vLLM to see a plain function tool named {expected_flat_name}"
        );

        let flattened = flatten_tools_for_upstream(Some(&tools)).expect("flattened tools");
        assert_eq!(flattened.len(), 1);
        assert!(
            matches!(
                &flattened[0],
                ResponsesTool::Function(function) if function.name.as_str() == expected_flat_name
            ),
            "{filename} turn {i}: flattening should preserve already-flat direct vLLM function"
        );
    }
}
