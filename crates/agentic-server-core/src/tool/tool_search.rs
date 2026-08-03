use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};

use crate::events::WireEvent;
use crate::types::io::{
    FunctionTool, InputItem, OutputItem, ResponsesInput, ToolSearchCall, ToolSearchOutput, ToolSearchStatus,
};
use crate::types::tools::{ToolSearchExecution, ToolSearchToolParam};
use crate::utils::common::deserialize_from_str_opt;

pub(crate) const TOOL_SEARCH_NAME: &str = "tool_search";

/// Convert the public tool-search declaration into the ordinary function
/// shape accepted by providers without native tool-search support.
///
/// The caller controls both the description and search argument schema, so
/// neither field may be replaced with gateway defaults.
pub(crate) fn tool_search_function_tool(declaration: &ToolSearchToolParam) -> FunctionTool {
    FunctionTool {
        type_: "function".to_owned(),
        name: TOOL_SEARCH_NAME.to_owned(),
        description: declaration.description.clone(),
        parameters: declaration.parameters.clone(),
        strict: None,
        defer_loading: None,
    }
}

/// Return valid client tool-search outputs in input order.
///
/// An output is trusted for provider promotion only when it is completed,
/// carries a non-empty call ID, and follows a completed client search call with
/// that ID. The first valid output for each call ID wins; later duplicates or
/// conflicting outputs are preserved on the wire but ignored for promotion.
fn valid_client_tool_search_outputs(input: &ResponsesInput) -> Vec<&ToolSearchOutput> {
    let ResponsesInput::Items(items) = input else {
        return Vec::new();
    };
    let mut calls = HashSet::new();
    let mut completed_outputs = HashSet::new();
    let mut outputs = Vec::new();

    for item in items {
        match item {
            InputItem::ToolSearchCall(call)
                if call.execution == Some(ToolSearchExecution::Client)
                    && call.status == Some(ToolSearchStatus::Completed) =>
            {
                if let Some(call_id) = call.call_id.as_deref().filter(|call_id| !call_id.is_empty()) {
                    calls.insert(call_id);
                }
            }
            InputItem::ToolSearchOutput(output)
                if output.execution == Some(ToolSearchExecution::Client)
                    && output.status == Some(ToolSearchStatus::Completed)
                    && output
                        .call_id
                        .as_deref()
                        .filter(|call_id| !call_id.is_empty())
                        .is_some_and(|call_id| calls.contains(call_id) && completed_outputs.insert(call_id)) =>
            {
                outputs.push(output);
            }
            _ => {}
        }
    }

    outputs
}

fn top_level_function_names(outputs: &[&ToolSearchOutput]) -> HashSet<String> {
    outputs
        .iter()
        .flat_map(|output| &output.tools)
        .filter_map(|tool| {
            tool.as_object()
                .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
                .and_then(|tool| tool.get("name").and_then(Value::as_str))
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
        })
        .collect()
}

/// Build an unqualified member-name to namespace map from client-provided
/// `tool_search_output` items.
///
/// Native namespace-capable providers return a `namespace` on the eventual
/// function call. Responses-compatible providers that flatten the loaded
/// namespace may return only the member name. Ambiguous member names are
/// intentionally excluded instead of guessing a namespace.
pub(crate) fn loaded_namespace_members(input: &ResponsesInput) -> HashMap<String, String> {
    let outputs = valid_client_tool_search_outputs(input);
    let top_level_names = top_level_function_names(&outputs);
    let mut members = HashMap::<String, Option<String>>::new();
    for output in outputs {
        for tool in &output.tools {
            let Some(namespace) = tool
                .as_object()
                .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("namespace"))
                .and_then(|tool| tool.get("name").and_then(Value::as_str))
                .filter(|namespace| !namespace.is_empty())
            else {
                continue;
            };
            let Some(tools) = tool.get("tools").and_then(Value::as_array) else {
                continue;
            };
            for member in tools {
                let Some(name) = member
                    .as_object()
                    .filter(|member| member.get("type").and_then(Value::as_str) == Some("function"))
                    .and_then(|member| member.get("name").and_then(Value::as_str))
                    .filter(|name| !name.is_empty())
                else {
                    continue;
                };
                members
                    .entry(name.to_owned())
                    .and_modify(|existing| {
                        if existing.as_deref() != Some(namespace) {
                            *existing = None;
                        }
                    })
                    .or_insert_with(|| Some(namespace.to_owned()));
            }
        }
    }

    members
        .into_iter()
        .filter_map(|(name, namespace)| {
            (!top_level_names.contains(&name))
                .then_some(namespace)
                .flatten()
                .map(|namespace| (name, namespace))
        })
        .collect()
}

/// Convert uniquely named functions returned by client-side tool search into
/// provider-facing declarations for the next inference call.
///
/// Codex keeps loaded definitions inside `tool_search_output`. Providers with
/// native dynamic-tool support can consume those definitions from the input
/// item directly. Responses-compatible providers that only understand a flat
/// `tools` array need the selected definitions repeated there. The functions
/// are no longer marked deferred because the client has explicitly loaded
/// them. Their namespace is restored on the eventual call before it is
/// returned to Codex.
pub(crate) fn loaded_function_tools(input: &ResponsesInput) -> Vec<FunctionTool> {
    let outputs = valid_client_tool_search_outputs(input);
    let unique_namespaces = loaded_namespace_members(input);
    let mut emitted = HashSet::new();
    let mut loaded = Vec::new();

    for output in outputs {
        for tool in &output.tools {
            let Some(tool) = tool.as_object() else {
                continue;
            };
            match tool.get("type").and_then(Value::as_str) {
                Some("function") => {
                    if let Some(function) = function_tool_from_object(tool)
                        && emitted.insert(function.name.clone())
                    {
                        loaded.push(function);
                    }
                }
                Some("namespace") => {
                    let Some(namespace_name) = tool.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    let Some(members) = tool.get("tools").and_then(Value::as_array) else {
                        continue;
                    };
                    for member in members {
                        let Some(member) = member.as_object() else {
                            continue;
                        };
                        let Some(function) = function_tool_from_object(member) else {
                            continue;
                        };
                        if unique_namespaces.get(&function.name).map(String::as_str) == Some(namespace_name)
                            && emitted.insert(function.name.clone())
                        {
                            loaded.push(function);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    loaded
}

fn function_tool_from_object(tool: &Map<String, Value>) -> Option<FunctionTool> {
    if tool.get("type").and_then(Value::as_str) != Some("function") {
        return None;
    }
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())?;
    Some(FunctionTool {
        type_: "function".to_owned(),
        name: name.to_owned(),
        description: tool.get("description").and_then(Value::as_str).map(str::to_owned),
        parameters: tool.get("parameters").filter(|value| !value.is_null()).cloned(),
        strict: tool.get("strict").and_then(Value::as_bool),
        defer_loading: None,
    })
}

pub(crate) fn loaded_function_names(input: &ResponsesInput) -> HashSet<String> {
    let mut names = top_level_function_names(&valid_client_tool_search_outputs(input));
    names.extend(loaded_namespace_members(input).into_keys());
    names
}

/// Restore Responses-compatible providers' function-call fallback to the
/// canonical client-executed tool-search item.
///
/// Some providers accept a native `type: "tool_search"` declaration but emit
/// the selected invocation as `type: "function_call", name: "tool_search"`.
/// Codex dispatches search only when it receives `tool_search_call`, so normalize
/// that provider fallback at the same boundary where namespace calls are
/// restored. The conversion is enabled only when the request declared a
/// client-executed tool search.
pub(crate) fn restore_output_items(output: &mut [OutputItem], enabled: bool) {
    if !enabled {
        return;
    }

    for item in output {
        let OutputItem::FunctionCall(call) = item else {
            continue;
        };
        if call.name != TOOL_SEARCH_NAME || call.namespace.is_some() || call.call_id.is_empty() {
            continue;
        }
        let Some(arguments) = deserialize_from_str_opt::<Value>(&call.arguments) else {
            tracing::warn!(call_id = %call.call_id, "cannot restore tool_search call with invalid JSON arguments");
            continue;
        };

        let call_id = call.call_id.clone();
        *item = OutputItem::ToolSearchCall(ToolSearchCall {
            execution: Some(ToolSearchExecution::Client),
            call_id: Some(call_id.clone()),
            status: Some(call.status.into()),
            arguments,
            extra: HashMap::new(),
        });
        tracing::debug!(%call_id, "restored provider function_call fallback as tool_search_call");
    }
}

pub(crate) fn restore_loaded_namespace_output_items(
    output: &mut [OutputItem],
    loaded_namespaces: &HashMap<String, String>,
) {
    for item in output {
        let OutputItem::FunctionCall(call) = item else {
            continue;
        };
        if call.namespace.is_some() {
            continue;
        }
        let Some(namespace) = loaded_namespaces.get(&call.name) else {
            continue;
        };
        call.namespace = Some(namespace.clone());
        tracing::debug!(
            call_id = %call.call_id,
            %namespace,
            member = %call.name,
            "restored namespace on dynamically loaded tool call"
        );
    }
}

/// Restore a streamed function-call fallback in-place.
///
/// This handles output-item events and response envelopes. Function-argument
/// delta events are suppressed separately by the streaming executor.
pub(crate) fn restore_response_value(value: &mut Value, enabled: bool) -> bool {
    if !enabled {
        return false;
    }

    let mut changed = false;
    if let Some(item) = value.as_object_mut().and_then(|object| object.get_mut("item")) {
        changed |= restore_call_value(item);
    }
    changed |= restore_call_value(value);

    for key in ["response", "payload"] {
        if let Some(nested) = value.as_object_mut().and_then(|object| object.get_mut(key)) {
            changed |= restore_response_value(nested, enabled);
        }
    }
    if let Some(Value::Array(items)) = value.as_object_mut().and_then(|object| object.get_mut("output")) {
        for item in items {
            changed |= restore_call_value(item);
        }
    }

    changed
}

/// Restore tool-search function fallbacks inside a parsed streaming event.
pub(crate) fn restore_response_wire(wire: &mut WireEvent, enabled: bool) -> bool {
    if !enabled {
        return false;
    }
    restore_response_map(&mut wire.rest)
}

pub(crate) fn restore_loaded_namespace_response_value(
    value: &mut Value,
    loaded_namespaces: &HashMap<String, String>,
) -> bool {
    if loaded_namespaces.is_empty() {
        return false;
    }

    let mut changed = false;
    if let Some(item) = value.as_object_mut().and_then(|object| object.get_mut("item")) {
        changed |= restore_loaded_namespace_call_value(item, loaded_namespaces);
    }
    changed |= restore_loaded_namespace_call_value(value, loaded_namespaces);
    for key in ["response", "payload"] {
        if let Some(nested) = value.as_object_mut().and_then(|object| object.get_mut(key)) {
            changed |= restore_loaded_namespace_response_value(nested, loaded_namespaces);
        }
    }
    if let Some(Value::Array(items)) = value.as_object_mut().and_then(|object| object.get_mut("output")) {
        for item in items {
            changed |= restore_loaded_namespace_call_value(item, loaded_namespaces);
        }
    }
    changed
}

pub(crate) fn restore_loaded_namespace_response_wire(
    wire: &mut WireEvent,
    loaded_namespaces: &HashMap<String, String>,
) -> bool {
    if loaded_namespaces.is_empty() {
        return false;
    }
    restore_loaded_namespace_response_map(&mut wire.rest, loaded_namespaces)
}

fn restore_response_map(object: &mut Map<String, Value>) -> bool {
    let mut changed = false;
    if let Some(item) = object.get_mut("item") {
        changed |= restore_call_value(item);
    }
    for key in ["response", "payload"] {
        if let Some(nested) = object.get_mut(key) {
            changed |= restore_response_value(nested, true);
        }
    }
    if let Some(Value::Array(items)) = object.get_mut("output") {
        for item in items {
            changed |= restore_call_value(item);
        }
    }
    changed
}

fn restore_loaded_namespace_response_map(
    object: &mut Map<String, Value>,
    loaded_namespaces: &HashMap<String, String>,
) -> bool {
    let mut changed = false;
    if let Some(item) = object.get_mut("item") {
        changed |= restore_loaded_namespace_call_value(item, loaded_namespaces);
    }
    for key in ["response", "payload"] {
        if let Some(nested) = object.get_mut(key) {
            changed |= restore_loaded_namespace_response_value(nested, loaded_namespaces);
        }
    }
    if let Some(Value::Array(items)) = object.get_mut("output") {
        for item in items {
            changed |= restore_loaded_namespace_call_value(item, loaded_namespaces);
        }
    }
    changed
}

fn restore_call_value(value: &mut Value) -> bool {
    let Some(object) = value.as_object_mut() else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) != Some("function_call")
        || object.get("name").and_then(Value::as_str) != Some(TOOL_SEARCH_NAME)
        || object.get("namespace").and_then(Value::as_str).is_some()
    {
        return false;
    }
    if object
        .get("call_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .is_empty()
    {
        return false;
    }

    let arguments = object
        .get("arguments")
        .and_then(Value::as_str)
        .filter(|arguments| !arguments.is_empty())
        .and_then(deserialize_from_str_opt::<Value>)
        .unwrap_or_else(|| Value::Object(Map::new()));
    object.insert("type".to_owned(), Value::String("tool_search_call".to_owned()));
    object.insert("execution".to_owned(), Value::String("client".to_owned()));
    object.insert("arguments".to_owned(), arguments);
    object.remove("id");
    object.remove("name");
    object.remove("namespace");
    true
}

fn restore_loaded_namespace_call_value(value: &mut Value, loaded_namespaces: &HashMap<String, String>) -> bool {
    let Some(object) = value.as_object_mut() else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) != Some("function_call")
        || object.get("namespace").and_then(Value::as_str).is_some()
    {
        return false;
    }
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return false;
    };
    let Some(namespace) = loaded_namespaces.get(name) else {
        return false;
    };
    object.insert("namespace".to_owned(), Value::String(namespace.clone()));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::event::MessageStatus;
    use crate::types::io::FunctionToolCall;

    #[test]
    fn restores_final_function_call_fallback() {
        let mut output = vec![OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_search".to_owned(),
            call_id: "call_search".to_owned(),
            name: TOOL_SEARCH_NAME.to_owned(),
            namespace: None,
            arguments: r#"{"query":"calendar","limit":2}"#.to_owned(),
            status: MessageStatus::Completed,
        })];

        restore_output_items(&mut output, true);

        let OutputItem::ToolSearchCall(call) = &output[0] else {
            panic!("expected restored tool_search_call");
        };
        assert_eq!(call.call_id.as_deref(), Some("call_search"));
        assert_eq!(call.status, Some(ToolSearchStatus::Completed));
        assert_eq!(call.arguments["query"], "calendar");
        assert!(!call.extra.contains_key("id"));
    }

    #[test]
    fn restores_streamed_output_item_and_preserves_unrelated_function() {
        let mut event = serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_search",
                "call_id": "call_search",
                "name": "tool_search",
                "status": "completed",
                "arguments": "{\"query\":\"calendar\"}"
            }
        });
        assert!(restore_response_value(&mut event, true));
        assert_eq!(event["item"]["type"], "tool_search_call");
        assert_eq!(event["item"]["execution"], "client");
        assert_eq!(event["item"]["arguments"]["query"], "calendar");
        assert!(event["item"].get("name").is_none());

        let mut unrelated = serde_json::json!({
            "type": "function_call",
            "call_id": "call_other",
            "name": "other",
            "arguments": "{}"
        });
        assert!(!restore_response_value(&mut unrelated, true));
        assert_eq!(unrelated["type"], "function_call");
    }

    #[test]
    fn restores_namespace_for_a_dynamically_loaded_member() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "arguments": {"query": "echo_text"}
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "tools": [{
                    "type": "namespace",
                    "name": "mcp__fixture",
                    "description": "Fixture tools",
                    "tools": [{
                        "type": "function",
                        "name": "echo_text",
                        "defer_loading": true,
                        "parameters": {"type": "object"}
                    }]
                }]
            },
            {
                "type": "tool_search_output",
                "execution": "server",
                "call_id": "call_server_search",
                "status": "completed",
                "tools": [{
                    "type": "namespace",
                    "name": "server",
                    "tools": [{"type": "function", "name": "server_only"}]
                }]
            }
        ]))
        .unwrap();
        let namespaces = loaded_namespace_members(&input);
        assert_eq!(namespaces.get("echo_text").map(String::as_str), Some("mcp__fixture"));
        assert!(!namespaces.contains_key("server_only"));

        let mut event = serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "call_echo",
                "name": "echo_text",
                "arguments": "{\"text\":\"hello\"}"
            }
        });
        assert!(restore_loaded_namespace_response_value(&mut event, &namespaces));
        assert_eq!(event["item"]["namespace"], "mcp__fixture");
    }

    #[test]
    fn promotes_only_uniquely_namespaced_loaded_functions() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "arguments": {"query": "fixture tools"}
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "mcp__one",
                        "tools": [
                            {
                                "type": "function",
                                "name": "echo_text",
                                "description": "Echo text",
                                "parameters": {"type": "object"},
                                "strict": false,
                                "defer_loading": true
                            },
                            {"type": "function", "name": "ambiguous"}
                        ]
                    },
                    {
                        "type": "namespace",
                        "name": "mcp__two",
                        "tools": [{"type": "function", "name": "ambiguous"}]
                    }
                ]
            }
        ]))
        .unwrap();

        let loaded = loaded_function_tools(&input);

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "echo_text");
        assert_eq!(loaded[0].description.as_deref(), Some("Echo text"));
        assert_eq!(
            loaded[0].parameters.as_ref().and_then(|value| value.get("type")),
            Some(&Value::String("object".to_owned()))
        );
        assert_eq!(loaded[0].strict, Some(false));
        assert_eq!(loaded[0].defer_loading, None);
    }

    #[test]
    fn promotes_top_level_functions_and_unique_namespace_members() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "arguments": {"query": "tools"}
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "tools": [
                    {
                        "type": "function",
                        "name": "direct_lookup",
                        "description": "A direct function.",
                        "parameters": {"type": "object"},
                        "defer_loading": true
                    },
                    {
                        "type": "namespace",
                        "name": "mcp__fixture",
                        "tools": [{
                            "type": "function",
                            "name": "namespaced_lookup",
                            "description": "A namespace member.",
                            "parameters": {"type": "object"},
                            "defer_loading": true
                        }]
                    }
                ]
            }
        ]))
        .unwrap();

        let loaded = loaded_function_tools(&input);
        assert_eq!(
            loaded.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(),
            ["direct_lookup", "namespaced_lookup"]
        );
        let namespaces = loaded_namespace_members(&input);
        assert!(!namespaces.contains_key("direct_lookup"));
        assert_eq!(
            namespaces.get("namespaced_lookup").map(String::as_str),
            Some("mcp__fixture")
        );
    }

    #[test]
    fn promotion_requires_a_prior_matching_completed_client_call() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": null,
                "status": "completed",
                "tools": [{"type": "function", "name": "null_id"}]
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "unmatched",
                "status": "completed",
                "tools": [{"type": "function", "name": "unmatched"}]
            },
            {
                "type": "tool_search_call",
                "execution": "server",
                "call_id": "server",
                "status": "completed",
                "arguments": {}
            },
            {
                "type": "tool_search_output",
                "execution": "server",
                "call_id": "server",
                "status": "completed",
                "tools": [{"type": "function", "name": "server"}]
            },
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "in_progress",
                "status": "in_progress",
                "arguments": {}
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "in_progress",
                "status": "completed",
                "tools": [{"type": "function", "name": "in_progress"}]
            },
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "incomplete",
                "status": "incomplete",
                "arguments": {}
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "incomplete",
                "status": "incomplete",
                "tools": [{"type": "function", "name": "incomplete"}]
            },
            {
                "type": "tool_search_call",
                "call_id": "absent_fields",
                "arguments": {}
            },
            {
                "type": "tool_search_output",
                "call_id": "absent_fields",
                "tools": [{"type": "function", "name": "absent_fields"}]
            },
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "valid",
                "status": "completed",
                "arguments": {}
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "valid",
                "status": "completed",
                "tools": [{"type": "function", "name": "valid"}]
            }
        ]))
        .unwrap();

        let loaded = loaded_function_tools(&input);
        assert_eq!(
            loaded.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(),
            ["valid"]
        );
    }

    #[test]
    fn first_valid_output_for_a_call_id_wins_deterministically() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "arguments": {}
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "tools": [{"type": "function", "name": "first"}]
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "tools": [{"type": "function", "name": "conflicting_second"}]
            }
        ]))
        .unwrap();

        let loaded = loaded_function_tools(&input);
        assert_eq!(
            loaded.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(),
            ["first"]
        );
    }

    #[test]
    fn direct_function_name_wins_over_a_namespaced_member_collision() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "arguments": {}
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "tools": [
                    {"type": "namespace", "name": "ns", "tools": [{"type": "function", "name": "same"}]},
                    {"type": "function", "name": "same", "description": "direct"}
                ]
            }
        ]))
        .unwrap();

        let loaded = loaded_function_tools(&input);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "same");
        assert_eq!(loaded[0].description.as_deref(), Some("direct"));
        assert!(!loaded_namespace_members(&input).contains_key("same"));
    }

    #[test]
    fn namespaced_tool_search_function_is_not_rewritten_as_search_fallback() {
        let mut output = vec![OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_search".to_owned(),
            call_id: "call_search".to_owned(),
            name: TOOL_SEARCH_NAME.to_owned(),
            namespace: Some("legitimate_namespace".to_owned()),
            arguments: "{}".to_owned(),
            status: MessageStatus::Completed,
        })];

        restore_output_items(&mut output, true);
        assert!(matches!(output[0], OutputItem::FunctionCall(_)));
    }
}
