use std::collections::HashSet;

use serde_json::Value;

use crate::types::event::MessageStatus;
use crate::types::io::input::FunctionToolResultMessage;
use crate::types::io::{FunctionTool, FunctionToolCall, OutputItem, ToolChoice};
use crate::types::tools::{
    CodexNamespaceMember, CodexNamespaceToolParam, FunctionToolParam, NonEmptyToolName, ResponsesTool,
};

use super::handler::ToolOutput;

impl ResponsesTool {
    /// Normalise this tool declaration to the `FunctionTool` wire format that vLLM understands.
    ///
    /// `Function` variants convert directly. Codex namespace members should be flattened first
    /// with [`flatten_tools_for_upstream`], which turns each member into a `Function` variant.
    #[must_use]
    pub fn to_function_tool(&self) -> Option<FunctionTool> {
        match self {
            ResponsesTool::Function(p) => Some(FunctionTool::from(p)),
            ResponsesTool::Mcp(p) => {
                tracing::debug!(
                    server_label = %p.server_label,
                    "MCP tool skipped in normalize - handler not yet registered"
                );
                None
            }
            ResponsesTool::WebSearch(_) => {
                tracing::debug!("web_search tool skipped in normalize - handler not yet registered");
                None
            }
            ResponsesTool::FileSearch(_) => {
                tracing::debug!("file_search tool skipped in normalize - handler not yet registered");
                None
            }
            ResponsesTool::CodeInterpreter(_) => {
                tracing::debug!("code_interpreter tool skipped in normalize - handler not yet registered");
                None
            }
            ResponsesTool::Namespace(_) => {
                tracing::debug!("namespace tool skipped in normalize - flatten_tools_for_upstream should run first");
                None
            }
            ResponsesTool::ToolSearch(_) => {
                tracing::debug!("tool_search tool skipped in normalize - provider/client-owned");
                None
            }
            ResponsesTool::Custom(_) => {
                tracing::debug!("custom tool skipped in normalize - client-owned raw tool");
                None
            }
            ResponsesTool::Unknown(value) => {
                let tool_type = value.get("type").and_then(Value::as_str).unwrap_or("<missing>");
                tracing::debug!(tool_type, "unknown tool skipped in normalize");
                None
            }
        }
    }
}

impl From<ToolOutput> for FunctionToolResultMessage {
    fn from(o: ToolOutput) -> Self {
        Self {
            call_id: o.call_id,
            output: o.output,
        }
    }
}

pub const MODEL_VISIBLE_NAMESPACE_MEMBER_PREFIX: &str = "agentic_ns__";

#[must_use]
pub fn model_visible_namespace_member_name(namespace: &str, member: &str) -> String {
    format!("{MODEL_VISIBLE_NAMESPACE_MEMBER_PREFIX}{namespace}__{member}")
}

#[must_use]
pub fn legacy_model_visible_namespace_member_name(namespace: &str, member: &str) -> String {
    format!("{namespace}.{member}")
}

#[must_use]
pub fn alternate_model_visible_namespace_member_name(namespace: &str, member: &str) -> String {
    legacy_model_visible_namespace_member_name(namespace, member).replace('.', "_")
}

#[must_use]
pub fn flatten_tools_for_upstream(tools: Option<&[ResponsesTool]>) -> Option<Vec<ResponsesTool>> {
    tools.map(|tools| {
        let top_level_names = top_level_tool_names(tools);
        let mut upstream_tools = Vec::with_capacity(tools.len());
        for tool in tools {
            match tool {
                ResponsesTool::Namespace(namespace) => {
                    if namespace_has_flat_name_collision(namespace, &top_level_names) {
                        tracing::debug!(
                            namespace = %namespace.name,
                            "leaving namespace tool unflattened because a top-level tool uses a generated name"
                        );
                        upstream_tools.push(tool.clone());
                        continue;
                    }

                    let mut emitted_members = false;
                    for member in &namespace.tools {
                        if let CodexNamespaceMember::Function(function) = member {
                            let flat_name_text =
                                model_visible_namespace_member_name(&namespace.name, function.name.as_str());
                            let Ok(flat_name) = NonEmptyToolName::try_from(flat_name_text.clone()) else {
                                continue;
                            };
                            tracing::debug!(
                                namespace = %namespace.name,
                                member = %function.name.as_str(),
                                upstream_name = %flat_name_text,
                                "flattened namespace tool member for upstream"
                            );
                            let mut function = function.clone();
                            function.name = flat_name;
                            upstream_tools.push(ResponsesTool::Function(function));
                            emitted_members = true;
                        }
                    }
                    if !emitted_members {
                        tracing::debug!(
                            namespace = %namespace.name,
                            "leaving namespace tool unflattened because it has no function members"
                        );
                        upstream_tools.push(tool.clone());
                    }
                }
                ResponsesTool::Function(_)
                | ResponsesTool::Mcp(_)
                | ResponsesTool::WebSearch(_)
                | ResponsesTool::FileSearch(_)
                | ResponsesTool::CodeInterpreter(_)
                | ResponsesTool::ToolSearch(_)
                | ResponsesTool::Custom(_)
                | ResponsesTool::Unknown(_) => upstream_tools.push(tool.clone()),
            }
        }
        upstream_tools
    })
}

pub fn normalize_output_items_with_tools(output: &mut [OutputItem], tools: Option<&[ResponsesTool]>) {
    for item in output {
        if let OutputItem::FunctionCall(call) = item {
            normalize_function_call_with_tools(call, tools);
        }
    }
}

pub fn normalize_response_value_with_tools(value: &mut Value, tools: Option<&[ResponsesTool]>) -> bool {
    let Some(tools) = tools else {
        return false;
    };
    normalize_response_value_inner(value, tools)
}

fn normalize_response_value_inner(value: &mut Value, tools: &[ResponsesTool]) -> bool {
    let mut changed = false;

    if let Some(item) = value.as_object_mut().and_then(|object| object.get_mut("item")) {
        changed |= normalize_call_value_with_tools(item, tools);
    }

    changed |= normalize_call_value_with_tools(value, tools);

    for key in ["response", "payload"] {
        if let Some(nested) = value.as_object_mut().and_then(|object| object.get_mut(key)) {
            changed |= normalize_response_value_inner(nested, tools);
        }
    }

    if let Some(Value::Array(items)) = value.as_object_mut().and_then(|object| object.get_mut("output")) {
        for item in items {
            changed |= normalize_call_value_with_tools(item, tools);
        }
    }

    changed
}

fn normalize_call_value_with_tools(value: &mut Value, tools: &[ResponsesTool]) -> bool {
    let Some(object) = value.as_object_mut() else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) != Some("function_call") {
        return false;
    }
    if object.get("namespace").and_then(Value::as_str).is_some() {
        return false;
    }
    let Some(name) = object.get("name").and_then(Value::as_str).map(str::to_string) else {
        return false;
    };

    let had_arguments = object.contains_key("arguments");
    let arguments = object
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let original_name = name.clone();
    let original_arguments = arguments.clone();
    let mut call = FunctionToolCall {
        id: object.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
        call_id: object
            .get("call_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        name,
        namespace: None,
        arguments,
        status: MessageStatus::Completed,
    };

    normalize_function_call_with_tools(&mut call, Some(tools));
    if call.namespace.is_none() && call.name == original_name && call.arguments == original_arguments {
        return false;
    }

    object.insert("name".to_string(), Value::String(call.name));
    if let Some(namespace) = call.namespace {
        object.insert("namespace".to_string(), Value::String(namespace));
    }
    if had_arguments || call.arguments != original_arguments {
        object.insert("arguments".to_string(), Value::String(call.arguments));
    }
    true
}

fn normalize_function_call_with_tools(call: &mut FunctionToolCall, tools: Option<&[ResponsesTool]>) {
    if call.namespace.is_some() {
        return;
    }
    let Some(tools) = tools else {
        return;
    };

    if let Some((namespace, function)) = namespace_member_call(&call.name, tools) {
        tracing::debug!(
            upstream_name = %call.name,
            namespace = %namespace.name,
            member = %function.name.as_str(),
            match_kind = "namespace_member",
            "normalized upstream namespace function call"
        );
        apply_namespace_member_call(call, namespace, function);
        return;
    }

    if let Some((namespace, function)) = namespace_container_call(&call.name, tools) {
        tracing::debug!(
            upstream_name = %call.name,
            namespace = %namespace.name,
            member = %function.name.as_str(),
            match_kind = "namespace_container",
            stripped_container_arguments = true,
            "normalized upstream namespace function call"
        );
        apply_namespace_member_call(call, namespace, function);
        strip_namespace_container_arguments(&mut call.arguments);
        return;
    }

    if let Some((namespace, function)) = unambiguous_namespace_member_call(&call.name, tools) {
        tracing::debug!(
            upstream_name = %call.name,
            namespace = %namespace.name,
            member = %function.name.as_str(),
            match_kind = "bare_member",
            "normalized upstream namespace function call"
        );
        apply_namespace_member_call(call, namespace, function);
    }
}

fn apply_namespace_member_call(
    call: &mut FunctionToolCall,
    namespace: &CodexNamespaceToolParam,
    function: &FunctionToolParam,
) {
    call.namespace = Some(namespace.name.clone());
    call.name = function.name.as_str().to_string();
}

fn record_unique_namespace_match<'a>(
    found: &mut Option<(&'a CodexNamespaceToolParam, &'a FunctionToolParam)>,
    ambiguous: &mut bool,
    candidate: (&'a CodexNamespaceToolParam, &'a FunctionToolParam),
) {
    if found.is_some() {
        *ambiguous = true;
    } else {
        *found = Some(candidate);
    }
}

fn namespace_member_call<'a>(
    call_name: &str,
    tools: &'a [ResponsesTool],
) -> Option<(&'a CodexNamespaceToolParam, &'a FunctionToolParam)> {
    if tools.iter().any(|tool| top_level_tool_named(tool, call_name)) {
        return None;
    }

    let mut exact = None;
    let mut exact_ambiguous = false;
    let mut legacy = None;
    let mut legacy_ambiguous = false;
    let mut alternate = None;
    let mut alternate_ambiguous = false;

    for tool in tools {
        let ResponsesTool::Namespace(namespace) = tool else {
            continue;
        };
        if namespace_flat_name_collides(namespace, tools) {
            continue;
        }
        for member in &namespace.tools {
            let CodexNamespaceMember::Function(function) = member else {
                continue;
            };
            if call_name == model_visible_namespace_member_name(&namespace.name, function.name.as_str()) {
                record_unique_namespace_match(&mut exact, &mut exact_ambiguous, (namespace, function));
            }
            if call_name == legacy_model_visible_namespace_member_name(&namespace.name, function.name.as_str()) {
                record_unique_namespace_match(&mut legacy, &mut legacy_ambiguous, (namespace, function));
            }
            if call_name == alternate_model_visible_namespace_member_name(&namespace.name, function.name.as_str()) {
                record_unique_namespace_match(&mut alternate, &mut alternate_ambiguous, (namespace, function));
            }
        }
    }

    if !exact_ambiguous {
        if let Some(found) = exact {
            return Some(found);
        }
    }
    if !legacy_ambiguous {
        if let Some(found) = legacy {
            return Some(found);
        }
    }
    if alternate_ambiguous { None } else { alternate }
}

fn namespace_member_call_by_namespace<'a>(
    namespace_name: &str,
    member_name: &str,
    tools: &'a [ResponsesTool],
) -> Option<(&'a CodexNamespaceToolParam, &'a FunctionToolParam)> {
    for tool in tools {
        let ResponsesTool::Namespace(namespace) = tool else {
            continue;
        };
        if namespace.name != namespace_name {
            continue;
        }
        for member in &namespace.tools {
            let CodexNamespaceMember::Function(function) = member else {
                continue;
            };
            if function.name.as_str() == member_name {
                return Some((namespace, function));
            }
        }
    }
    None
}

fn top_level_tool_named(tool: &ResponsesTool, name: &str) -> bool {
    match tool {
        ResponsesTool::Function(function) => function.name.as_str() == name,
        ResponsesTool::Custom(custom) => custom.name == name,
        ResponsesTool::Unknown(value) => value.get("name").and_then(Value::as_str) == Some(name),
        ResponsesTool::Mcp(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Namespace(_)
        | ResponsesTool::ToolSearch(_) => false,
    }
}

fn top_level_tool_names(tools: &[ResponsesTool]) -> HashSet<String> {
    tools
        .iter()
        .filter_map(|tool| match tool {
            ResponsesTool::Function(function) => Some(function.name.as_str().to_string()),
            ResponsesTool::Custom(custom) => Some(custom.name.clone()),
            ResponsesTool::Unknown(value) => value.get("name").and_then(Value::as_str).map(str::to_string),
            ResponsesTool::Mcp(_)
            | ResponsesTool::WebSearch(_)
            | ResponsesTool::FileSearch(_)
            | ResponsesTool::CodeInterpreter(_)
            | ResponsesTool::Namespace(_)
            | ResponsesTool::ToolSearch(_) => None,
        })
        .collect()
}

fn namespace_has_flat_name_collision(namespace: &CodexNamespaceToolParam, top_level_names: &HashSet<String>) -> bool {
    namespace.tools.iter().any(|member| match member {
        CodexNamespaceMember::Function(function) => top_level_names.contains(&model_visible_namespace_member_name(
            &namespace.name,
            function.name.as_str(),
        )),
        CodexNamespaceMember::Unknown(_) => false,
    })
}

fn namespace_member_flat_name_collides(
    namespace: &CodexNamespaceToolParam,
    function: &FunctionToolParam,
    tools: &[ResponsesTool],
) -> bool {
    let flat_name = model_visible_namespace_member_name(&namespace.name, function.name.as_str());
    tools.iter().any(|tool| top_level_tool_named(tool, &flat_name))
}

fn namespace_flat_name_collides(namespace: &CodexNamespaceToolParam, tools: &[ResponsesTool]) -> bool {
    namespace.tools.iter().any(|member| match member {
        CodexNamespaceMember::Function(function) => namespace_member_flat_name_collides(namespace, function, tools),
        CodexNamespaceMember::Unknown(_) => false,
    })
}

fn namespace_container_call<'a>(
    call_name: &str,
    tools: &'a [ResponsesTool],
) -> Option<(&'a CodexNamespaceToolParam, &'a FunctionToolParam)> {
    if tools.iter().any(|tool| top_level_tool_named(tool, call_name)) {
        return None;
    }

    let mut found = None;
    let mut ambiguous = false;
    for tool in tools {
        let ResponsesTool::Namespace(namespace) = tool else {
            continue;
        };
        if call_name != namespace.name {
            continue;
        }
        if namespace_flat_name_collides(namespace, tools) {
            continue;
        }

        let mut function_members = namespace.tools.iter().filter_map(|member| match member {
            CodexNamespaceMember::Function(function) => Some(function),
            CodexNamespaceMember::Unknown(_) => None,
        });
        let Some(function) = function_members.next() else {
            continue;
        };
        if function_members.next().is_some() {
            continue;
        }

        record_unique_namespace_match(&mut found, &mut ambiguous, (namespace, function));
    }

    if ambiguous { None } else { found }
}

fn unambiguous_namespace_member_call<'a>(
    call_name: &str,
    tools: &'a [ResponsesTool],
) -> Option<(&'a CodexNamespaceToolParam, &'a FunctionToolParam)> {
    let mut found = None;
    for tool in tools {
        if top_level_tool_named(tool, call_name) {
            return None;
        }
        let ResponsesTool::Namespace(namespace) = tool else {
            continue;
        };
        if namespace_flat_name_collides(namespace, tools) {
            continue;
        }
        for member in &namespace.tools {
            let CodexNamespaceMember::Function(function) = member else {
                continue;
            };
            if function.name.as_str() != call_name {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some((namespace, function));
        }
    }
    found
}

fn strip_namespace_container_arguments(arguments: &mut String) {
    let Ok(mut value) = serde_json::from_str::<Value>(arguments) else {
        return;
    };
    let Some(object) = value.as_object_mut() else {
        return;
    };
    if object.remove("tools").is_some() {
        *arguments = serde_json::to_string(&value).unwrap_or_else(|_| std::mem::take(arguments));
    }
}

#[must_use]
pub fn flatten_tool_choice_for_upstream(choice: &ToolChoice, tools: Option<&[ResponsesTool]>) -> ToolChoice {
    let ToolChoice::Function { namespace, name } = choice else {
        return choice.clone();
    };
    let Some(tools) = tools else {
        return choice.clone();
    };

    let resolved = if let Some(namespace) = namespace {
        namespace_member_call_by_namespace(namespace, name, tools)
            .filter(|(namespace, _function)| !namespace_flat_name_collides(namespace, tools))
    } else {
        namespace_member_call(name, tools)
            .or_else(|| namespace_container_call(name, tools))
            .or_else(|| unambiguous_namespace_member_call(name, tools))
    };
    let Some((namespace, function)) = resolved else {
        return choice.clone();
    };

    ToolChoice::Function {
        namespace: None,
        name: model_visible_namespace_member_name(&namespace.name, function.name.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completed_call(name: &str, arguments: &str) -> OutputItem {
        OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_string(),
            call_id: "call_1".to_string(),
            name: name.to_string(),
            namespace: None,
            arguments: arguments.to_string(),
            status: crate::types::event::MessageStatus::Completed,
        })
    }

    #[test]
    fn function_tool_choice_flattens_unambiguous_namespace_member() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();
        let choice = ToolChoice::Function {
            namespace: None,
            name: "run".to_string(),
        };

        assert_eq!(
            flatten_tool_choice_for_upstream(&choice, Some(&tools)),
            ToolChoice::Function {
                namespace: None,
                name: "agentic_ns__mcp__shell__run".to_string()
            }
        );
    }

    #[test]
    fn namespaced_function_tool_choice_flattens_exact_member() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            },
            {
                "type": "namespace",
                "name": "mcp__git",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();
        let choice: ToolChoice = serde_json::from_value(serde_json::json!({
            "type": "function",
            "namespace": "mcp__git",
            "name": "run"
        }))
        .unwrap();

        assert_eq!(
            flatten_tool_choice_for_upstream(&choice, Some(&tools)),
            ToolChoice::Function {
                namespace: None,
                name: "agentic_ns__mcp__git__run".to_string()
            }
        );
    }

    #[test]
    fn flatten_tools_does_not_generate_colliding_namespace_member_name() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {"type": "function", "name": "agentic_ns__mcp__shell__run"},
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();

        let upstream = flatten_tools_for_upstream(Some(&tools)).expect("tools");
        let flat_function_count = upstream
            .iter()
            .filter(|tool| matches!(tool, ResponsesTool::Function(function) if function.name.as_str() == "agentic_ns__mcp__shell__run"))
            .count();

        assert_eq!(flat_function_count, 1);
        assert!(upstream.iter().any(|tool| matches!(
            tool,
            ResponsesTool::Namespace(namespace) if namespace.name == "mcp__shell"
        )));
    }

    #[test]
    fn namespace_container_call_normalizes_to_member_call() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__agentic_fixture",
                "tools": [
                    {"type": "function", "name": "run", "parameters": {"type": "object"}}
                ]
            }
        ]))
        .unwrap();
        let mut output = vec![completed_call(
            "mcp__agentic_fixture",
            "{\"tools\":\"opaque\",\"cmd\":\"echo namespace fixture\"}",
        )];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
        assert_eq!(call.name, "run");
        assert_eq!(call.arguments, "{\"cmd\":\"echo namespace fixture\"}");
    }

    #[test]
    fn flat_namespace_member_call_preserves_tools_argument() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__agentic_fixture",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();
        let mut output = vec![completed_call(
            "agentic_ns__mcp__agentic_fixture__run",
            "{\"tools\":\"legitimate\",\"cmd\":\"pwd\"}",
        )];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
        assert_eq!(call.name, "run");
        assert_eq!(call.arguments, "{\"tools\":\"legitimate\",\"cmd\":\"pwd\"}");
    }

    #[test]
    fn underscore_namespace_member_alias_normalizes_to_member_call() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__agentic_fixture",
                "tools": [{"type": "function", "name": "echo_text"}]
            }
        ]))
        .unwrap();
        let mut output = vec![completed_call("mcp__agentic_fixture_echo_text", "{\"text\":\"hi\"}")];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
        assert_eq!(call.name, "echo_text");
    }

    #[test]
    fn ambiguous_underscore_namespace_member_alias_is_not_normalized() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__a_b",
                "tools": [{"type": "function", "name": "c"}]
            },
            {
                "type": "namespace",
                "name": "mcp__a",
                "tools": [{"type": "function", "name": "b_c"}]
            }
        ]))
        .unwrap();
        let mut output = vec![completed_call("mcp__a_b_c", "{}")];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert!(call.namespace.is_none());
        assert_eq!(call.name, "mcp__a_b_c");
    }

    #[test]
    fn unambiguous_bare_namespace_member_normalizes_to_member_call() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__agentic_fixture",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();
        let mut output = vec![completed_call("run", "{\"cmd\":\"pwd\"}")];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
        assert_eq!(call.name, "run");
    }

    #[test]
    fn response_value_normalizes_nested_function_call_item() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__agentic_fixture",
                "tools": [{"type": "function", "name": "add_numbers"}]
            }
        ]))
        .unwrap();
        let mut value = serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": "agentic_ns__mcp__agentic_fixture__add_numbers",
                "call_id": "call_1",
                "arguments": "{\"numbers\":[8,0]}"
            }
        });

        assert!(normalize_response_value_with_tools(&mut value, Some(&tools)));
        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "add_numbers");
        assert_eq!(value["item"]["arguments"], "{\"numbers\":[8,0]}");
    }
}
