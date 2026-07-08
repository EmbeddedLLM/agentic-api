use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};

use crate::types::io::{FunctionTool, FunctionToolCall, OutputItem, ToolChoice};
use crate::types::tools::{
    CodexNamespaceMember, CodexNamespaceToolParam, NonEmptyToolName, RequestTool, ResponsesTool,
};

use super::handler::{ToolError, ToolHandler};
use super::registry::ToolType;

// Upstream Responses-compatible backends only see flat function names. Prefix
// flattened Codex namespace members so generated names are recognizable,
// unlikely to collide with user functions, and can be restored to
// `{ namespace, name }` on the way back to the client.
pub const MODEL_VISIBLE_NAMESPACE_MEMBER_PREFIX: &str = "agentic_ns__";

#[must_use]
pub fn model_visible_namespace_member_name(namespace: &str, member: &str) -> String {
    format!("{MODEL_VISIBLE_NAMESPACE_MEMBER_PREFIX}{namespace}__{member}")
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct NamespaceMemberName {
    namespace: String,
    name: String,
}

#[derive(Clone, Debug)]
struct NamespaceCallMapping {
    member: NamespaceMemberName,
    upstream_name: String,
}

#[derive(Clone, Debug, Default)]
struct NamespaceMap {
    calls: HashMap<String, NamespaceCallMapping>,
    members: HashMap<NamespaceMemberName, String>,
}

impl NamespaceMap {
    fn mapping_for_call(&self, name: &str) -> Option<&NamespaceCallMapping> {
        self.calls.get(name)
    }

    fn mapping_for_member(&self, namespace: &str, name: &str) -> Option<&NamespaceCallMapping> {
        let member = NamespaceMemberName {
            namespace: namespace.to_string(),
            name: name.to_string(),
        };
        self.members
            .get(&member)
            .and_then(|upstream_name| self.calls.get(upstream_name))
    }

    #[cfg(test)]
    fn contains_call(&self, upstream_name: &str) -> bool {
        self.calls.contains_key(upstream_name)
    }

    #[cfg(test)]
    fn contains_namespace_call(&self, namespace: &str) -> bool {
        self.calls.values().any(|mapping| mapping.member.namespace == namespace)
    }
}

#[derive(Clone, Debug, Default)]
pub struct RawCodexNamespaceNormalization {
    map: NamespaceMap,
    original_tools: Option<Value>,
}

impl RawCodexNamespaceNormalization {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.calls.is_empty() && self.original_tools.is_none()
    }

    #[cfg(test)]
    pub(crate) fn has_original_tools(&self) -> bool {
        self.original_tools.is_some()
    }

    #[cfg(test)]
    pub(crate) fn contains_call(&self, upstream_name: &str) -> bool {
        self.map.contains_call(upstream_name)
    }

    #[cfg(test)]
    pub(crate) fn contains_namespace_call(&self, namespace: &str) -> bool {
        self.map.contains_namespace_call(namespace)
    }
}

#[derive(Clone, Debug)]
pub struct CodexNamespaceRequestNormalization {
    pub tools: Option<Vec<ResponsesTool>>,
    pub tool_choice: ToolChoice,
}

#[derive(Default)]
struct NamespaceMapBuilder {
    top_level_names: HashSet<String>,
    map: NamespaceMap,
}

impl NamespaceMapBuilder {
    fn new(top_level_names: HashSet<String>) -> Self {
        Self {
            top_level_names,
            ..Self::default()
        }
    }

    fn namespace_has_flat_name_collision<'a>(
        &self,
        namespace_name: &str,
        member_names: impl IntoIterator<Item = &'a str>,
    ) -> bool {
        member_names.into_iter().any(|member_name| {
            self.top_level_names
                .contains(&model_visible_namespace_member_name(namespace_name, member_name))
        })
    }

    fn record_flat_member(&mut self, namespace_name: &str, member_name: &str) -> String {
        let flat_name = model_visible_namespace_member_name(namespace_name, member_name);
        let member = NamespaceMemberName {
            namespace: namespace_name.to_string(),
            name: member_name.to_string(),
        };
        let mapping = NamespaceCallMapping {
            member: member.clone(),
            upstream_name: flat_name.clone(),
        };

        self.map.members.insert(member, flat_name.clone());
        self.map.calls.insert(flat_name.clone(), mapping);
        flat_name
    }

    fn finish(self) -> NamespaceMap {
        self.map
    }
}

/// Handler for Codex `type: "namespace"` tools.
///
/// Namespace tools are client-owned, like plain function tools, but need a
/// request-scoped normalization pass to flatten members into model-visible
/// function names and restore model calls back to the public namespace shape.
#[derive(Debug)]
pub struct CodexNamespaceHandler;

impl CodexNamespaceHandler {
    #[must_use]
    pub fn normalize_request_for_upstream(
        &self,
        tools: Option<&[RequestTool]>,
        tool_choice: &ToolChoice,
    ) -> CodexNamespaceRequestNormalization {
        let Some(tools) = tools else {
            return CodexNamespaceRequestNormalization {
                tools: None,
                tool_choice: tool_choice.clone(),
            };
        };

        let mut builder = NamespaceMapBuilder::new(typed_top_level_tool_names(tools));
        let tools = flatten_typed_tools_with_builder(tools, &mut builder);
        let map = builder.finish();

        CodexNamespaceRequestNormalization {
            tools: Some(tools),
            tool_choice: rewrite_tool_choice_with_map(tool_choice, &map),
        }
    }

    pub fn restore_output_items(&self, output: &mut [OutputItem], tools: Option<&[RequestTool]>) {
        let Some(map) = namespace_map_from_tools(tools) else {
            return;
        };
        for item in output {
            if let OutputItem::FunctionCall(call) = item {
                restore_function_call_with_map(call, &map);
            }
        }
    }

    #[must_use]
    pub fn restore_response_value(&self, value: &mut Value, tools: Option<&[RequestTool]>) -> bool {
        let Some(map) = namespace_map_from_tools(tools) else {
            return false;
        };
        restore_response_value_with_map(value, &map)
    }

    #[must_use]
    pub fn flatten_raw_tools_for_upstream(
        &self,
        object: &mut Map<String, Value>,
        normalization: &mut RawCodexNamespaceNormalization,
    ) -> bool {
        let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut) else {
            return false;
        };

        let original_tools = tools.clone();
        let mut builder = NamespaceMapBuilder::new(raw_top_level_tool_names(tools));
        let mut upstream_tools = Vec::with_capacity(tools.len());
        let mut changed = false;

        for tool in std::mem::take(tools) {
            let Some(tool_object) = tool.as_object() else {
                upstream_tools.push(tool);
                continue;
            };
            if tool_object.get("type").and_then(Value::as_str) != Some("namespace") {
                upstream_tools.push(tool);
                continue;
            }
            let Some(namespace_name) = tool_object.get("name").and_then(Value::as_str) else {
                upstream_tools.push(tool);
                continue;
            };
            let Some(members) = tool_object.get("tools").and_then(Value::as_array) else {
                upstream_tools.push(tool);
                continue;
            };

            let function_member_names = raw_function_member_names(members);
            if function_member_names.is_empty() {
                upstream_tools.push(tool);
                continue;
            }
            if builder
                .namespace_has_flat_name_collision(namespace_name, function_member_names.iter().map(String::as_str))
            {
                tracing::debug!(
                    namespace = %namespace_name,
                    "leaving raw namespace tool unflattened because a top-level tool uses a generated name"
                );
                upstream_tools.push(tool);
                continue;
            }

            let mut emitted_namespace_members = false;
            for member in members
                .iter()
                .filter(|member| member.get("type").and_then(Value::as_str) == Some("function"))
            {
                let Some(member_name) = member.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let flat_name = builder.record_flat_member(namespace_name, member_name);
                tracing::debug!(
                    namespace = %namespace_name,
                    member = %member_name,
                    upstream_name = %flat_name,
                    "flattened raw namespace tool member for upstream"
                );
                let mut upstream_member = member.clone();
                if let Some(member_object) = upstream_member.as_object_mut() {
                    member_object.insert("name".to_string(), Value::String(flat_name));
                }
                upstream_tools.push(upstream_member);
                emitted_namespace_members = true;
                changed = true;
            }

            if !emitted_namespace_members {
                upstream_tools.push(tool);
            }
        }

        if changed {
            normalization.map = builder.finish();
            normalization.original_tools = Some(Value::Array(original_tools));
            *tools = upstream_tools;
        } else {
            *tools = original_tools;
        }
        changed
    }

    #[must_use]
    pub fn rewrite_raw_tool_choice_for_upstream(
        &self,
        object: &mut Map<String, Value>,
        normalization: &RawCodexNamespaceNormalization,
    ) -> bool {
        let Some(tool_choice) = object.get_mut("tool_choice") else {
            return false;
        };
        let Some(choice_object) = tool_choice.as_object_mut() else {
            return false;
        };

        if choice_object.get("type").and_then(Value::as_str) == Some("function") {
            return rewrite_raw_tool_choice_function_object(choice_object, &normalization.map);
        }

        choice_object
            .get_mut("function")
            .and_then(Value::as_object_mut)
            .is_some_and(|function| rewrite_raw_tool_choice_function_object(function, &normalization.map))
    }

    #[must_use]
    pub fn restore_raw_response_value(
        &self,
        value: &mut Value,
        normalization: &RawCodexNamespaceNormalization,
    ) -> bool {
        let mut changed = restore_raw_original_tools(value, normalization);
        changed |= restore_response_value_with_map(value, &normalization.map);
        changed
    }

    #[must_use]
    pub fn restore_raw_sse_line(&self, line: &str, normalization: &RawCodexNamespaceNormalization) -> String {
        let Some(data) = line.strip_prefix("data: ") else {
            return line.to_string();
        };
        if data == "[DONE]" {
            return line.to_string();
        }
        let Ok(mut value) = serde_json::from_str::<Value>(data) else {
            return line.to_string();
        };
        if !self.restore_raw_response_value(&mut value, normalization) {
            return line.to_string();
        }
        serde_json::to_string(&value).map_or_else(|_| line.to_string(), |json| format!("data: {json}"))
    }
}

impl ToolHandler for CodexNamespaceHandler {
    fn tool_type(&self) -> ToolType {
        ToolType::CodexNamespace
    }

    fn validate(&self, param: &Value) -> Result<(), ToolError> {
        serde_json::from_value::<CodexNamespaceToolParam>(param.clone())
            .map(|_| ())
            .map_err(|e| ToolError::Config(format!("invalid codex namespace tool config: {e}")))
    }

    fn normalize(&self, param: &Value) -> Vec<FunctionTool> {
        let Ok(namespace) = serde_json::from_value::<CodexNamespaceToolParam>(param.clone()) else {
            tracing::warn!("normalize() called with invalid codex namespace param - validate() must be called first");
            return vec![];
        };
        let tools = vec![RequestTool::Namespace(namespace)];
        self.normalize_request_for_upstream(Some(&tools), &ToolChoice::Auto)
            .tools
            .unwrap_or_default()
            .iter()
            .filter_map(ResponsesTool::to_function_tool)
            .collect()
    }
}

impl NamespaceMapBuilder {
    fn with_typed_tools(mut self, tools: &[RequestTool]) -> Self {
        let _ = flatten_typed_tools_with_builder(tools, &mut self);
        self
    }
}

fn namespace_map_from_tools(tools: Option<&[RequestTool]>) -> Option<NamespaceMap> {
    let tools = tools?;
    Some(
        NamespaceMapBuilder::new(typed_top_level_tool_names(tools))
            .with_typed_tools(tools)
            .finish(),
    )
}

fn flatten_typed_tools_with_builder(tools: &[RequestTool], builder: &mut NamespaceMapBuilder) -> Vec<ResponsesTool> {
    let mut upstream_tools = Vec::with_capacity(tools.len());
    for tool in tools {
        match tool {
            RequestTool::Namespace(namespace) => {
                flatten_typed_namespace_tool(namespace, builder, &mut upstream_tools);
            }
            RequestTool::Function(function) => upstream_tools.push(ResponsesTool::Function(function.clone())),
            RequestTool::Mcp(mcp) => upstream_tools.push(ResponsesTool::Mcp(mcp.clone())),
            RequestTool::WebSearch(web_search) => upstream_tools.push(ResponsesTool::WebSearch(web_search.clone())),
            RequestTool::FileSearch(file_search) => upstream_tools.push(ResponsesTool::FileSearch(file_search.clone())),
            RequestTool::CodeInterpreter(code_interpreter) => {
                upstream_tools.push(ResponsesTool::CodeInterpreter(code_interpreter.clone()));
            }
            RequestTool::Unknown => {}
        }
    }
    upstream_tools
}

fn flatten_typed_namespace_tool(
    namespace: &CodexNamespaceToolParam,
    builder: &mut NamespaceMapBuilder,
    upstream_tools: &mut Vec<ResponsesTool>,
) {
    let function_member_names = typed_function_member_names(namespace);
    if function_member_names.is_empty() {
        tracing::debug!(
            namespace = %namespace.name,
            "skipping namespace tool for upstream because it has no function members"
        );
        return;
    }
    if builder.namespace_has_flat_name_collision(&namespace.name, function_member_names.iter().map(String::as_str)) {
        tracing::debug!(
            namespace = %namespace.name,
            "skipping namespace tool for upstream because a top-level tool uses a generated name"
        );
        return;
    }

    let mut emitted_members = false;
    for member in &namespace.tools {
        let CodexNamespaceMember::Function(function) = member else {
            continue;
        };
        let flat_name_text = builder.record_flat_member(&namespace.name, function.name.as_str());
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

    if !emitted_members {
        tracing::debug!(namespace = %namespace.name, "skipping namespace tool for upstream because no valid members emitted");
    }
}

fn typed_top_level_tool_names(tools: &[RequestTool]) -> HashSet<String> {
    tools
        .iter()
        .filter_map(|tool| match tool {
            RequestTool::Function(function) => Some(function.name.as_str().to_string()),
            RequestTool::Mcp(_)
            | RequestTool::WebSearch(_)
            | RequestTool::FileSearch(_)
            | RequestTool::CodeInterpreter(_)
            | RequestTool::Namespace(_)
            | RequestTool::Unknown => None,
        })
        .collect()
}

fn raw_top_level_tool_names(tools: &[Value]) -> HashSet<String> {
    tools
        .iter()
        .filter_map(|tool| {
            let object = tool.as_object()?;
            if object.get("type").and_then(Value::as_str) == Some("namespace") {
                return None;
            }
            object.get("name").and_then(Value::as_str).map(str::to_string)
        })
        .collect()
}

fn typed_function_member_names(namespace: &CodexNamespaceToolParam) -> Vec<String> {
    namespace
        .tools
        .iter()
        .filter_map(|member| match member {
            CodexNamespaceMember::Function(function) => Some(function.name.as_str().to_string()),
            CodexNamespaceMember::Unknown => None,
        })
        .collect()
}

fn raw_function_member_names(members: &[Value]) -> Vec<String> {
    members
        .iter()
        .filter(|member| member.get("type").and_then(Value::as_str) == Some("function"))
        .filter_map(|member| member.get("name").and_then(Value::as_str).map(str::to_string))
        .collect()
}

fn restore_function_call_with_map(call: &mut FunctionToolCall, map: &NamespaceMap) -> bool {
    if call.namespace.is_some() {
        return false;
    }
    let Some(mapping) = map.mapping_for_call(&call.name) else {
        return false;
    };
    let original_name = call.name.clone();

    call.namespace = Some(mapping.member.namespace.clone());
    call.name.clone_from(&mapping.member.name);
    tracing::debug!(
        upstream_name = %original_name,
        namespace = %mapping.member.namespace,
        member = %mapping.member.name,
        "restored upstream namespace function call"
    );
    true
}

fn rewrite_tool_choice_with_map(choice: &ToolChoice, map: &NamespaceMap) -> ToolChoice {
    let ToolChoice::Function { namespace, name } = choice else {
        return choice.clone();
    };
    let mapping = namespace
        .as_deref()
        .and_then(|namespace| map.mapping_for_member(namespace, name))
        .or_else(|| namespace.is_none().then(|| map.mapping_for_call(name)).flatten());
    let Some(mapping) = mapping else {
        return choice.clone();
    };

    ToolChoice::Function {
        namespace: None,
        name: mapping.upstream_name.clone(),
    }
}

fn restore_response_value_with_map(value: &mut Value, map: &NamespaceMap) -> bool {
    let mut changed = false;

    if let Some(item) = value.as_object_mut().and_then(|object| object.get_mut("item")) {
        changed |= restore_call_value_with_map(item, map);
    }

    changed |= restore_call_value_with_map(value, map);

    for key in ["response", "payload"] {
        if let Some(nested) = value.as_object_mut().and_then(|object| object.get_mut(key)) {
            changed |= restore_response_value_with_map(nested, map);
        }
    }

    if let Some(Value::Array(items)) = value.as_object_mut().and_then(|object| object.get_mut("output")) {
        for item in items {
            changed |= restore_call_value_with_map(item, map);
        }
    }

    changed
}

fn restore_call_value_with_map(value: &mut Value, map: &NamespaceMap) -> bool {
    let Some(object) = value.as_object_mut() else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) != Some("function_call") {
        return false;
    }
    if object.get("namespace").and_then(Value::as_str).is_some() {
        return false;
    }
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return false;
    };
    let Some(mapping) = map.mapping_for_call(name) else {
        return false;
    };
    let original_name = name.to_string();

    object.insert("namespace".to_string(), Value::String(mapping.member.namespace.clone()));
    object.insert("name".to_string(), Value::String(mapping.member.name.clone()));
    tracing::debug!(
        upstream_name = %original_name,
        namespace = %mapping.member.namespace,
        member = %mapping.member.name,
        "restored upstream namespace function call"
    );
    true
}

fn rewrite_raw_tool_choice_function_object(function: &mut Map<String, Value>, map: &NamespaceMap) -> bool {
    let explicit_namespace = function.get("namespace").and_then(Value::as_str).map(str::to_string);
    let Some(name_text) = function.get("name").and_then(Value::as_str).map(str::to_string) else {
        return false;
    };
    let mapping = explicit_namespace
        .as_deref()
        .and_then(|namespace| map.mapping_for_member(namespace, &name_text))
        .or_else(|| {
            explicit_namespace
                .is_none()
                .then(|| map.mapping_for_call(&name_text))
                .flatten()
        });
    let Some(mapping) = mapping else {
        return false;
    };

    let mut changed = mapping.upstream_name != name_text;
    if changed {
        function.insert("name".to_string(), Value::String(mapping.upstream_name.clone()));
    }
    if explicit_namespace.is_some() {
        changed |= function.remove("namespace").is_some();
    }
    changed
}

fn restore_raw_original_tools(value: &mut Value, normalization: &RawCodexNamespaceNormalization) -> bool {
    let Some(original_tools) = &normalization.original_tools else {
        return false;
    };
    let Some(object) = value.as_object_mut() else {
        return false;
    };
    if !object.get("tools").is_some_and(Value::is_array) {
        return false;
    }
    object.insert("tools".to_string(), original_tools.clone());
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::event::MessageStatus;

    fn completed_call(name: &str, arguments: &str) -> OutputItem {
        OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_string(),
            call_id: "call_1".to_string(),
            name: name.to_string(),
            namespace: None,
            arguments: arguments.to_string(),
            status: MessageStatus::Completed,
        })
    }

    #[test]
    fn unqualified_function_tool_choice_is_not_rewritten_to_namespace_member() {
        let tools: Vec<RequestTool> = serde_json::from_value(serde_json::json!([
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

        let normalized = CodexNamespaceHandler.normalize_request_for_upstream(Some(&tools), &choice);

        assert_eq!(
            normalized.tool_choice,
            ToolChoice::Function {
                namespace: None,
                name: "run".to_string()
            }
        );
        assert!(matches!(
            normalized.tools.as_deref(),
            Some([ResponsesTool::Function(function)]) if function.name.as_str() == "agentic_ns__mcp__shell__run"
        ));
    }

    #[test]
    fn namespaced_function_tool_choice_flattens_exact_member() {
        let tools: Vec<RequestTool> = serde_json::from_value(serde_json::json!([
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

        let normalized = CodexNamespaceHandler.normalize_request_for_upstream(Some(&tools), &choice);

        assert_eq!(
            normalized.tool_choice,
            ToolChoice::Function {
                namespace: None,
                name: "agentic_ns__mcp__git__run".to_string()
            }
        );
    }

    #[test]
    fn flatten_tools_does_not_generate_colliding_namespace_member_name() {
        let tools: Vec<RequestTool> = serde_json::from_value(serde_json::json!([
            {"type": "function", "name": "agentic_ns__mcp__shell__run"},
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();

        let upstream = CodexNamespaceHandler
            .normalize_request_for_upstream(Some(&tools), &ToolChoice::Auto)
            .tools
            .expect("tools");
        let flat_function_count = upstream
            .iter()
            .filter(|tool| matches!(tool, ResponsesTool::Function(function) if function.name.as_str() == "agentic_ns__mcp__shell__run"))
            .count();

        assert_eq!(upstream.len(), 1);
        assert_eq!(flat_function_count, 1);
    }

    #[test]
    fn flat_namespace_member_call_preserves_tools_argument() {
        let tools: Vec<RequestTool> = serde_json::from_value(serde_json::json!([
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

        CodexNamespaceHandler.restore_output_items(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
        assert_eq!(call.name, "run");
        assert_eq!(call.arguments, "{\"tools\":\"legitimate\",\"cmd\":\"pwd\"}");
    }

    #[test]
    fn plain_function_call_round_trip() {
        let tools: Vec<RequestTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"}
            }
        ]))
        .unwrap();
        let upstream = CodexNamespaceHandler
            .normalize_request_for_upstream(Some(&tools), &ToolChoice::Auto)
            .tools
            .expect("tools");
        let mut output = vec![completed_call("get_weather", "{\"city\":\"SF\"}")];

        CodexNamespaceHandler.restore_output_items(&mut output, Some(&tools));

        assert!(matches!(
            upstream.as_slice(),
            [ResponsesTool::Function(function)] if function.name.as_str() == "get_weather"
        ));
        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert!(call.namespace.is_none());
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, "{\"city\":\"SF\"}");
    }

    #[test]
    fn response_value_normalizes_nested_function_call_item() {
        let tools: Vec<RequestTool> = serde_json::from_value(serde_json::json!([
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

        assert!(CodexNamespaceHandler.restore_response_value(&mut value, Some(&tools)));
        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "add_numbers");
        assert_eq!(value["item"]["arguments"], "{\"numbers\":[8,0]}");
    }
}
