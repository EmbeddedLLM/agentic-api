use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};

use crate::types::io::{FunctionTool, FunctionToolCall, OutputItem, ToolChoice};
use crate::types::tools::{CodexNamespaceMember, CodexNamespaceToolParam, NonEmptyToolName, ResponsesTool};

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

/// A pre-built, reusable namespace rename map, computed once per request from
/// the declared tools via [`CodexNamespaceHandler::build_namespace_map`].
///
/// Passing this into [`CodexNamespaceHandler::restore_output_items`],
/// [`CodexNamespaceHandler::restore_response_value`], and
/// [`CodexNamespaceHandler::resolve_tool_choice`] avoids
/// rebuilding the map on every call — important for streaming responses,
/// which call the restore path once per SSE line.
#[derive(Clone, Debug, Default)]
pub struct NamespaceMap {
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

#[derive(Default)]
struct NamespaceMapBuilder {
    top_level_names: HashSet<String>,
    map: NamespaceMap,
    #[cfg(test)]
    flat_name_collisions: usize,
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
        if let Some(existing) = self.map.calls.get(&flat_name) {
            if existing.member != member {
                #[cfg(test)]
                {
                    self.flat_name_collisions += 1;
                }
                tracing::warn!(
                    upstream_name = %flat_name,
                    namespace = %namespace_name,
                    member = %member_name,
                    existing_namespace = %existing.member.namespace,
                    existing_member = %existing.member.name,
                    "generated codex namespace member name collides with another namespace member"
                );
            }
        }
        let mapping = NamespaceCallMapping {
            member: member.clone(),
            upstream_name: flat_name.clone(),
        };

        self.map.members.insert(member, flat_name.clone());
        self.map.calls.insert(flat_name.clone(), mapping);
        flat_name
    }

    #[cfg(test)]
    fn flat_name_collision_count(&self) -> usize {
        self.flat_name_collisions
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
    /// Rewrites every `Namespace` tool's function members to their flat,
    /// model-visible names (see [`model_visible_namespace_member_name`]),
    /// given real collision detection against sibling top-level tool names.
    ///
    /// Tools stay `ResponsesTool::Namespace` — only the nested members'
    /// `name` fields change — so [`ResponsesTool::to_function_tools`] and
    /// [`super::registry::ToolRegistry::build_with_handlers`] can read each
    /// member's already-flat name directly, with no further namespace logic.
    ///
    /// A namespace whose flat member names would collide with a declared
    /// top-level tool name is left unrenamed (its members keep colliding
    /// with each other under one upstream name, so it's dropped downstream
    /// instead — see [`rename_namespace_members`]).
    #[must_use]
    pub fn resolve_namespace_members(&self, tools: &[ResponsesTool]) -> Vec<ResponsesTool> {
        let mut builder = NamespaceMapBuilder::new(typed_top_level_tool_names(tools));
        tools
            .iter()
            .map(|tool| match tool {
                ResponsesTool::Namespace(namespace) => {
                    ResponsesTool::Namespace(rename_namespace_members(namespace, &mut builder))
                }
                other => other.clone(),
            })
            .collect()
    }

    /// Builds a [`NamespaceMap`] once from a request's declared tools, for
    /// reuse across every subsequent restore/rewrite call on that request —
    /// see [`NamespaceMap`]'s docs for why this matters.
    #[must_use]
    pub fn build_namespace_map(&self, tools: Option<&[ResponsesTool]>) -> Option<NamespaceMap> {
        namespace_map_from_tools(tools)
    }

    /// Resolves the request's `tool_choice` (defaulting to `ToolChoice::Auto`
    /// when absent) and, if it's a namespaced `ToolChoice::Function {
    /// namespace, name }`, rewrites it to the flattened, model-visible name
    /// that [`ResponsesTool::to_function_tools`] produces for the matching
    /// namespace member — so `tool_choice` agrees with the tool names
    /// actually sent upstream.
    ///
    /// A no-op for `ToolChoice` variants other than `Function`, and for a
    /// `Function` choice that doesn't match any declared namespace member.
    #[must_use]
    pub fn resolve_tool_choice(&self, map: Option<&NamespaceMap>, tool_choice: Option<&ToolChoice>) -> ToolChoice {
        let tool_choice = tool_choice.unwrap_or(&ToolChoice::Auto);
        let Some(map) = map else {
            return tool_choice.clone();
        };
        rewrite_tool_choice_with_map(tool_choice, map)
    }

    pub fn restore_output_items(&self, output: &mut [OutputItem], map: Option<&NamespaceMap>) {
        let Some(map) = map else {
            return;
        };
        for item in output {
            if let OutputItem::FunctionCall(call) = item {
                restore_function_call_with_map(call, map);
            }
        }
    }

    #[must_use]
    pub fn restore_response_value(&self, value: &mut Value, map: Option<&NamespaceMap>) -> bool {
        let Some(map) = map else {
            return false;
        };
        restore_response_value_with_map(value, map)
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

    /// Converts an already-renamed namespace's function members straight to
    /// `FunctionTool`s. Callers must rename members to their flat, model-visible
    /// names first via [`CodexNamespaceHandler::rename_members_for_upstream`] —
    /// this method has no sibling-tool context to do that itself.
    fn normalize(&self, param: &Value) -> Vec<FunctionTool> {
        let Ok(namespace) = serde_json::from_value::<CodexNamespaceToolParam>(param.clone()) else {
            tracing::warn!("normalize() called with invalid codex namespace param - validate() must be called first");
            return vec![];
        };
        namespace
            .tools
            .iter()
            .filter_map(|member| match member {
                CodexNamespaceMember::Function(function) => Some(FunctionTool::from(function)),
                CodexNamespaceMember::Unknown => None,
            })
            .collect()
    }
}

fn namespace_map_from_tools(tools: Option<&[ResponsesTool]>) -> Option<NamespaceMap> {
    let tools = tools?;
    let mut builder = NamespaceMapBuilder::new(typed_top_level_tool_names(tools));
    for tool in tools {
        if let ResponsesTool::Namespace(namespace) = tool {
            let _ = rename_namespace_members(namespace, &mut builder);
        }
    }
    Some(builder.finish())
}

/// Returns `namespace` with its function members' names rewritten to their
/// flat, model-visible form, recording each rename in `builder` along the
/// way. On a flat-name collision with a declared top-level tool, members are
/// left unrenamed (the namespace still has a non-empty `tools` list, so
/// callers must not assume renaming always changes names).
fn rename_namespace_members(
    namespace: &CodexNamespaceToolParam,
    builder: &mut NamespaceMapBuilder,
) -> CodexNamespaceToolParam {
    let function_member_names = typed_function_member_names(namespace);
    if function_member_names.is_empty() {
        tracing::debug!(
            namespace = %namespace.name,
            "namespace tool has no function members to rename for upstream"
        );
        return namespace.clone();
    }
    if builder.namespace_has_flat_name_collision(&namespace.name, function_member_names.iter().map(String::as_str)) {
        tracing::debug!(
            namespace = %namespace.name,
            "leaving namespace tool members unrenamed because a top-level tool uses a generated name"
        );
        return namespace.clone();
    }

    let tools = namespace
        .tools
        .iter()
        .map(|member| {
            let CodexNamespaceMember::Function(function) = member else {
                return member.clone();
            };
            let flat_name_text = builder.record_flat_member(&namespace.name, function.name.as_str());
            let Ok(flat_name) = NonEmptyToolName::try_from(flat_name_text.clone()) else {
                return member.clone();
            };
            tracing::debug!(
                namespace = %namespace.name,
                member = %function.name.as_str(),
                upstream_name = %flat_name_text,
                "renamed namespace tool member for upstream"
            );
            let mut function = function.clone();
            function.name = flat_name;
            CodexNamespaceMember::Function(function)
        })
        .collect();

    CodexNamespaceToolParam {
        tools,
        ..namespace.clone()
    }
}

fn typed_top_level_tool_names(tools: &[ResponsesTool]) -> HashSet<String> {
    tools
        .iter()
        .filter_map(|tool| match tool {
            ResponsesTool::Function(function) => Some(function.name.as_str().to_string()),
            ResponsesTool::Mcp(_)
            | ResponsesTool::WebSearch(_)
            | ResponsesTool::FileSearch(_)
            | ResponsesTool::CodeInterpreter(_)
            | ResponsesTool::Namespace(_)
            | ResponsesTool::Unknown => None,
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

        let map = CodexNamespaceHandler.build_namespace_map(Some(&tools));
        let rewritten = CodexNamespaceHandler.resolve_tool_choice(map.as_ref(), Some(&choice));

        assert_eq!(
            rewritten,
            ToolChoice::Function {
                namespace: None,
                name: "run".to_string()
            }
        );
        let resolved = CodexNamespaceHandler.resolve_namespace_members(&tools);
        assert!(matches!(
            resolved.as_slice(),
            [ResponsesTool::Namespace(namespace)]
                if matches!(&namespace.tools[0], CodexNamespaceMember::Function(f) if f.name.as_str() == "agentic_ns__mcp__shell__run")
        ));
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

        let map = CodexNamespaceHandler.build_namespace_map(Some(&tools));
        let rewritten = CodexNamespaceHandler.resolve_tool_choice(map.as_ref(), Some(&choice));

        assert_eq!(
            rewritten,
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

        let resolved = CodexNamespaceHandler.resolve_namespace_members(&tools);
        let flat_function_count = resolved
            .iter()
            .filter(|tool| matches!(tool, ResponsesTool::Function(function) if function.name.as_str() == "agentic_ns__mcp__shell__run"))
            .count();
        let ResponsesTool::Namespace(namespace) = &resolved[1] else {
            panic!("expected namespace tool");
        };

        // The namespace's member keeps its bare name — it isn't renamed to
        // the colliding flat name already used by the top-level function.
        assert_eq!(flat_function_count, 1);
        assert!(matches!(&namespace.tools[0], CodexNamespaceMember::Function(f) if f.name.as_str() == "run"));
    }

    #[test]
    fn namespace_map_builder_detects_flat_name_collision_between_namespace_members() {
        let mut builder = NamespaceMapBuilder::new(HashSet::new());

        assert_eq!(builder.record_flat_member("a__b", "c"), "agentic_ns__a__b__c");
        assert_eq!(builder.flat_name_collision_count(), 0);
        assert_eq!(builder.record_flat_member("a", "b__c"), "agentic_ns__a__b__c");
        assert_eq!(builder.flat_name_collision_count(), 1);
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

        let map = CodexNamespaceHandler.build_namespace_map(Some(&tools));
        CodexNamespaceHandler.restore_output_items(&mut output, map.as_ref());

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
        assert_eq!(call.name, "run");
        assert_eq!(call.arguments, "{\"tools\":\"legitimate\",\"cmd\":\"pwd\"}");
    }

    #[test]
    fn plain_function_call_round_trip() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"}
            }
        ]))
        .unwrap();
        let resolved = CodexNamespaceHandler.resolve_namespace_members(&tools);
        let mut output = vec![completed_call("get_weather", "{\"city\":\"SF\"}")];

        let map = CodexNamespaceHandler.build_namespace_map(Some(&tools));
        CodexNamespaceHandler.restore_output_items(&mut output, map.as_ref());

        assert!(matches!(
            resolved.as_slice(),
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

        let map = CodexNamespaceHandler.build_namespace_map(Some(&tools));
        assert!(CodexNamespaceHandler.restore_response_value(&mut value, map.as_ref()));
        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "add_numbers");
        assert_eq!(value["item"]["arguments"], "{\"numbers\":[8,0]}");
    }
}
