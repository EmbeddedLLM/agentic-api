use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::codex::insert_namespace_entries;
use super::executors::GatewayExecutors;
use super::function::insert_function_entry;
use super::mcp::{insert_mcp_entry, maybe_mcp_function};
use super::tool_search;
use super::web_search::insert_web_search_entry;
use super::{CodexNamespaceHandler, GatewayExecutor, NamespaceMap, ToolError, ToolOutput};
use crate::types::io::OutputItem;
use crate::types::io::output::FunctionToolCall;
use crate::types::tools::{CodeInterpreterToolParam, FileSearchToolParam, ResponsesTool, ToolSearchExecution};
use crate::utils::common::serialize_to_value_or_custom_default;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
    CodexNamespace,
    Mcp,
    /// Internal routing discriminant. Serializes as `"web_search"`.
    /// Note: the corresponding `ResponsesTool` wire tag is `"web_search_preview"`.
    /// `ToolType` is not used in wire-facing types so the names differ intentionally.
    WebSearch,
    FileSearch,
    CodeInterpreter,
}

impl ToolType {
    #[must_use]
    pub(crate) const fn description(self) -> &'static str {
        match self {
            Self::Function => "function tool",
            Self::CodexNamespace => "Codex namespace tool",
            Self::Mcp => "MCP tool",
            Self::WebSearch => "web search tool",
            Self::FileSearch => "file search tool",
            Self::CodeInterpreter => "code interpreter tool",
        }
    }

    #[must_use]
    pub const fn is_gateway_owned(self) -> bool {
        !matches!(self, Self::Function | Self::CodexNamespace)
    }
}

/// Per-request routing entry keyed by the tool name the model will call.
#[derive(Clone)]
pub struct ToolEntry {
    pub tool_type: ToolType,
    /// Full serialised tool param for the executor (used during dispatch).
    pub config: Value,
    /// For MCP tools: which server this tool belongs to.
    pub server_label: Option<String>,
    pub handler: Option<Arc<dyn GatewayExecutor>>,
}

impl std::fmt::Debug for ToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolEntry")
            .field("tool_type", &self.tool_type)
            .field("config", &self.config)
            .field("server_label", &self.server_label)
            .field("handler", &self.handler.is_some())
            .finish()
    }
}

pub struct GatewayDispatchResult {
    pub tool_type: ToolType,
    pub output: Result<ToolOutput, ToolError>,
}

// TODO: move to a dedicated file_search module alongside its `ToolHandler`
// once file_search execution is implemented.
fn insert_file_search_entry(
    entries: &mut HashMap<String, ToolEntry>,
    p: &FileSearchToolParam,
    handler: Option<Arc<dyn GatewayExecutor>>,
) {
    serialize_to_value_or_custom_default(
        p,
        "file_search tool config serialization failed",
        |config| {
            entries.insert(
                "file_search".to_owned(),
                ToolEntry {
                    tool_type: ToolType::FileSearch,
                    config,
                    server_label: None,
                    handler,
                },
            );
        },
        (),
    );
}

// TODO: move to a dedicated code_interpreter module alongside its `ToolHandler`
// once code_interpreter execution is implemented.
fn insert_code_interpreter_entry(
    entries: &mut HashMap<String, ToolEntry>,
    p: &CodeInterpreterToolParam,
    handler: Option<Arc<dyn GatewayExecutor>>,
) {
    serialize_to_value_or_custom_default(
        p,
        "code_interpreter tool config serialization failed",
        |config| {
            entries.insert(
                "code_interpreter".to_owned(),
                ToolEntry {
                    tool_type: ToolType::CodeInterpreter,
                    config,
                    server_label: None,
                    handler,
                },
            );
        },
        (),
    );
}

/// Request-scoped registry built from `RequestPayload.tools`.
/// Maps the name the LLM sees → routing metadata.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    entries: HashMap<String, ToolEntry>,
    /// Built once from the declared tools, so `restore_final_payload_output`
    /// and `restore_stream_event_value` — the latter called once per SSE line
    /// during streaming — don't rebuild it on every call.
    namespace_map: Option<NamespaceMap>,
    client_tool_search: bool,
    loaded_tool_namespaces: HashMap<String, String>,
    tool_search_name_owned: bool,
}

impl ToolRegistry {
    /// Build a registry from declared tools and attach gateway handlers for dispatchable tool types.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when Codex namespace member flattening
    /// would collide with another declared tool name.
    ///
    /// # Panics
    ///
    /// Panics if serialization of a tool param struct fails, which cannot happen
    /// for the types defined in this module (`#[derive(Serialize)]` on plain structs).
    pub async fn build_with_handlers(tools: &[ResponsesTool], executors: &GatewayExecutors) -> Result<Self, ToolError> {
        let mut entries = HashMap::with_capacity(tools.len());
        // Namespace members must be keyed by the same flat, model-visible name
        // the model will call, so resolve them first — the same pure pass used
        // to build the upstream request.
        let resolved_tools = CodexNamespaceHandler.resolve_namespace_members(tools)?;

        for tool in &resolved_tools {
            match tool {
                ResponsesTool::Function(p) => match maybe_mcp_function(p) {
                    Some(mcp_params) if !mcp_params.is_empty() => {
                        let handler = executors.mcp_read_resource_handler(&mcp_params).await;
                        for declaration_param in &mcp_params {
                            insert_mcp_entry(&mut entries, declaration_param, handler.clone()).await;
                        }
                    }
                    _ => insert_function_entry(&mut entries, p),
                },
                ResponsesTool::Mcp(p) => {
                    let handler = executors.mcp_handler(p).await;
                    insert_mcp_entry(&mut entries, p, handler).await;
                }
                ResponsesTool::WebSearch(p) => {
                    insert_web_search_entry(&mut entries, p, executors.web_search_handler());
                }
                ResponsesTool::FileSearch(p) => insert_file_search_entry(&mut entries, p, None),
                ResponsesTool::CodeInterpreter(p) => insert_code_interpreter_entry(&mut entries, p, None),
                ResponsesTool::Namespace(p) => insert_namespace_entries(&mut entries, p),
                ResponsesTool::Custom(p) => {
                    tracing::debug!(name = %p.name, "client-owned custom tool skipped in function registry");
                }
                ResponsesTool::ToolSearch(p) => {
                    tracing::debug!(
                        execution = ?p.execution,
                        "tool_search skipped in function registry"
                    );
                }
                ResponsesTool::Unknown => {
                    tracing::debug!("unknown tool declared but skipped in registry");
                }
            }
        }

        let namespace_map = CodexNamespaceHandler.build_namespace_map((!tools.is_empty()).then_some(tools))?;
        let client_tool_search = tools.iter().any(|tool| {
            matches!(
                tool,
                ResponsesTool::ToolSearch(search)
                    if search.execution == Some(ToolSearchExecution::Client)
            )
        });
        let tool_search_name_owned = entries.contains_key(tool_search::TOOL_SEARCH_NAME);

        Ok(Self {
            entries,
            namespace_map,
            client_tool_search,
            loaded_tool_namespaces: HashMap::new(),
            tool_search_name_owned,
        })
    }

    #[must_use]
    pub fn lookup(&self, tool_name: &str) -> Option<&ToolEntry> {
        self.entries.get(tool_name)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn restore_final_payload_output(&self, output: &mut [OutputItem]) {
        CodexNamespaceHandler.restore_output_items(output, self.namespace_map.as_ref());
        tool_search::restore_loaded_namespace_output_items(output, &self.loaded_tool_namespaces);
        tool_search::restore_output_items(output, self.can_restore_tool_search_fallback());
    }

    pub fn restore_stream_event_value(&self, value: &mut Value) -> bool {
        let mut changed = CodexNamespaceHandler.restore_response_value(value, self.namespace_map.as_ref());
        changed |= tool_search::restore_loaded_namespace_response_value(value, &self.loaded_tool_namespaces);
        changed |= tool_search::restore_response_value(value, self.can_restore_tool_search_fallback());
        changed
    }

    #[must_use]
    pub(crate) const fn can_restore_tool_search_fallback(&self) -> bool {
        self.client_tool_search && !self.tool_search_name_owned
    }

    pub(crate) fn load_tool_search_output(&mut self, input: &crate::types::io::ResponsesInput) {
        self.loaded_tool_namespaces = tool_search::loaded_namespace_members(input);
        self.loaded_tool_namespaces
            .retain(|name, _| !self.entries.contains_key(name));
        self.tool_search_name_owned |=
            tool_search::loaded_function_names(input).contains(tool_search::TOOL_SEARCH_NAME);
    }

    /// Returns the subset of `calls` whose names map to gateway-owned tools.
    #[must_use]
    pub fn gateway_owned<'a>(&self, calls: &'a [FunctionToolCall]) -> Vec<&'a FunctionToolCall> {
        calls
            .iter()
            .filter(|c| {
                self.entries
                    .get(&c.name)
                    .is_some_and(|e| e.tool_type.is_gateway_owned())
            })
            .collect()
    }

    #[must_use]
    pub fn is_gateway_owned_name(&self, name: &str) -> bool {
        self.entries
            .get(name)
            .is_some_and(|entry| entry.tool_type.is_gateway_owned())
    }

    /// Returns the subset of `calls` whose names map to client-owned tools
    /// (`Function`, Codex namespace members, or unknown names).
    #[must_use]
    pub fn client_owned<'a>(&self, calls: &'a [FunctionToolCall]) -> Vec<&'a FunctionToolCall> {
        calls
            .iter()
            .filter(|c| {
                self.entries
                    .get(&c.name)
                    .is_none_or(|e| !e.tool_type.is_gateway_owned())
            })
            .collect()
    }

    pub async fn dispatch(&self, call: &FunctionToolCall) -> Option<GatewayDispatchResult> {
        let entry = self.entries.get(&call.name)?;
        let handler = entry.handler.clone()?;
        let tool_type = entry.tool_type;
        let config = entry.config.clone();
        Some(GatewayDispatchResult {
            tool_type,
            output: handler
                .execute(&call.call_id, &call.name, &call.arguments, &config)
                .await,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::model_visible_namespace_member_name;
    use crate::types::event::MessageStatus;
    use crate::types::io::FunctionToolCall;

    fn function_call(name: impl Into<String>, namespace: Option<&str>) -> OutputItem {
        OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: name.into(),
            namespace: namespace.map(str::to_owned),
            arguments: "{}".to_owned(),
            status: MessageStatus::Completed,
        })
    }

    #[tokio::test]
    async fn declared_function_named_tool_search_owns_its_call() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {"type": "function", "name": "tool_search"},
            {"type": "tool_search", "execution": "client"}
        ]))
        .unwrap();
        let registry = ToolRegistry::build_with_handlers(&tools, &GatewayExecutors::default())
            .await
            .unwrap();
        let mut output = vec![function_call("tool_search", None)];

        registry.restore_final_payload_output(&mut output);

        assert!(matches!(output[0], OutputItem::FunctionCall(_)));
        assert!(!registry.can_restore_tool_search_fallback());
    }

    #[tokio::test]
    async fn declared_namespace_member_named_tool_search_is_restored_before_fallback_classification() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "fixture",
                "tools": [{"type": "function", "name": "tool_search"}]
            },
            {"type": "tool_search", "execution": "client"}
        ]))
        .unwrap();
        let registry = ToolRegistry::build_with_handlers(&tools, &GatewayExecutors::default())
            .await
            .unwrap();
        let flat_name = model_visible_namespace_member_name("fixture", "tool_search");
        let mut output = vec![function_call(flat_name, None)];

        registry.restore_final_payload_output(&mut output);

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("namespace member must remain a function call");
        };
        assert_eq!(call.name, "tool_search");
        assert_eq!(call.namespace.as_deref(), Some("fixture"));
    }

    #[tokio::test]
    async fn dynamically_loaded_namespace_member_named_tool_search_owns_its_call() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {"type": "tool_search", "execution": "client"}
        ]))
        .unwrap();
        let mut registry = ToolRegistry::build_with_handlers(&tools, &GatewayExecutors::default())
            .await
            .unwrap();
        let input = serde_json::from_value(serde_json::json!([
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "arguments": {"query": "search helper"}
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "call_search",
                "status": "completed",
                "tools": [{
                    "type": "namespace",
                    "name": "fixture",
                    "tools": [{"type": "function", "name": "tool_search"}]
                }]
            }
        ]))
        .unwrap();
        registry.load_tool_search_output(&input);
        let mut output = vec![function_call("tool_search", None)];

        registry.restore_final_payload_output(&mut output);

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("loaded namespace member must remain a function call");
        };
        assert_eq!(call.name, "tool_search");
        assert_eq!(call.namespace.as_deref(), Some("fixture"));
        assert!(!registry.can_restore_tool_search_fallback());
    }

    #[tokio::test]
    async fn unqualified_provider_fallback_is_restored_when_name_is_unowned() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {"type": "tool_search", "execution": "client"}
        ]))
        .unwrap();
        let registry = ToolRegistry::build_with_handlers(&tools, &GatewayExecutors::default())
            .await
            .unwrap();
        let mut output = vec![function_call("tool_search", None)];

        registry.restore_final_payload_output(&mut output);

        assert!(matches!(output[0], OutputItem::ToolSearchCall(_)));
        assert!(registry.can_restore_tool_search_fallback());
    }
}
