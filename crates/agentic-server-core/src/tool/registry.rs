use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::codex::insert_namespace_entries;
use super::executors::GatewayExecutors;
use super::function::insert_function_entry;
use super::mcp::{insert_discovered_mcp_entry, insert_read_resource_entry, maybe_mcp_function};
use super::web_search::insert_web_search_entry;
use super::{CodexNamespaceHandler, GatewayExecutor, NamespaceMap, ToolError, ToolOutput};
use crate::types::io::OutputItem;
use crate::types::io::output::FunctionToolCall;
use crate::types::tools::{CodeInterpreterToolParam, FileSearchToolParam, ResponsesTool};
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
    pub async fn build_with_handlers(
        tools: &mut [ResponsesTool],
        executors: &mut GatewayExecutors,
    ) -> Result<Self, ToolError> {
        let mut entries = HashMap::with_capacity(tools.len());
        let mut mcp_server_labels = HashSet::new();
        // Namespace members must be keyed by the same flat, model-visible name
        // the model will call, so resolve them first — the same pure pass used
        // to build the upstream request.
        let resolved_tools = CodexNamespaceHandler.resolve_namespace_members(tools)?;

        for tool in &resolved_tools {
            let ResponsesTool::Mcp(param) = tool else {
                continue;
            };
            if !mcp_server_labels.insert(param.server_label.clone()) {
                return Err(ToolError::Config(format!(
                    "duplicate MCP declarations are not allowed for server_label '{}'",
                    param.server_label
                )));
            }
        }

        for (index, tool) in resolved_tools.iter().enumerate() {
            match tool {
                ResponsesTool::Function(p) => match maybe_mcp_function(p) {
                    Some(mcp_params) if !mcp_params.is_empty() => {
                        let handler = executors.mcp_read_resource_handler(&mcp_params).await;
                        if let Some(declaration_param) = mcp_params.first() {
                            insert_read_resource_entry(&mut entries, declaration_param, handler);
                        }
                    }
                    _ => insert_function_entry(&mut entries, p),
                },
                ResponsesTool::Mcp(p) => {
                    let handlers = executors.mcp_handler(p).await?;
                    if let ResponsesTool::Mcp(declaration) = &mut tools[index] {
                        declaration.discovered_tools = handlers.iter().map(|item| item.param.clone()).collect();
                    }
                    for discovered in handlers {
                        insert_discovered_mcp_entry(&mut entries, discovered);
                    }
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
                ResponsesTool::Unknown => {
                    tracing::debug!("unknown tool declared but skipped in registry");
                }
            }
        }

        let namespace_map = CodexNamespaceHandler.build_namespace_map((!tools.is_empty()).then_some(tools))?;

        Ok(Self { entries, namespace_map })
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
    }

    pub fn restore_stream_event_value(&self, value: &mut Value) -> bool {
        CodexNamespaceHandler.restore_response_value(value, self.namespace_map.as_ref())
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
    use crate::tool::executors::GatewayExecutorRegistration;
    use crate::tool::mcp::{McpDiscoveredHandler, McpHandler};
    use crate::types::tools::McpDiscoveredToolParam;

    fn declaration(server_label: &str) -> ResponsesTool {
        serde_json::from_value(serde_json::json!({
            "type": "mcp",
            "server_label": server_label,
            "server_url": "http://127.0.0.1:8000/mcp"
        }))
        .expect("MCP declaration")
    }

    fn discovered_handler() -> McpDiscoveredHandler {
        let param = McpDiscoveredToolParam {
            server_label: "counter".to_owned(),
            tool_name: "increment".to_owned(),
            internal_name: "mcp__counter__increment".to_owned(),
            tool: serde_json::from_value(serde_json::json!({
                "name": "increment",
                "description": "Increment the counter",
                "inputSchema": {"type": "object"}
            }))
            .expect("discovered MCP tool"),
        };
        McpDiscoveredHandler {
            param,
            handler: Arc::new(McpHandler::discovered_tool_spec_only()),
        }
    }

    #[tokio::test]
    async fn cached_mcp_handler_populates_declaration_and_registry() {
        let mut executors = GatewayExecutors::default();
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "counter".to_owned(),
            handlers: vec![discovered_handler()],
        });
        let mut tools = vec![declaration("counter")];

        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry with cached MCP handler");

        assert!(registry.lookup("mcp__counter__increment").is_some());
        let ResponsesTool::Mcp(declared) = &tools[0] else {
            panic!("expected MCP declaration");
        };
        assert_eq!(declared.discovered_tools.len(), 1);
        let normalized = tools[0].to_function_tools();
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].name, "mcp__counter__increment");
    }

    #[tokio::test]
    async fn duplicate_mcp_server_labels_are_rejected() {
        let mut tools = vec![declaration("counter"), declaration("counter")];
        let mut executors = GatewayExecutors::default();

        let error = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect_err("duplicate server_label must fail");

        assert!(
            matches!(error, ToolError::Config(message) if message.contains("duplicate MCP declarations") && message.contains("counter"))
        );
    }
}
