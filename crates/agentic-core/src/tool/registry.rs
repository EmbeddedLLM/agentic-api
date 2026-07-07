use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::mcp::READ_MCP_RESOURCE_TOOL_NAME;
use super::{GatewayExecutor, ToolError, ToolOutput};
use crate::types::io::output::FunctionToolCall;
use crate::types::tools::ResponsesTool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
    Mcp,
    /// Internal routing discriminant. Serializes as `"web_search"`.
    /// Note: the corresponding `ResponsesTool` wire tag is `"web_search_preview"`.
    /// `ToolType` is not used in wire-facing types so the names differ intentionally.
    WebSearch,
    FileSearch,
    CodeInterpreter,
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

/// Request-scoped registry built from `RequestPayload.tools`.
/// Maps the name the LLM sees → routing metadata.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    entries: HashMap<String, ToolEntry>,
}

impl ToolRegistry {
    /// Build a registry from the declared tools.
    ///
    /// Function tools with empty names are skipped with a warning. Duplicate
    /// tool names result in last-write-wins, also logged at `warn` level.
    ///
    /// # Panics
    ///
    /// Panics if serialization of a tool param struct fails, which cannot happen
    /// for the types defined in this module (`#[derive(Serialize)]` on plain structs).
    #[must_use]
    pub fn build(tools: &[ResponsesTool]) -> Self {
        Self::build_with_handlers(tools, |_| None)
    }

    #[must_use]
    pub fn with_entries(entries: impl IntoIterator<Item = (String, ToolEntry)>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
        }
    }

    #[must_use]
    /// Build a registry from declared tools and attach gateway handlers for dispatchable tool types.
    ///
    /// # Panics
    ///
    /// Panics if serialization of a tool param struct fails, which cannot happen
    /// for the types defined in this module (`#[derive(Serialize)]` on plain structs).
    pub fn build_with_handlers(
        tools: &[ResponsesTool],
        mut handler_for: impl FnMut(ToolType) -> Option<Arc<dyn GatewayExecutor>>,
    ) -> Self {
        let mut entries = HashMap::with_capacity(tools.len());

        for tool in tools {
            match tool {
                ResponsesTool::Function(p) => {
                    // p.name is NonEmptyToolName — empty names are impossible here
                    // (serde rejects them at deserialization time).
                    if entries
                        .insert(
                            p.name.as_str().to_owned(),
                            ToolEntry {
                                tool_type: ToolType::Function,
                                config: serde_json::to_value(p).expect("serialization of known struct is infallible"),
                                server_label: None,
                                handler: None,
                            },
                        )
                        .is_some()
                    {
                        tracing::warn!(name = %p.name, "duplicate tool name — previous definition overwritten");
                    }
                }
                ResponsesTool::Mcp(p) => {
                    if p.name.as_str() == READ_MCP_RESOURCE_TOOL_NAME {
                        if entries
                            .insert(
                                READ_MCP_RESOURCE_TOOL_NAME.to_owned(),
                                ToolEntry {
                                    tool_type: ToolType::Mcp,
                                    config: serde_json::to_value(p)
                                        .expect("serialization of known struct is infallible"),
                                    server_label: None,
                                    handler: handler_for(ToolType::Mcp),
                                },
                            )
                            .is_some()
                        {
                            tracing::warn!(name = %p.name, "duplicate MCP tool name — previous definition overwritten");
                        }
                    } else {
                        tracing::debug!(name = %p.name, "unknown MCP built-in skipped in registry");
                    }
                }
                ResponsesTool::WebSearch(p) => {
                    entries.insert(
                        "web_search".to_owned(),
                        ToolEntry {
                            tool_type: ToolType::WebSearch,
                            config: serde_json::to_value(p).expect("serialization of known struct is infallible"),
                            server_label: None,
                            handler: handler_for(ToolType::WebSearch),
                        },
                    );
                }
                ResponsesTool::FileSearch(p) => {
                    entries.insert(
                        "file_search".to_owned(),
                        ToolEntry {
                            tool_type: ToolType::FileSearch,
                            config: serde_json::to_value(p).expect("serialization of known struct is infallible"),
                            server_label: None,
                            handler: handler_for(ToolType::FileSearch),
                        },
                    );
                }
                ResponsesTool::CodeInterpreter(p) => {
                    entries.insert(
                        "code_interpreter".to_owned(),
                        ToolEntry {
                            tool_type: ToolType::CodeInterpreter,
                            config: serde_json::to_value(p).expect("serialization of known struct is infallible"),
                            server_label: None,
                            handler: handler_for(ToolType::CodeInterpreter),
                        },
                    );
                }
            }
        }

        Self { entries }
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

    /// Returns the subset of `calls` whose names map to gateway-owned tools
    /// (i.e. everything except `ToolType::Function`).
    #[must_use]
    pub fn gateway_owned<'a>(&self, calls: &'a [FunctionToolCall]) -> Vec<&'a FunctionToolCall> {
        calls
            .iter()
            .filter(|c| {
                self.entries
                    .get(&c.name)
                    .is_some_and(|e| e.tool_type != ToolType::Function)
            })
            .collect()
    }

    /// Returns the subset of `calls` whose names map to client-owned function
    /// tools (i.e. `ToolType::Function` or unknown names).
    #[must_use]
    pub fn client_owned<'a>(&self, calls: &'a [FunctionToolCall]) -> Vec<&'a FunctionToolCall> {
        calls
            .iter()
            .filter(|c| {
                self.entries
                    .get(&c.name)
                    .is_none_or(|e| e.tool_type == ToolType::Function)
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
    use std::sync::Arc;

    use super::{ToolRegistry, ToolType};
    use crate::tool::{McpClientPool, McpHandler, READ_MCP_RESOURCE_TOOL_NAME};
    use crate::types::tools::ResponsesTool;

    #[test]
    fn read_mcp_resource_function_stays_client_owned() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "name": READ_MCP_RESOURCE_TOOL_NAME,
            "description": "Read a resource by URI from a connected MCP server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "uri": {"type": "string"}
                },
                "required": ["server", "uri"]
            },
            "strict": false
        }]))
        .unwrap();

        let handler = Arc::new(McpHandler::read_resource(Arc::new(McpClientPool::default())));
        let registry = ToolRegistry::build_with_handlers(&tools, |tool_type| {
            (tool_type == ToolType::Mcp).then(|| handler.clone() as Arc<dyn crate::tool::GatewayExecutor>)
        });

        let entry = registry
            .lookup(READ_MCP_RESOURCE_TOOL_NAME)
            .expect("read_mcp_resource entry");
        assert_eq!(entry.tool_type, ToolType::Function);
        assert!(entry.handler.is_none());
    }

    #[test]
    fn read_mcp_resource_mcp_tool_registers_as_gateway_mcp_when_handler_exists() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([{
            "type": "mcp",
            "name": READ_MCP_RESOURCE_TOOL_NAME
        }]))
        .unwrap();

        let handler = Arc::new(McpHandler::read_resource(Arc::new(McpClientPool::default())));
        let registry = ToolRegistry::build_with_handlers(&tools, |tool_type| {
            (tool_type == ToolType::Mcp).then(|| handler.clone() as Arc<dyn crate::tool::GatewayExecutor>)
        });

        let entry = registry
            .lookup(READ_MCP_RESOURCE_TOOL_NAME)
            .expect("read_mcp_resource entry");
        assert_eq!(entry.tool_type, ToolType::Mcp);
        assert!(entry.handler.is_some());
    }
}
