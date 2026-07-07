use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    FunctionHandler, GatewayExecutor, ToolError, ToolHandler, ToolOutput, flatten_tool_choice_for_upstream,
    flatten_tools_for_upstream, restore_output_items_with_tools, restore_response_value_with_tools,
};
use crate::types::io::output::FunctionToolCall;
use crate::types::io::{FunctionTool, OutputItem, ToolChoice};
use crate::types::tools::{CodexNamespaceMember, ResponsesTool};

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
    declared_tools: Vec<ResponsesTool>,
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
                    // MCP tool names are discovered at request-time via `tools/list`.
                    // Without discovery, we cannot know which tool names to register —
                    // keying by server_label would cause all MCP calls to miss on lookup
                    // since gateway_owned/client_owned look up by tool name, not server.
                    // MCP entries will be populated in PR C once HttpMcpHandler
                    // implements discover() and the executor calls it before build().
                    tracing::debug!(
                        server_label = %p.server_label,
                        "MCP server declared but skipped in registry — tool names unknown until discovery (PR C)"
                    );
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
                ResponsesTool::Namespace(p) => {
                    for member in &p.tools {
                        let CodexNamespaceMember::Function(function) = member else {
                            continue;
                        };
                        let name = super::model_visible_namespace_member_name(&p.name, function.name.as_str());
                        if entries
                            .insert(
                                name.clone(),
                                ToolEntry {
                                    tool_type: ToolType::Function,
                                    config: serde_json::to_value(function)
                                        .expect("serialization of known struct is infallible"),
                                    server_label: Some(p.name.clone()),
                                    handler: None,
                                },
                            )
                            .is_some()
                        {
                            tracing::warn!(name = %name, namespace = %p.name, "duplicate tool name - previous definition overwritten");
                        }
                    }
                }
                ResponsesTool::Unknown => {
                    tracing::debug!("unknown tool declared but skipped in registry");
                }
            }
        }

        Self {
            entries,
            declared_tools: tools.to_vec(),
        }
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

    #[must_use]
    pub fn upstream_tools(&self) -> Option<Vec<FunctionTool>> {
        let flattened =
            flatten_tools_for_upstream((!self.declared_tools.is_empty()).then_some(self.declared_tools.as_slice()))?;
        let tools = flattened
            .iter()
            .flat_map(|tool| self.normalize_tool_for_upstream(tool))
            .collect::<Vec<_>>();
        (!tools.is_empty()).then_some(tools)
    }

    #[must_use]
    pub fn upstream_tool_choice(&self, choice: &ToolChoice) -> ToolChoice {
        flatten_tool_choice_for_upstream(
            choice,
            (!self.declared_tools.is_empty()).then_some(self.declared_tools.as_slice()),
        )
    }

    pub fn restore_output_items(&self, output: &mut [OutputItem]) {
        restore_output_items_with_tools(
            output,
            (!self.declared_tools.is_empty()).then_some(self.declared_tools.as_slice()),
        );
    }

    pub fn restore_response_value(&self, value: &mut Value) -> bool {
        restore_response_value_with_tools(
            value,
            (!self.declared_tools.is_empty()).then_some(self.declared_tools.as_slice()),
        )
    }

    fn normalize_tool_for_upstream(&self, tool: &ResponsesTool) -> Vec<FunctionTool> {
        match tool {
            ResponsesTool::Function(p) => {
                let param = serde_json::to_value(p).expect("serialization of known struct is infallible");
                FunctionHandler.normalize(&param)
            }
            ResponsesTool::WebSearch(_) => self
                .entries
                .get("web_search")
                .and_then(|entry| entry.handler.as_ref().map(|handler| handler.normalize(&entry.config)))
                .unwrap_or_else(|| tool.to_function_tool().into_iter().collect()),
            ResponsesTool::Mcp(_)
            | ResponsesTool::FileSearch(_)
            | ResponsesTool::CodeInterpreter(_)
            | ResponsesTool::Namespace(_)
            | ResponsesTool::Unknown => tool.to_function_tool().into_iter().collect(),
        }
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

    #[must_use]
    pub fn is_gateway_owned_name(&self, name: &str) -> bool {
        self.entries
            .get(name)
            .is_some_and(|entry| entry.tool_type != ToolType::Function)
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
