use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::io::output::FunctionToolCall;
use crate::types::tools::ResponsesTool;
use crate::utils::common::serialize_to_value;

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
    /// Any tool declared by Codex CLI's own wire format (`function`, `namespace`,
    /// `tool_search`, `custom`, or an unrecognized shape). All are client-executed
    /// and normalized together by [`crate::tool::codex::CodexToolHandler`].
    Codex,
}

/// Per-request routing entry keyed by the tool name the model will call.
#[derive(Debug, Clone)]
pub struct ToolEntry {
    pub tool_type: ToolType,
    /// Full serialised tool param for the executor (used during dispatch).
    pub config: Value,
    /// For MCP tools: which server this tool belongs to.
    pub server_label: Option<String>,
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
    /// Each tool is validated via its [`crate::tool::ToolHandler::validate`]
    /// (through [`ResponsesTool::handler`]) before being registered; tools that
    /// fail validation are skipped with a warning. `Function` and `Codex` are
    /// keyed by each normalized `FunctionTool`'s name (via
    /// [`ResponsesTool::to_function_tools`]), so a Codex `namespace` correctly
    /// registers one entry per subtool. `WebSearch`/`FileSearch`/`CodeInterpreter`
    /// have no handler yet, so they keep a fixed name so lookups work ahead of
    /// their handlers landing. `Mcp` is skipped entirely — its tool names are
    /// only known after request-time discovery (PR C). Duplicate tool names
    /// result in last-write-wins, logged at `warn` level.
    ///
    /// # Panics
    ///
    /// Panics if serialization of a tool param struct fails, which cannot happen
    /// for the types defined in this module (`#[derive(Serialize)]` on plain structs).
    #[must_use]
    pub fn build(tools: &[ResponsesTool]) -> Self {
        let mut entries = HashMap::with_capacity(tools.len());

        for tool in tools {
            if let ResponsesTool::Mcp(p) = tool {
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
                continue;
            }

            if let Some(handler) = tool.handler() {
                if let Err(e) = handler.validate(&serialize_to_value(tool)) {
                    tracing::warn!(error = %e, "tool failed validation — skipped in registry");
                    continue;
                }
            }

            let (tool_type, names): (ToolType, Vec<String>) = match tool {
                ResponsesTool::Function(_) | ResponsesTool::Codex(_) => {
                    let tool_type = if matches!(tool, ResponsesTool::Function(_)) {
                        ToolType::Function
                    } else {
                        ToolType::Codex
                    };
                    (
                        tool_type,
                        tool.to_function_tools().into_iter().map(|f| f.name).collect(),
                    )
                }
                ResponsesTool::WebSearch(_) => (ToolType::WebSearch, vec!["web_search".to_owned()]),
                ResponsesTool::FileSearch(_) => (ToolType::FileSearch, vec!["file_search".to_owned()]),
                ResponsesTool::CodeInterpreter(_) => (ToolType::CodeInterpreter, vec!["code_interpreter".to_owned()]),
                ResponsesTool::Mcp(_) => unreachable!("Mcp is handled above and continues before reaching here"),
            };
            // `names` is empty for a `Codex(CodexParams::Unknown)` shape — the
            // loop below is then a no-op, which is the desired "skip silently"
            // behavior for unrecognized tool types.

            for name in names {
                if entries
                    .insert(
                        name.clone(),
                        ToolEntry {
                            tool_type,
                            config: serialize_to_value(tool),
                            server_label: None,
                        },
                    )
                    .is_some()
                {
                    tracing::warn!(name = %name, "duplicate tool name — previous definition overwritten");
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
}
