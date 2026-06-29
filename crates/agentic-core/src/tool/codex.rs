//! Codex CLI tool normalization and execution.
//!
//! Defines [`IncomingTool`] — a tagged enum covering every tool variant Codex
//! CLI can send — and implements [`NormalizeTool`] and [`ExecuteTool`] on it.
//!
//! # Normalize variant mapping
//!
//! | Codex type         | Normalized to                                    |
//! |--------------------|--------------------------------------------------|
//! | `function`         | itself, unchanged                                |
//! | `namespace`        | one `FunctionTool` per subtool (flattened)       |
//! | `tool_search`      | one `FunctionTool` named `"tool_search"`         |
//! | `custom`           | one `FunctionTool` using the custom tool's name  |
//! | unknown / other    | dropped                                          |
//!
//! All tool execution is Codex CLI's responsibility. [`ExecuteTool::execute`]
//! always returns `None` — agentic-api passes `FunctionCall` items through
//! and Codex CLI drives the next turn with `function_call_output`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool::{ExecuteTool, NormalizeTool};
use crate::types::io::{FunctionTool, FunctionToolResultMessage};

/// All tool definition shapes that Codex CLI may include in a request.
///
/// Intermediate struct used during deserialization to peek at the `type` field.
#[derive(Deserialize)]
struct RawTool {
    #[serde(rename = "type")]
    type_: String,
    name: Option<String>,
    description: Option<String>,
    parameters: Option<Value>,
    format: Option<Value>,
    tools: Option<Vec<IncomingTool>>,
}

/// All tool definition shapes that Codex CLI may include in a request.
///
/// Uses a manual `Deserialize` impl (via `RawTool`) so that unknown tool types
/// with arbitrary extra fields (e.g. `web_search` with `external_web_access`)
/// are accepted and mapped to `Unknown` without errors. Serde's `#[serde(other)]`
/// on unit variants rejects objects with extra fields in internally-tagged enums.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IncomingTool {
    /// Standard OpenAI function tool — passed through unchanged.
    Function(FunctionTool),

    /// A named container of subtools (e.g. `mcp__github`).
    ///
    /// The model selects individual subtools by name, so normalization
    /// flattens the container into one `FunctionTool` per subtool.
    Namespace {
        name: String,
        description: Option<String>,
        tools: Vec<IncomingTool>,
    },

    /// Deferred tool discovery — normalized to a single `FunctionTool`
    /// named `"tool_search"`.
    ToolSearch {
        description: Option<String>,
        parameters: Option<Value>,
    },

    /// Freeform / plugin tool — normalized using its own `name`.
    Custom {
        name: String,
        description: Option<String>,
        /// Arbitrary format metadata carried as `parameters` so vLLM
        /// sees a schema-shaped object rather than an opaque blob.
        format: Option<Value>,
    },

    /// Any Codex type not listed above (e.g. `web_search`, `code_interpreter`)
    /// — dropped gracefully during normalization.
    Unknown,
}

impl<'de> Deserialize<'de> for IncomingTool {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawTool::deserialize(deserializer)?;
        Ok(match raw.type_.as_str() {
            "function" => {
                let name = raw.name.unwrap_or_default();
                Self::Function(FunctionTool {
                    type_: "function".into(),
                    name,
                    description: raw.description,
                    parameters: raw.parameters,
                    strict: None,
                })
            }
            "namespace" => Self::Namespace {
                name: raw.name.unwrap_or_default(),
                description: raw.description,
                tools: raw.tools.unwrap_or_default(),
            },
            "tool_search" => Self::ToolSearch {
                description: raw.description,
                parameters: raw.parameters,
            },
            "custom" => Self::Custom {
                name: raw.name.unwrap_or_default(),
                description: raw.description,
                format: raw.format,
            },
            _ => Self::Unknown,
        })
    }
}

impl NormalizeTool for IncomingTool {
    /// Flatten this tool into the `FunctionTool` shape vLLM accepts.
    ///
    /// `namespace` expands into its first subtool only — callers that need the
    /// full expansion should use [`normalize_tools`](crate::tool::normalize_tools)
    /// which iterates over the collection. For a single top-level namespace call
    /// `normalize_all` on the namespace's `tools` field instead.
    ///
    /// `code_interpreter`, `web_search`, and `unknown` return `None` — they are
    /// either handled server-side after the response or dropped entirely.
    fn normalize(&self) -> Option<FunctionTool> {
        match self {
            Self::Function(f) => Some(f.clone()),

            // Namespaces expand to N tools; normalize_tools handles the iteration.
            // A single normalize() call on a namespace returns the first subtool only.
            // Callers should prefer normalize_all_tools() for namespace expansion.
            Self::Namespace { tools, .. } => tools.first().and_then(|t| t.normalize()),

            Self::ToolSearch {
                description,
                parameters,
            } => Some(FunctionTool {
                type_: "function".into(),
                name: "tool_search".into(),
                description: description.clone(),
                parameters: parameters.clone(),
                strict: None,
            }),

            Self::Custom {
                name,
                description,
                format,
            } => Some(FunctionTool {
                type_: "function".into(),
                name: name.clone(),
                description: description.clone(),
                parameters: format.clone(),
                strict: None,
            }),

            // Unknown types: dropped from the vLLM tool list.
            Self::Unknown => None,
        }
    }
}

impl ExecuteTool for IncomingTool {
    async fn execute(&self, _call_id: &str, _arguments: &str) -> Option<FunctionToolResultMessage> {
        // All tool execution is Codex CLI's responsibility.
        None
    }
}

/// Normalize a collection of [`IncomingTool`]s, correctly expanding `namespace`
/// containers into individual `FunctionTool`s.
///
/// This is the correct entry point for building the vLLM tool list from a
/// `RequestPayload`. Unlike calling `normalize()` on each item, this function
/// recurses into `Namespace` variants rather than returning just the first subtool.
pub fn normalize_incoming_tools(tools: &[IncomingTool]) -> Vec<FunctionTool> {
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        flatten_one(tool, &mut out);
    }
    out
}

fn flatten_one(tool: &IncomingTool, out: &mut Vec<FunctionTool>) {
    match tool {
        IncomingTool::Function(f) => out.push(f.clone()),

        IncomingTool::Namespace { tools, .. } => {
            for subtool in tools {
                flatten_one(subtool, out);
            }
        }

        IncomingTool::ToolSearch {
            description,
            parameters,
        } => {
            out.push(FunctionTool {
                type_: "function".into(),
                name: "tool_search".into(),
                description: description.clone(),
                parameters: parameters.clone(),
                strict: None,
            });
        }

        IncomingTool::Custom {
            name,
            description,
            format,
        } => {
            out.push(FunctionTool {
                type_: "function".into(),
                name: name.clone(),
                description: description.clone(),
                parameters: format.clone(),
                strict: None,
            });
        }

        IncomingTool::Unknown => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::normalize_tools;
    use serde_json::json;

    fn function_tool(name: &str) -> IncomingTool {
        IncomingTool::Function(FunctionTool {
            type_: "function".into(),
            name: name.into(),
            description: None,
            parameters: None,
            strict: None,
        })
    }

    #[test]
    fn normalize_function_passes_through() {
        let tool = function_tool("my_fn");
        let out = tool.normalize().unwrap();
        assert_eq!(out.name, "my_fn");
        assert_eq!(out.type_, "function");
    }

    #[test]
    fn normalize_tool_search_becomes_function() {
        let params = json!({"type": "object", "properties": {"query": {"type": "string"}}});
        let tool = IncomingTool::ToolSearch {
            description: Some("Search".into()),
            parameters: Some(params.clone()),
        };
        let out = tool.normalize().unwrap();
        assert_eq!(out.name, "tool_search");
        assert_eq!(out.parameters.as_ref().unwrap(), &params);
    }

    #[test]
    fn normalize_custom_becomes_function() {
        let fmt = json!({"type": "grammar"});
        let tool = IncomingTool::Custom {
            name: "lark_parser".into(),
            description: None,
            format: Some(fmt.clone()),
        };
        let out = tool.normalize().unwrap();
        assert_eq!(out.name, "lark_parser");
        assert_eq!(out.parameters.as_ref().unwrap(), &fmt);
    }

    #[test]
    fn normalize_unknown_returns_none() {
        assert!(IncomingTool::Unknown.normalize().is_none());
    }

    #[test]
    fn normalize_incoming_namespace_flattens_subtools() {
        let tools = vec![IncomingTool::Namespace {
            name: "mcp__github".into(),
            description: None,
            tools: vec![function_tool("create_issue"), function_tool("search_code")],
        }];
        let out = normalize_incoming_tools(&tools);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "create_issue");
        assert_eq!(out[1].name, "search_code");
    }

    #[test]
    fn normalize_incoming_nested_namespace_recurses() {
        let inner = IncomingTool::Namespace {
            name: "inner".into(),
            description: None,
            tools: vec![function_tool("leaf")],
        };
        let tools = vec![IncomingTool::Namespace {
            name: "outer".into(),
            description: None,
            tools: vec![inner],
        }];
        let out = normalize_incoming_tools(&tools);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "leaf");
    }

    #[test]
    fn normalize_incoming_mixed_list() {
        let tools = vec![
            function_tool("fn_a"),
            IncomingTool::Namespace {
                name: "ns".into(),
                description: None,
                tools: vec![function_tool("ns_tool")],
            },
            IncomingTool::ToolSearch {
                description: None,
                parameters: None,
            },
            IncomingTool::Unknown,
        ];
        let out = normalize_incoming_tools(&tools);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].name, "fn_a");
        assert_eq!(out[1].name, "ns_tool");
        assert_eq!(out[2].name, "tool_search");
    }

    #[test]
    fn normalize_tools_uses_trait() {
        let tools = vec![function_tool("a"), IncomingTool::Unknown];
        let out = normalize_tools(&tools);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "a");
    }

    #[tokio::test]
    async fn execute_always_returns_none() {
        assert!(function_tool("my_fn").execute("call_1", "{}").await.is_none());
        assert!(IncomingTool::Unknown.execute("call_2", "{}").await.is_none());
    }

    #[tokio::test]
    async fn execute_namespace_returns_none() {
        let tool = IncomingTool::Namespace {
            name: "ns".into(),
            description: None,
            tools: vec![],
        };
        assert!(tool.execute("call_4", "{}").await.is_none());
    }

    #[test]
    fn deserialize_namespace_from_json() {
        let json = json!({
            "type": "namespace",
            "name": "mcp__github",
            "description": "GitHub tools",
            "tools": [
                {"type": "function", "name": "create_issue",
                 "parameters": {"type": "object", "properties": {}}}
            ]
        });
        let tool: IncomingTool = serde_json::from_value(json).unwrap();
        let out = normalize_incoming_tools(&[tool]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "create_issue");
    }

    #[test]
    fn deserialize_tool_search_from_json() {
        let json = json!({
            "type": "tool_search",
            "description": "Search for deferred tools",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        });
        let tool: IncomingTool = serde_json::from_value(json).unwrap();
        assert_eq!(tool.normalize().unwrap().name, "tool_search");
    }

    #[test]
    fn deserialize_custom_from_json() {
        let json = json!({
            "type": "custom",
            "name": "lark_parser",
            "description": "Parse Lark grammar files",
            "format": {"type": "grammar", "syntax": "lark"}
        });
        let tool: IncomingTool = serde_json::from_value(json).unwrap();
        assert_eq!(tool.normalize().unwrap().name, "lark_parser");
    }
}
