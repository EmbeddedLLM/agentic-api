//! Codex CLI's own tool shapes — the wire formats it sends that don't already
//! exist in the standard `OpenAI` Responses API taxonomy ([`ResponsesTool`](crate::types::tools::ResponsesTool)).
//!
//! Defines [`CodexParams`] — a tagged enum covering `namespace`, `tool_search`,
//! `custom`, and an `Unknown` catch-all — and [`CodexToolHandler`], the
//! [`ToolHandler`] implementation that validates and normalizes it. Codex's
//! `function` shape is wire-identical to `ResponsesTool::Function` and is
//! handled there directly. `ResponsesTool::Codex(CodexParams)` is the single
//! entry point: every client (Codex CLI or otherwise) declares tools through
//! `RequestPayload.tools: Vec<ResponsesTool>`.
//!
//! # Normalize variant mapping
//!
//! | Codex type         | Normalized to                                    |
//! |--------------------|--------------------------------------------------|
//! | `namespace`        | one `FunctionTool` per subtool (flattened)       |
//! | `tool_search`      | one `FunctionTool` named `"tool_search"`         |
//! | `custom`           | one `FunctionTool` using the custom tool's name  |
//! | unknown / other    | dropped                                          |
//!
//! All tool execution is Codex CLI's responsibility — agentic-api never
//! executes a Codex tool call, it only normalizes the declarations forwarded
//! upstream. Codex CLI drives the next turn with `function_call_output` itself.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool::handler::{ToolError, ToolHandler};
use crate::tool::registry::ToolType;
use crate::types::io::{FunctionTool, FunctionToolCall};
use crate::types::tools::ResponsesTool;

/// Codex CLI's tool shapes that have no equivalent in the standard `OpenAI`
/// Responses API taxonomy. Nested inside [`ResponsesTool::Codex`].
///
/// Internally-tagged on `type`, matching [`ResponsesTool`]'s own tagging so a
/// Codex tool declaration deserializes into the right place regardless of
/// which variant it lands in. Also serves as `ResponsesTool`'s own catch-all:
/// serde requires the `#[serde(untagged)]` `Codex` variant to be the last
/// variant in `ResponsesTool`, so `CodexParams::Unknown` (via its own
/// `#[serde(other)]`) is where any `type` value unmatched by `ResponsesTool`'s
/// other variants (e.g. `computer_use_preview`) actually lands.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodexParams {
    /// A named container of subtools (e.g. `mcp__github`).
    ///
    /// The model selects individual subtools by name, so normalization
    /// flattens the container into one `FunctionTool` per subtool. Subtools
    /// are `ResponsesTool` since a namespace commonly contains plain
    /// `function` tools.
    Namespace {
        name: String,
        description: Option<String>,
        tools: Vec<ResponsesTool>,
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

    /// Any tool `type` not recognized by `ResponsesTool` or the variants
    /// above (e.g. `web_search`, `code_interpreter` as sent by Codex CLI, or
    /// future Responses API additions) — dropped gracefully during
    /// normalization rather than erroring.
    #[serde(other)]
    Unknown,
}

/// Normalize a single [`CodexParams`] value, correctly expanding `namespace`
/// containers into individual `FunctionTool`s.
///
/// Used by [`CodexToolHandler::normalize`] and directly by
/// [`super::normalize::ResponsesTool::to_function_tools`] (which already holds
/// a typed `CodexParams` and does not need to round-trip through `Value`).
pub(crate) fn normalize_codex_tool(tool: &CodexParams) -> Vec<FunctionTool> {
    let mut out = Vec::new();
    push_codex_tool(tool, None, &mut out);
    out
}

/// Normalizes `tool`, prefixing produced names with `prefix__` when set (used
/// for subtools nested inside an enclosing `namespace`).
fn push_codex_tool(tool: &CodexParams, prefix: Option<&str>, out: &mut Vec<FunctionTool>) {
    match tool {
        CodexParams::Namespace { name, tools, .. } => {
            // Strip trailing underscores from the namespace name to match Codex's
            // own join_tool_name behaviour (e.g. "mcp__foo__" -> "mcp__foo").
            let ns_part = name.trim_end_matches('_');
            let combined;
            let ns = match prefix {
                Some(outer) => {
                    combined = format!("{outer}__{ns_part}");
                    combined.as_str()
                }
                None => ns_part,
            };
            tracing::debug!(
                raw_name = %name,
                stripped = %ns_part,
                prefix = ?prefix,
                resolved_ns = %ns,
                subtool_count = tools.len(),
                "flattening codex namespace"
            );
            for subtool in tools {
                push_responses_tool(subtool, ns, out);
            }
        }

        CodexParams::ToolSearch {
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

        CodexParams::Custom {
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

        CodexParams::Unknown => {}
    }
}

/// Prefix marking a flattened namespace member name as gateway-generated.
///
/// Codex CLI needs to tell apart "a plain tool it declared" from "a member of
/// a namespace it declared" when a `function_call` comes back, so it can
/// route the call to the right MCP server. Without a distinctive marker, a
/// flattened name like `mcp__agentic_fixture__add_numbers` is indistinguishable
/// from a plain tool that happens to contain `__`, and Codex rejects the call
/// as unsupported. See [`model_visible_namespace_member_name`].
pub(crate) const MODEL_VISIBLE_NAMESPACE_MEMBER_PREFIX: &str = "agentic_ns__";

/// Builds the flattened, model-visible name for a `namespace` subtool.
///
/// `ns` is the (possibly nested) namespace path already joined with `__`;
/// `member` is the subtool's own name.
pub(crate) fn model_visible_namespace_member_name(ns: &str, member: &str) -> String {
    format!("{MODEL_VISIBLE_NAMESPACE_MEMBER_PREFIX}{ns}__{member}")
}

/// Normalizes a [`ResponsesTool`] subtool nested inside a Codex `namespace`,
/// prefixing its name with the enclosing namespace path (`ns`).
///
/// Only `Function` and nested `Codex` subtools produce output — other
/// gateway-owned shapes (`Mcp`, `WebSearch`, …) do not make sense nested
/// inside a Codex namespace and are dropped. `#[non_exhaustive]` on
/// `ResponsesTool` requires the wildcard arm for variants added later.
fn push_responses_tool(tool: &ResponsesTool, ns: &str, out: &mut Vec<FunctionTool>) {
    match tool {
        ResponsesTool::Function(p) => {
            let f = FunctionTool::from(p);
            let flattened_name = model_visible_namespace_member_name(ns, &f.name);
            tracing::debug!(ns = %ns, original_name = %f.name, flattened_name = %flattened_name, "flattened codex namespace subtool");
            out.push(FunctionTool {
                name: flattened_name,
                ..f
            });
        }
        ResponsesTool::Codex(inner) => push_codex_tool(inner, Some(ns), out),
        _ => {}
    }
}

/// Searches `tool` for the namespace member whose flattened, model-visible
/// name equals `flat_name` (see [`model_visible_namespace_member_name`]),
/// returning the member's declaring namespace path and its own (bare) name.
///
/// Recurses into nested namespaces, tracking the joined `ns` path the same
/// way [`push_codex_tool`]/[`push_responses_tool`] build it going forward, so
/// the two directions agree on what a given flattened name means.
fn find_namespace_member_by_flat_name(flat_name: &str, tool: &CodexParams) -> Option<(String, String)> {
    let CodexParams::Namespace { name, tools, .. } = tool else {
        return None;
    };
    find_in_namespace(flat_name, name, tools)
}

fn find_in_namespace(flat_name: &str, ns: &str, members: &[ResponsesTool]) -> Option<(String, String)> {
    let ns_part = ns.trim_end_matches('_');
    for member in members {
        match member {
            ResponsesTool::Function(p) => {
                if flat_name == model_visible_namespace_member_name(ns_part, p.name.as_str()) {
                    return Some((ns_part.to_owned(), p.name.as_str().to_owned()));
                }
            }
            ResponsesTool::Codex(CodexParams::Namespace {
                name: inner_name,
                tools: inner_tools,
                ..
            }) => {
                let combined = format!("{ns_part}__{}", inner_name.trim_end_matches('_'));
                if let Some(found) = find_in_namespace(flat_name, &combined, inner_tools) {
                    return Some(found);
                }
            }
            _ => {}
        }
    }
    None
}

/// Handler for Codex CLI's own tool shapes (`namespace`, `tool_search`, `custom`).
///
/// These are all just wire-format flavors of "a tool Codex CLI declared" that
/// have no equivalent in the standard `OpenAI` taxonomy — they are always
/// client-executed and normalized together as one wire concept (namespaces
/// flatten into their subtools), mirroring how `FunctionHandler` is one
/// handler for the whole `OpenAI` `function` wire concept.
///
/// All Codex tools are client-owned — Codex CLI executes them and drives the next
/// turn itself. `CodexToolHandler` intentionally implements only [`ToolHandler`],
/// not [`crate::tool::handler::GatewayExecutor`] — the type system makes it
/// impossible to call `execute()` on a client-owned tool.
#[derive(Debug, Default)]
pub struct CodexToolHandler;

impl ToolHandler for CodexToolHandler {
    fn tool_type(&self) -> ToolType {
        ToolType::Codex
    }

    /// Permissive validation: deserialization failures and empty names on
    /// name-bearing shapes (`namespace`, `custom`) are rejected; `tool_search`
    /// is always accepted.
    fn validate(&self, param: &Value) -> Result<(), ToolError> {
        let tool: CodexParams =
            serde_json::from_value(param.clone()).map_err(|e| ToolError::Config(format!("invalid codex tool: {e}")))?;
        match tool {
            CodexParams::Namespace { name, .. } if name.is_empty() => Err(ToolError::Config(
                "codex namespace tool must have a non-empty name".into(),
            )),
            CodexParams::Custom { name, .. } if name.is_empty() => {
                Err(ToolError::Config("codex custom tool must have a non-empty name".into()))
            }
            _ => Ok(()),
        }
    }

    fn normalize(&self, param: &Value) -> Vec<FunctionTool> {
        match serde_json::from_value::<CodexParams>(param.clone()) {
            Ok(tool) => normalize_codex_tool(&tool),
            Err(e) => {
                tracing::warn!(
                    "normalize() called with invalid codex tool param: {e} — validate() must be called first"
                );
                vec![]
            }
        }
    }

    /// Reverses [`Self::normalize`]'s namespace flattening: if `call.name`
    /// matches a member this `namespace` declares (see
    /// [`model_visible_namespace_member_name`]), rewrites it to the bare
    /// member name and sets `call.namespace` to the declaring path.
    ///
    /// A no-op for `param` shapes other than `namespace`, and for calls that
    /// don't match any of `param`'s members — the caller tries every declared
    /// tool via `unnormalize` until one claims the call.
    fn unnormalize(&self, param: &Value, call: &mut FunctionToolCall) {
        if call.namespace.is_some() {
            return;
        }
        let Ok(tool) = serde_json::from_value::<CodexParams>(param.clone()) else {
            return;
        };
        let Some((ns, member_name)) = find_namespace_member_by_flat_name(&call.name, &tool) else {
            return;
        };
        tracing::debug!(
            upstream_name = %call.name,
            namespace = %ns,
            member = %member_name,
            "reverse-normalized flattened codex namespace call"
        );
        call.namespace = Some(ns);
        call.name = member_name;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::io::OutputItem;
    use crate::types::tools::FunctionToolParam;
    use serde_json::json;

    fn function_tool(name: &str) -> ResponsesTool {
        ResponsesTool::Function(FunctionToolParam {
            name: name.try_into().unwrap(),
            description: None,
            parameters: None,
            strict: None,
        })
    }

    fn normalize_one(tool: &CodexParams) -> Vec<FunctionTool> {
        CodexToolHandler.normalize(&serde_json::to_value(tool).unwrap())
    }

    #[test]
    fn normalize_tool_search_becomes_function() {
        let params = json!({"type": "object", "properties": {"query": {"type": "string"}}});
        let tool = CodexParams::ToolSearch {
            description: Some("Search".into()),
            parameters: Some(params.clone()),
        };
        let out = normalize_one(&tool);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "tool_search");
        assert_eq!(out[0].parameters.as_ref().unwrap(), &params);
    }

    #[test]
    fn normalize_custom_becomes_function() {
        let fmt = json!({"type": "grammar"});
        let tool = CodexParams::Custom {
            name: "lark_parser".into(),
            description: None,
            format: Some(fmt.clone()),
        };
        let out = normalize_one(&tool);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "lark_parser");
        assert_eq!(out[0].parameters.as_ref().unwrap(), &fmt);
    }

    #[test]
    fn normalize_namespace_flattens_subtools() {
        let tool = CodexParams::Namespace {
            name: "mcp__github".into(),
            description: None,
            tools: vec![function_tool("create_issue"), function_tool("search_code")],
        };
        let out = normalize_codex_tool(&tool);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "agentic_ns__mcp__github__create_issue");
        assert_eq!(out[1].name, "agentic_ns__mcp__github__search_code");
    }

    #[test]
    fn normalize_nested_namespace_recurses() {
        let inner = ResponsesTool::Codex(CodexParams::Namespace {
            name: "inner".into(),
            description: None,
            tools: vec![function_tool("leaf")],
        });
        let tool = CodexParams::Namespace {
            name: "outer".into(),
            description: None,
            tools: vec![inner],
        };
        let out = normalize_codex_tool(&tool);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "agentic_ns__outer__inner__leaf");
    }

    #[test]
    fn normalize_namespace_drops_non_function_subtools() {
        let tool = CodexParams::Namespace {
            name: "ns".into(),
            description: None,
            tools: vec![function_tool("ns_tool"), ResponsesTool::Codex(CodexParams::Unknown)],
        };
        let out = normalize_codex_tool(&tool);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "agentic_ns__ns__ns_tool");
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
        let tool: CodexParams = serde_json::from_value(json).unwrap();
        let out = normalize_codex_tool(&tool);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "agentic_ns__mcp__github__create_issue");
    }

    #[test]
    fn deserialize_tool_search_from_json() {
        let json = json!({
            "type": "tool_search",
            "description": "Search for deferred tools",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        });
        let tool: CodexParams = serde_json::from_value(json).unwrap();
        assert_eq!(normalize_one(&tool)[0].name, "tool_search");
    }

    #[test]
    fn deserialize_custom_from_json() {
        let json = json!({
            "type": "custom",
            "name": "lark_parser",
            "description": "Parse Lark grammar files",
            "format": {"type": "grammar", "syntax": "lark"}
        });
        let tool: CodexParams = serde_json::from_value(json).unwrap();
        assert_eq!(normalize_one(&tool)[0].name, "lark_parser");
    }

    #[test]
    fn responses_tool_routes_namespace_to_codex_variant() {
        let json = json!({"type": "namespace", "name": "ns", "tools": []});
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        assert!(matches!(tool, ResponsesTool::Codex(CodexParams::Namespace { .. })));
    }

    #[test]
    fn responses_tool_routes_function_to_function_variant_not_codex() {
        let json = json!({"type": "function", "name": "f1"});
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        assert!(matches!(tool, ResponsesTool::Function(_)));
    }

    #[test]
    fn validate_rejects_empty_namespace_name() {
        let param = json!({"type": "namespace", "name": "", "tools": []});
        assert!(CodexToolHandler.validate(&param).is_err());
    }

    #[test]
    fn validate_rejects_empty_custom_name() {
        let param = json!({"type": "custom", "name": ""});
        assert!(CodexToolHandler.validate(&param).is_err());
    }

    #[test]
    fn validate_accepts_tool_search() {
        let param = json!({"type": "tool_search"});
        assert!(CodexToolHandler.validate(&param).is_ok());
    }

    fn function_call(name: &str) -> FunctionToolCall {
        FunctionToolCall {
            id: "fc_1".into(),
            call_id: "call_1".into(),
            name: name.into(),
            arguments: "{}".into(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        }
    }

    #[test]
    fn unnormalize_splits_flattened_member_name_back_apart() {
        let namespace = CodexParams::Namespace {
            name: "mcp__agentic_fixture".into(),
            description: None,
            tools: vec![function_tool("add_numbers")],
        };
        let param = serde_json::to_value(&namespace).unwrap();
        let mut call = function_call("agentic_ns__mcp__agentic_fixture__add_numbers");

        CodexToolHandler.unnormalize(&param, &mut call);

        assert_eq!(call.name, "add_numbers");
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
    }

    #[test]
    fn unnormalize_ignores_unrelated_call_names() {
        let namespace = CodexParams::Namespace {
            name: "mcp__agentic_fixture".into(),
            description: None,
            tools: vec![function_tool("add_numbers")],
        };
        let param = serde_json::to_value(&namespace).unwrap();
        let mut call = function_call("exec_command");

        CodexToolHandler.unnormalize(&param, &mut call);

        assert_eq!(call.name, "exec_command");
        assert!(call.namespace.is_none());
    }

    #[test]
    fn unnormalize_is_noop_once_namespace_already_set() {
        let namespace = CodexParams::Namespace {
            name: "mcp__agentic_fixture".into(),
            description: None,
            tools: vec![function_tool("add_numbers")],
        };
        let param = serde_json::to_value(&namespace).unwrap();
        let mut call = function_call("agentic_ns__mcp__agentic_fixture__add_numbers");
        call.namespace = Some("already_resolved".into());

        CodexToolHandler.unnormalize(&param, &mut call);

        assert_eq!(call.name, "agentic_ns__mcp__agentic_fixture__add_numbers");
        assert_eq!(call.namespace.as_deref(), Some("already_resolved"));
    }

    #[test]
    fn unnormalize_output_items_round_trips_through_full_request_cycle() {
        let json = json!([{
            "type": "namespace",
            "name": "mcp__agentic_fixture",
            "tools": [
                {"type": "function", "name": "add_numbers", "parameters": {"type": "object"}}
            ]
        }]);
        let tools: Vec<ResponsesTool> = serde_json::from_value(json).unwrap();

        let normalized = tools
            .iter()
            .flat_map(ResponsesTool::to_function_tools)
            .collect::<Vec<_>>();
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].name, "agentic_ns__mcp__agentic_fixture__add_numbers");

        let mut output = vec![OutputItem::FunctionCall(function_call(&normalized[0].name))];
        ResponsesTool::unnormalize_output_items(Some(&tools), &mut output);

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.name, "add_numbers");
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
    }
}
