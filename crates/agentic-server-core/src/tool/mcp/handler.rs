use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::tool::{GatewayExecutor, ToolError, ToolHandler, ToolOutput, ToolType};
use crate::types::io::FunctionTool;
use crate::types::io::output::{FunctionToolCall, GatewayCallStatus, McpToolCall, OutputItem};
use crate::types::tools::McpDiscoveredToolParam;
use crate::utils::common::{deserialize_from_str_opt, deserialize_from_value, serialize_to_string};

use super::McpClient;

#[must_use]
pub(crate) fn output_item(
    call: &FunctionToolCall,
    output: &ToolOutput,
    status: GatewayCallStatus,
    config: &Value,
) -> OutputItem {
    let identity = mcp_call_identity(call, config);
    let arguments =
        deserialize_from_str_opt::<Value>(&call.arguments).unwrap_or_else(|| Value::Object(serde_json::Map::new()));
    let parsed_output = deserialize_from_str_opt::<Value>(&output.output);
    let error = if status == GatewayCallStatus::Failed {
        Some(error_text_from_output(&output.output))
    } else {
        None
    };
    let result = (status == GatewayCallStatus::Completed)
        .then(|| parsed_output.unwrap_or_else(|| Value::String(output.output.clone())));

    OutputItem::McpToolCall(McpToolCall::new(
        call_output_id(call),
        identity.server_label,
        identity.name,
        arguments,
        status,
        result,
        error,
    ))
}

#[must_use]
pub(crate) fn started_output_item(call: &FunctionToolCall, config: &Value) -> OutputItem {
    let identity = mcp_call_identity(call, config);
    let arguments =
        deserialize_from_str_opt::<Value>(&call.arguments).unwrap_or_else(|| Value::Object(serde_json::Map::new()));

    OutputItem::McpToolCall(McpToolCall::new(
        call_output_id(call),
        identity.server_label,
        identity.name,
        arguments,
        GatewayCallStatus::InProgress,
        None,
        None,
    ))
}

/// Executes one tool discovered from an MCP server.
///
/// A handler with no client is used only while normalizing the discovered tool
/// metadata stored on `McpToolParam` into model-visible function tools.
pub struct McpHandler {
    client: Option<Arc<McpClient>>,
}

#[derive(Deserialize)]
struct McpToolNormalizationParams {
    #[serde(rename = "_agentic_discovered_tools", default)]
    discovered_tools: Vec<McpDiscoveredToolParam>,
}

#[derive(Clone)]
pub struct McpDiscoveredHandler {
    pub param: McpDiscoveredToolParam,
    pub handler: Arc<McpHandler>,
}

impl McpHandler {
    #[must_use]
    pub const fn discovered_tool_spec_only() -> Self {
        Self { client: None }
    }

    #[must_use]
    pub fn tool_call(client: Arc<McpClient>) -> Self {
        Self { client: Some(client) }
    }

    pub async fn discovered_tool_handlers(
        server_label: &str,
        client: Arc<McpClient>,
        allowed_tools: Option<&[String]>,
    ) -> Vec<McpDiscoveredHandler> {
        let tools = match client.list_tools().await {
            Ok(tools) => tools,
            Err(error) => {
                tracing::warn!(
                    server_label,
                    error = %error,
                    "failed to list MCP tools"
                );
                return Vec::new();
            }
        };

        let mut discovered_handlers = Vec::new();
        let mut internal_names = HashMap::new();
        for tool in tools {
            let tool_name = tool.name.to_string();
            if allowed_tools.is_some_and(|allowed| !allowed.iter().any(|name| name == &tool_name)) {
                continue;
            }
            let internal_name = internal_mcp_tool_name(server_label, &tool_name, &mut internal_names);
            discovered_handlers.push(McpDiscoveredHandler {
                param: McpDiscoveredToolParam {
                    server_label: server_label.to_owned(),
                    tool_name,
                    internal_name,
                    tool,
                },
                handler: Arc::new(Self::tool_call(Arc::clone(&client))),
            });
        }

        discovered_handlers
    }

    /// Returns the spec-only MCP tool handler used during request normalization.
    #[must_use]
    pub const fn spec_from_param(_param: &Value) -> Self {
        Self::discovered_tool_spec_only()
    }
}

impl ToolHandler for McpHandler {
    fn tool_type(&self) -> ToolType {
        ToolType::Mcp
    }

    fn validate(&self, _param: &Value) -> Result<(), ToolError> {
        Ok(())
    }

    fn normalize(&self, param: &Value) -> Vec<FunctionTool> {
        match deserialize_from_value::<McpToolNormalizationParams>(param.clone()) {
            Ok(params) => params
                .discovered_tools
                .iter()
                .map(discovered_mcp_function_tool)
                .collect(),
            Err(error) => {
                tracing::warn!(error = %error, "invalid MCP tool param");
                Vec::new()
            }
        }
    }
}

impl GatewayExecutor for McpHandler {
    fn execute(
        &self,
        call_id: &str,
        _tool_name: &str,
        arguments: &str,
        config: &Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let call_id = call_id.to_owned();
        let arguments = arguments.to_owned();
        let config = config.clone();

        Box::pin(async move {
            let Some(client) = &self.client else {
                return Err(ToolError::Config(
                    "MCP tool spec-only handler cannot execute tools".to_owned(),
                ));
            };
            let param = mcp_tool_param(&config)?;
            let output = execute_tool_call(client, &param.server_label, &param.tool_name, &arguments).await?;

            Ok(ToolOutput { call_id, output })
        })
    }
}

async fn execute_tool_call(
    client: &McpClient,
    server_label: &str,
    mcp_tool_name: &str,
    arguments: &str,
) -> Result<String, ToolError> {
    let args = deserialize_from_str_opt::<Value>(arguments);

    let result = client
        .call_tool(mcp_tool_name, args)
        .await
        .map_err(|error| ToolError::Execution(format!("tools/call failed for MCP server '{server_label}': {error}")))?;

    mcp_tool_result_text(&result)
}

fn mcp_tool_result_text(result: &rmcp::model::CallToolResult) -> Result<String, ToolError> {
    let text = result
        .content
        .iter()
        .filter_map(|content| content.as_text().map(|text| text.text.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    let output = if !text.is_empty() {
        text
    } else if let Some(structured_content) = &result.structured_content {
        serialize_to_string(structured_content)
            .map_err(|error| ToolError::Execution(format!("failed to serialize MCP structured content: {error}")))?
    } else {
        serialize_to_string(&result.content)
            .map_err(|error| ToolError::Execution(format!("failed to serialize MCP tool content: {error}")))?
    };

    if result.is_error == Some(true) {
        Err(ToolError::Execution(output))
    } else {
        Ok(output)
    }
}

fn mcp_tool_param(value: &Value) -> Result<McpDiscoveredToolParam, ToolError> {
    deserialize_from_value::<McpDiscoveredToolParam>(value.clone())
        .map_err(|error| ToolError::Config(format!("invalid MCP tool config: {error}")))
}

pub(crate) fn discovered_mcp_function_tool(param: &McpDiscoveredToolParam) -> FunctionTool {
    mcp_tool_to_function_tool(&param.internal_name, &param.tool)
}

#[cfg(test)]
const INTERNAL_DISCOVERED_TOOLS_KEY: &str = "_agentic_discovered_tools";
const INTERNAL_MCP_PREFIX: &str = "mcp__";
const MAX_INTERNAL_TOOL_NAME_LEN: usize = 64;

fn internal_mcp_tool_name(server_label: &str, tool_name: &str, used: &mut HashMap<String, (String, String)>) -> String {
    let identity = (server_label.to_owned(), tool_name.to_owned());
    let base = sanitize_internal_tool_name(&format!("{INTERNAL_MCP_PREFIX}{server_label}__{tool_name}"));
    if base.len() <= MAX_INTERNAL_TOOL_NAME_LEN && used.get(&base).is_none_or(|existing| existing == &identity) {
        used.insert(base.clone(), identity);
        return base;
    }

    let mut attempt = 0_u32;
    loop {
        let hash_input = if attempt == 0 {
            format!("{server_label}:{tool_name}")
        } else {
            format!("{server_label}:{tool_name}:{attempt}")
        };
        let suffix = format!("__{:010x}", stable_name_hash(&hash_input) & 0xff_ffff_ffff);
        let prefix_len = MAX_INTERNAL_TOOL_NAME_LEN.saturating_sub(suffix.len());
        let candidate = format!("{}{}", &base[..base.len().min(prefix_len)], suffix);
        if used.get(&candidate).is_none_or(|existing| existing == &identity) {
            used.insert(candidate.clone(), identity);
            return candidate;
        }
        attempt = attempt.saturating_add(1);
    }
}

fn sanitize_internal_tool_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn stable_name_hash(value: &str) -> u64 {
    value.as_bytes().iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn mcp_tool_to_function_tool(name: &str, tool: &rmcp::model::Tool) -> FunctionTool {
    let mut parameters = Value::Object(tool.input_schema.as_ref().clone());

    if let Value::Object(object) = &mut parameters
        && object.get("properties").is_none_or(Value::is_null)
    {
        object.insert("properties".to_owned(), Value::Object(serde_json::Map::new()));
    }

    FunctionTool {
        type_: "function".to_owned(),
        name: name.to_owned(),
        description: tool.description.as_ref().map(ToString::to_string),
        parameters: Some(parameters),
        strict: Some(false),
    }
}

fn error_text_from_output(output: &str) -> String {
    deserialize_from_str_opt::<Value>(output)
        .and_then(|value| value.get("error").and_then(Value::as_str).map(str::to_owned))
        .filter(|error| !error.trim().is_empty())
        .unwrap_or_else(|| output.to_owned())
}

struct McpCallIdentity {
    server_label: String,
    name: String,
}

fn mcp_call_identity(call: &FunctionToolCall, config: &Value) -> McpCallIdentity {
    deserialize_from_value::<McpDiscoveredToolParam>(config.clone()).map_or_else(
        |error| {
            tracing::warn!(error = %error, "invalid MCP tool identity config");
            McpCallIdentity {
                server_label: String::new(),
                name: call.name.clone(),
            }
        },
        |discovered| McpCallIdentity {
            server_label: discovered.server_label,
            name: discovered.tool_name,
        },
    )
}

fn call_output_id(call: &FunctionToolCall) -> String {
    if let Some(suffix) = call.id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("mcp_{suffix}");
    }
    if let Some(suffix) = call.call_id.strip_prefix("call_").filter(|suffix| !suffix.is_empty()) {
        return format!("mcp_{suffix}");
    }
    crate::utils::uuid7_str("mcp_")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discovered_param() -> McpDiscoveredToolParam {
        McpDiscoveredToolParam {
            server_label: "counter".to_owned(),
            tool_name: "increment".to_owned(),
            internal_name: "mcp__counter__increment".to_owned(),
            tool: serde_json::from_value(serde_json::json!({
                "name": "increment",
                "description": "Increment the counter",
                "inputSchema": {"type": "object"}
            }))
            .expect("valid MCP tool"),
        }
    }

    #[test]
    fn native_mcp_param_without_discovery_normalizes_to_no_functions() {
        let param = serde_json::json!({
            "server_label": "counter",
            "server_url": "http://127.0.0.1:8000/mcp"
        });

        let handler = McpHandler::spec_from_param(&param);

        assert!(handler.normalize(&param).is_empty());
    }

    #[test]
    fn discovered_tool_normalizes_to_function_tool() {
        let handler = McpHandler::discovered_tool_spec_only();
        let config = serde_json::json!({
            (INTERNAL_DISCOVERED_TOOLS_KEY): [discovered_param()]
        });

        let normalized = handler.normalize(&config);

        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].name, "mcp__counter__increment");
        assert_eq!(
            normalized[0].parameters.as_ref().unwrap()["properties"],
            serde_json::json!({})
        );
    }

    #[test]
    fn internal_tool_names_include_server_and_tool_identity() {
        let mut used = HashMap::new();

        let name = internal_mcp_tool_name("counter server", "increment/value", &mut used);

        assert_eq!(name, "mcp__counter_server__increment_value");
    }

    #[test]
    fn discovered_tool_output_uses_public_mcp_identity() {
        let call = FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "mcp__counter__increment".to_owned(),
            arguments: "{}".to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let output = ToolOutput {
            call_id: call.call_id.clone(),
            output: "1".to_owned(),
        };
        let config = serde_json::to_value(discovered_param()).expect("serializable discovered tool");

        let OutputItem::McpToolCall(item) = output_item(&call, &output, GatewayCallStatus::Completed, &config) else {
            panic!("expected mcp_tool_call");
        };

        assert_eq!(item.server, "counter");
        assert_eq!(item.tool, "increment");
        assert_eq!(item.arguments, serde_json::json!({}));
        assert_eq!(item.result, Some(serde_json::json!(1)));
    }

    #[test]
    fn successful_mcp_result_exposes_text_instead_of_protocol_envelope() {
        let result = serde_json::from_value::<rmcp::model::CallToolResult>(serde_json::json!({
            "content": [{"type": "text", "text": "42"}],
            "isError": false
        }))
        .expect("valid MCP result");

        assert_eq!(mcp_tool_result_text(&result).unwrap(), "42");
    }

    #[test]
    fn mcp_error_result_becomes_execution_failure() {
        let result = serde_json::from_value::<rmcp::model::CallToolResult>(serde_json::json!({
            "content": [{"type": "text", "text": "missing field `b`"}],
            "isError": true
        }))
        .expect("valid MCP result");

        let error = mcp_tool_result_text(&result).unwrap_err();
        assert!(matches!(error, ToolError::Execution(message) if message == "missing field `b`"));
    }

    #[test]
    fn failed_mcp_output_uses_execution_error_text() {
        let call = FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "mcp__counter__sum".to_owned(),
            arguments: r#"{"a":40}"#.to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let output = ToolOutput {
            call_id: call.call_id.clone(),
            output: r#"{"error":"missing field `b`"}"#.to_owned(),
        };
        let mut param = discovered_param();
        param.tool_name = "sum".to_owned();
        let config = serde_json::to_value(param).expect("serializable discovered tool");

        let item = output_item(&call, &output, GatewayCallStatus::Failed, &config);
        let json = serde_json::to_value(item).expect("serializable mcp_tool_call");

        assert_eq!(json["status"], "failed");
        assert!(json["result"].is_null());
        assert_eq!(json["error"], "missing field `b`");
    }
}
