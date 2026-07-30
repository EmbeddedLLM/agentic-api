use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::tool::{GatewayExecutor, ToolError, ToolHandler, ToolOutput, ToolType};
use crate::types::io::FunctionTool;
use crate::types::io::output::{FunctionToolCall, GatewayCallStatus, McpCall, McpCallError, McpCallStatus, OutputItem};
use crate::types::tools::{McpDiscoveredToolParam, ResponsesTool};
use crate::utils::common::{
    deserialize_from_str, deserialize_from_str_opt, deserialize_from_value, serialize_to_string,
};

use super::{McpClient, McpError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct McpToolRef {
    server_label: String,
    tool_name: String,
}

impl From<&McpDiscoveredToolParam> for McpToolRef {
    fn from(param: &McpDiscoveredToolParam) -> Self {
        Self {
            server_label: param.server_label.clone(),
            tool_name: param.tool_name.clone(),
        }
    }
}

/// Request-scoped mapping from model-visible internal names to public MCP
/// server and tool identities.
#[derive(Clone, Debug, Default)]
pub(crate) struct McpToolMap {
    calls: HashMap<String, McpToolRef>,
}

impl McpToolMap {
    pub(crate) fn record(&mut self, internal_name: String, tool_ref: McpToolRef) {
        debug_assert!(self.calls.insert(internal_name, tool_ref).is_none());
    }

    pub(crate) fn tool_ref(&self, internal_name: &str) -> Option<&McpToolRef> {
        self.calls.get(internal_name)
    }

    pub(crate) fn contains_server_label(&self, server_label: &str) -> bool {
        self.calls
            .values()
            .any(|tool_ref| tool_ref.server_label == server_label)
    }
}

#[must_use]
pub(crate) fn output_item(
    call: &FunctionToolCall,
    output: &ToolOutput,
    status: GatewayCallStatus,
    tool_ref: &McpToolRef,
) -> OutputItem {
    let error = if status == GatewayCallStatus::Failed {
        Some(McpCallError::tool_execution(error_text_from_output(&output.output)))
    } else {
        None
    };
    let successful_output = (status == GatewayCallStatus::Completed).then(|| output.output.clone());

    OutputItem::McpCall(McpCall::new(
        call_output_id(call),
        tool_ref.server_label.clone(),
        tool_ref.tool_name.clone(),
        call.arguments.clone(),
        status.into(),
        successful_output,
        error,
    ))
}

#[must_use]
pub(crate) fn started_output_item(call: &FunctionToolCall, tool_ref: &McpToolRef) -> OutputItem {
    OutputItem::McpCall(McpCall::new(
        call_output_id(call),
        tool_ref.server_label.clone(),
        tool_ref.tool_name.clone(),
        "",
        McpCallStatus::InProgress,
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
    /// Validates request-level MCP server identities before any discovery I/O.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when multiple MCP declarations use the
    /// same `server_label`.
    pub(crate) fn validate_server_labels(tools: &[ResponsesTool]) -> Result<(), ToolError> {
        let mut server_labels = HashSet::new();
        for param in tools.iter().filter_map(|tool| match tool {
            ResponsesTool::Mcp(param) => Some(param),
            _ => None,
        }) {
            if !server_labels.insert(param.server_label.clone()) {
                return Err(ToolError::Config(format!(
                    "duplicate MCP declarations are not allowed for server_label '{}'",
                    param.server_label
                )));
            }
        }
        Ok(())
    }

    #[must_use]
    pub const fn discovered_tool_spec_only() -> Self {
        Self { client: None }
    }

    #[must_use]
    pub fn tool_call(client: Arc<McpClient>) -> Self {
        Self { client: Some(client) }
    }

    /// Discovers and normalizes the tools exposed by one MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Execution`] when the server's `tools/list`
    /// operation fails or times out.
    pub async fn discovered_tool_handlers(
        server_label: &str,
        client: Arc<McpClient>,
        allowed_tools: Option<&[String]>,
    ) -> Result<Vec<McpDiscoveredHandler>, ToolError> {
        let tools = client
            .list_tools()
            .await
            .map_err(|error| mcp_discovery_error(server_label, &error))?;

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

        Ok(discovered_handlers)
    }

    /// Returns the spec-only MCP tool handler used during request normalization.
    #[must_use]
    pub const fn spec_from_param(_param: &Value) -> Self {
        Self::discovered_tool_spec_only()
    }
}

fn mcp_discovery_error(server_label: &str, error: &McpError) -> ToolError {
    ToolError::Execution(format!("tools/list failed for MCP server '{server_label}': {error}"))
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
    let args = parse_tool_arguments(arguments)?;

    let result = client
        .call_tool(mcp_tool_name, Some(args))
        .await
        .map_err(|error| ToolError::Execution(format!("tools/call failed for MCP server '{server_label}': {error}")))?;

    mcp_tool_result_text(&result)
}

fn parse_tool_arguments(arguments: &str) -> Result<Value, ToolError> {
    let arguments = deserialize_from_str::<Value>(arguments)
        .map_err(|error| ToolError::Execution(format!("invalid MCP tool arguments: {error}")))?;
    if !arguments.is_object() {
        return Err(ToolError::Execution(
            "MCP tool arguments must be a JSON object".to_owned(),
        ));
    }
    Ok(arguments)
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

fn call_output_id(call: &FunctionToolCall) -> String {
    if let Some(suffix) = call.id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("mcp_{suffix}");
    }
    if let Some(suffix) = call.call_id.strip_prefix("call_").filter(|suffix| !suffix.is_empty()) {
        return format!("mcp_{suffix}");
    }
    let source_identity = format!("{}\0{}", call.id, call.call_id);
    format!("mcp_{:016x}", stable_name_hash(&source_identity))
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
    fn tools_list_failure_preserves_upstream_cause_as_execution_error() {
        let upstream_error = super::super::McpError::Timeout {
            operation: super::super::McpOperation::ListTools,
        };

        let error = mcp_discovery_error("counter", &upstream_error);

        assert!(matches!(error, ToolError::Execution(_)));
        assert!(error.to_string().contains("tools/list failed for MCP server 'counter'"));
        assert!(error.to_string().contains("timed out during tools/list"));
    }

    #[test]
    fn mcp_tool_arguments_require_valid_json_object() {
        assert_eq!(
            parse_tool_arguments(r#"{"amount":1}"#).unwrap(),
            serde_json::json!({"amount": 1})
        );

        let malformed = parse_tool_arguments(r#"{"amount":"#).unwrap_err();
        assert!(matches!(
            malformed,
            ToolError::Execution(message) if message.contains("invalid MCP tool arguments")
        ));

        let non_object = parse_tool_arguments("null").unwrap_err();
        assert!(matches!(
            non_object,
            ToolError::Execution(message) if message == "MCP tool arguments must be a JSON object"
        ));
    }

    #[test]
    fn internal_tool_names_include_server_and_tool_identity() {
        let mut used = HashMap::new();

        let name = internal_mcp_tool_name("counter server", "increment/value", &mut used);

        assert_eq!(name, "mcp__counter_server__increment_value");
    }

    #[test]
    fn tool_map_resolves_internal_name_to_public_mcp_identity() {
        let param = discovered_param();
        let tool_ref = McpToolRef::from(&param);
        let mut map = McpToolMap::default();

        map.record(param.internal_name.clone(), tool_ref.clone());

        assert_eq!(map.tool_ref(&param.internal_name), Some(&tool_ref));
        assert!(map.contains_server_label("counter"));
        assert!(!map.contains_server_label("missing"));
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
        let tool_ref = McpToolRef::from(&discovered_param());

        let OutputItem::McpCall(item) = output_item(&call, &output, GatewayCallStatus::Completed, &tool_ref) else {
            panic!("expected mcp_call");
        };

        assert_eq!(item.server_label, "counter");
        assert_eq!(item.name, "increment");
        assert_eq!(item.arguments, "{}");
        assert_eq!(item.output.as_deref(), Some("1"));
    }

    #[test]
    fn prefixless_function_ids_reuse_public_mcp_id_across_lifecycle() {
        let call = FunctionToolCall {
            id: "provider-item-1".to_owned(),
            call_id: "provider-call-1".to_owned(),
            name: "mcp__counter__increment".to_owned(),
            arguments: "{}".to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let output = ToolOutput {
            call_id: call.call_id.clone(),
            output: "1".to_owned(),
        };
        let tool_ref = McpToolRef::from(&discovered_param());

        let OutputItem::McpCall(started) = started_output_item(&call, &tool_ref) else {
            panic!("expected started mcp_call");
        };
        let OutputItem::McpCall(completed) = output_item(&call, &output, GatewayCallStatus::Completed, &tool_ref)
        else {
            panic!("expected completed mcp_call");
        };

        assert!(started.id.starts_with("mcp_"));
        assert_eq!(started.id, completed.id);
    }

    #[test]
    fn idless_parallel_calls_receive_distinct_public_mcp_ids() {
        let calls = [
            serde_json::from_value::<FunctionToolCall>(serde_json::json!({
                "name": "mcp__counter__increment",
                "arguments": "{}"
            }))
            .expect("valid first function call"),
            serde_json::from_value::<FunctionToolCall>(serde_json::json!({
                "name": "mcp__counter__increment",
                "arguments": "{}"
            }))
            .expect("valid second function call"),
        ];
        let tool_ref = McpToolRef::from(&discovered_param());

        let public_ids = calls
            .iter()
            .map(|call| {
                let OutputItem::McpCall(started) = started_output_item(call, &tool_ref) else {
                    panic!("expected started mcp_call");
                };
                let output = ToolOutput {
                    call_id: call.call_id.clone(),
                    output: "1".to_owned(),
                };
                let OutputItem::McpCall(completed) =
                    output_item(call, &output, GatewayCallStatus::Completed, &tool_ref)
                else {
                    panic!("expected completed mcp_call");
                };

                assert_eq!(started.id, completed.id);
                started.id
            })
            .collect::<Vec<_>>();

        assert_ne!(public_ids[0], public_ids[1]);
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
    fn failed_mcp_output_uses_openai_structured_error() {
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
        let tool_ref = McpToolRef::from(&param);

        let item = output_item(&call, &output, GatewayCallStatus::Failed, &tool_ref);
        let json = serde_json::to_value(item).expect("serializable mcp_call");

        assert_eq!(json["status"], "failed");
        assert!(json["output"].is_null());
        assert_eq!(json["error"]["type"], "mcp_tool_execution_error");
        assert_eq!(json["error"]["content"][0]["type"], "text");
        assert_eq!(json["error"]["content"][0]["text"], "missing field `b`");
        assert!(json["error"]["content"][0]["annotations"].is_null());
        assert!(json["error"]["content"][0]["meta"].is_null());
    }
}
