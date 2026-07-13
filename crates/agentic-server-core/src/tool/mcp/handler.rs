use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::tool::{GatewayExecutor, ToolError, ToolHandler, ToolOutput, ToolType};
use crate::types::io::FunctionTool;
use crate::types::io::output::{FunctionToolCall, GatewayCallStatus, McpToolCall, OutputItem};
use crate::types::tools::{McpDiscoveredToolParam, McpToolParam};
use crate::utils::common::{
    deserialize_from_str, deserialize_from_str_opt, deserialize_from_value, deserialize_from_value_opt,
    serialize_to_string,
};

use super::{McpClient, McpClientPool, READ_MCP_RESOURCE_TOOL_NAME, ReadResourceArgs, read_mcp_resource_spec};

#[must_use]
pub(crate) fn output_item(call: &FunctionToolCall, output: &ToolOutput, status: GatewayCallStatus) -> OutputItem {
    let arguments = arguments_value(&call.arguments);
    let server = server_from_arguments(&arguments).unwrap_or_default();
    let parsed_output = deserialize_from_str_opt::<Value>(&output.output);
    let error = if status == GatewayCallStatus::Failed {
        parsed_output
            .as_ref()
            .and_then(error_from_output)
            .or_else(|| Some(output.output.clone()))
    } else {
        None
    };
    let result = (status == GatewayCallStatus::Completed)
        .then(|| parsed_output.unwrap_or_else(|| Value::String(output.output.clone())));

    OutputItem::McpToolCall(McpToolCall::new(
        call_output_id(call),
        server,
        call.name.clone(),
        arguments,
        status,
        result,
        error,
    ))
}

#[must_use]
pub(crate) fn started_output_item(call: &FunctionToolCall) -> OutputItem {
    let arguments = arguments_value(&call.arguments);
    let server = server_from_arguments(&arguments).unwrap_or_default();

    OutputItem::McpToolCall(McpToolCall::new(
        call_output_id(call),
        server,
        call.name.clone(),
        arguments,
        GatewayCallStatus::InProgress,
        None,
        None,
    ))
}

/// `*Spec` variants map a declared/discovered tool's shape to its upstream
/// `FunctionTool` form for `normalize()` — no live MCP connection involved.
/// `ReadResource`/`ToolCall` carry a real `pool`/`client`, built by the
/// registry once a connection exists, and are the only variants `execute()`
/// can dispatch through.
pub enum McpHandlerKind {
    ReadResourceSpec,
    ReadResource { pool: Arc<McpClientPool> },
    ToolCallSpec,
    ToolCall { client: Arc<McpClient> },
}

pub struct McpHandler {
    kind: McpHandlerKind,
}

impl McpHandler {
    #[must_use]
    pub fn read_resource_spec_only() -> Self {
        Self {
            kind: McpHandlerKind::ReadResourceSpec,
        }
    }

    #[must_use]
    pub fn read_resource(pool: Arc<McpClientPool>) -> Self {
        Self {
            kind: McpHandlerKind::ReadResource { pool },
        }
    }

    pub async fn from_params(params: &[McpToolParam]) -> Self {
        let pool = Arc::new(McpClientPool::from_params(params).await);
        Self::read_resource(pool)
    }

    #[must_use]
    pub fn pool(&self) -> Option<Arc<McpClientPool>> {
        match &self.kind {
            McpHandlerKind::ReadResource { pool } => Some(Arc::clone(pool)),
            McpHandlerKind::ReadResourceSpec | McpHandlerKind::ToolCallSpec | McpHandlerKind::ToolCall { .. } => None,
        }
    }

    #[must_use]
    pub fn discovered_tool_spec_only() -> Self {
        Self {
            kind: McpHandlerKind::ToolCallSpec,
        }
    }

    #[must_use]
    pub fn tool_call(client: Arc<McpClient>) -> Self {
        Self {
            kind: McpHandlerKind::ToolCall { client },
        }
    }

    /// Spec-only handler for normalizing a `ToolEntry` config with no live
    /// connection, picking `ReadResourceSpec` vs `ToolCallSpec` by inspecting
    /// `param`'s shape — mirrors the split `build_mcp_registry` makes per entry.
    #[must_use]
    pub fn spec_from_param(param: &Value) -> Self {
        let is_read_resource = deserialize_from_value_opt::<McpToolParam>(param.clone())
            .is_some_and(|declared| declared.name.as_str() == READ_MCP_RESOURCE_TOOL_NAME);
        if is_read_resource {
            Self::read_resource_spec_only()
        } else {
            Self::discovered_tool_spec_only()
        }
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
        match &self.kind {
            McpHandlerKind::ReadResourceSpec | McpHandlerKind::ReadResource { .. } => vec![read_mcp_resource_spec()],
            McpHandlerKind::ToolCallSpec | McpHandlerKind::ToolCall { .. } => {
                match deserialize_from_value::<McpDiscoveredToolParam>(param.clone()) {
                    Ok(discovered) => vec![mcp_tool_to_function_tool(&discovered.exposed_name, &discovered.tool)],
                    Err(error) => {
                        tracing::warn!(error = %error, "invalid MCP tool param");
                        Vec::new()
                    }
                }
            }
        }
    }
}

impl GatewayExecutor for McpHandler {
    fn execute(
        &self,
        call_id: &str,
        tool_name: &str,
        arguments: &str,
        config: &Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let call_id = call_id.to_owned();
        let tool_name = tool_name.to_owned();
        let arguments = arguments.to_owned();
        let config = config.clone();

        Box::pin(async move {
            let output = match &self.kind {
                McpHandlerKind::ReadResourceSpec => {
                    return Err(ToolError::Config(
                        "read_mcp_resource spec-only handler cannot execute tools".to_owned(),
                    ));
                }
                McpHandlerKind::ToolCallSpec => {
                    return Err(ToolError::Config(
                        "MCP tool spec-only handler cannot execute tools".to_owned(),
                    ));
                }
                McpHandlerKind::ReadResource { pool } => execute_read_resource(pool, &tool_name, &arguments).await?,
                McpHandlerKind::ToolCall { client } => {
                    let param = mcp_tool_param(&config)?;
                    execute_tool_call(client, &param.server_label, &param.tool_name, &arguments).await?
                }
            };

            Ok(ToolOutput { call_id, output })
        })
    }
}

async fn execute_read_resource(pool: &McpClientPool, tool_name: &str, arguments: &str) -> Result<String, ToolError> {
    if tool_name != READ_MCP_RESOURCE_TOOL_NAME {
        return Err(ToolError::Config(format!(
            "read_mcp_resource handler cannot execute tool '{tool_name}'"
        )));
    }

    let args = deserialize_from_str::<ReadResourceArgs>(arguments)
        .map_err(|error| ToolError::Execution(format!("invalid read_mcp_resource arguments: {error}")))?;

    let client = pool
        .get(&args.server)
        .ok_or_else(|| match pool.connection_error(&args.server) {
            Some(error) => ToolError::Execution(format!("MCP server '{}' failed to connect: {error}", args.server)),
            None => ToolError::Execution(format!("unknown MCP server: {}", args.server)),
        })?;

    let result = client
        .read_resource(&args.uri)
        .await
        .map_err(|error| ToolError::Execution(format!("resources/read failed: {error}")))?;

    serialize_mcp_result(&result, "resources/read")
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

    serialize_mcp_result(&result, "tools/call")
}

fn serialize_mcp_result(result: &impl serde::Serialize, operation: &str) -> Result<String, ToolError> {
    serialize_to_string(result)
        .map_err(|error| ToolError::Execution(format!("failed to serialize {operation} result: {error}")))
}

fn mcp_tool_param(value: &Value) -> Result<McpDiscoveredToolParam, ToolError> {
    deserialize_from_value::<McpDiscoveredToolParam>(value.clone())
        .map_err(|error| ToolError::Config(format!("invalid MCP tool config: {error}")))
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

fn arguments_value(arguments: &str) -> Value {
    deserialize_from_str_opt(arguments).unwrap_or_else(|| Value::Object(serde_json::Map::new()))
}

fn server_from_arguments(arguments: &Value) -> Option<String> {
    arguments
        .get("server")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|server| !server.is_empty())
        .map(str::to_owned)
}

fn error_from_output(output: &Value) -> Option<String> {
    output
        .get("error")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|error| !error.is_empty())
        .map(str::to_owned)
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
