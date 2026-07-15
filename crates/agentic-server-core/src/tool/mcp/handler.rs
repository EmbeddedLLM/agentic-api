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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpSpec {
    Resources,
    Tool,
}

impl McpSpec {
    #[must_use]
    pub fn from_param_name(name: &str) -> Self {
        if name == READ_MCP_RESOURCE_TOOL_NAME {
            Self::Resources
        } else {
            Self::Tool
        }
    }
}

/// Maps the model-facing MCP spec shape to the resource needed to execute it.
pub enum McpHandlerKind {
    ReadResource {
        spec: McpSpec,
        resource: Option<Arc<McpClientPool>>,
    },
    ToolCall {
        spec: McpSpec,
        client: Option<Arc<McpClient>>,
    },
}

pub struct McpHandler {
    kind: McpHandlerKind,
}

#[derive(Debug, Default)]
pub struct McpHandlerFactory;

pub struct McpDiscoveredHandler {
    pub param: McpDiscoveredToolParam,
    pub handler: Arc<McpHandler>,
}

impl McpHandlerFactory {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub async fn from_params(&self, param: &McpToolParam) -> Option<McpHandler> {
        let resource = Arc::new(McpClientPool::from_params(std::slice::from_ref(param)).await);
        match McpSpec::from_param_name(param.name.as_str()) {
            McpSpec::Resources => Some(self.read_resource(resource)),
            McpSpec::Tool => resource.client_for_param(param).map(|client| self.tool_call(client)),
        }
    }

    #[must_use]
    pub const fn read_resource_spec(&self) -> McpSpec {
        McpSpec::Resources
    }

    #[must_use]
    pub const fn tool_call_spec(&self) -> McpSpec {
        McpSpec::Tool
    }

    #[must_use]
    pub fn read_resource(&self, resource: Arc<McpClientPool>) -> McpHandler {
        McpHandler::with_kind(McpHandlerKind::ReadResource {
            spec: self.read_resource_spec(),
            resource: Some(resource),
        })
    }

    #[must_use]
    pub fn tool_call(&self, client: Arc<McpClient>) -> McpHandler {
        McpHandler::with_kind(McpHandlerKind::ToolCall {
            spec: self.tool_call_spec(),
            client: Some(client),
        })
    }
}

impl ToolHandler for McpHandlerFactory {
    fn tool_type(&self) -> ToolType {
        ToolType::Mcp
    }

    fn validate(&self, _param: &Value) -> Result<(), ToolError> {
        Ok(())
    }

    fn normalize(&self, param: &Value) -> Vec<FunctionTool> {
        McpHandler::spec_from_param(param).normalize(param)
    }
}

impl McpHandler {
    #[must_use]
    pub fn with_kind(kind: McpHandlerKind) -> Self {
        Self { kind }
    }

    #[must_use]
    pub const fn spec(&self) -> McpSpec {
        match &self.kind {
            McpHandlerKind::ReadResource { spec, .. } | McpHandlerKind::ToolCall { spec, .. } => *spec,
        }
    }

    #[must_use]
    pub fn read_resource_spec_only() -> Self {
        Self::with_kind(McpHandlerKind::ReadResource {
            spec: McpSpec::Resources,
            resource: None,
        })
    }

    #[must_use]
    pub fn read_resource(pool: Arc<McpClientPool>) -> Self {
        Self::with_kind(McpHandlerKind::ReadResource {
            spec: McpSpec::Resources,
            resource: Some(pool),
        })
    }

    #[must_use]
    pub fn discovered_tool_spec_only() -> Self {
        Self::with_kind(McpHandlerKind::ToolCall {
            spec: McpSpec::Tool,
            client: None,
        })
    }

    #[must_use]
    pub fn tool_call(client: Arc<McpClient>) -> Self {
        Self::with_kind(McpHandlerKind::ToolCall {
            spec: McpSpec::Tool,
            client: Some(client),
        })
    }

    pub async fn discovered_tool_handlers(&self, factory: &McpHandlerFactory) -> Vec<McpDiscoveredHandler> {
        let McpHandlerKind::ReadResource {
            resource: Some(pool), ..
        } = &self.kind
        else {
            return Vec::new();
        };

        let mut discovered_handlers = Vec::new();
        for (server_label, client) in pool.iter() {
            let tools = match client.list_tools().await {
                Ok(tools) => tools,
                Err(error) => {
                    tracing::warn!(
                        server_label = %server_label,
                        error = %error,
                        "failed to list MCP tools"
                    );
                    continue;
                }
            };

            for tool in tools {
                let tool_name = tool.name.to_string();
                let exposed_name = format!("{server_label}__{tool_name}");
                discovered_handlers.push(McpDiscoveredHandler {
                    param: McpDiscoveredToolParam {
                        server_label: server_label.clone(),
                        tool_name,
                        exposed_name,
                        tool,
                    },
                    handler: Arc::new(factory.tool_call(Arc::clone(client))),
                });
            }
        }

        discovered_handlers
    }

    /// Spec-only handler for normalizing a `ToolEntry` config with no live
    /// connection, picking resource vs tool-call shape by inspecting `param`.
    #[must_use]
    pub fn spec_from_param(param: &Value) -> Self {
        match deserialize_from_value_opt::<McpToolParam>(param.clone()).map_or(McpSpec::Tool, |declared| {
            McpSpec::from_param_name(declared.name.as_str())
        }) {
            McpSpec::Resources => Self::read_resource_spec_only(),
            McpSpec::Tool => Self::discovered_tool_spec_only(),
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
            McpHandlerKind::ReadResource {
                spec: McpSpec::Resources,
                ..
            } => vec![read_mcp_resource_spec()],
            McpHandlerKind::ToolCall {
                spec: McpSpec::Tool, ..
            } => match deserialize_from_value::<McpDiscoveredToolParam>(param.clone()) {
                Ok(discovered) => vec![mcp_tool_to_function_tool(&discovered.exposed_name, &discovered.tool)],
                Err(error) => {
                    tracing::warn!(error = %error, "invalid MCP tool param");
                    Vec::new()
                }
            },
            McpHandlerKind::ReadResource {
                spec: McpSpec::Tool, ..
            }
            | McpHandlerKind::ToolCall {
                spec: McpSpec::Resources,
                ..
            } => {
                tracing::warn!("invalid MCP handler kind/spec pairing");
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
            let output = match &self.kind {
                McpHandlerKind::ReadResource { resource, .. } => {
                    let Some(pool) = resource else {
                        return Err(ToolError::Config(
                            "read_mcp_resource spec-only handler cannot execute tools".to_owned(),
                        ));
                    };
                    execute_read_resource(pool, &arguments).await?
                }
                McpHandlerKind::ToolCall { client, .. } => {
                    let Some(client) = client else {
                        return Err(ToolError::Config(
                            "MCP tool spec-only handler cannot execute tools".to_owned(),
                        ));
                    };
                    let param = mcp_tool_param(&config)?;
                    execute_tool_call(client, &param.server_label, &param.tool_name, &arguments).await?
                }
            };

            Ok(ToolOutput { call_id, output })
        })
    }
}

async fn execute_read_resource(pool: &McpClientPool, arguments: &str) -> Result<String, ToolError> {
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
