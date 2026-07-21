use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::tool::{GatewayExecutor, ToolError, ToolHandler, ToolOutput, ToolType};
use crate::types::io::FunctionTool;
use crate::types::io::output::{FunctionToolCall, GatewayCallStatus, McpCall, OutputItem};
use crate::types::tools::{McpDiscoveredToolParam, McpToolParam};
use crate::utils::common::{
    deserialize_from_str, deserialize_from_str_opt, deserialize_from_value, serialize_to_string,
};

use super::{McpClient, McpClientPool, READ_MCP_RESOURCE_TOOL_NAME, ReadResourceArgs, read_mcp_resource_spec};

#[must_use]
pub(crate) fn output_item(
    call: &FunctionToolCall,
    output: &ToolOutput,
    status: GatewayCallStatus,
    config: &Value,
) -> OutputItem {
    let identity = mcp_call_identity(call, config);
    let parsed_output = deserialize_from_str_opt::<Value>(&output.output);
    let error = if status == GatewayCallStatus::Failed {
        parsed_output
            .as_ref()
            .and_then(error_from_output)
            .or_else(|| Some(output.output.clone()))
    } else {
        None
    };
    let successful_output = (status == GatewayCallStatus::Completed).then(|| output.output.clone());

    OutputItem::McpCall(McpCall::new(
        call_output_id(call),
        identity.server_label,
        identity.name,
        call.arguments.clone(),
        status,
        successful_output,
        error,
    ))
}

#[must_use]
pub(crate) fn started_output_item(call: &FunctionToolCall, config: &Value) -> OutputItem {
    let identity = mcp_call_identity(call, config);

    OutputItem::McpCall(McpCall::new(
        call_output_id(call),
        identity.server_label,
        identity.name,
        "",
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
    /// Selects the MCP operation represented by an internal/model-visible tool
    /// name. `read_mcp_resource` is currently the only built-in resource
    /// operation; names produced by MCP `tools/list` are tool calls.
    #[must_use]
    pub fn from_internal_name(internal_name: &str) -> Self {
        if internal_name == READ_MCP_RESOURCE_TOOL_NAME {
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

#[derive(Deserialize)]
struct McpToolNormalizationParams {
    #[serde(rename = "_agentic_discovered_tools", default)]
    discovered_tools: Vec<McpDiscoveredToolParam>,
}

#[derive(Debug, Default)]
pub struct McpHandlerFactory;

#[derive(Clone)]
pub struct McpDiscoveredHandler {
    pub param: McpDiscoveredToolParam,
    pub handler: Arc<McpHandler>,
}

impl McpHandlerFactory {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub async fn from_params(&self, internal_name: &str, param: &McpToolParam) -> Option<McpHandler> {
        let pool = Arc::new(McpClientPool::from_params(std::slice::from_ref(param)).await);
        match McpSpec::from_internal_name(internal_name) {
            McpSpec::Resources => Some(self.read_resource(pool)),
            McpSpec::Tool => pool.client_for_param(param).map(|client| self.tool_call(client)),
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

    pub async fn discovered_tool_handlers(
        &self,
        factory: &McpHandlerFactory,
        allowed_tools: Option<&[String]>,
    ) -> Vec<McpDiscoveredHandler> {
        let McpHandlerKind::ReadResource {
            resource: Some(pool), ..
        } = &self.kind
        else {
            return Vec::new();
        };

        let mut discovered_handlers = Vec::new();
        let mut internal_names = HashMap::new();
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
                if allowed_tools.is_some_and(|allowed| !allowed.iter().any(|name| name == &tool_name)) {
                    continue;
                }
                let internal_name = internal_mcp_tool_name(server_label, &tool_name, &mut internal_names);
                discovered_handlers.push(McpDiscoveredHandler {
                    param: McpDiscoveredToolParam {
                        server_label: server_label.clone(),
                        tool_name,
                        internal_name,
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
        let spec = deserialize_from_value::<McpToolNormalizationParams>(param.clone())
            .ok()
            .and_then(|params| params.discovered_tools.into_iter().next())
            .map_or(McpSpec::Tool, |tool| McpSpec::from_internal_name(&tool.internal_name));

        match spec {
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
            } => match deserialize_from_value::<McpToolNormalizationParams>(param.clone()) {
                Ok(params) => params
                    .discovered_tools
                    .iter()
                    .map(discovered_mcp_function_tool)
                    .collect(),
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

struct McpCallIdentity {
    server_label: String,
    name: String,
}

fn mcp_call_identity(call: &FunctionToolCall, config: &Value) -> McpCallIdentity {
    if let Ok(discovered) = deserialize_from_value::<McpDiscoveredToolParam>(config.clone()) {
        return McpCallIdentity {
            server_label: discovered.server_label,
            name: discovered.tool_name,
        };
    }

    let arguments = arguments_value(&call.arguments);
    let declared_server = deserialize_from_value::<McpToolParam>(config.clone())
        .ok()
        .map(|declared| declared.server_label);
    McpCallIdentity {
        server_label: server_from_arguments(&arguments)
            .or(declared_server)
            .unwrap_or_default(),
        name: READ_MCP_RESOURCE_TOOL_NAME.to_owned(),
    }
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
    fn internal_name_selects_mcp_spec() {
        assert_eq!(
            McpSpec::from_internal_name(READ_MCP_RESOURCE_TOOL_NAME),
            McpSpec::Resources
        );
        assert_eq!(McpSpec::from_internal_name("mcp__counter__increment"), McpSpec::Tool);
    }

    #[test]
    fn native_mcp_param_without_discovery_stays_tool_spec() {
        let param = serde_json::json!({
            "server_label": "counter",
            "server_url": "http://127.0.0.1:8000/mcp"
        });

        let handler = McpHandler::spec_from_param(&param);

        assert_eq!(handler.spec(), McpSpec::Tool);
        assert!(handler.normalize(&param).is_empty());
    }

    #[test]
    fn builtin_resource_internal_name_selects_resource_spec() {
        let mut resource = discovered_param();
        resource.internal_name = READ_MCP_RESOURCE_TOOL_NAME.to_owned();
        let param = serde_json::json!({
            (INTERNAL_DISCOVERED_TOOLS_KEY): [resource]
        });

        let handler = McpHandler::spec_from_param(&param);

        assert_eq!(handler.spec(), McpSpec::Resources);
        assert_eq!(handler.normalize(&param)[0].name, READ_MCP_RESOURCE_TOOL_NAME);
    }

    #[test]
    fn tool_spec_normalizes_discovered_tool_to_function_tool() {
        let handler = McpHandler::discovered_tool_spec_only();
        assert_eq!(handler.spec(), McpSpec::Tool);
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
    fn resource_spec_normalizes_to_read_resource_function() {
        let handler = McpHandler::read_resource_spec_only();
        assert_eq!(handler.spec(), McpSpec::Resources);

        let normalized = handler.normalize(&Value::Null);

        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].name, super::super::READ_MCP_RESOURCE_TOOL_NAME);
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

        let OutputItem::McpCall(item) = output_item(&call, &output, GatewayCallStatus::Completed, &config) else {
            panic!("expected mcp_call");
        };

        assert_eq!(item.server_label, "counter");
        assert_eq!(item.name, "increment");
        assert_eq!(item.arguments, "{}");
        assert_eq!(item.output.as_deref(), Some("1"));
    }

    #[test]
    fn read_resource_output_is_also_a_public_mcp_call() {
        let call = FunctionToolCall {
            id: "fc_2".to_owned(),
            call_id: "call_2".to_owned(),
            name: READ_MCP_RESOURCE_TOOL_NAME.to_owned(),
            arguments: r#"{"server":"repo","uri":"repo://fixture"}"#.to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let output = ToolOutput {
            call_id: call.call_id.clone(),
            output: r#"{"contents":[]}"#.to_owned(),
        };
        let config = serde_json::json!({
            "server_label": "repo",
            "server_url": "http://localhost:8000/mcp"
        });

        let OutputItem::McpCall(item) = output_item(&call, &output, GatewayCallStatus::Completed, &config) else {
            panic!("expected mcp_call");
        };

        assert_eq!(item.server_label, "repo");
        assert_eq!(item.name, READ_MCP_RESOURCE_TOOL_NAME);
        assert_eq!(item.arguments, call.arguments);
        assert_eq!(item.output.as_deref(), Some(r#"{"contents":[]}"#));
    }
}
