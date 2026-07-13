use std::collections::HashMap;
use std::sync::Arc;

use crate::tool::{GatewayExecutor, ToolEntry, ToolType};
use crate::types::tools::McpToolParam;
use crate::utils::common::serialize_to_value_or_custom_default;

use super::{McpSpec, READ_MCP_RESOURCE_TOOL_NAME};

/// Registers `p` for gateway dispatch by connecting to the request-declared
/// MCP server and discovering its tools via [`build_mcp_registry`].
pub async fn insert_mcp_entry<S: std::hash::BuildHasher>(
    entries: &mut HashMap<String, ToolEntry, S>,
    p: &McpToolParam,
    handler: Option<Arc<dyn GatewayExecutor>>,
) {
    let Some(handler) = handler else {
        tracing::debug!(name = %p.name, "MCP tool skipped because no MCP handler is configured");
        return;
    };

    build_mcp_registry(p, entries, handler).await;
}

/// Registers the request-scoped MCP handler into `entries`, keyed by the name
/// the model will call.
pub async fn build_mcp_registry<S: std::hash::BuildHasher>(
    param: &McpToolParam,
    entries: &mut HashMap<String, ToolEntry, S>,
    handler: Arc<dyn GatewayExecutor>,
) {
    match McpSpec::from_param_name(param.name.as_str()) {
        McpSpec::Resources => register_read_resource(param, entries, handler),
        McpSpec::Tool => register_declared_tool_call(param, entries, handler),
    }
}

fn register_read_resource<S: std::hash::BuildHasher>(
    param: &McpToolParam,
    entries: &mut HashMap<String, ToolEntry, S>,
    handler: Arc<dyn GatewayExecutor>,
) {
    let config = serialize_to_value_or_custom_default(
        param,
        "MCP read_resource config serialization failed",
        |config| config,
        serde_json::Value::Null,
    );
    entries.insert(
        READ_MCP_RESOURCE_TOOL_NAME.to_owned(),
        ToolEntry {
            tool_type: ToolType::Mcp,
            config,
            server_label: None,
            handler: Some(handler),
        },
    );
}

fn register_declared_tool_call<S: std::hash::BuildHasher>(
    param: &McpToolParam,
    entries: &mut HashMap<String, ToolEntry, S>,
    handler: Arc<dyn GatewayExecutor>,
) {
    let config = serialize_to_value_or_custom_default(
        param,
        "MCP tool-call config serialization failed",
        |config| config,
        serde_json::Value::Null,
    );
    if entries
        .insert(
            param.name.as_str().to_owned(),
            ToolEntry {
                tool_type: ToolType::Mcp,
                config,
                server_label: param.server_label.clone(),
                handler: Some(handler),
            },
        )
        .is_some()
    {
        tracing::warn!(name = %param.name, "duplicate MCP tool name — previous definition overwritten");
    }
}
