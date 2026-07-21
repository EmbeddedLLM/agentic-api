use std::collections::HashMap;
use std::sync::Arc;

use crate::tool::{GatewayExecutor, ToolEntry, ToolType};
use crate::types::tools::{McpDiscoveredToolParam, McpToolParam};
use crate::utils::common::serialize_to_value_or_custom_default;

use super::{McpDiscoveredHandler, READ_MCP_RESOURCE_TOOL_NAME};

/// Registers Codex's compatibility `read_mcp_resource` function bridge.
pub fn insert_read_resource_entry<S: std::hash::BuildHasher>(
    entries: &mut HashMap<String, ToolEntry, S>,
    p: &McpToolParam,
    handler: Option<Arc<dyn GatewayExecutor>>,
) {
    let Some(handler) = handler else {
        tracing::debug!("read_mcp_resource skipped because no MCP handler is configured");
        return;
    };
    let config = serialize_to_value_or_custom_default(
        p,
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

/// Registers one tool returned by MCP `tools/list`, keyed by its internal
/// model-visible name while retaining the raw server and tool identity in its
/// serialized config.
pub fn insert_discovered_mcp_entry<S: std::hash::BuildHasher>(
    entries: &mut HashMap<String, ToolEntry, S>,
    discovered: McpDiscoveredHandler,
) {
    let McpDiscoveredHandler { param, handler } = discovered;
    let config = serialize_to_value_or_custom_default(
        &param,
        "MCP tool-call config serialization failed",
        |config| config,
        serde_json::Value::Null,
    );
    let McpDiscoveredToolParam {
        server_label,
        internal_name,
        ..
    } = param;
    if entries
        .insert(
            internal_name.clone(),
            ToolEntry {
                tool_type: ToolType::Mcp,
                config,
                server_label: Some(server_label),
                handler: Some(handler),
            },
        )
        .is_some()
    {
        tracing::warn!(name = %internal_name, "duplicate discovered MCP tool name — previous definition overwritten");
    }
}
