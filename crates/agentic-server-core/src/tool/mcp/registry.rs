use std::collections::HashMap;

use crate::tool::{ToolEntry, ToolType};
use crate::types::tools::McpDiscoveredToolParam;
use crate::utils::common::serialize_to_value_or_custom_default;

use super::McpDiscoveredHandler;

/// Registers one tool returned by MCP `tools/list`, keyed by its internal
/// model-visible name while retaining the raw server and tool identity in its
/// serialized config.
///
pub(crate) fn insert_discovered_mcp_entry(entries: &mut HashMap<String, ToolEntry>, discovered: McpDiscoveredHandler) {
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
    entries.insert(
        internal_name,
        ToolEntry {
            tool_type: ToolType::Mcp,
            config,
            server_label: Some(server_label),
            handler: Some(handler),
        },
    );
}
