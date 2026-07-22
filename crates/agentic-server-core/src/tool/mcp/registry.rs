use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::tool::{ToolEntry, ToolError, ToolType};
use crate::types::tools::McpDiscoveredToolParam;
use crate::utils::common::serialize_to_value_or_custom_default;

use super::McpDiscoveredHandler;

/// Registers one tool returned by MCP `tools/list`, keyed by its internal
/// model-visible name while retaining the raw server and tool identity in its
/// serialized config.
///
/// # Errors
///
/// Returns [`ToolError::Config`] when the derived internal name collides with
/// an entry that was already registered for the request.
pub(crate) fn insert_discovered_mcp_entry(
    entries: &mut HashMap<String, ToolEntry>,
    discovered: McpDiscoveredHandler,
) -> Result<(), ToolError> {
    let McpDiscoveredHandler { param, handler } = discovered;
    let config = serialize_to_value_or_custom_default(
        &param,
        "MCP tool-call config serialization failed",
        |config| config,
        serde_json::Value::Null,
    );
    let McpDiscoveredToolParam {
        server_label,
        tool_name,
        internal_name,
        ..
    } = param;
    match entries.entry(internal_name) {
        Entry::Occupied(existing) => Err(ToolError::Config(format!(
            "discovered MCP tool '{server_label}/{tool_name}' conflicts with existing registry entry for internal name '{}'",
            existing.key()
        ))),
        Entry::Vacant(entry) => {
            entry.insert(ToolEntry {
                tool_type: ToolType::Mcp,
                config,
                server_label: Some(server_label),
                handler: Some(handler),
            });
            Ok(())
        }
    }
}
