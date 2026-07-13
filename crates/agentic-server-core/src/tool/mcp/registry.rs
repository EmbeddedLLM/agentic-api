use std::collections::HashMap;
use std::sync::Arc;

use crate::tool::{ToolEntry, ToolType};
use crate::types::tools::{McpDiscoveredToolParam, McpToolParam};
use crate::utils::common::{serialize_to_value, serialize_to_value_or_custom_default};

use super::{McpHandler, READ_MCP_RESOURCE_TOOL_NAME};

/// Registers `p` for gateway dispatch by connecting to the request-declared
/// MCP server and discovering its tools via [`build_mcp_registry`].
pub async fn insert_mcp_entry<S: std::hash::BuildHasher>(
    entries: &mut HashMap<String, ToolEntry, S>,
    p: &McpToolParam,
) {
    build_mcp_registry(p, entries).await;
}

/// Connects to the MCP server declared by `param`, then registers the
/// `read_mcp_resource` built-in and every tool the server exposes into
/// `entries`, keyed by the name the model will call.
pub async fn build_mcp_registry<S: std::hash::BuildHasher>(
    param: &McpToolParam,
    entries: &mut HashMap<String, ToolEntry, S>,
) {
    let read_handler = Arc::new(McpHandler::from_params(std::slice::from_ref(param)).await);
    let Some(pool) = read_handler.pool() else {
        tracing::warn!("MCP read_resource handler did not expose a client pool");
        return;
    };
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
            handler: Some(read_handler),
        },
    );

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
            let param = McpDiscoveredToolParam {
                server_label: server_label.clone(),
                tool_name,
                exposed_name: exposed_name.clone(),
                tool,
            };
            let config = match serialize_to_value(&param) {
                Ok(config) => config,
                Err(error) => {
                    tracing::warn!(
                        server_label = %server_label,
                        exposed_name = %exposed_name,
                        error = %error,
                        "failed to serialize discovered MCP tool config"
                    );
                    continue;
                }
            };
            let handler = Arc::new(McpHandler::tool_call(Arc::clone(client)));

            if entries
                .insert(
                    exposed_name.clone(),
                    ToolEntry {
                        tool_type: ToolType::Mcp,
                        config,
                        server_label: Some(server_label.clone()),
                        handler: Some(handler),
                    },
                )
                .is_some()
            {
                tracing::warn!(name = %exposed_name, "duplicate MCP tool name — previous definition overwritten");
            }
        }
    }
}
