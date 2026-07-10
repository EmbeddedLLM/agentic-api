use std::collections::HashMap;
use std::sync::Arc;

use crate::tool::{GatewayExecutor, ToolEntry, ToolType};
use crate::types::tools::{McpDiscoveredToolParam, McpToolParam};
use crate::utils::common::{serialize_to_value, serialize_to_value_or_custom_default};

use super::{McpClientPool, McpHandler, READ_MCP_RESOURCE_TOOL_NAME};

/// Registers `p` for gateway dispatch: reuses an externally-supplied MCP
/// handler when `handler_for` provides one, otherwise connects to `p`'s
/// declared server and discovers its tools via [`build_mcp_registry`].
pub async fn insert_mcp_entry<S: std::hash::BuildHasher>(
    entries: &mut HashMap<String, ToolEntry, S>,
    p: &McpToolParam,
    handler_for: &mut impl FnMut(ToolType) -> Option<Arc<dyn GatewayExecutor>>,
) {
    let Some(handler) = handler_for(ToolType::Mcp) else {
        build_mcp_registry(p, entries).await;
        return;
    };

    serialize_to_value_or_custom_default(
        p,
        "MCP tool config serialization failed",
        |config| {
            if entries
                .insert(
                    p.name.as_str().to_owned(),
                    ToolEntry {
                        tool_type: ToolType::Mcp,
                        config,
                        server_label: None,
                        handler: Some(handler),
                    },
                )
                .is_some()
            {
                tracing::warn!(name = %p.name, "duplicate MCP tool name — previous definition overwritten");
            }
        },
        (),
    );
}

/// Connects to the MCP server declared by `param`, then registers the
/// `read_mcp_resource` built-in and every tool the server exposes into
/// `entries`, keyed by the name the model will call.
pub async fn build_mcp_registry<S: std::hash::BuildHasher>(
    param: &McpToolParam,
    entries: &mut HashMap<String, ToolEntry, S>,
) {
    let pool = Arc::new(McpClientPool::from_params(std::slice::from_ref(param)).await);

    let read_handler = Arc::new(McpHandler::read_resource(Arc::clone(&pool)));
    entries.insert(
        READ_MCP_RESOURCE_TOOL_NAME.to_owned(),
        ToolEntry {
            tool_type: ToolType::Mcp,
            config: serde_json::Value::Null,
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
