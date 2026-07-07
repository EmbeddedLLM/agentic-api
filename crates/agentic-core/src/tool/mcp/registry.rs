use std::sync::Arc;

use serde_json::Value;

use crate::tool::{ToolEntry, ToolHandler, ToolRegistry, ToolType};
use crate::types::io::FunctionTool;
use crate::types::tools::McpDiscoveredToolParam;
use crate::utils::common::serialize_to_value;

use super::{McpClientPool, McpHandler, READ_MCP_RESOURCE_TOOL_NAME};

pub async fn build_mcp_registry(pool: Arc<McpClientPool>) -> (Vec<FunctionTool>, ToolRegistry) {
    let mut specs = Vec::new();
    let mut entries = Vec::new();

    let read_handler = Arc::new(McpHandler::read_resource(Arc::clone(&pool)));
    specs.extend(read_handler.normalize(&Value::Null));
    entries.push((
        READ_MCP_RESOURCE_TOOL_NAME.to_owned(),
        ToolEntry {
            tool_type: ToolType::Mcp,
            config: Value::Null,
            server_label: None,
            handler: Some(read_handler),
        },
    ));

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

            specs.extend(handler.normalize(&config));
            entries.push((
                exposed_name,
                ToolEntry {
                    tool_type: ToolType::Mcp,
                    config,
                    server_label: Some(server_label.clone()),
                    handler: Some(handler),
                },
            ));
        }
    }

    (specs, ToolRegistry::with_entries(entries))
}
