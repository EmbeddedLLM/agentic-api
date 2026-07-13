//! Bridge support for Codex's local `read_mcp_resource` function tool.
//!
//! Codex should be configured with the MCP server label and URL in
//! `$CODEX_HOME/config.toml`, for example:
//!
//! ```toml
//! [mcp_servers.server_label]
//! url = "http://localhost:8000/mcp"
//! default_tools_approval_mode = "approve"
//! ```
//!
//! When Codex is pointed at agentic-api as its Responses provider, the gateway
//! uses metadata on the `read_mcp_resource` function declaration to map the
//! model-provided `server` argument (for example `server_label`) to that MCP URL.

use std::collections::HashMap;

use serde::Deserialize;

use crate::types::tools::{FunctionToolParam, McpToolParam, NonEmptyToolName};
use crate::utils::common::deserialize_from_value_or_custom_default;

use super::READ_MCP_RESOURCE_TOOL_NAME;

#[derive(Debug, Deserialize)]
struct McpFunctionMetadata {
    #[serde(default)]
    mcp_servers: HashMap<String, McpFunctionServer>,
    #[serde(default)]
    server_label: Option<String>,
    #[serde(default, alias = "url")]
    server_url: Option<String>,
    #[serde(default)]
    headers: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
struct McpFunctionServer {
    #[serde(alias = "url")]
    server_url: String,
    #[serde(default)]
    headers: Option<HashMap<String, String>>,
}

#[must_use]
pub fn maybe_mcp_function(p: &FunctionToolParam) -> Option<Vec<McpToolParam>> {
    (p.name.as_str() == READ_MCP_RESOURCE_TOOL_NAME).then(|| mcp_params_from_function(p))
}

fn mcp_params_from_function(p: &FunctionToolParam) -> Vec<McpToolParam> {
    let Some(metadata) = p.extra.get("metadata") else {
        return Vec::new();
    };

    deserialize_from_value_or_custom_default(
        metadata.clone(),
        "read_mcp_resource metadata is not MCP metadata",
        mcp_params_from_metadata,
        Vec::new(),
    )
}

fn mcp_params_from_metadata(metadata: McpFunctionMetadata) -> Vec<McpToolParam> {
    let Ok(name) = NonEmptyToolName::try_from(READ_MCP_RESOURCE_TOOL_NAME) else {
        return Vec::new();
    };

    if metadata.mcp_servers.is_empty() {
        return match (metadata.server_label, metadata.server_url) {
            (Some(server_label), Some(server_url)) => vec![McpToolParam {
                name,
                server_label: Some(server_label),
                server_url: Some(server_url),
                headers: metadata.headers,
            }],
            _ => Vec::new(),
        };
    }

    let server_count = metadata.mcp_servers.len();
    if server_count == 1 {
        let Some((server_label, server)) = metadata.mcp_servers.into_iter().next() else {
            return Vec::new();
        };
        return vec![McpToolParam {
            name,
            server_label: Some(server_label),
            server_url: Some(server.server_url),
            headers: server.headers,
        }];
    }

    let mut params = Vec::with_capacity(server_count);
    for (server_label, server) in metadata.mcp_servers {
        params.push(McpToolParam {
            name: name.clone(),
            server_label: Some(server_label),
            server_url: Some(server.server_url),
            headers: server.headers,
        });
    }

    params
}
