use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::READ_MCP_RESOURCE_TOOL_NAME;
use super::client::McpClient;
use crate::types::tools::McpToolParam;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpServerEntry {
    Http {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        headers: Option<HashMap<String, String>>,
    },
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        env: Option<HashMap<String, String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
    },
}

#[derive(Default)]
pub struct McpClientPool {
    clients: HashMap<String, Arc<McpClient>>,
    connection_errors: HashMap<String, String>,
}

impl McpClientPool {
    pub async fn from_params(params: &[McpToolParam]) -> Self {
        let servers: HashMap<String, McpServerEntry> = params.iter().filter_map(server_entry_from_param).collect();
        Self::from_config(servers).await
    }

    pub async fn from_config(servers: HashMap<String, McpServerEntry>) -> Self {
        let mut clients = HashMap::with_capacity(servers.len());
        let mut connection_errors = HashMap::new();

        for (server_label, entry) in servers {
            let result = match entry {
                McpServerEntry::Http { url, headers } => McpClient::connect(&url, headers).await,
                McpServerEntry::Stdio {
                    command,
                    args,
                    env,
                    cwd,
                } => McpClient::connect_stdio(&command, &args, env.as_ref(), cwd.as_deref()).await,
            };

            match result {
                Ok(client) => {
                    clients.insert(server_label, Arc::new(client));
                }
                Err(error) => {
                    let error_message = error.to_string();
                    tracing::warn!(
                        server_label = %server_label,
                        error = %error_message,
                        "failed to connect MCP server from config"
                    );
                    connection_errors.insert(server_label, error_message);
                }
            }
        }

        Self {
            clients,
            connection_errors,
        }
    }

    #[must_use]
    pub fn get(&self, server_label: &str) -> Option<&Arc<McpClient>> {
        self.clients.get(server_label)
    }

    #[must_use]
    pub fn connection_error(&self, server_label: &str) -> Option<&str> {
        self.connection_errors.get(server_label).map(String::as_str)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Arc<McpClient>)> {
        self.clients.iter()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }
}

fn server_entry_from_param(param: &McpToolParam) -> Option<(String, McpServerEntry)> {
    if param.name.as_str() != READ_MCP_RESOURCE_TOOL_NAME {
        return None;
    }

    let Some(server_label) = clean_string(param.server_label.as_deref()) else {
        tracing::debug!(name = %param.name, "MCP tool param has no server_label");
        return None;
    };

    if let Some(url) = clean_string(param.server_url.as_deref()) {
        return Some((
            server_label,
            McpServerEntry::Http {
                url,
                headers: param.headers.clone(),
            },
        ));
    }

    if let Some(command) = clean_string(param.command.as_deref()) {
        return Some((
            server_label,
            McpServerEntry::Stdio {
                command,
                args: param.args.clone(),
                env: param.env.clone(),
                cwd: param.cwd.clone(),
            },
        ));
    }

    tracing::warn!(
        server_label,
        name = %param.name,
        "MCP tool param has no server_url or command"
    );
    None
}

fn clean_string(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::McpServerEntry;

    #[test]
    fn mcp_server_entry_deserializes_http_config() {
        let entry = serde_json::from_value::<McpServerEntry>(serde_json::json!({
            "url": "http://localhost:9000",
            "headers": {"Authorization": "Bearer token"}
        }))
        .unwrap();

        match entry {
            McpServerEntry::Http { url, headers } => {
                assert_eq!(url, "http://localhost:9000");
                assert_eq!(headers.unwrap()["Authorization"], "Bearer token");
            }
            McpServerEntry::Stdio { .. } => panic!("expected HTTP MCP config"),
        }
    }

    #[test]
    fn mcp_server_entry_deserializes_stdio_config() {
        let entry = serde_json::from_value::<McpServerEntry>(serde_json::json!({
            "command": "python3",
            "args": ["/tmp/server.py"],
            "env": {"TOKEN": "secret"},
            "cwd": "/tmp"
        }))
        .unwrap();

        match entry {
            McpServerEntry::Stdio {
                command,
                args,
                env,
                cwd,
            } => {
                assert_eq!(command, "python3");
                assert_eq!(args, vec!["/tmp/server.py".to_owned()]);
                assert_eq!(env.unwrap()["TOKEN"], "secret");
                assert_eq!(cwd.as_deref(), Some("/tmp"));
            }
            McpServerEntry::Http { .. } => panic!("expected stdio MCP config"),
        }
    }
}
