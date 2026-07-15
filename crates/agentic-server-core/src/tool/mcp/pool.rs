use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;

use reqwest::Url;
use serde::{Deserialize, Serialize};

use super::client::McpClient;
use crate::types::tools::McpToolParam;

// Hostnames configured here are a trust boundary: validation compares the
// hostname string but does not pin its DNS resolution to the transport. Only
// add names whose DNS records are controlled by a trusted administrator.
const MCP_ALLOWED_HOSTS_ENV: &str = "AGENTIC_MCP_ALLOWED_HOSTS";

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

    pub fn client_for_param(&self, param: &McpToolParam) -> Option<Arc<McpClient>> {
        let Some(server_label) = clean_string(param.server_label.as_deref()) else {
            tracing::debug!(name = %param.name, "MCP tool param has no server_label");
            return None;
        };

        let Some(client) = self.get(&server_label).cloned() else {
            if let Some(error) = self.connection_error(&server_label) {
                tracing::warn!(
                    server_label,
                    name = %param.name,
                    error,
                    "MCP server failed to connect"
                );
            } else {
                tracing::warn!(
                    server_label,
                    name = %param.name,
                    "MCP server is not connected"
                );
            }
            return None;
        };

        Some(client)
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
    let Some(server_label) = clean_string(param.server_label.as_deref()) else {
        tracing::debug!(name = %param.name, "MCP tool param has no server_label");
        return None;
    };

    if let Some(url) = clean_string(param.server_url.as_deref()) {
        let url = match validate_request_server_url(&url) {
            Ok(url) => url,
            Err(reason) => {
                tracing::warn!(
                    server_label,
                    name = %param.name,
                    url,
                    reason,
                    "MCP tool param server_url rejected"
                );
                return None;
            }
        };

        return Some((
            server_label,
            McpServerEntry::Http {
                url,
                headers: param.headers.clone(),
            },
        ));
    }

    tracing::warn!(
        server_label,
        name = %param.name,
        "MCP tool param has no server_url"
    );
    None
}

fn validate_request_server_url(value: &str) -> Result<String, String> {
    let url = Url::parse(value).map_err(|error| format!("invalid URL: {error}"))?;
    match url.scheme() {
        "http" | "https" => {}
        _ => return Err("URL scheme must be http or https".to_owned()),
    }

    if !url.username().is_empty() || url.password().is_some() {
        return Err("URL must not include credentials".to_owned());
    }

    let host = url.host_str().ok_or_else(|| "URL must include a host".to_owned())?;
    if is_allowed_request_host(host) {
        return Ok(value.to_owned());
    }

    Err(format!(
        "MCP server_url host is not allowed; set {MCP_ALLOWED_HOSTS_ENV} to allow it"
    ))
}

fn is_allowed_request_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
        || host_allowed_by_env(host)
}

fn host_allowed_by_env(host: &str) -> bool {
    std::env::var(MCP_ALLOWED_HOSTS_ENV).is_ok_and(|allowed_hosts| {
        allowed_hosts
            .split(',')
            .map(str::trim)
            .any(|allowed_host| allowed_host.eq_ignore_ascii_case(host))
    })
}

fn clean_string(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::{McpServerEntry, server_entry_from_param, validate_request_server_url};
    use crate::types::tools::McpToolParam;

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

    #[test]
    fn request_server_url_allows_loopback_http() {
        let url = validate_request_server_url("http://127.0.0.1:8000/mcp").unwrap();
        assert_eq!(url, "http://127.0.0.1:8000/mcp");
    }

    #[test]
    fn request_server_url_rejects_unallowlisted_host() {
        let error = validate_request_server_url("http://169.254.169.254/mcp").unwrap_err();
        assert!(error.contains("not allowed"));
    }

    #[test]
    fn request_params_do_not_accept_stdio_command() {
        let param = serde_json::from_value::<McpToolParam>(serde_json::json!({
            "name": "read_mcp_resource",
            "server_label": "repo",
            "command": "python3",
            "args": ["/tmp/server.py"]
        }))
        .unwrap();

        assert!(server_entry_from_param(&param).is_none());
    }
}
