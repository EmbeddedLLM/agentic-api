use std::collections::HashMap;
use std::sync::Arc;

use super::mcp::{McpClientPool, McpDiscoveredHandler, McpHandler};
use super::registry::ToolType;
use super::web_search::WebSearchHandler;
use super::{GatewayExecutor, ToolError};
use crate::types::tools::McpToolParam;

pub enum GatewayExecutorRegistration {
    Shared(Arc<dyn GatewayExecutor>),
    Mcp {
        server_label: String,
        handlers: Vec<McpDiscoveredHandler>,
    },
}

impl<T> From<Arc<T>> for GatewayExecutorRegistration
where
    T: GatewayExecutor,
{
    fn from(executor: Arc<T>) -> Self {
        Self::Shared(executor)
    }
}

impl From<Arc<dyn GatewayExecutor>> for GatewayExecutorRegistration {
    fn from(executor: Arc<dyn GatewayExecutor>) -> Self {
        Self::Shared(executor)
    }
}

/// Shared, per-server registry of gateway-owned tool executors.
///
/// Built once at startup ([`GatewayExecutors::from_env`]) and reused across
/// every request. MCP tools are the exception: their handler depends on the
/// per-request `McpToolParam`, so discovery builds them lazily unless a handler
/// has been pre-registered via [`GatewayExecutors::insert`].
#[derive(Clone, Default)]
pub struct GatewayExecutors {
    mcp: HashMap<String, Vec<McpDiscoveredHandler>>,
    web_search: Option<Arc<dyn GatewayExecutor>>,
}

impl GatewayExecutors {
    #[must_use]
    pub fn from_env(client: Arc<reqwest::Client>) -> Self {
        Self {
            mcp: HashMap::new(),
            web_search: Some(Arc::new(WebSearchHandler::from_env(client))),
        }
    }

    pub fn insert(&mut self, registration: impl Into<GatewayExecutorRegistration>) {
        match registration.into() {
            GatewayExecutorRegistration::Shared(executor) => match executor.tool_type() {
                ToolType::WebSearch => self.web_search = Some(executor),
                ToolType::Mcp => {
                    tracing::debug!("MCP executors must be registered with a server_label and discovered handlers");
                }
                other => tracing::debug!(tool_type = ?other, "gateway executor type has no executor slot"),
            },
            GatewayExecutorRegistration::Mcp { server_label, handlers } => {
                if handlers.is_empty() {
                    tracing::debug!(server_label, "empty MCP discovered handler registration skipped");
                    return;
                }
                if self.mcp.insert(server_label.clone(), handlers).is_some() {
                    tracing::debug!(server_label, "replaced MCP discovered handler registration");
                }
            }
        }
    }

    #[must_use]
    pub fn web_search_handler(&self) -> Option<Arc<dyn GatewayExecutor>> {
        self.web_search.clone()
    }

    #[must_use]
    pub(crate) fn request_scoped(&self) -> Self {
        self.clone()
    }

    /// Returns the discovered handlers for one request-declared MCP server.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for an invalid declaration or an empty
    /// allowed tool set, and an execution error when the server cannot connect.
    pub async fn mcp_handler(&mut self, param: &McpToolParam) -> Result<Vec<McpDiscoveredHandler>, ToolError> {
        let server_label = param.server_label.trim();
        if server_label.is_empty() {
            return Err(ToolError::Config(
                "MCP declaration requires a non-empty server_label".to_owned(),
            ));
        }
        if let Some(cached) = self.mcp.get(server_label) {
            return Ok(cached.clone());
        }

        let pool = McpClientPool::from_params(std::slice::from_ref(param)).await;
        let Some(client) = pool.get(server_label).cloned() else {
            return Err(pool.connection_error(server_label).map_or_else(
                || {
                    ToolError::Config(format!(
                        "MCP server '{server_label}' has no valid request-declared configuration"
                    ))
                },
                |error| ToolError::Execution(format!("MCP server '{server_label}' failed to connect: {error}")),
            ));
        };
        let discovered =
            McpHandler::discovered_tool_handlers(server_label, client, param.allowed_tools.as_deref()).await;
        if discovered.is_empty() {
            return Err(ToolError::Config(format!(
                "MCP server '{server_label}' has an empty final allowed tool set"
            )));
        }
        self.mcp.insert(server_label.to_owned(), discovered.clone());
        Ok(discovered)
    }
}

impl std::fmt::Debug for GatewayExecutors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayExecutors")
            .field("mcp_server_handlers", &self.mcp.len())
            .field("web_search", &self.web_search.is_some())
            .finish()
    }
}
