use std::collections::HashMap;
use std::sync::Arc;

use super::mcp::handler::McpServerToolSet;
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
        Ok(self.mcp_server_tools(param).await?.discovered_handlers)
    }

    /// Returns the request-scoped tools and public discovery item for one MCP server.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for an invalid declaration or an empty
    /// allowed tool set, and an execution error when the server cannot connect.
    pub(crate) async fn mcp_server_tools(&mut self, param: &McpToolParam) -> Result<McpServerToolSet, ToolError> {
        validate_mcp_execution_options(param)?;

        let server_label = param.server_label.trim();
        if server_label.is_empty() {
            return Err(ToolError::Config(
                "MCP declaration requires a non-empty server_label".to_owned(),
            ));
        }
        if let Some(cached) = self.mcp.get(server_label) {
            let discovered_handlers = require_non_empty_mcp_handlers(
                server_label,
                filter_allowed_mcp_handlers(cached, param.allowed_tools.as_deref()),
            )?;
            return Ok(McpHandler::server_tool_set_from_handlers(
                server_label,
                discovered_handlers,
            ));
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
        let McpServerToolSet {
            discovered_handlers,
            list_tools_item,
        } = McpHandler::discover_tools(server_label, client, param.allowed_tools.as_deref()).await?;
        let discovered_handlers = require_non_empty_mcp_handlers(server_label, discovered_handlers)?;
        self.mcp.insert(server_label.to_owned(), discovered_handlers.clone());
        Ok(McpServerToolSet {
            discovered_handlers,
            list_tools_item,
        })
    }
}

fn filter_allowed_mcp_handlers(
    handlers: &[McpDiscoveredHandler],
    allowed_tools: Option<&[String]>,
) -> Vec<McpDiscoveredHandler> {
    handlers
        .iter()
        .filter(|handler| {
            allowed_tools.is_none_or(|allowed| allowed.iter().any(|name| name == &handler.param.tool_name))
        })
        .cloned()
        .collect()
}

fn require_non_empty_mcp_handlers(
    server_label: &str,
    handlers: Vec<McpDiscoveredHandler>,
) -> Result<Vec<McpDiscoveredHandler>, ToolError> {
    if handlers.is_empty() {
        return Err(ToolError::Config(format!(
            "MCP server '{server_label}' has an empty final allowed tool set"
        )));
    }
    Ok(handlers)
}

fn validate_mcp_execution_options(param: &McpToolParam) -> Result<(), ToolError> {
    if param.connector_id.is_some() {
        return Err(ToolError::Config(
            "MCP connector_id is not supported; configure server_url instead".to_owned(),
        ));
    }
    if param.require_approval.as_deref() != Some("never") {
        return Err(ToolError::Config(
            "MCP require_approval must be explicitly set to 'never'; approval gating is not yet supported".to_owned(),
        ));
    }
    Ok(())
}

impl std::fmt::Debug for GatewayExecutors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayExecutors")
            .field("mcp_server_handlers", &self.mcp.len())
            .field("web_search", &self.web_search.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{GatewayExecutorRegistration, GatewayExecutors, validate_mcp_execution_options};
    use crate::tool::mcp::{McpDiscoveredHandler, McpHandler};
    use crate::types::tools::{McpDiscoveredToolParam, McpToolParam};

    fn mcp_param(value: serde_json::Value) -> McpToolParam {
        serde_json::from_value(value).unwrap()
    }

    fn discovered_handler(tool_name: &str) -> McpDiscoveredHandler {
        McpDiscoveredHandler {
            param: McpDiscoveredToolParam {
                server_label: "counter".to_owned(),
                tool_name: tool_name.to_owned(),
                internal_name: format!("mcp__counter__{tool_name}"),
                tool: serde_json::from_value(serde_json::json!({
                    "name": tool_name,
                    "inputSchema": {"type": "object"}
                }))
                .unwrap(),
            },
            handler: Arc::new(McpHandler::discovered_tool_spec_only()),
        }
    }

    #[test]
    fn mcp_execution_allows_explicit_never_approval_policy() {
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "server_url": "http://localhost:8000/mcp",
            "require_approval": "never"
        }));

        validate_mcp_execution_options(&param).unwrap();
    }

    #[test]
    fn mcp_execution_rejects_omitted_approval_policy() {
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "server_url": "http://localhost:8000/mcp"
        }));

        let error = validate_mcp_execution_options(&param).unwrap_err();
        assert!(error.to_string().contains("must be explicitly set to 'never'"));
    }

    #[test]
    fn mcp_execution_rejects_unsupported_approval_policy() {
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "server_url": "http://localhost:8000/mcp",
            "require_approval": "always"
        }));

        let error = validate_mcp_execution_options(&param).unwrap_err();
        assert!(error.to_string().contains("approval gating is not yet supported"));
    }

    #[test]
    fn mcp_execution_rejects_connector_id() {
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "connector_id": "connector_dropbox"
        }));

        let error = validate_mcp_execution_options(&param).unwrap_err();
        assert!(error.to_string().contains("connector_id is not supported"));
    }

    #[tokio::test]
    async fn cached_mcp_server_tools_apply_request_allowed_tools_with_fresh_output_id() {
        let mut executors = GatewayExecutors::default();
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "counter".to_owned(),
            handlers: vec![discovered_handler("read"), discovered_handler("delete")],
        });
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "allowed_tools": ["read"],
            "require_approval": "never"
        }));

        let first = executors.mcp_server_tools(&param).await.unwrap();
        let first_output_id = first.list_tools_item.id.clone();
        let second = executors.mcp_server_tools(&param).await.unwrap();

        assert_eq!(first.discovered_handlers.len(), 1);
        assert_eq!(first.discovered_handlers[0].param.tool_name, "read");
        assert_eq!(first.list_tools_item.tools.len(), 1);
        assert_eq!(first.list_tools_item.tools[0].name, "read");
        assert_ne!(first_output_id, second.list_tools_item.id);
    }

    #[tokio::test]
    async fn cached_mcp_handlers_reject_empty_final_allowed_set() {
        let mut executors = GatewayExecutors::default();
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "counter".to_owned(),
            handlers: vec![discovered_handler("delete")],
        });
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "allowed_tools": ["read"],
            "require_approval": "never"
        }));

        let Err(error) = executors.mcp_server_tools(&param).await else {
            panic!("expected empty allowed set to be rejected");
        };

        assert!(error.to_string().contains("empty final allowed tool set"));
    }
}
