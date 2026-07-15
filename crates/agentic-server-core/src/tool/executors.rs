use std::sync::Arc;

use super::GatewayExecutor;
use super::mcp::{McpClientPool, McpHandler, McpHandlerFactory};
use super::registry::ToolType;
use super::web_search::WebSearchHandler;
use crate::types::tools::McpToolParam;

/// Shared, per-server registry of gateway-owned tool executors.
///
/// Built once at startup ([`GatewayExecutors::from_env`]) and reused across
/// every request. MCP tools are the exception: their handler depends on the
/// per-request `McpToolParam`, so [`GatewayExecutors::mcp_handler`] builds one
/// lazily unless a handler has been pre-registered via [`GatewayExecutors::insert`].
#[derive(Clone, Default)]
pub struct GatewayExecutors {
    mcp: Option<Arc<dyn GatewayExecutor>>,
    web_search: Option<Arc<dyn GatewayExecutor>>,
}

impl GatewayExecutors {
    #[must_use]
    pub fn from_env(client: Arc<reqwest::Client>) -> Self {
        Self {
            // MCP handlers need request payload information from the MCP tool
            // params, so the default executor is created in `mcp_handler`.
            mcp: None,
            web_search: Some(Arc::new(WebSearchHandler::from_env(client))),
        }
    }

    pub fn insert(&mut self, executor: Arc<dyn GatewayExecutor>) {
        match executor.tool_type() {
            ToolType::Mcp => self.mcp = Some(executor),
            ToolType::WebSearch => self.web_search = Some(executor),
            other => tracing::debug!(tool_type = ?other, "gateway executor type has no executor slot"),
        }
    }

    #[must_use]
    pub fn web_search_handler(&self) -> Option<Arc<dyn GatewayExecutor>> {
        self.web_search.clone()
    }

    pub async fn mcp_handler(&self, param: &McpToolParam) -> Option<Arc<dyn GatewayExecutor>> {
        if let Some(handler) = self.mcp.clone() {
            return Some(handler);
        }

        McpHandlerFactory::new()
            .from_params(param)
            .await
            .map(|handler| Arc::new(handler) as Arc<dyn GatewayExecutor>)
    }

    pub async fn mcp_read_resource_handler(&self, params: &[McpToolParam]) -> Option<Arc<dyn GatewayExecutor>> {
        if let Some(handler) = self.mcp.clone() {
            return Some(handler);
        }

        let pool = Arc::new(McpClientPool::from_params(params).await);
        Some(Arc::new(McpHandler::read_resource(pool)))
    }
}

impl std::fmt::Debug for GatewayExecutors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayExecutors")
            .field("mcp", &self.mcp.is_some())
            .field("web_search", &self.web_search.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::GatewayExecutors;
    use crate::tool::mcp::READ_MCP_RESOURCE_TOOL_NAME;
    use crate::types::tools::McpToolParam;

    #[tokio::test]
    async fn from_env_builds_request_scoped_mcp_handler_from_params() {
        let executors = GatewayExecutors::from_env(Arc::new(reqwest::Client::new()));
        let param: McpToolParam = serde_json::from_value(serde_json::json!({
            "name": READ_MCP_RESOURCE_TOOL_NAME,
            "server_label": "missing"
        }))
        .expect("mcp tool param");

        assert!(executors.mcp_handler(&param).await.is_some());
        assert!(executors.web_search_handler().is_some());
    }
}
