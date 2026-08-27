//! Whether a tool is executed by the gateway itself or handed back to the client.

use std::sync::Arc;

use super::handler::GatewayExecutor;

/// A resolved gateway handler plus its per-tool-name concurrency policy.
pub struct GatewayBinding {
    pub handler: Arc<dyn GatewayExecutor>,
    /// `Some` when this handler must not run concurrently with a second call
    /// to the SAME tool name (built from `!handler.supports_parallel_execution()`
    /// at registration time); `None` when it's safe to call itself concurrently.
    /// Never gates against other tool names.
    pub self_exclusion: Option<Arc<tokio::sync::Semaphore>>,
}

impl Clone for GatewayBinding {
    fn clone(&self) -> Self {
        Self {
            handler: Arc::clone(&self.handler),
            self_exclusion: self.self_exclusion.clone(),
        }
    }
}

impl GatewayBinding {
    #[must_use]
    pub fn new(handler: Arc<dyn GatewayExecutor>) -> Self {
        let self_exclusion = (!handler.supports_parallel_execution()).then(|| Arc::new(tokio::sync::Semaphore::new(1)));
        Self {
            handler,
            self_exclusion,
        }
    }
}

pub enum ToolOwnership {
    Client,
    /// `None` means this tool type is gateway-owned in principle but has no
    /// handler implemented yet (e.g. `FileSearch`/`CodeInterpreter` today).
    Gateway(Option<GatewayBinding>),
}

impl Clone for ToolOwnership {
    fn clone(&self) -> Self {
        match self {
            Self::Client => Self::Client,
            Self::Gateway(binding) => Self::Gateway(binding.clone()),
        }
    }
}

impl ToolOwnership {
    #[must_use]
    pub fn is_gateway(&self) -> bool {
        matches!(self, Self::Gateway(_))
    }
}
