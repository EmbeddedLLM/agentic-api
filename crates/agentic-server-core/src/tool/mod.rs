//! Tool framework — registry, handler trait, and normalization pipeline.
//!
//! Wire format types (`ResponsesTool`, param structs) live in [`crate::types::tools`].
//! This module owns the behavioral layer: routing, handler interface, and normalization.

pub mod codex;
pub mod executors;
pub mod function;
pub mod handler;
pub mod mcp;
pub mod normalize;
pub mod registry;
mod tool_search;
pub mod web_search;

pub use codex::{CodexNamespaceHandler, NamespaceMap, model_visible_namespace_member_name};
pub use executors::{GatewayExecutorRegistration, GatewayExecutors};
pub use function::FunctionHandler;
pub use handler::{GatewayExecutor, ToolError, ToolHandler, ToolOutput};
pub use mcp::{McpClient, McpClientPool, McpDiscoveredHandler, McpError, McpHandler, McpOperation, McpServerEntry};
pub use registry::{GatewayDispatchResult, ToolEntry, ToolRegistry, ToolType};
pub(crate) use tool_search::{
    TOOL_SEARCH_NAME, loaded_function_identities, loaded_function_names, loaded_function_tools,
};
pub use web_search::WebSearchHandler;
