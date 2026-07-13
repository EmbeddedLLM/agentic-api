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
pub mod web_search;

pub use codex::{CodexNamespaceHandler, NamespaceMap, model_visible_namespace_member_name};
pub use executors::GatewayExecutors;
pub use function::FunctionHandler;
pub use handler::{GatewayExecutor, ToolError, ToolHandler, ToolOutput};
pub use mcp::{
    McpClient, McpClientPool, McpDiscoveredHandler, McpError, McpHandler, McpHandlerFactory, McpHandlerKind,
    McpOperation, McpServerEntry, McpSpec, READ_MCP_RESOURCE_TOOL_NAME, ReadResourceArgs, build_mcp_registry,
    read_mcp_resource_spec,
};
pub use registry::{GatewayDispatchResult, ToolEntry, ToolRegistry, ToolType};
pub use web_search::WebSearchHandler;
