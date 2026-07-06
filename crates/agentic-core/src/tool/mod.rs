//! Tool framework — registry, handler trait, and normalization pipeline.
//!
//! Wire format types (`ResponsesTool`, param structs) live in [`crate::types::tools`].
//! This module owns the behavioral layer: routing, handler interface, and normalization,
//! including Codex CLI's own tool taxonomy ([`codex::CodexParams`]) via [`codex::CodexToolHandler`].

pub mod codex;
pub mod function;
pub mod handler;
pub mod normalize;
pub mod registry;

pub use codex::{CodexParams, CodexToolHandler};
pub use function::FunctionHandler;
pub use handler::{GatewayExecutor, ToolError, ToolHandler, ToolOutput};
pub use registry::{ToolEntry, ToolRegistry, ToolType};
