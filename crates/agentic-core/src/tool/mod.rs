//! Tool framework — registry, handler trait, and normalization pipeline.
//!
//! Wire format types (`ResponsesTool`, param structs) live in [`crate::types::tools`].
//! This module owns the behavioral layer: routing, handler interface, and normalization.

pub mod function;
pub mod handler;
pub mod normalize;
pub mod registry;
pub mod web_search;

pub use function::FunctionHandler;
pub use handler::{GatewayExecutor, ToolError, ToolHandler, ToolOutput};
pub use normalize::{
    alternate_model_visible_namespace_member_name, flatten_tool_choice_for_upstream, flatten_tools_for_upstream,
    legacy_model_visible_namespace_member_name, model_visible_namespace_member_name, restore_output_items_with_tools,
    restore_response_value_with_tools,
};
pub use registry::{GatewayDispatchResult, ToolEntry, ToolRegistry, ToolType};
pub use web_search::WebSearchHandler;
