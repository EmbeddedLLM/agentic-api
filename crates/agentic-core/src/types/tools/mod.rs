//! Wire format types for the tool framework.
//!
//! This module contains only serde shapes (serialization/deserialization types).
//! Behavioral logic (registry, handler trait, normalization) lives in [`crate::tool`].

pub mod params;

pub use params::{
    CodeInterpreterToolParam, CodexCustomToolParam, CodexNamespaceMember, CodexNamespaceToolParam,
    CodexToolSearchToolParam, EmptyToolNameError, FileSearchToolParam, FunctionToolParam, McpToolParam,
    NonEmptyToolName, ResponsesTool, ToolSearchExecution, WebSearchToolParam,
};
