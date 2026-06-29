pub mod codex;
pub mod dispatch;

pub use codex::IncomingTool;
pub use dispatch::execute_server_tool_calls;

use std::future::Future;

use crate::types::io::{FunctionTool, FunctionToolResultMessage};

/// Convert a tool definition into the `FunctionTool` shape vLLM understands.
///
/// Returns `None` for variants that should be dropped before forwarding upstream
/// (e.g. `code_interpreter`, `web_search`, unknown types that agentic-api does
/// not yet handle server-side).
pub trait NormalizeTool {
    fn normalize(&self) -> Option<FunctionTool>;
}

/// Execute a tool call received from the model and return the result.
///
/// Returns `Some(result)` when agentic-api executes the tool server-side.
/// Returns `None` when the tool is Codex CLI's responsibility — the caller
/// passes the `FunctionCall` through unchanged and Codex drives the next turn.
pub trait ExecuteTool {
    fn execute(&self, call_id: &str, arguments: &str)
    -> impl Future<Output = Option<FunctionToolResultMessage>> + Send;
}

/// Normalize a slice of tools into the `Vec<FunctionTool>` forwarded to vLLM.
///
/// Replaces the old free function `normalize_for_vllm`. Drops any tool whose
/// `normalize()` returns `None`.
pub fn normalize_tools<T: NormalizeTool>(tools: &[T]) -> Vec<FunctionTool> {
    tools.iter().filter_map(|t| t.normalize()).collect()
}
