use crate::types::io::FunctionTool;
use crate::types::io::input::FunctionToolResultMessage;
use crate::types::tools::ResponsesTool;

use super::handler::ToolOutput;
use super::web_search::web_search_function_tool;

impl ResponsesTool {
    /// Normalise this tool declaration to the `FunctionTool` wire format that vLLM understands.
    ///
    /// `Function` variants convert directly, and `WebSearch` maps to the gateway-owned
    /// function shim. Codex namespace tools are represented by [`crate::types::RequestTool`]
    /// and flattened before they become `ResponsesTool` values.
    ///
    /// Handler-less provider tools (`Mcp`, `FileSearch`, `CodeInterpreter`) plus
    /// `Unknown` return `None` and log at `debug` level. This keeps the upstream request
    /// body as `Vec<FunctionTool>` after [`crate::tool::ToolRegistry`] has applied any
    /// request-scoped tool handlers.
    #[must_use]
    pub fn to_function_tool(&self) -> Option<FunctionTool> {
        match self {
            // name is NonEmptyToolName — empty names are rejected by serde at
            // deserialization time, so no runtime check is needed here.
            Self::Function(p) => Some(FunctionTool::from(p)),
            Self::Mcp(p) => {
                tracing::debug!(
                    server_label = %p.server_label,
                    "MCP tool skipped in normalize - handler not yet registered"
                );
                None
            }
            Self::WebSearch(_) => Some(web_search_function_tool()),
            Self::FileSearch(_) => {
                tracing::debug!("file_search tool skipped in normalize - handler not yet registered");
                None
            }
            Self::CodeInterpreter(_) => {
                tracing::debug!("code_interpreter tool skipped in normalize - handler not yet registered");
                None
            }
            Self::Unknown => {
                tracing::debug!("unknown tool skipped in normalize");
                None
            }
        }
    }
}

impl From<ToolOutput> for FunctionToolResultMessage {
    fn from(o: ToolOutput) -> Self {
        Self {
            call_id: o.call_id,
            output: o.output,
        }
    }
}
