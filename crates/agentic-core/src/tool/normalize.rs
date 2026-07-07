use crate::types::io::FunctionTool;
use crate::types::io::input::FunctionToolResultMessage;
use crate::types::tools::ResponsesTool;
use crate::utils::common::serialize_to_value;

use super::function::FunctionHandler;
use super::handler::{ToolHandler, ToolOutput};
use super::mcp::{McpHandler, READ_MCP_RESOURCE_TOOL_NAME};
use super::web_search::web_search_function_tool;

impl ResponsesTool {
    /// Normalise this tool declaration to the `FunctionTool` wire format that vLLM understands.
    ///
    /// - `Function` variants convert via [`From<&FunctionToolParam>`] for `FunctionTool`.
    ///   Returns `None` and logs at `debug` level if the name is empty.
    /// - `Mcp` variants convert gateway MCP built-ins to the function specs
    ///   vLLM can call.
    /// - Unimplemented variants (`FileSearch`, `CodeInterpreter`) return
    ///   `None` and emit a `tracing::debug!`.
    ///
    /// This is the entry point called by `RequestPayload::to_upstream_request()` so that
    /// vLLM always receives a `Vec<FunctionTool>`, never a raw `ResponsesTool` enum.
    #[must_use]
    pub fn to_function_tool(&self) -> Option<FunctionTool> {
        match self {
            // name is NonEmptyToolName — empty names are rejected by serde at
            // deserialization time, so no runtime check is needed here.
            ResponsesTool::Function(p) => {
                let config = serialize_to_value(p)
                    .inspect_err(|error| tracing::debug!(error = %error, "function tool config serialization failed"))
                    .ok()?;
                FunctionHandler.normalize(&config).into_iter().next()
            }
            ResponsesTool::Mcp(p) if p.name.as_str() == READ_MCP_RESOURCE_TOOL_NAME => {
                let config = serialize_to_value(p)
                    .inspect_err(|error| tracing::debug!(error = %error, "MCP tool config serialization failed"))
                    .ok()?;
                McpHandler::read_resource_spec_only()
                    .normalize(&config)
                    .into_iter()
                    .next()
            }
            ResponsesTool::Mcp(p) => {
                tracing::debug!(name = %p.name, "unknown MCP built-in skipped in normalize");
                None
            }
            ResponsesTool::WebSearch(_) => Some(web_search_function_tool()),
            ResponsesTool::FileSearch(_) => {
                tracing::debug!("file_search tool skipped in normalize — handler not yet registered");
                None
            }
            ResponsesTool::CodeInterpreter(_) => {
                tracing::debug!("code_interpreter tool skipped in normalize — handler not yet registered");
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
