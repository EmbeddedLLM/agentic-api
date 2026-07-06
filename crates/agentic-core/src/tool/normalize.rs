use crate::types::io::FunctionTool;
use crate::types::io::input::FunctionToolResultMessage;
use crate::types::tools::ResponsesTool;
use crate::utils::common::serialize_to_value;

use super::codex::CodexToolHandler;
use super::function::FunctionHandler;
use super::handler::{ToolHandler, ToolOutput};

impl ResponsesTool {
    /// Returns the [`ToolHandler`] responsible for validating and normalizing
    /// this variant, or `None` for variants whose handler has not landed yet
    /// (`Mcp`, `WebSearch`, `FileSearch`, `CodeInterpreter`).
    ///
    /// `Codex` covers both Codex CLI's own shapes and, via
    /// [`super::codex::CodexTools::Unknown`], any tool `type` unrecognized by
    /// the other variants (`#[non_exhaustive]` catch-all — see
    /// [`crate::types::tools::ResponsesTool`]'s docs).
    ///
    /// Single dispatch point shared by [`Self::to_function_tools`] and
    /// [`super::registry::ToolRegistry::build`] so variant → handler routing
    /// only lives in one place.
    #[must_use]
    pub fn handler(&self) -> Option<&'static dyn ToolHandler> {
        match self {
            ResponsesTool::Function(_) => Some(&FunctionHandler),
            ResponsesTool::Codex(_) => Some(&CodexToolHandler),
            ResponsesTool::Mcp(_)
            | ResponsesTool::WebSearch(_)
            | ResponsesTool::FileSearch(_)
            | ResponsesTool::CodeInterpreter(_) => None,
        }
    }

    /// Normalise this tool declaration to the `FunctionTool` wire format(s) that
    /// vLLM understands, via [`Self::handler`].
    ///
    /// `Codex(Namespace)` fans out into one `FunctionTool` per subtool, so this
    /// returns a `Vec` rather than a single value. Variants with no handler
    /// (`Mcp`, `WebSearch`, `FileSearch`, `CodeInterpreter`) return an empty
    /// `Vec` and emit a `tracing::debug!` — their full handlers have not
    /// landed yet.
    ///
    /// Callers building a `Vec<FunctionTool>` from a tool list should
    /// `.flat_map(ResponsesTool::to_function_tools)` so `Codex` namespace
    /// expansion is not silently truncated to one subtool.
    #[must_use]
    pub fn to_function_tools(&self) -> Vec<FunctionTool> {
        let Some(handler) = self.handler() else {
            tracing::debug!(tool = ?self, "tool shape skipped in normalize — no handler registered");
            return vec![];
        };
        handler.normalize(&serialize_to_value(self))
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
