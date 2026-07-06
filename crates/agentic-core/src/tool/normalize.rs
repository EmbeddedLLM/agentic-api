use crate::types::io::input::FunctionToolResultMessage;
use crate::types::io::{FunctionTool, OutputItem};
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
    /// [`super::codex::CodexParams::Unknown`], any tool `type` unrecognized by
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

    /// Reverses tool-declaration normalization on every `function_call` in
    /// `output`, via each declared tool's [`ToolHandler::unnormalize`].
    ///
    /// Needed because some declarations (e.g. a Codex `namespace`) generate a
    /// model-visible name that differs from what the call should look like
    /// once it reaches the client — see
    /// [`super::codex::CodexToolHandler::unnormalize`]. Tries every declared
    /// tool against each call until one claims it (sets `call.namespace`); a
    /// call that already has `namespace` set, or that doesn't match any
    /// declared tool's generated name, is left unchanged.
    pub fn unnormalize_output_items(tools: Option<&[ResponsesTool]>, output: &mut [OutputItem]) {
        let Some(tools) = tools else {
            return;
        };
        for item in output {
            let OutputItem::FunctionCall(call) = item else {
                continue;
            };
            for tool in tools {
                let Some(handler) = tool.handler() else {
                    continue;
                };
                handler.unnormalize(&serialize_to_value(tool), call);
                if call.namespace.is_some() {
                    break;
                }
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
