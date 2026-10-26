//! The model's view of a request's input.
//!
//! `types::io::input` projects search items, replayed gateway calls and MCP names; this resolves namespaced calls
//! against the request's declarations and lowers async calls, so a live round, a continuation, a replay and a
//! compaction summary show the model its history the same way.

use std::borrow::Cow;

use super::RequestPayload;
use crate::tool::async_execution::lower_async_calls;
use crate::tool::{CodexNamespaceHandler, NamespaceMap, ToolDeclaration};
use crate::types::io::ResponsesInput;
use crate::types::io::input::{InputValidationError, McpToolNames};

impl RequestPayload {
    /// The input as the model reads it, given the request's `declarations` and their namespace `map`.
    ///
    /// # Errors
    ///
    /// Returns a file that cannot be converted to text.
    pub(super) fn upstream_input(
        &self,
        declarations: Option<&[ToolDeclaration]>,
        map: Option<&NamespaceMap>,
    ) -> Result<Cow<'_, ResponsesInput>, InputValidationError> {
        let mcp_names = McpToolNames::from_tools(self.tools.as_deref().unwrap_or_default());
        // A call naming itself is to a searched function only if this request declares that function or a search
        // output in its input loaded it (a compaction summary declares no tools).
        let mut functions = self.input.searched_function_names();
        functions.extend(
            declarations
                .into_iter()
                .flatten()
                .filter_map(ToolDeclaration::function_name),
        );
        Ok(lower_async_calls(CodexNamespaceHandler.resolve_input(
            map,
            &functions,
            self.input.normalized_model_input_with(&mcp_names)?,
        )))
    }
}
