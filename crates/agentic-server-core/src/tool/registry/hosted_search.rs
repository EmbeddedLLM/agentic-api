//! Registry entries for tool search: one kind, owned where its declaration runs it, the hosted search's per-response
//! catalog binding, and the tools a search loads.

use std::collections::HashMap;
use std::future::{Future, ready};
use std::pin::Pin;
use std::sync::Arc;

use super::super::catalog::ToolCatalog;
use super::super::declaration::{ToolDeclaration, responses_declarations};
use super::super::executors::GatewayExecutors;
use super::super::handler::{GatewayExecutor, GatewayToolEventPlan, PlannedResult, ToolOutput};
use super::super::ownership::{GatewayBinding, ToolOwnership};
use super::super::tool_search::{TOOL_SEARCH_NAME, ToolSearchHandler};
use super::super::{CodexNamespaceHandler, ToolError};
use super::{CustomHandler, ToolEntry, ToolRegistry, ToolType};
use crate::types::io::OutputItem;
use crate::types::io::input::hosted_search_output_text;
use crate::types::io::output::FunctionToolCall;
use crate::types::io::{ToolSearchCall, ToolSearchOutputMessage};
use crate::types::tools::ResponsesTool;
use crate::types::tools::{ToolSearchStatus, ToolSearchToolParam};

impl ToolRegistry {
    /// Adds client entries for tools a hosted search loaded mid-response, and
    /// rebuilds the namespace and custom maps that restore their public shape.
    /// Gateway entries, including discovered MCP tools, are kept as built, so
    /// no MCP server is listed again.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when a loaded name collides with another tool.
    pub(crate) async fn refresh_client_tools(&mut self, tools: &[ResponsesTool]) -> Result<(), ToolError> {
        let declarations = responses_declarations(tools);
        let mut client_declarations: Vec<_> = declarations
            .iter()
            .filter(|declaration| {
                matches!(
                    declaration,
                    ToolDeclaration::Function(_) | ToolDeclaration::Namespace(_) | ToolDeclaration::Custom(_)
                )
            })
            .cloned()
            .collect();
        let client = Self::build_with_handlers(&mut client_declarations, &mut GatewayExecutors::default()).await?;
        for (name, entry) in client.entries {
            self.entries.entry(name).or_insert(entry);
        }
        self.namespace_map =
            CodexNamespaceHandler.build_namespace_map((!declarations.is_empty()).then_some(&declarations))?;
        self.custom_tool_map = CustomHandler::build_tool_map(&declarations);
        Ok(())
    }

    /// Binds the hosted `tool_search` entry to this response's catalog.
    pub(crate) fn bind_hosted_tool_search(&mut self, catalog: ToolCatalog) {
        if self.is_hosted_tool_search_name(TOOL_SEARCH_NAME) {
            self.entries
                .insert(TOOL_SEARCH_NAME.to_owned(), hosted_tool_search_entry(catalog));
        }
    }

    /// Whether `name` is the hosted `tool_search` the gateway runs. Its model
    /// call is replaced by the call result's own model input and history.
    #[must_use]
    pub fn is_hosted_tool_search_name(&self, name: &str) -> bool {
        self.entries
            .get(name)
            .is_some_and(|entry| entry.tool_type == ToolType::ToolSearch && entry.ownership.is_gateway())
    }

    /// Whether an upstream output item is recorded as the model saw it. A
    /// hosted search call is recorded through its call result instead.
    #[must_use]
    pub(crate) fn records_output_item(&self, item: &OutputItem) -> bool {
        !matches!(item, OutputItem::FunctionCall(call) if self.is_hosted_tool_search_name(&call.name))
    }
}

/// A hosted search is bound to an empty catalog here; the agent turn binds the
/// response's catalog once MCP discovery has run.
pub(crate) fn insert_tool_search_entry(entries: &mut HashMap<String, ToolEntry>, param: &ToolSearchToolParam) {
    let entry = if param.is_hosted() {
        hosted_tool_search_entry(ToolCatalog::default())
    } else {
        tool_search_entry(ToolOwnership::Client)
    };
    if entries.insert(TOOL_SEARCH_NAME.to_owned(), entry).is_some() {
        tracing::warn!(
            name = TOOL_SEARCH_NAME,
            "duplicate tool name — previous definition overwritten"
        );
    }
}

/// The gateway runs a hosted search with [`ToolSearchHandler`] over this response's catalog.
pub(crate) fn hosted_tool_search_entry(catalog: ToolCatalog) -> ToolEntry {
    tool_search_entry(ToolOwnership::Gateway(Some(GatewayBinding::new(
        Arc::new(ToolSearchHandler),
        Arc::new(catalog),
    ))))
}

/// The one tool-search kind; `ownership` comes from the declaration's `execution`.
fn tool_search_entry(ownership: ToolOwnership) -> ToolEntry {
    ToolEntry {
        tool_type: ToolType::ToolSearch,
        server_label: None,
        ownership,
        async_execution: false,
    }
}

/// A hosted search runs in the gateway: a pure lookup over the response's catalog, so the public
/// `tool_search_output` is already known when the call is planned. Validation and lowering are the handler's own.
impl GatewayExecutor for ToolSearchHandler {
    type ExecutionParams = Arc<ToolCatalog>;

    /// The planned result answers a hosted search (see [`Self::plan_gateway_events`]); this runs the same search for
    /// the trait.
    fn execute(
        &self,
        call_id: &str,
        _tool_name: &str,
        arguments: &str,
        params: &Self::ExecutionParams,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let output = hosted_search_output_text(&params.search_raw(arguments));
        Box::pin(ready(Ok(ToolOutput::success(call_id, output))))
    }

    fn supports_parallel_execution(&self) -> bool {
        true
    }

    /// The search is pure, so its result is known now: the gateway records the public pair without executing the
    /// call, and the upstream projection shows the model its `tool_search` call and output.
    fn plan_gateway_events(&self, call: &FunctionToolCall, params: &Self::ExecutionParams) -> GatewayToolEventPlan {
        let output = ToolSearchOutputMessage::hosted(call, params.search_raw(&call.arguments));
        GatewayToolEventPlan::new(Some(OutputItem::ToolSearchCall(ToolSearchCall::hosted(
            call,
            ToolSearchStatus::InProgress,
        ))))
        .with_planned_result(PlannedResult::search_pair(
            ToolSearchCall::hosted(call, ToolSearchStatus::Completed),
            output,
        ))
    }
}
