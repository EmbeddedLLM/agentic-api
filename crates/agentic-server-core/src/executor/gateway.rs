use std::sync::OnceLock;
use std::time::Duration;

use futures::StreamExt;
use futures::stream as futures_stream;

use crate::config::DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS;
use crate::events::SSEEventType;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::gateway_accumulator::{GatewayStreamAccumulator, StreamEvent, emit_sse_frame, synthetic_event};
use crate::executor::request::RequestContext;
use crate::tool::{ToolError, ToolOutput, ToolOwnership, ToolRegistry, ToolType};
use crate::types::io::output::{FunctionToolCall, GatewayCallStatus, McpCallStatus};
use crate::types::io::{InputItem, OutputItem, ResponsesInput};
use crate::types::request_response::ResponsePayload;
use crate::utils::common::{serialize_to_string, serialize_to_value};

/// Upper bound on gateway tool calls executing at once within a round. A sliding window:
/// as one call finishes, the next is admitted, so a round with N calls never
/// runs more than this many concurrently but still drains all N. Bounds
/// outbound fan-out without a hard per-round count cap.
/// Per-tool self-exclusion may reduce actual concurrency below this limit.
///
/// The call count is bounded upstream by the model's output size — there is no
/// unbounded in-memory materialisation from the model emitting arbitrarily many
/// tool calls. The window + per-call timeout bound outbound HTTP and latency.
///
/// Resolved once from [`Config::tools.max_concurrent_gateway_calls`](crate::config::ToolRuntimeConfig::max_concurrent_gateway_calls)
/// via [`set_max_concurrent_gateway_calls`] when `ExecutionContext` is built from
/// config; falls back to [`DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS`] otherwise (e.g.
/// in tests that construct `ExecutionContext::new` directly).
static MAX_CONCURRENT_GATEWAY_CALLS: OnceLock<usize> = OnceLock::new();

/// Sets the process-wide gateway concurrency limit. Called once, from
/// `ExecutionContext::from_config`. Subsequent calls are no-ops: the value is
/// fixed for the lifetime of the process.
pub fn set_max_concurrent_gateway_calls(value: usize) {
    let _ = MAX_CONCURRENT_GATEWAY_CALLS.set(value);
}

fn max_concurrent_gateway_calls() -> usize {
    *MAX_CONCURRENT_GATEWAY_CALLS.get_or_init(|| DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS)
}

/// Per-call wall-clock budget. A tool exceeding this yields an error output fed
/// back to the model (never a whole-request failure). `Duration::ZERO` disables
/// the timeout — for providers that manage their own.
///
/// Note: this bounds a single call, not the whole request. Worst-case request
/// latency scales with rounds and fan-out; an outer request-level deadline
/// would be the place to cap total time end-to-end.
const GATEWAY_TOOL_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub(super) struct GatewayCallResult {
    pub(super) call: FunctionToolCall,
    pub(super) input_item: InputItem,
    pub(super) public_output: Option<OutputItem>,
}

/// Supplies the public output that completes a gateway event plan.
pub(super) trait GatewayPublicOutputSource {
    fn public_output(&self) -> Option<&OutputItem>;
}

impl GatewayPublicOutputSource for GatewayCallResult {
    fn public_output(&self) -> Option<&OutputItem> {
        self.public_output.as_ref()
    }
}

impl GatewayPublicOutputSource for OutputItem {
    fn public_output(&self) -> Option<&OutputItem> {
        Some(self)
    }
}

#[derive(Clone)]
pub(super) struct GatewayEventPlan {
    output_index: u32,
    started_output: Option<OutputItem>,
    completed_output: Option<OutputItem>,
    arguments: Option<String>,
}

fn function_calls(output_items: &[OutputItem]) -> Vec<FunctionToolCall> {
    output_items
        .iter()
        .filter_map(|item| match item {
            OutputItem::FunctionCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

pub(super) fn has_client_owned_calls(output_items: &[OutputItem], registry: &ToolRegistry) -> bool {
    output_items.iter().any(|item| item.requires_client_action(registry))
}

fn execution_error_output(call: &FunctionToolCall, message: &str) -> ExecutorResult<ToolOutput> {
    let output = serialize_to_string(&serde_json::json!({ "error": message })).map_err(ExecutorError::JsonError)?;
    Ok(ToolOutput {
        call_id: call.call_id.clone(),
        output,
    })
}

/// Executes one round's gateway-owned calls with bounded concurrency and
/// per-tool same-name exclusion.
pub(super) struct GatewayRound {
    timeout: Duration,
}

impl GatewayRound {
    pub(super) fn new() -> Self {
        Self {
            timeout: GATEWAY_TOOL_TIMEOUT,
        }
    }

    /// Same as [`Self::new`] but with an injectable per-call timeout, so tests
    /// can exercise the timeout path without waiting out the real budget.
    #[cfg(test)]
    fn with_timeout(timeout: Duration) -> Self {
        Self { timeout }
    }

    /// Runs every gateway-owned call in `output_items` under the
    /// `max_concurrent_gateway_calls()` upper bound. `buffered` admits the next
    /// call as soon as one finishes; each resolved binding may additionally
    /// serialize calls to its own tool name. Result order matches call order
    /// regardless of completion order.
    pub(super) async fn execute(
        &self,
        output_items: &[OutputItem],
        registry: &ToolRegistry,
    ) -> ExecutorResult<Vec<GatewayCallResult>> {
        let calls = function_calls(output_items);
        let gateway_calls = registry.gateway_owned(&calls);

        futures_stream::iter(
            gateway_calls
                .into_iter()
                .cloned()
                .map(|call| self.run_one(call, registry)),
        )
        .buffered(max_concurrent_gateway_calls())
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect()
    }

    /// Resolves the call's `GatewayBinding`, takes its `self_exclusion` permit
    /// if it has one (gating only concurrent calls to this SAME tool name —
    /// never other tools), dispatches with a timeout, and shapes the public
    /// output via the resolved handler.
    async fn run_one(&self, call: FunctionToolCall, registry: &ToolRegistry) -> ExecutorResult<GatewayCallResult> {
        let Some(entry) = registry.lookup(&call.name) else {
            return Err(ExecutorError::InvalidRequest(format!(
                "gateway tool '{}' was not dispatchable",
                call.name
            )));
        };
        let ToolOwnership::Gateway(binding) = &entry.ownership else {
            return Err(ExecutorError::InvalidRequest(format!(
                "'{}' is not a gateway tool",
                call.name
            )));
        };
        let config = entry.config.clone();

        let Some(binding) = binding else {
            // Registered as gateway-owned but no handler implemented yet
            // (`FileSearch`/`CodeInterpreter` today). Surface an error output
            // fed back to the model rather than failing the whole request.
            let output = execution_error_output(
                &call,
                &format!("gateway tool '{}' has no registered handler", call.name),
            )?;
            return Ok(GatewayCallResult {
                call,
                input_item: InputItem::FunctionCallOutput(output.into()),
                public_output: None,
            });
        };

        let _permit = match &binding.self_exclusion {
            Some(semaphore) => Some(semaphore.acquire().await.expect("semaphore is never closed")),
            None => None,
        };

        // Per-call timeout: a hung tool becomes an error output fed back to
        // the model, never a whole-request failure. A zero timeout opts out
        // (used by tests exercising the "no registered handler" path without
        // waiting).
        let dispatched = if self.timeout.is_zero() {
            binding
                .handler
                .execute(&call.call_id, &call.name, &call.arguments, &config)
                .await
        } else {
            match tokio::time::timeout(
                self.timeout,
                binding
                    .handler
                    .execute(&call.call_id, &call.name, &call.arguments, &config),
            )
            .await
            {
                Ok(output) => output,
                Err(_elapsed) => Err(ToolError::Execution(format!(
                    "gateway tool '{}' timed out after {:?}",
                    call.name, self.timeout
                ))),
            }
        };
        let (output, status) = match dispatched {
            Ok(output) => (output, GatewayCallStatus::Completed),
            Err(ToolError::Execution(message) | ToolError::Config(message)) => {
                (execution_error_output(&call, &message)?, GatewayCallStatus::Failed)
            }
            // No `GatewayExecutor::execute` implementation ever returns this —
            // it's only raised as a request-validation error before dispatch
            // begins (`executor::rehydrate`). Propagate rather than treat as a
            // per-call failure if that invariant is ever violated.
            Err(error @ ToolError::MissingOutput { .. }) => return Err(ExecutorError::from(error)),
        };
        let public_output = binding.handler.public_output(&call, &output, status);
        Ok(GatewayCallResult {
            call,
            input_item: InputItem::FunctionCallOutput(output.into()),
            public_output,
        })
    }
}

pub(super) fn public_output_items(
    output_items: &[OutputItem],
    registry: &ToolRegistry,
    gateway_results: &[GatewayCallResult],
) -> Vec<OutputItem> {
    output_items
        .iter()
        .map(|item| match item {
            OutputItem::FunctionCall(call) if registry.is_client_custom_name(&call.name) => {
                crate::tool::CustomHandler::output_item(call)
            }
            OutputItem::FunctionCall(call) if registry.is_gateway_owned_name(&call.name) => gateway_results
                .iter()
                .find(|result| result.call.call_id == call.call_id)
                .and_then(|result| result.public_output.clone())
                .unwrap_or_else(|| OutputItem::FunctionCall(call.clone())),
            other => other.clone(),
        })
        .collect()
}

pub(super) fn gateway_event_plans(
    output_items: &[OutputItem],
    registry: &ToolRegistry,
    output_offset: usize,
) -> Vec<GatewayEventPlan> {
    let mut output_index = output_offset;
    let mut plans = Vec::new();
    for item in output_items {
        if let OutputItem::FunctionCall(call) = item
            && let Some(entry) = registry.lookup(&call.name)
            && let ToolOwnership::Gateway(Some(binding)) = &entry.ownership
        {
            plans.push(GatewayEventPlan {
                output_index: u32::try_from(output_index).unwrap_or(u32::MAX),
                arguments: (entry.tool_type == ToolType::Mcp).then(|| call.arguments.clone()),
                started_output: binding.handler.started_output(call),
                completed_output: None,
            });
        }
        output_index = output_index.saturating_add(1);
    }
    plans
}

pub(super) fn mcp_list_tools_event_plans(
    public_output_items: &[OutputItem],
    output_offset: usize,
) -> Vec<GatewayEventPlan> {
    public_output_items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let OutputItem::McpListTools(list_tools) = item else {
                return None;
            };
            Some(GatewayEventPlan {
                output_index: u32::try_from(output_offset.saturating_add(index)).unwrap_or(u32::MAX),
                started_output: Some(crate::tool::mcp::handler::started_list_tools_output_item(list_tools)),
                completed_output: Some(item.clone()),
                arguments: None,
            })
        })
        .collect()
}

pub(super) fn compaction_event_plans(
    public_output_items: &[OutputItem],
    output_offset: usize,
) -> Vec<GatewayEventPlan> {
    public_output_items
        .iter()
        .enumerate()
        .filter(|(_, item)| matches!(item, OutputItem::Compaction(_)))
        .map(|(index, item)| GatewayEventPlan {
            output_index: u32::try_from(output_offset.saturating_add(index)).unwrap_or(u32::MAX),
            started_output: Some(item.clone()),
            completed_output: Some(item.clone()),
            arguments: None,
        })
        .collect()
}

fn output_item_value(item: &OutputItem) -> ExecutorResult<serde_json::Value> {
    serde_json::to_value(item).map_err(ExecutorError::JsonError)
}

pub(super) fn emit_response_start_events(
    payload: &ResponsePayload,
    stream_accumulator: &mut GatewayStreamAccumulator,
    stream_sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
) -> ExecutorResult<()> {
    let mut response = payload.clone();
    "in_progress".clone_into(&mut response.status);
    response.output.clear();
    response.usage = None;
    let response = serialize_to_value(&response).map_err(ExecutorError::JsonError)?;
    for event_type in [SSEEventType::ResponseCreated, SSEEventType::ResponseInProgress] {
        let mut event = synthetic_event(event_type, [("response".to_owned(), response.clone())])?;
        emit_gateway_event(&mut event, stream_accumulator, stream_sender)?;
    }
    Ok(())
}

pub(super) fn complete_gateway_event_plans<T: GatewayPublicOutputSource>(
    plans: &mut [GatewayEventPlan],
    completed: &[T],
) {
    for (plan, source) in plans.iter_mut().zip(completed) {
        plan.completed_output = source.public_output().cloned();
    }
}

pub(super) fn emit_gateway_start_events(
    plans: &[GatewayEventPlan],
    stream_accumulator: &mut GatewayStreamAccumulator,
    stream_sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
) -> ExecutorResult<()> {
    for plan in plans {
        let Some(output_item) = &plan.started_output else {
            continue;
        };
        let item = output_item_value(output_item)?;
        let mut added_event = synthetic_event(
            SSEEventType::OutputItemAdded,
            [
                ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                ("item".to_owned(), item),
            ],
        )?;
        emit_gateway_event(&mut added_event, stream_accumulator, stream_sender)?;
        match output_item {
            OutputItem::WebSearchCall(web_search_call) => {
                let mut in_progress_event = synthetic_event(
                    SSEEventType::WebSearchCallInProgress,
                    [
                        ("item_id".to_owned(), serde_json::json!(web_search_call.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut in_progress_event, stream_accumulator, stream_sender)?;
                let mut searching_event = synthetic_event(
                    SSEEventType::WebSearchCallSearching,
                    [
                        ("item_id".to_owned(), serde_json::json!(web_search_call.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut searching_event, stream_accumulator, stream_sender)?;
            }
            OutputItem::McpCall(mcp_call) => {
                let mut in_progress_event = synthetic_event(
                    SSEEventType::McpCallInProgress,
                    [
                        ("item_id".to_owned(), serde_json::json!(mcp_call.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut in_progress_event, stream_accumulator, stream_sender)?;
                let arguments = plan.arguments.as_deref().unwrap_or_default();
                let mut arguments_delta_event = synthetic_event(
                    SSEEventType::McpCallArgumentsDelta,
                    [
                        ("delta".to_owned(), serde_json::json!(arguments)),
                        ("item_id".to_owned(), serde_json::json!(mcp_call.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut arguments_delta_event, stream_accumulator, stream_sender)?;
                let mut arguments_done_event = synthetic_event(
                    SSEEventType::McpCallArgumentsDone,
                    [
                        ("arguments".to_owned(), serde_json::json!(arguments)),
                        ("item_id".to_owned(), serde_json::json!(mcp_call.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut arguments_done_event, stream_accumulator, stream_sender)?;
            }
            OutputItem::McpListTools(list_tools) => {
                let mut in_progress_event = synthetic_event(
                    SSEEventType::McpListToolsInProgress,
                    [
                        ("item_id".to_owned(), serde_json::json!(list_tools.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut in_progress_event, stream_accumulator, stream_sender)?;
            }
            OutputItem::Message(_)
            | OutputItem::FunctionCall(_)
            | OutputItem::CustomToolCall(_)
            | OutputItem::Reasoning(_)
            | OutputItem::Compaction(_)
            | OutputItem::Unknown => {}
        }
    }
    Ok(())
}

pub(super) fn emit_gateway_completed_events<T: GatewayPublicOutputSource>(
    results: &[T],
    plans: &[GatewayEventPlan],
    stream_accumulator: &mut GatewayStreamAccumulator,
    stream_sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
) -> ExecutorResult<()> {
    for (index, plan) in plans.iter().enumerate() {
        let Some(public_output) = plan
            .completed_output
            .as_ref()
            .or_else(|| results.get(index).and_then(GatewayPublicOutputSource::public_output))
        else {
            continue;
        };
        let output_index = plan.output_index;
        let completed_event = match public_output {
            OutputItem::WebSearchCall(web_search_call) => {
                Some((SSEEventType::WebSearchCallCompleted, web_search_call.id.as_str()))
            }
            OutputItem::McpCall(mcp_call) => Some((
                if mcp_call.status == Some(McpCallStatus::Failed) {
                    SSEEventType::McpCallFailed
                } else {
                    SSEEventType::McpCallCompleted
                },
                mcp_call.id.as_str(),
            )),
            OutputItem::McpListTools(list_tools) => Some((
                if list_tools.error.is_some() {
                    SSEEventType::McpListToolsFailed
                } else {
                    SSEEventType::McpListToolsCompleted
                },
                list_tools.id.as_str(),
            )),
            OutputItem::Compaction(_) => None,
            OutputItem::Message(_)
            | OutputItem::FunctionCall(_)
            | OutputItem::CustomToolCall(_)
            | OutputItem::Reasoning(_)
            | OutputItem::Unknown => continue,
        };
        let item = output_item_value(public_output)?;
        if let Some((event_type, item_id)) = completed_event {
            let mut completed_fields = serde_json::Map::from_iter([
                ("item_id".to_owned(), serde_json::json!(item_id)),
                ("output_index".to_owned(), serde_json::json!(output_index)),
            ]);
            if matches!(public_output, OutputItem::WebSearchCall(_)) {
                completed_fields.insert("item".to_owned(), item.clone());
            }
            let mut completed_event = synthetic_event(event_type, completed_fields)?;
            emit_gateway_event(&mut completed_event, stream_accumulator, stream_sender)?;
        }
        let mut done_event = synthetic_event(
            SSEEventType::OutputItemDone,
            [
                ("output_index".to_owned(), serde_json::json!(output_index)),
                ("item".to_owned(), item),
            ],
        )?;
        emit_gateway_event(&mut done_event, stream_accumulator, stream_sender)?;
    }
    Ok(())
}

pub(super) async fn execute_and_emit_output_calls(
    output_items: &[OutputItem],
    registry: &ToolRegistry,
    output_offset: usize,
    mut stream: Option<(
        &mut GatewayStreamAccumulator,
        &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    )>,
) -> ExecutorResult<Vec<GatewayCallResult>> {
    let mut event_plans = gateway_event_plans(output_items, registry, output_offset);
    if let Some((stream_accumulator, stream_sender)) = stream.as_mut() {
        emit_gateway_start_events(&event_plans, stream_accumulator, stream_sender)?;
    }
    let gateway_results = GatewayRound::new().execute(output_items, registry).await?;
    complete_gateway_event_plans(&mut event_plans, &gateway_results);
    if let Some((stream_accumulator, stream_sender)) = stream.as_mut() {
        emit_gateway_completed_events(&gateway_results, &event_plans, stream_accumulator, stream_sender)?;
    }
    Ok(gateway_results)
}

fn emit_gateway_event(
    frame: &mut crate::events::EventFrame,
    stream_accumulator: &mut GatewayStreamAccumulator,
    stream_sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
) -> ExecutorResult<()> {
    if stream_accumulator.process_event(frame, 0) {
        emit_sse_frame(stream_sender, frame)?;
    }
    Ok(())
}

pub(super) fn append_input_item(input: &mut ResponsesInput, item: InputItem) {
    match input {
        ResponsesInput::Items(items) => items.push(item),
        ResponsesInput::Text(text) => {
            let text_input = ResponsesInput::Text(std::mem::take(text));
            let mut items = Vec::<InputItem>::from(&text_input);
            items.push(item);
            *input = ResponsesInput::Items(items);
        }
    }
}

pub(super) fn append_output_items_to_input(input: &mut ResponsesInput, output_items: &[OutputItem]) {
    for input_item in output_items.iter().flat_map(OutputItem::to_input_item) {
        append_input_item(input, input_item);
    }
}

pub(super) fn append_tool_outputs(ctx: &mut RequestContext, tool_outputs: Vec<InputItem>) {
    for output in tool_outputs {
        ctx.new_input_items.push(output.clone());
        append_input_item(&mut ctx.enriched_request.input, output);
    }
}

pub(super) fn append_gateway_calls_to_new_input(
    ctx: &mut RequestContext,
    output_items: &[OutputItem],
    registry: &ToolRegistry,
) {
    ctx.new_input_items.extend(output_items.iter().filter_map(|item| {
        let OutputItem::FunctionCall(call) = item else {
            return None;
        };
        registry
            .is_gateway_owned_name(&call.name)
            .then(|| InputItem::FunctionCall(call.clone().into()))
    }));
}

#[cfg(test)]
mod tests {
    use super::GatewayCallResult;
    use crate::executor::accumulator::ResponseAccumulator;
    use crate::types::io::output::{FunctionToolCall, McpListTool, McpListTools};
    use crate::types::io::{CompactionItem, InputItem, McpCallStatus};
    use tokio::sync::mpsc;

    fn parse_named_sse_event(content: &str) -> Value {
        let body = content.strip_suffix("\n\n").expect("SSE event terminator");
        let (event_line, data_line) = body.split_once('\n').expect("named SSE event and data lines");
        let event_name = event_line.strip_prefix("event: ").expect("SSE event prefix");
        let data = data_line.strip_prefix("data: ").expect("SSE data prefix");
        let event = serde_json::from_str::<Value>(data).expect("event JSON");
        assert_eq!(event["type"].as_str(), Some(event_name));
        event
    }

    use std::pin::Pin;
    use std::sync::Arc;

    use serde_json::Value;

    use super::GatewayRound;
    use crate::tool::{GatewayExecutor, GatewayExecutors, ToolError, ToolHandler, ToolOutput, ToolRegistry, ToolType};
    use crate::types::io::OutputItem;
    use crate::types::io::output::GatewayCallStatus;
    use crate::types::io::tools::FunctionTool;
    use crate::types::tools::ResponsesTool;

    /// A gateway executor that sleeps ~50ms — comfortably longer than the tiny
    /// timeout the test injects, forcing the timeout path without a paused clock.
    struct SlowExecutor;

    impl ToolHandler for SlowExecutor {
        fn tool_type(&self) -> ToolType {
            ToolType::WebSearch
        }
        fn validate(&self, _param: &Value) -> Result<(), ToolError> {
            Ok(())
        }
        fn normalize(&self, _param: &Value) -> Vec<FunctionTool> {
            Vec::new()
        }
    }

    impl GatewayExecutor for SlowExecutor {
        fn execute(
            &self,
            call_id: &str,
            _tool_name: &str,
            _arguments: &str,
            _config: &Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            let call_id = call_id.to_owned();
            Box::pin(async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(ToolOutput {
                    call_id,
                    output: "unreachable".to_owned(),
                })
            })
        }

        fn supports_parallel_execution(&self) -> bool {
            true
        }

        fn started_output(&self, call: &FunctionToolCall) -> Option<OutputItem> {
            Some(crate::tool::web_search::started_output_item(call))
        }

        fn public_output(
            &self,
            call: &FunctionToolCall,
            output: &ToolOutput,
            status: GatewayCallStatus,
        ) -> Option<OutputItem> {
            Some(crate::tool::web_search::output_item(call, output, status))
        }
    }

    fn web_search_call(call_id: &str) -> FunctionToolCall {
        FunctionToolCall {
            id: format!("fc_{call_id}"),
            call_id: call_id.to_owned(),
            name: "web_search".to_owned(),
            arguments: "{}".to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        }
    }

    #[tokio::test]
    async fn hung_gateway_call_times_out_into_error_output() {
        let web_search: ResponsesTool =
            serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param");
        let mut executors = GatewayExecutors::default();
        executors.insert(Arc::new(SlowExecutor));
        let mut tools = [web_search];
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry builds");

        // 1ms budget vs a 50ms tool → the timeout fires. Must return (not hang):
        // the stuck call becomes an error output the loop can feed back.
        let result = GatewayRound::with_timeout(std::time::Duration::from_millis(1))
            .run_one(web_search_call("call_hang"), &registry)
            .await
            .expect("timeout is isolated as an error output, not a dispatch failure");

        assert_eq!(result.call.call_id, "call_hang");
        // A failed web_search still yields a public web_search_call item.
        assert!(matches!(result.public_output, Some(OutputItem::WebSearchCall(_))));
        // The fed-back tool output is an error JSON mentioning the timeout.
        let InputItem::FunctionCallOutput(msg) = &result.input_item else {
            panic!("expected a function_call_output");
        };
        let body = serde_json::to_string(msg).expect("serialize output");
        assert!(
            body.contains("timed out"),
            "error output should mention the timeout: {body}"
        );
    }

    #[tokio::test]
    async fn gateway_call_without_configured_provider_becomes_error_output() {
        // Declare web_search but build the registry with NO provider for it —
        // `web_search_handler()` falls back to `WebSearchHandler::spec_only()`,
        // so the entry still has a real handler (public_output/started_output
        // work normally) but `execute()` fails. This must surface an error
        // output, not fail the whole request.
        let web_search: ResponsesTool =
            serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param");
        let mut tools = [web_search];
        let mut executors = GatewayExecutors::default();
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry builds");

        let result = GatewayRound::with_timeout(std::time::Duration::ZERO)
            .run_one(web_search_call("call_no_handler"), &registry)
            .await
            .expect("a missing provider is isolated as an error output, not a dispatch failure");

        assert_eq!(result.call.call_id, "call_no_handler");
        assert!(matches!(result.public_output, Some(OutputItem::WebSearchCall(_))));
        let InputItem::FunctionCallOutput(msg) = &result.input_item else {
            panic!("expected a function_call_output");
        };
        let body = serde_json::to_string(msg).expect("serialize output");
        assert!(
            body.contains("spec-only handler cannot execute tools"),
            "error output should mention the missing provider: {body}"
        );
    }

    /// Proves that a handler which opts into same-tool parallel execution really
    /// overlaps calls under the global window. Four 50ms calls sequentially would
    /// take ~200ms; with the default window of five they should finish near one slot.
    #[tokio::test]
    async fn gateway_round_overlaps_parallel_safe_same_tool_calls() {
        let web_search: ResponsesTool =
            serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param");
        let mut executors = GatewayExecutors::default();
        executors.insert(Arc::new(SlowExecutor));
        let mut tools = [web_search];
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry builds");

        let output_items: Vec<OutputItem> = ["call_a", "call_b", "call_c", "call_d"]
            .into_iter()
            .map(|call_id| OutputItem::FunctionCall(web_search_call(call_id)))
            .collect();

        let started = std::time::Instant::now();
        let results = GatewayRound::new()
            .execute(&output_items, &registry)
            .await
            .expect("all calls execute");
        let elapsed = started.elapsed();

        assert_eq!(results.len(), 4);
        // Well under the ~200ms four sequential 50ms calls would take, and close to
        // one 50ms slot -- proves the calls actually overlapped in wall-clock time.
        assert!(
            elapsed < std::time::Duration::from_millis(150),
            "four concurrent 50ms calls took {elapsed:?}, expected well under 150ms"
        );
        // buffered() preserves call order regardless of completion order.
        assert_eq!(
            results.iter().map(|r| r.call.call_id.as_str()).collect::<Vec<_>>(),
            vec!["call_a", "call_b", "call_c", "call_d"]
        );
    }

    /// A handler that inherits the conservative `false` default serializes calls
    /// to its own tool name. This does not prevent different tool names from
    /// overlapping elsewhere in the same round.
    struct ExclusiveSlowExecutor;

    impl ToolHandler for ExclusiveSlowExecutor {
        fn tool_type(&self) -> ToolType {
            ToolType::WebSearch
        }
        fn validate(&self, _param: &Value) -> Result<(), ToolError> {
            Ok(())
        }
        fn normalize(&self, _param: &Value) -> Vec<FunctionTool> {
            Vec::new()
        }
    }

    impl GatewayExecutor for ExclusiveSlowExecutor {
        fn execute(
            &self,
            call_id: &str,
            _tool_name: &str,
            _arguments: &str,
            _config: &Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            let call_id = call_id.to_owned();
            Box::pin(async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(ToolOutput {
                    call_id,
                    output: "unreachable".to_owned(),
                })
            })
        }
    }

    #[tokio::test]
    async fn gateway_round_serializes_non_parallel_safe_same_tool_calls() {
        let web_search: ResponsesTool =
            serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param");
        let mut executors = GatewayExecutors::default();
        executors.insert(Arc::new(ExclusiveSlowExecutor));
        let mut tools = [web_search];
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry builds");

        let output_items: Vec<OutputItem> = ["call_a", "call_b", "call_c", "call_d"]
            .into_iter()
            .map(|call_id| OutputItem::FunctionCall(web_search_call(call_id)))
            .collect();

        let started = std::time::Instant::now();
        let results = GatewayRound::new()
            .execute(&output_items, &registry)
            .await
            .expect("all calls execute");
        let elapsed = started.elapsed();

        assert_eq!(results.len(), 4);
        // Four 50ms calls run sequentially (one at a time) should take close to
        // 200ms -- well above what four *concurrent* 50ms calls would take.
        assert!(
            elapsed >= std::time::Duration::from_millis(180),
            "four exclusive 50ms calls took {elapsed:?}, expected close to 200ms"
        );
    }

    #[test]
    fn mcp_list_tools_uses_shared_gateway_event_lifecycle() {
        let list_tools = McpListTools::new(
            "mcpl_1",
            "counter",
            vec![McpListTool::new(
                "increment",
                Some("Increment the counter".to_owned()),
                serde_json::json!({"type": "object", "properties": {}}),
                Some(serde_json::json!({"read_only": false})),
            )],
        );
        let discovered_output = crate::tool::mcp::handler::list_tools_output_item(&list_tools);
        let public_output = super::public_output_items(&[discovered_output], &ToolRegistry::default(), &[]);
        let plans = super::mcp_list_tools_event_plans(&public_output, 0);
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut stream_accumulator = crate::executor::gateway_accumulator::GatewayStreamAccumulator::new();
        stream_accumulator
            .process_sse_line(r#"data: {"type":"response.created"}"#, 0)
            .expect("response.created");
        stream_accumulator
            .process_sse_line(r#"data: {"type":"response.in_progress"}"#, 0)
            .expect("response.in_progress");

        super::emit_gateway_start_events(&plans, &mut stream_accumulator, &sender).expect("start events");
        super::emit_gateway_completed_events(&public_output, &plans, &mut stream_accumulator, &sender)
            .expect("completed events");

        let events = std::iter::from_fn(|| receiver.try_recv().ok())
            .map(|event| parse_named_sse_event(&event.content))
            .collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "response.output_item.added",
                "response.mcp_list_tools.in_progress",
                "response.mcp_list_tools.completed",
                "response.output_item.done",
            ]
        );
        assert_eq!(
            events
                .iter()
                .map(|event| event["sequence_number"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![2, 3, 4, 5]
        );
        assert_eq!(events[0]["item"]["tools"], serde_json::json!([]));
        assert_eq!(events[3]["item"]["tools"][0]["name"], "increment");
    }

    #[test]
    fn compaction_uses_shared_gateway_event_lifecycle_without_intermediate_event() {
        let public_output = [OutputItem::Compaction(CompactionItem {
            id: Some("cmp_1".to_owned()),
            encrypted_content: "durable summary".to_owned(),
        })];
        let plans = super::compaction_event_plans(&public_output, 0);
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut stream_accumulator = crate::executor::gateway_accumulator::GatewayStreamAccumulator::new();

        super::emit_gateway_start_events(&plans, &mut stream_accumulator, &sender).expect("start events");
        super::emit_gateway_completed_events(&public_output, &plans, &mut stream_accumulator, &sender)
            .expect("completed events");

        let chunks = std::iter::from_fn(|| receiver.try_recv().ok())
            .map(|event| event.content)
            .collect::<Vec<_>>();
        let events = chunks
            .iter()
            .map(|chunk| parse_named_sse_event(chunk))
            .collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["response.output_item.added", "response.output_item.done"]
        );
        assert_eq!(events[0]["item"], events[1]["item"]);
        assert_eq!(events[1]["item"]["encrypted_content"], "durable summary");

        let data_lines = chunks
            .iter()
            .filter_map(|chunk| chunk.lines().find(|line| line.starts_with("data: ")).map(str::to_owned));
        let response = ResponseAccumulator::from_sse_lines(data_lines, None).finalize("test-model", None, None);
        assert_eq!(response.output.len(), 1);
        assert!(matches!(response.output[0], OutputItem::Compaction(_)));
    }

    #[test]
    fn mcp_gateway_events_follow_openai_lifecycle() {
        let call = FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "mcp__counter__increment".to_owned(),
            arguments: "{}".to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let started = OutputItem::McpCall(crate::types::io::McpCall::new(
            "mcp_1",
            "counter",
            "increment",
            "",
            McpCallStatus::InProgress,
            None,
            None,
        ));
        let mut plans = vec![super::GatewayEventPlan {
            output_index: 0,
            started_output: Some(started),
            completed_output: None,
            arguments: Some(call.arguments.clone()),
        }];
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut stream_accumulator = crate::executor::gateway_accumulator::GatewayStreamAccumulator::new();

        super::emit_gateway_start_events(&plans, &mut stream_accumulator, &sender).expect("start events");

        let mut start_events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            start_events.push(parse_named_sse_event(&event.content));
        }
        assert_eq!(
            start_events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "response.output_item.added",
                "response.mcp_call.in_progress",
                "response.mcp_call_arguments.delta",
                "response.mcp_call_arguments.done"
            ]
        );
        assert_eq!(start_events[0]["item"]["type"], "mcp_call");
        assert_eq!(start_events[0]["item"]["arguments"], "");
        assert_eq!(start_events[2]["delta"], "{}");
        assert_eq!(start_events[3]["arguments"], "{}");
        assert_eq!(
            start_events
                .iter()
                .map(|event| event["sequence_number"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );

        let final_item = OutputItem::McpCall(crate::types::io::McpCall::new(
            "mcp_1",
            "counter",
            "increment",
            "{}",
            McpCallStatus::Completed,
            Some("1".to_owned()),
            None,
        ));
        let results = vec![GatewayCallResult {
            call,
            input_item: InputItem::FunctionCallOutput(
                ToolOutput {
                    call_id: "call_1".to_owned(),
                    output: "1".to_owned(),
                }
                .into(),
            ),
            public_output: Some(final_item),
        }];

        super::complete_gateway_event_plans(&mut plans, &results);
        super::emit_gateway_completed_events(&results, &plans, &mut stream_accumulator, &sender)
            .expect("completed events");

        let completed = receiver.try_recv().expect("mcp_call.completed");
        let completed = parse_named_sse_event(&completed.content);
        assert_eq!(completed["type"], "response.mcp_call.completed");
        assert_eq!(completed["sequence_number"], 4);
        assert!(completed.get("item").is_none());

        let done = receiver.try_recv().expect("output_item.done");
        let done = parse_named_sse_event(&done.content);
        assert_eq!(done["type"], "response.output_item.done");
        assert_eq!(done["sequence_number"], 5);
        assert_eq!(done["item"]["type"], "mcp_call");
        assert_eq!(done["item"]["output"], "1");
    }

    #[test]
    fn failed_mcp_gateway_events_keep_contiguous_sequence_numbers() {
        let call = FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "mcp__counter__increment".to_owned(),
            arguments: "{}".to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let mut plans = vec![super::GatewayEventPlan {
            output_index: 0,
            started_output: Some(OutputItem::McpCall(crate::types::io::McpCall::new(
                "mcp_1",
                "counter",
                "increment",
                "",
                McpCallStatus::InProgress,
                None,
                None,
            ))),
            completed_output: None,
            arguments: Some(call.arguments.clone()),
        }];
        let results = vec![GatewayCallResult {
            call,
            input_item: InputItem::FunctionCallOutput(
                ToolOutput {
                    call_id: "call_1".to_owned(),
                    output: r#"{"error":"boom"}"#.to_owned(),
                }
                .into(),
            ),
            public_output: Some(OutputItem::McpCall(crate::types::io::McpCall::new(
                "mcp_1",
                "counter",
                "increment",
                "{}",
                McpCallStatus::Failed,
                None,
                Some(crate::types::io::McpCallError::tool_execution("boom")),
            ))),
        }];
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut stream_accumulator = crate::executor::gateway_accumulator::GatewayStreamAccumulator::new();

        super::emit_gateway_start_events(&plans, &mut stream_accumulator, &sender).expect("start events");
        super::complete_gateway_event_plans(&mut plans, &results);
        super::emit_gateway_completed_events(&results, &plans, &mut stream_accumulator, &sender)
            .expect("failed events");

        let events = std::iter::from_fn(|| receiver.try_recv().ok())
            .map(|event| parse_named_sse_event(&event.content))
            .collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "response.output_item.added",
                "response.mcp_call.in_progress",
                "response.mcp_call_arguments.delta",
                "response.mcp_call_arguments.done",
                "response.mcp_call.failed",
                "response.output_item.done",
            ]
        );
        assert_eq!(
            events
                .iter()
                .map(|event| event["sequence_number"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
    }
}
