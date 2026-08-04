use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::StreamExt;
use serde_json::Value;

use crate::events::{EventFrame, EventPayload, SSEEventType, SSEItemType, WireEvent};
use crate::executor::accumulator::ResponseAccumulator;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::gateway_accumulator::{GatewayStreamAccumulator, StreamEvent, emit_sse_frame, synthetic_event};
use crate::executor::inference::{call_inference, fetch_response_json};
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::tool::ToolRegistry;
use crate::types::request_response::ResponsePayload;
use crate::utils::common::serialize_to_string;

struct StreamEmitContext<'a> {
    request: &'a RequestContext,
    registry: &'a ToolRegistry,
    sender: &'a tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    accumulator: &'a mut GatewayStreamAccumulator,
    output_offset: usize,
}

pub(super) struct StreamPayload {
    pub(super) payload: ResponsePayload,
    pub(super) deferred_events: Vec<EventFrame>,
}

pub(super) async fn fetch_blocking_payload(
    ctx: &RequestContext,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
) -> ExecutorResult<ResponsePayload> {
    let url = exec_ctx.responses_url();
    // Non-streaming request: stream=false -> full JSON body -> from_json.
    let upstream_request = ctx.enriched_request.to_upstream_request(false)?;
    let upstream_json = serialize_to_string(&upstream_request).map_err(ExecutorError::JsonError)?;

    let body = fetch_response_json(upstream_json, &url, &exec_ctx.client, auth).await?;

    let acc = ResponseAccumulator::from_json(&body, ctx.conversation_id.as_deref())?;
    let mut payload = acc.finalize(
        &ctx.enriched_request.model,
        ctx.original_request.previous_response_id.as_deref(),
        ctx.original_request.instructions.as_deref(),
    );
    ctx.inject_ids(&mut payload);

    Ok(payload)
}

pub(super) async fn fetch_stream_payload(
    ctx: &RequestContext,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
    registry: &ToolRegistry,
    mut stream: Option<(
        &mut GatewayStreamAccumulator,
        &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    )>,
    output_offset: usize,
) -> ExecutorResult<StreamPayload> {
    let url = exec_ctx.responses_url();
    let upstream_request = ctx.enriched_request.to_upstream_request(true)?;
    let upstream_json = serialize_to_string(&upstream_request).map_err(ExecutorError::JsonError)?;
    let mut line_stream = Box::pin(call_inference(
        upstream_json,
        url,
        Arc::clone(&exec_ctx.client),
        auth.map(str::to_owned),
        exec_ctx.streaming_timeout,
    ));
    let mut acc = ResponseAccumulator::new(ctx.response_id.clone(), ctx.conversation_id.clone());
    let mut hidden_gateway_item_ids = HashSet::new();
    let mut fallback_tool_search_item_ids = HashSet::new();
    let mut pending_unnamed_function_events = HashMap::<String, Vec<EventFrame>>::new();
    let mut defer_from_output_index = None;
    let mut deferred_events = Vec::new();
    while let Some(line_result) = line_stream.next().await {
        let line = line_result?;
        if let Some(frame) = acc.process_sse_line(&line) {
            log_upstream_failure(&frame, &ctx.response_id);
            if let Some((accumulator, sender)) = stream.as_mut() {
                let mut emit_ctx = StreamEmitContext {
                    request: ctx,
                    registry,
                    sender,
                    accumulator,
                    output_offset,
                };
                emit_upstream_stream_event(
                    frame,
                    &mut emit_ctx,
                    &mut hidden_gateway_item_ids,
                    &mut fallback_tool_search_item_ids,
                    &mut pending_unnamed_function_events,
                    &mut defer_from_output_index,
                    &mut deferred_events,
                )?;
            }
        }
    }
    acc.finish_stream();
    let mut payload = acc.finalize(
        &ctx.enriched_request.model,
        ctx.original_request.previous_response_id.as_deref(),
        ctx.original_request.instructions.as_deref(),
    );
    ctx.inject_ids(&mut payload);
    Ok(StreamPayload {
        payload,
        deferred_events,
    })
}

fn log_upstream_failure(frame: &EventFrame, gateway_response_id: &str) {
    if frame.event_type != SSEEventType::ResponseFailed {
        return;
    }

    let response = frame.wire.rest.get("response").unwrap_or(&Value::Null);
    let error = &response["error"];
    let error_code = error.get("code").and_then(Value::as_str).unwrap_or_default();
    let error_message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())
        .unwrap_or_default();
    let incomplete_reason = response["incomplete_details"]
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or_default();

    tracing::warn!(
        response_id = %gateway_response_id,
        upstream_response_id = response["id"].as_str().unwrap_or_default(),
        error_code,
        error_message,
        incomplete_reason,
        "upstream response failed"
    );
}

fn emit_upstream_stream_event(
    frame: EventFrame,
    emit_ctx: &mut StreamEmitContext<'_>,
    hidden_gateway_item_ids: &mut HashSet<String>,
    fallback_tool_search_item_ids: &mut HashSet<String>,
    pending_unnamed_function_events: &mut HashMap<String, Vec<EventFrame>>,
    defer_from_output_index: &mut Option<u64>,
    deferred_events: &mut Vec<EventFrame>,
) -> ExecutorResult<()> {
    if handle_fallback_tool_search_event(
        &frame,
        emit_ctx,
        fallback_tool_search_item_ids,
        pending_unnamed_function_events,
        *defer_from_output_index,
        deferred_events,
    )? {
        return Ok(());
    }
    defer_after_gateway_call(&frame, emit_ctx.registry, defer_from_output_index);
    if should_hide_upstream_event(
        frame.event_type,
        &frame.payload,
        emit_ctx.registry,
        hidden_gateway_item_ids,
    ) || is_terminal_response_event(frame.event_type)
    {
        drop_pending_function_events(&frame.payload, pending_unnamed_function_events);
        return Ok(());
    }
    let Some(frame) = defer_or_flush_function_event(
        frame,
        emit_ctx,
        hidden_gateway_item_ids,
        pending_unnamed_function_events,
        defer_from_output_index,
        deferred_events,
    )?
    else {
        return Ok(());
    };

    emit_or_defer_stream_frame(frame, emit_ctx, *defer_from_output_index, deferred_events)
}

fn handle_fallback_tool_search_event(
    frame: &EventFrame,
    emit_ctx: &mut StreamEmitContext<'_>,
    fallback_item_ids: &mut HashSet<String>,
    pending_unnamed_function_events: &mut HashMap<String, Vec<EventFrame>>,
    defer_from_output_index: Option<u64>,
    deferred_events: &mut Vec<EventFrame>,
) -> ExecutorResult<bool> {
    if !emit_ctx.registry.can_restore_tool_search_fallback() {
        return Ok(false);
    }

    match (&frame.event_type, &frame.payload) {
        (
            SSEEventType::OutputItemAdded,
            EventPayload::OutputItemAdded {
                item_id,
                item_type: SSEItemType::FunctionCall,
                name: Some(name),
                namespace: None,
                call_id: Some(call_id),
                ..
            },
        ) if name == crate::tool::TOOL_SEARCH_NAME && !call_id.is_empty() => {
            fallback_item_ids.insert(item_id.clone());
            Ok(false)
        }
        (
            SSEEventType::FunctionCallArgumentsDelta | SSEEventType::FunctionCallArgumentsDone,
            EventPayload::FunctionCallArgsDelta { item_id, .. } | EventPayload::FunctionCallArgsDone { item_id, .. },
        ) if fallback_item_ids.contains(item_id) => Ok(true),
        (
            SSEEventType::FunctionCallArgumentsDone,
            EventPayload::FunctionCallArgsDone {
                item_id,
                call_id: Some(call_id),
                name,
                output_index,
                ..
            },
        ) if name == crate::tool::TOOL_SEARCH_NAME
            && !call_id.is_empty()
            && pending_function_is_unqualified(item_id, pending_unnamed_function_events) =>
        {
            pending_unnamed_function_events.remove(item_id);
            fallback_item_ids.insert(item_id.clone());
            let added = fallback_tool_search_added_frame(call_id, *output_index)?;
            emit_or_defer_stream_frame(added, emit_ctx, defer_from_output_index, deferred_events)?;
            Ok(true)
        }
        (
            SSEEventType::OutputItemDone,
            EventPayload::OutputItemDone {
                item_id,
                item_type: SSEItemType::FunctionCall,
                item,
                output_index,
            },
        ) if is_unqualified_tool_search_function(item) => {
            if !fallback_item_ids.contains(item_id)
                && let Some(call_id) = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .filter(|call_id| !call_id.is_empty())
            {
                fallback_item_ids.insert(item_id.clone());
                let added = fallback_tool_search_added_frame(call_id, *output_index)?;
                emit_or_defer_stream_frame(added, emit_ctx, defer_from_output_index, deferred_events)?;
            }
            pending_unnamed_function_events.remove(item_id);
            Ok(false)
        }
        _ => Ok(false),
    }
}

fn pending_function_is_unqualified(
    item_id: &str,
    pending_unnamed_function_events: &HashMap<String, Vec<EventFrame>>,
) -> bool {
    pending_unnamed_function_events
        .get(item_id)
        .and_then(|events| events.first())
        .is_some_and(|frame| {
            matches!(
                frame.payload,
                EventPayload::OutputItemAdded {
                    item_type: SSEItemType::FunctionCall,
                    namespace: None,
                    ..
                }
            )
        })
}

fn is_unqualified_tool_search_function(item: &Value) -> bool {
    item.get("name").and_then(Value::as_str) == Some(crate::tool::TOOL_SEARCH_NAME)
        && item.get("namespace").and_then(Value::as_str).is_none()
}

fn fallback_tool_search_added_frame(call_id: &str, output_index: u32) -> ExecutorResult<EventFrame> {
    let mut frame = synthetic_event(
        SSEEventType::OutputItemAdded,
        [(
            "item".to_owned(),
            serde_json::json!({
                "type": "tool_search_call",
                "execution": "client",
                "call_id": call_id,
                "status": "in_progress",
                "arguments": {}
            }),
        )],
    )?;
    frame.wire.output_index = Some(u64::from(output_index));
    Ok(frame)
}

pub(super) fn emit_deferred_stream_events(
    deferred_events: Vec<EventFrame>,
    request: &RequestContext,
    registry: &ToolRegistry,
    accumulator: &mut GatewayStreamAccumulator,
    sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    output_offset: usize,
) -> ExecutorResult<()> {
    let mut emit_ctx = StreamEmitContext {
        request,
        registry,
        sender,
        accumulator,
        output_offset,
    };
    for mut frame in deferred_events {
        emit_stream_frame(&mut frame, &mut emit_ctx)?;
    }
    Ok(())
}

fn defer_after_gateway_call(frame: &EventFrame, registry: &ToolRegistry, defer_from_output_index: &mut Option<u64>) {
    let EventPayload::OutputItemAdded {
        item_type: SSEItemType::FunctionCall,
        name: Some(name),
        ..
    } = &frame.payload
    else {
        return;
    };
    if registry.is_gateway_owned_name(name) {
        record_first_hidden_gateway_output_index(frame, defer_from_output_index);
    }
}

fn record_first_hidden_gateway_output_index(frame: &EventFrame, defer_from_output_index: &mut Option<u64>) {
    let Some(output_index) = frame.wire.output_index else {
        return;
    };
    if defer_from_output_index.is_none_or(|first_hidden_index| output_index < first_hidden_index) {
        *defer_from_output_index = Some(output_index);
    }
}

fn should_defer_stream_event(frame: &EventFrame, defer_from_output_index: Option<u64>) -> bool {
    defer_from_output_index.is_some_and(|first_hidden_index| {
        frame
            .wire
            .output_index
            .is_some_and(|output_index| output_index >= first_hidden_index)
    })
}

fn emit_stream_frame(frame: &mut EventFrame, emit_ctx: &mut StreamEmitContext<'_>) -> ExecutorResult<()> {
    apply_context_response_ids(&mut frame.wire, emit_ctx.request);
    emit_ctx.registry.restore_stream_event_wire(&mut frame.wire);
    if emit_ctx.accumulator.process_event(frame, emit_ctx.output_offset) {
        emit_sse_frame(emit_ctx.sender, frame)?;
    }
    Ok(())
}

fn emit_or_defer_stream_frame(
    mut frame: EventFrame,
    emit_ctx: &mut StreamEmitContext<'_>,
    defer_from_output_index: Option<u64>,
    deferred_events: &mut Vec<EventFrame>,
) -> ExecutorResult<()> {
    if should_defer_stream_event(&frame, defer_from_output_index) {
        deferred_events.push(frame);
        return Ok(());
    }
    emit_stream_frame(&mut frame, emit_ctx)
}

fn defer_or_flush_function_event(
    frame: EventFrame,
    emit_ctx: &mut StreamEmitContext<'_>,
    hidden_gateway_item_ids: &mut HashSet<String>,
    pending_unnamed_function_events: &mut HashMap<String, Vec<EventFrame>>,
    defer_from_output_index: &mut Option<u64>,
    deferred_events: &mut Vec<EventFrame>,
) -> ExecutorResult<Option<EventFrame>> {
    match &frame.payload {
        EventPayload::OutputItemAdded {
            item_id,
            item_type,
            name: None,
            ..
        } if *item_type == SSEItemType::FunctionCall => {
            let item_id = item_id.clone();
            pending_unnamed_function_events.entry(item_id).or_default().push(frame);
            Ok(None)
        }
        EventPayload::FunctionCallArgsDelta { item_id, .. }
            if pending_unnamed_function_events.contains_key(item_id) =>
        {
            let item_id = item_id.clone();
            pending_unnamed_function_events.entry(item_id).or_default().push(frame);
            Ok(None)
        }
        EventPayload::FunctionCallArgsDone { item_id, name, .. } => {
            if emit_ctx.registry.is_gateway_owned_name(name) {
                hidden_gateway_item_ids.insert(item_id.clone());
                record_first_hidden_gateway_output_index(&frame, defer_from_output_index);
                pending_unnamed_function_events.remove(item_id);
                return Ok(None);
            }
            flush_pending_function_events(
                item_id,
                emit_ctx,
                pending_unnamed_function_events,
                *defer_from_output_index,
                deferred_events,
            )?;
            Ok(Some(frame))
        }
        EventPayload::OutputItemDone {
            item_id,
            item_type,
            item,
            ..
        } if *item_type == SSEItemType::FunctionCall => {
            if item
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| emit_ctx.registry.is_gateway_owned_name(name))
            {
                hidden_gateway_item_ids.insert(item_id.clone());
                record_first_hidden_gateway_output_index(&frame, defer_from_output_index);
                pending_unnamed_function_events.remove(item_id);
                return Ok(None);
            }
            flush_pending_function_events(
                item_id,
                emit_ctx,
                pending_unnamed_function_events,
                *defer_from_output_index,
                deferred_events,
            )?;
            Ok(Some(frame))
        }
        _ => Ok(Some(frame)),
    }
}

fn flush_pending_function_events(
    item_id: &str,
    emit_ctx: &mut StreamEmitContext<'_>,
    pending_unnamed_function_events: &mut HashMap<String, Vec<EventFrame>>,
    defer_from_output_index: Option<u64>,
    deferred_events: &mut Vec<EventFrame>,
) -> ExecutorResult<()> {
    let Some(frames) = pending_unnamed_function_events.remove(item_id) else {
        return Ok(());
    };
    for frame in frames {
        emit_or_defer_stream_frame(frame, emit_ctx, defer_from_output_index, deferred_events)?;
    }
    Ok(())
}

fn drop_pending_function_events(
    payload: &EventPayload,
    pending_unnamed_function_events: &mut HashMap<String, Vec<EventFrame>>,
) {
    match payload {
        EventPayload::OutputItemDone { item_id, .. }
        | EventPayload::FunctionCallArgsDelta { item_id, .. }
        | EventPayload::FunctionCallArgsDone { item_id, .. } => {
            pending_unnamed_function_events.remove(item_id);
        }
        EventPayload::OutputItemAdded { .. }
        | EventPayload::TextDelta { .. }
        | EventPayload::TextDone { .. }
        | EventPayload::CustomToolCallInputDelta { .. }
        | EventPayload::CustomToolCallInputDone { .. }
        | EventPayload::ReasoningDelta { .. }
        | EventPayload::ReasoningDone { .. }
        | EventPayload::Response { .. }
        | EventPayload::Raw(_)
        | EventPayload::None => {}
    }
}

fn should_hide_upstream_event(
    event_type: SSEEventType,
    payload: &EventPayload,
    registry: &ToolRegistry,
    hidden_gateway_item_ids: &mut HashSet<String>,
) -> bool {
    match (event_type, payload) {
        (
            SSEEventType::OutputItemAdded,
            EventPayload::OutputItemAdded {
                item_id,
                item_type,
                name: Some(name),
                ..
            },
        ) if *item_type == SSEItemType::FunctionCall && registry.is_gateway_owned_name(name) => {
            hidden_gateway_item_ids.insert(item_id.clone());
            true
        }
        (SSEEventType::OutputItemDone, EventPayload::OutputItemDone { item_id, item_type, .. })
            if *item_type == SSEItemType::FunctionCall && hidden_gateway_item_ids.contains(item_id) =>
        {
            true
        }
        (
            SSEEventType::FunctionCallArgumentsDelta | SSEEventType::FunctionCallArgumentsDone,
            EventPayload::FunctionCallArgsDelta { item_id, .. } | EventPayload::FunctionCallArgsDone { item_id, .. },
        ) => hidden_gateway_item_ids.contains(item_id),
        _ => false,
    }
}

fn is_terminal_response_event(event_type: SSEEventType) -> bool {
    matches!(
        event_type,
        SSEEventType::ResponseCompleted | SSEEventType::ResponseFailed | SSEEventType::ResponseIncomplete
    )
}

fn apply_context_response_ids(wire: &mut WireEvent, ctx: &RequestContext) {
    let Some(response) = wire.rest.get_mut("response").and_then(Value::as_object_mut) else {
        return;
    };
    response.insert("id".to_owned(), Value::String(ctx.response_id.clone()));
    if let Some(previous_response_id) = &ctx.original_request.previous_response_id {
        response.insert(
            "previous_response_id".to_owned(),
            Value::String(previous_response_id.clone()),
        );
    }
    if let Some(conversation_id) = &ctx.conversation_id {
        response.insert("conversation_id".to_owned(), Value::String(conversation_id.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::normalize_sse_line;
    use crate::tool::GatewayExecutors;
    use crate::types::request_response::RequestPayload;

    fn emitted_event(receiver: &mut tokio::sync::mpsc::UnboundedReceiver<StreamEvent>) -> Value {
        let event = receiver.try_recv().expect("emitted SSE event");
        let data = event
            .content
            .strip_prefix("data: ")
            .and_then(|line| line.strip_suffix("\n\n"))
            .expect("SSE data framing");
        let value: Value = serde_json::from_str(data).expect("valid emitted JSON");
        assert_eq!(value["sequence_number"], event.sequence_number);
        value
    }

    async fn client_tool_search_fixture() -> (RequestContext, ToolRegistry) {
        let mut request: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "find a tool",
            "tools": [{
                "type": "tool_search",
                "execution": "client",
                "description": "Search deferred tools",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }]
        }))
        .expect("valid request");
        let mut executors = GatewayExecutors::default();
        let registry =
            ToolRegistry::build_with_handlers(request.tools.as_deref_mut().expect("declared tools"), &mut executors)
                .await
                .expect("valid registry");
        let context = RequestContext {
            original_request: request.clone(),
            enriched_request: request,
            new_input_items: Vec::new(),
            response_id: "resp_gateway".to_owned(),
            conversation_id: None,
            conversation_version: None,
        };
        (context, registry)
    }

    #[tokio::test]
    async fn fallback_stream_emits_only_canonical_tool_search_lifecycle() {
        let (context, registry) = client_tool_search_fixture().await;

        let cases = [
            [
                r#"data: {"type":"response.output_item.added","output_index":2,"item":{"id":"fc_search","type":"function_call","call_id":"call_search","name":"tool_search","status":"in_progress","arguments":""}}"#,
                r#"data: {"type":"response.function_call_arguments.delta","item_id":"fc_search","output_index":2,"call_id":"call_search","delta":"{\"query\":\"shell\"}"}"#,
                r#"data: {"type":"response.function_call_arguments.done","item_id":"fc_search","output_index":2,"call_id":"call_search","name":"tool_search","arguments":"{\"query\":\"shell\"}"}"#,
                r#"data: {"type":"response.output_item.done","output_index":2,"item":{"id":"fc_search","type":"function_call","call_id":"call_search","name":"tool_search","status":"completed","arguments":"{\"query\":\"shell\"}"}}"#,
            ],
            [
                r#"data: {"type":"response.output_item.added","output_index":2,"item":{"id":"fc_search","type":"function_call","call_id":"call_search"}}"#,
                r#"data: {"type":"response.function_call_arguments.delta","item_id":"fc_search","output_index":2,"call_id":"call_search","delta":"{\"query\":"}"#,
                r#"data: {"type":"response.function_call_arguments.done","item_id":"fc_search","output_index":2,"call_id":"call_search","name":"tool_search","arguments":"{\"query\":\"shell\"}"}"#,
                r#"data: {"type":"response.output_item.done","output_index":2,"item":{"id":"fc_search","type":"function_call","call_id":"call_search","name":"tool_search","status":"completed","arguments":"{\"query\":\"shell\"}"}}"#,
            ],
        ];

        for lines in cases {
            let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
            let mut accumulator = GatewayStreamAccumulator::new();
            let mut emit_context = StreamEmitContext {
                request: &context,
                registry: &registry,
                sender: &sender,
                accumulator: &mut accumulator,
                output_offset: 4,
            };
            let mut hidden_ids = HashSet::new();
            let mut fallback_ids = HashSet::new();
            let mut pending = HashMap::new();
            let mut defer_from_output_index = None;
            let mut deferred = Vec::new();

            for line in lines {
                let frame = normalize_sse_line(line).expect("valid upstream SSE event");
                emit_upstream_stream_event(
                    frame,
                    &mut emit_context,
                    &mut hidden_ids,
                    &mut fallback_ids,
                    &mut pending,
                    &mut defer_from_output_index,
                    &mut deferred,
                )
                .expect("event emission succeeds");
            }

            assert!(deferred.is_empty());
            let added = emitted_event(&mut receiver);
            assert_eq!(added["type"], "response.output_item.added");
            assert_eq!(added["sequence_number"], 0);
            assert_eq!(added["output_index"], 6);
            assert_eq!(added["item"]["type"], "tool_search_call");
            assert_eq!(added["item"]["execution"], "client");
            assert_eq!(added["item"]["call_id"], "call_search");
            assert_eq!(added["item"]["status"], "in_progress");
            assert_eq!(added["item"]["arguments"], serde_json::json!({}));
            assert!(added["item"].get("id").is_none());
            assert!(added["item"].get("name").is_none());

            let done = emitted_event(&mut receiver);
            assert_eq!(done["type"], "response.output_item.done");
            assert_eq!(done["sequence_number"], 1);
            assert_eq!(done["output_index"], 6);
            assert_eq!(done["item"]["type"], "tool_search_call");
            assert_eq!(done["item"]["execution"], "client");
            assert_eq!(done["item"]["call_id"], "call_search");
            assert_eq!(done["item"]["status"], "completed");
            assert_eq!(done["item"]["arguments"]["query"], "shell");
            assert!(done["item"].get("id").is_none());
            assert!(done["item"].get("name").is_none());
            assert!(receiver.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn malformed_known_tool_search_without_call_id_passes_through() {
        let (context, registry) = client_tool_search_fixture().await;
        let cases = [
            [
                r#"data: {"type":"response.output_item.added","output_index":2,"item":{"id":"fc_search","type":"function_call","name":"tool_search","status":"in_progress","arguments":""}}"#,
                r#"data: {"type":"response.function_call_arguments.delta","item_id":"fc_search","output_index":2,"delta":"{\"query\":"}"#,
                r#"data: {"type":"response.function_call_arguments.done","item_id":"fc_search","output_index":2,"name":"tool_search","arguments":"{\"query\":\"shell\"}"}"#,
                r#"data: {"type":"response.output_item.done","output_index":2,"item":{"id":"fc_search","type":"function_call","name":"tool_search","status":"completed","arguments":"{\"query\":\"shell\"}"}}"#,
            ],
            [
                r#"data: {"type":"response.output_item.added","output_index":2,"item":{"id":"fc_search","type":"function_call","call_id":"","name":"tool_search","status":"in_progress","arguments":""}}"#,
                r#"data: {"type":"response.function_call_arguments.delta","item_id":"fc_search","output_index":2,"call_id":"","delta":"{\"query\":"}"#,
                r#"data: {"type":"response.function_call_arguments.done","item_id":"fc_search","output_index":2,"call_id":"","name":"tool_search","arguments":"{\"query\":\"shell\"}"}"#,
                r#"data: {"type":"response.output_item.done","output_index":2,"item":{"id":"fc_search","type":"function_call","call_id":"","name":"tool_search","status":"completed","arguments":"{\"query\":\"shell\"}"}}"#,
            ],
        ];

        for lines in cases {
            let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
            let mut accumulator = GatewayStreamAccumulator::new();
            let mut emit_context = StreamEmitContext {
                request: &context,
                registry: &registry,
                sender: &sender,
                accumulator: &mut accumulator,
                output_offset: 0,
            };
            let mut hidden_ids = HashSet::new();
            let mut fallback_ids = HashSet::new();
            let mut pending = HashMap::new();
            let mut defer_from_output_index = None;
            let mut deferred = Vec::new();

            for line in lines {
                let frame = normalize_sse_line(line).expect("valid upstream SSE event");
                emit_upstream_stream_event(
                    frame,
                    &mut emit_context,
                    &mut hidden_ids,
                    &mut fallback_ids,
                    &mut pending,
                    &mut defer_from_output_index,
                    &mut deferred,
                )
                .expect("event emission succeeds");
            }

            let emitted = std::array::from_fn::<_, 4, _>(|_| emitted_event(&mut receiver));
            assert_eq!(emitted[0]["type"], "response.output_item.added");
            assert_eq!(emitted[0]["item"]["type"], "function_call");
            assert_eq!(emitted[1]["type"], "response.function_call_arguments.delta");
            assert_eq!(emitted[2]["type"], "response.function_call_arguments.done");
            assert_eq!(emitted[3]["type"], "response.output_item.done");
            assert_eq!(emitted[3]["item"]["type"], "function_call");
            assert!(receiver.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn done_only_tool_search_synthesizes_one_added_event() {
        let (context, registry) = client_tool_search_fixture().await;
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut accumulator = GatewayStreamAccumulator::new();
        let mut emit_context = StreamEmitContext {
            request: &context,
            registry: &registry,
            sender: &sender,
            accumulator: &mut accumulator,
            output_offset: 3,
        };
        let mut hidden_ids = HashSet::new();
        let mut fallback_ids = HashSet::new();
        let mut pending = HashMap::new();
        let mut defer_from_output_index = None;
        let mut deferred = Vec::new();
        let line = r#"data: {"type":"response.output_item.done","output_index":2,"item":{"id":"fc_search","type":"function_call","call_id":"call_search","name":"tool_search","status":"completed","arguments":"{\"query\":\"shell\"}"}}"#;

        for _ in 0..2 {
            let frame = normalize_sse_line(line).expect("valid upstream SSE event");
            emit_upstream_stream_event(
                frame,
                &mut emit_context,
                &mut hidden_ids,
                &mut fallback_ids,
                &mut pending,
                &mut defer_from_output_index,
                &mut deferred,
            )
            .expect("event emission succeeds");
        }

        let added = emitted_event(&mut receiver);
        assert_eq!(added["type"], "response.output_item.added");
        assert_eq!(added["output_index"], 5);
        assert_eq!(added["item"]["type"], "tool_search_call");
        assert_eq!(added["item"]["call_id"], "call_search");
        for expected_sequence in [1, 2] {
            let done = emitted_event(&mut receiver);
            assert_eq!(done["type"], "response.output_item.done");
            assert_eq!(done["sequence_number"], expected_sequence);
            assert_eq!(done["output_index"], 5);
            assert_eq!(done["item"]["type"], "tool_search_call");
        }
        assert!(receiver.try_recv().is_err());
    }
}
