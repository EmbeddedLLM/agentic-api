use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::events::{EventFrame, EventPayload, SSEEventType, SSEItemType};
use crate::executor::accumulator::AccumulatedFunctionCall;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::gateway_accumulator::synthetic_event;
use crate::tool::{ToolType, search};
use crate::types::io::OutputItem;
use crate::utils::common::{serialize_to_string, serialize_to_value};

const MAX_PENDING_FUNCTION_BYTES: usize = 256 * 1024;
const MAX_PENDING_FUNCTION_CALLS: usize = 128;

#[derive(Debug)]
enum FunctionCallShape {
    PublicFunction,
    GatewayOwned,
    Custom(CustomCallState),
    ToolSearch(ToolSearchCallState),
}

#[derive(Debug)]
struct ToolSearchCallState {
    upstream_item_id: String,
    public_item_id: String,
    call_id: String,
    output_index: u32,
    accounted_argument_bytes: usize,
    arguments_done: bool,
}

#[derive(Debug)]
struct CustomCallState {
    public_item_id: String,
    output_index: u32,
    emitted_input: String,
    input_start: Option<usize>,
    input_cursor: usize,
    input_done: bool,
}

#[derive(Debug, Default)]
struct PendingFunctionCall {
    output_index: u32,
    frames: Vec<EventFrame>,
    bytes: usize,
}

#[derive(Debug, Default)]
pub(super) struct FunctionSseTranslation {
    pub(super) frames: Vec<EventFrame>,
    pub(super) defer_from_output_index: Option<u32>,
}

/// Restores normalized upstream function-call SSE to the public call shape.
/// Tool routing remains outside this type; it receives only the request's
/// model-visible name-to-type mapping.
#[derive(Debug, Default)]
pub(super) struct FunctionSseTranslator {
    tool_types: HashMap<String, ToolType>,
    active: HashMap<u32, FunctionCallShape>,
    pending_unnamed: HashMap<u32, PendingFunctionCall>,
    pending_bytes: usize,
    first_gateway_output_index: Option<u32>,
    search_call_seen: bool,
    tool_search_enabled: bool,
    withheld_function_names: HashSet<String>,
    upstream_terminal_failure: bool,
}

impl FunctionSseTranslator {
    pub(super) fn new(tool_types: HashMap<String, ToolType>) -> Self {
        let tool_search_enabled = tool_types.get("tool_search") == Some(&ToolType::ToolSearch);
        Self {
            tool_types,
            tool_search_enabled,
            ..Self::default()
        }
    }

    pub(super) fn with_withheld_function_names(mut self, names: &HashSet<String>) -> Self {
        self.withheld_function_names.clone_from(names);
        self
    }

    pub(super) fn translate(
        &mut self,
        frame: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        if matches!(
            frame.event_type,
            SSEEventType::ResponseFailed | SSEEventType::ResponseIncomplete
        ) {
            self.upstream_terminal_failure = true;
        }
        let mut translated = match &frame.payload {
            EventPayload::OutputItemAdded {
                item_id,
                item_type: SSEItemType::FunctionCall,
                output_index,
                name: Some(name),
                ..
            } => self.start_call(item_id, name, *output_index, Some(frame.clone()), call),
            EventPayload::OutputItemAdded {
                item_id: _,
                item_type: SSEItemType::FunctionCall,
                output_index,
                name: None,
                ..
            } => self.buffer_unnamed(*output_index, frame),
            EventPayload::FunctionCallArgsDelta {
                item_id, output_index, ..
            } => self.translate_delta(item_id, *output_index, frame.clone(), call),
            EventPayload::FunctionCallArgsDone {
                item_id,
                call_id,
                name,
                output_index,
                ..
            } => self.finish_arguments(item_id, name, *output_index, call_id.as_deref(), frame.clone(), call),
            EventPayload::OutputItemDone {
                item_id,
                item_type: SSEItemType::FunctionCall,
                output_index,
                item,
            } => {
                let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                self.finish_call(item_id, name, *output_index, frame.clone(), call)
            }
            _ => Ok(FunctionSseTranslation {
                frames: vec![frame],
                defer_from_output_index: None,
            }),
        }?;
        translated.defer_from_output_index = self.defer_from_output_index();
        Ok(translated)
    }

    pub(super) fn finish(&self) -> ExecutorResult<()> {
        if !self.upstream_terminal_failure
            && (self
                .active
                .values()
                .any(|shape| matches!(shape, FunctionCallShape::ToolSearch(_)))
                || (self.tool_search_enabled && !self.pending_unnamed.is_empty()))
        {
            return Err(search::invalid_upstream_search_call().into());
        }
        Ok(())
    }

    pub(super) fn unfinished_search_item_ids(&self) -> HashSet<&str> {
        let mut item_ids = self
            .active
            .values()
            .filter_map(|shape| match shape {
                FunctionCallShape::ToolSearch(state) => Some(state.upstream_item_id.as_str()),
                FunctionCallShape::PublicFunction | FunctionCallShape::GatewayOwned | FunctionCallShape::Custom(_) => {
                    None
                }
            })
            .collect::<HashSet<_>>();
        if self.tool_search_enabled {
            item_ids.extend(self.pending_unnamed.values().flat_map(|pending| {
                pending.frames.iter().filter_map(|frame| match &frame.payload {
                    EventPayload::OutputItemAdded { item_id, .. } if !item_id.is_empty() => Some(item_id.as_str()),
                    _ => None,
                })
            }));
        }
        item_ids
    }

    pub(super) fn validate_before_accumulation(
        &mut self,
        frame: &EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<()> {
        self.validate_withheld_function_names(frame)?;
        match &frame.payload {
            EventPayload::OutputItemAdded {
                item_type: SSEItemType::FunctionCall,
                output_index,
                name: None,
                ..
            } => self.validate_pending_frame_before_accumulation(*output_index, frame),
            EventPayload::FunctionCallArgsDone {
                arguments,
                item_id,
                name,
                output_index,
                call_id,
            } if self.tool_search_enabled
                && (name == "tool_search"
                    || matches!(self.active.get(output_index), Some(FunctionCallShape::ToolSearch(_)))) =>
            {
                validate_wire_output_index(frame, *output_index)?;
                if arguments.len() > MAX_PENDING_FUNCTION_BYTES {
                    return Err(search::invalid_upstream_search_call().into());
                }
                if let Some(FunctionCallShape::ToolSearch(state)) = self.active.get(output_index) {
                    validate_active_tool_search_done_name(frame)?;
                    validate_stream_linkage(state, item_id, call_id.as_deref())?;
                }
                Ok(())
            }
            EventPayload::OutputItemDone {
                item_type: SSEItemType::FunctionCall,
                output_index,
                item,
                ..
            } if self.tool_search_enabled
                && (item.get("name").and_then(Value::as_str) == Some("tool_search")
                    || matches!(self.active.get(output_index), Some(FunctionCallShape::ToolSearch(_)))) =>
            {
                validate_wire_output_index(frame, *output_index)?;
                let object = item.as_object().ok_or_else(search::invalid_upstream_search_call)?;
                let arguments = object
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or_else(search::invalid_upstream_search_call)?;
                if arguments.len() > MAX_PENDING_FUNCTION_BYTES {
                    return Err(search::invalid_upstream_search_call().into());
                }
                let public = search::public_output_item_from_raw(object)?;
                if let Some(FunctionCallShape::ToolSearch(state)) = self.active.get(output_index) {
                    let OutputItem::ToolSearchCall(public) = public else {
                        return Err(search::invalid_upstream_search_call().into());
                    };
                    if public.id != state.public_item_id || public.call_id != state.call_id {
                        return Err(search::invalid_upstream_search_call().into());
                    }
                }
                Ok(())
            }
            EventPayload::FunctionCallArgsDelta {
                delta,
                call_id,
                item_id,
                output_index,
            } => {
                let Some(shape) = self.active.get_mut(output_index) else {
                    return self.validate_pending_frame_before_accumulation(*output_index, frame);
                };
                match shape {
                    FunctionCallShape::Custom(_) => {
                        let current = call.map_or(0, |call| call.arguments().len());
                        ensure_function_call_size_for(current, delta.len())
                    }
                    FunctionCallShape::ToolSearch(state) => {
                        validate_wire_output_index(frame, *output_index)?;
                        validate_stream_linkage(state, item_id, call_id.as_deref())?;
                        if state.accounted_argument_bytes.saturating_add(delta.len()) > MAX_PENDING_FUNCTION_BYTES {
                            return Err(search::invalid_upstream_search_call().into());
                        }
                        state.accounted_argument_bytes = state.accounted_argument_bytes.saturating_add(delta.len());
                        Ok(())
                    }
                    FunctionCallShape::PublicFunction | FunctionCallShape::GatewayOwned => Ok(()),
                }
            }
            _ => Ok(()),
        }
    }

    fn validate_withheld_function_names(&self, frame: &EventFrame) -> ExecutorResult<()> {
        let terminal_has_withheld_call = frame.event_type == SSEEventType::ResponseCompleted
            && frame
                .wire
                .rest
                .get("response")
                .and_then(|response| response.get("output"))
                .and_then(Value::as_array)
                .is_some_and(|output| {
                    output.iter().any(|item| {
                        item.get("type").and_then(Value::as_str) == Some("function_call")
                            && item
                                .get("name")
                                .and_then(Value::as_str)
                                .is_some_and(|name| self.withheld_function_names.contains(name))
                    })
                });
        let lifecycle_name = match &frame.payload {
            EventPayload::OutputItemAdded {
                item_type: SSEItemType::FunctionCall,
                name: Some(name),
                ..
            }
            | EventPayload::FunctionCallArgsDone { name, .. } => Some(name.as_str()),
            EventPayload::OutputItemDone {
                item_type: SSEItemType::FunctionCall,
                item,
                ..
            } => item.get("name").and_then(Value::as_str),
            _ => None,
        };
        if terminal_has_withheld_call || lifecycle_name.is_some_and(|name| self.withheld_function_names.contains(name))
        {
            return Err(search::invalid_upstream_withheld_function_call().into());
        }
        Ok(())
    }

    fn validate_pending_frame_before_accumulation(&self, output_index: u32, frame: &EventFrame) -> ExecutorResult<()> {
        if !self.pending_unnamed.contains_key(&output_index) && self.pending_unnamed.len() >= MAX_PENDING_FUNCTION_CALLS
        {
            return Err(self.pending_limit_error(format!(
                "unnamed function-call SSE exceeded {MAX_PENDING_FUNCTION_CALLS} pending calls"
            )));
        }
        let bytes = serialize_to_string(&frame.wire)
            .map_err(ExecutorError::JsonError)?
            .len();
        if self.pending_bytes.saturating_add(bytes) > MAX_PENDING_FUNCTION_BYTES {
            return Err(self.pending_limit_error(format!(
                "unnamed function-call SSE exceeded {MAX_PENDING_FUNCTION_BYTES} buffered bytes"
            )));
        }
        Ok(())
    }

    fn pending_limit_error(&self, message: String) -> ExecutorError {
        if self.tool_search_enabled {
            search::invalid_upstream_search_call().into()
        } else {
            ExecutorError::StreamError(message)
        }
    }

    fn start_call(
        &mut self,
        item_id: &str,
        name: &str,
        output_index: u32,
        original: Option<EventFrame>,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        match self.tool_type(name) {
            ToolType::Custom => {
                let public_item_id = call.as_ref().map_or_else(
                    || crate::tool::custom::public_item_id(item_id),
                    |call| crate::tool::custom::public_item_id(&call.item.id),
                );
                self.active.insert(
                    output_index,
                    FunctionCallShape::Custom(CustomCallState {
                        public_item_id,
                        output_index,
                        emitted_input: String::new(),
                        input_start: None,
                        input_cursor: 0,
                        input_done: false,
                    }),
                );
                Ok(FunctionSseTranslation {
                    frames: call
                        .map(|call| custom_added_frame(&call))
                        .transpose()?
                        .into_iter()
                        .collect(),
                    defer_from_output_index: None,
                })
            }
            ToolType::Mcp | ToolType::WebSearch | ToolType::FileSearch | ToolType::CodeInterpreter => {
                if self.first_gateway_output_index.is_none_or(|first| output_index < first) {
                    self.first_gateway_output_index = Some(output_index);
                }
                self.active.insert(output_index, FunctionCallShape::GatewayOwned);
                Ok(FunctionSseTranslation::default())
            }
            ToolType::ToolSearch => {
                if self.search_call_seen
                    || self.active.contains_key(&output_index)
                    || self.pending_unnamed.contains_key(&output_index)
                {
                    return Err(search::invalid_upstream_search_call().into());
                }
                let call = call.ok_or_else(search::invalid_upstream_search_call)?;
                if call.item.namespace.is_some() {
                    return Err(search::invalid_upstream_search_call().into());
                }
                let original = original.as_ref().ok_or_else(search::invalid_upstream_search_call)?;
                validate_wire_output_index(original, output_index)?;
                let public_item = search::public_added_item(item_id, &call.item.call_id)?;
                let public_item_id = public_item["id"].as_str().unwrap_or_default().to_owned();
                self.search_call_seen = true;
                self.active.insert(
                    output_index,
                    FunctionCallShape::ToolSearch(ToolSearchCallState {
                        upstream_item_id: item_id.to_owned(),
                        public_item_id,
                        call_id: call.item.call_id.clone(),
                        output_index,
                        accounted_argument_bytes: call.arguments().len(),
                        arguments_done: false,
                    }),
                );
                Ok(FunctionSseTranslation {
                    frames: vec![tool_search_frame(
                        SSEEventType::OutputItemAdded,
                        output_index,
                        public_item,
                    )?],
                    defer_from_output_index: None,
                })
            }
            ToolType::Function | ToolType::CodexNamespace => {
                self.active.insert(output_index, FunctionCallShape::PublicFunction);
                Ok(FunctionSseTranslation {
                    frames: original.into_iter().collect(),
                    defer_from_output_index: None,
                })
            }
        }
    }

    fn translate_delta(
        &mut self,
        item_id: &str,
        output_index: u32,
        original: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        match self.active.get_mut(&output_index) {
            Some(FunctionCallShape::PublicFunction) => Ok(FunctionSseTranslation {
                frames: vec![original],
                defer_from_output_index: None,
            }),
            Some(FunctionCallShape::GatewayOwned) => Ok(FunctionSseTranslation::default()),
            Some(FunctionCallShape::Custom(state)) => {
                let frame = match call {
                    Some(call) => incremental_custom_delta(state, call.arguments())?,
                    None => None,
                };
                Ok(FunctionSseTranslation {
                    frames: frame.into_iter().collect(),
                    defer_from_output_index: None,
                })
            }
            Some(FunctionCallShape::ToolSearch(state)) => {
                let event_call_id = match &original.payload {
                    EventPayload::FunctionCallArgsDelta { call_id, .. } => call_id.as_deref(),
                    _ => None,
                };
                validate_stream_linkage(state, item_id, event_call_id)?;
                Ok(FunctionSseTranslation::default())
            }
            None => self.buffer_unnamed(output_index, original),
        }
    }

    fn finish_arguments(
        &mut self,
        item_id: &str,
        name: &str,
        output_index: u32,
        event_call_id: Option<&str>,
        original: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        let mut translated = self.resolve_pending(item_id, name, output_index, call)?;
        match self.active.get_mut(&output_index) {
            Some(FunctionCallShape::PublicFunction) | None => translated.frames.push(original),
            Some(FunctionCallShape::GatewayOwned) => {}
            Some(FunctionCallShape::Custom(state)) => {
                if let Some(call) = call {
                    translated.frames.extend(finish_custom_input(state, call.arguments())?);
                }
            }
            Some(FunctionCallShape::ToolSearch(state)) => {
                validate_active_tool_search_done_name(&original)?;
                validate_stream_linkage(state, item_id, event_call_id)?;
                let call = call.ok_or_else(search::invalid_upstream_search_call)?;
                validate_search_call_state(state, &call, item_id)?;
                let public = search::public_output_item(&call.item.id, &call.item.call_id, call.arguments())?;
                let OutputItem::ToolSearchCall(public) = public else {
                    return Err(search::invalid_upstream_search_call().into());
                };
                if public.id != state.public_item_id || public.call_id != state.call_id {
                    return Err(search::invalid_upstream_search_call().into());
                }
                state.arguments_done = true;
            }
        }
        Ok(translated)
    }

    fn finish_call(
        &mut self,
        item_id: &str,
        name: &str,
        output_index: u32,
        original: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        let mut translated = self.resolve_pending(item_id, name, output_index, call)?;
        match self.active.remove(&output_index) {
            Some(FunctionCallShape::PublicFunction) | None => translated.frames.push(original),
            Some(FunctionCallShape::GatewayOwned) => {}
            Some(FunctionCallShape::Custom(mut state)) => {
                if let Some(call) = call {
                    translated
                        .frames
                        .extend(finish_custom_input(&mut state, call.arguments())?);
                    translated.frames.push(custom_done_frame(&state, &call)?);
                }
            }
            Some(FunctionCallShape::ToolSearch(state)) => {
                let call = call.ok_or_else(search::invalid_upstream_search_call)?;
                validate_search_call_state(&state, &call, item_id)?;
                let object = original
                    .wire
                    .rest
                    .get("item")
                    .and_then(Value::as_object)
                    .ok_or_else(search::invalid_upstream_search_call)?;
                let public = search::public_output_item_from_raw(object)?;
                let OutputItem::ToolSearchCall(public_call) = &public else {
                    return Err(search::invalid_upstream_search_call().into());
                };
                if public_call.id != state.public_item_id
                    || public_call.call_id != state.call_id
                    || public_call.arguments != search_arguments(&call)?
                    || (!state.arguments_done && call.arguments().is_empty())
                {
                    return Err(search::invalid_upstream_search_call().into());
                }
                let item = serialize_to_value(&public).map_err(ExecutorError::JsonError)?;
                translated.frames.push(tool_search_frame(
                    SSEEventType::OutputItemDone,
                    state.output_index,
                    item,
                )?);
            }
        }
        Ok(translated)
    }

    fn resolve_pending(
        &mut self,
        item_id: &str,
        name: &str,
        output_index: u32,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        if self.active.contains_key(&output_index) {
            return Ok(FunctionSseTranslation::default());
        }

        let pending = self.take_pending(output_index);
        let added = pending.iter().filter(|frame| {
            matches!(
                frame.payload,
                EventPayload::OutputItemAdded {
                    item_type: SSEItemType::FunctionCall,
                    ..
                }
            )
        });
        let added = added.collect::<Vec<_>>();
        if self.tool_type(name) == ToolType::ToolSearch && added.len() != 1 {
            return Err(search::invalid_upstream_search_call().into());
        }
        let original_added = added.first().copied();
        let start_item_id = original_added.and_then(|frame| match &frame.payload {
            EventPayload::OutputItemAdded { item_id, .. } => Some(item_id.as_str()),
            _ => None,
        });
        let mut translated = self.start_call(
            start_item_id.unwrap_or(item_id),
            name,
            output_index,
            original_added.cloned(),
            call,
        )?;

        for frame in pending {
            if let EventPayload::FunctionCallArgsDelta {
                item_id, output_index, ..
            } = &frame.payload
            {
                let delta = self.translate_delta(item_id, *output_index, frame.clone(), call)?;
                translated.frames.extend(delta.frames);
            }
        }
        Ok(translated)
    }

    fn tool_type(&self, name: &str) -> ToolType {
        self.tool_types.get(name).copied().unwrap_or(ToolType::Function)
    }

    fn defer_from_output_index(&self) -> Option<u32> {
        self.first_gateway_output_index
            .into_iter()
            .chain(self.pending_unnamed.values().map(|pending| pending.output_index))
            .min()
    }

    fn buffer_unnamed(&mut self, output_index: u32, frame: EventFrame) -> ExecutorResult<FunctionSseTranslation> {
        let bytes = serialize_to_string(&frame.wire)
            .map_err(ExecutorError::JsonError)?
            .len();
        if self.pending_bytes.saturating_add(bytes) > MAX_PENDING_FUNCTION_BYTES {
            return Err(self.pending_limit_error(format!(
                "unnamed function-call SSE exceeded {MAX_PENDING_FUNCTION_BYTES} buffered bytes"
            )));
        }
        if !self.pending_unnamed.contains_key(&output_index) && self.pending_unnamed.len() >= MAX_PENDING_FUNCTION_CALLS
        {
            return Err(self.pending_limit_error(format!(
                "unnamed function-call SSE exceeded {MAX_PENDING_FUNCTION_CALLS} pending calls"
            )));
        }
        let pending = self
            .pending_unnamed
            .entry(output_index)
            .or_insert_with(|| PendingFunctionCall {
                output_index,
                ..PendingFunctionCall::default()
            });
        pending.frames.push(frame);
        pending.bytes = pending.bytes.saturating_add(bytes);
        self.pending_bytes = self.pending_bytes.saturating_add(bytes);
        Ok(FunctionSseTranslation::default())
    }

    fn take_pending(&mut self, output_index: u32) -> Vec<EventFrame> {
        let Some(pending) = self.pending_unnamed.remove(&output_index) else {
            return Vec::new();
        };
        self.pending_bytes = self.pending_bytes.saturating_sub(pending.bytes);
        pending.frames
    }
}

fn validate_search_call_state(
    state: &ToolSearchCallState,
    call: &AccumulatedFunctionCall<'_>,
    event_item_id: &str,
) -> ExecutorResult<()> {
    ensure_function_call_size(call.arguments())?;
    if call.output_index != state.output_index
        || call.item.call_id != state.call_id
        || call.item.id != state.upstream_item_id
        || event_item_id != state.upstream_item_id
    {
        return Err(search::invalid_upstream_search_call().into());
    }
    Ok(())
}

fn validate_stream_linkage(state: &ToolSearchCallState, item_id: &str, call_id: Option<&str>) -> ExecutorResult<()> {
    if item_id != state.upstream_item_id || call_id.is_some_and(|call_id| call_id != state.call_id) {
        return Err(search::invalid_upstream_search_call().into());
    }
    Ok(())
}

fn validate_active_tool_search_done_name(frame: &EventFrame) -> ExecutorResult<()> {
    if frame
        .wire
        .rest
        .get("name")
        .is_some_and(|name| name.as_str() != Some("tool_search"))
    {
        return Err(search::invalid_upstream_search_call().into());
    }
    Ok(())
}

fn validate_wire_output_index(frame: &EventFrame, output_index: u32) -> ExecutorResult<()> {
    if frame.wire.output_index != Some(u64::from(output_index)) {
        return Err(search::invalid_upstream_search_call().into());
    }
    Ok(())
}

fn search_arguments(call: &AccumulatedFunctionCall<'_>) -> ExecutorResult<serde_json::Map<String, Value>> {
    let OutputItem::ToolSearchCall(public) =
        search::public_output_item(&call.item.id, &call.item.call_id, call.arguments())?
    else {
        return Err(search::invalid_upstream_search_call().into());
    };
    Ok(public.arguments)
}

fn tool_search_frame(event_type: SSEEventType, output_index: u32, item: Value) -> ExecutorResult<EventFrame> {
    let mut frame = synthetic_event(event_type, [("item".to_owned(), item)])?;
    frame.wire.output_index = Some(u64::from(output_index));
    Ok(frame)
}

fn custom_added_frame(call: &AccumulatedFunctionCall<'_>) -> ExecutorResult<EventFrame> {
    custom_frame(
        SSEEventType::OutputItemAdded,
        call.output_index,
        [(
            "item".to_owned(),
            serde_json::json!({
                "id": crate::tool::custom::public_item_id(&call.item.id),
                "type": "custom_tool_call",
                "status": "in_progress",
                "call_id": call.item.call_id,
                "input": "",
                "name": call.item.name,
            }),
        )],
    )
}

fn incremental_custom_delta(state: &mut CustomCallState, arguments: &str) -> ExecutorResult<Option<EventFrame>> {
    ensure_function_call_size(arguments)?;
    let Some(delta) = partial_custom_input(state, arguments)? else {
        return Ok(None);
    };
    state.emitted_input.push_str(&delta);
    custom_frame(
        SSEEventType::CustomToolCallInputDelta,
        state.output_index,
        [
            ("delta".to_owned(), Value::String(delta)),
            ("item_id".to_owned(), Value::String(state.public_item_id.clone())),
        ],
    )
    .map(Some)
}

fn finish_custom_input(state: &mut CustomCallState, arguments: &str) -> ExecutorResult<Vec<EventFrame>> {
    if state.input_done {
        return Ok(Vec::new());
    }
    ensure_function_call_size(arguments)?;
    let input = crate::tool::custom::input_from_arguments(arguments);
    let Some(remaining) = input.strip_prefix(&state.emitted_input) else {
        return Err(ExecutorError::StreamError(
            "authoritative custom tool input contradicts streamed custom tool input".to_owned(),
        ));
    };
    let remaining = (!remaining.is_empty()).then(|| remaining.to_owned());
    state.emitted_input.clone_from(&input);
    state.input_done = true;

    let mut frames = Vec::with_capacity(2);
    if let Some(delta) = remaining {
        frames.push(custom_frame(
            SSEEventType::CustomToolCallInputDelta,
            state.output_index,
            [
                ("delta".to_owned(), Value::String(delta)),
                ("item_id".to_owned(), Value::String(state.public_item_id.clone())),
            ],
        )?);
    }
    frames.push(custom_frame(
        SSEEventType::CustomToolCallInputDone,
        state.output_index,
        [
            ("input".to_owned(), Value::String(input)),
            ("item_id".to_owned(), Value::String(state.public_item_id.clone())),
        ],
    )?);
    Ok(frames)
}

fn custom_done_frame(state: &CustomCallState, call: &AccumulatedFunctionCall<'_>) -> ExecutorResult<EventFrame> {
    custom_frame(
        SSEEventType::OutputItemDone,
        state.output_index,
        [(
            "item".to_owned(),
            serde_json::json!({
                "id": state.public_item_id,
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": call.item.call_id,
                "input": state.emitted_input,
                "name": call.item.name,
            }),
        )],
    )
}

fn custom_frame(
    event_type: SSEEventType,
    output_index: u32,
    fields: impl IntoIterator<Item = (String, Value)>,
) -> ExecutorResult<EventFrame> {
    let mut frame = synthetic_event(event_type, fields)?;
    frame.wire.output_index = Some(u64::from(output_index));
    Ok(frame)
}

fn ensure_function_call_size(arguments: &str) -> ExecutorResult<()> {
    if arguments.len() > MAX_PENDING_FUNCTION_BYTES {
        return Err(ExecutorError::StreamError(format!(
            "function-call SSE exceeded {MAX_PENDING_FUNCTION_BYTES} buffered bytes"
        )));
    }
    Ok(())
}

fn ensure_function_call_size_for(current: usize, additional: usize) -> ExecutorResult<()> {
    if current.saturating_add(additional) > MAX_PENDING_FUNCTION_BYTES {
        return Err(ExecutorError::StreamError(format!(
            "function-call SSE exceeded {MAX_PENDING_FUNCTION_BYTES} buffered bytes"
        )));
    }
    Ok(())
}

fn partial_custom_input(state: &mut CustomCallState, arguments: &str) -> ExecutorResult<Option<String>> {
    let input_start = if let Some(input_start) = state.input_start {
        input_start
    } else {
        let Some(input_start) = custom_input_start(arguments) else {
            return Ok(None);
        };
        state.input_start = Some(input_start);
        state.input_cursor = input_start;
        input_start
    };
    if state.input_cursor < input_start || state.input_cursor > arguments.len() {
        return Ok(None);
    }
    let encoded = &arguments[state.input_cursor..];
    let end = complete_json_string_prefix(encoded);
    if end == 0 {
        return Ok(None);
    }
    let candidate = format!("\"{}\"", &encoded[..end]);
    let delta = serde_json::from_str::<String>(&candidate)
        .map_err(|error| ExecutorError::StreamError(format!("invalid custom tool input string: {error}")))?;
    state.input_cursor = state.input_cursor.saturating_add(end);
    Ok((!delta.is_empty()).then_some(delta))
}

fn complete_json_string_prefix(value: &str) -> usize {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => return index,
            b'\\' => {
                let Some(escape) = bytes.get(index + 1) else {
                    return index;
                };
                if *escape == b'u' {
                    let unicode_end = index.saturating_add(6);
                    if unicode_end > bytes.len() {
                        return index;
                    }
                    let Some(code_unit) = json_hex_quad(&bytes[index + 2..unicode_end]) else {
                        index = unicode_end;
                        continue;
                    };
                    if (0xD800..=0xDBFF).contains(&code_unit) {
                        let pair_end = index.saturating_add(12);
                        if pair_end > bytes.len() {
                            return index;
                        }
                        index = pair_end;
                    } else {
                        index = unicode_end;
                    }
                } else {
                    index = index.saturating_add(2);
                }
            }
            _ => index = index.saturating_add(1),
        }
    }
    index
}

fn json_hex_quad(bytes: &[u8]) -> Option<u16> {
    if bytes.len() != 4 {
        return None;
    }
    bytes.iter().try_fold(0_u16, |value, byte| {
        let digit = byte.to_ascii_lowercase();
        let digit = match digit {
            b'0'..=b'9' => u16::from(digit - b'0'),
            b'a'..=b'f' => u16::from(digit - b'a' + 10),
            _ => return None,
        };
        value.checked_mul(16)?.checked_add(digit)
    })
}

fn custom_input_start(arguments: &str) -> Option<usize> {
    let original_len = arguments.len();
    let arguments = arguments.trim_start();
    let arguments = arguments.strip_prefix("{}").unwrap_or(arguments).trim_start();
    let encoded = arguments
        .strip_prefix('{')?
        .trim_start()
        .strip_prefix("\"input\"")?
        .trim_start()
        .strip_prefix(':')?
        .trim_start()
        .strip_prefix('"')?;
    Some(original_len.saturating_sub(encoded.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::accumulator::ResponseAccumulator;

    fn sse(value: &Value) -> String {
        format!("data: {value}")
    }

    fn translate(
        accumulator: &mut ResponseAccumulator,
        translator: &mut FunctionSseTranslator,
        value: &Value,
    ) -> FunctionSseTranslation {
        accumulator
            .process_sse_line_with_translator(&sse(value), translator)
            .expect("translation succeeds")
            .expect("SSE event")
    }

    fn search_event_sequence(item_id: &str, call_id: &str, arguments: &str) -> [Value; 5] {
        let split = arguments.len() / 2;
        let (first, second) = arguments.split_at(split);
        [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": item_id, "type": "function_call", "call_id": call_id,
                    "name": "tool_search", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": item_id, "call_id": call_id, "delta": first
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": item_id, "call_id": call_id, "delta": second
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.done", "output_index": 0,
                "item_id": item_id, "call_id": call_id, "name": "tool_search", "arguments": arguments
            }),
            serde_json::json!({
                "type": "response.output_item.done", "output_index": 0,
                "item": {"id": item_id, "type": "function_call", "call_id": call_id,
                    "name": "tool_search", "arguments": arguments, "status": "completed"}
            }),
        ]
    }

    #[test]
    fn tool_search_stream_emits_only_public_added_and_done_with_stable_identity() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([
            ("tool_search".to_owned(), ToolType::ToolSearch),
            ("weather".to_owned(), ToolType::Function),
        ]));
        let arguments = r#"{"query":"weather"}"#;
        let mut frames = Vec::new();

        for (index, event) in search_event_sequence("fc_search", "call_search", arguments)
            .into_iter()
            .enumerate()
        {
            frames.extend(translate(&mut accumulator, &mut translator, &event).frames);
            if index == 1 {
                let ordinary = serde_json::json!({
                    "type": "response.output_item.added", "output_index": 1,
                    "item": {"id": "fc_weather", "type": "function_call", "call_id": "call_weather",
                        "name": "weather", "arguments": "", "status": "in_progress"}
                });
                frames.extend(translate(&mut accumulator, &mut translator, &ordinary).frames);
            }
        }

        assert_eq!(
            frames.iter().map(|frame| frame.event_type).collect::<Vec<_>>(),
            [
                SSEEventType::OutputItemAdded,
                SSEEventType::OutputItemAdded,
                SSEEventType::OutputItemDone
            ]
        );
        let search_frames = frames
            .iter()
            .filter(|frame| frame.wire.output_index == Some(0))
            .collect::<Vec<_>>();
        assert_eq!(search_frames.len(), 2);
        assert_eq!(search_frames[0].wire.rest["item"]["type"], "tool_search_call");
        assert_eq!(search_frames[0].wire.rest["item"]["status"], "in_progress");
        assert_eq!(search_frames[0].wire.rest["item"]["arguments"], serde_json::json!({}));
        assert_eq!(search_frames[1].wire.rest["item"]["type"], "tool_search_call");
        assert_eq!(search_frames[1].wire.rest["item"]["status"], "completed");
        assert_eq!(
            search_frames[1].wire.rest["item"]["arguments"],
            serde_json::json!({"query": "weather"})
        );
        assert_eq!(search_frames[0].wire.rest["item"]["id"], "tsc_search");
        assert_eq!(
            search_frames[0].wire.rest["item"]["id"],
            search_frames[1].wire.rest["item"]["id"]
        );
        assert_eq!(search_frames[0].wire.rest["item"]["call_id"], "call_search");
        assert_eq!(
            search_frames[0].wire.rest["item"]["call_id"],
            search_frames[1].wire.rest["item"]["call_id"]
        );
        assert_eq!(frames[1].wire.rest["item"]["type"], "function_call");

        let blocking = search::public_output_item("fc_search", "call_search", arguments)
            .expect("blocking translation uses the same identity helper");
        let blocking = serialize_to_value(&blocking).expect("blocking item serializes");
        let replay: crate::types::io::InputItem =
            serde_json::from_value(blocking.clone()).expect("public item replays");
        let replay = serialize_to_value(&replay).expect("replay serializes");
        assert_eq!(blocking, search_frames[1].wire.rest["item"]);
        assert_eq!(replay, search_frames[1].wire.rest["item"]);
    }

    #[test]
    fn tool_search_stream_rejects_malformed_arguments_and_empty_call_id() {
        for (call_id, arguments) in [("call_search", "[1]"), ("call_search", "{"), ("", "{}")] {
            let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
            let mut translator =
                FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
            let mut error = None;
            for event in search_event_sequence("fc_search", call_id, arguments) {
                match accumulator.process_sse_line_with_translator(&sse(&event), &mut translator) {
                    Ok(_) => {}
                    Err(found) => {
                        error = Some(found);
                        break;
                    }
                }
            }
            assert!(
                error
                    .expect("invalid synthetic search stream must fail")
                    .to_string()
                    .contains("invalid tool-search call")
            );
        }
    }

    #[test]
    fn tool_search_stream_rejects_linkage_changes_second_call_and_premature_eof() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator =
            FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let added = &search_event_sequence("fc_search", "call_search", r#"{"query":"weather"}"#)[0];
        translate(&mut accumulator, &mut translator, added);

        let wrong_item = serde_json::json!({
            "type": "response.function_call_arguments.delta", "output_index": 0,
            "item_id": "fc_other", "delta": "{}"
        });
        assert!(
            accumulator
                .process_sse_line_with_translator(&sse(&wrong_item), &mut translator)
                .expect_err("changed item ID must fail")
                .to_string()
                .contains("invalid tool-search call")
        );
        assert!(
            translator.finish().is_err(),
            "an unfinished search call must fail at EOF"
        );

        let mut unnamed_accumulator = ResponseAccumulator::new("resp_unnamed".to_owned(), None);
        let mut unnamed_translator =
            FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        for event in [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "fc_expected", "type": "function_call", "call_id": "call_expected",
                    "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "fc_wrong", "delta": "{}"
            }),
        ] {
            unnamed_accumulator
                .process_sse_line_with_translator(&sse(&event), &mut unnamed_translator)
                .expect("unnamed frames buffer before resolution");
        }
        let resolving_done = serde_json::json!({
            "type": "response.function_call_arguments.done", "output_index": 0,
            "item_id": "fc_expected", "name": "tool_search", "arguments": "{}"
        });
        assert!(
            unnamed_accumulator
                .process_sse_line_with_translator(&sse(&resolving_done), &mut unnamed_translator)
                .expect_err("buffered delta item linkage must be validated")
                .to_string()
                .contains("invalid tool-search call")
        );

        let mut second_accumulator = ResponseAccumulator::new("resp_2".to_owned(), None);
        let mut second_translator =
            FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        for event in search_event_sequence("fc_first", "call_first", "{}") {
            translate(&mut second_accumulator, &mut second_translator, &event);
        }
        let second_added = serde_json::json!({
            "type": "response.output_item.added", "output_index": 1,
            "item": {"id": "fc_second", "type": "function_call", "call_id": "call_second",
                "name": "tool_search", "arguments": "", "status": "in_progress"}
        });
        assert!(
            second_accumulator
                .process_sse_line_with_translator(&sse(&second_added), &mut second_translator)
                .expect_err("a second synthetic search call must fail")
                .to_string()
                .contains("invalid tool-search call")
        );
    }

    #[test]
    fn tool_search_stream_accepts_authoritative_done_without_argument_deltas() {
        let mut accumulator = ResponseAccumulator::new("resp_done_only".to_owned(), None);
        let mut translator =
            FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let events = search_event_sequence("fc_search", "call_search", r#"{"query":"weather"}"#);

        let frames = [0, 3, 4]
            .into_iter()
            .flat_map(|index| translate(&mut accumulator, &mut translator, &events[index]).frames)
            .collect::<Vec<_>>();

        assert_eq!(
            frames.iter().map(|frame| frame.event_type).collect::<Vec<_>>(),
            [SSEEventType::OutputItemAdded, SSEEventType::OutputItemDone]
        );
        assert_eq!(
            frames[1].wire.rest["item"]["arguments"],
            serde_json::json!({"query": "weather"})
        );
        assert!(translator.finish().is_ok());
    }

    #[test]
    fn tool_search_stream_accepts_omitted_done_name_for_active_call() {
        let mut accumulator = ResponseAccumulator::new("resp_omitted_done_name".to_owned(), None);
        let mut translator =
            FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let arguments = r#"{"query": "add numbers"}"#;
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 1,
                "item": {"id": "fc_search", "type": "function_call", "call_id": "call_search",
                    "name": "tool_search", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 1,
                "item_id": "fc_search", "delta": "{}"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 1,
                "item_id": "fc_search", "delta": "{\"query\": \""
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 1,
                "item_id": "fc_search", "delta": "add numbers"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 1,
                "item_id": "fc_search", "delta": "\"}"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.done", "output_index": 1,
                "item_id": "fc_search", "arguments": arguments
            }),
            serde_json::json!({
                "type": "response.output_item.done", "output_index": 1,
                "item": {"id": "fc_search", "type": "function_call", "call_id": "call_search",
                    "name": "tool_search", "arguments": arguments, "status": "completed"}
            }),
        ];

        let frames = events
            .iter()
            .flat_map(|event| translate(&mut accumulator, &mut translator, event).frames)
            .collect::<Vec<_>>();

        assert_eq!(
            frames.iter().map(|frame| frame.event_type).collect::<Vec<_>>(),
            [SSEEventType::OutputItemAdded, SSEEventType::OutputItemDone]
        );
        assert_eq!(frames[0].wire.output_index, Some(1));
        assert_eq!(frames[1].wire.output_index, Some(1));
        assert_eq!(frames[0].wire.rest["item"]["id"], "tsc_search");
        assert_eq!(frames[0].wire.rest["item"]["call_id"], "call_search");
        assert_eq!(frames[0].wire.rest["item"]["execution"], "client");
        assert_eq!(frames[0].wire.rest["item"]["status"], "in_progress");
        assert_eq!(frames[1].wire.rest["item"]["id"], "tsc_search");
        assert_eq!(frames[1].wire.rest["item"]["call_id"], "call_search");
        assert_eq!(frames[1].wire.rest["item"]["execution"], "client");
        assert_eq!(frames[1].wire.rest["item"]["status"], "completed");
        assert_eq!(
            frames[1].wire.rest["item"]["arguments"],
            serde_json::json!({"query": "add numbers"})
        );
        assert!(translator.finish().is_ok());
    }

    #[test]
    fn tool_search_stream_rejects_authoritative_shape_and_linkage_mismatches() {
        let added = search_event_sequence("fc_search", "call_search", "{}")[0].clone();
        for (label, followup) in [
            (
                "wrong call id",
                serde_json::json!({
                    "type": "response.function_call_arguments.delta", "output_index": 0,
                    "item_id": "fc_search", "call_id": "call_other", "delta": "{}"
                }),
            ),
            (
                "missing output index",
                serde_json::json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_search", "call_id": "call_search", "delta": "{}"
                }),
            ),
            (
                "wrong done name",
                serde_json::json!({
                    "type": "response.function_call_arguments.done", "output_index": 0,
                    "item_id": "fc_search", "call_id": "call_search", "name": "weather",
                    "arguments": "{}"
                }),
            ),
            (
                "empty done name",
                serde_json::json!({
                    "type": "response.function_call_arguments.done", "output_index": 0,
                    "item_id": "fc_search", "call_id": "call_search", "name": "", "arguments": "{}"
                }),
            ),
            (
                "null done name",
                serde_json::json!({
                    "type": "response.function_call_arguments.done", "output_index": 0,
                    "item_id": "fc_search", "call_id": "call_search", "name": null, "arguments": "{}"
                }),
            ),
            (
                "non-string done name",
                serde_json::json!({
                    "type": "response.function_call_arguments.done", "output_index": 0,
                    "item_id": "fc_search", "call_id": "call_search", "name": 7, "arguments": "{}"
                }),
            ),
        ] {
            let mut accumulator = ResponseAccumulator::new(format!("resp_{label}"), None);
            let mut translator =
                FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
            translate(&mut accumulator, &mut translator, &added);

            let error = accumulator
                .process_sse_line_with_translator(&sse(&followup), &mut translator)
                .expect_err(label);
            assert!(error.is_invalid_upstream_tool_search(), "{label}: {error}");
        }

        for (label, invalid_added) in [
            (
                "namespace on added",
                serde_json::json!({
                    "type": "response.output_item.added", "output_index": 0,
                    "item": {"id": "fc_search", "type": "function_call", "call_id": "call_search",
                        "name": "tool_search", "namespace": "tools", "arguments": "", "status": "in_progress"}
                }),
            ),
            (
                "missing added output index",
                serde_json::json!({
                    "type": "response.output_item.added",
                    "item": {"id": "fc_search", "type": "function_call", "call_id": "call_search",
                        "name": "tool_search", "arguments": "", "status": "in_progress"}
                }),
            ),
        ] {
            let mut accumulator = ResponseAccumulator::new(format!("resp_{label}"), None);
            let mut translator =
                FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
            let error = accumulator
                .process_sse_line_with_translator(&sse(&invalid_added), &mut translator)
                .expect_err(label);
            assert!(error.is_invalid_upstream_tool_search(), "{label}: {error}");
        }

        let mut accumulator = ResponseAccumulator::new("resp_index_collision".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([
            ("tool_search".to_owned(), ToolType::ToolSearch),
            ("weather".to_owned(), ToolType::Function),
        ]));
        let ordinary = serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"id": "fc_weather", "type": "function_call", "call_id": "call_weather",
                "name": "weather", "arguments": "", "status": "in_progress"}
        });
        translate(&mut accumulator, &mut translator, &ordinary);
        let error = accumulator
            .process_sse_line_with_translator(&sse(&added), &mut translator)
            .expect_err("search must not overwrite an active output index");
        assert!(error.is_invalid_upstream_tool_search(), "{error}");
    }

    #[test]
    fn unfinished_search_ids_include_pending_candidates_with_other_loaded_tools_only_until_resolved() {
        let tool_types = HashMap::from([
            ("tool_search".to_owned(), ToolType::ToolSearch),
            ("weather".to_owned(), ToolType::Function),
        ]);
        let unnamed = serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"id": "fc_candidate", "type": "function_call", "call_id": "call_candidate",
                "arguments": "", "status": "in_progress"}
        });

        let mut pending_accumulator = ResponseAccumulator::new("resp_pending".to_owned(), None);
        let mut pending_translator = FunctionSseTranslator::new(tool_types.clone());
        translate(&mut pending_accumulator, &mut pending_translator, &unnamed);
        assert_eq!(
            pending_translator.unfinished_search_item_ids(),
            HashSet::from(["fc_candidate"])
        );

        let mut ordinary_accumulator = ResponseAccumulator::new("resp_ordinary".to_owned(), None);
        let mut ordinary_translator = FunctionSseTranslator::new(tool_types);
        translate(&mut ordinary_accumulator, &mut ordinary_translator, &unnamed);
        let resolved = serde_json::json!({
            "type": "response.function_call_arguments.done", "output_index": 0,
            "item_id": "fc_candidate", "call_id": "call_candidate", "name": "weather",
            "arguments": "{}"
        });
        translate(&mut ordinary_accumulator, &mut ordinary_translator, &resolved);
        assert!(ordinary_translator.unfinished_search_item_ids().is_empty());
    }

    #[test]
    fn upstream_failure_may_terminate_an_incomplete_search_without_false_completion() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator =
            FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let added = &search_event_sequence("fc_search", "call_search", "{}")[0];
        translate(&mut accumulator, &mut translator, added);
        let failed = serde_json::json!({
            "type": "response.failed",
            "response": {
                "id": "upstream_failed", "status": "failed", "usage": null,
                "error": {"code": "provider_failure", "message": "provider stopped"},
                "incomplete_details": {"reason": "upstream_error"}
            }
        });
        let translated = translate(&mut accumulator, &mut translator, &failed);

        assert_eq!(translated.frames.len(), 1);
        assert_eq!(translated.frames[0].event_type, SSEEventType::ResponseFailed);
        assert!(translator.finish().is_ok());
    }

    #[test]
    fn pending_function_stream_state_has_aggregate_byte_and_call_count_limits() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::new());
        for output_index in 0..MAX_PENDING_FUNCTION_CALLS {
            let unnamed = serde_json::json!({
                "type": "response.output_item.added", "output_index": output_index,
                "item": {"id": format!("fc_{output_index}"), "type": "function_call",
                    "call_id": format!("call_{output_index}"), "arguments": "", "status": "in_progress"}
            });
            translate(&mut accumulator, &mut translator, &unnamed);
        }
        let over_count = serde_json::json!({
            "type": "response.output_item.added", "output_index": MAX_PENDING_FUNCTION_CALLS,
            "item": {"id": "fc_over", "type": "function_call", "call_id": "call_over",
                "arguments": "", "status": "in_progress"}
        });
        let count_error = accumulator
            .process_sse_line_with_translator(&sse(&over_count), &mut translator)
            .expect_err("pending call count must be bounded");
        assert!(count_error.to_string().contains("pending calls"));

        let mut bytes_accumulator = ResponseAccumulator::new("resp_2".to_owned(), None);
        let mut bytes_translator = FunctionSseTranslator::new(HashMap::new());
        let mut byte_error = None;
        for output_index in 0..MAX_PENDING_FUNCTION_CALLS {
            let unnamed = serde_json::json!({
                "type": "response.output_item.added", "output_index": output_index,
                "item": {"id": format!("fc_bytes_{output_index}"), "type": "function_call",
                    "call_id": format!("call_bytes_{output_index}"), "arguments": "", "status": "in_progress"}
            });
            if let Err(error) =
                bytes_accumulator.process_sse_line_with_translator(&sse(&unnamed), &mut bytes_translator)
            {
                byte_error = Some(error);
                break;
            }
            let delta = serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": output_index,
                "item_id": format!("fc_bytes_{output_index}"), "delta": "x".repeat(4 * 1024)
            });
            match bytes_accumulator.process_sse_line_with_translator(&sse(&delta), &mut bytes_translator) {
                Ok(_) => {}
                Err(error) => {
                    byte_error = Some(error);
                    break;
                }
            }
        }
        assert!(
            byte_error
                .expect("aggregate pending bytes must be bounded")
                .to_string()
                .contains("unnamed function-call SSE exceeded")
        );

        let mut search_accumulator = ResponseAccumulator::new("resp_search_pending".to_owned(), None);
        let mut search_translator = FunctionSseTranslator::new(HashMap::from([
            ("tool_search".to_owned(), ToolType::ToolSearch),
            ("weather".to_owned(), ToolType::Function),
        ]));
        for output_index in 0..MAX_PENDING_FUNCTION_CALLS {
            let unnamed = serde_json::json!({
                "type": "response.output_item.added", "output_index": output_index,
                "item": {"id": format!("fc_search_{output_index}"), "type": "function_call",
                    "call_id": format!("call_search_{output_index}"), "arguments": "", "status": "in_progress"}
            });
            translate(&mut search_accumulator, &mut search_translator, &unnamed);
        }
        let over_count = serde_json::json!({
            "type": "response.output_item.added", "output_index": MAX_PENDING_FUNCTION_CALLS,
            "item": {"id": "fc_search_over", "type": "function_call", "call_id": "call_search_over",
                "arguments": "", "status": "in_progress"}
        });
        let error = search_accumulator
            .process_sse_line_with_translator(&sse(&over_count), &mut search_translator)
            .expect_err("search-active pending overflow must use invalid-search classification");
        assert!(error.is_invalid_upstream_tool_search(), "{error}");
    }

    #[test]
    fn tool_search_argument_buffer_accepts_exact_limit_and_rejects_one_more_byte() {
        let prefix = r#"{"query":""#;
        let suffix = r#""}"#;
        let exact_arguments = format!(
            "{prefix}{}{suffix}",
            "x".repeat(MAX_PENDING_FUNCTION_BYTES - prefix.len() - suffix.len())
        );
        assert_eq!(exact_arguments.len(), MAX_PENDING_FUNCTION_BYTES);

        let mut exact_accumulator = ResponseAccumulator::new("resp_exact".to_owned(), None);
        let mut exact_translator =
            FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let exact_events = search_event_sequence("fc_exact", "call_exact", &exact_arguments);
        translate(&mut exact_accumulator, &mut exact_translator, &exact_events[0]);
        assert!(
            exact_accumulator
                .process_sse_line_with_translator(&sse(&exact_events[1]), &mut exact_translator)
                .is_ok()
        );
        assert!(
            exact_accumulator
                .process_sse_line_with_translator(&sse(&exact_events[2]), &mut exact_translator)
                .is_ok()
        );

        let over_arguments = format!("{exact_arguments}x");
        let mut over_accumulator = ResponseAccumulator::new("resp_over".to_owned(), None);
        let mut over_translator =
            FunctionSseTranslator::new(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let over_events = search_event_sequence("fc_over", "call_over", &over_arguments);
        translate(&mut over_accumulator, &mut over_translator, &over_events[0]);
        assert!(
            over_accumulator
                .process_sse_line_with_translator(&sse(&over_events[1]), &mut over_translator)
                .is_ok()
        );
        assert!(
            over_accumulator
                .process_sse_line_with_translator(&sse(&over_events[2]), &mut over_translator)
                .expect_err("one byte beyond the search-call limit must fail")
                .to_string()
                .contains("invalid tool-search call")
        );
    }

    #[test]
    fn custom_function_arguments_are_emitted_incrementally() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let mut frames = Vec::new();

        for event in [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_custom",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_custom",
                    "name": "raw_echo",
                    "arguments": ""
                }
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_custom",
                "call_id": "call_custom",
                "delta": "{\"in"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_custom",
                "call_id": "call_custom",
                "delta": "put\":\"hello "
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_custom",
                "call_id": "call_custom",
                "delta": "world\"}"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "item_id": "fc_custom",
                "call_id": "call_custom",
                "name": "raw_echo",
                "arguments": "{\"input\":\"hello world\"}"
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_custom",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_custom",
                    "name": "raw_echo",
                    "arguments": "{\"input\":\"hello world\"}"
                }
            }),
        ] {
            frames.extend(translate(&mut accumulator, &mut translator, &event).frames);
        }

        assert_eq!(
            frames.iter().map(|frame| frame.event_type).collect::<Vec<_>>(),
            [
                SSEEventType::OutputItemAdded,
                SSEEventType::CustomToolCallInputDelta,
                SSEEventType::CustomToolCallInputDelta,
                SSEEventType::CustomToolCallInputDone,
                SSEEventType::OutputItemDone,
            ]
        );
        assert_eq!(frames[0].wire.rest["item"]["type"], "custom_tool_call");
        assert_eq!(frames[1].wire.rest["delta"], "hello ");
        assert_eq!(frames[2].wire.rest["delta"], "world");
        assert_eq!(frames[3].wire.rest["input"], "hello world");
        assert_eq!(frames[4].wire.rest["item"]["input"], "hello world");
    }

    #[test]
    fn custom_input_deltas_match_authoritative_done_input() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                    "name": "raw_echo", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "fc_1", "call_id": "call_1", "delta": "{\"input\":\"hello\""
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.done", "output_index": 0,
                "item_id": "fc_1", "call_id": "call_1", "name": "raw_echo",
                "arguments": "{\"input\":\"hello\",\"extra\":true}"
            }),
            serde_json::json!({
                "type": "response.output_item.done", "output_index": 0,
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                    "name": "raw_echo", "arguments": "{\"input\":\"hello\",\"extra\":true}", "status": "completed"}
            }),
        ];

        let mut frames = Vec::new();
        for event in events {
            frames.extend(translate(&mut accumulator, &mut translator, &event).frames);
        }
        let deltas = frames
            .iter()
            .filter(|frame| frame.event_type == SSEEventType::CustomToolCallInputDelta)
            .filter_map(|frame| frame.wire.rest["delta"].as_str())
            .collect::<String>();
        let done = frames
            .iter()
            .find(|frame| frame.event_type == SSEEventType::CustomToolCallInputDone)
            .and_then(|frame| frame.wire.rest["input"].as_str())
            .expect("input.done");

        assert_eq!(deltas, done);
    }

    #[test]
    fn custom_input_rejects_authoritative_value_that_contradicts_deltas() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                    "name": "raw_echo", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "fc_1", "call_id": "call_1", "delta": "{\"input\":\"hello\"}"
            }),
        ];
        for event in events {
            translate(&mut accumulator, &mut translator, &event);
        }
        let done = serde_json::json!({
            "type": "response.function_call_arguments.done", "output_index": 0,
            "item_id": "fc_1", "call_id": "call_1", "name": "raw_echo",
            "arguments": "{\"input\":\"bye\"}"
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&done), &mut translator)
            .expect_err("contradictory final input must fail");
        assert!(error.to_string().contains("contradicts streamed custom tool input"));
    }

    #[test]
    fn malformed_custom_input_escape_is_rejected() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let added = serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                "name": "raw_echo", "arguments": "", "status": "in_progress"}
        });
        translate(&mut accumulator, &mut translator, &added);
        let delta = serde_json::json!({
            "type": "response.function_call_arguments.delta", "output_index": 0,
            "item_id": "fc_1", "call_id": "call_1", "delta": r#"{"input":"\q"#
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&delta), &mut translator)
            .expect_err("invalid JSON string escape must fail");
        assert!(error.to_string().contains("invalid custom tool input"));
    }

    #[test]
    fn custom_input_waits_for_split_unicode_surrogate_pair() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                    "name": "raw_echo", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "fc_1", "call_id": "call_1", "delta": r#"{"input":"hi \uD83D"#
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "fc_1", "call_id": "call_1", "delta": r#"\uDE00"}"#
            }),
        ];

        let frames = events
            .iter()
            .flat_map(|event| translate(&mut accumulator, &mut translator, event).frames)
            .collect::<Vec<_>>();
        let input = frames
            .iter()
            .filter(|frame| frame.event_type == SSEEventType::CustomToolCallInputDelta)
            .filter_map(|frame| frame.wire.rest["delta"].as_str())
            .collect::<String>();

        assert_eq!(input, "hi 😀");
    }

    #[test]
    fn custom_input_over_limit_is_rejected() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let added = serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                "name": "raw_echo", "arguments": "", "status": "in_progress"}
        });
        translate(&mut accumulator, &mut translator, &added);
        let oversized = serde_json::json!({
            "type": "response.function_call_arguments.delta", "output_index": 0,
            "item_id": "fc_1", "call_id": "call_1",
            "delta": format!("{{\"input\":\"{}", "x".repeat(MAX_PENDING_FUNCTION_BYTES + 1))
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&oversized), &mut translator)
            .expect_err("oversized custom input must fail");
        assert!(error.to_string().contains("function-call SSE exceeded"));
    }

    #[test]
    fn ordinary_functions_pass_through_unchanged() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([("echo".to_owned(), ToolType::Function)]));
        let event = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 3,
            "item": {
                "id": "fc_echo",
                "type": "function_call",
                "call_id": "call_echo",
                "name": "echo",
                "arguments": ""
            }
        });

        let translated = translate(&mut accumulator, &mut translator, &event);

        assert_eq!(translated.frames.len(), 1);
        assert_eq!(translated.frames[0].event_type, SSEEventType::OutputItemAdded);
        assert_eq!(translated.frames[0].wire.rest["item"]["type"], "function_call");
        assert_eq!(translated.defer_from_output_index, None);
    }

    #[test]
    fn unnamed_function_frames_are_recovered_by_output_index_when_done_changes_id() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([("echo".to_owned(), ToolType::Function)]));
        let mut frames = Vec::new();
        let mut defer_boundaries = Vec::new();

        for event in [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "fc_transient",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_echo",
                    "arguments": ""
                }
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": "fc_transient",
                "call_id": "call_echo",
                "delta": "{\"value\":1}"
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "id": "fc_stable",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_echo",
                    "name": "echo",
                    "arguments": "{\"value\":1}"
                }
            }),
        ] {
            let translated = translate(&mut accumulator, &mut translator, &event);
            defer_boundaries.push(translated.defer_from_output_index);
            frames.extend(translated.frames);
        }

        assert_eq!(
            frames.iter().map(|frame| frame.event_type).collect::<Vec<_>>(),
            [
                SSEEventType::OutputItemAdded,
                SSEEventType::FunctionCallArgumentsDelta,
                SSEEventType::OutputItemDone,
            ]
        );
        assert_eq!(frames[0].wire.rest["item"]["id"], "fc_transient");
        assert_eq!(frames[1].wire.rest["item_id"], "fc_transient");
        assert_eq!(frames[2].wire.rest["item"]["id"], "fc_stable");
        assert_eq!(defer_boundaries, [Some(1), Some(1), None]);
    }

    #[test]
    fn parallel_unnamed_functions_with_empty_ids_remain_distinct() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([
            ("first".to_owned(), ToolType::Function),
            ("second".to_owned(), ToolType::Function),
        ]));
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "", "type": "function_call", "arguments": ""}
            }),
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 1,
                "item": {"id": "", "type": "function_call", "arguments": ""}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "", "delta": "{\"value\":\"a\"}"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 1,
                "item_id": "", "delta": "{\"value\":\"b\"}"
            }),
            serde_json::json!({
                "type": "response.output_item.done", "output_index": 0,
                "item": {"id": "fc_first", "type": "function_call", "call_id": "call_first",
                    "name": "first", "arguments": "{\"value\":\"a\"}", "status": "completed"}
            }),
            serde_json::json!({
                "type": "response.output_item.done", "output_index": 1,
                "item": {"id": "fc_second", "type": "function_call", "call_id": "call_second",
                    "name": "second", "arguments": "{\"value\":\"b\"}", "status": "completed"}
            }),
        ];

        let mut frames = Vec::new();
        for event in events {
            frames.extend(translate(&mut accumulator, &mut translator, &event).frames);
        }

        assert_eq!(
            frames
                .iter()
                .filter(|frame| frame.event_type == SSEEventType::OutputItemAdded)
                .count(),
            2
        );
        assert_eq!(
            frames
                .iter()
                .filter(|frame| frame.event_type == SSEEventType::OutputItemDone)
                .count(),
            2
        );
    }

    #[test]
    fn parallel_named_custom_functions_with_empty_ids_remain_distinct() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([
            ("first".to_owned(), ToolType::Custom),
            ("second".to_owned(), ToolType::Custom),
        ]));
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "", "type": "function_call", "call_id": "call_first",
                    "name": "first", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 1,
                "item": {"id": "", "type": "function_call", "call_id": "call_second",
                    "name": "second", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "", "call_id": "call_first", "delta": "{\"input\":\"a\"}"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 1,
                "item_id": "", "call_id": "call_second", "delta": "{\"input\":\"b\"}"
            }),
        ];

        let frames = events
            .iter()
            .flat_map(|event| translate(&mut accumulator, &mut translator, event).frames)
            .collect::<Vec<_>>();
        let deltas = frames
            .iter()
            .filter(|frame| frame.event_type == SSEEventType::CustomToolCallInputDelta)
            .map(|frame| (frame.wire.output_index, frame.wire.rest["delta"].as_str()))
            .collect::<Vec<_>>();

        assert_eq!(deltas, [(Some(0), Some("a")), (Some(1), Some("b"))]);
    }

    #[test]
    fn unnamed_custom_function_with_empty_id_uses_one_public_id() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "", "type": "function_call", "call_id": "call_1", "arguments": ""}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "", "call_id": "call_1", "delta": "{\"input\":\"hello\"}"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.done", "output_index": 0,
                "item_id": "", "call_id": "call_1", "name": "raw_echo",
                "arguments": "{\"input\":\"hello\"}"
            }),
        ];

        let frames = events
            .iter()
            .flat_map(|event| translate(&mut accumulator, &mut translator, event).frames)
            .collect::<Vec<_>>();
        let added_id = frames
            .iter()
            .find(|frame| frame.event_type == SSEEventType::OutputItemAdded)
            .and_then(|frame| frame.wire.rest["item"]["id"].as_str())
            .expect("custom item id");
        let lifecycle_ids = frames.iter().filter_map(|frame| {
            matches!(
                frame.event_type,
                SSEEventType::CustomToolCallInputDelta | SSEEventType::CustomToolCallInputDone
            )
            .then(|| frame.wire.rest["item_id"].as_str())
            .flatten()
        });

        assert!(lifecycle_ids.eq(std::iter::repeat_n(added_id, 2)));
    }

    #[test]
    fn gateway_owned_functions_are_suppressed_and_mark_the_defer_boundary() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator =
            FunctionSseTranslator::new(HashMap::from([("web_search".to_owned(), ToolType::WebSearch)]));
        let added = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                "id": "fc_search",
                "type": "function_call",
                "call_id": "call_search",
                "name": "web_search",
                "arguments": ""
            }
        });
        let delta = serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 2,
            "item_id": "fc_search",
            "call_id": "call_search",
            "delta": "{}"
        });

        let added = translate(&mut accumulator, &mut translator, &added);
        let delta = translate(&mut accumulator, &mut translator, &delta);

        assert!(added.frames.is_empty());
        assert_eq!(added.defer_from_output_index, Some(2));
        assert!(delta.frames.is_empty());
        assert_eq!(delta.defer_from_output_index, Some(2));
    }
}
