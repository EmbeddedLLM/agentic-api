use std::collections::HashMap;

use serde_json::Value;

use crate::events::{EventFrame, EventPayload, SSEEventType, SSEItemType};
use crate::executor::accumulator::AccumulatedFunctionCall;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::gateway_accumulator::synthetic_event;
use crate::tool::ToolType;
use crate::utils::common::serialize_to_string;

const MAX_PENDING_FUNCTION_BYTES: usize = 256 * 1024;

#[derive(Debug)]
enum FunctionCallShape {
    PublicFunction { output_index: u32 },
    GatewayOwned { output_index: u32 },
    Custom(CustomCallState),
}

impl FunctionCallShape {
    const fn output_index(&self) -> u32 {
        match self {
            Self::PublicFunction { output_index }
            | Self::GatewayOwned { output_index }
            | Self::Custom(CustomCallState { output_index, .. }) => *output_index,
        }
    }
}

#[derive(Debug)]
struct CustomCallState {
    public_item_id: String,
    output_index: u32,
    emitted_input: String,
    input_done: bool,
}

#[derive(Debug, Default)]
struct PendingFunctionCall {
    frames: Vec<EventFrame>,
    bytes: usize,
}

#[derive(Debug, Default)]
pub(super) struct FunctionSseTranslation {
    pub(super) frames: Vec<EventFrame>,
    pub(super) gateway_output_index: Option<u32>,
}

/// Restores normalized upstream function-call SSE to the public call shape.
/// Tool routing remains outside this type; it receives only the request's
/// model-visible name-to-type mapping.
#[derive(Debug, Default)]
pub(super) struct FunctionSseTranslator {
    tool_types: HashMap<String, ToolType>,
    active: HashMap<String, FunctionCallShape>,
    pending_unnamed: HashMap<String, PendingFunctionCall>,
    pending_bytes: usize,
}

impl FunctionSseTranslator {
    pub(super) fn new(tool_types: HashMap<String, ToolType>) -> Self {
        Self {
            tool_types,
            ..Self::default()
        }
    }

    pub(super) fn translate(
        &mut self,
        frame: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        match &frame.payload {
            EventPayload::OutputItemAdded {
                item_id,
                item_type: SSEItemType::FunctionCall,
                output_index,
                name: Some(name),
                ..
            } => self.start_call(item_id, name, *output_index, Some(frame.clone()), call),
            EventPayload::OutputItemAdded {
                item_id,
                item_type: SSEItemType::FunctionCall,
                name: None,
                ..
            } => self.buffer_unnamed(item_id.clone(), frame),
            EventPayload::FunctionCallArgsDelta {
                item_id, output_index, ..
            } => self.translate_delta(item_id, *output_index, frame.clone(), call),
            EventPayload::FunctionCallArgsDone {
                item_id,
                name,
                output_index,
                ..
            } => self.finish_arguments(item_id, name, *output_index, frame.clone(), call),
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
                gateway_output_index: None,
            }),
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
                self.active.insert(
                    item_id.to_owned(),
                    FunctionCallShape::Custom(CustomCallState {
                        public_item_id: crate::tool::custom::public_item_id(item_id),
                        output_index,
                        emitted_input: String::new(),
                        input_done: false,
                    }),
                );
                Ok(FunctionSseTranslation {
                    frames: call
                        .map(|call| custom_added_frame(&call))
                        .transpose()?
                        .into_iter()
                        .collect(),
                    gateway_output_index: None,
                })
            }
            ToolType::Mcp | ToolType::WebSearch | ToolType::FileSearch | ToolType::CodeInterpreter => {
                self.active
                    .insert(item_id.to_owned(), FunctionCallShape::GatewayOwned { output_index });
                Ok(FunctionSseTranslation {
                    frames: Vec::new(),
                    gateway_output_index: Some(output_index),
                })
            }
            ToolType::Function | ToolType::CodexNamespace => {
                self.active
                    .insert(item_id.to_owned(), FunctionCallShape::PublicFunction { output_index });
                Ok(FunctionSseTranslation {
                    frames: original.into_iter().collect(),
                    gateway_output_index: None,
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
        let key = self.active_key(item_id, output_index);
        match key.as_deref().and_then(|key| self.active.get_mut(key)) {
            Some(FunctionCallShape::PublicFunction { .. }) => Ok(FunctionSseTranslation {
                frames: vec![original],
                gateway_output_index: None,
            }),
            Some(FunctionCallShape::GatewayOwned { .. }) => Ok(FunctionSseTranslation::default()),
            Some(FunctionCallShape::Custom(state)) => {
                let frame = match call {
                    Some(call) => incremental_custom_delta(state, call.arguments())?,
                    None => None,
                };
                Ok(FunctionSseTranslation {
                    frames: frame.into_iter().collect(),
                    gateway_output_index: None,
                })
            }
            None => self.buffer_unnamed(item_id.to_owned(), original),
        }
    }

    fn finish_arguments(
        &mut self,
        item_id: &str,
        name: &str,
        output_index: u32,
        original: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        let mut translated = self.resolve_pending(item_id, name, output_index, call)?;
        let key = self.active_key(item_id, output_index);
        match key.as_deref().and_then(|key| self.active.get_mut(key)) {
            Some(FunctionCallShape::PublicFunction { .. }) | None => translated.frames.push(original),
            Some(FunctionCallShape::GatewayOwned { .. }) => {}
            Some(FunctionCallShape::Custom(state)) => {
                if let Some(call) = call {
                    translated.frames.extend(finish_custom_input(state, call.arguments())?);
                }
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
        let key = self.active_key(item_id, output_index);
        match key.as_deref().and_then(|key| self.active.remove(key)) {
            Some(FunctionCallShape::PublicFunction { .. }) | None => translated.frames.push(original),
            Some(FunctionCallShape::GatewayOwned { .. }) => {}
            Some(FunctionCallShape::Custom(mut state)) => {
                if let Some(call) = call {
                    translated
                        .frames
                        .extend(finish_custom_input(&mut state, call.arguments())?);
                    translated.frames.push(custom_done_frame(&state, &call)?);
                }
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
        if self.active_key(item_id, output_index).is_some() {
            return Ok(FunctionSseTranslation::default());
        }

        let pending = self.take_pending(item_id);
        let original_added = pending.iter().find(|frame| {
            matches!(
                frame.payload,
                EventPayload::OutputItemAdded {
                    item_type: SSEItemType::FunctionCall,
                    ..
                }
            )
        });
        let mut translated = self.start_call(item_id, name, output_index, original_added.cloned(), call)?;

        for frame in pending {
            if let EventPayload::FunctionCallArgsDelta { output_index, .. } = &frame.payload {
                let delta = self.translate_delta(item_id, *output_index, frame.clone(), call)?;
                translated.frames.extend(delta.frames);
            }
        }
        Ok(translated)
    }

    fn tool_type(&self, name: &str) -> ToolType {
        self.tool_types.get(name).copied().unwrap_or(ToolType::Function)
    }

    fn active_key(&self, item_id: &str, output_index: u32) -> Option<String> {
        self.active
            .contains_key(item_id)
            .then(|| item_id.to_owned())
            .or_else(|| {
                self.active
                    .iter()
                    .find_map(|(key, shape)| (shape.output_index() == output_index).then(|| key.clone()))
            })
    }

    fn buffer_unnamed(&mut self, item_id: String, frame: EventFrame) -> ExecutorResult<FunctionSseTranslation> {
        let bytes = serialize_to_string(&frame.wire)
            .map_err(ExecutorError::JsonError)?
            .len();
        if self.pending_bytes.saturating_add(bytes) > MAX_PENDING_FUNCTION_BYTES {
            return Err(ExecutorError::StreamError(format!(
                "unnamed function-call SSE exceeded {MAX_PENDING_FUNCTION_BYTES} buffered bytes"
            )));
        }
        let pending = self.pending_unnamed.entry(item_id).or_default();
        pending.frames.push(frame);
        pending.bytes = pending.bytes.saturating_add(bytes);
        self.pending_bytes = self.pending_bytes.saturating_add(bytes);
        Ok(FunctionSseTranslation::default())
    }

    fn take_pending(&mut self, item_id: &str) -> Vec<EventFrame> {
        let Some(pending) = self.pending_unnamed.remove(item_id) else {
            return Vec::new();
        };
        self.pending_bytes = self.pending_bytes.saturating_sub(pending.bytes);
        pending.frames
    }
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
    let Some(input) = partial_custom_input(arguments) else {
        return Ok(None);
    };
    let Some(delta) = input
        .strip_prefix(&state.emitted_input)
        .filter(|delta| !delta.is_empty())
        .map(str::to_owned)
    else {
        return Ok(None);
    };
    state.emitted_input = input;
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
    let input = crate::tool::custom::input_from_arguments(arguments);
    let remaining = input
        .strip_prefix(&state.emitted_input)
        .filter(|delta| !delta.is_empty())
        .map(str::to_owned);
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

fn partial_custom_input(arguments: &str) -> Option<String> {
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
    let end = unescaped_quote(encoded).unwrap_or(encoded.len());
    let mut end = end;
    loop {
        let candidate = format!("\"{}\"", &encoded[..end]);
        if let Ok(input) = serde_json::from_str::<String>(&candidate) {
            return Some(input);
        }
        end = encoded[..end].rfind('\\')?;
    }
}

fn unescaped_quote(value: &str) -> Option<usize> {
    let mut escaped = false;
    value.char_indices().find_map(|(index, character)| {
        if escaped {
            escaped = false;
            return None;
        }
        match character {
            '\\' => escaped = true,
            '"' => return Some(index),
            _ => {}
        }
        None
    })
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
        assert_eq!(translated.gateway_output_index, None);
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
        assert_eq!(added.gateway_output_index, Some(2));
        assert!(delta.frames.is_empty());
        assert_eq!(delta.gateway_output_index, None);
    }
}
