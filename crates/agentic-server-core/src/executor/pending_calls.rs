//! Detects calls in an item sequence that never received a resolving output.
//!
//! Gateway-owned calls are always resolved within the same turn (their
//! output is appended before the turn ends), so anything still unresolved
//! after scanning a full item sequence is, by construction, something the
//! *client* owed a resolution for.

use crate::types::io::InputItem;

/// A client-owned call (plain `function`, Codex `namespace` member, or
/// `custom` tool) with no later matching output in the same item sequence.
pub(super) struct PendingCall {
    pub(super) call_id: String,
}

/// Scans `items` in order and returns every call left unresolved, in
/// emission order. A call counts as resolved once a later matching
/// `FunctionCallOutput`/`CustomToolCallOutput` with the same `call_id`
/// appears. Namespace member calls are represented as `InputItem::FunctionCall`
/// (their flattened name lives in the `name` field), so they're covered by
/// the same check as plain function calls.
pub(super) fn pending_calls(items: &[InputItem]) -> Vec<PendingCall> {
    let mut pending = Vec::new();
    for item in items {
        match item {
            InputItem::FunctionCall(call) => pending.push(call.call_id.clone()),
            InputItem::CustomToolCall(call) => pending.push(call.call_id.clone()),
            InputItem::FunctionCallOutput(output) => pending.retain(|call_id| *call_id != output.call_id),
            InputItem::CustomToolCallOutput(output) => pending.retain(|call_id| *call_id != output.call_id),
            InputItem::Message(_)
            | InputItem::Reasoning(_)
            | InputItem::McpListTools(_)
            | InputItem::Compaction(_)
            | InputItem::CompactionTrigger
            | InputItem::Unknown => {}
        }
    }
    pending.into_iter().map(|call_id| PendingCall { call_id }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::io::{
        CustomToolCall, CustomToolCallOutputMessage, FunctionToolResultMessage, InputFunctionToolCall, ToolCallOutput,
    };

    fn function_call(call_id: &str) -> InputItem {
        InputItem::FunctionCall(InputFunctionToolCall {
            id: None,
            call_id: call_id.to_owned(),
            name: "get_weather".to_owned(),
            namespace: None,
            arguments: "{}".to_owned(),
            status: None,
        })
    }

    fn function_call_output(call_id: &str) -> InputItem {
        InputItem::FunctionCallOutput(FunctionToolResultMessage {
            call_id: call_id.to_owned(),
            output: ToolCallOutput::Text(String::new()),
        })
    }

    fn custom_tool_call(call_id: &str) -> InputItem {
        InputItem::CustomToolCall(CustomToolCall {
            id: String::new(),
            status: None,
            call_id: call_id.to_owned(),
            name: "freeform".to_owned(),
            input: String::new(),
        })
    }

    fn custom_tool_call_output(call_id: &str) -> InputItem {
        InputItem::CustomToolCallOutput(CustomToolCallOutputMessage {
            call_id: call_id.to_owned(),
            name: None,
            output: ToolCallOutput::Text(String::new()),
        })
    }

    #[test]
    fn resolved_calls_are_not_pending() {
        let items = vec![function_call("call_1"), function_call_output("call_1")];
        assert!(pending_calls(&items).is_empty());
    }

    #[test]
    fn unresolved_function_call_is_reported_in_order() {
        let items = vec![
            function_call("call_1"),
            function_call("call_2"),
            function_call_output("call_1"),
        ];
        let pending = pending_calls(&items);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].call_id, "call_2");
    }

    #[test]
    fn unresolved_custom_tool_call_is_reported() {
        let items = vec![custom_tool_call("call_1")];
        let pending = pending_calls(&items);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].call_id, "call_1");
    }

    #[test]
    fn custom_tool_call_output_resolves_custom_tool_call() {
        let items = vec![custom_tool_call("call_1"), custom_tool_call_output("call_1")];
        assert!(pending_calls(&items).is_empty());
    }

    #[test]
    fn empty_items_have_no_pending_calls() {
        assert!(pending_calls(&[]).is_empty());
    }
}
