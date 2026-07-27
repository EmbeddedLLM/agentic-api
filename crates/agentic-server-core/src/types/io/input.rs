use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::output::{CustomToolCall, FunctionToolCall, ReasoningOutput, ToolSearchCall, ToolSearchOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTextContent {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputImageContent {
    pub image_url: Option<String>,
    pub detail: Option<String>,
}

/// Content item inside a message input.
///
/// Uses an internally-tagged enum — serde consumes `"type"` for the variant
/// discriminant so the inner structs must NOT redeclare a `type_` field.
/// `output_text` and `reasoning_text` reuse `InputTextContent` since they
/// carry only a `text` field; they are preserved so vLLM sees the full history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    InputText(InputTextContent),
    InputImage(InputImageContent),
    /// Assistant output text in rehydrated history.
    OutputText(InputTextContent),
    /// Reasoning step text in rehydrated history.
    ReasoningText(InputTextContent),
    /// Any other content type — drop silently.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMessage {
    pub role: String,
    pub content: InputMessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputMessageContent {
    Text(String),
    Parts(Vec<InputContent>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolResultMessage {
    pub call_id: String,
    pub output: String,
}

/// Client result for a freeform custom tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomToolCallOutputMessage {
    pub call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub output: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputItem {
    #[serde(rename = "message")]
    Message(InputMessage),
    /// The model's tool invocation — appears in rehydrated history so vLLM sees
    /// the full call/output pair across turns.
    #[serde(rename = "function_call")]
    FunctionCall(FunctionToolCall),
    #[serde(rename = "function_call_output")]
    FunctionCallOutput(FunctionToolResultMessage),
    /// The model's freeform invocation, retained when rehydrating the matching
    /// client-provided `custom_tool_call_output` on the next turn.
    #[serde(rename = "custom_tool_call")]
    CustomToolCall(CustomToolCall),
    #[serde(rename = "custom_tool_call_output")]
    CustomToolCallOutput(CustomToolCallOutputMessage),
    /// The model's request for the caller to discover deferred tools.
    #[serde(rename = "tool_search_call")]
    ToolSearchCall(ToolSearchCall),
    /// The tool definitions loaded by a hosted or client-executed search.
    #[serde(rename = "tool_search_output")]
    ToolSearchOutput(ToolSearchOutput),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningOutput),
    #[serde(other)]
    Unknown,
}

impl InputItem {
    #[must_use]
    pub(crate) fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_search_input_items_preserve_omitted_execution_and_status() {
        let expected = serde_json::json!([
            {
                "type": "tool_search_call",
                "call_id": "call_search",
                "arguments": {"query": "tools"}
            },
            {
                "type": "tool_search_output",
                "call_id": "call_search",
                "tools": []
            }
        ]);
        let input: ResponsesInput = serde_json::from_value(expected.clone()).unwrap();
        let ResponsesInput::Items(items) = &input else {
            panic!("expected input items");
        };
        let InputItem::ToolSearchCall(call) = &items[0] else {
            panic!("expected tool-search call");
        };
        assert_eq!(call.execution, None);
        assert_eq!(call.status, None);
        let InputItem::ToolSearchOutput(output) = &items[1] else {
            panic!("expected tool-search output");
        };
        assert_eq!(output.execution, None);
        assert_eq!(output.status, None);
        assert_eq!(serde_json::to_value(input).unwrap(), expected);
    }
}
