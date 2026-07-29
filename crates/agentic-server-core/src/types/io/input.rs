use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::event::MessageStatus;
use crate::utils::common::deserialize_from_value;

use super::output::{CustomToolCall, FunctionToolCall, ReasoningOutput};

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<MessageStatus>,
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

/// A model-generated function call replayed as Responses input.
///
/// Input replay is intentionally more permissive than [`FunctionToolCall`]
/// output: clients may omit `id` and `status` when passing prior items to a
/// later request or to the compact endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFunctionToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub call_id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<MessageStatus>,
}

impl From<FunctionToolCall> for InputFunctionToolCall {
    fn from(call: FunctionToolCall) -> Self {
        Self {
            id: Some(call.id),
            call_id: call.call_id,
            name: call.name,
            namespace: call.namespace,
            arguments: call.arguments,
            status: Some(call.status),
        }
    }
}

/// An opaque compacted context checkpoint accepted as Responses input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub encrypted_content: String,
}

/// Client result for a freeform custom tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomToolCallOutputMessage {
    pub call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub output: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum InputItem {
    #[serde(rename = "message")]
    Message(InputMessage),
    /// The model's tool invocation — appears in rehydrated history so vLLM sees
    /// the full call/output pair across turns.
    #[serde(rename = "function_call")]
    FunctionCall(InputFunctionToolCall),
    #[serde(rename = "function_call_output")]
    FunctionCallOutput(FunctionToolResultMessage),
    /// The model's freeform invocation, retained when rehydrating the matching
    /// client-provided `custom_tool_call_output` on the next turn.
    #[serde(rename = "custom_tool_call")]
    CustomToolCall(CustomToolCall),
    #[serde(rename = "custom_tool_call_output")]
    CustomToolCallOutput(CustomToolCallOutputMessage),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningOutput),
    #[serde(rename = "compaction")]
    Compaction(CompactionItem),
    #[serde(other)]
    Unknown,
}

impl<'de> Deserialize<'de> for InputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let item = match value.get("type").and_then(Value::as_str) {
            None | Some("message") => deserialize_from_value(value).map(Self::Message),
            Some("function_call") => deserialize_from_value(value).map(Self::FunctionCall),
            Some("function_call_output") => deserialize_from_value(value).map(Self::FunctionCallOutput),
            Some("custom_tool_call") => deserialize_from_value(value).map(Self::CustomToolCall),
            Some("custom_tool_call_output") => deserialize_from_value(value).map(Self::CustomToolCallOutput),
            Some("reasoning") => deserialize_from_value(value).map(Self::Reasoning),
            Some("compaction") => deserialize_from_value(value).map(Self::Compaction),
            Some(_) => return Ok(Self::Unknown),
        };
        item.map_err(serde::de::Error::custom)
    }
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompactionWindow {
    latest_index: usize,
    retained_start: usize,
}

impl CompactionWindow {
    #[must_use]
    pub(crate) const fn latest_index(self) -> usize {
        self.latest_index
    }

    #[must_use]
    pub(crate) fn retains_user_item(self, index: usize, item: &InputItem) -> bool {
        index >= self.retained_start
            && index < self.latest_index
            && matches!(item, InputItem::Message(message)
                if message.role == "user"
                    && message.id.is_some()
                    && message.status == Some(MessageStatus::Completed))
    }

    pub(crate) fn retained_user_items(self, items: &[InputItem]) -> impl Iterator<Item = &InputItem> {
        items
            .iter()
            .enumerate()
            .filter(move |(index, item)| self.retains_user_item(*index, item))
            .map(|(_, item)| item)
    }
}

#[must_use]
pub(crate) fn latest_compaction_window(items: &[InputItem]) -> Option<CompactionWindow> {
    let latest_index = items
        .iter()
        .rposition(|item| matches!(item, InputItem::Compaction(_)))?;
    let retained_start = items[..latest_index]
        .iter()
        .rposition(|item| matches!(item, InputItem::Compaction(_)))
        .map_or(0, |index| index + 1);
    Some(CompactionWindow {
        latest_index,
        retained_start,
    })
}

impl ResponsesInput {
    #[must_use]
    pub fn contains_compaction(&self) -> bool {
        matches!(self, Self::Items(items) if items.iter().any(|item| matches!(item, InputItem::Compaction(_))))
    }

    /// Return the canonical context sent to vLLM.
    ///
    /// vLLM does not understand public `compaction` items, so the latest item
    /// becomes an assistant message containing the locally generated summary.
    /// Items before that checkpoint are superseded and are omitted.
    #[must_use]
    pub fn model_input(&self) -> Cow<'_, Self> {
        let Self::Items(items) = self else {
            return Cow::Borrowed(self);
        };
        let Some(window) = latest_compaction_window(items) else {
            return Cow::Borrowed(self);
        };

        let model_items = window
            .retained_user_items(items)
            .chain(items[window.latest_index()..].iter())
            .map(|item| match item {
                InputItem::Compaction(compaction) => InputItem::Message(InputMessage {
                    id: None,
                    role: "assistant".to_owned(),
                    status: None,
                    content: InputMessageContent::Parts(vec![InputContent::OutputText(InputTextContent {
                        text: compaction.encrypted_content.clone(),
                    })]),
                }),
                other => other.clone(),
            })
            .collect();
        Cow::Owned(Self::Items(model_items))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_call_input_accepts_missing_status() {
        let item: InputItem = serde_json::from_value(serde_json::json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}"
        }))
        .expect("valid replay input");

        let InputItem::FunctionCall(call) = item else {
            panic!("expected function call");
        };
        assert_eq!(call.status, None);
    }

    #[test]
    fn malformed_known_type_is_not_reinterpreted_as_shorthand_message() {
        let result = serde_json::from_value::<InputItem>(serde_json::json!({
            "type": "function_call",
            "role": "user",
            "content": "not a function call"
        }));

        assert!(result.is_err());
    }

    #[test]
    fn compaction_item_becomes_assistant_model_context() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([{
            "type": "compaction",
            "id": "cmp_1",
            "encrypted_content": "summary"
        }]))
        .expect("valid compaction input");

        let model_input = input.model_input();
        let serialized = serde_json::to_value(model_input).expect("model input serializes");
        assert_eq!(serialized[0]["role"], "assistant");
        assert_eq!(serialized[0]["content"][0]["type"], "output_text");
        assert_eq!(serialized[0]["content"][0]["text"], "summary");
    }

    #[test]
    fn latest_compaction_preserves_canonical_user_messages_and_supersedes_prior_context() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "discard me"},
            {"type": "compaction", "encrypted_content": "old summary"},
            {"role": "assistant", "content": "also discard me"},
            {"type": "message", "id": "msg_keep", "role": "user", "status": "completed", "content": "retained user"},
            {"type": "compaction", "encrypted_content": "latest summary"},
            {"role": "user", "content": "keep me"}
        ]))
        .expect("valid compacted history");

        let serialized = serde_json::to_value(input.model_input()).expect("model input serializes");
        assert_eq!(serialized.as_array().map(Vec::len), Some(3));
        assert_eq!(serialized[0]["content"], "retained user");
        assert_eq!(serialized[1]["role"], "assistant");
        assert_eq!(serialized[1]["content"][0]["text"], "latest summary");
        assert_eq!(serialized[2]["content"], "keep me");
    }
}
