use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use super::output::{FunctionToolCall, ReasoningOutput};

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
/// Uses an internally-tagged enum. Unknown nested content deliberately fails
/// `InputMessage` deserialization so the containing `InputItem` can fall back
/// to unit `Unknown` and be filtered before the upstream request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    InputText(InputTextContent),
    InputImage(InputImageContent),
    /// Assistant output text in rehydrated history.
    OutputText(InputTextContent),
    /// Reasoning step text in rehydrated history.
    ReasoningText(InputTextContent),
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

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    Message(InputMessage),
    /// The model's tool invocation — appears in rehydrated history so vLLM sees
    /// the full call/output pair across turns.
    FunctionCall(FunctionToolCall),
    FunctionCallOutput(FunctionToolResultMessage),
    Reasoning(ReasoningOutput),
    Unknown,
}

impl<'de> Deserialize<'de> for InputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let Some(type_name) = value.get("type").and_then(Value::as_str).map(str::to_owned) else {
            return Ok(Self::Unknown);
        };

        match type_name.as_str() {
            "message" => Ok(serde_json::from_value(value).map_or(Self::Unknown, Self::Message)),
            "function_call" => Ok(serde_json::from_value(value).map_or(Self::Unknown, Self::FunctionCall)),
            "function_call_output" => Ok(serde_json::from_value(value).map_or(Self::Unknown, Self::FunctionCallOutput)),
            "reasoning" => Ok(serde_json::from_value(value).map_or(Self::Unknown, Self::Reasoning)),
            _ => Ok(Self::Unknown),
        }
    }
}

impl InputItem {
    #[must_use]
    pub(crate) fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }

    fn is_system_role_for_upstream(&self) -> bool {
        match self {
            Self::Message(message) => message.role == "system",
            Self::FunctionCall(_) | Self::FunctionCallOutput(_) | Self::Reasoning(_) | Self::Unknown => false,
        }
    }

    pub(crate) fn normalize_for_upstream(&mut self) {
        match self {
            Self::Message(message) if message.role == "developer" => {
                message.role = "system".to_string();
            }
            Self::Message(_)
            | Self::FunctionCall(_)
            | Self::FunctionCallOutput(_)
            | Self::Reasoning(_)
            | Self::Unknown => {}
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

impl ResponsesInput {
    pub(crate) fn prepend_system_text(&mut self, text: String) {
        let system = InputItem::Message(InputMessage {
            role: "system".to_string(),
            content: InputMessageContent::Text(text),
        });

        match self {
            Self::Text(user_text) => {
                *self = Self::Items(vec![
                    system,
                    InputItem::Message(InputMessage {
                        role: "user".to_string(),
                        content: InputMessageContent::Text(std::mem::take(user_text)),
                    }),
                ]);
            }
            Self::Items(items) => items.insert(0, system),
        }
    }

    pub(crate) fn normalize_for_upstream(&mut self) {
        if let Self::Items(items) = self {
            items.retain(|item| !item.is_unknown());
            for item in &mut *items {
                item.normalize_for_upstream();
            }
            items.sort_by_key(|item| !item.is_system_role_for_upstream());
        }
    }
}
