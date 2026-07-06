use serde::{Deserialize, Deserializer, Serialize, Serializer};
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
/// to raw JSON instead of silently dropping fields.
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

#[derive(Debug, Clone)]
pub enum InputItem {
    Message(InputMessage),
    /// The model's tool invocation — appears in rehydrated history so vLLM sees
    /// the full call/output pair across turns.
    FunctionCall(FunctionToolCall),
    FunctionCallOutput(FunctionToolResultMessage),
    ToolSearchCall(Value),
    CustomToolCall(Value),
    Reasoning(ReasoningOutput),
    Unknown(Value),
}

fn value_with_type<T: Serialize>(type_name: &str, value: &T) -> Result<Value, serde_json::Error> {
    let mut value = serde_json::to_value(value)?;
    if let Value::Object(map) = &mut value {
        map.insert("type".to_string(), Value::String(type_name.to_string()));
    }
    Ok(value)
}

fn raw_value_with_type(type_name: &str, value: &Value) -> Value {
    let mut value = value.clone();
    if let Value::Object(map) = &mut value {
        map.entry("type".to_string())
            .or_insert_with(|| Value::String(type_name.to_string()));
    }
    value
}

fn serialize_typed<S, T>(serializer: S, type_name: &str, value: &T) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    value_with_type(type_name, value)
        .map_err(serde::ser::Error::custom)?
        .serialize(serializer)
}

impl Serialize for InputItem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Message(item) => serialize_typed(serializer, "message", item),
            Self::FunctionCall(item) => serialize_typed(serializer, "function_call", item),
            Self::FunctionCallOutput(item) => serialize_typed(serializer, "function_call_output", item),
            Self::ToolSearchCall(item) => raw_value_with_type("tool_search_call", item).serialize(serializer),
            Self::CustomToolCall(item) => raw_value_with_type("custom_tool_call", item).serialize(serializer),
            Self::Reasoning(item) => serialize_typed(serializer, "reasoning", item),
            Self::Unknown(item) => item.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for InputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let Some(type_name) = value.get("type").and_then(Value::as_str) else {
            return Ok(Self::Unknown(value));
        };

        match type_name {
            "message" => Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::Message)),
            "function_call" => {
                Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::FunctionCall))
            }
            "function_call_output" => {
                Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::FunctionCallOutput))
            }
            "tool_search_call" => Ok(Self::ToolSearchCall(value)),
            "custom_tool_call" => Ok(Self::CustomToolCall(value)),
            "reasoning" => Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::Reasoning)),
            _ => Ok(Self::Unknown(value)),
        }
    }
}

fn normalize_raw_message_role_for_upstream(value: &mut Value) {
    let Some(object) = value.as_object_mut() else {
        return;
    };
    if object.get("role").and_then(Value::as_str) == Some("developer") {
        object.insert("role".to_string(), Value::String("system".to_string()));
    }
}

impl InputItem {
    fn is_system_role_for_upstream(&self) -> bool {
        match self {
            Self::Message(message) => message.role == "system",
            Self::ToolSearchCall(value) | Self::CustomToolCall(value) | Self::Unknown(value) => value
                .as_object()
                .and_then(|object| object.get("role"))
                .and_then(Value::as_str)
                .is_some_and(|role| role == "system"),
            Self::FunctionCall(_) | Self::FunctionCallOutput(_) | Self::Reasoning(_) => false,
        }
    }

    pub(crate) fn normalize_for_upstream(&mut self) {
        match self {
            Self::Message(message) if message.role == "developer" => {
                message.role = "system".to_string();
            }
            Self::ToolSearchCall(value) | Self::CustomToolCall(value) | Self::Unknown(value) => {
                normalize_raw_message_role_for_upstream(value);
            }
            Self::Message(_) | Self::FunctionCall(_) | Self::FunctionCallOutput(_) | Self::Reasoning(_) => {}
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
            for item in &mut *items {
                item.normalize_for_upstream();
            }
            items.sort_by_key(|item| !item.is_system_role_for_upstream());
        }
    }
}
