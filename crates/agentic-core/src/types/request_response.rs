use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use super::io::{
    InputItem, InputMessage, InputMessageContent, OutputItem, ResponseUsage, ResponsesInput, ResponsesTool, ToolChoice,
};
use crate::utils::common::serialize_to_string;

#[derive(Debug, Clone, Serialize)]
pub struct RequestPayload {
    pub model: String,
    pub input: ResponsesInput,
    pub instructions: Option<String>,
    pub previous_response_id: Option<String>,
    pub conversation_id: Option<String>,
    pub tools: Option<Vec<ResponsesTool>>,
    pub tool_choice: ToolChoice,
    #[serde(skip)]
    pub tool_choice_explicitly_set: bool,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_true")]
    pub store: bool,
    pub include: Option<Vec<String>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_output_tokens: Option<u32>,
    pub truncation: Option<String>,
    pub metadata: Option<Value>,
}

fn default_true() -> bool {
    true
}

impl<'de> Deserialize<'de> for RequestPayload {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireRequestPayload {
            model: String,
            input: ResponsesInput,
            instructions: Option<String>,
            previous_response_id: Option<String>,
            conversation_id: Option<String>,
            tools: Option<Vec<ResponsesTool>>,
            tool_choice: Option<ToolChoice>,
            #[serde(default)]
            stream: bool,
            #[serde(default = "default_true")]
            store: bool,
            include: Option<Vec<String>>,
            temperature: Option<f64>,
            top_p: Option<f64>,
            max_output_tokens: Option<u32>,
            truncation: Option<String>,
            metadata: Option<Value>,
        }

        let wire = WireRequestPayload::deserialize(deserializer)?;
        let tool_choice_explicitly_set = wire.tool_choice.is_some();
        Ok(Self {
            model: wire.model,
            input: wire.input,
            instructions: wire.instructions,
            previous_response_id: wire.previous_response_id,
            conversation_id: wire.conversation_id,
            tools: wire.tools,
            tool_choice: wire.tool_choice.unwrap_or_default(),
            tool_choice_explicitly_set,
            stream: wire.stream,
            store: wire.store,
            include: wire.include,
            temperature: wire.temperature,
            top_p: wire.top_p,
            max_output_tokens: wire.max_output_tokens,
            truncation: wire.truncation,
            metadata: wire.metadata,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct UpstreamRequest<'a> {
    pub model: &'a str,
    pub input: &'a ResponsesInput,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<&'a Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "is_default_tool_choice")]
    pub tool_choice: &'a ToolChoice,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<&'a Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<&'a Value>,
}

fn is_default_tool_choice(choice: &ToolChoice) -> bool {
    matches!(choice, ToolChoice::Auto)
}

impl RequestPayload {
    pub(crate) fn normalize_for_upstream(&mut self) {
        if let Some(instructions) = self.instructions.take().filter(|instructions| !instructions.is_empty()) {
            self.input.prepend_system_text(instructions);
        }
        self.input.normalize_for_upstream();
    }

    /// Construct an `UpstreamRequest` borrowing from this request, suitable for forwarding to vLLM.
    #[must_use]
    pub fn to_upstream_request(&self, stream: bool) -> UpstreamRequest<'_> {
        UpstreamRequest {
            model: &self.model,
            input: &self.input,
            stream,
            instructions: self.instructions.as_deref(),
            tools: self.tools.as_ref(),
            tool_choice: &self.tool_choice,
            include: self.include.as_ref(),
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_output_tokens,
            truncation: self.truncation.as_deref(),
            metadata: self.metadata.as_ref(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncompleteDetails {
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsePayload {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub model: String,
    pub status: String,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    pub usage: Option<ResponseUsage>,
    pub incomplete_details: Option<IncompleteDetails>,
    pub error: Option<Value>,
    pub previous_response_id: Option<String>,
    pub conversation_id: Option<String>,
    pub instructions: Option<String>,
}

impl ResponsePayload {
    #[must_use]
    pub fn as_responses_chunk(&self) -> String {
        let json_str = serialize_to_string(self).unwrap_or_else(|_| String::new());
        format!("data: {json_str}\n\n")
    }
}

impl From<&ResponsesInput> for Vec<InputItem> {
    fn from(input: &ResponsesInput) -> Self {
        match input {
            ResponsesInput::Text(text) => vec![InputItem::Message(InputMessage {
                role: "user".into(),
                content: InputMessageContent::Text(text.clone()),
            })],
            ResponsesInput::Items(items) => items.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_payload_tracks_explicit_tool_choice_presence() {
        let absent: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi"
        }))
        .unwrap();
        assert_eq!(absent.tool_choice, ToolChoice::Auto);
        assert!(!absent.tool_choice_explicitly_set);

        let explicit: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": "none"
        }))
        .unwrap();
        assert_eq!(explicit.tool_choice, ToolChoice::None);
        assert!(explicit.tool_choice_explicitly_set);
    }
}
