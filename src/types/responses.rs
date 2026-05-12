use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTextContent {
    #[serde(rename = "type")]
    pub type_: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputImageContent {
    #[serde(rename = "type")]
    pub type_: String,
    pub image_url: Option<String>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputContent {
    #[serde(rename = "input_text")]
    Text(InputTextContent),
    #[serde(rename = "input_image")]
    Image(InputImageContent),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputItem {
    #[serde(rename = "message")]
    Message(InputMessage),
    #[serde(rename = "function_call_output")]
    FunctionCallOutput(FunctionToolResultMessage),
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTextContent {
    #[serde(rename = "type")]
    pub type_: String,
    pub text: String,
    #[serde(default)]
    pub annotations: Vec<Value>,
}

impl OutputTextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            type_: "output_text".into(),
            text: text.into(),
            annotations: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMessage {
    pub id: String,
    pub role: String,
    pub status: String,
    #[serde(default)]
    pub content: Vec<OutputTextContent>,
}

impl OutputMessage {
    pub fn new(id: impl Into<String>, status: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: "assistant".into(),
            status: status.into(),
            content: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolCall {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message(OutputMessage),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionToolCall),
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputTokenDetails {
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputTokenDetails {
    pub reasoning_tokens: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub total_tokens: i64,
    #[serde(default)]
    pub input_tokens_details: InputTokenDetails,
    #[serde(default)]
    pub output_tokens_details: OutputTokenDetails,
}

// ---------------------------------------------------------------------------
// Tool types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionTool {
    #[serde(rename = "type")]
    pub type_: String,
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    pub strict: Option<bool>,
}

pub type ResponsesTool = FunctionTool;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function { name: String },
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Auto
    }
}

// ---------------------------------------------------------------------------
// Request / Response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    pub instructions: Option<String>,
    pub previous_response_id: Option<String>,
    pub conversation_id: Option<String>,
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(default)]
    pub tool_choice: ToolChoice,
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

impl ResponsesRequest {
    /// Produce a JSON value with gateway-only fields stripped, safe to forward to vLLM.
    pub fn to_upstream_body(&self, stream: bool) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "input": self.input,
            "stream": stream,
        });
        if let Some(v) = &self.instructions {
            body["instructions"] = serde_json::json!(v);
        }
        if let Some(v) = &self.tools {
            body["tools"] = serde_json::json!(v);
        }
        if let Some(v) = &self.include {
            body["include"] = serde_json::json!(v);
        }
        if let Some(v) = self.temperature {
            body["temperature"] = serde_json::json!(v);
        }
        if let Some(v) = self.top_p {
            body["top_p"] = serde_json::json!(v);
        }
        if let Some(v) = self.max_output_tokens {
            body["max_output_tokens"] = serde_json::json!(v);
        }
        if let Some(v) = &self.truncation {
            body["truncation"] = serde_json::json!(v);
        }
        if let Some(v) = &self.metadata {
            body["metadata"] = serde_json::json!(v);
        }
        body
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncompleteDetails {
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
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

impl ResponsesResponse {
    pub fn create_from_request(request: &ResponsesRequest) -> Self {
        Self {
            id: crate::utils::common::uuid7_str("resp_"),
            object: "response".into(),
            created_at: 0,
            model: request.model.clone(),
            status: "in_progress".into(),
            output: vec![],
            usage: None,
            incomplete_details: None,
            error: None,
            previous_response_id: request.previous_response_id.clone(),
            conversation_id: None,
            instructions: request.instructions.clone(),
        }
    }

    pub fn as_responses_chunk(&self) -> String {
        format!("data: {}\n\n", serde_json::to_string(self).unwrap_or_default())
    }
}

// ---------------------------------------------------------------------------
// Stream events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseEvent {
    #[serde(rename = "type")]
    pub type_: String,
    pub sequence_number: u64,
    pub response: ResponsesResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputItemEvent {
    #[serde(rename = "type")]
    pub type_: String,
    pub sequence_number: u64,
    pub output_index: usize,
    pub item: OutputItem,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartEvent {
    #[serde(rename = "type")]
    pub type_: String,
    pub sequence_number: u64,
    pub output_index: usize,
    pub content_index: usize,
    pub item_id: String,
    pub part: Option<OutputTextContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDeltaEvent {
    #[serde(rename = "type")]
    pub type_: String,
    pub sequence_number: u64,
    pub output_index: usize,
    pub item_id: String,
    #[serde(default)]
    pub content_index: usize,
    pub delta: Option<String>,
    pub text: Option<String>,
    pub arguments: Option<String>,
    #[serde(default)]
    pub logprobs: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    #[serde(rename = "type")]
    pub type_: String,
    pub sequence_number: u64,
    pub code: String,
    pub message: String,
    pub param: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum StreamEvent {
    Response(ResponseEvent),
    OutputItem(OutputItemEvent),
    ContentPart(ContentPartEvent),
    TextDelta(TextDeltaEvent),
    Error(ErrorEvent),
}

impl<'de> serde::Deserialize<'de> for StreamEvent {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        let type_str = v.get("type").and_then(Value::as_str).unwrap_or("");
        match type_str {
            "response.created"
            | "response.in_progress"
            | "response.completed"
            | "response.incomplete"
            | "response.failed" => serde_json::from_value(v)
                .map(Self::Response)
                .map_err(serde::de::Error::custom),
            "response.output_item.added" | "response.output_item.done" => serde_json::from_value(v)
                .map(Self::OutputItem)
                .map_err(serde::de::Error::custom),
            "response.content_part.added" | "response.content_part.done" => serde_json::from_value(v)
                .map(Self::ContentPart)
                .map_err(serde::de::Error::custom),
            "response.output_text.delta"
            | "response.output_text.done"
            | "response.function_call_arguments.delta"
            | "response.function_call_arguments.done" => serde_json::from_value(v)
                .map(Self::TextDelta)
                .map_err(serde::de::Error::custom),
            "response.error" => serde_json::from_value(v)
                .map(Self::Error)
                .map_err(serde::de::Error::custom),
            _ => Err(serde::de::Error::custom(format!(
                "unknown StreamEvent type: {type_str}"
            ))),
        }
    }
}

impl StreamEvent {
    pub fn type_str(&self) -> &str {
        match self {
            Self::Response(e) => &e.type_,
            Self::OutputItem(e) => &e.type_,
            Self::ContentPart(e) => &e.type_,
            Self::TextDelta(e) => &e.type_,
            Self::Error(e) => &e.type_,
        }
    }

    pub fn as_responses_chunk(&self) -> String {
        format!(
            "event: {}\ndata: {}\n\n",
            self.type_str(),
            serde_json::to_string(self).unwrap_or_default()
        )
    }
}
