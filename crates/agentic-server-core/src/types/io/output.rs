use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::events::EventPayload;
use crate::executor::error::ExecutorError;
use crate::tool::ToolRegistry;
use crate::types::event::MessageStatus;
use crate::utils::common::deserialize_from_value_opt;
use crate::utils::uuid7_str;

use super::input::{
    InputContent, InputFunctionToolCall, InputItem, InputMessage, InputMessageContent, InputTextContent,
};

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
    pub status: MessageStatus,
    #[serde(default)]
    pub content: Vec<OutputTextContent>,
}

impl OutputMessage {
    pub fn new(id: impl Into<String>, status: MessageStatus) -> Self {
        Self {
            id: id.into(),
            role: "assistant".into(),
            status,
            content: vec![],
        }
    }
}

impl TryFrom<&EventPayload> for OutputMessage {
    type Error = ExecutorError;

    fn try_from(payload: &EventPayload) -> Result<Self, Self::Error> {
        let EventPayload::OutputItemAdded { item_id, .. } = payload else {
            return Err(ExecutorError::ParseError("expected OutputItemAdded payload".into()));
        };
        let id = if item_id.is_empty() {
            uuid7_str("msg_")
        } else {
            item_id.clone()
        };
        Ok(Self::new(id, MessageStatus::InProgress))
    }
}

impl From<OutputMessage> for InputMessage {
    fn from(msg: OutputMessage) -> Self {
        let parts = msg
            .content
            .into_iter()
            .map(|c| InputContent::OutputText(InputTextContent { text: c.text }))
            .collect();
        Self {
            id: Some(msg.id),
            role: msg.role,
            status: Some(msg.status),
            content: InputMessageContent::Parts(parts),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolCall {
    #[serde(default = "default_function_call_id")]
    #[serde(deserialize_with = "deserialize_function_call_id")]
    pub id: String,
    #[serde(default)]
    pub call_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    #[serde(default)]
    pub arguments: String,
    #[serde(default = "default_completed_status")]
    #[serde(deserialize_with = "deserialize_status_or_default")]
    pub status: MessageStatus,
}

/// A freeform custom tool invocation.
///
/// `input` is opaque text and must not be parsed as function-call JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<MessageStatus>,
    #[serde(default)]
    pub call_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub input: String,
}

fn default_completed_status() -> MessageStatus {
    MessageStatus::Completed
}

fn default_function_call_id() -> String {
    uuid7_str("fc_")
}

fn deserialize_function_call_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?
        .filter(|id| !id.is_empty())
        .unwrap_or_else(default_function_call_id))
}

fn deserialize_status_or_default<'de, D>(deserializer: D) -> Result<MessageStatus, D::Error>
where
    D: Deserializer<'de>,
{
    let opt: Option<MessageStatus> = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or(MessageStatus::Completed))
}

impl TryFrom<&EventPayload> for FunctionToolCall {
    type Error = ExecutorError;

    fn try_from(payload: &EventPayload) -> Result<Self, Self::Error> {
        let EventPayload::OutputItemAdded {
            item_id,
            call_id,
            name,
            namespace,
            ..
        } = payload
        else {
            return Err(ExecutorError::ParseError("expected OutputItemAdded payload".into()));
        };
        let id = if item_id.is_empty() {
            uuid7_str("fc_")
        } else {
            item_id.clone()
        };
        Ok(Self {
            id,
            call_id: call_id.as_deref().unwrap_or_default().to_owned(),
            name: name.as_deref().unwrap_or_default().to_owned(),
            namespace: namespace.clone(),
            arguments: String::new(),
            status: MessageStatus::InProgress,
        })
    }
}

impl TryFrom<&EventPayload> for CustomToolCall {
    type Error = ExecutorError;

    fn try_from(payload: &EventPayload) -> Result<Self, Self::Error> {
        let EventPayload::OutputItemAdded {
            item_id, call_id, name, ..
        } = payload
        else {
            return Err(ExecutorError::ParseError("expected OutputItemAdded payload".into()));
        };
        let id = if item_id.is_empty() {
            uuid7_str("ctc_")
        } else {
            item_id.clone()
        };
        Ok(Self {
            id,
            status: Some(MessageStatus::InProgress),
            call_id: call_id.as_deref().unwrap_or_default().to_owned(),
            name: name.as_deref().unwrap_or_default().to_owned(),
            input: String::new(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GatewayCallStatus {
    InProgress,
    Completed,
    Failed,
}

impl GatewayCallStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

pub type WebSearchCallStatus = GatewayCallStatus;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpCallStatus {
    InProgress,
    Calling,
    Completed,
    Incomplete,
    Failed,
}

impl From<GatewayCallStatus> for McpCallStatus {
    fn from(status: GatewayCallStatus) -> Self {
        match status {
            GatewayCallStatus::InProgress => Self::InProgress,
            GatewayCallStatus::Completed => Self::Completed,
            GatewayCallStatus::Failed => Self::Failed,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchSource {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchActionSearch {
    #[serde(rename = "type")]
    pub type_: String,
    pub query: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<WebSearchSource>,
}

impl WebSearchActionSearch {
    #[must_use]
    pub fn new(query: impl Into<String>, sources: Vec<WebSearchSource>) -> Self {
        Self {
            type_: "search".to_owned(),
            query: query.into(),
            sources,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchCall {
    pub id: String,
    pub status: WebSearchCallStatus,
    pub action: WebSearchActionSearch,
}

impl WebSearchCall {
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        status: WebSearchCallStatus,
        query: impl Into<String>,
        sources: Vec<WebSearchSource>,
    ) -> Self {
        Self {
            id: id.into(),
            status,
            action: WebSearchActionSearch::new(query, sources),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpCallError {
    Text(String),
    ToolExecution(McpToolExecutionError),
    Unknown(Value),
}

impl McpCallError {
    #[must_use]
    pub fn tool_execution(text: impl Into<String>) -> Self {
        Self::ToolExecution(McpToolExecutionError {
            type_: "mcp_tool_execution_error".to_owned(),
            content: vec![McpToolExecutionErrorContent {
                type_: "text".to_owned(),
                text: text.into(),
                annotations: None,
                meta: None,
            }],
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolExecutionError {
    #[serde(rename = "type")]
    pub type_: String,
    pub content: Vec<McpToolExecutionErrorContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolExecutionErrorContent {
    #[serde(rename = "type")]
    pub type_: String,
    pub text: String,
    pub annotations: Option<Value>,
    pub meta: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpCall {
    pub id: String,
    pub server_label: String,
    pub name: String,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<McpCallStatus>,
    pub approval_request_id: Option<String>,
    pub output: Option<String>,
    pub error: Option<McpCallError>,
}

impl McpCall {
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        server_label: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
        status: McpCallStatus,
        output: Option<String>,
        error: Option<McpCallError>,
    ) -> Self {
        Self {
            id: id.into(),
            server_label: server_label.into(),
            name: name.into(),
            arguments: arguments.into(),
            status: Some(status),
            approval_request_id: None,
            output,
            error,
        }
    }
}

impl TryFrom<&EventPayload> for McpCall {
    type Error = ExecutorError;

    fn try_from(payload: &EventPayload) -> Result<Self, Self::Error> {
        let EventPayload::OutputItemAdded { item_id, name, .. } = payload else {
            return Err(ExecutorError::ParseError("expected OutputItemAdded payload".into()));
        };
        let id = if item_id.is_empty() {
            uuid7_str("mcp_")
        } else {
            item_id.clone()
        };
        Ok(Self::new(
            id,
            "",
            name.as_deref().unwrap_or_default(),
            "",
            McpCallStatus::InProgress,
            None,
            None,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTextContent {
    #[serde(rename = "type")]
    pub type_: String,
    pub text: String,
}

impl ReasoningTextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            type_: "reasoning_text".into(),
            text: text.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningOutput {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub content: Vec<ReasoningTextContent>,
    #[serde(default)]
    pub summary: Vec<Value>,
    pub encrypted_content: Option<Value>,
    pub status: Option<String>,
}

impl ReasoningOutput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: vec![],
            summary: vec![],
            encrypted_content: None,
            status: None,
        }
    }
}

impl TryFrom<&EventPayload> for ReasoningOutput {
    type Error = ExecutorError;

    fn try_from(payload: &EventPayload) -> Result<Self, Self::Error> {
        let EventPayload::OutputItemAdded { item_id, .. } = payload else {
            return Err(ExecutorError::ParseError("expected OutputItemAdded payload".into()));
        };
        let id = if item_id.is_empty() {
            uuid7_str("rs_")
        } else {
            item_id.clone()
        };
        Ok(Self::new(id))
    }
}

/// Applies a `*Done` event payload onto an in-flight output item.
///
/// `buffer` holds accumulated delta text/arguments. If the payload's own field
/// is empty the buffer is used as the final value and then cleared; otherwise
/// the buffer is discarded and the payload value is used directly.
pub trait ApplyDone {
    fn apply_done(&mut self, payload: &EventPayload, buffer: &mut String);
}

impl ApplyDone for ReasoningOutput {
    fn apply_done(&mut self, payload: &EventPayload, buffer: &mut String) {
        let EventPayload::ReasoningDone { text, .. } = payload else {
            return;
        };
        let text = if text.is_empty() {
            std::mem::take(buffer)
        } else {
            buffer.clear();
            text.clone()
        };
        if !text.is_empty() {
            self.content.push(ReasoningTextContent::new(text));
        }
    }
}

impl ApplyDone for FunctionToolCall {
    fn apply_done(&mut self, payload: &EventPayload, buffer: &mut String) {
        let EventPayload::FunctionCallArgsDone {
            arguments,
            call_id,
            name,
            ..
        } = payload
        else {
            return;
        };
        self.arguments = if arguments.is_empty() {
            std::mem::take(buffer)
        } else {
            buffer.clear();
            arguments.clone()
        };
        if let Some(cid) = call_id.as_deref().filter(|s| !s.is_empty()) {
            cid.clone_into(&mut self.call_id);
        }
        if !name.is_empty() {
            name.clone_into(&mut self.name);
        }
    }
}

impl ApplyDone for CustomToolCall {
    fn apply_done(&mut self, payload: &EventPayload, buffer: &mut String) {
        let EventPayload::CustomToolCallInputDone { input, .. } = payload else {
            return;
        };
        self.input = if input.is_empty() {
            std::mem::take(buffer)
        } else {
            buffer.clear();
            input.clone()
        };
    }
}

impl ApplyDone for McpCall {
    fn apply_done(&mut self, payload: &EventPayload, _buffer: &mut String) {
        let EventPayload::OutputItemDone { item, .. } = payload else {
            return;
        };
        if let Some(call) = deserialize_from_value_opt(item.clone()) {
            *self = call;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message(OutputMessage),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionToolCall),
    #[serde(rename = "custom_tool_call")]
    CustomToolCall(CustomToolCall),
    #[serde(rename = "web_search_call")]
    WebSearchCall(WebSearchCall),
    #[serde(rename = "mcp_call")]
    McpCall(McpCall),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningOutput),
    #[serde(other)]
    Unknown,
}

impl OutputItem {
    #[must_use]
    pub fn requires_client_action(&self, registry: &ToolRegistry) -> bool {
        match self {
            Self::FunctionCall(call) => registry
                .lookup(&call.name)
                .is_none_or(|entry| !entry.tool_type.is_gateway_owned()),
            Self::CustomToolCall(_) => true,
            Self::Message(_) | Self::WebSearchCall(_) | Self::McpCall(_) | Self::Reasoning(_) | Self::Unknown => false,
        }
    }

    #[must_use]
    pub fn to_input_item(&self) -> Option<InputItem> {
        match self {
            Self::Message(message) => Some(InputItem::Message(message.clone().into())),
            Self::Reasoning(reasoning) => Some(InputItem::Reasoning(reasoning.clone())),
            Self::FunctionCall(call) => Some(InputItem::FunctionCall(InputFunctionToolCall::from(call.clone()))),
            Self::CustomToolCall(call) => Some(InputItem::CustomToolCall(call.clone())),
            Self::WebSearchCall(_) | Self::McpCall(_) | Self::Unknown => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::io::InputItem;

    #[test]
    fn custom_tool_call_preserves_freeform_input_and_requires_client_action() {
        let item: OutputItem = serde_json::from_value(serde_json::json!({
            "id": "ctc_1",
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": "call_1",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch"
        }))
        .unwrap();

        assert!(item.requires_client_action(&ToolRegistry::default()));
        let OutputItem::CustomToolCall(call) = &item else {
            panic!("expected custom tool call");
        };
        assert_eq!(call.status, Some(MessageStatus::Completed));

        let Some(InputItem::CustomToolCall(call)) = item.to_input_item() else {
            panic!("custom call should rehydrate as input");
        };
        assert_eq!(call.name, "apply_patch");
        assert_eq!(call.input, "*** Begin Patch\n*** End Patch");
    }

    #[test]
    fn custom_tool_call_status_remains_optional_on_the_wire() {
        let call: CustomToolCall = serde_json::from_value(serde_json::json!({
            "id": "ctc_1",
            "call_id": "call_1",
            "name": "apply_patch",
            "input": "patch"
        }))
        .unwrap();

        assert_eq!(call.status, None);
        let serialized = serde_json::to_value(call).unwrap();
        assert!(serialized.get("status").is_none());
    }

    #[test]
    fn reasoning_output_round_trips_through_serde() {
        let json = serde_json::json!({
            "id": "rs_abc",
            "type": "reasoning",
            "summary": [],
            "content": [{"text": "Let me think...", "type": "reasoning_text"}],
            "encrypted_content": null,
            "status": null
        });
        let item: OutputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, OutputItem::Reasoning(_)));
        if let OutputItem::Reasoning(r) = &item {
            assert_eq!(r.id, "rs_abc");
            assert_eq!(r.content.len(), 1);
            assert_eq!(r.content[0].text, "Let me think...");
        }
        let serialized = serde_json::to_value(&item).unwrap();
        assert_eq!(serialized["type"], "reasoning");
        assert_eq!(serialized["id"], "rs_abc");
    }

    #[test]
    fn reasoning_input_round_trips_through_serde() {
        let reasoning = ReasoningOutput::new("rs_1");
        let item = InputItem::Reasoning(reasoning);
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "reasoning");
        let back: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(back, InputItem::Reasoning(_)));
    }

    #[test]
    fn mcp_call_serializes_as_openai_output_item() {
        let item = OutputItem::McpCall(McpCall::new(
            "mcp_1",
            "counter",
            "increment",
            "{}",
            McpCallStatus::Completed,
            Some("1".to_owned()),
            None,
        ));

        let json = serde_json::to_value(item).unwrap();
        assert_eq!(json["type"], "mcp_call");
        assert_eq!(json["id"], "mcp_1");
        assert_eq!(json["status"], "completed");
        assert_eq!(json["server_label"], "counter");
        assert_eq!(json["name"], "increment");
        assert_eq!(json["arguments"], "{}");
        assert_eq!(json["output"], "1");
        assert!(json["approval_request_id"].is_null());
        assert!(json["error"].is_null());
    }

    #[test]
    fn vllm_reasoning_response_deserializes() {
        let vllm_output = serde_json::json!([
            {
                "id": "rs_bb637a529f72b88d",
                "summary": [],
                "type": "reasoning",
                "content": [{"text": "2+2 is 4.", "type": "reasoning_text"}],
                "encrypted_content": null,
                "status": null
            },
            {
                "id": "msg_bb68f033f2ed1725",
                "content": [{"annotations": [], "text": "2+2 equals 4.", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message"
            }
        ]);
        let items: Vec<OutputItem> = serde_json::from_value(vllm_output).unwrap();
        assert_eq!(items.len(), 2);
        assert!(matches!(items[0], OutputItem::Reasoning(_)));
        assert!(matches!(items[1], OutputItem::Message(_)));
    }

    #[test]
    fn codex_response_items_round_trip_supported_shapes() {
        let function_call = serde_json::json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "run",
            "namespace": "mcp__shell",
            "arguments": "{\"cmd\":\"pwd\"}",
            "status": "completed"
        });
        let item: OutputItem = serde_json::from_value(function_call).unwrap();
        if let OutputItem::FunctionCall(call) = &item {
            assert_eq!(call.namespace.as_deref(), Some("mcp__shell"));
            assert_eq!(call.name, "run");
        } else {
            panic!("expected function call");
        }
        assert_eq!(serde_json::to_value(&item).unwrap()["namespace"], "mcp__shell");

        let future_item = serde_json::json!({
            "type": "future_item",
            "id": "future_1",
            "payload": {"a": 1}
        });
        let item: OutputItem = serde_json::from_value(future_item).unwrap();
        assert!(matches!(item, OutputItem::Unknown));

        let unknown = serde_json::json!({"type": "new_item", "payload": {"a": 1}});
        let item: InputItem = serde_json::from_value(unknown).unwrap();
        assert!(matches!(item, InputItem::Unknown));
    }

    #[test]
    fn known_items_with_new_nested_content_preserve_message_with_unknown_part() {
        let message = serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": "file_1"
                }
            ]
        });

        let item: InputItem = serde_json::from_value(message).unwrap();
        let InputItem::Message(message) = &item else {
            panic!("expected message item");
        };
        let InputMessageContent::Parts(parts) = &message.content else {
            panic!("expected message parts");
        };
        assert!(matches!(parts.as_slice(), [InputContent::Unknown]));
    }
}
