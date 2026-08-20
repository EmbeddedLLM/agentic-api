use std::borrow::Cow;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::io::{
    FunctionTool, InputItem, InputMessage, InputMessageContent, OutputItem, ResponseUsage, ResponsesInput, ToolChoice,
};
use super::tools::{CodexNamespaceMember, ResponsesTool};
use crate::tool::{CodexNamespaceHandler, CustomHandler, ToolError};
use crate::utils::common::serialize_to_string;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestPayload {
    pub model: String,
    pub input: ResponsesInput,
    pub instructions: Option<String>,
    pub previous_response_id: Option<String>,
    pub conversation_id: Option<String>,
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
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
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Vec<ContextManagement>>,
}

fn default_true() -> bool {
    true
}

/// Structural readiness result returned after validating public tool-search
/// declarations and replay-item wire shapes. The deterministic request-scoped
/// state builder consumes this result and owns ordered-history semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolSearchReadiness {
    Inactive,
    StatePreparationRequired,
}

#[derive(Debug, Serialize)]
pub struct UpstreamRequest<'a> {
    pub model: &'a str,
    pub input: Cow<'a, ResponsesInput>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<&'a str>,
    /// Tools forwarded to vLLM. Function-like declarations are normalized to
    /// ordinary function tools.
    /// Skipped when empty so vLLM does not receive an empty array.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<UpstreamTool>>,
    #[serde(
        skip_serializing_if = "is_absent_or_default_tool_choice",
        serialize_with = "serialize_upstream_tool_choice"
    )]
    pub tool_choice: Option<ToolChoice>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    pub cache_salt: Option<&'a str>,
}

/// A normalized tool declaration supported by the upstream Responses endpoint.
///
/// Gateway and client tool declarations are converted to function tools before
/// entering this upstream-only payload.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum UpstreamTool {
    Function(FunctionTool),
}

// serde's `skip_serializing_if` requires a `&Option<T>` receiver, so the
// idiomatic `Option<&T>` clippy suggests does not apply here.
#[allow(clippy::ref_option)]
fn is_absent_or_default_tool_choice(choice: &Option<ToolChoice>) -> bool {
    choice.as_ref().is_none_or(|choice| matches!(choice, ToolChoice::Auto))
}

// serde's `serialize_with` passes a reference to the field's concrete type.
#[allow(clippy::ref_option)]
fn serialize_upstream_tool_choice<S>(choice: &Option<ToolChoice>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    choice
        .as_ref()
        .map(ToolChoice::normalized_for_upstream)
        .serialize(serializer)
}

impl RequestPayload {
    /// Whether this request contains public tool-search history, an explicit
    /// search declaration, or any declaration whose schema is deferred.
    #[must_use]
    pub fn contains_tool_search_state(&self) -> bool {
        self.input.contains_tool_search_state()
            || self
                .tools
                .as_deref()
                .is_some_and(|tools| tools.iter().any(tool_activates_tool_search))
    }

    /// Validate the retained public tool-search contract without lowering or
    /// executing any declaration.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] for invalid cardinality, reserved-name or
    /// serial-execution conflicts, malformed replay items, and unsupported
    /// dynamically returned declarations.
    pub(crate) fn tool_search_readiness(&self) -> Result<ToolSearchReadiness, ToolError> {
        self.tool_search_readiness_for_input(&self.input)
    }

    pub(crate) fn tool_search_readiness_for_input(
        &self,
        input: &ResponsesInput,
    ) -> Result<ToolSearchReadiness, ToolError> {
        let tools = self.tools.as_deref().unwrap_or_default();
        let declaration_count = tools
            .iter()
            .filter(|tool| matches!(tool, ResponsesTool::ToolSearch(_)))
            .count();
        let input_items = match input {
            ResponsesInput::Text(_) => &[][..],
            ResponsesInput::Items(items) => items.as_slice(),
        };
        let active = input.contains_tool_search_state() || tools.iter().any(tool_activates_tool_search);
        if !active {
            return Ok(ToolSearchReadiness::Inactive);
        }

        if declaration_count > 1 {
            return Err(ToolError::Config(
                "tool search accepts at most one tool_search declaration".to_owned(),
            ));
        }
        if self.parallel_tool_calls == Some(true) {
            return Err(ToolError::Config(
                "parallel_tool_calls must be false when tool search is active".to_owned(),
            ));
        }

        for tool in tools {
            tool.validate()?;
            if let ResponsesTool::Namespace(namespace) = tool {
                validate_tool_search_namespace(namespace)?;
            }
            if has_reserved_tool_search_name(tool) {
                return Err(ToolError::Config(
                    "model-visible tool name 'tool_search' is reserved while tool search is active".to_owned(),
                ));
            }
        }

        for item in input_items {
            match item {
                InputItem::ToolSearchCall(call) => {
                    if call.id.trim().is_empty() {
                        return Err(ToolError::Config("tool_search_call id must not be blank".to_owned()));
                    }
                    if call.call_id.trim().is_empty() {
                        return Err(ToolError::Config(
                            "tool_search_call call_id must not be blank".to_owned(),
                        ));
                    }
                }
                InputItem::ToolSearchOutput(output) => {
                    if output.call_id.trim().is_empty() {
                        return Err(ToolError::Config(
                            "tool_search_output call_id must not be blank".to_owned(),
                        ));
                    }
                    for tool in &output.tools {
                        validate_loaded_tool(tool)?;
                    }
                }
                InputItem::Message(_)
                | InputItem::FunctionCall(_)
                | InputItem::FunctionCallOutput(_)
                | InputItem::CustomToolCall(_)
                | InputItem::CustomToolCallOutput(_)
                | InputItem::Reasoning(_)
                | InputItem::Compaction(_)
                | InputItem::Unknown => {}
            }
        }

        Ok(ToolSearchReadiness::StatePreparationRequired)
    }

    pub(crate) fn ensure_tool_search_ready(&self) -> Result<(), ToolError> {
        match self.tool_search_readiness()? {
            ToolSearchReadiness::Inactive => Ok(()),
            ToolSearchReadiness::StatePreparationRequired => Err(ToolError::Config(
                "tool_search requests require prepared request-scoped state before upstream conversion".to_owned(),
            )),
        }
    }

    /// Construct an `UpstreamRequest` suitable for forwarding to vLLM.
    ///
    /// Codex `namespace` tools' members are first renamed to their flat,
    /// model-visible names via [`CodexNamespaceHandler::resolve_namespace_members`].
    /// Namespace, gateway, and custom tools are then normalized to function
    /// declarations. `tool_choice` is resolved the same way via
    /// [`CodexNamespaceHandler::resolve_tool_choice`].
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when a Codex namespace member's generated
    /// flat name collides with a top-level function tool or another namespace
    /// member, or when a custom tool declares a format whose constrained
    /// decoding cannot be preserved upstream.
    pub fn to_upstream_request(&self, stream: bool) -> Result<UpstreamRequest<'_>, ToolError> {
        self.ensure_tool_search_ready()?;
        let has_built_in_tool = self.declares_built_in_tool();
        if has_built_in_tool && self.parallel_tool_calls == Some(true) {
            return Err(ToolError::Config(
                "parallel_tool_calls must be false when using built-in tools".into(),
            ));
        }
        let parallel_tool_calls = if has_built_in_tool {
            Some(false)
        } else {
            self.parallel_tool_calls
        };

        let renamed_tools = self
            .tools
            .as_deref()
            .map(|tools| CodexNamespaceHandler.resolve_namespace_members(tools))
            .transpose()?;
        if let Some(tools) = &renamed_tools {
            for tool in tools {
                tool.validate()?;
            }
        }
        let tools: Option<Vec<UpstreamTool>> = renamed_tools.map(|tools| {
            tools
                .iter()
                .flat_map(ResponsesTool::to_function_tools)
                .map(UpstreamTool::Function)
                .collect()
        });
        let tools = tools.filter(|tools| !tools.is_empty());
        let namespace_map = CodexNamespaceHandler.build_namespace_map(self.tools.as_deref())?;
        let input = CodexNamespaceHandler.resolve_input(namespace_map.as_ref(), self.input.model_input());
        let tool_choice = CodexNamespaceHandler.resolve_tool_choice(namespace_map.as_ref(), self.tool_choice.as_ref());
        CustomHandler::validate_tool_choice(self.tools.as_deref(), &tool_choice)?;
        Ok(UpstreamRequest {
            model: &self.model,
            input,
            stream,
            instructions: self.instructions.as_deref(),
            tools,
            tool_choice: Some(tool_choice),
            include: self.include.as_ref(),
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_output_tokens,
            truncation: self.truncation.as_deref(),
            metadata: self.metadata.as_ref(),
            parallel_tool_calls,
            cache_salt: self.cache_salt.as_deref(),
        })
    }

    fn declares_built_in_tool(&self) -> bool {
        self.tools
            .as_deref()
            .is_some_and(|tools| tools.iter().any(ResponsesTool::is_gateway_owned))
    }
}

fn has_reserved_tool_search_name(tool: &ResponsesTool) -> bool {
    match tool {
        ResponsesTool::Function(function) => function.name.as_str() == "tool_search",
        ResponsesTool::Custom(custom) => custom.name.as_str() == "tool_search",
        ResponsesTool::Namespace(namespace) => namespace.name == "tool_search",
        ResponsesTool::ToolSearch(_)
        | ResponsesTool::Mcp(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Unknown => false,
    }
}

fn tool_activates_tool_search(tool: &ResponsesTool) -> bool {
    match tool {
        ResponsesTool::ToolSearch(_) => true,
        ResponsesTool::Function(function) => function.defer_loading == Some(true),
        ResponsesTool::Mcp(mcp) => mcp.defer_loading == Some(true),
        ResponsesTool::Namespace(namespace) => namespace.tools.iter().any(
            |member| matches!(member, CodexNamespaceMember::Function(function) if function.defer_loading == Some(true)),
        ),
        ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Custom(_)
        | ResponsesTool::Unknown => false,
    }
}

fn validate_loaded_tool(tool: &ResponsesTool) -> Result<(), ToolError> {
    match tool {
        ResponsesTool::Function(_) | ResponsesTool::Mcp(_) => {}
        ResponsesTool::Namespace(namespace) => validate_tool_search_namespace(namespace)?,
        ResponsesTool::ToolSearch(_)
        | ResponsesTool::Custom(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Unknown => {
            return Err(ToolError::Config(
                "tool_search_output contains an unsupported tool type".to_owned(),
            ));
        }
    }
    tool.validate()?;
    if has_reserved_tool_search_name(tool) {
        return Err(ToolError::Config(
            "loaded model-visible tool name 'tool_search' is reserved".to_owned(),
        ));
    }
    Ok(())
}

fn validate_tool_search_namespace(namespace: &crate::types::tools::CodexNamespaceToolParam) -> Result<(), ToolError> {
    if namespace.tools.is_empty() {
        return Err(ToolError::Config(
            "tool-search namespaces must contain at least one function member".to_owned(),
        ));
    }
    if namespace
        .tools
        .iter()
        .any(|member| !matches!(member, CodexNamespaceMember::Function(_)))
    {
        return Err(ToolError::Config(
            "tool-search namespaces may contain only function members".to_owned(),
        ));
    }
    Ok(())
}

/// Server-side context management configuration for a Responses request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextManagement {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compact_threshold: Option<u64>,
}

/// Request body accepted by `POST /v1/responses/compact`.
#[derive(Debug, Clone, Deserialize)]
pub struct CompactRequest {
    pub model: String,
    #[serde(default)]
    pub input: Option<ResponsesInput>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// Compatibility fields sent by current SDK and Codex clients.
    #[serde(flatten)]
    pub compatibility: HashMap<String, Value>,
}

/// Result returned by `POST /v1/responses/compact`.
#[derive(Debug, Clone, Serialize)]
pub struct CompactedResponse {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub output: Vec<InputItem>,
    pub usage: ResponseUsage,
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
    pub fn as_created_response_chunk(&self) -> String {
        let mut response = self.clone();
        "in_progress".clone_into(&mut response.status);
        let event = json!({
            "type": "response.created",
            "response": response,
        });
        let json_str = serialize_to_string(&event).unwrap_or_else(|_| String::new());
        format!("data: {json_str}\n\n")
    }

    #[must_use]
    pub fn as_responses_chunk(&self) -> String {
        let json_str = serialize_to_string(self).unwrap_or_else(|_| String::new());
        format!("data: {json_str}\n\n")
    }

    #[must_use]
    pub fn as_terminal_response_chunk(&self) -> String {
        let event = json!({
            "type": self.terminal_event_type(),
            "response": self,
        });
        let json_str = serialize_to_string(&event).unwrap_or_else(|_| String::new());
        format!("data: {json_str}\n\n")
    }

    pub(crate) fn terminal_event_type(&self) -> &'static str {
        match self.status.as_str() {
            "incomplete" => "response.incomplete",
            "failed" | "error" => "response.failed",
            "in_progress" => "response.in_progress",
            _ => "response.completed",
        }
    }
}

impl From<&ResponsesInput> for Vec<InputItem> {
    fn from(input: &ResponsesInput) -> Self {
        match input {
            ResponsesInput::Text(text) => vec![InputItem::Message(InputMessage {
                id: None,
                role: "user".into(),
                status: None,
                content: InputMessageContent::Text(text.clone()),
            })],
            ResponsesInput::Items(items) => items
                .iter()
                .filter_map(|item| match item {
                    InputItem::Unknown => None,
                    InputItem::CustomToolCall(call) => Some(InputItem::FunctionCall(call.clone().into())),
                    InputItem::CustomToolCallOutput(output) => {
                        Some(InputItem::FunctionCallOutput(output.clone().into()))
                    }
                    item => Some(item.clone()),
                })
                .collect(),
        }
    }
}

impl From<ResponsesInput> for Vec<InputItem> {
    fn from(input: ResponsesInput) -> Self {
        match input {
            ResponsesInput::Text(text) => vec![InputItem::Message(InputMessage {
                id: None,
                role: "user".into(),
                status: None,
                content: InputMessageContent::Text(text),
            })],
            ResponsesInput::Items(items) => items
                .into_iter()
                .filter_map(|item| match item {
                    InputItem::Unknown => None,
                    InputItem::CustomToolCall(call) => Some(InputItem::FunctionCall(call.into())),
                    InputItem::CustomToolCallOutput(output) => Some(InputItem::FunctionCallOutput(output.into())),
                    item => Some(item),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_search_declaration() -> Value {
        serde_json::json!({
            "type": "tool_search",
            "execution": "client",
            "description": "Find a tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}}
            }
        })
    }

    fn tool_search_request(tools: Vec<Value>, input: Value, parallel_tool_calls: Option<bool>) -> RequestPayload {
        let mut request = serde_json::json!({
            "model": "test",
            "parallel_tool_calls": parallel_tool_calls
        });
        request["input"] = input;
        request["tools"] = Value::Array(tools);
        serde_json::from_value(request).expect("request fixture should deserialize")
    }

    #[test]
    fn tool_search_contract_rejects_request_wide_violations() {
        let cases = [
            (
                "multiple declarations",
                tool_search_request(
                    vec![tool_search_declaration(), tool_search_declaration()],
                    serde_json::json!("hi"),
                    None,
                ),
            ),
            (
                "parallel calling",
                tool_search_request(vec![tool_search_declaration()], serde_json::json!("hi"), Some(true)),
            ),
            (
                "reserved function name",
                tool_search_request(
                    vec![
                        tool_search_declaration(),
                        serde_json::json!({"type": "function", "name": "tool_search"}),
                    ],
                    serde_json::json!("hi"),
                    None,
                ),
            ),
            (
                "reserved custom name",
                tool_search_request(
                    vec![
                        tool_search_declaration(),
                        serde_json::json!({"type": "custom", "name": "tool_search"}),
                    ],
                    serde_json::json!("hi"),
                    None,
                ),
            ),
            (
                "unsupported returned tool",
                tool_search_request(
                    vec![tool_search_declaration()],
                    serde_json::json!([
                        {
                            "type": "tool_search_call",
                            "id": "tsc_1",
                            "call_id": "call_search_1",
                            "arguments": {"query": "weather"}
                        },
                        {
                            "type": "tool_search_output",
                            "call_id": "call_search_1",
                            "tools": [{"type": "custom", "name": "raw"}]
                        }
                    ]),
                    None,
                ),
            ),
        ];

        for (case, request) in cases {
            assert!(
                request.to_upstream_request(false).is_err(),
                "tool-search request should reject {case}"
            );
        }
    }

    fn request_with_returned_tools(returned_tools: Vec<Value>) -> RequestPayload {
        let returned_tools = Value::Array(returned_tools);
        tool_search_request(
            vec![tool_search_declaration()],
            serde_json::json!([
                {
                    "type": "tool_search_call",
                    "id": "tsc_1",
                    "call_id": "call_search_1",
                    "arguments": {"query": "weather"}
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_search_1",
                    "tools": returned_tools
                }
            ]),
            Some(false),
        )
    }

    #[test]
    fn tool_search_contract_accepts_supported_returned_tool_kinds() {
        for returned_tools in [
            vec![],
            vec![serde_json::json!({
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"}
            })],
            vec![serde_json::json!({
                "type": "namespace",
                "name": "weather",
                "tools": [{
                    "type": "function",
                    "name": "forecast",
                    "parameters": {"type": "object"}
                }]
            })],
            vec![serde_json::json!({
                "type": "mcp",
                "server_label": "weather",
                "server_url": "https://mcp.example.test"
            })],
        ] {
            assert_eq!(
                request_with_returned_tools(returned_tools)
                    .tool_search_readiness()
                    .expect("supported returned tools pass contract validation"),
                ToolSearchReadiness::StatePreparationRequired
            );
        }
    }

    #[test]
    fn tool_search_contract_accepts_declaration_free_manual_public_replay() {
        assert_eq!(
            tool_search_request(
                vec![],
                serde_json::json!([
                    {
                        "type": "tool_search_call",
                        "id": "tsc_1",
                        "call_id": "call_search_1",
                        "arguments": {"query": "weather"}
                    },
                    {
                        "type": "tool_search_output",
                        "call_id": "call_search_1",
                        "tools": []
                    }
                ]),
                Some(false),
            )
            .tool_search_readiness()
            .expect("manual public replay is valid without redeclaring tool_search"),
            ToolSearchReadiness::StatePreparationRequired
        );
    }

    #[test]
    fn deferred_declarations_require_tool_search_state_preparation() {
        for tool in [
            serde_json::json!({
                "type": "function",
                "name": "deferred_function",
                "defer_loading": true
            }),
            serde_json::json!({
                "type": "mcp",
                "server_label": "deferred_server",
                "server_url": "https://mcp.example.test/mcp",
                "defer_loading": true
            }),
            serde_json::json!({
                "type": "namespace",
                "name": "deferred_namespace",
                "tools": [{
                    "type": "function",
                    "name": "deferred_member",
                    "defer_loading": true
                }]
            }),
        ] {
            let request = tool_search_request(vec![tool], serde_json::json!("hi"), Some(false));
            assert!(request.contains_tool_search_state());
            assert_eq!(
                request
                    .tool_search_readiness()
                    .expect("deferred declaration is a valid tool-search trigger"),
                ToolSearchReadiness::StatePreparationRequired
            );
            assert!(request.to_upstream_request(false).is_err());
        }
    }

    #[test]
    fn tool_search_contract_rejects_every_unsupported_returned_tool_kind() {
        for returned_tool in [
            serde_json::json!({"type": "web_search_preview"}),
            serde_json::json!({"type": "file_search", "vector_store_ids": []}),
            serde_json::json!({"type": "code_interpreter"}),
            serde_json::json!({"type": "custom", "name": "raw"}),
            tool_search_declaration(),
            serde_json::json!({"type": "future_tool", "opaque": true}),
            serde_json::json!({
                "type": "namespace",
                "name": "mixed",
                "tools": [
                    {"type": "function", "name": "valid"},
                    {"type": "future_member", "opaque": true}
                ]
            }),
        ] {
            assert!(
                request_with_returned_tools(vec![returned_tool])
                    .tool_search_readiness()
                    .is_err(),
                "unsupported returned tool kind must fail validation"
            );
        }
    }

    #[test]
    fn to_upstream_request_rejects_unprepared_tool_search_state() {
        let request = tool_search_request(
            vec![tool_search_declaration()],
            serde_json::json!([
                {
                    "type": "tool_search_call",
                    "id": "tsc_1",
                    "call_id": "call_search_1",
                    "arguments": {"query": "weather"}
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_search_1",
                    "tools": []
                }
            ]),
            Some(false),
        );

        assert!(
            request.to_upstream_request(false).is_err(),
            "unprepared tool-search state must not be silently dropped or lowered"
        );
    }

    #[test]
    fn reserved_tool_search_name_is_allowed_without_tool_search_state() {
        let request = tool_search_request(
            vec![serde_json::json!({"type": "function", "name": "tool_search"})],
            serde_json::json!("hi"),
            Some(true),
        );

        let upstream = serde_json::to_value(
            request
                .to_upstream_request(false)
                .expect("reserved name applies only while tool search is active"),
        )
        .expect("upstream request serializes");
        assert_eq!(upstream["tools"][0]["name"], "tool_search");
        assert_eq!(upstream["parallel_tool_calls"], true);
    }

    #[test]
    fn compact_request_accepts_codex_compatibility_fields() {
        let request: CompactRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "input": [{"role": "user", "content": "hello"}],
            "tools": [],
            "parallel_tool_calls": true,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "low"}
        }))
        .expect("compact request should parse");

        assert_eq!(request.model, "test-model");
        assert!(request.input.is_some());
        assert_eq!(request.compatibility.len(), 4);
    }

    #[test]
    fn request_payload_forwards_cache_salt_upstream() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "input": "hello",
            "cache_salt": "tenant-a"
        }))
        .expect("request should deserialize");

        let upstream = serde_json::to_value(payload.to_upstream_request(false).expect("request should normalize"))
            .expect("upstream request should serialize");

        assert_eq!(upstream["cache_salt"], "tenant-a");
    }

    #[test]
    fn request_payload_uses_option_tool_choice_for_missing_vs_explicit() {
        let absent: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi"
        }))
        .unwrap();
        assert_eq!(absent.tool_choice, None);

        let explicit: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": "none"
        }))
        .unwrap();
        assert_eq!(explicit.tool_choice, Some(ToolChoice::None));
    }

    #[test]
    fn to_upstream_request_carries_instructions_forward() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "instructions": "rules",
            "input": "hi"
        }))
        .unwrap();

        assert_eq!(payload.instructions.as_deref(), Some("rules"));
        assert!(matches!(&payload.input, ResponsesInput::Text(text) if text == "hi"));

        let upstream = payload.to_upstream_request(false).expect("valid upstream request");
        let value = serde_json::to_value(upstream).unwrap();
        assert_eq!(value["instructions"], "rules");
        assert_eq!(value["input"], "hi");
    }

    #[test]
    fn to_upstream_request_preserves_parallel_tool_calls() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "parallel_tool_calls": false
        }))
        .unwrap();

        let upstream = payload.to_upstream_request(false).expect("valid upstream request");
        let value = serde_json::to_value(upstream).unwrap();
        assert_eq!(value["parallel_tool_calls"], false);
    }

    #[test]
    fn to_upstream_request_allows_parallel_tool_calls_for_client_function_tools() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "parallel_tool_calls": true,
            "tools": [{"type": "function", "name": "get_weather"}]
        }))
        .unwrap();

        let upstream = payload
            .to_upstream_request(false)
            .expect("function tools allow parallel calls");
        let value = serde_json::to_value(upstream).unwrap();
        assert_eq!(value["parallel_tool_calls"], true);
    }

    #[test]
    fn to_upstream_request_validates_parallel_tool_calls_for_mixed_tools() {
        for built_in_tool in builtin_tool_declarations() {
            for (parallel_tool_calls, should_reject) in [(false, false), (true, true)] {
                let payload: RequestPayload = serde_json::from_value(serde_json::json!({
                    "model": "test",
                    "input": "hi",
                    "parallel_tool_calls": parallel_tool_calls,
                    "tools": [
                        {"type": "function", "name": "get_weather"},
                        built_in_tool.clone()
                    ]
                }))
                .unwrap();

                let result = payload.to_upstream_request(false);
                if should_reject {
                    let err = result.expect_err("built-in tools should reject parallel tool calls");
                    assert!(err.to_string().contains("parallel_tool_calls must be false"));
                } else {
                    let value =
                        serde_json::to_value(result.expect("mixed built-in and function tools allow serial calls"))
                            .unwrap();
                    assert_eq!(value["parallel_tool_calls"], false);
                }
            }
        }
    }

    #[test]
    fn to_upstream_request_sets_serial_tool_calls_for_builtin_tools() {
        for tool in builtin_tool_declarations() {
            let payload: RequestPayload = serde_json::from_value(serde_json::json!({
                "model": "test",
                "input": "hi",
                "tools": [tool]
            }))
            .unwrap();

            let upstream = payload
                .to_upstream_request(false)
                .expect("built-in tools default to serial tool calls");
            let value = serde_json::to_value(upstream).unwrap();
            assert_eq!(value["parallel_tool_calls"], false);
        }
    }

    #[test]
    fn to_upstream_request_rejects_parallel_tool_calls_for_builtin_tools() {
        for tool in builtin_tool_declarations() {
            let payload: RequestPayload = serde_json::from_value(serde_json::json!({
                "model": "test",
                "input": "hi",
                "parallel_tool_calls": true,
                "tools": [tool]
            }))
            .unwrap();

            let Err(err) = payload.to_upstream_request(false) else {
                panic!("built-in tools should reject parallel_tool_calls=true");
            };

            assert!(err.to_string().contains("parallel_tool_calls must be false"));
        }
    }

    #[test]
    fn to_upstream_request_allows_builtin_tools_with_serial_tool_calls() {
        for tool in builtin_tool_declarations() {
            let payload: RequestPayload = serde_json::from_value(serde_json::json!({
                "model": "test",
                "input": "hi",
                "parallel_tool_calls": false,
                "tools": [tool]
            }))
            .unwrap();

            let upstream = payload
                .to_upstream_request(false)
                .expect("serial built-in tool request is valid");
            let value = serde_json::to_value(upstream).unwrap();
            assert_eq!(value["parallel_tool_calls"], false);
        }
    }

    fn builtin_tool_declarations() -> Vec<Value> {
        vec![
            serde_json::json!({
                "type": "mcp",
                "server_label": "repo",
                "server_url": "http://localhost:9001/mcp"
            }),
            serde_json::json!({"type": "web_search_preview"}),
            serde_json::json!({"type": "file_search", "vector_store_ids": ["vs_abc"]}),
            serde_json::json!({"type": "code_interpreter"}),
        ]
    }

    #[test]
    fn to_upstream_request_flattens_namespace_and_skips_unknown_tools() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__shell",
                    "tools": [
                        {"type": "function", "name": "run", "parameters": {"type": "object"}},
                        {"type": "future_member", "opaque": true}
                    ]
                },
                {"type": "future_tool", "opaque": true}
            ]
        }))
        .unwrap();

        let tools = payload.tools.as_ref().expect("tools should preserve explicit presence");
        assert_eq!(tools.len(), 2);
        let ResponsesTool::Namespace(namespace) = &tools[0] else {
            panic!("expected namespace tool");
        };
        assert_eq!(namespace.tools.len(), 2);

        let upstream = payload.to_upstream_request(false).expect("valid upstream request");
        let value = serde_json::to_value(upstream).unwrap();
        assert_eq!(value["tools"].as_array().expect("upstream tools").len(), 1);
        assert_eq!(value["tools"][0]["name"], "agentic_ns__mcp__shell__run");
    }

    #[test]
    fn to_upstream_request_rejects_namespace_collisions() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tools": [
                {"type": "function", "name": "agentic_ns__mcp__shell__run"},
                {
                    "type": "namespace",
                    "name": "mcp__shell",
                    "tools": [{"type": "function", "name": "run"}]
                }
            ]
        }))
        .unwrap();

        let Err(err) = payload.to_upstream_request(false) else {
            panic!("colliding namespace member should be rejected");
        };

        assert!(err.to_string().contains("collides with a declared function tool"));
    }

    #[test]
    fn to_upstream_request_normalizes_custom_tools_to_functions() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": {
                "type": "custom",
                "name": "apply_patch"
            },
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object"}
                },
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch.",
                    "x-provider-field": {"mode": "strict"}
                }
            ]
        }))
        .unwrap();

        let request = payload.to_upstream_request(false).unwrap();
        let tools = request.tools.as_ref().expect("mixed upstream tools");
        let UpstreamTool::Function(first) = &tools[0];
        let UpstreamTool::Function(second) = &tools[1];
        assert_eq!(first.name, "read_file");
        assert_eq!(second.name, "apply_patch");

        let upstream = serde_json::to_value(request).unwrap();
        assert_eq!(upstream["tools"][0]["type"], "function");
        assert_eq!(upstream["tools"][0]["name"], "read_file");
        assert_eq!(upstream["tools"][1]["type"], "function");
        assert_eq!(upstream["tools"][1]["name"], "apply_patch");
        let custom_description = upstream["tools"][1]["description"]
            .as_str()
            .expect("custom tool description");
        assert!(custom_description.contains("Apply a patch."));
        assert!(custom_description.contains("raw tool input in the `input` string field"));
        assert!(custom_description.contains("x-provider-field"));
        assert_eq!(
            upstream["tools"][1]["parameters"]["properties"]["input"]["type"],
            "string"
        );
        assert_eq!(upstream["tools"][1]["parameters"]["required"][0], "input");
        assert_eq!(upstream["tool_choice"]["type"], "function");
        assert_eq!(upstream["tool_choice"]["name"], "apply_patch");

        let deserialized: FunctionTool =
            serde_json::from_value(upstream["tools"][1].clone()).expect("upstream function tool should deserialize");
        assert_eq!(deserialized.name, "apply_patch");
    }

    #[test]
    fn to_upstream_request_rejects_custom_tool_grammar_formats() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tools": [{
                "type": "custom",
                "name": "constrained_input",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": "start: value"
                }
            }]
        }))
        .expect("request");

        let error = payload
            .to_upstream_request(false)
            .expect_err("unsupported grammar must fail closed");
        assert!(error.to_string().contains("cannot preserve constrained decoding"));
    }

    #[test]
    fn to_upstream_request_normalizes_custom_allowed_tool_choices() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [
                    {"type": "function", "name": "read_file"},
                    {"type": "custom", "name": "apply_patch"}
                ]
            },
            "tools": [
                {"type": "function", "name": "read_file"},
                {"type": "custom", "name": "apply_patch"}
            ]
        }))
        .unwrap();

        let public_choice = serde_json::to_value(payload.tool_choice.as_ref().unwrap()).unwrap();
        assert_eq!(public_choice["tools"][1]["type"], "custom");

        let upstream = serde_json::to_value(payload.to_upstream_request(false).unwrap()).unwrap();
        assert_eq!(upstream["tool_choice"]["type"], "allowed_tools");
        assert_eq!(upstream["tool_choice"]["mode"], "required");
        assert_eq!(upstream["tool_choice"]["tools"][0]["type"], "function");
        assert_eq!(upstream["tool_choice"]["tools"][1]["type"], "function");
        assert_eq!(upstream["tool_choice"]["tools"][1]["name"], "apply_patch");
    }

    #[test]
    fn to_upstream_request_rejects_custom_choice_for_function_declaration() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": {"type": "custom", "name": "echo"},
            "tools": [{"type": "function", "name": "echo"}]
        }))
        .expect("request");

        let error = payload
            .to_upstream_request(false)
            .expect_err("a custom selector must match a custom declaration");
        assert!(error.to_string().contains("no matching custom tool is declared"));
    }

    #[test]
    fn to_upstream_request_rejects_unknown_custom_allowed_tool() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "custom", "name": "missing"}]
            },
            "tools": [{"type": "custom", "name": "apply_patch"}]
        }))
        .expect("request");

        let error = payload
            .to_upstream_request(false)
            .expect_err("an allowed custom selector must match a custom declaration");
        assert!(error.to_string().contains("no matching custom tool is declared"));
    }

    #[test]
    fn responses_input_discards_unknown_items_when_converted_for_storage() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "future_item", "payload": {"a": 1}}
        ]))
        .unwrap();

        let items = Vec::<InputItem>::from(&input);
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], InputItem::Message(_)));
    }

    #[test]
    fn response_payload_terminal_chunk_uses_status_specific_event_type() {
        let mut payload = ResponsePayload {
            id: "resp_test".to_string(),
            object: "response".to_string(),
            created_at: 0,
            model: "test-model".to_string(),
            status: "completed".to_string(),
            output: Vec::new(),
            usage: None,
            incomplete_details: None,
            error: None,
            previous_response_id: None,
            conversation_id: None,
            instructions: None,
        };

        for (status, expected_type) in [
            ("completed", "response.completed"),
            ("incomplete", "response.incomplete"),
            ("failed", "response.failed"),
            ("error", "response.failed"),
            ("in_progress", "response.in_progress"),
        ] {
            payload.status = status.to_string();
            let chunk = payload.as_terminal_response_chunk();
            let data = chunk.trim().strip_prefix("data: ").unwrap();
            let event: Value = serde_json::from_str(data).unwrap();
            assert_eq!(event["type"], expected_type);
            assert_eq!(event["response"]["status"], status);
        }
    }

    #[test]
    fn response_payload_created_chunk_uses_in_progress_status() {
        let payload = ResponsePayload {
            id: "resp_test".to_string(),
            object: "response".to_string(),
            created_at: 0,
            model: "test-model".to_string(),
            status: "completed".to_string(),
            output: Vec::new(),
            usage: None,
            incomplete_details: None,
            error: None,
            previous_response_id: None,
            conversation_id: None,
            instructions: None,
        };

        let chunk = payload.as_created_response_chunk();
        let data = chunk.trim().strip_prefix("data: ").unwrap();
        let event: Value = serde_json::from_str(data).unwrap();
        assert_eq!(event["type"], "response.created");
        assert_eq!(event["response"]["id"], "resp_test");
        assert_eq!(event["response"]["status"], "in_progress");
    }
}
