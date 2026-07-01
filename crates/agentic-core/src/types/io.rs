use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de, ser::SerializeMap};
use serde_json::Value;

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

#[derive(Debug, Clone)]
pub enum InputItem {
    Message(InputMessage),
    FunctionCall(FunctionToolCall),
    FunctionCallOutput(FunctionToolResultMessage),
    ToolSearchCall(Value),
    CustomToolCall(Value),
    Reasoning(ReasoningOutput),
    Unknown(Value),
}

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

impl From<OutputMessage> for InputMessage {
    fn from(msg: OutputMessage) -> Self {
        let parts = msg
            .content
            .into_iter()
            .map(|c| {
                InputContent::Text(InputTextContent {
                    type_: c.type_,
                    text: c.text,
                })
            })
            .collect();
        Self {
            role: msg.role,
            content: InputMessageContent::Parts(parts),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolCall {
    #[serde(default)]
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
    #[serde(default)]
    pub status: String,
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

#[derive(Debug, Clone)]
pub enum OutputItem {
    Message(OutputMessage),
    FunctionCall(FunctionToolCall),
    ToolSearchCall(Value),
    CustomToolCall(Value),
    Reasoning(ReasoningOutput),
    Unknown(Value),
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct InputTokenDetails {
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct OutputTokenDetails {
    pub reasoning_tokens: i64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub total_tokens: i64,
    #[serde(default)]
    pub input_tokens_details: InputTokenDetails,
    #[serde(default)]
    pub output_tokens_details: OutputTokenDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesTool {
    Known(KnownResponsesTool),
    Unknown(Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum KnownResponsesTool {
    #[serde(rename = "function")]
    Function(ResponsesFunctionTool),
    #[serde(rename = "namespace")]
    Namespace(CodexNamespaceTool),
    #[serde(rename = "tool_search")]
    ToolSearch(CodexToolSearchTool),
    #[serde(rename = "custom")]
    Custom(CodexCustomTool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesFunctionTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    #[serde(default)]
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

pub type FunctionTool = ResponsesFunctionTool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexNamespaceTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub tools: Vec<CodexNamespaceMember>,
    #[serde(default)]
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CodexNamespaceMember {
    Function(ResponsesFunctionTool),
    Unknown(Value),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSearchExecution {
    Server,
    Client,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexToolSearchTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<ToolSearchExecution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(default)]
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexCustomTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    #[serde(default)]
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolName {
    pub namespace: Option<String>,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolExecutionOwner {
    Client,
    Gateway,
    Provider,
}

#[derive(Debug, Clone)]
pub struct ToolRegistryEntry {
    pub owner: ToolExecutionOwner,
    pub original_type: String,
    pub original_name: ToolName,
    pub model_visible_tool_name: ToolName,
    pub original_tool: Value,
}

#[derive(Debug, Clone, Default)]
pub struct ToolRegistry {
    pub by_model_name: HashMap<ToolName, ToolRegistryEntry>,
}

fn value_with_type<T: Serialize>(type_name: &str, value: &T) -> Result<Value, serde_json::Error> {
    let mut value = serde_json::to_value(value)?;
    if let Value::Object(map) = &mut value {
        map.insert("type".to_string(), Value::String(type_name.to_string()));
    }
    Ok(value)
}

#[allow(clippy::trivially_copy_pass_by_ref)]
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

impl Serialize for OutputItem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Message(item) => serialize_typed(serializer, "message", item),
            Self::FunctionCall(item) => serialize_typed(serializer, "function_call", item),
            Self::ToolSearchCall(item) => raw_value_with_type("tool_search_call", item).serialize(serializer),
            Self::CustomToolCall(item) => raw_value_with_type("custom_tool_call", item).serialize(serializer),
            Self::Reasoning(item) => serialize_typed(serializer, "reasoning", item),
            Self::Unknown(item) => item.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for OutputItem {
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
            "tool_search_call" => Ok(Self::ToolSearchCall(value)),
            "custom_tool_call" => Ok(Self::CustomToolCall(value)),
            "reasoning" => Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::Reasoning)),
            _ => Ok(Self::Unknown(value)),
        }
    }
}

impl ResponsesTool {
    #[must_use]
    pub fn original_type(&self) -> Option<&str> {
        match self {
            Self::Known(KnownResponsesTool::Function(_)) => Some("function"),
            Self::Known(KnownResponsesTool::Namespace(_)) => Some("namespace"),
            Self::Known(KnownResponsesTool::ToolSearch(_)) => Some("tool_search"),
            Self::Known(KnownResponsesTool::Custom(_)) => Some("custom"),
            Self::Unknown(value) => value.get("type").and_then(Value::as_str),
        }
    }

    #[must_use]
    pub fn to_raw_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

const MODEL_VISIBLE_NAMESPACE_MEMBER_PREFIX: &str = "agentic_ns__";

pub(crate) fn model_visible_namespace_member_name(namespace: &str, member: &str) -> String {
    format!("{MODEL_VISIBLE_NAMESPACE_MEMBER_PREFIX}{namespace}__{member}")
}

pub(crate) fn legacy_model_visible_namespace_member_name(namespace: &str, member: &str) -> String {
    format!("{namespace}.{member}")
}

pub(crate) fn alternate_model_visible_namespace_member_name(namespace: &str, member: &str) -> String {
    legacy_model_visible_namespace_member_name(namespace, member).replace('.', "_")
}

pub(crate) fn flatten_tools_for_upstream(tools: Option<&[ResponsesTool]>) -> Option<Vec<ResponsesTool>> {
    tools.map(|tools| {
        let top_level_names = top_level_tool_names(tools);
        let mut upstream_tools = Vec::with_capacity(tools.len());
        for tool in tools {
            match tool {
                ResponsesTool::Known(KnownResponsesTool::Namespace(namespace)) => {
                    if namespace_has_flat_name_collision(namespace, &top_level_names) {
                        upstream_tools.push(tool.clone());
                        continue;
                    }

                    let mut emitted_members = false;
                    for member in &namespace.tools {
                        if let CodexNamespaceMember::Function(function) = member {
                            let flat_name = model_visible_namespace_member_name(&namespace.name, &function.name);
                            let mut function = function.clone();
                            function.name = flat_name;
                            upstream_tools.push(ResponsesTool::Known(KnownResponsesTool::Function(function)));
                            emitted_members = true;
                        }
                    }
                    if !emitted_members {
                        upstream_tools.push(tool.clone());
                    }
                }
                ResponsesTool::Known(
                    KnownResponsesTool::Function(_) | KnownResponsesTool::ToolSearch(_) | KnownResponsesTool::Custom(_),
                )
                | ResponsesTool::Unknown(_) => upstream_tools.push(tool.clone()),
            }
        }
        upstream_tools
    })
}

impl ToolRegistry {
    #[must_use]
    pub fn from_tools(tools: Option<&[ResponsesTool]>) -> Self {
        let mut registry = Self::default();
        let Some(tools) = tools else {
            return registry;
        };

        for tool in tools {
            match tool {
                ResponsesTool::Known(KnownResponsesTool::Function(function)) => {
                    let name = ToolName {
                        namespace: None,
                        name: function.name.clone(),
                    };
                    registry.insert(
                        name.clone(),
                        ToolExecutionOwner::Client,
                        "function",
                        name,
                        tool.to_raw_value(),
                    );
                }
                ResponsesTool::Known(KnownResponsesTool::Namespace(namespace)) => {
                    for member in &namespace.tools {
                        if let CodexNamespaceMember::Function(function) = member {
                            let name = ToolName {
                                namespace: Some(namespace.name.clone()),
                                name: function.name.clone(),
                            };
                            registry.insert(
                                name.clone(),
                                ToolExecutionOwner::Client,
                                "namespace",
                                name,
                                tool.to_raw_value(),
                            );
                        }
                    }
                }
                ResponsesTool::Known(KnownResponsesTool::Custom(custom)) => {
                    let name = ToolName {
                        namespace: None,
                        name: custom.name.clone(),
                    };
                    registry.insert(
                        name.clone(),
                        ToolExecutionOwner::Client,
                        "custom",
                        name,
                        tool.to_raw_value(),
                    );
                }
                ResponsesTool::Known(KnownResponsesTool::ToolSearch(_)) | ResponsesTool::Unknown(_) => {}
            }
        }

        registry
    }

    fn insert(
        &mut self,
        model_visible_tool_name: ToolName,
        owner: ToolExecutionOwner,
        original_type: &str,
        original_name: ToolName,
        original_tool: Value,
    ) {
        self.by_model_name.insert(
            model_visible_tool_name.clone(),
            ToolRegistryEntry {
                owner,
                original_type: original_type.to_string(),
                original_name,
                model_visible_tool_name,
                original_tool,
            },
        );
    }

    #[must_use]
    pub fn owner_for(&self, name: &ToolName) -> Option<&ToolExecutionOwner> {
        self.by_model_name.get(name).map(|entry| &entry.owner)
    }
}

impl FunctionToolCall {
    #[must_use]
    pub fn tool_name(&self) -> ToolName {
        ToolName {
            namespace: self.namespace.clone(),
            name: self.name.clone(),
        }
    }
}

pub(crate) fn normalize_output_items_with_tools(output: &mut [OutputItem], tools: Option<&[ResponsesTool]>) {
    for item in output {
        if let OutputItem::FunctionCall(call) = item {
            normalize_function_call_with_tools(call, tools);
        }
    }
}

fn normalize_function_call_with_tools(call: &mut FunctionToolCall, tools: Option<&[ResponsesTool]>) {
    if call.namespace.is_some() {
        return;
    }
    let Some(tools) = tools else {
        return;
    };

    if let Some((namespace, function)) = namespace_member_call(&call.name, tools) {
        apply_namespace_member_call(call, namespace, function);
        return;
    }

    if let Some((namespace, function)) = namespace_container_call(&call.name, tools) {
        apply_namespace_member_call(call, namespace, function);
        strip_namespace_container_arguments(&mut call.arguments);
        return;
    }

    if let Some((namespace, function)) = unambiguous_namespace_member_call(&call.name, tools) {
        apply_namespace_member_call(call, namespace, function);
    }
}

fn apply_namespace_member_call(call: &mut FunctionToolCall, namespace: &CodexNamespaceTool, function: &FunctionTool) {
    call.namespace = Some(namespace.name.clone());
    call.name.clone_from(&function.name);
}

fn record_unique_namespace_match<'a>(
    found: &mut Option<(&'a CodexNamespaceTool, &'a FunctionTool)>,
    ambiguous: &mut bool,
    candidate: (&'a CodexNamespaceTool, &'a FunctionTool),
) {
    if found.is_some() {
        *ambiguous = true;
    } else {
        *found = Some(candidate);
    }
}

fn namespace_member_call<'a>(
    call_name: &str,
    tools: &'a [ResponsesTool],
) -> Option<(&'a CodexNamespaceTool, &'a FunctionTool)> {
    if tools.iter().any(|tool| top_level_tool_named(tool, call_name)) {
        return None;
    }

    let mut exact = None;
    let mut exact_ambiguous = false;
    let mut legacy = None;
    let mut legacy_ambiguous = false;
    let mut alternate = None;
    let mut alternate_ambiguous = false;

    for tool in tools {
        let ResponsesTool::Known(KnownResponsesTool::Namespace(namespace)) = tool else {
            continue;
        };
        if namespace_flat_name_collides(namespace, tools) {
            continue;
        }
        for member in &namespace.tools {
            let CodexNamespaceMember::Function(function) = member else {
                continue;
            };
            if call_name == model_visible_namespace_member_name(&namespace.name, &function.name) {
                record_unique_namespace_match(&mut exact, &mut exact_ambiguous, (namespace, function));
            }
            if call_name == legacy_model_visible_namespace_member_name(&namespace.name, &function.name) {
                record_unique_namespace_match(&mut legacy, &mut legacy_ambiguous, (namespace, function));
            }
            if call_name == alternate_model_visible_namespace_member_name(&namespace.name, &function.name) {
                record_unique_namespace_match(&mut alternate, &mut alternate_ambiguous, (namespace, function));
            }
        }
    }

    if !exact_ambiguous {
        if let Some(found) = exact {
            return Some(found);
        }
    }
    if !legacy_ambiguous {
        if let Some(found) = legacy {
            return Some(found);
        }
    }
    if alternate_ambiguous { None } else { alternate }
}

fn namespace_member_call_by_namespace<'a>(
    namespace_name: &str,
    member_name: &str,
    tools: &'a [ResponsesTool],
) -> Option<(&'a CodexNamespaceTool, &'a FunctionTool)> {
    for tool in tools {
        let ResponsesTool::Known(KnownResponsesTool::Namespace(namespace)) = tool else {
            continue;
        };
        if namespace.name != namespace_name {
            continue;
        }
        for member in &namespace.tools {
            let CodexNamespaceMember::Function(function) = member else {
                continue;
            };
            if function.name == member_name {
                return Some((namespace, function));
            }
        }
    }
    None
}

fn top_level_tool_named(tool: &ResponsesTool, name: &str) -> bool {
    match tool {
        ResponsesTool::Known(KnownResponsesTool::Function(function)) => function.name == name,
        ResponsesTool::Known(KnownResponsesTool::Custom(custom)) => custom.name == name,
        ResponsesTool::Unknown(value) => value.get("name").and_then(Value::as_str) == Some(name),
        ResponsesTool::Known(KnownResponsesTool::Namespace(_) | KnownResponsesTool::ToolSearch(_)) => false,
    }
}

fn top_level_tool_names(tools: &[ResponsesTool]) -> HashSet<String> {
    tools
        .iter()
        .filter_map(|tool| match tool {
            ResponsesTool::Known(KnownResponsesTool::Function(function)) => Some(function.name.clone()),
            ResponsesTool::Known(KnownResponsesTool::Custom(custom)) => Some(custom.name.clone()),
            ResponsesTool::Unknown(value) => value.get("name").and_then(Value::as_str).map(str::to_string),
            ResponsesTool::Known(KnownResponsesTool::Namespace(_) | KnownResponsesTool::ToolSearch(_)) => None,
        })
        .collect()
}

fn namespace_has_flat_name_collision(namespace: &CodexNamespaceTool, top_level_names: &HashSet<String>) -> bool {
    namespace.tools.iter().any(|member| match member {
        CodexNamespaceMember::Function(function) => {
            top_level_names.contains(&model_visible_namespace_member_name(&namespace.name, &function.name))
        }
        CodexNamespaceMember::Unknown(_) => false,
    })
}

fn namespace_member_flat_name_collides(
    namespace: &CodexNamespaceTool,
    function: &FunctionTool,
    tools: &[ResponsesTool],
) -> bool {
    let flat_name = model_visible_namespace_member_name(&namespace.name, &function.name);
    tools.iter().any(|tool| top_level_tool_named(tool, &flat_name))
}

fn namespace_flat_name_collides(namespace: &CodexNamespaceTool, tools: &[ResponsesTool]) -> bool {
    namespace.tools.iter().any(|member| match member {
        CodexNamespaceMember::Function(function) => namespace_member_flat_name_collides(namespace, function, tools),
        CodexNamespaceMember::Unknown(_) => false,
    })
}

fn namespace_container_call<'a>(
    call_name: &str,
    tools: &'a [ResponsesTool],
) -> Option<(&'a CodexNamespaceTool, &'a FunctionTool)> {
    if tools.iter().any(|tool| top_level_tool_named(tool, call_name)) {
        return None;
    }

    let mut found = None;
    let mut ambiguous = false;
    for tool in tools {
        let ResponsesTool::Known(KnownResponsesTool::Namespace(namespace)) = tool else {
            continue;
        };
        if call_name != namespace.name {
            continue;
        }
        if namespace_flat_name_collides(namespace, tools) {
            continue;
        }

        let mut function_members = namespace.tools.iter().filter_map(|member| match member {
            CodexNamespaceMember::Function(function) => Some(function),
            CodexNamespaceMember::Unknown(_) => None,
        });
        let Some(function) = function_members.next() else {
            continue;
        };
        if function_members.next().is_some() {
            continue;
        }

        record_unique_namespace_match(&mut found, &mut ambiguous, (namespace, function));
    }

    if ambiguous { None } else { found }
}

fn unambiguous_namespace_member_call<'a>(
    call_name: &str,
    tools: &'a [ResponsesTool],
) -> Option<(&'a CodexNamespaceTool, &'a FunctionTool)> {
    let mut found = None;
    for tool in tools {
        if top_level_tool_named(tool, call_name) {
            return None;
        }
        let ResponsesTool::Known(KnownResponsesTool::Namespace(namespace)) = tool else {
            continue;
        };
        if namespace_flat_name_collides(namespace, tools) {
            continue;
        }
        for member in &namespace.tools {
            let CodexNamespaceMember::Function(function) = member else {
                continue;
            };
            if function.name != call_name {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some((namespace, function));
        }
    }
    found
}

fn strip_namespace_container_arguments(arguments: &mut String) {
    let Ok(mut value) = serde_json::from_str::<Value>(arguments) else {
        return;
    };
    let Some(object) = value.as_object_mut() else {
        return;
    };
    if object.remove("tools").is_some() {
        *arguments = serde_json::to_string(&value).unwrap_or_else(|_| std::mem::take(arguments));
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

impl OutputItem {
    #[must_use]
    pub fn requires_client_action(&self, registry: &ToolRegistry) -> bool {
        match self {
            Self::FunctionCall(call) => {
                !matches!(registry.owner_for(&call.tool_name()), Some(ToolExecutionOwner::Gateway))
            }
            Self::ToolSearchCall(value) => value
                .get("execution")
                .and_then(Value::as_str)
                .is_some_and(|execution| execution == "client"),
            Self::CustomToolCall(_) => true,
            Self::Message(_) | Self::Reasoning(_) | Self::Unknown(_) => false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function {
        namespace: Option<String>,
        name: String,
    },
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function { namespace, name } => {
                let mut map = serializer.serialize_map(Some(2 + usize::from(namespace.is_some())))?;
                map.serialize_entry("type", "function")?;
                if let Some(namespace) = namespace {
                    map.serialize_entry("namespace", namespace)?;
                }
                map.serialize_entry("name", name)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(choice) => match choice.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::None),
                "required" => Ok(Self::Required),
                other => Err(de::Error::unknown_variant(
                    other,
                    &["auto", "none", "required", "function"],
                )),
            },
            Value::Object(object) => {
                if object.get("type").and_then(Value::as_str) == Some("function") {
                    let namespace = object.get("namespace").and_then(Value::as_str).map(str::to_string);
                    let name = object
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| de::Error::missing_field("name"))?;
                    return Ok(Self::Function {
                        namespace,
                        name: name.to_string(),
                    });
                }

                if let Some(function) = object.get("function").and_then(Value::as_object) {
                    let namespace = function.get("namespace").and_then(Value::as_str).map(str::to_string);
                    let name = function
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| de::Error::missing_field("name"))?;
                    return Ok(Self::Function {
                        namespace,
                        name: name.to_string(),
                    });
                }

                Err(de::Error::custom("expected tool_choice string or function object"))
            }
            _ => Err(de::Error::custom("expected tool_choice string or function object")),
        }
    }
}

#[must_use]
pub(crate) fn flatten_tool_choice_for_upstream(choice: &ToolChoice, tools: Option<&[ResponsesTool]>) -> ToolChoice {
    let ToolChoice::Function { namespace, name } = choice else {
        return choice.clone();
    };
    let Some(tools) = tools else {
        return choice.clone();
    };

    let resolved = if let Some(namespace) = namespace {
        namespace_member_call_by_namespace(namespace, name, tools)
            .filter(|(namespace, _function)| !namespace_flat_name_collides(namespace, tools))
    } else {
        namespace_member_call(name, tools)
            .or_else(|| namespace_container_call(name, tools))
            .or_else(|| unambiguous_namespace_member_call(name, tools))
    };
    let Some((namespace, function)) = resolved else {
        return choice.clone();
    };

    ToolChoice::Function {
        namespace: None,
        name: model_visible_namespace_member_name(&namespace.name, &function.name),
    }
}

/// Returns the effective tool list, preferring `request_tools` when explicitly
/// set by the caller, otherwise falling back to the stored configuration.
#[inline]
pub(crate) fn resolve_tools(
    request_tools: Option<&[ResponsesTool]>,
    stored_tools: Option<&[ResponsesTool]>,
    tools_explicitly_set: bool,
) -> Option<Vec<ResponsesTool>> {
    if tools_explicitly_set {
        request_tools
    } else {
        stored_tools
    }
    .map(<[_]>::to_vec)
}

/// Returns the effective tool choice using the same precedence as [`resolve_tools`].
#[inline]
pub(crate) fn resolve_tool_choice(
    request_choice: &ToolChoice,
    stored_choice: &ToolChoice,
    explicitly_set: bool,
) -> ToolChoice {
    if explicitly_set { request_choice } else { stored_choice }.clone()
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn codex_tool_shapes_round_trip() {
        let tools_json = serde_json::json!([
            {
                "type": "function",
                "name": "run",
                "description": "Run command",
                "parameters": {"type": "object"},
                "strict": true,
                "x-extra": "kept"
            },
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [
                    {"type": "function", "name": "run", "parameters": {"type": "object"}}
                ]
            },
            {
                "type": "tool_search",
                "execution": "client",
                "parameters": {"type": "object"}
            },
            {
                "type": "custom",
                "name": "apply_patch",
                "format": {"type": "grammar"},
                "defer_loading": true
            },
            {
                "type": "future_tool",
                "opaque": true
            }
        ]);

        let tools: Vec<ResponsesTool> = serde_json::from_value(tools_json).unwrap();
        assert!(matches!(
            tools[0],
            ResponsesTool::Known(KnownResponsesTool::Function(_))
        ));
        assert!(matches!(
            tools[1],
            ResponsesTool::Known(KnownResponsesTool::Namespace(_))
        ));
        assert!(matches!(
            tools[2],
            ResponsesTool::Known(KnownResponsesTool::ToolSearch(_))
        ));
        assert!(matches!(tools[3], ResponsesTool::Known(KnownResponsesTool::Custom(_))));
        assert!(matches!(tools[4], ResponsesTool::Unknown(_)));

        let serialized = serde_json::to_value(&tools).unwrap();
        assert_eq!(serialized[0]["x-extra"], "kept");
        assert_eq!(serialized[1]["tools"][0]["type"], "function");
        assert_eq!(serialized[4]["opaque"], true);
    }

    #[test]
    fn codex_response_items_round_trip_raw_shapes() {
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
            assert_eq!(call.tool_name().namespace.as_deref(), Some("mcp__shell"));
            assert_eq!(call.name, "run");
        } else {
            panic!("expected function call");
        }
        assert_eq!(serde_json::to_value(&item).unwrap()["namespace"], "mcp__shell");

        let custom_call = serde_json::json!({
            "type": "custom_tool_call",
            "id": "ctc_1",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch\n"
        });
        let item: OutputItem = serde_json::from_value(custom_call).unwrap();
        assert!(matches!(item, OutputItem::CustomToolCall(_)));
        assert_eq!(
            serde_json::to_value(&item).unwrap()["input"],
            "*** Begin Patch\n*** End Patch\n"
        );

        let unknown = serde_json::json!({"type": "new_item", "payload": {"a": 1}});
        let item: InputItem = serde_json::from_value(unknown).unwrap();
        assert!(matches!(item, InputItem::Unknown(_)));
        assert_eq!(serde_json::to_value(&item).unwrap()["payload"]["a"], 1);
    }

    #[test]
    fn known_items_with_new_nested_shapes_fall_back_to_raw() {
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
        assert!(matches!(item, InputItem::Unknown(_)));
        assert_eq!(serde_json::to_value(&item).unwrap()["content"][0]["type"], "input_file");
    }

    #[test]
    fn developer_messages_normalize_to_system_for_upstream() {
        let mut typed = ResponsesInput::Items(vec![InputItem::Message(InputMessage {
            role: "developer".to_string(),
            content: InputMessageContent::Text("rules".to_string()),
        })]);
        typed.normalize_for_upstream();
        if let ResponsesInput::Items(items) = typed {
            if let InputItem::Message(message) = &items[0] {
                assert_eq!(message.role, "system");
            } else {
                panic!("expected message");
            }
        }

        let mut raw = ResponsesInput::Items(vec![InputItem::Unknown(serde_json::json!({
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_file", "file_id": "file_1"}]
        }))]);
        raw.normalize_for_upstream();
        assert_eq!(serde_json::to_value(&raw).unwrap()[0]["role"], "system");

        let mut raw_without_type = ResponsesInput::Items(vec![InputItem::Unknown(serde_json::json!({
            "role": "developer",
            "content": "rules"
        }))]);
        raw_without_type.normalize_for_upstream();
        assert_eq!(serde_json::to_value(&raw_without_type).unwrap()[0]["role"], "system");
    }

    #[test]
    fn system_messages_move_to_front_for_upstream() {
        let mut input = ResponsesInput::Items(vec![
            InputItem::Message(InputMessage {
                role: "user".to_string(),
                content: InputMessageContent::Text("hi".to_string()),
            }),
            InputItem::Message(InputMessage {
                role: "developer".to_string(),
                content: InputMessageContent::Text("rules".to_string()),
            }),
        ]);

        input.normalize_for_upstream();
        let value = serde_json::to_value(&input).unwrap();
        assert_eq!(value[0]["role"], "system");
        assert_eq!(value[1]["role"], "user");
    }

    #[test]
    fn prepend_system_text_converts_text_input_to_items() {
        let mut input = ResponsesInput::Text("hi".to_string());
        input.prepend_system_text("rules".to_string());
        let value = serde_json::to_value(&input).unwrap();

        assert_eq!(value[0]["role"], "system");
        assert_eq!(value[0]["content"], "rules");
        assert_eq!(value[1]["role"], "user");
        assert_eq!(value[1]["content"], "hi");
    }

    #[test]
    fn tool_choice_function_uses_openai_shape() {
        let choice: ToolChoice = serde_json::from_value(serde_json::json!({
            "type": "function",
            "name": "run"
        }))
        .unwrap();

        assert_eq!(
            choice,
            ToolChoice::Function {
                namespace: None,
                name: "run".to_string()
            }
        );
        assert_eq!(
            serde_json::to_value(&choice).unwrap(),
            serde_json::json!({"type": "function", "name": "run"})
        );
    }

    #[test]
    fn function_tool_choice_flattens_unambiguous_namespace_member() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();
        let choice = ToolChoice::Function {
            namespace: None,
            name: "run".to_string(),
        };

        assert_eq!(
            flatten_tool_choice_for_upstream(&choice, Some(&tools)),
            ToolChoice::Function {
                namespace: None,
                name: "agentic_ns__mcp__shell__run".to_string()
            }
        );
    }

    #[test]
    fn namespaced_function_tool_choice_flattens_exact_member() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            },
            {
                "type": "namespace",
                "name": "mcp__git",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();
        let choice: ToolChoice = serde_json::from_value(serde_json::json!({
            "type": "function",
            "namespace": "mcp__git",
            "name": "run"
        }))
        .unwrap();

        assert_eq!(
            flatten_tool_choice_for_upstream(&choice, Some(&tools)),
            ToolChoice::Function {
                namespace: None,
                name: "agentic_ns__mcp__git__run".to_string()
            }
        );
    }

    #[test]
    fn flatten_tools_does_not_generate_colliding_namespace_member_name() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {"type": "function", "name": "agentic_ns__mcp__shell__run"},
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();

        let upstream = flatten_tools_for_upstream(Some(&tools)).expect("tools");
        let flat_function_count = upstream
            .iter()
            .filter(|tool| matches!(tool, ResponsesTool::Known(KnownResponsesTool::Function(function)) if function.name == "agentic_ns__mcp__shell__run"))
            .count();

        assert_eq!(flat_function_count, 1);
        assert!(upstream.iter().any(|tool| matches!(
            tool,
            ResponsesTool::Known(KnownResponsesTool::Namespace(namespace)) if namespace.name == "mcp__shell"
        )));
    }

    #[test]
    fn flatten_tools_keeps_colliding_namespace_whole_while_flattening_others() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {"type": "function", "name": "agentic_ns__mcp__shell__run"},
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [
                    {"type": "function", "name": "run"},
                    {"type": "function", "name": "status"}
                ]
            },
            {
                "type": "namespace",
                "name": "mcp__git",
                "tools": [{"type": "function", "name": "status"}]
            }
        ]))
        .unwrap();

        let upstream = flatten_tools_for_upstream(Some(&tools)).expect("tools");
        let shell_namespace = upstream
            .iter()
            .find_map(|tool| match tool {
                ResponsesTool::Known(KnownResponsesTool::Namespace(namespace)) if namespace.name == "mcp__shell" => {
                    Some(namespace)
                }
                _ => None,
            })
            .expect("shell namespace");
        let shell_member_names: Vec<&str> = shell_namespace
            .tools
            .iter()
            .filter_map(|member| match member {
                CodexNamespaceMember::Function(function) => Some(function.name.as_str()),
                CodexNamespaceMember::Unknown(_) => None,
            })
            .collect();

        assert_eq!(shell_member_names, vec!["run", "status"]);
        assert!(!upstream.iter().any(|tool| {
            matches!(
                tool,
                ResponsesTool::Known(KnownResponsesTool::Function(function))
                    if function.name == "agentic_ns__mcp__shell__status"
            )
        }));
        assert!(upstream.iter().any(|tool| {
            matches!(
                tool,
                ResponsesTool::Known(KnownResponsesTool::Function(function))
                    if function.name == "agentic_ns__mcp__git__status"
            )
        }));

        for member in ["status", "run"] {
            let choice = ToolChoice::Function {
                namespace: Some("mcp__shell".to_string()),
                name: member.to_string(),
            };
            assert_eq!(flatten_tool_choice_for_upstream(&choice, Some(&tools)), choice);
        }

        let git_choice = ToolChoice::Function {
            namespace: Some("mcp__git".to_string()),
            name: "status".to_string(),
        };
        assert_eq!(
            flatten_tool_choice_for_upstream(&git_choice, Some(&tools)),
            ToolChoice::Function {
                namespace: None,
                name: "agentic_ns__mcp__git__status".to_string()
            }
        );
    }

    #[test]
    fn flat_name_collision_is_not_normalized_to_namespace_member() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {"type": "function", "name": "agentic_ns__mcp__shell__run"},
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();
        let mut output = vec![OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_string(),
            call_id: "call_1".to_string(),
            name: "agentic_ns__mcp__shell__run".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            status: "completed".to_string(),
        })];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace, None);
        assert_eq!(call.name, "agentic_ns__mcp__shell__run");
    }

    #[test]
    fn namespaced_tool_choice_does_not_flatten_to_colliding_top_level_name() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {"type": "function", "name": "agentic_ns__mcp__shell__run"},
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();
        let choice = ToolChoice::Function {
            namespace: Some("mcp__shell".to_string()),
            name: "run".to_string(),
        };

        assert_eq!(flatten_tool_choice_for_upstream(&choice, Some(&tools)), choice);

        let bare_choice = ToolChoice::Function {
            namespace: None,
            name: "run".to_string(),
        };
        assert_eq!(
            flatten_tool_choice_for_upstream(&bare_choice, Some(&tools)),
            bare_choice
        );
    }

    #[test]
    fn registry_keys_namespaced_tools_by_split_name() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__a",
                "tools": [{"type": "function", "name": "run"}]
            },
            {
                "type": "namespace",
                "name": "mcp__b",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();

        let registry = ToolRegistry::from_tools(Some(&tools));
        assert_eq!(registry.by_model_name.len(), 2);
        assert_eq!(
            registry.owner_for(&ToolName {
                namespace: Some("mcp__a".to_string()),
                name: "run".to_string(),
            }),
            Some(&ToolExecutionOwner::Client)
        );
        assert_eq!(
            registry.owner_for(&ToolName {
                namespace: Some("mcp__b".to_string()),
                name: "run".to_string(),
            }),
            Some(&ToolExecutionOwner::Client)
        );
    }

    #[test]
    fn client_action_detection_respects_tool_search_execution() {
        let registry = ToolRegistry::default();
        let client_search = OutputItem::ToolSearchCall(serde_json::json!({
            "type": "tool_search_call",
            "execution": "client"
        }));
        let hosted_search = OutputItem::ToolSearchCall(serde_json::json!({
            "type": "tool_search_call",
            "execution": "server"
        }));
        let custom = OutputItem::CustomToolCall(serde_json::json!({
            "type": "custom_tool_call",
            "input": "free form"
        }));

        assert!(client_search.requires_client_action(&registry));
        assert!(!hosted_search.requires_client_action(&registry));
        assert!(custom.requires_client_action(&registry));
    }

    #[test]
    fn namespace_container_call_normalizes_to_member_call() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__agentic_fixture",
                "tools": [
                    {"type": "function", "name": "run", "parameters": {"type": "object"}}
                ]
            }
        ]))
        .unwrap();
        let mut output = vec![OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_string(),
            call_id: "call_1".to_string(),
            name: "mcp__agentic_fixture".to_string(),
            namespace: None,
            arguments: "{\"tools\":\"opaque\",\"cmd\":\"echo namespace fixture\"}".to_string(),
            status: "completed".to_string(),
        })];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
        assert_eq!(call.name, "run");
        assert_eq!(call.arguments, "{\"cmd\":\"echo namespace fixture\"}");
    }

    #[test]
    fn flat_namespace_member_call_preserves_tools_argument() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__agentic_fixture",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();
        let mut output = vec![OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_string(),
            call_id: "call_1".to_string(),
            name: "agentic_ns__mcp__agentic_fixture__run".to_string(),
            namespace: None,
            arguments: "{\"tools\":\"legitimate\",\"cmd\":\"pwd\"}".to_string(),
            status: "completed".to_string(),
        })];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
        assert_eq!(call.name, "run");
        assert_eq!(call.arguments, "{\"tools\":\"legitimate\",\"cmd\":\"pwd\"}");
    }

    #[test]
    fn underscore_namespace_member_alias_normalizes_to_member_call() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__agentic_fixture",
                "tools": [{"type": "function", "name": "echo_text"}]
            }
        ]))
        .unwrap();
        let mut output = vec![OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_string(),
            call_id: "call_1".to_string(),
            name: "mcp__agentic_fixture_echo_text".to_string(),
            namespace: None,
            arguments: "{\"text\":\"hi\"}".to_string(),
            status: "completed".to_string(),
        })];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
        assert_eq!(call.name, "echo_text");
    }

    #[test]
    fn ambiguous_underscore_namespace_member_alias_is_not_normalized() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__a_b",
                "tools": [{"type": "function", "name": "c"}]
            },
            {
                "type": "namespace",
                "name": "mcp__a",
                "tools": [{"type": "function", "name": "b_c"}]
            }
        ]))
        .unwrap();
        let mut output = vec![OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_string(),
            call_id: "call_1".to_string(),
            name: "mcp__a_b_c".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            status: "completed".to_string(),
        })];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert!(call.namespace.is_none());
        assert_eq!(call.name, "mcp__a_b_c");
    }

    #[test]
    fn unambiguous_bare_namespace_member_normalizes_to_member_call() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__agentic_fixture",
                "tools": [{"type": "function", "name": "run"}]
            }
        ]))
        .unwrap();
        let mut output = vec![OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_string(),
            call_id: "call_1".to_string(),
            name: "run".to_string(),
            namespace: None,
            arguments: "{\"cmd\":\"pwd\"}".to_string(),
            status: "completed".to_string(),
        })];

        normalize_output_items_with_tools(&mut output, Some(&tools));

        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__agentic_fixture"));
        assert_eq!(call.name, "run");
    }
}
