use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// Error returned when a tool name is empty.
///
/// Kept in `types/` so the wire-shape module stays self-contained and does
/// not import from the behavioral layer (`tool/`).
#[derive(Debug, thiserror::Error)]
#[error("tool name must not be empty")]
pub struct EmptyToolNameError;

/// A non-empty tool name, validated at construction.
///
/// Eliminates scattered empty-name checks by making the invalid state
/// (`name = ""`) unrepresentable. Use [`TryFrom<String>`] or
/// [`TryFrom<&str>`] to construct; serde rejects empty strings automatically.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct NonEmptyToolName(String);

impl NonEmptyToolName {
    /// Returns the name as a `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for NonEmptyToolName {
    type Error = EmptyToolNameError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s.is_empty() {
            Err(EmptyToolNameError)
        } else {
            Ok(Self(s))
        }
    }
}

impl TryFrom<&str> for NonEmptyToolName {
    type Error = EmptyToolNameError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::try_from(s.to_owned())
    }
}

impl From<NonEmptyToolName> for String {
    fn from(n: NonEmptyToolName) -> String {
        n.0
    }
}

impl AsRef<str> for NonEmptyToolName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NonEmptyToolName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Request-side tool params.
///
/// This enum covers the current generic tool framework plus Codex-specific
/// Responses shapes (`namespace`, `tool_search`, `custom`) and a raw fallback
/// so future provider tools do not make requests fail at the boundary.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ResponsesTool {
    Function(FunctionToolParam),
    Mcp(McpToolParam),
    WebSearch(WebSearchToolParam),
    FileSearch(FileSearchToolParam),
    CodeInterpreter(CodeInterpreterToolParam),
    Namespace(CodexNamespaceToolParam),
    ToolSearch(CodexToolSearchToolParam),
    Custom(CodexCustomToolParam),
    Unknown(Value),
}

/// Parameters for a user-defined function tool.
///
/// Does NOT carry a `type` field — serde consumes the tag during
/// deserialization and the payload struct must not also carry it.
///
/// `name` is a [`NonEmptyToolName`]: serde rejects empty strings at
/// deserialization time, making the invalid state unrepresentable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolParam {
    pub name: NonEmptyToolName,
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

/// Parameters for an MCP (Model Context Protocol) tool server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolParam {
    pub server_label: String,
    pub server_url: String,
    pub allowed_tools: Option<Vec<String>>,
    /// Per-server auth headers forwarded by the gateway (e.g. `Authorization: Bearer <token>`).
    pub headers: Option<HashMap<String, String>>,
}

/// Parameters for a web search tool (no required fields).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebSearchToolParam {}

/// Parameters for a file search tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileSearchToolParam {
    pub vector_store_ids: Option<Vec<String>>,
}

/// Parameters for a code interpreter tool (no required fields).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeInterpreterToolParam {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexNamespaceToolParam {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub tools: Vec<CodexNamespaceMember>,
    #[serde(default)]
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub enum CodexNamespaceMember {
    Function(FunctionToolParam),
    Unknown(Value),
}

impl Serialize for CodexNamespaceMember {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Function(function) => serialize_with_type(serializer, "function", function),
            Self::Unknown(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for CodexNamespaceMember {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        if value.get("type").and_then(Value::as_str) == Some("function") {
            return Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::Function));
        }
        Ok(Self::Unknown(value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSearchExecution {
    Server,
    Client,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexToolSearchToolParam {
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
pub struct CodexCustomToolParam {
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

fn value_with_type<T: Serialize>(type_name: &str, value: &T) -> Result<Value, serde_json::Error> {
    let mut value = serde_json::to_value(value)?;
    if let Value::Object(map) = &mut value {
        map.insert("type".to_string(), Value::String(type_name.to_string()));
    }
    Ok(value)
}

fn serialize_with_type<S, T>(serializer: S, type_name: &str, value: &T) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    value_with_type(type_name, value)
        .map_err(serde::ser::Error::custom)?
        .serialize(serializer)
}

impl Serialize for ResponsesTool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Function(param) => serialize_with_type(serializer, "function", param),
            Self::Mcp(param) => serialize_with_type(serializer, "mcp", param),
            Self::WebSearch(param) => serialize_with_type(serializer, "web_search_preview", param),
            Self::FileSearch(param) => serialize_with_type(serializer, "file_search", param),
            Self::CodeInterpreter(param) => serialize_with_type(serializer, "code_interpreter", param),
            Self::Namespace(param) => serialize_with_type(serializer, "namespace", param),
            Self::ToolSearch(param) => serialize_with_type(serializer, "tool_search", param),
            Self::Custom(param) => serialize_with_type(serializer, "custom", param),
            Self::Unknown(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ResponsesTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let Some(type_name) = value.get("type").and_then(Value::as_str) else {
            return Ok(Self::Unknown(value));
        };

        match type_name {
            "function" => Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::Function)),
            "mcp" => Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::Mcp)),
            "web_search_preview" | "web_search" => {
                Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::WebSearch))
            }
            "file_search" => Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::FileSearch)),
            "code_interpreter" => {
                Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::CodeInterpreter))
            }
            "namespace" => Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::Namespace)),
            "tool_search" => Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::ToolSearch)),
            "custom" => Ok(serde_json::from_value(value.clone()).map_or(Self::Unknown(value), Self::Custom)),
            _ => Ok(Self::Unknown(value)),
        }
    }
}

impl ResponsesTool {
    #[must_use]
    pub fn original_type(&self) -> Option<&str> {
        match self {
            Self::Function(_) => Some("function"),
            Self::Mcp(_) => Some("mcp"),
            Self::WebSearch(_) => Some("web_search_preview"),
            Self::FileSearch(_) => Some("file_search"),
            Self::CodeInterpreter(_) => Some("code_interpreter"),
            Self::Namespace(_) => Some("namespace"),
            Self::ToolSearch(_) => Some("tool_search"),
            Self::Custom(_) => Some("custom"),
            Self::Unknown(value) => value.get("type").and_then(Value::as_str),
        }
    }

    #[must_use]
    pub fn to_raw_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_empty_name_accepts_valid() {
        let n = NonEmptyToolName::try_from("get_weather").unwrap();
        assert_eq!(n.as_str(), "get_weather");
    }

    #[test]
    fn non_empty_name_rejects_empty() {
        assert!(NonEmptyToolName::try_from(String::new()).is_err());
        assert!(NonEmptyToolName::try_from("").is_err());
    }

    #[test]
    fn non_empty_name_serde_round_trips() {
        let json = serde_json::json!("get_weather");
        let n: NonEmptyToolName = serde_json::from_value(json).unwrap();
        assert_eq!(n.as_str(), "get_weather");
        assert_eq!(serde_json::to_value(&n).unwrap(), serde_json::json!("get_weather"));
    }

    #[test]
    fn non_empty_name_serde_rejects_empty() {
        assert!(serde_json::from_value::<NonEmptyToolName>(serde_json::json!("")).is_err());
    }

    #[test]
    fn responses_tool_function_round_trips() {
        let json = serde_json::json!({
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "x-extra": "kept"
        });
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        assert!(matches!(tool, ResponsesTool::Function(_)));
        if let ResponsesTool::Function(ref p) = tool {
            assert_eq!(p.name.as_str(), "get_weather");
        }
        let back = serde_json::to_value(&tool).unwrap();
        assert_eq!(back["type"], "function");
        assert_eq!(back["name"], "get_weather");
        assert_eq!(back["x-extra"], "kept");
    }

    #[test]
    fn responses_tool_mcp_round_trips_with_field_values() {
        let json = serde_json::json!({
            "type": "mcp",
            "server_label": "my_server",
            "server_url": "http://localhost:9000",
            "allowed_tools": ["search", "fetch"]
        });
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        let back = serde_json::to_value(&tool).unwrap();
        assert_eq!(back["type"], "mcp");
        assert_eq!(back["server_label"], "my_server");
        if let ResponsesTool::Mcp(ref p) = tool {
            assert_eq!(p.allowed_tools, Some(vec!["search".to_owned(), "fetch".to_owned()]));
        }
    }

    #[test]
    fn codex_tool_shapes_round_trip() {
        let tools_json = serde_json::json!([
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
        assert!(matches!(tools[0], ResponsesTool::Namespace(_)));
        assert!(matches!(tools[1], ResponsesTool::ToolSearch(_)));
        assert!(matches!(tools[2], ResponsesTool::Custom(_)));
        assert!(matches!(tools[3], ResponsesTool::Unknown(_)));

        let serialized = serde_json::to_value(&tools).unwrap();
        assert_eq!(serialized[0]["tools"][0]["type"], "function");
        assert_eq!(serialized[3]["opaque"], true);
    }
}
