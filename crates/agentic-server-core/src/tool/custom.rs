use std::collections::HashMap;

use crate::types::event::MessageStatus;
use crate::types::io::{CustomToolCall, FunctionTool, FunctionToolCall, OutputItem};
use crate::types::tools::CustomToolParam;
use crate::utils::common::serialize_to_value_or_custom_default;

use super::{ToolEntry, ToolError, ToolHandler, ToolType};

/// Handler for client-owned `type: "custom"` tools.
///
/// Custom tools are normalized for the model but are executed by the client,
/// so this intentionally implements [`ToolHandler`] without
/// [`super::GatewayExecutor`].
#[derive(Debug)]
pub struct CustomHandler;

impl CustomHandler {
    #[must_use]
    pub fn to_function_call(param: &CustomToolParam) -> FunctionTool {
        FunctionTool {
            type_: "function".to_owned(),
            name: param.name.as_str().to_owned(),
            description: Some(model_visible_description(param)),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Raw custom tool input. Follow the tool description and declared format exactly."
                    }
                },
                "required": ["input"],
                "additionalProperties": false
            })),
            strict: Some(true),
        }
    }

    #[must_use]
    pub(crate) fn output_item(call: &FunctionToolCall) -> OutputItem {
        OutputItem::CustomToolCall(CustomToolCall {
            id: public_item_id(&call.id),
            status: Some(call.status),
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            input: input_from_arguments(&call.arguments),
        })
    }

    #[must_use]
    pub(crate) fn started_output_item(call: &FunctionToolCall) -> OutputItem {
        OutputItem::CustomToolCall(CustomToolCall {
            id: public_item_id(&call.id),
            status: Some(MessageStatus::InProgress),
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            input: String::new(),
        })
    }
}

fn model_visible_description(param: &CustomToolParam) -> String {
    let mut fragments = Vec::new();
    if let Some(description) = param
        .description
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        fragments.push(description.to_owned());
    }

    fragments.push("Provide the raw tool input in the `input` string field.".to_owned());

    if let Some(format) = &param.format {
        let format_type = format.get("type").and_then(serde_json::Value::as_str);
        let syntax = format.get("syntax").and_then(serde_json::Value::as_str);
        let definition = format
            .get("definition")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());

        if format_type == Some("grammar")
            && matches!(syntax, Some("lark" | "regex"))
            && let (Some(syntax), Some(definition)) = (syntax, definition)
        {
            fragments.push(format!(
                "The string must match this {syntax} grammar exactly:\n{definition}"
            ));
        } else {
            fragments.push(format!(
                "The string must conform to this custom tool format declaration exactly:\n{format}"
            ));
        }
    }

    if !param.extra.is_empty()
        && let Ok(extra) = serde_json::to_string(&param.extra)
    {
        fragments.push(format!(
            "Additional custom tool declaration fields that must be respected:\n{extra}"
        ));
    }

    fragments.join("\n\n")
}

impl ToolHandler for CustomHandler {
    fn tool_type(&self) -> ToolType {
        ToolType::Custom
    }

    fn validate(&self, param: &serde_json::Value) -> Result<(), ToolError> {
        serde_json::from_value::<CustomToolParam>(param.clone())
            .map(|_| ())
            .map_err(|error| ToolError::Config(format!("invalid custom tool config: {error}")))
    }

    fn normalize(&self, param: &serde_json::Value) -> Vec<FunctionTool> {
        match serde_json::from_value::<CustomToolParam>(param.clone()) {
            Ok(param) => vec![Self::to_function_call(&param)],
            Err(error) => {
                tracing::warn!(%error, "invalid custom tool param");
                Vec::new()
            }
        }
    }
}

pub(crate) fn insert_custom_entry(entries: &mut HashMap<String, ToolEntry>, param: &CustomToolParam) {
    serialize_to_value_or_custom_default(
        param,
        "custom tool config serialization failed",
        |config| {
            entries.insert(
                param.name.as_str().to_owned(),
                ToolEntry {
                    tool_type: ToolType::Custom,
                    config,
                    server_label: None,
                    handler: None,
                },
            );
        },
        (),
    );
}

pub(crate) fn public_item_id(item_id: &str) -> String {
    if item_id.starts_with("ctc_") {
        return item_id.to_owned();
    }
    if let Some(suffix) = item_id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("ctc_{suffix}");
    }
    format!("ctc_{:016x}", stable_name_hash(item_id))
}

fn stable_name_hash(value: &str) -> u64 {
    value.as_bytes().iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub(crate) fn input_from_arguments(arguments: &str) -> String {
    try_input_from_arguments(arguments).unwrap_or_else(|| arguments.to_owned())
}

pub(crate) fn try_input_from_arguments(arguments: &str) -> Option<String> {
    match serde_json::from_str::<serde_json::Value>(arguments).ok()? {
        serde_json::Value::String(input) => Some(input),
        serde_json::Value::Object(fields) if fields.len() == 1 => fields
            .get("input")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_fallback_uses_public_custom_tool_shape() {
        let call = FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "raw_echo".to_owned(),
            namespace: None,
            arguments: r#"{"input":"hello"}"#.to_owned(),
            status: MessageStatus::Completed,
        };

        let OutputItem::CustomToolCall(completed) = CustomHandler::output_item(&call) else {
            panic!("expected custom output item");
        };
        assert_eq!(completed.id, "ctc_1");
        assert_eq!(completed.input, "hello");
        assert_eq!(completed.status, Some(MessageStatus::Completed));
    }

    #[test]
    fn custom_call_id_is_stable_for_every_source_item_id() {
        assert_eq!(public_item_id("fc_item"), "ctc_item");
        assert_eq!(public_item_id("ctc_item"), "ctc_item");
        assert_eq!(public_item_id("provider_item"), public_item_id("provider_item"));
    }

    #[test]
    fn custom_declaration_normalizes_to_function_with_raw_input() {
        let param = serde_json::from_value::<CustomToolParam>(serde_json::json!({
            "name": "raw_echo",
            "description": "Echo raw input.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": "start: \"CUSTOM_OK\""
            },
            "x-provider-field": {"mode": "strict"}
        }))
        .expect("custom tool");

        let tool = CustomHandler::to_function_call(&param);

        assert_eq!(tool.type_, "function");
        assert_eq!(tool.name, "raw_echo");
        assert_eq!(
            tool.parameters.as_ref().unwrap()["properties"]["input"]["type"],
            "string"
        );
        assert_eq!(tool.parameters.as_ref().unwrap()["required"][0], "input");
        let description = tool.description.as_deref().expect("model-visible description");
        assert!(description.contains("Echo raw input."));
        assert!(description.contains("raw tool input in the `input` string field"));
        assert!(description.contains("lark grammar exactly"));
        assert!(description.contains("start: \"CUSTOM_OK\""));
        assert!(description.contains("x-provider-field"));
        assert!(description.contains("strict"));
    }

    #[test]
    fn regex_grammar_is_preserved_in_model_visible_description() {
        let param = serde_json::from_value::<CustomToolParam>(serde_json::json!({
            "name": "timestamp",
            "description": "Save a timestamp.",
            "format": {
                "type": "grammar",
                "syntax": "regex",
                "definition": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
            }
        }))
        .expect("custom tool");

        let tool = CustomHandler::to_function_call(&param);
        let description = tool.description.as_deref().expect("model-visible description");
        assert!(description.contains("regex grammar exactly"));
        assert!(description.contains("^[0-9]{4}-[0-9]{2}-[0-9]{2}$"));
    }
}
