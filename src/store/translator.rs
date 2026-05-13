use serde::{Deserialize, Serialize};

use crate::types::responses::{
    FunctionToolResultMessage, InputItem, InputMessage, InputMessageContent, OutputItem, ResponsesInput, ResponsesTool,
    ToolChoice,
};

pub const ITEM_DATA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    Input,
    Output,
}

// ---------------------------------------------------------------------------
// InOutItem — holds either an InputItem or OutputItem for DB storage
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum InOutItem {
    Input(InputItem),
    Output(OutputItem),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemPayload {
    pub v: u32,
    pub kind: ItemKind,
    pub item: String, // pre-serialized JSON
}

impl ItemPayload {
    #[must_use]
    pub fn from_input(item: &InputItem) -> Self {
        Self {
            v: ITEM_DATA_VERSION,
            kind: ItemKind::Input,
            item: serde_json::to_string(item).unwrap_or_default(),
        }
    }

    #[must_use]
    pub fn from_output(item: &OutputItem) -> Self {
        Self {
            v: ITEM_DATA_VERSION,
            kind: ItemKind::Output,
            item: serde_json::to_string(item).unwrap_or_default(),
        }
    }

    #[must_use]
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    #[must_use]
    pub fn from_item_row(row: &crate::database::models::Item) -> Option<InOutItem> {
        let payload: Self = serde_json::from_str(&row.data).ok()?;
        match payload.kind {
            ItemKind::Input => serde_json::from_str(&payload.item).ok().map(InOutItem::Input),
            ItemKind::Output => serde_json::from_str(&payload.item).ok().map(InOutItem::Output),
        }
    }
}

#[must_use]
pub fn normalize_input(input: &ResponsesInput) -> Vec<InputItem> {
    match input {
        ResponsesInput::Text(text) => vec![InputItem::Message(InputMessage {
            role: "user".into(),
            content: InputMessageContent::Text(text.clone()),
        })],
        ResponsesInput::Items(items) => items.clone(),
    }
}

#[must_use]
pub fn wrap_tool_result(item: FunctionToolResultMessage) -> InputItem {
    InputItem::FunctionCallOutput(item)
}

#[must_use]
pub fn resolve_tools(
    request_tools: Option<&Vec<ResponsesTool>>,
    stored_tools: Option<&Vec<ResponsesTool>>,
    tools_explicitly_set: bool,
) -> Option<Vec<ResponsesTool>> {
    if tools_explicitly_set {
        request_tools
    } else {
        stored_tools
    }
    .cloned()
}

#[must_use]
pub fn resolve_tool_choice(
    request_tool_choice: &ToolChoice,
    stored_tool_choice: &ToolChoice,
    tool_choice_explicitly_set: bool,
) -> ToolChoice {
    if tool_choice_explicitly_set {
        request_tool_choice.clone()
    } else {
        stored_tool_choice.clone()
    }
}

/// Extract only `InputItem` variants from a history of `InOutItem`.
pub fn to_input_items(history: Vec<InOutItem>) -> Vec<InputItem> {
    history
        .into_iter()
        .filter_map(|i| match i {
            InOutItem::Input(item) => Some(item),
            InOutItem::Output(_) => None,
        })
        .collect()
}
