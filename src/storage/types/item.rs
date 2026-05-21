//! Domain types for conversation items.

use serde::{Deserialize, Serialize};

use crate::types::io::{InputItem, OutputItem};

/// Item kind (input vs output) for storage and retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    Input,
    Output,
}

/// Union type for conversation items (input or output).
#[derive(Debug, Clone)]
pub enum InOutItem {
    Input(InputItem),
    Output(OutputItem),
}

impl From<InputItem> for InOutItem {
    fn from(item: InputItem) -> Self {
        Self::Input(item)
    }
}

impl From<OutputItem> for InOutItem {
    fn from(item: OutputItem) -> Self {
        Self::Output(item)
    }
}

impl From<&InOutItem> for String {
    fn from(item: &InOutItem) -> Self {
        match item {
            InOutItem::Input(input) => serde_json::to_string(input).unwrap_or_default(),
            InOutItem::Output(output) => serde_json::to_string(output).unwrap_or_default(),
        }
    }
}

impl InOutItem {
    /// Extracts input items from a mixed history, filtering out output items.
    #[must_use]
    pub fn into_input_items(history: Vec<InOutItem>) -> Vec<InputItem> {
        history
            .into_iter()
            .filter_map(|i| match i {
                InOutItem::Input(item) => Some(item),
                InOutItem::Output(_) => None,
            })
            .collect()
    }
}
