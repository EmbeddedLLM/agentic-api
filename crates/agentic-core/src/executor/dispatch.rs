use crate::tool::ToolRegistry;
use crate::types::io::{InputItem, OutputItem};

#[derive(Debug, Clone)]
pub enum LoopDecision {
    Continue(Vec<InputItem>),
    RequiresClientAction(Vec<OutputItem>),
    Done,
    Incomplete(String),
}

#[must_use]
pub fn client_action_items(output: &[OutputItem], registry: &ToolRegistry) -> Vec<OutputItem> {
    output
        .iter()
        .filter(|item| item.requires_client_action(registry))
        .cloned()
        .collect()
}

#[must_use]
pub fn decide_client_action(output: &[OutputItem], registry: &ToolRegistry) -> LoopDecision {
    let items = client_action_items(output, registry);
    if items.is_empty() {
        LoopDecision::Done
    } else {
        LoopDecision::RequiresClientAction(items)
    }
}
