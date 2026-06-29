//! Tool call dispatch — matches model responses to tool definitions and executes
//! server-side tools, returning input items ready for the next turn.

use crate::tool::ExecuteTool;
use crate::tool::IncomingTool;
use crate::types::io::{InputItem, OutputItem};

/// Returns the effective name used to match a tool definition against a
/// `function_call` response item.
pub fn tool_name(tool: &IncomingTool) -> &str {
    match tool {
        IncomingTool::Function(f) => &f.name,
        IncomingTool::Custom { name, .. } => name,
        IncomingTool::ToolSearch { .. } => "tool_search",
        // Namespace containers and unknown types have no single call name.
        IncomingTool::Namespace { .. } | IncomingTool::Unknown => "",
    }
}

/// Match `FunctionCall` items in `output` against `tools` and execute any that
/// are handled server-side by agentic-api.
///
/// For each matched call where [`ExecuteTool::execute`] returns `Some`, appends
/// two `InputItem`s in order:
/// 1. `FunctionCall` — the model's original tool invocation (keeps history coherent)
/// 2. `FunctionCallOutput` — the execution result
///
/// Returns an empty `Vec` when all calls belong to Codex CLI. The caller uses
/// a non-empty return to decide whether to loop for another inference turn.
pub async fn execute_server_tool_calls(output: &[OutputItem], tools: &[IncomingTool]) -> Vec<InputItem> {
    let mut results = Vec::new();

    for item in output {
        let OutputItem::FunctionCall(call) = item else {
            continue;
        };

        let Some(tool) = tools.iter().find(|t| tool_name(t) == call.name) else {
            continue;
        };

        if let Some(result) = tool.execute(&call.call_id, &call.arguments).await {
            results.push(InputItem::FunctionCall(call.clone()));
            results.push(InputItem::FunctionCallOutput(result));
        }
    }

    results
}
