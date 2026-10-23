//! Replayed tool-search history: the definitions its search outputs loaded, and the search rules the pending-call
//! ledger cannot see. The model's view of the items comes from the upstream projection (`SearchHistory::lower`).

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::mem;

use crate::tool::codex::is_searched_function_call;
use crate::tool::handler::ToolError;
use crate::types::io::input::{SearchHistory, SearchPairing};
use crate::types::io::{InOutItem, InputFunctionToolCall, InputItem, ResponsesInput, ToolSearchOutputMessage};
use crate::types::tools::{ToolSearchExecution, ToolSearchStatus};

use super::definitions::load_definition;
use super::{DefinitionAccumulator, DefinitionViews, LoadedToolIdentity};

/// Loads the definitions each completed search output in `input` returned, in history order, and checks each
/// function call against them.
///
/// The ledger (`executor::pending_calls`) checks how client calls and outputs pair by `call_id`. This keeps the rules
/// it cannot see: a search call's item id and model call id are unique, a hosted output answers an earlier hosted
/// call and a hosted call is answered, and only a completed output loads definitions.
pub(super) fn load_history_definitions(
    input: &ResponsesInput,
    views: DefinitionViews<'_>,
    unqualified_call_positions: &mut HashMap<String, usize>,
) -> Result<(), ToolError> {
    let ResponsesInput::Items(items) = input else {
        return Ok(());
    };
    let items: Vec<Cow<'_, InputItem>> = items.iter().filter_map(InOutItem::as_input).collect();
    let search_history = SearchHistory::scan(&items);
    // A hosted call's model call id is its item id. It must not reuse another call's id, or the model would see two
    // calls with one id; the ledger checks how client call ids pair.
    let client_call_ids: HashSet<&str> = items.iter().filter_map(|item| item.call_id()).collect();
    let mut call_item_ids = HashSet::new();
    let mut hosted_call_ids = HashSet::new();
    let DefinitionViews {
        public_tools,
        definitions,
        definition_indexes,
        loaded_public_tools,
        withheld_function_names,
        mcp_loads,
    } = views;
    let mut definition_accumulator = DefinitionAccumulator {
        public_tools,
        definitions,
        definition_indexes,
        loaded_public_tools,
        withheld_function_names,
        mcp_loads,
        prior_unknown_namespace_calls: HashMap::new(),
        unqualified_call_positions: mem::take(unqualified_call_positions),
        current_history_position: None,
    };

    for (position, item) in items.iter().enumerate() {
        definition_accumulator.current_history_position = Some(position);
        match (item.as_ref(), search_history.pairing(position)) {
            (InputItem::ToolSearchCall(call), pairing) => {
                if !call_item_ids.insert(call.id.as_str()) {
                    return Err(ToolError::Config("duplicate tool_search_call item id".to_owned()));
                }
                if call.execution == ToolSearchExecution::Server
                    && let Some(SearchPairing::Paired(call_id)) = pairing
                    && (client_call_ids.contains(call_id.as_str()) || !hosted_call_ids.insert(call_id.as_str()))
                {
                    return Err(ToolError::Config("duplicate tool_search_call call_id".to_owned()));
                }
            }
            (InputItem::ToolSearchOutput(output), pairing) => {
                load_output_definitions(output, pairing, &mut definition_accumulator)?;
            }
            (InputItem::FunctionCall(call), _) => {
                let is_function = |name: &str| {
                    definition_accumulator
                        .definition_indexes
                        .get(name)
                        .is_some_and(|index| {
                            matches!(
                                definition_accumulator.definitions[*index].identity,
                                LoadedToolIdentity::Function(_)
                            )
                        })
                };
                let call = if is_searched_function_call(call, is_function) {
                    Cow::Owned(InputFunctionToolCall {
                        namespace: None,
                        ..call.clone()
                    })
                } else {
                    Cow::Borrowed(call)
                };
                ensure_history_call_is_available(&call, &mut definition_accumulator)?;
            }
            _ => {}
        }
    }

    let open_hosted_call = items.iter().enumerate().any(|(position, item)| {
        matches!(item.as_ref(), InputItem::ToolSearchCall(call) if call.execution == ToolSearchExecution::Server)
            && search_history.pairing(position) == Some(&SearchPairing::OpenCall)
    });
    if open_hosted_call {
        return Err(ToolError::Config(
            "unresolved tool_search_call requires a matching completed tool_search_output".to_owned(),
        ));
    }
    *unqualified_call_positions = definition_accumulator.unqualified_call_positions;
    Ok(())
}

fn load_output_definitions(
    output: &ToolSearchOutputMessage,
    pairing: Option<&SearchPairing>,
    definition_accumulator: &mut DefinitionAccumulator<'_>,
) -> Result<(), ToolError> {
    if output.execution == ToolSearchExecution::Server && pairing == Some(&SearchPairing::UnmatchedOutput) {
        return Err(ToolError::Config(
            "hosted tool_search_output has no preceding hosted tool_search_call".to_owned(),
        ));
    }
    if output.status != ToolSearchStatus::Completed {
        return Err(ToolError::Config(
            "tool_search_output must be completed before it may load tool definitions".to_owned(),
        ));
    }
    for tool in &output.tools {
        load_definition(tool, definition_accumulator)?;
    }
    Ok(())
}

fn ensure_history_call_is_available(
    call: &InputFunctionToolCall,
    definitions: &mut DefinitionAccumulator<'_>,
) -> Result<(), ToolError> {
    if let Some(namespace) = call.namespace.as_deref() {
        let member = definitions
            .definition_indexes
            .get(namespace)
            .and_then(|index| definitions.definitions.get(*index))
            .filter(|record| matches!(record.identity, LoadedToolIdentity::Namespace(_)))
            .and_then(|record| record.namespace_members.as_ref())
            .and_then(|members| members.indexes.get(&call.name).map(|index| &members.ordered[*index]));
        match member {
            Some(member) if !member.loaded => return Err(withheld_function_history_call()),
            Some(_) => {}
            None => {
                definitions
                    .prior_unknown_namespace_calls
                    .entry(namespace.to_owned())
                    .or_default()
                    .insert(call.name.clone());
            }
        }
    } else {
        if definitions.withheld_function_names.contains(&call.name) {
            return Err(withheld_function_history_call());
        }
        let position = definitions
            .current_history_position
            .ok_or_else(|| ToolError::Config("tool-search history position is unavailable".to_owned()))?;
        definitions
            .unqualified_call_positions
            .entry(call.name.clone())
            .or_insert(position);
    }
    Ok(())
}

pub(super) fn withheld_function_history_call() -> ToolError {
    ToolError::Config("request history calls a function before its definition is loaded".to_owned())
}
