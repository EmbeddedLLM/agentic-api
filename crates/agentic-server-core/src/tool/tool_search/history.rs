//! Replayed tool-search history: the definitions its search outputs loaded, and the search rules the pending-call
//! ledger cannot see. The model's view of the items comes from the upstream projection (`SearchHistory::lower`).

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::mem;

use crate::tool::catalog::{PriorCalls, ToolCatalog, ToolIdentity, history_call_before_load};
use crate::tool::codex::is_searched_function_call;
use crate::tool::handler::ToolError;
use crate::types::io::input::{SearchHistory, SearchPairing};
use crate::types::io::{InOutItem, InputFunctionToolCall, InputItem, ResponsesInput, ToolSearchOutputMessage};
use crate::types::tools::{ToolSearchExecution, ToolSearchStatus};

/// Loads the definitions each completed search output in `input` returned, in history order, and checks each
/// function call against them.
///
/// The ledger (`executor::pending_calls`) checks how client calls and outputs pair by `call_id`. This keeps the rules
/// it cannot see: a search call's item id and model call id are unique, a hosted output answers an earlier hosted
/// call and a hosted call is answered, and only a completed output loads definitions.
pub(super) fn load_history_definitions(
    input: &ResponsesInput,
    catalog: &mut ToolCatalog,
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
    let mut prior = PriorCalls {
        unknown_namespace_calls: HashMap::new(),
        unqualified_call_positions: mem::take(unqualified_call_positions),
    };

    for (position, item) in items.iter().enumerate() {
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
                load_output_definitions(output, pairing, catalog, &prior)?;
            }
            (InputItem::FunctionCall(call), _) => {
                let call = if is_searched_function_call(call, |name| catalog.is_function(name)) {
                    Cow::Owned(InputFunctionToolCall {
                        namespace: None,
                        ..call.clone()
                    })
                } else {
                    Cow::Borrowed(call)
                };
                ensure_history_call_is_available(&call, catalog, &mut prior, position)?;
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
    *unqualified_call_positions = prior.unqualified_call_positions;
    Ok(())
}

fn load_output_definitions(
    output: &ToolSearchOutputMessage,
    pairing: Option<&SearchPairing>,
    catalog: &mut ToolCatalog,
    prior: &PriorCalls,
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
        catalog.load(tool, prior)?;
    }
    Ok(())
}

fn ensure_history_call_is_available(
    call: &InputFunctionToolCall,
    catalog: &ToolCatalog,
    prior: &mut PriorCalls,
    position: usize,
) -> Result<(), ToolError> {
    if let Some(namespace) = call.namespace.as_deref() {
        let member = catalog
            .definition(namespace)
            .filter(|record| matches!(record.identity, ToolIdentity::Namespace(_)))
            .and_then(|record| record.namespace_members.as_ref())
            .and_then(|members| members.indexes.get(&call.name).map(|index| &members.ordered[*index]));
        match member {
            Some(member) if !member.loaded => return Err(history_call_before_load()),
            Some(_) => {}
            None => {
                prior
                    .unknown_namespace_calls
                    .entry(namespace.to_owned())
                    .or_default()
                    .insert(call.name.clone());
            }
        }
    } else {
        if catalog.withheld_function_names.contains(&call.name) {
            return Err(history_call_before_load());
        }
        prior
            .unqualified_call_positions
            .entry(call.name.clone())
            .or_insert(position);
    }
    Ok(())
}
