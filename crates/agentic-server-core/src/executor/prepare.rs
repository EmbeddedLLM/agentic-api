use crate::executor::error::ExecutorResult;
use crate::executor::request::RequestContext;
use crate::tool::ToolSearchState;

/// Prepare the one pure request-scoped tool-search state after rehydration.
///
/// Active state is shared by blocking and streaming execution, persistence,
/// continuation, replay, and compaction.
///
/// # Errors
///
/// Returns a client-visible configuration error for invalid state.
pub(crate) fn prepare_tool_search(ctx: &mut RequestContext) -> ExecutorResult<()> {
    let state = ToolSearchState::build_with_loaded_tools(
        &ctx.enriched_request,
        ctx.tool_search_loaded_tools.as_deref().unwrap_or_default(),
        ctx.original_request.tools.is_some(),
    )?;
    if !state.is_active() {
        ctx.tool_search_state = Some(state);
        return Ok(());
    }

    let private_request = state.private_inference_request(&ctx.enriched_request)?;
    ctx.tool_search_state = Some(state);
    ctx.tool_search_private_request = Some(Box::new(private_request));
    Ok(())
}
