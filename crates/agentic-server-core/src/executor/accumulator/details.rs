//! Retained terminal response metadata, independent of output-item assembly.

use super::ResponseAccumulator;
use crate::events::{EventFrame, SSEEventType};
use crate::executor::error::ExecutorResult;
use crate::executor::response_budget::RetainedSize;
use crate::types::request_response::IncompleteDetails;
use crate::utils::common::deserialize_from_value_opt;

impl ResponseAccumulator {
    pub(super) fn capture_terminal_details_if_needed(&mut self, frame: &EventFrame) -> ExecutorResult<()> {
        if !matches!(
            frame.event_type,
            SSEEventType::ResponseCompleted | SSEEventType::ResponseFailed | SSEEventType::ResponseIncomplete
        ) {
            return Ok(());
        }
        let Some(response) = frame.wire.rest.get("response") else {
            return Ok(());
        };
        let incomplete_details = response
            .get("incomplete_details")
            .cloned()
            .and_then(deserialize_from_value_opt::<IncompleteDetails>);
        let error = response.get("error").filter(|error| !error.is_null());
        self.terminal_details_account.reconcile(
            self.budget.as_ref(),
            incomplete_details.retained_bytes() + error.map_or(0, RetainedSize::retained_bytes),
        )?;
        self.incomplete_details = incomplete_details;
        self.error = error.cloned();
        Ok(())
    }
}
