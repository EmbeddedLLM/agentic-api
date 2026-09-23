//! Message and reasoning text state with incremental retained-byte accounting.

use super::active::Budget;
use crate::events::EventPayload;
use crate::executor::error::ExecutorResult;
use crate::executor::response_budget::{RETAINED_CONTAINER_OVERHEAD_BYTES, RetainedAccount, RetainedSize};
use crate::types::event::MessageStatus;
use crate::types::io::{ApplyDone, OutputItem, OutputMessage, OutputTextContent, ReasoningOutput};
use indexmap::IndexMap;
use std::collections::HashMap;

/// Text streamed for one message content part, and whether deltas carried it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct StreamedPart {
    pub(super) text: String,
    pub(super) streamed: bool,
}

/// Bytes retained by a streamed part: its container plus its text.
impl RetainedSize for StreamedPart {
    fn retained_bytes(&self) -> usize {
        RETAINED_CONTAINER_OVERHEAD_BYTES + "output_text".len() + self.text.len()
    }
}

/// Bytes retained by per-index streamed counters that hold no text yet: one
/// container per index plus the bytes charged for its deltas.
fn streamed_counter_bytes(counters: &HashMap<u32, usize>) -> usize {
    counters
        .values()
        .map(|bytes| RETAINED_CONTAINER_OVERHEAD_BYTES + bytes)
        .sum()
}

fn streamed_part_bytes(counters: &HashMap<u32, usize>, index: u32) -> usize {
    counters
        .get(&index)
        .map_or(0, |bytes| RETAINED_CONTAINER_OVERHEAD_BYTES + bytes)
}

/// Return the streamed part for `index`, charging its container before a new
/// entry exists so an empty part cannot grow the map for free.
fn part_mut<'a>(
    parts: &'a mut IndexMap<u32, StreamedPart>,
    index: u32,
    account: &mut RetainedAccount,
    budget: Budget<'_>,
) -> ExecutorResult<&'a mut StreamedPart> {
    if !parts.contains_key(&index) {
        account.charge(budget, StreamedPart::default().retained_bytes())?;
    }
    Ok(parts.entry(index).or_default())
}

/// Record streamed bytes for `index`, charging the container of a new index.
fn count_streamed(
    counters: &mut HashMap<u32, usize>,
    index: u32,
    delta: &str,
    account: &mut RetainedAccount,
    budget: Budget<'_>,
) -> ExecutorResult<()> {
    if !counters.contains_key(&index) {
        account.charge(budget, RETAINED_CONTAINER_OVERHEAD_BYTES)?;
    }
    account.charge(budget, delta.len())?;
    *counters.entry(index).or_default() += delta.len();
    Ok(())
}

#[derive(Clone)]
pub(super) struct MessageState {
    pub(super) item: OutputMessage,
    pub(super) parts: IndexMap<u32, StreamedPart>,
}

impl MessageState {
    pub(super) fn apply(
        &mut self,
        payload: &EventPayload,
        account: &mut RetainedAccount,
        budget: Budget<'_>,
    ) -> ExecutorResult<()> {
        match payload {
            EventPayload::TextDelta {
                delta, content_index, ..
            } => {
                let part = part_mut(&mut self.parts, *content_index, account, budget)?;
                account.charge(budget, delta.len())?;
                part.streamed = true;
                part.text.push_str(delta);
            }
            EventPayload::TextDone {
                text, content_index, ..
            } => {
                let part = part_mut(&mut self.parts, *content_index, account, budget)?;
                // A done-only part adopts the completed text; a delta-streamed
                // part already holds it and the snapshot is not charged twice.
                if !part.streamed {
                    account.grow(budget, part, |part| part.text.len(), |part| part.text.clone_from(text))?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn finalize(mut self) -> OutputItem {
        self.parts.sort_keys();
        for (_, part) in self.parts {
            if !part.text.is_empty() {
                self.item.content.push(OutputTextContent::new(part.text));
            }
        }
        self.item.status = MessageStatus::Completed;
        OutputItem::Message(self.item)
    }
}

impl RetainedSize for MessageState {
    fn retained_bytes(&self) -> usize {
        self.item.retained_bytes() + self.parts.values().map(RetainedSize::retained_bytes).sum::<usize>()
    }
}

/// Reasoning keeps per-index byte counters for streamed deltas and adopts the
/// completed text from `reasoning_text.done` / `reasoning_summary_text.done`,
/// which is the authoritative representation of each part.
#[derive(Clone)]
pub(super) struct ReasoningState {
    pub(super) item: ReasoningOutput,
    content_streamed: HashMap<u32, usize>,
    summary_streamed: HashMap<u32, usize>,
}

impl ReasoningState {
    pub(super) fn new(item: ReasoningOutput) -> Self {
        Self {
            item,
            content_streamed: HashMap::new(),
            summary_streamed: HashMap::new(),
        }
    }

    pub(super) fn apply(
        &mut self,
        payload: &EventPayload,
        account: &mut RetainedAccount,
        budget: Budget<'_>,
    ) -> ExecutorResult<()> {
        match payload {
            EventPayload::ReasoningTextDelta {
                delta, content_index, ..
            } => count_streamed(&mut self.content_streamed, *content_index, delta, account, budget),
            EventPayload::ReasoningSummaryTextDelta {
                delta, summary_index, ..
            } => count_streamed(&mut self.summary_streamed, *summary_index, delta, account, budget),
            EventPayload::ReasoningTextDone { content_index, .. } => {
                let count = self.item.content.len();
                let index = usize::try_from(*content_index).unwrap_or(usize::MAX).min(count);
                account.grow(
                    budget,
                    self,
                    |state| {
                        // ApplyDone inserts at this index (clamping sparse indexes), or
                        // retains nothing for empty text. Measure only its inserted part.
                        let inserted = if state.item.content.len() > count {
                            state.item.content[index].retained_bytes()
                        } else {
                            0
                        };
                        inserted + streamed_part_bytes(&state.content_streamed, *content_index)
                    },
                    |state| {
                        state.content_streamed.remove(content_index);
                        state.item.apply_done(payload, &mut String::new());
                    },
                )
            }
            EventPayload::ReasoningSummaryTextDone { summary_index, .. } => {
                let count = self.item.summary.len();
                let index = usize::try_from(*summary_index).unwrap_or(usize::MAX).min(count);
                account.grow(
                    budget,
                    self,
                    |state| {
                        let inserted = if state.item.summary.len() > count {
                            state.item.summary[index].retained_bytes()
                        } else {
                            0
                        };
                        inserted + streamed_part_bytes(&state.summary_streamed, *summary_index)
                    },
                    |state| {
                        state.summary_streamed.remove(summary_index);
                        state.item.apply_done(payload, &mut String::new());
                    },
                )
            }
            _ => Ok(()),
        }
    }
}

impl RetainedSize for ReasoningState {
    fn retained_bytes(&self) -> usize {
        self.item.retained_bytes()
            + streamed_counter_bytes(&self.content_streamed)
            + streamed_counter_bytes(&self.summary_streamed)
    }
}
