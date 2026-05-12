use async_stream::stream;
use futures::Stream;

use crate::types::responses::StreamEvent;

pub const DONE_MARKER: &str = "data: [DONE]\n\n";
pub const TERMINAL_EVENT_TYPES: &[&str] = &["response.completed", "response.failed", "response.incomplete"];

pub fn stream_responses_sse(events: impl Stream<Item = StreamEvent>) -> impl Stream<Item = String> {
    stream! {
        let mut done_emitted = false;
        for await event in events {
            yield event.as_responses_chunk();
            if !done_emitted && TERMINAL_EVENT_TYPES.contains(&event.type_str()) {
                yield DONE_MARKER.to_string();
                done_emitted = true;
            }
        }
        if !done_emitted {
            yield DONE_MARKER.to_string();
        }
    }
}
