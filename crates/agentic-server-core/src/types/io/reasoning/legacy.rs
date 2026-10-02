//! Reasoning items in the shape earlier releases accepted.
//!
//! Before typed reasoning, a reasoning item accepted any string content
//! discriminator and untyped summary, opaque state, and status values. Stored
//! history rows, stored response metadata, and lenient upstream ingestion can
//! still carry that shape. This decoder mirrors that schema exactly and keeps
//! every field that satisfies the typed schema. An item it rejects could not have
//! been accepted before either, so callers fail closed or drop it as they did then.

use serde::Deserialize;
use serde_json::Value;

use super::{OpaqueReasoning, ReasoningStatus, ReasoningSummaryContent, ReasoningTextContent};
use crate::types::io::{AgentAttribution, OutputItem, ReasoningOutput};
use crate::utils::common::serialized_size_up_to;

/// The reasoning item schema before typed reasoning.
#[derive(Deserialize)]
struct LegacyReasoning {
    #[serde(default)]
    agent: Option<AgentAttribution>,
    #[serde(default)]
    id: String,
    #[serde(default)]
    content: Option<Vec<LegacyText>>,
    #[serde(default)]
    summary: Option<Vec<Value>>,
    encrypted_content: Option<Value>,
    status: Option<String>,
}

/// Earlier releases required a string discriminator but accepted any spelling.
#[derive(Deserialize)]
struct LegacyText {
    #[serde(rename = "type")]
    _kind: String,
    text: String,
}

impl ReasoningOutput {
    /// Decode a reasoning item that satisfies the schema earlier releases accepted.
    ///
    /// Plaintext parts become `reasoning_text`. Summary parts, opaque state, and
    /// a status that don't satisfy the typed schema are dropped; dropped opaque
    /// state is logged by size. Items that are already typed decode unchanged.
    /// Returns `None` for anything else, and for an item whose only replay state
    /// was dropped.
    pub(crate) fn from_legacy_value(item: &Value) -> Option<Self> {
        if item.get("type").and_then(Value::as_str) != Some("reasoning") {
            return None;
        }
        let legacy = LegacyReasoning::deserialize(item).ok()?;
        let had_state = legacy.encrypted_content.as_ref().is_some_and(|state| !state.is_null());
        let reasoning = Self {
            agent: legacy.agent,
            id: legacy.id,
            content: legacy
                .content
                .unwrap_or_default()
                .into_iter()
                .map(|part| ReasoningTextContent::new(part.text))
                .collect(),
            summary: legacy
                .summary
                .unwrap_or_default()
                .iter()
                .filter_map(|part| ReasoningSummaryContent::deserialize(part).ok())
                .collect(),
            encrypted_content: legacy.encrypted_content.and_then(opaque_state),
            status: legacy
                .status
                .and_then(|status| ReasoningStatus::deserialize(Value::String(status)).ok()),
        };
        // Opaque state without plaintext can't be replayed to vLLM and is rejected
        // before inference. An item emptied by dropping that state would pass that
        // check and reach the model, so it is unreadable instead.
        let replayable = !had_state
            || reasoning.encrypted_content.is_some()
            || reasoning.content.iter().any(|part| !part.text.is_empty());
        replayable.then_some(reasoning)
    }
}

/// Keep opaque state that satisfies the typed schema. Dropping it loses the
/// provider's replay state, so log that, with its size only and never the state.
fn opaque_state(state: Value) -> Option<OpaqueReasoning> {
    let (bytes, reason) = match state {
        Value::Null => return None,
        Value::String(state) => {
            let bytes = state.len();
            match OpaqueReasoning::try_from(state) {
                Ok(state) => return Some(state),
                Err(_) => (bytes, "exceeds the size ceiling"),
            }
        }
        other => (
            serialized_size_up_to(&other, usize::MAX)
                .ok()
                .flatten()
                .unwrap_or_default(),
            "is not a string",
        ),
    };
    tracing::warn!(
        bytes,
        reason,
        "dropped reasoning opaque state that the typed schema rejects"
    );
    None
}

/// Replace each reasoning item in the earlier shape with its typed projection.
///
/// Other items stay unchanged, so a caller that decodes the result still fails
/// closed on anything the projection can't read.
pub(crate) fn upgrade_legacy_reasoning(items: &mut [Value]) {
    for item in items {
        if let Some(reasoning) = ReasoningOutput::from_legacy_value(item)
            && let Ok(upgraded) = serde_json::to_value(OutputItem::Reasoning(reasoning))
        {
            *item = upgraded;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::types::io::reasoning::MAX_OPAQUE_REASONING_BYTES;

    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl Write for Captured {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Decode `encrypted_content` through the projection and return what it logged.
    fn project(encrypted_content: &Value) -> (Option<OpaqueReasoning>, String) {
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let item = serde_json::json!({
            "type": "reasoning", "id": "rs_1", "status": "failed", "encrypted_content": encrypted_content,
            "content": [{"type": "reasoning_text", "text": "plaintext"}],
        });
        let reasoning = tracing::subscriber::with_default(subscriber, || ReasoningOutput::from_legacy_value(&item));
        let logs = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
        (reasoning.unwrap().encrypted_content, logs)
    }

    #[test]
    fn dropped_opaque_state_is_logged_by_size_only() {
        let object = serde_json::json!({"ciphertext": "provider secret"});
        let (state, logs) = project(&object);
        assert!(state.is_none());
        assert!(logs.contains(&format!("bytes={}", object.to_string().len())), "{logs}");
        assert!(
            logs.contains("is not a string") && !logs.contains("provider secret"),
            "{logs}"
        );

        let oversized = "s".repeat(MAX_OPAQUE_REASONING_BYTES + 1);
        let (state, logs) = project(&Value::String(oversized));
        assert!(state.is_none());
        assert!(
            logs.contains(&format!("bytes={}", MAX_OPAQUE_REASONING_BYTES + 1)),
            "{logs}"
        );
        assert!(
            logs.contains("exceeds the size ceiling") && !logs.contains("sss"),
            "{logs}"
        );

        // Kept or absent state is not a loss; only the unknown status is dropped.
        for (encrypted_content, kept) in [(Value::String("provider state".into()), true), (Value::Null, false)] {
            let (state, logs) = project(&encrypted_content);
            assert_eq!(state.is_some(), kept);
            assert!(logs.is_empty(), "{logs}");
        }
    }

    /// An item whose only replay state was dropped is unreadable; plaintext keeps it usable.
    #[test]
    fn opaque_only_items_are_unreadable_when_their_state_is_dropped() {
        let item = |content: &Value, state: &Value| serde_json::json!({"type": "reasoning", "id": "rs_1", "content": content, "encrypted_content": state});
        let plaintext = serde_json::json!([{"type": "reasoning_text", "text": "plaintext"}]);
        let no_plaintext = [
            serde_json::json!([]),
            serde_json::json!([{"type": "reasoning_text", "text": ""}]),
            Value::Null,
        ];
        let oversized = Value::String("s".repeat(MAX_OPAQUE_REASONING_BYTES + 1));
        for state in [serde_json::json!({"ciphertext": "object state"}), oversized] {
            for content in &no_plaintext {
                assert!(
                    ReasoningOutput::from_legacy_value(&item(content, &state)).is_none(),
                    "{content}"
                );
            }
            let kept = ReasoningOutput::from_legacy_value(&item(&plaintext, &state)).unwrap();
            assert!(kept.encrypted_content.is_none());
            assert_eq!(kept.content, vec![ReasoningTextContent::new("plaintext")]);
        }

        // Nothing was dropped, so an item without plaintext reads as it did before.
        for state in [Value::Null, Value::String("kept state".into())] {
            for content in &no_plaintext {
                let kept = ReasoningOutput::from_legacy_value(&item(content, &state)).unwrap();
                assert_eq!(kept.encrypted_content.is_some(), state.is_string());
            }
        }
    }
}
