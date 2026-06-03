//! Server-Sent Event (SSE) types and response status enums.

use std::convert::Infallible;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Response completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Response is being generated.
    #[default]
    InProgress,

    /// Response generation completed successfully.
    Completed,

    /// Response generation incomplete (e.g., stream interrupted).
    Incomplete,

    /// Response generation encountered an error.
    Error,
}

impl ResponseStatus {
    /// Returns the canonical wire string for this status.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::Error => "error",
        }
    }
}

impl FromStr for ResponseStatus {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "incomplete" => Self::Incomplete,
            _ => Self::Error,
        })
    }
}

/// Message item completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageStatus {
    /// Message is being generated.
    #[default]
    InProgress,

    /// Message generation completed.
    Completed,
}

impl MessageStatus {
    /// Returns the canonical wire string for this status.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
        }
    }
}

impl FromStr for MessageStatus {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "completed" => Self::Completed,
            _ => Self::InProgress,
        })
    }
}

/// Server-Sent Event types from LLM streaming responses.
///
/// Emitted by vLLM when `stream=true`. Each variant represents one step in the
/// response generation process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SSEEventType {
    /// Response object created; contains initial response metadata.
    ResponseCreated,

    /// Output item (message) added; marks the start of a new message.
    ResponseOutputItemAdded,

    /// Text delta; incremental token content added to the current message.
    ResponseOutputTextDelta,

    /// Response fully completed; no more events will follow.
    ResponseDone,

    /// Unknown or unhandled event type.
    #[default]
    Other,
}

impl FromStr for SSEEventType {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "response.created" => Self::ResponseCreated,
            "response.output_item.added" => Self::ResponseOutputItemAdded,
            "response.output_text.delta" => Self::ResponseOutputTextDelta,
            // vLLM uses `response.done`; OpenAI uses `response.completed`.
            "response.done" | "response.completed" => Self::ResponseDone,
            _ => Self::Other,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_event_type_from_str_created() {
        assert_eq!(
            "response.created".parse::<SSEEventType>().unwrap(),
            SSEEventType::ResponseCreated
        );
    }

    #[test]
    fn test_sse_event_type_from_str_delta() {
        assert_eq!(
            "response.output_text.delta".parse::<SSEEventType>().unwrap(),
            SSEEventType::ResponseOutputTextDelta
        );
    }

    #[test]
    fn test_sse_event_type_from_str_done() {
        assert_eq!(
            "response.done".parse::<SSEEventType>().unwrap(),
            SSEEventType::ResponseDone
        );
    }

    #[test]
    fn test_sse_event_type_from_str_unknown() {
        assert_eq!("unknown.event".parse::<SSEEventType>().unwrap(), SSEEventType::Other);
    }

    #[test]
    fn test_sse_event_type_from_str_empty() {
        assert_eq!("".parse::<SSEEventType>().unwrap(), SSEEventType::Other);
    }

    #[test]
    fn test_response_status_round_trip() {
        for (s, expected) in [
            ("in_progress", ResponseStatus::InProgress),
            ("completed", ResponseStatus::Completed),
            ("incomplete", ResponseStatus::Incomplete),
            ("error", ResponseStatus::Error),
        ] {
            let parsed: ResponseStatus = s.parse().unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.as_str(), s);
        }
    }

    #[test]
    fn test_message_status_round_trip() {
        assert_eq!("completed".parse::<MessageStatus>().unwrap(), MessageStatus::Completed);
        assert_eq!(
            "in_progress".parse::<MessageStatus>().unwrap(),
            MessageStatus::InProgress
        );
        assert_eq!("unknown".parse::<MessageStatus>().unwrap(), MessageStatus::InProgress);
    }
}
