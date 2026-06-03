//! Cassette-based integration tests for the Responses API (cases 1–5).
//!
//! Mirrors `test_responses_api.py`. Each test replays a YAML cassette
//! against a mock HTTP server and verifies `execute()` output.

mod support;

use agentic_core::executor::execute;
use std::sync::Arc;
use support::{TestFixture, collect_stream, expected_text, load_cassette, make_request, output_text, unwrap_blocking};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/text_only/responses");

/// Case 1 — single turn, non-streaming.
#[tokio::test]
async fn test_single_turn_nonstreaming() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/resp-single-gpt-4o-nonstreaming.yaml"));
    let t1 = &cassette.turns[0];
    let fixture = TestFixture::new(&[t1]).await;

    // Act
    let payload = unwrap_blocking(
        execute(
            make_request(&t1.request.body.input, t1.request.body.store, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("execute"),
    );

    // Assert
    assert!(payload.id.starts_with("resp_"), "id={}", payload.id);
    assert_eq!(payload.status, "completed");
    assert_eq!(output_text(&payload), expected_text(t1));
}

/// Case 2 — single turn, streaming.
#[tokio::test]
async fn test_single_turn_streaming() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/resp-single-gpt-4o-streaming.yaml"));
    let t1 = &cassette.turns[0];
    let fixture = TestFixture::new(&[t1]).await;

    // Act
    let payload = collect_stream(
        execute(
            make_request(&t1.request.body.input, t1.request.body.store, true, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("execute"),
    )
    .await;

    // Assert
    assert!(payload.id.starts_with("resp_"), "id={}", payload.id);
    assert_eq!(payload.status, "completed");
    assert_eq!(output_text(&payload), expected_text(t1));
}

/// Case 3 — two turns, non-streaming, chained via `previous_response_id`.
#[tokio::test]
async fn test_two_turn_nonstreaming_previous_response_id() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/resp-two-turn-gpt-4o-nonstreaming.yaml"));
    let (t1, t2) = (&cassette.turns[0], &cassette.turns[1]);
    let fixture = TestFixture::new(&[t1, t2]).await;

    // Act
    let p1 = unwrap_blocking(
        execute(
            make_request(&t1.request.body.input, true, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t1"),
    );
    let p2 = unwrap_blocking(
        execute(
            make_request(&t2.request.body.input, true, false, Some(p1.id.clone()), None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t2"),
    );

    // Assert
    assert!(p1.id.starts_with("resp_"));
    assert_eq!(p1.status, "completed");
    assert_eq!(output_text(&p1), expected_text(t1));
    assert_ne!(p2.id, p1.id);
    assert_eq!(p2.status, "completed");
    assert_eq!(p2.previous_response_id.as_deref(), Some(p1.id.as_str()));
    assert_eq!(output_text(&p2), expected_text(t2));
}

/// Case 4 — two turns, streaming, chained via `previous_response_id`.
#[tokio::test]
async fn test_two_turn_streaming_previous_response_id() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/resp-two-turn-gpt-4o-streaming.yaml"));
    let (t1, t2) = (&cassette.turns[0], &cassette.turns[1]);
    let fixture = TestFixture::new(&[t1, t2]).await;

    // Act
    let p1 = collect_stream(
        execute(
            make_request(&t1.request.body.input, true, true, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t1"),
    )
    .await;
    let p2 = collect_stream(
        execute(
            make_request(&t2.request.body.input, true, true, Some(p1.id.clone()), None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t2"),
    )
    .await;

    // Assert
    assert!(p1.id.starts_with("resp_"));
    assert_eq!(p1.status, "completed");
    assert_eq!(output_text(&p1), expected_text(t1));
    assert_ne!(p2.id, p1.id);
    assert_eq!(p2.status, "completed");
    assert_eq!(output_text(&p2), expected_text(t2));
}

/// Case 5 — `store=false` response cannot be used as `previous_response_id`.
#[tokio::test]
async fn test_store_disabled_not_reusable_as_previous_response_id() {
    // Arrange — only one mock needed; follow-up errors before hitting the LLM
    let cassette = load_cassette(&format!("{DIR}/resp-no-store-gpt-4o-nonstreaming.yaml"));
    let t1 = &cassette.turns[0];
    let fixture = TestFixture::new(&[t1]).await;

    // Act — turn 1, store=false
    let p1 = unwrap_blocking(
        execute(
            make_request(&t1.request.body.input, false, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t1"),
    );
    assert_eq!(p1.status, "completed");

    // Act — follow-up with the unstored id
    let result = execute(
        make_request("follow up", false, false, Some(p1.id.clone()), None),
        Arc::clone(&fixture.exec_ctx),
    )
    .await;

    // Assert — executor errors at rehydrate, before calling the LLM
    assert!(result.is_err(), "expected error for unstored previous_response_id");
}
