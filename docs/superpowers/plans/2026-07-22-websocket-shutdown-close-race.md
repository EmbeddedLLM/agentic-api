# WebSocket Shutdown Close-Race Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the active WebSocket response before completing a graceful close handshake, without starting requests received after shutdown.

**Architecture:** Keep the existing cancellation-aware upstream drain loop. Replace the fire-and-forget sink close with a small internal helper that sends the close frame and then consumes receiver frames until the peer closes, ends, or errors; harden the integration test with a ping/pong receipt barrier before releasing its gated upstream.

**Tech Stack:** Rust 2024, Tokio, Axum WebSockets, `futures`, `tokio-tungstenite`, Cargo test/Clippy/rustfmt.

## Global Constraints

- Preserve the existing eight-second gateway drain timeout as the outer bound.
- Do not start inference for any WebSocket request received after cancellation.
- Do not hold a lock or guard across `.await`.
- Keep receive errors during the close handshake at debug level because shutdown is already handling the connection.
- Do not modify unrelated files or existing user worktree changes.

---

### Task 1: Add the close-handshake regression and implementation

**Files:**
- Modify: `crates/agentic-server/src/handler/websocket/responses.rs:1-390`
- Test: `crates/agentic-server/src/handler/websocket/responses.rs:362-390`
- Test: `crates/agentic-server/tests/responses_websocket_test.rs:1213-1252`

**Interfaces:**
- Consumes: the existing `WsSender`, `WsReceiver`, cancellation-aware `stream_ws_response`, and WebSocket `Message` types.
- Produces: internal `close_ws<Sender, Receiver, SendError, ReceiveError>(sender: &mut Sender, receiver: &mut Receiver)` behavior used by `responses_ws_loop`.

- [ ] **Step 1: Write the failing close-handshake unit test and deterministic integration barrier**

Add `sink` and `StreamExt` to the unit-test imports, import `Message` and the wished-for `close_ws`, then add:

```rust
#[tokio::test]
async fn close_ws_ignores_late_frames_until_peer_close() {
    let mut sender = sink::drain();
    let mut receiver = stream::iter([
        Ok::<_, &'static str>(Message::Text("late request".into())),
        Ok(Message::Binary(vec![1].into())),
        Ok(Message::Close(None)),
        Err("must remain unread"),
    ]);

    close_ws(&mut sender, &mut receiver).await;

    assert!(matches!(receiver.next().await, Some(Err("must remain unread"))));
}
```

In `test_websocket_shutdown_drains_active_response_before_closing`, after sending the post-shutdown request and before `release.send(())`, add a ping/pong barrier:

```rust
let barrier = Bytes::from_static(b"shutdown-request-received");
ws.send(Message::Ping(barrier.clone())).await.unwrap();
loop {
    let message = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
        .await
        .expect("timed out waiting for shutdown barrier pong")
        .expect("websocket should yield a message")
        .expect("websocket message should be ok");
    match message {
        Message::Pong(payload) => {
            assert_eq!(payload, barrier);
            break;
        }
        Message::Ping(_) | Message::Frame(_) => {}
        Message::Text(text) => panic!("unexpected response before upstream release: {text}"),
        Message::Close(frame) => panic!("websocket closed before upstream release: {frame:?}"),
        Message::Binary(_) => panic!("unexpected binary websocket message"),
    }
}
```

- [ ] **Step 2: Run the new unit test to verify RED**

Run:

```bash
cargo test -p agentic-server handler::websocket::responses::tests::close_ws_ignores_late_frames_until_peer_close
```

Expected: compilation fails because `close_ws` does not exist yet. This proves the new test requires the missing close-handshake behavior.

- [ ] **Step 3: Implement the minimal close handshake**

Import `Sink`, then add the internal helper:

```rust
async fn close_ws<Sender, Receiver, SendError, ReceiveError>(sender: &mut Sender, receiver: &mut Receiver)
where
    Sender: Sink<Message, Error = SendError> + Unpin,
    Receiver: Stream<Item = Result<Message, ReceiveError>> + Unpin,
    SendError: std::fmt::Display,
    ReceiveError: std::fmt::Display,
{
    if let Err(error) = sender.close().await {
        debug!(%error, "failed to send responses websocket close frame");
        return;
    }

    while let Some(message) = receiver.next().await {
        match message {
            Ok(Message::Close(_)) => break,
            Ok(Message::Text(_) | Message::Binary(_) | Message::Ping(_) | Message::Pong(_)) => {}
            Err(error) => {
                debug!(%error, "responses websocket close handshake receive failed");
                break;
            }
        }
    }
}
```

Replace the existing `sender.close().await` block at the end of `responses_ws_loop` with:

```rust
close_ws(&mut sender, &mut receiver).await;
```

- [ ] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
cargo test -p agentic-server handler::websocket::responses::tests::close_ws_ignores_late_frames_until_peer_close
cargo test -p agentic-server --test responses_websocket_test test_websocket_shutdown_drains_active_response_before_closing -- --exact
cargo test -p agentic-server --test responses_websocket_test
```

Expected: the unit test, targeted integration test, and all WebSocket integration tests pass.

- [ ] **Step 5: Stress the original failure path**

Run the exact shutdown integration test 100 times, stopping on the first failure:

```bash
for iteration in $(seq 1 100); do
  cargo test -q -p agentic-server --test responses_websocket_test \
    test_websocket_shutdown_drains_active_response_before_closing -- --exact || exit 1
done
```

Expected: 100 successful iterations and no connection reset.

- [ ] **Step 6: Commit the verified implementation**

```bash
git add crates/agentic-server/src/handler/websocket/responses.rs \
  crates/agentic-server/tests/responses_websocket_test.rs \
  docs/superpowers/plans/2026-07-22-websocket-shutdown-close-race.md
git commit -s -m "fix: complete websocket shutdown handshake"
```

---

### Task 2: Verify and review the branch before publishing

**Files:**
- Review: every file changed from `origin/main`
- Modify only if a requested review reports an actionable issue in the scoped diff.

**Interfaces:**
- Consumes: the committed implementation from Task 1 and repository CI commands.
- Produces: a reviewed, pushed branch and a draft GitHub pull request targeting `main`.

- [ ] **Step 1: Run repository verification**

```bash
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test
pre-commit run --all-files
```

Expected: every command exits zero with no warnings promoted to errors.

- [ ] **Step 2: Run the requested gstack pre-landing review**

Follow `/Users/farceo/.agents/skills/gstack/review/SKILL.md` against `origin/main`. Apply and re-verify mechanical findings; stop for user input only if the review identifies a non-mechanical choice.

- [ ] **Step 3: Run the requested Claude read-only review**

```bash
~/.codex/skills/claude-review/scripts/claude-review "focus on WebSocket shutdown races, cancellation safety, close-handshake liveness, and test determinism"
```

Expected: no actionable findings. If Claude reports one, address it with a new failing test first and rerun both verification and Claude review.

- [ ] **Step 4: Re-verify the final reviewed commit**

```bash
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test
pre-commit run --all-files
git diff --check origin/main...HEAD
```

Expected: all commands exit zero and the diff contains only the design, plan, worktree ignore, WebSocket implementation, and its tests.

- [ ] **Step 5: Push and open the draft pull request**

```bash
git push -u origin codex/fix-websocket-shutdown-race
```

Create a draft PR targeting `main` with these required sections:

```markdown
## Summary

- complete the WebSocket close handshake after draining an active response during shutdown
- discard post-shutdown frames without starting another request
- synchronize the shutdown regression test with a ping/pong receipt barrier

## Test Plan

- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
- `pre-commit run --all-files`
- 100 repeated runs of the targeted WebSocket shutdown test
- gstack `/review`
- Claude read-only review focused on shutdown concurrency
```
