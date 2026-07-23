# WebSocket Shutdown Close-Race Design

## Problem

The WebSocket shutdown integration test intermittently receives `ConnectionReset` while waiting for the active
response's terminal event. The test cancels the gateway, writes a second request, and releases the gated upstream
response without proving that the server has consumed the second frame. If the upstream finishes first, the handler
can initiate shutdown while client data remains unread, and Linux may reset the connection instead of delivering the
queued terminal event and a clean WebSocket close.

## Desired behavior

Once shutdown starts, the gateway must:

1. finish forwarding the active upstream response, including its terminal event;
2. ignore any new response requests rather than starting more inference;
3. initiate a WebSocket close handshake; and
4. consume remaining client frames until the peer acknowledges the close or disconnects.

The existing gateway-level eight-second drain timeout remains the outer bound for an unresponsive upstream or peer.

## Design

Keep the current active-stream drain loop. After it finishes, send a close frame and continue reading the WebSocket
receiver until it yields a peer close, end-of-stream, or receive error. Text, binary, ping, and pong frames received
during this closing phase are discarded; no new request reaches the executor. A receive error during the close
handshake is logged at debug level because the response has already completed and shutdown is in progress.

Make the integration test deterministic by sending the post-shutdown request followed by a ping, then waiting for the
matching pong before releasing the upstream response. WebSocket frame ordering means the pong proves the server has
already consumed and rejected the preceding request. The test then asserts that the original response completes, the
connection closes, and the mock upstream observed exactly one request.

## Alternatives considered

- **Test-only synchronization:** smallest change, but leaves production clients exposed to the same late-frame reset
  window.
- **WebSocket task-ownership redesign:** could centralize shutdown state, but is unnecessary for this localized close
  sequencing issue.

## Verification

- Demonstrate the close-handshake regression test failing before the production change.
- Run the targeted shutdown test repeatedly after the fix.
- Run the complete WebSocket integration-test binary.
- Run formatting and Clippy for the affected workspace targets.
