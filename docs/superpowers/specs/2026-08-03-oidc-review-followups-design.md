# OIDC review follow-ups design

## Goal

Resolve all four actionable review threads on pull request #149 without broadening optional OpenID Connect (OIDC)
authentication into tenant authorization or changing pre-existing protocol behavior outside the reviewed paths.

## Considered approaches

### Focused compatibility fixes (selected)

Keep the gateway's single configured audience as the complete trust set, protect the existing JWKS refresh future with
a scoped cancellation guard, emit an OpenAI-compatible error event only for OIDC expiry in Responses WebSocket mode,
and add Anthropic request IDs only to authentication errors synthesized by this feature. This is the smallest approach
that satisfies the protocol contracts and preserves behavior newly tested on `main`.

### Detached JWKS refresh and configurable audience allowlist

Run refreshes in shared background tasks and add configuration for multiple trusted audiences. This makes cancellation
independent of request lifetime and supports deliberately multi-audience tokens, but adds task lifecycle, shutdown,
configuration, and documentation surface that issue #104 does not require.

### Normalize every gateway protocol error

Replace all Responses WebSocket and Anthropic error envelopes with fully typed protocol errors. This would improve
broader consistency, but current `main` explicitly tests the generic nested WebSocket envelope for persistence
failures, and non-authentication Anthropic request IDs are pre-existing behavior outside this pull request.

## Audience and authorized-party validation

`OIDC_AUDIENCE` is the gateway's only trusted audience. A scalar audience or every entry in an audience array must
equal that value. When `azp` is present, it must also equal the configured audience. Tokens containing an unconfigured
additional audience or a conflicting authorized party are rejected as invalid tokens.

Tests will cover scalar and array audiences, additional audiences, and `azp` values that are absent, matching, or
conflicting.

## Cancellation-safe JWKS refresh

The refresh gate continues to serialize provider requests. After setting `retry_after`, a scoped guard owns that exact
in-flight deadline. If the future is dropped before a provider result is processed, the guard clears the deadline only
when it still matches its own value. A completed provider request disarms the guard: successful refresh retains the
existing success state, while a real provider failure retains the intended 30-second retry backoff.

An integration test will pause a JWKS refresh, abort the authenticating task, release the provider, and prove that the
next refresh-dependent request is allowed to contact the provider instead of receiving a synthetic backoff response.

## Protocol error compatibility

When an authenticated Responses WebSocket principal expires between requests, the gateway sends a terminal error
event with top-level `type`, `code`, `message`, `param`, and `sequence_number`, then closes the connection. The event
uses sequence number zero because expiry is detected before processing the next `response.create`. Other `WsError`
variants retain their current envelope.

Anthropic authentication errors generate one `req_`-prefixed UUIDv7 using the existing core helper. The same value is
placed in the `request-id` response header and the top-level `request_id` error-body field. OpenAI-shaped authentication
errors remain unchanged.

## Rebase and verification

Rebase onto current `main` and combine the OIDC and PostgreSQL helper imports in the mechanical `main.rs` test conflict.
Each behavior change follows a failing-test-first cycle. After focused tests pass, run formatting, Clippy, the full Rust
test suite, pre-commit, documentation build, read-only Claude review, gstack pre-landing review, and GitHub CI.

## Out of scope

This follow-up does not add multiple trusted-audience configuration, change non-authentication WebSocket errors, add
request IDs to pre-existing non-authentication Anthropic paths, or implement tenant authorization from issue #107.
