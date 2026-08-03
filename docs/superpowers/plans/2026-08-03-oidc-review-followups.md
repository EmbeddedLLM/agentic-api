# OIDC Review Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all four actionable review threads on PR #149 while preserving unrelated protocol behavior.

**Architecture:** Keep the configured OIDC audience as the complete trust set, use a scoped RAII guard for refresh
cancellation, and add protocol-specific response data only at the two authentication boundaries under review. Rebase
first so every TDD cycle runs against current `main`.

**Tech Stack:** Rust 2024, Axum, Tokio, jsonwebtoken, serde/serde_json, reqwest, UUIDv7, cargo test.

## Global Constraints

- Rust MSRV is 1.85 and `unsafe` is forbidden.
- Do not hold `Mutex` or `RwLock` guards across `.await`.
- Keep non-authentication WebSocket error envelopes unchanged.
- Keep non-authentication Anthropic response behavior outside this follow-up.
- Follow `TERMINOLOGY.md` and preserve exact protocol field names.
- Every commit must use a conventional prefix and `git commit -s`.

---

### Task 1: Rebase onto current main

**Files:**

- Resolve: `crates/agentic-server/src/main.rs`

**Interfaces:**

- Consumes: the OIDC helpers from PR #149 and PostgreSQL helpers from current `main`.
- Produces: a branch based on current `origin/main` with both helper sets imported in the `main.rs` test module.

- [ ] **Step 1: Verify the worktree and remote branch are unchanged**

Run:

```bash
git status --short
git rev-parse HEAD
git rev-parse origin/codex/issue-102-oidc-auth
```

Expected: clean status and matching local/remote heads before history changes.

- [ ] **Step 2: Rebase onto current main**

Run:

```bash
git rebase origin/main
```

Expected: one conflict in the `crates/agentic-server/src/main.rs` test import block.

- [ ] **Step 3: Combine the imports**

Retain `oidc_config_from_values` alongside `database_configs_from_env`, `parse_env_duration_value`,
`parse_env_optional_duration_value`, and the existing integer/temp-store parsers. Retain both OIDC tests and every
PostgreSQL parser/configuration test from `main`.

- [ ] **Step 4: Continue and verify the rebase**

Run:

```bash
git add crates/agentic-server/src/main.rs
git rebase --continue
cargo test -p agentic-server --bin agentic-server
```

Expected: rebase completes and binary unit tests pass.

### Task 2: Enforce the complete audience trust set

**Files:**

- Modify: `crates/agentic-server/src/auth.rs`
- Test: `crates/agentic-server/tests/oidc_auth_test.rs`

**Interfaces:**

- Consumes: `IdentityClaims { aud: AudienceClaim, azp: Option<String> }` and the configured audience string.
- Produces: `IdentityClaims::audience_allows(&self, expected: &str) -> bool`, accepting only trusted audience values
  and an absent or matching authorized party.

- [ ] **Step 1: Write failing integration cases**

Extend `multi_audience_tokens_require_the_expected_authorized_party` so these literal cases are checked:

```rust
(&["agentic-api"][..], None, StatusCode::OK),
(&["agentic-api"][..], Some("agentic-api"), StatusCode::OK),
(&["agentic-api"][..], Some("other-client"), StatusCode::UNAUTHORIZED),
(&["agentic-api", "other-client"][..], Some("agentic-api"), StatusCode::UNAUTHORIZED),
(&["agentic-api", "other-client"][..], None, StatusCode::UNAUTHORIZED),
```

Add a scalar-audience token with `azp: "other-client"` and assert `401 Unauthorized` so scalar and array parsing both
exercise authorized-party validation.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p agentic-server --test oidc_auth_test multi_audience_tokens_require_the_expected_authorized_party
```

Expected: FAIL because an unconfigured additional audience and a conflicting scalar `azp` are currently accepted.

- [ ] **Step 3: Implement minimal validation**

Implement the equivalent of:

```rust
fn audience_allows(&self, expected: &str) -> bool {
    let audiences_match = match &self.aud {
        AudienceClaim::One(audience) => audience == expected,
        AudienceClaim::Many(audiences) => {
            !audiences.is_empty() && audiences.iter().all(|audience| audience == expected)
        }
    };
    audiences_match && self.azp.as_deref().is_none_or(|authorized_party| authorized_party == expected)
}
```

Use an MSRV-compatible expression if `Option::is_none_or` is unavailable on Rust 1.85.

- [ ] **Step 4: Verify GREEN**

Run the focused test from Step 2, then:

```bash
cargo test -p agentic-server --test oidc_auth_test configured_oidc_rejects_wrong_issuer_audience_and_expired_tokens
```

Expected: both tests pass.

### Task 3: Make JWKS refresh cancellation-safe

**Files:**

- Modify: `crates/agentic-server/src/auth.rs`
- Test: `crates/agentic-server/src/auth.rs`

**Interfaces:**

- Consumes: `Arc<std::sync::Mutex<RefreshState>>` and the exact `retry_after` deadline installed for one refresh.
- Produces: a private `RefreshAttempt` guard whose `Drop` clears only its own cancelled in-flight deadline and whose
  completion methods retain a real provider-failure backoff or the successful refresh state.

- [ ] **Step 1: Add a pausable provider and failing cancellation test**

Inside the existing `auth.rs` test module, start an Axum OIDC provider whose initial JWKS response contains `old-key`
with `max-age=0`. Later JWKS requests signal a zero-permit `Semaphore`, wait on a second semaphore, and return
`new-key`. Discover the authenticator, spawn `authenticate` with a token signed by `new-key`, wait for refresh start,
abort and await that task, release two provider requests, then authenticate again.

Assert that the second authentication succeeds and the JWKS request count reaches three: initial discovery, cancelled
refresh, and replacement refresh.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p agentic-server cancelled_jwks_refresh_does_not_install_backoff -- --exact
```

Expected: FAIL with `JwksRefreshBackoff` because the cancelled future leaves `retry_after` set.

- [ ] **Step 3: Implement the scoped refresh guard**

Add a private guard shaped like:

```rust
struct RefreshAttempt {
    state: Arc<std::sync::Mutex<RefreshState>>,
    retry_after: Instant,
    clear_on_drop: bool,
}
```

Its constructor installs the deadline. `retain_backoff` and `complete` set `clear_on_drop = false`. `Drop` takes the
synchronous lock and clears `retry_after` only when `clear_on_drop` is true and the stored deadline equals its own.
Create the guard after acquiring `refresh_gate`; retain backoff only when `fetch_jwks` returns a real error, and
complete it only after the success state has been written. Do not hold either lock across provider or Tokio lock
awaits.

- [ ] **Step 4: Verify GREEN and real-failure behavior**

Run the focused test from Step 2, then:

```bash
cargo test -p agentic-server --test oidc_auth_test jwks_refresh_failure_returns_protocol_specific_service_errors
```

Expected: cancellation test passes and genuine provider failures still use the existing backoff behavior.

### Task 4: Emit the canonical OIDC-expiry WebSocket event

**Files:**

- Modify: `crates/agentic-server/src/handler/websocket/responses.rs`
- Test: `crates/agentic-server/src/handler/websocket/responses.rs`

**Interfaces:**

- Consumes: `Option<&AuthenticatedPrincipal>` before processing a queued or newly received `response.create`.
- Produces: `Option<Value>` containing exactly the OIDC-expiry Responses WebSocket error event.

- [ ] **Step 1: Write the failing unit assertion**

Change the expiry test to require this literal value:

```rust
json!({
    "type": "error",
    "code": "invalid_token",
    "message": "OIDC bearer token expired",
    "param": null,
    "sequence_number": 0,
})
```

Keep a separate assertion proving the generic `WsError::to_ws_frame` behavior is untouched.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p agentic-server websocket_identity_expiry_uses_responses_error_event -- --exact
```

Expected: FAIL because expiry currently uses the nested generic WebSocket envelope.

- [ ] **Step 3: Implement the narrow event path**

Replace `websocket_identity_error` with an event helper returning the literal top-level fields above. In
`responses_ws_loop`, send that JSON value directly with `send_ws_json` and then close. Leave `WsError`,
`handle_ws_error`, and every non-authentication error path unchanged.

- [ ] **Step 4: Verify GREEN and existing WebSocket behavior**

Run the focused test from Step 2, then:

```bash
cargo test -p agentic-server --test responses_websocket_test
```

Expected: the expiry test and all existing WebSocket tests pass.

### Task 5: Add Anthropic authentication request IDs

**Files:**

- Modify: `crates/agentic-server/src/auth.rs`
- Test: `crates/agentic-server/tests/oidc_auth_test.rs`

**Interfaces:**

- Consumes: `agentic_core::utils::common::uuid7_str("req_")` when rendering an Anthropic authentication error.
- Produces: one identifier used by both the `request-id` response header and top-level `request_id` body field.

- [ ] **Step 1: Write the failing integration test**

Request `/v1/messages` without a bearer token, copy the `request-id` header before consuming the body, and assert:

```rust
assert!(request_id.starts_with("req_"));
assert_eq!(body["request_id"], request_id);
assert_eq!(body["error"]["type"], "authentication_error");
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p agentic-server --test oidc_auth_test anthropic_authentication_errors_include_matching_request_id
```

Expected: FAIL because the header and body field are absent.

- [ ] **Step 3: Implement header/body parity**

In `protocol_error`, generate a request ID only for `AuthErrorFormat::Anthropic`, include it as top-level
`request_id`, and add the same value through `Response::builder().header("request-id", request_id)`. Keep OpenAI-shaped
errors byte-for-byte unchanged apart from formatting performed by rustfmt.

- [ ] **Step 4: Verify GREEN and both dependency-error formats**

Run the focused test from Step 2, then:

```bash
cargo test -p agentic-server --test oidc_auth_test jwks_refresh_failure_returns_protocol_specific_service_errors
```

Expected: both tests pass.

### Task 6: Review, verify, and update PR #149

**Files:**

- Review: every file changed against `origin/main`
- Update: PR #149 Summary and Test Plan

**Interfaces:**

- Consumes: the rebased, locally verified branch and all four unresolved reviewer threads.
- Produces: signed commits, an updated remote branch, thread replies/resolutions, and green CI.

- [ ] **Step 1: Run local Rust gates**

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
git diff --check origin/main...HEAD
```

Expected: all commands pass without new warnings.

- [ ] **Step 2: Run repository gates**

```bash
uvx pre-commit run --all-files
uv run --with-requirements docs/requirements.txt mkdocs build
```

Expected: hooks pass; MkDocs has no warning attributable to this follow-up.

- [ ] **Step 3: Run required reviews in order**

Apply the Rust self-review checklist, run the read-only Claude review, then run the gstack pre-landing review. Fix every
actionable finding with focused tests and repeat relevant reviews until clean.

- [ ] **Step 4: Commit and push**

```bash
git add crates/agentic-server/src/auth.rs crates/agentic-server/src/handler/websocket/responses.rs \
  crates/agentic-server/tests/oidc_auth_test.rs docs/superpowers/specs docs/superpowers/plans
git commit -s -m "fix: address OIDC authentication review feedback"
git push --force-with-lease origin codex/issue-102-oidc-auth
```

Expected: the rewritten branch is pushed without overwriting an unexpected remote head.

- [ ] **Step 5: Update GitHub review state**

Reply in each inline thread with the exact fix and focused test, resolve only those four completed threads, update the
PR Summary/Test Plan, and monitor DCO, pre-commit, Rust, and container checks until terminal.
