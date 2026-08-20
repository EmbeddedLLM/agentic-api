# Agentic API CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Add a polished `agentic` binary that starts Agentic API, generates isolated Claude Code or Codex configuration,
launches the selected harness, and cleans up the session.

**Architecture:** Keep `agentic-server` as the existing gateway/vLLM runtime. Add a sibling `agentic` binary in the same
package that resolves `--upstream` and `--model` source/model options, starts `agentic-server` as a child process,
waits on gateway readiness, writes a temporary harness environment, and owns the harness/server child lifecycle.

**Tech Stack:** Rust 2024, Tokio process and signal APIs, Clap derive parsing, existing reqwest HTTP client, ANSI terminal
sequences without a new color dependency, and the existing SQLite/PostgreSQL `DATABASE_URL` path.

**Spec:** `docs/superpowers/specs/2026-08-14-agentic-api-cli-design.md`

## Global Constraints

- Preserve the existing `agentic-server` command and unrelated working-tree changes.
- Use `--database-url` with default `sqlite://./agentic_api.db` and `DATABASE_URL` fallback.
- Require `--upstream` or `--model`; when both are supplied, use the existing upstream and use `--model` as the harness model.
- Never modify a user’s real Codex or Claude configuration by default.
- Redact credentials in displayed URLs and errors.
- Support `--quiet` and `--no-color`; never emit ANSI styling when disabled.
- Follow Rust 2024, MSRV 1.85, `unsafe_code = "forbid"`, and existing clippy rules.
- Write tests before production code and verify each red-green cycle.

### Task 1: Add the `agentic` binary and CLI parser

**Files:**
- Modify: `crates/agentic-server/Cargo.toml`
- Create: `crates/agentic-server/src/bin/agentic.rs`
- Create: `crates/agentic-server/src/agentic_cli.rs`
- Test: `crates/agentic-server/src/agentic_cli.rs` unit tests

**Interfaces:**
- Produces `Cli`, `Command`, `Harness`, and `CommonOptions` types used by all later tasks.
- `Cli::parse_from` accepts `run codex`, `run claude`, `serve`, and `validate`.
- `SourceOptions` exposes `upstream: Option<String>`, `model: Option<String>`, and `llm_port`; `CommonOptions` exposes
  `database_url: String`, gateway host/port, API key, readiness controls, `quiet`, and `no_color`.

- [ ] Write failing parser tests for default SQLite, `DATABASE_URL`, harness selection, passthrough arguments after `--`,
  upstream/model combinations, and required source selection.
- [ ] Run `cargo test -p agentic-server agentic_cli` and confirm the new module/types are missing or assertions fail.
- [ ] Add the sibling `agentic` bin target and implement Clap structs with examples in top-level and subcommand help.
- [ ] Parse environment defaults without exposing secret values in generated help.
- [ ] Run the focused parser tests and `cargo test -p agentic-server --lib`.

### Task 2: Implement terminal presentation and URL redaction

**Files:**
- Create: `crates/agentic-server/src/agentic_output.rs`
- Modify: `crates/agentic-server/src/bin/agentic.rs`
- Test: `crates/agentic-server/src/agentic_output.rs` unit tests

**Interfaces:**
- `render_banner(color: bool) -> String` returns the fixed-width Agentic API frame.
- `redact_url(url: &str) -> String` removes password and credential-bearing URL components.
- `SessionPrinter` prints cyan branding, violet harness details, green readiness, amber warnings, and red failures while
  honoring quiet/no-color modes.

- [ ] Write failing tests for equal banner row widths, no ANSI escapes with `color = false`, and password redaction.
- [ ] Verify the tests fail for the absent renderer.
- [ ] Implement display-width padding around the lightning glyph and ANSI helpers with no output in quiet mode.
- [ ] Run the focused output tests and inspect `cargo run -p agentic-server --bin agentic -- --help`.

### Task 3: Generate isolated Codex and Claude configuration

**Files:**
- Create: `crates/agentic-server/src/agentic_harness.rs`
- Modify: `crates/agentic-server/src/bin/agentic.rs`
- Test: `crates/agentic-server/src/agentic_harness.rs` unit tests

**Interfaces:**
- `Harness::command_name(&self) -> &'static str` returns `codex` or `claude`.
- `prepare_codex_home(root: &Path, gateway_url: &str, model: &str, api_key: Option<&str>) -> Result<HarnessEnv, Error>`
  writes an isolated `config.toml` and model catalog path.
- `prepare_claude_env(gateway_url: &str, model: &str, api_key: Option<&str>) -> HarnessEnv` returns environment overrides
  for `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL`, and authentication behavior.
- `HarnessEnv` contains environment pairs, generated file paths, and a safe display summary.

- [ ] Write failing tests asserting Codex provider fields (`base_url`, `wire_api`, WebSockets, auth behavior), selected
  model, Claude environment fields, and absence of secret values from summaries.
- [ ] Verify the tests fail before writing files.
- [ ] Implement deterministic file generation under a caller-provided temporary directory; do not touch real home paths.
- [ ] Fetch the gateway’s `/v1/models` response only when needed to create the Codex model catalog, preserving the selected
  model in the generated catalog.
- [ ] Run focused generation tests, including a PostgreSQL URL summary test through the shared redactor.

### Task 4: Add gateway and harness process lifecycle

**Files:**
- Create: `crates/agentic-server/src/agentic_process.rs`
- Modify: `crates/agentic-server/src/bin/agentic.rs`
- Test: `crates/agentic-server/src/agentic_process.rs` unit tests and a small fixture-process integration test

**Interfaces:**
- `server_command(current_exe: &Path, options: &CommonOptions) -> Command` builds the sibling `agentic-server` invocation.
- `wait_for_gateway(client: &reqwest::Client, base_url: &str, timeout: Duration) -> Result<()>` waits for `/health` and
  `/ready` according to the selected readiness mode.
- `run_session(options: SessionOptions) -> Result<ExitCode>` starts server, waits, prepares harness, launches harness,
  forwards termination, reaps children, and returns the harness exit status.

- [ ] Write failing tests for server argument construction, gateway polling success/failure, harness argument forwarding,
  and exit-code propagation.
- [ ] Verify focused tests fail because lifecycle functions are absent.
- [ ] Implement process spawning with `tokio::process::Command`, inherited stdio, Ctrl-C handling, child reaping, and
  cleanup on every setup failure.
- [ ] Launch integrated mode through `agentic-server serve <model>` and standalone mode through `agentic-server
  --llm-api-base <upstream>`; pass host, port, API key, database, and readiness flags.
- [ ] Run lifecycle tests with fixture commands that exit successfully and unsuccessfully.

### Task 5: Implement `run`, `serve`, and `validate`

**Files:**
- Modify: `crates/agentic-server/src/bin/agentic.rs`
- Modify: `crates/agentic-server/src/agentic_cli.rs`
- Test: `crates/agentic-server/tests/agentic_cli_test.rs`

**Interfaces:**
- `run codex` selects the Codex binary from `AGENTIC_CODEX_BIN` or PATH and launches it with passthrough args.
- `run claude` selects the Claude Code binary from `AGENTIC_CLAUDE_BIN` or PATH and launches it with passthrough args.
- `serve` starts the gateway and waits for Ctrl-C without launching a harness.
- `validate` checks selected binaries, source mode, port availability, database URL/connectivity/migrations, and model
  launch prerequisites without starting the harness.

- [ ] Write failing integration tests for `--help`, `run codex`/`run claude` parsing, `validate` source errors, and a fake
  harness exit code.
- [ ] Verify the tests fail with the new binary absent or behavior unimplemented.
- [ ] Wire the commands to the parser, output, harness, and process modules.
- [ ] Add actionable missing-binary and readiness errors with redacted URLs.
- [ ] Run the integration tests and manually inspect `agentic validate --help` and `agentic run --help`.

### Task 6: Documentation, formatting, and full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/developing/getting-started.md`
- Modify: `crates/agentic-server/src/main.rs` only if compatibility help text needs adjustment

- [ ] Add the short workflow to README: `cargo run -p agentic-server --bin agentic -- run codex --model ...` and installed
  `agentic run claude --model ...` forms, including SQLite default and PostgreSQL `--database-url`.
- [ ] Document `agentic validate`, binary override variables, `--quiet`, and `--no-color`.
- [ ] Run `cargo fmt -- --check` and fix formatting.
- [ ] Run `cargo test -p agentic-server`.
- [ ] Run `cargo clippy --all-targets -- -D warnings`.
- [ ] Run `git diff --check` and review only the intended CLI/spec/docs files before reporting completion.
