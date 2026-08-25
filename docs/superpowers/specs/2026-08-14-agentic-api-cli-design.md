# Agentic API CLI Design

## Goal

Create a polished `agentic` CLI that makes it easy to run Agentic API with coding harnesses. A user should be able to
start a gateway, optionally start vLLM, generate the correct isolated harness configuration, launch Claude Code or
Codex, and have all child processes shut down cleanly when the harness exits.

The existing `agentic-server` invocation remains supported while the new CLI becomes the recommended local workflow.

## User experience

The primary commands are:

```text
agentic run codex [OPTIONS] -- [CODEX_ARGS...]
agentic run claude [OPTIONS] -- [CLAUDE_ARGS...]
agentic serve [OPTIONS]
agentic validate [OPTIONS]
```

Examples:

```bash
agentic run codex --model Qwen/Qwen3.5 -- exec "inspect this repo"
agentic run claude --model Qwen/Qwen3.5
agentic run codex --upstream http://127.0.0.1:8000 --database-url sqlite://./agentic_codex.db
```

The CLI uses `--database-url` as the canonical storage option and accepts `DATABASE_URL` as its environment fallback.
Its default is `sqlite://./agentic_api.db`. PostgreSQL URLs are passed through unchanged; SQLite-only tuning is applied
only when the URL uses SQLite.

The top-level help and each subcommand help include a short description, the relevant options, and copyable examples.
`--quiet` suppresses lifecycle decoration and `--no-color` disables ANSI styling for CI and redirected output.

## Visual identity

Interactive output uses an electric-cyan and violet terminal palette:

- electric cyan: Agentic API branding and gateway information
- violet: selected harness and generated configuration
- green: readiness and successful shutdown
- amber: warnings and fallback behavior
- red: actionable failures

The startup banner is a fixed-width rounded frame with programmatically padded rows:

```text
┌──────────────────────────────┐
│  ⚡  Agentic API              │
│      Local agent gateway     │
└──────────────────────────────┘
```

Padding is calculated from terminal display width rather than hard-coded around the lightning glyph, avoiding the
right-border drift visible in terminals with different Unicode-width behavior.

## Architecture

The CLI is a thin orchestration layer around the existing gateway runtime:

1. Parse shared server, upstream, database, harness, and output options.
2. Create a temporary session directory for generated harness files and runtime state.
3. Resolve the upstream mode. `--upstream` connects to an already-running OpenAI-compatible service. `--model` names
   the harness model and, when `--upstream` is absent, also starts vLLM using the existing integrated-server behavior.
   When both are supplied, the existing upstream wins and `--model` configures the harness.
4. Start the gateway with the selected upstream and database URL.
5. Wait for the gateway and upstream readiness checks.
6. Generate an isolated harness configuration and environment overlay.
7. Launch the requested harness binary with user arguments after `--`.
8. Forward termination, wait for the harness exit, terminate the gateway and optional vLLM child, remove temporary
   files, and return the harness exit code.

The harness runner must never edit the user’s existing Codex or Claude Code configuration by default. Binary paths are
overridable with `AGENTIC_CODEX_BIN` and `AGENTIC_CLAUDE_BIN`.

`validate` performs the same preflight checks without starting the gateway or harness: it verifies the selected harness
binary, upstream or vLLM launch prerequisites, gateway port, database URL/connectivity, and migration readiness. It
reports actionable failures with secrets redacted.

## Harness configuration

### Codex

Generate an isolated `CODEX_HOME` containing the Agentic API provider configuration, model metadata/catalog required by
the installed Codex version, and the selected model. The generated provider points to the gateway’s `/v1` endpoint,
uses Responses wire format, enables the supported WebSocket path, and reflects whether the gateway requires OpenAI
authentication. User-supplied Codex arguments are appended unchanged.

### Claude Code

Generate an isolated environment/settings overlay containing `ANTHROPIC_BASE_URL`, the selected API key behavior, and
the selected model. The gateway endpoint is configured for Claude Messages compatibility. User-supplied Claude Code
arguments are appended unchanged.

The generated files and effective environment are shown in the startup summary without printing secret values.

## Storage

The CLI forwards `--database-url` to the gateway. SQLite is the zero-configuration default and PostgreSQL is intended
for shared or multi-worker sessions. The gateway runs migrations before the harness starts. A database connection or
migration failure stops the session before launching the harness and reports the database URL with credentials redacted.

## Error handling and lifecycle

- Missing harness binaries produce an actionable error naming the override environment variable.
- `--model` without a usable vLLM/Python launch path reports the failed command and remediation.
- A failed readiness check prevents harness startup and cleans up already-started children.
- Ctrl-C reaches the harness and gateway shutdown path; child processes are reaped.
- The exit status is the harness status when the harness starts successfully, and a distinct nonzero CLI status for
  setup/readiness failures.
- Secrets are never included in banners, help output, errors, or generated summaries.

## Testing

Unit and integration tests cover:

- command parsing, shared options, mutual exclusion, defaults, and help examples
- fixed-width banner rendering with ANSI enabled/disabled and Unicode-width edge cases
- Codex and Claude configuration generation without writing to real user homes
- SQLite defaulting and PostgreSQL URL pass-through/redaction
- process argument forwarding and lifecycle cleanup using test doubles or local fixture processes
- exit-code propagation and setup failure behavior

Existing server tests remain the regression suite for gateway behavior. The implementation must not modify unrelated
working-tree changes already present in the repository.
