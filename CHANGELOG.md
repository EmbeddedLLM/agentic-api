# Changelog

All notable changes to Agentic API are documented here.

## [0.5.0] - 2026-08-25

### Changed

- Preserved Claude Code Messages transport fidelity across the gateway.
- Updated You.com web search integration to use GET query parameters.
- Aligned deployment and harness documentation with the 0.4.0 release.

### Testing

- Fixed web search test hangs in CI.

## [0.4.0] - 2026-08-23

### Added

- Added the Agentic API harness CLI for running Codex and Claude Code against Agentic API.
- Added home-based configuration and typed tool settings for standalone deployments.
- Added support for Codex CLI remote compaction V2.
- Added Kubernetes deployment guidance and architecture documentation.

### Changed

- Improved handling of Codex and Claude harness upstream configuration and compatible reasoning effort values.
- Preserved unsupported parallel tool calls through serialized upstream requests.
- Hardened MCP configuration and startup behavior.
- Improved Kubernetes health and readiness behavior for read-only container roots.

### Testing

- Added native Codex and Claude harness coverage and expanded compatibility tests.

## [0.3.0]

Initial documented release.
