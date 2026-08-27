#!/usr/bin/env python3
"""Structurally validate every recorded cassette under a directory.

Cassettes are YAML files with a ``turns`` list (see
crates/agentic-server-core/tests/cassettes/README.md). This checks the parts
every replay test relies on, independent of which upstream recorded them:

- ``turns`` is a non-empty list; each turn names a request ``method`` and
  ``path`` and carries a JSON-object ``body``.
- Each response has an integer ``status_code`` and exactly one of ``body``
  (non-streaming) or ``sse`` (streaming).
- Every ``data:`` line in an ``sse`` recording is valid JSON, and a 2xx
  streaming recording terminates: a ``response.completed``-style event, a
  ``message_stop`` event (Messages), or a ``data: [DONE]`` marker.

Usage: scripts/validate-cassettes.py [CASSETTE_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

DEFAULT_DIRS = [Path("crates/agentic-server-core/tests/cassettes")]
TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed", "message_stop"}


def sse_events(sse: list[str], where: str) -> tuple[list[dict], bool]:
    """Return the decoded ``data:`` events and whether a ``[DONE]`` marker was seen."""
    events = []
    done = False
    for raw in sse:
        for line in raw.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                done = True
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError as error:
                raise ValueError(f"{where}: SSE data line is not JSON ({error}): {payload[:120]}") from error
    return events, done


def validate_turn(turn: dict, where: str) -> None:
    request = turn.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"{where}: missing request")
    if not isinstance(request.get("path"), str) or not request["path"].startswith("/"):
        raise ValueError(f"{where}: request.path must be an absolute path")
    if not isinstance(request.get("method"), str):
        raise ValueError(f"{where}: request.method missing")
    if not isinstance(request.get("body"), dict):
        raise ValueError(f"{where}: request.body must be a JSON object")

    response = turn.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{where}: missing response")
    status = response.get("status_code")
    if not isinstance(status, int):
        raise ValueError(f"{where}: response.status_code must be an integer")
    has_body = response.get("body") is not None
    has_sse = response.get("sse") is not None
    if has_body == has_sse:
        raise ValueError(f"{where}: response must have exactly one of body or sse")
    if has_sse:
        if not isinstance(response["sse"], list):
            raise ValueError(f"{where}: response.sse must be a list of raw SSE strings")
        events, done = sse_events(response["sse"], where)
        if 200 <= status < 300:
            if not events:
                raise ValueError(f"{where}: streaming response has no events")
            if not done and events[-1].get("type") not in TERMINAL_EVENTS:
                raise ValueError(f"{where}: streaming response does not end with a terminal event or [DONE]")


def validate_cassette(path: Path) -> int:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    turns = data.get("turns") if isinstance(data, dict) else None
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{path}: expected a non-empty turns list")
    for index, turn in enumerate(turns, start=1):
        validate_turn(turn, f"{path} turn {index}")
    return len(turns)


def main(argv: list[str]) -> int:
    roots = [Path(arg) for arg in argv] or DEFAULT_DIRS
    files = sorted(file for root in roots for file in root.rglob("*.yaml") if "turns:" in file.read_text("utf-8"))
    if not files:
        print(f"no cassettes found under {', '.join(map(str, roots))}", file=sys.stderr)
        return 1
    failures = 0
    total_turns = 0
    for file in files:
        try:
            total_turns += validate_cassette(file)
        except (ValueError, yaml.YAMLError) as error:
            failures += 1
            print(f"FAIL {error}", file=sys.stderr)
    print(f"validated {len(files) - failures}/{len(files)} cassettes, {total_turns} turns")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
