#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

gateway_url="${AGENTIC_GATEWAY_URL:-http://127.0.0.1:3020}"
model="${AGENTIC_MODEL:-${V_MODEL:-agentic-api}}"
codex_home="${AGENTIC_CODEX_HOME:-/tmp/codex-agentic-3020}"
codex_bin="${CODEX_BIN:-}"

if [[ -z "$codex_bin" ]]; then
  codex_bin="$(command -v codex || true)"
fi
if [[ -z "$codex_bin" || ! -x "$codex_bin" ]]; then
  echo 'error: Codex CLI not found; set CODEX_BIN=/path/to/codex' >&2
  exit 127
fi
for required_command in curl jq; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "error: $required_command is required to create the isolated Codex model catalog" >&2
    exit 127
  fi
done

managed_marker="$codex_home/.agentic-managed"
if [[ -e "$codex_home" && ! -f "$managed_marker" ]]; then
  echo "error: refusing to replace an unmanaged Codex home: $codex_home" >&2
  echo 'set AGENTIC_CODEX_HOME to a new dedicated path and retry' >&2
  exit 2
fi
mkdir -p "$codex_home"
: >"$managed_marker"

catalog_tmp="$(mktemp "$codex_home/model_catalog.json.tmp.XXXXXX")"
config_tmp="$(mktemp "$codex_home/config.toml.tmp.XXXXXX")"
trap 'rm -f "$catalog_tmp" "$config_tmp"' EXIT

client_version="$($codex_bin --version | awk '{print $NF}')"
if [[ -z "$client_version" ]]; then
  echo 'error: unable to determine the Codex CLI version' >&2
  exit 2
fi
models_url="${gateway_url%/}/v1/models?client_version=${client_version}"
curl --fail --silent --show-error "$models_url" | jq --exit-status \
  --arg model "$model" \
  '{models: [.models[] | select(.slug == $model)]}
  | if (.models | length) == 1
    then .
    else error("gateway model catalog must contain exactly one requested model")
    end' >"$catalog_tmp"

model_toml="$(jq --null-input --arg value "$model" '$value')"
catalog_toml="$(jq --null-input --arg value "$codex_home/model_catalog.json" '$value')"
base_url_toml="$(jq --null-input --arg value "${gateway_url%/}/v1" '$value')"
{
  printf 'model = %s\n' "$model_toml"
  printf 'model_provider = "agentic-api"\n'
  printf 'model_catalog_json = %s\n\n' "$catalog_toml"
  printf '[model_providers.agentic-api]\n'
  printf 'name = "Agentic API"\n'
  printf 'base_url = %s\n' "$base_url_toml"
  printf 'wire_api = "responses"\n'
  printf 'requires_openai_auth = false\n'
  printf 'supports_websockets = true\n'
} >"$config_tmp"

mv "$catalog_tmp" "$codex_home/model_catalog.json"
mv "$config_tmp" "$codex_home/config.toml"
trap - EXIT

export CODEX_HOME="$codex_home"
codex_command=("$codex_bin" -C "$repo_root" --search --disable image_generation)
if [[ "${AGENTIC_YOLO:-0}" == "1" || "${AGENTIC_YOLO:-}" == "true" ]]; then
  codex_command+=(--dangerously-bypass-approvals-and-sandbox)
fi
exec "${codex_command[@]}" "$@"
