#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
WORKLOAD="${WORKLOAD:-history-rehydration}"
REQUESTS_PER_SESSION="${REQUESTS_PER_SESSION:-}"
DEPTHS="${DEPTHS:-}"
SESSIONS="${SESSIONS:-5}"
TRANSPORT_ROUNDS="${TRANSPORT_ROUNDS:-4}"
PROVIDER="${PROVIDER:-both}"
AGENTIC_URL="${AGENTIC_URL:-http://localhost:9000/v1}"
VLLM_URL="${VLLM_URL:-http://localhost:5050/v1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
SEED="${SEED:-20260817}"
RESULTS_ROOT="${RESULTS_ROOT:-${repository_root}/target/response-provider-benchmark}"
DATASET_QUESTIONS="${DATASET_QUESTIONS:-}"
DATASET_ANSWERS="${DATASET_ANSWERS:-}"
DATASET_OFFSET="${DATASET_OFFSET:-0}"
BFCL_ROOT="${BFCL_ROOT:-}"
BFCL_CATEGORY="${BFCL_CATEGORY:-simple_python}"
PRINT_PROMPTS="${PRINT_PROMPTS:-0}"

usage() {
  cat <<'EOF'
Run optimized Responses capability benchmarks from the agentic-server bench target.

Usage:
  ./scripts/run-response-provider-benchmark.sh

Core environment variables:
  MODEL                  Model served by all selected endpoints.
  WORKLOAD               transport, tool-call, or history-rehydration.
  SESSIONS               Parallel sessions per provider (default: 5).
  REQUESTS_PER_SESSION   Sequential turns per session; unset uses the workload default.
  DEPTHS                 Comma-separated fixed turn depths (history-rehydration only), e.g.
                         1,5,10,25,50,100. Each depth runs as its own independent batch of SESSIONS
                         sessions instead of bucketing one long run after the fact, so sample counts
                         stay even across depths. Overrides REQUESTS_PER_SESSION when set. Produces
                         depth_summary.md/.json under RESULTS_ROOT with request/response bytes and
                         accuracy versus turn depth.
  PROVIDER               all, both, agentic-api, agentic-api-http, agentic-api-json,
                         vllm, or vllm-json.
  AGENTIC_URL            Agentic API base URL (default: http://localhost:9000/v1).
  VLLM_URL               Direct vLLM base URL (default: http://localhost:5050/v1).
  TRANSPORT_ROUNDS       Sequential model/tool rounds in each transport turn (default: 4).
  SEED                   Deterministic prompt/case seed shared across separate runs.
  PRINT_PROMPTS          Set to 1 to validate and print generated prompt JSONL without running.

BFCL tool-call workload:
  BFCL_ROOT              Checkout of https://github.com/EnlightenedAI/BFCL.
  BFCL_CATEGORY          v4 filename suffix, such as simple_python or multiple.
  DATASET_QUESTIONS      Explicit BFCL question JSONL; overrides BFCL_ROOT derivation.
  DATASET_ANSWERS        Explicit BFCL possible-answer JSONL.
  DATASET_OFFSET         First deterministic case index (default: 0).

Other controls:
  TIMEOUT_SECONDS RESULTS_ROOT

Examples:
  WORKLOAD=transport PROVIDER=all TRANSPORT_ROUNDS=8 ./scripts/run-response-provider-benchmark.sh

  WORKLOAD=transport PROVIDER=agentic-api-json ./scripts/run-response-provider-benchmark.sh

  WORKLOAD=tool-call PROVIDER=agentic-api BFCL_ROOT=/path/to/BFCL \
    REQUESTS_PER_SESSION=1 ./scripts/run-response-provider-benchmark.sh

EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# != 0 )); then
  echo "error: unexpected argument: $1" >&2
  usage >&2
  exit 2
fi

for command_name in cargo tee; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "error: ${command_name} is required" >&2
    exit 2
  fi
done

for numeric_value in SESSIONS TRANSPORT_ROUNDS TIMEOUT_SECONDS; do
  if [[ ! "${!numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ${numeric_value} must be a positive integer" >&2
    exit 2
  fi
done
if [[ -n "$REQUESTS_PER_SESSION" && ! "$REQUESTS_PER_SESSION" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: REQUESTS_PER_SESSION must be a positive integer when set" >&2
  exit 2
fi
if [[ -n "$DEPTHS" ]]; then
  if [[ ! "$DEPTHS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "error: DEPTHS must be a comma-separated list of positive integers" >&2
    exit 2
  fi
  if [[ "$WORKLOAD" != "history-rehydration" ]]; then
    echo "error: DEPTHS is only supported for WORKLOAD=history-rehydration" >&2
    exit 2
  fi
fi
if [[ ! "$DATASET_OFFSET" =~ ^[0-9]+$ || ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "error: DATASET_OFFSET and SEED must be non-negative integers" >&2
  exit 2
fi
case "$WORKLOAD" in
  transport | tool-call | history-rehydration) ;;
  *)
    echo "error: unsupported WORKLOAD: ${WORKLOAD}" >&2
    exit 2
    ;;
esac
case "$PROVIDER" in
  all | both | agentic-api | agentic-api-http | agentic-api-json | vllm | vllm-json) ;;
  *)
    echo "error: unsupported PROVIDER: ${PROVIDER}" >&2
    exit 2
    ;;
esac

if [[ "$WORKLOAD" == "tool-call" && -z "$DATASET_QUESTIONS" && -n "$BFCL_ROOT" ]]; then
  bfcl_data_dir="${BFCL_ROOT%/}/berkeley-function-call-leaderboard/bfcl_eval/data"
  DATASET_QUESTIONS="${bfcl_data_dir}/BFCL_v4_${BFCL_CATEGORY}.json"
  DATASET_ANSWERS="${bfcl_data_dir}/possible_answer/BFCL_v4_${BFCL_CATEGORY}.json"
fi
if [[ "$WORKLOAD" == "tool-call" && ( -z "$DATASET_QUESTIONS" || -z "$DATASET_ANSWERS" ) ]]; then
  echo "error: tool-call requires BFCL_ROOT or both DATASET_QUESTIONS and DATASET_ANSWERS" >&2
  exit 2
fi
for dataset_path in "$DATASET_QUESTIONS" "$DATASET_ANSWERS"; do
  if [[ -n "$dataset_path" && ! -f "$dataset_path" ]]; then
    echo "error: dataset file not found: ${dataset_path}" >&2
    exit 2
  fi
done
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir="${RESULTS_ROOT%/}/${run_stamp}"
mkdir -p "$run_dir"

benchmark_command=(
  cargo bench --package agentic-server --bench response_provider --
  --model "$MODEL"
  --workload "$WORKLOAD"
  --sessions "$SESSIONS"
  --transport-rounds "$TRANSPORT_ROUNDS"
  --provider "$PROVIDER"
  --agentic-url "$AGENTIC_URL"
  --vllm-url "$VLLM_URL"
  --timeout-seconds "$TIMEOUT_SECONDS"
  --seed "$SEED"
  --dataset-offset "$DATASET_OFFSET"
  --output-dir "$run_dir"
  --live-jsonl
)
if [[ -n "$DEPTHS" ]]; then
  benchmark_command+=(--depths "$DEPTHS")
elif [[ -n "$REQUESTS_PER_SESSION" ]]; then
  benchmark_command+=(--requests-per-session "$REQUESTS_PER_SESSION")
fi
if [[ -n "$DATASET_QUESTIONS" ]]; then
  benchmark_command+=(--dataset-questions "$DATASET_QUESTIONS" --dataset-answers "$DATASET_ANSWERS")
fi
if [[ "$PRINT_PROMPTS" == "1" ]]; then
  benchmark_command+=(--print-prompts)
fi

{
  printf 'cd %q\n' "$repository_root"
  printf '%q ' "${benchmark_command[@]}"
  printf '\n'
} >"${run_dir}/command.sh"

echo "Benchmark workload: ${WORKLOAD}" >&2
echo "Benchmark results:  ${run_dir}" >&2
echo "Live JSONL:        ${run_dir}/live-events.jsonl" >&2
echo "Diagnostic log:    ${run_dir}/benchmark.log" >&2

cd "$repository_root"
if "${benchmark_command[@]}" \
  > >(tee "${run_dir}/live-events.jsonl") \
  2> >(tee "${run_dir}/benchmark.log" >&2); then
  benchmark_status=0
else
  benchmark_status=$?
fi

echo "Benchmark exit status: ${benchmark_status}" >&2
echo "Summary: ${run_dir}/summary.md" >&2
exit "$benchmark_status"
