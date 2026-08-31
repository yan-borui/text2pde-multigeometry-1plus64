#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: run_gpu_smoke.sh PYTHON_BIN --output-dir OUTPUT_DIR [SMOKE_ARGS...]" >&2
  exit 2
fi

python_input=$1
python_dir=$(cd "$(dirname "$python_input")" && pwd)
python_bin="$python_dir/$(basename "$python_input")"
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is missing: $python_bin" >&2
  exit 2
fi
shift
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
output_dir=""
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  if [[ "${arguments[$index]}" == "--output-dir" ]]; then
    if ((index + 1 >= ${#arguments[@]})); then
      echo "--output-dir requires a value" >&2
      exit 2
    fi
    output_dir=${arguments[$((index + 1))]}
  fi
done
if [[ -z "$output_dir" ]]; then
  echo "run_gpu_smoke.sh requires --output-dir" >&2
  exit 2
fi

if [[ -d /usr/lib/wsl/lib ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
cd "$repo_root"
set +e
"$python_bin" -m tools.cylinderflow.smoke "$@"
smoke_rc=$?
set -e
if [[ "$smoke_rc" -ne 0 ]]; then
  exit "$smoke_rc"
fi
if [[ ! -s "$output_dir/summary.json" ]]; then
  echo "GPU smoke returned without writing summary.json" >&2
  exit 90
fi
