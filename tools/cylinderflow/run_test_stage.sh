#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: run_test_stage.sh PYTHON_BIN META_JSON TEST_TFRECORD TRAIN_DATA_DIR RESULT_ROOT TEST_DATA_DIR" >&2
  exit 2
fi

python_input=$1
python_dir=$(cd "$(dirname "$python_input")" && pwd)
python_bin="$python_dir/$(basename "$python_input")"
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is missing: $python_bin" >&2
  exit 2
fi
meta_json=$(realpath "$2")
test_tfrecord=$(realpath "$3")
train_data_dir=$(realpath "$4")
result_root=$(realpath "$5")
test_data_dir=$6
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
test_output="$result_root/evaluation/test_v1"
ldm_config="$result_root/config/ldm_1plus24_raw.yaml"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
if [[ -d /usr/lib/wsl/lib ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$repo_root"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Test launch requires the clean locked source tree" >&2
  exit 50
fi
locked_commit=$(tr -d '\r\n' < "$result_root/identity/training_commit.txt")
current_commit=$(git rev-parse HEAD)
if [[ "$locked_commit" != "$current_commit" ]]; then
  echo "Test source commit differs from the locked training commit" >&2
  exit 51
fi
if [[ -e "$test_data_dir" || -e "$test_output" ]]; then
  echo "Test output paths already exist" >&2
  exit 52
fi

"$python_bin" -m tools.cylinderflow.verify_test_gate \
  --result-root "$result_root" \
  --data-dir "$train_data_dir" \
  > "$result_root/identity/test_gate_verification.json"

"$python_bin" -m tools.cylinderflow.prepare_test_data \
  --meta "$meta_json" \
  --test-tfrecord "$test_tfrecord" \
  --sealed-manifest "$train_data_dir/prepared/test_full_rollout64_sealed.json" \
  --output-dir "$test_data_dir" \
  --confirm-access-test \
  > "$result_root/logs/test_materialization.log"

selected_ae=$(tr -d '\r\n' < "$result_root/evaluation/locked_ae_checkpoint.txt")
selected_ldm=$(tr -d '\r\n' < "$result_root/evaluation/locked_ldm_checkpoint.txt")
"$python_bin" -m tools.cylinderflow.evaluate_rollout \
  --mode test \
  --config "$ldm_config" \
  --ae-checkpoint "$selected_ae" \
  --checkpoint "$selected_ldm" \
  --manifest "$test_data_dir/prepared/test_full_rollout64.json" \
  --output-dir "$test_output" \
  --seeds 0 1 2 \
  --ddim-steps 20 \
  --save-first-seed \
  --render-representative-gifs \
  > "$result_root/logs/test_full.log"
date --iso-8601=seconds > "$result_root/evaluation/test_complete"
