#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: run_ldm_stage.sh PYTHON_BIN DATA_DIR RESULT_ROOT SELECTED_AE_CHECKPOINT" >&2
  exit 2
fi

python_bin=$1
data_dir=$2
result_root=$3
selected_ae=$4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ldm_config="$result_root/config/ldm_1plus64.yaml"
prepared_dir="$data_dir/prepared"
train_dir="$result_root/ldm/formal"
selection_dir="$result_root/evaluation/ldm_selection_v1"
validation_dir="$result_root/evaluation/validation_v1"
test_dir="$result_root/evaluation/test_locked_v1"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

cd "$repo_root"
if [[ ! -f "$selected_ae" ]]; then
  echo "selected AE checkpoint is missing: $selected_ae" >&2
  exit 40
fi
mkdir -p "$train_dir" "$result_root/logs"
printf '%s\n' "$selected_ae" > "$train_dir/selected_ae_checkpoint.txt"
date --iso-8601=seconds > "$train_dir/started_at.txt"

resume_args=()
if [[ -f "$train_dir/checkpoints/last.ckpt" ]]; then
  resume_args=(--checkpoint "$train_dir/checkpoints/last.ckpt")
fi

"$python_bin" "$repo_root/train_ldm.py" \
  --config "$ldm_config" \
  --first-stage-checkpoint "$selected_ae" \
  "${resume_args[@]}" \
  > "$train_dir/train.log" 2>&1 &
train_pid=$!
printf '%s\n' "$train_pid" > "$train_dir/train.pid"
(
  printf 'timestamp,index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw\n'
  while kill -0 "$train_pid" 2>/dev/null; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep 30
  done
) > "$train_dir/gpu_resources.csv" 2> "$train_dir/gpu_resources.err" &
monitor_pid=$!
set +e
wait "$train_pid"
train_rc=$?
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
printf '%s\n' "$train_rc" > "$train_dir/train.exit"
date --iso-8601=seconds > "$train_dir/finished_at.txt"
if [[ "$train_rc" -ne 0 ]]; then
  exit "$train_rc"
fi

set +e
"$python_bin" -m tools.multigeometry.evaluate_ldm \
  --mode select \
  --config "$ldm_config" \
  --ae-checkpoint "$selected_ae" \
  --checkpoint-dir "$train_dir/checkpoints" \
  --manifest "$prepared_dir/validation_monitor_windows.json" \
  --output-dir "$selection_dir" \
  --seeds 0 1 2 \
  --ddim-steps 20 \
  > "$result_root/logs/ldm_selection.log" 2>&1
selection_rc=$?
set -e
printf '%s\n' "$selection_rc" > "$result_root/logs/ldm_selection.exit"
if [[ "$selection_rc" -ne 0 ]]; then
  exit "$selection_rc"
fi

selected_ldm=$(tr -d '\r\n' < "$selection_dir/selected_checkpoint.txt")
if [[ ! -f "$selected_ldm" ]]; then
  echo "selected LDM checkpoint is missing: $selected_ldm" >&2
  exit 41
fi

set +e
"$python_bin" -m tools.multigeometry.evaluate_ldm \
  --mode validation \
  --config "$ldm_config" \
  --ae-checkpoint "$selected_ae" \
  --checkpoint "$selected_ldm" \
  --manifest "$prepared_dir/validation_full_windows.json" \
  --output-dir "$validation_dir" \
  --seeds 0 1 2 \
  --ddim-steps 20 \
  --save-first-seed \
  --render-family-gifs \
  > "$result_root/logs/validation_full.log" 2>&1
validation_rc=$?
set -e
printf '%s\n' "$validation_rc" > "$result_root/logs/validation_full.exit"
if [[ "$validation_rc" -ne 0 ]]; then
  exit "$validation_rc"
fi

printf '%s\n' "$selected_ldm" > "$result_root/evaluation/locked_ldm_checkpoint.txt"
date --iso-8601=seconds > "$result_root/evaluation/test_unsealed_at.txt"
set +e
"$python_bin" -m tools.multigeometry.evaluate_ldm \
  --mode test \
  --config "$ldm_config" \
  --ae-checkpoint "$selected_ae" \
  --checkpoint "$selected_ldm" \
  --manifest "$prepared_dir/test_full_windows_sealed.json" \
  --output-dir "$test_dir" \
  --seeds 0 1 2 \
  --ddim-steps 20 \
  --save-first-seed \
  > "$result_root/logs/test_locked.log" 2>&1
test_rc=$?
set -e
printf '%s\n' "$test_rc" > "$result_root/logs/test_locked.exit"
exit "$test_rc"
