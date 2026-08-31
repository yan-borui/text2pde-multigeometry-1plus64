#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: run_ae_stage.sh PYTHON_BIN DATA_DIR RESULT_ROOT RUN_ID" >&2
  exit 2
fi

python_bin=$1
data_dir=$2
result_root=$3
run_id=$4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ae_config="$result_root/config/ae_1plus24_raw.yaml"
prepared_dir="$data_dir/prepared"
train_dir="$result_root/ae/formal"
selection_dir="$result_root/ae/selection_v1"
ldm_session="text2pde_raw_mgn_ldm_${run_id}"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
if [[ -d /usr/lib/wsl/lib ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$repo_root"
mkdir -p "$train_dir" "$result_root/logs"
date --iso-8601=seconds > "$train_dir/started_at.txt"
resume_args=()
if [[ -f "$train_dir/checkpoints/last.ckpt" ]]; then
  resume_args=(--checkpoint "$train_dir/checkpoints/last.ckpt")
fi

"$python_bin" "$repo_root/train_AE.py" \
  --config "$ae_config" \
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
"$python_bin" -m tools.cylinderflow.select_ae \
  --config "$ae_config" \
  --checkpoint-dir "$train_dir/checkpoints" \
  --manifest "$prepared_dir/validation_monitor_windows_25.json" \
  --output-dir "$selection_dir" \
  > "$result_root/logs/ae_selection.log" 2>&1
selection_rc=$?
set -e
printf '%s\n' "$selection_rc" > "$result_root/logs/ae_selection.exit"
if [[ "$selection_rc" -ne 0 ]]; then
  exit "$selection_rc"
fi

selected_checkpoint=$(tr -d '\r\n' < "$selection_dir/selected_checkpoint.txt")
if [[ ! -f "$selected_checkpoint" ]]; then
  echo "selected AE checkpoint is missing: $selected_checkpoint" >&2
  exit 31
fi
if tmux has-session -t "$ldm_session" 2>/dev/null; then
  echo "tmux session already exists: $ldm_session" >&2
  exit 32
fi

mkdir -p "$result_root/ldm"
tmux new-session -d -s "$ldm_session" \
  "bash '$repo_root/tools/cylinderflow/run_ldm_stage.sh' '$python_bin' '$data_dir' '$result_root' '$selected_checkpoint' > '$result_root/logs/ldm_stage_wrapper.log' 2>&1"
printf '%s\n' "$ldm_session" > "$result_root/ldm/tmux_session.txt"
date --iso-8601=seconds > "$result_root/ldm/launched_at.txt"
