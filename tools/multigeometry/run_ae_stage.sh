#!/usr/bin/env bash
set -euo pipefail

worktree=/home/shinku/projects/text2pde_matched_20260827_wt
result_root=/home/shinku/projects/dgn4cfd_runs/text2pde_multigeom_1plus64_20260827a
venv_python=/home/shinku/projects/text2pde_official_20260826/.venv/bin/python
ae_config="$worktree/configs/multigeometry/ae_1plus64.yaml"
prepared_dir="$result_root/data/prepared_v1"
train_dir="$result_root/ae/formal"
selection_dir="$result_root/ae/selection_v1"
ldm_session=text2pde_ldm_20260827a

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$worktree"
mkdir -p "$train_dir" "$result_root/logs"
git -C "$worktree" rev-parse HEAD > "$result_root/identity/training_commit.txt"
git -C "$worktree" status --short > "$result_root/identity/training_git_status.txt"
date --iso-8601=seconds > "$train_dir/started_at.txt"

resume_args=()
if [[ -f "$train_dir/checkpoints/last.ckpt" ]]; then
  resume_args=(--checkpoint "$train_dir/checkpoints/last.ckpt")
fi

"$venv_python" "$worktree/train_AE.py" \
  --config "$ae_config" \
  "${resume_args[@]}" \
  > "$train_dir/train.log" 2>&1 &
train_pid=$!
printf '%s\n' "$train_pid" > "$train_dir/train.pid"
(
  printf 'timestamp,index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw\n'
  while kill -0 "$train_pid" 2>/dev/null; do
    /usr/lib/wsl/lib/nvidia-smi \
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
"$venv_python" -m tools.multigeometry.select_ae \
  --config "$ae_config" \
  --checkpoint-dir "$train_dir/checkpoints" \
  --manifest "$prepared_dir/validation_monitor_windows.json" \
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
  echo "refusing to duplicate existing tmux session: $ldm_session" >&2
  exit 32
fi

tmux new-session -d -s "$ldm_session" \
  "bash '$worktree/tools/multigeometry/run_ldm_stage.sh' '$selected_checkpoint' > '$result_root/logs/ldm_stage_wrapper.log' 2>&1"
printf '%s\n' "$ldm_session" > "$result_root/ldm/tmux_session.txt"
date --iso-8601=seconds > "$result_root/ldm/launched_at.txt"
exit 0
