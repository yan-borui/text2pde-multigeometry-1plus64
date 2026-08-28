#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: launch_pipeline.sh PYTHON_BIN DATA_DIR RESULT_ROOT RUN_ID" >&2
  exit 2
fi

python_bin=$(realpath "$1")
data_dir=$(realpath "$2")
result_root=$3
run_id=$4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ae_session="text2pde_ae_${run_id}"

mkdir -p "$result_root/config" "$result_root/identity" "$result_root/logs" "$result_root/ae"
result_root=$(realpath "$result_root")

cd "$repo_root"
"$python_bin" -m tools.multigeometry.verify_handoff_data \
  --data-dir "$data_dir" \
  > "$result_root/identity/data_verification.json"

"$python_bin" -m tools.multigeometry.materialize_config \
  --template "$repo_root/configs/multigeometry/ae_1plus64.yaml" \
  --prepared-dir "$data_dir/prepared" \
  --result-root "$result_root" \
  --stage ae \
  --output "$result_root/config/ae_1plus64.yaml"
"$python_bin" -m tools.multigeometry.materialize_config \
  --template "$repo_root/configs/multigeometry/ldm_1plus64.yaml" \
  --prepared-dir "$data_dir/prepared" \
  --result-root "$result_root" \
  --stage ldm \
  --output "$result_root/config/ldm_1plus64.yaml"

git -C "$repo_root" rev-parse HEAD > "$result_root/identity/training_commit.txt"
git -C "$repo_root" status --short > "$result_root/identity/training_git_status.txt"
"$python_bin" -c "import h5py, lightning, numpy, open3d, torch, torch_scatter, yaml; import sys; print(f'python={sys.version.split()[0]}'); print(f'torch={torch.__version__}'); print(f'cuda={torch.version.cuda}'); print(f'lightning={lightning.__version__}'); print(f'numpy={numpy.__version__}'); print(f'h5py={h5py.__version__}'); print(f'open3d={open3d.__version__}'); print(f'torch_scatter={torch_scatter.__version__}'); print(f'pyyaml={yaml.__version__}')" \
  > "$result_root/identity/runtime_versions.txt"

if tmux has-session -t "$ae_session" 2>/dev/null; then
  echo "tmux session already exists: $ae_session" >&2
  exit 3
fi

tmux new-session -d -s "$ae_session" \
  "bash '$repo_root/tools/multigeometry/run_ae_stage.sh' '$python_bin' '$data_dir' '$result_root' '$run_id' > '$result_root/logs/ae_stage_wrapper.log' 2>&1"
printf '%s\n' "$ae_session" > "$result_root/ae/tmux_session.txt"
date --iso-8601=seconds > "$result_root/ae/launched_at.txt"
printf 'launched %s\n' "$ae_session"
