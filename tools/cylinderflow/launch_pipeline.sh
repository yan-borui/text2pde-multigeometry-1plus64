#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: launch_pipeline.sh PYTHON_BIN DATA_DIR RESULT_ROOT RUN_ID PREFLIGHT_SUMMARY" >&2
  exit 2
fi

python_input=$1
python_dir=$(cd "$(dirname "$python_input")" && pwd)
python_bin="$python_dir/$(basename "$python_input")"
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is missing: $python_bin" >&2
  exit 2
fi
data_dir=$(realpath "$2")
result_root=$3
run_id=$4
preflight_summary=$(realpath "$5")
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ae_session="text2pde_raw_mgn_ae_${run_id}"

mkdir -p "$result_root/config" "$result_root/identity" "$result_root/logs" "$result_root/ae"
result_root=$(realpath "$result_root")
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "formal launch requires a clean source tree" >&2
  exit 10
fi

"$python_bin" -m tools.cylinderflow.verify_data \
  --data-dir "$data_dir" \
  > "$result_root/identity/data_verification.json"
"$python_bin" -m tools.cylinderflow.verify_preflight \
  --summary "$preflight_summary" \
  --prepared-dir "$data_dir/prepared" \
  > "$result_root/identity/preflight_verification.json"

"$python_bin" -m tools.cylinderflow.materialize_config \
  --template "$repo_root/configs/cylinderflow/ae_1plus24_raw.yaml" \
  --prepared-dir "$data_dir/prepared" \
  --result-root "$result_root" \
  --stage ae \
  --output "$result_root/config/ae_1plus24_raw.yaml"
"$python_bin" -m tools.cylinderflow.materialize_config \
  --template "$repo_root/configs/cylinderflow/ldm_1plus24_raw.yaml" \
  --prepared-dir "$data_dir/prepared" \
  --result-root "$result_root" \
  --stage ldm \
  --output "$result_root/config/ldm_1plus24_raw.yaml"

git rev-parse HEAD > "$result_root/identity/training_commit.txt"
git status --short > "$result_root/identity/training_git_status.txt"
"$python_bin" -c "import h5py, lightning, numpy, open3d, torch, torch_scatter, yaml; import sys; print(f'python={sys.version.split()[0]}'); print(f'torch={torch.__version__}'); print(f'cuda={torch.version.cuda}'); print(f'lightning={lightning.__version__}'); print(f'numpy={numpy.__version__}'); print(f'h5py={h5py.__version__}'); print(f'open3d={open3d.__version__}'); print(f'torch_scatter={torch_scatter.__version__}'); print(f'pyyaml={yaml.__version__}')" \
  > "$result_root/identity/runtime_versions.txt"

if tmux has-session -t "$ae_session" 2>/dev/null; then
  echo "tmux session already exists: $ae_session" >&2
  exit 11
fi

tmux new-session -d -s "$ae_session" \
  "bash '$repo_root/tools/cylinderflow/run_ae_stage.sh' '$python_bin' '$data_dir' '$result_root' '$run_id' > '$result_root/logs/ae_stage_wrapper.log' 2>&1"
printf '%s\n' "$ae_session" > "$result_root/ae/tmux_session.txt"
date --iso-8601=seconds > "$result_root/ae/launched_at.txt"
printf 'launched %s\n' "$ae_session"
