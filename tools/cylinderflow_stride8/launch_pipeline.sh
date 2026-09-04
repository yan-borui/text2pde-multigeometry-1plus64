#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: launch_pipeline.sh PYTHON_BIN HDF5 MANIFEST NORMALIZER RESULT_ROOT RUN_ID GPU_ID" >&2
  exit 2
fi

python_input=$1
python_dir=$(cd "$(dirname "$python_input")" && pwd)
python_bin="$python_dir/$(basename "$python_input")"
data_path=$(realpath "$2")
manifest_path=$(realpath "$3")
normalizer_path=$(realpath "$4")
result_root=$5
run_id=$6
gpu_id=$7
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ae_session="text2pde_stride8_joint64_ae_${run_id}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is missing: $python_bin" >&2
  exit 2
fi
for required_file in "$data_path" "$manifest_path" "$normalizer_path"; do
  if [[ ! -f "$required_file" ]]; then
    echo "required data artifact is missing: $required_file" >&2
    exit 2
  fi
done
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer" >&2
  exit 2
fi

mkdir -p "$result_root/config" "$result_root/identity" "$result_root/logs" "$result_root/ae"
result_root=$(realpath "$result_root")
cd "$repo_root"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "formal launch requires a clean Text2PDE source tree" >&2
  exit 10
fi
if tmux has-session -t "$ae_session" 2>/dev/null; then
  echo "tmux session already exists: $ae_session" >&2
  exit 11
fi

"$python_bin" -m tools.cylinderflow_stride8.verify_data \
  --data "$data_path" \
  --manifest "$manifest_path" \
  > "$result_root/identity/data_verification.json"
"$python_bin" -m tools.cylinderflow_stride8.materialize_config \
  --template "$repo_root/configs/cylinderflow_stride8/ae_1plus64.yaml" \
  --data "$data_path" \
  --manifest "$manifest_path" \
  --normalizer "$normalizer_path" \
  --result-root "$result_root" \
  --stage ae \
  --output "$result_root/config/ae_1plus64.yaml"
"$python_bin" -m tools.cylinderflow_stride8.materialize_config \
  --template "$repo_root/configs/cylinderflow_stride8/ldm_1plus64.yaml" \
  --data "$data_path" \
  --manifest "$manifest_path" \
  --normalizer "$normalizer_path" \
  --result-root "$result_root" \
  --stage ldm \
  --output "$result_root/config/ldm_1plus64.yaml"

git rev-parse HEAD > "$result_root/identity/training_commit.txt"
git status --short > "$result_root/identity/training_git_status.txt"
"$python_bin" -c "import h5py, lightning, numpy, open3d, torch, torch_scatter, yaml; import sys; print(f'python={sys.version.split()[0]}'); print(f'torch={torch.__version__}'); print(f'cuda={torch.version.cuda}'); print(f'lightning={lightning.__version__}'); print(f'numpy={numpy.__version__}'); print(f'h5py={h5py.__version__}'); print(f'open3d={open3d.__version__}'); print(f'torch_scatter={torch_scatter.__version__}'); print(f'pyyaml={yaml.__version__}')" \
  > "$result_root/identity/runtime_versions.txt"

tmux new-session -d -s "$ae_session" \
  "bash '$repo_root/tools/cylinderflow_stride8/run_ae_stage.sh' '$python_bin' '$result_root' '$run_id' '$gpu_id' > '$result_root/logs/ae_stage_wrapper.log' 2>&1"
printf '%s\n' "$ae_session" > "$result_root/ae/tmux_session.txt"
date --iso-8601=seconds > "$result_root/ae/launched_at.txt"
printf 'launched %s\n' "$ae_session"
