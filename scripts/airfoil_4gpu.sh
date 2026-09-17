#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
: "${DATA_DIR:?Set DATA_DIR to the prepared Airfoil directory}"
python_bin=${PYTHON:-python}
action=${1:-train}
data="$DATA_DIR/airfoil_stride8_75frames.h5"
manifest="$DATA_DIR/airfoil_stride8_75frames_manifest.json"

case "$action" in
    train|resume) ;;
    *) printf 'Usage: bash scripts/airfoil_4gpu.sh train|resume\n' >&2; exit 2 ;;
esac

: "${RESULT_ROOT:?Set a new RESULT_ROOT for this training run}"
: "${CUDA_VISIBLE_DEVICES:?Set four allocated GPU IDs or use the scheduler mask}"
IFS=',' read -r -a assigned <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#assigned[@]} != 4 )); then printf 'Exactly four GPUs are required.\n' >&2; exit 2; fi
extra=()
if [[ "$action" == resume ]]; then extra=(--resume); fi

exec "$python_bin" -m tools.cylinderflow_stride8.run_four_gpu --data "$data" --manifest "$manifest" --normalizer "$DATA_DIR/text2pde_normalizer.pkl" --result-root "$RESULT_ROOT" "${extra[@]}"
