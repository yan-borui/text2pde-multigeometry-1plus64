#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
python_bin=${PYTHON:-python}
action=${1:-train}
case "$action" in
    data|prepare|train|resume|validate|test) ;;
    *) printf 'Usage: bash scripts/ellipse_4gpu.sh data|prepare|train|resume|validate|test\n' >&2; exit 2 ;;
esac
export RAW_DATA_DIR=${RAW_DATA_DIR:-/data/datasets/megaflow2d}
export DATA_DIR=${DATA_DIR:-${RAW_DATA_DIR}_ellipse_stride15}
if [[ "$action" != data && "$action" != prepare ]]; then
    : "${RESULT_ROOT:?Set RESULT_ROOT for this training run}"
    : "${CUDA_VISIBLE_DEVICES:?Set four allocated GPU IDs or use the scheduler mask}"
    IFS=',' read -r -a assigned <<< "$CUDA_VISIBLE_DEVICES"
    if (( ${#assigned[@]} != 4 )); then printf 'Exactly four GPUs are required.\n' >&2; exit 2; fi
    if [[ "$action" == resume && ! -d "$RESULT_ROOT" ]]; then
        printf 'Resume requires the existing RESULT_ROOT.\n' >&2; exit 2
    fi
    if [[ "$action" == train && -e "$RESULT_ROOT" ]]; then
        printf 'Choose a new RESULT_ROOT or use resume.\n' >&2; exit 2
    fi
fi
data_options=()
if [[ "${ELLIPSE_OFFLINE:-0}" == 1 ]]; then data_options=(--offline); fi
"$python_bin" -m megaflow_data.ensure --raw-dir "$RAW_DATA_DIR" --output-dir "$DATA_DIR" "${data_options[@]}"
if [[ "$action" == data ]]; then exit 0; fi
data="$DATA_DIR/ellipse_stride15_65frames.h5"
manifest="$DATA_DIR/ellipse_stride15_65frames_manifest.json"
extra=()
if [[ "$action" == resume ]]; then extra=(--resume); fi
split=validation
if [[ "$action" == test ]]; then split=test; fi
if [[ "$action" == validate || "$action" == test ]]; then
    export EVAL_DIR=${EVAL_DIR:-$RESULT_ROOT/${split}_ellipse_4gpu}
fi


if [[ "$action" == prepare ]]; then exit 0; fi
if [[ "$action" == validate || "$action" == test ]]; then
    ae_checkpoint=${AE_CHECKPOINT:-$(cat "$RESULT_ROOT/ae/selection_v1/selected_checkpoint.txt")}
    checkpoint=${CHECKPOINT:-$(cat "$RESULT_ROOT/evaluation/ldm_selection_v1/selected_checkpoint.txt")}
    exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=4 \
        --module tools.cylinderflow_stride8.evaluate_four_gpu --stage ldm --mode "$split" \
        --config "$RESULT_ROOT/config/ldm_1plus64.yaml" --ae-checkpoint "$ae_checkpoint" \
        --checkpoint "$checkpoint" --output-dir "$EVAL_DIR"
fi
exec "$python_bin" -m tools.cylinderflow_stride8.run_four_gpu --data "$data" --manifest "$manifest" \
    --normalizer "$DATA_DIR/text2pde_normalizer.pkl" --result-root "$RESULT_ROOT" "${extra[@]}"
