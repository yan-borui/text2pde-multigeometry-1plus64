#!/usr/bin/env bash
# Evaluate completed Airfoil weights through the existing NAS wrapper.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
: "${RESULT_ROOT:?Set RESULT_ROOT to the completed Airfoil training run}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new sampling result directory}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
if [[ -n "${NAS_ROOT:-}" ]]; then
    export RAW_DATA_DIR=${RAW_DATA_DIR:-$NAS_ROOT/data/airfoil_raw}
    export DATA_DIR=${DATA_DIR:-$NAS_ROOT/data/airfoil_uvp_stride8}
else
    export RAW_DATA_DIR=${RAW_DATA_DIR:-/data/datasets/meshgraphnets/airfoil}
    export DATA_DIR=${DATA_DIR:-${RAW_DATA_DIR}_uvp_stride8}
fi
config=${CONFIG:-$RESULT_ROOT/config/ldm_1plus64.yaml}
if [[ -n "${CHECKPOINT:-}" ]]; then
    checkpoint=$CHECKPOINT
else
    selection="$RESULT_ROOT/evaluation/ldm_selection_v1/selected_checkpoint.txt"
    [[ -s "$selection" ]] || { printf 'Missing LDM selection: %s\n' "$selection" >&2; exit 2; }
    checkpoint=$(< "$selection")
fi
if [[ -n "${AE_CHECKPOINT:-}" ]]; then
    ae_checkpoint=$AE_CHECKPOINT
else
    selection="$RESULT_ROOT/ae/selection_v1/selected_checkpoint.txt"
    [[ -s "$selection" ]] || { printf 'Missing AE selection: %s\n' "$selection" >&2; exit 2; }
    ae_checkpoint=$(< "$selection")
fi
exec bash "$code_root/scripts/nas.sh" python "${SAMPLING_ENTRYPOINT:-sampling_ensemble.py}" \
    --config "$config" --checkpoint "$checkpoint" --ae-checkpoint "$ae_checkpoint" \
    --output-dir "$OUTPUT_DIR" --device "${DEVICE:-cuda:0}" "$@"
