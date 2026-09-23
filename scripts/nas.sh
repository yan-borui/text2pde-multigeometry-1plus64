#!/usr/bin/env bash
# Use the selected environment and shared NAS paths for an existing entrypoint.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
if (( $# == 0 )); then
    printf 'Usage: bash scripts/nas.sh COMMAND [ARGUMENTS...]\nSee NAS.md for this experiment.\n' >&2
    exit 2
fi
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Set this before importing h5py, including in torchrun and scheduler children.
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-BEST_EFFORT}
if [[ -n "${NAS_ROOT:-}" ]]; then
    if [[ "$NAS_ROOT" != /* ]]; then
        printf 'NAS_ROOT must be an absolute shared directory.\n' >&2
        exit 2
    fi
fi
if [[ -f "$code_root/AIRFOIL.md" ]]; then
    if [[ -n "${NAS_ROOT:-}" ]]; then
        export RAW_DATA_DIR=${RAW_DATA_DIR:-$NAS_ROOT/data/airfoil_raw}
        export DATA_DIR=${DATA_DIR:-$NAS_ROOT/data/airfoil_uvp_stride8}
    fi
    : "${RAW_DATA_DIR:?Set NAS_ROOT or an absolute RAW_DATA_DIR; see NAS.md}"
    : "${DATA_DIR:?Set NAS_ROOT or an absolute DATA_DIR; see NAS.md}"
fi
if [[ "$1" == python ]]; then
    shift
    exec "${PYTHON:-python}" "$@"
fi
exec "$@"
