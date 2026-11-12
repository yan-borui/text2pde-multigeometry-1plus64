#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
action=${1:-quality}
if (( $# > 0 )); then shift; fi
case "$action" in quality|benchmark) ;; *) printf 'Use quality or benchmark\n' >&2; exit 2 ;; esac
: "${RESULT_ROOT:?Set the completed training result root}"
: "${OUTPUT_DIR:?Set a new output directory}"
python_bin=${PYTHON:-python}
args=(--method text2pde --output-dir "$OUTPUT_DIR" --split "${SPLIT:-validation}")
checkpoint=${CHECKPOINT:-$(cat "$RESULT_ROOT/evaluation/ldm_selection_v1/selected_checkpoint.txt")}
ae_checkpoint=${AE_CHECKPOINT:-$(cat "$RESULT_ROOT/ae/selection_v1/selected_checkpoint.txt")}
args+=(--config "$RESULT_ROOT/config/ldm_1plus64.yaml" --checkpoint "$checkpoint" --ae-checkpoint "$ae_checkpoint")
if [[ "$action" == benchmark ]]; then
    exec "$python_bin" ellipse_sampling.py "${args[@]}" --benchmark "$@"
fi
exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=4 ellipse_sampling.py "${args[@]}" "$@"
