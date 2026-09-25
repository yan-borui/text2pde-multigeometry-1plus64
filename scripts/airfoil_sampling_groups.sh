#!/usr/bin/env bash
# Retain each method's established data, model and environment defaults.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export SAMPLING_ENTRYPOINT=sampling_groups.py
exec bash "$code_root/scripts/airfoil_sampling.sh" "$@"
