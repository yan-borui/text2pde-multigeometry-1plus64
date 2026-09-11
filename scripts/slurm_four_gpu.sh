#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
set -euo pipefail
if [[ "$#" -lt 2 ]]; then
    echo "usage: sbatch scripts/slurm_four_gpu.sh PYTHON [run_four_gpu arguments]" >&2
    exit 2
fi
python_bin=$1
shift
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec "$python_bin" -m tools.cylinderflow_stride8.run_four_gpu "$@"

