#!/bin/bash
# Launch the full hyperparameter-search training sweep in the background.
#
# Usage:
#   ./run_hpsearch_train.sh                    # use the already-active environment
#   GAPA_CONDA_ENV=gender ./run_hpsearch_train.sh
#
# Set GAPA_CONDA_ENV to a conda env name or path to activate it first; otherwise
# whichever python is on PATH is used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载 conda（必要时）/ load conda when an env is requested
if [[ -n "${GAPA_CONDA_ENV:-}" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null)" || {
        echo "conda not found on PATH; unset GAPA_CONDA_ENV or install conda." >&2
        exit 1
    }
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${GAPA_CONDA_ENV}"
fi

which python
python --version

# 跑训练任务 / run the training sweep
nohup python "${SCRIPT_DIR}/run_experiments.py" --full > "${SCRIPT_DIR}/train.log" 2>&1 &
disown
echo "Training launched; log: ${SCRIPT_DIR}/train.log"
