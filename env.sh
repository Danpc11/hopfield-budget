#!/bin/bash
# Environment. Change this to match the modules on your cluster.
# module load python/3.11        # <-- uncomment and adapt

VENV="${VENV:-$HOME/.venvs/hopfield}"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install "numpy>=1.24" "scipy>=1.10" "cvxpy>=1.5" clarabel
fi
source "$VENV/bin/activate"

# For reproducible runs the solver must be single threaded and the Python hash
# seed must be fixed. RAYON is the one people forget: CLARABEL is written in Rust
# and opens its own thread pool, which the BLAS variables do not control.
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
