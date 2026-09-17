#!/bin/bash
# Set up the code on this machine. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"

echo "== environment =="
source env.sh
python -c "import numpy, scipy, cvxpy; print('cvxpy', cvxpy.__version__, cvxpy.installed_solvers())"
echo "cores available: $(python -c 'import os; print(os.cpu_count())')"

echo "== fast tests =="
pip install -q pytest
pytest -q

echo "== verdict without a sweep (epistasis + literature) =="
python verdict.py | tail -25

echo
echo "Done. Next step:"
echo "  python run_local.py --preset pilot"
