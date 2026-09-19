#!/usr/bin/env python3
"""
run_point.py -- evaluate ONE grid point and write one JSON file.

Useful for debugging a single point. To run a whole sweep on this machine use
run_local.py instead, which keeps the workers alive between points.

  python run_point.py --m 2 --F 50 --eps 4.2e-4 --J0 0.02 --nodes 60 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hopfield.runner import run_point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True, help="number of checking stages")
    ap.add_argument("--F", type=float, required=True, help="discrimination factor")
    ap.add_argument("--eps", type=float, required=True, help="error required")
    ap.add_argument("--J0", type=float, required=True, help="throughput required")
    ap.add_argument("--Atot", type=float, default=1.0, help="traffic budget")
    ap.add_argument("--tau", type=float, default=1e-8, help="minimum flux per edge")
    ap.add_argument("--nodes", type=int, default=60, help="branch and bound budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nsample", type=int, default=10, help="blind samples per node")
    ap.add_argument("--ndraw", type=int, default=300, help="samples in rate space")
    ap.add_argument("--solver", default="CLARABEL")
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    p = {k: v for k, v in vars(a).items() if k not in ("out", "force")}
    rec = run_point(p, out=a.out, force=a.force)
    if rec.get("status") == "skipped":
        print(f"[skip] {rec['path']}")
    else:
        print(f"[done] {os.path.basename(rec['path'])}  status={rec['status']}  "
              f"sigma={rec['sigma']}  {rec['seconds']}s")


if __name__ == "__main__":
    main()
