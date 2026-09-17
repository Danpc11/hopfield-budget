"""
make_tasks.py -- build the list of grid points, one line per job.

You normally do NOT need this: run_local.py builds the same grid in memory and
runs it directly. Use this only if you want the task list as a text file, for
example to feed it to GNU parallel or to another scheduler.

The eps grid is centred on the theoretical wall 1/F^m and extends above it, on a
log scale. The wall is read LATER, from the edge of the grid. There is no serial
bisection, so a badly solved point does not spoil the others.

  python make_tasks.py --preset coarse > tasks.txt
  python make_tasks.py --preset refine --from-results results > tasks.txt
  cat tasks.txt | xargs -P 8 -I{} sh -c "{}"

The `nodes` sweep is not optional: it is the convergence check. Missing an
incumbent makes the wall look higher, so the estimate must GO DOWN as the budget
grows and then settle. Without several `nodes` values you cannot say anything
about the shape of the curve.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

# The eps grid must resolve the wall better than 1/5 of the excess we want to
# measure. At small J0 that excess is about 2%, so a uniform grid over two orders
# of magnitude does NOT work (step 0.14 in log, when we need 0.004).
# That is why the sweep has TWO STAGES:
#   coarse : finds the wall with a wide, cheap grid
#   refine : a fine grid around the wall found in the first stage
# The coupling is only in how tasks are generated, not inside any calculation:
# every point is still independent.

PRESETS = {
    # about 500 points: a few hours on 50 cores. Use it to check the pipeline.
    "pilot": dict(m=[2], F=[50.0], n_eps=18, eps_lo=0.9, eps_hi=60.0,
                  J0=[5e-3, 2e-2, 8e-2], nodes=[30, 60], seeds=[0, 1]),
    # the production run.
    "coarse": dict(m=[2, 3], F=[50.0], n_eps=34,
                       eps_lo=0.9, eps_hi=200.0,
                       J0=list(np.geomspace(1e-3, 0.18, 12)),
                       nodes=[30, 60, 120, 240], seeds=[0, 1, 2]),
}

# Stage 2: fine grid around the wall found in stage 1.
REFINE = dict(span=2.5,        # multiplicative window around the wall
              log_step=0.004,  # log step: resolves an excess of 2%
              nodes=[120, 240, 480], seeds=[0, 1, 2, 3])


def coarse_walls(results_dir):
    """(m,F,J0) -> smallest FEASIBLE eps found, at the largest budget."""
    import glob, json
    from collections import defaultdict
    best = defaultdict(dict)
    for p in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("status") != "feasible":
            continue
        k = (r["m"], r["F"], r["J0"])
        n = r["nodes"]
        best[k][n] = min(best[k].get(n, np.inf), r["eps"])
    return {k: v[max(v)] for k, v in best.items() if v}


def emit_refine(results_dir, out, script):
    walls = coarse_walls(results_dir)
    if not walls:
        raise SystemExit("no walls from the coarse stage: run --preset coarse first")
    n = 0
    for (m, F, J0), w in sorted(walls.items()):
        lo, hi = w / REFINE["span"], w * REFINE["span"]
        k = max(2, int(np.log(hi / lo) / REFINE["log_step"]) + 1)
        for eps in np.geomspace(lo, hi, k):
            for nodes in REFINE["nodes"]:
                for seed in REFINE["seeds"]:
                    print(f"python {script} --m {m} --F {F:g} --eps {eps:.8e} "
                          f"--J0 {J0:.8e} --nodes {nodes} --seed {seed} --out {out}")
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS) + ["refine"], default="pilot")
    ap.add_argument("--from-results", default="results",
                    help="for --preset refine: the coarse sweep already done")
    ap.add_argument("--out", default="results")
    ap.add_argument("--script", default="run_point.py")
    a = ap.parse_args()
    if a.preset == "refine":
        n = emit_refine(a.from_results, a.out, a.script)
        print(f"# {n} tasks (refine: fine grid around the coarse wall)",
              file=sys.stderr)
        return
    c = PRESETS[a.preset]

    n = 0
    for m, F in itertools.product(c["m"], c["F"]):
        wall = F ** (-m)
        eps_grid = np.geomspace(c["eps_lo"] * wall, c["eps_hi"] * wall, c["n_eps"])
        for eps, J0, nodes, seed in itertools.product(
                eps_grid, c["J0"], c["nodes"], c["seeds"]):
            print(f"python {a.script} --m {m} --F {F:g} --eps {eps:.8e} "
                  f"--J0 {J0:.8e} --nodes {nodes} --seed {seed} --out {a.out}")
            n += 1
    print(f"# {n} tasks ({a.preset})", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:      # for example when piping to head
        os._exit(0)
