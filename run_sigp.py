#!/usr/bin/env python3
"""
run_sigp.py -- the error wall by signomial programming, in parallel.

Faster and more accurate than the branch and bound: one optimisation per
(m, F, J0) instead of a feasibility question at every point of an eps grid. At
m=2, F=50 it reproduces the closed-form law to four decimals, where the branch
and bound needed about 190 s for three points and sat 1% high.

  python run_sigp.py --m 2 --F 50 --starts 8 --sweeps 4
  python run_sigp.py --m 3 --F 20 --starts 48 --sweeps 8 --jobs 32 --out m3.json

THREADS
-------
Every solve is forced to ONE thread, and the threading is set here in the script
rather than left to the shell, so the behaviour does not depend on how the job
was launched. The limits must be set before numpy, cvxpy or the solver are
imported, which is why they sit at the top of the file.

RAYON_NUM_THREADS matters as much as the BLAS ones: CLARABEL is written in Rust
and opens its own thread pool, which none of the OMP/MKL/OpenBLAS variables
control. Without it, every worker would spawn a pool of its own and they would
all fight for the same cores.

PARALLELISM
-----------
Phase 1 is embarrassingly parallel: every (J0, start) pair is an independent
solve.

Phase 2 is continuation, which is serial along a chain of throughputs, so the
parallelism comes from running several INDEPENDENT chains at once: half sweeping
up in J0 and half sweeping down, each from a different phase-1 solution. The two
directions matter. Sweeping up only, the early points keep whatever local optimum
the random starts found while later ones inherit better solutions, and the wall
comes out decreasing with J0, which is impossible. The chains are merged by
taking the best result at each J0, and the next round is seeded from that merge.
"""
from __future__ import annotations

import os

# ---- thread control: must happen before numpy, cvxpy or the solver load ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "RAYON_NUM_THREADS"):          # RAYON: CLARABEL is Rust and threads on its own
    os.environ[_v] = "1"                  # forced, not a default

import argparse
import json
import signal
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hopfield.sigp import wall_sigp
from hopfield import analytic as an

_CFG = {}


def _init(cfg):
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[_v] = "1"              # again, in case the pool used spawn
    _CFG.update(cfg)
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the parent handles Ctrl-C


def _solve_one(task):
    """One independent solve: (J0, seed, x0 or None)."""
    J0, seed, x0 = task
    r = wall_sigp(_CFG["m"], _CFG["F"], J0, A=_CFG["A"], seed=seed, x0=x0,
                  iters=_CFG["iters"])
    return J0, (r["eps"], r["x"]) if r["eps"] is not None else (None, None)


def _run_chain(task):
    """One continuation chain: a direction and a starting point, walked serially
    over the whole list of throughputs."""
    direction, x0, J_list = task
    order = J_list if direction > 0 else list(reversed(J_list))
    warm, out = x0, {}
    for J0 in order:
        r = wall_sigp(_CFG["m"], _CFG["F"], J0, A=_CFG["A"], x0=warm,
                      iters=_CFG["iters"], seed=0)
        if r["eps"] is not None:
            out[J0] = (r["eps"], r["x"])
            warm = r["x"]
    return out


def merge(best, new):
    improved = False
    for J0, (eps, x) in new.items():
        if eps is None:
            continue
        if best.get(J0) is None or eps < best[J0][0] * (1 - 1e-9):
            best[J0] = (eps, x)
            improved = True
    return improved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2, help="number of checking stages")
    ap.add_argument("--F", type=float, default=50.0, help="discrimination factor")
    ap.add_argument("--J0", type=float, nargs="*", default=None,
                    help="throughputs; default is a log grid below the collapse point")
    ap.add_argument("--n-J0", type=int, default=8)
    ap.add_argument("--A", type=float, default=1.0, help="traffic budget")
    ap.add_argument("--starts", type=int, default=8, help="random starts per point")
    ap.add_argument("--sweeps", type=int, default=4, help="rounds of continuation")
    ap.add_argument("--chains", type=int, default=0,
                    help="chains per round (0 = one per worker)")
    ap.add_argument("--iters", type=int, default=40, help="condensation steps")
    ap.add_argument("--jobs", type=int, default=0, help="workers (0 = all cores)")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    a = ap.parse_args()

    jobs = a.jobs or (os.cpu_count() or 1)
    chains = a.chains or jobs
    Jc = a.A / (2 * (a.m + 1))                 # analytic collapse point
    J0 = [float(x) for x in (a.J0 or np.geomspace(Jc / 60, Jc * 0.55, a.n_J0))]
    J0.sort()

    print(f"m={a.m}  F={a.F:g}  A={a.A:g}   wall 1/F^m = {a.F ** -a.m:.3e}")
    print(f"analytic collapse J_c = A/(2(m+1)) = {Jc:.4f}")
    print(f"{len(J0)} throughputs, {a.starts} starts, {a.sweeps} rounds, "
          f"{chains} chains, {jobs} workers, 1 thread each\n")

    cfg = dict(m=a.m, F=a.F, A=a.A, iters=a.iters)
    best, t0 = {}, time.time()
    pool = Pool(jobs, initializer=_init, initargs=(cfg,))
    try:
        # ---- phase 1: independent solves, embarrassingly parallel ----
        tasks = [(j, s, None) for j in J0 for s in range(a.starts)]
        done = 0
        for J, res in pool.imap_unordered(_solve_one, tasks, chunksize=1):
            merge(best, {J: res})
            done += 1
            print(f"\rphase 1: {done}/{len(tasks)}  "
                  f"points solved {sum(1 for v in best.values() if v)}/{len(J0)}",
                  end="", flush=True)
        print()

        seeds = [v[1] for v in best.values() if v]
        if not seeds:
            print("no feasible point found; raise --starts")
            return

        # ---- phase 2: parallel continuation chains, both directions ----
        for rnd in range(a.sweeps):
            tsk = [(1 if c % 2 == 0 else -1,
                    seeds[c % len(seeds)], J0) for c in range(chains)]
            improved = False
            for out in pool.imap_unordered(_run_chain, tsk, chunksize=1):
                improved |= merge(best, out)
            seeds = [v[1] for v in best.values() if v] or seeds
            cov = sum(1 for v in best.values() if v)
            print(f"round {rnd + 1}/{a.sweeps}: {cov}/{len(J0)} points, "
                  f"{'improved' if improved else 'no change'}")
            if not improved:
                break
        pool.close()
    except KeyboardInterrupt:
        print("\nstopped by user; reporting the best found so far")
        pool.terminate()
    finally:
        pool.join()

    dt = time.time() - t0
    print(f"\n{'J0':>10} {'eps':>12} {'eps/(1/F^m)':>13} {'law':>9} {'monotone':>9}")
    prev, bad, res = None, 0, []
    for j in J0:
        v = best.get(j)
        eps = v[0] if v else None
        w = (eps * a.F ** a.m) if eps else None
        law = (an.wall(j, a.F) * a.F ** a.m) if a.m == 2 else None
        mono = "-" if prev is None else ("yes" if (w and prev and w >= prev * 0.999)
                                         else "NO")
        bad += int(mono == "NO")
        print(f"{j:10.4g} {eps if eps else float('nan'):12.4e} "
              f"{w if w else float('nan'):13.4f} "
              f"{law if law else float('nan'):9.4f} {mono:>9}")
        res.append(dict(J0=j, eps=eps, wall_ratio=w))
        prev = w if w else prev

    print(f"\n[{dt:.0f} s on {jobs} workers]")
    if bad:
        print(f"!! {bad} non-monotone points: NOT converged. "
              f"More throughput cannot lower the error floor. "
              f"Raise --starts and --sweeps.")
    if a.m == 2:
        ok = [r for r in res if r["wall_ratio"]]
        if ok:
            d = max(abs(r["wall_ratio"] - an.wall(r["J0"], a.F) * a.F ** 2)
                    / (an.wall(r["J0"], a.F) * a.F ** 2) for r in ok)
            print(f"largest gap against the closed-form law: {d:.2%}")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(m=a.m, F=a.F, A=a.A, starts=a.starts, sweeps=a.sweeps,
                           chains=chains, iters=a.iters, jobs=jobs,
                           seconds=round(dt, 1), Jc_analytic=Jc, points=res),
                      fh, indent=1)
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()
