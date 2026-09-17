#!/usr/bin/env python3
"""
run_local.py -- run the whole sweep on one machine, using all its cores.

No scheduler needed. It builds the grid, skips the points that are already done,
and runs the rest in a process pool.

  python run_local.py --preset pilot                 # check the pipeline
  python run_local.py --preset coarse --jobs 16      # stage 1
  python run_local.py --preset refine --jobs 16      # stage 2 (needs stage 1)

You can stop it with Ctrl-C at any time. Nothing is lost: every finished point is
already on disk, and running the same command again continues where it stopped.

Each solve is single threaded on purpose, and the parallelism comes from the
pool. That is why BLAS is limited to one thread: otherwise every worker would
open its own threads and they would fight for the cores.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # must be set before numpy is imported

import argparse
import itertools
import signal
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hopfield.runner import run_point, already_done
from make_tasks import PRESETS, REFINE, coarse_walls


def build_grid(preset: str, out: str):
    """Return the list of points for a preset."""
    if preset == "refine":
        walls = coarse_walls(out)
        if not walls:
            raise SystemExit("no walls from the coarse stage: "
                             "run --preset coarse first")
        pts = []
        for (m, F, J0), w in sorted(walls.items()):
            lo, hi = w / REFINE["span"], w * REFINE["span"]
            k = max(2, int(np.log(hi / lo) / REFINE["log_step"]) + 1)
            for eps in np.geomspace(lo, hi, k):
                for nodes in REFINE["nodes"]:
                    for seed in REFINE["seeds"]:
                        pts.append(dict(m=m, F=F, eps=float(eps), J0=float(J0),
                                        nodes=nodes, seed=seed))
        return pts

    c = PRESETS[preset]
    pts = []
    for m, F in itertools.product(c["m"], c["F"]):
        wall = F ** (-m)
        eps_grid = np.geomspace(c["eps_lo"] * wall, c["eps_hi"] * wall, c["n_eps"])
        for eps, J0, nodes, seed in itertools.product(
                eps_grid, c["J0"], c["nodes"], c["seeds"]):
            pts.append(dict(m=m, F=float(F), eps=float(eps), J0=float(J0),
                            nodes=nodes, seed=seed))
    return pts


def _work(args):
    p, out = args
    try:
        r = run_point(p, out=out)
        return p, r.get("status"), r.get("seconds", 0.0)
    except Exception as ex:
        return p, f"crash: {ex!r}", 0.0


def fmt(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pilot",
                    choices=sorted(PRESETS) + ["refine"])
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes (0 = all cores minus one)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--chunk", type=int, default=4,
                    help="points sent to a worker at a time")
    a = ap.parse_args()

    jobs = a.jobs or max(1, (os.cpu_count() or 2) - 1)
    pts = build_grid(a.preset, a.out)
    os.makedirs(a.out, exist_ok=True)
    todo = [p for p in pts if not already_done(p, a.out)]

    print(f"preset '{a.preset}': {len(pts)} points, {len(pts)-len(todo)} already "
          f"done, {len(todo)} to run on {jobs} workers")
    if not todo:
        print("nothing to do")
        return

    counts, t0, last = {}, time.time(), 0.0
    # workers must ignore Ctrl-C; the parent handles it
    orig = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = Pool(jobs)
    signal.signal(signal.SIGINT, orig)
    try:
        for i, (p, st, sec) in enumerate(
                pool.imap_unordered(_work, ((p, a.out) for p in todo),
                                    chunksize=a.chunk), start=1):
            counts[st] = counts.get(st, 0) + 1
            now = time.time()
            if now - last > 2.0 or i == len(todo):
                last = now
                el = now - t0
                eta = el / i * (len(todo) - i)
                bar = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                print(f"\r{i}/{len(todo)}  {100*i/len(todo):5.1f}%  "
                      f"elapsed {fmt(el)}  left {fmt(eta)}   {bar}   ",
                      end="", flush=True)
        pool.close()
    except KeyboardInterrupt:
        print("\nstopped by user. Finished points are saved; "
              "run the same command again to continue.")
        pool.terminate()
    finally:
        pool.join()

    print(f"\ndone in {fmt(time.time()-t0)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"\nNext: python verdict.py --results {a.out}")


if __name__ == "__main__":
    main()
