"""
hopfield.runner -- evaluate one grid point and write one JSON file.

This is shared by run_point.py (one point from the command line) and by
run_local.py (many points in a process pool). Keeping it here means the pool
does not have to start a new Python process for every point. Starting Python and
importing cvxpy costs about one second, which would waste hours over a full
sweep.

Writing is atomic: write to a temporary file, then rename. A job that is killed
never leaves a half-written JSON. If the output file already exists, the point is
not recomputed, so you can stop and restart a sweep for free.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import time

import numpy as np


def tag(p: dict) -> str:
    return (f"m{p['m']}_F{p['F']:g}_eps{p['eps']:.6e}_J{p['J0']:.6e}"
            f"_n{p['nodes']}_s{p['seed']}").replace("+", "")


def already_done(p: dict, out: str) -> bool:
    return os.path.exists(os.path.join(out, tag(p) + ".json"))


def run_point(p: dict, out: str = "results", force: bool = False) -> dict:
    """p needs: m, F, eps, J0, nodes, seed. Optional: Atot, tau, nsample,
    ndraw, solver."""
    from .bb import BilinearDesign
    from .models import proofreading, rate_space_seeder, default_vbox

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, tag(p) + ".json")
    if os.path.exists(path) and not force:
        return {"status": "skipped", "path": path}

    q = dict(Atot=1.0, tau=1e-8, nsample=10, ndraw=300, solver="CLARABEL")
    q.update(p)

    builder, nv = proofreading(q["m"], q["F"], q["eps"], q["J0"],
                               tau=q["tau"], Atot=q["Atot"])
    P = BilinearDesign(nv, builder, default_vbox(q["m"]), solver=q["solver"])
    P.set_seeder(rate_space_seeder(q["m"], q["F"], eps_target=q["eps"],
                                   n_draw=q["ndraw"]))

    t0 = time.time()
    try:
        res = P.solve(max_nodes=q["nodes"], seed=q["seed"], nsample=q["nsample"])
        err = None
    except Exception as ex:            # one bad point must not kill the sweep
        res = {"status": "error", "sigma": None, "lb": None, "gap": None,
               "nodes_explored": 0, "proven": False}
        err = repr(ex)

    rec = dict(q)
    rec.update(res)
    rec.update(wall_ref=float(q["F"] ** (-q["m"])),
               eps_over_wall=float(q["eps"] * q["F"] ** q["m"]),
               seconds=round(time.time() - t0, 3),
               host=socket.gethostname(),
               python=platform.python_version(),
               numpy=np.__version__,
               error=err)

    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)
    rec["path"] = path
    return rec
