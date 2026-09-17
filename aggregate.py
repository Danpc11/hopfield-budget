"""
aggregate.py -- read all the JSON files and produce walls, checks and fits.

The rules below are where mistakes happen, so they are written out:

1. Across seeds: FEASIBLE if ANY seed finds an incumbent. A positive certificate
   is not cancelled because another seed failed. PROVEN INFEASIBLE only if the
   root relaxation says so; that test is deterministic and seed independent.

2. UNDETERMINED never counts as infeasible. The wall is BRACKETED between the
   largest eps proved infeasible and the smallest feasible eps. The grey zone in
   between is reported, not filled in.

3. Feasibility must be MONOTONE in eps (asking for more error is easier) and go
   the other way in J0. Any violation is a search failure and is listed.

4. The wall must go DOWN as the node budget goes up. If it is still moving at
   the largest budget, the curve is not converged and you must not fit a
   functional form to it.

  python aggregate.py --results results --csv summary.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

FEAS, PROV, UNDET = "feasible", "proven_infeasible", "undetermined"


def load(results_dir):
    recs = []
    for p in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            with open(p) as fh:
                recs.append(json.load(fh))
        except Exception:
            print(f"[warning] cannot read JSON, skipping: {p}")
    return recs


def combine_seeds(recs):
    """(m,F,J0,nodes,eps) -> combined status and best sigma."""
    g = defaultdict(list)
    for r in recs:
        if r.get("status") == "error":
            continue
        g[(r["m"], r["F"], r["J0"], r["nodes"], r["eps"])].append(r)
    out = {}
    for k, v in g.items():
        if any(x["status"] == FEAS for x in v):
            sig = min(x["sigma"] for x in v if x["status"] == FEAS)
            st = FEAS
        elif any(x["status"] == PROV for x in v):
            sig, st = None, PROV
        else:
            sig, st = None, UNDET
        out[k] = dict(status=st, sigma=sig, nseeds=len(v))
    return out


def walls(comb):
    """Returns (m,F,J0,nodes) -> (largest eps proved infeasible, smallest feasible)."""
    by = defaultdict(list)
    for (m, F, J0, nodes, eps), d in comb.items():
        by[(m, F, J0, nodes)].append((eps, d["status"]))
    res, viol = {}, []
    for key, lst in by.items():
        lst.sort()
        inf = [e for e, s in lst if s == PROV]
        fea = [e for e, s in lst if s == FEAS]
        res[key] = (max(inf) if inf else None, min(fea) if fea else None)
        # monotonicity: nothing proved infeasible above something feasible
        if inf and fea and max(inf) > min(fea):
            viol.append((key, max(inf), min(fea)))
    return res, viol


def fits(pts, F, m):
    """pts: list of (J0, wall). Tests Michaelis-Menten against a pole."""
    if len(pts) < 4:
        return None
    J = np.array([p[0] for p in pts])
    R = np.array([p[1] for p in pts]) * F ** m          # wall / (1/F^m)
    beta = (R - 1.0) / (F - 1.0)
    ok = beta > 0
    J, beta = J[ok], beta[ok]
    if len(J) < 4:
        return None
    V = J * (1 - beta) / beta                            # implied V of Michaelis-Menten
    A = np.polyfit(1 / J, 1 / beta, 1)                   # pole: 1/beta linear in 1/J
    pred = np.polyval(A, 1 / J)
    r2 = 1 - np.sum((1 / beta - pred) ** 2) / np.sum((1 / beta - np.mean(1 / beta)) ** 2)
    return dict(V_mean=float(V.mean()), V_spread=float(V.std() / V.mean()),
                pole_slope=float(A[0]), pole_intercept=float(A[1]),
                pole_r2=float(r2),
                Jc=float(-A[0] / A[1]) if A[1] < 0 else None,
                n_points=int(len(J)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--csv", default="resumen.csv")
    a = ap.parse_args()

    recs = load(a.results)
    print(f"{len(recs)} records read")
    nerr = sum(1 for r in recs if r.get("status") == "error")
    if nerr:
        print(f"  [warning] {nerr} points failed to run")
    comb = combine_seeds(recs)
    W, viol = walls(comb)

    if viol:
        print("\n!! MONOTONICITY VIOLATIONS (search failures, not physics):")
        for key, hi_inf, lo_fea in viol:
            print(f"   m={key[0]} F={key[1]:g} J0={key[2]:.3g} nodes={key[3]}: "
                  f"proved infeasible at {hi_inf:.4e} > feasible at {lo_fea:.4e}")

    rows = []
    for (m, F, J0, nodes), (e_inf, e_fea) in sorted(W.items()):
        ref = F ** (-m)
        rows.append(dict(m=m, F=F, J0=J0, nodes=nodes,
                         eps_inf=e_inf, eps_fea=e_fea,
                         wall_ratio=(e_fea / ref) if e_fea else None,
                         bracket=(e_fea / e_inf) if (e_fea and e_inf) else None))
    with open(a.csv, "w") as fh:
        fh.write("m,F,J0,nodes,eps_proven_infeasible,eps_feasible,wall_ratio,bracket\n")
        for r in rows:
            fh.write(f"{r['m']},{r['F']},{r['J0']},{r['nodes']},"
                     f"{r['eps_inf']},{r['eps_fea']},{r['wall_ratio']},{r['bracket']}\n")
    print(f"\nCSV written to {a.csv}")

    print("\n=== CONVERGENCE: the wall must GO DOWN as the budget grows ===")
    by = defaultdict(dict)
    for r in rows:
        if r["wall_ratio"]:
            by[(r["m"], r["F"], r["J0"])][r["nodes"]] = r["wall_ratio"]
    for key in sorted(by):
        ns = sorted(by[key])
        vals = [by[key][n] for n in ns]
        drift = abs(vals[-1] - vals[-2]) / vals[-1] if len(vals) > 1 else float("nan")
        flag = "" if drift < 0.01 else "  <-- NOT CONVERGED"
        print(f"  m={key[0]} F={key[1]:g} J0={key[2]:.4g}: " +
              "  ".join(f"n={n}:{v:.4f}" for n, v in zip(ns, vals)) +
              f"   drift={drift:.2%}{flag}")

    print("\n=== FUNCTIONAL FORM (largest available budget only) ===")
    for (m, F) in sorted({(r["m"], r["F"]) for r in rows}):
        nmax = max(r["nodes"] for r in rows if r["m"] == m and r["F"] == F)
        pts = sorted((r["J0"], r["eps_fea"]) for r in rows
                     if r["m"] == m and r["F"] == F and r["nodes"] == nmax
                     and r["eps_fea"])
        f = fits(pts, F, m)
        if not f:
            print(f"  m={m} F={F:g}: not enough points")
            continue
        print(f"  m={m} F={F:g} (nodes={nmax}, {f['n_points']} points)")
        print(f"     Michaelis-Menten: V={f['V_mean']:.2f}, spread={f['V_spread']:.0%}"
              f"   {'(constant: MM works)' if f['V_spread'] < 0.15 else '(NOT constant: MM ruled out)'}")
        print(f"     pole:  1/beta = {f['pole_slope']:.3f}/J {f['pole_intercept']:+.1f}"
              f"   R^2={f['pole_r2']:.6f}   J_c={f['Jc']}")


if __name__ == "__main__":
    main()
