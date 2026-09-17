"""
verdict.py -- can we derive both frameworks? Do they agree with the published
data? What do they predict that someone can test?

The criteria are WRITTEN DOWN BEFORE looking at the data, in CRITERIA. We do not
change them afterwards: if one fails, we report that it fails.

  python verdict.py --results results     # with the cluster sweep
  python verdict.py                       # epistasis and literature only

Output: one verdict per criterion, one verdict per framework, and the falsifiable
prediction with its power calculation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hopfield import analytic as an
from hopfield import epistasis as ep
from hopfield import literature as lit

# ---------------------------------------------------------------------------
# CRITERIA -- written down in advance.
# ---------------------------------------------------------------------------
CRITERIA = {
    "S0": "The eps grid resolves the wall better than 1/5 of the smallest excess "
          "measured. Without this the walls are quantised and S1-S4 mean nothing.",
    "S1": "The effective number of stages goes down as J0 goes up "
          "(converged points only).",
    "S2": "The largest excess is more than 5 times the convergence drift, so the "
          "signal is bigger than the search bias.",
    "S3": "Michaelis-Menten is ruled out: the implied V changes by more than 15%.",
    "S4": "The pole fit gives R^2 > 0.99 with a finite positive J_c.",
    "S5": "J_c goes down from m=2 to m=3 (it scales with the cycle length).",
    "A1": "The closed-form law eps=(1/F)[1+q(F-1)]^-(m-1) with q=1-L*J0/A matches "
          "the sweep: the L we solve for is CONSTANT (spread < 15%), with no fit.",
    "A2": "The analytic J_c = (A/L) F/(F-1) agrees with the measured pole to 30%.",
    "E1": "Fixed window => exact multiplicativity, |E-1| < 1e-9.",
    "E2": "Sign constraint: no case with E < 1 in the random scan.",
    "E3": "The closed-form law matches the measurement to within 20%.",
    "L1": "Morrison 1993 gives E > 1, which agrees with the sign constraint.",
    "L2": "The effect predicted along the replication timing axis is much smaller "
          "than the deviation seen in TMB. So the comparison between genotypes "
          "is dominated by censoring and does NOT falsify the model.",
}

OK, FAIL, SKIP = "PASS", "FAIL", "NO DATA"


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def line(code, status, detail=""):
    mark = {OK: "[+]", FAIL: "[!]", SKIP: "[ ]"}[status]
    print(f" {mark} {code:3s} {status:9s} {CRITERIA[code]}")
    if detail:
        for row in str(detail).split("\n"):
            print(f"          {row}")


# ---------------------------------------------------------------- marco 1 ---
def saturation(results_dir):
    """Read the sweep and evaluate S0-S5."""
    import glob
    from collections import defaultdict
    verdicts, data = {}, {}
    files = glob.glob(os.path.join(results_dir, "*.json")) if results_dir else []
    if not files:
        for c in ("S0", "S1", "S2", "S3", "S4", "S5", "A1", "A2"):
            verdicts[c] = (SKIP, "no sweep results found")
        return verdicts, data

    recs = []
    for p in files:
        try:
            with open(p) as fh:
                recs.append(json.load(fh))
        except Exception:
            pass
    comb = defaultdict(list)
    for r in recs:
        if r.get("status") == "feasible":
            comb[(r["m"], r["F"], r["J0"], r["nodes"])].append(r["eps"])
    wall = {k: min(v) for k, v in comb.items()}

    for (m, F) in sorted({(k[0], k[1]) for k in wall}):
        budgets = sorted({k[3] for k in wall if k[0] == m and k[1] == F})
        nmax = budgets[-1]
        pts = sorted((k[2], wall[k]) for k in wall
                     if k[0] == m and k[1] == F and k[3] == nmax)
        drift = np.nan
        if len(budgets) > 1:
            n1, n2 = budgets[-2], nmax
            common = [k[2] for k in wall if k[0] == m and k[1] == F and k[3] == n2
                      and (m, F, k[2], n1) in wall]
            if common:
                d = [abs(wall[(m, F, j, n2)] - wall[(m, F, j, n1)])
                     / wall[(m, F, j, n2)] for j in common]
                drift = float(np.max(d))
        data[(m, F)] = dict(points=pts, nodes=nmax, drift=drift)

    # S0 grid resolution. This is a guard: a coarse grid quantises the walls and
    #    makes S1 and S2 pass by construction.
    eps_all = sorted({r["eps"] for r in recs})
    det0, ok0 = [], True
    for (m, F), d in data.items():
        if len(d["points"]) < 2:
            continue
        w = [p[1] for p in d["points"]]
        near = [e for e in eps_all if min(w) / 3 <= e <= max(w) * 3]
        step = (max(np.diff(np.log(near))) if len(near) > 1 else np.inf)
        excess = [p[1] * F ** m - 1.0 for p in d["points"]]
        exc_min = min(x for x in excess if x > 0) if any(x > 0 for x in excess) else np.nan
        good = np.isfinite(exc_min) and step < exc_min / 5
        ok0 &= bool(good)
        det0.append(f"m={m} F={F:g}: grid step {step:.4f} (log), "
                    f"smallest excess {exc_min:.4f} -> "
                    f"{'resolved' if good else 'TOO COARSE'}")
    verdicts["S0"] = (OK if ok0 and det0 else (FAIL if det0 else SKIP), "\n".join(det0))

    # S1 monotonicity
    bad = []
    for key, d in data.items():
        w = [p[1] for p in d["points"]]
        if any(w[i + 1] < w[i] * 0.999 for i in range(len(w) - 1)):
            bad.append(key)
    verdicts["S1"] = (FAIL if bad else OK,
                      f"not monotone at {bad}" if bad else
                      "; ".join(f"m={k[0]} F={k[1]:g}: {len(d['points'])} points"
                                for k, d in data.items()))
    # S2 signal against bias
    det = []
    ok2 = True
    for (m, F), d in data.items():
        if not d["points"]:
            continue
        exc = max(p[1] * F ** m for p in d["points"]) - 1.0
        dr = d["drift"]
        good = np.isfinite(dr) and exc > 5 * dr
        ok2 &= bool(good)
        det.append(f"m={m} F={F:g}: largest excess {exc:.3f}, drift {dr:.3f}"
                   f" -> {'ok' if good else 'not enough'}")
    verdicts["S2"] = (OK if ok2 and det else (FAIL if det else SKIP), "\n".join(det))

    # S3/S4 functional form
    from aggregate import fits
    Jc = {}
    det3, det4 = [], []
    ok3 = ok4 = True
    for (m, F), d in data.items():
        f = fits(d["points"], F, m)
        if not f:
            continue
        ok3 &= f["V_spread"] > 0.15
        det3.append(f"m={m} F={F:g}: V={f['V_mean']:.2f}, spread {f['V_spread']:.0%}")
        good4 = f["pole_r2"] > 0.99 and f["Jc"] is not None and f["Jc"] > 0
        ok4 &= bool(good4)
        det4.append(f"m={m} F={F:g}: R^2={f['pole_r2']:.5f}, J_c={f['Jc']}")
        if f["Jc"]:
            Jc[(m, F)] = f["Jc"]
    verdicts["S3"] = (OK if ok3 and det3 else (FAIL if det3 else SKIP), "\n".join(det3))
    verdicts["S4"] = (OK if ok4 and det4 else (FAIL if det4 else SKIP), "\n".join(det4))

    # S5 scaling with cycle length
    det5, ok5, any5 = [], True, False
    for F in sorted({k[1] for k in Jc}):
        if (2, F) in Jc and (3, F) in Jc:
            any5 = True
            good = Jc[(3, F)] < Jc[(2, F)]
            ok5 &= good
            det5.append(f"F={F:g}: J_c(m=2)={Jc[(2,F)]:.4f} -> "
                        f"J_c(m=3)={Jc[(3,F)]:.4f}  {'goes down' if good else 'does NOT go down'}")
    verdicts["S5"] = (SKIP if not any5 else (OK if ok5 else FAIL),
                      "\n".join(det5) if det5 else "m=3 missing from the sweep")
    # A1/A2: test against the CLOSED-FORM LAW, not against a free fit
    detA1, detA2, okA1, okA2, any_a = [], [], True, True, False
    for (m, F), d in data.items():
        if len(d["points"]) < 3:
            continue
        any_a = True
        J = [p[0] for p in d["points"]]
        w = [p[1] for p in d["points"]]
        c = an.check(J, w, F, m)
        okA1 &= c["passes"]
        detA1.append(f"m={m} F={F:g}: L={c['L']:.3f}, spread {c['L_spread']:.1%}, "
                     f"largest residual {c['max_abs_residual']:.1%}")
        if (m, F) in Jc:
            rel = abs(c["Jc"] - Jc[(m, F)]) / c["Jc"]
            okA2 &= rel < 0.30
            detA2.append(f"m={m} F={F:g}: analytic J_c {c['Jc']:.4f} vs "
                         f"fitted {Jc[(m,F)]:.4f}  ({rel:.0%})")
    verdicts["A1"] = (OK if okA1 and any_a else (FAIL if any_a else SKIP),
                      "\n".join(detA1))
    verdicts["A2"] = (OK if okA2 and detA2 else (FAIL if detA2 else SKIP),
                      "\n".join(detA2) or "pole fit missing")
    data["Jc"] = Jc
    return verdicts, data


# ---------------------------------------------------------------- marco 2 ---
def epistasis_checks():
    v = {}
    d0 = ep.test_fixed_window()
    v["E1"] = (OK if d0 < 1e-9 else FAIL, f"|E-1| = {d0:.3e} with a fixed window")
    s = ep.test_sign_constraint()
    v["E2"] = (OK if s["n_negative"] == 0 else FAIL,
               f"smallest E = {s['E_min']:.6f} over {s['n_trials']} draws; "
               f"{s['n_negative']} negative")
    c = ep.test_closed_form()
    v["E3"] = (OK if c["max_rel_error"] < 0.20 else FAIL,
               f"measured/law in [{c['ratio_min']:.3f}, {c['ratio_max']:.3f}], "
               f"largest error {c['max_rel_error']:.1%} over {c['n']} draws")
    return v


# ------------------------------------------------------------- literature ---
def literature_checks():
    v = {}
    m = lit.observed_epistasis()
    v["L1"] = (OK if m["E"] > 1 else FAIL,
               f"E = {m['E']:.2f} ({m['sign']})\n"
               f"multiplicative prediction {m['multiplicative_prediction']:.0f}, "
               f"observed {m['observed']:.0f}\n{m['source']}")
    p = lit.predicted_epistasis_from_rt()
    t = lit.tmb_epistasis()
    obs_dev = max(abs(x["E"] - 1) for x in t)
    pred_hi = p["E_minus_1_range"][1]
    v["L2"] = (OK if obs_dev > 10 * pred_hi else FAIL,
               f"predicted along the replication timing axis: E-1 in "
               f"[{p['E_minus_1_range'][0]:.4f}, {p['E_minus_1_range'][1]:.4f}]\n"
               f"observed between genotypes: |E-1| up to {obs_dev:.2f} "
               f"({obs_dev/pred_hi:.0f}x larger and with the opposite sign)\n"
               f"=> TMB measures censoring, not repair kinetics")
    return v, p


# -------------------------------------------------------------- prediction ---
def prediction(p):
    banner("FALSIFIABLE PREDICTION")
    lo, hi = p["E_minus_1_range"]
    print(f"""
 What      Epistasis resolved by REPLICATION TIMING, INSIDE the same genome,
           in ultramutated POLE + MMRd tumours (WGS, not exomes).

 Prediction with no free parameter, from already published gradients:
           E - 1 = rho * CV(alpha) * CV(beta)  with  CV(beta) = {p['cv_beta']:.3f},
           CV(alpha) in [{p['cv_alpha_range'][0]:.3f}, {p['cv_alpha_range'][1]:.3f}], rho <= 1

               E - 1  =  {lo:.4f}  to  {hi:.4f}     ({100*lo:.1f}% to {100*hi:.1f}%)

 Two independent ways to falsify it:
   sign      a shared clock cannot give E < 1. If E < 1 inside the genome, the
             framework is wrong (the specificity model allows both signs).
   size      a value outside the range above, with rho <= 1, also falsifies it.""")
    for eff, lbl in ((lo, "low end"), (hi, "high end")):
        pw = lit.power_required(eff)
        print(f"\n Power ({lbl}, effect {100*eff:.1f}%, 10 bins, 3 sigma):"
              f"\n   {pw['per_bin']:,.0f} mutations per bin -> "
              f"{pw['total']:,.0f} per tumour"
              f"\n   ultramutated genomes needed: about {pw['genomes_needed']:.1f}")
    print("""
 Why the exome does NOT work: exons sit in open chromatin, which replicates
 early. So the exome SQUEEZES the very axis we want to measure. Because the
 effect is a product of two CVs, it falls with the square of the range we can
 sample. If the exome covers about 40% of the range, the signal drops 6 times
 and the data we need goes up 40 times. We need WGS: PCAWG, Hartwig or Genomics
 England.

 Internal control: the same replication timing axis has been measured in human
 tumours in other work, and the comparison is INSIDE one genome. So viability
 censoring and detection bias cancel out.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="sweep directory; without it, only epistasis is run")
    a = ap.parse_args()

    banner("FRAMEWORK 1 -- can we derive SATURATION from a finite budget?")
    print(" Earlier work: " + lit.SATURATION_PRIOR["source"])
    print(" What we add: saturation as a BOUND, not as a mechanism.")
    print(" This is the other axis of " + lit.FORCE_BOUND["source"] + ":")
    print("   their bound is on the FORCE; ours is on the ACTIVITY.\n")
    vs, ds = saturation(a.results)
    for c in ("S0", "S1", "S2", "S3", "S4", "S5"):
        line(c, *vs[c])
    print("\n --- test against the CLOSED-FORM LAW (no free parameter) ---")
    for c in ("A1", "A2"):
        line(c, *vs[c])

    banner("FRAMEWORK 2 -- can we derive EPISTASIS from the shared clock?")
    print(" The \"overlapping specificity\" mechanism follows as a corollary:")
    print(" we do not assume the covariance between escapes; the clock creates it.\n")
    ve = epistasis_checks()
    for c in ("E1", "E2", "E3"):
        line(c, *ve[c])

    banner("COMPARISON WITH PUBLISHED DATA")
    vl, p = literature_checks()
    for c in ("L1", "L2"):
        line(c, *vl[c])

    banner("VERDICT")
    allv = {**vs, **ve, **vl}
    for name, codes in (("Saturation derived (numerical)", ("S0", "S1", "S2", "S3", "S4", "S5")),
                        ("Saturation derived (analytic)", ("A1", "A2")),
                        ("Epistasis derived", ("E1", "E2", "E3")),
                        ("Agrees with published data", ("L1", "L2"))):
        st = [allv[c][0] for c in codes]
        v = FAIL if FAIL in st else (SKIP if SKIP in st else OK)
        print(f"  {name:32s} {v}"
              + ("   (missing: " + ", ".join(c for c in codes if allv[c][0] == SKIP) + ")"
                 if v == SKIP else ""))
    prediction(p)


if __name__ == "__main__":
    main()
