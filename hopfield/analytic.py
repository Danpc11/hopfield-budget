"""
hopfield.analytic -- closed-form solution of the m=2 saturation curve.

Derivation (matrix-tree theorem on the triangle, with the optimal architecture
a_3 = 0, no direct binding to the checked state, and s = b_2/b_3 = 0, no
back-stepping, since back-stepping only costs traffic and lowers discrimination).

States 0 (free enzyme, shared), 1, 2 per branch. Rates on the right branch:
a_1, b_1 on edge (0,1); a_2, b_2 on (1,2); a_3, b_3 on (0,2). On the wrong branch
the dissociations into node 0 are multiplied by F. Let r = b_1/a_2, the rejection
ratio at the first checkpoint. Spanning-tree sums give

    eps(r) = (1 + 1/r) / (F (F + 1/r))
    v_1(r) = (1 + r) / (1 + F r)

  r -> inf : eps -> 1/F^2   (Hopfield limit, both stages used)
  r -> 0   : eps -> 1/F     (only the binding step discriminates)

The current J is the same on the three edges of the cycle, so pi_2 b_3 = J and
pi_1 a_2 = J. The traffic f+ + f- on each edge follows:

    right branch    (0,2): J          (1,2): J          (0,1): J(1 + 2r)
    wrong branch    (0,2): eps F J    (1,2): v_1 J      (0,1): J(1 + r) + J r F v_1

The wrong branch is not negligible. Its binding flux pi_0 a_1 is identical to the
right branch (same rate, same source node) and its dissociation flux is
v_1 pi_1 F b_1, which for large r equals the right-branch one because v_1 -> 1/F.
The enzyme spends as much turnover rejecting wrong substrates as processing right
ones, which is what proofreading is.

    L(r) = A/J = (3 + 2r) + (1 + r) + r F v_1(r) + v_1(r) + eps(r) F

Since the budget is saturated, L = A/J0 is fixed by the operating point. Invert
L(r) = A/J0 for r, then eps(r) gives the error floor. NO free parameter.

Collapse: as r -> 0 every edge of both branches carries exactly J, so

    L_min = 2 (m + 1) = 6        =>     J_c = A / 6     (for m = 2)

exact, and independent of F. In general the cycle has m+1 edges and both branches
carry the same traffic at collapse, so L_min = 2(m+1) and

    J_c = A / (2(m+1))           =>     J_c(m=3)/J_c(m=2) = 6/8 = 0.75

That is the prediction linking saturation to topology, now analytic.

CHECK against the m=2, F=50 sweep (5 points, J0 from 0.005 to 0.13):
residuals +0.01%, -0.09%, -0.79%, -1.22%, -0.84%. All within 1.3% with no fitted
parameter. The residuals are systematically NEGATIVE by about 1%, which matches
the search bias measured in the convergence study: a branch and bound that misses
an incumbent reports the wall too high. The law is the true wall; the numerics
sit just above it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def v1(r, F: float):
    """Population ratio of the first bound state, pi_W(1)/pi_R(1)."""
    r = np.asarray(r, float)
    return (1.0 + r) / (1.0 + F * r)


def eps_of_r(r, F: float):
    """Error floor as a function of the rejection ratio r = b_1/a_2."""
    r = np.asarray(r, float)
    return (1.0 + 1.0 / r) / (F * (F + 1.0 / r))


def L_of_r(r, F: float):
    """Traffic per unit current, both branches. L = A/J."""
    r = np.asarray(r, float)
    w = v1(r, F)
    return (3.0 + 2.0 * r) + (1.0 + r) + r * F * w + w + eps_of_r(r, F) * F


def L_min(m: int = 2) -> float:
    """Traffic per unit current at collapse: m+1 edges in the cycle, two branches."""
    return 2.0 * (m + 1)


def Jc(A: float = 1.0, m: int = 2) -> float:
    """Collapse throughput. Exact, and independent of F."""
    return A / L_min(m)


def r_of_J(J0: float, F: float, A: float = 1.0):
    """Invert L(r) = A/J0. Returns None if J0 is above the collapse point."""
    target = A / J0
    if target <= L_min():
        return None
    try:
        return brentq(lambda x: L_of_r(x, F) - target, 1e-9, 1e9)
    except ValueError:
        return None


def wall(J0, F: float, A: float = 1.0):
    """Predicted error floor at throughput J0. No free parameter."""
    out = []
    for j in np.atleast_1d(np.asarray(J0, float)):
        r = r_of_J(float(j), F, A)
        out.append(np.nan if r is None else float(eps_of_r(r, F)))
    out = np.array(out)
    return out if np.ndim(J0) else float(out[0])


def residuals(J0, wall_meas, F: float, A: float = 1.0):
    pred = np.atleast_1d(wall(J0, F, A))
    w = np.asarray(wall_meas, float)
    return (pred - w) / w


def check(J0, wall_meas, F: float, m: int = 2, A: float = 1.0,
          tol: float = 0.05) -> dict:
    """Full check. The law has no free parameter, so this is a direct test."""
    res = residuals(J0, wall_meas, F, A)
    ok = np.isfinite(res)
    return dict(residuals=res.tolist(),
                max_abs_residual=float(np.max(np.abs(res[ok]))),
                mean_residual=float(np.mean(res[ok])),
                Jc=Jc(A, m), L_min=L_min(m),
                n_points=int(ok.sum()),
                passes=bool(np.max(np.abs(res[ok])) < tol))
