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

Collapse. L(r) is NOT monotone. Writing u = F r and using eps(r) F = v_1(r),
which follows from the two expressions above, it collapses to

    L(u) = 4 + 3r + (1 + r) (u + 2) / (1 + u)
         = 5 + 1/u + 4u/F + O(1/F)          for u >> 1

whose minimum sits at u* = sqrt(F)/2, that is

    r* = 1 / (2 sqrt(F)),     L_min = 5 + 4/sqrt(F) + O(1/F)

At r = 0 one gets L = 2(m+1) = 6, twice the cycle length, because both branches
then carry the full current on every edge. The optimum is lower than that: a
small rejection ratio suppresses the wrong branch (v_1 < 1) and saves more
turnover than the futile cycle costs. So the machine does NOT switch proofreading
off at collapse; it keeps r* = 1/(2 sqrt(F)).

The collapse throughput is therefore

    J_c = A / L_min ,          L_min = min_r L(r)

which depends on F, weakly, through the 4/sqrt(F) term. For F = 50 this gives
J_c = 0.1818 at A = 1, against 0.182 from an independent free fit of the
numerics. The asymptotic form is accurate to 1.2% at F = 50 and to 0.1% at
F = 10^4; the code uses the numerical minimum, not the asymptotic one.

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


def r_star(F: float) -> float:
    """Rejection ratio that minimises the traffic, asymptotically 1/(2 sqrt(F))."""
    from scipy.optimize import minimize_scalar
    g = np.geomspace(1e-8, 1e2, 3000)
    L = np.array([L_of_r(r, F) for r in g])
    i = int(np.argmin(L))
    lo, hi = np.log(g[max(i - 1, 0)]), np.log(g[min(i + 1, len(g) - 1)])
    res = minimize_scalar(lambda t: L_of_r(np.exp(t), F),
                          bracket=(lo, np.log(g[i]), hi))
    return float(np.exp(res.x))


def L_min(F: float, m: int = 2) -> float:
    """Smallest traffic per unit current. L(r) is not monotone, so this is a
    minimum over r and not the value at r = 0, which is 2(m+1)."""
    return float(L_of_r(r_star(F), F))


def L_min_asymptotic(F: float) -> float:
    """Leading behaviour, 5 + 4/sqrt(F). Accurate to 1.2% at F = 50."""
    return 5.0 + 4.0 / np.sqrt(F)


def Jc(F: float, A: float = 1.0, m: int = 2) -> float:
    """Collapse throughput, A / L_min. Depends on F through the 4/sqrt(F) term."""
    return A / L_min(F, m)


def r_of_J(J0: float, F: float, A: float = 1.0):
    """Invert L(r) = A/J0. Returns None if J0 is above the collapse point."""
    target = A / J0
    rs = r_star(F)
    if target <= L_of_r(rs, F):
        return None                       # beyond the collapse point
    try:
        # L(r) is not monotone: the branch we want is the one above r*, where
        # more rejection buys more accuracy at the price of more traffic.
        return brentq(lambda x: L_of_r(x, F) - target, rs, 1e9)
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
                Jc=Jc(F, A, m), L_min=L_min(F, m), r_star=r_star(F),
                n_points=int(ok.sum()),
                passes=bool(np.max(np.abs(res[ok])) < tol))
