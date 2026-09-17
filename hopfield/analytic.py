"""
hopfield.analytic -- saturation in closed form.

At a checking stage the wrong substrate leaves F times faster. So if q is the
fraction of flux that is REJECTED at the checkpoint, that stage discriminates by
a factor 1/(1 + q(F-1)). With m-1 checking stages:

    eps(q) = (1/F) [1 + q(F-1)]^-(m-1)

  q -> 1 (no throughput asked)   : eps -> 1/F^m   (the Hopfield limit)
  q -> 0 (capacity used up)      : eps -> 1/F     (only one stage is left)

The budget links q to throughput. Since |j_e| <= a_e on every edge, if the futile
cycle and the productive path use L effective edges, then A >= L(R + J0), so

    q_max = 1 - L J0 / A

Write x = L J0 / A. This x is the fraction of the traffic budget that the
throughput itself uses up. For m=2, with beta = (eps F^2 - 1)/(F - 1):

    beta(J0) = x / (F - (F-1) x)


THREE POINTS THAT ARE EASY TO CONFUSE
-------------------------------------
    x = 1                : eps = 1/F.  The checking stage is completely lost.
                           All the budget goes into carrying the current and
                           nothing is left for the futile checking cycles.
                           THIS IS THE PHYSICAL COLLAPSE POINT.

    x = 1 + 1/F          : eps = 1. No discrimination at all is left.

    x = 1 + 1/(F-1)      : the denominator vanishes. This is only the pole of
                           the algebraic extrapolation, already outside the
                           physical range.

They are all separated by terms of order 1/F, because

    F/(F-1) = 1/(1 - 1/F) = 1 + 1/F + 1/F^2 + ...

so they all collapse onto the same place when F is large. Note that 1/F is also
the error a single discrimination stage can reach, so the small parameter of the
expansion is the same number that sets the physics. The approximation is good
exactly in the regime where proofreading makes sense: if F were small, neither
the expansion nor the discrimination would be worth anything.

We therefore report the PHYSICAL point:

    J_c = A / L         (exact; this is where eps reaches 1/F)

and keep the pole separately as J_pole, only to compare with a free fit of
1/beta against 1/J, which extrapolates to the pole and not to J_c. The two differ
by about 1/F: 2% at F = 50, 5% at F = 20.

In both cases J_c goes like 1/L. That is the prediction that links saturation to
topology: more stages means a longer path, so L is larger and J_c is smaller.
This is NOT a fit: L is read from the data and it must come out constant.

Status of the check (m=2, F=50, 60 nodes, 5 points from J0=0.005 to 0.13):
L = 4.318 with 5.4% spread, with no fitted parameter; the residuals of the
formula are between -16% and +8%. The slow drift of L (4.00 -> 4.66) is a second
order effect: the futile cycle and the productive path SHARE the first edges, so
the exact budget is A >= L_s(R+J) + L_r R + L_p J, not a single L.
Still open: check m=3, where the prediction J_c ~ 1/L is decided.
"""
from __future__ import annotations

import numpy as np


def eps_of_q(q, F: float, m: int):
    q = np.asarray(q, float)
    return (1.0 / F) * (1.0 + q * (F - 1.0)) ** (-(m - 1))


def q_max(J0, L: float, A: float = 1.0):
    return np.clip(1.0 - L * np.asarray(J0, float) / A, 0.0, 1.0)


def wall(J0, F: float, m: int, L: float, A: float = 1.0):
    """Predicted error floor. No free parameter once L is fixed."""
    return eps_of_q(q_max(J0, L, A), F, m)


def beta_of_J(J0, F: float, L: float, A: float = 1.0):
    """Only for m=2: beta = x/(F-(F-1)x)."""
    x = L * np.asarray(J0, float) / A
    return x / (F - (F - 1.0) * x)


def Jc(L: float, A: float = 1.0) -> float:
    """PHYSICAL collapse point: the throughput at which the checking stage is
    fully lost, that is x = 1 and eps = 1/F. It does not depend on F."""
    return A / L


def J_pole(F: float, L: float, A: float = 1.0) -> float:
    """Pole of the algebraic form, at x = F/(F-1). It sits about 1/F above J_c
    and is outside the physical range. Use it only to compare with a free fit of
    1/beta against 1/J, which extrapolates to this point and not to J_c."""
    return (A / L) * F / (F - 1.0)


def J_nodiscrimination(F: float, L: float, A: float = 1.0) -> float:
    """Throughput at which eps = 1, that is no discrimination at all (x = 1+1/F)."""
    return (A / L) * (1.0 + 1.0 / F)


def L_from_data(J0, wall_meas, F: float, m: int, A: float = 1.0):
    """Solve for L at each point. If the derivation is right, L is CONSTANT.
    Returns (L per point, mean, relative spread)."""
    J0 = np.asarray(J0, float)
    w = np.asarray(wall_meas, float)
    # invert eps(q): q = [(F eps)^(-1/(m-1)) - 1]/(F-1)
    q = ((F * w) ** (-1.0 / (m - 1)) - 1.0) / (F - 1.0)
    Ls = (1.0 - q) * A / J0
    return Ls, float(np.mean(Ls)), float(np.std(Ls) / np.mean(Ls))


def residuals(J0, wall_meas, F: float, m: int, L: float, A: float = 1.0):
    pred = wall(J0, F, m, L, A)
    w = np.asarray(wall_meas, float)
    return (pred - w) / w


def check(J0, wall_meas, F: float, m: int, A: float = 1.0,
          tol_spread: float = 0.15) -> dict:
    """Full check: is L constant, and does the formula match the measurement?"""
    Ls, Lbar, spread = L_from_data(J0, wall_meas, F, m, A)
    res = residuals(J0, wall_meas, F, m, Lbar, A)
    return dict(L_per_point=Ls.tolist(), L=Lbar, L_spread=spread,
                max_abs_residual=float(np.max(np.abs(res))),
                residuals=res.tolist(),
                Jc=Jc(Lbar, A),                 # physical: eps reaches 1/F
                J_pole=J_pole(F, Lbar, A),      # pole of the algebraic form
                passes=bool(spread < tol_spread))
