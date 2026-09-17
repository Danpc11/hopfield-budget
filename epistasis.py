"""
hopfield.epistasis -- epistasis as the SECOND moment of the shared budget.

Idea: polymerase proofreading and mismatch repair race against the SAME clock,
the time window of the replication fork. The escape probabilities are

    alpha = gamma/(gamma + r1)      (escapes proofreading)
    beta  = gamma/(gamma + r2)      (escapes mismatch repair)

where gamma is the rate at which the window closes. Multiplicative epistasis is

    E = mu_0 mu_AB / (mu_A mu_B) = E[alpha beta] / (E[alpha] E[beta])
      = 1 + Cov(alpha,beta) / (E[alpha] E[beta])

If gamma is FIXED, then E = 1 exactly: multiplicativity is exact.
If gamma VARIES, the shared clock makes the two escapes correlated BY
CONSTRUCTION. We do not need to assume that the two systems recognise the same
error classes ("overlapping specificity"): that follows.

Two things you can test:
  (i)  a closed-form law, E - 1 ~ CV^2(gamma) (1-alpha)(1-beta), with ONE free
       parameter instead of the six of the specificity model;
  (ii) a SIGN CONSTRAINT: a shared clock can only give E > 1, never E < 1. The
       specificity model allows both signs, so this model is more falsifiable.

In statistics this is a SHARED FRAILTY effect, standard in survival analysis.
What is specific here is to identify the replication window as that frailty.
"""
from __future__ import annotations

import numpy as np


def escapes(gamma, r1, r2):
    g = np.asarray(gamma, float)
    return g / (g + r1), g / (g + r2)


def epistasis(gamma, r1, r2, weights=None) -> float:
    a, b = escapes(gamma, r1, r2)
    w = np.ones_like(a) / len(a) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    return float(np.sum(w * a * b) / (np.sum(w * a) * np.sum(w * b)))


def lognormal_window(cv: float, n: int, rng, mean: float = 1.0) -> np.ndarray:
    if cv <= 0:
        return np.full(n, mean)
    s = np.sqrt(np.log(1 + cv ** 2))
    return mean * np.exp(rng.normal(-s * s / 2, s, n))


def law(cv: float, r1: float, r2: float, gbar: float = 1.0) -> float:
    """E - 1 ~ CV^2 * (fraction repaired by proofreading) * (fraction by MMR)."""
    return cv ** 2 * (r1 / (gbar + r1)) * (r2 / (gbar + r2))


def law_from_cvs(rho: float, cv_alpha: float, cv_beta: float) -> float:
    """Measurable form: E - 1 = rho * CV(alpha) * CV(beta), with rho <= 1."""
    return rho * cv_alpha * cv_beta


def cv_from_fold(R: float) -> float:
    """CV of a variable with a monotone gradient of range R = max/min."""
    return (R - 1.0) / ((R + 1.0) * np.sqrt(3.0))


# ---------------------------------------------------------------- checks ---
def test_fixed_window(n=200_000, r1=30.0, r2=10.0) -> float:
    """Fixed window => EXACT multiplicativity. Returns |E - 1|."""
    return abs(epistasis(np.full(n, 1.0), r1, r2) - 1.0)


def test_sign_constraint(n_trials=20_000, n=4000, seed=0) -> dict:
    """Can a shared clock give NEGATIVE epistasis? It should never happen."""
    rng = np.random.default_rng(seed)
    worst, n_neg = np.inf, 0
    for _ in range(n_trials):
        cv = rng.uniform(0.05, 4.0)
        g = lognormal_window(cv, n, rng)
        E = epistasis(g, 10 ** rng.uniform(-2, 3), 10 ** rng.uniform(-2, 3))
        worst = min(worst, E)
        n_neg += int(E < 1.0 - 1e-6)
    return dict(E_min=float(worst), n_negative=n_neg, n_trials=n_trials)


def test_closed_form(n_trials=12, n=400_000, seed=1) -> dict:
    """Does the closed-form law match the measurement when CV is small?"""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_trials):
        cv = rng.uniform(0.05, 0.6)
        r1, r2 = 10 ** rng.uniform(-1, 2), 10 ** rng.uniform(-1, 2)
        g = lognormal_window(cv, n, rng)
        meas = epistasis(g, r1, r2) - 1.0
        pred = law(cv, r1, r2)
        if pred > 0:
            ratios.append(meas / pred)
    r = np.array(ratios)
    return dict(ratio_mean=float(r.mean()), ratio_min=float(r.min()),
                ratio_max=float(r.max()),
                max_rel_error=float(np.max(np.abs(r - 1.0))), n=len(r))


def sweep_cv(cvs, r1=30.0, r2=10.0, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    return [(float(cv), epistasis(lognormal_window(cv, n, rng), r1, r2)) for cv in cvs]
