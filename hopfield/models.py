"""
hopfield.models -- an m-stage proofreading network, with m and F as parameters.

States: 0 = free enzyme (SHARED by both substrates),
        1..m    = complexes with the right substrate,
        m+1..2m = complexes with the wrong substrate.
The only physical asymmetry: the wrong substrate leaves the enzyme F times
faster. The product is formed in state m.

Outer variables of the branch and bound: v_1..v_m = pi_W(k)/pi_R(k), with
v_0 = 1 because state 0 is shared. The error is exactly v_m, so the functional
constraint is LINEAR in v.
"""
from __future__ import annotations

import itertools

import numpy as np
import cvxpy as cp

from .core import Net


def topology(m: int):
    tmpl = list(itertools.combinations(range(m + 1), 2))
    n = 2 * m + 1
    R = lambda k: 0 if k == 0 else k
    W = lambda k: 0 if k == 0 else m + k
    edges = []
    for (i, j) in tmpl:
        edges.append((R(i), R(j)))
        edges.append((W(i), W(j)))
    inc = np.zeros((n, len(edges)))
    for e, (u, w) in enumerate(edges):
        inc[u, e] -= 1
        inc[w, e] += 1
    key = (m - 1, m) if (m - 1, m) in tmpl else (0, m)
    return tmpl, n, inc, 2 * tmpl.index(key), R, W


def proofreading(m: int, F: float, eps: float, J0: float,
                 tau: float = 1e-8, Atot: float = 1.0):
    """min sigma  subject to  error <= eps, productive current >= J0,
    total traffic <= Atot. Returns the builder used by BilinearDesign."""
    tmpl, n, inc, prod_edge, _, _ = topology(m)
    me = 2 * len(tmpl)

    def builder(v, prod, box):
        def vv(k, fvar, idx, fL, fU):
            return fvar[idx] if k == 0 else prod(k - 1, fvar, idx, fL, fU)

        fp = cp.Variable(me, nonneg=True)
        fm = cp.Variable(me, nonneg=True)
        cons = [inc @ (fp - fm) == 0,
                cp.sum(fp + fm) <= Atot,
                (fp - fm)[prod_edge] >= J0,
                fp >= tau, fm >= tau]
        for t, (i, j) in enumerate(tmpl):
            eR, eW = 2 * t, 2 * t + 1
            Fd = F if i == 0 else 1.0
            cons += [fp[eW] == vv(i, fp, eR, 0.0, Atot),
                     fm[eW] == Fd * vv(j, fm, eR, 0.0, Atot)]
        cons += [v[m - 1] <= eps]                       # the error is v_m
        obj = cp.sum(cp.rel_entr(fp, fm) + cp.rel_entr(fm, fp))
        return cons, obj, None

    return builder, m


def _architecture_edges(m, F, r, w, tiny=1e-9):
    """Rates for the optimal architecture: a chain 0->1->...->m, no direct
    binding to checked states, no back-stepping, and a rejection edge k->0 from
    every bound state. r[k] is the rejection ratio at state k+1, w[k] the
    forward rate out of it."""
    tmpl, n, _, _, R, W = topology(m)
    edges = []
    for (i, j) in tmpl:
        if i == 0 and j == 1:
            kf, kb = w[0], r[0]
        elif i == 0:
            kf, kb = tiny, r[j - 1]          # no direct binding; rejection k->0
        elif j == i + 1:
            kf, kb = w[i], tiny              # forward step, no back-step
        else:
            kf, kb = tiny, tiny
        Fd = F if i == 0 else 1.0
        edges.append((R(i), R(j), kf, kb))
        edges.append((W(i), W(j), kf, kb * Fd))
    return edges, n, R, W


def _random_edges(m, F, rng, lo, hi, drive_max):
    """Random rates with a random cycle affinity injected."""
    tmpl, n, _, _, R, W = topology(m)
    kf = np.exp(rng.uniform(lo, hi, len(tmpl)))
    kb = np.exp(rng.uniform(lo, hi, len(tmpl)))
    internal = [t for t, (i, j) in enumerate(tmpl) if i > 0]
    if internal:
        d = rng.uniform(0.0, drive_max) / (2.0 * len(internal))
        for t in internal:
            kf[t] *= np.exp(d)
            kb[t] *= np.exp(-d)
    edges = []
    for t, (i, j) in enumerate(tmpl):
        Fd = F if i == 0 else 1.0
        edges.append((R(i), R(j), kf[t], kb[t]))
        edges.append((W(i), W(j), kf[t], kb[t] * Fd))
    return edges, n, R, W


def rate_space_seeder(m: int, F: float, eps_target=None, n_draw: int = 300,
                      lo: float = -4.0, hi: float = 4.0, drive_max=None,
                      keep: int = 40, arch_frac: float = 0.7,
                      r_lo: float = -1.0, r_hi: float = 5.0):
    """Candidate v values built in RATE space.

    Random rates alone do NOT work. Whatever driving you inject, the resulting
    v_m stays around 1/F while the wall is at 1/F^m. The candidates fall inside
    the box but break v_m <= eps, the branch and bound finds no incumbent, and
    the symptom is feasibility that is not monotone in eps, which is physically
    impossible. At m=3 this made every point undetermined.

    The fix is to build most seeds from the OPTIMAL ARCHITECTURE instead of
    sampling blindly: a chain with a rejection edge at every checkpoint, with the
    rejection ratios r_k as the free parameters. That family does reach 1/F^m.
    Measured at m=3, F=50: r ~ 1e2 gives v_m F^3 = 1.045, so the wall is
    reachable, while random sampling never got below about 1/F.

    A fraction (1 - arch_frac) still comes from random driven rates, because the
    architecture family is a low-dimensional slice and the true optimum can sit
    off it, in particular at high throughput where the architecture changes."""
    if drive_max is None:
        drive_max = 3.0 * m * np.log(F)
    n_arch = int(round(n_draw * arch_frac))

    def seeder(box, rng):
        scored = []
        for k in range(n_draw):
            if k < n_arch:
                r = 10.0 ** rng.uniform(r_lo, r_hi, m)
                if rng.random() < 0.5:            # half with a common ratio
                    r[:] = r[0]
                w = np.exp(rng.uniform(-1.0, 1.0, max(m, 1)))
                edges, n, R, W = _architecture_edges(m, F, r, w)
            else:
                edges, n, R, W = _random_edges(m, F, rng, lo, hi, drive_max)
            try:
                net = Net(n, edges)
            except Exception:
                continue
            pi = net.pi
            if np.any(pi <= 0):
                continue
            v = np.array([pi[W(k2)] / pi[R(k2)] for k2 in range(1, m + 1)])
            if not np.all(np.isfinite(v)):
                continue
            if not all(box[q][0] <= v[q] <= box[q][1] for q in range(m)):
                continue
            scored.append((float(v[m - 1]), v))
        if not scored:
            return []
        if eps_target is None:
            return [v for _, v in scored]
        hits = [v for e, v in scored if e <= eps_target]
        if hits:
            return hits[:keep]
        scored.sort(key=lambda z: z[0])        # keep the closest to the target
        return [v for _, v in scored[:keep]]

    return seeder

def default_vbox(m: int):
    """v_k = pi_W/pi_R for bound states. Discrimination means v_k < 1."""
    return [(1e-5, 1.0)] * m
