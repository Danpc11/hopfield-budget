"""
hopfield.sigp -- the error wall by signomial programming.

Why this exists
---------------
The flux formulation is convex, but the shared-rate constraint between the two
substrates is bilinear, and at small v the positive cone becomes empty in the
directions the branch and bound explores. At m=3 that made every point come out
"undetermined", which is a search failure, not physics.

This module removes v entirely. By the matrix-tree theorem each stationary
weight is a POSYNOMIAL in the rates,

    rho_i = sum over spanning trees rooted at i of the product of the rates
            directed towards i

so every quantity we need is a ratio of posynomials in the rate variables:

    error       eps = rho_{W(m)} / rho_{R(m)}        (the normalisation cancels)
    throughput  J   = (rho_i k_ij - rho_j k_ji) / Z,  Z = sum_i rho_i
    traffic     A   = sum_e (rho_i k_ij + rho_j k_ji) / Z

That is a signomial program. The standard solution is condensation
(Duffin, Peterson and Zener 1967; see Chiang's review): replace the posynomial
in the "bad" position by its monomial lower bound from the arithmetic-geometric
mean inequality at the current point, which turns each step into a geometric
program, convex in the log-rates, and iterate.

The monomial approximation of p(x) = sum_k c_k exp(a_k . x) at x0 is

    ln p_hat(x) = ln p(x0) + (sum_k u_k a_k) . (x - x0),
    u_k = c_k exp(a_k . x0) / p(x0)

and p_hat(x) <= p(x) everywhere, with equality at x0. So replacing p by p_hat on
the right of "<=" gives a conservative (inner) approximation: any solution of the
condensed problem is feasible for the original one.

Two things this buys
--------------------
* No bilinearity and no v, so the empty-cone problem disappears.
* The wall comes out of ONE optimisation per (m, F, J0), because we minimise the
  error directly instead of scanning eps and asking a feasibility question at
  each grid point.

Condensation converges to a KKT point, not a global optimum. The result is an
upper bound on the wall, like the branch and bound, but obtained without the
search failure. Use several starting points and report the best.
"""
from __future__ import annotations

import itertools

import numpy as np
import cvxpy as cp

from .models import topology


# ------------------------------------------------------------ posynomials ---
class Posy:
    """A posynomial, as coefficients and an exponent matrix over log-variables.
    p(x) = sum_k c_k exp(E[k] . x)."""

    def __init__(self, coef, expo):
        self.c = np.asarray(coef, float)
        self.E = np.asarray(expo, float).reshape(len(self.c), -1)

    @property
    def nvar(self):
        return self.E.shape[1]

    def __add__(self, other):
        return Posy(np.concatenate([self.c, other.c]),
                    np.vstack([self.E, other.E]))

    def scaled(self, s: float):
        return Posy(self.c * s, self.E)

    def times_monomial(self, idx: int, nvar: int):
        """Multiply by a single variable, i.e. add 1 to one exponent."""
        E = self.E.copy()
        E[:, idx] += 1.0
        return Posy(self.c, E)

    def simplify(self):
        """Merge terms with the same exponent vector. The traffic posynomial has
        one term per (edge, spanning tree) pair, and most of them repeat, so this
        is the difference between a tractable program and an intractable one."""
        key = {}
        for c, e in zip(self.c, self.E):
            k = tuple(np.round(e, 9))
            key[k] = key.get(k, 0.0) + c
        ks = list(key)
        return Posy(np.array([key[k] for k in ks]), np.array(ks, float))

    def value(self, x):
        return float(np.sum(self.c * np.exp(self.E @ np.asarray(x, float))))

    def log_expr(self, xvar):
        """log-sum-exp form: convex in the log-variables."""
        return cp.log_sum_exp(cp.log(self.c) + self.E @ xvar)

    def condense(self, x0):
        """Monomial lower bound at x0 (arithmetic-geometric mean).
        Returns (log_coefficient, exponent vector), so that
        ln p_hat(x) = log_coefficient + exponent . x  and  p_hat <= p."""
        x0 = np.asarray(x0, float)
        t = self.c * np.exp(self.E @ x0)
        p = t.sum()
        u = t / p
        a = u @ self.E
        return float(np.log(p) - a @ x0), a


# ------------------------------------------------- matrix-tree posynomials ---
def spanning_trees(n: int, edges):
    """All spanning trees, as tuples of edge indices."""
    out = []
    for sub in itertools.combinations(range(len(edges)), n - 1):
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        ok = True
        for e in sub:
            i, j = edges[e]
            ri, rj = find(i), find(j)
            if ri == rj:
                ok = False
                break
            parent[ri] = rj
        if ok:
            out.append(sub)
    return out


def rho_posynomials(n: int, edges, var_of):
    """rho_i for every node, as Posy objects over the log-rate variables.

    `edges` are undirected pairs (i, j). `var_of(i, j)` gives the index of the
    log-rate variable for the directed step i -> j."""
    nvar = max(max(var_of(i, j), var_of(j, i)) for i, j in edges) + 1
    trees = spanning_trees(n, edges)
    adj = {}
    for e, (i, j) in enumerate(edges):
        adj.setdefault(i, []).append((j, e))
        adj.setdefault(j, []).append((i, e))

    rhos = []
    for root in range(n):
        rows = []
        for tr in trees:
            tadj = {}
            for e in tr:
                i, j = edges[e]
                tadj.setdefault(i, []).append((j, e))
                tadj.setdefault(j, []).append((i, e))
            # orient every tree edge towards the root
            expo = np.zeros(nvar)
            seen = {root}
            stack = [root]
            while stack:
                u = stack.pop()
                for w, e in tadj.get(u, []):
                    if w in seen:
                        continue
                    seen.add(w)
                    expo[var_of(w, u)] += 1.0     # w -> u, towards the root
                    stack.append(w)
            rows.append(expo)
        rhos.append(Posy(np.ones(len(rows)), np.array(rows)))
    return rhos, nvar, trees


# --------------------------------------------------- the proofreading model ---
def build_model(m: int, F: float):
    """Directed rate variables and the matrix-tree posynomials for the
    m-stage proofreading network. The wrong branch shares every rate; only the
    dissociations into the free enzyme carry the factor F, which is a constant
    multiplier and therefore only changes a coefficient, not a variable."""
    tmpl, n, _, prod_edge, R, W = topology(m)
    edges, meta = [], []
    for (i, j) in tmpl:
        edges.append((R(i), R(j))); meta.append(("R", i, j))
        edges.append((W(i), W(j))); meta.append(("W", i, j))

    # one log-rate variable per directed template step, shared by both branches
    steps = {}
    for t, (i, j) in enumerate(tmpl):
        steps[(i, j)] = 2 * t
        steps[(j, i)] = 2 * t + 1
    nvar = 2 * len(tmpl)

    def var_of(a, b):
        """Directed step a -> b in the full graph, mapped to a template step."""
        for (br, i, j) in [meta[k] for k in range(len(meta))]:
            pass
        ia = 0 if a in (0,) else (a if a <= m else a - m)
        ib = 0 if b in (0,) else (b if b <= m else b - m)
        return steps[(ia, ib)]

    # constant multipliers: F on every dissociation of the wrong branch
    def coef_of(a, b):
        wrong = (a > m) or (b > m)
        into_free = (b == 0)
        return F if (wrong and into_free) else 1.0

    return dict(tmpl=tmpl, n=n, edges=edges, meta=meta, nvar=nvar,
                var_of=var_of, coef_of=coef_of, R=R, W=W, m=m, F=F,
                prod=(m - 1, m) if (m - 1, m) in tmpl else (0, m))


def rho_all(model):
    """rho_i with the F multipliers folded into the coefficients."""
    n, edges = model["n"], model["edges"]
    var_of, coef_of, nvar = model["var_of"], model["coef_of"], model["nvar"]
    trees = spanning_trees(n, edges)
    rhos = []
    for root in range(n):
        rows, cs = [], []
        for tr in trees:
            tadj = {}
            for e in tr:
                i, j = edges[e]
                tadj.setdefault(i, []).append((j, e))
                tadj.setdefault(j, []).append((i, e))
            expo = np.zeros(nvar); c = 1.0
            seen, stack = {root}, [root]
            while stack:
                u = stack.pop()
                for w, e in tadj.get(u, []):
                    if w in seen:
                        continue
                    seen.add(w)
                    expo[var_of(w, u)] += 1.0
                    c *= coef_of(w, u)
                    stack.append(w)
            rows.append(expo); cs.append(c)
        rhos.append(Posy(np.array(cs), np.array(rows)))
    return rhos


# -------------------------------------------------- the wall, by condensation ---
def _pieces(model, rhos, A: float):
    """Posynomials for the objective and the two constraints."""
    nvar, R, W, m = model["nvar"], model["R"], model["W"], model["m"]
    var_of, coef_of = model["var_of"], model["coef_of"]

    Z = rhos[0]
    for r in rhos[1:]:
        Z = Z + r

    i, j = model["prod"]                       # productive step in the right branch
    fwd = rhos[R(i)].times_monomial(var_of(R(i), R(j)), nvar)
    bwd = rhos[R(j)].times_monomial(var_of(R(j), R(i)), nvar)

    traffic = None
    for (a, b) in model["edges"]:
        for (u, w) in ((a, b), (b, a)):
            p = rhos[u].times_monomial(var_of(u, w), nvar).scaled(coef_of(u, w))
            traffic = p if traffic is None else traffic + p

    return dict(Z=Z.simplify(), fwd=fwd.simplify(), bwd=bwd.simplify(),
                traffic=traffic.simplify(),
                num=rhos[W(m)].simplify(), den=rhos[R(m)].simplify())


def wall_sigp(m: int, F: float, J0: float, A: float = 1.0, iters: int = 40,
              lo: float = -9.0, hi: float = 9.0, trust: float = 1.5,
              seed: int = 0, x0=None, solver: str = "CLARABEL", tol: float = 1e-7):
    """Lowest error at throughput J0, by signomial programming with condensation.

    Returns a dict with the error, the log-rates, and the convergence history.
    Each iteration is a geometric program, convex in the log-rates. Condensation
    replaces every posynomial on the right of a "<=" by its monomial lower bound
    at the current point, so each subproblem is an inner approximation and any
    solution it returns is feasible for the original problem."""
    model = build_model(m, F)
    rhos = rho_all(model)
    P = _pieces(model, rhos, A)
    nvar = model["nvar"]

    rng = np.random.default_rng(seed)
    x = np.asarray(x0, float) if x0 is not None else rng.uniform(-2.0, 2.0, nvar)

    hist, best = [], np.inf
    best_x = x.copy()
    tr = trust
    for it in range(iters):
        y = cp.Variable(nvar)
        cd, ad = P["den"].condense(x)                       # ln rho_R(m) >= cd + ad.y
        obj = P["num"].log_expr(y) - (cd + ad @ y)          # convex upper bound on ln eps

        cons = [y >= lo, y <= hi, cp.norm_inf(y - x) <= tr]

        lhs_t = P["Z"].scaled(J0) + P["bwd"]                # J0 Z + bwd <= fwd
        cf, af = P["fwd"].condense(x)
        cons.append(lhs_t.log_expr(y) <= cf + af @ y)

        cz, az = P["Z"].scaled(A).condense(x)               # traffic <= A Z
        cons.append(P["traffic"].log_expr(y) <= cz + az @ y)

        try:
            prob = cp.Problem(cp.Minimize(obj), cons)
            prob.solve(solver=solver)
        except Exception:
            tr *= 0.5
            if tr < 1e-4:
                break
            continue
        if y.value is None or not np.isfinite(prob.value):
            tr *= 0.5
            if tr < 1e-4:
                break
            continue

        xn = np.asarray(y.value, float)
        eps = P["num"].value(xn) / P["den"].value(xn)        # exact, not condensed
        feas = (P["fwd"].value(xn) - P["bwd"].value(xn) >= J0 * P["Z"].value(xn) * (1 - 1e-6)
                and P["traffic"].value(xn) <= A * P["Z"].value(xn) * (1 + 1e-6))
        hist.append(dict(it=it, eps=float(eps), feasible=bool(feas), trust=float(tr)))
        if feas and eps < best:
            best, best_x = float(eps), xn
        if np.max(np.abs(xn - x)) < tol:
            x = xn
            break
        x = xn
        tr = min(trust, tr * 1.2)

    return dict(eps=best if np.isfinite(best) else None, x=best_x.tolist(),
                iters=len(hist), history=hist,
                wall_ratio=(best * F ** m) if np.isfinite(best) else None)


def wall_multistart(m: int, F: float, J0: float, A: float = 1.0,
                    starts: int = 8, **kw) -> dict:
    """Condensation converges to a KKT point, so use several starting points."""
    best = None
    for s in range(starts):
        r = wall_sigp(m, F, J0, A=A, seed=s, **kw)
        if r["eps"] is not None and (best is None or r["eps"] < best["eps"]):
            best = r
    return best or dict(eps=None, wall_ratio=None, iters=0, history=[])


def wall_curve(m: int, F: float, J0_list, A: float = 1.0, starts: int = 6,
               sweeps: int = 3, **kw) -> list:
    """The wall over a list of throughputs, by continuation in J0.

    Condensation is an inner approximation, so it needs a feasible starting
    point. Random starts work at small J0, where almost anything satisfies the
    throughput constraint, but not at large J0. Continuation from the previous
    point gives a feasible start every time.

    The sweep must go BOTH ways. Walking up only, the early points keep whatever
    local optimum the random starts found, while the later ones inherit better
    and better solutions. The symptom is a wall that DECREASES with J0, which is
    impossible: more throughput cannot lower the error floor. Sweeping back down
    carries the good solutions to the early points, and repeating until nothing
    improves removes the artefact."""
    J = sorted(float(j) for j in J0_list)
    best = {j: None for j in J}

    for j in J:                                   # first pass: get anything feasible
        r = wall_multistart(m, F, j, A=A, starts=starts, **kw)
        if r["eps"] is not None:
            best[j] = r

    for sweep in range(sweeps):
        improved = False
        order = J if sweep % 2 == 0 else list(reversed(J))
        warm = None
        for j in order:
            if warm is not None:
                r = wall_sigp(m, F, j, A=A, x0=warm, **kw)
                if r["eps"] is not None and (best[j] is None
                                             or r["eps"] < best[j]["eps"] * (1 - 1e-9)):
                    best[j] = r
                    improved = True
            if best[j] is not None:
                warm = best[j]["x"]
        if not improved:
            break

    return [dict(J0=j, eps=(best[j]["eps"] if best[j] else None),
                 wall_ratio=(best[j]["wall_ratio"] if best[j] else None),
                 iters=(best[j]["iters"] if best[j] else 0)) for j in J]
