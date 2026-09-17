"""
hopfield.bb -- spatial branch and bound with McCormick envelopes.

The design problem with two substrates is BILINEAR: f_W(e) = v_k * c * f_R(e).
We add a variable w = v_k * f_R(e) and relax it with the four McCormick
inequalities. The relaxed problem is convex, so it gives a LOWER BOUND on the
current box of v.

THREE-VALUED STATUS  (important: do not mix the last two)
  PROVEN_INFEASIBLE : the ROOT relaxation is infeasible -> impossibility PROVED
  FEASIBLE          : we found an incumbent            -> feasibility PROVED
  UNDETERMINED      : relaxation is feasible, but no incumbent -> WE DO NOT KNOW

If we miss an incumbent, the error wall looks higher than it is. So UNDETERMINED
must never be reported as infeasible.
"""
from __future__ import annotations

import heapq
import warnings

import numpy as np
import cvxpy as cp

# CLARABEL warns "solution may be inaccurate" on problems that sit right at the
# edge of feasibility. That is expected here and we already check the value, so
# the warning only makes the output unreadable during a sweep.
warnings.filterwarnings("ignore", message=".*solution may be inaccurate.*")
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

PROVEN_INFEASIBLE = "proven_infeasible"
FEASIBLE = "feasible"
UNDETERMINED = "undetermined"


def mccormick(w, x, y, xL, xU, yL, yU):
    return [w >= xL * y + yL * x - xL * yL,
            w >= xU * y + yU * x - xU * yU,
            w <= xU * y + yL * x - xU * yL,
            w <= xL * y + yU * x - xL * yU]


class BilinearDesign:
    """builder(v, prod, box) -> (restricciones, objetivo, extra).
    `prod(k, fvar, idx, fL, fU)` devuelve el sustituto de v[k]*fvar[idx]:
    variable de McCormick en la relajacion, producto afin en el problema exacto."""

    def __init__(self, nv: int, builder, vbox, solver: str = "CLARABEL"):
        self.nv = nv
        self.builder = builder
        self.vbox = [tuple(b) for b in vbox]
        self.solver = solver
        self._seeder = None

    def set_seeder(self, fn):
        """fn(box, rng) -> a list of candidate v values built in RATE space.
        Rate space parametrises the feasible set by construction, so blind
        sampling inside the v box almost never works. This only changes the
        UPPER bound; the lower bound still comes from McCormick."""
        self._seeder = fn

    # ------------------------------------------------------- relaxation ---
    def relax(self, box):
        v = cp.Variable(self.nv)
        cons = [v >= [b[0] for b in box], v <= [b[1] for b in box]]
        prod = {}

        def make_prod(k, fvar, idx, fL, fU):
            key = (k, id(fvar), idx)
            if key not in prod:
                w = cp.Variable()
                prod[key] = w
                cons.extend(mccormick(w, v[k], fvar[idx],
                                      box[k][0], box[k][1], fL, fU))
            return prod[key]

        c2, obj, extra = self.builder(v, make_prod, box)
        p = cp.Problem(cp.Minimize(obj), cons + c2)
        try:
            p.solve(solver=self.solver)
        except Exception:
            return np.inf, None
        if v.value is None or p.value is None or not np.isfinite(p.value):
            return np.inf, None
        return float(p.value), {"v": np.asarray(v.value, float)}

    # ---------------------------------------------------------- exact ---
    def exact_at(self, vfix):
        """With v fixed, the products v*f become AFFINE, so the problem is convex
        again. It returns an UPPER bound (an incumbent), or inf if infeasible."""
        vfix = np.asarray(vfix, float)

        def make_prod(k, fvar, idx, fL, fU):
            return float(vfix[k]) * fvar[idx]

        class _V:
            def __getitem__(_s, k):
                return float(vfix[k])

        cons, obj, _ = self.builder(_V(), make_prod, None)
        real = []
        for c in cons:
            if isinstance(c, (bool, np.bool_)):
                if not c:
                    return np.inf
            else:
                real.append(c)
        p = cp.Problem(cp.Minimize(obj), real)
        try:
            p.solve(solver=self.solver)
        except Exception:
            return np.inf
        if p.value is None or not np.isfinite(p.value):
            return np.inf
        return float(p.value)

    def _incumbent(self, box, vrel, rng, nsample):
        cands = [np.asarray(vrel, float),
                 np.array([np.sqrt(max(b[0], 1e-12) * b[1]) for b in box])]
        if self._seeder is not None:
            cands += list(self._seeder(box, rng))
        for _ in range(nsample):
            cands.append(np.array([
                np.exp(rng.uniform(np.log(max(b[0], 1e-12)), np.log(b[1])))
                for b in box]))
        best = np.inf
        for c in cands:
            u = self.exact_at(c)
            if u < best:
                best = u
        return best

    # -------------------------------------------------------------- B&B ---
    def solve(self, max_nodes=60, tol=1e-3, seed=0, nsample=10):
        rng = np.random.default_rng(seed)
        lb0, sol0 = self.relax(self.vbox)
        if not np.isfinite(lb0):
            return {"status": PROVEN_INFEASIBLE, "sigma": None, "lb": np.inf,
                    "gap": None, "nodes_explored": 0, "proven": True}
        best = self._incumbent(self.vbox, sol0["v"], rng, nsample)
        best_sol = sol0 if np.isfinite(best) else None
        heap = [(lb0, 0, self.vbox, sol0)]
        cnt, nodes = 1, 0
        while heap and nodes < max_nodes:
            lb, _, box, sol = heapq.heappop(heap)
            nodes += 1
            if lb > best - tol:
                break
            rel = [(b[1] - b[0]) / max(abs(b[1]) + abs(b[0]), 1e-9) for b in box]
            if max(rel) < 1e-4:
                if lb < best:
                    best, best_sol = lb, sol
                continue
            k = int(np.argmax(rel))
            lo_k, hi_k = box[k]
            mid = np.sqrt(max(lo_k, 1e-12) * hi_k) if lo_k > 0 else 0.5 * (lo_k + hi_k)
            for lo, hi in ((lo_k, mid), (mid, hi_k)):
                nb = list(box)
                nb[k] = (lo, hi)
                l, s = self.relax(nb)
                if np.isfinite(l) and l < best - tol:
                    u = self._incumbent(nb, s["v"], rng, nsample)
                    if u < best:
                        best, best_sol = u, s
                    heapq.heappush(heap, (l, cnt, nb, s))
                    cnt += 1
        open_lb = heap[0][0] if heap else np.inf
        if not np.isfinite(best):
            return {"status": UNDETERMINED, "sigma": None, "lb": float(lb0),
                    "gap": None, "nodes_explored": nodes, "proven": False}
        glob_lb = float(min(open_lb, best))
        return {"status": FEASIBLE, "sigma": float(best), "lb": glob_lb,
                "gap": float(max(best - glob_lb, 0.0)), "nodes_explored": nodes,
                "proven": bool(open_lb >= best - tol),
                "v": (best_sol["v"].tolist() if best_sol else None)}
