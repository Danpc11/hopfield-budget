"""
hopfield.core -- exact and convex tools.

Net         : a Markov chain. It gives the steady state, the entropy production
              rate (EPR), the edge currents, and the EXACT current covariance
              (computed with the reduced Drazin inverse, not finite differences).
sigma_flux  : entropy production written as a CONVEX function of the one-way
              fluxes (DCP form, using relative entropy). It is the same as the
              perspective a*phi(j/a) with phi(s) = 2 s arcsinh(s), and the same
              as the cosh-type dissipation potential of Mielke, Peletier and
              Renger.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import null_space

try:
    import cvxpy as cp
except ImportError:  # the analysis code does not need cvxpy
    cp = None


# --------------------------------------------------------------------- Net ---
class Net:
    """Edges are (i, j, k_ij, k_ji): the rate i->j and the rate j->i."""

    def __init__(self, n: int, edges):
        self.n = n
        self.edges = list(edges)
        L = np.zeros((n, n))
        for i, j, kf, kb in self.edges:
            L[j, i] += kf
            L[i, j] += kb
        np.fill_diagonal(L, 0.0)
        np.fill_diagonal(L, -L.sum(axis=0))
        self.L = L
        v = null_space(L)
        if v.shape[1] == 0:
            raise ValueError("generator has no kernel: the network is not irreducible")
        p = np.abs(v[:, 0])
        self.pi = p / p.sum()

    # --- exact thermodynamics ---
    def epr(self) -> float:
        p, s = self.pi, 0.0
        for i, j, kf, kb in self.edges:
            a, b = kf * p[i], kb * p[j]
            if a > 0 and b > 0:
                s += (a - b) * np.log(a / b)
        return float(s)

    def mean_currents(self) -> np.ndarray:
        p = self.pi
        return np.array([kf * p[i] - kb * p[j] for i, j, kf, kb in self.edges])

    def _L1(self, e: int) -> np.ndarray:
        i, j, kf, kb = self.edges[e]
        M = np.zeros((self.n, self.n))
        M[j, i] = kf
        M[i, j] = -kb
        return M

    def covariance(self) -> np.ndarray:
        """EXACT long-time covariance of the edge currents.
        C_ab = 1^T L2_ab pi + 1^T L1_a x_b + 1^T L1_b x_a,
        where  L x_c = -(L1_c pi - J_c pi)  and  1^T x_c = 0.
        The rank of C is m-n+1: the currents live in the cycle space."""
        n, m = self.n, len(self.edges)
        p, J = self.pi, self.mean_currents()
        A = np.vstack([self.L, np.ones(n)])
        X = np.zeros((m, n))
        for c in range(m):
            rhs = -(self._L1(c) @ p - J[c] * p)
            X[c] = np.linalg.lstsq(A, np.append(rhs, 0.0), rcond=None)[0]
        ones = np.ones(n)
        C = np.zeros((m, m))
        for a in range(m):
            ia, ja, kfa, kba = self.edges[a]
            for b in range(a, m):
                t2 = (kfa * p[ia] + kba * p[ja]) if a == b else 0.0
                val = t2 + ones @ self._L1(a) @ X[b] + ones @ self._L1(b) @ X[a]
                C[a, b] = C[b, a] = val
        return C


# --------------------------------------------------------- convex objective ---
def sigma_flux(fp, fm):
    """sigma = sum_e (f+ - f-) ln(f+/f-), written in DCP form.
    It is jointly convex in (f+, f-): it is a sum of relative entropies."""
    if cp is None:
        raise ImportError("cvxpy is required")
    return cp.sum(cp.rel_entr(fp, fm) + cp.rel_entr(fm, fp))


def sigma_numeric(fp, fm) -> float:
    fp, fm = np.asarray(fp, float), np.asarray(fm, float)
    ok = (fp > 0) & (fm > 0)
    return float(np.sum((fp[ok] - fm[ok]) * np.log(fp[ok] / fm[ok])))


def perspective(j, a) -> np.ndarray:
    """sigma = a * phi(j/a) with phi(s) = 2 s arcsinh(s). This is the same value
    as sigma_numeric when a = 2 sqrt(f+ f-), the geometric traffic."""
    j, a = np.asarray(j, float), np.asarray(a, float)
    out = np.zeros_like(j)
    m = a > 0
    s = np.zeros_like(j)
    s[m] = j[m] / a[m]
    out[m] = a[m] * 2.0 * s[m] * np.arcsinh(s[m])
    return out
