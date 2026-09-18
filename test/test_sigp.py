"""
Tests for the signomial programming route (hopfield/sigp.py).

Slower than the rest (a few seconds each), so they are kept in their own file.

    pytest -q tests/test_sigp.py
"""
import numpy as np
import pytest

from hopfield.core import Net
from hopfield.sigp import build_model, rho_all, _pieces, wall_curve
from hopfield import analytic as an

M2_J0 = [5e-3, 2e-2, 4e-2, 8e-2, 1.3e-1]


def _explicit_net(model, x):
    """The same network built directly, to check the matrix-tree posynomials."""
    F, R, W = model["F"], model["R"], model["W"]
    k = np.exp(np.asarray(x, float))
    edges = []
    for t, (i, j) in enumerate(model["tmpl"]):
        kf, kb = k[2 * t], k[2 * t + 1]
        edges.append((R(i), R(j), kf, kb))
        edges.append((W(i), W(j), kf, kb * (F if i == 0 else 1.0)))
    return Net(model["n"], edges)


@pytest.mark.parametrize("m,F", [(2, 50.0), (3, 20.0)])
def test_matrix_tree_posynomials_give_the_exact_steady_state(m, F):
    """rho_i normalised must equal pi, or everything downstream is wrong."""
    model = build_model(m, F)
    rhos = rho_all(model)
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.uniform(-1.5, 1.5, model["nvar"])
        v = np.array([r.value(x) for r in rhos])
        v /= v.sum()
        assert np.max(np.abs(v - _explicit_net(model, x).pi)) < 1e-12


def test_simplify_merges_terms_without_changing_the_value():
    """The traffic posynomial has one term per (edge, tree) pair and most repeat.
    Merging them is what makes m=3 tractable, so it must be exact."""
    model = build_model(3, 20.0)
    P = _pieces(model, rho_all(model), 1.0)
    rng = np.random.default_rng(1)
    raw = _pieces(model, rho_all(model), 1.0)
    for _ in range(3):
        x = rng.uniform(-1.0, 1.0, model["nvar"])
        assert P["traffic"].value(x) == pytest.approx(raw["traffic"].value(x), rel=1e-12)
    assert len(P["traffic"].c) < 1000          # merged, not thousands of terms


def test_condensation_is_a_lower_bound_tight_at_the_point():
    """The arithmetic-geometric mean bound must satisfy p_hat <= p everywhere,
    with equality at the expansion point. That is what makes each condensed
    subproblem an inner approximation."""
    model = build_model(2, 50.0)
    P = _pieces(model, rho_all(model), 1.0)
    p = P["Z"]
    rng = np.random.default_rng(2)
    x0 = rng.uniform(-1.0, 1.0, model["nvar"])
    c, a = p.condense(x0)
    assert np.exp(c + a @ x0) == pytest.approx(p.value(x0), rel=1e-10)
    for _ in range(20):
        x = x0 + rng.uniform(-1.5, 1.5, model["nvar"])
        assert np.exp(c + a @ x) <= p.value(x) * (1 + 1e-9)


@pytest.mark.slow
def test_signomial_reproduces_the_closed_form_law():
    """Two independent methods, four decimals. This is the strongest check we
    have that the closed-form law is right."""
    res = wall_curve(2, 50.0, M2_J0, starts=4, sweeps=3)
    for r in res:
        assert r["wall_ratio"] is not None
        law = an.wall(r["J0"], 50.0) * 50.0 ** 2
        assert r["wall_ratio"] == pytest.approx(law, rel=2e-3)


@pytest.mark.slow
def test_wall_is_monotone_in_throughput():
    """More throughput cannot lower the error floor. A decreasing curve means
    the continuation only swept one way and the early points are stuck."""
    res = wall_curve(3, 20.0, [3e-3, 1e-2, 2e-2], starts=3, sweeps=2, iters=30)
    w = [r["wall_ratio"] for r in res]
    assert all(v is not None for v in w)
    assert all(w[i + 1] >= w[i] * 0.999 for i in range(len(w) - 1))
