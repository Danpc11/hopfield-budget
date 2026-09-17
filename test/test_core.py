"""
Fast tests (under one minute). They cover the exact identities and the checks
that, when they broke during development, produced wrong results.

    pytest -q
"""
import numpy as np
import pytest

from hopfield.core import Net, sigma_numeric, perspective
from hopfield import analytic as an
from hopfield import epistasis as ep
from hopfield import literature as lit


# ------------------------------------------------------------------ core ---
def unicycle(N=4, A=4.0, kb=1.0):
    f = kb * np.exp(A / N)
    return Net(N, [(i, (i + 1) % N, f, kb) for i in range(N)])


def test_epr_equals_current_times_affinity():
    """On a single cycle, sigma = J * A exactly (Schnakenberg)."""
    for A in (0.1, 1.0, 5.0, 20.0):
        net = unicycle(4, A)
        assert abs(net.epr() - net.mean_currents()[0] * A) < 1e-9 * max(net.epr(), 1)


def test_equilibrium_has_zero_epr():
    assert unicycle(4, 0.0).epr() < 1e-12


def test_covariance_rank_equals_cycle_rank():
    """Currents live in the cycle space, so rank(C) = m - n + 1."""
    rng = np.random.default_rng(0)
    n = 5
    edges = [(i, (i + 1) % n, 1.3, 0.7) for i in range(n)]
    edges += [(0, 2, 0.9, 1.1), (1, 4, 0.6, 1.4)]
    C = Net(n, edges).covariance()
    m = len(edges)
    w = np.linalg.eigvalsh(C)
    assert np.linalg.matrix_rank(C, tol=1e-8 * w.max()) == m - n + 1
    assert np.min(w) > -1e-10                      # positive semidefinite


def test_perspective_identity():
    """sigma = a phi(j/a) with a = 2 sqrt(f+ f-), the geometric traffic."""
    rng = np.random.default_rng(1)
    fp, fm = rng.uniform(0.01, 10, 500), rng.uniform(0.01, 10, 500)
    j, a = fp - fm, 2 * np.sqrt(fp * fm)
    assert np.max(np.abs(perspective(j, a) - (fp - fm) * np.log(fp / fm))) < 1e-10


def test_cosh_flux_force_relation():
    """j = a sinh(xi/2): the cosh gradient structure of Mielke, Peletier and Renger."""
    rng = np.random.default_rng(2)
    fp, fm = rng.uniform(0.01, 10, 500), rng.uniform(0.01, 10, 500)
    j, a, xi = fp - fm, 2 * np.sqrt(fp * fm), np.log(fp / fm)
    assert np.max(np.abs(j - a * np.sinh(xi / 2))) < 1e-10


def test_sigma_jointly_convex_in_one_way_fluxes():
    """Joint convexity: this is what makes the single-branch optimum global."""
    rng = np.random.default_rng(3)
    viol = 0
    for _ in range(4000):
        x, y = rng.uniform(0.02, 6, (2, 4)), rng.uniform(0.02, 6, (2, 4))
        t = rng.uniform()
        mid = sigma_numeric(t * x[0] + (1 - t) * y[0], t * x[1] + (1 - t) * y[1])
        ends = t * sigma_numeric(*x) + (1 - t) * sigma_numeric(*y)
        viol += int(mid > ends + 1e-9)
    assert viol == 0


# ------------------------------------------------------------- epistasis ---
def test_fixed_window_gives_exact_multiplicativity():
    assert ep.test_fixed_window(n=50_000) < 1e-9


def test_sign_constraint_never_negative():
    """A shared clock cannot give negative epistasis. This is what makes the model
    more falsifiable than the specificity model, which allows both signs."""
    r = ep.test_sign_constraint(n_trials=800, n=2000, seed=0)
    assert r["n_negative"] == 0
    assert r["E_min"] >= 1.0 - 1e-6


def test_closed_form_law():
    r = ep.test_closed_form(n_trials=6, n=120_000, seed=1)
    assert r["max_rel_error"] < 0.25


def test_epistasis_increases_with_window_heterogeneity():
    vals = [E for _, E in ep.sweep_cv([0.0, 0.5, 1.0, 2.0], n=60_000)]
    assert vals[0] == pytest.approx(1.0, abs=1e-9)
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


# -------------------------------------------------------------- analitico ---
M2_J0 = [5e-3, 2e-2, 4e-2, 8e-2, 1.3e-1]
M2_WALL_RATIO = [1.020, 1.088, 1.205, 1.538, 2.460]      # measured, F=50, 60 nodes


def test_analytic_law_recovers_constant_L():
    """The L we solve for must be CONSTANT. Nothing is fitted."""
    w = [r / 50.0 ** 2 for r in M2_WALL_RATIO]
    c = an.check(M2_J0, w, 50.0, 2)
    assert c["L_spread"] < 0.10
    assert c["max_abs_residual"] < 0.20
    assert 0.1 < c["Jc"] < 0.5


def test_analytic_limits():
    """q->1 gives 1/F^m (Hopfield); q->0 gives 1/F (only one stage left)."""
    for m in (2, 3, 4):
        assert an.eps_of_q(1.0, 50.0, m) == pytest.approx(50.0 ** -m, rel=1e-12)
        assert an.eps_of_q(0.0, 50.0, m) == pytest.approx(1 / 50.0, rel=1e-12)


def test_Jc_scales_inversely_with_L():
    """The prediction that links saturation to topology."""
    assert an.Jc(50.0, 8.0) < an.Jc(50.0, 4.0)
    assert an.Jc(50.0, 4.0) * 4.0 == pytest.approx(an.Jc(50.0, 8.0) * 8.0)


# ------------------------------------------------------------- literatura ---
def test_morrison_epistasis_is_supermultiplicative():
    """This agrees with the sign constraint of the shared clock."""
    assert lit.observed_epistasis()["E"] > 1.0


def test_tmb_deviation_dwarfs_predicted_effect():
    """The comparison between genotypes is dominated by censoring, not kinetics:
    we predict 1-5% and the data show 80%, with the opposite sign."""
    pred_hi = lit.predicted_epistasis_from_rt()["E_minus_1_range"][1]
    obs = max(abs(x["E"] - 1) for x in lit.tmb_epistasis())
    assert obs > 10 * pred_hi


# --------------------------------------------------------------------- bb ---
def test_bb_status_is_three_valued():
    """PROVED infeasible and not-found are DIFFERENT states. Mixing them makes the
    error wall look higher and invents transitions that do not exist."""
    from hopfield.bb import BilinearDesign, FEASIBLE, PROVEN_INFEASIBLE, UNDETERMINED
    from hopfield.models import proofreading, rate_space_seeder, default_vbox
    F, m, J0 = 50.0, 2, 1e-2
    seen = set()
    for ratio in (20.0, 1.0, 0.2):
        b, nv = proofreading(m, F, ratio * F ** -m, J0)
        P = BilinearDesign(nv, b, default_vbox(m))
        P.set_seeder(rate_space_seeder(m, F, eps_target=ratio * F ** -m, n_draw=120))
        seen.add(P.solve(max_nodes=12, seed=0)["status"])
    assert seen <= {FEASIBLE, PROVEN_INFEASIBLE, UNDETERMINED}
    assert FEASIBLE in seen                       # the loose requirement must be solvable
