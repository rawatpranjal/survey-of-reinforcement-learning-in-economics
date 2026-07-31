# Oracle tests for the Engine running example (sims/engine.py).
# Every published number is pinned here in exact rational arithmetic, and the float
# solvers are held to the rational answers. Run: python3 -m pytest tests/test_engine.py
# or python3 tests/test_engine.py.

import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sims.engine import (
    GAMMA,
    HIGH,
    KEEP,
    LOW,
    REPLACE,
    build_mdp,
    build_mdp_confounded,
    build_mdp_grid,
    build_mdp_grid_frac,
    discounted_occupancy,
    exact_value,
    exact_value_frac,
    fisher_matrix,
    natural_gradient,
    occupancy_frac,
    policy_from_logits,
    policy_gradient,
    policy_matrices,
    policy_performance,
    solve_ev,
    solve_optimal,
    solve_optimal_frac,
    stationary_distribution,
    stationary_frac,
    stochastic_policy_matrices,
)

# The logging policy the appendix settled on (keep with probability 0.1).
BETA = Fraction(1, 10)


def frac_mdp():
    return build_mdp_grid_frac(2)


def test_exact_optimal_values():
    P, r = frac_mdp()
    V, policy, Q = solve_optimal_frac(P, r)
    assert V == [Fraction(155, 29), Fraction(125, 29)]
    assert policy == [KEEP, REPLACE]
    assert Q[LOW][KEEP] == Fraction(155, 29)
    assert Q[LOW][REPLACE] == Fraction(125, 29)
    assert Q[HIGH][KEEP] == Fraction(1183, 290)
    assert Q[HIGH][REPLACE] == Fraction(125, 29)


def test_action_gaps():
    P, r = frac_mdp()
    _, _, Q = solve_optimal_frac(P, r)
    assert Q[LOW][KEEP] - Q[LOW][REPLACE] == Fraction(30, 29)
    assert Q[HIGH][REPLACE] - Q[HIGH][KEEP] == Fraction(67, 290)


def test_published_roundings():
    P, r = frac_mdp()
    V, _, Q = solve_optimal_frac(P, r)
    assert round(float(V[LOW]), 4) == 5.3448
    assert round(float(V[HIGH]), 4) == 4.3103
    assert round(float(Q[HIGH][KEEP]), 4) == 4.0793
    assert round(float(Q[LOW][KEEP] - Q[LOW][REPLACE]), 4) == 1.0345
    assert round(float(Q[HIGH][REPLACE] - Q[HIGH][KEEP]), 4) == 0.231


def test_float_solver_agrees_with_exact():
    P, r = build_mdp()
    Pf, rf = frac_mdp()
    V, greedy, q = solve_optimal(P, r, GAMMA)
    Vf, pf, Qf = solve_optimal_frac(Pf, rf)
    assert list(greedy) == pf
    assert np.max(np.abs(V - np.array([float(x) for x in Vf]))) < 1e-10
    assert np.max(np.abs(q - np.array([[float(x) for x in row] for row in Qf]))) < 1e-10
    # every deterministic policy's evaluation agrees too
    for policy in ([KEEP, KEEP], [KEEP, REPLACE], [REPLACE, KEEP], [REPLACE, REPLACE]):
        P_pi, r_pi = policy_matrices(P, r, policy)
        V_float = exact_value(P_pi, r_pi, GAMMA)
        V_exact = exact_value_frac(Pf, rf, list(policy))
        assert np.max(np.abs(V_float - np.array([float(x) for x in V_exact]))) < 1e-12


def test_stationary_distribution():
    Pf, rf = frac_mdp()
    P_pi = [Pf[LOW][KEEP], Pf[HIGH][REPLACE]]
    d = stationary_frac(P_pi)
    assert d == [Fraction(2, 3), Fraction(1, 3)]
    P, r = build_mdp()
    P_pi_f, _ = policy_matrices(P, r, [KEEP, REPLACE])
    d_float = stationary_distribution(P_pi_f)
    assert np.max(np.abs(d_float - np.array([2 / 3, 1 / 3]))) < 1e-12
    # second-largest eigenvalue modulus of P^pi* is exactly one half
    lam2 = sorted(np.abs(np.linalg.eigvals(P_pi_f)))[-2]
    assert abs(lam2 - 0.5) < 1e-12


def test_occupancy_and_concentrability():
    Pf, _ = frac_mdp()
    nu = [Fraction(1), Fraction(0)]
    P_pi = [Pf[LOW][KEEP], Pf[HIGH][REPLACE]]
    occ_pi = occupancy_frac(P_pi, nu)
    assert occ_pi == [Fraction(20, 29), Fraction(9, 29)]
    # logging policy: keep with probability 0.1 in both states
    P_mu = [
        [BETA * Pf[s][KEEP][t] + (1 - BETA) * Pf[s][REPLACE][t] for t in (LOW, HIGH)]
        for s in (LOW, HIGH)
    ]
    occ_mu = occupancy_frac(P_mu, nu)
    assert occ_mu == [Fraction(182, 191), Fraction(9, 191)]
    # state-level concentrability: max_s d^pi(s) / d^mu(s) = 191/29
    ratios = [occ_pi[s] / occ_mu[s] for s in (LOW, HIGH)]
    assert max(ratios) == Fraction(191, 29)
    assert round(float(max(ratios)), 4) == 6.5862
    # state-action concentrability: the target plays replace at high mileage, the log
    # plays it with probability 1 - beta there, so the binding ratio is 1910/261
    sa_ratio = occ_pi[HIGH] / (occ_mu[HIGH] * (1 - BETA))
    other = occ_pi[LOW] / (occ_mu[LOW] * BETA)
    assert other == Fraction(19100, 2639)
    assert max(sa_ratio, other) == Fraction(1910, 261)
    assert round(float(max(sa_ratio, other)), 4) == 7.318
    # float pipeline reproduces the same numbers
    P, r = build_mdp()
    P_pi_f, _ = policy_matrices(P, r, [KEEP, REPLACE])
    P_mu_f, _, b = stochastic_policy_matrices(P, r, 0.1)
    occ_pi_f = discounted_occupancy(P_pi_f, GAMMA, np.array([1.0, 0.0]))
    occ_mu_f = discounted_occupancy(P_mu_f, GAMMA, np.array([1.0, 0.0]))
    assert np.max(np.abs(occ_pi_f - np.array([20 / 29, 9 / 29]))) < 1e-12
    assert abs(occ_pi_f[HIGH] / (occ_mu_f[HIGH] * b[HIGH, REPLACE]) - 1910 / 261) < 1e-9


def test_grid_reduces_to_canonical():
    P2, r2 = build_mdp_grid(2)
    P, r = build_mdp()
    assert np.array_equal(P2, P)
    assert np.array_equal(r2, r)
    Pf, rf = build_mdp_grid_frac(2)
    assert rf == [[Fraction(1), Fraction(-1, 2)], [Fraction(1, 5), Fraction(-1, 2)]]
    assert Pf[LOW][KEEP] == [Fraction(1, 2), Fraction(1, 2)]
    assert Pf[HIGH][KEEP] == [Fraction(0), Fraction(1)]
    assert Pf[LOW][REPLACE] == [Fraction(1), Fraction(0)]
    assert Pf[HIGH][REPLACE] == [Fraction(1), Fraction(0)]


def test_grid_k3_hand_derived_oracle():
    # Independent pin for the K > 2 builder, derived by hand from the definition and
    # not from the module: at K = 3 the keep rewards are (1, 3/5, 1/5) and the
    # keep-everywhere policy satisfies V2 = (1/5)/(1/10) = 2,
    # 0.55 V1 = 3/5 + 0.45 * 2  => V1 = 30/11,
    # 0.55 V0 = 1 + 0.45 * 30/11 => V0 = 490/121.
    Pf, rf = build_mdp_grid_frac(3)
    assert rf[1][KEEP] == Fraction(3, 5)
    V = exact_value_frac(Pf, rf, [KEEP, KEEP, KEEP])
    assert V == [Fraction(490, 121), Fraction(30, 11), Fraction(2)]
    P, r = build_mdp_grid(3)
    P_pi, r_pi = policy_matrices(P, r, [KEEP, KEEP, KEEP])
    V_float = exact_value(P_pi, r_pi, GAMMA)
    assert np.max(np.abs(V_float - np.array([490 / 121, 30 / 11, 2.0]))) < 1e-12
    # and the exact and float optimal solvers agree at K = 3
    Vf, pf, _ = solve_optimal_frac(Pf, rf)
    V_opt, greedy, _ = solve_optimal(P, r, GAMMA)
    assert list(greedy) == pf
    assert np.max(np.abs(V_opt - np.array([float(x) for x in Vf]))) < 1e-10


def test_grid_sanity():
    for K in (3, 21):
        P, r = build_mdp_grid(K)
        assert np.allclose(P.sum(axis=2), 1.0)
        V, greedy, _ = solve_optimal(P, r, GAMMA)
        # value is non-increasing in mileage grade
        assert np.all(np.diff(V) <= 1e-12)
        # replacing at the top grade beats keeping there
        assert greedy[K - 1] == REPLACE


def test_ev_variant():
    P, r = build_mdp()
    V_star, greedy, _ = solve_optimal(P, r, GAMMA)
    # small shocks: the smoothed value approaches V* and the CCPs the greedy actions
    W, _, ccp = solve_ev(P, r, GAMMA, sigma=1e-4)
    assert np.max(np.abs(W - V_star)) < 1e-3
    assert list(ccp.argmax(axis=1)) == list(greedy)
    # moderate shocks: every CCP interior
    _, _, ccp = solve_ev(P, r, GAMMA, sigma=0.5)
    assert np.all(ccp > 0.0) and np.all(ccp < 1.0)


def test_iterative_solvers_fail_if_iteration_budget_is_exhausted():
    P, r = build_mdp()
    for solver, kwargs in (
        (solve_optimal, {}),
        (solve_ev, {"sigma": 0.5}),
    ):
        try:
            solver(P, r, GAMMA, max_iter=0, **kwargs)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"{solver.__name__} silently returned without converging"
            )


def test_confounded_variant():
    P_by_u, r, q = build_mdp_confounded(0.0)
    P, r0 = build_mdp()
    assert np.array_equal(P_by_u[0], P) and np.array_equal(P_by_u[1], P)
    assert np.array_equal(r, r0)
    # symmetric shift: the U-mixture kernel averages back to the canonical one
    P_by_u, _, q = build_mdp_confounded(0.2, q=0.5)
    assert np.allclose(q * P_by_u[1] + (1 - q) * P_by_u[0], P)


def test_policy_gradient_matches_finite_difference():
    P, r = build_mdp()
    nu = np.array([1.0, 0.0])
    theta = np.array([0.3, -0.4])
    grad, _ = policy_gradient(P, r, GAMMA, theta, nu)
    eps = 1e-6
    for s in range(2):
        tp, tm = theta.copy(), theta.copy()
        tp[s] += eps
        tm[s] -= eps
        Jp, _, _ = policy_performance(P, r, GAMMA, policy_from_logits(tp), nu)
        Jm, _, _ = policy_performance(P, r, GAMMA, policy_from_logits(tm), nu)
        assert abs((Jp - Jm) / (2 * eps) - grad[s]) < 1e-6


def test_natural_gradient_is_action_value_gap():
    P, r = build_mdp()
    nu = np.array([1.0, 0.0])
    theta = np.array([0.3, -0.4])
    ngrad, aux = natural_gradient(P, r, GAMMA, theta, nu)
    assert np.max(np.abs(ngrad - aux["gap"])) < 1e-12
    F = fisher_matrix(theta, aux["rho"])
    assert np.array_equal(F, np.diag(np.diag(F)))  # diagonal
    assert np.linalg.det(F) > 0.0


def test_module_never_calls_pinv():
    src = open(
        os.path.join(os.path.dirname(__file__), "..", "sims", "engine.py")
    ).read()
    assert "pinv" not in src


def test_policy_square_script_never_calls_pinv():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ch03_theory",
        "sims",
        "engine_policy_square.py",
    )
    assert "pinv" not in open(path).read()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")
