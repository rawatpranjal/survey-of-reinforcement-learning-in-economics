# The Engine Replacement MDP: a two-state machine-replacement model
# Appendix A - Mathematical Preliminaries
# Pins every number quoted in Appendix A.2 through A.7. A miniature of the bus-engine
# replacement problem of Chapters 4 and 5: a machine is good or worn, and each period the
# operator keeps it (earning output, letting it degrade) or replaces it (paying a fixed
# cost, returning it to good). Two states keep every object closed-form and drawable.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

# The MDP primitives and solvers live in the shared Engine module (sims/engine.py),
# the book-wide Engine Replacement MDP. This script narrates the same instance in the
# appendix's good/worn language and pins the numbers Appendix A quotes.
from sims.engine import (
    build_mdp,
    policy_matrices,
    stochastic_policy_matrices,
    exact_value,
    bellman_optimality,
    solve_optimal,
    stationary_distribution,
    discounted_occupancy,
    projected_modulus,
)

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCRIPT_NAME = "running_example"
OUTPUT_DIR = os.path.dirname(__file__)

GOOD, WORN = 0, 1
KEEP, REPLACE = 0, 1
STATE_NAMES = ["good", "worn"]
ACTION_NAMES = ["keep", "replace"]

# Layer 1: configuration. GAMMA is fixed by the survey's convention; the rest is searched.
GAMMA = 0.9

# The MDP primitives are round numbers, chosen so a reader can redo every calculation
# by hand, and then verified against the two properties the appendix needs (the optimal
# policy is keep-when-good and replace-when-worn, and the envelope kink sits strictly
# above the fixed point). The two geometry parameters that are not pinned by the
# economics, the feature ratio and the logging probability, are searched.
MDP_PRIMITIVES = {
    "gamma": GAMMA,
    "r_keep_good": 1.0,
    "r_keep_worn": 0.2,
    "replace_cost": 0.5,
    "degrade_prob": 0.5,
}

SEARCH_CONFIG = {
    **MDP_PRIMITIVES,
    # the single feature phi = (1, k), so k is the worn-to-good feature ratio
    "feature_ratio": list(np.round(np.arange(1.5, 5.01, 0.1), 2)),
    # the logging policy keeps with probability beta and replaces otherwise
    "behavior_keep_prob": list(np.round(np.arange(0.05, 0.51, 0.05), 2)),
    # margins the winning parameterization must clear
    "min_action_gap": 0.05,
    "min_kink_margin": 0.05,
    "min_off_modulus": 1.2,
    "version": 3,
}

VI_CONFIG = {**SEARCH_CONFIG, "vi_steps": 12}
NEUMANN_CONFIG = {**SEARCH_CONFIG, "truncations": [1, 3, 10]}
TD_CONFIG = {
    **SEARCH_CONFIG,
    "td_steps": 20000,
    "td_seeds": 20,
    "schedules": [
        "constant",
        "one_over_n",
        "one_over_n_pow_0.7",
        "one_over_n_squared",
    ],
    "constant_step": 0.05,
}


# ---------------------------------------------------------------------------
# Layer 2: the parameter search
# ---------------------------------------------------------------------------


def _search():
    """Verify the MDP primitives, then search the two free geometry parameters.

    The economics is fixed by round numbers so every value in the appendix can be
    redone by hand. Those numbers still have to satisfy two properties, and the
    function checks them rather than assuming them. The feature ratio and the logging
    probability are not pinned by the economics, so they are searched. Among every pair
    where the on-policy projected operator contracts and the off-policy one expands by
    at least min_off_modulus, the winner uses the closest induced stationary weighting
    among surviving grid points, followed by the smallest feature ratio and thus the
    smallest asymmetry between the two states. Maximizing the modulus instead would
    select the most extreme grid boundary and report an artifact of the grid.
    """
    cfg = SEARCH_CONFIG
    gamma = cfg["gamma"]
    r_kg = cfg["r_keep_good"]
    r_kw = cfg["r_keep_worn"]
    c = cfg["replace_cost"]
    p = cfg["degrade_prob"]

    print("Parameter selection")
    print("  MDP primitives, fixed at round values for hand-checkability:")
    print(f"    r(good, keep) = {r_kg}, r(worn, keep) = {r_kw}, replacement cost = {c}")
    print(f"    P(worn | good, keep) = {p}, gamma = {gamma}")

    P, r = build_mdp(r_kg, r_kw, c, p)
    V_star, greedy, q = solve_optimal(P, r, gamma)

    # Property 1: the optimal policy is keep-when-good, replace-when-worn, strictly.
    gap_good = float(q[GOOD, KEEP] - q[GOOD, REPLACE])
    gap_worn = float(q[WORN, REPLACE] - q[WORN, KEEP])
    ok_policy = (
        greedy[GOOD] == KEEP
        and greedy[WORN] == REPLACE
        and min(gap_good, gap_worn) >= cfg["min_action_gap"]
    )
    print(
        f"  property 1, optimal policy is (keep, replace): {ok_policy}, "
        f"action gaps {gap_good:.4f} at good and {gap_worn:.4f} at worn"
    )

    # Property 2: in the worn-state slice V = (V*_good, t), the keep line r_kw + gamma t
    # and the constant replace line -c + gamma V*_good cross strictly above the fixed
    # point V*_worn. That crossing is the kink drawn in the envelope diagram.
    kink = float((-c + gamma * V_star[GOOD] - r_kw) / gamma)
    kink_margin = float(kink - V_star[WORN])
    ok_kink = kink_margin >= cfg["min_kink_margin"]
    print(
        f"  property 2, envelope kink at t = {kink:.4f} sits {kink_margin:.4f} above "
        f"V*(worn) = {V_star[WORN]:.4f}: {ok_kink}"
    )

    pi_star = [KEEP, REPLACE]
    P_pi, _ = policy_matrices(P, r, pi_star)
    d_pi = stationary_distribution(P_pi)

    best = None
    n_pairs = 0
    n_survivors = 0
    for beta in cfg["behavior_keep_prob"]:
        P_mu, _, _ = stochastic_policy_matrices(P, r, beta)
        d_mu = stationary_distribution(P_mu)
        for k in cfg["feature_ratio"]:
            n_pairs += 1
            phi = np.array([1.0, float(k)])
            on_mod, _ = projected_modulus(phi, d_pi, P_pi, gamma)
            off_mod, _ = projected_modulus(phi, d_mu, P_pi, gamma)
            # Property 3: on-policy contracts at least as fast as gamma, off-policy expands.
            if on_mod >= gamma or off_mod < cfg["min_off_modulus"]:
                continue
            n_survivors += 1
            # Lexicographic: mildest logging bias first, then mildest feature asymmetry.
            mildness = (beta, -k)
            if best is None or mildness > (
                best["behavior_keep_prob"],
                -best["feature_ratio"],
            ):
                best = {
                    "gamma": gamma,
                    "r_keep_good": r_kg,
                    "r_keep_worn": r_kw,
                    "replace_cost": c,
                    "degrade_prob": p,
                    "feature_ratio": float(k),
                    "behavior_keep_prob": float(beta),
                    "on_modulus": float(on_mod),
                    "off_modulus": float(off_mod),
                    "kink": kink,
                    "kink_margin": kink_margin,
                    "action_gap_good": gap_good,
                    "action_gap_worn": gap_worn,
                }

    print(
        f"  property 3, searched {n_pairs} (logging keep-probability, feature ratio) pairs, "
        f"{n_survivors} give on-policy contraction and an off-policy modulus of at least "
        f"{cfg['min_off_modulus']}"
    )
    if best is None:
        raise RuntimeError("no (beta, feature ratio) pair satisfies property 3")
    print(
        f"    mildest survivor: logging keep-probability {best['behavior_keep_prob']}, "
        f"feature ratio {best['feature_ratio']}"
    )
    print(
        f"    on-policy modulus {best['on_modulus']:.4f}, "
        f"off-policy modulus {best['off_modulus']:.4f}"
    )
    # How the best attainable modulus falls away as the logging policy approaches the
    # target. Expansion is not a switch that flips, it is continuous in how far the
    # logging law has drifted from the law the projection ought to use.
    print(
        "    largest off-policy modulus attainable at each logging keep-probability, "
        "over the feature grid:"
    )
    print(
        f"      {'keep prob':>10s}  {'d^mu(good)':>11s}  {'best modulus':>13s}  {'at ratio':>9s}"
    )
    for beta in cfg["behavior_keep_prob"]:
        P_mu, _, _ = stochastic_policy_matrices(P, r, beta)
        d_mu = stationary_distribution(P_mu)
        mods = [
            (
                projected_modulus(np.array([1.0, float(k)]), d_mu, P_pi, gamma)[0],
                float(k),
            )
            for k in cfg["feature_ratio"]
        ]
        top_mod, top_k = max(mods)
        print(f"      {beta:10.2f}  {d_mu[GOOD]:11.4f}  {top_mod:13.4f}  {top_k:9.1f}")
    print(
        f"    the target policy's own stationary law is d^pi(good) = {d_pi[GOOD]:.4f}, and at a"
    )
    print(
        f"    logging keep-probability of {cfg['behavior_keep_prob'][-1]} the two laws coincide, so the"
    )
    print(
        "    modulus falls back below one; expansion grows continuously with the drift"
    )
    print()

    # The three properties, asserted so a later edit that breaks one fails loudly
    # rather than silently changing what the appendix claims.
    assert ok_policy, "optimal policy is not (keep, replace) with the required margin"
    assert ok_kink, "envelope kink is not strictly above the fixed point"
    assert best["on_modulus"] < gamma < 1.0 < best["off_modulus"], (
        "triad moduli do not straddle one"
    )
    return best


# ---------------------------------------------------------------------------
# Layer 2 (continued): everything the appendix quotes
# ---------------------------------------------------------------------------


def _shared(params):
    gamma = params["gamma"]
    P, r = build_mdp(
        params["r_keep_good"],
        params["r_keep_worn"],
        params["replace_cost"],
        params["degrade_prob"],
    )
    V_star, greedy, q_star = solve_optimal(P, r, gamma)

    print("=" * 70)
    print("A.2  THE MDP")
    print("=" * 70)
    print(f"  states  : {STATE_NAMES}")
    print(f"  actions : {ACTION_NAMES}")
    print(f"  gamma   : {gamma}")
    print("  reward r(s,a):")
    for s in range(2):
        for a in range(2):
            print(f"    r({STATE_NAMES[s]:5s}, {ACTION_NAMES[a]:8s}) = {r[s, a]:+.3f}")
    print("  transition P(s'|s,a):")
    for s in range(2):
        for a in range(2):
            print(
                f"    P(.|{STATE_NAMES[s]:5s}, {ACTION_NAMES[a]:8s}) = "
                f"[good {P[s, a, GOOD]:.3f}, worn {P[s, a, WORN]:.3f}]"
            )
    print()

    policies = {
        "keep-always": [KEEP, KEEP],
        "replace-always": [REPLACE, REPLACE],
        "optimal (keep, replace)": [KEEP, REPLACE],
        "reverse (replace, keep)": [REPLACE, KEEP],
    }
    policy_rows = {}
    print("  The four deterministic policies, with P^pi, r^pi and V^pi:")
    for name, pol in policies.items():
        P_pi, r_pi = policy_matrices(P, r, pol)
        V_pi = exact_value(P_pi, r_pi, gamma)
        policy_rows[name] = {
            "policy": pol,
            "P_pi": P_pi,
            "r_pi": r_pi,
            "V_pi": V_pi,
        }
        print(f"    {name}")
        print(
            f"      P^pi = [[{P_pi[0, 0]:.3f}, {P_pi[0, 1]:.3f}], [{P_pi[1, 0]:.3f}, {P_pi[1, 1]:.3f}]]"
        )
        print(f"      r^pi = [{r_pi[0]:+.3f}, {r_pi[1]:+.3f}]")
        print(f"      V^pi = [good {V_pi[GOOD]:.4f}, worn {V_pi[WORN]:.4f}]")
    print()
    print(f"  V* = [good {V_star[GOOD]:.4f}, worn {V_star[WORN]:.4f}]")
    print("  Q*(s,a):")
    for s in range(2):
        for a in range(2):
            print(
                f"    Q*({STATE_NAMES[s]:5s}, {ACTION_NAMES[a]:8s}) = {q_star[s, a]:.4f}"
            )
    print(f"  action gap at good (keep over replace) : {params['action_gap_good']:.4f}")
    print(f"  action gap at worn (replace over keep) : {params['action_gap_worn']:.4f}")
    print()

    return {
        "P": P,
        "r": r,
        "V_star": V_star,
        "greedy": greedy,
        "q_star": q_star,
        "policies": policy_rows,
        "params": params,
    }


def _value_iteration(shared):
    params = shared["params"]
    gamma = params["gamma"]
    P, r, V_star = shared["P"], shared["r"], shared["V_star"]
    n = VI_CONFIG["vi_steps"]

    print("=" * 70)
    print("A.3  VALUE ITERATION AND THE ENVELOPE")
    print("=" * 70)
    V = np.zeros(2)
    iterates = [V.copy()]
    errors = [float(np.max(np.abs(V - V_star)))]
    greedy_path = []
    print(
        f"  {'k':>2s}  {'V_k(good)':>10s}  {'V_k(worn)':>10s}  {'sup error':>10s}  {'ratio':>7s}  greedy"
    )
    print(f"  {0:2d}  {V[GOOD]:10.4f}  {V[WORN]:10.4f}  {errors[0]:10.4f}  {'':>7s}")
    for k in range(1, n + 1):
        V, greedy, _ = bellman_optimality(P, r, V, gamma)
        iterates.append(V.copy())
        err = float(np.max(np.abs(V - V_star)))
        errors.append(err)
        ratio = err / errors[-2] if errors[-2] > 0 else float("nan")
        greedy_path.append([int(greedy[0]), int(greedy[1])])
        print(
            f"  {k:2d}  {V[GOOD]:10.4f}  {V[WORN]:10.4f}  {err:10.4f}  {ratio:7.4f}  "
            f"({ACTION_NAMES[greedy[GOOD]]}, {ACTION_NAMES[greedy[WORN]]})"
        )
    ratios = [errors[i] / errors[i - 1] for i in range(1, len(errors))]
    max_ratio = max(ratios)
    tail_ratio = float(np.mean(ratios[-4:]))
    print(
        f"  largest ratio over the {len(ratios)} steps: {max_ratio:.4f}, which is at most gamma = {gamma}"
    )
    print(
        f"  mean of the last four ratios: {tail_ratio:.4f}, so the error settles into decay at rate gamma"
    )
    print(
        "  the early ratios fall below gamma because the second mode of gamma P^pi has"
    )
    print(
        "  modulus below gamma and dies out first; the contraction gives an upper bound"
    )
    print("  on each step, not an equality")
    assert max_ratio <= gamma + 1e-9, (
        "a value-iteration step contracted by more than gamma"
    )
    # The two modes of gamma P^pi, which is what actually sets the per-step ratios. The
    # appendix quotes the second modulus, so it is emitted here rather than derived by hand.
    P_pi_star, _ = policy_matrices(P, r, [KEEP, REPLACE])
    modes = np.sort(np.abs(np.linalg.eigvals(gamma * P_pi_star)))[::-1]
    print(
        f"  eigenvalue moduli of gamma P^pi: leading {modes[0]:.4f} (= gamma), "
        f"second {modes[1]:.4f}"
    )
    print("  the second mode decays faster, which is why early ratios fall below gamma")
    print()

    # The envelope slice used by diagram D4: hold V(good) at its optimal value and vary
    # V(worn) along t. The two action lines at the worn state are affine in t.
    r_kw = params["r_keep_worn"]
    c = params["replace_cost"]
    kink = params["kink"]
    print("  Envelope slice at the worn state, V = (V*(good), t):")
    print(
        f"    keep line    : (T^keep V)(worn)    = {r_kw:+.3f} + {gamma} t          (slope {gamma})"
    )
    print(
        f"    replace line : (T^replace V)(worn) = {-c:+.3f} + {gamma} * {V_star[GOOD]:.4f} = {-c + gamma * V_star[GOOD]:.4f}   (slope 0)"
    )
    print(f"    kink at t    = {kink:.4f}")
    print(
        f"    fixed point  V*(worn) = {V_star[WORN]:.4f}, which is {params['kink_margin']:.4f} below the kink"
    )
    print(
        f"    so replace is greedy at the optimum, and keep takes over for t > {kink:.4f}"
    )
    print()

    # Iterations to a fixed tolerance, the 1/(1-gamma) effective-horizon reading.
    print("  Iterations of value iteration to reach sup-norm error below 0.01:")
    print(f"    {'gamma':>6s}  {'1/(1-gamma)':>12s}  {'iterations':>10s}")
    horizon_rows = []
    for g in [0.5, 0.9, 0.99]:
        Vg, _, _ = solve_optimal(P, r, g)
        Vk = np.zeros(2)
        it = 0
        while np.max(np.abs(Vk - Vg)) >= 0.01 and it < 100000:
            Vk, _, _ = bellman_optimality(P, r, Vk, g)
            it += 1
        horizon_rows.append((g, 1.0 / (1.0 - g), it))
        print(f"    {g:6.2f}  {1.0 / (1.0 - g):12.1f}  {it:10d}")
    print()

    return {
        "iterates": np.array(iterates),
        "errors": np.array(errors),
        "greedy_path": greedy_path,
        "kink": kink,
        "horizon_rows": horizon_rows,
    }


def _neumann(shared):
    params = shared["params"]
    gamma = params["gamma"]
    P, r = shared["P"], shared["r"]
    P_pi, r_pi = policy_matrices(P, r, [KEEP, REPLACE])
    exact_inv = np.linalg.inv(np.eye(2) - gamma * P_pi)
    V_exact = exact_inv @ r_pi

    print("=" * 70)
    print("A.4  THE RESOLVENT AND THE NEUMANN SERIES")
    print("=" * 70)
    print("  Evaluating the optimal policy pi* = (keep, replace).")
    print(
        f"    P^pi   = [[{P_pi[0, 0]:.3f}, {P_pi[0, 1]:.3f}], [{P_pi[1, 0]:.3f}, {P_pi[1, 1]:.3f}]]"
    )
    print(f"    r^pi   = [{r_pi[0]:+.3f}, {r_pi[1]:+.3f}]")
    print(
        f"    I - gamma P^pi = [[{1 - gamma * P_pi[0, 0]:.4f}, {-gamma * P_pi[0, 1]:.4f}], "
        f"[{-gamma * P_pi[1, 0]:.4f}, {1 - gamma * P_pi[1, 1]:.4f}]]"
    )
    det = np.linalg.det(np.eye(2) - gamma * P_pi)
    print(f"    determinant = {det:.6f}")
    print(
        f"    (I - gamma P^pi)^-1 = [[{exact_inv[0, 0]:.4f}, {exact_inv[0, 1]:.4f}], "
        f"[{exact_inv[1, 0]:.4f}, {exact_inv[1, 1]:.4f}]]"
    )
    print(
        f"    row sums of the resolvent = [{exact_inv[0].sum():.4f}, {exact_inv[1].sum():.4f}], "
        f"both equal 1/(1-gamma) = {1 / (1 - gamma):.4f}"
    )
    print(f"    V^pi = [good {V_exact[GOOD]:.4f}, worn {V_exact[WORN]:.4f}]")
    print()

    print("  Truncating the Neumann series sum_{m=0}^{M} (gamma P^pi)^m r^pi:")
    print(
        f"    {'M':>3s}  {'V_M(good)':>10s}  {'V_M(worn)':>10s}  {'sup error':>10s}  {'bound':>10s}"
    )
    rows = []
    partial = np.zeros(2)
    term = np.eye(2)
    r_inf = float(np.max(np.abs(r_pi)))
    for M in range(0, max(NEUMANN_CONFIG["truncations"]) + 1):
        partial = partial + term @ r_pi
        term = gamma * term @ P_pi
        if M in NEUMANN_CONFIG["truncations"]:
            err = float(np.max(np.abs(partial - V_exact)))
            bound = gamma ** (M + 1) / (1 - gamma) * r_inf
            rows.append((M, partial.copy(), err, bound))
            print(
                f"    {M:3d}  {partial[GOOD]:10.4f}  {partial[WORN]:10.4f}  {err:10.4f}  {bound:10.4f}"
            )
    print(
        f"    the bound is gamma^(M+1) ||r^pi||_inf / (1 - gamma) with ||r^pi||_inf = {r_inf:.3f}"
    )
    print()
    return {"exact_inv": exact_inv, "V_exact": V_exact, "rows": rows, "r_inf": r_inf}


def _occupancy(shared):
    params = shared["params"]
    gamma = params["gamma"]
    beta = params["behavior_keep_prob"]
    P, r = shared["P"], shared["r"]

    P_pi, _ = policy_matrices(P, r, [KEEP, REPLACE])
    P_mu, _, b = stochastic_policy_matrices(P, r, beta)
    d_pi = stationary_distribution(P_pi)
    d_mu = stationary_distribution(P_mu)
    nu = np.array([1.0, 0.0])  # start in good
    occ_pi = discounted_occupancy(P_pi, gamma, nu)
    occ_mu = discounted_occupancy(P_mu, gamma, nu)

    print("=" * 70)
    print("A.5  MARKOV CHAINS, OCCUPANCY AND COVERAGE")
    print("=" * 70)
    print(
        f"  Target policy pi* = (keep, replace). Logging policy mu keeps with probability {beta}."
    )
    print(
        f"    P^pi = [[{P_pi[0, 0]:.3f}, {P_pi[0, 1]:.3f}], [{P_pi[1, 0]:.3f}, {P_pi[1, 1]:.3f}]]"
    )
    print(
        f"    P^mu = [[{P_mu[0, 0]:.3f}, {P_mu[0, 1]:.3f}], [{P_mu[1, 0]:.3f}, {P_mu[1, 1]:.3f}]]"
    )
    print(f"    stationary d^pi = [good {d_pi[GOOD]:.4f}, worn {d_pi[WORN]:.4f}]")
    print(f"    stationary d^mu = [good {d_mu[GOOD]:.4f}, worn {d_mu[WORN]:.4f}]")
    w_pi, _ = np.linalg.eig(P_pi)
    w_mu, _ = np.linalg.eig(P_mu)
    lam2_pi = float(np.sort(np.abs(w_pi))[-2])
    lam2_mu = float(np.sort(np.abs(w_mu))[-2])
    print(
        f"    second eigenvalue modulus |lambda_2|: pi {lam2_pi:.4f}, mu {lam2_mu:.4f}"
    )
    print(
        "    mixing: after k steps the distribution is within C |lambda_2|^k of stationary"
    )
    print()
    print(
        "  Discounted occupancy from nu = [1, 0] (start good), d = (1-gamma) nu (I - gamma P)^-1:"
    )
    print(
        f"    d^pi_nu = [good {occ_pi[GOOD]:.4f}, worn {occ_pi[WORN]:.4f}]  (sums to {occ_pi.sum():.4f})"
    )
    print(
        f"    d^mu_nu = [good {occ_mu[GOOD]:.4f}, worn {occ_mu[WORN]:.4f}]  (sums to {occ_mu.sum():.4f})"
    )
    print()
    ratio_state = occ_pi / occ_mu
    print("  Change of measure, state-level ratio w(s) = d^pi_nu(s) / d^mu_nu(s):")
    for s in range(2):
        print(f"    w({STATE_NAMES[s]:5s}) = {ratio_state[s]:.4f}")
    print(f"    concentrability C_inf = max_s w(s) = {ratio_state.max():.4f}")
    print()

    # State-action coverage: pi* is deterministic, mu is stochastic, so every
    # state-action pair the target uses is logged, but the worn-keep pair the target
    # never uses is over-represented and the good-replace pair is nearly absent.
    d_sa_pi = np.zeros((2, 2))
    d_sa_mu = np.zeros((2, 2))
    pi_star = [KEEP, REPLACE]
    for s in range(2):
        d_sa_pi[s, pi_star[s]] = occ_pi[s]
        d_sa_mu[s] = occ_mu[s] * b[s]
    print("  State-action occupancy:")
    print(f"    {'(s,a)':>18s}  {'d^pi':>8s}  {'d^mu':>8s}  {'ratio':>8s}")
    sa_rows = []
    for s in range(2):
        for a in range(2):
            ratio = d_sa_pi[s, a] / d_sa_mu[s, a] if d_sa_mu[s, a] > 0 else float("inf")
            sa_rows.append((s, a, d_sa_pi[s, a], d_sa_mu[s, a], ratio))
            print(
                f"    ({STATE_NAMES[s]:5s}, {ACTION_NAMES[a]:8s})  {d_sa_pi[s, a]:8.4f}  "
                f"{d_sa_mu[s, a]:8.4f}  {ratio:8.4f}"
            )
    finite = [row[4] for row in sa_rows if np.isfinite(row[4])]
    print(f"    state-action concentrability C_inf = {max(finite):.4f}")
    print()
    print("  Share of the log's discounted weight, by state-action pair:")
    for s, a, dpi_sa, dmu_sa, _ in sa_rows:
        used = "used by the target" if dpi_sa > 0 else "never used by the target"
        print(
            f"    ({STATE_NAMES[s]:5s}, {ACTION_NAMES[a]:8s}) {100 * dmu_sa:6.1f} percent, {used}"
        )
    print()

    return {
        "P_pi": P_pi,
        "P_mu": P_mu,
        "d_pi": d_pi,
        "d_mu": d_mu,
        "occ_pi": occ_pi,
        "occ_mu": occ_mu,
        "ratio_state": ratio_state,
        "sa_rows": sa_rows,
        "lam2_pi": lam2_pi,
        "lam2_mu": lam2_mu,
        "behavior_matrix": b,
    }


def _projection(shared, occ):
    params = shared["params"]
    gamma = params["gamma"]
    k = params["feature_ratio"]
    phi = np.array([1.0, k])
    Phi = phi.reshape(2, 1)
    P_pi = occ["P_pi"]
    d_pi, d_mu = occ["d_pi"], occ["d_mu"]
    P, r = shared["P"], shared["r"]
    _, r_pi = policy_matrices(P, r, [KEEP, REPLACE])
    V_pi = exact_value(P_pi, r_pi, gamma)

    print("=" * 70)
    print("A.6  APPROXIMATION GEOMETRY AND THE DEADLY TRIAD")
    print("=" * 70)
    print(
        f"  One feature, Phi = [{phi[GOOD]:.1f}, {phi[WORN]:.1f}]^T, so the representable"
    )
    print("  value functions are the line span(Phi) through the origin.")
    print(
        f"  V^pi points in the direction (1, {V_pi[WORN] / V_pi[GOOD]:.4f}), so the feature is"
    )
    print("  deliberately far from the direction that would approximate well.")
    print()

    out = {}
    for name, d in [("on-policy d^pi", d_pi), ("off-policy d^mu", d_mu)]:
        D = np.diag(d)
        Pi = Phi @ np.linalg.inv(Phi.T @ D @ Phi) @ Phi.T @ D
        mod, signed = projected_modulus(phi, d, P_pi, gamma)
        # the fixed point of Pi_d T^pi on span(Phi), when it exists
        A = np.eye(2) - Pi @ (gamma * P_pi)
        try:
            V_td = np.linalg.solve(A, Pi @ r_pi)
        except np.linalg.LinAlgError:
            V_td = np.array([np.nan, np.nan])
        proj_V = Pi @ V_pi
        print(f"  Weighting by {name} = [good {d[GOOD]:.4f}, worn {d[WORN]:.4f}]:")
        print(
            f"    Pi_d = [[{Pi[0, 0]:.4f}, {Pi[0, 1]:.4f}], [{Pi[1, 0]:.4f}, {Pi[1, 1]:.4f}]]"
        )
        print(
            f"    Pi_d is idempotent: max|Pi^2 - Pi| = {np.max(np.abs(Pi @ Pi - Pi)):.2e}"
        )
        print(
            f"    best representable approximation Pi_d V^pi = [good {proj_V[GOOD]:.4f}, worn {proj_V[WORN]:.4f}]"
        )
        print(
            f"    modulus of Pi_d T^pi on span(Phi) = {signed:.4f} (absolute value {mod:.4f})"
        )
        print(
            f"    fixed point of Pi_d T^pi = [good {V_td[GOOD]:.4f}, worn {V_td[WORN]:.4f}]"
        )
        print()
        out[name] = {
            "d": d,
            "Pi": Pi,
            "modulus": float(mod),
            "signed": float(signed),
            "V_td": V_td,
            "proj_V": proj_V,
        }

    print(f"  V^pi (exact) = [good {V_pi[GOOD]:.4f}, worn {V_pi[WORN]:.4f}]")
    print(
        f"  The on-policy modulus {out['on-policy d^pi']['modulus']:.4f} is below gamma = {gamma}."
    )
    print(
        f"  The off-policy modulus {out['off-policy d^mu']['modulus']:.4f} exceeds one, so the same"
    )
    print("  algebra now expands and the iterates diverge. Nothing about T^pi changed;")
    print("  only the weighting that defines the projection did.")
    print()

    # Sweep the weighting from d^pi to d^mu and locate the crossing.
    print(
        "  Tilting the weighting from d^pi toward d^mu, w(good) = mixture of the two:"
    )
    print(f"    {'alpha':>6s}  {'d(good)':>8s}  {'d(worn)':>8s}  {'modulus':>8s}")
    sweep = []
    for alpha in np.linspace(0.0, 1.0, 21):
        d = (1 - alpha) * d_pi + alpha * d_mu
        mod, _ = projected_modulus(phi, d, P_pi, gamma)
        sweep.append((float(alpha), float(d[GOOD]), float(d[WORN]), float(mod)))
        if abs(alpha * 20 - round(alpha * 20)) < 1e-9 and round(alpha * 20) % 4 == 0:
            print(f"    {alpha:6.2f}  {d[GOOD]:8.4f}  {d[WORN]:8.4f}  {mod:8.4f}")

    # Locate the crossing by bisection on the bracketing grid interval. The modulus is a
    # ratio of two affine functions of alpha, so it is not linear in alpha and linear
    # interpolation across a 0.05-wide bracket is wrong in the fourth decimal.
    def excess(a):
        d = (1 - a) * d_pi + a * d_mu
        return projected_modulus(phi, d, P_pi, gamma)[0] - 1.0

    crossing = None
    for i in range(1, len(sweep)):
        if sweep[i - 1][3] < 1.0 <= sweep[i][3]:
            lo, hi = sweep[i - 1][0], sweep[i][0]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if excess(mid) < 0.0:
                    lo = mid
                else:
                    hi = mid
            crossing = 0.5 * (lo + hi)
            break
    if crossing is None:
        print("    the modulus does not cross one on this path")
    else:
        print(f"    the modulus crosses one at alpha = {crossing:.4f}")
        assert abs(excess(crossing)) < 1e-9, (
            "bisection did not converge to the crossing"
        )
    print()

    return {
        "phi": phi,
        "sweep": sweep,
        "crossing": crossing,
        "V_pi": V_pi,
        **{"on_policy": out["on-policy d^pi"], "off_policy": out["off-policy d^mu"]},
    }


def _td_learning(shared, occ):
    params = shared["params"]
    gamma = params["gamma"]
    P, r = shared["P"], shared["r"]
    P_pi, r_pi = policy_matrices(P, r, [KEEP, REPLACE])
    V_pi = exact_value(P_pi, r_pi, gamma)
    pi_star = [KEEP, REPLACE]
    n_steps = TD_CONFIG["td_steps"]
    n_seeds = TD_CONFIG["td_seeds"]
    const = TD_CONFIG["constant_step"]

    print("=" * 70)
    print("A.7  STOCHASTIC APPROXIMATION: TABULAR TD(0) UNDER FOUR STEP SIZES")
    print("=" * 70)
    print(
        f"  Evaluating pi* = (keep, replace) from sampled transitions, {n_seeds} seeds,"
    )
    print(
        f"  {n_steps} steps each, target V^pi = [good {V_pi[GOOD]:.4f}, worn {V_pi[WORN]:.4f}]."
    )
    print()

    def step_size(schedule, t, visits):
        if schedule == "constant":
            return const
        if schedule == "one_over_n":
            return 1.0 / visits
        if schedule == "one_over_n_pow_0.7":
            return visits**-0.7
        if schedule == "one_over_n_squared":
            return 1.0 / (visits**2)
        raise ValueError(schedule)

    results = {}
    print(
        f"    {'schedule':>20s}  {'sum a_t':>12s}  {'sum a_t^2':>12s}  {'final sup error':>16s}  {'s.e.':>8s}"
    )
    for schedule in TD_CONFIG["schedules"]:
        final_errors = []
        curves = []
        alpha_sum = alpha_sq_sum = 0.0
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            V = np.zeros(2)
            s = GOOD
            visits = np.ones(2)
            curve = []
            for t in range(1, n_steps + 1):
                a = pi_star[s]
                s_next = rng.choice(2, p=P[s, a])
                alpha = step_size(schedule, t, visits[s])
                if seed == 0:
                    alpha_sum += alpha
                    alpha_sq_sum += alpha**2
                V[s] = V[s] + alpha * (r[s, a] + gamma * V[s_next] - V[s])
                visits[s] += 1
                s = s_next
                if t % (n_steps // 200) == 0:
                    curve.append(float(np.max(np.abs(V - V_pi))))
            final_errors.append(float(np.max(np.abs(V - V_pi))))
            curves.append(curve)
        mean_err = float(np.mean(final_errors))
        se_err = float(np.std(final_errors, ddof=1) / np.sqrt(n_seeds))
        results[schedule] = {
            "final_errors": final_errors,
            "mean": mean_err,
            "se": se_err,
            "curves": np.array(curves),
            "alpha_sum": alpha_sum,
            "alpha_sq_sum": alpha_sq_sum,
        }
        print(
            f"    {schedule:>20s}  {alpha_sum:12.2f}  {alpha_sq_sum:12.4f}  "
            f"{mean_err:16.4f}  {se_err:8.4f}"
        )
    print()
    print("  Reading the two Robbins-Monro conditions off the two sums, where n counts")
    print("  visits to the state being updated:")
    print(
        "    constant   : sum a_n diverges, which is what the first condition asks, but"
    )
    print(
        "                 sum a_n^2 diverges too, so the noise is never damped and the"
    )
    print("                 iterate settles into a floor set by the step size.")
    print("    1/n^2      : sum a_n^2 converges, but sum a_n converges as well, so the")
    print("                 total movement available is finite and the iterate stalls")
    print("                 short of the target no matter how long it runs.")
    print("    1/n        : both conditions hold, so this converges, but only in the")
    print("                 limit. Its total movement grows like log n, so after 20000")
    print(
        "                 steps it is still further from the target than the constant"
    )
    print("                 step, which converges fast and then stops improving. The")
    print("                 conditions are asymptotic and say nothing about the rate.")
    print(
        "    1/n^0.7    : both conditions also hold, and the total movement grows like"
    )
    print(
        "                 n^0.3 instead of log n, which is why the polynomial family is"
    )
    print("                 what reinforcement learning actually uses.")
    print()
    # State the comparison the table supports, rather than the one the theory suggests.
    ranked = sorted(results.items(), key=lambda kv: kv[1]["mean"])
    print("  Ranked by final sup-norm error at this budget:")
    for name, res in ranked:
        print(f"    {name:>20s}  {res['mean']:.4f}  (s.e. {res['se']:.4f})")
    best_name = ranked[0][0]
    print(f"  At {n_steps} steps the smallest error is {best_name}. Both Robbins-Monro")
    print(
        "  conditions are necessary for convergence and neither is sufficient for speed."
    )
    print()
    return {"results": results, "V_pi": V_pi}


# ---------------------------------------------------------------------------
# compute / outputs
# ---------------------------------------------------------------------------


def compute_data(force=None):
    force = force or set()
    params = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "search",
        SEARCH_CONFIG,
        _search,
        force=("search" in force),
    )
    cascade = "search" in force
    shared = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "shared",
        {**SEARCH_CONFIG, "p": params},
        _shared,
        params,
        force=("shared" in force or cascade),
    )
    vi = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "vi",
        {**VI_CONFIG, "p": params},
        _value_iteration,
        shared,
        force=("vi" in force or cascade),
    )
    neu = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "neumann",
        {**NEUMANN_CONFIG, "p": params},
        _neumann,
        shared,
        force=("neumann" in force or cascade),
    )
    occ = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "occupancy",
        {**SEARCH_CONFIG, "p": params},
        _occupancy,
        shared,
        force=("occupancy" in force or cascade),
    )
    proj = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "projection",
        {**SEARCH_CONFIG, "p": params},
        _projection,
        shared,
        occ,
        force=("projection" in force or cascade),
    )
    td = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "td",
        {**TD_CONFIG, "p": params},
        _td_learning,
        shared,
        occ,
        force=("td" in force or cascade),
    )
    return {
        "params": params,
        "shared": shared,
        "vi": vi,
        "neumann": neu,
        "occupancy": occ,
        "projection": proj,
        "td": td,
    }


def generate_outputs(data):
    """One LaTeX table summarizing the Engine Replacement MDP for Appendix A.2. No figure:
    the geometry of this example is drawn by the TikZ diagrams, not by matplotlib."""
    p = data["params"]
    shared = data["shared"]
    occ = data["occupancy"]
    proj = data["projection"]
    V_star = shared["V_star"]
    tex_path = os.path.join(OUTPUT_DIR, "running_example.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{The two-state machine-replacement example used throughout this "
            "appendix. Rewards and transitions are the primitives; the optimal value and "
            "the stationary and discounted state distributions are computed from them. "
            "The logging policy $\\mu$ keeps the machine "
            f"with probability {p['behavior_keep_prob']} in either state.}}\n"
        )
        f.write("\\label{tab:prelim_running_example}\n")
        f.write("\\begin{tabular}{lrr}\n\\hline\n")
        f.write(" & good & worn \\\\\n\\hline\n")
        f.write(
            f"$r(s, \\text{{keep}})$ & {p['r_keep_good']:.2f} & {p['r_keep_worn']:.2f} \\\\\n"
        )
        f.write(
            f"$r(s, \\text{{replace}})$ & {-p['replace_cost']:.2f} & {-p['replace_cost']:.2f} \\\\\n"
        )
        f.write(
            f"$P(\\text{{worn}} \\mid s, \\text{{keep}})$ & {p['degrade_prob']:.2f} & 1.00 \\\\\n"
        )
        f.write("\\hline\n")
        f.write(f"$V^\\star(s)$ & {V_star[GOOD]:.4f} & {V_star[WORN]:.4f} \\\\\n")
        f.write("optimal action & keep & replace \\\\\n")
        f.write(
            f"$d^{{\\pi^\\star}}(s)$ & {occ['d_pi'][GOOD]:.4f} & {occ['d_pi'][WORN]:.4f} \\\\\n"
        )
        f.write(
            f"$d^{{\\mu}}(s)$ & {occ['d_mu'][GOOD]:.4f} & {occ['d_mu'][WORN]:.4f} \\\\\n"
        )
        f.write(f"feature $\\phi(s)$ & 1.0 & {p['feature_ratio']:.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")

    # The four step-size schedules, as their own table. The appendix quotes these numbers
    # and must not point at the scalar Robbins-Monro table, a different experiment.
    td = data["td"]
    labels = {
        "constant": f"constant, $\\alpha = {TD_CONFIG['constant_step']}$",
        "one_over_n": "$1/n$",
        "one_over_n_pow_0.7": "$1/n^{0.7}$",
        "one_over_n_squared": "$1/n^2$",
    }
    verdict = {
        "constant": "$\\sum \\alpha_n^2$ diverges",
        "one_over_n": "both hold",
        "one_over_n_pow_0.7": "both hold",
        "one_over_n_squared": "$\\sum \\alpha_n$ converges",
    }
    td_path = os.path.join(OUTPUT_DIR, "running_example_td.tex")
    with open(td_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Tabular TD($0$) on the Engine Replacement MDP, evaluating $\\pi^\\star$ from "
            f"sampled transitions over {TD_CONFIG['td_seeds']} seeds and "
            f"{TD_CONFIG['td_steps']} steps each, against the exact $V^{{\\pi^\\star}}$. "
            "Here $n$ counts visits to the state being updated. The two sums are the "
            "quantities the Robbins-Monro conditions constrain, evaluated along one run. "
            "Error is the supremum-norm distance at the end of the budget, with its "
            "standard error across seeds in brackets.}\n"
        )
        f.write("\\label{tab:prelim_td_schedules}\n")
        f.write("\\begin{tabular}{lrrrl}\n\\hline\n")
        f.write(
            "step size & $\\sum \\alpha_n$ & $\\sum \\alpha_n^2$ & final error & conditions \\\\\n"
        )
        f.write("\\hline\n")
        for name in TD_CONFIG["schedules"]:
            res = td["results"][name]
            f.write(
                f"{labels[name]} & {res['alpha_sum']:.1f} & {res['alpha_sq_sum']:.2f} & "
                f"{res['mean']:.4f} ({res['se']:.4f}) & {verdict[name]} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {td_path}")


def main():
    parser = argparse.ArgumentParser(description="Appendix A Engine Replacement MDP")
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print("=" * 70)
    print("APPENDIX A RUNNING EXAMPLE: TWO-STATE MACHINE REPLACEMENT")
    print("=" * 70)
    print()
    if args.plots_only:
        generate_outputs(compute_data())
    elif args.data_only:
        compute_data(force=force)
    else:
        generate_outputs(compute_data(force=force))
    print("\nDone.")


if __name__ == "__main__":
    main()
