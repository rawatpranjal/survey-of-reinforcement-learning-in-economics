#!/usr/bin/env python3
"""
dtr_policy_learning.py
Chapter (OPE and Dynamic Treatment Effects), Section subsec:dynamic_offline_policy.

Offline dynamic policy LEARNING by doubly robust backward induction
(Sakaguchi 2024 / Athey-Wager AIPW scores), against single-nuisance
baselines, on a two-stage DGP whose optimal threshold regime and the value
of ANY threshold regime are computable to adaptive-integration precision.

Methods (shared cohorts, nuisances, policy class, and search per rep):
  AIPW  backward-induction policy learning on cross-fitted AIPW scores
  DM    plug-in policy learning on the fitted Q alone
  IPW   policy learning on inverse-propensity-weighted outcomes alone
  Behavior policy as the floor.

Policy class per stage: threshold rules pi_t = 1{S_t^(1) < c_t}; the argmax
over the class is exact (sort + prefix maximum, O(n log n)).

Experiments (cached as compute_or_load components; the cached unit is the
experiment, not the algorithm, because the methods must share datasets):
  oracle   c1*, c2*, V*, V(behavior), independent MC cross-checks
  sweep_n  regret vs n with correct nuisances, 40 seeds
  misspec  2x2 nuisance misspecification at n = 1000, 40 seeds
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, BENCH_STYLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

apply_style()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "dtr_policy_learning"

SEED_ROOT = 42

# ============================================================================
# Configuration
# ============================================================================
# DGP: S1 = (x, z) ~ N(0, I2); A1 ~ Bern(sigmoid(0.5 x + 0.5 z));
# S2^(1) = RHO x + TREAT_SHIFT A1 + eta1; S2^(2) = RHO z + eta2,
# eta ~ N(0, SIGMA_ETA^2); A2 ~ Bern(sigmoid(0.5 S2^(1) + 0.5 S2^(2)));
# Y = 1 - 0.5 S2^(1) + 0.4 z + 0.4 S2^(2)
#     + DELTA1 A1 (C_DIR - x) + TAU2 A2 (C2_STAR - S2^(1)) + N(0, SIGMA_Y^2).
DGP_PARAMS = {
    "RHO": 0.6,
    "TREAT_SHIFT": -0.6,
    "SIGMA_ETA": 0.5,
    "SIGMA_Y": 1.0,
    "DELTA1": 0.6,
    "C_DIR": -0.6,
    "TAU2": 0.8,
    "C2_STAR": 0.4,
    "BETA_PROP": 0.5,
}

BEHAVIOR_GH_NODES = 20
PROP_CLIP = 0.01
VALUE_METHOD = "adaptive_quad_split_at_c1_v2"
NUISANCE_SCHEME = "outer_fold_recursive_fqe_v2"

ORACLE_CONFIG = {
    **DGP_PARAMS,
    "BEHAVIOR_GH_NODES": BEHAVIOR_GH_NODES,
    "MC_CHECK": 1_000_000,
    "VALUE_METHOD": VALUE_METHOD,
}
SWEEP_CONFIG = {
    **DGP_PARAMS,
    "N_GRID": [250, 1000, 4000],
    "N_SEEDS": 40,
    "K_FOLDS": 2,
    "PROP_CLIP": PROP_CLIP,
    "VALUE_METHOD": VALUE_METHOD,
    "NUISANCE_SCHEME": NUISANCE_SCHEME,
}
MISSPEC_CONFIG = {
    **DGP_PARAMS,
    "N_MISSPEC": 1000,
    "N_SEEDS": 40,
    "K_FOLDS": 2,
    "PROP_CLIP": PROP_CLIP,
    "VALUE_METHOD": VALUE_METHOD,
    "NUISANCE_SCHEME": NUISANCE_SCHEME,
}

RHO = DGP_PARAMS["RHO"]
TREAT_SHIFT = DGP_PARAMS["TREAT_SHIFT"]
SIGMA_ETA = DGP_PARAMS["SIGMA_ETA"]
SIGMA_Y = DGP_PARAMS["SIGMA_Y"]
DELTA1 = DGP_PARAMS["DELTA1"]
C_DIR = DGP_PARAMS["C_DIR"]
TAU2 = DGP_PARAMS["TAU2"]
C2_STAR = DGP_PARAMS["C2_STAR"]
BETA_PROP = DGP_PARAMS["BETA_PROP"]

METHODS = ["AIPW", "DM", "IPW"]


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


# ============================================================================
# Environment
# ============================================================================
def generate_cohort(n, rng):
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    A1 = (rng.random(n) < sigmoid(BETA_PROP * (x + z))).astype(float)
    s21 = RHO * x + TREAT_SHIFT * A1 + SIGMA_ETA * rng.normal(size=n)
    s22 = RHO * z + SIGMA_ETA * rng.normal(size=n)
    A2 = (rng.random(n) < sigmoid(BETA_PROP * (s21 + s22))).astype(float)
    Y = (
        1.0
        - 0.5 * s21
        + 0.4 * z
        + 0.4 * s22
        + DELTA1 * A1 * (C_DIR - x)
        + TAU2 * A2 * (C2_STAR - s21)
        + SIGMA_Y * rng.normal(size=n)
    )
    return x, z, A1, s21, s22, A2, Y


# ============================================================================
# Oracle: exact value of any threshold pair by adaptive integration
# ============================================================================
def _gh_grid(n_nodes):
    t, w = np.polynomial.hermite.hermgauss(n_nodes)
    x = np.sqrt(2.0) * t
    weights = w / np.sqrt(np.pi)
    return x, weights


def policy_value(c1, c2):
    """E[Y] under threshold rules, with adaptive integration split at c1.

    Splitting is necessary because the stage-1 action jumps at c1. Applying a
    fixed Gauss-Hermite grid across that jump makes the value spuriously flat
    between adjacent nodes.
    """

    def branch_integrand(x, a1):
        mean = RHO * x + TREAT_SHIFT * a1
        u = (c2 - mean) / SIGMA_ETA
        stage2 = TAU2 * (
            (C2_STAR - mean) * norm.cdf(u) + SIGMA_ETA * norm.pdf(u)
        )
        conditional_value = (
            1.0 - 0.5 * mean + DELTA1 * a1 * (C_DIR - x) + stage2
        )
        return conditional_value * norm.pdf(x)

    treated = quad(branch_integrand, -np.inf, c1, args=(1.0,), epsabs=1e-10)[0]
    untreated = quad(branch_integrand, c1, np.inf, args=(0.0,), epsabs=1e-10)[0]
    return float(treated + untreated)


def _stage2_gain(mean, threshold=C2_STAR):
    """Expected stage-2 treatment gain under a threshold rule."""
    u = (threshold - mean) / SIGMA_ETA
    return TAU2 * (
        (C2_STAR - mean) * norm.cdf(u) + SIGMA_ETA * norm.pdf(u)
    )


def stage1_advantage(x):
    """Q_1(x, 1) - Q_1(x, 0) when stage 2 uses its oracle threshold."""
    mean0 = RHO * x
    mean1 = mean0 + TREAT_SHIFT
    return (
        -0.5 * (mean1 - mean0)
        + DELTA1 * (C_DIR - x)
        + _stage2_gain(mean1)
        - _stage2_gain(mean0)
    )


def behavior_value_quadrature(n_nodes):
    """Behavior-policy value by four-dimensional Gauss-Hermite quadrature."""
    nodes, weights = _gh_grid(n_nodes)
    x, z, e1, e2 = np.meshgrid(nodes, nodes, nodes, nodes, indexing="ij")
    wx, wz, w1, w2 = np.meshgrid(weights, weights, weights, weights, indexing="ij")
    joint_w = wx * wz * w1 * w2
    p1 = sigmoid(BETA_PROP * (x + z))
    value = np.zeros_like(x)
    for a1 in (0.0, 1.0):
        prob1 = p1 if a1 == 1.0 else 1.0 - p1
        s21 = RHO * x + TREAT_SHIFT * a1 + SIGMA_ETA * e1
        s22 = RHO * z + SIGMA_ETA * e2
        p2 = sigmoid(BETA_PROP * (s21 + s22))
        base = (
            1.0
            - 0.5 * s21
            + 0.4 * z
            + 0.4 * s22
            + DELTA1 * a1 * (C_DIR - x)
        )
        value += prob1 * (base + p2 * TAU2 * (C2_STAR - s21))
    return float(np.sum(joint_w * value))


def compute_oracle(cfg):
    # Stage 2 treats iff s21 < C2_STAR. The stage-1 advantage is strictly
    # decreasing, so its unique zero is the optimal stage-1 threshold.
    check_grid = np.linspace(-6.0, 6.0, 2001)
    check_adv = stage1_advantage(check_grid)
    if not np.all(np.diff(check_adv) < 0.0):
        raise RuntimeError("Stage-1 oracle advantage is not strictly decreasing")
    c1_star = float(brentq(stage1_advantage, -6.0, 6.0))
    V_star = policy_value(c1_star, C2_STAR)

    # MC cross-check of the quadrature pipeline at (c1*, c2*).
    rng = np.random.default_rng(np.random.SeedSequence([SEED_ROOT, 900]))
    M = cfg["MC_CHECK"]
    x = rng.normal(size=M)
    a1 = (x < c1_star).astype(float)
    s21 = RHO * x + TREAT_SHIFT * a1 + SIGMA_ETA * rng.normal(size=M)
    a2 = (s21 < C2_STAR).astype(float)
    # z, s22, and noise are mean-zero in E[Y]; simulate them anyway.
    z = rng.normal(size=M)
    s22 = RHO * z + SIGMA_ETA * rng.normal(size=M)
    Y = (
        1.0
        - 0.5 * s21
        + 0.4 * z
        + 0.4 * s22
        + DELTA1 * a1 * (C_DIR - x)
        + TAU2 * a2 * (C2_STAR - s21)
        + SIGMA_Y * rng.normal(size=M)
    )
    mc_mean = float(Y.mean())
    mc_se = float(Y.std(ddof=1) / np.sqrt(M))

    # Behavior-policy value by MC.
    rng_b = np.random.default_rng(np.random.SeedSequence([SEED_ROOT, 901]))
    xb, zb, A1b, s21b, s22b, A2b, Yb = generate_cohort(M, rng_b)
    V_beh = float(Yb.mean())
    V_beh_se = float(Yb.std(ddof=1) / np.sqrt(M))
    V_beh_quad = behavior_value_quadrature(cfg["BEHAVIOR_GH_NODES"])

    # Independent numerical checks guard the two quantities against which all
    # learned policies are scored. The Monte Carlo checks use their own seeds
    # and are not reused anywhere in estimation.
    if abs(mc_mean - V_star) > 4.0 * mc_se:
        raise RuntimeError("Oracle quadrature failed its independent MC check")
    if abs(V_beh - V_beh_quad) > 4.0 * V_beh_se:
        raise RuntimeError("Behavior quadrature failed its independent MC check")
    for dc1, dc2 in ((-0.10, 0.0), (0.10, 0.0), (0.0, -0.10), (0.0, 0.10)):
        if policy_value(c1_star + dc1, C2_STAR + dc2) > V_star + 1e-10:
            raise RuntimeError("Reported oracle is not locally optimal")

    treated1 = float(norm.cdf(c1_star))
    print(f"    c1* = {c1_star:.4f}, c2* = {C2_STAR:.4f}")
    print(f"    V* = {V_star:.6f} (adaptive quadrature split at c1*)")
    print(
        f"    MC check at (c1*, c2*): {mc_mean:.6f} (MC SE {mc_se:.6f}), "
        f"|diff|/MC SE = {abs(mc_mean - V_star) / mc_se:.2f}"
    )
    print(
        f"    V(behavior) = {V_beh:.6f} (MC SE {V_beh_se:.6f}); "
        f"quadrature = {V_beh_quad:.6f}, "
        f"|diff|/MC SE = {abs(V_beh - V_beh_quad) / V_beh_se:.2f}"
    )
    print(f"    Stage-1 treated share under c1*: {treated1:.3f}")
    return {
        "c1_star": c1_star,
        "c2_star": C2_STAR,
        "V_star": V_star,
        "mc_mean": mc_mean,
        "mc_se": mc_se,
        "V_beh": V_beh,
        "V_beh_se": V_beh_se,
        "V_beh_quad": V_beh_quad,
    }


# ============================================================================
# Nuisances (cross-fitted) and the exact threshold search
# ============================================================================
def fit_logistic(X, y, iters=25):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ b)
        W = p * (1.0 - p) + 1e-6
        grad = X.T @ (y - p)
        H = (X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        b = b + np.linalg.solve(H, grad)
    return b


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def q2_basis(x, z, A1, s21, s22, A2, correct):
    cols = [np.ones_like(x), x, z, A1, s21, s22, A2]
    if correct:
        cols += [A1 * x, A2 * s21]
    return np.column_stack(cols)


def q1_basis(x, z, A1, correct, c2):
    cols = [np.ones_like(x), x, z, A1]
    if correct:
        mean = RHO * x + TREAT_SHIFT * A1
        u = (c2 - mean) / SIGMA_ETA
        prob_below = norm.cdf(u)
        truncated_first_moment = mean * prob_below - SIGMA_ETA * norm.pdf(u)
        cols += [A1 * x, prob_below, truncated_first_moment]
    return np.column_stack(cols)


def best_threshold(xvals, gains):
    """Exact argmax_c of mean[ 1{x < c} g ] over all thresholds: sort by x,
    prefix maximum of the cumulative gains."""
    order = np.argsort(xvals)
    xs = xvals[order]
    cum = np.concatenate([[0.0], np.cumsum(gains[order])])
    j = int(np.argmax(cum))
    if j == 0:
        return float(xs[0] - 1.0)
    if j == len(xs):
        return float(xs[-1] + 1.0)
    return float(0.5 * (xs[j - 1] + xs[j]))


def check_threshold_search():
    """Compare the O(n log n) search with exhaustive prefix enumeration."""
    rng = np.random.default_rng(20240724)
    for n in (2, 7, 31):
        x = rng.normal(size=n)
        g = rng.normal(size=n)
        c = best_threshold(x, g)
        fast = np.sum(g[x < c])
        order = np.argsort(x)
        exhaustive = np.max(np.concatenate([[0.0], np.cumsum(g[order])]))
        if not np.isclose(fast, exhaustive, atol=1e-12, rtol=1e-12):
            raise RuntimeError("Exact threshold search failed exhaustive check")


def run_rep(n, rng, k_folds, q_correct, e_correct):
    """One replication: generate a cohort, cross-fit nuisances, learn
    (c1, c2) with each method. Returns {method: (c1, c2)}."""
    x, z, A1, s21, s22, A2, Y = generate_cohort(n, rng)
    fold = rng.permutation(n) % k_folds

    # Cross-fitted nuisance predictions for every subject.
    q2_hat_obs = np.zeros(n)  # Q2 at the observed A2
    q2_hat_a = np.zeros((n, 2))  # Q2 at A2 = 0, 1
    e2_hat = np.zeros(n)  # P(A2 = 1 | H2)
    e1_hat = np.zeros(n)  # P(A1 = 1 | H1)
    beta2_by_fold = {}
    for k in range(k_folds):
        tr = fold != k
        te = fold == k
        beta2 = ols(
            q2_basis(x[tr], z[tr], A1[tr], s21[tr], s22[tr], A2[tr], q_correct),
            Y[tr],
        )
        beta2_by_fold[k] = beta2
        q2_hat_obs[te] = (
            q2_basis(x[te], z[te], A1[te], s21[te], s22[te], A2[te], q_correct) @ beta2
        )
        for a in (0, 1):
            a_vec = np.full(te.sum(), float(a))
            q2_hat_a[te, a] = (
                q2_basis(x[te], z[te], A1[te], s21[te], s22[te], a_vec, q_correct)
                @ beta2
            )
        if e_correct:
            g2 = fit_logistic(
                np.column_stack([np.ones(tr.sum()), s21[tr], s22[tr]]), A2[tr]
            )
            e2_hat[te] = sigmoid(
                np.column_stack([np.ones(te.sum()), s21[te], s22[te]]) @ g2
            )
            g1 = fit_logistic(
                np.column_stack([np.ones(tr.sum()), x[tr], z[tr]]), A1[tr]
            )
            e1_hat[te] = sigmoid(
                np.column_stack([np.ones(te.sum()), x[te], z[te]]) @ g1
            )
        else:
            e2_hat[te] = A2[tr].mean()
            e1_hat[te] = A1[tr].mean()
    e2_hat = np.clip(e2_hat, PROP_CLIP, 1.0 - PROP_CLIP)
    e1_hat = np.clip(e1_hat, PROP_CLIP, 1.0 - PROP_CLIP)

    def gamma2(a_col):
        """Stage-2 score vector Gamma_2(a) for a in {0, 1}, per method."""
        prob_a = e2_hat if a_col == 1 else 1.0 - e2_hat
        match = (A2 == a_col).astype(float)
        resid = match / prob_a * (Y - q2_hat_obs)
        return {
            "AIPW": q2_hat_a[:, a_col] + resid,
            "DM": q2_hat_a[:, a_col],
            "IPW": match / prob_a * Y,
        }

    g2_by_a = {a: gamma2(a) for a in (0, 1)}
    out = {}
    for meth in METHODS:
        # Stage 2: exact threshold search on the stage-2 gains.
        gains2 = (g2_by_a[1][meth] - g2_by_a[0][meth]) / n
        c2_hat = best_threshold(s21, gains2)
        pi2 = (s21 < c2_hat).astype(int)
        pi2_score = np.where(pi2 == 1, g2_by_a[1][meth], g2_by_a[0][meth])

        if meth == "IPW":
            match2 = (A2 == pi2).astype(float)
            prob2 = np.where(pi2 == 1, e2_hat, 1.0 - e2_hat)
            w2Y = match2 / prob2 * Y
            g1 = {}
            for a in (0, 1):
                prob1 = e1_hat if a == 1 else 1.0 - e1_hat
                match1 = (A1 == a).astype(float)
                g1[a] = match1 / prob1 * w2Y
        else:
            # Fitted-Q evaluation for the continuation under pi2. For each
            # outer fold, both Q2 and Q1 are fitted without the scoring fold.
            # Reusing the globally cross-fitted q2_hat_a as the Q1 training
            # target would leak the scoring fold: with two folds, q2_hat_a on
            # the Q1 training observations was fitted on the Q1 test fold.
            q1_hat_a = np.zeros((n, 2))
            q1_hat_obs = np.zeros(n)
            for k in range(k_folds):
                tr = fold != k
                te = fold == k
                beta2_outer = beta2_by_fold[k]
                target_tr = (
                    q2_basis(
                        x[tr],
                        z[tr],
                        A1[tr],
                        s21[tr],
                        s22[tr],
                        pi2[tr].astype(float),
                        q_correct,
                    )
                    @ beta2_outer
                )
                beta1 = ols(
                    q1_basis(x[tr], z[tr], A1[tr], q_correct, c2_hat),
                    target_tr,
                )
                q1_hat_obs[te] = (
                    q1_basis(x[te], z[te], A1[te], q_correct, c2_hat) @ beta1
                )
                for a in (0, 1):
                    a_vec = np.full(te.sum(), float(a))
                    q1_hat_a[te, a] = (
                        q1_basis(x[te], z[te], a_vec, q_correct, c2_hat) @ beta1
                    )
            if meth == "DM":
                g1 = {a: q1_hat_a[:, a] for a in (0, 1)}
            else:  # AIPW
                v_tilde = pi2_score  # AIPW-corrected continuation value
                g1 = {}
                for a in (0, 1):
                    prob1 = e1_hat if a == 1 else 1.0 - e1_hat
                    match1 = (A1 == a).astype(float)
                    g1[a] = q1_hat_a[:, a] + match1 / prob1 * (v_tilde - q1_hat_obs)
        gains1 = (g1[1] - g1[0]) / n
        c1_hat = best_threshold(x, gains1)
        out[meth] = (c1_hat, c2_hat)
    return out


# ============================================================================
# Experiments
# ============================================================================
def run_sweep_n(cfg):
    n_grid = cfg["N_GRID"]
    n_seeds = cfg["N_SEEDS"]
    regret = {m: np.zeros((len(n_grid), n_seeds)) for m in METHODS}
    thresholds = {m: np.zeros((len(n_grid), n_seeds, 2)) for m in METHODS}
    # V* recomputed here (cheap) so the component is self-contained.
    c1_star = float(brentq(stage1_advantage, -6.0, 6.0))
    V_star = policy_value(c1_star, C2_STAR)
    for i, n in enumerate(n_grid):
        for rep in range(n_seeds):
            rng = np.random.default_rng(
                np.random.SeedSequence([SEED_ROOT, 100 + i, rep])
            )
            res = run_rep(n, rng, cfg["K_FOLDS"], True, True)
            for m in METHODS:
                c1_hat, c2_hat = res[m]
                v = policy_value(c1_hat, c2_hat)
                regret[m][i, rep] = V_star - v
                thresholds[m][i, rep] = (c1_hat, c2_hat)
        print(
            f"    n={n}: "
            + ", ".join(f"{m} regret {regret[m][i].mean():.4f}" for m in METHODS)
        )
    return {
        "regret": regret,
        "thresholds": thresholds,
        "N_grid": list(n_grid),
        "V_star": V_star,
    }


def run_misspec(cfg):
    n = cfg["N_MISSPEC"]
    n_seeds = cfg["N_SEEDS"]
    regimes = [
        ("both correct", True, True),
        ("Q wrong", False, True),
        ("e wrong", True, False),
        ("both wrong", False, False),
    ]
    c1_star = float(brentq(stage1_advantage, -6.0, 6.0))
    V_star = policy_value(c1_star, C2_STAR)
    regret = {m: np.zeros((len(regimes), n_seeds)) for m in METHODS}
    for r, (label, q_ok, e_ok) in enumerate(regimes):
        for rep in range(n_seeds):
            rng = np.random.default_rng(
                np.random.SeedSequence([SEED_ROOT, 300 + r, rep])
            )
            res = run_rep(n, rng, cfg["K_FOLDS"], q_ok, e_ok)
            for m in METHODS:
                v = policy_value(*res[m])
                regret[m][r, rep] = V_star - v
        print(
            f"    {label}: "
            + ", ".join(f"{m} regret {regret[m][r].mean():.4f}" for m in METHODS)
        )
    return {
        "regret": regret,
        "regime_labels": [r[0] for r in regimes],
        "V_star": V_star,
    }


def compute_data(force=None):
    force = force or set()
    oracle = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "oracle",
        ORACLE_CONFIG,
        compute_oracle,
        ORACLE_CONFIG,
        force=("oracle" in force),
    )
    sweep_n = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "sweep_n",
        SWEEP_CONFIG,
        run_sweep_n,
        SWEEP_CONFIG,
        force=("sweep_n" in force),
    )
    misspec = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "misspec",
        MISSPEC_CONFIG,
        run_misspec,
        MISSPEC_CONFIG,
        force=("misspec" in force),
    )
    return {"oracle": oracle, "sweep_n": sweep_n, "misspec": misspec}


# ============================================================================
# Outputs
# ============================================================================
def validate_results(data):
    """Hard gates for the oracle, robustness ablation, and learning signal."""
    oracle = data["oracle"]
    sweep = data["sweep_n"]
    misspec = data["misspec"]
    if abs(oracle["mc_mean"] - oracle["V_star"]) > 4.0 * oracle["mc_se"]:
        raise RuntimeError("Oracle failed its independent MC check")
    if abs(oracle["V_beh"] - oracle["V_beh_quad"]) > 4.0 * oracle["V_beh_se"]:
        raise RuntimeError("Behavior value failed its quadrature/MC check")
    if oracle["V_star"] - oracle["V_beh_quad"] < 0.50:
        raise RuntimeError("Policy-learning DGP has no material welfare signal")

    for method in METHODS:
        r = sweep["regret"][method]
        if not np.all(np.isfinite(r)) or np.min(r) < -1e-8:
            raise RuntimeError(f"Invalid regret draws for {method}")
        if r[-1].mean() >= r[0].mean():
            raise RuntimeError(f"{method} does not improve from smallest to largest n")

    labels = misspec["regime_labels"]
    idx = {name: labels.index(name) for name in labels}
    r = misspec["regret"]
    if max(r["AIPW"][idx["Q wrong"]].mean(), r["AIPW"][idx["e wrong"]].mean()) > 0.05:
        raise RuntimeError("AIPW failed a one-sided nuisance misspecification check")
    if r["DM"][idx["Q wrong"]].mean() < 5.0 * r["DM"][idx["both correct"]].mean():
        raise RuntimeError("Outcome misspecification demonstration is too weak")
    if r["IPW"][idx["e wrong"]].mean() < 2.0 * r["IPW"][idx["both correct"]].mean():
        raise RuntimeError("Propensity misspecification demonstration is too weak")


METHOD_COLORS = {
    "AIPW": COLORS["blue"],
    "DM": COLORS["orange"],
    "IPW": COLORS["red"],
}
METHOD_LABELS = {
    "AIPW": "Backward AIPW",
    "DM": "Plug-in $Q$",
    "IPW": "IPW",
}


def generate_outputs(data):
    oracle = data["oracle"]
    sweep = data["sweep_n"]
    misspec = data["misspec"]
    V_star = oracle["V_star"]
    n_grid = np.array(sweep["N_grid"])
    n_seeds = SWEEP_CONFIG["N_SEEDS"]

    print()
    print("=" * 72)
    print("  Doubly robust backward-induction offline policy learning")
    print("=" * 72)
    print()
    print(
        f"  Oracle: c1* = {oracle['c1_star']:.4f}, c2* = {oracle['c2_star']:.4f}, "
        f"V* = {V_star:.6f}"
    )
    print(
        f"  Quadrature vs MC: |diff|/MC SE = "
        f"{abs(oracle['mc_mean'] - V_star) / oracle['mc_se']:.2f}"
    )
    print(
        f"  Behavior quadrature vs MC: |diff|/MC SE = "
        f"{abs(oracle['V_beh'] - oracle['V_beh_quad']) / oracle['V_beh_se']:.2f}"
    )
    print(
        f"  V(behavior) = {oracle['V_beh']:.6f}; "
        f"behavior regret = {V_star - oracle['V_beh']:.4f}"
    )
    print()
    print(f"  Regret V* - V(pi_hat), correct nuisances, {n_seeds} seeds:")
    print(f"  {'n':>6}  " + "".join(f"{METHOD_LABELS[m]:>22}" for m in METHODS))
    for i, n in enumerate(n_grid):
        row = f"  {n:>6d}  "
        for m in METHODS:
            mean = sweep["regret"][m][i].mean()
            se = sweep["regret"][m][i].std(ddof=1) / np.sqrt(n_seeds)
            row += f"{mean:>13.4f} ({se:.4f})"
        print(row)
    paired = sweep["regret"]["AIPW"][-1] - sweep["regret"]["DM"][-1]
    print(
        "  Paired seed-wise regret contrast at largest n "
        "(AIPW - plug-in Q): "
        f"{paired.mean():+.4f} "
        f"(MC SE {paired.std(ddof=1) / np.sqrt(n_seeds):.4f})"
    )
    print()
    slopes = {}
    for m in METHODS:
        means = sweep["regret"][m].mean(axis=1)
        slope = np.polyfit(np.log(n_grid), np.log(means), 1)[0]
        slopes[m] = slope
        print(f"  log-log regret slope, {METHOD_LABELS[m]}: {slope:.2f}")
    print()
    print(
        f"  Misspecification design at n = {MISSPEC_CONFIG['N_MISSPEC']}, "
        f"{MISSPEC_CONFIG['N_SEEDS']} seeds (mean regret):"
    )
    print(f"  {'regime':>14}  " + "".join(f"{m:>10}" for m in METHODS))
    for r, label in enumerate(misspec["regime_labels"]):
        row = f"  {label:>14}  "
        for m in METHODS:
            row += f"{misspec['regret'][m][r].mean():>10.4f}"
        print(row)
    print()

    # ---- Figure: 1 x 3 panels ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel (a): regret vs n, log-log
    ax = axes[0]
    for m in METHODS:
        means = sweep["regret"][m].mean(axis=1)
        ses = sweep["regret"][m].std(axis=1, ddof=1) / np.sqrt(n_seeds)
        ax.errorbar(
            n_grid,
            means,
            yerr=1.96 * ses,
            marker="o",
            label=METHOD_LABELS[m],
            color=METHOD_COLORS[m],
            capsize=3,
        )
    ref = sweep["regret"]["AIPW"].mean(axis=1)[0] * np.sqrt(n_grid[0] / n_grid)
    ax.plot(n_grid, ref, **BENCH_STYLE, label=r"$n^{-1/2}$ rate")
    ax.axhline(
        V_star - oracle["V_beh"],
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.0,
        label="Behavior policy",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Trajectories $n$")
    ax.set_ylabel(r"Regret $V^* - V(\hat\pi)$")
    ax.set_title("(a) Regret vs sample size")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    # Panel (b): misspecification design, dot-and-whisker by regime
    ax = axes[1]
    n_reg = len(misspec["regime_labels"])
    offsets = {"AIPW": -0.18, "DM": 0.0, "IPW": 0.18}
    for m in METHODS:
        means = misspec["regret"][m].mean(axis=1)
        ses = misspec["regret"][m].std(axis=1, ddof=1) / np.sqrt(
            MISSPEC_CONFIG["N_SEEDS"]
        )
        xs = np.arange(n_reg) + offsets[m]
        ax.errorbar(
            xs,
            means,
            yerr=1.96 * ses,
            fmt="o",
            color=METHOD_COLORS[m],
            label=METHOD_LABELS[m],
            capsize=3,
        )
    ax.set_xticks(np.arange(n_reg))
    ax.set_xticklabels(misspec["regime_labels"], fontsize=8)
    ax.set_ylabel(r"Regret $V^* - V(\hat\pi)$")
    ax.set_title(f"(b) Nuisance misspecification, $n={MISSPEC_CONFIG['N_MISSPEC']}$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel (c): learned thresholds converge on the oracle pair (AIPW)
    ax = axes[2]
    th = sweep["thresholds"]["AIPW"]
    i_lo, i_hi = 0, len(n_grid) - 1
    ax.scatter(
        th[i_lo, :, 0],
        th[i_lo, :, 1],
        s=18,
        alpha=0.5,
        color=COLORS["orange"],
        label=f"$n={n_grid[i_lo]}$",
    )
    ax.scatter(
        th[i_hi, :, 0],
        th[i_hi, :, 1],
        s=18,
        alpha=0.7,
        color=COLORS["blue"],
        label=f"$n={n_grid[i_hi]}$",
    )
    ax.axvline(oracle["c1_star"], **BENCH_STYLE)
    ax.axhline(oracle["c2_star"], **BENCH_STYLE)
    ax.set_xlabel(r"$\hat c_1$")
    ax.set_ylabel(r"$\hat c_2$")
    ax.set_title(r"(c) Backward-AIPW thresholds vs $(c_1^*, c_2^*)$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "dtr_policy_learning.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {fig_path}")

    # ---- LaTeX results table: rank-ordered V(pi_hat)/V* at largest n ----
    i_hi = len(n_grid) - 1
    rows = [("Oracle regime $(c_1^*, c_2^*)$", 1.0, None)]
    for m in METHODS:
        vals = (V_star - sweep["regret"][m][i_hi]) / V_star
        rows.append(
            (
                f"{METHOD_LABELS[m]} ($n={n_grid[i_hi]}$)",
                vals.mean(),
                vals.std(ddof=1) / np.sqrt(n_seeds),
            )
        )
    rows.append(("Behavior policy", oracle["V_beh"] / V_star, None))
    rows.sort(key=lambda r: -r[1])
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & $V(\hat\pi) / V^*$ & MC SE \\",
        r"\midrule",
    ]
    for name, mean, se in rows:
        se_str = "({:.4f})".format(se) if se is not None else "--"
        lines.append("{} & {:.4f} & {} \\\\".format(name, mean, se_str))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    tab_path = os.path.join(OUTPUT_DIR, "dtr_policy_learning_results.tex")
    with open(tab_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Table:  {tab_path}")


def main():
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print("Config:")
    print(
        f"  DGP: rho={RHO}, treat_shift={TREAT_SHIFT}, sigma_eta={SIGMA_ETA}, "
        f"sigma_Y={SIGMA_Y}, delta1={DELTA1}, c_dir={C_DIR}, tau2={TAU2}, "
        f"c2*={C2_STAR}"
    )
    print(
        f"  Sweep: N_GRID={SWEEP_CONFIG['N_GRID']}, "
        f"N_SEEDS={SWEEP_CONFIG['N_SEEDS']}, K_FOLDS={SWEEP_CONFIG['K_FOLDS']}, "
        f"prop_clip={PROP_CLIP}"
    )
    print(
        f"  Misspec: n={MISSPEC_CONFIG['N_MISSPEC']}, "
        f"N_SEEDS={MISSPEC_CONFIG['N_SEEDS']}"
    )
    if force:
        print(f"  forcing recompute of: {sorted(force)}")

    check_threshold_search()
    if args.plots_only:
        data = compute_data()
    else:
        data = compute_data(force=force)
    validate_results(data)
    if not args.data_only:
        generate_outputs(data)


if __name__ == "__main__":
    main()
