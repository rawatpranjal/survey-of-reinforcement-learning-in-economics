# Off-policy evaluation estimators on a tabular customer-retention MDP.
# ch10b_rl_for_ci, Off-Policy Evaluation and Dynamic Treatment Effects.
#
# Monte Carlo study of the OPE estimator canon on a 5-state promotional
# targeting MDP with an exact dynamic-programming oracle. Three experiments:
# (a) sample-size sweep at H=16 (ground-truth recovery; bias/SD/RMSE),
# (b) horizon sweep at n=500 (curse of horizon; marginalized escape),
# (c) 2x2 nuisance-misspecification ablation at H=16, n=500 (double
#     robustness of DR: Q-model wrong / propensity wrong / both / neither).
#
# Estimators (references verified term-by-term against the papers in
# ch10b_rl_for_ci/papers/):
#   DM       model-based direct method: tabular MLE (P-hat, r-hat) + exact DP,
#            equal to tabular FQE at the MLE (Jiang & Li 2016, Eqns. 2-3).
#   IS       trajectory-wise importance sampling (Jiang & Li 2016, Eqn. 4).
#   PDIS     per-decision (step-wise) IS (Jiang & Li 2016, Eqn. 5;
#            Precup, Sutton & Singh 2000).
#   WIS      step-wise weighted IS, w_t = mean cumulative ratio
#            (Jiang & Li 2016, Eqn. 7).
#   DR       sequential doubly robust (Jiang & Li 2016, Eqn. 10), computed in
#            the equivalent non-recursive form of Thomas & Brunskill 2016
#            (Eqn. 2), with 2-fold cross-fitted Q-hat from the MLE model.
#   WDR      weighted doubly robust: same score with self-normalized weights
#            (Thomas & Brunskill 2016); under 2-fold cross-fitting the
#            normalization runs within each scoring fold, a consistent and
#            asymptotically equivalent variant of the full-dataset weights
#            w^i_t = rho^i_{1:t} / sum_j rho^j_{1:t} (Thomas & Brunskill 2016).
#   MIS      marginalized IS (Xie, Ma & Wang 2019, Eqns. 3.1-3.2): empirical
#            d-hat^{pi_b}_t, recursive d-hat^{pi_e}_t, per-state reward IS.
# MAGIC and DualDICE are discussed in the chapter prose only; implementing
# their blending / minimax machinery adds code risk with no extra qualitative
# point on a tabular oracle problem (explicit non-goal). High-confidence lower
# bounds are likewise out of scope here (they belong to the prose).
#
# Caching deviates from the ALGO_REGISTRY pattern: the cached unit is the
# EXPERIMENT (shared / sweep_n / sweep_h / ablation), not the algorithm,
# because all estimators are cheap and comparison fairness requires that they
# score the same logged datasets within each replication.

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, BENCH_STYLE, FIG_TRIPLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

apply_style()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "ope_estimators"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_PARAMS = {
    "N_STATES": 5,  # engagement level 0 (near-churn) .. 4 (loyal)
    "N_ACTIONS": 2,  # 0 = hold, 1 = promote
    "UP_HOLD": 0.15,
    "DOWN_HOLD": 0.30,  # a=0: up/stay/down
    "UP_PROMO": 0.45,
    "DOWN_PROMO": 0.15,  # a=1: up/stay/down
    "MARGIN": 0.25,
    "PROMO_COST": 0.5,
    "COST_OFFSET": 0.1,
    "INIT_STATES": (1, 2, 3),  # uniform initial distribution
    "BEHAVIOR_BASE": 0.6,
    "BEHAVIOR_SLOPE": 0.1,  # pi_b(1|s) = 0.6 - 0.1 s
    "EVAL_LOW": 0.9,
    "EVAL_HIGH": 0.1,
    "EVAL_CUT": 1,  # pi_e(1|s)
}

SHARED_CONFIG = {
    **ENV_PARAMS,
    "H_GRID": (4, 8, 16, 32, 64),
    "MC_CHECK_EPISODES": 100_000,
    "MC_CHECK_H": 16,
    "SEED_ROOT": 42,
}
N_REPS = 500
SWEEP_N_CONFIG = {
    **SHARED_CONFIG,
    "N_REPS": N_REPS,
    "H": 16,
    "N_GRID": (50, 100, 200, 500, 1000, 2000),
}
SWEEP_H_CONFIG = {
    **SHARED_CONFIG,
    "N_REPS": N_REPS,
    "N_EPISODES": 500,
    "H_SWEEP": (4, 8, 16, 32, 64),
}
ABLATION_CONFIG = {
    **SHARED_CONFIG,
    "N_REPS": N_REPS,
    "H": 16,
    "N_EPISODES": 500,
    "WRONG_PROPENSITY": 0.5,
}

ESTIMATORS = ["DM", "IS", "PDIS", "WIS", "DR", "WDR", "MIS"]
EST_COLORS = {
    "DM": COLORS["blue"],
    "IS": COLORS["red"],
    "PDIS": COLORS["orange"],
    "WIS": COLORS["brown"],
    "DR": COLORS["green"],
    "WDR": COLORS["olive"],
    "MIS": COLORS["purple"],
}

# ---------------------------------------------------------------------------
# Environment: promotional targeting MDP (tabular, finite horizon, gamma = 1)
# ---------------------------------------------------------------------------


def build_env(p):
    """Return (P, r, init_dist, pi_b, pi_e) as dense arrays."""
    S, A = p["N_STATES"], p["N_ACTIONS"]
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            up = p["UP_PROMO"] if a == 1 else p["UP_HOLD"]
            down = p["DOWN_PROMO"] if a == 1 else p["DOWN_HOLD"]
            stay = 1.0 - up - down
            # clip at boundaries, folding excess mass into "stay"
            if s == S - 1:
                stay += up
                up = 0.0
            if s == 0:
                stay += down
                down = 0.0
            if s + 1 < S:
                P[s, a, s + 1] = up
            P[s, a, s] = stay
            if s - 1 >= 0:
                P[s, a, s - 1] = down
    r = np.zeros((S, A))
    for s in range(S):
        r[s, 0] = p["MARGIN"] * s
        r[s, 1] = p["MARGIN"] * s - p["PROMO_COST"] * (1.0 - p["COST_OFFSET"] * s)
    init_dist = np.zeros(S)
    for s in p["INIT_STATES"]:
        init_dist[s] = 1.0 / len(p["INIT_STATES"])
    pi_b = np.zeros((S, A))
    pi_b[:, 1] = p["BEHAVIOR_BASE"] - p["BEHAVIOR_SLOPE"] * np.arange(S)
    pi_b[:, 0] = 1.0 - pi_b[:, 1]
    pi_e = np.zeros((S, A))
    pi_e[:, 1] = np.where(np.arange(S) <= p["EVAL_CUT"], p["EVAL_LOW"], p["EVAL_HIGH"])
    pi_e[:, 0] = 1.0 - pi_e[:, 1]
    return P, r, init_dist, pi_b, pi_e


def dp_value(P, r, init_dist, pi, H):
    """Exact finite-horizon policy value and per-stage (Q_t, V_t) under pi."""
    S, A = r.shape
    Q = np.zeros((H + 1, S, A))  # Q[t] for t = 1..H at index t-1; Q[H] = 0 pad
    V = np.zeros((H + 1, S))
    for t in range(H - 1, -1, -1):
        Q[t] = r + P @ V[t + 1]
        V[t] = (pi * Q[t]).sum(axis=1)
    return float(init_dist @ V[0]), Q[:H], V[: H + 1]


def state_marginals(P, init_dist, pi, H):
    """Exact d_t(s) for t = 1..H under pi (forward recursion)."""
    S = init_dist.shape[0]
    d = np.zeros((H, S))
    d[0] = init_dist
    P_pi = np.einsum("sa,sax->sx", pi, P)
    for t in range(1, H):
        d[t] = d[t - 1] @ P_pi
    return d


def simulate(P, r, init_dist, pi, H, n, rng):
    """Vectorized simulation of n trajectories under policy pi.

    Returns S_arr (n,H), A_arr (n,H), R_arr (n,H) with deterministic rewards
    r(s,a); all stochasticity is in actions and transitions.
    """
    S_n = init_dist.shape[0]
    S_arr = np.zeros((n, H), dtype=np.int64)
    A_arr = np.zeros((n, H), dtype=np.int64)
    R_arr = np.zeros((n, H))
    P_cum = np.cumsum(P, axis=2)  # (S, A, S)
    pi_cum = np.cumsum(pi, axis=1)  # (S, A)
    s = rng.choice(S_n, size=n, p=init_dist)
    for t in range(H):
        S_arr[:, t] = s
        u = rng.random(n)
        a = (u[:, None] > pi_cum[s]).sum(axis=1)
        A_arr[:, t] = a
        R_arr[:, t] = r[s, a]
        u2 = rng.random(n)
        s = (u2[:, None] > P_cum[s, a]).sum(axis=1)
    return S_arr, A_arr, R_arr


# ---------------------------------------------------------------------------
# Estimators (all operate on the same logged dataset)
# ---------------------------------------------------------------------------


def step_ratios(S_arr, A_arr, pi_e, pi_b_used):
    """Per-step ratios rho_t = pi_e(a_t|s_t) / pi_b(a_t|s_t), shape (n, H)."""
    return pi_e[S_arr, A_arr] / pi_b_used[S_arr, A_arr]


def fit_mle_model(S_arr, A_arr, R_arr, n_states, n_actions):
    """Tabular MLE (P-hat, r-hat) from logged transitions.

    Unvisited (s,a): uniform next-state distribution and reward 0.
    """
    n, H = S_arr.shape
    trans = np.zeros((n_states, n_actions, n_states))
    rsum = np.zeros((n_states, n_actions))
    cnt = np.zeros((n_states, n_actions))
    flat_sa = S_arr * n_actions + A_arr
    np.add.at(cnt.reshape(-1), flat_sa.ravel(), 1.0)
    np.add.at(rsum.reshape(-1), flat_sa.ravel(), R_arr.ravel())
    # transitions observed for t = 1..H-1 (last state has no successor logged)
    src = flat_sa[:, :-1].ravel()
    dst = S_arr[:, 1:].ravel()
    np.add.at(trans.reshape(n_states * n_actions, n_states), (src, dst), 1.0)
    tcnt = trans.sum(axis=2)
    P_hat = np.where(
        tcnt[:, :, None] > 0, trans / np.maximum(tcnt, 1)[:, :, None], 1.0 / n_states
    )
    r_hat = np.where(cnt > 0, rsum / np.maximum(cnt, 1), 0.0)
    return P_hat, r_hat


def fit_pooled_model(S_arr, A_arr, R_arr, n_states, n_actions):
    """Action-pooled (misspecified) model: P-hat(s'|s), r-hat(s), replicated
    across actions. A genuine projection that erases the treatment channel."""
    n, H = S_arr.shape
    trans = np.zeros((n_states, n_states))
    rsum = np.zeros(n_states)
    cnt = np.zeros(n_states)
    np.add.at(cnt, S_arr.ravel(), 1.0)
    np.add.at(rsum, S_arr.ravel(), R_arr.ravel())
    np.add.at(trans, (S_arr[:, :-1].ravel(), S_arr[:, 1:].ravel()), 1.0)
    tcnt = trans.sum(axis=1)
    P_s = np.where(
        tcnt[:, None] > 0, trans / np.maximum(tcnt, 1)[:, None], 1.0 / n_states
    )
    r_s = np.where(cnt > 0, rsum / np.maximum(cnt, 1), 0.0)
    P_hat = np.repeat(P_s[:, None, :], n_actions, axis=1)
    r_hat = np.repeat(r_s[:, None], n_actions, axis=1)
    return P_hat, r_hat


def model_q_functions(P_hat, r_hat, pi_e, H):
    """Per-stage (Q-hat_t, V-hat_t) under pi_e in the fitted model."""
    S, A = r_hat.shape
    Q = np.zeros((H, S, A))
    V = np.zeros((H + 1, S))
    for t in range(H - 1, -1, -1):
        Q[t] = r_hat + P_hat @ V[t + 1]
        V[t] = (pi_e * Q[t]).sum(axis=1)
    return Q, V


def est_dm(S_arr, V_hat):
    """Direct method: mean of V-hat_1 over logged initial states."""
    return float(V_hat[0][S_arr[:, 0]].mean())


def est_is(rho, R_arr):
    return float((rho.prod(axis=1) * R_arr.sum(axis=1)).mean())


def est_pdis(rho, R_arr):
    return float((np.cumprod(rho, axis=1) * R_arr).sum(axis=1).mean())


def est_wis(rho, R_arr):
    cum = np.cumprod(rho, axis=1)
    w = cum.mean(axis=0)  # w_t = mean cumulative ratio
    return float((cum / np.maximum(w, 1e-300) * R_arr).sum(axis=1).mean())


def dr_score(S_arr, A_arr, R_arr, rho, Q_hat, V_hat, weighted):
    """DR / WDR via the Thomas-Brunskill non-recursive form (their Eqn. 2).

    With w^i_t = rho^i_{1:t}/n this equals the Jiang-Li recursion (Eqn. 10);
    with self-normalized weights it is WDR. Q_hat/V_hat are per-stage arrays
    indexed [t][s(,a)].
    """
    n, H = R_arr.shape
    cum = np.cumprod(rho, axis=1)  # rho_{1:t}, t = 1..H
    if weighted:
        denom = cum.sum(axis=0)
        W = cum / np.maximum(denom, 1e-300)  # w^i_t, sums to 1 over i
        W_prev = np.concatenate([np.full((n, 1), 1.0 / n), W[:, :-1]], axis=1)
    else:
        W = cum / n
        W_prev = np.concatenate([np.full((n, 1), 1.0 / n), W[:, :-1]], axis=1)
    q_terms = np.stack([Q_hat[t][S_arr[:, t], A_arr[:, t]] for t in range(H)], axis=1)
    v_terms = np.stack([V_hat[t][S_arr[:, t]] for t in range(H)], axis=1)
    total = (W * R_arr).sum() - ((W * q_terms) - (W_prev * v_terms)).sum()
    return float(total)


def est_dr_crossfit(
    S_arr,
    A_arr,
    R_arr,
    rho,
    pi_e,
    H,
    n_states,
    n_actions,
    weighted=False,
    model_fitter=fit_mle_model,
):
    """2-fold cross-fit DR/WDR: fit the model on one fold, score the other."""
    n = S_arr.shape[0]
    half = n // 2
    idx = [np.arange(0, half), np.arange(half, n)]
    vals, weights = [], []
    for k in (0, 1):
        fit_idx, score_idx = idx[1 - k], idx[k]
        P_hat, r_hat = model_fitter(
            S_arr[fit_idx], A_arr[fit_idx], R_arr[fit_idx], n_states, n_actions
        )
        Q_hat, V_hat = model_q_functions(P_hat, r_hat, pi_e, H)
        vals.append(
            dr_score(
                S_arr[score_idx],
                A_arr[score_idx],
                R_arr[score_idx],
                rho[score_idx],
                Q_hat,
                V_hat,
                weighted,
            )
        )
        weights.append(len(score_idx))
    return float(np.average(vals, weights=weights))


def est_mis(S_arr, A_arr, R_arr, rho, n_states):
    """Marginalized IS (Xie, Ma & Wang 2019, Eqns. 3.1-3.2).

    d-hat^{pi_b}_t: empirical state frequencies. d-hat^{pi_e}_t: forward
    recursion through the ratio-weighted empirical transition operator.
    r-hat^{pi_e}_t(s): per-state per-step reward IS. The estimator equals
    sum_t sum_s d-hat^{pi_e}_t(s) r-hat^{pi_e}_t(s).
    """
    n, H = S_arr.shape
    d_pib = np.zeros((H, n_states))
    for t in range(H):
        np.add.at(d_pib[t], S_arr[:, t], 1.0)
    n_st = d_pib.copy()  # visitation counts n_{s_t}
    d_pib /= n
    d_pie = np.zeros((H, n_states))
    d_pie[0] = d_pib[0]  # identical initial distributions
    for t in range(1, H):
        # P-hat^{pi_e}_t(s_t | s_{t-1}): ratio-weighted empirical transitions
        Pt = np.zeros((n_states, n_states))
        np.add.at(Pt, (S_arr[:, t - 1], S_arr[:, t]), rho[:, t - 1])
        cnt_prev = n_st[t - 1]
        Pt = np.where(cnt_prev[:, None] > 0, Pt / np.maximum(cnt_prev, 1)[:, None], 0.0)
        d_pie[t] = d_pie[t - 1] @ Pt
    r_pie = np.zeros((H, n_states))
    for t in range(H):
        np.add.at(r_pie[t], S_arr[:, t], rho[:, t] * R_arr[:, t])
    r_pie = np.where(n_st > 0, r_pie / np.maximum(n_st, 1), 0.0)
    return float((d_pie * r_pie).sum())


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def _score_dataset(S_arr, A_arr, R_arr, pi_e, pi_b, H, n_states, n_actions):
    """All seven estimators on one logged dataset."""
    rho = step_ratios(S_arr, A_arr, pi_e, pi_b)
    P_hat, r_hat = fit_mle_model(S_arr, A_arr, R_arr, n_states, n_actions)
    Q_dm, V_dm = model_q_functions(P_hat, r_hat, pi_e, H)
    return {
        "DM": est_dm(S_arr, V_dm),
        "IS": est_is(rho, R_arr),
        "PDIS": est_pdis(rho, R_arr),
        "WIS": est_wis(rho, R_arr),
        "DR": est_dr_crossfit(S_arr, A_arr, R_arr, rho, pi_e, H, n_states, n_actions),
        "WDR": est_dr_crossfit(
            S_arr, A_arr, R_arr, rho, pi_e, H, n_states, n_actions, weighted=True
        ),
        "MIS": est_mis(S_arr, A_arr, R_arr, rho, n_states),
    }


def compute_shared(cfg):
    P, r, init_dist, pi_b, pi_e = build_env(cfg)
    truths, marg_ratio_max = {}, {}
    for H in cfg["H_GRID"]:
        J, _, _ = dp_value(P, r, init_dist, pi_e, H)
        truths[H] = J
        d_e = state_marginals(P, init_dist, pi_e, H)
        d_b = state_marginals(P, init_dist, pi_b, H)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(d_b > 1e-12, d_e / np.maximum(d_b, 1e-12), np.nan)
        marg_ratio_max[H] = float(np.nanmax(ratio))
    J_b, _, _ = dp_value(P, r, init_dist, pi_b, cfg["MC_CHECK_H"])
    # on-policy MC oracle check of the simulator against DP (audit point 5)
    rng = np.random.default_rng(np.random.SeedSequence([cfg["SEED_ROOT"], 999]))
    S_arr, A_arr, R_arr = simulate(
        P, r, init_dist, pi_e, cfg["MC_CHECK_H"], cfg["MC_CHECK_EPISODES"], rng
    )
    returns = R_arr.sum(axis=1)
    mc_mean = float(returns.mean())
    mc_se = float(returns.std(ddof=1) / np.sqrt(len(returns)))
    return {
        "truths": truths,
        "J_behavior_H16": J_b,
        "mc_mean": mc_mean,
        "mc_se": mc_se,
        "marg_ratio_max": marg_ratio_max,
        "rho_max": float((pi_e / pi_b).max()),
        "rho_min": float((pi_e / pi_b).min()),
    }


def _run_config(cfg, H, n, config_tag, model_fitters=None):
    """N_REPS Monte Carlo replications at one (H, n) cell; returns raw draws
    per estimator, all estimators scoring the same datasets per rep."""
    P, r, init_dist, pi_b, pi_e = build_env(cfg)
    S, A = r.shape
    draws = {name: np.zeros(cfg["N_REPS"]) for name in ESTIMATORS}
    for rep in range(cfg["N_REPS"]):
        rng = np.random.default_rng(
            np.random.SeedSequence([cfg["SEED_ROOT"], config_tag, rep])
        )
        S_arr, A_arr, R_arr = simulate(P, r, init_dist, pi_b, H, n, rng)
        vals = _score_dataset(S_arr, A_arr, R_arr, pi_e, pi_b, H, S, A)
        for name in ESTIMATORS:
            draws[name][rep] = vals[name]
    return draws


def compute_sweep_n(cfg):
    out = {}
    for j, n in enumerate(cfg["N_GRID"]):
        out[n] = _run_config(cfg, cfg["H"], n, config_tag=100 + j)
        print(f"  sweep_n: n={n} done")
    return out


def compute_sweep_h(cfg):
    out = {}
    for j, H in enumerate(cfg["H_SWEEP"]):
        out[H] = _run_config(cfg, H, cfg["N_EPISODES"], config_tag=200 + j)
        print(f"  sweep_h: H={H} done")
    return out


def compute_ablation(cfg):
    """2x2 double-robustness ablation + DM and PDIS anchors."""
    P, r, init_dist, pi_b, pi_e = build_env(cfg)
    S, A = r.shape
    H, n = cfg["H"], cfg["N_EPISODES"]
    pi_b_wrong = np.full_like(pi_b, cfg["WRONG_PROPENSITY"])
    cells = {
        "DR_QR_eR": [],
        "DR_QW_eR": [],
        "DR_QR_eW": [],
        "DR_QW_eW": [],
        "DM_QR": [],
        "DM_QW": [],
        "PDIS_eR": [],
        "PDIS_eW": [],
    }
    for rep in range(cfg["N_REPS"]):
        rng = np.random.default_rng(
            np.random.SeedSequence([cfg["SEED_ROOT"], 300, rep])
        )
        S_arr, A_arr, R_arr = simulate(P, r, init_dist, pi_b, H, n, rng)
        rho_r = step_ratios(S_arr, A_arr, pi_e, pi_b)
        rho_w = step_ratios(S_arr, A_arr, pi_e, pi_b_wrong)
        for tag, rho, fitter in [
            ("DR_QR_eR", rho_r, fit_mle_model),
            ("DR_QW_eR", rho_r, fit_pooled_model),
            ("DR_QR_eW", rho_w, fit_mle_model),
            ("DR_QW_eW", rho_w, fit_pooled_model),
        ]:
            cells[tag].append(
                est_dr_crossfit(
                    S_arr, A_arr, R_arr, rho, pi_e, H, S, A, model_fitter=fitter
                )
            )
        for tag, fitter in [("DM_QR", fit_mle_model), ("DM_QW", fit_pooled_model)]:
            P_hat, r_hat = fitter(S_arr, A_arr, R_arr, S, A)
            Q_hat, V_hat = model_q_functions(P_hat, r_hat, pi_e, H)
            cells[tag].append(est_dm(S_arr, V_hat))
        cells["PDIS_eR"].append(est_pdis(rho_r, R_arr))
        cells["PDIS_eW"].append(est_pdis(rho_w, R_arr))
    return {k: np.asarray(v) for k, v in cells.items()}


def compute_data(force=None):
    force = force or set()
    cascade = "shared" in force
    shared = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "shared",
        SHARED_CONFIG,
        compute_shared,
        SHARED_CONFIG,
        force=cascade,
    )
    sweep_n = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "sweep_n",
        SWEEP_N_CONFIG,
        compute_sweep_n,
        SWEEP_N_CONFIG,
        force=cascade or "sweep_n" in force,
    )
    sweep_h = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "sweep_h",
        SWEEP_H_CONFIG,
        compute_sweep_h,
        SWEEP_H_CONFIG,
        force=cascade or "sweep_h" in force,
    )
    ablation = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "ablation",
        ABLATION_CONFIG,
        compute_ablation,
        ABLATION_CONFIG,
        force=cascade or "ablation" in force,
    )
    return {
        "shared": shared,
        "sweep_n": sweep_n,
        "sweep_h": sweep_h,
        "ablation": ablation,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _stats(draws, truth):
    bias = float(draws.mean() - truth)
    sd = float(draws.std(ddof=1))
    rmse = float(np.sqrt(((draws - truth) ** 2).mean()))
    mc_se = sd / np.sqrt(len(draws))
    return bias, sd, rmse, mc_se


def generate_outputs(data):
    import matplotlib.pyplot as plt

    shared = data["shared"]
    truths = shared["truths"]
    J16 = truths[SWEEP_N_CONFIG["H"]]

    fig, axes = plt.subplots(1, 3, figsize=FIG_TRIPLE)

    # (a) relative RMSE vs n, log-log, with n^{-1/2} reference
    ax = axes[0]
    n_grid = list(SWEEP_N_CONFIG["N_GRID"])
    for name in ESTIMATORS:
        rmse = [_stats(data["sweep_n"][n][name], J16)[2] / abs(J16) for n in n_grid]
        ax.plot(n_grid, rmse, marker="o", ms=3.5, color=EST_COLORS[name], label=name)
    ref0 = _stats(data["sweep_n"][n_grid[0]]["DR"], J16)[2] / abs(J16)
    ax.plot(
        n_grid,
        [ref0 * np.sqrt(n_grid[0] / n) for n in n_grid],
        **BENCH_STYLE,
        label=r"$n^{-1/2}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"episodes $n$")
    ax.set_ylabel(r"relative RMSE")
    ax.set_title(f"(a) Sample-size sweep, $H={SWEEP_N_CONFIG['H']}$")
    ax.legend(fontsize=7, ncol=2)

    # (b) relative RMSE vs H, log y
    ax = axes[1]
    h_grid = list(SWEEP_H_CONFIG["H_SWEEP"])
    for name in ESTIMATORS:
        rmse = [
            _stats(data["sweep_h"][H][name], truths[H])[2] / abs(truths[H])
            for H in h_grid
        ]
        ax.plot(h_grid, rmse, marker="o", ms=3.5, color=EST_COLORS[name], label=name)
    ax.set_yscale("log")
    ax.set_xlabel(r"horizon $H$")
    ax.set_ylabel(r"relative RMSE")
    ax.set_title(f"(b) Horizon sweep, $n={SWEEP_H_CONFIG['N_EPISODES']}$")
    ax.legend(fontsize=7, ncol=2)

    # (c) ablation bias with 95% CI whiskers
    ax = axes[2]
    cells = [
        ("DR: both right", "DR_QR_eR", EST_COLORS["DR"]),
        ("DR: $\\hat Q$ wrong", "DR_QW_eR", EST_COLORS["DR"]),
        ("DR: $\\hat e$ wrong", "DR_QR_eW", EST_COLORS["DR"]),
        ("DR: both wrong", "DR_QW_eW", COLORS["black"]),
        ("DM: $\\hat Q$ wrong", "DM_QW", EST_COLORS["DM"]),
        ("PDIS: $\\hat e$ wrong", "PDIS_eW", EST_COLORS["PDIS"]),
    ]
    X_CLIP = 3.0  # off-scale points annotated at the clipped edge
    for y, (label, key, color) in enumerate(cells):
        d = data["ablation"][key]
        bias = d.mean() - J16
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        if abs(bias) > X_CLIP:
            ax.annotate(
                f"{bias:+.1f} $\\rightarrow$",
                (X_CLIP * 0.90, y),
                ha="right",
                va="center",
                fontsize=8,
                color=color,
            )
        else:
            ax.errorbar(bias, y, xerr=ci, fmt="o", ms=5, color=color, capsize=3)
    ax.set_xlim(-1.0, X_CLIP)
    ax.axvline(0.0, **BENCH_STYLE)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([c[0] for c in cells], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"bias of $\hat J(\pi_e)$")
    ax.set_title(
        f"(c) Double robustness, $H={ABLATION_CONFIG['H']}$, "
        f"$n={ABLATION_CONFIG['N_EPISODES']}$"
    )

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "ope_estimators.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # consolidated results table, rank-ordered by RMSE at n = 2000, H = 16
    n_big = SWEEP_N_CONFIG["N_GRID"][-1]
    H_big = SWEEP_H_CONFIG["H_SWEEP"][-1]
    rows = []
    for name in ESTIMATORS:
        bias, sd, rmse, _ = _stats(data["sweep_n"][n_big][name], J16)
        rmse_h = _stats(data["sweep_h"][H_big][name], truths[H_big])[2] / abs(
            truths[H_big]
        )
        rows.append((name, bias, sd, rmse, rmse_h))
    rows.sort(key=lambda x: x[3])
    tab_path = os.path.join(OUTPUT_DIR, "ope_estimators_results.tex")
    with open(tab_path, "w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Estimator & Bias & SD & RMSE & Rel. RMSE at $H=64$ \\\\\n\\midrule\n")
        for name, bias, sd, rmse, rmse_h in rows:
            f.write(
                f"{name} & {bias:+.4f} & {sd:.4f} & {rmse:.4f} & {rmse_h:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")

    # 2x2 DR ablation table + anchors
    ab = data["ablation"]
    ab_path = os.path.join(OUTPUT_DIR, "ope_estimators_dr_ablation.tex")
    label = {
        "DR_QR_eR": ("DR", "right", "right"),
        "DR_QW_eR": ("DR", "wrong", "right"),
        "DR_QR_eW": ("DR", "right", "wrong"),
        "DR_QW_eW": ("DR", "wrong", "wrong"),
        "DM_QR": ("DM", "right", "--"),
        "DM_QW": ("DM", "wrong", "--"),
        "PDIS_eR": ("PDIS", "--", "right"),
        "PDIS_eW": ("PDIS", "--", "wrong"),
    }
    order = [
        "DR_QR_eR",
        "DR_QW_eR",
        "DR_QR_eW",
        "DR_QW_eW",
        "DM_QR",
        "DM_QW",
        "PDIS_eR",
        "PDIS_eW",
    ]
    with open(ab_path, "w") as f:
        f.write("\\begin{tabular}{lllrr}\n\\toprule\n")
        f.write(
            "Estimator & $\\hat Q$ model & Propensity & Bias & RMSE \\\\\n\\midrule\n"
        )
        for key in order:
            est, qm, em = label[key]
            bias, sd, rmse, _ = _stats(ab[key], J16)
            f.write(f"{est} & {qm} & {em} & {bias:+.4f} & {rmse:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    return fig_path, tab_path, ab_path


def print_report(data):
    shared = data["shared"]
    truths = shared["truths"]
    J16 = truths[16]
    print("=" * 78)
    print("OPE estimators on the tabular retention MDP")
    print("=" * 78)
    print("States=5 Actions=2 gamma=1 init=uniform{1,2,3}")
    print("pi_b(1|s)=0.6-0.1s  pi_e(1|s)=0.9 (s<=1) / 0.1 (s>=2)")
    print(f"step ratio range: [{shared['rho_min']:.3f}, {shared['rho_max']:.3f}]")
    print(
        f"N_REPS={N_REPS}  n grid={SWEEP_N_CONFIG['N_GRID']}  "
        f"H grid={SWEEP_H_CONFIG['H_SWEEP']}"
    )
    print()
    print("Ground truth J(pi_e) by exact DP:")
    for H, J in truths.items():
        print(
            f"  H={H:3d}  J={J:+.6f}  max marginal state ratio={shared['marg_ratio_max'][H]:.3f}"
        )
    print(f"J(pi_b) at H=16: {shared['J_behavior_H16']:+.6f}")
    print(
        f"MC oracle check (100k episodes under pi_e, H=16): "
        f"{shared['mc_mean']:+.6f} (SE {shared['mc_se']:.6f}) vs DP {J16:+.6f}  "
        f"|diff|/SE = {abs(shared['mc_mean'] - J16) / shared['mc_se']:.2f}"
    )
    print()
    print(f"Experiment (a): sample-size sweep at H=16 (truth {J16:+.6f})")
    hdr = f"{'n':>6} " + "".join(f"{name:>22}" for name in ESTIMATORS)
    print(hdr)
    print(" " * 7 + "".join(f"{'bias (SE) / RMSE':>22}" for _ in ESTIMATORS))
    for n in SWEEP_N_CONFIG["N_GRID"]:
        cells = []
        for name in ESTIMATORS:
            bias, sd, rmse, mc_se = _stats(data["sweep_n"][n][name], J16)
            cells.append(f"{bias:+.3f}({mc_se:.3f})/{rmse:.3f}")
        print(f"{n:>6} " + "".join(f"{c:>22}" for c in cells))
    print()
    print(
        f"Experiment (b): horizon sweep at n={SWEEP_H_CONFIG['N_EPISODES']} "
        f"(relative RMSE)"
    )
    print(f"{'H':>6} " + "".join(f"{name:>10}" for name in ESTIMATORS))
    for H in SWEEP_H_CONFIG["H_SWEEP"]:
        cells = []
        for name in ESTIMATORS:
            rmse = _stats(data["sweep_h"][H][name], truths[H])[2] / abs(truths[H])
            cells.append(f"{rmse:.4f}")
        print(f"{H:>6} " + "".join(f"{c:>10}" for c in cells))
    print()
    print(
        f"Experiment (c): 2x2 double-robustness ablation at H=16, "
        f"n={ABLATION_CONFIG['N_EPISODES']} (truth {J16:+.6f})"
    )
    print(f"{'cell':>12} {'bias':>10} {'(MC SE)':>10} {'RMSE':>10}")
    for key in [
        "DR_QR_eR",
        "DR_QW_eR",
        "DR_QR_eW",
        "DR_QW_eW",
        "DM_QR",
        "DM_QW",
        "PDIS_eR",
        "PDIS_eW",
    ]:
        bias, sd, rmse, mc_se = _stats(data["ablation"][key], J16)
        print(f"{key:>12} {bias:>+10.4f} {mc_se:>10.4f} {rmse:>10.4f}")
    print()
    # theory-consistency checks (facts, computed from the draws above)
    is_bias_ok = all(
        abs(_stats(data["sweep_n"][n]["IS"], J16)[0])
        <= 3 * _stats(data["sweep_n"][n]["IS"], J16)[3]
        for n in SWEEP_N_CONFIG["N_GRID"]
    )
    pdis_bias_ok = all(
        abs(_stats(data["sweep_n"][n]["PDIS"], J16)[0])
        <= 3 * _stats(data["sweep_n"][n]["PDIS"], J16)[3]
        for n in SWEEP_N_CONFIG["N_GRID"]
    )
    var_pdis_le_is = all(
        data["sweep_n"][n]["PDIS"].std(ddof=1) <= data["sweep_n"][n]["IS"].std(ddof=1)
        for n in SWEEP_N_CONFIG["N_GRID"]
    )
    dr_le_pdis = (
        _stats(data["sweep_n"][2000]["DR"], J16)[2]
        <= _stats(data["sweep_n"][2000]["PDIS"], J16)[2]
    )
    is_h64 = _stats(data["sweep_h"][64]["IS"], truths[64])[2] / abs(truths[64])
    mis_h64 = _stats(data["sweep_h"][64]["MIS"], truths[64])[2] / abs(truths[64])
    print("Checks (3 MC-SE criterion for bias):")
    print(f"  IS bias within 3 MC-SE of 0 at every n:    {is_bias_ok}")
    print(f"  PDIS bias within 3 MC-SE of 0 at every n:  {pdis_bias_ok}")
    print(f"  SD(PDIS) <= SD(IS) at every n:             {var_pdis_le_is}")
    print(f"  RMSE(DR) <= RMSE(PDIS) at n=2000:          {dr_le_pdis}")
    print(
        f"  rel. RMSE at H=64, IS vs MIS:              {is_h64:.3f} vs {mis_h64:.3f} "
        f"(ratio {is_h64 / mis_h64:.1f}x)"
    )


def main():
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    data = compute_data(force=force)
    print_report(data)
    if not args.data_only:
        paths = generate_outputs(data)
        print()
        print("Output files:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
