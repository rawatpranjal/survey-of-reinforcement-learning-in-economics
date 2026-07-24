# Dynamic Double Machine Learning on a 2-stage SNMM.
# Chapter ch10b_rl_for_ci, Off-Policy Evaluation and Dynamic Treatment Effects.
#
# The numerical design uses a stylized Fast Track-inspired setting. X_1 and
# X_2 collect child, caregiver, and family measures; T_1 and T_2 are additional
# home visits. The design is not a reconstruction of the Fast Track trial.
#
# Reproduces the central claim of Lewis & Syrgkanis (NeurIPS 2021,
# "Double/Debiased Machine Learning for Dynamic Treatment Effects via
# g-Estimation," arXiv:2002.07285): a Neyman-orthogonal sequential
# residualisation moment delivers sqrt(n)-consistent, asymptotically normal
# estimates of the SNMM structural parameters even when nuisance functions are
# fit by machine learning on high-dimensional state. Naive OLS on the full
# panel is deliberately misspecified; correctly specified stabilized IPTW is
# included as a high-variance benchmark.
#
# Setting (Lewis-Syrgkanis Section 2.1, partially linear Markovian model):
#   X_1 ~ N(0, I_p)
#   T_1 ~ Bernoulli(sigmoid(gamma' X_1))             # confounded by X_1
#   X_2 = B X_1 + alpha T_1 + eta_1,    eta_1 ~ N(0, sigma_eta^2 I_p)
#   T_2 ~ Bernoulli(sigmoid(gamma' X_2))             # confounded by X_2
#   Y   = psi_1 T_1 + psi_2 T_2 + mu' X_1 + nu' eta_1 + eps,
#                                                     eps ~ N(0, sigma_eps^2)
# with sparse mu and nu (first s coordinates) and high-dimensional state
# p = 20. nu is aligned with gamma so eta_1 drives both Y and the period-2
# propensity, giving the canonical treatment-confounder-feedback channel.
# True structural parameters psi_1*, psi_2*.
#
# Estimators compared:
#   1. Naive OLS with initial controls: regress Y on (T_1, T_2, X_1). Biased
#      because it treats the longitudinal problem as static and omits the
#      period-2 treatment-confounder feedback through X_2.
#   2. IPTW-fitted MSM: marginal structural model E[Y(t_1, t_2)] = beta_0 + psi_1
#      t_1 + psi_2 t_2 fit by weighted least squares with stabilized weights
#      (Robins-Hernan-Brumback 2000). Correct under sequential ignorability.
#   3. Dynamic DML: the Lewis-Syrgkanis recursive estimator. Cross-fit Lasso
#      nuisances q_t and p_{j,t}; residualise outcome and treatments at each
#      period; peel off the period-2 effect; regress on residualised period-1
#      treatment to recover psi_1. Inference uses the full upper-triangular
#      Jacobian, so uncertainty in psi_2 propagates into the standard error
#      for psi_1.
#
# Outputs:
#   dynamic_dml_snmm_coverage.png  -- bias and 95%-CI coverage vs. sample size
#   dynamic_dml_snmm_results.tex   -- consolidated results table at n=4000
#   dynamic_dml_snmm_joint_inference.tex -- joint sandwich calibration
#   dynamic_dml_snmm_stdout.txt    -- numerical log

import argparse
import os
import sys
import warnings
import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import KFold
from scipy.stats import norm
from tqdm import tqdm

# Silence sklearn deprecation/future warnings unrelated to the algorithms here.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, BENCH_STYLE, FIG_DOUBLE
from sims.sim_cache import (
    compute_or_load,
    add_component_args,
    parse_force_set,
)

apply_style()
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "dynamic_dml_snmm"

# DGP parameters
P_STATE = 20  # state dimension (high relative to true effective dim)
S_SPARSE = 5  # number of nonzero coordinates in mu (outcome regression)
PSI_TRUE = np.array([1.0, 0.5])  # true (psi_1, psi_2)
GAMMA_NORM = 1.5  # confounding strength: || gamma ||_2
ALPHA_NORM = 1.0  # transition strength T_1 -> X_2 (T_1 shifts X_2[0] by 1)
B_OPNORM = 0.5  # || B ||_op for state dynamics (stability requires <1)
SIGMA_ETA = 0.6  # state innovation noise std (larger -> stronger feedback channel)
SIGMA_EPS = 0.5  # outcome noise std
NU_NORM = 2.0  # magnitude of nu (outcome-coupling on eta_1)
DGP_SEED = 12345  # fixes the population-level random matrices B, gamma, mu, alpha

# Sample sizes to sweep
N_GRID = [250, 500, 1000, 2000, 4000]
N_SEEDS = 200  # Monte Carlo replications per sample size
CI_LEVEL = 0.95  # nominal coverage

# Cross-fitting folds for Dynamic DML
K_FOLDS = 5

SHARED_CONFIG = {
    "P_STATE": P_STATE,
    "S_SPARSE": S_SPARSE,
    "PSI_TRUE": PSI_TRUE.tolist(),
    "GAMMA_NORM": GAMMA_NORM,
    "ALPHA_NORM": ALPHA_NORM,
    "B_OPNORM": B_OPNORM,
    "SIGMA_ETA": SIGMA_ETA,
    "SIGMA_EPS": SIGMA_EPS,
    "NU_NORM": NU_NORM,
    "DGP_SEED": DGP_SEED,
    "N_GRID": N_GRID,
    "N_SEEDS": N_SEEDS,
    "K_FOLDS": K_FOLDS,
    "SE_METHOD": "joint_upper_triangular_sandwich_v2",
}

NAIVE_CONFIG = {**SHARED_CONFIG, "method": "naive_ols"}
MSM_CONFIG = {**SHARED_CONFIG, "method": "msm_iptw"}
DML_CONFIG = {**SHARED_CONFIG, "method": "dynamic_dml"}
JOINT_INFERENCE_CONFIG = {
    **SHARED_CONFIG,
    "method": "dynamic_dml_joint_inference",
    "N": N_GRID[-1],
    "N_SEEDS": N_SEEDS,
    "COVARIANCE_METHOD": "full_joint_upper_triangular_sandwich_v1",
}


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_population_params(rng):
    """Sample the population-level matrices once, fixed across all Monte Carlo seeds.

    Returns dict with B (p x p), alpha (p,), gamma (p,), mu (p,).
    """
    p = P_STATE
    # B: random orthogonal projection scaled to operator norm B_OPNORM
    raw = rng.standard_normal((p, p)) / np.sqrt(p)
    u, _, vt = np.linalg.svd(raw)
    B = B_OPNORM * (u @ vt)
    # alpha: sparse, magnitude ALPHA_NORM concentrated on first coordinate
    alpha = np.zeros(p)
    alpha[0] = ALPHA_NORM
    # gamma: confounding direction in first SPARSE coordinates
    gamma = np.zeros(p)
    gamma[:S_SPARSE] = rng.standard_normal(S_SPARSE)
    gamma *= GAMMA_NORM / np.linalg.norm(gamma)
    # mu: sparse outcome-regression coefficients on X_1 (baseline state)
    mu = np.zeros(p)
    mu[:S_SPARSE] = rng.standard_normal(S_SPARSE)
    mu[:S_SPARSE] *= 1.0 / np.linalg.norm(mu[:S_SPARSE]) * 1.5  # || mu || = 1.5

    # nu: outcome-coupling on the period-1 state shock eta_1. This is the
    # "treatment-confounder feedback" channel: eta_1 enters both Y (directly)
    # and X_2 (which determines T_2), making T_2 endogenous conditional on
    # (X_1, T_1). We align nu *parallel to gamma* so the eta_1 -> Y channel
    # is maximally correlated with the eta_1 -> T_2 channel, giving a clean
    # bias signal for naive OLS that omits X_2.
    nu = NU_NORM * gamma / np.linalg.norm(gamma)
    return {"B": B, "alpha": alpha, "gamma": gamma, "mu": mu, "nu": nu}


def generate_panel(n, pop, rng):
    """Generate n iid two-stage trajectories.

    Y structural equation:
        Y = psi_1 T_1 + psi_2 T_2 + mu' X_1 + nu' eta_1 + eps

    where eta_1 is the period-1 state shock that also drives X_2. This is the
    canonical "treatment-confounder feedback" structure: eta_1 affects Y
    directly AND (via X_2) drives the period-2 propensity, so conditional on
    (X_1, T_1), T_2 is correlated with Y's noise through eta_1. The SNMM blip
    parameters are exactly (psi_1, psi_2): no intermediate state X_2 appears
    on the right-hand side of Y, so there is no indirect-effect leakage to
    confuse the target of estimation.

    Returns (X1, T1, X2, T2, Y) with shapes (n,p), (n,), (n,p), (n,), (n,).
    """
    p = P_STATE
    B, alpha, gamma, mu, nu = pop["B"], pop["alpha"], pop["gamma"], pop["mu"], pop["nu"]

    X1 = rng.standard_normal((n, p))
    pi_1 = _sigmoid(X1 @ gamma)
    T1 = (rng.uniform(size=n) < pi_1).astype(float)

    eta1 = SIGMA_ETA * rng.standard_normal((n, p))
    X2 = X1 @ B.T + np.outer(T1, alpha) + eta1
    pi_2 = _sigmoid(X2 @ gamma)
    T2 = (rng.uniform(size=n) < pi_2).astype(float)

    eps = SIGMA_EPS * rng.standard_normal(n)
    Y = PSI_TRUE[0] * T1 + PSI_TRUE[1] * T2 + X1 @ mu + eta1 @ nu + eps

    return X1, T1, X2, T2, Y


# ---------------------------------------------------------------------------
# Estimator 1 -- Naive OLS controlling only on initial state X_1
# ---------------------------------------------------------------------------
def fit_naive_ols(X1, T1, X2, T2, Y):
    """OLS of Y on (T1, T2, X1). Returns (psi_hat, se_hat) for (psi_1, psi_2).

    This is the "init-ctrls" baseline of Lewis-Syrgkanis 2021: control only on
    the initial state, treating dynamic treatment effects as if they were a
    static problem. It is biased because X_2 is omitted yet directly affects
    Y. The omitted-variable formula gives bias on psi_1 of approximately
    mu' alpha (the indirect path T_1 -> X_2 -> Y absorbed into the T_1
    coefficient) and bias on psi_2 from the X_2-omitted-confounder channel
    T_2 ~ pi(X_2), Y depends on X_2.
    """
    n = len(Y)
    Z = np.column_stack([np.ones(n), T1, T2, X1])
    beta, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    psi_hat = beta[1:3]
    resid = Y - Z @ beta
    XtX_inv = np.linalg.inv(Z.T @ Z)
    Omega = (Z * resid[:, None]).T @ (Z * resid[:, None])
    cov = XtX_inv @ Omega @ XtX_inv
    se_hat = np.sqrt(np.diag(cov)[1:3])
    return psi_hat, se_hat


# ---------------------------------------------------------------------------
# Estimator 2 -- Marginal structural model with stabilized IPTW weights
# ---------------------------------------------------------------------------
def fit_msm_iptw(X1, T1, X2, T2, Y):
    """Stabilized IPTW MSM regression Y ~ T1 + T2 (Robins-Hernan-Brumback 2000)."""
    n = len(Y)
    # Stage-1 propensity model: logistic in X1
    e1 = LogisticRegressionCV(Cs=10, cv=3, solver="liblinear", max_iter=500)
    e1.fit(X1, T1.astype(int))
    pi1_hat = e1.predict_proba(X1)[:, 1].clip(1e-3, 1 - 1e-3)

    # Marginal P(T1) for the stabilized-weight numerator
    p1_marg = max(min(T1.mean(), 1 - 1e-3), 1e-3)

    # Stage-2 propensity model: logistic in (X1, T1, X2)
    Z2 = np.column_stack([X1, T1, X2])
    e2 = LogisticRegressionCV(Cs=10, cv=3, solver="liblinear", max_iter=500)
    e2.fit(Z2, T2.astype(int))
    pi2_hat = e2.predict_proba(Z2)[:, 1].clip(1e-3, 1 - 1e-3)

    # Marginal P(T2 | T1) for the stabilized-weight numerator
    Z2_marg = np.column_stack([T1])
    e2_marg = LogisticRegressionCV(Cs=10, cv=3, solver="liblinear", max_iter=500)
    e2_marg.fit(Z2_marg, T2.astype(int))
    pi2_marg = e2_marg.predict_proba(Z2_marg)[:, 1].clip(1e-3, 1 - 1e-3)

    num1 = T1 * p1_marg + (1 - T1) * (1 - p1_marg)
    den1 = T1 * pi1_hat + (1 - T1) * (1 - pi1_hat)
    num2 = T2 * pi2_marg + (1 - T2) * (1 - pi2_marg)
    den2 = T2 * pi2_hat + (1 - T2) * (1 - pi2_hat)
    w = (num1 / den1) * (num2 / den2)

    # Trim extreme weights (standard MSM practice)
    w = np.clip(w, np.quantile(w, 0.005), np.quantile(w, 0.995))

    # Weighted OLS of Y on (1, T1, T2)
    Z = np.column_stack([np.ones(n), T1, T2])
    Wmat = w[:, None]
    WZ = Z * Wmat
    beta, *_ = np.linalg.lstsq(WZ.T @ Z, WZ.T @ Y, rcond=None)
    psi_hat = beta[1:3]

    # Sandwich variance (treating weights as fixed)
    resid = Y - Z @ beta
    bread = np.linalg.inv((Z * w[:, None]).T @ Z)
    meat = (Z * (w * resid)[:, None]).T @ (Z * (w * resid)[:, None])
    cov = bread @ meat @ bread
    se_hat = np.sqrt(np.diag(cov)[1:3])
    return psi_hat, se_hat


# ---------------------------------------------------------------------------
# Estimator 3 -- Dynamic DML (Lewis & Syrgkanis 2021)
# ---------------------------------------------------------------------------
def _crossfit_predict(estimator_factory, X, y, n_folds, rng_seed):
    """Cross-fit predict: for each fold, train on the complement and predict on the fold."""
    n = len(y)
    pred = np.zeros(n, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng_seed)
    for train_idx, test_idx in kf.split(X):
        est = estimator_factory()
        est.fit(X[train_idx], y[train_idx])
        if hasattr(est, "predict_proba"):
            pred[test_idx] = est.predict_proba(X[test_idx])[:, 1]
        else:
            pred[test_idx] = est.predict(X[test_idx])
    return pred


def fit_dynamic_dml(
    X1,
    T1,
    X2,
    T2,
    Y,
    n_folds=K_FOLDS,
    rng_seed=0,
    return_cov=False,
):
    """Lewis-Syrgkanis dynamic DML on a 2-stage SNMM with binary treatments.

    Algorithm 1 of the paper, specialized to T=2:
      1. Cross-fit nuisance models for stage 2 (last stage):
           q_2(X_2) = E[Y | X_2, T_1 included as covariate]
           p_2(X_2) = E[T_2 | X_2, T_1]
         These are computed on (X1, T1, X2) so the "history" at stage 2 is the
         full (X1, T1, X2) panel.
      2. Estimate psi_2 from residual-on-residual:
           Y_tilde_2 = Y - q_2_hat
           T_tilde_2 = T_2 - p_2_hat
           psi_2_hat = sum(Y_tilde_2 * T_tilde_2) / sum(T_tilde_2 ** 2)
      3. Cross-fit nuisance models for stage 1:
           q_1(X_1) = E[Y | X_1]                      # outcome residualisation
           p_{2,1}(X_1) = E[T_2 | X_1]               # future treatment residualisation
           p_{1,1}(X_1) = E[T_1 | X_1]               # current treatment residualisation
      4. Estimate psi_1 from residual-on-residual, with the period-2 effect
         peeled off through its residualized treatment (Theorem 2):
           Y_tilde_1 = Y - q_1_hat
           T_tilde_{2,1} = T_2 - p_{2,1}_hat
           T_tilde_{1,1} = T_1 - p_{1,1}_hat
           psi_1_hat from sum((Y_tilde_1 - psi_2_hat * T_tilde_{2,1}) * T_tilde_{1,1})
                       /  sum(T_tilde_{1,1} ** 2)
         The peel-off ensures the conditional moment restriction in Theorem 2
         identifies psi_1 only.
    """
    n = len(Y)

    # Lasso factories with internal CV; alpha selected from a small grid.
    def lasso_factory():
        return LassoCV(cv=3, max_iter=5000, n_alphas=10, random_state=rng_seed)

    def logit_factory():
        return LogisticRegressionCV(
            Cs=5,
            cv=3,
            solver="liblinear",
            max_iter=500,
            penalty="l1",
            random_state=rng_seed,
        )

    # === Stage 2 ===
    H2 = np.column_stack([X1, T1[:, None], X2])  # history at stage 2 = (X1, T1, X2)

    q2_hat = _crossfit_predict(lasso_factory, H2, Y, n_folds, rng_seed)
    p2_hat = _crossfit_predict(logit_factory, H2, T2.astype(int), n_folds, rng_seed + 1)
    p2_hat = np.clip(p2_hat, 1e-3, 1 - 1e-3)

    Y_til_2 = Y - q2_hat
    T_til_22 = T2 - p2_hat

    # OLS of Y_til_2 on T_til_22 (scalar regression -> closed form)
    denom2 = (T_til_22**2).sum()
    psi_2_hat = (Y_til_2 * T_til_22).sum() / denom2

    psi2_resid = Y_til_2 - psi_2_hat * T_til_22

    # === Stage 1 (Lewis-Syrgkanis Algorithm 1 + Theorem 2) ===
    # The algorithm regresses RAW Y on X_1 (not the calibrated outcome).
    # Then the conditional moment at t=1 reads
    #   E[((Y - q_1) - psi_2 * T_til_{2,1} - psi_1 * T_til_{1,1}) * T_til_{1,1} | X_1] = 0
    # so psi_1 is recovered from
    #   psi_1 = sum_i ((Y_i - q_1_i) - psi_2_hat * T_til_{2,1}_i) * T_til_{1,1}_i
    #          / sum_i T_til_{1,1}_i^2.
    H1 = X1  # history at stage 1 = X_1 only

    q1_hat = _crossfit_predict(lasso_factory, H1, Y, n_folds, rng_seed + 2)
    p11_hat = _crossfit_predict(
        logit_factory, H1, T1.astype(int), n_folds, rng_seed + 3
    )
    p11_hat = np.clip(p11_hat, 1e-3, 1 - 1e-3)
    # T_2 is binary, but as a regressor it lives on [0,1]; logistic is the right link.
    p21_raw = _crossfit_predict(
        logit_factory, H1, T2.astype(int), n_folds, rng_seed + 4
    )
    p21_hat = np.clip(p21_raw, 1e-3, 1 - 1e-3)

    Y_til_1 = Y - q1_hat
    T_til_11 = T1 - p11_hat
    T_til_21 = T2 - p21_hat

    denom1 = (T_til_11**2).sum()
    psi_1_hat = ((Y_til_1 - psi_2_hat * T_til_21) * T_til_11).sum() / denom1

    psi1_resid = Y_til_1 - psi_2_hat * T_til_21 - psi_1_hat * T_til_11

    # Joint Z-estimator sandwich. With parameters ordered (psi_1, psi_2),
    # Lewis-Syrgkanis's Jacobian is upper triangular:
    #
    #   J = [[E(T~_11^2), E(T~_11 T~_21)],
    #        [0,            E(T~_22^2)   ]].
    #
    # Using only the diagonal term for psi_1 would incorrectly treat psi_2 as
    # known and understate uncertainty whenever the two residualized
    # treatments remain correlated.
    J = np.array(
        [
            [
                np.mean(T_til_11**2),
                np.mean(T_til_11 * T_til_21),
            ],
            [
                0.0,
                np.mean(T_til_22**2),
            ],
        ]
    )
    scores = np.column_stack(
        [
            T_til_11 * psi1_resid,
            T_til_22 * psi2_resid,
        ]
    )
    influence = np.linalg.solve(J, scores.T).T
    # Full joint sandwich, including the off-diagonal covariance needed for
    # contrasts.  The score equations make the sample mean of the influence
    # values zero up to numerical precision.
    cov_hat = influence.T @ influence / n**2
    cov_hat = (cov_hat + cov_hat.T) / 2.0
    se_hat = np.sqrt(np.diag(cov_hat))

    if return_cov:
        return np.array([psi_1_hat, psi_2_hat]), se_hat, cov_hat
    return np.array([psi_1_hat, psi_2_hat]), se_hat


# ---------------------------------------------------------------------------
# Per-seed runner (shared across estimators)
# ---------------------------------------------------------------------------
def run_estimator(method, n_grid, n_seeds, pop):
    """For each n in n_grid and seed s in range(n_seeds), generate one panel and
    run the estimator. Returns dict of arrays of shape (len(n_grid), n_seeds, 2)."""
    psi_estimates = np.zeros((len(n_grid), n_seeds, 2))
    se_estimates = np.zeros((len(n_grid), n_seeds, 2))

    fn = {
        "naive_ols": fit_naive_ols,
        "msm_iptw": fit_msm_iptw,
        "dynamic_dml": fit_dynamic_dml,
    }[method]

    for i, n in enumerate(n_grid):
        for s in tqdm(
            range(n_seeds),
            desc=f"  {method} n={n}",
            leave=False,
            disable=not sys.stderr.isatty(),
        ):
            seed = (n * 10_007 + s) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            X1, T1, X2, T2, Y = generate_panel(n, pop, rng)
            if method == "dynamic_dml":
                psi_hat, se_hat = fn(X1, T1, X2, T2, Y, rng_seed=seed)
            else:
                psi_hat, se_hat = fn(X1, T1, X2, T2, Y)
            psi_estimates[i, s] = psi_hat
            se_estimates[i, s] = se_hat

    return {"psi": psi_estimates, "se": se_estimates, "n_grid": list(n_grid)}


def run_joint_inference(n, n_seeds, pop):
    """Dynamic-DML estimates and full sandwich covariance at one fixed n."""
    psi_estimates = np.zeros((n_seeds, 2))
    se_estimates = np.zeros((n_seeds, 2))
    cov_estimates = np.zeros((n_seeds, 2, 2))
    for s in tqdm(
        range(n_seeds),
        desc=f"  dynamic_dml joint covariance n={n}",
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
        seed = (n * 10_007 + s) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        X1, T1, X2, T2, Y = generate_panel(n, pop, rng)
        psi_hat, se_hat, cov_hat = fit_dynamic_dml(
            X1, T1, X2, T2, Y, rng_seed=seed, return_cov=True
        )
        psi_estimates[s] = psi_hat
        se_estimates[s] = se_hat
        cov_estimates[s] = cov_hat
    return {
        "psi": psi_estimates,
        "se": se_estimates,
        "cov": cov_estimates,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Compute_data: shared setup + 3 per-method computations
# ---------------------------------------------------------------------------
def compute_shared():
    """Fix the population parameters once."""
    rng = np.random.default_rng(DGP_SEED)
    pop = make_population_params(rng)
    return {"pop": pop}


def compute_data(force=None):
    force = force or set()

    shared = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "shared",
        SHARED_CONFIG,
        compute_shared,
        force=("shared" in force),
    )
    pop = shared["pop"]

    results = {}
    for name, config in [
        ("naive_ols", NAIVE_CONFIG),
        ("msm_iptw", MSM_CONFIG),
        ("dynamic_dml", DML_CONFIG),
    ]:
        results[name] = compute_or_load(
            CACHE_DIR,
            SCRIPT_NAME,
            name,
            config,
            run_estimator,
            name,
            N_GRID,
            N_SEEDS,
            pop,
            force=(name in force or "shared" in force),
        )

    joint_inference = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "joint_inference",
        JOINT_INFERENCE_CONFIG,
        run_joint_inference,
        JOINT_INFERENCE_CONFIG["N"],
        JOINT_INFERENCE_CONFIG["N_SEEDS"],
        pop,
        force=("joint_inference" in force or "shared" in force),
    )

    return {
        "shared": shared,
        "results": results,
        "joint_inference": joint_inference,
    }


# ---------------------------------------------------------------------------
# Summary statistics: bias, RMSE, coverage
# ---------------------------------------------------------------------------
def summarize(data):
    """Compute per-method bias / RMSE / coverage tables (shape: n_grid x 2)."""
    z = norm.ppf(0.5 + CI_LEVEL / 2)
    summary = {}
    for method in data["results"]:
        psi = data["results"][method]["psi"]  # (G, S, 2)
        se = data["results"][method]["se"]  # (G, S, 2)
        bias = psi.mean(axis=1) - PSI_TRUE  # (G, 2)
        rmse = np.sqrt(((psi - PSI_TRUE) ** 2).mean(axis=1))
        # 95% CI: psi_hat +/- z * se
        lo = psi - z * se
        hi = psi + z * se
        covered = ((lo <= PSI_TRUE) & (PSI_TRUE <= hi)).astype(float)
        coverage = covered.mean(axis=1)  # (G, 2)
        summary[method] = {
            "bias": bias,
            "rmse": rmse,
            "coverage": coverage,
            "mean_se": se.mean(axis=1),
        }
    summary["n_grid"] = N_GRID
    return summary


def validate_results(data):
    """Monte Carlo gates for recovery, standard errors, and coverage."""
    result = data["results"]["dynamic_dml"]
    if not np.all(np.isfinite(result["psi"])) or not np.all(np.isfinite(result["se"])):
        raise RuntimeError("Dynamic DML produced non-finite estimates")
    if np.any(result["se"] <= 0):
        raise RuntimeError("Dynamic DML produced a non-positive standard error")

    i = len(N_GRID) - 1
    psi = result["psi"][i]
    se = result["se"][i]
    bias = psi.mean(axis=0) - PSI_TRUE
    empirical_sd = psi.std(axis=0, ddof=1)
    se_ratio = se.mean(axis=0) / empirical_sd
    z = norm.ppf(0.5 + CI_LEVEL / 2)
    coverage = ((psi - z * se <= PSI_TRUE) & (PSI_TRUE <= psi + z * se)).mean(axis=0)
    left_miss = (PSI_TRUE < psi - z * se).mean(axis=0)
    right_miss = (PSI_TRUE > psi + z * se).mean(axis=0)

    if np.any(np.abs(bias) > 0.03):
        raise RuntimeError(f"Dynamic DML recovery failed at n={N_GRID[i]}: bias={bias}")
    if np.any((se_ratio < 0.80) | (se_ratio > 1.20)):
        raise RuntimeError(
            f"Formula and Monte Carlo standard errors disagree: ratio={se_ratio}"
        )
    if np.any((coverage < 0.90) | (coverage > 0.99)):
        raise RuntimeError(f"Dynamic DML coverage check failed: coverage={coverage}")
    if np.any(left_miss > 0.06) or np.any(right_miss > 0.06):
        raise RuntimeError(
            f"Dynamic DML tail coverage is asymmetric: left={left_miss}, right={right_miss}"
        )

    joint = data["joint_inference"]
    joint_psi = joint["psi"]
    joint_cov = joint["cov"]
    if not np.all(np.isfinite(joint_cov)):
        raise RuntimeError("Dynamic DML produced a non-finite joint covariance")
    if np.any(np.linalg.eigvalsh(joint_cov) < -1e-12):
        raise RuntimeError(
            "Dynamic DML produced a non-positive-semidefinite covariance"
        )

    # The dedicated joint run uses the same seed scheme as the largest-n cell.
    # Equality guards against the covariance path changing point estimates.
    if not np.array_equal(joint_psi, psi):
        raise RuntimeError("Joint-covariance path changed Dynamic DML point estimates")
    if not np.allclose(joint["se"], se, rtol=0.0, atol=1e-14):
        raise RuntimeError("Joint covariance diagonal disagrees with reported SEs")

    empirical_cov = np.cov(joint_psi, rowvar=False, ddof=1)
    mean_cov = joint_cov.mean(axis=0)
    cov_rel_error = np.linalg.norm(mean_cov - empirical_cov) / np.linalg.norm(
        empirical_cov
    )
    if cov_rel_error > 0.30:
        raise RuntimeError(
            f"Joint sandwich and Monte Carlo covariance disagree: "
            f"relative error={cov_rel_error:.3f}"
        )

    contrast = np.array([1.0, -1.0])
    contrast_truth = float(contrast @ PSI_TRUE)
    contrast_hat = joint_psi @ contrast
    contrast_se = np.sqrt(np.einsum("i,sij,j->s", contrast, joint_cov, contrast))
    contrast_empirical_sd = contrast_hat.std(ddof=1)
    contrast_se_ratio = contrast_se.mean() / contrast_empirical_sd
    contrast_lo = contrast_hat - z * contrast_se
    contrast_hi = contrast_hat + z * contrast_se
    contrast_coverage = np.mean(
        (contrast_lo <= contrast_truth) & (contrast_truth <= contrast_hi)
    )
    contrast_left = np.mean(contrast_truth < contrast_lo)
    contrast_right = np.mean(contrast_truth > contrast_hi)
    if not 0.80 <= contrast_se_ratio <= 1.20:
        raise RuntimeError(
            f"Dynamic DML contrast SE calibration failed: ratio={contrast_se_ratio:.3f}"
        )
    if not 0.90 <= contrast_coverage <= 0.99:
        raise RuntimeError(
            f"Dynamic DML contrast coverage failed: coverage={contrast_coverage:.3f}"
        )
    if max(contrast_left, contrast_right) > 0.06:
        raise RuntimeError(
            f"Dynamic DML contrast tail coverage failed: "
            f"left={contrast_left:.3f}, right={contrast_right:.3f}"
        )

    naive_bias_2 = abs(
        data["results"]["naive_ols"]["psi"][i, :, 1].mean() - PSI_TRUE[1]
    )
    if naive_bias_2 < 0.50:
        raise RuntimeError("Naive longitudinal-bias demonstration is too weak")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
LABELS = {
    "naive_ols": "Naive OLS",
    "msm_iptw": "IPTW MSM (naive SE)",
    "dynamic_dml": "Dynamic DML",
}
COLOR_MAP = {
    "naive_ols": COLORS["red"],
    "msm_iptw": COLORS["orange"],
    "dynamic_dml": COLORS["blue"],
}


def make_figure(summary):
    """Two-panel figure: bias and coverage vs n, for psi_2 (psi_1 in supplement)."""
    n_grid = summary["n_grid"]
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    for method in ["naive_ols", "msm_iptw", "dynamic_dml"]:
        bias_2 = summary[method]["bias"][:, 1]  # psi_2 bias
        rmse_2 = summary[method]["rmse"][:, 1]  # psi_2 RMSE
        cov_2 = summary[method]["coverage"][:, 1]
        axes[0].plot(
            n_grid,
            np.abs(bias_2),
            marker="o",
            label=LABELS[method],
            color=COLOR_MAP[method],
        )
        axes[1].plot(
            n_grid, cov_2, marker="o", label=LABELS[method], color=COLOR_MAP[method]
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"sample size $n$")
    axes[0].set_ylabel(r"$|\mathrm{bias}(\hat\psi_2)|$")
    axes[0].set_title(r"Second-visit effect bias")
    axes[0].legend(frameon=False)

    axes[1].axhline(CI_LEVEL, **BENCH_STYLE, label="nominal 95%")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"sample size $n$")
    axes[1].set_ylabel(r"95\% CI coverage of $\psi_2$")
    axes[1].set_title(r"Second-visit effect coverage")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(frameon=False, loc="lower right")

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "dynamic_dml_snmm_coverage.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out}")


def make_table(summary):
    """Consolidated results table at the largest n in the grid."""
    n_target = N_GRID[-1]
    idx = N_GRID.index(n_target)
    rows = []
    for method in ["naive_ols", "msm_iptw", "dynamic_dml"]:
        bias = summary[method]["bias"][idx]
        rmse = summary[method]["rmse"][idx]
        cov = summary[method]["coverage"][idx]
        rows.append(
            (
                LABELS[method],
                bias[0],
                rmse[0],
                cov[0],
                bias[1],
                rmse[1],
                cov[1],
            )
        )

    tex = []
    tex.append(r"\begin{tabular}{lrrrrrr}")
    tex.append(r"\toprule")
    tex.append(
        r" & \multicolumn{3}{c}{$\psi_1^* = " + f"{PSI_TRUE[0]:.2f}" + r"$}"
        r" & \multicolumn{3}{c}{$\psi_2^* = " + f"{PSI_TRUE[1]:.2f}" + r"$} \\"
    )
    tex.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    tex.append(r"Method & Bias & RMSE & Cov & Bias & RMSE & Cov \\")
    tex.append(r"\midrule")
    for r in rows:
        tex.append(
            f"{r[0]} & {r[1]:+.3f} & {r[2]:.3f} & {r[3]:.2f}"
            f" & {r[4]:+.3f} & {r[5]:.3f} & {r[6]:.2f} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    out = os.path.join(OUTPUT_DIR, "dynamic_dml_snmm_results.tex")
    with open(out, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"  Table saved: {out}")


def joint_inference_summary(data):
    """Full-covariance and linear-contrast repeated-sampling diagnostics."""
    joint = data["joint_inference"]
    psi = joint["psi"]
    cov = joint["cov"]
    empirical_cov = np.cov(psi, rowvar=False, ddof=1)
    mean_cov = cov.mean(axis=0)
    contrast = np.array([1.0, -1.0])
    contrast_truth = float(contrast @ PSI_TRUE)
    contrast_hat = psi @ contrast
    contrast_se = np.sqrt(np.einsum("i,sij,j->s", contrast, cov, contrast))
    # The same contrast under a sandwich that drops the cross-stage block, i.e.
    # that treats the two backward regressions as if they were fit on independent
    # samples. This is the comparison the chapter's warning is about, and it is
    # only meaningful stated as a magnitude.
    contrast_se_diag = np.sqrt(cov[:, 0, 0] + cov[:, 1, 1])
    z = norm.ppf(0.5 + CI_LEVEL / 2)
    lo = contrast_hat - z * contrast_se
    hi = contrast_hat + z * contrast_se
    lo_d = contrast_hat - z * contrast_se_diag
    hi_d = contrast_hat + z * contrast_se_diag
    emp_sd = contrast_hat.std(ddof=1)
    corr = mean_cov[0, 1] / np.sqrt(mean_cov[0, 0] * mean_cov[1, 1])
    return {
        "empirical_cov": empirical_cov,
        "mean_cov": mean_cov,
        "cov_rel_error": np.linalg.norm(mean_cov - empirical_cov)
        / np.linalg.norm(empirical_cov),
        "cross_stage_corr": float(corr),
        "contrast_bias": float(contrast_hat.mean() - contrast_truth),
        "contrast_empirical_sd": float(emp_sd),
        "contrast_mean_se": float(contrast_se.mean()),
        "contrast_se_ratio": float(contrast_se.mean() / emp_sd),
        "contrast_coverage": float(
            np.mean((lo <= contrast_truth) & (contrast_truth <= hi))
        ),
        "contrast_left": float(np.mean(contrast_truth < lo)),
        "contrast_right": float(np.mean(contrast_truth > hi)),
        "contrast_mean_se_diag": float(contrast_se_diag.mean()),
        "contrast_se_ratio_diag": float(contrast_se_diag.mean() / emp_sd),
        "contrast_coverage_diag": float(
            np.mean((lo_d <= contrast_truth) & (contrast_truth <= hi_d))
        ),
        "diag_vs_full_se": float(contrast_se_diag.mean() / contrast_se.mean()),
    }


def make_joint_inference_table(data):
    """Write the full covariance and psi_1 - psi_2 contrast diagnostics."""
    d = joint_inference_summary(data)
    out = os.path.join(OUTPUT_DIR, "dynamic_dml_snmm_joint_inference.tex")
    tex = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Quantity & Sandwich estimate & Monte Carlo target \\",
        r"\midrule",
        (
            r"$\operatorname{Var}(\hat\psi_1)$"
            f" & {d['mean_cov'][0, 0]:.6f} & {d['empirical_cov'][0, 0]:.6f} \\\\"
        ),
        (
            r"$\operatorname{Cov}(\hat\psi_1,\hat\psi_2)$"
            f" & {d['mean_cov'][0, 1]:+.6f} & {d['empirical_cov'][0, 1]:+.6f} \\\\"
        ),
        (
            r"$\operatorname{Var}(\hat\psi_2)$"
            f" & {d['mean_cov'][1, 1]:.6f} & {d['empirical_cov'][1, 1]:.6f} \\\\"
        ),
        (
            r"Cross-stage correlation"
            f" & {d['cross_stage_corr']:+.3f} & "
            f"{d['empirical_cov'][0, 1] / np.sqrt(d['empirical_cov'][0, 0] * d['empirical_cov'][1, 1]):+.3f} \\\\"
        ),
        r"\addlinespace",
        (
            r"SE$(\hat\psi_1-\hat\psi_2)$, joint sandwich"
            f" & {d['contrast_mean_se']:.4f} & {d['contrast_empirical_sd']:.4f} \\\\"
        ),
        (
            r"SE$(\hat\psi_1-\hat\psi_2)$, cross-stage block dropped"
            f" & {d['contrast_mean_se_diag']:.4f} & {d['contrast_empirical_sd']:.4f} \\\\"
        ),
        (
            r"95\% CI coverage, joint sandwich"
            f" & {d['contrast_coverage']:.3f} & {CI_LEVEL:.3f} \\\\"
        ),
        (
            r"95\% CI coverage, block dropped"
            f" & {d['contrast_coverage_diag']:.3f} & {CI_LEVEL:.3f} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(out, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"  Joint-inference table saved: {out}")


def print_stdout(summary, data):
    """Tabular stdout: per-method bias / RMSE / coverage at each n."""
    print()
    print("=" * 70)
    print("  Dynamic DML in a stylized Fast Track home-visiting SNMM")
    print("=" * 70)
    print(f"  True parameters: psi_1* = {PSI_TRUE[0]:.4f}, psi_2* = {PSI_TRUE[1]:.4f}")
    print(
        f"  State dim p = {P_STATE}, sparsity s = {S_SPARSE}, "
        f"n_seeds = {N_SEEDS}, K_folds = {K_FOLDS}"
    )
    print("  T_1 and T_2 denote additional home visits; Y is a terminal outcome.")
    print(
        f"  DGP: || B ||_op = {B_OPNORM}, || gamma || = {GAMMA_NORM}, "
        f"|| alpha || = {ALPHA_NORM}"
    )
    print(f"  Sample sizes: {N_GRID}")
    print(
        "  IPTW-MSM intervals use a naive fixed-weight sandwich; they do not "
        "propagate propensity estimation or trimming."
    )
    print()

    for method in ["naive_ols", "msm_iptw", "dynamic_dml"]:
        print(f"  --- {LABELS[method]} ---")
        print(
            f"  {'n':>6} {'bias(psi_1)':>14} {'rmse(psi_1)':>14} {'cov(psi_1)':>12}"
            f" {'bias(psi_2)':>14} {'rmse(psi_2)':>14} {'cov(psi_2)':>12}"
        )
        for i, n in enumerate(N_GRID):
            b = summary[method]["bias"][i]
            r = summary[method]["rmse"][i]
            c = summary[method]["coverage"][i]
            print(
                f"  {n:>6d} {b[0]:>14.4f} {r[0]:>14.4f} {c[0]:>12.3f}"
                f" {b[1]:>14.4f} {r[1]:>14.4f} {c[1]:>12.3f}"
            )
        print()

    d = joint_inference_summary(data)
    print("  --- Dynamic DML full joint sandwich at n=4000 ---")
    print("  Mean analytic covariance:")
    print(f"    [{d['mean_cov'][0, 0]:.6f}, {d['mean_cov'][0, 1]:+.6f}]")
    print(f"    [{d['mean_cov'][1, 0]:+.6f}, {d['mean_cov'][1, 1]:.6f}]")
    print("  Monte Carlo covariance across repeated datasets (ddof=1):")
    print(f"    [{d['empirical_cov'][0, 0]:.6f}, {d['empirical_cov'][0, 1]:+.6f}]")
    print(f"    [{d['empirical_cov'][1, 0]:+.6f}, {d['empirical_cov'][1, 1]:.6f}]")
    print(f"  Relative covariance error: {d['cov_rel_error']:.3f}")
    print(f"  Cross-stage correlation (sandwich): {d['cross_stage_corr']:+.3f}")
    print(
        "  Contrast psi_1 - psi_2: "
        f"bias={d['contrast_bias']:+.4f}, "
        f"mean analytic SE={d['contrast_mean_se']:.4f}, "
        f"Monte Carlo SD={d['contrast_empirical_sd']:.4f}, "
        f"SE ratio={d['contrast_se_ratio']:.3f}"
    )
    print(
        "  Same contrast with the cross-stage block dropped: "
        f"mean SE={d['contrast_mean_se_diag']:.4f}, "
        f"SE ratio={d['contrast_se_ratio_diag']:.3f}, "
        f"coverage={d['contrast_coverage_diag']:.3f}, "
        f"SE relative to joint={d['diag_vs_full_se']:.3f}"
    )
    print(
        "  Contrast 95% CI: "
        f"coverage={d['contrast_coverage']:.3f}, "
        f"tail misses={d['contrast_left']:.3f}/{d['contrast_right']:.3f}"
    )
    print()

    print("  Output files:")
    print("    ", os.path.join(OUTPUT_DIR, "dynamic_dml_snmm_coverage.png"))
    print("    ", os.path.join(OUTPUT_DIR, "dynamic_dml_snmm_results.tex"))
    print("    ", os.path.join(OUTPUT_DIR, "dynamic_dml_snmm_joint_inference.tex"))


def generate_outputs(data):
    summary = summarize(data)
    print_stdout(summary, data)
    make_figure(summary)
    make_table(summary)
    make_joint_inference_table(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()

    force = parse_force_set(args)

    if args.plots_only:
        # Load from cache without forcing
        data = compute_data(force=set())
    else:
        data = compute_data(force=force)

    validate_results(data)
    if not args.data_only:
        generate_outputs(data)


if __name__ == "__main__":
    main()
