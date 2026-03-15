"""
Brock-Mirman FVI vs FQI: Fitted Value Iteration, Fitted Q-Iteration,
Oracle-basis FQI, and Nonlinear Least Squares FQI.
Chapter 3 — Demonstrates that FQI's failure on Brock-Mirman is a basis
representability problem, not an algorithmic one. Oracle-basis FQI (known alpha)
and NLLS-FQI (estimated alpha) both converge, recovering the structural parameter.
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, FIG_SINGLE, FIG_DOUBLE
from sims.sim_cache import load_results, save_results, add_cache_args
apply_style()

# ============================================================================
# Parameters
# ============================================================================

ALPHA_PROD = 0.36       # capital share
BETA = 0.96             # discount factor
Z_VALS = np.array([0.9, 1.1])
PI_TRANS = np.array([[0.8, 0.2],
                     [0.2, 0.8]])

N_K = 50                # capital grid points
N_Z = len(Z_VALS)
N_S = N_K * N_Z         # 100 states
N_A = N_K               # 50 actions

# Basis: log(k) + polynomial features interacted with productivity indicators.
# phi(k,z) = [1, log(k), k^1, k^2, k^3] x [1_{z=z_low}, 1_{z=z_high}] in R^10.
# Including log(k) lets FVI represent V*(k,z) = A_z + B*log(k) exactly.
POLY_POWERS = [0, None, 1, 2, 3]  # None = log(k), rest = k^p
N_FEAT_PER_Z = len(POLY_POWERS)
N_BASIS = N_FEAT_PER_Z * N_Z  # 10 features total

MAX_ITER_VI  = 5000
MAX_ITER_FVI = 3000
MAX_ITER_FQI = 3000
TOL_VI  = 1e-10
TOL_FIT = 1e-6

SEED = 42
OUTPUT_DIR = os.path.dirname(__file__)

np.random.seed(SEED)

# ============================================================================
# Model helpers (replicated from bm_illustrated.py)
# ============================================================================

def build_grid(n_k):
    k_ss_high = (ALPHA_PROD * BETA * Z_VALS.max()) ** (1 / (1 - ALPHA_PROD))
    return np.linspace(0.01, 1.5 * k_ss_high, n_k)


def build_reward_and_transitions(k_grid):
    n_k = len(k_grid)
    n_z = len(Z_VALS)
    n_s = n_k * n_z
    n_a = n_k

    R = np.full((n_s, n_a), -np.inf)
    P = np.zeros((n_s, n_a, n_s))

    for ik in range(n_k):
        for iz in range(n_z):
            s = ik * n_z + iz
            output = Z_VALS[iz] * k_grid[ik] ** ALPHA_PROD
            for ia in range(n_a):
                c = output - k_grid[ia]
                if c > 1e-12:
                    R[s, ia] = np.log(c)
                    for iz_next in range(n_z):
                        s_next = ia * n_z + iz_next
                        P[s, ia, s_next] = PI_TRANS[iz, iz_next]

    return R, P


def closed_form_policy(k_grid):
    n_k = len(k_grid)
    n_z = len(Z_VALS)
    policy = np.zeros(n_k * n_z, dtype=int)
    for ik in range(n_k):
        for iz in range(n_z):
            s = ik * n_z + iz
            k_next = ALPHA_PROD * BETA * Z_VALS[iz] * k_grid[ik] ** ALPHA_PROD
            policy[s] = np.argmin(np.abs(k_grid - k_next))
    return policy


# ============================================================================
# Basis functions
# ============================================================================

def _feat(k, power, k_max):
    """Evaluate a single basis function at k."""
    if power is None:
        return np.log(k + 1e-12)  # log(k) feature
    elif power == 0:
        return np.ones_like(k)
    else:
        return (k / k_max) ** power  # normalize before forming powers


def build_feature_matrix(k_grid):
    """
    Log-polynomial features of capital interacted with productivity indicator.
    phi(k,z) = [1, log(k), (k/kmax)^1, (k/kmax)^2, (k/kmax)^3]
               x [1_{z=z_low}, 1_{z=z_high}]  in  R^10.
    Returns Phi: (N_S, N_BASIS) matrix.
    """
    k_max = k_grid[-1]
    Phi = np.zeros((N_S, N_BASIS))
    for ik in range(N_K):
        for iz in range(N_Z):
            s = ik * N_Z + iz
            for j, power in enumerate(POLY_POWERS):
                col = iz * N_FEAT_PER_Z + j
                Phi[s, col] = _feat(k_grid[ik], power, k_max)
    return Phi


# ============================================================================
# Exact Value Iteration (ground truth)
# ============================================================================

def value_iteration(R, P, gamma=BETA, tol=TOL_VI, max_iter=MAX_ITER_VI):
    n_s, n_a = R.shape
    V = np.zeros(n_s)
    for it in range(max_iter):
        Q = np.where(R > -1e30, R + gamma * (P @ V), -1e30)
        V_new = np.max(Q, axis=1)
        err = np.max(np.abs(V_new - V))
        V = V_new
        if err < tol:
            break
    policy = np.argmax(np.where(R > -1e30, R + gamma * (P @ V), -1e30), axis=1)
    return V, policy


# ============================================================================
# Linear FVI
# ============================================================================

def linear_fvi(R, P, Phi, V_star, gamma=BETA, max_iter=MAX_ITER_FVI, tol=TOL_FIT):
    """
    Fitted Value Iteration with linear function approximation.
      theta_{k+1} = (Phi^T Phi)^{-1} Phi^T V_target^k
    where V_target^k(s) = max_a [R(s,a) + gamma * (P @ Phi theta_k)(s,a)].
    Uses the full model (expectation, no sampling noise).
    """
    n_s, n_basis = Phi.shape
    PtP_inv = np.linalg.inv(Phi.T @ Phi)
    # P_Phi[s, a, :] = sum_{s'} P(s,a,s') * Phi(s', :)  shape: (n_s, n_a, n_basis)
    P_Phi = np.tensordot(P, Phi, axes=([2], [0]))

    theta = np.zeros(n_basis)
    errors_to_vstar = []

    for k in range(max_iter):
        EV = P_Phi @ theta                     # (n_s, n_a): E[V(s') | s, a]
        Q  = np.where(R > -1e30, R + gamma * EV, -1e30)
        V_target = np.max(Q, axis=1)
        theta_new = PtP_inv @ (Phi.T @ V_target)
        change = np.max(np.abs(Phi @ theta_new - Phi @ theta))
        errors_to_vstar.append(np.max(np.abs(Phi @ theta_new - V_star)))
        theta = theta_new
        if change < tol:
            break

    V_fvi = Phi @ theta
    policy = np.argmax(np.where(R > -1e30, R + gamma * P_Phi @ theta, -1e30), axis=1)
    return theta, V_fvi, policy, errors_to_vstar, k + 1


# ============================================================================
# Linear FQI
# ============================================================================

def linear_fqi(R, P, Phi, V_star, gamma=BETA, max_iter=MAX_ITER_FQI, tol=TOL_FIT):
    """
    Fitted Q-Iteration with per-action linear function approximation.
      theta_a^{k+1} = (Phi_a^T Phi_a + lam I)^{-1} Phi_a^T Q_target_a^k
    where Q_target_a^k(s) = R(s,a) + gamma * sum_{s'} P(s,a,s') max_{a'} Q_k(s',a').
    Uses the full model (expectation, no sampling noise).
    """
    n_s, n_basis = Phi.shape
    n_a = R.shape[1]
    lam = 1e-8  # ridge regularization for numerical stability

    # Theta[a, :] = theta_a; initialize at zero
    Theta = np.zeros((n_a, n_basis))
    # Precompute feasibility mask
    feas = R > -1e30  # (n_s, n_a)

    errors_to_vstar = []

    for k in range(max_iter):
        # Current Q-function and implied V
        Q_k = Phi @ Theta.T         # (n_s, n_a)
        Q_k = np.where(feas, Q_k, -1e30)
        V_k = np.max(Q_k, axis=1)  # (n_s,)

        Theta_new = np.zeros_like(Theta)
        for a in range(n_a):
            feasible_s = feas[:, a]
            if feasible_s.sum() < n_basis:
                Theta_new[a] = Theta[a]
                continue
            EV = P[:, a, :] @ V_k             # (n_s,): E[V_k(s') | s, a]
            Q_target = R[:, a] + gamma * EV   # (n_s,): only valid where feasible
            Phi_f = Phi[feasible_s]
            y_f   = Q_target[feasible_s]
            A = Phi_f.T @ Phi_f + lam * np.eye(n_basis)
            b = Phi_f.T @ y_f
            Theta_new[a] = np.linalg.solve(A, b)

        Q_new = np.where(feas, Phi @ Theta_new.T, -1e30)
        V_new = np.max(Q_new, axis=1)
        change = np.max(np.abs(V_new - V_k))
        errors_to_vstar.append(np.max(np.abs(V_new - V_star)))
        Theta = Theta_new
        if change < tol:
            break

    Q_final = np.where(feas, Phi @ Theta.T, -1e30)
    V_fqi   = np.max(Q_final, axis=1)
    policy  = np.argmax(Q_final, axis=1)
    return Theta, V_fqi, policy, errors_to_vstar, k + 1


# ============================================================================
# Oracle-basis FQI (known alpha, correct log-consumption feature)
# ============================================================================

def oracle_fqi(R, P, k_grid, V_star, gamma=BETA, max_iter=MAX_ITER_FQI, tol=TOL_FIT):
    """
    FQI with oracle log-consumption basis:
      phi_a(k,z) = [1_{z=z_low}, 1_{z=z_high}, log(z*k^alpha - k')]
    where alpha is the known production exponent.  Standard OLS per action.
    Demonstrates that FQI convergence is purely a basis choice problem.
    """
    n_s, n_a = R.shape
    feas = R > -1e30
    iz_all = np.arange(n_s) % N_Z
    ik_all = np.arange(n_s) // N_Z
    lam = 1e-8

    # Precompute log-consumption for all (s, a) using known alpha
    k_power = k_grid ** ALPHA_PROD
    output_flat = Z_VALS[iz_all] * k_power[ik_all]          # (N_S,)
    consumption = output_flat[:, None] - k_grid[None, :]     # (N_S, N_A)
    valid = consumption > 1e-12
    log_c = np.full_like(consumption, 0.0)
    log_c[valid] = np.log(consumption[valid])
    mask = feas & valid

    Theta = np.zeros((n_a, 3))  # (theta_zlow, theta_zhigh, theta_logc)
    errors_to_vstar = []

    for iteration in range(max_iter):
        # Q_k[s, a] = Theta[a, iz_s] + Theta[a, 2] * log_c[s, a]
        intercepts = Theta[:, iz_all].T                      # (N_S, N_A)
        slopes = Theta[:, 2]                                  # (N_A,)
        Q_k = np.where(mask, intercepts + slopes[None, :] * log_c, -1e30)
        V_k = np.max(Q_k, axis=1)

        # Targets: y(s,a) = R(s,a) + gamma * E[V_k(s') | s, a]
        EV = np.einsum('ijk,k->ij', P, V_k)
        targets = R + gamma * EV

        # Per-action OLS
        Theta_new = np.zeros_like(Theta)
        for a in range(n_a):
            s_m = np.where(mask[:, a])[0]
            if len(s_m) < 3:
                Theta_new[a] = Theta[a]
                continue
            Phi_a = np.zeros((len(s_m), 3))
            Phi_a[:, 0] = (iz_all[s_m] == 0).astype(float)
            Phi_a[:, 1] = (iz_all[s_m] == 1).astype(float)
            Phi_a[:, 2] = log_c[s_m, a]
            y_a = targets[s_m, a]
            A = Phi_a.T @ Phi_a + lam * np.eye(3)
            Theta_new[a] = np.linalg.solve(A, Phi_a.T @ y_a)

        # Updated V
        intercepts_new = Theta_new[:, iz_all].T
        slopes_new = Theta_new[:, 2]
        Q_new = np.where(mask, intercepts_new + slopes_new[None, :] * log_c, -1e30)
        V_new = np.max(Q_new, axis=1)

        change = np.max(np.abs(V_new - V_k))
        errors_to_vstar.append(np.max(np.abs(V_new - V_star)))
        Theta = Theta_new
        if change < tol:
            break

    policy = np.argmax(Q_new, axis=1)
    return Theta, V_new, policy, errors_to_vstar, iteration + 1


# ============================================================================
# NLLS-FQI (estimate alpha from data via concentrated least squares)
# ============================================================================

def nlls_fqi(R, P, k_grid, V_star, gamma=BETA, max_iter=MAX_ITER_FQI, tol=TOL_FIT):
    """
    FQI with nonlinear least squares:
      Q(k,z; theta_a) = theta_0(z) + theta_1 * log(z * k^{alpha} - k')
    where alpha is estimated from data.  Uses concentrated least squares:
    for each candidate alpha, solve OLS per action for (theta_0_zlow,
    theta_0_zhigh, theta_1), then optimize alpha to minimize total RSS.
    """
    n_s, n_a = R.shape
    feas = R > -1e30
    iz_all = np.arange(n_s) % N_Z
    ik_all = np.arange(n_s) // N_Z
    lam = 1e-8

    Theta = np.zeros((n_a, 3))
    alpha_est = 0.5  # deliberately wrong initial guess
    alpha_trajectory = [alpha_est]
    errors_to_vstar = []

    def _compute_log_c(alpha_val):
        """Log-consumption for all (s, a) at given alpha."""
        k_pow = k_grid ** alpha_val
        out = Z_VALS[iz_all] * k_pow[ik_all]
        cons = out[:, None] - k_grid[None, :]
        v = cons > 1e-12
        lc = np.full_like(cons, 0.0)
        lc[v] = np.log(cons[v])
        return lc, v

    def _eval_Q(Theta_val, log_c_val, valid_val):
        """Evaluate Q-function from weights and features."""
        intercepts = Theta_val[:, iz_all].T
        slopes = Theta_val[:, 2]
        return np.where(feas & valid_val,
                        intercepts + slopes[None, :] * log_c_val, -1e30)

    def _fit_per_action(targets_val, log_c_val, valid_val):
        """OLS per action given log_c features.  Returns Theta (N_A, 3)."""
        Th = np.zeros((n_a, 3))
        for a in range(n_a):
            m = feas[:, a] & valid_val[:, a]
            s_m = np.where(m)[0]
            if len(s_m) < 3:
                continue
            Phi_a = np.zeros((len(s_m), 3))
            Phi_a[:, 0] = (iz_all[s_m] == 0).astype(float)
            Phi_a[:, 1] = (iz_all[s_m] == 1).astype(float)
            Phi_a[:, 2] = log_c_val[s_m, a]
            y_a = targets_val[s_m, a]
            A = Phi_a.T @ Phi_a + lam * np.eye(3)
            Th[a] = np.linalg.solve(A, Phi_a.T @ y_a)
        return Th

    for iteration in range(max_iter):
        # Current Q and V
        log_c, valid = _compute_log_c(alpha_est)
        Q_k = _eval_Q(Theta, log_c, valid)
        V_k = np.max(Q_k, axis=1)

        # Targets
        EV = np.einsum('ijk,k->ij', P, V_k)
        targets = R + gamma * EV

        # Profile optimization: find alpha minimizing penalized RSS.
        # Penalty: each observation that is feasible under true model but
        # infeasible under alpha_candidate contributes mean(target^2) to RSS,
        # preventing the optimizer from exploiting observation dropout.
        target_var = np.mean(targets[feas] ** 2)

        def profile_rss(alpha_candidate):
            lc, vld = _compute_log_c(alpha_candidate)
            total_rss = 0.0
            for a in range(n_a):
                s_feas = np.where(feas[:, a])[0]
                if len(s_feas) < 3:
                    continue
                m = feas[:, a] & vld[:, a]
                n_lost = len(s_feas) - m.sum()
                total_rss += n_lost * target_var
                s_m = np.where(m)[0]
                if len(s_m) < 3:
                    continue
                Phi_a = np.zeros((len(s_m), 3))
                Phi_a[:, 0] = (iz_all[s_m] == 0).astype(float)
                Phi_a[:, 1] = (iz_all[s_m] == 1).astype(float)
                Phi_a[:, 2] = lc[s_m, a]
                y_a = targets[s_m, a]
                theta_a = np.linalg.lstsq(Phi_a, y_a, rcond=None)[0]
                total_rss += np.sum((Phi_a @ theta_a - y_a) ** 2)
            return total_rss

        result = minimize_scalar(profile_rss, bounds=(0.05, 0.95),
                                 method='bounded', options={'xatol': 1e-6})
        alpha_est = result.x
        alpha_trajectory.append(alpha_est)

        # Refit at optimal alpha
        log_c, valid = _compute_log_c(alpha_est)
        Theta_new = _fit_per_action(targets, log_c, valid)

        # New Q and V
        Q_new = _eval_Q(Theta_new, log_c, valid)
        V_new = np.max(Q_new, axis=1)

        change = np.max(np.abs(V_new - V_k))
        errors_to_vstar.append(np.max(np.abs(V_new - V_star)))
        Theta = Theta_new
        if change < tol:
            break

    policy = np.argmax(Q_new, axis=1)
    return Theta, V_new, policy, errors_to_vstar, iteration + 1, \
        np.array(alpha_trajectory)


# ============================================================================
# Caching
# ============================================================================

CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
SCRIPT_NAME = 'bm_fvi_fqi'
CONFIG = {
    'alpha_prod': ALPHA_PROD, 'beta': BETA,
    'z_vals': Z_VALS.tolist(), 'pi_trans': PI_TRANS.tolist(),
    'n_k': N_K, 'n_z': N_Z, 'n_basis': N_BASIS,
    'poly_powers': [p if p is not None else 'log' for p in POLY_POWERS],
    'max_iter_vi': MAX_ITER_VI, 'max_iter_fvi': MAX_ITER_FVI,
    'max_iter_fqi': MAX_ITER_FQI, 'tol_vi': TOL_VI, 'tol_fit': TOL_FIT,
    'seed': SEED, 'version': 1,
}


# ============================================================================
# Computation
# ============================================================================

def compute_data():
    """Run all computation phases and return results dict."""
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    print("Building model ...")
    k_grid = build_grid(N_K)
    R, P = build_reward_and_transitions(k_grid)
    cf_pol = closed_form_policy(k_grid)
    Phi = build_feature_matrix(k_grid)
    print(f"  k_grid: [{k_grid[0]:.4f}, {k_grid[-1]:.4f}]")
    print(f"  Feasible (s,a) pairs: {(R > -1e30).sum()} / {N_S * N_A}")
    print(f"  Phi shape: {Phi.shape}, rank: {np.linalg.matrix_rank(Phi)}")
    print()

    # Phase 1: Exact VI
    print("Phase 1: Exact Value Iteration (ground truth)")
    print("-" * 50)
    t0 = time.perf_counter()
    V_star, pol_vi = value_iteration(R, P)
    t_vi = time.perf_counter() - t0
    pol_agr_vi = np.mean(pol_vi == cf_pol) * 100
    print(f"  Converged. Time: {t_vi:.3f}s")
    print(f"  Policy agreement with closed-form: {pol_agr_vi:.1f}%")
    print(f"  V* range: [{V_star.min():.4f}, {V_star.max():.4f}]")
    B_theory = ALPHA_PROD / (1 - ALPHA_PROD * BETA)
    print(f"  Theoretical B = alpha/(1-alpha*beta) = {B_theory:.4f}")
    print()

    # Best linear approximation
    PtP_inv = np.linalg.inv(Phi.T @ Phi)
    theta_proj = PtP_inv @ (Phi.T @ V_star)
    V_proj = Phi @ theta_proj
    approx_err_vstar = np.max(np.abs(V_proj - V_star))
    print(f"  Best linear approx of V* in span(Phi): ||Phi theta - V*||_inf = {approx_err_vstar:.6f}")
    print(f"  (zero would mean V* is exactly representable; our log(k)+poly basis")
    print(f"   captures the dominant log(k) structure of BM value function)")
    print()

    # Phase 2: Linear FVI
    print("Phase 2: Linear Fitted Value Iteration")
    print("-" * 50)
    t0 = time.perf_counter()
    theta_V, V_fvi, pol_fvi, errors_fvi, n_iter_fvi = linear_fvi(R, P, Phi, V_star)
    t_fvi = time.perf_counter() - t0
    err_fvi_inf = np.max(np.abs(V_fvi - V_star))
    err_fvi_rms = np.sqrt(np.mean((V_fvi - V_star) ** 2))
    pol_agr_fvi = np.mean(pol_fvi == cf_pol) * 100
    converged_fvi = (n_iter_fvi < MAX_ITER_FVI)
    print(f"  {'Converged' if converged_fvi else 'Reached max iterations'} "
          f"in {n_iter_fvi} iterations. Time: {t_fvi:.3f}s")
    print(f"  ||V_FVI - V*||_inf = {err_fvi_inf:.6f}")
    print(f"  ||V_FVI - V*||_rms = {err_fvi_rms:.6f}")
    print(f"  Policy agreement with closed-form: {pol_agr_fvi:.1f}%")
    print()

    # Phase 3: Linear FQI
    print("Phase 3: Linear Fitted Q-Iteration")
    print("-" * 50)
    t0 = time.perf_counter()
    Theta_Q, V_fqi, pol_fqi, errors_fqi, n_iter_fqi = linear_fqi(R, P, Phi, V_star)
    t_fqi = time.perf_counter() - t0
    err_fqi_inf = np.max(np.abs(V_fqi - V_star))
    err_fqi_rms = np.sqrt(np.mean((V_fqi - V_star) ** 2))
    pol_agr_fqi = np.mean(pol_fqi == cf_pol) * 100
    converged_fqi = (n_iter_fqi < MAX_ITER_FQI)
    print(f"  {'Converged' if converged_fqi else 'Reached max iterations'} "
          f"in {n_iter_fqi} iterations. Time: {t_fqi:.3f}s")
    print(f"  ||V_FQI - V*||_inf = {err_fqi_inf:.6f}")
    print(f"  ||V_FQI - V*||_rms = {err_fqi_rms:.6f}")
    print(f"  Policy agreement with closed-form: {pol_agr_fqi:.1f}%")
    print()

    # Phase 3b: Oracle-basis FQI
    print("Phase 3b: Oracle-basis FQI (known alpha, log-consumption feature)")
    print("-" * 50)
    t0 = time.perf_counter()
    Theta_ora, V_ora, pol_ora, errors_ora, n_iter_ora = oracle_fqi(
        R, P, k_grid, V_star)
    t_ora = time.perf_counter() - t0
    err_ora_inf = np.max(np.abs(V_ora - V_star))
    err_ora_rms = np.sqrt(np.mean((V_ora - V_star) ** 2))
    pol_agr_ora = np.mean(pol_ora == cf_pol) * 100
    converged_ora = (n_iter_ora < MAX_ITER_FQI)
    print(f"  {'Converged' if converged_ora else 'Reached max iterations'} "
          f"in {n_iter_ora} iterations. Time: {t_ora:.3f}s")
    print(f"  ||V_Oracle - V*||_inf = {err_ora_inf:.6f}")
    print(f"  ||V_Oracle - V*||_rms = {err_ora_rms:.6f}")
    print(f"  Policy agreement with closed-form: {pol_agr_ora:.1f}%")
    print()

    # Phase 3c: NLLS-FQI
    print("Phase 3c: NLLS-FQI (estimate alpha via concentrated least squares)")
    print("-" * 50)
    t0 = time.perf_counter()
    Theta_nls, V_nls, pol_nls, errors_nls, n_iter_nls, alpha_traj = nlls_fqi(
        R, P, k_grid, V_star)
    t_nls = time.perf_counter() - t0
    err_nls_inf = np.max(np.abs(V_nls - V_star))
    err_nls_rms = np.sqrt(np.mean((V_nls - V_star) ** 2))
    pol_agr_nls = np.mean(pol_nls == cf_pol) * 100
    converged_nls = (n_iter_nls < MAX_ITER_FQI)
    print(f"  {'Converged' if converged_nls else 'Reached max iterations'} "
          f"in {n_iter_nls} iterations. Time: {t_nls:.3f}s")
    print(f"  ||V_NLLS - V*||_inf = {err_nls_inf:.6f}")
    print(f"  ||V_NLLS - V*||_rms = {err_nls_rms:.6f}")
    print(f"  Policy agreement with closed-form: {pol_agr_nls:.1f}%")
    print(f"  Recovered alpha: {alpha_traj[-1]:.6f} (true: {ALPHA_PROD})")
    print(f"  Alpha trajectory: "
          f"{' -> '.join(f'{a:.4f}' for a in alpha_traj[:6])}"
          f"{'...' if len(alpha_traj) > 6 else ''}")
    print()

    data = {
        'k_grid': k_grid, 'cf_pol': cf_pol, 'Phi': Phi,
        'V_star': V_star, 'pol_vi': pol_vi, 't_vi': t_vi,
        'pol_agr_vi': pol_agr_vi, 'approx_err_vstar': approx_err_vstar,
        # FVI
        'theta_V': theta_V, 'V_fvi': V_fvi, 'pol_fvi': pol_fvi,
        'errors_fvi': errors_fvi, 'n_iter_fvi': n_iter_fvi, 't_fvi': t_fvi,
        'err_fvi_inf': err_fvi_inf, 'err_fvi_rms': err_fvi_rms,
        'pol_agr_fvi': pol_agr_fvi, 'converged_fvi': converged_fvi,
        # FQI
        'Theta_Q': Theta_Q, 'V_fqi': V_fqi, 'pol_fqi': pol_fqi,
        'errors_fqi': errors_fqi, 'n_iter_fqi': n_iter_fqi, 't_fqi': t_fqi,
        'err_fqi_inf': err_fqi_inf, 'err_fqi_rms': err_fqi_rms,
        'pol_agr_fqi': pol_agr_fqi, 'converged_fqi': converged_fqi,
        # Oracle FQI
        'V_ora': V_ora, 'pol_ora': pol_ora,
        'errors_ora': errors_ora, 'n_iter_ora': n_iter_ora, 't_ora': t_ora,
        'err_ora_inf': err_ora_inf, 'err_ora_rms': err_ora_rms,
        'pol_agr_ora': pol_agr_ora, 'converged_ora': converged_ora,
        # NLLS FQI
        'V_nls': V_nls, 'pol_nls': pol_nls,
        'errors_nls': errors_nls, 'n_iter_nls': n_iter_nls, 't_nls': t_nls,
        'err_nls_inf': err_nls_inf, 'err_nls_rms': err_nls_rms,
        'pol_agr_nls': pol_agr_nls, 'converged_nls': converged_nls,
        'alpha_traj': alpha_traj,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


# ============================================================================
# Output generation (figures, tables, stdout diagnostics)
# ============================================================================

def generate_outputs(data):
    """Generate all figures, tables, and stdout diagnostics from cached data."""
    # Unpack
    k_grid = data['k_grid']
    Phi = data['Phi']
    V_star = data['V_star']
    theta_V = data['theta_V']
    V_fvi = data['V_fvi']
    Theta_Q = data['Theta_Q']
    V_fqi = data['V_fqi']
    errors_fvi = data['errors_fvi']
    errors_fqi = data['errors_fqi']
    errors_ora = data['errors_ora']
    errors_nls = data['errors_nls']
    alpha_traj = data['alpha_traj']
    n_iter_fvi = data['n_iter_fvi']
    n_iter_fqi = data['n_iter_fqi']
    n_iter_ora = data['n_iter_ora']
    n_iter_nls = data['n_iter_nls']
    t_vi = data['t_vi']
    t_fvi = data['t_fvi']
    t_fqi = data['t_fqi']
    t_ora = data['t_ora']
    t_nls = data['t_nls']
    pol_agr_vi = data['pol_agr_vi']
    pol_agr_fvi = data['pol_agr_fvi']
    pol_agr_fqi = data['pol_agr_fqi']
    pol_agr_ora = data['pol_agr_ora']
    pol_agr_nls = data['pol_agr_nls']
    err_fvi_inf = data['err_fvi_inf']
    err_fvi_rms = data['err_fvi_rms']
    err_fqi_inf = data['err_fqi_inf']
    err_fqi_rms = data['err_fqi_rms']
    err_ora_inf = data['err_ora_inf']
    err_ora_rms = data['err_ora_rms']
    err_nls_inf = data['err_nls_inf']
    err_nls_rms = data['err_nls_rms']
    approx_err_vstar = data['approx_err_vstar']

    # -------------------------------------------------------------------------
    # Stdout diagnostics: FVI weight vector
    # -------------------------------------------------------------------------
    feat_labels = ['1', 'log(k)', 'k/kmax', '(k/kmax)^2', '(k/kmax)^3']
    print("  FVI weight vector theta_V (10 components):")
    for iz in range(N_Z):
        label = f"z={Z_VALS[iz]}"
        theta_slice = theta_V[iz * N_FEAT_PER_Z:(iz + 1) * N_FEAT_PER_Z]
        formatted = ', '.join([f'{v:>10.4f}' for v in theta_slice])
        print(f"    {label} [{', '.join(feat_labels)}]: [{formatted}]")
    print()

    # -------------------------------------------------------------------------
    # Phase 4: Value-function comparison at convergence
    # -------------------------------------------------------------------------
    print("Phase 4: Value function comparison (FVI vs FQI at optimal action)")
    print("-" * 50)

    R, P = build_reward_and_transitions(k_grid)
    feas = R > -1e30
    Q_fqi_mat = np.where(feas, Phi @ Theta_Q.T, -1e30)
    opt_actions = np.argmax(Q_fqi_mat, axis=1)

    V_fvi_vals = V_fvi
    V_fqi_at_opt = np.array([Phi[s] @ Theta_Q[opt_actions[s]] for s in range(N_S)])

    diff_fvi_fqi = np.abs(V_fvi_vals - V_fqi_at_opt)
    print(f"  Per-state |V_FVI(s) - Q_FQI(s, a*(s))|:")
    print(f"    Max  difference: {diff_fvi_fqi.max():.6f}")
    print(f"    Mean difference: {diff_fvi_fqi.mean():.6f}")
    print()

    for iz in range(N_Z):
        states_z = [ik * N_Z + iz for ik in range(N_K)]
        diff_z = diff_fvi_fqi[states_z]
        print(f"  z={Z_VALS[iz]}: max diff = {diff_z.max():.6f}, "
              f"mean diff = {diff_z.mean():.6f}")

    print()
    print("  Detailed comparison (first 5 states per z level):")
    header = (f"  {'s':>4} {'iz':>3} {'k':>8} | "
              f"{'V*(s)':>10} {'V_FVI(s)':>10} {'Q_FQI(s,a*)':>12} | "
              f"{'|FVI-V*|':>10} {'|FQI-V*|':>10} {'|FVI-FQI|':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for iz in range(N_Z):
        for ik in range(min(5, N_K)):
            s = ik * N_Z + iz
            vstar = V_star[s]
            vfvi  = V_fvi_vals[s]
            vfqi  = V_fqi_at_opt[s]
            print(f"  {s:>4} {iz:>3} {k_grid[ik]:>8.4f} | "
                  f"{vstar:>10.4f} {vfvi:>10.4f} {vfqi:>12.4f} | "
                  f"{abs(vfvi-vstar):>10.6f} {abs(vfqi-vstar):>10.6f} {abs(vfvi-vfqi):>10.6f}")
    print()

    # -------------------------------------------------------------------------
    # Phase 5: Summary
    # -------------------------------------------------------------------------
    print("Phase 5: Results Summary")
    print("-" * 50)
    header = (f"  {'Method':<12} {'||V-V*||_inf':>14} {'||V-V*||_rms':>14} "
              f"{'Pol Agr %':>10} {'Iters':>7} {'Time(s)':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'VI (exact)':<12} {0.0:>14.6f} {0.0:>14.6f} "
          f"{pol_agr_vi:>9.1f}% {'--':>7} {t_vi:>8.3f}")
    print(f"  {'FVI':<12} {err_fvi_inf:>14.6f} {err_fvi_rms:>14.6f} "
          f"{pol_agr_fvi:>9.1f}% {n_iter_fvi:>7} {t_fvi:>8.3f}")
    print(f"  {'FQI':<12} {err_fqi_inf:>14.6f} {err_fqi_rms:>14.6f} "
          f"{pol_agr_fqi:>9.1f}% {n_iter_fqi:>7} {t_fqi:>8.3f}")
    print(f"  {'Oracle-FQI':<12} {err_ora_inf:>14.6f} {err_ora_rms:>14.6f} "
          f"{pol_agr_ora:>9.1f}% {n_iter_ora:>7} {t_ora:>8.3f}")
    alpha_str = f"(a={alpha_traj[-1]:.3f})"
    print(f"  {'NLLS-FQI':<12} {err_nls_inf:>14.6f} {err_nls_rms:>14.6f} "
          f"{pol_agr_nls:>9.1f}% {n_iter_nls:>7} {t_nls:>8.3f}  {alpha_str}")
    print()
    print(f"  Max |V_FVI - V_FQI| at optimal action: {diff_fvi_fqi.max():.6f}")
    print(f"  Best-representable error in span(Phi): {approx_err_vstar:.6f}")
    print()

    # -------------------------------------------------------------------------
    # Phase 6: Weight table (for LaTeX)
    # -------------------------------------------------------------------------
    print("Phase 6: Feature weight table (FVI vs FQI modal-optimal actions)")
    print("-" * 50)
    modal_actions = []
    for iz in range(N_Z):
        states_z = [ik * N_Z + iz for ik in range(N_K)]
        actions_z = opt_actions[states_z]
        modal_a = np.bincount(actions_z).argmax()
        modal_actions.append(modal_a)
        print(f"  z={Z_VALS[iz]}: modal a* = {modal_a} (k'={k_grid[modal_a]:.4f}), "
              f"appears in {(actions_z == modal_a).sum()} / {N_K} states")

    print()
    print(f"  {'Feature':<24} {'FVI theta_V':>12} | "
          f"{'FQI theta_a* z_low':>18} {'FQI theta_a* z_high':>19}")
    print("  " + "-" * 77)
    feat_labels_full = []
    for iz in range(N_Z):
        for name in feat_labels:
            feat_labels_full.append(f"z={Z_VALS[iz]}, {name}")

    for j, label in enumerate(feat_labels_full):
        tv  = theta_V[j]
        tql = Theta_Q[modal_actions[0], j]
        tqh = Theta_Q[modal_actions[1], j]
        print(f"  {label:<24} {tv:>12.4f} | {tql:>18.4f} {tqh:>19.4f}")
    print()

    for iz in range(N_Z):
        a_modal = modal_actions[iz]
        states_z = [ik * N_Z + iz for ik in range(N_K)]
        fvi_avg = np.mean(V_fvi_vals[states_z])
        fqi_avg = np.mean([Phi[s] @ Theta_Q[a_modal] for s in states_z])
        print(f"  z={Z_VALS[iz]}: avg V_FVI = {fvi_avg:.4f}, "
              f"avg phi(s)^T theta_{{a*}} = {fqi_avg:.4f}, "
              f"diff = {abs(fvi_avg - fqi_avg):.4f}")
    print()

    # -------------------------------------------------------------------------
    # Phase 7: Figures
    # -------------------------------------------------------------------------
    print("Phase 7: Generating convergence figure")
    print("-" * 50)
    fig, ax1 = plt.subplots(1, 1, figsize=FIG_SINGLE)

    ax1.plot(range(1, len(errors_fvi) + 1), errors_fvi,
             label='FVI (linear basis)', color='steelblue', linewidth=1.5)
    ax1.plot(range(1, len(errors_fqi) + 1), errors_fqi,
             label='FQI (linear basis)', color='darkorange', linewidth=1.5)
    ax1.plot(range(1, len(errors_ora) + 1), errors_ora,
             label='Oracle-FQI ($\\alpha$ known)', color='#2ca02c',
             linewidth=1.5)
    ax1.plot(range(1, len(errors_nls) + 1), errors_nls,
             label='NLLS-FQI ($\\alpha$ estimated)', color='#d62728',
             linewidth=1.5, linestyle='--')
    ax1.set_xlabel('Iteration $k$')
    ax1.set_ylabel('$\\|V_k - V^*\\|_\\infty$')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'bm_fvi_fqi.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # -------------------------------------------------------------------------
    # Phase 8: LaTeX weight table
    # -------------------------------------------------------------------------
    print("Phase 8: Generating LaTeX tables")
    print("-" * 50)

    lines_results = []
    lines_results.append(r'\begin{tabular}{lrrrr}')
    lines_results.append(r'\hline')
    lines_results.append(r'Method & $\|V - V^*\|_\infty$ & $\|V - V^*\|_\text{rms}$'
                         r' & Policy agreement (\%) & Iterations \\')
    lines_results.append(r'\hline')
    lines_results.append(f'Exact VI & 0.0000 & 0.0000 & {pol_agr_vi:.1f} & --- \\\\')
    lines_results.append(f'FVI (linear) & {err_fvi_inf:.4f} & {err_fvi_rms:.4f}'
                         f' & {pol_agr_fvi:.1f} & {n_iter_fvi} \\\\')
    lines_results.append(f'FQI (linear) & {err_fqi_inf:.4f} & {err_fqi_rms:.4f}'
                         f' & {pol_agr_fqi:.1f} & {n_iter_fqi} \\\\')
    lines_results.append(f'Oracle-FQI & {err_ora_inf:.4f} & {err_ora_rms:.4f}'
                         f' & {pol_agr_ora:.1f} & {n_iter_ora} \\\\')
    alpha_hat = alpha_traj[-1]
    lines_results.append(f'NLLS-FQI ($\\hat{{\\alpha}} = {alpha_hat:.4f}$)'
                         f' & {err_nls_inf:.4f} & {err_nls_rms:.4f}'
                         f' & {pol_agr_nls:.1f} & {n_iter_nls} \\\\')
    lines_results.append(r'\hline')
    lines_results.append(r'\end{tabular}')

    results_path = os.path.join(OUTPUT_DIR, 'bm_fvi_fqi_results.tex')
    with open(results_path, 'w') as f:
        f.write('\n'.join(lines_results))
    print(f"  Saved: {results_path}")

    lines_weights = []
    lines_weights.append(r'\begin{tabular}{lrrr}')
    lines_weights.append(r'\hline')
    lines_weights.append(r'Basis feature & FVI $\hat{\theta}_V$ & '
                         r'FQI $\hat{\theta}_{a^*(z_\ell)}$ & '
                         r'FQI $\hat{\theta}_{a^*(z_h)}$ \\')
    lines_weights.append(r'\hline')

    feat_tex = [r'$k^0$', r'$\log k$', r'$k/\bar{k}$',
                r'$(k/\bar{k})^2$', r'$(k/\bar{k})^3$']
    for iz in range(N_Z):
        z_str = f'$z = {Z_VALS[iz]}$'
        for p, ftex in enumerate(feat_tex):
            j = iz * N_FEAT_PER_Z + p
            tv  = theta_V[j]
            tql = Theta_Q[modal_actions[0], j]
            tqh = Theta_Q[modal_actions[1], j]
            label = f'{z_str}, {ftex}'
            lines_weights.append(f'{label} & {tv:.4f} & {tql:.4f} & {tqh:.4f} \\\\')
        lines_weights.append(r'\hline')

    lines_weights.append(r'\end{tabular}')

    weights_path = os.path.join(OUTPUT_DIR, 'bm_fvi_fqi_weights.tex')
    with open(weights_path, 'w') as f:
        f.write('\n'.join(lines_weights))
    print(f"  Saved: {weights_path}")
    print()

    print("Output files:")
    print(f"  {fig_path}")
    print(f"  {results_path}")
    print(f"  {weights_path}")
    print(f"  {os.path.join(OUTPUT_DIR, 'bm_fvi_fqi_stdout.txt')}  (this run)")
    print()
    print("Done.")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Brock-Mirman FVI vs FQI comparison')
    add_cache_args(parser)
    args = parser.parse_args()

    print("=" * 70)
    print("BROCK-MIRMAN: FITTED VALUE ITERATION vs FITTED Q-ITERATION")
    print("=" * 70)
    print()
    print("Parameters:")
    print(f"  alpha (capital share): {ALPHA_PROD}")
    print(f"  beta (discount factor): {BETA}")
    print(f"  z_vals: {Z_VALS.tolist()}")
    print(f"  PI_TRANS: {PI_TRANS.tolist()}")
    print(f"  N_K={N_K}, N_Z={N_Z}, N_S={N_S}, N_A={N_A}")
    feat_names = ['1 (const)', 'log(k)'] + [f'(k/k_max)^{p}' for p in [1,2,3]]
    print(f"  Basis (per z): {feat_names}  ->  {N_BASIS} total features")
    print(f"  Max iterations: FVI={MAX_ITER_FVI}, FQI={MAX_ITER_FQI}")
    print(f"  Convergence tolerance (change in V): {TOL_FIT}")
    print()

    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, "No cache found. Run without --plots-only first."
    else:
        data = compute_data()

    if not args.data_only:
        generate_outputs(data)


if __name__ == '__main__':
    main()
