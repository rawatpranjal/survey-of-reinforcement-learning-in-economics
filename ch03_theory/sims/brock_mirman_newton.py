"""
Brock-Mirman Optimal Growth: VI vs PI vs LP Dual
Chapter 3 — Demonstrates PI = Newton's method on the Bellman equation.

Five cached components:
  shared_r1     — grid, reward/transition matrices, closed-form policy (500 grid × 2 states)
  VI_r1         — value iteration on shared_r1
  PI_r1         — policy iteration on shared_r1
  lp_r2         — LP primal/dual (self-contained, 20 grid × 2 states)
  timing_sweep  — wall-clock VI vs PI across grid sizes 10..200
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, COLORS, CMAP_SEQ, FIG_SINGLE, FIG_DOUBLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
apply_style()

import argparse
import time

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# ============================================================================
# Parameters
# ============================================================================

ALPHA = 0.36        # capital share
BETA = 0.96         # discount factor
Z_VALS = np.array([0.9, 1.1])  # productivity states
PI_TRANS = np.array([[0.8, 0.2],
                     [0.2, 0.8]])  # Markov transition for z

SEED = 42
TOL = 1e-10
MAX_ITER_VI = 5000
MAX_ITER_PI = 50

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
SCRIPT_NAME = 'brock_mirman'

# ============================================================================
# Per-component config dicts
# ============================================================================

ENV_PARAMS = {
    'alpha': ALPHA, 'beta': BETA,
    'z': Z_VALS.tolist(), 'pi': PI_TRANS.tolist(),
    'seed': SEED, 'tol': TOL, 'version': 3,
}

SHARED_R1_CONFIG = {**ENV_PARAMS, 'n_k': 500}
VI_R1_CONFIG     = {**SHARED_R1_CONFIG, 'max_iter_vi': MAX_ITER_VI}
PI_R1_CONFIG     = {**SHARED_R1_CONFIG, 'max_iter_pi': MAX_ITER_PI}
LP_R2_CONFIG     = {**ENV_PARAMS, 'n_k': 20}
TIMING_CONFIG    = {**ENV_PARAMS, 'grid_sizes': [10, 20, 50, 100, 200]}

# Regime shorthand mapping for --algo r1, r2, r3
REGIME_COMPONENTS = {
    'r1': {'shared_r1', 'VI_r1', 'PI_r1'},
    'r2': {'lp_r2'},
    'r3': {'timing_sweep'},
}

# ============================================================================
# Model helpers
# ============================================================================

def build_grid(n_k):
    """Capital grid from near-zero to 150% of highest deterministic steady state."""
    k_ss_high = (ALPHA * BETA * Z_VALS.max()) ** (1 / (1 - ALPHA))
    return np.linspace(0.01, 1.5 * k_ss_high, n_k)


def utility(c):
    """Log utility; returns -inf for c <= 0."""
    with np.errstate(divide='ignore'):
        return np.where(c > 0, np.log(c), -np.inf)


def build_reward_and_transitions(k_grid, z_vals, pi_trans):
    """
    Build reward tensor R[s, a] and transition matrix P[s, a, s']
    where s = (k_idx, z_idx) flattened, a = k'_idx.
    """
    n_k = len(k_grid)
    n_z = len(z_vals)
    n_s = n_k * n_z
    n_a = n_k  # action = choice of next-period capital grid index

    R = np.full((n_s, n_a), -np.inf)
    P = np.zeros((n_s, n_a, n_s))

    for ik in range(n_k):
        for iz in range(n_z):
            s = ik * n_z + iz
            output = z_vals[iz] * k_grid[ik] ** ALPHA
            for ia in range(n_a):
                c = output - k_grid[ia]
                if c > 1e-12:
                    R[s, ia] = np.log(c)
                    for iz_next in range(n_z):
                        s_next = ia * n_z + iz_next
                        P[s, ia, s_next] = pi_trans[iz, iz_next]

    return R, P


def closed_form_policy(k_grid, z_vals):
    """Closed-form: k'(k,z) = αβ z k^α."""
    n_k = len(k_grid)
    n_z = len(z_vals)
    policy = np.zeros(n_k * n_z, dtype=int)
    for ik in range(n_k):
        for iz in range(n_z):
            s = ik * n_z + iz
            k_next = ALPHA * BETA * z_vals[iz] * k_grid[ik] ** ALPHA
            policy[s] = np.argmin(np.abs(k_grid - k_next))
    return policy


# ============================================================================
# Algorithms
# ============================================================================

def value_iteration(R, P, gamma, tol=TOL, max_iter=MAX_ITER_VI):
    """Standard VI. Returns V*, policy, successive-diff errors, V_history."""
    n_s, n_a = R.shape
    V = np.zeros(n_s)
    errors = []
    V_history = [V.copy()]
    for it in range(max_iter):
        Q = R + gamma * (P @ V)  # shape (n_s, n_a)
        V_new = np.max(Q, axis=1)
        err = np.max(np.abs(V_new - V))
        errors.append(err)
        V = V_new
        V_history.append(V.copy())
        if err < tol:
            break
    policy = np.argmax(R + gamma * (P @ V), axis=1)
    return V, policy, errors, V_history


def policy_iteration(R, P, gamma, max_iter=MAX_ITER_PI):
    """Howard PI. Returns V*, policy, error history, per-iteration timing, bellman_resids."""
    n_s, n_a = R.shape
    # Initialize with greedy policy from V=0
    policy = np.argmax(R, axis=1)
    errors = []
    timings = []
    bellman_resids = []

    V_history = []
    for it in range(max_iter):
        t0 = time.perf_counter()
        # Policy evaluation: solve (I - γ P^π) V = r^π
        r_pi = R[np.arange(n_s), policy]
        P_pi = P[np.arange(n_s), policy, :]  # (n_s, n_s)
        A_mat = np.eye(n_s) - gamma * P_pi
        V = np.linalg.solve(A_mat, r_pi)

        # Policy improvement + Bellman residual
        Q = R + gamma * (P @ V)
        new_policy = np.argmax(Q, axis=1)
        TV = np.max(Q, axis=1)
        bellman_resids.append(np.max(np.abs(V - TV)))

        t1 = time.perf_counter()
        timings.append(t1 - t0)

        V_history.append(V.copy())

        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    # Compute errors relative to final V
    V_star = V_history[-1]
    for V_h in V_history:
        errors.append(np.max(np.abs(V_h - V_star)))

    return V, policy, errors, timings, bellman_resids


def lp_primal(R, P, gamma):
    """
    Manne (1960) LP formulation.
    Primal: min  Σ_s α(s) V(s)
            s.t. V(s) >= R(s,a) + γ Σ_{s'} P(s,a,s') V(s')  ∀ feasible (s,a)
    Dual:   max  Σ_{s,a} R(s,a) μ(s,a)
            s.t. Σ_a μ(s,a) - γ Σ_{s',a'} P(s',a',s) μ(s',a') = α(s)  ∀ s
                 μ(s,a) >= 0
    """
    n_s, n_a = R.shape

    # --- Primal ---
    # Collect only feasible constraints
    feasible = [(s, a) for s in range(n_s) for a in range(n_a) if R[s, a] > -1e30]
    n_con = len(feasible)

    A_ub = np.zeros((n_con, n_s))
    b_ub = np.zeros(n_con)
    for idx, (s, a) in enumerate(feasible):
        # Constraint: -(e_s - γ P[s,a,:])^T V <= -R[s,a]
        A_ub[idx, s] = -1.0
        A_ub[idx, :] += gamma * P[s, a, :]
        b_ub[idx] = -R[s, a]

    c = np.ones(n_s) / n_s  # uniform initial distribution
    v_bounds = [(-1e6, 1e6)] * n_s  # explicit bounds (V can be negative with log utility)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=v_bounds, method='highs')
    assert result.success, f"LP primal failed: {result.message}"
    V_lp = result.x

    # --- Dual ---
    alpha = np.ones(n_s) / n_s

    # Only allow positive measure on feasible (s,a)
    feasible_set = set(feasible)
    c_dual = np.zeros(n_s * n_a)
    dual_bounds = []
    for s in range(n_s):
        for a in range(n_a):
            sa = s * n_a + a
            if (s, a) in feasible_set:
                c_dual[sa] = -R[s, a]  # negate for min
                dual_bounds.append((0, None))
            else:
                c_dual[sa] = 0
                dual_bounds.append((0, 0))  # force infeasible actions to zero

    # Flow conservation: Σ_a μ(s,a) - γ Σ_{s',a'} P(s',a',s) μ(s',a') = α(s)
    A_eq = np.zeros((n_s, n_s * n_a))
    b_eq = alpha.copy()

    for s in range(n_s):
        for a in range(n_a):
            sa = s * n_a + a
            A_eq[s, sa] += 1.0
        for s_prev in range(n_s):
            for a_prev in range(n_a):
                sa_prev = s_prev * n_a + a_prev
                A_eq[s, sa_prev] -= gamma * P[s_prev, a_prev, s]

    result_dual = linprog(c_dual, A_eq=A_eq, b_eq=b_eq, bounds=dual_bounds, method='highs')
    assert result_dual.success, f"LP dual failed: {result_dual.message}"
    mu = result_dual.x.reshape(n_s, n_a)

    # Extract policy from occupation measures
    mu_s = mu.sum(axis=1)
    policy_lp = np.zeros(n_s, dtype=int)
    for s in range(n_s):
        if mu_s[s] > 1e-12:
            policy_lp[s] = np.argmax(mu[s, :])
        else:
            policy_lp[s] = np.argmax(R[s, :])

    return V_lp, policy_lp, mu


# ============================================================================
# Per-component compute functions
# ============================================================================

def compute_shared_r1():
    """Grid, R, P, closed-form policy for Regime 1 (500 grid × 2 states)."""
    n_k = SHARED_R1_CONFIG['n_k']
    np.random.seed(SEED)
    k_grid = build_grid(n_k)
    R, P = build_reward_and_transitions(k_grid, Z_VALS, PI_TRANS)
    cf_pol = closed_form_policy(k_grid, Z_VALS)
    n_s = n_k * len(Z_VALS)

    print(f"  States: {n_s}, Actions: {n_k}")
    print(f"  Parameters: α={ALPHA}, β={BETA}")

    return {
        'k_grid': k_grid, 'R': R, 'P': P,
        'cf_pol': cf_pol, 'n_k': n_k, 'n_s': n_s,
    }


def compute_vi_r1(shared):
    """Value iteration on Regime 1 MDP."""
    R, P = shared['R'], shared['P']
    t0 = time.perf_counter()
    V_vi, pol_vi, errs_vi_succ, V_history_vi = value_iteration(R, P, BETA)
    t_vi = time.perf_counter() - t0

    vi_cf_match = np.mean(pol_vi == shared['cf_pol']) * 100
    print(f"  VI: {len(errs_vi_succ)} iterations, {t_vi:.2f}s, final error={errs_vi_succ[-1]:.2e}")
    print(f"  VI policy matches closed-form: {vi_cf_match:.1f}%")

    return {
        'V_vi': V_vi, 'pol_vi': pol_vi,
        'errs_vi': errs_vi_succ, 'V_history': V_history_vi,
        't_vi': t_vi, 'n_iters_vi': len(errs_vi_succ),
        'vi_cf_match': vi_cf_match,
    }


def compute_pi_r1(shared):
    """Policy iteration on Regime 1 MDP."""
    R, P = shared['R'], shared['P']
    t0 = time.perf_counter()
    V_pi, pol_pi, errs_pi, timings_pi, bellman_resids_pi = policy_iteration(R, P, BETA)
    t_pi = time.perf_counter() - t0

    pi_cf_match = np.mean(pol_pi == shared['cf_pol']) * 100
    print(f"  PI: {len(errs_pi)} iterations, {t_pi:.2f}s")
    print(f"  PI policy matches closed-form: {pi_cf_match:.1f}%")

    return {
        'V_pi': V_pi, 'pol_pi': pol_pi,
        'errs_pi': errs_pi, 'timings_pi': timings_pi,
        'bellman_resids_pi': bellman_resids_pi,
        't_pi': t_pi, 'n_iters_pi': len(errs_pi),
        'pi_cf_match': pi_cf_match,
    }


def compute_lp_r2():
    """Self-contained LP primal/dual regime (20 grid × 2 states)."""
    n_k = LP_R2_CONFIG['n_k']
    np.random.seed(SEED)
    k_grid = build_grid(n_k)
    R, P = build_reward_and_transitions(k_grid, Z_VALS, PI_TRANS)
    n_s = n_k * len(Z_VALS)

    print(f"  States: {n_s}, Actions: {n_k}, SA pairs: {n_s * n_k}")

    # VI as baseline
    t0 = time.perf_counter()
    V_vi, pol_vi, _, _ = value_iteration(R, P, BETA)
    t_vi = time.perf_counter() - t0
    print(f"  VI baseline: {t_vi:.3f}s")

    # LP primal + dual
    t0 = time.perf_counter()
    V_lp, pol_lp, mu = lp_primal(R, P, BETA)
    t_lp = time.perf_counter() - t0

    lp_vi_diff = np.max(np.abs(V_lp - V_vi))
    pol_match = np.mean(pol_lp == pol_vi) * 100
    print(f"  LP primal: {t_lp:.3f}s")
    print(f"  ||V_LP - V_VI||_inf = {lp_vi_diff:.2e}")
    print(f"  LP policy matches VI: {pol_match:.1f}%")

    mu_s = mu.sum(axis=1)
    print(f"  Occupation measures: min={mu_s.min():.4f}, max={mu_s.max():.4f}, sum={mu_s.sum():.4f}")

    return {
        'V_vi': V_vi, 'V_lp': V_lp,
        'pol_vi': pol_vi, 'pol_lp': pol_lp,
        'mu': mu, 'mu_s': mu_s,
        'lp_vi_diff': lp_vi_diff,
        'pol_match': pol_match,
        't_vi': t_vi, 't_lp': t_lp,
        'k_grid': k_grid,
        'n_s': n_s, 'n_a': n_k,
    }


def compute_timing_sweep():
    """Wall-clock VI vs PI across grid sizes 10..200."""
    grid_sizes = TIMING_CONFIG['grid_sizes']
    np.random.seed(SEED)

    timing_results = []
    for nk in grid_sizes:
        kg = build_grid(nk)
        Rg, Pg = build_reward_and_transitions(kg, Z_VALS, PI_TRANS)

        t0 = time.perf_counter()
        _, _, errs, _ = value_iteration(Rg, Pg, BETA)
        t_v = time.perf_counter() - t0

        t0 = time.perf_counter()
        _, _, errs_p, _, _ = policy_iteration(Rg, Pg, BETA)
        t_p = time.perf_counter() - t0

        timing_results.append({
            'n_k': nk, 'n_s': nk * len(Z_VALS),
            'vi_iters': len(errs), 'pi_iters': len(errs_p),
            't_vi': t_v, 't_pi': t_p,
        })
        print(f"  Grid {nk}: VI {len(errs)} iters ({t_v:.3f}s), PI {len(errs_p)} iters ({t_p:.3f}s)")

    # Cost summary from the n_k=20 entry
    r20 = next(r for r in timing_results if r['n_k'] == 20)
    n_s, n_a = r20['n_s'], 20
    vi_cost = n_s * n_a
    pi_cost = n_s**3 + n_s * n_a
    vi_total = r20['vi_iters'] * vi_cost
    pi_total = r20['pi_iters'] * pi_cost

    print(f"  VI per-iteration cost: O({n_s}x{n_a}) = {vi_cost}")
    print(f"  PI per-iteration cost: O({n_s}^3+{n_s}x{n_a}) = {pi_cost}")
    print(f"  VI total operations: {vi_total:,}")
    print(f"  PI total operations: {pi_total:,}")
    print(f"  Ratio (VI/PI): {vi_total/pi_total:.1f}x")

    return {
        'timing_results': timing_results,
        'vi_cost': vi_cost, 'pi_cost': pi_cost,
        'vi_total': vi_total, 'pi_total': pi_total,
        'n_s': n_s, 'n_a': n_a,
    }


# ============================================================================
# Cross-comparison assembly (not cached; fast arithmetic on cached data)
# ============================================================================

def assemble_regime1(vi_r1, pi_r1):
    """Compute cross-comparison metrics between VI and PI results."""
    V_star = pi_r1['V_pi']
    V_history = vi_r1['V_history']
    errs_vi_abs = [np.max(np.abs(Vh - V_star)) for Vh in V_history]
    theory_errors = [errs_vi_abs[0] * BETA**k for k in range(len(errs_vi_abs))]
    v_diff = np.max(np.abs(vi_r1['V_vi'] - pi_r1['V_pi']))

    print(f"  ||V_VI - V_PI||_inf = {v_diff:.2e}")

    return {
        'errs_vi_abs': errs_vi_abs,
        'theory_errors': theory_errors,
        'v_diff': v_diff,
    }


# ============================================================================
# Orchestration
# ============================================================================

def compute_data(force=None):
    force = force or set()
    # Expand regime shorthands
    expanded = set()
    for f in force:
        expanded |= REGIME_COMPONENTS.get(f, {f})
    force = expanded

    shared_r1 = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'shared_r1', SHARED_R1_CONFIG,
                                 compute_shared_r1, force=('shared_r1' in force))
    vi_r1 = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'VI_r1', VI_R1_CONFIG,
                             compute_vi_r1, shared_r1,
                             force=('VI_r1' in force or 'shared_r1' in force))
    pi_r1 = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'PI_r1', PI_R1_CONFIG,
                             compute_pi_r1, shared_r1,
                             force=('PI_r1' in force or 'shared_r1' in force))
    lp_r2 = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'lp_r2', LP_R2_CONFIG,
                             compute_lp_r2, force=('lp_r2' in force))
    timing = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'timing_sweep', TIMING_CONFIG,
                              compute_timing_sweep, force=('timing_sweep' in force))

    return {
        'shared_r1': shared_r1, 'VI_r1': vi_r1, 'PI_r1': pi_r1,
        'lp_r2': lp_r2, 'timing_sweep': timing,
    }


# ============================================================================
# Plotting
# ============================================================================

def _draw_operator_background(ax):
    """Draw shared background: policy lines, T envelope, 45-degree line, V*."""
    r_vals = [4.0, 3.5, 1.0]
    slopes = [0.20, 0.50, 0.82]
    labels = [r'$T^{\pi_1}$', r'$T^{\pi_2}$', r'$T^{\pi_3}$']

    V = np.linspace(-0.5, 9.5, 1000)

    for r, s, lbl in zip(r_vals, slopes, labels):
        line = r + s * V
        ax.plot(V, line, '--', color=COLORS['gray'], linewidth=0.7, zorder=1)
        ypos = r + s * 8.8
        if ypos < 9.5:
            ax.text(8.8, ypos, lbl, fontsize=7, color=COLORS['gray'], va='bottom')

    T_lines = np.array([r + s * V for r, s in zip(r_vals, slopes)])
    T_env = np.max(T_lines, axis=0)
    ax.plot(V, T_env, '-', color=COLORS['black'], linewidth=2.2, zorder=3)

    ax.plot(V, V, '-', color=COLORS['gray'], linewidth=0.8, zorder=1)

    ax.plot(7.0, 7.0, '*', color=COLORS['black'], markersize=10, zorder=5)
    ax.annotate('$V^*$', xy=(7.0, 7.0), xytext=(7.5, 6.2), fontsize=9,
                arrowprops=dict(arrowstyle='-', color=COLORS['gray'], lw=0.5))

    ax.set_xlabel('$V$')
    ax.set_ylabel('$TV$')
    ax.set_xlim(-0.5, 9.8)
    ax.set_ylim(-0.5, 9.8)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    return r_vals, slopes


def plot_convergence(vi_r1, pi_r1, r1_cross):
    """1x3 panel: (a) VI in V,TV space, (b) PI in V,TV space, (c) convergence."""
    VI_COLOR = ALGO_COLORS['VI']
    PI_COLOR = ALGO_COLORS['PI']
    THEORY_COLOR = COLORS['black']

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13, 4))

    # ---- Panel (a): VI staircase ----
    r_vals, slopes = _draw_operator_background(ax_a)
    V0 = 0.5
    v_curr = V0
    for _ in range(7):
        tv = max(r + s * v_curr for r, s in zip(r_vals, slopes))
        ax_a.plot([v_curr, v_curr], [v_curr, tv], '-', color=VI_COLOR,
                  linewidth=1.2, alpha=0.8, zorder=4)
        ax_a.plot([v_curr, tv], [tv, tv], '-', color=VI_COLOR,
                  linewidth=1.2, alpha=0.8, zorder=4)
        v_curr = tv
    ax_a.plot(V0, V0, 'o', color=VI_COLOR, markersize=3, zorder=5)
    ax_a.text(1.0, 8.5, 'VI', color=VI_COLOR, fontsize=9, fontstyle='italic')
    ax_a.set_title('(a)', loc='left', fontsize=10)

    # ---- Panel (b): PI Newton jumps ----
    r_vals, slopes = _draw_operator_background(ax_b)
    pi_segments = [
        (0.5, 0, 5.0),   # (V_start, policy_idx, V_end=fixpt)
        (5.0, 1, 7.0),
    ]
    for v_start, pi_idx, v_end in pi_segments:
        r, s = r_vals[pi_idx], slopes[pi_idx]
        tv_start = r + s * v_start
        ax_b.annotate('', xy=(v_start, tv_start), xytext=(v_start, v_start),
                      arrowprops=dict(arrowstyle='->', color=PI_COLOR, lw=1.8))
        ax_b.annotate('', xy=(v_end, v_end), xytext=(v_start, tv_start),
                      arrowprops=dict(arrowstyle='->', color=PI_COLOR, lw=1.8))
    ax_b.plot(V0, V0, 's', color=PI_COLOR, markersize=3, zorder=5)
    ax_b.text(2.5, 2.0, 'PI', color=PI_COLOR, fontsize=9, fontstyle='italic')
    ax_b.set_title('(b)', loc='left', fontsize=10)

    # ---- Panel (c): Sup-norm error convergence ----
    errs_vi_abs = r1_cross['errs_vi_abs']
    errs_pi = pi_r1['errs_pi']
    theory = r1_cross['theory_errors']
    n_vi = vi_r1['n_iters_vi']
    n_pi = pi_r1['n_iters_pi']

    ax_c.semilogy(range(len(errs_vi_abs)), errs_vi_abs, color=VI_COLOR,
                  linewidth=1.5, label=f'VI ({n_vi} iters)', zorder=3)
    ax_c.semilogy(range(len(theory)), theory, color=THEORY_COLOR,
                  linestyle='--', linewidth=1.0, alpha=0.5,
                  label=f'$\\beta^k$ bound ($\\beta$={BETA})')
    ax_c.semilogy(range(len(errs_pi)), errs_pi, color=PI_COLOR,
                  marker='o', markersize=5, linewidth=1.5,
                  label=f'PI ({n_pi} iters)', zorder=3)

    ax_c.set_xlabel('Iteration $k$')
    ax_c.set_ylabel('$\\|V_k - V^*\\|_\\infty$')
    ax_c.legend(fontsize=7, loc='upper right')
    ax_c.set_title('(c)', loc='left', fontsize=10)

    fig.tight_layout()

    # Save both PNG and PDF
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTPUT_DIR, f'brock_mirman_convergence.{ext}')
        fig.savefig(path, bbox_inches='tight')
        print(f"\nSaved: {path}")
    plt.close(fig)


def generate_table(data):
    """Consolidated results table from all components."""
    vi_r1 = data['VI_r1']
    pi_r1 = data['PI_r1']
    lp_r2 = data['lp_r2']
    timing = data['timing_sweep']

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Brock--Mirman Economy: VI vs PI vs LP}')
    lines.append(r'\label{tab:brock_mirman}')
    lines.append(r'\begin{tabular}{llrrr}')
    lines.append(r'\hline')
    lines.append(r'Regime & Method & Iterations & Time (s) & $\|V - V^*\|_\infty$ \\')
    lines.append(r'\hline')

    # Regime 1
    lines.append(f'1. Contraction & VI & {vi_r1["n_iters_vi"]} & {vi_r1["t_vi"]:.2f} & {vi_r1["errs_vi"][-1]:.1e} \\\\')
    lines.append(f' & PI & {pi_r1["n_iters_pi"]} & {pi_r1["t_pi"]:.2f} & {pi_r1["errs_pi"][-1]:.1e} \\\\')

    # Regime 2
    lines.append(f'2. LP dual & VI & --- & {lp_r2["t_vi"]:.3f} & --- \\\\')
    lines.append(f' & LP & --- & {lp_r2["t_lp"]:.3f} & {lp_r2["lp_vi_diff"]:.1e} \\\\')

    # Regime 3: timing across grid sizes
    tr = timing['timing_results']
    r3_small = tr[0]  # n_k=10
    r3_large = tr[-1]  # n_k=200
    lines.append(f'3. Rate ($n_k$=10) & VI & {r3_small["vi_iters"]} & {r3_small["t_vi"]:.3f} & --- \\\\')
    lines.append(f' & PI & {r3_small["pi_iters"]} & {r3_small["t_pi"]:.3f} & --- \\\\')
    lines.append(f'3. Rate ($n_k$=200) & VI & {r3_large["vi_iters"]} & {r3_large["t_vi"]:.3f} & --- \\\\')
    lines.append(f' & PI & {r3_large["pi_iters"]} & {r3_large["t_pi"]:.3f} & --- \\\\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    path = os.path.join(OUTPUT_DIR, 'brock_mirman_results.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {path}")


def generate_outputs(data):
    """Generate all plots and tables from cached component data."""
    vi_r1 = data['VI_r1']
    pi_r1 = data['PI_r1']

    r1_cross = assemble_regime1(vi_r1, pi_r1)
    plot_convergence(vi_r1, pi_r1, r1_cross)
    generate_table(data)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Brock-Mirman VI vs PI vs LP')
    add_component_args(parser)
    args = parser.parse_args()

    force = parse_force_set(args)

    print("Brock-Mirman Optimal Growth: VI vs PI vs LP")
    print(f"Parameters: α={ALPHA}, β={BETA}, z∈{Z_VALS.tolist()}")
    if force:
        print(f"Force recompute: {sorted(force)}")

    if args.plots_only:
        data = compute_data()  # all cache hits
        generate_outputs(data)
    elif args.data_only:
        compute_data(force=force)
    else:
        data = compute_data(force=force)
        generate_outputs(data)

    print("\nDone.")


if __name__ == '__main__':
    main()
