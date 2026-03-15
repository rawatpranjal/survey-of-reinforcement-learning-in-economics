"""
Q-Learning on the Bertsekas V,TV Geometry Diagram
Chapter 3 — Side exploration (not for the paper).

Visualizes VI staircase, PI Newton jumps, and Q-learning (dot cloud +
dampened micro-staircase) on the same (V, TV) axes at a single
representative state s_med of the Brock-Mirman model.
"""

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, ALGO_COLORS
apply_style()

import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# Parameters
# ============================================================================

ALPHA = 0.36          # capital share
BETA = 0.96           # discount factor
Z_VALS = np.array([0.9, 1.1])
PI_TRANS = np.array([[0.8, 0.2],
                     [0.2, 0.8]])
N_K = 50
N_Z = len(Z_VALS)
N_S = N_K * N_Z
N_A = N_K

SEED = 42
TOL = 1e-10
MAX_ITER_VI = 5000
MAX_ITER_PI = 50

# Q-learning hyperparameters
QL_EPISODES = 500_000
QL_HORIZON = 100
QL_ALPHA_C = 100.0
QL_EPS_START = 1.0
QL_EPS_END = 0.05
QL_EPS_DECAY = 0.99999

OUTPUT_DIR = os.path.dirname(__file__)

# Colors
VI_COLOR = ALGO_COLORS['VI']
PI_COLOR = ALGO_COLORS['PI']
QL_COLOR = ALGO_COLORS['Q-Learning']

# ============================================================================
# Model helpers (from brock_mirman_newton.py)
# ============================================================================

def build_grid(n_k):
    k_ss_high = (ALPHA * BETA * Z_VALS.max()) ** (1 / (1 - ALPHA))
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
            output = Z_VALS[iz] * k_grid[ik] ** ALPHA
            for ia in range(n_a):
                c = output - k_grid[ia]
                if c > 1e-12:
                    R[s, ia] = np.log(c)
                    for iz_next in range(n_z):
                        s_next = ia * n_z + iz_next
                        P[s, ia, s_next] = PI_TRANS[iz, iz_next]

    return R, P


def feasible_mask(R):
    return R > -1e30


# ============================================================================
# Algorithms
# ============================================================================

def value_iteration(R, P, gamma, tol=TOL, max_iter=MAX_ITER_VI):
    n_s, n_a = R.shape
    V = np.zeros(n_s)
    V_history = [V.copy()]
    for it in range(max_iter):
        Q = R + gamma * (P @ V)
        V_new = np.max(Q, axis=1)
        err = np.max(np.abs(V_new - V))
        V = V_new
        V_history.append(V.copy())
        if err < tol:
            break
    policy = np.argmax(R + gamma * (P @ V), axis=1)
    return V, policy, V_history


def policy_iteration(R, P, gamma, max_iter=MAX_ITER_PI):
    n_s, n_a = R.shape
    policy = np.argmax(R, axis=1)
    V_history = []
    for it in range(max_iter):
        r_pi = R[np.arange(n_s), policy]
        P_pi = P[np.arange(n_s), policy, :]
        A_mat = np.eye(n_s) - gamma * P_pi
        V = np.linalg.solve(A_mat, r_pi)
        V_history.append(V.copy())
        Q = R + gamma * (P @ V)
        new_policy = np.argmax(Q, axis=1)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return V, policy, V_history


# ============================================================================
# BrockMirmanEnv (from bm_illustrated.py)
# ============================================================================

class BrockMirmanEnv:
    def __init__(self, k_grid, R, P_mat):
        self.k_grid = k_grid
        self.R = R
        self.P = P_mat
        self.n_k = len(k_grid)
        self.n_z = N_Z
        self._feasible = feasible_mask(R)

    def state_to_index(self, s):
        ik, iz = s
        return ik * self.n_z + iz

    def reset(self):
        ik = np.random.randint(self.n_k)
        iz = np.random.randint(self.n_z)
        self._state = (ik, iz)
        return self._state

    def step(self, action):
        ik, iz = self._state
        s_idx = ik * self.n_z + iz
        r = self.R[s_idx, action]
        iz_next = np.random.choice(self.n_z, p=PI_TRANS[iz])
        self._state = (action, iz_next)
        return self._state, r, False


# ============================================================================
# Q-learning with per-visit recording at s_med
# ============================================================================

def run_q_learning_with_geometry(env, s_med, R, P):
    """Run Q-learning, recording (V_before, target, V_after) at every visit to s_med."""
    np.random.seed(SEED)
    feas = feasible_mask(R)

    Q = np.full((N_S, N_A), -1e6)
    for s in range(N_S):
        Q[s, feas[s]] = 0.0

    counts = np.zeros((N_S, N_A))
    epsilon = QL_EPS_START

    # Per-visit recording at s_med
    ql_V_before = []
    ql_target = []
    ql_V_after = []

    for ep in range(QL_EPISODES):
        s = env.reset()
        for t in range(QL_HORIZON):
            s_idx = env.state_to_index(s)

            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                feasible_a = np.where(feas[s_idx])[0]
                a = np.random.choice(feasible_a)
            else:
                a = np.argmax(Q[s_idx])

            ns, r, done = env.step(a)
            ns_idx = env.state_to_index(ns)

            if r > -1e30:
                # Record geometry at s_med BEFORE update
                if s_idx == s_med:
                    v_before = np.max(Q[s_idx])
                    td_target = r + BETA * np.max(Q[ns_idx])

                counts[s_idx, a] += 1
                alpha_sa = QL_ALPHA_C / (QL_ALPHA_C + counts[s_idx, a])
                max_q_next = np.max(Q[ns_idx])
                td_tgt = r + BETA * max_q_next
                Q[s_idx, a] += alpha_sa * (td_tgt - Q[s_idx, a])

                # Record geometry at s_med AFTER update
                if s_idx == s_med:
                    v_after = np.max(Q[s_idx])
                    ql_V_before.append(v_before)
                    ql_target.append(td_target)
                    ql_V_after.append(v_after)

            s = ns

        epsilon = max(QL_EPS_END, epsilon * QL_EPS_DECAY)

        if (ep + 1) % 100_000 == 0:
            V_l = np.max(Q, axis=1)
            print(f"  ep {ep+1:>7d}: V[s_med]={V_l[s_med]:.4f}, "
                  f"visits={len(ql_V_before)}")

    V_final = np.max(Q, axis=1)
    return (np.array(ql_V_before), np.array(ql_target),
            np.array(ql_V_after), V_final)


# ============================================================================
# Compute empirical T curve at s_med
# ============================================================================

def compute_T_at_s(V_vec, R, P, s):
    """Compute (TV)[s] = max_a { R[s,a] + gamma * sum_{s'} P[s,a,s'] V[s'] }."""
    Q_s = R[s, :] + BETA * (P[s, :, :] @ V_vec)
    return np.max(Q_s)


# ============================================================================
# Main
# ============================================================================

def generate_outputs():
    print("Q-Learning Geometry on Brock-Mirman (V, TV) Diagram")
    print(f"Parameters: alpha={ALPHA}, beta={BETA}, N_K={N_K}")
    print()

    # --- 1. Build model ---
    k_grid = build_grid(N_K)
    R, P = build_reward_and_transitions(k_grid)

    # Pick s_med: k near steady state, z_H=1.1 (index 1)
    k_ss = (ALPHA * BETA) ** (1 / (1 - ALPHA))
    ik_ss = np.argmin(np.abs(k_grid - k_ss))
    s_med = ik_ss * N_Z + 1  # z_H index
    print(f"Steady-state capital: k_ss={k_ss:.4f}, grid index={ik_ss}, "
          f"k_grid[ik_ss]={k_grid[ik_ss]:.4f}")
    print(f"s_med = {s_med} (ik={ik_ss}, iz=1, z=1.1)")
    print()

    # --- 2. Run VI ---
    print("Running VI...", flush=True)
    t0 = time.perf_counter()
    V_vi, pol_vi, V_history_vi = value_iteration(R, P, BETA)
    t_vi = time.perf_counter() - t0
    print(f"  VI: {len(V_history_vi)-1} iterations, {t_vi:.2f}s")

    # --- 3. Run PI ---
    print("Running PI...", flush=True)
    t0 = time.perf_counter()
    V_pi, pol_pi, V_history_pi = policy_iteration(R, P, BETA)
    t_pi = time.perf_counter() - t0
    print(f"  PI: {len(V_history_pi)} iterations, {t_pi:.2f}s")

    V_star = V_pi
    print(f"  V*[s_med] = {V_star[s_med]:.6f}")
    print()

    # --- 4. Compute empirical T curve at s_med ---
    # Dense curve: interpolate V vectors between V_0 and V* and beyond
    V_0 = V_history_vi[0]
    # alpha=0 → V*, alpha=1 → V_0; extend both ways
    alphas = np.concatenate([
        np.linspace(-0.3, 0.0, 20),   # beyond V* (below V*)
        np.linspace(0.0, 1.0, 200),   # V* to V_0
        np.linspace(1.0, 1.5, 20),    # beyond V_0
    ])
    curve_V = []
    curve_TV = []
    for a in alphas:
        V_synth = V_star + a * (V_0 - V_star)
        v_s = V_synth[s_med]
        tv_s = compute_T_at_s(V_synth, R, P, s_med)
        curve_V.append(v_s)
        curve_TV.append(tv_s)
    curve_V = np.array(curve_V)
    curve_TV = np.array(curve_TV)
    sort_idx = np.argsort(curve_V)
    curve_V = curve_V[sort_idx]
    curve_TV = curve_TV[sort_idx]

    print(f"Empirical T curve: {len(curve_V)} points, "
          f"V range [{curve_V[0]:.2f}, {curve_V[-1]:.2f}]")

    # --- 5. VI staircase data at s_med ---
    vi_V_seq = np.array([V_k[s_med] for V_k in V_history_vi])
    vi_TV_seq = np.array([compute_T_at_s(V_k, R, P, s_med)
                          for V_k in V_history_vi])

    # --- 6. PI trajectory at s_med ---
    pi_points = []
    for k, V_k in enumerate(V_history_pi):
        v_s = V_k[s_med]
        tv_s = compute_T_at_s(V_k, R, P, s_med)
        if k + 1 < len(V_history_pi):
            v_next = V_history_pi[k + 1][s_med]
        else:
            v_next = V_star[s_med]
        pi_points.append((v_s, tv_s, v_next))

    print(f"PI trajectory: {len(pi_points)} iterations")
    for i, (v, tv, vn) in enumerate(pi_points):
        print(f"  PI iter {i}: V={v:.4f}, TV={tv:.4f}, V_next={vn:.4f}")
    print()

    # --- 7. Q-learning ---
    print("Running Q-learning...", flush=True)
    env = BrockMirmanEnv(k_grid, R, P)
    t0 = time.perf_counter()
    ql_V_before, ql_target, ql_V_after, V_ql = run_q_learning_with_geometry(
        env, s_med, R, P)
    t_ql = time.perf_counter() - t0
    print(f"  Q-learning: {t_ql:.1f}s, {len(ql_V_before)} visits to s_med")
    print(f"  V_ql[s_med] = {V_ql[s_med]:.6f} (V* = {V_star[s_med]:.6f})")
    print()

    # --- 8. Plot ---
    print("Plotting...", flush=True)
    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')

    v_star_s = V_star[s_med]

    # Zoom: wide enough to see large VI steps (far from V*) and Q-learning
    # cloud (concentrated near V*). At V≈-15, VI steps are ~0.4 units visible.
    ZOOM_LO = v_star_s - 4.0   # about -29.4
    ZOOM_HI = v_star_s + 12.0  # about -13.4

    # Layer 1: Background — 45-degree line
    ax.plot([ZOOM_LO, ZOOM_HI], [ZOOM_LO, ZOOM_HI], '-', color=COLORS['gray'],
            linewidth=0.8, zorder=1, label='_nolegend_')

    # Layer 1: Background — Empirical T curve (clip to zoom range)
    mask = (curve_V >= ZOOM_LO - 1) & (curve_V <= ZOOM_HI + 1)
    ax.plot(curve_V[mask], curve_TV[mask], '-', color=COLORS['black'], linewidth=2.5,
            zorder=3, label='$T$ operator')

    # Layer 1: V* point on diagonal
    ax.plot(v_star_s, v_star_s, '*', color=COLORS['black'], markersize=14, zorder=10)
    ax.annotate('$V^*$', xy=(v_star_s, v_star_s),
                xytext=(v_star_s - 1.5, v_star_s + 0.5),
                fontsize=11, color=COLORS['black'],
                arrowprops=dict(arrowstyle='-', color=COLORS['gray'], lw=0.5))

    # Layer 2: Q-learning dot cloud (clip to zoom range)
    ql_mask = ((ql_V_before >= ZOOM_LO) & (ql_V_before <= ZOOM_HI) &
               (ql_target >= ZOOM_LO) & (ql_target <= ZOOM_HI))
    ax.scatter(ql_V_before[ql_mask], ql_target[ql_mask],
               s=1.0, c=QL_COLOR, alpha=0.03, zorder=2, rasterized=True,
               label='_nolegend_')
    # Invisible proxy for legend
    ax.scatter([], [], s=20, c=QL_COLOR, alpha=0.5, label='Q-learning targets')

    # Layer 3: Q-learning dampened staircase (subsample, only in zoom)
    # Use a relatively sparse subsample so individual micro-steps are visible
    n_ql = len(ql_V_before)
    step_every = max(1, n_ql // 200)
    for i in range(0, n_ql, step_every):
        v_b = ql_V_before[i]
        v_a = ql_V_after[i]
        if v_b < ZOOM_LO or v_b > ZOOM_HI:
            continue
        if v_a < ZOOM_LO or v_a > ZOOM_HI:
            continue
        # Vertical: from (v_b, v_b) to (v_b, v_a) — dampened step toward target
        ax.plot([v_b, v_b], [v_b, v_a], '-', color=QL_COLOR,
                linewidth=0.5, alpha=0.25, zorder=4)
        # Horizontal: from (v_b, v_a) to (v_a, v_a) — back to diagonal
        ax.plot([v_b, v_a], [v_a, v_a], '-', color=QL_COLOR,
                linewidth=0.5, alpha=0.25, zorder=4)

    # Layer 4: VI staircase — continuous zigzag path within zoom range.
    # Draw every Nth step as connected vertical-horizontal segments.
    vi_in_range = [i for i in range(len(vi_V_seq))
                   if ZOOM_LO - 1 <= vi_V_seq[i] <= ZOOM_HI + 1
                   and ZOOM_LO - 1 <= vi_TV_seq[i] <= ZOOM_HI + 1]

    # Subsample: every 3rd iteration for continuous-looking staircase
    sub = max(1, len(vi_in_range) // 80)
    vi_show = vi_in_range[::sub]

    # Build continuous zigzag path: (v, v) → (v, Tv) → (Tv, Tv) → ...
    zz_x, zz_y = [], []
    for idx in vi_show:
        v = vi_V_seq[idx]
        tv = vi_TV_seq[idx]
        zz_x.extend([v, v])
        zz_y.extend([v, tv])
        # Horizontal to diagonal: next V_k is tv
        zz_x.append(tv)
        zz_y.append(tv)

    ax.plot(zz_x, zz_y, '-', color=VI_COLOR, linewidth=1.5, alpha=0.85,
            zorder=6, label='VI staircase')

    # Layer 5: PI arrows — show iterations visible in zoom range
    first_pi = True
    for k, (v_s, tv_s, v_next) in enumerate(pi_points):
        # Skip if converged (no movement)
        if abs(v_s - v_next) < 1e-4:
            continue
        # Skip if entirely outside zoom
        if v_next > ZOOM_HI and v_s > ZOOM_HI:
            continue
        if v_s < ZOOM_LO and v_next < ZOOM_LO:
            continue

        lbl = 'PI (Newton jumps)' if first_pi else '_nolegend_'

        # Skip iterations starting outside zoom (e.g. iter 0 from -41)
        if v_s < ZOOM_LO or v_s > ZOOM_HI:
            continue
        else:
            # Vertical arrow: from diagonal to T curve
            ax.annotate('', xy=(v_s, tv_s), xytext=(v_s, v_s),
                        arrowprops=dict(arrowstyle='->', color=PI_COLOR,
                                        lw=2.2, shrinkA=0, shrinkB=1),
                        annotation_clip=True)
            # Horizontal arrow: T curve to next diagonal point
            ax.annotate('', xy=(v_next, v_next), xytext=(v_s, tv_s),
                        arrowprops=dict(arrowstyle='->', color=PI_COLOR,
                                        lw=2.2, shrinkA=0, shrinkB=1),
                        annotation_clip=True)
        if first_pi:
            ax.plot([], [], '-', color=PI_COLOR, linewidth=2.2, label=lbl)
        first_pi = False

    # Axes
    ax.set_xlabel('$V(s)$')
    ax.set_ylabel('$TV(s)$')
    ax.set_xlim(ZOOM_LO, ZOOM_HI)
    ax.set_ylim(ZOOM_LO, ZOOM_HI)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(False)

    path = os.path.join(OUTPUT_DIR, 'qlearning_geometry.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true',
                        help='No computation to cache (diagram-only script)')
    parser.add_argument('--plots-only', action='store_true',
                        help='Runs normally (same as no flags)')
    args = parser.parse_args()
    if args.data_only:
        print("No computation to cache (diagram-only script).")
        sys.exit(0)
    generate_outputs()
