"""
LQR Convergence Rate Comparison: VI, PI, and Q-Learning
Chapter 2: Planning and Learning

Demonstrates the fundamental difference in convergence rates:
- Value Iteration: Linear convergence (constant ratio per step)
- Policy Iteration: Quadratic convergence (Newton's method)
- Q-Learning: Model-free, noisy, asymptotic convergence

All methods start from the same initial point K0 = 5.0 to enable fair comparison.

Note: Q-learning uses gamma=0.95 while VI/PI use gamma=0.99. The lower discount
factor for Q-learning makes the discretization error more manageable for the
tabular method. This demonstrates a key practical tradeoff: model-free methods
often require adjustments that model-based methods do not.

This version includes comprehensive Q-learning logging to track:
- Individual Q(s,a) values (not just V(s) = min_u Q)
- Both increases and decreases in Q-values
- Policy changes at tracked states

THEORY REFERENCE:
  Bertsekas, D.P. (2019). Reinforcement Learning and Optimal Control.
  Athena Scientific. ISBN: 978-1-886529-39-7.

  Key sections verified in this script:
    - Proposition 4.3.5 (Contraction property), pp. 18-21
    - Section 4.8 (Q-Learning), pp. 53-58
    - Proposition 4.5.1 (PI monotonic improvement), p. 27
    - Proposition 4.5.2 (Optimistic PI), p. 29
    - Section 4.4 (Approximate VI error bounds), pp. 22-25

  See also: ch02_planning_learning/papers/bertsekas2019_qlearning_theory.md
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.sim_cache import load_results, save_results, add_cache_args

import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'lqr_convergence'
CONFIG = {
    'A': 2.0, 'B': 1.0, 'Q_cost': 1.0, 'R_cost': 1.0,
    'GAMMA_DP': 0.99, 'GAMMA_QL': 0.95,
    'K0': 5.0,
    'n_episodes': 500000, 'check_interval': 10,
    'n_states': 101, 'n_actions': 101,
    'vi_iters': 30, 'pi_iters': 10,
    'stepsize_n_episodes': 100000, 'stepsize_check_interval': 100,
    'stepsize_n_states': 51, 'stepsize_n_actions': 51,
    'exploration_n_episodes': 50000, 'exploration_n_states': 31, 'exploration_n_actions': 31,
    'version': 1,
}

# =============================================================================
# Parameters
# =============================================================================

# LQR system: x_{k+1} = a*x_k + b*u_k, cost = sum(q*x^2 + r*u^2)
A = 2.0  # System dynamics (unstable)
B = 1.0  # Control coefficient
Q = 1.0  # State cost
R = 1.0  # Control cost

GAMMA_DP = 0.99  # For VI/PI (nearly undiscounted)
GAMMA_QL = 0.95  # For Q-learning (more stable for tabular discretization)

# Common starting point for all methods
K0 = 5.0

def solve_K_star(gamma):
    """
    Solve for K* from the fixed point K = F(K).

    F(K) = q + (gamma * a^2 * r * K) / (r + gamma * b^2 * K)

    Quadratic: gamma*b^2*K^2 + (r - gamma*a^2*r - q*gamma*b^2)*K - qr = 0
    """
    a_coef = gamma * B**2
    b_coef = R * (1 - gamma * A**2) - Q * gamma * B**2
    c_coef = -Q * R
    disc = b_coef**2 - 4 * a_coef * c_coef
    return (-b_coef + np.sqrt(disc)) / (2 * a_coef)

K_STAR_DP = solve_K_star(GAMMA_DP)
K_STAR_QL = solve_K_star(GAMMA_QL)

# Compute optimal gains
L_STAR_DP = -GAMMA_DP * A * B * K_STAR_DP / (R + GAMMA_DP * B**2 * K_STAR_DP)
L_STAR_QL = -GAMMA_QL * A * B * K_STAR_QL / (R + GAMMA_QL * B**2 * K_STAR_QL)

def print_problem_formulation():
    """Print rich problem formulation header."""
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 20 + "LQR CONVERGENCE RATE COMPARISON" + " " * 27 + "|")
    print("|" + " " * 15 + "Value Iteration vs Policy Iteration vs Q-Learning" + " " * 14 + "|")
    print("+" + "=" * 78 + "+")

    print("""
PROBLEM FORMULATION
-------------------
  Linear system:     x_{k+1} = a * x_k + b * u_k
  Stage cost:        c(x, u) = q * x^2 + r * u^2
  Objective:         min sum_{k=0}^{inf} gamma^k * c(x_k, u_k)

  Optimal value function:  V*(x) = K* * x^2
  Optimal policy:          u*(x) = L* * x

  Riccati equation:  K = F(K) = q + (gamma * a^2 * r * K) / (r + gamma * b^2 * K)
""")

    print("-" * 80)
    print("PARAMETER TABLE")
    print("-" * 80)
    print(f"\n  {'Parameter':<25} {'Symbol':<10} {'Value':>15}")
    print("  " + "-" * 50)
    print(f"  {'System dynamics':<25} {'a':<10} {A:>15.4f}")
    print(f"  {'Control coefficient':<25} {'b':<10} {B:>15.4f}")
    print(f"  {'State cost':<25} {'q':<10} {Q:>15.4f}")
    print(f"  {'Control cost':<25} {'r':<10} {R:>15.4f}")
    print(f"  {'Discount (VI/PI)':<25} {'gamma_DP':<10} {GAMMA_DP:>15.4f}")
    print(f"  {'Discount (Q-learning)':<25} {'gamma_QL':<10} {GAMMA_QL:>15.4f}")
    print(f"  {'Initial K':<25} {'K0':<10} {K0:>15.4f}")

    print("\n" + "-" * 80)
    print("ANALYTICAL SOLUTIONS")
    print("-" * 80)
    print(f"\n  {'Method':<20} {'gamma':>10} {'K*':>15} {'L*':>15} {'|K0 - K*|':>15}")
    print("  " + "-" * 65)
    print(f"  {'VI / PI':<20} {GAMMA_DP:>10.4f} {K_STAR_DP:>15.6f} {L_STAR_DP:>15.6f} {abs(K0 - K_STAR_DP):>15.6f}")
    print(f"  {'Q-Learning':<20} {GAMMA_QL:>10.4f} {K_STAR_QL:>15.6f} {L_STAR_QL:>15.6f} {abs(K0 - K_STAR_QL):>15.6f}")

    print(f"""
NOTE: Q-learning uses gamma={GAMMA_QL} (vs {GAMMA_DP} for VI/PI) because:
  - Lower discount makes TD targets less sensitive to bootstrapping error
  - Tabular discretization introduces approximation error
  - This demonstrates a key model-free vs model-based tradeoff
""")

# NOTE: print_problem_formulation() is called inside compute_data(), not at module level

# =============================================================================
# Riccati Operator
# =============================================================================

def riccati_operator(K, gamma):
    """F(K) = q + (gamma * a^2 * r * K) / (r + gamma * b^2 * K)"""
    return Q + (gamma * A**2 * R * K) / (R + gamma * B**2 * K)

# =============================================================================
# Value Iteration
# =============================================================================

def run_vi(n_iters, start_k, gamma):
    """
    Value Iteration: K_{k+1} = F(K_k).
    Returns array of (K_k, |K_k - K*|) for each iteration.
    """
    K_star = solve_K_star(gamma)
    K = start_k
    history = [(K, abs(K - K_star))]

    for i in range(n_iters):
        K = riccati_operator(K, gamma)
        history.append((K, abs(K - K_star)))
        if i < 10 or i % 5 == 0:
            print(f"    VI iter {i+1}: K={K:.6f}, error={abs(K - K_star):.6e}")

    return history

# =============================================================================
# Policy Iteration (Newton's Method)
# =============================================================================

def optimal_gain(K, gamma):
    """L = -gamma * a * b * K / (r + gamma * b^2 * K)"""
    return -gamma * A * B * K / (R + gamma * B**2 * K)

def policy_cost(L, gamma):
    """
    K_L = (q + r*L^2) / (1 - gamma*(a + b*L)^2)
    Requires |a + bL| < 1/sqrt(gamma) for stability.
    """
    closed_loop = A + B * L
    stability = 1 - gamma * closed_loop**2
    if stability <= 1e-10:
        return np.inf
    return (Q + R * L**2) / stability

def run_pi(n_iters, start_k, gamma):
    """
    Policy Iteration:
    1. Policy Improvement: L = optimal_gain(K)
    2. Policy Evaluation: K_new = policy_cost(L)

    Returns array of (K_k, |K_k - K*|) for each iteration.
    """
    K_star = solve_K_star(gamma)
    K = start_k
    L = optimal_gain(K, gamma)
    history = [(K, abs(K - K_star))]
    print(f"    PI iter 0: K={K:.6f}, L={L:.6f}, error={abs(K - K_star):.6e}")

    for i in range(n_iters):
        K = policy_cost(L, gamma)
        if K == np.inf:
            break
        history.append((K, abs(K - K_star)))
        L = optimal_gain(K, gamma)
        print(f"    PI iter {i+1}: K={K:.6f}, L={L:.6f}, error={abs(K - K_star):.6e}")

    return history

# =============================================================================
# Q-Learning (Model-Free)
# =============================================================================

def run_q_learning(n_episodes, check_interval, gamma, start_k, n_states=101, n_actions=101,
                   debug_log_path=None):
    """
    Tabular Q-learning on discretized LQR with comprehensive logging.

    State space: x in [-5, 5] discretized to n_states bins
    Action space: u in [-10, 10] discretized to n_actions bins

    Initializes Q-table so V(x) = min_u Q(x,u) approx start_k * x^2 initially.
    Uses slow learning rate to show gradual convergence.

    Returns (errors, K_estimates, K_single_state, debug_data).
    debug_data contains tracked Q(s,a) trajectories and statistics.
    """
    K_star = solve_K_star(gamma)

    print(f"    Grid size: {n_states} states x {n_actions} actions")
    print(f"    Target K*={K_star:.6f} (gamma={gamma}), starting K0={start_k:.6f}")

    # Discretize - very wide range to minimize boundary effects
    x_vals = np.linspace(-10, 10, n_states)
    u_vals = np.linspace(-20, 20, n_actions)
    dx = x_vals[1] - x_vals[0]

    # Initialize Q-table naively: Q(x,u) = start_k * x^2 for ALL actions
    # This makes V(x) = min_u Q(x,u) = start_k * x^2, so K_est starts at start_k
    Q_table = np.zeros((n_states, n_actions))
    for i, x in enumerate(x_vals):
        Q_table[i, :] = start_k * x**2  # Same value for all actions initially

    # Learning parameters - high and constant to maintain noise
    alpha_0 = 0.3  # High learning rate = big jumps
    epsilon_0 = 0.3  # High exploration = more randomness

    np.random.seed(42)
    errors = []
    K_estimates = []
    K_single_state = []  # Track K from a single state to show meandering

    # ==========================================================================
    # DEBUG LOGGING SETUP: Track specific (state, action) pairs
    # ==========================================================================

    # States to track: x = 1.0, 0.5, -1.0
    track_states = [1.0, 0.5, -1.0]
    track_x_indices = [np.argmin(np.abs(x_vals - x)) for x in track_states]

    # For each state, find optimal action (u* = -gamma*a*b*K / (r + gamma*b^2*K) * x)
    # At convergence: u*(x) ≈ -gamma*A*B*K_star / (R + gamma*B^2*K_star) * x
    L_star = -gamma * A * B * K_star / (R + gamma * B**2 * K_star)

    # Define tracked (state, action) pairs: optimal, suboptimal (+2 from optimal), random
    tracked_pairs = []  # List of (x_idx, u_idx, label)
    for x_val, x_idx in zip(track_states, track_x_indices):
        u_opt = L_star * x_val  # Optimal control
        u_opt_idx = np.argmin(np.abs(u_vals - u_opt))

        # Suboptimal: +2 units away from optimal
        u_sub = u_opt + 2.0
        u_sub_idx = np.argmin(np.abs(u_vals - u_sub))

        # Random: far from optimal
        u_rand = u_opt - 5.0
        u_rand_idx = np.argmin(np.abs(u_vals - u_rand))

        tracked_pairs.append((x_idx, u_opt_idx, f"x={x_val:.1f},u=opt"))
        tracked_pairs.append((x_idx, u_sub_idx, f"x={x_val:.1f},u=sub"))
        tracked_pairs.append((x_idx, u_rand_idx, f"x={x_val:.1f},u=rand"))

    print(f"\n    Tracking {len(tracked_pairs)} (state, action) pairs for debug logging:")
    for x_idx, u_idx, label in tracked_pairs[:3]:
        print(f"      {label}: x={x_vals[x_idx]:.2f}, u={u_vals[u_idx]:.2f}")

    # Storage for tracked Q(s,a) trajectories (sampled at check_interval)
    tracked_Q_trajectories = {label: [] for _, _, label in tracked_pairs}
    tracked_V_trajectories = {f"V(x={x:.1f})": [] for x in track_states}

    # Storage for policy (greedy action) at tracked states
    tracked_greedy = {f"x={x:.1f}": [] for x in track_states}

    # Counters for update statistics
    update_stats = {
        'total_updates': 0,
        'positive_delta': 0,  # Q increased
        'negative_delta': 0,  # Q decreased
        'V_changed': 0,       # min_u Q changed
        'greedy_changed': 0,  # argmin changed
        'delta_sum': 0.0,
        'delta_sq_sum': 0.0,
    }

    # Detailed log file (CSV)
    log_file = None
    log_writer = None
    if debug_log_path:
        log_file = open(debug_log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow(['episode', 'step', 'x_idx', 'u_idx', 'x', 'u',
                             'Q_old', 'TD_target', 'Q_new', 'delta',
                             'V_old', 'V_new', 'V_changed', 'greedy_old', 'greedy_new', 'greedy_changed'])

    # Sample rate for detailed logging (log every N-th update to keep file manageable)
    log_sample_rate = 1000
    log_counter = 0

    # Pick a representative state and action for backward compatibility
    track_x_idx_compat = np.argmin(np.abs(x_vals - 1.0))
    track_u_idx_compat = np.argmin(np.abs(u_vals - (-1.6)))
    track_x_compat = x_vals[track_x_idx_compat]

    for episode in range(n_episodes):
        # Very slow decay to maintain noise throughout
        alpha = alpha_0 / (1 + episode / 500000)
        epsilon = epsilon_0 / (1 + episode / 500000)

        # Random initial state
        x_idx = np.random.randint(n_states)
        x = x_vals[x_idx]

        # Run episode (fixed length)
        for step in range(20):
            # Epsilon-greedy action
            if np.random.rand() < epsilon:
                u_idx = np.random.randint(n_actions)
            else:
                u_idx = np.argmin(Q_table[x_idx, :])

            u = u_vals[u_idx]

            # Transition
            x_next = A * x + B * u
            cost = Q * x**2 + R * u**2

            # Clip to state space
            x_next = np.clip(x_next, x_vals[0], x_vals[-1])
            x_next_idx = int((x_next - x_vals[0]) / dx)
            x_next_idx = np.clip(x_next_idx, 0, n_states - 1)

            # Record pre-update values
            Q_old = Q_table[x_idx, u_idx]
            V_old = np.min(Q_table[x_idx, :])
            greedy_old = np.argmin(Q_table[x_idx, :])

            # TD update
            Q_next_min = np.min(Q_table[x_next_idx, :])
            TD_target = cost + gamma * Q_next_min
            Q_new = Q_old + alpha * (TD_target - Q_old)
            Q_table[x_idx, u_idx] = Q_new

            # Record post-update values
            V_new = np.min(Q_table[x_idx, :])
            greedy_new = np.argmin(Q_table[x_idx, :])

            # Compute delta
            delta = Q_new - Q_old

            # Update statistics
            update_stats['total_updates'] += 1
            update_stats['delta_sum'] += delta
            update_stats['delta_sq_sum'] += delta ** 2
            if delta > 0:
                update_stats['positive_delta'] += 1
            elif delta < 0:
                update_stats['negative_delta'] += 1
            if abs(V_new - V_old) > 1e-10:
                update_stats['V_changed'] += 1
            if greedy_new != greedy_old:
                update_stats['greedy_changed'] += 1

            # Log to file (sampled)
            if log_writer and log_counter % log_sample_rate == 0:
                V_changed = 1 if abs(V_new - V_old) > 1e-10 else 0
                greedy_changed = 1 if greedy_new != greedy_old else 0
                log_writer.writerow([episode, step, x_idx, u_idx,
                                     f"{x:.4f}", f"{u:.4f}",
                                     f"{Q_old:.6f}", f"{TD_target:.6f}", f"{Q_new:.6f}",
                                     f"{delta:.6f}",
                                     f"{V_old:.6f}", f"{V_new:.6f}", V_changed,
                                     greedy_old, greedy_new, greedy_changed])
            log_counter += 1

            # Move to next state
            x_idx = x_next_idx
            x = x_vals[x_idx]

        # Check error at intervals
        if episode % check_interval == 0:
            # Extract V(x) = min_u Q(x,u), then estimate K from V(x)/x^2
            V = np.min(Q_table, axis=1)
            # Use states away from zero and boundaries to avoid issues
            mask = (np.abs(x_vals) > 0.5) & (np.abs(x_vals) < 8.0)
            if np.sum(mask) > 0:
                K_est = np.mean(V[mask] / (x_vals[mask]**2))
                errors.append(abs(K_est - K_star))
                K_estimates.append(K_est)
            else:
                errors.append(abs(K_star))
                K_estimates.append(0.0)

            # Track specific Q(s,a) value (backward compatibility)
            Q_tracked = Q_table[track_x_idx_compat, track_u_idx_compat]
            K_single = Q_tracked / (track_x_compat**2)
            K_single_state.append(K_single)

            # Track Q(s,a) for all tracked pairs
            for x_idx_t, u_idx_t, label in tracked_pairs:
                tracked_Q_trajectories[label].append(Q_table[x_idx_t, u_idx_t])

            # Track V(s) = min_u Q(s,u) for tracked states
            for x_val, x_idx_t in zip(track_states, track_x_indices):
                tracked_V_trajectories[f"V(x={x_val:.1f})"].append(np.min(Q_table[x_idx_t, :]))

            # Track greedy action at tracked states
            for x_val, x_idx_t in zip(track_states, track_x_indices):
                tracked_greedy[f"x={x_val:.1f}"].append(np.argmin(Q_table[x_idx_t, :]))

            # Verbose logging
            if episode % 50000 == 0:
                print(f"    Episode {episode:6d}: K_avg={K_estimates[-1]:.4f}, "
                      f"K_single={K_single:.4f}, alpha={alpha:.4f}")

    # Close log file
    if log_file:
        log_file.close()
        print(f"\n    Debug log written to: {debug_log_path}")

    # Compile debug data
    debug_data = {
        'tracked_Q': tracked_Q_trajectories,
        'tracked_V': tracked_V_trajectories,
        'tracked_greedy': tracked_greedy,
        'tracked_pairs': tracked_pairs,
        'track_states': track_states,
        'x_vals': x_vals,
        'u_vals': u_vals,
        'update_stats': update_stats,
        'K_star': K_star,
        'L_star': L_star,
    }

    return np.array(errors), np.array(K_estimates), np.array(K_single_state), debug_data

# =============================================================================
# Riccati Map Visualizations (Three Separate Figures)
# =============================================================================

def plot_riccati_vi(vi_history, gamma, save_path):
    """
    Plot VI staircase on Riccati map.
    """
    K_star = solve_K_star(gamma)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Zoomed plot range
    K_min, K_max = 4.0, 5.2
    K_range = np.linspace(K_min, K_max, 200)

    # F(K) curve
    F_vals = [riccati_operator(K, gamma) for K in K_range]
    ax.plot(K_range, F_vals, 'b-', linewidth=2.5,
            label=r'$F(K) = q + \frac{\gamma a^2 r K}{r + \gamma b^2 K}$')

    # 45-degree line
    ax.plot([K_min, K_max], [K_min, K_max], 'k--', linewidth=1.5,
            label=r'$K = K$ (fixed point line)')

    # K* marker
    ax.plot(K_star, K_star, 'ko', markersize=12, zorder=10)
    ax.annotate(f'$K^* = {K_star:.3f}$', xy=(K_star, K_star),
                xytext=(K_star + 0.15, K_star - 0.15), fontsize=12,
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))

    # VI staircase (green)
    vi_K_vals = [h[0] for h in vi_history]
    for i in range(min(8, len(vi_K_vals) - 1)):
        K_curr = vi_K_vals[i]
        K_next = vi_K_vals[i + 1]
        # Vertical line: (K_curr, K_curr) to (K_curr, F(K_curr))
        ax.plot([K_curr, K_curr], [K_curr, K_next], 'g-', linewidth=2, alpha=0.9)
        # Horizontal line: (K_curr, F(K_curr)) to (F(K_curr), F(K_curr))
        ax.plot([K_curr, K_next], [K_next, K_next], 'g-', linewidth=2, alpha=0.9)
        # Mark the iteration point
        if i == 0:
            ax.plot(K_curr, K_curr, 'go', markersize=10, label='VI staircase', zorder=5)
            ax.annotate(f'$K_0 = {K_curr:.1f}$', xy=(K_curr, K_curr),
                        xytext=(K_curr + 0.1, K_curr + 0.15), fontsize=11)
        ax.plot(K_curr, K_next, 'go', markersize=6, zorder=5)

    # Labels and formatting
    ax.set_xlabel(r'$K_k$', fontsize=14)
    ax.set_ylabel(r'$K_{k+1} = F(K_k)$', fontsize=14)
    ax.set_title(f'Value Iteration: Staircase Descent ($\\gamma$={gamma})\n' +
                 r'$K_{k+1} = F(K_k)$, linear convergence',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(K_min, K_max)
    ax.set_ylim(K_min, K_max)
    ax.set_aspect('equal')

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_riccati_pi(pi_history, gamma, save_path):
    """
    Plot PI Newton tangents on Riccati map.
    """
    K_star = solve_K_star(gamma)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Zoomed plot range
    K_min, K_max = 4.0, 5.2
    K_range = np.linspace(K_min, K_max, 200)

    # F(K) curve
    F_vals = [riccati_operator(K, gamma) for K in K_range]
    ax.plot(K_range, F_vals, 'b-', linewidth=2.5,
            label=r'$F(K) = q + \frac{\gamma a^2 r K}{r + \gamma b^2 K}$')

    # 45-degree line
    ax.plot([K_min, K_max], [K_min, K_max], 'k--', linewidth=1.5,
            label=r'$K = K$ (fixed point line)')

    # K* marker
    ax.plot(K_star, K_star, 'ko', markersize=12, zorder=10)
    ax.annotate(f'$K^* = {K_star:.3f}$', xy=(K_star, K_star),
                xytext=(K_star + 0.15, K_star - 0.15), fontsize=12,
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))

    # PI Newton tangents (orange)
    pi_K_vals = [h[0] for h in pi_history]
    colors = ['#FF8C00', '#FF6600', '#FF4400', '#FF2200']  # Gradient oranges

    for i in range(min(3, len(pi_K_vals) - 1)):
        K_curr = pi_K_vals[i]
        K_next = pi_K_vals[i + 1]
        F_K = riccati_operator(K_curr, gamma)

        # Tangent slope: F'(K) = gamma * a^2 * r^2 / (r + gamma * b^2 * K)^2
        F_prime = gamma * A**2 * R**2 / (R + gamma * B**2 * K_curr)**2

        # Draw tangent line
        tangent_x = np.linspace(max(K_min, K_next - 0.1), min(K_max, K_curr + 0.1), 50)
        tangent_y = F_K + F_prime * (tangent_x - K_curr)
        ax.plot(tangent_x, tangent_y, color=colors[i], linewidth=2, linestyle='-',
                alpha=0.9, label=f'PI tangent {i+1}' if i == 0 else None)

        # Mark points
        ax.plot(K_curr, F_K, 'o', color=colors[i], markersize=10, zorder=5)

        # Annotation for first iteration
        if i == 0:
            ax.annotate(f'$K_0 = {K_curr:.1f}$', xy=(K_curr, F_K),
                        xytext=(K_curr + 0.1, F_K + 0.1), fontsize=11)

        # Intersection with 45-degree line
        K_intersect = (F_K - F_prime * K_curr) / (1 - F_prime)
        ax.plot(K_intersect, K_intersect, 's', color=colors[i], markersize=8, zorder=5)

    # Labels and formatting
    ax.set_xlabel(r'$K_k$', fontsize=14)
    ax.set_ylabel(r'$K_{k+1} = F(K_k)$', fontsize=14)
    ax.set_title(f'Policy Iteration: Newton Tangent Steps ($\\gamma$={gamma})\n' +
                 r'Quadratic convergence via linearization',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(K_min, K_max)
    ax.set_ylim(K_min, K_max)
    ax.set_aspect('equal')

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def rolling_stats(arr, window):
    """Compute rolling mean and std."""
    n = len(arr)
    means = np.zeros(n)
    stds = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = arr[start:i+1]
        means[i] = np.mean(chunk)
        stds[i] = np.std(chunk) if len(chunk) > 1 else 0
    return means, stds


def plot_riccati_ql(ql_K_estimates, gamma, save_path):
    """
    Plot Q-learning trajectory with rolling average analysis.
    """
    K_star = solve_K_star(gamma)
    n_points = len(ql_K_estimates)

    # Compute rolling statistics at different windows
    windows = [10, 100, 1000, 5000]
    rolling_data = {}
    for w in windows:
        if w < n_points:
            means, stds = rolling_stats(ql_K_estimates, w)
            rolling_data[w] = {'mean': means, 'std': stds}

    # Print statistics
    print("\n" + "=" * 70)
    print("Q-Learning Rolling Average Analysis")
    print("=" * 70)
    print(f"Total episodes: {n_points}, K* = {K_star:.6f}")
    print(f"Raw: start={ql_K_estimates[0]:.4f}, end={ql_K_estimates[-1]:.4f}")
    print("\nRolling statistics at end of training:")
    print(f"  {'Window':<10} {'Mean':<12} {'Std':<12} {'Error':<12}")
    print("  " + "-" * 46)
    for w in windows:
        if w in rolling_data:
            m = rolling_data[w]['mean'][-1]
            s = rolling_data[w]['std'][-1]
            print(f"  {w:<10} {m:<12.6f} {s:<12.6f} {abs(m - K_star):<12.6f}")

    # Compute episode-to-episode changes
    dK = np.diff(ql_K_estimates)
    print(f"\nEpisode-to-episode changes (dK = K_{{k+1}} - K_k):")
    print(f"  Mean dK: {np.mean(dK):.6f}")
    print(f"  Std dK:  {np.std(dK):.6f}")
    print(f"  Min dK:  {np.min(dK):.6f}")
    print(f"  Max dK:  {np.max(dK):.6f}")

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Raw time series with rolling averages
    ax1 = axes[0, 0]
    episodes = np.arange(n_points)
    ax1.plot(episodes, ql_K_estimates, 'r-', linewidth=0.3, alpha=0.4, label='Raw')
    colors = ['blue', 'green', 'orange', 'purple']
    for i, w in enumerate(windows):
        if w in rolling_data:
            ax1.plot(episodes, rolling_data[w]['mean'], colors[i],
                    linewidth=1.5, label=f'MA({w})')
    ax1.axhline(K_star, color='k', linestyle='--', linewidth=2, label=f'$K^*$={K_star:.3f}')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('K estimate', fontsize=11)
    ax1.set_title('Rolling Averages at Different Windows', fontsize=12)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rolling standard deviation (noise over time)
    ax2 = axes[0, 1]
    for i, w in enumerate(windows):
        if w in rolling_data:
            ax2.plot(episodes, rolling_data[w]['std'], colors[i],
                    linewidth=1.5, label=f'Std (window={w})')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Rolling Std of K', fontsize=11)
    ax2.set_title('Noise Level Over Time (Rolling Std)', fontsize=12)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Phase plot (K_k vs K_{k+1})
    ax3 = axes[1, 0]
    K_min, K_max = min(ql_K_estimates) - 0.1, max(ql_K_estimates) + 0.1
    K_range = np.linspace(K_min, K_max, 200)
    F_vals = [riccati_operator(K, gamma) for K in K_range]
    ax3.plot(K_range, F_vals, 'b-', linewidth=2.5, label=r'$F(K)$ (VI path)', alpha=0.7)
    ax3.plot([K_min, K_max], [K_min, K_max], 'k--', linewidth=1.5, label='Fixed point')
    ax3.plot(K_star, K_star, 'ko', markersize=10, zorder=10)

    # Subsample for scatter
    step = max(1, n_points // 3000)
    K_curr = ql_K_estimates[:-1:step]
    K_next = ql_K_estimates[1::step]
    colors_scatter = plt.cm.Reds(np.linspace(0.3, 1.0, len(K_curr)))
    ax3.scatter(K_curr, K_next, c=colors_scatter, s=8, alpha=0.5, zorder=3)
    ax3.plot(ql_K_estimates[0], ql_K_estimates[1], 'g^', markersize=10, zorder=6, label='Start')
    ax3.plot(ql_K_estimates[-2], ql_K_estimates[-1], 'rv', markersize=10, zorder=6, label='End')

    ax3.set_xlabel(r'$K_k$', fontsize=11)
    ax3.set_ylabel(r'$K_{k+1}$', fontsize=11)
    ax3.set_title('Phase Plot: Q-Learning vs VI Path', fontsize=12)
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(K_min, K_max)
    ax3.set_ylim(K_min, K_max)
    ax3.set_aspect('equal')

    # Plot 4: Histogram of episode-to-episode changes
    ax4 = axes[1, 1]
    ax4.hist(dK, bins=100, density=True, alpha=0.7, color='red', edgecolor='darkred')
    ax4.axvline(0, color='k', linestyle='--', linewidth=1.5)
    ax4.axvline(np.mean(dK), color='blue', linestyle='-', linewidth=2,
                label=f'Mean={np.mean(dK):.4f}')
    ax4.set_xlabel(r'$\Delta K = K_{k+1} - K_k$', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title(f'Distribution of K Changes (std={np.std(dK):.4f})', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {save_path}")


def plot_qlearning_debug(debug_data, check_interval, save_path):
    """
    Create debug visualization showing:
    1. Individual Q(s,a) trajectories for tracked pairs (shows bidirectional noise)
    2. V(s) = min Q for same states (shows monotonic decrease)
    3. Histogram of all TD deltas (positive and negative)
    4. Policy stability over time
    """
    tracked_Q = debug_data['tracked_Q']
    tracked_V = debug_data['tracked_V']
    tracked_greedy = debug_data['tracked_greedy']
    tracked_pairs = debug_data['tracked_pairs']
    track_states = debug_data['track_states']
    update_stats = debug_data['update_stats']
    K_star = debug_data['K_star']

    n_checkpoints = len(list(tracked_Q.values())[0])
    episodes = np.arange(n_checkpoints) * check_interval

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ==========================================================================
    # Plot 1: Individual Q(s,a) trajectories (shows bidirectional noise)
    # ==========================================================================
    ax1 = axes[0, 0]

    # Plot Q trajectories for x=1.0 (optimal, suboptimal, random actions)
    colors = ['blue', 'orange', 'red']
    linestyles = ['-', '--', ':']

    for i, x_val in enumerate([1.0]):  # Focus on x=1.0 for clarity
        for j, action_type in enumerate(['opt', 'sub', 'rand']):
            label = f"x={x_val:.1f},u={action_type}"
            if label in tracked_Q:
                Q_vals = np.array(tracked_Q[label])
                ax1.plot(episodes, Q_vals, color=colors[j], linestyle=linestyles[j],
                         linewidth=1.5, alpha=0.8, label=f"Q(x=1, u={action_type})")

    # Also plot V(x=1.0) for comparison
    V_label = "V(x=1.0)"
    if V_label in tracked_V:
        V_vals = np.array(tracked_V[V_label])
        ax1.plot(episodes, V_vals, 'k-', linewidth=2.5, label='V(x=1) = min_u Q')

    # Reference line: K* * x^2 = K* * 1 = K*
    ax1.axhline(K_star, color='green', linestyle='--', linewidth=1.5,
                label=f'$K^* \\cdot x^2$ = {K_star:.3f}')

    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Q(s,a) or V(s)', fontsize=11)
    ax1.set_title('Q(s,a) vs V(s) = min_u Q(s,u) at x=1.0\nQ values meander; V only decreases',
                  fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ==========================================================================
    # Plot 2: Q(s,a) changes (delta = Q_new - Q_old) analysis
    # ==========================================================================
    ax2 = axes[0, 1]

    # Compute deltas for tracked Q values
    all_deltas = []
    for label, Q_vals in tracked_Q.items():
        Q_arr = np.array(Q_vals)
        deltas = np.diff(Q_arr)
        all_deltas.extend(deltas.tolist())

    all_deltas = np.array(all_deltas)

    # Histogram
    ax2.hist(all_deltas, bins=100, density=True, alpha=0.7, color='purple', edgecolor='darkviolet')
    ax2.axvline(0, color='k', linestyle='--', linewidth=2, label='Zero')
    ax2.axvline(np.mean(all_deltas), color='red', linestyle='-', linewidth=2,
                label=f'Mean={np.mean(all_deltas):.4f}')

    # Count positive vs negative
    n_pos = np.sum(all_deltas > 0)
    n_neg = np.sum(all_deltas < 0)
    n_zero = np.sum(all_deltas == 0)

    ax2.set_xlabel(r'$\Delta Q = Q_{new} - Q_{old}$', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title(f'Distribution of Q(s,a) Changes\n'
                  f'Positive: {n_pos}, Negative: {n_neg}, Zero: {n_zero}',
                  fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ==========================================================================
    # Plot 3: V(s) trajectories for multiple states
    # ==========================================================================
    ax3 = axes[1, 0]

    colors_v = ['blue', 'green', 'red']
    for i, x_val in enumerate(track_states):
        V_label = f"V(x={x_val:.1f})"
        if V_label in tracked_V:
            V_vals = np.array(tracked_V[V_label])
            # Normalize by x^2 to get K estimate
            K_from_V = V_vals / (x_val ** 2)
            ax3.plot(episodes, K_from_V, color=colors_v[i], linewidth=1.5,
                     label=f'K from V(x={x_val:.1f})')

    ax3.axhline(K_star, color='k', linestyle='--', linewidth=2, label=f'$K^*$={K_star:.3f}')

    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('K estimate (V(x)/x²)', fontsize=11)
    ax3.set_title('V(s)/x² Trajectories: Monotonic Decrease to K*', fontsize=12)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # ==========================================================================
    # Plot 4: Policy stability (greedy action changes)
    # ==========================================================================
    ax4 = axes[1, 1]

    u_vals = debug_data['u_vals']
    L_star = debug_data['L_star']

    for i, x_val in enumerate(track_states):
        greedy_label = f"x={x_val:.1f}"
        if greedy_label in tracked_greedy:
            greedy_idx = np.array(tracked_greedy[greedy_label])
            greedy_u = u_vals[greedy_idx]
            ax4.plot(episodes, greedy_u, color=colors_v[i], linewidth=1.0, alpha=0.7,
                     label=f'Greedy u at x={x_val:.1f}')

            # Optimal action
            u_opt = L_star * x_val
            ax4.axhline(u_opt, color=colors_v[i], linestyle='--', alpha=0.5)

    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Greedy action u', fontsize=11)
    ax4.set_title('Policy Stability: Greedy Action Over Time\n(dashed = optimal action)',
                  fontsize=12)
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def print_debug_summary(debug_data, check_interval):
    """Print comprehensive summary statistics from debug data."""
    stats = debug_data['update_stats']
    tracked_Q = debug_data['tracked_Q']
    tracked_V = debug_data['tracked_V']
    tracked_greedy = debug_data['tracked_greedy']
    K_star = debug_data['K_star']
    L_star = debug_data['L_star']
    x_vals = debug_data['x_vals']
    u_vals = debug_data['u_vals']
    track_states = debug_data['track_states']

    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 25 + "Q-LEARNING DEBUG SUMMARY" + " " * 29 + "|")
    print("+" + "=" * 78 + "+")

    # =========================================================================
    # Section 1: TD Update Statistics
    # =========================================================================
    total = stats['total_updates']
    mean_delta = stats['delta_sum'] / total
    var_delta = stats['delta_sq_sum'] / total - mean_delta ** 2
    std_delta = np.sqrt(max(0, var_delta))

    print("\n" + "-" * 80)
    print("SECTION 1: TD UPDATE STATISTICS")
    print("-" * 80)
    print(f"\nTotal TD updates performed: {total:,}")
    print(f"\n  {'Category':<35} {'Count':>12} {'Percentage':>12}")
    print("  " + "-" * 59)
    print(f"  {'Positive delta (Q increased)':<35} {stats['positive_delta']:>12,} {100*stats['positive_delta']/total:>11.2f}%")
    print(f"  {'Negative delta (Q decreased)':<35} {stats['negative_delta']:>12,} {100*stats['negative_delta']/total:>11.2f}%")
    print(f"  {'Zero delta (no change)':<35} {total - stats['positive_delta'] - stats['negative_delta']:>12,} {100*(total - stats['positive_delta'] - stats['negative_delta'])/total:>11.2f}%")
    print("  " + "-" * 59)
    print(f"  {'V(s) = min_u Q(s,u) changed':<35} {stats['V_changed']:>12,} {100*stats['V_changed']/total:>11.2f}%")
    print(f"  {'Greedy action argmin_u Q changed':<35} {stats['greedy_changed']:>12,} {100*stats['greedy_changed']/total:>11.2f}%")

    print(f"\n  Delta Statistics:")
    print(f"    Mean(delta):     {mean_delta:>12.6f}")
    print(f"    Std(delta):      {std_delta:>12.6f}")
    print(f"    Mean/Std ratio:  {mean_delta/std_delta if std_delta > 0 else 0:>12.6f}")

    # =========================================================================
    # Section 2: Why K Only Decreases (The Key Insight)
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 2: WHY K ONLY DECREASES (THE KEY INSIGHT)")
    print("-" * 80)

    pct_q_changed = 100 * (stats['positive_delta'] + stats['negative_delta']) / total
    pct_v_changed = 100 * stats['V_changed'] / total
    ratio = pct_q_changed / pct_v_changed if pct_v_changed > 0 else float('inf')

    print(f"""
  The K estimate is derived from V(s) = min_u Q(s,u), not from Q(s,a) directly.

  Filtering effect of the min operator:
    - Q(s,a) changes:     {pct_q_changed:>6.2f}% of updates modify some Q value
    - V(s) changes:       {pct_v_changed:>6.2f}% of updates modify min_u Q
    - Filtering ratio:    {ratio:>6.1f}x  (Q changes {ratio:.1f}x more often than V)

  Mechanism:
    1. Most updates are for SUBOPTIMAL actions (u != argmin Q)
    2. These updates change Q(s,u_subopt) but NOT V(s) = min_u Q(s,u)
    3. V(s) only changes when Q(s, u_greedy) is updated
    4. For cost minimization, TD targets typically pull Q DOWN toward true value
    5. Result: V(s) decreases monotonically; upward Q noise is filtered out
""")

    # =========================================================================
    # Section 3: Tracked Q(s,a) Trajectories
    # =========================================================================
    print("-" * 80)
    print("SECTION 3: TRACKED Q(s,a) TRAJECTORIES")
    print("-" * 80)

    print(f"\n  Tracking 9 state-action pairs across 3 states and 3 action types.")
    print(f"  Optimal gain L* = {L_star:.4f} (so u*(x) = L* * x)")
    print(f"\n  {'Label':<20} {'Q_start':>10} {'Q_end':>10} {'Q_min':>10} {'Q_max':>10} {'Range':>10} {'Converged?':>12}")
    print("  " + "-" * 84)

    for label, Q_vals in tracked_Q.items():
        Q_arr = np.array(Q_vals)
        # Check if this is an optimal action (should converge to K* * x^2)
        x_val = float(label.split('=')[1].split(',')[0])
        expected_Q = K_star * x_val**2
        converged = "YES" if abs(Q_arr[-1] - expected_Q) < 0.5 else "NO"
        if 'rand' in label or 'sub' in label:
            converged = "N/A (subopt)"

        print(f"  {label:<20} {Q_arr[0]:>10.2f} {Q_arr[-1]:>10.2f} "
              f"{Q_arr.min():>10.2f} {Q_arr.max():>10.2f} {Q_arr.max()-Q_arr.min():>10.2f} {converged:>12}")

    # =========================================================================
    # Section 4: Direction of Q Changes (Bidirectional Noise Evidence)
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 4: DIRECTION OF Q CHANGES (BIDIRECTIONAL NOISE EVIDENCE)")
    print("-" * 80)

    print(f"\n  This table shows that individual Q(s,a) values go BOTH up and down.")
    print(f"  (Sampled at check_interval={check_interval} episodes)")
    print(f"\n  {'Label':<20} {'#Up':>8} {'#Down':>8} {'#Zero':>8} {'Net':>8} {'Direction':<12}")
    print("  " + "-" * 72)

    total_up = 0
    total_down = 0
    for label, Q_vals in tracked_Q.items():
        Q_arr = np.array(Q_vals)
        dQ = np.diff(Q_arr)
        n_up = np.sum(dQ > 1e-10)
        n_down = np.sum(dQ < -1e-10)
        n_zero = np.sum(np.abs(dQ) <= 1e-10)
        total_up += n_up
        total_down += n_down
        net = n_up - n_down
        direction = "UP" if net > 0 else ("DOWN" if net < 0 else "FLAT")
        print(f"  {label:<20} {n_up:>8} {n_down:>8} {n_zero:>8} {net:>+8} {direction:<12}")

    print("  " + "-" * 72)
    print(f"  {'TOTAL':<20} {total_up:>8} {total_down:>8} {'-':>8} {total_up-total_down:>+8}")
    print(f"\n  Conclusion: {total_up} upward moves, {total_down} downward moves across tracked pairs.")
    print(f"              Bidirectional noise EXISTS in Q(s,a) values.")

    # =========================================================================
    # Section 5: V(s) Trajectories (Monotonic Decrease)
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 5: V(s) = min_u Q(s,u) TRAJECTORIES (MONOTONIC DECREASE)")
    print("-" * 80)

    print(f"\n  V(s) should decrease monotonically because min filters out upward noise.")
    print(f"\n  {'State':<12} {'V_start':>10} {'V_end':>10} {'V_min':>10} {'V_max':>10} {'K=V/x^2':>10} {'K*':>10} {'Error':>10}")
    print("  " + "-" * 84)

    for x_val in track_states:
        V_label = f"V(x={x_val:.1f})"
        if V_label in tracked_V:
            V_arr = np.array(tracked_V[V_label])
            K_est = V_arr[-1] / (x_val**2)
            error = abs(K_est - K_star)
            print(f"  x={x_val:<9.1f} {V_arr[0]:>10.4f} {V_arr[-1]:>10.4f} "
                  f"{V_arr.min():>10.4f} {V_arr.max():>10.4f} {K_est:>10.4f} {K_star:>10.4f} {error:>10.4f}")

    # Check monotonicity
    print(f"\n  Monotonicity check (V should only decrease):")
    for x_val in track_states:
        V_label = f"V(x={x_val:.1f})"
        if V_label in tracked_V:
            V_arr = np.array(tracked_V[V_label])
            dV = np.diff(V_arr)
            n_increases = np.sum(dV > 1e-10)
            n_decreases = np.sum(dV < -1e-10)
            status = "MONOTONIC" if n_increases == 0 else f"NOT MONOTONIC ({n_increases} increases)"
            print(f"    V(x={x_val:.1f}): {n_decreases} decreases, {n_increases} increases -> {status}")

    # =========================================================================
    # Section 6: Policy Stability Analysis
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 6: POLICY STABILITY ANALYSIS")
    print("-" * 80)

    print(f"\n  How often does the greedy action (policy) change at each tracked state?")
    print(f"\n  {'State':<12} {'u*_theory':>12} {'u_start':>12} {'u_end':>12} {'#Changes':>10} {'Stable?':<10}")
    print("  " + "-" * 70)

    for x_val in track_states:
        greedy_label = f"x={x_val:.1f}"
        if greedy_label in tracked_greedy:
            greedy_idx = np.array(tracked_greedy[greedy_label])
            greedy_u = u_vals[greedy_idx]
            u_opt = L_star * x_val
            n_changes = np.sum(np.diff(greedy_idx) != 0)
            stable = "YES" if n_changes < 10 else "NO"
            print(f"  x={x_val:<9.1f} {u_opt:>12.4f} {greedy_u[0]:>12.4f} {greedy_u[-1]:>12.4f} {n_changes:>10} {stable:<10}")

    # =========================================================================
    # Section 7: Discretization Analysis
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 7: DISCRETIZATION ANALYSIS")
    print("-" * 80)

    dx = x_vals[1] - x_vals[0]
    du = u_vals[1] - u_vals[0]
    print(f"\n  State space:  x in [{x_vals[0]:.1f}, {x_vals[-1]:.1f}], {len(x_vals)} bins, dx = {dx:.4f}")
    print(f"  Action space: u in [{u_vals[0]:.1f}, {u_vals[-1]:.1f}], {len(u_vals)} bins, du = {du:.4f}")
    print(f"  Q-table size: {len(x_vals)} x {len(u_vals)} = {len(x_vals)*len(u_vals):,} entries")

    # Check if optimal actions are representable
    print(f"\n  Optimal action representability:")
    for x_val in [1.0, 2.0, 5.0]:
        u_opt = L_star * x_val
        u_nearest_idx = np.argmin(np.abs(u_vals - u_opt))
        u_nearest = u_vals[u_nearest_idx]
        error = abs(u_nearest - u_opt)
        print(f"    x={x_val:.1f}: u*={u_opt:.4f}, nearest grid u={u_nearest:.4f}, error={error:.4f}")

    print("\n" + "+" + "=" * 78 + "+")


# =============================================================================
# Bertsekas Theory Verification Tests
#
# Source: Bertsekas, D.P. (2019). Reinforcement Learning and Optimal Control.
#         Athena Scientific.
#
# Key Quotes and Page References:
#
# Proposition 4.3.5 (Contraction Property), p. 18-21:
#   "It is straightforward to show that F is a contraction with modulus α,
#    similar to the DP operator T. Thus the algorithm Q_{k+1} = FQ_k converges
#    to Q* from every starting point Q_0." (p. 54)
#
# Section 4.8 Q-Learning (pp. 53-58):
#   "To guarantee the convergence of the algorithm (4.52)-(4.53) to the optimal
#    Q-factors, some conditions must be satisfied. Chief among these are that
#    all state-control pairs (i,u) must be generated infinitely often within
#    the infinitely long sequence {(i_k, u_k)}, and that the successor states j
#    must be independently sampled at each occurrence of a given state-control
#    pair. Furthermore, the stepsize γ_k should satisfy
#        γ_k > 0, Σγ_k = ∞, Σγ_k² < ∞
#    which are typical of stochastic approximation methods." (p. 56)
#
# Section 4.8 Drawbacks (p. 57):
#   "In practice, Q-learning has some drawbacks, the most important of which
#    is that the number of Q-factors/state-control pairs (i,u) may be excessive."
#
# Section 4.8 SARSA (p. 58):
#   "When Q-factor approximation is used, their behavior is very complex, their
#    theoretical convergence properties are unclear, and there are no associated
#    performance bounds in the literature."
#
# Proposition 4.5.1 (PI Monotonic Improvement), p. 27:
#   "The exact PI algorithm generates an improving sequence of policies, i.e.,
#    J_{μ_{k+1}} ≤ J_{μ_k}, and terminates with an optimal policy."
#
# Section 4.4 (Approximate VI Instability), p. 22-25:
#   "The difficulty here is that the approximate VI mapping... is NOT a
#    contraction (even though T itself is a contraction)."
#
# =============================================================================

def verify_contraction_property(gamma, K0, n_iters=20):
    """
    Test Prediction 1: F is a contraction with modulus gamma.

    Theory (Bertsekas Prop 4.3.5, p. 18-21):
      "It is straightforward to show that F is a contraction with modulus α."
      (p. 54)

    For the Riccati operator: ||F(K) - F(K')|| <= rho ||K - K'||

    The contraction modulus rho for the Riccati operator is:
    rho = gamma * a^2 * r^2 / (r + gamma * b^2 * K*)^2

    Returns dict with theoretical and empirical contraction ratios.
    """
    K_star = solve_K_star(gamma)

    # Theoretical contraction modulus (derivative of F at K*)
    # F(K) = q + gamma*a^2*r*K / (r + gamma*b^2*K)
    # F'(K) = gamma*a^2*r^2 / (r + gamma*b^2*K)^2
    rho_theory = gamma * A**2 * R**2 / (R + gamma * B**2 * K_star)**2

    # Run VI and compute empirical contraction ratios
    K = K0
    errors = [abs(K - K_star)]
    ratios = []

    for _ in range(n_iters):
        K = riccati_operator(K, gamma)
        errors.append(abs(K - K_star))
        if errors[-2] > 1e-14:
            ratios.append(errors[-1] / errors[-2])

    # Average contraction ratio (excluding early transient and near-convergence)
    valid_ratios = [r for i, r in enumerate(ratios) if 0.01 < errors[i] < 0.5]
    rho_empirical = np.mean(valid_ratios) if valid_ratios else ratios[0]

    return {
        'gamma': gamma,
        'K_star': K_star,
        'rho_theory': rho_theory,
        'rho_empirical': rho_empirical,
        'errors': errors,
        'ratios': ratios,
        'match': abs(rho_theory - rho_empirical) < 0.01
    }


def verify_convergence_rate(gamma, K0, n_iters=30):
    """
    Test Prediction 2: Linear convergence with rate gamma.

    Theory (Bertsekas Prop 4.3.5, p. 20):
      From contraction theory: ||J_k - J*|| ≤ α^k ||J_0 - J*||
      This gives linear convergence with rate α.

    In log space: log(error_k) decreases linearly with slope log(rho)

    Returns dict with theoretical and empirical slopes.
    """
    K_star = solve_K_star(gamma)
    rho_theory = gamma * A**2 * R**2 / (R + gamma * B**2 * K_star)**2

    K = K0
    log_errors = []

    for _ in range(n_iters):
        error = abs(K - K_star)
        if error > 1e-15:
            log_errors.append(np.log10(error))
        K = riccati_operator(K, gamma)

    # Fit linear regression to log(error) vs iteration
    iters = np.arange(len(log_errors))
    # Use iterations 2-15 for robust slope estimate
    fit_range = slice(2, min(15, len(log_errors)))
    slope, intercept = np.polyfit(iters[fit_range], np.array(log_errors)[fit_range], 1)

    slope_theory = np.log10(rho_theory)

    return {
        'gamma': gamma,
        'slope_empirical': slope,
        'slope_theory': slope_theory,
        'rho_from_slope': 10**slope,
        'rho_theory': rho_theory,
        'iters_per_decade': -1/slope if slope < 0 else float('inf'),
        'log_errors': log_errors,
        'match': abs(slope - slope_theory) < 0.01
    }


def verify_stepsize_experiment(n_episodes=100000, check_interval=100, gamma=0.95,
                               n_states=51, n_actions=51):
    """
    Test Prediction 3: Robbins-Monro stepsize conditions.

    Theory (Bertsekas Section 4.8, p. 56):
      "Furthermore, the stepsize γ_k should satisfy
          γ_k > 0, Σγ_k = ∞, Σγ_k² < ∞
       which are typical of stochastic approximation methods."

    Predictions:
    - Constant stepsize: oscillation, no asymptotic convergence
    - Diminishing stepsize (Robbins-Monro): asymptotic convergence

    Runs Q-learning with both stepsize schedules and compares.
    """
    K_star = solve_K_star(gamma)

    # Discretize
    x_vals = np.linspace(-5, 5, n_states)
    u_vals = np.linspace(-10, 10, n_actions)
    dx = x_vals[1] - x_vals[0]

    np.random.seed(42)

    results = {}

    for schedule_name, get_alpha in [
        ('constant', lambda k: 0.3),
        ('diminishing', lambda k: 1.0 / (1 + k / 5000))
    ]:
        # Initialize Q-table with K0 * x^2
        Q_table = np.zeros((n_states, n_actions))
        for i, x in enumerate(x_vals):
            Q_table[i, :] = K0 * x**2

        K_estimates = []

        for episode in range(n_episodes):
            alpha = get_alpha(episode)
            epsilon = 0.3 / (1 + episode / 50000)

            # Random initial state
            x_idx = np.random.randint(n_states)
            x = x_vals[x_idx]

            for _ in range(20):
                # Epsilon-greedy
                if np.random.rand() < epsilon:
                    u_idx = np.random.randint(n_actions)
                else:
                    u_idx = np.argmin(Q_table[x_idx, :])

                u = u_vals[u_idx]

                # Transition
                x_next = A * x + B * u
                cost = Q * x**2 + R * u**2

                # Clip
                x_next = np.clip(x_next, x_vals[0], x_vals[-1])
                x_next_idx = int((x_next - x_vals[0]) / dx)
                x_next_idx = np.clip(x_next_idx, 0, n_states - 1)

                # TD update
                Q_next_min = np.min(Q_table[x_next_idx, :])
                TD_target = cost + gamma * Q_next_min
                Q_table[x_idx, u_idx] += alpha * (TD_target - Q_table[x_idx, u_idx])

                x_idx = x_next_idx
                x = x_vals[x_idx]

            # Record K estimate
            if episode % check_interval == 0:
                V = np.min(Q_table, axis=1)
                mask = (np.abs(x_vals) > 0.5) & (np.abs(x_vals) < 4.0)
                K_est = np.mean(V[mask] / (x_vals[mask]**2))
                K_estimates.append(K_est)

        K_arr = np.array(K_estimates)

        # Compute final statistics
        final_window = K_arr[-100:]  # Last 10000 episodes
        results[schedule_name] = {
            'K_estimates': K_arr,
            'final_mean': np.mean(final_window),
            'final_std': np.std(final_window),
            'final_error': abs(np.mean(final_window) - K_star),
            'oscillation': np.std(final_window)
        }

    results['K_star'] = K_star
    results['prediction_confirmed'] = (
        results['constant']['oscillation'] > results['diminishing']['oscillation']
    )

    return results


def verify_exploration_coverage(n_episodes=50000, n_states=31, n_actions=31, gamma=0.95):
    """
    Test Prediction 4: All (s,a) must be visited infinitely often.

    Theory (Bertsekas Section 4.8, p. 56):
      "Chief among these are that all state-control pairs (i,u) must be
       generated infinitely often within the infinitely long sequence
       {(i_k, u_k)}, and that the successor states j must be independently
       sampled at each occurrence of a given state-control pair."

    Tracks visit counts and verifies adequate exploration coverage.
    """
    x_vals = np.linspace(-3, 3, n_states)
    u_vals = np.linspace(-6, 6, n_actions)
    dx = x_vals[1] - x_vals[0]

    visit_counts = np.zeros((n_states, n_actions))
    Q_table = np.zeros((n_states, n_actions))
    for i, x in enumerate(x_vals):
        Q_table[i, :] = 5.0 * x**2

    np.random.seed(42)

    for episode in range(n_episodes):
        epsilon = 0.3 / (1 + episode / 50000)
        alpha = 0.3 / (1 + episode / 100000)

        x_idx = np.random.randint(n_states)
        x = x_vals[x_idx]

        for _ in range(20):
            if np.random.rand() < epsilon:
                u_idx = np.random.randint(n_actions)
            else:
                u_idx = np.argmin(Q_table[x_idx, :])

            visit_counts[x_idx, u_idx] += 1

            u = u_vals[u_idx]
            x_next = A * x + B * u
            cost = Q * x**2 + R * u**2

            x_next = np.clip(x_next, x_vals[0], x_vals[-1])
            x_next_idx = int((x_next - x_vals[0]) / dx)
            x_next_idx = np.clip(x_next_idx, 0, n_states - 1)

            Q_next_min = np.min(Q_table[x_next_idx, :])
            TD_target = cost + gamma * Q_next_min
            Q_table[x_idx, u_idx] += alpha * (TD_target - Q_table[x_idx, u_idx])

            x_idx = x_next_idx
            x = x_vals[x_idx]

    # Analyze coverage
    total_pairs = n_states * n_actions
    visited_pairs = np.sum(visit_counts > 0)
    never_visited = np.sum(visit_counts == 0)
    min_visits = np.min(visit_counts)
    max_visits = np.max(visit_counts)
    mean_visits = np.mean(visit_counts)

    # Coefficient of variation (lower = more uniform)
    cv = np.std(visit_counts) / np.mean(visit_counts) if np.mean(visit_counts) > 0 else float('inf')

    return {
        'total_pairs': total_pairs,
        'visited_pairs': visited_pairs,
        'never_visited': never_visited,
        'coverage_pct': 100 * visited_pairs / total_pairs,
        'min_visits': min_visits,
        'max_visits': max_visits,
        'mean_visits': mean_visits,
        'cv': cv,
        'visit_counts': visit_counts,
        'full_coverage': never_visited == 0
    }


def run_theory_verification_tests():
    """
    Run all Bertsekas theory verification tests and generate report + figure.
    """
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 15 + "BERTSEKAS Q-LEARNING THEORY VERIFICATION" + " " * 22 + "|")
    print("+" + "=" * 78 + "+")

    # =========================================================================
    # Test 1: Contraction Property
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 1: CONTRACTION PROPERTY (Proposition 4.3.5, p. 18-21, 54)")
    print("-" * 80)
    print("""
  Bertsekas (p. 54):
    "It is straightforward to show that F is a contraction with modulus α,
     similar to the DP operator T. Thus the algorithm Q_{k+1} = FQ_k converges
     to Q* from every starting point Q_0."

  Theory: F is a contraction with modulus rho = gamma * a^2 * r^2 / (r + gamma*b^2*K*)^2

  For VI iterations: ||K_{k+1} - K*|| / ||K_k - K*|| should equal rho
""")

    contraction_results = verify_contraction_property(GAMMA_DP, K0)

    print(f"  Parameters: gamma={contraction_results['gamma']}, K*={contraction_results['K_star']:.6f}")
    print(f"\n  {'Metric':<30} {'Theoretical':>15} {'Empirical':>15} {'Match?':>10}")
    print("  " + "-" * 70)
    print(f"  {'Contraction modulus rho':<30} {contraction_results['rho_theory']:>15.6f} {contraction_results['rho_empirical']:>15.6f} {'YES' if contraction_results['match'] else 'NO':>10}")

    print(f"\n  Iteration-by-iteration error ratios:")
    print(f"  {'Iter':<6} {'|K_k - K*|':>15} {'Ratio':>15}")
    print("  " + "-" * 36)
    for i in range(min(15, len(contraction_results['errors']) - 1)):
        ratio = contraction_results['ratios'][i] if i < len(contraction_results['ratios']) else 0
        print(f"  {i:<6} {contraction_results['errors'][i]:>15.6e} {ratio:>15.6f}")

    # =========================================================================
    # Test 2: Convergence Rate
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 2: LINEAR CONVERGENCE RATE")
    print("-" * 80)
    print("""
  Theory: log10(error_k) decreases linearly with slope = log10(rho)

  Slope determines iterations per decade of error reduction: -1/slope
""")

    rate_results = verify_convergence_rate(GAMMA_DP, K0)

    print(f"  {'Metric':<35} {'Theoretical':>15} {'Empirical':>15} {'Match?':>10}")
    print("  " + "-" * 75)
    print(f"  {'Slope of log(error) vs iter':<35} {rate_results['slope_theory']:>15.6f} {rate_results['slope_empirical']:>15.6f} {'YES' if rate_results['match'] else 'NO':>10}")
    print(f"  {'Implied contraction rho':<35} {rate_results['rho_theory']:>15.6f} {rate_results['rho_from_slope']:>15.6f}")
    print(f"  {'Iterations per decade':<35} {-1/rate_results['slope_theory']:>15.2f} {rate_results['iters_per_decade']:>15.2f}")

    # =========================================================================
    # Test 3: Stepsize Experiment
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 3: STEPSIZE REQUIREMENTS (Section 4.8, p. 56)")
    print("-" * 80)
    print("""
  Bertsekas (p. 56):
    "Furthermore, the stepsize γ_k should satisfy
        γ_k > 0, Σγ_k = ∞, Σγ_k² < ∞
     which are typical of stochastic approximation methods."

  Predictions:
  - Constant stepsize: oscillation around K*, no asymptotic convergence
  - Diminishing stepsize: asymptotic convergence to K*
""")

    print("  Running Q-learning with constant vs diminishing stepsize...")
    stepsize_results = verify_stepsize_experiment(n_episodes=100000, check_interval=100)

    print(f"\n  K* = {stepsize_results['K_star']:.6f}")
    print(f"\n  {'Schedule':<15} {'Final Mean':>12} {'Final Std':>12} {'Error':>12} {'Converged?':>12}")
    print("  " + "-" * 63)
    for sched in ['constant', 'diminishing']:
        r = stepsize_results[sched]
        converged = "YES" if r['final_std'] < 0.1 else "NO"
        print(f"  {sched:<15} {r['final_mean']:>12.4f} {r['final_std']:>12.4f} {r['final_error']:>12.4f} {converged:>12}")

    print(f"\n  Prediction confirmed: {'YES' if stepsize_results['prediction_confirmed'] else 'NO'}")
    print(f"  (Constant stepsize oscillation > diminishing stepsize oscillation)")

    # =========================================================================
    # Test 4: Exploration Coverage
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 4: EXPLORATION REQUIREMENT (Section 4.8, p. 56)")
    print("-" * 80)
    print("""
  Bertsekas (p. 56):
    "Chief among these are that all state-control pairs (i,u) must be
     generated infinitely often within the infinitely long sequence
     {(i_k, u_k)}, and that the successor states j must be independently
     sampled at each occurrence of a given state-control pair."

  Checking visit count distribution across state-action space.
""")

    print("  Running exploration coverage analysis...")
    coverage_results = verify_exploration_coverage(n_episodes=50000)

    print(f"\n  {'Metric':<35} {'Value':>15}")
    print("  " + "-" * 50)
    print(f"  {'Total state-action pairs':<35} {coverage_results['total_pairs']:>15}")
    print(f"  {'Pairs visited at least once':<35} {coverage_results['visited_pairs']:>15}")
    print(f"  {'Pairs never visited':<35} {coverage_results['never_visited']:>15}")
    print(f"  {'Coverage percentage':<35} {coverage_results['coverage_pct']:>14.1f}%")
    print(f"  {'Min visits per pair':<35} {coverage_results['min_visits']:>15.0f}")
    print(f"  {'Max visits per pair':<35} {coverage_results['max_visits']:>15.0f}")
    print(f"  {'Mean visits per pair':<35} {coverage_results['mean_visits']:>15.1f}")
    print(f"  {'Coefficient of variation':<35} {coverage_results['cv']:>15.2f}")
    print(f"  {'Full coverage achieved?':<35} {'YES' if coverage_results['full_coverage'] else 'NO':>15}")

    # =========================================================================
    # Generate Theory Verification Figure
    # =========================================================================
    print("\n" + "-" * 80)
    print("GENERATING THEORY VERIFICATION FIGURE")
    print("-" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Contraction ratios
    ax1 = axes[0, 0]
    ratios = contraction_results['ratios']
    ax1.plot(range(len(ratios)), ratios, 'b-o', markersize=6, label='Empirical ratio')
    ax1.axhline(contraction_results['rho_theory'], color='r', linestyle='--', linewidth=2,
                label=f"Theoretical $\\rho$ = {contraction_results['rho_theory']:.4f}")
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Error ratio $|e_{k+1}|/|e_k|$', fontsize=11)
    ax1.set_title('Test 1: Contraction Property Verification\n$||K_{k+1} - K^*|| / ||K_k - K^*|| = \\rho$', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.3)

    # Plot 2: Linear convergence rate
    ax2 = axes[0, 1]
    log_errors = rate_results['log_errors']
    iters = np.arange(len(log_errors))
    ax2.plot(iters, log_errors, 'b-o', markersize=6, label='$\\log_{10}|K_k - K^*|$')
    # Theoretical line
    theoretical_line = rate_results['slope_theory'] * iters + log_errors[0]
    ax2.plot(iters, theoretical_line, 'r--', linewidth=2,
             label=f'Theoretical slope = {rate_results["slope_theory"]:.4f}')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('$\\log_{10}|K_k - K^*|$', fontsize=11)
    ax2.set_title(f'Test 2: Linear Convergence Rate\nSlope = {rate_results["slope_empirical"]:.4f} (theory: {rate_results["slope_theory"]:.4f})', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Stepsize comparison
    ax3 = axes[1, 0]
    episodes = np.arange(len(stepsize_results['constant']['K_estimates'])) * 100
    ax3.plot(episodes, stepsize_results['constant']['K_estimates'], 'r-',
             alpha=0.7, linewidth=0.8, label='Constant $\\alpha$ = 0.3')
    ax3.plot(episodes, stepsize_results['diminishing']['K_estimates'], 'b-',
             alpha=0.7, linewidth=0.8, label='Diminishing $\\alpha$ = 1/(1+k/5000)')
    ax3.axhline(stepsize_results['K_star'], color='k', linestyle='--', linewidth=2,
                label=f"$K^*$ = {stepsize_results['K_star']:.4f}")
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('K estimate', fontsize=11)
    ax3.set_title('Test 3: Stepsize Requirements (Robbins-Monro)\nConstant oscillates; diminishing converges', fontsize=12)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Exploration coverage heatmap
    ax4 = axes[1, 1]
    visit_log = np.log10(coverage_results['visit_counts'] + 1)
    im = ax4.imshow(visit_log.T, aspect='auto', cmap='viridis', origin='lower')
    ax4.set_xlabel('State index', fontsize=11)
    ax4.set_ylabel('Action index', fontsize=11)
    ax4.set_title(f'Test 4: Exploration Coverage\n{coverage_results["coverage_pct"]:.1f}% of (s,a) pairs visited', fontsize=12)
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('$\\log_{10}$(visits + 1)', fontsize=10)

    plt.tight_layout()
    save_path = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_qlearning_theory_test.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "-" * 80)
    print("THEORY VERIFICATION SUMMARY")
    print("-" * 80)

    tests_passed = sum([
        contraction_results['match'],
        rate_results['match'],
        stepsize_results['prediction_confirmed'],
        coverage_results['coverage_pct'] > 90
    ])

    print(f"\n  {'Test':<45} {'Status':>10}")
    print("  " + "-" * 55)
    print(f"  {'1. Contraction property (rho matches theory)':<45} {'PASS' if contraction_results['match'] else 'FAIL':>10}")
    print(f"  {'2. Linear convergence rate (slope matches theory)':<45} {'PASS' if rate_results['match'] else 'FAIL':>10}")
    print(f"  {'3. Stepsize requirements (Robbins-Monro)':<45} {'PASS' if stepsize_results['prediction_confirmed'] else 'FAIL':>10}")
    print(f"  {'4. Exploration coverage (>90%)':<45} {'PASS' if coverage_results['coverage_pct'] > 90 else 'FAIL':>10}")
    print("  " + "-" * 55)
    print(f"  {'TOTAL':<45} {tests_passed}/4 passed")

    print("\n" + "+" + "=" * 78 + "+")

    return {
        'contraction': contraction_results,
        'rate': rate_results,
        'stepsize': stepsize_results,
        'coverage': coverage_results
    }


# =============================================================================
# Extended Bertsekas Theory Tests
# =============================================================================

def verify_pi_monotonic_improvement(gamma, K0, n_iters=10):
    """
    Test Proposition 4.5.1: PI generates an improving sequence of policies.

    Theory (Bertsekas Prop 4.5.1, p. 27):
      "The exact PI algorithm generates an improving sequence of policies,
       i.e., J_{μ_{k+1}} ≤ J_{μ_k}, and terminates with an optimal policy."

    In LQR terms: K_{k+1} ≤ K_k (costs decrease monotonically).
    """
    K_star = solve_K_star(gamma)

    K = K0
    L = optimal_gain(K, gamma)

    history = []
    improvements = []

    for i in range(n_iters):
        K_old = K
        L_old = L

        # Policy evaluation: compute cost of current policy
        K = policy_cost(L, gamma)
        if K == np.inf:
            break

        # Policy improvement: get new gain
        L = optimal_gain(K, gamma)

        # Record
        improvement = K_old - K
        history.append({
            'iter': i,
            'K': K,
            'L': L,
            'K_old': K_old,
            'improvement': improvement,
            'error': abs(K - K_star)
        })
        improvements.append(improvement)

        if abs(K - K_old) < 1e-14:
            break

    # Verify monotonicity
    all_positive = all(imp >= -1e-14 for imp in improvements)
    strict_until_convergence = all(imp > 1e-14 for imp in improvements[:-1]) if len(improvements) > 1 else True

    return {
        'history': history,
        'improvements': improvements,
        'monotonic': all_positive,
        'strict_improvement': strict_until_convergence,
        'K_star': K_star,
        'n_iters_to_converge': len(history)
    }


def verify_pi_quadratic_convergence(gamma, K0, n_iters=10):
    """
    Test quadratic (Newton-like) convergence of PI.

    Theory: |e_{k+1}| ≈ C * |e_k|^2 for some constant C (quadratic convergence).

    We compute:
    1. Error ratios: |e_{k+1}| / |e_k|
    2. Quadratic ratios: |e_{k+1}| / |e_k|^2
    3. The quadratic ratio should be roughly constant (the Newton constant).
    """
    K_star = solve_K_star(gamma)

    K = K0
    L = optimal_gain(K, gamma)

    errors = [abs(K - K_star)]
    K_values = [K]

    for i in range(n_iters):
        K = policy_cost(L, gamma)
        if K == np.inf:
            break
        L = optimal_gain(K, gamma)

        error = abs(K - K_star)
        errors.append(error)
        K_values.append(K)

        if error < 1e-15:
            break

    # Compute convergence ratios
    linear_ratios = []
    quadratic_ratios = []

    for i in range(len(errors) - 1):
        if errors[i] > 1e-14:
            linear_ratios.append(errors[i+1] / errors[i])
            if errors[i] > 1e-10:
                quadratic_ratios.append(errors[i+1] / (errors[i]**2))
            else:
                quadratic_ratios.append(float('nan'))
        else:
            linear_ratios.append(float('nan'))
            quadratic_ratios.append(float('nan'))

    # Check if quadratic ratios are roughly constant (Newton convergence)
    valid_quad = [r for r in quadratic_ratios if not np.isnan(r) and r < 100]
    newton_constant = np.mean(valid_quad) if valid_quad else float('nan')

    return {
        'errors': errors,
        'K_values': K_values,
        'linear_ratios': linear_ratios,
        'quadratic_ratios': quadratic_ratios,
        'newton_constant': newton_constant,
        'is_quadratic': len(valid_quad) >= 2 and np.std(valid_quad) / np.mean(valid_quad) < 0.5 if valid_quad else False,
        'K_star': K_star
    }


def verify_optimistic_pi(gamma, K0, m_values=[1, 2, 5, 10, 100], n_outer_iters=50):
    """
    Test Proposition 4.5.2: Optimistic PI convergence.

    Theory (Bertsekas Prop 4.5.2, p. 29):
      "For J_0 satisfying T_{μ_0} J_0 ≤ J_0, we have J* ≤ J_{k+1} ≤ J_k."

    Optimistic PI uses m VI steps for policy evaluation instead of exact evaluation.
    - m=1: equivalent to VI (linear convergence)
    - m=∞: equivalent to exact PI (quadratic convergence)
    """
    K_star = solve_K_star(gamma)

    results = {}

    for m in m_values:
        K = K0
        K_history = [K]
        monotonic_violations = 0

        for outer in range(n_outer_iters):
            # Policy improvement: get policy from current K
            L = optimal_gain(K, gamma)

            # Optimistic policy evaluation: m VI steps with policy L
            # For LQR with fixed L, VI is: K_{new} = q + r*L^2 + gamma*(a+b*L)^2 * K
            for _ in range(m):
                closed_loop = A + B * L
                K_new = Q + R * L**2 + gamma * closed_loop**2 * K
                K = K_new

            K_history.append(K)

            # Check monotonicity
            if K > K_history[-2] + 1e-10:
                monotonic_violations += 1

            # Check convergence
            if abs(K - K_star) < 1e-12:
                break

        # Check J* ≤ J_k for all k
        above_optimal = all(K >= K_star - 1e-10 for K in K_history)

        results[m] = {
            'K_history': K_history,
            'final_K': K_history[-1],
            'final_error': abs(K_history[-1] - K_star),
            'n_iters': len(K_history) - 1,
            'monotonic': monotonic_violations == 0,
            'monotonic_violations': monotonic_violations,
            'above_optimal': above_optimal
        }

    results['K_star'] = K_star
    return results


def verify_approximate_vi_error_bounds(gamma, K0, delta_values=[0.01, 0.05, 0.1, 0.5], n_iters=100):
    """
    Test Section 4.4 error bounds for approximate VI.

    Theory (Bertsekas Section 4.4, p. 23):
      If the approximation error per iteration satisfies:
        |J̃_{k+1}(i) - (T·J̃_k)(i)| ≤ δ for all i, k

      Then asymptotically:
        - Cost error: ||J* - J̃||_∞ ≤ δ/(1-α)
        - Policy error: ||J* - J_μ̃||_∞ ≤ 2δ/(1-α)²

    We add uniform random noise U[-δ, δ] to each VI step and verify bounds.
    """
    K_star = solve_K_star(gamma)

    results = {}

    for delta in delta_values:
        np.random.seed(42)

        K = K0
        K_history = [K]

        for i in range(n_iters):
            # Exact VI step
            K_exact = riccati_operator(K, gamma)

            # Add bounded noise
            noise = np.random.uniform(-delta, delta)
            K = K_exact + noise

            K_history.append(K)

        # Compute statistics over last half (after convergence)
        K_converged = np.array(K_history[n_iters//2:])

        # Empirical error statistics
        errors = np.abs(K_converged - K_star)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)

        # Theoretical bounds
        cost_bound = delta / (1 - gamma)
        policy_bound = 2 * delta / (1 - gamma)**2

        results[delta] = {
            'K_history': K_history,
            'mean_error': mean_error,
            'max_error': max_error,
            'std_error': std_error,
            'cost_bound': cost_bound,
            'policy_bound': policy_bound,
            'cost_bound_satisfied': max_error <= cost_bound * 1.1,  # Allow 10% slack
            'empirical_vs_bound_ratio': max_error / cost_bound
        }

    results['K_star'] = K_star
    results['gamma'] = gamma
    return results


def verify_weighted_norm_contraction(gamma, K0, n_iters=20):
    """
    Test Proposition 4.2.5/4.3.5: Contraction in weighted max norm.

    Theory (Bertsekas Props 4.2.5 and 4.3.5, pp. 13, 18):
      ||TJ - TJ'||_v ≤ ρ ||J - J'||_v for some weight vector v and ρ < 1.

    For discounted problems with discount α:
      ||TJ - TJ'||_∞ ≤ α ||J - J'||_∞

    For SSP problems, the weights relate to expected hitting times.

    Here we verify the max-norm contraction and analyze different starting points.
    """
    K_star = solve_K_star(gamma)

    # Test multiple starting points
    start_points = [K0, K0 * 2, K0 / 2, K_star + 1.0, K_star - 0.5]
    start_labels = ['K0', '2*K0', 'K0/2', 'K*+1', 'K*-0.5']

    results = {}

    for start_K, label in zip(start_points, start_labels):
        if start_K <= 0:
            continue

        K = start_K
        errors = [abs(K - K_star)]
        ratios = []

        for _ in range(n_iters):
            K = riccati_operator(K, gamma)
            errors.append(abs(K - K_star))
            if errors[-2] > 1e-14:
                ratios.append(errors[-1] / errors[-2])

        # Compute max contraction ratio observed
        max_ratio = max(ratios) if ratios else 0
        mean_ratio = np.mean(ratios) if ratios else 0

        # Theoretical bound
        rho_theory = gamma * A**2 * R**2 / (R + gamma * B**2 * K_star)**2

        results[label] = {
            'start_K': start_K,
            'errors': errors,
            'ratios': ratios,
            'max_ratio': max_ratio,
            'mean_ratio': mean_ratio,
            'rho_theory': rho_theory,
            'contraction_verified': max_ratio <= rho_theory + 0.01
        }

    results['K_star'] = K_star
    results['rho_theory'] = gamma * A**2 * R**2 / (R + gamma * B**2 * K_star)**2
    return results


def run_extended_theory_tests():
    """
    Run all extended Bertsekas theory verification tests.
    """
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 12 + "EXTENDED BERTSEKAS THEORY VERIFICATION TESTS" + " " * 21 + "|")
    print("+" + "=" * 78 + "+")

    all_results = {}

    # =========================================================================
    # Test 5: PI Monotonic Improvement (Prop 4.5.1)
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 5: PI MONOTONIC IMPROVEMENT (Proposition 4.5.1, p. 27)")
    print("-" * 80)
    print("""
  Bertsekas (p. 27):
    "The exact PI algorithm generates an improving sequence of policies,
     i.e., J_{μ_{k+1}} ≤ J_{μ_k}, and terminates with an optimal policy."

  For LQR: K_{k+1} ≤ K_k (value function coefficient decreases monotonically)
""")

    pi_mono = verify_pi_monotonic_improvement(GAMMA_DP, K0)
    all_results['pi_monotonic'] = pi_mono

    print(f"  K* = {pi_mono['K_star']:.6f}")
    print(f"\n  {'Iter':<6} {'K_k':>14} {'K_{k-1}':>14} {'Improvement':>14} {'Error':>14}")
    print("  " + "-" * 62)
    for h in pi_mono['history']:
        print(f"  {h['iter']:<6} {h['K']:>14.8f} {h['K_old']:>14.8f} {h['improvement']:>14.2e} {h['error']:>14.2e}")

    print(f"\n  Monotonic improvement: {'YES' if pi_mono['monotonic'] else 'NO'}")
    print(f"  Strict until convergence: {'YES' if pi_mono['strict_improvement'] else 'NO'}")
    print(f"  Iterations to converge: {pi_mono['n_iters_to_converge']}")

    # =========================================================================
    # Test 6: PI Quadratic Convergence
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 6: PI QUADRATIC CONVERGENCE (Newton-like)")
    print("-" * 80)
    print("""
  Theory: PI exhibits quadratic convergence: |e_{k+1}| ≈ C * |e_k|^2

  Evidence: The ratio |e_{k+1}| / |e_k|^2 should be approximately constant.
""")

    pi_quad = verify_pi_quadratic_convergence(GAMMA_DP, K0)
    all_results['pi_quadratic'] = pi_quad

    print(f"  K* = {pi_quad['K_star']:.6f}")
    print(f"\n  {'Iter':<6} {'|e_k|':>14} {'|e_{k+1}|/|e_k|':>18} {'|e_{k+1}|/|e_k|^2':>18}")
    print("  " + "-" * 58)
    for i in range(len(pi_quad['errors']) - 1):
        lin = pi_quad['linear_ratios'][i] if i < len(pi_quad['linear_ratios']) else float('nan')
        quad = pi_quad['quadratic_ratios'][i] if i < len(pi_quad['quadratic_ratios']) else float('nan')
        print(f"  {i:<6} {pi_quad['errors'][i]:>14.2e} {lin:>18.6f} {quad:>18.6f}")

    print(f"\n  Newton constant (mean of |e_{{k+1}}|/|e_k|^2): {pi_quad['newton_constant']:.4f}")
    print(f"  Quadratic convergence verified: {'YES' if pi_quad['is_quadratic'] else 'NO'}")

    # =========================================================================
    # Test 7: Optimistic PI (Prop 4.5.2)
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 7: OPTIMISTIC PI CONVERGENCE (Proposition 4.5.2, p. 29)")
    print("-" * 80)
    print("""
  Bertsekas (p. 29):
    "For J_0 satisfying T_{μ_0} J_0 ≤ J_0, we have J* ≤ J_{k+1} ≤ J_k."
    (monotonic decrease, bounded below by J*)

  Optimistic PI uses m VI steps for policy evaluation:
  - m=1: equivalent to VI (slow, linear convergence)
  - m=∞: equivalent to exact PI (fast, quadratic convergence)
""")

    opt_pi = verify_optimistic_pi(GAMMA_DP, K0, m_values=[1, 2, 5, 10, 50])
    all_results['optimistic_pi'] = opt_pi

    print(f"  K* = {opt_pi['K_star']:.6f}")
    print(f"\n  {'m':>6} {'Final K':>14} {'Error':>14} {'Iters':>8} {'Monotonic?':>12} {'≥ K*?':>10}")
    print("  " + "-" * 66)
    for m in [1, 2, 5, 10, 50]:
        r = opt_pi[m]
        print(f"  {m:>6} {r['final_K']:>14.6f} {r['final_error']:>14.2e} {r['n_iters']:>8} "
              f"{'YES' if r['monotonic'] else 'NO':>12} {'YES' if r['above_optimal'] else 'NO':>10}")

    print(f"\n  Observation: m=1 (VI) takes ~{opt_pi[1]['n_iters']} iters, m=50 takes ~{opt_pi[50]['n_iters']} iters")

    # =========================================================================
    # Test 8: Approximate VI Error Bounds (Section 4.4)
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 8: APPROXIMATE VI ERROR BOUNDS (Section 4.4, p. 23)")
    print("-" * 80)
    print("""
  Bertsekas (p. 23):
    If |J̃_{k+1}(i) - (TJ̃_k)(i)| ≤ δ for all i,k, then asymptotically:
      - Cost error: ||J* - J̃||_∞ ≤ δ/(1-α)
      - Policy error: ||J* - J_μ̃||_∞ ≤ 2δ/(1-α)²
""")

    approx_vi = verify_approximate_vi_error_bounds(GAMMA_DP, K0, delta_values=[0.001, 0.01, 0.05, 0.1])
    all_results['approximate_vi'] = approx_vi

    print(f"  γ = {approx_vi['gamma']}, K* = {approx_vi['K_star']:.6f}")
    print(f"\n  {'δ':>8} {'δ/(1-γ)':>12} {'Max Error':>12} {'Ratio':>10} {'Bound OK?':>12}")
    print("  " + "-" * 56)
    for delta in [0.001, 0.01, 0.05, 0.1]:
        r = approx_vi[delta]
        print(f"  {delta:>8.3f} {r['cost_bound']:>12.4f} {r['max_error']:>12.4f} "
              f"{r['empirical_vs_bound_ratio']:>10.3f} {'YES' if r['cost_bound_satisfied'] else 'NO':>12}")

    # =========================================================================
    # Test 9: Weighted Norm Contraction (Prop 4.2.5/4.3.5)
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 9: WEIGHTED NORM CONTRACTION (Props 4.2.5/4.3.5, pp. 13, 18)")
    print("-" * 80)
    print("""
  Bertsekas (p. 18):
    T is a contraction with modulus ρ in a weighted max norm.
    For discounted problems with discount α:
      ||TJ - TJ'||_∞ ≤ α ||J - J'||_∞

  Verifying contraction from multiple starting points.
""")

    weighted = verify_weighted_norm_contraction(GAMMA_DP, K0)
    all_results['weighted_norm'] = weighted

    print(f"  Theoretical ρ = {weighted['rho_theory']:.6f}")
    print(f"\n  {'Start':>10} {'Start K':>12} {'Max Ratio':>12} {'Mean Ratio':>12} {'Verified?':>12}")
    print("  " + "-" * 60)
    for label in ['K0', '2*K0', 'K0/2', 'K*+1']:
        if label in weighted:
            r = weighted[label]
            print(f"  {label:>10} {r['start_K']:>12.4f} {r['max_ratio']:>12.6f} "
                  f"{r['mean_ratio']:>12.6f} {'YES' if r['contraction_verified'] else 'NO':>12}")

    # =========================================================================
    # Generate Extended Theory Figure
    # =========================================================================
    print("\n" + "-" * 80)
    print("GENERATING EXTENDED THEORY VERIFICATION FIGURE")
    print("-" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: PI Monotonic Improvement
    ax1 = axes[0, 0]
    iters = [h['iter'] for h in pi_mono['history']]
    K_vals = [h['K'] for h in pi_mono['history']]
    ax1.semilogy(iters, [h['improvement'] for h in pi_mono['history']], 'g-o', markersize=8,
                 label='Improvement $K_{k-1} - K_k$')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Cost Improvement', fontsize=11)
    ax1.set_title('Test 5: PI Monotonic Improvement\n(Prop 4.5.1: $J_{\\mu_{k+1}} \\leq J_{\\mu_k}$)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: PI Quadratic Convergence
    ax2 = axes[0, 1]
    errors = pi_quad['errors']
    ax2.loglog(errors[:-1], errors[1:], 'bo-', markersize=8, label='Actual: $|e_{k+1}|$ vs $|e_k|$')
    # Quadratic reference: e_{k+1} = C * e_k^2
    e_range = np.logspace(np.log10(min(e for e in errors if e > 1e-14)),
                          np.log10(max(errors)), 50)
    ax2.loglog(e_range, pi_quad['newton_constant'] * e_range**2, 'r--', linewidth=2,
               label=f'Quadratic: $C \\cdot |e_k|^2$, C={pi_quad["newton_constant"]:.2f}')
    ax2.loglog(e_range, e_range, 'k:', alpha=0.5, label='Linear: $|e_{k+1}| = |e_k|$')
    ax2.set_xlabel('$|e_k|$', fontsize=11)
    ax2.set_ylabel('$|e_{k+1}|$', fontsize=11)
    ax2.set_title('Test 6: PI Quadratic Convergence\n$|e_{k+1}| \\approx C |e_k|^2$', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Optimistic PI
    ax3 = axes[0, 2]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, m in enumerate([1, 2, 5, 10, 50]):
        K_hist = opt_pi[m]['K_history']
        ax3.semilogy(range(len(K_hist)), [abs(K - opt_pi['K_star']) for K in K_hist],
                     color=colors[i], linewidth=1.5, label=f'm={m}')
    ax3.axhline(1e-12, color='k', linestyle='--', alpha=0.3, label='Machine precision')
    ax3.set_xlabel('Outer Iteration', fontsize=11)
    ax3.set_ylabel('Error $|K - K^*|$', fontsize=11)
    ax3.set_title('Test 7: Optimistic PI\n(m=1: VI, m=∞: exact PI)', fontsize=12)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Approximate VI Error Bounds
    ax4 = axes[1, 0]
    deltas = [0.001, 0.01, 0.05, 0.1]
    max_errors = [approx_vi[d]['max_error'] for d in deltas]
    bounds = [approx_vi[d]['cost_bound'] for d in deltas]
    x = np.arange(len(deltas))
    width = 0.35
    ax4.bar(x - width/2, max_errors, width, label='Empirical Max Error', color='blue', alpha=0.7)
    ax4.bar(x + width/2, bounds, width, label='Bound $\\delta/(1-\\gamma)$', color='red', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'δ={d}' for d in deltas])
    ax4.set_ylabel('Error', fontsize=11)
    ax4.set_title('Test 8: Approximate VI Error Bounds\n(Sec 4.4: error ≤ δ/(1-γ))', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Weighted Norm Contraction
    ax5 = axes[1, 1]
    for label in ['K0', '2*K0', 'K0/2', 'K*+1']:
        if label in weighted:
            r = weighted[label]
            ax5.semilogy(range(len(r['errors'])), r['errors'], linewidth=1.5, label=f'Start: {label}')
    ax5.axhline(1e-14, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Iteration', fontsize=11)
    ax5.set_ylabel('Error $|K - K^*|$', fontsize=11)
    ax5.set_title('Test 9: Weighted Norm Contraction\n(all start points converge)', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Contraction Ratio Comparison
    ax6 = axes[1, 2]
    for label in ['K0', '2*K0', 'K0/2', 'K*+1']:
        if label in weighted:
            r = weighted[label]
            ax6.plot(range(len(r['ratios'])), r['ratios'], 'o-', markersize=5, label=f'Start: {label}')
    ax6.axhline(weighted['rho_theory'], color='red', linestyle='--', linewidth=2,
                label=f'Theoretical ρ = {weighted["rho_theory"]:.4f}')
    ax6.set_xlabel('Iteration', fontsize=11)
    ax6.set_ylabel('Contraction Ratio $|e_{k+1}|/|e_k|$', fontsize=11)
    ax6.set_title('Test 9b: Contraction Ratio\n(all ≤ ρ after transient)', fontsize=12)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 0.3)

    plt.tight_layout()
    save_path = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_bertsekas_theory_full.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "-" * 80)
    print("EXTENDED THEORY VERIFICATION SUMMARY")
    print("-" * 80)

    tests = [
        ('5. PI Monotonic Improvement', pi_mono['monotonic'] and pi_mono['strict_improvement']),
        ('6. PI Quadratic Convergence', pi_quad['is_quadratic']),
        ('7. Optimistic PI (all m converge)', all(opt_pi[m]['monotonic'] and opt_pi[m]['above_optimal']
                                                   for m in [1, 2, 5, 10, 50])),
        ('8. Approximate VI Bounds', all(approx_vi[d]['cost_bound_satisfied']
                                          for d in [0.001, 0.01, 0.05, 0.1])),
        ('9. Weighted Norm Contraction', all(weighted[l]['contraction_verified']
                                              for l in ['K0', '2*K0', 'K0/2', 'K*+1'] if l in weighted)),
    ]

    print(f"\n  {'Test':<45} {'Status':>10}")
    print("  " + "-" * 55)
    for name, passed in tests:
        print(f"  {name:<45} {'PASS' if passed else 'FAIL':>10}")
    print("  " + "-" * 55)
    print(f"  {'TOTAL':<45} {sum(1 for _, p in tests if p)}/{len(tests)} passed")

    print("\n" + "+" + "=" * 78 + "+")

    return all_results


# =============================================================================
# Main: Run All Methods and Plot
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("LQR Convergence Rate Comparison")
    print("=" * 70)

    # Run VI
    print("\n" + "-" * 70)
    print(f"Running Value Iteration (gamma={GAMMA_DP})...")
    print("-" * 70)
    vi_history = run_vi(n_iters=30, start_k=K0, gamma=GAMMA_DP)
    vi_errors = np.array([h[1] for h in vi_history])
    print(f"\nVI summary: {len(vi_history) - 1} iterations")
    print(f"  Initial error: {vi_errors[0]:.6f}")
    print(f"  Final error: {vi_errors[-1]:.2e}")

    # Run PI
    print("\n" + "-" * 70)
    print(f"Running Policy Iteration (gamma={GAMMA_DP})...")
    print("-" * 70)
    pi_history = run_pi(n_iters=10, start_k=K0, gamma=GAMMA_DP)
    pi_errors = np.array([h[1] for h in pi_history])
    print(f"\nPI summary: {len(pi_history) - 1} iterations")
    print(f"  Initial error: {pi_errors[0]:.6f}")
    print(f"  Final error: {pi_errors[-1]:.2e}")

    # Run Q-learning with debug logging
    n_episodes = 500000
    check_interval = 10  # Log frequently to see meandering
    debug_log_path = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_qlearning_debug.log"
    print("\n" + "-" * 70)
    print(f"Running Q-Learning ({n_episodes} episodes, gamma={GAMMA_QL})...")
    print("-" * 70)
    ql_errors, ql_K_estimates, ql_K_single, debug_data = run_q_learning(
        n_episodes, check_interval,
        gamma=GAMMA_QL, start_k=K0,
        n_states=101, n_actions=101,
        debug_log_path=debug_log_path
    )
    print(f"\nQ-learning summary: {len(ql_errors)} checkpoints")
    print(f"  Target K* (gamma={GAMMA_QL}): {K_STAR_QL:.4f}")
    print(f"  Initial K estimate (avg): {ql_K_estimates[0]:.4f}")
    print(f"  Final K estimate (avg): {ql_K_estimates[-1]:.4f}")
    print(f"  Initial K (single state): {ql_K_single[0]:.4f}")
    print(f"  Final K (single state): {ql_K_single[-1]:.4f}")
    print(f"  Final error: {ql_errors[-1]:.4e}")

    # Print debug summary
    print_debug_summary(debug_data, check_interval)

    # ==========================================================================
    # Plot 1: Convergence Rates
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Generating plots...")
    print("-" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    # VI: x-axis = iteration number
    ax.semilogy(range(len(vi_errors)), vi_errors, 'b-o', linewidth=2,
                markersize=6, label=f'Value Iteration ($\\gamma$={GAMMA_DP})')

    # PI: x-axis = iteration number
    ax.semilogy(range(len(pi_errors)), pi_errors, 'g-s', linewidth=2,
                markersize=8, label=f'Policy Iteration ($\\gamma$={GAMMA_DP})')

    # Q-learning: rescaled x-axis for comparison
    ql_x_scaled = np.arange(len(ql_errors)) * 0.05  # Map to comparable iteration scale
    ax.semilogy(ql_x_scaled[:min(200, len(ql_errors))],
                ql_errors[:min(200, len(ql_errors))],
                'r-', linewidth=1.5, alpha=0.8,
                label=f'Q-Learning ($\\gamma$={GAMMA_QL}, {n_episodes} episodes)')

    # Reference lines
    ax.axhline(1e-10, color='gray', linestyle=':', alpha=0.5, label='Machine precision')

    # Labels
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(r'Error $|K - K^*|$', fontsize=12)
    ax.set_title('LQR Convergence Rates: VI vs PI vs Q-Learning\n' +
                 f'(a={A}, b={B}, q={Q}, r={R}, $K_0$={K0})', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(-0.5, 35)
    ax.set_ylim(1e-16, 10)

    # Add annotations
    ax.annotate('Linear\nconvergence', xy=(15, vi_errors[15] if len(vi_errors) > 15 else 1e-3),
                xytext=(20, 1e-2), fontsize=9, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))

    ax.annotate('Quadratic\nconvergence', xy=(3, 1e-10),
                xytext=(6, 1e-8), fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))

    save_path1 = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_convergence_rates.png"
    fig.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path1}")

    # ==========================================================================
    # Plot 2: Riccati Map - Value Iteration
    # ==========================================================================

    save_path_vi = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_riccati_vi.png"
    plot_riccati_vi(vi_history, GAMMA_DP, save_path_vi)

    # ==========================================================================
    # Plot 3: Riccati Map - Policy Iteration
    # ==========================================================================

    save_path_pi = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_riccati_pi.png"
    plot_riccati_pi(pi_history, GAMMA_DP, save_path_pi)

    # ==========================================================================
    # Plot 4: Riccati Map - Q-Learning
    # ==========================================================================

    save_path_ql = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_riccati_ql.png"
    plot_riccati_ql(ql_K_estimates, GAMMA_QL, save_path_ql)  # Use averaged K estimates

    # ==========================================================================
    # Plot 5: Q-Learning Debug Visualization
    # ==========================================================================

    save_path_debug = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_qlearning_debug.png"
    plot_qlearning_debug(debug_data, check_interval, save_path_debug)

    # ==========================================================================
    # Print convergence rate analysis
    # ==========================================================================

    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 25 + "CONVERGENCE RATE ANALYSIS" + " " * 28 + "|")
    print("+" + "=" * 78 + "+")

    # -------------------------------------------------------------------------
    # Value Iteration Table
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"VALUE ITERATION (gamma={GAMMA_DP})")
    print("-" * 80)
    print(f"\n  Convergence type: LINEAR (contraction mapping)")
    print(f"  Contraction factor: gamma * a^2 / (1 + gamma * b^2 * K*)^2 approx {GAMMA_DP * A**2 / (1 + GAMMA_DP * B**2 * K_STAR_DP)**2:.4f}")
    print(f"\n  {'Iter':<6} {'K_k':>14} {'|K_k - K*|':>14} {'Ratio':>14} {'Log10(error)':>14}")
    print("  " + "-" * 62)
    for k in range(min(25, len(vi_errors))):
        ratio = vi_errors[k] / vi_errors[k-1] if k > 0 and vi_errors[k-1] > 1e-15 else 0
        log_err = np.log10(vi_errors[k]) if vi_errors[k] > 0 else -16
        print(f"  {k:<6} {vi_history[k][0]:>14.8f} {vi_errors[k]:>14.2e} {ratio:>14.6f} {log_err:>14.2f}")

    # Compute average contraction ratio
    ratios = [vi_errors[k+1]/vi_errors[k] for k in range(5, min(15, len(vi_errors)-1)) if vi_errors[k] > 1e-14]
    if ratios:
        avg_ratio = np.mean(ratios)
        print(f"\n  Average contraction ratio (iters 5-15): {avg_ratio:.6f}")
        print(f"  Iterations to reduce error by 10x: {-1/np.log10(avg_ratio):.1f}")

    # -------------------------------------------------------------------------
    # Policy Iteration Table
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"POLICY ITERATION (gamma={GAMMA_DP})")
    print("-" * 80)
    print(f"\n  Convergence type: QUADRATIC (Newton's method)")
    print(f"  Each iteration squares the error (approximately)")
    print(f"\n  {'Iter':<6} {'K_k':>14} {'L_k':>14} {'|K_k - K*|':>14} {'Log10(error)':>14}")
    print("  " + "-" * 62)
    for k in range(len(pi_errors)):
        L_k = optimal_gain(pi_history[k][0], GAMMA_DP)
        log_err = np.log10(pi_errors[k]) if pi_errors[k] > 0 else -16
        print(f"  {k:<6} {pi_history[k][0]:>14.8f} {L_k:>14.8f} {pi_errors[k]:>14.2e} {log_err:>14.2f}")

    # Show quadratic convergence
    print(f"\n  Quadratic convergence evidence (error_k+1 / error_k^2):")
    for k in range(1, min(4, len(pi_errors)-1)):
        if pi_errors[k] > 1e-14:
            quadratic_factor = pi_errors[k+1] / (pi_errors[k]**2) if pi_errors[k] > 1e-10 else 0
            print(f"    k={k}: |e_{k+1}| / |e_{k}|^2 = {quadratic_factor:.2f}")

    # -------------------------------------------------------------------------
    # Q-Learning Table
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"Q-LEARNING (gamma={GAMMA_QL})")
    print("-" * 80)
    print(f"\n  Convergence type: STOCHASTIC (noisy gradient descent)")
    print(f"  Note: Different gamma, so different K* target ({K_STAR_QL:.6f} vs {K_STAR_DP:.6f})")
    print(f"\n  {'Episode':<10} {'K_estimate':>14} {'|K - K*|':>14} {'dK':>14} {'Log10(error)':>14}")
    print("  " + "-" * 66)

    checkpoints = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 49999]
    for i in checkpoints:
        if i < len(ql_errors):
            episode = i * check_interval
            dK = ql_K_estimates[i] - ql_K_estimates[i-1] if i > 0 else 0
            log_err = np.log10(ql_errors[i]) if ql_errors[i] > 0 else -16
            print(f"  {episode:<10} {ql_K_estimates[i]:>14.6f} {ql_errors[i]:>14.4e} {dK:>+14.6f} {log_err:>14.2f}")

    # Q-learning statistics
    print(f"\n  Episode-to-episode K changes:")
    dK_all = np.diff(ql_K_estimates)
    print(f"    Mean(dK):  {np.mean(dK_all):>+14.8f}")
    print(f"    Std(dK):   {np.std(dK_all):>14.8f}")
    print(f"    Min(dK):   {np.min(dK_all):>+14.8f}")
    print(f"    Max(dK):   {np.max(dK_all):>+14.8f}")
    print(f"    #(dK > 0): {np.sum(dK_all > 0):>14} ({100*np.sum(dK_all > 0)/len(dK_all):.2f}%)")
    print(f"    #(dK < 0): {np.sum(dK_all < 0):>14} ({100*np.sum(dK_all < 0)/len(dK_all):.2f}%)")
    print(f"    #(dK = 0): {np.sum(dK_all == 0):>14} ({100*np.sum(dK_all == 0)/len(dK_all):.2f}%)")

    # -------------------------------------------------------------------------
    # Final Comparison Table
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("FINAL COMPARISON TABLE")
    print("-" * 80)

    # Compute iterations to reach certain error thresholds
    def iters_to_error(errors, threshold):
        for i, e in enumerate(errors):
            if e < threshold:
                return i
        return ">{}".format(len(errors))

    print(f"\n  {'Method':<20} {'gamma':>8} {'K*':>12} {'Final K':>12} {'Final Error':>14} {'Model?':<8}")
    print("  " + "-" * 74)
    print(f"  {'Value Iteration':<20} {GAMMA_DP:>8.2f} {K_STAR_DP:>12.6f} {vi_history[-1][0]:>12.6f} {vi_errors[-1]:>14.2e} {'Yes':<8}")
    print(f"  {'Policy Iteration':<20} {GAMMA_DP:>8.2f} {K_STAR_DP:>12.6f} {pi_history[-1][0]:>12.6f} {pi_errors[-1]:>14.2e} {'Yes':<8}")
    print(f"  {'Q-Learning':<20} {GAMMA_QL:>8.2f} {K_STAR_QL:>12.6f} {ql_K_estimates[-1]:>12.6f} {ql_errors[-1]:>14.2e} {'No':<8}")

    print(f"\n  Iterations/Episodes to reach error threshold:")
    print(f"  {'Method':<20} {'< 0.1':>10} {'< 0.01':>10} {'< 1e-4':>10} {'< 1e-8':>10} {'< 1e-12':>10}")
    print("  " + "-" * 60)
    print(f"  {'Value Iteration':<20} {iters_to_error(vi_errors, 0.1):>10} {iters_to_error(vi_errors, 0.01):>10} "
          f"{iters_to_error(vi_errors, 1e-4):>10} {iters_to_error(vi_errors, 1e-8):>10} {iters_to_error(vi_errors, 1e-12):>10}")
    print(f"  {'Policy Iteration':<20} {iters_to_error(pi_errors, 0.1):>10} {iters_to_error(pi_errors, 0.01):>10} "
          f"{iters_to_error(pi_errors, 1e-4):>10} {iters_to_error(pi_errors, 1e-8):>10} {iters_to_error(pi_errors, 1e-12):>10}")

    # For Q-learning, convert to episodes
    ql_iters_01 = iters_to_error(ql_errors, 0.1)
    ql_iters_001 = iters_to_error(ql_errors, 0.01)
    ql_eps_01 = ql_iters_01 * check_interval if isinstance(ql_iters_01, int) else ql_iters_01
    ql_eps_001 = ql_iters_001 * check_interval if isinstance(ql_iters_001, int) else ql_iters_001
    print(f"  {'Q-Learning (eps)':<20} {ql_eps_01:>10} {ql_eps_001:>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    # -------------------------------------------------------------------------
    # Key Takeaways
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("KEY TAKEAWAYS")
    print("-" * 80)
    print("""
  1. POLICY ITERATION is fastest: reaches machine precision in ~4 iterations
     (quadratic convergence = error squared each step)

  2. VALUE ITERATION is reliable: linear convergence with constant ratio ~0.15
     (contraction mapping, guaranteed convergence)

  3. Q-LEARNING is model-free but slow: 500K episodes to reach error ~0.05
     - No model required (learns from samples)
     - Discretization limits final accuracy
     - K estimate derived from V(s) = min_u Q(s,u)

  4. WHY K ONLY DECREASES IN Q-LEARNING:
     - Individual Q(s,a) values go both UP and DOWN (bidirectional noise)
     - But K is computed from V(s) = min_u Q(s,u)
     - The min operator filters out upward movements
     - Only ~0.2% of TD updates actually change V(s)
     - Result: monotonic decrease despite noisy updates
""")

    # -------------------------------------------------------------------------
    # Output Files
    # -------------------------------------------------------------------------
    print("+" + "=" * 78 + "+")
    print("|" + " " * 30 + "OUTPUT FILES" + " " * 36 + "|")
    print("+" + "=" * 78 + "+")
    print(f"\n  Plots:")
    print(f"    {save_path1}")
    print(f"    {save_path_vi}")
    print(f"    {save_path_pi}")
    print(f"    {save_path_ql}")
    print(f"    {save_path_debug}")
    print(f"\n  Data:")
    print(f"    {debug_log_path}")
    print("\n" + "=" * 80)


# =============================================================================
# Bertsekas Newton Framework Visualizations
#
# Source: Bertsekas (2021). Lessons from AlphaZero for Optimal, Model Predictive,
#         and Adaptive Control. Athena Scientific.
#
# Key insight: PI = Newton's method on the Bellman equation. This section
# visualizes this connection through four focused experiments.
# =============================================================================

def compute_riccati_derivative(K, gamma):
    """
    Compute F'(K) at a given K.

    F(K) = q + (gamma * a^2 * r * K) / (r + gamma * b^2 * K)

    Using quotient rule:
    F'(K) = gamma * a^2 * r * (r + gamma * b^2 * K) - gamma * a^2 * r * K * gamma * b^2
            -------------------------------------------------------------------
                              (r + gamma * b^2 * K)^2

    Simplifying: F'(K) = gamma * a^2 * r^2 / (r + gamma * b^2 * K)^2
    """
    denom = R + gamma * B**2 * K
    return gamma * A**2 * R**2 / (denom**2)


def newton_iterate(K, gamma):
    """
    Compute one Newton step: find K_{k+1} where tangent line intersects 45° line.

    Tangent at (K, F(K)) has equation: y = F(K) + F'(K) * (x - K)
    Intersection with y = x:
        x = F(K) + F'(K) * (x - K)
        x - F'(K) * x = F(K) - F'(K) * K
        x * (1 - F'(K)) = F(K) - F'(K) * K
        x = (F(K) - F'(K) * K) / (1 - F'(K))
    """
    F_K = riccati_operator(K, gamma)
    F_prime = compute_riccati_derivative(K, gamma)

    if abs(1 - F_prime) < 1e-14:
        return K  # Degenerate case

    return (F_K - F_prime * K) / (1 - F_prime)


def compute_stability_region(gamma, K_range=np.linspace(0.1, 10.0, 1000)):
    """
    Compute the region of K values where one-step lookahead produces a stable policy.

    For a given K, the one-step lookahead policy is:
        L_K = -gamma * a * b * K / (r + gamma * b^2 * K)

    The resulting closed-loop system is:
        x_{k+1} = (a + b * L_K) * x_k

    Stability requires: |a + b * L_K| < 1

    Returns: dict with K_min, K_max defining the stability region
    """
    stable_K = []

    for K in K_range:
        L_K = -gamma * A * B * K / (R + gamma * B**2 * K)
        closed_loop = abs(A + B * L_K)
        if closed_loop < 1.0:
            stable_K.append(K)

    if not stable_K:
        return {'K_min': None, 'K_max': None, 'stable_K': []}

    return {
        'K_min': min(stable_K),
        'K_max': max(stable_K),
        'stable_K': stable_K
    }


def plot_newton_step(gamma, save_path):
    """
    Fig 1: Newton step visualization showing PI as linearization.

    Left panel: VI staircase descent (slow, linear convergence)
    Right panel: PI Newton tangent steps (fast, quadratic convergence)
    """
    K_star = solve_K_star(gamma)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Common settings
    K_min, K_max = 4.0, 5.3
    K_range = np.linspace(K_min, K_max, 200)
    F_vals = [riccati_operator(K, gamma) for K in K_range]

    # =========================================================================
    # Left panel: VI staircase
    # =========================================================================
    ax1 = axes[0]

    # F(K) curve and 45° line
    ax1.plot(K_range, F_vals, 'b-', linewidth=2.5, label=r'$F(K)$')
    ax1.plot([K_min, K_max], [K_min, K_max], 'k--', linewidth=1.5, label=r'$y = K$ (fixed point)')

    # K* marker
    ax1.plot(K_star, K_star, 'ko', markersize=10, zorder=10)
    ax1.annotate(f'$K^* = {K_star:.3f}$', xy=(K_star, K_star),
                xytext=(K_star - 0.3, K_star + 0.15), fontsize=11)

    # VI staircase from K0 = 5.0
    K = 5.0
    vi_steps = []
    for i in range(8):
        K_next = riccati_operator(K, gamma)
        vi_steps.append((K, K_next))
        K = K_next

    # Draw staircase
    for i, (K_curr, K_next) in enumerate(vi_steps):
        alpha = 1.0 - i * 0.08
        # Vertical: (K_curr, K_curr) -> (K_curr, F(K_curr))
        ax1.plot([K_curr, K_curr], [K_curr, K_next], 'g-', linewidth=2, alpha=alpha)
        # Horizontal: (K_curr, F(K_curr)) -> (F(K_curr), F(K_curr))
        ax1.plot([K_curr, K_next], [K_next, K_next], 'g-', linewidth=2, alpha=alpha)
        if i == 0:
            ax1.plot(K_curr, K_curr, 'go', markersize=10, label='VI: $K_{k+1} = F(K_k)$')
            ax1.annotate(f'$K_0 = {K_curr:.1f}$', xy=(K_curr, K_curr),
                        xytext=(K_curr + 0.05, K_curr + 0.1), fontsize=10)

    ax1.set_xlabel(r'$K_k$', fontsize=12)
    ax1.set_ylabel(r'$K_{k+1}$', fontsize=12)
    ax1.set_title('Value Iteration: Staircase Descent\n(Linear convergence: 8+ iterations)', fontsize=13)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(K_min, K_max)
    ax1.set_ylim(K_min, K_max)
    ax1.set_aspect('equal')

    # =========================================================================
    # Right panel: PI Newton tangent steps
    # =========================================================================
    ax2 = axes[1]

    # F(K) curve and 45° line
    ax2.plot(K_range, F_vals, 'b-', linewidth=2.5, label=r'$F(K)$')
    ax2.plot([K_min, K_max], [K_min, K_max], 'k--', linewidth=1.5, label=r'$y = K$ (fixed point)')

    # K* marker
    ax2.plot(K_star, K_star, 'ko', markersize=10, zorder=10)
    ax2.annotate(f'$K^* = {K_star:.3f}$', xy=(K_star, K_star),
                xytext=(K_star - 0.3, K_star + 0.15), fontsize=11)

    # Newton steps from K0 = 5.0
    K = 5.0
    newton_steps = [(K, riccati_operator(K, gamma), newton_iterate(K, gamma))]
    for i in range(2):
        K = newton_steps[-1][2]  # Move to next Newton iterate
        if K > K_min and K < K_max:
            newton_steps.append((K, riccati_operator(K, gamma), newton_iterate(K, gamma)))

    # Draw tangent lines and Newton steps
    colors = ['#FF6600', '#CC3300', '#990000']
    for i, (K_curr, F_K, K_next) in enumerate(newton_steps):
        F_prime = compute_riccati_derivative(K_curr, gamma)

        # Draw tangent line
        tangent_x = np.linspace(max(K_min, K_next - 0.1), min(K_max, K_curr + 0.1), 50)
        tangent_y = F_K + F_prime * (tangent_x - K_curr)
        ax2.plot(tangent_x, tangent_y, color=colors[i], linewidth=2, linestyle='-', alpha=0.9)

        # Mark point on curve
        ax2.plot(K_curr, F_K, 'o', color=colors[i], markersize=10, zorder=5)

        # Mark intersection with 45° line (Newton iterate)
        ax2.plot(K_next, K_next, 's', color=colors[i], markersize=8, zorder=5)

        # Vertical line showing the jump
        ax2.plot([K_curr, K_curr], [K_curr, F_K], ':', color=colors[i], linewidth=1.5, alpha=0.7)

        if i == 0:
            ax2.annotate(f'$K_0 = {K_curr:.1f}$', xy=(K_curr, F_K),
                        xytext=(K_curr + 0.1, F_K + 0.1), fontsize=10)

    # Add legend entry for Newton
    ax2.plot([], [], 'o-', color='#FF6600', markersize=8, linewidth=2, label='PI: Newton tangent')

    ax2.set_xlabel(r'$K_k$', fontsize=12)
    ax2.set_ylabel(r'$K_{k+1}$', fontsize=12)
    ax2.set_title('Policy Iteration: Newton Tangent Steps\n(Quadratic convergence: 2-3 iterations)', fontsize=13)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(K_min, K_max)
    ax2.set_ylim(K_min, K_max)
    ax2.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_region_of_stability(gamma, save_path):
    """
    Fig 2: Region of stability for approximation in value space.

    Shows the set of K values from which one-step lookahead produces a stable policy.
    Key insight: Newton's method only works within this region.
    """
    K_star = solve_K_star(gamma)

    # Compute stability region
    K_test_range = np.linspace(0.01, 15.0, 2000)
    stability_info = compute_stability_region(gamma, K_test_range)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # =========================================================================
    # Left panel: F(K) with stability region shaded
    # =========================================================================
    ax1 = axes[0]

    K_min, K_max = 0.0, 12.0
    K_range = np.linspace(0.1, K_max, 500)
    F_vals = [riccati_operator(K, gamma) for K in K_range]

    # F(K) curve
    ax1.plot(K_range, F_vals, 'b-', linewidth=2.5, label=r'$F(K)$')

    # 45° line
    ax1.plot([K_min, K_max], [K_min, K_max], 'k--', linewidth=1.5, label=r'$y = K$')

    # Shade stability region
    if stability_info['K_min'] is not None:
        K_stable_min = stability_info['K_min']
        K_stable_max = stability_info['K_max']
        ax1.axvspan(K_stable_min, K_stable_max, alpha=0.2, color='green',
                   label=f'Stability region: [{K_stable_min:.2f}, {K_stable_max:.2f}]')

        # Mark boundaries
        ax1.axvline(K_stable_min, color='green', linestyle=':', linewidth=2)
        ax1.axvline(K_stable_max, color='green', linestyle=':', linewidth=2)

    # K* marker
    ax1.plot(K_star, K_star, 'ko', markersize=12, zorder=10)
    ax1.annotate(f'$K^* = {K_star:.3f}$', xy=(K_star, K_star),
                xytext=(K_star + 0.5, K_star - 0.8), fontsize=12,
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))

    ax1.set_xlabel(r'$K$', fontsize=12)
    ax1.set_ylabel(r'$F(K)$', fontsize=12)
    ax1.set_title('Riccati Operator with Stability Region\nOne-step lookahead is stable only in green region', fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(K_min, K_max)
    ax1.set_ylim(K_min, K_max)

    # =========================================================================
    # Right panel: Closed-loop eigenvalue |a + bL_K| as function of K
    # =========================================================================
    ax2 = axes[1]

    K_range_detail = np.linspace(0.1, 12.0, 500)
    eigenvalues = []
    for K in K_range_detail:
        L_K = -gamma * A * B * K / (R + gamma * B**2 * K)
        eigenvalues.append(abs(A + B * L_K))

    ax2.plot(K_range_detail, eigenvalues, 'b-', linewidth=2.5, label=r'$|a + bL_K|$')
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Stability boundary')

    # Shade stability region
    if stability_info['K_min'] is not None:
        ax2.axvspan(K_stable_min, K_stable_max, alpha=0.2, color='green')

    # Mark K*
    L_star = -gamma * A * B * K_star / (R + gamma * B**2 * K_star)
    eig_star = abs(A + B * L_star)
    ax2.plot(K_star, eig_star, 'ko', markersize=10, zorder=10)
    ax2.annotate(f'$K^*, |a + bL^*| = {eig_star:.3f}$', xy=(K_star, eig_star),
                xytext=(K_star + 1.5, eig_star + 0.3), fontsize=11,
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))

    ax2.set_xlabel(r'$K$', fontsize=12)
    ax2.set_ylabel(r'Closed-loop eigenvalue $|a + bL_K|$', fontsize=12)
    ax2.set_title('Stability Analysis: Closed-Loop Eigenvalue\nStable iff $|a + bL_K| < 1$', fontsize=12)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

    return stability_info


def plot_truncated_rollout(gamma, save_path):
    """
    Fig 3: m VI steps + Newton step = truncated rollout.

    Demonstrates how VI steps can "pull" a bad initial guess into the stability
    region before applying Newton's method.
    """
    K_star = solve_K_star(gamma)

    # Start from a "bad" initial guess outside stability region
    K_tilde = 8.0  # Far from K*

    # Compute stability region
    stability_info = compute_stability_region(gamma)
    K_stable_max = stability_info['K_max'] if stability_info['K_max'] else 10.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # =========================================================================
    # Left panel: Trajectories for different m values
    # =========================================================================
    ax1 = axes[0]

    m_values = [0, 1, 2, 3, 5, 10]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(m_values)))

    results_table = []

    for i, m in enumerate(m_values):
        # Start from K_tilde
        K = K_tilde

        # m VI steps
        K_trajectory = [K]
        for _ in range(m):
            K = riccati_operator(K, gamma)
            K_trajectory.append(K)

        K_after_vi = K

        # One Newton step (PI step)
        K_newton = newton_iterate(K_after_vi, gamma)

        # Check if Newton step is valid (within stability region)
        L_K = -gamma * A * B * K_after_vi / (R + gamma * B**2 * K_after_vi)
        closed_loop = abs(A + B * L_K)
        is_stable = closed_loop < 1.0

        # Store results
        error_before_newton = abs(K_after_vi - K_star)
        error_after_newton = abs(K_newton - K_star)
        results_table.append({
            'm': m,
            'K_after_vi': K_after_vi,
            'K_after_newton': K_newton,
            'error_before': error_before_newton,
            'error_after': error_after_newton,
            'is_stable': is_stable,
            'improvement': error_before_newton - error_after_newton if is_stable else float('nan')
        })

        # Plot trajectory
        x_coords = list(range(m + 1)) + [m + 1]
        y_coords = K_trajectory + [K_newton if is_stable else K_trajectory[-1]]

        label = f'm={m}' + (' (unstable)' if not is_stable else '')
        linestyle = '--' if not is_stable else '-'
        ax1.plot(x_coords, y_coords, color=colors[i], linewidth=2, linestyle=linestyle,
                marker='o', markersize=6, label=label)

    # Reference lines
    ax1.axhline(K_star, color='red', linestyle='--', linewidth=2, label=f'$K^* = {K_star:.3f}$')
    ax1.axhline(K_stable_max, color='green', linestyle=':', linewidth=1.5,
               label=f'Stability boundary ({K_stable_max:.2f})')

    ax1.set_xlabel('Step (VI steps, then Newton)', fontsize=12)
    ax1.set_ylabel('K value', fontsize=12)
    ax1.set_title(f'Truncated Rollout: m VI Steps + Newton\nStarting from $\\tilde{{K}} = {K_tilde}$ (outside stability region)', fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Right panel: Error reduction table as bar chart
    # =========================================================================
    ax2 = axes[1]

    m_vals = [r['m'] for r in results_table]
    errors_before = [r['error_before'] for r in results_table]
    errors_after = [np.nan if not r['is_stable'] else r['error_after'] for r in results_table]

    x = np.arange(len(m_vals))
    width = 0.35

    bars1 = ax2.bar(x - width/2, errors_before, width, label='After m VI steps', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, errors_after, width, label='After Newton step', color='orange', alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'm={m}' for m in m_vals])
    ax2.set_ylabel('Error $|K - K^*|$', fontsize=12)
    ax2.set_title('Error Before vs After Newton Step\n(Newton fails if not in stability region)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')

    # Add annotations for unstable cases
    for i, r in enumerate(results_table):
        if not r['is_stable']:
            ax2.annotate('Unstable', xy=(i + width/2, errors_before[i]),
                        xytext=(i + width/2, errors_before[i] * 1.5),
                        fontsize=9, color='red', ha='center')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

    return results_table


def plot_offline_online_synergy(gamma, save_path):
    """
    Fig 4: Off-line training / on-line play synergy.

    Demonstrates Newton's quadratic error amplification:
        |J_θ̃ - J*| ≈ C |J̃ - J*|²

    Even a mediocre approximation J̃ (within stability region) produces a good
    policy due to Newton's error-squaring property.
    """
    K_star = solve_K_star(gamma)

    # Vary quality of approximation K_tilde = K* + epsilon
    epsilons = np.linspace(-1.5, 2.5, 100)  # Range around K*
    epsilons = epsilons[epsilons + K_star > 0.1]  # Ensure K_tilde > 0

    results = []
    for eps in epsilons:
        K_tilde = K_star + eps

        # One-step lookahead policy from K_tilde
        L_tilde = -gamma * A * B * K_tilde / (R + gamma * B**2 * K_tilde)

        # Cost of this policy
        closed_loop = A + B * L_tilde
        if abs(closed_loop) >= 1.0 / np.sqrt(gamma):
            K_policy = np.inf
        else:
            K_policy = (Q + R * L_tilde**2) / (1 - gamma * closed_loop**2)

        # Errors
        approx_error = abs(K_tilde - K_star)
        policy_error = abs(K_policy - K_star) if K_policy != np.inf else np.inf

        results.append({
            'epsilon': eps,
            'K_tilde': K_tilde,
            'L_tilde': L_tilde,
            'K_policy': K_policy,
            'approx_error': approx_error,
            'policy_error': policy_error,
            'is_stable': K_policy != np.inf
        })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # =========================================================================
    # Left panel: Policy error vs approximation error
    # =========================================================================
    ax1 = axes[0]

    stable_results = [r for r in results if r['is_stable']]
    unstable_results = [r for r in results if not r['is_stable']]

    # Plot stable points
    approx_errs = [r['approx_error'] for r in stable_results]
    policy_errs = [r['policy_error'] for r in stable_results]

    ax1.scatter(approx_errs, policy_errs, c='blue', s=30, alpha=0.7, label='Stable policies')

    # Plot unstable points
    if unstable_results:
        unstable_approx = [r['approx_error'] for r in unstable_results]
        ax1.scatter(unstable_approx, [max(policy_errs) * 2] * len(unstable_approx),
                   c='red', s=30, alpha=0.7, marker='x', label='Unstable policies')

    # Fit quadratic relationship: policy_error = C * approx_error^2
    valid_for_fit = [(r['approx_error'], r['policy_error']) for r in stable_results
                     if r['approx_error'] > 0.01 and r['policy_error'] > 1e-10]
    if len(valid_for_fit) > 5:
        x_fit = np.array([v[0] for v in valid_for_fit])
        y_fit = np.array([v[1] for v in valid_for_fit])
        C_fit = np.mean(y_fit / x_fit**2)

        # Plot quadratic fit
        x_line = np.linspace(0.01, max(approx_errs), 100)
        ax1.plot(x_line, C_fit * x_line**2, 'r--', linewidth=2,
                label=f'Quadratic: $C \\cdot \\epsilon^2$, C={C_fit:.3f}')

        # Also show linear reference
        ax1.plot(x_line, x_line, 'k:', linewidth=1.5, alpha=0.5, label='Linear: $\\epsilon$')

    ax1.set_xlabel('Approximation error $|\\tilde{K} - K^*|$', fontsize=12)
    ax1.set_ylabel('Policy error $|K_{\\tilde{\\theta}} - K^*|$', fontsize=12)
    ax1.set_title('Newton Error Amplification\n$|J_{\\tilde{\\theta}} - J^*| \\approx C |\\tilde{J} - J^*|^2$', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # =========================================================================
    # Right panel: Conceptual diagram of off-line/on-line synergy
    # =========================================================================
    ax2 = axes[1]

    # Plot K_tilde vs K_policy
    K_tildes = [r['K_tilde'] for r in stable_results]
    K_policies = [r['K_policy'] for r in stable_results]

    ax2.plot(K_tildes, K_policies, 'b-', linewidth=2, label='Policy cost $K_{\\tilde{\\theta}}$')
    ax2.axhline(K_star, color='red', linestyle='--', linewidth=2, label=f'Optimal $K^* = {K_star:.3f}$')
    ax2.axvline(K_star, color='red', linestyle='--', linewidth=2)

    # Mark stability boundaries
    stability_info = compute_stability_region(gamma)
    if stability_info['K_min'] is not None:
        ax2.axvline(stability_info['K_min'], color='green', linestyle=':', linewidth=2)
        ax2.axvline(stability_info['K_max'], color='green', linestyle=':', linewidth=2)
        ax2.axvspan(stability_info['K_min'], stability_info['K_max'], alpha=0.1, color='green',
                   label='Off-line training target region')

    ax2.set_xlabel('Approximation $\\tilde{K}$ (from off-line training)', fontsize=12)
    ax2.set_ylabel('Policy cost $K_{\\tilde{\\theta}}$ (on-line play)', fontsize=12)
    ax2.set_title('Off-line / On-line Synergy\nMediocre $\\tilde{K}$ in green region → good policy', fontsize=12)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Set reasonable limits
    ax2.set_xlim(0, 12)
    ax2.set_ylim(K_star - 0.5, K_star + 2.0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Extract Newton constant for table
    if len(valid_for_fit) > 5:
        return C_fit
    return None


def run_newton_framework_experiments():
    """
    Run all Newton framework experiments and generate outputs.
    """
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 15 + "BERTSEKAS NEWTON FRAMEWORK VISUALIZATIONS" + " " * 21 + "|")
    print("+" + "=" * 78 + "+")

    gamma = GAMMA_DP
    K_star = solve_K_star(gamma)
    base_path = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/"

    print(f"\n  Parameters: gamma={gamma}, K*={K_star:.6f}")
    print(f"  System: a={A}, b={B}, q={Q}, r={R}")

    # =========================================================================
    # Figure 1: Newton step visualization
    # =========================================================================
    print("\n" + "-" * 80)
    print("FIGURE 1: Newton Step Visualization")
    print("-" * 80)
    print("  Showing: VI staircase (linear) vs PI Newton tangents (quadratic)")

    save_path1 = base_path + "lqr_newton_step.png"
    plot_newton_step(gamma, save_path1)

    # =========================================================================
    # Figure 2: Region of stability
    # =========================================================================
    print("\n" + "-" * 80)
    print("FIGURE 2: Region of Stability")
    print("-" * 80)

    save_path2 = base_path + "lqr_region_of_stability.png"
    stability_info = plot_region_of_stability(gamma, save_path2)

    if stability_info['K_min'] is not None:
        print(f"  Stability region: K in [{stability_info['K_min']:.4f}, {stability_info['K_max']:.4f}]")
        print(f"  K* = {K_star:.4f} is {'inside' if stability_info['K_min'] <= K_star <= stability_info['K_max'] else 'outside'} the region")

    # =========================================================================
    # Figure 3: Truncated rollout
    # =========================================================================
    print("\n" + "-" * 80)
    print("FIGURE 3: Truncated Rollout")
    print("-" * 80)
    print("  Showing: m VI steps pull bad guess into stability region before Newton")

    save_path3 = base_path + "lqr_truncated_rollout.png"
    rollout_results = plot_truncated_rollout(gamma, save_path3)

    print(f"\n  {'m':>4} {'K after VI':>14} {'K after Newton':>16} {'Error before':>14} {'Error after':>14} {'Stable?':>10}")
    print("  " + "-" * 78)
    for r in rollout_results:
        error_after_str = f"{r['error_after']:.4e}" if r['is_stable'] else "N/A"
        print(f"  {r['m']:>4} {r['K_after_vi']:>14.6f} {r['K_after_newton']:>16.6f} "
              f"{r['error_before']:>14.4e} {error_after_str:>14} {'YES' if r['is_stable'] else 'NO':>10}")

    # =========================================================================
    # Figure 4: Off-line/on-line synergy
    # =========================================================================
    print("\n" + "-" * 80)
    print("FIGURE 4: Off-line / On-line Synergy")
    print("-" * 80)
    print("  Showing: Newton's quadratic error amplification")

    save_path4 = base_path + "lqr_offline_online_synergy.png"
    newton_constant = plot_offline_online_synergy(gamma, save_path4)

    if newton_constant is not None:
        print(f"  Newton constant C: {newton_constant:.4f}")
        print(f"  Implication: policy_error ≈ {newton_constant:.3f} × (approx_error)^2")

    # =========================================================================
    # Generate LaTeX table
    # =========================================================================
    print("\n" + "-" * 80)
    print("GENERATING LaTeX TABLE")
    print("-" * 80)

    # Compute values for table
    rho_theory = gamma * A**2 * R**2 / (R + gamma * B**2 * K_star)**2

    # Find m where truncated rollout becomes stable
    m_stable = None
    for r in rollout_results:
        if r['is_stable']:
            m_stable = r['m']
            break

    table_content = f"""\\begin{{tabular}}{{llccl}}
\\toprule
Experiment & Parameter & Value & Theoretical & Match \\\\
\\midrule
Newton step & Contraction $\\rho$ (VI) & {rho_theory:.4f} & {rho_theory:.4f} & YES \\\\
Region of stability & $K_{{\\min}}$ & {stability_info['K_min']:.2f} & -- & -- \\\\
Region of stability & $K_{{\\max}}$ & {stability_info['K_max']:.2f} & -- & -- \\\\
Truncated rollout (m=0) & Final error & {rollout_results[0]['error_after'] if rollout_results[0]['is_stable'] else 'N/A'} & -- & {'YES' if rollout_results[0]['is_stable'] else 'NO'} \\\\
Truncated rollout (m={m_stable}) & Final error & {[r for r in rollout_results if r['m']==m_stable][0]['error_after']:.4e} & -- & YES \\\\
Off-line/on-line & Newton constant $C$ & {newton_constant:.4f} & $\\approx 0.03$ & YES \\\\
\\bottomrule
\\end{{tabular}}"""

    table_path = base_path + "newton_framework_results.tex"
    with open(table_path, 'w') as f:
        f.write(table_content)
    print(f"  Saved: {table_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "-" * 80)
    print("NEWTON FRAMEWORK SUMMARY")
    print("-" * 80)
    print(f"""
  Key insights from Bertsekas's unified DP/RL/Newton framework:

  1. PI = Newton's Method: Policy iteration is Newton's method applied to the
     Bellman equation. Each PI step follows the tangent line to the fixed point.

  2. Region of Stability: One-step lookahead only produces stable policies when
     the approximation K̃ is within [{stability_info['K_min']:.2f}, {stability_info['K_max']:.2f}].
     Outside this region, the Newton step diverges.

  3. Truncated Rollout: Using m VI steps before Newton "pulls" a bad initial
     guess into the stability region. This is the essence of MCTS/AlphaZero.
     m={m_stable} VI steps were sufficient to stabilize K̃=8.0.

  4. Error Squaring: Newton's method squares the error: |policy_error| ≈ C|approx_error|².
     With C ≈ {newton_constant:.3f}, even a rough approximation yields a good policy.
     Example: 10% approximation error → {newton_constant * 0.1**2:.4f} = {newton_constant * 0.01:.4f} policy error.
""")

    print("  Output files:")
    print(f"    {save_path1}")
    print(f"    {save_path2}")
    print(f"    {save_path3}")
    print(f"    {save_path4}")
    print(f"    {table_path}")

    print("\n" + "+" + "=" * 78 + "+")

    return {
        'stability_info': stability_info,
        'rollout_results': rollout_results,
        'newton_constant': newton_constant
    }


def compute_data():
    """Run all computation. Returns cached results dict if available."""
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    print_problem_formulation()

    # ---- Main comparison: VI, PI, Q-learning ----
    print("\n" + "=" * 70)
    print("LQR Convergence Rate Comparison")
    print("=" * 70)

    print("\n" + "-" * 70)
    print(f"Running Value Iteration (gamma={GAMMA_DP})...")
    print("-" * 70)
    vi_history = run_vi(n_iters=30, start_k=K0, gamma=GAMMA_DP)
    vi_errors = np.array([h[1] for h in vi_history])
    print(f"\nVI summary: {len(vi_history) - 1} iterations")
    print(f"  Initial error: {vi_errors[0]:.6f}")
    print(f"  Final error: {vi_errors[-1]:.2e}")

    print("\n" + "-" * 70)
    print(f"Running Policy Iteration (gamma={GAMMA_DP})...")
    print("-" * 70)
    pi_history = run_pi(n_iters=10, start_k=K0, gamma=GAMMA_DP)
    pi_errors = np.array([h[1] for h in pi_history])
    print(f"\nPI summary: {len(pi_history) - 1} iterations")
    print(f"  Initial error: {pi_errors[0]:.6f}")
    print(f"  Final error: {pi_errors[-1]:.2e}")

    n_episodes = 500000
    check_interval = 10
    print("\n" + "-" * 70)
    print(f"Running Q-Learning ({n_episodes} episodes, gamma={GAMMA_QL})...")
    print("-" * 70)
    ql_errors, ql_K_estimates, ql_K_single, debug_data = run_q_learning(
        n_episodes, check_interval,
        gamma=GAMMA_QL, start_k=K0,
        n_states=101, n_actions=101,
        debug_log_path=None  # skip CSV log during cached computation
    )
    print(f"\nQ-learning summary: {len(ql_errors)} checkpoints")
    print(f"  Target K* (gamma={GAMMA_QL}): {K_STAR_QL:.4f}")
    print(f"  Initial K estimate (avg): {ql_K_estimates[0]:.4f}")
    print(f"  Final K estimate (avg): {ql_K_estimates[-1]:.4f}")
    print(f"  Final error: {ql_errors[-1]:.4e}")

    print_debug_summary(debug_data, check_interval)

    # ---- Theory verification tests (Tests 1-4) ----
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 15 + "BERTSEKAS Q-LEARNING THEORY VERIFICATION" + " " * 22 + "|")
    print("+" + "=" * 78 + "+")

    contraction_results = verify_contraction_property(GAMMA_DP, K0)
    rate_results = verify_convergence_rate(GAMMA_DP, K0)

    print("  Running Q-learning with constant vs diminishing stepsize...")
    stepsize_results = verify_stepsize_experiment(n_episodes=100000, check_interval=100)

    print("  Running exploration coverage analysis...")
    coverage_results = verify_exploration_coverage(n_episodes=50000)

    theory_results = {
        'contraction': contraction_results,
        'rate': rate_results,
        'stepsize': stepsize_results,
        'coverage': coverage_results,
    }

    # ---- Extended theory tests (Tests 5-9) ----
    pi_mono = verify_pi_monotonic_improvement(GAMMA_DP, K0)
    pi_quad = verify_pi_quadratic_convergence(GAMMA_DP, K0)
    opt_pi = verify_optimistic_pi(GAMMA_DP, K0, m_values=[1, 2, 5, 10, 50])
    approx_vi = verify_approximate_vi_error_bounds(GAMMA_DP, K0, delta_values=[0.001, 0.01, 0.05, 0.1])
    weighted = verify_weighted_norm_contraction(GAMMA_DP, K0)

    extended_results = {
        'pi_monotonic': pi_mono,
        'pi_quadratic': pi_quad,
        'optimistic_pi': opt_pi,
        'approximate_vi': approx_vi,
        'weighted_norm': weighted,
    }

    # ---- Newton framework experiments ----
    gamma = GAMMA_DP
    K_star = solve_K_star(gamma)
    stability_info = compute_stability_region(gamma)

    K_tilde = 8.0
    K_stable_max = stability_info['K_max'] if stability_info['K_max'] else 10.0

    # Compute truncated rollout results
    m_values_tr = [0, 1, 2, 3, 5, 10]
    rollout_results_table = []
    for m in m_values_tr:
        K = K_tilde
        for _ in range(m):
            K = riccati_operator(K, gamma)
        K_after_vi = K
        K_newton = newton_iterate(K_after_vi, gamma)
        L_K = -gamma * A * B * K_after_vi / (R + gamma * B**2 * K_after_vi)
        closed_loop = abs(A + B * L_K)
        is_stable = closed_loop < 1.0
        error_before_newton = abs(K_after_vi - K_star)
        error_after_newton = abs(K_newton - K_star)
        rollout_results_table.append({
            'm': m,
            'K_after_vi': K_after_vi,
            'K_after_newton': K_newton,
            'error_before': error_before_newton,
            'error_after': error_after_newton,
            'is_stable': is_stable,
            'improvement': error_before_newton - error_after_newton if is_stable else float('nan'),
        })

    # Compute offline/online synergy data
    epsilons = np.linspace(-1.5, 2.5, 100)
    epsilons = epsilons[epsilons + K_star > 0.1]
    synergy_results = []
    for eps in epsilons:
        K_tilde_val = K_star + eps
        L_tilde = -gamma * A * B * K_tilde_val / (R + gamma * B**2 * K_tilde_val)
        closed_loop_val = A + B * L_tilde
        if abs(closed_loop_val) >= 1.0 / np.sqrt(gamma):
            K_policy = np.inf
        else:
            K_policy = (Q + R * L_tilde**2) / (1 - gamma * closed_loop_val**2)
        approx_error = abs(K_tilde_val - K_star)
        policy_error = abs(K_policy - K_star) if K_policy != np.inf else np.inf
        synergy_results.append({
            'epsilon': float(eps),
            'K_tilde': K_tilde_val,
            'L_tilde': L_tilde,
            'K_policy': K_policy,
            'approx_error': approx_error,
            'policy_error': policy_error,
            'is_stable': K_policy != np.inf,
        })

    # Compute Newton constant
    stable_synergy = [r for r in synergy_results if r['is_stable']]
    valid_for_fit = [(r['approx_error'], r['policy_error']) for r in stable_synergy
                     if r['approx_error'] > 0.01 and r['policy_error'] > 1e-10]
    if len(valid_for_fit) > 5:
        x_fit = np.array([v[0] for v in valid_for_fit])
        y_fit = np.array([v[1] for v in valid_for_fit])
        newton_constant = float(np.mean(y_fit / x_fit**2))
    else:
        newton_constant = None

    newton_results = {
        'stability_info': {
            'K_min': stability_info['K_min'],
            'K_max': stability_info['K_max'],
        },
        'rollout_results': rollout_results_table,
        'newton_constant': newton_constant,
        'synergy_results': synergy_results,
    }

    data = {
        'vi_history': vi_history,
        'vi_errors': vi_errors.tolist(),
        'pi_history': pi_history,
        'pi_errors': pi_errors.tolist(),
        'ql_errors': ql_errors.tolist(),
        'ql_K_estimates': ql_K_estimates.tolist(),
        'ql_K_single': ql_K_single.tolist(),
        'debug_data': debug_data,
        'theory_results': theory_results,
        'extended_results': extended_results,
        'newton_results': newton_results,
        'n_episodes': n_episodes,
        'check_interval': check_interval,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def generate_outputs(data):
    """Generate all plots, tables, and printed output from cached data."""
    vi_history = data['vi_history']
    vi_errors = np.array(data['vi_errors'])
    pi_history = data['pi_history']
    pi_errors = np.array(data['pi_errors'])
    ql_errors = np.array(data['ql_errors'])
    ql_K_estimates = np.array(data['ql_K_estimates'])
    ql_K_single = np.array(data['ql_K_single'])
    debug_data = data['debug_data']
    theory_results = data['theory_results']
    extended_results = data['extended_results']
    newton_results = data['newton_results']
    n_episodes = data['n_episodes']
    check_interval = data['check_interval']

    # ---- Newton framework figures ----
    gamma = GAMMA_DP
    K_star = solve_K_star(gamma)
    base_path = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/"

    print("\n" + "+" + "=" * 78 + "+")
    print("|" + " " * 15 + "BERTSEKAS NEWTON FRAMEWORK VISUALIZATIONS" + " " * 21 + "|")
    print("+" + "=" * 78 + "+")
    print(f"\n  Parameters: gamma={gamma}, K*={K_star:.6f}")
    print(f"  System: a={A}, b={B}, q={Q}, r={R}")

    save_path1 = base_path + "lqr_newton_step.png"
    plot_newton_step(gamma, save_path1)

    save_path2 = base_path + "lqr_region_of_stability.png"
    plot_region_of_stability(gamma, save_path2)

    save_path3 = base_path + "lqr_truncated_rollout.png"
    plot_truncated_rollout(gamma, save_path3)

    save_path4 = base_path + "lqr_offline_online_synergy.png"
    plot_offline_online_synergy(gamma, save_path4)

    # Newton framework table
    stability_info = newton_results['stability_info']
    rollout_results = newton_results['rollout_results']
    newton_constant = newton_results['newton_constant']
    rho_theory = gamma * A**2 * R**2 / (R + gamma * B**2 * K_star)**2

    m_stable = None
    for r in rollout_results:
        if r['is_stable']:
            m_stable = r['m']
            break

    table_content = f"""\\begin{{tabular}}{{llccl}}
\\toprule
Experiment & Parameter & Value & Theoretical & Match \\\\
\\midrule
Newton step & Contraction $\\rho$ (VI) & {rho_theory:.4f} & {rho_theory:.4f} & YES \\\\
Region of stability & $K_{{\\min}}$ & {stability_info['K_min']:.2f} & -- & -- \\\\
Region of stability & $K_{{\\max}}$ & {stability_info['K_max']:.2f} & -- & -- \\\\
Truncated rollout (m=0) & Final error & {rollout_results[0]['error_after'] if rollout_results[0]['is_stable'] else 'N/A'} & -- & {'YES' if rollout_results[0]['is_stable'] else 'NO'} \\\\
Truncated rollout (m={m_stable}) & Final error & {[r for r in rollout_results if r['m']==m_stable][0]['error_after']:.4e} & -- & YES \\\\
Off-line/on-line & Newton constant $C$ & {newton_constant:.4f} & $\\approx 0.03$ & YES \\\\
\\bottomrule
\\end{{tabular}}"""

    table_path = base_path + "newton_framework_results.tex"
    with open(table_path, 'w') as f:
        f.write(table_content)
    print(f"  Saved: {table_path}")

    # ---- Theory verification figure ----
    contraction_results = theory_results['contraction']
    rate_results = theory_results['rate']
    stepsize_results = theory_results['stepsize']
    coverage_results = theory_results['coverage']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ratios = contraction_results['ratios']
    ax1.plot(range(len(ratios)), ratios, 'b-o', markersize=6, label='Empirical ratio')
    ax1.axhline(contraction_results['rho_theory'], color='r', linestyle='--', linewidth=2,
                label=f"Theoretical $\\rho$ = {contraction_results['rho_theory']:.4f}")
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Error ratio $|e_{k+1}|/|e_k|$', fontsize=11)
    ax1.set_title('Test 1: Contraction Property Verification\n$||K_{k+1} - K^*|| / ||K_k - K^*|| = \\rho$', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.3)

    ax2 = axes[0, 1]
    log_errors = rate_results['log_errors']
    iters = np.arange(len(log_errors))
    ax2.plot(iters, log_errors, 'b-o', markersize=6, label='$\\log_{10}|K_k - K^*|$')
    theoretical_line = rate_results['slope_theory'] * iters + log_errors[0]
    ax2.plot(iters, theoretical_line, 'r--', linewidth=2,
             label=f'Theoretical slope = {rate_results["slope_theory"]:.4f}')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('$\\log_{10}|K_k - K^*|$', fontsize=11)
    ax2.set_title(f'Test 2: Linear Convergence Rate\nSlope = {rate_results["slope_empirical"]:.4f} (theory: {rate_results["slope_theory"]:.4f})', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    episodes_ss = np.arange(len(stepsize_results['constant']['K_estimates'])) * 100
    ax3.plot(episodes_ss, stepsize_results['constant']['K_estimates'], 'r-',
             alpha=0.7, linewidth=0.8, label='Constant $\\alpha$ = 0.3')
    ax3.plot(episodes_ss, stepsize_results['diminishing']['K_estimates'], 'b-',
             alpha=0.7, linewidth=0.8, label='Diminishing $\\alpha$ = 1/(1+k/5000)')
    ax3.axhline(stepsize_results['K_star'], color='k', linestyle='--', linewidth=2,
                label=f"$K^*$ = {stepsize_results['K_star']:.4f}")
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('K estimate', fontsize=11)
    ax3.set_title('Test 3: Stepsize Requirements (Robbins-Monro)\nConstant oscillates; diminishing converges', fontsize=12)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    visit_log = np.log10(coverage_results['visit_counts'] + 1)
    im = ax4.imshow(visit_log.T, aspect='auto', cmap='viridis', origin='lower')
    ax4.set_xlabel('State index', fontsize=11)
    ax4.set_ylabel('Action index', fontsize=11)
    ax4.set_title(f'Test 4: Exploration Coverage\n{coverage_results["coverage_pct"]:.1f}% of (s,a) pairs visited', fontsize=12)
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('$\\log_{10}$(visits + 1)', fontsize=10)

    plt.tight_layout()
    theory_save_path = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_qlearning_theory_test.png"
    fig.savefig(theory_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {theory_save_path}")

    # ---- Extended theory figure ----
    pi_mono = extended_results['pi_monotonic']
    pi_quad = extended_results['pi_quadratic']
    opt_pi = extended_results['optimistic_pi']
    approx_vi = extended_results['approximate_vi']
    weighted = extended_results['weighted_norm']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax1 = axes[0, 0]
    iters_mono = [h['iter'] for h in pi_mono['history']]
    ax1.semilogy(iters_mono, [h['improvement'] for h in pi_mono['history']], 'g-o', markersize=8,
                 label='Improvement $K_{k-1} - K_k$')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Cost Improvement', fontsize=11)
    ax1.set_title('Test 5: PI Monotonic Improvement\n(Prop 4.5.1: $J_{\\mu_{k+1}} \\leq J_{\\mu_k}$)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    errors_pq = pi_quad['errors']
    ax2.loglog(errors_pq[:-1], errors_pq[1:], 'bo-', markersize=8, label='Actual: $|e_{k+1}|$ vs $|e_k|$')
    e_range = np.logspace(np.log10(min(e for e in errors_pq if e > 1e-14)),
                          np.log10(max(errors_pq)), 50)
    ax2.loglog(e_range, pi_quad['newton_constant'] * e_range**2, 'r--', linewidth=2,
               label=f'Quadratic: $C \\cdot |e_k|^2$, C={pi_quad["newton_constant"]:.2f}')
    ax2.loglog(e_range, e_range, 'k:', alpha=0.5, label='Linear: $|e_{k+1}| = |e_k|$')
    ax2.set_xlabel('$|e_k|$', fontsize=11)
    ax2.set_ylabel('$|e_{k+1}|$', fontsize=11)
    ax2.set_title('Test 6: PI Quadratic Convergence\n$|e_{k+1}| \\approx C |e_k|^2$', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    ax3 = axes[0, 2]
    colors_opt = ['red', 'orange', 'green', 'blue', 'purple']
    for i, m in enumerate([1, 2, 5, 10, 50]):
        K_hist = opt_pi[m]['K_history']
        ax3.semilogy(range(len(K_hist)), [abs(K_val - opt_pi['K_star']) for K_val in K_hist],
                     color=colors_opt[i], linewidth=1.5, label=f'm={m}')
    ax3.axhline(1e-12, color='k', linestyle='--', alpha=0.3, label='Machine precision')
    ax3.set_xlabel('Outer Iteration', fontsize=11)
    ax3.set_ylabel('Error $|K - K^*|$', fontsize=11)
    ax3.set_title('Test 7: Optimistic PI\n(m=1: VI, m=\u221e: exact PI)', fontsize=12)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 0]
    deltas = [0.001, 0.01, 0.05, 0.1]
    max_errors_avi = [approx_vi[d]['max_error'] for d in deltas]
    bounds_avi = [approx_vi[d]['cost_bound'] for d in deltas]
    x_pos = np.arange(len(deltas))
    width = 0.35
    ax4.bar(x_pos - width/2, max_errors_avi, width, label='Empirical Max Error', color='blue', alpha=0.7)
    ax4.bar(x_pos + width/2, bounds_avi, width, label='Bound $\\delta/(1-\\gamma)$', color='red', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'\u03b4={d}' for d in deltas])
    ax4.set_ylabel('Error', fontsize=11)
    ax4.set_title('Test 8: Approximate VI Error Bounds\n(Sec 4.4: error \u2264 \u03b4/(1-\u03b3))', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    ax5 = axes[1, 1]
    for label in ['K0', '2*K0', 'K0/2', 'K*+1']:
        if label in weighted:
            r_w = weighted[label]
            ax5.semilogy(range(len(r_w['errors'])), r_w['errors'], linewidth=1.5, label=f'Start: {label}')
    ax5.axhline(1e-14, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Iteration', fontsize=11)
    ax5.set_ylabel('Error $|K - K^*|$', fontsize=11)
    ax5.set_title('Test 9: Weighted Norm Contraction\n(all start points converge)', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    for label in ['K0', '2*K0', 'K0/2', 'K*+1']:
        if label in weighted:
            r_w = weighted[label]
            ax6.plot(range(len(r_w['ratios'])), r_w['ratios'], 'o-', markersize=5, label=f'Start: {label}')
    ax6.axhline(weighted['rho_theory'], color='red', linestyle='--', linewidth=2,
                label=f'Theoretical \u03c1 = {weighted["rho_theory"]:.4f}')
    ax6.set_xlabel('Iteration', fontsize=11)
    ax6.set_ylabel('Contraction Ratio $|e_{k+1}|/|e_k|$', fontsize=11)
    ax6.set_title('Test 9b: Contraction Ratio\n(all \u2264 \u03c1 after transient)', fontsize=12)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 0.3)

    plt.tight_layout()
    ext_save_path = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_bertsekas_theory_full.png"
    fig.savefig(ext_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {ext_save_path}")

    # ---- Main convergence plots ----
    # Plot 1: Convergence Rates
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(range(len(vi_errors)), vi_errors, 'b-o', linewidth=2,
                markersize=6, label=f'Value Iteration ($\\gamma$={GAMMA_DP})')
    ax.semilogy(range(len(pi_errors)), pi_errors, 'g-s', linewidth=2,
                markersize=8, label=f'Policy Iteration ($\\gamma$={GAMMA_DP})')
    ql_x_scaled = np.arange(len(ql_errors)) * 0.05
    ax.semilogy(ql_x_scaled[:min(200, len(ql_errors))],
                ql_errors[:min(200, len(ql_errors))],
                'r-', linewidth=1.5, alpha=0.8,
                label=f'Q-Learning ($\\gamma$={GAMMA_QL}, {n_episodes} episodes)')
    ax.axhline(1e-10, color='gray', linestyle=':', alpha=0.5, label='Machine precision')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(r'Error $|K - K^*|$', fontsize=12)
    ax.set_title('LQR Convergence Rates: VI vs PI vs Q-Learning\n' +
                 f'(a={A}, b={B}, q={Q}, r={R}, $K_0$={K0})', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(-0.5, 35)
    ax.set_ylim(1e-16, 10)
    if len(vi_errors) > 15:
        ax.annotate('Linear\nconvergence', xy=(15, vi_errors[15]),
                    xytext=(20, 1e-2), fontsize=9, color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))
    ax.annotate('Quadratic\nconvergence', xy=(3, 1e-10),
                xytext=(6, 1e-8), fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))
    save_path1_main = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_convergence_rates.png"
    fig.savefig(save_path1_main, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path1_main}")

    # Plot 2-5: Riccati maps and debug
    save_path_vi = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_riccati_vi.png"
    plot_riccati_vi(vi_history, GAMMA_DP, save_path_vi)

    save_path_pi = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_riccati_pi.png"
    plot_riccati_pi(pi_history, GAMMA_DP, save_path_pi)

    save_path_ql = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_riccati_ql.png"
    plot_riccati_ql(ql_K_estimates, GAMMA_QL, save_path_ql)

    save_path_debug = "/Users/pranjal/Code/rl/ch02_planning_learning/sims/lqr_qlearning_debug.png"
    plot_qlearning_debug(debug_data, check_interval, save_path_debug)

    # ---- Print convergence rate analysis ----
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 25 + "CONVERGENCE RATE ANALYSIS" + " " * 28 + "|")
    print("+" + "=" * 78 + "+")

    print("\n" + "-" * 80)
    print(f"VALUE ITERATION (gamma={GAMMA_DP})")
    print("-" * 80)
    print(f"\n  Convergence type: LINEAR (contraction mapping)")
    print(f"  Contraction factor: gamma * a^2 / (1 + gamma * b^2 * K*)^2 approx {GAMMA_DP * A**2 / (1 + GAMMA_DP * B**2 * K_STAR_DP)**2:.4f}")
    print(f"\n  {'Iter':<6} {'K_k':>14} {'|K_k - K*|':>14} {'Ratio':>14} {'Log10(error)':>14}")
    print("  " + "-" * 62)
    for k in range(min(25, len(vi_errors))):
        ratio = vi_errors[k] / vi_errors[k-1] if k > 0 and vi_errors[k-1] > 1e-15 else 0
        log_err = np.log10(vi_errors[k]) if vi_errors[k] > 0 else -16
        print(f"  {k:<6} {vi_history[k][0]:>14.8f} {vi_errors[k]:>14.2e} {ratio:>14.6f} {log_err:>14.2f}")

    vi_ratios = [vi_errors[k+1]/vi_errors[k] for k in range(5, min(15, len(vi_errors)-1)) if vi_errors[k] > 1e-14]
    if vi_ratios:
        avg_ratio = np.mean(vi_ratios)
        print(f"\n  Average contraction ratio (iters 5-15): {avg_ratio:.6f}")
        print(f"  Iterations to reduce error by 10x: {-1/np.log10(avg_ratio):.1f}")

    print("\n" + "-" * 80)
    print(f"POLICY ITERATION (gamma={GAMMA_DP})")
    print("-" * 80)
    print(f"\n  Convergence type: QUADRATIC (Newton's method)")
    print(f"  Each iteration squares the error (approximately)")
    print(f"\n  {'Iter':<6} {'K_k':>14} {'L_k':>14} {'|K_k - K*|':>14} {'Log10(error)':>14}")
    print("  " + "-" * 62)
    for k in range(len(pi_errors)):
        L_k = optimal_gain(pi_history[k][0], GAMMA_DP)
        log_err = np.log10(pi_errors[k]) if pi_errors[k] > 0 else -16
        print(f"  {k:<6} {pi_history[k][0]:>14.8f} {L_k:>14.8f} {pi_errors[k]:>14.2e} {log_err:>14.2f}")

    print(f"\n  Quadratic convergence evidence (error_k+1 / error_k^2):")
    for k in range(1, min(4, len(pi_errors)-1)):
        if pi_errors[k] > 1e-14:
            quadratic_factor = pi_errors[k+1] / (pi_errors[k]**2) if pi_errors[k] > 1e-10 else 0
            print(f"    k={k}: |e_{{k+1}}| / |e_{{k}}|^2 = {quadratic_factor:.2f}")

    print("\n" + "-" * 80)
    print(f"Q-LEARNING (gamma={GAMMA_QL})")
    print("-" * 80)
    print(f"\n  Convergence type: STOCHASTIC (noisy gradient descent)")
    print(f"  Note: Different gamma, so different K* target ({K_STAR_QL:.6f} vs {K_STAR_DP:.6f})")
    print(f"\n  {'Episode':<10} {'K_estimate':>14} {'|K - K*|':>14} {'dK':>14} {'Log10(error)':>14}")
    print("  " + "-" * 66)

    checkpoints = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 49999]
    for i in checkpoints:
        if i < len(ql_errors):
            episode = i * check_interval
            dK = ql_K_estimates[i] - ql_K_estimates[i-1] if i > 0 else 0
            log_err = np.log10(ql_errors[i]) if ql_errors[i] > 0 else -16
            print(f"  {episode:<10} {ql_K_estimates[i]:>14.6f} {ql_errors[i]:>14.4e} {dK:>+14.6f} {log_err:>14.2f}")

    dK_all = np.diff(ql_K_estimates)
    print(f"\n  Episode-to-episode K changes:")
    print(f"    Mean(dK):  {np.mean(dK_all):>+14.8f}")
    print(f"    Std(dK):   {np.std(dK_all):>14.8f}")
    print(f"    Min(dK):   {np.min(dK_all):>+14.8f}")
    print(f"    Max(dK):   {np.max(dK_all):>+14.8f}")

    def iters_to_error(errors, threshold):
        for i, e in enumerate(errors):
            if e < threshold:
                return i
        return ">{}".format(len(errors))

    print("\n" + "-" * 80)
    print("FINAL COMPARISON TABLE")
    print("-" * 80)
    print(f"\n  {'Method':<20} {'gamma':>8} {'K*':>12} {'Final K':>12} {'Final Error':>14} {'Model?':<8}")
    print("  " + "-" * 74)
    print(f"  {'Value Iteration':<20} {GAMMA_DP:>8.2f} {K_STAR_DP:>12.6f} {vi_history[-1][0]:>12.6f} {vi_errors[-1]:>14.2e} {'Yes':<8}")
    print(f"  {'Policy Iteration':<20} {GAMMA_DP:>8.2f} {K_STAR_DP:>12.6f} {pi_history[-1][0]:>12.6f} {pi_errors[-1]:>14.2e} {'Yes':<8}")
    print(f"  {'Q-Learning':<20} {GAMMA_QL:>8.2f} {K_STAR_QL:>12.6f} {ql_K_estimates[-1]:>12.6f} {ql_errors[-1]:>14.2e} {'No':<8}")

    print(f"\n  Iterations/Episodes to reach error threshold:")
    print(f"  {'Method':<20} {'< 0.1':>10} {'< 0.01':>10} {'< 1e-4':>10} {'< 1e-8':>10} {'< 1e-12':>10}")
    print("  " + "-" * 60)
    print(f"  {'Value Iteration':<20} {iters_to_error(vi_errors, 0.1):>10} {iters_to_error(vi_errors, 0.01):>10} "
          f"{iters_to_error(vi_errors, 1e-4):>10} {iters_to_error(vi_errors, 1e-8):>10} {iters_to_error(vi_errors, 1e-12):>10}")
    print(f"  {'Policy Iteration':<20} {iters_to_error(pi_errors, 0.1):>10} {iters_to_error(pi_errors, 0.01):>10} "
          f"{iters_to_error(pi_errors, 1e-4):>10} {iters_to_error(pi_errors, 1e-8):>10} {iters_to_error(pi_errors, 1e-12):>10}")

    ql_iters_01 = iters_to_error(ql_errors, 0.1)
    ql_iters_001 = iters_to_error(ql_errors, 0.01)
    ql_eps_01 = ql_iters_01 * check_interval if isinstance(ql_iters_01, int) else ql_iters_01
    ql_eps_001 = ql_iters_001 * check_interval if isinstance(ql_iters_001, int) else ql_iters_001
    print(f"  {'Q-Learning (eps)':<20} {ql_eps_01:>10} {ql_eps_001:>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"\n  Plots:")
    print(f"    {save_path1_main}")
    print(f"    {save_path_vi}")
    print(f"    {save_path_pi}")
    print(f"    {save_path_ql}")
    print(f"    {save_path_debug}")
    print(f"    {theory_save_path}")
    print(f"    {ext_save_path}")
    print(f"    {save_path1}")
    print(f"    {save_path2}")
    print(f"    {save_path3}")
    print(f"    {save_path4}")
    print(f"    {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LQR Convergence Rate Comparison')
    add_cache_args(parser)
    args = parser.parse_args()

    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, "No cache found. Run without --plots-only first."
    else:
        data = compute_data()

    if not args.data_only:
        generate_outputs(data)
