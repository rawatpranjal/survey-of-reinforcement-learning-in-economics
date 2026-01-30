"""
RLHF vs DP: Gridworld Preference Learning
Chapter 7 - RLHF & Preference Learning

Compares Dynamic Programming (known rewards) vs RLHF (learns rewards from trajectory
preferences) on a gridworld navigation task. Demonstrates that RLHF can recover
reward functions from pairwise trajectory comparisons that enable policy learning
comparable to DP with known rewards.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# =============================================================================
# Configuration
# =============================================================================

np.random.seed(42)

# Gridworld parameters
N = 8  # Grid size (N x N)
GAMMA = 0.99  # Discount factor

# True reward parameters for gridworld
# 3-parameter model with state-dependent reward:
# r(s, a; phi) = phi[0] + phi[1] * distance_reduction + phi[2] * terminal_indicator
#
# phi[0]: step cost baseline (negative = penalty for each step)
# phi[1]: distance reduction bonus (positive = reward for moving closer)
# phi[2]: terminal reward (anchor for identification)
#
# The distance reduction is (dist_current - dist_next) / max_dist, so it's:
# - Positive when moving towards goal
# - Negative when moving away
# - Zero when staying same distance
PHI_TRUE = np.array([-0.1, 0.5, 10.0])

# Experiment parameters
N_SEEDS = 20
COMPARISON_COUNTS = [50, 100, 200, 500, 1000, 2000]
TRAJ_LENGTH = 30  # Long enough for most starts to reach goal

# Output
OUTPUT_DIR = 'ch07_rlhf/sims'
FIGURE_DPI = 300


# =============================================================================
# Gridworld Environment
# =============================================================================

class GridworldEnv:
    """NxN deterministic gridworld for RLHF comparison.

    State: (row, col) tuple
    Actions: 0=Left, 1=Right, 2=Up, 3=Down, 4=Stay
    Terminal: (N-1, N-1)
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    ACTION_NAMES = ['Left', 'Right', 'Up', 'Down', 'Stay']

    def __init__(self, grid_size):
        self.N = grid_size
        self.terminal = (grid_size - 1, grid_size - 1)
        self.initial = (0, 0)
        self.state = None
        self.reset()

    @property
    def num_states(self):
        return self.N * self.N

    @property
    def num_actions(self):
        return len(self.ACTIONS)

    def reset(self):
        self.state = self.initial
        return self.state

    def step(self, action):
        if self.state == self.terminal:
            return self.state, 0.0, True

        dr, dc = self.ACTIONS[action]
        r, c = self.state
        nr = max(0, min(self.N - 1, r + dr))
        nc = max(0, min(self.N - 1, c + dc))
        self.state = (nr, nc)

        done = self.state == self.terminal
        return self.state, 0.0, done  # Reward computed separately

    def state_to_index(self, state):
        return state[0] * self.N + state[1]

    def index_to_state(self, idx):
        return (idx // self.N, idx % self.N)

    def get_next_state(self, state, action):
        """Return next state without modifying environment."""
        dr, dc = self.ACTIONS[action]
        nr = max(0, min(self.N - 1, state[0] + dr))
        nc = max(0, min(self.N - 1, state[1] + dc))
        return (nr, nc)

    def get_all_states(self):
        return [self.index_to_state(i) for i in range(self.num_states)]


# =============================================================================
# Parametric Reward Function
# =============================================================================

def parametric_reward(s, a, phi, env):
    """Compute reward r(s, a; phi) for gridworld.

    3-parameter model with distance-based shaping:
    phi[0]: step cost baseline (negative = penalty per step)
    phi[1]: distance reduction bonus (positive = reward for progress)
    phi[2]: terminal reward (anchor for identification)
    """
    goal = env.terminal
    next_s = env.get_next_state(s, a)

    # Terminal transition
    if next_s == goal:
        return phi[2]

    # Compute distance reduction
    dist_current = abs(s[0] - goal[0]) + abs(s[1] - goal[1])
    dist_next = abs(next_s[0] - goal[0]) + abs(next_s[1] - goal[1])
    max_dist = 2 * (env.N - 1)

    # Progress towards goal (positive when moving closer)
    progress = (dist_current - dist_next) / max_dist

    return phi[0] + phi[1] * progress


def compute_all_rewards(phi, env):
    """Compute reward matrix R[state_idx, action] for all state-action pairs."""
    n_states = env.num_states
    n_actions = env.num_actions
    R = np.zeros((n_states, n_actions))

    for si in range(n_states):
        s = env.index_to_state(si)
        for a in range(n_actions):
            R[si, a] = parametric_reward(s, a, phi, env)

    return R


def verify_reward_structure(phi, env):
    """Print reward values at key states to verify structure."""
    print(f"Reward verification for phi = {phi}")
    states_to_check = [(0, 0), (3, 3), (6, 6), (7, 6), (6, 7)]
    for s in states_to_check:
        if s[0] < env.N and s[1] < env.N:
            for a in range(env.num_actions):
                r = parametric_reward(s, a, phi, env)
                ns = env.get_next_state(s, a)
                print(f"  s={s}, a={a} ({env.ACTION_NAMES[a]:5s}), next={ns}, r={r:.3f}")


# =============================================================================
# Value Iteration with Parametric Rewards
# =============================================================================

def solve_bellman_parametric(env, phi, gamma, tol=1e-8, max_iter=1000):
    """Solve Bellman equation using value iteration with parametric rewards.

    Returns:
        V: Value function array (n_states,)
        policy: Optimal policy array (n_states,)
    """
    n_states = env.num_states
    n_actions = env.num_actions

    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    for iteration in range(max_iter):
        V_new = np.zeros(n_states)

        for si in range(n_states):
            s = env.index_to_state(si)

            if s == env.terminal:
                V_new[si] = 0.0
                continue

            best_val = -np.inf
            best_a = 0

            for a in range(n_actions):
                r = parametric_reward(s, a, phi, env)
                ns = env.get_next_state(s, a)
                nsi = env.state_to_index(ns)
                val = r + gamma * V[nsi]

                if val > best_val:
                    best_val = val
                    best_a = a

            V_new[si] = best_val
            policy[si] = best_a

        diff = np.max(np.abs(V_new - V))
        V = V_new

        if diff < tol:
            break

    return V, policy


# =============================================================================
# Trajectory Generation
# =============================================================================

def generate_trajectory(env, policy, length, start=None):
    """Generate a trajectory following a given policy.

    Args:
        env: GridworldEnv instance
        policy: Either numpy array (state_idx -> action) or string ('random', 'greedy')
        length: Maximum trajectory length
        start: Optional starting state tuple

    Returns:
        states: List of state tuples
        actions: List of actions taken
    """
    if start is None:
        start = (np.random.randint(0, env.N), np.random.randint(0, env.N))

    states = []
    actions = []
    s = start

    for _ in range(length):
        if s == env.terminal:
            break

        states.append(s)
        si = env.state_to_index(s)

        if isinstance(policy, str):
            if policy == 'random':
                a = np.random.randint(env.num_actions)
            elif policy == 'greedy':
                # Greedy towards goal
                goal = env.terminal
                best_a = 4  # Stay
                best_dist = abs(s[0] - goal[0]) + abs(s[1] - goal[1])
                for a_cand in range(env.num_actions):
                    ns = env.get_next_state(s, a_cand)
                    d = abs(ns[0] - goal[0]) + abs(ns[1] - goal[1])
                    if d < best_dist:
                        best_dist = d
                        best_a = a_cand
                a = best_a
            else:
                a = np.random.randint(env.num_actions)
        else:
            a = policy[si]

        actions.append(a)
        s = env.get_next_state(s, a)

    return states, actions


def trajectory_total_reward(states, actions, phi, env):
    """Compute total reward for a trajectory."""
    total = 0.0
    for s, a in zip(states, actions):
        total += parametric_reward(s, a, phi, env)
    return total


def generate_comparisons(env, phi_true, n_comparisons, traj_length):
    """Generate pairwise trajectory comparisons.

    Uses diverse generation strategies that ensure terminal reward is observed:
    1. Optimal policy vs random (both from same start)
    2. Greedy vs random (both from same start)
    3. Short path (optimal from close start) vs long path (optimal from far start)
    4. Two random trajectories from same start

    Oracle preference: Higher total reward wins.

    Returns:
        comparisons: List of ((winner_states, winner_actions), (loser_states, loser_actions))
    """
    # Pre-compute optimal policy for generating comparisons
    _, policy_opt = solve_bellman_parametric(env, phi_true, GAMMA)

    comparisons = []

    for i in range(n_comparisons):
        strategy = i % 4

        if strategy == 0:
            # Strategy 0: Optimal vs random from same start
            # Start from middle region so optimal can reach goal
            row = np.random.randint(env.N // 4, 3 * env.N // 4)
            col = np.random.randint(env.N // 4, 3 * env.N // 4)
            start = (row, col)
            states1, actions1 = generate_trajectory(env, policy_opt, traj_length, start)
            states2, actions2 = generate_trajectory(env, 'random', traj_length, start)

        elif strategy == 1:
            # Strategy 1: Greedy vs random from same start
            row = np.random.randint(env.N // 4, 3 * env.N // 4)
            col = np.random.randint(env.N // 4, 3 * env.N // 4)
            start = (row, col)
            states1, actions1 = generate_trajectory(env, 'greedy', traj_length, start)
            states2, actions2 = generate_trajectory(env, 'random', traj_length, start)

        elif strategy == 2:
            # Strategy 2: Short path vs long path (both optimal, different starts)
            # Trajectory 1: start close to goal
            start1 = (env.N - 2, env.N - 2)
            states1, actions1 = generate_trajectory(env, policy_opt, traj_length, start1)
            # Trajectory 2: start far from goal
            start2 = (0, 0)
            states2, actions2 = generate_trajectory(env, policy_opt, traj_length, start2)

        else:
            # Strategy 3: Two random trajectories from same start (close to goal)
            row = np.random.randint(env.N // 2, env.N)
            col = np.random.randint(env.N // 2, env.N)
            start = (row, col)
            states1, actions1 = generate_trajectory(env, 'random', traj_length, start)
            states2, actions2 = generate_trajectory(env, 'random', traj_length, start)

        # Skip if either trajectory is empty (started at terminal)
        if len(states1) == 0 or len(states2) == 0:
            # Generate replacement comparison
            start = (0, 0)
            states1, actions1 = generate_trajectory(env, 'random', traj_length, start)
            states2, actions2 = generate_trajectory(env, 'random', traj_length, start)

        # Compute total rewards
        r1 = trajectory_total_reward(states1, actions1, phi_true, env)
        r2 = trajectory_total_reward(states2, actions2, phi_true, env)

        # Oracle preference: higher reward wins
        if r1 >= r2:
            comparisons.append(((states1, actions1), (states2, actions2)))
        else:
            comparisons.append(((states2, actions2), (states1, actions1)))

    return comparisons


# =============================================================================
# Bradley-Terry Estimation
# =============================================================================

def bradley_terry_loss(phi, comparisons, env):
    """Bradley-Terry negative log-likelihood for trajectory comparisons.

    phi: 3 parameters [phi[0], phi[1], phi[2]]
    Scale is NOT identified - all parameters estimated freely, normalized post-hoc.
    """
    loss = 0.0
    for (winner_s, winner_a), (loser_s, loser_a) in comparisons:
        r_winner = trajectory_total_reward(winner_s, winner_a, phi, env)
        r_loser = trajectory_total_reward(loser_s, loser_a, phi, env)

        diff = r_winner - r_loser
        # -log(sigmoid(diff)) = -diff + log(1 + exp(diff))
        loss += -diff + np.log(1 + np.exp(np.clip(diff, -500, 500)))

    return loss / len(comparisons)


def estimate_rlhf(comparisons, env):
    """Estimate reward parameters via Bradley-Terry MLE.

    All 3 parameters estimated freely. Scale is NOT identified by Bradley-Terry,
    so post-hoc normalization is needed using a known anchor.

    Returns:
        phi: Parameter vector (3,)
        result: Optimization result object
    """
    phi_init = np.array([-0.05, 0.3, 5.0])  # Initial guess

    result = minimize(
        bradley_terry_loss,
        phi_init,
        args=(comparisons, env),
        method='L-BFGS-B',
        bounds=[(-2.0, 0.5), (-2.0, 5.0), (1.0, 50.0)],
        options={'maxiter': 2000, 'ftol': 1e-12}
    )

    phi = result.x.copy()
    return phi, result


def normalize_phi(phi, anchor=10.0):
    """Normalize phi so that phi[2] = anchor.

    Bradley-Terry identifies parameters only up to affine transformation.
    Anchoring on known terminal reward fixes the scale.
    """
    if phi[2] > 0:
        scale = anchor / phi[2]
        return phi * scale
    return phi.copy()


# =============================================================================
# Evaluation Metrics
# =============================================================================

def parameter_mse(phi_est, phi_true):
    """Compute MSE between estimated and true parameters."""
    return np.mean((phi_est - phi_true)**2)


def policy_distance(policy1, policy2):
    """Compute fraction of states where policies disagree."""
    return np.mean(policy1 != policy2)


def evaluate_policy(env, policy, phi, gamma, n_episodes=100, max_steps=100):
    """Evaluate mean return of a policy under given reward parameters."""
    returns = []

    for _ in range(n_episodes):
        s = env.reset()
        total_return = 0.0
        discount = 1.0

        for _ in range(max_steps):
            if s == env.terminal:
                break

            si = env.state_to_index(s)
            a = policy[si]
            r = parametric_reward(s, a, phi, env)
            total_return += discount * r
            discount *= gamma

            s = env.get_next_state(s, a)

        returns.append(total_return)

    return np.mean(returns), np.std(returns)


def steps_to_goal(env, policy, start=None, max_steps=100):
    """Count steps to reach goal from start."""
    if start is None:
        start = env.initial

    s = start
    for step in range(max_steps):
        if s == env.terminal:
            return step

        si = env.state_to_index(s)
        a = policy[si]
        s = env.get_next_state(s, a)

    return max_steps  # Did not reach goal


# =============================================================================
# Experiment 1: Reward Recovery vs Sample Size
# =============================================================================

def run_experiment_1(env):
    """Experiment 1: Reward parameter recovery vs number of comparisons."""
    print("=" * 70)
    print("EXPERIMENT 1: REWARD RECOVERY VS SAMPLE SIZE")
    print("=" * 70)
    print()
    print(f"True parameters: phi = {PHI_TRUE.tolist()}")
    print(f"Identification: phi[2] (terminal) anchored at {PHI_TRUE[2]}")
    print()

    results = {M: [] for M in COMPARISON_COUNTS}

    for M in COMPARISON_COUNTS:
        print(f"--- M = {M} comparisons ---")

        # Header for per-seed output
        header = f"{'Seed':>4} | {'phi_est':>28} | {'MSE':>8} | {'Rho':>6} | {'Conv':>4}"
        print(header)
        print("-" * len(header))

        for seed in range(N_SEEDS):
            np.random.seed(seed)

            # Generate comparisons
            comparisons = generate_comparisons(env, PHI_TRUE, M, TRAJ_LENGTH)

            # Estimate parameters
            phi_raw, opt_result = estimate_rlhf(comparisons, env)
            phi_est = normalize_phi(phi_raw, anchor=PHI_TRUE[2])

            # Compute metrics
            mse = parameter_mse(phi_est, PHI_TRUE)
            rho, _ = spearmanr(phi_est, PHI_TRUE)

            result = {
                'seed': seed,
                'M': M,
                'phi_est': phi_est.copy(),
                'mse': mse,
                'rho': rho,
                'converged': opt_result.success,
                'loss': opt_result.fun,
            }
            results[M].append(result)

            # Print row
            phi_str = ', '.join([f'{p:.3f}' for p in phi_est])
            conv_str = 'Y' if opt_result.success else 'N'
            print(f"{seed:>4} | [{phi_str:>26}] | {mse:>8.5f} | {rho:>6.3f} | {conv_str:>4}")

        # Summary for this M
        mses = [r['mse'] for r in results[M]]
        rhos = [r['rho'] for r in results[M]]
        n_conv = sum(r['converged'] for r in results[M])

        print()
        print(f"Summary (M={M}):")
        print(f"  MSE: mean={np.mean(mses):.5f}, SE={np.std(mses)/np.sqrt(N_SEEDS):.5f}")
        print(f"  Spearman rho: mean={np.mean(rhos):.4f}, SE={np.std(rhos)/np.sqrt(N_SEEDS):.4f}")
        print(f"  Converged: {n_conv}/{N_SEEDS}")
        print()

    # Summary table
    print("=" * 70)
    print("REWARD RECOVERY SUMMARY")
    print("=" * 70)
    print(f"{'M':>6} | {'MSE (mean +/- SE)':>20} | {'Spearman (mean +/- SE)':>22} | {'Conv':>6}")
    print("-" * 70)
    for M in COMPARISON_COUNTS:
        mses = [r['mse'] for r in results[M]]
        rhos = [r['rho'] for r in results[M]]
        n_conv = sum(r['converged'] for r in results[M])
        mse_str = f"{np.mean(mses):.5f} +/- {np.std(mses)/np.sqrt(N_SEEDS):.5f}"
        rho_str = f"{np.mean(rhos):.4f} +/- {np.std(rhos)/np.sqrt(N_SEEDS):.4f}"
        print(f"{M:>6} | {mse_str:>20} | {rho_str:>22} | {n_conv:>3}/{N_SEEDS}")
    print()

    return results


# =============================================================================
# Experiment 2: Policy Quality Comparison
# =============================================================================

def run_experiment_2(env):
    """Experiment 2: Compare DP (true rewards) vs RLHF (learned rewards)."""
    print("=" * 70)
    print("EXPERIMENT 2: POLICY QUALITY COMPARISON")
    print("=" * 70)
    print()
    print(f"Fixed M = 500 comparisons")
    print()

    M_FIXED = 500

    # DP with true parameters (ground truth)
    V_dp, policy_dp = solve_bellman_parametric(env, PHI_TRUE, GAMMA)

    results = {
        'dp': [],
        'rlhf': [],
    }

    print("Per-seed results:")
    header = f"{'Seed':>4} | {'Method':>8} | {'Policy Dist':>12} | {'Mean Return':>12} | {'Steps to Goal':>14}"
    print(header)
    print("-" * len(header))

    for seed in range(N_SEEDS):
        np.random.seed(seed)

        # DP is deterministic, same every seed
        mean_ret_dp, _ = evaluate_policy(env, policy_dp, PHI_TRUE, GAMMA)
        steps_dp = steps_to_goal(env, policy_dp)

        results['dp'].append({
            'seed': seed,
            'policy': policy_dp.copy(),
            'policy_dist': 0.0,  # DP is the reference
            'mean_return': mean_ret_dp,
            'steps_to_goal': steps_dp,
        })

        # RLHF
        comparisons = generate_comparisons(env, PHI_TRUE, M_FIXED, TRAJ_LENGTH)
        phi_raw, _ = estimate_rlhf(comparisons, env)
        phi_est = normalize_phi(phi_raw, anchor=PHI_TRUE[2])
        V_rlhf, policy_rlhf = solve_bellman_parametric(env, phi_est, GAMMA)

        # Evaluate under TRUE rewards
        mean_ret_rlhf, _ = evaluate_policy(env, policy_rlhf, PHI_TRUE, GAMMA)
        steps_rlhf = steps_to_goal(env, policy_rlhf)
        p_dist = policy_distance(policy_rlhf, policy_dp)

        results['rlhf'].append({
            'seed': seed,
            'policy': policy_rlhf.copy(),
            'phi_est': phi_est.copy(),
            'policy_dist': p_dist,
            'mean_return': mean_ret_rlhf,
            'steps_to_goal': steps_rlhf,
            'V': V_rlhf.copy(),
        })

        # Print
        print(f"{seed:>4} | {'DP':>8} | {0.0:>12.4f} | {mean_ret_dp:>12.2f} | {steps_dp:>14}")
        print(f"{seed:>4} | {'RLHF':>8} | {p_dist:>12.4f} | {mean_ret_rlhf:>12.2f} | {steps_rlhf:>14}")

    # Store DP value function for visualization
    results['V_dp'] = V_dp

    print()
    print("=" * 70)
    print("POLICY QUALITY SUMMARY")
    print("=" * 70)
    print(f"{'Method':>10} | {'Policy Dist':>18} | {'Mean Return':>18} | {'Steps to Goal':>18}")
    print("-" * 70)

    for method in ['dp', 'rlhf']:
        dists = [r['policy_dist'] for r in results[method]]
        rets = [r['mean_return'] for r in results[method]]
        steps = [r['steps_to_goal'] for r in results[method]]

        dist_str = f"{np.mean(dists):.4f} +/- {np.std(dists)/np.sqrt(N_SEEDS):.4f}"
        ret_str = f"{np.mean(rets):.2f} +/- {np.std(rets)/np.sqrt(N_SEEDS):.2f}"
        step_str = f"{np.mean(steps):.1f} +/- {np.std(steps)/np.sqrt(N_SEEDS):.1f}"

        print(f"{method.upper():>10} | {dist_str:>18} | {ret_str:>18} | {step_str:>18}")

    print()

    return results


# =============================================================================
# Figure Generation
# =============================================================================

def plot_reward_recovery(results):
    """Plot MSE vs number of comparisons."""
    fig, ax = plt.subplots(figsize=(8, 5))

    mse_means = []
    mse_ses = []

    for M in COMPARISON_COUNTS:
        mses = [r['mse'] for r in results[M]]
        mse_means.append(np.mean(mses))
        mse_ses.append(np.std(mses) / np.sqrt(N_SEEDS))

    ax.errorbar(COMPARISON_COUNTS, mse_means, yerr=mse_ses, marker='o',
                capsize=3, linewidth=2, markersize=6, label='RLHF (Bradley-Terry)')

    # Add 1/M reference line
    M_ref = np.array(COMPARISON_COUNTS)
    mse_ref = mse_means[2] * (COMPARISON_COUNTS[2] / M_ref)  # Scale to match at M=200
    ax.plot(M_ref, mse_ref, 'k--', alpha=0.5, linewidth=1, label=r'$O(1/M)$ reference')

    ax.set_xlabel('Number of Pairwise Comparisons (M)', fontsize=11)
    ax.set_ylabel('Parameter MSE', fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Reward Recovery: MSE vs Sample Size', fontsize=12)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/gridworld_rlhf_reward_recovery.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/gridworld_rlhf_reward_recovery.png")


def plot_policy_comparison(results, env):
    """Plot value function heatmaps for DP vs RLHF."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # DP value function
    V_dp = results['V_dp'].reshape(env.N, env.N)

    # Average RLHF value function across seeds
    V_rlhf_list = [r['V'] for r in results['rlhf']]
    V_rlhf_mean = np.mean(V_rlhf_list, axis=0).reshape(env.N, env.N)

    # Value difference
    V_diff = V_rlhf_mean - V_dp

    # Plot DP
    im0 = axes[0].imshow(V_dp, cmap='viridis', origin='lower')
    axes[0].set_title('DP Value Function (True Rewards)', fontsize=11)
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    axes[0].scatter([env.terminal[1]], [env.terminal[0]], c='red', s=100, marker='*', label='Goal')
    axes[0].legend(loc='upper left')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Plot RLHF
    im1 = axes[1].imshow(V_rlhf_mean, cmap='viridis', origin='lower')
    axes[1].set_title('RLHF Value Function (Learned Rewards)', fontsize=11)
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    axes[1].scatter([env.terminal[1]], [env.terminal[0]], c='red', s=100, marker='*')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # Plot difference
    vmax = max(abs(V_diff.min()), abs(V_diff.max()))
    im2 = axes[2].imshow(V_diff, cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Value Difference (RLHF - DP)', fontsize=11)
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    axes[2].scatter([env.terminal[1]], [env.terminal[0]], c='black', s=100, marker='*')
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/gridworld_rlhf_policy_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/gridworld_rlhf_policy_comparison.png")


def generate_latex_table(recovery_results, policy_results):
    """Generate LaTeX results table."""
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"$M$ & MSE & Spearman $\rho$ & Policy Distance & Mean Return & Steps to Goal \\",
        r"\midrule",
    ]

    # DP baseline
    dp_rets = [r['mean_return'] for r in policy_results['dp']]
    dp_steps = [r['steps_to_goal'] for r in policy_results['dp']]
    lines.append(f"DP (oracle) & -- & -- & 0.0 & {np.mean(dp_rets):.2f} & {np.mean(dp_steps):.0f} \\\\")
    lines.append(r"\midrule")

    # RLHF at different M
    for M in COMPARISON_COUNTS:
        mses = [r['mse'] for r in recovery_results[M]]
        rhos = [r['rho'] for r in recovery_results[M]]

        mse_str = f"{np.mean(mses):.4f}"
        rho_str = f"{np.mean(rhos):.3f}"

        # For policy metrics, only have M=500
        if M == 500:
            dists = [r['policy_dist'] for r in policy_results['rlhf']]
            rets = [r['mean_return'] for r in policy_results['rlhf']]
            steps = [r['steps_to_goal'] for r in policy_results['rlhf']]
            dist_str = f"{np.mean(dists):.3f}"
            ret_str = f"{np.mean(rets):.2f}"
            step_str = f"{np.mean(steps):.1f}"
        else:
            dist_str = "--"
            ret_str = "--"
            step_str = "--"

        lines.append(f"RLHF ($M$={M}) & {mse_str} & {rho_str} & {dist_str} & {ret_str} & {step_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    latex_table = '\n'.join(lines)

    with open(f'{OUTPUT_DIR}/gridworld_rlhf_results.tex', 'w') as f:
        f.write(latex_table)

    print(f"Saved: {OUTPUT_DIR}/gridworld_rlhf_results.tex")
    print()
    print("Table contents:")
    print(latex_table)
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("RLHF vs DP: GRIDWORLD PREFERENCE LEARNING")
    print("Chapter 7 - RLHF & Preference Learning")
    print("=" * 70)
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"NumPy version: {np.__version__}")
    print()
    print("Configuration:")
    print(f"  Grid size: {N}x{N} ({N*N} states)")
    print(f"  Discount factor: {GAMMA}")
    print(f"  True phi: {PHI_TRUE.tolist()}")
    print(f"  Trajectory length: {TRAJ_LENGTH}")
    print(f"  Comparison counts: {COMPARISON_COUNTS}")
    print(f"  Seeds: {N_SEEDS}")
    print()

    # Create environment
    env = GridworldEnv(N)

    # Run experiments
    recovery_results = run_experiment_1(env)
    policy_results = run_experiment_2(env)

    # Generate outputs
    print("=" * 70)
    print("GENERATING OUTPUT FILES")
    print("=" * 70)
    print()

    plot_reward_recovery(recovery_results)
    plot_policy_comparison(policy_results, env)
    generate_latex_table(recovery_results, policy_results)

    # Verification summary
    print("=" * 70)
    print("NUMERICAL VERIFICATION")
    print("=" * 70)
    print()

    # Check 1: MSE decreases with M
    mse_50 = np.mean([r['mse'] for r in recovery_results[50]])
    mse_2000 = np.mean([r['mse'] for r in recovery_results[2000]])
    print(f"1. MSE decreases with M:")
    print(f"   MSE(M=50)   = {mse_50:.5f}")
    print(f"   MSE(M=2000) = {mse_2000:.5f}")
    print(f"   Ratio: {mse_50/mse_2000:.1f}x improvement")
    print(f"   CHECK: {'PASS' if mse_50 > mse_2000 else 'FAIL'}")
    print()

    # Check 2: Spearman correlation >= 0.9 for M >= 500
    rhos_500 = [r['rho'] for r in recovery_results[500]]
    mean_rho = np.mean(rhos_500)
    print(f"2. Spearman correlation for M=500:")
    print(f"   Mean rho: {mean_rho:.4f}")
    print(f"   CHECK: {'PASS' if mean_rho >= 0.9 else 'FAIL'} (threshold: 0.9)")
    print()

    # Check 3: RLHF policy achieves similar return to DP
    dp_rets = [r['mean_return'] for r in policy_results['dp']]
    rlhf_rets = [r['mean_return'] for r in policy_results['rlhf']]
    dp_mean = np.mean(dp_rets)
    rlhf_mean = np.mean(rlhf_rets)
    ratio = rlhf_mean / dp_mean if dp_mean != 0 else 0
    print(f"3. Policy return comparison (M=500):")
    print(f"   DP mean return:   {dp_mean:.2f}")
    print(f"   RLHF mean return: {rlhf_mean:.2f}")
    print(f"   Ratio: {ratio:.3f}")
    print(f"   CHECK: {'PASS' if ratio >= 0.9 else 'FAIL'} (threshold: 0.9)")
    print()

    # Check 4: Policy distance
    dists = [r['policy_dist'] for r in policy_results['rlhf']]
    mean_dist = np.mean(dists)
    print(f"4. Policy distance from DP optimal (M=500):")
    print(f"   Mean distance: {mean_dist:.4f}")
    print(f"   CHECK: {'PASS' if mean_dist <= 0.2 else 'FAIL'} (threshold: 0.2)")
    print()

    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"  {OUTPUT_DIR}/gridworld_rlhf_reward_recovery.png")
    print(f"  {OUTPUT_DIR}/gridworld_rlhf_policy_comparison.png")
    print(f"  {OUTPUT_DIR}/gridworld_rlhf_results.tex")
    print()


if __name__ == '__main__':
    main()
