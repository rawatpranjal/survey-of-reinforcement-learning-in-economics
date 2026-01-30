# Gridworld Navigation Benchmark: DP vs RL Methods
# Chapter 3 Benchmarks
# Compares Value Iteration and DQN on NxN gridworld navigation across multiple
# grid sizes, measuring wall-clock time and policy quality.

import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from econ_benchmark import (
    EconBenchmark, run_value_iteration, run_dqn,
    evaluate_dp_policy, evaluate_dqn_policy, evaluate_heuristic
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

SEEDS = [42, 123, 7]
GRID_SIZES = [5, 10, 20, 50]
GAMMA = 0.95
STEP_PENALTY = -0.1
TERMINAL_REWARD = 10.0

OUTPUT_DIR = Path(__file__).resolve().parent

# Actions: Left, Right, Up, Down, Stay
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
NUM_ACTIONS = len(ACTIONS)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Gridworld(EconBenchmark):
    """NxN deterministic gridworld. Terminal state at (N-1, N-1) with reward
    +TERMINAL_REWARD. All other transitions yield STEP_PENALTY."""

    complexity_param = 0  # Will be set per instance (N)
    dp_feasible = True

    def __init__(self, N):
        self.N = N
        self.terminal = (N - 1, N - 1)
        self._num_states = N * N
        self._num_actions = NUM_ACTIONS
        self.state = None
        self.reset()

    def reset(self):
        """Reset environment and return initial state tuple."""
        self.state = (0, 0)
        return self.state

    def step(self, action):
        """Take action, return (next_state, reward, done)."""
        if self.state == self.terminal:
            return self.state, 0.0, True

        dr, dc = ACTIONS[action]
        r, c = self.state
        nr = max(0, min(self.N - 1, r + dr))
        nc = max(0, min(self.N - 1, c + dc))
        self.state = (nr, nc)

        if self.state == self.terminal:
            return self.state, TERMINAL_REWARD, True
        return self.state, STEP_PENALTY, False

    @property
    def num_states(self):
        """Total number of states."""
        return self._num_states

    @property
    def num_actions(self):
        """Total number of actions."""
        return self._num_actions

    def state_to_index(self, state):
        """Map state tuple to integer index."""
        return state[0] * self.N + state[1]

    def index_to_state(self, idx):
        """Map integer index to state tuple."""
        return (idx // self.N, idx % self.N)

    def state_to_features(self, state):
        """Convert state tuple to normalized float array for neural network input."""
        return np.array([state[0] / max(self.N - 1, 1), state[1] / max(self.N - 1, 1)], dtype=np.float32)

    def transition_distribution(self, state, action):
        """Return list of (next_state, probability) pairs for DP."""
        dr, dc = ACTIONS[action]
        nr = max(0, min(self.N - 1, state[0] + dr))
        nc = max(0, min(self.N - 1, state[1] + dc))
        next_state = (nr, nc)
        return [(next_state, 1.0)]  # Deterministic transition

    def expected_reward(self, state, action):
        """Return expected immediate reward E[r | s, a] for DP."""
        if state == self.terminal:
            return 0.0
        dr, dc = ACTIONS[action]
        nr = max(0, min(self.N - 1, state[0] + dr))
        nc = max(0, min(self.N - 1, state[1] + dc))
        next_state = (nr, nc)
        if next_state == self.terminal:
            return TERMINAL_REWARD
        return STEP_PENALTY


# ---------------------------------------------------------------------------
# Heuristic: Greedy Manhattan
# ---------------------------------------------------------------------------
def heuristic_manhattan(state):
    """Greedy policy: move toward goal (N-1, N-1) via Manhattan distance."""
    # Get goal distance for each action
    goal = (4, 4)  # Placeholder, will be overridden in evaluation
    best_a = 0
    best_dist = float('inf')
    for a, (dr, dc) in enumerate(ACTIONS):
        nr, nc = state[0] + dr, state[1] + dc
        dist = abs(nr - goal[0]) + abs(nc - goal[1])
        if dist < best_dist:
            best_dist = dist
            best_a = a
    return best_a


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    results = {}

    for N in GRID_SIZES:
        print(f"\n{'='*60}")
        print(f"  Grid size N = {N}  ({N*N} states)")
        print(f"{'='*60}")
        env = Gridworld(N)

        entry = {'N': N, 'states': N * N}

        # --- Value Iteration (deterministic, single run) ---
        print(f"  Running Value Iteration...")
        V, policy, vi_time = run_value_iteration(env, gamma=GAMMA)
        entry['vi_time'] = vi_time
        vi_reward = evaluate_dp_policy(env, policy, gamma=GAMMA, n_episodes=200, horizon=N*4)
        entry['vi_reward'] = vi_reward
        print(f"    VI time={vi_time:8.3f}s  reward={vi_reward:.3f}")

        # --- DQN (multiple seeds) ---
        print(f"  Running DQN ({len(SEEDS)} seeds)...")
        dqn_times = []
        dqn_rewards = []

        # Scale episodes with grid size
        if N <= 10:
            num_episodes = 3000
        elif N <= 20:
            num_episodes = 6000
        else:
            num_episodes = 12000

        for seed in SEEDS:
            env_dqn = Gridworld(N)
            q_net, dqn_time = run_dqn(
                env_dqn, gamma=GAMMA, num_episodes=num_episodes,
                episode_horizon=N*4, seed=seed, replay_size=10_000,
                batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
                epsilon_decay_frac=0.6, target_update_freq=50
            )
            reward = evaluate_dqn_policy(env, q_net, n_episodes=200, horizon=N*4)
            dqn_times.append(dqn_time)
            dqn_rewards.append(reward)
            print(f"    seed={seed}: time={dqn_time:8.3f}s  reward={reward:.3f}")

        entry['dqn_time_mean'] = np.mean(dqn_times)
        entry['dqn_time_std'] = np.std(dqn_times)
        entry['dqn_reward_mean'] = np.mean(dqn_rewards)
        entry['dqn_reward_std'] = np.std(dqn_rewards)

        results[N] = entry

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figures(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    Ns = sorted(results.keys())
    vi_times = [results[n]['vi_time'] for n in Ns]
    dqn_times = [results[n]['dqn_time_mean'] for n in Ns]
    dqn_time_errs = [results[n]['dqn_time_std'] for n in Ns]

    vi_rewards = [results[n]['vi_reward'] for n in Ns]
    dqn_rewards = [results[n]['dqn_reward_mean'] for n in Ns]
    dqn_reward_errs = [results[n]['dqn_reward_std'] for n in Ns]

    # Left panel: wall-clock time (log scale)
    ax = axes[0]
    ax.plot(Ns, vi_times, 'o-', color='#1f77b4', label='Value Iteration', markersize=7, linewidth=2)
    ax.errorbar(Ns, dqn_times, yerr=dqn_time_errs, fmt='s-', color='#ff7f0e',
                label='DQN', capsize=3, markersize=7, linewidth=2)
    ax.set_xlabel('Grid size $N$', fontsize=12)
    ax.set_ylabel('Wall-clock time (s)', fontsize=12)
    ax.set_title('Computation Time vs Grid Size', fontsize=13)
    ax.set_yscale('log')
    ax.set_xticks(Ns)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # Right panel: reward
    ax = axes[1]
    ax.plot(Ns, vi_rewards, 'o-', color='#1f77b4', label='Value Iteration', markersize=7, linewidth=2)
    ax.errorbar(Ns, dqn_rewards, yerr=dqn_reward_errs, fmt='s-', color='#ff7f0e',
                label='DQN', capsize=3, markersize=7, linewidth=2)
    ax.set_xlabel('Grid size $N$', fontsize=12)
    ax.set_ylabel('Average episode reward', fontsize=12)
    ax.set_title('Policy Quality vs Grid Size', fontsize=13)
    ax.set_xticks(Ns)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / 'gridworld_scaling.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def make_latex_table(results):
    lines = []
    lines.append(r'\begin{tabular}{lrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Grid & States & VI Time (s) & VI Reward & DQN Time (s) & DQN Reward \\')
    lines.append(r'\midrule')

    for N in sorted(results.keys()):
        r = results[N]
        vi_time_str = f"${r['vi_time']:.3f}$"
        vi_reward_str = f"${r['vi_reward']:.2f}$"
        dqn_time_str = f"${r['dqn_time_mean']:.2f} \\pm {r['dqn_time_std']:.2f}$"
        dqn_reward_str = f"${r['dqn_reward_mean']:.2f} \\pm {r['dqn_reward_std']:.2f}$"
        lines.append(f'  ${N} \\times {N}$ & {r["states"]} & {vi_time_str} & {vi_reward_str} & {dqn_time_str} & {dqn_reward_str} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    out_path = OUTPUT_DIR / 'gridworld_results.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"LaTeX table saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    print("Gridworld Navigation Benchmark")
    print(f"Grid sizes: {GRID_SIZES}")
    print(f"Seeds for DQN: {SEEDS}")
    print(f"Gamma={GAMMA}, step penalty={STEP_PENALTY}, terminal reward={TERMINAL_REWARD}")

    results = run_benchmark()
    make_figures(results)
    make_latex_table(results)

    print("\nBenchmark complete.")
