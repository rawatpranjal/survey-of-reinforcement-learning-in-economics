# Gridworld Navigation Benchmark: DP vs DQN Learning Curves
# Chapter 3 Benchmarks
# Compares Value Iteration and DQN on NxN gridworld, showing DQN learning curves
# with convergence to DP optimal value.

import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from econ_benchmark import (
    EconBenchmark, run_value_iteration, run_dqn,
    evaluate_dp_policy, evaluate_dqn_policy, evaluate_heuristic,
    compute_policy_entropy, compute_policy_agreement, state_coverage
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

SEEDS = [42, 123, 7, 456, 789, 101, 202, 303, 404, 505]
N = 10  # Grid size for main experiment
GAMMA = 0.95
STEP_PENALTY = -0.1
TERMINAL_REWARD = 10.0

NUM_EPISODES = 3000
EPISODE_HORIZON = 40
EVAL_FREQ = 50
EVAL_EPISODES = 20

OUTPUT_DIR = Path(__file__).resolve().parent

# Actions: Left, Right, Up, Down, Stay
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
NUM_ACTIONS = len(ACTIONS)

# Figure styling
DP_COLOR = '#1f77b4'
DQN_COLOR = '#ff7f0e'
HEURISTIC_COLORS = ['#2ca02c', '#d62728', '#9467bd']


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Gridworld(EconBenchmark):
    """NxN deterministic gridworld. Terminal state at (N-1, N-1) with reward
    +TERMINAL_REWARD. All other transitions yield STEP_PENALTY."""

    complexity_param = 0
    dp_feasible = True

    def __init__(self, N):
        self.N = N
        self.terminal = (N - 1, N - 1)
        self._num_states = N * N
        self._num_actions = NUM_ACTIONS
        self.state = None
        self.reset()

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
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
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    def state_to_index(self, state):
        return state[0] * self.N + state[1]

    def index_to_state(self, idx):
        return (idx // self.N, idx % self.N)

    def state_to_features(self, state):
        return np.array([state[0] / max(self.N - 1, 1),
                         state[1] / max(self.N - 1, 1)], dtype=np.float32)

    def transition_distribution(self, state, action):
        dr, dc = ACTIONS[action]
        nr = max(0, min(self.N - 1, state[0] + dr))
        nc = max(0, min(self.N - 1, state[1] + dc))
        next_state = (nr, nc)
        return [(next_state, 1.0)]

    def expected_reward(self, state, action):
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
def make_heuristic_manhattan(N):
    """Create greedy policy that moves toward goal (N-1, N-1)."""
    goal = (N - 1, N - 1)

    def heuristic(state):
        best_a = 0
        best_dist = float('inf')
        for a, (dr, dc) in enumerate(ACTIONS):
            nr = max(0, min(N - 1, state[0] + dr))
            nc = max(0, min(N - 1, state[1] + dc))
            dist = abs(nr - goal[0]) + abs(nc - goal[1])
            if dist < best_dist:
                best_dist = dist
                best_a = a
        return best_a

    return heuristic


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    env = Gridworld(N)
    results = {}

    print(f"\n{'='*60}")
    print(f"  Gridworld {N}x{N} ({env.num_states} states, {env.num_actions} actions)")
    print(f"{'='*60}")

    # --- Value Iteration ---
    print("  Running Value Iteration...")
    V, policy, vi_metrics = run_value_iteration(env, gamma=GAMMA)
    dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                   n_episodes=200, horizon=EPISODE_HORIZON)
    print(f"    VI: {vi_metrics.iterations} iterations, "
          f"residual={vi_metrics.final_residual:.2e}, "
          f"time={vi_metrics.wall_time:.3f}s, reward={dp_reward:.3f}")

    results['dp'] = {
        'reward': dp_reward,
        'iterations': vi_metrics.iterations,
        'residual': vi_metrics.final_residual,
        'time': vi_metrics.wall_time,
        'entropy': 0.0,  # Deterministic policy
    }

    # --- DQN (multiple seeds) ---
    print(f"  Running DQN ({len(SEEDS)} seeds)...")
    all_curves = []
    dqn_rewards = []
    dqn_times = []
    dqn_transitions = []
    dqn_gradient_updates = []
    dqn_coverage = []
    dqn_entropy = []
    dqn_agreement = []
    convergence_episodes = []

    for seed in SEEDS:
        env_dqn = Gridworld(N)
        q_net, metrics = run_dqn(
            env_dqn, gamma=GAMMA, num_episodes=NUM_EPISODES,
            episode_horizon=EPISODE_HORIZON, seed=seed, replay_size=10_000,
            batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
            epsilon_decay_frac=0.6, target_update_freq=50,
            eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES
        )

        # Evaluate final policy
        reward = evaluate_dqn_policy(env, q_net, n_episodes=200, horizon=EPISODE_HORIZON)
        dqn_rewards.append(reward)
        dqn_times.append(metrics.wall_time)
        dqn_transitions.append(metrics.total_transitions)
        dqn_gradient_updates.append(metrics.total_gradient_updates)
        dqn_coverage.append(state_coverage(metrics.states_visited, env.num_states))

        # Compute policy metrics
        entropy = compute_policy_entropy(q_net, env, n_samples=500)
        dqn_entropy.append(entropy)
        agreement = compute_policy_agreement(q_net, policy, env, n_samples=500)
        dqn_agreement.append(agreement)

        # Find convergence episode (95% of final eval)
        if len(metrics.eval_checkpoints) > 0:
            final_perf = metrics.eval_checkpoints[-1]
            threshold = 0.95 * final_perf
            conv_ep = None
            for i, r in enumerate(metrics.eval_checkpoints):
                if r >= threshold:
                    conv_ep = metrics.checkpoint_episodes[i]
                    break
            convergence_episodes.append(conv_ep if conv_ep else NUM_EPISODES)
        else:
            convergence_episodes.append(NUM_EPISODES)

        all_curves.append(metrics.eval_checkpoints)
        print(f"    seed={seed}: reward={reward:.3f}, time={metrics.wall_time:.2f}s, "
              f"agreement={agreement:.1%}")

    # Store DQN results
    results['dqn'] = {
        'reward_mean': np.mean(dqn_rewards),
        'reward_std': np.std(dqn_rewards),
        'time_mean': np.mean(dqn_times),
        'convergence_mean': np.mean(convergence_episodes),
        'transitions_mean': np.mean(dqn_transitions),
        'gradient_updates_mean': np.mean(dqn_gradient_updates),
        'coverage_mean': np.mean(dqn_coverage),
        'entropy_mean': np.mean(dqn_entropy),
        'agreement_mean': np.mean(dqn_agreement),
        'agreement_std': np.std(dqn_agreement),
        'curves': all_curves,
        'checkpoint_episodes': metrics.checkpoint_episodes,
    }

    # --- Heuristic (Greedy Manhattan) ---
    print("  Running Heuristic (Greedy Manhattan)...")
    heuristic = make_heuristic_manhattan(N)
    h_reward = evaluate_heuristic(env, heuristic, n_episodes=200, horizon=EPISODE_HORIZON)
    print(f"    Heuristic: reward={h_reward:.3f}")

    results['heuristic'] = {
        'reward': h_reward,
    }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figure(results):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot DQN learning curves
    curves = results['dqn']['curves']
    episodes = results['dqn']['checkpoint_episodes']

    # Ensure all curves have same length
    min_len = min(len(c) for c in curves)
    curves_arr = np.array([c[:min_len] for c in curves])
    episodes = episodes[:min_len]

    mean_rewards = np.mean(curves_arr, axis=0)
    std_rewards = np.std(curves_arr, axis=0)

    ax.plot(episodes, mean_rewards, color=DQN_COLOR, linewidth=2, label='DQN')
    ax.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                    color=DQN_COLOR, alpha=0.2)

    # DP reference line
    dp_reward = results['dp']['reward']
    ax.axhline(dp_reward, color=DP_COLOR, linestyle='--', linewidth=1.5,
               label=f'DP Optimal ({dp_reward:.2f})')

    # Heuristic reference line
    h_reward = results['heuristic']['reward']
    ax.axhline(h_reward, color=HEURISTIC_COLORS[0], linestyle=':', linewidth=1.5,
               label=f'Greedy Heuristic ({h_reward:.2f})')

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Episode Reward', fontsize=11)
    ax.set_title(f'Gridworld {N}x{N} Learning Curve', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    out_path = OUTPUT_DIR / 'gridworld_learning_curve.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def make_latex_table(results):
    dp = results['dp']
    dqn = results['dqn']
    h = results['heuristic']

    lines = []
    lines.append(r'\begin{tabular}{lrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Method & Reward & Time (s) & Convergence & Transitions & Entropy & DP Agr. \\')
    lines.append(r'\midrule')

    # DP row
    lines.append(
        f"DP (VI) & ${dp['reward']:.2f}$ & ${dp['time']:.2f}$ & "
        f"{dp['iterations']} iter & --- & ${dp['entropy']:.2f}$ & 100\\% \\\\"
    )

    # DQN row
    lines.append(
        f"DQN & ${dqn['reward_mean']:.2f} \\pm {dqn['reward_std']:.2f}$ & "
        f"${dqn['time_mean']:.1f}$ & "
        f"{int(dqn['convergence_mean'])} ep & "
        f"{int(dqn['transitions_mean']/1000)}k & "
        f"${dqn['entropy_mean']:.2f}$ & "
        f"{dqn['agreement_mean']:.1%} \\\\"
    )

    # Heuristic row
    lines.append(
        f"Greedy & ${h['reward']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

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
    print(f"Grid size: {N}x{N}")
    print(f"Seeds for DQN: {SEEDS}")
    print(f"Gamma={GAMMA}, step penalty={STEP_PENALTY}, terminal reward={TERMINAL_REWARD}")

    results = run_benchmark()
    make_figure(results)
    make_latex_table(results)

    print("\nBenchmark complete.")
