# Bus Engine Replacement Benchmark: Learning Curves and Metrics
# Chapter 3 -- RL for Structural Estimation
# Benchmarks the multi-engine bus replacement problem (inspired by Rust 1987)
# comparing exact value iteration with DQN learning curves.

import itertools
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

MILEAGE_STATES = 6
ALPHA = 1.0
BETA = 5.0
GAMMA = 0.95
CAPACITY = 3

N = 4  # Number of engines for main experiment
SEEDS = [42, 123, 7, 456, 789, 101, 202, 303, 404, 505]

NUM_EPISODES = 10000
EPISODE_HORIZON = 20
EVAL_FREQ = 100
EVAL_EPISODES = 20

OUTPUT_DIR = Path(__file__).resolve().parent

# Figure styling
DP_COLOR = '#1f77b4'
DQN_COLOR = '#ff7f0e'
HEURISTIC_COLORS = ['#2ca02c', '#d62728', '#9467bd']


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class BusEngine(EconBenchmark):
    """Multi-engine bus replacement problem.

    State: tuple of N mileage levels, each in {0, ..., MILEAGE_STATES-1}
    Action: tuple of engine indices to replace (size <= CAPACITY)
    Reward: negative cost = -(ALPHA * operating + BETA * replacements)
    """

    dp_feasible = True

    def __init__(self, N):
        self.N = N
        self.complexity_param = N

        self._all_states = list(itertools.product(range(MILEAGE_STATES), repeat=N))
        self._state_to_idx = {s: i for i, s in enumerate(self._all_states)}

        self._all_actions = []
        engines = list(range(N))
        for k in range(min(CAPACITY, N) + 1):
            for combo in itertools.combinations(engines, k):
                self._all_actions.append(combo)

        self.state = None
        self.reset()

    def reset(self):
        self.state = tuple(np.random.randint(0, MILEAGE_STATES, size=self.N))
        return self.state

    def step(self, action):
        action_tuple = self._all_actions[action]
        cost = self._cost(self.state, action_tuple)
        self.state = self._transition(self.state, action_tuple)
        reward = -cost
        done = False
        return self.state, reward, done

    @property
    def num_states(self):
        return len(self._all_states)

    @property
    def num_actions(self):
        return len(self._all_actions)

    def state_to_index(self, state):
        return self._state_to_idx[state]

    def index_to_state(self, idx):
        return self._all_states[idx]

    def state_to_features(self, state):
        return np.array(state, dtype=np.float32) / (MILEAGE_STATES - 1)

    def transition_distribution(self, state, action):
        action_tuple = self._all_actions[action]
        next_state = self._transition(state, action_tuple)
        return [(next_state, 1.0)]

    def expected_reward(self, state, action):
        action_tuple = self._all_actions[action]
        cost = self._cost(state, action_tuple)
        return -cost

    def _cost(self, state, action_tuple):
        operating = sum(1 for m in state if m > 0)
        return ALPHA * operating + BETA * len(action_tuple)

    def _transition(self, state, action_tuple):
        replace_set = set(action_tuple)
        new_state = []
        for i, m in enumerate(state):
            if i in replace_set:
                new_state.append(0)
            else:
                new_state.append(min(MILEAGE_STATES - 1, m + 1))
        return tuple(new_state)


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def make_threshold_heuristic(env, threshold=3):
    """Replace engines with mileage >= threshold (up to CAPACITY)."""
    def heuristic(state):
        to_replace = tuple(sorted([i for i, m in enumerate(state) if m >= threshold])[:CAPACITY])
        for idx, action_tuple in enumerate(env._all_actions):
            if action_tuple == to_replace:
                return idx
        return 0
    return heuristic


def make_never_replace_heuristic(env):
    """Never replace any engine."""
    def heuristic(state):
        return 0
    return heuristic


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    env = BusEngine(N)
    results = {}

    print(f"\n{'='*60}")
    print(f"  Bus Engine Replacement (N={N}, {env.num_states} states, {env.num_actions} actions)")
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
        'entropy': 0.0,
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
        env_dqn = BusEngine(N)
        q_net, metrics = run_dqn(
            env_dqn, gamma=GAMMA, num_episodes=NUM_EPISODES,
            episode_horizon=EPISODE_HORIZON, seed=seed, replay_size=10_000,
            batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
            epsilon_decay_frac=0.6, target_update_freq=50,
            eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES
        )

        reward = evaluate_dqn_policy(env, q_net, n_episodes=200, horizon=EPISODE_HORIZON)
        dqn_rewards.append(reward)
        dqn_times.append(metrics.wall_time)
        dqn_transitions.append(metrics.total_transitions)
        dqn_gradient_updates.append(metrics.total_gradient_updates)
        dqn_coverage.append(state_coverage(metrics.states_visited, env.num_states))

        entropy = compute_policy_entropy(q_net, env, n_samples=500)
        dqn_entropy.append(entropy)
        agreement = compute_policy_agreement(q_net, policy, env, n_samples=500)
        dqn_agreement.append(agreement)

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

    # --- Heuristics ---
    print("  Running Heuristics...")
    threshold_h = make_threshold_heuristic(env, threshold=3)
    h_threshold_reward = evaluate_heuristic(env, threshold_h, n_episodes=200, horizon=EPISODE_HORIZON)
    print(f"    Threshold(3): reward={h_threshold_reward:.3f}")

    never_h = make_never_replace_heuristic(env)
    h_never_reward = evaluate_heuristic(env, never_h, n_episodes=200, horizon=EPISODE_HORIZON)
    print(f"    Never Replace: reward={h_never_reward:.3f}")

    results['heuristics'] = {
        'threshold': h_threshold_reward,
        'never': h_never_reward,
    }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figure(results):
    fig, ax = plt.subplots(figsize=(8, 5))

    curves = results['dqn']['curves']
    episodes = results['dqn']['checkpoint_episodes']

    min_len = min(len(c) for c in curves)
    curves_arr = np.array([c[:min_len] for c in curves])
    episodes = episodes[:min_len]

    mean_rewards = np.mean(curves_arr, axis=0)
    std_rewards = np.std(curves_arr, axis=0)

    ax.plot(episodes, mean_rewards, color=DQN_COLOR, linewidth=2, label='DQN')
    ax.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                    color=DQN_COLOR, alpha=0.2)

    dp_reward = results['dp']['reward']
    ax.axhline(dp_reward, color=DP_COLOR, linestyle='--', linewidth=1.5,
               label=f'DP Optimal ({dp_reward:.2f})')

    h_threshold = results['heuristics']['threshold']
    ax.axhline(h_threshold, color=HEURISTIC_COLORS[0], linestyle=':', linewidth=1.5,
               label=f'Threshold Heuristic ({h_threshold:.2f})')

    h_never = results['heuristics']['never']
    ax.axhline(h_never, color=HEURISTIC_COLORS[1], linestyle='-.', linewidth=1.5,
               label=f'Never Replace ({h_never:.2f})')

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Episode Reward', fontsize=11)
    ax.set_title(f'Bus Engine Replacement (N={N}) Learning Curve', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    out_path = OUTPUT_DIR / 'bus_engine_learning_curve.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def make_latex_table(results):
    dp = results['dp']
    dqn = results['dqn']
    h = results['heuristics']

    lines = []
    lines.append(r'\begin{tabular}{lrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Method & Reward & Time (s) & Convergence & Transitions & Entropy & DP Agr. \\')
    lines.append(r'\midrule')

    lines.append(
        f"DP (VI) & ${dp['reward']:.2f}$ & ${dp['time']:.2f}$ & "
        f"{dp['iterations']} iter & --- & ${dp['entropy']:.2f}$ & 100\\% \\\\"
    )

    lines.append(
        f"DQN & ${dqn['reward_mean']:.2f} \\pm {dqn['reward_std']:.2f}$ & "
        f"${dqn['time_mean']:.1f}$ & "
        f"{int(dqn['convergence_mean'])} ep & "
        f"{int(dqn['transitions_mean']/1000)}k & "
        f"${dqn['entropy_mean']:.2f}$ & "
        f"{dqn['agreement_mean']:.1%} \\\\"
    )

    lines.append(
        f"Threshold(3) & ${h['threshold']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(
        f"Never Replace & ${h['never']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    out_path = OUTPUT_DIR / 'bus_engine_results.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"LaTeX table saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    print("Bus Engine Replacement Benchmark")
    print(f"N={N} engines, {MILEAGE_STATES} mileage states")
    print(f"Seeds: {SEEDS}")
    print(f"Alpha={ALPHA}, Beta={BETA}, Gamma={GAMMA}")

    results = run_benchmark()
    make_figure(results)
    make_latex_table(results)

    print("\nBenchmark complete.")
