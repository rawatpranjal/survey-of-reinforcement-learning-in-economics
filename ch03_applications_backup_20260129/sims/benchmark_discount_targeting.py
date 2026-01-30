"""
Dynamic Discount Targeting Benchmark
Chapter 3 Benchmarks -- RL in Structural Estimation

Implements a ride-sharing coupon optimization environment (Uber/Lyft-style)
and benchmarks DQN learning curves against exact DP and heuristic baselines.
"""

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

GAMMA = 0.95
EPISODE_LENGTH = 50
BASE_REVENUE = 20.0
DISCOUNT_LEVELS = [0.0, 0.10, 0.20, 0.30]
N_ACTIONS = len(DISCOUNT_LEVELS)

RECENCY_BINS = 30
FREQUENCY_BINS = 20
DISC_HIST_LEVELS = 4
H_MAX = 5

BASE_RATE = -0.5
DISCOUNT_EFFECT = 4.0
RECENCY_EFFECT = -0.04
HABITUATION_EFFECT = 0.10

N_STATE = 3  # State complexity for main experiment
SEEDS = [42, 123, 7, 456, 789, 101, 202, 303, 404, 505]
NUM_EPISODES = 5000
EVAL_FREQ = 50
EVAL_EPISODES = 20

OUTPUT_DIR = Path(__file__).resolve().parent

# Figure styling
DP_COLOR = '#1f77b4'
DQN_COLOR = '#ff7f0e'
HEURISTIC_COLORS = ['#2ca02c', '#d62728', '#9467bd']


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class DiscountTargeting(EconBenchmark):
    """Dynamic discount targeting MDP.

    State components (controlled by n_state):
      1. recency       -- days since last ride
      2. frequency     -- lifetime rides
      3. disc_hist[-1] -- last discount given
      4. habituation   -- consecutive-discount counter
      5. disc_hist[-2] -- second-to-last discount
    """

    dp_feasible = True

    def __init__(self, n_state=3, habituation_effect=HABITUATION_EFFECT):
        self.n_state = n_state
        self.complexity_param = n_state
        self.habituation_effect = habituation_effect

        self._component_sizes = self._get_component_sizes()
        self.state_dim_int = len(self._component_sizes)
        self.total_states = int(np.prod(self._component_sizes))

        self.state = None
        self.t = 0
        self.reset()

    def _get_component_sizes(self):
        sizes = []
        if self.n_state >= 1:
            sizes.append(RECENCY_BINS)
        if self.n_state >= 2:
            sizes.append(FREQUENCY_BINS)
        if self.n_state >= 3:
            sizes.append(DISC_HIST_LEVELS)
        if self.n_state >= 4:
            sizes.append(H_MAX + 1)
        if self.n_state >= 5:
            sizes.append(DISC_HIST_LEVELS)
        return sizes

    def reset(self):
        self.recency = np.random.randint(0, RECENCY_BINS)
        self.frequency = np.random.randint(0, FREQUENCY_BINS)
        self.disc_hist = [0, 0]
        self.habituation = 0
        self.t = 0
        self.state = self._obs()
        return self.state

    def _obs(self):
        parts = []
        if self.n_state >= 1:
            parts.append(self.recency)
        if self.n_state >= 2:
            parts.append(self.frequency)
        if self.n_state >= 3:
            parts.append(self.disc_hist[-1])
        if self.n_state >= 4:
            parts.append(self.habituation)
        if self.n_state >= 5:
            parts.append(self.disc_hist[-2])
        return tuple(parts)

    @property
    def num_states(self):
        return self.total_states

    @property
    def num_actions(self):
        return N_ACTIONS

    def state_to_index(self, state):
        idx = 0
        for i, val in enumerate(state):
            idx = idx * self._component_sizes[i] + val
        return idx

    def index_to_state(self, idx):
        s = []
        for size in reversed(self._component_sizes):
            s.append(idx % size)
            idx //= size
        return tuple(reversed(s))

    def state_to_features(self, state):
        return np.array([state[i] / max(self._component_sizes[i] - 1, 1)
                         for i in range(len(state))], dtype=np.float32)

    def _conversion_prob(self, recency, habituation, discount_pct):
        logit = (BASE_RATE
                 - self.habituation_effect * habituation
                 + DISCOUNT_EFFECT * discount_pct
                 + RECENCY_EFFECT * recency)
        return 1.0 / (1.0 + np.exp(-logit))

    def step(self, action):
        discount_pct = DISCOUNT_LEVELS[action]
        p_convert = self._conversion_prob(self.recency, self.habituation, discount_pct)
        converted = (np.random.rand() < p_convert)

        if converted:
            reward = BASE_REVENUE * (1.0 - discount_pct)
        else:
            reward = 0.0

        if converted:
            self.recency = 0
            self.frequency = min(self.frequency + 1, FREQUENCY_BINS - 1)
            self.habituation = max(self.habituation - 1, 0)
        else:
            self.recency = min(self.recency + 1, RECENCY_BINS - 1)

        self.disc_hist.append(action)
        self.disc_hist = self.disc_hist[-2:]

        if action > 0:
            self.habituation = min(self.habituation + 1, H_MAX)
        else:
            self.habituation = max(self.habituation - 1, 0)

        self.t += 1
        done = (self.t >= EPISODE_LENGTH)
        self.state = self._obs()
        return self.state, reward, done

    def expected_reward(self, state, action):
        recency = state[0] if self.n_state >= 1 else 0
        habituation = state[3] if self.n_state >= 4 else 0
        discount_pct = DISCOUNT_LEVELS[action]
        p = self._conversion_prob(recency, habituation, discount_pct)
        return p * BASE_REVENUE * (1.0 - discount_pct)

    def transition_distribution(self, state, action):
        recency = state[0] if self.n_state >= 1 else 0
        frequency = state[1] if self.n_state >= 2 else 0
        disc_last = state[2] if self.n_state >= 3 else 0
        habituation = state[3] if self.n_state >= 4 else 0

        discount_pct = DISCOUNT_LEVELS[action]
        p = self._conversion_prob(recency, habituation, discount_pct)

        results = []

        for converted, prob in [(True, p), (False, 1.0 - p)]:
            if prob < 1e-12:
                continue
            r = recency
            f = frequency
            h = habituation

            if converted:
                r = 0
                f = min(f + 1, FREQUENCY_BINS - 1)
                h = max(h - 1, 0)
            else:
                r = min(r + 1, RECENCY_BINS - 1)

            if action > 0:
                h = min(h + 1, H_MAX)
            else:
                h = max(h - 1, 0)

            new_disc_hist_last = action
            new_disc_hist_prev = disc_last

            parts = []
            if self.n_state >= 1:
                parts.append(r)
            if self.n_state >= 2:
                parts.append(f)
            if self.n_state >= 3:
                parts.append(new_disc_hist_last)
            if self.n_state >= 4:
                parts.append(h)
            if self.n_state >= 5:
                parts.append(new_disc_hist_prev)

            ns = tuple(parts)
            results.append((ns, prob))

        return results


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def heuristic_static_ite(state):
    """Static ITE: always pick 10% discount."""
    return 1


def heuristic_threshold(state):
    """Give 30% discount if recency > 20, else 0%."""
    recency = state[0] if len(state) >= 1 else 0
    return 3 if recency > 20 else 0


def heuristic_no_discount(state):
    """Never give discount."""
    return 0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    env = DiscountTargeting(n_state=N_STATE)
    results = {}

    print(f"\n{'='*60}")
    print(f"  Discount Targeting (n_state={N_STATE}, {env.num_states} states)")
    print(f"{'='*60}")

    # --- Value Iteration ---
    print("  Running Value Iteration...")
    V, policy, vi_metrics = run_value_iteration(env, gamma=GAMMA)
    dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                   n_episodes=200, horizon=EPISODE_LENGTH)
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
        env_dqn = DiscountTargeting(n_state=N_STATE)
        q_net, metrics = run_dqn(
            env_dqn, gamma=GAMMA, num_episodes=NUM_EPISODES,
            episode_horizon=EPISODE_LENGTH, seed=seed, replay_size=10_000,
            batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
            epsilon_decay_frac=0.6, target_update_freq=50,
            eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES
        )

        reward = evaluate_dqn_policy(env, q_net, n_episodes=200, horizon=EPISODE_LENGTH)
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
    h_ite_reward = evaluate_heuristic(env, heuristic_static_ite,
                                       n_episodes=200, horizon=EPISODE_LENGTH)
    print(f"    Static ITE (10%): reward={h_ite_reward:.3f}")

    h_threshold_reward = evaluate_heuristic(env, heuristic_threshold,
                                             n_episodes=200, horizon=EPISODE_LENGTH)
    print(f"    Threshold: reward={h_threshold_reward:.3f}")

    h_no_discount_reward = evaluate_heuristic(env, heuristic_no_discount,
                                               n_episodes=200, horizon=EPISODE_LENGTH)
    print(f"    No Discount: reward={h_no_discount_reward:.3f}")

    results['heuristics'] = {
        'static_ite': h_ite_reward,
        'threshold': h_threshold_reward,
        'no_discount': h_no_discount_reward,
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

    h = results['heuristics']
    ax.axhline(h['static_ite'], color=HEURISTIC_COLORS[0], linestyle=':', linewidth=1.5,
               label=f'Static ITE ({h["static_ite"]:.2f})')

    ax.axhline(h['threshold'], color=HEURISTIC_COLORS[1], linestyle='-.', linewidth=1.5,
               label=f'Threshold ({h["threshold"]:.2f})')

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Episode Reward', fontsize=11)
    ax.set_title(f'Discount Targeting (n_state={N_STATE}) Learning Curve', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    out_path = OUTPUT_DIR / 'discount_targeting_learning_curve.png'
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
        f"Static ITE & ${h['static_ite']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(
        f"Threshold & ${h['threshold']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(
        f"No Discount & ${h['no_discount']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    out_path = OUTPUT_DIR / 'discount_targeting_results.tex'
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

    print("Dynamic Discount Targeting Benchmark")
    print(f"n_state={N_STATE}, episode_length={EPISODE_LENGTH}")
    print(f"Seeds: {SEEDS}")

    results = run_benchmark()
    make_figure(results)
    make_latex_table(results)

    print("\nBenchmark complete.")
