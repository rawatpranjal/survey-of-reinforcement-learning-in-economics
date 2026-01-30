# Real-Time Bidding Budget Pacing Benchmark: Learning Curves
# Chapter 3 -- Economic Benchmarks
# Budget-constrained bidding across second-price auctions with log-normal prices.

import matplotlib
matplotlib.use('Agg')

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import lognorm

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

T = 24  # Campaign length for main experiment
B_BINS = 10
WR_BINS = 3
TOTAL_BUDGET = 1000.0
BASE_BID = 2.0
BID_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0]
N_AUCTIONS = 20
CONVERSION_RATE = 0.05
PRICE_MU = 0.5
PRICE_SIGMA = 0.8
GAMMA = 1.0

SEEDS = [42, 123, 7, 456, 789, 101, 202, 303, 404, 505]
NUM_EPISODES = 1000
EVAL_FREQ = 20
EVAL_EPISODES = 20

OUTPUT_DIR = Path(__file__).resolve().parent

# Figure styling
DP_COLOR = '#1f77b4'
DQN_COLOR = '#ff7f0e'
HEURISTIC_COLORS = ['#2ca02c', '#d62728', '#9467bd']


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class RTBBiddingEnv(EconBenchmark):
    """RTB budget pacing MDP.

    State: (t, budget_bin, win_rate_bin).
    Action: bid multiplier index (0..4).
    """

    def __init__(self, T):
        self.T = T
        self._num_states = T * B_BINS * WR_BINS
        self._num_actions = len(BID_MULTIPLIERS)

        self.budget_edges = np.linspace(0, TOTAL_BUDGET, B_BINS + 1)
        self.wr_edges = np.linspace(0, 1.0, WR_BINS + 1)

        scale = np.exp(PRICE_MU)
        self._win_probs = []
        self._cond_prices = []
        for m in BID_MULTIPLIERS:
            bid = BASE_BID * m
            if bid > 0:
                pw = lognorm.cdf(bid, PRICE_SIGMA, scale=scale)
                cp = lognorm.expect(lambda x: x, args=(PRICE_SIGMA,),
                                    scale=scale, lb=0, ub=bid) / max(pw, 1e-12)
            else:
                pw = 0.0
                cp = 0.0
            self._win_probs.append(pw)
            self._cond_prices.append(cp)

        self.current_state = None
        self._budget = 0.0
        self._win_rate = 0.0
        self._t = 0

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    def _discretize_budget(self, b):
        return min(int(np.searchsorted(self.budget_edges[1:], b)), B_BINS - 1)

    def _discretize_wr(self, wr):
        return min(int(np.searchsorted(self.wr_edges[1:], wr)), WR_BINS - 1)

    def reset(self):
        self._t = 0
        self._budget = TOTAL_BUDGET
        self._win_rate = 0.5
        self.current_state = (0, self._discretize_budget(self._budget),
                              self._discretize_wr(self._win_rate))
        return self.current_state

    def step(self, action):
        bid = BASE_BID * BID_MULTIPLIERS[action]
        prices = np.random.lognormal(PRICE_MU, PRICE_SIGMA, N_AUCTIONS)

        wins = (bid >= prices)
        costs = prices[wins]
        cum = np.cumsum(costs)
        affordable = cum <= self._budget
        n_wins = int(affordable.sum())
        spend = cum[n_wins - 1] if n_wins > 0 else 0.0

        self._budget = max(0.0, self._budget - spend)
        wr_now = n_wins / N_AUCTIONS
        self._win_rate = 0.7 * self._win_rate + 0.3 * wr_now

        conversions = np.random.binomial(n_wins, CONVERSION_RATE)
        reward = float(conversions)

        self._t += 1
        done = (self._t >= self.T)

        self.current_state = (min(self._t, self.T - 1),
                              self._discretize_budget(self._budget),
                              self._discretize_wr(self._win_rate))
        return self.current_state, reward, done

    def state_to_index(self, state):
        t, b, w = state
        return t * (B_BINS * WR_BINS) + b * WR_BINS + w

    def index_to_state(self, idx):
        w = idx % WR_BINS
        idx //= WR_BINS
        b = idx % B_BINS
        t = idx // B_BINS
        return (t, b, w)

    def state_to_features(self, state):
        t, b, w = state
        return np.array([t / max(self.T - 1, 1),
                         b / max(B_BINS - 1, 1),
                         w / max(WR_BINS - 1, 1)], dtype=np.float32)

    def expected_reward(self, state, action):
        t, b, w = state
        pw = self._win_probs[action]
        exp_wins = N_AUCTIONS * pw
        cp = self._cond_prices[action]
        b_mid = (self.budget_edges[b] + self.budget_edges[min(b + 1, B_BINS)]) / 2
        exp_spend = exp_wins * cp
        if exp_spend > b_mid:
            exp_wins = b_mid / max(cp, 1e-6)
        return exp_wins * CONVERSION_RATE

    def transition_distribution(self, state, action):
        t, b, w = state
        if t >= self.T - 1:
            return [(state, 1.0)]

        pw = self._win_probs[action]
        cp = self._cond_prices[action]
        exp_wins = N_AUCTIONS * pw
        b_mid = (self.budget_edges[b] + self.budget_edges[min(b + 1, B_BINS)]) / 2

        exp_spend = exp_wins * cp
        if exp_spend > b_mid:
            exp_wins = b_mid / max(cp, 1e-6)
            exp_spend = b_mid

        next_b = max(0.0, b_mid - exp_spend)
        wr_mid = (self.wr_edges[w] + self.wr_edges[min(w + 1, WR_BINS)]) / 2
        wr_now = exp_wins / N_AUCTIONS
        next_wr = 0.7 * wr_mid + 0.3 * wr_now

        ns = (t + 1, self._discretize_budget(next_b), self._discretize_wr(next_wr))
        return [(ns, 1.0)]

    @staticmethod
    def _budget_mid(b):
        edges = np.linspace(0, TOTAL_BUDGET, B_BINS + 1)
        return (edges[b] + edges[min(b + 1, B_BINS)]) / 2


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
_CURRENT_T = T


def heuristic_uniform_pacing(state):
    """Spend budget/remaining_periods per period."""
    t, b, w = state
    b_mid = RTBBiddingEnv._budget_mid(b)
    remaining = max(1, _CURRENT_T - t)
    target = b_mid / remaining
    if target < 10:
        return 0
    elif target < 50:
        return 1
    elif target < 100:
        return 2
    elif target < 150:
        return 3
    else:
        return 4


def heuristic_greedy(state):
    """Always bid max until budget gone."""
    t, b, w = state
    b_mid = RTBBiddingEnv._budget_mid(b)
    return 4 if b_mid > 10 else 0


def heuristic_pid(state):
    """PID controller on spend rate."""
    t, b, w = state
    b_mid = RTBBiddingEnv._budget_mid(b)
    target_rate = TOTAL_BUDGET / max(_CURRENT_T, 1)
    actual_rate = (TOTAL_BUDGET - b_mid) / max(t, 1) if t > 0 else target_rate
    error = target_rate - actual_rate
    if error > 5:
        return 4
    elif error > 2:
        return 3
    elif error > -2:
        return 2
    elif error > -5:
        return 1
    else:
        return 0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    global _CURRENT_T
    _CURRENT_T = T
    env = RTBBiddingEnv(T)
    results = {}

    print(f"\n{'='*60}")
    print(f"  RTB Bidding (T={T}, {env.num_states} states)")
    print(f"{'='*60}")

    # --- Value Iteration ---
    print("  Running Value Iteration...")
    V, policy, vi_metrics = run_value_iteration(env, gamma=GAMMA)
    dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                   n_episodes=200, horizon=T)
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
        env_dqn = RTBBiddingEnv(T)
        q_net, metrics = run_dqn(
            env_dqn, gamma=GAMMA, num_episodes=NUM_EPISODES,
            episode_horizon=T, seed=seed, replay_size=10_000,
            batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
            epsilon_decay_frac=0.6, target_update_freq=50,
            eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES
        )

        reward = evaluate_dqn_policy(env, q_net, n_episodes=200, horizon=T)
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
    h_uniform_reward = evaluate_heuristic(env, heuristic_uniform_pacing,
                                           n_episodes=200, horizon=T)
    print(f"    Uniform Pacing: reward={h_uniform_reward:.3f}")

    h_greedy_reward = evaluate_heuristic(env, heuristic_greedy,
                                          n_episodes=200, horizon=T)
    print(f"    Greedy: reward={h_greedy_reward:.3f}")

    h_pid_reward = evaluate_heuristic(env, heuristic_pid,
                                       n_episodes=200, horizon=T)
    print(f"    PID: reward={h_pid_reward:.3f}")

    results['heuristics'] = {
        'uniform_pacing': h_uniform_reward,
        'greedy': h_greedy_reward,
        'pid': h_pid_reward,
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
    ax.axhline(h['uniform_pacing'], color=HEURISTIC_COLORS[0], linestyle=':', linewidth=1.5,
               label=f'Uniform Pacing ({h["uniform_pacing"]:.2f})')

    ax.axhline(h['pid'], color=HEURISTIC_COLORS[1], linestyle='-.', linewidth=1.5,
               label=f'PID ({h["pid"]:.2f})')

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Episode Conversions', fontsize=11)
    ax.set_title(f'RTB Bidding (T={T}) Learning Curve', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    out_path = OUTPUT_DIR / 'rtb_bidding_learning_curve.png'
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
    lines.append(r'Method & Conversions & Time (s) & Convergence & Transitions & Entropy & DP Agr. \\')
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
        f"Uniform Pacing & ${h['uniform_pacing']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(
        f"Greedy & ${h['greedy']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(
        f"PID & ${h['pid']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    out_path = OUTPUT_DIR / 'rtb_bidding_results.tex'
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

    print("RTB Budget Pacing Benchmark")
    print(f"T={T}, budget={TOTAL_BUDGET}")
    print(f"Seeds: {SEEDS}")

    results = run_benchmark()
    make_figure(results)
    make_latex_table(results)

    print("\nBenchmark complete.")
