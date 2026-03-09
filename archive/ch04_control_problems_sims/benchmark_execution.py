"""
Financial Order Execution Benchmark (Chapter 3)

Optimal execution of a large equity order over T periods. The agent splits
a parent order into child trades to minimize execution cost (market impact +
timing risk). Learning curve comparison of DP vs DQN.
"""

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

TOTAL_SHARES = 10
SHARES_BINS = TOTAL_SHARES + 1
VOLUME_BINS = 3
VOLATILITY_BINS = 3
NUM_TRADE_FRACTIONS = 6
TRADE_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

T = 20  # Number of periods for main experiment
SEEDS = [42, 123, 7, 456, 789, 101, 202, 303, 404, 505]
NUM_EPISODES = 1000
EVAL_FREQ = 20
EVAL_EPISODES = 20

MU = 0.0001
SIGMA = 0.02
INITIAL_PRICE = 100.0
TEMP_IMPACT = 0.001
PERM_IMPACT = 0.0005
RISK_AVERSION = 0.01

GAMMA = 1.0

OUTPUT_DIR = Path(__file__).resolve().parent

VOL_TRANSITION = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6],
])

# Figure styling
DP_COLOR = '#1f77b4'
DQN_COLOR = '#ff7f0e'
HEURISTIC_COLORS = ['#2ca02c', '#d62728', '#9467bd']


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class OrderExecution(EconBenchmark):
    """Order execution MDP.

    State: (time_remaining, shares_remaining, volume_bin, volatility_bin)
    Action: trade fraction index -> fraction of remaining shares to trade now
    Reward: negative execution cost
    """

    dp_feasible = True

    def __init__(self, T, gamma=GAMMA, horizon=None):
        self.T = T
        self.complexity_param = T
        self.gamma = gamma
        self.horizon = T if horizon is None else horizon

        self._num_states = T * SHARES_BINS * VOLUME_BINS * VOLATILITY_BINS
        self._num_actions = NUM_TRADE_FRACTIONS

        self.state = None
        self.price = INITIAL_PRICE
        self.t = 0
        self.reset()

    def reset(self):
        self.state = (self.T - 1, TOTAL_SHARES, 1, 1)
        self.price = INITIAL_PRICE
        self.t = 0
        return self.state

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    def state_to_index(self, state):
        t_rem, shares, vol, volat = state
        idx = t_rem
        idx = idx * SHARES_BINS + shares
        idx = idx * VOLUME_BINS + vol
        idx = idx * VOLATILITY_BINS + volat
        return idx

    def index_to_state(self, idx):
        volat = idx % VOLATILITY_BINS
        idx //= VOLATILITY_BINS
        vol = idx % VOLUME_BINS
        idx //= VOLUME_BINS
        shares = idx % SHARES_BINS
        idx //= SHARES_BINS
        t_rem = idx
        return (t_rem, shares, vol, volat)

    def state_to_features(self, state):
        t_rem, shares, vol, volat = state
        return np.array([
            t_rem / max(self.T - 1, 1),
            shares / TOTAL_SHARES,
            vol / max(VOLUME_BINS - 1, 1),
            volat / max(VOLATILITY_BINS - 1, 1),
        ], dtype=np.float32)

    def _compute_trade_cost(self, shares_to_trade, vol_bin, volat_bin):
        if shares_to_trade <= 0:
            return 0.0
        vol_scale = [1.5, 1.0, 0.7][vol_bin]
        volat_scale = [0.7, 1.0, 1.5][volat_bin]

        temp_cost = TEMP_IMPACT * INITIAL_PRICE * (shares_to_trade ** 2) * vol_scale
        perm_cost = PERM_IMPACT * INITIAL_PRICE * shares_to_trade * vol_scale
        risk_cost = RISK_AVERSION * INITIAL_PRICE * (shares_to_trade * SIGMA * volat_scale)

        return temp_cost + perm_cost + risk_cost

    def step(self, action):
        t_rem, shares_rem, vol_bin, volat_bin = self.state
        frac = TRADE_FRACTIONS[action]

        shares_to_trade = int(round(frac * shares_rem))
        shares_to_trade = min(shares_to_trade, shares_rem)

        if t_rem <= 0:
            shares_to_trade = shares_rem

        cost = self._compute_trade_cost(shares_to_trade, vol_bin, volat_bin)

        new_shares = shares_rem - shares_to_trade
        deadline_penalty = 0.0
        if t_rem <= 0 and new_shares > 0:
            deadline_penalty = 5.0 * INITIAL_PRICE * new_shares

        reward = -(cost + deadline_penalty)

        new_vol = np.random.choice(VOLUME_BINS, p=VOL_TRANSITION[vol_bin])
        new_volat = np.random.choice(VOLATILITY_BINS, p=VOL_TRANSITION[volat_bin])

        new_t_rem = max(0, t_rem - 1)
        self.state = (new_t_rem, new_shares, new_vol, new_volat)
        self.t += 1
        done = (self.t >= self.horizon) or (new_shares <= 0)

        return self.state, reward, done

    def transition_distribution(self, state, action):
        t_rem, shares_rem, vol_bin, volat_bin = state
        frac = TRADE_FRACTIONS[action]

        shares_to_trade = int(round(frac * shares_rem))
        shares_to_trade = min(shares_to_trade, shares_rem)
        if t_rem <= 0:
            shares_to_trade = shares_rem

        new_shares = shares_rem - shares_to_trade
        new_t_rem = max(0, t_rem - 1)

        transitions = []
        for nv in range(VOLUME_BINS):
            p_vol = VOL_TRANSITION[vol_bin, nv]
            if p_vol < 1e-10:
                continue
            for nvt in range(VOLATILITY_BINS):
                p_volat = VOL_TRANSITION[volat_bin, nvt]
                if p_volat < 1e-10:
                    continue
                prob = p_vol * p_volat
                ns = (new_t_rem, new_shares, nv, nvt)
                transitions.append((ns, prob))
        return transitions

    def expected_reward(self, state, action):
        t_rem, shares_rem, vol_bin, volat_bin = state
        frac = TRADE_FRACTIONS[action]

        shares_to_trade = int(round(frac * shares_rem))
        shares_to_trade = min(shares_to_trade, shares_rem)
        if t_rem <= 0:
            shares_to_trade = shares_rem

        cost = self._compute_trade_cost(shares_to_trade, vol_bin, volat_bin)

        new_shares = shares_rem - shares_to_trade
        deadline_penalty = 0.0
        if t_rem <= 0 and new_shares > 0:
            deadline_penalty = 5.0 * INITIAL_PRICE * new_shares

        return -(cost + deadline_penalty)


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def heuristic_twap(state):
    """Time-Weighted Average Price: split uniformly across time."""
    t_rem, shares_rem, vol_bin, volat_bin = state
    if shares_rem <= 0:
        return 0
    periods_left = t_rem + 1
    target = shares_rem / periods_left
    target_frac = target / max(shares_rem, 1)
    best_a = 0
    best_diff = float('inf')
    for i, f in enumerate(TRADE_FRACTIONS):
        diff = abs(f - target_frac)
        if diff < best_diff:
            best_diff = diff
            best_a = i
    return best_a


def heuristic_vwap(state):
    """Volume-Weighted Average Price: trade more in high volume."""
    t_rem, shares_rem, vol_bin, volat_bin = state
    if shares_rem <= 0:
        return 0
    periods_left = t_rem + 1
    base_frac = 1.0 / periods_left
    vol_multiplier = [0.6, 1.0, 1.5][vol_bin]
    target_frac = min(1.0, base_frac * vol_multiplier)
    best_a = 0
    best_diff = float('inf')
    for i, f in enumerate(TRADE_FRACTIONS):
        diff = abs(f - target_frac)
        if diff < best_diff:
            best_diff = diff
            best_a = i
    return best_a


def heuristic_greedy(state):
    """Greedy: execute all remaining shares immediately."""
    t_rem, shares_rem, vol_bin, volat_bin = state
    if shares_rem <= 0:
        return 0
    return len(TRADE_FRACTIONS) - 1


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    env = OrderExecution(T=T)
    results = {}

    print(f"\n{'='*60}")
    print(f"  Order Execution (T={T}, {env.num_states} states)")
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
        env_dqn = OrderExecution(T=T)
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
    h_twap_reward = evaluate_heuristic(env, heuristic_twap,
                                        n_episodes=200, horizon=T)
    print(f"    TWAP: reward={h_twap_reward:.3f}")

    h_vwap_reward = evaluate_heuristic(env, heuristic_vwap,
                                        n_episodes=200, horizon=T)
    print(f"    VWAP: reward={h_vwap_reward:.3f}")

    h_greedy_reward = evaluate_heuristic(env, heuristic_greedy,
                                          n_episodes=200, horizon=T)
    print(f"    Greedy: reward={h_greedy_reward:.3f}")

    results['heuristics'] = {
        'twap': h_twap_reward,
        'vwap': h_vwap_reward,
        'greedy': h_greedy_reward,
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
    ax.axhline(h['twap'], color=HEURISTIC_COLORS[0], linestyle=':', linewidth=1.5,
               label=f'TWAP ({h["twap"]:.2f})')

    ax.axhline(h['vwap'], color=HEURISTIC_COLORS[1], linestyle='-.', linewidth=1.5,
               label=f'VWAP ({h["vwap"]:.2f})')

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Episode Reward (negative cost)', fontsize=11)
    ax.set_title(f'Order Execution (T={T}) Learning Curve', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    out_path = OUTPUT_DIR / 'execution_learning_curve.png'
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
        f"TWAP & ${h['twap']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(
        f"VWAP & ${h['vwap']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(
        f"Greedy & ${h['greedy']:.2f}$ & --- & --- & --- & --- & --- \\\\"
    )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    out_path = OUTPUT_DIR / 'execution_results.tex'
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

    print("Financial Order Execution Benchmark")
    print(f"T={T} periods, {TOTAL_SHARES} shares")
    print(f"Seeds: {SEEDS}")

    results = run_benchmark()
    make_figure(results)
    make_latex_table(results)

    print("\nBenchmark complete.")
