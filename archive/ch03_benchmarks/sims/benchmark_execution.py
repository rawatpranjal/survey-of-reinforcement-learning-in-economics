"""
Financial Order Execution Benchmark (Chapter 3)

Optimal execution of a large equity order over T periods. The agent splits
a parent order into child trades to minimize execution cost (market impact +
timing risk). State space grows with T and discretization of shares/volume/volatility.
DP feasible for T <= 20. Compare DP, DQN, and classical heuristics (TWAP, VWAP, Greedy).
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
    evaluate_dp_policy, evaluate_dqn_policy, evaluate_heuristic
)

# =============================================================================
# Configuration
# =============================================================================
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

SEED = 42
DQN_SEEDS = [42, 123, 7]
OUTPUT_DIR = Path(__file__).resolve().parent

# Environment parameters
TOTAL_SHARES = 10          # Total shares to execute (discretized units)
SHARES_BINS = TOTAL_SHARES + 1   # 0..TOTAL_SHARES remaining
VOLUME_BINS = 3            # Low / Medium / High recent volume
VOLATILITY_BINS = 3        # Low / Medium / High recent volatility
NUM_TRADE_FRACTIONS = 6    # Action: trade 0%, 20%, 40%, 60%, 80%, 100% of remaining
TRADE_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

T_VALUES = [10, 20, 40, 80]   # Complexity dial: number of periods
DP_CUTOFF = 20                 # DP feasible for T <= 20

# Market parameters
MU = 0.0001           # Drift per period
SIGMA = 0.02          # Volatility per period
INITIAL_PRICE = 100.0
TEMP_IMPACT = 0.001   # Temporary impact coefficient
PERM_IMPACT = 0.0005  # Permanent impact coefficient
RISK_AVERSION = 0.01  # Penalty for variance / timing risk

GAMMA = 1.0           # Finite horizon, no discounting
EVAL_EPISODES = 50

# Volume/volatility transition probabilities (Markov)
# Each bin transitions to adjacent bins with some probability
VOL_TRANSITION = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6],
])

def get_dqn_episodes(T):
    if T <= 10:
        return 500
    elif T <= 20:
        return 1000
    elif T <= 40:
        return 2000
    else:
        return 3000


# =============================================================================
# Environment
# =============================================================================
class OrderExecution(EconBenchmark):
    """
    Order execution MDP.

    State: (time_remaining, shares_remaining, volume_bin, volatility_bin)
    Action: trade fraction index -> fraction of remaining shares to trade now
    Reward: negative execution cost = -(temporary_impact + permanent_impact + risk_penalty)

    Price follows GBM with temporary and permanent market impact.
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
        self.state = (self.T - 1, TOTAL_SHARES, 1, 1)  # Mid volume, mid volatility
        self.price = INITIAL_PRICE
        self.t = 0
        return self.state

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    def _encode_state(self, time_rem, shares_rem, vol_bin, volat_bin):
        return (time_rem, shares_rem, vol_bin, volat_bin)

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
        """Compute execution cost for trading a given number of shares."""
        if shares_to_trade <= 0:
            return 0.0
        # Volume scaling: low volume -> higher impact
        vol_scale = [1.5, 1.0, 0.7][vol_bin]
        # Volatility scaling: high volatility -> higher risk cost
        volat_scale = [0.7, 1.0, 1.5][volat_bin]

        frac = shares_to_trade / TOTAL_SHARES
        # Temporary impact: quadratic in trade size
        temp_cost = TEMP_IMPACT * INITIAL_PRICE * (shares_to_trade ** 2) * vol_scale
        # Permanent impact: linear in trade size
        perm_cost = PERM_IMPACT * INITIAL_PRICE * shares_to_trade * vol_scale
        # Risk penalty: proportional to remaining exposure and volatility
        risk_cost = RISK_AVERSION * INITIAL_PRICE * (shares_to_trade * SIGMA * volat_scale)

        return temp_cost + perm_cost + risk_cost

    def step(self, action):
        t_rem, shares_rem, vol_bin, volat_bin = self.state
        frac = TRADE_FRACTIONS[action]

        # Compute shares to trade (round to int)
        shares_to_trade = int(round(frac * shares_rem))
        shares_to_trade = min(shares_to_trade, shares_rem)

        # If last period, must trade everything remaining
        if t_rem <= 0:
            shares_to_trade = shares_rem

        # Compute cost
        cost = self._compute_trade_cost(shares_to_trade, vol_bin, volat_bin)

        # Penalty for not finishing by deadline
        new_shares = shares_rem - shares_to_trade
        deadline_penalty = 0.0
        if t_rem <= 0 and new_shares > 0:
            deadline_penalty = 5.0 * INITIAL_PRICE * new_shares

        reward = -(cost + deadline_penalty)

        # Transition volume and volatility bins (stochastic)
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


# =============================================================================
# Heuristic Policies
# =============================================================================
def heuristic_twap(state):
    """
    Time-Weighted Average Price: split remaining shares uniformly across
    remaining time periods.
    """
    t_rem, shares_rem, vol_bin, volat_bin = state
    if shares_rem <= 0:
        return 0  # Trade 0%
    periods_left = t_rem + 1
    target = shares_rem / periods_left
    target_frac = target / max(shares_rem, 1)
    # Find closest trade fraction
    best_a = 0
    best_diff = float('inf')
    for i, f in enumerate(TRADE_FRACTIONS):
        diff = abs(f - target_frac)
        if diff < best_diff:
            best_diff = diff
            best_a = i
    return best_a


def heuristic_vwap(state):
    """
    Volume-Weighted Average Price approximation: trade more in high-volume
    periods, less in low-volume periods.
    """
    t_rem, shares_rem, vol_bin, volat_bin = state
    if shares_rem <= 0:
        return 0
    periods_left = t_rem + 1
    base_frac = 1.0 / periods_left
    # Adjust by volume: trade more when volume is high
    vol_multiplier = [0.6, 1.0, 1.5][vol_bin]
    target_frac = min(1.0, base_frac * vol_multiplier)
    # Find closest trade fraction
    best_a = 0
    best_diff = float('inf')
    for i, f in enumerate(TRADE_FRACTIONS):
        diff = abs(f - target_frac)
        if diff < best_diff:
            best_diff = diff
            best_a = i
    return best_a


def heuristic_greedy(state):
    """
    Greedy: execute all remaining shares immediately.
    """
    t_rem, shares_rem, vol_bin, volat_bin = state
    if shares_rem <= 0:
        return 0
    return len(TRADE_FRACTIONS) - 1  # 100%


# =============================================================================
# Main Experiment
# =============================================================================
def main():
    results = {
        'T': [],
        'states': [],
        'dp_time': [],
        'dp_reward': [],
        'dqn_time': [],
        'dqn_reward': [],
        'dqn_std': [],
        'twap_reward': [],
        'vwap_reward': [],
        'greedy_reward': [],
    }

    for T in T_VALUES:
        print(f"\n{'='*60}")
        print(f"T = {T} periods")
        print(f"{'='*60}")

        env = OrderExecution(T=T)
        n_states = env.num_states
        n_actions = env.num_actions

        print(f"State space: {n_states}, Action space: {n_actions}")

        results['T'].append(T)
        results['states'].append(n_states)

        # Dynamic Programming
        if T <= DP_CUTOFF:
            print("\nRunning Value Iteration (DP)...")
            V, policy, dp_time = run_value_iteration(env, gamma=GAMMA)
            results['dp_time'].append(dp_time)

            print("Evaluating DP policy...")
            dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                           n_episodes=EVAL_EPISODES, horizon=T)
            results['dp_reward'].append(dp_reward)

            print(f"  DP time: {dp_time:.2f}s")
            print(f"  DP reward: {dp_reward:.2f}")
        else:
            print("\nSkipping DP (state space too large)")
            results['dp_time'].append(np.nan)
            results['dp_reward'].append(np.nan)

        # DQN
        print("\nRunning DQN...")
        dqn_episodes = get_dqn_episodes(T)
        dqn_times = []
        dqn_rewards = []

        for seed in DQN_SEEDS:
            env_dqn = OrderExecution(T=T)
            q_net, dqn_time = run_dqn(env_dqn, gamma=GAMMA,
                                       num_episodes=dqn_episodes,
                                       episode_horizon=T, seed=seed)
            dqn_reward = evaluate_dqn_policy(env, q_net,
                                              n_episodes=EVAL_EPISODES, horizon=T)
            dqn_times.append(dqn_time)
            dqn_rewards.append(dqn_reward)
            print(f"  seed={seed}: time={dqn_time:.2f}s, reward={dqn_reward:.2f}")

        results['dqn_time'].append(np.mean(dqn_times))
        results['dqn_reward'].append(np.mean(dqn_rewards))
        results['dqn_std'].append(np.std(dqn_rewards))

        # Heuristics
        print("\nEvaluating heuristics...")

        twap_reward = evaluate_heuristic(env, heuristic_twap,
                                          n_episodes=EVAL_EPISODES, horizon=T)
        results['twap_reward'].append(twap_reward)
        print(f"  TWAP: {twap_reward:.2f}")

        vwap_reward = evaluate_heuristic(env, heuristic_vwap,
                                          n_episodes=EVAL_EPISODES, horizon=T)
        results['vwap_reward'].append(vwap_reward)
        print(f"  VWAP: {vwap_reward:.2f}")

        greedy_reward = evaluate_heuristic(env, heuristic_greedy,
                                            n_episodes=EVAL_EPISODES, horizon=T)
        results['greedy_reward'].append(greedy_reward)
        print(f"  Greedy: {greedy_reward:.2f}")

    # =========================================================================
    # Generate Figure
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    T_array = np.array(results['T'])

    # Left panel: Computation time (log scale)
    dp_times_plot = [t if not np.isnan(t) else None for t in results['dp_time']]

    dp_T = [T_array[i] for i in range(len(T_array)) if dp_times_plot[i] is not None]
    dp_t = [dp_times_plot[i] for i in range(len(T_array)) if dp_times_plot[i] is not None]
    if dp_T:
        ax1.semilogy(dp_T, dp_t, 'o-', label='DP', linewidth=2, markersize=8)

    ax1.semilogy(T_array, results['dqn_time'], 's-', label='DQN', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Periods (T)', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds, log scale)', fontsize=12)
    ax1.set_title('Scaling: Computation Time', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(T_array)

    # Right panel: Reward comparison
    dp_T_r = [T_array[i] for i in range(len(T_array)) if not np.isnan(results['dp_reward'][i])]
    dp_r = [results['dp_reward'][i] for i in range(len(T_array)) if not np.isnan(results['dp_reward'][i])]
    if dp_T_r:
        ax2.plot(dp_T_r, dp_r, 'o-', label='DP', linewidth=2, markersize=8)

    ax2.errorbar(T_array, results['dqn_reward'], yerr=results['dqn_std'],
                 fmt='s-', label='DQN', linewidth=2, markersize=8, capsize=5)
    ax2.plot(T_array, results['twap_reward'], '^--', label='TWAP', linewidth=2, markersize=8, alpha=0.7)
    ax2.plot(T_array, results['vwap_reward'], 'v--', label='VWAP', linewidth=2, markersize=8, alpha=0.7)
    ax2.plot(T_array, results['greedy_reward'], 'd--', label='Greedy', linewidth=2, markersize=8, alpha=0.7)
    ax2.set_xlabel('Number of Periods (T)', fontsize=12)
    ax2.set_ylabel('Average Reward per Episode', fontsize=12)
    ax2.set_title('Performance Comparison', fontsize=13)
    ax2.legend(fontsize=10, loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(T_array)

    plt.tight_layout()

    fig_path = OUTPUT_DIR / "execution_scaling.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")

    # =========================================================================
    # Generate LaTeX Table
    # =========================================================================
    table_lines = []
    table_lines.append(r"\begin{tabular}{lrrrrrrr}")
    table_lines.append(r"\hline")
    table_lines.append(r"$T$ & States & DP Time & DP Reward & DQN Time & DQN Reward & TWAP & VWAP \\")
    table_lines.append(r"\hline")

    for i, T in enumerate(results['T']):
        states = results['states'][i]

        if np.isnan(results['dp_time'][i]):
            dp_time_str = "---"
            dp_reward_str = "---"
        else:
            dp_time_str = f"{results['dp_time'][i]:.2f}"
            dp_reward_str = f"{results['dp_reward'][i]:.2f}"

        dqn_time_str = f"{results['dqn_time'][i]:.2f}"
        dqn_reward_str = f"{results['dqn_reward'][i]:.2f}"
        twap_str = f"{results['twap_reward'][i]:.2f}"
        vwap_str = f"{results['vwap_reward'][i]:.2f}"

        table_lines.append(
            f"{T} & {states} & {dp_time_str} & {dp_reward_str} & "
            f"{dqn_time_str} & {dqn_reward_str} & {twap_str} & {vwap_str} \\\\"
        )

    table_lines.append(r"\hline")
    table_lines.append(r"\end{tabular}")

    table_content = "\n".join(table_lines)
    table_path = OUTPUT_DIR / "execution_results.tex"
    with open(table_path, 'w') as f:
        f.write(table_content)

    print(f"Table saved: {table_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
