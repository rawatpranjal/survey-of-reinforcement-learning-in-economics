# Real-Time Bidding Budget Pacing Benchmark: DP vs DQN
# Chapter 3 -- Economic Benchmarks
# Budget-constrained bidding across second-price auctions with log-normal prices.

import matplotlib
matplotlib.use('Agg')

import os
import time
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import lognorm

from econ_benchmark import (EconBenchmark, run_value_iteration, run_dqn,
                            evaluate_dp_policy, evaluate_dqn_policy, evaluate_heuristic)

# ---------------------------------------------------------------------------
# Random seeds
# ---------------------------------------------------------------------------
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
T_VALUES = [12, 24, 48]
DP_CUTOFF = 24
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

DQN_SEEDS = [42, 123, 7]
EVAL_EPISODES = 50
DQN_EPISODES = {12: 500, 24: 1000, 48: 1500}

OUTPUT_DIR = Path(__file__).resolve().parent


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

        # Precompute win probability per multiplier
        scale = np.exp(PRICE_MU)
        self._win_probs = []
        self._cond_prices = []
        for m in BID_MULTIPLIERS:
            bid = BASE_BID * m
            if bid > 0:
                pw = lognorm.cdf(bid, PRICE_SIGMA, scale=scale)
                # E[price | price <= bid]
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
        # Budget constraint: expected spend
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


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def heuristic_uniform_pacing(state):
    """Spend budget/remaining_periods per period."""
    t, b, w = state
    b_mid = (RTBBiddingEnv._budget_mid(b))
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

# Helper for heuristics to get budget midpoint from bin
@staticmethod
def _budget_mid_static(b):
    edges = np.linspace(0, TOTAL_BUDGET, B_BINS + 1)
    return (edges[b] + edges[min(b + 1, B_BINS)]) / 2

RTBBiddingEnv._budget_mid = _budget_mid_static

_CURRENT_T = 12  # set in main()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    global _CURRENT_T
    results = {}

    for T in T_VALUES:
        _CURRENT_T = T
        env = RTBBiddingEnv(T)

        print(f"\n{'='*60}")
        print(f"T = {T}  |  |S| = {env.num_states:,}")
        print(f"{'='*60}")

        entry = {"T": T, "state_space": env.num_states}

        # --- Exact DP ---
        if T <= DP_CUTOFF:
            print("  Running Value Iteration ...")
            V, policy, dp_time = run_value_iteration(env, gamma=GAMMA)
            np.random.seed(42)
            dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                           n_episodes=EVAL_EPISODES, horizon=T)
            print(f"  DP: time={dp_time:.2f}s  reward={dp_reward:.2f}")
            entry["dp_time"] = dp_time
            entry["dp_reward"] = dp_reward
        else:
            entry["dp_time"] = None
            entry["dp_reward"] = None

        # --- DQN ---
        print(f"  Running DQN ({len(DQN_SEEDS)} seeds) ...")
        dqn_times, dqn_rewards = [], []
        for seed in DQN_SEEDS:
            q_net, dqn_time = run_dqn(env, gamma=GAMMA,
                                       num_episodes=DQN_EPISODES[T],
                                       episode_horizon=T,
                                       seed=seed)
            np.random.seed(seed)
            rew = evaluate_dqn_policy(env, q_net, n_episodes=EVAL_EPISODES, horizon=T)
            dqn_times.append(dqn_time)
            dqn_rewards.append(rew)
            print(f"    seed={seed}: time={dqn_time:.2f}s  reward={rew:.2f}")

        entry["dqn_time_mean"] = np.mean(dqn_times)
        entry["dqn_time_std"] = np.std(dqn_times)
        entry["dqn_reward_mean"] = np.mean(dqn_rewards)
        entry["dqn_reward_std"] = np.std(dqn_rewards)

        # --- Heuristics ---
        print("  Running heuristics ...")
        for name, fn in [("Uniform Pacing", heuristic_uniform_pacing),
                          ("Greedy", heuristic_greedy),
                          ("PID", heuristic_pid)]:
            np.random.seed(42)
            h_rew = evaluate_heuristic(env, fn, n_episodes=EVAL_EPISODES, horizon=T)
            entry[f"h_{name}"] = h_rew
            print(f"    {name}: reward={h_rew:.2f}")

        results[T] = entry

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    dp_Ts = [T for T in T_VALUES if results[T]["dp_time"] is not None]
    dp_ts = [results[T]["dp_time"] for T in dp_Ts]
    if dp_ts:
        ax1.semilogy(dp_Ts, dp_ts, 's-', color='tab:blue', label='Exact DP (VI)', markersize=7)
    dqn_ts = [results[T]["dqn_time_mean"] for T in T_VALUES]
    dqn_errs = [results[T]["dqn_time_std"] for T in T_VALUES]
    ax1.errorbar(T_VALUES, dqn_ts, yerr=dqn_errs, fmt='o-', color='tab:red',
                 label='DQN', markersize=7, capsize=3)
    ax1.set_xlabel('Campaign Length ($T$)', fontsize=12)
    ax1.set_ylabel('Wall-clock Time (s, log scale)', fontsize=12)
    ax1.set_title('Computational Cost', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    methods = ['dp_reward', 'dqn_reward_mean', 'h_Uniform Pacing', 'h_Greedy', 'h_PID']
    labels_m = ['DP (VI)', 'DQN', 'Uniform Pacing', 'Greedy', 'PID']
    colors_m = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    for m, lbl, clr in zip(methods, labels_m, colors_m):
        xs = [T for T in T_VALUES if results[T].get(m) is not None]
        ys = [results[T][m] for T in xs]
        ax2.plot(xs, ys, 'o-', color=clr, label=lbl, markersize=7)
    ax2.set_xlabel('Campaign Length ($T$)', fontsize=12)
    ax2.set_ylabel('Mean Conversions', fontsize=12)
    ax2.set_title('Policy Quality', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'rtb_bidding_scaling.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {OUTPUT_DIR / 'rtb_bidding_scaling.png'}")

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------
    lines = []
    lines.append(r"\begin{tabular}{r r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"$T$ & $|\mathcal{S}|$ & DP Conv. & DP Time & DQN Conv. & Uniform & Greedy & PID \\")
    lines.append(r"\midrule")
    for T in T_VALUES:
        e = results[T]
        ss = e["state_space"]
        dp_r = f"{e['dp_reward']:.1f}" if e["dp_reward"] is not None else "---"
        dp_t = f"{e['dp_time']:.2f}" if e["dp_time"] is not None else "---"
        dqn_r = f"{e['dqn_reward_mean']:.1f} $\\pm$ {e['dqn_reward_std']:.1f}"
        up = f"{e['h_Uniform Pacing']:.1f}"
        gr = f"{e['h_Greedy']:.1f}"
        pi = f"{e['h_PID']:.1f}"
        lines.append(f"{T} & {ss:,} & {dp_r} & {dp_t} & {dqn_r} & {up} & {gr} & {pi} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    table_path = OUTPUT_DIR / 'rtb_bidding_results.tex'
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Table saved to {table_path}")


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    print("RTB Budget Pacing Benchmark")
    print(f"T values: {T_VALUES}")
    print(f"Seeds: {DQN_SEEDS}")
    main()
    print("\nBenchmark complete.")
