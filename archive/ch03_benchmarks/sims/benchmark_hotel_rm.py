# Hotel Revenue Management Benchmark: DP vs DQN
# Chapter 3 -- Economic Benchmarks
# Dynamic pricing of perishable hotel rooms over a finite booking horizon.

import matplotlib
matplotlib.use('Agg')

import os
import time
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import poisson, binom

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
T_HORIZON = 20
C_VALUES = [10, 20, 30]
DP_CUTOFF = 20
D_BINS = 3          # demand intensity: 0=low, 1=medium, 2=high
PRICE_LEVELS = [50, 80, 120, 180, 250]
N_PRICES = len(PRICE_LEVELS)
ALPHA_LOGIT = 0.02
REF_PRICE = 120.0
BASE_ARRIVAL_RATE = 3.0
MAX_ARRIVALS = 8
GAMMA = 1.0         # finite horizon, no discounting

DQN_SEEDS = [42, 123, 7]
EVAL_EPISODES = 50
DQN_EPISODES = {10: 500, 20: 1000, 30: 1500}

OUTPUT_DIR = Path(__file__).resolve().parent

# Demand rates by demand bin
LAMBDA_RATES = {0: 0.5 * BASE_ARRIVAL_RATE,
                1: BASE_ARRIVAL_RATE,
                2: 1.8 * BASE_ARRIVAL_RATE}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HotelRMEnv(EconBenchmark):
    """Hotel revenue management MDP.

    State: (t, I, d) — days until check-in, remaining rooms, demand intensity.
    Action: price tier index in {0,...,4}.
    """

    def __init__(self, capacity):
        self.C = capacity
        self.T = T_HORIZON
        self._num_states = self.T * (self.C + 1) * D_BINS
        self._num_actions = N_PRICES
        self.current_state = None
        self.steps = 0

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    def reset(self):
        self.current_state = (self.T - 1, self.C, 1)  # full inventory, medium demand
        self.steps = 0
        return self.current_state

    def step(self, action):
        t, I, d = self.current_state
        price = PRICE_LEVELS[action]
        lam = LAMBDA_RATES[d]

        # Sample arrivals and bookings
        arrivals = min(np.random.poisson(lam), MAX_ARRIVALS)
        q_p = 1.0 / (1.0 + np.exp(ALPHA_LOGIT * (price - REF_PRICE)))
        bookings = 0
        for _ in range(arrivals):
            if np.random.rand() < q_p:
                bookings += 1
        bookings = min(bookings, I)
        reward = price * bookings

        # Next state
        next_I = I - bookings
        next_d = _demand_transition(arrivals)
        next_t = t - 1
        self.steps += 1

        done = (next_t < 0) or (next_I == 0) or (self.steps >= self.T)
        if done:
            self.current_state = (0, next_I, next_d)
        else:
            self.current_state = (next_t, next_I, next_d)

        return self.current_state, reward, done

    def state_to_index(self, state):
        t, I, d = state
        return t * ((self.C + 1) * D_BINS) + I * D_BINS + d

    def index_to_state(self, idx):
        d = idx % D_BINS
        idx //= D_BINS
        I = idx % (self.C + 1)
        t = idx // (self.C + 1)
        return (t, I, d)

    def state_to_features(self, state):
        t, I, d = state
        return np.array([t / max(self.T - 1, 1),
                         I / max(self.C, 1),
                         d / max(D_BINS - 1, 1)], dtype=np.float32)

    def expected_reward(self, state, action):
        t, I, d = state
        if I == 0 or t <= 0:
            return 0.0
        price = PRICE_LEVELS[action]
        lam = LAMBDA_RATES[d]
        q_p = 1.0 / (1.0 + np.exp(ALPHA_LOGIT * (price - REF_PRICE)))
        # E[bookings] = E[min(Binom(Poisson(lam), q_p), I)]
        # Approximate: sum over arrivals
        exp_rew = 0.0
        for arr in range(MAX_ARRIVALS + 1):
            p_arr = poisson.pmf(arr, lam)
            if p_arr < 1e-10:
                continue
            exp_book = min(arr * q_p, I)
            exp_rew += p_arr * price * exp_book
        return exp_rew

    def transition_distribution(self, state, action):
        t, I, d = state
        if I == 0 or t <= 0:
            return [(state, 1.0)]

        price = PRICE_LEVELS[action]
        lam = LAMBDA_RATES[d]
        q_p = 1.0 / (1.0 + np.exp(ALPHA_LOGIT * (price - REF_PRICE)))
        next_t = t - 1

        result = {}
        for arr in range(MAX_ARRIVALS + 1):
            p_arr = poisson.pmf(arr, lam)
            if p_arr < 1e-10:
                continue
            next_d = _demand_transition(arr)
            max_book = min(arr, I)
            for book in range(max_book + 1):
                p_book = binom.pmf(book, arr, q_p)
                if p_book < 1e-10:
                    continue
                prob = p_arr * p_book
                ns = (next_t, I - book, next_d)
                result[ns] = result.get(ns, 0.0) + prob

        return list(result.items())


def _demand_transition(arrivals):
    if arrivals < 0.7 * BASE_ARRIVAL_RATE:
        return 0
    elif arrivals < 1.5 * BASE_ARRIVAL_RATE:
        return 1
    else:
        return 2


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def heuristic_fixed_price(state):
    """Always charge the reference price ($120, index 2)."""
    return 2


def heuristic_emsr_b(state):
    """EMSR-b: price based on inventory scarcity relative to time."""
    t, I, d = state
    if t <= 0 or I == 0:
        return 0
    # Use a global ref for capacity — will be set before evaluation
    C = _EMSR_CAPACITY
    inv_ratio = I / max(C, 1)
    time_ratio = t / T_HORIZON
    if inv_ratio > time_ratio + 0.2:
        return 0   # $50 — excess inventory
    elif inv_ratio > time_ratio:
        return 1   # $80
    elif inv_ratio > time_ratio - 0.2:
        return 2   # $120
    elif inv_ratio > time_ratio - 0.4:
        return 3   # $180
    else:
        return 4   # $250 — scarcity

_EMSR_CAPACITY = 10  # will be set in main()


def heuristic_myopic(state):
    """Maximize immediate expected revenue."""
    t, I, d = state
    if t <= 0 or I == 0:
        return 0
    lam = LAMBDA_RATES[d]
    best_a, best_r = 0, -1.0
    for a, price in enumerate(PRICE_LEVELS):
        q_p = 1.0 / (1.0 + np.exp(ALPHA_LOGIT * (price - REF_PRICE)))
        exp_book = 0.0
        for arr in range(MAX_ARRIVALS + 1):
            p_arr = poisson.pmf(arr, lam)
            exp_book += p_arr * min(arr * q_p, I)
        er = price * exp_book
        if er > best_r:
            best_r = er
            best_a = a
    return best_a


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    global _EMSR_CAPACITY
    results = {}

    for C in C_VALUES:
        env = HotelRMEnv(capacity=C)
        _EMSR_CAPACITY = C

        print(f"\n{'='*60}")
        print(f"C = {C}  |  |S| = {env.num_states:,}")
        print(f"{'='*60}")

        entry = {"C": C, "state_space": env.num_states}

        # --- Exact DP ---
        if C <= DP_CUTOFF:
            print("  Running Value Iteration ...")
            V, policy, dp_time = run_value_iteration(env, gamma=GAMMA)
            np.random.seed(42)
            dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                           n_episodes=EVAL_EPISODES, horizon=T_HORIZON)
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
                                       num_episodes=DQN_EPISODES[C],
                                       episode_horizon=T_HORIZON,
                                       seed=seed)
            np.random.seed(seed)
            rew = evaluate_dqn_policy(env, q_net,
                                      n_episodes=EVAL_EPISODES, horizon=T_HORIZON)
            dqn_times.append(dqn_time)
            dqn_rewards.append(rew)
            print(f"    seed={seed}: time={dqn_time:.2f}s  reward={rew:.2f}")

        entry["dqn_time_mean"] = np.mean(dqn_times)
        entry["dqn_time_std"] = np.std(dqn_times)
        entry["dqn_reward_mean"] = np.mean(dqn_rewards)
        entry["dqn_reward_std"] = np.std(dqn_rewards)

        # --- Heuristics ---
        print("  Running heuristics ...")
        for name, fn in [("Fixed Price", heuristic_fixed_price),
                          ("EMSR-b", heuristic_emsr_b),
                          ("Myopic", heuristic_myopic)]:
            np.random.seed(42)
            h_rew = evaluate_heuristic(env, fn, n_episodes=EVAL_EPISODES,
                                       horizon=T_HORIZON)
            entry[f"h_{name}"] = h_rew
            print(f"    {name}: reward={h_rew:.2f}")

        results[C] = entry

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    dp_Cs = [C for C in C_VALUES if results[C]["dp_time"] is not None]
    dp_ts = [results[C]["dp_time"] for C in dp_Cs]
    if dp_ts:
        ax1.semilogy(dp_Cs, dp_ts, 's-', color='tab:blue', label='Exact DP (VI)', markersize=7)
    dqn_ts = [results[C]["dqn_time_mean"] for C in C_VALUES]
    dqn_errs = [results[C]["dqn_time_std"] for C in C_VALUES]
    ax1.errorbar(C_VALUES, dqn_ts, yerr=dqn_errs, fmt='o-', color='tab:red',
                 label='DQN', markersize=7, capsize=3)
    ax1.set_xlabel('Hotel Capacity ($C$)', fontsize=12)
    ax1.set_ylabel('Wall-clock Time (s, log scale)', fontsize=12)
    ax1.set_title('Computational Cost', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    methods = ['dp_reward', 'dqn_reward_mean', 'h_Fixed Price', 'h_EMSR-b', 'h_Myopic']
    labels_m = ['DP (VI)', 'DQN', 'Fixed Price', 'EMSR-b', 'Myopic']
    colors_m = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    for m, lbl, clr in zip(methods, labels_m, colors_m):
        xs = [C for C in C_VALUES if results[C].get(m) is not None]
        ys = [results[C][m] for C in xs]
        ax2.plot(xs, ys, 'o-', color=clr, label=lbl, markersize=7)
    ax2.set_xlabel('Hotel Capacity ($C$)', fontsize=12)
    ax2.set_ylabel('Mean Episode Revenue', fontsize=12)
    ax2.set_title('Policy Quality', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'hotel_rm_scaling.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {OUTPUT_DIR / 'hotel_rm_scaling.png'}")

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------
    lines = []
    lines.append(r"\begin{tabular}{r r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"$C$ & $|\mathcal{S}|$ & DP Revenue & DP Time & DQN Revenue & Fixed & EMSR-b & Myopic \\")
    lines.append(r"\midrule")
    for C in C_VALUES:
        e = results[C]
        ss = e["state_space"]
        dp_r = f"{e['dp_reward']:.1f}" if e["dp_reward"] is not None else "---"
        dp_t = f"{e['dp_time']:.2f}" if e["dp_time"] is not None else "---"
        dqn_r = f"{e['dqn_reward_mean']:.1f} $\\pm$ {e['dqn_reward_std']:.1f}"
        fp = f"{e['h_Fixed Price']:.1f}"
        em = f"{e['h_EMSR-b']:.1f}"
        my = f"{e['h_Myopic']:.1f}"
        lines.append(f"{C} & {ss:,} & {dp_r} & {dp_t} & {dqn_r} & {fp} & {em} & {my} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    table_path = OUTPUT_DIR / 'hotel_rm_results.tex'
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Table saved to {table_path}")


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    print("Hotel Revenue Management Benchmark")
    print(f"Capacity values: {C_VALUES}")
    print(f"Seeds: {DQN_SEEDS}")
    main()
    print("\nBenchmark complete.")
