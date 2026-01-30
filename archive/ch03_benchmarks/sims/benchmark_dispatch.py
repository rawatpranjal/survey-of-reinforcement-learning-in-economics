# Zone-Based Ride Dispatch Benchmark: DP vs DQN
# Chapter 3 -- Economic Benchmarks
# DiDi-inspired zone rebalancing with Poisson demand arrivals.

import matplotlib
matplotlib.use('Agg')

import os
import time
import random
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import poisson

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
D_MAX = 3
Q_MAX = 3
FARE = 10.0
REBALANCE_COST = 2.0
GAMMA = 0.95

K_VALUES = [2, 3, 4]
DP_CUTOFF = 2
EPISODE_HORIZON = 20
DQN_SEEDS = [42, 123, 7]
EVAL_EPISODES = 50

DQN_EPISODES = {2: 500, 3: 1000, 4: 1500}
OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DispatchEnvironment(EconBenchmark):
    """Zone-based ride dispatch with Poisson arrivals.

    State: (d_1,...,d_K, q_1,...,q_K) — idle drivers + demand queue per zone.
    Action: 0 = no rebalance, k = send driver from zone k to highest-queue zone.
    """

    def __init__(self, K, demand_rates):
        self.K = K
        self.demand_rates = list(demand_rates[:K])
        self._num_states = (D_MAX + 1) ** K * (Q_MAX + 1) ** K
        self._num_actions = K + 1

        # Precompute truncated Poisson probabilities per zone
        self._poisson_probs = []
        for lam in self.demand_rates:
            probs = np.array([poisson.pmf(k, lam) for k in range(Q_MAX + 1)])
            probs[-1] += 1.0 - probs.sum()  # lump tail at Q_MAX
            self._poisson_probs.append(probs)

        self.complexity_param = K
        self.dp_feasible = (K <= DP_CUTOFF)
        self.current_state = None

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    def reset(self):
        d_init = [D_MAX // 2] * self.K
        q_init = [0] * self.K
        self.current_state = tuple(d_init + q_init)
        return self.current_state

    def step(self, action):
        s = self.current_state
        drivers = list(s[:self.K])
        queues = list(s[self.K:])

        # Rebalancing
        rebalance_penalty = 0.0
        if action > 0:
            zone = action - 1
            if drivers[zone] > 0:
                target = int(np.argmax(queues))
                if target != zone:
                    drivers[zone] -= 1
                    drivers[target] = min(drivers[target] + 1, D_MAX)
                    rebalance_penalty = REBALANCE_COST

        # Matches
        matches = [min(drivers[k], queues[k]) for k in range(self.K)]
        total_matches = sum(matches)
        reward = FARE * total_matches - rebalance_penalty

        # Post-match: drivers return, queues clear matched
        for k in range(self.K):
            queues[k] = queues[k] - matches[k]
            # matched drivers return as idle
            drivers[k] = min(drivers[k] - matches[k] + matches[k], D_MAX)

        # New Poisson arrivals
        for k in range(self.K):
            arrival = np.random.choice(Q_MAX + 1, p=self._poisson_probs[k])
            queues[k] = min(queues[k] + arrival, Q_MAX)

        self.current_state = tuple(drivers + queues)
        return self.current_state, reward, False

    def state_to_index(self, state):
        idx = 0
        base = 1
        # drivers first, then queues
        for i in range(self.K):
            idx += state[i] * base
            base *= (D_MAX + 1)
        for i in range(self.K):
            idx += state[self.K + i] * base
            base *= (Q_MAX + 1)
        return idx

    def index_to_state(self, idx):
        parts = []
        for _ in range(self.K):
            parts.append(idx % (D_MAX + 1))
            idx //= (D_MAX + 1)
        for _ in range(self.K):
            parts.append(idx % (Q_MAX + 1))
            idx //= (Q_MAX + 1)
        return tuple(parts)

    def state_to_features(self, state):
        drivers = np.array(state[:self.K], dtype=np.float32) / max(D_MAX, 1)
        queues = np.array(state[self.K:], dtype=np.float32) / max(Q_MAX, 1)
        return np.concatenate([drivers, queues])

    def expected_reward(self, state, action):
        drivers = list(state[:self.K])
        queues = list(state[self.K:])

        rebalance_penalty = 0.0
        if action > 0:
            zone = action - 1
            if drivers[zone] > 0:
                target = int(np.argmax(queues))
                if target != zone:
                    drivers[zone] -= 1
                    drivers[target] = min(drivers[target] + 1, D_MAX)
                    rebalance_penalty = REBALANCE_COST

        matches = sum(min(drivers[k], queues[k]) for k in range(self.K))
        return FARE * matches - rebalance_penalty

    def transition_distribution(self, state, action):
        drivers = list(state[:self.K])
        queues = list(state[self.K:])

        # Apply rebalancing
        if action > 0:
            zone = action - 1
            if drivers[zone] > 0:
                target = int(np.argmax(queues))
                if target != zone:
                    drivers[zone] -= 1
                    drivers[target] = min(drivers[target] + 1, D_MAX)

        # Matches
        matches = [min(drivers[k], queues[k]) for k in range(self.K)]
        for k in range(self.K):
            queues[k] -= matches[k]

        # Enumerate all Poisson arrival combinations
        arrival_ranges = [range(Q_MAX + 1) for _ in range(self.K)]
        result = {}
        for arrivals in product(*arrival_ranges):
            prob = 1.0
            for k in range(self.K):
                prob *= self._poisson_probs[k][arrivals[k]]
            if prob < 1e-12:
                continue

            new_queues = [min(queues[k] + arrivals[k], Q_MAX) for k in range(self.K)]
            ns = tuple(drivers + new_queues)
            result[ns] = result.get(ns, 0.0) + prob

        return list(result.items())


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def heuristic_nearest_match(state):
    K = len(state) // 2
    drivers = list(state[:K])
    queues = list(state[K:])
    max_d_zone = int(np.argmax(drivers))
    max_q_zone = int(np.argmax(queues))
    if drivers[max_d_zone] > 0 and queues[max_q_zone] > 0 and max_d_zone != max_q_zone:
        return max_d_zone + 1
    return 0


def heuristic_no_rebalance(state):
    return 0


def heuristic_proportional(state):
    K = len(state) // 2
    drivers = np.array(state[:K], dtype=float)
    queues = np.array(state[K:], dtype=float)
    ratio = np.where(queues > 0, drivers / (queues + 1e-6), drivers + 10.0)
    max_zone = int(np.argmax(ratio))
    min_zone = int(np.argmin(ratio))
    if drivers[max_zone] > 0 and max_zone != min_zone:
        return max_zone + 1
    return 0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    results = {}

    for K in K_VALUES:
        demand_rates = [2.0, 1.5, 1.0, 0.5, 0.8][:K]
        env = DispatchEnvironment(K, demand_rates)

        print(f"\n{'='*60}")
        print(f"K = {K}  |  |S| = {env.num_states:,}")
        print(f"{'='*60}")

        entry = {"K": K, "state_space": env.num_states}

        # --- Exact DP ---
        if K <= DP_CUTOFF:
            print("  Running Value Iteration ...")
            V, policy, dp_time = run_value_iteration(env, gamma=GAMMA)
            np.random.seed(42)
            dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                           n_episodes=EVAL_EPISODES, horizon=EPISODE_HORIZON)
            print(f"  DP: time={dp_time:.2f}s  reward={dp_reward:.2f}")
            entry["dp_time"] = dp_time
            entry["dp_reward"] = dp_reward
        else:
            entry["dp_time"] = None
            entry["dp_reward"] = None

        # --- DQN ---
        print(f"  Running DQN ({len(DQN_SEEDS)} seeds) ...")
        dqn_times = []
        dqn_rewards = []
        for seed in DQN_SEEDS:
            q_net, dqn_time = run_dqn(env, gamma=GAMMA,
                                       num_episodes=DQN_EPISODES[K],
                                       episode_horizon=EPISODE_HORIZON,
                                       seed=seed)
            np.random.seed(seed)
            rew = evaluate_dqn_policy(env, q_net,
                                      n_episodes=EVAL_EPISODES, horizon=EPISODE_HORIZON)
            dqn_times.append(dqn_time)
            dqn_rewards.append(rew)
            print(f"    seed={seed}: time={dqn_time:.2f}s  reward={rew:.2f}")

        entry["dqn_time_mean"] = np.mean(dqn_times)
        entry["dqn_time_std"] = np.std(dqn_times)
        entry["dqn_reward_mean"] = np.mean(dqn_rewards)
        entry["dqn_reward_std"] = np.std(dqn_rewards)

        # --- Heuristics ---
        print("  Running heuristics ...")
        for name, fn in [("Nearest-Match", heuristic_nearest_match),
                          ("No-Rebalance", heuristic_no_rebalance),
                          ("Proportional", heuristic_proportional)]:
            np.random.seed(42)
            h_rew = evaluate_heuristic(env, fn, n_episodes=EVAL_EPISODES,
                                       horizon=EPISODE_HORIZON)
            entry[f"h_{name}"] = h_rew
            print(f"    {name}: reward={h_rew:.2f}")

        results[K] = entry

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: wall-clock time
    dp_Ks = [K for K in K_VALUES if results[K]["dp_time"] is not None]
    dp_ts = [results[K]["dp_time"] for K in dp_Ks]
    if dp_ts:
        ax1.semilogy(dp_Ks, dp_ts, 's-', color='tab:blue', label='Exact DP (VI)', markersize=7)

    dqn_ts = [results[K]["dqn_time_mean"] for K in K_VALUES]
    dqn_errs = [results[K]["dqn_time_std"] for K in K_VALUES]
    ax1.errorbar(K_VALUES, dqn_ts, yerr=dqn_errs, fmt='o-', color='tab:red',
                 label='DQN', markersize=7, capsize=3)
    ax1.set_xlabel('Number of Zones ($K$)', fontsize=12)
    ax1.set_ylabel('Wall-clock Time (s, log scale)', fontsize=12)
    ax1.set_title('Computational Cost', fontsize=13)
    ax1.set_xticks(K_VALUES)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: reward comparison
    methods = ['dp_reward', 'dqn_reward_mean', 'h_Nearest-Match',
               'h_No-Rebalance', 'h_Proportional']
    labels = ['DP (VI)', 'DQN', 'Nearest-Match', 'No-Rebalance', 'Proportional']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    for m, lbl, clr in zip(methods, labels, colors):
        xs = [K for K in K_VALUES if results[K].get(m) is not None]
        ys = [results[K][m] for K in xs]
        ax2.plot(xs, ys, 'o-', color=clr, label=lbl, markersize=7)
    ax2.set_xlabel('Number of Zones ($K$)', fontsize=12)
    ax2.set_ylabel('Mean Episode Reward', fontsize=12)
    ax2.set_title('Policy Quality', fontsize=13)
    ax2.set_xticks(K_VALUES)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'dispatch_scaling.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {OUTPUT_DIR / 'dispatch_scaling.png'}")

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------
    lines = []
    lines.append(r"\begin{tabular}{r r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"$K$ & $|\mathcal{S}|$ & DP Reward & DP Time & DQN Reward & Nearest & No-Rebal & Proportional \\")
    lines.append(r"\midrule")
    for K in K_VALUES:
        e = results[K]
        ss = e["state_space"]
        dp_r = f"{e['dp_reward']:.1f}" if e["dp_reward"] is not None else "---"
        dp_t = f"{e['dp_time']:.2f}" if e["dp_time"] is not None else "---"
        dqn_r = f"{e['dqn_reward_mean']:.1f} $\\pm$ {e['dqn_reward_std']:.1f}"
        nm = f"{e['h_Nearest-Match']:.1f}"
        nr = f"{e['h_No-Rebalance']:.1f}"
        pr = f"{e['h_Proportional']:.1f}"
        lines.append(f"{K} & {ss:,} & {dp_r} & {dp_t} & {dqn_r} & {nm} & {nr} & {pr} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    table_path = OUTPUT_DIR / 'dispatch_results.tex'
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Table saved to {table_path}")


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    print("Zone-Based Ride Dispatch Benchmark")
    print(f"K values: {K_VALUES}")
    print(f"Seeds: {DQN_SEEDS}")
    main()
    print("\nBenchmark complete.")
