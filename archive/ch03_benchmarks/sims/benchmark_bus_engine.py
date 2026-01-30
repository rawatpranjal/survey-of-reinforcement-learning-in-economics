# Bus Engine Replacement Benchmark: Exact DP vs DQN
# Chapter 3 -- RL for Structural Estimation
# Benchmarks the multi-engine bus replacement problem (inspired by Rust 1987)
# across increasing fleet sizes N, comparing exact value iteration with DQN.

import itertools
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from econ_benchmark import (
    EconBenchmark, run_value_iteration, run_dqn,
    evaluate_dp_policy, evaluate_dqn_policy, evaluate_heuristic
)

# ---------------------------------------------------------------------------
# Random seeds
# ---------------------------------------------------------------------------
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MILEAGE_STATES = 6          # mileage in {0, 1, ..., 5}
ALPHA = 1.0                 # per-engine operating cost weight
BETA = 5.0                  # replacement cost weight
GAMMA = 0.95                # discount factor
CAPACITY = 3                # max engines replaceable per period

N_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
DP_CUTOFF = 6               # run exact DP only for N <= this

# Evaluation
EVAL_EPISODES = 100
EVAL_HORIZON = 20
DQN_SEEDS = [42, 123, 7]

OUTPUT_DIR = Path(__file__).resolve().parent


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

        # Enumerate all states and actions once
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
        """Reset and return initial state tuple."""
        self.state = tuple(np.random.randint(0, MILEAGE_STATES, size=self.N))
        return self.state

    def step(self, action):
        """Take action (integer index), return (next_state, reward, done)."""
        action_tuple = self._all_actions[action]
        cost = self._cost(self.state, action_tuple)
        self.state = self._transition(self.state, action_tuple)
        reward = -cost
        done = False  # Infinite horizon problem
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
        """Normalize mileage to [0, 1]."""
        return np.array(state, dtype=np.float32) / (MILEAGE_STATES - 1)

    def transition_distribution(self, state, action):
        """Return list of (next_state, probability) pairs. Deterministic transition."""
        action_tuple = self._all_actions[action]
        next_state = self._transition(state, action_tuple)
        return [(next_state, 1.0)]

    def expected_reward(self, state, action):
        """Return expected immediate reward (negative cost)."""
        action_tuple = self._all_actions[action]
        cost = self._cost(state, action_tuple)
        return -cost

    def _cost(self, state, action_tuple):
        """Immediate cost c(s,a) = ALPHA * sum(1{m_i>0}) + BETA * |a|."""
        operating = sum(1 for m in state if m > 0)
        return ALPHA * operating + BETA * len(action_tuple)

    def _transition(self, state, action_tuple):
        """Deterministic transition: replaced engines -> 0, others -> min(5, m+1)."""
        replace_set = set(action_tuple)
        new_state = []
        for i, m in enumerate(state):
            if i in replace_set:
                new_state.append(0)
            else:
                new_state.append(min(MILEAGE_STATES - 1, m + 1))
        return tuple(new_state)


# ---------------------------------------------------------------------------
# Heuristic: Replace engines with mileage >= threshold
# ---------------------------------------------------------------------------
def heuristic_threshold(state):
    """Replace all engines with mileage >= 3 (up to CAPACITY)."""
    threshold = 3
    to_replace = [i for i, m in enumerate(state) if m >= threshold]
    # Limit to CAPACITY
    to_replace = to_replace[:CAPACITY]
    # Find action index matching this tuple
    # For evaluation, we need to return the action index
    # We'll use a closure or env reference. For now, return 0 (no replacement) as fallback.
    return 0  # Placeholder; will be overridden in actual evaluation


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    results = {}

    for N in N_VALUES:
        state_space_size = MILEAGE_STATES ** N
        print(f"\n{'='*60}")
        print(f"N = {N}  |  |S| = {state_space_size}")
        print(f"{'='*60}")

        env = BusEngine(N)
        entry = {"N": N, "state_space": state_space_size}

        # --- Exact DP ---
        if N <= DP_CUTOFF:
            print(f"  Running exact Value Iteration ...")
            V, policy, dp_time = run_value_iteration(env, gamma=GAMMA)
            np.random.seed(42)
            dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA, n_episodes=EVAL_EPISODES, horizon=EVAL_HORIZON)
            print(f"  DP time: {dp_time:.2f}s  |  DP reward: {dp_reward:.3f}")
            entry["dp_time"] = dp_time
            entry["dp_reward"] = dp_reward
        else:
            entry["dp_time"] = None
            entry["dp_reward"] = None

        # --- DQN ---
        print(f"  Running DQN ({len(DQN_SEEDS)} seeds) ...")

        # Scale episodes with N
        if N <= 3:
            num_episodes = 5000
        elif N <= 6:
            num_episodes = 15000
        else:
            num_episodes = 25000

        dqn_times = []
        dqn_rewards = []
        for seed in DQN_SEEDS:
            env_dqn = BusEngine(N)
            q_net, dqn_time = run_dqn(
                env_dqn, gamma=GAMMA, num_episodes=num_episodes,
                episode_horizon=EVAL_HORIZON, seed=seed, replay_size=10_000,
                batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
                epsilon_decay_frac=0.6, target_update_freq=50
            )
            np.random.seed(seed)
            reward = evaluate_dqn_policy(env, q_net, n_episodes=EVAL_EPISODES, horizon=EVAL_HORIZON)
            dqn_times.append(dqn_time)
            dqn_rewards.append(reward)
            print(f"    seed={seed}: time={dqn_time:.2f}s  reward={reward:.3f}")

        entry["dqn_time_mean"] = np.mean(dqn_times)
        entry["dqn_time_std"] = np.std(dqn_times)
        entry["dqn_reward_mean"] = np.mean(dqn_rewards)
        entry["dqn_reward_std"] = np.std(dqn_rewards)

        results[N] = entry

    # --- Post-process: DP time projection for N=7,8 via log-linear regression ---
    dp_ns = [n for n in N_VALUES if n <= DP_CUTOFF]
    dp_times = [results[n]["dp_time"] for n in dp_ns]
    log_times = np.log(np.array(dp_times) + 1e-12)
    ns_arr = np.array(dp_ns, dtype=float)
    # Fit log(t) = a + b*N
    coeffs = np.polyfit(ns_arr, log_times, 1)
    projected = {}
    for n in N_VALUES:
        if n > DP_CUTOFF:
            projected[n] = np.exp(coeffs[1] + coeffs[0] * n)

    # ------------------------------------------------------------------
    # Figure: two-panel scaling plot
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: wall-clock time (log scale)
    dp_plot_ns = [n for n in N_VALUES if results[n]["dp_time"] is not None]
    dp_plot_times = [results[n]["dp_time"] for n in dp_plot_ns]
    ax1.semilogy(dp_plot_ns, dp_plot_times, "s-", color="tab:blue", label="Exact DP (VI)", markersize=7)

    # Projected DP times
    if projected:
        proj_ns = sorted(projected.keys())
        proj_times = [projected[n] for n in proj_ns]
        ax1.semilogy(proj_ns, proj_times, "s", color="tab:blue", fillstyle="none",
                      linestyle="--", markersize=7, label="DP projected")
        # Connect last actual to first projected with dashed line
        ax1.semilogy(
            [dp_plot_ns[-1]] + proj_ns,
            [dp_plot_times[-1]] + proj_times,
            "--", color="tab:blue", alpha=0.5
        )

    dqn_plot_ns = N_VALUES
    dqn_plot_times = [results[n]["dqn_time_mean"] for n in dqn_plot_ns]
    dqn_plot_errs = [results[n]["dqn_time_std"] for n in dqn_plot_ns]
    ax1.errorbar(dqn_plot_ns, dqn_plot_times, yerr=dqn_plot_errs,
                 fmt="o-", color="tab:red", label="DQN (mean +/- std)", markersize=7, capsize=3)

    ax1.set_xlabel("Number of Engines (N)", fontsize=12)
    ax1.set_ylabel("Wall-Clock Time (seconds, log scale)", fontsize=12)
    ax1.set_title("Computational Cost", fontsize=13)
    ax1.set_xticks(N_VALUES)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right panel: expected reward
    dp_reward_ns = [n for n in N_VALUES if results[n]["dp_reward"] is not None]
    dp_rewards = [results[n]["dp_reward"] for n in dp_reward_ns]
    ax2.plot(dp_reward_ns, dp_rewards, "s-", color="tab:blue", label="Exact DP (VI)", markersize=7)

    dqn_rewards_mean = [results[n]["dqn_reward_mean"] for n in N_VALUES]
    dqn_rewards_std = [results[n]["dqn_reward_std"] for n in N_VALUES]
    ax2.errorbar(N_VALUES, dqn_rewards_mean, yerr=dqn_rewards_std,
                 fmt="o-", color="tab:red", label="DQN (mean +/- std)", markersize=7, capsize=3)

    ax2.set_xlabel("Number of Engines (N)", fontsize=12)
    ax2.set_ylabel("Expected Discounted Reward", fontsize=12)
    ax2.set_title("Policy Quality", fontsize=13)
    ax2.set_xticks(N_VALUES)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "bus_engine_scaling.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------
    lines = []
    lines.append(r"\begin{tabular}{r r r r r r}")
    lines.append(r"\hline")
    lines.append(r"$N$ & $|\mathcal{S}|$ & DP Time (s) & DP Reward & DQN Time (s) & DQN Reward \\")
    lines.append(r"\hline")
    for N in N_VALUES:
        e = results[N]
        ss = e["state_space"]
        if e["dp_time"] is not None:
            dp_t = f"{e['dp_time']:.2f}"
            dp_r = f"{e['dp_reward']:.2f}"
        else:
            dp_t = "---"
            dp_r = "---"
        dqn_t = f"{e['dqn_time_mean']:.2f} $\\pm$ {e['dqn_time_std']:.2f}"
        dqn_r = f"{e['dqn_reward_mean']:.2f} $\\pm$ {e['dqn_reward_std']:.2f}"
        lines.append(f"{N} & {ss:,} & {dp_t} & {dp_r} & {dqn_t} & {dqn_r} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    table_path = OUTPUT_DIR / "bus_engine_results.tex"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to {table_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    for N in N_VALUES:
        e = results[N]
        dp_str = f"DP={e['dp_reward']:.2f}" if e["dp_reward"] is not None else "DP=n/a"
        print(f"  N={N}: |S|={e['state_space']:>8,}  {dp_str}  "
              f"DQN={e['dqn_reward_mean']:.2f}+/-{e['dqn_reward_std']:.2f}")


if __name__ == "__main__":
    main()
