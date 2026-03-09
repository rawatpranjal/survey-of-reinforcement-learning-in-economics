# Zone-Based Ride Dispatch Benchmark: Scaling Analysis
# Chapter 3 -- Economic Benchmarks
# DiDi-inspired zone rebalancing with Poisson demand arrivals.
# Systematic scaling sweep across number of zones.

import matplotlib
matplotlib.use('Agg')

import random
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import poisson

from econ_benchmark import (
    EconBenchmark, run_value_iteration, run_dqn,
    evaluate_dp_policy, evaluate_dqn_policy, evaluate_heuristic,
    compute_policy_entropy, compute_policy_agreement, state_coverage,
    compute_q_error, evaluate_with_decomposition, capture_stdout,
    make_scaling_table, DP_FEASIBLE_THRESHOLD
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
D_MAX = 3
Q_MAX = 3
FARE = 10.0
REBALANCE_COST = 2.0
GAMMA = 0.95

# Scaling sweep configuration
# K=2: 256 states (DP feasible), K=5: 1M+ states (DP infeasible)
COMPLEXITY_SWEEP = [2, 5]  # Number of zones: small (DP works) vs large (DP breaks)
SEEDS = [42, 123, 7]  # 3 seeds for faster runs
NUM_EPISODES = 500
EPISODE_HORIZON = 20
EVAL_FREQ = 50
EVAL_EPISODES = 10

OUTPUT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = 'dispatch'

# Figure styling
DP_COLOR = '#1f77b4'
DQN_COLOR = '#ff7f0e'
HEURISTIC_COLORS = ['#2ca02c', '#d62728', '#9467bd']


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class DispatchEnvironment(EconBenchmark):
    """Zone-based ride dispatch with Poisson arrivals.

    State: (d_1,...,d_K, q_1,...,q_K) -- idle drivers + demand queue per zone.
    Action: 0 = no rebalance, k = send driver from zone k to highest-queue zone.
    """

    def __init__(self, K, demand_rates):
        self.K = K
        self.demand_rates = list(demand_rates[:K])
        self._num_states = (D_MAX + 1) ** K * (Q_MAX + 1) ** K
        self._num_actions = K + 1

        self._poisson_probs = []
        for lam in self.demand_rates:
            probs = np.array([poisson.pmf(k, lam) for k in range(Q_MAX + 1)])
            probs[-1] += 1.0 - probs.sum()
            self._poisson_probs.append(probs)

        self.complexity_param = K
        self.dp_feasible = self._num_states <= DP_FEASIBLE_THRESHOLD
        self.current_state = None

        # Track last step for decomposition
        self._last_matches = None
        self._last_rebalance = 0.0

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

        rebalance_penalty = 0.0
        if action > 0:
            zone = action - 1
            if drivers[zone] > 0:
                target = int(np.argmax(queues))
                if target != zone:
                    drivers[zone] -= 1
                    drivers[target] = min(drivers[target] + 1, D_MAX)
                    rebalance_penalty = REBALANCE_COST

        matches = [min(drivers[k], queues[k]) for k in range(self.K)]
        total_matches = sum(matches)
        reward = FARE * total_matches - rebalance_penalty

        # Store for decomposition
        self._last_matches = matches
        self._last_rebalance = rebalance_penalty

        for k in range(self.K):
            queues[k] = queues[k] - matches[k]
            drivers[k] = min(drivers[k] - matches[k] + matches[k], D_MAX)

        for k in range(self.K):
            arrival = np.random.choice(Q_MAX + 1, p=self._poisson_probs[k])
            queues[k] = min(queues[k] + arrival, Q_MAX)

        self.current_state = tuple(drivers + queues)
        return self.current_state, reward, False

    def state_to_index(self, state):
        idx = 0
        base = 1
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

        if action > 0:
            zone = action - 1
            if drivers[zone] > 0:
                target = int(np.argmax(queues))
                if target != zone:
                    drivers[zone] -= 1
                    drivers[target] = min(drivers[target] + 1, D_MAX)

        matches = [min(drivers[k], queues[k]) for k in range(self.K)]
        for k in range(self.K):
            queues[k] -= matches[k]

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
# Decomposition function for dispatch
# ---------------------------------------------------------------------------
def dispatch_decompose(state, action, reward, next_state, env):
    """Decompose reward into components for dispatch environment."""
    K = env.K
    matches = env._last_matches if env._last_matches else [0] * K
    rebalance_cost = env._last_rebalance

    components = {
        'total_matches': sum(matches),
        'fare_revenue': FARE * sum(matches),
        'rebalance_cost': rebalance_cost,
        'rebalanced': 1.0 if rebalance_cost > 0 else 0.0,
    }

    # Per-zone matches
    for k in range(K):
        components[f'zone_{k}_matches'] = matches[k]

    return components


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
# Run single complexity level
# ---------------------------------------------------------------------------
def run_single_complexity(K, seeds):
    """Run benchmark for a single complexity level (K zones)."""
    demand_rates = [2.0, 1.5, 1.0, 0.5, 0.8][:K]
    env = DispatchEnvironment(K, demand_rates)
    result = {
        'complexity': K,
        'states': env.num_states,
        'dp_feasible': env.dp_feasible,
    }

    print(f"\n  K={K}: {env.num_states:,} states, DP feasible={env.dp_feasible}")

    # --- Value Iteration (if feasible) ---
    V, policy, vi_metrics = None, None, None
    if env.dp_feasible:
        print("    Running Value Iteration...")
        V, policy, vi_metrics = run_value_iteration(env, gamma=GAMMA)
        dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                       n_episodes=200, horizon=EPISODE_HORIZON)
        result['dp_reward'] = dp_reward
        result['dp_time'] = vi_metrics.wall_time
        result['dp_iterations'] = vi_metrics.iterations
        print(f"      VI: {vi_metrics.iterations} iter, time={vi_metrics.wall_time:.2f}s, "
              f"reward={dp_reward:.3f}")
    else:
        result['dp_reward'] = None
        result['dp_time'] = None
        result['dp_iterations'] = None
        print("    DP skipped (state space too large)")

    # --- DQN (multiple seeds) ---
    print(f"    Running DQN ({len(seeds)} seeds)...")
    dqn_rewards = []
    dqn_times = []
    dqn_q_errors = []
    dqn_agreements = []
    all_curves = []
    checkpoint_episodes = None

    for i, seed in enumerate(seeds):
        env_dqn = DispatchEnvironment(K, demand_rates)
        q_net, metrics = run_dqn(
            env_dqn, gamma=GAMMA, num_episodes=NUM_EPISODES,
            episode_horizon=EPISODE_HORIZON, seed=seed, replay_size=10_000,
            batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
            epsilon_decay_frac=0.6, target_update_freq=50,
            eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES,
            desc=f"K={K} seed {i+1}/{len(seeds)}"
        )

        reward = evaluate_dqn_policy(env, q_net, n_episodes=200, horizon=EPISODE_HORIZON)
        dqn_rewards.append(reward)
        dqn_times.append(metrics.wall_time)
        all_curves.append(metrics.eval_checkpoints)
        checkpoint_episodes = metrics.checkpoint_episodes

        if V is not None:
            q_err = compute_q_error(q_net, V, env, n_samples=500, gamma=GAMMA)
            dqn_q_errors.append(q_err)
            agreement = compute_policy_agreement(q_net, policy, env, n_samples=500)
            dqn_agreements.append(agreement)

        print(f"      seed={seed}: reward={reward:.3f}, time={metrics.wall_time:.2f}s")

    result['dqn_reward_mean'] = np.mean(dqn_rewards)
    result['dqn_reward_std'] = np.std(dqn_rewards)
    result['dqn_rewards'] = dqn_rewards
    result['dqn_time_mean'] = np.mean(dqn_times)
    result['dqn_curves'] = all_curves
    result['checkpoint_episodes'] = checkpoint_episodes

    if dqn_q_errors:
        result['q_error'] = np.mean(dqn_q_errors)
        result['agreement'] = np.mean(dqn_agreements)
    else:
        result['q_error'] = None
        result['agreement'] = None

    # --- Heuristics ---
    print("    Running Heuristics...")
    h_nearest = evaluate_heuristic(env, heuristic_nearest_match,
                                   n_episodes=200, horizon=EPISODE_HORIZON)
    h_no_reb = evaluate_heuristic(env, heuristic_no_rebalance,
                                  n_episodes=200, horizon=EPISODE_HORIZON)
    h_prop = evaluate_heuristic(env, heuristic_proportional,
                                n_episodes=200, horizon=EPISODE_HORIZON)

    result['heuristics'] = {
        'nearest_match': h_nearest,
        'no_rebalance': h_no_reb,
        'proportional': h_prop,
    }
    print(f"      Nearest={h_nearest:.2f}, NoReb={h_no_reb:.2f}, Prop={h_prop:.2f}")

    # --- Decomposition (using best DQN policy) ---
    if all_curves:
        # Retrain one network for decomposition analysis
        print("    Running decomposition analysis...")
        env_decomp = DispatchEnvironment(K, demand_rates)
        q_net_decomp, _ = run_dqn(
            env_decomp, gamma=GAMMA, num_episodes=NUM_EPISODES,
            episode_horizon=EPISODE_HORIZON, seed=42, replay_size=10_000,
            batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
            epsilon_decay_frac=0.6, target_update_freq=50,
            eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES,
            show_progress=False
        )

        def dqn_policy(s):
            s_feat = env_decomp.state_to_features(s)
            with torch.no_grad():
                q_vals = q_net_decomp(torch.FloatTensor(s_feat).unsqueeze(0))
                return q_vals.argmax(dim=1).item()

        decomp = evaluate_with_decomposition(
            env_decomp, dqn_policy, n_episodes=100, horizon=EPISODE_HORIZON,
            decompose_fn=dispatch_decompose
        )
        result['decomposition'] = decomp.component_rewards

    return result


# ---------------------------------------------------------------------------
# Scaling sweep
# ---------------------------------------------------------------------------
def run_scaling_sweep():
    """Run benchmark across all complexity levels."""
    print("=" * 70)
    print("  Zone-Based Ride Dispatch: Scaling Analysis")
    print("=" * 70)
    print(f"  Complexity sweep: K in {COMPLEXITY_SWEEP}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Episodes: {NUM_EPISODES}, Horizon: {EPISODE_HORIZON}")

    results = []
    for K in COMPLEXITY_SWEEP:
        result = run_single_complexity(K, SEEDS)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figures(results):
    """Generate scaling comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Reward vs complexity
    ax = axes[0]
    complexities = [r['complexity'] for r in results]
    dp_rewards = [r['dp_reward'] if r['dp_reward'] else np.nan for r in results]
    dqn_means = [r['dqn_reward_mean'] for r in results]
    dqn_stds = [r['dqn_reward_std'] for r in results]

    x = np.arange(len(complexities))
    width = 0.35

    ax.bar(x - width/2, dp_rewards, width, label='DP', color=DP_COLOR, alpha=0.8)
    ax.bar(x + width/2, dqn_means, width, yerr=dqn_stds, label='DQN',
           color=DQN_COLOR, alpha=0.8, capsize=3)

    ax.set_xlabel('Number of Zones (K)', fontsize=11)
    ax.set_ylabel('Episode Reward', fontsize=11)
    ax.set_title('Dispatch: Reward vs Complexity', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(complexities)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Learning curves for each complexity
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))

    for i, r in enumerate(results):
        K = r['complexity']
        curves = r['dqn_curves']
        episodes = r['checkpoint_episodes']

        if curves and episodes:
            min_len = min(len(c) for c in curves)
            curves_arr = np.array([c[:min_len] for c in curves])
            episodes = episodes[:min_len]

            mean_rewards = np.mean(curves_arr, axis=0)
            std_rewards = np.std(curves_arr, axis=0)

            ax.plot(episodes, mean_rewards, color=colors[i], linewidth=2, label=f'K={K}')
            ax.fill_between(episodes, mean_rewards - std_rewards,
                           mean_rewards + std_rewards, color=colors[i], alpha=0.15)

        # Add DP baseline if available
        if r['dp_reward']:
            ax.axhline(r['dp_reward'], color=colors[i], linestyle='--',
                      linewidth=1, alpha=0.7)

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Episode Reward', fontsize=11)
    ax.set_title('Dispatch: Learning Curves by Complexity', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f'{SCRIPT_NAME}_scaling.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def make_latex_tables(results):
    """Generate LaTeX tables for scaling results."""
    # Main scaling table
    tex = make_scaling_table(results, 'K', 'Dispatch')
    out_path = OUTPUT_DIR / f'{SCRIPT_NAME}_results.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"LaTeX table saved to {out_path}")

    # Decomposition table for largest complexity that ran
    largest = results[-1]
    if 'decomposition' in largest and largest['decomposition']:
        lines = []
        lines.append(r'\begin{tabular}{lr}')
        lines.append(r'\toprule')
        lines.append(r'Component & Mean per Episode \\')
        lines.append(r'\midrule')

        decomp = largest['decomposition']
        lines.append(f"Total Matches & ${decomp.get('total_matches', 0):.2f}$ \\\\")
        lines.append(f"Fare Revenue & ${decomp.get('fare_revenue', 0):.2f}$ \\\\")
        lines.append(f"Rebalance Cost & ${decomp.get('rebalance_cost', 0):.2f}$ \\\\")
        lines.append(f"Rebalance Freq. & ${decomp.get('rebalanced', 0):.2%}$ \\\\")

        K = largest['complexity']
        for k in range(K):
            key = f'zone_{k}_matches'
            if key in decomp:
                lines.append(f"Zone {k} Matches & ${decomp[key]:.2f}$ \\\\")

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')

        decomp_tex = '\n'.join(lines)
        decomp_path = OUTPUT_DIR / f'{SCRIPT_NAME}_decomposition.tex'
        with open(decomp_path, 'w') as f:
            f.write(decomp_tex)
        print(f"Decomposition table saved to {decomp_path}")


# ---------------------------------------------------------------------------
# Detailed stdout output
# ---------------------------------------------------------------------------
def print_detailed_results(results):
    """Print detailed results to stdout."""
    print("\n" + "=" * 70)
    print("  DETAILED RESULTS")
    print("=" * 70)

    # Scaling summary table
    print("\n  Scaling Summary:")
    print("  " + "-" * 66)
    print(f"  {'K':>3} | {'States':>10} | {'DP Reward':>10} | {'DQN Reward':>16} | "
          f"{'Q-Error':>8} | {'Agr.':>6}")
    print("  " + "-" * 66)

    for r in results:
        K = r['complexity']
        states = r['states']
        dp_r = f"{r['dp_reward']:.2f}" if r['dp_reward'] else "---"
        dqn_r = f"{r['dqn_reward_mean']:.2f} +/- {r['dqn_reward_std']:.2f}"
        q_err = f"{r['q_error']:.3f}" if r['q_error'] else "---"
        agr = f"{r['agreement']:.1%}" if r['agreement'] else "---"
        print(f"  {K:>3} | {states:>10,} | {dp_r:>10} | {dqn_r:>16} | {q_err:>8} | {agr:>6}")

    print("  " + "-" * 66)

    # Per-seed results
    print("\n  Per-Seed DQN Rewards:")
    for r in results:
        K = r['complexity']
        rewards = r['dqn_rewards']
        print(f"    K={K}: " + ", ".join(f"{rw:.2f}" for rw in rewards))

    # Heuristic comparison
    print("\n  Heuristic Comparison:")
    print("  " + "-" * 50)
    print(f"  {'K':>3} | {'Nearest':>10} | {'No Rebal.':>10} | {'Proportional':>12}")
    print("  " + "-" * 50)
    for r in results:
        K = r['complexity']
        h = r['heuristics']
        print(f"  {K:>3} | {h['nearest_match']:>10.2f} | {h['no_rebalance']:>10.2f} | "
              f"{h['proportional']:>12.2f}")
    print("  " + "-" * 50)

    # Decomposition for largest K
    largest = results[-1]
    if 'decomposition' in largest and largest['decomposition']:
        print(f"\n  Reward Decomposition (K={largest['complexity']}, DQN policy):")
        decomp = largest['decomposition']
        for key, val in sorted(decomp.items()):
            print(f"    {key}: {val:.3f}")

    # Output files
    print("\n  Output Files:")
    print(f"    {OUTPUT_DIR / f'{SCRIPT_NAME}_scaling.png'}")
    print(f"    {OUTPUT_DIR / f'{SCRIPT_NAME}_results.tex'}")
    print(f"    {OUTPUT_DIR / f'{SCRIPT_NAME}_stdout.txt'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    stdout_path = OUTPUT_DIR / f'{SCRIPT_NAME}_stdout.txt'
    with capture_stdout(stdout_path):
        print("Zone-Based Ride Dispatch Benchmark")
        print(f"Complexity sweep: K in {COMPLEXITY_SWEEP}")
        print(f"Seeds: {SEEDS}")

        results = run_scaling_sweep()
        make_figures(results)
        make_latex_tables(results)
        print_detailed_results(results)

        print("\nBenchmark complete.")
