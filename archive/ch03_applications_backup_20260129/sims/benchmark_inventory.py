"""
Multi-Echelon Inventory Management Benchmark: Scaling Analysis (Chapter 3)

Serial inventory system with K echelons. Echelon 1 faces stochastic customer demand.
Systematic scaling sweep across number of echelons.
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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
I_MAX = 8
Q_MAX = 5

DEMAND_LAMBDA = 4.0
HOLDING_COST = 1.0
BACKORDER_COST = 10.0
ORDER_COST = 0.5

GAMMA = 0.95
EPISODE_HORIZON = 20

# Scaling sweep configuration
# K=2: 81 states (DP feasible), K=6: ~530K states (DP infeasible)
COMPLEXITY_SWEEP = [2, 6]  # Number of echelons: small (DP works) vs large (DP breaks)
SEEDS = [42, 123, 7]  # 3 seeds for faster runs
NUM_EPISODES = 500
EVAL_FREQ = 50
EVAL_EPISODES = 10

MAX_DEMAND = int(2 * DEMAND_LAMBDA + 10)

OUTPUT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = 'inventory'

# Figure styling
DP_COLOR = '#1f77b4'
DQN_COLOR = '#ff7f0e'
HEURISTIC_COLORS = ['#2ca02c', '#d62728', '#9467bd']


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class MultiEchelonInventory(EconBenchmark):
    """Serial multi-echelon inventory system.

    State: (I_1, ..., I_K) where I_k in {0, ..., I_MAX}
    Action: order quantity q in {0, ..., Q_MAX} for echelon K
    Demand: D ~ Poisson(DEMAND_LAMBDA) at echelon 1
    """

    def __init__(self, K, gamma=GAMMA, horizon=EPISODE_HORIZON):
        self.K = K
        self.complexity_param = K
        self.gamma = gamma
        self.horizon = horizon

        self._num_states = (I_MAX + 1) ** K
        self._num_actions = Q_MAX + 1

        self.dp_feasible = self._num_states <= DP_FEASIBLE_THRESHOLD

        self.state = None
        self.t = 0

        self.demand_probs = np.array([poisson.pmf(d, DEMAND_LAMBDA) for d in range(MAX_DEMAND + 1)])
        self.demand_probs /= self.demand_probs.sum()

        self.initial_inventory = I_MAX // 2

        # Track last step for decomposition
        self._last_holding_cost = 0.0
        self._last_backorder_cost = 0.0
        self._last_order_cost = 0.0
        self._last_demand = 0
        self._last_stockout = False
        self._last_inventory = None

        self.reset()

    def reset(self):
        self.state = tuple([self.initial_inventory] * self.K)
        self.t = 0
        return self.state

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    def state_to_index(self, state):
        idx = 0
        multiplier = 1
        for i in range(self.K):
            idx += state[i] * multiplier
            multiplier *= (I_MAX + 1)
        return idx

    def index_to_state(self, idx):
        state = []
        for _ in range(self.K):
            state.append(idx % (I_MAX + 1))
            idx //= (I_MAX + 1)
        return tuple(state)

    def state_to_features(self, state):
        return np.array(state, dtype=np.float32) / I_MAX

    def step(self, action):
        demand = np.random.choice(MAX_DEMAND + 1, p=self.demand_probs)
        next_state, reward = self._transition(self.state, action, demand)

        self.state = next_state
        self.t += 1
        done = (self.t >= self.horizon)

        return next_state, reward, done

    def _transition(self, state, action, demand):
        inventory = list(state)
        q = action

        # Sales and backorders at echelon 1
        sales = min(inventory[0], demand)
        backorders = max(0, demand - inventory[0])
        inventory[0] = max(0, inventory[0] - demand)

        # Store stockout info
        self._last_stockout = backorders > 0
        self._last_demand = demand

        # Shipments between echelons
        for k in range(1, self.K):
            deficit = max(0, self.initial_inventory - inventory[k-1])
            shipment = min(inventory[k], deficit)
            inventory[k] -= shipment
            inventory[k-1] += shipment

        # Order arrival at last echelon
        inventory[self.K - 1] = min(I_MAX, inventory[self.K - 1] + q)

        # Costs
        holding_cost = HOLDING_COST * sum(inventory)
        backorder_cost = BACKORDER_COST * backorders
        order_cost = ORDER_COST * q

        # Store for decomposition
        self._last_holding_cost = holding_cost
        self._last_backorder_cost = backorder_cost
        self._last_order_cost = order_cost
        self._last_inventory = list(inventory)

        reward = -(holding_cost + backorder_cost + order_cost)

        next_state = tuple(min(I_MAX, max(0, inv)) for inv in inventory)

        return next_state, reward

    def transition_distribution(self, state, action):
        transitions = []
        for demand in range(MAX_DEMAND + 1):
            prob = self.demand_probs[demand]
            if prob > 1e-10:
                next_state, reward = self._transition(state, action, demand)
                transitions.append((next_state, prob))
        return transitions

    def expected_reward(self, state, action):
        expected_r = 0.0
        for demand in range(MAX_DEMAND + 1):
            prob = self.demand_probs[demand]
            if prob > 1e-10:
                _, reward = self._transition(state, action, demand)
                expected_r += prob * reward
        return expected_r


# ---------------------------------------------------------------------------
# Decomposition function for inventory
# ---------------------------------------------------------------------------
def inventory_decompose(state, action, reward, next_state, env):
    """Decompose cost into components for inventory environment."""
    K = env.K

    components = {
        'total_cost': -reward,
        'holding_cost': env._last_holding_cost,
        'backorder_cost': env._last_backorder_cost,
        'order_cost': env._last_order_cost,
        'demand': env._last_demand,
        'stockout': 1.0 if env._last_stockout else 0.0,
        'order_qty': action,
    }

    # Per-echelon inventory
    if env._last_inventory:
        for k in range(K):
            components[f'echelon_{k}_inv'] = env._last_inventory[k]
        components['total_inv'] = sum(env._last_inventory)

    return components


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def heuristic_base_stock(state):
    """Newsvendor-style base-stock policy for echelon 1."""
    critical_fractile = BACKORDER_COST / (BACKORDER_COST + HOLDING_COST)
    S_star = poisson.ppf(critical_fractile, DEMAND_LAMBDA)
    S_star = int(min(S_star, I_MAX))

    total_inventory = sum(state)
    target_order = max(0, S_star - total_inventory)
    return min(target_order, Q_MAX)


def heuristic_s_S(state):
    """(s, S) policy: if total inventory < s, order up to S; else order 0."""
    s = int(DEMAND_LAMBDA - 1)
    S = int(DEMAND_LAMBDA + 3)
    S = min(S, I_MAX)

    total_inventory = sum(state)

    if total_inventory < s:
        target_order = S - total_inventory
        return min(max(0, target_order), Q_MAX)
    else:
        return 0


def heuristic_fixed_order(state):
    """Fixed order quantity: always order the mean demand."""
    q_fixed = int(round(DEMAND_LAMBDA))
    return min(q_fixed, Q_MAX)


# ---------------------------------------------------------------------------
# Run single complexity level
# ---------------------------------------------------------------------------
def run_single_complexity(K, seeds):
    """Run benchmark for a single complexity level (K echelons)."""
    env = MultiEchelonInventory(K=K)
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
              f"reward={dp_reward:.2f}")
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
        env_dqn = MultiEchelonInventory(K=K)
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

        print(f"      seed={seed}: reward={reward:.2f}, time={metrics.wall_time:.2f}s")

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
    h_base_stock_reward = evaluate_heuristic(env, heuristic_base_stock,
                                              n_episodes=200, horizon=EPISODE_HORIZON)
    h_s_S_reward = evaluate_heuristic(env, heuristic_s_S,
                                       n_episodes=200, horizon=EPISODE_HORIZON)
    h_fixed_reward = evaluate_heuristic(env, heuristic_fixed_order,
                                         n_episodes=200, horizon=EPISODE_HORIZON)

    result['heuristics'] = {
        'base_stock': h_base_stock_reward,
        's_S': h_s_S_reward,
        'fixed_order': h_fixed_reward,
    }
    print(f"      Base-stock={h_base_stock_reward:.2f}, (s,S)={h_s_S_reward:.2f}, "
          f"Fixed={h_fixed_reward:.2f}")

    # --- Decomposition (using DQN policy) ---
    if all_curves:
        print("    Running decomposition analysis...")
        env_decomp = MultiEchelonInventory(K=K)
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
            decompose_fn=inventory_decompose
        )
        result['decomposition'] = decomp.component_rewards
        result['stockout_rate'] = decomp.component_rewards.get('stockout', 0)

    return result


# ---------------------------------------------------------------------------
# Scaling sweep
# ---------------------------------------------------------------------------
def run_scaling_sweep():
    """Run benchmark across all complexity levels."""
    print("=" * 70)
    print("  Multi-Echelon Inventory: Scaling Analysis")
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

    ax.set_xlabel('Number of Echelons (K)', fontsize=11)
    ax.set_ylabel('Episode Reward (negative cost)', fontsize=11)
    ax.set_title('Inventory: Reward vs Complexity', fontsize=12)
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

        if r['dp_reward']:
            ax.axhline(r['dp_reward'], color=colors[i], linestyle='--',
                      linewidth=1, alpha=0.7)

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Episode Reward (negative cost)', fontsize=11)
    ax.set_title('Inventory: Learning Curves by Complexity', fontsize=12)
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
    tex = make_scaling_table(results, 'K', 'Inventory')
    out_path = OUTPUT_DIR / f'{SCRIPT_NAME}_results.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"LaTeX table saved to {out_path}")

    # Decomposition table for largest complexity
    largest = results[-1]
    if 'decomposition' in largest and largest['decomposition']:
        lines = []
        lines.append(r'\begin{tabular}{lr}')
        lines.append(r'\toprule')
        lines.append(r'Cost Component & Mean per Episode \\')
        lines.append(r'\midrule')

        decomp = largest['decomposition']
        lines.append(f"Total Cost & ${decomp.get('total_cost', 0):.2f}$ \\\\")
        lines.append(f"Holding Cost & ${decomp.get('holding_cost', 0):.2f}$ \\\\")
        lines.append(f"Backorder Cost & ${decomp.get('backorder_cost', 0):.2f}$ \\\\")
        lines.append(f"Order Cost & ${decomp.get('order_cost', 0):.2f}$ \\\\")
        lines.append(f"Stockout Rate & ${decomp.get('stockout', 0):.2%}$ \\\\")
        lines.append(f"Avg Order Qty & ${decomp.get('order_qty', 0):.2f}$ \\\\")

        K = largest['complexity']
        for k in range(K):
            key = f'echelon_{k}_inv'
            if key in decomp:
                lines.append(f"Echelon {k} Avg Inv & ${decomp[key]:.2f}$ \\\\")

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
    print("  " + "-" * 70)
    print(f"  {'K':>3} | {'States':>10} | {'DP Reward':>10} | {'DQN Reward':>18} | "
          f"{'Q-Error':>8} | {'Agr.':>6}")
    print("  " + "-" * 70)

    for r in results:
        K = r['complexity']
        states = r['states']
        dp_r = f"{r['dp_reward']:.2f}" if r['dp_reward'] else "---"
        dqn_r = f"{r['dqn_reward_mean']:.2f} +/- {r['dqn_reward_std']:.2f}"
        q_err = f"{r['q_error']:.3f}" if r['q_error'] else "---"
        agr = f"{r['agreement']:.1%}" if r['agreement'] else "---"
        print(f"  {K:>3} | {states:>10,} | {dp_r:>10} | {dqn_r:>18} | {q_err:>8} | {agr:>6}")

    print("  " + "-" * 70)

    # Per-seed results
    print("\n  Per-Seed DQN Rewards:")
    for r in results:
        K = r['complexity']
        rewards = r['dqn_rewards']
        print(f"    K={K}: " + ", ".join(f"{rw:.2f}" for rw in rewards))

    # Heuristic comparison
    print("\n  Heuristic Comparison:")
    print("  " + "-" * 55)
    print(f"  {'K':>3} | {'Base-stock':>12} | {'(s,S)':>10} | {'Fixed':>10}")
    print("  " + "-" * 55)
    for r in results:
        K = r['complexity']
        h = r['heuristics']
        print(f"  {K:>3} | {h['base_stock']:>12.2f} | {h['s_S']:>10.2f} | "
              f"{h['fixed_order']:>10.2f}")
    print("  " + "-" * 55)

    # Decomposition for largest K
    largest = results[-1]
    if 'decomposition' in largest and largest['decomposition']:
        print(f"\n  Cost Decomposition (K={largest['complexity']}, DQN policy):")
        decomp = largest['decomposition']
        print(f"    Total cost: {decomp.get('total_cost', 0):.2f}")
        print(f"    Holding cost: {decomp.get('holding_cost', 0):.2f}")
        print(f"    Backorder cost: {decomp.get('backorder_cost', 0):.2f}")
        print(f"    Order cost: {decomp.get('order_cost', 0):.2f}")
        print(f"    Stockout rate: {decomp.get('stockout', 0):.2%}")
        print(f"    Avg order qty: {decomp.get('order_qty', 0):.2f}")

        K = largest['complexity']
        for k in range(K):
            key = f'echelon_{k}_inv'
            if key in decomp:
                print(f"    Echelon {k} avg inventory: {decomp[key]:.2f}")

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
        print("Multi-Echelon Inventory Benchmark")
        print(f"Complexity sweep: K in {COMPLEXITY_SWEEP}")
        print(f"Seeds: {SEEDS}")

        results = run_scaling_sweep()
        make_figures(results)
        make_latex_tables(results)
        print_detailed_results(results)

        print("\nBenchmark complete.")
