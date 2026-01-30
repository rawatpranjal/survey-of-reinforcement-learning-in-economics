"""
Multi-Echelon Inventory Management Benchmark (Chapter 3)

Serial inventory system with K echelons. Echelon 1 faces stochastic customer demand.
Upstream echelons supply downstream. State space grows exponentially in K.
DP feasible for K <= 3. Compare DP, DQN, and classical heuristics (base-stock, (s,S), fixed-order).
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
I_MAX = 8               # Max inventory per echelon
Q_MAX = 5               # Max order quantity
K_VALUES = [2, 3, 4]    # Number of echelons (complexity dial)
DP_CUTOFF = 2           # DP only feasible for K <= 2

DEMAND_LAMBDA = 4.0     # Poisson demand rate
HOLDING_COST = 1.0      # Cost per unit held per period
BACKORDER_COST = 10.0   # Cost per unit backordered per period
ORDER_COST = 0.5        # Cost per unit ordered

GAMMA = 0.95
EPISODE_HORIZON = 20
EVAL_EPISODES = 50

# DQN training episodes (scales with problem size)
def get_dqn_episodes(K):
    if K <= 2:
        return 500
    elif K == 3:
        return 1000
    else:
        return 1500

# Demand support (for exact DP transition probabilities)
MAX_DEMAND = int(2 * DEMAND_LAMBDA + 10)


# =============================================================================
# Environment
# =============================================================================
class MultiEchelonInventory(EconBenchmark):
    """
    Serial multi-echelon inventory system.

    State: (I_1, ..., I_K) where I_k in {0, ..., I_MAX}
    Action: order quantity q in {0, ..., Q_MAX} for echelon K
    Demand: D ~ Poisson(DEMAND_LAMBDA) at echelon 1

    Dynamics per period:
    1. Customer demand D arrives at echelon 1
    2. Sales at echelon 1: min(I_1, D)
    3. Backorders: max(0, D - I_1)
    4. Each echelon k > 1 ships min(I_k, demand_from_{k-1}) to echelon k-1
    5. Order q arrives at echelon K from external supplier
    6. Reward: -(holding_cost * sum(I_k) + backorder_cost * backorders + order_cost * q)
    """

    dp_feasible = True

    def __init__(self, K, gamma=GAMMA, horizon=EPISODE_HORIZON):
        self.K = K
        self.complexity_param = K
        self.gamma = gamma
        self.horizon = horizon

        # State: tuple of K inventory levels
        self._num_states = (I_MAX + 1) ** K
        self._num_actions = Q_MAX + 1

        # Current state and time
        self.state = None
        self.t = 0

        # Demand distribution
        self.demand_probs = np.array([poisson.pmf(d, DEMAND_LAMBDA) for d in range(MAX_DEMAND + 1)])
        self.demand_probs /= self.demand_probs.sum()  # Normalize

        # Initial state: half-full inventory at all echelons
        self.initial_inventory = I_MAX // 2

        self.reset()

    def reset(self):
        """Reset to initial state tuple."""
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
        """Convert state tuple to flat index for DP."""
        idx = 0
        multiplier = 1
        for i in range(self.K):
            idx += state[i] * multiplier
            multiplier *= (I_MAX + 1)
        return idx

    def index_to_state(self, idx):
        """Convert flat index to state tuple."""
        state = []
        for _ in range(self.K):
            state.append(idx % (I_MAX + 1))
            idx //= (I_MAX + 1)
        return tuple(state)

    def state_to_features(self, state):
        """Convert state tuple to normalized feature vector."""
        return np.array(state, dtype=np.float32) / I_MAX

    def step(self, action):
        """
        Execute one step.

        Args:
            action: order quantity q in {0, ..., Q_MAX}

        Returns:
            (next_state_tuple, reward, done)
        """
        # Sample demand
        demand = np.random.choice(MAX_DEMAND + 1, p=self.demand_probs)

        # Compute next state and reward
        next_state, reward = self._transition(self.state, action, demand)

        self.state = next_state
        self.t += 1
        done = (self.t >= self.horizon)

        return next_state, reward, done

    def _transition(self, state, action, demand):
        """
        Compute deterministic transition given demand realization.

        Returns:
            (next_state, reward)
        """
        inventory = list(state)  # Current inventory levels
        q = action  # Order quantity

        # Echelon 1: satisfy customer demand
        sales = min(inventory[0], demand)
        backorders = max(0, demand - inventory[0])
        inventory[0] = max(0, inventory[0] - demand)

        # Upstream echelons ship to downstream
        # Each echelon k > 1 tries to replenish echelon k-1
        for k in range(1, self.K):
            # Deficit at echelon k-1 (try to restore to initial level)
            deficit = max(0, self.initial_inventory - inventory[k-1])
            shipment = min(inventory[k], deficit)
            inventory[k] -= shipment
            inventory[k-1] += shipment

        # Order arrives at echelon K (the most upstream)
        inventory[self.K - 1] = min(I_MAX, inventory[self.K - 1] + q)

        # Compute cost components
        holding_cost = HOLDING_COST * sum(inventory)
        backorder_cost = BACKORDER_COST * backorders
        order_cost = ORDER_COST * q

        reward = -(holding_cost + backorder_cost + order_cost)

        # Clip inventory to valid range
        next_state = tuple(min(I_MAX, max(0, inv)) for inv in inventory)

        return next_state, reward

    def transition_distribution(self, state, action):
        """
        Return list of (next_state, probability) for all possible demand realizations.
        """
        transitions = []
        for demand in range(MAX_DEMAND + 1):
            prob = self.demand_probs[demand]
            if prob > 1e-10:
                next_state, reward = self._transition(state, action, demand)
                transitions.append((next_state, prob))
        return transitions

    def expected_reward(self, state, action):
        """Return expected immediate reward over demand distribution."""
        expected_r = 0.0
        for demand in range(MAX_DEMAND + 1):
            prob = self.demand_probs[demand]
            if prob > 1e-10:
                _, reward = self._transition(state, action, demand)
                expected_r += prob * reward
        return expected_r


# =============================================================================
# Heuristic Policies
# =============================================================================
def heuristic_base_stock(state):
    """
    Newsvendor-style base-stock policy for echelon 1.

    Order up to S* = F^{-1}(b / (b + h)) where F is demand CDF.
    Critical fractile for Poisson(lambda) with b=10, h=1 -> S* ~ lambda + 2.
    """
    # Compute base-stock level using newsvendor critical fractile
    critical_fractile = BACKORDER_COST / (BACKORDER_COST + HOLDING_COST)
    S_star = poisson.ppf(critical_fractile, DEMAND_LAMBDA)
    S_star = int(min(S_star, I_MAX))

    # Total inventory position
    total_inventory = sum(state)

    # Order up to S_star
    target_order = max(0, S_star - total_inventory)
    return min(target_order, Q_MAX)


def heuristic_s_S(state):
    """
    (s, S) policy: if total inventory < s, order up to S; else order 0.

    Use s = lambda - 1, S = lambda + 3 as reasonable defaults.
    """
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
    """
    Fixed order quantity: always order the mean demand.
    """
    q_fixed = int(round(DEMAND_LAMBDA))
    return min(q_fixed, Q_MAX)


# =============================================================================
# Main Experiment
# =============================================================================
def main():
    results = {
        'K': [],
        'states': [],
        'dp_time': [],
        'dp_reward': [],
        'dqn_time': [],
        'dqn_reward': [],
        'dqn_std': [],
        'base_stock_reward': [],
        's_S_reward': [],
        'fixed_order_reward': [],
    }

    for K in K_VALUES:
        print(f"\n{'='*60}")
        print(f"K = {K} echelons")
        print(f"{'='*60}")

        env = MultiEchelonInventory(K=K)
        n_states = env.num_states
        n_actions = env.num_actions

        print(f"State space: {n_states}, Action space: {n_actions}")

        results['K'].append(K)
        results['states'].append(n_states)

        # -------------------------------
        # Dynamic Programming
        # -------------------------------
        if K <= DP_CUTOFF:
            print("\nRunning Value Iteration (DP)...")
            V, policy, dp_time = run_value_iteration(env, gamma=GAMMA)
            results['dp_time'].append(dp_time)

            print("Evaluating DP policy...")
            dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA, n_episodes=EVAL_EPISODES, horizon=EPISODE_HORIZON)
            results['dp_reward'].append(dp_reward)

            print(f"  DP time: {dp_time:.2f}s")
            print(f"  DP reward: {dp_reward:.2f}")
        else:
            print("\nSkipping DP (state space too large)")
            results['dp_time'].append(np.nan)
            results['dp_reward'].append(np.nan)

        # -------------------------------
        # DQN
        # -------------------------------
        print("\nRunning DQN...")
        dqn_episodes = get_dqn_episodes(K)
        dqn_times = []
        dqn_rewards = []

        for seed in DQN_SEEDS:
            env_dqn = MultiEchelonInventory(K=K)
            q_net, dqn_time = run_dqn(env_dqn, gamma=GAMMA, num_episodes=dqn_episodes,
                                      episode_horizon=EPISODE_HORIZON, seed=seed)
            dqn_reward = evaluate_dqn_policy(env, q_net, n_episodes=EVAL_EPISODES, horizon=EPISODE_HORIZON)
            dqn_times.append(dqn_time)
            dqn_rewards.append(dqn_reward)
            print(f"  seed={seed}: time={dqn_time:.2f}s, reward={dqn_reward:.2f}")

        results['dqn_time'].append(np.mean(dqn_times))
        results['dqn_reward'].append(np.mean(dqn_rewards))
        results['dqn_std'].append(np.std(dqn_rewards))

        print(f"  DQN time: {np.mean(dqn_times):.2f}s")
        print(f"  DQN reward: {np.mean(dqn_rewards):.2f} +/- {np.std(dqn_rewards):.2f}")

        # -------------------------------
        # Heuristics
        # -------------------------------
        print("\nEvaluating heuristics...")

        # Base-stock
        base_stock_reward = evaluate_heuristic(env, heuristic_base_stock, n_episodes=EVAL_EPISODES, horizon=EPISODE_HORIZON)
        results['base_stock_reward'].append(base_stock_reward)
        print(f"  Base-stock: {base_stock_reward:.2f}")

        # (s,S) policy
        s_S_reward = evaluate_heuristic(env, heuristic_s_S, n_episodes=EVAL_EPISODES, horizon=EPISODE_HORIZON)
        results['s_S_reward'].append(s_S_reward)
        print(f"  (s,S) policy: {s_S_reward:.2f}")

        # Fixed order
        fixed_order_reward = evaluate_heuristic(env, heuristic_fixed_order, n_episodes=EVAL_EPISODES, horizon=EPISODE_HORIZON)
        results['fixed_order_reward'].append(fixed_order_reward)
        print(f"  Fixed order: {fixed_order_reward:.2f}")

    # =========================================================================
    # Generate Figure
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    K_array = np.array(results['K'])
    states_array = np.array(results['states'])

    # Left panel: Computation time (log scale)
    dp_times_plot = [t if not np.isnan(t) else None for t in results['dp_time']]
    dqn_times_plot = results['dqn_time']

    # Plot DP where available
    dp_K = [K_array[i] for i in range(len(K_array)) if dp_times_plot[i] is not None]
    dp_t = [dp_times_plot[i] for i in range(len(K_array)) if dp_times_plot[i] is not None]
    if dp_K:
        ax1.semilogy(dp_K, dp_t, 'o-', label='DP', linewidth=2, markersize=8)

    ax1.semilogy(K_array, dqn_times_plot, 's-', label='DQN', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Echelons (K)', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds, log scale)', fontsize=12)
    ax1.set_title('Scaling: Computation Time', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(K_array)

    # Right panel: Reward comparison
    dp_K_reward = [K_array[i] for i in range(len(K_array)) if not np.isnan(results['dp_reward'][i])]
    dp_r = [results['dp_reward'][i] for i in range(len(K_array)) if not np.isnan(results['dp_reward'][i])]
    if dp_K_reward:
        ax2.plot(dp_K_reward, dp_r, 'o-', label='DP', linewidth=2, markersize=8)

    ax2.errorbar(K_array, results['dqn_reward'], yerr=results['dqn_std'],
                 fmt='s-', label='DQN', linewidth=2, markersize=8, capsize=5)
    ax2.plot(K_array, results['base_stock_reward'], '^--', label='Base-stock', linewidth=2, markersize=8, alpha=0.7)
    ax2.plot(K_array, results['s_S_reward'], 'v--', label='(s,S)', linewidth=2, markersize=8, alpha=0.7)
    ax2.plot(K_array, results['fixed_order_reward'], 'd--', label='Fixed order', linewidth=2, markersize=8, alpha=0.7)
    ax2.set_xlabel('Number of Echelons (K)', fontsize=12)
    ax2.set_ylabel('Average Reward per Episode', fontsize=12)
    ax2.set_title('Performance Comparison', fontsize=13)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(K_array)

    plt.tight_layout()

    fig_path = OUTPUT_DIR / "inventory_scaling.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")

    # =========================================================================
    # Generate LaTeX Table
    # =========================================================================
    table_lines = []
    table_lines.append(r"\begin{tabular}{lrrrrrrr}")
    table_lines.append(r"\hline")
    table_lines.append(r"$K$ & States & DP Time & DP Reward & DQN Time & DQN Reward & Base-stock & (s,S) \\")
    table_lines.append(r"\hline")

    for i, K in enumerate(results['K']):
        states = results['states'][i]

        if np.isnan(results['dp_time'][i]):
            dp_time_str = "---"
            dp_reward_str = "---"
        else:
            dp_time_str = f"{results['dp_time'][i]:.2f}"
            dp_reward_str = f"{results['dp_reward'][i]:.2f}"

        dqn_time_str = f"{results['dqn_time'][i]:.2f}"
        dqn_reward_str = f"{results['dqn_reward'][i]:.2f}"
        base_stock_str = f"{results['base_stock_reward'][i]:.2f}"
        s_S_str = f"{results['s_S_reward'][i]:.2f}"

        table_lines.append(
            f"{K} & {states} & {dp_time_str} & {dp_reward_str} & "
            f"{dqn_time_str} & {dqn_reward_str} & {base_stock_str} & {s_S_str} \\\\"
        )

    table_lines.append(r"\hline")
    table_lines.append(r"\end{tabular}")

    table_content = "\n".join(table_lines)
    table_path = OUTPUT_DIR / "inventory_results.tex"
    with open(table_path, 'w') as f:
        f.write(table_content)

    print(f"Table saved: {table_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
