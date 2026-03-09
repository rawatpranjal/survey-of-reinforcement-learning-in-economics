#!/usr/bin/env python3
"""
structural_pricing_misra.py
Chapter 7: Economic Bandits
UCB with Partial Identification for Dynamic Pricing — Misra, Schwartz, Abernethy (2019)

Implements UCB-PI which leverages economic structure (WARP: if consumer buys at price p,
they would buy at any price p' < p) to learn demand bounds and turn off dominated prices.

Key insight: Partial identification from purchase decisions bounds segment valuations,
enabling tighter profit bounds and faster elimination of suboptimal prices.

Reference: Misra, Schwartz & Abernethy (2019), "Dynamic Online Pricing with Incomplete
Information Using Multi-Armed Bandit Experiments", Marketing Science, Vol. 38, No. 2.

Implementation follows paper's setup (Section 4):
- K = 100 prices (1-cent grid from $0 to $1)
- S = 1,000 consumer segments
- T = 200,000 rounds
- δ = 0.1 (10% within-segment heterogeneity)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm
import sys

# =============================================================================
# Configuration — Matching Paper's Setup (Section 4, p. 238)
# =============================================================================
K = 100                      # Number of discrete prices (paper: 100)
T = 200_000                  # Time horizon (paper: 200,000)
N_SEEDS = 10                 # Number of replications
N_SEGMENTS = 1000            # Number of consumer segments (paper: 1,000)
DELTA_TRUE = 0.1             # True within-segment heterogeneity (paper: 0.1)

# Price grid: $0.01 to $1.00 in 1-cent increments (normalized)
PRICES = np.linspace(0.01, 1.0, K)  # Paper uses [0,1] range
V_L, V_H = 0.0, 1.0          # Valuation bounds known to firm

np.random.seed(42)

# =============================================================================
# Segmented Demand Model (Misra et al. Section 2.1)
# =============================================================================
class SegmentedDemandModel:
    """
    Demand model with consumer segments per Misra et al. (2019).

    Each segment s has midpoint valuation v_s drawn uniformly from [V_L+δ, V_H-δ].
    Consumer i in segment s has v_i ~ Uniform(v_s - δ, v_s + δ).
    Consumer buys iff v_i >= p (WARP: Weak Axiom of Revealed Preference).

    Key equation (Equation 1): v_i ∈ [v_s - δ, v_s + δ] for all i in segment s.
    """

    def __init__(self, n_segments: int, delta: float, seed: int = 42):
        self.n_segments = n_segments
        self.delta = delta

        # Draw segment midpoint valuations uniformly from [delta, 1-delta]
        rng = np.random.RandomState(seed)
        self.segment_valuations = rng.uniform(V_L + delta, V_H - delta, n_segments)

        # Equal segment weights
        self.segment_weights = np.ones(n_segments) / n_segments

    def true_demand(self, price: float) -> float:
        """Compute true aggregate demand at price."""
        demand = 0.0
        for s in range(self.n_segments):
            v_s = self.segment_valuations[s]
            if price <= v_s - self.delta:
                prob = 1.0
            elif price >= v_s + self.delta:
                prob = 0.0
            else:
                prob = (v_s + self.delta - price) / (2 * self.delta)
            demand += self.segment_weights[s] * prob
        return demand

    def true_profit(self, price: float) -> float:
        """Expected profit at price p (marginal cost = 0)."""
        return price * self.true_demand(price)


# Create canonical demand model for computing optimal arm
canonical_demand = SegmentedDemandModel(N_SEGMENTS, DELTA_TRUE, seed=42)
TRUE_PROFITS = np.array([canonical_demand.true_profit(p) for p in PRICES])
OPTIMAL_ARM = np.argmax(TRUE_PROFITS)
OPTIMAL_PROFIT = TRUE_PROFITS[OPTIMAL_ARM]
OPTIMAL_PRICE = PRICES[OPTIMAL_ARM]


# =============================================================================
# Algorithm Implementations (Vectorized for Speed)
# =============================================================================

class Oracle:
    """Oracle: always plays the optimal price."""

    def __init__(self, n_arms: int, optimal_arm: int):
        self.optimal_arm = optimal_arm
        self.counts = np.zeros(n_arms)

    def select_arm(self, t: int) -> int:
        return self.optimal_arm

    def update(self, arm: int, reward: float, segment_id: int, purchased: bool):
        self.counts[arm] += 1


class UCB1:
    """Standard UCB1 algorithm."""

    def __init__(self, n_arms: int, prices: np.ndarray):
        self.n_arms = n_arms
        self.prices = prices
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self, t: int) -> int:
        if t < self.n_arms:
            return t
        ucb = self.values + np.sqrt(2 * np.log(t + 1) / np.maximum(self.counts, 1))
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float, segment_id: int, purchased: bool):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class UCB_PI:
    """
    UCB with Partial Identification (Misra, Schwartz & Abernethy 2019).

    Key features per paper:
    1. Tracks segment-level demand observations to bound valuations
    2. Uses WARP: if buy at p, would buy at any p' < p
    3. Computes demand bounds via partial identification (Equations 5-6)
    4. Turns off dominated prices (where UB(profit) <= max LB(profit))
    5. Scales exploration bonus by price (Equation 7)
    """

    def __init__(self, prices: np.ndarray, n_segments: int, segment_weights: np.ndarray):
        self.prices = prices
        self.K = len(prices)
        self.n_segments = n_segments
        self.segment_weights = segment_weights

        # Per-price statistics
        self.counts = np.zeros(self.K)
        self.values = np.zeros(self.K)
        self.sum_sq = np.zeros(self.K)

        # Per-segment partial identification bounds
        # Paper naming: p_min = highest price where D̂=1, p_max = lowest price where D̂=0
        self.segment_p_min = np.full(n_segments, V_L)
        self.segment_p_max = np.full(n_segments, V_H)

        # Per-(segment, price) observation tracking for demand rates
        self.obs_counts = np.zeros((n_segments, self.K), dtype=int)
        self.obs_purchases = np.zeros((n_segments, self.K), dtype=int)

        self.delta_hat = (V_H - V_L) / 2
        self.dominated_counts = []

    def _update_partial_identification(self, arm: int, segment_id: int, purchased: bool):
        """Update segment valuation bounds using demand rates (paper Section 2.3.1).

        Tracks per-(segment, price) purchase counts to compute empirical demand rates D̂(p_k)_s.
        p^min_s = highest price where D̂ = 1 (all observed consumers purchased)
        p^max_s = lowest price where D̂ = 0 (no observed consumer purchased)
        """
        self.obs_counts[segment_id, arm] += 1
        if purchased:
            self.obs_purchases[segment_id, arm] += 1

        # Recompute bounds for this segment from demand rates
        s = segment_id
        counts_s = self.obs_counts[s]
        purchases_s = self.obs_purchases[s]
        observed = counts_s > 0

        # p^min = highest price where D̂ = 1 (all observed consumers purchased)
        full_buy = observed & (purchases_s == counts_s)
        if full_buy.any():
            self.segment_p_min[s] = self.prices[np.where(full_buy)[0][-1]]
        else:
            self.segment_p_min[s] = V_L

        # p^max = lowest price where D̂ = 0 (no observed consumer purchased)
        no_buy = observed & (purchases_s == 0)
        if no_buy.any():
            self.segment_p_max[s] = self.prices[np.where(no_buy)[0][0]]
        else:
            self.segment_p_max[s] = V_H

    def _estimate_delta(self) -> float:
        """Estimate δ using bias-corrected maximum (paper Section 2.3.1, p. 235).

        Paper naming convention:
          segment_p_min[s] = p^min_s = highest price where D̂=1 → v_s - δ
          segment_p_max[s] = p^max_s = lowest price where D̂=0 → v_s + δ
        Well-identified segments have both bounds informative and properly ordered.
        """
        has_lb = self.segment_p_min > V_L
        has_ub = self.segment_p_max < V_H
        crossed = has_lb & has_ub & (self.segment_p_max > self.segment_p_min)

        if crossed.sum() >= 2:
            delta_estimates = (self.segment_p_max[crossed] - self.segment_p_min[crossed]) / 2
            delta_max = delta_estimates.max()
            mean_delta = delta_estimates.mean()
            # Paper's bias correction (Karunamuni & Alberts 2005):
            # γ̂ = max - mean; δ̂ = max + γ̂ (converges from above)
            gamma_hat = delta_max - mean_delta
            self.delta_hat = delta_max + gamma_hat
        elif crossed.sum() == 1:
            self.delta_hat = (self.segment_p_max[crossed] - self.segment_p_min[crossed])[0] / 2
        # else: initial delta_hat = (V_H - V_L) / 2 persists
        return self.delta_hat

    def _compute_profit_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute profit bounds for all prices (vectorized)."""
        # Segment midpoints: (p^min + p^max)/2 ≈ v_s
        v_hat = (self.segment_p_min + self.segment_p_max) / 2  # (S,)
        v_min = v_hat - self.delta_hat  # (S,)
        v_max = v_hat + self.delta_hat  # (S,)

        # For each price, compute demand bounds
        # v_min >= price means segment definitely buys
        # v_max >= price means segment might buy
        lb_demand = np.zeros(self.K)
        ub_demand = np.zeros(self.K)

        for k in range(self.K):
            lb_demand[k] = self.segment_weights[v_min >= self.prices[k]].sum()
            ub_demand[k] = self.segment_weights[v_max >= self.prices[k]].sum()

        lb_profits = self.prices * lb_demand
        ub_profits = self.prices * ub_demand

        return lb_profits, ub_profits

    def select_arm(self, t: int) -> int:
        """Select price using UCB-PI index (Equation 7)."""
        if t < self.K:
            return t

        self._estimate_delta()
        lb_profits, ub_profits = self._compute_profit_bounds()
        max_lb = lb_profits.max()

        # Compute UCB-PI index
        ucb_pi = np.full(self.K, -np.inf)
        dominated = ub_profits <= max_lb

        active = ~dominated
        if active.any():
            exploration = self.prices[active] * np.sqrt(
                2 * np.log(t + 1) / np.maximum(self.counts[active], 1)
            )
            ucb_pi[active] = self.values[active] + exploration

        self.dominated_counts.append(dominated.sum())
        return int(np.argmax(ucb_pi))

    def update(self, arm: int, reward: float, segment_id: int, purchased: bool):
        """Update statistics after observing outcome."""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.sum_sq[arm] += reward ** 2
        self._update_partial_identification(arm, segment_id, purchased)


class UCB_PI_Tuned(UCB_PI):
    """UCB-PI with variance tuning (Equation 9)."""

    def select_arm(self, t: int) -> int:
        if t < self.K:
            return t

        self._estimate_delta()
        lb_profits, ub_profits = self._compute_profit_bounds()
        max_lb = lb_profits.max()

        ucb_pi = np.full(self.K, -np.inf)
        dominated = ub_profits <= max_lb
        active = ~dominated

        if active.any():
            # Compute variance term
            counts_active = np.maximum(self.counts[active], 1)
            mean_sq = self.sum_sq[active] / counts_active
            variance = np.maximum(0, mean_sq - self.values[active] ** 2)
            V_kt = variance + np.sqrt(2 * np.log(t + 1) / counts_active)

            exploration = 2 * self.prices[active] * self.delta_hat * np.sqrt(
                (np.log(t + 1) / counts_active) * np.minimum(0.25, V_kt)
            )
            ucb_pi[active] = self.values[active] + exploration

        self.dominated_counts.append(dominated.sum())
        return int(np.argmax(ucb_pi))


class LearnThenEarn:
    """Learn-Then-Earn baseline (Section 4.2)."""

    def __init__(self, n_arms: int, prices: np.ndarray, tau: float = 0.05, T_total: int = T):
        self.n_arms = n_arms
        self.prices = prices
        self.T_explore = int(tau * T_total)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.best_arm = None

    def select_arm(self, t: int) -> int:
        if t < self.T_explore:
            return np.random.randint(self.n_arms)
        else:
            if self.best_arm is None:
                self.best_arm = int(np.argmax(self.values))
            return self.best_arm

    def update(self, arm: int, reward: float, segment_id: int, purchased: bool):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSampling:
    """Thompson Sampling for pricing."""

    def __init__(self, n_arms: int, prices: np.ndarray):
        self.n_arms = n_arms
        self.prices = prices
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.counts = np.zeros(n_arms)

    def select_arm(self, t: int) -> int:
        theta = np.random.beta(self.alpha, self.beta)
        expected_profit = self.prices * theta
        return int(np.argmax(expected_profit))

    def update(self, arm: int, reward: float, segment_id: int, purchased: bool):
        self.counts[arm] += 1
        if purchased:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


# =============================================================================
# Print Configuration
# =============================================================================
print("=" * 70)
print("UCB WITH PARTIAL IDENTIFICATION — MISRA ET AL. (2019)")
print("=" * 70)
print()
print("Configuration (matching paper's Section 4):")
print(f"  Number of prices (K):       {K}")
print(f"  Time horizon (T):           {T:,}")
print(f"  Number of seeds:            {N_SEEDS}")
print(f"  Price range:                [{PRICES[0]:.2f}, {PRICES[-1]:.2f}]")
print(f"  Number of segments (S):     {N_SEGMENTS:,}")
print(f"  True delta (unknown):       {DELTA_TRUE}")
print()
print("Demand Model: Segmented with within-segment heterogeneity")
print(f"  Segment valuations v_s ~ Uniform([{V_L + DELTA_TRUE:.1f}, {V_H - DELTA_TRUE:.1f}])")
print(f"  Consumer valuation v_i ~ Uniform(v_s - δ, v_s + δ)")
print()

# Print demand curve at selected prices
print("Expected Profit by Price (selected):")
print("-" * 60)
print(f"{'Price':>8}  {'Demand':>10}  {'E[Profit]':>12}  {'Optimal':>8}")
print("-" * 60)
sample_indices = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
for i in sample_indices:
    if i < K:
        opt_marker = "  <--" if i == OPTIMAL_ARM else ""
        print(f"{PRICES[i]:8.2f}  {canonical_demand.true_demand(PRICES[i]):10.4f}  "
              f"{TRUE_PROFITS[i]:12.4f}{opt_marker}")
print("-" * 60)
print(f"Optimal price: {OPTIMAL_PRICE:.2f} (arm {OPTIMAL_ARM}), "
      f"expected profit: {OPTIMAL_PROFIT:.4f}")
print()


# =============================================================================
# Run Experiments
# =============================================================================

def run_experiment(seed: int, pbar: tqdm = None) -> dict:
    """Run one replication with all algorithms."""
    rng = np.random.RandomState(seed)

    # Create demand model with this seed
    demand_model = SegmentedDemandModel(N_SEGMENTS, DELTA_TRUE, seed=seed)

    # Compute true optimal for this realization
    true_profits_seed = np.array([demand_model.true_profit(p) for p in PRICES])
    opt_arm = np.argmax(true_profits_seed)

    # Pre-generate all random data for speed
    segment_ids = rng.choice(N_SEGMENTS, size=T, p=demand_model.segment_weights)
    # Pre-generate consumer valuations: v_i = v_s + uniform(-delta, delta)
    valuation_offsets = rng.uniform(-DELTA_TRUE, DELTA_TRUE, size=T)

    # Initialize algorithms
    oracle = Oracle(K, opt_arm)
    ucb1 = UCB1(K, PRICES)
    ucb_pi = UCB_PI(PRICES, N_SEGMENTS, demand_model.segment_weights)
    ucb_pi_tuned = UCB_PI_Tuned(PRICES, N_SEGMENTS, demand_model.segment_weights)
    lte = LearnThenEarn(K, PRICES, tau=0.05, T_total=T)
    thompson = ThompsonSampling(K, PRICES)

    algorithms = {
        'oracle': oracle,
        'ucb1': ucb1,
        'ucb_pi': ucb_pi,
        'ucb_pi_tuned': ucb_pi_tuned,
        'lte': lte,
        'thompson': thompson,
    }

    # Track cumulative profits (subsampled for memory)
    sample_interval = 1000
    n_samples = T // sample_interval
    profits = {name: np.zeros(n_samples) for name in algorithms}
    cumulative = {name: 0.0 for name in algorithms}
    arms_selected = {name: np.zeros(K) for name in algorithms}

    # Main simulation loop
    for t in range(T):
        segment_id = segment_ids[t]
        v_s = demand_model.segment_valuations[segment_id]
        v_i = v_s + valuation_offsets[t]

        for name, alg in algorithms.items():
            arm = alg.select_arm(t)
            price = PRICES[arm]

            # Consumer buys iff v_i >= price (WARP)
            purchased = (v_i >= price)
            reward = price if purchased else 0.0

            alg.update(arm, reward, segment_id, purchased)

            cumulative[name] += reward
            arms_selected[name][arm] += 1

        # Record at sample points
        if (t + 1) % sample_interval == 0:
            idx = (t + 1) // sample_interval - 1
            for name in algorithms:
                profits[name][idx] = cumulative[name]

        # Update progress bar
        if pbar is not None and t % 10000 == 0:
            pbar.update(10000)

    # Final update for progress bar
    if pbar is not None:
        pbar.update(T % 10000)

    # Get final statistics
    final_best = {
        'ucb1': int(np.argmax(ucb1.values)),
        'ucb_pi': int(np.argmax(ucb_pi.values)),
        'ucb_pi_tuned': int(np.argmax(ucb_pi_tuned.values)),
        'lte': lte.best_arm if lte.best_arm is not None else int(np.argmax(lte.values)),
        'thompson': int(np.argmax(thompson.prices * thompson.alpha / (thompson.alpha + thompson.beta))),
    }

    delta_estimates = {
        'ucb_pi': ucb_pi.delta_hat,
        'ucb_pi_tuned': ucb_pi_tuned.delta_hat,
    }

    dominated_final = {
        'ucb_pi': ucb_pi.dominated_counts[-1] if ucb_pi.dominated_counts else 0,
        'ucb_pi_tuned': ucb_pi_tuned.dominated_counts[-1] if ucb_pi_tuned.dominated_counts else 0,
    }

    # Sample dominated counts for plotting
    dom_sample_interval = 5000
    dominated_over_time = {
        'ucb_pi': ucb_pi.dominated_counts[::dom_sample_interval],
        'ucb_pi_tuned': ucb_pi_tuned.dominated_counts[::dom_sample_interval],
    }

    return {
        'profits': profits,
        'arms': arms_selected,
        'final_best': final_best,
        'delta_estimates': delta_estimates,
        'dominated_final': dominated_final,
        'dominated_over_time': dominated_over_time,
        'opt_arm': opt_arm,
        'cumulative': cumulative,
    }


print("Running experiments...")
print("-" * 50)
sys.stdout.flush()

# Collect results with progress bar
all_results = []
total_iterations = N_SEEDS * T

with tqdm(total=total_iterations, desc="Simulating", unit="rounds",
          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
    for seed in range(N_SEEDS):
        pbar.set_description(f"Seed {seed}/{N_SEEDS}")
        results = run_experiment(seed, pbar)
        all_results.append(results)
        tqdm.write(f"  Seed {seed}: UCB-PI δ̂={results['delta_estimates']['ucb_pi']:.3f}, "
                   f"dominated={results['dominated_final']['ucb_pi']}/{K}")

# Aggregate results
sample_interval = 1000
time_points = np.arange(sample_interval, T + 1, sample_interval)

alg_names = ['oracle', 'ucb1', 'ucb_pi', 'ucb_pi_tuned', 'lte', 'thompson']
profit_arrays = {name: np.array([r['profits'][name] for r in all_results])
                 for name in alg_names}

mean_profits = {name: arr.mean(axis=0) for name, arr in profit_arrays.items()}
se_profits = {name: arr.std(axis=0) / np.sqrt(N_SEEDS) for name, arr in profit_arrays.items()}

# Final cumulative profits
final_profits = {name: np.mean([r['cumulative'][name] for r in all_results])
                 for name in alg_names}
final_profits_se = {name: np.std([r['cumulative'][name] for r in all_results]) / np.sqrt(N_SEEDS)
                    for name in alg_names}

# Aggregate arm selections
arms_arrays = {name: np.array([r['arms'][name] for r in all_results])
               for name in alg_names}
mean_arms = {name: arr.mean(axis=0) for name, arr in arms_arrays.items()}

# Per-seed fraction at per-seed optimal arm
frac_optimal = {}
for name in alg_names:
    fracs = [r['arms'][name][r['opt_arm']] / T for r in all_results]
    frac_optimal[name] = np.mean(fracs)

# Compute regret
regret = {name: mean_profits['oracle'] - mean_profits[name]
          for name in alg_names if name != 'oracle'}

# Final best arm accuracy
final_best_correct = {
    name: sum(1 for r in all_results if r['final_best'].get(name) == r['opt_arm'])
    for name in ['ucb1', 'ucb_pi', 'ucb_pi_tuned', 'lte', 'thompson']
}

# Delta estimation
delta_estimates_mean = {
    'ucb_pi': np.mean([r['delta_estimates']['ucb_pi'] for r in all_results]),
    'ucb_pi_tuned': np.mean([r['delta_estimates']['ucb_pi_tuned'] for r in all_results]),
}
delta_estimates_se = {
    'ucb_pi': np.std([r['delta_estimates']['ucb_pi'] for r in all_results]) / np.sqrt(N_SEEDS),
    'ucb_pi_tuned': np.std([r['delta_estimates']['ucb_pi_tuned'] for r in all_results]) / np.sqrt(N_SEEDS),
}

# Dominated counts
dominated_final_mean = {
    'ucb_pi': np.mean([r['dominated_final']['ucb_pi'] for r in all_results]),
    'ucb_pi_tuned': np.mean([r['dominated_final']['ucb_pi_tuned'] for r in all_results]),
}

# Ex-post profit ratios (paper's main metric)
ex_post_ratios = {}
for name in alg_names:
    ratios = []
    for r in all_results:
        oracle_profit = r['cumulative']['oracle']
        alg_profit = r['cumulative'][name]
        ratios.append(alg_profit / oracle_profit if oracle_profit > 0 else 0)
    ex_post_ratios[name] = {
        'mean': np.mean(ratios),
        'std': np.std(ratios),
        'min': np.min(ratios),
        'max': np.max(ratios),
    }


# =============================================================================
# Print Results
# =============================================================================
print()
print("=" * 70)
print("RESULTS: ALGORITHM COMPARISON")
print("=" * 70)
print()

# Ex-post profit ratio table (paper's primary metric)
print("Ex Post Profit Ratio (Algorithm / Oracle):")
print("-" * 85)
print(f"{'Algorithm':<16}  {'Mean':>10}  {'Std Dev':>10}  {'Min':>10}  {'Max':>10}  {'Range':>12}")
print("-" * 85)
for name, label in [('oracle', 'Oracle'), ('ucb_pi_tuned', 'UCB-PI-tuned'),
                    ('ucb_pi', 'UCB-PI'), ('ucb1', 'UCB1'),
                    ('lte', 'LTE (5%)'), ('thompson', 'Thompson')]:
    r = ex_post_ratios[name]
    range_str = f"[{100*r['min']:.0f}-{100*r['max']:.0f}%]"
    print(f"{label:<16}  {100*r['mean']:>10.1f}%  {100*r['std']:>10.1f}%  "
          f"{100*r['min']:>10.1f}%  {100*r['max']:>10.1f}%  {range_str:>12}")
print("-" * 85)
print()

# Cumulative profit table
print(f"Cumulative Profit at T = {T:,}:")
print("-" * 85)
print(f"{'Algorithm':<16}  {'Profit':>14}  {'Std Err':>10}  {'% of Oracle':>12}  {'Profit Lost':>14}")
print("-" * 85)
for name, label in [('oracle', 'Oracle'), ('ucb_pi_tuned', 'UCB-PI-tuned'),
                    ('ucb_pi', 'UCB-PI'), ('ucb1', 'UCB1'),
                    ('lte', 'LTE (5%)'), ('thompson', 'Thompson')]:
    profit = final_profits[name]
    se = final_profits_se[name]
    pct = 100 * profit / final_profits['oracle']
    lost = final_profits['oracle'] - profit
    print(f"{label:<16}  {profit:>14,.0f}  {se:>10,.0f}  {pct:>12.2f}%  {lost:>14,.0f}")
print("-" * 85)
print()

# Fraction at optimal
print("Fraction of Time at Optimal Price:")
print("-" * 65)
print(f"{'Algorithm':<16}  {'Frac Optimal':>14}  {'Final Best = Opt':>18}")
print("-" * 65)
for name, label in [('oracle', 'Oracle'), ('ucb_pi_tuned', 'UCB-PI-tuned'),
                    ('ucb_pi', 'UCB-PI'), ('ucb1', 'UCB1'),
                    ('lte', 'LTE (5%)'), ('thompson', 'Thompson')]:
    frac = frac_optimal[name]
    if name in final_best_correct:
        correct = f"{final_best_correct[name]}/{N_SEEDS}"
    else:
        correct = "N/A"
    print(f"{label:<16}  {frac:>14.4f}  {correct:>18}")
print("-" * 65)
print()

# UCB-PI diagnostics
print("UCB-PI Partial Identification Diagnostics:")
print("-" * 65)
print(f"True δ (unknown to algorithm):       {DELTA_TRUE:.3f}")
print(f"UCB-PI estimated δ:                  {delta_estimates_mean['ucb_pi']:.3f} ± "
      f"{delta_estimates_se['ucb_pi']:.3f}")
print(f"UCB-PI-tuned estimated δ:            {delta_estimates_mean['ucb_pi_tuned']:.3f} ± "
      f"{delta_estimates_se['ucb_pi_tuned']:.3f}")
print()
print(f"Avg dominated prices at T (UCB-PI):       {dominated_final_mean['ucb_pi']:.1f} / {K}")
print(f"Avg dominated prices at T (UCB-PI-tuned): {dominated_final_mean['ucb_pi_tuned']:.1f} / {K}")
print("-" * 65)
print()

# Regret scaling
print("Profit Loss Scaling Analysis (Loss / log(t)):")
print("-" * 85)
checkpoints = [10000, 25000, 50000, 100000, 200000]
checkpoint_indices = [(c // sample_interval) - 1 for c in checkpoints]
print(f"{'t':>10}  {'UCB-PI-tuned':>14}  {'UCB-PI':>12}  {'UCB1':>12}  {'LTE':>12}  {'Thompson':>12}")
print("-" * 85)
for t_check, idx in zip(checkpoints, checkpoint_indices):
    if idx < len(regret['ucb_pi']):
        print(f"{t_check:>10}  "
              f"{regret['ucb_pi_tuned'][idx] / np.log(t_check):>14.1f}  "
              f"{regret['ucb_pi'][idx] / np.log(t_check):>12.1f}  "
              f"{regret['ucb1'][idx] / np.log(t_check):>12.1f}  "
              f"{regret['lte'][idx] / np.log(t_check):>12.1f}  "
              f"{regret['thompson'][idx] / np.log(t_check):>12.1f}")
print("-" * 85)
print()


# =============================================================================
# Generate Figures
# =============================================================================
print("Generating figures...")

# Figure 1: Cumulative Profit
fig, ax = plt.subplots(figsize=(10, 6))

colors = {'oracle': 'black', 'ucb1': 'C0', 'ucb_pi': 'C1', 'ucb_pi_tuned': 'C3',
          'lte': 'C2', 'thompson': 'C4'}
labels_plot = {'oracle': 'Oracle', 'ucb1': 'UCB1', 'ucb_pi': 'UCB-PI',
               'ucb_pi_tuned': 'UCB-PI-tuned', 'lte': 'LTE (5%)', 'thompson': 'Thompson'}

for name in ['oracle', 'ucb_pi_tuned', 'ucb_pi', 'ucb1', 'lte', 'thompson']:
    lw = 2 if name == 'oracle' else 1.5
    ls = '--' if name == 'oracle' else '-'
    ax.plot(time_points, mean_profits[name], label=labels_plot[name], color=colors[name],
            linewidth=lw, linestyle=ls)
    if name != 'oracle':
        ax.fill_between(time_points, mean_profits[name] - 2*se_profits[name],
                        mean_profits[name] + 2*se_profits[name], alpha=0.15, color=colors[name])

ax.set_xlabel('Customers $t$', fontsize=12)
ax.set_ylabel('Cumulative Profit', fontsize=12)
ax.set_title(f'Dynamic Pricing with Partial Identification (K={K}, S={N_SEGMENTS:,})', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, T)
ax.set_ylim(0, None)

plt.tight_layout()
fig.savefig('/Users/pranjal/Code/rl/ch07_bandits/sims/structural_pricing_misra_profit.png',
            dpi=300, bbox_inches='tight')
print("  Saved: structural_pricing_misra_profit.png")

# Figure 2: Profit Loss (Regret)
fig, ax = plt.subplots(figsize=(10, 6))

for name in ['ucb_pi_tuned', 'ucb_pi', 'ucb1', 'lte', 'thompson']:
    ax.plot(time_points, regret[name], label=labels_plot[name], color=colors[name], linewidth=1.5)

midpoint = len(time_points) // 2
c_log = regret['ucb_pi_tuned'][midpoint] / np.log(time_points[midpoint])
ax.plot(time_points, c_log * np.log(time_points), '--', color='gray',
        label=r'$O(\log T)$ ref', linewidth=1, alpha=0.7)

ax.set_xlabel('Customers $t$', fontsize=12)
ax.set_ylabel('Cumulative Profit Loss vs Oracle', fontsize=12)
ax.set_title(f'Cost of Learning (K={K} prices)', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, T)
ax.set_ylim(0, None)

plt.tight_layout()
fig.savefig('/Users/pranjal/Code/rl/ch07_bandits/sims/structural_pricing_misra_regret.png',
            dpi=300, bbox_inches='tight')
print("  Saved: structural_pricing_misra_regret.png")

# Figure 3: Arm Selection Distribution
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for ax, name in zip(axes, ['oracle', 'ucb_pi_tuned', 'ucb1', 'thompson']):
    n_bins = 20
    bin_size = K // n_bins
    binned_arms = np.array([mean_arms[name][i*bin_size:(i+1)*bin_size].sum()
                           for i in range(n_bins)]) / T
    bin_centers = np.array([(i + 0.5) * bin_size for i in range(n_bins)])
    bin_prices = PRICES[bin_centers.astype(int)]

    ax.bar(range(n_bins), binned_arms, color=colors[name], alpha=0.7,
           edgecolor='black', linewidth=0.5)

    opt_bin = OPTIMAL_ARM // bin_size
    ax.axvline(opt_bin, color='red', linestyle='--', linewidth=1.5,
               label=f'Opt (p={OPTIMAL_PRICE:.2f})')

    ax.set_xlabel('Price Bin', fontsize=11)
    ax.set_title(labels_plot[name], fontsize=12)
    ax.set_xticks(range(0, n_bins, 4))
    ax.set_xticklabels([f'{bin_prices[i]:.2f}' for i in range(0, n_bins, 4)])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

axes[0].set_ylabel('Fraction of Pulls', fontsize=11)

plt.tight_layout()
fig.savefig('/Users/pranjal/Code/rl/ch07_bandits/sims/structural_pricing_misra_arms.png',
            dpi=300, bbox_inches='tight')
print("  Saved: structural_pricing_misra_arms.png")

# Figure 4: Dominated prices over time
fig, ax = plt.subplots(figsize=(10, 5))

# Use first seed's data for illustration
dom_sample_interval = 5000
n_dom_samples = len(all_results[0]['dominated_over_time']['ucb_pi'])
dom_time_points = np.arange(K, T, dom_sample_interval)[:n_dom_samples]

dom_ucb_pi = np.zeros(n_dom_samples)
dom_ucb_pi_tuned = np.zeros(n_dom_samples)
for r in all_results:
    dom_ucb_pi += np.array(r['dominated_over_time']['ucb_pi'][:n_dom_samples])
    dom_ucb_pi_tuned += np.array(r['dominated_over_time']['ucb_pi_tuned'][:n_dom_samples])
dom_ucb_pi /= N_SEEDS
dom_ucb_pi_tuned /= N_SEEDS

ax.plot(dom_time_points, dom_ucb_pi, label='UCB-PI', color='C1', linewidth=1.5)
ax.plot(dom_time_points, dom_ucb_pi_tuned, label='UCB-PI-tuned', color='C3', linewidth=1.5)
ax.axhline(K - 1, color='gray', linestyle='--', alpha=0.5, label=f'Max possible ({K-1})')

ax.set_xlabel('Round $t$', fontsize=12)
ax.set_ylabel('Number of Dominated Prices', fontsize=12)
ax.set_title('Partial Identification: Dominated Prices Over Time', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, T)
ax.set_ylim(0, K)

plt.tight_layout()
fig.savefig('/Users/pranjal/Code/rl/ch07_bandits/sims/structural_pricing_misra_bounds.png',
            dpi=300, bbox_inches='tight')
print("  Saved: structural_pricing_misra_bounds.png")

# Figure 5: Delta estimation
fig, ax = plt.subplots(figsize=(10, 5))

print("  Tracking δ estimation for seed 0...")
rng = np.random.RandomState(0)
demand_model = SegmentedDemandModel(N_SEGMENTS, DELTA_TRUE, seed=0)
ucb_pi_track = UCB_PI(PRICES, N_SEGMENTS, demand_model.segment_weights)

delta_history = []
delta_sample_interval = 2000
segment_ids = rng.choice(N_SEGMENTS, size=T, p=demand_model.segment_weights)
valuation_offsets = rng.uniform(-DELTA_TRUE, DELTA_TRUE, size=T)

for t in range(T):
    arm = ucb_pi_track.select_arm(t)
    v_s = demand_model.segment_valuations[segment_ids[t]]
    v_i = v_s + valuation_offsets[t]
    purchased = (v_i >= PRICES[arm])
    reward = PRICES[arm] if purchased else 0.0
    ucb_pi_track.update(arm, reward, segment_ids[t], purchased)

    if t >= K and (t - K) % delta_sample_interval == 0:
        delta_history.append(ucb_pi_track.delta_hat)

delta_time_points = np.arange(K, T + 1, delta_sample_interval)
ax.plot(delta_time_points[:len(delta_history)], delta_history, color='C1',
        linewidth=1.5, label=r'$\hat{\delta}_t$')
ax.axhline(DELTA_TRUE, color='red', linestyle='--', linewidth=2,
           label=f'True $\\delta$ = {DELTA_TRUE}')

ax.set_xlabel('Round $t$', fontsize=12)
ax.set_ylabel(r'Estimated $\hat{\delta}$', fontsize=12)
ax.set_title('Partial Identification: Heterogeneity Parameter Estimation', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, T)

plt.tight_layout()
fig.savefig('/Users/pranjal/Code/rl/ch07_bandits/sims/structural_pricing_misra_delta.png',
            dpi=300, bbox_inches='tight')
print("  Saved: structural_pricing_misra_delta.png")

# =============================================================================
# NEW VALIDATION FIGURES
# =============================================================================

# Figure 6: Regret Scaling (Log-Log) — validates O(log T) vs O(sqrt(T))
fig, ax = plt.subplots(figsize=(10, 6))

for name in ['ucb_pi_tuned', 'ucb_pi', 'ucb1', 'thompson']:
    ax.loglog(time_points, regret[name] + 1, label=labels_plot[name],
              color=colors[name], linewidth=1.5)

# Reference lines for theoretical rates
ax.loglog(time_points, 100 * np.log(time_points), '--', color='gray',
          alpha=0.7, linewidth=1.5, label=r'$O(\log T)$')
ax.loglog(time_points, 0.5 * np.sqrt(time_points), ':', color='gray',
          alpha=0.7, linewidth=1.5, label=r'$O(\sqrt{T})$')

ax.set_xlabel('Customers $t$', fontsize=12)
ax.set_ylabel('Cumulative Regret', fontsize=12)
ax.set_title('Regret Scaling: Log-Log Plot', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(time_points[0], time_points[-1])

plt.tight_layout()
fig.savefig('/Users/pranjal/Code/rl/ch07_bandits/sims/structural_pricing_misra_regret_scaling.png',
            dpi=300, bbox_inches='tight')
print("  Saved: structural_pricing_misra_regret_scaling.png")

# Figure 7: Dominated Arms Over Time with Checkpoints
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(dom_time_points, dom_ucb_pi, label='UCB-PI', color='C1', linewidth=1.5)
ax.plot(dom_time_points, dom_ucb_pi_tuned, label='UCB-PI-tuned', color='C3', linewidth=1.5)

# Mark checkpoints with vertical lines and annotations
checkpoints_dom = [10000, 50000, 100000, 200000]
for t_check in checkpoints_dom:
    ax.axvline(t_check, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    # Find dominated count at this checkpoint
    idx = np.searchsorted(dom_time_points, t_check)
    if idx < len(dom_ucb_pi_tuned):
        ax.annotate(f'{dom_ucb_pi_tuned[idx]:.0f}',
                    xy=(t_check, dom_ucb_pi_tuned[idx] + 3),
                    fontsize=9, ha='center', color='C3')
        ax.annotate(f'{dom_ucb_pi[idx]:.0f}',
                    xy=(t_check, dom_ucb_pi[idx] - 5),
                    fontsize=9, ha='center', color='C1')

ax.set_xlabel('Round $t$', fontsize=12)
ax.set_ylabel('Number of Dominated Prices', fontsize=12)
ax.set_title('Partial Identification: Price Elimination Checkpoints', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, T)
ax.set_ylim(0, K)

plt.tight_layout()
fig.savefig('/Users/pranjal/Code/rl/ch07_bandits/sims/structural_pricing_misra_dominated_checkpoints.png',
            dpi=300, bbox_inches='tight')
print("  Saved: structural_pricing_misra_dominated_checkpoints.png")

# Figure 8: Price Selection Frequency (Histogram)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, name in zip(axes.flat, ['ucb_pi_tuned', 'ucb_pi', 'ucb1', 'thompson']):
    # Direct histogram of arm selections
    ax.bar(PRICES, mean_arms[name] / T, width=0.008, color=colors[name], alpha=0.7)
    ax.axvline(OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2,
               label=f'$p^*$ = {OPTIMAL_PRICE:.2f}')
    ax.set_xlabel('Price', fontsize=11)
    ax.set_ylabel('Selection Frequency', fontsize=11)
    ax.set_title(labels_plot[name], fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Price Selection Distribution (T={T:,})', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig('/Users/pranjal/Code/rl/ch07_bandits/sims/structural_pricing_misra_price_histogram.png',
            dpi=300, bbox_inches='tight')
print("  Saved: structural_pricing_misra_price_histogram.png")

# Figure 9: Convergence Trajectory — moving average of selected prices
# Re-use the UCB-PI tracking run for convergence data
print("  Computing convergence trajectory...")

# Track price selections during a fresh simulation run
rng_conv = np.random.RandomState(0)
demand_model_conv = SegmentedDemandModel(N_SEGMENTS, DELTA_TRUE, seed=0)
segment_ids_conv = rng_conv.choice(N_SEGMENTS, size=T, p=demand_model_conv.segment_weights)
valuation_offsets_conv = rng_conv.uniform(-DELTA_TRUE, DELTA_TRUE, size=T)

# Initialize algorithms for tracking
ucb_pi_tuned_track = UCB_PI_Tuned(PRICES, N_SEGMENTS, demand_model_conv.segment_weights)
ucb_pi_track_conv = UCB_PI(PRICES, N_SEGMENTS, demand_model_conv.segment_weights)
ucb1_track = UCB1(K, PRICES)
thompson_track = ThompsonSampling(K, PRICES)

track_algs = {
    'ucb_pi_tuned': ucb_pi_tuned_track,
    'ucb_pi': ucb_pi_track_conv,
    'ucb1': ucb1_track,
    'thompson': thompson_track,
}

# Subsample price selections (every 100th) to save memory
subsample = 100
prices_subsampled = {name: [] for name in track_algs}

for t in range(T):
    v_s = demand_model_conv.segment_valuations[segment_ids_conv[t]]
    v_i = v_s + valuation_offsets_conv[t]

    for name, alg in track_algs.items():
        arm = alg.select_arm(t)
        if t % subsample == 0:
            prices_subsampled[name].append(PRICES[arm])
        purchased = (v_i >= PRICES[arm])
        reward = PRICES[arm] if purchased else 0.0
        alg.update(arm, reward, segment_ids_conv[t], purchased)

# Compute moving average
fig, ax = plt.subplots(figsize=(10, 6))

window = 50  # 50 subsampled points = 5000 actual rounds
conv_time_points = np.arange(0, T, subsample)

for name in ['ucb_pi_tuned', 'ucb_pi', 'ucb1', 'thompson']:
    prices_array = np.array(prices_subsampled[name])
    # Compute moving average with np.convolve
    kernel = np.ones(window) / window
    moving_avg = np.convolve(prices_array, kernel, mode='valid')
    time_for_avg = conv_time_points[window-1:]

    ax.plot(time_for_avg, moving_avg, label=labels_plot[name], color=colors[name], linewidth=1.5)

ax.axhline(OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2,
           label=f'$p^*$ = {OPTIMAL_PRICE:.2f}')

ax.set_xlabel('Round $t$', fontsize=12)
ax.set_ylabel('Moving Average of Selected Price', fontsize=12)
ax.set_title(f'Price Convergence (window = {window * subsample:,} rounds)', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, T)
ax.set_ylim(0.2, 0.7)

plt.tight_layout()
fig.savefig('/Users/pranjal/Code/rl/ch07_bandits/sims/structural_pricing_misra_convergence.png',
            dpi=300, bbox_inches='tight')
print("  Saved: structural_pricing_misra_convergence.png")

plt.close('all')


# =============================================================================
# Generate LaTeX Table
# =============================================================================
latex_table = r"""\begin{tabular}{lcccccc}
\toprule
Algorithm & Ex Post \%% & Range & Profit & Frac.\ Opt & Dominated \\
\midrule
Oracle & 100.0\%% & [100-100\%%] & %.0f & %.3f & -- \\
UCB-PI-tuned & %.1f\%% & [%.0f-%.0f\%%] & %.0f $\pm$ %.0f & %.3f & %.0f \\
UCB-PI & %.1f\%% & [%.0f-%.0f\%%] & %.0f $\pm$ %.0f & %.3f & %.0f \\
UCB1 & %.1f\%% & [%.0f-%.0f\%%] & %.0f $\pm$ %.0f & %.3f & -- \\
LTE (5\%%) & %.1f\%% & [%.0f-%.0f\%%] & %.0f $\pm$ %.0f & %.3f & -- \\
Thompson & %.1f\%% & [%.0f-%.0f\%%] & %.0f $\pm$ %.0f & %.3f & -- \\
\bottomrule
\end{tabular}
""" % (
    final_profits['oracle'], frac_optimal['oracle'],
    100*ex_post_ratios['ucb_pi_tuned']['mean'],
    100*ex_post_ratios['ucb_pi_tuned']['min'], 100*ex_post_ratios['ucb_pi_tuned']['max'],
    final_profits['ucb_pi_tuned'], 2*final_profits_se['ucb_pi_tuned'],
    frac_optimal['ucb_pi_tuned'], dominated_final_mean['ucb_pi_tuned'],
    100*ex_post_ratios['ucb_pi']['mean'],
    100*ex_post_ratios['ucb_pi']['min'], 100*ex_post_ratios['ucb_pi']['max'],
    final_profits['ucb_pi'], 2*final_profits_se['ucb_pi'],
    frac_optimal['ucb_pi'], dominated_final_mean['ucb_pi'],
    100*ex_post_ratios['ucb1']['mean'],
    100*ex_post_ratios['ucb1']['min'], 100*ex_post_ratios['ucb1']['max'],
    final_profits['ucb1'], 2*final_profits_se['ucb1'],
    frac_optimal['ucb1'],
    100*ex_post_ratios['lte']['mean'],
    100*ex_post_ratios['lte']['min'], 100*ex_post_ratios['lte']['max'],
    final_profits['lte'], 2*final_profits_se['lte'],
    frac_optimal['lte'],
    100*ex_post_ratios['thompson']['mean'],
    100*ex_post_ratios['thompson']['min'], 100*ex_post_ratios['thompson']['max'],
    final_profits['thompson'], 2*final_profits_se['thompson'],
    frac_optimal['thompson'],
)

with open('/Users/pranjal/Code/rl/ch07_bandits/sims/structural_pricing_misra_results.tex', 'w') as f:
    f.write(latex_table)

print("  Saved: structural_pricing_misra_results.tex")
print()

print("=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print()
print("Output files:")
print("  - structural_pricing_misra_profit.png")
print("  - structural_pricing_misra_regret.png")
print("  - structural_pricing_misra_arms.png")
print("  - structural_pricing_misra_bounds.png")
print("  - structural_pricing_misra_delta.png")
print("  - structural_pricing_misra_regret_scaling.png")
print("  - structural_pricing_misra_dominated_checkpoints.png")
print("  - structural_pricing_misra_price_histogram.png")
print("  - structural_pricing_misra_convergence.png")
print("  - structural_pricing_misra_results.tex")
print()
print("Summary:")
print(f"  UCB-PI-tuned achieves {100*ex_post_ratios['ucb_pi_tuned']['mean']:.1f}% of oracle "
      f"(range {100*ex_post_ratios['ucb_pi_tuned']['min']:.0f}-{100*ex_post_ratios['ucb_pi_tuned']['max']:.0f}%)")
print(f"  UCB-PI achieves {100*ex_post_ratios['ucb_pi']['mean']:.1f}% of oracle "
      f"(range {100*ex_post_ratios['ucb_pi']['min']:.0f}-{100*ex_post_ratios['ucb_pi']['max']:.0f}%)")
print(f"  UCB1 achieves {100*ex_post_ratios['ucb1']['mean']:.1f}% of oracle "
      f"(range {100*ex_post_ratios['ucb1']['min']:.0f}-{100*ex_post_ratios['ucb1']['max']:.0f}%)")
print(f"  δ estimation: true={DELTA_TRUE:.2f}, UCB-PI={delta_estimates_mean['ucb_pi']:.3f}")
print(f"  Dominated prices at T: {dominated_final_mean['ucb_pi']:.0f}/{K}")
