#!/usr/bin/env python3
"""
knowledge_ladder.py
Chapter 7: Economic Bandits
Progressive structural knowledge and regret rates in dynamic pricing.

Six algorithms with increasing structural knowledge on the same demand
environment demonstrate the concrete value of economic structure:
  Level 0: ε-greedy (no adaptive exploration)   — Θ(T) regret
  Level 1: LTE (separate explore/exploit)        — O(T^{2/3}) regret
  Level 2: UCB1 (adaptive, no demand structure)  — O(√(KT)) regret
  Level 3: Thompson Sampling (Bayesian per arm)  — O(√(KT)) regret
  Level 4: UCB-PI (WARP + partial ID)            — O(log T) regret
  Level 5: UCB-PI-tuned (WARP + partial ID + var)— O(log T) regret
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_SINGLE
from sims.sim_cache import load_results, save_results, add_cache_args
apply_style()

# =============================================================================
# Configuration
# =============================================================================
K = 100
T = 200_000
N_SEEDS = 10
N_SEGMENTS = 1000
DELTA_TRUE = 0.1
PRICES = np.linspace(0.01, 1.0, K)
V_L, V_H = 0.0, 1.0
EPSILON = 0.1  # fixed ε for ε-greedy
LTE_TAU = 0.05  # fraction of T for LTE exploration

CHECKPOINTS = [10_000, 50_000, 100_000, 200_000]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
SCRIPT_NAME = 'knowledge_ladder'
CONFIG = {
    'K': K,
    'T': T,
    'N_SEEDS': N_SEEDS,
    'N_SEGMENTS': N_SEGMENTS,
    'DELTA_TRUE': DELTA_TRUE,
    'PRICES': PRICES.tolist(),
    'V_L': V_L,
    'V_H': V_H,
    'EPSILON': EPSILON,
    'LTE_TAU': LTE_TAU,
    'version': 1,
}

np.random.seed(42)

# =============================================================================
# Demand Model (reused from structural_pricing_misra.py)
# =============================================================================
class SegmentedDemandModel:
    def __init__(self, n_segments, delta, seed=42):
        self.n_segments = n_segments
        self.delta = delta
        rng = np.random.RandomState(seed)
        self.segment_valuations = rng.uniform(V_L + delta, V_H - delta, n_segments)
        self.segment_weights = np.ones(n_segments) / n_segments

    def true_demand(self, price):
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

    def true_profit(self, price):
        return price * self.true_demand(price)


canonical_demand = SegmentedDemandModel(N_SEGMENTS, DELTA_TRUE, seed=42)
TRUE_PROFITS = np.array([canonical_demand.true_profit(p) for p in PRICES])
OPTIMAL_ARM = np.argmax(TRUE_PROFITS)
OPTIMAL_PROFIT = TRUE_PROFITS[OPTIMAL_ARM]
OPTIMAL_PRICE = PRICES[OPTIMAL_ARM]

# =============================================================================
# Algorithm Implementations
# =============================================================================

class EpsilonGreedy:
    """Fixed ε-greedy: no adaptive exploration."""
    def __init__(self, n_arms, prices, epsilon=EPSILON):
        self.n_arms = n_arms
        self.prices = prices
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self, t, rng):
        if t < self.n_arms:
            return t
        if rng.rand() < self.epsilon:
            return rng.randint(self.n_arms)
        return int(np.argmax(self.values))

    def update(self, arm, reward, segment_id, purchased):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class LearnThenEarn:
    """Learn-Then-Earn: uniform random exploration for first τ fraction, then commit."""
    def __init__(self, n_arms, prices, tau=LTE_TAU, T_total=T):
        self.n_arms = n_arms
        self.prices = prices
        self.T_explore = int(tau * T_total)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.best_arm = None

    def select_arm(self, t, rng):
        if t < self.T_explore:
            return rng.randint(self.n_arms)
        if self.best_arm is None:
            self.best_arm = int(np.argmax(self.values))
        return self.best_arm

    def update(self, arm, reward, segment_id, purchased):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class UCB1:
    """Standard UCB1."""
    def __init__(self, n_arms, prices):
        self.n_arms = n_arms
        self.prices = prices
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self, t, rng):
        if t < self.n_arms:
            return t
        ucb = self.values + np.sqrt(2 * np.log(t + 1) / np.maximum(self.counts, 1))
        return int(np.argmax(ucb))

    def update(self, arm, reward, segment_id, purchased):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSampling:
    """Thompson Sampling for pricing (Beta-Bernoulli on purchase, scaled by price)."""
    def __init__(self, n_arms, prices):
        self.n_arms = n_arms
        self.prices = prices
        self.alpha = np.ones(n_arms)
        self.beta_param = np.ones(n_arms)
        self.counts = np.zeros(n_arms)

    def select_arm(self, t, rng):
        theta = rng.beta(self.alpha, self.beta_param)
        expected_profit = self.prices * theta
        return int(np.argmax(expected_profit))

    def update(self, arm, reward, segment_id, purchased):
        self.counts[arm] += 1
        if purchased:
            self.alpha[arm] += 1
        else:
            self.beta_param[arm] += 1


class UCB_PI:
    """UCB with Partial Identification — WARP-based dominance elimination,
    price-scaled exploration bonus (Misra, Schwartz & Abernethy 2019)."""
    def __init__(self, prices, n_segments, segment_weights):
        self.prices = prices
        self.K = len(prices)
        self.n_segments = n_segments
        self.segment_weights = segment_weights
        self.counts = np.zeros(self.K)
        self.values = np.zeros(self.K)
        self.sum_sq = np.zeros(self.K)
        self.segment_p_min = np.full(n_segments, V_L)
        self.segment_p_max = np.full(n_segments, V_H)
        self.obs_counts = np.zeros((n_segments, self.K), dtype=int)
        self.obs_purchases = np.zeros((n_segments, self.K), dtype=int)
        self.delta_hat = (V_H - V_L) / 2

    def _update_partial_identification(self, arm, segment_id, purchased):
        self.obs_counts[segment_id, arm] += 1
        if purchased:
            self.obs_purchases[segment_id, arm] += 1
        s = segment_id
        counts_s = self.obs_counts[s]
        purchases_s = self.obs_purchases[s]
        observed = counts_s > 0
        full_buy = observed & (purchases_s == counts_s)
        if full_buy.any():
            self.segment_p_min[s] = self.prices[np.where(full_buy)[0][-1]]
        else:
            self.segment_p_min[s] = V_L
        no_buy = observed & (purchases_s == 0)
        if no_buy.any():
            self.segment_p_max[s] = self.prices[np.where(no_buy)[0][0]]
        else:
            self.segment_p_max[s] = V_H

    def _estimate_delta(self):
        has_lb = self.segment_p_min > V_L
        has_ub = self.segment_p_max < V_H
        crossed = has_lb & has_ub & (self.segment_p_max > self.segment_p_min)
        if crossed.sum() >= 2:
            delta_estimates = (self.segment_p_max[crossed] - self.segment_p_min[crossed]) / 2
            delta_max = delta_estimates.max()
            mean_delta = delta_estimates.mean()
            gamma_hat = delta_max - mean_delta
            self.delta_hat = delta_max + gamma_hat
        elif crossed.sum() == 1:
            self.delta_hat = (self.segment_p_max[crossed] - self.segment_p_min[crossed])[0] / 2
        return self.delta_hat

    def _compute_profit_bounds(self):
        v_hat = (self.segment_p_min + self.segment_p_max) / 2
        v_min = v_hat - self.delta_hat
        v_max = v_hat + self.delta_hat
        lb_demand = np.zeros(self.K)
        ub_demand = np.zeros(self.K)
        for k in range(self.K):
            lb_demand[k] = self.segment_weights[v_min >= self.prices[k]].sum()
            ub_demand[k] = self.segment_weights[v_max >= self.prices[k]].sum()
        return self.prices * lb_demand, self.prices * ub_demand

    def select_arm(self, t, rng):
        if t < self.K:
            return t
        self._estimate_delta()
        lb_profits, ub_profits = self._compute_profit_bounds()
        max_lb = lb_profits.max()
        ucb_pi = np.full(self.K, -np.inf)
        dominated = ub_profits <= max_lb
        active = ~dominated
        if active.any():
            exploration = self.prices[active] * np.sqrt(
                2 * np.log(t + 1) / np.maximum(self.counts[active], 1)
            )
            ucb_pi[active] = self.values[active] + exploration
        return int(np.argmax(ucb_pi))

    def update(self, arm, reward, segment_id, purchased):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.sum_sq[arm] += reward ** 2
        self._update_partial_identification(arm, segment_id, purchased)


class UCB_PI_Tuned(UCB_PI):
    """UCB-PI with variance tuning and WARP-based dominance elimination."""
    def select_arm(self, t, rng):
        if t < self.K:
            return t
        self._estimate_delta()
        lb_profits, ub_profits = self._compute_profit_bounds()
        max_lb = lb_profits.max()
        ucb_pi = np.full(self.K, -np.inf)
        dominated = ub_profits <= max_lb
        active = ~dominated
        if active.any():
            counts_active = np.maximum(self.counts[active], 1)
            mean_sq = self.sum_sq[active] / counts_active
            variance = np.maximum(0, mean_sq - self.values[active] ** 2)
            V_kt = variance + np.sqrt(2 * np.log(t + 1) / counts_active)
            exploration = 2 * self.prices[active] * self.delta_hat * np.sqrt(
                (np.log(t + 1) / counts_active) * np.minimum(0.25, V_kt)
            )
            ucb_pi[active] = self.values[active] + exploration
        return int(np.argmax(ucb_pi))


# =============================================================================
# Algorithm registry
# =============================================================================
alg_names = ['eps_greedy', 'lte', 'ucb1', 'thompson', 'ucb_pi', 'ucb_pi_tuned']
alg_labels = {
    'eps_greedy': r'$\varepsilon$-greedy ($\varepsilon=0.1$)',
    'lte': 'Learn-Then-Earn (5%)',
    'ucb1': 'UCB1',
    'thompson': 'Thompson Sampling',
    'ucb_pi': 'UCB-PI (WARP)',
    'ucb_pi_tuned': 'UCB-PI-tuned (WARP)',
}

# =============================================================================
# Run Experiments
# =============================================================================

def run_experiment(seed):
    rng = np.random.RandomState(seed)
    demand_model = SegmentedDemandModel(N_SEGMENTS, DELTA_TRUE, seed=seed)
    true_profits_seed = np.array([demand_model.true_profit(p) for p in PRICES])
    opt_arm = np.argmax(true_profits_seed)
    opt_profit = true_profits_seed[opt_arm]

    segment_ids = rng.choice(N_SEGMENTS, size=T, p=demand_model.segment_weights)
    valuation_offsets = rng.uniform(-DELTA_TRUE, DELTA_TRUE, size=T)

    # Per-algorithm RNG for reproducible arm selection
    rng_eps = np.random.RandomState(seed + 1000)
    rng_lte = np.random.RandomState(seed + 1500)
    rng_ucb = np.random.RandomState(seed + 2000)
    rng_ts = np.random.RandomState(seed + 3000)
    rng_pi = np.random.RandomState(seed + 4000)
    rng_pit = np.random.RandomState(seed + 5000)

    eps_greedy = EpsilonGreedy(K, PRICES)
    lte = LearnThenEarn(K, PRICES)
    ucb1 = UCB1(K, PRICES)
    thompson = ThompsonSampling(K, PRICES)
    ucb_pi = UCB_PI(PRICES, N_SEGMENTS, demand_model.segment_weights)
    ucb_pi_tuned = UCB_PI_Tuned(PRICES, N_SEGMENTS, demand_model.segment_weights)

    algorithms = {
        'eps_greedy': (eps_greedy, rng_eps),
        'lte': (lte, rng_lte),
        'ucb1': (ucb1, rng_ucb),
        'thompson': (thompson, rng_ts),
        'ucb_pi': (ucb_pi, rng_pi),
        'ucb_pi_tuned': (ucb_pi_tuned, rng_pit),
    }

    # Track cumulative regret at every round (subsampled)
    sample_interval = 100
    n_samples = T // sample_interval
    cum_regret = {name: np.zeros(n_samples) for name in algorithms}
    running_regret = {name: 0.0 for name in algorithms}

    for t in range(T):
        segment_id = segment_ids[t]
        v_s = demand_model.segment_valuations[segment_id]
        v_i = v_s + valuation_offsets[t]

        for name, (alg, alg_rng) in algorithms.items():
            arm = alg.select_arm(t, alg_rng)
            price = PRICES[arm]
            purchased = (v_i >= price)
            reward = price if purchased else 0.0
            alg.update(arm, reward, segment_id, purchased)
            running_regret[name] += (opt_profit - reward)

        if (t + 1) % sample_interval == 0:
            idx = (t + 1) // sample_interval - 1
            for name in algorithms:
                cum_regret[name][idx] = running_regret[name]

    return cum_regret, opt_profit


# =============================================================================
# Compute
# =============================================================================
def compute_data():
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    # Print Configuration
    print("=" * 70)
    print("KNOWLEDGE LADDER — PROGRESSIVE STRUCTURAL KNOWLEDGE IN PRICING")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Prices (K):       {K}")
    print(f"  Horizon (T):      {T:,}")
    print(f"  Seeds:            {N_SEEDS}")
    print(f"  Segments (S):     {N_SEGMENTS:,}")
    print(f"  Heterogeneity δ:  {DELTA_TRUE}")
    print(f"  ε-greedy ε:       {EPSILON}")
    print(f"  LTE explore:      {LTE_TAU*100:.0f}% ({int(LTE_TAU*T):,} rounds)")
    print(f"  Optimal price:    {OPTIMAL_PRICE:.2f} (arm {OPTIMAL_ARM})")
    print(f"  Optimal profit:   {OPTIMAL_PROFIT:.4f}/customer")
    print()
    sys.stdout.flush()

    print("Running experiments...")
    sys.stdout.flush()

    all_regrets = []
    all_opt_profits = []

    with tqdm(total=N_SEEDS, desc="Seeds", unit="seed") as pbar:
        for seed in range(N_SEEDS):
            regrets, opt_p = run_experiment(seed)
            all_regrets.append(regrets)
            all_opt_profits.append(opt_p)
            pbar.update(1)

    # Aggregate
    sample_interval = 100
    time_points = np.arange(sample_interval, T + 1, sample_interval)

    regret_arrays = {name: np.array([r[name] for r in all_regrets]) for name in alg_names}
    mean_regret = {name: arr.mean(axis=0) for name, arr in regret_arrays.items()}
    se_regret = {name: arr.std(axis=0) / np.sqrt(N_SEEDS) for name, arr in regret_arrays.items()}

    # Print Results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Checkpoint table
    print("Cumulative Regret at Checkpoints:")
    print("-" * 120)
    header = f"{'T':>10}"
    for name in alg_names:
        header += f"  {alg_labels[name]:>18}"
    print(header)
    print("-" * 120)

    for cp in CHECKPOINTS:
        idx = cp // sample_interval - 1
        row = f"{cp:>10,}"
        for name in alg_names:
            vals = regret_arrays[name][:, idx]
            m = vals.mean()
            se = vals.std() / np.sqrt(N_SEEDS)
            row += f"  {m:>14,.0f} ({se:>4.0f})"
        print(row)
    print("-" * 120)
    print()

    # Rate diagnostics table
    print("Rate Diagnostics (which normalization stabilizes?):")
    print("-" * 140)
    print(f"{'':>10}", end="")
    for name in alg_names:
        label = alg_labels[name][:16]
        print(f"  {'--- ' + label + ' ---':>22}", end="")
    print()
    print(f"{'T':>10}", end="")
    for name in alg_names:
        if name == 'eps_greedy':
            print(f"  {'R/T':>8} {'R/√T':>8}", end="")
        elif name == 'lte':
            print(f"  {'R/T':>8} {'R/T^⅔':>8}", end="")
        elif name in ('ucb_pi', 'ucb_pi_tuned'):
            print(f"  {'R/T':>8} {'R/logT':>8}", end="")
        else:
            print(f"  {'R/T':>8} {'R/√T':>8}", end="")
    print()
    print("-" * 140)

    for cp in CHECKPOINTS:
        idx = cp // sample_interval - 1
        row = f"{cp:>10,}"
        for name in alg_names:
            m = mean_regret[name][idx]
            if name == 'eps_greedy':
                row += f"  {m/cp:>8.4f} {m/np.sqrt(cp):>8.1f}"
            elif name == 'lte':
                row += f"  {m/cp:>8.4f} {m/cp**(2/3):>8.1f}"
            elif name in ('ucb_pi', 'ucb_pi_tuned'):
                row += f"  {m/cp:>8.4f} {m/np.log(cp):>8.1f}"
            else:
                row += f"  {m/cp:>8.4f} {m/np.sqrt(cp):>8.1f}"
        print(row)
    print("-" * 140)
    print()

    # Summary: final regret
    print("Final Regret at T = 200,000:")
    print("-" * 80)
    print(f"{'Algorithm':<28}  {'Regret':>10}  {'SE':>8}  {'Regret/T':>10}  {'Rate':>16}")
    print("-" * 80)
    for name in alg_names:
        vals = regret_arrays[name][:, -1]
        m = vals.mean()
        se = vals.std() / np.sqrt(N_SEEDS)
        if name == 'eps_greedy':
            rate_str = f"R/T = {m/T:.4f}"
        elif name == 'lte':
            rate_str = f"R/T^(2/3) = {m/T**(2/3):.1f}"
        elif name in ('ucb1', 'thompson'):
            rate_str = f"R/√T = {m/np.sqrt(T):.1f}"
        else:
            rate_str = f"R/logT = {m/np.log(T):.1f}"
        print(f"{alg_labels[name]:<28}  {m:>10,.0f}  {se:>8,.0f}  {m/T:>10.4f}  {rate_str:>16}")
    print("-" * 80)
    print()

    data = {
        'regret_arrays': regret_arrays,
        'mean_regret': mean_regret,
        'se_regret': se_regret,
        'time_points': time_points,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def generate_outputs(data):
    regret_arrays = data['regret_arrays']
    mean_regret = data['mean_regret']
    se_regret = data['se_regret']
    time_points = data['time_points']
    sample_interval = 100

    # =============================================================================
    # Generate Figure: Log-Log Cumulative Regret
    # =============================================================================
    print("Generating figure...")

    alg_colors = {
        'eps_greedy': COLORS['gray'],
        'lte': COLORS['brown'],
        'ucb1': COLORS['blue'],
        'thompson': COLORS['orange'],
        'ucb_pi': COLORS['purple'],
        'ucb_pi_tuned': COLORS['red'],
    }

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    for name in alg_names:
        ax.plot(time_points, mean_regret[name], label=alg_labels[name],
                color=alg_colors[name], linewidth=1.8)
        ax.fill_between(time_points,
                        mean_regret[name] - 2 * se_regret[name],
                        mean_regret[name] + 2 * se_regret[name],
                        alpha=0.15, color=alg_colors[name])

    # Reference lines (dashed)
    t_ref = np.linspace(1000, T, 500)
    eps_final = mean_regret['eps_greedy'][-1]
    ucb_final = mean_regret['ucb1'][-1]
    pi_final = mean_regret['ucb_pi_tuned'][-1]

    # Θ(T): linear reference
    c_linear = eps_final / T
    ax.plot(t_ref, c_linear * t_ref, '--', color='black', alpha=0.4, linewidth=1.0,
            label=r'$\Theta(T)$ reference')

    # O(√T): square root reference
    c_sqrt = ucb_final / np.sqrt(T)
    ax.plot(t_ref, c_sqrt * np.sqrt(t_ref), '--', color=COLORS['purple'], alpha=0.4,
            linewidth=1.0, label=r'$O(\sqrt{T})$ reference')

    # O(log T): logarithmic reference
    c_log = pi_final / np.log(T)
    ax.plot(t_ref, c_log * np.log(t_ref), '--', color=COLORS['green'], alpha=0.4,
            linewidth=1.0, label=r'$O(\log T)$ reference')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Customers $t$')
    ax.set_ylabel('Cumulative Regret')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.set_xlim(1000, T)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'knowledge_ladder_regret.png'))
    print("  Saved: knowledge_ladder_regret.png")


    # =============================================================================
    # Generate LaTeX Table
    # =============================================================================
    print("Generating LaTeX table...")

    tex_path = os.path.join(OUT_DIR, 'knowledge_ladder_results.tex')
    with open(tex_path, 'w') as f:
        f.write("\\begin{tabular}{llrrrrl}\n")
        f.write("\\toprule\n")
        f.write("Level & Algorithm & $T{=}10$K & $T{=}50$K & $T{=}100$K & $T{=}200$K & Rate \\\\\n")
        f.write("\\midrule\n")

        level_map = {
            'eps_greedy': 0, 'lte': 1, 'ucb1': 2,
            'thompson': 3, 'ucb_pi': 4, 'ucb_pi_tuned': 5,
        }
        rate_labels = {
            'eps_greedy': r'$\Theta(T)$',
            'lte': r'$O(T^{2/3})$',
            'ucb1': r'$O(\sqrt{KT})$',
            'thompson': r'$O(\sqrt{KT})$',
            'ucb_pi': r'$O(\log T)$',
            'ucb_pi_tuned': r'$O(\log T)$',
        }
        tex_labels = {
            'eps_greedy': r'$\varepsilon$-greedy',
            'lte': 'LTE (5\\%)',
            'ucb1': 'UCB1',
            'thompson': 'Thompson',
            'ucb_pi': 'UCB-PI',
            'ucb_pi_tuned': 'UCB-PI-tuned',
        }

        for name in alg_names:
            level = level_map[name]
            row = f"{level} & {tex_labels[name]}"
            for cp in CHECKPOINTS:
                idx = cp // sample_interval - 1
                vals = regret_arrays[name][:, idx]
                m = vals.mean()
                row += f" & {m:,.0f}"
            row += f" & {rate_labels[name]} \\\\\n"
            f.write(row)

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"  Saved: {tex_path}")
    print()

    # =============================================================================
    # Rate diagnostic table (LaTeX)
    # =============================================================================
    tex_diag_path = os.path.join(OUT_DIR, 'knowledge_ladder_diagnostics.tex')
    with open(tex_diag_path, 'w') as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("$T$ & $R/T$ ($\\varepsilon$-gr.) & $R/T^{2/3}$ (LTE) & $R/\\sqrt{T}$ (UCB1) & $R/\\sqrt{T}$ (TS) & $R/\\log T$ (UCB-PI) & $R/\\log T$ (tuned) \\\\\n")
        f.write("\\midrule\n")
        for cp in CHECKPOINTS:
            idx = cp // sample_interval - 1
            r_eps = mean_regret['eps_greedy'][idx]
            r_lte = mean_regret['lte'][idx]
            r_ucb = mean_regret['ucb1'][idx]
            r_ts = mean_regret['thompson'][idx]
            r_pi = mean_regret['ucb_pi'][idx]
            r_pit = mean_regret['ucb_pi_tuned'][idx]
            f.write(f"{cp:,} & {r_eps/cp:.4f} & {r_lte/cp**(2/3):.1f} & {r_ucb/np.sqrt(cp):.1f} & {r_ts/np.sqrt(cp):.1f} & {r_pi/np.log(cp):.1f} & {r_pit/np.log(cp):.1f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"  Saved: {tex_diag_path}")
    print()
    print("Output files:")
    print(f"  {os.path.join(OUT_DIR, 'knowledge_ladder_regret.png')}")
    print(f"  {tex_path}")
    print(f"  {tex_diag_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_cache_args(parser)
    args = parser.parse_args()
    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, "No cache found. Run without --plots-only first."
    else:
        data = compute_data()
    if not args.data_only:
        generate_outputs(data)
