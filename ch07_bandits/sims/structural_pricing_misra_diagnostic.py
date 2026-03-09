#!/usr/bin/env python3
"""
structural_pricing_misra_diagnostic.py
Chapter 7: Economic Bandits
Single-seed diagnostic run to investigate UCB-PI internals vs Thompson Sampling.

Logs δ estimation, profit bounds, dominated arms, UCB indices, and Thompson
posteriors at checkpoints every 10,000 rounds. Output: diagnostic .txt file.

Self-contained: copies needed classes to avoid importing the main sim
(which runs the full 10-seed experiment at module level).
"""

import numpy as np
import sys
import os
from typing import Tuple

# =============================================================================
# Configuration (matching main sim)
# =============================================================================
K = 100
T = 200_000
N_SEGMENTS = 1000
DELTA_TRUE = 0.1
PRICES = np.linspace(0.01, 1.0, K)
V_L, V_H = 0.0, 1.0

CHECKPOINT_INTERVAL = 10_000
SEED = 0
KEY_ARMS = list(range(35, 56))  # Arms near optimal (prices $0.36–$0.56)


# =============================================================================
# Classes (copied from structural_pricing_misra.py to avoid triggering main sim)
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


class ThompsonSampling:
    def __init__(self, n_arms, prices):
        self.n_arms = n_arms
        self.prices = prices
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.counts = np.zeros(n_arms)

    def select_arm(self, t):
        theta = np.random.beta(self.alpha, self.beta)
        expected_profit = self.prices * theta
        return int(np.argmax(expected_profit))

    def update(self, arm, reward, segment_id, purchased):
        self.counts[arm] += 1
        if purchased:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


# =============================================================================
# Diagnostic UCB-PI — Caches all intermediate computations
# =============================================================================

class UCB_PI_Diag:
    """UCB-PI with cached internals for diagnostic logging."""

    def __init__(self, prices, n_segments, segment_weights):
        self.prices = prices
        self.K = len(prices)
        self.n_segments = n_segments
        self.segment_weights = segment_weights

        self.counts = np.zeros(self.K)
        self.values = np.zeros(self.K)
        self.sum_sq = np.zeros(self.K)

        # Paper naming: p_min = highest price where D̂=1, p_max = lowest price where D̂=0
        self.segment_p_min = np.full(n_segments, V_L)
        self.segment_p_max = np.full(n_segments, V_H)

        # Per-(segment, price) observation tracking for demand rates
        self.obs_counts = np.zeros((n_segments, self.K), dtype=int)
        self.obs_purchases = np.zeros((n_segments, self.K), dtype=int)

        self.delta_hat = (V_H - V_L) / 2
        self.dominated_counts = []

        # Diagnostic caches
        self._last_lb = np.zeros(self.K)
        self._last_ub = np.zeros(self.K)
        self._last_ucb = np.full(self.K, -np.inf)
        self._last_dominated = np.zeros(self.K, dtype=bool)
        self._last_max_lb = 0.0
        self._last_n_crossed = 0
        self._last_delta_estimates = np.array([])

    def _update_partial_identification(self, arm, segment_id, purchased):
        """Update segment valuation bounds using demand rates (paper Section 2.3.1)."""
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

    def _estimate_delta(self):
        has_lb = self.segment_p_min > V_L
        has_ub = self.segment_p_max < V_H
        crossed = has_lb & has_ub & (self.segment_p_max > self.segment_p_min)
        self._last_n_crossed = int(crossed.sum())
        if crossed.sum() >= 2:
            delta_estimates = (self.segment_p_max[crossed] - self.segment_p_min[crossed]) / 2
            self._last_delta_estimates = delta_estimates.copy()
            delta_max = delta_estimates.max()
            mean_delta = delta_estimates.mean()
            # Paper's bias correction: γ̂ = max - mean; δ̂ = max + γ̂
            gamma_hat = delta_max - mean_delta
            self.delta_hat = delta_max + gamma_hat
        elif crossed.sum() == 1:
            delta_estimates = (self.segment_p_max[crossed] - self.segment_p_min[crossed]) / 2
            self._last_delta_estimates = delta_estimates.copy()
            self.delta_hat = delta_estimates[0]
        else:
            self._last_delta_estimates = np.array([])
        return self.delta_hat

    def _compute_profit_bounds(self):
        # Segment midpoints: (p^min + p^max)/2 ≈ v_s
        v_hat = (self.segment_p_min + self.segment_p_max) / 2
        v_min = v_hat - self.delta_hat
        v_max = v_hat + self.delta_hat

        lb_demand = np.zeros(self.K)
        ub_demand = np.zeros(self.K)
        for k in range(self.K):
            lb_demand[k] = self.segment_weights[v_min >= self.prices[k]].sum()
            ub_demand[k] = self.segment_weights[v_max >= self.prices[k]].sum()

        lb_profits = self.prices * lb_demand
        ub_profits = self.prices * ub_demand

        self._last_lb = lb_profits.copy()
        self._last_ub = ub_profits.copy()
        return lb_profits, ub_profits

    def select_arm(self, t):
        if t < self.K:
            return t

        self._estimate_delta()
        lb_profits, ub_profits = self._compute_profit_bounds()
        max_lb = lb_profits.max()
        self._last_max_lb = max_lb

        ucb_pi = np.full(self.K, -np.inf)
        dominated = ub_profits <= max_lb
        self._last_dominated = dominated.copy()

        active = ~dominated
        if active.any():
            exploration = self.prices[active] * np.sqrt(
                2 * np.log(t + 1) / np.maximum(self.counts[active], 1)
            )
            ucb_pi[active] = self.values[active] + exploration

        self._last_ucb = ucb_pi.copy()
        self.dominated_counts.append(dominated.sum())
        return int(np.argmax(ucb_pi))

    def update(self, arm, reward, segment_id, purchased):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.sum_sq[arm] += reward ** 2
        self._update_partial_identification(arm, segment_id, purchased)


class UCB_PI_Tuned_Diag(UCB_PI_Diag):
    """UCB-PI-tuned with cached internals for diagnostic logging."""

    def select_arm(self, t):
        if t < self.K:
            return t

        self._estimate_delta()
        lb_profits, ub_profits = self._compute_profit_bounds()
        max_lb = lb_profits.max()
        self._last_max_lb = max_lb

        ucb_pi = np.full(self.K, -np.inf)
        dominated = ub_profits <= max_lb
        self._last_dominated = dominated.copy()

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

        self._last_ucb = ucb_pi.copy()
        self.dominated_counts.append(dominated.sum())
        return int(np.argmax(ucb_pi))


# =============================================================================
# Checkpoint Logging
# =============================================================================

def log_checkpoint(t, ucb_pi, ucb_pi_tuned, thompson, demand_model, f):
    """Write one diagnostic checkpoint block."""
    f.write(f"\n{'=' * 70}\n")
    f.write(f"CHECKPOINT t = {t:,}\n")
    f.write(f"{'=' * 70}\n\n")

    # --- δ estimation internals ---
    f.write("--- delta Estimation Internals ---\n")
    for name, alg in [("UCB-PI", ucb_pi), ("UCB-PI-tuned", ucb_pi_tuned)]:
        n_crossed = alg._last_n_crossed
        de = alg._last_delta_estimates
        has_lb = alg.segment_p_min > V_L
        has_ub = alg.segment_p_max < V_H
        n_no_obs = int(np.sum(~has_lb & ~has_ub))
        f.write(f"  {name}:\n")
        f.write(f"    delta_hat = {alg.delta_hat:.6f}  (true = {DELTA_TRUE})\n")
        f.write(f"    crossed segments: {n_crossed} / {N_SEGMENTS}\n")
        if len(de) > 0:
            f.write(f"    per-segment delta estimates: mean={de.mean():.6f}, "
                    f"min={de.min():.6f}, max={de.max():.6f}, std={de.std():.6f}\n")
        f.write(f"    segments with no observations: {n_no_obs}\n")
    f.write("\n")

    # --- Profit bounds for key arms ---
    f.write("--- Profit Bounds (arms 35-55, prices $0.36-$0.56) ---\n")
    true_profits_key = np.array([demand_model.true_profit(PRICES[k]) for k in KEY_ARMS])
    opt_k = KEY_ARMS[np.argmax(true_profits_key)]

    # Header: UCB-PI columns then UCB-PI-tuned columns
    f.write(f"  {'Arm':>4} {'Price':>6} {'TrueP':>7}  |  "
            f"{'LB':>7} {'UB':>7} {'UCBidx':>7} {'Dom':>4} {'Pulls':>6}  |  "
            f"{'LB_t':>7} {'UB_t':>7} {'UCBidx':>7} {'Dom':>4} {'Pulls':>6}\n")
    f.write("  " + "-" * 105 + "\n")

    for k in KEY_ARMS:
        tp = demand_model.true_profit(PRICES[k])
        marker = " *" if k == opt_k else "  "
        f.write(f"  {k:>4} {PRICES[k]:>6.2f} {tp:>7.4f}{marker}|  "
                f"{ucb_pi._last_lb[k]:>7.4f} {ucb_pi._last_ub[k]:>7.4f} "
                f"{ucb_pi._last_ucb[k]:>7.4f} "
                f"{'Y' if ucb_pi._last_dominated[k] else 'N':>4} "
                f"{ucb_pi.counts[k]:>6.0f}  |  "
                f"{ucb_pi_tuned._last_lb[k]:>7.4f} {ucb_pi_tuned._last_ub[k]:>7.4f} "
                f"{ucb_pi_tuned._last_ucb[k]:>7.4f} "
                f"{'Y' if ucb_pi_tuned._last_dominated[k] else 'N':>4} "
                f"{ucb_pi_tuned.counts[k]:>6.0f}\n")

    f.write(f"\n  max_lb (UCB-PI): {ucb_pi._last_max_lb:.6f}   "
            f"max_lb (UCB-PI-tuned): {ucb_pi_tuned._last_max_lb:.6f}\n")
    f.write(f"  Dominated total (UCB-PI): {ucb_pi._last_dominated.sum()}   "
            f"(UCB-PI-tuned): {ucb_pi_tuned._last_dominated.sum()}\n\n")

    # --- Full dominated arm list ---
    f.write("--- Dominated Arms (UCB-PI) ---\n")
    dom_arms = np.where(ucb_pi._last_dominated)[0]
    if len(dom_arms) > 0:
        f.write(f"  Count: {len(dom_arms)}\n")
        f.write(f"  Arms: {[int(a) for a in dom_arms]}\n")
        f.write(f"  Price range: [{PRICES[dom_arms[0]]:.2f} ... {PRICES[dom_arms[-1]]:.2f}]\n")
    else:
        f.write("  None\n")

    dom_arms_t = np.where(ucb_pi_tuned._last_dominated)[0]
    f.write("--- Dominated Arms (UCB-PI-tuned) ---\n")
    if len(dom_arms_t) > 0:
        f.write(f"  Count: {len(dom_arms_t)}\n")
        f.write(f"  Arms: {[int(a) for a in dom_arms_t]}\n")
        f.write(f"  Price range: [{PRICES[dom_arms_t[0]]:.2f} ... {PRICES[dom_arms_t[-1]]:.2f}]\n")
    else:
        f.write("  None\n")
    f.write("\n")

    # --- Arm selection counts for key arms ---
    f.write("--- Arm Selection Counts (arms 35-55) ---\n")
    f.write(f"  {'Arm':>4} {'Price':>6}  {'UCB-PI':>8} {'UCB-PI-t':>8} {'Thompson':>8}\n")
    for k in KEY_ARMS:
        f.write(f"  {k:>4} {PRICES[k]:>6.2f}  "
                f"{ucb_pi.counts[k]:>8.0f} {ucb_pi_tuned.counts[k]:>8.0f} "
                f"{thompson.counts[k]:>8.0f}\n")
    f.write(f"\n  Total pulls: UCB-PI={ucb_pi.counts.sum():.0f}  "
            f"UCB-PI-tuned={ucb_pi_tuned.counts.sum():.0f}  "
            f"Thompson={thompson.counts.sum():.0f}\n\n")

    # --- Segment identification quality ---
    f.write("--- Segment Identification (10 segments near v_s ~ 0.45) ---\n")
    vs = demand_model.segment_valuations
    near_optimal = np.argsort(np.abs(vs - 0.45))[:10]
    f.write(f"  {'Seg':>5} {'v_s':>7} {'p_min':>7} {'p_max':>7} {'v_hat':>7} "
            f"{'|err|':>7} {'Cross':>6}\n")
    for s in near_optimal:
        v_hat = (ucb_pi.segment_p_min[s] + ucb_pi.segment_p_max[s]) / 2
        crossed = (ucb_pi.segment_p_min[s] > V_L) and (ucb_pi.segment_p_max[s] < V_H) and (ucb_pi.segment_p_max[s] > ucb_pi.segment_p_min[s])
        err = abs(vs[s] - v_hat)
        f.write(f"  {s:>5} {vs[s]:>7.4f} {ucb_pi.segment_p_min[s]:>7.4f} "
                f"{ucb_pi.segment_p_max[s]:>7.4f} {v_hat:>7.4f} "
                f"{err:>7.4f} {'Y' if crossed else 'N':>6}\n")
    f.write("\n")

    # --- Thompson posteriors ---
    f.write("--- Thompson Posteriors (arms 35-55) ---\n")
    f.write(f"  {'Arm':>4} {'Price':>6} {'alpha':>8} {'beta':>8} {'E[theta]':>8} "
            f"{'E[profit]':>10} {'Pulls':>7}\n")
    for k in KEY_ARMS:
        e_theta = thompson.alpha[k] / (thompson.alpha[k] + thompson.beta[k])
        e_profit = PRICES[k] * e_theta
        f.write(f"  {k:>4} {PRICES[k]:>6.2f} {thompson.alpha[k]:>8.1f} "
                f"{thompson.beta[k]:>8.1f} {e_theta:>8.4f} {e_profit:>10.4f} "
                f"{thompson.counts[k]:>7.0f}\n")
    f.write("\n")


# =============================================================================
# Main Diagnostic Run
# =============================================================================

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "structural_pricing_misra_diagnostic.txt")

    print(f"Diagnostic run: seed={SEED}, T={T:,}, checkpoints every {CHECKPOINT_INTERVAL:,}")
    print(f"Output: {out_path}")
    sys.stdout.flush()

    # Setup
    rng = np.random.RandomState(SEED)
    demand_model = SegmentedDemandModel(N_SEGMENTS, DELTA_TRUE, seed=SEED)

    true_profits = np.array([demand_model.true_profit(p) for p in PRICES])
    opt_arm = np.argmax(true_profits)
    print(f"Optimal arm: {opt_arm} (price {PRICES[opt_arm]:.2f}, profit {true_profits[opt_arm]:.4f})")

    segment_ids = rng.choice(N_SEGMENTS, size=T, p=demand_model.segment_weights)
    valuation_offsets = rng.uniform(-DELTA_TRUE, DELTA_TRUE, size=T)

    ucb_pi = UCB_PI_Diag(PRICES, N_SEGMENTS, demand_model.segment_weights)
    ucb_pi_tuned = UCB_PI_Tuned_Diag(PRICES, N_SEGMENTS, demand_model.segment_weights)
    thompson = ThompsonSampling(K, PRICES)

    algorithms = {
        'ucb_pi': ucb_pi,
        'ucb_pi_tuned': ucb_pi_tuned,
        'thompson': thompson,
    }

    cumulative = {name: 0.0 for name in algorithms}

    with open(out_path, 'w') as f:
        # Header
        f.write("STRUCTURAL PRICING DIAGNOSTIC -- UCB-PI vs Thompson\n")
        f.write(f"Seed: {SEED}  T: {T:,}  K: {K}  S: {N_SEGMENTS:,}  delta_true: {DELTA_TRUE}\n")
        f.write(f"Optimal arm: {opt_arm} (price {PRICES[opt_arm]:.2f}, "
                f"profit {true_profits[opt_arm]:.4f})\n")

        # Main loop
        for t in range(T):
            segment_id = segment_ids[t]
            v_s = demand_model.segment_valuations[segment_id]
            v_i = v_s + valuation_offsets[t]

            for name, alg in algorithms.items():
                arm = alg.select_arm(t)
                price = PRICES[arm]
                purchased = (v_i >= price)
                reward = price if purchased else 0.0
                alg.update(arm, reward, segment_id, purchased)
                cumulative[name] += reward

            # Checkpoint
            if (t + 1) % CHECKPOINT_INTERVAL == 0:
                log_checkpoint(t + 1, ucb_pi, ucb_pi_tuned, thompson, demand_model, f)

                # Print progress to console
                print(f"  t={t+1:>7,}  delta_hat={ucb_pi.delta_hat:.4f}  "
                      f"dom={ucb_pi._last_dominated.sum():>3}  "
                      f"cum: PI={cumulative['ucb_pi']:.0f} "
                      f"PI-t={cumulative['ucb_pi_tuned']:.0f} "
                      f"TS={cumulative['thompson']:.0f}")
                sys.stdout.flush()

        # Final summary
        f.write(f"\n{'=' * 70}\n")
        f.write(f"FINAL SUMMARY\n")
        f.write(f"{'=' * 70}\n\n")
        for name in algorithms:
            f.write(f"  {name}: cumulative profit = {cumulative[name]:.0f}\n")

        # Compute oracle profit for this seed
        oracle_profit = 0.0
        for t in range(T):
            v_i = demand_model.segment_valuations[segment_ids[t]] + valuation_offsets[t]
            if v_i >= PRICES[opt_arm]:
                oracle_profit += PRICES[opt_arm]
        f.write(f"  oracle (computed): cumulative profit = {oracle_profit:.0f}\n\n")

        for name in algorithms:
            ratio = 100 * cumulative[name] / oracle_profit if oracle_profit > 0 else 0
            f.write(f"  {name}: {ratio:.2f}% of oracle\n")

        f.write(f"\n  Final delta_hat (UCB-PI): {ucb_pi.delta_hat:.6f}\n")
        f.write(f"  Final delta_hat (UCB-PI-tuned): {ucb_pi_tuned.delta_hat:.6f}\n")
        f.write(f"  Optimal arm dominated (UCB-PI)? "
                f"{'YES' if ucb_pi._last_dominated[opt_arm] else 'NO'}\n")
        f.write(f"  Optimal arm dominated (UCB-PI-tuned)? "
                f"{'YES' if ucb_pi_tuned._last_dominated[opt_arm] else 'NO'}\n")

    print(f"\nDiagnostic complete. Output: {out_path}")
