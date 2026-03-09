# Misra, Schwartz & Abernethy (2019) - Deep Summary

**Paper**: "Dynamic Online Pricing with Incomplete Information Using Multiarmed Bandit Experiments"
**Journal**: Marketing Science, Vol. 38, No. 2, pp. 226-252

---

## 1. Problem Setting

Online retailers face a unique pricing challenge:
- Must set real-time prices for millions of products
- Incomplete information about each product's demand curve
- Need to balance **earning** (exploiting best known price) vs **learning** (exploring to find optimal price)

Traditional approach ("learn then earn"): Run balanced field experiment, then set optimal price.
**Problem**: High opportunity cost during learning phase.

**Their solution**: UCB-PI - extends multiarmed bandits with economic theory (partial identification).

---

## 2. Demand Model Assumptions

### 2.1 Consumer Side (Section 2.1, p. 231)

Each consumer has:
1. **Stable preferences** v_i (doesn't change over time)
2. **Stable budget**
3. **Stable outside option**
4. **WARP** (Weak Axiom of Revealed Preference): if buys at price p, would buy at any p' < p

**Key formulation**: For consumer i in segment s:
```
v_i = v_s + n_i,  where n_i ∈ [-δ, δ]
```

Thus:
```
v_i ∈ [v_s - δ, v_s + δ]  ∀i ∈ s       (Equation 1)
```

- v_s = segment midpoint valuation (unknown)
- δ = within-segment heterogeneity (unknown, same across all segments)
- S segments, with segment proportions ψ_s

**Critical assumption**: δ is the SAME across all segments. This allows cross-sectional learning.

### 2.2 Firm Side

- Monopolist, constant marginal cost (normalized to 0)
- Does NOT know consumer valuations
- Only knows valuations are in [v_L, v_H]
- Observes consumer segment membership
- Can change prices frequently (Amazon: every 15 minutes)

---

## 3. Partial Identification from Price Experiments (Section 2.3)

### 3.1 Per-Segment Learning (p. 234)

After observing segment s's demand at various prices, define:

```
p^min_{s,t} = max{p_k | D(p_k)_{s,t} = 1}   (highest price where ALL in segment bought)
p^max_{s,t} = min{p_k | D(p_k)_{s,t} = 0}   (lowest price where NONE in segment bought)
```

**Key insight**: From WARP, we know:
- If D(p_k)_{s,t} = 1 → all consumers in s have v_i ≥ p_k
- If D(p_k)_{s,t} = 0 → all consumers in s have v_i < p_k

Therefore: v_{i,s} ∈ [p^min_{s,t}, p^max_{s,t}] for all i in segment s.

### 3.2 Estimating Segment Midpoint and δ (p. 235)

```
v̂_{s,t} = (p^max_{s,t} + p^min_{s,t}) / 2    (midpoint estimate)
δ̂_{s,t} = (p^max_{s,t} - p^min_{s,t}) / 2    (segment-level δ estimate)
```

**Global δ estimation**:
```
δ̂_t = max{δ̂_{s,t} | s ∈ S} + γ̂_t    (bias correction)
```

Where bias correction: γ̂_t = Σ_s (max{δ̂_{s,t}} - δ̂_{s,t}) × f(δ̂_{s,t})

### 3.3 Valuation Bounds per Segment

```
v̂^min_{s,t} = v̂_{s,t} - δ̂_t
v̂^max_{s,t} = v̂_{s,t} + δ̂_t
```

So: H_t[v_{i,s}] = [v̂^min_{s,t}, v̂^max_{s,t}]

### 3.4 Aggregate Demand Bounds (Equation 5, p. 235)

```
H_t[D(p_k)] = [Σ_s ψ_s · 1(v̂^min_{s,t} ≥ p_k),  Σ_s ψ_s · 1(v̂^max_{s,t} ≥ p_k)]
               └─────────────────────────────┘  └─────────────────────────────┘
                      Lower Bound                      Upper Bound
```

### 3.5 Profit Bounds (Equation 6, p. 237)

```
LB_t(π(p_k)) = p_k × Σ_s ψ_s · 1(v̂^min_{s,t} ≥ p_k)
UB_t(π(p_k)) = p_k × Σ_s ψ_s · 1(v̂^max_{s,t} ≥ p_k)
```

---

## 4. UCB-PI Algorithm

### 4.1 UCB-PI-untuned (Equation 7, p. 237)

```
UCB-PI_{kt} = π̄_{kt} + p_k × √(2 log t / n_{kt})   if UB_t(π(p_k)) > max_l LB_t(π(p_l))
            = 0                                      otherwise (price is DOMINATED)
```

**Two key innovations over standard UCB1**:

1. **Dominance elimination**: Turn off prices where upper bound ≤ best lower bound
2. **Price-scaled exploration bonus**: Multiply by p_k because π(p_k) ∈ [0, p_k]

### 4.2 UCB-PI-tuned (Equation 9, p. 238)

```
V_{kt} = (1/n_{kt} Σ π²_{kτ}) - π̄²_{kt} + √(2 log t / n_{kt})

UCB-PI-tuned_{kt} = π̄_{kt} + 2 × p_k × δ̂ × √((log t / n_{kt}) × min(1/4, V_{kt}))
                    if not dominated, else 0
```

The 2δ̂ factor accounts for the partially identified interval size.

---

## 5. Theoretical Results

**Main Theorem (p. 237)**: UCB-PI achieves log-regret bound:

```
E[Regret_T(UCB-PI)] ≤ 8 × Σ_{k≠1} (p_k × log T) / Δ_k + O(1)
```

Where Δ_k = π* - π(p_k) is the suboptimality gap.

**Key insight**: Regret scales with p_k (which is ≤ 1 after normalization), so UCB-PI has strictly lower regret than UCB1 (which uses 1 instead of p_k).

---

## 6. Why Our Implementation Underperforms

### 6.1 The Fundamental Mismatch

The paper's partial identification requires observing **aggregate segment demand rates**:
- D(p_k)_{s,t} = 1 means 100% of segment s purchased at price p_k
- D(p_k)_{s,t} = 0 means 0% of segment s purchased at price p_k

**Our implementation**: We observe individual binary outcomes (buy/not buy) for one consumer per round.

With 20 segments and T=20,000 rounds:
- Average observations per segment: 1,000
- But spread across K=20 prices
- Average observations per (segment, price) pair: ~50

**Problem**: We rarely observe enough consumers from the same segment at the same price to determine D(p_k)_{s,t} = 0 or 1. We mostly see D(p_k)_{s,t} ∈ (0, 1).

### 6.2 The δ Estimation Problem

The paper's δ̂ estimation requires:
1. Observing p^min_s (highest price where ALL purchased) - requires seeing multiple purchases
2. Observing p^max_s (lowest price where NONE purchased) - requires seeing multiple non-purchases

**Our implementation**: We update bounds based on single observations:
- One purchase at p → p^min_s = max(p^min_s, p)
- One non-purchase at p → p^max_s = min(p^max_s, p)

This conflates **individual** purchase decisions with **segment-level** demand.

**Result**: δ̂ ≈ 0.7 when true δ = 0.2 (massive overestimate)

### 6.3 The Discrete Price Grid Problem

Paper's simulation (Section 4): K = 100 prices from $0 to $1 in 1¢ increments.

**Our simulation**: K = 20 prices from $0.50 to $10.00 in $0.50 increments.

With price spacing of $0.50 and true δ = $0.20:
- Even perfect identification would give δ̂ ≥ 0.25 (half the price gap)
- The price grid is too coarse relative to δ

### 6.4 The Paper's Actual Setup

From Section 4 (p. 238):
- K = 100 prices
- S = 1,000 segments
- T = 200,000 rounds
- Price changes every 10 consumers
- δ = 0.1 (10% of price range)

**Critical**: With 1,000 segments and 200,000 rounds, each segment sees ~200 consumers on average. This is enough to estimate segment-level demand rates.

### 6.5 Why UCB1 Outperforms in Our Setup

UCB1 doesn't attempt partial identification. It just:
1. Estimates π̄_{kt} directly from observed profits
2. Adds exploration bonus √(2 log t / n_{kt})

**Advantage**: No dependence on correctly estimating δ or segment bounds.

UCB-PI's partial identification HELPS when:
- Many segments (S large)
- Many observations per segment (T/S large)
- Fine price grid (small spacing relative to δ)
- Good δ estimation

UCB-PI HURTS when:
- Few observations per (segment, price) pair
- Coarse price grid
- δ overestimated → demand bounds too wide → fewer prices dominated → more exploration

---

## 7. Recommendations to Fix Implementation

### Option A: Match Paper's Setup
- Increase S to 1,000 segments
- Increase K to 100 prices
- Increase T to 200,000+ rounds
- Set δ = 0.1 (10% of price range)

### Option B: Change the Partial Identification Logic
- Don't update p^min_s / p^max_s from single observations
- Require multiple observations: only update when confident D(p)_s = 0 or 1
- Use Bayesian inference on segment-level demand rates

### Option C: Simplify to Match Paper's Spirit
- The paper's key insight: **demand is monotonic** (WARP)
- Exploit monotonicity without full segment tracking
- Example: if price p_k has UB(profit) ≤ LB(profit) at some p_l, eliminate p_k

---

## 8. Key Equations Summary

| Concept | Equation |
|---------|----------|
| Valuation bounds | v_i ∈ [v_s - δ, v_s + δ] |
| Segment midpoint | v̂_s = (p^max_s + p^min_s) / 2 |
| Segment δ | δ̂_s = (p^max_s - p^min_s) / 2 |
| Global δ | δ̂ = max(δ̂_s) + bias_correction |
| Demand LB | LB(D(p_k)) = Σ_s ψ_s · 1(v̂_s - δ̂ ≥ p_k) |
| Demand UB | UB(D(p_k)) = Σ_s ψ_s · 1(v̂_s + δ̂ ≥ p_k) |
| Profit LB | LB(π(p_k)) = p_k × LB(D(p_k)) |
| Profit UB | UB(π(p_k)) = p_k × UB(D(p_k)) |
| Dominance | Price k dominated if UB(π(p_k)) ≤ max_l LB(π(p_l)) |
| UCB-PI index | π̄_{kt} + p_k × √(2 log t / n_{kt}) if not dominated |

---

## 9. Paper's Simulation Results (Section 4-5)

| Algorithm | Ex Post Profit (mean) | Range |
|-----------|----------------------|-------|
| UCB-PI-tuned | 96% | 91-99% |
| Learn-then-earn (5%) | 95% | 66-100% |
| UCB-tuned | 89% | 86-91% |
| UCB1-untuned | 67-78% | varies |

**Key finding**: UCB-PI has both higher mean AND lower variance than alternatives.

In ZipRecruiter calibration (Section 5):
- UCB-PI: 98% of optimal, range 96-99%
- Learn-then-earn (7.9%): 94% of optimal, range 83-97%
- UCB-PI achieves **43% higher profits during first month** of testing

---

## 10. Implementation Details

### 10.1 Algorithm Pseudocode (UCB-PI)

```python
# Initialization
for k in 1..K:
    play price p_k once
    observe segment s, reward r_k
    n_k = 1
    sum_k = r_k
    sum_sq_k = r_k^2

for s in 1..S:
    p_min[s] = v_L  # lowest possible valuation
    p_max[s] = v_H  # highest possible valuation

# Main loop
for t in K+1..T:
    # Update partial identification bounds
    for s in 1..S:
        v_hat[s] = (p_max[s] + p_min[s]) / 2
        delta_hat[s] = (p_max[s] - p_min[s]) / 2

    delta_hat_global = max(delta_hat) + bias_correction()

    # Compute profit bounds for each price
    for k in 1..K:
        LB[k] = p_k * sum_s(psi_s * indicator(v_hat[s] - delta_hat_global >= p_k))
        UB[k] = p_k * sum_s(psi_s * indicator(v_hat[s] + delta_hat_global >= p_k))

    best_LB = max(LB)

    # Compute UCB index with dominance elimination
    for k in 1..K:
        if UB[k] <= best_LB:
            ucb[k] = 0  # DOMINATED - never play
        else:
            mean_k = sum_k / n_k
            ucb[k] = mean_k + p_k * sqrt(2 * log(t) / n_k)

    # Play highest UCB price
    k_star = argmax(ucb)
    observe segment s_t, reward r_t

    # Update statistics
    n[k_star] += 1
    sum[k_star] += r_t
    sum_sq[k_star] += r_t^2

    # Update segment bounds based on observation
    if r_t > 0:  # purchased
        p_min[s_t] = max(p_min[s_t], p[k_star])
    else:  # did not purchase
        p_max[s_t] = min(p_max[s_t], p[k_star])
```

### 10.2 Bias Correction for δ̂ (Equation 8, p. 237)

The raw max estimator δ̂_t = max_s{δ̂_{s,t}} is biased upward. Bias correction:

```
γ̂_t = Σ_s w_s × (δ̂_max - δ̂_{s,t})

where:
  δ̂_max = max_s{δ̂_{s,t}}
  w_s = n_s / Σ_s' n_s'  (weight by observations per segment)
```

The paper notes this is a "shrinkage" estimator that pulls toward the mean.

### 10.3 UCB-Tuned Variance Estimation (Equation 9)

For UCB-tuned variants, compute empirical variance:

```python
# Per-price variance estimate
V_k = (1/n_k) * sum_sq_k - (sum_k / n_k)^2 + sqrt(2 * log(t) / n_k)

# Clipped variance (never exceeds 1/4 for [0,1] rewards)
V_k_clipped = min(1/4, V_k)

# UCB-tuned index
ucb_tuned[k] = mean_k + sqrt((log(t) / n_k) * V_k_clipped)
```

For UCB-PI-tuned, scale by 2δ̂ instead of 1:

```python
ucb_pi_tuned[k] = mean_k + 2 * p_k * delta_hat * sqrt((log(t) / n_k) * V_k_clipped)
```

### 10.4 Dominance Testing Implementation

```python
def is_dominated(k, UB, LB):
    """Check if price k is dominated by any other price."""
    best_LB = max(LB[l] for l in range(K) if l != k)
    return UB[k] <= best_LB

def get_active_prices(UB, LB):
    """Return indices of non-dominated prices."""
    active = []
    for k in range(K):
        if not is_dominated(k, UB, LB):
            active.append(k)
    return active
```

### 10.5 Price Grid Construction

Paper recommends:
1. **Fine grid**: K ≥ 50 prices for good resolution
2. **Span full range**: prices from v_L to v_H
3. **Uniform spacing**: Δp = (v_H - v_L) / (K-1)
4. **Normalization**: Scale prices to [0,1] for theoretical guarantees

```python
# Paper's setup
K = 100
v_L, v_H = 0.0, 1.0
prices = np.linspace(v_L, v_H, K)  # 1-cent increments
```

### 10.6 Segment Proportion Estimation

If segment proportions ψ_s unknown, estimate from data:

```python
# Track arrivals per segment
segment_counts = np.zeros(S)
for t in range(T):
    s_t = observe_segment()
    segment_counts[s_t] += 1

# Estimate proportions
psi_hat = segment_counts / segment_counts.sum()
```

---

## 11. Heuristic Methods and Baselines

### 11.1 Learn-Then-Earn (LTE)

Traditional approach: dedicate fraction τ of horizon to pure exploration.

```python
# Parameters
tau = 0.05  # 5% of time for exploration (paper's default)
T_explore = int(tau * T)

# Exploration phase: uniform random pricing
for t in range(T_explore):
    k = random.choice(range(K))
    play(prices[k])
    update_statistics(k)

# Exploitation phase: play empirical best
k_best = argmax([sum_k / n_k for k in range(K)])
for t in range(T_explore, T):
    play(prices[k_best])
```

**Trade-off**: Higher τ → better learning but more regret during exploration.
Paper finds τ = 5-10% works well in practice.

### 11.2 ε-Greedy

```python
epsilon = 0.1  # exploration probability

for t in range(T):
    if random.random() < epsilon:
        k = random.choice(range(K))  # explore
    else:
        k = argmax([sum_k / n_k for k in range(K)])  # exploit
    play(prices[k])
```

**Issue**: Linear regret O(εT) vs logarithmic for UCB.

### 11.3 Thompson Sampling for Pricing

Not in original paper, but natural alternative:

```python
# Beta prior on demand at each price
alpha = np.ones(K)  # successes + 1
beta_param = np.ones(K)  # failures + 1

for t in range(T):
    # Sample demand from posterior
    theta = [random.beta(alpha[k], beta_param[k]) for k in range(K)]

    # Compute expected profit under sample
    expected_profit = [prices[k] * theta[k] for k in range(K)]

    # Play price with highest sampled profit
    k = argmax(expected_profit)
    play(prices[k])

    # Update posterior
    if purchased:
        alpha[k] += 1
    else:
        beta_param[k] += 1
```

### 11.4 Monotonicity-Constrained Estimation

Exploit WARP without full PI tracking:

```python
# Isotonic regression on empirical demand
from sklearn.isotonic import IsotonicRegression

# Empirical demands (decreasing in price)
d_hat = [sum_k / n_k for k in range(K)]

# Fit monotone decreasing function
ir = IsotonicRegression(increasing=False)
d_monotone = ir.fit_transform(prices, d_hat)

# Expected profits under monotone demand
profit_monotone = [prices[k] * d_monotone[k] for k in range(K)]
```

### 11.5 KL-UCB for Bounded Rewards

More sophisticated UCB using KL divergence:

```python
def kl_divergence(p, q):
    """KL divergence for Bernoulli distributions."""
    if p == 0:
        return -np.log(1 - q)
    if p == 1:
        return -np.log(q)
    return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

def kl_ucb_index(mean_k, n_k, t, c=3):
    """Compute KL-UCB index."""
    threshold = (np.log(t) + c * np.log(np.log(t))) / n_k

    # Binary search for largest q such that kl(mean_k, q) <= threshold
    lo, hi = mean_k, 1.0
    for _ in range(50):  # sufficient precision
        mid = (lo + hi) / 2
        if kl_divergence(mean_k, mid) <= threshold:
            lo = mid
        else:
            hi = mid
    return lo
```

---

## 12. Practical Implementation Considerations

### 12.1 Cold Start Problem

When n_k = 0 for some prices:
- **Option 1**: Initialize each price once (paper's approach)
- **Option 2**: Use UCB default of infinity for unplayed arms
- **Option 3**: Prior mean (e.g., assume 50% conversion)

### 12.2 Non-Stationary Demand

Paper assumes stationarity. For non-stationary:
- **Sliding window**: Only use last W observations
- **Discounting**: Weight recent observations more heavily
- **Change detection**: Reset when distribution shift detected

```python
# Sliding window UCB
W = 1000  # window size
recent_rewards = deque(maxlen=W)
recent_arms = deque(maxlen=W)

# Compute statistics only from window
for k in range(K):
    mask = [a == k for a in recent_arms]
    n_k = sum(mask)
    if n_k > 0:
        sum_k = sum(r for r, m in zip(recent_rewards, mask) if m)
```

### 12.3 Batched Updates

Real systems may update less frequently than every customer:

```python
batch_size = 10  # update every 10 customers

batch_rewards = []
batch_arms = []

for t in range(T):
    # Play current best
    k = select_arm()
    r = play(prices[k])

    batch_rewards.append(r)
    batch_arms.append(k)

    if len(batch_rewards) >= batch_size:
        # Batch update
        for r, k in zip(batch_rewards, batch_arms):
            update_statistics(k, r)
        update_partial_identification()
        batch_rewards = []
        batch_arms = []
```

### 12.4 Multiple Products

Extend to M products with shared δ:

```python
# Separate tracking per product
for m in range(M):
    n[m, k] = 0
    sum_rewards[m, k] = 0
    p_min[m, s] = v_L
    p_max[m, s] = v_H

# Global δ estimated across all products
delta_estimates = []
for m in range(M):
    for s in range(S):
        delta_estimates.append((p_max[m,s] - p_min[m,s]) / 2)
delta_hat_global = max(delta_estimates) + bias_correction()
```

### 12.5 Computational Complexity

Per-round complexity:
- UCB1: O(K) for argmax
- UCB-PI: O(K × S) for profit bounds computation
- UCB-PI with dominance: O(K × S) but fewer active arms over time

Memory:
- UCB1: O(K) for arm statistics
- UCB-PI: O(K + S) for arm statistics + segment bounds

---

## 13. Diagnostic Metrics

### 13.1 Regret Decomposition

```python
# Instantaneous regret
regret_t = optimal_profit - actual_profit_t

# Cumulative regret
cumulative_regret = sum(regret_t for t in range(T))

# Regret by source
exploration_regret = sum(regret_t for t where arm != optimal)
estimation_regret = cumulative_regret - exploration_regret
```

### 13.2 Arm Selection Frequency

```python
# Fraction of time at optimal
frac_optimal = n[k_optimal] / T

# Convergence: final arm selection
final_arm = mode(arms[-1000:])  # last 1000 plays
converged_to_optimal = (final_arm == k_optimal)
```

### 13.3 Partial Identification Quality

```python
# δ estimation error
delta_error = abs(delta_hat - delta_true) / delta_true

# Bound tightness (smaller = better)
bound_width = mean(UB - LB)

# Dominated arm count over time
dominated_count_t = K - len(active_prices_t)
```

### 13.4 Expected vs Realized Performance

```python
# Ex-ante optimal (before experiment)
ex_ante_profit = T * max(p_k * E[D(p_k)] for k in range(K))

# Ex-post optimal (with hindsight)
ex_post_profit = T * max(empirical_profit[k] for k in range(K))

# Algorithm performance
algorithm_profit = sum(rewards)

# Ratios
ex_post_ratio = algorithm_profit / ex_post_profit
ex_ante_ratio = algorithm_profit / ex_ante_profit
```
