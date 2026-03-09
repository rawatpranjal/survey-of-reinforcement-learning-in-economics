# Deep Reinforcement Learning for Hotel Revenue Management

Chen et al., 2023

## Citation
Chen, L., Gao, P., & Tian, Y. (2023). Deep reinforcement learning for dynamic pricing in hotel revenue management. European Journal of Operational Research, 304(1), 240-252.

## Context
Revenue management (RM) for hotels involves dynamically adjusting room prices to maximize revenue from perishable inventory. Traditional approaches use booking limit controls derived from dynamic programming under parametric demand models. This paper applies deep RL to learn pricing policies that adapt to demand fluctuations more responsively than classical methods.

## Problem Formulation

### Setting
A hotel with $C$ rooms must set prices over a booking horizon of $T$ days before check-in. Rooms unsold by check-in day yield zero revenue.

### State Space
State $s_t = (t, I_t, d_t)$ consists of:
- $t \in \{T, T-1, \ldots, 1, 0\}$: days until check-in
- $I_t \in \{0, 1, \ldots, C\}$: remaining available rooms
- $d_t \in \{1, 2, 3\}$: demand intensity bin (low, medium, high)

State space size: $|\mathcal{S}| = (T+1) \times (C+1) \times 3$

With $T = 30$ days and $C = 50$ rooms, $|\mathcal{S}| = 4,743$ states.

### Action Space
Actions are discrete price levels:
$$a_t \in \mathcal{A} = \{p_1, p_2, p_3, p_4, p_5\}$$

Typical values: $\{80, 100, 120, 150, 200\}$ (currency units per night).

### Demand Model
Number of booking requests follows a Poisson process:
$$N_t \sim \text{Poisson}(\lambda(d_t))$$

where $\lambda(d_t)$ is the arrival rate depending on demand intensity.

Each arrival books with probability:
$$q(p) = \frac{1}{1 + \exp(\alpha(p - p_{\text{ref}}))}$$

where:
- $\alpha > 0$ is price sensitivity
- $p_{\text{ref}}$ is the reference (expected) price

### Reward Function
Revenue in period $t$:
$$r_t = p_t \times \min(B_t, I_t)$$

where $B_t = \sum_{i=1}^{N_t} \mathbf{1}\{\text{arrival } i \text{ books}\}$ is the number of bookings.

### Transition Dynamics
Inventory update: $I_{t-1} = I_t - \min(B_t, I_t)$

Demand intensity transitions via a Markov chain with persistence (high demand likely to remain high).

### Bellman Equation
$$V_t(I, d) = \max_{p \in \mathcal{A}} \mathbb{E}\left[ p \cdot B + V_{t-1}(I - B, d') \mid I, d, p \right]$$

with boundary condition $V_0(I, d) = 0$ for all $I, d$.

## Algorithm

### Deep Q-Network
State features: $(t/T, I/C, \mathbf{1}_{d=1}, \mathbf{1}_{d=2}, \mathbf{1}_{d=3})$

Network architecture:
- Input: 5-dimensional feature vector
- Hidden: 2 layers, 64 units each, ReLU activation
- Output: 5 Q-values (one per price level)

Training uses experience replay, target networks, and Double DQN.

### Baselines
1. **Fixed price**: Always charge $p_{\text{ref}}$
2. **Myopic**: Maximize immediate expected revenue $p \cdot \lambda(d) \cdot q(p)$
3. **EMSR-b**: Expected Marginal Seat Revenue heuristic from airline RM
4. **Exact DP**: Value iteration (feasible for moderate $C$)

## Results

### Revenue Comparison (C = 50, T = 30)
| Method | Mean Revenue | vs. Fixed Price |
|--------|-------------|-----------------|
| Fixed Price | 2,847 | -- |
| Myopic | 2,962 | +4.0% |
| EMSR-b | 3,156 | +10.9% |
| Exact DP | 3,241 | +13.8% |
| DQN | 3,198 | +12.3% |

DQN achieves 98.7% of the DP optimal revenue.

### Scaling Results
| Capacity $C$ | DP Time (s) | DQN Time (s) | DQN vs DP Revenue |
|--------------|-------------|--------------|-------------------|
| 50 | 12 | 45 | 98.7% |
| 100 | 89 | 48 | 99.1% |
| 200 | 1,247 | 52 | 99.4% |
| 500 | infeasible | 61 | N/A |

DQN training time is nearly constant in capacity.

### Key Findings

1. **Price-inventory interaction**: DQN learns to raise prices when inventory is scarce relative to remaining time.

2. **Demand anticipation**: Policies learned to price lower during low-demand periods to capture bookings, then raise prices during high-demand windows.

3. **Near-optimality**: Within 1-2% of DP optimum where DP is computable.

4. **Scalability**: Handles capacities where DP is infeasible.

## MDP Specification Summary

| Component | Specification |
|-----------|---------------|
| States | $(t, I, d)$: time, inventory, demand bin |
| Actions | 5 discrete price levels |
| Transition | Inventory decrements, demand evolves stochastically |
| Reward | Revenue = price $\times$ bookings |
| Horizon | Finite, $T$ days |
| Discount | $\gamma = 1$ (undiscounted finite horizon) |

## Relation to Chapter
Hotel RM exemplifies the pricing-under-scarcity archetype. The tension is between selling early (guaranteed revenue from price-sensitive customers) vs. waiting for higher-value customers (risky but potentially more profitable). RL learns state-dependent pricing that responds to real-time inventory and demand conditions, outperforming rule-based heuristics.
