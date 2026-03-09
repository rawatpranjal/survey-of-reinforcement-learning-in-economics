# Gijsbrechts et al. (2022): Can Deep Reinforcement Learning Improve Inventory Management?

**Full citation:** Gijsbrechts, Joren, Robert N. Boute, Jan A. Van Mieghem, and Dennis J. Zhang. "Can Deep Reinforcement Learning Improve Inventory Management? Performance on Dual Sourcing, Lost Sales, and Multi-Echelon Problems." *Manufacturing & Service Operations Management* 24(3): 1349-1368, 2022.

## Problem Statement

The paper systematically evaluates deep reinforcement learning against classical inventory control policies across three canonical problems where optimal policies are either known or well-approximated:

1. **Lost sales problem**: Demand that cannot be met is lost (not backordered)
2. **Dual sourcing problem**: Two suppliers with different lead times and costs
3. **Multi-echelon problem**: Serial supply chain with K stages

The key question: Can DRL discover policies that match or exceed carefully calibrated classical solutions?

## Classical Benchmark: Clark-Scarf Echelon Base-Stock

For the multi-echelon serial system, the classical benchmark is the echelon base-stock policy from Clark and Scarf (1960). Each stage k maintains an echelon inventory position (local inventory plus all downstream inventory and in-transit) and orders to bring this position up to a base-stock level $S_k$. The echelon formulation decouples the stages, making optimization tractable.

For a single-echelon system with lead time L, holding cost h, and backorder cost b, the optimal base-stock level solves the newsvendor critical fractile:
$$S^* = F^{-1}\left(\frac{b}{b+h}\right)$$

The Clark-Scarf result extends this to serial systems: under specific cost accounting, echelon base-stock policies are optimal.

## MDP Formulation

**State:** $(I_1, \ldots, I_K)$ where $I_k$ is on-hand inventory at stage k, plus pipeline inventory in transit

**Action:** Order quantity $q$ placed at the most upstream stage (stage K)

**Transitions:**
- Demand $D_t \sim F_D$ realized at stage 1 (customer-facing)
- Shipments flow downstream with lead times
- Inventory at stage 1 satisfies demand; excess demand is backordered or lost

**Costs per period:**
$$c_t = h \sum_{k=1}^{K} I_k^+ + b \cdot \max(0, D_t - I_1) + c_o \cdot q_t$$
where $I_k^+ = \max(0, I_k)$ is on-hand inventory, h is holding cost, b is backorder cost, and $c_o$ is ordering cost.

**Objective:** Minimize expected discounted total cost:
$$\min_\pi \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t c_t \mid \pi\right]$$

## State Space Growth

The curse of dimensionality is severe:
- K=2 echelons: Moderate state space
- K=4 echelons: ~10^5 states
- K=6 echelons: ~10^8 states

This motivates the appeal of DRL, which uses function approximation to handle large state spaces.

## Algorithms Tested

1. **A3C (Asynchronous Advantage Actor-Critic)**: Policy gradient with parallel workers
2. **PPO (Proximal Policy Optimization)**: Clipped policy gradient
3. **DQN (Deep Q-Network)**: Value-based with experience replay and target networks

All use feedforward neural networks with 2-3 hidden layers of 32-64 units.

## Key Results

### Lost Sales Problem
- DRL (PPO) achieves costs within 1-2% of optimal base-stock
- Performance degrades with longer lead times due to larger effective state space

### Dual Sourcing Problem
- DRL matches tailored base-surge heuristics in simple cases
- Struggles when lead time difference exceeds 2-3 periods

### Multi-Echelon Problem
- **K=2:** DRL matches echelon base-stock within 3%
- **K=3-4:** DRL costs 5-15% higher than tuned base-stock
- **K=6+:** DRL fails to converge; base-stock policies dominate

### Summary Table (from paper Table 5)

| Problem | Echelons | DRL Gap vs. Base-Stock |
|---------|----------|------------------------|
| Serial  | 2        | +1.2%                  |
| Serial  | 3        | +6.8%                  |
| Serial  | 4        | +12.3%                 |
| Serial  | 6        | Did not converge       |

## Key Insights

1. **Base-stock is a strong baseline.** For single-echelon and simple multi-echelon systems, properly calibrated base-stock policies are near-optimal. DRL adds little value when classical solutions exist.

2. **DRL struggles with multi-echelon coupling.** The inter-stage dependencies create complex credit assignment problems. An order at stage K affects costs at stage 1 many periods later.

3. **Hyperparameter sensitivity is severe.** DRL results vary substantially with network architecture, learning rate, and training duration. The authors report extensive tuning.

4. **Sample complexity is high.** DRL requires millions of transitions to approach base-stock performance that can be computed analytically.

5. **When DRL adds value:** Problems without closed-form solutions (non-stationary demand, complex constraints, lost sales with lead times) are where DRL has potential.

## Relevance to Chapter

This paper provides the empirical grounding for discussing RL in inventory management. The results are sobering: classical operations research methods, developed over 60 years, remain highly competitive. RL's value proposition in inventory is for problems where analytical solutions are unavailable or require unrealistic modeling assumptions.

The Gijsbrechts results mirror findings across RL applications: the method works when it works, but practitioners should first ask whether a simpler baseline exists.
