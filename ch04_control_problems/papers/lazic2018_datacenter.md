# Data Center Cooling Using Model-Predictive Control

Lazic et al., NeurIPS 2018

## Citation
Lazic, N., Boutilier, C., Lu, T., Wong, E., Roy, B., Ryu, M., & Imwalle, G. (2018). Data center cooling using model-predictive control. Advances in Neural Information Processing Systems, 31.

## Context and Deployment
Google deployed machine learning for autonomous data center cooling control starting in 2016. The system operates continuously in production across multiple Google data centers, controlling fans, chillers, pumps, and other HVAC equipment to maintain safe temperatures while minimizing energy consumption. At Google's scale, even small percentage improvements in Power Usage Effectiveness (PUE) translate to substantial energy savings.

## Problem Formulation

### State Space
The state vector includes over 100 sensor readings:
- Zone temperatures (server inlet, exhaust)
- Coolant temperatures (supply, return)
- Flow rates (water, air)
- External weather conditions (temperature, humidity)
- Server workload proxies (power draw per rack)

State dimension: $s_t \in \mathbb{R}^{d}$ where $d > 100$.

### Action Space
Actions are setpoints for cooling equipment:
- Chiller setpoints (temperature targets)
- Cooling tower fan speeds
- Pump flow rates
- Computer Room Air Handler (CRAH) fan speeds

Action dimension: $a_t \in \mathbb{R}^{m}$ where $m \approx 20$.

### Dynamics Model
The system learns a neural network dynamics model $\hat{f}$ from historical data:
$$s_{t+1} = \hat{f}(s_t, a_t) + \epsilon_t$$

The model is trained on millions of 5-minute intervals of operational data.

### Objective Function
The objective balances energy efficiency against safety:
$$J = \sum_{t=0}^{T} \left[ c_{\text{energy}}(a_t) + \lambda \cdot \text{penalty}(s_t) \right]$$

where:
- $c_{\text{energy}}(a_t)$ = energy cost of cooling actions (proportional to PUE)
- $\text{penalty}(s_t)$ = constraint violation penalty for temperatures exceeding thresholds
- $\lambda$ = large constant ensuring safety constraints dominate

### Safety Constraints
Hard constraints on zone temperatures:
$$T_{\min} \leq T_{\text{zone},i}(s_t) \leq T_{\max} \quad \forall i$$

Typical bounds: $T_{\min} = 18°C$, $T_{\max} = 27°C$ for server inlet temperatures.

## Algorithm

### Model-Predictive Control (MPC)
At each control interval (5 minutes):
1. Observe current state $s_t$
2. Use learned model $\hat{f}$ to simulate trajectories over horizon $H$
3. Optimize action sequence to minimize $J$ subject to safety constraints
4. Apply first action $a_t^*$, observe $s_{t+1}$, repeat

### Safe Exploration
The system uses a constrained optimization approach:
- Actions are projected onto the feasible set defined by safety constraints
- Uncertainty estimates from the learned model are used to maintain conservative margins
- Human operators can override at any time

### Online Learning
The dynamics model is continuously updated with new operational data, adapting to seasonal variations, equipment degradation, and workload changes.

## Results

### Energy Reduction
- Initial deployment (Gao 2014): 40% reduction in cooling energy
- Subsequent refinements (Lazic et al. 2018): Additional 12% improvement
- Overall PUE improvement: ~15%

### Numerical Results from Paper
| Metric | Baseline | MPC | Improvement |
|--------|----------|-----|-------------|
| PUE | 1.20 | 1.10 | 8.3% |
| Cooling energy (relative) | 1.00 | 0.60 | 40% |
| Temperature violations | Rare | Zero | N/A |

### Deployment Scale
- Operates 24/7 across multiple data centers
- Controls thousands of individual setpoints
- Processes millions of sensor readings per day

## Key Insights

1. **Model-based RL with safety**: The MPC approach provides interpretability and formal safety guarantees that pure model-free RL cannot.

2. **Domain knowledge integration**: Physics-informed features (thermal mass, heat transfer coefficients) improve model accuracy.

3. **Conservative action selection**: Maintaining safety margins is more important than maximizing efficiency; the system never sacrifices safety for performance.

4. **Continuous adaptation**: Online model updates handle non-stationarity (weather, load patterns, equipment aging).

## Relation to Chapter
This deployment demonstrates RL solving a high-dimensional continuous control problem where:
- State and action spaces are too large for tabular methods
- Safety constraints are paramount
- The system must operate continuously without human intervention
- Percentage improvements at scale yield massive absolute savings
