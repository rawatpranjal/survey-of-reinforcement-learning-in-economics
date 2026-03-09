# Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher

Rust, 1987

## Citation
Rust, J. (1987). Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher. Econometrica, 55(5), 999-1033.

## Context
This paper introduced the nested fixed-point (NFXP) algorithm for estimating dynamic discrete choice models. Harold Zurcher is the (pseudonymous) superintendent of a bus fleet in Madison, Wisconsin. The paper models his monthly replacement decisions as solutions to a dynamic programming problem with unknown structural parameters, then estimates those parameters from observed replacement data.

## Problem Formulation

### Decision Problem
Each month, the fleet manager observes the accumulated mileage $x_t$ on each bus engine and decides whether to:
- $a_t = 0$: Continue operating the engine
- $a_t = 1$: Replace the engine (reset mileage to 0)

### State Space
State $x_t$ is discretized accumulated mileage since last replacement:
$$x_t \in \{0, 1, 2, \ldots, M-1\}$$

where each bin represents a mileage interval (e.g., 5,000 miles per bin). Rust uses $M = 90$ bins, covering 0 to 450,000 miles.

### Transition Dynamics
Mileage evolves stochastically:
$$x_{t+1} = x_t + \xi_t$$

where $\xi_t \in \{0, 1, 2, 3\}$ is the random mileage increment. Rust estimates the transition probabilities $\theta_3 = \{\theta_{30}, \theta_{31}, \theta_{32}\}$ from data:
- $P(\xi = 0) = \theta_{30}$
- $P(\xi = 1) = \theta_{31}$
- $P(\xi = 2) = \theta_{32}$
- $P(\xi = 3) = 1 - \theta_{30} - \theta_{31} - \theta_{32}$

### Cost Function
The per-period utility (negative cost) is:
$$u(x_t, a_t, \theta_1) = \begin{cases}
-c(x_t, \theta_1) & \text{if } a_t = 0 \text{ (continue)} \\
-RC - c(0, \theta_1) & \text{if } a_t = 1 \text{ (replace)}
\end{cases}$$

where:
- $c(x, \theta_1) = \theta_{11} x$ (linear maintenance cost)
- $RC$ = replacement cost

Rust also considers $c(x, \theta_1) = \theta_{11} x + \theta_{12} x^2$ (quadratic).

### Bellman Equation
The value function satisfies:
$$V(x) = \max_{a \in \{0,1\}} \left\{ u(x, a, \theta_1) + \epsilon_a + \beta \mathbb{E}[V(x') | x, a] \right\}$$

where $\epsilon_a$ is a Type I extreme value shock (logit errors). Under the extreme value assumption, the expected value function is:
$$\bar{V}(x) = \log\left( \exp(v_0(x)) + \exp(v_1(x)) \right) + \gamma$$

where $v_a(x) = u(x, a) + \beta \mathbb{E}[\bar{V}(x') | x, a]$ and $\gamma$ is Euler's constant.

### Choice Probabilities
The probability of replacement given state $x$:
$$P(a = 1 | x) = \frac{\exp(v_1(x))}{\exp(v_0(x)) + \exp(v_1(x))}$$

## Estimation

### Nested Fixed Point Algorithm
1. **Outer loop**: Search over structural parameters $\theta = (\theta_1, RC)$
2. **Inner loop**: For each $\theta$, solve the fixed-point problem $\bar{V} = \Gamma(\bar{V}; \theta)$ via value iteration
3. **Likelihood**: Compute log-likelihood of observed choices given $\bar{V}(\theta)$
4. **Optimize**: Use Newton-Raphson to find $\hat{\theta}$

### Data
- 162 buses in 8 groups
- Monthly observations from December 1974 to May 1985
- Total: 15,072 bus-month observations
- 4,256 observed replacements

## Results

### Parameter Estimates (Linear Cost Specification)
| Parameter | Estimate | Std. Error |
|-----------|----------|------------|
| $\theta_{11}$ (mileage cost) | 0.00107 | 0.00011 |
| $RC$ (replacement cost) | 9.76 | 0.53 |

(Costs in units of $\$ \times 10^{-3}$, so $RC \approx \$9,760$)

### Discount Factor
Rust estimates or calibrates $\beta = 0.9999$ (monthly discount).

### Transition Probabilities
| Parameter | Estimate |
|-----------|----------|
| $\theta_{30}$ | 0.392 |
| $\theta_{31}$ | 0.562 |
| $\theta_{32}$ | 0.046 |

### Model Fit
The model correctly predicts the steep increase in replacement probability at high mileages. Replacement hazard rises from near 0 at low mileage to approximately 0.15 at 250,000+ miles.

## MDP Specification Summary

| Component | Specification |
|-----------|---------------|
| States | Discretized mileage $x \in \{0, \ldots, 89\}$ |
| Actions | $\{0, 1\}$ (continue, replace) |
| Transition | Stochastic mileage increments |
| Reward | $-c(x) - RC \cdot a$ plus logit shock |
| Horizon | Infinite |
| Discount | $\beta = 0.9999$ |

## Significance

1. **Foundational for structural econometrics**: The NFXP algorithm became the standard approach for estimating DDC models.

2. **Validates rational behavior**: Observed replacement patterns are consistent with forward-looking optimization.

3. **Connects economics and RL**: The Bellman equation is the same; the difference is that economics estimates parameters while RL learns policies.

4. **Benchmark problem**: Bus engine replacement remains a standard testbed for DDC estimation methods and RL algorithms applied to economic models.

## Relation to Chapter
The bus engine problem is the canonical economics benchmark for durable goods replacement. We use a multi-engine fleet extension where the state space grows exponentially with fleet size, creating a setting where exact DP becomes infeasible and RL methods are needed.
