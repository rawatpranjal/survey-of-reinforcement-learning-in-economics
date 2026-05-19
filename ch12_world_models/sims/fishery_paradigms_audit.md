# Fishery Paradigms Sim — 7-Point Audit

Pre-implementation audit for `ch12_world_models/sims/fishery_paradigms.py`. Six paradigms on a single-agent fishery with logistic-growth stock dynamics. Audit must pass before code is shipped. The sim is the §9.2 panel of the dual simulation; it complements the cobweb panel by isolating sample efficiency in an exogenous environment (no self-referential price mechanism).

## 1. Algorithm Identity

Each paradigm matches its canonical source. The fishery action is a non-negative scalar harvest $h_t$; the state is a non-negative scalar stock $s_t$.

### Oracle (knows true parameters)

Solves the infinite-horizon discounted dynamic program by value iteration on a finite stock grid. The fishery is non-linear (logistic growth) but scalar, so DP on a one-dimensional grid is exact up to discretization error. The oracle has $(r, K, p, c, \sigma)$ and applies the greedy policy $g^\star(s) = \arg\max_h [p h - (c/2) h^2 + \gamma \mathbb{E} V^\star(s')]$ where $V^\star$ is the value function fixed point.

### Naive (no learning)

Constant rule $h_t = h_{\text{MSY}}$ where $h_{\text{MSY}} = r K / 4$ is the analytic maximum-sustainable-yield harvest of the deterministic logistic model. The agent is given the formula but not the parameters; for the audit's "no information leakage" requirement, naive is initialized with a guess at $r K / 4$ from rough priors and never updates. Concretely: the naive agent plays $h = 0.5$ (a reasonable steady-state guess that lies between the regime's MSY and zero).

### RLS adaptive learning

Marcet-Sargent style RLS, adapted to the non-linear logistic growth. The agent fits a linearized one-step growth equation $\Delta s_{t+1} \equiv s_{t+1} - s_t + h_t \approx \alpha_0 + \alpha_1 s_t + \eta_t$ by recursive least squares on accumulated $(s_t, h_t, s_{t+1})$ data, with the implicit identification $\alpha_0 \approx 0$, $\alpha_1 \approx r$ near $s = 0$, and a curvature correction $\alpha_2 s_t^2$ for the $-r s_t^2 / K$ term. Once $(\hat r, \hat K)$ are recovered from the regression, the agent re-solves the DP with point estimates and known cost parameters $(p, c)$ and plays the resulting greedy policy.

### Q-learning (tabular)

Discretize $s \in [0, 1.5 K]$ to a 30-point grid and $h$ to a 21-point grid over $[0, h_{\max}]$ with $h_{\max} = 1.5 \cdot h_{\text{MSY}}$. Standard $\varepsilon$-greedy with $\alpha = 0.1$, $\varepsilon$ decaying from $0.3$ to $0.01$ over the episode.

### Arifovic GA

Population of $N_{\text{pop}} = 30$ binary-encoded constant harvest rules with $L = 10$ bits encoding $h \in [0, h_{\max}]$. Selection by fitness, crossover with $p_c = 0.6$, mutation with $p_m = 0.0033$, election operator on. Generation length $10$ steps.

### MBPO-style MBRL

Learns $(r, K, p, c)$ jointly by least squares on accumulated $(s_t, h_t, s_{t+1}, r_t)$ data. The growth equation $s_{t+1} - s_t + h_t = r s_t (1 - s_t / K) + \eta_t$ is non-linear in parameters; for tractability the agent fits the linear-in-parameters form $\Delta s_t + h_t = r s_t - (r/K) s_t^2 + \eta_t$ with two coefficients $(r, r/K)$. Reward parameters $(p, c)$ are recovered from a linear regression of $r_t$ on $(h_t, h_t^2 / 2)$. With current estimates, the agent re-solves the DP and acts with Gaussian exploration noise around the planner output (std decays over the episode).

## 2. Environment / MDP Fidelity

Logistic-growth fishery per the chapter's §9.2 spec:

- **State.** $s_t \in [0, s_{\max}]$ with $s_{\max} = 1.5 K$ to allow for shocks above steady state.
- **Action.** $h_t \in [0, \min(s_t, h_{\max})]$ with $h_{\max} = 1.5 \cdot r K / 4$. The constraint $h_t \le s_t$ enforces non-negative stock after harvest.
- **Dynamics.** $s_{t+1} = \max(0, s_t + r s_t (1 - s_t / K) - h_t + \varepsilon_t)$ with $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$.
- **Reward.** $r_t = p h_t - (c / 2) h_t^2$.
- **Discount.** $\gamma = 0.95$.
- **Episode length.** $T = 500$ steps.
- **Parameters.** $r = 0.4$, $K = 10$, $p = 2.0$, $c = 0.2$, $\sigma = 0.3$. These give a deterministic MSY of $h_{\text{MSY}} = 1.0$ at the steady-state stock $s^\star = K/2 = 5$ and an oracle per-period reward at steady state of $p h_{\text{MSY}} - (c/2) h_{\text{MSY}}^2 = 2.0 - 0.1 = 1.9$ in units consistent with the cobweb regret scale.
- **Initial state.** $s_0 = K = 10$ (start at carrying capacity).

The environment is exogenous in the chapter's sense: the agent's action affects future state through the stock-depletion term $-h_t$ but not through expectations or self-referential pricing. The growth process $r s_t (1 - s_t / K)$ is fixed by the natural environment.

## 3. Data Integrity

Per-seed pipeline:

`compute_data()` → per-paradigm → per-seed → `rollout(paradigm, env, T, seed)` → list of $(s_t, h_t, r_t, s_{t+1})$ tuples → per-step reward vector → cumulative regret against the oracle's realized return on the same noise sequence. Every entry in the results table comes from `data['results'][paradigm][regime]['regret_curve']` aggregated across seeds; the table writer reads only from this dict. No hardcoded numbers.

## 4. Comparison Fairness

- All six paradigms see the same per-seed environment, same noise sequence, same episode length $T = 500$, same evaluation protocol (within-seed regret against the oracle's realized return on the same noise path).
- Q-learning's discrete action grid is sized to give resolution finer than $\sigma$: action grid $\Delta h = h_{\max} / 21 \approx 0.07$, which is below the noise floor.
- Regret is cumulative undiscounted reward gap, the same convention as the cobweb panel.
- All paradigms are initialized with the same initial stock $s_0 = K$.

## 5. Theoretical Sanity Checks

Before declaring trustworthy:

1. **Oracle DP fixed point.** The grid value iteration converges to within $10^{-6}$ in supremum norm in fewer than $200$ iterations.
2. **Oracle steady state.** The greedy policy applied to $s_0 = K$ drives stock toward the steady-state pair $(s^\star, h_{\text{MSY}}) = (5, 1)$ within $50$ steps in the deterministic limit ($\sigma = 0$).
3. **Oracle dominates.** Oracle cumulative regret is exactly zero per seed; every other paradigm's mean regret is positive.
4. **RLS recovers $(r, K)$.** After $T = 500$ steps, RLS's point estimates $(\hat r, \hat K)$ are within $10\%$ of the true values in mean across seeds.
5. **MBPO recovers all four parameters.** Similar tolerance on $(r, K)$; near-machine-precision on $(p, c)$ because the reward signal pins them down once $h_t$ varies.
6. **Sample-efficiency ordering matches cobweb.** Expected ordering by cumulative regret: Oracle, RLS, MBPO, Naive, Arifovic GA, Q-Learning, mirroring the cobweb panel's rank order.

## 6. No Information Leakage

- Oracle agent receives $(r, K, p, c, \sigma, \gamma)$ at construction; this is by definition.
- All learners see only $(s_0, p_{\min}, p_{\max} = $ action bounds$, \gamma, T)$ and the observed sequence $(s_\tau, h_\tau, r_\tau)$. They do not see $(r, K, p, c, \sigma)$.
- The reward function shape (linear-quadratic in harvest) is known to all learners, the same convention as the cobweb panel: the cost structure is an economic primitive, the demand-side parameters are estimated. For RLS and MBPO, $(p, c)$ are known by RLS and estimated by MBPO.

## 7. Seed and Reproducibility

- $N_{\text{seeds}} = 20$, matching the cobweb panel.
- Each seed sets `np.random.seed(seed)` at the top of the per-seed rollout.
- Environment uses `np.random.default_rng(seed)` for noise; paradigm-internal RNGs use `np.random.default_rng(seed + offset)` to keep paradigm randomness independent of env noise.
- All results reported as mean $\pm$ standard error across seeds.

## Hyperparameter Reference

| Item            | Value     | Source / rationale                                |
| --------------- | --------- | ------------------------------------------------- |
| $T$             | 500       | Same as cobweb panel                              |
| $\gamma$        | 0.95      | Standard                                          |
| $\sigma$        | 0.3       | Small relative to stock range                     |
| $r$             | 0.4       | Moderate growth                                   |
| $K$             | 10        | Carrying capacity                                 |
| $p$             | 2.0       | Price per unit harvest                            |
| $c$             | 0.2       | Quadratic harvest cost                            |
| $N_{\text{seeds}}$ | 20     | Matches cobweb                                    |
| Grid (DP)       | 60 stock × 31 action | Fine enough for visible curvature      |
| Grid (Q-learning) | 30 stock × 21 action | Coarser, as in cobweb's QL spec      |
| GA pop          | 30        | Arifovic 1994                                     |
| MBPO ensemble   | 1         | Linear-Gaussian, ensemble vacuous as in cobweb    |

## Sign-off

Audit written before implementation. Deviations are documented in the audit and in the §9.2 prose.
