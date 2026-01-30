# Econ-GYM: Scalable DP Problems Where Policy Search Beats Value Iteration

This note frames the Chapter 3 benchmarks using Ben Recht's "RL Minimalism vs Maximalism" taxonomy. The goal: identify economic problems that are solvable by DP in principle but intractable for value iteration, tractable for policy gradient methods, and verifiable against known theory. These problems form the core of Econ-GYM, a proposed benchmark suite for RL in economics.

---

## Part 1: The Insight -- Recht's RL Minimalism vs Maximalism

Ben Recht's "A Tour of Reinforcement Learning: The View from Continuous Control" (2019) draws a sharp distinction between two cultures in sequential decision-making:

**RL Minimalism.** The classical control and operations research view. Start with a model. Exploit structure (linearity, convexity, decomposability). Use the simplest method that works: certainty equivalence, LQR, base-stock policies, EMSR heuristics. If the problem admits a closed-form or tractable DP solution, use it. Don't introduce neural networks, replay buffers, or policy gradient estimators when a Riccati equation suffices.

**RL Maximalism.** The modern deep RL view. Assume minimal structure. Learn everything end-to-end from interaction data. Use flexible function approximators (neural networks), generic algorithms (PPO, SAC, DQN), and massive compute. The pitch: one algorithm to rule them all, from Atari to robotics to economics.

Recht's critique of RL Maximalism is pointed: on problems where classical methods apply (LQR, inventory with known demand distributions, single-leg revenue management), deep RL is slower, less sample-efficient, harder to debug, and often worse. The "Maximalist" approach conflates generality with quality. Many celebrated RL results are existence proofs on toy problems, not demonstrations of practical advantage.

### Recht's Taxonomy

Recht implicitly organizes problems along three axes:

| Axis | Minimalist End | Maximalist End |
|------|---------------|----------------|
| Action space | Few discrete actions | Continuous, high-dimensional |
| Model knowledge | Known dynamics, known rewards | Unknown, must be learned |
| Decision horizon | Single-step or myopic | Long sequential, delayed rewards |

Problems in the "Minimalist corner" (few actions, known model, short horizon) are solved by classical OR/control. Problems in the "Maximalist corner" (continuous actions, unknown dynamics, long horizon) arguably need deep RL. The interesting territory is in between: Recht's Cases 6-7, where the problem has enough structure that DP can be formulated, but the state space is large enough that value iteration on a grid is infeasible. Policy gradient methods can exploit the problem's smoothness without requiring full state-space enumeration.

### The Sweet Spot for Econ-GYM

The benchmark problems we want satisfy three conditions simultaneously:

1. **DP-formable:** The problem has a well-defined Bellman equation. An oracle with infinite compute could solve it exactly.
2. **VI-intractable:** The state space is too large or too continuous for tabular value iteration or grid-based DP. The curse of dimensionality bites.
3. **Policy-gradient-tractable:** The value function or policy is smooth enough that gradient-based optimization in policy space converges to a good solution.
4. **Theory-verifiable:** Known analytical results (closed-form solutions in special cases, bounds, asymptotic behavior) provide ground truth against which RL output can be checked.

This is the honest case for RL in economics. Not "RL solves everything" (Maximalism). Not "RL is never needed" (Minimalism). Rather: here are specific, economically meaningful problems where RL is the right tool, and here is how to verify it works.

---

## Part 2: Five Problem Classes

All five problems live in Recht's Cases 6-7: DP-formable, VI-intractable, policy-gradient-tractable, theory-verifiable.

### Problem 1: Optimal Stopping with Learning (Job Search / McCall)

**Economic setting.** A worker samples wage offers from an unknown distribution. Each period: accept (absorbing state) or reject and draw again. The worker pays a search cost $c$ per period and discounts future payoffs at rate $\gamma$. The twist: the wage distribution $F$ is unknown and must be learned from observed offers.

**DP formulation.**

$$V(b_t) = \max\left\{ \mathbb{E}_{w \sim b_t}[w], \; -c + \gamma \int V(b_{t+1}(b_t, w)) \, dF_{b_t}(w) \right\}$$

where $b_t$ is the agent's belief (posterior) over the wage distribution parameters.

**Why VI fails.** The state is the belief $b_t$, which lives in a continuous space (e.g., the space of Beta or Normal-Inverse-Gamma posteriors). Discretizing beliefs on a grid is feasible for 1-2 parameters but explodes for richer models (mixture distributions, non-parametric beliefs).

**Why policy gradient works.** The optimal policy is a threshold: accept if the expected wage exceeds a reservation wage $\bar{w}(b_t)$. Parameterize this threshold as a function of sufficient statistics of the belief. Policy gradient optimizes $\bar{w}(\cdot)$ directly without enumerating belief states.

**Theory verification.**
- Known result (McCall 1970): When $F$ is known, the reservation wage satisfies $\bar{w} = c + \gamma \int_{\bar{w}}^{\infty} (w - \bar{w}) \, dF(w)$.
- Check: In the limit of infinite data (belief converges to truth), the RL policy's threshold should converge to the McCall reservation wage.
- Check: The value of information (difference between Bayesian-optimal and myopic policies) should be positive and decreasing in sample size.

**Benchmark table.**

| Method | Utility (known F) | Utility (unknown F, T=100) | Theory checks | Wall-clock |
|--------|-------------------|---------------------------|---------------|------------|
| Oracle (known F, exact DP) | 1.000 | -- | all pass | -- |
| Grid DP on beliefs | -- | baseline | all pass | slow |
| PPO | -- | ? | ? | ? |
| SAC | -- | ? | ? | ? |
| CEM | -- | ? | ? | ? |

### Problem 2: Consumption-Savings with Stochastic Income

**Economic setting.** An infinitely-lived agent chooses consumption $c_t$ from wealth $w_t$, which evolves as $w_{t+1} = R(w_t - c_t) + y_{t+1}$, where $R$ is the gross interest rate and $y_{t+1}$ is stochastic income. The agent maximizes $\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t u(c_t)\right]$ with CRRA utility $u(c) = c^{1-\sigma}/(1-\sigma)$.

**DP formulation.**

$$V(w) = \max_{0 \leq c \leq w} \left\{ u(c) + \gamma \mathbb{E}[V(R(w-c) + y)] \right\}$$

**Why VI fails.** With one wealth dimension, grid DP works. Add heterogeneous income processes (regime-switching, persistent AR(1) shocks), multiple assets, housing, health shocks, or family composition, and the state space becomes 5-10 dimensional. Krusell-Smith (1998) style models with aggregate and idiosyncratic shocks are the canonical example of curse-of-dimensionality in macro.

**Why policy gradient works.** The consumption function $c(w, \theta)$ is smooth and monotonically increasing. Parameterize it with a neural network or polynomial and optimize expected lifetime utility via policy gradient. The smoothness of CRRA utility provides well-behaved gradients.

**Theory verification.**
- Euler equation: $u'(c_t) = \gamma R \, \mathbb{E}[u'(c_{t+1})]$ must hold at the RL policy.
- Carroll (2006) buffer-stock target: wealth-to-income ratio converges to a known target.
- Boundary behavior: consumption approaches income as wealth approaches infinity (no precautionary motive at high wealth).

**Benchmark table.**

| Method | Euler eq. residual | Buffer-stock target | Consumption fn. MSE | Wall-clock |
|--------|-------------------|--------------------|--------------------|------------|
| Endogenous grid (1D) | 0.000 | exact | baseline | fast |
| Grid DP (5D) | -- | -- | -- | infeasible |
| PPO | ? | ? | ? | ? |
| Neural network policy | ? | ? | ? | ? |

### Problem 3: Dynamic Pricing with Unknown Demand

**Economic setting.** A firm sells a product over $T$ periods with initial inventory $I_0$. Each period, the firm sets price $p_t$; demand is $d_t = D(p_t, \theta) + \varepsilon_t$ where $\theta$ is an unknown demand parameter. The firm learns $\theta$ from sales data while simultaneously optimizing revenue. This is the exploration-exploitation tradeoff in a pricing context.

**DP formulation.**

$$V(I_t, b_t, t) = \max_{p_t \geq 0} \left\{ p_t \cdot \mathbb{E}[\min(d_t, I_t) \mid b_t] + \gamma \, \mathbb{E}[V(I_{t+1}, b_{t+1}, t+1) \mid b_t] \right\}$$

where $b_t$ is the posterior belief over $\theta$ and $I_{t+1} = I_t - \min(d_t, I_t)$.

**Why VI fails.** State is $(I_t, b_t, t)$. Inventory is discrete (manageable), but the belief $b_t$ is continuous. With multi-dimensional demand parameters (intercept, slope, cross-elasticities), belief space explodes.

**Why policy gradient works.** The pricing policy $p_t(I_t, b_t)$ is smooth in both arguments. Thompson Sampling provides a natural exploration mechanism compatible with policy gradient (sample $\hat{\theta} \sim b_t$, optimize myopically). Policy gradient can learn when to explore (early, high inventory) vs. exploit (late, low inventory).

**Theory verification.**
- Myopic optimality at $T=1$: price equals known monopoly price.
- Gallego-van Ryzin (1994) upper bound: revenue should approach the deterministic optimum as $T \to \infty$.
- Regret scaling: Bayesian regret should be $O(\sqrt{T})$ for well-designed policies.

### Problem 4: Multi-Agent Coordination (Decentralized Inventory)

**Economic setting.** $N$ retailers each manage local inventory, sharing a common supplier with limited capacity. Each retailer observes only local demand and inventory. Centralized optimization would solve a joint MDP over all retailers; instead, each retailer runs a local policy that must implicitly coordinate through shared supply constraints.

**DP formulation (centralized).**

$$V(\mathbf{I}_t) = \max_{\mathbf{a}_t} \left\{ \sum_{i=1}^N r_i(I_{i,t}, a_{i,t}) + \gamma \, \mathbb{E}[V(\mathbf{I}_{t+1})] \right\} \quad \text{s.t.} \quad \sum_i a_{i,t} \leq K$$

where $\mathbf{I}_t = (I_{1,t}, \ldots, I_{N,t})$ is the joint inventory state.

**Why VI fails.** Joint state space is $|\mathcal{I}|^N$, exponential in the number of retailers. Even $N=10$ retailers with 100 inventory levels each yields $100^{10} = 10^{20}$ states.

**Why policy gradient works.** Each retailer's policy $\pi_i(a_i | I_i)$ depends only on local state. Independent PPO with shared reward signals (CTDE: centralized training, decentralized execution) scales linearly in $N$. Mean-field approximations further reduce complexity.

**Theory verification.**
- $N=2$ analytical solution exists for symmetric retailers; RL should match it.
- Base-stock policies are optimal for uncapacitated systems; RL should recover them when $K \to \infty$.
- Price of anarchy: ratio of decentralized to centralized welfare should match game-theoretic predictions.

### Problem 5: Macroeconomic Policy (Simplified DSGE)

**Economic setting.** A central bank chooses interest rate $i_t$ to minimize a loss function over output gap and inflation deviations, subject to a New Keynesian Phillips Curve and IS curve. The state includes current output gap, inflation, and potentially lagged variables or belief states about structural parameters.

**DP formulation.**

$$V(x_t) = \min_{i_t} \left\{ L(\pi_t, y_t) + \gamma \, \mathbb{E}[V(x_{t+1})] \right\}$$

where $x_t = (\pi_t, y_t, \ldots)$ and the transition follows the NK model equations.

**Why VI fails.** The basic 2-equation NK model is tractable. Add heterogeneous agents (HANK), financial frictions, learning about structural parameters, zero lower bound constraints, or forward guidance, and the state space becomes high-dimensional. Fernandez-Villaverde et al. (2020) document this as a key computational barrier.

**Why policy gradient works.** The Taylor Rule $i_t = \phi_\pi \pi_t + \phi_y y_t$ is already a parameterized policy. Policy gradient generalizes this to nonlinear rules $i_t = f(x_t; \theta)$ that can handle ZLB constraints, state-dependent responses, and uncertainty about model parameters. The loss function is smooth and differentiable in policy parameters.

**Theory verification.**
- Linear-quadratic case: RL should recover the optimal Taylor Rule coefficients.
- Certainty equivalence: with known parameters and quadratic loss, RL policy should match the certainty-equivalent solution.
- ZLB episodes: RL should learn asymmetric responses (aggressive easing, gradual tightening) consistent with optimal policy under occasionally binding constraints.

---

## Part 3: Econ-GYM API Design

The benchmark suite follows the Gymnasium (OpenAI Gym) interface standard, enabling direct use with stable_baselines3 and other RL libraries.

### Interface Sketch

```python
import econ_gym
from stable_baselines3 import PPO

# Create environment
env = econ_gym.make('JobSearch-v0', wage_dist='lognormal', unknown_params=True)

# Standard Gym loop
obs, info = env.reset(seed=42)
for t in range(1000):
    action = env.action_space.sample()  # or rl_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

# Train with stable_baselines3
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)

# Theory verification
from econ_gym.verification import check_theory
results = check_theory(env, model, checks=[
    'reservation_wage_convergence',
    'value_of_information_positive',
    'known_F_limit'
])
print(results)
# {'reservation_wage_convergence': PASS (0.98 vs 0.97 analytical),
#  'value_of_information_positive': PASS (delta_V = 0.034),
#  'known_F_limit': PASS (threshold within 1.2% of McCall)}
```

### Proposed Environments

| Environment ID | Problem | State dim | Action | Theory checks |
|----------------|---------|-----------|--------|---------------|
| `JobSearch-v0` | Optimal stopping with learning | 2-5 (belief params) | Binary (accept/reject) | Reservation wage, VOI |
| `ConsumptionSavings-v0` | Buffer-stock consumption | 1-10 (wealth + shocks) | Continuous (consumption) | Euler equation, buffer target |
| `DynamicPricing-v0` | Revenue management with learning | 3+ (inventory, belief, time) | Continuous (price) | Gallego-vR bound, regret |
| `DecentralizedInventory-v0` | Multi-agent supply chain | $N \times$ local state | Discrete (order qty) | Base-stock limit, POA |
| `MonetaryPolicy-v0` | NK model with frictions | 2-20 (macro vars + beliefs) | Continuous (interest rate) | Taylor rule, CE solution |

### Evaluation Protocol

```python
from econ_gym.evaluation import benchmark_suite

results = benchmark_suite(
    env_id='ConsumptionSavings-v0',
    methods={
        'oracle': econ_gym.baselines.EndogenousGrid,
        'vi_grid': econ_gym.baselines.ValueIterationGrid,
        'ppo': PPO("MlpPolicy", env),
        'sac': SAC("MlpPolicy", env),
        'cem': econ_gym.baselines.CrossEntropyMethod,
        'nelder_mead': econ_gym.baselines.NelderMead,
    },
    seeds=range(50),
    theory_checks=True,
)
results.to_latex('ch03_benchmarks/sims/consumption_savings_benchmark.tex')
results.plot_learning_curves('ch03_benchmarks/sims/consumption_savings_curves.png')
```

---

## Part 4: Benchmark Template

Following Recht's style: compare methods honestly, include classical baselines, report wall-clock time, and verify against theory.

### Template Table

| Method | Category | Utility (mean +/- SE) | Theory checks passed | Wall-clock (s) | Notes |
|--------|----------|----------------------|---------------------|----------------|-------|
| Oracle (exact DP) | Exact | 1.000 +/- 0.000 | 5/5 | -- | Available only for small instances |
| DP + belief grid | Approximate DP | ? | ?/5 | ? | Curse of dimensionality beyond 3D |
| Value Iteration (tabular) | DP | ? | ?/5 | ? | Infeasible beyond small state spaces |
| PPO | Policy gradient | ? | ?/5 | ? | Generic, no problem structure |
| SAC | Policy gradient | ? | ?/5 | ? | Continuous actions, entropy regularization |
| CEM | Derivative-free | ? | ?/5 | ? | Simple, embarrassingly parallel |
| Nelder-Mead | Derivative-free | ? | ?/5 | ? | No gradients needed |

### What the Table Reveals

The honest outcome will likely show:

- **Oracle** is best but doesn't scale.
- **Grid DP** works for low dimensions, fails for high dimensions.
- **PPO/SAC** scale to high dimensions but are noisier and slower to converge.
- **CEM/Nelder-Mead** are competitive for low-dimensional policy parameterizations.
- **Theory checks** are the key differentiator: methods that pass all checks are trustworthy; methods that fail checks despite high utility are likely overfitting or exploiting simulator artifacts.

This is Recht's point: don't just report reward. Report whether the solution makes economic sense. Theory verification is the contribution.

---

## Part 5: Existing OR Simulators (Reference Catalog)

The following catalogs existing simulator platforms relevant to OR and economics benchmarking. This section is preserved from earlier notes as reference material for the Econ-GYM design.

### OR-Gym: The Standard for OR+RL

**What it is**: Open-source library bridging operations research and reinforcement learning. Created by Carnegie Mellon and Dow Chemical researchers.

**GitHub**: https://github.com/hubbs5/or-gym
**Paper**: Hubbs et al. (2020) - "A Reinforcement Learning Library for Operations Research Problems"

Available Environments:

| Problem Type | Environment Name | Description | Benchmark Results |
|--------------|-----------------|-------------|-------------------|
| Knapsack | `Knapsack-v0`, `Knapsack-v1`, `Knapsack-v2` | Binary, bounded, stochastic knapsack | PPO achieves 95-98% of optimal |
| Bin Packing | `BinPacking-v0` to `BinPacking-v5` | 1D, 2D, 3D bin packing with online arrivals | RL matches heuristics, 5-10% below optimal |
| Multi-Echelon Inventory | `InvManagement-v0`, `InvManagement-v1` | 2-4 echelon supply chains, stochastic demand | RL outperforms base-stock by 8-12% |
| Vehicle Routing | `VehicleRouting-v0` | Capacitated VRP with stochastic demands | RL within 3-5% of LKH3 |
| TSP | `TSP-v0`, `TSP-v1` | Traveling salesman with time windows | Attention-based RL competitive with heuristics |
| Portfolio Optimization | `Portfolio-v0` | Multi-period asset allocation | RL adapts to non-stationary returns |
| Newsvendor | `Newsvendor-v0` | Classic single-period inventory problem | RL learns critical fractile implicitly |

Key Features:
- OpenAI Gym API: Standard `reset()`, `step()`, `render()` interface
- Benchmarks included: MIP solvers (Gurobi), heuristics, RL baselines (PPO, A2C, DQN)
- Customizable: Config dictionaries to adjust problem parameters

```python
import or_gym
from stable_baselines3 import PPO

env = or_gym.make('InvManagement-v1')
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### RL4CO: State-of-the-Art for Combinatorial Optimization

**What it is**: Unified library for neural combinatorial optimization with 20+ problem types and 15+ RL algorithms. Built on PyTorch Lightning for GPU acceleration.

**GitHub**: https://github.com/ai4co/rl4co
**Paper**: Berto et al. (2024) - "RL4CO: An Extensive Reinforcement Learning for Combinatorial Optimization Benchmark"

Supported Problems:

| Problem | Variants | Best Method (from benchmarks) | Gap to Optimal |
|---------|----------|-------------------------------|----------------|
| TSP | Standard, euclidean, asymmetric | POMO + Augmentation | 0.03-0.08% |
| CVRP | Capacitated vehicle routing | Attention Model | 0.5-1.2% |
| VRPTW | VRP with time windows | Sym-NCO | 1.5-2.5% |
| PDPTW | Pickup-delivery with time windows | POMO | 2-4% |
| OP | Orienteering problem | Attention Model | 0.8-1.5% |

Key Algorithms: Attention Model (Kool et al., 2019), POMO (Kwon et al., 2020), Sym-NCO (Kim et al., 2022), REINFORCE, A2C, PPO, MatNet (Kwon et al., 2021).

```python
from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModel
from rl4co.utils.trainer import RL4COTrainer

env = CVRPEnv(num_loc=50, min_demand=1, max_demand=10)
model = AttentionModel(env, embedding_dim=128, num_encoder_layers=3)
trainer = RL4COTrainer(max_epochs=100, accelerator="gpu")
trainer.fit(model)
```

### PyVRP: Production-Grade Vehicle Routing Solver

**What it is**: State-of-the-art hybrid genetic search VRP solver. Winner of DIMACS 2021 and EURO-NeurIPS 2022 competitions.

**GitHub**: https://github.com/PyVRP/PyVRP
**Paper**: Wouda et al. (2024) - "PyVRP: A High-Performance VRP Solver Package"

PyVRP is a classical metaheuristic, not an RL solver. It matters for RL researchers because: (1) RL methods must beat PyVRP to claim improvement, (2) PyVRP solutions can warm-start RL policy training, (3) hybrid RL-HGS architectures can learn heuristics that guide PyVRP's search.

Supported variants: CVRP, VRPTW, simultaneous pickup/delivery, heterogeneous fleet, multi-depot, site-dependent.

Performance (VRPTW, 100 customers): 0.1-0.5% gap to best-known solutions in 10-60 seconds on CPU.

### SimPy: Discrete-Event Simulation for Custom Supply Chains

**What it is**: General-purpose discrete-event simulation framework for modeling complex supply chain dynamics.

**Website**: https://simpy.readthedocs.io

Best for: multi-echelon inventory with complex lead times, manufacturing with machine breakdowns, warehouse operations, any system where events trigger other events.

SimPy environments can be wrapped as Gymnasium environments for RL training.

### Salina: Combinatorial Optimization with Multi-Agent RL

**What it is**: Flexible job shop scheduling environment with multi-agent RL support.

**GitHub**: https://github.com/facebookresearch/salina

Features: graph-based state representation (operations as nodes), multi-agent coordination (machines as agents), benchmarks on standard FJSSP instances (Brandimarte, Hurink).

### Commercial Simulation Platforms

| Platform | Use Case | RL Integration | Cost |
|----------|----------|---------------|------|
| AnyLogic | Enterprise supply chain, logistics | Java API allows Gym wrapper | $3,000-$10,000/year |
| Arena Simulation | Manufacturing, healthcare, call centers | VBA macros + Python bridge | Enterprise pricing |
| FlexSim | Warehouse, production lines, ports | C++ API | ~$5,000/year |

### Domain-Specific Simulators

- **CityLearn** (energy): Building energy management, grid optimization. Native Gym API. GitHub: https://github.com/intelligent-environments-lab/CityLearn
- **SUMO + Flow** (traffic): Traffic signal control, autonomous vehicle routing. RLlib + SUMO. GitHub: https://github.com/flow-project/flow
- **ns-3 Gym** (telecom): Network routing, resource allocation. Gym wrapper for ns-3. GitHub: https://github.com/tkn-tub/ns3-gym

### Simulator Comparison Matrix

| Simulator | Best For | Ease of Use | Performance | Flexibility | Documentation |
|-----------|----------|-------------|-------------|-------------|---------------|
| OR-Gym | Classical OR problems | 5/5 | 3/5 | 4/5 | 5/5 |
| RL4CO | Routing/combinatorial | 4/5 | 5/5 | 5/5 | 4/5 |
| PyVRP | Baseline/warm-start | 5/5 | 5/5 | 3/5 | 5/5 |
| SimPy | Custom supply chains | 3/5 | 3/5 | 5/5 | 4/5 |
| AnyLogic | Enterprise deployment | 3/5 | 4/5 | 5/5 | 4/5 |

### Practical Workflow

1. Problem Identification: Inventory -> OR-Gym; Routing -> RL4CO + PyVRP baseline; Scheduling -> OR-Gym or SimPy; TSP -> RL4CO.
2. Baseline Establishment: Always benchmark against classical methods first (MIP solvers, known heuristics).
3. RL Training: Use stable_baselines3 with eval callbacks.
4. Deployment Validation: Test on held-out instances with varied parameters.

---

## Part 6: Why This Frames the Problem Correctly

### Recht's Criticism of RL Maximalism

The standard pitch for RL in economics goes something like: "Economic problems are sequential decision problems. RL solves sequential decision problems. Therefore RL should be applied to economics." This is RL Maximalism, and Recht's critique applies in full force.

The problems with the Maximalist pitch:
- It ignores that economics has 70+ years of DP/structural estimation methods that work well for many problems.
- It conflates "can be formulated as an MDP" with "needs RL to be solved."
- It produces papers that solve toy versions of problems that economists already solve with existing tools, adding computational overhead without intellectual content.
- It doesn't explain when RL is actually needed, which is the only interesting question.

### The Honest Pitch

Econ-GYM makes an honest, narrow claim: there exist economically important problems where (1) the DP formulation is known, (2) value iteration is computationally infeasible, (3) policy gradient methods offer a tractable alternative, and (4) the solution can be verified against economic theory. These problems are not artificial. They arise whenever economists add realistic features (heterogeneity, learning, strategic interaction, continuous state spaces) to canonical models.

The contribution is not "RL beats everything." The contribution is:
- Identifying exactly which economic problems benefit from RL (and which don't).
- Providing reproducible benchmarks with theory-based verification.
- Reporting honest comparisons including classical methods, derivative-free optimization, and simple heuristics alongside deep RL.

### Flipping from Hype to Science

The existing literature on RL in economics suffers from a hype problem. Papers either oversell RL's capabilities or dismiss it as unnecessary. Econ-GYM takes a different approach: treat RL as a computational tool with specific strengths and weaknesses, benchmark it honestly against alternatives, and let the results speak.

This is Recht's core message applied to economics: the interesting question is not whether RL "works" (it always "works" on some metric) but whether it outperforms the methods economists already use, on problems economists actually care about, in ways that are verifiable against known theory. If the answer is yes for specific problem classes, that is a genuine scientific contribution. If the answer is no, that is equally valuable information.

The five problem classes in Part 2 are chosen because existing evidence suggests RL has genuine advantages on them. The benchmark template in Part 4 is designed to make the comparison fair. The theory verification checks ensure that any claimed advantage is not an artifact of simulator exploitation or reward hacking. This is what computational science looks like when done honestly.
