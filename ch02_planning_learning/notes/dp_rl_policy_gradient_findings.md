# DP, RL, and Policy Gradient: Unified Findings

Notes from two research passes through the `ch02_planning_learning/papers/` collection, January 2026.

## Pass 1: DP + RL Unified Results (Bertsekas Papers)

Seven major results showing DP and RL as unified rather than separate paradigms.

### 1. The Big Unification: Newton's Method

**Paper:** `bertsekas2024_mpc_rl_unified_dp.md`

Central framework: DP, RL, and MPC are all unified through Newton's method applied to Bellman's equation. The architecture has two synergistic components:

- Off-line training (learn a value function approximation)
- On-line play (policy improvement via lookahead)

Key result: the performance error `||J_mu - J*||` is superlinearly related to the approximation error `||J_tilde - J*||`. Longer lookahead expands the region of convergence, explaining why tree search compensates for imperfect value functions. The paper notes that "the synergy between off-line training and on-line play also underlies MPC, and indeed the MPC design architecture is very similar to the one of AlphaZero and TD-Gammon."

### 2. Rollout as the Bridge Between DP and RL

**Papers:** `bertsekas2020_rollout_policy_iteration_distributed_rl.md`, `bertsekas2020_multiagent_rollout_sinica.md`

The fundamental policy improvement property: for any base policy mu, the rollout policy satisfies `J_tilde_mu(x) <= J_mu(x)` for all states x. This holds whether the cost function is exact or approximate, deterministic or stochastic. Rollout is a policy improvement operator that works identically in DP (with a model) and RL (without one).

### 3. Approximate PI Error Bounds

**Paper:** `bertsekas2011_approximate_policy_iteration_survey.md`

Proposition 6: when policy evaluation approximates the true cost within error delta, the resulting policy satisfies `||J_mu_k - J*|| <= 2*alpha/(1-alpha) * delta`. This guarantees convergence of approximate DP/RL methods with bounded errors.

Also establishes that TD(0), TD(lambda), and LSPE are all projected value iteration (Proposition 3), meaning classical VI projected onto a feature subspace. The DP and RL versions are the same algorithm.

### 4. Lambda-PI: Interpolating Between VI and PI

**Paper:** `bertsekas2011_lambda_policy_iteration.md`

Lambda-PI is a continuous interpolation between value iteration (lambda=0) and policy iteration (lambda=1). Introduces geometric sampling (multiple short trajectories vs. one long trajectory), offering practical advantages for approximate implementations.

### 5. Newton Step Interpretation

**Paper:** `bertsekas2022_newton_method_rl_mpc.md`

The one-step lookahead with an approximate value function is literally a Newton step on Bellman's equation. This explains the stability region: better approximations and longer lookahead both expand it, and they interact synergistically.

### 6. AlphaZero as DP+RL

**Paper:** `bertsekas2021_lessons_from_alphazero.md`

AlphaZero = off-line neural net training (RL) + on-line tree search (DP planning). The principal contributor to success is long lookahead with tree pruning, which enlarges Newton's region of convergence even when the learned value function is imperfect.

### 7. Q-Learning = Asynchronous VI

**Paper:** `bertsekas2025_course_reinforcement_learning_2ed.md`

Q-learning is asynchronous value iteration on the (state, action) Bellman equation. The RL and DP literatures describe the same algorithm with different terminology.

**Through-line:** DP and RL are not separate paradigms but endpoints of a spectrum, connected by Newton's method, rollout, and approximation in value space.

---

## Pass 2: Policy Gradient Connections to DP

Five major categories of connection between policy gradient methods and DP.

### 1. Policy Gradient = Approximate Policy Iteration (in the limit)

**Paper:** `xiao2022_convergence_rates_policy_gradient.md` -- The cleanest result. Policy Mirror Descent (PMD) with step sizes eta_k -> infinity converges to classical Policy Iteration. With finite steps, PMD is inexact PI. Geometrically increasing step sizes (eta_k = eta_0 / gamma^k) yield linear convergence.

**Paper:** `agarwal2021_policy_gradient_theory.md` -- Lemma 5.1: NPG with softmax parametrization is literally a soft policy iteration step. Lemma 4.1: the gradient dominance property guarantees global convergence despite non-convexity. Theorem 16: NPG converges at O(1/k), dimension-free (no dependence on |S| or |A|).

### 2. Actor-Critic = Two-Timescale Policy Iteration

**Paper:** `wu2020_two_timescale_actor_critic.md` -- Critic updates fast (TD learning = policy evaluation), actor updates slow (gradient ascent = policy improvement). This is PI with embedded approximate value function updates. Finite-time bound: O(epsilon^{-2.5}) samples.

**Paper:** `tian2023_actor_critic_multilayer_nn.md` -- Extends to deep nets. TD learning interpreted as approximate gradient descent. Convergence rate O(T^{-0.5}) with function approximation.

### 3. NPG as Newton's Method / Mirror Descent

**Paper:** `muller2024_fisher_rao_npg.md` -- NPG viewed as mirror descent with Fisher-Rao metric (Bregman divergence). The Fisher information matrix provides adaptive preconditioning, connecting to Newton's method in policy space. Achieves geometric (linear) convergence.

**Paper:** `cen2022_fast_convergence_npg.md` -- NPG with entropy regularization = soft policy iteration. The soft Bellman operator is a gamma-contraction (just like the standard one). Linear convergence rate gamma^k.

### 4. Regularization Bridges PG and PI

**Paper:** `geist2019_regularized_mdp.md` -- Adding entropy or KL regularization to the Bellman operator yields "soft" PI. The regularized operators are contractions (Definition 1), and error propagation bounds match classical approximate DP.

**Paper:** `vieillard2020_kl_regularization_rl.md` -- Mirror Descent MPI (MD-MPI): KL-regularized policy iteration. Soft Actor-Critic is an instance. Regularization improves error propagation bounds compared to unregularized PI.

**Paper:** `grill2020_mcts_regularized_policy.md` -- AlphaZero's MCTS tracks a regularized policy optimization solution. Without regularization (R=0), it reduces to classical PI. With a single gradient step, it becomes regularized policy gradient.

### 5. Error Propagation in Approximate PI/PG

**Papers:** `munos2003_error_bounds_api.md`, `farahmand2010_error_propagation_api_avi.md` -- Bound performance loss `V* - V_pi` using concentrability coefficients. These bounds apply equally to policy gradient methods (which are approximate PI with specific approximation architectures).

**Paper:** `scherrer2015_approx_modified_policy_iteration_tetris.md` -- Unified error analysis for VI, PI, and modified PI (m Bellman applications). PG methods fit into this AMPI framework.

### The VI-to-PI Spectrum (Step Size Controls Position)

| Step size (eta_k) | Method |
|---|---|
| eta -> 0 | Vanilla policy gradient (small steps) |
| eta finite | Policy mirror descent / NPG |
| eta -> infinity | Exact policy iteration |
| + entropy regularization | Soft PI / SAC |
| + KL regularization | Mirror descent MPI |
| + Fisher metric | Natural policy gradient (Newton-like) |

**Unifying insight:** Policy gradient methods are continuous-time, approximate, regularized versions of policy iteration. The step size controls where you sit on the VI-to-PI spectrum, regularization controls the softness of the greedy operator, and the Fisher metric provides second-order (Newton-like) acceleration.

---

## Cross-Cutting Themes

1. **Newton's method appears twice**: Bertsekas shows one-step lookahead is a Newton step on Bellman's equation (Pass 1), and NPG is Newton's method in policy space via the Fisher information matrix (Pass 2). These are dual perspectives on the same acceleration principle.

2. **Approximate PI is the universal framework**: TD methods are projected VI (Pass 1, Bertsekas 2011). Actor-critic is two-timescale PI (Pass 2, Wu 2020). Policy gradient with large steps converges to PI (Pass 2, Xiao 2022). Nearly everything reduces to approximate policy iteration with different approximation strategies.

3. **Regularization unifies everything**: Lambda-PI interpolates between VI and PI via lambda (Pass 1). Entropy/KL regularization yields soft PI which connects to SAC and AlphaZero's MCTS (Pass 2). The greedy operator becomes "soft," and all classical contraction/error bounds carry over.

4. **The error bound `2*alpha/(1-alpha)*delta` is universal**: Whether you use approximate PI (Bertsekas 2011), policy gradient (Munos 2003, Farahmand 2010), or modified PI (Scherrer 2015), the same concentrability-based error propagation structure governs performance.

5. **AlphaZero is the concrete exemplar**: It appears in both passes as the canonical instance of DP+RL unification (off-line RL training + on-line DP tree search) and as an instance of regularized policy optimization (MCTS as regularized PI).
