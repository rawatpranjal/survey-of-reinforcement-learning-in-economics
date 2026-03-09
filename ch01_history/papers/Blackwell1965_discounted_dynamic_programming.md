# Blackwell (1965) — Discounted Dynamic Programming

## Bibliographic Information
- **Author:** David Blackwell
- **Title:** Discounted Dynamic Programming
- **Journal:** The Annals of Mathematical Statistics
- **Volume:** 36, No. 1
- **Pages:** 226-235
- **Year:** February 1965
- **Support:** National Science Foundation, Grant GP-2593

## Problem Addressed

Blackwell provides the first rigorous measure-theoretic foundation for discounted infinite-horizon dynamic programming with general (possibly uncountable) state and action spaces. While Bellman (1957) developed the conceptual framework and Howard (1960) gave complete analysis for finite state/action spaces, neither addressed the technical difficulties arising with continuous spaces: Does an optimal policy exist? Is the optimal value function the unique solution to the Bellman equation? Can non-stationary policies outperform stationary ones?

Blackwell resolves these questions, showing that under discounting with bounded rewards, the Bellman operator is a contraction, stationary policies are essentially optimal, and the optimal value function is characterized by the optimality equation.

## Key Definitions and Notation

**Dynamic Programming Problem:** Specified by four objects (S, A, q, r) where:
- S, A are non-empty Borel sets (state space, action space)
- q(·|s, a) ∈ Q(S|SA) is a transition kernel: probability on S given state-action pair
- r ∈ M(SAS) is a bounded Baire reward function r(s, a, s')
- β ∈ [0, 1) is the discount factor

**Plan:** A sequence π = (π₁, π₂, ...) where πₙ ∈ Q(A|Hₙ) specifies the (possibly randomized) action choice given history Hₙ = SA...S (2n-1 factors).

**Markov Plan:** π = (f₁, f₂, ...) where each fₙ: S → A is a Baire function (action depends only on current state, not history).

**Stationary Plan:** f^(∞) where fₙ = f for all n (same decision rule at every period).

**Total Discounted Reward:** For plan π starting in state s:
$$I(\pi)(s) = \mathbb{E}\left[\sum_{n=1}^{\infty} \beta^{n-1} r(\sigma_n, \alpha_n, \sigma_{n+1})\right]$$

**Operator T_f:** For stationary policy f, the operator T: M(S) → M(S) defined by:
$$Tu = fq(r + \beta u)$$
where u on the right depends only on the next state. Tu is the expected income if we use f for one period, then receive terminal value u.

**(p, ε)-optimal:** Plan π* is (p, ε)-optimal if for every π: p{I(π) > I(π*) + ε} = 0

**ε-optimal:** Plan π* is ε-optimal if I(π) ≤ I(π*) + ε for all π, s.

**Optimal:** Plan π* is optimal if I(π*) ≥ I(π) for all π, s.

## Main Results

### Contraction Property (Theorem 5)
Any operator U satisfying monotonicity (u ≤ v implies Uu ≤ Uv) and the discount shift property (U(u + c) = Uu + βc) is a contraction with modulus β:
$$\|Uu - Uv\| \leq \beta\|u - v\|$$

By Banach's fixed-point theorem, U has a unique fixed point u*, and U^n u → u* geometrically.

### Existence Results
1. **(p, ε)-optimal plans always exist** (Theorem 1): For any probability p on S and any ε > 0, there exists a (p, ε)-optimal plan.

2. **(p, ε)-optimal stationary plans always exist** (Theorem 6(b)): For any p ∈ P(S), ε > 0, there is a (p, ε)-optimal stationary plan.

3. **ε-optimal plans may not exist** (Example 2): When S = A = [0,1] with a non-Borel reward structure, no plan can be ε-optimal for small ε.

### Optimality of Stationary Plans

**Theorem 6(a):** For any Markov π = (f₁, f₂, ...), let U = supₙ Tₙ where Tₙ is associated with fₙ. The fixed point u* of U is the optimal return among π-generated plans. Any f with Tu* ≥ u* - ε(1-β) satisfies I(f^(∞)) ≥ u* - ε.

**Theorem 6(c):** If there exists an ε-optimal π*, then there exists an ε/(1-β)-optimal stationary plan.

**Theorem 6(f):** A plan π is optimal if and only if I(π) satisfies the optimality equation.

### The Optimality Equation

**Theorem 6(d):** Any u with T_a u ≤ u for all a ∈ A is an upper bound on incomes: I(π) ≤ u for all π.

**Theorem 6(e):** If for every ε > 0 there exists an ε-optimal plan, then the optimal return u* is a Baire function satisfying the optimality equation:
$$u^* = \sup_a T_a u^*$$

### Countable and Finite Action Spaces

**Theorem 7(a):** If A is countable (or essentially countable), the fixed point u* of U = sup_a T_a is the optimal return, u* is the unique bounded solution of the optimality equation, and for every ε > 0 there exists an ε-optimal stationary plan.

**Theorem 7(b):** If A is finite (or essentially finite), there exists an optimal stationary plan.

### Policy Improvement (Theorem 8)

**(a) Howard Improvement:** If I(g, π) ≥ I(π), then I(g^(∞)) ≥ I(g, π) ≥ I(π).

**(b) Eaton-Zadeh Improvement:** For any f, g: S → A, define h = f on {I(f^(∞)) ≥ I(g^(∞))}, h = g otherwise. Then I(h^(∞)) ≥ max(I(f^(∞)), I(g^(∞))).

## Counterexamples

**Example 1 (No p-optimal plans):** S = {0}, A = {1, 2, 3, ...}, r(0, a, 0) = (a-1)/a. No plan achieves I(π) = 1/(1-β), but sup_π I(π) = 1/(1-β).

**Example 2 (No ε-optimal plans):** S = A = [0,1] with a non-Borel reward indicator. For any π, there exists s₀ where I(π)(s₀) ≤ β/(1-β), but sup_π I(π)(s₀) = 1/(1-β).

**Example 3 (Non-Markov beating stationary):** A plan π that cannot be ε-dominated uniformly by any Markov plan, showing that stationary policies are optimal only in the (p, ε) sense, not uniformly.

## Implications for Reinforcement Learning

1. **Contraction Mapping Framework:** The proof that the Bellman operator is a contraction with modulus β provides the theoretical foundation for value iteration convergence. This is the basis for Q-learning's convergence proofs.

2. **Sufficiency of Stationary Policies:** The result that stationary policies are (p, ε)-optimal justifies focusing on deterministic, memoryless policies in RL. Agents need not maintain complex history-dependent strategies.

3. **Uniqueness of Value Function:** The unique fixed point characterization means there is a well-defined "optimal value function" that all convergent algorithms approach.

4. **Finite vs. Infinite Action Spaces:** The distinction between finite A (optimal policy exists) and countable A (only ε-optimal policies exist) is foundational for understanding when exact solutions are possible.

5. **Policy Improvement Theorems:** Theorem 8 provides the mathematical justification for Howard's policy iteration and extends it to the general measurable setting.

6. **Measurability Subtleties:** The counterexamples warn that in continuous spaces, technical conditions matter. RL algorithms on continuous spaces must handle measurability carefully.

## Notable Quotes

On the problem of undiscounted rewards:
> "This total expected reward may well be infinite, for example, if r ≡ 1. Or it may well be undefined. For example, if S has two elements 0, 1, A has only a single element, q is deterministic with 0 → 1, 1 → 0, and the transition 0 → 1 yields $1, while 1 → 0 costs $1, the series of rewards, starting in state 0, is 1 - 1 + 1 - 1 + ..."

On the relationship to prior work:
> "The first development of a general theory underlying these methods is due to Karlin [6], and a rather complete analysis of the finite case was given by Howard [5]."

On proving optimality:
> "(d) is extremely useful in proving optimality; if u is known to be the return from a policy π and u satisfies T_a u ≤ u for all a, (d) implies that π is optimal."

## Historical Significance

This paper, together with Blackwell's 1962 paper "Discrete Dynamic Programming" (which treated the undiscounted case), established the mathematical foundations of Markov Decision Processes as a branch of mathematics. The framework became standard in operations research, economics, and later reinforcement learning. The paper is notable for its economy: in 10 pages, Blackwell establishes essentially complete results for the discounted infinite-horizon case with general state and action spaces.
