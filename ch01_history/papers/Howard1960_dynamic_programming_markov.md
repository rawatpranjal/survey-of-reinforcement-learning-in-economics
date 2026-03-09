# Howard (1960) — Dynamic Programming and Markov Processes

## Bibliographic Information
- **Author:** Ronald A. Howard
- **Title:** Dynamic Programming and Markov Processes
- **Publisher:** The Technology Press of MIT and John Wiley & Sons
- **Year:** 1960
- **Context:** Outgrowth of an Sc.D. thesis submitted to the Department of Electrical Engineering, MIT, June 1958

## Problem Addressed

Howard tackles the computational challenge of finding optimal policies for sequential decision processes modeled as Markov Decision Processes (MDPs). While Bellman (1957) provided the theoretical framework via dynamic programming, direct application through value iteration requires iterating over potentially infinite time horizons. Howard develops the policy iteration algorithm, which finds optimal policies for infinite-horizon problems by solving finite systems of linear equations.

The motivating problem: given a system that can occupy states 1, 2, ..., N, with transition probabilities and rewards that depend on the chosen "alternative" (action) in each state, find the policy (mapping from states to alternatives) that maximizes long-run average reward.

## Key Definitions and Notation

**Markov Process:** A system occupying states i = 1, 2, ..., N with transition probabilities p_ij satisfying:
- 0 ≤ p_ij ≤ 1
- Σ_j p_ij = 1

**Markov Process with Rewards:** Each transition from state i to state j yields a reward r_ij. The expected immediate reward in state i is:

q_i = Σ_j p_ij r_ij

**Sequential Decision Process:** In each state i, the decision-maker chooses among K_i alternatives. Alternative k in state i determines:
- Transition probabilities p_ij^k
- Rewards r_ij^k
- Expected immediate reward q_i^k = Σ_j p_ij^k r_ij^k

**Policy:** A vector d = (d_1, d_2, ..., d_N) specifying which alternative to use in each state.

**Gain:** For a completely ergodic process with limiting state probabilities π_i, the gain (average reward per transition) is:

g = Σ_i π_i q_i

**Relative Values:** For large n, the total expected reward from state i satisfies:

v_i(n) = ng + v_i

where v_i is the relative value of state i. The difference v_i - v_j represents the value of starting in state i rather than state j.

## Main Results

### The Value Iteration Equation

For finite-horizon problems with n stages remaining:

v_i(n + 1) = max_k [q_i^k + Σ_j p_ij^k v_j(n)]

This is the application of Bellman's Principle of Optimality to Markovian decision processes.

### The Value-Determination Operation

For a fixed policy with gain g and relative values v_i, the following linear system holds:

g + v_i = q_i + Σ_j p_ij v_j,  for i = 1, 2, ..., N

Setting v_N = 0 (normalization), this yields N equations in N unknowns (g, v_1, ..., v_{N-1}).

### The Policy-Improvement Routine

Given relative values v_i from a current policy, improve by selecting for each state i the alternative k* that maximizes:

q_i^k + Σ_j p_ij^k v_j

This alternative becomes the new decision d_i for state i.

### The Policy Iteration Algorithm

1. **Initialization:** Start with an arbitrary policy (e.g., maximize expected immediate reward in each state)
2. **Value Determination:** Solve the linear system for g and v_i under current policy
3. **Policy Improvement:** For each state, find the alternative maximizing the test quantity
4. **Convergence Check:** If policy unchanged, stop; otherwise return to step 2

### Convergence Theorem

Howard proves:
1. Each iteration produces a policy with gain at least as large as the previous
2. If the policy changes, the gain strictly increases
3. The algorithm terminates in a finite number of iterations at the optimal policy

The proof uses the fact that y_i = [q_i^B + Σ_j p_ij^B v_j^A] - [q_i^A + Σ_j p_ij^A v_j^A] ≥ 0 for all states when policy B is chosen over policy A, with strict inequality for at least one state unless the policies are equivalent.

## Applications Mentioned

Howard presents several extended examples:

1. **The Toymaker Problem:** A business with states "successful toy" and "unsuccessful toy," with alternatives for advertising and research that affect transition probabilities and costs.

2. **Taxicab Operation:** A cab driver choosing between serving different parts of a city based on location and fare probabilities.

3. **Baseball Strategy:** Optimal decision-making for a batter based on balls and strikes count.

4. **Automobile Replacement:** When to replace an aging car given maintenance costs and replacement prices.

## Extensions Covered

- **Discounting (Chapter 7):** For discount factor α < 1, the optimality equations become:
  v_i = max_k [q_i^k + α Σ_j p_ij^k v_j]

- **Continuous-Time Processes (Chapter 8):** Extension to processes where time between transitions is random.

- **Multichain Processes (Chapter 6):** Processes with multiple recurrent classes where limiting behavior depends on starting state.

## Implications for Reinforcement Learning

1. **Policy Iteration:** Howard's algorithm is a foundational RL method, forming one of the two main approaches (alongside value iteration) for solving MDPs exactly.

2. **Average Reward Criterion:** The focus on gain (average reward) provides an alternative to discounted reward, important for continuing tasks without natural termination.

3. **Relative Values:** The concept that v_i - v_j represents the advantage of starting in state i over j anticipates the advantage function A(s, a) = Q(s, a) - V(s) central to modern policy gradient methods.

4. **Computational Efficiency:** Policy iteration typically converges in far fewer iterations than value iteration, though each iteration is more expensive (requiring solution of a linear system).

5. **Finite Convergence:** Unlike value iteration which converges asymptotically, policy iteration terminates exactly in a finite number of steps.

6. **Actor-Critic Structure:** The separation into value determination (evaluating a policy) and policy improvement (finding a better policy) prefigures the actor-critic architecture: the "critic" evaluates the current policy, and the "actor" improves it.

## Notable Quotes

On the motivation for policy iteration:
> "It does not seem efficient to have to iterate v_i(n) for n = 1, 2, 3, and so forth, until we have a sufficiently large n that termination is very remote. We would much rather have a method that directed itself to the problem of analyzing processes of indefinite duration."

On relative values:
> "The difference in the relative values of the two states v_1 - v_2 is equal to the amount that a rational man would be just willing to pay in order to start his transitions from state 1 rather than state 2 if he is going to operate the system for many, many transitions."

On the number of policies:
> "A problem with 50 states and 50 alternatives in each state contains 50^50 (~10^85) policies."
