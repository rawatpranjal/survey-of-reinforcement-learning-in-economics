# Dynamic Programming and Markov Processes (Howard, 1960)

## The Problem (Layperson)

Bellman's dynamic programming solved sequential decision problems with known time horizons. But many real problems continue indefinitely: a machine must be maintained forever, a business must allocate resources month after month, a gambler faces an endless sequence of bets.

For these infinite-horizon problems, backward induction cannot start from a terminal period that does not exist. Moreover, the optimal decision in any state should not depend on how much time has passed if the problem structure is stationary. A new computational approach was needed.

## What Didn't Work (Alternatives)

Finite-horizon approximation truncated the problem at some period $T$ and applied standard backward induction. This introduced approximation error that depended on the choice of $T$, and different choices could yield different policies. The method was ad hoc and lacked theoretical justification.

Direct solution of the infinite-horizon Bellman equation treated $V(s) = \max_a \{r(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')\}$ as a system of simultaneous equations. For problems with continuous states or many discrete states, this was intractable.

Exhaustive policy enumeration considered all possible stationary policies (mappings from states to actions) and evaluated each. With $|A|$ actions and $|S|$ states, there are $|A|^{|S|}$ stationary policies. This grew exponentially and was practical only for tiny problems.

## The Key Insight

Howard introduced policy iteration: alternate between policy evaluation (computing the value of a fixed policy) and policy improvement (finding a better policy given current values).

For a fixed policy $\pi$, the value function $V^\pi$ satisfies a system of linear equations:
$$V^\pi(s) = r(s,\pi(s)) + \gamma \sum_{s'} P(s'|s,\pi(s)) V^\pi(s')$$

This can be solved exactly by matrix inversion or iteratively. Given $V^\pi$, a strictly improved policy $\pi'$ is found by:
$$\pi'(s) = \arg\max_a \{r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\}$$

If $\pi' = \pi$, then $\pi$ is optimal. Otherwise, $\pi'$ is strictly better, and we repeat. Since there are finitely many policies and each iteration improves, the algorithm must terminate at the optimal policy.

## The Method

**Policy Iteration Algorithm:**

1. Initialize with any policy $\pi_0$

2. **Policy Evaluation:** Solve for $V^{\pi_k}$:
   $$V^{\pi_k}(s) = r(s,\pi_k(s)) + \gamma \sum_{s'} P(s'|s,\pi_k(s)) V^{\pi_k}(s')$$
   This is a linear system in $|S|$ unknowns.

3. **Policy Improvement:** Compute the improved policy:
   $$\pi_{k+1}(s) = \arg\max_a \{r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^{\pi_k}(s')\}$$

4. If $\pi_{k+1} = \pi_k$, stop; $\pi_k$ is optimal. Otherwise, set $k \leftarrow k+1$ and go to step 2.

**Value Iteration** (an alternative): repeatedly apply the Bellman operator:
$$V_{k+1}(s) = \max_a \{r(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s)\}$$

This converges to $V^*$ for any initial $V_0$, with the greedy policy converging to optimal.

## The Result

Policy iteration typically converges in surprisingly few iterations. For many practical problems, convergence occurs in fewer than 10 iterations regardless of problem size. Each iteration requires solving a linear system (polynomial time), so the total computation is polynomial.

Howard applied the method to equipment replacement (when to replace aging machines), highway maintenance, baseball strategy, and various economic models. The formalism of Markov decision processes (MDPs) provided a general framework for sequential decision problems under uncertainty.

The work established the canonical formulation: states $S$, actions $A$, transitions $P(s'|s,a)$, rewards $r(s,a)$, and discount $\gamma$. This vocabulary and mathematical structure became the foundation of reinforcement learning.

## Worked Example

Consider a simple machine replacement problem. A machine can be "good" or "bad." A good machine stays good with probability 0.9 and produces profit 10. A bad machine stays bad with probability 0.8 and produces profit 2. At any time, we can pay 15 to replace any machine with a new (good) one. Discount factor $\gamma = 0.9$.

States: $S = \{G, B\}$
Actions: $A = \{\text{keep}, \text{replace}\}$

**Iteration 0:** Start with policy "always keep."

Policy evaluation:
$$V^{\pi_0}(G) = 10 + 0.9[0.9 V^{\pi_0}(G) + 0.1 V^{\pi_0}(B)]$$
$$V^{\pi_0}(B) = 2 + 0.9[0.2 V^{\pi_0}(G) + 0.8 V^{\pi_0}(B)]$$

Solving: $V^{\pi_0}(G) = 84.21$, $V^{\pi_0}(B) = 53.68$

Policy improvement:
- In state G: keep gives $10 + 0.9[0.9(84.21) + 0.1(53.68)] = 78.95$
            replace gives $-15 + 0.9[0.9(84.21) + 0.1(53.68)] = 63.95$
  Keep is better.

- In state B: keep gives $2 + 0.9[0.2(84.21) + 0.8(53.68)] = 56.44$
            replace gives $-15 + 0.9[0.9(84.21) + 0.1(53.68)] = 63.95$
  Replace is better!

**Iteration 1:** New policy: keep if good, replace if bad.

Policy evaluation:
$$V^{\pi_1}(G) = 10 + 0.9[0.9 V^{\pi_1}(G) + 0.1 V^{\pi_1}(B)]$$
$$V^{\pi_1}(B) = -15 + 0.9[0.9 V^{\pi_1}(G) + 0.1 V^{\pi_1}(B)]$$

Note: after replacing, machine is good, so we use transition probabilities for good machine.

Solving: $V^{\pi_1}(G) = 90.57$, $V^{\pi_1}(B) = 65.57$

Policy improvement: both states retain their actions (verify by calculation).

$\pi_1$ is optimal. The algorithm converged in 2 iterations.

## Subtleties

Policy iteration solves a linear system at each iteration, which costs $O(|S|^3)$ for direct methods. For large state spaces, this is expensive. Modified policy iteration performs only partial evaluation (a few Bellman backups) before improving, interpolating between policy iteration and value iteration.

Value iteration avoids solving linear systems but may require many iterations to converge. The convergence rate depends on the discount factor: smaller $\gamma$ means faster convergence. Near $\gamma = 1$, value iteration can be very slow.

The average-reward formulation handles undiscounted problems where total reward is infinite. Howard developed techniques for this case, which arises in many economic applications.

Continuous state spaces require approximation. Howard's presentation focused on finite MDPs, but the ideas extend to function approximation settings that became central to reinforcement learning.

Partial observability, where the state is not directly observable, complicates the analysis substantially. Howard's framework assumed full state observability; POMDPs would later extend the theory.

## Critical Debates

Discounting versus average reward: The discount factor $\gamma$ determines how much future rewards matter. Howard analyzed both formulations, but their relationship to economic concepts like interest rates and time preference sparked debate.

Computation versus theory: Howard's work was practical, aimed at solving real problems. Some mathematicians criticized the lack of rigorous convergence proofs. Subsequent work (notably by Blackwell and others) provided the missing rigor.

Model-based versus model-free: Howard's methods require knowing the transition probabilities. This assumption was acceptable for many operations research problems but limiting for learning agents who must discover dynamics through experience.

The relationship to control theory: Optimal control developed independently in engineering, using continuous-time differential equations (the Hamilton-Jacobi-Bellman equation). Howard's discrete-time, probabilistic framework and control theory's continuous-time, deterministic framework represent two traditions that have gradually merged.

## Key Quotes

"The policy-improvement routine... constitutes one of the basic ideas underlying all Markovian decision processes." (Chapter 2)

"Each iteration of this algorithm requires the solution of a system of linear equations and the comparison of each of a finite number of alternatives. Therefore, each iteration requires only a finite amount of computation." (On computational tractability)

"The sequence of policies generated by this procedure forms a monotonically improving sequence." (On convergence)

## Citation

Howard, R. A. (1960). *Dynamic Programming and Markov Processes*. MIT Press.
