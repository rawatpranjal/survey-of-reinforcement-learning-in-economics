# 2.3: Rollout for Discrete Optimization

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 177-207
**Topics:** rollout algorithms, discrete optimization, sequential consistency, sequential improvement, fortified rollout, parallel rollout, truncated rollout

---

and

<!-- formula-not-decoded -->

otherwise. The sell or don't sell decision of the rollout algorithm is made on-line according to the preceding criterion, at each state x k encountered during on-line operation.

Figure 1.8.4 shows the rollout policy, which is computed by the preceding equations using the rewards-to-go of the base heuristic J x k k ( x k ), as given in Fig. 1.8.3. Once the rollout policy is computed, the corresponding reward function ˜ J k ( x k ) can be calculated similar to the case of the base heuristic. Of course, during on-line operation, the rollout decision need only be computed for the states x k encountered on-line.

The important observation when comparing Figs. 1.8.3 and 1.8.4 is that the rewards-to-go of the rollout policy are greater or equal to the ones for the base heuristic. In particular, starting from x 0 , the rollout policy attains reward 2.269, and the base heuristic attains reward 2.268. The optimal policy attains reward 2.4. The rollout policy reward is slightly closer to the optimal than the base heuristic reward.

The rollout reward-to-go values shown in Fig. 1.8.4 are 'exact,' and correspond to the favorable case where the heuristic rewards needed at x k , J x k +1 k +1 ( x k +1), J x k k +1 ( x k ), and J x k -1 k +1 ( x k -1), are computed exactly by DP or by infinite-sample Monte Carlo simulation.

When finite-sample Monte Carlo simulation is used to approximate the needed base heuristic rewards at state x k , i.e., J x k +1 k +1 ( x k +1), J x k k +1 ( x k ), and J x k -1 k +1 ( x k -1), the performance of the rollout algorithm will be degraded. In particular, by using a computer program to implement rollout with Monte Carlo simulation, it can be shown that when J x k +1 k +1 ( x k +1), J x k k +1 ( x k ), and J x k -1 k +1 ( x k -1) are approximated using a 20-sample Monte-Carlo simulation per reward value, the rollout algorithm achieves reward 2.264 starting from x 0 . This reward is evaluated by (almost exact) 400-sample Monte Carlo simulation of the rollout algorithm.

When J x k +1 k +1 ( x k + 1), J x k k +1 ( x k ), and J x k -1 k +1 ( x k -1) were approximated using a 200-sample Monte-Carlo simulation per reward value, the rollout algorithm achieves reward 2.273 [as evaluated by (almost exact) 400-sample Monte Carlo simulation of the rollout algorithm]. Thus with 20-sample simulation, the rollout algorithm performs worse than the base heuristic starting from x 0 . With the more accurate 200-sample simulation, the rollout algorithm performs better than the base heuristic starting from x 0 , and performs nearly as well as the optimal policy (but still somewhat worse than in the case where exact values of the needed base heuristic rewards are used (based on an 'infinite' number Monte Carlo samples).

It is worth noting here that the heuristic is not a legitimate policy because at any state x n is makes a decision that depends on the state x k where it started. Thus the heuristic's decision at x n depends not just on x n , but also on the starting state x k . However, the rollout algorithm is always an approximation in value space scheme with approximation reward ˜ J k ( x k ) defined by the heuristic, and it provides a legitimate policy.

12

2

Xo

HO

Expected Rewards (Rollout w/ Exact DP Base Heuristic; B = 1.4)

10

9

3

2

1

7

Expected Reward

5.0

6

3.084

3.062

1.286

4.015

2.231

1

3.043 3.026 3.013

1.208 1.165

6

4.004

2.193

0.355

1.248

2

9

4

3

2.236

2.016

4

3.004

2.154 2.115 2.077 2.043

0.323

0.34

3

6

4

5

7

Policy (Rollout w/ Exact DP Base Heuristic; B = 1.4)

12

11

* 10

9

5

4

2

1

10

9

1

N - 1

2.269

Xo

D

1

D

2

D

D

D

3

S

D

D

D

D

4

S

D

D

D

D

D

5

<!-- image -->

nan

S

S

D

D

D

D

6

Figure 1.8.4 Table of values of reward-to-go and decisions applied by the rollout policy that corresponds to the base heuristic with β = 1 glyph[triangleright] 4.

- (d) Repeat part (c) but with two-step instead of one-step lookahead minimiza-

Answer : The implementation is very similar to the one-step lookahead case. The main di ff erence is that at state x k , the rollout algorithm needs to calculate the base heuristic reward values J x k +2 k +2 ( x k +2), J x k +1 k +2 ( x k +1), J x k k +2 ( x k ), J x k -1 k +2 ( x k -1), and J x k -2 k +2 ( x k -2). Thus the on-line Monte Carlo simulation work is accordingly increased. Generally the simulation work per stage of the rollout algorithm is proportional to 2 /lscript + 1, when /lscript -stage lookahead minimization is used, since the number of leafs at the end of the

- tion. lookahead tree is 2 /lscript +1.

## 1.5 (Computational Exercise - Linear Quadratic Problem)

In a more realistic version of the cruise control system of Example 1.3.1, the system has the form

<!-- formula-not-decoded -->

where the coe ffi cient a satisfies 0 &lt; a ≤ 1, and the disturbance w k has zero mean and variance σ 2 . The cost function has the form

<!-- formula-not-decoded -->

where ¯ x 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ¯ x N are given nonpositive target values (a velocity profile) that serve to adjust the vehicle's velocity, in order to maintain a safe distance from

S

S

N = 10

* = 10

p+ = p = 0.25

x0 = 2

the vehicle ahead, etc. In a practical setting, the velocity profile is recalculated by using on-line radar measurements.

Design an experiment to compare the performance of a fixed linear policy π , derived for a fixed nominal velocity profile, and the performance of the algorithm that uses on-line replanning, whereby the optimal policy π ∗ is recalculated each time the velocity profile changes. Compare with the performance of the rollout policy ˜ π that uses π as the base policy and on-line replanning.

## 1.6 (Computational Exercise - Parking Problem)

In reference to Example 1.6.4, a driver aims to park at an inexpensive space on the way to his destination. There are L parking spaces available and a garage at the end. The driver can move in either direction. For example if he is in space i he can either move to i -1 with a cost t -i , or to i +1 with a cost t + i , or he can park at a cost c ( i ) (if the parking space i is free). The only exception is when he arrives at the garage (indicated by index N ) and he has to park there at a cost C . Moreover, after the driver visits a parking space he remembers its free/taken status and has an option to return to any parking space he has already visited. However, the driver must park within a given number of stages N , so that the problem has a finite horizon. The initial probability of space i being free is given, and the driver can only observe the free/taken status of a parking only after he/she visits the space. Moreover, the free/taken status of a parking visited so far does not change over time.

Write a program to calculate the optimal solution using exact dynamic programming over a state space that is as small as possible. Try to experiment with di ff erent problem data, and try to visualize the optimal cost/policy with suitable graphical plots. Comment on the run-time as you increase the number of parking spots L .

## 1.7 (Newton's Method for Solving the Riccati Equation)

The classical form of Newton's method applied to a scalar equation of the form H ( K ) = 0 takes the form

<!-- formula-not-decoded -->

where ∂ H ( K k ) ∂ K is the derivative of H , evaluated at the current iterate K k . This exercise shows algebraically (rather than graphically), within the context of linear quadratic problems, that in approximation in value space with quadratic cost approximation, the cost function of the corresponding one-step lookahead policy is the result of a Newton step for solving the Riccati equation. To this end, we will apply Newton's method to the solution of the Riccati Eq. (1.42), which we write in the form H ( K ) = 0 ↪ where

<!-- formula-not-decoded -->

- (a) Show that the operation that generates K L starting from K is a Newton iteration of the form (1.104). In other words, show that for all K that lead to a stable one-step lookahead policy, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where is the quadratic cost coe ffi cient of the one-step lookahead linear policy θ ( x ) = Lx corresponding to the cost function approximation J ( x ) = Kx 2 :

<!-- formula-not-decoded -->

Proof : Our approach for showing the Newton step formula (1.106) is to express each term in this formula in terms of L , and then show that the formula holds as an identity for all L . To this end, we first note from Eq. (1.108) that K can be expressed in terms of L as

<!-- formula-not-decoded -->

Furthermore, by using Eqs. (1.108) and (1.109), H ( K ) as given in Eq. (1.105), can be expressed in terms of L as follows:

<!-- formula-not-decoded -->

Moreover, by di ff erentiating the function H of Eq. (1.105), we obtain after a straightforward calculation

<!-- formula-not-decoded -->

where the second equation follows from Eq. (1.108). Having expressed all the terms in the Newton step formula (1.106) in terms of L through Eqs. (1.107), (1.109), (1.110), and (1.111), we can write this formula in terms of L only as

<!-- formula-not-decoded -->

or equivalently as

<!-- formula-not-decoded -->

Figure 1.8.5 Illustration of the performance errors of the one-step and two-step lookahead policies as a function of ˜ K ; cf. Exercise 1.8.

<!-- image -->

A straightforward calculation now shows that this equation holds as an identity for all L .

- (b) What happens when K lies outside the region of stability?
- (c) Show that in the case of /lscript -step lookahead, the analog of the quadratic convergence rate estimate has the form

<!-- formula-not-decoded -->

∣ ∣ where F /lscript -1 ( ˜ K ) is the result of the ( /lscript -1)-fold application of the mapping F to ˜ K . Thus a stronger bound for ♣ K ˜ L -K ∗ ♣ is obtained.

## 1.8 (Error Bounds and Region of Stability)

Consider a one-dimensional linear quadratic problem with problem data a = 2, b = 1, q = 1, r = 1.

- (a) Plot the regions of stability for one-step, two-step, and four-step lookahead (cf. Fig. 1.5.9 in Section 1.5).
- (b) Let ˜ θ 1 and ˜ θ 2 be the one-step and two-step lookahead policies for a given quadratic cost approximation coe ffi cient ˜ K . Verify that the performance errors ♣ K ˜ θ 1 -K ∗ ♣ and ♣ K ˜ θ 2 -K ∗ ♣ as a function of ˜ K are as shown in the plot of Fig. 1.8.5. Interpret the figure in terms of your results of part (a). Verify that longer lookahead expands the region of stability, and that the performance error increases sharply as ˜ K approaches the boundary of the region of stability.
- (c) Experiment with other problem data of your choice, including a range of values of a . Verify that for a system that is already stable ( ♣ a ♣ &lt; 1), the region of stability includes the entire nonnegative axis.

## 1.9 (Region of Stability and the Role of Multistep Lookahead)

In Section 1.5, we discussed the concept of the region of stability in the context of linear quadratic problems. The concept extends to far more general infinite horizon problems (see e.g., the book [Ber22a], Section 3.3). The idea is to call a stationary policy θ unstable if J θ ( x ) = ∞ for some states x , and call it stable otherwise. For /lscript ≥ 1, the /lscript -step region of stability is the set of cost function approximations ˜ J for which the corresponding /lscript -step lookahead policy is stable.

Generally, the /lscript -step region of stability expands as /lscript increases. Note also that in finite-state discounted problems all policies are stable, so all ˜ J belong to the region of stability. However, for SSP this is not so: there are policies, called improper , that do not terminate with positive probability for some initial states (this is very common, for example, in nonacyclic deterministic shortest path problems; see the books [Ber12] and [Ber22b] for extensive discussions). Such policies can be unstable. The following example illustrates di ffi cult problems where the region of instability includes functions that are very close to J ∗ , even with large /lscript . This example involves small stage costs, a class of problems that pose challenges for approximation in value space; see Section 2.6.

Consider a deterministic shortest path problem with a single state 1, plus the termination state t . At state 1 we can either stay at that state at cost /epsilon1 &gt; 0 or move to the state t at cost 1. Thus the optimal policy at state 1 is to move to t , the optimal cost J ∗ (1) = 1, and is the unique solution of Bellman's equation

<!-- formula-not-decoded -->

(In shortest path problems the optimal cost at t is 0 by assumption, and Bellman's equation involves only the costs of the states other than t .)

- (a) Show that the one-step region of stability is the set of all ˜ J (1) &gt; 1 -/epsilon1 . What happens in the case where ˜ J (1) = 1 -/epsilon1 ? Show also that the /lscript -step region of stability is the set of all ˜ J (1) &gt; 1 -/lscript/epsilon1 . Note : The /lscript -step region of stability becomes arbitrarily large for su ffi ciently large /lscript . However, the boundary of the /lscript -step region of stability is arbitrarily close to J ∗ (1) for su ffi ciently small /epsilon1 .
- (b) What happens in the case where there are additional states i = 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , and for each of these states i there is the option to stay at i at cost /epsilon1 or to move to i -1 at cost 0? Partial answer : The one-step region of stability consists of all ˜ J = ( ˜ J ( n ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ J (1) ) such that /epsilon1 + ˜ J ( i ) &gt; ˜ J ( i -1) for all i ≥ 2 and /epsilon1 + ˜ J (1) &gt; 1.

## Approximation in Value Space

## - Rollout Algorithms

| Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2.1. Deterministic Finite Horizon Problems . . . . . . . . p. 166 2.2. Approximation in Value Space - Deterministic Problems p. 173 2.3. Rollout Algorithms for Discrete Optimization . . . . . p. 177 2.3.1. Cost Improvement with Rollout - Sequential Consistency, Sequential Improvement . . . . . . . . . . . . p. 183 2.3.2. The Fortified Rollout Algorithm . . . . . . . . . p. 189 2.3.3. Using Multiple Base Heuristics - Parallel Rollout . p. 192 2.3.4. Using Multiple Rollout Trajectories - General Rollout p. 194 2.3.5. Simplified Rollout Algorithms . . . . . . . . . . p. 195 2.3.6. Truncated Rollout with Terminal Cost Approximation p. 196 2.3.7. Rollout with an Expert - Model-Free Rollout . . . p. 197 2.3.8. Using a World Model for Rollout . . . . . . . . p. 202 2.3.9. Local Search with Rollout for Discrete Optimization p. 204 2.3.10. Inference in n -Grams, Transformers, HMMs, and . . . . Markov Chains . . . . . . . . . . . . . . . . p. 208 2.4. Approximation in Value Space with Multistep Lookahead p. 221 |

| 2.7. Approximation in Value Space - Stochastic Problems . p. 257 2.7.1. Simplified Rollout and Policy Iteration . . . . . . p. 262 2.7.2. Certainty Equivalence Approximations . . . . . . p. 261 2.7.3. Simulation-Based Implementation of the Rollout . . . . . Algorithm . . . . . . . . . . . . . . . . . . . p. 265   |                |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| 2.7.4. Variance Reduction in Rollout - Comparing Advantagesp. 267                                                                                                                                                                                                                                                          |                |
| 2.7.5. Monte Carlo Tree Search . . . . . . . . . . . .                                                                                                                                                                                                                                                                     | p. 270         |
| 2.7.6. Randomized Policy Improvement by Monte Carlo Tree Search . . . . . . . . . . . . . . . . .                                                                                                                                                                                                                          | . . . . p. 274 |
| . 2.8. Rollout for Infinite-Spaces Problems - Optimization . . .                                                                                                                                                                                                                                                           | . . . p. 274   |
| Heuristics . . . . . . . . . . . . . . . . . . . .                                                                                                                                                                                                                                                                         | p. 275         |
| 2.8.1. Rollout for Infinite-Spaces Deterministic Problems                                                                                                                                                                                                                                                                  | . p. 279       |
| 2.8.2. Rollout Based on Stochastic Programming . . . 2.8.3. Stochastic Rollout with Certainty Equivalence . .                                                                                                                                                                                                              | . p. 281       |
| 2.9. Multiagent Rollout . . . . . . . . . . . . . . .                                                                                                                                                                                                                                                                      | . p. 283       |
| 2.9.1. Asynchronous and Autonomous Multiagent Rollout                                                                                                                                                                                                                                                                      | p. 294         |
| 2.10. Rollout for Bayesian Optimization and Sequential . . . . . .                                                                                                                                                                                                                                                         | . . .          |
| Estimation . . . . . . . . . . . . . . .                                                                                                                                                                                                                                                                                   | p.             |
| 2.11. Adaptive Control by Rollout with a POMDP Formulationp. 309                                                                                                                                                                                                                                                           | 297            |
|                                                                                                                                                                                                                                                                                                                            | p. 317         |
| 2.12. Minimax Control and Reinforcement Learning . . . . 2.12.1. Exact Dynamic Programming for Minimax Problems 2.12.2. Minimax Approximation in Value Space and Rollout                                                                                                                                                   | p. 318 p. 321  |
| 2.12.3. Combined Approximation in Value and Policy Space                                                                                                                                                                                                                                                                   | .              |
| for Minimax Control . . . . . . . . . . . .                                                                                                                                                                                                                                                                                | p. 327         |
| A Meta Algorithm for Computer Chess Based on                                                                                                                                                                                                                                                                               | .              |
| Reinforcement Learning . . . . . . . . . . . . 2.12.5. Combined Approximation in Value and Policy                                                                                                                                                                                                                          | p. 328         |
| Space                                                                                                                                                                                                                                                                                                                      | .              |
| for Sequential Noncooperative Games . . . . . .                                                                                                                                                                                                                                                                            | p. 332         |
| 2.13. Notes, Sources, and Exercises . . . . . . . . . . .                                                                                                                                                                                                                                                                  | p.             |
|                                                                                                                                                                                                                                                                                                                            | 334            |
| 2.12.4.                                                                                                                                                                                                                                                                                                                    |                |
| .                                                                                                                                                                                                                                                                                                                          |                |

In this chapter, we discuss various aspects of algorithms that are based on prediction of future costs starting from the current state, such as approximation in value space and rollout algorithms. We focus primarily on the case where the state and control spaces are finite. In Sections 2.1-2.6, we consider finite horizon deterministic problems, which in addition to arising often in practice, o ff er some important advantages in the context of RL. In particular, a finite horizon is well suited for the use of rollout, while the deterministic character of the problem eliminates the need for costly on-line Monte Carlo simulation.

An interesting aspect of our methodology for discrete deterministic problems is that it admits extensions that we have not discussed so far. The extensions include multistep lookahead variants, as well as variants that apply to constrained forms of DP, which involve constraints on the entire system trajectory, and also allow the use of heuristic algorithms that are more general than policies within the context of rollout. These variants rely on the problem's deterministic structure, and do not extend to stochastic problems.

Another interesting aspect of finite state deterministic problems is that they can serve as a framework for an important class of commonly encountered discrete optimization problems, including integer programming and combinatorial optimization problems such as scheduling, assignment, routing, etc. This brings to bear the methodology of approximation in value space, rollout, adaptive control, and MPC, and provides e ff ective suboptimal solution methods for these problems.

In Sections 2.7-2.11, we consider various problems that involve stochastic uncertainty. In Section 2.12, we consider minimax problems that involve set membership uncertainty. The present chapter draws heavily on Chapters 2 and 3 of the book [Ber20a], and Chapter 6 of the book [Ber22a]. These books may be consulted for more details and additional examples.

While our focus in this chapter will be on finite horizon problems, our discussion applies to infinite horizon problems as well, because approximation in value space and rollout are essentially finite-stages algorithms, while the nature of the original problem horizon (be it finite or infinite) a ff ects only the terminal cost function approximation. Thus in implementing onestep or multistep approximation in value space, it makes little di ff erence whether the original problem has finite or infinite horizon. At the same time, for conceptual purposes, we can argue that finite horizon problems, even when they involve a nonstationary system and cost per stage, can be transformed to infinite horizon problems, by introducing an artificial costfree termination state that the system moves into at the end of the horizon; see Sections 1.6.3 and 1.6.4. Through this transformation, the synergy of o ff -line training and on-line play based on Newton's method is brought to bear, and the insights that we discussed in Chapter 1 in the context of an infinite horizon apply and explain the good performance of our methods in practice.

Control Uk

Deterministic Transition

Xk+ 1 = fk (Xk, Uk)

Cost gk (Xk, Uk)

Stage k

Terminal Cost

9N (XN)

Future Stages

Figure 2.1.1 Illustration of a deterministic N -stage optimal control problem. Starting from state x k , the next state under control u k is generated nonrandomly, according to

<!-- image -->

<!-- formula-not-decoded -->

and a stage cost g k ( x k ↪ u k ) is incurred.

## 2.1 DETERMINISTIC FINITE HORIZON PROBLEMS

We recall from Chapter 1, Section 1.2, that in deterministic finite horizon DP problems, the state is generated nonrandomly over N stages, through a system equation of the form

<!-- formula-not-decoded -->

where k is the time index, and x k is the state of the system, an element of some state space X k ,

u k is the control or decision variable, to be selected at time k from some given set U k ( x k ), a subset of a control space U k , that depends on x k , f k is a function of ( x k ↪ u k ) that describes the mechanism by which the state is updated from time k to time k +1.

The state space X k and control space U k are arbitrary sets and may depend on k . Similarly, the system function f k can be arbitrary and may depend on k . The cost incurred at time k is denoted by g k ( x k ↪ u k ), and the function g k may depend on k . For a given initial state x 0 , the total cost of a control sequence ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ is

<!-- formula-not-decoded -->

where g N ( x N ) is a terminal cost incurred at the end of the process. This is a well-defined number, since the control sequence ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ together with x 0 determines exactly the state sequence ¶ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♦ via the system equation (2.1); see Figure 2.1.1. We want to minimize the cost (2.2) over all sequences ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ that satisfy the control constraints, thereby obtaining the optimal value as a function of x 0

<!-- formula-not-decoded -->

Xk

Xk+1)

Notice an important di ff erence from the stochastic case: we optimize over sequences of controls ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ , rather than over policies that consist of a sequence of functions π = ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ ↪ where θ k maps states x k into controls u k = θ k ( x k ), and satisfies the control constraints θ k ( x k ) ∈ U k ( x k ) for all x k . It is well-known that in the presence of stochastic uncertainty, policies are more e ff ective than control sequences, and can result in improved cost. On the other hand for deterministic problems, minimizing over control sequences yields the same optimal cost as over policies, since the cost of any policy starting from a given state determines with certainty the controls applied at that state and the future states, and hence can also be achieved by the corresponding control sequence. This point of view allows more general forms of rollout, which we will discuss in this chapter: instead of using a policy for rollout, we will allow the use of more general heuristics for choosing future controls.

The DP algorithm for finite horizon deterministic problems was derived in Section 1.2. It constructs functions

<!-- formula-not-decoded -->

sequentially, starting from J * N , and proceeding backwards to J * N -1 ↪ J * N -2 ↪ etc. The value J * k ( x k ) will be viewed as the optimal cost of the tail subproblem that starts at state x k at time k and ends at some state x N .

## DP Algorithm for Deterministic Finite Horizon Problems

Start with

<!-- formula-not-decoded -->

and for k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, let

<!-- formula-not-decoded -->

The key fact about the DP algorithm is that for every initial state x 0 , the number J * 0 ( x 0 ) obtained at the last step, is equal to the optimal cost J * ( x 0 ). Indeed, a more general fact was shown in Section 1.2, namely that for all k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, and all states x k at time k , we have

<!-- formula-not-decoded -->

where J ( x k ; u k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ) is the cost generated by starting at x k and using subsequent controls u k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 :

<!-- formula-not-decoded -->

Thus, J * k ( x k ) is the optimal cost for an ( N -k )-stage tail subproblem that starts at state x k and time k , and ends at time N . Based on this interpretation of J ∗ k ( x k ), we have called it the optimal cost-to-go from state x k at stage k , and refer to J ∗ k as the optimal cost-to-go function or optimal cost function at time k .

We have also discussed in Section 1.2 the construction of an optimal control sequence. Once the functions J * 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ J * N have been obtained, we can use a forward algorithm to construct an optimal control sequence ¶ u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ and state trajectory ¶ x ∗ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x ∗ N ♦ for a given initial state x 0 .

Construction of Optimal Control Sequence ¶ u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦

Set and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Sequentially, going forward, for k = 1 ↪ 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, set

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Note an interesting conceptual division of the optimal control sequence construction: there is o ff -line training to obtain J * k by precomputation [cf. the DP Eqs. (2.3)-(2.4)], which is followed by on-line play to obtain u ∗ k [cf. Eq. (2.6)]. This is analogous to the two algorithmic processes described in Section 1.1 in connection with computer chess and backgammon.

## Finite-State Deterministic Problems

For the first five sections of this chapter, we will consider the case where the state and control spaces are discrete and consist of a finite number of elements. As we have noted in Section 1.2, such problems can be described with an acyclic graph specifying for each state x k the possible transitions to next states x k +1 . The nodes of the graph correspond to states x k and the arcs of the graph correspond to state-control pairs ( x k ↪ u k ). Each arc with start node x k corresponds to a choice of a single control u k ∈ U k ( x k ) and has as end node the next state f k ( x k ↪ u k ). The cost of an arc ( x k ↪ u k ) is

Initial States

Stage 0

Cost 91(x1, U1)

State Transition

X2 = f1 (X1, U1)

X2

U1

Stage 1

X1

Stage 2

XN-1

UN-

••• Stage N -1

X2

State Space Partition Initial States

Initial State

Terminal Arcs

Cost gN(IN)

Artificial Terminal

<!-- image -->

Current Position

Current Position

Figure 2.1.2 Illustration of a deterministic finite-state DP problem. Nodes correspond to states x k . Arcs correspond to state-control pairs ( x k ↪ u k ). An arc ( x k ↪ u k ) has start and end nodes x k and x k +1 = f k ( x k ↪ u k ) ↪ respectively. The cost g k ( x k ↪ u k ) of the transition is the length of this arc. An artificial terminal node t is connected with an arc of cost g N ( x N ) with each state x N . The problem is equivalent to finding a shortest path from initial nodes of stage 0 to node t .

defined as g k ( x k ↪ u k ); see Fig. 2.1.2. To handle the final stage, an artificial terminal node t is added. Each state x N at stage N is connected to the terminal node t with an arc having cost g N ( x N ). The control sequences ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ correspond to paths originating at the initial state (a node at stage 0) and terminating at one of the nodes corresponding to the final stage N . With this description it can be seen that a deterministic finitestate finite-horizon problem is equivalent to finding a minimum-length (or shortest) path from the initial nodes of the graph (stage 0) to the terminal node t , as we have discussed in Section 1.2.

Shortest path problems arise in a great variety of application domains. While there are quite a few e ffi cient polynomial algorithms for solving them, some practical shortest path problems are extraordinarily di ffi cult because they involve an astronomically large number of nodes. For example deterministic scheduling problems of the type discussed in Example 1.2.1 can be formulated as shortest path problems, but with a number of nodes that grows exponentially with the number of tasks. For such problems neither exact DP nor any other shortest path algorithm can compute an exact optimal solution in practice. In what follows, we will aim to show that suboptimal solution methods, and rollout algorithms in particular, o ff er a viable alternative.

Many types of search problems involving games and puzzles also admit in principle exact solution by DP, but have to be solved by suboptimal methods in practice. The following is a characteristic example.

## Example 2.1.1 (The Four Queens Problem)

Four queens must be placed on a 4 × 4 portion of a chessboard so that no queen can attack another. In other words, the placement must be such that with Cost Equal

to Ter-

Terminal Arcs with Cost Equal

to Ter-

Dead-End Position

Starting Position

Root Node s

Dead-End Position wil

Length = 0

Length = 1

<!-- image -->

Dead-End Position

Figure 2.1.3 A finite horizon deterministic DP formulation of the four queens problem. Symmetric positions resulting from placing a queen in one of the rightmost squares in the top row have been ignored. Squares containing a queen have been darkened. All arcs have length zero except for those connecting dead-end positions to the artificial terminal node.

every row, column, or diagonal of the 4 × 4 board contains at most one queen. Equivalently, we can view the problem as a sequence of problems; first, placing a queen in one of the first two squares in the top row, then placing another queen in the second row so that it is not attacked by the first, and similarly placing the third and fourth queens. (It is su ffi cient to consider only the first two squares of the top row, since the other two squares lead to symmetric

Solution

Starting

positions; this is an example of a situation where we have a choice between several possible state spaces, but we select the one that is smallest.)

We can associate positions with nodes of an acyclic graph where the root node s corresponds to the position with no queens and the terminal nodes correspond to the positions where no additional queens can be placed without some queen attacking another. Let us connect each terminal position with an artificial terminal node t by means of an arc. Let us also assign to all arcs cost zero except for the artificial arcs connecting terminal positions with less than four queens with the artificial node t . These latter arcs are assigned a cost of 1 (see Fig. 2.1.3) to express the fact that they correspond to dead-end positions that cannot lead to a solution. Then, the four queens problem reduces to finding a minimal cost path from node s to node t , with an optimal sequence of queen placements corresponding to cost 0.

Note that once the states/nodes of the graph are enumerated, the problem is essentially solved. In this 4 × 4 problem the states are few and can be easily enumerated. However, we can think of similar problems with much larger state spaces. For example consider the problem of placing N queens on an N × N board without any queen attacking another. Even for moderate values of N , the state space for this problem can be extremely large (for N = 8 the number of possible placements with exactly one queen in each row is 8 8 = 16 ↪ 777 ↪ 216). It can be shown that there exist solutions to the N queens problem for all N ≥ 4 (for N = 2 and N = 3, clearly there is no solution). Moreover e ff ective (non-DP) search algorithms have been devised for its solution up to very large values of N .

The preceding example illustrates some of the di ffi culties of applying exact DP to discrete/combinatorial problems with the type of formulation that we have described. The state space typically becomes very large, particularly as k increases. In the preceding example, to start a backward DP algorithm, we need to consider all the possible terminal positions, which are too many when N is large. There is an alternative exact DP algorithm for deterministic problems, which proceeds forwards from the initial state. It is simply the backward DP algorithm applied to an equivalent shortest path problem, derived form one of Fig. 2.1.2 by reversing the directions of all the arcs, and exchanging the roles of the origin and the destination. It will be discussed in Section 2.4; see also [Ber17a], Chapter 2. Still, however, this forward DP algorithm cannot overcome the di ffi culty with a very large state space.

## Discrete Optimization Problems

A major class of deterministic problems that can be formulated as DP problems involves the minimization of a cost function G ( u ) over all u within a constraint set U . For the purposes of this chapter, we assume that U is finite, although a similar DP formulation is also possible for the more general case of an infinite set U . In Section 1.6.3, we discussed the case

Artificial

Initial State

Stage 1

U1

States

(чо)

Stage 2

States

Stage 3

Stage N

) Approximate ..

<!-- image -->

) Approximate ..

) Approximate ..

) Approximate ..

) Approximate ..

) Approximate ..

Figure 2.1.4 A DP formulation of the discrete optimization problem of minimizing a cost function G ( u ) over all u within a finite set U , for the case where u consists of N components (cf. Section 1.6.3).

where u consists of N components,

<!-- formula-not-decoded -->

the system equation takes the simple form

<!-- formula-not-decoded -->

and there is just a terminal cost G ( x N ) = G ( u ); see Fig. 1.6.2, repeated here for convenience as Fig. 2.1.4.

The following is a simple but challenging example.

## Example 2.1.2 (Constraint Programming)

An interesting special case of the general optimization problem min u ∈ U G ( u ), where u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ), is a feasibility problem , where G ( u ) ≡ 0, and the problem reduces to finding a value of u that satisfies the constraint. Generally, the structure of the constraint set U is encoded in a graph representing the problem such as the one of Fig. 2.1.4; cf. Section 1.6.3. This type of feasibility problem is also known as a constraint programming problem . The four queens problem (Example 2.1.1) provides an illustration. Another example is the breakthrough problem to be discussed in Section 2.3.

Constraint programming problems can also be transformed into equivalent unconstrained (or less constrained) problems by using problem-dependent penalty functions that eliminate constraints while quantifying the level of constraint violation. As an illustration, consider the case where the problem is to find a feasible solution of a system of constraints of the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This problem can be transformed into the equivalent DP problem of minimizing

<!-- formula-not-decoded -->

subject to the system equation x k +1 = u k ↪ and the control constraints u k ∈ U k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1. Other penalty functions can also be used, such as a quadratic; see the author's nonlinear programming text [Ber16]. This approach is convenient, but it o ff ers no guarantee that it can find a complete feasible solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ), even if one exists. It simply aims to minimize (suboptimally) a measure of the total constraint violation. However, in the process it may be able to find a complete feasible solution, or an infeasible solution that is adequate for practical purposes.

## 2.2 APPROXIMATION IN VALUE SPACE - DETERMINISTIC PROBLEMS

The forward optimal control sequence construction of Eq. (2.6) is possible only after we have computed J * k ( x k ) by DP for all x k and k . Unfortunately, in practice this is often prohibitively time-consuming. However, a similar forward algorithmic process can be used if the optimal cost-to-go functions J * k are replaced by some approximations ˜ J k . This is the idea of approximation in value space that we discussed in Section 1.2.3. It constructs a suboptimal solution ¶ ˜ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u N -1 ♦ in place of the optimal ¶ u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ , by using ˜ J k in place of J * k in the DP procedure (2.6).

Approximation in Value Space - Use of ˜ J k in Place of J * k

Start with and set

<!-- formula-not-decoded -->

Sequentially, going forward, for k = 1 ↪ 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, set

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The expression

<!-- formula-not-decoded -->

which is minimized in approximation in value space [cf. Eq. (2.7)] is known as the (approximate) Q-factor of ( x k ↪ u k ). Note that the computation of the suboptimal control (2.7) can be done through the Q-factor minimization

<!-- formula-not-decoded -->

This suggests the possibility of using approximate o ff -line trained Q-factors in place of cost functions in approximation in value space schemes. However, contrary to the cost approximation scheme (2.7) and its multistep counterparts, the performance may be degraded through the errors in the o ff -line training of the Q-factors (depending on how the training is done).

## Exploiting Structure to Expedite the Lookahead Minimization

An important practical idea is to choose the cost function approximation ˜ J k +1 in Eq. (2.7) in a way that exploits the problem's structure to expedite the computation of the one-step lookahead minimizing control ˜ u k . A noteworthy example arises in multiagent problems with decomposable structure, where the use of separable function approximations ˜ J k +1 leads to decomposition of the minimization of Eq. (2.7) (see the books [Ber19a], Section 2.3.1, and [Ber20a], Section 3.2.4; the author has first encountered this idea through his supervision of the MIT Ph.D. thesis of J. Kimemia [Kim82], [KGB82], which dealt with multi-machine flexible manufacturing). The following example, which is based on recent research by Musunuru et al. [MLW24], illustrates a similar idea but in a di ff erent context.

## Example 2.2.1 (Multidimensional Assignment)

Let us consider multidimensional assignment problems, a class of combinatorial problems that have both a temporal and a spacial allocation structure. They arise in various settings including scheduling, resource allocation, and data association. The general idea is to group together tasks, or resources, or data points, so as to optimize some objective. An example is when we are given a set of m jobs, m persons, and m machines, and we want to select m non-overlapping (job, person, machine) triplets that correspond to minimum cost. Another example is data association problems, whereby we have collected data relating to the movement of some entities (e,g., persons, vehicles) sequentially over a number of time periods, and we want to group together data points that correspond to distinct entities for better inference purposes.

Mathematically, multidimensional assignment problems involve graphs consisting of N +1 subsets of nodes N 0 ↪ N 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N N , and referred to as layers . The arcs of the graphs are directed and are of the form ( i↪ j ), where i is a node in a layer N k , k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, and j is a node in the corresponding

Three-Dimensional Assignment Problem 3 Node Layers Jobs

Figure 2.2.1 Illustration of the graph of an ( N +1)-dimensional assignment problem (here N = 5). There are N + 1 node layers each consisting of m nodes (here m = 4). Each grouping consists of N +1 nodes, one from each layer, and N corresponding arcs. An ( N +1)-dimensional assignment consists of m node-disjoint groupings, where each node belongs to one and only one grouping (illustrated in the figure with thick lines). For each grouping, there is an associated cost that depends on the N -tuple of arcs comprising the grouping. The cost of an ( N + 1)-dimensional assignment is the sum of the costs of its m groupings. The di ffi culty here is that the cost of a grouping does not decompose into a sum of its N arc costs, so the problem cannot be solved by solving N decoupled 2-dimensional assignment problems (for a suboptimal approach based on enforced decoupling, see [Ber20a], Section 3.4.2).

<!-- image -->

next layer N k +1 . Thus we have a directed graph with nodes arranged in N +1 layers, and arcs connecting the nodes of each layer to the nodes in their adjacent layers; see Fig. 2.2.1.

We assume that N ≥ 2, so there are at least three layers. For simplicity, we also assume that each of the layers N k contains the same number of nodes, say m , and that there is a unique arc connecting each node in a given layer with each of the nodes of the adjacent layers. We will present an approximation in value space approach that can be implemented by solving 2-dimensional assignment problems for which very fast algorithms exist, such as the Hungarian method and the auction algorithm (see e.g., [Ber98]).

We consider subsets of N + 1 nodes, referred to as groupings , which consist of a single node from every layer. For each grouping, there is an associated cost, which depends on the N -tuple of arcs that comprise the grouping. A partition of the set of nodes into m disjoint groupings (so that each node belongs to one and only one grouping) is called an ( N +1) -dimensional assignment . The cost of an ( N +1)-dimensional assignment is the sum of the costs of its m groupings. The problem is to find an ( N +1)-dimensional assignment of minimum cost. The exact solution of the problem is very di ffi cult when the cost of a grouping does not decompose into the sum of costs of the N arcs of the grouping (in which case the problem decouples into N easily solvable 2-dimensional assignment problems).

We formulate the problem as an N -stage sequential decision problem where at the k th stage we select the assignment arcs

<!-- formula-not-decoded -->

which connect the nodes i n of layer N k to nodes j n of layer N k +1 on a one-toone basis. The control constraint set U k is the set of legitimate assignments, namely those involving exactly one incident arc per node i n ∈ N k , and exactly one incident arc per node j n of layer N k +1 .

We assume a cost function that has the general form

<!-- formula-not-decoded -->

We use as state x k the partial solution up to time k ,

<!-- formula-not-decoded -->

so the system equation takes the simple form

<!-- formula-not-decoded -->

cf. Fig. 2.1.4 and Section 1.6.3.

The exact DP algorithm takes the form

<!-- formula-not-decoded -->

and,

<!-- formula-not-decoded -->

cf. Eqs. (2.3) and (2.4).

Since this DP algorithm is intractable, we resort to approximation of J ∗ k +1 by a function ˜ J k +1 . The one-step lookahead minimization becomes

<!-- formula-not-decoded -->

but is still formidable because its search space U k is very large. However, it can be greatly simplified by using a cost function approximation ˜ J k +1 with a special structure that is suitable for the use of fast 2-dimensional assignment algorithms. This cost function approximation has the form

<!-- formula-not-decoded -->

where ¶ ( i↪ j ) ∈ u k ♦ denotes the set of m arcs ( i↪ j ) that correspond to the 2-dimensional assignment u k , cf. Eq. (2.8) [thus the dependence of the righthand side on u k comes through the choice of arcs ( i↪ j ) specified by u k ]. The arc costs c ij k +1 ( x k ) in this equation must be calculated for every possible arc ( i↪ j ) that connects a node i ∈ N k with a node j ∈ N k +1 ; see Fig. 2.2.2.

Note that the arc costs c ij k +1 ( x k ) may depend in a complicated way on the previous assignment choices x k = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ), and future assignment choices, which may possibly be determined by a heuristic. For problems that involve tracking the movement of people or vehicles over time, the computation of c ij k +1 ( x k ) may rely on sensor data and problem-dependent circumstances. For other problems, enforced decomposition methods may be useful; see [Ber20a], Section 3.4.2 and the survey by Emami et al. [EPE20]. For an implementation of the approach of the present example and computational results, see Musunuru et al. [MLW24].

(Function of the State) Changing Fixed

Input (Control) Output (Function of the State) Changing Fixed nction of the State) Changing Fixed

Input (Control) Output (Function of the State) Changing Fixed

<!-- image -->

nction of the State) Changing Fixed

Figure 2.2.2 Illustration of a scheme for approximation in value space in a multiassignment problem. At time k , the next states have the form ( x k ↪ u k ), where x k = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ) is the multiassignment trajectory generated up to time k . The scheme calculates the arc cost c ij k +1 ( x k ) for each arc ( i↪ j ) connecting a node i of the time k layer with a node j of the time k +1 layer by extending heuristically the multiassignment trajectory along the 'most likely' assignment arcs over some truncated horizon of m steps ( m = 2 in the figure), and calculating c ij k +1 ( x k ) as the 'cost' of the multiassignment trajectory that starts at time 0, extends to time k + m and passes through arc ( i↪ j ).

Once the arc costs c ij k +1 ( x k ) have been calculated, the assignment u k and corresponding multiassignment trajectory x k +1 are obtained by solving a 2-dimensional assignment problem, using one the fast available algorithms.

## Multistep Lookahead Minimization

The approximation in value space algorithm (2.7) involves a one-step lookahead minimization, since it solves a one-stage DP problem for each k . We may also consider /lscript -step lookahead , which involves the solution of an /lscript -step deterministic DP problem, where /lscript is an integer, 1 &lt; /lscript &lt; N -k , with a terminal cost function approximation ˜ J k + /lscript .

As we have noted in Chapter 1, multistep lookahead typically provides better performance over one-step lookahead in approximation in value space schemes. For example in AlphaZero chess, long multistep lookahead is critical for good on-line performance. On the negative side, the solution of the multistep lookahead minimization problem is more time consuming than its one-step lookahead counterpart. However, the deterministic character of the lookahead minimization problem and the fact that it is solved for the single initial state x k at each time k helps to limit the growth of the lookahead tree and to keep the computation manageable. Moreover, one may try to approximate the solution of the multistep lookahead minimization problem (see Section 2.4).

## 2.3 ROLLOUTALGORITHMSFORDISCRETEOPTIMIZATION

The construction of suitable approximate cost-to-go functions ˜ J k +1 for ap-

-

‖

-

XO

X1

Current State

•••

Tk

Next States

Xk+1

Xk+1

Ик

Heuristic

Heuristic

Heuristic

XN

X'N

x"

N

Figure 2.3.1 Schematic illustration of rollout with one-step lookahead. At state x k , for every pair ( x k ↪ u k ), u k ∈ U k ( x k ), the base heuristic generates a Q-factor

<!-- image -->

<!-- formula-not-decoded -->

and the rollout algorithm selects the control ˜ θ k ( x k ) with minimal Q-factor.

proximation in value space can be done in many di ff erent ways, including some of the principal RL methods. A method of particular interest for our course is rollout , whereby the approximate values ˜ J k +1 ( x k +1 ) in Eq. (2.7) are obtained when needed by running for each u k ∈ U k ( x k ) a heuristic control scheme, called base heuristic , for a suitably large number of steps, starting from x k +1 = f k ( x k ↪ u k ).

The base heuristic can be any method, which starting from a state x k +1 generates a sequence of controls u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 , the corresponding sequence of states x k +2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N , and the cost of the heuristic starting from x k +1 , which we will generically denote by H k +1 ( x k +1 ) in this chapter:

<!-- formula-not-decoded -->

This value of H k +1 ( x k +1 ) is the one used as the approximate cost ˜ J k +1 ( x k +1 ) in the corresponding approximation in value space scheme (2.7).

In this section, we will develop in more detail the theory of rollout with one-step lookahead minimization for deterministic problems, including the important issue of cost improvement. We will also illustrate several variants of the method, and we will consider questions of e ffi cient implementation. We will then discuss examples of discrete optimization applications.

Let us consider a deterministic DP problem with a finite number of controls and a given initial state (so the number of states that can be reached from the initial state is also finite). We first focus on the pure form of rollout that uses one-step lookahead without truncation, and hence no terminal cost approximation. Given a state x k at time k , this algorithm considers the tail subproblems that start at every possible next state x k +1 , and solves them suboptimally with the base heuristic.

Thus when at x k , rollout generates on-line the next states x k +1 that correspond to all u k ∈ U k ( x k ), and uses the base heuristic to compute the sequence of states ¶ x k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♦ and controls ¶ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ such that

<!-- formula-not-decoded -->

and the corresponding cost

<!-- formula-not-decoded -->

The rollout algorithm then applies the control that minimizes over u k ∈ U k ( x k ) the tail cost expression for stages k to N :

<!-- formula-not-decoded -->

Equivalently, and more succinctly, the rollout algorithm applies at state x k the control ˜ θ k ( x k ) given by the minimization

<!-- formula-not-decoded -->

where ˜ Q k ( x k ↪ u k ) is the approximate Q-factor defined by

<!-- formula-not-decoded -->

see Fig. 2.3.1. The rollout algorithm thus defines a suboptimal policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ ↪ referred to as the rollout policy , where for each x k and k , ˜ θ k ( x k ) is the control produced by the Q-factor minimization (2.9).

Note that the rollout algorithm requires running the base heuristic for a number of times that is bounded by Nn , where n is an upper bound on the number of control choices available at each state. Thus if n is small relative to N , the algorithm requires computation equal to a small multiple of N times the computation time for a single application of the base heuristic. Similarly, if n is bounded by a polynomial in N , the ratio of the rollout algorithm computation time to the base heuristic computation time is a polynomial in N .

In Section 1.2 we considered an example of rollout involving the traveling salesman problem and the nearest neighbor heuristic (cf. Examples 1.2.2 and 1.2.3). Let us consider another example, which involves a classical discrete optimization problem.

## Example 2.3.1 (Multi-Vehicle Routing)

Consider m vehicles that move along the arcs of a given graph. Some of the nodes of the graph include a task to be performed by the vehicles. Each task will be performed only once, immediately after some vehicle reaches the corresponding node for the first time. We assume a horizon that is large

12

11

10

9

8

7

$

5

4

6

Base heuristic

10 11 12

towards its nearest pending task, until all tasks are performed

Move each vehicle one step at a time towards its nearest pending task, Move each vehicle one step at a time towards its nearest pending task, until all tasks are performed

Vehicle 2

2

Optimal

Figure 2.3.2 An instance of the vehicle routing problem of Example 2.3.1. The two vehicles aim to collectively perform the two tasks, at nodes 7 and 9, as fast as possible, by each moving to a neighboring node at each step. The optimal routes are shown.

<!-- image -->

enough to allow every task to be performed. The problem is to find a route for each vehicle so that the tasks are collectively performed by the vehicles in a minimum number of moves. To express this objective, we assume that for each move by a vehicle there is a cost of one unit. These costs are summed up to the point where all the tasks have been performed.

For a large number m of vehicles and a complicated graph, this is a nontrivial combinatorial problem. It can be approached by DP, like any discrete deterministic optimization problem, as we have discussed. In particular, we can view as state at a given stage the m -tuple of current positions of the vehicles together with the list of pending tasks. Unfortunately, however, the number of these states can be enormous (it increases exponentially with the number of tasks and the number of vehicles), so an exact DP solution is intractable.

This motivates an optimization in value space approach based on rollout. For this we need an easily implementable base heuristic that will solve suboptimally the problem starting from any state x k +1 , and will provide the cost approximation ˜ J k +1 ( x k +1 ) in Eq. (2.7). One possibility is based on the vehicles choosing their actions selfishly and without coordination, along shortest paths to their nearest pending task.

To illustrate, consider the two-vehicle problem of Fig. 2.3.2. The base heuristic is to move each vehicle one step at a time towards its nearest pending task, until all tasks have been performed.

The rollout algorithm will work as follows. At a given state x k [involving

for example vehicle positions at the node pair (1 ↪ 2) and tasks at nodes 7 and 9, as in Fig. 2.3.2], we consider all possible joint vehicle moves (the controls u k at the state) resulting in the node pairs (3,5), (4,5), (3,4), (4,4), corresponding to the next states x k +1 [thus, as an example (3,5) corresponds to vehicle 1 moving from 1 to 3, and vehicle 2 moving from 2 to 5]. We then run the base heuristic starting from each of these node pairs, and accumulate the incurred costs up to the time when both tasks are completed. For example starting from the vehicle positions/next state (3,5), the heuristic will produce the following sequence of moves:

- ÷ Vehicles 1 and 2 move from (3,5) to (6,2).
- ÷ Vehicles 1 and 2 move from (6,2) to (9,4), and the task at 9 is performed.
- ÷ Vehicles 1 and 2 move from (9,4) to (12,7), and the task at 7 is performed.

The two tasks are thus performed in a total of 6 vehicles moves once the move to (3,5) has been made.

The process of running the heuristic is repeated from the other three vehicle position pairs/next states (4,5), (3,4) (4,4), and the heuristic cost (number of moves) is recorded. We then choose the next state that corresponds to minimum cost. In our case the joint move to state x k +1 that involves the pair (3 ↪ 4) produces the sequence

- ÷ Vehicles 1 and 2 move from (3,4) to (6,7), and the task at 7 is performed.
- ÷ Vehicles 1 and 2 move from (6,7) to (9,4), and the task at 9 is performed.

and performs the two tasks in a total of 6 vehicle moves. It can be verified that it yields minimum first stage cost plus heuristic cost from the next state, as per Eq. (2.7). Thus, the rollout algorithm will choose to move the vehicles to state (3,4) from state (1,2). At that state the rollout process will be repeated, i.e., consider the possible next joint moves to the node pairs (6,7), (6,2), (6,1), (1,7), (1,2), (1,1), perform a heuristic calculation from each, compare, etc.

It can be verified that the rollout algorithm starting from the state (1,2) shown in Fig. 2.3.2 will attain the optimal cost (a total of 6 vehicle moves). It will perform much better than the heuristic, which starting from state (1,2), will move the two vehicles together to state (4,4), then to (7,7), then to (10,10), then to (12,12), and finally to (9,9), (a total of 10 vehicle moves). This is an instance of the cost improvement property of the rollout algorithm: it performs better than its base heuristic (under appropriate conditions).

Let us finally note that the computation required by in rollout algorithm increases exponentially with the number m of vehicles, since the number of m -tuples of moves at each stage increases exponentially with m . This is the type of problem where multiagent rollout can attain great computational savings; cf. Section 1.6.7, and Section 2.9.

Here is an example of a search problem, whose exact solution complexity grows exponentially with the problem size, but can be addressed with a greedy heuristic as well as with the corresponding rollout algorithm.

Continue Terminate Instruction Accept

Figure 2.3.3 Binary tree for the breakthrough problem. Each arc is either free or is blocked (crossed out in the figure). The problem is to find a path from the root to one of the leaves, which is free (such as the one shown with thick lines). Note that the problem is related to the constraint programming problem discussed earlier in Section 2.1.

<!-- image -->

## Example 2.3.2 (The Breakthrough Problem)

Consider a binary tree with N stages as shown in Fig. 2.3.3. Stage k of the tree has 2 k nodes, with the node of stage 0 called root and the nodes of stage N called leaves . There are two types of tree arcs: free and blocked . A free (or blocked) arc can (cannot, respectively) be traversed in the direction from the root to the leaves. The objective is to break through the graph with a sequence of free arcs (a free path) starting from the root, and ending at one of the leaves. (A variant of this problem is to introduce a positive cost c &gt; 0 for traversing a blocked arc, and 0 cost for traversing a free arc.)

One may use DP to discover a free path (if one exists) by starting from the last stage and by proceeding backwards to the root node. The k th step of the algorithm determines for each node of stage N -k whether there is a free path from that node to some leaf node, by using the results of the preceding step. The amount of calculation at the k th step is O (2 N -k ). Adding the computations for the N stages, we see that the total amount of calculation is O ( N 2 N ), so it increases exponentially with the number of stages. For this reason it is interesting to consider heuristics requiring computation that is linear or polynomial in N , but may sometimes fail to determine a free path, even when a free path exists.

Thus, one may suboptimally use a greedy algorithm, which starts at the root node, selects a free outgoing arc (if one is available), and tries to construct a free path by adding successively nodes to the path. At the current node, if one of the outgoing arcs is free and the other is blocked, the greedy algorithm selects the free arc. Otherwise, it selects one of the two outgoing arcs according to some fixed rule that depends only on the current node (and not on the status of other arcs). Clearly, the greedy algorithm may fail to find a free path even if such a path exists, as can be seen from Fig. 2.3.3. On the other hand the amount of computation associated with the greedy

algorithm is O ( N ), which is much faster than the O ( N 2 N ) computation of the DP algorithm. Thus we may view the greedy algorithm as a fast heuristic, which is suboptimal in the sense that there are problem instances where it fails while the DP algorithm succeeds.

One may also consider a rollout algorithm that uses the greedy algorithm as the base heuristic. There is an analysis that compares the probability of finding a breakthrough solution with the greedy and with the rollout algorithm for random instances of binary trees (each arc is independently free or blocked with given probability p ). This analysis is given in Section 6.4 of the book [Ber17a], and shows that asymptotically, the rollout algorithm requires O ( N ) times more computation, but has an O ( N ) times larger probability of finding a free path than the greedy algorithm.

This tradeo ff is qualitatively typical: the rollout algorithm achieves a substantial performance improvement over the base heuristic at the expense of extra computation that is equal to the computation time of the base heuristic times a factor that is a low order polynomial of the problem size.

## 2.3.1 Cost Improvement with Rollout - Sequential Consistency, Sequential Improvement

The definition of the rollout algorithm leaves open the choice of the base heuristic. There are several types of suboptimal solution methods that can be used as base heuristics, such as greedy algorithms, local search, genetic algorithms, and others.

Intuitively, we expect that the rollout policy's performance is no worse than the one of the base heuristic: since rollout optimizes over the first control before applying the heuristic, it makes sense to conjecture that it performs better than applying the heuristic without the first control optimization. However, some special conditions must hold in order to guarantee this cost improvement property. We provide two such conditions, sequential consistency and sequential improvement , introduced in the paper by Bertsekas, Tsitsiklis, and Wu [BTW97], and we later show how to modify the algorithm to deal with the case where these conditions are not met.

Definition 2.3.1: We say that the base heuristic is sequentially consistent if it has the property that when it generates the sequence

<!-- formula-not-decoded -->

starting from state x k , it also generates the sequence

<!-- formula-not-decoded -->

starting from state x k +1 .

In other words, the base heuristic is sequentially consistent if it 'stays the course': when the starting state x k is moved forward to the next state x k +1 of its state trajectory, the heuristic will not deviate from the remainder of the trajectory.

As an example, the reader may verify that the nearest neighbor heuristic described in the traveling salesman Example 1.2.3 and the heuristics used in the multi-vehicle routing Example 2.3.1 are sequentially consistent. Similar examples include the use of various types of greedy/myopic heuristics (Section 6.4 of the book [Ber17a] provides additional examples). Generally most heuristics used in practice satisfy the sequential consistency condition at 'most' states x k . However, some heuristics of interest may violate this condition at some states.

A sequentially consistent base heuristic can be recognized by the fact that it will apply the same control u k at a state x k , no matter what position x k occupies in a trajectory generated by the base heuristic. Thus a base heuristic is sequentially consistent if and only if it defines a legitimate DP policy . This is the policy that moves from x k to the state x k +1 that lies on the state trajectory ¶ x k ↪ x k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♦ that the base heuristic generates. Similarly the policy moves from x n to the state x n +1 for n = k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1.

We will now show that the rollout algorithm obtained with a sequentially consistent base heuristic has a fundamental cost improvement property: it yields no worse cost than the base heuristic. The amount of cost improvement cannot be easily quantified, but is determined by the performance of the Newton step associated with the rollout policy, so it can be very substantial; cf. the discussion of Chapter 1.

Proposition 2.3.1: (Cost Improvement Under Sequential Consistency) Consider the rollout policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ obtained with a sequentially consistent base heuristic, and let J k↪ ˜ π ( x k ) denote the cost obtained with ˜ π starting from x k at time k . Then we have

<!-- formula-not-decoded -->

where H k ( x k ) denotes the cost of the base heuristic starting from x k .

A subtle but important point relates to how one breaks ties while implementing greedy base heuristics. For sequential consistency, one must break ties in a consistent way at various states, i.e., using a fixed rule at each state encountered by the base heuristic. In particular, randomization among multiple controls, which are ranked as equal by the greedy optimization of the heuristic, violates sequential consistency, and can lead to serious degradation of the corresponding rollout algorithm's performance.

Proof: We prove the inequality (2.11) by induction. Clearly it holds for k = N , since

<!-- formula-not-decoded -->

Assume that it holds for index k +1. For any state x k , let u k be the control applied by the base heuristic at x k . Then we have

<!-- formula-not-decoded -->

where:

- ÷ The first equality is the DP equation for the rollout policy ˜ π .
- ÷ The first inequality holds by the induction hypothesis.
- ÷ The second equality holds by the definition of the rollout algorithm.
- ÷ The third equality is the DP equation for the policy that corresponds to the base heuristic (this is the step where we need sequential consistency).

This completes the proof of the cost improvement property (2.11). Q.E.D.

## Sequential Improvement

We will next show that the rollout policy has no worse performance than its base heuristic under a condition that is weaker than sequential consistency. Let us recall that the rollout algorithm ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ is defined by the minimization

<!-- formula-not-decoded -->

where ˜ Q k ( x k ↪ u k ) is the approximate Q-factor defined by

<!-- formula-not-decoded -->

[cf. Eq. (2.10)], and H k +1 ( f k ( x k ↪ u k ) ) denotes the cost of the trajectory of the base heuristic starting from state f k ( x k ↪ u k ).

Definition 2.3.2: We say that the base heuristic is sequentially improving if for all x k and k , we have

<!-- formula-not-decoded -->

In words, the sequential improvement property (2.13) states that

Minimal heuristic Q-factor at x k ≤ Heuristic cost at x k glyph[triangleright]

Note that when the heuristic is sequentially consistent it is also sequentially improving . This follows from the preceding relation, since for a sequentially consistent heuristic, the heuristic cost at x k is equal to the Q-factor of the control u k that the heuristic applies at x k ,

<!-- formula-not-decoded -->

which is greater or equal to the minimal Q-factor at x k . This implies Eq. (2.13). A sequentially improving heuristic yields policy improvement as the next proposition shows.

Proposition 2.3.2: (Cost Improvement Under Sequential Improvement) Consider the rollout policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ obtained with a sequentially improving base heuristic, and let J k↪ ˜ π ( x k ) denote the cost obtained with ˜ π starting from x k at time k . Then

<!-- formula-not-decoded -->

where H k ( x k ) denotes the cost of the base heuristic starting from x k .

Proof: Follows from the calculation of Eq. (2.12), by replacing the last two steps (which rely on sequential consistency) with Eq. (2.13). Q.E.D.

Thus the rollout algorithm obtained with a sequentially improving base heuristic, will improve or at least will perform no worse than the base heuristic, from every starting state x k . In fact the algorithm has a monotonic improvement property, whereby it discovers a sequence of improved trajectories . In particular, let us denote the trajectory generated by the base heuristic starting from x 0 by

<!-- formula-not-decoded -->

ũo

й1

X2

.. •

Xk-1

Monotonicity Property

Under Sequential Improvement

Cost of Tk ≥ Cost of Tk+1

Current Trajectory Tk+1

Optimal Base Rollout Terminal Score Approximation Current

Figure 2.3.4 Proof of the monotonicity property (2.14). At ˜ x k , the k th state generated by the rollout algorithm, we compare the 'current' trajectory T k whose cost is the sum of the cost of the current partial trajectory ( x 0 ↪ ˜ u 0 ↪ ˜ x 1 ↪ ˜ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ x k ) and the cost H k (˜ x k ) of the base heuristic starting from ˜ x k , and the trajectory T k +1 whose cost is the sum of the cost of the partial rollout trajectory ( x 0 ↪ ˜ u 0 ↪ ˜ x 1 ↪ ˜ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ x k ), and the Q-factor ˜ Q k (˜ x k ↪ ˜ u k ) of the base heuristic starting from (˜ x k ↪ ˜ u k ). The sequential improvement condition guarantees that

<!-- image -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If strict inequality holds, the rollout algorithm will switch from T k and follow T k +1 ; cf. the traveling salesman Example 1.2.3.

and the final trajectory generated by the rollout algorithm starting from x 0 by

<!-- formula-not-decoded -->

Consider also the intermediate trajectories generated by the rollout algorithm given by

<!-- formula-not-decoded -->

where

Trajectory Tk

Base Heuristic Cost Hk (k)

Xk+1

which implies that

Monotonicity Property Under Sequential Improvement

<!-- formula-not-decoded -->

is the trajectory generated by the base heuristic starting from ˜ x k . Then, by using the sequential improvement condition, it can be proved (see Fig. 2.3.4) that

<!-- formula-not-decoded -->

Empirically, it has been observed that the cost improvement obtained by rollout with a sequentially improving heuristic is typically considerable and often dramatic. In particular, many case studies, dating to the middle