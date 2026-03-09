# 1.4: Infinite Horizon Problems Overview

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 42-55
**Topics:** infinite horizon, infinite horizon methodology, approximation in value space, understanding approximation

---

optimal Q-factors , defined for all pairs ( x k ↪ u k ) and k by

<!-- formula-not-decoded -->

Thus the optimal Q-factors are simply the expressions that are minimized in the right-hand side of the DP equation (1.5).

Note that the optimal cost function J * k can be recovered from the optimal Q-factor Q * k by means of the minimization

<!-- formula-not-decoded -->

Moreover, the DP algorithm (1.5) can be written in an essentially equivalent form that involves Q-factors only [cf. Eqs. (1.9)-(1.10)]:

<!-- formula-not-decoded -->

Exact and approximate forms of this and other related algorithms, including counterparts for stochastic optimal control problems, comprise an important class of RL methods known as Q-learning .

## 1.2.3 Approximation in Value Space and Rollout

The forward optimal control sequence construction of Eq. (1.8) is possible only after we have computed J * k ( x k ) by DP for all x k and k . Unfortunately, in practice this is often prohibitively time-consuming, because the number of possible x k and k can be very large. However, a similar forward algorithmic process can be used if the optimal cost-to-go functions J * k are replaced by some approximations ˜ J k . This is the basis for an idea that is central in RL: approximation in value space . ‡ It constructs a suboptimal solution ¶ ˜ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u N -1 ♦ in place of the optimal ¶ u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ , based on using ˜ J k in place of J * k in the DP algorithm (1.8).

The term 'Q-factor' has been used in the books [BeT96], [Ber19a], [Ber20a] and is adopted here as well. Another term used is 'action value' (at a given state). The terms 'state-action value' and 'Q-value' are also common in the literature. The name 'Q-factor' originated in reference to the notation used in an influential Ph.D. thesis [Wat89] that proposed the use of Q-factors in RL.

‡ Approximation in value space (sometimes called 'search' or 'tree search' in the AI literature) is a simple idea that has been used quite extensively for deterministic problems, well before the development of the modern RL methodology. For example it conceptually underlies the widely used A ∗ method for computing approximate solutions to large scale shortest path problems. For a view of A ∗ that is consistent with our approximate DP framework, the reader may consult the author's DP book [Ber17a].

## Approximation in Value Space - Use of ˜ J k in Place of J *

Start with and set

<!-- formula-not-decoded -->

Sequentially, going forward, for k = 1 ↪ 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, set

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In approximation in value space the calculation of the suboptimal sequence ¶ ˜ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u N -1 ♦ is done by going forward (no backward calculation is needed once the approximate cost-to-go functions ˜ J k are available). This is similar to the calculation of the optimal sequence ¶ u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ , and is independent of how the functions ˜ J k are computed. The motivation for approximation in value space for stochastic DP problems is vastly reduced computation relative to the exact DP algorithm (once ˜ J k have been obtained): the minimization (1.11) needs to be performed only for the N states x 0 ↪ ˜ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ x N -1 that are encountered during the on-line control of the system, and not for every state within the potentially enormous state space, as is the case for exact DP.

The algorithm (1.11) is said to involve a one-step lookahead minimization , since it solves a one-stage DP problem for each k . In what follows we will also discuss the possibility of multistep lookahead , which involves the solution of an /lscript -step DP problem, where /lscript is an integer, 1 &lt; /lscript &lt; N -k , with a terminal cost function approximation ˜ J k + /lscript . Multistep lookahead typically (but not always) provides better performance over one-step lookahead in RL approximation schemes. For example in AlphaZero chess, long multistep lookahead is critical for good on-line performance. The intuitive reason is that with /lscript stages being treated 'exactly' (by optimization), the e ff ect of the approximation error

<!-- formula-not-decoded -->

tends to become less significant as /lscript increases. However, the solution of the multistep lookahead optimization problem, instead of the one-step lookahead counterpart of Eq. (1.11), becomes more time consuming.

k

## Rollout with a Base Heuristic for Deterministic Problems

A major issue in value space approximation is the construction of suitable approximate cost-to-go functions ˜ J k . This can be done in many di ff erent ways, giving rise to some of the principal RL methods. For example, ˜ J k may be constructed with a sophisticated o ff -line training method, as discussed in Section 1.1. Alternatively, ˜ J k may be obtained on-line with rollout , which will be discussed in detail in this book. In rollout, the approximate values ˜ J k ( x k ) are obtained when needed by running a heuristic control scheme, called base heuristic or base policy , for a suitably large number of stages, starting from the state x k , and accumulating the costs incurred at these stages.

The major theoretical property of rollout is cost improvement : the cost obtained by rollout using some base heuristic is less or equal to the corresponding cost of the base heuristic. This is true for any starting state, provided the base heuristic satisfies some simple conditions, which will be discussed in Chapter 2.

There are also several variants of rollout, including versions involving multiple heuristics, combinations with other forms of approximation in value space methods, multistep lookahead, and stochastic uncertainty. We will discuss such variants later. For the moment we will focus on a deterministic DP problem with a finite number of controls. Given a state x k at time k , this algorithm considers all the tail subproblems that start at every possible next state x k +1 , and solves them suboptimally by using some algorithm, referred to as base heuristic.

Thus when at x k , rollout generates on-line the next states x k +1 that correspond to all u k ∈ U k ( x k ), and uses the base heuristic to compute the sequence of states ¶ x k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♦ and controls ¶ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ such that

<!-- formula-not-decoded -->

and the corresponding cost

<!-- formula-not-decoded -->

The rollout algorithm then applies the control that minimizes over u k ∈ U k ( x k ) the tail cost expression for stages k to N :

<!-- formula-not-decoded -->

For an intuitive justification of the cost improvement mechanism, note that the rollout control ˜ u k is calculated from Eq. (1.11) to attain the minimum over u k over the sum of two terms: the first stage cost g k (˜ x k ↪ u k ) plus the cost of the remaining stages ( k +1 to N ) using the heuristic controls. Thus rollout involves a first stage optimization (rather than just using the base heuristic), which accounts for the cost improvement. This reasoning also explains why multistep lookahead tends to provide better performance than one-step lookahead in rollout schemes.

XO

X1

Current State

•••

Xk

Next States

Xk+1

Xk+1

XN

Figure 1.2.8 Schematic illustration of rollout with one-step lookahead for a deterministic problem. At state x k , for every pair ( x k ↪ u k ), u k ∈ U k ( x k ), the base heuristic generates an approximate Q-factor

<!-- image -->

<!-- formula-not-decoded -->

and selects the control ˜ θ k ( x k ) with minimal Q-factor.

Equivalently, and more succinctly, the rollout algorithm applies at state x k the control ˜ θ k ( x k ) given by the minimization

<!-- formula-not-decoded -->

where ˜ Q k ( x k ↪ u k ) is the approximate Q-factor defined by see Fig. 1.2.8.

<!-- formula-not-decoded -->

Note that the rollout algorithm requires running the base heuristic for a number of times that is bounded by Nn , where n is an upper bound on the number of control choices available at each state. Thus if n is small relative to N , it requires computation equal to a small multiple of N times the computation time for a single application of the base heuristic. Similarly, if n is bounded by a polynomial in N , the ratio of the rollout algorithm computation time to the base heuristic computation time is a polynomial in N .

## Example 1.2.3 (Traveling Salesman Problem - Continued)

Let us consider the traveling salesman problem of Example 1.2.2, whereby a salesman wants to find a minimum cost tour that visits each of N given cities c = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 exactly once and returns to the city he started from. With each pair of distinct cities c , c ′ , we associate a traversal cost g ( c↪ c ′ ). Note

Heuristic

Initial City

Xo

X1

Next Partial

Tours

Next Cities

Current

Partial Tour

Xk

X k+1

7k+1

X'N

Figure 1.2.9 Rollout with the nearest neighbor heuristic for the traveling salesman problem of Example 1.2.3. The initial state x 0 consists of a single city. The final state x N is a complete tour of N cities, containing each city exactly once.

<!-- image -->

that we assume that we can go directly from every city to every other city. There is no loss of generality in doing so because we can assign a very high cost g ( c↪ c ′ ) to any pair of cities ( c↪ c ′ ) that is precluded from participation in the solution. The problem is to find a visit order that goes through each city exactly once and whose sum of costs is minimum.

/negationslash

There are many heuristic approaches for solving the traveling salesman problem. For illustration purposes, let us focus on the simple nearest neighbor heuristic, which starts with a partial tour, i.e., an ordered collection of distinct cities, and constructs a sequence of partial tours, adding to the each partial tour a new city that does not close a cycle and minimizes the cost of the enlargement. In particular, given a sequence ¶ c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ♦ (with k &lt; N -1) consisting of distinct cities, the nearest neighbor heuristic adds a city c k +1 that minimizes g ( c k ↪ c k +1 ) over all cities c k +1 = c 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k , thereby forming the sequence ¶ c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ↪ c k +1 ♦ . Continuing in this manner, the heuristic eventually forms a sequence of N cities, ¶ c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c N -1 ♦ , thus yielding a complete tour with cost

<!-- formula-not-decoded -->

/negationslash

We can formulate the traveling salesman problem as a DP problem as we discussed in Example 1.2.2. We choose a starting city, say c 0 , as the initial state x 0 . Each state x k corresponds to a partial tour ( c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ) consisting of distinct cities. The states x k +1 , next to x k , are sequences of the form ( c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ↪ c k +1 ) that correspond to adding one more unvisited city c k +1 = c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k (thus the unvisited cities are the feasible controls at a given partial tour/state). The terminal states x N are the complete tours of the form ( c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c N -1 ↪ c 0 ), and the cost of the corresponding sequence of city choices is the cost of the corresponding complete tour given by Eq. (1.14). Note that the number of states at stage k increases exponentially with k , and so does the computation required to solve the problem by exact DP.

Let us now use as a base heuristic the nearest neighbor method. The corresponding rollout algorithm operates as follows: After k &lt; N -1 iterations, we have a state x k , i.e., a sequence ¶ c 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ♦ consisting of distinct cities. At the next iteration, we add one more city by running the

Xk+1

Complete Tours

Nearest Neighbor

XN

Heuristic

A AB AC AD ABC ABD ACB ACD ADB ADC

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Matrix of Intercity Travel Costs

Figure 1.2.10 Rollout with the nearest neighbor base heuristic, applied to a traveling salesman problem. At city A, the nearest neighbor heuristic generates the tour ACDBA (labelled T 0 ). At city A, the rollout algorithm compares the tours ABCDA, ACDBA, and ADCBA, finds ABCDA (labelled T 1 ) to have the least cost, and moves to city B. At AB, the rollout algorithm compares the tours ABCDA and ABDCA, finds ABDCA (labelled T 2 ) to have the least cost, and moves to city D. The rollout algorithm then moves to cities C and A (it has no other choice). The final tour T 2 generated by rollout turns out to be optimal in this example, while the tour T 0 generated by the base heuristic is suboptimal. This is suggestive of a general result: the rollout algorithm for deterministic problems generates a sequence of solutions of decreasing cost under some conditions on the base heuristic that we will discuss in Chapter 2, and which are satisfied by the nearest neighbor heuristic.

<!-- image -->

/negationslash nearest neighbor heuristic starting from each of the sequences of the form ¶ c 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ↪ c ♦ where c = c 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k . We then select as next city c k +1 the city c that yielded the minimum cost tour under the nearest neighbor heuristic; see Fig. 1.2.9. The overall computation for the rollout solution is bounded by a polynomial in N , and is much smaller than the exact DP computation. Figure 1.2.10 provides an example where the nearest neighbor heuristic and the corresponding rollout algorithm are compared; see also Exercise 1.1.

A AB AC AD ABC ABD ACB ACD ADB ADC

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

## 1.3 STOCHASTIC EXACT AND APPROXIMATE DYNAMIC PROGRAMMING

We will now extend the DP algorithm and our discussion of approximation in value space to problems that involve stochastic uncertainty in their system equation and cost function. We will first discuss the finite horizon case, and the extension of the ideas underlying the principle of optimality and approximation in value space schemes. We will then consider the infinite horizon version of the problem, and provide an overview of the underlying theory and algorithmic methodology.

## 1.3.1 Finite Horizon Problems

The stochastic optimal control problem di ff ers from its deterministic counterpart primarily in the nature of the discrete-time dynamic system that governs the evolution of the state x k . This system includes a random 'disturbance' w k with a probability distribution P k ( · ♣ x k ↪ u k ) that may depend explicitly on x k and u k , but not on values of prior disturbances w k -1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w 0 . The system has the form

<!-- formula-not-decoded -->

where as earlier x k is an element of some state space, the control u k is an element of some control space. The cost per stage is denoted by g k ( x k ↪ u k ↪ w k ) and also depends on the random disturbance w k ; see Fig. 1.3.1. The control u k is constrained to take values in a given subset U k ( x k ), which depends on the current state x k .

Given an initial state x 0 and a policy π = ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ , the future states x k and disturbances w k are random variables with distributions defined through the system equation

<!-- formula-not-decoded -->

The discrete equation format and corresponding x -u -w notation is standard in the optimal control literature. For finite-state stochastic problems, also called Markovian Decision Problems (MDP), the system is often represented conveniently in terms of control-dependent transition probabilities. A common notation in the RL literature is p ( s↪ a↪ s ′ ) for transition probability from s to s ′ under action a . This type of notation is not well suited for deterministic problems, which involve no probabilistic structure at all and are of major interest in this book. The transition probability notation is also cumbersome for problems with a continuous state space; see Sections 1.7.1 and 1.7.2 for further discussion. The reader should note, however, that mathematically the system equation and transition probabilities are equivalent, and any analysis that can be done in one notational system can be translated to the other notational system.

Figure 1.3.1 Illustration of an N -stage stochastic optimal control problem. Starting from state x k , the next state under control u k is generated randomly, according to x k +1 = f k ( x k ↪ u k ↪ w k ) ↪ where w k is the random disturbance, and a random stage cost g k ( x k ↪ u k ↪ w k ) is incurred.

<!-- image -->

and the given distributions P k ( · ♣ x k ↪ u k ). Thus, for given functions g k , k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N , the expected cost of π starting at x 0 is

<!-- formula-not-decoded -->

where the expected value operation E ¶·♦ is taken with respect to the joint distribution of all the random variables w k and x k . An optimal policy π ∗ is one that minimizes this cost; i.e.,

<!-- formula-not-decoded -->

where Π is the set of all policies.

An important di ff erence from the deterministic case is that we optimize not over control sequences ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ [cf. Eq. (1.3)], but rather over policies (also called closed-loop control laws , or feedback policies ) that consist of a sequence of functions

<!-- formula-not-decoded -->

where θ k maps states x k into controls u k = θ k ( x k ), and satisfies the control constraints, i.e., is such that θ k ( x k ) ∈ U k ( x k ) for all x k . Policies are more general objects than control sequences, and in the presence of stochastic uncertainty, they can result in improved cost, since they allow choices of controls u k that incorporate knowledge of the state x k . Without this knowledge, the controller cannot adapt appropriately to unexpected values of the state, and as a result the cost can be adversely a ff ected. This is a fundamental distinction between deterministic and stochastic optimal control problems.

We assume an introductory probability background on the part of the reader. For an account that is consistent with our use of probability in this book, see the textbook by Bertsekas and Tsitsiklis [BeT08].

The optimal cost depends on x 0 and is denoted by J * ( x 0 ); i.e.,

<!-- formula-not-decoded -->

We view J * as a function that assigns to each initial state x 0 the optimal cost J * ( x 0 ), and call it the optimal cost function or optimal value function .

## Stochastic Dynamic Programming

The DP algorithm for the stochastic finite horizon optimal control problem has a similar form to its deterministic version, and shares several of its major characteristics:

- (a) Using tail subproblems to break down the minimization over multiple stages to single stage minimizations.
- (b) Generating backwards for all k and x k the values J * k ( x k ), which give the optimal cost-to-go starting from state x k at stage k .
- (c) Obtaining an optimal policy by minimization in the DP equations.
- (d) A structure that is suitable for approximation in value space, whereby we replace J * k by approximations ˜ J k , and obtain a suboptimal policy by the corresponding minimization.

## DP Algorithm for Stochastic Finite Horizon Problems

Start with

<!-- formula-not-decoded -->

and for k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, let

<!-- formula-not-decoded -->

For each x k and k , define θ ∗ k ( x k ) = u ∗ k where u ∗ k attains the minimum in the right side of this equation. Then, the policy π ∗ = ¶ θ ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ ∗ N -1 ♦ is optimal.

The key fact is that starting from any initial state x 0 , the optimal cost is equal to the number J * 0 ( x 0 ), obtained at the last step of the above DP algorithm. This can be proved by induction similar to the deterministic case; we will omit the proof (which incidentally involves some mathematical fine points; see the discussion of Section 1.3 in the textbook [Ber17a]).

Simultaneously with the o ff -line computation of the optimal costto-go functions J * 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ J * N , we can compute and store an optimal policy π ∗ = ¶ θ ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ ∗ N -1 ♦ by minimization in Eq. (1.15). We can then use this

policy on-line to retrieve from memory and apply the control θ ∗ k ( x k ) once we reach state x k . The alternative is to forego the storage of the policy π ∗ and to calculate the control θ ∗ k ( x k ) by executing the minimization (1.15) on-line.

There are a few favorable cases where the optimal cost-to-go functions J * k and the optimal policies θ ∗ k can be computed analytically using the stochastic DP algorithm. A prominent such case involves a linear system and a quadratic cost function, which is a fundamental problem in control theory. We illustrate the scalar version of this problem next. The analysis can be generalized to multidimensional systems (see optimal control textbooks such as [Ber17a]).

## Example 1.3.1 (Linear Quadratic Optimal Control)

Here the system is linear,

<!-- formula-not-decoded -->

and the state and control are scalars. Moreover, the disturbance w k has zero mean and given variance σ 2 . The cost is quadratic of the form:

<!-- formula-not-decoded -->

where q and r are known positive weighting parameters. We assume no constraints on x k and u k (in reality such problems include constraints, but it is common to neglect the constraints initially, and check whether they are seriously violated later).

As an illustration, consider a vehicle that moves on a straight-line road under the influence of a force u k and without friction. Our objective is to maintain the vehicle's velocity at a constant level ¯ v (as in an oversimplified cruise control system). The velocity v k at time k , after time discretization of its Newtonian dynamics and addition of stochastic noise, evolves according to

<!-- formula-not-decoded -->

where w k is a stochastic disturbance. By introducing x k = v k -¯ v , the deviation between the vehicle's velocity v k at time k from the desired level ¯ v , we obtain the system equation

<!-- formula-not-decoded -->

Here the coe ffi cient b relates to a number of problem characteristics including the weight of the vehicle and the road conditions. The cost function expresses our desire to keep x k near zero with relatively little force.

We will apply the DP algorithm, and derive the optimal cost-to-go functions J ∗ k and optimal policy. We have

<!-- formula-not-decoded -->

and by applying Eq. (1.15), we obtain

<!-- formula-not-decoded -->

and finally, using the assumptions E ¶ w N -1 ♦ = 0, E ¶ w 2 N -1 ♦ = σ 2 , and bringing out of the minimization the terms that do not depend on u N -1 ,

<!-- formula-not-decoded -->

The expression minimized over u N -1 in the preceding equation is convex quadratic in u N -1 , so by setting to zero its derivative with respect to u N -1 ,

<!-- formula-not-decoded -->

we obtain the optimal policy for the last stage:

<!-- formula-not-decoded -->

Substituting this expression into Eq. (1.17), we obtain with a straightforward calculation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

We can now continue the DP algorithm to obtain J ∗ N -2 from J ∗ N -1 . An important observation is that J ∗ N -1 is quadratic (plus an inconsequential constant term), so with a similar calculation we can derive θ ∗ N -2 and J ∗ N -2 in closed form, as a linear and a quadratic (plus constant) function of x N -2 , respectively. This process can be continued going backwards, and it can be verified by induction that for all k , we obtain the optimal policy and optimal cost-to-go function in the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

and the sequence ¶ K k ♦ is generated backwards by the equation

<!-- formula-not-decoded -->

starting from the terminal condition K N = qglyph[triangleright]

The process by which we obtained an analytical solution in this example is noteworthy. A little thought while tracing the steps of the algorithm will convince the reader that what simplifies the solution is the quadratic nature of the cost and the linearity of the system equation. Indeed, it can be shown in generality that when the system is linear and the cost is quadratic, the optimal policy and cost-to-go function are given by closed-form expressions, even for multi-dimensional linear systems (see [Ber17a], Section 3.1). The optimal policy is a linear function of the state, and the optimal cost function is a quadratic in the state plus a constant.

Another remarkable feature of this example, which can also be extended to multi-dimensional systems, is that the optimal policy does not depend on the variance of w k , and remains una ff ected when w k is replaced by its mean (which is zero in our example). This is known as certainty equivalence , and occurs in several types of problems involving a linear system and a quadratic cost; see [Ber17a], Sections 3.1 and 4.2. For example it holds even when w k has nonzero mean. For other problems, certainty equivalence can be used as a basis for problem approximation, e.g., assume that certainty equivalence holds (i.e., replace stochastic quantities by some typical values, such as their expected values) and apply exact DP to the resulting deterministic optimal control problem. This is an important part of the RL methodology, which we will discuss later in this chapter, and in more detail in Chapter 2.

Note that the linear quadratic type of problem illustrated in the preceding example is exceptional in that it admits an elegant analytical solution. Most DP problems encountered in practice require a computational solution.

## Q-Factors and Q-Learning for Stochastic Problems

Similar to the case of deterministic problems [cf. Eq. (1.9)], we can define optimal Q-factors for a stochastic problem, as the expressions that are minimized in the right-hand side of the stochastic DP equation (1.15). They are given by

<!-- formula-not-decoded -->

The optimal cost-to-go functions J * k can be recovered from the optimal Q-factors Q * k by means of

<!-- formula-not-decoded -->

and the DP algorithm can be written in terms of Q-factors as

<!-- formula-not-decoded -->

We will later be interested in approximate Q-factors, where J * k +1 in Eq. (1.20) is replaced by an approximation ˜ J k +1 . Again, the Q-factor corresponding to a state-control pair ( x k ↪ u k ) is the sum of the expected first stage cost using ( x k ↪ u k ), plus the expected cost of the remaining stages starting from the next state as estimated by the function ˜ J k +1 .

## 1.3.2 Approximation in Value Space for Stochastic DP

Generally the computation of the optimal cost-to-go functions J * k can be very time-consuming or impossible. One of the principal RL methods to deal with this di ffi culty is approximation in value space. Here approximations ˜ J k are used in place of J * k , similar to the deterministic case; cf. Eqs. (1.8) and (1.11).

## Approximation in Value Space - Use of ˜ J k in Place of J * k

At any state x k encountered at stage k , set

<!-- formula-not-decoded -->

The one-step lookahead minimization (1.21) needs to be performed only for the N states x 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N -1 that are encountered during the on-line control of the system. By contrast, exact DP requires that this type of minimization be done for every state and stage.

## The Three Approximations

When designing approximation in value space schemes, one may consider several interesting simplification ideas, which are aimed at alleviating the computational overhead. Aside from cost function approximation (use ˜ J k +1 in place of J * k +1 ), there are other possibilities. One of them is to simplify the lookahead minimization over u k ∈ U k ( x k ) [cf. Eq. (1.15)] by replacing U k ( x k ) with a suitably chosen subset of controls that are viewed as most promising based on some heuristic criterion.

In Section 1.6.7, we will discuss a related idea for control space simplification for the multiagent case where the control consists of multiple

components, u k = ( u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k ). Then, a sequence of m single component minimizations can be used instead, with potentially enormous computational savings resulting.

Another type of simplification relates to approximations in the computation of the expected value in Eq. (1.21) by using limited Monte Carlo simulation. The Monte Carlo Tree Search method, which will be discussed in Chapter 2, Section 2.7.5, is one possibility of this type.

Still another type of expected value simplification is based on the certainty equivalence approach , to be discussed in more detail in Chapter 2, Section 2.7.2. In this approach, at stage k , we replace the future random variables w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w k + m by some deterministic values w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w k + m , such as their expected values. We may also view this as a form of problem approximation, whereby for the purpose of computing ˜ J k +1 ( x k +1 ), we 'pretend' that the problem is deterministic, with the future random quantities replaced by deterministic typical values. This is one of the most e ff ective techniques to make approximation in value space for stochastic problems computationally tractable, particularly when it is combined with multistep lookahead minimization, as we will discuss later.

Figure 1.3.2 illustrates the three approximations involved in approximation in value space for stochastic problems: cost-to-go approximation, simplified minimization, and expected value approximation . They may be designed largely independently of each other, and may be implemented with a variety of methods. Much of the discussion in this book will revolve around di ff erent ways to organize these three approximations for both cases of one-step and multistep lookahead.

As indicated in Fig. 1.3.2, an important approach for cost-to-go approximation is problem approximation , whereby the functions ˜ J k +1 in Eq. (1.21) are obtained as the optimal or nearly optimal cost functions of a simplified optimization problem, which is more convenient for computation. Simplifications may include exploiting decomposable structure, ignoring various types of uncertainties, and reducing the size of the state space. Several types of problem approximation approaches are discussed in the author's RL book [Ber19a]. A major approach is aggregation , which will be discussed in Section 3.6. In this book, problem approximation will not receive much attention, despite the fact that it can often be combined very e ff ectively with the approximation in value space methodology that is our main focus.

Another important approach for on-line cost-to-go approximation is rollout, which we discuss next. This is similar to the rollout approach for deterministic problems, discussed in Section 1.2.

## Rollout for Stochastic Problems - Truncated Rollout

In the rollout approach, we select ˜ J k +1 in Eq. (1.21) to be the cost function of a suitable base policy (perhaps with some approximation). Note that