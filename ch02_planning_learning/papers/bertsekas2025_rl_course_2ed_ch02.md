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

1990s, indicate consistently good performance of rollout; see the last section of this chapter for a bibliography. The DP textbook [Ber17a] provides some detailed worked-out examples (Chapter 6, Examples 6.4.2, 6.4.5, 6.4.6, and Exercises 6.11, 6.14, 6.15, 6.16). The price for the performance improvement is extra computation that is typically equal to the computation time of the base heuristic times a factor that is a low order polynomial of N . It is generally hard to quantify the amount of performance improvement, but the computational results obtained from the case studies are consistent with the Newton step interpretations that we discussed in Chapter 1.

The books [Ber19a] (Section 2.5.1) and [Ber20a] (Section 3.1) show that the sequential improvement condition is satisfied in the context of MPC, and is the underlying reason for the stability properties of the MPC scheme. On the other hand the base heuristic underlying the classical form of the MPC scheme is not sequentially consistent (see the preceding references).

Generally, the sequential improvement condition may not hold for a given base heuristic. This is not surprising since any heuristic (no matter how inconsistent or silly) is in principle admissible to use as base heuristic. Here is an example:

## Example 2.3.3 (Sequential Improvement Violation)

Consider the 2-stage problem shown in Fig. 2.3.5, which involves two states at each of stages 1 and 2, and the controls shown. Suppose that the unique optimal trajectory is ( x 0 ↪ u ∗ 0 ↪ x ∗ 1 ↪ u ∗ 1 ↪ x ∗ 2 ), and that the base heuristic produces this optimal trajectory starting at x 0 . The rollout algorithm chooses a control at x 0 as follows: it runs the base heuristic to construct a trajectory starting from x ∗ 1 and ˜ x 1 , with corresponding costs H 1 ( x ∗ 1 ) and H 1 (˜ x 1 ). If

<!-- formula-not-decoded -->

the rollout algorithm rejects the optimal control u ∗ 0 in favor of the alternative control ˜ u 0 . The inequality above will occur if the base heuristic chooses ¯ u 1 at x ∗ 1 (there is nothing to prevent this from happening, since the base heuristic is arbitrary), and moreover the cost g 1 ( x ∗ 1 ↪ ¯ u 1 ) + g 2 (˜ x 2 ), which is equal to H 1 ( x ∗ 1 ) is high enough.

Let us also verify that if the inequality (2.15) holds then the heuristic is not sequentially improving at x 0 , i.e., that

<!-- formula-not-decoded -->

Indeed, this is true because H 0 ( x 0 ) is the optimal cost

<!-- formula-not-decoded -->

and must be smaller than both

<!-- formula-not-decoded -->

ne bast ce u1 :

Optimal Trajectory

P

к, Йк, Tk+1, Uk+

Nn"

1, IN

Chosen by Base Heuristic at xo

*5

N

High Cost Transition

Chosen by Heuristic at xi

й1

Optimal Trajectory Chosen by Base Heuristic at

High Cost Transition Chosen by Heuristic at

Violates Sequential Improvement 2.4.3, 2.4.4 2.4.2 3.3,

Violates Sequential Improvement 2.4.3, 2.4.4 2.4.2 3.3,

<!-- image -->

Rollout Choice

Figure 2.3.5 A 2-stage problem with states x ∗ 1 ↪ ˜ x 1 at stage 1, and states x ∗ 2 ↪ ˜ x 2 at stage 2. The controls and corresponding transitions are as shown in the figure. The rollout choice at the initial state x 0 is strictly suboptimal, while the base heuristic choice is optimal. The reason is that the base heuristic is not sequentially improving and makes the suboptimal choice u 1 at x ∗ 1 , but makes the di ff erent (optimal) choice u ∗ 1 when run from x 0 .

which is the cost of the trajectory ( x 0 ↪ u ∗ 0 ↪ x ∗ 1 ↪ u 1 ↪ ˜ x 2 ), and

<!-- formula-not-decoded -->

which is the cost of the trajectory ( x 0 ↪ ˜ u 0 ↪ ˜ x 1 ↪ ˜ u 1 ↪ ˜ x 2 ).

The preceding example and the monotonicity property (2.14) suggest a simple enhancement to the rollout algorithm, which detects when the sequential improvement condition is violated and takes corrective measures. In this algorithmic variant, called fortified rollout , we maintain the best trajectory obtained so far, and keep following that trajectory up to the point where we discover another trajectory that has improved cost (see the next section).

## 2.3.2 The Fortified Rollout Algorithm

In this section we describe a rollout variant that implicitly enforces the sequential improvement property. This variant, called the fortified rollout algorithm , starts at x 0 , and generates step-by-step a sequence of states ¶ x 0 ↪ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♦ and corresponding sequence of controls. Upon reaching state x k we have the trajectory

<!-- formula-not-decoded -->

that has been constructed by rollout, called permanent trajectory , and we also store a tentative best trajectory

<!-- formula-not-decoded -->

Rollout

Choice

Il end- trajectory and

every

к, Йк)

algori to Tr.

., UN-1, IN.

of Tk,

Uk)

le perr d (Uk, Fk+1) to

It (It, Ut)

NIN

v. To se Tr.

inchanged: Tk+

Pk

Tk+

= Tk.

C(T1)

XO

Tentative Best Trajectory Tk

Xk+1

Current State

Xk

йк

Permanent trajectory P

Min Q-factor choice

Figure 2.3.6 Schematic illustration of fortified rollout. After k steps, we have constructed the permanent trajectory

<!-- image -->

<!-- formula-not-decoded -->

and the tentative best trajectory

<!-- formula-not-decoded -->

the best end-to-end trajectory computed so far. We now run the rollout algorithm at x k , i.e., we find the control ˜ u k that minimizes over u k the sum of g k ( x k ↪ u k ) plus the heuristic cost from the state x k +1 = f k ( x k ↪ u k ), and the corresponding trajectory

<!-- formula-not-decoded -->

If the cost of the end-to-end trajectory ˜ T k is lower than the cost of T k , we add (˜ u k ↪ ˜ x k +1 ) to the permanent trajectory and set the tentative best trajectory to T k +1 = ˜ T k . Otherwise we add ( u k ↪ x k +1 ) to the permanent trajectory and keep the tentative best trajectory unchanged: T k +1 = T k glyph[triangleright]

with corresponding cost

<!-- formula-not-decoded -->

The tentative best trajectory T k is the end-to-end trajectory that has minimum cost out of all end-to-end trajectories computed up to stage k of the algorithm. Initially, T 0 is the trajectory generated by the base heuristic starting at the initial state x 0 . The idea now is to discard the suggestion of the rollout algorithm at every state x k where it produces a trajectory that is inferior to T k , and use T k instead (see Fig. 2.3.6).

In particular, upon reaching state x k , we run the rollout algorithm as earlier, i.e., for every u k ∈ U k ( x k ) and next state x k +1 = f k ( x k ↪ u k ) ↪ we

Heuristic

run the base heuristic from x k +1 , and find the control ˜ u k that gives the best trajectory, denoted

<!-- formula-not-decoded -->

with corresponding cost

<!-- formula-not-decoded -->

Whereas the ordinary rollout algorithm would choose control ˜ u k and move to ˜ x k +1 , the fortified algorithm compares C ( T k ) and C ( ˜ T k ), and depending on which of the two is smaller, chooses u k or ˜ u k and moves to x k +1 or to ˜ x k +1 , respectively. In particular, if

<!-- formula-not-decoded -->

the algorithm sets the next state and corresponding tentative best trajectory to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

it sets the next state and corresponding tentative best trajectory to

<!-- formula-not-decoded -->

In other words the fortified rollout at x k follows the current tentative best trajectory T k unless a lower cost trajectory ˜ T k is discovered by running the base heuristic from all possible next states x k +1 . It follows that at every state the tentative best trajectory has no larger cost than the initial tentative best trajectory, which is the one produced by the base heuristic starting from x 0 . Moreover, it can be seen that if the base heuristic is sequentially improving, the rollout algorithm and its fortified version coincide. Experimental evidence suggests that it is often important to use the fortified version if the base heuristic is not known to be sequentially improving. Fortunately, the fortified version involves hardly any additional computational cost.

As expected, when the base heuristic generates an optimal trajectory, the fortified rollout algorithm will also generate the same trajectory. This is illustrated by the following example.

The base heuristic may also be run from a subset of the possible next states x k +1 , as in the case where a simplified version of rollout is used; cf. Section 2.3.4. Then fortified rollout will still guarantee a cost improvement property.

and if

## Example 2.3.4

Let us consider the application of the fortified rollout algorithm to the problem of Example 2.3.3 and see how it addresses the issue of cost improvement. The fortified rollout algorithm stores as initial tentative best trajectory the optimal trajectory ( x 0 ↪ u ∗ 0 ↪ x ∗ 1 ↪ u ∗ 1 ↪ x ∗ 2 ) generated by the base heuristic at x 0 . Then, starting at x 0 , it runs the heuristic from x ∗ 1 and ˜ x 1 , and (despite the fact that the ordinary rollout algorithm prefers going to ˜ x 1 rather than x ∗ 1 ) it discards the control ˜ u 0 in favor of u ∗ 0 , which is dictated by the tentative best trajectory. It then sets the tentative best trajectory to ( x 0 ↪ u ∗ 0 ↪ x ∗ 1 ↪ u ∗ 1 ↪ x ∗ 2 ).

We finally note that the fortified rollout algorithm can be used in a di ff erent setting to restore and maintain the cost improvement property. Suppose in particular that the rollout minimization at each step is performed with approximations. For example the control u k may have multiple independently constrained components, i.e.,

<!-- formula-not-decoded -->

Then, to take advantage of distributed computation, it may be attractive to decompose the optimization over u k in the rollout algorithm,

<!-- formula-not-decoded -->

into an (approximate) parallel optimization over the components u i k (or subgroups of these components). However, as a result of approximate optimization over u k , the cost improvement property may be degraded, even if the sequential improvement assumption holds. In this case by maintaining the tentative best trajectory, starting with the one produced by the base heuristic at the initial condition, we can ensure that the fortified rollout algorithm, even with approximate minimization, will not produce an inferior solution to the one of the base heuristic.

## 2.3.3 Using Multiple Base Heuristics - Parallel Rollout

In many problems, several promising heuristics may be available. It is then possible to use all of these heuristics in the rollout framework. The idea is to construct a superheuristic , which selects the best out of the trajectories produced by the entire collection of heuristics. The superheuristic can then be used as the base heuristic for a rollout algorithm.

A related practically interesting possibility is to introduce a partition of the state space into subsets, and a collection of multiple heuristics that are specially tailored to the subsets. We may then select the appropriate heuristic to use on each subset of the partition. In fact one may use a collection of multiple heuristics tailored to each subset of the state space partition, and at each state, select out of all the heuristics that apply, the one that yields minimum cost.

X1

Current State

•••

States at

Time N

Heuristic 1

Heuristic 2

Next States

Xk+1

Uk

Heuristic 3

Minimal Q-Factor

<!-- image -->

) States at Time

Heuristic 1 Heuristic 2 Heuristic 3

Heuristic 1 Heuristic 2 Heuristic 3

Figure 2.3.7 Schematic illustration of parallel rollout. From every possible next state x k +1 , we run each of the heuristics, we compute the Q-factor corresponding to each possible control u k and heuristic, and select the control with minimal Qfactor. Thus the number of Q-factors that are compared is equal to m · h , where m is the number of controls available at state x k and h is the number of heuristics used in the parallel rollout. In the figure, there are six Q-factors to compare, and the best/minimal Q-factor corresponds to the pair ( u ′ k ↪ Heuristic 1), as indicated, thus leading to selection of u ′ k as the rollout control at x k .

In particular, let us assume that we have m heuristics, and that the /lscript th of these, given a state x k +1 , produces a trajectory

<!-- formula-not-decoded -->

and corresponding cost C ( ˜ T /lscript k +1 ). The superheuristic then produces at x k +1 the trajectory ˜ T /lscript k +1 for which C ( ˜ T /lscript k +1 ) is minimum. The rollout algorithm selects at state x k the control u k that minimizes the minimal Q-factor:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is the cost of the trajectory ( x k ↪ u k ↪ ˜ T /lscript k +1 ). Note that the Q-factors of the di ff erent heuristics can be computed independently and in parallel. In view of this fact, the rollout scheme just described is sometimes referred to as parallel rollout ; see Fig. 2.3.7.

An interesting property, which can be readily verified by using the definitions, is that if all the heuristics are sequentially improving, the same is true for the superheuristic , something that is also suggested by the proof of the cost improvement property of Fig. 2.3.4. Indeed, let us write the sequential improvement condition (2.13) for each of the base heuristics

<!-- formula-not-decoded -->

l=1

l=1

., m

..., m

XO

Cont

X1

Current State

.. •

Xk

States at

Time N

Trajectory 1

Trajectory 2

Next States

Xk+1

Uk

Trajectory 3

Minimal Q-Factor

<!-- image -->

) States at Time

Trajectory 1 Trajectory 2 Trajectory 3

Trajectory 1 Trajectory 2 Trajectory 3

Figure 2.3.8 Schematic illustration of general rollout. From every possible next state x k +1 , we run a number of trajectories, we compute the Q-factor corresponding to all the possible (control, trajectory) pairs, and select the control with minimal Q-factor. The number of Q-factors compared is at most m · h , where m is the number of controls available at state x k and h is the maximum number of trajectories generated from each possible next state x k +1 . In the figure, there are five Q-factors to compare, and the best/minimal Q-factor corresponds to the pair ( u ′ k ↪ Trajectory 1), as indicated, thus leading to selection of u ′ k as the rollout control at x k .

where ˜ Q /lscript k ( x k ↪ u k ) and H /lscript k ( x k ) are Q-factors and heuristic costs that correspond to the /lscript th heuristic. Then by taking minimum over /lscript , we have

<!-- formula-not-decoded -->

for all x k and k . By interchanging the order of the minimizations of the left side, we then obtain

<!-- formula-not-decoded -->

which is precisely the sequential improvement condition (2.13) for the superheuristic.

## 2.3.4 Using Multiple Rollout Trajectories - General Rollout

A generalization of the parallel rollout scheme, called general rollout , involves the generation of multiple trajectories starting from each possible next state x k +1 (see Fig. 2.3.8). Here, similar to parallel rollout, for each possible x k +1 , we compute multiple Q-factors, but we do not require that

each of these Q-factors correspond to a distinct base heuristic. Instead the Q-factors may be associated with multiple trajectories, however generated, which start at x k +1 and end at the terminal time N .

Note that some of the trajectories that start from a state x k +1 may share some intermediate states from times k +2 to N , and they may also be randomly generated. As a result the general rollout framework is well suited for the use of large language models (LLM) as base heuristics. The reason is that the trajectories generated by LLM often involve randomizations and may also involve suboptimization over multiple trajectories. For example, LLM may consider multiple next word choices at selected states, and generate multiple trajectories corresponding to each of these choices.

Similar to parallel rollout, general rollout is not di ff erent conceptually from the basic rollout algorithm with a single superheuristic that involves optimization over multiple trajectories starting at any possible next state x k +1 (cf. Section 2.3.3). One possibility in general rollout is to carry out this optimization by DP, over a tree of trajectories that is rooted at x k +1 . We finally note an important point: with multiple trajectories, the verification of properties such as sequential consistency and sequential improvement, may become complicated. Thus the use of rollout fortification (cf. Section 2.3.2) may be essential to guarantee a cost improvement property.

## 2.3.5 Simplified Rollout Algorithms

We will now consider a rollout variant, called simplified rollout , which is motivated by problems where the control constraint set U k ( x k ) is either infinite or finite but very large. Then the minimization

<!-- formula-not-decoded -->

[cf. Eqs. (2.9) and (2.10)], may be unwieldy, since the number of Q-factors

<!-- formula-not-decoded -->

is accordingly infinite or large.

To remedy this situation, we may replace U k ( x k ) with a smaller finite subset U k ( x k ):

<!-- formula-not-decoded -->

The rollout control ˜ θ k ( x k ) in this variant is one that attains the minimum of ˜ Q k ( x k ↪ u k ) over u k ∈ U k ( x k ):

<!-- formula-not-decoded -->

An example is when U k ( x k ) results from discretization of an infinite set U k ( x k ). Another possibility is when by using some preliminary approximate optimization, perhaps using a trained neural network or other heuristic method, we can identify a subset U k ( x k ) of promising controls, and to

save computation, we restrict attention to this subset. A related possibility is to generate U k ( x k ) by some iterative or random search method that explores intelligently the set U k ( x k ) with the aim to minimize ˜ Q k ( x k ↪ u k ) [cf. Eq. (2.16)].

It turns out that the proof of the cost improvement property of Prop. 2.3.2,

<!-- formula-not-decoded -->

goes through if the following modified sequential improvement property holds:

<!-- formula-not-decoded -->

This can be seen by verifying that Eq. (2.18) is su ffi cient to guarantee that the monotone improvement Eq. (2.14) is satisfied. The condition (2.18) is very simple to satisfy if the base heuristic is sequentially consistent, in which case the control u k selected by the base heuristic satisfies

<!-- formula-not-decoded -->

In particular, for the property (2.18) to hold, it is su ffi cient that U k ( x k ) contains the base heuristic choice u k .

The idea of replacing the minimization (2.16) by the simpler minimization (2.17) can be extended. In particular, by working through the preceding argument, it can be seen that any policy

<!-- formula-not-decoded -->

such that ˜ θ k ( x k ) satisfies the condition

<!-- formula-not-decoded -->

for all x k and k , guarantees the modified sequential improvement property (2.18), and hence also the cost improvement property . A prominent example of such an algorithm arises in the multiagent case where u has m components, u = ( u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ), and the minimization over U 1 k ( x k ) × · · · × U m k ( x k ) is replaced by a sequence of single component minimizations, one-componentat-a-time; cf. Section 1.6.7. Of course in the multiagent case, the onecomponent-at-a-time implementation has an additional favorable property: it can be viewed as rollout (without simplification) for a modified but equivalent DP problem (see Section 1.6.7).

## 2.3.6 Truncated Rollout with Terminal Cost Approximation

An important variation of rollout algorithms is truncated rollout with terminal cost approximation. Here the rollout trajectories are obtained by running the base policy from the leaf nodes of the lookahead tree, but they

are truncated after a given number of steps, while a terminal cost approximation is added to the heuristic cost to compensate for the resulting error. This is important for problems with a large number of stages, and it is also essential for infinite horizon problems where the rollout trajectories have infinite length.

One possibility that works well for many problems is to simply set the terminal cost approximation to zero. Alternatively, the terminal cost function approximation may be obtained by using some sophisticated o ff -line training process that may involve an approximation architecture such as a neural network, or by using some heuristic calculation based on a simplified version of the problem. This form of truncated rollout may also be viewed as an intermediate approach between standard rollout where there is no truncation (and hence no cost function approximation), and approximation in value space without any rollout.

## 2.3.7 Rollout with an Expert - Model-Free Rollout

We will now consider a rollout algorithm for discrete deterministic optimization for the case where we do not know the cost function and the constraints of the problem . Instead we have access to a base heuristic, and also a human or software 'expert' who can rank any two feasible solutions without assigning numerical values to them.

We consider the general discrete optimization problem of selecting a control sequence u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ) to minimize a function G ( u ). For simplicity we assume that each component u k is constrained to lie in a given constraint set U k , but extensions to more general constraint sets are possible. We assume the following:

- (a) A base heuristic with the following property is available: Given any k &lt; N -1, and a partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), it generates, for every ˜ u k +1 ∈ U k +1 , a complete feasible solution by concatenating the given partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ) with a sequence (˜ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u N -1 ). This complete feasible solution is denoted

<!-- formula-not-decoded -->

The base heuristic is also used to start the algorithm from an artificial empty solution, by generating all components ˜ u 0 ∈ U 0 and a complete feasible solution (˜ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u N -1 ), starting from each ˜ u 0 ∈ U 0 .

- (b) An 'expert' is available that can compare any two feasible solutions u and u , in the sense that he/she can determine whether

<!-- formula-not-decoded -->

It can be seen that deterministic rollout can be applied to this problem, even though the cost function G is unknown. The reason is that the

Base

Heuristic

Complete

Solutions www

Expert Ranks Complete Solutions

Sk U0, ..., Ик, k+1), k+1 € Uk+1

Base Heuristic Expert Ranks Complete Solutions

Base Heuristic Expert Ranks Complete Solutions www

Figure 2.3.9 Schematic illustration of the rollout with an expert for minimizing G ( u ) subject to

<!-- image -->

<!-- formula-not-decoded -->

We assume that we do not know G and/or U 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ U N -1 . Instead we have a base heuristic, which given a partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), outputs all next controls ˜ u k +1 ∈ U k +1 , and generates from each a complete solution

<!-- formula-not-decoded -->

Also, we have a human or software 'expert' that can rank any two complete solutions without assigning numerical values to them. The control that is selected from U k +1 by the rollout algorithm is the one whose corresponding complete solution is ranked best by the expert.

rollout algorithm uses the cost function only as a means of ranking complete solutions in terms of their cost. Hence, if the ranking of any two solutions can be revealed by the expert, this is all that is needed. In fact, the constraint sets U 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ U N -1 need not be known either, as long as they can be generated by the base heuristic. Thus, the rollout algorithm can be described as follows (see Fig. 2.3.9):

We start with an artificial empty solution, and at the typical step, given the partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), k &lt; N -1, we use the base heuristic

Note that for this to be true, it is important that the problem is deterministic, and that the expert ranks solutions using some underlying (though unknown) cost function. In particular, the expert's rankings should have a transitivity property: if u is ranked better than u ′ and u ′ is ranked better than u ′′ , then u is ranked better than u ′′ .

Current Partial Solution glyph[triangleright] ↪

to generate all possible one-step-extended solutions

<!-- formula-not-decoded -->

and the set of complete solutions

<!-- formula-not-decoded -->

We then use the expert to rank this set of complete solutions. Finally, we select the component u k +1 that is ranked best by the expert, extend the partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ) by adding u k +1 , and repeat with the new partial solution ( u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ↪ u k +1 ).

Except for the (mathematically inconsequential) use of an expert rather than a cost function, the preceding rollout algorithm can be viewed as a special case of the one given earlier. As a result several of the rollout variants that we have discussed so far (rollout with multiple heuristics, simplified rollout, and fortified rollout) can also be easily adapted.

## Example 2.3.5 (Active Learning - Dataset Enrichment)

Let us consider a machine learning context whereby we have a dataset that we want to enrich with data selected from another dataset. The enrichment is to be done sequentially, one data point at a time, up to a given maximum size of N data points. We are interested in finding a sequential enrichment policy that optimizes some terminal cost function. This is the active learning context that we discussed briefly in Section 1.7.5.

We can approach the problem in terms of the discrete optimization framework of this section. The state is a partial data set ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), consisting of the initial data set, denoted u 0 , and the k additional data points u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k that have been selected up to time k . The current state ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ) is augmented with a data point u k +1 from the set of the remaining (unused) data points, and each possible new state is evaluated using a base heuristic and an expert, according to our framework of this section. An interesting variation here is to allow the option for an early termination of the dataset enrichment process; this can be done by allowing a dummy/empty data point selection at each time. An additional possibility for early termination is to introduce a cost for each new data point addition, so that the dataset enrichment process stops naturally, when there is little cost improvement.

## Example 2.3.6 (Using a Large Language Model for Rollout)

The problem of minimizing G ( u ) over a constraint set can be viewed as an N -gram optimization problem, discussed in Example 1.6.2, where the text window consists of a partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ↪ u k +1 ) (preceded by N -k -2 'default' words to bring the total to N ). We noted in that example that a GPT can be used as a policy that generates next words within the context of N -gram optimization. Thus the problem can be addressed within the model-free rollout framework of this section, whereby a GPT is used as a

Partial Folding Software Critic Software

Complete Folding Current Partial Folding

Partial Folding Software Critic Software

Figure 2.3.10 Schematic illustration of rollout for the RNA folding problem. The current state is the partial folding depicted on the left side. There are at most three choices for control at each state.

<!-- image -->

base heuristic for completion of partial solutions. The main issues with this approach are how to train a GPT for the problem at hand, and also in the absence of an explicit cost function G ( u ), how to properly design the expert software for comparing complete solutions. Both of these issues are actively researched at present.

## Example 2.3.7 (RNA Folding)

In a classical problem from computational biology, we are given a sequence of nucleotides, represented by circles in Fig. 2.3.10, and we want to 'fold' the sequence in an 'interesting' way (introduce pairings of nucleotides that result in an 'interesting' structure). There are some constraints on which pairings are possible, but we will not go into the details of this (some types of constraints may require the use of the constrained rollout framework of Section 2.5). A common constraint is that the pairings should not 'cross,' i.e., given a pairing ( i 1 ↪ i 2 ) there should be no pairing ( i 3 ↪ i 4 ) where either i 3 &lt; i 1 and i 1 &lt; i 4 &lt; i 2 , or i 1 &lt; i 3 &lt; i 2 and i 2 &lt; i 4 . This type of problem has a long history of solution by DP, starting with the paper by Zuker and Stiegler [ZuS81]. There are several formulations, where the aim is to optimize some criterion, e.g., the number of pairings, or the 'energy' of the folding. However, biologists do not agree on a suitable criterion, and have developed software to generate 'reasonable' foldings, based on semi-heuristic reasoning. We will develop a rollout approach that makes use of such software without discussing their underlying principles.

We formulate the folding problem as a discrete optimization problem involving a pairing decision at each nucleotide in the sequence with at most three choices (open a pairing, close a pairing, do nothing); see Fig. 2.3.10. To apply rollout, we need a base heuristic, which given a partial folding, generates a complete folding (this is the partial folding software shown in Fig.

Expert Rollout with Base O

Value iterations Compares Complete Foldings

2.3.10). Two complete foldings can be compared by some other software, called the expert software . An interesting aspect of this problem is that there is no explicit cost function here (it is internal to the expert software). Thus by trying di ff erent partial folding and expert software, we may obtain multiple solutions, which may be used for further screening and/or experimental evaluation. For a recent implementation and variations, see Liu et al. [LPS21].

One more aspect of the problem that is worth noting is that there are at most three choices for control at each state, while the problem is deterministic. As a result, the problem is a good candidate for the use of multistep lookahead. In particular, with /lscript -step lookahead, the number of Q-factors to be computed at each state increases from 3 (or less) to 3 /lscript (or less).

## Perpetual Rollout with an Expert

We have considered so far a rollout algorithm that makes a single pass through the variables u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 in a fixed order. However, it is clear that any order of the variables is possible, as long as the base heuristic can deal with that order. In an o ff -line setting, where rollout is used simply as a suboptimal method to minimize G ( u ), this allows the possibility of repeating the rollout with multiple di ff erent orders. In particular, the final feasible solution obtained through each rollout pass can be used as the initial base heuristic solution for the next rollout pass. If the fortified version of rollout is used, then the algorithm produces a sequence of solutions with nonincreasing cost, and terminates when the set of selected variable orders is exhausted without any cost improvement.

## Learning to Imitate the Expert

To implement model-free rollout, we need both a base heuristic and an expert. None of these may be readily available, particularly the expert, which may involve a hidden cost function that is implicitly used to rank complete solutions. Within this context, it is worth considering the case where an expert is not available but can be emulated by training with the use of data. In particular, suppose that we are given a set of control sequence pairs ( u s ↪ u s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , with

<!-- formula-not-decoded -->

which we can use for training. Such a set may be obtained in a variety of ways, including querying the expert. We may then train a parametric approximation architecture such as a neural network to produce a function ˜ G ( u↪ r ), where r is a parameter vector, and use this function in place of the unknown G ( u ) to implement the preceding rollout algorithm.

A method, known as comparison training , has been suggested for this purpose, and has been used in a variety of game contexts, including backgammon and chess by Tesauro [Tes89b], [Tes01]. Briefly, given

the training set of pairs ( u s ↪ u s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , which satisfy Eq. (2.19), we generate for each ( u s ↪ u s ), two solution-cost pairs ( u s ↪ 1) ↪ ( u s ↪ -1). A parametric architecture ˜ G ( · ↪ r ), involving a parameter vector r , such as a neural network, is then trained by some form of regression with these data to produce an approximation ˜ G ( · ↪ ¯ r ) to be used in place of G ( · ) in a rollout scheme. There is also a related methodology, known as inverse reinforcement learning , which aims to use an expert's history of choices to learn the expert's cost function; see the RL survey by Arora and Doshi [ArD21], and the influential early work in econometrics by Rust [Rus88].

## Learning the Base Policy's Q-Factors

In another type of imitation approach, we view the base policy decisions as being selected by a process the mechanics of which are not observed except through its generated cost samples at the various stages. In particular, the stage costs starting from any given partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ) are added to form samples of the base policy's Q-factors Q k ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ). In this way we can obtain Q-factor samples starting from many partial solutions ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ). Moreover, a single complete solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ) generated by the base policy provides multiple Q-factor samples, one for each of the partial solutions ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ).

We can then use the sample (partial solution, cost) pairs in conjunction with a training method (see Chapter 3) in order to construct parametric approximations ˜ Q k ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ↪ r k ), k = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N↪ to the true Q-factors Q k ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), where r k is the parameter vector. Once the training has been completed and the Q-factors ˜ Q k ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ↪ r k ) have been obtained for all k , we can construct complete solutions step-by-step, by selecting the next component ˜ u k +1 , given the partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), through the minimization

<!-- formula-not-decoded -->

Note that even though we are 'learning' the base policy, our aim is not to imitate it, but rather to generate a rollout policy. The latter policy will make better decisions than the base policy, thanks to the cost improvement property of rollout. This points to an important issue of exploration : we must ensure that the training set of sample (partial solution, cost) pairs is broadly representative, in the sense that it is not unduly biased towards sample pairs that are generated by the base policy.

## 2.3.8 Using a World Model for Rollout

We now consider another optimal control problem context that bears some similarity with the model-free rollout methodology of the preceding section.

There is a true system and a cost per stage,

<!-- formula-not-decoded -->

but the controller does not have access to their mathematical description, so the system equation f k and cost function g k are unknown. Still, however, we assume that at time k , the controller can observe the current state x k , and can predict all the possible next states x k +1 and the corresponding one-stage costs resulting from the possible control choices u k . In other words, upon arrival to state x k , the controller can calculate all the triplets of the form

<!-- formula-not-decoded -->

corresponding to x k [only these one-step triplets, and not full access to the true model (2.20)]. These local one-step triplet predictions may take a variety of forms in practical contexts. For example in robotics, the robot can typically predict accurately the immediate outcome of small control actions using its sensors, but its predictions about future environments may be much less precise. A similar situation occurs in many other contexts, such as autonomous driving, healthcare treatment planning, supply chain systems, financial trading and execution, etc.

The idea is that, by using data triplets from past interactions with the system or by pretraining a neural network or transformer, the controller can learn an approximate predictive model of system evolution and stage costs, possibly in a simplified or implicit representation:

<!-- formula-not-decoded -->

This type of learned model is an instance of what is called a world model in contemporary AI terminology, (starting with the paper by Ha and Schmidhuber [HaS18]; see e.g., Hafner et al. [HLB19], and Moerland et al. [MBP23]), specialized here to a control-theoretic setting with explicit state and cost representations. It can be used as a computational surrogate for the true model for the purposes of rollout in conjunction with a given base policy.

In particular, at each state x k , and given the possible next states upon arrival [cf. Eq. (2.21)], the world model (2.22) is used to construct system trajectories starting from each candidate next state. These trajectories are then used to compute the associated Q-factors under a given base policy, consistently with the approximation in value space framework of this chapter: the control that corresponds to the minimal Q-factor is applied at time k . The resulting next state is then observed and the process is repeated.

Note that there is no guarantee of cost improvement over the base policy, unless the exact model (2.20) and world model (2.22) act identically at x k . Nevertheless, in a sequential decision framework, state observations

from the real system can compensate for world model prediction inaccuracies, consistent with the replanning character of approximation in value space and model predictive control (cf. Section 1.6.9).

In particular, our Newton step interpretation of approximation in value space, and the associated superlinear error bound guarantees apply; cf. our discussion of Section 1.5.3. As we have discussed earlier, good performance relies in an accurate model for the first step of the lookahead optimization, while the cost function approximation of the subsequent steps can be fairly inaccurate, subject to staying within the region of convergence of the Newton step; cf. the discussion of Section 1.5.3.

We finally note that with appropriate modifications, a world model can still be used for approximate rollout in a stochastic problem setting, with system and cost per stage

<!-- formula-not-decoded -->

where f k and g k are unknown. What is necessary here is that at the current state x k , the one-step triplets

<!-- formula-not-decoded -->

are known to the controller, together with the probability distribution of w k . A world model can then be used for approximate rollout by predicting trajectories starting from the next states. The world model may be simplified; for example it may be deterministic, based on the use of certainty equivalence (see our discussion of MPC in Section 1.6.9).

## 2.3.9 Local Search with Rollout for Discrete Optimization

We will now introduce a di ff erent rollout approach to the solution of the discrete optimization problem of Section 2.3.7:

<!-- formula-not-decoded -->

where U is a given finite set of feasible solutions. In particular, we will adopt a perspective, which aligns with local search methods, such as those based on arc exchanges for scheduling and traveling salesman problems, tabu search, genetic algorithms, and others (see Ch. 10 of the author's network optimization book [Ber98], which contains an accessible account). These methods are iterative, and generate a sequence of solutions ¶ u k ♦ ⊂ U , starting from some initial u 0 . Note that the points u k are feasible solutions (not partial solutions), and indeed u need not have the N -component structure u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ) of Section 2.3.7.

The general form of an iterative local search method is as follows: At the current iterate u it generates (in some unspecified way) a subset

Next States/Tours u'

Current State/Tour

и'

Selected State û

û E ar®

Next States/Tours Current State/Tour

Figure 2.3.11 Schematic representation of a greedy base heuristic, such as k -OPT, for the traveling salesman problem. The current tour is u , the possible next tours u ′ ∈ B ( u ), where B ( u ) is the neighborhood of u , and the next tour selected by the base heuristic is

<!-- image -->

<!-- formula-not-decoded -->

B ( u ) ⊂ U , called the neighborhood of u , it evaluates the cost G ( u ) for each point of B ( u ), and it moves to the point ˆ u ∈ B ( u ) that has minimal cost. It then replaces u with ˆ u , and continues up to when there is no further progress. We assume that each u is contained within its neighborhood, i.e., u ∈ B ( u ) for all u ∈ U ; this guarantees that there is no cost deterioration at any iteration, so the method terminates (some local search methods, like tabu search, allow a limited amount of cost deterioration, and introduce a suitable termination mechanism). Note the di ff erence from the method of Section 2.3.7. Here, we are generating iteratively a sequence of complete solutions (not partial solutions).

An important point here is that a local search method of the type just described can be viewed as a dynamic system. The state is u , the control at u is the selection of a point u ′ ∈ B ( u ), the set of next states is B ( u ), and the system function generates the next state ˆ u according to the selection policy. Finally, once we choose a finite horizon for the corresponding optimal control problem, the cost function is G ( u T ), where u T is the terminal state.

## Example 2.3.8 (Traveling Salesman Problem)

For an example of a local search method, and the dynamic system and optimal control problem that it represents, consider the N -city traveling salesman problem that we discussed in Example 1.2.2. Here the states are the complete tours u (cycles that contain all of the N cities exactly once). In a popular approach, a neighborhood of a tour u is defined as the set B ( u ) of all tours that can be obtained from u by exchanging k arcs that belong to u with another k arcs that do not belong to u (where k is some positive integer). This neighborhood defines a system equation where the decision/control variable min

G(u')

consists of the two sets of k arcs that are being exchanged. Such a transition from the current state/tour to the next state/tour is called a k -interchange . A choice, known as the k -OPT heuristic , which has been successful in practice, is a greedy heuristic, whereby the two sets of k arcs are chosen to optimize the cost of the new tour; see Fig. 2.3.11. The method stops when no improvement of the current tour is possible through a k -interchange.

The k -OPT heuristic or similar heuristics can also be used to define a rollout algorithm, which improves the cost of the final tour, based on our earlier analysis. On the other hand, this requires extra computation, which is potentially significant (an increase by a factor proportional to N 2 ). A truncated rollout approach, coupled with parallel computation, can reduce this extra computation drastically.

There are other possible methods to construct tour neigborhoods and base heuristics for the traveling salesman problem and its variants, including some that are defined with the assistance of trained neural networks or large language models.

Here is another example of local search in the context of prompting a pretrained large language model (LLM):

## Example 2.3.9 (Context Adaptation as Local Search in LLM)

Let us discuss how our local search and rollout framework applies to a class of optimization problems arising in artificial intelligence systems based on LLMs, originally addressed by Zhang et al. [ZHU25]. Here the object being optimized is not a direct solution to a combinatorial problem (such as a tour or a schedule), but rather a context consisting of system instructions (prompts) to the LLM. Examples of such instructions include reusable solution fragments or templates, or domain-specific constraints or failure modes, which guide the generation of solutions by the LLM.

In this setting, the state is a context c , which plays a role analogous to a tour c in the traveling salesman Example 2.3.8, except that it represents a mechanism that generates an output trajectory from the LLM. In particular, each context c induces a method for performing a task via the LLM. The heuristic, starting from c , produces a trajectory of reasoning steps, actions, and intermediate results, culminating in an output O ( c ). The performance of this output can be evaluated through a cost function G ( · ), which may be defined via task success, constraint satisfaction, execution correctness, or other externally observable criteria. This process defines the discrete optimization problem where C is a known set of feasible contexts, which is structured according to the task at hand, and can be very large.

<!-- formula-not-decoded -->

The neighborhood of a context c , B ( c ), is a subset of the set of possible contexts C , and is defined by a set of local modifications of c , such as: adding a new rule, refining or specializing an existing rule, removing or disabling a rule, merging or deduplicating similar context elements.

Next States/Contexts

Current State/Context

Selected State/Context

C'E B(c) : LLM O(c) :7

ê E arg mind'eB(c) G(O(c'))

Next States/Contexts Current State/Context Selected State

Figure 2.3.12 Schematic representation of a greedy base heuristic defined by an LLM for the context adaptation problem. The current state is c , the possible next states are denoted c ′ ∈ B ( c ), and the next state selected by the base heuristic is

<!-- image -->

where O ( c ′ ) is the output of the LLM when prompted by c ′ .

<!-- formula-not-decoded -->

These neighborhood operations are analogous to the arc exchanges described for the preceding traveling salesman example. Similar to successful local search methods, the neighborhood is restricted, and arbitrary global rewrites of the context are typically excluded, since they may destroy useful structure accumulated through prior iterations. In fact the neighborhood B ( c ) may be constructed by using some kind of neural network.

For a given LLM, a base heuristic in this setting consists of a procedure for executing tasks and observing feedback; see Fig. 2.3.12. Starting from a given context/state c , the base heuristic generates task executions by applying the LLM to the possible next contexts.

From the viewpoint of local search and rollout methods, context adaptation for LLMs represents a form of meta-level discrete optimization, where the objects being optimized are contexts, that are evaluated using the LLM. The same principles that govern e ff ective local search in classical combinatorial optimization, restricted neighborhoods, incremental improvement, memory of past failures, and selective use of rollout, apply in this LLM setting as well.

The preceding example illustrates a mechanism that can bring to bear neural network technology on local search methods. In particular, the elements of the neighborhood of the current iterate can be evaluated with a neural network, which may take into account the potential of the next iterate to lead to improved solutions at future iterations.

## Rollout

Our discussion so far has focused on how a local search algorithm can be viewed in terms of an optimal control problem, involving a dynamic

system and a base policy. We can now take this view as a starting point for approximation in value space and rollout.

In particular, starting with the current point u , the base heuristic underlying the method can generate truncated trajectories of points, with some terminal cost function approximation. For each truncated trajectory, the terminal cost at the final point of the trajectory serves as its Q-factor. The next point selected by the rollout method is the point in B ( u ) that corresponds to the minimal Q-factor. This type of algorithm brings to bear the benefits of a longer prediction horizon that underlies rollout and the model predictive control methodology (cf. Section 1.6.9).

## 2.3.10 Inference in n -Grams, Transformers, HMMs, and Markov Chains

In this section we consider a type of deterministic sequential decision problem involving n -grams and transformers (cf. Section 1.6, Example 1.6.2). We assume that the n -gram provides next word probabilities that can be used to generate word sequences, and we consider methods for computing N -step word sequences that are highly likely, based on these probabilities. Computing the optimal (i.e., most likely) word sequence starting with a given initial state is an intractable problem for n -grams with large vocabularies, so we consider rollout algorithms that compute highly likely N -word sequences in time that is a low order polynomial in N and in the vocabulary size of the n -gram. This may be useful in several contexts: for example when the n -gram is a transformer that serves as a predictive model of some dynamic process. Then, a likely future word sequence can serve as a basis for present action selection, e.g., in model predictive control.

Our n -gram model generates a sequence ¶ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♦ of text strings, starting from some initial string x 0 (here n and N are fixed positive integers). Each string x k consists of a sequence of n words, chosen from a given list (the vocabulary of the n -gram). The k th string x k is transformed into the next string x k +1 by adding a word at the front end of x k and deleting the word at the back end of x k ; see Example 1.6.2.

Given a text string x k , the n -gram provides probabilities p ( x k +1 ♣ x k ) for the next text string x k +1 . The probabilities can be viewed as the transition probabilities of a stationary Markov chain, whose state space is the set of all n -word sequences x k . ‡ Bearing this context in mind, we also refer to x k as the state (of the underlying Markov chain).

This section is based on joint work with Yuchao Li; see the paper by Li and Bertsekas [LiB24], which also contains extensive computational experimentation.

‡ The stationarity assumption simplifies our notation, but is not essential to our methodology, as we will discuss later. The probabilities may also depend on external information, such as user-provided prompts.

The transition probabilities p ( x k +1 ♣ x k ) provide guidance for generating state sequences with some specific purpose in mind. To this end, a transformer may use a (next word) selection policy , i.e., a (possibly timedependent) function θ k , which selects the text string that follows x k as

<!-- formula-not-decoded -->

We are generally interested in selection policies that give preference to highprobability future words. In particular, we will focus on the computation of the most likely N -step state sequence generated by a general finite-state Markov chain. This is the sequence that maximizes p ( x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♣ x 0 ), where x 0 is a given initial condition. Equivalently, it is the sequence that maximizes

<!-- formula-not-decoded -->

We will see that this is a problem that can be addressed by DP, despite the multiplicative character of the above expression. Indeed DP applies to problems of multiplicative (rather than additive) cost functions, provided the multiplicative terms are positive. The reason is that maximization (or minimization) of the product of positive terms is equivalent to maximization (or minimization) of the sum of the corresponding logarithms.

## State Estimation in Hidden Markov Models

The most likely N -step sequence problem also arises in inference problems involving Hidden Markov Models (HMM), a problem area with a vast range of applications. We will briefly describe this context, before focusing on the problem of generating a most likely sequence in a Markov chain.

In the HMM context, again we have a Markov chain with state at time k denoted by x k , and transition probabilities, which are known and are denoted by

<!-- formula-not-decoded -->

However, the states x k are not observable (i.e., they are 'hidden'). Instead, when a transition from x k to x k +1 occurs, we obtain some data about the transition (cf. Fig. 2.3.13). In particular, we observe the value of a random variable z k generated with probability

<!-- formula-not-decoded -->

The problem is to estimate the N -step trajectory ( x 1 ↪ x 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ) given an observed data sequence ( z 0 ↪ z 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ z N -1 ).

Transition probabilities are 'modified' by data

Replaced by

Figure 2.3.13 Illustration of the state trajectory generated by an HMM. Given a data sequence ( z 0 ↪ z 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ z N -1 ), the most likely trajectory is the one that maximizes the product of weights

<!-- image -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

## Example 2.3.10 (Language Translation)

Suppose that we wish to translate an English phrase consisting of N words into French. The N words of the English phrase are the known data z k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, the hidden states x k of the HMM are French words that may possibly correspond to these English words, and any sequence ( x 1 ↪ x 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ) provides a translation of the English phrase into French. The transition probabilities p ( x k +1 ♣ x k ) and observation probabilities p ( z k ♣ x k ↪ x k +1 ) are assumed known (they can be derived from linguistic data). We wish to find the 'most likely' French translation of the given English phrase, in the sense that it corresponds to the most likely path, the one that maximizes the product of 'modified probabilities'/weights

<!-- formula-not-decoded -->

The product of the weights expresses the likelihood of the state sequence given the observed data sequence. This approach to language translation is known as statistical machine translation , and has been used for many years. Over time it has been replaced by deep neural network and transformer-based translation approaches, which can capture the interdependencies between multiple adjacent phrases.

The HMM inference problem has a long history. It has been addressed principally with the Viterbi algorithm, a shortest path algorithm originally developed for coding/decoding in an information theory context.

A DP-oriented discussion of the Viterbi algorithm, based on the original sources [Vit67], [For73], is given in Section 2.2.2 of the author's book [Ber17]. Related methods such as Viterbi training, the Baum-Welch algorithm, and others, are used widely for the estimation of the most likely sequence, and also for the problem of estimating the transition probabilities p ( x k +1 ♣ x k ) using data. These methods play an important role in several diverse fields, such as speech recognition [Rab69], [EpM02], computational linguistics [JuM23], [MaS99], coding and error correction [PrS01], [PrS08], bioinformatics [Edd96], [DEK98], computational finance [MaE07], and others; see the edited volumes by Bouguila, Fan, and Amayri [BFA22], Mamon and Elliott [MaE14], and Westhead and Vijayabaskar [WeV17]. Compared to these fields, the transformer/ n -gram context tends to involve a much larger state space. Still, however, there are HMM problems where the state space is intractably large, and which can benefit from the rollout methodology to be developed in this section.

In what follows, we will not consider explicitly issues of inference in HMMs. However, our Markov chain/most likely sequence generation methods fully apply to this context, once we modify appropriately the transition probabilities to account for the observed data. In particular, the transition probabilities p ( x k +1 ♣ x k ) should be replaced by the datadependent weights

<!-- formula-not-decoded -->

Then, given the data sequence, ( z 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ z N -1 ), the problem is to find the most likely state sequence ( x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ) that corresponds to this data, i.e., the one that maximizes

<!-- formula-not-decoded -->

cf. Fig. 2.3.13. The exact and approximate DP algorithms that we will discuss in this section fully apply to this problem.

## Computing the Most Likely Sequence in a Markov Chain

We will next consider a finite-state stationary Markov chain and various policies for generating highly likely sequences according to the transition probabilities of the Markov chain. We will generally use the symbols x and y for states, and we will denote the chain's transition probabilities by p ( y ♣ x ). We assume that given a state x , the probabilities p ( y ♣ x ) are either known or can be generated on-line by means of software such as a transformer.

We assume stationarity of the Markov chain in part to alleviate an overburdened notation, and also because n -gram and transformer models are typically assumed to be stationary. However, the rollout methodology

and the manner in which we use it do not depend at all on stationarity of the transition probabilities, or infinite horizon properties of Markov chains, such as ergodic classes, transient states, etc. In fact, they also do not depend on the stationarity of the state space either. Only the Markov property is used in our discussion, i.e., the probability of the next state depends on the immediately preceding state, and not on earlier states.

A selection policy π is a sequence of functions ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ , which given the current state x k , determines the next state x k +1 as

<!-- formula-not-decoded -->

Note that for a given π , the state evolution is deterministic ; so for a given π and x 0 , the generated state sequence ¶ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♦ is fully determined. Moreover the choice of the policy π is arbitrary, although we are primarily interested in π that give preference to high probability next states.

Given a policy π = ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ and a starting state x at time k , the state at future times m&gt;k is denoted by y m↪k ( x↪ π ):

<!-- formula-not-decoded -->

The state trajectory generated by a policy π , starting at state x at time k , is the sequence

<!-- formula-not-decoded -->

(cf. Fig. 2.3.14), and the probability of its occurrence in the given Markov chain is

<!-- formula-not-decoded -->

according to the multiplication rule for conditional probabilities.

## Optimal/Most Likely Selection Policy

The most likely selection policy , denoted by π ∗ = ¶ θ ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ ∗ N -1 ♦ , maximizes over all policies π the probabilities P k ( x↪ π ) for every initial state x and time k . The corresponding probabilities of π ∗ , starting at state x at time k , are denoted by P ∗ k ( x ):

<!-- formula-not-decoded -->

This is similar to our finite-horizon DP context, except that we consider a multiplicative reward function, instead of an additive cost function [ P ∗ k ( x ) can be viewed as an optimal reward-to-go from state x at time k , in place of

State Time

Figure 2.3.14 Illustration of the state trajectory generated by a policy π , starting at state x at time k . The probability of its occurrence, P k ( x↪ π ), is the product of the transition probabilities along the N -k steps of the trajectory [cf. Eq. (2.23)].

<!-- image -->

the optimal cost-to-to J * k ( x ) that we have considered so far in the additive DP case].

The policy π ∗ and its probabilities P ∗ k ( x ) can be generated by the following DP-like algorithm, which operates in two stages:

- (a) It first computes the probabilities P ∗ k ( x ) backwards, for all x , according to

<!-- formula-not-decoded -->

starting with

<!-- formula-not-decoded -->

- (b) It then generates sequentially the selections x ∗ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x ∗ N of π ∗ according to

<!-- formula-not-decoded -->

starting with x ∗ 0 = x 0 .

This algorithm is equivalent to the usual DP algorithm for multistage additive costs, after we take logarithms of the multiplicative expressions defining the probabilities P k ( x↪ π ).

Generally multiplicative reward problems can be converted to additive cost DP problems involving negative logarithms of the multiplicative reward factors (assuming they are positive).

## Greedy Policy

At any given state x k , the greedy policy produces the next state by maximization of the corresponding transition probability over all y :

<!-- formula-not-decoded -->

We assume that ties in the above maximization are broken according to some prespecified deterministic rule. For example if the states are labeled by distinct integers, one possibility is to specify the greedy selection at x k as the state y with minimal label, among those that attain the maximum above. Note that the greedy policy is not only deterministic, but it is also stationary (its selections depend only on the current state and not on the time k ). We will consequently use the notation π = ¶ θ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ ♦ for the greedy policy, where

<!-- formula-not-decoded -->

and θ ( x k ) is uniquely defined according to our deterministic convention for breaking ties in the maximization above. The corresponding probabilities P k ( x k ↪ π ) are given by the DP-like algorithm

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equivalently, we can compute P k ( x↪ π ) by using forward multiplication of the transition probabilities along the trajectory generated by the greedy policy, starting from x ; cf. Eq. (2.23).

The limitation of the greedy policy is that it chooses the locally optimal next state without considering the impact of this choice on future state selections. The rollout approach, to be discussed next, mitigates this limitation with a mechanism for looking into the future, and balancing the desire for a high-probability next transition with the potential undesirability of low-probability future transitions.

## Rollout Policy

At a given state x k , the rollout policy with one-step lookahead produces the next state, denoted ˜ θ k ( x k ), by maximizing p ( y ♣ x k ) P k +1 ( y↪ π ) over all y :

<!-- formula-not-decoded -->

Thus it optimizes the selection of the first state y , assuming that the subsequent states will be chosen using the greedy policy .

starting with

By comparing the maximization (2.28) with the one for the most likely selection policy [cf. Eq. (2.25)], we see that it chooses the next state similarly, except that P ∗ k +1 ( y ) (which is hard to compute) is replaced by the (much more easily computable) probability P k +1 ( y↪ π ). In particular, the latter probability is computed for every y by running the greedy policy forward starting from y and multiplying the corresponding transition probabilities along the generated state trajectory. This is a polynomial computation, which is roughly larger by a factor N over the greedy selection method. However, there are ways to reduce this computation, including the use of parallel processing and other possibilities, which we will discuss later.

The expression p ( y ♣ x k ) P k +1 ( y↪ π ) that is maximized over y in Eq. (2.28) can be viewed as the Q-factor of the pair ( x k ↪ y ) corresponding to the base policy π , and is denoted by Q π ↪k ( x k ↪ y ):

<!-- formula-not-decoded -->

This is similar to the approximation in value space context, except that we consider a multiplicative reward function, whereby at state x k we choose the action y that yields the maximal Q-factor.

## Rollout Policy with /lscript -Step Lookahead

Another rollout possibility includes rollout with /lscript -step lookahead ( /lscript &gt; 1), whereby given x k we maximize over all sequences ¶ y 1 ↪ y 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ y /lscript ♦ up to /lscript steps ahead, the /lscript -step Q-factor

<!-- formula-not-decoded -->

and if ¶ ˜ y 1 ↪ ˜ y 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ y /lscript ♦ is the maximizing sequence, we select ˜ y 1 at x k , and discard the remaining states ˜ y 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ y /lscript . In practice the performance of /lscript -step lookahead rollout policies almost always improves with increasing /lscript . However, artificial examples have been constructed where this is not so; see the book [Ber19a], Section 2.1.1. Moreover, the computational overhead of /lscript -step lookahead increases with /lscript .

## Simplified and Truncated Rollout

As we have already noted, one of the di ffi culties that arises in the application of rollout is the potentially very large number of the Q-factors that need to be calculated at each time step at the current state x [it is equal to the number of states y for which p ( y ♣ x ) &gt; 0]. In practice the computation of Q-factors can be restricted to a subset of most probable next

If /lscript &gt; N -k , then /lscript must be reduced to N -k , to take into account end-of-horizon e ff ects.

states, as per the transition probabilities p ( y ♣ x ) (this is similar to simplified rollout that we discussed in Section 2.3.5). For example, often many of the transition probabilities p ( y ♣ x ) are very close to 0, and can be safely ignored.

Another possibility to reduce computation is to truncate the trajectories generated from the next states y by the greedy policy, up to m steps (assuming that k + m &lt; N , i.e., if we are more than m steps away from the end of the horizon). This is essentially the truncated rollout algorithm, discussed in Section 2.3.6, whereby we maximize over y the m -step Q-factor of the greedy policy π :

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is the m -step product of probabilities along the path generated by the greedy policy π starting from y at time k +1 [cf. Eq. (2.23)]. By contrast, in rollout without truncation, we maximize over y

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is the ( N -k -1)-step product of probabilities along the path generated by the greedy policy starting from y at time k +1; cf. Eqs. (2.23) and (2.28).

## Multiple On-line Policy Iterations - Double Rollout

Still another possibility is to apply the rollout approach successively, in multiple policy iterations , by using the rollout policy obtained at each iteration as base policy for the next iteration. This corresponds to the fundamental DP algorithm of policy iteration .

Performing on-line just two policy iterations amounts to using the rollout algorithm as a base policy for another rollout algorithm. This has been called double rollout , and it has been discussed in Section 2.3.5 of the book [Ber20] and Section 6.5 of the book [Ber22]. It amounts to two successive Newton steps for solving Bellman's equation. Generally, one-step lookahead rollout requires O ( q · N ) applications of the base policy where q

is the number of Q-factors calculated at each time step. Thus with each new policy iteration, there is an amplification factor O ( q · N ) of the computational requirements. Still, however, the multiple iteration approach may be viable, even on-line, when combined with some of the other time-saving computational devices described above (e.g., truncation and simplification to reduce q ), in view of the relative simplicity of the calculations involved and their suitability for parallel computation. This is particularly so for double rollout. Policy iteration/double rollout is discussed by Yan et al. [YDR04] in the context of the game of solitaire, and by Silver and Barreto [SiB22] in the context of a broader class of search methods.

We next show that the rollout selection policy with one-step lookahead has a performance improvement property : it generates more likely state sequences than the greedy policy, starting from any state, and the improvement is often very substantial.

## Performance Improvement Properties of Rollout Policies

We will show by induction a performance improvement property of the rollout algorithm with one-step lookahead, namely that for all states x ∈ X and k , we have

<!-- formula-not-decoded -->

i.e., the probability of the sequence generated by the rollout policy is greater or equal to the probability of the sequence generated by the greedy policy; this is true for any starting state x at any time k . This is similar to our earlier rollout cost improvement results in Section 2.3.1.

Indeed, for k = N this relation holds, since we have

<!-- formula-not-decoded -->

Assuming that

<!-- formula-not-decoded -->

For a more accurate estimate of the complexity of the greedy, rollout, and double rollout algorithms, note that the basic operation of the greedy operation is the maximization over the q numbers p ( y ♣ x k ). Thus m steps of the greedy algorithm, as in an m -step Q-factor calculation, costs q · m comparisons. In m -step truncated rollout, we compare q greedy Q-factors so the number of comparisons per rollout time step is q 2 m + q . Over N time steps the total is ( q 2 m + q ) · N comparisons, while for the greedy algorithm starting from the initial state x 0 , the corresponding number is q · N . Thus there is an amplification factor of qm +1 for the computation of simplified m -step truncated rollout over the greedy policy. Similarly it can be estimated that there is an extra amplification factor of no more than qm +1 for using double rollout with (single) rollout as a base policy.

we will show that

<!-- formula-not-decoded -->

Indeed, we use the preceding relations to write

<!-- formula-not-decoded -->

where

- ÷ The first equality holds from the definition of the probabilities corresponding to the rollout policy ˜ π .
- ÷ The first inequality holds from the definition by the induction hypothesis.
- ÷ The second inequality holds from the fact that the rollout choice ˜ θ k ( x ) maximizes the Q-factor p ( ˜ y ♣ x ) P k +1 ( y↪ π ) over y .
- ÷ The second equality holds from the definition of the probabilities corresponding to the greedy policy π .

Thus the induction proof of the improvement property (2.32) is complete. Clearly, the performance improvement property continues to hold in successive multiple iterations of the rollout policy, and in fact it can be shown that after a su ffi ciently large number of iterations it yields the most likely selection policy. This is a consequence of classical policy iteration convergence results, which establish the finite convergence of the policy iteration algorithm for finite-state Markovian decision problems, see e.g., [Ber12], [Ber17a], [Ber19a].

Performance improvement can also be established for the /lscript -step lookahead version of the rollout policy, using an induction proof that is similar to the one given above for the one-step lookahead case. Moreover, the books [Ber20a], [Ber22a] describe conditions under which simplified rollout maintains the performance improvement property. However, it is not necessarily true that the performance of the /lscript -step lookahead rollout policy improves as /lscript increases; see an example in the book [Ber19a], Section 2.1.1. Similarly, it is not necessarily true that the m -step truncated rollout policy performs better than the greedy policy. On the other hand, known performance deterioration examples of this type are artificial and are apparently rare in practice.

It performs better than an m -step version of the greedy policy, which generates a sequence of m +1 states, starting from the current state and using the greedy policy.

## Computational Comparison of Greedy, Optimal, and Rollout Policies

We will now present some illustrative computational comparisons, using Markov chains that are small enough for the optimal/most likely selection policy to be computed exactly via the DP-like algorithm (2.24)-(2.25). Thus, the performance di ff erences between the rollout, greedy, and optimal policies can be accurately assessed. The experiments are presented in more detail in the paper [LiB24], which also contains qualitatively similar computational results with much larger Markov chains involving n -grams, and a trained transformer neural network. We used Markov chains where there is a fixed number q of distinct states y such that p ( y ♣ x ) &gt; 0, with q being the same for all states x . These states were selected according to a uniform distribution, whereby all y with p ( y ♣ x ) &gt; 0 are equally likely. The probabilities p ( y ♣ x ) were also generated according to a uniform distribution.

Let us denote the state space by X and the number of states by ♣ X ♣ . We refer to the ratio qglyph[triangleleft] ♣ X ♣ (in percent) as the branching factor of the chain. We represent the probability of an entire sequence generated by a policy as the average of its constituent transition probabilities (i.e., a geometric mean over N as will be described below). In particular, given a sample set C of Markov chains, we compute the optimal occurrence probability of generated sequences, averaged over all chains, states, and transitions, and denoted by ( P ∗ 0 ) 1 glyph[triangleleft]N , according to the average geometric mean formula

<!-- formula-not-decoded -->

where P ∗ 0 ↪c ( x ) is the optimal occurrence probability with x 0 = x and Markov chain c in the sample set. Similarly, we compute the occurrence probabilities of sequences generated by the greedy policy averaged over all chains, states, and transitions, and denoted by ( P 0 ) 1 glyph[triangleleft]N , according to

<!-- formula-not-decoded -->

where P 0 ↪c ( x↪ π ) is the transition probability of the sequence generated by the greedy policy with x 0 = x and Markov chain indexed by c . For the rollout algorithm (or variants thereof), we compute its averaged occurrence probability ( ˜ P 0 ) 1 glyph[triangleleft]N similar to Eq. (2.33) with ˜ π in place of π . Then the performance of the rollout algorithm can be measured by its percentage recovery of the optimality loss of the greedy policy, given by

<!-- formula-not-decoded -->

Performance recovery (%)

100 -

80 -

60 -

40 -

20 -

0

64.3

82.89

87.62

79.64

Without Truncation

58.36

Figure 2.3.15 Percentage recovery of the optimality loss of the greedy policy through the use of rollout and its variants, applied to sequence selection problems with N = 100 for 50 randomly generated Markov chains with 100 states and 5% branching factor. Here ˜ π /lscript is rollout with /lscript -step lookahead, and ˜ π m /lscript represents m -step truncated rollout for m = 10 with /lscript -step lookahead. It can be seen that on average, rollout and its variants provide a substantial improvement over the greedy policy, the improvement increases with the size of the lookahead, and truncated rollout methods perform comparable to their exact counterparts.

<!-- image -->

This performance measure describes accurately how the rollout performance compares with the greedy policy and how close it comes to optimality.

In the experiments presented here we have used small Markov chains with ♣ X ♣ = 100 states, branching factor q = 5%, and sequence length N = 100. In summary, the percentage recovery has ranged roughly from 60% to 90% for one-step to five-step lookahead, untruncated and truncated rollout with m = 10 steps up to truncation; see Fig. 2.3.15. The performance improves as the length of the lookahead increases, but seems remarkably una ff ected by the 90% truncation of the rollout horizon (the relative insensitivity of the performance of truncated rollout to the number of rollout steps m has been observed in other application contexts as well). The figure has been generated with a sample of 50 di ff erent Markov chains with ♣ X ♣ = 100 states, branching factor equal to 5%, and sequence length N = 100. The figure shows the results obtained by rollout with one-step and multi-step lookahead (ranging from 2 to 5 steps), and their m -step truncated counterparts with m = 10. Their percentage recovery, evaluated according to Eq. (2.34), is given in Fig. 2.3.15, where ˜ π /lscript denotes rollout with /lscript -step lookahead, and ˜ π m /lscript denotes m -step truncated rollout with /lscript -step lookahead.

Figure 2.3.16 provides corresponding results using double rollout, which show a significant improvement over the case of single rollout. Moreover, truncating the rollout horizon by 90% has remarkably small e ff ect on

91.51

73.02

84.18 84.37

92.79

(P.*) 1/100

Percentage recovery (%)

100 -

80 -

60 -

40 -

20 -

0

64.3

87.97 89.84 90.72 91.47 92.73

72

Without Truncation

85.31

80.75

Figure 2.3.16 Percentage recovery of the optimality loss of the greedy policy through the use of double rollout and its variants, applied to sequence selection problems with N = 100 for 50 randomly generated Markov chains with 100 states and 5% branching factor. Here ˜ π /lscript is rollout with /lscript -step lookahead, and ˜ π m /lscript represents m -step truncated rollout for m = 10 with /lscript -step lookahead.

<!-- image -->

the percentage recovery, similar to the case of a single rollout.

The preceding results are consistent with those of other computational studies using rollout and its variants. The sequences produced by rollout improve substantially over those generated by the base policy. Moreover, there is typically a relatively small degradation of performance when applying the truncated rollout compared with untruncated rollout. This is significant as truncated rollout greatly reduces the computation if m is substantially smaller than N , and also makes possible the use of double rollout.

## 2.4 APPROXIMATION IN VALUE SPACE WITH MULTISTEP LOOKAHEAD

We will now consider approximation in value space with multistep lookahead minimization, possibly also involving some form of rollout. Figure 2.4.1 describes the case of pure (nontruncated) form of rollout with twostep lookahead for deterministic problems. In particular, suppose that after k steps we have reached state x k . We then consider the set of all possible two-step-ahead states x k +2 , we run the base heuristic starting from each of them, and compute the two-stage cost to get from x k to x k +2 , plus the cost of the base heuristic from x k +2 . We select the state, say ˜ x k +2 , that is associated with minimum cost, compute the controls ˜ u k and ˜ u k +1 that lead from x k to ˜ x k +2 , choose ˜ u k as the next control and x k +1 = f k ( x k ↪ ˜ u k ) as the next state, and discard ˜ u k +1 .

89.27 89.75

91.34

(P*) 1/100

XO

X1

Xk

Uk

Figure 2.4.1 Illustration of multistep rollout with /lscript = 2 for deterministic problems. We run the base heuristic from each leaf x k + /lscript at the end of the lookahead graph. We then construct an optimal solution for the lookahead minimization problem, where the heuristic cost is used as terminal cost approximation. We thus obtain an optimal /lscript -step control sequence through the lookahead graph, use the first control in the sequence as the rollout control, discard the remaining controls, move to the next state, and repeat. Note that the multistep lookahead minimization may involve approximations aimed at simplifying the associated computations.

<!-- image -->

The extension of the algorithm to lookahead of more than two steps is straightforward: instead of the two-step-ahead states x k +2 , we run the base heuristic starting from all the possible /lscript -step ahead states x k + /lscript , etc. For cases where the /lscript -step lookahead minimization is very time consuming, we may consider variants involving approximations aimed at simplifying the associated computations.

An important variation is truncated rollout with terminal cost approximation . Here the rollout trajectories are obtained by running the base heuristic from the leaf nodes of the lookahead graph, and they are truncated after a given number of steps, while a terminal cost approximation is added to the heuristic cost to compensate for the resulting error; see Fig. 2.4.2. One possibility that works well for many problems, particularly when the combined lookahead for minimization and base heuristic simulation is long, is to simply set the terminal cost approximation to zero. Alternatively, the terminal cost function approximation can be obtained by problem approximation or by using some sophisticated o ff -line training process that may involve an approximation architecture such as a neural network. Generally, the terminal cost approximation is especially important if a large portion of the total cost is incurred upon termination (this

States at the End of the Lookahead

Uk+1

Xk+2

Heuristic

Xk+1

Final States

Lookahead Tree

Current State

Xk)

States Xk+1

Rollout

-Factors Current State

Figure 2.4.2 Illustration of truncated rollout with two-step lookahead and a terminal cost approximation ˜ J . The base heuristic is used for a limited number of steps and the terminal cost is added to compensate for the remaining steps.

<!-- image -->

is true for example in games).

Note that the preceding algorithmic scheme can be viewed as multistep approximation in value space, and it can be interpreted as a Newton step , with suitable starting point that is determined by the truncated rollout with the base heuristic, and the terminal cost approximation. This interpretation is possible once the discrete optimal control problem is reformulated to an equivalent infinite horizon SSP problem; cf. the discussion of Sections 1.6.2 and 2.1. Thus the algorithm inherits the fast convergence property of the Newton step, which we have discussed in the context of infinite horizon problems in Section 1.5; see also the book [Ber22a].

The architecture of Fig. 2.4.2 contains as a special case the general multistep approximation in value space scheme, where there is no rollout at all; i.e., the leaves of the multistep lookahead tree are evaluated with the function ˜ J . Figure 2.4.3 illustrates this special case, where for notational simplicity we have denoted the current state by x 0 . The illustration involves an acyclic graph with a single root (the current state) and /lscript layers, with the n th layer consisting of the states x n that are reachable from x 0 with a feasible sequence of n controls. In particular, there is an arc for every state x 1 of the 1st layer that can be reached from x 0 with a feasible control, and similarly an arc for every pair of states ( x n ↪ x n +1 ), of layers n and n +1, respectively, for which x n +1 can be reached from x n with a feasible control. The cost of each of these arcs is the stage cost of the corresponding state-

Terminal Cost

Approximation J

Truncated Rollout Terminal Cost Approximation

Move Chosen

XI Q

— *—

Terminal Cost Approximation xo (Current State)

- Layer 1

- Layer 2

Shortest Path

Layer n

Figure 2.4.3 Illustration of the general /lscript -step approximation in value space scheme with a terminal cost approximation ˜ J where x 0 denotes the current state. It involves an acyclic graph of /lscript layers, with layer n , n = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ /lscript , consisting of all the states x n that can be reached from x 0 with a sequence of n feasible controls. In /lscript -step approximation in value space, we obtain a trajectory

<!-- image -->

<!-- formula-not-decoded -->

that minimizes the shortest distance from x 0 to x /lscript plus ˜ J ( x /lscript ). We then use the control that corresponds to the first move x 0 → x ∗ 1 .

control pair, minimized over all possible controls that correspond to the same pair ( x n ↪ x n +1 ). Mathematically, the cost of the arc ( x n ↪ x n +1 ) is

<!-- formula-not-decoded -->

For the states x /lscript of the last layer there is also the terminal cost approximation ˜ J ( x /lscript ), which may be obtained through o ff -line training or some other means. It can be thought of as the cost of an artificial arc connecting x /lscript to an artificial termination state.

Once we have computed all the shortest distances D ( x /lscript ) from x 0 to all states x /lscript of the last layer /lscript , we obtain the /lscript -step lookahead control to be applied at the current state x 0 , by minimizing over x /lscript the sum

<!-- formula-not-decoded -->

If x ∗ /lscript is the state that attains the minimum, we generate the corresponding trajectory ( x 0 ↪ x ∗ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x ∗ /lscript ), and then use the control that corresponds to

the first move x 0 → x ∗ 1 ; see Fig. 2.4.3. Note that the shortest path problems from x 0 to all states x n of all the layers n = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ /lscript can be solved simultaneously by backward DP (start from layer /lscript and go back towards x 0 ).

## Long Lookahead for Deterministic Problems

The architecture of Figs. 2.4.2 and 2.4.3 is similar to the one we discussed in Section 1.1 for AlphaZero and related programs. However, because it is adapted to deterministic problems, it is much simpler to implement and to use. In particular, the truncated rollout portion does not involve expensive Monte Carlo simulation, while the multistep lookahead minimization portion involves a deterministic shortest path problem, which is much easier to solve than its stochastic counterpart. These favorable characteristics can be exploited to facilitate implementations that involve very long lookahead.

Generally speaking, longer lookahead is desirable because it typically results in improved performance . We will adopt this as a working hypothesis. It is typically true in practice, although it cannot be established analytically in the absence of additional assumptions. On the other hand, the on-line computational cost of multistep lookahead increases, often exponentially, with the length of lookahead. We conclude that we should aim to use a lookahead that is as long as is allowed by the on-line computational budget (the amount of time that is available for calculating a control to apply at the current state).

## Long Lookahead by Rollout is Far More Economical than Long Lookahead Minimization

Our preceding discussion leads to the question of how to economize in computation in order to e ff ectively increase the length of the multistep lookahead within a given on-line computational budget. One way to do this, which we have already discussed, is the use of truncated rollout that explores forward through a deterministic base policy at far less computational cost than lookahead minimization of equal length. As an example, let us consider the possibility of starting with a terminal cost function ˜ J , possibly generated by o ff -line training, and use as base policy for rollout the one-step lookahead policy ˜ θ , defined by ˜ J using the equation ‡

<!-- formula-not-decoded -->

Indeed, there are examples where as the size /lscript of the lookahead becomes longer, the performance of the multistep lookahead policy deteriorates (see [Ber17a], Section 6.1.2, or [Ber19a], Section 2.2.1). However, these examples are isolated and artificial. They are not representative of practical experience.

‡ For simplicity, we use stationary system notation, omitting the time subscripts of U , g , and f .

Let us assume that the principal computation in the minimization of Eq. (2.36) is the calculation of ˜ J ( f ( x↪ u ) ) , and compare two possibilities:

- (a) Using /lscript -step lookahead minimization with ˜ J as the terminal cost approximation without any rollout ; cf. Fig. 2.4.3.
- (b) Using one-step lookahead minimization, with ( /lscript -1) -step truncated rollout and ˜ J as the terminal cost approximation .

Note that scheme (b) is the one used by the TD-Gammon program of Tesauro and Galperin [TeG96], out of necessity: multistep lookahead minimization is very expensive in backgammon, due to the rapid growth of the lookahead graph as /lscript increases (cf. the discussion of Section 1.1).

Suppose that the control set U ( x ) has m elements for every x . Then the /lscript -step lookahead minimization scheme (a) requires the calculation of as many as m /lscript values of ˜ J , because the number of leaves of the m -step lookahead graph are as many as m /lscript .

The corresponding number of calculations of the value of ˜ J for scheme (b) is clearly much smaller. To verify this, let us calculate this number as a function of m and /lscript . In particular, the first lookahead stage starting from the current state x k requires m calculations corresponding to the m controls in U ( x k ), and yields corresponding states x k +1 , which are as many as m . For each of these states x k +1 , we must calculate a sequence of /lscript -1 controls using the base policy (2.36) for stages ( k +1) to ( k + /lscript ). Each of these /lscript -1 controls requires m calculations of the value of ˜ J . Thus, for the /lscript -1 stages of truncated rollout, there are m · ( /lscript -1) calculations of the value of ˜ J per state x k +1 , for a total of as many as m 2 · ( /lscript -1) calculations. Adding the m calculations at state x k , we conclude that scheme (b) requires a total of as many as m 2 · /lscript calculations of the value of ˜ J .

In conclusion, both schemes (a) and (b) above look forward for /lscript stages, but their associated total computation grows exponentially and linearly with /lscript , respectively . Thus, for a given computational budget, short lookahead minimization with long truncated rollout, can increase the total amount of lookahead and improve the performance of approximation in value space schemes. This is particularly so since based on the Newton step interpretations of approximation in value space of Section 1.5, truncated rollout with a reasonably good (e.g., stable) base policy often works about as well as long lookahead minimization. Extensive computational practice, starting with the rollout/TD-Gammon scheme of [TeG96], is consistent with this assessment.

In the following two sections, we will explore two alternative ways to speed up the lookahead minimization calculation, thereby allowing a larger number /lscript of computational stages for a given on-line computational budget. These are based on iterative deepening of the shortest path computation, and pruning of the lookahead minimization graph.

Dn (Xn): Shortest distance from xo to state n

of layer n

In

In+1,

Xe

X2

X1

Xo (Current State)

- Layer 1

• —

Layer 2

<!-- image -->

): Shortest distance from

Multistep Lookahead

Figure 2.4.4 Illustration of the forward DP algorithm for computing the shortest distances from the current state x 0 to all the states x n of the layers n = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ /lscript . The shortest distance D n +1 ( x n +1 ) to a state x n +1 of layer n +1 is obtained by minimizing over all predecessor states x n the sum

<!-- formula-not-decoded -->

## 2.4.1 Iterative Deepening Using Forward Dynamic Programming

As noted earlier, the shortest path problems from x 0 to x /lscript in Fig. 2.4.3 can be solved simultaneously by the familiar backward DP that starts from layer /lscript and goes towards x 0 . An important alternative for solving these problems is the forward DP algorithm. This is the same as the backwards DP algorithm with the direction of the arcs reversed (start from x 0 and go towards layer /lscript ). In particular, the shortest distances D n +1 ( x n +1 ) to layer n + 1 states are obtained from the shortest distances D n ( x n ) to layer n states through the equation

<!-- formula-not-decoded -->

which is also illustrated in Fig. 2.4.4. Here ˆ g n ( x n ↪ x n +1 ) is the cost (or length) of the arc ( x n ↪ x n +1 ); cf. Eq. (2.35).

In particular, the solution of the /lscript -step lookahead problem is obtained from the shortest path to the state x ∗ /lscript of layer /lscript that minimizes D /lscript ( x /lscript ) + ˜ J ( x /lscript ). The idea of iterative deepening is to progressively solve the n -step lookahead problem first for n = 1 , then for n = 2 , and so on, until our

• Layer n + 1

Xe

Pruned States

Xn

Xn+1

Terminal Cost Approximation

Multistep Lookahead xo (Current State)

Layer 1

- Layer 2

Pruned States

- Layer n + 1

Figure 2.4.5 Illustration of iterative deepening with pruning within the context of forward DP.

<!-- image -->

on-line computational budget is exhausted . In addition to fitting perfectly the mechanism of the forward DP algorithm, this scheme has the character of an 'anytime' algorithm ; i.e., it returns the shortest distances to some depth n ≤ /lscript , which can in turn yield a solution of an n -step lookahead minimization after adding a suitable terminal cost function. In practice, this is an important advantage, well known from chess programming, which allows us to keep on aiming for longer lookahead minimization, within the limit imposed by our computational budget constraint.

## Iterative Deepening Combined with Pruning

A principal di ffi culty in approximation in value space with /lscript -step lookahead stems from the rapid expansion of the lookahead graph as /lscript increases. One way to mitigate this di ffi culty is to 'prune' the lookahead minimization graph, i.e., to delete some of its arcs in order to expedite the shortest path computations from the current state to the states of subsequent layers; see Fig. 2.4.5. One possibility is to combine pruning with iterative deepening by eliminating from the computation states ˆ x n of layer n such that the n -step lookahead cost D n (ˆ x n ) + ˜ J (ˆ x n ) is 'far from the minimum' over x n . This in turn prunes automatically some of the states of the next layer n +1. The rationale is that such states are 'unlikely' to be part of the shortest path that we aim to compute. Note that this type of pruning is progressive, i.e., we prune states in layer n before pruning states in layer n +1.

X2

X1

S x

Expansion of 'Most Promising' Leaf Node of

Figure 2.4.6 Illustration of the /lscript -step lookahead minimization problem and its suboptimal solution with the IMR algorithm. The algorithm maintains a connected acyclic subgraph S as shown. At each iteration it expands S by selecting a leaf node of S and by adding its neighbor nodes to S (if not already in S ). The leaf node, denoted x ∗ , is the 'most promising': the one that minimizes over all leaf nodes x of S the sum of the shortest distance D ( x ) from x 0 to x and a 'heuristic cost' H ( x ).

<!-- image -->

## 2.4.2 Incremental Multistep Rollout

We will now consider a more flexible form of the rollout scheme, which we call incremental multistep rollout (IMR). It applies a base heuristic and a forward DP computation to a sequence of subgraphs of a multistep lookahead graph, with the size of the subgraphs expanding iteratively. In particular, in incremental rollout a connected subgraph of multiple paths is iteratively extended starting from the current state going towards the end of the lookahead horizon, instead of extending a single path as in rollout. This is similar to what is done in Monte Carlo Tree Search (MCTS, to be discussed later), which is also designed to solve approximately general multistep lookahead minimization problems (including stochastic ones), and involves iterative expansion of an acyclic lookahead graph to new nodes, as well as backtracking to previously encountered nodes. However, incremental rollout seems to be more appropriate than MCTS for deterministic problems, where there are no random variables in the problem's model and therefore Monte Carlo simulation does not make sense.

The IMR algorithm starts with and maintains a connected acyclic subgraph S of the given multistep lookahead graph G , which contains x 0 . At each iteration it expands S by selecting a leaf node of S and by adding its neighbor nodes to S (if not already in S ); see Fig. 2.4.6. This leaf node, denoted x ∗ , is the 'most promising' one, in the sense that it minimizes (over all leaf nodes x of S ) the sum

<!-- formula-not-decoded -->

where

D ( x ) is the shortest distance from x 0 to the leaf node x using only arcs that belong to S . This can be computed by forward DP. [A noteworthy possibility is to replace D ( x ) with a conveniently computed approximation, which may be problem-specific. We will discuss such schemes later in this section.]

H ( x ) is a 'heuristic cost' corresponding to x . This is defined as the sum of three terms:

- (a) The cost of the base heuristic starting from node x and ending at one of the states x /lscript in the last layer /lscript .
- (b) The terminal cost approximation ˜ J ( x /lscript ), where x /lscript is the state obtained via the base heuristic as in (a) above.
- (c) An additional loss term L ( x ) that depends on the layer to which x belongs. As an example, we will assume here that

<!-- formula-not-decoded -->

where δ is a positive parameter. Thus L ( x ) adds a cost of δ for each extra arc to reach x from x 0 , and penalizes nodes x that lie in more distant layers from the root x 0 . It encourages the algorithm to 'backtrack' and select nodes x ∗ that lie in layers closer to x 0 . [Other ways to define L ( x ) may also be considered.]

The algorithm starts with a connected acyclic subgraph S of the given multistep lookahead graph G , which contains x 0 , and with a path P that starts at x 0 , goes through S and ends at one of the terminal nodes of S (one possibility is take S and P equal to just the root node x 0 ). It terminates when the end node of P belongs to layer /lscript (or earlier if a computational budget constraint is reached).

The role of the parameter δ is noteworthy and a ff ects significantly the nature of the algorithm. When δ = 0, the initial graph S consists of the single state x 0 , and the base heuristic is sequentially improving, it can be seen that IMR performs exactly like the rollout algorithm for solving the /lscript -step lookahead minimization problem. On the other hand when δ is large enough, the algorithm operates like the forward DP algorithm. The reason is that a very large value of δ forces the algorithm to expand all nodes of a given layer before proceeding to the next layer.

Generally, as δ increases, the algorithm tends to backtrack more often, and to generate more paths through the graph, thereby visiting more nodes and increasing the number of applications of the base heuristic. Thus δ may be viewed as an exploration parameter ; when δ is large the algorithm tends to explore more paths thereby improving the quality of the multistep lookahead minimization, at the expense of greater computational e ff ort. In the absence of additional problem-specific information, favorable values

of δ should be obtained through experimentation. One may also consider alternative and more adaptive schemes; for example with a δ that depends on x 0 , and is adjusted in the course of the computation. Finally, we note that if the base heuristic used in the calculation of H ( x ) is sequentially consistent, a local optimality property of the incremental rollout trajectory can be shown (cf. Section 2.3.1).

## Approximations in the Incremental Multistep Rollout Scheme

Let us now consider variants of the IMR scheme, which are aimed towards expediting its calculations, possibly at the expense of some degradation in its performance guarantees. To this end, it is useful to view the algorithm as maintaining a list L of the current leaf nodes of S . At each step a node is removed from L , and its neighbor nodes are added to S and to L , if not already there. This process continues until a path through S that starts at the root node x 0 and ends at some node of the last layer /lscript is constructed. In particular, each step of the IMR algorithm consists of:

- (a) The selection of a node x ∗ of L for expansion.
- (b) The addition to S and to L of each neighbor node x of the expanded node x ∗ , if x does not already belong to S and/or L , respectively. The values D ( x ) and H ( x ) are also computed at the time when x enters the list L .
- (c) The removal of x ∗ from L .

Since evidently a node can enter the list L at most once, the algorithm will terminate with a path that starts at x 0 and ends at some node of layer /lscript . Moreover, this will happen regardless of which leaf node of the current subgraph S is chosen at each step for expansion.

It follows from the preceding argument that selecting the leaf node x ∗ that minimizes D ( x ) + H ( x ) at each step is not essential to the algorithm's termination. This allows some flexibility in designing variations of the IMR algorithm, which aim at expediting the computation. In particular, we may consider selecting a leaf node x that has a 'low' value of D ( x ) + H ( x ) instead of selecting the one that has minimum value, while maintaining an appropriate cost improvement property. Two possibilities of this type are as follows:

- (1) Replace D ( x ), the shortest distance from x 0 to x through the subgraph S , with an upper bound D ( x ), which is the shortest distance from x 0 to x through the subgraph S at the time that x becomes a leaf node of S and enters the list L of leaf nodes. Note that as the set S grows, D ( x ) may become smaller than D ( x ), as more nodes are added to S and additional paths from x 0 to x through S are created. The important point here is that in general D ( x ) is in many cases likely to be either equal or not too di ff erent than D ( x ). Moreover,

D ( x ) is computed only once, at the time when x becomes a leaf node of S , thus saving in computational overhead.

- (2) Organize L as a priority queue and instead of selecting a node x ∗ that minimizes D ( x ) + H ( x ) or D ( x ) + H ( x ) over all nodes x of L , simply let x ∗ be the top node of the queue. This will save the overhead for minimizing over the nodes of L , which can be very large in number, depending on the problem at hand. We note here that it is possible to reduce this overhead by organizing L with a heap or bucket data structure, whereby the minimizing node is e ffi ciently selected; such schemes are well known from implementations of label setting shortest path computations (i.e., Dijkstra's algorithm, see e.g., [Ber98], Ch. 2). However, simpler schemes to organize L are possible, which place nodes with small value of D ( x )+ H ( x ) near the top of the queue. Such schemes have been proposed by the author for label correcting methods for shortest paths, in the context of approximations to Dijkstra's algorithm. We refer to the SLF (Small Label First), LLL (Last Label Last), and threshold shortest path algorithms, which are described in the network optimization book [Ber98] (Chapter 2) and the references given in that book. In practice, these algorithms work very well, and typically better than the heap or bucket schemes.

An important point is that when δ is small or is 0, the preceding variants of the IMR algorithm can be easily modified to coincide with the rollout algorithm, and inherit the corresponding cost improvement property. Moreover the cost improvement property can be restored by using the fortified rollout ideas of Section 2.3. In addition, the flexibility a ff orded by the modifications (1) and (2) above allow variations of the IMR scheme that are tailored to the problem at hand.

## 2.5 CONSTRAINED FORMS OF ROLLOUT ALGORITHMS

In this section we will discuss constrained deterministic DP problems, including challenging combinatorial optimization and integer programming problems. We introduce a rollout algorithm, which relies on a base heuristic and applies to problems with general trajectory constraints. Under suitable assumptions, we will show that if the base heuristic produces a feasible solution, the rollout algorithm has a cost improvement property: it produces a feasible solution, whose cost is no worse than the base heuristic's cost.

Before going into formal descriptions of the constrained DP problem formulation and the corresponding algorithms, it is worth to revisit the broad outline of the rollout algorithm for deterministic DP:

- (a) It constructs a sequence ¶ T 0 ↪ T 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ T N ♦ of complete system trajectories with monotonically nonincreasing cost (assuming a sequential improvement condition).

- (b) The initial trajectory T 0 is the one generated by the base heuristic starting from x 0 , and the final trajectory T N is the one generated by the rollout algorithm.
- (c) For each k , the trajectories T k ↪ T k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ T N share the same initial portion ( x 0 ↪ ˜ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u k -1 ↪ ˜ x k ).
- (d) For each k , the base heuristic is used to generate a number of candidate trajectories, all of which share the initial portion with T k , up to state ˜ x k . These candidate trajectories correspond to the controls u k ∈ U k ( x k ). (In the case of fortified rollout, these trajectories include the current 'tentative best' trajectory.)
- (e) For each k , the next trajectory T k +1 is the candidate trajectory that is best in terms of total cost.

In our constrained DP formulation, to be described shortly, we introduce a trajectory constraint T ∈ C , where C is some subset of admissible trajectories. A consequence of this is that some of the candidate trajectories in (d) above, may be infeasible. Our modification to deal with this situation is simple: we discard all the candidate trajectories that violate the constraint, and we choose T k +1 to be the best of the remaining candidate trajectories, the ones that are feasible .

Of course, for this modification to be viable, we have to guarantee that at least one of the candidate trajectories will satisfy the constraint for every k . For this we will rely on a sequential improvement condition that we will introduce shortly. For the case where this condition does not hold, we will introduce a fortified version of the algorithm, which requires only that the base heuristic generates a feasible trajectory T 0 starting from the initial condition x 0 . Thus to apply reliably the constrained rollout algorithm, we only need to know a single feasible solution , i.e., a trajectory T 0 that starts at x 0 and satisfies the constraint T 0 ∈ C .

## Constrained Problem Formulation

We assume that the state x k takes values in some (possibly infinite) set and the control u k takes values in some finite set. The finiteness of the control space is only needed for implementation purposes of the rollout algorithms to be described shortly. The algorithm can be defined without the finiteness condition, and makes sense, provided the implementation issues associated with infinite control spaces can be dealt with. A sequence of the form

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is referred to as a complete trajectory . Our problem is stated succinctly as

<!-- formula-not-decoded -->

where G is some cost function and C is the constraint set.

Note that G need not have the additive form

<!-- formula-not-decoded -->

which we have assumed so far. Thus, except for the finiteness of the control space, which is needed for implementation of rollout, this is a very general optimization problem. In fact, later we will simplify the problem further by eliminating the state transition structure of Eq. (2.38).

Trajectory constraints can arise in a number of ways. A relatively simple example is the standard problem formulation for deterministic DP: an additive cost of the form (2.40), where the controls satisfy the timeuncoupled constraints u k ∈ U k ( x k ) [so here C is the set of trajectories that are generated by the system equation with controls satisfying u k ∈ U k ( x k )]. In a more complicated constrained DP problem, there may be constraints that couple the controls of di ff erent stages such as

<!-- formula-not-decoded -->

where g m k and b m are given functions and scalars, respectively. Examples of this type include multiobjective or Pareto optimization problems, where there are multiple cost functions of interest, and all but one of the cost functions are treated through constraints (see e.g., [Ber17a], Ch. 2). Examples where di ffi cult trajectory constraints arise also include situations where the control contains some discrete components, which once chosen must remain fixed for multiple time periods.

Here is a discrete optimization example involving the traveling salesman problem. A related classical example is the knapsack problem, described in most books on discrete optimization algorithms.

## Example 2.5.1 (A Constrained Form of the Traveling Salesman Problem)

Let us consider a constrained version of the traveling salesman problem of Example 1.2.2. We want to find a minimum travel cost tour that additionally

Actually, similar to our discussion on model-free rollout in Section 2.3.6, it is not essential that we know the explicit form of the cost function G and the constraint set C . For our constrained rollout algorithms, it is su ffi cient to have access to a human or software expert that can determine whether a given trajectory T is feasible, i.e., satisfies the constraint T ∈ C , and also to be able to compare any two feasible trajectories T 1 and T 2 , based on some internal process that is unknown to us, without assigning numerical values to them.

ABC

1

ABCD

5

4

ABD

3

ABDC

20

Matrix of Intercity

Travel Costs

5

1

20

20

20

1

20

4

1

3

4

1

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Matrix of Intercity Travel Costs

1 2 3 4 5 6 7 8 9 10 11 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Initial State xo

A

1

AC

AD

<!-- image -->

A AB AC AD ABC ABD ACB ACD ADB ADC

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Safety Costs of Complete Tours ABCDA

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

ABCDA ABDCA ACBDA ACDBA ADBCA ADCBA

ABCDA ABDCA ACBDA ACDBA ADBCA ADCBA

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

ABCDA ABDCA ACBDA ACDBA ADBCA ADCBA

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Figure 2.5.1 An example of a constrained traveling salesman problem; cf. Example 2.5.1. We want to find a minimum cost tour that has safety cost less or equal to 10. The safety costs of the six possible tours are given in the table on the right. The (unconstrained) minimum cost tour, ABDCA, does not satisfy the safety constraint. The optimal constrained tour is ABCDA.

satisfies a safety constraint that the 'safety cost' of the tour should be less than a certain threshold; see Fig. 2.5.1. This constraint need not have the additive structure of Eq. (2.41). We are simply given a safety cost for each tour (see the table at the bottom right), which is calculated in a way that is of no further concern to us. In this example, for a tour to be admissible, its safety cost must be less than or equal to 10. Note that the (unconstrained) minimum cost tour, ABDCA, does not satisfy the safety constraint.

## Using a Base Heuristic for Constrained Rollout

We will now describe formally the constrained rollout algorithm. We assume the availability of a base heuristic, which for any given partial trajectory

<!-- formula-not-decoded -->

can produce a (complementary) partial trajectory

<!-- formula-not-decoded -->

AB

1

ACB

4

ACBD

20\

A

Terminal State t

Constraint:

ACD

20

ADB

ADC

ỹo

й1

X2

Tk

Xk+1 Xk+2

UN -1

О-

XN-1

Th (Tk, Uk) = (Tk, Uk, R(yk+1)) € C

˜

XN

Figure 2.5.2 The trajectory generation mechanism of the rollout algorithm. At stage k , and given the current partial trajectory

<!-- image -->

<!-- formula-not-decoded -->

which starts at ˜ x 0 and ends at ˜ x k , we consider all possible next states x k +1 = f k (˜ x k ↪ u k ), run the base heuristic starting at

<!-- formula-not-decoded -->

and form the complete trajectory T k (˜ y k ↪ u k ). Then the rollout algorithm:

- (b) Extends ˜ y k by ( ˜ u k ↪ f k (˜ x k ↪ ˜ u k ) ) to form ˜ y k +1 .
- (a) Finds ˜ u k , the control that minimizes the cost G ( T k (˜ y k ↪ u k ) ) over all u k for which the complete trajectory T k (˜ y k ↪ u k ) is feasible.

that starts at x k and satisfies the system equation

<!-- formula-not-decoded -->

Thus, given y k and any control u k , we can use the base heuristic to obtain a complete trajectory as follows:

- (a) Generate the next state x k +1 = f k ( x k ↪ u k ).
- (b) Extend y k to obtain the partial trajectory
- (c) Run the base heuristic from y k +1 to obtain the partial trajectory R ( y k +1 ).

<!-- formula-not-decoded -->

- (d) Join the two partial trajectories y k +1 and R ( y k +1 ) to obtain the complete trajectory ( y k ↪ u k ↪ R ( y k +1 ) ) , which is denoted by T k ( y k ↪ u k ):

This process is illustrated in Fig. 2.5.2. Note that the partial trajectory R ( y k +1 ) produced by the base heuristic depends on the entire partial trajectory y k +1 , not just the state x k +1 .

<!-- formula-not-decoded -->

ũk -1

Tk - 1

Tk

Uk+1

10 ....

...

..•

A complete trajectory T k ( y k ↪ u k ) of the form (2.42) is generally feasible for only the subset ˆ U k ( y k ) of controls u k that maintain feasibility:

<!-- formula-not-decoded -->

Our rollout algorithm starts from a given initial state ˜ y 0 = ˜ x 0 , and generates successive partial trajectories

<!-- formula-not-decoded -->

of the form

<!-- formula-not-decoded -->

where ˜ x k is the last state component of ˜ y k , and ˜ u k is a control that minimizes the heuristic cost G ( T k (˜ y k ↪ u k ) ) over all u k for which T k (˜ y k ↪ u k ) is feasible. Thus at stage k , the algorithm forms the set U k (˜ y k ) [cf. Eq. (2.43)] and selects from U k (˜ y k ) a control ˜ u k that minimizes the cost of the complete trajectory T k (˜ y k ↪ u k ):

<!-- formula-not-decoded -->

see Fig. 2.5.2. The objective is to produce a feasible final complete trajectory ˜ y N , which has a cost G (˜ y N ) that is no larger than the cost of R (˜ y 0 ) produced by the base heuristic starting from ˜ y 0 , i.e.,

<!-- formula-not-decoded -->

Note that T k (˜ y k ↪ u k ) is not guaranteed to be feasible for any given u k (i.e., may not belong to C ), but we will assume that the constraint set U k (˜ y k ) of problem (2.45) is nonempty, so that our rollout algorithm is welldefined. We will later modify our algorithm so that it is well-defined under the weaker assumption that just the complete trajectory generated by the base heuristic starting from the initial state ˜ y 0 is feasible , i.e., R (˜ y 0 ) ∈ Cglyph[triangleright]

## Constrained Rollout Algorithm

The algorithm starts at stage 0 and sequentially proceeds to the last stage. At the typical stage k , it has constructed a partial trajectory

<!-- formula-not-decoded -->

that starts at the given initial state ˜ y 0 = ˜ x 0 , and is such that

<!-- formula-not-decoded -->

The algorithm then forms the set of controls

<!-- formula-not-decoded -->

that is consistent with feasibility [cf. Eq. (2.43)], and chooses a control ˜ u k ∈ U k (˜ y k ) according to the minimization

<!-- formula-not-decoded -->

[cf. Eq. (2.45)], where

<!-- formula-not-decoded -->

[cf. Eq. (2.42)]. Finally, the algorithm sets

<!-- formula-not-decoded -->

[cf. Eq. (2.44)], thus obtaining the partial trajectory ˜ y k +1 to start the next stage.

It can be seen that our constrained rollout algorithm is not much more complicated or computationally demanding than its unconstrained version where the constraint T ∈ C is not present (as long as checking feasibility of a complete trajectory T is not computationally demanding). Note, however, that our algorithm makes essential use of the deterministic character of the problem, and does not admit a straightforward extension to stochastic problems, since checking feasibility of a complete trajectory is typically di ffi cult in the context of these problems.

The rollout algorithm just described is illustrated in Fig. 2.5.3 for our earlier traveling salesman Example 2.5.1. Here we want to find a minimum travel cost tour that additionally satisfies a safety constraint, namely that the 'safety cost' of the tour should be less than a certain threshold. This cost may be given by a table as in Fig. 2.5.3, or it may be generated on line with an algorithm for any given tour. Note that the minimum cost tour, ABDCA, in this example does not satisfy the safety constraint. Moreover, the tour ABCDA obtained by the rollout algorithm has barely smaller cost than the tour ACDBA generated by the base heuristic starting from A. In fact if the travel cost D → A were larger, say 25, the tour produced by constrained rollout would be more costly than the one produced by the base heuristic starting from A. This points to the need for a constrained

Rollout Choice

Heuristic.

AB

from AB 1

ABC

1

ABCD

4

ABD

3

ABDC

Stages Beyond Truncation Rollout Choice

Heuristic from AB

Matrix of Intercity

Travel Costs

5

1

20

20

1 20

20|

4

1

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

4

Matrix of Intercity Travel Costs

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Initial State xo

Rollout Choice

A

from A

ACD

Heuristic

5

Heuristic from AC

АСВ

4

1

AC

AD

Heuristic from AD

ADC

<!-- image -->

Heuristic Partial Tour

A AB AC AD ABC ABD ACB ACD ADB ADC

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Safety Costs of Complete Tours ABCDA

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

ABCDA ABDCA ACBDA ACDBA ADBCA ADCBA

ABCDA ABDCA ACBDA ACDBA ADBCA ADCBA

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Figure 2.5.3 The constrained traveling salesman problem; cf. Example 2.5.1, and its rollout solution using the base heuristic shown, which completes a partial tour as follows:

At A it yields ACDBA.

At AB it yields ABCDA.

At AC it yields ACBDA.

At AD it yields ADCBA.

This base heuristic is not assumed to have any special structure. It is just capable of completing every partial tour without regard to any additional considerations. Thus for example the heuristic generates at A the complete tour ACDBA, and it switches to the tour ACBDA once the salesman moves to AC.

At city A, the rollout algorithm:

- (a) Considers the partial tours AB, AC, and AD.
- (b) Uses the base heuristic to obtain the corresponding complete tours ABCDA, ACBDA, and ADCBA.
- (c) Discards ADCBA as being infeasible.
- (d) Compares the other two tours, ABCDA and ACBDA, finds ABCDA to have smaller cost, and selects the partial tour AB.
- (e) At AB, it considers the partial tours ABC and ABD.
- (f) It uses the base heuristic to obtain the corresponding complete tours ABCDA and ABDCA, and discards ABDCA as being infeasible.
- (g) It finally selects the complete tour ABCDA.

1

Terminal State t

20

ADB

version of the notion of sequential improvement and for a fortified variant of the algorithm, which we discuss next.

## Sequential Consistency, Sequential Improvement, and the Cost Improvement Property

We will now introduce sequential consistency and sequential improvement conditions guaranteeing that the control set U k (˜ y k ) in the minimization (2.47) is nonempty, and that the costs of the complete trajectories

<!-- formula-not-decoded -->

are improving with each k in the sense that

<!-- formula-not-decoded -->

while at the first step of the algorithm we have

<!-- formula-not-decoded -->

It will then follow that the cost improvement property holds.

<!-- formula-not-decoded -->

Definition 2.5.1: We say that the base heuristic is sequentially consistent if whenever it generates a partial trajectory

<!-- formula-not-decoded -->

starting from a partial trajectory y k , it also generates the partial trajectory

<!-- formula-not-decoded -->

starting from the partial trajectory y k +1 = ( y k ↪ u k ↪ x k +1 ) .

As we have noted in the context of unconstrained rollout, greedy heuristics tend to be sequentially consistent. Also any policy [a sequence of feedback control functions θ k ( y k ), k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1] for the DP problem of minimizing the terminal cost G ( y N ) subject to the system equation

<!-- formula-not-decoded -->

and the feasibility constraint y N ∈ C can be seen to be sequentially consistent. For an example where sequential consistency is violated, consider the base heuristic of the traveling salesman Example 2.5.1. From Fig. 2.5.3, it can be seen that the base heuristic at A generates ACDBA, but from AC it generates ACBDA, thus violating sequential consistency.

For a given partial trajectory y k , let us denote by y k ∪ R ( y k ) the complete trajectory obtained by joining y k with the partial trajectory generated by the base heuristic starting from y k . Thus if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we have

<!-- formula-not-decoded -->

Definition 2.5.2: We say that the base heuristic is sequentially improving if for every k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 and partial trajectory y k for which y k ∪ R ( y k ) ∈ C , the set ˆ U k ( y k ) is nonempty, and we have

<!-- formula-not-decoded -->

Note that for a base heuristic that is not sequentially consistent, the condition y k ∪ R ( y k ) ∈ C does not imply that the set ˆ U k ( y k ) is nonempty. The reason is that starting from the next state

<!-- formula-not-decoded -->

the base heuristic may generate a di ff erent trajectory than from y k , even if it applies u k at y k . Thus we need to include nonemptiness of ˆ U k ( y k ) as a requirement in the preceding definition of sequential improvement (in the fortified version of the algorithm to be discussed shortly, this requirement will be removed).

On the other hand, if the base heuristic is sequentially consistent, it is also sequentially improving. The reason is that for a sequentially consistent heuristic, y k ∪ R ( y k ) is equal to one of the trajectories contained in the set

<!-- formula-not-decoded -->

Our main result is contained in the following proposition.

Proposition 2.5.1: (Cost Improvement for Constrained Rollout) Assume that the base heuristic is sequentially improving and generates a feasible complete trajectory starting from the initial state ˜ y 0 = ˜ x 0 , i.e., R (˜ y 0 ) ∈ C . Then for each k , the set U k (˜ y k ) is nonempty, and we have

<!-- formula-not-decoded -->

where cf. Eq. (2.42). In particular, the final trajectory ˜ y N generated by the constrained rollout algorithm is feasible and has no larger cost than the trajectory R (˜ y 0 ) generated by the base heuristic starting from the initial state.

<!-- formula-not-decoded -->

Proof: Consider R (˜ y 0 ), the complete trajectory generated by the base heuristic starting from ˜ y 0 . Since ˜ y 0 ∪ R (˜ y 0 ) = R (˜ y 0 ) ∈ C by assumption, it follows from the sequential improvement definition, that the set U 0 (˜ y 0 ) is nonempty and we have

<!-- formula-not-decoded -->

[cf. Eq. (2.48)], while T 0 (˜ y 0 ↪ ˜ u 0 ) ∈ C .

The preceding argument can be repeated for the next stage, by replacing ˜ y 0 with ˜ y 1 , and R (˜ y 0 ) with T 0 (˜ y 0 ↪ ˜ u 0 ). Since ˜ y 1 ∪ R (˜ y 1 ) = T 0 (˜ y 0 ↪ ˜ u 0 ) ∈ C , from the sequential improvement definition, the set U 1 (˜ y 1 ) is nonempty and we have

[cf. Eq. (2.48)], while T 1 (˜ y 1 ↪ ˜ u 1 ) ∈ C . Similarly, the argument can be successively repeated for every k , to verify that U k (˜ y k ) is nonempty and that G ( T k (˜ y k ↪ ˜ u k ) ) ≥ G ( T k +1 (˜ y k +1 ↪ ˜ u k +1 ) ) for all k . Q.E.D.

<!-- formula-not-decoded -->

Proposition 2.5.1 establishes the fundamental cost improvement property for constrained rollout under the sequential improvement condition. On the other hand we may construct examples where the sequential improvement condition (2.48) is violated and the cost of the solution produced by rollout is larger than the cost of the solution produced by the

base heuristic starting from the initial state (cf. the unconstrained rollout Example 2.3.3).

In the case of the traveling salesman Example 2.5.1, it can be verified that the base heuristic specified in Fig. 2.5.3 is sequentially improving. However, if the travel cost D → A were larger, say 25, then it can be verified that the definition of sequential improvement would be violated at A, and the tour produced by constrained rollout would be more costly than the one produced by the base heuristic starting from A.

## The Fortified Rollout Algorithm and Other Variations

We will now discuss some variations and extensions of the constrained rollout algorithm. Let us first consider the case where the sequential improvement assumption is not satisfied. Then it may happen that given the current partial trajectory ˜ y k , the set of controls U k (˜ y k ) that corresponds to feasible trajectories T k (˜ y k ↪ u k ) [cf. Eq. (2.43)] is empty, in which case the rollout algorithm cannot extend the partial trajectory ˜ y k further. To bypass this di ffi culty, we introduce a fortified constrained rollout algorithm , patterned after the fortified algorithm given earlier. For validity of this algorithm, we require that the base heuristic generates a feasible complete trajectory R (˜ y 0 ) starting from the initial state ˜ y 0 .

The fortified constrained rollout algorithm, in addition to the current partial trajectory

<!-- formula-not-decoded -->

maintains a complete trajectory ˆ T k , called tentative best trajectory , which is feasible (i.e., ˆ T k ∈ C ) and agrees with ˜ y k up to state ˜ x k , i.e., ˆ T k has the form

<!-- formula-not-decoded -->

for some u k ↪ x k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ↪ x N such that

<!-- formula-not-decoded -->

Initially, ˆ T 0 is the complete trajectory R (˜ y 0 ), generated by the base heuristic starting from ˜ y 0 , which is assumed to be feasible. At stage k , the algorithm forms the subset ˆ U k (˜ y k ) of controls u k ∈ U k (˜ y k ) such that the corresponding T k (˜ y k ↪ u k ) is not only feasible, but also has cost that is no larger than the one of the current tentative best trajectory:

<!-- formula-not-decoded -->

There are two cases to consider at state k :

- (1) The set ˆ U k (˜ y k ) is nonempty . Then the algorithm forms the partial trajectory ˜ y k +1 = (˜ y k ↪ ˜ u k ↪ ˜ x k +1 ) ↪ where

<!-- formula-not-decoded -->

and sets T k (˜ y k ↪ ˜ u k ) as the new tentative best trajectory, i.e.,

<!-- formula-not-decoded -->

- (2) The set ˆ U k (˜ y k ) is empty . Then, the algorithm forms the partial trajectory ˜ y k +1 = ( ˜ y k ↪ ˜ u k ↪ ˜ x k +1 ) ↪ where

<!-- formula-not-decoded -->

and u k ↪ x k +1 are the control and state subsequent to ˜ x k in the current tentative best trajectory ˆ T k [cf. Eq. (2.49)], and leaves ˆ T k unchanged, i.e.,

<!-- formula-not-decoded -->

It can be seen that the fortified constrained rollout algorithm will follow the initial complete trajectory ˆ T 0 , the one generated by the base heuristic starting from ˜ y 0 , up to a stage k where it will discover a new feasible complete trajectory with smaller cost to replace ˆ T 0 as the tentative best trajectory. Similarly, the new tentative best trajectory ˆ T k may be subsequently replaced by another feasible trajectory with smaller cost, etc.

Note that if the base heuristic is sequentially improving, and the fortified rollout algorithm will generate the same complete trajectory as the (nonfortified) rollout algorithm given earlier, with the tentative best trajectory ˆ T k +1 being equal to the complete trajectory T k (˜ y k ↪ ˜ u k ) for all k . The reason is that if the base heuristic is sequentially improving, the controls ˜ u k generated by the nonfortified algorithm belong to the set ˆ U k (˜ y k ) [by Prop. 2.5.1, case (1) above will hold].

However, it can be verified that even when the base heuristic is not sequentially improving, the fortified rollout algorithm will generate a complete trajectory that is feasible and has cost that is no worse than the cost of the complete trajectory generated by the base heuristic starting from ˜ y 0 . This is because each tentative best trajectory has a cost that is no worse than the one of its predecessor, and the initial tentative best trajectory is just the trajectory generated by the base heuristic starting from the initial condition ˜ y 0 .

## Tree-Based Constrained Rollout Algorithms

It is possible to improve the performance of the rollout algorithm at the expense of maintaining more than one partial trajectory. In particular,

instead of the partial trajectory ˜ y k of Eq. (2.46), we can maintain a tree of partial trajectories that is rooted at ˜ y 0 . These trajectories need not have equal length, i.e., they need not involve the same number of stages. At each step of the algorithm, we select a single partial trajectory from this tree, and execute the rollout algorithm's step as if this partial trajectory were the only one. Let this partial trajectory have k stages and denote it by ˜ y k . Then we extend ˜ y k similar to our earlier rollout algorithm, with possibly multiple feasible trajectories. There is also a fortified version of this algorithm where a tentative best trajectory is maintained, which is the minimum cost complete trajectory generated thus far.

The aim of the tree-based algorithm is to obtain improved performance, essentially because it can go back and extend partial trajectories that were generated and temporarily abandoned at previous stages. The net result is a more flexible algorithm that is capable of examining more alternative trajectories. Note also that there is considerable freedom to select the number of partial trajectories maintained in the tree.

We finally mention a drawback of the tree-based algorithm: it is suitable for o ff -line computation, but it cannot be applied in an on-line context, where the rollout control selection is made after the current state becomes known as the system evolves in real-time .

## 2.5.1 Constrained Rollout for Discrete Optimization and Integer Programming

As noted in Section 2.1, general discrete optimization problems may be formulated as DP problems, which in turn can be addressed with rollout. The following is an example of a classical problem that involves both discrete and continuous variables. It can also be viewed as an instance of a 0-1 integer programming problem, and in fact this is the way it is usually addressed in the literature; see e.g., the book [DrH01]. The author's rollout book [Ber20a] contains additional examples.

## Example 2.5.2 (Facility Location)

We are given a candidate set of N locations, and we want to place in some of these locations a 'facility' that will serve the needs of a total of M 'clients.' Each client i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ M has a demand d i for services that may be satisfied at a location k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 at a cost a ik per unit. If a facility is placed at location k , it has capacity to serve demand up to a known level c k .

We introduce a 0-1 integer variable u k to indicate with u k = 1 that a facility is placed at location k at a cost b k and with u k = 0 that a facility is not placed at location k . Thus if y ik denotes the amount of demand of client i to be served at facility k , the constraints are

<!-- formula-not-decoded -->

Clients

Yik

Clients

Locations

Uk = 0 or 1

Clients

ABCDA ABDCA ACBDA ACDBA ADBCA ADCBA Cost Function:

Controlled Markov Chain Locations i=1 k=0

Set of Yik ≥ 0 such that

C: Set of (20, ..., UN-1) such that uk € {0, 1}

and can satisfy the demand and other constraints

(e.g., public policy constraints)

Cost Function:

M N-1

, UN -1) =

min

N-1

k=0

Figure 2.5.4 Schematic illustration of the facility location problem; cf. Example 2.5.2. Clients are matched to facilities, and the locations of the facilities are subject to optimization.

<!-- image -->

<!-- formula-not-decoded -->

together with

<!-- formula-not-decoded -->

We wish to minimize the cost

<!-- formula-not-decoded -->

subject to the preceding constraints; see Fig. 2.5.4. The essence of the problem is to place enough facilities at favorable locations to satisfy the clients' demand at minimum cost. This can be a very di ffi cult mixed integer programming problem.

On the other hand, when all the variables u k are fixed at some 0 or 1 values, the problem belongs to the class of linear transportation problems (see e.g., [Ber98]), and can be solved by fast polynomial algorithms. Thus the essential di ffi culty of the problem is how to select the integer variables u k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1. This can be viewed as a discrete optimization problem of the type discussed in Section 1.6.3 (cf. Fig. 1.6.2). In terms of the notation of this figure, the control components are u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 , where u k can take the values 0 or 1.

To address the problem suboptimally by rollout, we must define a base heuristic at a 'state' ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), where u j = 1 or u j = 0 specifies that a facility is or is not placed at location j , respectively. A suitable base heuristic at that state is to place a facility at all of the remaining locations (i.e., u j = 1 for j = k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1), and its cost is obtained by solving the corresponding linear transportation problem of minimizing the cost (2.53) subject to the constraints (2.50)-(2.52), with the variables u j , j = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ k , fixed at the previously chosen values, and the variables u j , j = k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N , fixed at 1.

G(40, •

∑

To illustrate, at the initial state where no placement decision has been made, we set u 0 = 1 (a facility is placed at location 0) or u 0 = 0 (a facility is not placed at location 0), we solve the two corresponding transportation problems, and we fix u 0 , depending on which of the two resulting costs is smallest. Having fixed the status of location 0, we repeat with location 1: set the variable u 1 to 1 and to 0, solve the corresponding two transportation problems, and fix u 1 , depending on which of the two resulting costs is smallest, etc.

It is easily seen that if the initial base heuristic choice (placing a facility at every candidate location) is feasible, i.e.,

<!-- formula-not-decoded -->

the rollout algorithm will yield a feasible solution with cost that is no larger than the cost corresponding to the initial application of the base heuristic. In fact it can be verified that the base heuristic here is sequentially consistent, so it is not necessary to use the fortified version of the algorithm. Regarding computational costs, the number of transportation problems to be solved is at first count 2 N , but it can be reduced to N +1 by exploiting the fact that one of the two transportation problems at each stage after the first has been solved at an earlier stage.

It is worth noting, for readers that are familiar with the integer programming method of branch-and-bound, that the graph of Fig. 2.1.4 corresponds to the branch-and-bound tree for the problem, so the rollout algorithm amounts to a quick (and imperfect) method to traverse the branch-and-bound tree. This observation may be useful if we wish to use integer programming techniques to add improvements to the rollout algorithm.

We finally note that the rollout algorithm requires the solution of many linear transportation problems, which are defined by fairly similar data. It is thus important to use an algorithm that is capable of using e ff ectively the final solution of one transportation problem as a starting point for the solution of the next. The auction algorithm for transportation problems (Bertsekas and Casta˜ non [BeC89]) is particularly well-suited for this purpose.

## Example 2.5.3 (Constrained Shortest Paths and Directed Spanning Trees)

Let us consider a spanning tree-type problem involving a directed graph with nodes 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N . At each node k ∈ ¶ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 ♦ there is a set of outgoing arcs u k ∈ U k . Node N is special: it is viewed as a 'root' node and has no outgoing arc. We are interested in collections of arcs involving a single outgoing arc per node,

<!-- formula-not-decoded -->

with u k ∈ U k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1. We require that these arcs do not form a cycle, so that u specifies a directed spanning tree that is rooted at node

3

2

5

6

0 1 2 3 4 5 6

<!-- image -->

/negationslash

Figure 2.5.5 Schematic illustration of a constrained shortest path problem with root node N = 6. Given the current feasible spanning tree solution (indicated with solid line arcs), the rollout algorithm, considers a node k (in the figure k = 0) and the spanning tree arcs ¶ u i ♣ i = k ♦ that are outgoing from the nodes i = k . It then considers the spanning trees that correspond to the outgoing arcs u k from k that do not close a cycle with the set ¶ u i ♣ i = k ♦ and are feasible [in the figure, these are the arcs indicated with broken lines, plus the arc (0,4)], and selects the arc that forms a spanning tree solution of minimum cost.

/negationslash

N . Note that for every node k , such a spanning tree specifies a unique path that starts at k , lies on the spanning tree, and ends at node N . We wish to find u that minimizes a given cost function G ( u ) subject to certain additional constraints, which we do not specify further. The set of all constraints on u (including the constraint that the arcs form a directed spanning tree) is denoted abstractly as u ∈ U , so the problem comes within our constrained optimization framework of this section.

Note that this problem contains as a special case the classical shortest path problem, where we have a length for every arc and the objective is to find a tree of shortest paths to node N from all the nodes 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1. Here U is just the constraint that the set of arcs u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ) form a directed spanning tree that is rooted at node N , and G ( u ) is the sum of the lengths of all the paths specified by u , summed over all the start nodes k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1. Other shortest path-type problems, involving constraints, are included as special cases. For example, there may be a constraint that all the paths to N that are specified by the spanning tree corresponding to u contain a number of arcs that does not exceed a given upper bound.

Suppose that we have an initial solution/directed spanning tree

<!-- formula-not-decoded -->

which is feasible (note here that finding such an initial solution may be a challenge). Let us apply the constrained rollout algorithm with a base heuristic that operates as follows: given a partial trajectory

<!-- formula-not-decoded -->

/negationslash

i.e., a sequence of k arcs, each outgoing from one of the nodes 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ k -1, it generates the complete trajectory/directed spanning tree

<!-- formula-not-decoded -->

Thus the rollout algorithm, given a partial trajectory

<!-- formula-not-decoded -->

considers the set ˆ U k (˜ y k ) of all outgoing arcs u k from node k , such that the complete trajectory

<!-- formula-not-decoded -->

is feasible. It then selects the arc u k ∈ ˆ U k (˜ y k ) that minimizes the cost

<!-- formula-not-decoded -->

see Fig. 2.5.5. It can be seen by induction, starting from ¯ u , that the set of arcs ˆ U k (˜ y k ) is nonempty, and that the algorithm generates a sequence of feasible solutions/spanning trees, each with cost no worse than the preceding one.

Note that throughout the rollout process, a rooted spanning tree is maintained, and at each stage k , a single arc ¯ u k that is outgoing from node k is replaced by the outgoing arc ˜ u k . Thus two successive rooted spanning trees generated by the algorithm, di ff er by at most a single arc.

An interesting aspect of this rollout algorithm is that it can be applied multiple times with the final solution of one rollout application used to specify the base heuristic of the next rollout application. Moreover, a di ff erent order of nodes may be used in each rollout application. This can be viewed as a form of policy iteration, of the type that we have discussed. The algorithm will eventually terminate, in the sense that it can make no further progress. More irregular/heuristic orders of node selections are also possible; for example some nodes may be selected multiple times before others will be selected for the first time. However, there is no guarantee that the final solution thus obtained will be optimal.

## 2.6 SMALL STAGE COSTS AND LONG HORIZON CONTINUOUS-TIME ROLLOUT

Let us consider the deterministic one-step approximation in value space scheme

<!-- formula-not-decoded -->

In the context of rollout, ˜ J k +1 ( f k ( x k ↪ u k ) ) is either the cost of the trajectory generated by the base heuristic starting from the next state f k ( x k ↪ u k ), or some approximation that may involve truncation and terminal cost function approximation, as in the truncated rollout scheme of Section 2.3.6.

There is a special di ffi culty in this context, which is often encountered in practice. It arises when the cost per stage g k ( x k ↪ u k ) is either 0 or is small relative to the cost-to-go approximation ˜ J k +1 ( f k ( x k ↪ u k ) ) . Then there is a potential pitfall to contend with: the cost approximation errors that are inherent in the term ˜ J k +1 ( f k ( x k ↪ u k ) ) may overwhelm the first stage cost term g k ( x k ↪ u k ), with unpredictable consequences for the quality of the one-step-lookahead policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ .

The most straightforward way to address this issue is to use longer lookahead; this is typically what is done in the context of MPC (cf. Section 1.6.9). The di ffi culty here is that long lookahead minimization may require extensive on-line computation. In this case, creative application of the lookahead tree pruning and incremental rollout ideas, discussed in Section 2.4.2, may be helpful.

We will next discuss the di ffi culty with small stage costs for an alternative context, which arises from discretization of a continuous-time optimal control problem.

## Continuous-Time Optimal Control and Approximation in Value Space

Consider a problem that involves a vector di ff erential equation of the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where x ( t ) ∈ /Rfractur n is the state vector at time t , ˙ x ( t ) ∈ /Rfractur n is the vector of first order time derivatives of the state at time t , u ( t ) ∈ U ⊂ /Rfractur m is the control vector at time t , where U is the control constraint set, and T is a given terminal time. Starting from a given initial state x (0), we want to find a feasible control trajectory { u ( t ) ♣ t ∈ [0 ↪ T ] } , which together with its corresponding state trajectory { x ( t ) ♣ t ∈ [0 ↪ T ] } , minimizes a cost function of the form where g represents cost per unit time, and G is a terminal cost function. This is a classical problem with a long history.

Let us consider a simple conversion of the preceding continuous-time problem to a discrete-time problem, while treading lightly over some of the associated mathematical fine points. We introduce a small discretization increment δ &gt; 0, such that T = δ N where N is a large integer, and we replace the di ff erential equation (2.55) by

<!-- formula-not-decoded -->

Here the function h k is given by

<!-- formula-not-decoded -->

where we view ¶ x k ♣ k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 ♦ and ¶ u k ♣ k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 ♦ as state and control trajectories, respectively, which approximate the corresponding continuous-time trajectories:

<!-- formula-not-decoded -->

We also replace the cost function (2.56) by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Thus the approximation in value space scheme with time discretization takes the form where ˜ J k +1 is the function that approximates the cost-to-go starting from a state at time k +1. We note here that the ratio of the terms δ · g k ( x k ↪ u k ) and ˜ J k +1 ( x k + δ · h k ( x k ↪ u k ) ) is likely to tend to 0 as δ → 0, since ˜ J k +1 ( x k + δ · h k ( x k ↪ u k ) ) ordinarily stays roughly constant at a nonzero level as δ → 0. This suggests that the one-step lookahead minimization may be degraded substantially by discretization, and other errors, including rollout truncation and terminal cost approximation. Note that a similar sensitivity to errors may occur in other discrete-time models that involve frequent selection of decisions, with cost per stage that is very small relative to the cumulative cost over many stages and/or the terminal cost.

<!-- formula-not-decoded -->

To deal with this di ffi culty, we subtract the constant ˜ J k ( x k ) in the one-step-lookahead minimization (2.57), and write

<!-- formula-not-decoded -->

since ˜ J k ( x k ) does not depend on u k , the results of the minimization are not a ff ected. Assuming ˜ J k is di ff erentiable with respect to its argument, we can write where ∇ x ˜ J k denotes the gradient of J k (a column vector), and prime denotes transposition. By dividing with δ , and taking informally the limit as δ → 0, we can write the one-step lookahead minimization (2.58) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ J t ( x ) is the continuous-time cost function approximation and ∇ x ˜ J t ( x ) is its gradient with respect to x . This is the correct analog of the approximation in value space scheme (2.54) for continuous-time problems.

## Rollout for Continuous-Time Optimal Control

In view of the value approximation scheme of Eq. (2.59), it is natural to speculate that the continuous-time analog of rollout with a base policy of the form where θ t ( x ( t ) ) ∈ U for all x ( t ) and t , has the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here J π ↪t ( x ( t ) ) is the cost of the base policy π starting from state x ( t ) at time t , and satisfies the terminal condition

<!-- formula-not-decoded -->

Computationally, the inner product in the right-hand side of the above minimization can be approximated using the finite di ff erence formula

<!-- formula-not-decoded -->

which can be calculated by running the base policy π starting from x ( t ) and from x ( t ) + δ · h ( x ( t ) ↪ u ( t ) ↪ t ) . (This finite di ff erencing operation may involve tricky computational issues, but we will not get into this.)

An important question is how to select the base policy π . A choice that is often sensible and convenient is to choose π to be a 'short-sighted' policy, which takes into account the 'short term' cost from the current state (say for a very small horizon starting from the current time t ), but ignores the remaining cost. An extreme case is the myopic policy, given by

<!-- formula-not-decoded -->

This policy is the continuous-time analog of the greedy policy that we discussed in the context of discrete-time problems, and the traveling salesman Example 1.2.3 in particular.

The following example illustrates the rollout algorithm (2.61) with a problem that has a special property: the base policy cost J π ↪t ( x ( t ) ) is independent of x ( t ) (it depends only on t ), so that ∇ x J π ↪t ( x ( t ) ) ≡ 0 glyph[triangleright] In this case, in view of Eq. (2.59), the rollout policy is myopic. It turns out that the optimal policy in this example is also myopic, so that the rollout policy is optimal, even though the base policy is very poor.

Given Point Given Line

Figure 2.6.1 Problem of finding a curve of minimum length from a given point to a given line, and its formulation as a calculus of variations problem.

<!-- image -->

## Example 2.6.1 (A Calculus of Variations Problem)

This is a simple example from the classical context of calculus of variations (see [Ber17a], Example 7.1.3). The problem is to find a minimum length curve that starts at a given point and ends at a given line. Without loss of generality, let (0 ↪ 0) be the given point, and let the given line be the vertical line that passes through ( T↪ 0), as shown in Fig. 2.6.1.

Let ( t↪ x ( t ) ) be the points of the curve, where 0 ≤ t ≤ T . The portion of the curve joining the points ( t↪ x ( t ) ) and ( t + dt↪ x ( t + dt ) ) can be approximated, for small dt , by the hypotenuse of a right triangle with sides dt and ˙ x ( t ) dt . Thus the length of this portion is which is equal to

The length of the entire curve is the integral over [0 ↪ T ] of this expression, so the problem is to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To reformulate the problem as a continuous-time optimal control problem, we introduce a control u and the system equation

<!-- formula-not-decoded -->

Our problem then takes the form

<!-- formula-not-decoded -->

This is a problem that fits our continuous-time optimal control framework, with

<!-- formula-not-decoded -->

Consider now a base policy π whereby the control depends only on t and not on x . Such a policy has the form

<!-- formula-not-decoded -->

where β ( t ) is some scalar function. For example, β ( t ) may be constant, β ( t ) ≡ ¯ β for some scalar ¯ β , which yields a straight line trajectory that starts at (0 ↪ 0) and makes an angle φ with the horizontal with tan( φ ) = ¯ β . The cost function of the base policy is

<!-- formula-not-decoded -->

which is independent of x ( t ), so that ∇ x J π ↪t ( x ( t ) ) ≡ 0. Thus, from the minimization of Eq. (2.61), we have

<!-- formula-not-decoded -->

and the rollout policy is

This is the optimal policy: it corresponds to the horizontal straight line that starts at (0 ↪ 0) and ends at ( T↪ 0).

<!-- formula-not-decoded -->

## Rollout with General Base Heuristics - Sequential Improvement

An extension of the rollout algorithm (2.61) is to use a more general base heuristic whose cost function H t ( x ( t ) ) can be evaluated by simulation. This rollout algorithm has the form

<!-- formula-not-decoded -->

Here the policy cost function J π ↪t is replaced by a more general di ff erentiable function H t , obtainable through a base heuristic, which may lack the sequential consistency property that is inherent in policies.

We will now show a cost improvement property of the rollout algorithm based on the natural condition

<!-- formula-not-decoded -->

and the assumption

<!-- formula-not-decoded -->

for all ( x ( t ) ↪ t ) , where ∇ x H t denotes gradient with respect to x , and ∇ t H t denotes gradient with respect to t . This assumption is the continuous-time analog of the sequential improvement condition of Definition 2.3.2 [cf. Eq. (2.13)]. Under this assumption, we will show that

<!-- formula-not-decoded -->

Indeed, let { ˜ x ( t ) ♣ t ∈ [0 ↪ T ] } and { ˜ u ( t ) ♣ t ∈ [0 ↪ T ] } be the state and control trajectories generated by the rollout policy starting from x (0). Then the sequential improvement condition (2.63) yields i.e., the cost of the rollout policy starting from the initial state x (0) is no worse than the base heuristic cost starting from the same initial state.

<!-- formula-not-decoded -->

for all t , and by integration over [0 ↪ T ], we obtain

<!-- formula-not-decoded -->

The second integral above can be written as

<!-- formula-not-decoded -->

and its integrand is the total di ff erential with respect to time: d dt ( H t ( ˜ x ( t ) ) ) glyph[triangleright] Thus we obtain from Eq. (2.65)

<!-- formula-not-decoded -->

Since

[cf. Eq. (2.62)] and ˜ x (0) = x (0), from Eq. (2.66) [which is a direct consequence of the sequential improvement condition (2.63)], it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

thus proving the cost improvement property (2.64).

Note that the sequential improvement condition (2.63) is satisfied if H t is the cost function J π ↪t corresponding to a base policy π . The reason is that for any policy π = { θ t ( x ( t )) ♣ 0 ≤ t ≤ T } [cf. Eq. (2.60)], the analog of the DP algorithm (under the requisite mathematical conditions) is

In continuous-time optimal control theory, this is known as the HamiltonJacobi-Bellman equation . It is a partial di ff erential equation, which may be viewed as the continuous-time analog of the DP algorithm for a single policy; there is also a Hamilton-Jacobi-Bellman equation for the optimal cost function J ∗ t ( x ( t ) ) (see optimal control textbook accounts, such as [Ber17a], Section 7.2, and the references cited there). As illustration, the reader may verify that the cost function of the base policy used in the calculus of variations problem of Example 2.6.1 satisfies this equation. It can be seen from the Hamilton-Jacobi-Bellman Eq. (2.67) that when H t = J π ↪t , the sequential improvement condition (2.63) and the cost improvement property (2.64) hold.

<!-- formula-not-decoded -->

## Approximating Cost Function Di ff erences

The preceding analysis suggests that when dealing with a discrete-time problem with a long horizon N , a system equation x k +1 = f k ( x k ↪ u k ), and a small cost per stage g k ( x k ↪ u k ) relative to the optimal cost-to-go function J ∗ k +1 ( f k ( x k ↪ u k ) ) , it is worth considering an alternative implementation of the approximation in value space scheme. In particular, we should consider approximating the cost di ff erences instead of approximating the optimal cost-to-go functions J ∗ k +1 ( f k ( x k ↪ u k ) ) . The one-step-lookahead minimization (2.54) should then be replaced by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ D k is the approximation to D ∗ k .

Note also that while for continuous-time problems, the idea of approximating the gradient of the optimal cost function is essential and comes out naturally from the analysis, for discrete-time problems, approximating cost-to-go di ff erences rather than cost functions is optional and should be considered in the context of a given problem, possibly in conjunction with increased lookahead. Methods along this line include advantage updating, cost shaping, biased aggregation, and the use of baselines, for which we refer to the books [BeT96], [Ber19a], and [Ber20a]. A special method to explicitly approximate cost function di ff erences is di ff erential training , which was proposed in the author's paper [Ber97b], and was also discussed in Section 4.3.4 of the book [Ber20a].

## The Case of Zero Cost per Stage

The most extreme and challenging case of small stage costs arises when the cost per stage is zero for all states, while a nonzero cost may be incurred only at termination. This type of cost structure occurs, among others, in games such as chess and backgammon (another interesting context is solving the Rubik's cube [AMS19], [MAS19]). It also occurs in several other contexts, including constraint programming problems (Section 2.1), where there is not even a terminal cost, just constraints to be satisfied.

Under these circumstances, the idea of approximating cost-to-go differences that we have just discussed may not be e ff ective, and applying approximation in value space may involve serious challenges. An advisable remedy is to resort to long lookahead, either through multistep lookahead minimization (possibly augmented by pruning or incremental rollout ideas; cf. Section 2.4), or through some form of truncated rollout.

It may also be important to introduce a terminal cost function approximation by using problem simplification (solving a simpler problem, in place of the original), or neural network training. Chess, Go, backgammon, and other games are prime examples of zero cost per stage problems, where the use of a terminal cost function approximation, obtained through a neural network training, has been a critical part of the solution methodology.

## 2.7 APPROXIMATION IN VALUE SPACE - STOCHASTIC PROBLEMS

We will now discuss approximation in value space for stochastic problems. We first focus on rollout algorithms for finite state, control, and disturbance spaces. We will restrict ourselves to the case where the base heuristic is a policy π = ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ . The rollout policy applies at state x k the control ˜ θ k ( x k ) given by the minimization

<!-- formula-not-decoded -->

Equivalently, the rollout policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ is obtained by minimization over the Q-factors Q k↪ π ( x k ↪ u k ) of the base policy:

<!-- formula-not-decoded -->

where

We first establish that the cost improvement property that we showed for deterministic problems under the sequential consistency condition carries through for stochastic problems. In particular, let us denote by J k↪ π ( x k )

<!-- formula-not-decoded -->

the cost corresponding to starting the base policy at state x k , and by J k↪ ˜ π ( x k ) the cost corresponding to starting the rollout algorithm at state x k . We claim that

<!-- formula-not-decoded -->

We prove this inequality by induction similar to the deterministic case [cf. Eq. (2.12)]. Clearly it holds for k = N , since

<!-- formula-not-decoded -->

Assuming that it holds for index k +1, we have for all x k ,

<!-- formula-not-decoded -->

where:

- (a) The first equality is the DP equation for the rollout policy ˜ π .
- (b) The first inequality holds by the induction hypothesis.
- (c) The second equality holds by the definition of the rollout algorithm.
- (d) The final equality is the DP equation for the base policy π .

The induction proof of the cost improvement property is thus complete.

The preceding cost improvement argument assumes that the cost functions J k +1 ↪ π of the base policy are calculated exactly. In practice, truncated rollout with terminal cost function approximation and limited simulation may be used to approximate J k +1 ↪ π . In this case the cost function of the rollout policy can still be viewed as the result of a Newton step in the context of an approximation in value space scheme. Moreover, the cost improvement property can still be proved under some conditions that we will not discuss in this book; see the books [Ber12], [Ber19a], and [Ber20a].

## Some Rollout Examples

Similar to deterministic problems, it has been observed empirically that for stochastic problems the rollout policy not only does not deteriorate

the performance of the base policy, but also typically produces substantial cost improvement, thanks to its underlying Newton step; see also the case studies referenced at the end of the chapter. To emphasize this point, we provide here an example of a nontrivial optimal stopping problem where the rollout policy is actually optimal, despite the fact that the base policy is rather naive. Such behavior is of course special and nontypical, but highlights the nature of the cost improvement property of rollout.

## Example 2.7.1 (Optimal Stopping and Rollout Optimality)

Optimal stopping problems are characterized by the availability, at each state, of a control that stops the evolution of the system. We will consider a problem with two control choices: at each stage we observe the current state of the system and decide whether to continue or to stop the process. We formulate this as an N -stage problem where stopping is mandatory at or before stage N .

Consider a stationary version of the problem (state and disturbance spaces, disturbance distribution, control constraint set, and cost per stage are the same for all times). At each state x k and at time k , if we stop, the system moves to a termination state at a cost C ( x k ) and subsequently remains there at no cost. If we do not stop, the system moves to state x k +1 = f ( x k ↪ w k ) at cost g ( x k ↪ w k ). The terminal cost, assuming stopping has not occurred by the last stage, is C ( x N ). An example is a problem of optimal exercise of a financial option where x is the asset's price, C ( x ) = x , and g ( x↪w ) ≡ 0.

The DP algorithm (for states other than the termination state) is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and it is optimal to stop at time k for states x in the set

<!-- formula-not-decoded -->

Consider now the rather primitive base policy π , whereby we stop at every state x . Thus we have for all x k and k ,

<!-- formula-not-decoded -->

The rollout policy is stationary and can be computed on-line relatively easily, since J k↪ π is available in closed form. In particular, the rollout policy is to stop at x k if

<!-- formula-not-decoded -->

i.e., if x k is in the set S N -1 , and otherwise to continue.

The rollout policy also has an intuitive interpretation: it stops at the states for which it is better to stop rather than continue for one more stage

and then stop. A policy of this type turns out to be optimal in several types of stopping applications. Let us provide a condition that guarantees its optimality.

We have from the DP Eqs. (2.70)-(2.71),

<!-- formula-not-decoded -->

and using this fact in the DP equation (2.71), we obtain inductively

<!-- formula-not-decoded -->

Using this fact and the definition of S k we see that

<!-- formula-not-decoded -->

We will now consider a condition guaranteeing that all the stopping sets S k are equal. Suppose that the set S N -1 is absorbing in the sense that if a state belongs to S N -1 and we decide to continue, the next state will also be in S N -1 :

<!-- formula-not-decoded -->

We will show that equality holds in Eq. (2.72) and for all k we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and using Eq. (2.73) we obtain for x ∈ S N -1

Therefore, stopping is optimal for all x N -2 ∈ S N -1 or equivalently S N -1 ⊂ S N -2 . This together with Eq. (2.72) implies S N -2 = S N -1 . Proceeding similarly, we obtain S k = S N -1 for all k . Thus the optimal policy is to stop if and only if the state is within the set S N -1 , which is precisely the set of states where the rollout policy stops.

<!-- formula-not-decoded -->

In conclusion, if condition (2.73) holds (the one-step stopping set S N -1 is absorbing), the rollout policy is optimal. Moreover, the preceding analysis [cf. Eq. (2.72)] can be used to show that even if the one-step stopping set S N -1 is not absorbing, the rollout policy stops and is optimal within the set of states x ∈ ∩ k S k , and correctly continues within the set of states x glyph[triangleleft] ∈ S N -1 . Contrary to the optimal policy, it also stops within the subset of states x ∈ S N -1 that are not in ∩ k S k . Thus, even in the absence of condition (2.73), the rollout policy is quite sensible even though the base policy is not.

We next discuss a special case of the preceding example. Again the one-step lookahead/rollout policy is optimal, despite the fact that the base policy is poor. Related examples can be found in Chapter 3 of the DP textbook [Ber17a].

## Example 2.7.2 (The Rational Burglar)

A burglar may at any night k choose to retire with his accumulated earnings x k or enter a house and bring home a random amount w k . However, in the latter case he gets caught with probability p , and then he is forced to terminate his activities and forfeit all of his earnings thus far. The amounts w k are independent, identically distributed with mean w . The problem is to find a policy that maximizes the burglar's expected earnings over N nights.

We can formulate this problem as a stopping problem with two actions (retire or continue) and a state space consisting of the real line, the retirement state, and a special state corresponding to the burglar getting caught. The DP algorithm is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The one-step stopping set is

<!-- formula-not-decoded -->

(more accurately this set together with the special state corresponding to the burglar's arrest). Since this set is absorbing in the sense of Eq. (2.73), we see that the one-step lookahead/rollout policy by which the burglar retires when his earnings reach or exceed (1 -p ) wglyph[triangleleft]p is optimal. Note that the base policy of the burglar is the 'timid' policy of always retiring, regardless of his accumulated earnings, which is far from optimal.

## 2.7.1 Simplified Rollout and Policy Iteration

The cost improvement property (2.68) also holds for the simplified version of the rollout algorithm (cf. Section 2.3.4) where the rollout policy is defined by

<!-- formula-not-decoded -->

for a subset U k ( x k ) ⊂ U k ( x k ) that contains the base policy control θ k ( x k ). The proof is obtained by replacing the last inequality in the argument of Eq. (2.69),

<!-- formula-not-decoded -->

with the inequality

<!-- formula-not-decoded -->

The simplified rollout algorithm (2.74) may be implemented in a number of ways, including control constraint discretization/approximation, a random search algorithm, or a one-agent-at-a-time minimization process, as in multiagent rollout.

The simplified rollout idea can also be used within the infinite horizon policy iteration (PI) context. In particular, instead of the minimization

<!-- formula-not-decoded -->

in the policy improvement operation, it is su ffi cient for cost improvement to generate a new policy ˜ θ that satisfies for all x ,

<!-- formula-not-decoded -->

This cost improvement property is the critical argument for proving convergence of the PI algorithm and its variations to the optimal cost function and policy; see the corresponding proofs in the books [Ber17a] and [Ber19a].

## 2.7.2 Certainty Equivalence Approximations

As in the case of deterministic DP problems, it is possible to use /lscript -step lookahead, with the aim to improve the performance of the policy obtained through approximation in value space. This, however, can be computationally expensive, because the lookahead graph expands fast as /lscript increases, due to the stochastic character of the problem. Using certainty equivalence (CE for short) is an important approximation approach for dealing with this di ffi culty, as it reduces the size of the /lscript -step lookahead graph. Moreover, CE mitigates the potentially excessive simulation because it reduces the stochastic variance of the Q-factors calculated at each stage.

In the pure but somewhat flawed version of this approach, when solving the /lscript -step lookahead minimization problem, we simply replace all of the uncertain quantities w k ↪ w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w k + /lscript -1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 by some nominal value w , thus making that problem fully deterministic. Unfortunately, this a ff ects significantly the character of the approximation: when w k is replaced by a deterministic quantity the Newton step interpretation of the underlying approximation in value space scheme is lost to a great extent.

Still, we may largely correct this di ffi culty, while retaining substantial simplification, by using CE for only after the first stage of the /lscript -step lookahead. We can do this with a CE scheme whereby only the uncertain quantities w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 are replaced by a deterministic value w , while w k is treated as a stochastic quantity.

Variants of the CE approach, based on less drastic simplifications of the probability distributions of the uncertain quantities, are given in the books [Ber17a], Section 6.2.2 and [Ber19a], Section 2.3.2.

The CE approach, first proposed in the paper by Bertsekas and Casta˜ non [BeC99], has an important property: it maintains the Newton step character of the approximation in value space scheme . In particular, the function J ˜ θ of the /lscript -step lookahead policy ˜ θ obtained is generated by a Newton step, applied to the function obtained by the last /lscript -1 minimization steps (modified by CE, and applied to the terminal cost function approximation); see the monograph [Ber20a] for a discussion. Thus the benefit of the fast convergence of Newton's method is restored. In fact based on insights derived from this Newton step interpretation, it appears that the performance penalty for the CE approximation is typically small. At the same time the /lscript -step lookahead minimization involves only one stochastic step, the first one, and hence potentially a much 'thinner' lookahead graph, than the /lscript -step minimization that does not involve any CE-type approximations; see Fig. 2.7.1. Moreover, the ideas of tree pruning and iterative deepening, which we have discussed in Section 2.4 for deterministic multistep lookahead, come into play when the CE approximation is used.

## Certainty Equivalence Approximations in Rollout

Certainty equivalence ideas can also be used in the context of rollout with a base policy. The cost function J θ ( f ( x↪ u↪ w ) ) of the base policy θ in the rollout control calculation

<!-- formula-not-decoded -->

[cf. Eq. (2.75)], can be approximated by the cost function

<!-- formula-not-decoded -->

of θ for a deterministic problem where the stochastic disturbances w are replaced by 'typical' deterministic quantities w . If w is a continuous random variable, then it is natural to use w = E ¶ w ♦ . If w is a discrete random variable, then one possibility is to use as w the most probable value of w under the state-control pair ( x↪ θ ( x ) ) . The resulting scheme can be viewed as an approximate rollout algorithm in the context of approximation in value space, but with exact minimization at the first step of lookahead [cf. Eq. (2.76)]. For this reason it still implements a Newton step for solving the Bellman equation and inherits the attendant cost improvement properties.

Another possibility, which applies to problems of finite horizon N , is to calculate the rollout control at state x k by computing the most likely future disturbance trajectory ( w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 ) under the base policy; cf. Section 2.3.7. Similar ideas also apply to truncated rollout schemes that incorporate CE approximations.

Xe-10

Ие-1

l' = 2l - 1

Xe!-

Ul' -1

Xe'

U1

oxo (Current State)

11Q

W1

X20

-Layer 1

- Layer 2

Without CE

Layer l - 1

<!-- image -->

1 Layer

With CE Lookahead Length Increases Without

1 Layer

Figure 2.7.1 Illustration of the computational savings with the CE approximation, applied at the states after the first layer of states of the multistep lookahead tree. The figure on the top (or the bottom) illustrates the lookahead tree without (or with, respectively) CE. It can be seen that with CE, the lookahead tree grows much faster (the layers contain more states). In particular, the 'height' of the /lscript -step lookahead graph without CE is the same as the 'height' of a /lscript ′ -step lookahead graph with CE, where /lscript ′ = 2 /lscript -1. Moreover, with a number m of controls per state, and a number n of disturbances per state-control pair, the number of leaves of the /lscript -step lookahead tree is estimated as O ( mn /lscript ) without CE and O ( m ( n + /lscript ) ) with CE.

## Example 2.7.3 (Markov Jump Parameter Problems)

Consider an N -stage horizon problem where the uncertainty in the system equation comes from a parameter θ k that is perfectly observed, but evolves according to a Markov chain with transition probabilities

<!-- formula-not-decoded -->

where 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r , are the possible values of the parameter. For example, in a maintenance-and-repair context, the system can be in one of two states, 'good' and 'bad', and the problem may be to schedule preventive maintenance of the system to maximize adherence to a certain production objective.

Stochastic Rollout uo axo (Current State)

U1

x2Q

Deterministic Rollout

To capture the correlation between successive parameter values, we introduce an augmented state ( x k ↪ θ k ), and policies θ k ( x k ↪ θ k ) that depend on this augmented state. To this end, we must define a stochastic dynamic system that describes the evolution from the augmented state ( x k ↪ θ k ) to the next augmented state ( x k +1 ↪ θ k +1 ). We thus introduce a stochastic variable w k that models the next value of the parameter, θ k +1 = w k , and changes according to the given Markov chain transition probabilities; see our discussion of state augmentation in Section 1.6.5. The augmented state evolves according to

<!-- formula-not-decoded -->

where w k takes the values i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r with probabilities p k i θ k ( x k ↪ u k ).

Suppose now that we want to apply rollout with a given base policy, which at state parameter pair ( x k ↪ θ k ) applies control θ k ( x k ↪ θ k ). To implement a CE approximation, we must calculate the base policy controls for the future time periods, k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, while assuming a deterministic trajectory of parameter values after θ k +1 . One possibility is to compute this trajectory as the most likely future trajectory ( w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 ) under the base policy. This computation must be done for each of the possible pairs ( x k +1 ↪ θ k +1 ), using the Markov chain probabilities that correspond to the state and control pairs that are generated by the base policy starting with ( x k +1 ↪ θ k +1 ).

## 2.7.3 Simulation-Based Implementation of the Rollout Algorithm

A conceptually straightforward way to compute the rollout control at a given state x k and time k is to consider each possible control u k ∈ U k ( x k ), and to generate a 'large' number of simulated trajectories of the system starting from ( x k ↪ u k ). Thus a simulated trajectory is obtained from

<!-- formula-not-decoded -->

where ¶ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ is the tail portion of the base policy, the starting state of the simulated trajectory is

<!-- formula-not-decoded -->

and the disturbance sequence ¶ w k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 ♦ is obtained by random sampling. The costs of the trajectories corresponding to a pair ( x k ↪ u k ) can be viewed as samples of the Q-factor

<!-- formula-not-decoded -->

where J k +1 ↪ π is the cost-to-go function of the base policy, i.e., J k +1 ↪ π ( x k +1 ) is the cost of using the base policy starting from x k +1 . For problems with a large number of stages, it is also common to truncate the rollout trajectories

Possible Moves Average Score by Monte-Carlo Simulation

Possible Moves Average Score by Monte-Carlo Simulation

Moves Average Score by Monte-Carlo Simulation

<!-- image -->

Possible Moves Average Score by Monte-Carlo Simulation

Figure 2.7.2 Illustration of rollout for backgammon. At a given position and roll of the dice, the set of all possible moves is generated, and the outcome of the game for each move is evaluated by 'rolling out' (simulating to the end) many games using a suboptimal/heuristic backgammon player (the TD-Gammon player was used for this purpose in [TeG96]), and by Monte Carlo averaging the scores. The move that results in the best average score is selected for play.

and add a terminal cost function approximation as compensation for the resulting error.

By Monte Carlo averaging of the costs of the sample trajectories plus the terminal cost (if any), we obtain an approximation to the Q-factor Q k↪ π ( x k ↪ u k ) for each u k ∈ U k ( x k ), denoted by ˜ Q k↪ π ( x k ↪ u k ) glyph[triangleright] We then compute the (approximate) rollout control ˜ θ k ( x k ) with the minimization

<!-- formula-not-decoded -->

## Example 2.7.4 (Backgammon)

The first impressive application of rollout was given for the ancient two-player game of backgammon, in the paper by Tesauro and Galperin [TeG96]; see Fig. 2.7.2. They implemented a rollout algorithm, which attained a level of play that was better than all computer backgammon programs, and eventually better than the best humans. Tesauro had proposed earlier the use of one-step and two-step lookahead with lookahead cost function approximation provided by a neural network, resulting in a backgammon program called TDGammon [Tes89a], [Tes89b], [Tes92], [Tes94], [Tes95], [Tes02]. TD-Gammon was trained with an approximate policy iteration method, and was used as the base policy (for each of the two players) to simulate game trajectories.

Possible Moves Average Score by Monte-Carlo Simulation

The rollout algorithm also involved truncation of long game trajectories, using a terminal cost function approximation based on TD-Gammon's position evaluation. Game trajectories are of course random, since they involve the use of dice at each player's turn. Thus the scores of many trajectories have to be generated and averaged with the Monte Carlo method to assess the probability of a win from a given position.

An important issue to consider here is that backgammon is a two-player game and not an optimal control problem that involves a single decision maker. While there is a DP theory for sequential zero-sum games, this theory has not been covered in this book. Thus how are we to interpret rollout algorithms in the context of two-player games, with both players using some base policy? The answer is to view the game as a (one-player) optimal control problem, where one of the two players passively uses the base policy exclusively (TD-Gammon in the present example). The other player takes the role of the optimizer, and actively tries to improve on his base policy (TDGammon) by using rollout. Thus 'policy improvement' in the context of the present example means that when playing against a TD-Gammon opponent, the rollout player achieves a better score on the average than if he/she were to play with the TD-Gammon strategy. In particular, the theory does not guarantee that a rollout player that is trained using TD-Gammon for both players will do better than TD-Gammon would against a non-TD-Gammon opponent. While this is a plausible practical hypothesis, it is one that can only be tested empirically. In fact relevant counterexamples have been constructed for the game of Go using 'adversarial' optimization techniques; see Wang et al. [WGB22], and also our discussion on minimax problems in Section 2.12.

Most of the currently existing computer backgammon programs descend from TD-Gammon. Rollout-based backgammon programs are the most powerful in terms of performance, consistent with the principle that a rollout algorithm performs better than its base heuristic. However, they are too timeconsuming for real-time play (without parallel computing hardware), because of the extensive on-line simulation requirement at each move. They have been used in a limited diagnostic way to assess the quality of neural networkbased programs (many articles and empirical works on computer backgammon are posted on-line; see e.g., http://www.bkgm.com/articles/page07.html).

## 2.7.4 Variance Reduction in Rollout - Comparing Advantages

When using simulation, sampling is often organized to e ff ect variance reduction . By this we mean that for a given problem, the collection and use of samples is structured so that the variance of the simulation error is made smaller, with the same amount of simulation e ff ort. There are several methods of this type for which we refer to textbooks on simulation (see, e.g., Ross [Ros12], and Rubinstein and Kroese [RuK1]).

The situation in backgammon is exacerbated by its high branching factor, i.e., for a given position, the number of possible successor positions is quite large, as compared for example with chess.

In this section we discuss a method to reduce the e ff ects of the simulation error in the calculation of the Q-factors in the context of rollout. The key idea is that the selection of the rollout control depends on the values of the Q-factor di ff erences

<!-- formula-not-decoded -->

for all pairs of controls ( u k ↪ ˆ u k ). These values must be computed accurately, so that the controls u k and ˆ u k can be accurately compared. On the other hand, the simulation/approximation errors in the computation of the individual Q-factors ˜ Q k↪ π ( x k ↪ u k ) may be magnified through the preceding di ff erencing operation.

An approach to counteract this type of simulation error magnification is to approximate the Q-factor di ff erence ˜ Q k↪ π ( x k ↪ u k ) -˜ Q k↪ π ( x k ↪ ˆ u k ) by sampling the di ff erence

<!-- formula-not-decoded -->

where w k = ( w k ↪ w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 ) is the same disturbance sequence for the two controls u k and ˆ u k , and

<!-- formula-not-decoded -->

with ¶ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ being the tail portion of the base policy.

For a simple example that illustrates how this form of variance reduction works, suppose we want to calculate the di ff erence q 1 -q 2 of two numbers q 1 and q 2 by subtracting two simulation samples s 1 = q 1 + w 1 and s 2 = q 2 + w 2 , where w 1 and w 2 are zero mean random variables. Then s 1 -s 2 is unbiased in the sense that its mean is equal to q 1 -q 2 . However, the variance of s 1 -s 2 decreases as the correlation of w 1 and w 2 increases. It is maximized when w 1 and w 2 are uncorrelated, and it is minimized (it is equal to 0) when w 1 and w 2 are equal.

The preceding example suggests a simulation scheme that is based on the di ff erence (2.78) and involves a common disturbance w k for u k and ˆ u k . In particular, it may be far more accurate than the one obtained by di ff erencing samples of C k ( x k ↪ u k ↪ w k ) and C k ( x k ↪ ˆ u k ↪ ˆ w k ), which involve two di ff erent disturbances w k and ˆ w k . Indeed, by introducing the zero mean sample errors

<!-- formula-not-decoded -->

For this to be possible, we need to assume that the probability distribution of each disturbance w i does not depend on x i and u i .

it can be seen that the variance of the error in estimating ˜ Q k↪ π ( x k ↪ u k ) -˜ Q k↪ π ( x k ↪ ˆ u k ) with the former method will be no larger than with the latter method if and only if

<!-- formula-not-decoded -->

By expanding the quadratic forms and using the fact E { D k ( x k ↪ u k ↪ w k ) } = 0, we see that this condition is equivalent to

<!-- formula-not-decoded -->

i.e., the errors D k ( x k ↪ u k ↪ w k ) and D k ( x k ↪ ˆ u k ↪ w k ) being nonnegatively correlated. A little thought should convince the reader that this property is likely to hold in many types of problems.

Roughly speaking, the relation (2.79) holds if changes in the value of u k (at the first stage) have little e ff ect on the value of the error D k ( x k ↪ u k ↪ w k ) relative to the e ff ect induced by the randomness of w k . To see this, suppose that there exists a scalar γ &lt; 1 such that, for all x k , u k , and ˆ u k , there holds

<!-- formula-not-decoded -->

Then we have, by using the generic relation ab ≥ a 2 -♣ a ♣ · ♣ b -a ♣ for two scalars a and b ,

<!-- formula-not-decoded -->

from which we obtain

<!-- formula-not-decoded -->

where for the second inequality we use the generic relation

<!-- formula-not-decoded -->

for two scalars a and b , and for the third inequality we use Eq. (2.80).

Thus, under the assumption (2.80), the condition (2.79) holds and guarantees that by averaging cost di ff erence samples rather than di ff erencing (independently obtained) averages of cost samples, the simulation error variance does not increase.

Let us finally note the potential benefit of using Q-factor di ff erences in contexts other than rollout. In particular when approximating Q-factors Q k↪ π ( x k ↪ u k ) using parametric architectures (Section 3.3 in Chapter 3), it may be important to approximate and compare instead the di ff erences

<!-- formula-not-decoded -->

The function A k↪ π ( x k ↪ u k ) is also known as the advantage of the pair ( x k ↪ u k ), and can serve just as well as Q k↪ π ( x k ↪ u k ) for the purpose of comparing controls, but may work better in the presence of approximation errors. The use of advantages will be discussed further in Chapter 3.

## 2.7.5 Monte Carlo Tree Search

In our earlier discussion of simulation-based rollout implementation, we implicitly assumed that once we reach state x k , we generate the same large number of trajectories starting from each pair ( x k ↪ u k ), with u k ∈ U ( x k ), to the end of the horizon. The drawback of this is threefold:

- (a) The trajectories may be too long because the horizon length N is large (or infinite, in an infinite horizon context).
- (b) Some of the controls u k may be clearly inferior to others, and may not be worth as much sampling e ff ort.
- (c) Some of the controls u k that appear to be promising, may be worth exploring better through multistep lookahead.

This has motivated multistep lookahead variants, generally referred to as Monte Carlo tree search (MCTS for short), which aim to trade o ff computational economy with a hopefully small risk of degradation in performance. Such variants involve, among others, early discarding of controls deemed to be inferior based on the results of preliminary calculations, and simulation that is limited in scope (either because of a reduced number of simulation samples, or because of a shortened horizon of simulation, or both).

A simple remedy for (a) above is to use rollout trajectories of reasonably limited length, with some terminal cost approximation at the end (in

an extreme case, the rollout may be skipped altogether for some states, i.e., rollout trajectories have zero length). The terminal cost function may be very simple (such as zero) or may be obtained through some auxiliary calculation. In fact the base policy used for rollout may be used to construct the terminal cost function approximation, as noted for the rollout-based backgammon algorithm of Example 2.7.4. In particular, an approximation to the cost function of the base policy may be obtained by training some approximation architecture, such as a neural network (see Chapter 3), and may be used as a terminal cost function.

A simple but less straightforward remedy for (b) is to use some heuristic or statistical test to discard some of the controls u k , as soon as this is suggested by the early results of simulation, or even before any simulation, based for example on the suggestions of a trained neural network. Similarly, to implement (c) one may use some heuristic to increase the length of lookahead selectively for some of the controls u k . This is similar to the incremental multistep rollout scheme for deterministic problems that we discussed in Section 2.4.2; see Fig. 2.4.6.

The MCTS approach can be based on sophisticated procedures for implementing and combining the remedies just described. The general idea is to use the interim results of the computation and statistical tests to focus the simulation e ff ort along the most promising directions. Thus to implement MCTS with multistep lookahead, one needs to maintain a lookahead tree, which is expanded as the relevant Q-factors are evaluated by simulation, and which balances the competing desires of exploitation and exploration (generate and evaluate controls that seem most promising in terms of performance versus assessing the potential of inadequately explored controls). Ideas that were developed in the context of multiarmed bandit problems have played an important role in the construction of this type of MCTS procedures (see the end-of-chapter references).

In the simple case of one-step lookahead, with Q-factors calculated by Monte Carlo simulation, MCTS fundamentally aims to find e ffi ciently the minimum of the expected values of a finite number of random variables. This is illustrated in the following example.

## Example 2.7.5 (Statistical Tests for Adaptive Sampling with One-Step Lookahead)

Let us consider a typical one-step lookahead selection strategy that is based on adaptive sampling. We are at a state x k and we try to find a control ˜ u k that minimizes an approximate Q-factor

<!-- formula-not-decoded -->

over u k ∈ U k ( x k ), with ˜ Q k ( x k ↪ u k ) computed by averaging samples of the expression within braces. We assume that U k ( x k ) contains m elements, which

-Factors Current State

Simulation Nearest Neighbor Heuristic Move to the Rig

Simulation Nearest Neighbor Heuristic Move to the Rig

Simulation Nearest Neighbor Heuristic Move to the Rig

Figure 2.7.3 Illustration of one-step lookahead MCTS at a state x k . The Q-factor sampled next corresponds to the control i with minimum sum of exploitation index (here taken to be the running average Q i↪n ) and exploration index ( R i↪n , possibly given by the UCB rule).

<!-- image -->

for simplicity are denoted 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m . At the /lscript th sampling period, knowing the outcomes of the preceding sampling periods, we select one of the m controls, say i /lscript , and we draw a sample of ˜ Q k ( x k ↪ i /lscript ), whose value is denoted by S i /lscript . Thus after the n th sampling period we have an estimate Q i↪n of the Q-factor of each control i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m that has been sampled at least once, given by where

<!-- formula-not-decoded -->

/negationslash

Thus Q i↪n is the empirical mean of the Q-factor of control i (total sample value divided by total number of samples), assuming that i has been sampled at least once.

<!-- formula-not-decoded -->

After n samples have been collected, with each control sampled at least once, we may declare the control i that minimizes Q i↪n as the 'best' one, i.e., the one that truly minimizes the Q-factor Q k ( x k ↪ i ). However, there is a positive probability that there is an error: the selected control may not minimize the true Q-factor. In adaptive sampling, roughly speaking, we want to design the sample selection strategy and the criterion to stop the sampling, in a way that keeps the probability of error small (by allocating some sampling e ff ort to all controls), and the number of samples limited (by not wasting samples on controls i that appear inferior based on their empirical mean Q i↪n ).

Intuitively, a good sampling policy will balance at time n the desires of exploitation and exploration (i.e., sampling controls that seem most promising, in the sense that they have a small empirical mean Q i↪n , versus assessing the potential of inadequately explored controls, those i that have been sampled a small number of times). Thus it makes sense to sample next the control i that minimizes the sum

<!-- formula-not-decoded -->

of two indexes: an exploitation index T i↪n and an exploration index R i↪n . Usually the exploitation index is chosen to be the empirical mean Q i↪n ; see Fig. 2.7.3. The exploration index is based on a confidence interval formula and depends on the sample count

<!-- formula-not-decoded -->

of control i . A frequently suggested choice is the UCB rule (upper confidence bound), which sets

<!-- formula-not-decoded -->

where c is a positive constant that is selected empirically (some analysis suggests values near c = √ 2, assuming that Q i↪n is normalized to take values in the range [ -1 ↪ 0]). The UCB rule, first proposed in the paper by Auer, CesaBianchi, and Fischer [ACF02], has been extensively discussed in the literature both for one-step and for multistep lookahead [where it is called UCT (UCB applied to trees; see Kocsis and Szepesvari [KoS06])].

Its justification is based on probabilistic analyses that relate to the multiarmed bandit problem, and is beyond our scope. Alternatives to the UCB formula have been suggested, and in fact in the AlphaZero program, the exploitation term has a di ff erent form than the one above, and depends on the depth of lookahead (see Silver et al. [SHS17]).

Sampling policies for MCTS with multistep lookahead are based on similar sampling ideas to the case of one-step lookahead. A simulated trajectory is run from a node i of the lookahead tree that minimizes the sum T i↪n + R i↪n of an exploitation index and an exploration index. There are several schemes of this type, but the details are beyond our scope and are often problem-dependent (see the end-of-chapter references).

A major success has been the use of MCTS in two-player game contexts, such as the AlphaGo program (Silver et al. [SHM16]), which performs better than the best humans in the game of Go. This program integrates several of the techniques discussed in this book, including MCTS and rollout using a base policy that is trained o ff -line using a deep neural network. The AlphaZero program, which has performed spectacularly well against humans and other programs in the games of Go and chess (Silver et al. [SHS17]), bears some similarity with AlphaGo, and critically relies on MCTS, but does not use rollout in its on-line playing mode (it relies primarily on very long lookahead). Let us also note that based on early tests, MCTS is not e ff ective in computer backgammon and it has not been adopted in commercial programs. More generally, the success of MCTS appears to be problem-dependent as well as implementation-dependent.

The paper [ACF02] refers to the rule given here as UCB1 and credits its motivation to the paper by Agrawal [Agr95]. The book by Lattimore and Szepesvari [LaS20] provides an extensive discussion of the UCB rule and its generalizations.

## 2.7.6 Randomized Policy Improvement by Monte Carlo Tree Search

We have described rollout and MCTS as schemes for policy improvement: start with a base policy, and compute an improved policy based on the results of one-step lookahead or multistep lookahead followed by simulation with the base policy. We have implicitly assumed that both the base policy and the rollout policy are deterministic in the sense that they map each state x k into a unique control ˜ θ k ( x k ) [cf. Eq. (2.77)]. In some (even nonstochastic) contexts, success has been achieved with randomized policies , which map a state x k to a probability distribution over the set of controls U k ( x k ), rather than mapping onto a single control. In particular, the AlphaGo and AlphaZero programs use MCTS to generate and use for training purposes randomized policies, which specify at each board position the probabilities with which the various moves are selected.

A randomized policy can be used as a base policy in a rollout context in exactly the same way as a deterministic policy: for a given state x k , we just generate sample trajectories and associated sample Q-factors, using probabilistically selected controls, starting from each leaf-state of the lookahead tree that is rooted at x k . We then average the corresponding Q-factor samples. The rollout/improved policy, as described here, is a deterministic policy, i.e., it applies at x k the control ˜ θ k ( x k ) that is 'best' according to the results of the rollout [cf. Eq. (2.77)]. Still, however, if we wish to generate an improved policy that is randomized, we can simply change the probabilities of di ff erent controls in the direction of the deterministic rollout policy. This can be done by increasing by some amount the probability of the 'best' control ˜ θ k ( x k ) from its base policy level, while proportionally decreasing the probabilities of the other controls.

The use of MCTS provides a related method to 'improve' a randomized policy. In the process of the adaptive simulation that is used in MCTS, we generate frequency counts of the di ff erent controls in U k ( x k ), i.e., the proportion of rollout trajectories associated with each u k ∈ U k ( x k ). We can then obtain the rollout randomized policy by moving the probabilities of the base policy in the direction suggested by the frequency counts, i.e., increase the probability of high-count controls and reduce the probability of the others. This type of policy improvement is reminiscent of gradient-type methods, and has been successful in some contexts; see the end-of-chapter references for such policy improvement implementations in AlphaGo, AlphaZero, and other applications.

## 2.8 ROLLOUT FOR INFINITE-SPACES PROBLEMS OPTIMIZATION HEURISTICS

We have considered so far finite control space applications of rollout, so there is a finite number of relevant Q-factors at each state x k , which are

evaluated by simulation and are exhaustively compared. When the control constraint set is infinite, to implement this approach the constraint set must be replaced by a finite set, obtained by some form of discretization or random sampling, which can be inconvenient and ine ff ective. In this section we will discuss an alternative approach to deal with an infinite number of controls and Q-factors at x k . The idea is to use a base heuristic that involves a continuous optimization , and to rely on a linear or nonlinear programming method to solve the corresponding lookahead optimization problem.

## 2.8.1 Rollout for Infinite-Spaces Deterministic Problems

To develop the basic idea of how to deal with infinite control spaces, we first consider deterministic problems, involving a system x k +1 = f k ( x k ↪ u k ) ↪ and a cost per stage g k ( x k ↪ u k ). The rollout minimization is

<!-- formula-not-decoded -->

where ˜ Q k ( x k ↪ u k ) is the approximate Q-factor

<!-- formula-not-decoded -->

with H k +1 ( x k +1 ) being the cost of the base heuristic starting from state x k +1 [cf. Eq. (2.10)]. Suppose that we have a di ff erentiable closed-form expression for H k +1 , and the functions g k and f k are known and are di ff erentiable with respect to u k . Then the Q-factor ˜ Q k ( x k ↪ u k ) of Eq. (2.82) is also di ff erentiable with respect to u k , and its minimization (2.81) may be addressed with one of the many gradient-based methods that are available for di ff erentiable unconstrained and constrained optimization.

The preceding approach requires that the heuristic cost H k +1 ( x k +1 ) can be di ff erentiated, so it should either be available in closed form, which is quite restrictive, or that it can be di ff erentiated numerically, which may be inconvenient and/or unreliable. These di ffi culties can be circumvented by using a base heuristic that is itself based on multistep optimization . In particular, suppose that H k +1 ( x k +1 ) is the optimal cost of some ( /lscript -1)stage deterministic optimal control problem that is related to the original problem. Then the rollout algorithm (2.81)-(2.82) can be implemented by solving the /lscript -stage deterministic optimal control problem, which seamlessly concatenates the first stage minimization over u k [cf. Eq. (2.81)], with the ( /lscript -1) -stage minimization of the base heuristic ; see Fig. 2.8.1. This /lscript -stage problem may be solvable on-line by standard continuous spaces nonlinear

¿ Minto Lamaron that fon thia to

Next States

Xk+1

Current State

Control

Stage k

Base Heuristic

Minimization

-Factors Current State k+1,..., k+l-1

Current State

Figure 2.8.1 Schematic illustration of rollout for a deterministic problem with infinite control spaces. The base heuristic is to solve an ( /lscript -1)-stage deterministic optimal control problem, which together with the k th stage minimization over u k ∈ U k ( x k ), seamlessly forms an /lscript -stage continuous spaces optimal control/nonlinear programming problem that starts at state x k .

<!-- image -->

programming or optimal control methods. A major paradigm of methods of this type is model predictive control, which we have discussed in Chapter 1 (cf. Section 1.6.9). In the present section we will discuss a few other possibilities. The following is a simple example of an important class of inventory storage and supply chain management processes.

## Example 2.8.1 (Supply Chain Management)

Let us consider a supply chain system, where a certain item is produced at a production center and fulfilled at a retail center. Stock of the item is shipped from the production center to the retail center, where it arrives with a delay of τ ≥ 1 time units, and is used to fulfill a known stream of demands d k over an N -stage horizon; see Fig. 2.8.2. We denote:

- x 1 k : The stock at hand at the production center at time k .
- x 2 k : The stock at hand at the retail center at time k , and used to fulfill demand (both positive and negative x 2 k are allowed; a negative value indicates that there is backordered demand).

Note, however, that for this to be possible, it is necessary to have a mathematical model of the system; a simulator is not su ffi cient. Another di ffi culty occurs when the control space is the union of a discrete set and a continuous set. Then it may be necessary to use some type of mixed integer programming technique to solve the /lscript -stage problem. Alternatively, it may be possible to handle the discrete part by brute force enumeration, followed by continuous optimization.

States

Xk+l

Controller Production Center Delay Retail Storage Demand

<!-- image -->

Controller Production Center Delay Retail Storage Demand

Figure 2.8.2. Illustration of a simple supply chain system for Example 2.8.1.

u 1 k : The amount produced at time k .

u 2 k : The amount shipped at time k (and arriving at the retail center τ time units later).

The state at time k is the stock available at the production and retail centers, x 1 k ↪ x 2 k , plus the stock amounts that are in transit and have not yet arrived at the retail center u 2 k -τ -1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u 2 k -1 . The control u k = ( u 1 k ↪ u 2 k ) is chosen from some constraint set that may depend on the current state, and is subject to production capacity and transport availability constraints. The system equation is

<!-- formula-not-decoded -->

and involves the delayed control component u 2 k -τ . Thus the exact DP algorithm involves state augmentation as introduced in Section 1.6.5, and may thus be much more complicated than in the case where there are no delays.

The cost at time k consists of three components: a production cost that depends on x 1 k and u 1 k , a transportation cost that depends on u 2 k , and a fulfillment cost that depends on x 2 k [which includes positive costs for both excess inventory (i.e., x 2 k &gt; d k ) and for backordered demand (i.e., x 2 k &lt; d k )]. The precise forms of these cost components are immaterial for the purposes of this example.

Here the control vector u k is often continuous (or a mixture of discrete and continuous components), so it may be essential for the purposes of rollout to use the continuous optimization framework of this section. In particular, at the current stage k , we know the current state, which includes x 1 k , x 2 k , and the amounts of stock in transit together with their scheduled arrival times at the retail center. We then apply some heuristic optimization to determine the stream of future production and shipment levels over /lscript steps, and use the first component of this stream as the control applied by rollout. As an example we may use as base policy one that brings the retail inventory to some target value /lscript stages ahead, and possibly keep it at that value for a portion of the remaining periods. This is a nonlinear programming or mixed integer programming problem that may be solvable with available software far more e ffi ciently than by a discretized form of DP.

Despite the fact that with large delays, the size of the augmented state space can become very large (cf. Section 1.6.5), the implementation of rollout schemes is not a ff ected much by this increase in size. For this reason, rollout can be very well suited for problems involving delayed e ff ects of past states and controls.

A major benefit of rollout in the supply chain context is that it can readily incorporate on-line replanning. This is necessary when unexpected demand changes, production or transport equipment failures occur, or updated forecasts become available.

The following example deals with a common class of problems of resource allocation over time.

## Example 2.8.2 (Multistage Linear and Mixed Integer Programming)

Let us consider a deterministic optimal control problem with linear system equation

<!-- formula-not-decoded -->

where A k and B k are known matrices of appropriate dimension, d k is a known vector, and x k and u k are column vectors. The cost function is linear of the form

<!-- formula-not-decoded -->

where c k and d k are known column vectors of appropriate dimension, and a prime denotes transpose. The terminal state and state-control pairs ( x k ↪ u k ) are constrained by

<!-- formula-not-decoded -->

where T and P k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 ↪ are given sets, which are specified by linear and possibly integer constraints.

As an example, consider a multi-item production system, where the state is x k = ( x 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x n k ) and x i k represents stock of item i available at the start of period k . The state evolves according to the system equation

<!-- formula-not-decoded -->

where u ij k is the amount of product i that is used during time k for the manufacture of product j , a ij k are known scalars that are related to the underlying production process, and d i k is a deterministic demand of product i that is fulfilled at time k . One constraint here is that

<!-- formula-not-decoded -->

and there are additional linear and integer constraints on ( x k ↪ u k ), which are collected in a general constraint of the form ( x k ↪ u k ) ∈ P k (e.g., nonnegativity, production capacity, storage constraints, etc). Note that the problem

may be further complicated by production delays, as in the preceding supply chain Example 2.8.1. Moreover, while in this section we focus on deterministic problems, we may envision a stochastic version of the problem where the demands d i k are random with given probability distributions, which are subject to revisions based on randomly received forecasts.

The problem may be solved using a linear or mixed integer programming algorithm, but this may be very time-consuming when N is large. Moreover, the problem will need to be resolved on-line if some of the problem data changes and replanning is necessary. A suboptimal alternative is to use truncated rollout with an /lscript -stage mixed integer optimization, and a polyhedral terminal cost function ˜ J k + /lscript to provide a terminal cost optimization. A simple possibility is no terminal cost [ ˜ J k + /lscript ( x k + /lscript ) ≡ 0], and another possibility is a polyhedral lower bound approximation that can be based on relaxing the integer constraints after stage k + /lscript , or some kind of training approach that uses data.

We will next discuss how rollout can accommodate stochastic disturbances by using deterministic optimization ideas based on certainty equivalence and the methodology of stochastic programming.

## 2.8.2 Rollout Based on Stochastic Programming

We have focused so far in this section on rollout that relies on deterministic continuous optimization. There is an important class of methods, known as stochastic programming , which can be used for stochastic optimal control, but bears a close connection to continuous spaces deterministic optimization. We will first describe this connection for two-stage problems, then discuss extensions to many-stages problems, and finally show how rollout can be brought to bear for their approximate solution.

## Example 2.8.3 (Two-Stage Stochastic Programming)

Consider a stochastic problem of optimal decision making over two stages: In the first stage we will choose a finite-dimensional vector u 0 from a subset U 0 with cost g 0 ( u 0 ). Then an uncertain event represented by a random variable w 0 will occur, whereby w 0 will take one of the values w 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w m with corresponding probabilities p 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ p m . Once w 0 occurs, we will know its value w i , and we must then choose at the second stage a vector u i 1 from a subset U 1 ( u 0 ↪ w i ) at a cost g 1 ( u i 1 ↪ w i ). The objective is to minimize the expected cost

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

subject to

1st Stage

йо

wl wl

w2

and can satisfy the demand and other constraints 1st Stage 2nd Stage

U1

U1

<!-- image -->

can satisfy the demand and other constraints 1st Stage 2nd S

Figure 2.8.3. Illustration of the DP problem associated with two-stage stochastic programming; cf. Example 2.8.3. The figure depicts the case where each variable u 0 , w 0 , and u 1 can take only two values. A similar conversion to a DP problem is possible for a multistage stochastic programming problem, involving multiple choices of decisions, each followed by an uncertain event whose outcome is perfectly observed by the decision maker.

We can view this problem as a two-stage DP problem, where x 1 = w 0 is the system equation, the disturbance w 0 can take the values w 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w m with probabilities p 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ p m , the cost of the first stage is g 0 ( u 0 ), the cost of the second stage is g 1 ( x 1 ↪ u 1 ), and the terminal cost is 0. The intuitive meaning is that since at time 0 we don't know yet which of the m values w i of w 0 will occur, we must calculate (in addition to u 0 ) a separate second stage decision u i 1 for each i , which will be used after we know that the value of w 0 is w i .

However, if u 0 and u 1 take values in a continuous space such as the Euclidean spaces /Rfractur d 0 and /Rfractur d 1 , respectively, we can also equivalently view the problem as a nonlinear programming problem of dimension ( d 0 + md 1 ) (the optimization variables are u 0 and u i 1 , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m ).

For a generalization of the preceding example, consider the stochastic DP problem of Section 1.3 for the case where there are only two stages, and the disturbances w 0 and w 1 can independently take one of the m values w 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w m with corresponding probabilities p 1 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ p m 0 and p 1 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ p m 1 , respectively. The optimal cost function J 0 ( x 0 ) is given by the two-stage

DP algorithm

<!-- formula-not-decoded -->

By bringing the inner minimization outside the inner brackets, we see that this DP algorithm is equivalent to solving the nonlinear programming problem

<!-- formula-not-decoded -->

If the controls u 0 and u i 1 are elements of /Rfractur d , this problem involves d (1 + m ) scalar variables. An example is the multi-item production problem described in Example 2.8.2 in the case where the demands w i k and/or the production coe ffi cients a ij k are stochastic.

We can also consider an N -stage stochastic optimal control problem. A similar reformulation as a nonlinear programming problem is possible. It converts the N -stage stochastic problem into a deterministic optimization problem of dimension that grows exponentially with the number of stages N . In particular, for an N -stage problem, the number of control variables expands by a factor m with each additional stage. The total number of variables is bounded by

<!-- formula-not-decoded -->

where m is the maximum number of values that a disturbance can take at each stage and d is the dimension of the control vector.

## 2.8.3 Stochastic Rollout with Certainty Equivalence

The dimension of the preceding nonlinear programming formulation of the multistage stochastic optimal control problem with continuous control spaces can be very large. This motivates a variant of a rollout algorithm

that relies on a stochastic optimization for the current stage, and a deterministic optimization that relies on (assumed) certainty equivalence for the remaining stages, where the base policy is used. In this way, the dimension of the nonlinear programming problem to be solved by rollout is drastically reduced.

This rollout algorithm operates as follows: Given a state x k and control u k ∈ U k ( x k ), we consider the next states x i k +1 that correspond to the m possible values w i k , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , which occur with the known probabilities p i k , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m . We then consider the approximate Q-factors

<!-- formula-not-decoded -->

where ˜ H k +1 ( x i k +1 ) is the cost of a base policy, which starting at stage k +1 from

<!-- formula-not-decoded -->

optimizes the cost-to-go starting from x i k +1 , while assuming that the future disturbances w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 , will take some nominal (nonrandom) values w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 . The rollout control ˜ θ k ( x k ) computed by this algorithm is

<!-- formula-not-decoded -->

Note that this rollout algorithm does not have the cost improvement property, because it involves an approximation: the cost ˜ H k +1 ( x i k +1 ) used in Eq. (2.84) is an approximation to the cost of a policy. It is the cost of a policy applied to the certainty equivalent version of the original stochastic problem.

The key fact now is that the problem (2.85) can be viewed as a seamless ( N -k )-stage deterministic optimization, which involves the control u 0 , and for each value w i k of the disturbance w k , the sequence of controls ( u i k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u i N -1 ). If the controls are elements of /Rfractur d , this deterministic optimization involves a total of

<!-- formula-not-decoded -->

scalar variables. Currently available deterministic optimization software can deal with quite large numbers of variables, particularly in the context of linear programming, so by using rollout in combination with certainty equivalence, very large problems with continuous state and control variables may be addressed. We refer to the paper by Hu et al. [HWP22] for an application of this idea to problems of maintenance scheduling.

Another possibility is to use multistep lookahead that aims to represent better the stochastic character of the uncertainty. Here at state x k we solve an ( N -k )-stage optimal control problem, where the uncertainty

is fully taken into account in the first /lscript stages, similar to stochastic programming, and in the remaining N -k -/lscript stages, the uncertainty is dealt with by certainty equivalence, by fixing the disturbances w k + /lscript ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 at some nominal values (we assume here for simplicity that /lscript &lt; N -k ). If the controls are elements of /Rfractur d , and the number of values that the disturbances w 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 can take is m , the total number of control variables of this problem is

<!-- formula-not-decoded -->

[this is the /lscript -step lookahead generalization of the formula (2.86)]. Once the optimal policy ¶ ˜ u k ↪ ˜ θ k +1 ↪ ˜ θ k +2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ for this problem is obtained, the first control component ˜ u k is applied at x k and the remaining components ¶ ˜ θ k +1 ↪ ˜ θ k +2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ are discarded. Note also that this multistep lookahead approach may be combined with the ideas of multiagent rollout, which will be discussed in the next section.

## 2.9 MULTIAGENT ROLLOUT

We will now consider a special structure of the control space, whereby the control u k consists of m components, u k = ( u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k ), with a separable control constraint structure u /lscript k ∈ U /lscript k ( x k ), /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m . The control constraint set is the Cartesian product

<!-- formula-not-decoded -->

Conceptually, each component u /lscript k , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , is chosen at stage k by a separate 'agent' (a decision making entity), and for the sake of the following discussion, we assume that each set U /lscript k ( x k ) is finite. We discussed this type of problem briefly in Section 1.6.7, and we will discuss it in this section in greater detail.

The one-step lookahead minimization

<!-- formula-not-decoded -->

where π is a base policy, involves as many as n m Q-factors, where n is the maximum number of elements of the sets U /lscript k ( x k ) [so that n m is an upper bound to the number of controls in U k ( x k ), in view of the Cartesian product structure (2.87)]. As a result, the standard rollout algorithm requires an exponential [order O ( n m )] number of base policy cost computations per stage, which can be overwhelming even for moderate values of m .

This motivates an alternative and more e ffi cient rollout algorithm, called multiagent rollout also referred to as agent-by-agent rollout , that still achieves the cost improvement property

<!-- formula-not-decoded -->

) Random cost

Figure 2.9.1 Equivalent formulation of the N -stage stochastic optimal control problem for the case where the control u k consists of m components u 1 k ↪ u 2 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k :

<!-- image -->

<!-- formula-not-decoded -->

cf. Section 1.6.7. The figure depicts the k th stage transitions. Starting from state x k , we generate the intermediate states

<!-- formula-not-decoded -->

using the respective controls u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m -1 k . The final control u m k leads from ( x k ↪ u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m -1 k ) to

<!-- formula-not-decoded -->

and a stage cost g k ( x k ↪ u k ↪ w k ) is incurred. All of the preceding transitions, which involve the controls u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m -1 k , incur zero cost.

where J k↪ ˜ π ( x k ), k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N , is the cost-to-go of the rollout policy ˜ π starting from state x k . Indeed we will exploit the multiagent structure to construct an algorithm that maintains the cost improvement property at much smaller computational cost, namely requiring order O ( nm ) base policy cost computations per stage.

A key idea here is that the computational requirements of the rollout one-step minimization (2.88) are proportional to the size of the control space and are independent of the size of the state space. We consequently reformulate the problem so that control space complexity is traded o ff with state space complexity, as discussed in Section 1.6.7. This is done by 'unfolding' the control u k into its m components u 1 k ↪ u 2 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k . At the same time, between x k and the next state x k +1 = f k ( x k ↪ u k ↪ w k ), we introduce artificial intermediate 'states' and corresponding transitions; see Fig. 2.9.1, given in Section 1.6.7 and repeated here for convenience.

It can be seen that this reformulated problem is equivalent to the original, since any control choice that is possible in one problem is also possible in the other problem, while the cost structure of the two problems is the same: each policy of the reformulated problem corresponds to a policy of the original problem, with the same cost function, and reversely.

A fine point here is that policies of the original problem involve functions

Consider now the standard rollout algorithm applied to the reformulated problem of Fig. 2.9.1, with a given base policy π = ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ ↪ which is also a policy of the original problem [so that θ k = ( θ 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m k ), with each θ /lscript k , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , being a function of just x k ]. The algorithm involves a minimization over only one control component at the states x k and at the intermediate states

<!-- formula-not-decoded -->

In particular, for each stage k , the algorithm requires a sequence of m minimizations, once over each of the agent controls u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k , with the past controls determined by the rollout policy, and the future controls determined by the base policy. Assuming a maximum of n elements in the constraint sets U /lscript k ( x k ), the computation required at each stage k is of order O ( n ) for each of the 'states'

<!-- formula-not-decoded -->

for a total of order O ( nm ) computation.

To elaborate, at ( x k ↪ u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u /lscript -1 k ) with /lscript ≤ m , and for each of the controls u /lscript k ∈ U /lscript k ( x k ), we generate by simulation a number of system trajectories up to stage N , with all future controls determined by the base policy. We average the costs of these trajectories, thereby obtaining the Q -factors corresponding to ( x k ↪ u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u /lscript -1 k ↪ u /lscript k ), for all values u /lscript k ∈ U /lscript k ( x k ) (with the preceding controls u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u /lscript -1 k held at the values computed earlier, and the future controls u /lscript +1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k ↪ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 determined by the base policy). We then select the control u /lscript k ∈ U /lscript k ( x k ) that corresponds to the minimal Q -factor.

Prerequisite assumptions for the preceding algorithm to work in an on-line multiagent setting are:

- (a) All agents have access to the current state x k as well as the base policy (including the control functions θ /lscript n , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , n = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 of all agents).
- (b) There is an order in which agents compute and apply their local controls.
- (c) The agents share their information, so agent /lscript knows the local controls u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u /lscript -1 k computed by the predecessor agents 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ /lscript -1 in the given order.

of x k , while policies of the reformulated problem involve functions of the choices of the preceding agents, as well as x k . However, by successive substitution of the control functions of the preceding agents, we can view control functions of each agent as depending exclusively on x k . It follows that the multi-transition structure of the reformulated problem cannot be exploited to reduce the cost function beyond what can be achieved with a single-transition structure.

Spider 1 Spider 2 Fly 1 Fly 2

Figure 2.9.2 Illustration of the two-spiders and two-flies problem. The spiders move along integer points of a line. The two flies stay still at some integer locations. The character of the optimal policy is to move the two spiders towards two di ff erent flies.

<!-- image -->

Multiagent rollout with the given base policy starts with spider 1 at location n , and calculates the two Q-factors of moving to locations n -1 and n + 1, assuming that the remaining moves of the two spiders will be made using the go-towards-the-nearest-fly base policy. The Q-factor of going to n -1 is smallest because it saves in unnecessary moves of spider 1 towards fly 2, so spider 1 will move towards fly 1. The trajectory generated by multiagent rollout is to move spiders 1 and 2 towards flies 1 and 2, respectively, then spider 2 first captures fly 2, and then spider 1 captures fly 1.

Note that the rollout policy obtained from the reformulated problem may be di ff erent from the rollout policy obtained from the original problem. However, the former rollout algorithm is far more e ffi cient than the latter in terms of required computation, while still maintaining the cost improvement property (2.89).

## Illustrative Examples

The following spiders-and-flies example illustrates how multiagent rollout may exhibit intelligence and agent coordination that is totally lacking from the base policy. This behavior has been supported by computational experiments and analysis with larger (two-dimensional) spiders-and-flies problems.

## Example 2.9.1 (Spiders and Flies)

We have two spiders and two flies moving along integer locations on a straight line. For simplicity we assume that the flies' positions are fixed at some integer locations, although the problem is qualitatively similar when the flies move randomly. The spiders have the option of moving either left or right by one unit; see Fig. 2.9.2. The objective is to minimize the time to capture both flies. The problem has essentially a finite horizon since the spiders can force the capture of the flies within a known number of steps.

The salient feature of the optimal policy here is to move the two spiders towards di ff erent flies. The minimal time to capture is the maximum of the initial distances of the two spider-fly pairs of the optimal policy.

Let us apply multiagent rollout with the base policy that directs each spider to move one unit towards the closest fly position (a tie is broken by moving towards the right-side fly). The base policy is poor because it may unnecessarily move both spiders in the same direction, when in fact only one

is needed to capture the fly. This limitation is due to the lack of coordination between the spiders: each acts selfishly, ignoring the presence of the other. We will see that rollout restores a significant degree of coordination between the spiders through an optimization that takes into account the long-term consequences of the spider moves.

According to the multiagent rollout mechanism, the spiders choose their moves one-at-a-time, optimizing over the two Q-factors corresponding to the right and left moves, while assuming that future moves will be chosen according to the base policy. Let us consider a stage, where the two flies are alive, while both spiders are closest to fly 2, as in Fig. 2.9.2. Then the rollout algorithm will start with spider 1 and calculate two Q-factors corresponding to the right and left moves, while using the base heuristic to obtain the next move of spider 2, and the remaining moves of the two spiders. Depending on the values of the two Q-factors, spider 1 will move to the right or to the left, and it can be seen that it will choose to move away from spider 2 even if doing so increases its distance to its closest fly contrary to what the base heuristic will do . Then spider 2 will act similarly and the process will continue. Intuitively, at the state of Fig. 2.9.2, spider 1 moves away from spider 2 and fly 2, because it recognizes that spider 2 will capture earlier fly 2, so it might as well move towards the other fly.

Thus the multiagent rollout algorithm induces implicit move coordination , i.e., each spider moves in a way that takes into account future moves of the other spider. In fact it can be verified that the algorithm will produce an optimal sequence of moves starting from any initial spider positions. It can also be seen that ordinary rollout (both flies move at once) will also produce an optimal move sequence.

The example illustrates how a poor base heuristic can produce an excellent rollout solution, something that can be observed frequently in many other problems. Intuitively, the key fact is that rollout is 'farsighted' in the sense that it can benefit from control calculations that reach far into future stages.

A two-dimensional generalization of the example is also interesting. Here the flies are at two corners of a square in the plane. It can be shown that the two spiders, starting from the same position within the square, will separate under the rollout policy, with each moving towards a di ff erent spider, while under the base policy, they will move in unison along the shortest path to the closest surviving fly. Again this will happen for both standard and multiagent rollout.

Let us consider another example of a discrete optimization problem that can be solved e ffi ciently with multiagent rollout.

## Example 2.9.2 (Multi-Vehicle Routing)

Consider a multi-vehicle routing problem, whereby m vehicles move along the arcs of a given graph, aiming to perform tasks located at the nodes of the graph; see Fig. 2.9.3. When a vehicle reaches a task, it performs it, and can move on to perform another task. We wish to perform the tasks in a minimum number of individual vehicle moves.

12

11

10

9

8

7

5

4

6

Base heuristic

10 11 12

towards its nearest pending task, until all tasks are performed

1 2 3 4 5 6 7 8 9 Vehicle 1 Vehicle 2

Move each vehicle one step at a time towards its nearest pending task, Move each vehicle one step at a time towards its nearest pending task, until all tasks are performed

Vehicle 2

2

Optimal

Figure 2.9.3 An instance of the vehicle routing problem of Example 2.9.2, and the multiagent rollout approach. The two vehicles aim to collectively perform the two tasks as fast as possible. Here, we should avoid sending both vehicles to node 4, towards the task at node 7; sending only vehicle 2 towards that task, while sending vehicle 1 towards the task at node 9 is clearly optimal. However, the base heuristic has 'limited vision' and does not perceive this. By contrast the standard and the one-vehicle-at-a-time rollout algorithms look beyond the first move and avoid this ine ffi ciency: they examine both moves of vehicle 1 to nodes 3 and 4, and use the base heuristic to explore the corresponding trajectories to the end of the horizon, and discover that vehicle 2 can reach quickly node 7, and that it is best to send vehicle 1 towards node 9.

<!-- image -->

In particular, the one-vehicle-at-a-time rollout algorithm will operate as follows: given the starting position pair (1 ↪ 2) of the vehicles and the current pending tasks at nodes 7 and 9, we first compare the Q-factors of the two possible moves of vehicle 1 (to nodes 3 and 4), assuming that all the remaining moves will be selected by the base heuristic at the beginning of each stage. Thus vehicle 1 will choose to move to node 3. Then with knowledge of the move of vehicle 1 from 1 to 3, we select the move of vehicle 2 by comparing the Q-factors of its two possible moves (to nodes 4 and 5), taking also into account the fact that the remaining moves will be made according to the base heuristic. Thus vehicle 2 will choose to move to node 4.

We then continue at the next state [vehicle positions at (3,4) and pending tasks at nodes 7 and 9], select the base heuristic moves of vehicles 1 and 2 on the path to the closest pending tasks [(9 and 7), respectively], etc. Eventually the rollout finds the optimal solution (move vehicle 1 to node 9 in three moves and move vehicle 2 to node 7 in two moves), which has a total cost of 5. By contrast it can be seen that the base heuristic at the initial state will move both vehicles to node 4 (towards the closest pending task), and generate a trajectory that moves vehicle 1 along the path 1 → 4 → 7 and vehicle 2 along the path 2 → 4 → 7 → 10 → 12 → 9, while incurring a total cost of 7.

For a large number of vehicles and a complicated graph, this is a nontrivial combinatorial problem. The problem can be formulated as a discrete deterministic optimization problem, and addressed by approximate DP methods. The state at a given stage is the m -tuple of current positions of the vehicles together with the list of pending tasks, but the number of these states can be enormous (it increases exponentially with the number of nodes and the number of vehicles). Moreover the number of joint move choices by the vehicles also increases exponentially with the number of vehicles.

We are thus motivated to use a multiagent rollout approach. We define a base heuristic as follows: at a given stage and state (vehicle positions and pending tasks), it finds the closest pending task (in terms of number of moves needed to reach it) for each of the vehicles and moves each vehicle one step towards the corresponding closest pending task (this is a legitimate base heuristic: it assigns to each state a vehicle move for every vehicle).

In the multiagent rollout algorithm, at a given stage and state, we take up each vehicle in the order 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , and we compare the Q-factors of the available moves to that vehicle while assuming that all the remaining moves will be made according to the base heuristic, and taking into account the moves that have been already made and the tasks that have already been performed; see the illustration of Fig. 2.9.3. In contrast to all-vehicles-at-once rollout, the one-vehicle-at-a-time rollout algorithm considers a polynomial (in m ) number of moves and corresponding shortest path problems at each stage. In the example of Fig. 2.9.3, the one-vehicle-at-a-time rollout finds the optimal solution, while the base heuristic starting from the initial state does not.

## The Cost Improvement Property

Generally, it is unclear how the two rollout policies (standard/all-agents-atonce and agent-by-agent) perform relative to each other in terms of attained cost. ‡ On the other hand, both rollout policies perform no worse than the

There is an alternative version of the base heuristic, which makes selections one-vehicle-at-a-time: at a given stage and state (vehicle positions and pending tasks), it finds the closest pending task (in terms of number of moves needed to reach it) for vehicle 1 and moves this vehicle one step towards this closest pending task. Then it finds the closest pending task for vehicle 2 (the pending status of the tasks, however, may have been a ff ected by the move of vehicle 1) and moves this vehicle one step towards this closest pending task, and continues similarly for vehicles 3 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n . There is a subtle di ff erence between the two base heuristics: for example they may make di ff erent choices when vehicle 1 reaches a pending task in a single move, thereby changing the status of that task, and a ff ecting the choice of the base heuristic for vehicle 2, etc.

‡ For an example where the standard rollout algorithm works better, consider a single-stage problem, where the objective is to minimize the first stage cost g 0 ( u 1 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m 0 ) glyph[triangleright] Let u 0 = ( u 1 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m 0 ) be the control applied by the base policy, and assume that u 0 is not optimal. Suppose that starting at u 0 , the cost cannot be improved by varying any single control component. Then the multiagent rollout algorithm stays at the suboptimal u 0 , while the standard rollout algorithm finds

base policy, since the performance of the base policy is identical for both the reformulated and the original problems. This cost improvement property can also be shown analytically as follows by induction, by modifying the standard rollout cost improvement proof; cf. Section 2.7.

Proposition 2.9.1: (Cost Improvement for Multiagent Rollout) The rollout policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ obtained by multiagent rollout satisfies

<!-- formula-not-decoded -->

where π is the base policy.

Proof: We will show the inequality (2.90) by induction, but for simplicity, we will give the proof for the case of just two agents, i.e., m = 2. Clearly the inequality holds for k = N , since J N↪ ˜ π = J N↪ π = g N . Assuming that it holds for index k +1, we have for all x k ,

<!-- formula-not-decoded -->

an optimal control. Thus, for one-stage problems, the standard rollout algorithm will perform no worse than the multiagent rollout algorithm.

The example just given is best seen within the framework of the classical coordinate descent method for minimizing a function of m components. This method can get stuck at a nonoptimal point in the absence of appropriate conditions on the cost function, such as di ff erentiability and/or convexity. However, within our context of multistage rollout and possibly stochastic disturbances, it appears that the consequences of such a phenomenon may not be serious. In fact, one can construct multi-stage examples where multiagent rollout performs better than the standard rollout.

where:

- (a) The first equality is the DP equation for the rollout policy ˜ π .
- (b) The first inequality holds by the induction hypothesis.
- (c) The second equality holds by the definition of the rollout algorithm as it pertains to agent 2.
- (d) The third equality holds by the definition of the rollout algorithm as it pertains to agent 1.
- (e) The fourth equality is the DP equation for the base policy π .

The induction proof of the cost improvement property (2.90) is thus complete for the case m = 2. The proof for an arbitrary number of agents m is entirely similar. Q.E.D.

## Optimizing the Agent Order in Agent-by-Agent Rollout Multiagent Parallelization

In the multiagent rollout algorithm described so far, the agents optimize the control components sequentially in a fixed order. It is possible to improve performance by trying to optimize at each stage k the order of the agents.

/negationslash

An e ffi cient way to do this is to first optimize over all single agent Qfactors, by solving the m minimization problems that correspond to each of the agents /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m being first in the multiagent rollout order. If /lscript 1 is the agent that produces the minimal Q-factor, we fix /lscript 1 to be the first agent in the multiagent rollout order. Then we optimize over all single agent Qfactors, by solving the m -1 minimization problems that correspond to each of the agents /lscript = /lscript 1 being second in the multiagent rollout order. Let /lscript 2 be the agent that produces the minimal Q-factor, fix /lscript 2 to be the second agent in the multiagent rollout order, and continue in this manner. In the end, after

<!-- formula-not-decoded -->

minimizations, we obtain an agent order /lscript 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ /lscript m that produces a potentially much reduced Q-factor value, as well as the corresponding rollout control component selections.

<!-- formula-not-decoded -->

The method just described likely produces substantially better performance, and eliminates the need for guessing a good agent order, but it increases the number of Q-factor calculations needed per stage roughly by a factor ( m + 1) glyph[triangleleft] 2. Still this is much better than the all-agents-atonce approach, which requires an exponential number of Q -factor calculations. Moreover, the Q-factor minimizations of the above process can be parallelized, so with m parallel processors, we can perform the number of m ( m +1) glyph[triangleleft] 2 minimizations derived above in just m batches of parallel minimizations, which require about the same time as in the case where the agents are selected for Q-factor minimization in a fixed order. We finally note that our earlier cost improvement proof goes through again by induction, when the order of agent selection is variable at each stage k .

## Multiagent Rollout Variants

The agent-by-agent rollout algorithm admits several variants. We describe briefly a few of these variants.

- (a) We may use multiagent truncated rollout and terminal cost function approximation, as described earlier. Of course, in this case the cost improvement property need not hold.
- (b) When the control constraint sets U /lscript k ( x k ) are infinite, multiagent rollout still applies, based on the tradeo ff between control and state space complexity, cf. Fig. 2.9.1. In particular, when the sets U /lscript k ( x k ) are intervals of the real line, each agent's lookahead minimization problem can be performed with the aid of one-dimensional search methods.
- (c) When the problem is deterministic there are additional possible variants of the multiagent rollout algorithm. In particular, for deterministic problems, we may use a more general base policy, i.e., a heuristic that is not defined by an underlying policy; cf. Section 2.3.1. In this case, if the sequential improvement assumption for the modified problem of Fig. 2.9.1 is not satisfied, then the cost improvement property may not hold. However, cost improvement may be restored by introducing fortification, as discussed in Section 2.3.2.
- (d) The multiagent rollout algorithm can be simply modified to apply to infinite horizon problems. In this context, we may also consider policy iteration methods, which can be viewed as repeated rollout. These methods may involve agent-by-agent policy improvement, and value and policy approximations of intermediately generated policies (see the RL book [Ber19a], Section 5.7.3; also the paper [MLB25]). The paper [Ber21a] provides an analysis of methods involving exact policy evaluation and agent-by-agent policy improvement, and shows convergence to an agent-by-agent optimal policy, a form of locally optimal policy.

- (e) The multiagent rollout algorithm can be simply modified to apply to deterministic continuous-time optimal control problems; cf. Section 2.6. The idea is again to simplify the minimization over u ( t ) in the case where u ( t ) consists of multiple components u 1 ( t ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ( t ).
- (f) We can implement within the agent-by-agent rollout context the use of Q-factor di ff erences. The motivation is similar: deal with the approximation errors that are inherent in the estimated cost of the base policy, ˜ J k +1 ↪ π ( f k ( x k ↪ u k ) ) , and may overwhelm the current stage cost term g k ( x k ↪ u k ). This may seriously degrade the quality of the rollout policy; see also the discussion of advantage updating and di ff erential training in Chapter 3.

## Constrained Multiagent Rollout

Let us consider a special structure of the control space, where the control u k consists of m components, u k = ( u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k ), each belonging to a corresponding set U /lscript k ( x k ), /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m . Thus the control space at stage k is the Cartesian product

<!-- formula-not-decoded -->

We refer to this as the multiagent case , motivated by the special case where each component u /lscript k is chosen by a separate agent /lscript at stage k .

Similar to the unconstrained case, we can introduce a modified but equivalent problem, involving one-at-a-time agent control selection. In particular, at the generic state x k , we break down the control u k into the sequence of the m controls u 1 k ↪ u 2 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k , and between x k and the next state x k +1 = f k ( x k ↪ u k ), we introduce artificial intermediate 'states'

<!-- formula-not-decoded -->

and corresponding transitions. The choice of the last control component u m k at 'state' ( x k ↪ u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m -1 k ) marks the transition at cost g k ( x k ↪ u k ) to the next state x k +1 = f k ( x k ↪ u k ) according to the system equation. It is evident that this reformulated problem is equivalent to the original, since any control choice that is possible in one problem is also possible in the other problem, with the same cost.

By working with the reformulated problem, we can consider a rollout algorithm requires a sequence of m minimizations per stage, one over each of the control components u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k , with the past controls already determined by the rollout algorithm, and the future controls determined by running the base heuristic. Assuming a maximum of n elements in the control component spaces U /lscript k ( x k ), /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , the computation required for the m single control component minimizations is of order O ( nm ) per stage. By contrast the standard rollout minimization (2.47) involves the computation of as many as n m terms G ( T k (˜ y k ↪ u k ) ) per stage.

## 2.9.1 Asynchronous and Autonomous Multiagent Rollout

In this section we consider multiagent rollout algorithms that are distributed and asynchronous in the sense that the agents may compute their rollout controls in parallel rather than in sequence, aiming at computational speedup. An example of such an algorithm is obtained when at a given stage, agent /lscript computes the rollout control ˜ u /lscript k before knowing the rollout controls of some of the agents 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ /lscript -1, and uses the controls θ 1 k ( x k ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ /lscript -1 k ( x k ) of the base policy in their place.

This algorithm often works well and it may be worth trying on a given problem. However, it does not possess the cost improvement property, and may not work well at all for some problems and specially selected initial states. In fact we can construct a simple example with a single state, two agents, and two controls per agent, where the second agent does not take into account the control applied by the first agent, and as a result the rollout policy performs worse than the base policy for some initial states.

## Example 2.9.3 (Cost Deterioration in the Absence of Adequate Agent Coordination)

/negationslash

Consider a problem with two agents ( m = 2) and a single state. Thus the state does not change and the costs of di ff erent stages are decoupled (the problem is essentially static). Each of the two agents has two controls: u 1 k ∈ ¶ 0 ↪ 1 ♦ and u 2 k ∈ ¶ 0 ↪ 1 ♦ . The cost per stage g k is equal to 0 if u 1 k = u 2 k , is equal to 1 if u 1 k = u 2 k = 0, and is equal to 2 if u 1 k = u 2 k = 1. Suppose that the base policy applies u 1 k = u 2 k = 0. Then it can be seen that when executing rollout, the first agent applies u 1 k = 1, and in the absence of knowledge of this choice, the second agent also applies u 2 k = 1 (thinking that the first agent will use the base policy control u 1 k = 0). Thus the cost of the rollout policy is 2 per stage, while the cost of the base policy is 1 per stage. By contrast the rollout algorithm that takes into account the first agent's control when selecting the second agent's control applies u 1 k = 1 and u 2 k = 0, thus resulting in a rollout policy with the optimal cost of 0 per stage.

The di ffi culty here is inadequate coordination between the two agents. In particular, each agent uses rollout to compute the local control, thinking that the other will use the base policy control. If instead the two agents coordinated their control choices, they would have applied an optimal policy.

The simplicity of the preceding example raises serious questions as to whether the cost improvement property (2.90) can be easily maintained by a distributed rollout algorithm where the agents do not know the controls applied by the preceding agents in the given order of local control selection, and use instead the controls of the base policy. One may speculate that if the agents are naturally 'weakly coupled' in the sense that their choice of control has little impact on the desirability of various controls of other

agents, then a more flexible inter-agent communication pattern may be su ffi cient for cost improvement.

An important question is to clarify the extent to which agent coordination is essential. In what follows in this section, we will discuss a distributed asynchronous multiagent rollout scheme, which is based on the use of a signaling policy that provides estimates of coordinating information once the current state is known.

## Autonomous Multiagent Rollout - Signaling Policies

An interesting possibility for autonomous control selection by the agents is to use a distributed rollout algorithm, which is augmented by a precomputed signaling policy that embodies agent coordination. ‡ The idea is to assume that the agents do not communicate their computed rollout control components to the subsequent agents in the given order of local control selection. Instead, once the agents know the state, they use precomputed (or easily computed) approximations to the control components of the preceding agents , and compute their own control components in parallel and asynchronously. We call this algorithm autonomous multiagent rollout . While this type of algorithm involves a form of redundant computation, it allows for additional speedup through parallelization.

/negationslash

The algorithm at the k th stage uses a base policy θ k = ¶ θ 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m -1 k ♦ , but also uses a second policy ̂ θ k = ¶ ̂ θ 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ̂ θ m -1 k ♦ , called the signaling policy , which is computed o ff -line, is known to all the agents for on-line use, and is designed to play an agent coordination role. Intuitively, ̂ θ /lscript k ( x k ) provides an intelligent 'guess' about what agent /lscript will do at state x k . This is used in turn by all other agents i = /lscript to compute asynchronously their own rollout control components on-line.

More precisely, the autonomous multiagent rollout algorithm uses the base and signaling policies to generate a rollout policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦

In particular, one may divide the agents in 'coupled' groups, and require coordination of control selection only within each group, while the computation of di ff erent groups may proceed in parallel. Note that the 'coupled' group formations may change over time, depending on the current state. For example, in applications where the agents' locations are distributed within some geographical area, it may make sense to form agent groups on the basis of geographic proximity, i.e., one may require that agents that are geographically near each other (and hence are more coupled) coordinate their control selections, while agents that are geographically far apart (and hence are less coupled) forego any coordination.

‡ The general idea of coordination by sharing information about the agents' policies arises also in other multiagent algorithmic contexts, including some that involve forms of policy gradient methods and Q-learning; see the surveys of the relevant research cited earlier. The survey by Matignon, Laurent, and Le FortPiat [MLL12] focuses on coordination problems from an RL point of view.

as follows. At stage k and state x k , ˜ θ k ( x k ) = ( ˜ θ 1 k ( x k ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ m k ( x k ) ) ↪ is obtained according to

<!-- formula-not-decoded -->

Note that the preceding computation of the controls ˜ θ 1 k ( x k ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ m k ( x k ) can be done asynchronously and in parallel, and without direct agent coordination, since the signaling policy values ̂ θ 1 k ( x k ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ̂ θ m -1 k ( x k ) are precomputed and are known to all the agents.

<!-- formula-not-decoded -->

The simplest choice is to use as signaling policy ̂ θ the base policy θ . However, this choice does not guarantee policy improvement as evidenced by Example 2.9.3. In fact performance deterioration with this choice is not uncommon, and can be observed in more complicated examples, including the following.

## Example 2.9.4 (Spiders and Flies - Use of the Base Policy for Signaling)

Consider the problem of Example 2.9.1, which involves two spiders and two flies on a line, and the base policy θ that moves a spider towards the closest surviving fly (and in case where a spider starts at the midpoint between the two flies, moves the spider to the right). Assume that we use as signaling policy ̂ θ the base policy θ . It can then be verified that if the spiders start from di ff erent positions, the rollout policy will be optimal (will move the spiders in opposite directions). If, however, the spiders start from the same position , a completely symmetric situation is created, whereby the rollout controls move both flies in the direction of the fly furthest away from the spiders' position (or to the left in the case where the spiders start at the midpoint between the two flies). Thus, the flies end up oscillating around the middle of the interval between the flies and never catch the flies!

The preceding example is representative of a broad class of counterexamples that involve multiple identical agents. If the agents start at

the same initial state, with a base policy that has identical components, and use the base policy for signaling, the agents will select identical controls under the corresponding multiagent rollout policy, ending up with a potentially serious cost deterioration.

/negationslash

This example also highlights an e ff ect of the sequential choice of the control components u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k , based on the reformulated problem of Fig. 2.9.1: it tends to break symmetries and 'group think' that guides the agents towards selecting the same controls under identical conditions. Generally, any sensible multiagent policy must be able to deal in some way with this 'group think' issue. One simple possibility is for each agent /lscript to randomize somehow the control choices of other agents j = /lscript when choosing its own control, particularly in 'tightly coupled' cases where the choice of agent /lscript is 'strongly' a ff ected by the choices of the agents j = /lscript .

/negationslash

An alternative idea is to choose the signaling policy ̂ θ k to approximate the sequential multiagent rollout policy (the one computed with each agent knowing the controls applied by the preceding agents), or some other policy that is known to embody coordination between the agents. In particular, we may obtain ̂ θ k as the multiagent rollout policy for a related but simpler problem, such as a certainty equivalent version of the original problem, whereby the stochastic system is replaced by a deterministic one.

Another interesting possibility is to compute ̂ θ k = ( ̂ θ 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ̂ θ m k ) by o ff -line training of a neural network (or m networks, one per agent) with training samples generated through the sequential multiagent rollout policy. We intuitively expect that if the neural network provides a signaling policy that approximates well the sequential multiagent rollout policy, we would obtain better performance than the base policy. This expectation was confirmed in a case study involving a large-scale multi-robot repair application (see [BKB20]).

The advantage of autonomous multiagent rollout with neural network or other approximations is that it may lead to approximate policy improvement, while at the same time allowing asynchronous agent operation without coordination through communication of their rollout control values (but still assuming knowledge of the exact state by all agents).

## 2.10 ROLLOUT FOR BAYESIAN OPTIMIZATION AND SEQUENTIAL ESTIMATION

In this section, we discuss a wide class of problems that has been studied intensively in statistics and related fields since the 1940s. Roughly speaking, in these problems we use observations and sampling for the purpose of inference, but the number and the type of observations are not fixed in advance. Instead, the outcomes of the observations are sequentially evaluated on-line with a view towards stopping or modifying the observation process. This involves sequential decision making, thus bringing to bear exact and

Observation Type Selection

Observation Type Selection

System Observation Outcome Decision

System Observation Outcome Decision on Next Observation

Figure 2.10.1 Illustration of sequential estimation of a parameter θ . At each time a decision is made to select one of several observation types relating to θ , each of di ff erent cost, or stop the observations and provide a final estimate of θ .

<!-- image -->

approximate DP. A central issue here is to estimate an m -dimensional random vector θ , using optimal sequential selection of observations, which are based on feedback from preceding observations; see Fig. 2.10.1. Here is a simple but historically important illustrative example, where θ represents a binary hypothesis.

## Example 2.10.1 (Hypothesis Testing - Sequential Probability Ratio Test)

Consider a hypothesis testing problem whereby we can make observations, at a cost C each, relating to two hypotheses. Given a new observation, we can either accept one of the hypotheses or delay the decision for one more period, pay the cost C , and obtain a new observation. At issue is trading o ff the cost of observation with the higher probability of accepting the wrong hypothesis. As an example, in a quality control setting, the two hypotheses may be that a certain product meets or does not meet a certain level of quality, while the observations may consist of quantitative tests of the quality of the product.

Intuitively, one expects that once the conditional probability of one of the hypotheses, given the observations thus far, gets su ffi ciently close to 1, we should stop the observations. Indeed classical DP analyses bear this out; see e.g., the books by Cherno ff [Che72], DeGroot [DeG70], Whittle [Whi82], and the references quoted therein. In particular, the simple version of the hypothesis testing problem just described admits a simple and elegant optimal solution, known as the sequential probability ratio test . On the other hand more complex versions of the problem, involving for example multiple hypotheses and/or multiple types of observations, are computationally intractable, thus necessitating the use of suboptimal approaches.

Observation Type Selection

An important distinction in sequential estimation problems is whether the current choice of observation a ff ects the cost and the availability of future observations. If this is so, the problem can often be viewed most fruitfully as a combined estimation and control problem , and is related to a type of adaptive control problem that we will discuss in the next section. As an example we will consider there sequential decoding, whereby we search for a hidden code word by using a sequence of queries, in the spirit of the Wordle puzzle and the family of Mastermind games [see, e.g., the Wikipedia page for 'Mastermind (board game)'].

If the observation choices are 'independent' and do not a ff ect the cost or availability of future observations, the problem is substantially simplified. We will discuss problems of this type in the present section, starting with the cases of surrogate and Bayesian optimization.

## Surrogate Optimization

Surrogate optimization refers to a collection of methods, which address suboptimally a broad range of minimization problems, beyond the realm of DP. The problem is to minimize approximately a function that is given as a 'black box.' By this we mean a function whose analytical expression is unknown, and whose values at any one point may be hard-to-compute, e.g., may requite costly simulation or experimentation. The idea is to replace such a cost function with a 'surrogate' whose values are easier to compute.

Here we introduce a model of the cost function that is parametrized by a parameter θ ; see Fig. 2.10.2. We observe sequentially the cost function at a few observation points, construct a model of the cost function (the surrogate) by estimating θ based on the results of the observations, and minimize the surrogate to obtain a suboptimal solution. The question is how to select observation points sequentially, using feedback from previous observations. This selection process often embodies an explorationexploitation tradeo ff : Observing at points likely to have near-optimal value vs observing at points in relatively unexplored areas of the search space.

Surrogate optimization at its core involves construction from data of functions of interest. Thus the ideas to be presented apply to other domains, e.g., the construction of probability density functions from data.

## Bayesian Optimization

Bayesian optimization (BO) has been used widely for the approximate optimization of functions whose values at given points can only be obtained through time-consuming calculation, simulation, or experimentation. A classical application from geostatistical interpolation, pioneered by the statisticians Matheron and Krige, was to identify locations of high gold distribution in South Africa based on samples from a few boreholes (the name 'kriging' is often used to refer to this type of application; see the review by Kleijnen [Kle09]). As another example, BO has been used to select the

{ Mans samalan fanma afarmanat

Black

Box

Function

Observation

Surrogate Model

Unknown Parameter

0

Next Observation

Black Box Function

System Observation Outcome Decision on Next Observation

Figure 2.10.2 Illustration of the construction of a surrogate for a 'black box' function f whose values are hard-to-compute. We replace f with a parametric model that involves a parameter θ to be estimated by using observations at some points. The points are selected sequentially, using the results of earlier observations. Eventually, the observation process is stopped (often when an observation/computation budget limit is reached), and the final estimate of θ is used to construct the surrogate to be minimized in place of f .

<!-- image -->

hyperparameters of machine-learning models, including the architectural parameters of the deep neural network of AlphaZero; see [SHS17].

In this section, we will focus on a relatively simple BO formulation that can be viewed as the special case of surrogate optimization. In particular, we will discuss the case where the surrogate function is parametrized by the collection of its values at the points where it is defined. See the references cited later in this section. Formally, we want to minimize a real-valued function f , defined over a set of m points, which we denote by 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m . These m points lie in some space, which we leave unspecified for the moment. ‡ The values of the function are not readily available, but can be estimated with observations that may be imperfect. However, the observations are so costly that we can only hope to observe the function at a limited number of points. Once the function has been estimated with this type of observation process, we obtain a surrogate cost function, which may be minimized to obtain an approximately optimal solution.

More complex forms of surrogates are obtained through linear combinations of some basis functions, with the parameter vector θ consisting of the weights of the basis functions.

‡ We restrict the domain of definition of f to be the finite set ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m ♦ in order to facilitate the implementation of the rollout algorithm to be discussed in what follows. However, in a more general and sometimes more convenient formulation, the domain of f can be an infinite set, such as a subset of a finitedimensional Euclidean space.

Function f (u) = Ou

21 = 01 + W1

• 01

1

• 02

Function

.

2

Figure 2.10.3 Illustration of a function f that we wish to estimate. The function is defined at the points u = 1 ↪ 2 ↪ 3 ↪ 4, and is represented by a vector θ = ( θ 1 ↪ θ 2 ↪ θ 3 ↪ θ 4 ) ∈ /Rfractur 4 , in the sense that f ( u ) = θ u for all u . The prior distribution of θ is given, and is used to construct the posterior distribution of θ given noisy observations z u = θ u + w u at some of the points u .

<!-- image -->

We denote the value of f at a point u by θ u :

<!-- formula-not-decoded -->

Thus the m -dimensional vector θ = ( θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m ) belongs to /Rfractur m and represents the function f . We assume that we obtain sequentially noisy observations of values f ( u ) = θ u at suitably selected points u . These values are used to estimate the vector θ (i.e., the function f ), and to ultimately minimize (approximately) f over the m points u = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m . The essence of the problem is to select points for observation based on an explorationexploitation tradeo ff (exploring the potential of relatively unexplored candidate solutions and improving the estimate of promising candidate solutions). The fundamental idea of the BO methodology is that the function value changes relatively slowly, so that observing the function value at some point provides information about the function values at neighboring points. Thus a limited number of strategically chosen observations can provide reasonable approximation to the true cost function over a large portion of the search space.

For a mathematical formulation of a BO framework, we assume that at each of N successive times k = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N , we select a single point u k ∈ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m ♦ , and observe the corresponding component θ u k of θ (i.e., the function value at u k ) with some noise w u k , i.e.,

<!-- formula-not-decoded -->

see Fig. 2.10.3. We view the observation points u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N as the optimization variables (or controls/actions in a DP/RL context), and consider policies for selecting u k with knowledge of the preceding observations

0 03

24 = 04 + W4

004

Using measurements of the form

z u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ z u k -1 that have resulted from the selections u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 . We assume that the noise random variables w u , u ∈ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m ♦ are independent and that their distributions are given. Moreover, we assume that θ has a given a priori distribution on the space of m -dimensional vectors /Rfractur m , which we denote by b 0 . The posterior distribution of θ , given any subset of observations

<!-- formula-not-decoded -->

is denoted by b k .

An important special case arises when b 0 and the distributions of w u , u ∈ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m ♦ , are Gaussian. In this case b 0 is a multidimensional Gaussian distribution, defined by its mean (based on prior knowledge, or an equal value for all u = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m in case of absence of such knowledge) and its covariance matrix [implying greater correlation for pairs ( u↪ u ′ ) that are 'close' to each other in some problem-specific sense, e.g., exponentially decreasing with the Euclidean distance between u and u ′ ]. A key consequence of this assumption is that the posterior distribution b k is multidimensional Gaussian, and can be calculated in closed form by using well-known formulas.

More generally, b k evolves according to an equation of the form

<!-- formula-not-decoded -->

Thus given the set of observations up to time k , and the next choice u k +1 , resulting in an observation value z u k +1 , the function B k gives the formula for updating b k to b k +1 , and may be viewed as a recursive estimator of b k . In the Gaussian case, the function B k can be written in closed form, using standard formulas for Gaussian random vector estimation. In other cases where no closed form expression is possible, B k can be implemented through simulation that computes (approximately) the new posterior b k +1 using samples generated from the current posterior b k .

At the end of the sequential estimation process, after the complete observation set

<!-- formula-not-decoded -->

has been obtained, we have the posterior distribution b N of θ , which we can use to compute a surrogate of f . As an example we may use as surrogate the posterior mean

<!-- formula-not-decoded -->

and declare as minimizer of f over u the point u ∗ with minimum posterior mean:

<!-- formula-not-decoded -->

see Fig. 2.10.4.

There is a large literature relating to the surrogate and Bayesian optimization methodology and its applications, particularly for the Gaussian

Posterior 610

<!-- image -->

True Cost Function f(u)

Figure 2.10.4 Illustration of the true cost function f , defined over an interval of the real line, and the posterior distribution b 10 after noise-free measurements at 10 points. The shaded area represents the interval of the mean plus/minus the standard deviation of the posterior b 10 at the points u . The mean of the finally obtained posterior, as a function of u , may be viewed as a surrogate cost function that can be minimized in place of f . Note that since the observations are assumed noise-free, the mean of the posterior is exact at the observation points.

case. We refer to the books by Rasmussen and Williams [RaW06], Powell and Ryzhov [PoR12], the highly cited papers by Saks et al. [SWM89], Jones, Schonlau, and Welch [JSW98], and Queipo et al. [QHS05], the reviews by Sasena [Sas02], Powell and Frazier [PoF08], Forrester and Keane [FoK09], Kleijnen [Kle09], Brochu, Cora, and De Freitas [BCD10], Ryzhov, Powell, and Frazier [RPF12], Ghavamzadeh, Mannor, Pineau, and Tamar [GMP15], Shahriari et al. [SSW16], and Frazier [Fra18], and the references quoted there. Our purpose here is to focus on the aspects of the subject that are most closely connected to exact and approximate DP.

## A Dynamic Programming Formulation

The sequential estimation problem just described, viewed as a DP problem, involves a state at time k , which is the posterior (or belief state) b k , and a control/action at time k , which is the point index u k +1 selected for observation. The transition equation according to which the state evolves, is

<!-- formula-not-decoded -->

cf. Eq. (2.94). To complete the DP formulation, we need to introduce a cost structure. To this end, we assume that observing θ u , as per Eq. (2.93), incurs a cost c ( u ), and that there is a terminal cost G ( b N ) that depends of the final posterior distribution; as an example, the function G may involve the mean and covariance corresponding to b N .

The corresponding DP algorithm is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and proceeds backwards from the terminal condition

<!-- formula-not-decoded -->

The expected value in the right side of the DP equation (2.95) is taken with respect to the conditional distribution of z u k +1 , given b k and the choice u k +1 . The observation cost c ( u ) may be 0 or a constant for all u , but it can also have a more complicated dependence on u . The terminal cost G ( b N ) may be a suitable measure of surrogate 'fidelity' that depends on the posterior mean and covariance of θ corresponding to b N .

Generally, executing the DP algorithm (2.95) is practically infeasible, because the space of posterior distributions is infinite-dimensional. In the Gaussian case where the a priori distribution b 0 is Gaussian and the noise variables w u are Gaussian, the posterior b k is m -dimensional Gaussian, so it is characterized by its mean and covariance, and can be specified by a finite set of numbers. Despite this simplification, the DP algorithm (2.95) is prohibitively time-consuming even under Gaussian assumptions, except for simple special cases. We consequently resort to approximation in value space, whereby the function J ∗ k +1 in the right side of Eq. (2.95) is replaced by an approximation ˜ J k +1 .

## Approximation in Value Space

The most popular BO methodology makes use of a myopic/greedy policy θ k +1 , which at each time k and given b k , selects a point ˆ u k +1 = θ k +1 ( b k ) for the next observation, using some calculation involving an acquisition function . This function, denoted A k ( b k ↪ u k +1 ), quantifies some form of 'expected benefit' for an observation at u k +1 , given the current posterior b k . The myopic policy selects the next point at which to observe, ˆ u k +1 ,

Acommon type of acquisition function is the upper confidence bound , which has the form

<!-- formula-not-decoded -->

where T k ( b k ↪ u ) is the negative of the mean of f ( u ) under the posterior distribution b k , R k ( b k ↪ u ) is the standard deviation of f ( u ) under the posterior distribution b k , and β is a tunable positive scalar parameter. Thus T k ( b k ↪ u ) can be

by maximizing the acquisition function:

<!-- formula-not-decoded -->

Several ways to define suitable acquisition functions have been proposed, and an important issue is to be able to calculate economically its values A k ( b k ↪ u k +1 ) for the purposes of the maximization in Eq. (2.97). Another important issue of course is to be able to calculate the posterior b k economically.

Approximation in value space is an alternative approach, which is based on the DP formulation of the preceding section. In particular, in this approach we approximate the DP algorithm (2.95) by replacing J ∗ k +1 with an approximation ˜ J k +1 in the minimization of the right side. Thus we select the next observation at point ˜ u k +1 according to

<!-- formula-not-decoded -->

where Q k ( b k ↪ u k +1 ) is the Q-factor corresponding to the pair ( b k ↪ u k +1 ), given by

<!-- formula-not-decoded -->

The expected value in the preceding equation is taken with respect to the conditional probability distribution of z u k +1 given ( b k ↪ u k +1 ), which can be computed using b k and the given distribution of the noise w u k +1 . Thus if b k and ˜ J k +1 are available, we may use Monte Carlo simulation to determine the Q-factors Q k ( b k ↪ u k +1 ) for all u k +1 ∈ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m ♦ , and select as next point for observation the one that corresponds to the minimal Q-factor [cf. Eq. (2.98)].

## Rollout Algorithms for Bayesian Optimization

A special case of approximation in value space is the rollout algorithm, whereby the function J ∗ k +1 in the right side of the DP Eq. (2.95) is replaced by the cost function of some base policy θ k +1 ( b k ), k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1. Thus, viewed as an exploitation index (encoding our desire to search within parts of the space where f takes low value), while R k ( b k ↪ u ) can be viewed as an exploration index (encoding our desire to search within parts of the space that are relatively unexplored). There are several other popular acquisition functions, which directly or indirectly embody a tradeo ff between exploitation and exploration. A popular example is the expected improvement acquisition function, which is equal to the expected value of the reduction of f ( u ) relative to the minimal value of f obtained up to time k (under the posterior distribution b k ).

¿ Mho allant olanditha fa DA ..

Possible

Observations

Uk+1

Current

Posterior

(bo

Rollout with

Stages given a base policy the rollout algorithm uses the cost function of this policy as the function ˜ J k +1 in the approximation in value space scheme (2.98)(2.99). The values of ˜ J k +1 needed for the Q-factor calculations in Eq. (2.99) can be computed or approximated by simulation. Greedy/myopic policies based on an acquisition function [cf. Eq. (2.97)] have been suggested as base policies in various rollout proposals.

Possible

Posteriors 0k+1

One-Step or Multistep Lookahead for stages Possible

Figure 2.10.5 Illustration of rollout at the current posterior b k . For each u k +1 ∈ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m ♦ , we compute the Q-factor Q k ( b k ↪ u k +1 ) by using Monte-Carlo simulation with samples from w u k +1 and a base heuristic that uses an acquisition function starting from each possible posterior b k +1 . The rollout may extend to the end of the horizon N , or it may be truncated after a few steps.

<!-- image -->

In particular, given b k , the rollout algorithm computes for each u k +1 ∈

The rollout algorithm for BO was first proposed under Gaussian assumptions by Lam, Wilcox, and Wolpert [LWW16]. It was further discussed by Jiang et al. [JJB20], [JCG20], Lee at al. [LEC20], Lee [Lee20], Yue and Kontar [YuK20], Lee et al. [LEP21], Paulson, Sorouifar, and Chakrabarty [PSC22], where it is also referred to as 'nonmyopic BO' or 'nonmyopic sequential experimental design.' For related work, see Gerlach, Ho ff mann, and Charlish [GHC21]. These papers also discuss various approximations to the rollout approach, and generally report encouraging computational results. Section 3.5 of the author's book [Ber20a] focuses on rollout algorithms for surrogate and Bayesian optimization.

Rollout with Base Policy Using an Acquisition Function

Stages Beyond Truncation

Stages Beyond Truncation

¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m ♦ a Q-factor value Q k ( b k ↪ u k +1 ) by simulating the base policy for multiple time steps starting from all possible posteriors b k +1 that can be generated from ( b k ↪ u k +1 ), and by accumulating the corresponding cost [including a terminal cost such as G ( b N )]; see Fig. 2.10.5. It then selects the next point ˜ u k +1 for observation by using the Q-factor minimization of Eq. (2.98).

Note that the equation

<!-- formula-not-decoded -->

which governs the evolution of the posterior distribution (or belief state), is stochastic because z u k +1 involves the stochastic noise w u k +1 . Thus some Monte Carlo simulation is unavoidable in the calculation of the Q-factors Q k ( b k ↪ u k +1 ). On the other hand, one may greatly reduce the Monte Carlo computational burden by employing a certainty equivalence approximation, which at stage k , treats only the noise w u k +1 as stochastic, and replaces the noise variables w u k +2 ↪ w u k +3 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] , after the first stage of the calculation, by deterministic quantities such as their means ˆ w u k +2 ↪ ˆ w u k +3 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] .

The simulation of the Q-factor values may also involve other approximations, some of which have been suggested in various proposals for rolloutbased BO. For example, if the number of possible observations m is very large, we may compute and compare the Q-factors of only a subset. In particular, at a given time k , we may rank the observations by using an acquisition function, select a subset U k +1 of most promising observations, compute their Q-factors Q k ( b k ↪ u k +1 ), u k +1 ∈ U k +1 , and select the observation whose Q-factor is minimal; this idea has been used in the case of the Wordle puzzle in the papers by Bhambri, Bhattacharjee, and Bertsekas [BBB22], [BBB23], which will be discussed in the next section.

## Multiagent Rollout for Bayesian Optimization

In some BO applications there arises the possibility of simultaneously performing multiple observations before receiving feedback about the corresponding observation outcomes. This occurs, among others, in two important contexts:

- (a) In parallel computation settings, where multiple processors are used to perform simultaneously expensive evaluations of the function f at multiple points u . These evaluations may involve some form of truncated simulation, so they yield evaluations of the form z u = θ u + w u , where w u is the simulation noise.
- (b) In distributed sensor systems, where a number of sensors provide in parallel relevant information about the random vector θ that we want to estimate; see e.g., the recent paper by Li, Krakow, and Gopalswamy [LKG21], which describes related multisensor estimation problems, based on the multiagent rollout methodology of Section 2.9.

Of course in such cases we may treat the entire set of simultaneous observations as a single observation within an enlarged Cartesian product space of observations, but there is a fundamental di ffi culty: the size of the observation space (and hence the number of Q-factors to be calculated by rollout at each time step) grows exponentially with the number of simultaneous observations. This in turn greatly increases the computational requirements of the rollout algorithm.

To address this di ffi culty, we may employ the methodology of multiagent rollout whereby the policy improvement is done one-agent-at-a-time in a given order, with (possibly partial) knowledge of the choices of the preceding agents in the order. As a result, the amount of computation for each policy improvement grows linearly with the number of agents, as opposed to exponentially for the standard all-agents-at-once method. At the same time the theoretical cost improvement property of the rollout algorithm can be shown to be preserved, while the empirical evidence suggests that great computational savings are achieved with hardly any performance degradation.

## Generalization to Sequential Estimation of Random Vectors

Aside from BO, there are several other types of simple sequential estimation problems, which involve 'independent sampling,' i.e., problems where the choice of an observation type does not a ff ect the quality, cost, or availability of observations of other types. A common class of problems that contains BO as a special case and admits a similar treatment, is to sequentially estimate an m -dimensional random vector θ = ( θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m ) by using N linear observations of θ of the form

<!-- formula-not-decoded -->

where n is some integer. Here w u are independent random variables with given probability distributions, the m -dimensional vectors a u are known, and a ′ u θ denotes the inner product of a u and θ . Similar to the case of BO, the problem simplifies if the given a priori distribution of θ is Gaussian, and the random variables w u are independent and Gaussian. Then, the posterior distribution of θ , given any subset of observations, is Gaussian (thanks to the linearity of the observations), and can be calculated in closed form.

Observations are generated sequentially at times 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N , one at a time and with knowledge of the outcomes of the preceding observations, by choosing an index u k ∈ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ at time k , at a cost c ( u k ). Thus u k are the optimization variables, and a ff ect both the quality of estimation of θ and the observation cost. The objective, roughly speaking, is to select N observations to estimate θ in a way that minimizes an appropriate cost function; for example, one that penalizes some form of estimation

error plus the cost of the observations. We can similarly formulate the corresponding optimization problem in terms of N -stage DP, and develop rollout algorithms for its approximate solution.

## 2.11 ADAPTIVE CONTROL BY ROLLOUT WITH A POMDP FORMULATION

In this section, we discuss various approaches for the approximate solution of Partially Observed Markovian Decision Problems (POMDP) with a special structure, which is well-suited for adaptive control, as well as other contexts that involve search for a hidden object. It is well known that POMDP are among the most challenging DP problems, and nearly always require the use of approximations for (suboptimal) solution.

The application and implementation of rollout and approximate PI methods to general finite-state POMDP is described in the author's RL book [Ber19a] (Section 5.7.3). Here we will focus attention on a special class of POMDP where the state consists of two components:

- (a) A perfectly observed component x k that evolves over time according to a discrete-time equation.
- (b) An unobserved component θ that stays constant and is estimated through the perfect observations of the component x k .

We view θ as a parameter in the system equation that governs the evolution of x k , hence the connection with adaptive control. Thus we have

<!-- formula-not-decoded -->

where u k is the control at time k , selected from a set U k ( x k ), and w k is a random disturbance with given probability distribution that depends on ( x k ↪ θ ↪ u k ). We will assume that θ can take one of m known values θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m :

<!-- formula-not-decoded -->

see Fig. 2.11.1.

The a priori probability distribution of θ is given and is updated based on the observed values of the state components x k and the applied controls u k . In particular, we assume that the information vector

<!-- formula-not-decoded -->

In Section 1.6.8, we discussed the indirect adaptive control approach, which enforces a separation of the controller into a system identification algorithm and a policy reoptimization algorithm. The POMDP approach of this section (also summarized in Section 1.6.6), does not assume such an a priori separation, and is thus founded on a more principled algorithmic framework.

Figure 2.11.1 Illustration of an adaptive control scheme involving perfect state observation of a system with an unknown parameter θ . At each time a decision is made to select a control and (possibly) one of several observation types, each of di ff erent cost.

<!-- image -->

is available at time k , and is used to compute the conditional probabilities

<!-- formula-not-decoded -->

These probabilities form a vector

<!-- formula-not-decoded -->

which together with the perfectly observed state x k , form the pair ( x k ↪ b k ) that is commonly called the belief state of the POMDP at time k .

Note that according to the classical methodology of POMDP (see e.g., [Ber17a], Chapter 4), the belief component b k +1 is determined by the belief state ( x k ↪ b k ), the control u k , and the observation obtained at time k +1, i.e., x k +1 . Thus b k can be updated according to an equation of the form

<!-- formula-not-decoded -->

where B k is an appropriate function, which can be viewed as a recursive estimator of θ . There are several approaches to implement this estimator (perhaps with some approximation error), including the use of Bayes' rule and the simulation-based method of particle filtering.

The preceding mathematical model forms the basis for a classical adaptive control formulation, where each θ i represents a set of system parameters, and the computation of the belief probabilities b k↪i can be viewed as the outcome of a system identification algorithm. In this context, the problem becomes one of dual control , a classical type of combined identification and control problem, whose optimal solution is notoriously di ffi cult.

Another interesting context arises in search problems, where θ specifies the locations of one or more objects of interest within a given space. Some puzzles, including the popular Wordle game, fall within this category, as we will discuss briefly later in this section.

## The Exact DP Algorithm - Approximation in Value Space

We will now describe an exact DP algorithm that operates in the space of information vectors I k . To describe this algorithm, let us denote by J k ( I k ) the optimal cost starting at information vector I k at time k . We can view I k as a state of the POMDP, which evolves over time according to the equation

<!-- formula-not-decoded -->

Viewing this as a system equation, whose right hand side involves the state I k , the control u k , and the disturbance w k , the DP algorithm takes the form

<!-- formula-not-decoded -->

for k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, with J N ( I N ) = g N ( x N ); see e.g., the DP textbook [Ber17a], Section 4.1.

The algorithm (2.101) is typically very hard to implement, in part because of the dependence of J ∗ k +1 on the entire information vector I k +1 , which expands in size according to

<!-- formula-not-decoded -->

To address this di ffi culty, we may use approximation in value space, based on replacing J ∗ k +1 ( I k +1 ) in the DP algorithm (2.101) with some function ˜ J k +1 ( I k +1 ) such that the expected value

<!-- formula-not-decoded -->

can be obtained with a tractable computation for any ( I k ↪ u k ). A useful possibility arises when the cost function approximations

<!-- formula-not-decoded -->

can be obtained for each fixed value of θ i with a tractable computation. In this case, we may compute the cost function approximation (2.102) by using the formula

<!-- formula-not-decoded -->

which follows from the law of iterated expectations,

<!-- formula-not-decoded -->

We will now discuss some choices of functions ˜ J k +1 with a structure that facilitates the implementation of the corresponding approximation in value space scheme. One possibility is to use the optimal cost functions corresponding to the m parameters θ i ,

<!-- formula-not-decoded -->

In particular, ˆ J i k +1 ( x k +1 ) is the optimal cost that would be obtained starting from state x k +1 under the assumption that θ = θ i ; this corresponds to a perfect state information problem. Then an approximation in value space scheme with one-step lookahead minimization is given by

<!-- formula-not-decoded -->

In particular, instead of the optimal control, which minimizes the optimal Q-factor of ( I k ↪ u k ) appearing in the right side of Eq. (2.101), we apply control ˜ u k that minimizes the expected value over θ of the optimal Qfactors that correspond to fixed values of θ .

In the case where the horizon is infinite, it is reasonable to expect that an improving estimate of the parameter θ can be obtained over time, and that with a suitable estimation scheme, it converges asymptotically to the correct value of θ , call it θ ∗ , i.e.,

/negationslash

<!-- formula-not-decoded -->

Then it can be seen that the generated one-step lookahead controls ˜ u k are asymptotically obtained from the Bellman equation that corresponds to the correct parameter θ ∗ , and are typically optimal in some asymptotic sense. Schemes of this type have been extensively discussed in the adaptive control literature since the 70s; see the end-of-chapter references and discussion.

Generally, the optimal costs ˆ J i k +1 ( x k +1 ) of Eq. (2.103), which correspond to the di ff erent parameter values θ i , may be hard to compute, despite the fact that they correspond to perfect state information problems. An alternative possibility is to use o ff -line trained approximations to ˆ J i k +1 ( x k +1 ) involving neural networks or other approximation architectures. Still another possibility, described next, is to use a rollout approach.

In favorable special cases, such as linear quadratic problems, the optimal costs ˆ J i k +1 ( x k +1 ) may be easily calculated in closed form. Still, however, even in such cases the calculation of the belief probabilities b k↪i may not be simple, and may require the use of a system identification algorithm.

## Rollout and Cost Improvement

A simpler possibility for approximation in value space is to use the cost of a given policy π i in place of the optimal cost ˆ J i k +1 ( x k +1 ) of Eq. (2.103) that corresponds to θ i . In this case the one-step lookahead scheme (2.104) takes the form

<!-- formula-not-decoded -->

with π i = ¶ θ i 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ i N -1 ♦ , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , being known policies, with components θ i k that depend only on x k . Here, the term

<!-- formula-not-decoded -->

in Eq. (2.105) is the cost of the base policy π i , calculated starting from the next state

<!-- formula-not-decoded -->

under the assumption that θ will stay fixed at the value θ = θ i until the end of the horizon. Note that the cost function of π i , conditioned on θ = θ i , x k , and u k , which is needed in Eq. (2.105), can be calculated by Monte Carlo simulation. This is made possible by the fact that the components θ i k of π i depend only on x k [rather than I k or the belief state ( x k ↪ b k )].

The preceding scheme has the character of a rollout algorithm, but strictly speaking, it does not qualify as a rollout algorithm because the policy components θ i k involve a dependence on i in addition to the dependence on x k . On the other hand if we restrict all the policies π i to be the same for all i , the corresponding functions θ k depend only on x k and not on i , thus defining a legitimate base policy. In this case the rollout scheme (2.105) amounts to replacing

<!-- formula-not-decoded -->

in the DP algorithm (2.101) with

<!-- formula-not-decoded -->

Similar to Section 2.7, a cost improvement property can then be shown.

Within our rollout context, a policy π such that π i = π for all i should be a robust policy, in the sense that it should work adequately well for all parameter values θ i . The method to obtain such a policy is likely problem-dependent. On the other hand robust policies have a long history in the context of adaptive control, and have been discussed widely (see e.g., the book by Jiang and Jiang [JiJ17], and the references quoted therein).

## The Case of a Deterministic System

Let us now consider the case where the system (2.100) is deterministic of the form

<!-- formula-not-decoded -->

Then, while the problem still has a stochastic character due to the uncertainty about the value of θ , the DP algorithm (2.101) and its approximation in value space counterparts are greatly simplified because there is no expectation over w k to contend with. Indeed, given a state x k , a parameter θ i , and a control u k , the on-line computation of the control of the rollout-like algorithm (2.105), takes the form

<!-- formula-not-decoded -->

The computation of ˆ J i k +1 ↪ π i ( f k ( x k ↪ θ i ↪ u k ) ) involves a deterministic propagation from the state x k +1 of Eq. (2.106) up to the end of the horizon, using the base policy π i , while assuming that θ is fixed at the value θ i .

In particular, the term

<!-- formula-not-decoded -->

appearing on the right side of Eq. (2.107) is viewed as a Q-factor that must be computed for every pair ( u k ↪ θ i ), u k ∈ U k ( x k ), i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , using the base policy π i . The expected value of this Q-factor,

<!-- formula-not-decoded -->

must then be calculated for every u k ∈ U k ( x k ), and the computation of the rollout control ˜ u k is obtained from the minimization

<!-- formula-not-decoded -->

cf. Eq. (2.107). This computation is illustrated in Fig. 2.11.2.

The case of a deterministic system is particularly interesting because we can typically expect that the true parameter θ ∗ is identified in a finite number of stages, since at each stage k , we are receiving a noiseless measurement relating to θ , namely the state x k . Once this happens, the problem becomes one of perfect state information.

An illustration similar to the one of Fig. 2.11.2 applies to the rollout scheme (2.105) for the case of a stochastic system. In this case, a Q-factor

<!-- formula-not-decoded -->

XO

X1

Xk

Next States

01

Xk +1

Base Policy 71

02

Final States

Base Policy 72

Figure 2.11.2 Schematic illustration of adaptive control by rollout for deterministic systems; cf. Eqs. (2.108) and (2.109). The Q-factors Q k ( x k ↪ u k ↪ θ i ) are averaged over θ i , using the current belief distribution b k , and the control applied is the one that minimizes over u k ∈ U k ( x k ) the averaged Q-factor

<!-- image -->

<!-- formula-not-decoded -->

must be calculated for every triplet ( u k ↪ θ i ↪ w k ), using the base policy π i . The rollout control ˜ u k is obtained by minimizing the expected value of this Q-factor [averaged using the distribution of ( θ ↪ w k )]; cf. Eq. (2.105).

An interesting and intuitive example that demonstrates the deterministic system case is the popular Wordle puzzle.

## Example 2.11.1 (The Wordle Puzzle)

In the classical form of this puzzle, we try to guess a mystery word θ ∗ out of a known finite collection of 5-letter words. This is done with sequential guesses each of which provides additional information on the correct word θ ∗ , by using certain given rules to shrink the current mystery list (the smallest list that contains θ ∗ , based on the currently available information). The objective is to minimize the number of guesses to find θ ∗ (using more than 6 guesses is considered to be a loss). This type of puzzle descends from the classical family of Mastermind puzzles that centers around decoding a secret sequence of objects (e.g., letters or colors) using partial observations.

The rules for shrinking the mystery list relate to the common letters between the word guesses and the mystery word θ ∗ , and they will not be described here (there is a large literature regarding the Wordle puzzle). More-

Observation Type Selection

System Observation Outcome Decision

System Observation Outcome Decision on Next Observation

Figure 2.11.3 A view of sequential estimation as an adaptive control problem. The system function f k does not depend on the current state x k , so the system provides a decision-dependent noisy observation of θ .

<!-- image -->

over, θ ∗ is assumed to be chosen from the initial collection of 5-letter words according to a uniform distribution. Under this assumption, it can be shown that the belief distribution b k at stage k continues to be uniform over the mystery list. As a result, we may use as state x k the mystery list at stage k , which evolves deterministically according to an equation of the form (2.106), where u k is the guess word at stage k . There are several base policies to use in the rollout-like algorithm (2.107), which are described in the papers by Bhambri, Bhattacharjee, and Bertsekas [BBB22], [BBB23], together with computational results, which show that the corresponding rollout algorithm (2.107) performs remarkably close to the optimal policy (first obtained with a very computationally intensive exact DP calculation by Selby in 2022).

The rollout approach also applies to several variations of the Wordle puzzle. Such variations may include for example a larger length /lscript &gt; 5 of mystery words, and/or a known nonuniform distribution over the initial collection of /lscript -letter words; see [BBB22].

## The Case of Sequential Estimation - Alternative Base Policies

We finally note that the adaptive control framework of this section contains as a special case the sequential estimation framework of the preceding section. Here the problem formulation involves a dynamic system of the form

<!-- formula-not-decoded -->

where the state x k +1 is the observation at time k +1 and exhibits no explicit dependence on the preceding observation x k , but depends on the stochastic disturbance w k , and on the decision u k ; cf. Figs. 2.11.1 and 2.11.3. This decision may involve a cost and determines the type of next observation out of a collection of possible types.

Observation Type Selection

While the rollout methodology of the present section applies to sequential estimation problems, other rollout algorithms may also be used, depending on the problem's detailed structure. In particular, the rollout algorithms for Bayesian optimization of the works noted in Section 2.10 involve base policies that depend on the current belief state b k , rather than the current state x k . Another example of rollout for adaptive control, which uses a base policy that depends on the current belief state is given in Section 6.7 of the book [Ber22a]. For work on related stochastic optimal control problems that involve observation costs and the rollout approach, see Antunes and Heemels [AnH14], and Khashooei, Antunes, and Heemels [KAH15].

## 2.12 MINIMAX CONTROL AND REINFORCEMENTLEARNING

The problem of optimal control of uncertain systems is usually treated within a stochastic framework, whereby all disturbances w k are described by probability distributions, and the expected value of the cost is minimized. However, in many practical situations a stochastic description of the disturbances may not be available, but one may have information with less detailed structure, such as bounds on their magnitude. In other words, one may know a set within which the disturbances are known to lie, but may not know the corresponding probability distribution. Under these circumstances we can use a minimax approach, i.e., try to minimize a cost function assuming that the worst possible values of the disturbances will occur. In this approach, we assume an antagonistic opponent, called the maximizer , who chooses w k with the aim to maximize the cost. The controller, who chooses the controls u k with the aim to minimize the cost, will be referred to as the minimizer .

In this section, we consider a variety of RL schemes for minimax control. We start with exact DP and the corresponding Bellman equation for finite and infinite horizon problems, in order to provide the foundation for approximate DP/RL approaches. The main di ff erence from the stochastic control problems we have discussed earlier is that the disturbance choices w k are made by maximization rather than by randomization. Accordingly, the expected value operation is replaced by maximization over w k in the

The minimax approach to decision and control has its origins in the 50s and 60s. It is also referred to by other names, depending on the underlying context, such as robust control , robust optimization , control with a set membership description of the uncertainty , and games against nature . In this book, we will be using the 'minimax control' name. The minimax approach is also connected with two-player games, when in lack of information about the opponent, we adopt a worst case viewpoint during on-line play, as well as with contexts where we wish to guard against adversarial attacks.

DP algorithm and in Bellman's equation. We then turn to approximate DP and we discuss two distinct RL schemes.

The first RL scheme (Section 2.12.2) is similar to the ones we have considered so far in this chapter. We approximate the optimal cost function J * (or cost functions J * 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ J * N -1 , in the case of a finite horizon) by an approximation ˜ J (or approximations ˜ J 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ J N -1 , respectively), and we assume that the minimizer chooses the control u k by one-step or multistep lookahead minimization, while the maximizer chooses the disturbances by exact cost maximization. Rollout is the special case where ˜ J is equal to the cost function of some policy for the minimizer.

The second RL scheme (Section 2.12.3) is based on approximation in policy space, in addition to approximation in value space. We again introduce a cost function approximation ˜ J . However, instead of choice by maximization, we introduce a policy ν ( x↪ u ) for the maximizer, i.e., a rule by which the disturbance w is chosen at state-control pair ( x↪ u ). Naturally, the maximizer's policy ν is chosen to emulate approximately maximizing selections, and we will discuss a few possibilities along this line. However, once the maximizer's policy ν has been chosen, the minimizer's problem becomes a one-player optimization that can be dealt with by using the methods that we have discussed so far in this chapter. This brings to bear the full spectrum of approximation in value space techniques of the preceding sections, including problem approximation and various types of rollout. We discuss briefly an extension of this second RL scheme to distributionally robust control, a set of models that receives much attention at present. Finally, in Section 2.12.4, we illustrate the scheme of Section 2.12.3 within the context of computer chess and other two-person games for which suitable policies ν for the maximizer can be implemented through sophisticated computer engines.

## 2.12.1 Exact Dynamic Programming for Minimax Problems

Let us first consider a finite horizon case, and assume that the disturbances w 0 ↪ w 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 do not have a probabilistic description, but rather are known to belong to corresponding given sets W k ( x k ↪ u k ), k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, which may depend on the current state x k and control u k . The minimax control problem is to find a policy π = ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ with θ k ( x k ) ∈ U k ( x k ) for all x k and k , which minimizes the cost function

<!-- formula-not-decoded -->

The DP algorithm for this problem takes the following form, which resembles the one corresponding to the stochastic DP problem (maximization is used in place of expectation):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This algorithm can be explained by using a principle of optimality type of argument. In particular, we consider the tail subproblem whereby we are at state x k at time k , and we wish to minimize the 'cost-to-go'

<!-- formula-not-decoded -->

We argue that if π ∗ = ¶ θ ∗ 0 ↪ θ ∗ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ ∗ N -1 ♦ is an optimal policy for the minimax problem, then the tail of the policy ¶ θ ∗ k ↪ θ ∗ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ ∗ N -1 ♦ is optimal for the tail subproblem. The optimal cost of this subproblem is J ∗ k ( x k ), as given by the DP algorithm (2.110)-(2.111). The algorithm expresses the intuitive fact that when at state x k at time k , then regardless of what happened in the past, we should choose u k that minimizes the worst/maximum value over w k of the sum of the current stage cost plus the optimal cost of the tail subproblem that starts from the next state. This argument requires a mathematical proof, which turns out to involve a few fine points. For a detailed mathematical derivation, we refer to the author's textbook [Ber17a], Section 1.6. However, the DP algorithm (2.110)-(2.111) is correct assuming finite state and control spaces, among other cases.

## Minimax Control and Zero-Sum Game Theory

The theory of minimax control is intimately connected with the theory of dynamic zero-sum game problems, which essentially involve two minimax control problems:

- (a) The min-max problem , where the minimizer chooses a policy first and the maximizer chooses a policy second with knowledge of the minimizer's policy. The DP algorithm for this problem has the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (b) The max-min problem , where the maximizer chooses policy first and the minimizer chooses policy second with knowledge of the maximizer's policy. The DP algorithm for this problem has the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A basic and easily seen fact is that

Max-Min optimal value ≤ Min-Max optimal value glyph[triangleright]

There is an extensive and time-honored theory for dynamic zero-sum games, which is focused on conditions that guarantee that

<!-- formula-not-decoded -->

However, this question is of limited interest in engineering contexts that involve worst-case design. Moreover, the validity of the minimax equality (2.112) is beyond the range of practical RL, and thus will not be discussed here. The main reason is that once approximations are introduced, the delicate assumptions that guarantee the minimax equality are disrupted.

## Minimax Control Over an Infinite Horizon

The formulation of the infinite horizon version of the preceding minimax control problem follows similar lines to its stochastic counterpart (cf. Section 1.4). The system equation f , cost per stage g , control constraint sets U , and disturbance constraint sets W do not depend on the time k . For a discounted version of the problem, the cost function of a policy π = ¶ θ 0 ↪ θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ has the form

<!-- formula-not-decoded -->

where α &lt; 1 is the discount factor. When the range of values of the stage cost g is bounded, the discount factor guarantees that the limit defining J π ( x 0 ) exists and is finite. It can then be shown that the optimal cost function J ∗ also takes finite values, and is the unique solution of the Bellman equation

<!-- formula-not-decoded -->

Moreover, a stationary policy θ ∗ policy is optimal if and only if θ ∗ ( x ) attains the minimum in Bellman's equation for all x . Straightforward analogs of the value and policy iteration algorithms are also valid under the same circumstances. These results follow from general analyses of abstract DP models under conditions that guarantee that the Bellman operator is a contraction mapping; see the books [Ber12], [Ber22b].

The policy iteration algorithm involves some subtleties and requires modifications, which have been the subject of quite a bit of research. We will not discuss this issue here; see the author's paper [Ber21c] and book [Ber22b] (Chapter 5).

Undiscounted problems, where α = 1, usually involve a special costfree termination state. They are generally more complicated than discounted problems, but the preceding results hold under the assumption that there is a bound on the number of stages needed to reach the termination state, regardless of the choices of the minimizer and the maximizer. This condition is fulfilled, for example, in many computer games, like chess, which will be discussed later in this section.

## 2.12.2 Minimax Approximation in Value Space and Rollout

The approximation ideas for stochastic optimal control that we have discussed in this chapter are also relevant within the minimax context. In particular, for finite horizon problems, approximation in value space with one-step lookahead applies at state x k a control

<!-- formula-not-decoded -->

where ˜ J k +1 ( x k +1 ) is an approximation to the optimal cost-to-go J ∗ k +1 ( x k +1 ) from state x k +1 .

Rollout is obtained when this approximation is the tail cost of some base policy π = ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ :

<!-- formula-not-decoded -->

Given π , we can compute J k +1 ↪ π ( x k +1 ) by solving a deterministic maximization DP problem with the disturbances w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 playing the role of 'optimization variables/controls.' For finite state, control, and disturbance spaces, this is a longest path problem defined on an acyclic graph, since the control variables u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 are determined by the base policy. It is then straightforward to implement rollout: at x k we generate all next states of the form

<!-- formula-not-decoded -->

corresponding to all possible values of u k ∈ U k ( x k ) and w k ∈ W k ( x k ↪ u k ). We then run the maximization/longest path problem described above to compute J k +1 ↪ π ( x k +1 ) from each of these possible next states x k +1 . Finally, we obtain the rollout control ˜ u k by solving the minimax problem in Eq. (2.115). Moreover, it is possible to use truncated rollout to approximate the tail cost of the base policy J k +1 ↪ π ( x k +1 ).

Note that like all rollout algorithms, the minimax rollout algorithm is well-suited for on-line replanning in problems where data may be changing or may be revealed during the process of control selection.

For a more detailed discussion of this implementation, see the author's paper [Ber19b] (Section 5.4).

We mentioned earlier that deterministic problems allow a more general form of rollout, whereby we may use a base heuristic that need not be a legitimate policy, i.e., it need not be sequentially consistent. For cost improvement it is su ffi cient that the heuristic be sequentially improving. A similarly more general view of rollout is not easily constructed for stochastic problems, but is possible for minimax control.

In particular, suppose that at any state x k there is a heuristic that generates a sequence of feasible controls and disturbances, and corresponding states,

<!-- formula-not-decoded -->

with corresponding cost

<!-- formula-not-decoded -->

Then the rollout algorithm applies at state x k a control

<!-- formula-not-decoded -->

This does not preclude the possibility that the disturbances w k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 are chosen by an antagonistic opponent, but allows more general choices of disturbances, obtained for example, by some form of approximate maximization. For example, when the disturbance involves multiple components, w k = ( w 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w m k ), corresponding to multiple opponent agents, the heuristic may involve an agent-by-agent maximization strategy .

The sequential improvement condition, similar to the deterministic case, is that for all x k and k ,

<!-- formula-not-decoded -->

It guarantees cost improvement, i.e., that for all x k and k , the rollout policy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, generally speaking, minimax rollout is fairly similar to rollout for deterministic as well as stochastic DP problems. The main di ff erence with deterministic (or stochastic) problems is that to compute the Q-factor of a control u k , we need to solve a maximization problem, rather than carry out a deterministic (or Monte-Carlo, respectively) simulation with the given base policy.

satisfies

## Example 2.12.1 (Pursuit-Evasion Problems)

Consider a pursuit-evasion problem with state x k = ( x 1 k ↪ x 2 k ), where x 1 k is the location of the minimizer/pursuer and x 2 k is the location of the maximizer/evader, at stage k , in a (finite node) graph defined in two- or threedimensional space. There is also a cost-free and absorbing termination state that consists of a subset of pairs ( x 1 ↪ x 2 ) that includes all pairs with x 1 = x 2 , and possibly some other pairs for which x 1 and x 2 are 'close enough' for the pursuer to capture the evader.

The pursuer chooses one out of a finite number of actions u k ∈ U k ( x k ) at each stage k , when at state x k , and if the state is x k and the pursuer selects u k , the evader may choose from a known set X k +1 ( x k ↪ u k ) of next states x k +1 , which depends on ( x k ↪ u k ). The objective of the pursuer is to minimize a nonnegative terminal cost g ( x 1 N ↪ x 2 N ) at the end of N stages (or reach the termination state, which has cost 0 by assumption). A reasonable base policy for the pursuer can be precomputed by DP as follows: given the current (nontermination) state x k = ( x 1 k ↪ x 2 k ), make a move along the path that starts from x 1 k and minimizes the terminal cost after N -k stages, under the assumption that the evader will stay motionless at his current location x 2 k . (In a variation of this policy, the DP computation is done under the assumption that the evader will follow some nominal sequence of moves.)

For the on-line computation of the rollout control, we need the maximal value of the terminal cost that the evader can achieve starting from every x k +1 ∈ X k +1 ( x k ↪ u k ), assuming that the pursuer will follow the base policy (which has already been computed). We denote this maximal value by ˜ J k +1 ( x k +1 ). The required values ˜ J k +1 ( x k +1 ) can be computed by an ( N -k )-stage DP computation involving the optimal choices of the evader, while assuming the pursuer uses the (already computed) base policy. Then the rollout control for the pursuer is obtained from the minimization

<!-- formula-not-decoded -->

Note that the preceding algorithm can be adapted for the imperfect information case where the pursuer knows x 2 k imperfectly. This is possible by using a form of assumed certainty equivalence: the pursuer's base policy and the evader's maximization can be computed by using an estimate of the current location x 2 k instead of the unknown true location.

In the preceding pursuit-evasion example, the choice of the base policy was facilitated by the special structure of the problem. Generally, however, finding a suitable base policy that can be conveniently implemented is an important problem-dependent issue.

## Variants of Minimax Rollout

Several of the variants of rollout discussed earlier have analogs in the minimax context, e.g., truncation with terminal cost approximation, multistep

and selective step lookahead, and multiagent rollout. In particular, in the /lscript -step lookahead variant, we solve the /lscript -stage problem

<!-- formula-not-decoded -->

we find an optimal solution ˜ u k ↪ ˜ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ k + /lscript -1 , and we apply the first component ˜ u k of that solution. As an example, this type of problem is solved at each move of some chess programs, where the terminal cost function is encoded through a position evaluator. In fact when multistep lookahead is used, special techniques such as alpha-beta pruning may be used to accelerate the computations by eliminating unnecessary portions of the lookahead graph. These techniques are well-known in the context of the two-person computer game methodology, and are used widely in games such as chess.

It is interesting to note that, contrary to the case of stochastic optimal control, there is an on-line constrained form of rollout for minimax control. Here there are some additional trajectory constraints of the form

<!-- formula-not-decoded -->

where C is an arbitrary set. The modification needed is similar to the one of Section 2.5: at partial trajectory

<!-- formula-not-decoded -->

generated by rollout, we use a heuristic with cost function H k +1 to compute the Q-factor

<!-- formula-not-decoded -->

for each u k in the set ˜ U k (˜ y k ) that guarantee feasibility [we can check feasibility here by running some algorithm that verifies whether the future disturbances w k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 can be chosen to violate the constraint under the base policy, starting from (˜ y k ↪ u k )]. Once the set of 'feasible controls' ˜ U k (˜ y k ) is computed, we can obtain the rollout control by the Q-factor minimization:

<!-- formula-not-decoded -->

We may also use fortified versions of the unconstrained and constrained rollout algorithms, which guarantee a feasible cost-improved rollout policy. This requires the assumption that the base heuristic at the initial state produces a trajectory that is feasible for all possible disturbance sequences.

Truncated and multiagent versions of the minimax rollout algorithm are possible. The following example describes the multiagent case.

## Example 2.12.2 (Multiagent Minimax Rollout)

Let us consider a minimax problem where the minimizer's choice involves the collective decision of m agents, u = ( u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ), with u /lscript corresponding to agent /lscript , and constrained to lie within a finite set U /lscript . Thus u must be chosen from within the set

<!-- formula-not-decoded -->

which is finite but grows exponentially in size with m . The maximizer's choice w is constrained to belong to a finite set W . We consider multiagent rollout for the minimizer, and for simplicity, we focus on a two-stage problem. However, there are straightforward extensions to a more general multistage framework.

In particular, we assume that the minimizer knowing an initial state x 0 , chooses u = ( u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ), with u /lscript ∈ U /lscript , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , and a state transition

<!-- formula-not-decoded -->

occurs with cost g 0 ( x 0 ↪ u ) glyph[triangleright] Then the maximizer, knowing x 1 , chooses w ∈ W , and a terminal state

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The problem is to select u ∈ U , to minimize

<!-- formula-not-decoded -->

The exact DP algorithm for this problem is given by

<!-- formula-not-decoded -->

This DP algorithm is computationally intractable for large m . The reason is that the set of possible minimizer choices u grows exponentially with m , and for each of these choices the value of J ∗ 1 ( f 0 ( x 0 ↪ u ) ) must be computed.

<!-- formula-not-decoded -->

However, the problem can be solved approximately with multiagent rollout, using a base policy θ = ( θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m ). Then the number of times is generated with cost

J ∗ 1 ( f 0 ( x 0 ↪ u ) ) needs to be computed is dramatically reduced. This computation is done sequentially, one-agent-at-a-time, as follows:

<!-- formula-not-decoded -->

When the number of stages is larger than two, a similar algorithm can be used. Essentially, the one-stage maximizer's cost function J ∗ 1 must be replaced by the optimal cost function of a multistage maximization problem, where the minimizer is constrained to use the base policy (see also the paper [Ber19b], Section 5.4).

In this algorithm, the number of times for which J ∗ 1 ( f 0 ( x 0 ↪ u ) ) must be computed grows linearly with m .

An interesting question is how do various algorithms work when approximations are used in the min-max and max-min problems? We can certainly improve the minimizer's policy assuming a fixed policy for the maximizer ; this will be the basis for the scheme of the next section. However, it is unclear how to improve both the minimizer's and the maximizer's policies simultaneously. In practice, in symmetric games , like chess, a common policy is trained for both players. In particular, in the AlphaZero and TD-Gammon programs this strategy is computationally expedient and has worked well. However, there is no reliable theory to guide the simultaneous training of policies for both maximizer and minimizer, and it is quite plausible that unusual behavior may arise in exceptional cases.

Another interesting fact is that even the exact policy iteration method (given in 1969 by Pollatschek and Avi-Itzhak [PoA69]) encounters serious convergence di ffi culties. The reason is that Newton's method applied to solution of the Bellman equation for minimax problems (which is equivalent to policy iteration) exhibits more complex and unreliable behavior than

Indeed such exceptional cases have been reported for the AlphaGo program in late 2022, when humans defeated an AlphaGo look-alike, KataGo, 'by using adversarial techniques that take advantage of KataGo's blind spots' (according to the reports); see Wang et al. [WGB22].

its one-player counterpart. Mathematically, this is because the Bellman operator T for infinite horizon problems, given by

<!-- formula-not-decoded -->

is neither convex nor concave as a function of J . To see this, note that the function viewed as a function of J [for fixed ( x↪ u )], is convex, and when minimized over u ∈ U ( x ), it becomes neither convex nor concave. As a result there are special di ffi culties in connection with convergence of Newton's method and the natural form of policy iteration. The author's paper [Ber21c] and book [Ber22b] (Chapter 5) address these convergence issues with modified versions of the policy iteration method, and give many earlier references on policy iteration and other related methods for minimax problems.

<!-- formula-not-decoded -->

## 2.12.3 Combined Approximation in Value and Policy Space for Minimax Control

We will now discuss ways to simplify the approximation in value space scheme of the preceding section. For an infinite horizon problem, this scheme applies at state x the control

<!-- formula-not-decoded -->

where ˜ J is an approximation to the optimal cost function J ∗ . The di ffi culty here is that the maximization over w ∈ W ( x↪ u ) must be performed for each u ∈ U ( x ), and complicates the minimization over u ∈ U ( x ).

In an alternative scheme that alleviates this di ffi culty, we may introduce an approximation of the maximizer's optimal policy in addition to the cost function approximation ˜ J . This is a policy ν ( x↪ u ) that depends on both x and u , and is used to approximate the minimax scheme of Eq. (2.116) with the minimization scheme

<!-- formula-not-decoded -->

One possibility to obtain ν is o ff -line training, using samples from the maximization

<!-- formula-not-decoded -->

of Eq. (2.116). If the policy ν ( x↪ u ) is optimal for the maximizer, this scheme coincides with the minimax approximation in value space scheme

of the preceding section and inherits the corresponding Newton step-related superlinear cost improvement guarantees. Otherwise, cost improvement is not guaranteed, but can be expected under favorable conditions where ν ( x↪ u ) is near-optimal.

Once ν ( x↪ u ) has been determined, the simplified scheme (2.117) is just a one-player approximation in value space for a problem defined by the system equation and the cost per stage

<!-- formula-not-decoded -->

It is important to note that this problem is deterministic, and is suitable for the special search techniques and simplifications that we have discussed in connection with deterministic problems, including the rollout schemes of Sections 2.3-2.6. Moreover the superlinear error bounds associated with Newton's method (cf. Section 1.5.3), apply to this problem.

<!-- formula-not-decoded -->

## Extension to Distributionally Robust Control Models

There is an important minimax-type model where the disturbance w k is a probability distribution and the set W k ( x k ↪ u k ) is a given set of probability distributions for each pair ( x k ↪ u k ). This problem formulation is known as a distributionally robust model . It dates to the early days of statistics (see e.g., the classical book by Blackwell and Girshick [BlG54], and the more recent book by Chen and Paschalidis [ChP20]). Within the RL context of this section, it can be treated as a special case of the schemes that we have discussed. In particular, by allowing the policy ν ( x↪ u ) to take probability distribution values, the approximation in value space scheme (2.117) applies. Note, however, that in this case, the dynamic system (2.118) is stochastic, and as a result the implementation of the corresponding approximation in value space scheme is more complicated.

## 2.12.4 A Meta Algorithm for Computer Chess Based on Reinforcement Learning

In this section, we will apply the approach of approximating the maximizer's action with a suitably constructed policy to computer chess, based

A subtle but important point here is that in single-player DP contexts [cf. Eq. (2.117)], the Bellman equation has concave components. This turns out to have a beneficial e ff ect on the convergence properties of Newton's method. By contrast, in two-player games the components of Bellman's equation may be neither concave nor convex, and this complicates significantly the convergence properties of Newton's method; see the author's paper [Ber21b] and abstract DP book [Ber22b] (Chapter 5).

on the work by Gundawar, Li, and Bertsekas [GLB24]. A key fact is that for computer chess there are several programs (commonly called engines ), which supply a move selection policy [this is the maximizer's policy approximation ν in Eq. (2.117)]. Engines can also be used to supply board position evaluations [this is the cost function approximation ˜ J in Eq. (2.117)]. ‡

For a brief summary, our scheme is built around two components:

- (a) The position evaluator . This is implemented with one of the many publicly available chess engines, such as the popular and freely available champion program Stockfish, which produces an evaluation of any given position (normalized in a way that is standard in computer chess).
- (b) The nominal opponent . This is an approximation to the true opponent engine or human, whom we intend to play with. It outputs deterministically a move at each given position, against which we expect to play. In the absence of a known opponent, a reasonable choice is to use a competent chess engine as nominal opponent, such as for example the one used to provide position evaluations (e.g., Stockfish). In any case, it is important not to use a relative poor nominal opponent, which would lead us to underestimate the true opponent.

The nominal opponent and the position evaluator may be implemented with di ff erent chess engines. Moreover the nominal opponent may be changed from game to game to adapt to the real opponent at hand. An important fact is that stored knowledge of the opponent and evaluator engines, such as an opening book or an endgame database, are indirectly incorporated into the approximation in value space scheme.

To make the connection with our mathematical framework for minimax problems, we use the following notation:

- ÷ x k is the chess position of our player at time k .
- ÷ u k is the move choice of our player at time k in response to position x k .
- ÷ w k is the move choice of the nominal opponent at time k in response to position x k followed by move u k .

The resulting position at time k +1 is given by

<!-- formula-not-decoded -->

The ideas of the present section apply more broadly to any two-person antagonistic game, which uses computer engines to supply cost function approximation and move selection policies.

‡ We view the approach of this section as a meta algorithm , which is a broad term that describes an algorithm that 'provides a framework or strategy to develop, combine, or enhance other algorithms' (according to ChatGPT).

Current Position

Xk

Move Uk

All Logal moves

88!

88

888

888

$ 4&amp;

Stockfish

888

Nominal Opponent

Stockfish

Position

Evaluation

<!-- image -->

Position Evaluation

Figure 2.12.1 Illustration of the sequence of calculations of a one-step lookahead scheme for computer chess. Here we use the chess engine Stockfish for both opponent move generation and for position evaluation. At the current position x k , we generate all legal moves u k , and for each pair ( x k ↪ u k ), the opponent engine generates a single best move w k , resulting in the position

<!-- formula-not-decoded -->

We then use the position evaluation engine to evaluate each of the possible x k +1 .

where f is a known function. This corresponds to the dynamic system, where x k is viewed as the state, u k is viewed as the control, and w k is viewed as a known or random disturbance.

At the current position x k , our scheme operates as follows:

- (1) We generate all legal moves u k .
- (2) For each pair ( x k ↪ u k ), the nominal opponent generates a single best move w k , resulting in the position

<!-- formula-not-decoded -->

- (3) We evaluate each of the possible positions x k +1 .

Thus, there are a total of at most m position evaluations and m nominal opponent move generations, where m is an upper bound to the number of legal moves at any position.

Next Position

Xk+1

8 ₴

Figure 2.12.1 describes our scheme for the case of one-step lookahead, with Stockfish used for nominal opponent moves and position evaluations. Note that the scheme is well suited for parallel computation. In particular, the moves of the nominal opponent can be computed in parallel, and the positions following the generation of the opponent's moves can also be evaluated in parallel. Thus, with su ffi cient parallel computing resources, our scheme requires roughly twice as much computation as the underlying engines to generate a move.

The paper [GLB24] presents test results using strong chess engines, such as Stockfish, as well as other weaker engines. These results show that, similar to rollout, our scheme improves the performance of the position evaluation engine on which it is based, and for relatively weak engines, dramatically so. This finding generally assumes a reasonable choice for opponent move selection, such as for example the engine that is used for position evaluation. In our tests, a scheme that uses Stockfish of various strength levels as the engine for both nominal opponent and position evaluator has beaten the engine itself by a significant margin, although the winning margin diminishes as the strength of the engine increases. This is to be expected since Stockfish plays near-perfect chess at its highest levels. Qualitatively, this is similar to the performance of approximation in value space and rollout methods, which emulate a Newton step and attain a superlinear convergence, but with the amount of cost improvement diminishing near J * .

## Multistep Lookahead and Other Extensions

It is possible to introduce two-step and multistep lookahead in the preceding algorithm: the lookahead stage of Fig. 2.12.1 is replicated over multiple stages; see Fig. 2.12.2. For an /lscript -step lookahead scheme, the computation needed per move is m + m 2 + · · · + m /lscript opponent move generations plus m /lscript position evaluations. This computation time can be reduced if the lookahead tree is suitably pruned at the second stage. Moreover, it can be seen that given su ffi cient parallel computing resources, to generate a move, an /lscript -step lookahead scheme requires roughly /lscript + 1 times as much computation as the underlying engines. For more details, extensions, and computational testing, see the paper [GLB24].

Let us also note the possibility of replacing the single nominal opponent move with multiple (presumably good) moves. This creates a reduced minimax tree, where the moves at each stage, after the first one, are pruned, except for a few top moves, selected using some form of position evaluation. The player's move at the current position is then found with a minimax search over the reduced tree that consists of the moves that have not been pruned. Note that it is important not to prune any moves of the first stage in order to preserve the Newton step character of the approximation in value space scheme. On the other hand, the (pruned) multistep minimax

Current Position

Tk

Move uk

All Legal moves

888

888

888

888

Stockfish

83

Next Position

Xk+1

888

Nominal Opponent

18.2

82

Figure 2.12.2 Illustration of the two-step lookahead version of the computer chess scheme of Fig. 2.12.1. Again we use Stockfish for both opponent move generation and for position evaluation.

<!-- image -->

search can be costly, and may require a lot of position evaluations and move generations, so for the scheme to be viable, the pruning should be aggressive and the length of the lookahead should be limited.

## 2.12.5 Combined Approximation in Value and Policy Space for Sequential Noncooperative Games

Noncooperative games (also called nonzero-sum games, or Nash games) represent a significant extension of the zero-sum games that we discussed earlier. Here there are a finite number of agents who choose distinct controls, interact with each other, and aim to optimize their private cost functions. The analysis of the agents' choices in the absence of complete or partial information regarding the choices of the other agents has fascinated engineers, mathematicians, and social scientists since the seminal (26 pages!) Ph.D. dissertation of J. Nash [Nas50].

In this section we will discuss a relatively simple special class of nonzero-sum games, which involves sequential choices of the controls of the agents, with complete communication of these controls to the other agents. Still their treatment by DP is very challenging in general. In fact,

Uk+1

Wk+1

Stockfish

Xk+2

Position Evaluation

contrary to the minimax problem, where there is a Bellman equation (albeit complicated by components that are neither convex nor concave; cf. the discussion at the end of Section 2.12.2), for the class of problems of this section there may not be a Bellman equation that can be used as a basis for approximation in value space.

We will now focus on an important example of sequential noncooperative game, a leader-follower problem , also known as a Stackelberg game in economics and as a bilevel optimization problem in mathematical programming. It involves a dominant decision maker, called the leader , and m other decision makers, called the followers . Once the followers observe the decision of the leader, they optimize their choices according to their private cost functions. We make no assumption on the state and control spaces, other than that the constraints of the followers are decoupled.

Let us first consider for simplicity a two-stage stochastic environment with perfect state information, and the special case where both leader and followers act cooperatively in the sense that they minimize a common cost function. Thus, we assume that the leader knowing an initial state x 0 , makes a decision u 0 ∈ U 0 , and a state transition

<!-- formula-not-decoded -->

occurs with cost g 0 ( x 0 ↪ u 0 ↪ w 0 ) ↪ where w 0 is a random disturbance. Then the followers, knowing x 1 , choose decisions u /lscript 1 ∈ U /lscript 1 , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , and a terminal state

<!-- formula-not-decoded -->

is generated with cost

<!-- formula-not-decoded -->

where w 1 is a random disturbance, and g 2 is the terminal cost function. The problem is to select u 0 and follower policies θ /lscript 1 , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , so as to minimize the total expected cost.

Clearly this is a two-stage multiagent DP problem, which can be very hard to solve for large m , because of the large size of the control space of the second stage and the coupling of the followers' actions through the terminal cost. This two-stage problem can be solved approximately with multiagent rollout; see Section 2.9. Moreover, the extension from a two-stage to a multistage framework is straightforward.

However, the multiagent formulation just described fails when the followers do not share with the leader the same cost function. In this case it seems very di ffi cult to apply approximation in value space methods. On the other hand, a convenient approach is to convert the problem to a single

Approximation in policy space approaches are possible, and have been suggested in the literature, but have met with mixed success so far.

agent type of problem, involving just the leader and nominal policies of the followers , similar to the approach of Section 2.12.3, and the computer chess paradigm of Section 2.12.4. These policies, can be obtained using some form of policy training, and produce follower actions u /lscript 1 ∈ U /lscript 1 , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m↪ that depend predictably on the current state x 1 and possibly the preceding action u 0 of the leader. The advantage of this approach is that the Newton step framework fully applies to the leader's problem, and the corresponding on-line implementation of the leader's policy is tractable, based on oneplayer approximation in value space schemes. Limited preliminary results using this approach have been encouraging, but more research along this line is needed.

## 2.13 NOTES, SOURCES, AND EXERCISES

In this chapter, we have first considered deterministic problems, then stochastic problems, and finally minimax problems. We have emphasized the e ff ectiveness of approximation in value space, which derives from its connection to Newton's method. We have discussed several types of approximation in value space schemes, with an emphasis on rollout and its variations, which are very e ff ective, reliable, and easily implementable.

Section 2.1: Our focus in this section has been on finite horizon problems, possibly involving a nonstationary system and cost per stage. However, the insights that can be obtained from the infinite horizon/stationary context fully apply. These include the interpretation of approximation in value space as a Newton step, and of rollout as a single step of the policy iteration method. The reason is that an N -step finite horizon/nonstationary problem can be converted to an infinite horizon/stationary problem with a termination state to which the system moves at the N th stage; see Section 1.6.4.

Section 2.2: Approximation in value space has been considered in an ad hoc manner since the early days of DP, motivated by the curse of dimensionality. Moreover, the idea of /lscript -step lookahead minimization with horizon truncation beyond the /lscript steps has a long history and is often referred to as 'rolling horizon' or 'receding horizon' optimization. Approximation in value space was reframed in the late 80s and was coupled with model-free simulation methods that originated in artificial intelligence.

Section 2.3: The main idea of rollout algorithms, obtaining an improved policy starting from some other suboptimal policy, has appeared in several DP contexts, including games; see e.g., Abramson [Abr90], and Tesauro and Galperin [TeG96]. The name 'rollout' was coined in [TeG96] in the context of backgammon; see Example 2.7.4. The use of the name 'rollout' has gradually expanded beyond its original context; for example samples

collected through trajectory simulation are referred to as 'rollouts' by some authors.

In this book, we will adopt the original intended meaning: rollout is an algorithm that provides policy improvement starting from a base policy, which is evaluated with some form of Monte Carlo simulation, perhaps augmented by some other calculation that may include a terminal cost function approximation. The author's rollout book [Ber20a] provides a more extensive discussion of rollout algorithms and their applications.

Following the original works on rollout for discrete deterministic and stochastic optimization (Bertsekas, Tsitsiklis, and Wu [BTW97], Bertsekas and Casta˜ non [BeC99], and the neuro-dynamic programming book [BeT96]), there has been a lot of research on rollout algorithms, which we list selectively in chronological order: Christodouleas [Chr97], Duin and Voss [DuV99], Secomandi [Sec00], [Sec01], [Sec03], Ferris and Voelker [FeV02], [FeV04], McGovern, Moss, and Barto [MMB02], Savagaonkar, Givan, and Chong [SGC02], Wu, Chong, and Givan [WCG02], [WCG03], Bertsimas and Popescu [BeP03], Guerriero and Mancini [GuM03], Tu and Pattipati [TuP03], Meloni, Pacciarelli, and Pranzo [MPP04], Yan et al. [YDR04], Nedich, Schneider, and Washburn [NSW05], Han, Lai, and Spivakovsky [HLS06], Lee et al. [LSG07], An et al. [ASP08], Berger et al. [BAP08], Besse and Chaib-draa [BeC08], Patek, Breton, and Kovatchev [PBK08], Sun et al. [SZL08], Tian, Bar-Shalom, and Pattipati [TBP08], Novoa and Storer [NoS09], Mishra et al. [MCT10], Malikopoulos [Mal10], Bertazzi et al. [BBG13], Sun et al. [SLJ13], Tesauro et al. [TGL13], Antunes and Heemels [AnH14], Beyme and Leung [BeL14], Goodson, Thomas, and Ohlmann [GTO15], [GTO17], Khashooei, Antunes, Heemels [KAH15], Li and Womer [LiW15], Mastin and Jaillet [MaJ15], Simroth, Holfeld, and Brunsch [SHB15], Huang, Jia, and Guan [HJG16], Lan, Guan, and Wu [LGW16], Lam, Willcox, and Wolpert [LWW16], Gommans et al. [GTA17], Lam and Willcox [LaW17], Ulmer [Ulm17], Bertazzi and Secomandi [BeS18], Sarkale et al. [SNC18], Ulmer at al. [UGM18], Zhang, Ohlmann, and Thomas [ZOT18], Arcari, Hewing, and Zeilinger [AHZ19], Chu, Xu, and Li [CXL19], Goodson, Bertazzi, and Levary [GBL19], Guerriero, Di Puglia, and Macrina [GDM19], Ho, Liu, and Zabinsky [HLZ19], Liu et al. [LLL19], Nozhati et al. [NSE19], Singh and Kumar [SiK19], Yu et al. [YYM19], Yuanhong [Yua19], Andersen, Stidsen, and Reinhardt [ASR20], Durasevic and Jakobovic [DuJ20], Issakkimuthu, Fern, and Tadepalli [IFT20], Lee et al. [LEC20], Li et al. [LZS20], Lee [Lee20], Montenegro et al. [MLM20], Meshram and Kaza [MeK20], Schope, Driessen, and Yarovoy [SDY20], Yan, Wang, and Xu [YWX20], Yue and Kontar [YuK20], Zhang, Kafouros, and Yu [ZKY20], Ho ff man et al. [HCR21], Houy and Flaig [HoF21], Li, Krakow, and Gopalswamy [LKG21], Liu et al. [LPS21], Nozhati [Noz21], Rim­ el­ e et al. [RGG21], Tuncel et al. [TBP21], Xie, Li, and Xu [XLX21], Bertsekas [Ber22d], Paulson, Sonouifar, and Chakrabarty [PSC22], Bai et al. [BLJ23], Rusmevichientong et al. [RST23], Wu and Zeng

[WuZ23], Gerlach and Piatkowski [GeP24], Samani, Hammar, and Stadler [SHS24], Samani et al. [SLD24], Yilmaz, Xiang, and Klein [YXK24], Zhang et al. [ZLZ24], Wang et al. [WTL25].

These references collectively include a large number of computational studies, discuss variants and problem-specific adaptations of rollout algorithms for a broad variety of practical problems, and consistently report favorable computational experience. The size of the cost improvement over the base policy is often impressive, evidently owing to the fast convergence rate of Newton's method that underlies rollout. Moreover these works illustrate some of the other important advantages of rollout: reliability, simplicity, suitability for on-line replanning, and the ability to interface with other RL techniques, such as neural network training, which can be used to provide suitable base policies and/or approximations to their cost functions.

The adaptation of rollout algorithms to discrete deterministic optimization problems, the notions of sequential consistency, sequential improvement, fortified rollout, and the use of multiple heuristics for parallel rollout were first given in the paper by Bertsekas, Tsitsiklis, and Wu [BTW97], and were also discussed in the neuro-dynamic programming book [BeT96]. Rollout algorithms for stochastic problems were further formalized in the papers by Bertsekas [Ber97b], and Bertsekas and Casta˜ non [BeC99]. Extensions to constrained rollout were first given in the author's papers [Ber05a], [Ber05b]. A survey of rollout in discrete optimization was given by the author in [Ber13a].

The model-free rollout algorithm, in the form given here, was first discussed in the RL book [Ber19a]. It is related to the method of comparison training, proposed by Tesauro [Tes89a], [Tes89b], [Tes01], and discussed by several other authors (see [DNW16], [TCW19]). This is a general method for training an approximation architecture to choose between two alternatives, using a dataset of expert choices in place of an explicit cost function.

The material on most likely sequence generation for n -grams, HMMs, and Markov Chains is recent, and was developed in the paper by Li and Bertsekas [LiB24]. As we have noted, our rollout-based most likely sequence generation algorithm can be useful to all contexts where the Viterbi algorithm is used. This includes algorithms for HMM parameter inference, which use the Viterbi algorithm as a subroutine.

Section 2.4: Our discussion of rollout, iterative deepening, and pruning in the context of multistep approximation in value space for deterministic problems contains some original ideas. In particular, the incremental multistep rollout algorithm and variations of Section 2.4.2 are presented here for the first time.

Note also that the multistep lookahead approximations described in Section 2.4 can be used more broadly within algorithms that employ forms of multistep lookahead search as subroutines. In particular, local search

algorithms, such as tabu search, genetic algorithms, and others, which are commonly used for discrete and combinatorial optimization, may be modified along the lines of Section 2.4 to incorporate RL and approximate DP ideas.

Section 2.5: Constrained forms of rollout were introduced in the author's papers [Ber05a] and [Ber05b]. The paper [Ber05a] also discusses rollout and approximation in value space for stochastic problems in the context of so-called restricted structure policies . The idea here is to simplify the problem by selectively restricting the information and/or the controls available to the controller, thereby obtaining a restricted but more tractable problem structure, which can be used conveniently in a one-step lookahead context. An example of such a structure is one where fewer observations are obtained, or one where the control constraint set is restricted to a single or a small number of given controls at each state.

Section 2.6: Rollout for continuous-time optimal control was first discussed in the author's rollout book [Ber20a]. A related discussion of policy iteration, including the motivation for approximating the gradient of the optimal cost-to-go ∇ x J t rather than the optimal cost-to-go J t , has been given in Section 6.11 of the neuro-dynamic programming book [BeT96]. This discussion also includes the use of value and policy networks for approximate policy evaluation and policy improvement for continuous-time optimal control. The underlying ideas have long historical roots, which are recounted in detail in the book [BeT96].

Section 2.7: The idea of the certainty equivalence approximation in the context of rollout for stochastic systems (Section 2.7.2) was proposed in the paper by Bertsekas and Casta˜ non [BeC99], together with extensive empirical justification. However, the associated theoretical insight into this idea was established more recently, through the interpretation of approximation in value space as a Newton step, which suggests that the lookahead minimization after the first step can be approximated with small degradation of performance. This point is emphasized in the author's book [Ber22a], and the papers [Ber22c], [Ber24].

Markov jump problems (Example 2.7.3) have an interesting theory and several diverse applications (see e.g., Sworder [Swo69], Wonham [Won70], Chizeck, Willsky, and Casta˜ non [CWC86], Abou-Kandil, Freiling, and Jank [AKFJ95], Costa and Do Val [CDV02], and Li and Bertsekas [LiB25a]. For related monographs, see Mariton [Mar90], Costa, Fragoso, and Marques [CFM05]). They seem to be particularly well suited for the application of approximation in value space, rollout schemes, and the use of certainty equivalence.

The idea of variance reduction in the context of rollout (Section 2.7.4) was proposed by the author in the paper [Ber97b]. See also the DP textbook [Ber17a], Section 6.5.2.

The paper by Chang, Hu, Fu, and Marcus [CHF05], and the 2007 first edition of their monograph proposed and analyzed adaptive sampling in connection with DP, as well as early forms of Monte Carlo tree search, including statistical tests to control the sampling process (a second edition, [CHF13], appeared in 2013). The name 'Monte Carlo tree search' has become popular, and in its current use, it encompasses a variety of methods that involve adaptive sampling, rollout, and extensions to sequential games. We refer to the papers by Coulom [Cou06], and Chang et al. [CHF13], the discussion by Fu [Fu17], and the survey by Browne et al. [BPW12].

Statistical tests for adaptive sampling has been inspired by works on multiarmed bandit problems; see Lai and Robbins [LaR85], Agrawal [Agr95], Burnetas and Katehakis [BuK97], Meuleau and Bourgine [MeB99], Auer, Cesa-Bianchi, and Fischer [ACF02], Kocsis and Szepesvari [KoS06], Dimitrakakis and Lagoudakis [DiL08], Audibert, Munos, and Szepesvari [AMS09], and Munos [Mun14]. The book by Lattimore and Szepesvari [LaS20] focuses on multiarmed bandit methods, and provides an extensive account of the UCB rule. For recent work on the theoretical properties of the UCB and UCT rules, see Shah, Xie, and Xu [SXX22], and Chang [Cha24].

Adaptive sampling and MCTS may be viewed within the context of a broader class of on-line lookahead minimization techniques, sometimes called on-line search methods. These techniques are based on a variety of ideas, such as random search and intelligent pruning of the lookahead tree. One may naturally combine them with approximation in value space and (possibly) rollout, although it is not necessary to do so (the multistep minimization horizon may extend to the terminal time N ). For representative works, some of which apply to continuous spaces problems, including POMDP, see Hansen and Zilberstein [HaZ01], Kearns, Mansour, and Ng [KMN02], Peret and Garcia [PeG04], Ross et al. [RPP08], Silver and Veness [SiV10], Hostetler, Fern, and Dietterich [HFD17], and Ye et al. [YSH17]. The multistep lookahead approximation ideas of Section 2.4 may also be viewed within the context of on-line search methods.

Another rollout idea for stochastic problems, which we have not discussed in this book, is the open-loop feedback controller (OLFC), a suboptimal control scheme that dates to the 60s; see Dreyfus [Dre65]. The OLFC applies to POMDP as well, and uses an open-loop optimization over the future evolution of the system. In particular, it uses the current information vector I k to determine the belief state b k . It then solves the open-loop problem of minimizing

<!-- formula-not-decoded -->

subject to the constraints

<!-- formula-not-decoded -->

and applies the first control u k in the optimal open-loop control sequence ¶ u k ↪ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ . It is easily seen that the OLFC is a rollout algorithm that uses as base policy the optimal open-loop policy for the problem (the one that ignores any state or observation feedback).

For a detailed discussion of the OLFC, we refer to the author's survey paper [Ber05a] (Section 4) and DP textbook [Ber17a] (Section 6.4.4). The survey [Ber05a] discusses also a generalization of the OLFC, called partial open-loop-feedback-control , which calculates the control input on the basis that some (but not necessarily all) of the observations will in fact be taken in the future, and the remaining observations will not be taken. This method often allows one to deal with those observations that are troublesome and complicate the solution, while taking into account the future availability of other observations that can be reasonably dealt with. A computational case study for hydrothermal power system scheduling is given by Martinez and Soares [MaS02]. A variant of the OLFC, which also applies to minimax control problems, is given in the author's paper [Ber72b], together with a proof of a cost improvement property over the optimal open-loop policy.

Section 2.8: The role of stochastic programming in providing a link between stochastic DP and continuous spaces deterministic optimization is well known; see the texts by Birge and Louveaux [BiL97], Kall and Wallace [KaW94], and Prekopa [Pre95], and the survey by Ruszczynski and Shapiro [RuS03]. Stochastic programming has been applied widely, and there is much to be gained from its combination with RL. The material of this section comes from the author's rollout book [Ber20a], Section 2.5.2. For a computational study that has tested the ideas of this section on a problem of maintenance scheduling, see Hu et al. [HWP22]. For a related application, see Gioia, Fadda, and Brandimarte [GFB24].

Section 2.9: The multiagent rollout algorithm was proposed in the author's papers [Ber19c], [Ber20b]. The paper [Ber21a] provides an extensive overview of this research. For followup work, see the sources given for Section 1.6 of Chapter 1.

Section 2.10: The material on rollout for Bayesian optimization and sequential estimation comes from a recent paper by the author [Ber22d]. This work is also the basis for the adaptive control material of Section 2.11, and has been included in the book [Ber22a]. The paper by Bhambri, Bhattacharjee, and Bertsekas [BBB22] discusses this material for the case of a deterministic system, applies rollout to sequential decoding in the context of the challenging Wordle puzzle, and provides an implementation using some popular base heuristics, with performance that is very close to optimal. For related work see Loxley and Cheung [LoC23].

Section 2.11: The POMDP framework for adaptive control dates to the 60s, and has stimulated substantial theoretical investigations; see Mandl

[Man74], Borkar and Varaiya [BoV79], Doshi and Shreve [DoS80], Kumar and Lin [KuL82], and the survey by Kumar [Kum85]. Some of the pitfalls of performing parameter estimation while simultaneously applying adaptive control have been described by Borkar and Varaiya [BoV79], and by Kumar [Kum83]; see [Ber17a], Section 6.8 for a related discussion.

The papers just mentioned have proposed on-line estimation of the unknown parameter θ and the use at each time period of a policy that is optimal for the current estimate. The papers provide nontrivial analyses that assert asymptotic optimality of the resulting adaptive control schemes under appropriate conditions. If parameter estimation schemes are similarly used in conjunction with rollout, as suggested in Section 2.11 [cf. Eq. (2.105)], one may conjecture that an asymptotic cost improvement property can be proved for the rollout policy, again under appropriate conditions.

Section 2.12: The treatment of sequential minimax problems by DP has a long history. For some early influential works, see Blackwell and Girshick [BlG54], Shapley [Sha53], and Witsenhausen [Wit66]. In minimax control problems, the maximizer is assumed to make choices with perfect knowledge of the minimizer's policy. If the roles of maximizer and minimizer are reversed, i.e., the maximizer has a policy (a sequence of functions of the current state) and the minimizer makes choices with perfect knowledge of that policy, the minimizer gains an advantage, the problem may genuinely change, and the optimal value may be reduced. Thus 'min-max' and 'max-min' are generally two di ff erent problems. In classical two-person zero-sum game theory, however, the main focus is on situations where the min-max and max-min are equal. By contrast, in engineering worst case design, the min-max and max-min values are typically unequal.

There is substantial literature on sequential zero-sum games in the context of DP, often called Markov games . The classical paper by Shapley [Sha53] addresses discounted infinite horizon games. A PI algorithm for finite-state Markov games was proposed by Pollatschek and Avi-Itzhak [PoA69], and was interpreted as a Newton method for solving the associated Bellman equation. They have also shown that the algorithm may not converge to the optimal cost function. Computational studies have verified that the Pollatschek and Avi-Itzhak algorithm converges much faster than its competitors, when it converges (see Breton et al. [BFH86], and also Filar and Tolwinski [FiT91], who proposed a modification of the algorithm). Related methods have been discussed for Markov games by van der Wal [Van78] and Tolwinski [Tol89]. Raghavan and Filar [RaF91], and Filar and Vrieze [FiV96] provide extensive surveys of the research up to 1996.

The author's paper [Ber21b] has explained the reason behind the unreliable behavior of the Pollatschek and Avi-Itzhak algorithm. This explanation relies on the Newton step interpretation of PI given in Chapter 1: in the case of Markov games, the Bellman operator does not have the concavity property that is typical of one-player games. The paper [Ber21b] has

also provided a modified algorithm with solid convergence properties, which applies to very general types of sequential zero-sum games and minimax control. The algorithm is based on the idea of constructing a special type of uniform contraction mapping, whose fixed point is the solution of the underlying Bellman equation; this idea was first suggested for discounted and stochastic shortest path problems by Bertsekas and Yu [BeY12], [YuB13]. Because the uniform contraction property is with respect to a sup-norm, the corresponding PI algorithm is convergent even under a totally asynchronous implementation. The algorithm, its variations, and the analysis of the paper [Ber21b] were incorporated as Chapter 5 in the 3rd edition of the author's abstract DP book [Ber22b].

The paper by Yu [Yu14] provides an analysis of stochastic shortest path games, where the termination state may not be reachable under some policies, following earlier work by Patek and Bertsekas [PaB99]. The paper [Yu14] also includes a rigorous analysis of the Q-learning algorithm for stochastic shortest path games (without any cost function approximation). The papers by Perolat et al. [PSP15], [PPG16], and the survey by Zhang, Yang, and Basar [ZYB21] discuss alternative RL methods for games.

The author's paper [Ber19b] develops VI, PI, and Dijkstra-like finitely terminating algorithms for exact solution of 'robust shortest path planning' problems, which involve finding a shortest path assuming adversarial uncertainty in the state transitions. The paper also discusses related rollout algorithms for approximate solution.

Combining approximation in value and policy space for minimax control (Section 2.12.3) and noncooperative games (Section 2.12.5) is formally presented here for the first time, but it is a natural idea, which undoubtedly has occurred to several researchers. The computer chess methodology of Section 2.12.4 is based on this idea, and was introduced in the paper by Gundawar, Li, and Bertsekas [GLB24], as an application of model predictive control to computer chess.

but from AC it generates

## E X E R C I S E S

## 2.1 (A Traveling Salesman Rollout Example with a Sequentially Improving Heuristic)

Consider the traveling salesman problem of Example 1.2.3 and Fig. 1.2.11, and the rollout algorithm starting from city A.

- (a) Assume that the base heuristic is chosen to be the farthest neighbor heuristic, which completes a partial tour by successively moving to the farthest neighbor city not visited thus far. Show that this base heuristic is sequentially consistent. What are the tours produced by this base heuristic and the corresponding rollout algorithm? Answer : The base heuristic will produce the tour A → AD → ADB → ADBC → A with cost 45. The rollout algorithm will produce the tour A → AB → ABD → ABDC → A with cost 13.
- (b) Assume that the base heuristic at city A is the nearest neighbor heuristic, while at the partial tours AB, AC, and AD it is the farthest neighbor heuristic. Show that this base heuristic is sequentially improving but not sequentially consistent. Compute the final tour generated by rollout.

Solution of part (b): Clearly the base heuristic is not sequentially consistent, since from A it generates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

However, it is seen that the sequential improvement criterion (2.13) holds at each of the states A, AB, AC, and AD (and also trivially for the remaining states).

The base heuristic at A is the nearest neighbor heuristic so it generates

<!-- formula-not-decoded -->

The rollout algorithm at state A looks at the three successor states AB, AC, AD, and runs the farthest neighbor heuristic from each, and generates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so the rollout algorithm will move from A to AB.

Then the rollout algorithm looks at the two successor states ABC, ABD, and runs the base heuristic (whatever that may be; it does not matter) from each. The paths generated are:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so the rollout algorithm will move from AB to ABD.

Thus the final tour generated by the rollout algorithm is

<!-- formula-not-decoded -->

## 2.2 (A Generic Example of a Base Heuristic that is not Sequentially Improving)

Consider a rollout algorithm for a deterministic problem with a base heuristic that produces an optimal control sequence at the initial state x 0 , and uses the (optimal) first control u 0 of this sequence to move to the (optimal) next state x 1 . Suppose that the base heuristic produces a strictly suboptimal sequence from every successor state x 2 = f 1 ( x 1 ↪ u 1 ) ↪ u 1 ∈ U 1 ( x 1 ) ↪ so that the rollout yields a control u 1 that is strictly suboptimal. Show that the trajectory produced by the rollout algorithm starting from the initial state x 0 is strictly inferior to the one produced by the base heuristic starting from x 0 , while the sequential improvement condition does not hold.

## 2.3 (Computational Exercise - Parking with Problem Approximation and Rollout)

In this computational exercise we consider a more complex, imperfect state information version of the one-directional parking problem of Example 1.6.1. Recall that in this problem a driver is looking for a free parking space in an area consisting of N spaces arranged in a line, with a garage at the end of the line (space N ). The driver starts at space 0 and traverses the parking spaces sequentially, i.e., from each space he/she goes to the next space, up to when he/she decides to park in space k at cost c ( k ), if space k is free. Upon reaching the garage, parking is mandatory at cost C .

In Example 1.6.1, we assumed that the driver knows the probabilities p ( k + 1) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ p ( N -1) of the parking spaces ( k + 1) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ( N -1), respectively, being free. Under this assumption, the state at stage k is either the termination state t (if already parked), or it is F (location k free), or it is F (location k taken), and the DP algorithm has the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for the states other than the termination state t , while for t we have J ∗ k ( t ) = 0 for all k .

We will now consider the more complex variant of the problem where the probabilities p (0) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ p ( N -1) do not change over time, but are unknown to

the driver, so that he/she cannot use the exact DP algorithm (2.119)-(2.120). Instead, the driver considers a one-step lookahead approximation in value space scheme, which uses empirical estimates of these probabilities that are based on the ratio f k k +1 ↪ where f k is the number of free spaces seen up to space k , after the free/taken status of spaces 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ k has been observed. In particular, these empirical estimates are given by

<!-- formula-not-decoded -->

where f k is the number of free spaces seen up to space k , and γ and ¯ p ( m ) are fixed numbers between 0 and 1. Of course the values f k observed by the driver evolve according to the true (and unknown) probabilities p (0) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ p ( N -1) according to

<!-- formula-not-decoded -->

For the solution of this exercise you may assume any reasonable values you wish for N , p ( m ), ¯ p ( m ), and γ . Recommended values are N ≥ 100, and probabilities p ( m ) and ¯ p ( m ) that are nonincreasing with m .

The decision made by the approximation in value space scheme is to park at space k if and only if it is free and in addition

<!-- formula-not-decoded -->

where ˜ J k +1 ( F ) and ˜ J k +1 ( F ) are the cost-to-go approximations from stage k +1. Consider the following two di ff erent methods to compute ˜ J k +1 ( F ) and ˜ J k +1 ( F ) for use in Eq. (2.123):

- (1) Here the approximate cost function values ˜ J k +1 ( F ) and ˜ J k +1 ( F ) are obtained by using problem approximation, whereby at time k it is assumed that the probabilities of free/taken status at the future spaces m = k + 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N 1 are b k ( m↪f k ), m = k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N 1, as given by Eq. (2.121).
2. --

More specifically, ˜ J k +1 ( F ) and ˜ J k +1 ( F ) are obtained by solving optimally the problem whereby we use the probabilities b k ( m↪f k ) of Eq. (2.121) in place of the unknown p ( m ) in the DP algorithm (2.119)-(2.120):

<!-- formula-not-decoded -->

where ˆ J k +1 ( F ) and ˆ J k +1 ( F ) are given at the last step of the DP algorithm

<!-- formula-not-decoded -->

- (2) Here for each k , the approximate cost function values ˜ J k +1 ( F ) and ˜ J k +1 ( F ) are obtained by using rollout with a greedy base heuristic (park as soon as possible), and Monte Carlo simulation. In particular, according to this greedy heuristic, we have ˜ J k +1 ( F ) = c ( k + 1). To compute ˜ J k +1 ( F ) we generate many random trajectories by running the greedy heuristic forward from space k +1 assuming the probabilities b k ( m +1 ↪ f k ) of Eq. (2.121) in place of the unknown p ( m +1), m = k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, and we average the cost results obtained.
2. (a) Use Monte Carlo simulation to compute the expected cost from spaces 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, when using each of the two schemes (1) and (2).
3. (b) Compare the performance of the schemes of part (a) with the following:
4. (i) The optimal expected costs J ∗ k ( F ) and J ∗ k ( F ) from k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, using the DP algorithm (2.119)-(2.120), and the probabilities p ( m ), m = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, that you used for the random generation of the numbers of free spaces f k [cf. Eq. (2.122)].
5. (ii) The expected costs ˆ J k ( F ) and ˆ J k ( F ) from k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 that are attained by using the greedy base heuristic. Argue that these are given by

<!-- formula-not-decoded -->

- (c) Argue that scheme (1) becomes superior to scheme (2) in terms of cost attained as γ ≈ 1 and ¯ p ( m ) ≈ p ( m ). Are your computational results in rough agreement with this assertion?
- (d) Argue that as γ ≈ 0 and N &gt;&gt; 1, scheme (1) becomes superior to scheme (2) in terms of cost attained from parking spaces k &gt;&gt; 1.
- (e) What happens if the probabilities p ( m ) do not change much with m ?

## 2.4 (Breakthrough Problem with a Random Base Heuristic)

Consider the breakthrough problem of Example 2.3.2 with the di ff erence that instead of the greedy heuristic, we use the random heuristic, which at a given node selects one of the two outgoing arcs with equal probability. Denote by

<!-- formula-not-decoded -->

the probability of success of the random heuristic in a graph of k stages, and by R k the probability of success of the corresponding rollout algorithm. Show that for all k

<!-- formula-not-decoded -->

and that

<!-- formula-not-decoded -->

Conclude that R k glyph[triangleleft]D k increases exponentially with k .

## 2.5 (Breakthrough Problem with Truncated Rollout)

Consider the breakthrough problem of Example 2.3.2 and consider a truncated rollout algorithm that uses a greedy base heuristic with /lscript -step lookahead. This is the same algorithm as the one described in Example 2.3.2, except that if both outgoing arcs of the current node at stage k are free, the rollout algorithm considers the two end nodes of these arcs, and from each of them it runs the greedy algorithm for min ¶ l↪ N -k -1 ♦ steps. Consider a Markov chain with l + 1 states, where states i = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ l -1 correspond to the path generated by the greedy algorithm being blocked after i arcs. State /lscript corresponds to the path generated by the greedy algorithm being unblocked after /lscript arcs.

- (a) Derive the transition probabilities for this Markov chain so that it models the operation of the rollout algorithm.
- (b) Use computer simulation to generate the probability of a breakthrough, and to demonstrate that for large values of N , the optimal value of /lscript is roughly constant and much smaller than N (this can also be justified analytically, by using properties of Markov chains).

## 2.6 (Incremental Truncated Rollout Algorithm for Constraint Programming)

Consider a discrete N -stage optimization problem, involving a tree with a root node s that plays the role of an artificial initial state, and N layers of states x 1 = ( u 0 ) ↪ x 2 = ( u 0 ↪ u 1 ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ), as shown in Fig. 2.1.4. We allow deadend nodes in the graph of this problem, i.e., states that have no successor states, and thus cannot be part of any feasible solution. We also assume that all stage costs as well as the terminal cost are 0. The problem is to find a feasible solution, i.e., a sequence of N transitions through the graph that starts at the initial state s and ends at some node of the last layer of states x N .

- (a) Argue that this is a discrete spaces formulation of a constraint programming problem, such as the one described in Section 2.1.
- (b) Describe in detail an incremental rollout algorithm with /lscript -step lookahead minimization and m -step rollout truncation, which is similar to the IMR algorithm of Section 2.4.2 and operates as follows: The algorithm maintains a connected subtree S that contains the initial state s . The base policy at a state x k either generates a feasible sequence of m arcs starting at x k , where m is an integer that satisfies 1 ≤ m ≤ min ¶ m↪N -k ♦ , or determines that such a sequence does not exist. In the former case the node x k is expanded by adding all of its neighbor nodes to S . In the latter case, the node x k is deleted from S . The algorithm terminates once a state x N of the last layer is added to S .
- (c) Argue that since the algorithm cannot keep deleting nodes indefinitely, one of two things will eventually happen:
- (1) The graph S will be reduced to just the root node s , proving that there is no feasible solution.
- (2) The algorithm will terminate with a feasible solution.

- (d) Suppose that the algorithm is operated so that the selected node x k at each iteration is a leaf node of S , which is at maximum arc distance from s (the number of arcs of the path connecting s and x k is maximized). Show that the subtree S always consists of just a path of nodes, together with all the neighbor nodes of the nodes of the path. Conclude that in this case, the algorithm can be implemented so that it requires O ( Nd ) memory storage, where d is the maximum node degree. How does this algorithm compare with a depth-first search algorithm for finding a feasible solution?
- (e) Describe an adaptation of the algorithm of part (d) for the case where most of the arcs have cost 0 but there are some arcs with positive cost.

## 2.7 (Purchasing Over Time with Multiagent Rollout)

Consider a market that makes available for purchase m products over N time periods, and a buyer that may or may not buy any one of these products subject to cash availability. For each product i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m and time period k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, we denote:

- a i k : The asking price of product i at time k (the case where product i is unavailable for purchase is modeled by setting a i k to ∞ ).
- v i k : The value to the buyer of product i at time k .
- u i k : The decision to buy ( u i k = 1) or not to buy ( u i k = 0) product i at time k .

The conditional distributions P ( a i k +1 ♣ a i k ↪ u i k = 1) and P ( a i k +1 ♣ a i k ↪ u i k = 0) are given. (Thus when u i k = 1, product i will be made available at the next time period at a possibly di ff erent price a i k +1 ; however, it may also be unavailable, i.e., a i k +1 = ∞ .)

The amount of cash available to the buyer at time k is denoted by c k , and evolves according to

<!-- formula-not-decoded -->

The initially available cash c 0 is a given positive number. Moreover, we have the constraint

<!-- formula-not-decoded -->

i.e., the buyer may not borrow to buy products. The buyer aims to maximize the total value obtained over the N time periods, plus the remaining cash at time N :

<!-- formula-not-decoded -->

- (a) Formulate the problem as a finite horizon DP problem by identifying the state, control, and disturbance spaces, the system equation, the cost function, and the probability distribution of the disturbance. Write the corresponding exact DP algorithm.

- (b) Introduce a suitable base policy and formulate a corresponding multiagent rollout algorithm for addressing the problem.

## 2.8 (Treasure Hunting Using Adaptive Control and Rollout)

Consider a problem of sequentially searching for a treasure of known value v among n given locations. At each time period we may either select a location i to search at cost c i &gt; 0, or we may stop searching. Moreover, if the search location i is di ff erent from the location j where we currently are, we incur an additional switching cost s ij ≥ 0. If the treasure is at location i , a search at that location will find it with known probability β i &lt; 1. Our initial location is given, and the a priori probabilities p i , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , that the treasure is at location i are also given. We assume that ∑ n i =1 p i &lt; 0, so there is positive probability that there is no treasure at any one of the n locations.

- (a) Formulate the problem as a special case of the adaptive control problem of Section 2.11, with the parameter θ taking one of ( n + 1) values, θ 0 ↪ θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ n , where θ 0 corresponds to the case where there is no treasure at any location, and θ i , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , corresponds to the case where the treasure is at location i . Use as state the current location together with the current probability distribution of θ , and use as control the choice between stopping the search or continuing the search at one of the n locations.
- (b) Consider the special case where there is only one location. Show that the optimal policy is to continue searching up to the point where the conditional expected benefit of the search falls below a certain threshold, and that the optimal cost can be computed very simply. (The proof is given in Example 4.3.1 of the DP textbook [Ber17a].)
- (c) Formulate the rollout algorithm of Section 2.11 with two di ff erent base policies:
- (1) A policy that is optimal among the policies that never switch to another location (they continue to search the same location up to stopping).
- (2) A policy that is optimal among the policies that may stay at the current location or may switch to the location that is most likely to contain the treasure according to the current probability distribution of θ .
- (d) Implement the preceding two rollout algorithms using reasonable problem data of your choice.