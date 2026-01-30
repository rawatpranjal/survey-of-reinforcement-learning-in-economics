# 2.13: Notes, Sources, and Exercises Ch2

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 341-348
**Topics:** exercises, notes, sources, chapter 2

---

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