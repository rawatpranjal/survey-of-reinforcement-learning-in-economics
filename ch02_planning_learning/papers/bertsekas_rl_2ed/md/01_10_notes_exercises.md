# 1.8: Notes, Sources, and Exercises Ch1

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 141-165
**Topics:** exercises, notes, sources, chapter 1

---

-Factors Current State

Current State

<!-- image -->

Sample Q-Factors Simulation Control 1 Control 2 Control 3

1)-Stages Base Heuristic Minimization

Figure 1.6.17 Illustration of the problem solved by a classical form of MPC at state x k . We minimize the cost function over the next /lscript stages while imposing the requirement that x k + /lscript = 0. We then apply the first control of the optimizing sequence. In the context of rollout, the minimization over u k is the one-step lookahead, while the minimization over u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k + /lscript -1 that drives x k + /lscript to 0 is the base heuristic.

and the terminal state constraint

<!-- formula-not-decoded -->

Here /lscript is an integer with /lscript &gt; 1, which is chosen in some largely empirical way.

- (b) If ¶ ˜ u k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u k + /lscript -1 ♦ is the optimal control sequence of this problem, we apply ˜ u k and we discard the other controls ˜ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u k + /lscript -1 .
- (c) At the next stage, we repeat this process, once the next state x k +1 is revealed.

To make the connection of the preceding MPC algorithm with rollout, we note that the one-step lookahead function ˜ J implicitly used by MPC [cf. Eq. (1.85)] is the cost function of a certain stable base policy . This is the policy that drives to 0 the state after /lscript -1 stages ( not /lscript stages ) and keeps the state at 0 thereafter, while observing the state and control constraints, and minimizing the associated ( /lscript -1)-stages cost. This rollout view of MPC was first discussed in the author's paper [Ber05]. It is useful for making a connection with the approximate DP/RL, rollout, and its interpretation in terms of Newton's method. In particular, an important consequence is that the MPC policy is stable , since rollout with a stable base policy can be shown to yield a stable policy under very general conditions, as we have noted earlier for the special case of linear quadratic problems in Section 1.5; cf. Fig. 1.5.10.

## Terminal Cost Approximation - Stability Issues

In a common variant of MPC, the requirement of driving the system state to 0 in /lscript steps in the /lscript -stage MPC problem (1.85), is replaced by a terminal cost G ( x k + /lscript ), which is positive everywhere except at 0. Thus at state x k , we solve the problem

<!-- formula-not-decoded -->

instead of problem (1.85) where we require that x k + /lscript = 0. This variant can be viewed as rollout with one-step lookahead, and a base policy, which at state x k +1 applies the first control ˜ u k +1 of the sequence ¶ ˜ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u k + /lscript -1 ♦ that minimizes

<!-- formula-not-decoded -->

On the other hand, this MPC variant can also be viewed as approximation in value space with /lscript -step lookahead minimization and terminal cost approximation given by G . It can be interpreted in terms of a Newton step, as illustrated in Fig. 1.5.7 for the case of one-step lookahead, and in Fig. 1.5.8 for the case of multistep lookahead.

An important question is how to choose the terminal cost approximation so that the resulting MPC controller is stable. Our discussion of Section 1.5 on the region of stability of approximation in value space schemes applies here. In particular, under the nonnegative cost assumption of this section, the MPC controller can be proved to be stable if a single value iteration (VI) starting from G produces a function that takes uniformly smaller values than G :

<!-- formula-not-decoded -->

This is also known as the Lyapunov condition in MPC. Figure 1.6.18 provides a graphical illustration, showing how this condition guarantees that successive iterates of value iteration, as implemented through multistep lookahead, lie within the region of stability, so that the policy produced by MPC is stable.

We expect that as the length /lscript of the lookahead minimization is increased, the stability properties of the MPC controller are improved. In particular, given G ≥ 0 , the resulting MPC controller is likely to be stable for /lscript su ffi ciently large , since the VI algorithm ordinarily converges to J * , which lies within the region of stability. Results of this type are known within the MPC framework under various conditions (see the papers by Mayne at al. [MRR00], Magni et al. [MDM01], the MPC book [RMD17],

Value Iterations

Bellman Operator

-

TJ.

Slope = 1

Optimal cost

J* = TJ*

Cost of

MPC Policy й

MPC Policy й

l = 3

Figure 1.6.18 Illustration of the Bellman operator, defined by

<!-- image -->

<!-- formula-not-decoded -->

The condition in (1.90) can be written compactly as ( TG )( x ) ≤ G ( x ) for all x . When satisfied by the terminal cost function G , it guarantees stability of the MPC policy ˜ θ with /lscript -step lookahead minimization. In this figure, /lscript = 3.

and the author's book [Ber20a], Section 3.1.2). Our discussion of stability in Section 1.5 is also relevant within this context; cf. Fig. 1.5.8.

In another variant of MPC, in addition to the terminal cost function approximation G , we use truncated rollout, which involves running some stable base policy θ for a number of steps m ; see Fig. 1.6.19. This is quite similar to standard truncated rollout, except that the computational solution of the lookahead minimization problem (1.89) may become complicated when the control space is infinite. As discussed in Section 1.5, increasing the length of the truncated rollout enlarges the region of stability of the MPC controller . The reason is that by increasing this length, we push the start of the Newton step towards of the cost function J θ of the stable policy, which lies within the region of stability. The base policy may also be used to address state constraints; see the papers by Rosolia and Borelli [RoB17], [RoB19], Li et al. [LJM21], and the discussions in the author's RL books [Ber20a], [Ber22a].

G

Base Policy

TuJ

TJ

Slope = 1

Defined by

Cost-to-go approximation Expected value approximation

Optimal cost

J* = TJ*

Stability Region l-Step

Lookahead

Minimization

Te

Cost of

MPC Policy й

m-Step Truncated

Rollout with

Stable Policy H

<!-- image -->

) Yields Truncated Rollout Policy ˜

Terminal Cost Approximation

Figure 1.6.19 An MPC scheme with /lscript -step lookahead minimization, m -step truncated rollout with a stable base policy θ , and a terminal cost function approximation G , together with its interpretation as a Newton step. In this figure, /lscript = 2 and m = 4. The truncated rollout with base policy θ consists of m value iterations, starting with the function G , and using the Bellman operator corresponding to θ , which is given by

<!-- formula-not-decoded -->

Thus, truncated rollout yields the function T θ m G . Then /lscript -1 value iterations are applied to this function through the ( /lscript -1)-step minimization, yielding the function

<!-- formula-not-decoded -->

Finally, the Newton step is applied to this function to yield the cost function of the MPC policy ˜ θ . As m increases, the starting point for the Newton step moves closer to J θ , which lies within the region of stability.

Finally, let us note that when faced with changing problem parameters, it is natural to consider on-line replanning as per our earlier adaptive control discussion. In this context, once new estimates of system and/or cost function parameters become available, MPC can readily adapt by introducing the new parameter estimates into the /lscript -stage optimization problem in (a) above. This is an important and often decisive advantage of

MPC over approximation in policy space for problems with changing environments.

## State Constraints and Invariant Sets

Our discussion so far has skirted a major issue in MPC, which is that there may be additional state constraints of the form x k ∈ X , for all k , where X is some subset of the true state space. Indeed much of the original work on MPC was motivated by control problems with state constraints, imposed by the physics of the problem, which could not be handled e ff ectively with the nice unconstrained framework of the linear quadratic problem that we discussed in Section 1.5.

To deal with additional state constraints of the form x k ∈ X , where X is some subset of the state space, the MPC problem to be solved at the k th stage [cf. Eq. (1.89)] must be modified. Assuming that the current state x k belongs to the constraint set X , the MPC problem should take the form

<!-- formula-not-decoded -->

subject to the control constraints

<!-- formula-not-decoded -->

and the state constraints

<!-- formula-not-decoded -->

The control ˜ u k thus obtained will generate a state

<!-- formula-not-decoded -->

that will belong to X , and similarly the entire state trajectory thus generated will satisfy the state constraint x t ∈ X for all t , assuming that the initial state does.

However, there is an important di ffi culty with the preceding MPC scheme, namely there is no guarantee that the problem (1.91)-(1.93) has a feasible solution for all initial states x k ∈ X . Here is a simple example.

## Example 1.6.8 (State Constraints in MPC)

Consider the scalar system with control constraint

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∈

Bellman Operator Value Iterations Largest Invariant Set

β

Figure 1.6.20 Illustration of invariance of a state constraint set X . Here the sets of the form X = { x k ♣ ♣ x k ♣ ≤ β } are invariant for β ≤ 1. For β = 1, we obtain the largest invariant set (the one that contains all other invariant sets). The figure shows some state trajectories produced by MPC. Note that starting with an initial condition x 0 with ♣ x 0 ♣ &gt; 1 (or ♣ x 0 ♣ &lt; 1) the closed-loop system obtained by MPC is unstable (or stable, respectively); cf. the red and green trajectories shown.

<!-- image -->

and state constraints of the form x k ∈ X , for all k , where

<!-- formula-not-decoded -->

Then if β &gt; 1, the state constraint cannot be satisfied for all initial states x 0 ∈ X . In particular, if we take x 0 = β , then 2 x 0 &gt; 2 and x 1 = 2 x 0 + u 0 will satisfy x 1 &gt; x 0 = β for any value of u 0 with ♣ u 0 ♣ ≤ 1. Similarly the entire sequence of states ¶ x k ♦ generated by any set of feasible controls will satisfy

<!-- formula-not-decoded -->

The state constraint can be satisfied only for initial states x 0 in the set ˆ X given by

<!-- formula-not-decoded -->

see Fig. 1.6.20, which also illustrates the trajectories generated by the MPC scheme of Eq. (1.89), which does not involve state constraints.

The preceding example illustrates a fundamental point in state-constrained MPC: the state constraint set X must be invariant in the sense that starting from any one of its points x k there must exist a control u k ∈ U ( x k ) for which the next state x k +1 = f ( x k ↪ u k ) must belong to X . Mathematically, X is invariant if for every x ∈ X , there exists u ∈ U ( x ) such that f ( x↪ u ) ∈ Xglyph[triangleright]

In particular, it can be seen that the set X of Eq. (1.94) is invariant if and only if β ≤ 1.

Given an MPC calculation of the form (1.91)-(1.93), we must make sure that the set X is invariant, or else it should be replaced by an invariant subset ˆ X ⊂ X . Then the MPC calculation (1.91)-(1.93) will be feasible provided the initial state x 0 belongs to ˆ X .

This brings up the question of how we compute an invariant subset of a given constraint set. This is typically an o ff -line calculation that cannot be performed during on-line play. It turns out that given X there exists a largest possible invariant subset of X , which can be computed in the limit with an algorithm that resembles value iteration. In particular, starting with X 0 = X , we obtain a nested sequence of subsets through the recursion

<!-- formula-not-decoded -->

Clearly, we have X k +1 ⊂ X k for all k , and under mild conditions it can be shown that the intersection set ˆ X = ∩ ∞ k =0 X k ↪ is the largest invariant subset of X ; see the author's PhD thesis [Ber71] and subsequent paper [Ber72a], which introduced the concept of invariance and its use in satisfying state constraints in control over a finite and an infinite horizon.

To illustrate, in the preceding Example 1.6.8, the sequence of value iterates (1.95) starting with the set X 0 = ¶ x ♣ ♣ x ♣ ≤ β ♦ , where β &gt; 1, is given by

<!-- formula-not-decoded -->

It can be seen that we have β k +1 &lt; β k for all k and β k ↓ 1, so the intersection ˆ X = ∩ ∞ k =0 X k yields the largest invariant set ˆ X = { x k ♣ ♣ x k ♣ ≤ 1 } glyph[triangleright]

## Suboptimal Invariant Subsets

Since computing the largest invariant subset of a constraint set X is often computationally intractable, one may consider using smaller invariant subsets of X . A relatively simple possibility is to compute an invariant subset ˆ X that corresponds to some nominal policy ˆ θ [i.e., starting from any point x ∈ ˆ X , the state f ( x↪ ˆ θ ( x ) ) belongs to ˆ X ]. Such an invariant subset may be obtained or approximated with some form of simulation using the policy ˆ θ . Moreover, ˆ θ can also be used for truncated rollout and also provide a terminal cost function approximation. An alternative for the case of a linear system driven by ellipsoid-bounded disturbances, is to construct an ellipsoidal invariant subset; the author's PhD thesis [Ber71] provided an algorithm for doing so.

For more broadly applicable possibilities, we refer to the MPC literature; see e.g., the book by Rawlings, Mayne, and Diehl [RMD17] (Chapter

The term used in [Ber71] and [Ber72a] is reachability of a target tube ¶ X↪X↪glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ , which is synonymous to invariance of X .

3), and the surveys by Mayne [May14], and Houska, Muller, and Villanueva [HMV24], which give additional references. An important point is that the computation of an invariant subset of the given constraint set X must be done o ff -line, thus becoming part of the o ff -line training phase, along with the terminal cost function G .

To deal with state constraints in the context of a partially unknown or changing system model, combinations of MPC with robust control ideas have been suggested [see [RMD17] (Chapter 3), where robust tube-based MPC ideas are discussed]. An alternative is to replace the state constraints with penalty or barrier functions as part of the cost per stage. This approach has received attention, using what is known as control barrier functions and control Lyapunov functions . We refer to the several survey papers in the literature, and to the book by Xiao, Cassandras, and Belta [XCB23], for accounts of this research direction.

## Stochastic MPC by Certainty Equivalence

We note that while we have focused on deterministic problems in this section, there are variants of MPC that include the treatment of uncertainty. The books and papers cited earlier contain several ideas along these lines; see e.g. the books by Kouvaritakis and Cannon [KoC16], Rawlings, Mayne, and Diehl [RMD17], and the survey by Mesbah [Mes16].

In this connection, it is also worth mentioning the certainty equivalence approach that we discussed briefly earlier. In particular, upon reaching state x k we may perform the MPC calculations after replacing the uncertain quantities w k +1 ↪ w k +2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] with deterministic quantities w k +1 ↪ w k +2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] , while allowing for the stochastic character of the disturbance w k of just the current stage k . Note that only the first step of this MPC calculation is stochastic. Thus the calculation needed per stage is not much more di ffi cult than the one for deterministic problems, while still implementing a Newton step for solving the associated Bellman equation; see our earlier discussion, and also Section 2.5.3 of the RL book [Ber19a], Section 3.2 of the book [Ber22a], and the MPC overview paper [Ber24].

## Data-Driven MPC

One of the principal limitations of the mainstream methodology of MPC is the need for a mathematical model. In many cases, such a model may not be available or may be di ffi cult to derive. To deal with this situation, data-driven versions of MPC have been suggested, where sampled triplets, consisting of state, control, and next state, are used for prediction of system trajectories.

In this approach, the system is implicitly represented by a dataset of observed trajectories, rather than an explicit mathematical model, an approach pioneered by J. C. Willems; see Markovsky and Dorfler [MaD21] for a recent review. The data consists of many sample triplets ( x k ↪ u k ↪ x k +1 ),

and the idea is to use this data to approximate the system dynamics and perform MPC. We can do this by fitting a parametric system equation model to the data, and predicting future system trajectories using this model. Among other approaches, we may use a neural network (possibly a transformer) or other type of system identification method for this purpose. Nonparametric models are also possible, based on statistical models, such as Bayesian process regression.

Once the data-approximated model has been constructed o ff -line, it can be used on-line to predict and optimize the future system trajectory over a finite prediction horizon, and to apply the first control in the optimal sequence at the current state. Naturally, the quality of the model has a major impact on the performance of the corresponding MPC policy. Some challenges here are the choice of state, and also the fact that the learned model may be inaccurate outside the training set. Moreover, major di ffi -culties arise in an adaptive control environment, where the model must be modified on-line by using data. For some representative relevant papers, see [CLD19], [GrZ19], [HWM20], [KRW21], [CLR21], [BGH22], [CWA22], [GrZ22], [MHD23], and [BeA24].

## MPC with Imperfect State Information

The MPC methodology, in the form we have described it here, requires full state feedback. When the exact state is not available, an estimate may be used instead. In classical control methodology, it is common to use a Kalman filtering algorithm or an observer to reconstruct the state. This is possible when the state is continuous and a system model is known. Also, in some noncontinuous state problems, it may be possible to use some inference method to obtain a state estimate, and use it in the MPC algorithm as if it were the exact state. Otherwise, it is necessary to view the problem as one of imperfect state information, introduce belief states in place of the unobservable states, and adopt the corresponding POMDP-like methodology. This is, however, much more demanding computationally, both for training a terminal cost function approximation and for on-line computation of the MPC controls.

The alternative for problems with imperfect state information is a non-MPC/approximation in policy space approach, which may also have some serious drawbacks:

- (a) The training of suitable policies can be complicated and unreliable due to local minima and other di ffi culties (see Chapter 3).
- (b) In an adaptive control/changing model setting, it may be necessary to retrain policies on-line, which can be very sample-ine ffi cient or practically impossible.
- (c) Restricting the structure of the policy to be a function of just part of the available information (e.g., the current system output) places a

limit on performance: we are not using the full information available.

- (d) Fundamentally, MPC/approximation in value space has a generic advantage over approximation in policy space: the MPC policy is the result of a Newton step for solving Bellman's equation. Thus the error from optimality J ˜ θ -J * of the MPC policy ˜ θ is governed by a superlinear relation, as we have discussed, and is negligible if the terminal cost approximation error ˜ J -J * is small.

## Model Mismatch and Disturbance Estimation in MPC

In practical implementations of MPC, an important source of performance degradation is model mismatch. Even when a reasonably accurate model is available to start with, unmodeled dynamics, parameter drift, or varying workloads, may cause the true evolution of the system to di ff er from the predictions used by the controller. If such mismatch is ignored, the closed-loop system may exhibit long-term state deviations from their desired targets or degraded constraint handling. For this reason, many practical MPC schemes for control system design incorporate some mechanism for estimating and compensating for the dominant model errors and other disturbances during real-time operation.

A viewpoint, emphasized in chemical process control (see the classical textbook by Stephanopoulos [Ste84] and its updated sequel [Ste25]), is that in many industrial settings, modeling errors can be significant, but can e ff ectively be viewed as a constant (or slowly varying) disturbance. If the controller can infer this disturbance from available measurements, it can suitably compensate for it; this is an idea that stems from the so called internal model principle , articulated by Francis and Wonham [FrW76]. The simplest instance of this is the integral term in a PID control scheme (cf. Section 1.6.8), which acts as a dynamic estimator of the unknown average disturbance: by accumulating the output error, it gradually identifies the control action needed to make the measured output coincide with the desired one.

A similar strategy is adopted in MPC formulations in a more general and systematic way. Rather than relying solely on integral action, one may introduce a small number of disturbance variables into the system model and allow an observer to update their values based on the mismatch between predicted and measured state evolution. Within our DP framework, this can be implemented through state augmentation (cf. Section 1.6.5). For a simple example, let the system x k +1 = f ( x k ↪ u k ) be replaced by

<!-- formula-not-decoded -->

where w k is a disturbance that represents a constant (or slowly varying) modeling error, so that

<!-- formula-not-decoded -->

We may then introduce w k as an additional state variable and an augmented system with state ˜ x k = ( x k ↪ w k ). If the state x k is observed perfectly, the same is true for the augmented state ˜ x k , since w k can be computed exactly as the di ff erence x k -f ( x k -1 ↪ u k -1 ).

A popular generalization of this approach, is to model the e ff ect of the disturbance through a system equation of the form

<!-- formula-not-decoded -->

where E is a known matrix, which models the e ff ects of the components of w k on di ff erent state variables/components of x k . When the state is not measured exactly, the MPC controller is naturally combined with a Kalman filter or related estimator. This estimator updates both the state and disturbance estimates at each stage, essentially producing a continually refined model of the system. The MPC controller then applies its optimization step as if the estimates of state and disturbance were exact.

For many systems of interest in chemical process control, the combination of disturbance modeling and state estimation, outlined above, captures the main benefits of more elaborate adaptive or robust MPC techniques; see the book by Stephanopoulos [Ste25] noted earlier. For larger and more rapidly varying uncertainties, more sophisticated identification or robust formulations may be necessary; we refer to the MPC literature for further discussion, particularly the textbooks by Rawlings, Mayne, and Diehl [RMD17], and by Borrelli, Bemporad, and Morari [BBM17].

## 1.7 REINFORCEMENTLEARNINGANDDECISION/CONTROL

The current state of RL has greatly benefited from the cross-fertilization of ideas from decision and control, and from artificial intelligence; see Fig. 1.7.1. The strong connections between these two fields are now widely recognized. Still, however, there are cultural di ff erences, including the traditional reliance on mathematical analysis for the decision and control field, and the emphasis on challenging problem implementations in the artificial intelligence (AI) field. Moreover, substantial di ff erences in language and emphasis remain between RL-based discussions (where AI-related terminology is used) and DP-based discussions (where optimal control-related terminology is used).

## 1.7.1 Di ff erences in Terminology

The terminology used in this book is standard in DP and optimal control, and in an e ff ort to forestall confusion of readers that are accustomed to either the AI or the decision and control terminology, we provide a list of terms commonly used in AI/RL, and their optimal control counterparts.

- (a) Environment = System.
- (b) Agent = Decision maker or controller.

Decision/

Control/DP

Principle of

Optimality

Markov Decision

Problems

POMDP

Policy Iteration

Value Iteration

Complementary

Late 80s-Early 90s

AIRL

Learning through

Data/Experience

Figure 1.7.1 A schematic illustration of the synergy of ideas between decision and control on one hand, and artificial intelligence on the other.

<!-- image -->

- (c) Action = Decision or control.
- (d) Reward of a stage = (Opposite of) Cost of a stage.
- (e) State value = (Opposite of) Cost starting from a state.
- (f) Value (or reward) function = (Opposite of) Cost function.
- (g) Maximizing the value function = Minimizing the cost function.
- (h) Action (or state-action) value = Q-factor (or Q-value) of a statecontrol pair. (Q-value is also used often in RL.)
- (i) Planning = Solving a DP problem with a known mathematical model. (Often related to MPC and approximation in value space.)
- (j) Learning = Solving a DP problem without using a known mathematical model. (This is the principal meaning of the term 'learning' in AI/RL. Other meanings are also common.)
- (k) Self-learning (or self-play in the context of games) = Solving a DP problem using some form of policy iteration.
- (l) Deep reinforcement learning = Approximate DP using value and/or policy approximation with deep neural networks.
- (m) Prediction = Policy evaluation.
- (n) Generalized policy iteration = Optimistic policy iteration.
- (o) State abstraction = State aggregation.
- (p) Temporal abstraction = Time aggregation.
- (q) Learning a model = System identification.

- (r) Episodic task or episode = Finite-step system trajectory.
- (s) Continuing task = Infinite-step system trajectory.
- (t) Experience replay = Reuse of samples in a simulation process.
- (u) Bellman operator = DP mapping or operator.
- (v) Backup = Applying the DP operator at some state.
- (w) Sweep = Applying the DP operator at all states.
- (x) Greedy policy with respect to a cost function J = Minimizing policy in the DP expression defined by J .
- (y) Afterstate = Post-decision state.
- (z) Ground truth = Empirical evidence or information provided by direct observation.

Some of the preceding terms will be introduced in future chapters; see also the RL textbook [Ber19a]. The reader may then wish to return to this section as an aid in connecting with the relevant RL literature.

## 1.7.2 Di ff erences in Notation

Unfortunately, the confusion caused by di ff ering terminology has been further compounded by the use of inconsistent notations across various sources. This book adheres to the 'standard' notation that emerged during the Bellman/Pontryagin optimal control era; see e.g., the books by Athans and Falb [AtF66], Bellman [Bel67], and Bryson and Ho [BrH75]. This notation is consistent with the author's other DP books and is the most appropriate for a unified treatment of the subject, which simultaneously addresses discrete and continuous spaces problems.

A summary of the most prominently used symbols in our notational system is as follows:

- (a) x : state (also i for finite-state systems).
- (b) u : control.
- (c) w : stochastic disturbance.
- (d) J : cost function.
- (e) f : system function. For deterministic systems,

<!-- formula-not-decoded -->

and for stochastic systems,

<!-- formula-not-decoded -->

Also f k in place of f for time-varying systems.

- (f) g : cost per stage [ g ( x↪ u ) for deterministic systems, and g ( x↪ u↪ w ) for stochastic systems; also g k in place of g for time-varying systems].
- (g) p xy ( u ): transition probability from state x to state y under control u in finite-state systems [also p ij ( u )].
- (h) α : discount factor in discounted problems.

The x -u -J notation is standard in deterministic optimal control textbooks (e.g., the classical books [AtF66] and [BrH75], noted earlier, as well as the more recent books by Stengel [Ste94], Kirk [Kir04], and Liberzon [Lib11]). The symbols f (system function) and g (cost per stage) are also widely used in both early and later optimal control literature (unfortunately the more natural symbol ' c ' has not been used much in place of ' g ' for the cost per stage).

The notations i (state) and p ij ( u ) (transition probability) are common in the discrete-state Markov decision process (MDP) and operations research literature. Sometimes the alternative notation p ( j ♣ i↪ u ) is used for the transition probabilities.

In the artificial intelligence literature, the focus is primarily on finitestate MDPs, particularly discounted and stochastic shortest path infinite horizon problems. The most commonly used notation is s for state, a for action, r ( s↪ a↪ s ′ ) for reward per stage, p ( s ′ ♣ s↪ a ) or p ( s↪ a↪ s ′ ) for transition probability from s to s ′ under action a , and γ for discount factor. While this notation is well-suited to finite-state problems, it is not ideal for continuous spaces models. The reason is that it requires the use of transition probability distributions defined over continuous spaces, and leads to more complex and less intuitive mathematics. Moreover, for deterministic problems, which lack a probabilistic component, the transition probability notation becomes cumbersome and unnecessary.

## 1.7.3 Relations Between DP and RL

When comparing the RL and DP methodologies, it is important to recognize that they are fundamentally connected by their shared focus on sequential decision making. Thus, any problem that can be addressed by DP can, in principle, also be addressed by RL, and vice versa.

One may argue that the RL algorithmic methodology is broader than that of DP. It includes the use of gradient descent and random search algorithms, simulation-based methods, statistical methods of sampling and performance evaluation, and neural network design and training ideas. However, methods of this type have also been considered in DP-related research and applications for many years, albeit less intensively.

In the artificial intelligence view of RL, a machine learns through trial and error by interacting with an environment. In practical terms,

Acommon description is that 'the machine learns sequentially how to make

this is more or less the same as what DP aims to do, but in RL there is often an emphasis on the presence of uncertainty and exploration of the environment. In the decision, control, and optimization community, there is a lot of interest in using RL methods to address intractable problems, including deterministic discrete/integer optimization, which need not involve data collection, interaction with the environment, uncertainty, and learning (adaptive control is the only decision and control problem type, where uncertainty and exploration arise in a significant way).

In terms of applications, DP was originally developed in the 1950s and 1960s as part of the then emerging methodologies of operations research and optimal control. These methodologies are now mature and provide important tools and perspectives, as well as a rich variety of applications, such as robotics, autonomous transportation, and aerospace, which can benefit from the use of RL. Moreover, DP has been used in a broad range of applications in industrial engineering, operations research, economics, and finance, so these applications can also benefit from the use of RL methods and perspectives.

At the same time, RL and machine learning have ushered opportunities for the application of DP techniques in new domains, such as machine translation, image recognition, knowledge representation, database organization, large language models, and automated planning, where they can have a significant practical impact. We may also add that RL has brought into the field of sequential decision making a fresh and ambitious spirit that has made possible the solution of problems thought to be well outside the capabilities of DP. Indeed, before the connections between RL and DP were recognized, large dimensional problems, like those involving a Euclidean state space of even moderate dimension, or POMDP problems, were considered totally intractable with the DP methodology.

## 1.7.4 Synergy Between Large Language Models and DP/RL

Can RL and large language models (LLMs) work synergistically? This is an important question, as these two AI paradigms operate with distinct methodologies, objectives, and capabilities. While RL focuses on sequential decision-making, LLMs specialize in natural language understanding and generation, including computer code.

In particular, RL is designed to optimize policies for sequential control tasks, excelling in applications such as robotics and resource allocation, where adaptive decision-making is essential. In contrast, LLMs process and generate human-like text, enabling them to perform tasks such as translation, summarization, sentiment analysis, and code generation. Moreover,

decisions that maximize a reward signal, based on the feedback received from the environment.'

by leveraging vast amounts of pre-trained knowledge, LLMs can generalize across diverse contexts.

Despite their di ff erences, the capabilities of RL and LLMs are complementary , making them powerful tools when used in combination. Let us now summarize ways in which the synergy between RL and LLMs can manifest itself in practice.

The advent of pre-trained transformers, such as ChatGPT, has revolutionized natural language processing. These transformers can undergo further refinement through o ff -line training to specialize in specific tasks or mitigate undesirable biases. Notably, RL methodologies, particularly policy-space approximation techniques, have played a crucial role in this refinement process, as discussed in Section 3.5 of Chapter 3. Thus, RL has been instrumental in enhancing the capabilities of LLMs through o ff -line optimization techniques.

Conversely, LLMs serve as catalysts for RL by injecting domain knowledge, improving interpretability, and enabling more human-aligned training. By leveraging natural language input, LLMs facilitate the design of RL policies that are more transparent and adaptable. Additionally, LLMs support RL applications by assisting with mathematical formulation, algorithm selection, and code generation. This interplay between the two fields continues to evolve, driving new innovations in AI.

In conclusion, the interaction of RL and LLMs is not merely additive, it is multiplicative: RL equips LLMs with the ability to learn from interaction and feedback, while LLMs equip RL with contextual awareness, explainability, accessibility, and code generation capability. Together, they pave the way for enhanced applications, which align with human intent, and can also communicate and explain their reasoning.

## 1.7.5 Machine Learning and Optimization

Machine learning and optimization are closely intertwined fields, sharing similar mathematical models and computational algorithms. However, they di ff er in their cultures and application contexts, so it is worth reflecting on their similarities and di ff erences.

Machine learning can be broadly categorized into three main types of methods, all of which involve the collection and use of data in some form:

- (a) Supervised learning : Here a dataset of many input-output pairs (also called labeled data) is collected. An optimization algorithm is used to create a parametrized function that fits well the data, as well as make accurate predictions on new, unseen data. Supervised learning problems are typically formulated as optimization problems, examples

Both fields are also closely connected to the field of statistical analysis. However, in this section, we will not focus on this connection, as it is less relevant to the content of this book.

of which we will see in Chapter 3. A common algorithmic approach is to use a gradient-type algorithm to minimize a loss function that measures the di ff erence between the actual outputs of the dataset and the predicted outputs of the parametrized model.

- (b) Unsupervised learning : Here the dataset is 'unlabeled' in the sense that the data are not separated into input and matching output pairs. Unsupervised learning algorithms aim to identify patterns or structures within the data, which is useful for tasks like clustering, dimensionality reduction, and density estimation. The objective is to extract meaningful insights from the data. Some unsupervised learning methods can be related to DP, but the connection is not strong. Generally speaking, unsupervised learning does not seem to align well with the types of sequential decision making applications of this book.
- (c) Reinforcement learning : RL di ff ers in an important way from supervised and unsupervised learning. It does not use a dataset as a starting point . Instead, it generates data on-line or o ff -line as dictated by the needs of the optimization algorithm it uses, be it multistep lookahead minimization, approximate policy iteration and rollout, or approximation in policy space. A further complication in RL is that the generated data depends on the policy that is used to control the system. Ideally, the data should be generated using an optimal or near-optimal policy, but such a policy is unlikely to be available. We are thus forced to collect data using a sequence of (hopefully) improving policies, which is the essence of the approximate policy iteration algorithm of DP. This is a primary reason why this algorithm and its variations will be a focal point for our discussions in this book.

Another type of machine learning approach, which relates to DP/RL methods, is semi-supervised learning . It involves training a model using a dataset containing both labeled and unlabeled data. Here, some initial labeled data are sequentially augmented with unlabeled data, with the aim of constructing an 'informative' data set that enhances machine learning tasks such as classification. This approach lies between supervised learning (which requires all data to be labeled) and unsupervised learning (which works with exclusively unlabeled data). Semi-supervised learning is related to the field of active learning , where DP-like methods are used to augment sequentially the labeled set; see e.g., the monograph by Zhu and Goldberg [ZhG22], the survey by Van Engelen and Hoos [VaH20], and the illustrative application papers by Marchesoni-Acland et al. [MMK23], and Bhusal, Miller, and Merkurjev [BMM24].

Optimization problems and algorithms on the other hand may or may not involve the collection and use of data. They involve data only in the

A variant of RL called o ffl ine RL or batch RL , starts from a historical dataset, and does not explore the environment to collect new data.

context of special applications, most of which are related to machine learning. In theoretical terms, optimization problems are categorized in terms of their mathematical structure, which is the primary determinant of the suitability of particular types of methods for their solution. In particular, it is common to distinguish between static optimization problems and dynamic optimization problems . The latter problems involve sequential decision making, with feedback between decisions, while the former problems involve a single decision.

Stochastic problems with perfect or imperfect state observations are dynamic (unless they involve open-loop decision making without the use of any feedback), and they require the use of DP for their optimal solution. Deterministic problems can be formulated as static, but they can also be formulated as dynamic for reasons of algorithmic expediency. In this case, the decision making process is (sometimes artificially) broken down into stages, as is often done in this book for discrete optimization and other contexts.

Another important categorization of optimization problems is based on whether their search space is discrete or is continuous . Discrete problems include deterministic problems such as integer and combinatorial optimization problems, and can be addressed by formal methods of integer programming as well as by DP. These problems tend to be challenging, so they are often addressed (suboptimally) with the use of heuristics.

Continuous problems are usually addressed with very di ff erent methods, which are based on calculus and convexity, such as Lagrange multiplier theory and duality, and the computational machinery of linear, nonlinear, and convex programming. Some discrete problems, particularly those that involve graphs (such as matching, transportation, and transshipment problems), can be addressed using continuous spaces network optimization methods that rely on linear programming and duality. Hybrid problems, which combine discrete and continuous variables, usually require discrete optimization techniques, but can also benefit from convex duality methods, which are fundamentally continuous.

The DP methodology, generally speaking, applies to just about any kind of optimization problem, deterministic or stochastic, static or dynamic, discrete or continuous, as long as it is formulated as a sequential decision problem , in the manner described in Sections 1.2-1.4. In terms of algorithmic structure, DP di ff ers significantly from other optimization techniques, particularly those based on calculus and convexity. Notably, DP can handle both discrete and continuous problems and is not concerned with local minima, focusing instead on finding global minima.

Notice a qualitative di ff erence between optimization and machine learning: the former is mostly organized around mathematical structures and the analysis of the foundational issues of the corresponding algorithms, while the latter is mostly organized around how data is collected, used, and analyzed, often with a strong emphasis on statistical issues . This is an im-

portant distinction, which a ff ects profoundly the perspectives of researchers in the two fields.

## 1.7.6 Mathematics in Machine Learning and Optimization

Let us now discuss some di ff erences between the research cultures of the optimization and machine learning fields, as they pertain to the use of mathematics. In optimization, the emphasis is often on general purpose methods that o ff er broad and mathematically rigorous performance guarantees, for a wide variety of problems. In particular, it is generally believed that a solid mathematical foundation for a given optimization methodology enhances its reliability and clarifies the boundaries of its applicability. Furthermore, it is recognized that formulating practical problems and matching them to the right algorithms is greatly enhanced by one's understanding of the mathematical structure of the underlying optimization methodology.

Machine learning research includes important lines of analysis with a strongly mathematical character, particularly relating to theoretical computer science, complexity theory, and statistical analysis. At the same time, in machine learning there are eminently useful algorithmic structures, such as neural networks, large language models, and image generative models, which are not well-understood mathematically and defy to a large extent mathematical analysis. This can add to a perception that focusing on rigorous mathematics, as opposed to practical implementation, may be a low payo ff investment in many practical machine learning contexts.

Moreover, the starting point in machine learning is often a specific dataset or a specialized type of training problem (e.g., language translation or image recognition). The priority is to find a method that works well for that specific dataset or problem, even if it is not generalizable to others. Thus specialized approximation architectures, implementation techniques, and heuristics, which perform well for the given problem and dataset type, may be perfectly acceptable in a machine learning context, even if they do not provide rigorous and generally applicable performance guarantees.

In conclusion, both optimization and machine learning involve mathematical models and rigorous analysis in important ways, and often overlap in the techniques and tools that they use, as well as in the practical applications that they address. However, depending on the problem at hand, there may be di ff erences in the emphasis and priority placed on mathe-

As an illustration, the paper by He et al., 'Deep Residual Learning for Image Recognition,' published in Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition, 2016, has been cited over 296,276 times as of November 2025, and contains only two equations. The famous neural network architecture paper by Vaswani et al., 'Attention is all you Need,' published in NIPS, 2017, which laid the foundation for GPT, has been cited over 205,945 times as of November 2025, and contains only six equations.

matical analysis, insight, and generality versus practical e ff ectiveness and problem-specific e ffi ciency. This can lead to some tension, as di ff erent fields may not fully appreciate each other's perspective.

## 1.8 NOTES, SOURCES, AND EXERCISES

We will now summarize this chapter and describe how it can be used flexibly as a foundation for a few di ff erent courses. We will also provide a selective overview of the DP and RL literature, and give a few exercises that have been used in ASU classes.

## Chapter Summary

In this chapter, we have aimed to provide an overview of the approximate DP/RL landscape, which can serve as the foundation for a deeper in-class development of other RL topics. In particular, we have described in varying levels of depth the following:

- (a) The algorithmic foundation of exact DP in all its major forms: deterministic and stochastic, discrete and continuous, finite and infinite horizon.
- (b) Approximation in value space with one-step and multistep lookahead, the workhorse of RL, which underlies its major success stories, including AlphaZero. We contrasted approximation in value space with approximation in policy space, and discussed how the two may be combined.
- (c) The important division between o ff -line training and on-line play in the context of approximation in value space. We highlighted how their synergy can be intuitively explained in terms of Newton's method.
- (d) The fundamental methods of policy iteration and rollout, the former being primarily an o ff -line method, and the latter being primarily a less ambitious on-line method. Both methods and their variants bear close relation to Newton's method and draw their e ff ectiveness from this relation.
- (e) Some major models with a broad range of applications, such as discrete optimization, POMDP, multiagent problems, adaptive control, and model predictive control. We delineated their principal characteristics and the major RL implementation issues within their contexts.
- (f) The use of function approximation, which has been a recurring theme in our presentation. We have touched upon some of the principal schemes for approximation, e.g., neural networks and feature-based architectures.

One of the principal aims of this chapter was to provide a foundational platform for a range of RL courses that explore at a deeper level various algorithmic methodologies, such as:

- (1) Rollout and policy iteration.
- (2) Neural networks and other approximation architectures for o ff -line training.
- (3) Aggregation, which can be used for cost function approximation in the context of approximation in value space.
- (4) A broader discussion of sequential decision making in contexts involving changing system parameters, sequential estimation, and simultaneous system identification and control.
- (5) Stochastic algorithms, such as temporal di ff erence methods and Qlearning, which can be used for o ff -line policy evaluation in the context of approximate policy iteration.
- (6) Sampling methods to collect data for o ff -line training in the context of cost and policy approximations.
- (7) Statistical estimates and e ffi ciency enhancements of various sampling methods used in simulation-based schemes. This includes confidence intervals and computational complexity estimates.
- (8) On-line methods for specially structured contexts, including problems of the multi-armed bandit type.
- (9) Simulation-based algorithms for approximation in policy space, including policy gradient and random search methods.
- (10) A deeper exploration of control system design methodologies such as model predictive control and adaptive control, and their applications in robotics and automated transportation.

In our course we have focused selectively on the methodologies (1)(4), with a limited coverage of (9) in Section 3.5. In a di ff erent course, other choices from the above list may be made, by building on the content of the present chapter.

## Notes and Sources for Individual Sections

In the literature review that follows, we will focus primarily on textbooks, research monographs, and broad surveys, which supplement our discussions, present related viewpoints, and collectively provide a guide to the literature. Inevitably, our selection reflects a certain cultural bias and an overemphasis on sources that are familiar to the author and aligned in style with this book (including the author's own works). We acknowledge in advance that this may lead to omissions of research references that fall outside

our own understanding and perspective on the field, and we apologize for any such exclusions.

Sections 1.1-1.4 : Our discussion of exact DP in this chapter has been brief since our focus in this book will be on approximate DP and RL. For a more comprehensive treatment of finite-horizon exact DP and its applications to both discrete and continuous space problems, the author's DP textbook [Ber17a] provides an extensive overview, using notation and style consistent with this book. The books by Puterman [Put94] (written from an operations research perspective) and by the author [Ber12] (written from a decision and control perspective) provide detailed (but substantially different) treatments of infinite horizon finite-state stochastic DP problems. The book [Ber12] also covers continuous/infinite state and control spaces problems, including the linear quadratic problems that we have discussed for one-dimensional problems in this chapter. Continuous spaces problems present special analytical and computational challenges, which are at the forefront of research of the RL methodology. The author's 1976 DP textbook [Ber76] was the first to develop discrete-time DP within a general framework that allows arbitrary state, control, and disturbance spaces.

Some of the more complex mathematical aspects of exact DP were addressed in the monograph by Bertsekas and Shreve [BeS78], particularly the probabilistic/measure-theoretic issues associated with stochastic optimal control, including partial state information problems. This monograph provides an extensive treatment of these issues. The followup work by Huizhen Yu and the author [YuB15] addresses the special measurability issues that relate to policy iteration, and provides further analysis relating to the convergence of value iteration. The second volume of the author's DP book [Ber12], Appendix A, includes an accessible summary introduction of the measure-theoretic framework of the book [BeS78]. In the RL literature, the mathematical di ffi culties around measurability are usually

The rigorous mathematical theory of stochastic optimal control, including the development of an appropriate Borel space measure-theoretic framework, originated in the 60s, with the work of Blackwell [Bla65], [Bla67]. It relies on the theory of analytic sets of descriptive set theory, introduced in 1917 by M. Suslin, a young Russian mathematician, and further developed by his mentor N. Luzin. It culminated in the Bertsekas and Shreve monograph [BeS78], which provides the now 'standard' framework, based on the formalism of Borel spaces, lower semianalytic functions, and universally measurable policies. This development involves daunting mathematical complications, which stem, among others, from the observation that when a Borel measurable function F ( x↪ u ), of the two variables x and u , is minimized with respect to u , the resulting function G ( x ) = min u F ( x↪ u ) need not be Borel measurable (it belongs to the broader class of lower semianalytic functions); this key fact was the starting point of Suslin's analysis. Moreover, even if the minimum is attained by several policies θ , i.e., G ( x ) = F ( x↪ θ ( x ) ) for all x , it is possible that none of these θ is Borel

neglected (as they are in this book), and this is fine because they do not play an important role in practical applications. Moreover, measurability issues do not arise for problems involving finite or countably infinite state and control spaces. We note, however, that there are quite a few published works in RL as well as exact DP, which purport to address measurability issues with a mathematical narrative that is either confusing or plain incorrect.

The third edition of the author's abstract DP monograph [Ber22b], expands on the original 2013 first edition, and aims at a unified development of the core theory and algorithms of total cost sequential decision problems. It addresses simultaneously stochastic, minimax, game, risksensitive, and other DP problems, through the use of abstract DP operators (or Bellman operators as we call them here). The idea is to gain insight through abstraction. In particular, the structure of a DP model is encoded in its abstract Bellman operator, which serves as the 'mathematical signature' of the model. Thus, characteristics of this operator (such as monotonicity and contraction) largely determine the analytical results and computational algorithms that can be applied to that model. Abstract DP ideas are also useful for visualizations and interpretations of RL methods using the Newton method formalism that we have discussed somewhat briefly in this book in the context of linear quadratic problems.

Approximation in value space, rollout, and policy iteration are the principal subjects of this book. These are very powerful and general techniques: they can be applied to deterministic and stochastic problems, finite and infinite horizon problems, discrete and continuous spaces problems, and mixtures thereof. Moreover, rollout is reliable, easy to implement, and can be used in conjunction with on-line replanning. It is also compatible measurable (however, there does exist a minimizing policy that belongs to the broader class of universally measurable policies). Thus, starting with a Borel measurability framework for cost functions and policies, we quickly get outside that framework when executing DP algorithms, such as value and policy iteration. The broader framework of universal measurability, introduced in [BeS78], is required to correct this deficiency, in the absence of additional (fairly strong) assumptions.

The name 'rollout' (also called 'policy rollout') was introduced by Tesauro and Galperin [TeG96] in the context of rolling the dice in the game of backgammon. In Tesauro's proposal, a given backgammon position is evaluated by 'rolling out' many games starting from that position to the end of the game. To quote from the paper [TeG96]: 'In backgammon parlance, the expected value of a position is known as the 'equity' of the position, and estimating the equity by Monte-Carlo sampling is known as performing a 'rollout.' This involves playing the position out to completion many times with di ff erent random dice sequences, using a fixed policy to make move decisions for both sides.'

with new and exciting technologies such as transformer networks and large language models (see Section 2.3.7).

As we have noted, rollout with a given base policy is simply the first iteration of the policy iteration algorithm starting from the base policy. Truncated rollout can be interpreted as an 'optimistic' form of a single policy iteration, whereby a policy is evaluated inexactly, by using a limited number of value iterations; see the books [Ber20a], [Ber22a].

Policy iteration, which can be seen as repeated rollout, is more ambitious and challenging than rollout. It requires o ff -line training, possibly in conjunction with the use of neural networks. Together with its neural network and distributed implementations, it will be discussed in more detail later. Note that rollout does not require any o ff -line training, once the base policy is available; this is its principal advantage over policy iteration.

Section 1.5: There is a vast literature on linear quadratic problems. The connection of policy iteration with Newton's method within this context was first derived by Kleinman [Kle68], as part of his doctoral research at MIT, under the supervision of M. Athans. Kleinman's work focused on continuous-time linear quadratic problems (see Hewer [Hew71] for the discrete-time case). For followup work, which relates to approximate policy iteration, see Feitzinger, Hylla, and Sachs [FHS09], and Hylla [Hyl11].

The general relation of approximation in value space with Newton's method, beyond policy iteration, and its connections with MPC and adaptive control was first presented in the author's book [Ber20a], the papers [Ber21b], [Ber22c], and in the book [Ber22a], which contains an extensive discussion. This relation provides the starting point for an in-depth understanding of the synergy between the o ff -line training and the on-line play components of the approximation in value space architecture, including the role of multistep lookahead in enhancing the starting point of the Newton

Truncated rollout was also proposed in the context of backgammon in the paper [TeG96]. To quote from this paper: 'Using large multi-layer networks to do full rollouts is not feasible for real-time move decisions, since the large networks are at least a factor of 100 slower than the linear evaluators described previously. We have therefore investigated an alternative Monte-Carlo algorithm, using so-called 'truncated rollouts.' In this technique trials are not played out to completion, but instead only a few steps in the simulation are taken, and the neural net's equity estimate of the final position reached is used instead of the actual outcome. The truncated rollout algorithm requires much less CPU time, due to two factors: First, there are potentially many fewer steps per trial. Second, there is much less variance per trial, since only a few random steps are taken and a real-valued estimate is recorded, rather than many random steps and an integer final outcome. These two factors combine to give at least an order of magnitude speed-up compared to full rollouts, while still giving a large error reduction relative to the base player.' Analysis and computational experience with truncated rollout since 1996 are consistent with the preceding assessment.

step. The monograph [Ber22a] (Appendix A) also provides analysis of variants of Newton's method for nondi ff erentiable fixed point problems, such as the ones arising in Bellman's equation (which involves a nondi ff erentiable right-hand side in finite-control space problems, among others).

Note that in approximation in value space, we are applying Newton's method to the solution of a system of equations (the Bellman equation). This context has no connection with the 'gradient descent' methods that are popular for the solution of special types of optimization problems in RL, arising for example in neural network training problems (see Chapter 3). In particular, there are no gradient descent methods that can be used for the solution of systems of equations such as the Bellman equation. There are, however, 'first order' deterministic algorithms such as the Gauss-Seidel and Jacobi methods (and stochastic asynchronous extensions) that can be applied to the solution of systems of equations with special structure, including Bellman equations. Such methods include various Q-learning algorithms, which are discussed in the neuro-dynamic programming book by Bertsekas and Tsitsiklis [BeT89], as well as the recent book by Meyn [Mey22]. While these methods can be useful, they are much slower than Newton's method and have limited utility in the context of on-line play.

Section 1.6: Many applications of DP are discussed in the 1st volume of the author's DP book [Ber17a]. This book also covers a broad variety of state augmentation and problem reformulation techniques, including the mathematics of how problems with imperfect state information can be transformed to perfect state information problems. In Section 1.6 we have aimed to provide an overview, with an emphasis on the use of approximations. In what follows we provide some related historical notes.

Multiagent problems : This subject has a long history (Marschak [Mar55], Radner [Rad62], Witsenhausen [Wit68], [Wit71a], [Wit71b]), and was researched extensively in the 70s; see the review paper by Ho [Ho80] and the references cited there. The names used at that time were team theory and decentralized control . For a sampling of subsequent works in team theory and multiagent optimization, we refer to the papers by Krainak, Speyer, and Marcus [KLM82a], [KLM82b], and de Waal and van Schuppen [WaS00]. For more recent works, see Nayyar, Mahajan, and Teneketzis [NMT13], Nayyar and Teneketzis [NaT19], Li et al. [LTZ19], Qu and Li [QuL19], Gupta [Gup20], the book by Zoppoli, Sanguineti, Gnecco, and Parisini [ZSG20], and the references quoted there. In addition to the aforementioned works, surveys of multiagent DP from an RL perspective were given by Busoniu, Babuska, and De Schutter [BBD08], [BBD10b].

The term 'multiagent' has been used with various meanings in the literature. Some authors emphasize scenarios where agents lack common information when making their decisions, leading to sequential decision problems with 'nonclassical information patterns.' These problems are particularly complex because they cannot be solved using exact DP techniques.