# 1.7: Reinforcement Learning and Decision/Control

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 131-140
**Topics:** RL terminology, RL notation, DP vs RL, LLM synergy, machine learning, optimization

---

theory literature, but the corresponding methodology is complex and beyond our scope in this book. However, assuming that we can make the estimation phase work somehow, we are free to revise the controller using the newly estimated parameters in a variety of ways, in an on-line replanning process.

Unfortunately, there is still another di ffi culty with this type of online replanning: it may be hard to recompute an optimal or near-optimal policy on-line, using a newly identified system model. In particular, it may be impossible to use time-consuming methods that involve for example the training of a neural network or discrete/integer control constraints. A simpler possibility is to use rollout, which we discuss next.

## Adaptive Control by Rollout and On-Line Replanning

We will now consider an approach for dealing with unknown or changing parameters, which is based on on-line replanning. We have discussed this approach in the context of rollout and multiagent rollout, where we stressed the importance of fast on-line policy improvement.

Let us assume that some problem parameters change and the current controller becomes aware of the change 'instantly' (i.e., very quickly before the next stage begins). The method by which the problem parameters are recalculated or become known is immaterial for the purposes of the following discussion. It may involve a limited form of parameter estimation, whereby the unknown parameters are 'tracked' by data collection over a few time stages, with due attention paid to issues of parameter identifiability; or it may involve new features of the control environment, such as a changing number of servers and/or tasks in a service system (think of new spiders and/or flies appearing or disappearing unexpectedly in the spiders-and-flies Example 1.6.5).

We thus assume away/ignore issues of parameter estimation, and focus on revising the controller by on-line replanning based on the newly obtained parameters. This revision may be based on any suboptimal method, but rollout with the current policy used as the base policy is particularly attractive. Here the advantage of rollout is that it is simple and reliable. In particular, it does not require a complicated training procedure to re-

Another possibility is to deal with this di ffi culty by precomputation. In particular, assume that the set of problem parameters may take a known finite set of values (for example each set of parameter values may correspond to a distinct maneuver of a vehicle, motion of a robotic arm, flying regime of an aircraft, etc). Then we may precompute a separate controller for each of these values. Once the control scheme detects a change in problem parameters, it switches to the corresponding predesigned current controller. This is sometimes called a multiple model control design or gain scheduling , and has a long history of success in various settings over the years; see e.g., Athans et al., [ACD77].

Lookahead

Minimization

Xk

Possible States

Xk+ 1

Rollout with

Base Policy

Changing System,

<!-- image -->

Changing System, Cost, and Constraint Parameters

Changing System, Cost, and Constraint Parameters

Multiagent Q-factor minimization

Possible States

Figure 1.6.12 Schematic illustration of adaptive control by rollout. One-step lookahead is followed by simulation with the base policy, which stays fixed. The system, cost, and constraint parameters are changing over time, and the most recent values are incorporated into the lookahead minimization and rollout operations. For the discussion in this section, we may assume that all the changing parameter information is provided by some computation and sensor 'cloud' that is beyond our control. The base policy may also be revised based on various criteria.

vise the current policy, based for example on the use of neural networks or other approximation architectures, so no new policy is explicitly computed in response to the parameter changes . Instead the current policy is used as the base policy for rollout, and the available controls at the current state are compared by a one-step or mutistep minimization, with cost function approximation provided by the base policy (cf. Fig. 1.6.12).

Note that over time the base policy may also be revised (on the basis of an unspecified rationale), in which case the rollout policy will be revised both in response to the changed current policy and in response to the changing parameters. This is necessary in particular when the constraints of the problem change.

The principal requirement for using rollout in an adaptive control context is that the rollout control computation should be fast enough to be performed between stages. Note, however, that accelerated/truncated versions of rollout, as well as parallel computation, can be used to meet

this time constraint.

The following example considers on-line replanning with the use of rollout in the context of the simple one-dimensional linear quadratic problem that we discussed earlier in this chapter. The purpose of the example is to illustrate analytically how rollout with a policy that is optimal for a nominal set of problem parameters works well when the parameters change from their nominal values. This property is not practically useful in linear quadratic problems because when the parameter change, it is possible to calculate the new optimal policy in closed form, but it is indicative of the performance robustness of rollout in other contexts. Generally, adaptive control by rollout and on-line replanning makes sense in situations where the calculation of the rollout controls for a given set of problem parameters is faster and/or more convenient than the calculation of the optimal controls for the same set of parameter values. These problems include cases involving nonlinear systems and/or di ffi cult (e.g., integer) constraints.

## Example 1.6.7 (On-Line Replanning for Linear Quadratic Problems Based on Rollout)

Consider the deterministic undiscounted infinite horizon linear quadratic problem. It involves the linear system

<!-- formula-not-decoded -->

and the quadratic cost function

<!-- formula-not-decoded -->

The optimal cost function is given by

<!-- formula-not-decoded -->

where K ∗ is the unique positive solution of the Riccati equation

<!-- formula-not-decoded -->

The optimal policy has the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

As an example, consider the optimal policy that corresponds to the nominal problem parameters b = 2 and r = 0 glyph[triangleright] 5: this is the policy (1.81)(1.82), with K obtained as the positive solution of the quadratic Riccati Eq. (1.80) for b = 2 and r = 0 glyph[triangleright] 5. In particular, we can verify that

<!-- formula-not-decoded -->

From Eq. (1.82) we then obtain

<!-- formula-not-decoded -->

Wewill now consider changes of the values of b and r while keeping L constant, and we will compare the quadratic cost coe ffi cient of the following three cost functions as b and r vary:

- (a) The optimal cost function K ∗ x 2 , where K ∗ is given by the positive solution of the Riccati Eq. (1.80).
- (b) The cost function K L x 2 that corresponds to the base policy

<!-- formula-not-decoded -->

where L is given by Eq. (1.83). From our earlier discussion, we have

<!-- formula-not-decoded -->

- (c) The cost function ˜ K L x 2 that corresponds to the rollout policy

<!-- formula-not-decoded -->

obtained by using the policy θ L as base policy. Using the formulas given earlier, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Figure 1.6.13 shows the coe ffi cients K ∗ , K L , and ˜ K L for a range of values of r and b . We have

<!-- formula-not-decoded -->

The di ff erence K L -K ∗ is indicative of the robustness of the policy θ L , i.e., the performance loss incurred by ignoring the values of b and r , and continuing to use the policy θ L , which is optimal for the nominal values b = 2 and r = 0 glyph[triangleright] 5, but suboptimal for other values of b and r . The di ff erence ˜ K L -K ∗ is indicative of the performance loss due to using on-line replanning by rollout rather than using optimal replanning. Finally, the di ff erence K L -˜ K L is indicative of the performance improvement due to on-line replanning using rollout rather than keeping the policy θ L unchanged.

Note that Fig. 1.6.13 illustrates the behavior of the error ratio

<!-- formula-not-decoded -->

where for a given initial state, ˜ J is the rollout performance, J ∗ is the optimal performance, and J is the base policy performance. This ratio approaches 0 as J -J ∗ becomes smaller because of the quadratic convergence rate of Newton's method that underlies the rollout algorithm.

Exactly Reoptimized Policy

Approximately Reoptimized Rollout Policy

Approximately Reoptimized Rollout Policy

Figure 1.6.13 Illustration of adaptive control by rollout under changing problem parameters. The quadratic cost coe ffi cients K ∗ (optimal, denoted by solid line), K L (base policy, denoted by circles), and ˜ K L (rollout policy, denoted by asterisks) for the two cases where r = 0 glyph[triangleright] 5 and b varies, and b = 2 and r varies. The value of L is fixed at the value that is optimal for b = 2 and r = 0 glyph[triangleright] 5 [cf. Eq. (1.83)].

<!-- image -->

The rollout policy performance is very close to the one of the exactly reoptimized policy, while the base policy yields much worse performance. This is a consequence of the quadratic convergence rate of Newton's method that underlies rollout:

<!-- formula-not-decoded -->

where for a given initial state, ˜ J is the rollout performance, J ∗ is the optimal performance, and J is the base policy performance.

## Adaptive Control as POMDP

The preceding adaptive control formulation strictly separates the dual objective of estimation and control: first parameter identification and then controller reoptimization (either exact or rollout-based). In an alternative adaptive control formulation, the parameter estimation and the application of control are done simultaneously, and indeed part of the control e ff ort may be directed towards improving the quality of future estimation. This alternative (and more principled) approach is based on a view of adaptive control as a partially observed Markovian decision problem (POMDP) with a special structure. We will see in Section 2.11 that this approach is well-suited for approximation in value space schemes, including forms of rollout.

To describe briefly the adaptive control reformulation as POMDP, we introduce a system whose state consists of two components:

- (a) A perfectly observed component x k that evolves over time according to a discrete-time equation.
- (b) A component θ which is unobserved but stays constant, and is estimated through the perfect observations of the component x k .

We view θ as a parameter in the system equation that governs the evolution of x k . Thus we have

<!-- formula-not-decoded -->

where u k is the control at time k , selected from a set U k ( x k ), and w k is a random disturbance with given probability distribution that depends on ( x k ↪ θ ↪ u k ). For convenience, we will assume that θ can take one of m known values θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m .

The a priori probability distribution of θ is given and is updated based on the observed values of the state components x k and the applied controls u k . In particular, the information vector

<!-- formula-not-decoded -->

is available at time k , and is used to compute the conditional probabilities

<!-- formula-not-decoded -->

These probabilities form a vector

<!-- formula-not-decoded -->

which together with the perfectly observed state x k , form the pair ( x k ↪ b k ), which is the belief state of the POMDP at time k . The overall control scheme takes the form illustrated in Fig. 1.6.14.

System State Data Control Parameter Estimation

k

) Belief Estimator

) Belief Estimator

Figure 1.6.14 Schematic illustration of simultaneous control and belief estimation for the unknown system parameter θ . The control applied is a function of the current belief state ( x k ↪ b k ), where b k is the conditional probability distribution of θ given the observations accumulated up to time k (the current and past states x k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x 0 , and the past controls u k -1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u 0 ).

<!-- image -->

As discussed in Section 1.6.6, an exact DP algorithm can be written for the equivalent POMDP, and this algorithm is suitable for the use of approximation in value space and rollout. We will describe this approach in some detail in Section 2.11. Related ideas will also be discussed in the context of Bayesian estimation and sequential estimation in Section 2.10.

Note that the case of a deterministic system

<!-- formula-not-decoded -->

is particularly interesting, because we can then typically expect that the true parameter θ ∗ will be identified in a finite number of stages. The reason is that at each stage k , we are receiving a noiseless observation relating to θ , namely the state x k . Once the true parameter θ ∗ is identified, the problem becomes one of perfect state information.

## 1.6.9 Model Predictive Control

In this section, we will provide a brief summary of the model predictive control (MPC) methodology for control system design, with a view towards its connection with DP/RL, approximation in value space, and rollout schemes. We will focus on the classical control problem: keeping the state of a deterministic system close to some reference point, taken here to be the origin of the state space (see Fig. 1.6.15). Another type of classical

An extensive overview of the connections of the conceptual framework of this book with model predictive and adaptive control is given in the author's paper [Ber24]. The corresponding video is a good supplement to the present section and can be found at https://www.youtube.com/watch?v=UeVs0Op-Ui4 and a related video can also be found at https://www.youtube.com/watch?v=ZBRouvMat2Q

m

REGULATION PROBLEM

Keep the state near some given point

## PATH PLANNING FOLLOW A GIVEN TRAJECTORY REGULATION PROBLEM

-Component Control

Figure 1.6.15 Illustration of a classical regulation problem, known as the 'cartpole problem' or 'inverse pendulum problem.' The state is the two-dimensional vector of angular position and angular velocity. We aim to keep the pole at the upright position (state equal to 0) by exerting horizontal force u on the cart.

<!-- image -->

control problem is to keep the system close to a given trajectory (see Fig. 1.6.16). It can also be treated by forms of MPC, but will not be addressed in this book.

We discussed earlier the linear quadratic approach, whereby the system is represented by a linear model, the cost is quadratic in the state and the control, and there are no state and control constraints. The linear quadratic and other approaches based on state variable system representations and optimal control became popular, starting in the late 50s and early 60s. Unfortunately, however, the analytically convenient linear quadratic problem formulations are often not satisfactory. There are two main reasons for this:

- (a) The system may be nonlinear, and it may be inappropriate to use for control purposes a model that is linearized around the desired point or trajectory. Moreover, some of the control variables may be naturally discrete, and this is incompatible with the linear system viewpoint.
- (b) There may be control and/or state constraints, which are not handled adequately through quadratic penalty terms in the cost function. For example, the motion of a car may be constrained by the presence of obstacles and hardware limitations (see Fig. 1.6.16). The solution obtained from a linear quadratic model may not be suitable for such a problem, because quadratic penalties treat constraints 'softly' and may produce trajectories that violate the constraints.

These inadequacies of the linear quadratic formulation have motivated

FOLLOW A

GIVEN TRAJECTORY

Moving Obstacle

Fixed Obstacles

PATH PLANNING FOLLOW A GIVEN TRAJECTORY States at the End of t

Best Score Fixed Obstacles

Velocity

Constraints

Acceleration

<!-- image -->

Must Deal with State and Control Constraints Linear-Quadratic F

Must Deal with State and Control Constraints Linear-Quadratic F

Figure 1.6.16 Illustration of constrained motion of a car from point A to point B. There are state (position/velocity) constraints, and control (acceleration) constraints. When there are mobile obstacles, the state constraints may change unpredictably, necessitating on-line replanning.

MPC, which combines elements of several ideas that we have discussed so far, such as multistep lookahead, rollout with a base policy, and certainty equivalence. Aside from dealing adequately with state and control constraints, MPC is well-suited for on-line replanning, like all approximation in value space methods.

Note that the ideas of MPC were developed independently of the approximate DP/RL methodology. However, the two fields are closely related, and there is much to be gained from understanding one field within the context of the other, as the subsequent development will aim to show. A major di ff erence between MPC and finite-state stochastic control problems that are popular in the RL/artificial intelligence literature is that in MPC the state and control spaces are continuous/infinite, such as for example in self-driving cars, the control of aircraft and drones, or the operation of chemical processes. At the same time, at a fundamental level, this di ff erence turns out to be inconsequential, because the key underlying framework for approximation in value space, which is based on Newton's method, is valid for both discrete and continuous state and control spaces.

In this section, we will primarily focus on the undiscounted infinite

horizon deterministic problem, which involves the system

<!-- formula-not-decoded -->

whose state x k and control u k are finite-dimensional vectors. The cost per stage is assumed nonnegative

<!-- formula-not-decoded -->

(e.g., a positive definite quadratic cost). There are control constraints u k ∈ U ( x k ) ↪ and to simplify the following discussion, we will initially consider no state constraints. We assume that the system can be kept at the origin at zero cost, i.e.,

<!-- formula-not-decoded -->

For a given initial state x 0 , we want to obtain a sequence ¶ u 0 ↪ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ that satisfies the control constraints, while minimizing the total cost.

This is a classical problem in control system design, known as the regulation problem , where the aim is to keep the state of the system near the origin (or more generally some desired set point), in the face of disturbances and/or parameter changes. In an important variant of the problem, there are additional state constraints of the form x k ∈ X , and there arises the issue of maintaining the state within X , not just at the present time but also in future times. We will address this issue later in this section.

## The Classical Form of MPC - View as a Rollout Algorithm

We will first focus on a classical form of the MPC algorithm, discussed in the form given here by Keerthi and Gilbert [KeG88], with a view towards stability analysis. In this algorithm, at each encountered state x k , we apply a control ˜ u k that is computed as follows; see Fig. 1.6.17:

- (a) We solve an /lscript -stage optimal control problem involving the same cost function and the requirement that the state after /lscript steps is driven to 0, i.e., x k + /lscript = 0. This is the problem

<!-- formula-not-decoded -->

subject to the system equation constraints

<!-- formula-not-decoded -->

the control constraints

<!-- formula-not-decoded -->