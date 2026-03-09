# 3.5: Policy Gradient and Related Methods

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 406-426
**Topics:** policy gradient, gradient methods, proximal policy optimization, PPO, random direction, random search, cross-entropy

---

where r is the parameter vector obtained from the policy evaluation formula (3.23).

An important alternative for approximate policy improvement, is to compute a set of pairs ( i s ↪ ˜ θ ( i s ) ) , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , using Eq. (3.24), and fit these pairs with a policy approximation architecture (see the next section on approximation in policy space). The overall scheme then becomes a form of PI that is based on approximation in both value and policy spaces.

At the end of the last policy evaluation step of PI, we have obtained a final Q-factor approximation ˜ Q ( i↪ u↪ ˜ r ). Then, in on-line play mode, we may apply the policy

<!-- formula-not-decoded -->

i.e., use the (would be) next policy iterate. Alternatively, we may apply the one-step lookahead policy

<!-- formula-not-decoded -->

or its multistep lookahead version. The latter alternative implements a Newton step and will likely result in substantially better performance. However, it is more time consuming, particularly if it is implemented by using a computer model and model-free simulation. Still another possibility, which also implements a Newton step, is to replace the function

<!-- formula-not-decoded -->

in the preceding Eq. (3.25) with an o ff -line trained approximation.

## Challenges Relating to Approximate Policy Iteration

Approximate PI in its various forms has been the subject of extensive theoretical and applied research. A comprehensive discussion is beyond our scope, and we refer to the literature, for detailed accounts, including the DP textbook [Ber12] or the RL textbook [Ber19a]. Let us provide a few comments relating to the challenges of approximate PI implementation.

- (a) Architectural issues : The architecture ˜ Q θ ( i↪ u↪ r ) may involve the use of features, and it could be linear, or it could be nonlinear such as a neural network. A major advantage of a linear feature-based architecture is that the policy evaluation (3.23) is a linear least squares problem, which admits a closed-form solution. Moreover, when linear architectures are used, there is a broader variety of approximate policy evaluation methods with solid theoretical performance guarantees,

such as TD( λ ), LSTD( λ ), and LSPE( λ ), which are not described in this book, but are discussed extensively in the literature. Generally, identifying an architecture that fits the problem well and training it e ff ectively can be a challenging and time-intensive task.

- (b) Exploration issues : Generating an appropriate set of training triplets ( i s ↪ u s ↪ β s ) at the policy evaluation step poses considerable challenges. A major di ffi culty has to do with inadequate exploration , the Achilles' heel of approximate PI. In particular, straightforward ways to evaluate a policy θ , typically rely on Q -factor samples of θ starting from states frequently visited by θ . Unfortunately, this may bias the simulation by underrepresenting states that are unlikely to occur under θ . As a result, the Q -factor estimates of these underrepresented states may be highly inaccurate, potentially causing serious errors in the calculation of the improved control policy ˜ θ via the policy improvement Eq. (3.24).

One way to address this issue is to use a large number of initial states to form a rich and representative subset of the state space. To keep simulation costs manageable, it may be necessary to use relatively short trajectories. However, when using short trajectories it may be important to introduce a terminal cost function approximation in the policy evaluation step in order to make the cost sample β s more accurate. Other approaches to enhance exploration include the use of a so-called o ff -policy , i.e., a policy θ ′ other than the currently evaluated policy θ , which tends to visit states that are unlikely to be visited using θ . For further discussion, see Section 6.4 of the DP textbook [Ber12].

- (c) Oscillation issues : Contrary to exact PI, which is guaranteed to yield an optimal policy, approximate PI produces a sequence of policies, which are only guaranteed to lie asymptotically within a certain error bound from the optimal; see the books [BeT96], Section 6.2.2, [Ber12], Section 2.5, and [Ber19a], Section 5.3.5. Moreover, the generated policies may oscillate. By this we mean that after a few iterations, policies tend to repeat in cycles.

The associated parameter vectors r may also tend to oscillate, although it is possible that there is convergence in parameter space and oscillation in policy space. This phenomenon, known as chattering , is explained in the author's survey papers [Ber10c], [Ber11b], and book [Ber12] (Section 6.4.3), and can be quite problematic, because there is no guarantee that the policies involved in the oscillation are 'good' policies, and it is often di ffi cult to assess their performance relative to the optimal. We note, however, that oscillations can be avoided and approximate PI can be shown to converge under special conditions, which arise in particular when an aggregation approach is

used; see the approximate PI survey [Ber11b].

We refer to the literature for further discussion of the preceding issues, as well as a variety of other approximate PI methods.

## 3.3.4 Optimistic Policy Iteration with Parametric Q-Factor Approximation - SARSA and DQN

There are also 'optimistic' approximate PI methods with Q-factor approximation, and/or a few samples in between policy updates. We view these primarily as o ff -line training methods, but because of the limited number of samples between policy updates, they have the potential of on-line implementation. In this case, however, a number of di ffi culties must be overcome, as we will explain later in this section.

As an example, let us consider an extreme version of Q-factor parametric approximation that uses a single sample between policy updates. At the start of iteration k , we have the current parameter vector r k , we are at some state i k , and we have chosen a control u k . Then:

- (1) We simulate the next transition ( i k ↪ i k +1 ) using the transition probabilities p i k j ( u k ) glyph[triangleright]
- (2) We generate the control u k +1 with the minimization

<!-- formula-not-decoded -->

[In some schemes, to enhance exploration, u k +1 is chosen with a small probability to be a random element of U ( i k +1 ) or one that is ' /epsilon1 -greedy,' i.e., attains within some /epsilon1 the minimum above. This is commonly referred to as the use of an o ff -policy .]

- (3) We update the parameter vector via

<!-- formula-not-decoded -->

where γ k is a positive stepsize, and ∇ ( · ) denotes gradient with respect to r evaluated at the current parameter vector r k . In the special case where ˜ Q is a linear feature-based architecture, ˜ Q ( i↪ u↪ r ) = φ ( i↪ u ) ′ r , the gradient ∇ ˜ Q ( i k ↪ u k ↪ r k ) is just the feature vector φ ( i k ↪ u k ), and iteration (3.27) becomes

<!-- formula-not-decoded -->

Thus r k is changed in an incremental gradient direction : the one opposite to the gradient (with respect to r ) of the incremental error

<!-- formula-not-decoded -->

evaluated at the current iterate r k .

The process is now repeated with r k +1 , i k +1 , and u k +1 replacing r k , i k , and u k , respectively.

Extreme optimistic schemes of the type just described have received a lot of attention, in part because they admit a model-free implementation [i.e., the use of a simulator, which provides for each pair ( i k ↪ u k ), the next state i k +1 and corresponding cost g ( i k ↪ u k ↪ i k +1 ) that are needed in Eq. (3.27)]. They are often referred to as SARSA (State-Action-Reward-StateAction); see e.g., the books [BeT96], [BBD10], [SuB18]. When Q-factor approximation is used, their behavior is very complex, their theoretical convergence properties are unclear, and there are no associated performance bounds in the literature. In practice, SARSA is often implemented in a less extreme/optimistic form, whereby several (perhaps many) statecontrol-transition cost-next state samples are batched together and suitably averaged before updating the vector r k .

Other variants of the method attempt to reduce sampling e ff ort by storing the generated samples in a bu ff er for reuse in subsequent iterations through some randomized process (cf. our earlier discussion of exploration). This is also called sometimes experience replay , an idea that has been used since the early days of RL, both to save in sampling e ff ort and to enhance exploration. The DQN (Deep Q Network) scheme, championed by DeepMind (see Mnih et al. [MKS15]), is based on this idea (the term 'Deep' is a reference to DeepMind's a ffi nity for deep neural networks, but the idea of experience replay does not depend on the use of a deep neural network architecture).

Another interesting idea from DeepMind [MRM16] is to introduce asynchronous parallel computation into the algorithm, based on the theory of distributed asynchronous methods in DP, gradient optimization, and RL, by Bertsekas and Tsitsiklis [Ber82a], [TBA86], [BeT89], [Tsi94], [BeT00], [Ber19a].

## Q-Learning Algorithms and On-Line Play

Algorithms that approximate Q-factors, including SARSA and DQN, are fundamentally o ff -line training algorithms. This is because their training process is long and requires the collection of many samples before reaching a stage that resembles parameter convergence. It can thus be unreliable to use the interim approximate Q-factors for on-line decision making, particularly in an adaptive context that involves changing system parameters, thereby requiring on-line replanning.

On the other hand, compared to the approximate PI method of Section 3.3.3, SARSA and DQN are far better suited for on-line implementation, because the control generation process of Eq. (3.26) can also be used to select controls on-line, thereby facilitating the combination of training and on-line control selection. To this end, it is important, among others,

to make sure that the parameters r k stay at 'safe' levels during the on-line control process, which can be a challenge. Still, even if this di ffi culty can be overcome in the context of a given problem, there are a number of other di ffi culties that SARSA and DQN can encounter during on-line play.

- (a) On-line exploration issues : There is a need to occasionally select controls using an o ff -policy in order to enhance exploration, and finding an adequate o ff -line policy in a given practical context can be a challenge. Moreover, the o ff -policy controls may improve exploration, but may be of poor quality, and in some contexts, may induce instability.
- (b) Robustness and replanning issues : In an adaptive control context where the problem parameters are changing, the algorithm may be too slow to adapt to the changes.
- (c) Performance degradation issues : Similar to our earlier discussion [cf. the comparison of Eqs. (3.24) and (3.25)], the minimization of Eq. (3.26) does not implement a Newton step, thereby resulting in performance loss. The alternative implementation

<!-- formula-not-decoded -->

which is patterned after Eq. (3.26), is better in this regard, but is computationally more costly, and thus less suitable for on-line implementation.

Generally speaking, the combination of o ff -line training and on-line play with the use of SARSA and DQN poses serious challenges. Nevertheless, encouraging results have been achieved in specific contexts, often with 'manual tuning,' i.e., tuning tailored to the problem at hand. Moreover, the popularity of these methods has been bolstered by the availability of open-source software that allow model-free implementations.

## 3.3.5 Approximate Policy Iteration for Infinite Horizon POMDP

In this section, we consider partial observation Markovian decision problems (POMDP) with a finite number of states and controls, and discounted additive cost over an infinite horizon. As discussed in Section 1.6.6, the optimal solution is typically intractable, so approximate DP/RL approaches must be used. In this section we focus on PI methods that are based on rollout, and approximations in policy and value space. They update a policy by using truncated rollout with that policy and a terminal cost function approximation. We focus on cost function approximation schemes, but Q-factor approximation is also possible.

Due to its simulation-based rollout character, the methodology of this section relies critically on the finiteness of the control space. It can be extended to POMDP with infinite state space but finite control space, although we will not consider this possibility in this section.

We assume that there are n states denoted by i ∈ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ and a finite set of controls U at each state. We denote by p ij ( u ) and g ( i↪ u↪ j ) the transition probabilities and corresponding transition costs, from i to j under u ∈ U . The cost is accumulated over an infinite horizon and is discounted by α ∈ (0 ↪ 1). At each new generated state j , an observation z from a finite set Z is obtained with known probability p ( z ♣ j↪ u ) that depends on j and the control u that was applied prior to the generation of j . The objective is to select each control optimally as a function of the prior history of observations and controls.

A classical approach to this problem is to convert it to a perfect state information problem whose state is the current belief b = ( b 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ b n ), where b i is the conditional distribution of the state i given the prior history. As noted in Section 1.6.6, b is a su ffi cient statistic, which can serve as a substitute for the set of available observations, in the sense that optimal control can be achieved with knowledge of just b .

In this section, we consider a generalized form of su ffi cient statistic, which we call the feature state and we denote by y . We require that the feature state y subsumes the belief state b . By this we mean that b can be computed exactly knowing y . One possibility is for y to be the union of b and some distinguishable characteristics of the belief state, or some compact representation of the measurement history up to the current time (such as a number of most recent measurements, or the state of a finitestate controller).

We further assume that y can be sequentially generated using a known feature estimator F ( y↪ u↪ z ). By this we mean that given that the current feature state is y , control u is applied, and observation z is obtained, the next feature can be exactly predicted as F ( y↪ u↪ z ).

Clearly, since b is a su ffi cient statistic, the same is true for y . Thus the optimal costs achievable by the policies that depend on y and on b are the same. However, specific suboptimal schemes may become more e ff ective with the use of the feature state y instead of just the belief state b .

The optimal cost J * ( y ), as a function of the su ffi cient statistic/feature state y , is the unique solution of the corresponding Bellman equation

<!-- formula-not-decoded -->

Here we use the following notation:

b y is the belief state that corresponds to feature state y , with components denoted by b y↪i , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n .

Transition Cost ik+1

Original

Observer

2k+1

Feature Estimator

F (y, u, 2)

System

Policy yk+1

4k+1

44+1

Figure 3.3.1 Composite system simulator for POMDP for a given policy. The starting state i k at stage k of a trajectory is generated randomly using the belief state b k , which is in turn computed from the feature state y k .

<!-- image -->

ˆ g ( y↪ u ) is the expected cost per stage

<!-- formula-not-decoded -->

ˆ p ( z ♣ b y ↪ u ) is the conditional probability that the next observation will be z given the current belief state b y and control u

F is the feature state estimator. In particular, F ( y↪ u↪ z ) is the next feature vector, when the current feature state is y , control u is applied, and observation z is obtained.

The feature space reformulation of the problem can serve as the basis for approximation in value space, whereby J * is replaced in Bellman's equation by some function ˜ J after one-step or multistep lookahead. For example a one-step lookahead scheme yields the suboptimal policy ˜ θ given by

<!-- formula-not-decoded -->

In /lscript -step lookahead schemes, ˜ J is used as terminal cost function in an /lscript -step horizon version of the original infinite horizon problem. In the standard form of a rollout algorithm, ˜ J is the cost function of some base policy. We will next discuss a rollout scheme with /lscript -step lookahead, which involves rollout truncation and terminal cost approximation.

## Truncated Rollout with Terminal Cost Function Approximation

In the pure form of the rollout algorithm, the cost function approximation ˜ J is the cost function J θ of a known base policy θ , and its value ˜ J ( y ) = J θ ( y )

"k+1

Transition Cost ix+1

Original

System

Uk +1

Observer

24 +2

Feature Estimator

1k +2

Policy ik +2

4k+2

at any y is obtained by first extracting b from y , and then running a simulator starting from b , and using the system model, the feature generator, and θ . In the truncated form of rollout, ˜ J ( y ) is obtained by running the simulator of θ for a given number of steps m , and then adding a terminal cost approximation ˆ J (¯ y ) for each terminal feature state ¯ y that is obtained at the end of the m steps of the simulation with θ (see Fig. 3.3.1).

Thus the rollout policy is defined by the base policy θ , the terminal cost function approximation ˆ J , the number of steps m after which the simulated trajectory with θ is truncated, and the lookahead size /lscript . The choices of m and /lscript are typically made by trial and error, based on computational tractability among other considerations, while ˆ J may be chosen on the basis of problem-dependent insight or through the use of some o ff -line approximation method. In some variants of the method, the multistep lookahead may be implemented approximately using a Monte Carlo tree search or adaptive sampling scheme.

Using m -step rollout between the /lscript -step lookahead and the terminal cost approximation gives the method the character of a single PI. We will use repeated truncated rollout as the basis for constructing a PI algorithm, which we will discuss next.

## Supervised Learning of Rollout Policies and Cost Functions Approximate Policy Iteration

The rollout algorithm uses multistep lookahead and on-line simulation of the base policy to generate the rollout control at any feature state of interest. To avoid the cost of on-line simulation, we can approximate the rollout policy o ff -line by using some approximation architecture, potentially involving a neural network. This is policy approximation built on top of the rollout scheme.

To this end, we may introduce a parametric family/architecture of policies of the form ˆ θ ( y↪ r ), where r is a parameter vector. We then construct a training set that consists of a large number of sample feature state-control pairs ( y s ↪ u s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , such that for each s , u s is the rollout control at feature state y s . We use this data set to obtain a parameter ¯ r by solving a corresponding classification problem, which associates each feature state y with a control ˆ θ ( y↪ ¯ r ). The parameter ¯ r defines a classifier, which given a feature state y , classifies y as requiring control ˆ θ ( y↪ ¯ r ) (see Section 3.4).

We can also apply the rollout policy approximation to the context of PI. The idea is to view rollout as a single policy improvement, and to view the PI algorithm as a perpetual rollout process , which performs multiple policy improvements, using at each iteration the current policy as the base policy, and the next policy as the corresponding rollout policy.

In particular, we consider a PI algorithm where at the typical iteration we have a policy θ , which we use as the base policy to generate

many feature state-control sample pairs ( y s ↪ u s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , where u s is the rollout control corresponding to feature state y s . We then obtain an 'improved' policy ˆ θ ( y↪ r ) with an approximation architecture and a classification algorithm, as described above. The 'improved' policy is then used as a base policy to generate samples of the corresponding rollout policy, which is approximated in policy space, etc.

To use truncated rollout in this PI scheme, a terminal cost approximation is required, which can take a variety of forms. Using zero is a simple possibility, which may work well if either the size /lscript of multistep lookahead or the length m of the rollout is relatively large. Another possibility is to use as terminal cost in the truncated rollout an approximation of the cost function of some base policy, which may be obtained with a neural network-based approximation architecture.

In particular, at any policy iteration with a given base policy, once the rollout data is collected, one or two neural networks are constructed: A policy network that approximates the rollout policy, and (in the case of rollout with truncation) a value network that constructs a cost function approximation for that rollout policy. Thus, we may consider two types of methods:

- (a) Approximate rollout and PI with truncation , where each generated policy as well as its cost function are approximated by a policy and a value network, respectively. The cost function approximation of the current policy is used to truncate the rollout trajectories that are used to train the next policy.
- (b) Approximate rollout and PI without truncation , where each generated policy is approximated using a policy network, but the rollout trajectories are continued up to a large maximum number of stages (enough to make the cost of the remaining stages insignificant due to discounting) or upon reaching a termination state. The advantage of this scheme is that only a policy network is needed; a value network is unnecessary since there is no rollout truncation with cost function approximation at the end.

Note that as in all approximate PI schemes, the sampling of feature states used for training is subject to exploration concerns. In particular, for each policy approximation, it is important to include in the sample set ¶ y s ♣ s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q ♦ , a subset of feature states that are 'favored' by the rollout trajectories; e.g., start from some initial subset of feature states y s and selectively add to this subset feature states that are encountered along the rollout trajectories. This is a challenging issue, which must be approached with care.

An extensive case study of the methodology of this section was given in the paper by Bhattacharya et al. [BBW20], for the case of a pipeline repair problem. The implementation used there also includes the use of

a partitioned state space architecture and an asynchronous distributed algorithm for o ff -line training; see Section 3.4.2. See also the paper by Bhattacharya et al. [BKB23], which deals with large multiagent POMDP problems, involving multiple robots operating on a network.

## 3.3.6 Advantage Updating - Approximating Q-Factor Di ff erences

Let us now explore an important alternative to computing Q-factor approximations. It is motivated by the potential benefit of approximating Q-factor di ff erences rather than Q-factors. In this method, called advantage updating , instead of computing and comparing Q * k ( x k ↪ u k ) for all u k ∈ U k ( x k ), we compute

<!-- formula-not-decoded -->

The function A k ( x k ↪ u k ) can serve to compare controls, i.e., at state x k select

<!-- formula-not-decoded -->

and this can also be done when A k ( x k ↪ u k ) is approximated with a value network.

Note that in the absence of approximations, selecting controls by advantage updating is equivalent to selecting controls by comparing their Q-factors. By contrast, when approximation is involved, comparing advantages instead of Q-factors can be important, because the former may have a much smaller range of values than the latter. In particular, Q * k may embody sizable quantities that depend on x k but are independent of u k , and which may interfere with algorithms such as the fitted value iteration (3.19)-(3.20). Thus, when training an architecture to approximate Q * k , the training algorithm may naturally try to capture the large scale behavior of Q * k , which may be irrelevant because it may not be reflected in the Q-factor di ff erences A k . However, with advantage updating, we may instead focus the training process on finer scale variations of Q * k , which may be all that matters. Here is an example (first given in the book [BeT96]) of what can happen when trained approximations of Q-factors are used.

## Example 3.3.1

Consider the deterministic scalar linear system

<!-- formula-not-decoded -->

and the quadratic cost per stage

<!-- formula-not-decoded -->

where δ is a very small positive constant [think of δ -discretization of a continuoustime problem involving the di ff erential equation dx ( t ) glyph[triangleleft]dt = u ( t )]. Let us focus on the stationary policy π , which applies at state x the control

<!-- formula-not-decoded -->

and view it as the base policy of a rollout algorithm. The Q-factors of π over an infinite number of stages can be calculated to be

<!-- formula-not-decoded -->

(We omit the details of this calculation, which is based on the classical analysis of linear-quadratic optimal control problems; see e.g., Section 1.5, or [Ber17a], Section 3.1.) Thus the important part of Q π ( x↪ u ) for the purpose of rollout policy computation is

<!-- formula-not-decoded -->

However, when a value network is trained to approximate Q π ( x↪ u ), the approximation will be dominated by 5 x 2 4 , and the important part (3.28) will be 'lost' when δ is very small. By contract, the advantage function can be calculated to be

<!-- formula-not-decoded -->

and when approximated with a value network, the approximation will be essentially una ff ected by δ .

## The Use of a Baseline

The idea of advantage updating is also related to the useful technique of subtracting a suitable constant (often called a baseline ) from a quantity that is estimated; see Fig. 3.3.2 (in the case of advantage updating, the baselines depend on x k , but the same general idea applies). This idea can also be used in the context of the fitted value iteration method given earlier, as well as in conjunction with other simulation-based methods in RL.

Example 3.1.1 also points to the connection between the ideas underlying advantage updating and the rollout methods for small stage costs relative to the cost function approximation, which we discussed in Section 2.6. In both cases it is necessary to avoid including terms of disproportionate size in the target function that is being approximated. The remedy in both cases is to subtract from the target function a suitable state-dependent baseline.

u u

Figure 3.3.2 Illustration of the idea of subtracting a baseline constant from a cost or Q-factor approximation. Here we have samples h ( u 1 ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ h ( u q ) of a scalar function h ( u ) at sample points u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u q , and we want to approximate h ( u ) with a linear function ˜ h ( u↪ r ) = ru , where r is a scalar tunable weight. We subtract a baseline constant b from the samples, and we solve the problem

<!-- image -->

<!-- formula-not-decoded -->

By properly adjusting b , we can improve the quality of the approximation, which after subtracting b from all the sample values, takes the form ˜ h ( u↪ b↪ r ) = b + ruglyph[triangleright] Conceptually, b serves as an additional weight (multiplying the basis function 1), which enriches the approximation architecture.

## 3.3.7 Di ff erential Training of Cost Di ff erences for Rollout

Let us now consider ways to approximate Q-factor di ff erences (cf. our advantage updating discussion of the preceding section) by approximating cost function di ff erences first. We recall here that given a base policy π = ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ , the o ff -line computation of an approximate rollout policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ consists of two steps:

- (1) In a preliminary phase, we compute approximations ˜ J k to the cost functions J k↪ π of the base policy π , possibly using simulation and a least squares fit from a parametrized class of functions.
- (2) Given ˜ J k and a state x k at time k , we compute the approximate Q -factor

<!-- formula-not-decoded -->

for all u ∈ U k ( x k ), and we obtain the (approximate) rollout control ˜ θ k ( x k ) from the minimization

<!-- formula-not-decoded -->

Unfortunately, this method also su ff ers from the error magnification inherent in the Q -factor di ff erencing operation. This motivates an alternative approach, called di ff erential training , which is based on cost-to-go di ff erence approximations. To this end, we note that to compute the rollout control ˜ θ k ( x k ), it is su ffi cient to have the di ff erences of costs-to-go

<!-- formula-not-decoded -->

where θ k ( x k ) is the control applied by the base policy at x k .

We thus consider a function approximation approach, whereby given any two states x k +1 and ˆ x k +1 , we obtain an approximation ˜ G k +1 ( x k +1 ↪ ˆ x k +1 ) of the cost di ff erence (3.29). We then compute the rollout control by

<!-- formula-not-decoded -->

where θ k ( x k ) is the control applied by the base policy at x k . Note that the minimization (3.30) aims to simply subtract the approximate Q-factor of the base policy control θ k ( x k ) from the approximate Q-factor of every other control u ∈ U k ( x k ).

An important point here is that the training of an approximation architecture to obtain ˜ G k +1 can be done using any of the standard training methods, and a 'di ff erential' system, whose 'states' are pairs ( x k ↪ ˆ x k ) and will be described shortly. To see this, let us denote for all k and pair of states ( x k ↪ ˆ x k )

<!-- formula-not-decoded -->

the cost function di ff erences corresponding to the base policy π . We consider the DP equations corresponding to π , and to x k and ˆ x k :

<!-- formula-not-decoded -->

and we subtract these equations to obtain

<!-- formula-not-decoded -->

for all ( x k ↪ ˆ x k ) and k . Therefore, G k can be viewed as the cost-to-go function for a problem involving a fixed policy (the base policy), the state ( x k ↪ ˆ x k ), the cost per stage

<!-- formula-not-decoded -->

and the system equation

<!-- formula-not-decoded -->

Thus, it can be seen that any of the standard methods that can be used to train architectures that approximate J k↪ π , can also be used for training architectures that approximate G k . For example, one may use simulationbased methods that generate pairs of trajectories starting at the pair of initial states ( x k ↪ ˆ x k ), and generated according to Eq. (3.32) by using the base policy π . Note that a single random sequence ¶ w 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 ♦ may be used to simultaneously generate samples of G k ( x k ↪ ˆ x k ) for several triples ( x k ↪ ˆ x k ↪ k ), and in fact this may have a substantial beneficial e ff ect.

A special case of interest arises when a linear, feature-based architecture is used for the approximator ˜ G k . In particular, let φ k be a feature extraction mapping that associates a feature vector φ k ( x k ) with state x k and time k , and let ˜ G k be of the form

<!-- formula-not-decoded -->

where r k is a tunable weight vector of the same dimension as φ k ( x k ) and prime denotes transposition. The rollout policy is generated by

<!-- formula-not-decoded -->

which corresponds to using r ′ k +1 φ k +1 ( x k +1 ) (plus an unknown inconsequential constant) as an approximation to J k +1 ↪ π ( x k +1 ). Thus, in this approach, we essentially use a linear feature-based architecture to approximate the cost functions J k↪ π of the base policy, but we train this architecture using the di ff erential system (3.32) and the di ff erential cost per stage of Eq. (3.31) . This is done by selecting pairs of initial states, running in parallel the corresponding trajectories using the base policy, and subtracting the resulting trajectory costs from each other.

## 3.4 LEARNING A POLICY IN APPROXIMATE DP

We have focused so far on approximation in value space using parametric architectures. In this section we will discuss how the cost function approximation methods discussed earlier this chapter can be adapted for

the purpose of approximation in policy space, whereby we approximate a given policy by using optimization over a parametric family of some form. Throughout this section we focus on a fixed policy and we focus on the o ff -line training of that policy.

In particular, suppose that for a given stage k , we have access to a dataset of sample state-control pairs ( x s k ↪ u s k ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q↪ obtained through some unspecified control process, such as rollout or problem approximation. We may then wish to 'learn' this process by training the parameter vector r k of a parametric family of policies ˜ θ k ( x k ↪ r k ) ↪ using least squares minimization/regression:

<!-- formula-not-decoded -->

cf. our discussion of approximation in policy space in Section 1.3.3.

## 3.4.1 The Use of Classifiers for Approximation in Policy Space

As we have noted in Section 3.1, in the case of a continuous control space, training of a parametric architecture for policy approximation is similar to training for a cost approximation. In the case where the control space is finite, however, it is useful to make the connection of approximation in policy space with classification ; cf. Fig. 3.1.2 and the discussion of Section 3.1.

Classification is an important subject in machine learning. The objective is to construct an algorithm, called a classifier , which assigns a given 'object' to one of a finite number of 'categories' based on its 'characteristics.' Here we use the term 'object' generically. In some cases, the classification may relate to persons or situations. In other cases, an object may represent a hypothesis, and the problem is to decide which of the hypotheses is true, based on some data. In the context of approximation in policy space, objects correspond to states, and categories correspond to controls to be applied at the di ff erent states . Thus in this case, we view each sample x s k ↪ u s k as an object-category pair.

( ) Generally, in classification we assume that we have a population of objects, each belonging to one of m categories c = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m . We want to be able to assign a category to any object that is presented to us. Mathematically, we represent an object with a vector x (e.g., some raw description or a vector of features of the object), and we aim to construct a rule that assigns to every possible object x a unique category c .

To illustrate a popular classification method, let us assume that if we draw an object x at random from this population, the conditional probability of the object being of category c is p ( c ♣ x ). If we know the probabilities p ( c ♣ x ), we can use a classical statistical approach, whereby we assign x to

the category c ∗ ( x ) that has maximal posterior probability, i.e.,

<!-- formula-not-decoded -->

This is called the Maximum a Posteriori rule (or MAP rule for short; see for example the book [BeT08], Section 8.2, for a discussion).

When the probabilities p ( c ♣ x ) are unknown, we may try to estimate them using a least squares optimization, based on the following property, whose proof is outlined in Exercise 3.1.

Proposition 3.4.1: (Least Squares Property of Conditional Probabilities) Let ξ ( x ) be any prior distribution of x , so that the joint distribution of ( c↪ x ) is

<!-- formula-not-decoded -->

For a pair of classes ( c↪ c ′ ), define z ( c↪ c ′ ) by

<!-- formula-not-decoded -->

and for a fixed class c and any function h of ( c↪ x ), consider

<!-- formula-not-decoded -->

the expected value with respect to the distribution ζ ( c ′ ↪ x ) of the random variable ( z ( c↪ c ′ ) -h ( c↪ x ) ) 2 . Then p ( c ♣ x ) minimizes this expected value over all functions h ( c↪ x ), i.e., for all functions h and all classes c , we have

<!-- formula-not-decoded -->

The proposition states that p ( c ♣ x ) is the function of ( c↪ x ) that minimizes

<!-- formula-not-decoded -->

over all functions h of ( c↪ x ), for any prior distribution of x and class c . This suggests that we can obtain approximations to the probabilities p ( c ♣ x ), c = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , by minimizing an empirical/simulation-based approximation of the expected value (3.36).

More specifically, let us assume that we have a training set consisting of q object-category pairs ( x s ↪ c s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , and corresponding vectors

<!-- formula-not-decoded -->

and let us adopt a parametric approach. In particular, for each category c = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , we approximate the probability p ( c ♣ x ) with a function ˜ h ( c↪ x↪ r ) that is parametrized by a vector r , and optimize over r the empirical approximation to the expected squared error of Eq. (3.36). Thus we can obtain r by the least squares regression:

<!-- formula-not-decoded -->

perhaps with some quadratic regularization added. The functions ˜ h ( c↪ x↪ r ) may be provided for example by a feature-based architecture or a neural network.

/negationslash

Note that each training pair ( x s ↪ c s ) is used to generate m examples for use in the regression problem (3.37): m -1 'negative' examples of the form ( x s ↪ 0), corresponding to the m -1 categories c = c s , and one 'positive' example of the form ( x s ↪ 1), corresponding to c = c s . Note also that the incremental gradient method can be applied to the solution of this problem.

The regression problem (3.37) approximates the minimization of the expected value (3.36), so we conclude that its solution ˜ h ( c↪ x↪ ¯ r ), c = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , approximates the probabilities p ( c ♣ x ). Once this solution is obtained, we may use it to classify a new object x according to the rule

<!-- formula-not-decoded -->

which approximates the MAP rule (3.34); cf. Fig. 3.4.1.

Returning to approximation in policy space, for a given training set ( x s k ↪ u s k ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q↪ the classifier just described provides (approximations to) the 'probabilities' of using the controls u k ∈ U k ( x k ) at the states x k , so it yields a 'randomized' policy ˜ h ( u↪ x k ↪ r k ) for stage k [once the values ˜ h ( u↪ x k ↪ r k ) are normalized so that, for any given x k , they add to 1]; cf. Fig. 3.4.2. In practice, this policy is usually approximated by the deterministic policy ˜ θ k ( x k ↪ r k ) that uses at state x k the control of maximal probability at that state; cf. Eq. (3.38).

For the simpler case of a classification problem with just two categories, say A and B , a similar formulation is to hypothesize a relation of the following form between object x and its category:

<!-- formula-not-decoded -->

k)

Data-Trained

Maxu

Classifier

Idealized

Maxc

Maxc

Data-Trained

MAP Classifier

(e.g., a NN)

Classifier

2 Illustration of classifica

τ

τ

<!-- image -->

Next Partial Tours, MAP Classifier Data-Trained Max MAX max

Figure 3.4.1 Illustration of the MAP classifier c ∗ ( x ) for the case where the probabilities p ( c ♣ x ) are known [cf. Eq. (3.34)], and its data-trained version ˜ c ( x↪ ¯ r ) [cf. Eq. (3.38)]. The classifier may be obtained by using the data set ( x s k ↪ u s k ) ↪ s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q↪ and an approximation architecture such as a feature-based architecture or a neural network.

Next Partial Tours, MAP Classifier Data-Trained Max MAX max

Figure 3.4.2 Illustration of classification-based approximation in policy space. The classifier, defined by the parameter r k , is constructed by using the training set ( x s k ↪ u s k ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q . It yields a randomized policy that consists of the probability ˜ h ( u↪ x k ↪ r k ) of using control u ∈ U k ( x k ) at state x k . This policy is approximated by the deterministic policy ˜ θ k ( x k ↪ r k ) that uses at state x k the control that maximizes over u ∈ U k ( x k ) the probability ˜ h ( u↪ x k ↪ r k ) [cf. Eq. (3.38)].

<!-- image -->

where ˜ h is a given function and r is the unknown parameter vector. Given a set of q object-category pairs ( x 1 ↪ z 1 ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ( x q ↪ z q ) where

<!-- formula-not-decoded -->

we obtain r by the least squares regression:

<!-- formula-not-decoded -->

) System PID Controller

The optimal parameter vector ¯ r is used to classify a new object with data vector x according to the rule

<!-- formula-not-decoded -->

In the context of DP and approximation in policy space, this classifier may be used, among others, in stopping problems where there are just two controls available at each state: stopping (i.e., moving to a termination state) and continuing (i.e., moving to some nontermination state).

There are several variations of the preceding classification schemes, for which we refer to the specialized literature. Moreover, there are several commercially and publicly available software packages for solving the associated regression problems and their variants. They can be brought to bear on the problem of parametric approximation in policy space using any training set of state-control pairs, regardless of how it was obtained.

## 3.4.2 Policy Iteration with Value and Policy Networks

We noted earlier that contrary to rollout, approximate policy iteration (PI) is fundamentally an o ff -line training algorithm, because for a large scale problem, it is necessary to represent the cost functions or Q-factors of the successively generated policies with an approximation architecture. Thus, in a typical implementation, approximate PI involves the successive use of value networks to represent the cost functions of the generated policies, and one-step or multistep lookahead minimization to implement policy improvement.

On the other hand, it is also possible to use policy networks to approximate the results of policy improvement. In particular, we can start with a base policy and a terminal cost approximation, and generate state-control samples of the corresponding truncated rollout policy. These samples can be used with an approximation in policy space scheme to train a policy network that approximates the truncated rollout policy.

Then the cost function of the policy network can be approximated with a value network using the cost approximation methodology that we have discussed in this chapter. This value network can be used in turn as a terminal cost approximation in a truncated rollout algorithm where the previously obtained policy network can be used as a base policy. A new policy network can then be trained using samples of this rollout policy, etc. Thus a perpetual rollout scheme is obtained, which involves a sequence of value and policy networks.

One may also consider approximate PI algorithms that do not use a value network at all. Indeed the value network is only used to provide the approximate cost function values of the current policy, which are needed to calculate samples of the improved policy and train the corresponding

State Space Partition

Initial State Truncated Rollout Using Local Policy Network

Terminal Cost Supplied bu Local Value Network

Terminal Cost Supplied by Local Value Network

Figure 3.4.3 Illustration of a truncated rollout scheme with a partitioned architecture. A local value network is used for terminal cost function approximation for each subset of the partition.

<!-- image -->

policy network. On the other hand the samples of the improved policy can also be computed by rollout, using simulation-generated cost function values of the current policy. If the rollout can be suitably implemented with simulation, the training of a value network may be unnecessary.

## Multiprocessor Parallelization

We have noted earlier that parallelization and distributed computation can be used in several di ff erent ways in rollout and PI schemes, including Qfactor, Monte Carlo, and multiagent parallelization. It is also possible to consider the use of multiple neural networks in the implementation of rollout or approximate PI. For example, when feature-based partitioning of the state space is used (cf. Example 3.1.8), we may consider a multiprocessor parallelization scheme, which involves multiple local value and/or policy networks, which operate locally within a subset of the state space partition; see Figs. 3.4.3 and 3.4.4.

Let us finally note that multiprocessor parallelization leads to the idea of an approximation architecture that involves a graph. Each node of the graph consists of a neural network and each arc connecting a pair of nodes corresponds to data transfer between the corresponding neural networks. The question of how to train such an architecture is quite complex and one may think of several alternative possibilities. For example the training may be collaborative with the exchange of training results and/or training data communicated periodically or asynchronously; see the book [Ber20a], Section 5.8.

and a Local Policy Network

Each Set Has a Local Value Network and a Local Policy Network

Each Set Has a Local Value Network and a Local Policy Network

State Space Partition

<!-- image -->

Initial State Truncated Rollout Using Local Policy Network

Terminal Cost Supplied bu Local Value Network

Terminal Cost Supplied by Local Value Network

Figure 3.4.4 Illustration of a perpetual truncated rollout scheme with a partitioned architecture. A local value network and a local policy network are used for each subset of the partition. The policy network is used as the base policy and the value network is used to provide a terminal cost function approximation.

State-control training pairs for the corresponding rollout policy are obtained by starting at an initial state within some subset of the partition, generating rollout trajectories using the local policy network, which are truncated once the state enters a di ff erent subset of the partition, with the corresponding terminal cost function approximation supplied by the value network of that subset.

When a separate processor is used for each subset of partition, the corresponding value networks are communicated between processors. This can be done asynchronously, with each processor sharing its value network as it becomes available. In a variation of this scheme, the local policy networks may also be shared selectively among processors for selective use in the truncated rollout process.

## 3.4.3 Why Use On-Line Play and not Just Train a Policy Network to Emulate the Lookahead Minimization?

This is a sensible and common question, which stems from the mindset that neural networks have extraordinary function approximation properties. In other words, why go through the arduous and time-consuming process of on-line lookahead minimization, if we can do the same thing o ff -line and represent the lookahead policy with a trained policy network? In particular, we can select the policy from a suitably restricted class of policies, such as a parametric class of the form θ ( x↪ r ) ↪ where r is a parameter vector. We may then estimate r using some type of o ff -line training. Then the on-line computation of controls θ ( x↪ r ) can be much faster compared with on-line lookahead minimization.

On the negative side, because parametrized approximations often involve substantial calculations, they are not well suited for on-line replan-