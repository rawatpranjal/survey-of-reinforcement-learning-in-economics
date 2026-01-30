## Convergence of Actor-Critic Methods with Multi-Layer Neural Networks

## Haoxing Tian, Ioannis Ch. Paschalidis, Alex Olshevsky

Department of Electrical and Computer Engineering Boston University Boston, MA 02215, USA { tianhx, yannisp, alexols } @bu.edu

## Abstract

The early theory of actor-critic methods considered convergence using linear function approximators for the policy and value functions. Recent work has established convergence using neural network approximators with a single hidden layer. In this work we are taking the natural next step and establish convergence using deep neural networks with an arbitrary number of hidden layers, thus closing a gap between theory and practice. We show that actor-critic updates projected on a ball around the initial condition will converge to a neighborhood where the average of the squared gradients is ˜ O (1 / √ m ) + O ( ϵ ) , with m being the width of the neural network and ϵ the approximation quality of the best critic neural network over the projected set.

## 1 Introduction

Reinforcement Learning (RL) has emerged as a powerful tool for solving decision-making problems in a model-free way. Among the various RL algorithms, the Actor-Critic (AC) method (Konda &amp; Tsitsiklis (1999); Barto et al. (1983)) has shown great success in various domains, including robotics, game playing, and control systems (LeCun et al. (2015); Mnih et al. (2016); Silver et al. (2017)). AC involves simultaneous updates of two networks: an actor network that employs policy gradient (Sutton et al. (1999)) to update a parameterized policy, and a critic network which is driven by the Temporal Differences (TD) in the estimated value function. While AC methods with neural networks used for both actor and critic have achieved widespread use in practice, a fully satisfactory analysis of their convergence guarantees is currently lacking.

In recent years, a number of theoretical studies of AC have obtained provable convergence rates and performance analyses. Almost all works in this area assumed linear, rather than neural networkbased, approximators for both actor and critic. A 'two-timescale' linear AC was analysed in Wu et al. (2020), with a convergence rate of ˜ O ( T -1 / 4 ) , where T is the total number of iterations and ˜ O ( · ) refers to potential logarithmic terms omitted from the notation; the term 'two-timescale' refers to the fact that the stepsizes for the actor update and critic update are not proportional to each other, but rather the actor steps are asymptotically negligible compared to the critic steps. A 'singletimescale' linear AC method was considered in Olshevsky &amp; Gharesifard (2022); Chen et al. (2021) and both works obtained a convergence rate of O ( T -0 . 5 ) under an i.i.d. sampling assumption on the underlying MDP. The more realistic Markov sampling case was analyzed in the recent paper Chen &amp; Zhao (2022), which also established a convergence rate of ˜ O ( T -0 . 5 ) . All these results relied on linear approximations.

To our knowledge, convergence rates for AC with neural approximators were analyzed only in two recent works Wang et al. (2019); Cayci et al. (2022). Both of these papers considered neural net-

works with a single hidden layer. The paper Wang et al. (2019) obtained a convergence rate of O ( T -0 . 5 ) with a final error of O ( m -0 . 25 ) under i.i.d. sampling, where m is the width of hidden layer. The case of Markov sampling was considered in Cayci et al. (2022) which improved this to ˜ O ( T -0 . 5 ) and ˜ O ( m -0 . 5 ) , respectively. Further, both Wang et al. (2019); Cayci et al. (2022) considered 'double-loop' methods where, in the inner loop, the critic takes sufficiently many steps to accurately estimate Q -values. Such double-loop methods do not match prevailing practice and are considerably easier to analyze since they can be shown to approximate gradient descent.

Further, Cayci et al. (2022) required a projection onto a ball of radius O ( m -1 / 2 ) around the initial condition. Although a full representation theory for such neural networks is unknown, this is clearly limiting as compared to Wang et al. (2019) which only required projection onto a ball of constant radius. For nonlinear approximations, such projections are usually needed to stabilize the algorithm; without them, AC can diverge both in theory and practice.

Table 1: Comparisons with previous work.

| Reference                     | Algorithm                    | Sampling   | Approximation       | Projection   | Convergence rate                       | Convergence rate   |
|-------------------------------|------------------------------|------------|---------------------|--------------|----------------------------------------|--------------------|
|                               |                              |            |                     | Radius       | w.r.t. T                               | w.r.t. m           |
| Wu et al. (2020)              | Two-timescale Single-loop    | Markov     | Linear              | N/A          | ˜ O ( T - 0 . 4 )                      | N/A                |
| Olshevsky &Gharesifard (2022) | Single-timescale Single-loop | I.i.d.     | Linear              | N/A          | O ( T - 0 . 5 )                        | N/A                |
| Chen et al. (2021)            | Single-timescale Single-loop | I.i.d .    | Linear              | N/A          | O ( T - 0 . 5 )                        | N/A                |
| Chen &Zhao (2022)             | Single-timescale Single-loop | Markov     | Linear              | N/A          | ˜ O ( T - 0 . 5 )                      | N/A                |
| Wang et al. (2019)            | Double-loop                  | I.i.d.     | Single hidden layer | Constant     | O ( T - 0 . 5 )                        | O ( m - 0 . 25 )   |
| Cayci et al. (2022)           | Double-loop                  | Markov     | Single hidden layer | Decaying     | ˜ O ( T - 0 . 5 ) m sufficiently large | ˜ O ( m - 0 . 5 )  |
| Ours                          | Single-timescale Single-loop | Markov     | Any depth           | Constant     | ˜ O ( T - 0 . 5 )                      | ˜ O ( m - 0 . 5 )  |

The main contribution of this paper is to provide the first analysis of AC with neural networks of arbitrary depth. While replicating the earlier results of a ˜ O ( T -0 . 5 ) convergence rate and ˜ O ( m -0 . 5 ) error, our work considers a single-loop method with proportional step-sizes (sometimes called 'singletimescale'). We prove this result under Markov sampling and project onto a ball of constant radius around the initial condition. An explicit comparison of our result to previous work is given in Table 1. A more technical comparison is also given later after the statement of our main result.

Our main technical tool is the so-called 'gradient splitting' view of TD learning. This idea began with the paper Ollivier (2018) which observed that TD learning is exactly gradient descent when the underlying policy is such that the state transition matrix is reversible. In Liu &amp; Olshevsky (2021), this was generalized to non-reversible policies by introducing the notion of a 'gradient splitting' (discussed formally later in this work) and observing that, for linear approximation, TD updates are an example of gradient splitting. Gradient splitting is closely related to gradient descent, and the two processes can be analyzed similarly. A generalization to neural TD learning was given in Tian et al. (2023), which argued for an interpretation of nonlinear TD as approximate gradient splitting.

The analysis of AC that we perform in this work is trickier because both actor and critic updates rely on each other, and one must prove that the resulting errors in each process do not compound in interaction with each other. This difficulty arises because we do not consider the 'double loop' case where the actor can effectively wait for the critic to converge, so that actor steps resemble gradient steps with error; rather both actor and critic update simultaneously their (imperfect) estimates. Similarly to what was done in Olshevsky &amp; Gharesifard (2022), we show that we can draw on some ideas from control theory to prove that the resulting process converges with a so-called 'small-gain' analysis.

## 2 Preliminaries

We begin by standardizing notation and stating the key concepts that will enable us to formulate our results alongside all the assumptions they require.

## 2.1 Markov Decision Processes (MDP)

A finite discounted-reward MDP can be described by a tuple ( S, A, P env , r, γ ) where S is a finite state-space whose elements are vectors, and we use s 0 ∈ S to denote the starting state; A is a finite action space with cardinality n a ; P env = ( P env ( s ′ | s, a )) s,s ′ ∈ S,a ∈ A is the transition probability matrix, where P env ( s ′ | s, a ) is the probability of transitioning from s to s ′ after taking action a ; r : S × A → R is the reward function, where r ( s, a ) stands for the expected reward at state s and taking action a ; and γ ∈ (0 , 1) is the discount factor.

A policy π is a mapping π : S × A → [0 , 1] where π ( a | s ) is the probability that the agent takes action a in state s . Given a policy π , we can define the state transition matrix P ′ π = ( P ′ π ( s ′ | s )) s,s ′ ∈ S and the state-action transition matrix P π = ( P π ( s ′ , a ′ | s, a )) ( s,a ) , ( s ′ ,a ′ ) ∈ S × A as

<!-- formula-not-decoded -->

The stationary distribution over state-action pairs µ π is defined to be a nonnegative vector with coordinates summing to one and satisfying µ T π = µ T π P π , while the stationary distribution over states µ ′ π is defined similarly with µ ′ π T = µ ′ π T P ′ π . The Perron-Frobenius theorem guarantees that such a µ π and µ ′ π exist and are unique subject to some conditions on P ′ π , P π , e.g., aperiodicity and irreducibility (Gantmacher (1964)). We use µ π ( s, a ) to denote each entry of µ π and µ ′ π ( s ) each entry of µ ′ π . Clearly,

<!-- formula-not-decoded -->

The value function and the Q -function of a policy π is defined as:

<!-- formula-not-decoded -->

Here, E s,a,π stands for the expectation when action a is chosen in state s and all subsequent actions are chosen according to policy π . Throughout the paper, if π can be parameterized by θ , then we will use θ as a subscript instead of π , e.g., by writing V ∗ θ ( s ) instead of V ∗ π θ ( s ) .

If π is parameterized by θ , the Q -values satisfy the Bellman equation

<!-- formula-not-decoded -->

which can be stated in matrix notation as

<!-- formula-not-decoded -->

where Q ∗ θ = ( Q ∗ θ ( s, a )) ( s,a ) ∈ S × A and R = ( R ( s, a )) ( s,a ) ∈ S × A are vectors that stack up the Q -values and rewards, respectively. We will assume rewards are bounded:

Assumption 2.1 (Bounded Reward) . For any s, a ∈ S × A , | r ( s, a ) | ≤ r max .

This assumption is commonly adopted throughout the literature, e.g., among the previous literature in Cayci et al. (2022); Wu et al. (2020). An obvious implication of this is an upper bound on the Q -values for any policy:

## 2.2 The Policy Gradient Theorem

We introduce the quantity ϕ θ ( s ) , commonly called the discounted occupation measure which is defined as

<!-- formula-not-decoded -->

where P θ ( S t = s ) is the probability of being in state s after t steps -- and recall that we always begin in state s 0 . Next, we define ϕ θ ( s, a ) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the sum of both ϕ ( s ) and ϕ ( s, a ) equal to (1 -γ ) -1 rather than 1 :

<!-- formula-not-decoded -->

Now we are prepared to state the policy gradient theorem Sutton &amp; Barto (2018).

## Theorem 2.1. (Policy Gradient Theorem)

<!-- formula-not-decoded -->

It is standard to write this as

<!-- formula-not-decoded -->

which can be further rewritten in matrix form as

<!-- formula-not-decoded -->

where Φ θ is a diagonal matrix stacking up the ϕ θ ( s, a ) as its diagonal entries.

## 2.3 Parameterized Value Function and Policy

We will now state the various assumptions we have on the policies and their parametrizations. We will say that a function f : R → R is L -Lipschitz if

<!-- formula-not-decoded -->

and a differentiable function f : R → R is H -smooth if

<!-- formula-not-decoded -->

We will be using a multi-layer neural network to approximate the Q values under a policy. We basically follow the same setting as in Liu et al. (2020), with some changes as far as notation goes. Specifically, we define the following recursion

<!-- formula-not-decoded -->

where σ is an activation function and x ( k ) stands for the value of k 'th layer ( x (0) ∈ S × A is the input to this neural network). The neural network outputs Q ( s, a, w ) , which is defined as

<!-- formula-not-decoded -->

Notice that the output is linear to x ( K ) as no activation function is applied here. While this formulation does not have a bias, it is equivalent to a formulation with a bias if we pad all inputs with a single 1 , and add an additional node to every hidden layer that propagates this 1 to subsequent layers. We will assume that all the hidden layers have the same width which we denote by m , i.e., all the matrices w ( k ) have m rows and all the vector x ( k ) , k ≥ 1 are m -dimensional. The total number of layers in the neural network is denoted by K .

For simplicity, we will make the following assumption on the neural network. Throughout the paper, we will use || · || for the standard l 2 -norm.

Assumption 2.2. (Neural architecture and initialization) Suppose the neural network satisfies the following properties:

- (Input assumption) Any input to the neural network satisfies || x (0) || ≤ 1 .
- (Activation function assumption) σ is L σ -Lipschitz and H σ -smooth.
- (Initialization assumption) Each entry of the vector b satisfies | b r | ≤ 1 , ∀ r , and each entry of w ( k ) is randomly chosen from N (0 , 1) , independently across entries.

Liu et al. (2020) showed that with these assumptions, the following result holds with high probability - which we state as an assumption for our work.

Assumption 2.3. The absolute value of each entry of x ( k ) (the output of layer k of the neural network) is ˜ O m (1) at initialization.

Next, we will stack up the weights of different layers into a column vector w consisting of the entries of the matrices w (1) , . . . , w ( K ) , with its norm defined by

<!-- formula-not-decoded -->

where || · || F is the Frobenius norm. During the training process, only the weights w will be updated while the final weights b will be left to their initial value. For convenience, we define the vector Q ( w ) = ( Q ( s, a, w )) ( s,a ) ∈ S × A which stacks up Q ( s, a, w ) over all state-action pairs ( s, a ) . While this vector will never be actually used in the execution of any algorithm we consider due to its high dimensionality, it will be useful in some of the arguments we will make. Finally, we assume the parametrization of the policy π is smooth.

Assumption 2.4 (Smooth parametrization) . For all s, a , the quantities π ( a | s, θ ) , ln π ( a | s, θ ) are L π -Lipschitz and L ′ π -Lipschitz with respect to θ , respectively.

Note that this forces us to use a smooth activation function and rules out non-differentiable activation functions such as ReLU. If a RELU-like activation is needed, one could use a GeLU or ELU activation (which are smooth versions of ReLU) and still satisfy the above assumption. Note, also, that this assumption implicitly assumes that all policies are exploratory in the sense of assigning a positive probability to each action, since the derivative of ln x blows up as x → 0 .

## 2.4 Neural Actor-Critic

Wewill use Proj W {·} refer to projection onto a ball with constant radius around the initial condition of the critic, where

<!-- formula-not-decoded -->

We now introduce the neural AC, which updates the actor and critic parameters as

<!-- formula-not-decoded -->

where δ t is the TD error defined by

<!-- formula-not-decoded -->

and the samples are obtained as follows:

1. the state s t is generated by taking a step in the Markov chain P env from s t -1 ;
2. the action a t is chosen according to the policy π ( a | s t , θ t ) ;
3. the next state s ′ t , i.e, s ′ t = s t +1 , is determined according to the transition probability P env of the MDP;
4. the action a ′ t is an action chosen at the next state according to the policy π ( a | s ′ t , θ t ) ;
5. the state-action pair (ˆ s t , ˆ a t ) is obtained by first sampling a geometric random variable T with distribution { P ( T = t ) = (1 -γ ) γ t , t ≥ 0 } , and second obtaining T transitions by starting at s 0 and taking actions according to π ( a | s, θ t ) . Note that this update has to be re-done at every step, i.e., every t requires Geom( γ ) steps.

The above algorithm will be referred to as actor-critic with Markov sampling . It is also possible to consider a simplified variant, where step 1 is slightly altered as follows: the state s t is instead chosen i.i.d. at every step from the stationary distribution of µ θ t of the policy π θ t . This is referred to as actor-critic with i.i.d. sampling .

## 2.4.1 Approximation Assumptions

It is evident that any performance bound on AC will depend on how well the neural network used for the critic can approximate the true value function. If we choose a neural network architecture for which universal approximation theorems do not apply and it happens to poorly approximate the true Q -functions, we will likely obtain poor results. Here, we will largely sidestep this issue by defining ϵ to be the approximation quality of the critic; our final performance results will be in terms of ϵ .

Formally, we say that the vector Q is an ϵ -approximation to the true value function Q ∗ θ t of the policy π θ t if

<!-- formula-not-decoded -->

We then make the following assumption.

Assumption 2.5. (Approximation capabilities of critic) For all θ , there exists some set of weights ˆ w θ which give rise to an ϵ -approximation of Q ∗ θ .

Note that, since we do not say what ϵ is, this assumption could well be a definition of ϵ . Throughout the paper we will use ˆ Q ∗ θ t to denote an ϵ -approximation to Q ∗ θ t guaranteed by the above assumption. Thus,

<!-- formula-not-decoded -->

Further, we will assume that ˆ w θ is a smooth function of θ in the sense of its first and second derivatives.

Assumption 2.6. (Smoothness of critic approximation) Suppose there exists scalars L w ( i ) and H w ( i ) such that for all θ ,

<!-- formula-not-decoded -->

where λ max {·} stands for the largest eigenvalue.

For convenience, we define

<!-- formula-not-decoded -->

Finally, we need an additional assumption on the critic neural network. It should be obvious that any analysis of actor-critic has to assume that the critic is capable of approximating the correct Q -values. One part of this was already assumed earlier in Assumption 2.5, where we assumed that an approximation exists. However, it should be clear that in the nonlinear case this is insufficient: just because there exists an approximation which is good doesn't follow that it will be found during training, which is not known converge to the global minimizer in the nonlinear case, but rather only to a critical point.

We thus need something to rule out the possibility that the critic training gets stuck at a bad crticial point. It turns out that it suffices to assume (a quantitative version of the fact that) the critic is one-to-one map from weights to value functions.

Assumption 2.7. (State regularity) There exists some constant λ ′ &gt; 0 such that

<!-- formula-not-decoded -->

̸

Let us parse the meaning of this assumption. Because Q ( ˆ w ∗ θ ) = ˆ Q ∗ θ , it is appropriately viewed as a quantitative version of the statement that if w 1 = w 2 , then Q ( w 1 ) = Q ( w 2 ) . To see why this makes sense, note that the number of states is typically many magnitudes larger than the number of parameters in the critic. For example, in many applications the number of states often corresponds to the number of images (when states are captured through images) which is astronomical. Thus Q ( w ) will map w to a much higher dimensional space.

̸

If the states s are generated from a probability distribution which has a continuous density, and the activation functions are continuous and increasing, the chance that Q w 1 ( s ) = Q w 2 ( s ) even for one state s is zero. That is why we label it 'state regularity' as above (and recall that Q ( w ) stacks up Q w ( s ) for every state s ).

On a technical level, this property ensures that critic actually finds a good critic approximation in spite of the nonlinearity of the update. If the features are linear, this reduces to the assumption that the features are linearly independent, an assumption which is made in all previous and related work on AC method (Wu et al. (2020); Olshevsky &amp; Gharesifard (2022); Chen &amp; Zhao (2022); Kumar et al. (2023)) and TD Learning (Liu &amp; Olshevsky (2021); Xu &amp; Gu (2020); Cai et al. (2019); Zou et al. (2019)).

## 2.5 The Mixing of Markov Chains

It is standard to make an assumption to the effect that all the Markov chains that can arise satisfy a mixing condition. Otherwise, it is possible under Markov sampling for the state to fail to explore the entire state-space. This assumption, first introduced by Bhandari et al. (2018) in TD learning, now is commonly used in AC analysis (Olshevsky &amp; Gharesifard (2022); Wu et al. (2020); Chen &amp; Zhao (2022)).

Assumption 2.8 (Markov chain mixing) . There exists constants C &gt; 0 and β ∈ [0 , 1) with the following property: for all θ , if we consider a Markov chain generated by a t ∼ π ( ·| s t , θ ) , s t +1 ∼ P env ( ·| s t , a t ) starting from state s , then

<!-- formula-not-decoded -->

where p τ is the probability distribution of the state of this Markov chain after τ steps.

To assure AC explores every possible state, we make the following assumption:

Assumption 2.9. (Exploration) Suppose there exists some constant µ min &gt; 0 such that, for all θ , µ ′ θ is uniformly bounded away from 0 . In other words,

<!-- formula-not-decoded -->

Recall that µ θ was defined earlier to be the stationary distribution of the transition matrix associated with the policy π θ . A key point is that the constants C , β and µ min in the above assumptions do not depend on θ .

We note that there is some redundancy in our assumptions. As discussed above, we require ln π θ ( a | s ) to have a smooth gradient for all s, a , which ensures that π θ assigns a strictly positive probability to every action. This implies Assumptions 2.8 and 2.9 which can therefore be made into propositions. Nevertheless, we explicitly make Assumptions 2.8 and 2.9 (even though both of them are actually implied by our earlier assumption) since the quantities appearing in them (specifically, the mixing time β and the constant µ min ) appear in various bounds we will derive.

More precisely, we follow the earlier literature by setting Cβ τ to be proportional to T -0 . 5 , the typical of stepsize in Stochastic Gradient Descent. We call the smallest τ such that Cβ τ ≤ O ( T -0 . 5 ) the mixing time and denote it by τ mix . It is easy to see that τ mix = O ( (1 -β ) -1 log T ) . The quantity τ mix will appear throughout our paper.

## 2.6 D -norm and Dirichlet Norm in MDPs

A key ingredient is our analysis is the choice of norm: we have found that a certain norm originally introduced in Ollivier (2018) significantly simplifies analysis of the problem. We next introduce this norm and state our assumptions about it.

Let D θ = diag ( µ θ ( s, a )) be the diagonal matrix whose elements are given by the entries of the stationary distribution µ θ associated with the policy π θ . Given a function f : S × A → R , its D -norm is defined as

<!-- formula-not-decoded -->

The D -norm is similar to the Euclidean norm except each entry is weighted proportionally to the stationary distribution. We also define the Dirichlet semi-norm of f :

<!-- formula-not-decoded -->

A semi-norm satisfies the axioms of a norm except that it may be equal to zero at a non-zero vector. Note that || f || Dir depends on the policy both through the stationary distribution µ θ ( s, a ) as well as through the transition matrix P θ .

Finally, following Ollivier (2018), the weighted combination of the D -norm and the Dirichlet seminorm is denoted as N θ ( f ) will be defined

<!-- formula-not-decoded -->

Note that as long as µ θ ( s, a ) &gt; 0 , which is stated in Assumption 2.9, for all s, a , we have that √ N θ ( f ) is a valid norm.

## 3 Our Main Results

To simplify the expression that follow, we will adopt the notations ∆ V and ∆ Q for the two losses that we want to bound in our paper:

<!-- formula-not-decoded -->

Intuitively, ∆ V corresponds to the actor error: ideally, we want to reach a point where the gradient of the actor value function is zero. Note that, since the value function is not convex in general, the actor error is measured in terms of distance to a stationary point as above.

Similarly, ∆ Q is a measure of the critic error: it equals zero precisely if Q ( w t ) , the approximator of Q -function, equals ˆ Q θ t . Of course, as discussed above, the critic neural network may not be able to perfectly represent the true Q -function. Now we are ready to state our main results.

Theorem 3.1. Consider the neural AC algorithm mentioned in Section 2.4. Suppose Assumptions 2.1-2.9 hold and the step-sizes α θ and α w are both chosen to decay proportionally to O ( T -0 . 5 ) .

1. In the i.i.d. sampling case,

<!-- formula-not-decoded -->

2. In the Markov sampling case,

<!-- formula-not-decoded -->

In all O ( · ) notations above, we treat factors that do not depend on T, ϵ, m as constants.

We next provide a more detailed comparison to the previous works of (Wang et al. (2019); Cayci et al. (2022)). Our discussion partially reprises the discussion in the Introduction, but can now be discussed at a greater level of detail:

- Arbitrary depth/single-timescale. The main contribution of this paper to provide an analysis that applies to neural networks of arbitrary depth. Moreover, we do so in a singleloop/single-timescale method where the critic and actor iterate simultaneously, which is matching what is typically done in practice. Such an analysis is inherently more technically challenging, since when the actor can wait for the critic to go through sufficiently many iterations, one could argue that the resulting Q -values are approximately accurate and the process resembles gradient descent.
- Representability. Both previous works for the single-layer case assume the Q -function lies in some function class, which, as discussed after Assumption 6 in Farahmand et al. (2016), is one kind of 'no function approximation error' assumption. By contrast, we make no such assumption: rather we allow any approximation error for the critic ϵ , and our final result is given in terms of ϵ .

- Lower bound on m . Previous works require m , the width of neural network, to be sufficiently large. In Wang et al. (2019), given that m is sufficiently large, Section 3.1 and Corollary A.3 argue that the gradient, denoted by ¯ ϕ θ and ¯ ϕ w , can be well approximated by the 'centered feature mapping corresponding to the initialization', denoted by ¯ ϕ 0 . In Cayci et al. (2022), this dependency is even more emphasised since the upper bound shown in Theorem 2 could diverge with small m .
- Relation to NTK theory. NTKtheory (Jacot et al. (2018)) tells us that neural networks get more linear as m →∞ . The classic analyses of this proceed by arguing that as m →∞ , the neural network stays close to its initialization during training Chizat et al. (2019). In that sense, we should expect to get a convergence result for AC as m →∞ , but if the critic neural network stays close to its initial condition, the algorithm will effectively be using random linear features at initialization. For this reason, it is desirable not to argue that the critic neural network always stays close to its initial condition. We do not use such an argument in this work, whereas both Wang et al. (2019) and Cayci et al. (2022) obtain their results by arguing that the critic neural network stays close to its initial condition. This theoretical distinction is shown in Tian et al. (2023) to match what happens in simulations, which shows empirically that even for projected neural TD, the critic neural network will move to the boundary of the projection ball.
- Linearization. Previous works assume some kind of linearization around the initial point. The objective is explicitly linearized in Wang et al. (2019).In Cayci et al. (2022), while the objective is not linearized, the neural networks weights are projected onto a radius of size O (1 / √ m ) around the initial point.

## 4 Tools in Our Analysis

## 4.1 Choice of Norm and Gradient Splitting

A linear function h ( θ ) is said to be a gradient splitting of a convex quadratic f ( θ ) minimized at θ = a if

<!-- formula-not-decoded -->

In other words, a splitting h ( θ ) has exactly the same inner product with the 'direction to the optimal solution' as the true gradient of f ( θ ) (up to the factor of 1 / 2) . The connection between this idea and RL was made in the following papers:

- In Ollivier (2018) it was shown that in TD Learning, if the matrix P corresponds to a reversible Markov chain, then E [¯ g ( θ t )] = ∇ θ N ( f ) for some f . This makes Neural TD easy to analyze in the reversible case as it is exactly gradient descent.
- In Liu &amp; Olshevsky (2021), it was shown how to further use the function N ( · ) to analyze TD learning with linear approximation when the policy is not necessarily reversible. In particular, it was shown that the mean update of TD with linear approximation is a gradient splitting of the function N ( · ) . This is one of the crucial ideas we build on in this paper.

## 4.2 Nonlinear Small-Gain Theorem

Inspired by Olshevsky &amp; Gharesifard (2022), our second main tool is a nonlinear version of the small-gain theorem . Because the actor and critic update simultaneously, we need to rule out the possibility that errors in the actor compound with errors in the critic to create divergence. For example, it is conceivable that, when the policy is fixed, the critic converges to a reasonable approximation; when the critic is fixed, the actor converges to an approximate of the stationary point; but both updating simultaneously results in divergence.

The core idea of small-gain is to write these updates in such a way so that one can argue that if certain coefficients are small enough, this 'interconnection' of the actor and critic systems converges. The

Figure 1: Key property of gradient splitting: h ( θ ) has the same inner product with a -θ as ∇ f ( θ ) up to a factor of 1 / 2 .

<!-- image -->

small-gain theorem we use is a nonlinear version of the textbook version Drazin (1992). This is a widely-used trick in control theory that avoids the necessity of explicitly finding a Lyapunov function.

## 5 Conclusion

We have provided an analysis of Neural AC using a convex combination of the D -norm and the Dirichlet semi-norm to describe the error. Our main result is an error rate of O ( T -0 . 5 + ϵ ) + ˜ O ( m -0 . 5 ) under the i.i.d. sampling and O ( (log T ) 2 · T -0 . 5 + ϵ ) + ˜ O ( m -0 . 5 ) under the Markov sampling for neural networks of arbitrary depth. Crucially, our proof does not make assumptions that force the neural networks to stay close to their initial conditions, relying instead on arguments that show that neural networks which are not 'too nonlinear' will still converge to an approximate minimum.

## Acknowledgments and Disclosure of Funding

This research was partially supported by the NSF under grants CCF-2200052, DMS-1664644, and IIS-1914792, by the ONR under grant N00014-19-1-2571, by the DOE under grant DE-AC0205CH11231, by the NIH under grant UL54 TR004130, and by Boston University.

## References

- Andrew G Barto, Richard S Sutton, and Charles W Anderson. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE transactions on systems, man, and cybernetics , (5):834-846, 1983.
- Jalaj Bhandari, Daniel Russo, and Raghav Singal. A finite time analysis of temporal difference learning with linear function approximation. In Conference on learning theory , pp. 1691-1692. PMLR, 2018.
- Qi Cai, Zhuoran Yang, Jason D Lee, and Zhaoran Wang. Neural temporal-difference learning converges to global optima. Advances in Neural Information Processing Systems , 32, 2019.
- Semih Cayci, Niao He, and R Srikant. Finite-time analysis of entropy-regularized neural natural actor-critic algorithm. arXiv preprint arXiv:2206.00833 , 2022.
- Tianyi Chen, Yuejiao Sun, and Wotao Yin. Closing the gap: Tighter analysis of alternating stochastic gradient methods for bilevel problems. Advances in Neural Information Processing Systems , 34: 25294-25307, 2021.
- Xuyang Chen and Lin Zhao. Finite-time analysis of single-timescale actor-critic. arXiv preprint arXiv:2210.09921 , 2022.
- Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. Advances in neural information processing systems , 32, 2019.
- Philip G Drazin. Nonlinear systems . Number 10. Cambridge University Press, 1992.
- Amir-massoud Farahmand, Mohammad Ghavamzadeh, Csaba Szepesv´ ari, and Shie Mannor. Regularized policy iteration with nonparametric function spaces. The Journal of Machine Learning Research , 17(1):4809-4874, 2016.
- FR Gantmacher. The theory of matrices. New York , 1964.
- Arthur Jacot, Franck Gabriel, and Cl´ ement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems , 31, 2018.
- Vijay Konda and John Tsitsiklis. Actor-critic algorithms. Advances in neural information processing systems , 12, 1999.

- Harshat Kumar, Alec Koppel, and Alejandro Ribeiro. On the sample complexity of actor-critic method for reinforcement learning with function approximation. Machine Learning , pp. 1-35, 2023.
- Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature , 521(7553):436-444, 2015.
- Chaoyue Liu, Libin Zhu, and Misha Belkin. On the linearity of large non-linear models: when and why the tangent kernel is constant. Advances in Neural Information Processing Systems , 33: 15954-15964, 2020.
- Rui Liu and Alex Olshevsky. Temporal difference learning as gradient splitting. In International Conference on Machine Learning , pp. 6905-6913. PMLR, 2021.
- A Yu Mitrophanov. Sensitivity and convergence of uniformly ergodic markov chains. Journal of Applied Probability , 42(4):1003-1014, 2005.
- Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning , pp. 1928-1937. PMLR, 2016.
- Yann Ollivier. Approximate temporal difference learning is a gradient descent for reversible policies. arXiv preprint arXiv:1805.00869 , 2018.
- Alex Olshevsky and Bahman Gharesifard. A small gain analysis of single timescale actor critic. arXiv preprint arXiv:2203.02591 , 2022.
- David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature , 550(7676):354-359, 2017.
- Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction . MIT press, 2018.
- Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems , 12, 1999.
- Haoxing Tian, Ioannis Paschalidis, and Alex Olshevsky. On the performance of temporal difference learning with neural networks. In The Eleventh International Conference on Learning Representations , 2023.
- Lingxiao Wang, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural policy gradient methods: Global optimality and rates of convergence. arXiv preprint arXiv:1909.01150 , 2019.
- Yue Frank Wu, Weitong Zhang, Pan Xu, and Quanquan Gu. A finite-time analysis of two time-scale actor-critic methods. Advances in Neural Information Processing Systems , 33:17617-17628, 2020.
- Pan Xu and Quanquan Gu. A finite-time analysis of q-learning with neural network function approximation. In International Conference on Machine Learning , pp. 10555-10565. PMLR, 2020.
- Shaofeng Zou, Tengyu Xu, and Yingbin Liang. Finite-sample analysis for sarsa with linear function approximation. Advances in neural information processing systems , 32, 2019.
- Dan Zwillinger. CRC standard mathematical tables and formulas . chapman and hall/CRC, 2018.

## A Sketch of Proof

In this section we give a basic idea of how we prove Theorem 3.1. Briefly speaking, our idea contains two directions: First, the Critic error (captured by ∆ Q ) can be upper-bounded by the Actor error (captured by ∆ V ); Next, the Actor error can also be upper-bounded by the Critic error. Therefore, both errors are bounded and converge to 0 . Based on this idea, our proof can be divided into three steps:

Step 1: Analysis of Actor update.

In Appendix B, we first bound ∆ V by ∆ Q through Actor update. On one hand, by considering Actor update and comparing it with mean-path update (where we replace g by ¯ g ), one would have

<!-- formula-not-decoded -->

where D Q = ∇ ln π ( θ t ) T Φ θ t [ Q ( w t ) -Q ∗ θ t ] , F t = ( w t , θ t ) and U g is defined in Lemma C.13.

On the other hand, Lemma C.12 suggests V ∗ θ is smooth w.r.t. θ . Hence,

<!-- formula-not-decoded -->

Our claim is a combination of the above facts and some simple calculations:

<!-- formula-not-decoded -->

We successfully bound ∆ V by ∆ Q .

Step 2: Analysis of Critic update.

In Appendix C, we next bound ∆ Q by ∆ V through Critic update. Here we perform classical way of analysis, which begins with

<!-- formula-not-decoded -->

We treat the above three terms respectively. To address I 1 , by comparing with mean-path update:

<!-- formula-not-decoded -->

Now let us examine this equation carefully. I 1 , 1 is the inner product between w t -ˆ w ∗ θ t and the mean-path update ¯ f ( w t , θ t ) , which can be captured by gradient splitting; I 1 , 2 decays as α 2 t , so a loose bound on || f ( O t , w t ) || 2 is enough (See Lemma C.15); I 1 , 3 is Markov sampling noise, which is handled using the same procedure as in Bhandari et al. (2018).

To discuss more about how to address Markov sampling noise, the idea is to using Assumption 2.8 to show that, after τ mix steps, the distance between distribution of agent and the stationary distribution

decaying geometrically, and thus I 1 , 3 also decays geometrically. However, there is still a lot of difficulties to apply the same analysis in our work since TD(0) is considered in Bhandari et al. (2018) while Actor-Critic methods is considered here. The difficulties is induced by the constant changing of policy in every time steps during training. We introduce an auxiliary chain (See the definitions before Lemma C.10) to further address the changing of policy problem inspired by Zou et al. (2019); Wu et al. (2020); Chen &amp; Zhao (2022).

Now we move on to I 2 . we notice that the dominate term is E [ 2( ˆ w ∗ θ t -ˆ w ∗ θ t +1 ) T ( w t -ˆ w ∗ θ t ) ] since the remaining term E [ 2( ˆ w ∗ θ t -ˆ w ∗ θ t +1 ) T α w f ( O t , w t ) ] decays as α θ α w ( α θ comes from || ˆ w ∗ θ t -ˆ w ∗ θ t +1 || which can be seen using Assumption 2.6). To handle the dominate term, we first view ˆ w ∗ θ as a function of θ and use a second order expansion as follows. Then the problem get solved after noticing that we already derive relationships on θ t +1 -θ t in Step 1.

<!-- formula-not-decoded -->

To address I 3 , we notice that it decays as α w 2 as a direct result of Assumption 2.6.

Combine all of the above result we can finally arrive at the relationship between ∆ Q and ∆ V .

Step 3: Combine result from Step 1 and 2 by small-gain theorem.

Now we are ready to use the Small Gain theorem. We fit the results from Step 1 and Step 2 by the following form:

<!-- formula-not-decoded -->

Then, Small Gain theorem implies that y can be upper bounded by the following inequality:

<!-- formula-not-decoded -->

Once we have a bound for y , we can easily compute a bound for x .

## B Actor-Critic

In this section, we will review and clarify the AC algorithm being considered in this paper. Recall that in Eq.(8), we defined the set W as W = { w | || w -w 0 || ≤ σ w } and the TD error δ t as

<!-- formula-not-decoded -->

With δ t , we now define function f and g such that

<!-- formula-not-decoded -->

where we denote by O t = ( s t , a t , s ′ t , a ′ t ) ∈ S × A × S × A the tuple of s t , a t , s ′ t , a ′ t and by ˆ O t = (ˆ s t , ˆ a t ) ∈ S × A the ˆ s t , ˆ a t pair. The way of sampling O t and ˆ O t is mentioned in Section 2.4.

With these notations, the AC update mentioned in Section 2.4 can be written as

<!-- formula-not-decoded -->

We find it useful to talk about the 'mean path update'. This just means that the functions f ( · , · ) and g ( · , · , · ) in Eq.(15) are replaced by their means, assuming that ( s t , a t ) is sampled from µ θ t while (ˆ s t , ˆ a t ) is sampled from (1 -γ ) ϕ θ t . More formally, the mean-path update functions ¯ f ( · , · ) and ¯ g ( · , · ) are defined as

<!-- formula-not-decoded -->

where F t = ( w t , θ t ) and E O t , E ˆ O t assume O t follows µ θ t and ˆ O t follows (1 -γ ) ϕ θ t . To show the latter one, as we discussed in Section 2.4, we first sample T such that P ( T = t ) = (1 -γ ) γ t . We then perform T transition starting from s 0 . This mean that by total probability,

<!-- formula-not-decoded -->

Thus, if the policy here is given by θ t , it follows immediately that

<!-- formula-not-decoded -->

Notice that under these notations, we have E ˆ O t [ g ( ˆ O t , w t , θ t )] = E [ g ( ˆ O t , w t , θ t ) |F t ] .

Algorithm 1 details the algorithm considered in this paper.

## Algorithm 1 Actor-Critic

Require: Numbers of iterations T , learning rate α w and α θ , projection set W .

Initialize θ 0 , b r and w ( k ) such that | b r | ≤ 1 , ∀ r and every entry of w ( k ) is chosen from N (0 , 1) . Initialize the starting state-action pair s 0 , a 0 .

<!-- formula-not-decoded -->

Sample ˆ O t by first sampling a random variable T with P ( T = t ) = (1 -γ ) γ t , and second obtaining T transitions by starting at s 0 and taking actions according to π ( a | s, θ t ) .

Compute δ t , f ( O t , w t ) , g ( ˆ O t , w t , θ t ) , and update w t +1 and θ t +1 as

<!-- formula-not-decoded -->

end for

## C Auxiliary Lemmas

In this section, we will present all the auxiliary lemmas needed to prove Theorem 3.1.

## C.1 Properties of the Neural Network

In this section, we will show that the neural network has Lipschitzness and smoothness properties. The following result is based on Liu et al. (2020) and has been talked about in Tian et al. (2023).

LemmaC.1. For any ( s, a ) ∈ S × A , there exists scalars L Q ( s, a ) , H Q ( s, a ) such that for w 1 , w 2 ∈ W ,

<!-- formula-not-decoded -->

If we further define

<!-- formula-not-decoded -->

then L Q = O (1) and H Q = ˜ O ( 1 √ m ) with respect to m .

Proof. The Lipschitzness property is proved in Tian et al. (2023) while the smoothness property is a direct result of Liu et al. (2020).

## C.2 Properties of the Operator N

In this section, we will show several results about the operator N θ defined in Eq.(12).

Lemma C.2. For any function f defined on S × A ,

<!-- formula-not-decoded -->

Proof. The proof is given by Lemma A.1 in Tian et al. (2023).

Lemma C.3. There exists λ min &gt; 0 and λ ′ min &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and where λ min = (1 -γ ) µ min λ ′ 2 and λ ′ min is given by λ ′ min = (1 -γ ) µ min λ ′ 2 L 2 Q .

Proof. To show the first part,

<!-- formula-not-decoded -->

where the first line is the definition of N ( · ) while the last line uses Assumption 2.9. We can set λ min = (1 -γ ) µ min λ ′ 2 and we finish the proof for the first part.

The second part is an obvious result that simply combines the first part and Lemma C.1.

Lemma C.4. Suppose D Q = ∇ ln π ( θ t ) T Φ θ t [ Q ( w t ) -Q ∗ θ t ] . The relationship between D Q and N θ t ( Q ( w t ) -ˆ Q ∗ θ t ) can be described as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. One can easily show that

<!-- formula-not-decoded -->

As assumed in Assumption 2.4, ||∇ ln π ( a | s, θ t ) || ≤ L ′ π . Hence,

<!-- formula-not-decoded -->

Using the facts that ( E [ X ]) 2 ≤ E [ X 2 ] ,

<!-- formula-not-decoded -->

On the other hand,

<!-- formula-not-decoded -->

where we use Assumption 2.5 which tells us | ˆ Q ∗ θ t ( s, a ) -Q ∗ θ t ( s, a ) | ≤ ϵ . Hence,

<!-- formula-not-decoded -->

Combine with Lemma C.3 and the fact that (1 -γ ) ϕ θ t ( s, a ) ≤ 1 ,

<!-- formula-not-decoded -->

This finishes the proof.

## C.3 Mean-value Theorem and Extensions

Lemma C.5. These following lemmas generalize the mean-value theorem to higher dimensional input and output cases.

(a) Let h : R → R be any differentiable function. For any x, y ∈ R , there exists λ ∈ (0 , 1) and z = λx +(1 -λ ) y such that

<!-- formula-not-decoded -->

(b) Let ξ : R a → R be any differentiable function. For any x, y ∈ R a , there exists λ ∈ (0 , 1) and z = λx +(1 -λ ) y such that

<!-- formula-not-decoded -->

(c) Let f : R a → R b be any differentiable function and e ∈ R b be any vector. For any x, y ∈ R a , there exists λ ∈ (0 , 1) and z = λx +(1 -λ ) y such that

<!-- formula-not-decoded -->

where f ′ ( z ) is the Jacobian at z .

Proof. This proof is given by Lemma A.2 in Tian et al. (2023).

## C.4 The Mixing of Two Markov Chains

In this section, we argue that if two Markov chains satisfy Assumption 2.8, then the difference between their distributions could be very small. This is inspired by and follows the same logic as Chen &amp; Zhao (2022); Zou et al. (2019). Before that, we will first introduce the total variation norm for vectors and matrices, which can be used to measure a difference between distributions.

Denote f : X → R to be any real value function. We can define the total variation norm of f , denoted by || f || TV , as

<!-- formula-not-decoded -->

For matrix A , we can also define || A || TV to be

<!-- formula-not-decoded -->

If f is some probability measure, since f ( x ) ∈ [0 , 1] , it is easy to conclude that || f || TV = 1 . Likewise, if A is a Markov transition matrix, we can show that || A || TV = 1 .

The following lemma establishes the relationship between the total variation norm with the more familiar 1 -norm and ∞ -norm.

Lemma C.6. The following statements are true:

- a. For any vector f , || f || TV = || f || 1 .
- b. For any matrix A , || A || TV = || A || ∞ .

Proof. The lemma is obvious so we omit the proof here.

Based on Assumption 2.8, we have the following result:

Lemma C.7. If the Markov chain has transition probability matrix A , then we have

<!-- formula-not-decoded -->

Further, if the Markov chain satisfies Assumption 2.8, then we have

<!-- formula-not-decoded -->

Proof. The first part of this lemma is obvious because A is a stochastic (Markov) matrix and by Lemma C.6, || · || TV is just the same as || · || ∞ .

For the second part, by the definition of total variation norm,

<!-- formula-not-decoded -->

where e i means the all-zero vector except a 1 at the i 'th entry. By Assumption 2.8,

<!-- formula-not-decoded -->

The following theorem, which is inspired by Theorem 3.1 in Mitrophanov (2005), is very important in many analyses of AC that take Markov sampling into consideration (i.e., Wu et al. (2020); Chen &amp;Zhao (2022)). However, since our settings are slightly different, we provide our own version.

Lemma C.8. Suppose we have the following two Markov Chains which of both satisfy Assumption 2.8,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where p A i , p B i stand for the probability at step i under transition matrix A,B , respectively. The following inequality holds:

<!-- formula-not-decoded -->

Proof. First, we prove that

<!-- formula-not-decoded -->

by induction. If t = 1 , by definition we know

<!-- formula-not-decoded -->

where the third line is because we assume the result holds for the t = k case. Now, we can take the total variation norm on both sides:

<!-- formula-not-decoded -->

where the third line utilizes Lemma C.6 and Lemma C.7.

With Lemma C.8, we can derive many useful results. The following result is similar to Lemma 3 in Zou et al. (2019), Lemma B.1 in Wu et al. (2020), and Lemma B.4 in Chen &amp; Zhao (2022), which shows that both of µ ′ θ , the stationary distribution over states, and µ θ , the stationary distribution over state-action pairs, are Lipschitz with respect to θ .

Lemma C.9. The following statements hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where n a is the number of actions. In other words, n a = | A | .

Proof. Recall the state transition matrix P ′ θ is defined by P θ ( s ′ | s ) = ∑ a P env ( s ′ | s, a ) π ( a | s, θ ) and corresponding stationary distribution µ ′ θ T = µ ′ θ T P ′ θ . Use Lemma C.9, we know that

<!-- formula-not-decoded -->

Notice that

<!-- formula-not-decoded -->

where we use Assumption 2.4 in the second line. Hence,

<!-- formula-not-decoded -->

Taking lim t →∞ on both sides, we derive

<!-- formula-not-decoded -->

This finishes the proof of the first part.

Now, Eq.(1) implies

<!-- formula-not-decoded -->

In order to show the following lemmas, we need to introduce the following auxiliary Markov chain:

<!-- formula-not-decoded -->

For reference, the original Markov chain around t is

<!-- formula-not-decoded -->

For the consistency of notations, we will denote O τ = ( s τ , a τ , s ′ τ , a ′ τ ) and ˜ O τ = (˜ s τ , ˜ a τ , ˜ s ′ τ , ˜ a ′ τ ) where in this case we have s ′ τ = s τ +1 , ˜ s ′ τ = ˜ s τ +1 and a ′ τ ∼ π ( a | s, θ τ ) , ˜ a ′ τ ∼ π ( a | s, θ t -τ mix ) . This kind of notations will immediately implies P ( ˜ O t -τ mix -1 ∈ · ) = P ( O t -τ mix -1 ∈ · ) .

The following lemma claims that the distribution difference between the two Markov chains above will be very small.

Lemma C.10. The following statements are true:

- a. For any possible τ ∈ { t -τ mix , t -τ mix +1 , . . . , t }

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- b. For any possible τ ∈ { t -τ mix , t -τ mix +1 , . . . , t }

<!-- formula-not-decoded -->

- c. Consider P ( O t ∈ · ) and P ( ˜ O t ∈ · ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. For the first part,

<!-- formula-not-decoded -->

For the second part, conditioned on θ and s , we denote

<!-- formula-not-decoded -->

which will be useful later. Using notations from the original Markov Chain,

<!-- formula-not-decoded -->

Similarly, the auxiliary Markov Chain gives us

<!-- formula-not-decoded -->

We now rephrase the left-hand side term we want to prove,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

,

For I 2 ,

<!-- formula-not-decoded -->

For I 3 ,

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

which finishes the proof for part b .

It is easy to check that part a implies the following:

<!-- formula-not-decoded -->

The above fact, along with the result from part b , tells us the following:

<!-- formula-not-decoded -->

Repeat the inequality above over t to t -τ mix we have

<!-- formula-not-decoded -->

## C.5 Smoothness of the State-Value Function

The following two lemmas show that the state value function is actually smooth with respect to θ . The idea here is inspired by Olshevsky &amp; Gharesifard (2022). The first lemma requires the following basic identity from Zwillinger (2018): for matrix A θ ,

<!-- formula-not-decoded -->

Lemma C.11. For two vectors u, v ∈ R n whose entries are bounded. Suppose

<!-- formula-not-decoded -->

Then, there exists a constant L q such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, there exists a constant L ′ q such that

<!-- formula-not-decoded -->

Similarly, if

The following lemma, which is our goal in this section, claims that V ∗ θ is smooth with respect to θ .

Lemma C.12. V ∗ θ t is H V -smoothness with respect to θ t .

Proof. Using Theorem 2.1, we obtain the following result:

<!-- formula-not-decoded -->

We now show that all I 1 , I 2 , I 3 can be bounded by a multiple of || θ 1 -θ 2 || .

For I 1 , by Assumption 2.4 we know that ∇ ln π ( a | s, θ ) is Lipschitz, which, together with ϕ θ ( s, a ) ≤ 1 1 -γ and Q ∗ θ ( s, a ) ≤ r max 1 -γ , implies that I 1 can be upper bounded by a multiple of || θ 1 -θ 2 || .

For I 2 , since Q ∗ θ satisfies Bellman equation, we can write Q ∗ θ using matrix multiplication, which is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where e s,a has only one non-zero entry of one corresponding to the pair ( s, a ) . Hence, Q ∗ θ ( s, a ) is Lipschitz with respect to θ .

For I 3 , by definition,

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where e s has only one non-zero entry of one corresponding to s . Again, by Lemma C.11, ϕ θ ( s, a ) is Lipschitz with respect to θ .

By Lemma C.11, this implies

## C.6 Properties of the Actor Update

The following lemma shows that the incremental in actor update is bounded.

Lemma C.13. For g ( ˆ O t , w t , θ t ) and ¯ g ( w t , θ t ) , we have the following properties:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Recall that in Eq.(15), g ( ˆ O t , w t , θ t ) is defined to be

<!-- formula-not-decoded -->

To bound || g ( ˆ O t , w t , θ t ) || tions

, by Assumption 2.4 and Lemma C.1, we can do the following manipula-

<!-- formula-not-decoded -->

Because Eq.(16) implies that ¯ g ( w t , θ t ) is some expectation of g ( ˆ O t , w t , θ t ) with a coefficient 1 1 -γ , the second part of this lemma is a direct result of the first part in Lemma C.13.

## C.7 Properties of the Critic Update

In this section, we will introduce some properties that we find useful in analyzing critic update.

Notice that, as defined in Eq.(8), δ t actually depends on O t and w t . Here we make this dependency explicitly and thus write δ t as δ t = δ ( O t , w t ) . In this following lemma, we explore some properties of δ t .

Lemma C.14. For δ t , we have the following two results:

a. δ ( O t , w ) is L δ -Lipschitz with respect to w ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

b. | δ ( O t , w ) | is upper bounded by U δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Using Eq.(8),

<!-- formula-not-decoded -->

where the last line uses Lemma C.1. On the other hand,

<!-- formula-not-decoded -->

where the second equation uses Assumption 2.1, Lemma C.1, Eq.(9) and Eq.(5).

With the above properties of δ t , now we can further consider f , ¯ f , and F defined in Lemma C.17. Lemma C.15. For f ( O t , w ) defined in Eq.(15), we have the following two results:

a. f ( O t , w ) is L f -Lipschitz with respect to w ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

b. || f ( O t , w ) || can be upper bounded by U f ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By definition,

Hence,

<!-- formula-not-decoded -->

For I 1 , by Lemma C.14 and Lemma C.1, we perform the following manipulations:

<!-- formula-not-decoded -->

For I 2 , by Lemma C.14 and Lemma C.1, we derive

<!-- formula-not-decoded -->

Combining I 1 and I 2 we prove the first part of this lemma.

For the second part, by Lemma C.14 and Lemma C.1,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma C.16. For ¯ f ( w,θ ) defined in Eq.(16), we have the following results:

a. || ¯ f ( w,θ ) || can be upper bounded by U f ,

<!-- formula-not-decoded -->

where U f = L Q U δ .

b. ¯ f ( w,θ ) is L f -Lipschitz with respect to w ,

<!-- formula-not-decoded -->

where L f = (1 + γ ) L 2 Q + H Q U δ .

c. ¯ f ( w,θ ) is L ¯ f -Lipschitz with respect to θ ,

<!-- formula-not-decoded -->

where L ¯ f = L Q U δ (2 + τ mix + C 1 1 -β ) n a L π .

Proof. Because of Eq.(16), the first and second part of this lemma is a direct result of Lemma C.15. For the third part,

<!-- formula-not-decoded -->

where the last line is by Lemma C.15. Further notice that

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

where the last line is by Lemma C.9. This implies

<!-- formula-not-decoded -->

Lemma C.17. Denote F ( O,w,θ ) = ( w -ˆ w ∗ θ ) T [ f ( O,w ) -¯ f ( w,θ ) ] . The following results hold:

a.

F

(

O,w,θ

where

)

is

L

F

θ

-Lipschitz with respect to

θ

,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- b. F ( O,w,θ ) is L F w -Lipschitz with respect to w ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- c. Conditioned on θ t -τ mix and s t -τ mix +1 ,

<!-- formula-not-decoded -->

- d. Conditioned on θ t -τ mix and s t -τ mix +1 ,

<!-- formula-not-decoded -->

Proof. First, we observe that

<!-- formula-not-decoded -->

where

For I 1 , we have

<!-- formula-not-decoded -->

which is by Lemma C.15 and Assumption 2.6.

For I 2 , we have

<!-- formula-not-decoded -->

which is by Lemma C.16. Combining the above two facts we end the proof of the first part. For the second part,

<!-- formula-not-decoded -->

where the last line is by Lemma C.15 and Lemma C.16.

For the third part, conditioned on θ t -τ mix and s t -τ mix +1 ,

<!-- formula-not-decoded -->

where we use Lemma C.15 and Lemma C.10.

For the fourth part, we first denote O + = ( s + , a + , s + ′ , a + ′ ) such that ( s + , a + ) ∼ µ θ t -τ mix , s + ′ ∼ P env ( s ′ | s, a ) and a + ′ ∼ π ( a ′ | s ′ , θ t -τ mix ) . Under this definition, we have

<!-- formula-not-decoded -->

By Assumption 2.8, we have

<!-- formula-not-decoded -->

Hence, conditioned on θ t -τ mix and s t -τ mix +1 ,

<!-- formula-not-decoded -->

where the last line is because of Lemma C.15. However,

<!-- formula-not-decoded -->

Hence, conditioned on θ t -τ mix and s t -τ mix +1 ,

<!-- formula-not-decoded -->

The following lemma, which reveals a useful property for critic update, is inspired by Olshevsky &amp; Gharesifard (2022).

Lemma C.18. For critic update, we have the following two results:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Proof. If we use x ( i ) to denote the i 'th entry of vector x ,

<!-- formula-not-decoded -->

We can view ˆ w ∗ θ as a function of θ . Using a second order expansion,

<!-- formula-not-decoded -->

If we take expectation conditioned on F t ,

<!-- formula-not-decoded -->

This leads to

<!-- formula-not-decoded -->

where in the last line we use Assumption 2.6 and Lemma C.13. Now we go back to what we really care about,

<!-- formula-not-decoded -->

where the third line is because the second term is constant conditioned on F t .

For the second part, notice that we already have the following result

<!-- formula-not-decoded -->

where we simply bound || ¯ g ( w t , θ t ) || by U g (this result is from Lemma C.13). On the other hand, by Lemma C.15 and Lemma C.16, a rough bound for f ( O t , w t ) -¯ f ( w t , θ t ) would be simply

<!-- formula-not-decoded -->

Now we go back to what we really care about,

<!-- formula-not-decoded -->

where the third line is because the two terms are independent when conditioned on F t . The second part of this lemma is proved after we take expectation on both sides.

## D Actor Update Analysis

Lemma D.1.

<!-- formula-not-decoded -->

Proof. Actor update says the following:

<!-- formula-not-decoded -->

By the definition of ¯ g ( w t , θ t ) in Eq.(16),

<!-- formula-not-decoded -->

where we use the fact in Eq.(7) .

For simplicity, denote D Q = ∇ ln π ( θ t ) T Φ θ t [ Q ( w t ) -Q ∗ θ t ] . So far we have the following result:

<!-- formula-not-decoded -->

On one hand, if we take expectation (conditioned on F t ) on both sides, we get

<!-- formula-not-decoded -->

where we use the fact in Eq.(16) that E [ 1 1 -γ g ( ˆ O t , w t , θ t ) ] = ¯ g ( w t , θ t ) . On the other hand,

<!-- formula-not-decoded -->

where we keep using E [ 1 1 -γ g ( ˆ O t , w t , θ t ) ] = ¯ g ( w t , θ t ) . Using Lemma C.12,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking expectation (conditioned on F t ) on both sides and we obtain

<!-- formula-not-decoded -->

Plug in the facts about θ t +1 -θ t , we know

<!-- formula-not-decoded -->

We can use the facts that 2 ab ≤ a 2 + b 2 and ( a + b ) 2 ≤ 2 a 2 +2 b 2 to obtain

<!-- formula-not-decoded -->

where the last inequality we uses the fact from LemmaC.4. We can rewrite it as

<!-- formula-not-decoded -->

Taking expectation on both sides and telescoping sum:

<!-- formula-not-decoded -->

If we use notations from Eq.(13), the above fact can be rewritten as

<!-- formula-not-decoded -->

## E Critic Update Analysis under the Markov Sampling Case

Lemma E.1. In the Markov sampling case,

<!-- formula-not-decoded -->

where, for simplicity, we denote

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Proof. Recall the critic update is

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

We can take expectation on both sides,

<!-- formula-not-decoded -->

To analyze I 1 , we derive

<!-- formula-not-decoded -->

To analyze I 1 , 1 , we perform the following of manipulations:

<!-- formula-not-decoded -->

where the second line is by the definition in Eq.(16), the sixth line is by Lemma (C.2) and Lemma C.5, and the eighth line is by Lemma C.1. Here, λ ∈ [0 , 1] is some scalar and w mid = λw t +(1 -λ ) ˆ w ∗ θ t . So we arrive at the final bound for I 1 , 1 :

<!-- formula-not-decoded -->

To analyze I 1 , 2 , we conclude

<!-- formula-not-decoded -->

To analyze I 1 , 3 , for simplicity, we denote F ( O t , w, θ ) = ( w -ˆ w ∗ θ ) T [ f ( O t , w ) -¯ f ( w,θ ) ] . We have

<!-- formula-not-decoded -->

For J 1 , we have

For J 2 , we have

<!-- formula-not-decoded -->

For J 3 , by Lemma C.17, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which means

<!-- formula-not-decoded -->

For J 4 , by Lemma C.17, we have

<!-- formula-not-decoded -->

which means

<!-- formula-not-decoded -->

Hence, for I 1 , 3 , we have

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Hence, for I 1 ,

<!-- formula-not-decoded -->

where, for simplicity, we denote

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

To analyze I 2 , we derive

<!-- formula-not-decoded -->

To analyze I 2 , 1 first, by lemma C.18, we already know

<!-- formula-not-decoded -->

For I 2 , 1 , 2 , a rough bound would be

<!-- formula-not-decoded -->

Notice that this term is very similar to what we have in Eq.(18). The two difference are (a) all the f in Eq.(18) are replaced by ¯ f and (b) expectation is removed. In this case, I 1 , 3 will be 0 and a similar bound for I 1 , 1 and I 1 , 2 will also hold. This implies the following result:

<!-- formula-not-decoded -->

where the last line uses Lemma C.3. For I 2 , 1 , 1 , recall that

<!-- formula-not-decoded -->

Hence, we have the following bound for I 2 , 1 , 1 :

<!-- formula-not-decoded -->

Hence, we have the following result:

<!-- formula-not-decoded -->

After taking expectation on both sides, we know

<!-- formula-not-decoded -->

which is the bound for I 2 , 1 .

For I 2 , 2 , by Lemma C.18, we have

<!-- formula-not-decoded -->

This ends the bound for I 2 , which, after combining the bound for I 2 , 1 and I 2 , 2 , will be

<!-- formula-not-decoded -->

To analyze I 3 , we have

<!-- formula-not-decoded -->

Recall that

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

where we use the fact that E [ ¯ g ( w t , θ t ) -1 1 -γ g ( ˆ O t , w t , θ t ) |F t ] = 0 . Hence,

<!-- formula-not-decoded -->

After taking expectation on both side, we will arrive at the bound for I 3 , which is

<!-- formula-not-decoded -->

Now, go back to Eq.(17) and we obtain

<!-- formula-not-decoded -->

Now, we can do a telescoping sum for i to T :

<!-- formula-not-decoded -->

which, if we adopt notations from Eq.(13), can be rewritten as

<!-- formula-not-decoded -->

## F Critic Update Analysis under the i.i.d. Sampling Case

Lemma F.1. In the i.i.d. sampling case,

<!-- formula-not-decoded -->

Proof. The i.i.d. assumption implies that f ( O t , w t ) is replaced by ¯ f ( w t , θ t ) . This will bring a change in both the analysis for I 1 and I 2 .

First, we will figure out how the i.i.d. sampling effect I 1 . Now we know that I 1 , 3 in Eq.(18) is 0 . That means, for I 1 in Eq.(17),

<!-- formula-not-decoded -->

where C w, 1 is defined the same as before:

<!-- formula-not-decoded -->

Next, after a removal of I 2 , 2 term ( I 2 , 2 will just be 0 if we replace f by ¯ f ), we can derive the new bound for I 2 , which is

<!-- formula-not-decoded -->

Based on the new bounds for I 1 and I 2 , the critic update now gives the following:

<!-- formula-not-decoded -->

## G Small Gain Theorem and Small Gain Analysis

## G.1 Small Gain Theorem

Now we introduce the small gain theorem.

Lemma G.1. Suppose x and y satisfy the following two inequalities:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where all coefficients are non-negative. Then, y can be upper bounded by the following inequality:

<!-- formula-not-decoded -->

Proof. Proof of this lemma can be found in Olshevsky &amp; Gharesifard (2022).

## G.2 Small Gain Analysis under i.i.d. Sampling

Now recall the result from Actor analysis is

<!-- formula-not-decoded -->

and the one from Critic analysis is

<!-- formula-not-decoded -->

What we really care about is the relationship between T, ϵ and m . So from now on, we will use O ( · ) and ˜ O ( · ) ( ˜ O ( · ) hides the potential logarithm factor of m ) notations and only consider these variables. First, observe the following dependency on T, m and ϵ :

<!-- formula-not-decoded -->

If we choose α w = α θ = 1 √ T and given that all other coefficients are independent with ϵ , T and m , we conclude

<!-- formula-not-decoded -->

We can set

<!-- formula-not-decoded -->

Now we can apply Small Gain Theorem, where we conclude

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## G.3 Small Gain Analysis in the Markov Sampling Case

Now recall the result from Actor analysis is

<!-- formula-not-decoded -->

and

and the one from Critic analysis is

<!-- formula-not-decoded -->

If we choose α w = α θ = 1 √ T and given that all other coefficients are independent with ϵ , T and m , we conclude and

<!-- formula-not-decoded -->

We can set

<!-- formula-not-decoded -->

Now we can apply Small Gain Theorem, where we conclude and

<!-- formula-not-decoded -->