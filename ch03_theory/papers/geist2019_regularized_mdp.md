## A Theory of Regularized Markov Decision Processes

Matthieu Geist 1 Bruno Scherrer 2 Olivier Pietquin 1

## Abstract

Many recent successful (deep) reinforcement learning algorithms make use of regularization, generally based on entropy or Kullback-Leibler divergence. We propose a general theory of regularized Markov Decision Processes that generalizes these approaches in two directions: we consider a larger class of regularizers, and we consider the general modified policy iteration approach, encompassing both policy iteration and value iteration. The core building blocks of this theory are a notion of regularized Bellman operator and the Legendre-Fenchel transform, a classical tool of convex optimization. This approach allows for error propagation analyses of general algorithmic schemes of which (possibly variants of) classical algorithms such as Trust Region Policy Optimization, Soft Q-learning, Stochastic Actor Critic or Dynamic Policy Programming are special cases. This also draws connections to proximal convex optimization, especially to Mirror Descent.

## 1. Introduction

Many reinforcement learning algorithms make use of some kind of entropy regularization, with various motivations, such as improved exploration and robustness. Trust Region Policy Optimization (TRPO) (Schulman et al., 2015) is a policy iteration scheme where the greedy step is penalized with a Kullback-Leibler (KL) penalty between two consecutive policies. Dynamic Policy Programming (DPP) (Azar et al., 2012) is a reparametrization of a value iteration scheme regularized by a KL penalty between consecutive policies. Soft Q-learning, eg. (Fox et al., 2016; Schulman et al., 2017; Haarnoja et al., 2017), uses a Shannon entropy regularization in a value iteration scheme, while Soft Actor Critic (SAC) (Haarnoja et al., 2018a) uses it in a policy iteration scheme. Value iteration has also been combined with a

1 Google Research, Brain Team. 2 Universit´ e de Lorraine, CNRS, Inria, IECL, F-54000 Nancy, France. Correspondence to: Matthieu Geist &lt; mfgeist@google.com &gt; .

Proceedings of the 36 th International Conference on Machine Learning , Long Beach, California, PMLR 97, 2019. Copyright 2019 by the author(s).

Tsallis entropy (Lee et al., 2018), with the motivation of having a sparse regularized greedy policy. Other approaches are based on a notion of temporal consistency equation, somehow extending the notion of Bellman residual to the regularized case (Nachum et al., 2017; Dai et al., 2018; Nachum et al., 2018), or on policy gradient (Williams, 1992; Mnih et al., 2016).

This non-exhaustive set of algorithms share the idea of using regularization, but they are derived from sometimes different principles, consider each time a specific regularization, and have ad-hoc analysis, if any. Here, we propose a general theory of regularized Markov Decision Processes (MDPs). To do so, a key observation is that (approximate) dynamic programming, or (A)DP, can be derived solely from the core definition of the Bellman evaluation operator. The framework we propose is built upon a regularized Bellman operator, and on an associated Legendre-Fenchel transform. We study the theoretical properties of these regularized MDPs and of the related regularized ADP schemes. This generalizes many existing theoretical results and provides new ones. Notably, it allows for an error propagation analysis for many of the aforementioned algorithms. This framework also draws connections to convex optimization, especially to Mirror Descent (MD).

A unified view of entropy-regularized MDPs has already been proposed by Neu et al. (2017). They focus on regularized DP through linear programming for the average reward case. Our contribution is complementary to this work (different MDP setting, we do not regularize the same quantity, we do not consider the same DP approach). Our use of the Legendre-Fenchel transform is inspired by Mensch &amp; Blondel (2018), who consider smoothed finite horizon DP in directed acyclic graphs. Our contribution is also complementary to this work, that does not allow recovering aforementioned algorithms nor analyzing them. After a brief background, we introduce regularized MDPs and various related algorithmic schemes based on approximate modified policy iteration (Scherrer et al., 2015), as well as their analysis. All proofs are provided in the appendix.

## 2. Background

In this section, we provide the necessary background for building the proposed regularized MDPs. We write ∆ X the

set of probability distributions over a finite set X and Y X the set of applications from X to the set Y . All vectors are column vectors, except distributions, for left multiplication. We write 〈· , ·〉 the dot product and ‖ · ‖ p the glyph[lscript] p -norm.

## 2.1. Unregularized MDPs

An MDP is a tuple {S , A , P, r, γ } with S the finite 1 state space, A the finite action space, P ∈ ∆ S×A S the Markovian transition kernel ( P ( s ′ | s, a ) denotes the probability of transiting to s ′ when action a is applied in state s ), r ∈ R S×A the reward function and γ ∈ (0 , 1) the discount factor.

A policy π ∈ ∆ S A associates to each state a distribution over actions. The associated Bellman operator is defined as, for any function v ∈ R S ,

<!-- formula-not-decoded -->

This operator is a γ -contraction in supremum norm and its unique fixed-point is the value function v π . With r π ( s ) = E a ∼ π ( . | s ) [ r ( s, a )] and P π ( s ′ | s ) = E a ∼ π ( . | s ) [ P ( s ′ | s, a )]) , the operator can be written as T π v = r π + γP π v . For any function v ∈ R S , we associate the function q ∈ R S×A ,

<!-- formula-not-decoded -->

Thus, the Bellman operator can also be written as [ T π v ]( s ) = 〈 π ( ·| s ) , q ( s, · ) 〉 = 〈 π s , q s 〉 . With a slight abuse of notation, we will write T π v = 〈 π, q 〉 = ( 〈 π s , q s 〉 ) s ∈S .

From this evaluation operator, one can define the Bellman optimality operator as, for any v ∈ R S ,

<!-- formula-not-decoded -->

This operator is also a γ -contraction in supremum norm, and its fixed point is the optimal value function v ∗ . From the same operator, one can also define the notion of a policy being greedy respectively to a function v ∈ R S :

<!-- formula-not-decoded -->

Given this, we could derive value iteration, policy iteration, modified policy iteration, and so on. Basically, we can do all these things from the core definition of the Bellman evaluation operator. We'll do so from a notion of regularized Bellman evaluation operator.

## 2.2. Legendre-Fenchel transform

Let Ω : ∆ A → R be a strongly convex function. The Legendre-Fenchel transform (or convex conjugate) of Ω is Ω ∗ : R A → R , defined as

<!-- formula-not-decoded -->

1 We assume a finite space for simplicity of exposition, our results extend to more general cases.

We'll make use of the following properties (Hiriart-Urruty &amp;Lemar´ echal, 2012; Mensch &amp; Blondel, 2018).

Proposition 1. Let Ω be strongly convex, we have the following properties.

- i Unique maximizing argument: ∇ Ω ∗ is Lipschitz and satisfies ∇ Ω ∗ ( q s ) = argmax π s ∈ ∆ A 〈 π s , q s 〉 -Ω( π s ) .
- ii Boundedness: if there are constants L Ω and U Ω such that for all π s ∈ ∆ A , we have L Ω ≤ Ω( π s ) ≤ U Ω , then max a ∈A q s ( a ) -U Ω ≤ Ω ∗ ( q s ) ≤ max a ∈A q s ( a ) -L Ω .
- iii Distributivity: for any c ∈ R (and 1 the vector of ones), we have Ω ∗ ( q s + c 1 ) = Ω ∗ ( q s ) + c .

<!-- formula-not-decoded -->

A classical example is the negative entropy Ω( π s ) = ∑ a π s ( a ) ln π s ( a ) . Its convex conjugate is the smoothed maximum Ω ∗ ( q s ) = ln ∑ a exp q s ( a ) and the unique maximizing argument is the usual softmax ∇ Ω ∗ ( q s ) = exp q s ( a ) ∑ b exp q s ( b ) . For a positive regularizer, one can consider Ω( π s ) = ∑ a π s ( a ) ln π s ( a ) + ln |A| , that is the KL divergence between π s and a uniform distribution. Its convex conjugate is Ω ∗ ( q s ) = ln ∑ a 1 |A| exp q s ( a ) , that is the Mellowmax operator (Asadi &amp; Littman, 2017). The maximizing argument is still the softmax. Another less usual example is the negative Tsallis entropy (Lee et al., 2018), Ω( π s ) = 1 2 ( ‖ π s ‖ 2 2 -1) . The analytic convex conjugate is more involved, but it leads to the sparsemax as the maximizing argument (Martins &amp; Astudillo, 2016).

## 3. Regularized MDPs

The core idea of our contribution is to regularize the Bellman evaluation operator. Recall that [ T π v ]( s ) = 〈 π s , q s 〉 . A natural idea is to replace it by [ T π, Ω v ]( s ) = 〈 π s , q s 〉 -Ω( π s ) . To get the related optimality operator, one has to perform state-wise maximization over π s ∈ ∆ A , which gives the Legendre-Fenchel transform of [ T π, Ω v ]( s ) . This defines a smoothed maximum (Nesterov, 2005). The related maximizing argument defines the notion of greedy policy.

## 3.1. Regularized Bellman operators

We now define formally these regularized Bellman operators. With a slight abuse of notation, we write Ω( π ) = (Ω( π s )) s ∈S (and similarly for Ω ∗ and ∇ Ω ∗ ).

Definition 1 (Regularized Bellman operators) . Let Ω : ∆ A → R be a strongly convex function. For any v ∈ R S define q ∈ R S×A as q ( s, a ) = r ( s, a ) + γ E s ′ | s,a [ v ( s ′ )] . The regularized Bellman evaluation operator is defined as

<!-- formula-not-decoded -->

that is, state-wise, [ T π, Ω v ]( s ) = 〈 π s , q s 〉 -Ω( π s ) . The regularized Bellman optimality operator is defined as

<!-- formula-not-decoded -->

that is, state-wise, [ T ∗ , Ω v ]( s ) = Ω ∗ ( q s ) . For any function v ∈ R S , the associated unique greedy policy is defined as

<!-- formula-not-decoded -->

that is, state-wise, π ′ = ∇ Ω ∗ ( q s )

<!-- formula-not-decoded -->

To be really useful, these operators should satisfy the same properties as the classical ones. It is indeed the case (we recall that all proofs are provided in the appendix).

Proposition 2. The operator T π, Ω is affine and we have the following properties.

- i Monotonicity: let v 1 , v 2 ∈ R S such that v 1 ≥ v 2 . Then,

<!-- formula-not-decoded -->

- ii Distributivity: for any c ∈ R , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- iii Contraction: both operators are γ -contractions in supremum norm. For any v 1 , v 2 ∈ R S ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 3.2. Regularized value functions

The regularized operators being contractions, we can define regularized value functions as their unique fixed-points. Notice that from the following definitions, we could also easily derive regularized Bellman operators on q -functions.

Definition 2 (Regularized value function of policy π ) . Noted v π, Ω , it is defined as the unique fixed point of the operator T π, Ω : v π, Ω = T π, Ω v π, Ω . We also define the associated state-action value function q π, Ω as

<!-- formula-not-decoded -->

Thus, the regularized value function is simply the unregularized value of π for the reward r π -Ω( π ) , that is v π, Ω = ( I -γP π ) -1 ( r π -Ω( π )) .

Definition 3 (Regularized optimal value function) . Noted v ∗ , Ω , it is the unique fixed point of the operator T ∗ , Ω : v ∗ , Ω = T ∗ , Ω v ∗ , Ω . We also define the associated state-action value function q ∗ , Ω ( s, a ) as

<!-- formula-not-decoded -->

The function v ∗ , Ω is indeed the optimal value function, thanks to the following result.

Theorem 1 (Optimal regularized policy) . The policy π ∗ , Ω = G Ω ( v ∗ , Ω ) is the unique optimal regularized policy, in the sense that for all π ∈ ∆ S A , v π ∗ , Ω , Ω = v ∗ , Ω ≥ v π, Ω .

When regularizing the MDP, we change the problem at hand. The following result relates value functions in (un)regularized MDPs.

Proposition 3. Assume that L Ω ≤ Ω ≤ U Ω . Let π be any policy. We have that v π -U Ω 1 -γ 1 ≤ v π, Ω ≤ v π -L Ω 1 -γ 1 and v ∗ -U Ω 1 -γ 1 ≤ v ∗ , Ω ≤ v ∗ -L Ω 1 -γ 1 .

Regularization changes the optimal policy, the next result shows how it performs in the original MDP.

Theorem 2. Assume that L Ω ≤ Ω ≤ U Ω . We have that

<!-- formula-not-decoded -->

## 3.3. Related Works

Some of these results already appeared in the literature, in different forms and with specific regularizers. For example, the contraction of T ∗ , Ω (Prop. 2) was shown in various forms, e.g. (Fox et al., 2016; Asadi &amp; Littman, 2017; Dai et al., 2018), as well as the relation between (un)regularized optimal value functions (Th. 2), e.g. (Lee et al., 2018; Dai et al., 2018). The link to Legendre-Fenchel has also been considered before, e.g. (Dai et al., 2018; Mensch &amp; Blondel, 2018; Richemond &amp; Maginnis, 2017).

The core contribution of Sec. 3 is the regularized Bellman operator, inspired by Nesterov (2005) and Mensch &amp; Blondel (2018). It allows building in a principled and general way regularized MDPs, and generalizing existing results easily. More importantly, it is the core building block of regularized (A)DP, studied in the next sections. The framework and analysis we propose next rely heavily on this formalism.

## 4. Regularized Modified Policy Iteration

Having defined the notion of regularized MDPs, we still need algorithms that solve them. As the regularized Bellman operators have the same properties as the classical ones, we can apply classical dynamic programming. Here, we consider directly the modified policy iteration approach (Puterman &amp; Shin, 1978), that we regularize (reg-MPI for short):

<!-- formula-not-decoded -->

Given an initial v 0 , reg-MPI iteratively performs a regularized greedy step to get π k +1 and a partial regularized evaluation step to get v k +1 .

With m = 1 , we retrieve a regularized value iteration algorithm, that can be simplified as v k +1 = T ∗ , Ω v k (as π k +1 is greedy resp. to v k , we have T π k +1 , Ω v k = T ∗ , Ω v k ). With m = ∞ , we obtain a regularized policy iteration algorithm, that can be simplified as π k +1 = G Ω ( v π k , Ω ) (indeed, with a slight abuse of notation, ( T π k , Ω ) ∞ v k -1 = v π k , Ω ).

Before studying the convergence and rate of convergence of this general algorithmic scheme (with approximation), we discuss its links to state of the art algorithms (and more generally how it can be practically instantiated).

## 4.1. Related algorithms

Most existing schemes consider the negative entropy as the regularizer. Usually, it is also more convenient to work with q-functions. First, we consider the case m = 1 . In the exact case, the regularized value iteration scheme can be written

<!-- formula-not-decoded -->

In the entropic case, Ω ∗ ( q k ( s, · )) = ln ∑ a exp q k ( s, a ) . In an approximate setting, the q-function can be parameterized by parameters θ (for example, the weights of a neural network), write ¯ θ the target parameters (computed during the previous iteration) and ˆ E the empirical expectation over sampled transitions ( s i , a i , r i , s ′ i ) , an iteration amounts to minimize the expected loss

<!-- formula-not-decoded -->

Getting a practical algorithm may require more work, for example for estimating Ω ∗ ( q ¯ θ ( s ′ i , · )) in the case of continuous actions (Haarnoja et al., 2017), but this is the core principle of soft Q-learning (Fox et al., 2016; Schulman et al., 2017). This idea has also been applied using the Tsallis entropy as the regularizer (Lee et al., 2018).

Alternatively, assume that q k has been estimated. One could compute the regularized greedy policy analytically, π k +1 ( ·| s ) = ∇ Ω ∗ ( q k ( s, · )) . Instead of computing this for any state-action couple, one can generalize this from observed transitions to any state-action couple through a parameterized policy π w , by minimizing the KL divergence between both distributions:

<!-- formula-not-decoded -->

This is done in SAC (Haarnoja et al., 2018a), with an entropic regularizer (and thus ∇ Ω ∗ ( q k ( s, . )) = exp q k ( s, · ) ∑ a exp q k ( s,a ) ). This is also done in Maximum A Posteriori Policy Optimization (MPO) (Abdolmaleki et al., 2018b) with a KL regularizer (a case we discuss Sec. 5), or by Abdolmaleki et al. (2018a) with more general 'conservative' greedy policies.

Back to SAC, q k is estimated using a TD-like approach, by minimizing 2 for the current policy π :

<!-- formula-not-decoded -->

For SAC, we have Ω( π ( · , s )) = E a ∼ π ( ·| s ) [ln π ( a | s )] specifically (negative entropy). This approximate evaluation step corresponds to m = 1 , and SAC is therefore more a VI scheme than a PI scheme, as presented by Haarnoja et al. (2018a) (the difference with soft Q-learning lying in how the greedy step is performed, implicitly or explicitly). It could be extended to the case m&gt; 1 in two ways. One possibility is to minimize m times the expected loss (4), updating the target parameter vector ¯ θ between each optimization, but keeping the policy π fixed. Another possibility is to replace the 1-step rollout of Eq. (4) by an m -step rollout (similar to classical m -step rollouts, up to the additional regularizations correcting the rewards). Both are equivalent in the exact case, but not in the general case.

Depending on the regularizer, Ω ∗ or ∇ Ω ∗ might not be known analytically. In this case, one can still solve the greedy step directly. Recall that the regularized greedy policy satisfies π k +1 = max π T π, Ω v k . In an approximate setting, this amounts to maximize 3

<!-- formula-not-decoded -->

This improvement step is used by Riedmiller et al. (2018) with an entropy, as well as by TRPO (up to the fact that the objective is constrained rather than regularized), with a KL regularizer (see Sec. 5).

To sum up, for any regularizer Ω , with m = 1 one can concatenate greedy and evaluation steps as in Eq. (2), with m ≥ 1 one can estimate the greedy policy using either Eqs. (3) or (5), and estimate the q-function using Eq. 4, either performed m times repeatedly or combined with m -step rollouts, possibly combined with off-policy correction such as importance sampling or Retrace (Munos et al., 2016).

## 4.2. Analysis

We analyze the propagation of errors of the scheme depicted in Eq. (1), and as a consequence, its convergence and rate of convergence. To do so, we consider possible errors in both the (regularized) greedy and evaluation steps,

<!-- formula-not-decoded -->

2 Actually, a separate network is used to estimate the value function, but it is not critical here.

3 One could add a state-dependant baseline to q k , eg. v k , this does not change the maximizer but can reduce the variance.

with π k +1 = G glyph[epsilon1] ′ k +1 Ω ( v k ) meaning that for any policy π , we have T π, Ω v k ≤ T π k +1 , Ω v k + glyph[epsilon1] ′ k +1 . The following analysis is basically the same as the one of Approximate Modified Policy Iteration (AMPI) (Scherrer et al., 2015), thanks to the results of Sec. 3 (especially Prop. 2).

The distance we bound is the loss l k, Ω = v ∗ , Ω -v π k , Ω . The bound will involve the terms d 0 = v ∗ , Ω -v 0 and b 0 = v 0 -T π 1 , Ω v 0 . It requires also defining the following.

Definition 4 ( Γ -matrix (Scherrer et al., 2015)) . For n ∈ N ∗ , P n is the set of transition kernels defined as 1) for any set of n policies { π 1 , . . . , π n } , ∏ n i =1 ( γP π i ) ∈ P n and 2) for any α ∈ (0 , 1) and ( P 1 , P 2 ) ∈ P n × P n , αP 1 +(1 -α ) P 2 ∈ P n . Any element of P n is denoted Γ n .

We first state a point-wise bound on the loss. This is the same bound as for AMPI, generalized to regularized MDPs.

Theorem 3. After k iterations of scheme (6) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we provide a bound on the weighted glyph[lscript] p -norm of the loss, defined for a distribution ρ as ‖ l k ‖ p p,ρ = ρ | l k | p . Again, this is the AMPI bound generalized to regularized MDPs.

Corollary 1. Let ρ and µ be distributions. Let p , q and q ′ such that 1 q + 1 q ′ = 1 . Define the concentrability coefficients

<!-- formula-not-decoded -->

k iterations of scheme (6) , the loss satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As this is the same bound (up to the fact that it deals with regularized MDPs) as the one of AMPI, we refer to Scherrer et al. (2015) for a broad discussion about it. It is similar to other error propagation analyses in reinforcement learning, and generalizes those that could be obtained for regularized value or policy iteration. The factor m does not appear in the bound. This is also discussed by Scherrer et al. (2015), but basically this depends on where the error is injected. We could derive a regularized version of Classification-based Modified Policy Iteration (CBMPI, see Scherrer et al. (2015) again) and make it appear.

So, we get the same bound for reg-MPI that for unregularized AMPI, no better nor worse. This is a good thing, as it justifies considering regularized MDPs, but it does no explain the good empirical results of related algorithms.

With regularization, policies will be more stochastic than in classical approximate DP (that tends to produce deterministic policies). Such stochastic policies can induce lower concentrability coefficients. We also hypothesize that regularizing the greedy step helps controlling the related approximation error, that is the ‖ glyph[epsilon1] ′ k -i ‖ pq ′ ,µ terms. Digging this question would require instantiating more the algorithmic scheme and performing a finite sample analysis of the resulting optimization problems. We left this for future work, and rather pursue the general study of solving regularized MDPs, with varying regularizers now.

## 5. Mirror Descent Modified Policy Iteration

Solving a regularized MDP provides a solution that differs from the one of the unregularized MDP (see Thm. 2). The problem we address here is estimating the original optimal policy while solving regularized greedy steps. Instead of considering a fixed regularizer Ω( π ) , the key idea is to penalize a divergence between the policy π and the policy obtained at the previous iteration of an MPI scheme. We consider more specifically the Bregman divergence generated by the strongly convex regularizer Ω .

Let π ′ be some given policy (typically π k , when computing π k +1 ), the Bregman divergence generated by Ω is

<!-- formula-not-decoded -->

For example, the KL divergence is generated by the negative entropy: KL( π s || π ′ s ) = ∑ a π s ( a ) ln π s ( a ) π ′ s ( a ) . With a slight abuse of notation, as before, we will write

<!-- formula-not-decoded -->

This divergence is always positive, it satisfies Ω π ′ ( π ′ ) = 0 , and it is strongly convex in π (so Prop. 1 applies).

We consider a reg-MPI algorithmic scheme with a Bregman divergence replacing the regularizer. For the greedy step, we simply consider π k +1 = G Ω π k ( v k ) , that is

<!-- formula-not-decoded -->

This is similar to the update of the Mirror Descent (MD) algorithm in its proximal form (Beck &amp; Teboulle, 2003), with -q k playing the role of the gradient in MD. Therefore, we will call this approach Mirror Descent Modified Policy Iteration (MD-MPI). For the partial evaluation step, we can regularize according to the previous policy π k , that is v k +1 = ( T π k +1 , Ω π k ) m v k , or according to the current policy π k +1 , that is v k +1 = ( T π k +1 , Ω π k +1 ) m v k . As

Ω π k +1 ( π k +1 ) = 0 , this simplifies as v k +1 = ( T π k +1 ) m v k , that is a partial unregularized evaluation.

To sum up, we will consider two general algorithmic schemes based on a Bregman divergence, MD-MPI types 1 and 2 respectively defined as

<!-- formula-not-decoded -->

and both initialized with some v 0 and π 0 .

## 5.1. Related algorithms

To derive practical algorithms, the recipes provided in Sec. 4.1 still apply, just replacing Ω by Ω π k . If m = 1 , greedy and evaluation steps can be concatenated (only for MD-MPI type 1). In the general case ( m ≥ 1 ) the greedy policy (for MD-MPI types 1 and 2) can be either directly estimated (Eq. (5)) or trained to generalize the analytical solution (Eq. (3)). The partial evaluation can be done using a TD-like approach, either done repeatedly while keeping the policy fixed or considering m -step rollouts. Specifically, in the case of a KL divergence, one could use the fact that Ω ∗ π k ( q k ( s, · )) = ln ∑ a π k ( a | s ) exp q k ( s, a ) and that ∇ Ω ∗ π k ( q k ( s, · )) = π k ( ·| s ) exp q k ( s, · ) ∑ a π k ( a | s ) exp q k ( s,a ) .

This general algorithmic scheme allows recovering state of the art algorithms. For example, MD-MPI type 2 with m = ∞ and a KL divergence as the regularizer is TRPO (Schulman et al., 2015) (with a direct optimization of the regularized greedy step, as in Eq. (5), up to the use of a constraint instead of a regularization). DPP can be seen as a reparametrization 4 of MD-MPI type 1 with m = 1 (Azar et al., 2012, Appx. A). MPO (Abdolmaleki et al., 2018b) is derived from an expectation-maximization principle, but it can be seen as an instantiation of MD-MPI type 2, with a KL divergence, a greedy step similar to Eq. (3) (up to additional regularization) and an evaluation step similar to Eq. (4) (without regularization, as in type 2, with m-step return and with the Retrace off-policy correction). This also generally applies to the approach proposed by Abdolmaleki et al. (2018a) (up to an additional subtelty in the greedy step consisting in decoupling updates for the mean and variance in the case of a Gaussian policy).

## 5.2. Analysis

Here, we propose to analyze the error propagation of MDMPI (and thus, its convergence and rate of convergence). We think this is an important topic, as it has only been partly

4 Indeed, if one see MD-MPI as a Mirror Descent approach, one can see DPP as a dual averaging approach, somehow updating a kind of cumulative q-functions directly in the dual. However, how to generalize this beyond the specific DPP algorithm is unclear, and we let it for future work.

studied for the special cases discussed in Sec. 5.1. For example, DPP enjoys an error propagation analysis in supremum norm (yet it is a reparametrization of a special case of MDMPI, so not directly covered here), while TRPO or MPO are only guaranteed to have monotonic improvements, under some assumptions. Notice that we do not claim that our analysis covers all these cases, but it will provide the key technical aspects to analyze similar schemes (much like CBMPI compared to AMPI, as discussed in Sec. 4.2 or by Scherrer et al. (2015); where the error is injected changes the bounds).

In Sec. 4.2, the analysis was a straightforward adaptation of the one of AMPI, thanks to the results of Sec. 3 (the regularized quantities behave like their unregularized counterparts). It is no longer the case here, as the regularizer changes over iterations, depending on what has been computed so far. We will notably need a slightly different notion of approximate regularized greediness.

Definition 5 (Approximate Bregman divergence-regularized greediness) . Write J k ( π ) the (negative) optimization problem corresponding to the Bregman divergenceregularized greediness (that is, negative regularized Bellman operator of π applied to v k ):

<!-- formula-not-decoded -->

We write π k +1 ∈ G glyph[epsilon1] ′ k +1 Ω π k ( v k ) if for any policy π the policy π k +1 satisfies

<!-- formula-not-decoded -->

In other words, π k +1 ∈ G glyph[epsilon1] ′ k +1 Ω π k ( v k ) means that π k +1 is glyph[epsilon1] ′ k +1 -close to satisfying the optimality condition, which might be slightly stronger than being glyph[epsilon1] ′ k +1 -close to the optimal (as for AMPI or reg-MPI). Given this, we consider MD-MPI with errors in both greedy and evaluation steps, type 1

<!-- formula-not-decoded -->

and type 2

<!-- formula-not-decoded -->

.

The quantity we are interested in is v ∗ -v π k , that is suboptimality in the unregularized MDP, while the algorithms compute new policies with a regularized greedy operator. So, we need to relate regularized and unregularized quantities when using a Bregman divergence based on the previous policy. The next lemma is the key technical result that allows analyzing MD-MPI.

Lemma 1. Assume that π k +1 ∈ G glyph[epsilon1] ′ k +1 Ω π k ( v k ) , as defined in Def. 5. Then, the policy π k +1 is glyph[epsilon1] ′ k +1 -close to the regularized greedy policy, in the sense that for any policy π

<!-- formula-not-decoded -->

Moreover, we can relate the (un)regularized Bellman operators applied to v k . For any policy π (so notably for the unregularized optimal policy π ∗ ), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We're interested in bounding the loss l k = v ∗ -v π k , or some related quantity, for each type of MD-MPI. To do so, we introduce quantities similar to the ones of the AMPI analysis (Scherrer et al., 2015), defined respectively for types 1 and 2: 1) The distance between the optimal value function and the value before approximation at the k th iteration, d 1 k = v ∗ -( T π k , Ω π k -1 ) m v k -1 = v ∗ -( v k -glyph[epsilon1] k ) and d 2 k = v ∗ -( T π k ) m v k -1 = v ∗ -( v k -glyph[epsilon1] k ) ; 2) The shift between the value before approximation and the policy value a iteration k , s 1 k = ( T π k , Ω π k -1 ) m v k -1 -v π k = ( v k -glyph[epsilon1] k ) -v π k and s 2 k = ( T π k ) m v k -1 -v π k = ( v k -glyph[epsilon1] k ) -v π k ; 3) the Bellman residual at iteration k , b 1 k = v k -T π k +1 , Ω π k v k and b 2 k = v k -T π k +1 v k .

For both types ( h ∈ { 1 , 2 } ), we have that l h k = d h k + s h k , so bounding the loss requires bounding these quantities, which is done in the following lemma (quantities related to both types enjoy the same bounds).

<!-- formula-not-decoded -->

These bounds are almost the same as the ones of AMPI (Scherrer et al., 2015, Lemma 2), up to the additional δ k ( π ∗ ) term in the bound of the distance d h k . One can notice that summing these terms gives a telescopic sum: ∑ K -1 k =0 δ k ( π ∗ ) = D Ω ( π ∗ || π 0 ) -D Ω ( π ∗ || π K ) ≤ D Ω ( π ∗ || π 0 ) ≤ sup π D Ω ( π || π 0 ) . For example, if D Ω is the KL divergence and π 0 the uniform policy, then ‖ sup π D Ω ( π || π 0 ) ‖ ∞ = ln |A| . This suggests that we must bound the regret L K defined as

<!-- formula-not-decoded -->

Theorem 4. Define R Ω π 0 = ‖ sup π D Ω ( π || π 0 ) ‖ ∞ , after K iterations of MD-MPI, for h = 1 , 2 , the regret satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From this,we can derive an glyph[lscript] p -bound for the regret.

Corollary 2. Let ρ and µ be distributions over states. Let p , q and q ′ be such that 1 q + 1 q ′ = 1 . Define the concentrability coefficients C i q as in Cor. 1. After K iterations, the regret satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This result bounds the regret, while it is usually the loss that is bounded. Both can be related as follows.

Proposition 4. For any p ≥ 1 and distribution ρ , we have min 1 ≤ k ≤ K ‖ v ∗ -v π k ‖ 1 ,ρ ≤ 1 K ‖ L K ‖ p,ρ .

This means that if we can control the average regret, then we can control the loss of the best policy computed so far. This suggests that practically we should not use the last policy, but this best policy.

From Cor. 2 can be derived the convergence and rate of convergence of MD-MPI in the exact case.

Corollary 3. Both MD-MPI type 1 and 2 enjoy the following rate of convergence, when no approximation is done ( glyph[epsilon1] k = glyph[epsilon1] ′ k = 0 ),

<!-- formula-not-decoded -->

In classical DP and in regularized DP (see Cor. 1), there is a linear convergence rate (the bound is 2 γ K 1 -γ ‖ v ∗ -v 0 ‖ ∞ ), while in this case we only have a logarithmic convergence rate. We also pay an horizon factor (square dependency in 1 1 -γ instead of linear). This is normal, as we bound the regret instead of the loss. Bounding the regret in classical DP would lead to the bound of Cor. 3 (without the R Ω π 0 term).

The convergence rate of the loss of MD-MPI is an open question, but a sublinear rate is quite possible. Compared to classical DP, we slow down greediness by adding the Bregman divergence penalty. Yet, this kind of regularization is used in an approximate setting, where it favors stability empirically (even if studying this further would require much more work regarding the ‖ glyph[epsilon1] ′ k ‖ term, as discussed in Sec. 4.2).

As far as we know, the only other approach that studies a DP scheme regularized by a divergence and that offers a convergence rate is DPP, up to the reparameterization we discussed earlier. MD-MPI has the same upper-bound as DPP in the exact case (Azar et al., 2012, Thm. 2). However, DPP bounds the loss, while we bound a regret. This means that if the rate of convergence of our loss can be sublinear, it is superlogarithmic (as the rate of the regret is logarithmic), while the rate of the loss of DPP is logarithmic.

To get more insight on Cor. 2, we can group the terms differently, by grouping the errors.

Corollary 4. With the same notations as Cor. 2, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Compared to the bound of AMPI (Scherrer et al., 2015, Thm. 7), instead of propagating the errors, we propagate the sum of errors over previous iterations normalized by the total number of iterations. So, contrary to approximate DP, it is no longer the last iterations that have the highest influence on the regret. Yet, we highlight again the fact that we bound a regret, and bounding the regret of AMPI would provide a similar result.

Our result is similar to the error propagation of DPP (Azar et al., 2012, Thm. 5), except that we sum norms of errors, instead of norming a sum of errors, the later being much better (as it allows the noise to cancel over iterations). Yet, as said before, DPP is not a special case of our framework, but a reparameterization of such one. Consequently, while we estimate value functions, DPP estimate roughly at iteration k a sum of k advantage functions (converging to -∞ for any suboptimal action in the exact case). As explained before, where the error is injected does matter. Knowing if the DPP's analysis can be generalized to our framework (MPI scheme, glyph[lscript] p bounds) remains an open question.

To get further insight, we can express the bound using different concentrability coefficients.

Corollary 5. Define the concentrability coefficient C l,k q as

C l,k q = (1 -γ ) 2 γ l -γ k ∑ k -1 i = l ∑ ∞ j = i c q ( j ) , the regret then satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We observe again that contrary to ADP, the last iteration does not have the highest influence, and we do not enjoy a decrease of influence at the exponential rate γ towards the initial iterations. However, we bound a different quantity (regret instead of loss), that explains this behavior. Here again, bounding the regret in AMPI would lead to the same bound (up to the term R Ω π 0 ). Moreover, sending p and K to infinity, defining glyph[epsilon1] = sup j ‖ glyph[epsilon1] j ‖ ∞ and glyph[epsilon1] ′ = sup j ‖ glyph[epsilon1] ′ j ‖ ∞ , we get lim sup K →∞ 1 K ‖ L K ‖ ∞ ≤ 2 γglyph[epsilon1] + glyph[epsilon1] ′ (1 -γ ) 2 , which is the classical asymptotical bound for approximate value and policy iterations (Bertsekas &amp; Tsitsiklis, 1996) (usually stated without greedy error). It is generalized here to an approximate MPI scheme regularized with a Bregman divergence.

## 6. Conclusion

We have introduced a general theory of regularized MDPs, where the usual Bellman evaluation operator is modified by either a fixed convex function or a Bregman divergence between consecutive policies. For both cases, we proposed a general algorithmic scheme based on MPI. We shown how many (variations of) existing algorithms could be derived from this general algorithmic scheme, and also analyzed and discussed the related propagation of errors.

We think that this framework can open many perspectives, among which links between (approximate) DP and proximal convex optimization (going beyond mirror descent), temporal consistency equations (roughly regularized Bellman residuals), regularized policy search (maximizing the expected regularized value function), inverse reinforcement learning (thanks to uniqueness of greediness in this regularized framework) or zero-sum Markov games (regularizing the two-player Bellman operators). We develop more these points in the appendix.

This work also lefts open questions, such as combining the propagation of errors with a finite sample analysis, or what specific regularizer one should choose for what context. Some approaches also combine a fixed regularizer and a divergence (Akrour et al., 2018), a case not covered here and worth being investigated.

## References

- Abdolmaleki, A., Springenberg, J. T., Degrave, J., Bohez, S., Tassa, Y., Belov, D., Heess, N., and Riedmiller, M. Relative entropy regularized policy iteration. arXiv preprint arXiv:1812.02256 , 2018a.
- Abdolmaleki, A., Springenberg, J. T., Tassa, Y., Munos, R., Heess, N., and Riedmiller, M. Maximum a posteriori policy optimisation. In International Conference on Learning Representations (ICLR) , 2018b.
- Akrour, R., Abdolmaleki, A., Abdulsamad, H., Peters, J., and Neumann, G. Model-free trajectory-based policy optimization with monotonic improvement. The Journal of Machine Learning Research (JMLR) , 19(1):565-589, 2018.
- Asadi, K. and Littman, M. L. An alternative softmax operator for reinforcement learning. In International Conference on Machine Learning (ICML) , 2017.
- Azar, M. G., G´ omez, V., and Kappen, H. J. Dynamic policy programming. Journal of Machine Learning Research (JMLR) , 13(Nov):3207-3245, 2012.
- Beck, A. and Teboulle, M. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters , 31(3):167-175, 2003.
- Bertsekas, D. P. and Tsitsiklis, J. N. Neuro-Dynamic Programming . Athena Scientific, 1st edition, 1996. ISBN 1886529108.
- Dai, B., Shaw, A., Li, L., Xiao, L., He, N., Liu, Z., Chen, J., and Song, L. Sbeed: Convergent reinforcement learning with nonlinear function approximation. In International Conference on Machine Learning (ICML) , 2018.
- Finn, C., Levine, S., and Abbeel, P. Guided cost learning: Deep inverse optimal control via policy optimization. In International Conference on Machine Learning (ICML) , 2016.
- Fox, R., Pakman, A., and Tishby, N. Taming the noise in reinforcement learning via soft updates. In Conference on Uncertainty in Artificial Intelligence (UAI) , 2016.
- Fu, J., Luo, K., and Levine, S. Learning robust rewards with adversarial inverse reinforcement learning. In International Conference on Representation Learning , 2018.
- Haarnoja, T., Tang, H., Abbeel, P., and Levine, S. Reinforcement learning with deep energy-based policies. In International Conference on Machine Learning (ICML) , 2017.
- Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning (ICML) , 2018a.
- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., et al. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905 , 2018b.
- Hiriart-Urruty, J.-B. and Lemar´ echal, C. Fundamentals of convex analysis . Springer Science &amp; Business Media, 2012.
- Lee, K., Choi, S., and Oh, S. Sparse markov decision processes with causal sparse tsallis entropy regularization for reinforcement learning. IEEE Robotics and Automation Letters , 3(3):1466-1473, 2018.
- Levine, S. Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review. arXiv preprint arXiv:1805.00909 , 2018.
- Martins, A. and Astudillo, R. From softmax to sparsemax: A sparse model of attention and multi-label classification. In International Conference on Machine Learning (ICML) , 2016.
- Mensch, A. and Blondel, M. Differentiable dynamic programming for structured prediction and attention. In International Conference on Machine Learning (ICML) , 2018.
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning (ICML) , 2016.
- Morgenstern, O. and Von Neumann, J. Theory of games and economic behavior . Princeton university press, 1953.
- Munos, R., Stepleton, T., Harutyunyan, A., and Bellemare, M. Safe and efficient off-policy reinforcement learning. In Advances in Neural Information Processing Systems (NIPS) , pp. 1054-1062, 2016.
- Nachum, O., Norouzi, M., Xu, K., and Schuurmans, D. Bridging the gap between value and policy based reinforcement learning. In Advances in Neural Information Processing Systems (NIPS) , 2017.
- Nachum, O., Chow, Y., and Ghavamzadeh, M. Path consistency learning in tsallis entropy regularized mdps. arXiv preprint arXiv:1802.03501 , 2018.
- Nemirovski, A. Prox-method with rate of convergence o (1 /t ) for variational inequalities with lipschitz continuous monotone operators and smooth convex-concave

- saddle point problems. SIAM Journal on Optimization , 15(1):229-251, 2004.
- Nesterov, Y. Smooth minimization of non-smooth functions. Mathematical programming , 103(1):127-152, 2005.
- Nesterov, Y. Primal-dual subgradient methods for convex problems. Mathematical programming , 120(1):221-259, 2009.
- Neu, G., Jonsson, A., and G´ omez, V. A unified view of entropy-regularized Markov decision processes. arXiv preprint arXiv:1705.07798 , 2017.
- Ng, A. Y., Harada, D., and Russell, S. Policy invariance under reward transformations: Theory and application to reward shaping. In International Conference on Machine Learning (ICML) , 1999.
- Perolat, J., Scherrer, B., Piot, B., and Pietquin, O. Approximate dynamic programming for two-player zero-sum markov games. In International Conference on Machine Learning (ICML) , 2015.
- Peters, J., Mulling, K., and Altun, Y . Relative entropy policy search. In AAAI Conference on Artificial Intelligence , 2010.
- Puterman, M. L. and Shin, M. C. Modified policy iteration algorithms for discounted markov decision problems. Management Science , 24(11):1127-1137, 1978.
- Richemond, P. H. and Maginnis, B. A short variational proof of equivalence between policy gradients and soft q learning. arXiv preprint arXiv:1712.08650 , 2017.
- Riedmiller, M., Hafner, R., Lampe, T., Neunert, M., Degrave, J., Wiele, T., Mnih, V., Heess, N., and Springenberg, J. T. Learning by playing solving sparse reward tasks from scratch. In International Conference on Machine Learning (ICML) , pp. 4341-4350, 2018.
- Scherrer, B., Ghavamzadeh, M., Gabillon, V., Lesner, B., and Geist, M. Approximate modified policy iteration and its application to the game of tetris. Journal of Machine Learning Research (JMLR) , 16:1629-1676, 2015.
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In International Conference on Machine Learning (ICML) , 2015.
- Schulman, J., Chen, X., and Abbeel, P. Equivalence between policy gradients and soft q-learning. arXiv preprint arXiv:1704.06440 , 2017.
- Sutton, R. S., McAllester, D. A., Singh, S. P., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems (NIPS) , 2000.
- Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning , 8(3-4):229-256, 1992.
- Ziebart, B. D., Maas, A. L., Bagnell, J. A., and Dey, A. K. Maximum entropy inverse reinforcement learning. In AAAI Conference on Artificial Intelligence (AAAI) , 2008.

This appendices provide the proofs for all stated results (Appx. A to C) and discuss in more details the perspectives mentioned in Sec. 5 (Appx. D).

## A. Proofs of section 3

In this section, we prove the results of Sec. 3. We start with the properties of the regularized Bellman operators.

Proof of Proposition 2. We can write T π, Ω v = r π -Ω( π ) + γP π v , it is obviously affine (in v ). Then, we show that the operators are monotonous. For the evaluation operator, we have

<!-- formula-not-decoded -->

For the optimality operator, we have

<!-- formula-not-decoded -->

Then, we show the distributivity property. For the evaluation operator, we have

<!-- formula-not-decoded -->

For the optimality operator, for any s ∈ S , we have

<!-- formula-not-decoded -->

Lastly, we study the contraction of both operators. For the evaluation operator, we have

<!-- formula-not-decoded -->

So, the contraction is the same as the one of the unregularized operator. For the optimality operator, we have that

<!-- formula-not-decoded -->

Pick s ∈ S , and without loss of generality assume that [ T ∗ , Ω v 1 ]( s ) ≥ [ T ∗ , Ω v 2 ]( s ) . Write also π 1 = G Ω ( v 1 ) and π 2 = G Ω ( v 2 ) . We have

<!-- formula-not-decoded -->

The stated result follows immediately.

Then, we show that in a regularized MDP, the policy greedy respectively to the optimal value function is indeed the optimal policy, and is unique.

Proof of Theorem 1. The uniqueness of π ∗ , Ω is a consequence of the strong convexity of Ω , see Prop. 1. On the other hand, by definition of the greediness, we have

<!-- formula-not-decoded -->

This proves that v ∗ , Ω is the value function of π ∗ , Ω . Next, for any function v ∈ R S and any policy π , we have

<!-- formula-not-decoded -->

Using monotonicity, we have that

<!-- formula-not-decoded -->

By direct induction, for any n ≥ 1 , T n ∗ , Ω v ≥ T n π, Ω v . Taking the limit as n →∞ , we conclude that v ∗ , Ω ≥ v π, Ω .

Next, we relate regularized and unregularized value functions (for a given policy, and for the optimal value function).

Proof of Proposition 3. We start by linking (un)regularized values of a given policy. Let v ∈ R S . As T π, Ω v = T π v -Ω( π ) , we have that

<!-- formula-not-decoded -->

We work on the left inequality first. We have

<!-- formula-not-decoded -->

By direct induction, for any n ≥ 1 ,

Taking the limit as n →∞ we obtain

## B. Proofs of section 4

The results of section 4 do not need to be proven. Indeed, we have shown in Sec. 3 that all involved quantities of regularized MDPs satisfy the same properties as their unregularized counterpart. Therefore, the proofs of these results are identical to the proofs provided by Scherrer et al. (2015), up to the replacement of value functions, Bellman operators, and so on, by their regularized counterparts. The proofs for Mirror Descent Modified Policy Iteration (Sec. 5) are less straightforward.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof is similar for the right inequality. Next, we link the (un)regularized optimal values. As a direct corollary of Prop. 1, for any v ∈ R S we have

<!-- formula-not-decoded -->

Then, the proof is the same as above, switching evaluation and optimality operators.

Lastly, we show how good is the optimal policy of the regularized MDP for the original problem (unregularized MDP).

Proof of Theorem 2. The right inequality is obvious, as for any π , v ∗ ≥ v π . For the left inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C. Proofs of section 5

As a prerequisite of Lemma 1, we need the following result.

Lemma 3 (Three-point identity) . Let π be any policy, we have that

<!-- formula-not-decoded -->

Proof. This is the classical three-point identity of Bregman divergences, and can be checked by calculus:

<!-- formula-not-decoded -->

Now, we can prove the key lemma of MD-MPI.

Proof of Lemma 1. Let J k be as defined in Def. 5,

<!-- formula-not-decoded -->

and let π k +1 ∈ G glyph[epsilon1] ′ k +1 Ω π k ( v k ) , that is 〈∇ J k ( π k +1 ) , π -π k +1 〉 + glyph[epsilon1] ′ k +1 ≥ 0 . By convexity of J k , for any policy π , we have

<!-- formula-not-decoded -->

This is the first result stated in Lemma 1.

Next, we relate (un)regularized quantities. We start with the following decomposition

<!-- formula-not-decoded -->

Taking the gradient of J k (by using the definition of the Bregman divergence), we get

<!-- formula-not-decoded -->

Injecting Eq. (8) into Eq. (7), we get

<!-- formula-not-decoded -->

where we used in the last inequality the fact that 〈 q k , π 〉 = T π v k . From the definition of the regularized Bellman operator, we have that T π k +1 v k -D Ω ( π k +1 || π k ) = T π k +1 , Ω π k v k , so Eq. (9) is equivalent to the second result of Lemma 1:

<!-- formula-not-decoded -->

As the Bregman divergence is positive, -D Ω ( π || π k +1 ) ≤ 0 , and thus Eq. (9) implies the last result of Lemma 1:

<!-- formula-not-decoded -->

This concludes the proof.

Next, we prove the bounds for b h k , s h k and d h k , for h = 1 , 2 (if the bounds are the same, the proofs differ).

Proof of Lemma 2. We start by bounding the quantities for MD-MPI type 1. First, we consider the Bellman residual:

<!-- formula-not-decoded -->

In the previous equations, we used the following facts:

- (a) T π k v k = T π k , Ω π k v k as Ω π k ( π k ) = 0 .
- (b) We used two facts. First, T π k v k ≥ T π k v k -Ω π k -1 ( π k ) = T π k , Ω π k -1 v k , as Ω π k -1 ( π k ) ≥ 0 . Second, by Lemma 1, T π k , Ω π k v k -T π k +1 , Ω π k v k ≤ glyph[epsilon1] ′ k +1 .
- (c) We used two facts. First, generally speaking, we have T π, Ω ( v 1 + v 2 ) = T π, Ω v 1 + γP π v 2 (as T π, Ω is affine). Second, by definition x k = ( I -γP π k ) glyph[epsilon1] k + glyph[epsilon1] ′ k +1 .
- (d) By definition, v k = ( T π k , Ω π k -1 ) m v k -1 + glyph[epsilon1] k .

Next, we bound the shift s 1 k :

<!-- formula-not-decoded -->

In the previous equations, we used the following facts:

- (a) Generally speaking, if Ω ≥ 0 then v π, Ω -v π ≤ 0 (see Prop. 3).
- (b) With a slight abuse of notation, for any v ∈ R S , v π, Ω = T ∞ π, Ω v .

Then we bound the distance d 1 k :

<!-- formula-not-decoded -->

In the previous equations, we used the following facts:

- (a) By Lemma 1,
- (b) For this step, we used:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proofs for the quantities involved in MD-MPI type 2 are similar, even if their definition differ. First, we consider the Bellman residual

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the previous equations, we used the following facts:

- (a) This is because Ω π k ( π k +1 ) ≥ 0 and by definition of T π, Ω v = T π v -Ω( π ) .
- (b) It uses the fact that T π k v k = T π k , Ω π k v k (as Ω π k ( π k ) = 0 ).
- (c) This is by Lemma 1.
- (d) This is by definition of x k = glyph[epsilon1] k -γP π k glyph[epsilon1] k + glyph[epsilon1] ′ k +1 .
- (e) This is by definition of v k = ( T π k ) m v k -1 + glyph[epsilon1] k .

Then, we bound the shift s 2 k , the technique being the same as before:

<!-- formula-not-decoded -->

To finish with, we prove the bound on the distance

<!-- formula-not-decoded -->

Now, we will show the component-wise bound on the regret L k of Thm. 4

Proof of Theorem 4. The proof is similar to the one of AMPI (Scherrer et al., 2015, Lemma 2), up to the additional term δ k ( π ∗ ) and to the different bounded quantity. We will make use of the notation Γ , defined in Def. 4, and we will write Γ ∗ if only the stochastic kernel induced by the optimal policy π ∗ is involved. In other words, we write Γ j ∗ = ( γP π ∗ ) j .

From Lemma 2, we have that

<!-- formula-not-decoded -->

As the bound is the same for h = 1 , 2 , we remove the upperscript (and reintroduce it only when necessary, that is when going back to the core definition of these quantities). By direct induction, we get

<!-- formula-not-decoded -->

Therefore, the loss l k can be bounded as (defining L k at the same time)

<!-- formula-not-decoded -->

The loss L k is exactly of the same form as the one of AMPI, we'll take advantage of this later. From this bound on the loss l k , we can bound the regret as:

<!-- formula-not-decoded -->

We will first work on the last double sum. For this, we define ∆ j ( π ∗ ) = ∑ j k =0 δ k ( π ∗ ) . We have

<!-- formula-not-decoded -->

For the last line, we used the fact that Γ ∗ only involves the P π ∗ transition kernel, that does not depend on any iteration. Now, we can bound the term ∆ k ( π ∗ ) , for any k ≥ 0 as follows. Let write R Ω π 0 = ‖ sup π D Ω ( π || π 0 ) ‖ ∞ , we have

<!-- formula-not-decoded -->

Given the definition of Γ , we have that Γ j 1 = γ j 1 , so

<!-- formula-not-decoded -->

Next, we work on the term L k . As stated before, it is exactly the same as the one of the AMPI analysis, and the proof of Scherrer et al. (2015, Lemma 2) applies almost readily, we do not repeat it fully here. The only difference that appears and that induces a slight modification of the bound (on L k ) ) is how b 0 and d 0 are related, that will modify the η k term of the original proof. We will link b 0 and d 0 for both types of MD-MPI.

For MD-MPI type 1 (with the natural convention that glyph[epsilon1] 0 = 0 ), we have that

<!-- formula-not-decoded -->

The Bellman residual can be written as

<!-- formula-not-decoded -->

where we used in the penultimate line the fact that δ 0 ( π ∗ ) = D Ω ( π ∗ || π 0 ) -D Ω ( π ∗ || π 1 ) ≤ D Ω ( π ∗ || π 0 ) and in the last line the same bounding as before. So, the link between b 1 0 and d 1 0 is the same as for AMPI, up to the additional R Ω π 0 1 term.

For MD-MPI type 2, we have (with the same convention)

<!-- formula-not-decoded -->

Working on the Bellman residual

<!-- formula-not-decoded -->

So, we have the same bound.

Combining this with the part of the proof that does not change, and that we do not repeat, we get the following bound on L k :

<!-- formula-not-decoded -->

with h ( k ) being defined as

<!-- formula-not-decoded -->

Combining Eqs (11) and (12) into Eq. (10), we can bound the regret:

<!-- formula-not-decoded -->

This concludes the proof.

To prove Cor. 2, we will need a result from Scherrer et al. (2015), that we recall first.

Lemma 4 (Lemma 6 of Scherrer et al. (2015)) . Let I and ( J i ) i ∈I be sets of non-negative integers, {I 1 , . . . , I n } be a partition of I , and f and ( g i ) i ∈I be functions satisfying

<!-- formula-not-decoded -->

Then, for all p , q such that 1 p + 1 q = 1 and for all distributions ρ and µ , we have

<!-- formula-not-decoded -->

with the following concentrability coefficients,

<!-- formula-not-decoded -->

Now, we can prove the stated result.

Proof of Corollary 2. The proof is an application of Lemma 4 to Thm. 4. We define I = { 1 , 2 , . . . , K 2 + K +1 } and the associated trivial partition (that is, I i = { i } ). For each i ∈ I we define

<!-- formula-not-decoded -->

With this, Thm. 4 rewrites as

<!-- formula-not-decoded -->

The results follows by applying Lemma 4 and using the fact that ∑ j ≥ i γ j = γ i 1 -γ

The proof of Prop. 4 is a basic application of the H¨ older inequality.

Proof of Proposition 4. We write ◦ the Hadamard product. Recall that l k = v ∗ -v π k ≥ 0 and that L K = ∑ K k =1 l k . Notice that the sequence v π k is not necessarily monotone, even without approximation error. On one side, we have that

<!-- formula-not-decoded -->

On the other side, with q such that 1 p + 1 q = 1 we have that

<!-- formula-not-decoded -->

where we used the H¨ older inequality. Combining both equations provides the stated result.

Cor. 3 is a direct consequence of Cor. 2.

Proof of Corollary 3. Taking the limit of Cor. 2 as p →∞ , when the errors are null, gives

<!-- formula-not-decoded -->

where we used the fact that ∑ K k =1 γ k = γ 1 -γ K 1 -γ . The result follows by grouping terms and using d 0 = v ∗ -v 0 .

The proof of Cor. 4 is mainly a manipulation of sums.

Proof of Corollary 4. From Cor. 2, we have that

<!-- formula-not-decoded -->

We only need to work on the first two sums. To shorten the notations, write α i = γ i 1 -γ ( C i q ) 1 p , β i = ‖ glyph[epsilon1] k -i ‖ pq ′ ,µ and β ′ i = ‖ glyph[epsilon1] ′ k -i ‖ pq ′ ,µ . We have:

<!-- formula-not-decoded -->

The result follows by reinjecting this in Cor. 2 after having replaced α i , β i and β ′ i .

The proof of Cor. 5 basically consists in the application of Lemma 4 to a rewriting of Thm. 4.

Proof of Corollary 5. By Thm. 4, we have

<!-- formula-not-decoded -->

We start by rewriting the two first sums. For the first one, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, the bound on the loss can be writen as

<!-- formula-not-decoded -->

In order to apply Lemma 4 to this bound, we consider I = { 1 , 2 , . . . 2 K +1 } and the associated trivial partition I i = { i } .

Similarly, we have that

For each i ∈ I , we define:

With this, Eq. (13) rewrites as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With the c q term defined in Lemma 4, using the fact that ∑ k -1 i = l ∑ ∞ j = i γ j = γ l -γ k (1 -γ ) 2 , as well as the following concentrability coefficient,

<!-- formula-not-decoded -->

we get the following bound on the regret:

<!-- formula-not-decoded -->

## D. Perspectives on regularized MDPs

Here, we discuss in more details the perspectives briefly mentioned in Sec. 6.

## D.1. Dynamic programming and optimization

We have shown how MPI regularized by a Bregman divergence is related to Mirror Descent. The computation of the regularized greedy policy is similar to a mirror descent step, where the q -function plays the role of the negative subgradient. In this sense, the policy lives in the primal space while the q -function lives in the dual space. It would be interesting to take inspiration from proximal convex optimization to derive new dynamic programming approaches, for example based on Dual Averaging (Nesterov, 2009) or Mirror Prox (Nemirovski, 2004).

For example, consider the case of a fixed regularizer. We have seen that it leads to a different solution than the original one (see Thm. 2). Usually, one considers a scaled negative entropy as such a regularizer 5 , Ω( π ( ·| s )) = α ∑ a π ( a | s ) ln π ( a | s ) . The choice of this parameter is important practically and problem-dependent: too high and the solution of the regularized problem will be very different from the original one, too low and the algorithm will not benefit from the regularization. A natural idea is to vary the weight of the regularizer over iterations (Peters et al., 2010; Abdolmaleki et al., 2018a; Haarnoja et al., 2018b), much like a learning rate in a gradient descent approach.

More formally, in our framework, write Ω k = α k Ω and consider the following weighted reg-MPI scheme,

<!-- formula-not-decoded -->

5 This is the case for SAC or soft Q-learning, for example. Sometimes, it is the reward that is scaled, but both are equivalent.

<!-- formula-not-decoded -->

with G glyph[epsilon1] ′ k +1 Ω k ( v k ) as defined in Sec. 4.2: for any policy π , T π, Ω k v k ≤ T π k +1 , Ω k v k + glyph[epsilon1] ′ k +1 . By applying the proof techniques developed previously, one can obtain easily the following result.

Theorem 5. Define R Ω = ‖ sup π Ω( π ) ‖ ∞ . Assume that the series ( α k ) k ≥ 0 is positive and decreasing, and that the regularizer Ω is positive (without loss of generality). After K iterations of the preceding weighted reg-MPI scheme, the regret satisfies

<!-- formula-not-decoded -->

with h ( k ) = 2 ∑ ∞ j = k Γ j | d 0 | or h ( k ) = 2 ∑ ∞ j = k Γ j | b 0 | .

Proof. The proof is similar to the one of MD-MPI, as it bounds the regret, but also simpler as it does not require Lemma 1. The principle is still to bound the distance d k and the shift s k , that both require bounding the Bellman residual b k . From this, one can bound the loss l k = d k + s k , and then the regret L K = ∑ K k =1 l k . We only specify what changes compared to the previous proofs.

Bounding the shift s k is done as before, and one get the same bound:

<!-- formula-not-decoded -->

For the distance d k , we have

<!-- formula-not-decoded -->

The rest of the bounding is similar to MD-MPI, and we get

<!-- formula-not-decoded -->

So, this is similar to the bound of MD-MPI, with the term α k R Ω 1 replacing the term δ k ( π ∗ ) . For the Bellman residual, we have

<!-- formula-not-decoded -->

where we used the facts that α k ≤ α k -1 and that Ω ≥ 0 for bounding the first term:

<!-- formula-not-decoded -->

The rest of the bounding is as before and gives b k ≤ ( γP π k ) m b k -1 + x k . From these bounds, one can bound L k as previously.

From Thm. 5, we can obtain an glyph[lscript] p -bound on the regret, and from this the rate of convergence of the average regret. For MD-MPI, the rate of convergence was in O ( 1 K ) . Here, it depends on the weighting of the regularizer. For example, if α k is in O ( 1 k ) , then the average regret will be in O ( ln K K ) . If α k is in O ( 1 √ k ) , then the average regret will be in O ( 1 √ K ) . This illustrates the kind of things that can be done with the proposed framework of regularized MDPs.

## D.2. Temporal consistency equations

Thanks to Ω , the regularized greedy policies are unique, and thus is the regularized optimal policy. The pair of optimal policy and optimal value function can be characterized as follows.

Corollary 6. The optimal policy and optimal value function in a regularized MDP are the unique functions satisfying

<!-- formula-not-decoded -->

.

Proof. This is a direct consequence of Thm. 1

This provides a general way to estimate the optimal policy. For example, if Ω is the negative entropy, this set of equations simplifies to

<!-- formula-not-decoded -->

This has been used by Nachum et al. (2017) or Dai et al. (2018), where it is called 'temporal consistency equation', to estimate the optimal value-policy pair by minimizing the related residual,

<!-- formula-not-decoded -->

This idea has also been extended to Tsallis entropy (Nachum et al., 2018). In this case, the set of equations does not simplify as nicely as with the Shannon entropy. Instead, the approach consists in considering the Lagrangian derived from the Legendre-Fenchel transform (and the resulting temporal consistency equation involves Lagrange multipliers, that have to be learnt too). This idea could be extended to other regularizers. One could also replace the regularizer Ω by a Bregman divergence, and estimate the optimal policy by solving a sequence of temporal consistency equations.

## D.3. Regularized policy gradient

Policy search approaches often combine policy gradient with an entropic regularization, typically to prevent the policy from becoming too quickly deterministic (Williams, 1992; Mnih et al., 2016). The policy gradient theorem (Sutton et al., 2000) can easily be extended to the proposed framework.

Let ν be a (user-defined) state distribution, the classical policy search approach consists in maximizing J ( π ) = E s ∼ ν [ v π ( s )] = νv π . This principle can easily be extended to regularized MDPs, by maximizing

<!-- formula-not-decoded -->

Write d ν,π the γ -weighted occupancy measure induced by the policy π when the initial state is sampled from ν , defined as d ν,π = (1 -γ ) ν ( I -γP π ) -1 ∈ ∆ S . Slightly abusing notations, we'll also write d ν,π ( s, a ) = d ν,π ( s ) π ( a | s ) .

Theorem 6 (Policy gradient for regularized MDPs) . The gradient of J Ω is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We have to study the gradient of the value function. For this, the nabla-log trick is useful: ∇ π = π ∇ ln π .

<!-- formula-not-decoded -->

So, the components of ∇ v π, Ω are the (unregularized) value functions corresponding to the rewards being the components of q π, Ω ( s, a ) ∇ ln π ( a | s ) -∇ Ω( π ( . | s )) . Consequently,

<!-- formula-not-decoded -->

Proof. We have that

Using the chain-rule, we have that

<!-- formula-not-decoded -->

Injecting this in the previous result concludes the proof.

Even in the entropic case, it might be not exactly the same as the usual regularized policy gradient, notably because our result involves the regularized q -function. Again, the regularizer Ω could be replaced by a Bregman divergence, to ultimately estimate the optimal policy of the original MDP. It would be interesting to compare empirically the different resulting policy gradients approaches, we left this for future work.

## D.4. Regularized inverse reinforcement learning

Inverse reinforcement learning (IRL) consists in finding a reward function that explains the behavior of an expert which is assumed to act optimally. It is often said that it is an ill-posed problem. The classical example is the null reward function that explains any behavior (as all policies are optimal in this case).

We argue that in this regularized framework, the problem is not ill-posed, because the optimal policy is unique, thanks to the regularization. For example, if one consider the negative entropy as the regularizer, with a null reward, the optimal policy will be that of maximum entropy, so the uniform policy, and it is unique.

Notice that if for a reward, the associated regularized optimal policy is unique, the converse is not true. For example, the uniform policy is optimal for any constant reward. More generally, reward shaping (Ng et al., 1999) still holds true for regularized MDPs (this being thanks to the results of Sec. 3, again).

This being said, assume that the model (dynamic, discount factor, regularizer) and that the optimal regularized policy π ∗ , Ω are known. It is possible to retrieve a reward function such that π ∗ , Ω is the unique optimal policy.

Proposition 5. Let ˆ q ∈ R S×A be any function satisfying

<!-- formula-not-decoded -->

then the reward ˆ r ( s, a ) defined as

<!-- formula-not-decoded -->

has π ∗ , Ω as the unique corresponding optimal policy.

Proof. First, recall (see Prop. 1) that if for any q , there is a unique regularized greedy policy, the converse is not true (simply by the fact that for any v ∈ R S , we have ∇ Ω ∗ ( q ( s, · ) + v ( s )) = ∇ Ω ∗ ( q ( s, · )) ). By assumption and by uniqueness of regularized greediness, π ∗ , Ω is the unique regularized policy corresponding to ˆ q . Then, with the above defined reward function, ˆ q is unique the solution of the regularized Bellman optimality equation. This shows the stated result.

If this result shows that IRL is well-defined in a regularized framework, it is not very practical (for example, in the entropic case, it tells that ˆ r ( s, a ) = ln π ∗ , Ω ( s, a ) is such a reward function). Yet, we think that the proposed general framework could lead to more practical algorithm.

For example, many IRL algorithms are based on the maximum-entropy principle, eg. (Ziebart et al., 2008; Finn et al., 2016; Fu et al., 2018). This maximum-entropy IRL framework can be linked to probabilistic inference, that can itself be shown to be equivalent, in some specific cases (deterministic dynamics), to entropy-regularized reinforcement learning (Levine, 2018). We think this to be an interesting connection, and maybe that our proposed regularized framework could allow to generalize or analyze some of these approaches.

## D.5. Regularized zero-sum Markov games

A zero-sum Markov game can be seen as a generalization of MDPs. It is a tuple {S , A 1 , A 2 , P, r, γ } with S the state space common to both players, A j the action space of player j , P ∈ ∆ S×A 1 ×A 2 S the transition kernel ( P ( s ′ | s, a 1 , a 2 ) is the probability of transiting to state s ′ when player 1 played a 1 and player 2 played a 2 in s ), r ∈ R S×A 1 ×A 2 the reward function of both players (one tries to maximize it, the other one to minimize it) and γ the discount factor. We write µ ∈ ∆ S A 1 a policy of the maximizer, and ν ∈ ∆ S A 2 a policy of the minimizer.

As in the case of classical MDPs, everything can be constructed from an evaluation operator, defined as

<!-- formula-not-decoded -->

of fixed point v µ,ν . From this, the following operators are defined:

<!-- formula-not-decoded -->

We also define the greedy operator as µ ∈ G ( v ) ⇔ Tv = T µ v = min ν T µ,ν v . Thanks to the Von Neumann's minimax theorem (Morgenstern &amp; Von Neumann, 1953), we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The modified policy iteration for this kind of games is

<!-- formula-not-decoded -->

As in MDPs, we can regularize the evaluation operator, and construct regularized zero-sum Markov games from this. Let Ω 1 and Ω 2 be two strongly convex reguralizers on ∆ A 1 and ∆ A 2 , and define the regularized evaluation operator as

<!-- formula-not-decoded -->

From this, we can construct a theory of regularized zero-sum Markov games as it was done for MDPs. The Von Neumann's minimax theorem does not only hold for affine operators, but for convex-concave operators, so we're fine.

Notably, the unregularized error propagation analysis of (14) by Perolat et al. (2015) could be easily adapted to the regularized case (much like how the analysis of AMPI directly led to the analysis of Sec. 4.2, thanks to the results of Sec. 3). We left its extension to regularization with a Bregman divergence as future work.

and the optimal value function satisfies