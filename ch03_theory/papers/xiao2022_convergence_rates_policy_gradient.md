## On the Convergence Rates of Policy Gradient Methods

## Lin Xiao

Meta AI Research Seattle, WA 98109, USA

linx@fb.com

## Abstract

We consider infinite-horizon discounted Markov decision problems with finite state and action spaces and study the convergence rates of the projected policy gradient method and a general class of policy mirror descent methods, all with direct parametrization in the policy space. First, we develop a theory of weak gradient-mapping dominance and use it to prove sharper sublinear convergence rate of the projected policy gradient method. Then we show that with geometrically increasing step sizes, a general class of policy mirror descent methods, including the natural policy gradient method and a projected Q-descent method, all enjoy a linear rate of convergence without relying on entropy or other strongly convex regularization. Finally, we also analyze the convergence rate of an inexact policy mirror descent method and estimate its sample complexity under a simple generative model.

Keywords: discounted Markov decision problem, policy gradient, gradient domination, policy mirror descent, sample complexity.

## 1. Introduction

Markov decision process (MDP) is a fundamental model for sequential decision-making. In this paper, we consider infinite-horizon, discounted Markov decision problems (DMDPs) with finite state and action spaces. They are specified as a 5-tuple ( S , A , P, R, γ ), where S is a finite state space with cardinality |S| , A is a finite action space with cardinality |A| , P is a transition probability function with P ( s ′ | s, a ) denoting the probability of transitioning to s ′ when taking action a from state s , R : S × A → [0 , 1] is a reward function with R s,a or R ( s, a ) being the (expected) reward of taking action a from state s , and finally γ ∈ [0 , 1) is a discount factor applied to the reward one-step in the future.

Starting from an initial state s 0 ∈ S , an agent takes an action a t ∈ A at each time step t = 0 , 1 , 2 , . . . , which leads to the next state s t +1 with probability P ( s t +1 | s t , a t ), and obtains the immediate reward r t = R ( s t , a t ). Such interactions generate a trajectory

<!-- formula-not-decoded -->

The goal of the agent is to find a policy of choosing the actions a 0 , a 1 , a 2 , . . . that maximizes the discounted cumulative reward E [∑ ∞ t =0 γ t r t ] . Here the expectation is taken with respect to the possible randomness in s 0 , any randomness in choosing the actions a t , and the randomness of state transitions prescribed by P .

In general, a policy that determines the action at time t may depends on the whole history of the trajectory up to time t . A stationary policy π specifies a decision rule that depends only on the current state. Specifically, we let π s ∈ ∆( A ) be the decision rule at state s , where ∆( A ) denotes the probability simplex supported on A , and π s,a denotes the

probability of taking action a at state s . The value of a stationary policy π ∈ ∆( A ) |S| starting from an arbitrary state s is defined as

<!-- formula-not-decoded -->

where the expectation is taken with respect to a t ∼ π s t and s t +1 ∼ P ( ·| s t , a t ) for all t ≥ 0. We define V : ∆( A ) |S| → R |S| as a vector-valued function with components V s ( π ). By the assumption that R ( s, a ) ∈ [0 , 1] for all ( s, a ) ∈ S × A , we immediately have

<!-- formula-not-decoded -->

The conventional formulation of DMDP is about maximizing the discounted total reward. In this paper, we adopt a minimization formulation in order to better align with conventions in the optimization literature. To this end, we regard each R ( s, a ) ∈ [0 , 1] as a value measuring regret rather than reward. Given a reward matrix R , we can reset R ( s, a ) ← 1 -R ( s, a ) for all ( s, a ) ∈ S ×A to turn it into a regret matrix. Suppose ρ ∈ ∆( S ) is an arbitrary initial state distribution. We consider the problem of minimizing

<!-- formula-not-decoded -->

For infinite-horizon DMDPs with finite state and actions spaces, there exists a (deterministic) stationary policy π /star that is simultaneously optimal in minimizing V s ( · ) for all s ∈ S (e.g., Puterman, 1994, Section 6.2.4). Such a solution is insensitive to the choice of ρ .

In this paper, we focus on policy gradient methods for minimizing the weighted value function V ρ . These methods generate a sequence of policies { π ( k ) } through repeated evaluation of the policy gradient ∇ V µ , where µ ∈ ∆( S ) is not necessarily equal to ρ . The most straightforward variant is the projected policy gradient method,

<!-- formula-not-decoded -->

where η k is the step size, Π := ∆( A ) |S| is the set of feasible policies, and proj Π ( · ) denotes projection onto Π in the Euclidean norm. More generally, policy gradient methods can be derived from the mirror-descent form

<!-- formula-not-decoded -->

where D k ( · , · ) is a distance-like function that may depend on π ( k ) . For example, setting D k as the squared Euclidean distance yields the projected policy gradient method (4). Shani et al. (2020) showed that by setting D k as an appropriately weighted Kullback-Leibler (KL) divergence, one recovers the natural policy gradient (NPG) method of Kakade (2001). In general, we can think of (5) as a class of preconditioned policy gradient methods. The main results of this paper concern the convergence rates of such methods.

## 1.1 Previous Work

Many classical algorithms for DMDP are based on dynamical programming (Bellman, 1957), including value iteration, policy iteration, temporal difference learning and Q-learning (see, e.g., Puterman, 1994; Bertsekas and Tsitsiklis, 1996; Sutton and Barto, 2018). Analyses of these methods in the tabular case mostly rely on the contraction property of the Bellman operator, which are difficult to extend with nonlinear function approximation and policy parametrization. In contrast, policy gradient methods (Williams, 1992; Sutton et al., 2000; Konda and Tsitsiklis, 2000; Kakade, 2001) aim to find a local minimum of an expected value function, thus are applicable to any differentiable policy parametrization and admit easy extensions to function approximation. In particular, they appear to work well when parametrized with modern deep neural networks (Schulman et al., 2015, 2017).

Despite the long history and empirical successes of policy gradient methods, their convergence properties are not well understood until recently. For example, it was widely accepted that they converge asymptotically to a stationary point or a local minimum because the objective function is nonconvex in general. However, Fazel et al. (2018) show that for linear quadratic control problems, policy gradient methods converge to the global optimal solution despite the nonconvex cost function, thanks to a gradient dominance property (Polyak, 1963). Agarwal et al. (2021) derive a variational gradient-dominance property and use it to obtain global convergence of the projected policy gradient method (4). Bhandari and Russo (2019) identify more general structural properties of policy gradient methods to ensure gradient domination and hence convergence to global optimum.

Using direct policy parametrization (over π ∈ ∆( A ) |S| ), Agarwal et al. (2021) show that the projected policy gradient method (4) converges to a global optimum at an O (1 / √ k ) sublinear rate. Specifically, the number of iterations to obtain V ρ ( π ( k ) ) -V /star ρ ≤ /epsilon1 is

<!-- formula-not-decoded -->

where d ρ ( π /star ) ∈ ∆( S ) is a discounted state-visitation distribution and ∥ ∥ d ρ ( π /star ) /µ ∥ ∥ ∞ is a distribution mismatch coefficient (see Section 2.1 for definition and explanation). Zhang et al. (2020) develop a variational policy gradient framework and use it to show that the projected policy gradient method converges to global optimum at a faster O (1 /k ) rate. In both cases, the constants in the iteration complexity are very large and depend on the Lipschitz constant characterizing the smoothness of the objective function.

Shani et al. (2020) show that the natural policy gradient (NPG) method (Kakade, 2001) can be cast as a special case of policy mirror descent method (5) and has an O (1 / √ k ) convergence rate. Agarwal et al. (2021) improve the convergence rate of NPG to O (1 /k ); more concretely, the number of iterations to obtain V ρ ( π ( k ) ) -V /star ρ ≤ /epsilon1 is

<!-- formula-not-decoded -->

which is independent of the dimensions |S| and |A| or any distribution mismatch coefficient. Interestingly, the step sizes that guarantee such a rate can be chosen arbitrarily large, regardless of the Lipschitz constant of the policy gradient.

With entropy regularization (added to the DMDP objective), Cen et al. (2020) show that the NPG method has linear (geometric) convergence. Their approach rely on the contraction property of a generalized Bellman operator and the convergence guarantees are in terms of the infinity norm of the 'soft' Q -functions. With appropriate choice of the regularization parameter and step size, they obtain iteration complexity on the order of

<!-- formula-not-decoded -->

Lan (2021) proposes a general policy mirror descent method that is similar to (5) with either convex or strongly convex regularizations. He focuses on the case of minimizing V ρ /star where ρ /star is the stationary distribution of the MDP under the optimal policy π /star , which avoids any distribution mismatch coefficient in the analysis. In order to guarantee V ρ /star ( π ( k ) ) -V /star ρ /star ≤ /epsilon1 , Lan (2021) obtains iteration complexity on the orders of (7) and (8) for the settings without and with entropy regularization, respectively. More interestingly, Lan (2021) also obtained linear convergence for the un-regularized DMDP using diminishing regularization combined with increasing step sizes (while maintaining a constant product of the two).

More recently, Zhan et al. (2021) extend the framework of Lan (2021) to accommodate a broader class of convex regularizers including those that are nonsmooth. For un-regularized DMDP, Khodadadian et al. (2021) show that the NPG method can obtain linear convergence with an adaptive step-size rule, and Bhandari and Russo (2021) show that several variants of policy gradient methods has linear convergence with exact line search.

For the exact policy gradient method with softmax parametrization, Agarwal et al. (2021) show that it converges asymptotically to a global optimum, and attains an O (1 / √ k ) rate with log barrier regularization. Mei et al. (2020) derive an O (1 /k ) convergence rate and Mei et al. (2021) further improve it to linear convergence by exploiting non-uniform variants of the smoothness and gradient dominance properties. However, these fast rates are associated with problem-dependent constants that can be very large (Li et al., 2021).

## 1.2 Contributions and Outline

In this paper, we present a systematic study of policy gradient methods with direct policy parametrization, focusing on their convergence rates for minimizing V ρ over π ∈ ∆( A ) |S| .

Section 2 contains an overview of structural properties of DMDP that are well-known but essential for the main results of the paper.

In Section 3, we develop a theory of weak gradient-mapping domination for general nonconvex composite optimization, and use it to obtain an O (1 /k ) convergence rate for the projected policy gradient method. Concretely, our result on iteration complexity replaces /epsilon1 -2 in (6) with /epsilon1 -1 and (1 -γ ) -6 with (1 -γ ) -5 . Although this result is the same as the one obtained by Zhang et al. (2020), our analysis are quite different. Zhang et al. (2020) exploit the bijection structure of the primal-dual DMDP formulations, while we derive this result as a special case of nonconvex optimization with weak gradient-mapping domination which, to our best knowledge, is new and of independent interest.

In Section 4, we study exact policy mirror descent methods of the form (5). First, we show that with a constant step size (which can be arbitrarily large), they obtain the same dimension-free iteration complexity (7). This result extend the one of Agarwal et al. (2021) on NPG with KL-divergence to a general class of Bregman divergences, including a

projected Q -descent method derived with squared Euclidean distance. Second, we show that with geometrically increasing step sizes, as simple as η k +1 = η k /γ , policy mirror descent methods enjoy linear convergence without relying on any regularization. Specifically, their iteration complexity for reaching V ρ ( π ( k ) ) -V /star ρ ≤ /epsilon1 is

If ρ is set to be the stationary distribution under the optimal policy π /star , then the distribution mismatch coefficient ∥ ∥ d ρ ( π /star ) /ρ ∥ ∥ ∞ = 1 and we recover (8). In addition, we discuss conditions for superlinear convergence and make connections with the classical Policy Iteration method.

<!-- formula-not-decoded -->

In Section 5, we investigate the iteration complexity of inexact policy mirror descent methods and show that the geometrically increasing step sizes do not cause instability even with errors in evaluating the policy gradients or Q -functions. They converge with the same linear rate up to an asymptotic error floor. With a simple Q -estimator by repeated simulation of truncated trajectories, we obtain a sample complexity of

<!-- formula-not-decoded -->

where the notation ˜ O ( · ) hides poly-logarithmic factors of |S||A| , 1 / (1 -γ ) and 1 //epsilon1 . Finally, in Section 6, we discuss the limitations of our work and possible extensions.

## 2. Preliminaries on DMDP

In this section, we overview the structural properties of DMDP that are essential for the developments in later sections. We start with a few definitions. Let ∆( S ) denote the probability simplex defined over the state space S , i.e.,

Similarly, ∆( A ) denotes the probability simplex over the action space A . The set of admissible policies for DMDP is defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- P : Π → R |S|×|S| : a matrix function with entries P s,s ′ ( π ) = ∑ a ∈A π s,a P ( s ′ | s, a );

With slight abuse of notation, we define the following functions of π ∈ Π:

- r : Π → R |S| : a vector function with components r s ( π ) = ∑ a ∈A π s,a R s,a .

Using the definitions above, the value function V : Π → R |S| , whose components V s are defined in (1), admits the following analytic form (see, e.g., Puterman, 1994, Section 6.1)

<!-- formula-not-decoded -->

Since P ( π ) is a row stochastic matrix and 0 ≤ γ &lt; 1, the spectral norm of γP ( π ) is strictly less than one (by the Perron-Frobenius theorem) and thus I -γP ( π ) is always invertible. Given ρ ∈ ∆( S ), the weighted value function V ρ defined in (3) can be written as

<!-- formula-not-decoded -->

Here we treat ρ as a column vector and use matrix multiplication conventions.

## 2.1 Distribution Mismatch Coefficient

Starting from s ∈ S , the discounted state-visitation distribution under a policy π is a vector d s ( π ) ∈ ∆( S ) whose components are defined as

<!-- formula-not-decoded -->

The coefficient 1 -γ ensures that ∑ s ′ ∈S d s,s ′ ( π ) = 1. In fact, d s,s ′ ( π ) is the ( s, s ′ ) entry of the matrix (1 -γ )( I -γP ( π )) -1 . In other words, if we define e s ∈ R |S| with components e s,s ′ = 1 if s = s ′ and 0 otherwise, then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Given an initial state distribution ρ ∈ ∆( S ), we define d ρ ( π ) ∈ ∆( S ) with components

Some useful facts from the above definitions are:

<!-- formula-not-decoded -->

For any ρ, µ ∈ ∆( S ), we define the distribution mismatch of ρ from µ as

∥ ∥ with the convention 0 / 0 = 1. This is an asymmetric measure of mismatch and it is finite if and only if the support (set of indices with nonzero entries) of µ contains that of ρ . If µ is the uniform distribution, then the mismatch is bounded by |S| .

<!-- formula-not-decoded -->

The convergence properties of policy gradient methods often depend on the distribution mismatch coefficients between two discounted state-visitation distributions (e.g., Kakade and Langford, 2002; Agarwal et al., 2021). According to (13), we have for any ρ, µ ∈ ∆( S ) and π, π ′ ∈ Π,

Our results in this paper mostly concern the case with ρ = µ and π = π /star , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In order for C /star ρ to be finite, it suffices to assume ρ &gt; 0, which means ρ s &gt; 0 for all s ∈ S .

The distribution mismatch coefficient is closely related to the concentrability coefficients in the analysis of approximate dynamic programming algorithms (Munos, 2003, 2005; Szepesv´ ari and Munos, 2008). In fact, C /star ρ is considered the 'best' one among all concentrability coefficients in the sense that it does not impose any restrictions on the MDP dynamics and it can be finite when other concentrability coefficients are infinite (Scherrer, 2014). See Agarwal et al. (2021, Section 2) for further discussions.

If ρ is chosen as the stationary distribution of the MDP under the optimal policy π /star , denoted as ρ /star , then we have d ρ /star ( π /star ) = ρ /star and hence C /star ρ /star = 1. This is the setting adopted by Liu et al. (2019) and Lan (2021), which leads to simplified analysis for minimizing V ρ /star . For DMDP with entropy regularization (Lan, 2021; Cen et al., 2020), the resulting ρ /star always have full support over S . However, in general ρ /star may not have full support over S unless the underlying MDP is ergodic (Puterman, 1994, Section A.2).

## 2.2 Q -functions, Policy Gradient and Performance Difference Lemma

For each pair ( s, a ) ∈ S × A , the state-action value function Q s,a : Π → R is defined as where the expectation is taken with respect to a t ∼ π s t and s t +1 ∼ P ( ·| s t , a t ) for all t ≥ 0. It is straightforward to verify that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let Q s ( π ) ∈ R |A| denote the vector with components Q s,a ( π ) for all a ∈ A . Then,

<!-- formula-not-decoded -->

where 〈· , ·〉 denotes the inner product of two vectors.

Policy gradients refer to the gradients of the value functions V s ( π ) and V ρ ( π ). We can obtain their expressions as special cases of the policy gradient theorem (Sutton et al., 2000) which covers the general case with policy parametrization. For easy reference, we give a simple, self-contained derivation in the Appendix (Section A.1). Specifically, we have

<!-- formula-not-decoded -->

and ∇ V ρ is the concatenation of ∇ s V ρ for all s ∈ S . In other words, policy gradients are weighted Q -functions where the weights are block-diagonal and proportional to the discounted state-visitation probabilities.

A fundamental result for analyzing DMDP and related algorithms is the performance difference lemma of Kakade and Langford (2002). In this paper, we mostly rely on the following variant, which has appeared in Liu et al. (2019) and Lan (2021). For completeness, here we provide an alternative proof.

Lemma 1 (Performance difference lemma) For any π, ˜ π ∈ Π , it holds that

<!-- formula-not-decoded -->

Proof Using the relation (17) on both π and ˜ π and the definition of Q -functions, we obtain

<!-- formula-not-decoded -->

Define u ∈ R |S| with components u s = 〈 Q s (˜ π ) , π s -˜ π s 〉 . Then the above result leads to which further implies

<!-- formula-not-decoded -->

Using the expression of d s,s ′ ( π ) in (12), we write the above equality component-wise as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is the same as (19).

The weighted version of the performance difference lemma is,

<!-- formula-not-decoded -->

Considering the expression of policy gradient in (18), the above characterization resembles that of a linear function (precisely so if the expectation were taken with respect to s ′ ∼ d ρ (˜ π ) instead of s ′ ∼ d ρ ( π )). The performance difference lemma is directly responsible for variational gradient domination (Agarwal et al., 2021, Lemma 4), which is a convexity-like property, and also for descent with arbitrarily large step sizes (see Section 4), which is a concavity-like property.

## 3. Projected Policy Gradient Method

In this section, we analyze the projected policy gradient method for solving the problem

<!-- formula-not-decoded -->

where V ρ ( π ) is defined in (3) or equivalently (10). We assume that the policy gradients are computed with respect to an initial state distribution µ , which may be different from the performance evaluation distribution ρ .

Starting from an initial policy π (0) ∈ Π, the projected policy gradient method generates a sequence π ( k ) for k = 1 , 2 , . . . as follows:

where η k is the step size and proj Π ( · ) denotes projection onto Π in Euclidean norm, i.e., proj Π ( π ) = arg min π ′ ∈ Π ‖ π ′ -π ‖ 2 2 . Since Π = ∆( A ) |S| is a Cartesian product, the projections associated with different states can be done separately:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Agarwal et al. (2021, Theorem 5) show that with a constant step size η k = (1 -γ ) 3 2 γ |A| for all k ≥ 0, the projected policy gradient method converges at an O ( 1 / √ k ) rate. More precisely, where ∇ s V µ is given by (18) with ρ replaced by µ .

min 0 &lt;k ≤ K { V ρ ( π ( k ) ) -V /star ρ } ≤ /epsilon1 whenever K &gt; 64 γ |S||A| /epsilon1 2 (1 -γ ) 6 ∥ ∥ ∥ ∥ d ρ ( π /star ) µ ∥ ∥ ∥ ∥ 2 ∞ . (24) The following two ingredients are key to their analysis.

<!-- formula-not-decoded -->

- Smoothness (Agarwal et al., 2021, Lemma 54): For any π, π ′ ∈ Π, it holds that
- Variational gradient domination (Agarwal et al., 2021, Lemma 4): For any π ∈ Π,

Here 'variational' refers to the term max π ′ ∈ Π 〈∇ V µ ( π ) , π -π ′ 〉 , which is different from ‖∇ V µ ( π ) ‖ as in gradient dominance conditions for unconstrained optimization.

<!-- formula-not-decoded -->

Based on the same two results above, we show that the projected policy gradient method enjoys a faster O (1 /k ) convergence rate. The following theorem holds for the case ρ = µ .

Theorem 2 Suppose ρ = µ and the step size η k = (1 -γ ) 3 2 γ |A| for all k ≥ 0 . Then the projected policy gradient method (22) generates a sequence of policies π ( k ) satisfying

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

The general case with ρ = µ can be handled with an additional distribution mismatch coefficient. Concretely,

Then applying Theorem 2 with ρ replaced by µ yields

<!-- formula-not-decoded -->

We prove Theorem 2 as a special case of a more general result on gradient-mapping domination , which we present next.

Comparing with (24), in addition to improving the rate from O (1 / √ k ) to O (1 /k ), our bound also has better dependence on the discount factor: 1 / (1 -γ ) 5 as opposed to 1 / (1 -γ ) 6 . However, our bound uses a different distribution mismatch coefficient and has an additional factor of ‖ ρ/µ ‖ ∞ .

## 3.1 An Interlude on Gradient-Mapping Domination

In this section, we consider the following composite optimization problem

<!-- formula-not-decoded -->

where f is smooth and Ψ is convex and lower semi-continuous. More specifically, we assume that there exists a constant L &gt; 0 such that

<!-- formula-not-decoded -->

The MDP formulation in (21) is a special case of (27) with the mappings x ← π , f ← V ρ and Ψ as the indicator function of Π, i.e., Ψ( π ) = 0 if π ∈ Π and ∞ otherwise.

For any convex function φ , the prox operator is defined as

<!-- formula-not-decoded -->

A generic algorithm for solving problem (27) is the proximal gradient method :

<!-- formula-not-decoded -->

where η k is the step size. If Ψ is the indicator function of Π, then prox η k Ψ becomes the projection operator proj Π for any η k &gt; 0. In this section, we focus on the proximal gradient method with the constant step size η k = 1 /L . To simplify presentation, we define

<!-- formula-not-decoded -->

thus the proximal gradient method (29) can be written simply as x ( k +1) = T L ( x ( k ) ). The gradient mapping associated with problem (27) is defined as

<!-- formula-not-decoded -->

In the special case Ψ ≡ 0, we have G L ( x ) = ∇ f ( x ) for any L &gt; 0. The norm of the gradient mapping, ‖ G L ( x ( k ) ) ‖ , can serve as a measure of closeness to a first-order stationary point.

Key to the convergence analysis of the proximal gradient method is the following descent property (Nesterov, 2013, Theorem 1),

<!-- formula-not-decoded -->

Summing up over k = 0 , 1 , . . . , K , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, together with the fact F ( x ( K +1) ) ≥ F /star , implies (Beck, 2017, Theorem 10.15)

The O (1 / √ k ) convergence rate stated in (24) is obtained by combining (33) with (26), which is the approach taken by Agarwal et al. (2021) and Bhandari and Russo (2019).

We show that under similar conditions, the proximal gradient method actually enjoy a faster O (1 /k ) rate of convergence. To this end, the following notion of (weak) gradientmapping domination is a proper extension of (weak) gradient domination.

Definition 3 (weak gradient-mapping domination) Suppose F := f + Ψ where f is L -smooth and Ψ is proper, convex and closed. We say that F satisfies a weak gradientmapping dominance condition if there exists ω &gt; 0 such that

<!-- formula-not-decoded -->

where F /star = min x F ( x ) and T L and G L are defined in (30) and (31) respectively.

This weak version of gradient-mapping domination corresponds to the Kurdyka-/suppress Lojasiewicz (K/suppress L) condition with K/suppress L exponent 1, instead of the usual exponent 1 / 2 that leads to linear convergence (Kurdyka, 1998; Karimi et al., 2016; Li and Pong, 2018). We discuss the stronger notion of gradient-mapping dominance in Appendix A.2.

In the following theorem, we prove the O (1 /k ) convergence rate for problems satisfying weak gradient-mapping domination.

Theorem 4 Consider the problem of minimizing F := f + Ψ where f is L -smooth and Ψ is proper, convex and closed. Suppose F is weakly gradient-mapping dominant with parameter ω and let F /star = min x F ( x ) . Then the proximal gradient method (29) with a constant step size η k = 1 /L generates a sequence { x ( k ) } that satisfies, for all k ≥ 0 ,

<!-- formula-not-decoded -->

Proof Combining the descent property (32) with the inequality (34) yields

<!-- formula-not-decoded -->

Let δ k = F ( x ( k ) ) -F /star ≥ 0 Then we have

<!-- formula-not-decoded -->

We divide both sides of the above inequality by δ k δ k +1 to obtain

<!-- formula-not-decoded -->

Then telescoping sum over iterations 0 , 1 , . . . , k -1 yields

<!-- formula-not-decoded -->

Notice that due to the descent property, we always have δ i +1 ≤ δ i and thus δ i +1 /δ i ≤ 1.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Otherwise, we must have n ( k, r ) &lt; ck , which means that δ i +1 /δ i &lt; r at least /ceilingleft (1 -c ) k /ceilingright times. Noticing that δ i +1 /δ i ≤ 1 for all i , we arrive at

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the above two cases and using the fact that r, c ∈ (0 , 1) can be chosen arbitrarily, we conclude that

Simply setting r = c = 1 / 2 gives the desired result (35).

## 3.2 Proof of Theorem 2

In order to prove Theorem 2, we only need to verify that the weak gradient-mapping domination holds for the weighted value function V ρ . This is the result of the next lemma.

For any two constant r, c ∈ (0 , 1), let's define n ( k, r ) be the number of times that the ratio δ i +1 /δ i is at least r among the first k iterations. If n ( k, r ) ≥ ck , then δ i +1 /δ i ≥ r at least /ceilingleft ck /ceilingright times, thus which implies

Lemma 5 Consider the problem of minimizing V ρ over Π and suppose that V ρ is L -smooth. We have where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof Applying a result of Nesterov (2013, Theorem 1) to our setting yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the facts T L ( π ) ∈ Π and ‖ π ′′ -π ′ ‖ 2 ≤ √ 2 |S| for any π ′′ , π ′ ∈ Π, we obtain

Combining the above bound with (26) yields the desired result.

An argument equivalent to Lemma 5 was used by Agarwal et al. (2021) which relies on a result of Ghadimi and Lan (2016). Combining (36) with (33) gives the result in (24). On the other hand, we recognize that with ρ = µ , inequality (36) implies (34) with

<!-- formula-not-decoded -->

Now we can apply Theorem 4. Notice that in this case, the exponential decay part in (35) is always smaller than the sublinear part, which leads to V ρ ( π ( k ) ) -V /star ρ ≤ 4 L/ ( ωk ), i.e.,

This finishes the proof of Theorem 2.

<!-- formula-not-decoded -->

Zhang et al. (2020, Theorem 5) have also established the O (1 /k ) rate of the projected policy gradient method with direct parametrization. However, their proof appears to be quite different from ours, which leverages the dual linear programming parametrization (Puterman, 1994, Section 6.9). In contrast, our approach is based on a novel notion of weak gradient-mapping domination and applies to general nonconvex composite optimization problems.

## 4. Exact Policy Mirror Descent Methods

Mirror descent (Nemirovski and Yudin, 1983) is a general framework for the construction and analysis of optimization algorithms, which covers the projected gradient method as a special case. Here we adopt the form of mirror descent based on proximal minimization with respect to a Bregman divergence (Beck and Teboulle, 2003).

Let h : ∆( A ) → R be a strictly convex function and continuously differentiable on the relative interior of ∆( A ), denoted as rint ∆( A ). The Bregman divergence generated by h is a distance-like function defined as

<!-- formula-not-decoded -->

Two most popular examples of Bregman divergence are:

- Squared Euclidean distance, generated by the squared 2-norm:

<!-- formula-not-decoded -->

- Kullback-Leibler (KL) divergence, generated by the negative entropy:

<!-- formula-not-decoded -->

Notice that the gradient of negative entropy vanishes on the boundary of the simplex. Therefore we need to restrict the second argument p ′ to lie within the relative interior of ∆( A ). We shall address such subtleties later in the convergence analysis.

Recall that the set of feasible policies is Π = ∆( A ) |S| , which is a Cartesian product of |S| copies of ∆( A ). For any ρ ∈ ∆( S ), we define a weighted divergence function

<!-- formula-not-decoded -->

This function satisfies the basic properties of a Bregman divergence; in particular, it is nonnegative and equals to 0 if and only if π = π ′ .

Following the derivations of Shani et al. (2020), we consider policy mirror descent (PMD) methods with dynamically weighted divergences:

<!-- formula-not-decoded -->

where η k is the step size, µ ∈ ∆( S ) is an arbitrary state distribution and d µ ( π ( k ) ) is the discounted state-visitation distribution under the policy π ( k ) . Using the fact

<!-- formula-not-decoded -->

and plugging in the policy gradient formula (18), we obtain

<!-- formula-not-decoded -->

which can be written separately for each state as

<!-- formula-not-decoded -->

Notice that the above update rule is independent of the choice of µ . This is the result of adaptive preconditioning with a dynamically weighted divergence: the weight for each state in D d µ ( π ( k ) ) matches the coefficient of Q s ( π ( k ) ) in the policy gradient ∇ V µ ( π ( k ) ).

For the two prominent examples of Bregman divergence listed before, the corresponding PMD methods have closed-form update rules:

- Projected Q -descent. If D ( · , · ) is the squared Euclidean distance, then (37) becomes

Compared with the projected policy gradient method (23), we replaced the policy gradient ∇ s V µ ( π ( k ) ) by Q s ( π ( k ) ) as the result of adaptive preconditioning.

<!-- formula-not-decoded -->

- Exponentiated Q -descent. If D ( · , · ) is the KL-divergence, then (37) takes the form

where

This is exactly the Natural Policy Gradient (NPG) method (Kakade, 2001) expressed in the policy space (Agarwal et al., 2021).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the rest of this section, we investigate the convergence rate of the PMD method (37). We show that with a constant step size, it has O (1 /k ) convergence rate. When the step size increases exponentially as η k = η 0 /γ k , we have linear convergence and the convergnece rate depends on the distribution mismatch coefficient ‖ d ρ ( π /star ) /ρ ‖ ∞ . In addition, we discuss situations of super-linear convergence and connections to policy iteration.

Our results hold for PMD methods constructed with general Bregman divergences, matching or improving over the best known convergence rates. In particular, the projected Q -descent method has the same rate of convergence as NPG. We show that the key ingredient for fast convergence of the PMD method is the adaptive preconditioning using weighted divergence functions. The adopted local Bregman divergence, being KL-divergence or squared Euclidean distance, does not make much difference.

## 4.1 Sublinear Convergence

Our analysis is based on two key ingredients: the performance difference lemma (Lemma 1) and a three-point descent lemma on proximal optimization with Bregman divergences.

In order to cover both the squared Euclidean distance and KL-divergence without loss of rigor, we need some technical conditions. Specifically, we say a function h is of Legendre type (Rockafellar, 1970, Section 26) if it is essentially smooth and strictly convex in the

relative interior of dom h , denoted as rint dom h . Essential smoothness means that h is differentiable and ‖∇ h ( x k ) ‖ → ∞ for every sequence { x k } converging to a boundary point of dom h . The following result is a slight variation of Chen and Teboulle (1993, Lemma 3.2), where we replaced the original assumption of h being a Bregman function with h being of Legendre type. The proof essentially follows the same arguments and thus is omitted here.

Lemma 6 (Three-point descent lemma) Suppose that C ⊂ R n is a closed convex set, φ : C → R is a proper, closed convex function, D ( · , · ) is the Bregman divergence generated by a function h of Legendre type and rint dom h ∩C /negationslash = ∅ . For any x ∈ rint dom h , let

Then x + ∈ rint dom h ∩ C and for any u ∈ C ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the context of the PMD method (37), C = ∆( A ) and φ is the linear function η k 〈 Q s ( π ( k ) ) , · 〉 . There are some subtle differences between the two Bregman divergences we consider, as explained below.

- For the squared Euclidean distance, h ( · ) = (1 / 2) ‖ · ‖ 2 2 is of Legendre type with rint dom h = R |A| and thus rint dom h ∩ C = ∆( A ). Therefore each iterate generated by the PMD method, specifically (38), can be on the boundary of ∆( A ).
- For the KL divergence, h is the negative entropy function, which is also of Legendre type, but with rint dom h ∩ C = rint dom h = rint ∆( A ). Therefore, if we start with an initial point in rint ∆( A ), then every iterates will stay in rint ∆( A ).

We first use Lemma 6 to prove a descent property of PMD. This result is elementary and has appeared in various forms before (e.g., Liu et al., 2019; Lan, 2021). We present the proof for completeness as we will need to refer to some intermediate steps in it later.

Lemma 7 (Descent property of PMD) Suppose the initial point π (0) ∈ rint Π . Then the sequences generated by the PMD method (37) satisfy and for any ρ ∈ ∆( s ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof Applying Lemma 6 to the update rule (37) with C = ∆( A ) and φ ( · ) = η k 〈 Q s ( π k ) , · 〉 , we obtain that for any p ∈ ∆( A ),

Rearranging terms and dividing both sides by η k , we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Letting p = π ( k ) s in (42) yields

<!-- formula-not-decoded -->

which implies (40) since the Bregman divergence D ( · , · ) is always nonnegative. By the performance difference lemma, specifically the weighted version (20), we have

<!-- formula-not-decoded -->

which is the same as (41).

The next result is a generalization of the O (1 /k ) convergence rate of the NPG method obtained by Agarwal et al. (2021, Theorem 16), where they focused on the setting of KLdivergence and their proof also relies on specific properties of the KL-divergence. Here we extend it to more general Bregman divergence. Lan (2021, Theorem 2) derived a similar result using techniques that works for general Bregman divergence. However, he worked with the special objective function V ρ /star where ρ /star is the stationary distribution of the optimal policy π /star . As a result, the proof of Lan (2021, Theorem 2) avoids some subtle arguments required for the more general objective function V ρ where ρ ∈ ∆( S ) can be arbitrary.

In order to simplify presentation, we use the following notation throughout this paper:

<!-- formula-not-decoded -->

where d ρ ( π /star ) is the state-visitation distribution under π /star with initial state distribution ρ . Although ρ does not appear in the notation D /star k , we hope it is clear from the context.

Theorem 8 Consider the policy mirror descent method (37) with π (0) ∈ rint Π and constant step size η k = η for all k ≥ 0 . For any ρ ∈ ∆( S ) , we have for all k ≥ 0 ,

<!-- formula-not-decoded -->

Proof Consider the inequality (42), we let p = π /star s and subtract and add π ( k ) s within the inner product term, which leads to

<!-- formula-not-decoded -->

Notice that we dropped the nonnegative term (1 /η k ) D ( π ( k +1) s , π ( k ) s ) on the left side of the inequality. Taking expectation with respect to the distribution d ρ ( π /star ) on both sides of the above inequality and using the notation in (43), we obtain

<!-- formula-not-decoded -->

For the first expectation in (44), we have

<!-- formula-not-decoded -->

where the inequality holds because of (40) and the fact, due to (13), that

<!-- formula-not-decoded -->

The last equality in (45) is due to the performance difference lemma. For the second expectation in (44), we again use the performance difference lemma to obtain

<!-- formula-not-decoded -->

Substituting the two results above into (44) leads to

<!-- formula-not-decoded -->

Setting η k = η for all k ≥ 0 and summing up over k :

<!-- formula-not-decoded -->

Since V ρ ( π ( k ) ) is monotone non-increasing in k (see Lemma 7), we conclude that

<!-- formula-not-decoded -->

Finally, bounding V d ρ ( π /star ) ( π (0) ) by 1 / (1 -γ ) as in (2) gives the desired result.

As a result of Theorem 8, whenever η ≥ (1 -γ ) D d ρ ( π /star ) ( π /star , π (0) ), we have

<!-- formula-not-decoded -->

In other words, the number of iterations to reach V ρ ( π ( k ) ) -V /star ρ ≤ /epsilon1 is at most

<!-- formula-not-decoded -->

which is independent of the problem dimensions |S| and |A| . More specifically,

- For the projected Q -descent method (38), since D ( π s , π ′ s ) = (1 / 2) ‖ π s -π ′ s ‖ 2 ≤ 1 for any π s , π ′ s ∈ ∆( A ), we have D ρ ( π, π ′ ) = ∑ s ∈S ρ s D ( π s , π ′ s ) ≤ 1 for any ρ ∈ ∆( S ). Therefore in order for (47) to hold, it suffices to have η ≥ (1 -γ ).
- For the exponentiated Q -descent method (39), if we choose the uniform initial policy, i.e., π (0) s,a = 1 / |A| for all ( s, a ) ∈ S × A , then D ρ ( π /star , π (0) ) ≤ log |A| for all ρ ∈ ∆( S ). Therefore in order for (47) to hold, it suffices to have η ≥ (1 -γ ) log |A| .

The above analysis indicates that the projected Q -descent method may have a slight advantage over the exponentiated variant (NPG) in terms of having a wider range of η to enjoy the same dimensional independent convergence guarantee (47).

A more curious fact is that for both variants, the step size η does not have an upper bound and can be as large as possible. This is in contrast to the classical analysis of smooth optimization, where the step size is usually upper bounded by 2 /L with L being the Lipschitz constant of the gradient; see, e.g., the approach taken in Section 3.1. Here the fact the step sizes can be arbitrarily large is due to the unique structure of DMDP. Indeed, we show next that PMD has linear convergence if the step size grows exponentially.

## 4.2 Linear Convergence

Consider again the policy mirror descent algorithm (37). In order to simplify the presentation, we define two more notations: the optimality gap

<!-- formula-not-decoded -->

which is always nonnegative, and the per-iteration distribution mismatch coefficient

∥ ∥ The following result is the basis for establishing the linear convergence and also for discussions on possible superlinear convergence.

<!-- formula-not-decoded -->

Proposition 9 Consider the policy mirror descent method (37) with π (0) ∈ rint Π and η k &gt; 0 for all k ≥ 0 . Then for any ρ ∈ ∆( S ) , we have for all k ≥ 0 , where δ k , ϑ k and D /star k are defined in (48) , (49) and (43) , respectively.

<!-- formula-not-decoded -->

Proof We start with the inequality (44) and bound the first expectation as follows:

<!-- formula-not-decoded -->

where the inequality holds because of (40), and the last equality is due to the performance difference lemma, specifically (20). Substituting the above bound and (46) into (44) and dividing both sides by 1 -γ yield the desired result.

The next theorem is our main result on linear convergence. The convergence rate depends on the performance evaluation distribution ρ through the following quantity:

<!-- formula-not-decoded -->

which is an upper bound on ϑ k for all k ≥ 0.

Theorem 10 Consider the policy mirror descent method (37) with π (0) ∈ rint Π . Suppose the step sizes satisfy η 0 &gt; 0 and

<!-- formula-not-decoded -->

then we have for each k ≥ 0 ,

<!-- formula-not-decoded -->

Proof Using (13), specifically d ρ,s ( π ( k ) ) ≥ (1 -γ ) ρ s for all s ∈ S , we have ϑ k ≤ ϑ ρ for all k ≥ 0. In addition, by Lemma 7, we have δ k +1 -δ k ≤ 0 for all k ≥ 0. Therefore (50) still holds if we replace ϑ k +1 by its upper bound ϑ ρ , i.e.,

<!-- formula-not-decoded -->

Dividing both sides by ϑ ρ and rearranging terms, we obtain

<!-- formula-not-decoded -->

If the step sizes satisfy (52), i.e., η k +1 ( ϑ ρ -1) ≥ η k ϑ ρ , then we have

<!-- formula-not-decoded -->

This forms a recursion and results in

<!-- formula-not-decoded -->

Finally, using the fact ϑ ρ ≥ 1 / (1 -γ ), we derive

<!-- formula-not-decoded -->

Substituting the above bound into the right side of (54), and considering the nonnegativity of D /star k on the left side, we arrive at the desired bound (53).

The exact value of ϑ ρ is hard to estimate in practice, which hinders the use of the step size rule (52). However, we can replace it with the more aggressive increasing rule

<!-- formula-not-decoded -->

which always implies (52). To see this, we use ϑ ρ ≥ 1 / (1 -γ ) to derive

According to Theorem 10, in order to guarantee V ρ ( π ( k ) ) -V /star ρ ≤ /epsilon1 , the required number of iterations of the PMD method is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the bound (2) and assuming η 0 ≥ 1 -γ γ D /star 0 , the iteration complexity becomes

Next we discuss a special choice of the performance evaluation distribution ρ .

Special case of ρ = ρ /star . Let ρ /star ∈ ∆( S ) be the stationary state distribution of the MDP under the optimal policy π /star . If the MDP starts with s ∼ ρ /star and following π /star , then the visit probability at every step is ρ /star and so is the discounted sum of them. Therefore we have d ρ /star ( π /star ) = ρ /star , which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In this case, with the step size rule η 0 ≥ 1 -γ γ D /star 0 and η k +1 ≥ η k /γ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the iteration complexity for V ρ /star ( π ( k ) ) -V ρ /star ( π /star ) ≤ /epsilon1 is dimension-independent:

However, unless the MDP is ergodic, the support of ρ /star may not cover the full state space S .

Several recent work studied policy mirror descent method for entropy-regularized MDP and obtained similar linear convergence rates (Cen et al., 2020; Lan, 2021; Zhan et al., 2021). With entropy regularization, the resulting MDP is always ergodic and the support of any stationary distribution covers the full state space S , i.e., ρ /star &gt; 0. Lan (2021) only considers ρ /star as the performance evaluation distribution; Cen et al. (2020) and Zhan et al. (2021) rely on the contraction properties of a generalized Bellman operator and obtain guarantees of the form ‖ Q ( π ( k ) ) -Q ( π /star ) ‖ ∞ ≤ /epsilon1 where Q is the 'soft' Q -function with regularization. Our analysis closely resembles that of Lan (2021), with the following differences:

- We consider the standard DMDP and show that linear convergence can be obtained without entropy regularization. Since the support of ρ /star may not cover the entire state space, we give a general analysis for any ρ ∈ ∆( S ) and characterize the convergence rate in terms of the distribution mismatch coefficient ‖ d ρ ( π /star ) /ρ ‖ ∞ .
- For DMDP without regularization, Lan (2021) also obtains a slower linear convergence rate ( γ k/ 2 instead of γ k ), through an approximate policy mirror descent (APMD) method. This method employs exponentially diminishing regularization and exponentially increasing step sizes, and the analysis is considerably more technical.

## 4.3 Superlinear Convergence

Under additional conditions, the PMD method (37) may exhibit superlinear convergence. We revisit Proposition 9 and start by rewriting the inequality (50) as

<!-- formula-not-decoded -->

If the step sizes satisfy η k ≥ ϑ k ϑ k +1 -1 η k -1 starting with some η -1 &gt; 0, then we have

Therefore, we have superlinear convergence of δ k if ϑ k → 1.

<!-- formula-not-decoded -->

Recall the definition of ϑ k in (49), we have ϑ k → 1 if and only if d ρ ( π ( k ) ) → d ρ ( π /star ). Apparently, a sufficient condition is π ( k ) → π /star . However, this is hard to establish without additional assumptions, e.g., by assuming that the optimal policy π /star is unique. Alternatively, since D /star k = D d ρ ( π /star ) ( π /star , π k ) → 0 implies π k → π /star , a reasonable attempt is to show the convergence of D /star k by further leveraging (57). In particular, we can show

<!-- formula-not-decoded -->

Nevertheless, we list here two sufficient conditions for superlinear convergence that are weaker than directly assuming π ( k ) → π /star . Both conditions have been used to establish superlinear convergence of the classical Policy Iteration algorithm (Puterman, 1994, Corollary 6.4.10 and Theorem 6.4.8, respectively).

at the same speed as δ k → 0, which is at least linear with an uniform upper bound on ϑ k as we have done in Section 4.2. However, the step-size condition η k ≥ ϑ k ϑ k +1 -1 η k -1 implies that the factor 1 / ( ϑ k η k -1 ) itself converges at the same rate, thus we can not guarantee D /star k → 0.

- Convergence of the transition probability matrix P ( π k ). Specifically,

<!-- formula-not-decoded -->

where ‖ · ‖ is any matrix norm. Under this condition, we have d ρ ( π ( k ) ) → d ρ ( π /star ) and thus ϑ k → 1 because d ρ ( π ( k ) ) = ( I -γP ( π ( k ) ) ) -T ρ is a continuous function.

- There exists a finite constant C &gt; 0 such that for all k = 1 , 2 , . . .

<!-- formula-not-decoded -->

This condition is stronger than the previous one because we already established linear convergence of V ρ ( π k ) -V /star ρ . As a result, it leads to local quadratic convergence.

Khodadadian et al. (2021) showed that under a variant of the second condition above, the NPG method converges superlinerly. With entropy regularization, the optimal policy π /star is unique and Cen et al. (2020) established local quadratic convergence of the regularized PMD method.

## 4.4 Connection with Policy Iteration

Our analysis of the PMD method does not impose any upper bound on the step sizes: they can be either arbitrarily large constant (Section 4.1) or gemmetrically increasing (Section 4.2). If we allow η k →∞ for all iterations, the limit of the PMD method (37) becomes

<!-- formula-not-decoded -->

which is precisely the classical Policy Iteration method (e.g., Puterman, 1994; Bertsekas, 2012). In fact, our analysis still holds in the limiting case and the result corresponding to Theorem 10 is

<!-- formula-not-decoded -->

where δ k = V ρ ( π ( k ) ) -V /star ρ . Recall the definition of ϑ ρ in (51). If ρ = ρ /star , then we have ϑ ρ /star = 1 / (1 -γ ) and

<!-- formula-not-decoded -->

which has the same convergence rate as Policy Iteration (e.g., Puterman, 1994; Ye, 2011). In general, we have the trivial bound which leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This convergence rate is the same as that established for several variants of policy gradient methods by Bhandari and Russo (2021, Theorem 1), which requires exact line search. Khodadadian et al. (2021) show that the NPG method with an adaptive step size rule can also achieve linear convergence. In contrast, our results in Section 4.2 show that the simple, non-adaptive step size schedule of η k = η 0 /γ k is suffice to obtain linear convergence of a general class of policy mirror descent methods.

## 5. Inexact Policy Mirror Descent Methods

For DMDP problems with large state and action spaces, computing the exact policy gradients or Q -functions are very costly and infeasible in practice. In this section, we consider the following inexact PMD method where ̂ Q s ( π ( k ) ) is an inexact evaluation of Q s ( π ( k ) ). We first study the convergence properties of (58) under the following assumption on the evaluation error.

<!-- formula-not-decoded -->

Assumption 1 The inexact Q -function evaluations Q ( π ( k ) ) satisfy

The following result is the counterpart of Lemma 7 for the inexact PMD method.

<!-- formula-not-decoded -->

Lemma 11 Consider the inexact PMD method (58) with π (0) ∈ rint Π and suppose that Assumption 1 holds. Then we have for all k ≥ 0 , and for any ρ ∈ ∆( S ) ,

<!-- formula-not-decoded -->

Proof The proof of (60) follows the same arguments as in Lemma 7. However, due to the inexact Q -function evaluations, the objectives V ρ ( π ( k ) ) are no longer monotone decreasing. We use the performance difference lemma to deduct:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that the first term on the right-hand side is non-positive due to (60). For the second term, we use H¨ older's inequality to obtain, for all s ∈ S , where the second inequality is due to ∥ ∥ π ( k +1) s -π ( k ) s ∥ ∥ 1 ≤ ∥ ∥ π ( k +1) s ∥ ∥ 1 + ∥ ∥ π ( k ) s ∥ ∥ 1 ≤ 2, and the last inequality is due to Assumption 1. Combining (62) with the previous inequality yields (61).

We will need the following simple fact, whose proof is straightforward and thus omitted.

Lemma 12 Suppose 0 &lt; α &lt; 1 , b &gt; 0 , and a nonnegative sequence { a k } satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then for all k ≥ 0 ,

The following theorem characterizes the convergence of the inexact PMD method under Assumption 1. We keep using the notations D /star k and ϑ ρ defined in (43) and (51), respectively.

Theorem 13 Consider the inexact PMD method (58) with π (0) ∈ rint Π and suppose that Assumption 1 holds. If the step sizes satisfy η 0 ≥ 1 -γ γ D /star 0 and η k +1 ≥ η k /γ , then we have for all k ≥ 0 ,

<!-- formula-not-decoded -->

Proof Applying Lemma 6 to the update in (58) and following the same arguments in the proof of Theorem 8, we arrive at the following counterpart of (44):

<!-- formula-not-decoded -->

For the first expectation in (64), we follow the proof of Proposition 9 to obtain

<!-- formula-not-decoded -->

where the last inequality is due to the performance difference lemma and (62). For the second expectation in (64), we again use the performance difference lemma and H¨ older's inequality to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting the last two bounds into (64) and dividing both sides by 1 -γ , we get

where δ k := V ρ ( π ( k +1) ) -V /star ρ . Since δ k +1 -δ k -2 τ 1 -γ ≤ 0 (Lemma 11) and ϑ k +1 ≤ ϑ ρ , the above inequality still holds with ϑ k +1 replaced by ϑ ρ , which leads to

<!-- formula-not-decoded -->

Dividing both sides by ϑ ρ and rearranging terms, we get

<!-- formula-not-decoded -->

If the step sizes satisfy η k +1 ( ϑ ρ -1) ≥ η k ϑ ρ , which is implied by η k +1 ≥ η k /γ , then

<!-- formula-not-decoded -->

where we also used 1 + 1 /ϑ ρ &lt; 2 because ϑ ρ &gt; 1. Next we invoke Lemma 12 with

<!-- formula-not-decoded -->

which leads to

<!-- formula-not-decoded -->

Finally applying (55) and η 0 ≥ 1 -γ γ D /star 0

gives the desired result (63).

As a result of Theorem 13, we have the following asymptotic error bound:

<!-- formula-not-decoded -->

which agrees with that of conservative policy iteration (CPI) of Kakade and Langford (2002, Theorem 6.2). It is also similar to the asymptotic error bound of many approximate dynamical programming algorithms (e.g., Bertsekas, 2012), with the additional factor of distribution mismatch coefficient.

## 5.1 Sample Complexity under a Generative Model

One way to ensure Assumption 1 hold with high probability is through multiple independent simulations (rollouts) of the MDP under a fixed policy. In this section, we analyze the sample complexity of this approach.

Suppose that for a given policy π ( k ) and any state-action pair ( s, a ) ∈ S × A , we can generate a set of M k independent, truncated trajectories of horizon H , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 14 Consider the Q -estimator given in (65) . For any δ ∈ (0 , 1) , if M k satisfies

The following lemma gives a high-probability bound on the error ‖ ̂ Q ( π ( k ) ) -Q ( π ( k ) ) ‖ ∞ .

<!-- formula-not-decoded -->

then we have with probability at least 1 -δ ,

Proof We first define the expectation of the Q -estimator in (65):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which holds for any i = 1 , . . . , M k . Recall the definition of Q s,a in (15). Since R ( s, a ) ≥ 0, we always have Q s,a ( π ( k ) ) -Q s,a ( π ( k ) ) ≥ 0. On the other hand, which holds for all ( s, a ) ∈ S × A . Therefore,

Next that we can decompose the estimation error into two parts:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last term is bounded by (67), so we need to bound ∥ ∥ ̂ Q ( π ( k ) ) -Q ( π ( k ) ) ∥ ∥ ∞ . To this end, we notice that the random variables ̂ Q ( i ) s,a ( π ( k ) ) are bounded in the interval [0 , 1 / (1 -γ )]. Therefore Hoeffding's inequality (Hoeffding, 1963) implies that for any σ k &gt; 0,

Applying the union bound across all ( s, a ) ∈ S × A , we obtain

<!-- formula-not-decoded -->

Therefore, for any δ ∈ (0 , 1), if we choose M k large enough, i.e., then ∥ ∥ ̂ Q ( π ( k ) ) -Q ( π ( k ) ) ∥ ∥ ∞ &lt; σ k with probability at least 1 -δ . Combining with (67) and (68), we conclude that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally setting σ k = γ H / (1 -γ ) gives the desired result.

The next theorem characterizes the sample complexity of the inexact PMD method with the simple Q -estimator.

Theorem 15 Consider using the Q -estimator (65) in the inexact PMD method (58) , with the step sizes satisfying η 0 ≥ 1 -γ γ D /star 0 and η k +1 ≥ 1 γ η k for all k ≥ 0 . For any δ ∈ (0 , 1) and integers H &gt; 0 and K &gt; 0 , suppose the batch sizes M k satisfy

<!-- formula-not-decoded -->

Then we have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

In addition, for any /epsilon1 &gt; 0 , we have V ρ ( π ( K ) ) -V /star ρ ≤ /epsilon1 with probability at least 1 -δ if

<!-- formula-not-decoded -->

The corresponding sample complexity of state-action pairs is

<!-- formula-not-decoded -->

Proof Suppose the total number of iterations is K . In order to have (66) hold for all k = 0 , 1 , . . . , K -1, we need to apply the union bound across all K iterations, which imposes an additional factor K on the right-hand side of (69). Consequently, we can extend Lemma 14 to ensure that the event where the notation ˜ O ( · ) hides poly-logarithmic factors of 1 / (1 -γ ) , 1 //epsilon1 and |S||A| /δ .

<!-- formula-not-decoded -->

occurs with probability at least 1 -δ provided that (71) holds. Then (72) follows directly from Theorem 13 with τ = 2 γ H / (1 -γ ).

In order to have V ρ ( π ( k ) ) -V /star ρ ≤ /epsilon1 within K iterations, it suffices to have each of the two terms on the right-hand side of (72) less than /epsilon1/ 2, i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which translate into the conditions on K and H in (73). Correspondingly, the batch sizes need to satisfy M k ≥ M where

The total number of state-action samples can be estimated as

<!-- formula-not-decoded -->

Finally, plugging in the definition ϑ ρ = 1 1 -γ ∥ ∥ ∥ d ρ ( π /star ) ρ ∥ ∥ ∥ ∞ gives the estimate in (74).

<!-- formula-not-decoded -->

The sample complexity obtained in Theorems 15 has O ( /epsilon1 -2 ) dependence on /epsilon1 . This is better than that of O ( /epsilon1 -4 ) obtained by Shani et al. (2020) and Agarwal et al. (2021) and O ( /epsilon1 -3 ) by Liu et al. (2020) for policy gradient type of methods (without regularization). Cen et al. (2020) remarked that O ( /epsilon1 -2 ) sample complexity can be obtained with entropy regularization. Their approach leads to a result without the factor of the distribution mismatch coefficient, but with the same 1 / (1 -γ ) 8 factor. Lazaric et al. (2016) derived an O ( /epsilon1 -2 ) sample complexity for a variant of the policy iteration method, with a factor of at least 1 / (1 -γ ) 7 . Lan (2021) studies sample complexity in expectation instead of with high probability and obtains similar results with weaker dependence on 1 / (1 -γ ). Yuan et al. (2021) characterize the sample complexity of vanilla policy gradient method (such as REINFORCE (Williams, 1992)) under a variety of different assumptions on the parametrized value function.

Much progresses have been made for understanding the sample complexity of DMDP in the tabular setting. Azar et al. (2013) established a lower bound of ˜ Ω ( |S||A| (1 -γ ) 3 /epsilon1 2 ) for DMDP under a generative model , which allows drawing random state-transitions repeatedly under any policy. The simple Q-estimator we use in this section fits this sample oracle model, but the dependence of our results on 1 / (1 -γ ) is much worse than the lower bound. On the other hand, this lower bound has been matched or nearly matched by several recent work based on variance-reduced Value Iteration (Sidford et al., 2018) and Q -learning (Wainwright, 2019). There are interesting work to be done for improving the sample complexity of stochastic policy gradient methods.

## 6. Conclusion and Discussion

We developed a general theory of weak gradient-mapping dominance and used it to obtain an improved sublinear convergence rate of the projected policy gradient methods. By exploiting additional structure of discounted Markov decision problem (DMDP), we show that with a simple, non-adaptive rule of geometrically increasing the step sizes, policy mirror descent methods enjoy linear convergence without relying on entropy or other strongly convex regularizations. In fact, the convergence rates obtained with strongly convex regularizations (Cen et al., 2020; Lan, 2021; Zhan et al., 2021) are no better than γ k regardless of the regularization strength.

Our results on policy mirror descent methods show that dynamic preconditioning using discounted state-visitation distributions is critical for obtaining fast convergence rates that are (almost) independent of problem dimensions. The adopted local Bregman divergence, being KL-divergence or squared Euclidean distance, does not make much difference. Indeed, when the step sizes grow to infinity, preconditioned policy mirror descent methods derived with different Bregman divergences all reduce to the classical Policy Iteration algorithm. Essentially, such methods with finite step sizes can be viewed as inexact Policy Iteration methods, much like many approximate dynamic programming algorithms.

The major limitation of this work is our restriction to direct policy parametrization. (We note that the NPG method with tabular softmax parametrization has an equivalent mirror-descent form expressed in the policy space, therefore is included in our study.) A natural extension is to consider general policy parametrizations of the form π ( θ ) where the dimension of θ is much smaller than |S||A| . There are two ways to proceed. The first approach is to simply treat it as a nonlinear optimization problem of minimizing the composite objective J ρ ( θ ) = V ρ ( π ( θ )). This approach may lose some important structure of DMDP. In particular, the parametrized objective function J ρ ( θ ) may no longer be quasiconvex or quasi-concave. As a result, it will be hard to establish convergence to global optimum and we may have to rely on standard theory of smooth nonconvex optimization, which imposes bounded step sizes and leads to relatively slow convergence rates.

The second approach is to follow the framework of compatible function approximation (Sutton et al., 2000; Kakade, 2001), which is extensively developed by Agarwal et al. (2021). This approach facilitates the extension of our results on inexact policy mirror descent to general policy parametrization. In particular, our results in Section 5 show that geometrically increasing step sizes do not cause instability even if the Q -functions are evaluated inaccurately. In fact, inexact policy mirror descent methods converge linearly up to an asymptotic error floor, which immediately leads to an O ( /epsilon1 -2 ) sample complexity as we have shown. It is of great interest to reduce the dependence of sample complexity on 1 / (1 -γ ) and the distribution mismatch coefficient.

## Acknowledgments

The author is grateful to Lihong Li and Simon S. Du for helpful discussions and feedback. Parts of the results in this paper were obtained by the author while preparing for a tutorial jointly with Lihong Li at the SIAM Conference on Optimization held in July 2021.

The author is indebted to Marek Petrik and Julien Grand-Clement, who found a mistake in a previous version of this paper stating that the weighted value function is quasi-convex and quasi-concave. They gave a simple counter-example and pointed out the mistake in the proof. Indeed, it is neither quasi-convex nor quasi-concave. Fortunately this mistake does not affect the rest of the results that are contained in this version.

## A. Appendix

## A.1 Derivation of Policy Gradient using Matrix Calculus

We derive the policy gradient formula (18) using simple matrix calculus. Let e s ∈ R |S| be a vector with components e s,s ′ = 1 if s = s ′ and 0 otherwise. From the expression of V ( π ) in (9), we can write its components as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the matrix calculus formula ∂X -1 ∂π = -X -1 ∂X ∂π X -1 with X = ( I -γP ( π )), we have where in the last equality we used ∂r ( π ) /∂π s ′ ,a ′ = R s ′ ,a ′ e s ′ and the definition of V ( π ). From the definition of P ( π ), we have ∂P ( π ) /∂π s ′ ,a ′ = e s ′ P ( ·| s ′ , a ′ ), which is a rank-one matrix with P ( ·| s ′ , a ′ ) acting as a row vector. Therefore,

where we used the expression of d s,s ′ ( π ) in (12) and the definition of Q s ′ ,a ′ ( π ). This gives the component-wise expression for policy gradient, which leads to the aggregated form (18).

<!-- formula-not-decoded -->

## A.2 Strong Gradient-Mapping Domination

Following the setting in Section 3.1, we define a stronger notion of gradient-mapping domination and show that it leads to geometric convergence to a global optimum.

Definition 16 (strong gradient-mapping domination) Suppose F := f +Ψ where f is L -smooth and Ψ is proper, convex and closed. We say that F satisfies a strong gradientmapping dominance condition if there exists µ &gt; 0 such that

<!-- formula-not-decoded -->

where F /star = min x F ( x ) and T L and G L are defined in (30) and (31) respectively.

Consider the composite optimization problem of minimizing F := f + Ψ where f is L -smooth and Ψ is proper, convex and closed. If F satisfies the strong gradient-mapping domination condition, then the proximal gradient method (29) converges geometrically to a global minimum. To see this, we simply combine the descent property (32) with strong gradient-mapping dominance condition (75) to obtain

<!-- formula-not-decoded -->

Rearranging terms, we obtain

<!-- formula-not-decoded -->

This leads to a geometric recursion and we have

<!-- formula-not-decoded -->

Connections with other notions of gradient dominance. The classical Kurdyka/suppress Lojasiewicz (K/suppress L) condition with exponent 1 / 2 (Kurdyka, 1998) can be expressed as

<!-- formula-not-decoded -->

where ∂F ( x ) denotes the set of subgradients (subdifferential) of F at x . Karimi et al. (2016) derived a proximal Polyak-/suppress Lojasiewicz (P/suppress L) condition where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and showed that it is equivalent to the K/suppress L condition (76) in the sense that they imply each other albeit with different constants ˜ µ and µ . Interestingly, it can be shown that there is an interlacing relationship between our definition of gradient-mapping domination and the proximal P/suppress L condition:

<!-- formula-not-decoded -->

The proximal P/suppress L condition (77) takes the first and the third terms in the above inequality chain, while our gradient-mapping dominance condition (75) takes the second and the fourth. We conjecture that these two conditions also imply each other.

## References

- Alekh Agarwal, Sham M. Kakade, Jason D. Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research , 22(98):1-76, 2021.
- Mohammad Gheshlaghi Azar, R´ emi Munos, and Hilbert J. Kappen. Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine Learning , 91(3):325-349, 2013.
- Amir Beck. First-Order Methods in Optimization . MOS-SIAM Series on Optimization. SIAM, 2017.
- Amir Beck and Marc Teboulle. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters , 31(3):167-175, 2003.
- Richard Bellman. Dynamic Programming . Princeton University Press, Princeton, NJ, USA, 1957.
- Dimitri P. Bertsekas. Dynamic Programming and Optimal Control , volume 2: Approximate Dynamic Programming . Athena Scientific, 4th edition, 2012.
- Dimitri P. Bertsekas and John N. Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1996.
- Jalaj Bhandari and Daniel Russo. Global optimality guarantees for policy gradient methods. arXiv preprint, arXiv:1906.01786, 2019.
- Jalaj Bhandari and Daniel Russo. On the linear convergence of policy gradient methods for finite mdps. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics , volume 130 of Proceedings of Machine Learning Research , pages 2386-2394. PMLR, 13-15 Apr 2021.
- Shicong Cen, Chen Cheng, Yuxin Chen, Yuting Wei, and Yuejie Chi. Fast global convergence of natural policy gradient methods with entropy regularization. arXiv preprint, arXiv:2007.06558, 2020.
- Gong Chen and Marc Teboulle. Convergence analysis of a proximal-like minimization algorithm using Bregman functions. SIAM Journal on Optimization , 3(3):538-543, 1993.
- Maryam Fazel, Rong Ge, Sham Kakade, and Mehran Mesbahi. Global convergence of policy gradient methods for the linear quadratic regulator. In Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pages 1467-1476. PMLR, 10-15 Jul 2018.
- Saeed Ghadimi and Guanghui Lan. Accelerated gradient methods for nonconvex nonlinear and stochastic programming. Mathematical Programming , 156(1-2):59-99, 2016.
- Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association , 58:13-30, 1963.

- Sham Kakade. A natural policy gradient. In Proceedings of the 14th International Conference on Neural Information Processing Systems (NIPS'01) , pages 1531-1538, 2001.
- Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the 19th International Conference on Machine Learning (ICML) , volume 2, pages 267-274, 2002.
- Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximal-gradient methods under the Polyak-/suppress lojasiewicz condition. In Paolo Frasconi, Niels Landwehr, Giuseppe Manco, and Jilles Vreeken, editors, Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2016) , volume 9851 of Lectur Notes in Computer Sciencce . Springer, 2016.
- Sajad Khodadadian, Prakirt Raj Jhunjhunwala, Sushil Mahavir Varma, and Siva Theja Maguluri. On the linear convergence of natural policy gradient algorithm. arXiv preprint, arXiv:2105.01424, 2021.
- Vijay Konda and John Tsitsiklis. Actor-critic algorithms. In Advances in Neural Information Processing Systems , volume 12, pages 1008-1014. MIT Press, 2000.
- Krzysztof Kurdyka. On gradients of functions definable in o-minimal structures. Annales de l'institut Fourier , 48(3):769-783, 1998.
- Guanghui Lan. Policy mirror descent for reinforcement learning: Linear convergence, new sampling complexity, and generalized problem classes. Preprint, arXiv:2102.00135, 2021.
- Alessandro Lazaric, Mohammad Ghavamzadeh, and R´ emi Munos. Analysis of classificationbased policy iteration algorithms. Journal of Machine Learning Research , 17:1-30, 2016.
- Gen Li, Yuting Wei, Yuejie Chi, Yuantao Gu, and Yuxin Chen. Softmax policy gradient methods can take exponential time to converge. In Proceedings of Thirty Fourth Conference on Learning Theory , volume 134 of Proceedings of Machine Learning Research , pages 3107-3110. PMLR, 15-19 Aug 2021.
- Guoyin Li and Ting Kei Pong. Calculus of the exponent of Kurdyka-/suppress Ljasiewicz inequality and its applications to linear convergence of first-order methods. Foundations of Computational Mathematics , 18:1199-1232, 2018.
- Boyi Liu, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural trust region/proximal policy optimization attains globally optimal policy. In Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019.
- Yanli Liu, Kaiqing Zhang, Tamer Basar, and Wotao Yin. An improved analysis of (variancereduced) policy gradient and natural policy gradient methods. In Advances in Neural Information Processing Systems , volume 33, pages 7624-7636. Curran Associates, Inc., 2020.
- Jincheng Mei, Chenjun Xiao, Csaba Szepesv´ ari, and Dale Schuurmans. On the global convergence rates of softmax policy gradient methods. In Proceedings of the 37 th International Conference on Machine Learning (ICML) , 2020.

- Jincheng Mei, Yue Gao, Bo Dai, Csaba Szepesv´ ari, and Dale Schuurmans. Leveraging non-uniformity in first-order non-convex optimization. In Proceedings of the 38 th International Conference on Machine Learning (ICML) , 2021.
- R´ emi Munos. Error bounds for approximate policy iteration. In Proceedings of the 20th International Conference on Machine Learning (ICML'03) , pages 560-567, 2003.
- R´ emi Munos. Error bounds for approximate value iteration. In Proceedings of the 20th National Conference on Artificial Intelligence (AAAI'05) , pages 1006-1011, 2005.
- A. Nemirovski and D. Yudin. Problem Complexity and Method Efficiency in Optimization . Wiley Interscience, 1983.
- Yurii Nesterov. Gradient methods for minimizing composite functions. Mathematical Programming , 140:125-161, 2013.
- Boris T. Polyak. Gradient methods for minimizing functionals. USSR Computational Mathematics and Mathematical Physics , 3(4):864-878, 1963.
- Martin L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . Wiley Series in Probability and Statistics. John Wiley and Sons, Inc., 1994.
- R. Tyrrell Rockafellar. Convex Analysis . Princeton University Press, 1970.
- Bruno Scherrer. Approximate policy iteration schemes: A comparison. In Proceedings of the 31st International Conference on Machine Learning , volume 32 of Proceedings of Machine Learning Research , pages 1314-1322, Bejing, China, 22-24 Jun 2014.
- John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning , volume 37 of Proceedings of Machine Learning Research , pages 1889-1897, Lille, France, 07-09 Jul 2015.
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint, arXiv:1707.06347, 2017.
- Lior Shani, Yonathan Efroni, and Shie Mannor. Adaptive trust region policy optimization: Global convergence and faster rates for regularized mdps. In The Thirty-Fourth AAAI Conference on Artificial Intelligence , pages 5668-5675. AAAI Press, 2020.
- Aaron Sidford, Mengdi Wang, Xian Wu, Lin Yang, and Yinyu Ye. Near-optimal time and sample complexities for solving markov decision processes with a generative model. In Advances in Neural Information Processing Systems , volume 31. Curran Associates, Inc., 2018.
- Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction . Adaptive Computation and Machine Learning. The MIT Press, 2nd edition, 2018.
- Richard S. Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems , volume 12, pages 1057-1063. MIT Press, 2000.

- Csaba Szepesv´ ari and R´ emi Munos. Finite time bounds for fitted value iteration. Journal of Machine Learning Research , pages 815-857, 2008.
- Martin J. Wainwright. Variance-reduced Q-learning is minimax optimal. arXiv e-Preprint, arXiv:1906.04697, 2019.
- Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning , 8:229-256, 1992.
- Yinyu Ye. The simplex and policy-iteration methods are strongly polynomial for the markov decision problem with a fixed discount rate. Mathematics of Operations Research , 36(4): 593-603, 2011.
- Rui Yuan, Robert M. Gower, and Alessandro Lazaric. A general sample complexity analysis of vanilla policy gradient. arXiv e-Preprint, arXiv:2107.11433, 2021.
- Wenhao Zhan, Shicong Cen, Baihe Huang, Yuxin Chen, Jason D. Lee, and Yuejie Chi. Policy mirror descent for regularized reinforcement learning: A generalized framework with linear convergence. Preprint, arXiv:2105.11066, 2021.
- Junyu Zhang, Alec Koppel, Amrit Singh Bedi, Csaba Szepesvari, and Mengdi Wang. Variational policy gradient method for reinforcement learning with general utilities. In Advances in Neural Information Processing Systems , volume 33, pages 4572-4583. Curran Associates, Inc., 2020.