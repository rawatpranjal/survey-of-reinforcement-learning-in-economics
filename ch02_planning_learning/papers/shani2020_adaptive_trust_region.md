## Adaptive Trust Region Policy Optimization: Global Convergence and Faster Rates for Regularized MDPs

Lior Shani † , Yonathan Efroni † , Shie Mannor

† equal contribution Technion - Israel Institute of Technology Haifa, Israel

## Abstract

Trust region policy optimization (TRPO) is a popular and empirically successful policy search algorithm in Reinforcement Learning (RL) in which a surrogate problem, that restricts consecutive policies to be 'close' to one another, is iteratively solved. Nevertheless, TRPO has been considered a heuristic algorithm inspired by Conservative Policy Iteration (CPI). We show that the adaptive scaling mechanism used in TRPO is in fact the natural 'RL version' of traditional trust-region methods from convex analysis. We first analyze TRPO in the planning setting, in which we have access to the model and the entire state space. Then, we consider sample-based TRPO and establish ˜ O (1 / √ N ) convergence rate to the global optimum. Importantly, the adaptive scaling mechanism allows us to analyze TRPO in regularized MDPs for which we prove fast rates of ˜ O (1 /N ) , much like results in convex optimization. This is the first result in RL of better rates when regularizing the instantaneous cost or reward.

## 1 Introduction

The field of Reinforcement learning (RL) (Sutton and Barto 2018) tackles the problem of learning how to act optimally in an unknown dynamic environment. The agent is allowed to apply actions on the environment, and by doing so, to manipulate its state. Then, based on the rewards or costs it accumulates, the agent learns how to act optimally. The foundations of RL lie in the theory of Markov Decision Processes (MDPs), where an agent has an access to the model of the environment and can plan to act optimally.

Trust Region Policy Optimization (TRPO): Trust region methods are a popular class of techniques to solve an RL problem and span a wide variety of algorithms including Non-Euclidean TRPO (NETRPO) (Schulman et al. 2015) and Proximal Policy Optimization (Schulman et al. 2017). In these methods a sum of two terms is iteratively being minimized: a linearization of the objective function and a proximity term which restricts two consecutive updates to be 'close' to each other, as in Mirror Descent (MD) (Beck and Teboulle 2003).

Copyright c © 2020, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

In spite of their popularity, much less is understood in terms of their convergence guarantees and they are considered heuristics (Schulman et al. 2015; Papini, Pirotta, and Restelli 2019) (see Figure 1).

TRPO and Regularized MDPs: Trust region methods are often used in conjunction with regularization. This is commonly done by adding the negative entropy to the instantaneous cost (Nachum et al. 2017; Schulman et al. 2017). The intuitive justification for using entropy regularization is that it induces inherent exploration (Fox, Pakman, and Tishby 2016), and the advantage of 'softening' the Bellman equation (Chow, Nachum, and Ghavamzadeh 2018; Dai et al. 2018). Recently, Ahmed et al. (2019) empirically observed that adding entropy regularization results in a smoother objective which in turn leads to faster convergence when the learning rate is chosen more aggressively. Yet, to the best of our knowledge, there is no finite-sample analysis that demonstrates faster convergence rates for regularization in MDPs. This comes in stark contrast to well established faster rates for strongly convex objectives w.r.t. convex ones (Nesterov 1998). In this work we refer to regularized MDPs as describing a more general case in which a strongly convex function is added to the immediate cost.

The goal of this work is to bridge the gap between the practicality of trust region methods in RL and the scarce theoretical guarantees for standard (unregularized) and regularized MDPs. To this end, we revise a fundamental question in this context:

## What is the proper form of the proximity term in trust region methods for RL?

In Schulman et al. (2015), two proximity terms are suggested which result in two possible versions of trust region methods for RL. The first (Schulman et al. 2015, Algorithm 1) is motivated by Conservative Policy Iteration (CPI) (Kakade and others 2003) and results in an improving and thus converging algorithm in its exact error-free version. Yet, it seems computationally infeasible to produce a sample-based version of this algorithm. The second algorithm, with an adaptive proximity term which depends on the current policy (Schulman et al. 2015, Equation 12), is described as a heuristic approximation of the first, with no

Figure 1: The adaptive TRPO: a solid line implies a formal relation; a dashed line implies a heuristic relation.

<!-- image -->

convergence guarantees, but leads to NE-TRPO, currently among the most popular algorithms in RL (see Figure 1).

In this work, we focus on tabular discounted MDPs and study a general TRPO method which uses the latter adaptive proximity term. Unlike the common belief, we show this adaptive scaling mechanism is 'natural' and imposes the structure of RL onto traditional trust region methods from convex analysis. We refer to this method as adaptive TRPO, and analyze two of its instances: NE-TRPO (Schulman et al. 2015, Equation 12) and Projected Policy Gradient (PPG), as illustrated in Figure 1. In Section 2, we review results from convex analysis that will be used in our analysis. Then, we start by deriving in Section 4 a closed form solution of the linearized objective functions for RL. In Section 5, using the closed form of the linearized objective, we formulate and analyze Uniform TRPO. This method assumes simultaneous access to the state space and that a model is given. In Section 6, we relax these assumptions and study SampleBased TRPO, a sample-based version of Uniform TRPO, while building on the analysis of Section 5. The main contributions of this paper are:

- We establish ˜ O (1 / √ N ) convergence rate to the global optimum for both Uniform and Sample-Based TRPO.
- We prove a faster rate of ˜ O (1 /N ) for regularized MDPs. To the best of our knowledge, it is the first evidence for faster convergence rates using regularization in RL.
- The analysis of Sample-Based TRPO, unlike CPI, does not rely on improvement arguments. This allows us to choose a more aggressive learning rate relatively to CPI which leads to an improved sample complexity even for the unregularized case.

## 2 Mirror Descent in Convex Optimization

Mirror descent (MD) (Beck and Teboulle 2003) is a well known first-order trust region optimization method for solving constrained convex problems, i.e, for finding

<!-- formula-not-decoded -->

where f is a convex function and C is a convex compact set. In each iteration, MD minimizes a linear approximation of the objective function, using the gradient ∇ f ( x k ) , together with a proximity term by which the updated x k +1 is 'close'

to x k . Thus, it is considered a trust region method, as the iterates are 'close' to one another. The iterates of MD are

<!-- formula-not-decoded -->

where B ω ( x, x k ) := ω ( x ) -ω ( x k ) - 〈∇ ω ( x k ) , x -x k 〉 is the Bregman distance associated with a strongly convex ω and t k is a stepsize (see Appendix A). In the general convex case, MD converges to the optimal solution of (1) with a rate of ˜ O (1 / √ N ) , where N is the number of MD iterations (Beck and Teboulle 2003; Juditsky, Nemirovski, and others 2011), i.e., f ( x k ) -f ∗ ≤ ˜ O (1 / √ k ) , where f ∗ = f ( x ∗ ) .

The convergence rate can be further improved when f is a part of special classes of functions. One such class is the set of λ -strongly convex functions w.r.t. the Bregman distance. We say that f is λ -strongly convex w.r.t. the Bregman distance if f ( y ) ≥ f ( x ) + 〈∇ f ( x ) , y -x 〉 + λB ω ( y, x ) . For such f , improved convergence rate of ˜ O (1 /N ) can be obtained (Juditsky, Nemirovski, and others 2011; Nedic and Lee 2014). Thus, instead of using MD to optimize a convex f , one can consider the following regularized problem,

<!-- formula-not-decoded -->

where g is a strongly convex regularizer with coefficient λ &gt; 0 . Define F λ ( x ) := f ( x ) + λg ( x ) , then, each iteration of MD becomes,

<!-- formula-not-decoded -->

Solving (4) allows faster convergence, at the expense of adding a bias to the solution of (1). Trivially, by setting λ = 0 , we go back to the unregularized convex case.

In the following, we consider two common choices of ω which induce a proper Bregman distance: (a) The euclidean case , with ω ( · ) = 1 2 ‖·‖ 2 2 and the resulting Bregman distance is the squared euclidean norm B ω ( x, y ) = 1 2 ‖ x -y ‖ 2 2 . In this case, (2) becomes the Projected Gradient Descent algorithm (Beck 2017, Section 9.1), where in each iteration, the update step goes along the direction of the gradient at x k , ∇ f ( x k ) , and then projected back to the convex set C , x k +1 = P c ( x k -t k ∇ f ( x k )) , where P c ( x ) = min y ∈ C 1 2 ‖ x -y ‖ 2 2 is the orthogonal projection operator w.r.t. the euclidean norm.

- (b) The non-euclidean case , where ω ( · ) = H ( · ) is the negative entropy, and the Bregman distance then becomes the Kullback-Leibler divergence, B ω ( x, y ) = d KL ( x || y ) . In this case, MD becomes the Exponentiated Gradient Descent algorithm. Unlike the euclidean case, where we need to project back into the set, when choosing ω as the negative entropy, (2) has a closed form solution (Beck 2017, Example 3.71), x i k +1 = x i k e -t k ∇ i f ( x k ) ∑ j x j k e -t k ∇ j f ( x k ) , where x i k and ∇ i f are the i -th coordinates of x k and ∇ f .

## 3 Preliminaries and Notations

We consider the infinite-horizon discounted MDP which is defined as the 5-tuple ( S , A , P, C, γ ) (Sutton and Barto 2018), where S and A are finite state and action sets with cardinality of S = |S| and A = |A| , respectively. The transition kernel is P ≡ P ( s ′ | s, a ) , C ≡ c ( s, a ) is a cost function bounded in [0 , C max ] ∗ , and γ ∈ (0 , 1) is a discount factor. Let π : S → ∆ A be a stationary policy, where ∆ A is the set probability distributions on A . Let v π ∈ R S be the value of a policy π, with its s ∈ S entry given by v π ( s ) := E π [ ∑ ∞ t =0 γ t r ( s t , π ( s t )) | s 0 = s ] , and E π [ · | s 0 = s ] denotes expectation w.r.t. the distribution induced by π and conditioned on the event { s 0 = s } . It is known that v π = ∑ ∞ t =0 γ t ( P π ) t c π = ( I -γP π ) -1 c π , with the component-wise values [ P π ] s,s ′ := P ( s ′ | s, π ( s )) and [ c π ] s := c ( s, π ( s )) . Our goal is to find a policy π ∗ yielding the optimal value v ∗ such that

<!-- formula-not-decoded -->

This goal can be achieved using the classical operators:

<!-- formula-not-decoded -->

where T π is a linear operator, T is the optimal Bellman operator and both T π and T are γ -contraction mappings w.r.t. the max-norm. The fixed points of T π and T are v π and v ∗ .

A large portion of this paper is devoted to analysis of regularized MDPs: A regularized MDP is an MDP with a shaped cost denoted by c π λ for λ ≥ 0 . Specifically, the cost of a policy π on a regularized MDP translates to c π λ ( s ) := c π ( s ) + λω ( s ; π ) where ω ( s ; π ) := ω ( π ( · | s )) and ω : ∆ A → R is a 1 -strongly convex function. We denote ω ( π ) ∈ R S as the corresponding state-wise vector. See that for λ = 0 , the cost c π is recovered. In this work we consider two choices of ω : the euclidean case ω ( s ; π ) = 1 2 ‖ π ( · | s ) ‖ 2 2 and non-euclidean case ω ( s ; π ) = H ( π ( · | s ))+log A . By this choice we have that 0 ≤ c π λ ( s ) ≤ C max ,λ where C max ,λ = C max + λ and C max ,λ = C max + λ log A , for the euclidean and non-euclidean cases, respectively. With some abuse of notation we omit ω from C max ,λ .

The value of a stationary policy π on the regularized MDP is v π λ = ( I -γP π ) -1 c π λ . Furthermore, the optimal value v ∗ λ , optimal policy π ∗ λ and Bellman operators of the regularized MDP are generalized as follows,

<!-- formula-not-decoded -->

As Bellman operators for MDPs, both T π λ , T are γ -contractions with fixed points v π λ , v ∗ λ (Geist, Scherrer, and Pietquin 2019). Denoting c π λ ( s, a ) = c ( s, a ) + λω ( s ; π ) , the q -function of a policy π for a regularized MDP is defined as q π λ ( s, a ) = c π λ ( s, a ) + γ ∑ s ′ p π ( s ′ | s ) v π λ ( s ′ ) .

∗ We work with costs instead of rewards to comply with convex analysis. All results are valid to the case where a reward is used.

When the state space is small and the dynamics of environment is known (5), (7) can be solved using DP approaches. However, in case of a large state space it is expected to be computationally infeasible to apply such algorithms as they require access to the entire state space. In this work, we construct a sample-based algorithm which minimizes the following scalar objective instead of (5), (7),

<!-- formula-not-decoded -->

where µ ( · ) is a probability measure over the state space. Using this objective, one wishes to find a policy π which minimizes the expectation of v π λ ( s ) under a measure µ . This objective is widely used in the RL literature (Sutton et al. 2000; Kakade and Langford 2002; Schulman et al. 2015).

Here, we always choose the regularization function ω to be associated with the Bregman distance used, B ω . This simplifies the analysis as c π λ is λ -strongly convex w.r.t. B ω by definition. Given two policies π 1 , π 2 , we denote their Bregman distance as B ω ( s ; π 1 , π 2 ) := B ω ( π 1 ( · | s ) , π 2 ( · | s )) and B ω ( π 1 , π 2 ) ∈ R S is the corresponding state-wise vector. The euclidean choice for ω leads to B ω ( s ; π 1 , π 2 ) = 1 2 ‖ π 1 ( · | s ) -π 2 ( · | s ) ‖ 2 2 , and the non-euclidean choice to B ω ( s ; π 1 , π 2 ) = d KL ( π 1 ( · | s ) || π 2 ( · | s )) . In the results we use the following ω -dependent constant, C ω, 1 = √ A in the euclidean case, and C ω, 1 = 1 in the non-euclidean case.

For brevity, we omit constant and logarithmic factors when using O ( · ) , and omit any factors other than non-logarithmic factors in N , when using ˜ O ( · ) . For x, y ∈ R S × A , the state-action inner product is 〈 x, y 〉 = ∑ s,a x ( s, a ) y ( s, a ) , and the fixed-state inner product is 〈 x ( s, · ) , y ( s, · ) 〉 = ∑ a x ( s, a ) y ( s, a ) . Lastly, when x ∈ R S × S × A (e.g., first claim of Proposition 1) the inner product 〈 x, y 〉 is a vector in R S where 〈 x, y 〉 ( s ) := 〈 x ( s, · , · ) , y 〉 , with some abuse of notation.

## 4 Linear Approximation of a Policy's Value

As evident from the updating rule of MD (2), a crucial step in adapting MD to solve MDPs is studying the linear approximation of the objective, 〈∇ f ( x ) , x ′ -x 〉 , i.e., the directional derivative in the direction of an element from the convex set. The objectives considered in this work are (7), (8), and the optimization set is the convex set of policies ∆ S A . Thus, we study 〈∇ v π λ , π ′ -π 〉 and 〈∇ µv π λ , π ′ -π 〉 , for which the following proposition gives a closed form:

Proposition 1 (Linear Approximation of a Policy's Value) . Let π, π ′ ∈ ∆ S A , and d µ,π = (1 -γ ) µ ( I -γP π ) -1 . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof, supplied in Appendix B, is a direct application of a Policy Gradient Theorem (Sutton et al. 2000) derived for regularized MDPs. Importantly, the linear approximation is scaled by ( I -γP π ) -1 or 1 1 -γ d µ,π , the discounted visitation frequency induced by the current policy. In what follows, we use this understanding to properly choose an adaptive scaling for the proximity term of TRPO, which allows us to use methods from convex optimization.

## 5 Uniform Trust Region Policy Optimization

In this section we formulate Uniform TRPO , a trust region planning algorithm with an adaptive proximity term by which (7) can be solved, i.e., an optimal policy which jointly minimizes the vector v π λ is acquired. We show that the presence of the adaptive term simplifies the update rule of Uniform TRPO and then analyze its performance for both the regularized ( λ &gt; 0 ) and unregularized ( λ = 0 ) cases. Despite the fact (7) is not a convex optimization problem, the presence of the adaptive term allows us to use techniques applied for MD in convex analysis and establish convergence to the global optimum with rates of ˜ O (1 / √ N ) and ˜ O (1 /N ) for the unregularized and regularized case, respectively.

## Algorithm 1 Uniform TRPO

```
initialize: t k , γ , λ , π 0 is the uniform policy. for k = 0 , 1 , ... do v π k = ( I -γP π k ) -1 c π k λ for ∀ s ∈ S do for ∀ a ∈ A do q π k λ ( s, a ) ← c π k λ ( s, a ) + γ ∑ s ′ p ( s ′ | s, a ) v π k λ ( s ′ ) end for π k +1 ( ·| s ) ← PolicyUpdate( π ( ·| s ) , q π k λ ( s, · ) , t k , λ ) end for end for
```

Uniform TRPO repeats the following iterates

<!-- formula-not-decoded -->

The update rule resembles MD's updating-rule (2). The updated policy minimizes the linear approximation while being not 'too-far' from the current policy due to the presence of B ω ( π, π k ) . However, and unlike MD's update rule, the Bregman distance is scaled by the adaptive term ( I -γP π k ) -1 . Applying Proposition 1, we see why this adaptive term is so natural for RL,

<!-- formula-not-decoded -->

Since ( I -γP π k ) -1 ≥ 0 component-wise, minimizing (12) is equivalent to minimizing the vector ( ∗ ) . This results in a simplified update rule: instead of minimizing over ∆ S A we minimize over ∆ A for each s ∈ S independently (see Appendix C.1). For each s ∈ S the policy is updated by

<!-- formula-not-decoded -->

This is the update rule of Algorithm 1. Importantly, the update rule is a direct consequence of choosing the adaptive scaling for the Bregman distance in (11), and without it, the trust region problem would involve optimizing over ∆ S A .

## Algorithm 2 PolicyUpdate: PPG

<!-- formula-not-decoded -->

## Algorithm 3 PolicyUpdate: NE-TRPO

```
input: π ( · | s ) , q ( s, · ) , t k , λ for a ∈ A do π ( a | s ) ← π ( a | s ) exp( -t k ( q ( s,a )+ λ log π k ( a | s ))) ∑ a ′ ∈A π ( a ′ | s ) exp( -t k ( q ( s,a ′ )+ λ log π k ( a ′ | s ))) end for return π ( · | s )
```

By instantiating the PolicyUpdate procedure with Algorithms 2 and 3 we get the PPG and NE-TRPO, respectively, which are instances of Uniform TRPO. Instantiating PolicyUpdate is equivalent to choosing ω and the induced Bregman distance B ω . In the euclidean case, ω ( · ) = 1 2 ‖·‖ 2 2 (Alg. 2), and in the non-euclidean case, ω ( · ) = H ( · ) (Alg. 3). This comes in complete analogy to the fact Projected Gradient Descent and Exponentiated Gradient Descent are instances of MD with similar choices of ω (Section 2).

With the analogy to MD (2) in mind, one would expect Uniform TRPO, to converge with rates ˜ O (1 / √ N ) and ˜ O (1 /N ) for the unregularized and regularized cases, respectively, similarly to MD. Indeed, the following theorem formalizes this intuition for a proper choice of learning rate. The proof of Theorem 2 extends the techniques of Beck (2017) from convex analysis to the non-convex optimization problem (5), by relying on the adaptive scaling of the Bregman distance in (11) (see Appendix C).

Theorem 2 (Convergence Rate: Uniform TRPO) . Let { π k } k ≥ 0 be the sequence generated by Uniform TRPO. Then, the following holds for all N ≥ 1 :

<!-- formula-not-decoded -->

Theorem 2 establishes that regularization allows faster convergence of ˜ O (1 /N ). It is important to note using such regularization leads to a 'biased' solution: Generally ∥ ∥ v π ∗ λ -v ∗ ∥ ∥ ∞ &gt; 0 , where we denote π ∗ λ as the optimal policy of the regularized MDP. In other words, the optimal policy of the regularized MDP evaluated on the unregularized MDP is not necessarily the optimal one. However, when adding such regularization to the problem, it becomes easier to solve, in the sense Uniform TRPO converges faster (for a proper choice of learning rate).

In the next section, we extend the analysis of Uniform TRPO to Sample-Based TRPO, and relax the assumption of having access to the entire state space in each iteration, while still securing similar convergence rates in N .

## 6 Exact and Sample-Based TRPO

In the previous section we analyzed Uniform TRPO, which uniformly minimizes the vector v π . Practically, in largescale problems, such an objective is infeasible as one cannot access the entire state space, and less ambitious goal is usually defined (Sutton et al. 2000; Kakade and Langford 2002; Schulman et al. 2015). The objective usually minimized is the scalar objective (8), the expectation of v π λ ( s ) under a measure µ , min π ∈ ∆ S A E s ∼ µ [ v π λ ( s )] = min π ∈ ∆ S A µv π λ .

Starting from the seminal work on CPI, it is common to assume access to the environment in the form of a ν -restart model . Using a ν -restart model, the algorithm interacts with an MDP in an episodic manner. In each episode k , the starting state is sampled from the initial distribution s 0 ∼ ν , and the algorithm samples a trajectory ( s 0 , r 0 , s 1 , r 1 , ... ) by following a policy π k . As mentioned in Kakade and others (2003), a ν -restart model is a weaker assumption than an access to the true model or a generative model, and a stronger assumption than the case where no restarts are allowed.

To establish global convergence guarantees for CPI, Kakade and Langford (2002) have made the following assumption, which we also assume through the rest of this section:

<!-- formula-not-decoded -->

∥ ∥ ∣ ∣ The term C π ∗ is known as a concentrability coefficient and appears often in the analysis of policy search algorithms (Kakade and Langford 2002; Scherrer and Geist 2014; Bhandari and Russo 2019). Interestingly, C π ∗ is considered the 'best' one among all other existing concentrability coefficients in approximate Policy Iteration schemes (Scherrer 2014), in the sense it can be finite when the rest of them are infinite.

## 6.1 Warm Up: Exact TRPO

We split the discussion on the sample-based version of TRPO: we first discuss Exact TRPO which minimizes the scalar µv π λ (8) instead of minimizing the vector v π λ (7) as Uniform TRPO, while having an exact access to the gradients. Importantly, its updating rule is the same update rule used in NE-TRPO (Schulman et al. 2015, Equation 12), which uses the adaptive proximity term, and is described there as a heuristic. Specifically, there are two minor discrepancies between NE-TRPO and Exact TRPO: 1) We use a penalty formulation instead of a constrained optimization problem. 2) The policies in the Kullback-Leibler divergence are reversed. Exact TRPO is a straightforward adaptation of Uniform TRPO to solve (8) instead of (7) as we establish in Proposition 3. Then, in the next section, we extend Exact TRPO to a sample-based version with provable guarantees.

With the goal of minimizing the objective µv π λ , Exact TRPO repeats the following iterates

<!-- formula-not-decoded -->

Its update rule resembles MD's update rule (11), but uses the ν -restart distribution for the linearized term. Unlike in MD(2), the Bregman distance is scaled by an adaptive scaling factor d ν,π k , using ν and the policy π k by which the algorithm interacts with the MDP. This update rule is motivated by the one of Uniform TRPO analyzed in previous section (11) as the following straightforward proposition suggests (Appendix D.2):

Proposition 3 (Uniform to Exact Updates) . For any π, π k ∈ ∆ S A

<!-- formula-not-decoded -->

Meaning, the proximal objective solved in each iteration of Exact TRPO (14) is the expectation w.r.t. the measure ν of the objective solved in Uniform TRPO (11).

Similarly to the simplified update rule for Uniform TRPO (12), by using the linear approximation in Proposition 1, it can be easily shown that using the adaptive proximity term allows to obtain a simpler update rule for Exact TRPO. Unlike Uniform TRPO which updates all states, Exact TRPO updates only states for which d ν,π k ( s ) &gt; 0 . Denote S d ν,π k = { s : d ν,π k ( s ) &gt; 0 } as the set of these states. Then, Exact TRPO is equivalent to the following update rule (see Appendix D.2), ∀ s ∈ S d ν,π k :

<!-- formula-not-decoded -->

i.e., it has the same updates as Uniform TRPO, but updates only states in S d ν,π k . Exact TRPO converges with similar rates for both the regularized and unregularized cases, as Uniform TRPO. These are formally stated in Appendix D.

## 6.2 Sample-Based TRPO

In this section we derive and analyze the sample-based version of Exact TRPO, and establish high-probability convergence guarantees in a batch setting. Similarly to the previous section, we are interested in minimizing the scalar objective µv π λ (8). Differently from Exact TRPO which requires an access to a model and to simultaneous updates in all states in S d ν,π k , Sample-Based TRPO assumes access to a ν -restart model. Meaning, it can only access sampled trajectories and restarts according to the distribution ν .

## Algorithm 4 Sample-Based TRPO

<!-- formula-not-decoded -->

Sample-Based TRPO samples M k trajectories per episode. In every trajectory of the k -th episode, it first samples s m ∼ d ν,π k and takes an action a m ∼ U ( A ) where U ( A ) is the uniform distribution on the set A . Then, by following the current policy π k , it estimates q π k λ ( s m , a m ) using a rollout (possibly truncated in the infinite horizon case). We denote this estimate as ˆ q π k λ ( s m , a m , m ) and observe it is (nearly) an unbiased estimator of q π k λ ( s m , a m ) . We assume that each rollout runs sufficiently long so that the bias is small enough (the sampling process is fully described in Appendix E.2). Based on this data, Sample-Based TRPO updates the policy at the end of the k -th episode, by the following proximal problem,

<!-- formula-not-decoded -->

/BD The following proposition motivates the study of this update rule and formalizes its relation to Exact TRPO:

where the estimation of the gradient is ˆ ∇ νv π k λ [ m ] := 1 1 -γ ( A ˆ q π k λ ( s m , · , m ) {· = a m } + λ ∇ ω ( s m ; π k )) .

Proposition 4 (Exact to Sample-Based Updates) . Let F k be the σ -field containing all events until the end of the k -1 episode. Then, for any π, π k ∈ ∆ S A and every sample m ,

<!-- formula-not-decoded -->

Meaning, the expectation of the proximal objective of Sample-Based TRPO (15) is the proximal objective of Exact TRPO (14). This fact motivates us to study this algorithm, anticipating it inherits the convergence guarantees of its exacted counterpart.

Like Uniform and Exact TRPO, Sample-Based TRPO has a simpler update rule, in which, the optimization takes place on every visited state at the k -th episode. This comes in contrast to Uniform and Exact TRPO which require access to all states in S or S d ν,π k , and is possible due to the sample-based adaptive scaling of the Bregman distance. Let S k M be the set of visited states at the k -th episode, n ( s, a ) the number of times ( s m , a m ) = ( s, a ) at the k -th episode, and

<!-- formula-not-decoded -->

is the empirical average of all rollout estimators for q π k λ ( s, a ) gathered in the k -th episode ( m i is the episode in which ( s m , a m ) = ( s, a ) for the i -th time). If the state action pair ( s, a ) was not visited at the k -th episode then ˆ q π k λ ( s, a ) = 0 . Given these definitions, Sample-Based TRPO updates the policy for all s ∈ S k M by a simplified update rule:

<!-- formula-not-decoded -->

As in previous sections, the euclidean and non-euclidean choices of ω correspond to a PPG and NE-TRPO instances of Sample-Based TRPO. The different choices correspond to instantiating PolicyUpdate with the subroutines 2 or 3. Generalizing the proof technique of Exact TRPO and using standard concentration inequalities, we derive a highprobability convergence guarantee for Sample-Based TRPO (see Appendix E). An additional important lemma for the proof is Lemma 27 provided in the appendix. This lemma bounds the change ∇ ω ( π k ) - ∇ ω ( π k +1 ) between consecutive episodes by a term proportional to t k . Had this bound been t k -independent, the final results would deteriorate significantly.

Theorem 5 (Convergence Rate: Sample-Based TRPO) . Let { π k } k ≥ 0 be the sequence generated by Sample-Based TRPO, using M k ≥ O ( A 2 C 2 max ,λ ( S log A +log 1 /δ ) (1 -γ ) 2 /epsilon1 2 ) samples in each iteration, and { µv k best } k ≥ 0 be the sequence of best achieved values, µv N best := arg min k =0 ,...,N µv π k λ -µv ∗ λ . Then, with probability greater than 1 -δ for every /epsilon1 &gt; 0 the following holds for all N ≥ 1 :

<!-- formula-not-decoded -->

Table 1: The sample complexity of Sample-Based TRPO (TRPO) and CPI. For TRPO, the best policy so far is returned, where for CPI, the last policy π N is returned.

| Method                       | Sample Complexity                                                      |
|------------------------------|------------------------------------------------------------------------|
| TRPO (this work)             | C 2 ω, 1 A 2 C 4 max ( S +log 1 δ ) (1 - γ ) 3 /epsilon1 4             |
| Regularized TRPO (this work) | C 2 ω, 1 C ω, 2 A 2 C 4 max ,λ ( S +log 1 δ ) λ (1 - γ ) 4 /epsilon1 3 |
| CPI (Kakade and Langford)    | A 2 C 4 max ( S +log 1 δ ) (1 - γ ) 5 /epsilon1 4                      |

Where C ω, 2 = 1 for the euclidean case, and C ω, 2 = A 2 for the non-euclidean case.

Similarly to Uniform TRPO, the convergence rates are ˜ O (1 / √ N ) and ˜ O (1 /N ) for the unregularized and regularized cases, respectively. However, the Sample-Based TRPO converges to an approximate solution, similarly to CPI. The sample complexity for a C π ∗ /epsilon1 (1 -γ ) 2 error, the same as the error of CPI, is given in Table 6.2. Interestingly, SampleBased TRPO has better polynomial sample complexity in (1 -γ ) -1 relatively to CPI. Importantly, the regularized versions have a superior sample-complexity in /epsilon1 , which can explain the empirical success of using regularization.

Remark 1 (Optimization Perspective) . From an optimization perspective, CPI can be interpreted as a sample-based Conditional Gradient Descent (Frank-Wolfe) for solving MDPs (Scherrer and Geist 2014). With this in mind, the two analyzed instances of Sample-Based TRPO establish the convergence of sample-based projected and exponentiated gradient descent methods for solving MDPs: PPG and NETRPO. It is well known that a convex problem can be solved with any one of the three aforementioned methods. The convergence guarantees of CPI together with the ones of Sample-Based TRPO establish the same holds for RL.

Remark 2 (Is Improvement and Early Stopping Needed?) . Unlike CPI, Sample-Based TRPO does not rely on improvement arguments or early stopping. Even so, its asymptotic performance is equivalent to CPI, and its sample complexity has better polynomial dependence in (1 -γ ) -1 . This questions the necessity of ensuring improvement for policy search methods, heavily used in the analysis of these methods, yet less used in practice, and motivated by the analysis of CPI.

## 7 Related Works

The empirical success of policy search and regularization techniques in RL (Peters and Schaal 2008; Mnih et al. 2016; Schulman et al. 2015; Schulman et al. 2017) led to nonnegligible theoretical analysis of these methods. Gradient based policy search methods were mostly analyzed in the function approximation setting, e.g., (Sutton et al. 2000; Bhatnagar et al. 2009; Pirotta, Restelli, and Bascetta 2013; Dai et al. 2018; Papini, Pirotta, and Restelli 2019; Bhandari and Russo 2019). There, convergence to a local optimum was established under different conditions and several aspects of policy search methods were investigated. In this work, we study a trust-region based, as opposed to gradient based, policy search method in tabular RL and establish global convergence guarantees. Regarding regularization in TRPO, in Neu, Jonsson, and G´ omez (2017) the authors analyzed entropy regularized MDPs from a linear programming perspective for average-reward MDPs. Yet, convergence rates were not supplied, as opposed to this paper.

In Geist, Scherrer, and Pietquin (2019) different aspects of regularized MDPs were studied, especially, when combined with MD-like updates in an approximate PI scheme (with partial value updates). The authors focus on update rules which require uniform access to the state space of the form π k +1 = arg min π ∈ ∆ S A 〈 q k , π -π k 〉 + B ω ( π, π k ) , similarly to the simplified update rule of Uniform TRPO (13) with a fixed learning rate, t k = 1 . In this paper, we argued it is instrumental to view this update rule as an instance of the more general update rule (11), i.e., MD with an adaptive proximity term. This view allowed us to formulate and analyze the adaptive Sample-Based TRPO, which does not require uniform access to the state space. Moreover, we proved Sample-Based TRPO inherits the same asymptotic performance guarantees of CPI. Specifically, the quality of the policy Sample-Based TRPO outputs depends on the concentrability coefficient C π ∗ . The results of Geist, Scherrer, and Pietquin (2019) in the approximate setting led to a worse concentrability coefficient, C i q , which can be infinite even when C π ∗ is finite (Scherrer 2014) as it depends on the worst case of all policies.

In a recent work of Agarwal et al. (2019), Section 4.2, the authors study a variant of Projected Policy Gradient Descent and analyze it under the assumption of exact gradients and uniform access to the state space. The proven convergence rate depends on both S and C π ∗ whereas the convergence rate of Exact TRPO (Section 6.1) does not depend on S nor on C π ∗ (see Appendix D.4), and is similar to the guarantees of Uniform TRPO (Theorem 2). Furthermore, the authors do not establish faster rates for regularized MDPs. It is important to note their projected policy gradient algorithm is different than the one we study, which can explain the discrepancy between our results. Their projected policy gradient updates by π k +1 ∈ P ∆ S A ( π k -η ∇ µv π k ) , whereas, the Projected Policy Gradient studied in this work applies a different update rule based on the adaptive scaling of the Bregman distance.

Lastly, in another recent work of Liu et al. (2019) the authors established global convergence guarantees for a sampled-based version of TRPO when neural networks are used as the q -function and policy approximators. The sample complexity of their algorithm is O ( /epsilon1 -8 ) (as opposed to O ( /epsilon1 -4 ) we obtained) neglecting other factors. It is an interesting question whether their result can be improved.

## 8 Conclusions and Future Work

We analyzed the Uniform and Sample-Based TRPO methods. The first is a planning, trust region method with an adaptive proximity term, and the latter is an RL sample- based version of the first. Different choices of the proximity term led to two instances of the TRPO method: PPG and NE-TRPO.For both, we proved ˜ O (1 / √ N ) convergencerate to the global optimum, and a faster ˜ O (1 /N ) rate for regularized MDPs. Although Sample-Based TRPO does not necessarily output an improving sequence of policies, as CPI, its best policy in hindsight does improve. Furthermore, the asymptotic performance of Sample-Based TRPO is equivalent to that of CPI, and its sample complexity exhibits better dependence in (1 -γ ) -1 . These results establish the popular NE-TRPO (Schulman et al. 2015) should not be interpreted as an approximate heuristic to CPI but as a viable alternative.

In terms of future work, an important extension of this study is deriving algorithms with linear convergence, or, alternatively, establish impossibility results for such rates in RL problems. Moreover, while we proved positive results on regularization in RL, we solely focused on the question of optimization. We believe that establishing more positive as well as negative results on regularization in RL is of value. Lastly, studying further the implication of the adaptive proximity term in RL is of importance due to the empirical success of NE-TRPO and its now established convergence guarantees.

## 9 Acknowledgments

We would like to thank Amir Beck for illuminating discussions regarding Convex Optimization and Nadav Merlis for helpful comments. This work was partially funded by the Israel Science Foundation under ISF grant number 1380/16.

## References

- [Agarwal et al. 2019] Agarwal, A.; Kakade, S. M.; Lee, J. D.; and Mahajan, G. 2019. Optimality and approximation with policy gradient methods in markov decision processes. arXiv preprint arXiv:1908.00261 .
- [Ahmed et al. 2019] Ahmed, Z.; Le Roux, N.; Norouzi, M.; and Schuurmans, D. 2019. Understanding the impact of entropy on policy optimization. In International Conference on Machine Learning , 151-160.
- [Beck and Teboulle 2003] Beck, A., and Teboulle, M. 2003. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters 31(3):167-175.
- [Beck 2017] Beck, A. 2017. First-order methods in optimization , volume 25. SIAM.
- [Bertsimas and Tsitsiklis 1997] Bertsimas, D., and Tsitsiklis, J. N. 1997. Introduction to linear optimization , volume 6. Athena Scientific Belmont, MA.
- [Bhandari and Russo 2019] Bhandari, J., and Russo, D. 2019. Global optimality guarantees for policy gradient methods. arXiv preprint arXiv:1906.01786 .
- [Bhatnagar et al. 2009] Bhatnagar, S.; Sutton, R. S.; Ghavamzadeh, M.; and Lee, M. 2009. Natural actor-critic algorithms. Automatica 45(11):2471-2482.
- [Chow, Nachum, and Ghavamzadeh 2018] Chow, Y.; Nachum, O.; and Ghavamzadeh, M. 2018. Path consistency
- learning in tsallis entropy regularized mdps. In International Conference on Machine Learning , 978-987.
- [Dai et al. 2018] Dai, B.; Shaw, A.; Li, L.; Xiao, L.; He, N.; Liu, Z.; Chen, J.; and Song, L. 2018. Sbeed: Convergent reinforcement learning with nonlinear function approximation. In Dy, J., and Krause, A., eds., Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , 1125-1134. Stockholmsmssan, Stockholm Sweden: PMLR.

[Farahmand, Szepesv´ ari, and Munos 2010] Farahmand,

- A. M.; Szepesv´ ari, C.; and Munos, R. 2010. Error propagation for approximate policy and value iteration. In Advances in Neural Information Processing Systems , 568-576.
2. [Fox, Pakman, and Tishby 2016] Fox, R.; Pakman, A.; and Tishby, N. 2016. Taming the noise in reinforcement learning via soft updates. In Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence , 202-211. AUAI Press.
3. [Geist, Scherrer, and Pietquin 2019] Geist, M.; Scherrer, B.; and Pietquin, O. 2019. A theory of regularized markov decision processes. In International Conference on Machine Learning , 2160-2169.
4. [Juditsky, Nemirovski, and others 2011] Juditsky, A.; Nemirovski, A.; et al. 2011. First order methods for nonsmooth convex large-scale optimization, i: general purpose methods. Optimization for Machine Learning 121-148.
5. [Kakade and Langford 2002] Kakade, S., and Langford, J. 2002. Approximately optimal approximate reinforcement learning. In ICML , volume 2, 267-274.
6. [Kakade and others 2003] Kakade, S. M., et al. 2003. On the sample complexity of reinforcement learning . Ph.D. Dissertation, University of London London, England.
7. [Liu et al. 2019] Liu, B.; Cai, Q.; Yang, Z.; and Wang, Z. 2019. Neural proximal/trust region policy optimization attains globally optimal policy. arXiv preprint arXiv:1906.10306 .
8. [Mnih et al. 2016] Mnih, V.; Badia, A. P.; Mirza, M.; Graves, A.; Lillicrap, T.; Harley, T.; Silver, D.; and Kavukcuoglu, K. 2016. Asynchronous methods for deep reinforcement learning. In International conference on machine learning , 1928-1937.
9. [Nachum et al. 2017] Nachum, O.; Norouzi, M.; Xu, K.; and Schuurmans, D. 2017. Trust-pcl: An off-policy trust region method for continuous control. arXiv preprint arXiv:1707.01891 .
10. [Nedic and Lee 2014] Nedic, A., and Lee, S. 2014. On stochastic subgradient mirror-descent algorithm with weighted averaging. SIAM Journal on Optimization 24(1):84-107.
11. [Nesterov 1998] Nesterov, Y. 1998. Introductory lectures on convex programming volume i: Basic course . Springer, New York, NY.
12. [Neu, Jonsson, and G´ omez 2017] Neu, G.; Jonsson, A.; and G´ omez, V. 2017. A unified view of entropyregularized markov decision processes. arXiv preprint arXiv:1705.07798 .
13. [Papini, Pirotta, and Restelli 2019] Papini, M.; Pirotta, M.; and Restelli, M. 2019. Smoothing policies and safe policy gradients. arXiv preprint arXiv:1905.03231 .
14. [Peters and Schaal 2008] Peters, J., and Schaal, S. 2008. Natural actor-critic. Neurocomputing 71(7-9):1180-1190.
15. [Pirotta, Restelli, and Bascetta 2013] Pirotta, M.; Restelli, M.; and Bascetta, L. 2013. Adaptive step-size for policy gradient methods. In Advances in Neural Information Processing Systems , 1394-1402.
16. [Scherrer and Geist 2014] Scherrer, B., and Geist, M. 2014. Local policy search in a convex space and conservative policy iteration as boosted policy search. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases , 35-50. Springer.
17. [Scherrer 2014] Scherrer, B. 2014. Approximate policy iteration schemes: a comparison. In International Conference on Machine Learning , 1314-1322.
18. [Schulman et al. 2015] Schulman, J.; Levine, S.; Abbeel, P.; Jordan, M.; and Moritz, P. 2015. Trust region policy optimization. In International Conference on Machine Learning , 1889-1897.
19. [Schulman et al. 2017] Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 .
20. [Sutton and Barto 2018] Sutton, R. S., and Barto, A. G. 2018. Reinforcement learning: An introduction . MIT press.
21. [Sutton et al. 2000] Sutton, R. S.; McAllester, D. A.; Singh, S. P.; and Mansour, Y. 2000. Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems , 10571063.

## List of Appendices

| A   | Assumptions of Mirror Descent                 | Assumptions of Mirror Descent                                                                           | 11   |
|-----|-----------------------------------------------|---------------------------------------------------------------------------------------------------------|------|
| B   | Policy                                        | Gradient, and Directional Derivatives for Regularized MDPs                                              | 11   |
|     | B.1 . . . . .                                 | Extended Value Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          | 11   |
|     | B.2                                           | Policy Gradient Theorem for Regularized MDPs . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | 13   |
|     | B.3                                           | The Linear Approximation of the Policy's Value and The Directional Derivative for Regularized MDPs      | 14   |
| C   | Uniform Trust Region Policy Optimization      | Uniform Trust Region Policy Optimization                                                                | 16   |
|     | C.1 . . . .                                   | Uniform TRPO Update Rule . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          | 16   |
|     | C.2 . . . .                                   | The PolicyUpdate procedure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        | 17   |
|     | C.3                                           | Fundamental Inequality for Uniform TRPO . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 18   |
|     | C.4 . . . . . . . .                           | Proof of Theorem 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .              | 19   |
| D   | Exact Trust Region Policy Optimization        | Exact Trust Region Policy Optimization                                                                  | 22   |
|     | D.1                                           | Relation Between Uniform and Exact TRPO . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 23   |
|     | D.2 . . . . . .                               | Exact TRPO Update rule . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .            | 24   |
|     | D.3                                           | Fundamental Inequality of Exact TRPO . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | 25   |
|     | D.4                                           | Convergence proof of Exact TRPO . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 27   |
| E   | Sample-Based Trust Region Policy Optimization | Sample-Based Trust Region Policy Optimization                                                           | 31   |
|     | E.1                                           | Relation Between Exact and Sample-Based TRPO . . . . . . . . . . . . . . . . . . . . . . . . . . . .    | 31   |
|     | E.2 .                                         | Sample-Based TRPO Update Rule . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     | 32   |
|     | E.3 . . . . .                                 | Proof Sketch of Theorem 5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         | 34   |
|     | E.4                                           | Fundamental Inequality of Sample-Based TRPO . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 35   |
|     | E.5 . . . .                                   | Approximation Error Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         | 36   |
|     | E.6 . . . . . . . . .                         | Proof of Theorem 5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                | 45   |
|     | E.7                                           | Sample Complexity of Sample-Based TRPO . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    | 48   |
| F   | Useful Lemmas                                 | Useful Lemmas                                                                                           | 51   |
| G   | Useful Lemmas from Convex Analysis            | Useful Lemmas from Convex Analysis                                                                      |      |

## A Assumptions of Mirror Descent

Assumption 2 (properties of Bregman distance) .

- (A) ω is proper closed and convex.
- (B) ω is differentiable over dom ( ∂ω ) .
- (C) C ⊆ dom ( ω )
- (D) ω + δ C is σ -strongly convex ( σ &gt; 0 )

Assumption 2 is the main assumption regarding the underlying Bregman distance used in Mirror Descent. In our analysis, we have two common choice of ω : a) the negative entropy function, denoted as H ( · ) , for which the corresponding Bregman distance is B ω ( · , · ) = d KL ( ·||· ) . b) the euclidean norm ω ( · ) = 1 2 ‖·‖ 2 , for which the resulting Bregman distance is the euclidean distance. The convex optimization domain C is in our case ∆ S A , the state-wise unit simplex over the space of actions. For both choices, the assumption holds. Finally, δ C ( x ) is an extended real valued function which describes the optimization domain C . It is defined as follows: For x ∈ C , δ C ( x ) = 0 . For x / ∈ C , δ C ( x ) = ∞ . For more details, see (Beck 2017).

We go on to define the second assumption regarding the optimization problem:

## Assumption 3.

- (A) f : E → ( -∞ , ∞ ] is proper closed.
- (B) C ⊆ E is nonempty closed and convex.
- (C) C ⊆ int ( dom ( f )) .
- (D) The optimal set of (P) is nonempty.

## B Policy Gradient, and Directional Derivatives for Regularized MDPs

In this section we re-derive the Policy Gradient Theorem (Sutton et al. 2000) for regularized MDPs when tabular representation is used. Meaning, we explicitly calculate the derivative ∇ π v π λ ( s ) . Based on this result, we derive the directional derivative, or the linear approximation of the objective functions, 〈∇ π v π λ ( s ) , π -π ′ 〉 , 〈∇ π µv π λ ( s ) , π -π ′ 〉 .

## B.1 Extended Value Functions

To formally study ∇ π v π λ ( s ) we need to define value functions v π when π is outside of the simplex ∆ S A , since when π ( a | s ) changes infinitesimally, π ( · | s ) does not remain a valid probability distribution. To this end, we study extended value functions denoted by v ( y ) ∈ R S for y ∈ R S × A , and denote v s ( y ) as the component of v ( y ) which corresponds to the state s . Furthermore, we define the following cost and dynamics,

<!-- formula-not-decoded -->

Definition 1 (Extended value and q functions.) . An extended value function is a mapping v : R S × A → R S , such that for y ∈ R S × A

where ω s ( y ) := ω ( y ( · | s )) for ω : R A → R , p y ∈ R S × S and c y λ ∈ R S .

<!-- formula-not-decoded -->

Similarly, an extended q -function is a mapping q : R S × A → R S × A , such that its s, a element is given by

<!-- formula-not-decoded -->

When y ∈ ∆ S A is a policy, π , we denote v ( π ) := v π λ ∈ R S , q ( π ) = q π λ ∈ R S × A .

Note that in this section we use different notations than the rest of the paper, in order to generalize the discussion and keep it out of the regular RL conventions.

The following proposition establishes that v ( y ) the fixed point of a corresponding Bellman operator when y is close to the simplex component-wise.

Lemma 6. Let y ∈ { y ′ ∈ R S × A : ∀ s, ∑ a | y ′ ( a | s ) | &lt; 1 γ } . Define the operator T y : R S → R S , such that for any v ∈ R S ,

Then,

1. T y is a contraction operator in the max norm.
2. Its fixed-point is v ( y ) and satisfies v s ( y ) = ( T y v ( y )) s .

Proof. We start by proving the first claim. Unlike in classical results on MDPs, y is not a policy. However, since it is not 'too far' from being a policy we get the usual contraction property by standard proof techniques.

Let v ′ , v ∈ R S , and assume ( T y v ′ ) s ≥ ( T y v ) s .

In the fourth relation we used the assumption that γ ∑ a | y ( a | s ) | &lt; 1 . Repeating the same proof for the other case where ( T y v ′ ) s &lt; ( T y v ) s , concludes the proof of the first claim.

<!-- formula-not-decoded -->

To prove the second claim, we use the definition of v ( y ) .

<!-- formula-not-decoded -->

In the third relation we used the distributive property of matrix multiplication and in the forth relation we used the definition of v ( y ) . Thus, v ( y ) = T y v ( y ) , i.e., v ( y ) is the fixed point of the operator T y .

<!-- formula-not-decoded -->

## B.2 Policy Gradient Theorem for Regularized MDPs

We now derive the Policy Gradient Theorem for regularized MDPs for tabular policy representation. Specifically, we use the notion of an extended value function and an extended q -functions defined in the previous section.

Lemma 7. Let y ∈ { y ′ ∈ R S × A : ∀ s, ∑ a | y ′ ( a | s ) | &lt; 1 γ } . Then,

Proof. Using (17), we get

<!-- formula-not-decoded -->

where the last equality is by the fixed-point property of Lemma 6.

We now derive the Policy Gradient Theorem for extended (regularized) value functions.

Theorem 8 (Policy Gradient for Extended Regularized Value Functions) . Let y ∈ { y : ∀ s, ∑ a | y ( a | s ) | &lt; 1 γ } . Furthermore, consider a fixed s, a and ¯ s . Then, where p y ( s t | s ) = ∑ s 1 ,..,s t p y ( s t | s t -1 ) · · · p y ( s 1 | s ) , and p y t ( s 0 | s ) = 1 .

<!-- formula-not-decoded -->

Proof. Following similar derivation to the original Policy Gradient Theorem (Sutton et al. 2000), for every s ,

<!-- formula-not-decoded -->

We now explicitly write the last term,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging this back yields,

<!-- formula-not-decoded -->

Iteratively applying this relation yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where,

y

<!-- formula-not-decoded -->

Returning to the specific notation for RL, defined in Section 3, by setting y = π , i.e., when y is a policy, we get the Policy Gradient Theorem for regularized MDPs, since for all s , ∑ a ′ π ( a ′ | s ) = 1 .

Corollary 9 (Policy Gradient for Regularized MDPs) . Let π ∈ ∆ S A . Then, ∇ π v π ∈ R S × S × A and

<!-- formula-not-decoded -->

## B.3 The Linear Approximation of the Policy's Value and The Directional Derivative for Regularized MDPs

In this section, we derive the directional derivative in policy space for regularized MDPs with tabular policy representation.

The linear approximation of the value function of the policy π ′ , around the policy π , is given by

<!-- formula-not-decoded -->

In the MD framework, we take the arg min w.r.t. to this linear approximation. Note that the minimizer is independent on the zeroth term, v π λ , and thus the optimization problem depends only on the directional derivative, 〈∇ π v π λ , π ′ -π 〉 . To keep track with the MD formulation, we chose to refer to Proposition 1 as the 'linear approximation of a policy's value', even though it is actually the directional derivative.

Proposition 1 (Linear Approximation of a Policy's Value) . Let π, π ′ ∈ ∆ S A , and d µ,π = (1 -γ ) µ ( I -γP π ) -1 . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

See that (10) is a vector in R S , whereas (9) is a scalar.

Proof. We start by proving the first claim. Consider the inner product, 〈 ∇ π ( ·| ¯ s ) v π ( s ) , π ′ ( · | ¯ s ) -π ( · | ¯ s ) 〉 . By the linearity of the inner product and using Corollary 9 we get,

<!-- formula-not-decoded -->

The following relations hold.

<!-- formula-not-decoded -->

The third relation holds by the fixed-point property of v π λ , and the last relation is by the definition of the regularized Bellman operator.

Plugging this back into (18), we get,

<!-- formula-not-decoded -->

Thus, we have that

Where the third relation is by (20), the forth by defining the matrix ∑ ∞ t =0 γ t P π = ( I -γP π ) -1 , and the fifth by the definition of matrix-vector product.

<!-- formula-not-decoded -->

To prove the second claim, multiply both sides of the first relation (9) by µ . For the LHS we get,

<!-- formula-not-decoded -->

In the first and second relation we used the linearity of the inner product and the derivative, and in the third relation the definition of µv π . Lastly, observe that multiplying the RHS by µ yields µ ( I -γP π ) -1 = 1 1 -γ d µ,π .

## C Uniform Trust Region Policy Optimization

In this Appendix, we derive the Uniform TRPO algorithm (Algorithm 1) and prove its convergence for both the unregularized and regularized versions. As discussed in Section 5, both Uniform Projected Policy Gradient and Uniform NE-TRPO are instances of Uniform TRPO, by a proper choice of the Bregman distance. In Appendix C.1, we explicitly show that the iterates

<!-- formula-not-decoded -->

result in algorithm 1. In Appendix C.2, we derive the updates of the PolicyUpdate procedure, Algorithms 2 and 3. Then, we turn to analyze Uniform TRPO and its instances in Appendix C.3. Specifically, we derive the fundamental inequality for Unifom TRPO, similarly to the fundamental inequality for Mirror Descent (Beck 2017, Lemma-9.13). Although the objective is not convex, we show that due to the adaptive scaling, by applying the linear approximation of the value of regularized MDPs (Proposition 1), we can repeat similar derivation to that of MD, with some modifications. Finally, in Appendix C.4, we go on to prove convergence rates for both the unregularized ( λ = 0 ) and regularized ( λ &gt; 0 ) versions of Uniform TRPO, using a right choice of stepsizes.

## C.1 Uniform TRPO Update Rule

In each TRPO step, we solve the following optimization problem:

<!-- formula-not-decoded -->

where the second transition holds by plugging in the linear approximation (Proposition 1), and the last transition holds since ( I -γP π k ) -1 &gt; 0 and does not depend on π . Thus, we have,

<!-- formula-not-decoded -->

By discarding terms which do not depend on π , we get

<!-- formula-not-decoded -->

We are now ready to write (13), using the fact that (23), can be written as the following state-wise optimization problem: For every s ∈ S ,

<!-- formula-not-decoded -->

## C.2 The PolicyUpdate procedure

Next, we write the solution for the optimization problem for each of the cases:

By plugging Lemma 24 into (22)

<!-- formula-not-decoded -->

Or again in a state-wise form,

<!-- formula-not-decoded -->

Using (24), we can plug in the solution of the MD iteration for each of the different cases.

Euclidean Case: For ω chosen to be the L 2 norm, the solution to (24) is the orthogonal projection. For all s ∈ S the policy is updated according to

<!-- formula-not-decoded -->

where P ∆ A is the orthogonal projection operator over the simplex. Refer to (Beck 2017) for details.

Finally, dividing by the constant 1 -λt k does not change the optimizer. Thus,

<!-- formula-not-decoded -->

Non-Euclidean Case: For ω chosen to be the negative entropy, (24) has the following analytic solution for all s ∈ S ,

<!-- formula-not-decoded -->

where the first transition is by substituting ω and the Bregman distance, the second is by the definition of the Bregman distance, and the last transition is by omitting constant factors.

By using (Beck 2017, Example 3.71), we get

<!-- formula-not-decoded -->

Now, using the derivative of the negative entropy function H ( · ) , we have that for every s, a , which concludes the result.

<!-- formula-not-decoded -->

## C.3 Fundamental Inequality for Uniform TRPO

Central to the following analysis is Lemma 10, which we prove in this section. This lemma replaces Lemma (Beck 2017)[9.13] from which it inherits its name, for the RL non-convex case. It has two main differences relatively to Lemma (Beck 2017)[9.13]: (a) The inequality is in vector form (statewise). (b) The non-convexity of f demands replacing the gradient inequality with different proof mechanism, i.e., the directional derivative in RL (see Proposition 1).

Lemma10 (fundamental inequality for Uniform TRPO) . Let { π k } k ≥ 0 be the sequence generated by the uniform TRPO method with stepsizes { t k } k ≥ 0 . Then, for every π and k ≥ 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where h ω is defined in the second claim of Lemma 25, and e is a vector of ones.

Proof. First, notice that assumptions 2 and 3 hold. Assumption 2 is a regular assumption on the Bregman distance, which holds trivially both in the euclidean and non-euclidean case, where the optimization domain is the ∆ S A . Assumption 3 deals with the optimization problem itself and is similar to (Beck 2017, Assumption 9.1) over ∆ A . The only difference is that in our case, the optimization objective v π is non-convex.

Define ψ ( π ) ≡ t k ( I -γP π k ) 〈∇ v π k λ , π 〉 + δ ∆ S A ( π ) where δ ∆ S A ( π ) = 0 when π ∈ ∆ S A and infinite otherwise. Observe it is a convex function in π , as a sum of two convex functions: The first term is linear in π for any π ∈ ∆ S A , and thus convex, and δ ∆ S A ( π ) is convex since ∆ S A is a convex set. Applying the non-euclidean second prox theorem (Theorem 31), with a = π k , b = π k +1 , we get that for any π ∈ ∆ S A ,

<!-- formula-not-decoded -->

By the three-points lemma (30),

<!-- formula-not-decoded -->

which, combined with (27), gives,

<!-- formula-not-decoded -->

Therefore, by simple algebraic mainpulation, we get

<!-- formula-not-decoded -->

where the last equality is due to Proposition 1, and using ( I -γP π k )( I -γP π k ) -1 = I.

Rearranging we get

<!-- formula-not-decoded -->

where the last inequality follows since the Bregman distance is 1 -strongly-convex for our choices of B ω (e.g., Beck 2017, Lemma 9.4(a)).

Furthermore, for every state s ∈ S , where the first inequality is due to the Fenchel's inequality on the convex ‖·‖ 2 and its convex conjugate ‖·‖ 2 ∗ , and the last equality uses the fact that ‖ c ( s, · ) + γ ∑ s ′ p ( s ′ | s, · ) v π k λ ( s ′ ) ‖ ∗ ≤ ‖ c λ ( s, · ) + γ ∑ s ′ p ( s ′ | s, · ) v π k λ ( s ′ ) ‖ ∗ = ‖ q π k λ ( s, · ) ‖ ∗ , and using the repsective bound in Lemma 25.

<!-- formula-not-decoded -->

Plugging the last inequality into (29),

<!-- formula-not-decoded -->

where e is a vector of all ones.

By using Proposition 1 on the LHS, we get,

<!-- formula-not-decoded -->

Lastly,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first relation holds by the second claim in Lemma 29.

## C.4 Proof of Theorem 2

Before proving the theorem, we establish that the policy improves in k for the chosen learning rates.

Lemma 11 (Uniform TRPO Policy Improvement) . Let { π k } k ≥ 0 be the sequence generated by Uniform TRPO. Then, for both the euclidean and non-euclidean versions of the algorithm, for any λ ≥ 0 , the value improves for all k ,

Proof. Restating (28), we have that for any π ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging the closed form of the directional derivative (Proposition (1)), setting π = π k , using B ω ( π k , π k ) = 0 , we get,

<!-- formula-not-decoded -->

The choice of the learning rate and the fact that the Bregman distance is non negative ( λ &gt; 0 , λt k = 1 k +2 ≤ 1 and for λ = 0 the RHS of (30) is positive) implies that

<!-- formula-not-decoded -->

Applying iteratively T π k +1 λ and using its monotonicty we obtain,

<!-- formula-not-decoded -->

where in the last relation we used the fact T π k +1 λ is a contraction operator and its fixed point is v π k +1 λ which proves the claim.

For the sake of completeness and readability, we restate here Theorem 2, this time including all logarithmic factors:

Theorem (Convergence Rate: Uniform TRPO) . Let π

{ k } k ≥ 0 be the sequence generated by Uniform TRPO,

Then, the following holds for all N ≥ 1 .

1. (Unregularized) Let λ = 0 , t k = (1 -γ ) C ω, 1 C max √ k +1 then

<!-- formula-not-decoded -->

2. (Regularized) Let λ &gt; 0 , t k = 1 λ ( k +2) then

<!-- formula-not-decoded -->

Where C ω, 1 = √ A,C ω, 3 = 1 for the euclidean case, and C ω, 1 = 1 , C ω, 3 = log A for the non-euclidean case.

We are now ready to prove Theorem 2, while following arguments from (Beck 2017, Theorem 9.18).

## The Unregularized case

Proof. Applying Lemma 10 with π = π ∗ and λ = 0 (the unregularized case) and let e ∈ R S , a vector ones, the following relations hold.

<!-- formula-not-decoded -->

Summing the above inequality over k = 0 , 1 , ..., N , and noticing we get a telescopic sum gives

<!-- formula-not-decoded -->

where the second relation holds since B ω ( π ∗ , π N +1 ) ≥ 0 component-wise. From which we get the following relations,

<!-- formula-not-decoded -->

In the second relation we multiplied both sides of inequality by ( I -γP π ∗ ) -1 ≥ 0 component-wise. In the third relation we used ( I -γP π ) -1 e = 1 1 -γ e for any π . By Lemma (11) the policies are improving, from which, we get

<!-- formula-not-decoded -->

Combining (33), (34) , and dividing by N ∑ k =0 t k we get the following component-wise inequality,

<!-- formula-not-decoded -->

By plugging in the stepsizes, t k = 1 h ω √ k +1 we get,

<!-- formula-not-decoded -->

Plugging in Lemma 28 and bounding the sums (e.g., by using Beck 2017, Lemma 8.27(a)) yields,

<!-- formula-not-decoded -->

Plugging the expressions for h ω , D ω in Lemma 25 and Lemma 28 we conclude the proof.

## The Regularized case

Proof. Applying Lemma 10 with π = π ∗ and λ &gt; 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging t k = 1 λ ( k +2) and multiplying by λ ( k +2) ,

<!-- formula-not-decoded -->

Summing the above inequality over k = 0 , ..., N yields

<!-- formula-not-decoded -->

as the summation results in a telescopic sum.

Observe that for any π, π ′ and both our choices of ω , ω ( π ) -ω ( π ′ ) ≤ max π | ω ( π ) | . For the euclidean case max π | ω ( π ) | &lt; 1 and for the non euclidean case max π | ω ( π ) | ≤ log A . These bounds are the same bounds as the bound for the Bregman distance, D ω (see Lemma 28). Thus, for both our choices of ω we can bound ω ( π ) -ω ( π ′ ) &lt; D ω .

Furthermore, since B ω ( π ∗ , π N +1 ) ≥ 0 the following bound holds:

<!-- formula-not-decoded -->

and in the third relation we multiplied both side by ( I -γP π ∗ ) -1 ≥ 0 component-wise and used ( I -γP π ) -1 e = 1 1 -γ e for any π .

By Lemma 11 the value v π k λ decreases in k , and, thus,

<!-- formula-not-decoded -->

Combining (35), (36), and dividing by N +1 we get the following component-wise inequality,

<!-- formula-not-decoded -->

Using the fact that N +1 ∑ k =1 1 k ∈ O (log n ) , we get

<!-- formula-not-decoded -->

Plugging the expressions for h ω , D ω in Lemma 25 and Lemma 28 we conclude the proof.

## D Exact Trust Region Policy Optimization

The derivation of Exact TRPO is similar in spirit to the derivation of Uniform TRPO (Appendix C). However, instead of minimizing a vector, the objective to be minimized in this section is the scalar µv π (8). This fact complicates the analysis and requires us assuming a finite concentrability coefficient C π ∗ = ∥ ∥ ∥ d µ,π ∗ ν ∥ ∥ ∥ ∞ &lt; ∞ (Assumption 1), a common assumption in the

RL literature (Kakade and Langford 2002; Farahmand, Szepesv´ ari, and Munos 2010; Scherrer 2014; Scherrer and Geist 2014). This assumption alleviates the need to deal with exploration and allows us to focus on the optimization problem in MDPs in which the stochasticity of the dynamics induces sufficient exploration. We note that assuming a finite C π ∗ is the weakest assumptions among all other existing concentrability coefficients (Scherrer 2014).

The Exact TRPO algorithm is as follows:

```
Algorithm 5 Exact TRPO initialize: t k , γ , λ , π 0 is the uniform policy. for k = 0 , 1 , ... do v π k ← µ ( I -γP π k ) -1 c π k λ S d ν,π k = { s ∈ S : d ν,π k ( s ) > 0 } for ∀ s ∈ S d ν,π k do for ∀ a ∈ A do q π k λ ( s, a ) ← c π λ ( s, a ) + γ ∑ s ′ p ( s ′ | s, a ) v π k λ ( s ′ ) end for π k +1 ( ·| s ) = PolicyUpdate( π k ( ·| s ) , q π k λ ( s, · ) , t k , λ ) end for end for
```

Similarly to Uniform TRPO, the euclidean and non-euclidean choices of ω correspond to a PPG and NE-TRPO instances of Exact TRPO: by instantiating PolicyUpdate with the subroutines 2 or 3 we get the instances of Exact TRPO respectively. A

The main goal of this section is to create the infrastructure for the analysis of Sample-Based TRPO, which is found in Appendix E. Sample-Based TRPO is a sample-based version of Exact TRPO, and for pedagogical reasons we start by analyzing the latter from which the analysis of the first is better motivated.

In this section we prove convergence for Exact TRPO which establishes similar convergence rates as for the Uniform TRPO in the previous section. We now describe the content of each of the subsections: First, in Appendix D.1, we show the connection between Exact TRPO and Uniform TRPO by proving Proposition 4. In Appendix D.2, we formalize the exact version of TRPO. Then, we derive a fundamental inequality that will be used to prove convergence for the exact algorithms (Appendix D.3). This inequality is a scalar version of the vector fundamental inequality derived for Uniform TRPO (Lemma 10). This is done by first deriving a state-wise inequality, and then using Assumption 1 to connect the state-wise local guarantee to a global guarantee w.r.t. the optimal policy π ∗ . Finally, we use the fundamental inequality for Exact TRPO to prove the convergence rates of Exact TRPO for both the unregularized and regularized version (Appendix D.4).

## D.1 Relation Between Uniform and Exact TRPO

Before diving into the proof of Exact TRPO, we prove Proposition 3, which connects the update rules for Uniform and Exact TRPO:

Proposition 3 (Uniform to Exact Updates) . For any π, π k ∈ ∆ S A

<!-- formula-not-decoded -->

Proof. First, notice that for every s ′

<!-- formula-not-decoded -->

where in the second and third transition we used the linearity of the inner product and the derivative, and in the last transition we used the definition of νv π k λ .

Thus, we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second transition is by plugging in (37) and the last transition is by the definition of the stationary distribution d ν,π k .

## D.2 Exact TRPO Update rule

Exact TRPO repeatedly updates the policy by the following update rule (see (14)),

<!-- formula-not-decoded -->

Note that differently than regular MD, the gradient here is w.r.t. to νv π k λ , and not µv π k λ which is the true scalar objective (8). This is due to the fact that d ν,π k is the proper scaling for solving the MDP using the ν -restart model, as can be seen in (39).

Using Proposition 1, the update rule can be written as follows,

<!-- formula-not-decoded -->

Much like the arguments we followed in Section 5, since d ν,π k ≥ 0 component-wise, minimizing (39) is equivalent to minimizing T π λ v π k λ ( s ) -v π k λ ( s ) + 1 t k B ω ( s ; π, π k ) for all s for which d ν,π k ( s ) &gt; 0 . Meaning, the update rule takes the following form,

<!-- formula-not-decoded -->

which can be written equivalently using Lemma 24

<!-- formula-not-decoded -->

which will be use in the next section.

The minimization problem is solved component-wise as in Appendix C.1, equations (25) and (26) for the euclidean and noneuclidean cases, respectively. Thus, the solution of (38) is equivalent to a single iteration of Exact TRPO as given in Algorithm 5.

Remark 3. Interestingly, the analysis does not depend on the updates in states for which d ν,π k ( s ) = 0 . Although this might seem odd, the reason for this indifference is Assumption 1, by which ∀ s, k, d µ,π ∗ ( s ) &gt; 0 → d ν,π k ( s ) &gt; 0 . Meaning, by Assumption 1 in each iteration we update all the states for which d µ,π ∗ ( s ) &gt; 0 . This fact is sufficient to prove the convergence of Exact TRPO, with no need to analyze the performance at states for which d µ,π ∗ ( s ) = 0 and d ν,π k ( s ) &gt; 0 .

## D.3 Fundamental Inequality of Exact TRPO

In this section we will develop the fundamental inequality for Exact TRPO (Lemma 14) based on its updating rule (40). We derive this inequality using two intermediate lemmas: First, in Lemma 12 we derive a state-wise inequality which holds for all states s for which d ν,π k ( s ) &gt; 0 . Then, in Lemma 13, we use Lemma 12 together with Assumption 1 to prove an inequality related to the stationary distribution of the optimal policy d µ,π ∗ . Finally, we prove the fundamental inequality for Exact TRPO using Lemma 29, which allows us to use the local guarantees of the inequality in Lemma 13 for a global guarantee w.r.t. the optimal value, µv ∗ λ .

Lemma 12 (exact state-wise inequality) . For all states s for which d ν,π k ( s ) &gt; 0 the following inequality holds:

<!-- formula-not-decoded -->

where h ω is defined at the third claim of Lemma 25.

Proof. Start by observing that the update rule (41) is applied in any state s for which d ν,π k ( s ) &gt; 0 . By the first order optimality condition for the solution of (41), for any policy π ∈ ∆ A at state s ,

<!-- formula-not-decoded -->

The first term can be bounded as follows.

<!-- formula-not-decoded -->

where the last relation follows from Fenchel's inequality using the euclidean or non-euclidean norm ‖·‖ , and where ‖·‖ ∗ is its dual norm, which is L 2 in the euclidean case, and L ∞ in the non-euclidean case. Note that the norms are applied over the action

space. Furthermore, by adding and subtracting λω ( s ; π ) ,

<!-- formula-not-decoded -->

where the second transition follows the same steps as in equation (19) in the proof of Proposition 1, and the third transition is by the definition of the Bregman distance of ω . Note that (43) is actually given in Lemma 24, but is re-derived here for readability.

From which, we conclude that

<!-- formula-not-decoded -->

where in the last transition we used the third claim of Lemma 25,

We now continue analyzing (2) .

<!-- formula-not-decoded -->

The first relation, ∇ π k +1 B ω ( s ; π k +1 , π k ) = ∇ ω ( s ; π k +1 ) -∇ ω ( s ; π k ) , holds by simply taking the derivative of any Bregman distance w.r.t. π k +1 . The second relation holds by the three-points lemma (Lemma 30). The third relation holds by the strong convexity of the Bregman distance, i.e., 1 2 ‖ x -y ‖ 2 ≤ B ω ( x, y ) , which is straight forward in the euclidean case, and is the well known Pinsker's inequality in the non-euclidean case.

Plugging the above upper bounds for (1) and (2) into (42) we get,

<!-- formula-not-decoded -->

and conclude the proof.

We now turn state another lemma, which connects the state-wise inequality using the discounted stationary distribution of the optimal policy d µ,π ∗

Lemma 13. Assuming 1, the following inequality holds for all π .

<!-- formula-not-decoded -->

where h ω is defined at the third claim of Lemma 25.

Proof. By Assumption 1, for all s for which d µ,π ∗ ( s ) &gt; 0 it also holds that d ν,π k ( s ) &gt; 0 . Thus, for all s for which d µ,π ∗ ( s ) &gt; 0 the component-wise relation in Lemma 12 holds. By multiplying each inequality by the positive number d µ,π ∗ ( s ) and summing over all s we get,

<!-- formula-not-decoded -->

which concludes the proof.

Using the previous lemma, we are ready to prove the following Lemma:

Lemma 14 (fundamental inequality of exact TRPO) . Let { π k } k ≥ 0 be the sequence generated by the TRPO method using stepsizes { t k } k ≥ 0 . Then, for all k ≥ 0

<!-- formula-not-decoded -->

where h ω ( k ; λ ) is defined in Lemma 25.

Proof. Setting π = π ∗ in Lemma 13 we get that for any k ,

<!-- formula-not-decoded -->

Furthermore, by the third claim in Lemma 29,

<!-- formula-not-decoded -->

Combining the two relations and taking expectation on both sides we conclude the proof.

We are ready to prove the convergence rates for the unregularized and regularized algorithms, much like the equivalent proofs in the case of Uniform TRPO in Appendix C.4.

## D.4 Convergence proof of Exact TRPO

Before proving the theorem, we establish that the policy improves in k for the chosen learning rates.

Lemma 15 (Exact TRPO Policy Improvement) . Let { π k } k ≥ 0 be the sequence generated by Exact TRPO. Then, for both the euclidean and non-euclidean versions of the algorithm, for any λ ≥ 0 , the value improves for all k , and, thus, µv π k λ ≥ µv π k +1 λ

<!-- formula-not-decoded -->

Proof. By (42), for any state s for which d ν,π k ( s ) &gt; 0 , and for any policy π ∈ ∆ A at state s ,

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

where the first relation is by the derivative of the Bregman distance and the second is by the three-point lemma (30).

By choosing π = π k ( · | s ) ,

<!-- formula-not-decoded -->

where we used the fact that B ω ( s, π k ) π k = 0 .

Now, Using equation (43) (see Lemma 24), we get

<!-- formula-not-decoded -->

The choice of the learning rate and the fact that the Bregman distance is non negative ( λ &gt; 0 , λt k = 1 k +2 ≤ 1 and for λ = 0 the RHS of (44) is positive), implies that for all s ∈ { s ′ : d ν,π k ( s ′ ) &gt; 0 } .

<!-- formula-not-decoded -->

For all states s ∈ S for which d ν,π k ( s ) = 0 , as we do not update the policy in these states we have that π k +1 ( · | s ) = π k ( · | s ) . Thus, for all s ∈ { s ′ : d ν,π k ( s ′ ) = 0 } ,

<!-- formula-not-decoded -->

Combining (45), (46) we get that for all s ∈ S ,

<!-- formula-not-decoded -->

Applying iteratively T π k +1 λ and using its monotonicty we obtain,

<!-- formula-not-decoded -->

where in the last relation we used the fact T π k +1 λ is a contraction operator and its fixed point is v π k +1 λ .

Finally we conclude the proof by multiplying both sides with µ which gives µv π k +1 λ ≤ µv π k λ

The following theorem establish the convergence rates of the Exact TRPO algorithms.

Theorem 16 (Convergence Rate: Exact TRPO) . Let { π k } k ≥ 0 be the sequence generated by Exact TRPO Then, the following holds for all N ≥ 1 .

1. (Unregularized) Let λ = 0 , t k = (1 -γ ) C ω, 1 C max √ k +1 then

<!-- formula-not-decoded -->

2. (Regularized) Let λ &gt; 0 , t k = 1 λ ( k +2) then

<!-- formula-not-decoded -->

Where C ω, 1 = √ A,C ω, 3 = 1 for the euclidean case, and C ω, 1 = 1 , C ω, 3 = log A for the non-euclidean case.

## The Unregularized case

Proof. Applying Lemma 14 and λ = 0 (the unregularized case),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Summing the above inequality over k = 0 , 1 , ..., N , gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the second relation we used B ω ( π ∗ , π N +1 ) ≥ 0 and thus d µ,π ∗ B ω ( π ∗ , π N +1 ) ≥ 0 , and in the third relation Lemma 28.

By the improvement lemma (Lemma 15),

<!-- formula-not-decoded -->

and by some algebraic manipulations, we get

<!-- formula-not-decoded -->

Plugging in the stepsizes t k = 1 h ω √ k , we get,

<!-- formula-not-decoded -->

Bounding the sums using (Beck 2017, Lemma 8.27(a)) yields,

<!-- formula-not-decoded -->

Plugging the expressions for h ω and D ω in Lemma 25 and Lemma 28, we get for the euclidean case,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and for the non-euclidean case,

## The Regularized case

Proof. Applying Lemma 14 and setting t k = 1 λ ( k +2) , we get,

<!-- formula-not-decoded -->

where in the second relation we used that fact h ω ( k ; λ ) is a non-decreasing function of k for both the euclidean and noneuclidean cases.

Next, multiplying both sides by λ ( k +2) , summing both sides from k = 0 to N and using the linearity of expectation, we get,

<!-- formula-not-decoded -->

where the second relation holds by the positivity of the Bregman distance, and the third relation by Lemma 28 for uniformly initialized π 0 .

Bounding ∑ N k =0 1 k +2 ≤ O (log N ) , we get

<!-- formula-not-decoded -->

Since N ( µv π N λ -µv ∗ ) ≤ N ∑ k =0 µv π k -µv ∗ by Lemma 15 and some algebraic manipulations, we obtain

<!-- formula-not-decoded -->

By Plugging the bounds D ω , h ω and C max ,λ , we get in the euclidean case,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and in the non-euclidean case,

## E Sample-Based Trust Region Policy Optimization

Sample-Based TRPO is a sample-based version of Exact TRPO which was analyzed in previous section (see Appendix D). Unlike Uniform TRPO (see Appendix C) which accesses the entire state and computes v π ∈ R S in each iteration, SampleBased TRPO requires solely the ability to sample from an MDP using a ν -restart model. Similarly to (Kakade and others 2003) it requires Assumption 1 to be satisfied. Thus, Sample-Based TRPO operates under much more realistic assumptions, and, more importantly, puts formal ground to first-order gradient based methods such as NE-TRPO (Schulman et al. 2015), which was so far considered a heuristic method motivated by CPI (Kakade and Langford 2002).

In this section we prove Sample-Based TRPO (Section 6.2, Theorem 5) converges to an approximately optimal solution with high probability. The analysis in this section relies heavily on the analysis of Exact TRPO in Appendix D. We now describe the content of each of the subsections: First, in Appendix E.1, we show the connections between Sample-Based TRPO (using unbiased estimation) and Exact TRPO by proving Proposition 4. In Appendix E.2, we analyze the Sample-Based TRPO update rule and formalize the truncated sampling process. In Appendix E.3, we give a detailed proof sketch of the convergence theorem for Sample-Based TRPO, in order to ease readability. Then, we derive a fundamental inequality that will be used to prove the convergence of both unregularized and regularized versions (Appendix E.4). This inequality is almost identical to the fundamental inequality derived for Exact TRPO (Lemma 14), but with an additional term which arises due to the approximation error. In Appendix E.5, we analyze the sample complexity needed to bound this approximation error. We go on to prove the convergence rates of Sample-Based TRPO for both the unregularized and regularized version (Appendix E.6). Finally, in Appendix E.7, we calculate the overall sample complexity of both the unregularized and regularized Sample-Based TRPO and compare it to CPI.

## E.1 Relation Between Exact and Sample-Based TRPO

Before diving into the proof of Sample-Based TRPO, we prove Proposition 4, which connects the update rules of Exact TRPO and Sample-Based TRPO (in case of an unbiased estimator for q π k λ ):

Proposition 4 (Exact to Sample-Based Updates) . Let F k be the σ -field containing all events until the end of the k -1 episode. Then, for any π, π k ∈ ∆ S A and every sample m ,

<!-- formula-not-decoded -->

Proof. For any m = 1 , ..., M , we take expectation over the sampling process given the filtration F k , i.e., s m ∼ d ν,π k , a m ∼ U ( A ) , ˆ q π k λ ∼ q π k λ (we assume here an unbiased estimation process where we do not truncate the sample trajectories),

<!-- formula-not-decoded -->

/negationslash where first transition is by the definition of ˆ ∇ νv π k λ [ m ] , the second by the smoothing theorem, the third transition is due to the linearity of expectation and the fourth transition is by taking the expectation and due to the fact that /BD { a = a m } is zero for any a = a m .

<!-- formula-not-decoded -->

where the second transition is by taking the expectation over a m , the third transition is by the linearity of the inner product and due to the fact that 〈∇ ω ( s m ; π k ) , π ( · | s m ) -π k ( · | s m ) 〉 and B ω ( s m ; π, π k ) are independent of a m .

Now, taking the expectation over s m ∼ d ν,π k , where the second transition is by taking the expectation w.r.t. to s m , the the fourth is by using the lemma 24 which connects the bellman operator and the q -functions, and the last transition is due to (10) in Proposition 1, which concludes the proof.

<!-- formula-not-decoded -->

## E.2 Sample-Based TRPO Update Rule

In each step, we solve the following optimization problem (15):

<!-- formula-not-decoded -->

where s m ∼ d ν,π k ( · ) , a m ∼ U ( A ) , and ˆ q π k λ ( s m , a m , m ) is the truncated Monte Carlo estimator of q π k λ ( s m , a m ) in the m -th trajectory. The notation ˆ q π k λ ( s m , · , m ) /BD {· = a m } is a vector with the estimator value at the index a m , and zero elsewhere. Also, we remind the reader we use the notation A := |A| . We can obtain a sample s m ∼ d ν,π k ( · ) by a similar process as described in (Kakade and Langford 2002; Kakade and others 2003). Draw a start state s from the ν -restart distribution. Then, s m = s is chosen w.p. γ . Otherwise, w.p. 1 -γ , an action is sampled according to a ∼ π k ( s ) to receive the next state s . This process is

repeated until s m is chosen. If the time T = 1 1 -γ log /epsilon1 8 r ω ( k,λ ) is reached, we accept the current state as s m . Note that r ω ( k, λ ) is defined in Lemma 20, and /epsilon1 is the required final error. Finally, when s m is chosen, an action a m is drawn from the uniform distribution, and then the trajectory is unrolled using the current policy π k for T = 1 1 -γ log /epsilon1 8 r ω ( k,λ ) time-steps, to calculate ˆ q π k λ ( s m , a m , m ) . Note that this introduces a bias into the estimation of q π k λ (Kakade and others 2003)[Sections 2.3.3 and 7.3.4]. Lastly, note that the A factor in the estimator is due to importance sampling.

First, the update rule of Sample-Based TRPO can be written as a state-wise update rule for any s ∈ S . Observe that,

The first relation is the definition of the update rule (15) without the constant factor 1 M . See that multiplying the optimization problem by the constant M does not change the minimizer. In the second relation we used the fact that summation on ∑ s { s = s m } leaves the optimization problem unchanged (as the indicator function is 0 for all states that are not s m ).

<!-- formula-not-decoded -->

/BD Thus, using this update rule we can solve the optimization problem individually per s ∈ S ,

Note that using this representation optimization problem, the solution for states which were not encountered in the k -th iteration, s / ∈ S k M , is arbitrary. To be consistent, we always choose to keep the current policy, π k +1 ( · | s ) = π k ( · | s ) .

<!-- formula-not-decoded -->

Now, similarly to Uniform and Exact TRPO, the update rule of Sample-Based TRPO can be written such that the optimization problem is solved individually per visited state s ∈ S k M . This results in the final update rule used in Algorithm 4.

To prove this, let n ( s ) = ∑ a n ( s, a ) be the number of times the state s was observed at the k -th episode. Using this notation and (48), the update rule has the following equivalent forms,

<!-- formula-not-decoded -->

In the third relation we used the fact for any π, π k

<!-- formula-not-decoded -->

The fourth relation holds as the optimization problem is not affected by s / ∈ S k M , and the last relation holds by dividing by n ( s ) &gt; 0 as s ∈ S k M and using linearity of inner product.

Lastly, we observe that (50) is a sum of functions of π ( · | s ) , i.e.,

  where f = 〈 g s , π ( · | s ) 〉 + 1 t k B ω ( s ; π, π k ) , g s ∈ R A is the vector inside the inner product of (50). Meaning, the minimization problem is a sum of independent summands. Thus, in order to minimize the function on ∆ S A it is enough to minimize independently each one of the summands. From this observation, we conclude that the update rule (15) is equivalent to update the policy for all s ∈ S k M by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, by plugging in ˆ q π k λ ( s, a ) = 1 n ( s ) ∑ n ( s,a ) i =1 ˆ q π k λ ( s, a, m i ) , we get where m i is the trajectory index of the i -th occurrence of the state s .

## E.3 Proof Sketch of Theorem 5

In order to keep things organized for an easy reading, we first go through the proof sketch in high level, which serves as map for reading the proof of Theorem 5 in the following sections.

1. We use the Sample-Based TRPO optimization problem described in E.2, to derive a fundamental inequality in Lemma 19 for the sample-based case (in Appendix E.4):
2. (a) We derive a state-wise inequality by applying similar analysis to Exact TRPO, but for the Sample-Based TRPO optimization problem. By adding and subtracting a term which is similar to (42) in the state-wise inequality of Exact TRPO (Lemma 12), we write this inequality as a sum between the expected error and an approximation error term.
3. (b) For each state, we employ importance sampling of d µ,π ∗ ( s ) d ν,π k ( s ) to relate the derived state-wise inequality, to a global guarantee w.r.t. the optimal policy π ∗ and measure µ . This importance sampling procedure is allowed by assumption 1, which states that for any s such that d µ,π ∗ ( s ) &gt; 0 it also holds that ν ( s ) &gt; 0 , and thus d ν,π k ( s ) &gt; 0 since d ν,π k ( s ) ≥ (1 -γ ) ν ( s ) .
4. (c) By summing over all states we get the required fundamental inequality which resembles the fundamental inequality of Exact TRPO with an additional term due to the approximation error.
2. In Appendix E.5, we show that the approximation error term is made of two sources of errors: (a) a sampling error due to the finite number of trajectories in each iteration; (b) a truncation error due to the finite length of each trajectory, even in the infinite-horizon case.
6. (a) In Lemma 20 we deal with the sampling error. We show that this error is caused by the difference between an empirical mean of i.i.d. random variables and their expected value. Using Lemma 26 and Lemma 27, we show that these random variables are bounded, and also that they are proportional to the step size t k . Then, similarly to (Kakade and others 2003), we use Hoeffding's inequality and the union bound over the policy space (in our case, the space of deterministic policies), in order to bound this error term uniformly. This enables us to find the number of trajectories needed in the k -th iteration to reach an error proportional to C π ∗ t k /epsilon1 = ∥ ∥ ∥ d µ,π ∗ ν ∥ ∥ ∥ ∞ t k /epsilon1 with high probability. The common concentration efficient C π ∗ , arises due to d µ,π ∗ ( s ) d ν,π k ( s ) , the importance sampling ratio used for the global convergence guarantee.

Finally, in Lemma 23, we use the union bound over all k ∈ N in order to uniformly bound the error propagation over N iterations of Sample-Based TRPO.

- (b) In Lemma 21 we deal with the truncation error. We show that we can bound this error to be proportional to C π ∗ t k /epsilon1 , by using O ( 1 1 -γ ) samples in each trajectory.
3. In Appendix E.6 we use a similar analysis to the one used for the rates guarantees of Exact TRPO (Appendix D.4), using the above results. The only difference is the approximation term which we bound in E.5. There, we make use of the fact that the approximation term is proportional to the step size t k and thus decreasing with the number of iterations, to prove a bounded approximation error for any N .

4. Lastly, in Appendix E.7, we calculate the overall sample complexity - previously we bounded the number of needed iterations and the number of samples needed in every iteartion - for each of the four cases of Sample-Based TRPO (euclidean vs. noneuclidean, unregularized vs. regularized).

## E.4 Fundamental Inequality of Sample-Based TRPO

Lemma 17 (sample-based state-wise inequality) . Let { π k } k ≥ 0 be the sequence generated by Aproximate TRPO using stepsizes { t k } k ≥ 0 . Then, for all states s for which d ν,π k ( s ) &gt; 0 the following inequality holds for all π ∈ ∆ S A ,

<!-- formula-not-decoded -->

where h ω is defined at the third claim of Lemma 26.

Proof. Using the first order optimality condition for the update rule (49), the following holds for any s ∈ S and thus for any s ∈ { s ′ : d ν,π k ( s ) &gt; 0 } ,

Dividing by d ν,π k ( s ) which is strictly positive for all s such that { s = s m } = 1 and adding and subtracting the term

<!-- formula-not-decoded -->

we get

<!-- formula-not-decoded -->

where we defined /epsilon1 k ( s, π ) ,

/epsilon1 k ( s, π )

<!-- formula-not-decoded -->

By bounding ( ∗ ) in (52) using the exact same analysis of Lemma 12 we conclude the proof.

Now, we state another lemma which connects the state-wise inequality using the discounted stationary distribution of the optimal policy d µ,π ∗ , similarly to Lemma 13.

Lemma 18. Let Assumption 1 hold and let { π k } k ≥ 0 be the sequence generated by Aproximate TRPO using stepsizes { t k } k ≥ 0 . Then, for all k ≥ 0 Then, the following inequality holds for all π ,

<!-- formula-not-decoded -->

where h ω is defined in the third claim of Lemma 26.

Proof. By Assumption 1, for all s for which d µ,π ∗ ( s ) &gt; 0 it also holds that d ν,π k ( s ) &gt; 0 . Thus, for all s for which d µ,π ∗ ( s ) &gt; 0 the component-wise relation in Lemma 17 holds. By multiplying each inequality by the positive number d µ,π ∗ ( s ) and summing over all s we get,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 19 (fundamental inequality of Sample-Based TRPO.) . Let { π k } k ≥ 0 be the sequence generated by Aproximate TRPO using stepsizes { t k } k ≥ 0 . Then, for all k ≥ 0

<!-- formula-not-decoded -->

where h ω ( k ; λ ) is defined in Lemma 26 and /epsilon1 k := /epsilon1 k ( · , π ∗ ) where the latter defined in (53) .

Proof. Setting π = π ∗ in Lemma 18 and denoting /epsilon1 k := /epsilon1 k ( · , π ∗ ) , we get that for any k ,

<!-- formula-not-decoded -->

Furthermore, by the third claim of Lemma 29,

<!-- formula-not-decoded -->

Combining the two relations on both sides we concludes the proof.

## E.5 Approximation Error Bound

In this section we deal with the approximation error, the term d µ,π ∗ /epsilon1 k in Lemma 19. Two factors effects d µ,π ∗ /epsilon1 k : (1) the error due to Monte-Carlo sampling, which we bound using Hoeffding's inequality and the union bound; (2) the error due to the truncation in the sampling process (see Appendix E.2). The next two lemmas bound these two sources of error. We first discuss the analysis of using an unbiased sampling process (Lemma 20), i.e., when no truncation is taking place, and then move to discuss the use of the truncated trajectories (Lemma 21). Finally, in Lemma 22 we combine the two results to bound d µ,π ∗ /epsilon1 k in the case of the full truncated sampling process discussed in Appendix E.2.

The unbiased q -function estimator uses a full unrolling of a trajectory, i.e., calculates the (possibly infinite) sum of retrieved costs following the policy π k in the m -th trajecotry of the k -th iteration,

<!-- formula-not-decoded -->

/negationslash where the notation s k,m t refer to the state encountered in the m -th trajectory of the k -th iteration, at the t step of estimating the q π k λ function. Moreover, ( s m , a m ) = ( s k,m 0 , a k,m 0 ) and ˆ q π k λ ( s, a, m ) = 0 for any ( s, a ) = ( s m , a m ) .

The truncated biased q -function estimator, truncates the trajectory after T interactions with the MDP, where T is predefined:

<!-- formula-not-decoded -->

The following lemma describes the number of trajectories needed in the k -th update, in order to bound the error to be proportional to /epsilon1 w.p. 1 -δ ′ , using an unbiased estimator.

<!-- formula-not-decoded -->

Lemma 20 (Approximation error bound with unbiased sampling) . For any /epsilon1, ˜ δ &gt; 0 , if the number of trajectories in the k -th iteration is

<!-- formula-not-decoded -->

then with probability of 1 -˜ δ ,

∥ ∥ where r ω ( k, λ ) = 4 A C max ,λ 1 -γ and r ω ( k, λ ) = 4 A C max ,λ 1 -γ (1 + /BD { λ = 0 } log k ) in the euclidean and non-euclidean settings respectively.

<!-- formula-not-decoded -->

/negationslash

Proof. Plugging the definition of /epsilon1 k := /epsilon1 k ( · , π ∗ ) in (53), we get, d /epsilon1

<!-- formula-not-decoded -->

/negationslash where in the last transition we used the fact that for every s = s m the identity function /BD { s = s m } = 0 . We define,

<!-- formula-not-decoded -->

Using this definition, we have,

<!-- formula-not-decoded -->

In order to remove the dependency on the randomness of π k +1 , we can bound this term in a uniform way:

<!-- formula-not-decoded -->

In this lemma, we analyze the case where no truncation is taken into account. In this case we, we will now show that for any π ′

<!-- formula-not-decoded -->

which means that 1 M k ∑ M k m =1 ∑ s /BD { s = s m } d µ,π ∗ ( s ) d ν,π k ( s ) 〈 ˆ X k ( s, · , m ) , π ∗ ( · | s m ) -π ′ ( · | s m ) 〉 is an unbiased estimator. This fact comes from the from the following relations:

/negationslash

<!-- formula-not-decoded -->

where the first transition is by law of total expectation; the second transition is by the fact the indicator function is zero for every s = s m ; the third transition is by the fact s m is not random given s m ; the fourth transition is by the linearity of expectation and the fact that π ∗ ( · | s m ) -π ′ ( · | s m ) is not random given s m ; the fifth transition is by taking the expectation of ˆ X in the state s m ; finally, the sixth transition is by explicitly taking the expectation over the probability that s m is drawn from d ν,π k in the m -th trajectory (by following π k from the restart distribution ν ).

Meaning, (57) is a difference between an empirical mean of M k random variables and their mean for a the fixed policy π ′ , which maximizes the following expression

<!-- formula-not-decoded -->

As we wish to obtain a uniform bound on π ′ , we can use the common approach of bounding (59) uniformly for all π ′ ∈ ∆ S A using the union bound. Note that the above optimization problem is a linear programming optimization problem in π ′ , where π ′ ∈ ∆ S A . It is a well known fact that for linear programming, there is an extreme point which is the optimal solution of the problem (Bertsimas and Tsitsiklis 1997)[Theorem 2.7]. The set of extreme points of ∆ S A is the set of all deterministic policies denoted by Π det . Therefore, in order to bound the maximum in (59), it suffices to uniformly bound all policies π ′ ∈ Π det .

Now, notice that

∥ ∥ where the second transition is due to H¨ older's inequality; the third transition is due to the bound of the TV distance between two random variables; the sixth transition is due to the triangle inequality; finally, the seventh transition is by plugging in the bounds in Lemma 26 and Lemma 27. Also, we defined r ω ( k, λ ) = 4 A C max ,λ 1 -γ and r ω ( k, λ ) = 4 A C max ,λ 1 -γ (1 + /BD { λ = 0 } log k ) in the euclidean and non-euclidean cases respectively.

Thus, by Hoeffding and the union bound over the set of deterministic policies,

<!-- formula-not-decoded -->

In other words, in order to guarantee that we need the number of trajectories M k to be at least

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used the fact that there are | Π det | = A S deterministic policies. which concludes the result.

The following lemma described with error due to the use of truncated trajectories:

Lemma 21 (Truncation error bound) . The bias of the truncated sampling process in the k -th iteration, with maximal trajectory length of T = 1 1 -γ log /epsilon1 8 r ω ( k,λ ) is t k ∥ ∥ ∥ d µ,π ∗ d ν,π k ∥ ∥ ∥ ∞ /epsilon1 4 , where r ω ( k, λ ) = 4 A C max ,λ 1 -γ and r ω ( k, λ ) = 2 A C max ,λ 1 -γ ( 1 1 -λt k +1+ λ log k ) in the euclidean and non-euclidean settings respectively.

<!-- formula-not-decoded -->

/negationslash

Proof. We start this proof by defining notation related to the truncated sampling process. First, denote d trunc ν,π k ( s ) , the probability to choose a state s , using the truncated biased sampling process of length T , as described in Appendix E.2. Observe that

<!-- formula-not-decoded -->

We also make use in this proof in the following definitions (see (54) and (55)),

<!-- formula-not-decoded -->

Lastly, we denote the expectation of ˆ X k ( s, · , m ) using the truncated sampling process as X trunc k ( s, · ) ,

<!-- formula-not-decoded -->

Now, we move on to the proof. We first split the bias to two different sources of bias:

<!-- formula-not-decoded -->

The first source of bias is due to the truncation of the state sampling after T iterations, and the second source of bias is due to the truncation done in the estimation of q π k λ ( s, a ) , for the chosen state s and action a .

First, we bound the first error term. Observe that for any s ,

<!-- formula-not-decoded -->

where the third transition is due to the triangle inequality, the fourth transition is due to the fact that for any t , γ t p ( s t | ν, π k ) ≥ 0 and the sixth transition is by the fact that ∑ s p ( s t = s | ν, π k ) ≤ 1 for any t as a probability distribution.

Thus,

<!-- formula-not-decoded -->

where the fourth transition is by plugging in (61) and the last transition is by repeating similar analysis to (60).

Now, by simple arithmetic, for any /epsilon1 &gt; 0 , if the trajectory length T &gt; 1 1 -γ log /epsilon1 16 r ω ( k,λ ) , we get that the first bias term is bounded,

<!-- formula-not-decoded -->

Next, we bound the second error term.

First, observe that for any s, a ,

<!-- formula-not-decoded -->

Now,

<!-- formula-not-decoded -->

∥ ∥ where the first transition is due to the linearity of expectation, the third transition is by the fact the summation of d ν,π k is convex, the fourth transition is by the fact d µ,π ∗ ( s ) d ν,π k ( s ) is non-negative for any s and by maximizing each term separately, the fifth transition is by using the definitions of X k and X trunc k , the sixth is using H¨ older's inequality and the last transition is due to (63).

Now, using the same T , by the fact r ω ( k, λ ) &gt; 2 C max ,λ 1 -γ , we have that

<!-- formula-not-decoded -->

Finally, combining (62) and (64) concludes the results.

In the next lemma we combine the results of Lemmas 20 and 21 to bound the overall approximation error due to both sampling and truncation.

Lemma 22 (Approximation error bound using truncated biased sampling) . For any /epsilon1, ˜ δ &gt; 0 , if the number of trajectories in the k -th iteration is and the number of samples in the truncated sampling process is of length

then with probability of 1 -˜ δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the overall number of interaction with the MDP is in the k -th iteration is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where r ω ( k, λ ) = 4 A C max ,λ 1 -γ and r ω ( k, λ ) = 2 A C max ,λ 1 -γ ( 1 1 -λt k +1+ λ log k ) in the euclidean and non-euclidean settings respectively.

Proof. Repeating the same steps of Lemma 20, we re-derive equation (57),

<!-- formula-not-decoded -->

Now, we move on to deal with a truncated trajectory: In Appendix E.2 we defined a nearly unbiased estimation process for q π k λ , i.e., 1 M k ∑ M k m =1 ∑ s /BD { s = s m } d µ,π ∗ ( s ) d ν,π k ( s ) 〈 ˆ X k ( s, · , m ) , π ∗ ( · | s m ) -π ′ ( · | s m ) 〉 is no longer an unbiased estimator as in Lemma 20. In what follows we divide the error to two sources of error, one due to the finite sampling error (finite number of trajectories) and the other due to the bias admitted by the truncation.

For any π ′ , denote the following variables,

<!-- formula-not-decoded -->

By plugging this new notation in (57), we can write,

<!-- formula-not-decoded -->

where the first inequality is by plugging in the definition of Y ( π ′ ) , ˆ Y M ( π ′ ) in (57) and the last transition is by maximizing each of the terms in the sum independently. Note that (1) describes the error due to the finite sampling and (2) describes the error due to the truncation of the trajectories. Importantly, notice that in the case where we do not truncate the trajectory, the second term (2) equals zero by (58). We will now use Lemma 20 and Lemma 21 to bound (1) and (2) respectively:

First, look at the first term (1). By definition it an unbiased estimation process. Furthermore, by equation (60), ˆ Y m ( π ′ ) is bounded for all s m and π ′ by

<!-- formula-not-decoded -->

Thus by applying Lemma 20 we get that in order to guarantee that we need the number of trajectories M k to be at least

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we bound the second term (2). By Lemma 21, using a trajectory of maximal length 1 1 -γ log /epsilon1 8 r ω ( k,λ ) , the errors due to the truncated estimation process are bounded as follows,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Bounding the two terms by (66) and (67), and plugging them back in (65), we get that using M k trajectories, where each trajectory is of length O ( 1 1 -γ log /epsilon1 ) , we have that w.p. 1 -˜ δ

which concludes the result.

So far, we proved the number of samples needed for a bounded error with high probability in the k -th iteration of Sample-Based TRPO. The following Lemma gives a bound for the accumulative error of Sample-Based TRPO after k iterations.

Lemma 23 (Cumulative approximation error) . For any /epsilon1, δ &gt; 0 , if the number of trajectories in the k -th iteration is and the number of samples in the truncated sampling process is of length

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then, with probability greater than 1 -δ , uniformly on all k ∈ N ,

/negationslash

∥ ∥ where r ω ( k, λ ) = 4 A C max ,λ 1 -γ and r ω ( k, λ ) = 4 A C max ,λ 1 -γ (1 + /BD { λ = 0 } log k ) in the euclidean and non-euclidean settings respectively.

Proof. Using Lemma 22 with ˜ δ = 6 π 2 δ ( k +1) 2 and the union bound over all k ∈ N , we get that w.p. bigger than for any k , the following inequality holds

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, by summing the inequalities for k = 0 , 1 , ..., N , we obtain

∥ ∥ where we used the solution to Basel's problem (the sum of reciprocals of the squares of the natural numbers) for calculating ∑ ∞ k =0 1 ( k +1) 2 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, Using the fact that ∥ ∥ ∥ d µ,π ∗ d ν,π k ∥ ∥ ∥ ∞ ≤ 1 1 -γ ∥ ∥ d µ,π ∗ ν ∥ ∥ ∞ , we have that w.p. of at least δ ,

Lastly, by bounding π 2 / 6 ≤ 2 we conclude the proof.

We are ready to prove the convergence rates for the unregularized and regularized algorithms, similarly to the proofs of Exact TRPO (see Appendix D.4).

## E.6 Proof of Theorem 5

For the sake of completeness and readability, we restate here Theorem 5, this time including all logarithmic factors, but excluding higher orders in λ (All constants are in the proof):

Theorem (Convergence Rate: Sample-Based TRPO) . Let { π k } k ≥ 0 be the sequence generated by Sample-Based TRPO, using M k ≥ r ω ( N,λ ) 2 2 /epsilon1 2 ( S log 2 A +log π 2 ( k +1) 2 / 6 δ ) trajectories in each iteration, and { µv k best } k ≥ 0 be the sequence of best achieved values, µv N best := arg min k =0 ,...,N µv π k λ -µv ∗ λ . Then, with probability greater than 1 -δ for every /epsilon1 &gt; 0 the following holds for all N ≥ 1 .

<!-- formula-not-decoded -->

2. (Regularized) Let λ &gt; 0 , t k = 1 λ ( k +2) then

<!-- formula-not-decoded -->

/negationslash

Where C ω, 1 = √ A,C ω, 2 = 1 , C ω, 3 = 1 , r ω ( k, λ ) = 4 A C max ,λ 1 -γ for the euclidean case, and C ω, 1 = 1 , C ω, 2 = A 2 , C ω, 3 = log A,r ω ( k, λ ) = 4 A C max ,λ 1 -γ (1 + { λ = 0 } log k ) for the non-euclidean case.

/BD The proof of this theorem follows the almost identical steps as the proof of Theorem 16 in Appendix D.4, but two differences: The first, is the fact we also have the additional approximation error term d µ,π ∗ /epsilon1 k . The second, is that for the sample-based case, as we don't have improvement guarantees such as in Lemma 15, we prove convergence for best policy in hindsight, which have the value µv N best := arg min k =0 ,...,N µv π k λ -µv ∗ λ .

## The Unregularized Case

Proof. Applying Lemma 19 and λ = 0 (the unregularized case),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Summing the above inequality over k = 0 , 1 , ..., N , gives

<!-- formula-not-decoded -->

where in the second relation we used B ω ( π ∗ , π N +1 ) ≥ 0 and thus d µ,π ∗ B ω ( π ∗ , π N +1 ) ≥ 0 , and in the third relation Lemma 28.

Using the definition of v N best , we have that

<!-- formula-not-decoded -->

and by some algebraic manipulations, we get

<!-- formula-not-decoded -->

Plugging in the stepsizes t k = 1 h ω √ k , we get,

<!-- formula-not-decoded -->

Bounding the sums using (Beck 2017, Lemma 8.27(a)) yields,

<!-- formula-not-decoded -->

Plugging in Lemma 23, we get that for any ( /epsilon1, δ ) , if the number of trajectories in the k -th iteration is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then, with probability greater than 1 -δ , where r ω ( k, λ ) = 4 A C max ,λ 1 -γ and r ω ( k, λ ) = 4 A C max ,λ 1 -γ (1 + /BD { λ = 0 } log k ) in the euclidean and non-euclidean settings respectively.

By rearranging, we get,

<!-- formula-not-decoded -->

/negationslash

Thus, for the euclidean case, and for the non-euclidean case,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## The Regularized Case

Proof. Applying Lemma 19 and setting t k = 1 λ ( k +2) , we get,

<!-- formula-not-decoded -->

where in the second relation we used that fact h ω ( k ; λ ) is a non-decreasing function of k for both the euclidean and noneuclidean cases.

Next, multiplying both sides by λ ( k +2) , summing both sides from k = 0 to N and using the linearity of expectation, we get,

<!-- formula-not-decoded -->

where the second relation holds by the positivity of the Bregman distance, the third relation by Lemma 28 for uniformly initialized π 0 , and the last relation by plugging back t k = 1 λ ( k +2) in the last term..

Bounding ∑ N k =0 1 k +2 ≤ O (log N ) , we get

<!-- formula-not-decoded -->

By the definition of v N best , which gives ( N +1) ( µv N best -µv ∗ ) ≤ N ∑ k =0 µv π k -µv ∗ , and some algebraic manipulations, we obtain

<!-- formula-not-decoded -->

Plugging in Lemma 22, we get that for any ( /epsilon1, δ ) , if the number of trajectories in the k -th iteration is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash

By Plugging the bounds D ω , h ω and C max ,λ , we get in the euclidean case, then with probability of at least 1 -δ ,

∥ ∥ where r ω ( k, λ ) = 4 A C max ,λ 1 -γ and r ω ( k, λ ) = 4 A C max ,λ 1 -γ (1 + /BD { λ = 0 } log k ) in the euclidean and non-euclidean settings respectively.

and in the non-euclidean case,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E.7 Sample Complexity of Sample-Based TRPO

In this section we calculate the overall sample complexity of Sample-Based TRPO, i.e., the number interactions with the MDP the algorithm does in order to reach a close to optimal solution.

/negationslash

Therefore, the number of samples in each iteration required to guarantee a 1 (1 -γ ) 2 ∥ ∥ d µ,π ∗ ν ∥ ∥ ∞ /epsilon1 2 error is

By Lemma 23, in order to have 1 (1 -γ ) 2 ∥ ∥ ∥ d µ,π ∗ ν ∥ ∥ ∥ ∞ /epsilon1 2 approximation error, we need M k ≥ O ( r ω ( k,λ ) 2 /epsilon1 2 ( S log 2 A +log( k +1) 2 /δ ) ) trajectories in each iteration, and the number of samples in each truncated trajectory is T k ≥ O ( 1 1 -γ log /epsilon1 r ω ( k,λ ) ) , where r ω ( k, λ ) = 4 A C max ,λ 1 -γ (1 + /BD { λ = 0 } log k ) in the euclidean and non-euclidean settings respectively.

<!-- formula-not-decoded -->

The overall sample complexity is acquired by multiplying the number of iterations N required to reach an /epsilon1/ 2 (1 -γ ) 2 optimization error multiplied with the iteration-wise sample complexity, given above. Combining the two errors and using the fact that C π ∗ ≥ 1 , we have that the overall error

In other words, the overall error of the algorithm is bounded by 1 (1 -γ ) 2 C π ∗ /epsilon1

<!-- formula-not-decoded -->

| Euclidean                                                     |                                                               | Non-Euclidean (KL)   |
|---------------------------------------------------------------|---------------------------------------------------------------|----------------------|
| A 3 C 4 max (1 - γ ) 3 /epsilon1 4 ( log | Π det | +log 1 δ ) | A 2 C 4 max (1 - γ ) 3 /epsilon1 4 ( log | Π det | +log 1 δ ) | Unregularized        |
| A 3 C 4 max ,λ λ (1 γ ) 4 /epsilon1 3 log | Π det | +log 1 δ  | A 4 C 4 max ,λ λ (1 γ ) 4 /epsilon1 3 log | Π det | +log 1 δ  | Regularized          |

-

(

)

-

(

)

Finally, the sample complexity to reach a 1 (1 -γ ) 2 C π ∗ /epsilon1 error for the different cases is arranged in the following table (the complete analysis is provided the the next section):

The same bound for CPI as given in (Kakade and others 2003) is

<!-- formula-not-decoded -->

where we omitted logarithmic factors in 1 -γ and /epsilon1 . Notice that this bound is similar to the bound of Sample-Based TRPO observed in this paper, as expected.

In order to translate this bound using our notation bound, we used (Kakade and others 2003)[Theorem 7.3.3] with H = 1 1 -γ , which states that in order to guarantee a bounded advantage of for any policy π ′ , A π ( ν, π ′ ) ≤ (1 -γ ) /epsilon1 we need O ( log /epsilon1 (log Π det +log 1 δ (1 -γ ) 5 /epsilon1 4 ) samples. Then, by (Kakade and Langford 2002)[Corollary 4.5] with A π ( ν, π ′ ) ≤ (1 -γ ) /epsilon1 we get that (1 -γ )( µv π -µv ∗ ) ≤ /epsilon1 1 -γ ∥ ∥ ∥ d µ,π ∗ ν ∥ ∥ ∥ ∞ , or µv π -µv ∗ ≤ /epsilon1 (1 -γ ) 2 ∥ ∥ ∥ d µ,π ∗ ν ∥ ∥ ∥ ∞ . Finally, the C 4 max factor comes from using a nonnormalized MDP, where the maximum reward is C max. We get C 2 max from number of iterations needed for convergence, and the number of samples in each iteration is also proportional to C 2 max

## The Unregularized Case

The euclidean case: The error after N iterations is bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, in order to reach an error of 1 (1 -γ ) 2 C π ∗ /epsilon1 error, we need

Thus, the sample complexity to reach 1 (1 -γ ) 2 C π ∗ /epsilon1 error when logarithmic factors are omitted is

<!-- formula-not-decoded -->

The non-euclidean case: The error after N iterations is bounded by

<!-- formula-not-decoded -->

Thus, in order to reach an error of 1 (1 -γ ) 2 C π ∗ /epsilon1 error, we need

<!-- formula-not-decoded -->

Thus, the sample complexity to reach 1 (1 -γ ) 2 C π ∗ /epsilon1 error when logarithmic factors are omitted is

<!-- formula-not-decoded -->

## The Regularized Case

The euclidean case: The error after N iterations is bounded by

<!-- formula-not-decoded -->

Thus, in order to reach an error of 1 (1 -γ ) 2 C π ∗ /epsilon1 error, we need

<!-- formula-not-decoded -->

Thus, the sample complexity to reach 1 (1 -γ ) 2 C π ∗ /epsilon1 error when logarithmic factors are omitted is

<!-- formula-not-decoded -->

The non-euclidean case: The error after N iterations is bounded by

Rearranging, we get,

<!-- formula-not-decoded -->

which can also be written with C 2 max ,λ

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, in order to reach an error of 1 (1 -γ ) 2 C π ∗ /epsilon1 error, we need

<!-- formula-not-decoded -->

omitting logarithmic factors.

Thus, the sample complexity to reach 1 (1 -γ ) 2 C π ∗ /epsilon1 error when logarithmic factors are omitted is

<!-- formula-not-decoded -->

Proof. First, note that for any s

<!-- formula-not-decoded -->

where the second transition is by the definition of q π λ , the third is by the definition of c π λ , the fourth is by adding and subtracting λω ( s ; π ′ ) , the fifth is by the fact λω ( s ; π ′ ) is independent of a and the seventh is by the definition of the regularized Bellman operator.

Thus,

<!-- formula-not-decoded -->

Now, note that by the definition of the q -function 〈 q π λ , π 〉 = v π λ and thus,

<!-- formula-not-decoded -->

Finally, by adding to both sides 〈 λ ∇ ω ( π ) , π ′ -π 〉 , we get,

<!-- formula-not-decoded -->

To conclude the proof, note that by the definition of the Bregman distance we have,

<!-- formula-not-decoded -->

Lemma 25 (Bounds regarding the updates of Uniform and Exact TRPO) . For any k ≥ 0 and state s , which is updated in the k -th iteration, the following relations hold for both Uniform TRPO (40) and Exact TRPO (21) :

1. ‖∇ ω ( π k ( ·| s )) ‖ ∗ ≤ O (1) and ‖∇ ω ( π k ( ·| s )) ‖ ∗ ≤ O ( C max ,λ log k λ (1 -γ ) ) , in the euclidean and non-euclidean cases, respectively.
2. ‖ q π k λ ( s, · ) ‖ ∗ ≤ h ω , where h ω = O ( √ A C max ,λ 1 -γ ) and h ω = O ( C max ,λ 1 -γ ) in the euclidean and non-euclidean cases, respectively.

## F Useful Lemmas

The next lemmas will provide useful bounds for uniform, exact and Sample-Based TRPO. In this section, we define ‖·‖ ∗ to be the dual norm of ‖·‖ .

Lemma 24 (Connection between the regularized Bellman operator and the q -function) . For any π, π ′ the following holds:

<!-- formula-not-decoded -->

/negationslash

/negationslash

Where for every state s , ‖·‖ ∗ denotes the dual norm over the action space, which is L 1 in the euclidean case, and L ∞ in non-euclidean cases.

3. ‖ q π k λ ( s, · ) + λ ∇ ω ( π k ( ·| s )) ‖ ∗ ≤ h ω ( k ; λ ) , where h ω ( k ; λ ) = O ( √ A C max ,λ 1 -γ ) and h ω ( k ; λ ) = O ( C max ,λ (1+ /BD { λ =0 } log k ) 1 -γ ) in the euclidean and non-euclidean cases, respectively, and /BD { λ = 0 } = 0 in the unregularized case ( λ =0) and /BD { λ = 0 } = 1 otherwise.

Proof. We start by proving the first claim :

For the euclidean case , ω ( · ) = 1 2 ‖·‖ 2 2 . Thus, for every state s ,

<!-- formula-not-decoded -->

where the inequality is due to the fact that ‖·‖ 2 ≤ ‖·‖ 1 .

The statement holds by the properties of 1 2 ‖·‖ 2 2 and thus holds for both the uniform and exact versions.

For the non-euclidean case , ω ( · ) = H ( · ) + log A . Now, consider exact TRPO (21). By taking the logarithm of (26), we have,

<!-- formula-not-decoded -->

Notice that for k ≥ 0 , for every state-action pair, q π k λ ( a | s ) ≥ 0 . Thus,

Where the first relation holds since q π λ ( s, a ) ≥ 0 . Applying Jensen's inequality we can further bound the above.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the third relation we applied Jensen's inequality for concave functions. As 0 ≤ 1 -λt k ≤ 1 (by the choice of the learning rate in the regularized case) we have that X 1 -λt k is a concave function in X , and thus ∑ A a ′ =1 1 A π 1 -λt k k ( a ′ | s ) ≤ ( ∑ A a ′ =1 1 A π k ( a ′ | s ) ) 1 -λt k by Jensen's inequality. Combining this inequality with the fact that A is positive and log is monotonic function establishes the third relation.

/negationslash

Furthermore, note that for every k , and for every s, a

Plugging (70) and (71) in (68), we get,

<!-- formula-not-decoded -->

where the second relation holds by unfolding the recursive formula for each k and the fourth by plugging in the stepsizes for the regularized case, i.e. t k = 1 λ ( k +2) . The final relation holds since C max ,λ = C max + λ log A .

To conclude, since log π k ( a | s ) ≤ 0 and ∇ ω ( π ) = ∇ H ( π ) = 1 + log π , we get that for the non-euclidean case,

<!-- formula-not-decoded -->

This concludes the proof of the first claim for both the euclidean and non-euclidean cases, in both exact scenarios. Interestingly, in the non-euclidean case, the gradients can grow to infinity due to the fact that the gradient of the entropy of a deterministic policy is unbounded. However, this result shows that a deterministic policy can only be obtained after an infinite time, as the gradient is bounded by a logarithmic rate.

Next, we prove the second claim :

It holds that for any state-action pair q π k λ ( s, a ) ∈ [ 0 , C max ,λ 1 -γ ] .

For the euclidean case , we have that

<!-- formula-not-decoded -->

For the non-euclidean case , we have that which concludes the proof of the second claim.

<!-- formula-not-decoded -->

Finally, we prove the third claim : For any state s , by the triangle inequality,

<!-- formula-not-decoded -->

by plugging the two former claims for the euclidean and non-euclidean cases, we get the required result.

<!-- formula-not-decoded -->

The next lemma follows similar derivation to Lemma 25, with small changes tailored for the sample-based case. Note that in the sample-based case, and A factor is added in claims 1,3 and 4.

Lemma 26 (Bounds regarding the updates of Sample-Based TRPO) . For any k ≥ 0 and state s , which is updated in the k -th iteration, the following relations hold for Sample-Based TRPO (51) :

1. ‖∇ ω ( π k ( ·| s )) ‖ ∗ ≤ O (1) and ‖∇ ω ( π k ( ·| s )) ‖ ∗ ≤ O ( A C max ,λ log k λ (1 -γ ) ) , in the euclidean and non-euclidean cases, respectively.
2. ‖ q π k λ ( s, · ) ‖ ∗ ≤ h ω , where h ω = O ( √ A C max ,λ 1 -γ ) and h ω = O ( C max ,λ 1 -γ ) in the euclidean and non-euclidean cases, respectively.

/negationslash

3. ‖ q π k λ ( s, · ) + λ ∇ ω ( π k ( ·| s )) ‖ ∗ ≤ h ω ( k ; λ ) , where h ω ( k ; λ ) = O ( √ A C max ,λ 1 -γ ) and h ω ( k ; λ ) = O ( C max ,λ (1+ /BD { λ =0 } A log k ) 1 -γ ) in the euclidean and non-euclidean cases, respectively, and /BD { λ = 0 } = 0 in the unregularized case ( λ =0) and /BD { λ = 0 } = 1 in the regularized case ( λ &gt; 0 ).

/negationslash

/negationslash

/negationslash

4. ‖ A ˆ q π k λ ( s, · , m ) + λ ∇ ω ( π k ( ·| s )) ‖ ∞ ≤ ˆ h ω ( k ; λ ) , where ˆ h ω ( k ; λ ) = O ( A C max ,λ 1 -γ ) and ˆ h ω ( k ; λ ) = O ( A C max ,λ (1+ /BD { λ =0 } log k ) 1 -γ ) in the euclidean and non-euclidean cases, respectively, and /BD { λ = 0 } = 0 in the unregularized case ( λ =0) and /BD { λ = 0 } = 1 in the regularized case ( λ &gt; 0 ).

/negationslash

/negationslash

Where for every state s , ‖·‖ ∗ denotes the dual norm over the action space, which is L 1 in the euclidean case, and L ∞ in non-euclidean cases.

Proof. We start by proving the first claim :

For the euclidean case , in the same manner as in the exact cases, ω ( · ) = 1 2 ‖·‖ 2 2 . Thus, for every state s ,

<!-- formula-not-decoded -->

where the inequality is due to the fact that ‖·‖ 2 ≤ ‖·‖ 1 .

For the non-euclidean case , ω ( · ) = H ( · ) + log A . The bound for the sample-based version for the non-euclidean choice of ω follows similar reasoning with mild modification. By (50), in the sample-based case, a state s is updated in the k -th iteration using the approximation of the q π k λ ( s, a ) in this state,

<!-- formula-not-decoded -->

where we denoted n ( s ) = ∑ a n ( s, a ) the number of times the state s was observed at the k -th episode and used the fact ˆ q π k λ ( s m , · , m i ) is sampled by unrolling the MDP. Thus, it holds that

<!-- formula-not-decoded -->

Interestingly, because we use the importance sampling factor A in the approximation of q π k λ , we obtain an additional A factor.

Thus, by repeating the analysis in Lemma 25, equation (72), we obtain,

<!-- formula-not-decoded -->

where the second relation holds by unfolding the recursive formula for each k and the fourth by plugging in the stepsizes for the regularized case, i.e. t k = 1 λ ( k +2) . The final relation holds since C max ,λ = C max + λ log A . Thus,

<!-- formula-not-decoded -->

This concludes the proof of the first claim for both the euclidean and non-euclidean cases.

As in the exact case, in the non-euclidean case, the gradients can grow to infinity due to the fact that the gradient of the entropy of a deterministic policy is unbounded. However, this result shows that a deterministic policy can only be obtained after an infinite time, as the gradient is bounded by a logarithmic rate.

Next, we prove the second claim :

It holds that for any state-action pair q π k λ ( s, a ) ∈ [ 0 , C max ,λ 1 -γ ] .

For the euclidean case , we have that

<!-- formula-not-decoded -->

For the non-euclidean case , we have that

<!-- formula-not-decoded -->

which concludes the proof of the second claim.

Next, we prove the third claim : For any state s , by the triangle inequality,

<!-- formula-not-decoded -->

by plugging the two former claims for the euclidean and non-euclidean cases, we get the required result.

Finally, the fourth claim is the same as the third claim, but with an additional A factor due to the importance sampling factor,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the same techniques of the last lemma, we prove the following technical lemma, regarding the change in the gradient of the Bregman generating function ω of two consecutive iterations of TRPO, in the sample-based case.

Lemma 27 (bound on the difference of the gradient of ω between two consecutive policies in the sample-based case) . For each state-action pair, s, a , the difference between two consecutive policies of Sample-Based TRPO is bounded by:

<!-- formula-not-decoded -->

where A ω ( k ) = t k A 3 / 2 C max ,λ 1 -γ and A ω ( k ) = t k A C max ,λ log k 1 -γ in the euclidean and non-euclidean cases respectively, k is the iteration number and t k is the step size used in the update.

Proof. In both the euclidean in non-euclidean cases, we discuss optimization problem (51) for the sample-based case. Thus, for any visited state in the k -th iteration, s ∈ S k M := { s ′ ∈ S : ∑ M m =1 { s ′ = s m } &gt; 0 } , by (50)

<!-- formula-not-decoded -->

where we denoted n ( s ) = ∑ a n ( s, a ) the number of times the state s was observed at the k -th episode and used the fact ˆ q π k λ ( s m , · , m i ) is sampled by unrolling the MDP. Thus, it holds that

Interestingly, because we use the importance sampling factor A in the approximation of q π k λ , we obtain an additional A factor.

<!-- formula-not-decoded -->

First, notice that for states which were not encountered in the k -th iteration, i.e., all states s for which ∑ M m =1 /BD { s = s m } = 0 , the solution of the optimization problem is π k +1 ( · | s ) = π k ( · | s ) . Thus, ∇ ω ( s ; π k +1 ) = ∇ ω ( s ; π k ) and the inequality trivially holds.

We now turn to discuss the case where ∑ M m =1 /BD { s = s m } &gt; 0 , i.e., s ∈ S k M . We separate here the analysis for the euclidean and non-euclidean cases:

For the euclidean case , ω ( · ) = 1 2 ‖·‖ 2 2 . Thus, the derivative of ω at a state s is,

<!-- formula-not-decoded -->

By the first order optimality condition, for any state s and policy π ,

<!-- formula-not-decoded -->

Plugging in π := π k ( · | s ) , we get

<!-- formula-not-decoded -->

Plugging in (74), we have that

<!-- formula-not-decoded -->

which can be also written as

<!-- formula-not-decoded -->

Bounding the RHS using the Cauchy-Schwartz inequality, we get,

<!-- formula-not-decoded -->

which is the same as

<!-- formula-not-decoded -->

Dividing by ‖∇ ω ( s ; π k +1 ) -∇ ω ( s ; π k ) ‖ 2 &gt; 0 and noticing that in case it is 0 the bound is trivially satisfied,

<!-- formula-not-decoded -->

Finally, using the norm equivalence we get,

<!-- formula-not-decoded -->

Using the fourth claim of Lemma 26 (in the euclidean setting), and the fact the this inequality holds uniformly for all s ∈ S k M concludes the result.

For the non-euclidean case , ω ( s ; π ) = ∑ a π ( a | s ) log π ( a | s ) . Thus, the derivative at the state action pair, s, a , is ∇ π ( a | s ) ω ( s ; π ) = 1 + log π ( a | s ) .

Thus, the difference between two consecutive policies is:

<!-- formula-not-decoded -->

Restating (68),

<!-- formula-not-decoded -->

First, we will bound log π k +1 ( a | s ) -log π k ( a | s ) from below:

Similarly to equation 70, bounding the last term in the RHS,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Together with the fact that λt k log π k ( a | s ) ≤ 0 , we obtain, where the last relation is by the definition of C max ,λ

Next, it is left to bound log π k +1 ( a | s ) -log π k ( a | s ) from above. Notice that,

<!-- formula-not-decoded -->

where in the first transition we used the fact that in the sample-based case ‖ ˆ q π k λ ‖ ∞ , ∞ ≤ A C max ,λ 1 -γ due to the importance sampling applied in the estimation process, in the second transition we used the fact that the exponent is minimized when λt k log π ( a ′ | s ) is maximized and the fact that log π ( a ′ | s ) ≤ 0 , and the last transition is by the fact ∑ a ′ π k ( a ′ | s ) = 0 . Thus, we have

<!-- formula-not-decoded -->

where the third transition is due to (73), and the last transition is by the the definition of C max ,λ .

Combining the two bounds we have,

<!-- formula-not-decoded -->

which concludes the proof.

Lemma 28 (bounds on initial distance D ω ) . Let π 0 be the uniform policy over all states, and D ω be an upper bound on max π ‖ B ω ( π 0 , π ) ‖ ∞ , i.e., max π ‖ B ω ( π 0 , π ) ‖ ≤ D ω . Then, the following claims hold.

1. For ω ( · ) = 1 2 ‖·‖ 2 2 , D ω = 1 .
2. For ω ( · ) = H ( · ) , D ω = log A.

Proof. For brevity, without loss of generality we omit the dependency on the state s . We start by proving the first claim. For the euclidean case,

<!-- formula-not-decoded -->

where the fifth relation holds since x 2 ≤ x for x ∈ [0 , 1] , and the sixth relation holds since π is a probability measure.

For the non-euclidean case the following relation holds.

<!-- formula-not-decoded -->

where H is the negative entropy. Since H ( π ) ≤ 0 we get that B ω ( π, π 0 ) ≤ log A and conclude the proof.

The following Lemma as many instances in previous literature (e.g., (Scherrer and Geist 2014)[Lemma 1]) in the unregularized case, when λ = 0 . Here we generalize it to the regularized case, for λ &gt; 0 .

Lemma 29 (value difference to Bellman differences) . For any policies π and π ′ , the following claims hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The first claim holds by the following relations.

<!-- formula-not-decoded -->

The second claim follows by multiplying both sides by ( I -γP π ′ ) . The third claim holds by multiplying both sides of the first claim by µ and using the definition d µ,π ′ = (1 -γ ) µ ( I -γP π ′ ) -1 .

## G Useful Lemmas from Convex Analysis

We state two basic results which are essential to the analysis of convergence. A full proof can be found in (Beck 2017).

Lemma 30 (Beck 2017, Lemma 9.11, three-points lemma) . Suppose that ω : E → ( -∞ , ∞ ] is proper closed and convex. Suppose in addition that ω is differentiable over dom ( ∂ω ) . Assume that a , b ∈ dom ( ∂ω ) and c ∈ dom ( ω ) . Then the following equality holds:

<!-- formula-not-decoded -->

Theorem 31 (Beck 2017, Theorem 9.12, non-euclidean second prox theorem) .

- ω : E → ( -∞ , ∞ ] be a proper closed and convex function differentiable over dom ( ∂ω ) .
- ψ : E → ( -∞ , ∞ ] be a proper closed and convex function satisfying dom ( ψ ) ⊆ dom ( ω ) .
- ω + δ dom ( ψ ) be σ -strongly convex ( σ &gt; 0 ).

Assume that b ∈ dom ( ∂ω ) , and let a be defined by

<!-- formula-not-decoded -->

Then a ∈ dom ( ∂ω ) and for all u ∈ dom ( ψ ) ,

<!-- formula-not-decoded -->