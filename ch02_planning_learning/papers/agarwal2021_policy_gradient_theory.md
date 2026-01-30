## On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift

Alekh Agarwal * Sham M. Kakade † Jason D. Lee ‡ Gaurav Mahajan §

## Abstract

Policy gradient methods are among the most effective methods in challenging reinforcement learning problems with large state and/or action spaces. However, little is known about even their most basic theoretical convergence properties, including: if and how fast they converge to a globally optimal solution or how they cope with approximation error due to using a restricted class of parametric policies. This work provides provable characterizations of the computational, approximation, and sample size properties of policy gradient methods in the context of discounted Markov Decision Processes (MDPs). We focus on both: 'tabular' policy parameterizations, where the optimal policy is contained in the class and where we show global convergence to the optimal policy; and parametric policy classes (considering both log-linear and neural policy classes), which may not contain the optimal policy and where we provide agnostic learning results. One central contribution of this work is in providing approximation guarantees that are average case - which avoid explicit worst-case dependencies on the size of state space - by making a formal connection to supervised learning under distribution shift . This characterization shows an important interplay between estimation error, approximation error, and exploration (as characterized through a precisely defined condition number).

## 1 Introduction

Policy gradient methods have a long history in the reinforcement learning (RL) literature [Williams, 1992, Sutton et al., 1999, Konda and Tsitsiklis, 2000, Kakade, 2001] and are an attractive class of algorithms as they are applicable to any differentiable policy parameterization; admit easy extensions to function approximation; easily incorporate structured state and action spaces; are easy to implement in a simulation based, model-free manner. Owing to their flexibility and generality, there has also been a flurry of improvements and refinements to make these ideas work robustly with deep neural network based approaches (see e.g. Schulman et al. [2015, 2017]).

* Microsoft Research, Redmond, WA 98052. Email: alekha@microsoft.com

† University of Washington, Seattle, WA 98195 &amp; Microsoft Research. Email: sham@cs.washington.edu

‡ Princeton University, Princeton, NJ 08540. Email: jasonlee@princeton.edu

§ University of California San Diego, La Jolla, CA 92093. Email: gmahajan@eng.ucsd.edu

Despite the large body of empirical work around these methods, their convergence properties are only established at a relatively coarse level; in particular, the folklore guarantee is that these methods converge to a stationary point of the objective, assuming adequate smoothness properties hold and assuming either exact or unbiased estimates of a gradient can be obtained (with appropriate regularity conditions on the variance). However, this local convergence viewpoint does not address some of the most basic theoretical convergence questions, including: 1) if and how fast they converge to a globally optimal solution (say with a sufficiently rich policy class); 2) how they cope with approximation error due to using a restricted class of parametric policies; or 3) their finite sample behavior. These questions are the focus of this work.

Overall, the results of this work place policy gradient methods under a solid theoretical footing, analogous to the global convergence guarantees of iterative value function based algorithms.

## 1.1 Our Contributions

This work focuses on first-order and quasi second-order policy gradient methods which directly work in the space of some parameterized policy class (rather than value-based approaches). We characterize the computational, approximation, and sample size properties of these methods in the context of a discounted Markov Decision Process (MDP). We focus on: 1) tabular policy parameterizations , where there is one parameter per state-action pair so the policy class is complete in that it contains the optimal policy, and 2) function approximation , where we have a restricted class or parametric policies which may not contain the globally optimal policy. Note that policy gradient methods for discrete action MDPs work in the space of stochastic policies, which permits the policy class to be differentiable. We now discuss our contributions in the both of these contexts.

Tabular case: We consider three algorithms: two of which are first order methods, projected gradient ascent (on the simplex) and gradient ascent (with a softmax policy parameterization); and the third algorithm, natural policy gradient ascent, can be viewed as a quasi second-order method (or preconditioned first-order method). Table 1 summarizes our main results in this case: upper bounds on the number of iterations taken by these algorithms to find an /epsilon1 -optimal policy, when we have access to exact policy gradients.

Arguably, the most natural starting point for an analysis of policy gradient methods is to consider directly doing gradient ascent on the policy simplex itself and then to project back onto the simplex if the constraint is violated after a gradient update; we refer to this algorithm as projected gradient ascent on the simplex. Using a notion of gradient domination [Polyak, 1963], our results provably show that any first-order stationary point of the value function results in an approximately optimal policy, under certain regularity assumptions; this allows for a global convergence analysis by directly appealing to standard results in the non-convex optimization literature.

A more practical and commonly used parameterization is the softmax parameterization, where the simplex constraint is explicitly enforced by the exponential parameterization, thus avoiding projections. This work provides the first global convergence guarantees using only first-order gradient information for the widely-used softmax parameterization. Our first result for this parameterization establishes the asymptotic convergence of the policy gradient algorithm; the analysis

Table 1: Iteration Complexities with Exact Gradients for the Tabular Case: A summary of the number of iterations required by different algorithms to find a policy π such that V /star ( s 0 ) -V π ( s 0 ) ≤ /epsilon1 for some fixed s 0 , assuming access to exact policy gradients . The first three algorithms optimize the objective E s ∼ µ [ V π ( s )] , where µ is the starting state distribution for the algorithms. The MDP has |S| states, |A| actions, and discount factor 0 ≤ γ &lt; 1 . The quantity D ∞ := max s ( d π /star s 0 ( s ) µ ( s ) ) is termed the distribution mismatch coefficient , where, roughly speaking, d π /star s 0 ( s ) is the fraction of time spent in state s when executing an optimal policy π /star , starting from the state s 0 (see (4)). The NPG algorithm directly optimizes V π ( s 0 ) for any state s 0 . In contrast to the complexities of the previous three algorithms, NPG has no dependence on the coefficient D ∞ , nor does it depend on the choice of s 0 . Both the MDP Experts Algorithm [Even-Dar et al., 2009] and MD-MPIalgorithm [Geist et al., 2019] (see Corollary 3 of their paper) also yield guarantees for the same update rule as NPG for the softmax parameterization, though at a worse rate. See Section 2 for further discussion.

| Algorithm                                                                        | Iteration complexity                           |
|----------------------------------------------------------------------------------|------------------------------------------------|
| Projected Gradient Ascent on Simplex (Thm 4.1)                                   | O ( D 2 ∞ |S||A| (1 - γ ) 6 /epsilon1 2 )      |
| Policy Gradient, softmax parameterization (Thm 5.1)                              | asymptotic                                     |
| Policy Gradient + log barrier regularization, softmax parameterization (Cor 5.1) | O ( D 2 ∞ |S| 2 |A| 2 (1 - γ ) 6 /epsilon1 2 ) |
| Natural Policy Gradient (NPG), softmax parameterization (Thm 5.3)                | 2 (1 - γ ) 2 /epsilon1                         |

challenge here is that the optimal policy (which is deterministic) is attained by sending the softmax parameters to infinity.

In order to establish a finite time, convergence rate to optimality for the softmax parameterization, we then consider a log barrier regularizer and provide an iteration complexity bound that is polynomial in all relevant quantities. The use of our log barrier regularizer is critical to avoiding the issue of gradients becomingly vanishingly small at suboptimal near-deterministic policies, an issue of significant practical relevance. The log barrier regularizer can also be viewed as using a relative entropy regularizer; here, we note the general approach of entropy based regularization is common in practice (e.g. see [Williams and Peng, 1991, Mnih et al., 2016, Peters et al., 2010, Abdolmaleki et al., 2018, Ahmed et al., 2019]). One notable distinction, which we discuss later, is that our analysis is for the log barrier regularization rather than the entropy regularization.

For these aforementioned algorithms, our convergence rates depend on the optimization measure having coverage over the state space, as measured by the distribution mismatch coefficient D ∞ (see Table 1 caption). In particular, for the convergence rates shown in Table 1 (for the afore-

mentioned algorithms), we assume that the optimization objective is the expected (discounted) cumulative value where the initial state is sampled under some distribution, and D ∞ is a measure of the coverage of this initial distribution. Furthermore, we provide a lower bound that shows such a dependence is unavoidable for first-order methods, even when exact gradients are available.

We then consider the Natural Policy Gradient (NPG) algorithm [Kakade, 2001] (also see Bagnell and Schneider [2003], Peters and Schaal [2008]), which can be considered a quasi secondorder method due to the use of its particular preconditioner, and provide an iteration complexity to achieve an /epsilon1 -optimal policy that is at most 2 (1 -γ ) 2 /epsilon1 iterations, improving upon the previous related results of [Even-Dar et al., 2009, Geist et al., 2019] (see Section 2). Note the convergence rate has no dependence on the number of states or the number of actions, nor does it depend on the distribution mismatch coefficient D ∞ . We provide a simple and concise proof for the convergence rate analysis by extending the approach developed in [Even-Dar et al., 2009], which uses a mirror descent style of analysis [Nemirovsky and Yudin, 1983, Cesa-Bianchi and Lugosi, 2006] and also handles the non-concavity of the policy optimization problem.

This fast and dimension free convergence rate shows how the variable preconditioner in the natural gradient method improves over the standard gradient ascent algorithm. The dimension free aspect of this convergence rate is worth reflecting on, especially given the widespread use of the natural policy gradient algorithm along with variants such as the Trust Region Policy Optimization (TRPO) algorithm [Schulman et al., 2015]; our results may help to provide analysis of a more general family of entropy based algorithms (see for example Neu et al. [2017]).

Function Approximation: We now summarize our results with regards to policy gradient methods in the setting where we work with a restricted policy class, which may not contain the optimal policy. In this sense, these methods can be viewed as approximate methods. Table 2 provides a summary along with the comparisons to some relevant approximate dynamic programming methods.

A long line of work in the function approximation setting focuses on mitigating the worst-case ' /lscript ∞ ' guarantees that are inherent to approximate dynamic programming methods [Bertsekas and Tsitsiklis, 1996] (see the first row in Table 2). The reason to focus on average case guarantees is that it supports the applicability of supervised machine learning methods to solve the underlying approximation problem. This is because supervised learning methods, like classification and regression, typically have bounds on the expected error under a distribution, as opposed to worst-case guarantees over all possible inputs.

The existing literature largely consists of two lines of provable guarantees that attempt to mitigate the explicit /lscript ∞ error conditions of approximate dynamic programming: those methods which utilize a problem dependent parameter (the concentrability coefficient [Munos, 2005]) to provide more refined dynamic programming guarantees (e.g. see Munos [2005], Szepesv´ ari and Munos [2005], Antos et al. [2008], Farahmand et al. [2010]) and those which work with a restricted policy class, making incremental updates, such as Conservative Policy Iteration (CPI) [Kakade and Langford, 2002, Scherrer and Geist, 2014], Policy Search by Dynamic Programming (PSDP) [Bagnell et al., 2004], and MD-MPI Geist et al. [2019]. Both styles of approaches give guarantees based on worstcase density ratios, i.e. they depend on a maximum ratio between two different densities over the state space. As discussed in[Scherrer, 2014], the assumptions in the latter class of algorithms

are substantially weaker, in that the worst-case density ratio only depends on the state visitation distribution of an optimal policy (also see Table 2 caption and Section 2).

With regards to function approximation, our main contribution is in providing performance bounds that, in some cases, have milder dependence on these density ratios. We precisely quantify an approximation/estimation error decomposition relevant for the analysis of the natural gradient method; this decomposition is stated in terms of the compatible function approximation error as introduced in Sutton et al. [1999]. More generally, we quantify our function approximation results in terms of a precisely quantified transfer error notion, based on approximation error under distribution shift . Table 2 shows a special case of our convergence rates of NPG, which is governed by four quantities: /epsilon1 stat , /epsilon1 approx , κ , and D ∞ .

For the realizable case, where all policies have values which are linear in the given features (such as in linear MDP models of [Jin et al., 2019, Yang and Wang, 2019, Jiang et al., 2017]), we have that the approximation error /epsilon1 approx is 0 . Here, our guarantees yield a fully polynomial and sample efficient convergence guarantee, provided the condition number κ is bounded. Importantly, there always exists a good (universal) initial measure that ensures κ is bounded by a quantity that is only polynomial in the dimension of the features, d , as opposed to an explicit dependence on the size of the (infinite) state space (see Remark 6.3). Such a guarantee would not be implied by algorithms which depend on the coefficients C ∞ or D ∞ . 1

Let us discuss the important special case of log-linear policies (i.e. policies that take the softmax of linear functions in a given feature space) where the relevant quantities are as follows: /epsilon1 stat is a bound on the excess risk (the estimation error) in fitting linearly parameterized value functions, which can be driven to 0 with more samples (at the usual statistical rate of O (1 / √ N ) where N is the number of samples); /epsilon1 approx is the usual notion of average squared approximation error where the target function may not be perfectly representable by a linear function; κ can be upper bounded with an inverse dependence on the minimal eigenvalue of the feature covariance matrix of the fitting measure (as such it can be viewed as a dimension dependent quantity but not necessarily state dependent); and D ∞ is as before.

Our results are also suggestive that a broader class of incremental algorithms - such as CPI [Kakade and Langford, 2002], PSDP [Bagnell et al., 2004], and MD-MPI Geist et al. [2019] which make small changes to the policy from one iteration to the next - may also permit a sharper analysis, where the dependence of worst-case density ratios can be avoided through an appropriate approximation/estimation decomposition; this is an interesting direction for future work (a point which we return to in Section 7). One significant advantage of NPG is that the explicit parametric policy representation in NPG (and other policy gradient methods) leads to a succinct policy representation in comparison to CPI, PSDP, or related boosting-style methods [Scherrer and Geist, 2014], where the representation complexity of the policy of the latter class of methods grows linearly in the number of iterations (since these methods add one policy to the ensemble per iteration). This representation complexity is likely why the latter class of algorithms are less widely used in practice.

1 Bounding C ∞ would require a restriction on the dynamics of the MDP (see Chen and Jiang [2019] and Section 2). Bounding D ∞ would require an initial state distribution that is constructed using knowledge of π /star , through d π /star . In contrast, κ can be made O ( d ) , with an initial state distribution that only depends on the geometry of the features (and does not depend on any other properties of the MDP). See Remark 6.3.

Table 2: Overview of Approximate Methods: The suboptimality, V /star ( s 0 ) -V π ( s 0 ) , after T iterations for various approximate algorithms, which use different notions of approximation error (sample complexities are not directly considered but instead may be thought of as part of /epsilon1 1 and /epsilon1 stat . See Section 2 for further discussion). Order notation is used to drop constants, and we assume |A| = 2 for ease of exposition. For approximate dynamic programming methods, the relevant error is the worst case, /lscript ∞ -error in approximating a value function, e.g. /epsilon1 ∞ = max s,a | Q π ( s, a ) -̂ Q π ( s, a ) | , where ̂ Q π is what an estimation oracle returns during the course of the algorithm. The second row (see Lemma 12 in Antos et al. [2008]) is a refinement of this approach, where /epsilon1 1 is an /lscript 1 -average error in fitting the value functions under the fitting (state) distribution µ , and, roughly, C ∞ is a worst case density ratio between the state visitation distribution of any non-stationary policy and the fitting distribution µ . For Conservative Policy Iteration, /epsilon1 1 is a related /lscript 1 -average case fitting error with respect to a fitting distribution µ , and D ∞ is as defined as before, in the caption of Table 1 (see also [Kakade and Langford, 2002]); here, D ∞ ≤ C ∞ (e.g. see Scherrer [2014]). For NPG, /epsilon1 stat and /epsilon1 approx measure the excess risk (the regret) and approximation errors in fitting the values. Roughly speaking, /epsilon1 stat is the excess squared loss relative to the best fit (among an appropriately defined parametric class) under our fitting distribution (defined with respect to the state distribution µ ). Here, /epsilon1 approx is the approximation error: the minimal possible error (in our parametric class) under our fitting distribution. The condition number κ is a relative eigenvalue condition between appropriately defined feature covariances with respect to the state visitation distribution of an optimal policy, d π /star s 0 , and the state fitting distribution µ . See text for further discussion, and Section 6 for precise statements as well as a more general result not explicitly dependent on D ∞ . 6

| Algorithm                                                                                                                  | Suboptimality after T Iterations                                     | Relevant Quantities                                                                                                             |
|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Approx. Value/Policy Iteration [Bertsekas and Tsitsiklis, 1996]                                                            | /epsilon1 ∞ (1 - γ ) 2 + γ T (1 - γ ) 2                              | /epsilon1 ∞ : /lscript ∞ error of values                                                                                        |
| Approx. Policy Iteration, with concentrability [Munos, 2005, Antos et al., 2008]                                           | C ∞ /epsilon1 1 (1 - γ ) 2 + γ T (1 - γ ) 2                          | /epsilon1 1 : an /lscript 1 average error C ∞ : concentrability (max density ratio)                                             |
| Conservative Policy Iteration [Kakade and Langford, 2002] Related: PSDP [Bagnell et al., 2004], MD-MPI Geist et al. [2019] | D ∞ /epsilon1 1 (1 - γ ) 2 + 1 (1 - γ ) √ T                          | /epsilon1 1 : an /lscript 1 average error D ∞ : max density ratio to opt., D ∞ ≤ C ∞                                            |
| Natural Policy Gradient (Cor. 6.1 and Thm. 6.2)                                                                            | √ κ/epsilon1 stat + D ∞ /epsilon1 approx (1 - γ ) 3 + 1 (1 - γ ) √ T | /epsilon1 stat : excess risk /epsilon1 approx : approx. error κ : a condition number D ∞ : max density ratio to opt., D ∞ ≤ C ∞ |

## 2 Related Work

We now discuss related work, roughly in the order which reflects our presentation of results in the previous section.

For the direct policy parameterization in the tabular case, we make use of a gradient dominationlike property, namely any first-order stationary point of the policy value is approximately optimal up to a distribution mismatch coefficient. A variant of this result also appears in Theorem 2 of Scherrer and Geist [2014], which itself can be viewed as a generalization of the approach in Kakade and Langford [2002]. In contrast to CPI [Kakade and Langford, 2002] and the more general boosting-based approach in Scherrer and Geist [2014], we phrase this approach as a Polyaklike gradient domination property [Polyak, 1963] in order to directly allow for the transfer of any advances in non-convex optimization to policy optimization in RL. More broadly, it is worth noting the global convergence of policy gradients for Linear Quadratic Regulators [Fazel et al., 2018] also goes through a similar proof approach of gradient domination.

Empirically, the recent work of Ahmed et al. [2019] studies entropy based regularization and shows the value of regularization in policy optimization, even with exact gradients. This is related to our use of the log barrier regularization.

For our convergence results of the natural policy gradient algorithm in the tabular setting, there are close connections between our results and the works of Even-Dar et al. [2009], Geist et al. [2019]. Even-Dar et al. [2009] provides provable online regret guarantees in changing MDPs utilizing experts algorithms (also see Neu et al. [2010], Abbasi-Yadkori et al. [2019a]); as a special case, their MDP Experts Algorithm is equivalent to the natural policy gradient algorithm with the softmax policy parameterization. While the convergence result due to Even-Dar et al. [2009] was not specifically designed for this setting, it is instructive to see what it implies due to the close connections between optimization and regret [Cesa-Bianchi and Lugosi, 2006, Shalev-Shwartz et al., 2012]. The Mirror Descent-Modified Policy Iteration (MD-MPI) algorithm [Geist et al., 2019] with negative entropy as the Bregman divergence results is an identical algorithm as NPG for softmax parameterization in the tabular case; Corollary 3 [Geist et al., 2019] applies to our updates, leading to a bound worse by a 1 / (1 -γ ) factor and also has logarithmic dependence on |A| . Our proof for this case is concise and may be of independent interest. Also worth noting is the Dynamic Policy Programming of Azar et al. [2012], which is an actor-critic algorithm with a softmax parameterization; this algorithm, even though not identical, comes with similar guarantees in terms of its rate (it is weaker in terms of an additional 1 / (1 -γ ) factor) than the NPG algorithm.

Wenowturn to function approximation, starting with a discussion of iterative algorithms which make incremental updates in which the next policy is effectively constrained to be close to the previous policy, such as in CPI and PSDP [Bagnell et al., 2004]. Here, the work in Scherrer and Geist [2014] show how CPI is part of broader family of boosting-style methods. Also, with regards to PSDP, the work in Scherrer [2014] shows how PSDP actually enjoys an improved iteration complexity over CPI, namely O (log 1 //epsilon1 opt ) vs. O (1 //epsilon1 2 opt ) . It is worthwhile to note that both NPG and projected gradient ascent are also both incremental algorithms.

We now discuss the approximate dynamic programming results characterized in terms of the concentrability coefficient. Broadly we use the term approximate dynamic programming to refer to fitted value iteration, fitted policy iteration and more generally generalized policy iteration

schemes such as classification-based policy iteration as well, in addition to the classical approximate value/policy iteration works. While the approximate dynamic programming results typically require /lscript ∞ bounded errors, which is quite stringent, the notion of concentrability (originally due to [Munos, 2003, 2005]) permits sharper bounds in terms of average case function approximation error, provided that the concentrability coefficient is bounded (e.g. see Munos [2005], Szepesv´ ari and Munos [2005], Antos et al. [2008], Lazaric et al. [2016]). Chen and Jiang [2019] provide a more detailed discussion on this quantity. Based on this problem dependent constant being bounded, Munos [2005], Szepesv´ ari and Munos [2005], Antos et al. [2008] and Lazaric et al. [2016] provide meaningful sample size and error bounds for approximate dynamic programming methods, where there is a data collection policy (under which value-function fitting occurs) that induces a concentrability coefficient. In terms of the concentrability coefficient C ∞ and the 'distribution mismatch coefficient' D ∞ in Table 2 , we have that D ∞ ≤ C ∞ , as discussed in [Scherrer, 2014] (also see the table caption). Also, as discussed in Chen and Jiang [2019], a finite concentrability coefficient is a restriction on the MDP dynamics itself, while a bounded D ∞ does not require any restrictions on the MDP dynamics. The more refined quantities defined by Farahmand et al. [2010] (for the approximate policy iteration result) partially alleviate some of these concerns, but their assumptions still implicitly constrain the MDP dynamics, like the finiteness of the concentrability coefficient.

Assuming bounded concentrability coefficient, there are a notable set of provable average case guarantees for the MD-MPI algorithm [Geist et al., 2019] (see also [Azar et al., 2012, Scherrer et al., 2015]), which are stated in terms of various norms of function approximation error. MD-MPI is a class of algorithms for approximate planning under regularized notions of optimality in MDPs. Specifically, Geist et al. [2019] analyze a family of actor-critic style algorithms, where there are both approximate value functions updates and approximate policy updates. As a consequence of utilizing approximate value function updates for the critic, the guarantees of Geist et al. [2019] are stated with dependencies on concentrability coefficients.

When dealing with function approximation, computational and statistical complexities are relevant because they determine the effectiveness of approximate updates with finite samples. With regards to sample complexity, the work in Szepesv´ ari and Munos [2005], Antos et al. [2008] provide finite sample rates (as discussed above), further generalized to actor-critic methods in Azar et al. [2012], Scherrer et al. [2015]. In our policy optimization approach, the analysis of both computational and statistical complexities are straightforward, since we can leverage known statistical and computational results from the stochastic approximation literature; in particular, we use the stochastic projected gradient ascent to obtain a simple, linear time method for the critic estimation step in the natural policy gradient algorithm.

In terms of the algorithmic updates for the function approximation setting, our development of NPGbears similarity to the natural actor-critic algorithm Peters and Schaal [2008], for which some asymptotic guarantees under finite concentrability coefficients are obtained in Bhatnagar et al. [2009]. While both updates seek to minimize the compatible function approximation error, we perform streaming updates based on stochastic optimization using Monte Carlo estimates for values. In contrast Peters and Schaal [2008] utilize Least Squares Temporal Difference methods [Boyan, 1999] to minimize the loss. As a consequence, their updates additionally make linear approximations to

the value functions in order to estimate the advantages; our approach is flexible in allowing for wide family of smoothly differentiable policy classes (including neural policies).

Finally, we remark on some concurrent works. The work of Bhandari and Russo [2019] provides gradient domination-like conditions under which there is (asymptotic) global convergence to the optimal policy. Their results are applicable to the projected gradient ascent algorithm; they are not applicable to gradient ascent with the softmax parameterization (see the discussion in Section 5 herein for the analysis challenges). Bhandari and Russo [2019] also provide global convergence results beyond MDPs. Also, Liu et al. [2019] provide an analysis of the TRPO algorithm [Schulman et al., 2015] with neural network parameterizations, which bears resemblance to our natural policy gradient analysis. In particular, Liu et al. [2019] utilize ideas from both Even-Dar et al. [2009] (with a mirror descent style of analysis) along with Cai et al. [2019] (to handle approximation with neural networks) to provide conditions under which TRPO returns a near optimal policy. Liu et al. [2019] do not explicitly consider the case where the policy class is not complete (i.e when there is approximation). Another related work of Shani et al. [2019] considers the TRPO algorithm and provides theoretical guarantees in the tabular case; their convergence rates with exact updates are O (1 / √ T ) for the (unregularized) objective function of interest; they also provide faster rates on a modified (regularized) objective function. They do not consider the case of infinite state spaces and function approximation. The closely related recent papers [Abbasi-Yadkori et al., 2019a,b] also consider closely related algorithms to the Natural Policy Gradient approach studied here, in an infinite horizon, average reward setting. Specifically, the EE-POLITEX algorithm is closely related to the Q-NPG algorithm which we study in Section 6.2, though our approach is in the discounted setting. We adopt the name Q-NPG to capture its close relationship with the NPG algorithm, with the main difference being the use of function approximation for the Q -function instead of advantages. We refer the reader to Section 6.2 (and Remark 6.5) for more discussion of the technical differences between the two works.

## 3 Setting

A (finite) Markov Decision Process (MDP) M = ( S , A , P, r, γ, ρ ) is specified by: a finite state space S ; a finite action space A ; a transition model P where P ( s ′ | s, a ) is the probability of transitioning into state s ′ upon taking action a in state s ; a reward function r : S × A → [0 , 1] where r ( s, a ) is the immediate reward associated with taking action a in state s ; a discount factor γ ∈ [0 , 1) ; a starting state distribution ρ over S .

A policy induces a distribution over trajectories τ = ( s t , a t , r t ) ∞ t =0 , where s 0 is drawn from the starting state distribution ρ , and, for all subsequent timesteps t , a t ∼ π ( ·| s t ) and s t +1 ∼ P ( ·| s t , a t ) . The value function V π : S → R is defined as the discounted sum of future rewards starting at state

A deterministic, stationary policy π : S → A specifies a decision-making strategy in which the agent chooses actions adaptively based on the current state, i.e., a t = π ( s t ) . The agent may also choose actions according to a stochastic policy π : S → ∆( A ) (where ∆( A ) is the probability simplex over A ), and, overloading notation, we write a t ∼ π ( ·| s t ) .

s and executing π , i.e.

where the expectation is with respect to the randomness of the trajectory τ induced by π in M . Since we assume that r ( s, a ) ∈ [0 , 1] , we have 0 ≤ V π ( s ) ≤ 1 1 -γ . We overload notation and define V π ( ρ ) as the expected value under the initial state distribution ρ , i.e.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The action-value (or Q-value) function Q π : S × A → R and the advantage function A π : S × A → R are defined as:

<!-- formula-not-decoded -->

The goal of the agent is to find a policy π that maximizes the expected value from the initial state, i.e. the optimization problem the agent seeks to solve is:

<!-- formula-not-decoded -->

where the max is over all policies. The famous theorem of Bellman and Dreyfus [1959] shows there exists a policy π /star which simultaneously maximizes V π ( s 0 ) , for all states s 0 ∈ S .

Policy Parameterizations. This work studies ascent methods for the optimization problem:

<!-- formula-not-decoded -->

where { π θ | θ ∈ Θ } is some class of parametric (stochastic) policies. We consider a number of different policy classes. The first two are complete in the sense that any stochastic policy can be represented in the class. The final class may be restrictive. These classes are as follows:

- Direct parameterization: The policies are parameterized by

<!-- formula-not-decoded -->

- Softmax parameterization: For unconstrained θ ∈ R |S||A| ,

where θ ∈ ∆( A ) |S| , i.e. θ is subject to θ s,a ≥ 0 and ∑ a ∈A θ s,a = 1 for all s ∈ S and a ∈ A .

<!-- formula-not-decoded -->

The softmax parameterization is also complete.

<!-- image -->

Figure 1: (Non-concavity example) A deterministic MDP corresponding to Lemma 3.1 where V π θ ( s ) is not concave. Numbers on arrows represent the rewards for each action.

<!-- image -->

a

Figure 2: (Vanishing gradient example) A deterministic, chain MDP of length H + 2 . We consider a policy where π ( a | s i ) = θ s i ,a for i = 1 , 2 , . . . , H . Rewards are 0 everywhere other than r ( s H +1 , a 1 ) = 1 . See Proposition 4.1.

- Restricted parameterizations: We also study parametric classes { π θ | θ ∈ Θ } that may not contain all stochastic policies. In particular, we pay close attention to both log-linear policy classes and neural policy classes (see Section 6). Here, the best we may hope for is an agnostic result where we do as well as the best policy in this class.

While the softmax parameterization is the more natural parametrization among the two complete policy classes, it is also informative to consider the direct parameterization.

It is worth explicitly noting that V π θ ( s ) is non-concave in θ for both the direct and the softmax parameterizations, so the standard tools of convex optimization are not applicable. For completeness, we formalize this as follows (with a proof in Appendix A, along with an example in Figure 1):

Lemma 3.1. There is an MDP M (described in Figure 1) such that the optimization problem V π θ ( s ) is not concave for both the direct and softmax parameterizations.

Policy gradients. In order to introduce these methods, it is useful to define the discounted state visitation distribution d π s 0 of a policy π as:

<!-- formula-not-decoded -->

where Pr π ( s t = s | s 0 ) is the state visitation probability that s t = s , after we execute π starting at state s 0 . Again, we overload notation and write:

<!-- formula-not-decoded -->

where d π ρ is the discounted state visitation distribution under initial distribution ρ .

The policy gradient functional form (see e.g. Williams [1992], Sutton et al. [1999]) is then:

<!-- formula-not-decoded -->

Furthermore, if we are working with a differentiable parameterization of π θ ( ·| s ) that explicitly constrains π θ ( ·| s ) to be in the simplex, i.e. π θ ∈ ∆( A ) |S| for all θ , then we also have:

<!-- formula-not-decoded -->

Note the above gradient expression (Equation 6) does not hold for the direct parameterization, while Equation 5 is valid. 2

The performance difference lemma. The following lemma is helpful throughout:

Lemma 3.2. (The performance difference lemma [Kakade and Langford, 2002]) For all policies π, π ′ and states s 0 ,

<!-- formula-not-decoded -->

For completeness, we provide a proof in Appendix A.

The distribution mismatch coefficient. We often characterize the difficulty of the exploration problem faced by our policy optimization algorithms when maximizing the objective V π ( µ ) through the following notion of distribution mismatch coefficient .

Definition 3.1 (Distribution mismatch coefficient) . Given a policy π and measures ρ, µ ∈ ∆( S ) , we refer to ∥ ∥ ∥ d π ρ µ ∥ ∥ ∥ ∞ as the distribution mismatch coefficient of π relative to µ . Here, d π ρ µ denotes componentwise division.

We often instantiate this coefficient with µ as the initial state distribution used in a policy optimization algorithm, ρ as the distribution to measure the sub-optimality of our policy (this is the start state distribution of interest), and where π above is often chosen to be π /star ∈ argmax π ∈ Π V π ( ρ ) , given a policy class Π .

Notation. Following convention, we use V /star and Q /star to denote V π /star and Q π /star respectively. For iterative algorithms which obtain policy parameters θ ( t ) at iteration t , we let π ( t ) , V ( t ) and A ( t ) denote the corresponding quantities parameterized by θ ( t ) , i.e. π θ ( t ) , V θ ( t ) and A θ ( t ) , respectively. For vectors u and v , we use u v to denote the componentwise ratio; u ≥ v denotes a componentwise inequality; we use the standard convention where ‖ v ‖ 2 = √∑ i v 2 i , ‖ v ‖ 1 = ∑ i | v i | , and ‖ v ‖ ∞ = max i | v i | .

2 This is due to ∑ a ∇ θ π θ ( a | s ) = 0 not explicitly being maintained by the direct parameterization.

## 4 Warmup: Constrained Tabular Parameterization

Our starting point is, arguably, the simplest first-order method: we directly take gradient ascent updates on the policy simplex itself and then project back onto the simplex if the constraints are violated after a gradient update. This algorithm is projected gradient ascent on the direct policy parametrization of the MDP, where the parameters are the state-action probabilities, i.e. θ s,a = π θ ( a | s ) (see (2)). As noted in Lemma 3.1, V π θ ( s ) is non-concave in the parameters π θ . Here, we first prove that V π θ ( µ ) satisfies a Polyak-like gradient domination condition [Polyak, 1963], and this tool helps in providing convergence rates. The basic approach was also used in the analysis of CPI [Kakade and Langford, 2002]; related gradient domination-like lemmas also appeared in Scherrer and Geist [2014].

It is instructive to consider this special case due to the connections it makes to the non-convex optimization literature. We also provide a lower bound that rules out algorithms whose runtime appeals to the curvature of saddle points (e.g. [Nesterov and Polyak, 2006, Ge et al., 2015, Jin et al., 2017]).

For the direct policy parametrization where θ s,a = π θ ( a | s ) , the gradient is:

<!-- formula-not-decoded -->

using (5). In particular, for this parameterization, we may write ∇ π V π ( µ ) instead of ∇ θ V π θ ( µ ) .

## 4.1 Gradient Domination

Informally, we say a function f ( θ ) satisfies a gradient domination property if for all θ ∈ Θ ,

<!-- formula-not-decoded -->

where θ /star ∈ argmax θ ′ ∈ Θ f ( θ ′ ) and where G ( θ ) is some suitable scalar notion of first-order stationarity, which can be considered a measure of how large the gradient is (see [Karimi et al., 2016, Bolte et al., 2007, Attouch et al., 2010]). Thus if one can find a θ that is (approximately) a firstorder stationary point, then the parameter θ will be near optimal (in terms of function value). Such conditions are a standard device to establishing global convergence in non-convex optimization, as they effectively rule out the presence of bad critical points. In other words, given such a condition, quantifying the convergence rate for a specific algorithm, like say projected gradient ascent, will require quantifying the rate of its convergence to a first-order stationary point, for which one can invoke standard results from the optimization literature.

The following lemma shows that the direct policy parameterization satisfies a notion of gradient domination. This is the basic approach used in the analysis of CPI [Kakade and Langford, 2002]; a variant of this lemma also appears in Scherrer and Geist [2014]. We give a proof for completeness.

Even though we are interested in the value V π ( ρ ) , it is helpful to consider the gradient with respect to another state distribution µ ∈ ∆( S ) .

Lemma 4.1 (Gradient domination) . For the direct policy parameterization (as in (2) ), for all state distributions µ, ρ ∈ ∆( S ) , we have where the max is over the set of all policies, i.e. ¯ π ∈ ∆( A ) |S| .

<!-- formula-not-decoded -->

Before we provide the proof, a few comments are in order with regards to the performance measure ρ and the optimization measure µ . Subtly, note that although the gradient is with respect to V π ( µ ) , the final guarantee applies to all distributions ρ . The significance is that even though we may be interested in our performance under ρ , it may be helpful to optimize under the distribution µ . To see this, note the lemma shows that a sufficiently small gradient magnitude in the feasible directions implies the policy is nearly optimal in terms of its value, but only if the state distribution of π , i.e. d π µ , adequately covers the state distribution of some optimal policy π /star . Here, it is also worth recalling the theorem of Bellman and Dreyfus [1959] which shows there exists a single policy π /star that is simultaneously optimal for all starting states s 0 . Note that the hardness of the exploration problem is captured through the distribution mismatch coefficient (Definition 3.1).

Proof: [ of Lemma 4.1 ] By the performance difference lemma (Lemma 3.2),

<!-- formula-not-decoded -->

where the last inequality follows since max ¯ a A π ( s, ¯ a ) ≥ 0 for all states s and policies π . We wish

to upper bound (8). We then have:

<!-- formula-not-decoded -->

where the first step follows since max ¯ π is attained at an action which maximizes A π ( s, · ) (per state); the second step follows as ∑ a π ( a | s ) A π ( s, a ) = 0 ; the third step uses ∑ a (¯ π ( a | s ) -π ( a | s )) V π ( s ) = 0 for all s ; and the final step follows from the gradient expression (see (7)). Using this in (8), where the last step follows due to max ¯ π ∈ ∆( A ) |S| (¯ π -π ) /latticetop ∇ π V π ( µ ) ≥ 0 for any policy π and d π µ ( s ) ≥ (1 -γ ) µ ( s ) (see (4)).

<!-- formula-not-decoded -->

In a sense, the use of an appropriate µ circumvents the issues of strategic exploration. It is natural to ask whether this additional term is necessary, a question which we return to. First, we provide a convergence rate for the projected gradient ascent algorithm.

## 4.2 Convergence Rates for Projected Gradient Ascent

Using this notion of gradient domination, we now give an iteration complexity bound for projected gradient ascent over the space of stochastic policies, i.e. over ∆( A ) |S| . The projected gradient ascent algorithm updates

<!-- formula-not-decoded -->

where P ∆( A ) |S| is the projection onto ∆( A ) |S| in the Euclidean norm.

Theorem 4.1. The projected gradient ascent algorithm (9) on V π ( µ ) with stepsize η = (1 -γ ) 3 2 γ |A| satisfies for all distributions ρ ∈ ∆( S ) ,

<!-- formula-not-decoded -->

A proof is provided in Appendix B.1. The proof first invokes a standard iteration complexity result of projected gradient ascent to show that the gradient magnitude with respect to all feasible directions is small. More concretely, we show the policy is /epsilon1 -stationary 3 , that is, for all π θ + δ ∈ ∆( A ) |S| and ‖ δ ‖ 2 ≤ 1 , δ /latticetop ∇ π V π θ ( µ ) ≤ /epsilon1 . We then use Lemma 4.1 to complete the proof.

Note that the guarantee we provide is for the best policy found over the T rounds, which we obtain from a bound on the average norm of the gradients. This type of a guarantee is standard in the non-convex optimization literature, where an average regret bound cannot be used to extract a single good solution, e.g. by averaging. In the context of policy optimization, this is not a serious limitation as we collect on-policy trajectories for each policy in doing sample-based gradient estimation, and these samples can be also used to estimate the policy's value. Note that the evaluation step is not required for every policy, and can also happen on a schedule, though we still need to evaluate O ( T ) policies to obtain the convergence rates described here.

## 4.3 A Lower Bound: Vanishing Gradients and Saddle Points

To understand the necessity of the distribution mismatch coefficient in Lemma 4.1 and Theorem 4.1, let us first give an informal argument that some condition on the state distribution of π , or equivalently µ , is necessary for stationarity to imply optimality. For example, in a sparse-reward MDP (where the agent is only rewarded upon visiting some small set of states), a policy that does not visit any rewarding states will have zero gradient, even though it is arbitrarily suboptimal in terms of values. Below, we give a more quantitative version of this intuition, which demonstrates that even if π chooses all actions with reasonable probabilities (and hence the agent will visit all states if the MDP is connected), then there is an MDP where a large fraction of the policies π have vanishingly small gradients, and yet these policies are highly suboptimal in terms of their value.

Concretely, consider the chain MDP of length H + 2 shown in Figure 2. The starting state of interest is state s 0 and the discount factor γ = H/ ( H + 1) . Suppose we work with the direct parameterization, where π θ ( a | s ) = θ s,a for a = a 1 , a 2 , a 3 and π θ ( a 4 | s ) = 1 -θ s,a 1 -θ s,a 2 -θ s,a 3 . Note we do not over-parameterize the policy. For this MDP and policy structure, if we were to initialize the probabilities over actions, say deterministically, then there is an MDP (obtained by permuting the actions) where all the probabilities for a 1 will be less than 1 / 4 .

The following result not only shows that the gradient is exponentially small in H , it also shows that many higher order derivatives, up to O ( H/ log H ) , are also exponentially small in H .

Proposition 4.1 (Vanishing gradients at suboptimal parameters) . Consider the chain MDP of Figure 2, with H +2 states, γ = H/ ( H +1) , and with the direct policy parameterization (with 3 |S| parameters, as described in the text above). Suppose θ is such that 0 &lt; θ &lt; 1 (componentwise) and θ s,a 1 &lt; 1 / 4 (for all states s ). For all k ≤ H 40 log(2 H ) -1 , we have ‖∇ k θ V π θ ( s 0 ) ‖ ≤ (1 / 3) H/ 4 , where ∇ k θ V π θ ( s 0 ) is a tensor of the k th order derivatives of V π θ ( s 0 ) and the norm is the operator norm of the tensor. 4 Furthermore, V /star ( s 0 ) -V π θ ( s 0 ) ≥ ( H +1) / 8 -( H +1) 2 / 3 H .

3 See Appendix B.1 for discussion on this definition.

4 The operator norm of a k th -order tensor J ∈ R d ⊗ k is defined as sup u 1 ,...,u k ∈ R d : ‖ u i ‖ 2 =1 〈 J, u 1 ⊗ . . . ⊗ u d 〉 .

This lemma also suggests that results in the non-convex optimization literature, on escaping from saddle points, e.g. [Nesterov and Polyak, 2006, Ge et al., 2015, Jin et al., 2017], do not directly imply global convergence due to that the higher order derivatives are small.

Remark 4.1. (Exact vs. Approximate Gradients) The chain MDP of Figure 2, is a common example where sample based estimates of gradients will be 0 under random exploration strategies; there is an exponentially small in H chance of hitting the goal state under a random exploration strategy. Note that this lemma is with regards to exact gradients. This suggests that even with exact computations (along with using exact higher order derivatives) we might expect numerical instabilities.

Remark 4.2. (Comparison with the upper bound) The lower bound does not contradict the upper bound of Theorem 4.1 (where a small gradient is turned into a small policy suboptimality bound), as the distribution mismatch coefficient, as defined in Definition 3.1, could be infinite in the chain MDPofFigure 2, since the start-state distribution is concentrated on one state only. More generally,

∥ ∥ Remark 4.3. (Comparison with information-theoretic lower bounds) The lower bound here is not information theoretic , in that it does not present a hard problem instance for all algorithms. Indeed, exploration algorithms for tabular MDPs starting from E 3 [Kearns and Singh, 2002], RMAX [Brafman and Tenne 2003] and several subsequent works yield polynomial sample complexities for the chain MDP. Proposition 4.1 should be interpreted as a hardness result for the specific class of policy gradient like approaches that search for a policy with a small policy gradient, as these methods will find the initial parameters to be valid in terms of the size of (several orders of) gradients. In particular, it precludes any meaningful claims on global optimality, based just on the size of the policy gradients, without additional assumptions as discussed in the previous remark.

<!-- formula-not-decoded -->

The proof is provided in Appendix B.2. The lemma illustrates that lack of good exploration can indeed be detrimental in policy gradient algorithms, since the gradient can be small either due to π being near-optimal, or, simply because π does not visit advantageous states often enough. In this sense, it also demonstrates the necessity of the distribution mismatch coefficient in Lemma 4.1.

## 5 The Softmax Tabular Parameterization

We now consider the softmax policy parameterization (3). Here, we still have a non-concave optimization problem in general, as shown in Lemma 3.1, though we do show that global optimality can be reached under certain regularity conditions. From a practical perspective, the softmax parameterization of policies is preferable to the direct parameterization, since the parameters θ are unconstrained and standard unconstrained optimization algorithms can be employed. However, optimization over this policy class creates other challenges as we study in this section, as the optimal policy (which is deterministic) is attained by sending the parameters to infinity.

We study three algorithms for this problem. The first performs direct policy gradient ascent on the objective without modification, while the second adds a log barrier regularizer to keep the

parameters from becoming too large, as a means to ensure adequate exploration. Finally, we study the natural policy gradient algorithm and establish a global optimality result with no dependence on the distribution mismatch coefficient or dimension-dependent factors.

For the softmax parameterization, the gradient takes the form:

<!-- formula-not-decoded -->

(see Lemma C.1 for a proof).

## 5.1 Asymptotic Convergence, without Regularization

Due to the exponential scaling with the parameters θ in the softmax parameterization, any policy that is nearly deterministic will have gradients close to 0 . In spite of this difficulty, we provide a positive result that gradient ascent asymptotically converges to the global optimum for the softmax parameterization.

The update rule for gradient ascent is:

<!-- formula-not-decoded -->

Theorem 5.1 (Global convergence for softmax parameterization) . Assume we follow the gradient ascent update rule as specified in Equation (11) and that the distribution µ is strictly positive i.e. µ ( s ) &gt; 0 for all states s . Suppose η ≤ (1 -γ ) 3 8 , then we have that for all states s , V ( t ) ( s ) → V /star ( s ) as t →∞ .

Remark 5.1. (Strict positivity of µ and exploration) Theorem 5.1 assumed that optimization distribution µ was strictly positive, i.e. µ ( s ) &gt; 0 for all states s . We leave it is an open question of whether or not gradient ascent will globally converge if this condition is not met. The concern is that if this condition is not met, then gradient ascent may not globally converge due to that d π θ µ ( s ) effectively scales down the learning rate for the parameters associated with state s (see (10)).

The complete proof is provided in the Appendix C.1. We now discuss the subtleties in the proof and show why the softmax parameterization precludes a direct application of the gradient domination lemma. In order to utilize the gradient domination property (in Lemma 4.1), we would desire to show that: ∇ π V π ( µ ) → 0 . However, using the functional form of the softmax parameterization (see Lemma C.1) and (7), we have that:

<!-- formula-not-decoded -->

Hence, we see that even if ∇ θ V π θ ( µ ) → 0 , we are not guaranteed that ∇ π V π θ ( µ ) → 0 .

We now briefly discuss the main technical challenges in the proof. The proof first shows that the sequence V ( t ) ( s ) is monotone increasing pointwise, i.e. for every state s , V ( t +1) ( s ) ≥ V ( t ) ( s ) (Lemma C.2). This implies the existence of a limit V ( ∞ ) ( s ) by the monotone convergence theorem

(Lemma C.3). Based on the limiting quantities V ( ∞ ) ( s ) and Q ( ∞ ) ( s, a ) , which we show exist, define the following limiting sets for each state s :

<!-- formula-not-decoded -->

The challenge is to then show that, for all states s , the set I s + is the empty set, which would immediately imply V ( ∞ ) ( s ) = V /star ( s ) . The proof proceeds by contradiction, assuming that I s + is non-empty. Using that I s + is non-empty and that the gradient tends to zero in the limit, i.e. ∇ θ V π θ ( µ ) → 0 , we have that for all a ∈ I s + , π ( t ) ( a | s ) → 0 (see (10)). This, along with the functional form of the softmax parameterization, implies that there must be divergence (in magnitude) among the set of parameters associated with some action a at state s , i.e. that max a ∈A | θ ( t ) s,a | → ∞ . The primary technical challenge in the proof is to then use this divergence, along with the dynamics of gradient ascent, to show that I s + is empty via a contradiction.

We leave it as a question for future work as to characterizing the convergence rate, which we conjecture is exponentially slow in some of the relevant quantities, such as in terms of the size of state space. Here, we turn to a regularization based approach to ensure convergence at a polynomial rate in all relevant quantities.

## 5.2 Polynomial Convergence with Log Barrier Regularization

Due to the exponential scaling with the parameters θ , policies can rapidly become near deterministic, when optimizing under the softmax parameterization, which can result in slow convergence. Indeed a key challenge in the asymptotic analysis in the previous section was to handle the growth of the absolute values of parameters as they tend to infinity. A common practical remedy for this is to use entropy-based regularization to keep the probabilities from getting too small [Williams and Peng, 1991, Mnih et al., 2016], and we study gradient ascent on a similarly regularized objective in this section. Recall that the relative-entropy for distributions p and q is defined as: KL ( p, q ) := E x ∼ p [ -log q ( x ) /p ( x )] . Denote the uniform distribution over a set X by Unif X , and define the following log barrier regularized objective as:

<!-- formula-not-decoded -->

where λ is a regularization parameter. The constant (i.e. the last term) is not relevant with regards to optimization. This regularizer is different from the more commonly utilized entropy regularizer as in Mnih et al. [2016], a point which we return to in Remark 5.2.

The policy gradient ascent updates for L λ ( θ ) are given by:

<!-- formula-not-decoded -->

Our next theorem shows that approximate first-order stationary points of the entropy-regularized objective are approximately globally optimal, provided the regularization is sufficiently small.

Theorem 5.2. (Log barrier regularization) Suppose θ is such that:

<!-- formula-not-decoded -->

and /epsilon1 opt ≤ λ/ (2 |S| |A| ) . Then we have that for all starting state distributions ρ :

Proof: The proof consists of showing that max a A π θ ( s, a ) ≤ 2 λ/ ( µ ( s ) |S| ) for all states. To see that this is sufficient, observe that by the performance difference lemma (Lemma 3.2),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which would then complete the proof.

We now proceed to show that max a A π θ ( s, a ) ≤ 2 λ/ ( µ ( s ) |S| ) . For this, it suffices to bound A π θ ( s, a ) for any state-action pair s, a where A π θ ( s, a ) ≥ 0 else the claim is trivially true. Consider an ( s, a ) pair such that A π θ ( s, a ) &gt; 0 . Using the policy gradient expression for the softmax parameterization (see Lemma C.1),

<!-- formula-not-decoded -->

The gradient norm assumption ‖∇ θ L λ ( θ ) ‖ 2 ≤ /epsilon1 opt implies that:

<!-- formula-not-decoded -->

where we have used A π θ ( s, a ) ≥ 0 . Rearranging and using our assumption /epsilon1 opt ≤ λ/ (2 |S| |A| ) ,

<!-- formula-not-decoded -->

Solving for A π θ ( s, a ) in (14), we have:

where the penultimate step uses /epsilon1 opt ≤ λ/ (2 |S| |A| ) and the final step uses d π θ µ ( s ) ≥ (1 -γ ) µ ( s ) . This completes the proof.

<!-- formula-not-decoded -->

By combining the above theorem with standard results on the convergence of gradient ascent (to first order stationary points), we obtain the following corollary.

Corollary 5.1. (Iteration complexity with log barrier regularization) Let β λ := 8 γ (1 -γ ) 3 + 2 λ |S| . Starting from any initial θ (0) , consider the updates (13) with λ = /epsilon1 (1 -γ ) 2 ∥ ∥ ∥ ∥ d π /star ρ µ ∥ ∥ ∥ ∥ ∞ and η = 1 /β λ . Then for all starting state distributions ρ , we have

<!-- formula-not-decoded -->

See Appendix C.2 for the proof. The corollary shows the importance of balancing how the regularization parameter λ is set relative to the desired accuracy /epsilon1 , as well as the importance of the initial distribution µ to obtain global optimality.

Remark 5.2. (Entropy vs. log barrier regularization) The more commonly considered regularizer is the entropy [Mnih et al., 2016] (also see Ahmed et al. [2019] for a more detailed empirical investigation), where the regularizer would be:

<!-- formula-not-decoded -->

Note the entropy is far less aggressive in penalizing small probabilities, in comparison to the log barrier, which is equivalent to the relative entropy. In particular, the entropy regularizer is always bounded between 0 and log |A| , while the relative entropy (against the uniform distribution over actions), is bounded between 0 and infinity, where it tends to infinity as probabilities tend to 0 . We leave it is an open question if a polynomial convergence rate 5 is achievable with the more common entropy regularizer; our polynomial convergence rate using the KL regularizer crucially relies on the aggressive nature in which the relative entropy prevents small probabilities (the proof shows that any action, with a positive advantage, has a significant probability for any near-stationary policy of the regularized objective).

5 Here, ideally we would like to be poly in |S| , |A| , 1 / (1 -γ ) , 1 //epsilon1 , and the distribution mismatch coefficient, which we conjecture may not be possible.

## 5.3 Dimension-free Convergence of Natural Policy Gradient Ascent

We now show the Natural Policy Gradient algorithm, with the softmax parameterization (3), obtains an improved iteration complexity. The NPG algorithm defines a Fisher information matrix (induced by π ), and performs gradient updates in the geometry induced by this matrix as follows:

<!-- formula-not-decoded -->

where M † denotes the Moore-Penrose pseudoinverse of the matrix M . Throughout this section, we restrict to using the initial state distribution ρ ∈ ∆( S ) in our update rule in (15) (so our optimization measure µ and the performance measure ρ are identical). Also, we restrict attention to states s ∈ S reachable from ρ , since, without loss of generality, we can exclude states that are not reachable under this start state distribution 6 .

We leverage a particularly convenient form the update takes for the softmax parameterization (see Kakade [2001]). For completeness, we provide a proof in Appendix C.3.

Lemma 5.1. (NPG as soft policy iteration) For the softmax parameterization (3) , the NPG updates (15) take the form:

<!-- formula-not-decoded -->

The updates take a strikingly simple form in this special case; they are identical to the classical multiplicative weights updates [Freund and Schapire, 1997, Cesa-Bianchi and Lugosi, 2006] for online linear optimization over the probability simplex, where the linear functions are specified by the advantage function of the current policy at each iteration. Notably, there is no dependence on the state distribution d ( t ) ρ , since the pseudoinverse of the Fisher information cancels out the effect of the state distribution in NPG. We now provide a dimension free convergence rate of this algorithm.

<!-- formula-not-decoded -->

Theorem 5.3 (Global convergence for NPG) . Suppose we run the NPG updates (15) using ρ ∈ ∆( S ) and with θ (0) = 0 . Fix η &gt; 0 . For all T &gt; 0 , we have:

<!-- formula-not-decoded -->

In particular, setting η ≥ (1 -γ ) 2 log |A| , we see that NPG finds an /epsilon1 -optimal policy in a number of iterations that is at most:

<!-- formula-not-decoded -->

6 Specifically, we restrict the MDP to the set of states { s ∈ S : ∃ π such that d π ρ ( s ) &gt; 0 } .

which has no dependence on the number of states or actions, despite the non-concavity of the underlying optimization problem.

The proof strategy we take borrows ideas from the online regret framework in changing MDPs (in [Even-Dar et al., 2009]); here, we provide a faster rate of convergence than the analysis implied by Even-Dar et al. [2009] or by Geist et al. [2019]. We also note that while this proof is obtained for the NPG updates, it is known in the literature that in the limit of small stepsizes, NPG and TRPO updates are closely related (e.g. see Schulman et al. [2015], Neu et al. [2017], Rajeswaran et al. [2017]).

First, the following improvement lemma is helpful:

Lemma 5.2 (Improvement lower bound for NPG) . For the iterates π ( t ) generated by the NPG updates (15) , we have for all starting state distributions µ

<!-- formula-not-decoded -->

Proof: First, let us show that log Z t ( s ) ≥ 0 . To see this, observe:

<!-- formula-not-decoded -->

where the inequality follows by Jensen's inequality on the concave function log x and the final equality uses ∑ a π ( t ) ( a | s ) A ( t ) ( s, a ) = 0 . Using d ( t +1) as shorthand for d ( t +1) µ , the performance difference lemma implies:

<!-- formula-not-decoded -->

where the last step uses that d ( t +1) = d ( t +1) µ ≥ (1 -γ ) µ , componentwise (by (4)), and that log Z t ( s ) ≥ 0 .

With this lemma, we now prove Theorem 5.3.

Proof: [ of Theorem 5.3 ] Since ρ is fixed, we use d /star as shorthand for d π /star ρ ; we also use π s as

shorthand for the vector of π ( ·| s ) . By the performance difference lemma (Lemma 3.2), where we have used the closed form of our updates from Lemma 5.1 in the second step. By applying Lemma 5.2 with d /star as the starting state distribution, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which gives us a bound on E /star log Z ( s ) .

Using the above equation and that V ( t +1) ( ρ ) ≥ V ( t ) ( ρ ) (as V ( t +1) ( s ) ≥ V ( t ) ( s ) for all states s by Lemma 5.2), we have:

s ∼ d t

<!-- formula-not-decoded -->

The proof is completed using that V ( T ) ( ρ ) ≥ V ( T -1) ( ρ ) .

## 6 Function Approximation and Distribution Shift

We now analyze the case of using parametric policy classes:

<!-- formula-not-decoded -->

where Π may not contain all stochastic policies (and it may not even contain an optimal policy). In contrast with the tabular results in the previous sections, the policy classes that we are often interested in are not fully expressive, e.g. d /lessmuch |S||A| (indeed |S| or |A| need not even be finite for the results in this section); in this sense, we are in the regime of function approximation.

We focus on obtaining agnostic results, where we seek to do as well as the best policy in this class (or as well as some other comparator policy). While we are interested in a solution to the (unconstrained) policy optimization problem

<!-- formula-not-decoded -->

(for a given initial distribution ρ ), we will see that optimization with respect to a different distribution will be helpful, just as in the tabular case,

We will consider variants of the NPG update rule (15):

<!-- formula-not-decoded -->

Our analysis will leverage a close connection between the NPG update rule (15) with the notion of compatible function approximation [Sutton et al., 1999], as formalized in Kakade [2001]. Specifically, it can be easily seen that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where w /star is a minimizer of the following regression problem:

The above is a straightforward consequence of the first order optimality conditions (see (50)). The above regression problem can be viewed as 'compatible' function approximation: we are approximating A π θ ( s, a ) using the ∇ θ log π θ ( ·| s ) as features. We also consider a variant of the above update rule, Q -NPG, where instead of using advantages in the above regression we use the Q -values.

This viewpoint provides a methodology for approximate updates, where we can solve the relevant regression problems with samples. Our main results establish the effectiveness of NPG updates where there is error both due to statistical estimation (where we may not use exact gradients) and approximation (due to using a parameterized function class); in particular, we provide a novel estimation/approximation decomposition relevant for the NPG algorithm. For these algorithms, we will first consider log linear policies classes (as a special case) and then move on to more general policy classes (such as neural policy classes). Finally, it is worth remarking that the results herein provide one of the first provable approximation guarantees where the error conditions required do not have explicit worst case dependencies over the state space.

## 6.1 NPG and Q -NPG Examples

In practice, the most common policy classes are of the form:

<!-- formula-not-decoded -->

where f θ is a differentiable function. For example, the tabular softmax policy class is one where f θ ( s, a ) = θ s,a . Typically, f θ is either a linear function or a neural network. Let us consider the NPG algorithm, and a variant Q -NPG, in each of these two cases.

## 6.1.1 Log-linear Policy Classes and Soft Policy Iteration

For any state-action pair ( s, a ) , suppose we have a feature mapping φ s,a ∈ R d . Each policy in the log-linear policy class is of the form:

<!-- formula-not-decoded -->

With regards to compatible function approximation for the log-linear policy class, we have:

with θ ∈ R d . Here, we can take f θ ( s, a ) = θ · φ s,a .

<!-- formula-not-decoded -->

that is, φ θ s,a is the centered version of φ s,a . With some abuse of notation, we accordingly also define ¯ φ π for any policy π . Here, using (17), the NPG update rule (16) is equivalent to:

(We have rescaled the learning rate η in comparison to (16)). Note that we recompute w /star for every update of θ . Here, the compatible function approximation error measures the expressivity of our parameterization in how well linear functions of the parameterization can capture the policy's advantage function.

<!-- formula-not-decoded -->

We also consider a variant of the NPG update rule (16), termed Q -NPG , where:

Note we do not center the features for Q -NPG; observe that Q π ( s, a ) is also not 0 in expectation under π ( ·| s ) , unlike the advantage function.

<!-- formula-not-decoded -->

Remark 6.1. (NPG/ Q -NPG and Soft-Policy Iteration) We now see how we can view both NPG and Q -NPG as an incremental (soft) version of policy iteration, just as in Lemma 5.1 for the tabular case. Rather than writing the update rule in terms of the parameter θ , we can write an equivalent update rule directly in terms of the (log-linear) policy π :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Z s is normalization constant. While the policy update uses the original features φ instead of φ π , whereas the quadratic error minimization is terms of the centered features φ π , this distinction is not relevant due to that we may also instead use φ π (in the policy update) which would result in an equivalent update; the normalization makes the update invariant to (constant) translations of the features. Similarly, an equivalent update for Q -NPG, where we update π directly rather than θ , is:

Remark 6.2. (On the equivalence of NPG and Q -NPG) If it is the case that the compatible function approximation error is 0 , then it straightforward to verify that the NPG and Q -NPG are equivalent algorithms, in that their corresponding policy updates will be equivalent to each other.

## 6.1.2 Neural Policy Classes

Now suppose f θ ( s, a ) is a neural network parameterized by θ ∈ R d , where the policy class Π is of form in (18). Observe:

<!-- formula-not-decoded -->

and, using (17), the NPG update rule (16) is equivalent to:

<!-- formula-not-decoded -->

(Again, we have rescaled the learning rate η in comparison to (16)).

The Q -NPG variant of this update rule is:

<!-- formula-not-decoded -->

## 6.2 Q -NPG: Performance Bounds for Log-Linear Policies

For a state-action distribution υ , define:

<!-- formula-not-decoded -->

The iterates of the Q -NPG algorithm can be viewed as minimizing this loss under some (changing) distribution υ .

We now specify an approximate version of Q -NPG. It is helpful to consider a slightly more general version of the algorithm in the previous section, where instead of optimizing under a starting state distribution ρ , we have a different starting state-action distribution ν . Analogous to the definition of the state visitation measure, d π µ , we can define a visitation measure over states and actions induced by following π after s 0 , a 0 ∼ ν . We overload notation using d π ν to also refer to the state-action visitation measure; precisely,

<!-- formula-not-decoded -->

where Pr π ( s t = s, a t = a | s 0 , a 0 ) is the probability that s t = s and a t = a , after starting at state s 0 , taking action a 0 , and following π thereafter. While we overload notation for visitation distributions ( d π µ ( s ) and d π ν ( s, a ) ) for notational convenience, note that the state-action measure d π ν uses the subscript ν , which is a state-action measure.

- Q -NPG will be defined with respect to the on-policy state action measure starting with s 0 , a 0 ∼ ν . As per our convention, we define

<!-- formula-not-decoded -->

The approximate version of this algorithm is:

<!-- formula-not-decoded -->

where the above update rule also permits us to constrain the norm of the update direction w ( t ) (alternatively, we could use /lscript 2 regularization as is also common in practice). The exact minimizer is denoted as:

<!-- formula-not-decoded -->

Note that w ( t ) /star depends on the current parameter θ ( t ) .

Our analysis will take into account both the excess risk (often also referred to as estimation error) and the transfer error . Here, the excess risk will be due to that w ( t ) may not be equal w ( t ) /star , and the approximation error will be due to that even the best linear fit using w ( t ) /star may not perfectly match the Q -values, i.e. L ( w ( t ) /star ; θ ( t ) ; d ( t ) ) is unlikely to be 0 in practical applications.

We now formalize these concepts in the following assumption:

Assumption 6.1 (Estimation/Transfer errors) . Fix a state distribution ρ ; a state-action distribution ν ; an arbitrary comparator policy π /star (not necessarily an optimal policy). With respect to π /star , define the state-action measure d /star as

<!-- formula-not-decoded -->

i.e. d /star samples states from the comparators state visitation measure, d π /star ρ and actions from the uniform distribution. Let us permit the sequence of iterates w (0) , w (1) , . . . w ( T -1) used by the Q -NPG algorithm to be random, where the randomness could be due to sample-based, estimation error. Suppose the following holds for all t &lt; T :

1. ( Excess risk ) Assume that the estimation error is bounded as follows:

<!-- formula-not-decoded -->

Note that using a sample based approach we would expect /epsilon1 stat = O (1 / √ N ) or better, where N is the number of samples used to estimate. w ( t ) /star We formalize this in Corollary 6.2.

2. ( Transfer error ) Suppose that the best predictor w ( t ) /star has an error bounded by /epsilon1 bias , in expectation, with respect to the comparator's measure of d ∗ . Specifically, assume:

<!-- formula-not-decoded -->

We refer to /epsilon1 bias as the transfer error (or transfer bias ); it is the error where relevant distribution is shifted to d /star . For the softmax policy parameterization for tabular MDPs, /epsilon1 bias = 0 (see remark 6.4 for another example).

In both conditions, the expectations are with respect to the randomness in the sequence of iterates w (0) , w (1) , . . . w ( T -1) , e.g. the approximate algorithm may be sample based.

Shortly, we discuss how the transfer error relates to the more standard approximation-estimation decomposition. Importantly, with the transfer error, it is always defined with respect to a single, fixed measure, d /star .

Assumption 6.2 (Relative condition number) . Consider the same ρ , ν , and π /star as in Assumption 6.1. With respect to any state-action distribution υ , define:

and define

Assume that κ is finite.

Remark 6.3 discusses why it is reasonable to expect that κ is not a quantity related to the size of the state space. 7

Our main theorem below shows how the approximation error, the excess risk, and the conditioning, determine the final performance. Note that both the transfer error /epsilon1 bias and κ are defined with respect to the comparator policy π /star .

Theorem 6.1. (Agnostic learning with Q -NPG) Fix a state distribution ρ ; a state-action distribution ν ; an arbitrary comparator policy π /star (not necessarily an optimal policy). Suppose Assumption 6.2 holds and ‖ φ s,a ‖ 2 ≤ B for all s, a . Suppose the Q -NPG update rule (in (20) ) starts with θ (0) = 0 , η = √ 2 log |A| / ( B 2 W 2 T ) , and the (random) sequence of iterates satisfies Assumption 6.1. We have that

<!-- formula-not-decoded -->

The proof is provided in Section 6.4.

Note when /epsilon1 bias = 0 , our convergence rate is O ( √ 1 /T ) plus a term that depends on the excess risk; hence, provided we obtain enough samples, then /epsilon1 stat will also tend to 0 , and we will be competitive with the comparison policy π /star . When /epsilon1 bias = 0 and /epsilon1 stat = 0 , as in the tabular setting with exact gradients, the additional two terms become 0 , consistent with Theorem 5.3 except that the convergence rate is O ( √ 1 /T ) rather than the faster rate of O (1 /T ) . Obtaining a faster rate in the function approximation regime appears to require stronger conditions on how the approximation errors are controlled at each iteration.

7 Technically, we only need the relative condition number sup w ∈ R d w /latticetop Σ d /star w w /latticetop Σ π ( t ) w to be bounded for all t . We state this as a sufficient condition based on the initial distribution ν due to: this is more interpretable, and, as per Remark 6.3, this quantity can be bounded in a manner that is independent of the sequence of iterates produced by the algorithm.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The usual approximation-estimation error decomposition is that we can write our error as:

<!-- formula-not-decoded -->

As we obtain more samples, we can drive the excess risk (the estimation error) to 0 (see Corollary 6.2). The approximation error above is due to modeling error. Importantly, for our Q -NPG performance bound, it is not this standard approximation error notion which is relevant, but it is this error under a different measure d /star , i.e. L ( w ( t ) /star ; θ ( t ) , d /star ) . One appealing aspect about the transfer error is that this error is with respect to a fixed measure, namely d /star . Furthermore, in practice, modern machine learning methods often performs favorably with regards to transfer learning, substantially better than worst case theory might suggest.

The following corollary provides a performance bound in terms of the usual notion of approximation error, at the cost of also depending on the worst case distribution mismatch ratio. The corollary disentangles the estimation error from the approximation error.

Corollary 6.1. (Estimation error/Approximation error bound for Q -NPG) Consider the same setting as in Theorem 6.1. Rather than assuming the transfer error is bounded (part 2 in Assumption 6.1), suppose that, for all t ≤ T ,

We have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above also shows the striking difference between the effects of estimation error and approximation error. The proof shows how the transfer error notion is weaker than previous conditions based on distribution mistmatch coefficients or concentrability coefficients. Also, as discussed in Scherrer [2014], the (distribution mismatch) coefficient ∥ ∥ ∥ d /star ν ∥ ∥ ∥ ∞ is already weaker than the more standard concentrability coefficients.

where the last step uses the defintion of d ( t ) (see (19)). This implies /epsilon1 bias ≤ 1 1 -γ ∥ ∥ ∥ d /star ν ∥ ∥ ∥ ∞ /epsilon1 approx , and the corollary follows.

A few additional remarks are now in order. We now make a few observations with regards to κ .

Remark 6.3. (Dimension dependence in κ and the importance of ν ) It is reasonable to think about κ as being dimension dependent (or worse), but it is not necessarily related to the size of the state

space. For example, if ‖ φ s,a ‖ 2 ≤ B , then κ ≤ B 2 σ min ( E s,a ∼ ν [ φ s,a φ /latticetop s,a ]) though this bound may be pessimistic. Here, we also see the importance of choice of ν in having a small (relative) condition number; in particular, this is the motivation for considering the generalization which allows for a starting state-action distribution ν vs. just a starting state distribution µ (as we did in the tabular case). Roughly speaking, we desire a ν which provides good coverage over the features. As the following lemma shows, there always exists a universal distribution ν , which can be constructed only with knowledge of the feature set (without knowledge of d /star ), such that κ ≤ d .

Lemma 6.1. ( κ ≤ d is always possible) Let Φ = { φ ( s, a ) | ( s, a ) ∈ S × A} ⊂ R d and suppose Φ is a compact set. There always exists a state-action distribution ν , which is supported on at most d 2 state-action pairs and which can be constructed only with knowledge of Φ (without knowledge of the MDP or d /star ), such that:

<!-- formula-not-decoded -->

Proof: The distribution can be found through constructing the minimal volume ellipsoid containing Φ , i.e. the Lo ¨ wner-John ellipsoid [John, 1948]. In particular, this ν is supported on the contact points between this ellipsoid and Φ ; the lemma immediately follows from properties of this ellipsoid (e.g. see Ball [1997], Bubeck et al. [2012]).

It is also worth considering a more general example (beyond tabular MDPs) in which /epsilon1 bias = 0 for the log-linear policy class.

Remark 6.4. ( /epsilon1 bias = 0 for 'linear' MDPs) In the recent linear MDP model of Jin et al. [2019], Yang and Wang [2019], Jiang et al. [2017], where the transition dynamics are low rank, we have that /epsilon1 bias = 0 provided we use the features of the linear MDP. Our guarantees also permit model misspecification of linear MDPs, with non worst-case approximation error where /epsilon1 bias = 0 .

/negationslash

Remark 6.5. (Comparison with POLITEX and EE-POLITEX) Compared with POLITEX [Abbasi-Yadkori et al., 2019a], Assumption 6.2 is substantially milder, in that it just assumes a good relative condition number for one policy rather than all possible policies (which cannot hold in general even for tabular MDPs). Changing this assumption to an analog of Assumption 6.2 is the main improvement in the analysis of the EE-POLITEX [Abbasi-Yadkori et al., 2019b] algorithm. They provide a regret bound for the average reward setting, which is qualitatively different from the suboptimality bound in the discounted setting that we study. They provide a specialized result for linear function approximation, similar to Theorem 6.1.

## 6.2.1 Q -NPG Sample Complexity

Assumption 6.3 (Episodic Sampling Oracle) . For a fixed state-action distribution ν , we assume the ability to: start at s 0 , a 0 ∼ ν ; continue to act thereafter in the MDP according to any policy π ; and terminate this 'rollout' when desired. With this oracle, it is straightforward to obtain unbiased samples of Q π ( s, a ) (or A π ( s, a ) ) under s, a ∼ d π ν for any π ; see Algorithms 1 and 3.

Algorithm 2 provides a sample based version of the Q -NPG algorithm; it simply uses stochastic projected gradient ascent within each iteration. The following corollary shows this algorithm suffices to obtain an accurate sample based version of Q -NPG.

Algorithm 1 Sampler for: s, a ∼ d π ν and unbiased estimate of Q π ( s, a )

Require: Starting state-action distribution ν .

- 1: Sample s 0 , a 0 ∼ ν .
- 2: Sample s, a ∼ d π ν as follows: at every timestep h , with probability γ , act according to π ; else, accept ( s h , a h ) as the sample and proceed to Step 4. See (19).
- 3: From s h , a h , continue to execute π , and use a termination probability of 1 -γ . Upon termination, set ̂ Q π ( s h , a h ) as the undiscounted sum of rewards from time h onwards.

Corollary 6.2. (Sample complexity of Q -NPG) Assume we are in the setting of Theorem 6.1 and that we have access to an episodic sampling oracle (i.e. Assumption 6.3). Suppose that the Sample Based Q -NPG Algorithm (Algorithm 2) is run for T iterations, with N gradient steps per iteration, with an appropriate setting of the learning rates η and α . We have that:

- 4: return ( s h , a h ) and ̂ Q π ( s h , a h ) .

<!-- formula-not-decoded -->

Furthermore, since each episode has expected length 2 / (1 -γ ) , the expected number of total samples used by Q -NPG is 2 NT/ (1 -γ ) .

Proof: Note that our sampled gradients are bounded by G := 2 B ( BW + 1 1 -γ ) . Using α = W G √ N , a standard analysis for stochastic projected gradient ascent (Theorem E.3) shows that:

<!-- formula-not-decoded -->

The proof is completed via substitution.

Remark 6.6. (Improving the scaling with N ) Our current rate of convergence is 1 /N 1 / 4 due to our use of stochastic projected gradient ascent. Instead, for the least squares estimator, /epsilon1 stat would be O ( d/N ) provided certain further regularity assumptions hold (a bound on the minimal eigenvalue of Σ ν would be sufficient but not necessary. See Hsu et al. [2014] for such conditions). With such further assumptions, our rate of convergence would be O (1 / √ N ) .

## 6.3 NPG: Performance Bounds for Smooth Policy Classes

We now return to the analyzing the standard NPG update rule, which uses advantages rather than Q -values (see Section 6.1). It is helpful to define

<!-- formula-not-decoded -->

## Algorithm 2 Sample-based Q -NPG for Log-linear Policies

Require: Learning rate η ; SGD learning rate α ; number of SGD iterations N

- 1: Initialize θ (0) = 0 .
- 2: for t = 0 , 1 , . . . , T -1 do
- 3: Initialize w 0 = 0
- 4: for n = 0 , 1 , . . . , N -1 do

<!-- formula-not-decoded -->

- 5: Call Algorithm 1 to obtain s, a ∼ d ( t ) and an unbiased estimate ̂ Q ( s, a ) . 6: Update:

where W = { w : ‖ w ‖ 2 ≤ W } .

- 7: end for
- ̂ 9: Update θ ( t +1) = θ ( t ) + η ̂ w ( t ) . 10: end for
- 8: Set w ( t ) = 1 N ∑ N n =1 w n .

where υ is state-action distribution, and the subscript of A denotes the loss function uses advantages (rather than Q -values). The iterates of the NPG algorithm can be viewed as minimizing this loss under some appropriately chosen measure.

We now consider an approximate version of the NPG update rule:

<!-- formula-not-decoded -->

where again we use the on-policy, fitting distribution d ( t ) . As with Q -NPG, we also permit the use of a starting state-action distribution ν as opposed to just a starting state distribution (see Remark 6.3). Again, we let w ( t ) /star denote the minimizer, i.e. w ( t ) /star ∈ argmin ‖ w ‖ 2 ≤ W L A ( w ; θ ( t ) , d ( t ) ) .

For this section, our analysis will focus on more general policy classes, beyond log-linear policy classes. In particular, we make the following smoothness assumption on the policy class:

Assumption 6.4. (Policy Smoothness) Assume for all s ∈ S and a ∈ A that log π θ ( a | s ) is a β -smooth function of θ (to recall the definition of smoothness, see (24) ).

It is not to difficult to verify that the tabular softmax policy parameterization is a 1 -smooth policy class in the above sense. The more general class of log-linear policies is also smooth as we remark below.

Remark 6.7. (Smoothness of the log-linear policy class) For the log-linear policy class (see Section 6.1.1), smoothness is implied if the features φ have bounded Euclidean norm. Precisely, if the feature mapping φ satisfies ‖ φ s,a ‖ 2 ≤ B , then it is not difficult to verify that log π θ ( a | s ) is a B 2 -smooth function.

For any state-action distribution υ , define:

<!-- formula-not-decoded -->

and, again, we use Σ ( t ) υ as shorthand for Σ θ ( t ) υ .

Assumption 6.5. (Estimation/Transfer/Conditioning) Fix a state distribution ρ ; a state-action distribution ν ; an arbitrary comparator policy π /star (not necessarily an optimal policy). With respect to π /star , define the state-action measure d /star as

<!-- formula-not-decoded -->

Note that, in comparison to Assumption 6.1, d /star is the state-action visitation measure of the comparator policy. Let us permit the sequence of iterates w (0) , w (1) , . . . w ( T -1) used by the NPG algorithm to be random, where the randomness could be due to sample-based, estimation error. Suppose the following holds for all t &lt; T :

1. (Excess risk) Assume the estimation error is bounded as:

i.e. the above conditional expectation is bounded (with probability one). 8 As we see in Corollary 6.2, we can guarantee /epsilon1 stat to drop as √ 1 /N . 2. (Transfer error) Suppose that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

3. (Relative condition number) For all iterations t , assume the average relative condition number is bounded as follows:

Note that term inside the expectation is a random quantity as θ ( t ) is random.

In the above conditions, the expectation is with respect to the randomness in the sequence of iterates w (0) , w (1) , . . . w ( T -1) .

Analogous to our Q -NPG theorem, our main theorem for NPG shows how the transfer error is relevant in addition the statistical error /epsilon1 stat .

Theorem 6.2. (Agnostic learning with NPG) Fix a state distribution ρ ; a state-action distribution ν ; an arbitrary comparator policy π /star (not necessarily an optimal policy). Suppose Assumption 6.4 holds. Suppose the NPG update rule (in (21) ) starts with π (0) being the uniform distribution (at each state), η = √ 2 log |A| / ( βW 2 T ) , and the (random) sequence of iterates satisfies Assumption 6.5. We have that

8 The use of a conditional expectation here (vs. the unconditional one in Assumption 6.1) permits the assumption to hold even in settings where we may reuse data in the sample-based approximation of L A . Also, the expectation over the iterates allows a more natural assumption on the relative condition number, relevant for the more general case of smooth policies.

<!-- formula-not-decoded -->

The proof is provided in Section 6.4.

Remark 6.8. (The |A| dependence: NPG vs. Q -NPG) Observe there is no polynomial dependence on |A| in the rate for NPG (in constrast to Theorem 6.1); also observe that here we define d /star as the state-action distribution of π /star in Assumption 6.5, as opposed to a uniform distribution over the actions, as in Assumption 6.1. The main difference arises in the analysis in that, even for Q -NPG, we need to bound the error in fitting the advantage estimates; this leads to the dependence on |A| (which can be removed with a path dependent bound, i.e. a bound which depends on the sequence of iterates produced by the algorithm) 9 . For NPG, the direct fitting of the advantage function sidesteps this conversion step. Note that the relative condition number assumption in Q -NPG (Assumption 6.2) is a weaker assumption, due to that it can be bounded independently of the path of the algorithm (see Remark 6.2), while NPG's centering of the features makes the assumption on the relative condition number depend on the path of the algorithm.

Remark 6.9. (Generalizing Q -NPG for smooth policies) A similar reasoning as the analysis here can be also used to establish a convergence result for the Q -NPG algorithm in this more general setting of smooth policy classes. Concretely, we can analyze the Q -NPG update described for neural policy classes in Section 6.1.2, assuming that the function f θ is Lipschitz-continuous in θ . Like for Theorem 6.2, the main modification is that Assumption 6.2 on relative condition numbers is now defined using the covariance matrix for the features f θ ( s, a ) , which depend on θ , as opposed to some a feature map φ ( s, a ) in the log-linear case. The rest of the analysis follows with an appropriate adaptation of the results above.

## 6.3.1 NPG Sample Complexity

Algorithm 4 provides a sample based version of the NPG algorithm, again using stochastic projected gradient ascent; it uses a slight modification of the Q -NPG algorithm to obtain unbiased gradient estimates. The following corollary shows that this algorithm provides an accurate sample based version of NPG.

Corollary 6.3. (Sample complexity of NPG) Assume we are in the setting of Theorem 6.2 and that we have access to an episodic sampling oracle (i.e. Assumption 6.3). Suppose that the Sample Based NPG Algorithm (Algorithm 4) is run for T iterations, with N gradient steps per iteration. Also, suppose that ‖∇ θ log π ( t ) ( a | s ) ‖ 2 ≤ B holds with probability one. There exists a setting of η and α such that:

<!-- formula-not-decoded -->

Furthermore, since each episode has expected length 2 / (1 -γ ) , the expected number of total samples used by NPG is 2 NT/ (1 -γ ) .

9 For Q -NPG, we have to bound two distribution shift terms to both π /star and π ( t ) at step t of the algorithm.

Algorithm 3 Sampler for: s, a ∼ d π ν and unbiased estimate of A π ( s, a )

Require: Starting state-action distribution ν .

- 2: Start at state s 0 ∼ ν . Sample a 0 ∼ ν ( ·| s 0 ) (though do not necessarily execute a 0 ).
- 1: Set ̂ Q π = 0 and ̂ V π = 0 .
- With probability γ , execute a h , transition to s h +1 , and sample a h +1 ∼ π ( ·| s h +1 ) .
- 3: ( d π ν sampling) At every timestep h ≥ 0 ,
- Else accept ( s h , a h ) as the sample and proceed to Step 4.
- 4: ( A π ( s, a ) sampling) Set SampleQ = True with probability 1 / 2 .
- If SampleQ = True, execute a h at state s h and then continue executing π with a termination probability of 1 -γ . Upon termination, set ̂ Q π as the undiscounted sum of rewards from time h onwards.
- 5: return ( s h , a h ) and ̂ A π ( s h , a h ) = 2( ̂ Q π -̂ V π ) .
- Else sample a ′ h ∼ π ( ·| s h ) . Then execute a ′ h at state s h and then continue executing π with a termination probability of 1 -γ . Upon termination, set ̂ V π as the undiscounted sum of rewards from time h onwards.

Proof: Let us see that the update direction in Step 6 of Algorithm 4 uses an unbiased estimate of the true gradient of the loss function L A :

<!-- formula-not-decoded -->

where the last step follows due to that sampling procedure in Algorithm 3 produces a conditionally unbiased estimate.

Since ‖∇ θ log π ( t ) ( a | s ) ‖ 2 ≤ B and since ̂ A ( s, a ) ≤ 2 / (1 -γ ) , our sampled gradients are bounded by G := 8 B ( BW + 1 1 -γ ) . The remainder of the proof follows that of Corollary 6.2

## 6.4 Analysis

We first proceed by providing a general analysis of NPG, for arbitrary sequences. We then specialize it to complete the proof of our two main theorems in this section.

## Algorithm 4 Sample-based NPG

Require: Learning rate η ; SGD learning rate α ; number of SGD iterations N

- 1: Initialize θ (0) = 0 .
- 2: for t = 0 , 1 , . . . , T -1 do
- 3: Initialize w 0 = 0
- 4: for n = 0 , 1 , . . . , N -1 do
- 5: Call Algorithm 3 to obtain s, a d ( t ) , and an unbiased estimate A ( s, a ) of A ( t ) ( s, a ) .
- 6:

<!-- formula-not-decoded -->

- ∼ ̂ Update:

<!-- formula-not-decoded -->

- 7: end for
- 8: Set ̂ w ( t ) = 1 N ∑ N n =1 w n . 9: Update θ ( t +1) = θ ( t ) + η ̂ w ( t ) . 10: end for

## 6.4.1 The NPG 'Regret Lemma'

It is helpful for us to consider NPG more abstractly, as an update rule of the form

<!-- formula-not-decoded -->

Wewill now provide a lemma where w ( t ) is an arbitrary (bounded) sequence, which will be helpful when specialized.

Recall a function f : R d → R is said to be β -smooth if for all x, x ′ ∈ R d :

<!-- formula-not-decoded -->

and, due to Taylor's theorem, recall that this implies:

<!-- formula-not-decoded -->

The following analysis of NPG is based on the mirror-descent approach developed in [Even-Dar et al., 2009], which motivates us to refer to it as a 'regret lemma'.

Lemma 6.2. (NPG Regret Lemma) Fix a comparison policy ˜ π and a state distribution ρ . Assume for all s ∈ S and a ∈ A that log π θ ( a | s ) is a β -smooth function of θ . Consider the update rule (23) , where π (0) is the uniform distribution (for all states) and where the sequence of weights w (0) , . . . , w ( T ) , satisfies ‖ w ( t ) ‖ 2 ≤ W (but is otherwise arbitrary). Define:

<!-- formula-not-decoded -->

We have that:

Proof: By smoothness (see (24)),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We use ˜ d as shorthand for d ˜ π ρ (note ρ and ˜ π are fixed); for any policy π , we also use π s as shorthand for the vector π ( ·| s ) . Using the performance difference lemma (Lemma 3.2),

<!-- formula-not-decoded -->

Rearranging, we have:

Proceeding,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which completes the proof.

## 6.4.2 Proofs of Theorem 6.1 and 6.2

Proof: (of Theorem 6.1) Using the NPG regret lemma (Lemma 6.2) and the smoothness of the log-linear policy class (see Example 6.7),

<!-- formula-not-decoded -->

where we have used our setting of η .

We make the following decomposition of err t :

For the first term, using that ∇ θ log π θ ( a | s ) = φ s,a -E a ′ ∼ π θ ( ·| s ) [ φ s,a ′ ] (see Section 6.1.1), we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used the definition of d /star and L ( w ( t ) /star ; θ ( t ) , d /star ) in the last step.

For the second term, let us now show that:

<!-- formula-not-decoded -->

To see this, first observe that a similar argument to the above leads to:

<!-- formula-not-decoded -->

where we use the notation ‖ x ‖ 2 M := x /latticetop Mx for a matrix M and a vector x . From the definition of κ ,

<!-- formula-not-decoded -->

using that (1 -γ ) ν ≤ d π ( t ) ν (see (19)). Due to that w ( t ) /star minimizes L ( w ; θ ( t ) , d ( t ) ) over the set W := { w : ‖ w ‖ 2 ≤ W } , for any w ∈ W the first-order optimality conditions for w ( t ) /star imply that:

<!-- formula-not-decoded -->

Therefore, for any w ∈ W ,

<!-- formula-not-decoded -->

Noting that w ( t ) ∈ W by construction in Algorithm 4 yields the claimed bound on the second term in (26).

Using the bounds on the first and second terms in (25) and (26), along with concavity of the square root function, we have that:

<!-- formula-not-decoded -->

The proof is completed by substitution and using our assumptions on /epsilon1 stat and /epsilon1 bias .

The following proof for the NPG algorithm follows along similar lines.

Proof: (of Theorem 6.2) Using the NPG regret lemma and our setting of η ,

<!-- formula-not-decoded -->

where the expectation is with respect to the sequence of iterates w (0) , w (1) , . . . w ( T -1) . Again, we make the following decomposition of err t :

For the first term,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used the definition of L A ( w ( t ) /star ; θ ( t ) , d /star ) in the last step.

For the second term, a similar argument leads to:

<!-- formula-not-decoded -->

Define κ ( t ) := ‖ (Σ ( t ) ν ) -1 / 2 Σ d /star (Σ ( t ) ν ) -1 / 2 ‖ 2 , which is the relative condition number at iteration t . We have

<!-- formula-not-decoded -->

where the last step uses that w ( t ) /star is a minimizer of L A over W and that w ( t ) is feasible as before (see the proof of Theorem 6.1). Now taking an expectation we have:

<!-- formula-not-decoded -->

where we have used our assumption on κ and /epsilon1 stat .

The proof is completed by substitution and using the concavity of the square root function.

## 7 Discussion

This work provides a systematic study of the convergence properties of policy optimization techniques, both in the tabular and the function approximation settings. At the core, our results imply that the non-convexity of the policy optimization problem is not the fundamental challenge for typical variants of the policy gradient approach. This is evidenced by the global convergence results which we establish and that demonstrate the relative niceness of the underlying optimization problem. At the same time, our results highlight that insufficient exploration can lead to the convergence to sub-optimal policies, as is also observed in practice; technically, we show how this is an issue of conditioning. Conversely, we can expect typical policy gradient algorithms to find the best policy from amongst those whose state-visitation distribution is adequately aligned with the policies we discover, provided a distribution-shifted notion of approximation error is small.

In the tabular case, our results show that the nature and severity of the exploration/distribution mismatch term differs in different policy optimization approaches. For instance, we find that doing policy gradient in its standard form for both the direct and softmax parameterizations can be

slow to converge, particularly in the face of distribution mismatch, even when policy gradients are computed exactly. Natural policy gradient, on the other hand, enjoys a fast dimension-free convergence when we are in tabular settings with exact gradients. On the other hand, for the function approximation setting, or when using finite samples, all algorithms suffer to some degree from the exploration issue captured through a conditioning effect.

With regards to function approximation, the guarantees herein are the first provable results that permit average case approximation errors, where the guarantees do not have explicit worst case dependencies over the state space. These worst case dependencies are avoided by precisely characterizing an approximation/estimation error decomposition, where the relevant approximation error is under distribution shift to an optimal policies measure. Here, we see that successful function approximation relies on two key aspects: good conditioning (related to exploration) and low distribution-shifted, approximation error. In particular, these results identify the relevant measure of the expressivity of a policy class, for the natural policy gradient.

With regards to sample size issues, we showed that simply using stochastic (projected) gradient ascent suffices for accurate policy optimization. However, in terms of improving sample efficiency and polynomial dependencies, there are number of important questions for future research, including variance reduction techniques along with data re-use.

There are number of compelling directions for further study. The first is in understanding how to remove the density ratio guarantees among prior algorithms; our results are suggestive that the incremental policy optimization approaches, including CPI [Kakade and Langford, 2002], PSDP [Bagnell et al., 2004], and MD-MPI Geist et al. [2019], may permit such an improved analysis. The question of understanding what representations are robust to distribution shift is wellmotivated by the nature of our distribution-shifted, approximation error (the transfer error). Finally, we hope that policy optimization approaches can be combined with exploration approaches, so that, provably, these approaches can retain their robustness properties (in terms of their agnostic learning guarantees) while mitigating the need for a well conditioned initial starting distribution.

## Acknowledgments

We thank the anonymous reviewers who provided detailed and constructive feedback that helped us significantly improve the presentation and exposition. Sham Kakade and Alekh Agarwal gratefully acknowledge numerous helpful discussions with Wen Sun with regards to the Q -NPG algorithm and our notion of transfer error. We also acknowledge numerous helpful comments from Ching-An Cheng and Andrea Zanette on an earlier draft of this work. We thank Nan Jiang, Bruno Scherrer, and Matthieu Geist for their comments with regards to the relationship between concentrability coefficients, the condition number, and the transfer error; this discussion ultimately lead to Corollary 6.1. Sham Kakade acknowledges funding from the Washington Research Foundation for Innovation in Data-intensive Discovery, the ONR award N00014-18-1-2247, and the DARPA award FA8650-18-2-7836. Jason D. Lee acknowledges support of the ARO under MURI Award W911NF-11-1-0303. This is part of the collaboration between US DOD, UK MOD and UK Engineering and Physical Research Council (EPSRC) under the Multidisciplinary University Research Initiative.

## References

- Yasin Abbasi-Yadkori, Peter Bartlett, Kush Bhatia, Nevena Lazic, Csaba Szepesvari, and Gell´ ert Weisz. POLITEX: Regret bounds for policy iteration using expert prediction. In International Conference on Machine Learning , pages 3692-3702, 2019a.
- Yasin Abbasi-Yadkori, Nevena Lazic, Csaba Szepesvari, and Gellert Weisz. Exploration-enhanced politex. arXiv preprint arXiv:1908.10479 , 2019b.
- Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a posteriori policy optimisation. In International Conference on Learning Representations , 2018.
- Zafarali Ahmed, Nicolas Le Roux, Mohammad Norouzi, and Dale Schuurmans, editors. Understanding the impact of entropy on policy optimization , 2019. URL https://arxiv.org/abs/1811.11214 .
- Andr´ as Antos, Csaba Szepesv´ ari, and R´ emi Munos. Learning near-optimal policies with bellmanresidual minimization based fitted policy iteration and a single sample path. Machine Learning , 71(1):89-129, 2008.
- H´ edy Attouch, J´ erˆ ome Bolte, Patrick Redont, and Antoine Soubeyran. Proximal alternating minimization and projection methods for nonconvex problems: An approach based on the kurdykałojasiewicz inequality. Mathematics of Operations Research , 35(2):438-457, 2010.
- Mohammad Gheshlaghi Azar, Vicenc ¸ G´ omez, and Hilbert J. Kappen. Dynamic policy programming. J. Mach. Learn. Res. , 13(1), November 2012. ISSN 1532-4435.
- J. A. Bagnell, Sham M Kakade, Jeff G. Schneider, and Andrew Y. Ng. Policy search by dynamic programming. In S. Thrun, L. K. Saul, and B. Sch¨ olkopf, editors, Advances in Neural Information Processing Systems 16 , pages 831-838. MIT Press, 2004.
- J. Andrew Bagnell and Jeff Schneider. Covariant policy search. In Proceedings of the 18th International Joint Conference on Artificial Intelligence , IJCAI'03, pages 1019-1024, San Francisco, CA, USA, 2003. Morgan Kaufmann Publishers Inc.
- Keith Ball. An elementary introduction to modern convex geometry. Flavors of geometry , 31: 1-58, 1997.
- A. Beck. First-Order Methods in Optimization . Society for Industrial and Applied Mathematics, Philadelphia, PA, 2017. doi: 10.1137/1.9781611974997.
- Richard Bellman and Stuart Dreyfus. Functional approximations and dynamic programming. Mathematical Tables and Other Aids to Computation , 13(68):247-251, 1959.
- Dimitri P Bertsekas and John N Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, Belmont, MA, 1996.

- Jalaj Bhandari and Daniel Russo. Global optimality guarantees for policy gradient methods. CoRR , abs/1906.01786, 2019. URL http://arxiv.org/abs/1906.01786 .
- Shalabh Bhatnagar, Richard S Sutton, Mohammad Ghavamzadeh, and Mark Lee. Natural actorcritic algorithms. Automatica , 45(11):2471-2482, 2009.
- J´ erˆ ome Bolte, Aris Daniilidis, and Adrian Lewis. The łojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems. SIAM Journal on Optimization , 17(4):1205-1223, 2007.
- Justin A Boyan. Least-squares temporal difference learning. In Proceedings of the Sixteenth International Conference on Machine Learning , pages 49-56. Morgan Kaufmann Publishers Inc., 1999.
- Ronen I Brafman and Moshe Tennenholtz. R-max-a general polynomial time algorithm for nearoptimal reinforcement learning. The Journal of Machine Learning Research , 3:213-231, 2003.
- S´ ebastien Bubeck, Nicol` o Cesa-Bianchi, and Sham M. Kakade. Towards minimax policies for online linear optimization with bandit feedback. In COLT 2012 - The 25th Annual Conference on Learning Theory, June 25-27, 2012, Edinburgh, Scotland , volume 23 of JMLR Proceedings , pages 41.1-41.14, 2012.
- Qi Cai, Zhuoran Yang, Jason D. Lee, and Zhaoran Wang. Neural temporaldifference learning converges to global optima. CoRR , abs/1905.10027, 2019. URL http://arxiv.org/abs/1905.10027 .
- Nicolo Cesa-Bianchi and Gabor Lugosi. Prediction, Learning, and Games . Cambridge University Press, New York, NY, USA, 2006. ISBN 0521841089.
- Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning , pages 1042-1051, 2019.
- Eyal Even-Dar, Sham M Kakade, and Yishay Mansour. Online Markov decision processes. Mathematics of Operations Research , 34(3):726-736, 2009.
- Amir-massoud Farahmand, Csaba Szepesv´ ari, and R´ emi Munos. Error propagation for approximate policy and value iteration. In Advances in Neural Information Processing Systems , pages 568-576, 2010.
- Maryam Fazel, Rong Ge, Sham M Kakade, and Mehran Mesbahi. Global convergence of policy gradient methods for the linear quadratic regulator. arXiv preprint arXiv:1801.05039 , 2018.
- Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences , 55(1):119-139, 1997.
- Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points - online stochastic gradient for tensor decomposition. Proceedings of The 28th Conference on Learning Theory, COLT 2015, Paris, France, July 3-6, 2015 , 2015.

- Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. arXiv preprint arXiv:1901.11275 , 2019.
- Saeed Ghadimi and Guanghui Lan. Accelerated gradient methods for nonconvex nonlinear and stochastic programming. Mathematical Programming , 156(1-2):59-99, 2016.
- Daniel Hsu, Sham M. Kakade, and Tong Zhang. Random design analysis of ridge regression. Foundations of Computational Mathematics , 14(3):569-600, 2014.
- Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low bellman rank are PAC-learnable. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 , pages 1704-1713. JMLR. org, 2017.
- Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M. Kakade, and Michael I. Jordan. How to escape saddle points efficiently. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017 , pages 1724-1732, 2017.
- Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. arXiv preprint arXiv:1907.05388 , 2019.
- Fritz John. Extremum problems with inequalities as subsidiary conditions. Interscience Publishers, 1948.
- S. Kakade. A natural policy gradient. In NIPS , 2001.
- Sham Kakade and John Langford. Approximately Optimal Approximate Reinforcement Learning. In Proceedings of the 19th International Conference on Machine Learning , volume 2, pages 267-274, 2002.
- Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximalgradient methods under the polyak-łojasiewicz condition. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases , pages 795-811. Springer, 2016.
- Michael Kearns and Satinder Singh. Near-optimal reinforcement learning in polynomial time. Machine Learning , 49(2-3):209-232, 2002.
- Vijay R Konda and John N Tsitsiklis. Actor-critic algorithms. In Advances in neural information processing systems , pages 1008-1014, 2000.
- Alessandro Lazaric, Mohammad Ghavamzadeh, and R´ emi Munos. Analysis of classification-based policy iteration algorithms. The Journal of Machine Learning Research , 17(1):583-612, 2016.
- Boyi Liu, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural proximal/trust region policy optimization attains globally optimal policy. CoRR , abs/1906.10306, 2019. URL http://arxiv.org/abs/1906.10306 .

- Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning , pages 1928-1937, 2016.
- R´ emi Munos. Error bounds for approximate policy iteration. In ICML , volume 3, pages 560-567, 2003.
- R´ emi Munos. Error bounds for approximate value iteration. In AAAI , 2005.
- Arkadii Semenovich Nemirovsky and David Borisovich Yudin. Problem complexity and method efficiency in optimization. 1983.
- Yurii Nesterov and Boris T. Polyak. Cubic regularization of newton method and its global performance. Math. Program. , pages 177-205, 2006.
- Gergely Neu, Andras Antos, Andr´ as Gy¨ orgy, and Csaba Szepesv´ ari. Online markov decision processes under bandit feedback. In Advances in Neural Information Processing Systems 23 . Curran Associates, Inc., 2010.
- Gergely Neu, Anders Jonsson, and Vicenc ¸ G´ omez. A unified view of entropy-regularized markov decision processes. CoRR , abs/1705.07798, 2017.
- Jan Peters and Stefan Schaal. Natural actor-critic. Neurocomput. , 71(7-9):1180-1190, 2008. ISSN 0925-2312.
- Jan Peters, Katharina M¨ ulling, and Yasemin Alt¨ un. Relative entropy policy search. In Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence (AAAI 2010) , pages 1607-1612. AAAI Press, 2010.
- B. T. Polyak. Gradient methods for minimizing functionals. USSR Computational Mathematics and Mathematical Physics , 3(4):864-878, 1963.
- Aravind Rajeswaran, Kendall Lowrey, Emanuel V. Todorov, and Sham M Kakade. Towards generalization and simplicity in continuous control. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30 , pages 6550-6561. Curran Associates, Inc., 2017.
- Bruno Scherrer. Approximate policy iteration schemes: A comparison. In Proceedings of the 31st International Conference on International Conference on Machine Learning - Volume 32 , ICML'14. JMLR.org, 2014.
- Bruno Scherrer and Matthieu Geist. Local policy search in a convex space and conservative policy iteration as boosted policy search. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases , pages 35-50. Springer, 2014.
- Bruno Scherrer, Mohammad Ghavamzadeh, Victor Gabillon, Boris Lesner, and Matthieu Geist. Approximate modified policy iteration and its application to the game of tetris. Journal of Machine Learning Research , 16:1629-1676, 2015.

- John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning , pages 1889-1897, 2015.
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- S. Shalev-Shwartz and S. Ben-David. Understanding Machine Learning: From Theory to Algorithms . Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, 2014. ISBN 9781107057135. URL https://books.google.com/books?id=ttJkAwAAQBAJ .
- Shai Shalev-Shwartz et al. Online learning and online convex optimization. Foundations and Trends in Machine Learning , 4(2):107-194, 2012.
- Lior Shani, Yonathan Efroni, and Shie Mannor. Adaptive trust region policy optimization: Global convergence and fa ster rates for regularized mdps, 2019.
- Richard S Sutton, David A McAllester, Satinder P Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems , volume 99, pages 1057-1063, 1999.
- Csaba Szepesv´ ari and R´ emi Munos. Finite time bounds for sampling based fitted value iteration. In Proceedings of the 22nd international conference on Machine learning , pages 880-887. ACM, 2005.
- Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning , 8(3-4):229-256, 1992.
- Ronald J Williams and Jing Peng. Function optimization using connectionist reinforcement learning algorithms. Connection Science , 3(3):241-268, 1991.
- Lin F. Yang and Mengdi Wang. Sample-optimal parametric q-learning using linearly additive features. In International Conference on Machine Learning , pages 6995-7004, 2019.

## A Proofs for Section 3

Proof: [ of Lemma 3.1 ] Recall the MDP in Figure 1. Note that since actions in terminal states s 3 , s 4 and s 5 do not change the expected reward, we only consider actions in states s 1 and s 2 . Let the 'up/above' action as a 1 and 'right' action as a 2 . Note that

<!-- formula-not-decoded -->

Consider

<!-- formula-not-decoded -->

where θ is written as a tuple ( θ a 1 ,s 1 , θ a 2 ,s 1 , θ a 1 ,s 2 , θ a 2 ,s 2 ) . Then, for the softmax parameterization, we have and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Also, for θ ( mid ) = θ (1) + θ (2) 2 ,

<!-- formula-not-decoded -->

This gives

<!-- formula-not-decoded -->

which shows that V π is non-concave.

Proof: [ of Lemma 3.2 ] Let Pr π ( τ | s 0 = s ) denote the probability of observing a trajectory τ when starting in state s and following the policy π . Using a telescoping argument, we have:

where ( a ) rearranges terms in the summation and cancels the V π ′ ( s 0 ) term with the -V π ′ ( s ) outside the summation, and ( b ) uses the tower property of conditional expectations and the final equality follows from the definition of d π s .

<!-- formula-not-decoded -->

## B Proofs for Section 4

## B.1 Proofs for Section 4.2

We first define first-order optimality for constrained optimization.

Definition B.1 (First-order Stationarity) . A policy π θ ∈ ∆( A ) | S | is /epsilon1 -stationary with respect to the initial state distribution µ if

<!-- formula-not-decoded -->

where ∆( A ) | S | is the set of all policies.

Due to that we are working with the direct parameterization (see (2)), we drop the θ subscript.

Remark B.1. If /epsilon1 = 0 , then the definition simplifies to δ /latticetop ∇ π V π ( µ ) ≤ 0 . Geometrically, δ is a feasible direction of movement since the probability simplex ∆( A ) | S | is convex. Thus the gradient is negatively correlated with any feasible direction of movement, and so π is first-order stationary.

Proposition B.1. Let V π ( µ ) be β -smooth in π . Define the gradient mapping

<!-- formula-not-decoded -->

and the update rule for the projected gradient is π + = π + ηG η ( π ) . If ‖ G η ( π ) ‖ 2 ≤ /epsilon1 , then

<!-- formula-not-decoded -->

Proof: By Theorem E.2,

<!-- formula-not-decoded -->

where B 2 is the unit /lscript 2 ball, and N ∆( A ) |S| is the normal cone of the product simplex ∆( A ) |S| . Since ∇ π V π + ( µ ) is /epsilon1 ( ηβ + 1) distance from the normal cone and δ is in the tangent cone, then δ /latticetop ∇ π V π + ( µ ) ≤ /epsilon1 ( ηβ +1) .

Proof: [ of Theorem 4.1 ] Recall the definition of gradient mapping

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Lemma D.3, we have V π ( s ) is β -smooth for all states s (and also hence V π ( µ ) is also β -smooth) with β = 2 γ |A| (1 -γ ) 3 . Then, from standard result (Theorem E.1), we have that for G η ( π ) with step-size η = 1 β ,

Then, from Proposition B.1, we have

<!-- formula-not-decoded -->

Observe that

<!-- formula-not-decoded -->

where the last step follows as ‖ ¯ π -π ‖ 2 ≤ 2 √ |S| . And then using Lemma 4.1 and ηβ = 1 , we have

We can get our required bound of /epsilon1 , if we set T such that

<!-- formula-not-decoded -->

or, equivalently,

<!-- formula-not-decoded -->

∥ ∥ Using V /star ( µ ) -V (0) ( µ ) ≤ 1 1 -γ and β = 2 γ |A| (1 -γ ) 3 from Lemma D.3 leads to the desired result.

<!-- formula-not-decoded -->

## B.2 Proofs for Section 4.3

Recall the MDP in Figure 2. Each trajectory starts from the initial state s 0 , and we use the discount factor γ = H/ ( H +1) . Recall that we work with the direct parameterization, where π θ ( a | s ) = θ s,a for a = a 1 , a 2 , a 3 and π θ ( a 4 | s ) = 1 -θ s,a 1 -θ s,a 2 -θ s,a 3 . Note that since states s 0 and s H +1 only have once action, therefore, we only consider the parameters for states s 1 to s H . For this policy class and MDP, let P θ be the state transition matrix under π θ , i.e. [ P θ ] s,s ′ is the probability of going from state s to s ′ under policy π θ :

<!-- formula-not-decoded -->

For the MDP illustrated in Figure 2, the entries of this matrix are given as:

<!-- formula-not-decoded -->

With this definition, we recall that the value function in the initial state s 0 is given by

<!-- formula-not-decoded -->

where e 0 is an indicator vector for the starting state s 0 . From the form of the transition probabilities (27), it is clear that the value function only depends on the parameters θ s,a 1 in any state s . While care is needed for derivatives as the parameters across actions are related by the simplex feasibility constraints, we have assumed each parameter is strictly positive, so that an infinitesimal change to any parameter other than θ s,a 1 does not affect the policy value and hence the policy gradients. With this understanding, we succinctly refer to θ s,a 1 as θ s in any state s . We also refer to the state s i simply as i to reduce subscripts.

For convenience, we also define ¯ p (resp. p ) to be the largest (resp. smallest) of the probabilities θ s across the states s ∈ [1 , H ] in the MDP.

It is easily checked that V π θ ( s 0 ) = M θ 0 ,H +1 , where

In this section, we prove Proposition 4.1, that is: for 0 &lt; θ &lt; 1 (componentwise across states and actions), ¯ p ≤ 1 / 4 , and for all k ≤ H 40 log(2 H ) -1 , we have ‖∇ k θ V π θ ( s 0 ) ‖ ≤ (1 / 3) H/ 4 , where ∇ k θ V π θ ( s 0 ) is a tensor of the k th order. Furthermore, we seek to show V /star ( s 0 ) -V π θ ( s 0 ) ≥ ( H +1) / 8 -( H +1) 2 / 3 H (where θ /star are the optimal policy's parameters).

<!-- formula-not-decoded -->

since the only rewards are obtained in the state s H +1 . In order to bound the derivatives of the expected reward, we first establish some properties of the matrix M θ .

<!-- formula-not-decoded -->

1. M θ a,b ≤ α b -a -1 1 -γ for 0 ≤ a ≤ b ≤ H .

<!-- formula-not-decoded -->

Proof: Let ρ k a,b be the normalized discounted probability of reaching b , when the initial state is a , in k steps, that is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we recall the convention that U 0 is the identity matrix for any square matrix U . Observe that 0 ≤ ρ k a,b ≤ 1 , and, based on the form (27) of P θ , we have the recursive relation for all k &gt; 0 :

/negationslash

Note that ρ 0 a,b = 0 for a = b and ρ 0 a,b = 1 -γ for a = b . Now let us inductively prove that for all k ≥ 0

<!-- formula-not-decoded -->

/negationslash

Clearly this holds for k = 0 since ρ 0 a,b = 0 for a = b and ρ 0 a,b = 1 -γ for a = b . Now, assuming the bound for all steps till k -1 , we now prove it for k case by case.

<!-- formula-not-decoded -->

For 1 &lt; b &lt; H and a &lt; b , observe that the recursion (29) and the inductive hypothesis imply that

<!-- formula-not-decoded -->

For b = H and a &lt; H , we observe that where the last inequality follows since α 2 γ (1 -p ) -α + γ ¯ p ≤ 0 due to that α is within the roots of this quadratic equation. Note the discriminant term in the square root is non-negative provided ¯ p &lt; 1 / 4 , since the condition along with the knowledge that p ≤ ¯ p ensures that 4 γ 2 ¯ p (1 -p ) ≤ 1 .

<!-- formula-not-decoded -->

This proves the inductive claim (note that the cases of b = a = 1 and b = a = H are already handled in the first part above). Next, we prove that for all k ≥ 0

<!-- formula-not-decoded -->

Clearly this holds for k = 0 and b = 0 since ρ 0 0 ,b = 0 . Furthermore, for all k ≥ 0 and b = 0 ,

<!-- formula-not-decoded -->

since α ≤ 1 by construction and b = 0 . Now, we consider the only remaining case when k &gt; 0 and b ∈ [1 , H +1] . By (27), observe that for k &gt; 0 and b ∈ [1 , H +1] ,

<!-- formula-not-decoded -->

For a = b the result follows since

/negationslash

for all i ≥ 1 . Using the definition of ρ k a,b (28) for k &gt; 0 and b ∈ [1 , H +1] ,

<!-- formula-not-decoded -->

Hence, for all k ≥ 0

In conjunction with Equation (30), the above display gives for all k ≥ 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Also observe that

Since the above bound holds for all k ≥ 0 , it also applies to the limiting value M θ a,b , which shows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which completes the proof of the first part of the lemma.

For the second claim, from recursion (29) and b = H +1 and a &lt; H +1

<!-- formula-not-decoded -->

Taking the limit of k →∞ , we see that

<!-- formula-not-decoded -->

Rearranging the terms in the above bound yields the second claim in the lemma.

Using the lemma above, we now bound the derivatives of M θ .

Lemma B.2. The k th order partial derivatives of M satisfy:

<!-- formula-not-decoded -->

where β denotes a k dimensional vector with entries in { 1 , 2 , . . . , H } .

Proof: Since the parameter θ is fixed throughout, we drop the superscript in M θ for brevity. Using ∇ θ M = -M ∇ θ ( I -γP θ ) M , using the form of P θ in (27), we get for any h ∈ [1 , H ]

<!-- formula-not-decoded -->

where the second equality follows since P h,h +1 = θ h and P h,h -1 = 1 -θ h are the only two entries in the transition matrix which depend on θ h for h ∈ [1 , H ] .

1. | c n | = γ k and N ≤ 2 k k ! ,

Next, let us consider a k th order partial derivative of M 0 ,H +1 , denoted as ∂ k M 0 ,H +1 ∂θ β . Note that β can have repeated entries to capture higher order derivative with respect to some parameter. We prove by induction for all k ≥ 1 , -∂ k M 0 ,H +1 ∂θ β can be written as ∑ N n =1 c n ζ n where

2. Each monomial ζ n is of the form M i 1 ,j 1 . . . M i k +1 ,j k +1 , i 1 = 0 , j k +1 = H + 1 , j l ≤ H and i l +1 = j l ± 1 for all l ∈ [1 , k ] .

The base case k = 1 follows from Equation (32), as we can write for any h ∈ [ H ]

<!-- formula-not-decoded -->

Clearly, the induction hypothesis is true with | c n | = γ , N = 2 , i 1 = 0 , j 2 = H +1 , j 1 ≤ H and i 2 = j 1 ± 1 . Now, suppose the claim holds till k -1 . Then by the chain rule:

<!-- formula-not-decoded -->

where β /i is the vector β with the i th entry removed. By inductive hypothesis,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

2. Each monomial ζ n is of the form M i 1 ,j 1 . . . M i k ,j k , i 1 = 0 , j k = H + 1 , j l ≤ H and i l +1 = j l ± 1 for all l ∈ [1 , k -1] .

In order to compute the ( k ) th derivative of M 0 ,H +1 , we have to compute derivative of each monomial ζ n with respect to θ β 1 . Consider one of the monomials in the ( k -1) th derivative, say, ζ = M i 1 ,j 1 . . . M i k ,j k . We invoke the chain rule as before and replace one of the terms in ζ , say M i m ,j m , with γM i m ,β 1 M β 1 -1 ,j m -γM i m ,β 1 M β 1 +1 ,j m using Equation 32. That is, the derivative of each entry gives rise to two monomials and therefore derivative of ζ leads to 2 k monomials which can be written in the form ζ ′ = M i ′ 1 ,j ′ 1 . . . M i ′ k +1 ,j ′ k +1 where we have the following properties (by appropriately reordering terms)

1. i ′ l , j ′ l = i l , j l for l &lt; m
2. i ′ l , j ′ l = i l -1 , j l -1 for l &gt; m +1
3. i ′ m , j ′ m = i m , β 1 and i ′ m +1 , j ′ m +1 = j m ± 1 , j m

Using the induction hypothesis, we can write

<!-- formula-not-decoded -->

where

1. | c ′ n | = γ | c n | = γ k , since as shown above each coefficient gets multiplied by ± γ .
2. N ′ ≤ 2 k 2 k -1 ( k -1)! = 2 k k ! , since as shown above each monomial ζ leads to 2 k monomials ζ ′ .
3. Each monomial ζ ′ n is of the form M i 1 ,j 1 . . . M i k +1 ,j k +1 , i 1 = 0 , j k +1 = H + 1 , j l ≤ H and i l +1 = j l ± 1 for all l ∈ [1 , k ] .

This completes the induction.

Next we prove a bound on the magnitude of each of the monomials which arise in the derivatives of M 0 ,H +1 . Specifically, we show that for each monomial ζ = M i 1 ,j 1 . . . M i k +1 ,j k +1 , we have

<!-- formula-not-decoded -->

We observe that it suffices to only consider pairs of indices i l , j l where i l &lt; j l . Since | M i,j | ≤ 1 1 -γ for all i, j ,

(by the inductive claim shown above)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(using Lemma B.1, parts 1 and 2 on the first and last terms resp.)

<!-- formula-not-decoded -->

The last step follows from H +1 = j ′ k +1 ≥ i ′ k +1 . Note that

<!-- formula-not-decoded -->

where the first inequality follows from adding only non-positive terms to the sum, the second equality follows from rearranging terms and the third inequality follows from i ′ 1 = 0 , j ′ k +1 = H +1 and i ′ l +1 = j ′ l ± 1 for all l ∈ [1 , k ] . Therefore,

<!-- formula-not-decoded -->

Using Equation (34) and α ≤ 1 with above display gives

This proves the bound. Now using the claim that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where | c n | = γ k and N ≤ 2 k k ! , we have shown that which completes the proof.

<!-- formula-not-decoded -->

We are now ready to prove Proposition 4.1.

Proof: [Proof of Proposition 4.1] The k th order partial derivative of V π θ ( s 0 ) is equal to

<!-- formula-not-decoded -->

Given vectors u 1 , . . . , u k which are unit vectors in R H k (we denote the unit sphere by S H k ), the norm of this gradient tensor is given by:

<!-- formula-not-decoded -->

where the last inequality follows from Lemma B.2. In order to proceed further, we need an upper bound on the smallest admissible value of α . To do so, let us consider all possible parameters θ such that ¯ p ≤ 1 / 4 in accordance with the theorem statement. In order to bound α , it suffices to place an upper bound on the lower end of the range for α in Lemma B.1 (note Lemma B.1 holds for any choice of α in the range). Doing so, we see that

<!-- formula-not-decoded -->

where the first inequality uses √ x -y ≥ √ x - √ y , by triangle inequality while the last inequality uses p ≤ ¯ p ≤ 1 / 4 .

Hence, we have the bound

<!-- formula-not-decoded -->

where ( a ) uses γ = H/ ( H +1) , ( b ) follows since ¯ p ≤ 1 , H,k ≥ 1 , γ ≤ 1 and k ≤ H . Requiring that the gradient norm be no larger than ( 4¯ p 3 ) H/ 4 , we would like to satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (a) follows from a + b ≤ 2 ab when a, b ≥ 1 , (b) follows from H ≥ 1 and ¯ p ≤ 1 / 4 . Therefore, in order to obtain the smallest value of k 0 for all choices of 0 ≤ ¯ p &lt; 1 / 4 , we further lower bound k 0 as

<!-- formula-not-decoded -->

Thus, the norm of the gradient is bounded by ( 4¯ p 3 ) H/ 4 ≤ (1 / 3) H/ 4 for all k ≤ H 40 log(2 H ) -1 as long as ¯ p ≤ 1 / 4 , which gives the first part of the lemma.

for which it suffices to have

Since,

For the second part, note that the optimal policy always chooses the action a 1 , and gets a discounted reward of where the final inequality uses (1 -1 /x ) x ≥ 1 / 8 for x ≥ 1 . On the other hand, the value of π θ is upper bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This gives the second part of the lemma.

## C Proofs for Section 5

We first give a useful lemma about the structure of policy gradients for the softmax parameterization. We use the notation Pr π ( τ | s 0 = s ) to denote the probability of observing a trajectory τ when starting in state s and following the policy π and Pr π µ ( τ ) be E s ∼ µ [Pr π ( τ | s 0 = s )] for a distribution µ over states.

Lemma C.1. For the softmax policy class, we have:

Proof: First note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using this along with the policy gradient expression (6), we have:

where 1 |E ] is the indicator of E being true.

<!-- formula-not-decoded -->

where the second to last step uses that for any policy ∑ a π ( a | s ) A π ( s, a ) = 0 .

## C.1 Proofs for Section 5.1

We now prove Theorem 5.1, i.e. we show that for the updates given by

<!-- formula-not-decoded -->

policy gradient converges to optimal policy for the softmax parameterization.

Weprove this theorem by first proving a series of supporting lemmas. First, we show in Lemma C.2, that V ( t ) ( s ) is monotonically increasing for all states s using the fact that for appropriately chosen stepsizes GD makes monotonic improvement for smooth objectives.

Lemma C.2 (Monotonic Improvement in V ( t ) ( s ) ) . For all states s and actions a , for updates (36) with learning rate η ≤ (1 -γ ) 2 5 , we have

<!-- formula-not-decoded -->

Proof: The proof will consist of showing that:

<!-- formula-not-decoded -->

holds for all states s . To see this, observe that since the above holds for all states s ′ , the performance difference lemma (Lemma 3.2) implies

<!-- formula-not-decoded -->

which would complete the proof.

Let us use the notation θ s ∈ R |A| to refer to the vector of θ s, · for some fixed state s . Define the function where c ( s, a ) is constant, which we later set to be A ( t ) ( s, a ) ; note we do not treat c ( s, a ) as a function of θ . Thus,

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

Taking c ( s, a ) to be A ( t ) ( s, a ) implies ∑ a ′ ∈A π ( t ) ( a ′ | s ) c ( s, a ′ ) = ∑ a ′ ∈A π ( t ) ( a ′ | s ) A ( t ) ( s, a ′ ) = 0 ,

Observe that for the softmax parameterization,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∇ s is gradient w.r.t. θ s and from Lemma C.1 that:

<!-- formula-not-decoded -->

This gives using Equation (39)

<!-- formula-not-decoded -->

Recall that for a β smooth function, gradient ascent will decrease the function value provided that η ≤ 1 /β (Theorem E.1). Because F s ( θ s ) is β -smooth for β = 5 1 -γ (Lemma D.1 and ∣ ∣ A ( t ) ( s, a ) ∣ ∣ ≤ 1 1 -γ ), then our assumption that

<!-- formula-not-decoded -->

implies that η 1 1 -γ d π ( t ) µ ( s ) ≤ 1 /β , and so we have

<!-- formula-not-decoded -->

which implies (37).

Next, we show the limit for iterates V ( t ) ( s ) and Q ( t ) ( s, a ) exists for all states s and actions a .

Lemma C.3. For all states s and actions a , there exists values V ( ∞ ) ( s ) and Q ( ∞ ) ( s, a ) such that as t →∞ , V ( t ) ( s ) → V ( ∞ ) ( s ) and Q ( t ) ( s, a ) → Q ( ∞ ) ( s, a ) . Define

<!-- formula-not-decoded -->

/negationslash where A ( ∞ ) ( s, a ) = Q ( ∞ ) ( s, a ) -V ( ∞ ) ( s ) . Furthermore, there exists a T 0 such that for all t &gt; T 0 , s ∈ S , and a ∈ A , we have

<!-- formula-not-decoded -->

Proof: Observe that Q ( t +1) ( s, a ) ≥ Q ( t ) ( s, a ) (by Lemma C.2) and Q ( t ) ( s, a ) ≤ 1 1 -γ , therefore by monotone convergence theorem, Q ( t ) ( s, a ) → Q ( ∞ ) ( s, a ) for some constant Q ( ∞ ) ( s, a ) . Similarly it follows that V ( t ) ( s ) → V ( ∞ ) ( s ) for some constant V ( ∞ ) ( s ) . Due to the limits existing, this implies we can choose T 0 , such that the result (40) follows.

Based on the limits V ( ∞ ) ( s ) and Q ( ∞ ) ( s, a ) , define following sets:

<!-- formula-not-decoded -->

In the following lemmas C.5- C.11, we first show that probabilities π ( t ) ( a | s ) → 0 for actions a ∈ I s + ∪ I s -as t → ∞ . We then show that for actions a ∈ I s -, lim t →∞ θ ( t ) s,a = -∞ and for all actions a ∈ I s + , θ ( t ) ( a | s ) is bounded from below as t →∞ .

Lemma C.4. We have that there exists a T 1 such that for all t &gt; T 1 , s ∈ S , and a ∈ A , we have

<!-- formula-not-decoded -->

Proof: Since, V ( t ) ( s ) → V ( ∞ ) ( s ) , we have that there exists T 1 &gt; T 0 such that for all t &gt; T 1 ,

<!-- formula-not-decoded -->

Using Equation (40), it follows that for , for s t &gt; T 1 &gt; T 0 a ∈ I -

<!-- formula-not-decoded -->

Similarly A ( t ) ( s, a ) = Q ( t ) ( s, a ) -V ( t ) ( s ) &gt; ∆ / 4 for a ∈ I s + as

<!-- formula-not-decoded -->

which completes the proof.

LemmaC.5. ∂V ( t ) ( µ ) ∂θ s,a → 0 as t →∞ for all states s and actions a . This implies that for a ∈ I s + ∪ I s -, π ( t ) ( a | s ) → 0 and that ∑ a ∈ I s 0 π ( t ) ( a | s ) → 1 .

Proof: Because V π θ ( µ ) is smooth (Lemma D.4) as a function of θ , it follows from standard optimization results (Theorem E.1) that ∂V ( t ) ( µ ) ∂θ s,a → 0 for all states s and actions a . We have from Lemma C.1

Since, | A ( t ) ( s, a ) | &gt; ∆ 4 for all t &gt; T 1 (from Lemma C.4) for all a ∈ I s -∪ I s + and d π ( t ) µ ( s ) ≥ µ ( s ) 1 -γ &gt; 0 (using the strict positivity of µ in our assumption in Theorem 5.1), we have π ( t ) ( a | s ) → 0 .

<!-- formula-not-decoded -->

Lemma C.6. (Monotonicity in θ ( t ) s,a ). For all a ∈ I s + , θ ( t ) s,a is strictly increasing for t ≥ T 1 . For all a ∈ I s -, θ ( t ) s,a is strictly decreasing for t ≥ T 1 .

Proof: We have from Lemma C.1

<!-- formula-not-decoded -->

From Lemma C.4, we have for all t &gt; T 1

<!-- formula-not-decoded -->

Since d π ( t ) µ ( s ) &gt; 0 and π ( t ) ( a | s ) &gt; 0 for the softmax parameterization, we have for all t &gt; T 1

<!-- formula-not-decoded -->

This implies for all a ∈ I s + , θ ( t +1) s,a -θ ( t ) s,a = ∂V ( t ) ( µ ) ∂θ s,a &gt; 0 i.e. θ ( t ) s,a is strictly increasing for t ≥ T 1 . The second claim follows similarly.

Lemma C.7. For all s where I s + = ∅ , we have that:

/negationslash

<!-- formula-not-decoded -->

Proof: Since I s + = ∅ , there exists some action a + ∈ I s + . From Lemma C.5,

<!-- formula-not-decoded -->

or equivalently by softmax parameterization,

<!-- formula-not-decoded -->

From Lemma C.6, for any action a ∈ I s + and in particular for a + , θ ( t ) s,a + is monotonically increasing for t &gt; T 1 . That is the numerator in previous display is monotonically increasing. Therefore, the denominator should go to infinity i.e.

<!-- formula-not-decoded -->

/negationslash

From Lemma C.5, or equivalently

Since, denominator goes to ∞ , which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note this also implies max a ∈A θ ( t ) s,a → ∞ . The last part of the proof is completed using that the gradients sum to 0 , i.e. ∑ a ∂V ( t ) ( µ ) ∂θ s,a = 0 . From gradient sum to 0 , we get that ∑ a ∈A θ ( t ) s,a = ∑ a ∈A θ (0) s,a := c for all t &gt; 0 where c is defined as the sum (over A ) of initial parameters. That is min a ∈A θ ( t ) s,a &lt; -1 |A| max a ∈A θ ( t ) s,a + c . Since, max a ∈A θ ( t ) s,a →∞ , the result follows.

Lemma C.8. Suppose a + ∈ I s + . For any a ∈ I s 0 , if there exists a t ≥ T 0 such that π ( t ) ( a | s ) ≤ π ( t ) ( a + | s ) , then for all τ ≥ t , π ( τ ) ( a | s ) ≤ π ( τ ) ( a + | s ) .

Proof: The proof is inductive. Suppose π ( t ) ( a | s ) ≤ π ( t ) ( a + | s ) , this implies from Lemma C.1

<!-- formula-not-decoded -->

where the second to last step follows from Q ( t ) ( s, a + ) ≥ Q ( ∞ ) ( s, a + ) -∆ / 4 ≥ Q ( ∞ ) ( s, a ) + ∆ -∆ / 4 &gt; Q ( t ) ( s, a ) for t &gt; T 0 . This implies that π ( t +1) ( a | s ) ≤ π ( t +1) ( a + | s ) which completes the proof.

Consider an arbitrary a + ∈ I s + . Let us partition the set I s 0 into B s 0 ( a + ) and ¯ B s 0 ( a + ) as follows: B s 0 ( a + ) is the set of all a ∈ I s 0 such that for all t ≥ T 0 , π ( t ) ( a + | s ) &lt; π ( t ) ( a | s ) , and ¯ B s 0 ( a + ) contains the remainder of the actions from I s 0 . We drop the argument ( a + ) when clear from the context.

Lemma C.9. Suppose I s + = ∅ . For all a + ∈ I s + , we have that B s 0 ( a + ) = ∅ and that

This implies that:

/negationslash

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash

Proof: Let a + ∈ I s + . Consider any a ∈ ¯ B s 0 . Then, by definition of ¯ B s 0 , there exists t ′ &gt; T 0 such that π ( t ) ( a + | s ) ≥ π ( t ) ( a | s ) . From Lemma C.8, for all τ &gt; t π ( τ ) ( a + | s ) ≥ π ( τ ) ( a | s ) . Also, since π ( t ) ( a + | s ) → 0 , this implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since, B s 0 ∪ ¯ B s 0 = I s 0 and ∑ a ∈ I s 0 π ( t ) ( a | s ) → 1 (from Lemma C.5), this implies that B s 0 = ∅ and that means

/negationslash which completes the proof of the first claim. The proof of the second claim is identical to the proof in Lemma C.7 where instead of ∑ a ∈ I s 0 π ( t ) ( a | s ) → 1 , we use ∑ a ∈ B s 0 π ( t ) ( a | s ) → 1 .

Lemma C.10. Consider any s where I s + = ∅ . Then, for any a + ∈ I s + , there exists an iteration T a + such that for all t &gt; T a + , for all a ∈ ¯ B s 0 ( a + ) .

/negationslash

<!-- formula-not-decoded -->

Proof: The proof follows from definition of ¯ B s 0 ( a + ) . That is if a ∈ ¯ B s 0 ( a + ) , then there exists a iteration t a &gt; T 0 such that π ( t a ) ( a + | s ) &gt; π ( t a ) ( a | s ) . Then using Lemma C.8, for all τ &gt; t a , π ( τ ) ( a + | s ) &gt; π ( τ ) ( a | s ) . Choosing completes the proof.

Lemma C.11. For all actions a ∈ I s + , we have that θ ( t ) s,a is bounded from below as t →∞ . For all actions a ∈ I s -, we have that θ ( t ) s,a →-∞ as t →∞ .

<!-- formula-not-decoded -->

Proof: For the first claim, from Lemma C.6, we know that after T 1 , θ ( t ) s,a is strictly increasing for a ∈ I s + , i.e. for all t &gt; T 1

For the second claim, we know that after T 1 , θ ( t ) s,a is strictly decreasing for a ∈ I s -(Lemma C.6). Therefore, by monotone convergence theorem, lim t →∞ θ ( t ) s,a exists and is either -∞ or some constant θ 0 . We now prove the second claim by contradiction. Suppose a ∈ I s -and that there exists a θ 0 , such that θ ( t ) s,a &gt; θ 0 , for t ≥ T 1 . By Lemma C.7, there must exist an action where a ′ ∈ A such that

Let us consider some δ &gt; 0 such that θ ( T 1 ) s,a ′ ≥ θ 0 -δ . Now for t ≥ T 1 define τ ( t ) as follows: τ ( t ) = k if k is the largest iteration in the interval [ T 1 , t ] such that θ ( k ) s,a ′ ≥ θ 0 -δ (i.e. τ ( t ) is the latest iteration before θ s,a ′ crosses below θ 0 -δ ). Define T ( t ) as the subsequence of iterations τ ( t ) &lt; t ′ &lt; t such that θ ( t ′ ) s,a ′ decreases, i.e.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define Z t as the sum (if T ( t ) = ∅ , we define Z t = 0 ):

<!-- formula-not-decoded -->

For non-empty T ( t ) , this gives:

<!-- formula-not-decoded -->

where we have used that | ∂V ( t ′ ) ( µ ) ∂θ s,a ′ | ≤ 1 / (1 -γ ) . By (43), this implies that:

<!-- formula-not-decoded -->

For any T ( t ) = ∅ , this implies that for all t ′ ∈ T ( t ) , from Lemma C.1

/negationslash where we have used that | A ( t ′ ) ( s, a ′ ) | ≤ 1 / (1 -γ ) and | A ( t ′ ) ( s, a ) | ≥ ∆ 4 for all t ′ &gt; T 1 (from Lemma C.4). Note that since ∂V ( t ′ ) ( µ ) ∂θ s,a &lt; 0 and ∂V ( t ′ ) ( µ ) ∂θ s,a ′ &lt; 0 over the subsequence T ( t ) , the sign of the inequality reverses. In particular, for any T ( t ) = ∅

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

where the first step follows from that θ ( t ) s,a is monotonically decreasing, i.e. ∂V ( t ) ( µ ) ∂θ s,a &lt; 0 for t / ∈ T (Lemma C.6). Since,

<!-- formula-not-decoded -->

this contradicts that θ ( t ) s,a is lower bounded from below, which completes the proof.

Lemma C.12. Consider any s where I s + = ∅ . Then, for any a + ∈ I s + ,

/negationslash

<!-- formula-not-decoded -->

Proof: Consider any a ∈ B s 0 . We have by definition of B s 0 that π ( t ) ( a + | s ) &lt; π ( t ) ( a | s ) for all t &gt; T 0 . This implies by the softmax parameterization that θ ( t ) s,a + &lt; θ ( t ) s,a . Since, θ ( t ) s,a + is lower bounded as t → ∞ (using Lemma C.11), this implies θ ( t ) s,a is lower bounded as t → ∞ for all a ∈ B s 0 . This in conjunction with max a ∈ B s 0 ( a + ) θ ( t ) s,a →∞ implies which proves this claim.

We are now ready to complete the proof for Theorem 5.1. We prove it by showing that I s + is empty for all states s or equivalently V ( t ) ( s 0 ) → V /star ( s 0 ) as t →∞ .

<!-- formula-not-decoded -->

Proof: [Proof for Theorem 5.1] Suppose the set I s + is non-empty for some s , else the proof is complete. Let a + ∈ I s + . Then, from Lemma C.12,

Now we proceed by showing a contradiction. For a ∈ I s -, we have that since π ( t ) ( a | s ) π ( t ) ( a + | s ) = exp( θ ( t ) s,a -θ ( t ) s,a + ) → 0 (as θ ( t ) s,a + is lower bounded and θ ( t ) s,a → -∞ by Lemma C.11), there exists T 2 &gt; T 0 such that or, equivalently,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For a ∈ ¯ B s 0 , we have A ( t ) ( s, a ) → 0 (by definition of set I s 0 and ¯ B s 0 ⊂ I s 0 ) and 1 &lt; π ( t ) ( a + | s ) π ( t ) ( a | s ) for all t &gt; T a + from Lemma C.10. Thus, there exists T 3 &gt; T 2 , T a + such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We have for t &gt; T 3 , from ∑ a ∈A π ( t ) ( a | s ) A ( t ) ( s, a ) = 0 , 0 = ∑ a ∈ I s 0 π ( t ) ( a | s ) A ( t ) ( s, a ) + ∑ a ∈ I s + π ( t ) ( a | s ) A ( t ) ( s, a ) + ∑ a ∈ I s -π ( t ) ( a | s ) A ( t ) ( s, a ) ( a ) ≥ ∑ a ∈ B s 0 π ( t ) ( a | s ) A ( t ) ( s, a ) + ∑ a ∈ ¯ B s 0 π ( t ) ( a | s ) A ( t ) ( s, a ) + π ( t ) ( a + | s ) A ( t ) ( s, a + ) + ∑ a ∈ I s -π ( t ) ( a | s ) A ( t ) ( s, a ) ∑ ∑ ∑ ( t )

where in the step (a), we used A ( t ) ( s, a ) &gt; 0 for all actions a ∈ I s + for t &gt; T 3 &gt; T 1 from Lemma C.4. In the step (b), we used A ( t ) ( s, a + ) ≥ ∆ 4 for t &gt; T 3 &gt; T 1 from Lemma C.4 and A ( t ) ( s, a ) ≥ -1 1 -γ . In the step (c), we used Equation (46) and left inequality in (47). This implies that for all t &gt; T 3

<!-- formula-not-decoded -->

This contradicts Equation (45) which requires

<!-- formula-not-decoded -->

Therefore, the set I s + must be empty, which completes the proof.

## C.2 Proofs for Section 5.2

Proof: [ of Corollary 5.1 ] Using Theorem 5.2, the desired optimality gap /epsilon1 will follow if we set

<!-- formula-not-decoded -->

and if ‖∇ θ L λ ( θ ) ‖ 2 ≤ λ/ (2 |S| |A| ) . In order to complete the proof, we need to bound the iteration complexity of making the gradient sufficiently small.

Since the optimization is deterministic and unconstrained, we can appeal to standard results (Theorem E.1) which give that after T iterations of gradient ascent with stepsize of 1 /β λ , we have

<!-- formula-not-decoded -->

where β λ is an upper bound on the smoothness of L λ ( θ ) . We seek to ensure

<!-- formula-not-decoded -->

Choosing T ≥ 8 β λ |S| 2 |A| 2 (1 -γ ) λ 2 satisfies the above inequality. By Lemma D.4, we can take β λ = 8 γ (1 -γ ) 3 + 2 λ |S| , and so where we have used that λ &lt; 1 . This completes the proof.

<!-- formula-not-decoded -->

## C.3 Proofs for Section 5.3

Proof: [ of Lemma 5.1 ] Following the definition of compatible function approximation in Sutton et al. [1999], which was also invoked in Kakade [2001], for a vector w ∈ R |S||A| , we define the error function

<!-- formula-not-decoded -->

Let w /star θ be the minimizer of L θ ( w ) with the smallest /lscript 2 norm. Then by definition of MoorePenrose pseudoinverse, it is easily seen that

<!-- formula-not-decoded -->

In other words, w /star θ is precisely proportional to the NPG update direction. Note further that for the Softmax policy parameterization, we have by (35),

<!-- formula-not-decoded -->

Since ∑ a ∈A π ( a | s ) A π ( s, a ) = 0 , this immediately yields that L θ ( A π θ ) = 0 . However, this might not be the unique minimizer of L θ , which is problematic since w /star ( θ ) as defined in terms of the Moore-Penrose pseudoinverse is formally the smallest norm solution to the least-squares problem, which A π θ may not be. However, given any vector v ∈ R |S||A| , let us consider solutions of the form A π θ + v . Due to the form of the derivatives of the policy for the softmax parameterization (recall Equation 35), we have for any state s, a such that s is reachable under ρ ,

<!-- formula-not-decoded -->

Note that here we have used that π θ is a stochastic policy with π θ ( a | s ) &gt; 0 for all actions a in each state s , so that if a state is reachable under ρ , it will also be reachable using π θ , and hence the zero derivative conditions apply at each reachable state. For A π θ + v to minimize L θ , we would like v /latticetop ∇ θ log π θ ( a | s ) = 0 for all s, a so that v s,a is independent of the action and can be written as a constant c s for each s by the above equality. Hence, the minimizer of L θ ( w ) is determined up to a state-dependent offset, and where v s,a = c s for some c s ∈ R for each state s and action a . Finally, we observe that this yields the updates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Owing to the normalization factor Z t ( s ) , the state dependent offset c s cancels in the updates for π , so that resulting policy is invariant to the specific choice of c s . Hence, we pick c s ≡ 0 , which yields the statement of the lemma.

## D Smoothness Proofs

Various convergence guarantees we show leverage results from smooth, non-convex optimization. In this section, we collect the various results on smoothness of policies and value functions in the different parameterizations which are needed in our analysis.

Define the Hadamard product of two vectors:

<!-- formula-not-decoded -->

Define diag ( x ) for a column vector x as the diagonal matrix with diagonal as x .

Lemma D.1 (Smoothness of F (see Equation 38) ) . Fix a state s . Let θ s ∈ R |A| be the column vector of parameters for state s . Let π θ ( ·| s ) be the corresponding vector of action probabilities given by the softmax parameterization. For some fixed vector c ∈ R |A| , define:

<!-- formula-not-decoded -->

Then where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: For notational convenience, we do not explicitly state the s dependence. For the softmax parameterization, we have that

<!-- formula-not-decoded -->

We can then write (as ∇ θ π θ is symmetric),

<!-- formula-not-decoded -->

and therefore

For the first term, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the second term, we can decompose by chain rule

<!-- formula-not-decoded -->

Substituting these back, we get

<!-- formula-not-decoded -->

Note that which gives

Before we prove the smoothness results for ∇ π V π ( s 0 ) and ∇ θ V π θ ( s 0 ) , we prove the following helpful lemma. This lemma is general and not specific to the direct or softmax policy parameterizations.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma D.2. Let π α := π θ + αu and let ˜ V ( α ) be the corresponding value at a fixed state s 0 , i.e.

<!-- formula-not-decoded -->

Assume that

Then

<!-- formula-not-decoded -->

Proof: Consider a unit vector u and let P ( α ) be the state-action transition matrix under π , i.e.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can differentiate ˜ P ( α ) w.r.t α to get

<!-- formula-not-decoded -->

For an arbitrary vector x ,

<!-- formula-not-decoded -->

and therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By definition of /lscript ∞ norm,

Similarly, differentiating P ( α ) twice w.r.t. α , we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

An identical argument leads to that, for arbitrary x ,

<!-- formula-not-decoded -->

Let Q α ( s 0 , a 0 ) be the corresponding Q -function for policy π α at state s 0 and action a 0 . Observe that Q α ( s 0 , a 0 ) can be written as:

<!-- formula-not-decoded -->

where M ( α ) := ( I -γ P ( α )) -1 and differentiating twice w.r.t α gives:

By using power series expansion of matrix inverse, we can write M ( α ) as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies that M ( α ) ≥ 0 (componentwise) and M ( α ) 1 = 1 1 -γ 1 , i.e. each row of M ( α ) is positive and sums to 1 / (1 -γ ) . This implies:

<!-- formula-not-decoded -->

This gives using expression for d 2 Q α ( s 0 ,a 0 ) ( dα ) 2 and dQ α ( s 0 ,a ) dα ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consider the identity:

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

which completes the proof.

Using this lemma, we now establish smoothness for: the value functions under the direct policy parameterization and the log barrier regularized objective 12 for the softmax parameterization.

Lemma D.3 (Smoothness for direct parameterization) . For all starting states s 0 ,

<!-- formula-not-decoded -->

Proof: By differentiating π α w.r.t α gives

<!-- formula-not-decoded -->

and differentiating again w.r.t α gives

<!-- formula-not-decoded -->

Using this with Lemma D.2 with C 1 = √ |A| and C 2 = 0 , we get which completes the proof.

We now present a smoothness result for the entropy regularized policy optimization problem which we study for the softmax parameterization.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma D.4 (Smoothness for log barrier regularized softmax) . For the softmax parameterization and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have that where

Proof: Let us first bound the smoothness of V π θ ( µ ) . Consider a unit vector u . Let θ s ∈ R |A| denote the parameters associated with a given state s . We have:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where e a is a standard basis vector and π ( ·| s ) is a vector of probabilities. We also have by differentiating π α ( a | s ) once w.r.t α ,

Similarly, differentiating once again w.r.t. α , we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using this with Lemma D.2 for C 1 = 2 and C 2 = 6 , we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now let us bound the smoothness of the regularizer λ |S| R ( θ ) , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash where β = 8 (1 -γ ) 3 .

We have

Equivalently,

Hence,

For any vector u s ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ∇ θ s ∇ θ s ′ R ( θ ) = 0 for s = s ′ ,

Thus R is 2 -smooth and λ |S| R is 2 λ |S| -smooth, which completes the proof.

## E Standard Optimization Results

In this section, we present the standard optimization results from Ghadimi and Lan [2016], Beck [2017] used in our proofs. We consider solving the following problem

<!-- formula-not-decoded -->

with C being a nonempty closed and convex set. We assume the following

Assumption E.1. f : R d → ( -∞ , ∞ ) is proper and closed, dom ( f ) is convex and f is β smooth over int ( dom ( f )) .

Throughout the section, we will denote the optimal f value by f ( x ∗ ) .

Definition E.1 (Gradient Mapping) . We define the gradient mapping G η ( x ) as

<!-- formula-not-decoded -->

where P C is the projection onto C .

Note that when C = R d , the gradient mapping G η ( x ) = ∇ f ( x ) .

Theorem E.1 (Theorem 10.15 Beck [2017]) . Suppose that Assumption E.1 holds and let { x k } k ≥ 0 be the sequence generated by the gradient descent algorithm for solving the problem (54) with the stepsize η = 1 /β . Then,

1. The sequence { F ( x t ) } t ≥ 0 is non-increasing.
2. G η ( x t ) → 0 as t →∞

<!-- formula-not-decoded -->

Theorem E.2 (Lemma 3 Ghadimi and Lan [2016]) . Suppose that Assumption E.1 holds. Let x + = x -ηG η ( x ) . Then,

<!-- formula-not-decoded -->

where B 2 is the unit /lscript 2 ball, and N C is the normal cone of the set C .

We now consider the stochastic projected gradient descent algorithm where at each time step t , we update x t by sampling a random v t such that

<!-- formula-not-decoded -->

Theorem E.3 (Theorem 14.8 and Lemma 14.9 Shalev-Shwartz and Ben-David [2014]) . Assume C = { x : ‖ x ‖ ≤ B } , for some B &gt; 0 . Let f be a convex function and let x ∗ ∈ argmin x : ‖ x ‖≤ B f ( w ) . Assume also that for all t , ‖ v t ‖ ≤ ρ , and that stochastic projected gradient descent is run for N iterations with η = √ B 2 ρ 2 N . Then,

<!-- formula-not-decoded -->