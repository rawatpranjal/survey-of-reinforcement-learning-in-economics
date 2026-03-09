## Towards Minimax Optimality of Model-based Robust Reinforcement Learning

Pierre Clavier 1,2,3

Erwan Le Pennec 1

Matthieu Geist 4

1 CMAP, CNRS, Ecole Polytechnique, Institut Polytechnique de Paris, 91120 Palaiseau, France, 2 INRIA Paris, HeKA, France,

3 Centre de Recherche des Cordeliers, INSERM, Universite de Paris, Sorbonne Universite, F-75006 Paris, France, 4 Cohere.

## Abstract

We study the sample complexity of obtaining an /epsilon1 -optimal policy in Robust discounted Markov Decision Processes (RMDPs), given only access to a generative model of the nominal kernel. This problem is widely studied in the non-robust case, and it is known that any planning approach applied to an empirical MDP estimated with ˜ O ( H 3 | S || A | /epsilon1 2 ) samples provides an /epsilon1 -optimal policy, which is minimax optimal. Results in the robust case are much more scarce. For sa - (resp s -) rectangular uncertainty sets, until recently the best-known sample complexity was ˜ O ( H 4 | S | 2 | A | /epsilon1 2 ) (resp. ˜ O ( H 4 | S | 2 | A | 2 /epsilon1 2 ) ), for specific algorithms and when the uncertainty set is based on the total variation (TV), the KL or the Chi-square divergences. In this paper, we consider uncertainty sets defined with an L p -ball (recovering the TV case), and study the sample complexity of any planning algorithm (with high accuracy guarantee on the solution) applied to an empirical RMDP estimated using the generative model. In the general case, we prove a sample complexity of ˜ O ( H 4 | S || A | /epsilon1 2 ) for both the sa -and s -rectangular cases (improvements of | S | and | S || A | respectively). When the size of the uncertainty is small enough, we improve the sample complexity to ˜ O ( H 3 | S || A | /epsilon1 2 ) , recovering the lower-bound for the non-robust case for the first time and a robust lower-bound. Finally, we also introduce simple and efficient algorithms for solving the studied L p robust MDPs.

## 1 INTRODUCTION

Reinforcement learning (RL) [Sutton and Barto, 2018], often modelled as learning and decision-making in a Markov decision process (MDP), has attracted increasing interest in recent years due to its remarkable success in practice. A major goal of RL is to find a strategy or policy, based on a collection of data samples, that can predict the expected cumulative rewards in an MDP, without direct access to a detailed description of the underlying model. However, Mannor et al. [2004] showed that the policy and the value function could sometimes be sensitive to estimation errors of the reward and transition probabilities, meaning that a very small perturbation of the reward and transition probabilities could lead to a significant change in the value function.

Robust MDPs [Iyengar, 2005, Nilim and El Ghaoui, 2005] (RMDPs) have been proposed to handle these problems by letting the transition probability vary in an uncertainty (or ambiguity) set. In this way, the solution of robust MDPs is less sensitive to model estimation errors with a properly chosen uncertainty set. An RMDP problem is usually formulated as a max-min problem, where the objective is to find the policy that maximizes the value function for the worst possible model that lies within an uncertainty set around a nominal model. Initially, RMPDs [Iyengar, 2005, Nilim and El Ghaoui, 2005] were developed because the solution of MDPs can be very sensitive to the model parameters [Zhao et al., 2019, Packer et al., 2018]. However, as the solution of robust MDPs is NP-hard for general uncertainty sets Nilim and El Ghaoui [2005], the uncertainty set is usually assumed to be rectangular (meaning that it can be decomposed as a product of uncertainty sets for each state or state-action pair), which allows tractability Iyengar [2005], Ho et al. [2021]. These two kinds of sets are called respectively s - and sa -rectangular sets. A fundamental difference between them is that the greedy and optimal policy in sa -rectangular robust MDPs is deterministic, as in non-robust MDPs, but can be stochastic in the s -rectangular case Wiesemann et al. [2013]. Compared to sa -rectangular robust MDPs, s -rectangular robust MDPs are less restrictive but much more difficult to handle. Under this rectangularity assumption, many structural properties of MDPs remain intact Iyengar [2005] and methods such as robust value iteration, robust modified policy iteration, or partial robust policy iteration Ho et al. [2021] can be used to solve them. It is also known that the uncertainty in the reward can be easily handled, while handling uncertainty in the transition kernel is much more difficult Kumar et al. [2022], Derman et al. [2021]. Finally, Deep Robust RL algorithms Pinto et al. [2017], Clavier et al. [2022], Tanabe et al. [2022] have been proposed to tackle the problem of Robust MDPS with continuous state-action space.

In this work, we consider robust MDPs, with both sa - and s -rectangular uncertainty sets, consisting of L p -balls centered around the nominal model P 0 . We assume access to a generative model, which can sample a next state from any state-action pair from the nominal model. The question we address is to know how many samples are required to compute an /epsilon1 -optimal policy. This classic abstraction, which allows studying the sample complexity of planning over a long horizon, is widely studied in the non-robust setting Singh and Yee [1994], Sidford et al. [2018], Azar et al. [2013], Agarwal et al. [2020], Li et al. [2020], Kozuno et al. [2022], but much less in the robust setting [Yang et al., 2021, Panaganti and Kalathil, 2022, Shi and Chi, 2022, Xu et al., 2023, Shi et al., 2023]. We consider more specifically model-based robust RL. We call the generative model the same number of times for each state-action pair, to build a maximum likelihood estimate of the nominal model, and use any planning algorithm for robust MDPs (with high accuracy guarantee on the solution) on this empirical model. This setting will be discussed further later, but we insist right away that it is especially meaningful in the robust setting, as it is a good abstraction of sim2real. The research question we address is:

How many samples are required for guaranteeing an /epsilon1 -optimal policy with high probability?

Our first contribution is to prove that for both s and sa -rectangular sets based on L p -balls, the sample complexity of the proposed approach is ˜ O ( H 4 | S || A | /epsilon1 2 ) , with H = (1 -γ ) -1 being the horizon term. Previous works [Yang et al., 2021, Panaganti and Kalathil, 2022, Shi and Chi, 2022, Xu et al., 2023] study different sets, based on the Kullback-Leibler (KL) divergence, Chi-square divergence, and total variation (TV). We have the TV in common ( L 1 -ball up to a normalizing factor), and, in this case, we improve these existing results by | S | for the sa -rectangular case, and by | S || A | for the s -rectangular case, which is significant for large state-action spaces. On the technical side, our results build heavily upon the dual view of robust Bellman operators [Derman et al., 2021, Kumar et al., 2022]. However, we deviate from this line of work by enforcing the uncertainty set to belong to the simplex. This allows ensuring that the robust operators are overly conservative while ensuring they are γ -contractions, which is important for the theoretical analysis. On the negative side, the algorithms they introduce are no longer applicable, which calls for new algorithmic design.

Our second contribution is to show that, if the uncertainty set is small enough, then we have a sample complexity of ˜ O ( H 3 | S || A | /epsilon1 2 ) . This is a further improvement by H of the previous bound, and it matches the known lower bound for the non-robust case [Azar et al., 2013]. On the technical side, it again builds upon the dual view of robust Bellman operators with the deviation mentioned above.[Derman et al., 2021, Kumar et al., 2022]. In addition to that, it adapts two proof techniques of the non-robust case: The total variance technique of Azar et al. [2013] to reduce the dependency to the horizon, and the absorbing MDP construction of Agarwal et al. [2020] to allow for a wider range of valid /epsilon1 .As mentioned earlier,[Derman et al., 2021, Kumar et al., 2022] algorithms are not applicable to the more realistic uncertainty sets we consider.

Our third contribution is an algorithm DRVI L P (see Alg. 1, for Distributionally Robust Value Iteration for L P in sa rectangular case that solves exactly RMDPs in the case of valid robust transition that belongs to the simplex contrary to Kumar et al. [2022].

## 2 RELATED WORK

The question of sample complexity when having access to a generative model has been widely studied in the non-robust setting Singh and Yee [1994], Sidford et al. [2018], Azar et al. [2013], Agarwal et al. [2020], Li et al. [2020], Kozuno et al. [2022]. Notably, Azar et al. [2013] provide a lowerbound of this sample complexity, ˜ Ω( | S || A | H 3 /epsilon1 2 ) , and show that (tabular) model-based RL reaches this lower-bound, making it minimax optimal (up to polylog factors). This bound relies on the so-called total variance technique, that we adapt to the robust setting. However, their result is only true for small enough /epsilon1 , in the range (0 , √ H/ | S | ) . This was later improved to (0 , √ H ) by Agarwal et al. [2020], thanks to a novel absorbing MDP construction, that we also adapt to the robust setting.

Closer to our contributions are the works that study the sample complexity in the robust setting Yang et al. [2021], Panaganti and Kalathil [2022], Xu et al. [2023], Shi and Chi [2022]. The study of sample complexity of specific algorithms (respectively either empirical robust value or Robust Phased Value Learning) is studied by Panaganti and Kalathil [2022], Xu et al. [2023], while our results apply to any oracle planning (applied to the empirical model), as long as it provides a solution with enough accuracy. We consider both s - and sa -rectangular uncertainty sets, as Yang et al. [2021], while Panaganti and Kalathil [2022], Xu et al. [2023], Shi and Chi [2022] only consider the simpler sa -rectangular sets. They all study either TV, KL or Chisquare balls, while we study L p -balls. Shi and Chi [2022] improved the KL bound compared to Yang et al. [2021], Panaganti and Kalathil [2022] in the sa rectangular case.

The framework of Xu et al. [2023] is slightly different as they consider finite horizon which adds a factor H in all bounds. All previous results are not minimax optimal in terms of the horizon factor.

We rely more specifically on a simple optimization dual expression of the minimization problem over models. As such, we do not cover the KL and Chi-square cases, which do not have such a simple form even if there can also be written as simple scalar optimization problem. However, we have in common with Yang et al. [2021], Panaganti and Kalathil [2022] the total variation case, which corresponds to a (scaled) L 1 -ball. For this case, we can compare our sample complexities. Without assumption on the size of the uncertainty set, we improve the existing sample complexities by | S | and | S || A | respectively (for sa - or s -rectangularity). Also, our bounds have no dependency on the size of the uncertainty set. Notice that as we consider a generic oracle planning algorithm, our bounds apply to the algorithms they consider in Panaganti and Kalathil [2022], Xu et al. [2023]. If we further assume that the uncertainty set is small enough, then we improve the bound by an additional H factor, reaching the minimax sample complexity of the non-robust case. Table 1 summarizes the difference in sample complexity, and we'll discuss them again after stating our theorems.

Finally, the archival version of this contribution predates the concurrent work of Shi et al. [2023] that studies the sample complexity of RMDPs for TV and χ 2 divergence. In the very specific case of sa - rectangular for TV which in this case coincides with L 1 norm, Shi et al. [2023] retrieves our upper bound which is minimax optimal in the regime where the radius of the uncertainty set is small and improves our result in the regime where the radius of the uncertainty set is bigger than 1 -γ . However, our results hold more generally for the s -rectangular case are still state-of-the-art for s -rectangular case with p ≥ 1 and for sa -rectangular with p &gt; 1 . Notice also that the proof techniques are very different, and it is an interesting research direction to know if their bound for the regime where the radius of the uncertainty set is bigger than 1 -γ or their lower-bound would extend to the more general case studied here.

## 3 PRELIMINARIES

For finite sets S and A , we write respectively | S | and | A | their cardinality. We write ∆ A := { p : A → R | p ( a ) ≥ 0 , ∑ a ∈ A p ( a ) = 1 } the simplex over A . For v ∈ R S the classic L q norm is ‖ v ‖ q q = ∑ s v ( s ) q . The unitary vector of dimension | S | is denoted 1 S . Finally, we denote ˜ O the O notation up to logarithm factor.

## 3.1 MARKOVDECISION PROCESS

A Markov Decision Process (MDP) is defined by M = ( S , A , P, R, γ, µ ) where S and A are the finite state and action spaces, P : S × A → ∆ S is the transition kernel, R : S × A → [0 , 1] is the reward function, µ ∈ ∆ S is the initial distribution over states and γ ∈ [0 , 1) is the discount factor. A stationary policy π : S → ∆ A maps states to probability distributions over actions. We write P s,a the vector P ( ·| s, a ) . We also define P π to be the transition matrix on state-action pairs induced by a policy π : P π ( s,a ) , ( s ′ ,a ′ ) = P ( s ′ | s, a ) π ( a ′ | s ′ ) . Slightly abusing notations, for V ∈ R S , we define the vector Var P ( V ) ∈ R S× A as Var P ( V )( s, a ) := Var P ( ·| s,a ) ( V ) , so that Var P ( V ) = P ( V ) 2 -( PV ) 2 (with the square understood component-wise). Usually, the goal is to estimate the value function defined as:

<!-- formula-not-decoded -->

The value function V π P,R for policy π , is the fixed point of the Bellmen operator T P,R , defined as

<!-- formula-not-decoded -->

We also define the optimal Bellman operator: T ∗ P,R V ( s ) = max π s ∈ ∆ A ( T π s P,R V ) ( s ) . Both optimal and classical Bellman operators are γ -contractions Sutton and Barto [2018]. This is why sequences { V π n | n ≥ 0 } , and { V ∗ n | n ≥ 0 } , defined as

<!-- formula-not-decoded -->

converge linearly to V π P,R and V ∗ P,R , respectively the value function following π and the optimal value function. Finally, we can define the Q-function,

<!-- formula-not-decoded -->

The value function and Q-function are linked with the relation V π P,R ( s ) = 〈 ( π s , Q π P,R ( s ) 〉 A . With these notations, we can define Q-functions for transition probability transition P following policy π such as

<!-- formula-not-decoded -->

## 3.2 ROBUST MARKOV DECISION PROCESS

Once classical MDPs defined, we can define robust (optimal) Bellman operators T π U and T ∗ U ,

<!-- formula-not-decoded -->

Table 1: Sample Complexity of TV for s - or sa rectangular with β (see Def 3.2) the radius of uncertainty set (see also Tab. 2 in the appendix for a complete table with different norms)

|            | Panaganti and Kalathil [2022]                             | Yang et al. [2021]                                                        | Our β ≥ 0                                              | Our 1 / (2 Hγ ) > β > 0                                | Shi et al. [2023]                                                   |
|------------|-----------------------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------|
| sa - rect. | ˜ O ( &#124; S &#124; 2 &#124; A &#124; H 4 /epsilon1 2 ) | ˜ O ( &#124; S &#124; 2 &#124; A &#124; H 4 (2+ β ) 2 /epsilon1 2 β 2 )   | ˜ O ( &#124; S &#124;&#124; A &#124; H 4 /epsilon1 2 ) | ˜ O ( &#124; S &#124;&#124; A &#124; H 3 /epsilon1 2 ) | ˜ O ( &#124; S &#124;&#124; A &#124; H 2 /epsilon1 2 min(1 /H,β ) ) |
| s -rect.   | ×                                                         | ˜ O ( &#124; S &#124; 2 &#124; A &#124; 2 H 4 (2+ β ) 2 /epsilon1 2 β 2 ) | ˜ O ( &#124; S &#124;&#124; A &#124; H 4 /epsilon1 2 ) | ˜ O ( &#124; S &#124;&#124; A &#124; H 3 /epsilon1 2 ) | ×                                                                   |

<!-- formula-not-decoded -->

where P and R belong to the uncertainty set U . The optimal robust Bellman operator T ∗ U and robust Bellman operator T π U are γ -contraction maps for any policy π [Iyengar, 2005, Thm. 3.2] if the adversarial kernel P ∈ ∆ s to obtain a valid transition kernel :

<!-- formula-not-decoded -->

Finally, for any initial values V π 0 , V ∗ 0 , sequences defined as V π n +1 := T π U V π n and V ∗ n +1 := T ∗ U V ∗ n converge linearly to their respective fixed points, that is V π n → V π U and V ∗ n → V ∗ U . This makes robust value iteration an attractive method for solving robust MDPs. In order to obtain tractable forms of RMDPs, one has to make assumptions about the uncertainty sets and give them a rectangularity structure Iyengar [2005]. In the following, we will use an L p norm as the distance between distributions. The s - and sa -rectangular assumptions can be defined as follows, with R 0 and P 0 being called the nominal reward and kernel.

Assumption 3.1. ( sa -rectangularity) We define sa -rectangular L p -constrained uncertainty set as

<!-- formula-not-decoded -->

Assumption 3.2. ( s -rectangularity) We define srectangular L p -constrained uncertainty set as

<!-- formula-not-decoded -->

We write β = sup s,a β s,a for sa -rectangular assumptions or β = sup s β s for s -rectangular assumptions and with the same manner α = sup s,a α s,a . Moreover, we write P ∈ P 0 ,s,a for P = P 0 ,s,a + P ′ with P ′ ∈ P s,a and P ∈ P 0 ,s for P = P π 0 ,s + P ′ with P ′ ∈ P s , P π 0 ,s ( s ′ ) = ∑ a π ( a | s ) P 0 ,s,a ( s ′ ) ∈ R S .

In comparison to sa -rectangular robust MDPs, s -rectangular robust MDPs are less restrictive but much more difficult to deal with. Using rectangular assumptions and constraints defined with L p -balls, it is possible to derive simple dual forms for the (optimal) robust Bellman operators for the minimization problem that involves the seminorm defined below:

Definition 3.1 (Span seminorm [Puterman, 1990]) . Let q be such that it satisfies the Holder's equality, i.e. 1 p + 1 q = 1 . Let q -variance or span-seminorm function sp q ( . ) : S → R and q -mean function ω q : S → R be defined as

<!-- formula-not-decoded -->

One can think of those span-seminorms as semi-meancentered-norms. The main problem is that these quantities represent the dispersion of a distribution around its mean, and there are no order relations for this type of object. Seminorms appear in the (non-robust) RL community for other reasons Puterman [1990], Scherrer [2013]. For p = 1, 2 and ∞ , a closed form can be derived, corresponding to median, variance and range. This is not the case for arbitrary p but span-seminorms can be efficiently computed in practice, see Kumar et al. [2022]. Once span-seminorms defined, we introduced the dual of the inner minimization problem.

Lemma3.3 (Duality for sa rectangular case with L p norm) . For any V ∈ R S , P 0 ,s,a = P 0 ( . | s, a ) ∈ R S and µ ∈ R S

<!-- formula-not-decoded -->

Lemma 3.4 (Duality for s rectangular case.) . Consider the probability kernel P π 0 ,s = Π π P 0 ,s,a ∈ R s with Π π a projection matrix associated with a given policy π such that P π 0 ,s ( s ′ ) = ∑ a π ( a | s ) P 0 ,s,a ( s ′ ) ∈ R S . For any V ∈ R S :

<!-- formula-not-decoded -->

Proofs car be found in Appendix B.5 ,3.4. These results allow computing robust value and Q-functions. Close to our work, Derman et al. [2021], Kumar et al. [2022] do not assume that robust kernel belongs to the simplex and in that sense, their formulation is a relaxation of the framework of RMPDs. Using this relaxation, closed form of robust Bellman operator can be obtained, see Th. 1 in Kumar et al. [2022]. In our work, we assume a valid transition kernel in the simplex ( P s,a ≥ 0 or P s ≥ 0 for respectively sa -or s -rectangular case.) that leads to dual form that has not a closed form but which is a simple scalar optimization problem. A complete discussion can be found in Appendix A.2.

Finally, we denote robust Q function for sa -and s -rectangular respectively Q π sa and Q π s and we define them from robust value function V π sa , V π s as :

<!-- formula-not-decoded -->

Lemma 3.5. For sa -and s -rectangular,

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Robust Q functions and dual forms of the robust Bellman operators will be central to our analysis of the sample complexity of model-based robust RL. They allow improving the bound by a factor | S | or | S || A | compared to existing results (Sec. 4). With additional technical subtleties, adapted from the non-robust setting, and assuming the uncertainty set is small enough, they even allow improving the bound by a factor | S | H or | S || A | H (Sec. 5).

## 3.3 GENERATIVE MODEL FRAMEWORK

We consider the setting where we have access to a generative model, or sampler, that gives us samples s ′ ∼ P 0 ( · | s, a ) , from the nominal model and from arbitrary stateaction couples. Suppose we call our sampler N times on each state-action pair ( s, a ) . Let ̂ P be our empirical model, the maximum likelihood estimate of P 0 , where count( s ′ , s, a ) represents the number of times the state-action pair ( s, a ) transitions to state s ′ . Moreover,

<!-- formula-not-decoded -->

we define ̂ M as the empirical RMDP identical to the original M except that it uses ̂ P instead of P 0 for the transition kernel. We denote by ̂ V π and ̂ Q π the value functions of a policy π in ̂ M , and ̂ π /star , ̂ Q /star and ̂ V /star denote the optimal policy and its value functions in ̂ M . It is assumed that the reward function R 0 is known and deterministic and therefore exactly identical in M and ̂ M . Moreover, we write P ∈ ˆ P s,a for P = ˆ P s,a + P ′ with P ′ ∈ P s,a and P ∈ ˆ P s for P = ˆ P π s + P ′ with P ′ ∈ P s , ˆ P π s ( s ′ ) = ∑ a π ( a | s ) ˆ P s,a ( s ′ ) ∈ R S .

Notice that our analysis would easily account for an estimated reward (the hard part being handling the estimated transition model). This generative model framework, when we can only sample from the nominal kernel, is classic and appears for both non-robust and robust MDPs [Agarwal et al., 2020, Panaganti et al., 2022, Azar et al., 2013, Xu et al., 2023]. In the robust case, it is especially relevant as an abstraction of "sim-to-real", the simulator giving access to the nominal kernel for learning a robust policy to be deployed in the real world (assumed to belong to the uncertainty set).

The question of how to solve RMDPs and the related computational complexity are complementary, but different from Theorems 4.1and 5.1. Indeed, an important point that differentiates us from [Panaganti and Kalathil, 2022] is the use of a robust optimization oracle . In (model-based) sample complexity analysis, the goal is to determine the smallest sample size N such that a planner executed in ̂ M yields a near-optimal policy in the RMDP M . To decouple the statistical and computational aspects of planning with respect to an approximate model ̂ M , we will use an optimization oracle that takes as input an (empirical) RMDP and returns a policy ˆ π that satisfies ‖ ˆ Q ∗ -ˆ Q ˆ π ‖ ∞ ≤ /epsilon1 opt . Our final bound will depend on /epsilon1 , the error made from finite sample complexity, and /epsilon1 opt . In practice, the error /epsilon1 opt is typically decreasing at a linear speed of γ k at the k th iteration of the algorithm, as in classical MDPs because (optimal) Bellman operators are γ -contraction in both classic and robust settings when robust kernel in assuming in the simplex.

The computational cost of RMDPs is addressed by Iyengar [2005] but not in the L p . Kumar et al. [2022] address this question, in this case, using the regularized form of robust MDPs obtained with relaxed hypothesis on the kernel (See Appendix A.2). The conclusions of the latter are that L p robust MDPs are computationally as easy as non-robust MDPs for regularized forms, at least for some choices of p for their relaxation. However, in their analysis, the use of γ -contraction of the Robust Bellman Operator is needed, whereas this is not always the case for sufficiently large β . Indeed, assuming robust kernel is not anymore in the simplex, Robust Bellman Operator is not anymore a γ -contraction but an /epsilon1 -contraction for /epsilon1 close to 1 and only for a small range of β . (See Derman et al. [2021]

Th. 5.1). We address the question of solving RMPDs in the L p case with a valid robust kernel in Alg. 1 as it is required to obtain an /epsilon1 ops solution in our analysis.

## 4 SAMPLE COMPLEXITY WITH L p -BALLS

The aim of this section is to obtain an upper-bound on the sample complexity of RMDPs. This result is true for sa -and s -rectangular sets and for any L p norm with p ≥ 1 .

Theorem 4.1. Assume δ &gt; 0 , /epsilon1 &gt; 0 and β &gt; 0 . Let ̂ π be any /epsilon1 opt -optimal policy for ̂ M , i.e. ‖ ̂ Q ̂ π -̂ Q /star ‖ ∞ ≤ /epsilon1 opt . With N calls to the sampler per state-action pair, such that N ≥ Cγ 2 L ′′ (1 -γ ) 4 /epsilon1 2 , with L ′′ = log( 32 SAN ‖ 1 s ‖ q δ (1 -γ ) ) we obtain the following guarantee for policy ˆ π ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -δ , where C is an absolute constant. Finally, for N total = N |S||A| and H = 1 / (1 -γ ) , we get an overall complexity of

## 4.1 DISCUSSION

This result says that the policy ˆ π computed by the planner on the empirical RMDP ˆ M will be ( /epsilon1 opt + /epsilon1 ) -optimal in the original RMDP M . As explained before, 1 planning algorithms for RMDPs that guarantee arbitrary small /epsilon1 opt , such as robust value iteration considered by Panaganti and Kalathil [2022]. It will also apply to future planners, as long as they come with a convergence guarantee. The error term /epsilon1 is controlled by the number of samples: N tot = ˜ O ( H 4 | S || A | /epsilon1 -2 ) calls to the generative models allow guaranteeing an error /epsilon1 . This is a gain in terms of sample complexity of | S | compared to Panaganti and Kalathil [2022], for the sa -rectangular assumption. Our bound also holds for both s - and sa -rectangular uncertainty sets. Panaganti et al. [2022] do not study the s -rectangular case, while Yang et al. [2021] do, but have a worst dependency to | A | in this case. Their bounds also have additional dependencies on the size of the uncertainty set, which we do not have. We recall that we do not cover the same cases, we do not analyze the KL and Chi-Square robust set, while they do not analyze the L p robust set for p &gt; 1 . However, the above comparison holds for the total variation case that we have in common ( p = 1 ). These bounds are clearly stated in Table 1. In the non-robust setting, Azar et al. [2013] show that there exist MDPs where the sample complexity is at least ˜ Ω ( H 3 | A || S | /epsilon1 2 ) . Section 5 gives a new upper-bound in H 3 which matches this lower-bound for non-robust MDPs with an extra condition on the range of β (the uncertainty set should be small enough).

## 4.2 SKETCH OF PROOF

This first proof is the simpler one, it relies notably on Hoeffding's concentration arguments. We provide a sketch, the full proof can be found in Appendix B. The resulting bound is not optimal in terms of the horizon H , but it also does not impose any condition on the range of /epsilon1 or β , contrary to the (better) bound of Sec. 5. We would like to bound the supremum norm of the difference between the optimal Q-function and the one of the policy computed by the planner in the empirical RMDP, according to the true RMDP, ‖ Q ∗ -Q ˆ π ‖ ∞ . Using a simple decomposition and the fact that π ∗ is not optimal in the empirical RMDP ( ˆ Q π ∗ ≤ ˆ Q ∗ = ˆ Q ˆ π ∗ ), we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The second term is easy to bound, by the assumption of the planning oracle we have ‖ ˆ Q ∗ -ˆ Q ˆ π ‖ ∞ ≤ /epsilon1 opt. The two other terms are similar in nature. They compare the Qfunctions of the same policy (either π ∗ the optimal one of the original RMDP, or ˆ π the output of the planning algorithm) but for different RMPDs, either the original one or the empirical one. For bounding the remaining terms, we need to introduce the following notation. For any set D and a vector v , let define κ D ( v ) = inf { u /latticetop v : u ∈ D } . This quantity corresponds to the inf form of the robust Bellman operator. The following lemma provides a data-dependent bound of the two terms of interest.

Lemma 4.2. We have with P s,a defined in Assumption 3.1 and ˆ P s,a the robust set centered around the empirical MDPs that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For proving these inequalities, we rely on fundamental properties of the (robust) Bellman operator, such as γ -contraction. This lemma is written for sa -rectangular assumption but is also true for s -rectangular assumption, replacing notation of robust set P s,a by P s . Now, we need to bound the resulting terms, which is done by the following lemma.

Lemma 4.3. With probability at least 1 -δ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Again, this also holds for s -rectangular sets. This inequality relies on Hoeffding's based concentration argument coupled with absorbing MDPs of Agarwal et al. [2020] and smoothness of the L p norm. Putting everything together, we have just shown that :

<!-- formula-not-decoded -->

Solving in /epsilon1 for the second term of the right-hand side gives the stated result as the term proportional to 1 /N is small compared to the second one for sufficiently small /epsilon1 .

## 5 TOWARDMINIMAX OPTIMAL SAMPLE COMPLEXITY

Now, we provide a better bound in terms of the horizon H , reaching (up to log factors) the lower-bound in H 3 for non-robust MDPs. Recall β = sup s,a β s,a for the sa -rectangular assumption or β = sup s β s for the s -rectangular assumption. For the following result to hold, we need to assume that the uncertainty set is small enough: we will require

<!-- formula-not-decoded -->

The following theorem is true for both sa -and s -rectangular uncertainty sets, and for any L p norm with p ≥ 1 .

Theorem 5.1. let β 0 ∈ (0 , 1 2( H -1) | S | 1 /q ] , for any κ &gt; 0 and any /epsilon1 0 ≤ κ √ H it exists a C β 0 ,/epsilon1 0 &gt; 0 independent of H such that for any β ∈ (0 , β 0 ) and any /epsilon1 ∈ (0 , /epsilon1 0 ) , whenever N the number of calls to the sampler per state-action pair satisfies N ≥ C β 0 ,/epsilon1 0 Lγ 2 H 3 /epsilon1 2 where L = log(8 |S||A| / ((1 -γ ) δ )) , it holds that if ̂ π is any /epsilon1 opt -optimal policy for ̂ M , that is when ‖ ̂ Q ̂ π -̂ Q /star ‖ ∞ ≤ /epsilon1 opt , then with probability at least 1 -δ .

<!-- formula-not-decoded -->

So N total = N |S||A| as an overall sample complexity for any /epsilon1 &lt; /epsilon1 0 .

<!-- formula-not-decoded -->

## 5.1 DISCUSSION

The constants of Theorem 5.1 are explicitly given in Appendix C. For instance, for β 0 = 1 8( H -1) and /epsilon1 0 = √ 16 H , we have C = 1024 , other choices being possible. Recall that in the non-robust case, the lower-bound is ˜ Ω ( H 3 | S || A | /epsilon1 2 ) Azar et al. [2013]. Our theorem states that any model-based robust RL approach, in the generative model setting, with an accurate enough planner applied to the empirical RMDP, reaches this lower bound, up to log terms. As far as we know, it is the first time that one shows that solving an RMDP in this setting does not require more samples than solving a non-robust MDP, provided that the uncertainty set is small enough. Our bound on /epsilon1 is similar to the one of Agarwal et al. [2020] in the robust case with their range [0 , √ H ) , we differ only by giving more flexibility in the choice of the constant C . The best range of /epsilon1 for non-robust MDPs is (0 , H ) [Li et al., 2020], we let its extension to the robust case for future work. So far, we discussed the lower-bound for the non-robust case, that we reach. Indeed, non-robust MDPs can be considered as a special case of MDPs with β = 0 . As far as we know, the only robust-specific lower-bounds on the sample complexity have been proposed by Yang et al. [2021]. They propose two lower-bounds accounting for the size of the uncertainty set, one for the Chi-square case, and one for the total variation case, which coincide with our L p framework for p = 1 This bound is

<!-- formula-not-decoded -->

This lower bound has two cases, depending on the size of the uncertainty set. If β ≤ (1 -γ ) = 1 /H , we retrieve the non-robust lower bound ˜ Ω ( |S||A| H 3 ε 2 ) . Therefore, for a L 1 -ball, our upper-bound matches the lower-bound, and we have proved that model-based robust RL in the generative model setting is minimax optimal for any accurate enough planner. Their condition for this bound, β ≤ 1 /H , is close to our condition, β &lt; 1 / (4( H -1) . This suggests that our condition on β is not just a proof artifact. In the second case, if β &gt; 1 -γ , the lower-bound is ˜ Ω ( |S‖A| (1 -γ ) ε 2 β 4 ) . In this case, our theorem does not hold, and we only currently get a bound in H 4 (see Sec. 4), which doesn't match this lower-bound.

In the case of TV , we know from posterior work Shi et al. [2023] that it is possible to get a tighter bound in the regime β &gt; 1 -γ but in the case of L P norm, it is still an open question. In the case where β is too large, the question arises whether RMDPs are useful as long as there is little to control when the transition kernel can be too arbitrary.

To sum up, to the best of our knowledge, with a small enough uncertainty set, our work delivers the first-ever minimax-optimal guarantee for RMDPs according to the non-robust lower-bound for L p -balls, and the first ever minimax-optimal guarantee according to the robust lowerbound for the total variation case for a sufficiently small radius of the uncertainty set, which has been later on the larger set of β by Shi et al. [2023]. '

## 5.2 SKETCH OF PROOF

The full proof is provided in Appendix C. As in Sec. 4.2, we start from the inequality

<!-- formula-not-decoded -->

where the second term of the right-hand side can again be readily bounded, ‖ ˆ Q ∗ -ˆ Q ˆ π ‖ ∞ ≤ /epsilon1 opt . To bound the remaining two terms, if we want to obtain a tighter final bound, the contracting property of the robust Bellman operator will not be enough, we need a finer analysis. To achieve this, we rely on the total variance technique introduced by Azar et al. [2013] for the non-robust case, combined with the absorbing MDP construction of Agarwal et al. [2020], also for the non-robust case, which allows improving the range of valid /epsilon1 . The key underlying idea is to rely on a Bernstein concentration inequality rather than a Hoeffding one, therefore considering the variance of the random variable rather than its range, tightening the bound. Working with a Bernstein inequality will require controlling the variance of the return. A key result was provided by Azar et al. [2013], that we extend to the robust setting,

<!-- formula-not-decoded -->

Naively bounding the left-hand side would provide a bound in H 2 , while this (non-obvious) bound in √ H 3 is crucial for obtaining on overall dependency in H 3 in the end. Now, we come back to the terms ‖ Q ∗ -ˆ Q ˆ π ∗ ‖ ∞ and ‖ Q ˆ π -ˆ Q ˆ π ‖ ∞ that we have to bound. This bound should involve a term proportional to ( I -γP π 0 ) -1 to leverage later Eq. (1). The following lemma is inspired by Agarwal et al. [2020], and its proof relies crucially on having a simple dual of robust Bellman operator.

## Lemma 5.2.

<!-- formula-not-decoded -->

We see that the term β appears in the bound. This comes from the need to control the difference in penalization between seminorms of value functions, from a technical viewpoint. Indeed, the terms 2 γβ 1 -γ ‖ Q π -ˆ Q π ‖ ∞ (with π being either ˆ π or π ∗ ) are not present in the non-robust version of the bound, and are one of the main differences from the derivation of Agarwal et al. [2020]. The first term of the righthand side of each bound ‖ ( I -γP π 0 ) -1 ( P 0 -̂ P ) ˆ V π ‖ ∞ (with π being either ˆ π or π ∗ , again) will be upper-bounded using a Bernstein argument, leveraging also Eq. (1). The resulting lemma is the following.

Lemma 5.3. With probability at least 1 -δ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For this result to be exploitable, we have to ensure that C N + C β &lt; 1 , which leads to β ≤ 1 -γ 2 γ | S | 1 /q , and then C N + C β &lt; 1 leads to a constraint on N in Theorem 5.1. Eventually, injecting the result of this last lemma in the initial bound, keeping the dominant term in 1 / √ N and solving for /epsilon1 provides the stated result, cf Appendix C.

## 6 CONCLUSION

In this paper, we have studied the question of the sample complexity of model-based robust reinforcement learning. To decouple this from the problem of exploration, we have considered the classic (in non-robust RL) generative model setting, where a sampler can provide next-state samples from the nominal kernel and from arbitrary state-action couples. We focused our study more specifically on sa - and s -rectangular uncertainty sets corresponding to L p -balls around the nominal.

Without any restriction on the size of uncertainty set ( β ), we have shown that the sample complexity of the studied general setting is ˜ O ( | S || A | H 4 /epsilon1 2 ) , already significantly improving existing results [Yang et al., 2021, Panaganti and Kalathil, 2022]. Our bound holds for both the sa - and s -rectangular cases, and improves existing results (for the total variation) by respectively | S | and | S || A | . By assuming a small enough uncertainty set, and for a small enough /epsilon1 , we further improved this bound to ˜ O ( | S || A | H 3 /epsilon1 2 ) , adapting proof techniques from the non-robust case [Azar et al., 2013, Agarwal et al., 2020]. This is a significant improvement. Our bound again holds for both the sa - and s - rectangular cases, it matches the lower-bound for the non-robust case Azar et al. [2013], and it matches the total variation lower-bound for the robust case when the uncertainty set is small enough [Yang et al., 2021]. We think this is an important step towards minimax optimal robust reinforcement learning.

There are a number of natural perspectives, such as knowing if we could extend our results to other kinds of uncertainty sets, or to extend our last bound to larger uncertainty sets (despite the fact that if the dynamics are too unpredictable, there may be little left to be controlled). Our results build heavily on the simple dual form of the robust Bellman operator, which prevents us from considering, for the moment, uncertainty sets based on the KL or Chi-square divergence. Beyond their theoretical advantages, these simple dual forms also provide practical and computationally efficient planning algorithms. Therefore, another interesting research direction would be to know if one could derive additional useful uncertainty sets relying primarily on the regularization viewpoint.

## 7 ACKNOWLEDGEMENTS

Pierre Clavier has been supported by a grant from Région Île-de-France.

## References

- Alekh Agarwal, Sham Kakade, and Lin F Yang. Modelbased reinforcement learning with a generative model is minimax optimal. In Conference on Learning Theory , pages 67-83. PMLR, 2020.
- Mohammad Azar, Rémi Munos, and Hilbert J Kappen. Minimax pac bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3):325-349, 2013.
- Pierre Clavier, Stéphanie Allassonière, and Erwan Le Pennec. Robust reinforcement learning with distributional risk-averse formulation. arXiv preprint arXiv:2206.06841 , 2022.
- Esther Derman, Matthieu Geist, and Shie Mannor. Twice regularized mdps and the equivalence between robustness and regularization. Advances in Neural Information Processing Systems , 34:22274-22287, 2021.
- Chin Pang Ho, Marek Petrik, and Wolfram Wiesemann. Partial policy iteration for l1-robust markov decision processes. J. Mach. Learn. Res. , 22:275-1, 2021.
- Wassily Hoeffding. Probability inequalities for sums of bounded random variables. In The collected works of Wassily Hoeffding , pages 409-426. Springer, 1994.
- Garud N Iyengar. Robust dynamic programming. Mathematics of Operations Research , 30(2):257-280, 2005.
- William Karush. Minima of functions of several variables with inequalities as side conditions. In Traces and emergence of nonlinear programming , pages 217-245. Springer, 2013.
- Tadashi Kozuno, Wenhao Yang, Nino Vieillard, Toshinori Kitamura, Yunhao Tang, Jincheng Mei, Pierre Ménard, Mohammad Gheshlaghi Azar, Michal Valko, Rémi Munos, et al. Kl-entropy-regularized rl with a generative model is minimax optimal. arXiv preprint arXiv:2205.14211 , 2022.
- Navdeep Kumar, Kfir Levy, Kaixin Wang, and Shie Mannor. Efficient policy iteration for robust markov decision processes via regularization. arXiv preprint arXiv:2205.14327 , 2022.
- Gen Li, Yuting Wei, Yuejie Chi, Yuantao Gu, and Yuxin Chen. Breaking the sample size barrier in model-based reinforcement learning with a generative model. Advances in neural information processing systems , 33: 12861-12872, 2020.
- Shie Mannor, Duncan Simester, Peng Sun, and John N Tsitsiklis. Bias and variance in value function estimation. In Proceedings of the twenty-first international conference on Machine learning , page 72, 2004.
- Colin McDiarmid et al. On the method of bounded differences. Surveys in combinatorics , 141(1):148-188, 1989.
- Arnab Nilim and Laurent El Ghaoui. Robust control of markov decision processes with uncertain transition matrices. Operations Research , 53(5):780-798, 2005.
- Charles Packer, Katelyn Gao, Jernej Kos, Philipp Krähenbühl, Vladlen Koltun, and Dawn Song. Assessing generalization in deep reinforcement learning. arXiv preprint arXiv:1810.12282 , 2018.
- Kishan Panaganti and Dileep Kalathil. Sample complexity of robust reinforcement learning with a generative model. In International Conference on Artificial Intelligence and Statistics , pages 9582-9602. PMLR, 2022.
- Kishan Panaganti, Zaiyan Xu, Dileep Kalathil, and Mohammad Ghavamzadeh. Robust reinforcement learning using offline data. arXiv preprint arXiv:2208.05129 , 2022.
- Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta. Robust adversarial reinforcement learning. In International Conference on Machine Learning , pages 2817-2826. PMLR, 2017.
- Martin L Puterman. Markov decision processes. Handbooks in operations research and management science , 2:331-434, 1990.
- Bruno Scherrer. Performance bounds for λ policy iteration and application to the game of tetris. Journal of Machine Learning Research , 14(4), 2013.
- Laixi Shi and Yuejie Chi. Distributionally robust modelbased offline reinforcement learning with near-optimal sample complexity. arXiv preprint arXiv:2208.05767 , 2022.
- Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, Matthieu Geist, and Yuejie Chi. The curious price of distributional robustness in reinforcement learning with a generative model. arXiv preprint arXiv:2305.16589 , 2023.
- Aaron Sidford, Mengdi Wang, Xian Wu, Lin Yang, and Yinyu Ye. Near-optimal time and sample complexities for solving markov decision processes with a generative model. Advances in Neural Information Processing Systems , 31, 2018.
- Satinder P Singh and Richard C Yee. An upper bound on the loss from approximate optimal-value functions. Machine Learning , 16(3):227-233, 1994.
- Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction . MIT press, 2018.
- Takumi Tanabe, Rei Sato, Kazuto Fukuchi, Jun Sakuma, and Youhei Akimoto. Max-min off-policy actor-critic method focusing on worst-case robustness to model misspecification. Advances in Neural Information Processing Systems , 35:6967-6981, 2022.
- Roman Vershynin. High dimensional probability, 2017.
- J. von Neumann. Zur theorie der gesellschaftsspiele. Mathematische annalen , 100(1):295-320, 1928.
- Wolfram Wiesemann, Daniel Kuhn, and Berç Rustem. Robust markov decision processes. Mathematics of Operations Research , 38(1):153-183, 2013.
- Zaiyan Xu, Kishan Panaganti, and Dileep Kalathil. Improved sample complexity bounds for distributionally robust reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 97289754. PMLR, 2023.
- Wei H Yang. On generalized holder inequality. Nonlinear Analysis, Theory, Methods &amp; Applications , 16(5), 1991.
- Wenhao Yang, Liangyu Zhang, and Zhihua Zhang. Towards theoretical understandings of robust markov decision processes: Sample complexity and asymptotics. arXiv preprint arXiv:2105.03863 , 2021.
- Chenyang Zhao, Olivier Sigaud, Freek Stulp, and Timothy M Hospedales. Investigating generalisation in continuous deep reinforcement learning. arXiv preprint arXiv:1902.07015 , 2019.

## Towards Minimax Optimality of Model-based Robust Reinforcement Learning (Supplementary Material)

Pierre Clavier 1,2,3

Erwan Le Pennec 1

Matthieu Geist 4

1 CMAP, CNRS, Ecole Polytechnique, Institut Polytechnique de Paris, 91120 Palaiseau, France, 2 INRIA Paris, HeKA, France,

4 Cohere.

3 Centre de Recherche des Cordeliers, INSERM, Universite de Paris, Sorbonne Universite, F-75006 Paris, France,

## A OVERVIEW AND USEFUL INEQUALITIES

The appendix is organized as follows

- In Appendix A.1, a comprehensive table with state-of-the-art complexity for every distance.
- In Appendix A.2, we provide more details/explanations on the difference between our formulation on the one of Kumar et al. [2022] and Derman et al. [2021].
- In Appendix A.3, we give more details about our algorithm : DRVI L P
- In Appendix A.4, we give some useful inequalities frequently used in the proofs.
- In Appendix B, we prove Theorem 4.1.
- In Appendix C, we prove Theorem 5.1.

Finally, the proofs for the s -rectangular and sa -rectangular cases are often very similar. If this is true, we will combine them in a single proof with the two cases detailed when needed.

## A.1 TABLE OF SAMPLE COMPLEXITY

Table 2: Sample Complexity for different metric and s - or sa rectangular assumptions with β the radius of uncertainty set, H the horizon factor, /epsilon1 the precicion, ¯ p , β 0 ,p = (1 -γ ) / (2 γ | S | 1 /q ) . the smallest positive state transition probability of the nominal kernel visited by the optimal robust policy (see Yang et al. [2021]).

|            | Panaganti and Kalathil [2022]                              | Yang et al. [2021]                                                         | Shi and Chi [2022]                                          | Our β ≥ 0                                                | Our β 0 ,p > β > 0                                       | Shi et al. [2023] β > 1 - γ                              | Shi et al. [2023] 0 < β < 1 - γ                         |
|------------|------------------------------------------------------------|----------------------------------------------------------------------------|-------------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|---------------------------------------------------------|
| TV ( sa )  | ˜ O ( &#124; S &#124; 2 &#124; A &#124; H 4 /epsilon1 2 )  | ˜ O ( &#124; S &#124; 2 &#124; A &#124; H 4 (2+ β ) 2 /epsilon1 2 β 2 )    | ×                                                           | ˜ O ( &#124; S &#124;&#124; A &#124; H 4 /epsilon1 2 ) ) | ˜ O ( &#124; S &#124;&#124; A &#124; H 3 /epsilon1 2 )   | ˜ O ( &#124; S &#124;&#124; A &#124; H 2 /epsilon1 2 β ) | ˜ O ( &#124; S &#124;&#124; A &#124; H 3 /epsilon1 2 )  |
| TV ( s )   | ×                                                          | ˜ O ( &#124; S &#124; 2 &#124; A &#124; 2 H 4 (2+ β ) 2 /epsilon1 2 β 2 )  | ×                                                           | ˜ O ( &#124; S &#124;&#124; A &#124; H 4 /epsilon1 2     | ˜ O ( &#124; S &#124;&#124; A &#124; H 3 /epsilon1 2 ) ) | ×                                                        | ×                                                       |
| Lp ( sa )  | ×                                                          | ×                                                                          | ×                                                           | ˜ O ( &#124; S &#124;&#124; A &#124; H 4 /epsilon1 2 )   | ˜ O ( &#124; S &#124;&#124; A &#124; H 3 /epsilon1 2     | ×                                                        | ×                                                       |
| Lp ( s )   | ×                                                          | ×                                                                          | ×                                                           | ˜ O ( &#124; S &#124;&#124; A &#124; H 4 /epsilon1 2 )   | ˜ O ( &#124; S &#124;&#124; A &#124; H 3 /epsilon1 2 )   | ×                                                        | ×                                                       |
| χ 2 ( sa ) | ˜ O ( &#124; S &#124; 2 &#124; A &#124; βH 4 /epsilon1 2 ) | ˜ O ( &#124;S&#124; 2 &#124;A&#124; (1+ β ) 2 H 4 ε 2 ( √ 1+ β - 1) 2 )    | ×                                                           | ×                                                        | ×                                                        | ˜ O ( &#124; S &#124;&#124; A &#124; βH 4 /epsilon1 2 )  | ˜ O ( &#124; S &#124;&#124; A &#124; βH 4 /epsilon1 2 ) |
| χ 2 ( s )  | ×                                                          | ˜ O ( &#124;S&#124; 2 &#124;A 3 &#124; (1+ β ) 2 H 4 ε 2 ( √ 1+ β - 1) 2 ) | ×                                                           | ×                                                        | ×                                                        |                                                          | ×                                                       |
| KL ( sa )  | ˜ O ( &#124;S&#124; 2 &#124;A&#124; exp( H ) H 4 β 2 ε 2 ) | ˜ O ( &#124; S &#124; 2 &#124; A &#124; H 4 ¯ p 2 /epsilon1 2 β 2 )        | ˜ O ( &#124; S &#124;&#124; A &#124; H 4 ¯ p/epsilon1 2 β 4 | ×                                                        | ×                                                        | ×                                                        | ×                                                       |
| KL ( s )   | ×                                                          | ˜ O ( &#124; S &#124; 2 &#124; A &#124; 2 H 4 ¯ p 2 /epsilon1 2 β 2 )      | ×                                                           | ×                                                        | ×                                                        | ×                                                        | ×                                                       |

## A.2 RELATION WITH THE WORK OF Kumar et al. [2022] AND Derman et al. [2021]

In the work of Derman et al. [2021] close forms for RMDPs with L p norms are derived assuming the following uncertainty set :

Assumption A.1. ( sa -rectangularity in Derman et al. [2021])

<!-- formula-not-decoded -->

Using these uncertainty sets leads to the following Bellman Operator :

Theorem A.2 (Derman et al. [2021]) . The sa -rectangular Robust Bellman operator is equivalent to a regularized nonrobust Bellman operator: for r s,a V,π ( s, a ) = -( α s + γβ s,a ‖ V ‖ q ) + R 0 ( s, a ) as we have

<!-- formula-not-decoded -->

Using this formulation, they get a closed form for the inner minimization problem and for the Robust Bellman Operator

The work Kumar et al. [2022] modifies the work of Derman et al. [2021] using Kernel that sum to 1 , ∑ s ′ P s,a ( s ′ ) = 0 in their definition, but using this uncertainty set, it is still possible to get a robust kernel out of the simplex. Using this formulation, they also get a closed form for the inner minimization problem and for the Robust Bellman Operator.

Assumption A.3. ( sa -rectangularity in Kumar et al. [2022])

<!-- formula-not-decoded -->

Using these uncertainty sets where robust Kernel may not belong anymore to the simplex as they do not assume P 0 + P s,a ≥ 0 . This leads to the following Bellman Operator :

Theorem A.4 (Kumar et al. [2022]) . The sa -rectangular Robust Bellman operator is equivalent to a regularized non-robust Bellman operator: for r s,a V,π ( s, a ) = -( α s + γβ s,a sp q ( V ) ) + R 0 ( s, a ) , as we have

<!-- formula-not-decoded -->

where sp q ( V ) in defined in Def. 3.1.These results are due to the following lemma.

Lemma A.5 ( Kumar et al. [2022]. Duality for the minimization problem for sa rectangular case with L p norm without simplex constrain) .

Our analysis assumes the positivity of the kernel function, P 0 + P s ≥ 0 in s-rectangular or P 0 + P s,a ≥ 0 for sa -rectangular case. Using this more realistic assumption, we can not obtain a closed form of the robust Bellman operator. However, we are still able to compute a dual form for the inner minimization problem of RMDPs. With our definition of rectangularity in the simplex:

<!-- formula-not-decoded -->

Assumption A.6. ( sa -rectangularity) We define sa -rectangular L p -constrained uncertainty set as

<!-- formula-not-decoded -->

and using κ D ( v ) = inf { u /latticetop v : u ∈ D } . , we obtain :

Lemma A.7 (Duality for the minimization problem for sa rectangular case with L p norm) .

<!-- formula-not-decoded -->

Proof can be found on Appendix B.5

Contrary to previous lemma in Kumar et al. [2022], there is an additional max operator in our dual formulation. Interestingly, their formulation is a relaxation of our Lemmas 3.3 as their formulation does not assume the positivity of the kernel. Their relaxation allows practical algorithms with close form, but still suffer from non-exact formulation of RMDPs with robust Kernel that are not in the simplex.

One crucial point in our analysis is that Bellman Operator for RMDPs is a γ - contraction for robust kernel in the simplex for any radius β (see Iyengar [2005]). For Kumar et al. [2022] and Derman et al. [2021] the range of β where their Robust Bellman Operator is a contraction is smaller than 1 -γ γ | S | 1 /q (see Proposition 4 of Derman et al. [2021]) which is the range where we have minimax optimality in our Theorem 5.1. For β &gt; 1 -γ γ | S | 1 /q , there is no contraction anymore. In the following, we will assume that robust kernels belong to the simplex to use γ -contraction in our proof of sample complexity and ensure convergence of the following Distributionally Robust value Iteration for L p norms for any β Algoritm 1.

## A.3 MODELBASED DRVI L P ALGORITHM

```
Algorithm 1: DRVI L P : Distributionally robust value iteration DRVI for L P norms with sa -rectangular assuptions 1 input: empirical nominal transition kernel ̂ P 0 ; reward function r ; uncertainty level β . 2 initialization: ̂ Q 0 ( s, a ) = 0 , ̂ V 0 ( s ) = 0 for all ( s, a ) ∈ S × A . 3 for t = 1 , 2 , · · · , T do 4 for ∀ s ∈ S, a ∈ A do 5 Set ̂ Q t ( s, a ) according to (2) for sa -rectangular ; 6 for ∀ s ∈ S do 7 Set ̂ V t ( s ) = max a ̂ Q t ( s, a ) ; 8 output: ̂ Q T , ̂ V T and ̂ π obeying ̂ π ( s ) = arg max a ̂ Q T ( s, a ) .
```

We propose Alg. 1 to solve robust MDPs in the case of L P norms using value Iteration with sa - rectangularity assumptions. First, we can remark that directly solving classical RMDPs formulation is computationally costly as it requires an optimization over an S -dimensional probability simplex at each iteration, especially when the dimension of the state space S is large. However, using strong duality like Iyengar [2005] for the TV , one can also solve using the dual problem of this formulation. The equivalence between the two formulations can be found in Lemma 3.3. Using the dual form, the optimization (3) reduces to a 2-dimensional optimization problem that can be solved efficiently using any 2 -dimensional convex solver if there exists an analytic form of the span-semi norm. Then the iterates { ̂ Q t } t ≥ 0 of DRVI for L P norms converge linearly to the fixed point ̂ Q /star , owing to the appealing γ -contraction property of robust MDPs in the simplex. From an initialization ̂ Q 0 = 0 , the update rule at the t -th ( t ≥ 1 ) iteration can be formulated as for sa -rectangular case as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the variational family A λ,ω ˆ P is a 2 -dimensional variational family defined in (8). The specific form of the dual problem depends on the choice of the norm. In the case of L 1 , L 2 , or L ∞ , span semi-norms involved in dual problems have closed form (respectively equals to median, variance, or span), and equation 3 corresponds to a 2 -D minimization problem.

But in general cases, one has to compute span-semi norms that can be easily computed using binary search solving

<!-- formula-not-decoded -->

to compute ω q and then setting the semi norm sp q ( v ) = ‖ v -ω q ‖ . Recall the q -variance function sp q : S → R and q -mean function ω q : S → R be defined as

<!-- formula-not-decoded -->

See Kumar et al. [2022] for discussion about computing span semi norms. So in the general case, we can also compute the maximum solving :

<!-- formula-not-decoded -->

Finally, in the sa -case we compute the best policy which is the greedy policy of the final Q-estimates ̂ Q T as the final policy ̂ π :

Using any 2 -D convex optimization algorithm solves the problem as this problem is jointly concave in ( λ, w ) because ( λ, w ) →-∥ ∥ ∥ [ ̂ V t -1 ] α λ,ω ˆ P -w ∥ ∥ ∥ q is concave using norm property and ( λ, w ) → ̂ P [ ̂ V t -1 ] α λ,ω ˆ P also. Then the sum is concave.

<!-- formula-not-decoded -->

## A.4 USEFUL INEQUALITIES AND NOTATIONS

Here we present some useful inequalities used frequently in the derivation. Consider any P a transition matrix and β s for s rectangular uncertain sets or β sa for sa - uncertainty sets, then for I = (1 , 1 , ..., 1) /latticetop :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Eq. (4) is true, taking the supremum norm of the left-hand side inequality. Eq. (5) and Eq. (6) come from properties of norms, see Eq. (1) from Scherrer [2013].

Finally we denote the truncation operator for a vector α ∈ R S ,

<!-- formula-not-decoded -->

## A.5 ROBUST BELLMAN OPERATOR AND ROBUST Q VALUES

This is proof of Lemma 3.5:

Lemma A.8. Robust Bellman Operator for sa -and s -rectangular are :

<!-- formula-not-decoded -->

Proof. For sa -rectangular: by rectangularity

<!-- formula-not-decoded -->

For s -rectangular case :

<!-- formula-not-decoded -->

where (a) comes from Holder's inequality.

Lemma A.9. For sa -and s -rectangular,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The result comes directly as for sa -rectangular the following relations hold,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then using fixed point equation of Bellman operator: T π U s p V π s ( s ) = V π s ( s ) or T π U sa p V π sa ( s ) = V π sa ( s ) and previous Lemma A.8 for the expression of T π U s p V π s ( s ) , we can identify the robust Q values that give the result

## B AN H 4 BOUND FOR L p -BALLS

To lighten notations, we remove subscript s in most places and denote for example V π instead of V π s for s -rectangular sets.

Lemma B.1 (Decomposition of the bound) .

<!-- formula-not-decoded -->

with and for s -rectangular case

Proof.

<!-- formula-not-decoded -->

This decomposition is the starting point of our proofs for both Theorems 4.1 and 5.1. In this decomposition, the second term satisfies ‖ Q ∗ -ˆ Q π ∗ ‖ ∞ ≤ /epsilon1 opt by definition. This term goes to 0 exponentially fast as the robust Bellman operator is a γ -contraction. The two last terms ‖ Q ∗ -ˆ Q π ∗ ‖ ∞ and ‖ ˆ Q ˆ π -Q ˆ π ‖ ∞ need to be controlled using concentration inequalities between the true MDP and the estimated one. To do so, we need concentration inequalities such as the following Lemma B.2.

Lemma B.2 (Hoeffding's inequality for V ) . For any V ∈ R |S| with ‖ V ‖ ∞ ≤ H , with probability at least 1 -δ , we have

Proof. For any ( s, a ) pair, assume a discrete random variable taking value V ( i ) with probability P 0 ,s,a ( i ) for all i ∈ { 1 , 2 , · · · , |S|} . Using Hoeffding's inequality [Hoeffding, 1994] and ‖ V ‖ ∞ ≤ H :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, taking ε = H √ 2 log(2 |S||A| /δ ) N , we get

Finally, using a union bound:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This completes the concentration proof. Next we will look at the contraction argument of the robust Bellman operator.

Lemma B.3 (Contraction of infimum operator) . For D = P s,a or P s , the function is 1-Lipchitz.

Proof. We have that

<!-- formula-not-decoded -->

Then ∀ ε &gt; 0 , there exists P s,a ∈ P s,a such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using those two properties,

<!-- formula-not-decoded -->

where we used the Holder's inequality. Since ε is arbitrary small, we obtain, κ P s,a ( V 1 ) -κ P s,a ( V 2 ) ≤ ‖ V 1 -V 2 ‖ . Exchanging the roles of V 1 and V 2 give the result.

The proof is similar for P s .

Note that an immediate consequence is the already known γ - contraction of the robust Bellman operator.

<!-- formula-not-decoded -->

Proof. For the first inequality, since we can rewrite the robust Q-function for any uncertainty sets on the dynamics as Q ˆ π ( s, a ) = r -α s,a + γκ P 0 ,s,a ( V ˆ π ) (see Eq. (3.5)), or replacing α s,a by α s ( ˆ π s ( a ) ‖ ˆ π s ‖ q ) q -1 in the s - rectangular case:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with P s,a defined in Assumption 3.1 and ˆ P s,a with the same definition but centered around the empirical MDP. Hence, taking the supremum norm ‖ . ‖ ∞ ,

∥ ∥ ∣ ∣ Line (a) comes from the rectangularity assumption, (b) uses the triangular inequality and the 1-contraction of the infimum in Lemma B.3, (c) uses the fact that ‖ V π -V π ‖ ∞ ≤ ‖ Q π -Q π ‖ ∞ for any π . As 1 -γ &lt; 1 , we get the first stated result.

<!-- formula-not-decoded -->

̂ ̂ One can note that the proof is true for any policy, so it is also true for both ˆ π and π ∗ which concludes the proof. This proof is written for the sa -rectangular assumption, it is also true for the s -rectangular case with slightly different notations, replacing D = P 0 ,s,a by D = P 0 ,s . Now we need to find new form for κ for both s and sa rectangular assumptions.

For the second claim, we are using a slightly different modification:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

using the same arguments as in the first inequality. Solving gives the result.

We denote [ V ] α as its clipped version by some non-negative vector α , namely,

<!-- formula-not-decoded -->

Defining the gradient of P ↦→‖ P ‖ as ∇‖ P ‖ , λ &gt; 0 , a positive scalar and ω is the generalized mean defined as the argmin in the definition of the span semi norm in Def.3.1, we derive two optimization lemmas.

Lemma B.5 (Duality for the minimization problem for sa rectangular case.) . Denoting ̂ P the vector ̂ P s,a or P 0 for P 0 ,s,a ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(10)

where

<!-- formula-not-decoded -->

and with [ V ] α := { α ( s ) , if V ( s ) &gt; α ( s ) V ( s ) , otherwise.

For L 1 or TV , case , the vector α λ,ω P reduces to a 1 dimensional scalar such as α ∈ [0 , 1 / (1 -γ )] .

Proof. First, we will show that

The second equation of this lemma is the same as the first one, replacing the center of the ball constrain ̂ P s,a by P 0 ,s,a and ˆ π by π ∗ . By definition,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use the change of variable y ( s ′ ) = P ( s ′ ) -ˆ P ( s ′ ) . Then writing the Lagrangian we get for µ ∈ R | S | + , γ ∈ R the Lagrangian variables:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (a) is true using the equality case of Holder's inequality and (b) is the definition of the span semi-norm (see Def. 3.1). The value that maximizes the inner maximization problem in 12 in ν is the q -mean (see Def. 3.1) by definition denoted ω . Now the aim is to prove that

<!-- formula-not-decoded -->

First, as the norm is differentiable (which true for L p , p ≥ 2 ), we have that the equality (a) comes from the generalized Holder's inequality for arbitrary norms Yang [1991], namely, defining z = ( ˆ V ˆ π -µ -ω ) , it satisfies

<!-- formula-not-decoded -->

The quantity ν is replaced by the generalized mean for equality in (b) while (14) comes from Yang [1991]. Using complementary slackness Karush [2013]we define B = { s ∈ S : µ ( s ) &gt; 0 }

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which leads to the following equality by plugging the previous (15) in (14) and defining z ∗ = ˆ V ˆ π -µ ∗ -ω :

or by letting λ = ‖ z ∗ ‖ q ∈ R + . Note that for s ∈ B , ∇‖ y ‖ p = ∇‖ P ‖ p only depends on P ( s ) and not on other coordinates due to definition of L p norm.

We can remark that v -µ ∗ is P dependent, but if P is known, the best µ ∗ is only determined by one 2 dimensional parameters λ = ‖ v -µ ∗ -ν ‖ q and ω ∈ R + . Moreover, when ̂ P is fixed, the scalar ω is a constant is fully determined by P , v and µ ∗ . This is why the quantity defined α λ ̂ P varies through 2 parameter λ and ω . Given this observation, we can rewrite the optimization problem as :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we defined the maximization problem on µ not in R S but at the optimal in the variational family denote M λ,ω P = { µ λ,ω P = ˆ V ˆ π -α λ,ω P , λ, ω ∈ R + , P ∈ ∆( S ) , µ ∈ R S + , µ λ,ω P = [ 0 , 1 1 -γ ] S } .

We can rewrite the optimization problem in terms of α P with

<!-- formula-not-decoded -->

Note that for TV or L 1 , this lemma holds, but the vector α λ,ω ̂ P reduces to a positive scalar denoted α which is equal to ∥ ∥ ∥ ˆ V ˆ π -µ ∗ ∥ ∥ ∥ ∞ according to Iyengar [2005]. The thing which is of capital importance is that the second part of the equation sp q ([ ˆ V ˆ π ] α ) does not depend on ̂ P .

Lemma B.6 (Duality for the minimization problem for s rectangular case.) . Considering a projection matrix associated with a given policy π such that P π s ( s ′ ) = ∑ a π ( a | s ) P s,a ( s ′ ) and denoting ̂ P π ∈ R s the vector ̂ P π s ( . ) or P π 0 for P π 0 ,s ( . ) , we have:

with [ V ] α ( s ) := α ( s ) , if V ( s ) V ( s ) , otherwise.

<!-- formula-not-decoded -->

Proof. The second equation is the same replacing the center of the ball constrain ̂ P π s by P π 0 and ˆ π by π ∗ . By definition,

<!-- formula-not-decoded -->

where we use the change of variable y ( s ′ ) = P s,a ( s ′ ) -ˆ P s,a ( s ′ ) in (a). Then we case use the previous lemma for sa rectangular assumption, Lemma 3.3. Then,

<!-- formula-not-decoded -->

we can exchange the min and the max as we get concave-convex problems in β s,a and µ , ([von Neumann, 1928]) in the second line and using Holder's inequality in the last line. Finally, we obtain:

<!-- formula-not-decoded -->

where in (a) we use Lemma 3.3. Second claim is the same replacing ˆ V ˆ π by V ∗ , ˆ π by π ∗ and ˆ P by P 0 . Then we derive a new decomposition of the difference the two minimum.

Lemma B.7. For s and sa rectangular assumptions,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

where in the first equality we use Lemma B.5. The final inequality is a consequence of the 1 -Lipschitzness of the max operator. Taking the supremum over s, a gives the result. Replacing V ∗ by ˆ V ˆ π gives the other inequality. The result for s rectangular are the same as

<!-- formula-not-decoded -->

Note that at this point, quantities for s and sa rectangular is the same as the part with span semi norms cancelled. Now, note that the main problem is that we can not apply classical Hoeffding's inequality as ̂ P is dependent of data as ˆ V ˆ π . We need to decouple ˆ V ˆ π using s absorbing MDPS as in Agarwal et al. [2020] but using Hoeffding arguments. First, we will use a concentration for V ∗ .

Lemma B.8. For sa and s -rectangular, with probability 1 -δ , it holds:

with L = log(18 ‖ 1 ‖ q SAN/δ )

<!-- formula-not-decoded -->

Proof. First, we can use previous Lemma B.7

<!-- formula-not-decoded -->

First, we control g s,a ( α λ,ω P , V ∗ ) . To do so, we use for a fixed α λ,ω P and any vector V ∗ that is independent with ̂ P 0 , the Hoeffding's inequality, one has with probability at least 1 -δ with sa -rectangular notations,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Once pointwise concentration derived, we will use uniform concentration to yield this lemma. First, union bound, is obtained noticing that g s,a ( α λ,ω P , V ∗ ) is 1 -Lipschitz w.r.t. λ and ω as it is linear in λ and ω . Moreover, λ ∗ = ‖ V ∗ -µ ∗ -ω ‖ q obeying λ ∗ ≤ ‖ 1 ‖ q 1 -γ . The quantity ω ∈ [0 , 1 / (1 -γ )] as it is always smaller that V ∗ by definition. We construct then a 2 -dimensional a ε 1 -net N ε 1 over λ ∗ ∈ [0 , ‖ 1 ‖ q 1 -γ ] and ω ∈ [0 , 1 / (1 -γ )] whose size satisfies | N ε 1 | ≤ ( 3 ‖ 1 ‖ q ε 1 (1 -γ ) ) 2 [Vershynin, 2017]. Using union bound and (32), it holds with probability at least 1 -δ SA that for all λ ∈ N ε 1 ,

Using the previous equation and also (27), it results in using notation log( 18 SAN δ ) = L ,

<!-- formula-not-decoded -->

where (a) is because the optimal α ∗ falls into the ε 1 -ball centered around some point inside N ε 1 and g s,a ( α λ P , V ∗ ) is 1 -Lipschitz with regard to λ and ω , (b) is due to Eq. (33), (c) arises from taking ε 1 = log( 2 SA | Nε 1 | δ ) 3 N (1 -γ ) , (d) is verified by | N ε 1 | ≤ ( 3 ‖ 1 ‖ q ε 1 (1 -γ ) ) 2 ≤ 9 N ‖ 1 ‖ q and that variance of a ceiling function of a vector is smaller than the variance of non-ceiling vector.

For L p with p ≥ 2 , contrary to the previous term, the second term g s,a ( α λ ˆ P , V ) is more difficult as we need concentration, but there is an extra dependency in the data thought the parameter α λ ˆ P . Note that this term does not exist as α is a constant for TV . We need to decouple this problem using absorbing MDPs. Then it leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the first equality, we add the term µ λ,ω P 0 ,s,a to retrieve the previous concentration problem, fixing P 0 ,s,a and optimizing λ, ω . In the second, we extend the max using triangular inequality. The first term in the last equality is exactly the term we have controlled previously, while the second one needs more attention. We decouple the dependency of the data, and then controlling the difference between the µ . Then using the characterization of the optimal µ from equation (17):

<!-- formula-not-decoded -->

As the norm is C 2 for p ≥ 2 , using Mean value theorem, we know that

<!-- formula-not-decoded -->

For L p = ‖ x ‖ p norms, p ≥ 2 , we have simple taking derivative twice:

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

and L p the norm, where Diag is the diagonal matrix. However, as x ≤ L p , A ≤ I , we get

<!-- formula-not-decoded -->

where the 1 /L p is minimized for the uniform distribution. Then using Cauchy-Swartz inequality, it holds

<!-- formula-not-decoded -->

Then the question is how to bound the quantity ∥ ∥ ∥ ( P 0 ,s,a -̂ P 0 ,s,a )∥ ∥ ∥ 2 2 . To do so, we will use Mac Diarmid inequality. Definition B.1. Bounded difference property

A function f : X 1 × . . . X n → R satisfies the bounded difference property if for each i = 1 , . . . , n the change of coordinate from s i to s ′ i may change the value of the function at most on c i

<!-- formula-not-decoded -->

In our case, we consider f ( X 1 , . . . , X n ) = ‖ ∑ n k =1 X k ‖ 2 . Then we can notice that by triangle inequality for any x 1 , . . . , x n and x ′ k with X i,s ′ = P i 0 ,s,a ( s ′ ) -P 0 ,s,a ( s ′ ) ( index i holds for index of sample generated from the generative model) that

<!-- formula-not-decoded -->

Theorem B.9. (McDiarmid's inequality). McDiarmid et al. [1989] Let f : X 1 × . . . X n → R be a function satisfying the bounded difference property with bounds c 1 , . . . , c n . Consider independent random variables X 1 , . . . , X n , X i ∈ X i for all i . Then for any t &gt; 0

<!-- formula-not-decoded -->

Using McDiarmid's inequality and union bound, we can bound the term as here

<!-- formula-not-decoded -->

with probability 1 -δ/ ( | S || A | ) . Moreover, the additional term can be bounded as follows:

<!-- formula-not-decoded -->

with X i,s ′ = P i 0 ,s,a ( s ′ ) -P 0 ,s,a ( s ′ ) is one sample sampled from the generative model. Then

<!-- formula-not-decoded -->

where (a) the last equality comes from the independence of the random variables and where the last inequality comes from the fact the maximum of two elements in the simplex is bounded by 2 . Finally, regrouping the two terms, we obtain with probability 1 -δ/ ( | S || A | ) :

<!-- formula-not-decoded -->

with L ′ = 6 log( | S || A | / ( δ )) . Finally, plugging the previous equation in (41):

<!-- formula-not-decoded -->

This term can be easily controlled by taking the supremum over λ which is a 1 dimensional parameter. Then we can bound λ ∈ [0 , H ‖ 1 S ‖ q ] . Indeed,

<!-- formula-not-decoded -->

Finally, we obtain:

Regrouping all terms:

<!-- formula-not-decoded -->

(43)

For the specific case of TV which is not C 2 smooth, this lemma still holds as in (27), we only need to control one term without the dependency on data in the supremum as α λ P reduces to a scalar α which does not depend on P . Then extra decomposition using smoothness of the norm is not needed, as the only remaining term in the max in (27) is the left hand side term.

## Lemma B.10 ( s -absorbing MDPs for Hoeffding's concentration Inequalities) .

As in Agarwal paper Agarwal et al. [2020], we define for a state s and a scalar u , the MDP called M s,u such that: M s,u is identical to M except that state s is absorbing in M s,u , i.e. P M s,u ( s | s, a ) = 1 for all a , and the reward at state s in M s,u is (1 -γ ) u . The remainder of the transition model and reward function are identical to those in M . In the following, we will use V π s,u to denote the value function V π M s,u and correspondingly for Q and reward and transition functions to avoid notational clutter. Then, we have that for all policies π :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

because s is absorbing with reward (1 -γ ) u . For some state s , we will only consider the MDP M s,u for u in a finite set U s with

<!-- formula-not-decoded -->

with ∆ δ,N := γ (1 -γ ) 2 ( 2 √ L 2 N + 2 L | S | 1 /q ‖ 1 S ‖ q ( p -1) N ) The set U s consists of evenly spaced elements in this interval, where we set the size of | U s | appropriately later on. As before, we let ̂ M s,u denote the MDP that uses the empirical model ̂ P instead of P , at all non-absorbing states and abbreviate the value functions in ̂ M s,u as ̂ V π s,u . Then we have for a fix a state s , action a , a finite set U s , and δ ≥ 0 , that for all u ∈ U s : with probability greater than 1 -δ , it holds :

<!-- formula-not-decoded -->

Lemma B.11 (Agarwal et al. [2020], Lemma 7) . Let u ∗ = V /star M ( s ) and u π = V π M ( s ) . We have

This is exactly B.8 in equation (27) to the finite set U s as now V ˆ π u and ̂ P s,a are now independent.

<!-- formula-not-decoded -->

Proof can be found in Agarwal et al. [2020], Lemma 7.

Lemma B.12. For any u, u ′ , s and policy π :

<!-- formula-not-decoded -->

Proof. To obtain the result in our robust MDP setting, we need a similar stability property like in Lemma 8 of Agarwal et al. [2020], but for the robust value functions. It turns out that this a direct consequence of the property for classical MDP. Agarwal in Agarwal et al. [2020] show equation 45 for classical MPDs, then we have for RMDPs:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which concludes the proof for RMDPs.

Lemma B.13 (Hoeffding's Concentration for dependent variables) . Removing s, a notations for kernels,

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) is 44 or Hoeffding's inequality for s-absorbing MDPs. By Lemmas B.11 and B.12,

<!-- formula-not-decoded -->

which is point ( b ) . The last min operator in the result comes from the fact that the previous equation holds for all u ∈ U s , we take the best possible choice, which completes the proof of the first claim.

Lemma B.14 (Crude bound for Robust MDPs) . This lemma is needed for next Lemma B.15 but the proof differs from the classical MDP setting. For s and sa rectangular assumptions,

<!-- formula-not-decoded -->

Proof. For the first claim :

<!-- formula-not-decoded -->

where we use contraction of κ , lemma B.3 in (a) and ∥ ∥ ∥ Q π -ˆ Q π ∥ ∥ ∥ ∞ ≤ ∥ ∥ ∥ V π -ˆ V π ∥ ∥ ∥ ∞ in (c) for any π . Solving we get :

Then using Lemma B.7, we obtain :

<!-- formula-not-decoded -->

Taking π = π ∗ , V π ∗ is independent of the data and we can use Lemma B.8. Finally, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(56)

For the second point, using s or sa rectangular assumptions,

Then using Lemma B.7, and solving we get :

<!-- formula-not-decoded -->

Finally using Lemma B.8, we obtain which concludes the proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma B.15 (Similar to Agarwal, Agarwal et al. [2020] lemma 9 but for RMPDs) . With probability 1 -δ , we have:

Proof. The proof can be found in Agarwal et al. [2020] and is similar for RMDs than for classical MPDs and consists in choosing U s to be the evenly spaced elements in the interval [ V /star ( s ) -∆ δ/ 2 ,N V /star ( s ) + ∆ δ/ 2 ,N ] , then finally the size of U s is chosen to be | U s | = 1 (1 -γ ) 2 . Using lemma , with probability greater than 1 -δ/ 2 , we have ̂ V /star ( s ) ∈ [ V /star ( s ) -∆ δ/ 2 ,N V /star ( s ) + ∆ δ/ 2 ,N ] for all s according to Lemma B.14. This implies using that that ˆ V π ∗ will land in one of | U s | -1 evenly sized sub-intervals of length 2∆ δ/ 2 ,N :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma B.16 (Relation between concentration of robust and non-robust MDPs) . With probability 1 -δ , we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with L ′′ = log( 32 SAN ‖ 1 ‖ q δ (1 -γ )

Proof. Using Lemma B.7, we directly have the first inequality equality part of the first statement:

is bounded by either by

<!-- formula-not-decoded -->

or

We know that in both cases that

<!-- formula-not-decoded -->

using | [ ˆ V ˆ π ] α λ,ω P |-| [ ˆ V ∗ ] α λ,ω P | ≤ | ([ ˆ V ˆ π -ˆ V ∗ ] α λ,ω P ) | ≤ | ( ˆ V ˆ π -ˆ V ∗ ) | and combining Lemma B.13 and B.15, for | U s | = 1 (1 -γ ) 2 , with probability 1 -δ , we have :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof is exactly the same by replacing ˆ π by π ∗ but without the 2 /epsilon1 opt , which gives the second stated result. Again, this proof is written for the sa -rectangular assumption, it is also true for the s -rectangular case with slightly different notations, replacing D = P 0 ,s,a by D = P 0 ,s .

These two inequalities are the core of our proof, as the closed form solution of the min problem in the robust setting only depends on α, β and the current value function.

<!-- formula-not-decoded -->

Theorem B.17. Suppose δ &gt; 0 , /epsilon1 &gt; 0 and β &gt; 0 , let ̂ π be any /epsilon1 opt -optimal policy for ̂ M , i.e. ∥ ∥ ∥ ̂ Q ̂ π -̂ Q /star ∥ ∥ ∥ ∞ ≤ /epsilon1 opt . If we get

with probability at least 1 -δ , where C is an absolute constant. Finally, for N total = N |S||A| and H = 1 / (1 -γ ) , we get an overall complexity of

Proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Inequality (a) is due to Lemma B.1. Inequality (b) comes from Lemma B.4. Finally, inequality (c) comes from Lemma B.16 and inequality (d) from the form of N in the theorem. For N ≥ H 4 SA , the second term proportional to 1 /N is very small compared to the asymptotic term in 1 / √ N for small /epsilon1 . Note that S 1 /q ‖ 1 S ‖ q = | S | for L 2 norm for example. This proof holds for both s - and sa -rectangular assumptions.

## C TOWARDS MINIMAX OPTIMAL BOUNDS

We start from the same decomposition as the proof of Theorem 4.1 proved in Lemma B.1:

<!-- formula-not-decoded -->

However, we need tighter concentration arguments for this proof.

In the following, we will frequently use the fact that, for any policy π , written below for the s -rectangular case (a similar expression can be obtained for the sa -rectangular case, adapting the regularized reward),

Recall, the fix point equation for Q π can be written as :

<!-- formula-not-decoded -->

It will be applied notably to ˆ π and π ∗ (recall that Q ∗ = Q π ∗ ), in the RMDP but also in the empirical one.

Lemma C.1. For s -rectangular we have

<!-- formula-not-decoded -->

and for optimal policy

<!-- formula-not-decoded -->

The solution is a bit different as r s ˆ Q π s is the regularized form of the L p optimization problem with simplex constraints which correspond to r s ˆ Q π s = R 0 -( π ∗ s ‖ π ∗ s ‖ q ) q -1 α s + γ inf P π ∈P s P π ˆ V π or for sa case : r ( s,a ) ˆ Q π sa = R 0 -α sa + γ inf P π ∈P s P π ˆ V π

<!-- formula-not-decoded -->

Indeed, even without close form, we can write the problem with an expectation over the nominal and the infimum problem.

Lemma C.2 (Upper bound on Q ∗ -ˆ Q π ∗ and on Q ˆ π -ˆ Q ˆ π , all Q values are now with robust under simplex constraints.) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in (a) we use previous Lemma C.1.

Hence, taking the supremum norm ‖ . ‖ ∞ , where (b) is the triangular inequality, (c) Eq. (4), (d) is the triangular inequality for seminorms, (d) is | inf A f -inf A g | ≤ sup A | f -g | . , (e) is a relaxation (f) is the relation between sup and inf, (g) is lemma 1 of Kumar et al. [2022]), (h) is inequality for seminorms and norms (5).

<!-- formula-not-decoded -->

For brevity in the remaining analysis, let us define the shorthand:

<!-- formula-not-decoded -->

Recall, slightly abusing the notation, for V ∈ R S , we define the vector Var P ( V ) ∈ R S× A as Var P ( V ) = P ( V ) 2 -( PV ) 2 .

Lemma C.3 (Agarwal et al. [2020], Lemma 9) . With probability greater than 1 -δ ,

<!-- formula-not-decoded -->

Proof. The proof of Agarwal et al. [2020] holds for classical MDP but can be adapted to the robust setting using all lemmas proved for the bound in H 4 previously. Lemma B.11,B.12 ,B.14,B.15,45 are needed but the main difference is that we are using Berstein's inequality and not Hoeffding's inequality. The idea is first, as in the previous proof, to apply Berstein's inequality to independent variables using s absorbing MDPs then using Lemma B.15.

Proof. Similar to Agarwal et al. [2020], we first show that

<!-- formula-not-decoded -->

First, with probability greater than 1 -δ , we have that for all u ∈ U s .

<!-- formula-not-decoded -->

using the triangle inequality in (a), (b) classical Berstein's inequality, (d) for variance and Lemmas B.11 and B.12 such as

∥ ∥ ∣ It is true for u ∈ U s , so we take the best possible choice, which completes the proof of the first claim. The proof of the second claim is similar. Then using Lemma B.15 gives the final concentration theorem.

<!-- formula-not-decoded -->

Lemma C.4 (Azar et al. [2013], Lemma 7) . This is an adaptation of Azar et al. [2013] to RMDPs. For any policy π , where P 0 is the nominal transition model of M .

<!-- formula-not-decoded -->

Proof. This proof is exactly the same for Robust and non robust MDPs, as it uses only standard computations such as the Jensen inequality and no robust form which are specific to this problem. The main difference is that we are doing the proof on the nominal of our robust set P 0 , considering the regularized robust Bellman operator and associated regularized reward functions.

Azar et al. [2013] introduce the variance of the sum of discounted rewards starting at state-action ( s, a ) ,

<!-- formula-not-decoded -->

and we defined the same variance for robust MDPs using robust rewards r ( s,a ) Q π sa and r s Q π s and using robust Q-function instead of classical Q-function in the definition of Σ . Then, in their Lemma 6 they show that, for any π :

<!-- formula-not-decoded -->

which is, in fact, a Bellman equation for the variance. The proof is exactly the same for RMDPs considering our robust reward r ( s,a ) Q π sa or r s Q π s and not classical R 0 . Note that this is thanks to the regularized form of robust RMDPs. Finally, Lemma C.4 is the same as their Lemma 7 considering robust rewards. This lemma is usually called the total variance lemma. This completes the proof.

Lemma C.5. The following upper bound holds with probability 1 -δ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

We have that (a) is true by Lemma C.2, (b) is by the triangular inequality using ̂ V ̂ π = ̂ V ̂ π + ̂ V /star -̂ V /star , (c) is from the definition of /epsilon1 opt and Eq. (4), (d) is by positivity of the classic horizon inverse matrix, that is ( I -γP ) -1 = ∑ t&gt; 0 γ t P t &gt; 0 , (e) is by Lemma C.3, (f) is by the triangular inequality for the variance (which is, in fact, a seminorm) and decomposing ̂ V /star = ̂ V /star + ̂ V ̂ π -̂ V ̂ π + V ̂ π -V ̂ π , (g) is by Lemma C.4, uses the definition of /epsilon1 opt and takes the sup over ( s, a ) of the variance in the second term, and eventually (h) is because we have that ‖ V π -̂ V π ‖ ∞ ≤ ‖ Q π -̂ Q π ‖ ∞ for any π .

<!-- formula-not-decoded -->

Lemma C.6. The following upper bound holds with probability 1 -δ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

We have that (a) is true by Lemma C.2, (b) is by the positivity of the classic horizon inverse matrix, (c) is by Lemma (C.3), (d) is by the triangular inequality for the variance (which is a seminorm), (e) is by Lemma C.4 and taking the sup over ( s, a ) of the variance in the second term, and eventually (h) is because ‖ V π -̂ V π ‖ ∞ ≤ ‖ Q π -̂ Q π ‖ ∞ for any π .

with C N = γ 1 -γ √ 8 L N and C β = 2 γβ | S | 1 /q 1 -γ .

As the event on which ∆ ′ δ,N is the same in the two previous Lemma C.5 and Lemma C.6, we can obtain the following.

Theorem C.7. For 0 &lt; C β ≤ 1 / 2 and 0 &lt; C N + C β &lt; 1 , with probability 1 -δ , we get:

Proof. This result is obtained by combining the two previous Lemmas C.5 and C.6 and passing the term in ( C N + C β ) to the left-hand side.

<!-- formula-not-decoded -->

Note that C β + C N &lt; 1 implies C β = 2 γβ | S | 1 /q 1 -γ &lt; 1 and hence β &lt; 1 -γ 2 γ | S | 1 /q . Now we need to pick C N &lt; 1 -C β . Let C N ≤ 1 -C β -η , for any 0 &lt; η &lt; 1 -C β the previous inequality becomes

As ∆ ′ δ,N = √ cL N + cL (1 -γ ) N , the term in 1 / √ N is given by 8 γ √ LH 3 / 2 η √ N ( 1 + 1 / 4 √ c/H ) and is smaller than /epsilon1 whenever

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will use c &lt; 16 and H ≥ 1 and use the stronger constraint

<!-- formula-not-decoded -->

Along the same line, the term in 1 /N is 2 γcLH 2 ηN which is smaller than /epsilon1 whenever

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, C N &lt; 1 -η -C β means hence

We deduce that whenever

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the error is smaller than 2 /epsilon1 up to the /epsilon1 opt terms.

This bounds reduces to with C = 256 /η 2 if

Note that /epsilon1 ∈ [0 , H ) and η &lt; 1 so that the previous condition simplifies to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we want to obtain an arbitrary /epsilon1 0 , it suffices thus to take η arbitrarily small leading to the constant C = 256 /η 2 to be arbitrarily large.

Note that if /epsilon1 0 ≥ O ( H 1 / 2+ δ ) then 1 /η &gt; O ( H δ ) which adds a H 2 δ factor to the bound on N .

However, for any κ √ H and for any C β , it exists an η independent of H so that /epsilon1 0 = 8 √ H 1 -η -C β η = κ √ H , hence the result stated in Theorem 5.1.

Now, as L = log(8 |S||A| / ((1 -γ ) δ )) , the previous condition can be summarized by provided /epsilon1 &lt; /epsilon1 0 .

<!-- formula-not-decoded -->

Finally, taking β 0 = 1 -γ 8 γ which gives C β = 1 / 4 and η = 1 / 2 so that C N ≤ 1 / 4 , we obtain C = 1024 and /epsilon1 0 = √ 16 H .