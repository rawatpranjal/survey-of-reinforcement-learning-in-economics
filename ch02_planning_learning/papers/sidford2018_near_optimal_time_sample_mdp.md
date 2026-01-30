## Near-Optimal Time and Sample Complexities for Solving Discounted Markov Decision Process with a Generative Model

Aaron Sidford Stanford University sidford@stanford.edu

Mengdi Wang Princeton University mengdiw@princeton.edu

Lin F. Yang Princeton University lin.yang@princeton.edu

June 7, 2019

## Abstract

In this paper we consider the problem of computing an /epsilon1 -optimal policy of a discounted Markov Decision Process (DMDP) provided we can only access its transition function through a generative sampling model that given any state-action pair samples from the transition function in O (1) time. Given such a DMDP with states S , actions A , discount factor γ ∈ (0 , 1), and rewards in range [0 , 1] we provide an algorithm which computes an /epsilon1 -optimal policy with probability 1 -δ where both the time spent and number of sample taken are upper bounded by

<!-- formula-not-decoded -->

For fixed values of /epsilon1 ∈ (0 , 1), this improves upon the previous best known bounds by a factor of (1 -γ ) -1 and matches the sample complexity lower bounds proved in [AMK13] up to logarithmic factors. We also extend our method to computing /epsilon1 -optimal policies for finite-horizon MDP with a generative model and provide a nearly matching sample complexity lower bound.

Xian Wu Stanford University xwu20@stanford.edu

Yinyu Ye Stanford University yyye@stanford.edu

## 1 Introduction

Markov decision processes (MDPs) are a fundamental mathematical abstraction used to model sequential decision making under uncertainty and are a basic model of discrete-time stochastic control and reinforcement learning (RL). Particularly central to RL is the case of computing or learning an approximately optimal policy when the MDP itself is not fully known beforehand. One of the simplest such settings is when the states, rewards, and actions are all known but the transition between states when an action is taken is probabilistic, unknown, and can only be sampled from.

Computing an approximately optimal policy with high probability in this case is known as PAC RL with a generative model. It is a well studied problem with multiple existing results providing algorithms with improved the sample complexity (number of sample transitions taken) and running time (the total time of the algorithm) under various MDP reward structures, e.g. discounted infinite-horizon, finite-horizon, etc. (See Section 2 for a detailed review of the literature.)

In this work, we consider this well studied problem of computing approximately optimal policies of discounted infinite-horizon Markov Decision Processes (DMDP) under the assumption we can only access the DMDP by sampling state transitions. Formally, we suppose that we have a DMDP with a known set of states, S , a known set of actions that can be taken at each states, A , a known reward r s,a ∈ [0 , 1] for taking action a ∈ A at state s ∈ S , and a discount factor γ ∈ (0 , 1). We assume that taking action a at state s probabilistically transitions an agent to a new state based on a fixed, but unknown probability vector P s,a . The objective is to maximize the cumulative sum of discounted rewards in expectation. Throughout this paper, we assume that we have a generative model , a notion introduced by [Kak03], which allows us to draw random state transitions of the DMDP. In particular, we assume that we can sample from the distribution defined by P s,a for all ( s, a ) ∈ S × A in O (1) time. This is a natural assumption and can be achieved in expectation in certain computational models with linear time preprocessing of the DMDP. 1

The main result of this paper is that we provide the first algorithm that is sample-optimal and runtime-optimal (up to polylogarithmic factors) for computing an /epsilon1 -optimal policy of a DMDP with a generative model (in the regime of 1 / √ (1 -γ ) |S| ≤ /epsilon1 ≤ 1). In particular, we develop a randomized Variance-Reduced Q-Value Iteration (vQVI) based algorithm that computes an /epsilon1 -optimal policy with probability 1 -δ with a number of samples, i.e. queries to the generative model, bound by

This result matches (up to polylogarithmic factors) the following sample complexity lower bound established in [AMK13] for finding /epsilon1 -optimal policies with probability 1 -δ (see Appendix D):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, we show that the algorithm can be implemented using sparse updates such that the overall run-time complexity is equal to its sample complexity up to constant factors, as long as each sample transition can be generated in O (1) time. Consequently, up to logarithmic factors our run time complexity is optimal as well. In addition, the algorithm's space complexity is Θ( |S||A| ).

Our method and analysis builds upon a number of prior works. (See Section 2 for an indepth comparison.) The paper [AMK13] provided the first algorithm that achieves the optimal

1 If instead the oracle needed time τ , every running time result in this paper should be multiplied by τ .

sample complexity for finding /epsilon1 -optimal value functions (rather than /epsilon1 -optimal policy), as well as the matching lower bound. Unfortunately an /epsilon1 -optimal value function does not imply an /epsilon1 -optimal policy and if we directly use the method of [AMK13] to get an /epsilon1 -optimal policy for constant /epsilon1 , the best known sample complexity is ˜ O ( |S||A| (1 -γ ) -5 /epsilon1 -2 ). 2 This bound is known to be improvable through related work of [SWWY18] which provides a method for computing an /epsilon1 -optimal policy using ˜ O ( |S||A| (1 -γ ) -4 /epsilon1 -2 ) samples and total runtime and the work of [AMK13] which in the regime of small approximation error, i.e. where /epsilon1 = O ((1 -γ ) -1 / 2 |S| -1 / 2 ), already provides a method that achieves the optimal sample complexity. However, when the approximation error takes fixed values, e.g. /epsilon1 ≥ Ω((1 -γ ) -1 / 2 |S| -1 / 2 ), there remains a gap between the best known runtime and sample complexity for computing an /epsilon1 -optimal policy and the theoretical lower bounds. For fixed values of /epsilon1 , which mostly occur in real applications, our algorithm improves upon the previous best sample and time complexity bounds by a factor of (1 -γ ) -1 where γ ∈ (0 , 1), the discount factor, is typically close to 1.

We achieve our results by combining and strengthening techniques from both [AMK13] and [SWWY18]. On the one hand, in [AMK13] the authors showed that simply constructing a 'sparsified' MDP model by taking samples and then solving this model to high precision yields a sample optimal algorithm in our setting for computing the approximate value of every state. On the other hand, [SWWY18] provided faster algorithms for solving explicit DMDPs and improved sample and time complexities given a sampling oracle. In fact, as we show in Appendix B.1, simply combining these two results yields the first nearly optimal runtime for approximately learning the value function with a generative model. Unfortunately, it is known that an approximate-optimal value function does not immediately yield an approximate-optimal policy of comparable quality (see e.g. [Ber13]) and it is was previously unclear how to combine these methods to improve upon previous known bounds for computing an approximate policy. To achieve our policy computation algorithm we therefore open up both the algorithms and the analysis in [AMK13] and [SWWY18], combining them in nontrivial ways. Our proofs leverage techniques ranging from standard probabilistic analysis tools such as Hoeffding and Bernstein inequalities, to optimization techniques such as variance reduction, to properties specific to MDPs such as the Bellman fixed-point recursion for expectation and variance of the optimal value vector, and monotonicity of value iteration.

Finally, we extend our method to finite-horizon MDPs, which are also occurred frequently in real applications. We show that the number of samples needed by this algorithm is ˜ O ( H 3 |S||A| /epsilon1 -2 ) , in order to obtain an /epsilon1 -optimal policy for H -horizon MDP (see Appendix F). We also show that the preceding sample complexity is optimal up to logarithmic factors by providing a matching lower bound. We hope this work ultimately opens the door for future practical and theoretical work on solving MDPs and efficient RL more broadly.

## 2 Comparison to Previous Work

There exists a large body of literature on MDPs and RL (see e.g. [Kak03, SLL09, KBJ14, DB15] and reference therein). The classical MDP problem is to compute an optimal policy exactly or

2 [AMK13] showed that one can obtain /epsilon1 -optimal value v (instead of /epsilon1 -optimal policy) using sample size ∝ (1 -γ ) -3 /epsilon1 -2 . By using this /epsilon1 -optimal value v , one can get a greedy policy that is [(1 -γ ) -1 /epsilon1 ]-optimal. By setting /epsilon1 → (1 -γ ) /epsilon1 , one can obtain an /epsilon1 -optimal policy, using the number of samples ∝ (1 -γ ) -5 /epsilon1 -2 .

3 Although not explicitly stated, an immediate derivation shows that obtaining an /epsilon1 -optimal policy in [AMK13] requires O ( | S || A | (1 -γ ) -5 /epsilon1 -2 ) samples.

Table 1: Sample Complexity to Compute /epsilon1 -Approximate Policies Using the Generative Sampling Model : Here |S| is the number of states, |A| is the number of actions per state, γ ∈ (0 , 1) is the discount factor, and C is an upper bound on the ergodicity. Rewards are bounded between 0 and 1.

| Algorithm                            | Sample Complexity                                                             | References   |
|--------------------------------------|-------------------------------------------------------------------------------|--------------|
| Phased Q-Learning                    | ˜ O ( C |S||A| (1 - γ ) 7 /epsilon1 2 )                                       | [KS99]       |
| Empirical QVI                        | ˜ O ( |S||A| (1 - γ ) 5 /epsilon1 2 ) 3                                       | [AMK13]      |
| Empirical QVI                        | ˜ O ( |S||A| (1 - γ ) 3 /epsilon1 2 ) if /epsilon1 = ˜ O ( 1 √ (1 - γ ) |S| ) | [AMK13]      |
| Randomized Primal-Dual Method        | ˜ O ( C |S||A| (1 - γ ) 4 /epsilon1 2 )                                       | [Wan17]      |
| Sublinear Randomized Value Iteration | ˜ O ( |S||A| (1 - γ ) 4 /epsilon1 2 )                                         | [SWWY18]     |
| Sublinear Randomized QVI             | ˜ O ( |S||A| (1 - γ ) 3 /epsilon1 2 )                                         | This Paper   |

approximately, when the full MDP model is given as input. For a survey on existing complexity results when the full MDP model is given, see Appendix A.

Despite the aforementioned results of [Kak03, AMK13, SWWY18], there exists only a handful of additional RL methods that achieve a small sample complexity and a small run-time complexity at the same time for computing an /epsilon1 -optimal policy. A classical result is the phased Q-learning method by [KS99], which takes samples from the generative model and runs a randomized value iteration. The phased Q-learning method finds an /epsilon1 -optimal policy using O ( |S||A| /epsilon1 -2 / poly(1 -γ )) samples/updates, where each update uses ˜ O (1) run time. 4 Another work [Wan17] gave a randomized mirror-prox method that applies to a special Bellman saddle point formulation of the DMDP. They achieve a total runtime of ˜ O ( |S| 3 |A| /epsilon1 -2 (1 -γ ) -6 ) for the general DMDP and ˜ O ( C |S||A| /epsilon1 -2 (1 -γ ) -4 ) for DMDPs that are ergodic under all possible policies, where C is a problem-specific ergodicity measure. A recent closely related work is [SWWY18] which gave a variance-reduced randomized value iteration that works with the generative model and finds an /epsilon1 -approximate policy in sample size/run time ˜ O ( |S||A| /epsilon1 -2 (1 -γ ) -4 ), without requiring any ergodicity assumption.

Finally, in the case where /epsilon1 = O ( 1 / √ (1 -γ ) -1 |S| ) , [AMK13] showed that the solution obtained by performing exact PI on the empirical MDP model provides not only an /epsilon1 -optimal value but also an /epsilon1 -optimal policy. In this case, the number of samples is ˜ O ( |S||A| (1 -γ ) -3 /epsilon1 -2 ) and matches the sample complexity lower bound. Although this sample complexity is optimal, it requires solving the empirical MDP exactly (see Appendix B), and is no longer sublinear in the size of the MDP model because of the very small approximation error /epsilon1 = O (1 / √ (1 -γ ) |S| ). See Table 1 for a list of comparable sample complexity results for solving MDP based on the generative model.

## 3 Preliminaries

We use calligraphy upper case letters for sets or operators, e.g., S , A and T . We use bold small case letters for vectors, e.g., v , r . We denote v s or v ( s ) as the s -th entry of vector v . We denote matrix as bold upper case letters, e.g., P . We denote constants as normal upper case letters, e.g.,

4 The dependence on (1 -γ ) in [KS99] is not stated explicitly but we believe basic calculations yield O (1 / (1 -γ ) 7 ).

M . For a vector v ∈ R N for index set N , we denote √ v , | v | , and v 2 vectors in R N with √ · , | · | , and ( · ) 2 acting coordinate-wise. For two vectors v , u ∈ R N , we denote by v ≤ u as coordinate-wise comparison, i.e., ∀ i ∈ N : v ( i ) ≤ u ( i ). The same definition are defined to relations ≤ , &lt; and &gt; .

We describe a DMDP by the tuple ( S , A , P , r , γ ), where S is a finite state space, A is a finite action space, P ∈ R S×A×S is the state-action-state transition matrix, r ∈ R S×A is the state-action reward vector, and γ ∈ (0 , 1) is a discount factor. We use P s,a ( s ′ ) to denote the probability of going to state s ′ from state s when taking action a . We also identify each P s,a as a vector in R S . We use r s,a to denote the reward obtained from taking action a ∈ A at state s ∈ S and assume r ∈ [0 , 1] S×A . 5 For a vector v ∈ R S , we denote Pv ∈ R S×A as ( Pv ) s,a = P /latticetop s,a v . A policy π : S → A maps each state to an action. The objective of MDP is to find the optimal policy π ∗ that maximizes the expectation of the cumulative sum of discounted rewards.

In the remainder of this section we give definitions for several prominent concepts in MDP analysis that we use throughout the paper.

Definition 3.1 (Bellman Value Operator) . For a given DMDP the value operator T : R S ↦→ R S is defined for all u ∈ R S and s ∈ S by T ( u ) s = max a ∈A [ r a ( s ) + γ · P /latticetop s,a v ] , and we let v ∗ denote the value of the optimal policy π ∗ , which is the unique vector such that T ( v ∗ ) = v ∗ .

Note that T π can be viewed as the value operator for the modified MDP where the only available action from each state is given by the policy π . Note that this modified MDP is essentially just an uncontrolled Markov Chain, i.e. there are no action choices that can be made.

Definition 3.2 (Policy) . We call any vector π ∈ A S a policy and say that the action prescribed by policy π to be taken at state s ∈ S is π s . We let T π : R S ↦→ R S denote the value operator associated with π defined for all u ∈ R S and s ∈ S by T π ( u ) s = r s,π ( s ) + γ · P /latticetop s,π ( s ) u , and we let v π denote the values of policy π , which is the unique vector such that T π ( v π ) = v π .

Definition 3.3 ( /epsilon1 -optimal value and policy) . We say values u ∈ R S are /epsilon1 -optimal if ‖ v ∗ -u ‖ ∞ ≤ /epsilon1 and policy π ∈ A S is /epsilon1 -optimal if ‖ v ∗ -v π ‖ ∞ ≤ /epsilon1 , i.e. the values of π are /epsilon1 -optimal.

Definition 3.4 (Q-function) . For any policy π , we define the Q-function of a MDP with respect to π as a vector Q ∈ R S×A such that Q π ( s, a ) = r ( s, a ) + γ P /latticetop s,a v π . The optimal Q -function is defined as Q ∗ = Q π ∗ . We call any vector Q ∈ R S×A a Q-function even though it may not relate to a policy or a value vector and define v ( Q ) ∈ R S and π ( Q ) ∈ A S as the value and policy implied by Q , by

<!-- formula-not-decoded -->

For a policy π , let P π Q ∈ R S×A be defined as ( P π Q )( s, a ) = ∑ s ′ ∈S P s,a ( s ′ ) Q ( s ′ , π ( s ′ )).

## 4 Technique Overview

In this section we provide a more detailed and technical overview of our approach. At a high level, our algorithm shares a similar framework as the variance reduction algorithm presented in [SWWY18]. This algorithm used two crucial algorithmic techniques, which are also critical in this paper. We call these techniques as the monotonicity technique and the variance reduction technique. Our algorithm and the results of this paper can be viewed as an advanced, non-trivial integration of these two methods, augmented with a third technique which we refer to as a total-variation

5 A general r ∈ R S×A can always be reduced to this case by shifting and scaling.

technique which was discovered in several papers [MM99, LH12, AMK13]. In the remainder of this section we give an overview of these techniques and through this, explain our algorithm.

The Monotonicity Technique Recall that the classic value iteration algorithm for solving a MDP repeatedly applies the following rule

<!-- formula-not-decoded -->

A greedy policy π ( i ) can be obtained at each iteration i by

<!-- formula-not-decoded -->

For any u &gt; 0, it can be shown that if one can approximate v ( i ) ( s ) with ̂ v ( i ) ( s ) such that ‖ ̂ v ( i ) -v ( i ) ‖ ∞ ≤ (1 -γ ) u and run the above value iteration algorithm using these approximated values, then after Θ((1 -γ ) -1 log[ u -1 (1 -γ ) -1 ]) iterations, the final iteration gives an value function that is u -optimal ([Ber13]). However, a u -optimal value function only yields a u/ (1 -γ )-optimal greedy policy (in the worst case), even if (4.2) is precisely computed. To get around this additional loss, a monotone-VI algorithm was proposed in [SWWY18] as follows. At each iteration, this algorithm maintains not only an approximated value v ( i ) but also a policy π ( i ) . The key for improvement is to keep values as a lower bound of the value of the policy on a set of sample paths with high probability. In particular, the following monotonicity condition was maintained with high probability

<!-- formula-not-decoded -->

By the monotonicity of the Bellman's operator, the above equation guarantees that v ( i ) ≤ v π ( i ) . If this condition is satisfied, then, if after R iterations of approximate value iteration we obtain an value ̂ v ( R ) that is u -optimal then we also obtain a policy π ( R ) which by the monotonicity condition and the monotonicity of the Bellman operator T π ( R ) yields

<!-- formula-not-decoded -->

and therefore this π ( R ) is an u -optimal policy. Ultimately, this technique avoids the standard loss of a (1 -γ ) -1 factor when converting values to policies.

The Variance Reduction Technique Suppose now that we provide an algorithm that maintains the monotonicity condition using random samples from P s,a to approximately compute (4.1). Further, suppose we want to obtain a new value function and policy that is at least ( u/ 2)-optimal. In order to obtain the desired accuracy, we need to approximate P /latticetop s,a v ( i ) up to error at most (1 -γ ) u/ 2. Since ‖ v ( i ) ‖ ∞ ≤ (1 -γ ) -1 , by Hoeffding bound, ˜ O ((1 -γ ) -4 u -2 ) samples suffices. Note that the number of samples also determines the computation time and therefore each iteration takes ˜ O ((1 -γ ) -4 u -2 |S||A| ) samples/computation time and ˜ O ((1 -γ ) -1 ) iterations for the value iteration to converge. Overall, this yields a sample/computation complexity of ˜ O ((1 -γ ) -5 u -2 |S||A| ). To reduce the (1 -γ ) -5 dependence, [SWWY18] uses properties of the input (and the initialization) vectors: ‖ v (0) -v ∗ ‖ ∞ ≤ u and rewrites value iteration (4.1) as follows

<!-- formula-not-decoded -->

Notice that P /latticetop s,a v (0) is shared over all iterations and we can approximate it up to error (1 -γ ) u/ 4 using only ˜ O ((1 -γ ) -4 u -2 ) samples. For every iteration, we have ‖ v ( i -1) -v (0) ‖ ∞ ≤ u (recall that we demand the monotonicity is satisfied at each iteration). Hence P /latticetop s,a ( v ( i -1) -v (0) ) can be approximated up to error (1 -γ ) u/ 4 using only ˜ O ((1 -γ ) -2 ) samples (note that there is no u -dependence here). By this technique, over ˜ O ((1 -γ ) -1 ) iterations only ˜ O ((1 -γ ) -4 u -2 +(1 -γ ) -3 ) samples/computation per state action pair are needed, i.e. there is a (1 -γ ) improvement.

The Total-Variance Technique By combining the monotonicity technique and variance reduction technique, one can obtain a ˜ O ((1 -γ ) -4 ) sample/running time complexity (per state-action pair) on computing a policy; this was one of the results [SWWY18]. However, there is a gap between this bound and the best known lower bound of ˜ Ω[ | S || A | /epsilon1 -2 (1 -γ ) -3 ] [AMK13]. Here we show how to remove the last (1 -γ ) factor by better exploiting the structure of the MDP. In [SWWY18] the update error in each iteration was set to be at most (1 -γ ) u/ 2 to compensate for error accumulation through a horizon of length (1 -γ ) -1 (i.e., the accumulated error is sum of the estimation error at each iteration). To improve we show how to leverage previous work to show that the true error accumulation is much less. To see this, let us now switch to Bernstein inequality. Suppose we would like to estimate the value function of some policy π . The estimation error vector of the value function is upper bounded by ˜ O ( √ σ π /m ), where σ π ( s ) = Var s ′ ∼ P s,π ( s ) ( v π ( s ′ )) denotes the variance of the value of the next state if starting from state s by playing policy π , and m is the number of samples collected per state-action pair. The accumulated error due to estimating value functions can be shown to obey the following inequality (upper to logarithmic factors)

<!-- formula-not-decoded -->

where c 1 is a constant and the inequality follows from a Cauchy-Swartz-like inequality. According to the law of total variance , for any given policy π (in particular, the optimal policy π ∗ ) and initial state s , the expected sum of variance of the tail sums of rewards, ∑ γ 2 i P i π σ π , is exactly the variance of the total return by playing the policy π . This observation was previously used in the analysis of [MM99, LH12, AMK13]. Since the upper bound on the total return is (1 -γ ) -1 , it can be shown that ∑ i γ 2 i P i π σ π ≤ (1 -γ ) -2 · 1 and therefore the total error accumulation is √ (1 -γ ) -3 /m . Thus picking m ≈ (1 -γ ) -3 /epsilon1 -2 is sufficient to control the accumulated error (instead of (1 -γ ) -4 ). To analyze our algorithm, we will apply the above inequality to the optimal policy π ∗ to obtain our final error bound.

Putting it All Together In the next section we show how to combine these three techniques into one algorithm and make them work seamlessly. In particular, we provide and analyze how to combine these techniques into an Algorithm 1 which can be used to at least halve the error of a current policy. Applying this routine a logarithmic number of time then yields our desired bounds. In the input of the algorithm, we demand the input value v (0) and π (0) satisfies the required monotonicity requirement, i.e., v (0) ≤ T π (0) ( v (0) ) (in the first iteration, the zero vector 0 and an arbitrary policy π satisfies the requirement). We then pick a set of samples to estimate Pv (0) accurately with ˜ O ((1 -γ ) -3 /epsilon1 -2 ) samples per state-action pair. The same set of samples is used to estimate the variance vector σ v ∗ . These estimates serve as the initialization of the algorithm. In each iteration i , we draw fresh new samples to compute estimate of P ( v ( i ) -v (0) ). The sum of the

estimate of Pv (0) and P ( v ( i ) -v (0) ) gives an estimate of Pv ( i ) . We then make the above estimates have one-sided error by shifting them according to their estimation errors (which is estimated from the Bernstein inequality). These one-side error estimates allow us to preserve monotonicity, i.e., guarantees the new value is always improving on the entire sample path with high probability. The estimate of Pv ( i ) is plugged in to the Bellman's operator and gives us new value function, v ( i +1) and policy π ( i +1) , satisfying the monotonicity and advancing accuracy. Repeating the above procedure for the desired number of iterations completes the algorithm.

```
Algorithm 1 Variance-Reduced QVI 1: Input: A sampling oracle for DMDP M = ( S , A , r , P , γ ) 2: Input: Upper bound on error u ∈ [0 , (1 -γ ) -1 ] and error probability δ ∈ (0 , 1) 3: Input: Initial values v (0) and policy π (0) such that v (0) ≤ T π (0) v (0) , and v ∗ -v (0) ≤ u 1 ; 4: Output: v , π such that v ≤ T π ( v ) and v ∗ -v ≤ ( u/ 2) · 1 . 5: 6: INITIALIZATION: 7: Let β ← (1 -γ ) -1 , and R ←/ceilingleft c 1 β ln[ βu -1 ] /ceilingright for constant c 1 ; 8: Let m 1 ← c 2 β 3 u -2 log(8 |S||A| δ -1 ) for constant c 2 ; 9: Let m 2 ← c 3 β 2 log[2 R |S||A| δ -1 ] for constant c 3 ; 10: Let α 1 ← m 1 -1 log(8 |S||A| δ -1 ); 11: For each ( s, a ) ∈ S × A , sample independent samples s (1) s,a , s (2) s,a , . . . , s ( m 1 ) s,a from P s,a ; 12: Initialize w = ˜ w = ̂ σ = Q (0) ← 0 S×A , and i ← 0; 13: for each ( s, a ) ∈ S × A do 14: \\ Compute empirical estimates of P /latticetop s,a v (0) and σ v (0) ( s, a ) 15: Let ˜ w ( s, a ) ← 1 m 1 ∑ m 1 j =1 v (0) ( s ( j ) s,a ) 16: Let ̂ σ ( s, a ) ← 1 m 1 ∑ m 1 j =1 ( v (0) ) 2 ( s ( j ) s,a ) -˜ w 2 ( s, a ) 17: 18: \\ Shift the empirical estimate to have one-sided error and guarantee monotonicity 19: w ( s, a ) ← ˜ w ( s, a ) -√ 2 α 1 ̂ σ ( s, a ) -4 α 3 / 4 1 ‖ v (0) ‖ ∞ -(2 / 3) α 1 ‖ v (0) ‖ ∞ 20: 21: \\ Compute coarse estimate of the Q -function 22: Q (0) ( s, a ) ← r ( s, a ) + γ w ( s, a ) 23: 24: REPEAT: \\ successively improve 25: for i = 1 to R do 26: \\ Compute g ( i ) the estimate of P [ v ( i ) -v (0) ] with one-sided error 27: Let v ( i ) ← v ( Q ( i -1) ), π ( i ) ← π ( Q ( i -1) ); \\ let ˜ v ( i ) ← v ( i ) , ˜ π ( i ) ← π ( i ) (for analysis) ; 28: For each s ∈ S , if v ( i ) ( s ) ≤ v ( i -1) ( s ), then v ( i ) ( s ) ← v ( i -1) ( s ) and π ( i ) ( s ) ← π ( i -1) ( s ); 29: For each ( s, a ) ∈ S × A , draw independent samples ˜ s (1) s,a , ˜ s (2) s,a , . . . , ˜ s ( m 2 ) s,a from P s,a ; 30: Let g ( i ) ( s, a ) ← 1 m 2 ∑ m 2 j =1 [ v ( i ) (˜ s ( j ) s,a ) -v (0) (˜ s ( j ) s,a ) ] -(1 -γ ) u/ 8; 31: 32: \\ Improve Q ( i ) 33: Q ( i ) ← r + γ · [ w + g ( i ) ]; 34: return v ( R ) , π ( R ) .
```

## 5 Algorithm and Analysis

In this section we provide and analyze our near sample/time optimal /epsilon1 -policy computation algorithm. As discussed in Section 4 our algorithm combines three main ideas: variance reduction, the monotone value/policy iteration, and the reduction of accumulated error via Bernstein inequality. These ingredients are used in the Algorithm 1 to provide a routine which halves the error of a given policy. We analyze this procedure in Section 5.1 and use it to obtain our main result in Section 5.2.

## 5.1 The Analysis of the Variance Reduced Algorithm

In this section we analyze Algorithm 1, showing that each iteration of the algorithm approximately contracts towards the optimal value and policy and that ultimately the algorithm halves the error of the input value and policy with high probability. All proofs in this section are deferred to Appendix E.1.

We start with bounding the error of ˜ w and ̂ σ defined in Line 15 and 16 of Algorithm 1. Notice that these are the empirical estimations of P /latticetop s,a v (0) and σ v (0) ( s, a ).

Lemma 5.1 (Empirical Estimation Error) . Let ˜ w and ̂ σ be computed in Line 15 and 16 of Algorithm 1. Recall that ˜ w and ̂ σ are empirical estimates of Pv and σ v = Pv 2 -( Pv ) 2 using m 1 samples per ( s, a ) pair. With probability at least 1 -δ , for L def = log(8 |S||A| δ -1 ) , we have and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof is a straightforward application of Bernstein's inequality and Hoeffding's inequality. Next we show that the difference between σ v (0) and σ v ∗ is also bounded.

<!-- formula-not-decoded -->

Next we show that in Line 30, the computed g ( i ) concentrates to and is an overestimate of P [ v ( i ) -v (0) ] with high probability.

<!-- formula-not-decoded -->

Lemma 5.3. Let g ( i ) be the estimate of P [ v ( i ) -v (0) ] defined in Line 30 of Algorithm 1. Then conditioning on the event that ‖ v ( i ) -v (0) ‖ ∞ ≤ 2 u , with probability at least 1 -δ/R , provided appropriately chosen constants c 1 , c 2 , and c 3 in Algorithm 1.

Now we present the key contraction lemma, in which we set the constants, c 1 , c 2 , c 3 , in Algorithm 1 to be sufficiently large (e.g., c 1 ≥ 4 , c 2 ≥ 8192 , c 3 ≥ 128). Note that these constants only need to be sufficiently large so that the concentration inequalities hold.

Lemma 5.4. Let Q ( i ) be the estimated Q -function of v ( i ) in Line 33 of Algorithm 1. Let π ( i ) and v ( i ) be estimated in iteration i , as defined in Line 27 and 28. Then, with probability at least 1 -2 δ , for all 1 ≤ i ≤ R , where for α 1 = m -1 1 L &lt; 1 the error vector ξ satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some sufficiently large constant C ≥ 8 .

Using the previous lemmas we can prove the guarantees of Algorithm 1.

Proposition 5.4.1. Onan input value vector v (0) , policy π (0) , and parameters u ∈ (0 , (1 -γ ) -1 ] , δ ∈ (0 , 1) such that v (0) ≤ T π (0) [ v (0) ], and v ∗ -v (0) ≤ u 1 , Algorithm 1 halts in time

<!-- formula-not-decoded -->

and outputs values v and policy π such that v ≤ T π ( v ) and v ∗ -v ≤ ( u/ 2) 1 with probability at least 1 -δ , provided appropriately chosen constants, c 1 , c 2 , c 3 .

We prove this proposition by iteratively applying Lemma 5.4. Suppose v ( R ) is the output of the algorithm, after R iterations. We show v ∗ -v ( R ) ≤ γ R -1 P π ∗ [ Q ∗ -Q 0 ] + ( I -γ P π ∗ ) -1 ξ . Notice that ( I -γ P π ∗ ) -1 ξ is related to ( I -γ P π ∗ ) -1 √ σ v ∗ . We then apply the variance analytical tools presented in Section C to show that ( I -γ P π ∗ ) -1 ξ ≤ ( u/ 4) 1 when setting the constants properly in Algorithm 1. We refer this technique as the total-variance technique , since ‖ ( I -γ P π ∗ ) -1 √ σ v ∗ ‖ 2 ∞ ≤ O [(1 -γ ) -3 ] instead of a na¨ ıve bound of (1 -γ ) -4 . We complete the proof by choosing R = ˜ Θ((1 -γ ) -1 log( u -1 )) and showing that γ R -1 P π ∗ [ Q ∗ -Q 0 ] ≤ ( u/ 4) 1 .

## 5.2 From Halving the Error to Arbitrary Precision

In the previous section, we provided an algorithm that on an input policy, outputs a policy with value vector that has /lscript ∞ distance to the optimal value vector only half of that of the input one. In this section, we give a complete policy computation algorithm by by showing that it is possible to apply this error 'halving' procedure iteratively. We summarize our meta algorithm in Algorithm 2. Note that in the algorithm, each call of HalfErr draws new samples from the sampling oracle. We refer in this section to Algorithm 1 as a subroutine HalfErr , which given an input MDP M with a sampling oracle, an input value function v ( i ) , and an input policy π ( i ) , outputs an value function v ( i +1) and a policy π ( i +1) .

Combining Algorithm 2 and Algorithm 1, we are ready to present main result.

Theorem 5.5. Let M = ( S , A , P , r , γ ) be a DMDP with a generative model. Suppose we can sample a state from each probability vector P s,a within time O (1) . Then for any /epsilon1, δ ∈ (0 , 1) , there exists an algorithm that halts in time

<!-- formula-not-decoded -->

## Algorithm 2 Meta Algorithm

```
1: Input: A sampling oracle of some M = ( S , A , r , P , γ ), /epsilon1 > 0 , δ ∈ (0 , 1) 2: Initialize: v (0) ← 0 , π (0) ← arbitrary policy, R ← Θ[log( /epsilon1 -1 (1 -γ ) -1 )] 3: for i = { 1 , 2 , . . . , R } do 4: // HalfErr is initialized with QVI ( u = 2 -i +1 (1 -γ ) -1 , δ, v (0) = v ( i -1) , π (0) = π ( i -1) ) 5: v ( i ) , π ( i ) ← HalfErr ← v ( i -1) , π ( i -1) 6: Output: v ( R ) , π ( R ) .
```

and obtains a policy π such that v ∗ -/epsilon1 1 ≤ v π ≤ v ∗ , with probability at least 1 -δ where v ∗ is the optimal value of M . The algorithm uses space O ( |S||A| ) and queries the generative model for at most O ( T ) fresh samples.

Remark 5.6 . In the above theorem, we require /epsilon1 ∈ (0 , 1). For /epsilon1 ≥ 1, our sample complexity may fail to be optimal. We leave this for a future project.

Remark 5.7 . The full analysis of the halving algorithm is presented in Section E.2. Our algorithm can be implemented in space O ( |S||A| ) since in Algorithm 1, the initialization phase can be done for each ( s, a ) and compute w ( s, a ) , ˜ w ( s, a ) , ̂ σ ( s, a ) , Q (0) ( s, a ) without storing the samples. The updates can be computed in space O ( |S||A| ) as well.

## 6 Concluding Remark

In summary, for a discounted Markov Decision Process (DMDP) M = ( S , A , P , r , γ ) provided we can only access the transition function of the DMDP through a generative sampling model, we provide an algorithm which computes an /epsilon1 -approximate optimal (for /epsilon1 ∈ (0 , 1)) policy with probability 1 -δ where both the time spent and number of sample taken is upper bounded by ˜ O ((1 -γ ) -3 /epsilon1 -2 |S||A| ). This improves upon the previous best known bounds by a factor of 1 / (1 -γ ) and matches the the lower bounds proved in [AMK13] up to logarithmic factors.

The appendix is structured as follows. Section A surveys the existing runtime results for solving the DMDP when a full model is given. Section B provides an runtime optimal algorithm for computing approximate value functions (by directly combining [AMK13] and [SWWY18]). Section C gives technical analysis and variance upper bounds for the total-variance technique. Section D discusses sample complexity lower bounds for obtaining approximate policies with a generative sampling model. Section E provides proofs to lemmas, propositions and theorems in the main text of the paper. Section F extends our method and results to the finite-horizon MDP and provides a nearly matching sample complexity lower bound.

## References

- [AMK13] Mohammad Gheshlaghi Azar, R´ emi Munos, and Hilbert J Kappen. Minimax pac bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3):325-349, 2013.
- [Bel57] Richard Bellman. Dynamic Programming . Princeton University Press, Princeton, NJ, 1957.
- [Ber13] Dimitri P Bertsekas. Abstract dynamic programming . Athena Scientific, Belmont, MA, 2013.
- [Dan16] George Dantzig. Linear Programming and Extensions . Princeton University Press, Princeton, NJ, 2016.
- [DB15] Christoph Dann and Emma Brunskill. Sample complexity of episodic fixed-horizon reinforcement learning. In Advances in Neural Information Processing Systems , pages 2818-2826, 2015.
- [d'E63] F d'Epenoux. A probabilistic production and inventory problem. Management Science , 10(1):98-108, 1963.
- [DG60] Guy De Ghellinck. Les problemes de decisions sequentielles. Cahiers du Centre dEtudes de Recherche Op´ erationnelle , 2(2):161-179, 1960.
- [HMZ13] Thomas Dueholm Hansen, Peter Bro Miltersen, and Uri Zwick. Strategy iteration is strongly polynomial for 2-player turn-based stochastic games with a constant discount factor. J. ACM , 60(1):1:1-1:16, February 2013.
- [How60] Ronald A. Howard. Dynamic programming and Markov processes . The MIT press, Cambridge, MA, 1960.
- [Kak03] Sham M Kakade. On the sample complexity of reinforcement learning . PhD thesis, University of London London, England, 2003.
- [KBJ14] Dileep Kalathil, Vivek S Borkar, and Rahul Jain. Empirical q-value iteration. arXiv preprint arXiv:1412.0180 , 2014.
- [KS99] Michael J Kearns and Satinder P Singh. Finite-sample convergence rates for q-learning and indirect algorithms. In Advances in neural information processing systems , pages 996-1002, 1999.
- [LDK95] Michael L Littman, Thomas L Dean, and Leslie Pack Kaelbling. On the complexity of solving Markov decision problems. In Proceedings of the Eleventh conference on Uncertainty in artificial intelligence , pages 394-402. Morgan Kaufmann Publishers Inc., 1995.
- [LH12] Tor Lattimore and Marcus Hutter. Pac bounds for discounted mdps. In International Conference on Algorithmic Learning Theory , pages 320-334. Springer, 2012.

- [LS14] Yin Tat Lee and Aaron Sidford. Path finding methods for linear programming: Solving linear programs in o (vrank) iterations and faster algorithms for maximum flow. In Foundations of Computer Science (FOCS), 2014 IEEE 55th Annual Symposium on , pages 424-433. IEEE, 2014.
- [LS15] Yin Tat Lee and Aaron Sidford. Efficient inverse maintenance and faster algorithms for linear programming. In Foundations of Computer Science (FOCS), 2015 IEEE 56th Annual Symposium on , pages 230-249. IEEE, 2015.
- [MM99] Remi Munos and Andrew W Moore. Variable resolution discretization for highaccuracy solutions of optimal control problems. Robotics Institute , page 256, 1999.
- [MS99] Yishay Mansour and Satinder Singh. On the complexity of policy iteration. In Proceedings of the Fifteenth conference on Uncertainty in artificial intelligence , pages 401-408. Morgan Kaufmann Publishers Inc., 1999.
- [Sch13] Bruno Scherrer. Improved and generalized upper bounds on the complexity of policy iteration. In Advances in Neural Information Processing Systems , pages 386-394, 2013.
- [SLL09] Alexander L Strehl, Lihong Li, and Michael L Littman. Reinforcement learning in finite mdps: Pac analysis. Journal of Machine Learning Research , 10(Nov):2413-2444, 2009.
- [SWWY18] Aaron Sidford, Mengdi Wang, Xian Wu, and Yinyu Ye. Variance reduced value iteration and faster algorithms for solving markov decision processes. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 770-787. SIAM, 2018.
- [Tse90] Paul Tseng. Solving h-horizon, stationary markov decision problems in time proportional to log (h). Operations Research Letters , 9(5):287-297, 1990.
- [Wan17] Mengdi Wang. Randomized linear programming solves the discounted Markov decision problem in nearly-linear running time. arXiv preprint arXiv:1704.01869 , 2017.
- [Ye05] Yinyu Ye. A new complexity result on solving the Markov decision problem. Mathematics of Operations Research , 30(3):733-749, 2005.
- [Ye11] Yinyu Ye. The simplex and policy-iteration methods are strongly polynomial for the Markov decision problem with a fixed discount rate. Mathematics of Operations Research , 36(4):593-603, 2011.

## A Previous Work on Solving DMDP with a Full Model

Value iteration was proposed by [Bel57] to compute an exact optimal policy of a given DMDP in time O ((1 -γ ) -1 |S| 2 |A| L log((1 -γ ) -1 )), where L is the total number of bits needed to represent the input; and it can find an approximate /epsilon1 -approximate solution in time O ( |S| 2 |A| (1 -γ ) -1 log(1 //epsilon1 (1 -γ ))); see e.g. [Tse90, LDK95]. The policy iteration was introduced by [How60] shortly after, where the policy is monotonically improved according to its associated value function. Its complexity has also been analyzed extensively; see e.g. [MS99, Ye11, Sch13]. Ye [Ye11] showed that policy iteration and the simplex method are strongly polynomial for DMDP and terminates in O ( |S| 2 |A| (1 -γ ) -1 log( |S| (1 -γ ) -1 )) number of iterations. Later [HMZ13] and [Sch13] improved the iteration bound to O ( |S||A| (1 -γ ) -1 log((1 -γ ) 1 )) for Howard's policy iteration method. A third approach is to formulate the nonlinear Bellman equation into a linear program [d'E63, DG60], and solve it using standard linear program solvers, such as the simplex method by Dantzig [Dan16] and the combinatorial interior-point algorithm by [Ye05]. [LS14, LS15] showed that one can solve linear programs in ˜ O ( √ rank( A )) number of linear system solves, which, applied to DMDP, yields to a running time of ˜ O ( |S| 2 . 5 |A| L ) for computing the exact policy and ˜ O ( |S| 2 . 5 |A| log(1 //epsilon1 )) for computing an /epsilon1 -optimal policy. [SWWY18] further improved the complexity of value iteration by using randomization and variance reduction. See Table 2 for comparable run-time results or computing the optimal policy when the MDP model is fully given.

Table 2: Running Times to Solve DMDPs Given the Full MDP Model : In this table, |S| is the number of states, |A| is the number of actions per state, γ ∈ (0 , 1) is the discount factor, and L is a complexity measure of the linear program formulation that is at most the total bit size to present the DMDP input. Rewards are bounded between 0 and 1.

| Algorithm                                 | Complexity                                                      | References     |
|-------------------------------------------|-----------------------------------------------------------------|----------------|
| Value Iteration (exact)                   | |S| 2 |A| L log(1 / (1 - γ )) 1 - γ                             | [Tse90, LDK95] |
| Value Iteration                           | |S| 2 |A| log(1 / (1 - γ ) /epsilon1 ) 1 - γ                    | [Tse90, LDK95] |
| Policy Iteration (Block Simplex)          | |S| 4 |A| 2 1 - γ log( 1 1 - γ )                                | [Ye11],[Sch13] |
| Recent Interior Point Methods             | ˜ O ( |S| 2 . 5 |A| L ) ˜ O ( |S| 2 . 5 |A| log(1 //epsilon1 )) | [LS14]         |
| Combinatorial Interior Point Algorithm    | |S| 4 |A| 4 log |S| 1 - γ                                       | [Ye05]         |
| High Precision Randomized Value Iteration | ˜ O [ ( nnz( P )+ |S||A| (1 - γ ) 3 ) log ( 1 /epsilon1δ ) ]    | [SWWY18]       |

## B Sample and Time Efficient Value Computation

In this section, we describe an algorithm that obtains an /epsilon1 -optimal values in time ˜ O ( /epsilon1 -2 (1 -γ ) -3 |S||A| ). Note that the time and number of samples of this algorithm is optimal (up to logarithmic factors) due to the lower bound in [AMK13] which also established this upper bound on the sample complexity (but not time complexity) of the problem.

We achieve this by combining the algorithms in [AMK13] and [SWWY18]. First, we use the

ideas and analysis of [AMK13] to construct a sparse MDP where the optimal value function of this MDP approximates the optimal value function of the original MDP and then we run the high precision algorithm in [SWWY18] on this sparsified MDP. We show that [SWWY18] runs in nearly linear time on sparsified MDP. Since the number of samples taken to construct the sparsified MDP was the the optimal number of samples, to solve the problem, the ultimate running time we thereby achieve is nearly optimal as any algorithm needs spend time at least the number of samples to obtain these samples.

We include this for completeness but note that the approximate value function we show how to compute here does not suffice to compute policy of the MDP of comparable quality. The greedy policy of an /epsilon1 -optimal value function is an /epsilon1/ (1 -γ )-optimal policy in the worst case. It has been shown in [AMK13] that the greedy policy of their value function is /epsilon1 -optimal if /epsilon1 ≤ (1 -γ ) 1 / 2 |S| -1 / 2 . However, when /epsilon1 is so small, the seemingly sublinear runtime ˜ O ((1 -γ ) -3 S||A| //epsilon1 2 ) essentially means a linear running time and sample complexity as O ((1 -γ ) -3 |S| 2 |A| ). The running time can be obtained by merely applying the result in [SWWY18] (although with a slightly different computation model).

## B.1 The Sparsified DMDP

Suppose we are given a DMDP M = ( S , A , r , P , γ ) with a sampling oracle. To approximate the optimal value of this MDP, we perform a spasification procedure as in [AMK13]. Sparsification of DMDP is conducted as follows. Let δ &gt; 0 , /epsilon1 &gt; 0 be arbitrary. First we pick a number

For each s ∈ S and each a ∈ A , we generate a sequence of independent samples from S using the probability vector P s,a

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next we construct a new and sparse probability vector P s,a ∈ ∆ |S| as

Combining these |S||A| new probability vectors, we obtain a new probability transition matrix ̂ P ∈ R S×A×S with number of non-zeros

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem B.1 ([AMK13]) . Let M be the original DMDP and ̂ M be the corresponding sparsified version. Let Q ∗ be the optimal Q -function vector of the original DMDP and ̂ Q ∗ be the optimal Q -function of ̂ M . Then with probability at least 1 -δ (over the randomness of the samples),

Denote ̂ M = ( S , A , r , ̂ P , γ ) as the sparsified DMDP. In the rest of this section, we use ̂ · to represent the quantities corresponding to DMDP ̂ M , e.g., ̂ v ∗ for the optimal value function, ̂ π ∗ for a optimal policy, and ̂ Q ∗ for the optimal Q -function. There is a strong approximation guarantee of the optimal Q -function of the sparsified MDP, presented as follows.

<!-- formula-not-decoded -->

Recall that v ∗ and ̂ v ∗ are the optimal value functions of M and ̂ M . From Theorem B.1, we immediately have with probability at least 1 -δ .

<!-- formula-not-decoded -->

## B.2 High Precision Algorithm in the Sparsified MDP

Next we shall use the high precision algorithm of the [SWWY18] which has the following guarantee.

Theorem B.2 ([SWWY18]) . There is an algorithm which given an input DMDP M = ( S , A , r , P , γ ) in time 6

and outputs a vector ˜ v ∗ such that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where v ∗ is the optimal value of M .

Combining the above two theorems, we immediately obtain an algorithm for finding /epsilon1 -optimal value functions. It works by first generating enough samples for each state-action pair and then call the high-precision MDP solver by [SWWY18]. It does not sample transitions adaptively. We show that it achieves an optimal running time guarantee (up to poly log factors) of obtaining the value function under the sampling oracle model.

Theorem B.3. Given an input DMDP M = ( S , A , r , P , γ ) with a sampling oracle and optimal value function v ∗ , there exists an algorithm, that runs in time

<!-- formula-not-decoded -->

and outputs a vector ̂ v ∗ such that ‖ ̂ v ∗ -v ∗ ‖ ∞ ≤ O ( /epsilon1 ) with probability at least 1 -O ( δ ) . Proof. We first obtain a sparsified MDP ̂ M = ( S , A , r , ̂ P , γ ) using the procedure described in Section B.1. This procedure runs in time O ( |S||A| m ), recalling that m is the number of samples per ( s, a ), defined in (B.1). Let ̂ u ∗ be the optimal value function of ̂ M . By Theorem B.1, with probability at least 1 -δ , ‖ ̂ u ∗ -v ∗ ‖ ≤ /epsilon1 , which we condition on for the rest of the proof. Calling the algorithm in Theorem B.2, we obtain a vector ˜ u ∗ in time and that with probability at least 1 -δ , ‖ ˜ u ∗ -̂ u ∗ ‖ ≤ /epsilon1 , which we condition on. By triangle inequality, we have

<!-- formula-not-decoded -->

This concludes the proof.

6 ˜ O ( f ) denotes O ( f · log O (1) f ).

<!-- formula-not-decoded -->

## C Variance Bounds

In this section, we study some properties of a DMDP. Most of the content in this section is similar to [AMK13]. We provide slight modifications and improvement to make the results fit to our application. The main result of this section is to show the following lemma.

Lemma C.1 (Upper Bound on Variance) . For any π , we have

<!-- formula-not-decoded -->

Before we prove this lemma, we introduce another notation. We define Σ π ∈ R |S||A| for all ( s, a ) ∈ S × A by where σ v π = P π ( v π ) 2 -( P π v π ) 2 is the 'one-step' variance of playing policy π .

<!-- formula-not-decoded -->

where a t = π ( s t ). Thus Σ π is the variance of the reward of starting with ( s, a ) and play π for infinite steps. The crucial observation of obtaining the near-optimal sample complexity is the following 'Bellman Equation' for variance. It is a consequence of 'the law of total variance'.

Lemma C.2 (Bellman Equation for variance) . Σ π satisfies the Bellman equation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By direct expansion,

<!-- formula-not-decoded -->

Combining the above two equations, we conclude the proof.

As a remark, we note that

<!-- formula-not-decoded -->

Furthermore, by definition, we have

<!-- formula-not-decoded -->

The next lemma is crucial in proving the error bounds.

Lemma C.3. Let P ∈ R n × n be a non-negative matrix in which every row has /lscript 1 norm at most 1 , i.e. /lscript ∞ operator norm at most 1 . Then for all γ ∈ (0 , 1) and v ∈ R n ≥ 0 we have

Proof. Since, every row of P has /lscript 1 norm at most 1, by Cauchy-Schwarz for i ∈ [ n ] we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since v is non-negative and applying P preserves non-negativity, applying this inequality repeatedly yields that P k √ v ≤ √ P k v entrywise for all k &gt; 0. Consequently, Cauchy-Schwarz again yields

Next, as ( I -γ P )( I + γ P ) = ( I -γ P 2 ) we see that ( I -γ P ) -1 = ( I + γ P )( I -γ 2 P ) -1 . Furthermore, as ‖ P x ‖ ∞ ≤ ‖ x ‖ ∞ for all x we have ‖ ( I + γ P ) x ‖ ∞ ≤ (1 + γ ) ‖ x ‖ ∞ for all x and therefore ‖ ( I -γ P ) -1 v ‖ ∞ ≤ (1 + γ ) ‖ ( I -γ 2 P ) -1 v ‖ ∞ as desired.

<!-- formula-not-decoded -->

We are now ready to prove Lemma C.1.

Proof of Lemma C.1. The lemma follows directly from the application of Lemma C.3. This proof is slightly simpler, tighter, and more general than the one in [AMK13].

## D Lower Bounds on Policy

Lemma D.1. Suppose M = ( S , A , P, γ, r ) is a DMDP with an sampling oracle. Suppose π is a given policy. Then there is an algorithm, halts in ˜ O ((1 -γ ) -3 /epsilon1 -2 |S| ) time, outputs a vector v such that, with high probability, ‖ v π -v ‖ ∞ ≤ /epsilon1 .

Proof. The lemma follows from a direct application of Theorem B.2.

Remark D.2 . Suppose |A| = ˜ Ω(1). Suppose there is an algorithm that obtains an /epsilon1 -optimal policy with Z samples, then the above lemma implies an algorithm for obtaining an /epsilon1 -optimal value function with Z + ˜ O ((1 -γ ) -3 /epsilon1 -2 |S| ) samples. By the Ω((1 -γ ) -3 /epsilon1 -2 |S||A| ) sample bound on obtaining approximate value functions given in [AMK13], the above lemma implies a

<!-- formula-not-decoded -->

sample lower bound for obtaining an /epsilon1 -optimal policy.

## E Missing Proofs

Here are several standard properties of the Bellman value operator (see, e.g., [Ber13]).

Fact 1. Let v 1 , v 2 ∈ R S be two vectors. Let T be a value operator of a DMDP with discount factor γ . Let π ∈ A S be an arbitrary policy. Then the follows hold.

- Monotonicity : If v 1 ≤ v 2 then T ( v 1 ) ≤ T ( v 2 ) ;
- Contraction : ‖T ( v 1 ) -T ( v 2 ) ‖ ∞ ≤ γ ‖ v 1 -v 2 ‖ ∞ and ‖T π ( v 1 ) -T π ( v 2 ) ‖ ∞ ≤ γ ‖ v 1 -v 2 ‖ ∞ .

## E.1 Missing Proofs from Section 5

To begin, we introduce two standard concentration results. Let p ∈ ∆ S be a probability vector, and v ∈ R S be a vector. Let p m ∈ ∆ S be empirical estimations of p using m i.i.d. samples from the distribution p . For instance, let these samples be s 1 , s 2 , . . . , s m ∈ S , then ∀ s ∈ S : p m ( s ) = ∑ m j =1 1 ( s j = s ) /m .

<!-- formula-not-decoded -->

Theorem E.1 (Hoeffding Inequality) . Let δ ∈ (0 , 1) be a parameter, vectors p , p m and v defined above. Then with probability at least 1 -δ ,

Theorem E.2 (Bernstein Inequality) . Let δ ∈ (0 , 1) be a parameter, vectors p , p m and v defined as in Theorem E.1. Then with probability at least 1 -δ

where Var s ′ ∼ p ( v ( s ′ )) = p /latticetop v 2 -( p /latticetop v ) 2 .

<!-- formula-not-decoded -->

Proof of Lemma 5.1. By Theorem E.2 and a union bound over all ( s, a ) pairs, with probability at least 1 -δ/ 4, for every ( s, a ), we have which is the first inequality.

Next, by Theorem E.1 and a union bound over all ( s, a ) pairs, with probability at least 1 -δ/ 4, for every ( s, a ), we have

<!-- formula-not-decoded -->

which we condition on. Thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since P /latticetop s,a v (0) ≤ ‖ v (0) ‖ ∞ , we obtain provided 2 m -1 1 L ≤ 1. Next by Lemma E.1 and a union bound over all ( s, a ) pairs, with probability at least 1 -δ/ 4, for every ( s, a ), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By a union bound, we obtain, with probability at least 1 -δ/ 2,

<!-- formula-not-decoded -->

By a union bound, with probability at least 1 -δ , both (E.1) and (E.2) hold, concluding the proof.

Proof of Lemma 5.2. Since for each ( s, a ), σ v ( s, a ) is a variance, then we have triangle inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We conclude the proof by taking a square root of all three sides of the above inequality.

Proof of Lemma 5.3. Recall that for each ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

where m 2 = 128(1 -γ ) -2 · log(2 |S||A| R/δ ) and s (1) s,a , s (2) s,a , . . . , s ( m 2 ) s,a is a sequence of independent samples from P s,a . Thus by Theorem E.1 and a union bound over S ×A , with probability at least 1 -δ/R , we have

Finally by shifting the estimate to have one-sided error, we obtain the one-side error (1 -γ ) u/ 4 in the statement of this lemma.

<!-- formula-not-decoded -->

Observing that

Proof of Lemma 5.4. For i = 0, Q (0) = r + γ w . By Lemma 5.1, with probability at least 1 -δ , and

<!-- formula-not-decoded -->

which we condition on. We have

Thus

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By (E.3) and Lemma 5.2, we have we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the rest of the proof, we condition on the event that (E.4) and (E.5) hold, which happens with probability at least 1 -δ . Denote v ( -1) = 0 . Thus we have v ( -1) ≤ v (0) ≤ T π (0) ( v (0) ). Next we prove the lemma by induction on i . Assume for some i ≥ 1, with probability at least 1 -( i -1) δ ′ the following holds,

<!-- formula-not-decoded -->

which we condition on. Next we show that the lemma statement holds for k = i . By definition of v ( i ) (Line 27 and 28),

Furthermore, since v (0) ≤ v (1) ≤ . . . ≤ v ( i -1) ≤ T π i -1 v ( i -1) ≤ T v ( i -1) ≤ T ∞ v ( i -1) = v ∗ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Lemma 5.3, we have, with probability at least 1 -δ ′

<!-- formula-not-decoded -->

which we condition on for the rest of the proof. Thus we have

<!-- formula-not-decoded -->

To show v ( i ) ≤ T π ( i ) v ( i ) , we notice that if for some s , π ( i ) ( s ) = π ( i -1) ( s ), then

/negationslash

<!-- formula-not-decoded -->

where the first inequality follows from v ( i ) ( s ) ≤ r ( s, π ( i ) ( s )) + γ P /latticetop s,π ( i ) ( s ) v ( i -1) = T π ( i ) v ( i -1) . On the other hand, if π ( i ) ( s ) = π ( i -1) ( s ), then

<!-- formula-not-decoded -->

This completes the induction step. Lastly, combining (E.5) and (E.6), we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

where α 1 = log(8 |S||A| δ -1 ) /m 1 ≤ 1. Mover, since v ( Q ( i -1) ) ≤ v ( i ) , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where π ∗ is an arbitrary optimal policy and we use the fact that max a Q ∗ ( s, a ) = Q ∗ ( s, π ∗ ( s )). This completes the proof of the lemma.

Proof of Proposition 5.4.1. Recall that we are able to sample a state from each P s,a with time O (1). Let β = (1 -γ ) -1 , R = /ceilingleft c 1 β ln[ βu -1 ] /ceilingright , m 1 = c 2 β 3 u -2 · log(8 |S||A| δ -1 ) and m 2 = c 3 β 2 · log[2 R |S||A| δ -1 ] for some constants c 1 , c 2 and c 3 required in Algorithm 1. In the following proof, we set c 1 , c 2 , c 3 to be sufficiently large but otherwise arbitrary absolute constants (e.g., c 1 ≥ 4 , c 2 ≥ 8192 , c 3 ≥ 128). By Lemma 5.4, with probability at least 1 -2 δ for each 1 ≤ i ≤ R , we have v ( i -1) ≤ v ( i ) ≤ T π ( i ) v ( i ) , and Q ( i ) ≤ r + γ Pv ( i ) , where

<!-- formula-not-decoded -->

for α 1 = log(8 |S||A| δ -1 ) /m 1 and sufficiently large constant C . Solving the recursion, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first apply a na¨ ıve bound ‖ P π ∗ [ Q ∗ -Q 0 ] ‖ ∞ ≤ (1 -γ ) -1 . Hence

where R = /ceilingleft (1 -γ ) -1 ln[4(1 -γ ) -1 u -1 ] /ceilingright +1. The next step is the key to the improvement in our analysis. We further apply the bound in Lemma C.1, given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality follows since min(2 γ -1 , (1 -γ ) -1 / 2 ) ≤ 3. With ‖ ( I -γ P π ∗ ) -1 1 ‖ ∞ ≤ (1 -γ ) -1 and ‖ v (0) ‖ ∞ ≤ (1 -γ ) -1 , we have, for some sufficiently large C ′ and C ′′ , which depend on c 1 , c 2 and c 3 . Since v ( Q ( R -1) ) ≤ v ( R ) , we have

This completes the proof of the correctness. It remains to bound the time complexity. The initialization stage costs O ( m 1 ) time per ( s, a ). Each iteration costs O ( m 2 ) time per ( s, a ). We thus have the total time complexity as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since log[(1 -γ ) -1 u -1 ] = O (log[(1 -γ ) -1 ] u -2

<!-- formula-not-decoded -->

## E.2 Missing Analysis of Halving Errors

We refer in this section to Algorithm 1 as a subroutine HalfErr , which given an input MDP M with a sampling oracle, an input value function v ( i ) and an input policy π ( i ) , outputs an value function v ( i +1) and a policy π ( i +1) such that, with high probability (over the new samples of the sampling oracle),

<!-- formula-not-decoded -->

After log[ /epsilon1 -1 (1 -γ ) -1 ] calls of the subroutine HalfErr , the final output policy and value functions are /epsilon1 -close to the optimal ones with high probability.

We summarize our meta algorithm in Algorithm 2. Note that in the algorithm, each call of HalfErr will draw new samples from the sampling oracle. These new samples guarantee the independence of successive improvements and also save space of the algorithm. For instance, the algorithm HalfErr only needs to use O ( |S||A| ) words of memory instead of storing all the samples. The guarantee of the algorithm is summarized in Proposition E.2.1.

Proposition E.2.1. Let M = ( S , A , r , P , γ ) with a sampling oracle. Suppose HalfErr is an algorithm that takes an input v ( i ) and an input policy π ( i ) and a number u ∈ [0 , (1 -γ ) -1 ] satisfying v ∗ -u 1 ≤ v ( i ) ≤ v π ( i ) , halts in time τ and outputs a v ( i +1) and a policy π ( i +1) satisfying,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -(1 -γ ) · /epsilon1 · δ (over the randomness of the new samples given by the sampling oracle), then the meta algorithm described in Algorithm 2, given input M and the sampling oracle, halts in τ · log( /epsilon1 -1 · (1 -γ ) -1 ) and outputs an policy π ( R ) such that

<!-- formula-not-decoded -->

with probability at least 1 -δ (over the randomness of all samples drawn from the sampling oracle). Moreover, if HalfErr uses space s , then the meta algorithm uses space s + O ( |S||A| ). If each call of HalfErr takes m samples from the oracle, then the overall samples taken by Algorithm 2 is m · log( /epsilon1 -1 · (1 -γ ) -1 ).

The proof of this proposition is a straightforward application of conditional probability.

Proof of Proposition E.2.1. The proof follows from a straightforward induction. For simplicity, denote β = (1 -γ ) -1 . In the meta-algorithm, the initialization is v (0) = 0 and π (0) is an arbitrary policy. Thus v ∗ -β · 1 ≤ v (0) ≤ v π (0) . By running the meta-algorithm, we obtain a sequence of value functions and policies: { v ( i ) } R i =0 and { π ( i ) } R i =0 . Since each call of the HalfErr uses new samples from the oracle, the sequence of value functions and policies satisfies strong Markov property (given ( v ( i ) , π ( i ) ), ( v ( i +1) , π ( i +1) ) is independent with { ( v ( j ) , π ( j ) ) } i -1 j =0 ). Thus

<!-- formula-not-decoded -->

Since 2 -R (1 -γ ) -1 ≤ /epsilon1 , we conclude the proof.

Proof of Theorem 5.5. Our algorithm is simply plugging in Algorithm 1 as the HalfErr subroutine in Algorithm 2. The correctness is guaranteed by Proposition E.2.1 and Proposition 5.4.1. The running time guarantee follows from a straightforward calculation.

## F Extension to Finite Horizon

In this section we show how to apply similar techniques to achieve improved sample complexities for solving finite Horizon MDPs given a generative model and we prove that the sample complexity we achieve is optimal up to logarithmic factors.

The finite horizon problem is to compute an optimal non-stationary policy over a fixed time horizon H , i.e. a policy of the form π ( s, h ) for s ∈ S and h ∈ { 0 , . . . H } ), where the reward is the expected cumulative (un-discounted) reward for following this policy. In classic value iteration, this is typically done using a backward recursion from time H,H -1 , . . . 0. We show how to use the ideas in this paper to solve for an /epsilon1 -approximate policy. As we have shown in the discounted case, it is suffice to show an algorithm that decrease the error of the value at each stage by half. Our algorihtm is presented in Algorithm 3.

To analyze the algorithm, we first provide an analogous lemma of Lemma 5.1,

Lemma F.1 (Empirical Estimation Error) . Let ˜ w h and ̂ σ h be computed in Line 10 of Algorithm 3. Recall that ˜ w h and ̂ σ h are empirical estimates of Pv h and σ v h = Pv 2 h -( Pv h ) 2 using m 1 samples per ( s, a ) pair. Then with probability at least 1 -δ , for L def = log(8 |S||A| δ -1 ) and every h = 1 , 2 , . . . , H , we have and

<!-- formula-not-decoded -->

Proof. The proof of this lemma is identical to that of Lemma 5.1.

<!-- formula-not-decoded -->

An analogous lemma to Lemma 5.3 is also presented here.

Lemma F.2. Let g ( i ) h be the estimate of P [ v ( i ) h -v (0) h ] defined in Line 27 of Algorithm 3. Then conditioning on the event that ‖ v ( i ) h -v (0) h ‖ ∞ ≤ 2 /epsilon1 , with probability at least 1 -δ/H , provided appropriately chosen constants in Algorithm 3.

<!-- formula-not-decoded -->

Proof. The proof of this lemma is identical to that of Lemma 5.3 except that (1 -γ ) -1 is replaced with H .

Similarly, we can show the following improvement lemma.

Lemma F.3. Let Q h be the estimated Q -function of v h +1 in Line 30 of Algorithm 3. Let Q ∗ h = r + P h v ∗ h +1 be the optimal Q -function of the DMDP. Let π ( · , h ) and v h be estimated in iteration h , as defined in Line 24 and 25. Let π ∗ be an optimal policy for the DMDP. For a policy π , let P π h Q ∈ R S×A be defined as ( P π h Q )( s, a ) = ∑ s ′ ∈S P s,a ( s ′ ) Q ( s ′ , π ( s ′ , h )) . Suppose for all h ∈ [ H -1] , v (0) h ≤ T π (0) ( · ,h ) v (0) h +1 . Let v H +1 def = 0 and Q H +1 def = 0 . Then, with probability at least 1 -2 δ , for all 1 ≤ h ≤ H , v (0) h ≤ v h ≤ T π ( · ,h ) v h +1 ≤ v ∗ h , Q h ≤ r + P h v h +1 and where the error vector ξ h satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and α 1 = log(8 |S||A| Hδ -1 ) /m 1 .

Proof of Lemma F.3. By Lemma 5.1, for any h = 1 , 2 , . . . , H , with probability at least 1 -δ/H , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which we condition on. We have

Thus

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

By (E.3) and Lemma 5.2, we have we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the rest of the proof, we condition on the event that (F.4) and (F.5) hold for all h = 1 , 2 , . . . , H , which happens with probability at least 1 -δ . Denote v ∗ H +1 = v H +1 = v (0) H +1 = 0 . Thus we have v (0) H +1 ≤ v H +1 ≤ v ∗ H +1 . Next we prove the lemma by induction on h . Assume for some h , with probability at least 1 -( h -1) δ/H the following holds, for all h ′ = h +1 , h +2 , . . . , H,

<!-- formula-not-decoded -->

which we condition on. Next we show that the lemma statement holds for h as well. By definition of v h (Line 27 and 28),

<!-- formula-not-decoded -->

Furthermore, since v (0) h +1 ≤ v ∗ h +1 ≤ v (0) h +1 + u 1 we have

<!-- formula-not-decoded -->

By Lemma 5.3, we have, with probability at least 1 -δ ′

which we condition on for the rest of the proof. Thus we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To show v h ≤ T π ( · ,h ) v h +1 , we notice that if for some s , π ( s, h ) = π (0) ( s, h ), then,

/negationslash

<!-- formula-not-decoded -->

On the other hand, if π ( s, h ) = π (0) ( s, h ), then

<!-- formula-not-decoded -->

This completes the induction step. Lastly, combining (F.5) and (F.6), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where where α 1 = log(8 |S||A| δ -1 ) /m 1 . Mover, since v ( Q h +1 ) ≤ v h +1 , we obtain

where π ∗ is an arbitrary optimal policy and we use the fact that max a Q ∗ h ( s, a ) = Q ∗ h ( s, π ∗ ( s, h )). This completes the proof of the lemma.

<!-- formula-not-decoded -->

Furthermore, we show an analogous lemma of Lemma C.1.

Lemma F.4 (Upper Bound on Variance) . For any π , we have

Proof. First, by Cauchy-Swartz inequality, we have

<!-- formula-not-decoded -->

Next, by a similar argument of the proof of Lemma C.2, we can show that

<!-- formula-not-decoded -->

This completes the proof.

<!-- formula-not-decoded -->

We are now ready to present the guarantee of the algorithm 3.

Proposition F.4.1. On an input value vectors v (0) 1 , v (0) 2 , . . . , v (0) H , policy π (0) , and parameters u ∈ (0 , β ] , δ ∈ (0 , 1) such that v (0) h ≤ T π (0) ( · ,h ) v (0) h +1 for all h ∈ [ H -1], and v (0) h ≤ v ∗ h ≤ v (0) h + u 1 , Algorithm 3 halts in time O [ u -2 · H 4 |S||A| · log( |S||A δ -1 Hu -1 )] and outputs v 1 , v 2 , . . . , v H and π : S × [ H ] →A such that

<!-- formula-not-decoded -->

with probability at least 1 -δ , provided appropriately chosen constants, c 1 , c 2 and c 3 , in Algorithm 3. Moreover, the algorithm uses O [ u -2 · H 3 |S||A| · log( |S||A δ -1 Hu -1 )] samples from the sampling oracle.

Proof of Proposition F.4.1. Recall that we are able to sample a state from each P s,a with time O (1). Let R = /ceilingleft c 1 H ln[ Hu -1 ] /ceilingright , m 1 = c 2 H 3 u -2 · log(8 |S||A| δ -1 ) and m 2 = c 3 H 2 · log[2 R |S||A| δ -1 ] for some constants c 1 , c 2 and c 3 required in Algorithm 1. In the following proof, we set c 1 = 4 , c 2 = 8192 , c 3 = 128. By Lemma 5.4, with probability at least 1 -2 δ for each 1 ≤ h ≤ H , we have v (0) h ≤ v h ≤ T π ( · ,h ) v h , and Q h ≤ r + Pv h +1 , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and α 1 = log(8 |S||A| δ -1 ) /m 1 . Notice that v (0) H = v ∗ H = v ( r ), thus the v H -v ∗ H = 0 . Solving the recursion, we obtain

<!-- formula-not-decoded -->

The next step is the key to the improvement in our analysis. We further apply the bound in Lemma C.1, given by

With ‖ ∑ H -1 h ′ = h ∏ h ′ i = h +1 P π ∗ i 1 ‖ ∞ ≤ H -h +1 and ‖ v (0) h ‖ ∞ ≤ H , we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

provided

Since v ( Q h ) ≤ v h , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This completes the proof of the correctness. It remains to bound the time complexity. The initialization stage costs O ( m 1 ) time per ( s, a ) per stage h . Each iteration costs O ( m 2 ) time per ( s, a ). We thus have the total time complexity as

<!-- formula-not-decoded -->

The total number of samples used is

<!-- formula-not-decoded -->

This completes the proof.

We can then use our meta-algorithm and obtain the following theorem.

Theorem F.5. Let M = ( S , A , P , r , H ) be a H -MDP with a sampling oracle. Suppose we can sample a state from each probability vector P s,a within time O (1) . Then there exists an algorithm that runs in time and obtains a policy π such that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where v ∗ h is the optimal value of M at stage h . Moreover, the number of samples used by the algorithm is

<!-- formula-not-decoded -->

## F.1 Sample Lower Bound On H -MDP

In this section we show that the sample complexity obtained by the algorithm in the last section is essentially tight. Our proof idea is simple, we will reduce the H -MDP problem to a discounted MDP problem. If there is an algorithm that solves an H -MDP to obtain an /epsilon1 -optimal value, it also gives an value function to the discounted MDP. Therefore, the lower bound of solving H -MDP inherits from that of the discounted MDP. The formal guarantee is presented in the following theorem.

Theorem F.6. Let S and A be finite sets of states and actions. Let H &gt; 0 be a positive integer and /epsilon1 ∈ (0 , 1 / 2) be an error parameter. Let K be an algorithm that, on input an H -MDP M def = ( S , A , P, r ) with a sampling oracle, outputs a value function v 1 for the first stage, such that ‖ v 1 -v ∗ 1 ‖ ∞ ≤ /epsilon1 with probability at least 0 . 9 . Then K calls the sampling oracle at least Ω( H -3 /epsilon1 -2 |S||A| / log /epsilon1 -1 ) times on some input P and r ∈ [0 , 1] S .

Proof. Let s 0 ∈ S be a state. Denote S ′ = S\{ s 0 } be a subset of S . Let γ ∈ (0 , 1) be such that (1 -γ ) -1 log /epsilon1 -1 ≤ H . Suppose we have an DMDP M ′ = ( S ′ , A , P ′ , γ, r ′ ) with a sampling oracle. Let v ∗ ′ be the optimal value function of M ′ . Note that v ∗ ′ ∈ R S ′ . We will show, in the next paragraph, an H -MDP M = ( S , A , P, H, r ) with first stage value v ∗ 1 , such that ‖ v ∗ 1 | S ′ -v ∗ ′ ‖ ≤ /epsilon1 . Therefore, an /epsilon1 -approximation of v ∗ 1 gives a 2 /epsilon1 -approximation to v ∗ . We show that K can be used to obtain an /epsilon1 -approximate value v 1 for v ∗ 1 of M and thus K inherits the lower bound for obtaining (2 /epsilon1 )-approximated value for γ -DMDPs.

For M , in each state s ∈ S ′ , for any action there is a (1 -γ ) probability transiting to s 0 and γ probability to do the original transitions in M ′ ; for s 0 , no matter what action taken, it transits to itself with probability 1. Formally, for each state s, s ′ ∈ S ′ , a ∈ A , P ( s ′ | s, a ) = γ · P ′ ( ·| s, a ) and P ( s 0 | s, a ) = (1 -γ ); P ( s ′ | s 0 , a ) = 0 and P ( s 0 | s 0 , a ) = 1. For r , we set r ( s 0 , · ) = 0 and

<!-- formula-not-decoded -->

## Algorithm 3 FiniteHorizonRandomQVI

```
1: Input: M = ( S , A , r , P ) with a sampling oracle, v (0) 1 , v (0) 2 , . . . , v (0) H , π (0) : S × [ H ] →A , u, δ (0 , 1); 2: \\ u is the initial error, π (0) is the input policy, and δ is the error probability 3: Output: v 1 , v 2 , . . . , v H , π 4: 5: INITIALIZATION: 6: Let m 1 ← c 1 H 3 u -2 log(8 |S||A| δ -1 ) for constant c 1 ; 7: Let m 2 ← c 2 H 2 log[2 H |S||A| δ -1 ] for constant c 2 ; 8: Let α 1 ← m -1 1 log(8 |S||A| δ -1 ); 9: For each ( s, a ) ∈ S × A , sample independent samples s (1) s,a , s (2) s,a , . . . , s ( m 1 ) s,a from P s,a ; 10: Initialize w h = ˜ w h = ̂ σ h = Q (0) h ← 0 S×A for all h ∈ [ H ], and i ← 0; 11: Denote v H +1 ← 0 and Q H +1 ← 0 12: for each ( s, a ) ∈ S × A , h ∈ [ H ] do 13: \\ Compute empirical estimates of P /latticetop s,a v (0) h and σ v (0) h ( s, a ) 14: Let ˜ w h ( s, a ) ← 1 m 1 ∑ m 1 j =1 v (0) h ( s ( j ) s,a ) 15: Let ̂ σ h ( s, a ) ← 1 m 1 ∑ m 1 j =1 ( v (0) h ) 2 ( s ( j ) s,a ) -˜ w 2 h ( s, a ) 16: 17: \\ Shift the empirical estimate to have one-sided error 18: w h ( s, a ) ← ˜ w h ( s, a ) -√ 2 α 1 ̂ σ h ( s, a ) -4 α 3 / 4 1 ‖ v (0) h ‖ ∞ -(2 / 3) α 1 ‖ v (0) h ‖ ∞ 19: Let v H +1 ← 0 and Q H +1 ← 0 . 20: 21: REPEAT: \\ successively improve 22: for h = H,H -1 to 1 do 23: \\ Compute P /latticetop s,a [ v h -v (0) h ] with one-sided error 24: Let ˜ v h ← v h ← v ( Q h +1 ), ˜ π ( · , h ) ← π ( · , h ) ← π ( Q h +1 ), v h ← ˜ v h ; 25: For each s ∈ S , if ˜ v h ( s ) ≤ v (0) h ( s ), then v h ( s ) ← v (0) h ( s ) and π ( s, h ) ← π (0) ( s, h ); 26: For each ( s, a ) ∈ S × A , sample independent samples ˜ s (1) s,a , ˜ s (2) s,a , . . . , ˜ s ( m 2 ) s,a from P s,a ; 27: Let g h ( s, a ) ← m -1 2 ∑ m 2 j =1 [ v h (˜ s ( j ) s,a ) -v (0) h (˜ s ( j ) s,a ) ] -H -1 u/ 8; 28: 29: \\ Improve Q h : 30: Q h ← r + w h + g h ; 31: return v 1 , v 2 , . . . , v H , π .
```

- ∈

r ( s, · ) = r ′ ( s, · ) for s ∈ S ′ . It remains to show that ‖ v ∗ 1 | S ′ -v ∗ ‖ ∞ ≤ /epsilon1 . First we note that v ( r ) = v ∗ H ≤ v ∗ . Then, by monotonicity of the T operator, we have, for all h ∈ [ H -1] and s ∈ S ′ ,

<!-- formula-not-decoded -->

In particular, v ∗ 1 | S ′ ≤ v ∗ ′ . Since the optimal policy π ∗ ′ of M ′ can be used as a policy for the H -MDP as a non-optimal one, we have

<!-- formula-not-decoded -->

This completes the proof.

The above lower bound with our algorithm also implies a sample lower bound for an /epsilon1 -policy.

Corollary F.7. Let S and A be finite sets of states and actions. Let H &gt; 0 be a positive integer and /epsilon1 ∈ (0 , 1 / 2) be an error parameter. Let K be an algorithm that, on input an H -MDP M := ( S , A , P, r ) with a sampling oracle, outputs a policy π : S× [ H ] →A , such that ∀ h : ‖ v π h -v ∗ h ‖ ∞ ≤ /epsilon1 with probability at least 0 . 9 . Then K calls the sampling oracle at least Ω( H -3 /epsilon1 -2 |S||A| / log /epsilon1 -1 ) times on the worst case input P and r ∈ [0 , 1] S .