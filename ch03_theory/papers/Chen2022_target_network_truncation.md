## TARGET NETWORK AND TRUNCATION OVERCOME THE DEADLY TRIAD IN Q -LEARNING

BY ZAIWEI CHEN 1,* , JOHN-PAUL CLARKE 2,‡ , AND SIVA THEJA MAGULURI 1,†

1 Georgia Institute of Technology, * zchen458@gatech.edu; † siva.theja@gatech.edu

2 The University of Texas at Austin, ‡ johnpaul@utexas.edu

Q -learning with function approximation is one of the most empirically successful while theoretically mysterious reinforcement learning (RL) algorithms, and was identified in Sutton (1999) as one of the most important theoretical open problems in the RL community. Even in the basic linear function approximation setting, there are well-known divergent examples. In this work, we show that target network and truncation together are enough to provably stabilize Q -learning with linear function approximation, and we establish the finite-sample guarantees. The result implies an ˜ O ( glyph[epsilon1] -2 ) sample complexity up to a function approximation error. Moreover, our results do not require strong assumptions or modifying the problem parameters as in existing literature.

1. Introduction. The Deep Q -Network (Mnih et al., 2015), as a typical example of Q -learning with function approximation, is one of the most successful algorithms to solve the reinforcement learning (RL) problem, and hence is viewed as a milestone in the development of modern RL. On the other hand, the behavior of Q -learning with function approximation is theoretically not well understood, and was identified in Sutton (1999) as one of four most important theoretical open problems. In fact, the infamous deadly triad (Sutton, 2015) is present in Q -learning with function approximation, and hence even in the basic setting where linear function approximation is used, the algorithm was shown to be unstable in general (Baird, 1995).

While theoretically unclear, it was empirically evident from Mnih et al. (2015) that the following three ingredients: experience replay , target network , and truncation together overcome the divergence of Q -learning with function approximation. In this work, we focus on Q -learning with linear function approximation for infinite horizon discounted Markov decision processes (MDPs), and show theoretically that target network together with truncation is sufficient to provably stabilize Q -learning. The main contributions of this work are summarized in the following.

- Finite-Sample Guarantees. We establish finite-sample guarantees of the output of Q -learning with target network and truncation to the optimal Q -function Q ∗ up to a function approximation error. This is the first variant of Q -learning with linear function approximation that is provably stable (without needing strong assumptions), and uses a single trajectory of Markovian samples. The result implies an ˜ O ( glyph[epsilon1] -2 ) sample complexity, which matches with the sample complexity of Q -learning in the tabular setting, and is known to be optimal up to a logarithmic factor. The function approximation error in our finite-sample bound well captures the approximation power of the chosen function class. In the special case of tabular setting, or assuming the function class is closed under Bellman operator, our result implies asymptotic convergence in the mean-square sense to the optimal Q -function Q ∗ .
- Broad Applicability. In existing literature, to stabilize Q -learning with linear function approximation, one usually requires strong assumptions on the underlying MDP and/or the approximating function class. Those assumptions include but not limited to the function class being complete with respect to the Bellman operator, the MDP being linear (or close to linear), and a so-called strong negative drift assumption, etc. In this work, we do not require any of those assumptions. Specifically, our result holds as long as the policy used to collect samples enables the agent to sufficiently explore the state-action space, which is to some extent a necessary requirement to find an optimal policy in RL.

1.1. Related Work. The Q -learning algorithm was first proposed in Watkins and Dayan (1992). Since then, theoretically understanding the behavior of Q -learning has been a major topic in the RL community. In particular, the asymptotic convergence of Q -learning was established in Tsitsiklis (1994); Jaakkola, Jordan and Singh (1994); Borkar and Meyn (2000); Lee and He (2020), and the asymptotic convergence rate in Szepesvári (1998); Devraj and Meyn (2017). Beyond the asymptotic behavior, recently there has been an increasing interest in studying finite-sample convergence guarantees of Q -learning. Here is a nonexhaustive list: Even-Dar and Mansour (2003); Beck and Srikant (2012, 2013); Chen et al. (2020, 2021); Chandak and Borkar (2021); Borkar (2021); Li et al. (2020, 2021a,b); Wainwright (2019a,b); Jin et al. (2018); Qu and Wierman (2020), leading to the optimal ˜ O ( glyph[epsilon1] -2 ) sample complexity of Q -learning. Other variants of Q -learning such as zap Q -learning and double Q -learning were proposed and studied in Devraj and Meyn (2017) and Hasselt (2010), respectively.

When using function approximation, the infamous deadly triad (i.e., function approximation, off-policy sampling, and bootstrapping) (Sutton and Barto, 2018) appears in Q -learning, and the algorithm can be unstable even when linear function approximation is used. This is evident from the divergent MDP example constructed in Baird (1995). There are many attempts to stabilize Q -learning with linear function approximation, which are summarised in the following. See Appendix E for a more detailed survey about existing results and their limitations.

Strong Negative Drift Assumption. The asymptotic convergence of Q -learning with linear function approximation was established in Melo, Meyn and Ribeiro (2008) under a 'negative drift' assumption. Under similar assumptions, the finite-sample analysis of Q -learning, as well as its on-policy variant SARSA, was performed in Chen et al. (2019); Gao et al. (2021); Lee and He (2020); Zou, Xu and Liang (2019) for using linear function approximation, and in Xu and Gu (2020); Cai et al. (2019) for using neural network approximation. However, such negative drift assumption is highly artificial, highly restrictive, and is impossible to satisfy unless the discount factor of the MDP is extremely small. See Appendix E.1 for a more detailed explanation. In this work, we do not require such negative drift assumption or any of its variants to stabilize Q -learning with linear function approximation.

Modifying the Problem Discount Factor. Very recently, new convergent variants of Q -learning with linear function approximation were proposed in Carvalho, Melo and Santos (2020); Zhang, Yao and Whiteson (2021), where target network was used in the algorithm. However, as we will see later in Section 4.3, target network alone is not sufficient to provably stabilize Q -learning. The reason that Carvalho, Melo and Santos (2020); Zhang, Yao and Whiteson (2021) achieve convergence of Q -learning is by implicitly modifying the discount factor. In fact, the problem they are effectively solving is no longer the original MDP, but an MDP with a much smaller discount factor, which is the reason why their algorithms do not converge to the optimal Q -function Q ∗ in the tabular setting. See Appendices E.2 and E.3 for more details. In this work we do not modify the original problem parameters to achieve stability, and in the special case of tabular RL, we have convergence to Q ∗ .

The Greedy-GQ Algorithm. Atwotime-scale variant of Q -learning with linear function approximation, known as Greedy-GQ, was proposed in Maei et al. (2010). The algorithm is designed based on minimizing the projected Bellman error using stochastic gradient descent. Although the Greedy-GQ algorithm is stable without needing the negative drift assumption, since the Bellman error is in general non-convex, GreedyGQ algorithm can only guarantee convergence to stationary points. As a result, there are no performance guarantees on how well the limit point approximates the optimal Q -function Q ∗ . Although finite-sample bounds for Greedy-GQ were recently established in Wang and Zou (2020); Ma et al. (2021); Xu and Liang (2021), due to the lack of global optimality, the finite-sample bounds were on the gradient of the Bellman error rather than the distance to Q ∗ . In this work we provide finite-sample guarantees to the optimal Q -function Q ∗ (up to a function approximation error).

Fitted Q -Iteration and Its Variants. Fitted Q -iteration is proposed in Ernst, Geurts and Wehenkel (2005) as an offline variant of Q -learning. The finite-sample guarantees of fitted Q -iteration (or more generally fitted value iteration) were established in Szepesvári and Munos (2005); Munos and Szepesvári (2008). More recently, Xie and Jiang (2020) proposes a variant of batch RL algorithms called BVFT, where the

authors establish an ˜ O ( glyph[epsilon1] -4 ) sample complexity under the realizability assumption. Notably, Szepesvári and Munos (2005); Munos and Szepesvári (2008) employed truncation technique to ensure the boundedness of the function approximation class. Such truncation technique dates back to Györfi et al. (2002). We use the same truncation technique in this paper. In the special case of linear function approximation, Q -learning with target network can be viewed as an approximate way of implementing the fitted Q -iteration, where stochastic gradient descent was used as a way of performing such fitting. Compared to Szepesvári and Munos (2005); Munos and Szepesvári (2008), the main difference of this work is that our algorithm is implemented in an online manner, and is driven by a single trajectory of Markovian samples.

Another variant of fitted Q -iteration targeting finite horizon MDPs was proposed in Du et al. (2019) using a distribution shift checking oracle. However, Du et al. (2019) requires the approximating function class to contain the optimal Q -function, and only polynomial sample complexity, i.e., ˜ O ( glyph[epsilon1] -n ) for some positive integer n , was established. In this work, we do not require Q ∗ to be within our chosen function class, and our algorithm achieves the optimal ˜ O ( glyph[epsilon1] -2 ) sample complexity.

Linear MDP Model. In the special case that the MDP has linear transition dynamics and linear reward, convergent variants of Q -learning with linear function approximation were designed and analyzed in Yang and Wang (2019, 2020); Jin et al. (2020); Zhou, He and Gu (2021); He, Zhou and Gu (2021); Li et al. (2021c). Such linear model assumption can be relaxed to the case where the MDP is approximately linear. In this work, we do not make any assumption on the underlying structure of the MDP, except the uniform ergodicity of the Markov chain induced by the behavior policy.

Other Work. Du et al. (2020) studies Q -learning with function approximation for deterministic MDPs. The Deep Q -Network was studied in Fan et al. (2020). See Appendix D.2 for a more detailed discussion about the Deep Q -Network.

2. Background on RL and Q -Learning. We model the RL problem as an infinite horizon discounted MDP defined by a 5 -tuple ( S , A , P , R , γ ) , where S is a finite set of states, A is a finite set of actions, P = { P a ∈ R |S|×|S| | a ∈ A} is a set of unknown transition probability matrices, R : S × A ↦→ [0 , 1] is an unknown reward function, and γ ∈ (0 , 1) is the discount factor. Our results can be generalized to continuousstate finite-action MDPs. We restrict our attention to finite-state setting for ease of exposition.

Define the state-action value function of a policy π by Q π ( s, a ) = E [ ∑ ∞ k =0 γ k R ( S k , A k ) | S 0 = s, A 0 = a ] for all ( s, a ) . The goal is to find an optimal policy π ∗ so that its associated Q -function (denoted by Q ∗ ) is maximized uniformly for all ( s, a ) . A well-known relation between the optimal Q -function and any optimal policy π ∗ states that π ∗ ( ·| s ) is supported on the set argmax a ∈A Q ∗ ( s, a ) for all s . Therefore, to find an optimal policy, it is enough the find the optimal Q -function, which is the motivation of the Q -learning algorithm. The Q -learning algorithm is designed to find Q ∗ by solving the Bellman equation Q ∗ = H ( Q ∗ ) using stochastic approximation, and provably converges. However, Q -learning becomes intractable for MDPs with large state-action space. This motivates the use of function approximation, where the idea is to approximate the optimal Q -function from a pre-specified function class.

In this work, we focus on using linear function approximation. Let φ i ∈ R |S||A| , i = 1 , 2 , ..., d , be a set of basis vectors, and denote φ ( s, a ) = ( φ 1 ( s, a ) , · · · , φ d ( s, a )) ∈ R d for all ( s, a ) . We assume without loss of generality that the basis vectors { φ i } 1 ≤ i ≤ d are linearly independent, and are normalized so that ‖ φ ( s, a ) ‖ 1 ≤ 1 for all ( s, a ) , where ‖ · ‖ 1 stands for the glyph[lscript] 1 -norm. Let Φ ∈ R |S||A|× d be defined by

<!-- formula-not-decoded -->

Using the feature matrix Φ , the linear sub-space spanned by { φ i } 1 ≤ i ≤ d can be compactly written as W = { Q θ ∈ R |S||A| | Q θ =Φ θ, θ ∈ R d } . The goal of Q -learning with linear function approximation is to design a stable algorithm that provably finds an approximation of the optimal Q -function Q ∗ from the linear subspace W .

3. Algorithm and Finite-Sample Guarantees. In this section, we first present the algorithm of Q -learning with linear function approximation using target network and truncation. Then we provide the finitesample guarantees of our algorithm to the optimal Q -function Q ∗ up to a function approximation error. The detailed proofs of all technical results are provided in the Appendix.
2. 3.1. Stable Algorithm Design. To present our algorithm, we introduce the truncation operator glyph[ceilingleft]·glyph[ceilingright] in the following. For any vector x , let glyph[ceilingleft] x glyph[ceilingright] be the resulting vector of x component-wisely truncated from both above and below at r = 1 / (1 -γ ) , i.e., for each component glyph[ceilingleft] x glyph[ceilingright] i of glyph[ceilingleft] x glyph[ceilingright] , we have glyph[ceilingleft] x glyph[ceilingright] i = r if x i &gt; r , glyph[ceilingleft] x glyph[ceilingright] i = x i if x i ∈ [ -r, r ] , and glyph[ceilingleft] x glyph[ceilingright] i = -r if x i &lt; -r . The reason that we pick the truncation level r to be 1 / (1 -γ ) is that ‖ Q ∗ ‖ ∞ ≤ 1 / (1 -γ ) . Therefore by performing truncation we do not exclude Q ∗ .

## Algorithm 1 Q -Learning with Linear Function Approximation: Target Network and Truncation

```
1: Input: Integers T , K , initializations θ t, 0 = 0 for all t =0 , 1 , ..., T -1 and ˆ θ 0 = 0 , behavior policy π b 2: for t =0 , 1 , · · · , T -1 do 3: for k =0 , 1 , · · · , K -1 do 4: Sample A k ∼ π b ( ·| S k ) , S k +1 ∼ P A k ( S k , · ) 5: θ t,k +1 = θ t,k + α k φ ( S k , A k )( R ( S k , A k ) + γ max a ′ ∈A glyph[ceilingleft] φ ( S k +1 , a ′ ) glyph[latticetop] ˆ θ t glyph[ceilingright]φ ( S k , A k ) glyph[latticetop] θ t,k ) 6: end for 7: ˆ θ t +1 = θ t,K 8: S 0 = S K 9: end for 10: Output: ˆ θ T
```

Several remarks are in order. First of all, Algorithm 1 is simple, easy to implement, and can be generalized to using arbitrary parametric function approximation in a straightforward manner (see Appendix D.2). Second, in addition to { θ t,k } , we introduce { ˆ θ t } as the target network parameter, which is fixed in the inner loop where we update θ t,k , and is synchronized to the last iterate θ t,K in the outer loop. Target network was first introduced in Mnih et al. (2015) for the design of the celebrated Deep Q -Network. Finally, before using the Q -function estimate associated with the target network in the inner-loop, we first truncate it at level r (see line 5 of Algorithm 1).

Note that the location where we impose the truncation operator is different from that in the Deep Q -Network (Mnih et al., 2015), where instead of only truncating φ ( S k +1 , a ′ ) glyph[latticetop] ˆ θ t , truncation is performed for the entire temporal difference R ( S k , A k )+ γ max a ′ ∈A φ ( S k +1 , a ′ ) glyph[latticetop] ˆ θ t -φ ( S k , A k ) glyph[latticetop] θ t,k . Similar truncation technique has been employed in Munos and Szepesvári (2008); Jin et al. (2018). The reason that target network and truncation together ensure the stability of Q -learning with linear function approximation will be illustrated in detail in Section 4.

On the practical side, Algorithm 1 uses a single trajectory of Markovian samples generated by the behavior policy π b (see line 4 and line 8 of Algorithm 1). Therefore, the agent does not have to constantly reset the system. Our result can be easily generalized to the case where one uses time-varying behavior policy (i.e., the behavior policy is updated across the iterations of the target network) as long as it ensures sufficient exploration. For example, one can use the glyph[epsilon1] -greedy policy or the Boltzmann exploration policy (aka. softmax policy) with respect to the Q -function estimate associated with the target network Q ˆ θ t as the behavior policy.

- 3.2. Finite-Sample Guarantees. To present the finite-sample guarantees of Algorithm 1, we first formally state our assumption about the behavior policy π b and introduce necessary notation.

ASSUMPTION 3.1. The behavior policy π b satisfies π b ( a | s ) &gt; 0 for all ( s, a ) , and induces an irreducible and aperiodic Markov chain { S k } .

This assumption ensures that the behavior policy sufficient explores the state-action space, and is commonly imposed for value-based RL algorithms in the literature Tsitsiklis and Van Roy (1997). Note that Assumption 3.1 implies that the Markov chain { S k } admits a unique stationary distribution, denoted by µ ∈ ∆ |S| , and mixes at a geometric rate (Levin and Peres, 2017). As a result, letting t δ = min { k ≥ 0 : max s ∈S ‖ P k π b ( s, · ) -µ ( · ) ‖ TV ≤ δ } be the mixing time of the Markov chain { S k } (induced by π b ) with precision δ &gt; 0 , then under Assumption 3.1 we have t δ = O (log(1 /δ )) .

Under Assumption 3.1, the Markov chain { ( S k , A k ) } also has a unique stationary distribution. Let D ∈ R |S||A|×|S||A| be a diagonal matrix with the unique stationary distribution of { ( S k , A k ) } on its diagonal, i.e., D (( s, a ) , ( s, a )) = µ ( s ) π b ( a | s ) for all ( s, a ) . Moreover, let a norm ‖·‖ D be defined by ‖ x ‖ D =( x glyph[latticetop] Dx ) 1 / 2 . Denote λ min as the minimum eigenvalue of the positive definite matrix Φ glyph[latticetop] D Φ .

Let Proj W ( · ) be the projection operator onto the linear sub-space W with respect to the weighted glyph[lscript] 2 -norm ‖· ‖ D . Note that Proj W ( · ) is explicitly given by Proj W ( Q ) = Φ(Φ glyph[latticetop] D Φ) -1 Φ glyph[latticetop] DQ for any Q ∈ R |S||A| . Let E approx := sup Q ∈W : ‖ Q ‖ ∞ ≤ r ‖glyph[ceilingleft] Proj W H ( Q ) glyph[ceilingright] - H ( Q ) ‖ ∞ , which captures the approximation power of the chosen function class. Denote ˆ Q t = glyph[ceilingleft] Φ ˆ θ t glyph[ceilingright] as the truncated Q -function estimate associated with the target network ˆ θ t .

We next present the finite-sample bounds. For ease of exposition, we only present the case where we use constant stepsize in the inner-loop of Algorithm 1, i.e., α k ≡ α . The results for using various diminishing stepsizes are straightforward extensions Chen et al. (2019).

THEOREM 3.1. Consider ˆ θ T of Algorithm 1. Suppose that Assumption 3.1 is satisfied, the constant stepsize α is chosen such that α ≤ λ min (1 -γ ) 2 130 , and K ≥ t α +1 . Then we have for any T ≥ 0 that

<!-- formula-not-decoded -->

E 3 : Variance in the inner-loop E 4 : Function approximation error

As a result, to obtain E [ ‖ ˆ Q T -Q ∗ ‖ ∞ ] ≤ glyph[epsilon1] + E approx 1 -γ for a given accuracy glyph[epsilon1] , the sample complexity is

<!-- formula-not-decoded -->

REMARK. While commonly used in existing literature studying RL with function approximation, it was argued in Khodadadian, Chen and Maguluri (2021) that sample complexity is strictly speaking not well-defined when the asymptotic error is non-zero. Here we present the 'sample complexity' in the same sense as in existing literature to enable a fair comparison.

Theorem 3.1 is by far the strongest result of Q -learning with linear function approximation in the literature in that it achieves the optimal ˜ O ( glyph[epsilon1] -2 ) sample complexity without needing strong assumptions or modifying the problem parameters.

In our finite-sample bound, the term E 1 goes to zero geometrically fast as T goes to infinity. In fact, the term E 1 captures the error due to fixed-point iteration. That is, if we had a complete basis (hence no function approximation error), and were able to perform value iteration to solve the Bellman equation Q ∗ = H ( Q ∗ ) (hence no stochastic error), E 1 is the only error term.

The terms E 2 and E 3 represent the bias and variance in the inner-loop of Algorithm 1. Since the target network parameter ˆ θ t is fixed in the inner-loop, the update equation in Algorithm 1 line 5 can be viewed as a linear stochastic approximation algorithm under Markovian noise. When using constant stepsize, the bias

goes to zero geometrically fast as K goes to infinity but the variance is a constant proportional to √ αt α . Since geometric mixing implies t α = O (log(1 /α )) , the term √ αt α can be made arbitrarily small by using small enough constant stepsize. This agrees with existing literature studying linear stochastic approximation Srikant and Ying (2019). When using diminishing stepsizes with a suitable decay rate, one can easily show using results in Chen et al. (2019) that both E 1 and E 2 go to zero at a rate of O (1 / √ K ) , therefore the resulting sample complexity is the same as when using constant stepsize.

The term E 4 captures the error due to using function approximation. Recall that we define E approx = sup Q ∈W : ‖ Q ‖ ∞ ≤ r ‖glyph[ceilingleft] Proj W H ( Q ) glyph[ceilingright]-H ( Q ) ‖ ∞ . Therefore to make the function approximation error small, one only needs to approximate the functions that are one-step reachable under the Bellman operator. In addition, using truncation also helps reducing the function approximation error to some extend since ‖glyph[ceilingleft] Proj W H ( Q ) glyph[ceilingright]-H ( Q ) ‖ ∞ ≤‖ Proj W H ( Q ) -H ( Q ) ‖ ∞ for any Q such that ‖ Q ‖ ∞ ≤ r . The 1 / (1 -γ ) factor in E 4 also appears in TD-learning with linear function approximation Tsitsiklis and Van Roy (1997), where it was shown to be not removable in general. Observe that E 4 vanishes (and hence we have convergence to Q ∗ ) when (1) we are in the tabular setting, or (2) we use a complete basis (i.e., Φ being an invertible matrix), or (3) under the completeness assumption in existing literature, which requires H ( Q ) ∈W whenever Q ∈ W . In existing work Carvalho, Melo and Santos (2020); Zhang, Yao and Whiteson (2021), the algorithm does not converge to Q ∗ even in the tabular setting (see Appendix E).

4. The reason that Target Network and Truncation Stabilize Q -Learning. In the previous section, we presented the algorithm and the finite-sample guarantees. In this section, we elaborate in detail why target network and truncation together are enough to stabilize Q -learning.

Summary. We start with the classical semi-gradient Q -learning with linear function approximation algorithm in Section 4.1, which unfortunately is not necessarily stable, as evidenced by the divergent counterexample constructed in Baird (1995). In Section 4.2, We show that by adding target network to Q -learning, the resulting algorithm successfully overcomes the divergence in the MDP example in Baird (1995). However, beyond the example in Baird (1995), target network alone is not sufficient to stabilize Q -learning. In fact, we show in Section 4.3 that Q -learning with target network diverges for another MDP example constructed in Chen et al. (2019). In Section 4.4, we show that by further adding truncation, the resulting algorithm (i.e., Algorithm 1) is provably stable and achieves the optimal ˜ O ( glyph[epsilon1] -2 ) sample complexity. The reason that truncation successfully stabilizes Q -learning is due to an insightful observation regarding the relation between truncation and projection.

4.1. Classical Semi-Gradient Q -Learning. We begin with the classical semi-gradient Q -learning with linear function approximation algorithm (Bertsekas and Tsitsiklis, 1996; Sutton and Barto, 2018). With a trajectory of samples { ( S k , A k ) } collected under the behavior policy π b and an arbitrary initialization θ 0 , the semi-gradient Q -learning algorithm updates the parameter θ k according to the following formula:

<!-- formula-not-decoded -->

The reason that update (2) is called semi-gradient Q -learning is that it can be interpreted as a one step stochastic semi-gradient descent for minimizing the Bellman error. See Bertsekas and Tsitsiklis (1996) for more details. Unfortunately, Algorithm (2) does not necessarily converge, as evidenced by the divergent example provided in Baird (1995). The MDP example contructed in Baird (1995) has 7 states and 2 actions. To perform linear function approximation, 14 linearly independent basis vectors are chosen. See Appendix A for more details about this MDP. The important thing to notice about this example is that the number of basis vectors is equal to the size of the state-action space, i.e., d = |S||A| . Hence rather than doing function approximation, we are essentially doing a change of basis. Surprisingly even in this setting, Algorithm (2) diverges. Due to the divergence nature, Melo, Meyn and Ribeiro (2008); Chen et al. (2019); Lee and He (2020) impose strong negative drift assumptions to ensure the stability of Algorithm (2).

By viewing Algorithm (2) as a stochastic approximation algorithm, the target equation Algorithm (2) is trying to solve is E S k ∼ µ,A k ∼ π b ( ·| S k ) [ φ ( S k , A k )( R ( S k , A k ) + γ max a ′ ∈A φ ( S k +1 , a ′ ) glyph[latticetop] θ -φ ( S k , A k ) glyph[latticetop] θ )] = 0 . The previous equation can be written compactly using the Bellman optimality operator H ( · ) and the diagonal matrix D as

<!-- formula-not-decoded -->

and is further equivalent to the fixed-point equation

<!-- formula-not-decoded -->

where the operator H Φ : R d ↦→ R d is defined by H Φ ( θ ) = (Φ glyph[latticetop] D Φ) -1 Φ glyph[latticetop] D H (Φ θ ) . Eq. (4) is closely related to the so-called projected Bellman equation. To see this, since Φ is assumed to have linearly independent columns, Eq. (4) is equivalent to

<!-- formula-not-decoded -->

where Proj W denotes the projection operator onto the linear subspace W (which is spanned by the columns of Φ ) with respect to the weighted glyph[lscript] 2 -norm ‖ · ‖ D .

We next show that in the complete basis setting, i.e., d = |S||A| , which covers the Baird's counterexample as a special case, the operator H Φ ( · ) is in fact a contraction mapping with θ ∗ =Φ -1 Q ∗ being its unique fixed-point. This implies that the design of the classical semi-gradient Q -learning algorithm (2) is flawed because if it were designed as a stochastic approximation algorithm which is in effect performing fixed-point iteration to solve Eq. (4), it would converge . Instead, it was designed as a stochastic approximation algorithm based on Eq. (3). While Eq. (3) is equivalent to Eq. (4), their corresponding stochastic approximation algorithms have different behavior in terms of their convergence or divergence.

To show the contraction property of H Φ ( · ) , first observe that in the complete basis setting we have H Φ ( θ ) = (Φ glyph[latticetop] D Φ) -1 Φ glyph[latticetop] D H (Φ θ ) = Φ -1 H (Φ θ ) . Let ‖ · ‖ Φ , ∞ be a norm on R d defined by ‖ θ ‖ Φ , ∞ = ‖ Φ θ ‖ ∞ (the fact that it is indeed a norm can be easily verified). Then we have

<!-- formula-not-decoded -->

for all θ 1 , θ 2 ∈ R d , where the inequality follows from the Bellman optimality operator H ( · ) being an glyph[lscript] ∞ -norm contraction mapping. It follows that the operator H Φ ( · ) is a contraction mapping with respect to ‖ · ‖ Φ , ∞ . Moreover, since H Φ ( θ ∗ ) = Φ -1 H (Φ θ ∗ ) = Φ -1 H ( Q ∗ ) = Φ -1 Q ∗ = θ ∗ , the point θ ∗ is the unique fixed-point of the operator H Φ ( · ) . The previous analysis suggests that we should aim at designing Q -learning with linear function approximation algorithm as a fixed-point iteration (implemented in a stochastic manner due to sampling in RL) to solve Eq. (4). The resulting algorithm would at least converge for the Baird's MDP example.

4.2. Introducing Target Network. We begin with the following fixed-point iteration for solving the fixed-point equation (4):

<!-- formula-not-decoded -->

where we write H Φ ( · ) explicitly in terms of Φ , D , and H ( · ) . Update (6) is what we would like to perform if we had complete information about the dynamics of the underlying MDP. The question is that if there is a stochastic variant of such fixed-point iteration that can be actually implemented in the RL setting where the transition probabilities and the stationary distribution are unknown. The answer is Q -learning with target network.

We next elaborate on why Algorithm 2 can be viewed as a stochastic variant of the fixed-point iteration (6). Consider the update equation (line 5) in the inner-loop of Algorithm 2. Since the target network is fixed in the inner-loop, the update equation in terms of θ t,k is in fact a linear stochastic approximation algorithm for solving the following linear system of equations:

<!-- formula-not-decoded -->

## Algorithm 2 Q -Learning with Linear Function Approximation: Target Network and No Truncation

```
1: Input: Integers T , K , initializations θ t, 0 = 0 for all t =0 , 1 , ..., T -1 and ˆ θ 0 = 0 , behavior policy π b 2: for t =0 , 1 , · · · , T -1 do 3: for k =0 , 1 , · · · , K -1 do 4: Sample A k ∼ π b ( ·| S k ) , S k +1 ∼ P A k ( S k , · ) 5: θ t,k +1 = θ t,k + α k φ ( S k , A k )( R ( S k , A k ) + γ max a ′ ∈A φ ( S k +1 , a ′ ) glyph[latticetop] ˆ θ t -φ ( S k , A k ) glyph[latticetop] θ t,k ) 6: end for 7: ˆ θ t +1 = θ t,K 8: S 0 = S K 9: end for 10: Output: ˆ θ T
```

Since the matrix -Φ glyph[latticetop] D Φ is negative definite, the asymptotic convergence of the inner-loop update follows from standard results in the literature (Bertsekas and Tsitsiklis, 1996). Therefore, when the stepsize sequence { α k } is appropriately chosen and K is large, we expect θ t,K to approximate the solution of Eq. (7), i.e., θ t,K ≈ (Φ glyph[latticetop] D Φ) -1 Φ glyph[latticetop] D H (Φ ˆ θ t ) . Now in view of line 7 of Algorithm 2, the target network ˆ θ t +1 is synchronized to θ t,K . Therefore Q -learning with target network is in effect performing a stochastic variant of the fixed-point iteration (6).

Note that on an aside, Q -learning with target network can be viewed as an online version of fitted Q -iteration. To see this, recall that in the linear function approximation setting, fitted Q -iteration updates the corresponding parameter { ˜ θ t } iteratively according to

<!-- formula-not-decoded -->

where N = { ( s, a, s ′ ) } is a batch dataset generated in an i.i.d. manner as follows: s ∼ µ ( · ) , a ∼ π b ( ·| s ) , and s ′ ∼ P a ( s, · ) . Observe that Eq. (8) is an empirical version of

<!-- formula-not-decoded -->

In light of Eq. (9), the inner-loop of Algorithm 2 can be viewed as a stochastic gradient descent algorithm for solving the optimization problem in Eq. (9) with a single trajectory of Markovian samples.

Revisiting Baird's counter-example (where d = |S||A| ), recall that the fixed-point iteration (6) reduces to θ k +1 =Φ -1 H (Φ θ k ) = H Φ ( θ k ) . Since the operator H Φ ( · ) is a contraction mapping as shown in Section 4.1, the fixed-point iteration (6) provably converges. As a result, Q -learning with target network as a stochastic variant of the fixed-point iteration (6) also converges.

PROPOSITION 4.1. Consider Algorithm 2. Suppose that Assumption 3.1 is satisfied, the feature matrix Φ is a square matrix (i.e., d = |S||A| ), α k ≡ α ≤ λ min (1 -γ ) 2 130 , and K ≥ t α +1 . Then the sample complexity to achieve E [ ‖ Φ ˆ θ T -Q ∗ ‖ ∞ ] &lt;glyph[epsilon1] is ˜ O ( glyph[epsilon1] -2 ) .

To further verify the stability, we conduct numerical simulations for the MDP example constructed in Baird (1995). As we see, while classical semi-gradient Q -learning with linear function approximation diverges in Figure 1 (which agrees with Baird (1995)), Q -learning with target network converges as shown in Figure 2.

4.3. Insufficiency of Target Network. The reason that Q -learning with target network overcomes the divergence for Baird's MDP example is essentially that the projected Bellman operator reduces to the regular Bellman operator (which is a contraction mapping) when we have a complete basis . However, this is not the case in general. In the projected Bellman equation (5), the Bellman operator H ( · ) is a contraction mapping with respect to the glyph[lscript] ∞ -norm ‖·‖ ∞ , and the projection operator Proj W is a non-expansive mapping

60

400

300

₴ 40

200

220

g E|Ok

100

0

0

0

0

—Classical QLFA

—Classical QLFA

1

2

2

iterations (k) × 104

10

100

80

60

<!-- image -->

4

iterations (k) × 10t

FIG 1 . Classical Semi-Gradient Q -Learning

FIG 2 . Q -Learning with Target Network

with respect to the projection norm, in this case the weighted glyph[lscript] 2 -norm ‖·‖ D . Due to the norm mismatch, the composed operator Proj W H ( · ) in general is not a contraction mapping with respect to any norm. This is the fundamental reason for the divergence of Q -learning with linear function approximation, and introducing target network alone does not overcome this issue, as evidenced by the following MDP example.

EXAMPLE 4.1 (MDP Example in Chen et al. (2019)). Consider an MDP with state-space S = { s 1 , s 2 } and action-space A = { a 1 , a 2 } . Regardless of the present state, taking action a 1 results in state s 1 with probability 1 , and taking action a 2 results in state s 2 with probability 1 . The reward function is defined as R ( s 1 , a 1 ) = 1 , R ( s 1 , a 2 ) = R ( s 2 , a 1 ) = 2 , and R ( s 2 , a 2 ) = 4 . We construct the approximating linear subspace with a single basis vector: Φ=[ φ ( s 1 , a 1 ) , φ ( s 1 , a 2 ) , φ ( s 2 , a 1 ) , φ ( s 2 , a 2 )] glyph[latticetop] =[1 , 2 , 2 , 4] glyph[latticetop] . The behavior policy is to take each action with equal probability. In this example, after straightforward calculation, we have the following result.

<!-- formula-not-decoded -->

When the discount factor γ is in the interval (5 / 6 , 1) , for any positive initialization θ 0 &gt; 0 , it is clear that performing fixed-point iteration to solve Eq. (4) in this example leads to divergence. Since Q -learning with target network is a stochastic variant of such fixed-point iteration, it also diverges. Numerical simulations demonstrate that performing either classical semi-gradient Q -learning (cf. Figure 3) or Q -learning with target network (cf. Figure 4) leads to divergence for the MDP in Example 4.1.

FIG 3 . Classical Semi-Gradient Q -Learning

<!-- image -->

—QLFA with Target Network

FIG 4 . Q -Learning with Target Network

- 4.4. Truncation to the Rescue. The key ingredient we used to further overcome the divergence of Q -learning with target network is truncation. Recall from the previous section that Q -learning with target network is trying to perform a stochastic variant of the fixed-point iteration (6), which can be equivalently written as

<!-- formula-not-decoded -->

where we use ˜ Q t to denote the Q -function estimate associated with the target network ˆ θ t , i.e., ˜ Q t =Φˆ θ t . To motivate the truncation technique, we next analyze the update (10), whose behavior in terms of stability aligns with the behavior of Q -learning with target network, as explained in the previous section. First note that Eq. (10) is equivalent to

<!-- formula-not-decoded -->

A simple calculation using triangle inequality, the contraction property of H ( · ) , and telescoping yields the following error bound of the iterative algorithm (10):

<!-- formula-not-decoded -->

The problem with the previous analysis is that the term A i (which captures the error due to using linear function approximation) is not necessarily bounded unless using a complete basis or knowing in prior that { ˜ Q t } is always contained in a bounded set. The possibility that such function approximation error can be unbounded is an alternative explanation to the divergence of Q -learning with linear function approximation. This is true for arbitrary function approximation (including neural network) as well since it is in general not possible to uniformly approximate unbounded functions.

Suppose we are able to somehow control the size of the estimate ˜ Q t so that it is always contained in a bounded set. Then the term A i is guaranteed to be finite, and well captures the approximation power of the chosen function class. To achieve the boundedness of the associated Q -function estimate ˜ Q t of the target network, tracing back to Algorithm 2, a natural approach is to first project Φ ˆ θ t onto the glyph[lscript] ∞ -norm ball B r := { Q ∈ R |S||A| | ‖ Q ‖ ∞ ≤ r } before using it as the target Q -function in the inner-loop, resulting in Algorithm 3 presented in the following.

```
Algorithm 3 Impractical Q -Learning with Linear Function Approximation: Target Network and Projection 1: Input: Integers T , K , initializations θ t, 0 = 0 for all t =0 , 1 , ..., T -1 and ˆ θ 0 = 0 , behavior policy π b 2: for t =0 , 1 , · · · , T -1 do 3: for k =0 , 1 , · · · , K -1 do 4: Sample A k ∼ π b ( ·| S k ) , S k +1 ∼ P A k ( S k , · ) 5: θ t,k +1 = θ t,k + α k φ ( S k , A k )( R ( S k , A k ) + γ max a ′ ∈A ˜ Q t ( S k +1 , a ′ ) -φ ( S k , A k ) glyph[latticetop] θ t,k ) 6: end for 7: ˆ θ t +1 = θ t,K 8: ˜ Q t +1 =Π B r Φ ˆ θ t +1 9: S 0 = S K 10: end for 11: Output: ˆ θ T
```

In line 8 of Algorithm 3, the operator Π B r stands for the projection onto the glyph[lscript] ∞ -norm ball B r with respect to some suitable norm ‖ · ‖ . The specific norm ‖ · ‖ chosen to perform the projection turns out to be irrelevant as result of a key observation between truncation and projection.

Algorithm 3 although stabilizes the Q -function estimate ˜ Q t , it is not implementable in practice. To see this, recall that the whole point of using linear function approximation is to avoid working with |S||A| dimensional objects. However, to implement Algorithm 3 line 8, one has to first compute Φ ˆ θ t +1 ∈ R |S||A| ,

and then project it onto B r . Therefore, the last difficulty we need to overcome is to find a way to implement Algorithm 3 without working with |S||A| dimensional objects. The solution relies on the following observation.

LEMMA 4.2. For any x ∈ R |S||A| and any weighted glyph[lscript] p -norm ‖ · ‖ (the weights can be arbitrary and p ∈ [1 , ∞ ] ), we have glyph[ceilingleft] x glyph[ceilingright]∈ argmin y ∈ B r ‖ x -y ‖ .

REMARK. Note that argmin y ∈ B r ‖ x -y ‖ is in general a set because the projection may not be unique. As an example, observe that any point in the set { ( x, 1) | x ∈ [ -1 , 1] } is a projection of the point (0 , 2) onto the glyph[lscript] ∞ -norm unit ball { ( x,y ) | x,y ∈ [ -1 , 1] } with respect to the glyph[lscript] ∞ -norm.

Lemma 4.2 states that for any x ∈ R |S||A| , if we simply truncate x at r , the resulting vector must belong to the projection set of x onto the glyph[lscript] ∞ -norm ball with radius r , for a wide class of projection norms. This seemingly simple but important result enables us to replace projection Π B r ( · ) by truncation glyph[ceilingleft]·glyph[ceilingright] in line 8 of Algorithm 3:

<!-- formula-not-decoded -->

Unlike projection, truncation is a component-wise operation. Hence ˜ Q t +1 = glyph[ceilingleft] Φ ˆ θ t +1 glyph[ceilingright] is equivalent to ˜ Q t +1 ( s, a ) = glyph[ceilingleft] φ ( s, a ) glyph[latticetop] ˆ θ t +1 glyph[ceilingright] for all ( s, a ) .

The last issue is that we need to perform truncation for all state-action pairs ( s, a ) , which as illustrated earlier, violates the purpose of doing function approximation. However, observe that the target network is used in line 5 of Algorithm 3, where only the components of ˜ Q t visited by the sample trajectory is needed to perform the update. In light of this observation, instead of truncating φ ( s, a ) glyph[latticetop] ˆ θ t for all ( s, a ) , we only need to truncate φ ( S k +1 , a ′ ) glyph[latticetop] ˆ θ t in Algorithm 3 line 5, which leads to our stable version of Q -learning with linear function approximation in Algorithm 1. The following proposition shows that target network and truncation together stabilized Q -learning with linear function approximation, and serves as a middle step to prove Theorem 3.1.

PROPOSITION 4.2. The following inequality holds:

<!-- formula-not-decoded -->

Because of truncation, the error due to using function approximation is bounded, and is captured by E approx. This is crucial to prevent the divergence of Q -learning with linear function approximation. The last term in Eq. (11) captures the error in the inner-loop of Algorithm 1, and eventually contribute to the terms E 2 and E 3 in Eq. (1).

Revisiting Example 4.1, where either semi-gradient Q -learning or Q -learning with target network diverges, Algorithm 1 converges as demonstrated in Figure 5. Moreover, observe that Algorithm 1 seems to converge to a positive scalar, which we denote by θ ∗ . As a result, the policy π induced greedily from Φ θ ∗ is to always take action a 2 . It can be easily verified that π is indeed the optimal policy. This is an interesting observation since the optimal Q -function Q ∗ in this case does not belong to the linear sub-space W (which is spanned by a single basis vector (1 , 2 , 2 , 4) glyph[latticetop] ). Nevertheless performing Algorithm 1 converges and the induced policy is optimal. Figure 6 shows that Algorithm 1 also converges for the Baird's MDP example.

8

6

4

2

0

0

0.5

-Proposed Algorithm

1

iterations (k) × 104

1.5

- Proposed Algorithm

5

iterations (k) × 104

FIG 5 . Algorithm 1 for Baird's MDP Example

<!-- image -->

5. Conclusion and Future Work. This work makes contributions towards the understanding of Q -learning with function approximation. In particular, we show that by adding target network and truncation, the resulting Q -learning with linear function approximation is provably stable, and achieves the optimal ˜ O ( glyph[epsilon1] -2 ) sample complexity up to a function approximation error. Furthermore, the establishment of our results do not require strong assumptions (e.g. linear MDP, strong negative drift assumption, sufficiently small discount factor γ , etc.) as in related literature. There are two immediate future directions in this line of work. One is to improve the function approximation error, and the second is to extend the results of this work to using neural network approximation, i.e., the Deep Q -Network. The detailed plan is provided in Appendix D.

Acknowledgement. We thank Prof. Csaba Szepesvari from University of Alberta for the insightful comments and suggestions about this work.

## REFERENCES

- AGARWAL, A., KAKADE, S. M., LEE, J. D. and MAHAJAN, G. (2021). On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research 22 1-76.
- BAIRD, L. (1995). Residual algorithms: Reinforcement learning with function approximation. In Machine Learning Proceedings 1995 30-37. Elsevier.
- BECK, C. L. and SRIKANT, R. (2012). Error bounds for constant step-size Q -learning. Systems &amp; control letters 61 1203-1208.
- BECK, C. L. and SRIKANT, R. (2013). Improved upper bounds on the expected error in constant step-size Q -learning. In 2013 American Control Conference 1926-1931. IEEE.
- BERTSEKAS, D. P. and TSITSIKLIS, J. N. (1996). Neuro-dynamic programming . Athena Scientific.
- BORKAR, V. S. (2009). Stochastic approximation: a dynamical systems viewpoint 48 . Springer.
- BORKAR, V. S. (2021). A concentration bound for contractive stochastic approximation. Systems &amp; Control Letters 153 104947.
- BORKAR, V. S. and MEYN, S. P. (2000). The ODE method for convergence of stochastic approximation and reinforcement learning. SIAM Journal on Control and Optimization 38 447-469.
- CAI, Q., YANG, Z., LEE, J. D. and WANG, Z. (2019). Neural temporal-difference learning converges to global optima. In Proceedings of the 33rd International Conference on Neural Information Processing Systems 11315-11326.
- CARVALHO, D., MELO, F. S. and SANTOS, P. (2020). A new convergent variant of Q -learning with linear function approximation. Advances in Neural Information Processing Systems 33 .
- CHANDAK, S. and BORKAR, V. S. (2021). Concentration of Contractive Stochastic Approximation and Reinforcement Learning. Preprint arXiv:2106.14308 .
- CHEN, Z., ZHANG, S., DOAN, T. T., CLARKE, J.-P. and MAGULURI, S. T. (2019). Finite-Sample Analysis of Nonlinear Stochastic Approximation with Applications in Reinforcement Learning. Preprint arXiv:1905.11425 .
- CHEN, Z., MAGULURI, S. T., SHAKKOTTAI, S. and SHANMUGAM, K. (2020). Finite-Sample Analysis of Contractive Stochastic Approximation Using Smooth Convex Envelopes. Advances in Neural Information Processing Systems 33 .
- CHEN, Z., MAGULURI, S. T., SHAKKOTTAI, S. and SHANMUGAM, K. (2021). A Lyapunov Theory for Finite-Sample Guarantees of Asynchronous Q -Learning and TD-Learning Variants. Preprint arXiv:2102.01567 .
- DEVRAJ, A. M. and MEYN, S. (2017). Zap Q -learning. In Advances in Neural Information Processing Systems 2235-2244.

2

10

- DU, S. S., LUO, Y., WANG, R. and ZHANG, H. (2019). Provably efficient Q -learning with function approximation via distribution shift error checking oracle. In Proceedings of the 33rd International Conference on Neural Information Processing Systems 8060-8070.
- DU, S. S., LEE, J. D., MAHAJAN, G. and WANG, R. (2020). Agnostic Q -learning with Function Approximation in Deterministic Systems: Near-Optimal Bounds on Approximation Error and Sample Complexity. Advances in Neural Information Processing Systems 2020 .
- ERNST, D., GEURTS, P. and WEHENKEL, L. (2005). Tree-based batch mode reinforcement learning. Journal of Machine Learning Research 6 503-556.
- EVEN-DAR, E. and MANSOUR, Y. (2003). Learning rates for Q -learning. Journal of Machine Learning Research 5 1-25.
- FAN, J., WANG, Z., XIE, Y. and YANG, Z. (2020). A theoretical analysis of deep Q -learning. In Learning for Dynamics and Control 486-489. PMLR.
- GAO, Z., MA, Q., BA¸ SAR, T. and BIRGE, J. R. (2021). Finite-Sample Analysis of Decentralized Q -Learning for Stochastic Games. Preprint arXiv:2112.07859 .
- GYÖRFI, L., KOHLER, M., KRZYZAK, A., WALK, H. et al. (2002). A distribution-free theory of nonparametric regression 1 . Springer.
- HASSELT, H. (2010). Double Q -learning. Advances in neural information processing systems 23 2613-2621.
- HE, J., ZHOU, D. and GU, Q. (2021). Uniform-PAC Bounds for Reinforcement Learning with Linear Function Approximation. Advances in Neural Information Processing Systems, 34, 2021 .
- JAAKKOLA, T., JORDAN, M. I. and SINGH, S. P. (1994). Convergence of stochastic iterative dynamic programming algorithms. In Advances in neural information processing systems 703-710.
- JIN, C., ALLEN-ZHU, Z., BUBECK, S. and JORDAN, M. I. (2018). Is Q -learning provably efficient? In Proceedings of the 32nd International Conference on Neural Information Processing Systems 4868-4878.
- JIN, C., YANG, Z., WANG, Z. and JORDAN, M. I. (2020). Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory 2137-2143. PMLR.
- KHODADADIAN, S., CHEN, Z. and MAGULURI, S. T. (2021). Finite-Sample Analysis of Off-Policy Natural Actor-Critic Algorithm. In Proceedings of the 38th International Conference on Machine Learning . Proceedings of Machine Learning Research 139 5420-5431. PMLR.
- LEE, D. and HE, N. (2020). A unified switching system perspective and convergence analysis of Q -learning algorithms. In 34th Conference on Neural Information Processing Systems, NeurIPS 2020 . Conference on Neural Information Processing Systems.
- LEVIN, D. A. and PERES, Y. (2017). Markov chains and mixing times 107 . American Mathematical Soc.
- LI, G., WEI, Y., CHI, Y., GU, Y. and CHEN, Y. (2020). Sample Complexity of Asynchronous Q -Learning: Sharper Analysis and Variance Reduction. In Advances in Neural Information Processing Systems 33 7031-7043. Curran Associates, Inc.
- LI, G., CAI, C., CHEN, Y., GU, Y., WEI, Y. and CHI, Y. (2021a). Tightening the dependence on horizon in the sample complexity of Q -learning. In International Conference on Machine Learning 6296-6306. PMLR.
- LI, G., SHI, L., CHEN, Y., GU, Y. and CHI, Y. (2021b). Breaking the sample complexity barrier to regret-optimal model-free reinforcement learning. Advances in Neural Information Processing Systems 34 .
- LI, G., CHEN, Y., CHI, Y., GU, Y. and WEI, Y. (2021c). Sample-Efficient Reinforcement Learning Is Feasible for Linearly Realizable MDPs with Limited Revisiting. Advances in Neural Information Processing Systems 34 .
- MA, S., CHEN, Z., ZHOU, Y. and ZOU, S. (2021). Greedy-GQ with Variance Reduction: Finite-time Analysis and Improved Complexity. In International Conference on Learning Representations .
- MAEI, H. R., SZEPESVÁRI, C., BHATNAGAR, S. and SUTTON, R. S. (2010). Toward off-policy learning control with function approximation. In Proceedings of the 27th International Conference on Machine Learning (ICML-10) 719-726.
- MELO, F. S., MEYN, S. P. and RIBEIRO, M. I. (2008). An analysis of reinforcement learning with function approximation. In Proceedings of the 25th international conference on Machine learning 664-671.
- MNIH, V., KAVUKCUOGLU, K., SILVER, D., RUSU, A. A., VENESS, J., BELLEMARE, M. G., GRAVES, A., RIEDMILLER, M., FIDJELAND, A. K., OSTROVSKI, G. et al. (2015). Human-level control through deep reinforcement learning. nature 518 529533.
- MUNOS, R. and SZEPESVÁRI, C. (2008). Finite-Time Bounds for Fitted Value Iteration. Journal of Machine Learning Research 9 .
- QU, G. and WIERMAN, A. (2020). Finite-Time Analysis of Asynchronous Stochastic Approximation and Q -Learning. In Conference on Learning Theory 3185-3205. PMLR.
- ROBERTS, D. A., YAIDA, S. and HANIN, B. (2021). The Principles of Deep Learning Theory. Preprint arXiv:2106.10165 .
- SRIKANT, R. and YING, L. (2019). Finite-Time Error Bounds For Linear Stochastic Approximation and TD Learning. In Conference on Learning Theory 2803-2830.
- SUTTON, R. S. (1999). Open Theoretical Questions in Reinforcement Learning. In European Conference on Computational Learning Theory 11-17. Springer.
- SUTTON, R. S. (2015). Introduction to reinforcement learning with function approximation. In Tutorial at the Conference on Neural Information Processing Systems 33.
- SUTTON, R. S. and BARTO, A. G. (2018). Reinforcement learning: An introduction . MIT press.

00 + 201

- 010

1

1011

5

- SZEPESVÁRI, C. (1998). The asymptotic convergence-rate of Q -learning. In Advances in Neural Information Processing Systems 1064-1070.
- SZEPESVÁRI, C. and MUNOS, R. (2005). Finite time bounds for sampling based fitted value iteration. In Proceedings of the 22nd international conference on Machine learning 880-887.
- TSITSIKLIS, J. N. (1994). Asynchronous stochastic approximation and Q -learning. Machine learning 16 185-202.
- TSITSIKLIS, J. N. and VAN ROY, B. (1997). An analysis of temporal-difference learning with function approximation. IEEE transactions on automatic control 42 674-690.
- WAINWRIGHT, M. J. (2019a). Stochastic approximation with cone-contractive operators: Sharp glyph[lscript] ∞ -bounds for Q -learning. Preprint arXiv:1905.06265 .
- WAINWRIGHT, M. J. (2019b). Variance-reduced Q -learning is minimax optimal. Preprint arXiv:1906.04697 .
- WANG, Y. and ZOU, S. (2020). Finite-sample Analysis of Greedy-GQ with Linear Function Approximation under Markovian Noise. In Conference on Uncertainty in Artificial Intelligence 11-20. PMLR.
- WATKINS, C. J. and DAYAN, P. (1992). Q -learning. Machine learning 8 279-292.
- XIE, T. and JIANG, N. (2020). Batch value-function approximation with only realizability. Preprint arXiv:2008.04990 .
- XU, P. and GU, Q. (2020). A finite-time analysis of Q -learning with neural network function approximation. In International Conference on Machine Learning 10555-10565. PMLR.
- XU, T. and LIANG, Y. (2021). Sample complexity bounds for two timescale value-based reinforcement learning algorithms. In International Conference on Artificial Intelligence and Statistics 811-819. PMLR.
- YANG, L. and WANG, M. (2019). Sample-optimal parametric Q -learning using linearly additive features. In International Conference on Machine Learning 6995-7004. PMLR.
- YANG, L. and WANG, M. (2020). Reinforcement learning in feature space: Matrix bandit, kernels, and regret bound. In International Conference on Machine Learning 10746-10756. PMLR.
- ZHANG, S., YAO, H. and WHITESON, S. (2021). Breaking the Deadly Triad with a Target Network. In Proceedings of the 38th International Conference on Machine Learning . Proceedings of Machine Learning Research 139 12621-12631. PMLR.
- ZHOU, D., HE, J. and GU, Q. (2021). Provably efficient reinforcement learning for discounted MDPs with feature mapping. In International Conference on Machine Learning 12793-12802. PMLR.
- ZOU, S., XU, T. and LIANG, Y. (2019). Finite-sample analysis for SARSA with linear function approximation. In Advances in Neural Information Processing Systems 8668-8678.

## APPENDIX A: DIVERGENT MDP EXAMPLE IN Baird (1995)

The MDP instance constructed in Baird (1995) is presented in Figure 7. As we see, the state-space is S = { 1 , 2 , ..., 7 } and action-space is A = { solid , dash } . Regardless of the present state, the dash action takes the agent to one of the states 1 , 2 , ..., 6 , each with equal probability, while the solid action takes the agent to state 7 with probability 1 . The reward is identically equal to zero for all transitions, and the behavior policy π b is to take each action (solid or dash) with equal probability.

FIG 7 . Baird's counter-example (Baird, 1995)

<!-- image -->

The 14 basis vectors used for linear approximation are also presented in Figure 7. For example, the Q -function at state 1 taking solid action is approximated by θ 0 +2 θ 1 . One can easily check that the basis vectors are linearly independent. Hence this is essentially a change of basis. Performing classical semigradient Q -learning with linear function approximation leads to divergence in this example, as demonstrated in Baird (1995), and also in Figure 1 of this work.

1012

## APPENDIX B: PROOF OF THEOREM 3.1

B.1. Analysis of the Outer-Loop (Proof of Proposition 4.2). Recall that we denote ˆ Q t = glyph[ceilingleft] Φ ˆ θ t glyph[ceilingright] . Using the fact that Q ∗ = H ( Q ∗ ) , we have for any t ≥ 0 that

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

where the last line follows from H ( · ) being a γ -contraction mapping with respect to ‖·‖ ∞ , and the definition of E approx .

Repeatedly using the previous inequality and then taking expectation on both sides of the resulting inequality, and we have for any T ≥ 0 :

<!-- formula-not-decoded -->

This proves Proposition 4.2. The remaining task is to control E [ ‖ ˆ Q i +1 -glyph[ceilingleft] Proj W H ( ˆ Q i ) glyph[ceilingright]‖ ∞ ] for any i = 0 , ..., T -1 . First of all, since ˆ Q t = glyph[ceilingleft] Φ ˆ θ t glyph[ceilingright] = glyph[ceilingleft] Φ θ t -1 ,K glyph[ceilingright] and ‖glyph[ceilingleft] Q 1 glyph[ceilingright] -glyph[ceilingleft] Q 2 glyph[ceilingright]‖ ∞ ≤ ‖ Q 1 -Q 2 ‖ ∞ for any Q 1 , Q 2 ∈ R |S||A| , we have

<!-- formula-not-decoded -->

To further bound the RHS of the previous inequality, we need to analyze the inner-loop of Algorithm 1, which is done in the next section.

## B.2. Analysis of the Inner-Loop. We begin by presenting the inner-loop of Algorithm 1.

## Algorithm 4 Inner-Loop of Algorithm 1

- 1: Input: Integer K , initialization θ 0 = 0 , target network ˆ θ , behavior policy π b
- 2: for k =0 , 1 , · · · , K -1 do
- 3: Sample A k ∼ π b ( ·| S k ) , S k +1 ∼ P A k ( S k , · )
- 4: θ k +1 = θ k + α k φ ( S k , A k )( R ( S k , A k ) + γ max a ′ ∈A glyph[ceilingleft] φ ( S k +1 , a ′ ) glyph[latticetop] ˆ θ glyph[ceilingright]φ ( S k , A k ) glyph[latticetop] θ k )
- 5: end for
- 6: Output: θ K

In view of the main update equation, Algorithm 4 is a Markovian linear stochastic approximation algorithm for solving the following linear system of equations:

<!-- formula-not-decoded -->

Since the matrix -Φ glyph[latticetop] D Φ is negative definite, the finite-sample guarantees follow from standard results in the literature (Srikant and Ying, 2019; Chen et al., 2019). Specifically, we will apply Chen et al. (2019) Corollary 2.1 to establish the result. To make this paper self-contained, we first present Corollary 2.1 of Chen et al. (2019) in the following.

THEOREM B.1 (Corollary 2.1 of Chen et al. (2019)). Consider the Markovian stochastic approximation algorithm with an arbitrary initialization x 0 ∈ R d :

<!-- formula-not-decoded -->

Suppose that

- (1) The finite-state Markov chain { Y k } has a unique stationary distribution ν , and it holds for any k ≥ 0 that max y ∈Y ‖ P k ( y, · ) -ν ( · ) ‖ TV ≤ Cρ k for some constant C &gt; 0 and ρ ∈ (0 , 1) .
- (2) There exist constants L 1 , L 2 &gt; 0 such that the operator F ( · , · ) satisfies ‖ F ( x 1 , y ) -F ( x 2 , y ) ‖ 2 ≤ L 1 ‖ x 1 -x 2 ‖ 2 and ‖ F ( 0 , y ) ‖ 2 ≤ L 2 for any x 1 , x 2 ∈ R d and y ∈Y .
- (3) The equation ¯ F ( x ) = E Y ∼ ν [ F ( x,Y )] = 0 has a unique solution x ∗ ∈ R d , and the following inequality holds for all x ∈ R d : ( x -x ∗ ) glyph[latticetop] ¯ F ( x ) ≤-κ ‖ x -x ∗ ‖ 2 2 , where κ &gt; 0 is a positive constant.
- (4) The stepsize sequence { α k } is a constant sequence (i.e., α k ≡ α ), and α is chosen such that ατ α ≤ κ 130max( L 1 ,L 2 ) 2 , where τ α := min { k ≥ 0 : max y ∈Y ‖ P k ( y, · ) -ν ( · ) ‖ TV ≤ α } .

Then we have for any k ≥ τ α that

<!-- formula-not-decoded -->

To apply Theorem B.1, we first rewrite the update equation in line 4 of Algorithm 4 in the form of stochastic approximation algorithm (13). Then we verify that Conditions (1) - (4) are satisfied.

- Reformulation. For any k ≥ 0 , let Y k =( S k , A k , S k +1 ) , which is clearly a Markov chain with state-space given by Y = { y =( s, a, s ′ ) | s ∈ S , π b ( a | s ) &gt; 0 , P a ( s, s ′ ) &gt; 0 } . Define the function F : R d ×Y ↦→ R d by

<!-- formula-not-decoded -->

for any θ ∈ R d and y = ( s, a, s ′ ) ∈ Y . Then the update equation of Algorithm 4 can be equivalently written as

<!-- formula-not-decoded -->

- Verification of Condition (1). Under Assumption 3.1, the Markov chain { Y k } has a unique stationary distribution ν , which is given by ν ( s, a, s ′ ) = µ ( s ) π ( a | s ) P a ( s, s ′ ) for all ( s, a, s ′ ) ∈ Y . In addition, we have for any y =( s, a, s ′ ) ∈Y that

<!-- formula-not-decoded -->

- Verification of Condition (2). For any x 1 , x 2 ∈ R d and y =( s, a, s ′ ) ∈Y , we have

<!-- formula-not-decoded -->

Similarly, we have for any y =( s, a, s ′ ) ∈Y that

<!-- formula-not-decoded -->

- Verification of Condition (3). By definition of F ( · , · ) , we have

<!-- formula-not-decoded -->

where we recall that D ∈ R |S||A|×|S||A| is a diagonal matrix with diagonal entries { µ ( s ) π b ( a | s ) } ( s,a ) ∈S×A . Since Φ has linearly independent columns, the matrix Φ glyph[latticetop] D Φ is invertible. Solving ¯ F ( θ ) = 0 and we obtain θ ∗ =(Φ glyph[latticetop] D Φ) -1 Φ glyph[latticetop] D H ( glyph[ceilingleft] Φ ˆ θ glyph[ceilingright] ) . Furthermore, note that the matrix Φ glyph[latticetop] D Φ is positive definite, whose smallest eigenvalue is denoted by λ min . Therefore we have for any θ ∈ R d :

<!-- formula-not-decoded -->

- Verification of Condition (4). This is satisfied due to our choice of the constant stepsize α in Theorem 3.1.

Now that all Conditions are satisfied. Apply Theorem B.1 and we obtain for any k ≥ t α +1 :

<!-- formula-not-decoded -->

where we used θ 0 = 0 in Algorithm 4. The last step is to provide an upper bound on ‖ θ ∗ ‖ 2 . Note that

<!-- formula-not-decoded -->

Substituting the previous upper bound we obtained for ‖ θ ∗ ‖ 2 into Eq. (15) and we finally have for all k ≥ t α +1 :

<!-- formula-not-decoded -->

- B.3. Putting Together. In this section, we combine the analysis of the outer-loop and the inner-loop to establish the overall finite-sample bounds of Algorithm 1. Denote θ ∗ t =(Φ glyph[latticetop] D Φ) -1 Φ glyph[latticetop] D H ( ˆ Q t ) . Note that we have Φ θ ∗ t = Proj W H ( ˆ Q t ) . Using the fact that ‖ · ‖ ∞ ≤‖·‖ 2 and we obtain for any 0 ≤ i ≤ T :

<!-- formula-not-decoded -->

where the last line follows from √ a + b ≤ √ a + √ b for any a, b ≥ 0 .

Substituting the previous inequality into Eq. (12), and we obtain the overall finite-sample guarantees of Algorithm 1:

<!-- formula-not-decoded -->

In view of the finite-sample guarantee, to obtain E [ ‖ ˆ Q T -Q ∗ ‖ ∞ ] ≤ glyph[epsilon1] + E approx 1 -γ for a given accuracy glyph[epsilon1] , the number of sample required is of the size

<!-- formula-not-decoded -->

## APPENDIX C: PROOF OF ALL TECHNICAL RESULTS IN SECTION 4

- C.1. Proof of Proposition 4.1. The proof is identical to that of Theorem 3.1, and hence is omitted.
- C.2. Proof of Proposition 4.2. See Appendix B.1.
- C.3. Proof of Lemma 4.1. We first compute the transition probability matrix of the Markov chain { S k } under π b . Since

<!-- formula-not-decoded -->

and π ( a | s ) = 1 / 2 for any a ∈ { a 1 , a 2 } and s ∈ { s 1 , s 2 } , we have P π b = 1 2 I 2 . As a result, the unique stationary distribution µ of the Markov chain { S k } under π b is given by µ =(1 / 2 , 1 / 2) . Therefore, the matrix

D ∈ R |S||A|×|S||A| (defined before Theorem 3.1) is given by D = 1 4 I 4 . We next compute Eq. (4) in this example. First of all, by definition of the Bellman operator we have for any θ ∈ R that

<!-- formula-not-decoded -->

Similarly, we also have

<!-- formula-not-decoded -->

Therefore, Eq. (4) in the case of Example 4.1 is explicitly given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

C.4. Proof of Lemma 4.2. Let { ν ( s, a ) } ( s,a ) ∈S×A be any positive weights, and denote the weighted glyph[lscript] p -norm with weights { ν ( s, a ) } ( s,a ) ∈S×A by ‖ · ‖ ν,p . For any x ∈ R |S||A| , we have

<!-- formula-not-decoded -->

Therefore, we have glyph[ceilingleft] x glyph[ceilingright]∈ argmin y ∈ B r ‖ x -y ‖ ν,p .

## APPENDIX D: FUTURE WORK

D.1. Establishing the Asymptotic Convergence and Improving the Function Approximation Error. Although Theorem 3.1 establishes the mean-square error bound of Q -learning with linear function approximation, due to the function approximation error, the bound does not imply asymptotic convergence. In light of our discussion in Section 4, suppose Algorithm 1 indeed converges (as K,T →∞ and α → 0 ) . The corresponding Q -function estimate of the output, i.e., ˆ Q T = glyph[ceilingleft] Φ ˆ θ T glyph[ceilingright] , can only converge to the solution of the truncated projected Bellman equation :

<!-- formula-not-decoded -->

Unlike projected Bellman equation (5), which may not have a solution in general (cf. Example 4.1), since the truncated projected Bellman operator maps a compact set B r to itself, Eq. (17) must have at least one solution according to the Brouwer fixed-point theorem. However, whether the solution to Eq. (17) is unique or not is unclear. Therefore, it is also unclear if performing fixed-point iteration to solve Eq. (17), or its stochastic variant (i.e., Algorithm 1) can actually leads to asymptotic convergence. Further investigating the truncated projected Bellman equation to show asymptotic convergence is one of our immediate future directions.

Suppose we were able to show the asymptotic convergence of Algorithm 1 to the unique solution of the truncated projected Bellman equation (17), denoted by ¯ Q . Then, instead of establishing finite-sample bound of the form

<!-- formula-not-decoded -->

which is in fact what we did in this work, we would seek to establish the finite-sample bound of E [ ‖ ˆ Q T -¯ Q ‖ ∞ ] , and separately characterize the difference between Q ∗ and ¯ Q . This is in the same spirit of the seminal work Tsitsiklis and Van Roy (1997), which studies the TD-learning with linear function approximation algorithm for policy evaluation. There are two advantages of this alternative approach. One is that the sample complexity of ˆ Q T converging to ¯ Q is well-defined once we establish finite-sample convergence of E [ ‖ ˆ Q T -¯ Q ‖ ∞ ] to zero, while the sample complexity of convergence bounds of the form (18) is strictly speaking not well-defined because of the additive constant E 4 , and may lead to erroneous result, as illustrated in Khodadadian, Chen and Maguluri (2021) Appendix C. Second, this approach would enable us to reduce the function approximation error by removing the sup operator in E approx, i.e., from the current sup Q : ‖ Q ‖ ∞ ≤ r ‖glyph[ceilingleft] Proj W H ( Q ) glyph[ceilingright]-H ( Q ) ‖ ∞ to ‖glyph[ceilingleft] Proj W H ( ¯ Q ) glyph[ceilingright]-H ( ¯ Q ) ‖ ∞ .

Although the lack of asymptotic convergence is a major limitation of this work, we want to point out that such limitation is present in almost all related literature on both value-space and policy-space methods whenever function approximation is used. To our knowledge, the only exception is Tsitsiklis and Van Roy (1997) (as well as its follow-up work), where asymptotic convergence was established for TD-learning, and the limit was characterized as the unique solution of the projected Bellman equation. Other literature studying RL with function approximation either do not have asymptotic convergence Agarwal et al. (2021), or have asymptotic convergence without knowing where the limit is Maei et al. (2010).

D.2. The Deep Q Network. The ultimate goal of this line of work is to provide theoretical understanding to the celebrated Deep Q -Network. We first present the extension of our Algorithm 1 to the setting where we use arbitrary function approximation (cf. Algorithm 5). Let F = { f θ : S × A ↦→ R | θ ∈ R d } be a parametric function class (with parameter θ ). For example, F can be the set of functions representable by a certain neural network, and θ is the corresponding weight vector.

While the algorithm easily extends, the theoretical results do not. In particular, there are two major challenges.

- (1) With recent advances in deep learning Roberts, Yaida and Hanin (2021), it is possible to explicitly characterize the function approximation error E approx as a function of the hyper-parameters of the chosen neural network, such as the width, the number of layers, and the Hölder continuity parameter, etc.

```
1: Input: Integers T , K , initialization θ 0 , 0 = ˆ θ 0 = 0 2: for t =0 , 1 , · · · , T -1 do 3: for k =0 , 1 , · · · , K -1 do 4: Sample A k ∼ π b ( ·| S k ) , observe S k +1 ∼ P A k ( S k , · ) 5: θ t,k +1 = θ t,k + α k ∇ f θ t,k ( S k , A k )( R ( S k , A k ) + γ max a ′ ∈A glyph[ceilingleft] f ˆ θ t ( S k +1 , a ′ ) glyph[ceilingright]f θ t,k ( S k , A k )) 6: end for 7: ˆ θ t +1 = θ t,K 8: S 0 = S K 9: end for 10: Output: ˆ θ T
```

- (2) A more significant challenge is about the convergence of the inner-loop of Algorithm 5. Recall that in the linear function approximation setting, the inner loop (line 5 of Algorithm 1) can be viewed as a one-step Markovian stochastic approximation for solving the linear system of equations -Φ glyph[latticetop] D Φ θ + Φ glyph[latticetop] D H ( glyph[ceilingleft] Φ ˆ θ t glyph[ceilingright] ) = 0 , or a one-step Markovian stochastic gradient descent for minimizing a quadratic objective ‖ Φ θ -H ( glyph[ceilingleft] Φ ˆ θ t glyph[ceilingright] ) ‖ 2 D in terms of θ . In this case, convergence to the global optimal of the innerloop iterates is well established in the literature. Now consider using arbitrary function approximation in Algorithm 5. Although the inner-loop (line 5) is still performing a one-step Markovian stochastic gradient descent for minimizing ‖ f θ -H ( glyph[ceilingleft] f ˆ θ t glyph[ceilingright] ) ‖ 2 D in terms of θ , since the objective is now in general non-convex, the convergence to global optimal remains as a major theoretical open problem in the deep learning community.

Although the Deep Q -Network was previously studied in Fan et al. (2020), their results rely on the following two assumptions: (1) the function approximation space is closed under the Bellman operator, and (2) there exists an oracle that returns the global optimal of non-convex optimization problems. Under these two assumptions, both challenges described earlier are no longer present.

Once we explicitly characterize the function approximation error E approx, and show global convergence of the inner-loop, substituting the result into our analysis framework and we would be able to obtain finitesample guarantees of Deep Q -Network, thereby achieving the ultimate goal of this line of research.

## APPENDIX E: RELATED LITERATURE AND THEIR LIMITATIONS

To complement Section 1.1, we here present a more detailed discussion about related literature that requires strong negative drift assumptions, and that achieves stability of Q -learning with linear function approximation by implicitly changing the problem parameters.

- E.1. Strong Negative Drift Assumption. As mentioned in Section 1.1, classical Q -learning with linear function approximation (cf. Algorithm (2)) was studied in Melo, Meyn and Ribeiro (2008); Chen et al. (2019); Lee and He (2020); Xu and Gu (2020); Cai et al. (2019) and many other follow-up work under a strong negative drift assumption. While the specific assumption varies, they are all in the same spirit that the assumption should ensure that the associated ODE of Q -learning with linear function approximation is globally asymptotically stable, which essentially guarantees the stability of the algorithm Borkar (2009).

The negative drift assumptions are highly restrictive. To see this, we here present the assumption proposed in Melo, Meyn and Ribeiro (2008) as an illustrative example:

glyph[negationslash]

<!-- formula-not-decoded -->

where the factor of 2 is missing in Melo, Meyn and Ribeiro (2008). Since Condition (19) needs to hold for all θ = 0 , it is not clear if it can be satisfied even if we choose the optimal policy as the behavior policy. Intuitively, to satisfy Condition (19), the discount factor γ should be extremely small.

glyph[negationslash]

To see more explicitly the restrictiveness of Condition (19), we consider the case where d = |S||A| . A special case of this is when Φ is an identity matrix, which corresponds to the tabular setting. Since it is known that tabular Q -learning does not Condition (19) to converge (Tsitsiklis, 1994), we would expect that Condition (19) is automatically satisfied. However, the following result implies that Condition (19) remains highly restrictive even when d = |S||A| .

LEMMA E.1. When d = |S||A| , then it is not possible to satisfy Condition (19) when γ ≥ 1 √ 2 |A| , where |A| is the size of the action-space.

PROOF OF LEMMA E.1. Lemma E.1 is entirely similar to Chen et al. (2019) Proposition 3. We here present its proof to make this paper self-contained. Define glyph[negationslash]

<!-- formula-not-decoded -->

which is the orthogonal complement of the span of the feature vectors { φ ( s ′ , a ′ ) } ( s ′ ,a ′ ) =( s,a ) . Let θ ∈ Θ s,a satisfying φ ( s, a ) glyph[latticetop] θ &gt; 0 (which is always possible). Then Condition (19) implies

<!-- formula-not-decoded -->

which implies γ 2 &lt; π b ( a | s ) 2 . Since this is true for all ( s, a ) , we must have γ 2 &lt; min ( s,a ) π b ( a | s ) 2 ≤ 1 2 |A| . Therefore, when γ ≥ 1 √ 2 |A| , it is not possible to satisfy Condition (19).

In this work, we do not require any variants of the negative drift assumption to achieve the stability of Q -learning with linear function approximation. Removing such strong assumption in existing literature is a major contribution of this work.

E.2. A Variant of Q -Learning with Target Network Zhang, Yao and Whiteson (2021). A variant of the Q -learning with linear function approximation was proposed in Zhang, Yao and Whiteson (2021). To overcome the divergence issue, they introduced target network in the algorithm. However, as we have shown in Section 4.3, target network alone is not enough to stabilize Q -learning. The reason that Zhang, Yao and Whiteson (2021) achieves convergence is by implicitly modifying the problem discount factor. To see this, consider the tabular setting where φ i , 1 ≤ i ≤ d are chosen as the canonical basis vectors. Then the algorithm proposed in Zhang, Yao and Whiteson (2021) aims at solving the following modified Bellman equation (Eq. (11) in Zhang, Yao and Whiteson (2021)):

<!-- formula-not-decoded -->

where D is a diagonal matrix with the stationary distribution of the Markov chain { ( S k , A k ) } (induced by the behavior policy) on its diagonal, and η &gt; 0 is a tunable parameter introduced in Zhang, Yao and Whiteson (2021) to stabilize Q -learning.

glyph[negationslash]

Now note that as long as η =0 , Eq. (20) is not the same as the original Bellman equation Q = H ( Q ) , which implies that the algorithm in Zhang, Yao and Whiteson (2021) does not converge to Q ∗ even in the tabular setting. We next show that introducing η &gt; 0 in Eq. (20) is equivalent to artificially scaling down the discount factor γ of the problem.

For ease of exposition, suppose that we are in the ideal setting where D = 1 |S||A| I (i.e., uniform exploration). Then Eq. (20) can be equivalently written by

<!-- formula-not-decoded -->

Compared to the original Bellman equation Q = H ( Q ) , the modified Bellman equation (21) has two major modifications. First is that the reward function is scaled down by a factor of 1+ η |S||A| . This modifications glyph[negationslash]

does not change the optimal policy since the optimal policy is invariant to the scaling of the optimal Q -function. A more important modification is that the discount factor γ is scaled down by a factor of 1 + η |S||A| . This change potentially results in a different optimal policy compared to the original problem. In fact, since the tunable parameter η is first multiplied by |S||A| before it appears in the denominator of γ in Eq. (21), using postive η changes the discount factor of the problem drastically.

E.3. Coupled Q -Learning Carvalho, Melo and Santos (2020). A two time-scale variant of Q -learning with linear function approximation (called coupled Q -learning) was proposed in Carvalho, Melo and Santos (2020). It was shown in Carvalho, Melo and Santos (2020) that the limit points u ∗ and v ∗ of the coupled Q -learning algorithm satisfy the following systems of equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where D is a diagonal matrix with diagonal entries being the stationary distribution of the Markov chain { ( S k , A k ) } induced by the behavior policy π b .

Under the assumption that ‖ φ ( s, a ) ‖ 2 ≤ 1 for all ( s, a ) (Assumption 2 in Carvalho, Melo and Santos (2020)), and Φ glyph[latticetop] D Φ= σI d (Assumption 4 in Carvalho, Melo and Santos (2020)), where d is the number of basis vectors, a performance guarantee is provided regarding the distance between the Q -function estimate Q v ∗ associated with v ∗ and the optimal Q -function Q ∗ , and is presented in the following.

THEOREM E.1 (Theorem 2 in Carvalho, Melo and Santos (2020)). The limit point v ∗ satisfies

<!-- formula-not-decoded -->

where E σ = 1 -σ σ γ (1 -γ ) 2 .

Note that in Eq. (24) of Theorem E.1, in addition to the function approximation error, there is an additional error term E σ that does not vanish even in the tabular setting. Although the coupled Q -learning algorithm does not require strong assumptions to converge, we next show that in order for the performance bound (24) to be non-trivial, the discount factor γ must be sufficiently small.

Consider the error term E σ = 1 -σ σ γ 1 -γ 1 1 -γ . Since ‖ Q ∗ ‖ ∞ ≤ 1 1 -γ , in order for the performance bound of Theorem E.1 to be meaningful, we need to at least have

<!-- formula-not-decoded -->

otherwise simply choosing Q = 0 leads to a better performance guarantee. The above inequality implies γ &lt; σ . However, under the assumption that ‖ φ ( s, a ) ‖ 2 ≤ 1 for all ( s, a ) (Assumption 2 in Carvalho, Melo and Santos (2020)), and Φ glyph[latticetop] D Φ= σI d (Assumption 4 in Carvalho, Melo and Santos (2020)), we have

<!-- formula-not-decoded -->

which implies σ ≤ 1 d . As a result, Theorem E.1 provides a meaning performance guarantee on the limit point v ∗ only when γ ≤ 1 d , which is a restrictive requirement on the discount factor γ of the problem.

To see more explicitly the reason that the coupled Q -learning algorithm has an additional bias E σ , consider the tabular setting, i.e., Φ= I |S||A| . In this case Assumption 4 of Carvalho, Melo and Santos (2020) reduces to D = 1 |S||A| I |S||A| (uniform exploration). Then Eq. (22) is equivalent to

<!-- formula-not-decoded -->

As illustrated in the previous subsection, such modification of the Bellman equation is equivalent to artificially scaling down the discount factor γ of the original problem by a factor of |S||A| . In view of Eq. (23), when Φ= I |S||A| and D = 1 |S||A| I |S||A| , Q v ∗ is just a constant scaling of Q u ∗ . Hence the optimal policy induced from either Q v ∗ or Q u ∗ is the one that corresponds to the original problem with the discount factor γ being replaced by γ |S||A| . Because of this implicit modification on the problem discount factor, the coupled Q -learning algorithm although is stable, does not converge to Q ∗ even in the tabular setting.