## Confident Approximate Policy Iteration for Efficient Local Planning in /u1D45E /u1D70B -realizable MDPs

## Gellért Weisz

DeepMind, London, UK University College London, London, UK

## Tadashi Kozuno

András György

DeepMind, London, UK

## Csaba Szepesvári

University of Alberta, Edmonton, Canada Omron Sinic X, Tokyo, Japan

DeepMind, London, UK University of Alberta, Edmonton, Canada

## Abstract

We consider approximate dynamic programming in /u1D6FE -discounted Markov decision processes and apply it to approximate planning with linear value-function approximation. Our first contribution is a new variant of APPROXIMATE POLICY ITERATION (API), called CONFIDENT APPROXIMATE POLICY ITERATION (CAPI), which computes a deterministic stationary policy with an optimal error bound scaling linearly with the product of the effective horizon /u1D43B and the worstcase approximation error /u1D700 of the action-value functions of stationary policies. This improvement over API (whose error scales with /u1D43B 2 ) comes at the price of an /u1D43B -fold increase in memory cost. Unlike Scherrer and Lesner [2012], who recommended computing a non-stationary policy to achieve a similar improvement (with the same memory overhead), we are able to stick to stationary policies. This allows for our second contribution, the application of CAPI to planning with local access to a simulator and /u1D451 -dimensional linear function approximation. As such, we design a planning algorithm that applies CAPI to obtain a sequence of policies with successively refined accuracies on a dynamically evolving set of states. The algorithm outputs an ˜ O( √ /u1D451 /u1D43B /u1D700 ) -optimal policy after issuing ˜ O( /u1D451/u1D43B 4 / /u1D700 2 ) queries to the simulator, simultaneously achieving the optimal accuracy bound and the best known query complexity bound, while earlier algorithms in the literature achieve only one of them. This query complexity is shown to be tight in all parameters except /u1D43B . These improvements come at the expense of a mild (polynomial) increase in memory and computational costs of both the algorithm and its output policy.

## 1 Introduction

A key question in reinforcement learning is how to use value-function approximation to arrive at scaleable algorithms that can find near-optimal policies in Markov decision processes (MDPs). A flurry of recent results aims at solving this problem efficiently with varying models of interaction with the MDP. In this paper we focus on the problem of planning with a simulator when using linear function approximation. A simulator is a 'device' that, given a state-action pair as a query, returns a next state and reward generated from the transition kernel of the MDP that is simulated. Depending on the application, such a simulator is often readily available (e.g., in chess, go, Atari). Planning with simulator access comes with great benefits: for example, in a recent work, Wang et al. [2021] showed that under some conditions it is exponentially more efficient to find a near-optimal policy if

a simulator of the MDP (that can reset to a state) is available compared to the online case where a learner interacts with its environment by following trajectories but without the help of a simulator.

Our setting of offline, local planning considers the problem of finding a policy with near-optimal value at a given initial state /u1D460 0 in the MDP. The planner can issue queries to the simulator, and has to find and output a near-optimal policy with high probability. The efficiency of a planner is measured in four ways: the suboptimality of the policy found, that is, how far its value is from that of the optimal policy; the query cost , that is, the number of queries issued to the simulator; the computational cost , which is the number of operations used; and the memory cost , which is the amount of memory used (we adopt the real computation model for these costs). There are several interaction models between the planner and the simulator [Yin et al., 2022]. The most permissive one is called the generative model , or random access . Here, the planner receives the set of all states and is allowed to issue queries for any state and action. Coding a simulator that supports this model can be challenging, as oftentimes the set of states is computationally difficult to describe. Instead of random access , we consider the more practical and more challenging local access setting, where the planner only sees the initial state and the set of states received as a result to a query to the simulator. Consequently, the queries issued have to be for a state that has already been encountered this way (and any available action), while the simulator needs to support the ability to reset the MDP state only to previously seen states. A simple approach in practice to support this model is saving and reloading checkpoints during the operation of the simulator.

To handle large, possibly infinite state spaces, we use linear function approximation to approximate the action-value functions /u1D45E /u1D70B of stationary, deterministic policies /u1D70B (for background on MDPs, see the next section). A feature-map is a good fit to an MDP if the worst-case error of using the featuremap to approximate value functions of policies of the MDP is small:

Definition 1.1 ( /u1D45E /u1D70B -realizability: uniform policy value-function approximation error) . Given an MDP, the uniform policy value-function approximation error induced by a feature map /u1D711 , which maps state-action pairs ( /u1D460 , /u1D44E ) to the Euclidean ball of radius /u1D43F centered at zero in R /u1D451 , over a set of parameters belonging to the /u1D451 -dimensional centered Euclidean ball of radius /u1D435 is

<!-- formula-not-decoded -->

where the outermost supremum is over all possible stationary deterministic memoryless policies (i.e., maps from states to actions) of the MDP.

Our goal is to design algorithms that scale gracefully with the uniform approximation error /u1D700 at the expense of controlled computational cost. To achieve nontrivial guarantees, the uniform approximation error needs to be 'small'. This (implicit) assumption is stronger than the /u1D45E ★ -realizability assumption (where the approximation error is only considered for optimal policies), which Weisz et al. [2021] showed an exponential query complexity lower bound for. At the same time, it is (strictly) weaker than the linear MDP assumption [Zanette et al., 2020], for which there are efficient algorithms to find a near-optimal policy in the online setting (without a simulator) [Jin et al., 2020], even in the more challenging reward-free setting where the rewards are only revealed after exploration [Wagenmaker et al., 2022].

In the local access setting, the planner learns the features /u1D711 ( /u1D460 , /u1D44E ) of a state-action pair only for those states /u1D460 that have already been encountered. In contrast, in the random access setting, the whole feature map /u1D711 (· , ·) , of (possibly infinite) size /u1D451 |S||A| (where S and A are the state and action sets, resp.), is given to the planner as input. In the latter setting, when only the query cost is counted, Du et al. [2019] and Lattimore et al. [2020] proposed algorithms (the latter working in the misspecified, /u1D700 &gt; 0 regime) that issue a number of queries that is polynomial in the relevant parameters, but require a barycentric spanner or near-optimal design of the input features. In the worst case, computing any of these sets scales polynomially in |S| and |A| , which can be prohibitive.

In the case of local access , considered in this paper, the best known bound on the suboptimality of the computed policy is achieved by CONFIDENT MC-POLITEX [Yin et al., 2022]. In the more permissive random access setting, the best known query cost is achieved by Lattimore et al. [2020]. Our algorithm, CAPI-QPI-PLAN (given in Algorithm 3), achieves the best of both while only assuming local access . This is shown in the next theorem; in the theorem /u1D700 is as defined in Definition 1.1, /u1D6FE is the discount factor, and /u1D463 ★ and /u1D463 /u1D70B are the state value functions associated with the optimal policy and policy /u1D70B , respectively (precise definitions of these quantities are given in the next section). A

comparison to other algorithms in the literature is given in Table 1; there the accuracy parameter /u1D714 of the algorithms is set to /u1D700 , but a larger /u1D714 can be used to trade off suboptimality guarantees for an improved query cost.

Theorem 1.2. For any confidence parameter /u1D6FF ∈ ( 0 , 1 ] , accuracy parameter /u1D714 &gt; 0 , and initial state /u1D460 0 ∈ S , with probability at least 1 -/u1D6FF , CAPI-QPI-PLAN (Algorithm 3) finds a policy /u1D70B with while executing at most ˜ O ( /u1D451 ( 1 -/u1D6FE ) -4 /u1D714 -2 ) queries in the local access setting.

<!-- formula-not-decoded -->

CAPI-QPI-PLAN is based on CONFIDENT MC-LSPI, another algorithm of Yin et al. [2022], which relies on policy iteration from a core set of informative state-action pairs, but achieves inferior performance both in terms of suboptimality and query complexity. However, CAPI-QPI-PLAN's improvements come at the expense of increased memory and computational costs, as shown in the next theorem: compared to CONFIDENT MC-LSPI, the memory and computational costs of our algorithm increase by a factor of the effective horizon /u1D43B = ˜ O( 1 /( 1 -/u1D6FE )) , and the policy computed by CAPI-QPI-PLAN uses a /u1D451/u1D43B factor more memory for storage and a /u1D451 2 /u1D43B factor more computation to evaluate.

Theorem 1.3 (Memory and computational cost) . The memory and computational cost of running CAPI-QPI-PLAN (Algorithm 3) are ˜ O ( /u1D451 2 /( 1 -/u1D6FE ) ) and ˜ O ( /u1D451 4 |A|( 1 -/u1D6FE ) -5 /u1D714 -2 ) , respectively, while the memory and computational costs of storing and evaluating the final policy outputted by CAPI-QPI-PLAN , respectively, are ˜ O ( /u1D451 2 /( 1 -/u1D6FE ) ) and ˜ O ( /u1D451 3 |A|/( 1 -/u1D6FE ) ) .

Theorem 1.4 (Query cost lower bound) . Let /u1D6FC ∈ ( 0 , 0 . 05 /u1D6FE ( 1 -/u1D6FE ) ( 1 + /u1D6FE ) 2 ) , /u1D6FF ∈ ( 0 , 0 . 08 ] , /u1D6FE ∈ [ 7 12 , 1 ] , /u1D451 ≥ 3 , and /u1D700 ≥ 0 . Then there is a class M of MDPs with uniform policy value-function approximation error at most /u1D700 such that any planner that guarantees to find an /u1D6FC -optimal policy /u1D70B (i.e., /u1D463 ★ ( /u1D460 0 ) -/u1D463 /u1D70B ( /u1D460 0 ) ≤ /u1D6FC ) with probability at least 1 -/u1D6FF for all /u1D440 ∈ M when used with a simulator for /u1D440 with random access , the worst-case (over M ) expected number of queries issued by the planner is at least

Next we present a lower bound corresponding to Theorem 1.2 that holds even in the more permissive random access setting, and shows that CAPI-QPI-PLAN trades of the query cost and the suboptimality of the returned policy asymptotically optimally up to its dependence on 1 /( 1 -/u1D6FE ) :

<!-- formula-not-decoded -->

If /u1D714 is set to /u1D700 for CAPI-QPI-PLAN, the first term of Eq. (2) implies that any planner with an asymptotically smaller (apart from logarithmic factors) suboptimality guarantee than Eq. (1) executes exponentially many queries in expectation. The second term of Eq. (2), which is shown to be a lower bound in Theorem H.3 even in the more general setting of linear MDPs with zero misspecification ( /u1D700 = 0 ), matches the query complexity of Theorem 1.2 up to an ˜ O(( 1 -/u1D6FE ) 2 ) factor. Thus, the lower bound implies that the suboptimality and query cost bounds of Theorem 1.2 are tight up to logarithmic factors in all parameters except the 1 /( 1 -/u1D6FE ) -dependence of the query cost bound.

At the heart of our method is a new algorithm, which we call CONFIDENT APPROXIMATE POLICY ITERATION (CAPI). This algorithm, which belongs to the family of approximate dynamic programming algorithms [Bertsekas, 2012, Munos, 2003, 2005], is a novel variant of APPROXIMATE POLICY ITERATION (API) [Bertsekas and Tsitsiklis, 1996]: in the policy improvement step, CAPI only updates the policy in states where it is confident that the update will improve the performance. This simple modification allows CAPI to avoid the problem of 'classical' approximate dynamic programming algorithms (approximate policy and value iteration) of inflating the value function evaluation error by a factor of /u1D43B 2 where /u1D43B = ˜ O( 1 /( 1 -/u1D6FE )) (for discussions of this problem, see also the papers by Scherrer and Lesner, 2012 and Russo, 2020), and reduce this inflation factor to /u1D43B . A similar result has already been achieved by Scherrer and Lesner [2012], who proposed to construct a non-stationary policy that strings together all policies obtained while running either approximate value or policy iteration. However, applying this result to our planning problem is problematic, since the policies to be evaluated are non-stationary, and hence including them in the policy set we aim to approximate may drastically increase the error /u1D700 as compared to Definition 1.1, which only considers stationary memoryless policies.

Table 1: Comparison of suboptimality and query complexity guarantees of various planners (with the approximation accuracy parameter /u1D714 set to /u1D700 ). Drawbacks are highlighted with red, the best bounds with blue.

| Algorithm [Publication]                 | Query cost                                    | Suboptimality                               | Access model   |
|-----------------------------------------|-----------------------------------------------|---------------------------------------------|----------------|
| MC-LSPI [Lattimore et al., 2020]        | ˜ O ( /u1D451 /u1D700 2 ( 1 - /u1D6FE ) 4 )   | ˜ O ( /u1D700 √ /u1D451 ( 1 - /u1D6FE ) 2 ) | random access  |
| CONFIDENT MC-LSPI [Yin et al., 2022]    | ˜ O ( /u1D451 2 /u1D700 2 ( 1 - /u1D6FE ) 4 ) | ˜ O /u1D700 √ /u1D451 ( 1 - /u1D6FE ) 2 )   | local access   |
| CONFIDENT MC-POLITEX [Yin et al., 2022] | ˜ O /u1D451 /u1D700 4 ( 1 - /u1D6FE ) 5       | ( ˜ O /u1D700 √ /u1D451 1 - /u1D6FE         | local access   |
| CAPI-QPI-PLAN [This work]               | ( ) ˜ O /u1D451 /u1D700 2 ( 1 - /u1D6FE ) 4   | ( ) ˜ O /u1D700 √ /u1D451 1 - /u1D6FE       | local access   |

(

)

(

)

While the improvements provided by CAPI allows CAPI-QPI-PLAN to match the performance of CONFIDENT MC-POLITEX in terms of suboptimality, it is unlikely that a simple modification of CONFIDENT MC-POLITEX would lead to an algorithm which matches CAPI-QPI-PLAN's performance in terms of query cost (see Table 1): Both methods evaluate a sequence of policies at an ˜ O( /u1D700 ) accuracy each (requiring ˜ O( 1 / /u1D700 2 ) queries, omitting the dependence on other parameters). However, while CAPI-QPI-PLAN (and CONFIDENT MC-LSPI) evaluates O( log ( 1 / /u1D700 )) (again in terms of /u1D700 only) policies to find one which is ˜ O( /u1D700 ) -optimal, CONFIDENT MC-POLITEX needs to compute ˜ O( 1 / /u1D700 2 ) policies to achieve the same. As a consequence, CONFIDENT MC-POLITEX only achieves ˜ O( 1 / /u1D700 4 ) query complexity, and to match CAPI-QPI-PLAN's ˜ O( 1 / /u1D700 2 ) complexity, one would need to come up with either significantly better policy evaluation methods (potentially using the similarity in the subsequent policies) or a much faster (exponential vs. square-root) convergence rate in the suboptimality of the policy sequence produced by CONFIDENT MC-POLITEX.

The rest of the paper is organized as follows: The model and notation are introduced in Section 2. CAPI is introduced and analyzed in Section 3. Planning with /u1D45E /u1D70B -realizability is introduced in Section 4, with CAPI-QPI-PLAN being built-up and analyzed in Sections 4.1 and 4.2. In particular, the proof of Theorem 1.2 is given in Section 4.2. Several proofs are relegated to appendices, in particular, Theorem 1.3 is proved and implementation details of CAPI-QPI-PLAN are discussed in Appendix G, while Theorem 1.4 is proved in Appendix H.

## 2 Notation and preliminaries

Let N = { 0 , 1 , . . . } denote the set of natural numbers, N + = { 1 , 2 , . . . } the positive integers. For some integer /u1D456 , let [ /u1D456 ] = { 0 , . . . , /u1D456 -1 } . For /u1D465 ∈ R , let /ceilingleft /u1D465 /ceilingright denote the smallest integer i such that /u1D456 ≥ /u1D465 . For a positive definite /u1D449 ∈ R /u1D451 × /u1D451 and /u1D465 ∈ R /u1D451 , let ‖ /u1D465 ‖ 2 /u1D449 = /u1D465 /latticetop /u1D449 /u1D465 . For matrices /u1D434 and /u1D435 , we say that /u1D434 /followsequal /u1D435 if /u1D434 -/u1D435 is positive semidefinite. Let I be the /u1D451 -dimensional identity matrix. For compatible vectors /u1D465 , /u1D466 , let 〈 /u1D465 , /u1D466 〉 be their inner product: 〈 /u1D465 , /u1D466 〉 = /u1D465 /latticetop /u1D466 . Let M 1 ( /u1D44B ) denote the space of probability distributions supported on the set /u1D44B (throughout, we assume that the /u1D70E -algebra is implicit). We write /u1D44E ≈ /u1D700 /u1D44F for /u1D44E , /u1D44F , /u1D700 ∈ R if | /u1D44E -/u1D44F | ≤ /u1D700 . We denote by ˜ O(·) and ˜ Θ (·) the variants of the big-O notation that hide polylogarithmic factors.

A Markov Decision Process (MDP) is a tuple /u1D440 = (S , A , Q) , where S is a measurable state space, A is a finite action space, and Q : S × A → M 1 (S × [ 0 , 1 ]) is the transition-reward kernel. We define the transition and reward distributions /u1D443 : S × A → M 1 (S) and R : S × A → M 1 ([ 0 , 1 ]) as the marginals of Q . By a slight abuse of notation, for any /u1D460 ∈ S and /u1D44E ∈ A , let /u1D443 (·| /u1D460 , /u1D44E ) and R(·| /u1D460 , /u1D44E ) denote the distributions /u1D443 ( /u1D460 , /u1D44E ) and R( /u1D460 , /u1D44E ) , respectively. We further denote by /u1D45F ( /u1D460 , /u1D44E ) = ∫ 1 0 /u1D465 d R( /u1D465 | /u1D460 , /u1D44E ) the expected reward for an action /u1D44E ∈ A taken in a state /u1D460 ∈ S . Without loss of generality, we assume that there is a designated initial state /u1D460 0 ∈ S .

Starting from any state /u1D460 ∈ S , a stationary memoryless policy /u1D70B : S → M 1 (A) interacts with the MDP in a sequential manner for time-steps /u1D461 ∈ N , defining a probability distribution P /u1D70B ,/u1D460 over the episode trajectory { /u1D446 /u1D456 , /u1D434 /u1D456 , /u1D445 /u1D456 } /u1D456 ∈ N as follows: /u1D446 0 = /u1D460 deterministically, /u1D434 /u1D456 ∼ /u1D70B ( /u1D446 /u1D456 ) , and ( /u1D446 /u1D456 + 1 , /u1D445 /u1D456 ) ∼ Q( /u1D446 /u1D456 , /u1D434 /u1D456 ) . By a slight variation, let P /u1D70B ,/u1D460 , /u1D44E denote (for some /u1D44E ∈ A ) the distribution of the trajectory when /u1D434 0 = /u1D44E deterministically, while the distribution of the rest of the trajectory is defined analogously.

This allows us to conveniently define the expected state-value and action-value functions in the discounted setting we consider, for some discount factor 0 &lt; /u1D6FE &lt; 1 , respectively, as where throughout the paper we use the convention that E · is the expectation operator corresponding to a distribution P· (e.g., E /u1D70B ,/u1D460 is the expectation with respect to P /u1D70B ,/u1D460 ). It is well known (see, e.g., Puterman, 1994) that there exists an optimal stationary deterministic memoryless policy /u1D70B ★ such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let /u1D463 ★ = /u1D463 /u1D70B ★ and /u1D45E ★ = /u1D45E /u1D70B ★ . For any policy /u1D70B , /u1D463 /u1D70B and /u1D45E /u1D70B are known to satisfy the Bellman equations [Puterman, 1994]:

Finally, we call a policy /u1D70B deterministic if for all states, /u1D70B ( /u1D460 ) is a distribution that assigns unit weight to one action and zero weight to the others. With a slight abuse of notation, for a deterministic policy /u1D70B , we denote by /u1D70B ( /u1D460 ) the action /u1D70B chooses (deterministically) in state /u1D460 ∈ S .

## 3 Confident Approximate Policy Iteration

In this section we introduce CONFIDENT APPROXIMATE POLICY ITERATION (CAPI), our new approximate dynamic programming algorithm. In approximate dynamic programming, the methods are designed around oracles that return either an approximation to the application of the Bellman optimality operator to a value function ('approximate value iteration'), or an approximation to the value function of some policy ('approximate policy iteration'). Our setting is the second. The novelty is that we assume access to the accuracy of the approximation and use this knowledge to modify the policy update, which leads to improved guarantees on the suboptimality of the computed policy.

We present the pseudocodes of API [Bertsekas and Tsitsiklis, 1996] and CAPI jointly in Algorithm 1: starting from an arbitrary (deterministic) policy /u1D70B 0 , the algorithm iterates a policy estimation (Line 2) and a policy update step (Line 3) /u1D43C times. The policy update for API is greedy with respect to the action-value estimates ˆ /u1D45E and is defined as /u1D70B ˆ /u1D45E ( /u1D460 ) = arg max /u1D44E ∈A ˆ /u1D45E ( /u1D460 , /u1D44E ) . We assume that arg max /u1D44E ∈A breaks ties in a consistent manner by ordering the actions (using the notation A = (A 1 , . . . , A|A|) ) and always choosing action A /u1D456 with the lowest index /u1D456 that achieves the maximum. For CAPI, the policy update further relies on a global estimation-accuracy parameter /u1D714 , and a set of fixed-states S fix ⊆ S . For the purposes of this section, it is enough to keep S fix = {} . CAPI updates the policy to one that acts greedily with respect to ˆ /u1D45E only on states that are not in S fix and where it is confident that this leads to an improvement over the previous policy (Case 5a); otherwise, the new policy will return the same action as the previous one (Case 5b). To decide, ˆ /u1D45E ( /u1D460 , /u1D70B ( /u1D460 )) + /u1D714 is treated as the upper bound on the previous policy's value, and max /u1D44E ∈A ˆ /u1D45E ( /u1D460 , /u1D44E ) -/u1D714 as the lower bound of the action-value of the greedy action (Eq. 5):

Note that /u1D70B ˆ /u1D45E , /u1D70B, S fix also depends on /u1D714 , however, this dependence is omitted from the notation (as /u1D714 is kept fixed throughout).

<!-- formula-not-decoded -->

CAPI can also be seen as a refinement of CONSERVATIVE POLICY ITERATION (CPI) of Kakade and Langford [2002] with some important differences: While CPI introduces a global parameter to ensure the update stays close to the previous policy, CAPI has no such parameter, and it dynamically decides when to stay close to (more precisely, use) the previous policy, individually for every state, based on whether there is evidence for a guaranteed improvement.

Let /u1D70B be any stationary deterministic memoryless policy, ˆ /u1D45E /u1D70B : S×A → R be any function, /u1D714 ∈ R + , and S fix ⊆ S . First, we show that as long as ˆ /u1D45E /u1D70B is an /u1D714 -accurate estimate of /u1D45E /u1D70B , the CAPI policy update only improves the policy's values:

Algorithm 1 APPROXIMATE POLICY ITERATION (API) and CONFIDENT APPROXIMATE POLICY ITERATION (CAPI)

```
1: for /u1D456 = 1 to /u1D43C do 2: ˆ /u1D45E ← ESTIMATE ( /u1D70B /u1D456 -1 ) 3: /u1D70B /u1D456 ← { /u1D70B ˆ /u1D45E API /u1D70B ˆ /u1D45E , /u1D70B /u1D456 -1 , S fix CAPI 4: return /u1D70B /u1D43C
```

Lemma 3.1 (No deterioration) . Let /u1D70B ′ = /u1D70B ˆ /u1D45E /u1D70B , /u1D70B , S fix . Assume that for all /u1D460 ∈ S \ S fix and /u1D44E ∈ A , ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) ≈ /u1D714 /u1D45E /u1D70B ( /u1D460 , /u1D44E ) . Then, for any /u1D460 ∈ S , /u1D463 /u1D70B ′ ( /u1D460 ) ≥ /u1D463 /u1D70B ( /u1D460 ) .

Proof. Fix any /u1D460 ∈ S . If /u1D460 ∈ S fix or ˆ /u1D45E /u1D70B ( /u1D460 , /u1D70B ( /u1D460 )) + /u1D714 ≥ max /u1D44E ∈A ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) -/u1D714 , then /u1D70B ′ ( /u1D460 ) = /u1D70B ( /u1D460 ) and therefore /u1D45E /u1D70B ( /u1D460 , /u1D70B ′ ( /u1D460 )) = /u1D463 /u1D70B ( /u1D460 ) . Otherwise, /u1D460 ∉ S fix and ˆ /u1D45E /u1D70B ( /u1D460 , /u1D70B ( /u1D460 )) + /u1D714 ≤ max /u1D44E ∈A ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) -/u1D714 , hence /u1D70B ′ ( /u1D460 ) = arg max /u1D44E ∈A ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) , and it follows by our assumptions that /u1D45E /u1D70B ( /u1D460 , /u1D70B ′ ( /u1D460 )) ≥ ˆ /u1D45E /u1D70B ( /u1D460 , /u1D70B ′ ( /u1D460 )) -/u1D714 = max /u1D44E ∈A ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) -/u1D714 &gt; ˆ /u1D45E /u1D70B ( /u1D460 , /u1D70B ( /u1D460 )) + /u1D714 ≥ /u1D45E /u1D70B ( /u1D460 , /u1D70B ( /u1D460 )) = /u1D463 /u1D70B ( /u1D460 ) . Therefore, in any case, /u1D45E /u1D70B ( /u1D460 , /u1D70B ′ ( /u1D460 )) ≥ /u1D463 /u1D70B ( /u1D460 ) . Since this holds for any /u1D460 ∈ S , the Policy Improvement Theorem [Sutton and Barto, 2018, Section 4.2] implies that for any /u1D460 ∈ S , /u1D463 /u1D70B ′ ( /u1D460 ) ≥ /u1D463 /u1D70B ( /u1D460 ) . /square

Next we introduce two approximate optimality criterion for a policy on a set of states:

Definition 3.2 (Policy optimality on a set of states) . A policy /u1D70B is Δ -optimal (for some Δ ≥ 0 ) on a set of states S ′ ⊆ S , if for all /u1D460 ∈ S ′ , /u1D463 ★ ( /u1D460 ) -/u1D463 /u1D70B ( /u1D460 ) ≤ Δ .

Note that in the special case of S ′ = S the first property implies the second, that is, if /u1D70B is Δ -optimal on S , then it is also next-state Δ -optimal on S . Next, we show that the suboptimality of a policy updated by CAPI evolves as follows (the proof is relegated to Appendix A):

Definition 3.3 (Next-state optimality on a set of states) . A policy /u1D70B is next-state Δ -optimal (for some Δ ≥ 0 ) on a set of states S ′ ⊆ S , if for all /u1D460 ∈ S ′ and all actions /u1D44E ∈ A , ∫ /u1D460 ′ ∈S ( /u1D463 ★ ( /u1D460 ′ ) -/u1D463 /u1D70B ( /u1D460 ′ )) d /u1D443 ( /u1D460 ′ | /u1D460 , /u1D44E ) ≤ Δ .

Lemma 3.4 (Iteration progress) . Let /u1D70B ′ = /u1D70B ˆ /u1D45E /u1D70B , /u1D70B , S fix . Assume that for all /u1D460 ∈ S \ S fix and /u1D44E ∈ A , ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) ≈ /u1D714 /u1D45E /u1D70B ( /u1D460 , /u1D44E ) , and that /u1D70B is next-state Δ -optimal on S \ S fix. Then /u1D70B ′ is ( 4 /u1D714 + /u1D6FE Δ ) -optimal on S \ S fix .

## 3.1 CAPI guarantee with accurate estimation everywhere

To obtain a final suboptimality guarantee for CAPI, first consider the ideal scenario in which we assume that we have a mechanism to estimate /u1D45E /u1D70B ( /u1D460 , /u1D44E ) up to some /u1D714 accuracy for all /u1D460 ∈ S and /u1D44E ∈ A , and for any policy /u1D70B :

Theorem 3.6 (CAPI performance) . Assume CAPI (Algorithm 1) is run with S fix = {} , iteration count to /u1D43C = /ceilingleft log /u1D714 / log /u1D6FE /ceilingright , and suppose that the estimation used in Line 2 satisfies Assumption 3.5. Then the policy /u1D70B /u1D43C returned by the algorithm is 5 /u1D714 /( 1 -/u1D6FE ) -optimal on S .

Assumption 3.5. There is an oracle called ESTIMATE that accepts a policy /u1D70B and returns ˆ /u1D45E /u1D70B : S × A → R such that for all /u1D460 ∈ S and /u1D44E ∈ A , ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) ≈ /u1D714 /u1D45E /u1D70B ( /u1D460 , /u1D44E ) .

Proof. We prove by induction that policy /u1D70B /u1D456 is Δ /u1D456 -optimal on S for Δ /u1D456 = 4 /u1D714 ∑ /u1D457 ∈[ /u1D456 ] /u1D6FE /u1D457 + /u1D6FE /u1D456 1 -/u1D6FE . This holds immediately for the base case of /u1D456 = 0 , as rewards are bounded in [ 0 , 1 ] and thus /u1D463 ★ ( /u1D460 ) ≤ 1 /( 1 -/u1D6FE ) for any /u1D460 . Assuming now that the inductive hypothesis holds for /u1D456 -1 we observe that /u1D70B /u1D456 -1 is next-state Δ -optimal on S = S\S fix. Together with Assumption 3.5, this implies that the conditions of Lemma 3.4 are satisfied for /u1D70B = /u1D70B /u1D456 -1 , which yields /u1D463 ★ ( /u1D460 ) -/u1D463 /u1D70B /u1D456 ( /u1D460 ) ≤ 4 /u1D714 + /u1D6FE Δ /u1D456 -1 = Δ /u1D456 , finishing the induction. Finally, by the definition of /u1D43C , /u1D70B /u1D43C is Δ /u1D43C -optimal with Δ /u1D43C ≤ 4 /u1D714 1 -/u1D6FE + /u1D6FE /u1D43C 1 -/u1D6FE ≤ 5 /u1D714 1 -/u1D6FE . /square

## 4 Local access planning with /u1D45E /u1D70B -realizability

Our planner, CAPI-QPI-PLAN, is based on the CONFIDENT MC-LSPI algorithm of Yin et al. [2022]. This latter algorithm gradually builds a core set of state-action pairs whose corresponding features are informative. The /u1D45E -values of the state-action pairs in the core set are estimated

## Algorithm 2 MEASURE

```
1: Input: state /u1D460 , action /u1D44E , deterministic policy /u1D70B , set of states S ′ ⊆ S , accuracy /u1D714 > 0 , failure probability /u1D701 ∈ ( 0 , 1 ] 2: Initialize: /u1D43B ←/ceilingleft log (( /u1D714 / 4 )( 1 -/u1D6FE ))/ log /u1D6FE /ceilingright , /u1D45B ← ⌈ ( /u1D714 / 4 ) -2 ( 1 -/u1D6FE ) -2 log ( 2 / /u1D701 )/ 2 ⌉ 3: for /u1D456 = 1 to /u1D45B do 4: ( /u1D446 , /u1D445 /u1D456 , 0 ) ← SIMULATOR ( /u1D460 , /u1D44E ) 5: for ℎ = 1 to /u1D43B -1 do 6: if /u1D446 ∉ S ′ then return ( discover , /u1D446 ) 7: /u1D434 ← /u1D70B ( /u1D446 ) 8: ( /u1D446 , /u1D445 /u1D456 , ℎ ) ← SIMULATOR ( /u1D446 , /u1D434 ) ⊲ Call to the simulator oracle 9: return ( success , 1 /u1D45B ∑ /u1D45B /u1D456 = 1 ∑ /u1D43B -1 ℎ = 0 /u1D6FE ℎ /u1D445 /u1D456 , ℎ )
```

using rollouts. The procedure is restarted with an extended core set whenever the algorithm encounters a new informative feature. If such a new feature is not encountered, the estimation error can be controlled, and the estimation is extended to all state-action pairs using the least-squares estimator. Finally, the extended estimation is used in Line 2 of API.

CAPI-QPI-PLAN improves upon CONFIDENT MC-LSPI in two ways. First, using CAPI instead of API improves the final suboptimality bound by a factor of the effective horizon. Second, we apply a novel analysis on a more modular variant of the CONFIDENTROLLOUT subroutine used in CONFIDENT MC-LSPI, which delivers /u1D45E -estimation accuracy guarantees with respect to a large class of policies simultaneously. This allows for a dynamically evolving version of policy iteration, that does not have to restart whenever a new informative feature is encountered. Intuitively, this prevents duplication of work.

## 4.1 Estimation oracle

To obtain an algorithm for planning with local access whose performance degrades gracefully with the uniform approximation error, we must weaken Assumption 3.5. This is because under local access, we cannot guarantee to cover all states or hope to obtain accurate /u1D45E -value estimates for all states. Instead, we are interested in an accuracy guarantee that holds for /u1D45E -values only on some subset S ′ ⊆ S of states, but holds simultaneously for any policy that agrees with /u1D70B on S ′ but may take arbitrary values elsewhere. For this, we define the extended set of policies:

Definition 4.1. Let Π det be the set of all stationary deterministic memoryless policies, /u1D70B ∈ Π det, and S ′ ⊆ S . For ( /u1D70B , S ′ ) , we define Π /u1D70B, S ′ to be the set of policies that agree with /u1D70B on /u1D460 ∈ S ′ :

<!-- formula-not-decoded -->

We aim to first accurately estimate /u1D45E /u1D70B ( /u1D460 , /u1D44E ) for some specific ( /u1D460 , /u1D44E ) pairs, based on which we extend the estimates to other state-action pairs using least-squares. To this end, we first devise a subroutine called MEASURE (Algorithm 2). MEASURE is a modularized variant of the CONFIDENTROLLOUT subroutine of Yin et al. [2022]. The modularity of our variant is due to the parameter S ′ that corresponds to the set of states on which the planner is 'confident' for CONFIDENTROLLOUT. MEASURE unrolls the policy /u1D70B starting from ( /u1D460 , /u1D44E ) for a number of episodes, each lasting /u1D43B steps, and returns with the average measured reward. Throughout, we let /u1D43B = /ceilingleft log (( /u1D714 / 4 )( 1 -/u1D6FE ))/ log /u1D6FE /ceilingright be the effective horizon. At the end of this process, MEASURE returns status success along with the empirical average /u1D45E -value, where compared to Eq. (3), the discounted summation of rewards is truncated to /u1D43B . If, however, the algorithm encounters a state not in its input S ′ , it returns with status discover , along with that state. This is because in such cases, the algorithm could no longer guarantee an accurate estimation with respect to any member of the extended set of policies. The next lemma, proved in Appendix B, shows that MEASURE provides accurate estimates of the action-value functions for members of the extended policy set.

Lemma 4.2. For any input parameters /u1D460 ∈ S , /u1D44E ∈ A , /u1D70B ∈ Π det , S ′ ⊂ S , /u1D714 &gt; 0 , /u1D701 ∈ ( 0 , 1 ) , MEASURE either returns with ( discover , /u1D460 ′ ) for some /u1D460 ′ ∉ S ′ (Line 6), or it returns with ( success , ˜ /u1D45E ) such that with probability at least 1 -/u1D701 ,

<!-- formula-not-decoded -->

Suppose we have a list of state-action pairs /u1D436 = ( /u1D460 /u1D456 , /u1D44E /u1D456 ) /u1D456 ∈[| /u1D436 | ] and corresponding /u1D45E -estimates ¯ /u1D45E = ( ¯ /u1D45E /u1D456 ) /u1D456 ∈| /u1D436 | . We use the regularized least-squares estimator LSE (Eq. 8) to extend the estimates for all state-action pairs, with regularization parameter /u1D706 = /u1D714 2 / /u1D435 2 (recall that /u1D435 is defined in Definition 1.1):

∑ For /u1D436 = ¯ /u1D45E = () (the empty sequence), we define LSE /u1D436 , ¯ /u1D45E (· , ·) = 0 . This estimator satisfies the guarantee below.

<!-- formula-not-decoded -->

Lemma 4.3. Let /u1D70B be a stationary deterministic memoryless policy. Let /u1D436 = ( /u1D460 /u1D456 , /u1D44E /u1D456 ) /u1D456 ∈[ /u1D45B ] be sequences of state-action pairs of some length /u1D45B ∈ N and ¯ /u1D45E = ( ¯ /u1D45E /u1D456 ) /u1D456 ∈[ /u1D45B ] a sequence of corresponding reals such that for all /u1D456 ∈ [ /u1D45B ] , /u1D45E /u1D70B ( /u1D460 /u1D456 , /u1D44E /u1D456 ) ≈ /u1D714 ¯ /u1D45E /u1D456 . Then, for all /u1D460 , /u1D44E ∈ S × A , where /u1D700 is the uniform approximation error from Definition 1.1.

<!-- formula-not-decoded -->

The proof is given in Appendix C. The order of the estimation accuracy bound (Eq. 9) is optimal, as shown by the lower bounds of Du et al. [2019] and Lattimore et al. [2020].

We intend to use the LSE estimator given in Eq. (8) and the bound in Lemma 4.3 only for state-action pairs where ‖ /u1D711 ( /u1D460 , /u1D44E )‖ /u1D449 ( /u1D436 ) -1 ≤ 1 (and /u1D45B = ˜ O( /u1D451 ) ). We call these state-action pairs covered by /u1D436 , and we call a state /u1D460 covered by /u1D436 if for all their corresponding actions /u1D44E , the pair ( /u1D460 , /u1D44E ) is covered by /u1D436 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will use the parameter S fix of CAPI (see CAPI's update rule in Eq. 5) to ensure policies are only updated on covered states, where the approximation error is well-controlled by Eq. (9).

## 4.2 Main algorithm

Finally, we are ready to introduce CAPI-QPI-PLAN, presented in Algorithm 3, our algorithm for planning with local access under approximate /u1D45E /u1D70B -realizability. For this, we define levels /u1D459 = 0 , 1 , . . . , /u1D43B , and corresponding suboptimality requirements: For any /u1D459 ∈ [ /u1D43B + 1 ] , let for some ˜ /u1D451 = ˜ Θ ( /u1D451 ) defined in Eq. (13). For each level /u1D459 , the algorithm maintains a policy /u1D70B /u1D459 and a set of covered states on which it can guarantee that /u1D70B /u1D459 is a Δ /u1D459 -optimal policy. More specifically, this set is Cover ( /u1D436 /u1D459 ) , where /u1D436 /u1D459 is a list of state-action pairs with elements /u1D436 /u1D459 ,/u1D456 = ( /u1D460 /u1D456 /u1D459 , /u1D44E /u1D456 /u1D459 ) for /u1D456 ∈ [| /u1D436 /u1D459 |] . The algorithm maintains the following suboptimality guarantee below, which we prove in Appendix E after showing some further key properties of the algorithm.

Lemma 4.4. Assuming that Eq. (6) holds whenever MEASURE returns success , /u1D70B /u1D459 is Δ /u1D459 -optimal on Cover ( /u1D436 /u1D459 ) (Definition 3.2) for all /u1D459 ∈ [ /u1D43B + 1 ] at the end of every iteration of the main loop of CAPI-QPI-PLAN .

CAPI-QPI-PLAN aims to improve the policies, while propagating the members of /u1D436 /u1D459 to /u1D436 /u1D459 + 1 , and so on, all the way to /u1D436 /u1D43B . During this, whenever the algorithm discovers a state-action pair with a sufficiently 'new' feature direction, this pair is appended to the sequence /u1D436 0 corresponding to level 0 , as there are no suboptimality guarantees yet available for such a state. However, such a discovery can only happen ˜ O( /u1D451 ) times. When, eventually, all discovered state-action pairs end up in /u1D436 /u1D43B , the final suboptimality guarantee is reached, and the algorithm returns with the final policy. Note that in the local access setting we consider, the algorithm cannot enumerate the set Cover ( /u1D436 /u1D459 ) , but can answer membership queries, that is, for any /u1D460 ∈ S it encounters, it is able to decide if /u1D460 ∈ Cover ( /u1D436 /u1D459 ) . The algorithm maintains sequences ¯ /u1D45E /u1D459 , corresponding to /u1D436 /u1D459 , for each level /u1D459 . Whenever a new ( /u1D460 , /u1D44E ) pair is appended to the sequence /u1D436 /u1D459 , a corresponding ⊥ symbol is appended to the sequence ¯ /u1D45E /u1D459 , to signal that an estimate of /u1D45E /u1D70B /u1D459 ( /u1D460 , /u1D44E ) is not yet known.

## Algorithm 3 CAPI-QPI-PLAN

```
1: Input: initial state /u1D460 0 ∈ S , dimensionality /u1D451 , parameter bound /u1D435 , accuracy /u1D714 , failure probability /u1D6FF > 0 2: Initialize: /u1D43B ← /ceilingleft log (( /u1D714 / 4 )( 1 -/u1D6FE ))/ log /u1D6FE /ceilingright , for /u1D459 ∈ [ /u1D43B + 1 ] , /u1D436 /u1D459 ← () , ¯ /u1D45E /u1D459 ← () , /u1D70B /u1D459 ← policy that always returns action A 1 , /u1D706 ← /u1D714 2 / /u1D435 2 3: while True do ⊲ main loop 4: if ∃ /u1D44E ∈ A , ( /u1D460 0 , /u1D44E ) ∉ ActionCover ( /u1D436 0 ) then 5: append ( /u1D460 0 , /u1D44E ) to /u1D436 0 , append ⊥ to ¯ /u1D45E 0 6: break 7: let ℓ be the smallest integer such that ¯ /u1D45E ℓ,/u1D45A = ⊥ for some /u1D45A ; set ℓ = /u1D43B if no such /u1D459 exists 8: if ℓ = /u1D43B then return /u1D70B /u1D43B 9: ( status , result ) ← MEASURE ( /u1D460 /u1D45A ℓ , /u1D44E /u1D45A ℓ , /u1D70B ℓ , Cover ( /u1D436 ℓ ) , /u1D714, /u1D6FF /( ˜ /u1D451 /u1D43B )) ⊲ recall /u1D436 ℓ,/u1D45A = ( /u1D460 /u1D45A ℓ , /u1D44E /u1D45A ℓ ) 10: if status = discover then 11: append ( result , /u1D44E ) to /u1D436 0 for some /u1D44E such that ( result , /u1D44E ) ∉ ActionCover ( /u1D436 0 ) 12: append ⊥ to ¯ /u1D45E 0 13: break 14: ¯ /u1D45E ℓ,/u1D45A ← result 15: if /nexists /u1D45A ′ such that ¯ /u1D45E ℓ,/u1D45A ′ = ⊥ then 16: ˆ /u1D45E ← LSE /u1D436 ℓ , ¯ /u1D45E ℓ 17: /u1D70B ′ ← /u1D70B ˆ /u1D45E , /u1D70B ℓ , S\ Cover ( /u1D436 ℓ ) 18: /u1D70B ℓ + 1 ←( /u1D460 ↦→ /u1D70B ℓ + 1 ( /u1D460 ) if /u1D460 ∈ Cover ( /u1D436 ℓ + 1 ) else /u1D70B ′ ( /u1D460 )) 19: for ( /u1D460 , /u1D44E ) ∈ /u1D436 ℓ such that ( /u1D460 , /u1D44E ) ∉ /u1D436 ℓ + 1 do 20: append ( /u1D460 , /u1D44E ) to /u1D436 ℓ + 1 , ⊥ to ¯ /u1D45E ℓ + 1
```

After initializing /u1D436 0 to cover the initial state /u1D460 0 (Lines 4 to 6), the algorithm measures /u1D45E /u1D70B ℓ ( /u1D460 , /u1D44E ) for the smallest level ℓ for which there still exists a ⊥ in the corresponding ¯ /u1D45E ℓ . After a successful measurement, if there are no more ⊥ 's left at this level (i.e., in ¯ /u1D45E ℓ ), the algorithm executes a policy update on /u1D70B ℓ (Line 17) using the least-squares estimate obtained from the measurements at this level, but only for states in Cover ( /u1D436 ℓ ) (using S fix = S\ Cover ( /u1D436 ℓ ) ). Next, Line 18 merges this new policy /u1D70B ′ with the existing policy /u1D70B ℓ + 1 of the next level, setting /u1D70B ℓ + 1 to be the policy /u1D70B ′′ defined as

This ensures that the existing policy /u1D70B ℓ + 1 remains unchanged by /u1D70B ′′ (its replacement) on states that are already covered by /u1D436 ℓ + 1 , and therefore /u1D70B ′′ ∈ Π /u1D70B ℓ + 1 , Cover ( /u1D436 ℓ + 1 ) = Π /u1D70B ′′ , Cover ( /u1D436 ℓ + 1 ) . We also observe that /u1D436 /u1D459 can only grow for any /u1D459 (elements are never removed from these sequences), thus for any update where /u1D436 /u1D459 is assigned a new value /u1D436 ′ /u1D459 (Lines 5, 11, and 20), /u1D449 ( /u1D436 ′ /u1D459 ) /followsequal /u1D449 ( /u1D436 /u1D459 ) , and therefore Cover ( /u1D436 ′ /u1D459 ) ⊇ Cover ( /u1D436 /u1D459 ) and Π /u1D70B /u1D459 , Cover ( /u1D436 ′ /u1D459 ) ⊆ Π /u1D70B /u1D459 , Cover ( /u1D436 /u1D459 ) . Combining these properties yields the following result:

<!-- formula-not-decoded -->

Lemma 4.5. If for any /u1D459 ∈ [ /u1D43B ] , /u1D70B /u1D459 and /u1D436 /u1D459 take some values /u1D70B old /u1D459 and /u1D436 old /u1D459 at any point in the execution of the algorithm, then at any later point during the execution, /u1D70B /u1D459 ∈ Π /u1D70B /u1D459 , Cover ( /u1D436 /u1D459 ) ⊆ Π /u1D70B old /u1D459 , Cover ( /u1D436 old /u1D459 ) .

Any value in ¯ /u1D45E /u1D459 that is set to anything other than ⊥ will never change again. Since as long as the sample paths generated by MEASURE in Line 9 of CAPI-QPI-PLAN remain in Cover ( /u1D436 /u1D459 ) , their distribution is the same under any policy from Π /u1D70B /u1D459 , Cover ( /u1D436 /u1D459 ) , the ¯ /u1D45E /u1D459 estimates are valid for these policies, as well. Combined with Lemma 4.5, we get that the accuracy guarantees of Lemma 4.2 continue to hold throughout:

Lemma 4.6. Assuming that Eq. (6) holds whenever MEASURE returns success , for any level /u1D459 and index /u1D45A such that ¯ /u1D45E /u1D459 ,/u1D45A ≠ ⊥ , /u1D45E /u1D70B ′ ( /u1D460 /u1D45A /u1D459 , /u1D44E /u1D45A /u1D459 ) ≈ /u1D714 ¯ /u1D45E /u1D459 ,/u1D45A for all /u1D70B ′ ∈ Π /u1D70B /u1D459 , Cover ( /u1D436 /u1D459 ) throughout the execution of CAPI-QPI-PLAN .

Once /u1D70B ℓ + 1 is updated in Line 18, in Line 20 we append to the sequence /u1D436 ℓ + 1 all members of /u1D436 ℓ that are not yet in /u1D436 ℓ + 1 , while adding a corresponding ⊥ to ¯ /u1D45E ℓ + 1 indicating that these /u1D45E -values are not yet measured for policy /u1D70B ℓ + 1 . Thus, whenever all ⊥ values disappear from some level /u1D459 ∈ [ /u1D43B + 1 ] , by the end of that iteration /u1D436 /u1D459 + 1 = /u1D436 /u1D459 , and hence ActionCover ( /u1D436 /u1D459 ) = ActionCover ( /u1D436 /u1D459 + 1 ) . Together with

the fact that for any /u1D459 ∈ [ /u1D43B + 1 ] , whenever a new state-action pair is appended to /u1D436 /u1D459 , an ⊥ symbol is appended to ¯ /u1D45E /u1D459 , we have by induction the following result:

Lemma 4.7. Throughout the execution of CAPI-QPI-PLAN ,after Line 7 when ℓ is set,

ActionCover ( /u1D436 0 ) = ActionCover ( /u1D436 1 ) = · · · = ActionCover ( /u1D436 ℓ ) .

As a result, whenever the MEASURE call of Line 9 outputs ( discover , /u1D460 ) for some state /u1D460 , by Lemma 4.2, there is an action /u1D44E ∈ A such that ( /u1D460 , /u1D44E ) ∉ ActionCover ( /u1D436 ℓ ) = ActionCover ( /u1D436 0 ) . This explains why adding such an ( /u1D460 , /u1D44E ) pair to /u1D436 0 is always possible in Line 11. Consider the /u1D456 th time Line 11 is executed, and denote /u1D460 by /u1D460 /u1D456 and /u1D44E by /u1D44E /u1D456 , and /u1D449 /u1D456 = /u1D706 I + ∑ /u1D456 -1 /u1D461 = 1 /u1D711 ( /u1D460 /u1D461 , /u1D44E /u1D461 ) /u1D711 ( /u1D460 /u1D461 , /u1D44E /u1D461 ) /latticetop . Observe that as /u1D449 /u1D456 = /u1D449 ( /u1D436 ) , ( /u1D460 /u1D456 , /u1D44E /u1D456 ) ∉ ActionCover ( /u1D436 0 ) implies ‖ /u1D711 ( /u1D460 /u1D456 , /u1D44E /u1D456 )‖ /u1D449 -1 /u1D456 &gt; 1 . Therefore, ∑ /u1D456 /u1D461 = 1 min { 1 , ‖ /u1D711 ( /u1D460 /u1D461 , /u1D44E /u1D461 )‖ /u1D449 -1 /u1D461 } = /u1D456 , and thus by the elliptical potential

<!-- formula-not-decoded -->

lemma [Lattimore and Szepesvári, 2020, Lemma 19.4], /u1D456 ≤ 2 /u1D451 log ( /u1D451 /u1D706 + /u1D456 /u1D43F 2 /u1D451 /u1D706 ) . This inequality is satisfied by the largest value of /u1D456 , that is, the total number of times MEASURE returns with discover . Since any element of /u1D436 /u1D459 is also an element of /u1D436 0 for any /u1D459 ∈ [ /u1D43B + 1 ] , we have that at any time during the execution of CAPI-QPI-PLAN,

When CAPI-QPI-PLAN returns at Line 8 with the policy /u1D70B /u1D43B , it is Δ /u1D43B -optimal on Cover ( /u1D436 /u1D43B ) by Lemma 4.4 when the estimates of MEASURE are correct. Furthermore, /u1D460 0 ∈ Cover ( /u1D436 0 ) is guaranteed by Lines 4 to 6, and hence /u1D460 0 ∈ Cover ( /u1D436 /u1D43B ) by Lemma 4.7 when the algorithm finishes. Hence, bounding Δ /u1D43B using the definition of /u1D43B immediately gives the following result:

Lemma4.8. Assuming that Eq. (6) holds whenever MEASURE returns success , the policy /u1D70B returned by CAPI-QPI-PLAN is Δ -optimal on { /u1D460 0 } for

<!-- formula-not-decoded -->

To finish the proof of Theorem 1.2, we only need to analyze the query complexity and the failure probability (i.e., the probability of Eq. (6) not being satisfied for some MEASURE call that returns success ) of CAPI-QPI-PLAN:

Proof of Theorem 1.2. Both the total failure probability and query complexity of CAPI-QPI-PLAN depend on the number of times MEASURE is executed, as this is the only source of randomness and of interaction with the simulator. MEASURE can return discover at most | /u1D436 0 | times, which is bounded by ˜ /u1D451 by Eq. (13). For every /u1D459 ∈ [ /u1D43B ] , MEASURE is executed exactly once with returning success for each element of /u1D436 /u1D459 . Hence, by Eq. (13) again, MEASURE returns success at most ˜ /u1D451 /u1D43B times, each satisfying Eq. (6) with probability at least 1 -/u1D701 = 1 -/u1D6FF /( ˜ /u1D451 /u1D43B ) by Lemma 4.2. By the union bound, MEASURE returns success in all occasions with probability at least 1 -/u1D6FF . Hence Eq. (6) holds with probability at least 1 -/u1D6FF , which, combined with Lemma 4.8, proves Eq. (1).

Each successful run of MEASURE executes at most /u1D45B /u1D43B queries ( /u1D45B is set in Line 2 of Algorithm 2). Since /u1D43B &lt; ( 1 -/u1D6FE ) -1 log ( 4 /u1D714 -1 ( 1 -/u1D6FE ) -1 ) = ˜ O(( 1 -/u1D6FE ) -1 ) , in total CAPI-QPI-PLAN executes at most ˜ O ( /u1D451 ( 1 -/u1D6FE ) -4 /u1D714 -2 ) queries. As this happens at most ˜ /u1D451 /u1D43B times, we obtain the desired bound on the query complexity. /square

## 5 Conclusions and future work

In this paper we presented CONFIDENT APPROXIMATE POLICY ITERATION, a confident version of API, which can obtain a stationary policy with a suboptimality guarantee that scales linearly with the effective horizon /u1D43B = ˜ O( 1 /( 1 -/u1D6FE )) . This scaling is optimal as shown by Scherrer and Lesner [2012].

CAPI can be applied to local planning with approximate /u1D45E /u1D70B -realizability (yielding the CAPI-QPIPLAN algorithm) to obtain a sequence of policies with successively refined accuracies on a dynamically evolving set of states, resulting in a final, recursively defined policy achieving simultaneously the optimal suboptimality guarantee and best query cost available in the literature. More precisely,

CAPI-QPI-PLAN achieves ˜ O( /u1D700 √ /u1D451 /u1D43B ) suboptimality, where /u1D700 is the uniform policy value-function approximation error. We showed that this bound is the best (up to polylogarithmic factors) that is achievable by any planner with polynomial query cost. We also proved that the ˜ O ( /u1D451/u1D43B 4 /u1D700 -2 ) query cost of CAPI-QPI-PLAN is optimal up to polylogarithmic factors in all parameters except for /u1D43B ; whether the dependence on /u1D43B is optimal remains an open question.

Finally, our method comes at a memory and computational cost overhead, both for the final policy and the planner. It is an interesting question if this overhead necessarily comes with the API-style method we use (as it is also present in the works of Scherrer and Lesner, 2012, Scherrer, 2014), or if it is possible to reduce it by, for example, compressing the final policy into one that is greedy with respect to some action-value function realized with the features.

## Acknowledgements

The authors would like to thank Tor Lattimore and Qinghua Liu for helpful discussions. Csaba Szepesvári gratefully acknowledges the funding from Natural Sciences and Engineering Research Council (NSERC) of Canada, 'Design.R AI-assisted CPS Design' (DARPA) project and the Canada CIFAR AI Chairs Program for Amii.

## References

- Dimitri P. Bertsekas. Dynamic Programming and Optimal Control: Approximate dynamic programming , volume II. 4 edition, 2012.
- Dimitri P. Bertsekas and John N. Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1996.
- Simon S Du, Sham M Kakade, Ruosong Wang, and Lin F Yang. Is a good representation sufficient for sample efficient reinforcement learning? In International Conference on Learning Representations , 2019.
- Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pages 21372143. PMLR, 2020.
- Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In In Proc. 19th International Conference on Machine Learning . Citeseer, 2002.
- T. Lattimore and Cs. Szepesvári. Bandit Algorithms . Cambridge University Press, 2020.
- Tor Lattimore, Csaba Szepesvári, and Gellért Weisz. Learning with good feature representations in bandits and in RL with a generative model. In ICML , pages 9464-9472, 2020.
- A Woodbury Max. Inverting modified matrices. In Memorandum Rept. 42, Statistical Research Group , page 4. Princeton Univ., 1950.
- Remi Munos. Error bounds for approximate policy iteration. In ICML , pages 560-567, 2003.
- Remi Munos. Error bounds for approximate value iteration. In AAAI , pages 1006-1011, 2005.
- Martin L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . Wiley-Interscience, 1994.
- Walter Rudin et al. Principles of mathematical analysis , volume 3. McGraw-hill New York, 1976.
- Daniel Russo. Approximation benefits of policy gradient methods with aggregated states. arXiv preprint arXiv:2007.11684 , 2020.
- Bruno Scherrer. Approximate policy iteration schemes: a comparison. In International Conference on Machine Learning , pages 1314-1322. PMLR, 2014.
- Bruno Scherrer and Boris Lesner. On the use of non-stationary policies for stationary infinitehorizon Markov decision processes. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems , volume 25, 2012.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction . MIT press, 2018.

- Andrew Wagenmaker, Yifang Chen, Max Simchowitz, Simon S Du, and Kevin Jamieson. Rewardfree RL is no harder than reward-aware RL in linear Markov decision processes. arXiv preprint arXiv:2201.11206 , 2022.
- Yuanhao Wang, Ruosong Wang, and Sham Kakade. An exponential lower bound for linearly realizable MDP with constant suboptimality gap. Advances in Neural Information Processing Systems , 34, 2021.
- Gellért Weisz, Philip Amortila, and Csaba Szepesvári. Exponential lower bounds for planning in MDPswith linearly-realizable optimal action-value functions. In ALT , volume 132 of Proceedings of Machine Learning Research , pages 1237-1264, 2021.
- Chenjun Xiao, Ilbin Lee, Bo Dai, Dale Schuurmans, and Csaba Szepesvari. The curse of passive data collection in batch reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 8413-8438, 2022.
- Dong Yin, Botao Hao, Yasin Abbasi-Yadkori, Nevena Lazi´ c, and Csaba Szepesvári. Efficient local planning with linear function approximation. In International Conference on Algorithmic Learning Theory , pages 1165-1192. PMLR, 2022.
- Andrea Zanette, Alessandro Lazaric, Mykel Kochenderfer, and Emma Brunskill. Learning near optimal policies with low inherent Bellman error. In International Conference on Machine Learning , pages 10978-10989. PMLR, 2020.
- Dongruo Zhou, Jiafan He, and Quanquan Gu. Provably efficient reinforcement learning for discounted MDPs with feature mapping. arXiv preprint arXiv:2006.13165 , 2020.

## A Proof of Lemma 3.4

Take any /u1D460 ∈ S \ S fix .

where the first equality holds because /u1D70B ′ is deterministic, and the inequality is true because

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by Lemma 3.1. Next observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since, as /u1D460 ∉ S fix , either /u1D70B ′ ( /u1D460 ) is defined by Case 5a as /u1D70B ′ ( /u1D460 ) = arg max /u1D44E ∈A ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) and so ˆ /u1D45E /u1D70B ( /u1D460 , /u1D70B ′ ( /u1D460 )) = max /u1D44E ∈A ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) , or it is defined by Case 5b in which case ˆ /u1D45E /u1D70B ( /u1D460 , /u1D70B ′ ( /u1D460 )) = ˆ /u1D45E /u1D70B ( /u1D460 , /u1D70B ( /u1D460 )) ≥ max /u1D44E ∈A ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) -2 /u1D714 . Combining Eqs. (14) and (15), we obtain where in the first line we added and subtracted ˆ /u1D45E /u1D70B ( /u1D460 , /u1D70B ′ ( /u1D460 )) , and the second inequality holds as ˆ /u1D45E /u1D70B ( /u1D460 , /u1D44E ) ≈ /u1D714 /u1D45E /u1D70B ( /u1D460 , /u1D44E ) for /u1D460 ∉ S fix and /u1D44E ∈ A by the assumptions of the lemma.

<!-- formula-not-decoded -->

We continue by adding and subtracting max /u1D44E ∈A /u1D45E /u1D70B ( /u1D460 , /u1D44E ) :

where in the fifth line we used that /u1D70B is next-state Δ -optimal by assumption.

## B Proof of Lemma 4.2

For an episode trajectory { /u1D446 ℎ , /u1D434 ℎ , /u1D445 ℎ } ℎ ∈ N , let /u1D43E be the smallest positive integer such that /u1D446 /u1D43E ∉ S ′ . For any /u1D456 ∈ { 1 , . . . , /u1D45B } , let /u1D43C /u1D456 denote the indicator of the event that at the /u1D456 th iteration of the outer loop of Algorithm 2, the algorithm encounters /u1D446 ∉ S ′ in Line 6. Note that E /u1D70B ,/u1D460 , /u1D44E [ /u1D43C /u1D456 ] = P /u1D70B ,/u1D460 , /u1D44E [ 1 ≤ /u1D43E &lt; /u1D43B ] . Then, by Hoeffding's inequality (see, e.g., Lattimore and Szepesvári [2020]), with probability at least 1 -/u1D701 / 2 ,

MEASURE only returns success if all indicators are zero; therefore, the above inequality implies that if MEASURE returns success then, with probability at least 1 -/u1D701 / 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/square

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

another application of Hoeffding's inequality yields that /u1D45E /u1D70B ( /u1D460 , /u1D44E ) and ¯ /u1D45E are close with high probability: with probability at least 1 -/u1D701 / 2 ,

Pick any /u1D70B ′ ∈ Π /u1D70B, S ′ . Observe that for any /u1D460 ∈ S and /u1D44E ∈ A , the distribution of the trajectory /u1D446 0 , /u1D434 0 , /u1D445 0 , /u1D446 1 , /u1D434 1 , /u1D445 1 , . . . , /u1D434 /u1D43E -1 , /u1D445 /u1D43E -1 , /u1D446 /u1D43E is the same under P /u1D70B ′ ,/u1D460 , /u1D44E and P /u1D70B ,/u1D460 , /u1D44E , as /u1D70B and /u1D70B ′ select the same actions for states in S ′ . By Eqs. (3) to (4), we can write

<!-- formula-not-decoded -->

Combining Eqs. (16) to (18), it follows by the union bound that if MEASURE returns with ( success , ˜ /u1D45E ) , then with probability at least 1 -/u1D701 ,

## C Proof of Lemma 4.3

<!-- formula-not-decoded -->

We start the proof by showing that there exists a /u1D703 ∈ R /u1D451 such that

For any finite set /u1D44A ⊆ S × A , max ( /u1D460 , /u1D44E ) ∈ /u1D44A | /u1D45E /u1D70B ( /u1D460 , /u1D44E ) - 〈 /u1D711 ( /u1D460 , /u1D44E ) , /u1D703 ′ 〉 | is a continuous function of /u1D703 ′ , hence it attains its infimum on the compact set { /u1D703 ′ ∈ R /u1D451 : ‖ /u1D703 ′ ‖ 2 ≤ /u1D435 } . By Definition 1.1, this infimum is at most /u1D700 . Therefore, the compact sets Θ /u1D460 , /u1D44E = { /u1D703 ′ ∈ R /u1D451 : ‖ /u1D703 ′ ‖ 2 ≤ /u1D435 and | /u1D45E /u1D70B ( /u1D460 , /u1D44E ) -〈 /u1D711 ( /u1D460 , /u1D44E ) , /u1D703 ′ 〉 | ≤ /u1D700 } are non-empty for all ( /u1D460 , /u1D44E ) ∈ S × A , and any intersection of a finite collection of these sets is also non-empty. Therefore, ⋂ ( /u1D460 , /u1D44E ) ∈S×A Θ /u1D460 , /u1D44E is non-empty by [Rudin et al., 1976, Theorem 2.36], and any element /u1D703 of this set satisfies Eq. (19). For the remainder of this proof, fix such a /u1D703 .

<!-- formula-not-decoded -->

For any /u1D456 ∈ [ /u1D45B ] , with a slight abuse of notation, we introduce the shorthand /u1D711 /u1D456 = /u1D711 ( /u1D460 /u1D456 , /u1D44E /u1D456 ) , and let ˆ /u1D45E /u1D456 = 〈 /u1D703 , /u1D711 /u1D456 〉 and /u1D709 /u1D456 = ¯ /u1D45E /u1D456 -ˆ /u1D45E /u1D456 . Note that by the triangle inequality, | /u1D709 /u1D456 | ≤ | ¯ /u1D45E /u1D456 -/u1D45E /u1D70B ( /u1D460 /u1D456 , /u1D44E /u1D456 )| + | /u1D45E /u1D70B ( /u1D460 /u1D456 , /u1D44E /u1D456 ) -ˆ /u1D45E /u1D456 | ≤ /u1D714 + /u1D700 . Let ¯ /u1D703 = /u1D449 ( /u1D436 ) -1 ∑ /u1D456 ∈[ /u1D45B ] /u1D711 /u1D456 ¯ /u1D45E /u1D456 and ˆ /u1D703 = /u1D449 ( /u1D436 ) -1 ∑ /u1D456 ∈[ /u1D45B ] /u1D711 /u1D456 ˆ /u1D45E /u1D456 .

<!-- formula-not-decoded -->

where in the last line we used that /u1D449 ( /u1D436 ) /followsequal /u1D706 I .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds by the triangle inequality, the second by our bound on | /u1D709 /u1D456 | , the third by the Cauchy-Schwartz inequality, and the fourth by the positivity of /u1D706 . Putting it all together, for any /u1D460 ∈ S and /u1D44E ∈ A , using the previous bounds with /u1D463 = /u1D711 ( /u1D460 , /u1D44E ) , completing the proof.

## D Deriving next-state optimality of /u1D70B ℓ for Lemma 4.4

Lemma D.1. Assume that Eq. (6) holds whenever MEASURE returns success . At any point of CAPI-QPI-PLAN after Line 16 is executed, for any /u1D70B ′′ ∈ Π /u1D70B ℓ , Cover ( /u1D436 ℓ ) , /u1D460 ∈ Cover ( /u1D436 ℓ ) , and /u1D44E ∈ A ,

Proof. By Lemma 4.6 and Eq. (6), ¯ /u1D45E /u1D459 ,/u1D45A ≈ /u1D714 /u1D45E /u1D70B ′′ ( /u1D436 /u1D459 ,/u1D45A ) for all /u1D45A ∈ [| /u1D436 ℓ |] (recall that /u1D436 /u1D459 ,/u1D45A is the /u1D45A th state-action pair in /u1D436 /u1D459 ). Therefore, applying Lemma 4.3 with /u1D45E /u1D70B ′′ , /u1D436 ℓ and ¯ /u1D45E ℓ , as ˆ /u1D45E = LSE /u1D436 ℓ , ¯ /u1D45E ℓ , we get that for any /u1D460 ∈ Cover ( /u1D436 ℓ ) and all /u1D44E ∈ A ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality holds because ‖ /u1D711 ( /u1D460 , /u1D44E )‖ /u1D449 ( /u1D436 ℓ ) -1 ≤ 1 since /u1D460 ∈ Cover ( /u1D436 ℓ ) , | /u1D436 ℓ | ≤ ˜ /u1D451 by Eq. (13), and the definition of /u1D706 . /square

LemmaD.2. Assume that Eq. (6) holds whenever MEASURE returns success . Consider a time when Lines 17 to 20 of CAPI-QPI-PLAN are run and assume that at this time, for all /u1D459 ∈ [ /u1D43B + 1 ] , /u1D70B /u1D459 is Δ /u1D459 -optimal on Cover ( /u1D436 /u1D459 ) . Then, /u1D70B ℓ is next-state ( Δ ℓ + 4 ( /u1D714 + /u1D700 )( √ ˜ /u1D451 + 1 )/ /u1D6FE ) -optimal on Cover ( /u1D436 ℓ ) .

<!-- formula-not-decoded -->

Proof. Let /u1D70B + ℓ be defined as in Eq. (22). As /u1D70B + ℓ ∈ Π /u1D70B ℓ , Cover ( /u1D436 ℓ ) , by Lemma D.1, for any /u1D460 ∈ Cover ( /u1D436 ℓ ) and all /u1D44E ∈ A ,

Similarly, applying Lemma D.1 with /u1D70B ℓ (which trivially belongs to Π /u1D70B ℓ , Cover ( /u1D436 ℓ ) ), we also have

<!-- formula-not-decoded -->

Therefore,

Since /u1D70B ℓ is Δ ℓ -optimal on Cover ( /u1D436 ℓ ) by assumption, this makes /u1D70B + ℓ Δ -optimal on Cover ( /u1D436 ℓ ) for

<!-- formula-not-decoded -->

For a trajectory in the MDP, let the random variable /u1D70F be the first time the state is in Cover ( /u1D436 ℓ ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since /u1D70B + ℓ agrees with /u1D70B ★ on states not in Cover ( /u1D436 ℓ ) , the distribution of the trajectory up to and including /u1D446 /u1D70F is the same under both policies, starting from any state /u1D460 ∈ S . Therefore, for any /u1D460 ∈ S , as /u1D6FE /u1D70F ≤ 1 and /u1D70B + ℓ is Δ -optimal on Cover ( /u1D436 ℓ ) . That is, /u1D70B + ℓ is also Δ -optimal on S (with Δ defined in Eq. 21). Using this, for any /u1D460 ∈ Cover ( /u1D436 ℓ ) , and /u1D44E ∈ A , we have

where the third inequality holds by Eq. (20). Therefore /u1D70B ℓ is next-state ( Δ ℓ + 4 ( /u1D714 + /u1D700 )( √ ˜ /u1D451 + 1 )/ /u1D6FE )) -optimal on Cover ( /u1D436 ℓ ) . /square

<!-- formula-not-decoded -->

## E Poof of Lemma 4.4

Proof of Lemma 4.4. We prove by induction on the iterations of the main loop of CAPI-QPI-PLAN the inductive hypothesis: at the start of iteration /u1D456 , for all /u1D459 ∈ [ /u1D43B + 1 ] , /u1D70B /u1D459 is Δ /u1D459 -optimal on Cover ( /u1D436 /u1D459 ) . We first observe that after initialization, /u1D436 /u1D459 is the empty sequence for every /u1D459 , so we can apply Lemma 4.3 with /u1D45E ★ and empty sequences ( /u1D45B = 0 ) to get that for any /u1D460 ∈ Cover (()) and /u1D44E ∈ A , /u1D45E ★ ( /u1D460 , /u1D44E ) ≤ /u1D700 + √ /u1D706 /u1D435 = /u1D700 + /u1D714 . Then, /u1D463 ★ ( /u1D460 ) ≤ /u1D700 + /u1D714 ≤ Δ /u1D459 . Therefore, at initialization, any policy is Δ /u1D459 -optimal on Cover ( /u1D436 /u1D459 ) for any /u1D459 ∈ [ /u1D43B + 1 ] .

Assuming that the inductive hypothesis holds at the start of some iteration, it is left to prove that it continues to hold at the end of the iteration (assuming Eq. (6) holds whenever MEASURE returns success ); this implies that the hypothesis also holds at the start of the next iteration and hence also proves the lemma. For any ( /u1D460 , /u1D44E ) appended to /u1D436 0 , the inductive hypothesis trivially continues to hold as Δ 0 = 1 /( 1 -/u1D6FE ) ≥ /u1D463 ★ ( /u1D460 ) for any /u1D460 ∈ S because the rewards are bounded in [ 0 , 1 ] . The only other case in which /u1D436 /u1D459 or /u1D70B /u1D459 changes for any /u1D459 is in Lines 18 and 20, where the changes happen only for /u1D459 = ℓ + 1 .

We will use Lemma 3.4 to analyze the effect of these updates, thus next we show that the conditions of the lemma are satisfied:

(a) In Lemma D.2 we show that /u1D70B ℓ is next-state ( Δ ℓ + 4 ( /u1D714 + /u1D700 )( √ ˜ /u1D451 + 1 )/ /u1D6FE ) -optimal on Cover ( /u1D436 ℓ ) . In the proof of the lemma, we introduce a policy in Eq. (22) that acts as /u1D70B ℓ on states in Cover ( /u1D436 ℓ ) ,

and as an optimal stationary deterministic memoryless policy /u1D70B ★ otherwise:

<!-- formula-not-decoded -->

Intuitively, this policy corrects /u1D70B ℓ on the low-confidence states. The proof of Lemma D.2 then uses the fact that this policy is also /u1D45E /u1D70B -realizable (Definition 1.1) and satisfies /u1D70B + ℓ ∈ Π /u1D70B ℓ , Cover ( /u1D436 ℓ ) to show (i) that the /u1D45E -values of /u1D70B ℓ and /u1D70B + ℓ are close on the measured state-action pairs (via Lemma 4.6 and Lemma D.1); (ii) an optimality guarantee on /u1D70B + ℓ for all /u1D460 ∈ S ; and, as a consequence, (iii) the next-state optimality of /u1D70B ℓ .

(b) Next, to analyze the effect of Line 18, we introduce hypothetical /u1D45E -approximators ˜ /u1D45E /u1D459 for /u1D459 ∈ [ /u1D43B + 1 ] , defined as follows: At initialization, ˜ /u1D45E /u1D459 ( /u1D460 , /u1D44E ) = 0 for all /u1D459 ∈ [ /u1D43B + 1 ] , /u1D460 ∈ S , and /u1D44E ∈ A . It is updated every time after Line 16 of the algorithm is executed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In other words, ˜ /u1D45E ℓ is only updated to the newly computed ˆ /u1D45E for states that are not in Cover ( /u1D436 ℓ + 1 ) , and stays unchanged for other states. We show in Lemma F.2 that the new policy that /u1D70B ℓ + 1 is updated to, which is constructed in two steps (Lines 17-18), can be expressed as the result of a single CAPI policy update that uses ˜ /u1D45E :

<!-- formula-not-decoded -->

/u1D70B ℓ + 1 ← /u1D70B ˜ /u1D45E ℓ , /u1D70B ℓ , S\ Cover ( /u1D436 /u1D459 ) . We show in Lemma F.1 that ˜ /u1D45E ℓ ≈ /u1D714 ′ /u1D45E /u1D70B ℓ with /u1D714 ′ = ( /u1D714 + /u1D700 )( √ ˜ /u1D451 + 1 ) on Cover ( /u1D436 ℓ ) .

By the above, we can apply Lemma 3.4 with policy /u1D70B ℓ , /u1D45E -approximation ˜ /u1D45E ℓ (with approximation error guarantee /u1D714 ′ on Cover ( /u1D436 ℓ ) , and S fix = S \ Cover ( /u1D436 ℓ ) to get that the new value of /u1D70B ℓ + 1 is a Δ ℓ + 1 = ( 8 ( /u1D714 + /u1D700 )( √ ˜ /u1D451 + 1 ) + /u1D6FE Δ ℓ ) -optimal policy on Cover ( /u1D436 ℓ ) . By the end of the loop in Line 20, Cover ( /u1D436 ℓ + 1 ) = Cover ( /u1D436 ℓ ) , so /u1D70B ℓ + 1 is Δ ℓ + 1 -optimal on Cover ( /u1D436 ℓ + 1 ) . This finishes the proof that the inductive hypothesis continues to hold at the end of the iteration, finishing the proof of the lemma. /square

## F Auxiliary results for Lemma 4.4 about ˜ /u1D45E /u1D459

Throughout the execution of CAPI-QPI-PLAN, for /u1D459 ∈ [ /u1D43B + 1 ] , let ˜ /u1D45E -/u1D459 , /u1D70B -/u1D459 , /u1D436 -/u1D459 denote the values of variables ˜ /u1D45E ℓ , /u1D70B ℓ , /u1D436 ℓ , respectively, at the time when Lines 16-20 were most recently executed with ℓ = /u1D459 in a previous iteration of the main loop of CAPI-QPI-PLAN. If such a time does not exist, let their values be the initialization values. Thus, /u1D436 -/u1D459 may (only) change at the start of some iteration /u1D456 if Lines 16-20 were executed with ℓ = /u1D459 in the previous iteration /u1D456 -1 . Observe that whenever this happens, Lines 16-20 may also change /u1D436 ℓ + 1 in iteration /u1D456 -1 , and this is the only time /u1D436 /u1D459 + 1 can be changed for any /u1D459 ∈ [ /u1D43B ] . After this, at the beginning of iteration /u1D456 , /u1D436 /u1D459 + 1 always has the same elements as /u1D436 -/u1D459 . Therefore, since it also holds at the initialization of the algorithm, we conclude that at the start of each iteration,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma F.1. Assume that Eq. (6) holds whenever MEASURE returns success . Then, whenever Line 18 of CAPI-QPI-PLAN is executed, for all /u1D460 ∈ Cover ( /u1D436 ℓ ) and /u1D44E ∈ A ,

Proof. We prove this by induction for every time Line 18 is executed with any value of ℓ . We first observe that after initialization, /u1D436 /u1D459 is the empty sequence for every /u1D459 , so we can apply Lemma 4.3 with /u1D45E ★ and empty sequences ( /u1D45B = 0 ) to get that for any /u1D460 ∈ Cover (()) and /u1D44E ∈ A , /u1D45E /u1D70B ′′ ( /u1D460 , /u1D44E ) ≤ /u1D45E ★ ( /u1D460 , /u1D44E ) ≤ /u1D700 + √ /u1D706 /u1D435 = /u1D700 + /u1D714 . Also, ˜ /u1D45E /u1D459 (· , ·) = 0 at initialization, so Eq. (25) holds for any value of ℓ .

Consider a time when Line 18 is executed and assume the inductive hypothesis holds for the previous time Line 18 was executed with the same value of ℓ (or at the initialization if this is the first time), that is,

<!-- formula-not-decoded -->

To prove that the statement now holds for any /u1D460 ∈ Cover ( /u1D436 ℓ ) , first consider any /u1D460 ∈ Cover ( /u1D436 ℓ + 1 ) = Cover ( /u1D436 -ℓ ) . For such an /u1D460 , by Lemma 4.5 we have that Π /u1D70B ℓ , Cover ( /u1D436 ℓ ) ⊆ Π /u1D70B -ℓ , Cover ( /u1D436 -ℓ ) . Also, by definition, ˜ /u1D45E ℓ ( /u1D460 , ·) = ˜ /u1D45E -ℓ ( /u1D460 , ·) for /u1D460 ∈ Cover ( /u1D436 ℓ + 1 ) . Combining with the inductive hypothesis, it follows that Eq. (25) holds for /u1D460 ∈ Cover ( /u1D436 ℓ + 1 ) .

It remains to show that Eq. (25) also holds for /u1D460 ∈ Cover ( /u1D436 ℓ ) \ Cover ( /u1D436 ℓ + 1 ) . For such an /u1D460 , ˜ /u1D45E ℓ ( /u1D460 , ·) = ˆ /u1D45E ( /u1D460 , ·) by definition, and hence Lemma D.1 implies that Eq. (25) holds in this case.

Combining the two cases, it follows that the inductive hypothesis continues to hold when Line 18 is executed. /square

Lemma F.2. Throughout the execution of CAPI-QPI-PLAN , at the start of any iteration, for all /u1D459 ∈ [ /u1D43B ] ,

<!-- formula-not-decoded -->

Proof. We prove this by induction for the start of any iteration. Eq. (26) holds at the start of the algorithm due to its initialization (because at initialiaztion, ˜ /u1D45E -/u1D459 ( /u1D460 , /u1D44E ) = 0 for all /u1D460 , /u1D44E , and hence by our tie-breaking rule, the policy on the right-hand side of Eq. (26) always chooses action A 1 , which is the initial policy for /u1D70B /u1D459 ).

In what follows, we use the fact that for any /u1D45E : S × A → R , policy /u1D70B , and S fix ⊆ S , the CAPI policy update /u1D70B /u1D45E , /u1D70B, S fix is a policy whose value at any /u1D460 ∈ S only depends on /u1D45E ( /u1D460 , ·) , /u1D70B ( /u1D460 ) , and whether or not /u1D460 ∈ S fix, by definition (Eq. 5). Therefore, for an alternative /u1D45E ′ , /u1D70B ′ , S ′ fix , for any /u1D460 ∈ S , /u1D70B /u1D45E , /u1D70B, S fix ( /u1D460 ) = /u1D70B /u1D45E ′ , /u1D70B ′ , S ′ fix ( /u1D460 ) whenever the following three conditions hold: (C1) /u1D45E ( /u1D460 , /u1D44E ) = /u1D45E ′ ( /u1D460 , /u1D44E ) for all /u1D44E ∈ A ; (C2) /u1D70B ( /u1D460 ) = /u1D70B ′ ( /u1D460 ) ; and (C3) either both or none of S fix and S ′ fix include /u1D460 .

Assume the inductive hypothesis holds at the beginning of some iteration. Let /u1D70B ′′ be the policy Line 18 updates /u1D70B ℓ + 1 to, noting that this is the only place where policies are updated. All we need to prove is that /u1D70B ′′ is equal to

<!-- formula-not-decoded -->

First, for any /u1D460 ∉ Cover ( /u1D436 ℓ + 1 ) , /u1D70B ′′ ( /u1D460 ) = /u1D70B ′ ( /u1D460 ) = /u1D70B ˆ /u1D45E , /u1D70B ℓ , S\ Cover ( /u1D436 ℓ ) ( /u1D460 ) and ˆ /u1D45E ( /u1D460 , ·) = ˜ /u1D45E ℓ ( /u1D460 , ·) by definition. Hence, /u1D70B ′′ ( /u1D460 ) = /u1D70B ˆ /u1D45E , /u1D70B ℓ , S\ Cover ( /u1D436 ℓ ) ( /u1D460 ) = /u1D70B ˜ /u1D45E ℓ , /u1D70B ℓ , S\ Cover ( /u1D436 ℓ ) ( /u1D460 ) = ˜ /u1D70B ( /u1D460 ) , as all of conditions (C1)-(C3) are satisfied for /u1D460 (C2 and C3 hold trivially).

Next, take any /u1D460 ∈ Cover ( /u1D436 ℓ + 1 ) = Cover ( /u1D436 -ℓ ) . Then, by Line 18, /u1D70B ′′ ( /u1D460 ) = /u1D70B ℓ + 1 ( /u1D460 ) . By the inductive hypothesis, the current value of /u1D70B ℓ + 1 can be written as /u1D70B ˜ /u1D45E -ℓ , /u1D70B -ℓ , S\ Cover ( /u1D436 -ℓ ) . We prove that this policy takes the same value as ˜ /u1D70B at /u1D460 , by showing conditions (C1)-(C3). First, by Lemma 4.5, /u1D70B ℓ ∈ Π /u1D70B -ℓ , Cover ( /u1D436 -ℓ ) . Thus, as /u1D460 ∈ Cover ( /u1D436 -ℓ ) , /u1D70B ℓ ( /u1D460 ) = /u1D70B -ℓ ( /u1D460 ) , showing condition (C2). Furthermore, as /u1D460 ∈ Cover ( /u1D436 ℓ + 1 ) , by definition, ˜ /u1D45E ℓ ( /u1D460 , ·) = ˜ /u1D45E -ℓ ( /u1D460 , ·) , showing condition (C1). Finally, as /u1D460 ∈ Cover ( /u1D436 ℓ + 1 ) = Cover ( /u1D436 -ℓ ) ⊆ Cover ( /u1D436 ℓ ) , /u1D460 ∉ S \ Cover ( /u1D436 -ℓ ) and /u1D460 ∉ S \ Cover ( /u1D436 ℓ ) , showing condition (C3).

Combining the two cases, /u1D70B ′′ ( /u1D460 ) = ˜ /u1D70B ( /u1D460 ) for any /u1D460 ∈ S , finishing the induction.

/square

## G Efficient implementation and proof of Theorem 1.3

In this section we consider the efficient implementation of CAPI-QPI-PLAN in terms of memory and computational costs of both the algorithm itself and the final policy it outputs.

Focusing on the memory cost, first we can observe that throughout the execution of the algorithm, /u1D436 /u1D459 for all /u1D459 ∈ [ /u1D43B + 1 ] only stores up to ˜ /u1D451 unique state-action pairs altogether (cf. Eq. (13)), as they use the same pairs; let /u1D44A = ( /u1D460 /u1D456 , /u1D44E /u1D456 ) /u1D456 ∈ ˆ /u1D451 denote these for some ˆ /u1D451 ≤ ˜ /u1D451 . Furthermore, throughout the execution of the algorithm, for any level /u1D459 , the only features that /u1D70B /u1D459 depends on are the features associated with members of /u1D44A . Storing all these features takes /u1D451 ˆ /u1D451 memory. Denote all the policies that CAPI-QPI-PLAN constructs in Line 18, in order, as /u1D70B ( 0 ) , /u1D70B ( 1 ) , . . . , /u1D70B ( /u1D45B -1 ) , where /u1D45B is the number of times Line 18 is executed. Recall from the proof of Theorem 1.2 that the number of times MEASURE returns success , which is an upper bounds on /u1D45B , is itself bounded by ˜ /u1D451 /u1D43B , hence /u1D45B ≤ ˜ /u1D451 /u1D43B . Together, Lines 17-18 construct a policy that, for an /u1D460 ∈ S , decides whether the action should be arg max /u1D44E ∈A 〈 /u1D711 ( /u1D460 , /u1D44E ) , /u1D703 〉 for some /u1D703 given by LSE (Eq. (8)), or the value of the policy should be determined by a recursive call to a previously constructed policy, either /u1D70B ℓ + 1 or /u1D70B ℓ (through /u1D70B ′ ). Now

there exist some /u1D44E , /u1D44F ∈ [ /u1D45B ] such that /u1D70B ( /u1D44E ) = /u1D70B ℓ and /u1D70B ( /u1D44F ) = /u1D70B ℓ + 1 before the new policy is constructed in Line 18. To implement the new /u1D70B ℓ + 1 constructed policy, it is enough therefore to store, in addition to the existing policies, /u1D703 (from ˆ /u1D45E ), the decision rules, and the indices /u1D44E and /u1D44F . The decision rules are fully defined by /u1D703 , /u1D436 ℓ , and /u1D436 ℓ + 1 . It is therefore enough to further store /u1D436 ℓ , /u1D436 ℓ + 1 ⊆ /u1D44A , which can be encoded as ˆ /u1D451 -dimensional vectors each, storing the bitmask of which state-action pairs are included. We also store the current value of ℓ (the level) for the newly constructed policy. Together, a policy thus consumes 3 + /u1D451 + 2 ˆ /u1D451 memory. We store all policies constructed, along with the features of /u1D44A , and the final value of /u1D449 ( /u1D436 /u1D43B ) -1 , at a memory cost of /u1D451 ˆ /u1D451 + ˜ /u1D451 /u1D43B ( 3 + /u1D451 + ˆ /u1D451 ) + /u1D451 2 = ˜ O( /u1D451 2 /( 1 -/u1D6FE )) . This is the memory cost of the final policy outputted by CAPI-QPI-PLAN. The memory cost of running CAPI-QPI-PLAN itself is of the same order, as additionally storing /u1D436 /u1D459 , ¯ /u1D45E /u1D459 , and /u1D449 ( /u1D436 /u1D459 ) -1 for /u1D459 ∈ [ /u1D43B + 1 ] takes ˜ O( /u1D451 2 /( 1 -/u1D6FE )) memory.

To efficiently implement the final policy found by CAPI-QPI-PLAN with the stored information described above, we start from evaluating the last policy constructed, /u1D70B ( /u1D456 ) for /u1D456 = /u1D45B -1 . We introduce auxiliary variables ˜ /u1D449 ( /u1D436 /u1D459 ) -1 and ˜ /u1D436 /u1D459 for /u1D459 ∈ [ /u1D43B + 1 ] to efficiently track the required values of /u1D449 ( /u1D436 /u1D459 ) -1 and /u1D436 /u1D459 . We keep updating these variables so that for /u1D459 ∈ { ℓ, ℓ + 1 } , they match the values of /u1D449 ( /u1D436 /u1D459 ) -1 and /u1D436 /u1D459 , respectively, at the time of construction of the current policy /u1D70B ( /u1D456 ) under consideration, where ℓ is the (saved) level of /u1D70B ( /u1D456 ) . For /u1D456 = /u1D45B -1 , observe that when it was constructed, /u1D436 0 = /u1D436 1 = · · · = /u1D436 /u1D43B by Lemma 4.7. We therefore start by initializing variables ˜ /u1D449 ( /u1D436 0 ) -1 , . . . , ˜ /u1D449 ( /u1D436 /u1D43B ) -1 to the saved final value of /u1D449 ( /u1D436 /u1D43B ) -1 , and variables ˜ /u1D436 0 , . . . , ˜ /u1D436 /u1D43B to /u1D44A . Implementing the decisions of a policy takes an order of |A| /u1D451 2 computation ( |A| vector and matrix multiplications), after which we recover either the policy output or a previously constructed policy to recurse into. For the latter case, we have to consider the evaluation of this policy, denoted by /u1D70B ( /u1D456 ′ ) . Let the (saved) level of /u1D70B ( /u1D456 ′ ) be ℓ ′ . Before we set /u1D456 to /u1D456 ′ and start evaluating it, we need to update the values of ˜ /u1D449 ( /u1D436 /u1D459 ) and /u1D436 /u1D459 for /u1D459 ∈ { ℓ ′ , ℓ ′ + 1 } . The updates are needed for these two levels only, as the decision rule of policy /u1D456 ′ only depends on these levels, as shown before. Let us describe the update procedure for some /u1D459 ∈ { ℓ ′ , ℓ ′ + 1 } : Since /u1D70B ( /u1D456 ′ ) was constructed earlier than /u1D70B ( /u1D456 ) (i.e., /u1D456 ′ &lt; /u1D456 ), and /u1D436 /u1D459 ′ can only grow during the algorithm for any /u1D459 ′ ∈ [ /u1D43B + 1 ] , we only need to remove members of the variable ˜ /u1D436 /u1D459 to match the value of /u1D436 /u1D459 at the time of construction of /u1D70B ( /u1D456 ′ ) . The members to be removed are given by the difference of the members of ˜ /u1D436 /u1D459 and the bitmasks stored for /u1D70B ( /u1D456 ′ ) for level /u1D459 . For each state-action pair ( /u1D460 , /u1D44E ) removed, we also need to update ˜ /u1D449 ( /u1D436 /u1D459 ) -1 to ( ˜ /u1D449 ( /u1D436 /u1D459 ) -/u1D711 ( /u1D460 , /u1D44E ) /u1D711 ( /u1D460 , /u1D44E ) /latticetop ) -1 , which can be done in order /u1D451 2 computation using the Sherman-Morrison-Woodbury formula [Max, 1950]. The total number of such removal operations for any level /u1D459 is bounded by the sum of the number of state-action pairs in the initialization of ˜ /u1D436 /u1D459 ′ (for /u1D459 ′ ∈ [ /u1D43B + 1 ] ), that is, by ( /u1D43B + 1 ) ˆ /u1D451 . As a result, the computational cost of the final policy of CAPI-QPI-PLAN is ˜ O(( /u1D43B + 1 ) ˆ /u1D451 /u1D451 2 ) + /u1D45B ˜ O(|A| /u1D451 2 ) = ˜ O( /u1D451 3 |A|/( 1 -/u1D6FE )) .

Finally, we consider the computational cost of running CAPI-QPI-PLAN. The number of iterations of the outer loop is bounded by ˜ O( /u1D451/u1D43B ) = ˜ O( /u1D451 /( 1 -/u1D6FE )) , as each iteration involves either a MEASURE call that returns success , or a new member added to some /u1D436 /u1D459 . For each iteration, Line 4 takes ˜ O( /u1D451 2 |A|) , Line 7 takes ˜ O( /u1D451 /( 1 -/u1D6FE )) , Line 11 takes ˜ O( /u1D451 2 |A|) computation; for Line 16, calculating /u1D703 , the second component of the inner product of the least-squares predictor in Eq. (8) takes ˜ O( /u1D451 2 ) computation, and if /u1D436 /u1D459 ever changes for some /u1D459 , updating /u1D449 ( /u1D436 /u1D459 ) -1 by the Sherman-Morrison-Woodbury takes ˜ O( /u1D451 2 ) computation. Overall, all the operations except those associated to the MEASURE call of Line 9 take ˜ O( /u1D451 3 |A|/( 1 -/u1D6FE )) computation in total. We conclude our calculations by considering the computational cost of the MEASURE calls, which will dominate the overall computational cost. Line 6 of Algorithm 2 has a computational cost of order /u1D451 2 |A| , while the majority of the computational cost comes from evaluating the policy at Line 7. By our previous calculations, this takes ˜ O( /u1D451 3 |A|/( 1 -/u1D6FE )) computation and happens (at most) once for each simulator call. Using the query cost bound of Theorem 1.2, we conclude that the computational cost of CAPI-QPI-PLAN is ˜ O( /u1D451 4 |A|( 1 -/u1D6FE ) -5 /u1D714 -2 ) . /square

## H Query cost lower bounds with random access

In this section we prove lower bounds on the worst-case expected query cost of planning algorithms with a simulator supporting random access . Recall from Section 1 that in this setting a planner can issue queries for any state-action pair, not just the ones already visited. As this is a more powerful access to the simulator than local access , statements that hold for all planners using random

access (as such, all lower bounds presented in this section) trivially hold for planners using local access . We prove two bounds, Theorem H.2 and Theorem H.3, whose combination trivially implies Theorem 1.4.

Formally, the planner interacts with a random access simulator that simulates some MDP /u1D440 as follows: at step /u1D461 starting from 1 , given the whole interaction history /u1D43B /u1D461 = ( /u1D446 1 , /u1D434 1 , /u1D445 1 , /u1D446 ′ 1 , . . . , /u1D446 /u1D461 -1 , /u1D434 /u1D461 -1 , /u1D445 /u1D461 -1 , /u1D446 ′ /u1D461 -1 ) (where /u1D43B 1 is the empty sequence by definition), the planner either selects a state-action pair ( /u1D446 /u1D461 , /u1D434 /u1D461 ) , or halts and outputs a stationary memoryless policy. The planner is allowed to randomize. Let /u1D70F denote the number of queries the planner sends to the simulator before it halts, and /u1D70B /u1D70F the policy it outputs.If the planner does not stop, the simulator responds to the query ( /u1D446 /u1D461 , /u1D434 /u1D461 ) by returning ( /u1D446 ′ /u1D461 , /u1D445 /u1D461 ) sampled independently from the transition-reward kernel Q( /u1D446 /u1D461 , /u1D434 /u1D461 ) of /u1D440 . Let P /u1D440 denote the probability measure associated with this procedure, and let E /u1D440 denote the expectation operator corresponding to P /u1D440 . Both P /u1D440 and E /u1D440 implicitly depend on the planner, which is omitted in the notation for brevity but will always be clear from the context. Using this notation, clearly E /u1D440 ( /u1D70F ) is the expected query cost of the planner on /u1D440 .

As usual, we only consider the query complexity of planners which are reasonable in the sense that they can find a near-optimal policies for a class of MDPs:

Definition H.1 (Soundness and query complexity) . A planner is said to be ( /u1D6FC, /u1D6FF ) -sound for an MDP /u1D440 if, when used with a simulator of /u1D440 , it halts almost surely (i.e., P /u1D440 ( /u1D70F &lt; ∞) = 1 ) and outputs a policy /u1D70B /u1D70F that is /u1D6FC -optimal for /u1D440 with probability at least 1 -/u1D6FF , that is, where /u1D463 ★ and /u1D463 /u1D70B /u1D70F are the value-functions of the optimal policy and /u1D70B /u1D70F in the MDP /u1D440 and /u1D460 0 is the initial state of /u1D440 . A planner is ( /u1D6FC, /u1D6FF ) -sound for a class of MDPs M if it is ( /u1D6FC, /u1D6FF ) -sound for every MDP in the class. The query complexity of a planner over M is defined as the maximum of its expected query cost over the members of the class.

<!-- formula-not-decoded -->

In the rest of the section, for /u1D451 ≥ 1 and /u1D43F &gt; 0 , we use B /u1D451 ( /u1D43F ) = { /u1D465 ∈ R /u1D451 : ‖ /u1D465 ‖ ≤ /u1D43F } to denote the /u1D451 -dimensional Euclidean ball of radius /u1D43F centered at the origin.

## H.1 Exponential lower bound for planners with small suboptimality

We first show an exponential query complexity lower bound for sound planners that guarantee a small suboptimality bound. The result is a simple application of the techniques in Lattimore et al. [2020], and establishes the barrier for the suboptimality attainable by query-efficient planners:

Theorem H.2. Let /u1D6FF ≤ 0 . 9 , /u1D6FC ≤ 0 . 49 /( 1 -/u1D6FE ) , and /u1D700 ≥ 0 , /u1D451 ≥ 3 . There is a class of MDPs M with uniform policy value-function approximation error /u1D700 for some /u1D451 -dimensional feature map such that the query complexity of any ( /u1D6FC, /u1D6FF ) -sound planner over M is at least exp ( Ω ( /u1D451 ( /u1D700 /u1D6FC ( 1 -/u1D6FE ) ) 2 ) ) .

˜

Proof. Our proof is based on a similar complexity lower bound of Lattimore et al. [2020] for the multi-armed bandit setting, which is a special case of our problem. As such, we start by rewriting the class of bandit problems they used in their proof in our MDP framework, introducing a set of MDPs ˜ M each of which gets into a terminal state with no rewards after the first step. Let /u1D6FC ′ = 2 . 01 /u1D6FC ( 1 -/u1D6FE ) ≤ 1 and /u1D458 = ⌊ exp ( /u1D451 -2 8 ( /u1D700 /u1D6FC ′ ) 2 )⌋ . ˜ M = { } as follows: Each MDP in ˜ M has /u1D458 actions (i.e., A = [ /u1D458 ] ) and two states: S = ( /u1D460 0 , /u1D460 1 ) with /u1D460 0 being the initial state, and deterministic transitions /u1D443 ( /u1D460 1 | /u1D460 , /u1D44E ) = 1 and /u1D443 ( /u1D460 0 | /u1D460 , /u1D44E ) = 0 for all ( /u1D460 , /u1D44E ) ∈ S×A . For any /u1D456 ∈ [ /u1D458 ] , the reward distribution R /u1D456 for MDP ˜ /u1D440 /u1D456 is defined as follows: rewards for state /u1D460 1 are deterministically zero, that is, R /u1D456 ( 0 | /u1D460 1 , /u1D44E ) = 1 for all /u1D44E ∈ A , making /u1D460 1 an absorbing state with zero reward, while rewards for state /u1D460 0 are deterministically /u1D6FC ′ for action /u1D456 and zero otherwise, that is, R /u1D456 ( /u1D6FC ′ | /u1D460 0 , /u1D456 ) = 1 and R /u1D456 ( 0 | /u1D460 0 , /u1D457 ) = 1 for /u1D457 ∈ [A] with /u1D457 ≠ /u1D456 . Since this class of MDPs is equivalent to the class of muti-armed bandit problems defined by Lattimore et al. [2020], their proof of Corollary 3.3 implies that

/u1D440

1

, . . . ,

˜

/u1D440

/u1D458

is defined to be a set of

/u1D458

MDPs

- there exists a feature map ˜ /u1D711 : S × A → B /u1D451 -1 ( 1 ) such that /u1D700 is the maximum uniform policy value-function approximation error (Definition 1.1) over ˜ M equipped with features ˜ /u1D711 ; and

- any planner that almost surely outputs an /u1D6FC ′ -optimal deterministic policy for all ˜ /u1D440 ∈ ˜ M (when run with a random access simulator for ˜ /u1D440 ) executes at least

queries in expectation.

<!-- formula-not-decoded -->

We construct a new set M = { /u1D440 1 , . . . , /u1D440 /u1D458 } of /u1D458 MDPs where for each /u1D456 ∈ [ /u1D458 ] , /u1D440 /u1D456 is a slight modification of ˜ /u1D440 /u1D456 , always returning to the initial state /u1D460 0 instead of stopping after the first step: as such, the only modification is that the transition probabilities for all /u1D440 ∈ M are /u1D443 ( /u1D460 0 | /u1D460 , /u1D44E ) = 1 and /u1D443 ( /u1D460 1 | /u1D460 , /u1D44E ) = 0 for all ( /u1D460 , /u1D44E ) ∈ S × A . Let /u1D711 : S × A → B /u1D451 ( 2 ) be the features for all MDPs in M , where for all ( /u1D460 , /u1D44E ) ∈ S × A , /u1D711 ( /u1D460 , /u1D44E ) is a concatenation of the ( /u1D451 -1 ) -dimensional ˜ /u1D711 ( /u1D460 , /u1D44E ) and the scalar 1 , so that the /u1D451 th coordinate of /u1D711 ( /u1D460 , /u1D44E ) is /u1D711 ( /u1D460 , /u1D44E ) /u1D451 = 1 .

<!-- formula-not-decoded -->

Fix any /u1D456 ∈ [ /u1D458 ] and any stationary deterministic memoryless policy /u1D70B , and let ˜ /u1D703 be the parameter realizing the low approximation error for ˜ /u1D440 /u1D456 and ˜ /u1D711 , that is, satisfying Eq. (19) (see Appendix C for a proof that such a ˜ /u1D711 exists). In what follows, we denote /u1D45E - and /u1D463 -functions (with arbitrary superscripts) of an MDP /u1D440 by adding /u1D440 as a superscript to the corresponding function. Let /u1D703 be a concatenation of ˜ /u1D703 and the scalar /u1D6FE /u1D463 /u1D70B /u1D440 /u1D456 ( /u1D460 0 ) . For any ( /u1D460 , /u1D44E ) ∈ S × A ,

The uniform policy value-function approximation error therefore remains at most /u1D700 for /u1D440 /u1D456 with feature map /u1D711 , and this is true for any /u1D456 ∈ [ /u1D458 ] . We can therefore take any ( /u1D6FC, /u1D6FF ) -sound planner with query complexity /u1D447 (for some /u1D447 ≥ 0 ) over M , and provide it with a simulator of /u1D440 /u1D456 for any /u1D456 ∈ [ /u1D458 ] (which we can trivially build with access to a simulator of ˜ /u1D440 /u1D456 ), to get a policy /u1D70B that is /u1D6FC -optimal for /u1D440 /u1D456 with P /u1D440 /u1D456 -probability at least 1 -/u1D6FF . Recall that the rewards of /u1D440 /u1D456 are 0 for every action apart from a single optimal action, /u1D456 , where the reward is /u1D6FC ′ . Thus, /u1D463 ★ /u1D440 /u1D456 ( /u1D460 0 ) = /u1D6FC ′ /( 1 -/u1D6FE ) and /u1D463 /u1D70B /u1D440 /u1D456 ( /u1D460 0 ) = /u1D6FC ′ /u1D70B ( /u1D456 | /u1D460 0 )/( 1 -/u1D6FE ) = /u1D70B ( /u1D456 | /u1D460 0 ) /u1D463 ★ /u1D440 /u1D456 ( /u1D460 0 ) . Thus, with probability at least 1 -/u1D6FF , /u1D463 ★ /u1D440 /u1D456 ( /u1D460 0 ) -/u1D463 /u1D70B /u1D440 /u1D456 ( /u1D460 0 ) ≤ /u1D6FC &lt; 0 . 5 /u1D6FC ′ /( 1 -/u1D6FE ) = 0 . 5 /u1D463 ★ /u1D440 /u1D456 ( /u1D460 0 ) . Therefore, /u1D70B ( /u1D456 | /u1D460 0 ) &gt; 0 . 5 . As we know that the optimal action achieves a deterministic reward of /u1D6FC ′ , we can test with a single query whether the action that /u1D70B assigns the highest probability to is optimal. If not, we can run the planner again and repeat the check. Since each run of the planner is successful with probability at least 1 -/u1D6FF , independently of each other, almost surely one of the checks eventually passes and we output the deterministic policy that chooses the optimal action. Now the number of times the planner needs to be run is a stopping time (with respect to the sequence of the runs) with expectation at most 1 /( 1 -/u1D6FF ) , hence the expected query cost of the whole procedure is at most ( /u1D447 + 1 )/( 1 -/u1D6FF ) by Wald's equation. Note that the same policy is /u1D6FC ′ -optimal for ˜ /u1D440 /u1D456 . Therefore, the planner defined above almost surely outputs an /u1D6FC ′ -optimal deterministic policy for any MDP in ˜ M , and hence by Eq. (27) we have

Therefore /u1D447 = exp ( Ω ( /u1D451 ( /u1D700 /u1D6FC ( 1 -/u1D6FE ) ) 2 ) ) , finishing the proof.

<!-- formula-not-decoded -->

## H.2 Lower bound for linear MDPs

We close this section by proving a lower bounds on the query complexity of random access planners for linear MDPs (c.f. Theorem H.3).

We start by recalling the definition of linear MDPs [Zanette et al., 2020]: An MDP with countable state space is said to be linear if there exists a feature map /u1D711 : S × A → B /u1D451 ( /u1D43F ) , a state-transition feature map /u1D713 : S → R /u1D451 , and a reward parameter /u1D703 /u1D45F ∈ B /u1D451 ( /u1D435 ) such that /u1D45F ( /u1D460 , /u1D44E ) = 〈 /u1D711 ( /u1D460 , /u1D44E ) , /u1D703 /u1D45F 〉 and /u1D443 ( /u1D460 ′ | /u1D460 , /u1D44E ) = 〈 /u1D711 ( /u1D460 , /u1D44E ) , /u1D713 ( /u1D460 ′ )〉 for any ( /u1D460 , /u1D44E , /u1D460 ′ ) ∈ S × A × S , and ∑ /u1D460 ∈S ‖ /u1D713 ( /u1D460 )‖ 2 ≤ /u1D435 . Clearly, any linear MDP satisfies Definition 1.1 with /u1D700 = 0 . As such, the lower bounds presented below trivially transfer to the /u1D700 uniform policy value-function approximation error case for any /u1D700 ≥ 0 .

Theorem H.3. Let /u1D6FF ∈ ( 0 , 0 . 08 ] , /u1D6FE ∈ [ 7 12 , 1 ] , /u1D43B = 1 /( 1 -/u1D6FE ) , /u1D6FC ∈ ( 0 , 0 . 05 /u1D6FE /u1D43B /( 1 + /u1D6FE ) 2 ] , and /u1D451 ≥ 3 . Then there is a class of linear MDPs M such that the query complexity of any ( /u1D6FC, /u1D6FF ) -sound planner over M is at least Ω ( /u1D451 2 /u1D43B 3 / /u1D6FC 2 ) .

/square

In the remainder of the section we prove the above bound. Throughout we assume that the conditions in Theorem H.3 are satisfied. We start with the construction of the class M of MDPs, then prove several auxiliary results, before finally presenting the proof of the theorem.

The construction of M is based on a combination of hard tabular MDPs [Xiao et al., 2022] and hard linear bandit problems [Lattimore and Szepesvári, 2020, Section 24.1]. Each MDP in M has two states: S = { /u1D460 0 , /u1D460 1 } with /u1D460 0 being the initial state. The action space is the intersection of a unit sphere and a ( /u1D451 -2 ) -dimensional hypercube: A = {± 1 / √ /u1D451 -2 } /u1D451 -2 . We construct MDPs /u1D440 /u1D6FD for all /u1D6FD ∈ A , and let M = { /u1D440 /u1D6FD | /u1D6FD ∈ A} . The feature map /u1D711 is defined, for any /u1D44E ∈ A , as

<!-- formula-not-decoded -->

We define the linear MDPs /u1D440 /u1D6FD to have deterministic rewards for any /u1D6FD ∈ A . Thus, /u1D440 /u1D6FD is fully defined by its reward parameter /u1D703 /u1D45F and state-transition feature map /u1D713 , according to the definition of linear MDPs. Let /u1D703 /u1D45F = ( 1 , 0 , . . . , 0 ) /latticetop , making state /u1D460 0 the only rewarding state, as then for all /u1D44E ∈ A ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies that

<!-- formula-not-decoded -->

Our assumptions guarantee that /u1D443 /u1D6FD defines a valid transition kernel with probabilities in [ 0 , 1 ] . The MDP starts in /u1D460 0 and rewards are collected until the state transitions to /u1D460 1 , which is a terminal state with zero reward.

For the proof, we also need the following notation and supporting lemmas.

Notation. The probability measure P /u1D440 /u1D6FD induced by the interconnection of a planner and a simulator for /u1D440 /u1D6FD is written for simplicity as P /u1D6FD . Similarly, E /u1D440 /u1D6FD is written as E /u1D6FD . /u1D463 /u1D6FD (with arbitrary superscripts) denotes value functions (corresponding to the superscripts) of /u1D440 /u1D6FD . For any integer /u1D456 ∈ { 1 , . . . , /u1D451 -2 } , err /u1D456 ( /u1D70B , /u1D6FD ) = ∑ /u1D44E ∈A /u1D70B ( /u1D44E | /u1D460 0 ) /u1D43C sgn ( /u1D44E /u1D456 ) ≠ sgn ( /u1D6FD /u1D456 ) denotes the average error of a policy /u1D70B at the /u1D456 th coordinate, where /u1D44E /u1D456 and /u1D6FD /u1D456 are the /u1D456 th components of /u1D44E and /u1D6FD , respectively, and /u1D43C /u1D438 is the indicator function of event /u1D438 . With a slight abuse of notation, for a stationary memoryless policy /u1D70B , we let /u1D70B /latticetop /u1D6FD denote ∑ /u1D44E ∈A /u1D70B ( /u1D44E | /u1D460 0 ) /u1D44E /latticetop /u1D6FD .

Lemma H.4. For any /u1D440 /u1D6FD ∈ M , the value function of a stationary memoryless policy /u1D70B is given by

<!-- formula-not-decoded -->

Proof. It clearly holds that /u1D463 /u1D70B /u1D6FD ( /u1D460 1 ) = 0 . From the Bellman equation, /u1D463 /u1D70B /u1D6FD ( /u1D460 0 ) = 1 + /u1D6FE ( /u1D6FE + Δ /u1D70B /latticetop /u1D6FD ) /u1D463 /u1D70B /u1D6FD ( /u1D460 0 ) , and the claim follows from solving this equation for /u1D463 /u1D70B /u1D6FD ( /u1D460 0 ) . /square

It is easy to see that the optimal policy in /u1D440 /u1D6FD is defined by /u1D70B ★ /u1D6FD ( /u1D6FD | /u1D460 0 ) = 1 (the actions in /u1D460 1 do not matter). Hence, by the above lemma,

<!-- formula-not-decoded -->

Because 1 -/u1D70B /latticetop /u1D6FD = 2 ∑ /u1D451 -2 /u1D456 = 1 err /u1D456 ( /u1D70B , /u1D6FD )/( /u1D451 -2 ) ,

Accordingly, to prove a lower bound on the suboptimality of /u1D70B , we need a lower bound for the sum of errors, ∑ /u1D451 -2 /u1D456 = 1 err /u1D456 ( /u1D70B , /u1D6FD ) . To this end, Lemma H.5 below plays a key role.

<!-- formula-not-decoded -->

Lemma H.5 (Error Probability Lower Bound) . For any planner there exists a /u1D6FD ∈ A such that

<!-- formula-not-decoded -->

To prove Lemma H.5, we need some technical lemmas. First, let F /u1D461 for any /u1D461 ∈ N + denote the /u1D70E -algebra generated by random variables in /u1D43B /u1D461 , with F 1 being the trivial /u1D70E -algebra. F = (F /u1D461 ) ∞ /u1D461 = 1 is chosen to be the filtration. The following lemma is adopted from Exercise 15.7 of Lattimore and Szepesvári [2020] with a slight modification.

Lemma H.6 (KL-divergence decomposition) . Let /u1D440 and /u1D440 ′ be two MDPs differing only in their transition probability kernels, denoted by /u1D443 and /u1D443 ′ , respectively. Then, for any any F -adapted stopping time /u1D70F satisfying P /u1D440 ( /u1D70F &lt; ∞) = 1 , and an F /u1D70F -measurable 1 random variable /u1D44D , where P /u1D44D /u1D440 and P /u1D44D /u1D440 ′ are the laws of /u1D44D under P /u1D440 and P /u1D440 ′ , respectively, N /u1D461 ( /u1D460 , /u1D44E ) denotes the number of queries with ( /u1D460 , /u1D44E ) ∈ S × A up to time step /u1D461 , and KL (· , ·) denotes the Kullback-Leibler (KL-) divergence of two distributions.

<!-- formula-not-decoded -->

The next lemma provides an upper bound on the KL-divergence of certain next-state distributions. A similar result appears in the proof of Lemma 6.8 of Zhou et al. [2020], but it requires that /u1D6FE ≥ 2 / 3 ; ours only requires the weaker assumption that /u1D6FE ≥ 7 / 12 .

<!-- formula-not-decoded -->

Lemma H.7. Take any /u1D6FD, /u1D6FD ′ ∈ A that only differ at a single coordinate. Then for any action /u1D44E ∈ A ,

Proof. Our proof relies on Proposition 2 of Xiao et al. [2022]: for two Bernoulli distributions Ber ( /u1D45D ) and Ber ( /u1D45D ′ ) with parameters /u1D45D , /u1D45D ′ ∈ ( 0 , 1 ) , it holds that

<!-- formula-not-decoded -->

Since /u1D443 /u1D6FD ( /u1D460 1 | /u1D460 0 , /u1D44E ) = 1 -/u1D6FE -Δ /u1D6FD /latticetop /u1D44E and /u1D443 /u1D6FD ′ ( /u1D460 1 | /u1D460 0 , /u1D44E ) = 1 -/u1D6FE -Δ ( /u1D6FD ′ ) /latticetop /u1D44E ,

<!-- formula-not-decoded -->

for any action /u1D44E ∈ A . Note that

<!-- formula-not-decoded -->

where ( /u1D44E ) is due to the fact that /u1D465 ( 1 -/u1D465 ) is monotone decreasing for /u1D465 ≥ 0 . 5 and /u1D6FE + Δ /u1D6FD /latticetop /u1D44F ≥ /u1D6FE -Δ ≥ 0 . 5 since /u1D6FE ≥ 7 / 12 and Δ ≤ 0 . 2 ( 1 -/u1D6FE ) , ( /u1D44F ) follows since 0 . 5 ≤ /u1D6FE + Δ , and ( /u1D450 ) holds because Δ ≤ 0 . 2 ( 1 -/u1D6FE ) . Combining this result with Eq. (31) concludes the proof of the lemma. /square

Now we are ready to prove Lemma H.5.

1 By a slight abuse of notation, F /u1D70F is the /u1D70E -algebra generated by the random vector (with random length) ( /u1D446 1 , /u1D434 1 , /u1D445 1 , /u1D446 ′ 1 , . . . , /u1D446 /u1D70F -1 , /u1D434 /u1D70F -1 , /u1D445 /u1D70F -1 , /u1D446 ′ /u1D70F -1 ) .

Proof of Lemma H.5. Let /u1D6FD ( /u1D456 ) be a vector obtained by flipping the sign of /u1D6FD 's /u1D456 th coordinate. Then, where P err /u1D456 ( /u1D70B /u1D70F ,/u1D6FD ) /u1D6FD , P err /u1D456 ( /u1D70B /u1D70F ,/u1D6FD ) /u1D6FD ( /u1D456 ) ∈ M 1 ([ 0 , 1 ]) are the laws of the random variable err /u1D456 ( /u1D70B /u1D70F , /u1D6FD ) in /u1D440 /u1D6FD and /u1D440 /u1D6FD ( /u1D456 ) , respectively , and the last line follows from an improved Bretagnolle-Huber inequality (inequality (14.11) of Lattimore and Szepesvári [2020]). Applying Lemmas H.6 and H.7 to the KLdivergence in the exponent in the right hand side of the above inequality together with the fact that ∑ ( /u1D460 , /u1D44E ) ∈S×A E /u1D6FD [N /u1D70F ( /u1D460 , /u1D44E )] ≤ E /u1D6FD [ /u1D70F ] , we can further lower-bound the last line by

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

where the first equality holds because for any /u1D6FD , there is exactly one /u1D6FD ( /u1D456 ) in A . As max /u1D6FD ∈A /u1D453 ( /u1D6FD ) ≥ ∑ /u1D6FD ∈A /u1D453 ( /u1D6FD )/|A| for any /u1D453 : A → R , arg max /u1D6FD ∈A ∑ /u1D451 -2 /u1D456 = 1 P /u1D6FD ( err /u1D456 ( /u1D70B /u1D70F , /u1D6FD ( /u1D456 ) ) ≥ 1 / 2 ) satisfies the claim of the lemma. /square

<!-- formula-not-decoded -->

Now we are ready to prove Theorem H.3.

Proof of Theorem H.3. Take any ( /u1D6FC, /u1D6FF ) -sound planner on M . Let err ( /u1D70B , /u1D6FD ) : = ∑ /u1D451 -2 /u1D456 = 1 err /u1D456 ( /u1D70B , /u1D6FD ) for brevity. From Eq. (29),

/parenleftbtA /parenrightbtA where the first inequality holds because /u1D70B /latticetop /u1D6FD ≥ -1 for any stationary memoryless policy /u1D70B , the second inequality is due to the Markov inequality, while the last inequality holds by Lemma H.5. From Eq. (29) and /u1D70B /latticetop /u1D6FD ≤ 1 we also have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality holds because

<!-- formula-not-decoded -->

Combining this result with Eq. (33),

/parenleftbtA /parenrightbtA Note that err ( /u1D70B /u1D70F , /u1D6FD ) &gt; ( /u1D451 -2 )/ 8 implies that /u1D463 ★ /u1D6FD ( /u1D460 0 ) -/u1D463 /u1D70B /u1D70F /u1D6FD ( /u1D460 0 ) &gt; /u1D6FC since similarly to Eq. (32) (i.e., without the expectation)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equality follows because Δ = 4 ( 1 + /u1D6FE ) 2 /u1D6FC /( /u1D6FE /u1D43B 2 ) = 4 ( 1 -/u1D6FE 2 ) 2 /u1D6FC / /u1D6FE . Therefore,

<!-- formula-not-decoded -->

where ( /u1D44E ) follows since Δ ≤ 0 . 2 ( 1 -/u1D6FE ) , and ( /u1D44F ) follows since 0 ≤ 0 . 4 /u1D465 /( 1 + /u1D465 ) ≤ 0 . 2 for /u1D465 ∈ [ 0 , 1 ] . This implies that unless E /u1D6FD [ /u1D70F ] ≥ Ω ( /u1D451 2 /u1D43B 3 / /u1D6FC 2 ) , the algorithm is not ( /u1D6FC, /u1D6FF ) -sound. Indeed if it holds that P /u1D6FD ( /u1D463 ★ /u1D6FD ( /u1D460 0 ) -/u1D463 /u1D70B /u1D70F /u1D6FD ( /u1D460 0 ) &gt; /u1D6FC ) &gt; /u1D6FF , contradicting the assumption that the planner is ( /u1D6FC, /u1D6FF ) -sound on M (the upper bound /u1D6FF ≤ 0 . 08 &lt; 3 / 35 guarantees that the logarithmic term above is bounded by a constant). This concludes the proof. /square

<!-- formula-not-decoded -->