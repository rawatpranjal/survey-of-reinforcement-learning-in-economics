## OPTIMAL NON-ASYMPTOTIC RATES OF VALUE ITERATION FOR AVERAGE-REWARD MDPS

Jongmin Lee Seoul National University Department of Mathematical Sciences dlwhd2000@snu.ac.kr

Ernest K. Ryu UCLA Department of Mathematics eryu@math.ucla.edu

## ABSTRACT

While there is an extensive body of research on the analysis of Value Iteration (VI) for discounted cumulative-reward MDPs, prior work on analyzing VI for (undiscounted) average-reward MDPs has been limited, and most prior results focus on asymptotic rates in terms of Bellman error. In this work, we conduct refined non-asymptotic analyses of average-reward MDPs, obtaining a collection of convergence results that advance our understanding of the setup. Among our new results, most notable are the O (1 /k ) -rates of Anchored Value Iteration on the Bellman error under the multichain setup and the span-based complexity lower bound that matches the O (1 /k ) upper bound up to a constant factor of 8 in the weakly communicating and unichain setups.

## 1 INTRODUCTION

Average-reward Markov decision processes (MDPs) are a fundamental framework for modeling decision-making, where the goal is to maximize long-term, steady-state performance. However, compared to the discounted cumulative-reward counterpart, the average-reward setup is more complex to analyze, and there has been less prior work on it. It is known that while iterates of VI diverge to infinity, the normalized iterates and the Bellman error converge to the optimal average reward under a certain aperiodicity condition. However, despite this understanding of convergence, quantifying the convergence rates of such methods for various classes of average-reward MDPs has been open.

In this work, we conduct refined non-asymptotic analyses of average-reward MDPs, obtaining a collection of convergence results advancing our understanding of the setup. Notably, we establish O (1 /k ) convergence rates of Anchored Value Iteration on the Bellman error under the multichain setup, and we present a span-based complexity lower bound that matches the O (1 /k ) -upper bound up to a constant factor of 8 in the weakly communicating and unichain setups.

Table 1: Summary of our contributions. ( 1 : Federgruen et al. (1978), 2 : Van Der Wal (1981), 3 : Schweitzer &amp; Federgruen (1977), 4 : Schweitzer &amp; Federgruen (1979), 5 : Bertsekas (1998), 'Nonasym' stands for non-asymptotic convergence, 'Asym' for asymptotic convergence, 'multi' for multichain, 'w.c.' for weakly communicating, and 'uni' for unichain.) One check mark indicates a convergence result (upper bound) and two check marks with a strict inequality sign indicate a convergence result accompanied by a complexity lower bound but they do not match. Two check marks with an equal sign indicate a matching complexity lower bound. For multichain MDPs, we present the first non-asymptotic convergence result in Theorem 2. For weakly communicating MDPs, we present the first optimal complexity by matching the non-asymptotic convergence result in Corollary 2 with the complexity lower bound in Theorem 3.

| Prior works   | Non-asym multi MDP   | Asym multi MDP   | Non-asym w.c. MDP   | Asym w.c. MDP   | Non-asym uni MDP   | Asym uni MDP   |
|---------------|----------------------|------------------|---------------------|-----------------|--------------------|----------------|
| [1 , 2]       | ✗                    | ✗                | ✗ ✗ ✗               | ✗ ✓ ✗           | ✓ ✓ ✓              | ✓ ✓ ✓          |
| [3 , 4]       | ✗                    | ✓                |                     |                 |                    |                |
| [5]           | ✗                    | ✗                |                     |                 |                    |                |
| Our work      | ✓ > ✓                | ✓                | ✓ = ✓               | ✓               | ✓ = ✓              | ✓              |

## 1.1 NOTATION AND PRELIMINARIES

We quickly review basic definitions and concepts of average-reward Markov decision processes (MDPs) and reinforcement learning (RL). For further details, refer to standard references such as Puterman (2014); Bertsekas (2012); Sutton &amp; Barto (2018b).

Average-rewardMarkov decision processes. Let M ( X ) be the space of probability distributions over X . Write ( S , A , P, r ) to denote the infinite-horizon undiscounted MDP with finite state space S , finite action space A , transition matrix P : S × A → M ( S ) , and bounded reward r : S × A → R . Denote π : S → M ( A ) for a policy, g π ( s ) = liminf T →∞ 1 T E π [ ∑ T -1 t =0 r ( s t , a t ) | s 0 = s ] for average-reward of a given policy, where E π denotes the expected value over all trajectories ( s 0 , a 0 , s 1 , a 1 , . . . , s T -1 , a T -1 ) induced by P and π . We say g /star is optimal average reward if g /star ( s ) = max π g π ( s ) for all s ∈ S . We say π is an /epsilon1 -optimal policy if ‖ g /star -g π ‖ ∞ ≤ /epsilon1 .

Value Iteration. Let F ( X ) denote the space of bounded measurable real-valued functions over X . With the given undiscounted MDP ( S , A , P, r ) , for V ∈ F ( S ) , define the Bellman consistency operators T π as

<!-- formula-not-decoded -->

for all s ∈ S , and the Bellman optimality operators T as for all s ∈ S . For notational conciseness, we write T π V = r π + P π V , where r π ( s ) = E a ∼ π ( · | s ) [ r ( s, a )] is the reward induced by policy π and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is transition matrix induced by policy π . We define the standard Value Iteration (VI) for the Bellman optimality operator as

<!-- formula-not-decoded -->

where V 0 is an initial point. After executing K iterations, VI returns the near-optimal policy π K as a greedy policy satisfying T π K V K = TV K .

Fixed-point iterations. Given an operator T , classical Banach fixed-point theorem (Banach, 1922) states that if T is contractive, fixed point of T exists and following Picard iteration

<!-- formula-not-decoded -->

converges to the unique fixed point of T . If T is nonexpansive but not contractive such as the rotation operator, Picard iteration may not converge to a fixed point. (For undiscounted MDPs, the Bellman optimality operator is nonexpansive but not necessarily contractive.) In such cases, one may use Kransnosel'ski˘ ı-Mann iteration (Mann, 1953; Krasnosel'ski˘ ı, 1955)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where { λ k } k ∈ N ∈ [0 , 1] , or Halpern iteration (Halpern, 1967)

where x 0 is an initial point and { λ k } k ∈ N ∈ [0 , 1] , to guarantee convergence.

Classification of MDPs. MDPs are classified as follows by the structure of transition matrices. (For definitions of basic concepts of MDPs such as irreducible class, recurrent class, transient states, accessibility, etc., please refer to Puterman (2014, Appendix A.2).)

MDP is unichain if the transition matrix corresponding to every deterministic policy consists of a single irreducible recurrent class plus a possibly empty set of transient states. MDP is weakly communicating , if there exists a closed set of states where each state in that set is accessible from every other state in that set under some determinisitc policy, plus a possibly empty set of states which is transient under every policy. MDP is multichain , if the transition matrix corresponding to any deterministic policy contains one or more irreducible recurrent classes.

MDP is weakly communicating if MDP is unichain, and MDP is multichain if MDP is weakly communicating. Since every MDP is multichain, we use the expressions multichain and general interchangeably.

## 1.2 PRIOR WORKS

Average-reward MDP. The setup of average-reward MDP was first introduced by Howard (1960) in the dynamic programming literature. Blackwell (1962) provided a theoretical framework for analyzing average-reward MDP. Yushkevich (1974); Denardo &amp; Fox (1968) studied modified Bellman equations of multichain MDPs and solutions were characterized by Schweitzer &amp; Federgruen (1978); Schweitzer (1984). In reinforcement learning (RL), average-reward MDP was mainly considered in the sample-based setup to find optimal policy when the transition matrix and reward are unknown (Dewanto et al., 2020). For this setup, Burnetas &amp; Katehakis (1997); Jaksch et al. (2010) analyze regret minimization problem for unichain and communicating MDPs. Also, model-based algorithms (Zhang &amp; Ji, 2019), model-free algorithms (Wei et al., 2020; Wan et al., 2021), policy gradient method (Kakade, 2001), and finite time analysis (Zhang et al., 2021) have been studied.

Convergence of Value Iteration. Value iteration (VI) was first introduced in the DP literature (Bellman, 1957) and serves as a fundamental dynamic programming algorithm for computing the value functions. Its approximate and sample-based variants, such as Temporal Different Learning (Sutton, 1988), Fitted Value Iteration (Ernst et al., 2005; Munos &amp; Szepesvári, 2008), Deep QNetwork (Mnih et al., 2015), are the workhorses of modern RL algorithms (Bertsekas &amp; Tsitsiklis, 1996; Sutton &amp; Barto, 2018a; Szepesvári, 2010). VI is also routinely applied in diverse settings, including factored MDPs (Rosenberg &amp; Mansour, 2021), robust MDPs (Kumar et al., 2024), MDPs with reward machines (Bourel et al., 2023), and MDPs with options (Fruit et al., 2017).

The convergence of VI in average-reward MDPs has been extensively studied. For unichain MDPs, delta coefficient and ergodicity coefficient have been considered as the linear rate of VI (Seneta,

Figure 1: Classification of MDPs: Unichain ⊂ Weakly Communicating ⊂ Multichain (General)

<!-- image -->

Modified Bellman equations. Following Puterman (2014, Section 9.1.1), we consider the modified Bellman equations defined as

<!-- formula-not-decoded -->

for all s ∈ S , and we express these more concisely as

<!-- formula-not-decoded -->

We say ( g /star , h /star ) is a solution of the modified Bellman equations if ( g /star , h /star ) satisfies the two equations and there exists a policy π /star attaining maximum simultaneously. It is known that solutions of modified Bellman equations always exist (Puterman, 2014, Proposition 9.1.1). Furthermore, g /star is unique and it is equal to the optimal average reward (Puterman, 2014, Theorem 9.1.2, 9.1.6). Finally, a policy π /star simultaneously attaining the maximum in the modified Bellman equations is an optimal policy (Puterman, 2014, Theorem 9.1.7, 9.1.8).

If the MDP is weakly communicating or unichain, g /star ∈ R d is a uniform constant vector, i.e., g /star = c 1 for some c ∈ R , where 1 ∈ R n is the vector with entries all 1 (Puterman, 2014, Theorem 8.3.2, 8.4.1). Then, first modified Bellman equations holds automatically, and modified Bellman equations reduce to

<!-- formula-not-decoded -->

2006; Hübner, 1977), (Puterman, 2014, Theorem 6.6.1), and the J-stage span contraction demonstrates linear rate of VI for every J -th iterations in terms of span seminorm (Federgruen et al., 1978; Van Der Wal, 1981), (Puterman, 2014, Theorem 8.5.2). Bertsekas (1998) proposes λ -SSP, which exhibits non-asymptotic linear convergence under the recurrent assumption. When MDP is multichain, it is known that normalized iterates converge to the optimal average reward (Puterman, 2014, Theorem 9.4.1) while policy error might not converge to zero. Schweitzer &amp; Federgruen (1977; 1979) established necessary and sufficient conditions of convergence of VI and established asymptotic linear convergence on Bellman error.

For convergence of iterates to the h /star solution of modified Bellman equations, White (1963) introduced Relative Value Iteration (RVI) which subtracts a uniform constant for every iteration. Morton &amp; Wecker (1977) studied sufficient conditions of convergence of RVI. Bravo &amp; Cominetti (2024) studied asymptotic convergence rates of Rx-RVI on Bellman error in Q-learning setup, and Bravo &amp; Contreras (2024) also considered Q-learning version of Halpern iteration in average-reward MDP and study sample complexity.

In Section A of the appendix, we present several tables that thoroughly compare the our new results with the prior results of the literature, and refer to Della Vecchia et al. (2012) for further detailed conditions of convergence of VI.

Fixed point iterations. The Banach fixed-point theorem (Banach, 1922) establishes the convergence of the standard fixed-point iteration with a contractive operator. As a generalization of Picard iteration, Kransnosel'ski˘ ı-Mann iteration (KM) (Mann, 1953; Krasnosel'ski˘ ı, 1955) was introduced, and its convergence with general nonexpansive operators was shown by Martinet (1970). The Halpern iteration (Halpern, 1967) converges for nonexpansive operators on Hilbert spaces (Wittmann, 1992) and uniformly smooth Banach spaces (Reich, 1980; Xu, 2002).

When a nonexpansive operator T is assumed to have a fixed point, the fixed-point residual ‖ Tx k -x k ‖ is a commonly used error measure for fixed-point problems. In Hilbert spaces, the KM iteration with nonexpansive operators was shown to exhibit O (1 / √ k ) -rate by Matsushita (2017). Sabach &amp; Shtern (2017) first established an O (1 /k ) -rate for the Halpern iteration, and the constant was later improved by Lieder (2021); Kim (2021). In general normed spaces, KM iteration with nonexpansive opeator was proven to exhibit O (1 / √ k ) -rate (Baillon &amp; Bruck, 1992; Cominetti et al., 2014; Bravo &amp; Cominetti, 2018). The Halpern iteration was shown to exhibit O (1 /k ) -rate for (nonlinear) nonexpansive operators (Leustean, 2007; Sabach &amp; Shtern, 2017; Contreras &amp; Cominetti, 2022).

Inconsistent fixed-point iteration. A fixed-point iteration for a nonexpansive operator T without a fixed point is referred to as the inconsistent setup, and it is the analog relevant to the averagereward MDP setup. There exist a line of researches about convergence of inconsistent fixed-point iteration in both Hilbert space (Pazy, 1971; Applegate et al., 2024; Bauschke et al., 2014; Liu et al., 2019) and Banach space (Browder &amp; Petryshyn, 1966; Reich, 1973; Baillon, 1978; Reich &amp; Shafrir, 1987). Notably, Park &amp; Ryu (2023) studied sublinear convergence rates of KM iteration and Halpern iteration of the inconsistent setup in Hilbert spaces and established optimality by providing complexity lower bound.

Complexity lower bounds. With the information-based complexity analysis (Nemirovski, 1992), complexity lower bound on first-order methods for convex minimization problem has been thoroughly studied (Nesterov, 2018; Drori, 2017; Drori &amp; Taylor, 2022; Carmon et al., 2020; 2021; Drori &amp; Shamir, 2020). If a complexity lower bound matches an algorithm's convergence rate, it establishes optimality of the algorithm (Nemirovski, 1992; Kim &amp; Fessler, 2016; Salim et al., 2022; Taylor &amp; Drori, 2023; Drori &amp; Teboulle, 2016; Park &amp; Ryu, 2022). In Hilbert spaces, Park &amp; Ryu (2022) showed exact complexity lower bound on fixed-point residual for deterministic fixed-point iterations with contractive and nonexpansive operators. In fixed-point problems, Colao &amp; Marino (2021) established Ω ( 1 /k 1 -√ 2 /q ) lower bound on distance to solution for Halpern iteration with a nonexpansive operator in q -uniformly smooth Banach spaces. In general normed space, Contreras &amp; Cominetti (2022) provided Ω(1 /k ) lower bound on the fixed-point residual for the general Mann iteration with a nonexpansive linear operator, which includes Picard iteration, KM iteration, and Halpern iteration.

In discounted MDPs, Goyal &amp; Grand-Clément (2022) provided a lower bound on the Bellman error and distance to optimal value function for fixed-point iterations satisfying span condition with γ -contractive Bellman operators. Lee &amp; Ryu (2023) improved upon the prior lower bound on Bellman error by a factor 1 -γ k +1 , and further established Ω(1 /k ) bound in undiscounted MDP. However, none of these works consider the average-reward MDP setup. Zurek &amp; Chen (2023) studied sample complexity of learning a near-optimal policy in an average-reward MDP under generative model.

## 1.3 CONTRIBUTION

We summarize the contributions of this work as follows.

Non-asymptotic rates. For multichain MDPs, we establish the first non-asymptotic convergence rates on Bellman error. Theorems 1 and 2 and Corollary 1 and 2 present the non-asymptotic sublinear rates on both Bellman and policy errors in multichain MDPs (see Tables A.1 and A.2 of the Appendix). For the Relative Value Iteration (RVI) and its variants as described in Section 6, Theorems 5 and 6 establish the non-asymptotic sublinear rates on both Bellman and policy errors and point convergence in weakly communicating MDPs (see Tables A.5 and A.6 of the Appendix).

Complexity lower bound. Theorems 3 and 4 present the first complexity lower bounds for the average-reward MDP setup, one with a multichain MDP and another with a unichain MDP. These complexity lower bounds apply both to the Bellman error and normalized iterates for value-iterationtype methods satisfying the span condition.

Characterization of optimal complexity. Through our matching the convergence rates (upper bound) and the complexity lower bounds, we first establish the optimal complexity of standard VI in terms the normalized iterates and of Anc-VI in terms of the Bellman error.

## 2 PERFORMANCE MEASURES

We quickly review the standard performance measures used to quantify convergence rates of valueiteration-type methods for average-reward MDPs. Let T be the Bellman optimality operator of the given MDP, and suppose a method generates sequences { V k } k =0 , 1 ,... and { π k } k =0 , 1 ,... . We call V k -V 0 α k with an appropriate scaling factor α k &gt; 0 for k = 0 , 1 , . . . the normalized iterates . We call TV k -V k the Bellman error at V k for k = 0 , 1 , . . . . We call g /star -g π k the policy error at π k for k = 0 , 1 , . . . . Again, we call the V k = TV k -1 for k = 1 , 2 , . . . standard Value Iteration (VI) with greedy policy π k satisfying T π k V k = TV k .

Fact 1 (Classical result, (Puterman, 2014, Theorem 9.4.1)) . Consider a general (multichain) MDP. Then, for k ≥ 1 , the normalized iterates of standard VI with α k = k exhibit the rate

∥ ∥ Fact 1 shows that the normalized iterates converge to optimal average reward in multichain MDPs with a non-asymptotic rate. As we will later show with Theorem 4, the O (1 /k ) -rate on the normalized iterates of Fact 1 is exactly optimal. However, it is known that the convergence of normalized iterates does not guarantee convergence of policy error (Della Vecchia et al., 2012, Example 4).

<!-- formula-not-decoded -->

Fact 2 (Classical result, (Puterman, 2014, Theorem 9.1.7, 8.5.5)) . Consider a general (multichain) MDP. If ‖ TV -V -g /star ‖ ∞ = 0 , ‖ g /star -g π V ‖ ∞ = 0 , where π V is greedy policy satisfying T π V V = TV . Furthermore, if MDP is weakly communicating, ‖ g /star -g π V ‖ ∞ ≤ ‖ TV -V -g /star ‖ ∞ .

Fact 2 shows that, unlike normalized iterate, convergence of Bellman error guarantees convergence of policy error. But the classical asymptotic convergence results on the Bellman error of Fact 3 has

Fact 3 (Classical result, (Puterman, 2014, Theorem 9.4.5)) . Consider a general (multichain) MDP. Assume that the transition matrices corresponding to every average-optimal deterministic policy are aperiodic. Then, for standard VI, the Bellman error ∥ ∥ TV k -V k -g /star ∥ ∥ ∞ converges to zero. Furthermore, ‖ g π k -g /star ‖ ∞ also converges to zero.

no quantitative rate (and also additionally requires aperiodicity), so we establish several stronger non-asymptotic rates throughout this paper.

We also briefly mention another performance measure considered in average-reward MDPs. The span seminorm is defined as ‖ x ‖ sp = max i x i -min i x i for x ∈ R n . The span seminorm of the Bellman error ‖ TV k -V k ‖ sp has been considered for weakly communicating and unichain MDPs because in such setups, the optimal average reward g /star is a uniform constant and ‖ TV k -V k -g /star ‖ sp = ‖ TV k -V k ‖ sp . Therefore, unlike the ‖ · ‖ ∞ -norm of the Bellman error, the span seminorm is computable without knowledge of g /star . In this work, we primarily focus on convergence rates of the normalized iterates and ‖ · ‖ ∞ -norm of the Bellman errors. Nevertheless, we point out that our results on the latter measure imply rates on the span seminorm of the Bellman error in weakly communicating and unichain MDP, since ‖ TV -V ‖ sp ≤ 2 ‖ TV -V -g /star ‖ ∞ (Puterman, 2014, Section 6.6.1) in such setups.

## 3 RELAXED VALUE ITERATION

The Relaxed Value Iteration (Rx-VI) is

<!-- formula-not-decoded -->

for k = 1 , 2 , . . . , where T is the Bellman optimality operator, V 0 ∈ R n is a starting point, and 0 ≤ λ k &lt; 1 for k = 0 , 1 , . . . . π k is a greedy policy satisfying T π k V k = TV k for k = 0 , 1 , . . . . Notably, Rx-VI obtains the next iterate as a convex combination between the output of T and the current point V k -1 .

We now present our non-asymptotic sublinear converge rates of Rx-VI in terms of the Bellman and policy errors while deferring the proofs to Section F of the appendix.

Theorem 1. Consider a general (multichain) MDP. Let ( g /star , h /star ) be a solution of the modified Bellman equations. For k &gt; K , the Bellman and policy errors of Rx-VI with λ k = 1 / 2 exhibits the rate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and S is the set of all deterministic policies. Since π denotes the policy in this work, we write (3 . 141592 . . . ) to denote the mathematical constant usually written as π .

We clarify that for general (multichain) MDPs, the Bellman error does not bound the policy error. However, our analysis shows that the Bellman error does bound the policy error for k &gt; K .

The characterization of K in Theorem 1 is somewhat intricate when considering general MDPs. This is simplified if we focus on specific class of MDPs which includes weakly communicating and unichain MDPs.

Corollary 1. Consider a a general (multichain) MDP satsifying P π g /star = g /star for any policy π . Let ( g /star , h /star ) be a solution of the modified Bellman equations. For k ≥ 1 , the Bellman and policy errors of Rx-VI with λ k = 1 / 2 exhibit the rate

<!-- formula-not-decoded -->

Proof of Corollary 1. We apply Theorem 1. By assumption on MDP, S/ { π | P π g /star = g /star } = ∅ . So /epsilon1 = inf π ∈∅ ‖P π g /star -g /star ‖ ∞ = ∞ and K = 0 . Finally, we plug K = 0 into Theorem 1.

Note that the weakly communicating MDPs satisfy the assumption of Corollary 1 since g /star = c 1 for some c ∈ R (Puterman, 2014, Theorem 8.3.2) and so P π c 1 = c 1 for any policy π . In the

next section, we will show that the O (1 / √ k ) -rate with Rx-VI can be improved to O (1 /k ) -rate with Anc-VI. Section B of Appendix presents more general results establishing convergence rates for arbitrary λ k in terms of both the Bellman error and the normalized iterates.

Broadly speaking, Rx-VI is a well-studied algorithm. This averaging mechanism has been widely studied in fixed-point theory literature under the name Krasnosel'ski˘ ı-Mann iteration (Mann, 1953; Krasnosel'ski˘ ı, 1955; Bauschke &amp; Combettes, 2017; Baillon &amp; Bruck, 1992; Cominetti et al., 2014). In the dynamic programming literature, the aperiodic transformation (Puterman, 2014, Section 8.5.4), which averages the transition matrix and identity to make the transition matrix aperiodic, is closely related to this averaging mechanism. In the reinforcement learning literature, TD learning and Q learning use the averaging mechanism to stabilize randomness and ensure convergence (Sutton, 1988; Watkins, 1989; Bertsekas &amp; Tsitsiklis, 1995; Bravo &amp; Cominetti, 2024), and in tabular setup, Kushner &amp; Kleinman (1971); Porteus &amp; Totten (1978); Goyal &amp; Grand-Clément (2022); Akian et al. (2022) studied convergence of Rx-VI in discounted MDP setup. However, to the best of our knowledge, no prior work has established non-asymptotic rates of Rx-VI or any other valueiteration-type method for multichain MDPs. Only Schweitzer &amp; Federgruen (1977; 1979) established asymptotic convergence results for multichain MDPs.

## 4 ANCHORED VALUE ITERATION

The Anchored Value Iteration is for k = 1 , 2 , . . . , where T is the Bellman optimality operator, V 0 ∈ R n is a starting point, and 0 ≤ λ k &lt; 1 for k = 0 , 1 , . . . . π k is a greedy policy satisfying T π k V k = TV k for k = 0 , 1 , . . . . Notably, Anc-VI obtains the next iterate as a convex combination between the output of T and the starting point V 0 (note, Rx-VI uses V k -1 instead of V 0 ). We call the λ k V 0 term the anchor term since, loosely speaking, it serves to retract the iterates back toward the starting point V 0 . Generally, λ k is set to be a decreasing sequence, and then the strength of the anchor mechanism diminishes as the iteration progresses.

<!-- formula-not-decoded -->

We now present our non-asymptotic sublinear converge rates of Anc-VI in terms of the Bellman and policy errors while deferring the proofs to Section G of the Appendix.

Theorem 2. Consider a general (multichain) MDP. Let ( g /star , h /star ) be a solution of the modified Bellman equations. For k &gt; K , the Bellman and policy errors of Anc-VI with λ k = 2 k +2 . exhibits the rate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and S is the set of all deterministic policies.

As before, Theorem 2 claims that the Bellman error bounds the policy error for k &gt; K , and the characterization of K in Theorem 2 is simplified if we focus on a specific class of MDPs which includes weakly communicating and unichain MDPs.

Corollary 2. Consider a general (multichain) MDP satsifying P π g /star = g /star for any policy π . Let ( g /star , h /star ) be a solution of the modified Bellman equations. For k ≥ 1 , the Bellman and policy errors of Anc-VI with λ k = 2 k +2 exhibits the rate

Proof of Corollary 2.

<!-- formula-not-decoded -->

Follows from the same line of argument as for Corollary 1.

Note, the anchoring mechanism allows us to improve the rate to O (1 /k ) . In the next section, we will show that the O (1 /k ) rate is optimal in the weakly communicating setup by providing a matching complexity lower bound. Section C of Appendix presents more general results establishing convergence rates for arbitrary λ k in terms of both the Bellman error and the normalized iterates.

The anchor mechanism has been widely studied in minimax optimization and fixed-point problems (Halpern, 1967; Sabach &amp; Shtern, 2017; Lieder, 2021; Park &amp; Ryu, 2022; Contreras &amp; Cominetti, 2022; Yoon &amp; Ryu, 2021). In the context of reinforcement learning, Lee &amp; Ryu (2023) applied the anchoring mechanism to VI to achieve an accelerated convergencerate for cumulative-reward MDPs, and Bravo &amp; Contreras (2024) applied the anchoring mechanism to Q-learning for average-reward MDPs. However, to the best of our knowledge, no prior work established a non-asymptotic rate for value-iteration-type methods for multichain MDP.

We further clarify that our non-asymptotic convergence results in Section 3 and 4 are neither a direct application nor a direct adaptation of the prior convergence. VI for the average-reward MDP setup can be thought of as a fixed point iteration without a fixed point, and so most prior analyses assuming the existence of a fixed point do not apply. Bravo &amp; Contreras (2024); Bravo &amp; Cominetti (2024) study the convergence of Rx-RVI and Anc-RVI in unichain MDPs by applying results derived from the fixed-point iteration setup, but their analyses do not extend to mulichain MDPs. In the inconsistent fixed point iteration setup, analog relevant to the average-reward MDPs setup, prior analyses for Hilbert space (Pazy, 1971; Applegate et al., 2024; Bauschke et al., 2014; Liu et al., 2019; Park &amp; Ryu, 2023) are not applicable to Bellman operators since R d with ‖ · ‖ ∞ -norm is not Hilbert space. The prior analyses for Banach space assuming uniformly Gateaux differentiable norm (Browder &amp; Petryshyn, 1966; Reich, 1973; Reich &amp; Shafrir, 1987) or uniform convexity (Browder &amp; Petryshyn, 1966) are not applicable either since ‖ · ‖ ∞ -norm is not uniformly Gateaux differentiable norm and R d with ‖ · ‖ ∞ -norm is not uniformly convex space. We note that our analyses specifically utilize the structure of Bellman operators and modified Bellman equation to obtain a non-asymptotic convergence rate on both Bellman and policy errors, adapting proof techiniques from Cominetti et al. (2014); Contreras &amp; Cominetti (2022) to the multichain setup.

## 5 COMPLEXITY LOWER BOUND

We now present complexity lower bounds establishing optimality of Anc-VI in terms of the Bellman error and standard VI in terms of the normalized iterates. To the best of our knowledge, Theorems 3 and 4 are the first complexity lower bounds for value-iteration-type methods in the average-reward MDP setup.

Following the information-based complexity framework (Nemirovski, 1992), we consider the span condition

<!-- formula-not-decoded -->

where T is the Bellman optimality operator and span( A ) is set of all finite linear combinations of the elements of A . Standard VI, Rx-VI, and Anc-VI all satisfy equation 1.

Optimality of Anc-VI for Bellman error. We now establish the optimality of Anc-VI for weakly communicating and unichain MDPs in terms of the Bellman error.

Theorem 3. Let k ≥ 0 , n ≥ k +2 , and V 0 ∈ R n . Then there exists a unichain MDP with |S| = n and |A| = 1 such that its modified Bellman equations has a solution ( g /star , h /star ) satisfying

∥ ∥ for any iterates { V i } k i =0 satisfying the span condition equation 1 and any choice of real numbers { a i } k i =0 such that ∑ k i =0 a i = 1 .

<!-- formula-not-decoded -->

If we set a k = 1 in Theorem 3, we get ∥ ∥ TV k -V k -g /star ∥ ∥ ∞ ≥ 1 k +1 ∥ ∥ V 0 -h /star ∥ ∥ ∞ . Note that the construction of Theorem 3 is a unichain MDP, which is also a weakly communicating MDP. The lower bound matches the 8 k +1 ∥ ∥ V 0 -h /star ∥ ∥ ∞ upper bound of Corollary 5, which applies to both weakly communicating and unichain MDPs. The upper and lower bounds match up to constant of factor 8 , and we therefore conclude optimality for both weakly communicating and unichain MDPs.

Exact optimality of standard VI for normalized iterates. We now establish the optimality of standard VI for general (multichain) MDPs in terms of the normalized iterates.

Theorem 4. Let k ≥ 0 , n ≥ k +3 , and V 0 ∈ R n . Then there exists a multichain MDP with |S| = n and |A| = 1 such that its modified Bellman equations has a solution ( g /star , h /star ) satisfying

∥ ∥ for any iterates { V i } k i =0 satisfying the span condition equation 1 and any choice of real numbers { a i } k i =0 such that ∑ k i =0 a i = 1 .

<!-- formula-not-decoded -->

If we set a i = 1 k +1 for all i = 0 , . . . , k in Theorem 4, we get ∥ ∥ ∥ V k +1 -V 0 k +1 -g /star ∥ ∥ ∥ ∞ ≥ 2 k +1 ∥ ∥ V 0 -h /star ∥ ∥ ∞ . This lower bound exactly matches the 2 k +1 ∥ ∥ V 0 -h /star ∥ ∥ ∞ upper bound of Fact 1, and we therefore conclude exact optimality of standard VI in terms of the normalized iterates.

Discussion. To clarify, the unichain MDP construction of Theorem 3 is a multichain MDP, so Theorem 3 and Fact 1 together already establish optimality up to a constant factor of 2 . However, the multichain construction of Theorem 3 improves the lower bound by a constant factor of 2 , and this factor of 2 leads to the exact match.

The span condition used in Theorems 3 and 4 are arguably very natural and is satisfied by Standard VI, Rx-VI, and Anc-VI. The span condition is commonly used in the construction of complexity lower bounds for first-order optimization methods (Nesterov, 2018; Drori, 2017; Drori &amp; Taylor, 2022; Carmon et al., 2020; 2021; Park &amp; Ryu, 2022) and has been used in the lower bound for standard VI and Anc-VI (Goyal &amp; Grand-Clément, 2022; Lee &amp; Ryu, 2023). However, designing an algorithm that breaks the lower bound of Theorem 3 and 4 by violating the span condition remains a possibility. In optimization theory, there is precedence of lower bounds being broken by violating seemingly natural and minute conditions (Hannah et al., 2018; Golowich et al., 2020; Yoon &amp; Ryu, 2021).

## 6 RELAXED AND ANCHORED RELATIVE VALUE ITERATION

The iterates of standard VI, Rx-VI, and Anc-VI diverge . For example, the iterates of standard VI asymptotically behave as V k ∼ kg /star as k → ∞ by Fact 1. Of course, the normalized iterates do converge, but if we want the iterates themselves to converge, the algorithm must be modified.

The Relative Value Iteration (RVI) subtracts some uniform constant vector at each iteration:

<!-- formula-not-decoded -->

Fact 4 (Classical result, (Bertsekas, 2012, Theorem 4.3.2)) . Consider a unichain MDP. Assume that the transition matrices corresponding to every average-optimal deterministic policy are aperiodic, and f ( h ) = ( Th ) i for some fixed 1 ≤ i ≤ n . Then, for some solution of modified Bellman equations ( g /star , h /star ) , the iterates of standard RVI converge to h /star and ( Th k ) i 1 converges to g /star .

for k = 1 , 2 , . . . , where T is the bellman optimality operator, h 0 ∈ R n is a starting point, 1 ∈ R n is the uniform constant vector with all entries 1 , and f : R n → R is a continuous function satisfying f ( x + c 1 ) = f ( x ) + c for any c ∈ R . Following is one of known convergence results of RVI.

Like standard VI, iterates of Rx-VI and Anc-VI also diverge as we show in the Theorems 7 and 9 of the Appendix. To ensure convergence of the iterates, we can also subtract uniform constant vectors from the iterate. The Relaxed Relative Value Iteration is

<!-- formula-not-decoded -->

for k = 1 , 2 , . . . , where 0 ≤ λ k &lt; 1 and h 0 is starting point. π k is a greedy policy satisfying T π k h k = Th k for k = 0 , 1 , . . . . The Anchored Relative Value Iteration is

<!-- formula-not-decoded -->

for k = 1 , 2 , . . . , where 0 ≤ λ k &lt; 1 and h 0 is starting point. π k is a greedy policy satisfying T π k h k = Th k for k = 0 , 1 , . . . .

Now we present our non-asymptotic convergence rates of Rx-RVI and Anc-RVI in terms of Bellman and policy errors while deferring the proofs to Section I in Appendix.

Theorem 5. Consider a weakly communicating MDP. Let ( g /star , h /star ) be a solution of modified Bellman equations. For k ≥ 1 and , the Bellman and policy errors of Rx-RVI with λ k = 1 / 2 exhibits the rate

<!-- formula-not-decoded -->

Furthermore, h k → h ∞ and f ( h k ) 1 → g /star for some solution of modified Bellman equations ( g /star , h ∞ ) .

Theorem 6. Consider a weakly communicating MDP. Let ( g /star , h /star ) be a solution of modified Bellman equations. For k ≥ 1 , the Bellman and policy errors of Anc-RVI with λ k = 2 k +2 exhibits the rate

<!-- formula-not-decoded -->

Furthermore, if MDP is unichain, h k → h ∞ and f ( h k ) 1 → g /star for some solution of modified Bellman equations ( g /star , h ∞ ) .

Since Rx-RVI and Anc-RVI generate same policy as Rx-VI and Anc-VI, respectively, the rates of Bellman errors of Rx-RVI and Anc-RVI in Theorem 5 and 6 are immediately implied by the rates of Rx-VI and Anc-VI in Corollary 1 and 2, respectively. Therefore, the main substance of Corollary 1 and 2 are the convergence results ( h k , f ( h k ) 1 ) → ( h ∞ , g /star ) . Section D of Appendix presents more general results establishing convergence rates for arbitrary λ k in terms of the Bellman error and convergence of iterates. Lastly, we briefly note that for weakly communicating MDP, non-asymptotic rate on Bellman error can be obtained from results in Bravo &amp; Contreras (2024); Bravo &amp; Cominetti (2024) by leveraging their convergence analysis with uniform constant g /star in unichain MDP.

## 7 CONCLUSION

In this work, we present the first non-asymptotic convergence rates for multichain MDPs in terms of the Bellman error. We also provide complexity lower bounds matching the upper bound of Anc-VI in terms of the Bellman error up to a constant factor of 8 for weakly communicating and unichain MDPs. Finally, we also showed that standard VI is exactly optimal in terms of the normalized iterates for multichain MDPs. Our results and proof techniques open the door to future work on non-asymptotic, sublinear, and optimal rates for average-reward MDPs.

One future direction is to fully characterize the optimal non-asymptotic complexity on Bellman error for multichain MDPs, as our current upper bound of Theorem 2, with its dependence on K , does not exactly match the lower bound of Theorem 4. We aim to achieve this goal by enhancing our lower bound through the consideration of more delicate worst-case multichain MDPs.

Finally, we highlight an observation implied by our results: the 'correct' rates for (undiscounted) average-reward MDPs are sublinear, i.e., something like O (1 /k ) . This contrasts with the classical γ -discounted cumulative-reward MDP setup, where we are accustomed to O ( γ k ) -rates. We expect future work analyzing other average-reward MDP setups and algorithms to similarly discover optimal sublinear rates.

## 8 ACKNOWLEDGMENTS

This work is supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (No.RS-2024-00421203). We thank Taeho Yoon for providing valuable feedback.

## REFERENCES

Marianne Akian, Stéphane Gaubert, Zheng Qu, and Omar Saadi. Multiply accelerated value iteration for non-symmetric affine fixed point problems and application to Markov decision processes. SIAM Journal on Matrix Analysis and Applications , 43(1):199-232, 2022.

- David Applegate, Mateo Díaz, Haihao Lu, and Miles Lubin. Infeasibility detection with primaldual hybrid gradient for large-scale linear programming. SIAM Journal on Optimization , 34(1): 459-484, 2024.
- Jean-Bernard Baillon. On the asymptotic behavior of nonexpansive mappings and semigroups in banach spaces. Houston J. Math. , 4(1):1-9, 1978.
- Jean-Bernard Baillon and Ronald E Bruck. Optimal rates of asymptotic regularity for averaged nonexpansive mappings. Fixed Point Theory and Applications , 128:27-66, 1992.
- Stefan Banach. Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales. Fundamenta Mathematicae , 3(1):133-181, 1922.
- Heinz H Bauschke and Patrick L Combettes. Convex Analysis and Monotone Operator Theory in Hilbert Spaces . Springer, 2th edition, 2017.
- Heinz H Bauschke, Warren L Hare, and Walaa M Moursi. Generalized solutions for the sum of two maximally monotone operators. SIAM Journal on Control and Optimization , 52(2):1034-1047, 2014.
- Richard Bellman. A Markovian decision process. Journal of Mathematics and Mechanics , 6(5): 679-684, 1957.
- D. P. Bertsekas and J. N. Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1996.
- Dimitri P Bertsekas. A new value iteration method for the average cost dynamic programming problem. SIAM journal on control and optimization , 36(2):742-759, 1998.
- Dimitri P Bertsekas. Dynamic Programming and Optimal Control, volume II . 4th edition, 2012.
- Dimitri P Bertsekas and John N Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1995.
- David Blackwell. Discrete dynamic programming. The Annals of Mathematical Statistics , 33:719726, 1962.
- Hippolyte Bourel, Anders Jonsson, Odalric-Ambrym Maillard, and Mohammad Sadegh Talebi. Exploration in reward machines with low regret. International Conference on Artificial Intelligence and Statistics , 2023.
- Mario Bravo and Roberto Cominetti. Sharp convergence rates for averaged nonexpansive maps. Israel Journal of Mathematics , 227:163-188, 2018.
- Mario Bravo and Roberto Cominetti. Stochastic fixed-point iterations for nonexpansive maps: Convergence and error bounds. SIAM Journal on Control and Optimization , 62(1):191-219, 2024.
- Mario Bravo and Juan Pablo Contreras. Stochastic halpern iteration in normed spaces and applications to reinforcement learning. arXiv preprint arXiv:2403.12338 , 2024.
- Felix Earl Browder and Walter Petryshyn. The solution by iteration of nonlinear functional equations in banach spaces. Bulletin of the American Mathematical Society , 72:571-575, 1966.
- Apostolos N Burnetas and Michael N Katehakis. Optimal adaptive policies for markov decision processes. Mathematics of Operations Research , 22(1):222-255, 1997.
- Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points I. Mathematical Programming , 184(1-2):71-120, 2020.
- Yair Carmon, John C. Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points II: first-order methods. Mathematical Programming , 185(1-2):315-355, 2021.
- Vittorio Colao and Giuseppe Marino. On the rate of convergence of Halpern iterations. Journal of Nonlinear and Convex Analysis , 22(12):2639-2646, 2021.
- Roberto Cominetti, José A Soto, and José Vaisman. On the rate of convergence of Krasnosel'ski˘ ıMann iterations and their connection with sums of Bernoullis. Israel Journal of Mathematics , 199 (2):757-772, 2014.

- Juan Pablo Contreras and Roberto Cominetti. Optimal error bounds for non-expansive fixed-point iterations in normed spaces. Mathematical Programming , 199(1-2):343-374, 2022.
- Eugenio Della Vecchia, Silvia Di Marco, and Alain Jean-Marie. Illustrated review of convergence conditions of the value iteration algorithm and the rolling horizon procedure for average-cost MDPs. Annals of Operations Research , 199:193-214, 2012.
- Eric V Denardo and Bennett L Fox. Multichain Markov renewal programs. SIAM Journal on Applied Mathematics , 16(3):468-487, 1968.
- Vektor Dewanto, George Dunn, Ali Eshragh, Marcus Gallagher, and Fred Roosta. Average-reward model-free reinforcement learning: a systematic review and literature mapping. arXiv preprint arXiv:2010.08920 , 2020.
- Yoel Drori. The exact information-based complexity of smooth convex minimization. Journal of Complexity , 39:1-16, 2017.
- Yoel Drori and Ohad Shamir. The complexity of finding stationary points with stochastic gradient descent. International Conference on Machine Learning , 2020.
- Yoel Drori and Adrien Taylor. On the oracle complexity of smooth strongly convex minimization. Journal of Complexity , 68, 2022.
- Yoel Drori and Marc Teboulle. An optimal variant of Kelley's cutting-plane method. Mathematical Programming , 160(1-2):321-351, 2016.
- D. Ernst, P. Geurts, and L. Wehenkel. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research , 2005.
- Awi Federgruen, Paul J Schweitzer, and Hendrik Cornelis Tijms. Contraction mappings underlying undiscounted Markov decision problems. Journal of Mathematical Analysis and Applications , 65 (3):711-730, 1978.
- Ronan Fruit, Matteo Pirotta, Alessandro Lazaric, and Emma Brunskill. Regret minimization in mdps with options without prior knowledge. Neural Information Processing Systems , 2017.
- Noah Golowich, Sarath Pattathil, Constantinos Daskalakis, and Asuman Ozdaglar. Last iterate is slower than averaged iterate in smooth convex-concave saddle point problems. Conference on Learning Theory , 2020.
- Vineet Goyal and Julien Grand-Clément. A first-order approach to accelerated value iteration. Operations Research , 71(2):517-535, 2022.
- Benjamin Halpern. Fixed points of nonexpanding maps. Bulletin of the American Mathematical Society , 73(6):957-961, 1967.
- Robert Hannah, Yanli Liu, Daniel O'Connor, and Wotao Yin. Breaking the span assumption yields fast finite-sum minimization. Neural Information Processing Systems , 2018.
- Ronald A Howard. Dynamic Programming and Markov Processes. John Wiley and Sons, 1960.
- Gerhard Hübner. Improved procedures for eliminating suboptimal actions in markov programming by the use of contraction properties. Transactions of the Seventh Prague Conference on Information Theory, Statistical Decision Functions, Random Processes , 1977.
- Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(51):1563-1600, 2010.
- Sham M Kakade. A natural policy gradient. Neural Information Processing Systems , 2001.
- Donghwan Kim. Accelerated proximal point method for maximally monotone operators. Mathematical Programming , 190(1-2):57-87, 2021.
- Donghwan Kim and Jeffrey A Fessler. Optimized first-order methods for smooth convex minimization. Mathematical Programming , 159(1-2):81-107, 2016.

- Mark A Krasnosel'ski˘ ı. Two remarks on the method of successive approximations. Uspekhi Matematicheskikh Nauk , 10(1):123-127, 1955.
- Navdeep Kumar, Kaixin Wang, Kfir Yehuda Levy, and Shie Mannor. Efficient value iteration for s-rectangular robust markov decision processes. International Conference on Machine Learning , 2024.
- Harold Kushner and A Kleinman. Accelerated procedures for the solution of discrete Markov control problems. IEEE Transactions on Automatic Control , 16(2):147-152, 1971.
- Jongmin Lee and Ernest Ryu. Accelerating value iteration with anchoring. Neural Information Processing Systems , 2023.
- Laurentiu Leustean. Rates of asymptotic regularity for Halpern iterations of nonexpansive mappings. Journal of Universal Computer Science , 13(11):1680-1691, 2007.
- Felix Lieder. On the convergence rate of the Halpern-iteration. Optimization Letters , 15(2):405-418, 2021.
- Yanli Liu, Ernest K Ryu, and Wotao Yin. A new use of Douglas-Rachford splitting for identifying infeasible, unbounded, and pathological conic programs. Mathematical Programming , 177(1): 225-253, 2019.
- W Robert Mann. Mean value methods in iteration. Proceedings of the American Mathematical Society , 4(3):506-510, 1953.
- B. Martinet. Régularisation d'inéquations variationnelles par approximations successives. Revue Française de Informatique et Recherche Opérationnelle , 4(R3):154-158, 1970.
- Shin-Ya Matsushita. On the convergence rate of the Krasnosel'ski˘ ı-Mann iteration. Bulletin of the Australian Mathematical Society , 96(1):162-170, 2017.
- V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, and et al. Human-level control through deep reinforcement learning. Nature , 518(7540):529-533, 2015.
- Thomas E Morton and William E Wecker. Discounting, ergodicity and convergence for Markov decision processes. Management Science , 23(8):890-900, 1977.
- R. Munos and C. Szepesvári. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research , 2008.
- Arkadi Semenoviˇ c Nemirovski. Information-based complexity of linear operator equations. Journal of Complexity , 8(2):153-175, 1992.
- Yurii Nesterov. Lectures on Convex Optimization . Springer, 2nd edition, 2018.
- Jisun Park and Ernest K Ryu. Exact optimal accelerated complexity for fixed-point iterations. International Conference on Machine Learning , 2022.
- Jisun Park and Ernest K. Ryu. Accelerated infeasibility detection of constrained optimization and fixed-point iterations. International Conference on Machine Learning , 2023.
- A Pazy. Asymptotic behavior of contractions in hilbert space. Israel Journal of Mathematics , 9: 235-240, 1971.
- Evan L Porteus and John C Totten. Accelerated computation of the expected discounted return in a Markov chain. Operations Research , 26(2):350-358, 1978.
- Martin L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . John Wiley and Sons, 2nd edition, 2014.
- Simeon Reich. Asymptotic behavior of contractions in banach spaces. Journal of Mathematical Analysis and Applications , 44(1):57-70, 1973.

- Simeon Reich. Strong convergence theorems for resolvents of accretive operators in Banach spaces. Journal of Mathematical Analysis and Applications , 75(1):287-292, 1980.
- Simeon Reich and Itai Shafrir. The asymptotic behavior of firmly nonexpansive mappings. Proceedings of the American Mathematical Society , 101(2):246-250, 1987.
- Aviv Rosenberg and Yishay Mansour. Oracle-efficient regret minimization in factored mdps with unknown structure. Neural Information Processing Systems , 2021.
- Shoham Sabach and Shimrit Shtern. A first order method for solving convex bilevel optimization problems. SIAM Journal on Optimization , 27(2):640-660, 2017.
- Adil Salim, Laurent Condat, Dmitry Kovalev, and Peter Richtárik. An optimal algorithm for strongly convex minimization under affine constraints. International Conference on Artificial Intelligence and Statistics , 2022.
- Paul J Schweitzer. On the existence of relative values for undiscounted Markovian decision processes with a scalar gain rate. Journal of mathematical analysis and applications , 104(1):67-78, 1984.
- Paul J Schweitzer and Awi Federgruen. The asymptotic behavior of undiscounted value iteration in Markov decision problems. Mathematics of Operations Research , 2(4):360-381, 1977.
- Paul J Schweitzer and Awi Federgruen. The functional equations of undiscounted Markov renewal programming. Mathematics of Operations Research , 3(4):308-321, 1978.
- Paul J Schweitzer and Awi Federgruen. Geometric convergence of value-iteration in multichain Markov decision problems. Advances in Applied Probability , 11(1):188-217, 1979.
- E Seneta. Non-Negative Matrices and Markov Chains . Springer Science &amp; Business Media, 3th edition, 2006.
- R. S. Sutton. Learning to predict by the methods of temporal differences. Machine Learning , 1988.
- R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction . The MIT Press, second edition, 2018a.
- Richard S Sutton and Andrew G Barto. Reinforcement Learning: An introduction . MIT press, 2nd edition, 2018b.
- C. Szepesvári. Algorithms for Reinforcement Learning . Morgan Claypool Publishers, 2010.
- Adrien Taylor and Yoel Drori. An optimal gradient method for smooth strongly convex minimization. Mathematical Programming , 199(1-2):557-594, 2023.
- Johannes Van Der Wal. Stochastic dynamic programming: successive approximations and nearly optimal strategies for Markov decision processes and Markov games. 1981.
- Yi Wan, Abhishek Naik, and Richard S Sutton. Learning and planning in average-reward Markov decision processes. International Conference on Machine Learning , 2021.
- C. J. C. H. Watkins. Learning from Delayed Rewards . PhD thesis, 1989.
- Chen-Yu Wei, Mehdi Jafarnia Jahromi, Haipeng Luo, Hiteshi Sharma, and Rahul Jain. Model-free reinforcement learning in infinite-horizon average-reward Markov decision processes. International conference on machine learning , 2020.
- D. J. White. Dynamic programming, Markov chains, and the method of successive approximations. J. Math. Anal. Appl , 6(3):373-376, 1963.
- Rainer Wittmann. Approximation of fixed points of nonexpansive mappings. Archiv der Mathematik , 58(5):486-491, 1992.
- Hong-Kun Xu. Iterative algorithms for nonlinear operators. Journal of the London Mathematical Society , 66(1):240-256, 2002.

- TaeHo Yoon and Ernest K Ryu. Accelerated algorithms for smooth convex-concave minimax problems with O (1 /k 2 ) rate on squared gradient norm. International Conference on Machine Learning , 2021.
- AA Yushkevich. On a class of strategies in general Markov decision models. Theory of Probability &amp;Its Applications , 18(4):777-779, 1974.
- Sheng Zhang, Zhe Zhang, and Siva Theja Maguluri. Finite sample analysis of average-reward td learning and q -learning. Neural Information Processing Systems , 2021.
- Zihan Zhang and Xiangyang Ji. Regret minimization for reinforcement learning by evaluating the optimal bias function. Neural Information Processing Systems , 2019.
- Matthew Zurek and Yudong Chen. Span-based optimal sample complexity for average reward MDPs. arXiv preprint arXiv:2311.13469 , 2023.

## A COMPARISON WITH PRIOR CONVERGENCE RESULTS IN TERMS OF PERFORMANCE MEASURES

In this section, we present asymptotic and non-asymptotic convergence results of prior works and our work in terms of Bellman error, policy error, span seminorm, and normalized iterates. We denote the correspondence between numbers and prior works as follows. [ 1 : Federgruen et al. (1978), 2 : Van Der Wal (1981), 3 : Schweitzer &amp; Federgruen (1977), 4 : Schweitzer &amp; Federgruen (1979), 5 : Bertsekas (1998), 6: Puterman (2014), 7: Bravo &amp; Cominetti (2024), 8: Bravo &amp; Contreras (2024), 9: Bertsekas (2012)]

## A.1 BELLMAN ERROR

| Prior works     | Non-asym multi MDP   | Asym multi MDP   | Non-asym w.c. MDP   | Asym w.c. MDP   | Non-asym uni MDP   | Asym uni MDP   |
|-----------------|----------------------|------------------|---------------------|-----------------|--------------------|----------------|
| [1 , 2] [3 , 4] | ✗ ✗                  | ✗                | ✗ ✗                 | ✗ ✓ ✗           | ✓ ✓ ✓              | ✓ ✓ ✓          |
|                 |                      | ✓                |                     |                 |                    |                |
| [5]             | ✗                    | ✗                | ✗                   |                 |                    |                |
| Rx-VI Anc-VI    | ✓                    | ✓                | ✓ ✓                 | ✓               | ✓ ✓                | ✓ ✓            |
|                 | ✓                    | ✓                |                     | ✓               |                    |                |

## A.2 POLICY ERROR

| Prior works         | Non-asym multi MDP   | Asym multi MDP   | Non-asym w.c. MDP   | Asym w.c. MDP   | Non-asym uni MDP   | Asym uni MDP   |
|---------------------|----------------------|------------------|---------------------|-----------------|--------------------|----------------|
| [1 , 2] [3 , 4] [5] | ✗ ✗ ✗                | ✗ ✗              | ✗ ✗ ✗               | ✗ ✓ ✗           | ✓ ✓ ✓              | ✓ ✓ ✓          |
| Rx-VI               |                      | ✓                |                     |                 |                    |                |
| Anc-VI              | ✓ ✓                  | ✓ ✓              | ✓ ✓                 |                 | ✓ ✓                | ✓ ✓            |
|                     |                      |                  |                     | ✓ ✓             |                    |                |

## A.3 SPAN SEMINORM

| Prior works     | Non-asym multi MDP   | Asym multi MDP   | Non-asym w.c. MDP   | Asym w.c. MDP   | Non-asym uni MDP   | Asym uni MDP   |
|-----------------|----------------------|------------------|---------------------|-----------------|--------------------|----------------|
| [1 , 2] [3 , 4] | N/A N/A N/A          | N/A N/A N/A      | ✗ ✗ ✗               | ✗ ✓ ✗           | ✓ ✓ ✓              | ✓ ✓ ✓          |
| Rx-VI           |                      |                  |                     |                 |                    |                |
| [5]             |                      |                  |                     |                 |                    |                |
|                 | N/A                  | N/A              | ✓                   | ✓               | ✓                  | ✓              |
| Anc-VI          | N/A                  | N/A              | ✓                   | ✓               | ✓                  | ✓              |

## A.4 NORMALIZED ITERATES

| Prior works   | Non-asym multi MDP   | Asym multi MDP   | Non-asym w.c. MDP   | Asym w.c. MDP   | Non-asym uni MDP   | Asym uni MDP   |
|---------------|----------------------|------------------|---------------------|-----------------|--------------------|----------------|
| [6]           | ✓                    | ✓                | ✓                   | ✓               | ✓                  | ✓              |
| Rx-VI Anc-VI  | ✓                    | ✓                | ✓                   | ✓               | ✓                  | ✓ ✓            |
|               | ✓                    | ✓                | ✓                   | ✓               | ✓                  |                |

## A.5 BELLMAN ERROR (RVI)

| Prior works    | Non-asym multi MDP   | Asym multi MDP   | Non-asym w.c. MDP   | Asym w.c. MDP   | Non-asym uni MDP   | Asym uni MDP   |
|----------------|----------------------|------------------|---------------------|-----------------|--------------------|----------------|
| [7] [8] [9]    | N/A N/A N/A          | N/A N/A N/A      | ✗ ✗ ✗               | ✗ ✗ ✗           | ✓ ✓ ✓ ✓            | ✓ ✓ ✓          |
| Rx-RVI Anc-RVI | N/A                  | N/A              | ✓ ✓                 | ✓               |                    |                |
|                |                      |                  |                     |                 |                    | ✓ ✓            |
|                | N/A                  | N/A              |                     | ✓               | ✓                  |                |

## A.6 POLICY ERROR (RVI)

| Prior works    | Non-asym multi MDP   | Asym multi MDP   | Non-asym w.c. MDP   | Asym w.c. MDP   | Non-asym uni MDP   | Asym uni MDP   |
|----------------|----------------------|------------------|---------------------|-----------------|--------------------|----------------|
| [7] [8] [9]    | N/A N/A N/A          | N/A N/A N/A      | ✗ ✗ ✗               | ✗ ✗ ✗           | ✓ ✓ ✓              | ✓ ✓ ✓          |
| Rx-RVI Anc-RVI | N/A N/A              | N/A N/A          | ✓ ✓                 | ✓ ✓             | ✓ ✓                | ✓ ✓            |

## A.7 SPAN SEMINORM (RVI)

| Prior works    | Non-asym multi MDP   | Asym multi MDP   | Non-asym w.c. MDP   | Asym w.c. MDP   | Non-asym uni MDP   | Asym uni MDP   |
|----------------|----------------------|------------------|---------------------|-----------------|--------------------|----------------|
| [7] [8] [9]    | N/A N/A N/A          | N/A N/A N/A      | ✗ ✗ ✗               | ✗ ✗ ✗           | ✓ ✓ ✓              | ✓ ✓ ✓          |
| Rx-RVI Anc-RVI | N/A N/A              | N/A N/A          | ✓ ✓                 | ✓               |                    | ✓              |
|                |                      |                  |                     |                 |                    | ✓              |
|                |                      |                  |                     |                 | ✓                  |                |
|                |                      |                  |                     | ✓               | ✓                  |                |

## B CONVERGENCE RATES OF RX-VI WITH ARBITRARY λ k

In this section, we present the convergence rates of Rx-VI for arbitrary λ k in terms of both the Bellman error and the normalized iterates.

Theorem 7. Consider a general (multichain) MDP. Let ( g /star , h /star ) be a solution of the modified Bellman equations. For k &gt; K , the normalized iterates of Rx-VI with α k = ∑ k i =1 (1 -λ i ) exhibits the rate

∥ ∥ Theorem 8. Consider a general (multichain) MDP. Let ( g /star , h /star ) be a solution of the modified Bellman equations. Let 0 &lt; λ j for 1 ≤ j and lim sup λ j &lt; 1 . Then, there exist 0 &lt; K such that for K &lt; k , the Bellman and policy errors of Rx-VI exhibit the rates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Specifically, K is the minimum iteration number satisfying if K ≤ k , π k generated by Rx-VI satisfies P π k g /star = g /star , first modified Bellman equation.

Wedefer the proofs to Appendix F. Note that Theorems 7 and 8 imply the convergence of normalized iterate and Bellman error to g /star respectively when ∑ ∞ i =1 (1 -λ i ) = ∞ and ∑ ∞ i =1 λ i (1 -λ i ) = ∞ . Interestingly, for normalized iterate, Theorem 1 recovers rate of standard VI in Fact 1.

Corollary 3. Consider a general (multichain) MDP. Let ( g /star , h /star ) be a solution of the modified Bellman equations. The normalized iterate of Rx-VI in Theorem 7 is optimized when λ k = 0 with

<!-- formula-not-decoded -->

∥ ∥ Proof. By AM-GM inequality, we have Π k i =1 λ i ≤ (Π k i =1 λ i ) 1 /k ≤ ∑ k i =1 λ i k since λ i ≤ 1 . This implies 1 k ≤ 1 -Π k i =1 λ i ∑ k i =1 (1 -λ i ) and if λ i = 0 for all i , equality holds. Therefore, by plugging λ i = 0 in Theorem 7, we get the desired result.

Lastly, we present the non-asymptotic rate of Rx-VI with arbitrary λ k in specific class of MDPs which includes weakly communicating.

Corollary 4. Consider a a general (multichain) MDP satsifying P π g /star = g /star for any policy π . Let ( g /star , h /star ) be a solution of the modified Bellman equations. Let 0 &lt; λ j &lt; 1 for 1 ≤ j . Then, for 1 ≤ k , the Bellman error of Rx-VI exhibit the rates

<!-- formula-not-decoded -->

Proof. We apply Theorem 8. By assumption on MDP, S/ { π | P π g /star = g /star } = ∅ . So /epsilon1 = inf π ∈∅ ‖P π g /star -g /star ‖ ∞ = ∞ and K = 0 . Finally, we plug K = 0 into Theorem 8

## C CONVERGENCE RATES OF ANC-VI WITH ARBITRARY λ k

In this section, we present the convergence rates of Anc-VI for arbitrary λ k in terms of both the Bellman error and the normalized iterates.

Theorem 9. Consider a general (multichain) MDP. Let ( g /star , h /star ) be a solution of the modified Bellman equations. The normalized iterates of Anc-VI with α k = ∑ k i =1 Π k j = i (1 -λ j ) exhibits the rates

∥ ∥ Theorem 10. Consider a general (multichain) MDP. Let ( g /star , h /star ) be a solution of the modified Bellman equations. Let λ k +1 ≤ λ k &lt; 1 for 1 ≤ k and lim λ k = 0 . Then, there exist 0 &lt; K such that for K &lt; k , the Bellman and policy errors of Anc-VI exhibit the rates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Specifically, K is the minimum iteration number satisfying if K ≤ k , π k generated by Anc-VI satisfies P π k g /star = g /star , first modified Bellman equation.

We defer the proofs to Appendix G. Note that Theorems 3 and 4 imply the convergence of normalized iterate and Bellman error to g respectively when lim k →∞ ∑ k i =0 Π k j = i (1 -λ j ) = ∞ and lim k →∞ ∑ k i =1 λ i Π k j = i (1 -λ j ) = 1 . We briefly mention that like Rx-VI, convergence rate of normalized iterate of Anc-VI is optimized when λ k = 0 and recover rate of standard VI in Fact 1.

Lastly, we present the non-asymptotic rate of Anc-VI with arbitrary λ k in specific class of MDPs which includes weakly communicating.

Corollary 5. Consider a a general (multichain) MDP satsifying P π g /star = g /star for any policy π . Let ( g /star , h /star ) be a solution of the modified Bellman equations. Let λ k +1 ≤ λ k &lt; 1 for 1 ≤ k . Then, there exist 0 &lt; K such that for K &lt; k , the Bellman and policy errors of Anc-VI exhibit the rates

<!-- formula-not-decoded -->

Proof. We apply Theorem 10. By assumption on MDP, S/ { π | P π g /star = g /star } = ∅ . So /epsilon1 = inf π ∈∅ ‖P π g /star -g /star ‖ ∞ = ∞ and K = 0 . Finally, we plug K = 0 into Theorem 10

## D CONVERGENCE RATES OF RX-RVI AND ANC-RVI WITH ARBITRARY λ k

In this section, we present the convergence rates of Rx-RVI and Anc-RVI for arbitrary λ k in terms of both the Bellman error.

Theorem 11. Consider a weakly communicating MDP. Let ( g /star , h /star ) be a solution of the modified Bellman equations. Let 0 &lt; λ j &lt; 1 for 1 ≤ j . Then, for 1 ≤ k , Rx-RVI exhibit the rates and if lim sup λ j &lt; 1 , h k converges to h /star and f ( h k ) 1 converges to g /star for some solution of modified Bellman equations ( g /star , h ∞ ) . .

<!-- formula-not-decoded -->

Theorem 12. Consider a weakly communicating MDP. Let ( g /star , h /star ) be a solution of the modified Bellman equations. Let λ k +1 ≤ λ k &lt; 1 for 1 ≤ k . Then, Anc-RVI exhibit the rates and if lim λ k = 0 and MDP is unichain, h k converges to h /star and f ( h k ) 1 converges to g /star for some solution of modified Bellman equations ( g /star , h ∞ ) .

<!-- formula-not-decoded -->

We defer the proofs to Appendix I. We note that rates of Bellman errors of Rx-RVI and Anc-RVI in Theorem 11 and 12 are exactly match to the rates of Rx-VI and Anc-VI in Corollary 4 and 5, respectively.

## E PRELIMINARIES

In this section, we define some notations and introduce elementary propositions used in proofs.

## E.1 NOTATIONS

We denote V ≤ ˜ V if V ( s ) ≤ ˜ V ( s ) for all s ∈ S and V, ˜ V ∈ R n .

We denote P /star = lim k →∞ 1 k ∑ k i =0 P i for Cesaro limit of stochastic matrix P (Cesaro limit of stochastic matrix always exist (Puterman, 2014, Theorem A.6)).

We denote Π k i = j A i = A j A j +1 . . . A k (ascending order) and Π j i = k A i = A k A k -1 . . . A j (descending order) where 0 ≤ j ≤ k and A i ∈ R n × n for j ≤ i ≤ k . We define Π k i = j A i = 1 and ∑ k i = j A i = 0 if 0 ≤ k &lt; j .

## E.2 PROPOSITIONS

Proposition 1. B 1 ≤ A ≤ B 2 implies ‖ A ‖ ∞ ≤ max {‖ B 1 ‖ ∞ , ‖ B 2 ‖ ∞ }

Proof. By definition of ‖·‖ ∞

<!-- formula-not-decoded -->

Proposition 2. If P 1 , P 2 are stochastic matrices and 0 &lt; a, b , there exist stochastic matrix P such that aP 1 + bP 2 = ( a + b ) P .

Proof. Define P ( i, j ) = ( a + b ) -1 ( aP 1 ( i, j ) + bP 2 ( i, j )) . Then, by simple calculation, we get the desired result.

Proposition 3. (Bertsekas, 2012, Lemma 1.1.1) If V ≤ ˜ V , then T π U ≤ T π ˜ V , T /star V ≤ T /star ˜ V .

Proposition 4. For any policy π , P π is a nonexpansive linear operator such that if V ≤ ˜ V , P π V ≤ P π ˜ V .

Proof. If r ( s, a ) = 0 for all s ∈ S and a ∈ A , T π = P π . Then by Proposition 3, we have the desired result.

Proposition 5. For a stochastic matrix P , P /star P = PP /star = P /star .

Proof. By definition of

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F OMITTED PROOFS OF THEOREMS FOR SECTION 3 AND B

In this section, we present omitted proofs convergence theorems of Rx-VI. We prove Theorems 7, 8, and 1 in turn.

## F.1 PROOF OF THEOREM 7

First, we prove the following lemma by induction.

Lemma 1. For the iterates { V k } k =1 , 2 ,... of Rx-VI,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By induction,

<!-- formula-not-decoded -->

Now, we prove following key lemma.

<!-- formula-not-decoded -->

Proof. For the first inequality, we have V k

<!-- formula-not-decoded -->

where first equality comes form Lemma 1, first inequality follows from second Bellman equation, second inequality follows from first Bellman equation, and second equality is from telescoping-sum argument.

We now prove the second inequality.

V

k

<!-- formula-not-decoded -->

where first inequality follows from Lemma 3 and the fact that { π l } l =0 , 1 ,...,k are greedy policies, first equality comes from second Bellman equation, and second equality is from first Bellman equation.

We are now ready to prove Theorem 7 .

<!-- formula-not-decoded -->

Proof of Theorem 7 . By Lemma 2, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

By Proposition 1, this implies

<!-- formula-not-decoded -->

Hence, we conclude

<!-- formula-not-decoded -->

## F.2 PROOF OF THEOREM 8

First, define a k j = ( Π k i = j +1 λ i ) (1 -λ j ) for 0 ≤ j ≤ k and a 0 0 = 1 , where { λ k } k ∈ N ∈ [0 , 1] . Let λ 0 = 0 for computational conciseness. Following lemma will simplify calculation in later proof.

Lemma 3. For 0 ≤ k 2 &lt; k 1 ,

<!-- formula-not-decoded -->

Proof. First and second equality can be proved by simple calculation. With first and second equality, we prove third equality as follows.

<!-- formula-not-decoded -->

By simple calculation, for the iterates { V k } k =0 , 1 , 2 ,... of Rx-VI,

<!-- formula-not-decoded -->

where TV -1 = V 0 , and we have following lemma.

Lemma 4. For the iterates { V k } k =0 , 1 , 2 ,... of Rx-VI,

<!-- formula-not-decoded -->

for 0 ≤ k 2 ≤ k 1 .

Proof. By definition, we have

<!-- formula-not-decoded -->

where third equality is from Lemma 3.

Following lemma will be used in proof in later proof.

Lemma 5. For the iterates { V k } k =0 , 1 , 2 ,... of Rx-VI,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We have

<!-- formula-not-decoded -->

where first inequality is from Lemma 3 and 2, second inequality comes from second Bellman equation, and last equality follows from first Bellman equation.

Also, we have

<!-- formula-not-decoded -->

where first inequality is from Lemma 3 and 2 and the fact that { π l } l =0 , 1 ,...,k are greedy policies, first equality comes from second Bellman equation, and last equality follows from first Bellman equation.

We now prove one of key lemmas for Theorem 8. For that, define

<!-- formula-not-decoded -->

Lemma 6. For the iterates { V k } k =0 , 1 , 2 ,... of Rx-VI and 0 ≤ k 2 ≤ k 1 , for 0 ≤ k 2 &lt; k 1 and c n, -1 = 1 , c k,k = 0 for all 0 ≤ k . Note that a k j = ( Π k i = j +1 λ i ) (1 -λ j ) and a 0 0 = 1 for 0 ≤ j ≤ k . Then, we have following lemma.

<!-- formula-not-decoded -->

where S k 1 ,k 2 1 , S k 1 ,k 2 2 are stochastic matrices.

Proof. We use induction on k 2 . Let k 2 = 0 . Then, c k, 0 = ∑ k i =1 a k i = 1 -Π k i =1 λ i ( a 0 0 = 1) by Lemma 3. Also, by Lemma 2, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where first equality is from Lemma 4, first inequality comes from the fact that { π l } l =0 , 1 ,...,k 1 are greedy policies, second inequality follows from induction and Lemma 5, last equality is from Lemma 3, and

<!-- formula-not-decoded -->

To obtain lower bound of V k 1 -V k 2 , we need more sophisticated consideration, and following lemma is necessary for later argument.

Lemma 7. Let { V k } k =0 , 1 , 2 ,... be the iterates of Rx-VI. Let lim sup λ k &lt; 1 . Let E = { π : P π g /star = g /star } . Then there exist K such that if K ≤ k ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Suppose π is infinitely often repeated deterministic policy among { π k } k =0 , 1 , 2 ,... . Then there exist increasing sequence k n such that π k n = π and λ k n converge to some λ &lt; 1 . Then, since V k K +1 = λ k n +1 V n k +(1 -λ k n +1 ) TV k n , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If k n →∞ , lim sup λ k &lt; 1 implies ∑ ∞ j =1 (1 -λ j ) = ∞ . Then, by Theorem 7, we have

Thus g /star = P π g /star and this implies π ∈ E . By finiteness of action and state space, number of infinitely repeated policy π is also finite. Therefore there exist K such that TV k = max π ∈ E T π V k for K ≤ k .

We are now ready to prove left key lemma. To obtain proper lower bound of V k 1 -V k 2 , roughly speaking, we need to consider V K as initial point where N is iteration number in Lemma 13. For that, define { λ ′ k } K ≤ k such that λ ′ k = λ k for K + 1 ≤ k and λ ′ K = 0 , b k j = ( Π k i = j +1 λ ′ i ) (1 -λ ′ j ) for K ≤ j ≤ k , and b K K = 1 . Also, define

<!-- formula-not-decoded -->

for 0 ≤ k 2 &lt; k 1 and c K k,K -1 = 1 for all K ≤ k . Note that if K = 0 , λ ′ k = λ k , b k j = a k j , and c K k 1 ,k 2 = c k 1 ,k 2 for all 0 ≤ k 1 , k 2 , k, j, k .

Lemma 8. Let { V k } k =0 , 1 , 2 ,... be the iterates of Rx-VI. Suppose there exist K such that if K ≤ k , TV k = max π ∈ E T π V k where E = { π : P π g /star = g /star } . Then, for K ≤ k ′ 2 ≤ k ′ 1 ,

<!-- formula-not-decoded -->

where S k 1 ,k 2 1 ′ , S k 1 ,k 2 2 ′ are stochastic matrices.

Proof. For K ≤ k , by simple calculation, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where first equality comes from previous equality, first inequality follows the fact that { π l } l = K,K +1 ,...,k are greedy policies, second equality comes from second Bellman equation, and last equality is from first Bellman equation.

Now, we use induction on k 2 . If k 2 = K , by previous inequality,

<!-- formula-not-decoded -->

where second inequality comes from Lemma 2 and first Bellman equation (note that g /star terms cancel out), and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By induction,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where first equality is from similar argument in the proof of Lemma 4, first inequality comes from the fact that { π l } l = K,K +1 ,...,k 1 are greedy policies, second inequality follows from induction and simlilar argument in the proof of Lemma 5, last inequality is from Lemma 2 and first Bellman equation (note that g /star terms cancel out), second from the last equality is from same argument in the proof of Lemma 3, and

<!-- formula-not-decoded -->

For the explicit convergence rate of Theorem 8, we will use the following Fact.

Fact 5. (Cominetti et al., 2014, Section 2.3) For 0 &lt; k and K ≤ k ′ ,

Now, we are ready to prove Theorem 8.

<!-- formula-not-decoded -->

Proof of Theorem 8. First, by Lemma 6, we have

<!-- formula-not-decoded -->

Similarly, by Lemma 8, we have

<!-- formula-not-decoded -->

Thus, this two inequality implies that

<!-- formula-not-decoded -->

by Fact 5 and λ k = λ ′ k for K + 1 ≤ k . Finally, by applying the Proposition 6, we conclude proof.

## F.3 PROOF OF THEOREM 1

Let S be set of all deterministic policies and /epsilon1 = inf π ∈ S/ { π | P π g /star = g /star } ‖P π g /star -g /star ‖ ∞ (note that if S/ { π | P π g /star = g /star } = ∅ , /epsilon1 = ∞ ). By definition of Bellman optimality operator, there exist deterministic policy π k such that. By definition of Bellman optimality operator, there exist deterministic π such that

<!-- formula-not-decoded -->

for all k . By simple calculation, this is equivalent to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and this implies

Then, we have

This implies

<!-- formula-not-decoded -->

Then, if we take ‖·‖ ∞ in both sides of previous equality,

<!-- formula-not-decoded -->

∥ ∥ ∥ ∥ Thus, if we set K = ( 2 ‖ r ‖ ∞ +4 ∥ ∥ V 0 ∥ ∥ ∞ +16 ∥ ∥ V 0 -h /star ∥ ∥ ∞ +2 ‖ g /star ‖ ∞ ) /epsilon1 -1 , K satisfied conditions of Theorem 8. Therefore, by Theorem 8 with λ i = 1 / 2 for all i , we obtain desired rate of Bellman and policy errors.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## G OMITTED PROOFS OF SECTION 4 AND C

In this section, we present omitted proofs convergence theorems of Anc-VI. We prove Theorem 9, 10, and 2 in turn.

## G.1 PROOF OF THEOREM 9

Define λ 0 = 1 as coefficient of Anc-VI for computational conciseness.

First, we prove the following lemma by induction.

Lemma 9. For the iterates { V k } k =0 , 1 ,... of Anc-VI,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By induction,

<!-- formula-not-decoded -->

Now, we prove following key lemma.

Lemma 10. For the iterates { V k } k =0 , 1 ,... of Anc-VI,

<!-- formula-not-decoded -->

Proof. For the first inequality, we have

<!-- formula-not-decoded -->

where first equality follows from Lemma 9, first inequality comes from second Bellman equation, and second inequality is from first Bellman equation and telescoping-sum argument.

We now prove second inequality.

<!-- formula-not-decoded -->

where first inequality follows from the Lemma 3 and fact that { π l } l =0 , 1 ,...,k are greedy policies, first equality comes from second Bellman equation, and second equality is from first Bellman equation.

We now prove Theorem 9.

Proof of Theorem 9 . By Lemma 10, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we take ‖·‖ ∞ right side of first and second inequality, we have and this implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## G.2 PROOF OF THEOREM 10

Following lemma will be used in proof in later proof.

Lemma 11. For the iterates { V k } k =0 , 1 ,... of Anc-VI,

<!-- formula-not-decoded -->

Proof. For the first inequality, We have TV k

<!-- formula-not-decoded -->

where first inequality is from Lemma 10 and second inequality comes from Bellman equations. Now, we prove the second inequality.

TV k

<!-- formula-not-decoded -->

where first inequality is from Lemma 10 and the fact that π k is greedy policy and first equality comes from Bellman equations.

We now prove one of key lemmas.

Lemma 12. For the iterates { V k } k =1 , 2 ,... of Anc-VI,

<!-- formula-not-decoded -->

where S k 1 , S k 2 are stochastic matrices.

Proof. We use induction. If k = 1 ,

<!-- formula-not-decoded -->

where inequality follows from second Bellman equation.

By induction,

<!-- formula-not-decoded -->

where first inequality comes from the fact that π k , π k -1 are greedy policies, second inequality follows from induction and Lemma 11, last inequality is from the second Bellman equation, and

<!-- formula-not-decoded -->

Note that condition of leading coefficients positive.

To obtain lower bound of V k -V k -1 , we need more sophisticated consideration, and following lemma is necessary for later argument.

Lemma 13. Let { V k } k =0 , 1 , 2 ,... be the iterates of Anc-VI. Let λ k ≤ λ k -1 for 1 ≤ k and lim λ k = 0 . Let E = { π : P π g /star = g /star } . Then there exist K such that if K ≤ k ,

<!-- formula-not-decoded -->

Proof. Suppose π is infinitely often repeated deterministic policy among { π k } k =0 , 1 , 2 ,... . Then there exist increasing sequence k n such that π k n = π . Then, since V k n +1 = λ k n +1 V 0 + (1 -

λ k n +1 ) TV k n , we have

<!-- formula-not-decoded -->

If k n → ∞ , lim λ k = 0 implies lim k →∞ ∑ k i =0 Π k j = i (1 -λ j ) = ∞ by Lemma 14. Then, by Theorem 7, we have g /star = P π g /star , and this implies π ∈ E . By finiteness of action and state space, number of infinitely repeated policy π is also finite. Therefore there exist K such that TV k = max π ∈ E T π V k for K ≤ k .

<!-- formula-not-decoded -->

Proof. By condition, for any /epsilon1 &gt; 0 , there exist K /epsilon1 such that 1 -λ k &gt; 1 -/epsilon1 if K /epsilon1 ≤ k . Hence, lim inf k →∞ ∑ k i =0 Π k j = i (1 -λ j ) ≥ 1 //epsilon1 -1 . This concludes lemma.

The following lemma will be used in the proof of key lemma.

Lemma 15. Let { V k } k =1 , 2 ,... be the iterates of Anc-VI. For k ≤ K +1 ,

<!-- formula-not-decoded -->

where S k 1 ′ , S k 2 ′ , S k 3 ′ are stochastic matrices.

Proof. We use induction. If k = 1 , V 1 -V 0 = (1 -λ 1 )( TV 0 -V 0 ) ≥ (1 -λ 1 ) g /star +(1 -λ 1 )( P π /star -I )( V 0 -h /star ) .

By induction,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where first inequality comes from the fact that π k , π k -1 are greedy policies, second inequality follows from induction and Lemma 11, and

<!-- formula-not-decoded -->

(Note that λ k λ k +1 0 implies 1 λ k k Π k (1 λ j ) λ i 0 . )

<!-- formula-not-decoded -->

Now, we prove left key lemma.

Lemma 16. Let { V k } k =0 , 1 , 2 ,... be the iterates of Anc-VI. Suppose there exist K such that if K ≤ k , TV k = max π ∈ E T π V k where E = { π : P π g /star = g /star } . Then, for K + 1 ≤ k , For the iterates { V k } k = K +1 ,K +2 ,... of Anc-VI,

<!-- formula-not-decoded -->

where S k 1 ′ , S k 2 ′ , S k 3 ′ , S k 4 ′ are stochastic matrices.

Proof. We use induction. If k = K +1 , by Lemma 15, we have

<!-- formula-not-decoded -->

where S K +1 4 ′ = I .

By induction, for k ≥ K +2 ,

<!-- formula-not-decoded -->

where first inequality comes from the fact that π k , π k -1 are greedy policies, second inequality follows from Lemma 11 and induction, last equality is from first Bellman equation, and

<!-- formula-not-decoded -->

Now, we prove Theorem 10.

Proof of Theorem 10. First, we have

<!-- formula-not-decoded -->

where first inequality comes from the fact that π k , π k -1 are greedy policies, second inequality follows from induction and Lemma 11 and 12, and last equality is from the second Bellman equation.

Similarly,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where first inequality comes from the fact that π k , π k -1 are greedy policies, second inequality follows from Lemma 11 and 16, and last equality is from the second Bellman equation.

If we take ‖·‖ ∞ right side of first and second inequality, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since Π k j = K +1 (1 -λ j ) ( 1 -∑ K +1 i =1 λ K Π K j = i (1 -λ j ) ) ≤ Π k j = K (1 -λ j ) . Finally, by applying the Proposition 6, we conclude proof.

## G.3 PROOF OF THEOREM 2

Let S be set of all deterministic policies and /epsilon1 = inf π ∈ S/ { π | P π g /star = g /star } ‖P π g /star -g /star ‖ ∞ (note that if S/ { π | P π g /star = g /star } = ∅ , /epsilon1 = ∞ ). By definition of Bellman optimality operator, there exist deterministic policy π k such that

<!-- formula-not-decoded -->

for all k . By simple calculation, this is equivalent to

<!-- formula-not-decoded -->

Let V k -V 0 k/ 3 = g /star + /epsilon1 k . By Theorem 9 with λ k = 2 k +2 , we have and this implies

Then, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies

<!-- formula-not-decoded -->

Then, if we take ‖·‖ ∞ in both sides of previous equality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∥ ∥ Thus, if we set K = ( 3 ‖ r ‖ ∞ +12 ∥ ∥ V 0 -h /star ∥ ∥ ∞ +3 ‖ g /star ‖ ∞ ) /epsilon1 -1 , K satisfied conditions of Theorem 8. Therefore, by Theorem 8 with λ i = 2 i +2 for all i , we have desired rate of Bellman and policy errors.

## H OMITTED PROOFS IN SECTION 5

## H.1 PROOF OF THEOREM 3

First, we prove the case V 0 = 0 for n ≥ k +2 . Consider the MDP ( S , A , P, r ) such that

<!-- formula-not-decoded -->

where { s 1 , . . . , s n -1 } is closed irreducible set and { s n } is transient set. Thus, given MDP is unichain. Moreover, T = P π U +[1 , 0 , . . . , 0] ᵀ , and since ( P π ) m = ( P π ) m + n for 1 ≤ m ≤ n -1 , g /star = lim k →∞ ∑ k i =0 ( P π ) i k r π = [1 / ( n -1) , . . . , 1 / ( n -1)] ᵀ and h /star = [( n -1) / (2 n -2) , ( n -3) / (2 n -2) , . . . , -( n -3) / (2 n -2) , -( n -1) / (2 n -2)] ᵀ satisfy modified Bellman equation. Therefore, ∥ V 0 -h /star ∥ ∞ = 1 / 2 , and under the span condition, we can show following lemma.

∥ ∥ Lemma 17. Let T : R n → R n be defined as before. Then, under span condition, ( V i ) j = 0 for 0 ≤ i ≤ k , i +1 ≤ j ≤ n .

Proof. We use induction. Case i = 0 is obvious. By induction, ( V l ) j = 0 for 0 ≤ l ≤ i -1 , l + 1 ≤ j ≤ n . Then ( TV l ) j = 0 for 0 ≤ l ≤ i -1 , l + 2 ≤ j ≤ n and this implies that ( TV l -V l ) j = 0 for 0 ≤ l ≤ i -1 , l +2 ≤ j ≤ n . Therefore, ( V i ) j = 0 for i +1 ≤ j ≤ n .

Thus, under the span condition, for i ≤ k , we get and this implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then for 0 ≤ i ≤ k . If ∑ k i =0 a i = 1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and taking the absolute value on both sides,

<!-- formula-not-decoded -->

Since ( TV i -V i ) l = 0 for k +2 ≤ l , we have

Therefore, this implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∥ ∥ Since g /star = [1 / ( n -1) , 1 / ( n -1) , . . . , 1 / ( n -1)] . we conclude

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, we show that for any initial point V 0 ∈ R n , there exists an MDP which exhibits same lower bound with the case V 0 = 0 . Denote by MDP( 0 ) and T 0 the worst-case MDP and Bellman optimality operator constructed for V 0 = 0 . Define an MDP( V 0 ) ( S , A , P, r ) for V 0 = 0 as

Then, Bellman optimality operator T satisfies

/negationslash

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let ˜ g /star be average reward of T 0 and ˜ h /star solution of optimlaity equation. Then, since lim k →∞ ∑ k i =0 ( P π ) i k ( I - P π ) = 0 , g /star = ˜ g /star is average reward of T and h /star = V 0 + ˜ h /star is also solution of Bellman equation. Furthermore, if { V i } k i =0 satisfies span condition

<!-- formula-not-decoded -->

which is the same span condition in Theorem 4 with respect to T 0 . This is because

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for i = 0 , . . . , k . Thus, { ˜ U i } k i =0 is a sequence starting from 0 and satisfy the span condition for T 0 . This implies that

Hence, MDP( V 0 ) is indeed our desired worst-case instance.

<!-- formula-not-decoded -->

## H.2 PROOF OF THEOREM 4

We now present the proof of Theorem 4.

Proof of Theorem 4. First, we prove the case V 0 = 0 for n ≥ k +3 . Consider the MDP ( S , A , P, r ) such that

<!-- formula-not-decoded -->

where { s 1 } , { s n } are closed irreducible sets and { s 2 , . . . , s n -1 } is transient set. Thus, given MDP is multichain. Morevoer, T = P π U +[0 , 1 , 0 , . . . , 0 , 1] ᵀ , and since ( P π ) m = ( P π ) k +1 for m ≥ k +1 , g /star = lim k →∞ ∑ k i =0 ( P π ) i k r π = [0 , . . . , 0 , 1] ᵀ and h /star = [ -1 / 2 , 1 / 2 , 1 / 2 , . . . , 1 / 2 , 0] ᵀ which satisfy Bellman equation. Thus, ∥ ∥ V 0 -h /star ∥ ∥ ∞ = 1 / 2 . Under the span condition, we can show following lemma.

Proof. We use induction. Case i = 0 is obvious. By induction, ( V l ) 1 = 0 for 0 ≤ l ≤ i -1 . Then ( TV l ) 1 = 0 for 0 ≤ l ≤ i -1 . This implies that ( TV l -V l ) 1 = 0 for 0 ≤ l ≤ i -1 . Hence ( V i ) 1 = 0 . Again, by induction, ( V l ) j = 0 for 0 ≤ l ≤ i -1 , l +2 ≤ j ≤ n -1 . Then ( TV l ) j = 0 for 0 ≤ l ≤ i -1 , l +3 ≤ j ≤ n -1 and this implies that ( TV l -V l ) j = 0 for 0 ≤ l ≤ i -1 , l +3 ≤ j ≤ n -1 . Therefore, ( V i ) j = 0 for i +2 ≤ j ≤ n -1 .

Lemma 18. Let T : R n → R n be defined as before. Then, under span condition, ( V i ) 1 = 0 for 0 ≤ i ≤ k , and ( V i ) j = 0 for 0 ≤ i ≤ k and i +2 ≤ j ≤ n -1 .

Thus, under the span condition, for 0 ≤ i ≤ k , we get

<!-- formula-not-decoded -->

where g /star = e n . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for 0 ≤ i ≤ k . If ∑ k i =0 a i = 1 , we have and taking the absolute value on both sides,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∣ ∣ Since ( TV i -V i -g /star ) l = 0 for l = 1 and k +3 ≤ l , we have

Therefore, we conclude

With the same argument in proof of Theorem 3, we can extend this result to arbitrary V 0 .

<!-- formula-not-decoded -->

## I OMITTED PROOFS IN SECTION 6 AND D

## I.1 PROOF OF THEOREM 11

Consider Rx-VI and V 0 = h 0 . Let { V k } k =0 , 1 , 2 ,... be the iterates of Rx-VI. Then, since T ( v + c 1 ) = c 1 + T ( v ) for arbitrary v ∈ R n and c ∈ R , V k = h k + c k 1 for some c k ∈ R . This implies TV k -V k = Th k -h k and by Corollary 4, we have

Now, consider following iteration

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for 1 ≤ k where g /star is average reward of T and V 0 g = h 0 . Then, ∥ ∥ V k g -h /star ∥ ∥ ∞ ≤ ∥ ∥ V k -1 g -h /star ∥ ∥ ∞ for 1 ≤ k where h /star is solution of Bellman equation. This is because

∥ ∥ where second inequality is from the fact that h /star is fixed point of nonexpansive operator T ( · ) + g /star . This implies that V k g is bounded. Then, there exist convergent subsequence V k n g which converges to some V g ∈ R d . Since g is uniform constant vector, using previous argument and Theorem 8, we have ∥ ∥ TV k g -V k g -g /star ∥ ∥ ∞ ≤ 2 √ π ∑ k j =1 λ i (1 -λ i ) ∥ ∥ V 0 g -h /star ∥ ∥ ∞ by condition of λ k . This implies that

V g is solution of Bellman equation, and since ∥ ∥ V k g -V g ∥ ∥ ∞ ≤ ∥ ∥ V k -1 g -V g ∥ ∥ ∞ , we have V k g → V g . By previous argument, there exist c k ∈ R such that V k g = h k + c k 1 for 0 ≤ k where c 0 = 0 . Also, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where last equality comes from the property of f . This implies c k = λ k c k -1 +(1 -λ k )( f ( V k -1 g ) -ˆ g ) = ∑ k j =1 (Π k i = j +1 λ i )(1 -λ j )( f ( V j -1 g ) -ˆ g ) where g /star = ˆ g 1 . We now prove c k → f ( V g ) -ˆ g . Since f is a continuous function and V k g → V g , ∣ ∣ f ( V k g ) -ˆ g ∣ ∣ ≤ M for some 0 &lt; M , and there exist K such that | f ( V g ) -g -( f ( V k g ) -ˆ g ) | &lt; /epsilon1 for any 0 &lt; /epsilon1 and all K ≤ k . For K +1 ≤ k , we have

<!-- formula-not-decoded -->

Thus, as k →∞ , c k → f ( V g ) -ˆ g if lim sup λ j &lt; 1 . This implies h k converges to h /star solution of Bellman equations, and we have h /star = Th /star -f ( h ) 1 . By uniqueness of g /star , f ( h /star ) 1 = g /star .

## I.2 PROOF OF THEOREM 5

If λ i = 1 / 2 for all i , it satisfies condition of Theorem 11. Then, by Theorem 11 with λ i = 1 / 2 , we get the desired result.

## I.3 PROOF OF THEOREM 12

Consider Anc-VI and V 0 = h 0 . Let { V k } k =0 , 1 , 2 ,... be the iterates of Anc-VI. Then, since T ( v + c 1 ) = c 1 + T ( v ) for arbitrary v ∈ R n and c ∈ R , V k = h k + c k 1 for some c k ∈ R . This implies TV k -V k = Th k -h k and by Corollary 5, we have

Consider following iteration

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for 1 ≤ k where g /star is average reward of T and V 0 g = h 0 . Then, ∥ ∥ V k g -h /star ∥ ∥ ∞ ≤ ∥ ∥ V 0 g -h /star ∥ ∥ ∞ . By induction, we have

∥ ∥ where second inequality is from the fact that h /star is fixed point of nonexpansive operator of T ( · ) -g /star and last inequality comes from induction. This implies that V k g is bounded.

By previous argument, there exist c k ∈ R such that V k g = h k + c k 1 for 0 ≤ k since g /star is uniform constant vector. Also, we have

<!-- formula-not-decoded -->

where last equality comes from the property of f . This implies c k = (1 -λ k )( f ( V k -1 g ) -g /star ) . Since f is a continuous function, the boundedness of V k g implies the boundedness of c k and this also implies boundedness of h k .

Now for convergence of h k , we use folloiwng fact.

Fact 6 (Classical result, (Schweitzer &amp; Federgruen, 1978, Remark 3)) . If MDP is unichain and h is solution of modified Bellman equations, h = h /star + c 1 for arbitary c ∈ R and some fixed solution of modified Bellman equations h /star .

Suppose there exist convergent subsequence h k n of h k . Then, there also exist subsequence h k ′ n -1 of h k n -1 which converges to some h . By the previous convergence result, h must be solution of modified Bellman equation, and by Fact 6, h = h /star + c 1 for some constant c ∈ R and h /star a fixed solution of modified Bellman equation. This implies λ k ′ n -1 ( h k ′ n -1 ) + (1 -λ k ′ n -1 )( Th k ′ n -1 -f ( h k ′ n -1 ) 1 ) → Th /star + f ( h /star ) 1 . Since Th /star + f ( h /star ) 1 is fixed, h k is converge to h where h = Th -f ( h ) 1 . By uniqueness of g /star , f ( h /star ) 1 = g /star .

## I.4 PROOF OF THEOREM 6

If λ i = 2 / ( i + 2) for all i , it satisfies condition of Theorem 12. Then, by Theorem 12 with λ i = 2 / ( i +2) , we get the desired result.