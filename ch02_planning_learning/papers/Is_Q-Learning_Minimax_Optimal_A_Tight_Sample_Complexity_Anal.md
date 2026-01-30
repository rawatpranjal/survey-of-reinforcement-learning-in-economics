## Is Q-Learning Minimax Optimal? A Tight Sample Complexity Analysis

Gen Li ∗ UPenn

Changxiao Cai † UPenn

Yuxin Chen ∗ UPenn

Yuting Wei ∗ UPenn

Yuejie Chi ‡ CMU

February 2021;

Revised: October 2022

## Abstract

Q-learning, which seeks to learn the optimal Q-function of a Markov decision process (MDP) in a model-free fashion, lies at the heart of reinforcement learning. When it comes to the synchronous setting (such that independent samples for all state-action pairs are drawn from a generative model in each iteration), substantial progress has been made towards understanding the sample efficiency of Q-learning. Consider a γ -discounted infinite-horizon MDP with state space S and action space A : to yield an entrywise ε -approximation of the optimal Q-function, state-of-the-art theory for Q-learning requires a sample size exceeding the order of |S||A| (1 -γ ) 5 ε 2 , which fails to match existing minimax lower bounds. This gives rise to natural questions: what is the sharp sample complexity of Q-learning? Is Q-learning provably sub-optimal? This paper addresses these questions for the synchronous setting: (1) when |A| = 1 (so that Q-learning reduces to TD learning), we prove that the sample complexity of TD learning is minimax optimal and scales as |S| (1 -γ ) 3 ε 2 (up to log factor); (2) when |A| ≥ 2 , we settle the sample complexity of Q-learning to be on the order of |S||A| (1 -γ ) 4 ε 2 (up to log factor). Our theory unveils the strict sub-optimality of Q-learning when |A| ≥ 2 , and rigorizes the negative impact of over-estimation in Q-learning. Finally, we extend our analysis to accommodate asynchronous Q-learning (i.e., the case with Markovian samples), sharpening the horizon dependency of its sample complexity to be 1 (1 -γ ) 4 .

Keywords: Q-learning, temporal difference learning, effective horizon, sample complexity, minimax optimality, lower bound, over-estimation

## Contents

| 1   | Introduction                                              | Introduction                                                    |   2 |
|-----|-----------------------------------------------------------|-----------------------------------------------------------------|-----|
|     | 1.1                                                       | Main contributions . . . . . . . . . . . . . . . . . . . . . .  |   3 |
|     | 1.2                                                       | Related works . . . . . . . . . . . . . . . . . . . . . . . . . |   4 |
| 2   | Background and algorithms                                 | Background and algorithms                                       |   5 |
| 3   | Main results: sample complexity of synchronous Q-learning | Main results: sample complexity of synchronous Q-learning       |   7 |
|     | 3.1                                                       | Minimax optimality of TD learning . . . . . . . . . . . . .     |   7 |
|     | 3.2                                                       | Tight sample complexity and sub-optimality of Q-learning        |   9 |
| 4   | Key analysis ideas (the synchronous case)                 | Key analysis ideas (the synchronous case)                       |  11 |
|     | 4.1                                                       | Vector and matrix notation . . . . . . . . . . . . . . . . .    |  11 |
|     | 4.2                                                       | Proof outline for Theorem 2 . . . . . . . . . . . . . . . . .   |  12 |
|     | 4.3                                                       | Proof outline for Theorem 3 . . . . . . . . . . . . . . . . .   |  14 |

∗ Department of Statistics and Data Science, Wharton School, University of Pennsylvania, Philadelphia, PA 19104, USA.

† Department of Biostatistics, University of Pennsylvania, Philadelphia, PA 19104, USA.

‡ Department of Electrical and Computer Engineering, Carnegie Mellon University, Pittsburgh, PA 15213, USA.

| 5 Extension: sample complexity of asynchronous Q-learning   | 5 Extension: sample complexity of asynchronous Q-learning         | 5 Extension: sample complexity of asynchronous Q-learning         | 16   |
|-------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|------|
|                                                             | 5.1                                                               | Markovian samples and asynchronous Q-learning                     | 16   |
|                                                             | 5.2                                                               | Sample complexity of asynchronous Q-learning .                    | 17   |
| 6                                                           | Concluding                                                        | remarks                                                           | 18   |
| A                                                           | Freedman's inequality                                             | Freedman's inequality                                             | 18   |
| B                                                           | Upper bounds for Q-learning (Theorem 2)                           | Upper bounds for Q-learning (Theorem 2)                           | 19   |
|                                                             | B.1                                                               | Preliminaries . . . . . . . . . . . . . . . . . . . .             | 20   |
|                                                             | B.2                                                               | Proof of Lemma 1 . . . . . . . . . . . . . . . . .                | 21   |
|                                                             | B.3                                                               | Proof of Lemma 2 . . . . . . . . . . . . . . . . .                | 26   |
|                                                             | B.4                                                               | Solving the recurrence relation regarding ∆ t . . .               | 26   |
|                                                             | B.5                                                               | Proof of Lemma 5 . . . . . . . . . . . . . . . . .                | 28   |
| C                                                           | Analysis for TD learning (Theorem 1)                              | Analysis for TD learning (Theorem 1)                              | 30   |
|                                                             | C.1                                                               | Preliminary facts . . . . . . . . . . . . . . . . . .             | 30   |
|                                                             | C.2                                                               | Proof of Theorem 8 . . . . . . . . . . . . . . . . .              | 32   |
|                                                             | C.3                                                               | Proof for Remarks 2 and 3 . . . . . . . . . . . . .               | 35   |
| D                                                           | Lower bound: sub-optimality of synchronous Q-learning (Theorem 3) | Lower bound: sub-optimality of synchronous Q-learning (Theorem 3) | 36   |
|                                                             | D.1                                                               | Key quantities related to learning rates . . . . .                | 36   |
|                                                             | D.2                                                               | Preliminary calculations . . . . . . . . . . . . . .              | 37   |
|                                                             | D.3                                                               | Lower bounds for three cases . . . . . . . . . . .                | 39   |
|                                                             | D.4                                                               | Proof of Lemma 3 . . . . . . . . . . . . . . . . .                | 52   |
| E                                                           | Analysis for asynchronous Q-learning (Theorem 4)                  | Analysis for asynchronous Q-learning (Theorem 4)                  | 53   |
|                                                             | E.1 Notation and preliminary facts                                | . . . . . . . . . .                                               | 53   |
|                                                             | E.2                                                               | Main steps for proving Theorem 4 . . . . . . . .                  | 53   |
|                                                             | E.3                                                               | Proofs of technical lemmas . . . . . . . . . . . . .              | 58   |
|                                                             | F Lower bound for asynchronous Q-learning (Theorem 5)             | F Lower bound for asynchronous Q-learning (Theorem 5)             | 60   |
|                                                             | its values                                                        | . .                                                               |      |
|                                                             | F.1 Construction of a hard instance and . .                       | F.1 Construction of a hard instance and . .                       | 60   |
|                                                             | F.2                                                               | Analysis for the constructed MDP . . . . . .                      | 62   |

## 1 Introduction

Q-learning is arguably one of the most widely adopted model-free algorithms (Watkins and Dayan, 1992; Watkins, 1989). Characterizing its sample efficiency lies at the core of the statistical foundation of reinforcement learning (RL) (Sutton and Barto, 2018). While classical convergence analyses for Q-learning (Borkar and Meyn, 2000; Jaakkola et al., 1994; Szepesvári, 1998; Tsitsiklis, 1994) focused primarily on the asymptotic regime-in which the number of iterations tends to infinity with other problem parameters held fixedrecent years have witnessed a paradigm shift from asymptotic analyses towards a finite-sample / finite-time framework (Beck and Srikant, 2012; Chen et al., 2020, 2021; Even-Dar and Mansour, 2003; Kearns and Singh, 1999; Lee and He, 2018; Li et al., 2022c; Qu and Wierman, 2020; Wainwright, 2019b; Weng et al., 2020a; Xiong et al., 2020). Drawing insights from high-dimensional statistics (Wainwright, 2019a), a modern non-asymptotic framework unveils more clear and informative impacts of salient problem parameters upon the sample complexity, particularly for those applications with enormous state/action space and long horizon. Motivated by its practical value, a suite of non-asymptotic theory has been recently developed for Q-learning to accommodate multiple sampling mechanisms (Beck and Srikant, 2012; Even-Dar and Mansour, 2003; Jin et al., 2018; Li et al., 2022c; Qu and Wierman, 2020; Wainwright, 2019b).

In this paper, we revisit the sample complexity of Q-learning for tabular Markov decision processes (MDPs). For concreteness, let us consider the synchronous setting, which assumes access to a generative model or a simulator that produces independent samples for all state-action pairs in each iteration (Kakade,

Table 1: Comparisons of existing sample complexity upper bounds of synchronous Q-learning and TD learning for an infinite-horizon γ -discounted MDP with state space S and action space A , where 0 &lt; ε &lt; 1 is the target accuracy level. Here, sample complexity refers to the total number of samples needed to yield either max s,a | ̂ Q ( s, a ) -Q glyph[star] ( s, a ) | ≤ ε with high probability or E [ max s,a | ̂ Q ( s, a ) -Q glyph[star] ( s, a ) | ] ≤ ε , where ̂ Q is the estimate returned by Q-learning. All logarithmic factors are omitted in the table to simplify the expressions.

| paper                             | learning rates                              | sample complexity                                          |
|-----------------------------------|---------------------------------------------|------------------------------------------------------------|
| Even-Dar and Mansour (2003)       | linear: 1 t                                 | 2 1 1 - γ |S||A| (1 - γ ) 4 ε 2                            |
| Even-Dar and Mansour (2003)       | polynomial: 1 t ω , ω ∈ (1 / 2 , 1)         | |S||A| { ( 1 (1 - γ ) 4 ε 2 ) 1 /ω + ( 1 1 - γ ) 1 1 - ω } |
| Beck and Srikant (2012)           | constant: (1 - γ ) 4 ε 2 |S||A|             | |S| 2 |A| 2 (1 - γ ) 5 ε 2                                 |
| Wainwright (2019b)                | rescaled linear: 1 1+(1 - γ ) t             | |S||A| (1 - γ ) 5 ε 2                                      |
| Wainwright (2019b)                | polynomial: 1 t ω , ω ∈ (0 , 1)             | |S||A| { ( 1 (1 - γ ) 4 ε 2 ) 1 /ω + ( 1 1 - γ ) 1 1 - ω } |
| Chen et al. (2020)                | rescaled linear: 1 1 (1 - γ ) 2 +(1 - γ ) t | |S||A| (1 - γ ) 5 ε 2                                      |
| Chen et al. (2020)                | constant: (1 - γ ) 4 ε 2                    | |S||A| (1 - γ ) 5 ε 2                                      |
| this work (Q-learning, |A| ≥ 2 )  | rescaled linear: 1 1+(1 - γ ) t             | |S||A| (1 - γ ) 4 ε 2                                      |
| this work (Q-learning, |A| ≥ 2 )  | constant: (1 - γ ) 3 ε 2                    | |S||A| (1 - γ ) 4 ε 2                                      |
| this work (TD learning, |A| = 1 ) | rescaled linear: 1 1+(1 - γ ) t             | |S| (1 - γ ) 3 ε 2                                         |
| this work (TD learning, |A| = 1 ) | constant: (1 - γ ) 3 ε 2                    | |S| (1 - γ ) 3 ε 2                                         |

2003; Kearns et al., 2002); this setting is termed 'synchronous' as the estimates w.r.t. all state-action pairs are updated at once. We investigate the glyph[lscript] ∞ -based sample complexity, namely, the number of samples needed for synchronous Q-learning to yield an entrywise ε -accurate estimate of the optimal Q-function. Despite a number of prior works tackling this setting, the dependence of the sample complexity on the effective horizon 1 1 -γ remains unsettled. Take γ -discounted infinite-horizon MDPs for instance: the state-of-the-art sample complexity bounds (Chen et al., 2020; Wainwright, 2019b) scale on the order of |S||A| (1 -γ ) 5 ε 2 (up to some log factor), where S and A represent the state space and the action space, respectively. However, it is unclear whether this scaling is sharp for Q-learning, and whether it can be further improved via a more refined theory. On the one hand, the minimax lower limit for this setting has been shown to be on the order of |S||A| (1 -γ ) 3 ε 2 (up to some log factor) (Azar et al., 2013); this limit is achievable by model-based approaches (Agarwal et al., 2020; Li et al., 2023) and apparently smaller than prior sample complexity bounds for Q-learning. On the other hand, Wainwright (2019c) argued through numerical experiments that ' the usual Q-learning suffers from at least worst-case fourth-order scaling in the discount complexity 1 1 -γ , as opposed to the third-order scaling . . . ', although no rigorous justification was provided therein. Given the gap between the achievability bounds and lower bounds in the status quo, it is natural to seek answers to the following questions:

What is the tight sample complexity characterization of Q-learning?

How does it compare to the minimax sample complexity limit?

## 1.1 Main contributions

Focusing on γ -discounted infinite-horizon MDPs with state space S and action space A , this paper settles the glyph[lscript] ∞ -based sample complexity of synchronous Q-learning. Here and throughout, the standard notation f ( · ) = ˜ O ( g ( · )) (resp. f ( · ) = ˜ Ω( g ( · )) ) means that f ( · ) is orderwise no larger than (resp. no smaller than) g ( · )

modulo some logarithmic factors. Our main contributions regarding synchronous Q-learning are summarized below.

- When |A| = 1 , Q-learning coincides with temporal difference (TD) learning in a Markov reward process. For any 0 &lt; ε &lt; 1 , we prove that a total sample size of

<!-- formula-not-decoded -->

is sufficient for TD learning to guarantee ε -accuracy in an glyph[lscript] ∞ sense; see Theorem 1. This is sharp and minimax optimal (up to some log factor).

- Moving on to the case with |A| ≥ 2 , we demonstrate that a sample size of

<!-- formula-not-decoded -->

suffices for Q-learning to yield ε -accuracy in an glyph[lscript] ∞ sense for any 0 &lt; ε &lt; 1 ; see Theorem 2. Conversely, we construct a hard MDP instance with 4 states and 2 actions, for which Q-learning provably requires at least iterations to achieve ε -accuracy in an glyph[lscript] ∞ sense; see Theorem 3. These two theorems taken collectively lead to the first sharp characterization of the sample complexity of Q-learning, strengthening prior theory (Chen et al., 2020; Wainwright, 2019b) by a factor of 1 1 -γ . In addition, the discrepancy between our sharp characterization and the minimax lower bound makes clear that Q-learning is not minimax optimal when |A| ≥ 2 , and is outperformed by, say, the model-based approaches (Agarwal et al., 2020; Li et al., 2023) in terms of the sample efficiency.

<!-- formula-not-decoded -->

Our results cover both rescaled linear and constant learning rates; see Table 1 for more detailed comparisons with previous literature. On the technical side, (i) our analysis for the upper bound relies on a sort of crucial error decompositions and variance control that are previously unexplored, which might shed light on how to pin down the finite-sample efficacy of other variants of Q-learning such as double Q-learning; (ii) the development of our lower bound, which is inspired by Azar et al. (2013); Wainwright (2019c), puts the negative impact of over-estimation on sample efficiency on a rigorous footing.

Finally, we extend our analysis framework to accommodate the asynchronous setting, in which the samples are non-i.i.d. and take the form of a single Markovian trajectory. We show for the first time that the sample complexity of asynchronous Q-learning exhibits a 1 (1 -γ ) 4 scaling w.r.t. the effective horizon, which is nearly sharp and improves upon the prior state-of-the-art Li et al. (2022c).

## 1.2 Related works

There is a growing literature dedicated to analyzing the non-asymptotic behavior of value-based model-free RL algorithms in a variety of scenarios. In the discussion below, we subsample the literature and discuss a couple of papers that are the closest to ours.

Finite-sample glyph[lscript] ∞ -based guarantees for synchronous Q-learning and TD learning. The sample complexities derived in prior literature often rely crucially on the choices of learning rates. Even-Dar and Mansour (2003) studied the sample complexity of Q-learning with linear learning rates 1 /t or polynomial learning rates 1 /t ω , which scales as ˜ O ( |S||A| (1 -γ ) 5 ε 2 . 5 ) when optimized w.r.t. the effective horizon (attained when ω = 4 / 5 ). The resulting sample complexity, however, is sub-optimal in terms of its dependency on not only 1 1 -γ but also the target accuracy level ε . Beck and Srikant (2012) investigated the case of constant learning rates; however, their result suffered from an additional factor of |S||A| , which could be prohibitively large in practice. More recently, Chen et al. (2020); Wainwright (2019b) further analyzed the sample complexity of Q-learning with either constant learning rates or linearly rescaled learning rates, leading to the state-of-the-art bound ˜ O ( |S||A| (1 -γ ) 5 ε 2 ) . However, this result remains sub-optimal in terms of its scaling with 1 1 -γ . See Table 1

for details. In the special case with |A| = 1 , the recent works Khamaru et al. (2021a); Mou et al. (2020) developed instance-dependent results for TD learning with Polyak-Ruppert averaging, and studied the local (sub)-optimality of TD learning in a different local minimax framework.

Finite-sample glyph[lscript] ∞ -based guarantees for asynchronous Q-learning and TD learning. Moving beyond the synchronous model, Beck and Srikant (2012); Chen et al. (2021); Even-Dar and Mansour (2003); Li et al. (2022c); Qu and Wierman (2020); Shah and Xie (2018) developed non-asymptotic convergence guarantees for the asynchronous setting, where the data samples take the form of a single Markovian trajectory (following some behavior policy) and only a single state-action pair is updated in each iteration. A similar scaling of ˜ O ( 1 (1 -γ ) 5 ) also showed up in the state-of-the-art sample complexity bounds for asynchronous Q-learning (Li et al., 2022c), and our theory is the first to sharpen it to ˜ O ( 1 (1 -γ ) 4 ) . When it comes to the special case with |A| = 1 , the non-asymptotic performance guarantees for TD learning with Markovian sample trajectories (assuming that the behavior policy coincides with the target policy) have been recently derived by Bhandari et al. (2021); Mou et al. (2020); Srikant and Ying (2019).

Finite-sample glyph[lscript] ∞ -based guarantees of other Q-learning variants. With the aim of alleviating the sub-optimal dependency on the effective horizon in vanilla Q-learning and improving sample efficiency, several variants of Q-learning have been proposed and analyzed. Azar et al. (2011) proposed speedy Q-learning, which achieves a sample complexity of ˜ O ( |S||A| (1 -γ ) 4 ε 2 ) at the expense of doubling the computation and storage complexity. Our result on vanilla Q-learning matches that of speedy Q-learning in an order-wise sense. In addition, Wainwright (2019c) proposed a variance-reduced Q-learning algorithm that is shown to be minimax optimal in the range glyph[epsilon1] ∈ (0 , 1) with a sample complexity ˜ O ( |S||A| (1 -γ ) 3 ε 2 ) , which was subsequently generalized to the asynchronous setting by Li et al. (2022c). The glyph[lscript] ∞ statistical bounds for variance-reduced TD learning have been investigated in Khamaru et al. (2021a) for the synchronous setting, and in Li et al. (2022c) for the asynchronous setting. Last but not least, Xiong et al. (2020) established the finite-sample convergence of double Q-learning following the framework of Even-Dar and Mansour (2003); however, it is unclear whether double Q-learning can provably outperform vanilla Q-learning in terms of the sample efficiency.

Others. There are also several other strands of related papers that tackle model-free algorithms but do not pursue glyph[lscript] ∞ -based non-asymptotic guarantees. For instance, Bhandari et al. (2021); Chen et al. (2019); Doan et al. (2019); Gupta et al. (2019); Lakshminarayanan and Szepesvari (2018); Srikant and Ying (2019); Wu et al. (2020); Xu et al. (2019a,b) developed finite-sample (weighted) glyph[lscript] 2 convergence guarantees for several model-free algorithms, which also allow one to accommodate linear function approximation as well as off-policy evaluation. Another line of recent work (Bai et al., 2019; Jin et al., 2018; Li et al., 2021; Zhang et al., 2020) considered the sample efficiency of Q-learning type algorithms paired with proper exploration strategies (e.g., upper confidence bounds) under the framework of regret analysis. The asymptotic behaviors of some variants of Q-learning, e.g., double Q-learning (Weng et al., 2020b) and relative Q-learning (Devraj and Meyn, 2020) are also studied. In addition, Q-learning in conjunction with the pessimism principle has proven effective in dealing with offline data (Shi et al., 2022; Yan et al., 2022). The effect of more general function approximation schemes (e.g., certain families of neural network approximations) has been studied in Cai et al. (2019); Fan et al. (2019); Murphy (2005); Wai et al. (2019); Xu and Gu (2020), whereas the extension to multi-agent scenarios has been looked at in Hu and Wellman (2003); Li et al. (2022a). These are beyond the scope of the present paper.

## 2 Background and algorithms

This paper concentrates on discounted infinite-horizon MDPs (Bertsekas, 2017). We shall start by introducing some basics of tabular MDPs, followed by a description of both Q-learning and TD learning. Throughout this paper, we denote by S = { 1 , · · · , |S|} and A = { 1 , · · · , |A|} the state space and the action space of the MDP, respectively, and let ∆( S ) represent the probability simplex over the set S .

Basics of discounted infinite-horizon MDPs. Consider an infinite-horizon MDP as represented by a quintuple M = ( S , A , P, r, γ ) , where γ ∈ (0 , 1) indicates the discount factor, P : S × A → ∆( S ) represents the probability transition kernel (i.e., P ( s ′ | s, a ) is the probability of transiting to state s ′ from a state-action pair ( s, a ) ∈ S ×A ), and r : S ×A → [0 , 1] stands for the reward function (i.e., r ( s, a ) is the immediate reward collected in state s ∈ S when action a ∈ A is taken). Note that the immediate rewards are assumed to lie within [0 , 1] throughout this paper. Moreover, we let π : S → ∆( A ) represent a policy, so that π ( · | s ) ∈ ∆( A ) specifies the (possibly randomized) action selection rule in state s . If π is a deterministic policy, then we denote by π ( s ) the action selected by π in state s .

A common objective in RL is to maximize a sort of long-term rewards called value functions or Q-functions. Specifically, given a policy π , the associated value function and Q-function of π are defined respectively by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all s ∈ S , and for all ( s, a ) ∈ S × A . Here, { ( s k , a k ) } k ≥ 0 is a trajectory of the MDP induced by the policy π (except a 0 when evaluating the Q-function), and the expectations are evaluated with respect to the randomness of the MDP trajectory. Given that the immediate rewards fall within [0 , 1] , it can be straightforwardly verified that 0 ≤ V π ( s ) ≤ 1 1 -γ and 0 ≤ Q π ( s, a ) ≤ 1 1 -γ for any π and any state-action pair ( s, a ) . The optimal value function V glyph[star] and optimal Q-function Q glyph[star] are defined respectively as

<!-- formula-not-decoded -->

for any state-action pair ( s, a ) ∈ S × A . It is well known that there exists a deterministic optimal policy, denoted by π glyph[star] , that attains V glyph[star] ( s ) and Q glyph[star] ( s, a ) simultaneously for all ( s, a ) ∈ S × A (Sutton and Barto, 2018).

Algorithms: Q-learning and TD learning (the synchronous setting). The synchronous setting assumes access to a generative model (Kearns and Singh, 1999; Sidford et al., 2018) such that: in each iteration t , we collect an independent sample s t ( s, a ) ∼ P ( · | s, a ) for every state-action pair ( s, a ) ∈ S × A .

With this sampling model in place, the Q-learning algorithm (Watkins and Dayan, 1992) maintains a Q-function estimate Q t : S × A → R for all t ≥ 0 ; in each iteration t , the algorithm updates all entries of the Q-function estimate at once via the following update rule

<!-- formula-not-decoded -->

Here, η t ∈ (0 , 1] denotes the learning rate or the step size in the t -th iteration, and T t denotes the empirical Bellman operator constructed by samples collected in the t -th iteration, i.e.,

<!-- formula-not-decoded -->

for each state-action pair ( s, a ) ∈ S × A . Obviously, T t is an unbiased estimate of the celebrated Bellman operator T given by

<!-- formula-not-decoded -->

Note that the optimal Q-function Q glyph[star] is the unique fixed point of the Bellman operator (Bellman, 1952), that is, T ( Q glyph[star] ) = Q glyph[star] . Viewed in this light, synchronous Q-learning can be interpreted as a stochastic approximation scheme (Robbins and Monro, 1951) aimed at solving this fixed-point equation. Throughout this work, we initialize the algorithm in a way that obeys 0 ≤ Q 0 ( s, a ) ≤ 1 1 -γ for every state-action pair ( s, a ) . In addition, the corresponding value function estimate V t : S → R in the t -th iteration is defined as

<!-- formula-not-decoded -->

## Algorithm 1 Synchronous Q-learning for infinite-horizon discounted MDPs.

- 1: inputs: learning rates { η t } , number of iterations T , discount factor γ , initial estimate Q 0 .
- 3: Draw s t ( s, a ) ∼ P ( · | s, a ) for each ( s, a ) ∈ S × A .
- 2: for t = 1 , 2 , · · · , T do
- 4: Compute Q t according to (4) and (5).
- 5: end for

## Algorithm 2 Synchronous TD learning for infinite-horizon discounted MRPs.

- 1: inputs: learning rates { η t } , number of iterations T , discount factor γ , initial estimate V 0 .
- 3: Draw s t ( s ) ∼ P ( · | s ) for each s ∈ S .
- 2: for t = 1 , 2 , · · · , T do
- 4: Compute V t according to (7).
- 5: end for

The complete description of Q-learning is summarized in Algorithm 1.

As it turns out, TD learning (Bhandari et al., 2021; Sutton, 1988; Tsitsiklis and Van Roy, 1997) in the synchronous setting can be viewed as a special instance of Q-learning when the action set A is a singleton (i.e., |A| = 1 ). In such a case, the MDP reduces to a Markov reward process (MRP) (Bertsekas, 2017), and we shall abuse the notation to use P : S → ∆( S ) to describe the probability transition kernel, and employ r : S → [0 , 1] to represent the reward function (with r ( s ) indicating the immediate reward gained in state s ). The TD learning algorithm maintains an estimate V t : S → R of the value function in each iteration t , 1 and carries out the following iterative update rule

<!-- formula-not-decoded -->

for each state s ∈ S . As before, η t ∈ (0 , 1] is the learning rate at time t , the initial estimate V 0 ( s ) is taken to be within [ 0 , 1 1 -γ ] , and in each iteration, the samples { s t ( s ) | s ∈ S} are generated independently. The whole algorithm of TD learning is summarized in Algorithm 2.

Finally, while synchronous Q-learning is the main focal point of this paper, we shall also discuss the extension to asynchronous Q-learning, which we will elaborate on in Section 5.

## 3 Main results: sample complexity of synchronous Q-learning

With the above backgrounds in place, we are in a position to state formally our main findings in this section, concentrating on the synchronous setting.

## 3.1 Minimax optimality of TD learning

We start with the special with |A| = 1 and characterize the glyph[lscript] ∞ -based sample complexity of synchronous TD learning.

Theorem 1. Consider any δ ∈ (0 , 1) , ε ∈ (0 , 1] , and γ ∈ [1 / 2 , 1) . Suppose that for any 0 ≤ t ≤ T , the learning rates satisfy

<!-- formula-not-decoded -->

for some small enough universal constants c 1 ≥ c 2 &gt; 0 . Assume that the total number of iterations T obeys

<!-- formula-not-decoded -->

1 There is no need to maintain additional Q-estimates, as the Q-function and the value function coincide when |A| = 1 .

for some sufficiently large universal constant c 3 &gt; 0 . If the initialization obeys 0 ≤ V 0 ( s ) ≤ 1 1 -γ for all s ∈ S , then with probability at least 1 -δ , Algorithm 2 achieves

<!-- formula-not-decoded -->

Remark 1 (Mean estimation error) . This high-probability bound immediately translates to a mean estimation error guarantee. Recognizing the crude upper bound ∣ ∣ V T ( s ) -V glyph[star] ( s ) ∣ ∣ ≤ 1 1 -γ (see (104) in Section C.1) and taking δ ≤ ε (1 -γ ) , we reach

<!-- formula-not-decoded -->

provided that T ≥ c 3 (log 3 T ) ( log |S| T ε (1 -γ ) ) (1 -γ ) 3 ε 2 .

Given that each iteration of synchronous TD learning makes use of |S| samples, Theorem 1 implies that the sample complexity of TD learning is at most

<!-- formula-not-decoded -->

for any target accuracy level ε ∈ (0 , 1] . This non-asymptotic result is valid as long as the learning rates are chosen to be either a proper constant or rescaled linear (see (8a)). Compared to a large number of prior works studying the performance of TD learning (Bhandari et al., 2021; Borkar and Meyn, 2000; Chen et al., 2020; Khamaru et al., 2021a; Lakshminarayanan and Szepesvari, 2018; Wainwright, 2019b), Theorem 1 strengthens prior results by uncovering an improved scaling (i.e., 1 (1 -γ ) 3 ) in the effective horizon. In fact, prior results on plain TD learning were only able to obtain a scaling as 1 (1 -γ ) 5 (Wainwright, 2019b).

To assess the tightness of the above result, we take a moment to compare it with the minimax lower bound recently established in the context of value function estimation. Specifically, Pananjady and Wainwright (2020, Theorem 2(b)) asserted that no algorithm whatsoever can obtain an entrywise ε approximation of the value function-in a minimax sense-unless the total sample size exceeds

<!-- formula-not-decoded -->

In turn, this taken together with Theorem 1 unveils the minimax optimality of the sample complexity (modulo some logarithmic factor) of TD learning for the synchronous setting. While prior works have demonstrated how to attain the minimax limit (12) using model-based methods or variance-reduced model-free algorithms (e.g., Azar et al. (2013); Khamaru et al. (2021a); Li et al. (2023); Pananjady and Wainwright (2020)), our theory provides the first rigorous evidence that plain TD learning alone is already minimax optimal, without the need of Polyak-Ruppert averaging or variance reduction.

Remark 2 (Runtime-oblivious learning rates) . Careful readers might remark that the choice (8a) of the learning rates might still rely on prior knowledge on T (or log T ). Fortunately, Theorem 1 immediately leads to convergence guarantees for another choice of η t selected completely independent of T . More specifically, suppose that the learning rates obey

<!-- formula-not-decoded -->

for some universal constants ˜ c 1 , ˜ c 2 &gt; 0 . Then the claim (9) remains valid under this choice (13), provided that

<!-- formula-not-decoded -->

See Appendix C.3 for the proof.

Remark 3 (Polyak-Ruppert averaging) . The results claimed in Remark 2 further allow us to control the estimation error of TD learning under Polyak-Ruppert averaging (Polyak and Juditsky, 1992). More precisely, under the choice (13) of learning rates, the averaged iterates satisfy

<!-- formula-not-decoded -->

with probability exceeding 1 -δ . See Appendix C.3 for the proof.

Remark 4 . It is also noteworthy that: while the last iterate of plain TD learning is shown to be minimax optimal (which concerns worst-case optimality), it might not necessarily enjoy local optimality. As recently demonstrated by Khamaru et al. (2021b), additional algorithmic tricks like variance reduction might be needed in order to ensure local optimality.

## 3.2 Tight sample complexity and sub-optimality of Q-learning

Next, we move on to the more general case with |A| ≥ 2 and study the performance of Q-learning. As it turns out, Q-learning with |A| ≥ 2 is considerably more challenging to analyze than the TD learning case, due to the presence of the nonsmooth max operator. Our glyph[lscript] ∞ -based sample complexity bound for Q-learning is summarized as follows, strengthening the state-of-the-art results.

Theorem 2. Consider any δ ∈ (0 , 1) , ε ∈ (0 , 1] , and γ ∈ [1 / 2 , 1) . Suppose that for any 0 ≤ t ≤ T , the learning rates satisfy

<!-- formula-not-decoded -->

for some small enough universal constants c 1 ≥ c 2 &gt; 0 . Assume that the total number of iterations T obeys

<!-- formula-not-decoded -->

for some sufficiently large universal constant c 3 &gt; 0 . If the initialization obeys 0 ≤ Q 0 ( s, a ) ≤ 1 1 -γ for any ( s, a ) ∈ S × A , then Algorithm 1 achieves

<!-- formula-not-decoded -->

Remark 5 (Mean estimation error) . Repeating exactly the same argument as in Remark 1, one can readily translate this high-probability bound into the following mean estimation error guarantee:

with probability at least 1 -δ .

<!-- formula-not-decoded -->

holds as long as T ≥ c 3 (log 4 T ) ( log |S||A| T ε (1 -γ ) ) (1 -γ ) 4 ε 2 .

In a nutshell, Theorem 2 develops a non-asymptotic bound on the iteration complexity of Q-learning in the presence of the synchronous model. A few remarks and implications are in order.

Sample complexity and sharpened dependency on 1 1 -γ . Recognizing that |S||A| independent samples are drawn in each iteration, we can see from Theorem 2 the following sample complexity bound

<!-- formula-not-decoded -->

in order for Q-learning to attain ε -accuracy ( 0 &lt; ε &lt; 1 ) in an entrywise sense. To the best of our knowledge, this is the first result that breaks the |S||A| (1 -γ ) 5 ε 2 barrier that is present in all state-of-the-art analyses for vanilla Q-learning (Beck and Srikant, 2012; Chen et al., 2020; Li et al., 2022c; Qu and Wierman, 2020; Wainwright, 2019b).

Learning rates. Akin to the TD learning case, our result accommodates two commonly adopted learning rate schemes (cf. (16a)): (i) linearly rescaled learning rates 1 1+ c 2 (1 -γ ) log 2 T t , and (ii) iteration-invariant learning rates 1 1+ c 1 (1 -γ ) T log 2 T (which depend on the total number of iterations T but not the iteration number t ). In particular, when T = c 3 (log 4 T ) ( log |S||A| T δ ) (1 -γ ) 4 ε 2 , the constant learning rates can be taken to be on the order of

<!-- formula-not-decoded -->

which depends almost solely on the discount factor γ and the target accuracy ε . Interestingly, both learning rate schedules lead to the same glyph[lscript] ∞ -based sample complexity bound (in an order-wise sense), making them appealing for practical use.

Remark 6 (Runtime-oblivious learning rates and Polyak-Ruppert averaging) . Akin to Remark 2, Theorem 2 can be easily extended to accommodate a family of learning rates chosen without prior knowledge of T . More concretely, suppose that the learning rates obey

<!-- formula-not-decoded -->

for some suitable constants ˜ c 1 , ˜ c 2 &gt; 0 . Then the claim (17) continues to hold under this choice (20), provided that T/ 2 ≥ c 3 (log 4 T ) ( log |S||A| T δ ) (1 -γ ) 4 ε 2 . Additionally, similar to Remark 3, we can demonstrate that the averaged Q-learning iterates under the choice (20) of learning rates obey

<!-- formula-not-decoded -->

with probability exceeding 1 -δ . The proofs of these results are identical to those of Remarks 2-3 (see Appendix C.3), and are hence omitted.

A matching lower bound and sub-optimality. The careful reader might remark that there remains a gap between our sample complexity bound for Q-learning and the minimax lower bound (Azar et al., 2013). More specifically, the minimax lower bound scales on the order of |S||A| (1 -γ ) 3 ε 2 and is achievable-up to some logarithmic factor-by the model-based approach and variance-reduced methods (Agarwal et al., 2020; Azar et al., 2013; Li et al., 2023; Wainwright, 2019c). This raises natural questions regarding whether our sample complexity bound can be further improved, and whether there is any intrinsic bottleneck that prevents vanilla Q-learning from attaining optimal performance. To answer these questions, we develop the following lower bound for plain Q-learning, with the aim of confirming the sharpness of Theorem 2 and revealing the sub-optimality of Q-learning.

Theorem 3. Assume that 3 / 4 ≤ γ &lt; 1 and that T ≥ c 3 (1 -γ ) 2 for some sufficiently large constant c 3 &gt; 0 . Suppose that the initialization is Q 0 ≡ 0 , and that the learning rates are taken to be either (i) η t = 1 1+ c η (1 -γ ) t for all t ≥ 0 , or (ii) η t ≡ η for all t ≥ 0 . There exists a γ -discounted MDP with |S| = 4 and |A| = 2 such that Algorithm 1-with any c η &gt; 0 and any η ∈ (0 , 1) -obeys

<!-- formula-not-decoded -->

where c lb &gt; 0 is some universal constant.

Remark 7 . This theorem constructs a hard MDP instance with no more than 4 states and 2 actions, with the emphasis of unveiling the sub-optimality of horizon dependency. It can be generalized to accommodate larger state/action space, as we shall elucidate in Section 4.3.

Remark 8 . Theorem 3 concentrates on two families of learning rates-rescaled linear, and constant learning rates-that are most widely used in practice. Note, however, that our current analysis does not readily generalize to arbitrary learning rates, which we leave for future investigation.

Theorem 3 provides an algorithm-dependent lower bound for vanilla Q-learning. As asserted by this theorem, it is impossible for Q-learning to attain ε -accuracy (in the sense that max s E [∣ ∣ V T ( s ) -V glyph[star] ( s ) ∣ ∣ 2 ] ≤ ε 2 ) unless the number of iterations exceeds the order of

<!-- formula-not-decoded -->

up to some logarithmic factor. Consequently, the performance guarantees for Q-learning derived in Theorem 2 are sharp in terms of the dependency on the effective horizon 1 1 -γ . On the other hand, it has been shown in prior literature that the minimax sample complexity limit with a generative model is on the order of (Azar et al., 2013; Li et al., 2022c)

<!-- formula-not-decoded -->

this in turn reveals the sub-optimality of plain Q-learning, whose horizon scaling is larger than the minimax limit by a factor of 1 1 -γ . Hence, more sophisticated algorithmic tricks are necessary in order to further reduce the sample complexity. For instance, a variance-reduced variant of Q-learning-namely, leveraging the idea of variance reduction originating from stochastic optimization (Johnson and Zhang, 2013) to accelerate convergence of Q-learning-has been shown to attain minimax optimality (23) for any ε ∈ (0 , 1] ; see Wainwright (2019c) for more details.

## 4 Key analysis ideas (the synchronous case)

This section outlines the key ideas for the establishment of our main results of Q-learning for the synchronous case, namely Theorem 2 and Theorem 3. The proof for TD learning is deferred to Appendix C. Before delving into the proof details, we first introduce convenient vector and matrix notation that shall be used frequently.

## 4.1 Vector and matrix notation

To begin with, for any matrix M , the notation ‖ M ‖ 1 := max i ∑ j | M i,j | is defined as the largest row-wise glyph[lscript] 1 norm of M . For any vector a = [ a i ] n i =1 ∈ R n , we define √ · and | · | in a coordinate-wise manner, i.e. √ a := [ √ a i ] n i =1 ∈ R n and | a | := [ | a i | ] n i =1 ∈ R n . For a set of vectors a 1 , · · · , a m ∈ R n with a k = [ a k,j ] n j =1 ( 1 ≤ k ≤ m ), we define the max operator in an entrywise fashion such that max 1 ≤ k ≤ m a k := [max k a k,j ] n j =1 . For any vectors a = [ a i ] n i =1 ∈ R n and b = [ b i ] n i =1 ∈ R n , the notation a ≤ b (resp. a ≥ b ) means a i ≤ b i (resp. a i ≥ b i ) for all 1 ≤ i ≤ n . We also let a ◦ b = [ a i b i ] n i =1 denote the Hadamard product. In addition, we denote by 1 (resp. e i ) the all-one vector (resp. the i -th standard basis vector), and let I be the identity matrix.

We shall also introduce the matrix P ∈ R |S||A|×|S| to represent the probability transition kernel P , whose ( s, a ) -th row P s,a is a probability vector representing P ( · | s, a ) . Additionally, we define the square probability transition matrix P π ∈ R |S||A|×|S||A| (resp. P π ∈ R |S|×|S| ) induced by a deterministic policy π over the state-action pairs (resp. states) as follows:

<!-- formula-not-decoded -->

where Π π ∈ { 0 , 1 } |S|×|S||A| is a projection matrix associated with the deterministic policy π :

<!-- formula-not-decoded -->

with e i the i -th standard basis vector. Moreover, for any vector V ∈ R |S| , we define Var P ( V ) ∈ R |S||A| as follows:

<!-- formula-not-decoded -->

In other words, the ( s, a ) -th entry of Var P ( V ) corresponds to the variance Var s ′ ∼ P ( ·| s,a ) ( V ( s ′ )) w.r.t. the distribution P ( · | s, a ) .

Moreover, we use the vector r ∈ R |S||A| to represent the reward function r , so that for any ( s, a ) ∈ S × A , the ( s, a ) -th entry of r is given by r ( s, a ) . Analogously, we shall employ the vectors V π ∈ R |S| , V glyph[star] ∈ R |S| , V t ∈ R |S| , Q π ∈ R |S||A| , Q glyph[star] ∈ R |S||A| and Q t ∈ R |S||A| to represent V π , V glyph[star] , V t , Q π , Q glyph[star] and Q t , respectively. Additionally, we define π t to be the policy associated with Q t such that for any state-action pair ( s, a ) ,

<!-- formula-not-decoded -->

In other words, for any s ∈ S , the policy π t picks out the smallest indexed action that attains the largest Q-value in the estimate Q t ( s, · ) . As an immediate consequence, one can easily verify

<!-- formula-not-decoded -->

for any π, where P π is defined in (24). Further, we introduce a matrix P t ∈ { 0 , 1 } |S||A|×|S| such that

<!-- formula-not-decoded -->

for any ( s, a ) , which is an empirical transition matrix constructed using samples collected in the t -th iteration.

Finally, let X := ( |S| , |A| , 1 1 -γ , 1 ε ) . The notation f ( X ) = O ( g ( X )) or f ( X ) glyph[lessorsimilar] g ( X ) (resp. f ( X ) glyph[greaterorsimilar] g ( X ) ) means that there exists a universal constant C 0 &gt; 0 such that | f ( X ) | ≤ C 0 | g ( X ) | (resp. | f ( X ) | ≥ C 0 | g ( X ) | ). The notation f ( X ) glyph[equivasymptotic] g ( X ) means f ( X ) glyph[lessorsimilar] g ( X ) and f ( X ) glyph[greaterorsimilar] g ( X ) hold simultaneously. We define ˜ O ( · ) in the same way as O ( · ) except that it hides logarithmic factors.

## 4.2 Proof outline for Theorem 2

We are now positioned to describe how to establish Theorem 2, towards which we first express the Q-learning update rule (4) and (5) using the above matrix notation. As can be easily verified, Q-learning employs the samples in P t (cf. (29)) to perform the following update

<!-- formula-not-decoded -->

in the t -th iteration. In the sequel, we denote by

<!-- formula-not-decoded -->

the error of the Q-function estimate in the t -th iteration.

## 4.2.1 Basic decomposition

We start by decomposing the estimation error term ∆ t . In view of the update rule (30), we arrive at the following elementary decomposition:

<!-- formula-not-decoded -->

where the third line exploits the Bellman equation Q glyph[star] = r + γ PV glyph[star] . Further, the term P ( V t -1 -V glyph[star] ) can be linked with ∆ t -1 using the definition (27) of π t as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have made use of the relation (28). Substitute (33) into (32) to reach

<!-- formula-not-decoded -->

Applying these relations recursively, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we define

Comparisons to prior approaches. We take a moment to discuss how prior analyses handle the above elementary decomposition. Several prior works (e.g., Li et al. (2022c); Wainwright (2019b)) tackled the second term on the right-hand side of the relation (34) via the following crude bounds:

<!-- formula-not-decoded -->

which, however, are too loose when characterizing the dependency on 1 1 -γ . By contrast, expanding terms recursively without the above type of crude bounding and carefully analyzing the aggregate terms (e.g., ∑ t i =1 η ( t ) i P π i -1 ∆ i -1 ) play a major role in sharpening the dependence of sample complexity on the effective horizon.

## 4.2.2 Key intertwined relations underlying {‖ ∆ t ‖ ∞ }

By exploiting the crucial relations (35) derived above, we proceed to upper and lower bound ∆ t separately. To be more specific, defining

<!-- formula-not-decoded -->

for some constant c 4 &gt; 0 , one can further decompose the upper bound in (35) into several terms:

<!-- formula-not-decoded -->

Let us briefly remark on the effect of the first two terms:

- Each component in the first term ζ t is fairly small, given that η ( t ) i is sufficiently small for any i ≤ (1 -β ) t (meaning that each component has undergone contraction-the ones taking the form of 1 -η j -for sufficiently many times). As a result, the influence of ζ t becomes somewhat negligible.
- The second term ξ t , which can be controlled via Freedman's inequality (Freedman, 1975) due to its martingale structure, contributes to the main variance term in the above recursion. Note, however, that the resulting variance term also depends on { ∆ i } .

In summary, the right-hand side of the above inequality can be further decomposed into some weighted superposition of { ∆ i } in addition to some negligible effect. This is formalized in the following two lemmas, which make apparent the key intertwined relations underlying { ∆ i } .

Lemma 1. Suppose that c 1 c 2 ≤ c 4 / 8 . With probability at least 1 -δ ,

<!-- formula-not-decoded -->

holds simultaneously for all t ≥ T c 2 log T .

Lemma 2. Suppose that c 1 c 2 ≤ c 4 / 8 . With probability at least 1 -δ ,

<!-- formula-not-decoded -->

holds simultaneously for all t ≥ T c 2 log T .

Proof. The proofs of Lemma 1 and Lemma 2 are deferred to Appendices B.2 and B.3, respectively. As a remark, our analysis collects all the error terms accrued through the iterations-instead of bounding them individually-by conducting a high-order nonlinear expansion of the estimation error through recursion, followed by careful control of the main variance term leveraging the structure of the discounted MDP.

Putting the preceding bounds in Lemmas 1 and 2 together, we arrive at

<!-- formula-not-decoded -->

for all t ≥ T c 2 log T with probability exceeding 1 -2 δ , which forms the crux of our analysis. Employing elementary analysis tailored to the above recursive relation, one can demonstrate that

<!-- formula-not-decoded -->

with probability at least 1 -2 δ , which in turn allows us to establish the advertised result under the assumed sample size condition. The details are deferred to Appendix B.4.

## 4.3 Proof outline for Theorem 3

Construction of a hard instance with 4 states and 2 actions. Let us construct an MDP M hard with state space S = { 0 , 1 , 2 , 3 } (see a pictorial illustration in Figure 4.3). We shall denote by A s the action space associated with state s . The probability transition kernel and reward function of M hard are specified as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the parameter p is taken to be

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Before moving forward to analyze the behavior of Q-learning, we first characterize the optimal value function and Q-function of this MDP; the proof is postponed to Section D.4.

η

(

T

0

)

≤

&lt;latexi sh

1\_b

64="d o3M

AL

TBru/8

cKm

F

gQE

&gt;

H

V

NS

J

W

X

O

k

9

P

UD

2

w

7G

Iv y5

+

n

Y

f

q

Z

C

p

0

j

z

R

1

75

&lt;latexi sh

1\_b

64="TQ

SpB9YKW

CJG

5mU

H

2Rc

&gt;A

V

N

8

E

3

X

O

k

L

g

M

P

D

w

7

/

Iv

y

r

+

u

n

d

f

q

Z

0

j

z

F

o

&lt;latexi sh

1\_b

64="RS5MywIJv

EF

W

kj

H

Oq

L9

&gt;A

B7X

c

V

N

2

+

T

C

3

f

K

z

Z

0

/g

Y

uD

d

8

m

P

o

U

Q

G

r

n

p

&lt;latexi sh

1\_b

64="d o3M

AL

TBru/8

cKm

F

gQE

&gt;

H

V

NS

J

W

X

O

k

9

P

UD

2

w

7G

Iv y5

+

n

Y

f

q

Z

C

p

0

j

z

R

&lt;latexi sh

1\_b

64="U

WXQ

KG

85jd

FNvq

2w7O

o

&gt;A

B

H

c

V

S

EJ3

r

/

9L

0

C

Y

p

T

f

M

D

z

m

u

k

+

Z

g

n

y

I

R

P

&lt;latexi sh

1\_b

64="RS5MywIJv

EF

W

kj

H

Oq

L9

&gt;A

B7X

c

V

N

2

+

T

C

3

f

K

z

Z

0

/g

Y

uD

d

8

m

P

o

U

Q

G

r

n

p

&lt;latexi sh

1\_b

64="RS5MywIJv

EF

W

kj

H

Oq

L9

&gt;A

B7X

c

V

N

2

+

T

C

3

f

K

z

Z

0

/g

Y

uD

d

8

m

P

o

U

Q

G

r

n

p

&lt;latexi sh

1\_b

64="M

2rUwv

KTCSWc8Hzyng+L

O

&gt;A

B7

V

N

EJ3

q/

9

0

Q

Y

F

p

d

f

Xj

o

5

D

m

u

k

Z

G

I

R

P

&lt;latexi sh

1\_b

64="TQ

SpB9YKW

CJG

5mU

H

2Rc

&gt;A

V

N

8

E

3

X

O

k

L

g

M

P

D

w

7

/

Iv

y

r

+

u

n

d

f

q

Z

0

j

z

F

o

&lt;latexi sh

1\_b

64="TQ

SpB9YKW

CJG

5mU

H

2Rc

&gt;A

V

N

8

E

3

X

O

k

L

g

M

P

D

w

7

/

Iv

y

r

+

u

n

d

f

q

Z

0

j

z

F

o

Figure 1: The constructed hard MDP instance used in the analysis of Theorem 3, where p = 4 γ -1 3 γ and the specifications are described in (42).

<!-- image -->

Lemma 3. Consider the MDP M hard constructed in (42) . One has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recognizing the elementary decomposition

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any state s , our proof consists of lower bounding either the squared bias term ( E [ V glyph[star] ( s ) -V T ( s )] ) 2 or the variance term Var ( V T ( s ) ) . In short, we shall primarily analyze the dynamics w.r.t. state 2 to handle the case when the learning rates are either too small or too large, and analyze the dynamics w.r.t. state 1 to cope with the case with medium learning rates (with state 3 serving as a helper state to simplify the analysis). The latter case-corresponding to the learning rates adopted in establishing the upper bounds-is the most challenging: critically, from state 1 the agent can take one of two identical actions, whose value tends to be estimated with a high positive bias due to maximizing over the empirical state-action values, highlighting the well-recognized 'over-estimation' issue of Q-learning in practice (Hasselt, 2010). The complete proof is deferred to Appendix D.

Extension: lower bounds for larger |S| and |A| . For pedagogical reasons, the hard instance (42) constructed above contains no more than 4 states and 2 actions (as the focus has been to unveil sub-optimal dependency on the effective horizon). As it turns out, one can straightforwardly extend it to cover larger state and action spaces, with a more general hard instance constructed as follows.

- 1 · We begin by generating the following sub-MDP, denoted by M sub , which comprises 4 states { 1 , 2 , 3 , 4 } and no more than |A| ≥ 2 actions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where p is still set according to (43).

- The full MDP M full is then constructed by generating |S| / 4 independent copies of M sub .

As can be easily verified (which we omit here for the sake of brevity), our analysis developed for the smaller MDP (42) is directly applicable to studying the more general M full , revealing that the lower bound (55) w.r.t. the iteration number T remains valid. Recognizing that the total sample size scales as |S||A| T , we have established a general sample complexity lower bound |S||A| (1 -γ ) 4 ε 2 for synchronous Q-learning to yield ε -accuracy.

## 5 Extension: sample complexity of asynchronous Q-learning

Moving beyond the synchronous setting, another scenario of practical importance is the case where the acquired samples take the form of a single Markovian trajectory (Tsitsiklis, 1994). In this section, we extend our analysis framework for synchronous Q-learning to accommodate Markovian non-i.i.d. samples.

## 5.1 Markovian samples and asynchronous Q-learning

Markovian sample trajectory. Suppose that we obtain a Markovian sample trajectory { ( s t , a t , r t ) } ∞ t =0 , which is generated by the MDP of interest when a stationary behavior policy π b is employed; in other words,

<!-- formula-not-decoded -->

When π b is stationary, the trajectory { ( s t , a t ) } ∞ t =0 can be viewed as a sample path of a time-homogeneous Markov chain; in what follows, we shall denote by µ π b the stationary distribution of this Markov chain. Note that the behavior policy π b can often be quite different from the target optimal policy π glyph[star] .

Asynchronous Q-learning. In the presence of a single Markovian sample trajectory, the Q-learning algorithm implements the following iterative update rule

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

for all t ≥ 1 , where 0 &lt; η t ≤ 1 stands for the learning rate at time t . It is often referred to as asynchronous Q-learning , as only a single state-action pair is updated in each iteration (in contrast, synchronous Q-learning updates all state-action pairs simultaneously in each iteration). This also leads to the following estimate for the value function at time t :

<!-- formula-not-decoded -->

As can be expected, the presence of Markovian non-i.i.d. data considerably complicates the analysis for asynchronous Q-learning.

Assumptions. In order to ensure sufficient coverage of the sample trajectory over the state/action space, we make the following assumption throughout this section, which is also commonly imposed in prior literature.

Assumption 1. The Markov chain induced by the behavior policy π b is uniformly ergodic. 2

In addition, there are two crucial quantities concerning the sample trajectory that dictate the performance of asynchronous Q-learning. The first one is the minimum state-action occupancy probability of the sample trajectory, defined formally as

<!-- formula-not-decoded -->

This metric captures the information bottleneck incurred by the least visited state-action pair. The second key quantity is the mixing time associated with the sample trajectory, denoted by

<!-- formula-not-decoded -->

Here, d TV ( µ, ν ) := 1 2 ∑ x ∈X | µ ( x ) -ν ( x ) | indicates the total variation distance between two measures µ and ν over X (Tsybakov and Zaiats, 2009), whereas P t ( · | s, a ) stands for the distribution of ( s t , a t ) when the sample trajectory is initialized at ( s 0 , a 0 ) = ( s, a ) . In words, the mixing time reflects the time required for the Markov chain to become nearly independent of the initial states. See Li et al. (2022c, Section 2) for a more detailed account of these quantities and assumptions.

2 See Paulin (2015, Section 1.2) for the definition of uniform ergodicity.

## 5.2 Sample complexity of asynchronous Q-learning

While a number of previous works have been dedicated to understanding the performance of asynchronous Q-learning, its sample complexity bound remains loose when it comes to the dependency on the effective horizon 1 1 -γ . Encouragingly, the analysis framework laid out in this paper allows us to tighten the dependency on 1 1 -γ , as stated below.

Theorem 4. Consider any δ ∈ (0 , 1) , ε ∈ (0 , 1] , and γ ∈ [1 / 2 , 1) . Suppose that for any 0 ≤ t ≤ T , the learning rates satisfy

<!-- formula-not-decoded -->

for some universal constants 0 &lt; c 1 ≤ 1 . Assume that the total number of iterations T obeys

<!-- formula-not-decoded -->

for some sufficiently large universal constant c 2 &gt; 0 . If the initialization obeys 0 ≤ Q 0 ( s, a ) ≤ 1 1 -γ for all ( s, a ) ∈ S × A , then asynchronous Q-learning (cf. (48) ) satisfies

<!-- formula-not-decoded -->

Remark 9 . Similar to Remark 1 and Remark 5, one can immediately translate the above high-probability result into the following mean estimation error bound:

with probability at least 1 -δ .

<!-- formula-not-decoded -->

which holds as long as T ≥ c 2 log 2 |S||A| T ε (1 -γ ) µ min max { log 3 T (1 -γ ) 4 ε 2 , t mix 1 -γ } for some large enough constant c 2 &gt; 0 .

This theorem demonstrates that with high probability, the total sample size needed for asynchronous Q-learning to yield entrywise ε accuracy is

<!-- formula-not-decoded -->

provided that the learning rates are taken to be some proper constant (see (52a)). The first term in (54) resembles our sample complexity characterization of synchronous Q-learning (cf. (19)), except that we replace the number |S||A| of state-action pairs in (19) with 1 /µ min in order to account for non-uniformity across state-action pairs. The second term in (54) is nearly independent of the target accuracy (except for some logarithmic scaling), and can be viewed as the burn-in time taken for asynchronous Q-learning to mimic synchronous Q-learning despite Markovian data.

We now pause to compare Theorem 4 with prior non-asymptotic theory for asynchronous Q-learning. As far as we know, all existing sample complexity bounds (Beck and Srikant, 2012; Chen et al., 2021; Even-Dar and Mansour, 2003; Li et al., 2022c; Qu and Wierman, 2020) scale at least as 1 (1 -γ ) 5 in terms of the dependency on the effective horizon, with Theorem 4 being the first result to sharpen this dependency to 1 (1 -γ ) 4 . In particular, our sample complexity bound strengthens the state-of-the-art result Li et al. (2022c) by a factor up to 1 1 -γ , while improving upon Qu and Wierman (2020) by a factor of at least |S||A| 1 -γ min { t mix , 1 (1 -γ ) 3 ε 2 } . 3

Before concluding this section, we note that for a large enough sample size, the first term 1 µ min (1 -γ ) 4 ε 2 in (54) is essentially unimprovable (up to logarithmic factor). To make precise this statement, we develop a matching algorithm-dependent lower bound as follows, which parallels Theorem 3 previously developed for the synchronous case.

3 The sample complexity of Li et al. (2022c) scales as ˜ O ( 1 µ min (1 -γ ) 5 ε 2 + t mix µ min (1 -γ ) ) , while the sample complexity of Qu and Wierman (2020) scales as ˜ O ( t mix µ 2 min (1 -γ ) 5 ε 2 ) . It is worth noting that 1 /µ min ≥ |S||A| and is therefore a large factor.

Theorem 5. Consider any 0 . 95 ≤ γ &lt; 1 . Suppose that µ min ≤ 1 c 3 log 2 T and T ≥ c 3 log 3 T µ min (1 -γ ) 7 for some sufficiently large constant c 3 &gt; 0 . Assume that the initialization is Q 0 ≡ 0 , and that the learning rates are taken to be η t ≡ η for all t ≥ 0 . Then there exist a γ -discounted MDP with |S| = 4 and |A| = 3 and a behavior policy such that (i) the minimum state-action occupancy probability of the sample trajectory is given by µ min , and (ii) the asynchronous Q-learning update rule (48) -for any η ∈ (0 , 1) -obeys

<!-- formula-not-decoded -->

where c lb &gt; 0 is some universal constant.

In words, Theorem 5 asserts that, for large enough sample size T , in general one cannot hope to achieve glyph[lscript] ∞ -based ε -accuracy using fewer than ˜ O ( 1 µ min (1 -γ ) 4 ε 2 ) samples, thus confirming the sharpness of our upper bound. The proof of this theorem can be found in Appendix F.

## 6 Concluding remarks

In this paper, we have settled the sample complexity of synchronous Q-learning in γ -discounted infinite-horizon MDPs, which is shown to be on the order of ˜ O ( |S| (1 -γ ) 3 ε 2 ) when |A| = 1 and ˜ O ( |S||A| (1 -γ ) 4 ε 2 ) when |A| ≥ 2 . A matching lower bound has been developed when |A| ≥ 2 through studying the dynamics of Q-learning on a hard MDP instance, which unveils the negative impact of an inevitable over-estimation issue. Our theory has been further extended to accommodate asynchronous Q-learning, resulting in tight dependency of the sample complexity on the effective horizon. The analysis framework developed herein-which exploits novel error decompositions and variance control that differ substantially from prior approaches-might suggest a plausible path towards sharpening the sample complexity of, as well as understanding the algorithmic bottlenecks for, other model-free algorithms (e.g., double Q-learning (Hasselt, 2010)).

## Acknowledgements

Y. Chen is supported in part by the Alfred P. Sloan Research Fellowship, the Google Research Scholar Award, the AFOSR grant FA9550-22-1-0198, the ONR grant N00014-22-1-2354, and the NSF grants CCF-2221009, CCF-1907661, DMS-2014279, IIS-2218713 and IIS-2218773. Y. Wei is supported in part by the the NSF grants CCF-2106778, DMS-2147546/2015447 and CAREER award DMS-2143215. Y. Chi is supported in part by the grants ONR N00014-18-1-2142 and N00014-19-1-2404, the NSF grants CCF-1806154, CCF-2007911, CCF-2106778, ECCS-2126634, and DMS-2134080. The authors are grateful to Laixi Shi for helpful discussions about the lower bound, and thank Shaocong Ma for pointing out some errors in an early version of this work. Part of this work was done while G. Li, Y. Chen and Y. Wei were visiting the Simons Institute for the Theory of Computing.

## A Freedman's inequality

The analysis of this work relies heavily on Freedman's inequality (Freedman, 1975), which is an extension of the Bernstein inequality and allows one to establish concentration results for martingales. For ease of presentation, we include a user-friendly version of Freedman's inequality as follows.

Theorem 6. Suppose that Y n = ∑ n k =1 X k ∈ R , where { X k } is a real-valued scalar sequence obeying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define

where we write E k -1 for the expectation conditional on { X j } j : j&lt;k . Then for any given σ 2 ≥ 0 , one has

<!-- formula-not-decoded -->

In addition, suppose that W n ≤ σ 2 holds deterministically. For any positive integer K ≥ 1 , with probability at least 1 -δ one has

<!-- formula-not-decoded -->

Proof. See Freedman (1975); Tropp (2011) for the proof of (56). As an immediate consequence of (56), one has

<!-- formula-not-decoded -->

Next, we turn attention to (57). Consider any positive integer K . As can be easily seen, the event

<!-- formula-not-decoded -->

is contained within the union of the following K events

<!-- formula-not-decoded -->

where we define

<!-- formula-not-decoded -->

Invoking inequality (58) with σ 2 set to be σ 2 2 k -1 and δ set to be δ K , we arrive at P {B k } ≤ δ/K . Taken this fact together with the union bound gives

<!-- formula-not-decoded -->

This concludes the proof.

## B Upper bounds for Q-learning (Theorem 2)

In this section, we fill in the details for the proof idea outlined in Section 4.2 for synchronous Q-learning. In fact, our proof strategy leads to a more general version that accounts for the full ε -range ε ∈ ( 0 , 1 1 -γ ] , as stated below.

Theorem 7. Consider any γ ∈ (0 , 1) and any ε ∈ ( 0 , 1 1 -γ ] . Theorem 2 continues to hold if

<!-- formula-not-decoded -->

for some large enough universal constant c 3 &gt; 0 .

Remark 10 . Clearly, Theorem 7 subsumes Theorem 2 as a special case.

As one can anticipate, the proof of Theorem 7 for Q-learning includes many key ingredients for establishing Theorem 1 for TD learning. We will elaborate on how to modify the proof argument to establish Theorem 1 in Section C.

## B.1 Preliminaries

To begin with, we gather a few elementary facts that shall be used multiple times in the proof.

Ranges of Q t and V t . When properly initialized, the Q-function estimates and the value function estimates always fall within a suitable range, as asserted by the following lemma.

Lemma 4. Suppose that 0 ≤ η t ≤ 1 for all t ≥ 0 . Assume that 0 ≤ Q 0 ≤ 1 1 -γ 1 . Then for any t ≥ 0 ,

<!-- formula-not-decoded -->

Proof. We shall prove this by induction. First, our initialization trivially obeys (60) for t = 0 . Next, suppose that (60) is true for the ( t -1) -th iteration, namely,

<!-- formula-not-decoded -->

and we intend to justify the claim for the t -th iteration. Recognizing that 0 ≤ r ≤ 1 , P t ≥ 0 and ‖ P t ‖ 1 = 1 , one can straightforwardly see from the update rule (30) and the induction hypothesis (61) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, from the definition V t ( s ) := max a Q t ( s, a ) for all t ≥ 0 and all s ∈ S , it is easily seen that

<!-- formula-not-decoded -->

thus establishing (60) for the t -th iteration. Applying the induction argument then concludes the proof.

As a result of Lemma 4 and the fact 0 ≤ Q glyph[star] ≤ 1 1 -γ 1 , we have

<!-- formula-not-decoded -->

which also confirms that 0 ≤ ε ≤ 1 1 -γ is the full ε -range we need to consider. Further, we make note of a direct consequence of the claimed iteration number (59) when ε ≤ 1 1 -γ :

<!-- formula-not-decoded -->

which will be useful for subsequent analysis.

Several facts regarding the learning rates. Next, we gather a couple of useful bounds regarding the learning rates { η t } . To begin with, we find it helpful to introduce the following related quantities introduced previously in (36):

<!-- formula-not-decoded -->

and

We now take a moment to bound η ( t ) i . From our assumption (16a) and the condition (63), we know that the learning rate obeys

<!-- formula-not-decoded -->

for some constants c 1 , c 2 &gt; 0 . Recalling that for any τ ≤ t .

## B.2 Proof of Lemma 1

We shall exploit the relation (39) to prove this lemma. One of the key ingredients of our analysis lies in controlling the terms ζ t and ξ t introduced in (39), which in turn enables us to apply (39) recursively to control ∆ t .

<!-- formula-not-decoded -->

for some universal constant c 4 &gt; 0 and considering any t obeying

<!-- formula-not-decoded -->

we shall bound η ( t ) i by looking at two cases separately.

- For any 0 ≤ i ≤ (1 -β ) t , we can use (65) to show that

<!-- formula-not-decoded -->

where the last inequality holds as long as c 1 c 2 ≤ c 4 / 8 .

- When it comes to the case with i &gt; (1 -β ) t ≥ t/ 2 , one can upper bound

<!-- formula-not-decoded -->

where we have used the constraint (67).

Moreover, the sum of η ( t ) i over i obeys

<!-- formula-not-decoded -->

Repeating the same argument further allows us to derive

<!-- formula-not-decoded -->

Step 1: bounding ζ t . We start by developing an upper bound on ζ t (cf. (39)) for any t obeying T c 2 log T ≤ t ≤ T . Invoking the preceding upper bounds (68) on η ( t ) i implies that

<!-- formula-not-decoded -->

Here, (i) holds since ‖ P π i -1 ‖ 1 = ‖ P i ‖ 1 = ‖ P ‖ 1 = 1 (as they are all probability transition matrices), whereas (ii) arises from the previous bound (68a).

Step 2: bounding ξ t . Moving on to the term ξ t , let us express it as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is due to (68b), Lemma 4, and the fact ‖ P i ‖ 1 = ‖ P ‖ 1 = 1 .

- Next, we turn to certain variance terms. For any vector a = [ a j ] , let us use Var ( a | V i -1 , · · · , V 0 ) to denote a vector whose j -th entry is given by Var ( a j | V i -1 , · · · , V 0 ) . With this notation in place, and recalling the notation Var P ( z ) in (26), we obtain

<!-- formula-not-decoded -->

where the last inequality relies on the previous bounds (68b) and (69).

where the z i 's satisfy

This motivates us to invoke Freedman's inequality (see Theorem 6) to control ξ t for any t obeying T c 2 log T ≤ t ≤ T . Towards this, we need to calculate several quantities.

- First, it is seen that

- In the meantime, Theorem 4 leads us to the following trivial upper bound:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By setting K = ⌈ 2 log 2 1 1 -γ ⌉ , one has

Clearly, this sequence satisfies

<!-- formula-not-decoded -->

With the above bounds in place, applying the Freedman inequality in Theorem 6 and invoking the union bound over all the |S||A| entries of ξ t demonstrate that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -δ/T. Here, the second line holds due to (72) and the fact log 8 |S||A| T log 1 1 -γ δ ≤ 2 log |S||A| T δ (cf. (63)), whereas the last inequality makes use of the relation (71).

Step 3: using the bounds on ζ t and ξ t to control ∆ t . Let us define

<!-- formula-not-decoded -->

In view of the upper bounds derived in Steps 1 and 2, and β defined in (66), we have-with probability exceeding 1 -δ -that provided that T ≥ c 9 (log 4 T ) ( log |S||A| T δ ) (1 -γ ) 3 for some sufficiently large constant c 9 &gt; 0 . Substituting (74) into (39), we can upper bound ∆ t as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Further, we find it convenient to define { α ( t ) i } as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any t , where the first inequality results from (69). With these in place, we can write (75) as

<!-- formula-not-decoded -->

Given that (1 -β ) t ≥ 2 t/ 3 (see (66)), we can invoke this relation recursively to yield

<!-- formula-not-decoded -->

where the second inequality relies on (78), the third line uses the inequality η ( t ) i 1 +1 ≤ α ( t ) i 1 in (77), and the fourth line is valid since ∑ i 1 -1 i 2 =(1 -β ) i 1 α ( i 1 ) i 2 = 1 (see (77)).

We intend to continue invoking (78) recursively-similar to how we derive (79)-in order to control ∆ t . To do so, we are in need of some preparation. First, let us define

<!-- formula-not-decoded -->

for any t &gt; i 1 &gt; i 2 &gt; · · · &gt; i H , which clearly satisfies (see (77))

In addition, defining the index set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have

Additionally, recalling that β = c 4 (1 -γ ) / log T , we see that this choice of H satisfies

<!-- formula-not-decoded -->

for c 4 small enough, thus implying that

<!-- formula-not-decoded -->

This is an important property that allows one to invoke the relation (78). With these in place, applying the preceding relation (78) recursively-in a way similar to (79)-further leads to

<!-- formula-not-decoded -->

for all t ≥ T c 2 log T , where we recall the definition of the entrywise max operator in Section 4.1. Here, the last inequality relies on the fact that ∑ ( i 1 , ··· ,i H ) ∈I t α { i k } H k =1 = 1 (see (83)). It remains to control β 1 and β 2 , which we shall accomplish separately in the next two steps.

Step 4: bounding β 2 . The term β 2 defined in (84) is relatively easier to control. Observing that ∏ H k =1 P π i k is still a probability transition matrix, we can derive

<!-- formula-not-decoded -->

where (i) results from the crude bound (62). To justify the inequality (ii), we recall the definition (80) of H to see that

<!-- formula-not-decoded -->

where the inequality comes from the elementary fact that γ 1 1 -γ ≤ e -1 for any 0 &lt; γ &lt; 1 .

Step 5: bounding β 1 . When it comes to the term β 1 defined in (84), we can upper bound the entrywise square of β 1 -denoted by | β 1 | 2 -as follows

<!-- formula-not-decoded -->

Here, (i) follows from Jensen's inequality and the fact that ∏ h k =1 P π i k is a probability transition matrix; (ii) holds due to the Cauchy-Schwarz inequality; (iii) utilizes the definition of ϕ t in (73); (iv) follows since ∏ 1 ≤ k ≤ h P π i k 1 = 1 and ∑ 0 ≤ h&lt;H γ h ≤ 1 1 -γ . To further control the right-hand side of the above inequality, we resort to the following lemma.

Lemma 5. Suppose that t ≥ T c 2 log T . For any ( i 1 , · · · , i H ) ∈ I t , the following holds:

<!-- formula-not-decoded -->

Proof. This lemma, which is inspired by but significantly more complicated than Azar et al. (2013, Lemma 8), plays a key role in shaving one 1 1 -γ factor. See Section B.5 for the proof.

Therefore, the above result directly implies that

<!-- formula-not-decoded -->

Step 6: putting all this together. Substituting the preceding bounds for β 1 and β 2 into (84), we can demonstrate that: with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds simultaneously for all t ≥ T c 2 log T , where the second line is valid since 1 (1 -γ ) T ≤ √ (log 4 T ) ( log |S||A| T δ ) (1 -γ ) 4 T under our sample size condition (63).

## B.3 Proof of Lemma 2

Next, we move forward to develop an lower bound on ∆ t , which can be accomplished in an analogous manner as for the above upper bound. Applying a similar argument for (84) (except that we need to replace π i with π glyph[star] ), one can deduce that

<!-- formula-not-decoded -->

for any t ≥ c 2 T log 1 1 -γ . It is straightforward to bound the second term on the right-hand side of (88) as

<!-- formula-not-decoded -->

where the second inequality makes use of (62) as well as the fact that ∏ k P π glyph[star] is a probability transition matrix (so that ‖ ∏ k P π glyph[star] ∥ ∥ 1 = 1 ). As for the first term on the right-hand side of (88), we can invoke a similar argument for (86) to obtain

<!-- formula-not-decoded -->

Taking these two bounds together, we see that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

holds simultaneously for all t ≥ T c 2 log T .

## B.4 Solving the recurrence relation regarding ∆ t

Recall from (40) that with probability exceeding 1 -2 δ , the following recurrence relation

<!-- formula-not-decoded -->

holds, which plays a crucial role in establishing the desired estimation error bound. Specifically, for any k ≥ 0 , let us define

<!-- formula-not-decoded -->

To bound this sequence, we first obtain a crude bound as a result of (62):

<!-- formula-not-decoded -->

Next, it is directly seen from (90) and the definition of u k that

<!-- formula-not-decoded -->

for some constant c 6 = 20 /γ &gt; 0 . In order to analyze the size of u k , we divide into two cases.

- If u k ≤ 1 for some k ≥ 1 , then (93) tells us that

<!-- formula-not-decoded -->

as long as T ≥ 2 c 2 6 log 4 T log |S||A| T δ γ 2 (1 -γ ) 4 . In other words, once u k -1 drops below 1 , then all subsequent quantities will remain bounded above by 1, namely, max j : j ≥ k u j ≤ 1 . As a result,

<!-- formula-not-decoded -->

- Instead, suppose that u j &gt; 1 for all 0 ≤ j ≤ k . Then it is seen from (93) that

<!-- formula-not-decoded -->

This is equivalent to saying that

<!-- formula-not-decoded -->

where α u = c 6 √ 2 ( log 4 T )( log |S||A| T δ ) γ 2 (1 -γ ) 4 T . Invoking a standard analysis strategy for this type of recursive relations yields and hence

<!-- formula-not-decoded -->

log u j +1 ≤ 2 log α u + ( 1 2 ) j +1 (log u 0 -2 log α u ) for all j ≤ k.

This is equivalent to saying that

<!-- formula-not-decoded -->

Putting the above two cases together and using (92), we conclude that

<!-- formula-not-decoded -->

In particular, as long as k ≥ c 7 log log 1 1 -γ for some constant c 7 &gt; 0 , one has ( 1 1 -γ ) 1 / 2 k ≤ O (1) and

<!-- formula-not-decoded -->

As a result, the above bound simplifies to

<!-- formula-not-decoded -->

for some constant c 8 &gt; 0 .

Consequently, taking t = T and choosing k = c 7 log log 1 1 -γ for some appropriate constant c 7 &gt; 0 (so as to ensure 2 k T c 2 log T &lt; T ), we immediately see from the definition (91) of u k that

<!-- formula-not-decoded -->

with probability at least 1 -2 δ . To finish up, we note that the sample size assumption (59) is equivalent to

<!-- formula-not-decoded -->

When c 3 &gt; 0 is sufficiently large, substituting this relation into (94) gives

<!-- formula-not-decoded -->

as claimed in Theorem 7.

## B.5 Proof of Lemma 5

We first claim that

If this claim were valid (which we shall justify towards the end of this subsection), then it would lead to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It then boils down to bounding the first term on the right-hand side of (96). Let us first upper bound the variance term involving V glyph[star] . For any 0 ≤ h &lt; H , one can express (see (26))

<!-- formula-not-decoded -->

where (i) relies on the identity Q glyph[star] = r + γ PV glyph[star] , and (iii) holds since 0 &lt; γ &lt; 1 . To justify (ii), we make the following observation:

<!-- formula-not-decoded -->

where (iv) arises from the fact ‖ Pz ‖ ∞ ≤ ‖ P ‖ 1 ‖ z ‖ ∞ = ‖ z ‖ ∞ , (v) is valid because ‖ Q glyph[star] ‖ ∞ ≤ 1 / (1 -γ ) , (vi) follows from the fact that V i h +1 = Π π i h +1 Q i h +1 , and (vi) holds since ‖ V i h +1 -V glyph[star] ‖ ∞ ≤ ‖ Q i h +1 -Q glyph[star] ‖ ∞ .

As it turns out, the first term in (97) allows one to build a telescoping sum. Specifically, invoking (97) allows one to bound

<!-- formula-not-decoded -->

Here, (i) comes from the identity ∏ h k =1 P π i k 1 = 1 ; (ii) holds because each row of ∏ h k =1 P π i k has unit ‖ · ‖ 1 norm for any h ; (iii) arises from the bound ‖ Q glyph[star] ‖ ∞ ≤ 1 / (1 -γ ) . This completes the proof, as long as the claim (95) can be justified.

Proof of the inequality (95) . To validate this result, we make the observation that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the last inequality follows since (by applying Lemma 4)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A useful extension of Lemma 5. Before concluding, we make note of the following extension that proves useful for studying asynchronous Q-learning.

Lemma 6. Suppose that t ≥ T c 2 log T . Then one has

<!-- formula-not-decoded -->

for any set of policies { ̂ π k } obeying { ̂ π k } ⊆ Π . Here, we define

<!-- formula-not-decoded -->

The key difference between Lemma 6 and Lemma 5 is that: the components of ̂ π k corresponding to different states can be chosen in a separate manner. The proof follows from an identical argument as the above proof of Lemma 5, and is hence omitted.

## C Analysis for TD learning (Theorem 1)

As it turns out, if |A| = 1 (which reduces to the case of TD learning), we can further modify the previous analysis in Section B to yield an improved 1 (1 -γ ) 3 scaling. This forms the main content of this section, which leads to the proof of Theorem 1 for TD learning. Akin to the Q-learning case, we proceed to establish a more general version of Theorem 1 that covers the full ε -range. This is formally stated below, which subsumes Theorem 1 as a special case.

Theorem 8. Consider any γ ∈ (0 , 1) and any ε ∈ ( 0 , 1 1 -γ ] . Theorem 1 continues to hold if

<!-- formula-not-decoded -->

for some sufficiently large universal constant c 3 &gt; 0 .

## C.1 Preliminary facts

Before embarking on the analysis, we begin by presenting several useful preliminary facts. The first one is a direct consequence of the claimed iteration complexity (101) when ε ≤ 1 1 -γ :

<!-- formula-not-decoded -->

a simple fact that will be used multiple times. In addition, the update rule (7) of TD learning can be expressed using vector/matrix notation as follows

<!-- formula-not-decoded -->

where the matrix P t ∈ { 0 , 1 } |S|×|S| obeys

<!-- formula-not-decoded -->

for any s, s ′ ∈ S . In the sequel, we collect a few other facts concerning the range of V t and learning rates.

Range of V t . We claim that: when the initialization V 0 obeys 0 ≤ V 0 ≤ 1 1 -γ 1 , the TD learning iterates obey

<!-- formula-not-decoded -->

provided that 0 ≤ η t ≤ 1 for all t ≥ 0 . The proof follows immediately by repeating the proof of Lemma 4 (see Section B.1) with |A| = 1 , and is hence omitted for brevity.

Learning rates. We shall also collect several useful results concerning the learning rates { η t } . Let us abuse the notation by defining the following crucial quantities:

<!-- formula-not-decoded -->

Note that this definition (105) differs from the one (64) used for Q-learning, and will only be employed in this section. Consider any iteration number t satisfying

<!-- formula-not-decoded -->

Clearly, the learning rate η t under Assumption (8a) obeys

<!-- formula-not-decoded -->

In what follows, we intend to bound η ( t ) k for two cases separately.

- For any i obeying 0 ≤ i ≤ t/ 2 , it is easily seen from (107) that

<!-- formula-not-decoded -->

where the last inequality holds as long as c 1 c 2 ≤ 1 / 8 and (102) holds.

- When it comes to the case with i &gt; t/ 2 , we can develop the following upper bound

<!-- formula-not-decoded -->

which relies on Assumption (8a).

In addition, given that P k 1 = 1 for any integer k &gt; 0 , it can be easily verified that and as a result,

<!-- formula-not-decoded -->

## C.2 Proof of Theorem 8

Step 1: decomposing the error V t -V glyph[star] . Taking ∆ t := V t -V glyph[star] , via the basic relation (32), the TD learning update rule can be written as

<!-- formula-not-decoded -->

Invoking the above relation recursively then leads to

<!-- formula-not-decoded -->

Step 2: controlling the first term of (111) . With regards to the first term of (111), we make the observation that

<!-- formula-not-decoded -->

where the second line arises from (109), and the last inequality holds true due to (108a) as long as t ≥ T c 2 log T .

Step 3: controlling the second term of (111) . We then move on to the second term ξ t in (111), which admits the following expression

<!-- formula-not-decoded -->

Here, the summands { z k } clearly satisfy

<!-- formula-not-decoded -->

We then attempt to invoke the Freedman inequality (see Theorem 6) to control this term. Towards this end, there are several quantities that need to be calculated.

- First of all, we observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- The next step is to control certain variance terms. Towards this, we first make note of a userful fact. For any given non-negative vector u = [ u i ] 1 ≤ i ≤|S| ≥ 0 and any vector v , it is easily seen that

where the third line again makes use of the relation (109) and the last line follows the facts ‖ P k ‖ 1 = ‖ P ‖ 1 = 1 , ‖ V k -1 ‖ ∞ ≤ 1 / (1 -γ ) , as well as the properties (108).

<!-- formula-not-decoded -->

where we remind the reader of the notation Var P ( v ) in (26). Additionally, for any vector a = [ a j ] , let us employ the notation Var ( a | V k -1 , · · · , V 0 ) to represent a vector whose j -th entry is given by Var ( a j | V k -1 , · · · , V 0 ) . Armed with this notation, we obtain

<!-- formula-not-decoded -->

where the first inequality is a consequence of (115) and the definition of z k (cf. (113)), the second line arises from (109), and the last relation results from the definition of η ( t ) k . This in turn allows us to compute

<!-- formula-not-decoded -->

where the penultimate inequality results from (108); to see why the last inequality holds, observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used the fact that all entries of ( I -γ P ) -1 and I -η i ( I -γ P ) are non-negative.

- In addition, we also derive the following trivial upper bound based on (117):

<!-- formula-not-decoded -->

where we have invoked the fact that ‖ ( I -γ P ) -1 ‖ 1 = 1 / (1 -γ ) . Therefore, by setting K = ⌈ 2 log 2 1 1 -γ ⌉ , one arrives at

<!-- formula-not-decoded -->

Equipped with the preceding bounds, let us apply the Freedman inequality in Theorem 6 and invoke the union bound over all entries of ξ t to show that

<!-- formula-not-decoded -->

with probability at least 1 -δ/T. Here, the second line follows since

<!-- formula-not-decoded -->

as long as |S| T δ ≥ 8log 1 1 -γ , whereas the last line holds by using (114), (117) and (119). Further, we make the observation that

<!-- formula-not-decoded -->

where the second line makes use of the basic relation V glyph[star] = r + γ PV glyph[star] . As a consequence, we conclude

<!-- formula-not-decoded -->

Here, the first inequality arises from (95), while the second inequality holds due to the facts that ∥ ∥ ( I -γ P ) -1 ∥ ∥ 1 = 1 / (1 -γ ) .

Step 4: putting everything together. Consequently, substituting the bounds in Steps 2-3 into (111) yields

<!-- formula-not-decoded -->

Repeating the same argument as in Section B.4, we see that

<!-- formula-not-decoded -->

holds with probability at least 1 -δ , where c 9 &gt; 0 is some universal constant. As a result, one has

<!-- formula-not-decoded -->

as long as the sample size satisfies the following

<!-- formula-not-decoded -->

for some constant c 3 ≥ max { 1 , 2 c 9 } . This requirement is equivalent to condition (101) as claimed.

## C.3 Proof for Remarks 2 and 3

Proof for Remark 2. Let us divide the dynamics of the algorithm into two parts.

- For any 1 ≤ t ≤ T/ 2 , it follows from Lemma 4 that

<!-- formula-not-decoded -->

- Next, let us consider any T/ 2 &lt; t ≤ T , and set ˜ t = t -T/ 2 . It comes from the choice (13) that

<!-- formula-not-decoded -->

provided that ˜ c 1 and ˜ c 2 are suitably chosen. By treating V T/ 2 as the initial point and invoking Theorem 1, we immediately establish (9) under the choice (13) and the sample size condition (14).

Proof for Remark 3. Let us define, for each 1 ≤ t ≤ T ,

<!-- formula-not-decoded -->

Invoking the claim in Remark 2 in conjunction with Lemma 4 reveals that with probability at least 1 -δ/T ,

<!-- formula-not-decoded -->

for any given t ≤ T . Taking the union bound over all 1 ≤ t ≤ T implies that with probability exceeding 1 -δ ,

<!-- formula-not-decoded -->

where the last inequality results from the elementary inequality ∑ T t =1 1 / √ t ≤ 2 √ T . This finishes the proof.

## D Lower bound: sub-optimality of synchronous Q-learning (Theorem 3)

In this section, a main focus is to establish the lower bound claimed in Theorem 3 by analyzing synchronous Q-learning for the MDP instance constructed in Section 4.3. Without loss of generality, we assume

<!-- formula-not-decoded -->

throughout the proof; otherwise the lower bound in Theorem 3 is worse than the minimax lower bound 1 (1 -γ ) 3 T in Azar et al. (2013).

Throughout, we shall use P t to represent the sample transitions such that for any triple ( s, a, s ′ ) ,

<!-- formula-not-decoded -->

where s t ( s, a ) stands for the sample collected in the t -th iteration (see (5)). Recognizing that state 2 is associated with a singleton action space, we shall often write

<!-- formula-not-decoded -->

for notational simplicity.

## D.1 Key quantities related to learning rates

We find it convenient to define the following quantities (by abuse of notation)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is helpful to establish several basic properties about these quantities. As can be easily verified,

<!-- formula-not-decoded -->

where we denote ̂ η i := η i (1 -γp ) to simplify notation. Similarly, for any given integer 0 ≤ τ &lt; t one has

<!-- formula-not-decoded -->

## D.2 Preliminary calculations

Before moving forward, we record several basic relations as a result of the Q-learning update rule.

## D.2.1 Basic update rules and expansion

Given that Q 0 = V 0 = 0 and that state 0 is absorbing, the update rule (4) gives

<!-- formula-not-decoded -->

for all t ≥ 1 . Regarding state 2, the update rule (4) taken together with (130) leads to

<!-- formula-not-decoded -->

and for state 3,

Similarly, one also has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In what follows, we shall first determine a crude range for certain quantities relates to the learning rates η t , and then combine this with the above relations to establish the desired result.

Next, we record some elementary decomposition of V t (2) . For any iteration t and τ &lt; t , one can continue the derivation in (131) to obtain

<!-- formula-not-decoded -->

where the penultimate line arises from (129). In particular, in the special case where τ = 0 (so that V τ (2) = V 0 (2) = 0 ), this simplifies to

<!-- formula-not-decoded -->

which relies on the definition of η ( t ) 0 in (127). With similar derivation, (132) leads to

<!-- formula-not-decoded -->

## D.2.2 Mean and variance of V glyph[star] (2) -V T (2)

We start by computing the mean V glyph[star] (2) -E [ V t (2)] . From the construction (42), it is easily seen that E [ P k (2 | 2)] = p , which together with the identity (135) leads to

<!-- formula-not-decoded -->

Similarly, applying the above argument to (134) and rearranging terms, we immediately arrive at

<!-- formula-not-decoded -->

for any integer 0 ≤ τ &lt; T .

Next, we develop a lower bound on the variance Var ( V T (2) ) . Towards this end, consider first a martingale sequence { Z k } 0 ≤ k ≤ T adapted to a filtration F 0 ⊆ F 1 ⊆ · · · ⊆ F T , namely, E [ Z k +1 | F k ] = 0 and E [ Z k | F k ] = Z k for all 0 ≤ k ≤ T . In addition, consider any 0 ≤ τ &lt; T , and let W 0 be a random variable such that E [ W 0 | F τ ] = W 0 . Then the law of total variance together with basic martingale properties tells us that

<!-- formula-not-decoded -->

Consequently, for any 0 ≤ τ &lt; T -1 , it follows from the decomposition (134) (with τ replaced by τ +1 ) that

<!-- formula-not-decoded -->

where the first identity relies on the fact that P k (2 | 2) is a Bernoulli random variable with mean p , and the inequality comes from the definition of τ (see (146)) and the choice of p (see (43)). As an implication, the sum of squares of η ( T ) k plays a crucial role in determining the variance of V T (2) .

## D.3 Lower bounds for three cases

## D.3.1 Case 1: small learning rates ( c η ≥ log T or 0 ≤ η ≤ 1 (1 -γ ) T )

In this case, we focus on lower bounding V glyph[star] (2) -E [ V T (2) ] . In view of this identity (137), this boils down to controlling η ( T ) 0 .

Suppose that c η &gt; log T (for rescaled linear learning rates) or 0 ≤ η &lt; 1 (1 -γ ) T (for constant learning rates). A little algebra then gives

<!-- formula-not-decoded -->

for any t ≥ 1 , provided that T ≥ 15 . Consequently, one can derive

<!-- formula-not-decoded -->

where the first inequality holds due to the elementary fact log(1 -x ) ≥ -1 . 5 x for all 0 ≤ x ≤ 0 . 5 , and the last inequality follows from the following bound (which makes use of (141))

<!-- formula-not-decoded -->

Combining the above result with the properties (137) and (142) then yields

<!-- formula-not-decoded -->

This taken together with (45) gives

<!-- formula-not-decoded -->

## D.3.2 Case 2: large learning rates ( c η ≤ 1 -γ or η ≥ 1 (1 -γ ) 2 T )

By virtue of (138), the mean gap V glyph[star] (2) -E [ V T (2) ] depends on two factors: (i) the choice of the learning rates, and (ii) the gap between 1 1 -γp and E [ V τ (2) ] , where τ is an integer obeying 0 ≤ τ &lt; T . To control the factor (ii), we need to choose τ properly. Let us start by considering the simple scenario with E [( V T (2) ) 2 ] &lt; 1 4(1 -γ ) 2 , for which we have

<!-- formula-not-decoded -->

Here, we have used (44) and the elementary fact E [ X ] ≤ √ E [ X 2 ] . Consequently, it remains to look at the scenario obeying E [( V T (2) ) 2 ] ≥ 1 4(1 -γ ) 2 , towards which we propose to set τ as follows

<!-- formula-not-decoded -->

Clearly, τ is well-defined in this scenario and obeys (in view of both (146) and the initialization V 0 = 0 )

<!-- formula-not-decoded -->

Our analysis for this scenario is divided into three subcases based on the size of the learning rates.

Case 2.1. Consider the case where

Invoke (138) to deduce that

<!-- formula-not-decoded -->

where the second line makes use of the definition (43) and the elementary fact E [ X ] ≤ √ E [ X 2 ] , and the last line relies on the inequalities (147) and (148).

Case 2.2. We now move on to the case where

<!-- formula-not-decoded -->

We intend to demonstrate that the variance of V T (2) -and hence the typical size of its fluctuation-is too large. In view of the observation (140), it boils down to lower bounding ∑ T k = τ +2 ( η ( T ) k ) 2 , which we accomplish as follows.

- Consider constant learning rates η k = η , and suppose that η obeys 1 (1 -γ ) 2 T &lt; η ≤ 1 &lt; 1 1 -γp . It is readily seen that η ( T ) k = η ( 1 -η (1 -γp ) ) T -k for any k ≥ 1 . We claim that it suffices to focus on the scenario where

<!-- formula-not-decoded -->

In fact, if τ ≥ T -1 , then the definition (146) of τ necessarily requires that

<!-- formula-not-decoded -->

In view of (137) (with T replaced by T -1 ), a little algebra shows that this is equivalent to ( 1 -η (1 -γp ) ) T -1 ≥ 1 / 3 , and hence ( 1 -η (1 -γp ) ) T ≥ 1 / 9 . In turn, this combined with (137) leads to

<!-- formula-not-decoded -->

which already suffices for our purpose.

Next, assuming that (150) holds, one can derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds since (from the assumptions (149) and τ ≤ T -2 )

<!-- formula-not-decoded -->

and the last inequality follows since

<!-- formula-not-decoded -->

Substituting (152) into (140), we obtain

<!-- formula-not-decoded -->

provided that γ ≥ 3 / 4 (so that 4 γ -1 ≥ 2 ). Here, the last inequality is valid since either η ≥ 1 (1 -γ ) 2 T .

- We then move on to linearly rescaled learning rates with η t = 1 1+ c η (1 -γ ) t for some 0 ≤ c η &lt; 1 -γ . Towards this, we first make the observation that

<!-- formula-not-decoded -->

with the proviso that c η &lt; 1 -γ ≤ 1 / 3 (as long as γ ≥ 2 / 3 ). By defining τ ′ := T -1 (1 -γ ) η T , one can deduce that

<!-- formula-not-decoded -->

where the penultimate inequality comes from the Cauchy-Schwarz inequality. In addition, recognizing that η ( T ) k 1 ≤ ( 1 -(1 -γ ) η T ) k 2 -k 1 η ( T ) k 2 for any k 2 ≥ k 1 (see (154)), one has

<!-- formula-not-decoded -->

Summing these inequalities up and rearranging terms, we reach

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which relies on the fact ( 1 -(1 -γ ) η T ) T -τ ′ = ( 1 -1 / ( T -τ ′ ) ) T -τ ′ ≤ e -1 (using the definition of τ ′ ). Consequently, it is easily seen that

<!-- formula-not-decoded -->

Here, (i) and (ii) follow from (129) and (149), respectively, while (iii) holds since

<!-- formula-not-decoded -->

as long as γ ≥ 3 / 4 . Substitution into (155) yields

<!-- formula-not-decoded -->

Substituting the above bound into (140), we obtain

<!-- formula-not-decoded -->

provided that γ ≥ 3 / 4 (so that 4 γ -1 ≥ 2 ). Here, the last inequality is valid since η T = 1 1+ c η (1 -γ ) T ≥ 1 1+(1 -γ ) 2 T ≥ 1 2(1 -γ ) 2 T as long as T ≥ 1 (1 -γ ) 2 .

Putting all this together. With the above bounds in place, it is readily seen that either the bias is too large (see (151)) or the variance is too large (see (153) and (157)). These bounds taken collectively with (45) yield

<!-- formula-not-decoded -->

provided T ≥ 1 (1 -γ ) 2 .

D.3.3 Case 3: medium learning rates ( 1 -γ &lt; c η &lt; log T or 1 (1 -γ ) T ≤ η ≤ 1 (1 -γ ) 2 T )

Throughout this case, we assume that

<!-- formula-not-decoded -->

In fact, if η ( T ) 0 &gt; 1 / 75 , then the scenario becomes much easier to cope with. To see this, applying the previous result (143) and recalling the choice (43) of p immediately yield

<!-- formula-not-decoded -->

which together with (45) and the assumption T ≥ 1 (1 -γ ) 2 yields

<!-- formula-not-decoded -->

We now turn our attention to the dynamics w.r.t. state 1 and its associated value function V t (1) under the condition (159).

Two auxiliary sequences. Towards this, we first eliminate the effect of initialization on Q t (1 , a ) by introducing the following auxiliary sequence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we recall the value of Q glyph[star] (1 , a ) from Lemma 3. In other words, { ̂ Q t ( a ) } is essentially a Q-learning sequence when initialized at the ground truth. Despite the difference in initialization, we claim that the discrepancy between ̂ Q t ( a ) and Q t (1 , a ) can be well controlled in the following sense:

<!-- formula-not-decoded -->

which shall be justified in Section D.3.4. As we shall discuss momentarily, the gap 1 1 -γ ∏ t i =1 ( 1 -η i (1 -γ ) ) is sufficiently small for this case.

Further, in order to control ̂ Q t ( a ) , we find it convenient to introduce another auxiliary sequence as follows

<!-- formula-not-decoded -->

which can be interpreted as a Q-learning sequence when there is only a single action (so that there is no max operator involved). In view of the basic fact that ̂ V t = max a ̂ Q t ( a ) ≥ ̂ Q t (1) , we can easily verify that

<!-- formula-not-decoded -->

allowing one to lower bound ̂ V t by controlling Q t .

A useful lower bound on the auxiliary sequence (162) . In what follows, let us establish a useful lower bound on the sequence (162) introduced above. Then we claim that there exists some τ ≤ T (see (179) and (181)) such that

<!-- formula-not-decoded -->

The auxiliary sequence constructed in (164) plays a crucial role in establishing this claim.

Proof of the claim (166) . We intend to employ the sequence Q t (cf. (164)) to help control ̂ V t . It is first observed that the sequence Q t admits the following decomposition (akin to the derivation in (135))

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

where the last line results from (128). In order to lower bound Q t , it boils down to controlling ∑ k z k . Note that the sequence defined above is a martingale satisfying

<!-- formula-not-decoded -->

where the last inequality follows from the basic property 0 ≤ Q k -1 ≤ 1 1 -γ (akin to Lemma 4) and the fact that ∣ ∣ P k (1 | 1 , 1) -p ∣ ∣ ≤ max { p, 1 -p } = p since p = (4 γ -1) / (3 γ ) and γ ≥ 3 / 4 . We intend to invoke Freedman's inequality to control (167). Armed with these properties and the fact that P k (1 | 1 , 1) is a Bernoulli random variable with mean p , we obtain

<!-- formula-not-decoded -->

Here, the penultimate inequality relies on the fact 0 ≤ Q k -1 ≤ 1 1 -γ (akin to Lemma 4) and the choice of p (see definition (43)), whereas the last inequality results from the following condition (derived through (128))

<!-- formula-not-decoded -->

Applying Freedman's inequality (see (58)) then yields

<!-- formula-not-decoded -->

As an implication of the preceding result, a key ingredient towards bounding ∑ t k =1 z k lies in controlling the quantity max 1 ≤ k ≤ t η ( t ) k . To do so, we claim for the moment that there exists some τ ≤ T such that

<!-- formula-not-decoded -->

whose proof is postponed to Section D.3.4. In light of this claim, setting δ = 1 / 2 in the expression (168) yields

<!-- formula-not-decoded -->

with probably at least 1 / 2 . Combining this with the decomposition (167) and the property (165), we arrive at

<!-- formula-not-decoded -->

with probability at least 1 / 2 , where the last identity relies on the choice of p (see the definition (43)). This establishes the advertised claim (166).

{ z k }

Main proof. With the property (166) in place, we are positioned to prove our main result. Towards this, we find it convenient to define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The goal is thus to control ∆ T, max ; in fact, we intend to show that ∆ T, max is in expectation excessively large, resulting in an 'over-estimation' issue that hinders convergence. Towards this, it follows from the iterative update rule (162) that

<!-- formula-not-decoded -->

Here, the second line comes from the Bellman equation Q glyph[star] (1 , a ) = 1 + γpV glyph[star] (1) , whereas the last line holds since ̂ V t -1 -V glyph[star] (1) = max a ( ̂ Q t -1 ( a ) -V glyph[star] (1) ) = max a ∆ t -1 ( a ) (as a consequence of the relation (44)). Applying the above relation recursively leads to

<!-- formula-not-decoded -->

where we have used the initialization ∆ 0 ( a ) = 0 . Letting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

one can express the above relation as follows

<!-- formula-not-decoded -->

Next, we claim that E [ ξ t, max ] satisfies the following property

<!-- formula-not-decoded -->

for some universal constant c &gt; 0 , where

<!-- formula-not-decoded -->

whose existence is ensured under the condition (159). Given the validity of this claim (which we shall justify in Section D.3.4), we immediately arrive at

<!-- formula-not-decoded -->

In order to study the above recursion, it is helpful to look at the following sequence

<!-- formula-not-decoded -->

with x ̂ τ = 0 , where we recall the definition of ̂ τ in (174). In comparison to the iterative relation (175) which starts from E [∆ 0 , max ] = 0 (and hence E [∆ t, max ] ≥ 0 ), we let the sequence x t start from x ̂ τ = 0 , where ̂ τ is defined in (174). It is straightforward to verify that

<!-- formula-not-decoded -->

recognizing that

<!-- formula-not-decoded -->

A little algebra reveals that the sequence (176) obeys

<!-- formula-not-decoded -->

where the second equality arises from (129), and the last inequality holds as long as ∏ T i = ̂ τ ( 1 -η i (1 -γp ) ) ≤ 6 / 7 (see (189)). This taken together with (177) leads to

<!-- formula-not-decoded -->

Combining the above bound with (163) leads to

<!-- formula-not-decoded -->

Taking this together with (136), we arrive at

<!-- formula-not-decoded -->

This combined with (45) establishes the following desired lower bound:

<!-- formula-not-decoded -->

## D.3.4 Proofs of auxiliary results

Proof of the inequality (163) . We shall establish this claim by induction. To begin with, the inequality (163) holds trivially for the base case with t = 0 . Now, let us assume that the claim holds up to the ( t -1) -th

iteration, and we would like to justify it for the t -th iteration. As an immediate consequence of the claim (163) for the ( t -1) -th iteration and the definitions of V t -1 and ̂ V t -1 , we have

<!-- formula-not-decoded -->

By virtue of the respective update rules of Q t (1 , a ) and ̂ Q t ( a ) , we can express their difference as follows:

<!-- formula-not-decoded -->

where the first inequality invokes the induction hypothesis for the ( t -1) -th iteration. This establishes (163) for the t -th iteration, and hence the proof is complete via an induction argument.

Proof of the claim (169) . When taking the constant learning rates η t ≡ η ≤ 1 (1 -γ ) 2 T ≤ 1 50 (under the condition T ≥ 50 (1 -γ ) 2 ), one has

<!-- formula-not-decoded -->

thus allowing us to take τ = 1 for this case.

It then suffices to look at rescaled linear learning rates (i.e., η t = 1 1+ c η (1 -γ ) t ). As already calculated in the expression (154), the ratio of two consecutive quantities obeys

<!-- formula-not-decoded -->

In what follows, we divide into two cases, depending on whether this sequence is decreasing or increasing.

- The case with 4 / 3 ≤ c η &lt; log T . In this scenario, the ratio in (178) is larger than 1, and hence the sequence { η ( t ) k } decreases with k . Let us define

<!-- formula-not-decoded -->

which clearly satisfies τ ≤ T (in view of (159)). For all t ≥ τ , one has

<!-- formula-not-decoded -->

At the same time, we claim that one must have

<!-- formula-not-decoded -->

Otherwise, recalling η ( T ) 0 = ∏ T i =1 ( 1 -η i (1 -γp ) ) , we have

<!-- formula-not-decoded -->

which contradicts our assumption that η ( T ) 0 &gt; 1 / 75 (cf. (159)).

- The case with 1 -γ &lt; c η &lt; 4 / 3 . In this case, the sequence η ( t ) k increases with k . If we set

<!-- formula-not-decoded -->

then for all t ≥ τ we have

<!-- formula-not-decoded -->

Under the condition T ≥ 150 (1 -γ ) 2 ≥ 150 c η (1 -γ ) (so that T -τ +1 ≥ 100 c η (1 -γ ) ≥ 100 (1 -γ )4 / 3 ), one can show that

<!-- formula-not-decoded -->

- Putting these two cases together (with τ specified in (179) and (181)), we obtain

<!-- formula-not-decoded -->

for all t ≥ τ , thus establishing the desired inequality (169).

Proof of the inequality (173) . For every t , recalling the definition (172), it is convenient to write

<!-- formula-not-decoded -->

where we have used the fact that E [ ξ t ( a )] = 0 . To control the right-hand side of the above equation, let us define

<!-- formula-not-decoded -->

for any k ≥ 1 , where { z k } also forms a martingale sequence since

<!-- formula-not-decoded -->

As a consequence of Freedman's inequality, we claim that ζ t satisfies

<!-- formula-not-decoded -->

To verify this relation, we first notice that

<!-- formula-not-decoded -->

provided that max k η k ∏ t i = k +1 ( 1 -η i ) ≤ η t . To verify the condition max k η k ∏ t i = k +1 ( 1 -η i ) ≤ η t , one can check-similar to (154)-that

<!-- formula-not-decoded -->

which indicates that η k ∏ t i = k +1 ( 1 -η i ) is an increasing sequence as long as c η ≤ log T ≤ 1 1 -γ (see (125)). In addition to the boundedness condition (185), we can further calculate

<!-- formula-not-decoded -->

where the last inequality comes from the facts that ̂ V k -1 ≤ 1 1 -γ and the choice p = 4 γ -1 3 γ . These bounds taken together with Freedman's inequality (see (58)) validate (184).

By virtue of (184), setting δ = (1 -γ ) 2 2 E [ | ζ t | 2 ] yields that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

When T ≥ 1 (1 -γ ) 2 , one can ensure that

<!-- formula-not-decoded -->

Here, (i) holds since

<!-- formula-not-decoded -->

as a consequence of (185) and (69); (ii) holds by the choice of δ . It is thus sufficient to lower bound E [ | ζ t | 2 ] . Towards this, let us define

<!-- formula-not-decoded -->

which clearly satisfies τ ≤ ̂ τ ≤ T (in view of (180) and (182)). Then, for all t ≥ ̂ τ one has (which shall be proved towards the end of this subsection)

<!-- formula-not-decoded -->

We now proceed to lower bound E [ | ζ t | 2 ] for t ≥ ̂ τ . We first observe that for any t ≥ ̂ τ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first line relies on (139), and the last step makes use of the fact (166). To further control the right-hand side of the above inequality, we take τ ′ := max { t -1 η t/ 2 , 1 } and show that

<!-- formula-not-decoded -->

Here, (i) makes use of the constraint ̂ V k -1 ≥ 1 4(1 -γ ) , while (ii) makes use of (190), and (iii) are valid if the following property holds (which shall be proved towards the end of this subsection)

<!-- formula-not-decoded -->

We are now well-equipped to control E [ ξ t, max ] using the property (188). Recall the expression of B in (187), we know that bounding E [ | ζ t | 2 ] /B boils down to controlling

<!-- formula-not-decoded -->

- For the first term in (193), recalling that δ = (1 -γ ) 2 2 E [ | ζ t | 2 ] , we can demonstrate that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality makes use of the bound (191), and the second inequality arises from the fact η t ≥ 1 1+(1 -γ ) T log T (given the range of the learning rates in this case). Combining this with (191), we can guarantee that

- Moving to the second term in (193), one can ensure that

<!-- formula-not-decoded -->

Here, (i) follows from (191) and (194) since

<!-- formula-not-decoded -->

(ii) arises from (192); and (iii) relies on the fact η t glyph[greaterorsimilar] 1 (1 -γ ) T log T (given the range of the learning rates in this case).

Substituting the above relations into (188) and using the expression of B in (187), we reach at

<!-- formula-not-decoded -->

for some constant c &gt; 0 . Thus, this validates the inequality (173).

Proof of the claim (190) . By the definition of ̂ τ in (189), we have ∏ T i = ̂ τ ( 1 -η i (1 -γp ) ) ≤ 6 / 7 . An important observation is that

<!-- formula-not-decoded -->

Here, the relations (i) and (iii) arise from (70), and the inequality (ii) follows since

<!-- formula-not-decoded -->

where τ is defined in (179) and (181) for linearly rescaled learning rates and τ = 1 for constant learning rates, and we have also made use of (159), (180) and (182) in the penultimate inequality in (196).

With (195) in place, we can continue to prove the claim (190). Recognizing that η k [ ∏ t i = k +1 ( 1 -η i )] is increasing in k (see (186)), we can obtain

<!-- formula-not-decoded -->

where the last inequality comes from (195). With the preceding inequality in place, the claim (190) then follows by observing that

<!-- formula-not-decoded -->

where we make use of the monotonicity of η k [ ∏ t i = k +1 ( 1 -η i )] again.

Proof of the claim (192) . Note that for τ ′ := max { t -1 η t/ 2 , 1 } , one has

<!-- formula-not-decoded -->

as long as the following condition holds (recalling the definition of ̂ τ in (189))

<!-- formula-not-decoded -->

In addition, similar to (129), we can derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we once again use the condition (198), and the last inequality comes from the derivation in (196). Putting these two bounds together yields

<!-- formula-not-decoded -->

To finish up, it remains to justify (198). This condition is obvious for constant learning rates. As for rescaled learning rates, one can see that

<!-- formula-not-decoded -->

where τ := glyph[ceilingleft] 19 c η (1 -γ ) glyph[ceilingright] . This allows one to obtain

<!-- formula-not-decoded -->

provided that T ≥ c 1 (1 -γ ) 2 for some sufficiently large constant c 1 &gt; 0 and 1 -γ &lt; c η &lt; log T . Taking this together with (189) implies that ̂ τ ≥ τ and hence η ̂ τ ≤ η τ = 1 1+(1 -γ ) c η τ = 1 / 20 .

## D.4 Proof of Lemma 3

Given that state 0 is an absorbing state with zero immediate reward, it is easily seen that

<!-- formula-not-decoded -->

Moreover, by construction, taking action 1 and taking action 2 in state 1 result in the same behavior (in terms of both the reward function and the associated transition probability), and as a consequence,

<!-- formula-not-decoded -->

From Bellman's equation, we can thus deduce that

<!-- formula-not-decoded -->

which in conjunction with (199) and a little algebra leads to

<!-- formula-not-decoded -->

Here, the second identity follows since V glyph[star] (0) = 0 , and the third identity makes use of (43). The calculation for V glyph[star] (2) and Q glyph[star] (2 , 1) follows from an identical argument and is hence omitted.

Turning to state 3 , by Bellman's equation, we have

<!-- formula-not-decoded -->

which leads to V glyph[star] (3) = 1 1 -γ .

## E Analysis for asynchronous Q-learning (Theorem 4)

## E.1 Notation and preliminary facts

Vector and matrix notation. We shall adopt the vector notation Q t ∈ R |S||A| , V t ∈ R |S| , r ∈ R |S||A| in the same way as in Section 4.1. The sample transition matrix P t ∈ R |S||A|×|S| in the asynchronous case is defined such that

<!-- formula-not-decoded -->

It is also handy to introduce the diagonal matrix Λ t ∈ R |S||A|×|S||A| such that

<!-- formula-not-decoded -->

Armed with the above notation, the asynchronous Q-learning update rule (48) can be conveniently expressed as follows:

<!-- formula-not-decoded -->

Range of V t and Q t . Similar to the synchronous counterpart, we have the following elementary properties. Lemma 7. Suppose that 0 &lt; η t ≤ 1 for all t ≥ 0 . Assume that 0 ≤ Q 0 ≤ 1 1 -γ 1 . Then for any t ≥ 0 , one has

<!-- formula-not-decoded -->

Proof. The proof is the same as that of Lemma 4, and is hence omitted for brevity.

## E.2 Main steps for proving Theorem 4

We are now in a position to outline the main steps for the proof of Theorem 4.

Step 1: deriving basic recursions. According to the update rule (202), we can derive the following elementary decomposition

<!-- formula-not-decoded -->

where the penultimate identity follows from the Bellman optimality equation Q glyph[star] = r + PV glyph[star] . Combining (204) with the inequalities (33) and using the definition (27) of π t result in

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Apply the above two relations recursively to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By defining the following diagonal matrices

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and setting for some constant c 3 &gt; 0 , we can rearrange terms in the upper bound (206a) to reach

<!-- formula-not-decoded -->

In the subsequent steps, we shall first develop bounds on the sizes of the terms ζ t and ξ t in (209) separately, and then combine these bounds with (209) recursively in order to derive the advertised upper bound on ∆ t .

Step 2: bounding the terms ζ t and ξ t . The terms ζ t and ξ t defined in (209) can be bounded with high probability by the following lemmas.

Lemma 8. With probability at least 1 -δ , we have

<!-- formula-not-decoded -->

for all t obeying T c 4 log T ≤ t ≤ T . Here, c 4 &gt; 0 is some constant obeying c 4 ≤ c 1 c 3 / 4 , where the constants c 1 and c 3 appear in (52a) and (208) , respectively.

Proof. See Section E.3.1.

Lemma 9. Suppose that 0 &lt; η ≤ log 3 T (1 -γ ) Tµ min . With probability at least 1 -δ , one has

<!-- formula-not-decoded -->

for all t obeying T c 4 log T ≤ t ≤ T for some constant c 4 &gt; 0 .

Proof. See Section E.3.2.

Step 3: controlling ∆ t . Consider any t obeying T c 4 log T ≤ t ≤ T and any k obeying 2 t/ 3 ≤ k ≤ t . Under the sample size condition (52b), Lemmas 8-9 together with a little algebra lead to where we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining this inequality with (209) allows us to obtain

<!-- formula-not-decoded -->

Similar to the quantity α ( t ) i defined in (76), let us define

<!-- formula-not-decoded -->

which, according to (69) and the definition (201), clearly satisfies

<!-- formula-not-decoded -->

Set i 0 = t for notational convenience. With this set of notation and the property (215) in mind, we can derive the following bound

<!-- formula-not-decoded -->

Here, the first relation makes use of the second property in (215), the second relation further expands ∆ i 1 in the same way as in the first line of (216), whereas the third inequality relies on the first property in (215). Next, we intend to invoke the above relation multiple times to reach a simpler relation. Set

<!-- formula-not-decoded -->

Similar to the way we derive (84), we can apply the relation (216) recursively and use the basic relation | ∆ k | ≤ 1 1 -γ 1 for any k to show that

<!-- formula-not-decoded -->

where I t has been defined in (82). To further simplify (218), we need to control the two terms on the right-hand side of (218) separately.

- We shall begin with the first term on the right-hand side of (218). Towards this end, let us define a collection of policies { ̂ π k } recursively and backward as follows:

<!-- formula-not-decoded -->

or alternatively (in view of the definition (24) of P π ),

<!-- formula-not-decoded -->

Here, Π is a policy set satisfying

<!-- formula-not-decoded -->

in words, for any policy π belonging to Π , each π ( s ) coincides with one of the policy iterates π i ( s ) during the latest βt iterations, although we do not require all { π ( s ) } across different states to be associated with the same time stamp i . With this collection of policies in place, we can deduce that

<!-- formula-not-decoded -->

where we abbreviate ∑ ( i 1 , ··· ,i H ) ∈I t as ∑ i 1 , ··· ,i H as long as it is clear from the context. Here, (i) and (iii) arise from the second property in (215), while (ii) and (iv) are due to the construction (220). Continuing the derivation of the above inequality recursively, we arrive at

<!-- formula-not-decoded -->

- We now turn attention to the second term on the right-hand side of (218). It is seen that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first line follows from the first property in (215), the third line is due to the fact P π 1 = 1 for any π , and the fourth line arises from the second property in (215).

Substituting the above two bounds into (218) yields

<!-- formula-not-decoded -->

Step 4: putting all pieces together. Repeating our analysis for the term β 1 in Section B.2 (i.e., Step 5 of Section B.2 with Lemma 5 replaced by Lemma 6), we arrive at

<!-- formula-not-decoded -->

with probability at least 1 -δ . Substitution into (222) then yields

<!-- formula-not-decoded -->

holds simultaneously for all t ≥ T c 4 log T , provided that the sample size condition (52b) is satisfied. Similarly, we can also establish the following lower bound on ∆ t (which we omit the details for the sake of brevity)

<!-- formula-not-decoded -->

with probability at least 1 -δ . To summarize, it is seen that with probability exceeding 1 -2 δ ,

<!-- formula-not-decoded -->

This resembles the relation (40) derived for the synchronous case, except that T in the denominator is replaced with µ min T . As a result, we can readily repeat the argument in Appendix B.4 to reach

<!-- formula-not-decoded -->

which in turn establishes the claimed result in Theorem 4.

## E.3 Proofs of technical lemmas

## E.3.1 Proof of Lemma 8

In view of the definition of ζ t in (209) and the fact that Λ ( t ) 0 is a diagonal matrix, we can deduce that

<!-- formula-not-decoded -->

Here, (i) holds true since ‖ P π i -1 ‖ 1 = ‖ P i ‖ 1 = ‖ P ‖ 1 = 1 . To verify (ii), we first define t k ( s, a ) := the time stamp when the trajectory visits ( s, a ) for the k -th time (226)

and

<!-- formula-not-decoded -->

namely, the total number of times - before the t -th iteration - that the sample trajectory visits ( s, a ) . Then Li et al. (2022c, Lemma 8) tells us that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

holds uniformly for all ( s, a ) ∈ S × A and 0 ≤ t 2 ≤ t 1 ≤ T obeying

<!-- formula-not-decoded -->

This in turn implies that: if βt ≥ 886 t mix µ min log |S||A| T δ and i ≤ (1 -β ) t , then one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -δ , provided that ηβtµ min &gt; 4 log T . In other words, (229) holds with probability at least 1 -δ , as long as

<!-- formula-not-decoded -->

This taken together with the sample size assumption (52b) concludes the proof of Lemma 8.

## E.3.2 Proof of Lemma 9

Fix any state-action pair ( s, a ) ∈ S × A , and let us look at the ( s, a ) -th entry of ξ t , i.e., ξ t ( s, a ) . For notational simplicity, let Λ j ( s, a ) denote the ( s, a ) -th diagonal entry of the diagonal matrix Λ j , and P t ( s, a ) (resp. P ( s, a ) ) the ( s, a ) -th row of P t (resp. P ).

Using the definition of ξ t in (209) and the above notation, we can derive

<!-- formula-not-decoded -->

Equipped with the definitions of t k ( s, a ) (cf. (226)) and K t ( s, a ) (cf. (227)), we can further rewrite (230) as

<!-- formula-not-decoded -->

In what follows, we shall suppress the notation and write t k = t k ( s, a ) and K t = K t ( s, a ) to streamline notation.

The main step thus boils down to controlling (231). Towards this, we claim that: with probability at least 1 -δ ,

<!-- formula-not-decoded -->

holds simultaneously for all ( s, a ) ∈ S × A and all 1 ≤ K β ≤ K ≤ T , provided that 0 &lt; η ≤ log 3 T (1 -γ ) Tµ min . If this claim were true, then taking K β = K (1 -β ) t +1 and K = K t and substituting the bound (232) into the expression (231) would lead to

<!-- formula-not-decoded -->

thus concluding the proof of this lemma. To finish up, it is sufficient to justify the claim (232), which forms the content of the remainder of this proof.

Proof of the claim (232). Let us use the notation in (64) to express η ( K ) k = (1 -η ) K -k η . For any fixed integer K &gt; 0 , the following vectors

<!-- formula-not-decoded -->

are identically and independently distributed; see Li et al. (2022c, Section B.1). We can then express the term

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as follows:

where the z k 's satisfy

<!-- formula-not-decoded -->

We intend to invoke the Freedman inequality to control X K for any K obeying K ≤ T . Similar to the synchronous counterpart, we can see that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have made use of (69). In addition, we make note of a trivial upper bound on W as follows

<!-- formula-not-decoded -->

With the preceding bounds in place, applying the Freedman inequality in Theorem 6 and taking L = log 2 1 1 -γ imply that

<!-- formula-not-decoded -->

with probability at least 1 -δ |S||A| T 2 , provided that η ≤ log 3 T (1 -γ ) Tµ min . We can thus conclude the proof by taking the union bound over all ( s, a ) ∈ S × A and all 1 ≤ K β ≤ K ≤ T .

## F Lower bound for asynchronous Q-learning (Theorem 5)

This section establishes Theorem 5 by identifying a hard MDP instance satisfying the assumed conditions.

## F.1 Construction of a hard instance and its values

Let us construct an MDP M hard with state space S = { 0 , 1 , 2 , 3 } as follows, which is partly inspired by the idea from Li et al. (2022b). We shall denote by A s the action space associated with state s . The probability transition kernel and the reward function of M hard are specified as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the parameter µ = [ µ 0 , µ 1 , µ 2 , µ 3 ] ∈ ∆( S ) is set as

<!-- formula-not-decoded -->

for some sufficiently small quantity 0 &lt; c µ = O (1) . In particular, actions 1 and 2 from state 1 are identical, which will play a similar critical role as the case of synchronous Q-learning in pinpointing the 'over-estimation' issue.

In addition, let the behavior policy π b be uniform distributions such that

<!-- formula-not-decoded -->

Then the transition probability from state s to state s ′ under the behavior policy π b is given by

<!-- formula-not-decoded -->

With this in mind, it can be easily verified that the stationary distribution under π b is given by

<!-- formula-not-decoded -->

This together with (235) implies that

<!-- formula-not-decoded -->

Moreover, if the sample trajectory is initialized with an initial state distribution µ 0 , then the marginal state distribution at time t can be calculated as

<!-- formula-not-decoded -->

thus indicating that the total variation distance d TV between µ t and µ obeys

<!-- formula-not-decoded -->

Consequently, the mixing time of the sample trajectory (Paulin, 2015) obeys

<!-- formula-not-decoded -->

for some universal constants c mix , 1 , c mix , 2 &gt; 0 .

Before embarking on the analysis of the behavior of asynchronous Q-learning, let us first look at the optimal value function and Q-function of the constructed MDP.

Lemma 10. Consider the MDP M hard as constructed in (234) . It holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. In order to justify (238), let us begin by defining two vectors V and Q as follows:

<!-- formula-not-decoded -->

where r = [ r ( s, a )] denotes the reward vector. Then the claimed expressions of the value function in (238) are valid as long as we can validate V ( s ) = max a Q ( s, a ) (namely, they satisfy the Bellman equation). These are elementary calculations, which we omit for brevity. Once the expressions of both the value function and the Q-function are settled, the remaining set of advertised inequalities can be validated straightforwardly, which is omitted as well for conciseness.

## F.2 Analysis for the constructed MDP

We now proceed to analyze the dynamics of asynchronous Q-learning when applied to the above MDP instance, for which we divide into three cases based on the magnitudes of the learning rates. Throughout the proof, we denote by t k ( s, a ) the iteration number corresponding to the k -th time the state-action pair ( s, a ) is visited, and let N T ( s, a ) represent the total number of visits to ( s, a ) up to time T . We shall also reuse the notation of the sample transition matrix P t defined in (200), as well as the value estimate vector V t = [ V t ( s )] s ∈S as in Appendix E.1.

## F.2.1 Case 1: small learning rates ( η &lt; 1 µ min (1 -γ ) T )

In this case, we focus attention on analyzing state 3 . To begin with, we claim that if η ≤ c 8 (1 -γ ) 2 log T for some sufficiently small constant c 8 &gt; 0 , then one has, for all t ≤ T ,

<!-- formula-not-decoded -->

Given the assumption µ min T ≥ c 3 log T (1 -γ ) 4 , the regime η &lt; 1 µ min (1 -γ ) T clearly satisfies η ≤ c 8 (1 -γ ) 2 log T .

Proof of Claim (239) . We would like to prove the second inequality in (239) by induction. Clearly, it suffices to look at those iterations where the value of Q t (3 , 2) changes, namely, { t k (3 , 2) } k ≥ 1 . Assume for the moment that (239) holds for all t &lt; t k (3 , 2) . Taking V := [ 1 1 -γ , 1 1 -γ , 1 1 -γ , V glyph[star] (3) ] glyph[latticetop] , we can invoke the asynchronous Q-learning update rule iteratively to deduce that

<!-- formula-not-decoded -->

Recognizing that

<!-- formula-not-decoded -->

we can readily obtain

<!-- formula-not-decoded -->

Combine this with the Q-learning update rule to arrive at

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the second line holds since V serves as an entrywise upper bound on V t ; the third line follows since ∑ k i =1 η (1 -η ) k -i ≤ 1 ; the fourth line in (240) is valid since, according to the Bernstein inequality (see Li et al. (2022c, Lemma 1)),

<!-- formula-not-decoded -->

holds with probability at least 1 -1 /T ; and the validity of the last inequality can be shown by observing that

<!-- formula-not-decoded -->

where we have used the facts that V (3) = V glyph[star] (3) = 3 4(1 -γ ) , P (3 | 3 , 2) = 1 -2(1 -γ )(1 -µ 3 ) and γ &lt; 1 . Thus, standard induction arguments immediately validate the second inequality in the claim (239) for all t ≤ T .

Regarding the first inequality of (239), it is seen that for any k ≥ 1 ,

<!-- formula-not-decoded -->

where we have used the facts that Q 0 ≡ 0 and V (3) = V glyph[star] (3) = 3 4(1 -γ ) and the elementary inequality ∑ k i =1 η (1 -η ) k -i ≤ 1 . Given that the above bound holds for all k ≥ 1 (and { t k (3 , 1) } k ≥ 1 correspond to all iterations when the value of Q t (3 , 1) changes), we have established the first advertised inequality in (239).

Next, let us define

<!-- formula-not-decoded -->

From this definition and Claim (239), we know that for any k &gt; s , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality holds since V glyph[star] (3) = 3 4(1 -γ ) . This taken together with (241) (so that V glyph[star] (3) -Q t s (3 , 1) (3 , 1) ≥ 1 / 8 ) leads to

<!-- formula-not-decoded -->

for some constant c 9 &gt; 0 ; here, the last inequality holds since, according to Li et al. (2022c, Lemma 8),

<!-- formula-not-decoded -->

occurs with probability at least 1 -1 /T , provided that T ≥ 443 t mix log(10 T ) µ min . Note that according to (236) and (237), t mix µ min is on the order of log 2 T c µ (1 -γ ) .

Putting (239) and (242) together then reveals that with probability at least 1 -2 /T ,

<!-- formula-not-decoded -->

## F.2.2 Case 2: large learning rates ( η &gt; log T µ min (1 -γ ) 2 T )

Case 2.1: η ≥ c 8 (1 -γ ) 2 log T for some small enough constant c 8 &gt; 0 . Under the condition that η ≥ c 8 (1 -γ ) 2 log T , we claim that with probability at least 1 -γ 50 ,

<!-- formula-not-decoded -->

for some universal constant c 10 &gt; 0 . We shall first prove this claim.

Proof of Claim (244) . We shall focus attention on the case where ( s T -1 , a T -1 ) = (2 , 1) . Given that the stationary distribution obeys µ b (2 , 1) = 1 / 10 and that T is sufficiently large (so that the empirical distribution approaches the stationary distribution), we know that

<!-- formula-not-decoded -->

Let us first look at the case where ∣ ∣ V T -1 (0) -V glyph[star] (0) ∣ ∣ &gt; 1 27(1 -γ ) . It follows from P ( · | 0 , 1) = [1 , 0 , 0 , 0] that

<!-- formula-not-decoded -->

as long as γ ≥ 0 . 95 , which clearly satisfies (244). When it comes to the complement case where ∣ ∣ V T -1 (0) -V glyph[star] (0) ∣ ∣ ≤ 1 27(1 -γ ) , either of the following two scenarios will happen:

- If | V T -1 (2) -V T -1 (0) | ≤ 1 27(1 -γ ) , then the assumption ∣ ∣ V T -1 (0) -V glyph[star] (0) ∣ ∣ ≤ 1 27(1 -γ ) yields

<!-- formula-not-decoded -->

where the second line holds since, for γ ≥ 0 . 95 ,

<!-- formula-not-decoded -->

and the last inequality holds since Q glyph[star] (2 , 2) &gt; r (2 , 2) + γP (2 | 2 , 2) V glyph[star] (2) (from the Bellman equation).

- Consider instead the scenario with | V T -1 (2) -V T -1 (0) | &gt; 1 27(1 -γ ) . For notational convenience, define

<!-- formula-not-decoded -->

Recognizing that min { P (0 | 2 , 1) , P (2 | 2 , 1) } ≥ 2(1 -γ ) µ 0 , we can show that with probability at least (1 -γ ) µ 0 = 2(1 -γ ) 5 ,

<!-- formula-not-decoded -->

where the last inequality holds since η ≥ c 8 (1 -γ ) 2 log T and γ ≥ 0 . 95 .

We can thus conclude that for all the above scenarios, the claim (244) holds with probability at least 1 -γ 50 .

Case 2.2: log T µ min (1 -γ ) 2 T &lt; η c 8 (1 -γ ) 2 log T . Recall from Claim (239) that: when η c 8 (1 -γ ) 2 log T , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, for any k ≥ log T η (1 -γ ) , we can use P ( · | 3 , 1) = [0 , 0 , 0 , 1] and the Bellman equation to derive

<!-- formula-not-decoded -->

where the last line is valid since V glyph[star] (3) = 3 4(1 -γ ) and Q 0 (3 , 1) = 0 . It is also seen from Li et al. (2022c, Lemma 8) that, with probability at least 1 -1 /T ,

<!-- formula-not-decoded -->

as long as η ≥ log T µ min (1 -γ ) T . The above two results taken collectively yield

<!-- formula-not-decoded -->

Next, we move on to analyze state 2 . Towards this end, let us define

<!-- formula-not-decoded -->

Note that when η ≤ c 8 (1 -γ ) 2 log T , one has

<!-- formula-not-decoded -->

which combined with (248) implies that Q t k -1 (2 , 1) (2 , 2) &lt; Q t k -1 (2 , 1) (2 , 1) for any k &gt; s and hence

<!-- formula-not-decoded -->

This crucial identity together with the construction of P ( · | 2 , 1) in turn allows one to derive, for any k &gt; s ,

<!-- formula-not-decoded -->

thus leading to

<!-- formula-not-decoded -->

where the second inequality arises from the Bellman equation (so that V glyph[star] (2) ≥ 1 + 2 γ (1 -γ ) µ 0 V glyph[star] (0) ).

<!-- formula-not-decoded -->

- Otherwise, consider the case where ( 1 -η (1 -γ )(1 + 2 γµ 0 ) ) k -s &lt; 1 2 . The expression (250) allows one to control the variance as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third inequality holds since V t i (2 , 1) -1 (2) -V t i (2 , 1) -1 (0) ≥ V glyph[star] (2) -1 4 -V glyph[star] (0) &gt; 5 27(1 -γ ) (using the definition of s in (248)), and the last line uses γ ≥ 0 . 95 and µ 0 = 2 / 5 . As a result,

<!-- formula-not-decoded -->

Taking the above two results together reveals that

<!-- formula-not-decoded -->

Combining Case 2.1 and Case 2.2. Putting together (244) and (253) together leads to

<!-- formula-not-decoded -->

provided that η &gt; log T µ min (1 -γ ) 2 T .

## F.2.3 Case 3: medium learning rates ( 1 µ min (1 -γ ) T ≤ η ≤ log T µ min (1 -γ ) 2 T )

We now shift attention to the dynamics underlying state 1 , and look at its associated value function V t (1) .

Two auxiliary sequences. Before proceeding, we abuse notation by defining

<!-- formula-not-decoded -->

and letting N ( t k -1 ,t k ] (1 , a, 1) (resp. N ( t k -1 ,t k ] (1 , a ) ) denote the number of times the sample trajectory visits ( s, a, s ′ ) = (1 , a, 1) (resp. ( s, a ) = (1 , a ) ) within the time interval ( t k -1 , t k ] . From standard Bernstein's inequality (see Li et al. (2022c, Lemma 8)) and the definition of t i , one can easily see that with probability at least 1 -1 /T ,

<!-- formula-not-decoded -->

holds simultaneously for all t i ≤ T , where c 11 &gt; 0 is some suitable constant. In turn, this bound (255) also implies that

<!-- formula-not-decoded -->

for some constant c 12 &gt; 0 . These follow from fairly standard concentration arguments and are hence omitted.

Under our assumption on T , the sample trajectory mixes well after T/ 3 iterations. In order to further remove the effect of Q T/ 3 (1 , a ) , let us introduce the following auxiliary sequence

<!-- formula-not-decoded -->

where ̂ V k -1 := max a ̂ Q k -1 ( a ) ,

<!-- formula-not-decoded -->

Repeating the proof of (163) (which we omit here for brevity), we arrive at the following relation:

<!-- formula-not-decoded -->

Furthermore, in order to control ̂ Q k ( a ) , we construct an additional auxiliary sequence as follows

<!-- formula-not-decoded -->

From the basic fact that ̂ V k = max a ̂ Q k ( a ) ≥ ̂ Q k (1) , it can be easily verified that

<!-- formula-not-decoded -->

which in turn motivates us to lower bound ̂ V k by controlling Q k . Using similar analysis for (166), we reach

<!-- formula-not-decoded -->

Main proof. With the preceding auxiliary sequences in place, let us define (akin to the synchronous case)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Based on the iterative update rule (257), we can once again derive

<!-- formula-not-decoded -->

where p = 1 -3 2 (1 -γ ) µ 0 . Then adopting the same analysis as for the synchronous case, we arrive at

<!-- formula-not-decoded -->

for some constant c 13 &gt; 0 . Here, we have made use of the fact that

<!-- formula-not-decoded -->

an immediate consequence of (256) and the assumption that η (1 -γ ) ≥ 1 µ min T = 3 log 2 T c µ T .

## F.2.4 Putting all this together

Combining (243), (254), and (264) leads to

<!-- formula-not-decoded -->

for any 0 &lt; η &lt; 1 . Then the conclusion is handy under the proviso that T ≥ c 3 µ min (1 -γ ) 7 log T .

## References

- Agarwal, A., Kakade, S., and Yang, L. F. (2020). Model-based reinforcement learning with a generative model is minimax optimal. Conference on Learning Theory , pages 67-83.
- Azar, M. G., Munos, R., Ghavamzadeh, M., and Kappen, H. (2011). Reinforcement learning with a near optimal rate of convergence. Technical report, INRIA.
- Azar, M. G., Munos, R., and Kappen, H. J. (2013). Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3):325-349.
- Bai, Y., Xie, T., Jiang, N., and Wang, Y.-X. (2019). Provably efficient Q-learning with low switching cost. In Advances in Neural Information Processing Systems , pages 8002-8011.
- Beck, C. L. and Srikant, R. (2012). Error bounds for constant step-size Q-learning. Systems &amp; control letters , 61(12):1203-1208.
- Bellman, R. (1952). On the theory of dynamic programming. Proceedings of the National Academy of Sciences of the United States of America , 38(8):716.
- Bertsekas, D. P. (2017). Dynamic programming and optimal control (4th edition) . Athena Scientific.
- Bhandari, J., Russo, D., and Singal, R. (2021). A finite time analysis of temporal difference learning with linear function approximation. Operations Research .
- Borkar, V. S. and Meyn, S. P. (2000). The ODE method for convergence of stochastic approximation and reinforcement learning. SIAM Journal on Control and Optimization , 38(2):447-469.
- Cai, Q., Yang, Z., Lee, J. D., and Wang, Z. (2019). Neural temporal-difference and Q-learning converges to global optima. In Advances in Neural Information Processing Systems , pages 11312-11322.
- Chen, Z., Maguluri, S. T., Shakkottai, S., and Shanmugam, K. (2020). Finite-sample analysis of stochastic approximation using smooth convex envelopes. arXiv preprint arXiv:2002.00874 .
- Chen, Z., Maguluri, S. T., Shakkottai, S., and Shanmugam, K. (2021). A Lyapunov theory for finite-sample guarantees of asynchronous Q-learning and TD-learning variants. arXiv preprint arXiv:2102.01567 .
- Chen, Z., Zhang, S., Doan, T. T., Maguluri, S. T., and Clarke, J.-P. (2019). Performance of Q-learning with linear function approximation: Stability and finite-time analysis. arXiv preprint arXiv:1905.11425 .
- Devraj, A. M. and Meyn, S. P. (2020). Q-learning with uniformly bounded variance: Large discounting is not a barrier to fast learning. arXiv preprint arXiv:2002.10301 .
- Doan, T., Maguluri, S., and Romberg, J. (2019). Finite-time analysis of distributed TD(0) with linear function approximation on multi-agent reinforcement learning. In International Conference on Machine Learning , pages 1626-1635. PMLR.
- Even-Dar, E. and Mansour, Y. (2003). Learning rates for Q-learning. Journal of machine learning Research , 5(Dec):1-25.
- Fan, J., Wang, Z., Xie, Y., and Yang, Z. (2019). A theoretical analysis of deep Q-learning. arXiv preprint arXiv:1901.00137 .
- Freedman, D. A. (1975). On tail probabilities for martingales. the Annals of Probability , pages 100-118.
- Gupta, H., Srikant, R., and Ying, L. (2019). Finite-time performance bounds and adaptive learning rate selection for two time-scale reinforcement learning. In Advances in Neural Information Processing Systems , pages 4706-4715.
- Hasselt, H. (2010). Double Q-learning. Advances in neural information processing systems , 23:2613-2621.

- Hu, J. and Wellman, M. P. (2003). Nash Q-learning for general-sum stochastic games. Journal of machine learning research , 4(Nov):1039-1069.
- Jaakkola, T., Jordan, M. I., and Singh, S. P. (1994). Convergence of stochastic iterative dynamic programming algorithms. In Advances in neural information processing systems , pages 703-710.
- Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. (2018). Is Q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4863-4873.
- Johnson, R. and Zhang, T. (2013). Accelerating stochastic gradient descent using predictive variance reduction. In Advances in neural information processing systems , pages 315-323.
- Kakade, S. (2003). On the sample complexity of reinforcement learning . PhD thesis, University of London.
- Kearns, M., Mansour, Y., and Ng, A. Y. (2002). A sparse sampling algorithm for near-optimal planning in large Markov decision processes. Machine learning , 49(2-3):193-208.
- Kearns, M. J. and Singh, S. P. (1999). Finite-sample convergence rates for Q-learning and indirect algorithms. In Advances in neural information processing systems , pages 996-1002.
- Khamaru, K., Pananjady, A., Ruan, F., Wainwright, M. J., and Jordan, M. I. (2021a). Is temporal difference learning optimal? an instance-dependent analysis. SIAM Journal on Mathematics of Data Science , 3(4):1013-1040.
- Khamaru, K., Xia, E., Wainwright, M. J., and Jordan, M. I. (2021b). Instance-optimality in optimal value estimation: Adaptivity via variance-reduced Q-learning. arXiv preprint arXiv:2106.14352 .
- Lakshminarayanan, C. and Szepesvari, C. (2018). Linear stochastic approximation: How far does constant step-size and iterate averaging go? In International Conference on Artificial Intelligence and Statistics , pages 1347-1355.
- Lee, D. and He, N. (2018). Stochastic primal-dual Q-learning. arXiv preprint arXiv:1810.08298 .
- Li, G., Chi, Y., Wei, Y., and Chen, Y. (2022a). Minimax-optimal multi-agent RL in Markov games with a generative model. Neural Information Processing Systems (NeurIPS) .
- Li, G., Shi, L., Chen, Y., and Chi, Y. (2021). Breaking the sample complexity barrier to regret-optimal model-free reinforcement learning. accepted to Information and Inference: A Journal of the IMA .
- Li, G., Shi, L., Chen, Y., Chi, Y., and Wei, Y. (2022b). Settling the sample complexity of model-based offline reinforcement learning. arXiv preprint arXiv:2204.05275 .
- Li, G., Wei, Y., Chi, Y., and Chen, Y. (2023). Breaking the sample size barrier in model-based reinforcement learning with a generative model. accepted to Operations Research .
- Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2022c). Sample complexity of asynchronous Q-learning: Sharper analysis and variance reduction. IEEE Transactions on Information Theory , 68(1):448-473.
- Mou, W., Li, C. J., Wainwright, M. J., Bartlett, P. L., and Jordan, M. I. (2020). On linear stochastic approximation: Fine-grained Polyak-Ruppert and non-asymptotic concentration. arXiv preprint arXiv:2004.04719 .
- Murphy, S. (2005). A generalization error for Q-learning. Journal of Machine Learning Research , 6:1073-1097.
- Pananjady, A. and Wainwright, M. J. (2020). Instance-dependent glyph[lscript] ∞ -bounds for policy evaluation in tabular reinforcement learning. IEEE Transactions on Information Theory , 67(1):566-585.
- Paulin, D. (2015). Concentration inequalities for Markov chains by Marton couplings and spectral methods. Electronic Journal of Probability , 20.
- Polyak, B. T. and Juditsky, A. B. (1992). Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization , 30(4):838-855.

- Qu, G. and Wierman, A. (2020). Finite-time analysis of asynchronous stochastic approximation and Q-learning. Conference on Learning Theory , pages 3185-3205.
- Robbins, H. and Monro, S. (1951). A stochastic approximation method. The annals of mathematical statistics , pages 400-407.
- Shah, D. and Xie, Q. (2018). Q-learning with nearest neighbors. In Advances in Neural Information Processing Systems , pages 3111-3121.
- Shi, L., Li, G., Wei, Y., Chen, Y., and Chi, Y. (2022). Pessimistic Q-learning for offline reinforcement learning: Towards optimal sample complexity. International Conference on Machine Learning .
- Sidford, A., Wang, M., Wu, X., Yang, L., and Ye, Y. (2018). Near-optimal time and sample complexities for solving Markov decision processes with a generative model. In Advances in Neural Information Processing Systems , pages 5186-5196.
- Srikant, R. and Ying, L. (2019). Finite-time error bounds for linear stochastic approximation and TD learning. In Conference on Learning Theory , pages 2803-2830.
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine learning , 3(1):9-44.
- Sutton, R. S. and Barto, A. G. (2018). Reinforcement learning: An introduction . MIT press.
- Szepesvári, C. (1998). The asymptotic convergence-rate of Q-learning. In Advances in Neural Information Processing Systems , pages 1064-1070.
- Tropp, J. (2011). Freedman's inequality for matrix martingales. Electronic Communications in Probability , 16:262-270.
- Tsitsiklis, J. and Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. IEEE Transactions on Automatic Control , 42(5):674-690.
- Tsitsiklis, J. N. (1994). Asynchronous stochastic approximation and Q-learning. Machine learning , 16(3):185202.
- Tsybakov, A. B. and Zaiats, V. (2009). Introduction to nonparametric estimation , volume 11. Springer.
- Wai, H.-T., Hong, M., Yang, Z., Wang, Z., and Tang, K. (2019). Variance reduced policy evaluation with smooth function approximation. Advances in Neural Information Processing Systems , 32:5784-5795.
- Wainwright, M. (2019a). High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press.
- Wainwright, M. J. (2019b). Stochastic approximation with cone-contractive operators: Sharp glyph[lscript] ∞ -bounds for Q-learning. arXiv preprint arXiv:1905.06265 .
- Wainwright, M. J. (2019c). Variance-reduced Q-learning is minimax optimal. arXiv preprint arXiv:1906.04697 .
- Watkins, C. J. and Dayan, P. (1992). Q-learning. Machine learning , 8(3-4):279-292.
- Watkins, C. J. C. H. (1989). Learning from delayed rewards.
- Weng, B., Xiong, H., Zhao, L., Liang, Y., and Zhang, W. (2020a). Momentum Q-learning with finite-sample convergence guarantee. arXiv preprint arXiv:2007.15418 .
- Weng, W., Gupta, H., He, N., Ying, L., and Srikant, R. (2020b). The mean-squared error of double Q-learning. Advances in Neural Information Processing Systems , 33.
- Wu, Y., Zhang, W., Xu, P., and Gu, Q. (2020). A finite time analysis of two time-scale actor critic methods. arXiv preprint arXiv:2005.01350 .

- Xiong, H., Zhao, L., Liang, Y., and Zhang, W. (2020). Finite-time analysis for double Q-learning. Advances in Neural Information Processing Systems , 33.
- Xu, P. and Gu, Q. (2020). A finite-time analysis of Q-learning with neural network function approximation. In International Conference on Machine Learning , pages 10555-10565. PMLR.
- Xu, T., Wang, Z., Zhou, Y., and Liang, Y. (2019a). Reanalysis of variance reduced temporal difference learning. In International Conference on Learning Representations .
- Xu, T., Zou, S., and Liang, Y. (2019b). Two time-scale off-policy TD learning: Non-asymptotic analysis over Markovian samples. In Advances in Neural Information Processing Systems , pages 10633-10643.
- Yan, Y., Li, G., Chen, Y., and Fan, J. (2022). The efficacy of pessimism in asynchronous Q-learning. arXiv preprint arXiv:2203.07368 .
- Zhang, Z., Zhou, Y., and Ji, X. (2020). Almost optimal model-free reinforcement learning via referenceadvantage decomposition. Advances in Neural Information Processing Systems , 33.