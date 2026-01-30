## Finite-Time Analysis for Double Q-learning

Huaqing Xiong 1 , Lin Zhao 2 , Yingbin Liang 1 , and Wei Zhang ∗ .3,4

1

The Ohio State University 2 National University of Singapore 3 Southern University of Science and Technology

4 Peng Cheng Laboratory

1 {xiong.309, liang.889}@osu.edu;

2 elezhli@nus.edu.sg;

## Abstract

Although Q-learning is one of the most successful algorithms for finding the best action-value function (and thus the optimal policy) in reinforcement learning, its implementation often suffers from large overestimation of Q-function values incurred by random sampling. The double Q-learning algorithm proposed in Hasselt (2010) overcomes such an overestimation issue by randomly switching the update between two Q-estimators, and has thus gained significant popularity in practice. However, the theoretical understanding of double Q-learning is rather limited. So far only the asymptotic convergence has been established, which does not characterize how fast the algorithm converges. In this paper, we provide the first non-asymptotic (i.e., finite-time) analysis for double Q-learning. We show that both synchronous and asynchronous double Q-learning are guaranteed to converge to an glyph[epsilon1] -accurate

<!-- formula-not-decoded -->

iterations, where ω ∈ (0 , 1) is the decay parameter of the learning rate, and γ is the discount factor. Our analysis develops novel techniques to derive finite-time bounds on the difference between two inter-connected stochastic processes, which is new to the literature of stochastic approximation.

## 1 Introduction

Q-learning is one of the most successful classes of reinforcement learning (RL) algorithms, which aims at finding the optimal action-value function or Q-function (and thus the associated optimal policy) via off-policy data samples. The Q-learning algorithm was first proposed by Watkins and Dayan (1992), and since then, it has been widely used in various applications including robotics (Tai and Liu, 2016), autonomous driving (Okuyama et al., 2018), video games (Mnih et al., 2015), to name a few. Theoretical performance of Q-learning has also been intensively explored. The asymptotic convergence has been established in Tsitsiklis (1994); Jaakkola et al. (1994); Borkar and Meyn (2000); Melo (2001); Lee and He (2019). The non-asymptotic (i.e., finite-time) convergence rate of Q-learning was firstly obtained in Szepesvári (1998), and has been further studied in (Even-Dar and Mansour, 2003; Shah and Xie, 2018; Wainwright, 2019; Beck and Srikant, 2012; Chen et al., 2020) for synchronous Q-learning and in (Even-Dar and Mansour, 2003; Qu and Wierman, 2020) for asynchoronous Q-learning.

One major weakness of Q-learning arises in practice due to the large overestimation of the actionvalue function (Hasselt, 2010; Hasselt et al., 2016). Practical implementation of Q-learning involves using the maximum sampled Q-function to estimate the maximum expected Q-function (where the

∗ Corresponding author

3,4 zhangw3@sustech.edu.cn

expectation is taken over the randomness of reward). Such an estimation often yields a large positive bias error (Hasselt, 2010), and causes Q-learning to perform rather poorly. To address this issue, double Q-learning was proposed in Hasselt (2010), which keeps two Q-estimators (i.e., estimators for Q-functions), one for estimating the maximum Q-function value and the other one for update, and continuously changes the roles of the two Q-estimators in a random manner. It was shown in Hasselt (2010) that such an algorithm effectively overcomes the overestimation issue of the vanilla Q-learning. In Hasselt et al. (2016), double Q-learning was further demonstrated to substantially improve the performance of Q-learning with deep neural networks (DQNs) for playing Atari 2600 games. It inspired many variants (Zhang et al., 2017; Abed-alguni and Ottom, 2018), received a lot of applications (Zhang et al., 2018a,b), and have become one of the most common techniques for applying Q-learning type of algorithms (Hessel et al., 2018).

Despite its tremendous empirical success and popularity in practice, theoretical understanding of double Q-learning is rather limited. Only the asymptotic convergence was provided in Hasselt (2010); Weng et al. (2020c). There has been no non-asymptotic result on how fast double Q-learning converges. From the technical standpoint, such finite-time analysis for double Q-learning does not follow readily from those for the vanilla Q-learning, because it involves two randomly updated Q-estimators, and the coupling between these two random paths significantly complicates the analysis. This goes much more beyond the existing techniques for analyzing the vanilla Q-learning that handles the random update of a single Q-estimator. Thus, the goal of this paper is to develop new finite-time analysis techniques that handle the inter-connected two random path updates in double Q-learning and provide the convergence rate.

## 1.1 Our contributions

The main contribution of this paper lies in providing the first finite-time analysis for double Q-learning with both the synchronous and asynchronous implementations.

- We show that synchronous double Q-learning with a learning rate α t = 1 /t ω (where ω ∈ (0 , 1) ) attains an glyph[epsilon1] -accurate global optimum with at least the probability of 1 -δ by taking Ω ( ( 1 (1 -γ ) 6 glyph[epsilon1] 2 ln |S||A| (1 -γ ) 7 glyph[epsilon1] 2 δ ) 1 ω + ( 1 1 -γ ln 1 (1 -γ ) 2 glyph[epsilon1] ) 1 1 -ω ) iterations, where γ ∈ (0 , 1) is the discount factor, |S| and |A| are the sizes of the state space and action space, respectively.
- We further show that under the same accuracy and high probability requirements, asynchronous double Q-learning takes Ω ( ( L 4 (1 -γ ) 6 glyph[epsilon1] 2 ln |S||A| L 4 (1 -γ ) 7 glyph[epsilon1] 2 δ ) 1 ω + ( L 2 1 -γ ln 1 (1 -γ ) 2 glyph[epsilon1] ) 1 1 -ω ) iterations, where L is the covering number specified by the exploration strategy.

Our results corroborate the design goal of double Q-learning, which opts for better accuracy by making less aggressive progress during the execution in order to avoid overestimation. Specifically, our results imply that in the high accuracy regime, double Q-learning achieves the same convergence rate as vanilla Q-learning in terms of the order-level dependence on glyph[epsilon1] , which further indicates that the high accuracy design of double Q-learning dominates the less aggressive progress in such a regime. In the low-accuracy regime, which is not what double Q-learning is designed for, the cautious progress of double Q-learning yields a slightly weaker convergence rate than Q-learning in terms of the dependence on 1 -γ .

From the technical standpoint, our proof develops new techniques beyond the existing finite-time analysis of the vanilla Q-learning with a single random iteration path. More specifically, we model the double Q-learning algorithm as two alternating stochastic approximation (SA) problems, where one SA captures the error propagation between the two Q-estimators, and the other captures the error dynamics between the Q-estimator and the global optimum. For the first SA, we develop new techniques to provide the finite-time bounds on the two inter-related stochastic iterations of Q-functions. Then we develop new tools to bound the convergence of Bernoulli-controlled stochastic iterations of the second SA conditioned on the first SA.

## 1.2 Related work

Due to the rapidly growing literature on Q-learning, we review only the theoretical results that are highly relevant to our work.

Q-learning was first proposed in Watkins and Dayan (1992) under finite state-action space. Its asymptotic convergence has been established in Tsitsiklis (1994); Jaakkola et al. (1994); Borkar and Meyn (2000); Melo (2001) through studying various general SA algorithms that include Q-learning as a special case. Along this line, Lee and He (2019) characterized Q-learning as a switched linear system and applied the results of Borkar and Meyn (2000) to show the asymptotic convergence, which was also extended to other Q-learning variants. Another line of research focuses on the finite-time analysis of Q-learning which can capture the convergence rate. Such non-asymptotic results were firstly obtained in Szepesvári (1998). A more comprehensive work (Even-Dar and Mansour, 2003) provided finite-time results for both synchronous and asynchoronous Q-learning. Both Szepesvári (1998) and Even-Dar and Mansour (2003) showed that with linear learning rates, the convergence rate of Q-learning can be exponentially slow as a function of 1 1 -γ . To handle this, the so-called rescaled linear learning rate was introduced to avoid such an exponential dependence in synchronous Q-learning (Wainwright, 2019; Chen et al., 2020) and asynchronous Q-learning (Qu and Wierman, 2020). The finite-time convergence of Q-learning was also analyzed with constant step sizes (Beck and Srikant, 2012; Chen et al., 2020; Li et al., 2020). Moreover, the polynomial learning rate, which is also the focus of this work, was investigated for both synchronous (Even-Dar and Mansour, 2003; Wainwright, 2019) and asynchronous Q-learning (Even-Dar and Mansour, 2003). In addition, it is worth mentioning that Shah and Xie (2018) applied the nearest neighbor approach to handle MDPs on infinite state space.

Differently from the above extensive studies of vanilla Q-learning, theoretical understanding of double Q-learning is limited. The only theoretical guarantee was on the asymptotic convergence provided by Hasselt (2010); Weng et al. (2020c), which do not provide the non-asymptotic (i.e., finite-time) analysis on how fast double Q-learning converges. This paper provides the first finite-time analysis for double Q-learning.

The vanilla Q-learning algorithm has also been studied for the function approximation case, i.e., the Q-function is approximated by a class of parameterized functions. In contrast to the tabular case, even with linear function approximation, Q-learning has been shown not to converge in general (Baird, 1995). Strong assumptions are typically imposed to guarantee the convergence of Q-learning with function approximation (Bertsekas and Tsitsiklis, 1996; Zou et al., 2019; Chen et al., 2019; Du et al., 2019; Xu and Gu, 2019; Cai et al., 2019; Weng et al., 2020a,b). Regarding double Q-learning, it is still an open topic on how to design double Q-learning algorithms under function approximation and under what conditions they have theoretically guaranteed convergence.

## 2 Preliminaries on Q-learning and Double Q-learning

In this section, we introduce the Q-learning and the double Q-learning algorithms.

## 2.1 Q-learning

We consider a γ -discounted Markov decision process (MDP) with a finite state space S and a finite action space A . The transition probability of the MDP is given by P : S × A × S → [0 , 1] , that is, P ( ·| s, a ) denotes the probability distribution of the next state given the current state s and action a . We consider a random reward function R t at time t drawn from a fixed distribution φ : S × A × S ↦→ R , where E { R t ( s, a, s ′ ) } = R s ′ sa and s ′ denotes the next state starting from ( s, a ) . In addition, we assume | R t | ≤ R max . A policy π := π ( ·| s ) characterizes the conditional probability distribution over the action space A given each state s ∈ S .

The action-value function (i.e., Q-function) Q π ∈ R |S|×|A| for a given policy π is defined as

<!-- formula-not-decoded -->

where γ ∈ (0 , 1) is the discount factor. Q-learning aims to find the Q-function of an optimal policy π ∗ that maximizes the accumulated reward. The existence of such a π ∗ has been proved in the classical MDP theory (Bertsekas and Tsitsiklis, 1996). The corresponding optimal Q-function, denoted as Q ∗ ,

is known as the unique fixed point of the Bellman operator T given by

<!-- formula-not-decoded -->

where U ( s ′ ) ⊂ A is the admissible set of actions at state s ′ . It can be shown that the Bellman operator T is γ -contractive in the supremum norm ‖ Q ‖ := max s,a | Q ( s, a ) | , i.e., it satisfies

<!-- formula-not-decoded -->

The goal of Q-learning is to find Q ∗ , which further yields π ∗ ( s ) = arg max a ∈ U ( s ) Q ∗ ( s, a ) . In practice, however, exact evaluation of the Bellman operator (2) is usually infeasible due to the lack of knowledge of the transition kernel of MDP and the randomness of the reward. Instead, Q-learning draws random samples to estimate the Bellman operator and iteratively learns Q ∗ as

<!-- formula-not-decoded -->

where R t is the sampled reward, s ′ is sampled by the transition probability given ( s, a ) , and α t ( s, a ) ∈ (0 , 1] denotes the learning rate.

## 2.2 Double Q-learning

Although Q-learning is a commonly used RL algorithm to find the optimal policy, it can suffer from overestimation in practice (Smith and Winkler, 2006). To overcome this issue, Hasselt (2010) proposed double Q-learning given in Algorithm 1.

```
Algorithm 1 Synchronous Double Q-learning (Hasselt, 2010) 1: Input: Initial Q A 1 , Q B 1 . 2: for t = 1 , 2 , . . . , T do 3: Assign learning rate α t . 4: Randomly choose either UPDATE(A) or UPDATE(B) with probability 0.5, respectively. 5: for each ( s, a ) do 6: observe s ′ ∼ P ( ·| s, a ) , and sample R t ( s, a, s ′ ) . 7: if UPDATE(A) then 8: Obtain a ∗ = arg max a ′ Q A t ( s ′ , a ′ ) 9: Q A t +1 ( s, a ) = Q A t ( s, a ) + α t ( s, a )( R t ( s, a, s ′ ) + γQ B t ( s ′ , a ∗ ) -Q A t ( s, a )) 10: else if UPDATE(B) then 11: Obtain b ∗ = arg max b ′ Q B t ( s ′ , b ′ ) 12: Q B t +1 ( s, a ) = Q B t ( s, a ) + α t ( s, a )( R t ( s, a, s ′ ) + γQ A t ( s ′ , b ∗ ) -Q B t ( s, a )) 13: end if 14: end for 15: end for 16: Output: Q A T (or Q B T ).
```

Double Q-learning maintains two Q-estimators (i.e., Q-tables): Q A and Q B . At each iteration of Algorithm 1, one Q-table is randomly chosen to be updated. Then this chosen Q-table generates a greedy optimal action, and the other Q-table is used for estimating the corresponding Bellman operator for updating the chosen table. Specifically, if Q A is chosen to be updated, we use Q A to obtain the optimal action a ∗ and then estimate the corresponding Bellman operator using Q B . As shown in Hasselt (2010), E [ Q B ( s ′ , a ∗ )] is likely smaller than max a E [ Q A ( s ′ , a )] , where the expectation is taken over the randomness of the reward for the same state-action pair. In this way, such a two-estimator framework of double Q-learning can effectively reduce the overestimation.

Synchronous and asynchronous double Q-learning: In this paper, we study the finite-time convergence rate of double Q-learning in two different settings: synchronous and asynchronous implementations. For synchronous double Q-learning (as shown in Algorithm 1), all the state-action pairs of the chosen Q-estimator are visited simultaneously at each iteration. For the asynchronous case, only one state-action pair is updated in the chosen Q-table. Specifically, in the latter case, we sample a

trajectory { ( s t , a t , R t , i t ) } ∞ t =0 under a certain exploration strategy, where i t ∈ { A,B } denotes the index of the chosen Q-table at time t . Then the two Q-tables are updated based on the following rule:

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We next provide the boundedness property of the Q-estimators and the errors in the following lemma, which is typically necessary for the finite-time analysis.

Lemma 1. For either synchronous or asynchronous double Q-learning, let Q i t ( s, a ) be the value of either Q table corresponding to a state-action pair ( s, a ) at iteration t . Suppose ∥ ∥ Q i 0 ∥ ∥ ≤ R max 1 -γ . Then we have ∥ ∥ Q i t ∥ ∥ ≤ R max 1 -γ and ∥ ∥ Q i t -Q ∗ ∥ ∥ ≤ V max for all t ≥ 0 , where V max := 2 R max 1 -γ .

Lemma 1 can be proved by induction arguments using the triangle inequality and the uniform boundedness of the reward function, which is seen in Appendix A.

## 3 Main results

We present our finite-time analysis for the synchronous and asynchronous double Q-learning in this section, followed by a sketch of the proof for the synchronous case which captures our main techniques. The detailed proofs of all the results are provided in the Supplementary Materials.

## 3.1 Synchronous double Q-learning

Since the update of the two Q-estimators is symmetric, we can characterize the convergence rate of either Q-estimator, e.g., Q A , to the global optimum Q ∗ . To this end, we first derive two important properties of double Q-learning that are crucial to our finite-time convergence analysis.

The first property captures the stochastic error ∥ ∥ Q B t -Q A t ∥ ∥ between the two Q-estimators. Since double Q-learning updates alternatingly between these two estimators, such an error process must decay to zero in order for double Q-learning to converge. Furthermore, how fast such an error converges determines the overall convergence rate of double Q-learning. The following proposition (which is an informal restatement of Proposition 1 in Appendix B.1) shows that such an error process can be block-wisely bounded by an exponentially decreasing sequence G q = (1 -ξ ) q V max for q = 0 , 1 , 2 , . . . , and some ξ ∈ (0 , 1) . Conceptually, as illustrated in Figure 1, such an error process is upper-bounded by the blue-colored piece-wise linear curve.

Proposition 1. ( Informal ) Consider synchronous double Q-learning under a polynomial learning rate α t = 1 t ω with ω ∈ (0 , 1) . We divide the time horizon into blocks [ˆ τ q , ˆ τ q +1 ) for q ≥ 0 , where ˆ τ 0 = 0 and ˆ τ q +1 = ˆ τ q + c 1 ˆ τ ω q with some c 1 &gt; 0 . Fix ˆ glyph[epsilon1] &gt; 0 . Then for any n such that G n ≥ ˆ glyph[epsilon1] and under certain conditions on ˆ τ 1 (see Appendix B.1), we have

<!-- formula-not-decoded -->

where the positive constants c 2 and c 3 are specified in Appendix B.1.

Proposition 1 implies that the two Q-estimators approach each other asymptotically, but does not necessarily imply that they converge to the optimal action-value function Q ∗ . Then the next proposition (which is an informal restatement of Proposition 2 in Appendix B.2) shows that as long as the high probability event in Proposition 1 holds, the error process ∥ ∥ Q A t -Q ∗ ∥ ∥ between either Q-estimator (say Q A ) and the optimal Q-function can be block-wisely bounded by an exponentially decreasing sequence D k = (1 -β ) k V max σ for k = 0 , 1 , 2 , . . . , and β ∈ (0 , 1) . Conceptually, as illustrated in Figure 1, such an error process is upper-bounded by the yellow-colored piece-wise linear curve.

Proposition 2. ( Informal ) Consider synchronous double Q-learning using a polynomial learning rate α t = 1 t ω with ω ∈ (0 , 1) . We divide the time horizon into blocks [ τ k ,τ k +1 ) for k ≥ 0 , where

Do =

Vmax

DI

Go = Vmax

G1 = 7D1

TI (71)

т2 (72)

<!-- image -->

т3 (73)

Figure 1: Illustration of sequence { G k } k ≥ 0 as a block-wise upper bound on ∥ ∥ Q B t -Q A t ∥ ∥ , and sequence { D k } k ≥ 0 as a block-wise upper bound on ∥ ∥ Q A t -Q ∗ ∥ ∥ conditioned on the first upper bound event.

τ 0 = 0 and τ k +1 = τ k + c 4 τ ω k with some c 4 &gt; 0 . Fix ˜ glyph[epsilon1] &gt; 0 . Then for any m such that D m ≥ ˜ glyph[epsilon1] and under certain conditions on τ 1 (see Appendix B.2), we have

<!-- formula-not-decoded -->

where E and F denote certain events defined in (12) and (13) in Appendix B.2, and the positive constants c 4 , c 5 , and c 6 are specified Appendix B.2.

As illustrated in Figure 1, the two block sequences { ˆ τ q } q ≥ 0 in Proposition 1 and { τ q } q ≥ 0 in Proposition 2 can be chosen to coincide with each other. Then combining the above two properties followed by further mathematical arguments yields the following main theorem that characterizes the convergence rate of double Q-learning. We will provide a proof sketch for Theorem 1 in Section 3.3, which explains the main steps to obtain the supporting properties of Proposition 1 and 2 and how they further yield the main theorem.

Theorem 1. Fix glyph[epsilon1] &gt; 0 and γ ∈ (1 / 3 , 1) . Consider synchronous double Q-learning using a polynomial learning rate α t = 1 t ω with ω ∈ (0 , 1) . Let Q A T ( s, a ) be the value of Q A for a state-action pair ( s, a ) at time T . Then we have P ( ∥ ∥ Q A T -Q ∗ ∥ ∥ ≤ glyph[epsilon1] ) ≥ 1 -δ , given that

<!-- formula-not-decoded -->

where V max = 2 R max 1 -γ .

Theorem 1 provides the finite-time convergence guarantee in high probability sense for synchronous double Q-learning. Specifically, double Q-learning attains an glyph[epsilon1] -accurate optimal Q-function with high probability with at most Ω ( ( 1 (1 -γ ) 6 glyph[epsilon1] 2 ln 1 (1 -γ ) 7 glyph[epsilon1] 2 ) 1 ω + ( 1 1 -γ ln 1 (1 -γ ) 2 glyph[epsilon1] ) 1 1 -ω ) iterations. Such a result can be further understood by considering the following two regimes. In the high accuracy regime, in which glyph[epsilon1] glyph[lessmuch] 1 -γ , the dependence on glyph[epsilon1] dominates, and the time complexity is given by Ω ( ( 1 glyph[epsilon1] 2 ln 1 glyph[epsilon1] 2 ) 1 ω + ( ln 1 glyph[epsilon1] ) 1 1 -ω ) , which is optimized as ω approaches to 1. In the low accuracy regime, in which glyph[epsilon1] glyph[greatermuch] 1 -γ , the dependence on 1 1 -γ dominates, and the time complexity can be optimized at ω = 6 7 , which yields T = ˜ Ω ( 1 (1 -γ ) 7 glyph[epsilon1] 7 / 3 + 1 (1 -γ ) 7 ) = ˜ Ω ( 1 (1 -γ ) 7 glyph[epsilon1] 7 / 3 ) .

Furthermore, Theorem 1 corroborates the design effectiveness of double Q-learning, which overcomes the overestimation issue and hence achieves better accuracy by making less aggressive progress in each update. Specifically, comparison of Theorem 1 with the time complexity bounds of vanilla synchronous Q-learning under a polynomial learning rate in Even-Dar and Mansour (2003) and Wainwright (2019) indicates that in the high accuracy regime, double Q-learning achieves the same convergence rate as vanilla Q-learning in terms of the order-level dependence on glyph[epsilon1] . Clearly, the design of double Q-learning for high accuracy dominates the performance. In the low-accuracy regime

(which is not what double Q-learning is designed for), double Q-learning achieves a slightly weaker convergence rate than vanilla Q-learning in Even-Dar and Mansour (2003); Wainwright (2019) in terms of the dependence on 1 -γ , because its nature of less aggressive progress dominates the performance.

## 3.2 Asynchronous Double Q-learning

In this subsection, we study the asynchronous double Q-learning and provide its finite-time convergence result.

Differently from synchronous double Q-learning, in which all state-action pairs are visited for each update of the chosen Q-estimator, asynchronous double Q-learning visits only one state-action pair for each update of the chosen Q-estimator. Therefore, we make the following standard assumption on the exploration strategy (Even-Dar and Mansour, 2003):

Assumption 1. (Covering number) There exists a covering number L , such that in consecutive L updates of either Q A or Q B estimator, all the state-action pairs of the chosen Q-estimator are visited at least once.

The above conditions on the exploration are usually necessary for the finite-time analysis of asynchronous Q-learning. The same assumption has been taken in Even-Dar and Mansour (2003). Qu and Wierman (2020) proposed a mixing time condition which is in the same spirit.

Assumption 1 essentially requires the sampling strategy to have good visitation coverage over all state-action pairs. Specifically, Assumption 1 guarantees that consecutive L updates of Q A visit each state-action pair of Q A at least once, and the same holds for Q B . Since 2 L iterations of asynchronous double Q-learning must make at least L updates for either Q A or Q B , Assumption 1 further implies that any state-action pair ( s, a ) must be visited at least once during 2 L iterations of the algorithm. In fact, our analysis allows certain relaxation of Assumption 1 by only requiring each state-action pair to be visited during an interval with a certain probability. In such a case, we can also derive a finite-time bound by additionally dealing with a conditional probability.

Next, we provide the finite-time result for asynchronous double Q-learning in the following theorem.

Theorem 2. Fix glyph[epsilon1] &gt; 0 , γ ∈ (1 / 3 , 1) . Consider asynchronous double Q-learning under a polynomial learning rate α t = 1 t ω with ω ∈ (0 , 1) . Suppose Assumption 1 holds. Let Q A T ( s, a ) be the value of Q A for a state-action pair ( s, a ) at time T . Then we have P ( ∥ ∥ Q A T -Q ∗ ∥ ∥ ≤ glyph[epsilon1] ) ≥ 1 -δ , given that

<!-- formula-not-decoded -->

Comparison of Theorem 1 and 2 indicates that the finite-time result of asynchronous double Qlearning matches that of synchronous double Q-learning in the order dependence on 1 1 -γ and 1 glyph[epsilon1] . The difference lies in the extra dependence on the covering time L in Theorem 2. Since synchronous double Q-learning visits all state-action pairs (i.e., takes |S||A| sample updates) at each iteration, whereas asynchronous double Q-learning visits only one state-action pair (i.e., takes only one sample update) at each iteration, a more reasonable comparison between the two should be in terms of the overall sample complexity. In this sense, synchronous and asynchronous double Q-learning algorithms have the sample complexities of |S||A| T (where T is given in (5)) and T (where T is given in (6)), respectively. Since in general L glyph[greatermuch] |S||A| , synchronous double-Q is more efficient than asynchronous double-Q in terms of the overall sampling complexity.

## 3.3 Proof Sketch of Theorem 1

In this subsection, we provide an outline of the technical proof of Theorem 1 and summarize the key ideas behind the proof. The detailed proof can be found in Appendix B.

Our goal is to study the finite-time convergence of the error ∥ ∥ Q A t -Q ∗ ∥ ∥ between one Q-estimator and the optimal Q-function (this is without the loss of generality due to the symmetry of the two estimators). To this end, our proof includes: (a) Part I which analyzes the stochastic error propagation between the two Q-estimators ∥ ∥ Q B t -Q A t ∥ ∥ ; (b) Part II which analyzes the error dynamics between one Q-estimator and the optimum ∥ ∥ Q A t -Q ∗ ∥ ∥ conditioned on the error event in Part I; and (c) Part

III which bounds the unconditional error ∥ ∥ Q A t -Q ∗ ∥ ∥ . We describe each of the three parts in more details below.

Part I: Bounding ∥ ∥ Q B t -Q A t ∥ ∥ (see Proposition 1). The main idea is to upper bound ∥ ∥ Q B t -Q A t ∥ ∥ by a decreasing sequence { G q } q ≥ 0 block-wisely with high probability, where each block q (with q ≥ 0 ) is defined by t ∈ [ˆ τ q , ˆ τ q +1 ) . The proof consists of the following four steps.

Step 1 (see Lemma 2) : We characterize the dynamics of u BA t ( s, a ) := Q B ( s, a ) -Q A ( s, a ) as an SA algorithm as follows:

<!-- formula-not-decoded -->

where h t is a contractive mapping of u BA t , and z t is a martingale difference sequence.

Step 2 (see Lemma 3) : We derive lower and upper bounds on u BA t via two sequences X t ;ˆ τ q and Z t ;ˆ τ q as follows:

<!-- formula-not-decoded -->

for any t ≥ ˆ τ q , state-action pair ( s, a ) ∈ S × A , and q ≥ 0 , where X t ;ˆ τ q is deterministic and driven by G q , and Z t ;ˆ τ q is stochastic and driven by the martingale difference sequence z t .

Step 3 (see Lemma 5 and Lemma 6) : Weblock-wisely bound u BA t ( s, a ) using the induction arguments. Namely, we prove ∥ ∥ u BA t ∥ ∥ ≤ G q for t ∈ [ ˆ τ q , ˆ τ q +1 ) holds for all q ≥ 0 . By induction, we first observe for q = 0 , ∥ ∥ u BA t ∥ ∥ ≤ G 0 holds. Given any state-action pair ( s, a ) , we assume that ∥ ∥ u BA t ( s, a ) ∥ ∥ ≤ G q holds for t ∈ [ˆ τ q , ˆ τ q +1 ) . Then we show ∥ ∥ u BA t ( s, a ) ∥ ∥ ≤ G q +1 holds for t ∈ [ˆ τ q +1 , ˆ τ q +2 ) , which follows by bounding X t ;ˆ τ q and Z t ;ˆ τ q separately in Lemma 5 and Lemma 6, respectively.

Step 4 (see Appendix B.1.4) : We apply union bound (Lemma 8) to obtain the block-wise bound for all state-action pairs and all blocks.

Part II: Conditionally bounding ∥ ∥ Q A t -Q ∗ ∥ ∥ (see Proposition 2) . We upper bound ∥ ∥ Q A t -Q ∗ ∥ ∥ by a decreasing sequence { D k } k ≥ 0 block-wisely conditioned on the following two events:

Event E : ∥ ∥ u BA t ∥ ∥ is upper bounded properly (see (12) in Appendix B.2), and

Event F : there are sufficient updates of Q A t in each block (see (13) in Appendix B.2).

The proof of Proposition 2 consists of the following four steps.

Step 1 (see Lemma 10) : We design a special relationship (illustrated in Figure 1) between the block-wise bounds { G q } q ≥ 0 and { D k } k ≥ 0 and their block separations.

Step 2 (see Lemma 11) : We characterize the dynamics of the iteration residual r t ( s, a ) := Q A t ( s, a ) -Q ∗ ( s, a ) as an SA algorithm as follows: when Q A is chosen to be updated at iteration t ,

<!-- formula-not-decoded -->

where w t ( s, a ) is the error between the Bellman operator and the sample-based empirical estimator, and is thus a martingale difference sequence, and u BA t has been defined in Part I.

Step 3 (see Lemma 12) : We provide upper and lower bounds on r t via two sequences Y t ; τ k and W t ; τ k as follows:

<!-- formula-not-decoded -->

for all t ≥ τ k , all state-action pairs ( s, a ) ∈ S × A , and all q ≥ 0 , where Y t ; τ k is deterministic and driven by D k , and W t ; τ k is stochastic and driven by the martingale difference sequence w t . In particular, if Q A t is not updated at some iteration, then the sequences Y t ; τ k and W t ; τ k assume the same values from the previous iteration.

Step 4 (see Lemma 13, Lemma 14 and Appendix B.2.4) : Similarly to Steps 3 and 4 in Part I, we conditionally bound ‖ r t ‖ ≤ D k for t ∈ [ τ k , τ k +1 ) and k ≥ 0 via bounding Y t ; τ k and W t ; τ k and further taking the union bound.

Part III: Bounding ∥ ∥ Q A t -Q ∗ ∥ ∥ (see Appendix B.3). We combine the results in the first two parts, and provide high probability bound on ‖ r t ‖ with further probabilistic arguments, which exploit the high probability bounds on P ( E ) in Proposition 1 and P ( F ) in Lemma 15.

## 4 Conclusion

In this paper, we provide the first finite-time results for double Q-learning, which characterize how fast double Q-learning converges under both synchronous and asynchronous implementations. For the synchronous case, we show that it achieves an glyph[epsilon1] -accurate optimal Q-function with at least the

<!-- formula-not-decoded -->

Similar scaling order on 1 1 -γ and 1 glyph[epsilon1] also applies for asynchronous double Q-learning but with extra dependence on the covering number. We develop new techniques to bound the error between two correlated stochastic processes, which can be of independent interest.

## Acknowledgements

The work was supported in part by the U.S. National Science Foundation under the grant CCF1761506 and the startup fund of the Southern University of Science and Technology (SUSTech), China.

## Broader Impact

Reinforcement learning has achieved great success in areas such as robotics and game playing, and thus has aroused broad interests and more potential real-world applications. Double Q-learning is a commonly used technique in deep reinforcement learning to improve the implementation stability and speed of deep Q-learning. In this paper, we provided the fundamental analysis on the convergence rate for double Q-learning, which theoretically justified the empirical success of double Q-learning in practice. Such a theory also provides practitioners desirable performance guarantee to further develop such a technique into various transferable technologies.

## References

- Abed-alguni, B. H. and Ottom, M. A. (2018). Double delayed Q-learning. International Journal of Artificial Intelligence , 16(2):41-59.
- Azuma, K. (1967). Weighted sums of certain dependent random variables. Tohoku Mathematical Journal, Second Series , 19(3):357-367.
- Baird, L. (1995). Residual algorithms: Reinforcement learning with function approximation. In Machine Learning Proceedings 1995 , pages 30-37. Elsevier.
- Beck, C. L. and Srikant, R. (2012). Error bounds for constant step-size Q-learning. Systems &amp; Control Letters , 61(12):1203-1208.
- Bertsekas, D. P. and Tsitsiklis, J. N. (1996). Neuro-Dynamic Programming , volume 5. Athena Scientific.
- Borkar, V. S. and Meyn, S. P. (2000). The ODE method for convergence of stochastic approximation and reinforcement learning. SIAM Journal on Control and Optimization , 38(2):447-469.
- Cai, Q., Yang, Z., Lee, J. D., and Wang, Z. (2019). Neural temporal-difference learning converges to global optima. In Advances in Neural Information Processing Systems (NeurIPS) , pages 11312-11322.
- Chen, Z., Maguluri, S. T., Shakkottai, S., and Shanmugam, K. (2020). Finite-sample analysis of stochastic approximation using smooth convex envelopes. arXiv preprint arXiv:2002.00874 .
- Chen, Z., Zhang, S., Doan, T. T., Maguluri, S. T., and Clarke, J.-P. (2019). Finite-time analysis of Q-learning with linear function approximation. arXiv preprint arXiv:1905.11425 .
- Du, S. S., Luo, Y., Wang, R., and Zhang, H. (2019). Provably efficient Q-learning with function approximation via distribution shift error checking oracle. In Advances in Neural Information Processing Systems (NeurIPS) , pages 8058-8068.
- Even-Dar, E. and Mansour, Y. (2003). Learning rates for Q-learning. Journal of Machine Learning Research , 5(Dec):1-25.

- Hasselt, H. V. (2010). Double Q-learning. In Advances in Neural Information Processing Systems (NeurIPS) , pages 2613-2621.
- Hasselt, H. v., Guez, A., and Silver, D. (2016). Deep reinforcement learning with double q-learning. In Proc. AAAI Conference on Artificial Intelligence (AAAI) .
- Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., and Silver, D. (2018). Rainbow: Combining improvements in deep reinforcement learning. In Proc. AAAI Conference on Artificial Intelligence (AAAI) .
- Jaakkola, T., Jordan, M. I., and Singh, S. P. (1994). Convergence of stochastic iterative dynamic programming algorithms. In Advances in Neural Information Processing Systems (NeurIPS) , pages 703-710.
- Lee, D. and He, N. (2019). A unified switching system perspective and ODE analysis of Q-learning algorithms. arXiv preprint arXiv:1912.02270 .
- Li, G., Wei, Y., Chi, Y ., Gu, Y ., and Chen, Y . (2020). Sample complexity of asynchronous Q-learning: Sharper analysis and variance reduction. arXiv preprint arXiv:2006.03041 .
- Melo, F. S. (2001). Convergence of Q-learning: A simple proof. Institute of Systems and Robotics, Tech. Rep , pages 1-4.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. (2015). Human-level control through deep reinforcement learning. Nature , 518(7540):529.
- Okuyama, T., Gonsalves, T., and Upadhay, J. (2018). Autonomous driving system based on deep Q-learnig. In Proc. IEEE International Conference on Intelligent Autonomous Systems (ICoIAS) , pages 201-205.
- Qu, G. and Wierman, A. (2020). Finite-time analysis of asynchronous stochastic approximation and Q-learning. arXiv preprint arXiv:2002.00260 .
- Shah, D. and Xie, Q. (2018). Q-learning with nearest neighbors. In Advances in Neural Information Processing Systems (NeurIPS) , pages 3111-3121.
- Smith, J. E. and Winkler, R. L. (2006). The optimizer's curse: Skepticism and postdecision surprise in decision analysis. Management Science , 52(3):311-322.
- Szepesvári, C. (1998). The asymptotic convergence-rate of Q-learning. In Advances in Neural Information Processing Systems (NeurIPS) , pages 1064-1070.
- Tai, L. and Liu, M. (2016). A robot exploration strategy based on Q-learning network. In Proc. IEEE International Conference on Real-time Computing and Robotics (RCAR) , pages 57-62.
- Tsitsiklis, J. N. (1994). Asynchronous stochastic approximation and Q-learning. Machine Learning , 16(3):185-202.
- Wainwright, M. J. (2019). Stochastic approximation with cone-contractive operators: Sharp glyph[lscript] ∞ -bounds for Q-learning. arXiv preprint arXiv:1905.06265 .
- Watkins, C. J. and Dayan, P. (1992). Q-learning. Machine Learning , 8(3-4):279-292.
- Weng, B., Xiong, H., Liang, Y., and Zhang, W. (2020a). Analysis of Q-learning with adaptation and momentum restart for gradient descent. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20) , pages 3051-3057.
- Weng, B., Xiong, H., Zhao, L., Liang, Y., and Zhang, W. (2020b). Momentum Q-learning with finite-sample convergence guarantee. arXiv preprint arXiv:2007.15418 .
- Weng, W., Gupta, H., He, N., Ying, L., and R., S. (2020c). Provably-efficient double Q-learning. arXiv preprint arXiv:arXiv:2007.05034 .
- Xu, P. and Gu, Q. (2019). A finite-time analysis of Q-learning with neural network function approximation. arXiv preprint arXiv:1912.04511 .
- Zhang, Q., Lin, M., Yang, L. T., Chen, Z., Khan, S. U., and Li, P. (2018a). A double deep Qlearning model for energy-efficient edge scheduling. IEEE Transactions on Services Computing , 12(5):739-749.
- Zhang, Y., Sun, P., Yin, Y., Lin, L., and Wang, X. (2018b). Human-like autonomous vehicle speed control by deep reinforcement learning with double Q-learning. In Proc. IEEE Intelligent Vehicles Symposium (IV) , pages 1251-1256.

- Zhang, Z., Pan, Z., and Kochenderfer, M. J. (2017). Weighted double Q-learning. In International Joint Conferences on Artificial Intelligence , pages 3455-3461.
- Zou, S., Xu, T., and Liang, Y. (2019). Finite-sample analysis for SARSA with linear function approximation. In Advances in Neural Information Processing Systems (NeurIPS) , pages 8665-8675.

## Supplementary Materials

## A Proof of Lemma 1

We prove Lemma 1 by induction.

First, it is easy to guarantee that the initial case is satisfied, i.e., ∥ ∥ Q A 1 ∥ ∥ ≤ R max 1 -γ = V max 2 , ∥ ∥ Q B 1 ∥ ∥ ≤ V max 2 . (In practice we usually initialize the algorithm as Q A 1 = Q B 1 = 0 ). Next, we assume that ∥ ∥ Q A t ∥ ∥ ≤ V max 2 , ∥ ∥ Q B t ∥ ∥ ≤ V max 2 . It remains to show that such conditions still hold for t +1 . Observe that

<!-- formula-not-decoded -->

Similarly, we can have ∥ ∥ Q B t +1 ( s, a ) ∥ ∥ ≤ V max 2 . Thus we complete the proof.

## B Proof of Theorem 1

In this appendix, we will provide a detailed proof of Theorem 1. Our proof includes: (a) Part I which analyzes the stochastic error propagation between the two Q-estimators ∥ ∥ Q B t -Q A t ∥ ∥ ; (b) Part II which analyzes the error dynamics between one Q-estimator and the optimum ∥ ∥ Q A t -Q ∗ ∥ ∥ conditioned on the error event in Part I; and (c) Part III which bounds the unconditional error ∥ ∥ Q A t -Q ∗ ∥ ∥ . We describe each of the three parts in more details below.

## B.1 Part I: Bounding ∥ ∥ Q B t -Q A t ∥ ∥

The main idea is to upper bound ∥ ∥ Q B t -Q A t ∥ ∥ by a decreasing sequence { G q } q ≥ 0 block-wisely with high probability, where each block or epoch q (with q ≥ 0 ) is defined by t ∈ [ˆ τ q , ˆ τ q +1 ) .

Proposition 1. Fix glyph[epsilon1] &gt; 0 , κ ∈ (0 , 1) , σ ∈ (0 , 1) and ∆ ∈ (0 , e -2) . Consider synchronous double Q-learning using a polynomial learning rate α t = 1 t ω with ω ∈ (0 , 1) . Let G q = (1 -ξ ) q G 0 with G 0 = V max and ξ = 1 -γ 4 . Let ˆ τ q +1 = ˆ τ q + 2 c κ ˆ τ ω q for q ≥ 1 with c ≥ ln(2+∆)+1 / ˆ τ ω 1 1 -ln(2+∆) -1 / ˆ τ ω 1 and ˆ τ 1 as the finishing time of the first epoch satisfying

<!-- formula-not-decoded -->

Then for any n such that G n ≥ σglyph[epsilon1] , we have

<!-- formula-not-decoded -->

The proof of Proposition 1 consists of the following four steps.

## B.1.1 Step 1: Characterizing the dynamics of Q B t ( s, a ) -Q A t ( s, a )

We first characterize the dynamics of u BA t ( s, a ) := Q B t ( s, a ) -Q A t ( s, a ) as a stochastic approximation (SA) algorithm in this step.

Lemma 2. Consider double Q-learning in Algorithm 1. Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

In addition, F t satisfies

<!-- formula-not-decoded -->

Proof. Algorithm 1 indicates that at each time, either Q A or Q B is updated with equal probability. When updating Q A at time t , for each ( s, a ) we have

<!-- formula-not-decoded -->

Similarly, when updating Q B , we have

<!-- formula-not-decoded -->

Therefore, we can rewrite the dynamics of u BA t as u BA t +1 ( s, a ) = (1 -α t ) u BA t ( s, a ) + α t F t ( s, a ) , where

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Next, we bound E s t +1 [ Q A t ( s t +1 , b ∗ ) -Q B t ( s t +1 , a ∗ ) ] . First, consider the case when E s t +1 Q A t ( s t +1 , b ∗ ) ≥ E s t +1 Q B t ( s t +1 , a ∗ ) . Then we have

<!-- formula-not-decoded -->

where (i) follow from the definition of a ∗ in Algorithm 1. Similarly, if E s t +1 Q A t ( s t +1 , b ∗ ) &lt; E s t +1 Q B t ( s t +1 , a ∗ ) , we have

<!-- formula-not-decoded -->

where (i) follows from the definition of b ∗ . Thus we can conclude that

<!-- formula-not-decoded -->

Then, we continue to bound (7), and obtain

<!-- formula-not-decoded -->

for all ( s, a ) pairs. Hence, ‖ E [ F t |F t ] ‖ ≤ 1+ γ 2 ∥ ∥ u BA t ∥ ∥ .

Applying Lemma 2, we write the dynamics of u BA t ( s, a ) in the form of a classical SA algorithm driven by a martingale difference sequence as follows:

<!-- formula-not-decoded -->

where h t ( s, a ) = E [ F t ( s, a ) |F t ] and z t ( s, a ) = F t ( s, a ) -E [ F t |F t ] . Then, we obtain E [ z t ( s, a ) |F t ] = 0 and ‖ h t ‖ ≤ 1+ γ 2 ∥ ∥ u BA t ∥ ∥ following from Lemma 2. We define u ∗ ( s, a ) = 0 , and treat h t as an operator over u BA t . Then h t has a contraction property as:

<!-- formula-not-decoded -->

where γ ′ = 1+ γ 2 ∈ (0 , 1) . Based on this SA formulation, we bound u BA t ( s, a ) block-wisely in the next step.

## B.1.2 Step 2: Constructing sandwich bounds on u BA t

We derive lower and upper bounds on u BA t via two sequences X t ;ˆ τ q and Z t ;ˆ τ q in the following lemma.

Lemma 3. Let ˆ τ q be such that ∥ ∥ u BA t ∥ ∥ ≤ G q for all t ≥ ˆ τ q . Define Z t ;ˆ τ q ( s, a ) , X t ;ˆ τ q ( s, a ) as

<!-- formula-not-decoded -->

Then for any t ≥ ˆ τ q and state-action pair ( s, a ) , we have

<!-- formula-not-decoded -->

Proof. We proceed the proof by induction. For the initial condition t = ˆ τ q , ∥ ∥ ∥ u BA ˆ τ q ∥ ∥ ∥ ≤ G q implies -G q ≤ u BA ˆ τ q ≤ G q . We assume the sandwich bound holds for time t . It remains to check that the bound also holds for t +1 .

<!-- formula-not-decoded -->

At time t +1 , we have

<!-- formula-not-decoded -->

where (i) follows from Lemma 2. Similarly, we can bound the other direction as

<!-- formula-not-decoded -->

## B.1.3 Step 3: Bounding X t ;ˆ τ q and Z t ;ˆ τ q for block q +1

We bound X t ;ˆ τ q and Z t ;ˆ τ q in Lemma 5 and Lemma 6 below, respectively. Before that, we first introduce the following technical lemma which will be useful in the proof of Lemma 5.

Lemma 4. Fix ω ∈ (0 , 1) . Let 0 &lt; t 1 &lt; t 2 . Then we have

<!-- formula-not-decoded -->

Proof. Since ln(1 -x ) ≤ -x for any x ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

Thus, fix ω ∈ (0 , 1) , let 0 &lt; t 1 &lt; t 2 , and then we have

<!-- formula-not-decoded -->

Define f ( t ) := t 1 -ω . Observe that f ( t ) is an increasing concave function. Then we have

<!-- formula-not-decoded -->

which immediately indicates the result.

We now derive a bound for X t ;ˆ τ q .

Lemma 5. Fix κ ∈ (0 , 1) and ∆ ∈ (0 , e -2) . Let { G q } be defined in Proposition 1. Consider synchronous double Q-learning using a polynomial learning rate α t = 1 t ω with ω ∈ (0 , 1) . Suppose that X t ;ˆ τ q ( s, a ) ≤ G q for any t ≥ ˆ τ q . Then for any t ∈ [ˆ τ q +1 , ˆ τ q +2 ) , given ˆ τ q +1 = ˆ τ q + 2 c κ ˆ τ ω q with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Observe that X ˆ τ q ;ˆ τ q ( s, a ) = G q = γ ′ G q +(1 -γ ′ ) G q := γ ′ G q + ρ ˆ τ q . We can rewrite the dynamics of X t ;ˆ τ q ( s, a ) as

<!-- formula-not-decoded -->

where ρ t +1 = (1 -α t ) ρ t . By the definition of ρ t , we obtain

<!-- formula-not-decoded -->

where (i) follows because α i is decreasing and t ≥ ˆ τ q +1 , (ii) follows from Lemma 4, (iii) follows because ˆ τ q ≥ ˆ τ 1 and

<!-- formula-not-decoded -->

and (iv) follows because 2 c κ ≥ c . Next, observing the conditions that ˆ τ ω 1 ≥ 1 1 -ln(2+∆) and c ≥ 1 1 -ln(2+∆) -1 / ˆ τ ω 1 -1 , we have

<!-- formula-not-decoded -->

Thus we have ρ t ≤ 1 -γ ′ 2+∆ G q . Finally, We finish our proof by further observing that 1 -γ ′ = 2 ξ .

Since we have bounded X t ;ˆ τ q ( s, a ) by ( γ ′ + 2 2+∆ ξ ) G q for all t ≥ ˆ τ q +1 , it remains to bound Z t ;ˆ τ q ( s, a ) by ( 1 -2 2+∆ ) ξG q for block q +1 , which will further yield ∥ ∥ u BA t ( s, a ) ∥ ∥ ≤ ( γ ′ + ξ ) G q = (1 -ξ ) G q = G q +1 for any t ∈ [ˆ τ q +1 , ˆ τ q +2 ) as desired. Differently from X t ;ˆ τ q ( s, a ) which is a deterministic monotonic sequence, Z t ;ˆ τ q ( s, a ) is stochastic. We need to capture the probability for a bound on Z t ;ˆ τ q ( s, a ) to hold for block q + 1 . To this end, we introduce a different sequence { Z l t ;ˆ τ q ( s, a ) } given by

<!-- formula-not-decoded -->

where φ q,t -1 i = α i ∏ t -1 j = i +1 (1 -α j ) . By the definition of Z t ;ˆ τ q ( s, a ) , one can check that Z t ;ˆ τ q ( s, a ) = Z t -1 -ˆ τ q t ;ˆ τ q ( s, a ) . Thus we have

<!-- formula-not-decoded -->

In the following lemma, we capture an important property of Z l t ;ˆ τ q ( s, a ) defined in (9).

Lemma 6. For any t ∈ [ˆ τ q +1 , ˆ τ q +2 ) and 1 ≤ l ≤ t -1 -ˆ τ q , Z l t ;ˆ τ q ( s, a ) is a martingale sequence and satisfies

<!-- formula-not-decoded -->

Proof. To show the martingale property, we observe that

<!-- formula-not-decoded -->

where the last equation follows from the definition of z t ( s, a ) .

In addition, based on the definition of φ q,t -1 i in (9) which requires i ≥ ˆ τ q , we have

<!-- formula-not-decoded -->

Further, since | F t | ≤ 2 R max 1 -γ = V max , we obtain | z t ( s, a ) | = | F t -E [ F t |F t ] | ≤ 2 V max . Thus

<!-- formula-not-decoded -->

Lemma 6 guarantees that Z l t ;ˆ τ q ( s, a ) is a martingale sequence, which allows us to apply the following Azuma's inequality.

Lemma 7. (Azuma, 1967) Let X 0 , X 1 , . . . , X n be a martingale sequence such that for each 1 ≤ k ≤ n ,

<!-- formula-not-decoded -->

where the c k is a constant that may depend on k . Then for all n ≥ 1 and any glyph[epsilon1] &gt; 0 ,

<!-- formula-not-decoded -->

By Azuma's inequality and the relationship between Z t ;ˆ τ q ( s, a ) and Z l t ;ˆ τ q ( s, a ) in (9), we obtain

<!-- formula-not-decoded -->

where (i) follows from Lemma 6, and (ii) follows because

<!-- formula-not-decoded -->

## B.1.4 Step 4: Unionizing all blocks and state-action pairs

Now we are ready to prove Proposition 1 by taking a union of probabilities over all blocks and state-action pairs. Before that, we introduce the following two preliminary lemmas, which will be used for multiple times in the sequel.

Lemma 8. Let { X i } i ∈I be a set of random variables. Fix glyph[epsilon1] &gt; 0 . If for any i ∈ I , we have P ( X i ≤ glyph[epsilon1] ) ≥ 1 -δ , then

<!-- formula-not-decoded -->

Proof. By union bound, we have

<!-- formula-not-decoded -->

Lemma 9. Fix positive constants a, b satisfying 2 ab ln ab &gt; 1 . If τ ≥ 2 ab ln ab , then

<!-- formula-not-decoded -->

Proof. Let c = ab . If τ ≤ c 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If τ ≥ c 2 , we have where the last inequality follows from ln x 2 = 2ln x ≤ x . Therefore, we obtain c ln τ = ab ln τ ≤ τ . Thus τ b ≤ exp ( τ a ) , which implies this lemma.

## Proof of Proposition 1

Based on the results obtained above, we are ready to prove Proposition 1. Applying Lemma 8, we have

<!-- formula-not-decoded -->

where (i) follows because G q ≥ G n ≥ σglyph[epsilon1] , (ii) follows from Lemma 9 by substituting that a = 64 c ( c + κ ) V 2 max κ 2 ( ∆ 2+∆ ) 2 σ 2 ξ 2 glyph[epsilon1] 2 , b = 1 and observing

<!-- formula-not-decoded -->

and (iii) follows because ˆ τ q ≥ ˆ τ 1 .

Finally, we complete the proof of Proposition 1 by observing that X t ;ˆ τ q is a deterministic sequence and thus

<!-- formula-not-decoded -->

## B.2 Part II: Conditionally bounding ∥ ∥ Q A t -Q ∗ ∥ ∥

In this part, we upper bound ∥ ∥ Q A t -Q ∗ ∥ ∥ by a decreasing sequence { D k } k ≥ 0 block-wisely conditioned on the following two events: fix a positive integer m , we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where I A k denotes the number of iterations updating Q A at epoch k , τ k +1 is the starting iteration index of the ( k +1) th block, and ω is the decay parameter of the polynomial learning rate. Roughly, Event E requires that the difference between the two Q-estimators are bounded appropriately, and Event F requires that Q A is sufficiently updated in each block.

Proposition 2. Fix glyph[epsilon1] &gt; 0 , κ ∈ (ln 2 , 1) and ∆ ∈ (0 , e κ -2) . Consider synchronous double Qlearning under a polynomial learning rate α t = 1 t ω with ω ∈ (0 , 1) . Let { G q } q ≥ 0 , { ˆ τ q } q ≥ 0 be defined in Proposition 1. Define D k = (1 -β ) k V max σ with β = 1 -γ (1+ σ ) 2 and σ = 1 -γ 2 γ . Let τ k = ˆ τ k for k ≥ 0 . Suppose that c ≥ κ (ln(2+∆)+1 /τ ω 1 ) 2( κ -ln(2+∆) -1 /τ ω 1 ) and τ 1 as the finishing time of the first block satisfies

<!-- formula-not-decoded -->

Then for any m such that D m ≥ glyph[epsilon1] , we have

<!-- formula-not-decoded -->

where the events E,F are defined in (12) and (13) , respectively.

The proof of Proposition 2 consists of the following four steps.

## B.2.1 Step 1: Designing { D k } k ≥ 0

The following lemma establishes the relationship (illustrated in Figure 1) between the block-wise bounds { G q } q ≥ 0 and { D k } k ≥ 0 and their block separations, such that Event E occurs with high probability as a result of Proposition 1.

Lemma 10. Let { G q } be defined in Proposition 1, and let D k = (1 -β ) k V max σ with β = 1 -γ (1+ σ ) 2 and σ = 1 -γ 2 γ . Then we have

<!-- formula-not-decoded -->

given that τ k = ˆ τ k

<!-- formula-not-decoded -->

Proof. Based on our choice of σ , we have

<!-- formula-not-decoded -->

Therefore, the decay rate of D k is the same as that of G q . Further considering G 0 = σD 0 , we can make the sequence { σD k } as an upper bound of { G q } for any time as long as we set the same starting point and ending point for each epoch.

In Lemma 10, we make G k = σD k at any block k and ξ = β = 1 -γ 4 by careful design of σ . In fact, one can choose any value of σ ∈ (0 , (1 -γ ) /γ ) and design a corresponding relationship between τ k and ˆ τ k as long as the sequence { σD k } can upper bound { G q } for any time. For simplicity of presentation, we keep the design in Lemma 10.

## B.2.2 Step 2: Characterizing the dynamics of Q A t ( s, a ) -Q ∗ ( s, a )

We characterize the dynamics of the iteration residual r t ( s, a ) := Q A t ( s, a ) -Q ∗ ( s, a ) as an SA algorithm in Lemma 11 below. Since not all iterations contribute to the error propagation due to the random update between the two Q-estimators, we introduce the following notations to label the valid iterations.

Definition 1. We define T A as the collection of iterations updating Q A . In addition, we denote T A ( t 1 , t 2 ) as the set of iterations updating Q A between time t 1 and t 2 . That is,

<!-- formula-not-decoded -->

Correspondingly, the number of iterations updating Q A between time t 1 and t 2 is the cardinality of T A ( t 1 , t 2 ) which is denoted as | T A ( t 1 , t 2 ) | .

Lemma 11. Consider double Q-learning in Algorithm 1. Then we have

<!-- formula-not-decoded -->

Proof. Following from Algorithm 1 and for t ∈ T A , we have

<!-- formula-not-decoded -->

where (i) follows because we denote T t Q A t ( s, a ) = R t + γQ A t ( s ′ , a ∗ ) . By subtracting Q ∗ from both sides, we complete the proof.

## B.2.3 Step 3: Constructing sandwich bounds on r t ( s, a )

We provide upper and lower bounds on r t by constructing two sequences Y t ; τ k and W t ; τ k in the following lemma.

Lemma 12. Let τ k be such that ‖ r t ‖ ≤ D k for all t ≥ τ k . Suppose that we have ∥ ∥ u BA t ∥ ∥ ≤ σD k with σ = 1 -γ 2 γ for all t ≥ τ k . Define W t ; τ k ( s, a ) as

<!-- formula-not-decoded -->

where W τ k ; τ k ( s, a ) = 0 and define Y t ; τ k ( s, a ) as

<!-- formula-not-decoded -->

where Y τ k ; τ k ( s, a ) = D k and γ ′′ = γ (1 + σ ) . Then for any t ≥ τ k and state-action pair ( s, a ) , we have

<!-- formula-not-decoded -->

Proof. We proceed the proof by induction. For the initial condition t = τ k , we have ‖ r t ( s, a ) ‖ ≤ D k , and thus it holds that -D k ≤ r τ k ( s, a ) ≤ D k . We assume the sandwich bound holds for time t ≥ τ k . It remains to check whether this bound holds for t +1 .

If t / ∈ T A , then r t +1 ( s, a ) = r t ( s, a ) , W t +1; τ k ( s, a ) = W t ; τ k ( s, a ) , Y t +1; τ k ( s, a ) = Y t ; τ k ( s, a ) . Thus the sandwich bound still holds.

If t ∈ T A , we have

<!-- formula-not-decoded -->

where (i) follows from the contraction property of the Bellman operator, and (ii) follows from the condition ∥ ∥ u BA t ∥ ∥ ≤ σD k .

Similarly, we can bound the other direction as

<!-- formula-not-decoded -->

## B.2.4 Step 4: Bounding Y t ; τ k ( s, a ) and W t ; τ k ( s, a ) for epoch k +1

Similarly to Steps 3 and 4 in Part I, we conditionally bound ‖ r t ‖ ≤ D k for t ∈ [ τ k , τ k +1 ) and k = 0 , 1 , 2 , . . . by the induction arguments followed by the union bound. We first bound Y t ; τ k ( s, a ) and W t ; τ k ( s, a ) in Lemma 13 and Lemma 14, respectively.

Lemma 13. Fix κ ∈ (ln 2 , 1) and ∆ ∈ (0 , e κ -2) . Let { D k } be defined in Lemma 10. Consider synchronous double Q-learning using a polynomial learning rate α t = 1 t ω with ω ∈ (0 , 1) . Suppose that Y t ; τ k ( s, a ) ≤ D k for any t ≥ τ k . At block k , we assume that there are at least cτ ω k iterations updating Q A , i.e., | T A ( τ k , τ k +1 ) | ≥ cτ ω k . Then for any t ∈ [ τ k +1 , τ k +2 ) , we have

<!-- formula-not-decoded -->

Proof. Since we have defined τ k = ˆ τ k in Lemma 10, we have τ k +1 = τ k + 2 c κ τ ω k .

Observe that Y τ k ; τ k ( s, a ) = D k = γ ′′ D k + (1 -γ ′′ ) D k := γ ′′ D k + ρ τ k . We can rewrite the dynamics of Y t ; τ k ( s, a ) as

<!-- formula-not-decoded -->

where ρ t +1 = (1 -α t ) ρ t for t ∈ T A . By the definition of ρ t , we obtain

<!-- formula-not-decoded -->

where (i) follows because α i &lt; 1 and t ≥ τ k +1 , (ii) follows because | T A ( τ k , τ k +1 -1) | ≥ cτ ω k where T A ( t 1 , t 2 ) and | T A ( t 1 , t 2 ) | are defined in Definition 1, (iii) follows from Lemma 9, and (iv) holds because τ + k ≥ τ 1 and

<!-- formula-not-decoded -->

Next we check the value of the power -c 1+ 2 c κ + 1 τ ω 1 . Since κ ∈ (ln 2 , 1) and ∆ ∈ (0 , e κ -2) , we have ln(2 + ∆) ∈ (0 , κ ) . Further, observing τ ω 1 &gt; 1 κ -ln(2+∆) , we obtain ln(2 + ∆) + 1 τ ω 1 ∈ (0 , κ ) . Last, since c ≥ κ 2 ( 1 1 -ln(2+∆)+1 /τ ω 1 κ -1 ) = κ (ln(2+∆)+1 /τ ω 1 ) 2( κ -ln(2+∆) -1 /τ ω 1 ) , we have -c 1+ 2 c κ + 1 τ ω 1 ≤ -ln(2 + ∆) .

Thus, we have ρ t ≤ 1 -γ ′′ 2+∆ D k . Finally, we finish our proof by further observing that 1 -γ ′′ = 2 β .

It remains to bound | W t ; τ k ( s, a ) | ≤ ( 1 -2 2+∆ ) βD k for t ∈ [ τ k +1 , τ k +2 ) . Combining the bounds of Y t ; τ k and W t ; τ k yields ( γ ′′ + β ) D k = (1 -β ) D k = D k +1 . Since W t ; τ k is stochastic, we need to derive the probability for the bound to hold. To this end, we first rewrite the dynamics of W t ; τ k defined in Lemma 12 as

<!-- formula-not-decoded -->

Next, we introduce a new sequence { W l t ; τ k ( s, a ) } as

<!-- formula-not-decoded -->

Thus we have W t ; τ k ( s, a ) = W t -1 -τ k t ; τ k ( s, a ) . Then we have the following lemma.

Lemma14. For any t ∈ [ τ k +1 , τ k +2 ] and 1 ≤ l ≤ t -τ k -1 , { W l t ; τ k ( s, a ) } is a martingale sequence and satisfies

<!-- formula-not-decoded -->

Proof. Observe that

<!-- formula-not-decoded -->

Since E [ w t |F t -1 ] = 0 , we have

<!-- formula-not-decoded -->

Thus { W l t ; τ k ( s, a ) } is a martingale sequence. In addition, since l ≥ 1 and α t ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

Further, we obtain | w t ( s, a ) | = |T t Q A t ( s, a ) -T Q A t ( s, a ) | ≤ 2 Q max 1 -γ = V max . Thus

<!-- formula-not-decoded -->

Next, we bound W t ; τ k ( s, a ) . Fix ˜ glyph[epsilon1] &gt; 0 . Then for any t ∈ [ τ k +1 , τ k +2 ) , we have P [ | W t ; τ k ( s, a ) | &gt; ˜ glyph[epsilon1] | t ∈ [ τ k +1 , τ k +2 ) , E, F ]

<!-- formula-not-decoded -->

where (i) follows from Lemma 7, (ii) follows from Lemma 14, (iii) follows because | T A ( t 1 , t 2 ) | ≤ t 2 -t 1 +1 and (iv) holds because

<!-- formula-not-decoded -->

## Proof of Proposition 2

Now we bound ‖ r t ‖ by combining the bounds of Y t ; τ k and W t ; τ k . Applying the union bound in Lemma 8 yields

<!-- formula-not-decoded -->

where (i) follows because D k ≥ D m ≥ glyph[epsilon1] , and (ii) follows from Lemma 9 by substituting a = 16 c ( c + κ ) V 2 max κ 2 ( ∆ 2+∆ ) 2 β 2 glyph[epsilon1] 2 , b = 1 and observing that

<!-- formula-not-decoded -->

Note that Y t ; τ k ( s, a ) is deterministic. We complete this proof by observing that

<!-- formula-not-decoded -->

## B.3 Part III: Bounding ∥ ∥ Q A t -Q ∗ ∥ ∥

We combine the results in the first two parts, and provide a high probability bound on ‖ r t ‖ with further probabilistic arguments, which exploit the high probability bounds on P ( E ) in Proposition 1 and P ( F ) in the following lemma.

Lemma 15. Let the sequence τ k be the same as given in Lemma 10, i.e. τ k +1 = τ k + 2 c κ τ ω k for k ≥ 1 . Then we have

<!-- formula-not-decoded -->

where I A k denotes the number of iterations updating Q A at epoch k .

Proof. The event updating Q A is a binomial random variable. To be specific, at iteration t we define

<!-- formula-not-decoded -->

Clearly, the events are independent across iterations. Therefore, for a given epoch [ τ k , τ k +1 ) , I A k = ∑ τ k +1 -1 t = τ k J A t is a binomial random variable satisfying the distribution Binomial ( τ k +1 -τ k , 0 . 5) . In the following, we use the tail bound of a binomial random variable. That is, if a random variable X ∼ Binomial ( n, p ) , by Hoeffding's inequality we have P ( X ≤ x ) ≤ exp ( -2( np -x ) 2 n ) for x &lt; np , which implies P ( X ≤ κnp ) ≤ exp ( -2 np 2 (1 -κ ) 2 ) for any fixed κ ∈ (0 , 1) .

If k = 0 , I A 0 ∼ Binomial ( τ 1 , 0 . 5) . Thus the tail bound yields

<!-- formula-not-decoded -->

If k ≥ 1 , since τ k +1 -τ k = 2 c κ τ ω k , we have I A k ∼ Binomial ( 2 c κ τ ω k , 0 . 5 ) . Thus the tail bound of a binomial random variable gives

<!-- formula-not-decoded -->

Then by the union bound, we have

<!-- formula-not-decoded -->

We further give the following Lemma 16 and Lemma 17 before proving Theorem 1. Lemma 16 characterizes the number of blocks to achieve glyph[epsilon1] -accuracy given D k defined in Lemma 10.

Lemma 16. Let D k +1 = (1 -β ) D k with β = 1 -γ 4 , D 0 = 2 γV max 1 -γ . Then for m ≥ 4 1 -γ ln 2 γV max glyph[epsilon1] (1 -γ ) , we have D m ≤ glyph[epsilon1] .

Proof. By the definition of D k , we have D k = (1 -β ) k D 0 . Then we obtain

<!-- formula-not-decoded -->

Further observe that ln 1 1 -x ≤ x if x ∈ (0 , 1) . Thus we have

<!-- formula-not-decoded -->

From the above lemma, it suffices to find the starting time at epoch m ∗ = ⌈ 4 1 -γ ln 2 γV max glyph[epsilon1] (1 -γ ) ⌉ .

The next lemma is useful to calculate the total iterations given the initial epoch length and number of epochs.

Lemma 17. (Even-Dar and Mansour, 2003, Lemma 32) Consider a sequence { x k } satisfying

<!-- formula-not-decoded -->

Then for any constant ω ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

## Proof of Theorem 1

Now we are ready to prove Theorem 1 based on the results obtained so far.

Let m ∗ = ⌈ 4 1 -γ ln 2 γV max glyph[epsilon1] (1 -γ ) ⌉ , then G m ∗ -1 ≥ σglyph[epsilon1],D m ∗ -1 ≥ glyph[epsilon1] . Thus we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) follows from Lemma 10, (ii) follows from Proposition 1 and 2 and (iii) holds due to the fact that

<!-- formula-not-decoded -->

By setting

<!-- formula-not-decoded -->

we obtain

<!-- formula-not-decoded -->

Considering the conditions on ˆ τ 1 in Proposition 1 and Proposition 2, we choose

<!-- formula-not-decoded -->

Finally, applying the number of iterations m ∗ = ⌈ 4 1 -γ ln 2 γV max glyph[epsilon1] (1 -γ ) ⌉ and Lemma 17, we conclude that it suffices to let

<!-- formula-not-decoded -->

to attain an glyph[epsilon1] -accurate Q-estimator.

## C Proof of Theorem 2

The main idea of this proof is similar to that of Theorem 1 with further efforts to characterize the effects of asynchronous sampling. The proof also consists of three parts: (a) Part I which analyzes the stochastic error propagation between the two Q-estimators ∥ ∥ Q B t -Q A t ∥ ∥ ; (b) Part II which analyzes the error dynamics between one Q-estimator and the optimum ∥ ∥ Q A t -Q ∗ ∥ ∥ conditioned on the error event in Part I; and (c) Part III which bounds the unconditional error ∥ ∥ Q A t -Q ∗ ∥ ∥ .

To proceed the proof, we first introduce the following notion of valid iterations for any fixed stateaction pair ( s, a ) .

<!-- formula-not-decoded -->

Definition 2. We define T ( s, a ) as the collection of iterations if a state-action pair ( s, a ) is used to update the Q-function Q A or Q B , and T A ( s, a ) as the collection of iterations specifically updating Q A ( s, a ) . In addition, we denote T ( s, a, t 1 , t 2 ) and T A ( s, a, t 1 , t 2 ) as the set of iterations updating ( s, a ) and Q A ( s, a ) between time t 1 and t 2 , respectively. That is,

<!-- formula-not-decoded -->

Correspondingly, the number of iterations updating ( s, a ) between time t 1 and t 2 equals the cardinality of T ( s, a, t 1 , t 2 ) which is denoted as | T ( s, a, t 1 , t 2 ) | . Similarly, the number of iterations updating Q A ( s, a ) between time t 1 and t 2 is denoted as | T A ( s, a, t 1 , t 2 ) | .

Given Assumption 1, we can obtain some properties of the quantities defined above.

Lemma 18. It always holds that | T ( s, a, t 1 , t 2 ) | ≤ t 2 -t 1 +1 and | T A ( s, a, t 1 , t 2 ) | ≤ t 2 -t 1 +1 . In addition, suppose that Assumption 1 holds. Then we have T ( s, a, t, t +2 kL -1) ≥ k for any t ≥ 0 .

Proof. Since in a consecutive 2 L running iterations of Algorithm 1, either Q A or Q B is updated at least L times. Then following from Assumption 1, ( s, a ) is visited at least once for each 2 L running iterations of Algorithm 1, which immediately implies this proposition.

Now we proceed our proof by three parts.

## C.1 Part I: Bounding ∥ ∥ Q B t -Q A t ∥ ∥

Weupper bound ∥ ∥ Q B t -Q A t ∥ ∥ block-wisely using a decreasing sequence { G q } q ≥ 0 as defined in Proposition 3 below.

Proposition 3. Fix glyph[epsilon1] &gt; 0 , κ ∈ (ln 2 , 1) and ∆ ∈ (0 , e κ -2) . Consider asynchronous double Q-learning using a polynomial learning rate α t = 1 t ω with ω ∈ (0 , 1) . Suppose that Assumption 1 holds. Let G q = (1 -ξ ) q G 0 with G 0 = V max and ξ = 1 -γ 4 . Let ˆ τ q +1 = ˆ τ q + 2 cL κ ˆ τ ω q for q ≥ 1 with c ≥ Lκ (ln(2+∆)+1 /τ ω 1 ) 2( κ -ln(2+∆) -1 /τ ω 1 ) and ˆ τ 1 as the finishing time of the first block satisfying

<!-- formula-not-decoded -->

Then for any n such that G n ≥ σglyph[epsilon1] , we have

<!-- formula-not-decoded -->

The proof of Proposition 3 consists of the following steps. Since the main idea of the proofs is similar to that of Proposition 1, we will focus on pointing out the difference. We continue to use the notation u BA t ( s, a ) := Q B t ( s, a ) -Q A t ( s, a ) .

## Step 1: Characterizing the dynamics of u BA t

First, we observe that when ( s, a ) is visited at time t , i.e., t ∈ T ( s, a ) , Lemmas 2 and 3 still apply. Otherwise, u BA is not updated. Thus, we have

<!-- formula-not-decoded -->

where F t satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For t ∈ T ( s, a ) , we rewrite the dynamics of u BA t ( s, a ) as

<!-- formula-not-decoded -->

In the following steps, we use induction to proceed the proof of Proposition 3. Given G q defined in Proposition 3, since ∥ ∥ u BA t ∥ ∥ ≤ G 0 holds for all t , and thus it holds for t ∈ [0 , ˆ τ 1 ] . Now suppose ˆ τ q satisfies that ∥ ∥ u BA t ∥ ∥ ≤ G q for any t ≥ ˆ τ q . Then we will show there exists ˆ τ q +1 = ˆ τ q + 2 cL κ ˆ τ ω q such that ∥ ∥ u BA t ∥ ∥ ≤ G q +1 for any t ≥ ˆ τ q +1 .

## Step 2: Constructing sandwich bounds

We first observe that the following sandwich bound still holds for all t ≥ ˆ τ q .

<!-- formula-not-decoded -->

where Z t ;ˆ τ q ( s, a ) is defined as

<!-- formula-not-decoded -->

with the initial condition Z ˆ τ q ;ˆ τ q ( s, a ) = 0 , and X t ;ˆ τ q ( s, a ) is defined as

<!-- formula-not-decoded -->

with X ˆ τ q ;ˆ τ q ( s, a ) = G q , γ ′ = 1+ γ 2 .

This claim can be shown by induction. This bound clearly holds for the initial case with t = ˆ τ q . Assume that it still holds for iteration t . If t ∈ T ( s, a ) , the proof is the same as that of Lemma 3. If t / ∈ T ( s, a ) , since all three sequences do not change from time t to time t +1 , the sandwich bound still holds. Thus we conclude this claim.

## Step 3: Bounding X t ;ˆ τ q ( s, a )

Next, we bound the deterministic sequence X t ;ˆ τ q ( s, a ) . Observe that X t ;ˆ τ q ( s, a ) ≤ G q for any t ≥ ˆ τ q . We will next show that X t ;ˆ τ q ( s, a ) ≤ ( γ ′ + 2 2+∆ ξ ) G q for any t ∈ [ˆ τ q +1 , ˆ τ q +2 ) where ˆ τ q +1 = ˆ τ q + 2 cL κ ˆ τ ω q .

Similarly to the proof of Lemma 5, we still rewrite X ˆ τ q ;ˆ τ q ( s, a ) as X ˆ τ q ;ˆ τ q ( s, a ) = G q = γ ′ G q + (1 -γ ′ ) G q := γ ′ G q + ρ ˆ τ q . However, in this case the dynamics of X t ;ˆ τ q ( s, a ) is different, which is represented as where

<!-- formula-not-decoded -->

where (i) follows from Lemma 18, (ii) follows Lemma 4, and (iii) follows because ˆ τ q ≥ ˆ τ 1 and

<!-- formula-not-decoded -->

Since κ ∈ (ln 2 , 1) and ∆ ∈ (0 , e κ -2) , we have ln(2 + ∆) ∈ (0 , κ ) . Further, observing ˆ τ 1 ω &gt; 1 κ -ln(2+∆) , we obtain ln(2 + ∆) + 1 ˆ τ 1 ω ∈ (0 , κ ) . Last, since c ≥ Lκ (ln(2+∆)+1 / ˆ τ 1 ω ) 2( κ -ln(2+∆) -1 / ˆ τ 1 ω ) , we have -c 1+ 2 c κ + 1 ˆ τ 1 ω ≤ -ln(2 + ∆) .

Finally, combining the above observations with the fact 1 -γ ′ = 2 ξ , we conclude that for any t ≥ ˆ τ q +1 = ˆ τ q + 2 cL κ ˆ τ ω q ,

<!-- formula-not-decoded -->

## Step 4: Bounding Z t ;ˆ τ q ( s, a )

It remains to bound the stochastic sequence Z t ;ˆ τ q ( s, a ) by ∆ 2+∆ ξG q at epoch q +1 . We define an auxiliary sequence { Z l t ;ˆ τ q ( s, a ) } (which is different from that in (9)) as:

<!-- formula-not-decoded -->

Following the same arguments as the proof of Lemma 6, we conclude that { Z l t ;ˆ τ q ( s, a ) } is a martingale sequence and satisfies

<!-- formula-not-decoded -->

In addition, note that

<!-- formula-not-decoded -->

Then we apply Azuma' inequality in Lemma 7 and obtain

<!-- formula-not-decoded -->

where (i) follows from Lemma 18.

## Step 5: Taking union over all blocks

Finally, using the union bound of Lemma 8 yields

<!-- formula-not-decoded -->

where (i) follows from G q ≥ G n ≥ σglyph[epsilon1] , (ii) follows from Lemma 9 by substituting a = 64 cL ( cL + κ ) V 2 max κ 2 ( ∆ 2+∆ ) 2 ξ 2 σ 2 glyph[epsilon1] 2 , b = 1 and observing that

<!-- formula-not-decoded -->

and (iii) follows from ˆ τ q ≥ ˆ τ 1 .

## C.2 Part II: Conditionally bounding ∥ ∥ Q A t -Q ∗ ∥ ∥

We upper bound ∥ ∥ Q A t -Q ∗ ∥ ∥ block-wisely by a decreasing sequence { D k } k ≥ 0 conditioned on the following two events: fix a positive integer m ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where I A k denotes the number of iterations updating Q A at epoch k , τ k is the starting iteration index of the k +1 th block, and ω is the parameter of the polynomial learning rate. Roughly, Event G requires that the difference between the two Q-function estimators are bounded appropriately, and Event H requires that Q A is sufficiently updated in each epoch. Again, we will design { D k } k ≥ 0 in a way such that the occurrence of Event G can be implied from the event that ∥ ∥ u BA t ∥ ∥ is bounded by { G q } q ≥ 0 (see Lemma 19 below). A lower bound of the probability for Event H to hold is characterized in Lemma 15 in Part III.

Proposition 4. Fix glyph[epsilon1] &gt; 0 , κ ∈ (ln 2 , 1) and ∆ ∈ (0 , e κ -2) . Consider asynchronous double Q-learning using a polynomial learning rate α t = 1 t ω with ω ∈ (0 , 1) . Let { G q } , { ˆ τ q } be as defined in Proposition 3. Define D k = (1 -β ) k V max σ with β = 1 -γ (1+ σ ) 2 and σ = 1 -γ 2 γ . Let τ k = ˆ τ k for

k ≥ 0 . Suppose that c ≥ L (ln(2+∆)+1 /τ ω 1 ) 2( κ -ln(2+∆) -1 /τ ω 1 ) and τ 1 = ˆ τ 1 as the finishing time of the first epoch satisfies

<!-- formula-not-decoded -->

Then for any m such that D m ≥ glyph[epsilon1] , we have

<!-- formula-not-decoded -->

Recall that in the proof of Proposition 2, Q A is not updated at each iteration and thus we introduced notations T A and T A ( t 1 , t 2 ) in Definition 1 to capture the convergence of the error ∥ ∥ Q A -Q ∗ ∥ ∥ . In this proof, the only difference is that when choosing to update Q A , only one ( s, a ) -pair is visited. Therefore, the proof of Proposition 4 is similar to that of Proposition 2, where most of the arguments simply substitute T A , T A ( t 1 , t 2 ) in the proof of Proposition 2 by T A ( s, a ) , T A ( s, a, t 1 , t 2 ) in Definition 2, respectively. Certain bounds are affected by such substitutions. In the following, we proceed the proof of Proposition 4 in five steps, and focus on pointing out the difference from the proof of Proposition 2. More details can be referred to Appendix B.2.

## Step 1: Coupling { D k } k ≥ 0 and { G q } q ≥ 0

We establish the relationship between { D k } k ≥ 0 and { G q } q ≥ 0 in the same way as Lemma 10. For the convenience of reference, we restate Lemma 10 in the following.

Lemma 19. Let { G q } be defined in Proposition 3, and let D k = (1 -β ) k V max σ with β = 1 -γ (1+ σ ) 2 and σ = 1 -γ 2 γ . Then we have

<!-- formula-not-decoded -->

given that τ k = ˆ τ k .

## Step 2: Constructing sandwich bounds

Let r t ( s, a ) = Q A ( s, a ) -Q ∗ ( s, a ) and τ k be such that ‖ r t ‖ ≤ D k for all t ≥ τ k . The requirement of Event G yields

<!-- formula-not-decoded -->

where W t ; τ k ( s, a ) is defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with Y τ k ; τ k ( s, a ) = D k and γ ′′ = γ (1 + σ ) .

## Step 3: Bounding Y t ; τ k ( s, a )

Next, we first bound Y t ; τ k ( s, a ) . Observe that Y t ; τ k ( s, a ) ≤ D k for any t ≥ τ k . We will bound Y t ; τ k ( s, a ) by ( γ ′′ + 2 2+∆ β ) D k for block k +1 .

We use a similar representation of Y t ; τ k ( s, a ) as in the proof of Lemma 13, which is given by

<!-- formula-not-decoded -->

where ρ t +1 = (1 -α t ) ρ t for t ∈ T A ( s, a ) . By the definition of ρ t , we obtain

<!-- formula-not-decoded -->

where (i) follows because α i &lt; 1 and t ≥ τ k +1 , (ii) follows from Proposition 18 and the requirement of event H , (iii) follows from Lemma 9, and (iv) holds because τ + k ≥ τ 1 and

<!-- formula-not-decoded -->

Since κ ∈ (ln 2 , 1) and ∆ ∈ (0 , e κ -2) , we have ln(2 + ∆) ∈ (0 , κ ) . Further, observing ˆ τ 1 ω &gt; 1 κ -ln(2+∆) , we obtain ln(2 + ∆) + 1 ˆ τ 1 ω ∈ (0 , κ ) . Last, since c ≥ L (ln(2+∆)+1 / ˆ τ 1 ω ) 2( κ -ln(2+∆) -1 / ˆ τ 1 ω ) , we have -c 1+ 2 c κ + 1 ˆ τ 1 ω ≤ -ln(2 + ∆) .

Then, we have ρ t ≤ 1 -γ ′′ 2+∆ D k . Thus we conclude that for any t ∈ [ τ k +1 , τ k +2 ] ,

<!-- formula-not-decoded -->

## Step 4: Bounding W t ; τ k ( s, a )

It remains to bound | W t ; τ k ( s, a ) | ≤ ( 1 -2 2+∆ ) βD k for t ∈ [ τ k +1 , τ k +2 ) .

Similarly to Appendix B.2.4, we define a new sequence { W l t ; τ k ( s, a ) } as

<!-- formula-not-decoded -->

The same arguments as the proof of Lemma 14 yields

<!-- formula-not-decoded -->

If we fix ˜ glyph[epsilon1] &gt; 0 , then for any t ∈ [ τ k +1 , τ k +2 ) we have

P

[

|

W

t

;

τ

k

(

s, a

)

|

&gt;

˜

glyph[epsilon1]

|

t

∈

[

τ

k

+1

, τ

k

+2

)

, G, H

]

<!-- formula-not-decoded -->

where (i) follows from Proposition 18 and (ii) holds because

<!-- formula-not-decoded -->

## Step 5: Taking union over all blocks

Applying the union bound in Lemma 8, we obtain

<!-- formula-not-decoded -->

where (i) follows because D k ≥ D m ≥ glyph[epsilon1] , and (ii) follows from Lemma 9 by substituting a = 16 cL ( cL + κ ) V 2 max κ 2 ( ∆ 2+∆ ) 2 β 2 glyph[epsilon1] 2 , b = 1 and observing that

<!-- formula-not-decoded -->

## C.3 Part III: Bound ∥ ∥ Q A t -Q ∗ ∥ ∥

In order to obtain the unconditional high-probability bound on ∥ ∥ Q A t -Q ∗ ∥ ∥ , we first characterize a lower bound on the probability of Event H . Note that the probability of Event G is lower bounded in Proposition 3.

Lemma 20. Let the sequence τ k be the same as given in Lemma 19, i.e. τ k +1 = τ k + 2 cL κ τ ω k for k ≥ 1 . Define I A k as the number of iterations updating Q A at epoch k . Then we have

<!-- formula-not-decoded -->

Proof. We use the same idea as the proof of Lemma 15. Since we only focus on the blocks with k ≥ 1 , I A k ∼ Binomial ( 2 cL κ τ ω k , 0 . 5 ) in such a case. Thus the tail bound of a binomial random

variable gives

<!-- formula-not-decoded -->

Then by the union bound, we have

<!-- formula-not-decoded -->

Following from Lemma 16, it suffices to determine the starting time at epoch m ∗ = ⌈ 4 1 -γ ln 2 γV max glyph[epsilon1] (1 -γ ) ⌉ . This can be done by using Lemma 17 if we have ˆ τ 1 .

Now we are ready to prove the main result of Theorem 2. By the definition of m ∗ , we know D m ∗ -1 ≥ glyph[epsilon1],G m ∗ -1 ≥ σglyph[epsilon1] . Then we obtain

<!-- formula-not-decoded -->

where (i) follows from Lemma 19, (ii) follows from Propositions 3 and 4 and (iii) holds due to the fact that

<!-- formula-not-decoded -->

By setting

<!-- formula-not-decoded -->

we obtain

<!-- formula-not-decoded -->

Combining with the requirement of ˆ τ 1 in Propositions 3 and 4, we can choose

<!-- formula-not-decoded -->

Finally, applying m ∗ = ⌈ 4 1 -γ ln 2 γV max glyph[epsilon1] (1 -γ ) ⌉ and Lemma 17, we conclude that it suffices to let

<!-- formula-not-decoded -->

to attain an glyph[epsilon1] -accurate Q-estimator.