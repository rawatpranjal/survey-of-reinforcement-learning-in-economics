## Settling the Sample Complexity of Online Reinforcement Learning

Zihan Zhang ∗ Princeton

Yuxin Chen † UPenn

Jason D. Lee ∗ Princeton

July 2023;

Revised:

April 2025

## Abstract

A central issue lying at the heart of online reinforcement learning (RL) is data efficiency. While a number of recent works achieved asymptotically minimal regret in online RL, the optimality of these results is only guaranteed in a 'large-sample' regime, imposing enormous burn-in cost in order for their algorithms to operate optimally. How to achieve minimax-optimal regret without incurring any burn-in cost has been an open problem in RL theory.

We settle this problem for finite-horizon inhomogeneous Markov decision processes. Specifically, we prove that a modified version of MVP (Monotonic Value Propagation), an optimistic model-based algorithm proposed by Zhang et al. (2021a), achieves a regret on the order of (modulo log factors)

<!-- formula-not-decoded -->

where S is the number of states, A is the number of actions, H is the horizon length, and K is the total number of episodes. This regret matches the minimax lower bound for the entire range of sample size K ≥ 1, essentially eliminating any burn-in requirement. It also translates to a PAC sample complexity (i.e., the number of episodes needed to yield ε -accuracy) of SAH 3 ε 2 up to log factor, which is minimaxoptimal for the full ε -range. Further, we extend our theory to unveil the influences of problem-dependent quantities like the optimal value/cost and certain variances. The key technical innovation lies in a novel analysis paradigm (based on a new concept called 'profiles') to decouple complicated statistical dependency across the sample trajectories - a long-standing challenge facing the analysis of online RL in the sample-starved regime.

## Contents

| 1   | Introduction                                         | Introduction                                         |   2 |
|-----|------------------------------------------------------|------------------------------------------------------|-----|
|     | 1.1                                                  | Inadequacy of prior art: enormous burn-in cost       |   3 |
|     | 1.2                                                  | A peek at our main contributions . . . . . . . .     |   5 |
|     | 1.3                                                  | Related works . . . . . . . . . . . . . . . . . . .  |   6 |
|     | 1.4                                                  | Notation . . . . . . . . . . . . . . . . . . . . . . |   7 |
| 2   | Problem formulation                                  | Problem formulation                                  |   8 |
| 3   | A model-based algorithm: Monotonic Value Propagation | A model-based algorithm: Monotonic Value Propagation |   9 |
| 4   | Key technical innovations                            | Key technical innovations                            |  11 |
|     | 4.1                                                  | Technical barriers in prior theory for UCBVI . .     |  12 |
|     | 4.2                                                  | Our approach . . . . . . . . . . . . . . . . . . .   |  13 |

∗ Department of Electrical and Computer Engineering, Princeton University; email: {zz5478,jasonlee}@princeton.edu .

† Department of Statistics and Data Science, University of Pennsylvania; email: yuxinc@wharton.upenn.edu .

‡ Paul G. Allen School of Computer Science and Engineering, University of Washington; email: ssdu@cs.washington.edu .

Simon S. Du ‡

U. Washington

| 5 Proof of Theorem 1   | 5 Proof of Theorem 1                                               | 5 Proof of Theorem 1                                               |   16 |
|------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|------|
| 6                      | Extensions                                                         | Extensions                                                         |   20 |
|                        | 6.1                                                                | Value-based regret bounds                                          |   20 |
|                        | 6.2                                                                | Cost-based regret bounds .                                         |   20 |
|                        | 6.3                                                                | Variance-dependent regret bound . . . . . . . . . . . . .          |   21 |
| 7                      | Discussion                                                         | Discussion                                                         |   22 |
| A                      | Preliminary facts                                                  | Preliminary facts                                                  |   23 |
| B                      | Proofs of key lemmas in Section 4                                  | Proofs of key lemmas in Section 4                                  |   25 |
|                        | B.1                                                                | Proof of Lemma 5 . . . . . . . . . .                               |   25 |
|                        | B.2                                                                | Proof of Lemma 6 . . . . . . . . . . . . . . . . .                 |   26 |
|                        | B.3                                                                | Proof of Lemma 7 . .                                               |   27 |
| C                      | Proofs of auxiliary lemmas in Section 5                            | Proofs of auxiliary lemmas in Section 5                            |   29 |
|                        | C.1                                                                | Proof of Lemma 8                                                   |   29 |
|                        | C.2                                                                | Proof of Lemma 10 . .                                              |   31 |
|                        | C.3                                                                | Proof of Lemma 11 .                                                |   32 |
| D                      | Proof of the value-based regret bound (proof of Theorem 2)         | Proof of the value-based regret bound (proof of Theorem 2)         |   34 |
| E                      | Proof of the cost-based regret bound (proof of Theorem 3)          | Proof of the cost-based regret bound (proof of Theorem 3)          |   37 |
| F                      | Proof of the variance-dependent regret bounds (proof of Theorem 4) | Proof of the variance-dependent regret bounds (proof of Theorem 4) |   40 |
|                        | F.1                                                                | Proof of Lemma 24                                                  |   41 |
|                        | F.2                                                                | Proof of Lemma 25 .                                                |   49 |
| G                      | Minimax lower bounds                                               | Minimax lower bounds                                               |   53 |
|                        | G.1                                                                | Proof of Theorem 12                                                |   53 |
|                        | G.2                                                                | Proof of Theorem 13 . . . . .                                      |   54 |
|                        | G.3                                                                | Proof of Theorem 14 . . . .                                        |   55 |

## 1 Introduction

In reinforcement learning (RL), an agent is often asked to learn optimal decisions (i.e., the ones that maximize cumulative reward) through real-time 'trial-and-error' interactions with an unknown environment. This task is commonly dubbed as online RL , underscoring the critical role of adaptive online data collection and differentiating it from other RL settings that rely upon pre-collected data. A central challenge in achieving sample-efficient online RL boils down to how to optimally balance exploration and exploitation during data collection, namely, how to trade off the potential revenue of exploring unknown terrain/dynamics against the benefit of exploiting past experience. While decades-long effort has been invested towards unlocking the capability of online RL, how to fully characterize - and more importantly, attain - its fundamental performance limit remains largely unsettled.

In this paper, we take an important step towards settling the sample complexity limit of online RL, focusing on tabular Markov Decision Processes (MDPs) with finite horizon and finite state-action space. More concretely, imagine that one seeks to learn a near-optimal policy of a time-inhomogeneous MDP with S states, A actions, and horizon length H , and is allowed to execute the MDP of interest K times to collect K sample episodes each of length H . This canonical problem is among the most extensively studied in the

RL literature, with formal theoretical pursuit dating back to more than 25 years ago (e.g., Kearns and Singh (1998b)). Numerous works have since been devoted to improving the sample efficiency and/or refining the analysis framework (Azar et al., 2017; Bai et al., 2019; Brafman and Tennenholtz, 2003; Dann et al., 2017; Domingues et al., 2021; Jaksch et al., 2010; Jin et al., 2018; Kakade, 2003; Li et al., 2021a; Ménard et al., 2021; Zanette and Brunskill, 2019; Zhang et al., 2021a, 2020). As we shall elucidate momentarily, however, information-theoretic optimality has only been achieved in the 'large-sample' regime. When it comes to the most challenging sample-hungry regime, there remains a substantial gap between the state-of-the-art regret upper bound and the best-known minimax lower bound, which motivates the research of this paper.

## 1.1 Inadequacy of prior art: enormous burn-in cost

While past research has obtained asymptotically optimal (i.e., optimal when K approaches infinity) regret bounds in the aforementioned setting, all of these results incur an enormous burn-in cost - that is, the minimum sample size needed for an algorithm to operate sample-optimally - which we explain in the sequel. For simplicity of presentation, we assume that each immediate reward lies within the normalized range [0 , 1] when discussing the prior art.

Minimax lower bound. To provide a theoretical benchmark, we first make note of the best-known minimax regret lower bound developed by Domingues et al. (2021); Jin et al. (2018): 1

<!-- formula-not-decoded -->

assuming that the immediate reward at each step falls within [0 , 1] and imposing no restriction on K . Given that a regret of O ( HK ) can be trivially achieved (as the sum of rewards in any K episodes cannot exceed HK ), we shall sometimes drop the HK term and simply write

<!-- formula-not-decoded -->

Prior upper bounds and burn-in cost. We now turn to the upper bounds developed in prior literature. For ease of presentation, we shall assume

<!-- formula-not-decoded -->

in the rest of this subsection unless otherwise noted. Log factors are also ignored in the discussion below.

The first paper that achieves asymptotically optimal regret is Azar et al. (2017), which came up with a model-based algorithm called UCBVI that enjoys a regret bound ˜ O ( √ SAH 3 K + H 3 S 2 A ) . A close inspection reveals that this regret matches the minimax lower bound (2) if and only if due to the presence of the lower-order term H 3 S 2 A in the regret bound. This burn-in cost is clearly undesirable, since the sample size available in many practical scenarios might be far below this requirement.

<!-- formula-not-decoded -->

In light of its fundamental importance in contemporary RL applications (which often have very large dimensionality and relatively limited data collection capability), reducing the burn-in cost without compromising sample efficiency has emerged as a central problem in recent pursuit of RL theory (Agarwal et al., 2020; Dann et al., 2019; Li et al., 2022, 2021a,c; Ménard et al., 2021; Sidford et al., 2018b; Zanette and Brunskill, 2019; Zhang et al., 2021a; Zhou et al., 2023). The state-of-the-art regret upper bounds for finite-horizon inhomogeneous MDPs can be summarized below (depending on the size of K ):

<!-- formula-not-decoded -->

| Algorithm                                 | Regret upper bound                                    | Range of K that attains optimal regret   | Sample complexity (or PAC bound)     |
|-------------------------------------------|-------------------------------------------------------|------------------------------------------|--------------------------------------|
| MVP (this work, Theorem 1)                | min { √ SAH 3 K,HK }                                  | [1 , ∞ )                                 | SAH 3 ε 2                            |
| UCBVI (Azar et al., 2017)                 | min { √ SAH 3 K + S 2 AH 3 , HK }                     | [ S 3 AH 3 , ∞ )                         | SAH 3 ε 2 + S 2 AH 3 ε               |
| ORLC (Dann et al., 2019)                  | min { √ SAH 3 K + S 2 AH 4 , HK }                     | [ S 3 AH 5 , ∞ )                         | SAH 3 ε 2 + S 2 AH 4 ε               |
| EULER (Zanette and Brunskill, 2019)       | min { √ SAH 3 K + S 3 / 2 AH 3 ( √ S + √ H ) , HK }   | [ S 2 AH 3 ( √ S + √ H ) , ∞ )           | SAH 3 ε 2 + S 2 AH 3 ( √ S + √ H ε   |
| UCB - Adv (Zhang et al., 2020)            | min { √ SAH 3 K + S 2 A 3 / 2 H 33 / 4 K 1 / 4 , HK } | [ S 6 A 4 H 27 , ∞ )                     | SAH 3 ε 2 + S 8 / 3 A 2 H 11 ε 4 / 3 |
| MVP (Zhang et al., 2021a)                 | min { √ SAH 3 K + S 2 AH 2 , HK }                     | [ S 3 AH, ∞ )                            | SAH 3 ε 2 + S 2 AH 2 ε               |
| UCBMQ (Ménard et al., 2021)               | min { √ SAH 3 K + SAH 4 , HK }                        | [ SAH 5 , ∞ )                            | SAH 3 ε 2 + SAH 4 ε                  |
| Q - Earlysettled - Adv (Li et al., 2021a) | min { √ SAH 3 K + SAH 6 , HK }                        | [ SAH 9 , ∞ )                            | SAH 3 ε 2 + SAH 6 ε                  |
| Lower bound (Domingues et al., 2021)      | min √ SAH 3 K,HK                                      | n/a                                      | SAH 3 ε 2                            |

{

}

Table 1: Comparisons between our result and prior works that achieve asymptotically optimal regret for finite-horizon inhomogeneous MDPs (with all log factors omitted), where S (resp. A ) is the number of states (resp. actions), H is the planning horizon, and K is the number of episodes. The third column reflects the burn-in cost, and the sample complexity (or PAC bound) refers to the number of episodes needed to yield ε accuracy. The results provided here account for all K ≥ 1 or all ε ∈ (0 , H ]. Our paper is the only one that gives regret (resp. PAC) bound matching the minimax lower bound for the entire range of K (resp. ε ).

meaning that even the most advanced prior results fall short of sample optimality unless

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The interested reader is referred to Table 1 for more details about existing regret upper bounds and their associated sample complexities.

In summary, no prior theory was able to achieve optimal sample complexity in the data-hungry regime

<!-- formula-not-decoded -->

suffering from a significant barrier of either long horizon (as in the term SAH 5 ) or large state space (as in the term S 3 AH ). In fact, the information-theoretic limit is yet to be determined within this regime (i.e., neither the achievability results nor the lower bounds had been shown to be tight), although it has been conjectured by Ménard et al. (2021) that the lower bound (1) reflects the correct scaling for any sample size K . 2

Comparisons with other RL settings and key challenges. In truth, the incentives to minimize the burn-in cost and improve data efficiency arise in multiple other settings beyond online RL. For instance, in an idealistic setting that assumes access to a simulator (or a generative model) - a model that allows the learner to query arbitrary state-action pairs to draw samples - a recent work Li et al. (2024c) developed a perturbed model-based approach that is provably optimal without incurring any burn-in cost. Analogous results have been obtained in Li et al. (2021c) for offline RL - a setting that requires policy learning to be performed based on historical data - unveiling the full-range optimality of a pessimistic model-based algorithm.

Unfortunately, the algorithmic and analysis frameworks developed in the above two works fail to accommodate the online counterpart. The main hurdle stems from the complicated statistical dependency intrinsic

2 Note that the original conjecture in Ménard et al. (2021) was ˜ Θ ( √ SAH 3 K + SAH 2 ) . Combining it with the trivial upper bound HK allows one to remove the term SAH 2 (with a little algebra).

to episodic online RL; for instance, in online RL, the empirical transition probabilities and the running estimates of the value function are oftentimes statistically dependent in an intertwined manner (unless we waste data). How to decouple the intricate statistical dependency without compromising data efficiency constitutes the key innovation of this work. More precise, in-depth technical discussions will be provided in Section 4.

## 1.2 A peek at our main contributions

We are now positioned to summarize the main findings of this paper. Focusing on time-inhomogeneous finite-horizon MDPs, our main contributions can be divided into two parts: the first part fully settles the minimax-optimal regret and sample complexity of online RL, whereas the second part further extends and augments our theory to make apparent the impacts of certain problem-dependent quantities. Throughout this subsection, the regret metric Regret ( K ) captures the cumulative sub-optimality gap (i.e., the gap between the performance of the policy iterates and that of the optimal policy) over all K episodes, to be formally defined in (17).

## 1.2.1 Settling the optimal sample complexity with no burn-in cost

Our first result fully determines the sample complexity limit of online RL in a minimax sense, allowing one to attain the optimal regret regardless of the number K of episodes that can be collected.

Theorem 1. For any K ≥ 1 and any 0 &lt; δ &lt; 1 , there exists an algorithm (see Algorithm 1) obeying

<!-- formula-not-decoded -->

with probability at least 1 -δ .

The optimality of our regret bound (8) can be readily seen given that it matches the minimax lower bound (1) (modulo some logarithmic factor). One can also easily translate the above regret bound into sample complexity or probably approximately correct (PAC) bounds: the proposed algorithm is able to return an ε -suboptimal policy with high probability using at most

<!-- formula-not-decoded -->

(or equivalently, ˜ O ( SAH 4 ε 2 ) sample transitions as each episode has length H ). Remarkably, this result holds true for the entire ε range (i.e., any ε ∈ (0 , H ]), essentially eliminating the need of any burn-in cost. It is noteworthy that even in the presence of an idealistic generative model, this order of sample size is unimprovable (Azar et al., 2013; Li et al., 2024c).

The algorithm proposed herein is a modified version of MVP : Monotonic Value Propagation . Originally proposed by Zhang et al. (2021a), the MVP method falls under the category of model-based approaches, a family of algorithms that construct explicit estimates of the probability transition kernel before value estimation and policy learning. Notably, a technical obstacle that obstructs the progress in understanding model-based algorithms arises from the exceedingly large model dimensionality: given that the dimension of the transition kernel scales proportionally with S 2 , all existing analyses for model-based online RL fell short of effectiveness unless the sample size already far exceeds S 2 (Azar et al., 2017; Zhang et al., 2021a). To overcome this undesirable source of burn-in cost, a crucial step is to empower the analysis framework in order to accommodate the highly sub-sampled regime (i.e., a regime where the sample size scales linearly with S ), which we shall elaborate on in Section 4. The full proof of Theorem 1 will be provided in Section 5.

## 1.2.2 Extension: optimal problem-dependent regret bounds

In practice, RL algorithms often perform far more appealingly than what their worst-case performance guarantees would suggest. This motivates a recent line of works that investigate optimal performance in a more problem-dependent fashion (Dann et al., 2021; Fruit et al., 2018; Jin et al., 2020; Simchowitz and Jamieson,

2019; Talebi and Maillard, 2018; Tirinzoni et al., 2021; Wagenmaker et al., 2022; Xu et al., 2021; Yang et al., 2021; Zanette and Brunskill, 2019; Zhao et al., 2023; Zhou et al., 2023). Encouragingly, the proposed algorithm automatically achieves optimality on a more refined problem-dependent level, without requiring prior knowledge of additional problem-specific knowledge. This results in several extended theorems that take into account different problem-dependent quantities.

The first extension below investigates how the optimal value influences the regret bound.

Theorem 2 (Optimal value-dependent regret) . For any K ≥ 1 , Algorithm 1 satisfies where v /star is the value of the optimal policy averaged over the initial state distribution (to be formally defined in (43) ).

<!-- formula-not-decoded -->

Moreover, there is also no shortage of applications where the use of a cost function is preferred over a value function (Agarwal et al., 2017; Allen-Zhu et al., 2018; Lee et al., 2020; Wang et al., 2023). For this purpose, we provide another variation based upon the optimal cost.

Theorem 3 (Optimal cost-dependent regret) . For any K ≥ 1 and any 0 &lt; δ &lt; 1 , Algorithm 1 achieves with probability exceeding 1 -δ , where c /star denotes the cost of the optimal policy averaged over the initial state distribution (to be formally defined in (45) ).

<!-- formula-not-decoded -->

It is worth noting that: despite the apparent similarity between the statements of Theorem 2 and Theorem 3, they do not imply each other, although their proofs are very similar to each other.

Finally, we establish another regret bound that reflects the effect of certain variance quantities of interest.

Theorem 4 (Optimal variance-dependent regret) . For any K ≥ 1 and any 0 &lt; δ &lt; 1 , Algorithm 1 obeys with probability at least 1 -δ , where var is a certain variance-type metric (to be formally defined in (49) ).

<!-- formula-not-decoded -->

Two remarks concerning the above extensions are in order:

- In the worst-case scenarios, the quantities v /star , c /star and var can all be as large as the order of H , in which case Theorems 2-4 readily recover Theorem 1. In contrast, the advantages of Theorems 2-4 over Theorem 1 become more evident in those favorable cases (e.g., the situation where v /star /lessmuch H or c /star /lessmuch H , or the case when the environment is nearly deterministic (so that var ≈ 0)).
- Interestingly, the regret bounds in Theorems 2-4 all contain a lower-order term SAH 2 , and one might naturally wonder whether this term is essential. To demonstrate the unavoidable nature of this term and hence the optimality of Theorems 2-4, we will provide matching lower bounds, to be detailed in Section 6.

## 1.3 Related works

Let us take a moment to discuss several related theoretical works on tabular RL. Note that there has also been an active line of research that exploits low-dimensional function approximation to further reduce sample complexity, which is beyond the scope of this paper.

Our discussion below focuses on two mainstream approaches that have received widespread adoption: the model-based approach and the model-free approach. In a nutshell, model-based algorithms decouple model estimation and policy learning, and often use the learned transition kernel to compute the value function and find a desired policy. In stark contrast, the model-free approach attempts to estimate the optimal value function and optimal policy directly without explicit estimation of the model. In general, model-free algorithms only require O ( SAH ) memory - needed when storing the running estimates for Q-functions and value functions - while the model-based counterpart might require Ω( S 2 AH ) space in order to store the estimated transition kernel.

Sample complexity for RL with a simulator. As an idealistic setting that separates the consideration of exploration from that of estimation, RL with a simulator (or generative model) has been studied by numerous works, allowing the learner to draw independent samples for any state-action pairs (Agarwal et al., 2020; Azar et al., 2013; Beck and Srikant, 2012; Chen et al., 2020; Cui and Yang, 2021; Even-Dar and Mansour, 2003; Kakade, 2003; Kearns and Singh, 1998a; Li et al., 2024a, 2022, 2024c; Pananjady and Wainwright, 2020; Shi et al., 2023; Sidford et al., 2018a,b; Wainwright, 2019a,b). While both model-based and modelfree approaches are capable of achieving asymptotic sample optimality (Agarwal et al., 2020; Azar et al., 2013; Sidford et al., 2018b; Wainwright, 2019b), all model-free algorithms that enjoy asymptotically optimal sample complexity suffer from dramatic burn-in cost. Thus far, only the model-based approach has been shown to fully eliminate the burn-in cost for both discounted infinite-horizon MDPs and inhomogeneous finite-horizon MDPs (Li et al., 2024c). The full-range optimal sample complexity for time-homogeneous finite-horizon MDPs in the presence of a simulator remains open.

Sample complexity for offline RL. The subfield of offline RL is concerned with learning based purely on a pre-collected dataset (Levine et al., 2020). A frequently used mathematical model assumes that historical data are collected (often independently) using some behavior policy, and the key challenges (compared with RL with a simulator) come from distribution shift and incomplete data coverage. The sample complexity of offline RL has been the focus of a large strand of recent works, with asymptotically optimal sample complexity achieved by multiple algorithms (Jin et al., 2021; Li et al., 2024b, 2021b; Qu and Wierman, 2020; Rashidinejad et al., 2021; Ren et al., 2021; Shi et al., 2022; Wang et al., 2022; Xie et al., 2021; Yan et al., 2023; Yin et al., 2022). Akin to the simulator setting, the fully optimal sample complexity (without burn-in cost) has only been achieved via the model-based approach when it comes to discounted infinite-horizon and inhomogeneous finite-horizon settings (Li et al., 2024b). All asymptotically optimal model-free methods incur substantial burn-in cost. The case with time-homogeneous finite-horizon MDPs also remains unsettled.

Sample complexity for online RL. Obtaining optimal sample complexity (or regret bound) in online RL without incurring any burn-in cost has been one of the most fundamental open problems in RL theory. In fact, the past decades have witnessed a flurry of activity towards improving the sample efficiency of online RL, partial examples including Agrawal and Jia (2017); Bartlett and Tewari (2009); Brafman and Tennenholtz (2003); Cai et al. (2019); Dann and Brunskill (2015); Dann et al. (2017); Domingues et al. (2021); Dong et al. (2019); Efroni et al. (2019); Fruit et al. (2018); Jaksch et al. (2010); Ji and Li (2023); Jin et al. (2018); Kakade (2003); Kearns and Singh (1998b); Kolter and Ng (2009); Lattimore and Hutter (2012); Li et al. (2021a, 2024d,e, 2021c); Ménard et al. (2021); Neu and Pike-Burke (2020); Osband et al. (2013); Pacchiano et al. (2020); Russo (2019); Strehl et al. (2006); Strehl and Littman (2008); Szita and Szepesvári (2010); Tarbouriech et al. (2021); Wang et al. (2020); Xiong et al. (2022); Zanette and Brunskill (2019); Zhang et al. (2021a, 2022, 2020). Unfortunately, no work has been able to conquer this problem completely: the state-of-the-art result for model-based algorithms still incurs a burn-in that scales at least quadratically in S (Zhang et al., 2021a), while the burn-in cost of the best model-free algorithms (particularly with the aid of variance reduction introduced in Zhang et al. (2020)) still suffers from highly sub-optimal horizon dependency (Li et al., 2021a).

## 1.4 Notation

Before proceeding, let us introduce a set of notation to be used throughout. Let 1 and 0 indicate respectively the all-one vector and the all-zero vector. Let e s denote the s -th standard basis vector (which has 1 at the s -th coordinate and 0 otherwise). For any set X , ∆( X ) represents the set of probability distributions over the set X . For any positive integer N , we denote [ N ] = { 1 , . . . , N } . For any two vectors x, y with the same dimension, we use 〈 x, y 〉 (or x /latticetop y ) to denote the inner product of x and y . For any integer S &gt; 0, any probability vector p ∈ ∆([ S ]) and another vector v = [ v i ] 1 ≤ i ≤ S , we denote by the associated variance, where v 2 = [ v 2 i ] 1 ≤ i ≤ S represents element-wise square of v . For any two vectors a = [ a i ] 1 ≤ i ≤ n and b = [ b i ] 1 ≤ i ≤ n , the notation a ≥ b (resp. a ≤ b ) means a i ≥ b i (resp. a i ≤ b i ) holds

<!-- formula-not-decoded -->

simultaneously for all i . Without loss of generality, we assume throughout that K is a power of 2 to streamline presentation.

## 2 Problem formulation

In this section, we introduce the basics of tabular online RL, as well as some basic assumptions to be imposed throughout.

Basics of finite-horizon MDPs. This paper concentrates on time-inhomogeneous (or nonstationary) finite-horizon MDPs. Throughout the paper, we employ S = { 1 , . . . , S } to denote the state space, A = { 1 , . . . , A } the action space, and H the planning horizon. The notation P = { P h : S × A → ∆( S ) } 1 ≤ h ≤ H denotes the probability transition kernel of the MDP; for any current state s at any step h , if action a is taken, then the state at the next step h +1 of the environment is randomly drawn from P s,a,h := P h ( · | s, a ) ∈ ∆( S ). Also, the notation R = { R s,a,h ∈ ∆([0 , H ]) } 1 ≤ h ≤ H,s ∈S ,a ∈A indicates the reward distribution; that is, while executing action a in state s at step h , the agent receives an immediate reward - which is non-negative and possibly stochastic - drawn from the distribution R s,a,h . We shall also denote by r = { r h ( s, a ) } 1 ≤ h ≤ H,s ∈S ,a ∈A the mean reward function, so that r h ( s, a ) := E r ′ ∼ R s,a,h [ r ′ ] ∈ [0 , H ] for any ( s, a, h )-tuple. Additionally, a deterministic policy π = { π h : S → A} 1 ≤ h ≤ H stands for an action selection rule, so that the action selected in state s at step h is given by π h ( s ). The readers can consult standard textbooks (e.g., Bertsekas (2019)) for more extensive descriptions.

In each episode, a trajectory ( s 1 , a 1 , r ′ 1 , s 2 , . . . , s H , a H , r ′ H ) is rolled out as follows: the learner starts from an initial state s 1 independently drawn from some fixed (but unknown) distribution µ ∈ ∆( S ); for each step 1 ≤ h ≤ H , the learner takes action a h , gains an immediate reward r ′ h ∼ R s h ,a h ,h , and the environment transits to the state s h +1 at step h +1 according to P s h ,a h ,h . Note that both the reward and the state transition are independently drawn from their respective distributions, depending solely on the current state-action-step triple but not any previous outcomes. All of our results in this paper operate under the following assumption on the total reward.

Assumption 1. For any possible trajectory ( s 1 , a 1 , r ′ 1 , . . . , s H , a H , r ′ H ) , one always has 0 ≤ ∑ H h =1 r ′ h ≤ H .

As can be easily seen, Assumption 1 is less stringent than another common choice that assumes r ′ h ∈ [0 , 1] for any h in any episode. In particular, Assumption 1 allows for sparse and spiky rewards along an episode; more discussions can be found in (Jiang and Agarwal, 2018; Wang et al., 2020).

Value function and Q-function. For any given policy π , one can define the value function V π = { V π h : S → R } and the Q -function Q π = { Q π h : S × A → R } such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the expectation E π [ · ] is taken over the randomness of an episode { ( s h , a h , r ′ h ) } 1 ≤ h ≤ H generated under policy π , that is, a j = π j ( s j ) for every h ≤ j ≤ H (resp. h &lt; j ≤ H ) is chosen in the definition of V π h (resp. Q π h ). Accordingly, we define the optimal value function and the optimal Q -function respectively as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Throughout this paper, we shall often abuse the notation by letting both V π h and V /star h (resp. Q π h and Q /star h ) represent S -dimensional (resp. SA -dimensional) vectors containing all elements of the corresponding value

functions (resp. Q-functions). Two important properties are worth mentioning: (a) the optimal value and the optimal Q-function are linked by the Bellman equation:

<!-- formula-not-decoded -->

(b) there exists a deterministic policy, denoted by π /star , that achieves optimal value functions and Q-functions for all state-action-step tuples simultaneously, that is,

<!-- formula-not-decoded -->

Data collection protocol and performance metrics. During the learning process, the learner is allowed to collect K episodes of samples (using arbitrary policies it selects). More precisely, in the k -th episode, the learner is given an independently generated initial state s k 1 ∼ µ , and executes policy π k (chosen based on data collected in previous episodes) to obtain a sample trajectory { ( s k h , a k h , r k h ) } 1 ≤ h ≤ H , with s k h , a k h and r k h denoting the state, action and immediate reward at step h of this episode.

To evaluate the learning performance, a widely used metric is the (cumulative) regret over all K episodes:

<!-- formula-not-decoded -->

and our goal is to design an online RL algorithm that minimizes Regret ( K ) regardless of the allowable sample size K . It is also well-known (see, e.g., Jin et al. (2018)) that a regret bound can often be readily translated into a PAC sample complexity result, the latter of which counts the number of episodes needed to find an ε -optimal policy ̂ π in the sense that E s 1 ∼ µ [ V /star 1 ( s 1 ) -V ̂ π 1 ( s 1 ) ] ≤ ε . For instance, the reduction argument in Jin et al. (2018) reveals that: if an algorithm achieves Regret ( K ) ≤ f ( S, A, H ) K 1 -α for some function f and some parameter α ∈ (0 , 1), then by randomly selecting a policy from { π k } 1 ≤ k ≤ K as ̂ π one achieves E s 1 ∼ µ [ V /star 1 ( s 1 ) -V ̂ π 1 ( s 1 ) ] /lessorsimilar f ( S, A, H ) K -α , thus resulting in a sample complexity bound of ( f ( S,A,H ) ε ) 1 /α .

## 3 A model-based algorithm: Monotonic Value Propagation

In this section, we formally describe our algorithm: a simple variation of the model-based algorithm called Monotonic Value Propagation proposed by Zhang et al. (2021a). We present the full procedure in Algorithm 1, and point out several key ingredients.

- Optimistic updates using upper confidence bounds (UCB). The algorithm implements the optimism principle in the face of uncertainty by adopting the frequently used UCB-based framework (see, e.g., UCBVI by Azar et al. (2017)). More specifically, the learner calculates the optimistic Bellman equation backward (from h = H,.. . , 1): it first computes an empirical estimate ̂ P = { ̂ P h ∈ R SA × S } 1 ≤ h ≤ H of the transition probability kernel as well as an empirical estimate ̂ r = { ̂ r h ∈ R SA } 1 ≤ h ≤ H of the mean reward function, and then maintains upper estimates for the associated value function and Q-function using

<!-- formula-not-decoded -->

for all state-action pairs. Here, Q h (resp. V h ) indicates the running estimate for the Q-function (resp. value function), whereas b h ( s, a ) ≥ 0 is some suitably chosen bonus term that compensates for the uncertainty. The above opportunistic Q-estimate in turn allows one to obtain a policy estimate (via a simple greedy rule), which will then be executed to collect new data. The fact that we first estimate the model (i.e., the transition kernel and mean rewards) makes it a model-based approach. Noteworthily, the empirical model ( ̂ P, ̂ r ) shall be updated multiple times as new samples continue to arrive, and hence the updating rule (18) will be invoked multiple times as well.

## Algorithm 1: Monotonic Value Propagation ( MVP ) (Zhang et al., 2021a)

```
1 input: state space S , action space A , horizon H , total number of episodes K , confidence parameter δ , c 1 = 460 9 , c 2 = 2 √ 2, c 3 = 544 9 . 2 initialization: set δ ′ ← δ 200 SAH 2 K 2 , and for all ( s, a, s ′ , h ) ∈ S × A × S × [ H ], set θ h ( s, a ) ← 0, κ h ( s, a ) ← 0, N all h ( s, a, s ′ ) ← 0, N h ( s, a, s ′ ) ← 0, N h ( s, a ) ← 0, Q h ( s, a ) ← H , V h ( s ) ← H . 3 for k = 1 , 2 , . . . , K do 4 Set π k such that π k h ( s ) = arg max a Q h ( s, a ) for all s ∈ S and h ∈ [ H ]. /* policy update. */ 5 for h = 1 , 2 , ..., H do 6 Observe s k h , take action a k h = arg max a Q h ( s k h , a ), receive r k h , observe s k h +1 . /* sampling. */ 7 ( s, a, s ′ ) ← ( s k h , a k h , s k h +1 ). 8 Update N all h ( s, a ) ← N all h ( s, a ) + 1, N h ( s, a, s ′ ) ← N h ( s, a, s ′ ) + 1, θ h ( s, a ) ← θ h ( s, a ) + r k h , κ h ( s, a ) ← κ h ( s, a ) + ( r k h ) 2 . /* perform updates using data of this epoch. */ 9 if N all h ( s, a ) ∈ { 1 , 2 , . . . , 2 log 2 K } then 10 N h ( s, a ) ← ∑ ˜ s N h ( s, a, ˜ s ). // number of visits to ( s, a, h ) in this epoch. 11 ̂ r h ( s, a ) ← θ h ( s,a ) N h ( s,a ) . // empirical rewards of this epoch. 12 ̂ σ h ( s, a ) ← κ h ( s,a ) N h ( s,a ) . // empirical squared rewards of this epoch. 13 ̂ P s,a,h ( ˜ s ) ← N h ( s,a, ˜ s ) N h ( s,a ) for all ˜ s ∈ S . // empirical transition for this epoch. 14 Set TRIGGERED = TRUE, and θ h ( s, a ) ← 0, κ h ( s, a ) ← 0, N h ( s, a, ˜ s ) ← 0 for all ˜ s ∈ S . /* optimistic Q-estimation using empirical model of this epoch. */ 15 if TRIGGERED= TRUE then 16 Set TRIGGERED = FALSE, and V H +1 ( s ) ← 0 for all s ∈ S . 17 for h = H,H -1 , ..., 1 do 18 for ( s, a ) ∈ S × A do 19 b h ( s, a ) ← c 1 √ V ( ̂ P s,a,h , V h +1 ) log 1 δ ′ max { N h ( s, a ) , 1 } + c 2 √ ( ̂ σ h ( s, a ) -( ̂ r h ( s, a )) 2 ) log 1 δ ′ max { N h ( s, a ) , 1 } + c 3 H log 1 δ ′ max { N h ( s, a ) , 1 } , (19) Q h ( s, a ) ← min { ̂ r h ( s, a ) + 〈 ̂ P s,a,h , V h +1 〉 + b h ( s, a ) , H } , V h ( s ) ← max a Q h ( s, a ) . (20) 20
```

- An epoch-based procedure and a doubling trick. Compared to the original UCBVI (Azar et al., 2017), one distinguishing feature of MVP is to update the empirical transition kernel and empirical rewards in an epoch-based fashion, as motivated by a doubling update framework adopted in Jaksch et al. (2010). More concretely, the whole learning process is divided into consecutive epochs via a simple doubling rule; namely, whenever there exits a ( s, a, h )-tuple whose visitation count reaches a power of 2, we end the current epoch, reconstruct the empirical model (cf. lines 11 and 13 of Algorithm 1), compute the Q-function and value function using the newly updated transition kernel and rewards (cf. (20)), and then start a new epoch with an updated sampling policy. This stands in stark contrast with the original UCBVI , which computes new estimates for the transition model, Q-function and value function in every episode. With this doubling rule in place, the estimated transition probability vector for each ( s, a, h )-tuple will be updated by no more than log 2 K times, a feature that plays a pivotal role in significantly reducing some sort of covering number needed in our covering-based analysis (as we shall elaborate on shortly in Section 4). In each epoch, the learned policy is induced by the optimistic Qfunction estimate - computed based on the empirical transition kernel of the current epoch - which

will then be employed to collect samples in all episodes of the next epoch. More technical explanations of the doubling update rule will be provided in Section 4.2.

- Monotonic bonus functions. Another crucial step in order to ensure near-optimal regret lies in careful designs of the data-driven bonus terms { b h ( s, a ) } in (18a). Here, we adopt the monotonic Bernsteinstyle bonus function for MVP originally proposed in Zhang et al. (2021a), to be made precise in (19). Compared to the bonus function in Euler (Zanette and Brunskill, 2019) and UCBVI (Azar et al., 2017), the monotonic bonus form has a cleaner structure that effectively avoids large lower-order terms. Note that in order to enable variance-aware regret, we also need to keep track of the empirical variance of the (stochastic) immediate rewards.

Remark 1. We note that a doubling update rule has also been used in the original MVP (Zhang et al., 2021a). A subtle difference between our modified version and the original one lies in that: when the visitation count for some ( s, a, h ) reaches 2 i for some integer i ≥ 1 , we only use the second half of the samples (i.e., the { 2 i -1 + l } 2 i -1 l =1 -th samples) to compute the empirical model, whereas the original MVP makes use of all the 2 i samples. This modified step turns out to be helpful in our analysis, while still preserving sample efficiency in an orderwise sense (since the latest batch always contains at least half of the samples).

## 4 Key technical innovations

In this section, we point out the key technical hurdles the previous approach encounters when mitigating the burn-in cost, and put forward a new strategy to overcome such hurdles. For ease of presentation, let us introduce a set of augmented notation to indicate several running iterates in Algorithm 1, which makes clear the dependency on the episode number k and will be used throughout all of our analysis.

- ̂ P k s,a,h ∈ R S : the latest update of the empirical transition probability vector ̂ P s,a,h before the k -th episode.
- ̂ ̂ · b k h ( s, a ) ≥ 0: the latest update of the bonus term b h ( s, a ) before the k -th episode.
- ̂ r k h ( s, a ) ∈ [0 , H ]: the latest update of the empirical reward ̂ r h ( s, a ) before the k -th episode. · σ k h ( s, a ) ∈ [0 , H 2 ]: the latest update of the empirical squared reward σ h ( s, a ) before the k -th episode.
- N k, all h ( s, a ): the total visitation count of the ( s, a, h )-tuple before the beginning of the k -th episode.
- N k h ( s, a ): the visitation count N h ( s, a ) of the ( s, a, h )-tuple of the latest doubling batch used to compute ̂ P s,a,h before the k -th episode. When N k, all h ( s, a ) = 0, we define N k h ( s, a ) = 1 for ease of presentation. · V k h ∈ R S : the value function estimate V h before the beginning of the k -th episode.
- Q k h ∈ R SA : the Q-function estimate Q h before the beginning of the k -th episode.

Another notation for the empirical transition probability vector is also introduced below:

- For any j ≥ 2 (resp. j = 1), let ̂ P ( j ) s,a,h be the empirical transition probability vector for ( s, a, h ) computed using the j -th batch of data, i.e., the { 2 j -2 + i } 2 j -2 i =1 -th samples (resp. the 1st sample) for ( s, a, h ). For completeness, we take P (0) s,a,h = 1 S 1 for the 0-th batch.
- ̂ · Similarly, let ̂ r ( j ) h ( s, a ) (resp. ̂ σ ( j ) h ( s, a )) denote the empirical reward (resp. empirical squared reward) w.r.t. ( s, a, h ) based on the j -th batch of data.

## 4.1 Technical barriers in prior theory for UCBVI

Let us take a close inspection on prior regret analysis for UCB-based model-based algorithms, in order to illuminate the part that calls for novel analysis. To simplify presentation, this subsection assumes deterministic rewards so that each empirical reward is replaced by its mean.

Let us look at the original UCBVI algorithm proposed by Azar et al. (2017). Standard decomposition arguments employed in the literature (e.g., Azar et al. (2017); Jaksch et al. (2010); Zhang et al. (2021a)) decompose the regret as follows:

<!-- formula-not-decoded -->

see also the derivation in Section 5. Here, we abuse the notation by letting V k h +1 (resp. b k h ) be the value function estimate (resp. bonus term) of UCBVI before the k -th episode, and in the meantime, we let ̂ P k, all s,a,h represent the empirical transition probability for the ( s, a, h )-tuple computed using all samples before the k -th episode (note that we add the superscript all to differentiate it from its counterpart in our algorithm). In order to achieve full-range optimal regret, one needs to bound the three terms on the right-hand side of (21) carefully, among which two are easy to handle.

- It is known that the second term (i.e., the aggregate bonus) on the right-hand side of (21) can be controlled in a rate-optimal manner if we adopt suitably chosen Bernstein-style bonus; see, e.g., Zhang et al. (2021a), which will also be made clear shortly in Section 5.
- In the meantime, the third term on the right-hand side of (21) can be easily coped with by means of standard martingale concentration bounds (e.g., the Freedman inequality).

It then comes down to controlling the first term on the right-hand side of (21). This turns out to be the most challenging part, owing to the complicated statistical dependency between ̂ P k, all s k h ,a k h ,h and V k h +1 . To see this, note that ̂ P k, all s,a,h is constructed based on all previous samples of ( s, a, h ), which has non-negligible influences upon V k h +1 as V k h +1 is computed based on previous samples. At least two strategies have been proposed to circumvent this technical difficulty, which we take a moment to discuss.

- Strategy 1: replacing V k h +1 with V /star h +1 for large k . Most prior analysis for model-based algorithms (Azar et al., 2017; Dann et al., 2017; Zanette and Brunskill, 2019; Zhang et al., 2021a) decomposes

<!-- formula-not-decoded -->

The rationale behind this decomposition is as follows:

- (i) given that V /star h +1 is fixed and independent from the data, the first term on the right-hand side of (22) can be bounded easily using Freedman's inequality;
- (ii) the second term on the right-hand side of (22) would vanish as V k h +1 and V /star h +1 become exceedingly close (which would happen as k becomes large enough).

Such arguments, however, fall short of tightness when analyzing the initial stage of the learning process: given that V k h +1 -V /star h +1 cannot be sufficiently small at the beginning, this approach necessarily results in a huge burn-in cost.

- Strategy 2: a covering-based argument. Let us discuss informally another potential strategy that motivates our analysis. We first take a closer look at the relationship between ̂ P k, all s,a,h and V k h +1 . Abusing

notation by letting N k, all h ( s, a ) be the total number of visits to a ( s, a, h )-tuple before the k -th episode in UCBVI , we can easily observe that ̂ P k, all s,a,h and V k h +1 are statistically independent conditioned on the set { N k, all h ( s, a ) } ( s,a,k ) ∈S×A× [ K ] . Consequently, if we 'pretend' that { N k, all h ( s, a ) } are pre-fixed and independent of { ̂ P k, all s,a,h } , then one can invoke standard concentration inequalities to obtain a high-probability bound on ∑ k,h ( ̂ P k, all s k h ,a k h ,h -P s k h ,a k h ,h ) V k h +1 in a desired manner. The next step would then be to invoke a union bound over all possible configurations of { N k, all h ( s, a ) } , so as to eliminate the above independence assumption. The main drawback of this approach, however, is that there are exponentially many (e.g., in K ) possible choices of { N k, all h ( s, a ) } , inevitably loosening the regret bound.

## 4.2 Our approach

In light of the covering-based argument in Section 4.1, one can only hope this analysis strategy to work if substantial compression (i.e., a significantly reduced covering number) of the visitation counts is plausible. This motivates our introduction of the doubling batches as described in Section 3, so that for each ( s, a, h )-tuple, the empirical model ̂ P s,a,h and its associated visitation count N h ( s, a ) (for the associated batch) are updated at most log 2 K times (see line 9 of Algorithm 1). Compared to the original UCBVI that recomputes the transition model in every episode, our algorithm allows for significant reduction of the covering number of the visitation counts, thanks to its much less frequent updates.

Similar to (21), we are in need of bounding the following term when analyzing Algorithm 1:

<!-- formula-not-decoded -->

In what follows, we present our key ideas that enable tight analysis of this quantity, which constitute our main technical innovations. The complete regret analysis for Algorithm 1 is postponed to Section 5.

## 4.2.1 Key concept: profiles

One of the most important concepts underlying our analysis for Algorithm 1 is the so-called 'profile', defined below.

Definition 1 (Profile) . Consider any combination { N k, all h ( s, a ) } ( s,a,h,k ) ∈S×A× [ H ] × [ K ] . For any k ∈ [ K ] , define

<!-- formula-not-decoded -->

The profile for the k -th episode (1 ≤ k ≤ K ) and the total profile are then defined respectively as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Clearly, once a total profile I w.r.t. { N k, all h ( s, a ) } is given, one can write

<!-- formula-not-decoded -->

In order to quantify the degree of compression Definition 1 offers when representing the update times and locations, we provide an upper bound on the number of possible total profiles in the lemma below.

In other words, a total profile specifies all the time instances and locations when the empirical model is updated. Given that each N k h ( s, a ) is recomputed only when the associated empirical model is updated (see line 10 of Algorithm 1), the total profile also provides a succinct representation of the set { N k h ( s, a ) } .

Lemma 5. Suppose that K ≥ SAH log 2 K . Then the number of all possible total profiles w.r.t. Algorithm 1 is at most

<!-- formula-not-decoded -->

Proof. Define the following set (which will be useful in subsequent analysis as well)

<!-- formula-not-decoded -->

Due to the monotonicity constraints, it is easily seen that the total profile of any set { N k h ( s, a ) } must lie within C . It then boils down to proving that |C| ≤ (4 SAHK ) SAH log 2 K +1 , which can be accomplished via elementary combinatorial calculations. The complete proof is deferred to Appendix B.1.

In comparison to using { N k, all h ( s, a ) } to encode all update times and locations - which might have exponentially many (in K ) possibilities - the use of doubling batches in Algorithm 1 allows for remarkable compression (as the exponent of the number of possibilities only scales logarithmically in K ).

## 4.2.2 Decoupling the statistical dependency

An expanded view of randomness w.r.t. state transitions. To facilitate analysis, we find it helpful to look at a different yet closely related way to generate independent samples from a generative model.

Definition 2 (An expanded sample set from a generative model) . Let D expand be a set of SAHK independent samples generated as follows: for each ( s, a, h ) ∈ S × A × [ H ] , draw K independent samples ( s, a, h, s ′ , ( i ) ) obeying s ′ , ( i ) ind . ∼ P s,a,h ( 1 ≤ i ≤ K ).

Crucially, D expand can be viewed as an expansion of the original dataset - denoted by D original -collected in online learning, as we can couple the data collection processes of D original and D expand as follows:

- (i) generate D expand before the beginning of the online process;
- (ii) during the online learning process, whenever a sample needs to be drawn from ( s, a, h ), one can take an unused sample of ( s, a, h ) from D expand without replacement.

This allows one to conduct analysis alternatively based on the expanded sample set D expand , which is sometimes more convenient (as we shall detail momentarily). Unless otherwise noted, all analyses in our proof assume that D original and D expand are coupled through the above simulation process.

In the sequel, we let ̂ P ( j ) s,a,h (cf. the beginning of Section 4) denote the empirical probability vector based on the j -th batch of data from D original and D expand interchangeably, as long as it is clear from the context.

A starting point: a basic decomposition. We now describe our approach to tackling the complicated statistical dependency between ̂ P k s,a,h and V k h +1 . To begin with, from relation (25) we can write

<!-- formula-not-decoded -->

where I true = {I 1 , true , · · · , I K, true } with I k, true = { I k, true s,a,h } denotes the total profile w.r.t. the true visitation counts in the online learning process, k l,j,s,a,h denotes the episode index of the sample that visits ( s, a, h ) for the (2 l -1 + j )-th time in the online learning process, and we take V k h +1 = 0 for any k &gt; K . Here, the third line makes use of the fact that 0 ≤ V k h +1 ( s ) ≤ H for all s ∈ S . The decomposition (27) motivates us to first control the term ∑ s,a,h 〈 ̂ P ( l ) s,a,h -P s,a,h , V k l,j,s,a,h h +1 〉 , leading to the following 3-step analysis strategy.

- 1) For any given total profile I ∈ C and any fixed 1 ≤ l ≤ log 2 K , develop a high-probability bound on a weighted sum taking the following form

<!-- formula-not-decoded -->

- 2) Take the union bound over all possible I ∈ C -with the aid of Lemma 5 - to obtain a uniform control of the term (28), simultaneously accounting for all I ∈ C and all associated sequences { X h +1 ,s,a } .

where each vector X h +1 ,s,a is any deterministic function of I and the samples collected for steps h ′ ≥ h +1. Given the statistical independence between ̂ P ( l ) s,a,h and those samples for steps h ′ ≥ h +1 (in the view of D expand ), we can bound (28) using standard martingale concentration inequalities.

- 3) We then demonstrate that the above uniform bounds can be applied to the decomposition (27) to obtain a desired bound.

Main steps. We now carry out the above three steps.

Steps 1) and 2). Let us first specify the types of vectors { X h,s,a } mentioned above in (28). For each total profile I ∈ C (cf. (26)), consider any set { X h, I } 1 ≤ h ≤ H obeying: for each 1 ≤ h ≤ H ,

<!-- formula-not-decoded -->

- X h +1 , I is given by a deterministic function of I and
- ‖ X ‖ ∞ ≤ H for each vector X ∈ X h, I ;
- X h, I is a set of no more than K +1 non-negative vectors in R S , and contains the all-zero vector 0.

Lemma 6. Suppose that K ≥ SAH log 2 K , and construct a set { X h, I } 1 ≤ h ≤ H for each I ∈ C satisfying the above properties. Then with probability at least 1 -δ ′ ,

Given such a construction of { X h, I } , we can readily conduct Steps 1) and 2), with a uniform concentration bound stated below.

<!-- formula-not-decoded -->

holds simultaneously for all I ∈ C , all 2 ≤ l ≤ log 2 K +1 , and all sequences { X h,s,a } ( s,a,h ) ∈S×A× [ H ] obeying X h,s,a ∈ X h +1 , I , ∀ ( s, a, h ) ∈ S × A × [ H ] . Here, we recall that δ ′ = δ 200 SAH 2 K 2 .

Proof. We first invoke the Freedman inequality to bound the target quantity for any fixed I ∈ C , any fixed integer l , and any fixed feasible sequence { X h,s,a } , before applying the union bound to establish uniform control. See Appendix B.2 for details.

Step 3). Next, we turn to Step 3), which is accomplished via the following lemma. Note that we also provide upper bounds for two additional quantities: ∑ k,h max {〈 ̂ P k s k h ,a k h ,h -P s k h ,a k h ,h , V k h +1 〉 , 0 } and ∑ k,h 〈 ̂ P k s k h ,a k h ,h -P s k h ,a k h ,h , ( V k h +1 ) 2 〉 , which will be useful in subsequent analysis.

Lemma 7. Suppose that K ≥ SAH log 2 K . With probability exceeding 1 -δ ′ , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Proof. This result is proved by combining the uniform bound in Lemma 6 with the decomposition (27). See Appendix B.3.

Thus far, we have obtained high-probability bounds on the most challenging terms. The complete proof of Theorem 1 will be presented next in Section 5.

## 5 Proof of Theorem 1

This section is devoted to proving Theorem 1. For notational convenience, let B be a logarithmic term

<!-- formula-not-decoded -->

where we recall that δ is the confidence parameter in Algorithm 1 and δ ′ = δ 200 SAH 2 K 2 . When K ≤ BSAH , the claimed result in Theorem 1 holds trivially since

<!-- formula-not-decoded -->

As a result, it suffices to focus on the scenario with

<!-- formula-not-decoded -->

Our regret analysis for Algorithm 1 consists of several steps described below.

Step 1: the optimism principle. To begin with, we justify that the running estimates of Q-function and value function in Algorithm 1 are always upper bounds on the optimal Q-function and the optimal value function, respectively, thereby guaranteeing optimism in the face of uncertainty.

Lemma 8 (Optimism) . With probability exceeding 1 -4 SAHKδ ′ , one has

<!-- formula-not-decoded -->

for all ( s, a, h, k ) .

Proof. See Appendix C.1.

Step 2: regret decomposition. In view of the optimism shown in Lemma 8, the regret can be upper bounded by

<!-- formula-not-decoded -->

with probability at least 1 -4 SAHKδ ′ . In order to control the right-hand side of (33), we first make note of the following upper bound on V k 1 ( s k 1 ).

Lemma 9. For every 1 ≤ k ≤ K , one has

<!-- formula-not-decoded -->

Proof of Lemma 9. From the construction of V k h and Q k h , it is seen that, for each 1 ≤ h ≤ H ,

<!-- formula-not-decoded -->

Applying this relation recursively over 1 ≤ h ≤ H gives

<!-- formula-not-decoded -->

which combined with V k H +1 = 0 concludes the proof.

Combine Lemma 9 with (33) to show that, with probability at least 1 -4 SAHKδ ′ ,

<!-- formula-not-decoded -->

leaving us with four terms to control. In particular, T 1 has already been upper bounded in Section 4.2, and hence we shall describe how to bound T 2 , . . . , T 4 in the sequel.

Step 3.1: bounding the terms T 2 , T 3 and T 4 . In this section, we seek to bound the terms T 2 , T 3 and T 4 defined in the regret decomposition (34). To do so, we find it helpful to first introduce the following quantities that capture some sort of aggregate variances:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with T 5 denoting certain empirical variance and T 6 the true variance. With these quantities in place, we claim that the following bounds hold true.

Lemma 10. With probability exceeding 1 -15 SAH 2 K 2 δ ′ , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. See Appendix C.2.

Step 3.2: bounding the aggregate variances T 5 and T 6 . The previous bounds on T 2 and T 3 stated in Lemma 10 depend respectively on the aggregate variance T 5 and T 6 (cf. (35a) and (35b)), which we would like to control now. By introducing the following quantities:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we can upper bound T 5 and T 6 through the following lemma.

Lemma 11. With probability at least 1 -4 SAHKδ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. See Appendix C.3.

Step 3.3: bounding the terms T 1 , T 7 and T 9 . Taking a look at the above bounds on T 2 , . . . , T 6 , we see that one still needs to deal with the terms T 1 , T 7 and T 9 (see (34), (37a) and (37c), respectively). As

it turns out, these quantities have already been bounded in Section 4. Specifically, Lemma 7 tells us that: with probability at least 1 -δ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we recall that B = 4000(log 2 K ) 3 log(3 SAH ) log 1 δ ′ .

Step 4: putting all pieces together. The previous bounds (36), (38) and (39) indicate that: with probability at least 1 -100 SAH 2 K 2 δ ′ , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we again use B = 4000(log 2 K ) 3 log(3 SAH ) log 1 δ ′ .

<!-- formula-not-decoded -->

To solve the inequalities (40), we resort to the elementary AM-GM inequality: if a ≤ √ bc + d for some b, c ≥ 0, then it follows that a ≤ /epsilon1b + 1 2 /epsilon1 c + d for any /epsilon1 &gt; 0. This basic inequality combined with (40) gives which in turn result in

<!-- formula-not-decoded -->

By taking /epsilon1 = 1 / 20, we arrive at

<!-- formula-not-decoded -->

where the last relation holds due to our assumption K ≥ SAHB (cf. (31)). Substituting this into (40) yields

<!-- formula-not-decoded -->

provided that K ≥ SAHB . These bounds taken collectively with (34) readily give

<!-- formula-not-decoded -->

Combining the two scenarios (i.e., K ≥ BSAH and K ≤ BSAH ) reveals that with probability at least 1 -100 SAH 2 K 2 δ ′ ,

The proof of Theorem 1 is thus completed by recalling that δ ′ = δ 200 SAH 2 K 2 .

<!-- formula-not-decoded -->

## 6 Extensions

In this section, we develop more refined regret bounds for Algorithm 1 in order to reflect the role of several problem-dependent quantities. Detailed proofs are postponed to Appendix D and Appendix F.

## 6.1 Value-based regret bounds

Thus far, we have not yet introduced the crucial quantity v /star in Theorem 2, which we define now. When the initial states are drawn from µ , we define v /star to be the weighted optimal value:

<!-- formula-not-decoded -->

Encouragingly, the value-dependent regret bound we develop in Theorem 2 is still minimax-optimal, as asserted by the following lower bound.

Theorem 12. Consider any p ∈ [0 , 1] and K ≥ 1 . For any learning algorithm, there exists an MDP with S states, A actions and horizon H obeying v /star ≤ Hp and

In fact, the construction of the hard instance (as in the proof of Theorem 12) is quite simple. Design a new branch with 0 reward and set the probability of reaching this branch to be 1 -p . Also, with probability p , we direct the learner to a hard instance with regret Ω(min { √ SAH 3 Kp,KpH } ) and optimal value H . This guarantees that the optimal value obeys v /star ≤ Hp and that the expected regret is at least

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

See Appendix G for more details.

## 6.2 Cost-based regret bounds

Next, we turn to the cost-aware regret bound as in Theorem 3. Note that all other results except for Theorem 3 (and a lower bound in this subsection) are about rewards as opposed to cost. In order to facilitate discussion, let us first formally formulate the cost-based scenarios.

Suppose that the reward distributions { R h,s,a } ( s,a,h ) are replaced with the cost distributions { C h,s,a } ( s,a,h ) , where each distribution C h,s,a ∈ ∆([0 , H ]) has mean c h ( s, a ). In the h -th step of an episode, the learner pays an immediate cost c h ∼ C h,s h ,a h instead of receiving an immediate reward r h , and the objective of the

learner is instead to minimize the total cost ∑ H h =1 c h (in an expected sense). The optimal cost quantity c /star is then defined as

In this cost-based setting, we find it convenient to re-define the Q -function and value function as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we adopt different fonts to differentiate them from the original Q-function and value function. The optimal cost function is then define by

<!-- formula-not-decoded -->

Given the definitions above, we overload the notation Regret ( K ) to denote the regret for the cost-based scenario as

One can also simply regard the cost minimization problem as reward maximization with negative rewards by choosing r h = -c h . This way allows us to apply Algorithm 1 directly, except that (20) is replaced by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To confirm the tightness of Theorem 3, we develop the following matching lower bound, which resorts to a similar hard instance as in the proof of Theorem 12.

Theorem 13. Consider any p ∈ [0 , 1 / 4] and any K ≥ 1 . For any algorithm, one can construct an MDP with S states, A actions and horizon H obeying c /star /equivasymptotic Hp and

The proof of this lower bound can be found in Appendix G.2.

<!-- formula-not-decoded -->

## 6.3 Variance-dependent regret bound

The final regret bound presented in Theorem 4 depends on some sort of variance metrics. Towards this end, let us first make precise the variance metrics of interest:

- (i) The first variance metric is defined as

<!-- formula-not-decoded -->

where { ( s h , a h ) } 1 ≤ h ≤ H represents a sample trajectory under policy π . This captures the maximal possible expected sum of variance with respect to the optimal value function { V /star h } H h =1 .

- (ii) Another useful variance metric is defined as

<!-- formula-not-decoded -->

where { r h } 1 ≤ h ≤ H denotes a sample sequence of immediate rewards under policy π . This indicates the maximal possible variance of the accumulative reward.

The interested reader is referred to Zhou et al. (2023) for further discussion about these two metrics. Our final variance metric is then defined as

<!-- formula-not-decoded -->

With the above variance metrics in mind, we can then revisit Theorem 4. As a special case, when the transition model is fully deterministic, the regret bound in Theorem 4 simplifies to

Regret ( K ) ≤ ˜ O ( min { SAH 2 , HK }) for any K ≥ 1, which is roughly the cost of visiting each state-action pair. The full proof of Theorem 4 is postponed to Appenndix F.

To finish up, let us develop a matching lower bound to corroborate the tightness and optimality of Theorem 4.

Theorem 14. Consider any p ∈ [0 , 1] and any K ≥ 1 . For any algorithm, one can find an MDP instance with S states, A actions, and horizon H satisfying max { var 1 , var 2 } ≤ H 2 p and

The proof of Theorem 14 resembles that of Theorem 12, except that we need to construct a hard instance when K ≤ SAH/p . For this purpose, we construct a fully deterministic MDP (i.e., all of its transitions are deterministic and all rewards are fixed), and show that the learner has to visit about half of the state-actionlayer tuples in order to learn a near-optimal policy. The proof details are deferred to Appendix G.

## 7 Discussion

Focusing on tabular online RL in time-inhomogeneous finite-horizon MDPs, this paper has established the minimax-optimal regret (resp. sample complexity) - up to log factors - for the entire range of sample size K ≥ 1 (resp. target accuracy level ε ∈ (0 , H ]), thereby fully settling an open problem at the core of recent RL theory. The MVP algorithm studied herein is model-based in nature. Remarkably, the model-based approach remains the only family of algorithms that is capable of obtaining minimax optimality without burn-ins, regardless of the data collection mechanism in use (e.g., online RL, offline RL, and the simulator setting). We have further unlocked the optimality of this algorithm in a more refined manner, making apparent the effect of several problem-dependent quantities (e.g., optimal value/cost, variance statistics) upon the fundamental performance limits. The new analysis and algorithmic techniques put forward herein might shed important light on how to conquer other RL settings as well.

Moving forward, there are multiple directions that anticipate further theoretical pursuit. To begin with, is it possible to develop a model-free algorithm - which often exhibits more favorable memory complexity compared to the model-based counterpart - that achieves full-range minimax optimality? As alluded to previously, existing paradigms that rely on reference-advantage decomposition (or variance reduction) seem to incur a high burn-in cost (Li et al., 2021a; Zhang et al., 2020), thus calling for new ideas to overcome this barrier. Additionally, multiple other tabular settings (e.g., time-homogeneous finite-horizon MDPs, discounted infinite-horizon MDPs) have also suffered from similar issues regarding the burn-in requirements (Ji and Li, 2023; Zhang et al., 2021a). Take time-homogeneous finite-horizon MDPs for example: in order to achieve optimal sample efficiency, one needs to carefully deal with the statistical dependency incurred by aggregating data from across different time steps to estimate the same transition matrix (due to the homogeneous nature of P ), which results in more intricate issues than the time-homogeneous counterpart. We believe that resolving these two open problems will greatly enhance our theoretical understanding about online RL and beyond.

## Acknowledgement

We thank for Qiwen Cui for helpful discussions. Y. Chen is supported in part by the Alfred P. Sloan Research Fellowship, the Google Research Scholar Award, the AFOSR grant FA9550-22-1-0198, the ONR

<!-- formula-not-decoded -->

grant N00014-22-1-2354, and the NSF grants CCF-2221009 and CCF-1907661. JDL acknowledges support of Open Philanthropy, NSF IIS 2107304, NSF CCF 2212262, ONR Young Investigator Award, NSF CAREER Award 2144994, and NSF CCF 2019844. SSD acknowledges the support of NSF IIS 2110170, NSF DMS 2134106, NSF CCF 2212261, NSF IIS 2143493, NSF CCF 2019844, NSF IIS 2229881, and the Sloan Research Fellowship.

## A Preliminary facts

In this section, we gather several useful results that prove useful in our analysis. We use 1 {E} to denote the indicator of the event E . The first result below is a user-friendly version of the celebrated Freedman inequality (Freedman, 1975), a martingale counterpart to the Bernstein inequality. See Zhang et al. (2021b, Lemma 11) for the proof.

Lemma 15 (Freedman's inequality) . Let ( M n ) n ≥ 0 be a martingale such that M 0 = 0 and | M n -M n -1 | ≤ c ( ∀ n ≥ 1) hold for some quantity c &gt; 0 . Define Var n := ∑ n k =1 E [ ( M k -M k -1 ) 2 | F k -1 ] for every n ≥ 0 , where F k is the σ -algebra generated by ( M 1 , ..., M k ) . Then for any integer n ≥ 1 and any /epsilon1,δ &gt; 0 , one has

<!-- formula-not-decoded -->

Next, letting Var ( X ) represent the variance of X , we record a basic inequality connecting Var ( X 2 ) with Var ( X ) for any bounded random variable X .

Lemma 16 (Lemma 30 in (Chen et al., 2021)) . Let X be a random variable, and denote by C max the largest possible value of X . Then we have Var ( X 2 ) ≤ 4 C 2 max Var ( X ) .

Now, we turn to an intimate connection between the sum of a sequence of bounded non-negative random variables and the sum of their associated conditional random variables (with each random variable conditioned on the past), which is a consequence of basic properties about supermartingales.

Lemma 17 (Lemma 10 in (Zhang et al., 2022)) . Let X 1 , X 2 , . . . be a sequence of random variables taking value in [0 , l ] . For any k ≥ 1 , let F k be the σ -algebra generated by ( X 1 , X 2 , . . . , X k ) , and define Y k := E [ X k | F k -1 ] . Then for any δ &gt; 0 , we have

<!-- formula-not-decoded -->

The next two lemmas are concerned with concentration inequalities for the sum of i.i.d. bounded random variables: the first one is a version of the Bennet inequality, and the second one is an empirical Bernstein inequality (which replaces the variance in the standard Bernstein inequality with the empirical variance).

Lemma 18 (Bennet's inequality) . Let Z, Z 1 , ..., Z n be i.i.d. random variables with values in [0 , 1] and let δ &gt; 0 . Define V Z = E [ ( Z -E Z ) 2 ] . Then one has

∣ ∣ Lemma 19 (Theorem 4 in Maurer and Pontil (2009)) . Consider any δ &gt; 0 and any integer n ≥ 2 . Let Z, Z 1 , ..., Z n be a collection of i.i.d. random variables falling within [0 , 1] . Define the empirical mean Z := 1 n ∑ n i =1 Z i and empirical variance V n := 1 n ∑ n i =1 ( Z i -Z ) 2 . Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, we record a simple fact concerning the visitation counts { N k h ( s k h , a k h ) } .

Lemma 20. Recall the definition of N k h ( s k h , a k h ) in Algorithm 1. It holds that

<!-- formula-not-decoded -->

Proof. In view of the doubling batch update rule, it is easily seen that: for any given ( s, a, h ),

<!-- formula-not-decoded -->

since each ( s, a, h ) is associated with at most log 2 K epochs. Summing over ( s, a, h ) completes the proof.

As it turns out, Lemma 20 together with the Freedman inequality allows one to control the difference between the empirical rewards and the true mean rewards, as stated below.

Lemma 21. With probability exceeding 1 -2 SAHKδ ′ , it holds that

As an immediate consequence of Lemma 21 and the basic fact ∑ k,h r h ( s k h , a k h ) ≤ KH , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability exceeding 1 -2 SAHKδ ′ , where the last inequality holds true under the assumption 31.

Proof of Lemma 21. In view of Lemma 19 and the union bound, with probability 1 -2 SAHKδ ′ we have simultaneously for all ( s, a, h, k ) obeying N k h ( s, a ) &gt; 2, where we take advantage of the basic fact ̂ σ k h ( s k h , a k h ) ≤ H ̂ r k h ( s, a ) (since each immediate reward is upper bounded by H ). Solve the inequality above to obtain

<!-- formula-not-decoded -->

It is then seen that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the second inequality arises from Cauchy-Schwarz, whereas the term 4 SAH 2 accounts for those state-action pairs with N k h ( s, a ) ≤ 2 (since there are at most 2 SAH such occurances and it holds that ∣ r k h ( s k h , a k h ) -r h ( s k h , a k h ) ∣ ≤ 2 H ). This together with Lemma 20 then leads to

Moreover, the AM-GM inequality implies that

<!-- formula-not-decoded -->

thus concluding the proof.

## B Proofs of key lemmas in Section 4

## B.1 Proof of Lemma 5

It suffices to develop an upper bound on the cardinality of C (cf. (26)). Setting

<!-- formula-not-decoded -->

we find it helpful to introduce the following useful sets:

<!-- formula-not-decoded -->

/negationslash

In words, C distinct ( l ) can be viewed as the set of non-decreasing lengthl paths in { 0 , 1 , · · · , M } N , with all points on a path being distinct; C distinct thus consists of all such paths regardless of the length.

<!-- formula-not-decoded -->

We first establish a connection between |C| and ∣ ∣ C distinct ∣ ∣ . Define the operator Proj : C → C distinct that maps each I ∈ C to I distinct ∈ C distinct , where I distinct is composed of all distinct elements in I (in other words, this operator simply removes redundancy in I ). Let us looking at the following set for each I distinct ∈ C distinct . Since I distinct is a non-decreasing path with all its points being distinct, there are at most MN +1 elements in each I distinct . Hence, the size of B ( I distinct ) is at most the number of solutions to the following equations

<!-- formula-not-decoded -->

Elementary combinatorial arguments then reveal that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for each I distinct , provided that K ≥ MN = SAH log 2 K . We then arrive at

Everything then boils down to bounding |C distinct | . To do so, let us first look at the set C distinct ( MN +1), as each path in C distinct cannot have length more than MN +1. For each I distinct = { ˜ I 1 , ˜ I 2 , . . . , ˜ I MN +1 } ∈ C distinct ( MN +1), it is easily seen that

· For each 1 ≤ τ ≤ MN , ˜ I τ and ˜ I τ +1 differ only in one element (i.e., their Hamming distance is 1). In other words, we can view I distinct as an MN -step path from [0 , 0 , . . . , 0] /latticetop to [ M,M,...,M ] /latticetop , with each step moving in one dimension. Clearly, each step has at most N directions to choose from, meaning that there are at most N MN such paths. This implies that

<!-- formula-not-decoded -->

∣ ∣ C distinct ( MN +1) ∣ ∣ ≤ N MN . To finish up, we further observe that for each I distinct ∈ C distinct , there exists some ˜ I distinct ∈ C distinct ( MN +1) such that I distinct ⊆ ˜ I distinct . This observation together with basic combinatorial arguments indicates that which taken collectively with (56) leads to the advertised bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.2 Proof of Lemma 6

Let us begin by considering any fixed total profile I ∈ C , any fixed integer l obeying 2 ≤ l ≤ log 2 K + 1, and any given feasible sequence { X h,s,a } ( s,a,h ) ∈S×A× [ H ] . Recall that (i) ̂ P ( l ) s,a,h is computed based on the l -th batch of data comprising 2 l -2 independent samples from D expand (see Definition 2); and (ii) each X h +1 ,s,a is given by a deterministic function of I and the empirical models for steps h ′ ∈ [ h +1 , H ]. Consequently, Lemma 15 together with Definition 2 tells us that: with probability at least 1 -δ ′ , one has where we view the left-hand side of (57) as a martingale sequence from h = H back to h = 1.

<!-- formula-not-decoded -->

Moreover, given that each X h,s,a has at most K +1different choices (since we assume |X h, I | ≤ K +1), there are no more than ( K +1) SAH ≤ (2 K ) SAH possible choices of the feasible sequence { X h,s,a } ( s,a,h ) ∈S×A× [ H ] . In addition, it has been shown in Lemma 5 that there are no more than (4 SAHK ) 2 SAH log 2 K possibilities of the total profile I . Taking the union bound over all these choices and replacing δ ′ in (57) with δ ′ / ( (4 SAHK ) 2 SAH log 2 K (2 K ) SAH log 2 K ) , we can demonstrate that with probability at least 1 -δ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, recalling our assumption 0 ∈ X h +1 , I , we see that for every total profile I and its associated feasible sequence { X h,s,a } , holds simultaneously for all I ∈ C , all 2 ≤ l ≤ log 2 K +1, and all feasible sequences { X h,s,a } ( s,a,h ) ∈S×A× [ H ] .

<!-- formula-not-decoded -->

## B.3 Proof of Lemma 7

holds true. Consequently, the uniform upper bound on the right-hand side of (58) continues to be a valid upper bound on ∑ s,a,h max {〈 ̂ P ( l ) s,a,h -P s,a,h , X h +1 ,s,a 〉 , 0 } . This concludes the proof.

We begin by making the following claim, which we shall establish towards the end of this subsection. Claim 22. With probability exceeding 1 -δ ′ , holds simultaneously for all l = 1 , . . . , log 2 K and all j = 1 , . . . , 2 l -1 , where k l,j,s,a,h stands for the episode index of the sample that visits ( s, a, h ) for the (2 l -1 + j ) -th time in the online learning process.

<!-- formula-not-decoded -->

Assuming the validity of Claim 22 for the moment, we can combine this claim with the decomposition (27) and applying the Cauchy-Schwarz inequality to reach

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the last inequality is valid due to our assumption V k h +1 = 0 ( ∀ k &gt; K ) and the identity

<!-- formula-not-decoded -->

Before proceeding to the proof of Claim 22, we note that the other two quantities ∑ k,h max {〈 ̂ P k s k h ,a k h ,h -P s k h ,a k h ,h , V k h +1 〉 , 0 } and ∑ k,h 〈 ̂ P k s k h ,a k h ,h -P s k h ,a k h ,h , ( V k h +1 ) 2 〉 can be upper bounded using exactly the same arguments, which we omit for the sake of brevity. In particular, the latter quantity further satisfies

This establishes our advertised bound on ∑ k,h 〈 ̂ P k s k h ,a k h ,h -P s k h ,a k h ,h , V k h +1 〉 , provided that Claim 22 is valid.

<!-- formula-not-decoded -->

where the last inequality follows from Lemma 16 and the fact that 0 ≤ V k h +1 ( s ) ≤ H for all s ∈ S .

Proof of Claim 22. To invoke Lemma 7 to prove this claim, we need to choose the set {X h, I } properly to include the true value function estimates { V k h } . To do so, we find it helpful to first introduce an auxiliary algorithm tailored to each total profile. Specifically, for each I ∈ C (cf. (26)), consider the following updates operating upon the expanded sample set D expand .

<!-- formula-not-decoded -->

If we construct then it can be easily seen that {X h, I } satisfies the properties stated right before Lemma 6. As a consequence, applying Lemma 6 yields

<!-- formula-not-decoded -->

simultaneously for all l = 1 , . . . , log 2 K , all I ∈ C , and all sequences { X h,s,a } obeying X h,s,a ∈ X h, I , ∀ ( s, a, h ).

<!-- formula-not-decoded -->

To finish up, denote by I true the true total profile resulting from the online learning process. Given the way we couple D expand and D original (see the beginning of Section 4.2.2), we can easily see that the true value function estimate { V k h } obeys

<!-- formula-not-decoded -->

The claimed result then follows immediately from (62) and the uniform bound (61).

## C Proofs of auxiliary lemmas in Section 5

## C.1 Proof of Lemma 8

To begin with, we find it helpful to define the following function

<!-- formula-not-decoded -->

for any vector p ∈ ∆ S , any non-negative vector v ∈ R S obeying ‖ v ‖ ∞ ≤ H , and any positive integer n . We claim that

<!-- formula-not-decoded -->

To justify this claim, consider any 1 ≤ s ≤ S , and let us freeze p , n and all but the s -th entries of v . It then suffices to observe that (i) f is a continuous function, and (ii) except for at most two possible choices of v ( s ) that obey 20 3 √ V ( p,v ) log 1 δ ′ n = 400 9 H log 1 δ ′ n , one can use the properties of p and v to calculate

<!-- formula-not-decoded -->

thus establishing the claim (63).

We now proceed to the proof of Lemma 8. Consider any ( h, k, s, a ), and we divide into two cases.

Case 1: N k h ( s, a ) ≤ 2 . In this case, the following trivial bounds arise directly from the update rule (19):

<!-- formula-not-decoded -->

Case 2: N k h ( s, a ) &gt; 2 . Suppose now that Q k h +1 ≥ Q /star h +1 , which also implies that V k h +1 ≥ V /star h +1 . If Q k h ( s, a ) = H , then Q k h ( s, a ) ≥ Q /star h ( s, a ) holds trivially, and hence it suffices to look at the case with Q k h ( s, a ) &lt; H . According to the update rule in (19), it holds that

<!-- formula-not-decoded -->

for any ( s, a ), where the last inequality results from the claim (63) and the property V k h +1 ≥ V /star h +1 . Moreover, applying Lemma 19 and recalling the definition of σ k h ( s, a ), we have and

<!-- formula-not-decoded -->

These two inequalities imply that with probability exceeding 1 -4 δ ′ ,

<!-- formula-not-decoded -->

Substitution into (64) gives: with probability at least 1 -4 δ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Putting all this together. With the above two cases in place, one can invoke standard induction arguments to deduce that: with probability at least 1 -4 SAHKδ ′ , one has Q k h ( s, a ) ≥ Q /star h ( s, a ) and V k h = max a Q k h ( s, a ) ≥ max a Q /star h ( s, a ) = V /star h ( s ) for every ( s, a, h, k ). The proof is thus completed.

## C.2 Proof of Lemma 10

## C.2.1 Bounding T 2

We first establish the bound (36a) on T 2 . To begin with, T 2 can be decomposed using the definition (19) of the bonus term:

<!-- formula-not-decoded -->

Applying the Cauchy-Schwarz inequality and invoking Lemma 20, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the basic fact ̂ σ k h ( s k h , a k h ) ≤ H ̂ r k h ( s, a ) (since each immediate reward is at most H ) and the definition (35a) of T 5 , we can continue the bound in (67) to derive

Applying Lemma 21 to bound ∑ k,h ̂ r k h ( s k h , a k h ) and using the basic fact ∑ k,h r h ( s k h , a k h ) ≤ KH , we can employ a little algebra to deduce that with probability exceeding 1 -2 SAHKδ ′ .

<!-- formula-not-decoded -->

## C.2.2 Bounding T 3

Next, let us prove the bound (36b) on | T 3 | . Recall that V k h +1 ( s ) denotes the value function estimate of state s before the k -th episode, which corresponds to the value estimate computed at the end of the previous

epoch . This important fact implies that conditional on ( s k h , a k h ), the vector e s k h +1 is statistically independent of V k h +1 and has conditional mean P s k h ,a k h ,h , allowing us to invoke the Freedman inequality for martingales (see Lemma 15) to control the sum of 〈 P s k h ,a k h ,h -e s k h +1 , V k h +1 〉 . Recalling the definition of T 6 in (35b), we can see from Lemma 15 that

<!-- formula-not-decoded -->

with probability at least 1 -10 SAH 2 K 2 δ ′ .

## C.2.3 Bounding T 4

We now turn attention to the bound (36c) on | T 4 | . Recall that and we shall bound the two terms above separately.

<!-- formula-not-decoded -->

- Regarding the first term on the right-hand side of (70), we can apply Lemma 21 and the fact ∑ k,h r h ( s k h , a k h ) ≤ KH to show that

holds with probability at least 1 -2 SAHKδ ′ .

<!-- formula-not-decoded -->

- With regards to the second term on the right-hand side of (70), we note that conditional on π k , E k := ∑ H h =1 r h ( s k h , a k h ) -V π k 1 ( s k 1 ) is a zero-mean random variable bounded in magnitude by H . According to Lemma 15,

<!-- formula-not-decoded -->

holds with probability exceeding 1 -4 δ ′ log 2 ( KH ), where Var ( E k ) denotes the variance of E k conditioned on what happens before the k -th episode, and the last inequality follows since | E k | ≤ H always holds.

Substituting (71) and (72) into (70) reveals that with probability at least 1 -3 SAHKδ ′ ,

<!-- formula-not-decoded -->

## C.3 Proof of Lemma 11

Regarding the term T 5 , direct calculation gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -3 δ ′ log( KH 3 ). Here, the third line utilizes the fact that V k H +1 = 0, the first inequality holds since

<!-- formula-not-decoded -->

the penultimate line makes use of the property V k h ( s k h ) = Q k h ( s k h , a k h ) and the update rule (20), whereas the last line applies property (52) and the definition (36a) of T 2 .

Akin to the above bound on T 5 , we can show that with probability at least 1 -3 SAHKδ ′ ,

<!-- formula-not-decoded -->

Finally, note that the above bounds on T 5 and T 6 both depend on the term T 8 (cf. (37b)), which we would like to cope with now. Using Freedman's inequality (cf. Lemma 15) and the fact that Var ( X 2 ) ≤ 4 H 2 Var ( X ) for any random variable X with support on [ -H,H ] (cf. Lemma 16), we reach with probability at least 1 -3 δ ′ log( KH 3 ). Substitution into (75) and (77) establishes (38).

<!-- formula-not-decoded -->

## D Proof of the value-based regret bound (proof of Theorem 2)

Recall that

<!-- formula-not-decoded -->

Consider first the scenario where K ≤ BSAH 2 v /star : the regret bound can be upper bounded by

As a result, the remainder of the proof is dedicated to the the case with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To begin with, recall that the proof of Theorem 1 in Section 5 consists of bounding the quantities T 1 , . . . , T 9 (see (34), (35) and (37)) and recall that δ ′ = δ 200 SAH 2 K 2 . In order to establish Theorem 2, we need to develop tighter bounds on some of these quantities (i.e., T 2 , T 4 , T 5 and T 6 ) to reflect their dependency on v /star (cf. (43)).

Bounding T 2 . Recall that we have shown in (68) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In view of the definition of T 4 (cf. (34)) as well as the fact that ∑ K k =1 V /star 1 ( s k 1 ) ≤ 3 Kv /star + H log 1 δ ′ holds with probability at least 1 -δ ′ (see Lemma 17), we arrive at which in turn gives

<!-- formula-not-decoded -->

Bounding T 4 . When it comes to the quantity T 4 (cf. (34)), we make the observation that

<!-- formula-not-decoded -->

Repeating the arguments for (82) yields

<!-- formula-not-decoded -->

with probability at least 1 -δ ′ . Combining this with Lemma 21, we see that

<!-- formula-not-decoded -->

with probability exceeding 1 -3 SAHKδ ′ . In addition, Lemma 15 tells us that

<!-- formula-not-decoded -->

with probability at least 1 -2 SAHKδ ′ , where the expectation operator E π k ,s 1 ∼ µ [ · ] is taken over the randomness of a trajectory { ( s h , a h ) } generated under policy π k and initial state s 1 ∼ µ , the last line arises from the AM-GM inequality, and the penultimate line makes use of Assumption 1 and the fact that

<!-- formula-not-decoded -->

Taking (86), (87) and (88) together, we can demonstrate that with probability exceeding 1 -5 SAHKδ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substitution into (84) reveals that: with probability exceeding 1 -5 SAHKδ ′ ,

<!-- formula-not-decoded -->

Bounding T 5 . Recall that we have proven in (74) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With (85) and (88) in place, we can deduce that, with probability at least 1 -3 SAHKδ ′ ,

Moreover, under the assumption (81), we can further bound (89a) as

<!-- formula-not-decoded -->

with probability exceeding 1 -3 SAHKδ ′ , which combined with (92) and the assumption (81) results in

Substitution into (91) indicates that: with probability exceeding 1 -6 SAHKδ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Bounding T 6 . Making use of our bounds (76), (38c) and (93), we can readily derive

<!-- formula-not-decoded -->

with probability at least 1 -16 SAH 2 K 2 δ ′ .

Putting all pieces together. Recalling our choice of B (cf. (79)), we can see from (83), (36b), (90), (94), (95), (38c), (39a) and (39b) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Solving (96) under the assumption K ≥ BSAH 2 v /star allows us to demonstrate that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability exceeding 1 -200 SAH 2 K 2 δ ′ . Putting these bounds together with (34), we arrive at

<!-- formula-not-decoded -->

with probability exceeding 1 -200 SAH 2 K 2 δ ′ . Replacing δ ′ with δ 200 SAH 2 K 2 and taking δ = 1 2 KH give provided that K ≥ BSAH 2 v /star . Taking this collectively with (80) concludes the proof.

<!-- formula-not-decoded -->

## E Proof of the cost-based regret bound (proof of Theorem 3)

We now turn to the proof of Theorem 3. For notational convenience, we shall use r to denote the negative cost (namely, r h = -c h , ̂ r h = -̂ c h , and so on) throughout this section. We shall also use the following notation (and similar quantities like Q k h , V k h , . . . )

<!-- formula-not-decoded -->

in order to be consistent with the reward-based setting.

Akin to the proof of Theorem 2, we need to bound the quantities T 1 , . . . , T 9 introduced previously (see (34), (35) and (37)). We note that the analysis for T 1 , T 3 , T 7 , T 8 and T 9 in Appendix D readily applies to the negative reward case herein. Thus, it suffices to develop bounds on T 2 , T 4 , T 5 and T 6 to capture their dependency on c /star , which forms the main content of the remainder of this section.

Bounding T 2 . Recall from (66) that

<!-- formula-not-decoded -->

In what follows, let us bound the three terms on the right-hand side of (98) separately.

- For the first and the third terms on the right-hand side of (98), invoking the Cauchy-Schwarz inequality and Lemma 20 gives

<!-- formula-not-decoded -->

with T 5 defined in (35a), and in addition,

<!-- formula-not-decoded -->

- Let us turn to the second term on the right-hand side of (98). Observing the basic fact that

we can combine it with Lemma 20 to derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality invokes the definition of T 4 (see (34)). By virtue of Lemma 17 and the definition (45) of c /star , one can show that

<!-- formula-not-decoded -->

with probability exceeding 1 -δ ′ . In addition, we note that

<!-- formula-not-decoded -->

Taking these properties together with (101) yields

<!-- formula-not-decoded -->

Putting the above results together, we can deduce that, with probability exceeding 1 -δ ′ ,

<!-- formula-not-decoded -->

Bounding T 4 . When it comes to the quantity T 4 , we recall that

<!-- formula-not-decoded -->

To control T 4 , we first make note of the following result that bounds the empirical reward (for the case with negative rewards), which assists in bounding the term q T 1 .

Lemma 23. With probability at least 1 -2 SAHKδ ′ , it holds that

<!-- formula-not-decoded -->

Proof. The proof basically follows the same arguments as in the proof of Lemma 21, except that r is now replaced with -r .

Lemma 23 tells us that with probability at least 1 -3 SAHKδ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the last line uses the fact (see Lemma 17) that, with probability exceeding 1 -δ ′ ,

In addition, the Freedman inequality in Lemma 15 combined with (106) implies that, with probability at least 1 -3 SAHKδ ,

<!-- formula-not-decoded -->

Combining (105), (107) with (108) reveals that, with probability at least 1 -4 SAHKδ ,

<!-- formula-not-decoded -->

As a result, substitution into (104) leads to

<!-- formula-not-decoded -->

Bounding T 5 . Invoking the arguments in (38a) and recalling the update rule (46), we obtain

<!-- formula-not-decoded -->

Moreover, we recall that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By virtue of (106), one sees that with probability at least 1 -5 SAHKδ ,

Consequently, we arrive at

<!-- formula-not-decoded -->

with probability exceeding 1 5 SAHKδ

<!-- formula-not-decoded -->

Bounding T 6 . Invoking the arguments in (38a), (106) and (111), and recalling the update rule (46), we can demonstrate that

<!-- formula-not-decoded -->

with probability at least 1 -3 SAHKδ .

Putting all this together. Armed with the preceding bounds, we are ready to establish the claimed regret bound. By solving (103),(36b),(109),(113),(114),(38c),(39a) and (39b), we can show that, with probability exceeding 1 -100 SAH 2 Kδ ,

<!-- formula-not-decoded -->

We then readily conclude that the total regret is bounded by

<!-- formula-not-decoded -->

In addition, the regret bound is trivially upper bounded by O ( K ( H -c /star ) ) . The proof is thus completed by combining these two regret bounds and replacing δ ′ with δ 100 SAH 2 K .

## F Proof of the variance-dependent regret bounds (proof of Theorem 4)

In this section, we turn to establishing Theorem 4. The proof primarily contains two parts, as summarized in the following lemmas.

Lemma 24. With probability exceeding 1 -δ/ 2 , Algorithm 1 obeys

Lemma 25. With probability at least 1 -δ/ 2 , Algorithm 1 satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Putting these two regret bounds together and rescaling δ to δ/ 2, we immediately conclude the proof of Theorem 4. The remainder of this section is thus devoted to establishing Lemma 24 and Lemma 25.

## F.1 Proof of Lemma 24

Before proceeding, we recall that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and that

## F.1.1 Bounding T 2

Recall that when proving (36a), we have demonstrated that (see (67))

<!-- formula-not-decoded -->

Lemma 26. With probability at least 1 -4 SAHKδ ′ , one has

This motivates us to bound the sum ∑ k,h ( ̂ σ k h ( s k h , a k h ) -( ̂ r k h ( s k h , a k h ) ) 2 ) , which we accomplish via the following lemma.

<!-- formula-not-decoded -->

Combining Lemma 26 with (115), we can readily derive

<!-- formula-not-decoded -->

with probability at least 1 -4 SAHKδ ′ .

Proof of Lemma 26. For notational convenience, let us define the variance of R h ( s, a ) as v h ( s, a ).

Firstly, we control each ̂ σ k h ( s k h , a k h ) -( ̂ r k h ( s k h , a k h )) 2 with v h ( s, a ). Fix ( s, a, h, k ). Applying Lemma 17 shows that, with probability at least 1 -2 δ ′ ,

This allows us to deduce that, with probability exceeding 1 -2 SAHKδ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It then suffices to bound the sum ∑ k,h v h ( s k h , a k h ). Towards this end, let be the value function with rewards taken to be { v h ( s, a ) } and the policy selected as π k . It is clearly seen that

<!-- formula-not-decoded -->

V

h

˜

s, a

)

In view of Lemma 15, we can obtain

<!-- formula-not-decoded -->

with probability at least 1 -2 SAHKδ ′ . Moreover, invoking Lemma 15 once again reveals that

<!-- formula-not-decoded -->

with probability at least 1 -2 SAHKδ ′ . Combine (120) and (121) to yield

<!-- formula-not-decoded -->

with probability exceeding 1 -4 SAHKδ ′ .

## F.1.2 Bounding T 4

We now move on to the term T 4 , which can be written as T 4 = q T 1 + q T 2 with

<!-- formula-not-decoded -->

k

(

≤

H

2

.

<!-- formula-not-decoded -->

This leaves us with two quantities to control.

To begin with, let us look at q T 1 . In view of Lemma 18 and the union bound over ( s, a, h, k ), we see that, with probability at least 1 -2 SAHKδ ′ ,

As a result, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In view of (122), with probability exceeding 1 -4 SAHKδ ′ we have

<!-- formula-not-decoded -->

Consequently, we arrive at

<!-- formula-not-decoded -->

Next, we proceed to bound q T 2 . Towards this, we make the observation that

<!-- formula-not-decoded -->

Applying Lemma 15 shows that, with probability at least 1 -2 SAHKδ ′ ,

Continue the calculation to derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, (129) holds with probability at least 1 -2 SAHKδ ′ , a consequence of Lemma 15 and Lemma 16. To further bound the right-hand side of (129), we develop the following upper bound:

<!-- formula-not-decoded -->

Note that the first term on the right-hand side (130) is exactly Regret ( K ) = T 1 + T 2 + T 3 + T 4 , the second term on the right-hand side (130) corresponds to -T 4 , whereas the third term on the right-hand side (130) can be bounded by

<!-- formula-not-decoded -->

with probability at least 1 -2 SAHKδ ′ . It then implies the validity of the following bound with probability exceeding 1 -8 SAHKδ ′ :

<!-- formula-not-decoded -->

Combining these bounds with (129), we can use a little algebra to further obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -8 SAHKδ ′ . If we define T 10 = ∑ K k =1 ∑ H h =1 V ( P s k h ,a k h ,h , V /star h +1 ), then substituting (133) into (128) yields: with probability exceeding 1 -10 SAHKδ ′ ,

Combining the above bound on | q T 2 | with (126), with probability exceeding 1 -10 SAHKδ ′

<!-- formula-not-decoded -->

which together with a little algebra yields

<!-- formula-not-decoded -->

## F.1.3 Bounding T 5 and T 6

We now turn attention to the terms T 5 and T 6 . Towards this, we start with the following lemma.

Lemma 27. With probability at least 1 -2 SAHKδ ′ , one has

<!-- formula-not-decoded -->

Proof of Lemma 27. Direct computation gives

<!-- formula-not-decoded -->

Invoking Lemma 7 to bound T 7 and T 1 , we obtain

<!-- formula-not-decoded -->

with probability exceeding 1 -2 SAHKδ ′ .

<!-- formula-not-decoded -->

In view of Lemma 27, it suffices to bound T 6 = ∑ k,h V ( P s k h ,a k h ,h , V k h +1 ). Given that Var ( X + Y ) ≤ 2( Var ( X ) + Var ( Y )) holds for any two random variables X,Y , we have

To further upper bound the right-hand side of (140), we make note of the following lemmas.

Lemma 28. With probability at least 1 -4 SAHKδ ′ , it holds that

<!-- formula-not-decoded -->

Lemma 29. With probability at least 1 -2 δ ′ , it holds that

<!-- formula-not-decoded -->

Combining Lemma 28 and Lemma 29 with (140), we see that with probability at least 1 -6 SAHKδ ′ , and as a result,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This taken collectively with Lemma 27 yields, with probability at least 1 -8 SAHKδ ′ ,

To finish our bounds on T 5 and T 6 , it remains to establish Lemma 28 and Lemma 29.

<!-- formula-not-decoded -->

Proof of Lemma 28. Let R /star h ( s, a ) = V ( P s,a,h , V /star h +1 ), and define

<!-- formula-not-decoded -->

Then V k h ( s ) ≤ var 1 ≤ H 2 . It then follows that

<!-- formula-not-decoded -->

Note that V k depends only on π k , which is determined before the beginning of the k -th episode. Consequently, applying Lemma 15 reveals that, with probability at least 1 -2 SAHKδ ′ ,

<!-- formula-not-decoded -->

Regarding the sum of variance terms on the right-hand side of (145), one can further bound

<!-- formula-not-decoded -->

with probability at least 1 -2 SAHKδ ′ . Here, the last inequality arises from Lemma 15 and Lemma 16 as well as the fact that V k h ( s k h ) = R h ( s k h , a k h ) + P s k h ,a k h ,h V k h +1 . It then follows from elementary algebra that

<!-- formula-not-decoded -->

Substituting (147) into (145) gives

<!-- formula-not-decoded -->

thus indicating that

<!-- formula-not-decoded -->

The proof of Lemma 28 is thus completed.

Proof of Lemma 29. We make the observation that

<!-- formula-not-decoded -->

According to Lemma 15 and Lemma 16, we see that with probability exceeding 1 -δ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, with probability at least 1 -δ ′ one has

<!-- formula-not-decoded -->

It then follows that, with probability at least 1 -2 δ ′ ,

<!-- formula-not-decoded -->

thereby concluding the proof.

## F.1.4 Putting all this together

To finish up, let us rewrite the inequalities (40g) -(40f) as follows, with (40a), (40c), (40d) and (40e) replaced by (117), (135) (143) and (142), respectively:

<!-- formula-not-decoded -->

where we recall that B = 4000(log 2 K ) 3 log(3 SA ) log 1 δ ′ . In addition, it follows from Lemma 28 that

<!-- formula-not-decoded -->

Solving the inequalities above reveals that, with probability exceeding 1 -200 SAH 2 K 2 δ ′ ,

One can thus conclude the proof by recalling that δ ′ = δ 200 SAH 2 K 2 .

<!-- formula-not-decoded -->

## F.2 Proof of Lemma 25

Following similar arguments as in the proof of Lemma 24, we focus on bounding T 2 , T 4 , T 5 and T 6 in terms of var 2 .

## F.2.1 Bounding T 2

Recall that δ ′ is defined as δ ′ = δ 200 SAH 2 K 2 , and that we have demonstrated in (67) that

<!-- formula-not-decoded -->

To bound the right-hand side of (153), we resort to the following lemma.

Lemma 30. With probability at least 1 -4 SAHKδ ′ , one has

Proof. Recall that in Lemma 26, we have shown that with probability at least 1 -4 SAHKδ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We then complete the proof by observing that

<!-- formula-not-decoded -->

Combining Lemma 30 with (153) gives: with probability at least 1 -4 SAHKδ ′ ,

<!-- formula-not-decoded -->

## F.2.2 Bounding T 4

Recall that T 4 = q T 1 + q T 2 , where

<!-- formula-not-decoded -->

Repeating similar arguments employed in the proof of Lemma 26 and using (124), we see that with probability exceeding 1 -6 SAHKδ ′ ,

<!-- formula-not-decoded -->

In addition, from Lemma 15 and the definition of var 2 , we see that with probability at least 1 -2 SAHKδ ′ . Therefore, with probability at least 1 -8 SAHKδ ′ , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F.2.3 Bounding T 5 and T 6

Recall that Lemma 27 asserts that with probability exceeding 1 -2 δ ′ ,

Hence, it suffices to bound T 6 .

From the elementary inequality Var ( X + Y ) ≤ 2 Var ( X ) + 2 Var ( Y ), we obtain

To bound the right-hand side of (160), we resort to the following two lemmas.

<!-- formula-not-decoded -->

Lemma 31. With probability at least 1 -4 SAHKδ ′ , it holds that

Lemma 32. With probability exceeding 1 -4 SAKHδ ′ , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With Lemma 31 and Lemma 32 in place, we can demonstrate that with probability at least 1 -6 SAHKδ ′ ,

Taking this result together with Lemma 27 gives, with probability exceeding 1 -8 SAHKδ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To finish establishing the above bounds on T 5 and T 6 , it suffices to prove Lemma 31 and Lemma 32, which we accomplish in the sequel.

<!-- formula-not-decoded -->

Proof of Lemma 31. For notational convenience, define

<!-- formula-not-decoded -->

We also make the observation that

It is easily seen that q V k h ( s ) ≤ var 2 ≤ H 2 .

<!-- formula-not-decoded -->

Note that q V k only depends on π k , which is determined before the k -th episode starts. Lemma 15 then tells us that, with probability at least 1 -2 SAHKδ ′ ,

Further, it is observed that with probability at least 1 -2 SAHKδ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the last inequality results from Lemma 15, Lemma 16 and the fact that q V k h ( s k h ) = q R h ( s k h , a k h ) + 〈 P s k h ,a k h ,h , q V k h +1 〉 . It then follows that

<!-- formula-not-decoded -->

Taking (165) and (167) together leads to

<!-- formula-not-decoded -->

which further implies that

<!-- formula-not-decoded -->

This concludes the proof.

Proof of Lemma 29. A little algebra gives

<!-- formula-not-decoded -->

From Lemma 15 and Lemma 16, we can show that with probability 1 -2 SAKHδ ′ ,

<!-- formula-not-decoded -->

Additionally, with probability at least 1 -2 SAKHδ ′ ,

<!-- formula-not-decoded -->

It then follows that

<!-- formula-not-decoded -->

with probability at least 1 4 SAKHδ

-′ . The proof is thus complete.

## F.2.4 Putting all pieces together

Recall that B = 4000(log 2 K ) 3 log(3 SA ) log 1 δ ′ . The last step is to rewrite the inequalities (40g) -(40f) as follows with (40a), (40c), (40d) and (40e) replaced by (157),(159) (163) and (162) respectively:

<!-- formula-not-decoded -->

which are valid with probability at least 1 -200 SAH 2 K 2 δ ′ . Solving the inequalities listed above, we can readily conclude that

<!-- formula-not-decoded -->

This finishes the proof by recalling that δ ′ = δ 200 SAH 2 K 2 .

## G Minimax lower bounds

In this section, we establish the lower bounds advertised in this paper.

## G.1 Proof of Theorem 12

Consider any given ( S, A, H ). We start by establishing the following lemma.

Lemma 33. Consider any K ′ ≥ 1 . For any algorithm, there exists an MDP instance with S states, A actions, and horizon H , such that the regret in K ′ episodes is at least

<!-- formula-not-decoded -->

Proof of Lemma 33. Our construction of the hard instance is based on the hard instance JAO-MDP constructed in Jaksch et al. (2010); Jin et al. (2018). In Jin et al. (2018, Appendix.D), the authors already showed that when K ′ ≥ C 0 SAH for some constant C 0 &gt; 0, the minimax regret lower bound is Ω( √ SAH 3 K ′ ). Hence, it suffices for us to focus on the regime where K ′ ≤ C 0 SAH . Without loss of generality, we assume S = A = 2, and the argument to generalize it to arbitrary ( S, A ) is standard and henc omitted for brevity.

Recall the construction of JAO-MDP in Jaksch et al. (2010). Let the two states be x and y , and the two actions be a and b . The reward is always equal to x in state 1 and 1 / 2 in state y . The probability transition kernel is given by

<!-- formula-not-decoded -->

where we choose δ = C 1 /H and /epsilon1 = 1 /H . Then the mixing time of the MDP is roughly O ( H ). By choosing C 1 large enough, we can ensure that the MDP is C 3 -mixing after the first half of the horizons for some proper constant C 3 ∈ (0 , 1 / 2).

It is then easy to show that action b is the optimal action for state y . Moreover, whenever action a is chosen in state y , the learner needs to pay regret Ω( /epsilon1H ) = Ω(1). In addition, to differentiate action a from action b in state y with probability at least 1 -1 10 , the learner needs at least Ω( /epsilon1 δ 2 ) = Ω( H ) rounds - let us call it C 4 H rounds for some proper constant C 4 &gt; 0. As a result, in the case where K ′ ≤ C 4 H , the minimax regret is at least Ω( K ′ H 2 /epsilon1 ) = Ω( K ′ H ). When C 4 H ≤ K ′ ≤ C 0 SAH = 4 C 0 H , the minimax regret is at least Ω( C 4 H 2 ) = Ω( K ′ H ). This concludes the proof.

With Lemma 33, we are ready to prove Theorem 12. Let M be the hard instance for K ′ = max { 1 10 Kp, 1 } constructed in the proof of Lemma 33. We construct an MDP M ′ as below.

- In the first step, for any state s , with probability p , the leaner transitions to a copy of M , and with probability 1 -p , the learner transitions to a dumb state with 0 reward.

It can be easily verified that v /star ≤ pH . Let X = X 1 + X 2 + · · · + X k , where { X i } K i =1 are i.i.d. Bernoulli random variables with mean p . Let g ( X,K ′ ) denote the minimax regret on the hard instance M in X episodes. Given that g ( X,K ′ ) is non-decreasing in X , one sees that

<!-- formula-not-decoded -->

- In the case where Kp ≥ 10, Lemma 17 tells us that with probability at least 1 / 2, X ≥ 1 10 Kp = K ′ , and then it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- In the case where Kp &lt; 10, with probability exceeding 1 -(1 -p ) K ≥ (1 -e -Kp ) ≥ Kp 30 , one has X ≥ 1. Then one has

The preceding bounds taken together complete the proof.

## G.2 Proof of Theorem 13

Without loss of generality, assume that S = A = 2 (as in the proof of Theorem 12), and recall the assumption that p ≤ 1 / 4. In what follows, we construct a hard instance for which the learner needs to identify the correct action for each step.

Let S = { s 1 , s 2 } , and take the initial state to be s 1 . The transition kernel and cost are chosen as follows.

- Select a /star h ∈ { a 1 , a 2 } for every h ∈ [ H ].
- For each action a and each step h , set P s 2 ,a,h = e s 2 and c h ( s 2 , a ) = 0.
- For each step h and each action a = a /star h , set P s 1 ,a,h = e s 2 and c h ( s 1 , a ) = 1.

/negationslash

- Set P s 1 ,a /star h ,h = e s 1 and c h ( s 1 , a /star h ) = p .

It can be easily checked that c /star = Hp (the cost obtained by choosing action a /star h for each step h ).

Note that in the above construction, the a /star h 's are selected independently across different steps. Thus, to identify the optimal action a /star h for at least half of the H steps, we need at least Ω( H ) episodes. This implies that: there exists a constant C 5 &gt; 0 such that in the first K ≤ C 5 H episodes, the cost of the learner is at least Ω( H (1 -p )). As a result, the minimax regret is at least

<!-- formula-not-decoded -->

Ω ( K ( H -c /star ) ) = Ω ( KH (1 -p ) ) when K ≤ C 5 H . Similarly, in the case where C 5 H ≤ K ≤ 100 H p , the minimax regret is at least

We then turn to the case where K ≥ 100 H p . Let M be the hard instance having the same transition as the instance constructed in the proof of Lemma 33, and set the cost as 1 / 2 (resp. 1) for state x (resp. state y ), with respect to K ′ = Kp/ 10 ≥ 10 H (a quantity defined therein). Let M ′ be the MDP such that: in the first step, with probability p , the learner transitions to a copy of M , and with probability 1 -p , the learner transitions to a dumb state with 0 cost. Then we clearly have c /star = Θ( Hp ). It follows from Lemma 17 that, with probability exceeding 1 / 2, one has X ≥ 1 3 Kp -log 2 ≥ 1 6 Kp , where X is again defined in the proof of Lemma 33. Then one has

The proof is thus completed by combining the above minimax regret lower bounds for the three regimes K ∈ [1 , C 5 H ], K ∈ ( C 5 H, 100 H p ] and K ∈ ( 100 H p , ∞ ].

<!-- formula-not-decoded -->

## G.3 Proof of Theorem 14

When K ≥ SAH/p , the lower bound in Theorem 12 readily applies because the regret is at least Ω( √ SAH 3 Kp ) and the variance var is at most pH 2 . When SAH ≤ K ≤ SAH/p , the regret is at least Ω( SAH 2 ) = Ω(min { √ SAH 3 Kp + SAH 2 , KH } ). As a result, it suffices to focus on the case where 1 ≤ K ≤ SAH , Towards this end, we only need the following lemma, which suffices for us to complete the proof.

Lemma 34. Consider any 1 ≤ K ≤ SAH . There exists an MDP instance with S states, A actions, horizon H , and var 1 = var 2 = 0 , such that the regret is at least Ω( KH ) .

/negationslash

Proof. Let us construct an MDP with deterministic transition; more precisely, for each ( s, a, h ), there is some s ′ such that P s,a,h,s ′ = 1 and P s,a,h,s ′′ = 0 for any s ′′ = s ′ . The reward function is also chosen to be deterministic. In this case, it is easy to verify that var 1 = var 2 = 0.

/negationslash

We first assume S = 2. For any action a and horizon h , we set P s 2 ,a,h = e s 2 and r h ( s 2 , a ) = 0. For any action a = a /star and h , we also set P s 1 ,a,h = e s 2 and r h ( s 2 , a ) = 0. At last, we set P s 1 ,a /star ,h = e s 1 and r h ( s 1 , a /star ) = 1. In other words, there are a dumb state and a normal state in each step. The learner would naturally hope to find the correct action to avoid the dumb state. Obviously, V /star 1 ( s 1 ) = H . To find an H 2 -optimal policy, the learner needs to identify a /star for the first H 2 steps, requiring at least Ω( HA ) rounds in expectation. As a result, the minimax regret is at least Ω( KH ) when K ≤ cHA for some proper constant c &gt; 0.

Let us refer to the hard instance above as a hard chain . For general S , we can construct d := S 2 hard chains. Let the two states in the i -th hard chain be ( s 1 ( i ) , s 2 ( i )). We set the initial distribution to be the uniform distribution over { s 1 ( i ) } d i =1 . Then V /star 1 ( s 1 ( i )) = H holds for any 1 ≤ i ≤ d . Let Regret i ( K ) be the expected regret resulting from the i -th hard chain. When K ≥ 100 S , Lemma 17 tells us that with probability at least 1 2 , s 1 ( i ) is visited for at least K 10 S ≥ 10 times. As a result, we have

Summing over i , we see that the total regret is at least ∑ d i =1 Regret i ( K ) = Ω( KH ). When K &lt; 100 S , with probability at least 1 -(1 -1 S ) K ≥ 0 . 0001 K S , we know that s 1 ( i ) is visited for at least one time. Therefore, it holds that Regret i ( K ) ≥ Ω( KH S ). Summing over i , we obtain as claimed.

## References

Agarwal, A., Kakade, S., and Yang, L. F. (2020). Model-based reinforcement learning with a generative model is minimax optimal. In Conference on Learning Theory , pages 67-83.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Agarwal, A., Krishnamurthy, A., Langford, J., Luo, H., et al. (2017). Open problem: First-order regret bounds for contextual bandits. In Conference on Learning Theory , pages 4-7.
- Agrawal, S. and Jia, R. (2017). Optimistic posterior sampling for reinforcement learning: worst-case regret bounds. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R., editors, Advances in Neural Information Processing Systems 30 , pages 1184-1194. Curran Associates, Inc.
- Allen-Zhu, Z., Bubeck, S., and Li, Y. (2018). Make the minority great again: First-order regret bound for contextual bandits. In International Conference on Machine Learning , pages 186-194.
- Azar, M. G., Munos, R., and Kappen, H. J. (2013). Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3):325-349.
- Azar, M. G., Osband, I., and Munos, R. (2017). Minimax regret bounds for reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning , pages 263-272.
- Bai, Y., Xie, T., Jiang, N., and Wang, Y.-X. (2019). Provably efficient Q-learning with low switching cost. In Advances in Neural Information Processing Systems , pages 8004-8013.
- Bartlett, P. L. and Tewari, A. (2009). Regal: a regularization based algorithm for reinforcement learning in weakly communicating mdps. In Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI 2009)) .
- Beck, C. L. and Srikant, R. (2012). Error bounds for constant step-size Q-learning. Systems &amp; control letters , 61(12):1203-1208.
- Bertsekas, D. (2019). Reinforcement learning and optimal control . Athena Scientific.
- Brafman, R. I. and Tennenholtz, M. (2003). R-max - a general polynomial time algorithm for near-optimal reinforcement learning. J. Mach. Learn. Res. , 3(Oct):213-231.
- Cai, Q., Yang, Z., Jin, C., and Wang, Z. (2019). Provably efficient exploration in policy optimization. arXiv preprint arXiv:1912.05830 .
- Chen, L., Jafarnia-Jahromi, M., Jain, R., and Luo, H. (2021). Implicit finite-horizon approximation and efficient optimal algorithms for stochastic shortest path. Advances in Neural Information Processing Systems , 34.
- Chen, Z., Maguluri, S. T., Shakkottai, S., and Shanmugam, K. (2020). Finite-sample analysis of contractive stochastic approximation using smooth convex envelopes. Advances in Neural Information Processing Systems , 33:8223-8234.
- Cui, Q. and Yang, L. F. (2021). Minimax sample complexity for turn-based stochastic game. In Uncertainty in Artificial Intelligence , pages 1496-1504.
- Dann, C. and Brunskill, E. (2015). Sample complexity of episodic fixed-horizon reinforcement learning. In Advances in Neural Information Processing Systems , pages 2818-2826.
- Dann, C., Lattimore, T., and Brunskill, E. (2017). Unifying PAC and regret: Uniform PAC bounds for episodic reinforcement learning. Advances in Neural Information Processing Systems , 30.
- Dann, C., Li, L., Wei, W., and Brunskill, E. (2019). Policy certificates: Towards accountable reinforcement learning. In Proceedings of the 36th International Conference on Machine Learning , pages 1507-1516.
- Dann, C., Marinov, T. V., Mohri, M., and Zimmert, J. (2021). Beyond value-function gaps: Improved instance-dependent regret bounds for episodic reinforcement learning. Advances in Neural Information Processing Systems , 34:1-12.
- Domingues, O. D., Ménard, P., Kaufmann, E., and Valko, M. (2021). Episodic reinforcement learning in finite mdps: Minimax lower bounds revisited. In Algorithmic Learning Theory , pages 578-598.

- Dong, K., Wang, Y., Chen, X., and Wang, L. (2019). Q-learning with UCB exploration is sample efficient for infinite-horizon MDP. arXiv preprint arXiv:1901.09311 .
- Efroni, Y., Merlis, N., Ghavamzadeh, M., and Mannor, S. (2019). Tight regret bounds for model-based reinforcement learning with greedy policies. Advances in Neural Information Processing Systems , 32.
- Even-Dar, E. and Mansour, Y. (2003). Learning rates for Q-learning. Journal of Machine Learning Research , 5(Dec):1-25.
- Freedman, D. A. (1975). On tail probabilities for martingales. the Annals of Probability , 3(1):100-118.
- Fruit, R., Pirotta, M., Lazaric, A., and Ortner, R. (2018). Efficient bias-span-constrained explorationexploitation in reinforcement learning. In ICML 2018-The 35th International Conference on Machine Learning , volume 80, pages 1578-1586.
- Jaksch, T., Ortner, R., and Auer, P. (2010). Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(Apr):1563-1600.
- Ji, X. and Li, G. (2023). Regret-optimal model-free reinforcement learning for discounted MDPs with short burn-in time. Advances in neural information processing systems .
- Jiang, N. and Agarwal, A. (2018). Open problem: The dependence of sample complexity lower bounds on planning horizon. In Conference On Learning Theory , pages 3395-3398.
- Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. (2018). Is Q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4863-4873.
- Jin, C., Krishnamurthy, A., Simchowitz, M., and Yu, T. (2020). Reward-free exploration for reinforcement learning. International Conference on Machine Learning .
- Jin, Y., Yang, Z., and Wang, Z. (2021). Is pessimism provably efficient for offline RL? In International Conference on Machine Learning , pages 5084-5096.
- Kakade, S. M. (2003). On the sample complexity of reinforcement learning . PhD thesis, University of London London, England.
- Kearns, M. and Singh, S. (1998a). Finite-sample convergence rates for Q-learning and indirect algorithms. Advances in neural information processing systems , 11.
- Kearns, M. J. and Singh, S. P. (1998b). Near-optimal reinforcement learning in polynominal time. In Proceedings of the Fifteenth International Conference on Machine Learning , pages 260-268.
- Kolter, J. Z. and Ng, A. Y. (2009). Near-bayesian exploration in polynomial time. In Proceedings of the 26th annual international conference on machine learning , pages 513-520.
- Lattimore, T. and Hutter, M. (2012). PAC bounds for discounted MDPs. In International Conference on Algorithmic Learning Theory , pages 320-334. Springer.
- Lee, C.-W., Luo, H., Wei, C.-Y., and Zhang, M. (2020). Bias no more: high-probability data-dependent regret bounds for adversarial bandits and mdps. Advances in neural information processing systems , 33:15522-15533.
- Levine, S., Kumar, A., Tucker, G., and Fu, J. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643 .
- Li, G., Cai, C., Chen, Y., Wei, Y., and Chi, Y. (2024a). Is Q-learning minimax optimal? a tight sample complexity analysis. Operations Research , 72(1):222-236.
- Li, G., Chi, Y., Wei, Y., and Chen, Y. (2022). Minimax-optimal multi-agent RL in Markov games with a generative model. Advances in Neural Information Processing Systems , 35:15353-15367.

- Li, G., Shi, L., Chen, Y., Chi, Y., and Wei, Y. (2024b). Settling the sample complexity of model-based offline reinforcement learning. The Annals of Statistics , 52(1):233-260.
- Li, G., Shi, L., Chen, Y., Gu, Y., and Chi, Y. (2021a). Breaking the sample complexity barrier to regretoptimal model-free reinforcement learning. Advances in Neural Information Processing Systems , 34.
- Li, G., Wei, Y., Chi, Y., and Chen, Y. (2024c). Breaking the sample size barrier in model-based reinforcement learning with a generative model. Operations Research , 72(1):203-221.
- Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2021b). Sample complexity of asynchronous Q-learning: Sharper analysis and variance reduction. IEEE Transactions on Information Theory , 68(1):448-473.
- Li, G., Yan, Y., Chen, Y., and Fan, J. (2024d). Minimax-optimal reward-agnostic exploration in reinforcement learning. Conference on Learning Theory (COLT) .
- Li, G., Zhan, W., Lee, J. D., Chi, Y., and Chen, Y. (2024e). Reward-agnostic fine-tuning: Provable statistical benefits of hybrid reinforcement learning. Advances in Neural Information Processing Systems , 36.
- Li, Y., Wang, R., and Yang, L. F. (2021c). Settling the horizon-dependence of sample complexity in reinforcement learning. In IEEE Symposium on Foundations of Computer Science .
- Maurer, A. and Pontil, M. (2009). Empirical Bernstein bounds and sample variance penalization. In Conference on Learning Theory .
- Ménard, P., Domingues, O. D., Shang, X., and Valko, M. (2021). UCB momentum Q-learning: Correcting the bias without forgetting. In International Conference on Machine Learning , pages 7609-7618.
- Neu, G. and Pike-Burke, C. (2020). A unifying view of optimism in episodic reinforcement learning. arXiv preprint arXiv:2007.01891 .
- Osband, I., Russo, D., and Van Roy, B. (2013). (more) efficient reinforcement learning via posterior sampling. In Advances in Neural Information Processing Systems , pages 3003-3011.
- Pacchiano, A., Ball, P., Parker-Holder, J., Choromanski, K., and Roberts, S. (2020). On optimism in model-based reinforcement learning. arXiv preprint arXiv:2006.11911 .
- Pananjady, A. and Wainwright, M. J. (2020). Instance-dependent /lscript ∞ -bounds for policy evaluation in tabular reinforcement learning. IEEE Transactions on Information Theory , 67(1):566-585.
- Qu, G. and Wierman, A. (2020). Finite-time analysis of asynchronous stochastic approximation and Qlearning. In Proceedings of Thirty Third Conference on Learning Theory (COLT) .
- Rashidinejad, P., Zhu, B., Ma, C., Jiao, J., and Russell, S. (2021). Bridging offline reinforcement learning and imitation learning: A tale of pessimism. Advances in Neural Information Processing Systems , 34:1170211716.
- Ren, T., Li, J., Dai, B., Du, S. S., and Sanghavi, S. (2021). Nearly horizon-free offline reinforcement learning. Advances in neural information processing systems , 34:15621-15634.
- Russo, D. (2019). Worst-case regret bounds for exploration via randomized value functions. In Advances in Neural Information Processing Systems , pages 14433-14443.
- Shi, L., Li, G., Wei, Y., Chen, Y., and Chi, Y. (2022). Pessimistic Q-learning for offline reinforcement learning: Towards optimal sample complexity. In International Conference on Machine Learning , pages 19967-20025.
- Shi, L., Li, G., Wei, Y., Chen, Y., Geist, M., and Chi, Y. (2023). The curious price of distributional robustness in reinforcement learning with a generative model. Advances in Neural Information Processing Systems .

- Sidford, A., Wang, M., Wu, X., Yang, L., and Ye, Y. (2018a). Near-optimal time and sample complexities for solving Markov decision processes with a generative model. In Advances in Neural Information Processing Systems , pages 5186-5196.
- Sidford, A., Wang, M., Wu, X., and Ye, Y. (2018b). Variance reduced value iteration and faster algorithms for solving Markov decision processes. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 770-787. Society for Industrial and Applied Mathematics.
- Simchowitz, M. and Jamieson, K. G. (2019). Non-asymptotic gap-dependent regret bounds for tabular MDPs. In Advances in Neural Information Processing Systems , pages 1153-1162.
- Strehl, A. L., Li, L., Wiewiora, E., Langford, J., and Littman, M. L. (2006). PAC model-free reinforcement learning. In Proceedings of the 23rd international conference on Machine learning , pages 881-888. ACM.
- Strehl, A. L. and Littman, M. L. (2008). An analysis of model-based interval estimation for markov decision processes. Journal of Computer and System Sciences , 74(8):1309-1331.
- Szita, I. and Szepesvári, C. (2010). Model-based reinforcement learning with nearly tight exploration complexity bounds. In ICML .
- Talebi, M. S. and Maillard, O.-A. (2018). Variance-aware regret bounds for undiscounted reinforcement learning in mdps. arXiv preprint arXiv:1803.01626 .
- Tarbouriech, J., Zhou, R., Du, S. S., Pirotta, M., Valko, M., and Lazaric, A. (2021). Stochastic shortest path: Minimax, parameter-free and towards horizon-free regret. Advances in Neural Information Processing Systems , 34.
- Tirinzoni, A., Pirotta, M., and Lazaric, A. (2021). A fully problem-dependent regret lower bound for finitehorizon MDPs. arXiv preprint arXiv:2106.13013 .
- Wagenmaker, A. J., Chen, Y., Simchowitz, M., Du, S., and Jamieson, K. (2022). First-order regret in reinforcement learning with linear function approximation: A robust estimation approach. In International Conference on Machine Learning , pages 22384-22429.
- Wainwright, M. J. (2019a). Stochastic approximation with cone-contractive operators: Sharp /lscript ∞ -bounds for Q-learning. arXiv preprint arXiv:1905.06265 .
- Wainwright, M. J. (2019b). Variance-reduced Q-learning is minimax optimal. arXiv preprint arXiv:1906.04697 .
- Wang, K., Zhou, K., Wu, R., Kallus, N., and Sun, W. (2023). The benefits of being distributional: Small-loss bounds for reinforcement learning. arXiv preprint arXiv:2305.15703 .
- Wang, R., Du, S. S., Yang, L. F., and Kakade, S. M. (2020). Is long horizon reinforcement learning more difficult than short horizon reinforcement learning? In Advances in Neural Information Processing Systems .
- Wang, X., Cui, Q., and Du, S. S. (2022). On gap-dependent bounds for offline reinforcement learning. Advances in Neural Information Processing Systems , 35:14865-14877.
- Xie, T., Jiang, N., Wang, H., Xiong, C., and Bai, Y. (2021). Policy finetuning: Bridging sample-efficient offline and online reinforcement learning. Advances in neural information processing systems , 34:2739527407.
- Xiong, Z., Shen, R., Cui, Q., Fazel, M., and Du, S. S. (2022). Near-optimal randomized exploration for tabular markov decision processes. Advances in Neural Information Processing Systems , 35:6358-6371.
- Xu, H., Ma, T., and Du, S. (2021). Fine-grained gap-dependent bounds for tabular MDPs via adaptive multi-step bootstrap. In Conference on Learning Theory , pages 4438-4472.
- Yan, Y., Li, G., Chen, Y., and Fan, J. (2023). The efficacy of pessimism in asynchronous Q-learning. IEEE Transactions on Information Theory , 69(11):7185-7219.

- Yang, K., Yang, L., and Du, S. (2021). Q -learning with logarithmic regret. In International Conference on Artificial Intelligence and Statistics , pages 1576-1584.
- Yin, M., Duan, Y., Wang, M., and Wang, Y.-X. (2022). Near-optimal offline reinforcement learning with linear representation: Leveraging variance information with pessimism. arXiv preprint arXiv:2203.05804 .
- Zanette, A. and Brunskill, E. (2019). Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. In International Conference on Machine Learning , pages 7304-7312.
- Zhang, Z., Ji, X., and Du, S. (2021a). Is reinforcement learning more difficult than bandits? a near-optimal algorithm escaping the curse of horizon. In Conference on Learning Theory , pages 4528-4531.
- Zhang, Z., Ji, X., and Du, S. (2022). Horizon-free reinforcement learning in polynomial time: the power of stationary policies. In Conference on Learning Theory , pages 3858-3904.
- Zhang, Z., Zhou, Y., and Ji, X. (2020). Almost optimal model-free reinforcement learning via referenceadvantage decomposition. In Advances in Neural Information Processing Systems .
- Zhang, Z., Zhou, Y., and Ji, X. (2021b). Model-free reinforcement learning: from clipped pseudo-regret to sample complexity. In International Conference on Machine Learning , pages 12653-12662.
- Zhao, H., He, J., Zhou, D., Zhang, T., and Gu, Q. (2023). Variance-dependent regret bounds for linear bandits and reinforcement learning: Adaptivity and computational efficiency. arXiv preprint arXiv:2302.10371 .
- Zhou, R., Zihan, Z., and Du, S. S. (2023). Sharp variance-dependent bounds in reinforcement learning: Best of both worlds in stochastic and deterministic environments. In International Conference on Machine Learning , pages 42878-42914.