## Distributionally Robust Model-Based Offline Reinforcement Learning with Near-Optimal Sample Complexity

Laixi Shi ∗ Carnegie Mellon University laixis@andrew.cmu.edu

Yuejie Chi ∗ Carnegie Mellon University yuejiechi@cmu.edu

August 2022; revised December 2023

## Abstract

This paper concerns the central issues of model robustness and sample efficiency in offline reinforcement learning (RL), which aims to learn to perform decision making from history data without active exploration. Due to uncertainties and variabilities of the environment, it is critical to learn a robust policy-with as few samples as possible-that performs well even when the deployed environment deviates from the nominal one used to collect the history dataset. We consider a distributionally robust formulation of offline RL, focusing on tabular robust Markov decision processes with an uncertainty set specified by the Kullback-Leibler divergence in both finite-horizon and infinite-horizon settings. To combat with sample scarcity, a model-based algorithm that combines distributionally robust value iteration with the principle of pessimism in the face of uncertainty is proposed, by penalizing the robust value estimates with a carefully designed data-driven penalty term. Under a mild and tailored assumption of the history dataset that measures distribution shift without requiring full coverage of the state-action space, we establish the finite-sample complexity of the proposed algorithms. We further develop an information-theoretic lower bound, which suggests that learning RMDPs is at least as hard as the standard MDPs when the uncertainty level is sufficient small, and corroborates the tightness of our upper bound up to polynomial factors of the (effective) horizon length for a range of uncertainty levels. To the best our knowledge, this provides the first provably near-optimal robust offline RL algorithm that learns under model uncertainty and partial coverage.

Keywords: offline/batch reinforcement learning, distributional robustness, pessimism, model-based reinforcement learning, KL divergence uncertainty

## Contents

| 1   | Introduction                                        | Introduction                                                  |   2 |
|-----|-----------------------------------------------------|---------------------------------------------------------------|-----|
|     | 1.1                                                 | Challenges and premises in robust offline RL . . . . . .      |   2 |
|     | 1.2                                                 | Main contributions . . . . . . . . . . . . . . . . . . . . .  |   3 |
|     | 1.3                                                 | Related works . . . . . . . . . . . . . . . . . . . . . . . . |   4 |
|     | 1.4                                                 | Notation and paper organization . . . . . . . . . . . . .     |   6 |
| 2   | Problem formulation: episodic finite-horizon RMDPs  | Problem formulation: episodic finite-horizon RMDPs            |   6 |
|     | 2.1                                                 | Basics of finite-horizon episodic tabular MDPs . . . . .      |   6 |
|     | 2.2                                                 | Distributionally robust MDPs . . . . . . . . . . . . . . .    |   7 |
|     | 2.3                                                 | Distributionally robust offline RL . . . . . . . . . . . . .  |   8 |
| 3   | Algorithm and theory: episodic finite-horizon RMDPs | Algorithm and theory: episodic finite-horizon RMDPs           |   9 |
|     | 3.1                                                 | Building an empirical nominal MDP . . . . . . . . . . .       |   9 |
|     | 3.2                                                 | DRVI-LCB : a pessimistic variant of robust value iteration    |  10 |
|     | 3.3                                                 | Performance guarantees . . . . . . . . . . . . . . . . . .    |  11 |

∗ Department of Electrical and Computer Engineering, Carnegie Mellon University, Pittsburgh, PA 15213, USA.

| 4   | Robust offline RL for discounted infinite-horizon RMDPs   | Robust offline RL for discounted infinite-horizon RMDPs   |   13 |
|-----|-----------------------------------------------------------|-----------------------------------------------------------|------|
|     | 4.1                                                       | Backgrounds on discounted infinite-horizon RMDPs          |   13 |
|     | 4.2                                                       | Data collection and constructing the empirical MDP        |   15 |
|     | 4.3                                                       | DRVI-LCB for discounted infinite-horizon RMDPs .          |   15 |
|     | 4.4                                                       | Performance guarantees . . . . . . . . . . . . . . .      |   17 |
| 5   | Numerical experiments                                     | Numerical experiments                                     |   19 |
| 6   | Conclusion                                                | Conclusion                                                |   21 |
| A   | Preliminaries                                             | Preliminaries                                             |   25 |
|     | A.1                                                       | Properties of the robust Bellman operator . . . . .       |   25 |
|     | A.2                                                       | Concentration inequalities . . . . . . . . . . . . . .    |   26 |
|     | A.3                                                       | Kullback-Leibler (KL) divergence . . . . . . . . . .      |   26 |
| B   | Analysis: episodic finite-horizon RMDPs                   | Analysis: episodic finite-horizon RMDPs                   |   27 |
|     | B.1                                                       | Proof of Theorem 1 . . . . . . . . . . . . . . . . . .    |   27 |
|     | B.2                                                       | Proof of Lemma 10 . . . . . . . . . . . . . . . . . .     |   31 |
|     | B.3                                                       | Proof of Theorem 2 . . . . . . . . . . . . . . . . . .    |   35 |
| C   | Analysis: discounted infinite-horizon RMDPs               | Analysis: discounted infinite-horizon RMDPs               |   53 |
|     | C.1                                                       | Proof of Lemma 2 . . . . . . . . . . . . . . . . . .      |   53 |
|     | C.2                                                       | Proof of Lemma 3 . . . . . . . . . . . . . . . . . .      |   54 |
|     | C.3                                                       | Proof of Theorem 3 . . . . . . . . . . . . . . . . . .    |   55 |

C.4

Proof of Theorem 4 .

## 1 Introduction

Reinforcement learning (RL) concerns about finding an optimal policy that maximizes an agent's expected total reward in an unknown environment. A fundamental challenge of deploying RL to real-world applications is the limited ability to explore or interact with the environment, due to resources, time, or safety constraints. Offline RL, or batch RL, seeks to circumvent this challenge by resorting to history data-which are often collected by executing some possibly unknown behavior policy in the past-with the hope that the history data might already provide significant insights about the targeted optimal policy without further exploration (Levine et al., 2020).

Besides maximizing the expected total reward, perhaps an equally important goal-to say the least-for an RL agent is safety and robustness (Garcıa and Fernández, 2015), especially in high-stake applications such as robotics, autonomous driving, clinical trials, financial investments, and so on (Choi et al., 2009; Schulman et al., 2013). It has been observed that a standard RL agent trained in an ideal environment might be extremely sensitive and fail catastrophically when the deployed environment is subject to small adversarial perturbations (Zhang et al., 2020). Consequently, robust RL has attracted a surge of attentions with the goal to learn an optimal policy that is robust to environment perturbations. In fact, providing robustness guarantees becomes even more relevant in the offline setting, which can be formulated as robust offline RL , since the history data is often inevitably collected from a timeframe where it is no longer reasonable to assume model stillness, due to the highly non-stationary and time-varying dynamics of many real-world applications. Altogether, this naturally leads to a question:

Can we learn a near-optimal policy which is robust with respect to uncertainties and variabilities of the environments using as few history samples as possible?

## 1.1 Challenges and premises in robust offline RL

Despite significant amount of recent activities in robust RL and offline RL, addressing model uncertainty and sample efficiency simultaneously remains challenging due to several key issues that we single out below.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

64

- Distribution shift. The history data is generated by following some behavior policy in an outdated environment, which can result in a data distribution that is heavily deviated from the desired one, i.e., induced by the target policy in the deployed environment.
- Partial and limited coverage. The history data might only provide partial and limited coverage over the entire state-action space, where the limited sample size leads to a poor estimate of the associated model parameters, and consequently, unreliable policy learning outcomes.

Understanding the implications of-and designing algorithms that work around-these challenges play a major role in advancing the state-of-the-art of robust offline RL. In particular, two prevalent algorithmic ideas, distributional robustness and pessimism, are called out as our guiding principles.

- Distributional robustness. Instead of finding an optimal policy in a fixed environment, motivated by the literature in distributionally robust optimization (Delage and Ye, 2010), one might seek to find a policy that achieves the best worst-case performance for all the environments in some uncertainty set around the offline environment, as formalized in the framework of robust RL (Iyengar, 2005; Nilim and El Ghaoui, 2005).
- Pessimism. When the samples are scarce, it is wise to act with caution based on the principle of pessimism, where one subtracts a penalty term-representing the confidence of the corresponding estimate-from the value functions to avoid excessive risk. Encouragingly, pessimism has been recently shown as an indispensable ingredient to achieve sample efficiency in offline RL without requiring full coverage (Jin et al., 2021; Li et al., 2022; Rashidinejad et al., 2021), as long as the trajectory of the behavior policy provides sufficient overlap with that of the target policy.

While these two ideas have been proven useful for robust RL and offline RL separately , tackling robust offline RL needs novel ingredients that go significantly beyond a naïve combination of existing techniques. This is because, in robust offline RL, one needs to handle the distribution shift induced not only by the behavior policy, but also by model perturbations, thus the penalty term derived from the pessimism principle in standard offline RL is no longer applicable. Indeed, while the value function of standard RL depends linearly with respect to the transition kernel, the dependency between the nominal transition kernel and the robust value function unfortunately becomes highly nonlinear-even without a closed-form expressionmaking the control of statistical uncertainty extremely challenging in robust offline RL.

## 1.2 Main contributions

In this work, we provide an affirmative answer to the question raised earlier, by developing a provably efficient model-based algorithm that learns a near-optimal distributionally-robust policy from a minimal number of offline samples. Specifically, we consider a Robust Markov Decision Process (RMDP) with S states, A actions in both the nonstationary finite-horizon setting (with horizon length H ) and the discounted infinite-horizon setting (with discount factor γ ). Different from standard MDPs, RMDPs specify a family transition kernels, which lie within an uncertainty set taken as a small ball of size σ around a nominal transition kernel with respect to the Kullback-Leibler (KL) divergence. Given K episodes (resp. N transitions) of history data drawn by following some behavior policy π b under the nominal transition kernel in the finite-horizon (resp. infinite-horizon) setting, our goal is to learn the optimal robust policy π ⋆ in the maximin sense, which has the best worst-case value for all the transition kernels within the uncertainty set (Iyengar, 2005; Nilim and El Ghaoui, 2005). Our main results are summarized below.

- We introduce a notion called robust single-policy clipped concentrability coefficient C ⋆ rob ∈ [1 /S, ∞ ] to quantify the quality of history data, which measures the distribution shift between the behavior policy π b and the optimal robust policy π ⋆ in the presence of model perturbations, without requiring full coverage of the entire state-action space by the behavior policy. In contrast, prior algorithms (Panaganti and Kalathil, 2022; Yang et al., 2022; Zhou et al., 2021)-using simulator or offline dataall require full coverage of the entire state-action space.
- We propose a novel pessimistic variant of distributionally robust value iteration with a plug-in estimate of the nominal transition kernel (Iyengar, 2005; Nilim and El Ghaoui, 2005), called DRVI-LCB , by

penalizing the robust value estimates with a carefully designed data-driven penalty term. We demon- strate that DRVI-LCB finds an ε -optimal robust policy as soon as the sample size is above ˜ O ( SC ⋆ rob H 5 P ⋆ min σ 2 ε 2 ) for the finite-horizon setting and ˜ O ( SC ⋆ rob P ⋆ min σ 2 (1 -γ ) 4 ε 2 ) for the infinite-horizon setting, up to some logarithmic factor after a burn-in cost independent of ε . Here, P ⋆ min is the smallest positive state transition probability of the optimal robust policy π ⋆ under the nominal kernel.

- To complement the upper bound, we further develop information-theoretic lower bounds for a range of uncertainty levels, showing there exists some transition kernel such that at least Ω ( SC ⋆ rob H 4 ε 2 ) samples (resp. Ω ( SC ⋆ rob (1 -γ ) 3 ε 2 ) samples) are needed to find an ε -optimal robust policy when the uncertainty level σ ≲ 1 /H (resp. σ ≲ (1 -γ ) ), and at least Ω ( SC ⋆ rob H 3 P ⋆ min σ 2 ε 2 ) samples (resp. Ω ( SC ⋆ rob P ⋆ min σ 2 (1 -γ ) 2 ε 2 ) samples) are needed to find an ε -optimal robust policy when the uncertainty level σ ≍ log(1 /P ⋆ min ) , regardless of the choice of algorithms in the finite-horizon (resp. infinite-horizon) setting. Hence, this suggests that learning RMDPs is at least as hard as the standard MDP (Li et al., 2022) when the uncertainty level is sufficiently small, and corroborates the near-optimality of DRVI-LCB with respect to all key parameters up to a polynomial factor of the horizon length H (resp. the effective horizon length 1 1 -γ ) for a range of uncertainty levels ( σ ≍ log(1 /P ⋆ min ) ).

To the best of our knowledge, our paper is the first work to execute the principle of pessimism in a data-driven manner for robust offline RL, leading to the first provably efficient algorithm that learns under simultaneous model uncertainty and partial coverage of the history dataset. See Table 1 for a summary.

Comparison with prior art under full coverage. Prior works (Panaganti and Kalathil, 2022; Yang et al., 2022; Zhou et al., 2021) have only addressed the infinite-horizon setting under full coverage of the history data. Fortunately, our results also seamlessly cover this easier scenario, by replacing C ⋆ rob with A . Specializing our result to this setting to facilitate comparison, DRVI-LCB finds an ε -optimal robust policy with at most ˜ O ( SA P ⋆ min (1 -γ ) 4 σ 2 ε 2 ) samples, which depends linearly with respect to the size of the state space S (ignoring other parameters). In contrast, all prior works (Panaganti and Kalathil, 2022; Yang et al., 2022; Zhou et al., 2021) incur sample complexities that scale at least quadratically with respect to the size of the state space S . In addition, our bound improves the exponential dependency on 1 1 -γ of Panaganti and Kalathil (2022); Zhou et al. (2021) to a polynomial dependency, as well as the quadratic dependency on 1 /P min (which satisfies P min ≤ P ⋆ min ) of Yang et al. (2022) to a linear one on 1 /P ⋆ min . These improvements further corroborate the benefit of the proposed DRVI-LCB even under full coverage. See Table 2 for detailed comparisons.

## 1.3 Related works

We shall focus on the closely related works on offline RL and distributionally robust RL.

Offline RL. Focusing on the task of learning an optimal policy from offline data, a significant amount of prior arts sets to understand the sample complexity and efficacy of offline RL under different assumptions of the history dataset. A bulk of prior results requires the history data to cover all the state-action pairs, under assumptions such as uniformly bounded concentrability coefficients (Chen and Jiang, 2019; Munos, 2005) and uniformly lower bounded data visitation distribution (Yin et al., 2021; Yin and Wang, 2021), where the latter assumption is also related to studies of asynchronous Q-learning (Li et al., 2021). More recently, the principle of pessimism has been investigated for offline RL in both model-based (Jin et al., 2021; Li et al., 2022; Rashidinejad et al., 2021; Xie et al., 2021) and model-free algorithms (Kumar et al., 2020; Shi et al., 2022; Yan et al., 2023), without the stringent requirement of full coverage. In particular, Li et al. (2022) established the near-minimax optimality of a pessimistic variant of value iteration under the single-policy clipped concentrability of history data, which inspired our algorithm design in the distributionally robust setting.

Table 1: Our results for finding an ε -optimal robust policy in the infinite/finite-horizon robust MDPs with an uncertainty set measured with respect to the KL divergence using history data under partial coverage. The sample complexities included in the table are valid for sufficiently small ε , with all logarithmic factors omitted. Here, σ is the uncertainty level, S is the size of the state space, H is the horizon length for the finite-horizon setting, γ is the discount factor for the infinite-horizon setting, C ⋆ rob is the robust single-policy clipped concentrability coefficient, and P ⋆ min is the smallest positive state transition probability of the nominal kernel visited by the optimal robust policy π ⋆ .

| Horizon          | Algorithm               | Coverage   | Sample complexity                   | Uncertainty level    |
|------------------|-------------------------|------------|-------------------------------------|----------------------|
| infinite-horizon | DRVI-LCB (this work)    | partial    | SC ⋆ rob P ⋆ min (1 - γ ) 4 σ 2 ε 2 | full range           |
| infinite-horizon | Lower bound (this work) | partial    | SC ⋆ rob (1 - γ ) 3 ε 2             | σ ≲ (1 - γ )         |
| infinite-horizon | Lower bound (this work) | partial    | SC ⋆ rob P ⋆ min (1 - γ ) 2 σ 2 ε 2 | σ ≍ log(1 /P ⋆ min ) |
| finite-horizon   | DRVI-LCB (this work)    | partial    | SC ⋆ rob H 5 P ⋆ min σ 2 ε 2        | full range           |
| finite-horizon   | Lower bound (this work) | partial    | SC ⋆ rob H 4 ε 2                    | σ ≲ 1 /H             |
| finite-horizon   | Lower bound (this work) | partial    | SC ⋆ rob H 3 P ⋆ min σ 2 ε 2        | σ ≍ log(1 /P ⋆ min ) |

Distributionally robust RL. While distributionally robust optimization has been mainly investigated in the context of supervised learning (Bertsimas et al., 2018; Blanchet and Murthy, 2019; Duchi and Namkoong, 2021; Gao, 2022; Rahimian and Mehrotra, 2019; Sinha et al., 2018), distributionally robust dynamic programming has also attracted considerable amount of attention, e.g. Iyengar (2005); Nilim and El Ghaoui (2005); Nilim and Ghaoui (2003); Xu and Mannor (2012), where natural robust extensions to the standard Bellman machineries are developed under mild assumptions. Targeting robust MDPs, empirical and theoretical works have been widely explored under different forms of uncertainty sets (Abdullah et al., 2019; Badrinath and Kalathil, 2021; Derman and Mannor, 2020; Ding et al., 2023; Goyal and Grand-Clement, 2022; Ho et al., 2018, 2021; Hou et al., 2020; Iyengar, 2005; Kaufman and Schaefer, 2013; Smirnova et al., 2019; Song and Zhao, 2020; Tamar et al., 2014; Wang et al., 2022; Wolff et al., 2012; Xu and Mannor, 2012; Yang, 2017). Nonetheless, the majority of prior theoretical analyses focus on planning with an exact knowledge of the uncertainty set (Iyengar, 2005; Tamar et al., 2014; Xu and Mannor, 2012), or are asymptotic in nature (Roy et al., 2017).

Since the first appearance of our paper on arXiv in August 2022, a few more papers have emerged that also tackle the sample complexity of robust RL algorithms. For example, Wang et al. (2023a,b) developed finitesample complexity bounds for robust variants of Q-learning with the generative model when the uncertainty set is measured by KL divergence; in particular, the improved bound of variance-reduced robust Q-learning (Wang et al., 2023b) becomes independent of the size of the uncertainty set when it is sufficiently small with

A number of robust RL algorithms were proposed recently with an emphasis on finite-sample performance guarantees under different data generating mechanisms. Wang and Zou (2021) proposed a robust Q-learning algorithm with an R-contamination uncertain set for the online setting, which achieves a similar bound as its non-robust counterpart. Badrinath and Kalathil (2021) proposed a model-free algorithm for the online setting with linear function approximation to cope with large state spaces. Panaganti and Kalathil (2022); Yang et al. (2022) developed sample complexities for a model-based robust RL algorithm with a variety of uncertainty sets where the data are collected using a generative model. In addition, Zhou et al. (2021) examined the uncertainty set defined by the KL divergence for offline data with uniformly lower bounded data visitation distribution. These works all require full coverage of the state-action space, whereas ours is the first one to leverage the principle of pessimism in robust offline RL.

Table 2: Comparisons between our results and prior arts for finding an ε -optimal robust policy in the infinite/finite-horizon robust MDPs with an uncertainty set measured with respect to the KL divergence under full coverage of the history data. The sample complexities included in the table are valid for sufficiently small ε , with all logarithmic factors omitted. Here, σ is the uncertainty level, S is the size of the state space, A is the size of the action space, H is the horizon length for the finite-horizon setting, γ is the discount factor for the infinite-horizon setting, P ⋆ min is the smallest positive state transition probability of the nominal kernel visited by the optimal robust policy π ⋆ , and P min is the smallest positive state transition probability of the nominal kernel; it holds P min ≤ P ⋆ min .

| Problem type     | Algorithm                                | Coverage   | Sample complexity                              |
|------------------|------------------------------------------|------------|------------------------------------------------|
| infinite-horizon | DRVI (Zhou et al., 2021)                 | full       | S 2 A exp ( O ( 1 1 - γ ) ) (1 - γ ) 4 σ 2 ε 2 |
| infinite-horizon | REVI/DRVI (Panaganti and Kalathil, 2022) | full       | S 2 A exp ( O ( 1 1 - γ ) ) (1 - γ ) 4 σ 2 ε 2 |
| infinite-horizon | DRVI (Yang et al., 2022)                 | full       | S 2 A P 2 min (1 - γ ) 4 σ 2 ε 2               |
| infinite-horizon | DRVI-LCB (this work)                     | full       | SA P ⋆ min (1 - γ ) 4 σ 2 ε 2                  |
| finite-horizon   | DRVI-LCB (this work)                     | full       | SAH 5 P ⋆ min σ 2 ε 2                          |

respect to the minimal support probability of the nominal kernel at a price of worse dependency with 1 /P ⋆ min . Shi et al. (2023) provided near-optimal sample complexity bounds for model-based robust RL algorithms with the generative model when the uncertainty set is measured by the total variation or chi-square distances, which highlighted that different uncertainty sets can lead to drastically different sample complexities, and hence, statistical consequences.

## 1.4 Notation and paper organization

Throughout this paper, we denote by ∆( S ) the probability simplex over a set S , and introduce the notation [ H ] := { 1 , · · · , H } for any positive integer H &gt; 0 . In addition, for any vector x = [ x ( s, a ) ] ( s,a ) ∈S×A ∈ R SA (resp. x = [ x ( s ) ] s ∈S ∈ R S ) that constitutes certain values for each state-action pair (resp. state), we overload the notation by letting x 2 = [ x ( s, a ) 2 ] ( s,a ) ∈S×A (resp. x 2 = [ x ( s ) 2 ] s ∈S ). Moreover, for any two vectors x = [ x i ] 1 ≤ i ≤ n and y = [ y i ] 1 ≤ i ≤ n , the notation x ≤ y (resp. x ≥ y ) means x i ≤ y i (resp. x i ≥ y i ) for all 1 ≤ i ≤ n . Finally, the Kullback-Leibler (KL) divergence for any two distributions P and Q is denoted as KL ( P ∥ Q ) .

The rest of this paper is organized as follows. Section 2 provides the backgrounds and introduces the distributionally robust formulation of finite-horizon MDPs in the offline setting under partial coverage. Section 3 presents the proposed algorithm and provides sample complexity guarantees. Section 4 develops the corresponding results for the infinite-horizon setting. Section 5 demonstrate the performance of the proposed algorithm through numerical experiments. Finally, we conclude in Section 6. The detailed proofs are postponed to the appendix.

## 2 Problem formulation: episodic finite-horizon RMDPs

## 2.1 Basics of finite-horizon episodic tabular MDPs

Consider an episodic finite-horizon MDP, represented by M = ( S , A , H, P := { P h } H h =1 , { r h } H h =1 ) , where S = { 1 , · · · , S } and A = { 1 , · · · , A } are the finite state and action spaces, respectively, H is the horizon length, P h : S ×A → ∆( S ) (resp. r h : S ×A → [0 , 1] ) denotes the probability transition kernel (resp. reward

function) at step h (1 ≤ h ≤ H ) . 1 For any transition kernel P , we introduce the S -dimensional distribution vectors

<!-- formula-not-decoded -->

to represent the probability transition vector in state s when taking action a at step h .

Denote by π = { π h } H h =1 as the policy or action selection rule of an agent, where π h : S → ∆( A ) specifies the action selection probability over the action space; when the policy is deterministic, we slightly abuse the notation and refer to π h ( s ) as the action selected by policy π in state s at step h . The value function V π,P = { V π,P h } H h =1 of policy π with a transition kernel P is defined by

<!-- formula-not-decoded -->

where the expectation is taken over the randomness of the trajectory { s h , a h , r h } H h =1 generated by executing policy π , namely, a t ∼ π t ( s t ) , and s t +1 ∼ P t ( · | s t , a t ) . Similarly, the Q-function Q π,P = { Q π,P h } H h =1 of policy π is defined as

<!-- formula-not-decoded -->

where the expectation is again taken over the randomness of the trajectory.

Moreover, when the initial state s 1 is drawn from a given distribution ρ , let d π,P h ( s | ρ ) and d π,P h ( s, a | ρ ) denote respectively the state occupancy distribution and the state-action occupancy distribution induced by π at time step h ∈ [ H ] . In particular, we often dropped the dependency with respect to ρ whenever it is clear from the context, by simply writing d π,P h ( s ) := d π,P h ( s | ρ ) and d π,P h ( s, a ) := d π,P h ( s, a | ρ ) , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which are conditioned on s 1 ∼ ρ and the event that all actions and states are drawn according to policy π and transition kernel P .

## 2.2 Distributionally robust MDPs

In this section, we focus on finite-horizon episodic distributionally robust MDPs (RMDPs), denoted by M rob = ( S , A , H, U σ ( P 0 ) , { r h } H h =1 ) . Different from standard MDPs, we now consider an ensemble of probability transition kernels or models within an uncertainty set centered around a nominal one P 0 = { P 0 h } H h =1 , where the distance between the transition kernels is measured in terms of the Kullback-Leibler (KL) divergence. Specifically, given an uncertainty level σ &gt; 0 , the uncertainty set around P 0 , which satisfies the so-called ( s, a ) -rectangularity condition (Wiesemann et al., 2013), is specified as

<!-- formula-not-decoded -->

where ⊗ denote the Cartesian product. In words, the KL divergence between the true transition probability vector and the nominal one at each state-action pair is at most σ ; moreover, the RMDP reduces to the standard MDP when σ = 0 .

Instead of evaluating a policy in a fixed MDP, the performance of a policy in the RMDP is evaluated based on its worst-case-i.e., smallest-value function over all the instances in the uncertainty set. That is, we define the robust value function V π,σ = { V π,σ h } H h =1 and the robust Q-function Q π,σ = { Q π,σ h } H h =1 respectively as

<!-- formula-not-decoded -->

where the infimum is taken over the uncertainty set of transition kernels.

1 Without loss of generality, we assume the reward function is deterministic, fixed, and normalized to be within [0 , 1] ; it is straightforward to generalize our framework to incorporate random rewards with uncertainties.

Optimal robust policy. For finite-horizon RMDPs, it has been established that there exists at least one deterministic policy that maximizes the robust value function and the robust Q-function simultaneously (Iyengar, 2005; Nilim and El Ghaoui, 2005). In view of this, we shall denote a deterministic policy π ⋆ = { π ⋆ h } H h =1 as an optimal robust policy throughout this paper. The resulting optimal robust value function V ⋆,σ = { V ⋆,σ h } H h =1 and optimal robust Q-function Q ⋆,σ = { Q ⋆,σ h } H h =1 are denoted by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar to (4), we adopt the following short-hand notation for the occupancy distributions associated with the optimal policy:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Robust Bellman equations. It turns out the Bellman's principle of optimality can be extended naturally to its robust counterpart (Iyengar, 2005; Nilim and El Ghaoui, 2005), which plays a fundamental role in solving the RMDP. To begin with, for any policy π , the robust value function and robust Q-function satisfy the following robust Bellman consistency equation :

<!-- formula-not-decoded -->

Additionally, the optimal robust Q-function obeys the robust Bellman optimality equation :

<!-- formula-not-decoded -->

which can be solved efficiently via a robust variant of value iteration when the RMDP is known (Iyengar, 2005; Nilim and El Ghaoui, 2005).

## 2.3 Distributionally robust offline RL

Let D be a history/batch dataset, which consists of a collection of K independent episodes generated based on executing a behavior policy π b = { π b h } H h =1 in some nominal MDP M 0 = ( S , A , H, P 0 := { P 0 h } H h =1 , { r h } H h =1 ) . More specifically, for 1 ≤ k ≤ K , the k -th episode ( s k 1 , a k 1 , . . . , s k H , a k H , s k H +1 ) is generated according to

<!-- formula-not-decoded -->

Throughout the paper, ρ b represents for some initial distribution associated with the history dataset. Then, we introduce the following short-hand notation for the occupancy distribution w.r.t. π b :

<!-- formula-not-decoded -->

Goal. With the history dataset D in hand, our goal is to find a near-optimal robust policy ̂ π , which satisfies

<!-- formula-not-decoded -->

using as few samples as possible, where ε is the target accuracy level, and

<!-- formula-not-decoded -->

are evaluated when the initial state s 1 is drawn from a given distribution ρ .

Robust single-policy clipped concentrability. To quantify the quality of the history dataset to achieve the set goal, it is desirable to capture the distribution mismatch between the history dataset and the desired ones, inspired by the single-policy clipped concentrability assumption recently proposed by Li et al. (2022), we introduce a tailored assumption for robust MDPs as follows.

Assumption 1 (Robust single-policy clipped concentrability) . The behavior policy of the history dataset D satisfies

<!-- formula-not-decoded -->

for some quantity C ⋆ rob ∈ [ 1 S , ∞ ] . Here, we take C ⋆ rob to be the smallest quantity satisfying (14) , and refer to it as the robust single-policy clipped concentrability coefficient. In addition, we follow the convention 0 / 0 = 0 .

In words, C ⋆ rob measures the worst-case discrepancy-between the optimal robust policy π ⋆ (from initial state distribution ρ ) in any model P ∈ U σ ( P 0 ) within the uncertainty set and the behavior policy π b (from initial state distribution ρ b ) in the nominal model P 0 -in terms of the clipped maximum density ratio of the state-action occupancy distributions.

- Distribution shift. When the uncertainty level σ = 0 , Assumption 1 reduces back to the single-policy clipped concentrability in Li et al. (2022) for standard offline RL, a weaker notion that can be S times smaller than the single-policy concentrability adopted in, e.g., Rashidinejad et al. (2021); Shi et al. (2022); Xie et al. (2021). On the other end, whenever σ &gt; 0 , the proposed robust single-policy clipped concentrability accounts for the distribution shift not only due to the policies in use ( π ⋆ versus π b ) with respect to the respective initial state distributions, but also the underlying environments ( P ∈ U σ ( P 0 ) versus P 0 ), and therefore, is generally larger than that in the non-robust counterpart.
- Partial coverage. As long as C ⋆ rob is finite, i.e., C ⋆ rob &lt; ∞ , it admits the scenarios when the history dataset only provides partial coverage over the entire state-action space, as long as the behavior policy π b visits the state-action pairs that are visited by the optimal robust policy π ⋆ under at least one model in the uncertainty set.

Remark 1 . To facilitate comparison with prior works assuming full coverage, we can bound C ⋆ rob when the batch dataset is generated using a simulator (Panaganti and Kalathil, 2022; Yang et al., 2022); namely, we can generate sample state transitions based on the transition kernel of the nominal MDP for all state-action pairs at all time steps. In this case, it amounts to that d b h ( s, a ) = 1 SA for all ( s, a, h ) ∈ S × A × [ H ] , which directly leads to the bound C ⋆ rob = max ( s,a,h,P ) ∈S×A× [ H ] ×U σ ( P 0 ) min { d ⋆,P h ( s,a ) , 1 S } d b h ( s,a ) ≤ 1 /S 1 / ( SA ) = A .

## 3 Algorithm and theory: episodic finite-horizon RMDPs

In this section, we present a model-based algorithm-namely DRVI-LCB -for robust offline RL in the finitehorizon setting, along with its performance guarantees.

## 3.1 Building an empirical nominal MDP

For a moment, imagine we have access to N independent sample transitions D 0 := { ( h i , s i , a i , s ′ i ) } N i =1 drawn from the transition kernel P 0 of the nominal MDP M 0 , where each sample ( h i , s i , a i , s ′ i ) indicates the transition from state s i to state s ′ i when action a i is taken at step h i , drawn according to s ′ i ∼ P 0 h i ( · | s i , a i ) . It is then natural to build an empirical estimate ̂ P 0 = { ̂ P 0 h } H h =1 of P 0 based on the empirical frequencies of state transitions, where

<!-- formula-not-decoded -->

Algorithm 1: Two-fold subsampling trick for the finite-horizon setting.

- 1 input: a dataset D , probability δ .
- 2 data splitting: split D into two D main and D aux , where each contain K/ 2 trajectories.
- 3 lower bounding the number of transitions in D main : denote the number of transitions from state s at step h in D main (resp. D aux ) as N main h ( s ) (resp. N aux h ( s ) ), construct

<!-- formula-not-decoded -->

- 4 generate the subsampled dataset D trim : randomly sample the transitions (i.e., the quadruples taking the form ( s, a, h, s ′ ) ) from D main uniformly at random, such that for each ( s, h ) ∈ S × [ H ] , D trim contains min { N trim h ( s ) , N main h ( s ) } sample transitions.
- 5 output: set D 0 = D trim .

for any ( h, s, a, s ′ ) ∈ [ H ] ×S × A × S . Here, N h ( s, a ) denotes the total number of sample transitions from ( s, a ) at step h as

<!-- formula-not-decoded -->

While it is possible to directly break down the history dataset D into sample transitions, unfortunately, the sample transitions from the same episode are not independent, significantly hurdling the analysis. To alleviate this, Li et al. (2022, Algorithm 2) proposed a simple two-fold subsampling scheme to preprocess the history dataset D and decouple the statistical dependency, resulting into a distributionally equivalent dataset D 0 with independent samples; for completeness, we provide the procedure in Algorithm 1. We have the following lemma paraphrased from Li et al. (2022) for the obtained dataset D 0 .

Lemma 1 ((Li et al., 2022)) . With probability at least 1 -8 δ , the output dataset from the two-fold subsampling scheme in Li et al. (2022) is distributionally equivalent to D 0 , where { N h ( s, a ) } are independent of the sample transitions in D 0 and obey

<!-- formula-not-decoded -->

for all ( h, s, a ) ∈ [ H ] ×S × A .

Therefore, by invoking the two-fold sampling trick from Li et al. (2022), it is sufficient to treat the dataset D 0 with independent samples onwards with Lemma 1 in place, which greatly simplifies the analysis.

## 3.2 DRVI-LCB : a pessimistic variant of robust value iteration

Armed with the estimate ̂ P 0 of the nominal transition kernel P 0 , we are positioned to introduce our algorithm DRVI-LCB , summarized in Algorithm 2.

Distributionally robust value iteration. Before proceeding, let us recall the update rule of the classical distributionally robust value iteration ( DRVI ), which serves as the basis of our algorithmic development. Given an estimate of the nominal MDP ̂ P 0 and the radius σ of the uncertainty set, DRVI updates the robust value functions according to

<!-- formula-not-decoded -->

```
Algorithm 2: Robust value iteration with LCB ( DRVI-LCB ) for robust offline RL. 1 input: a dataset D 0 ; reward function r ; uncertainty level σ . 2 initialization: ̂ Q H +1 = 0 , ̂ V H +1 = 0 . 3 for h = H, · · · , 1 do 4 Compute the empirical nominal transition kernel ̂ P 0 h according to (15); 5 for s ∈ S , a ∈ A do 6 Compute the penalty term b h ( s, a ) according to (22); 7 Set ̂ Q h ( s, a ) according to (21); 8 for s ∈ S do 9 Set ̂ V h ( s ) = max a ̂ Q h ( s, a ) and ̂ π h ( s ) = arg max a ̂ Q h ( s, a ) ; 10 output: ̂ π = { ̂ π h } 1 ≤ h ≤ H .
```

which works backwards from h = H to h = 1 , with the terminal condition ̂ Q H +1 = 0 . Due to strong duality (Hu and Hong, 2013), the update rule of the robust Q-functions in (19) can be equivalently reformulated in its dual form as

<!-- formula-not-decoded -->

which can be solved efficiently (Iyengar, 2005; Panaganti and Kalathil, 2022; Yang et al., 2022).

Our algorithm DRVI-LCB . Motivated by the principle of pessimism in standard offline RL (Jin et al., 2021; Li et al., 2022; Rashidinejad et al., 2021; Xie et al., 2021), we propose to perform a pessimistic variant of DRVI , where the update rule of DRVI-LCB at step h is modified as

<!-- formula-not-decoded -->

Here, the robust Q -function estimate is adjusted by subtracting a carefully designed data-driven penalty term b h ( s, a ) that measures the uncertainty of the value estimates. Specifically, for some δ ∈ (0 , 1) and any ( s, a, h ) ∈ S × A × [ H ] , the penalty term b h ( s, a ) is defined as

<!-- formula-not-decoded -->

where c b is some universal constant, and

<!-- formula-not-decoded -->

The penalty term is novel and different from the one used in standard (no-robust) offline RL (Jin et al., 2021; Li et al., 2022; Rashidinejad et al., 2021; Shi et al., 2022; Xie et al., 2021), by taking into consideration the unique problem structure pertaining to robust MDPs. In particular, it tightly upper bounds the statistical uncertainty which carries a non-linear and implicit dependency w.r.t. the estimated nominal transition kernel induced by the uncertainty set U ( P 0 ) , addressing unique challenges not present for the standard MDP case.

## 3.3 Performance guarantees

Before stating the main theorems, let us first introduce several important metrics.

- P ⋆ min , which only depends on the state-action pairs covered by the optimal robust policy π ⋆ under the nominal model P 0 :

<!-- formula-not-decoded -->

In words, P ⋆ min is the smallest positive state transition probability of the optimal robust policy π ⋆ under the nominal kernel P 0 .

- Similarly, we introduce P b min which only depends on the state-action pairs covered by the behavior policy π b under the nominal model P 0 :

<!-- formula-not-decoded -->

In words, P b min is the smallest positive state transition probability of the behavior policy π b under the nominal kernel P 0 .

- Finally, let d b min denote the smallest positive state-action occupancy distribution of the behavior policy π b under the nominal model P 0 :

<!-- formula-not-decoded -->

We are now positioned to present the performance guarantees of DRVI-LCB for robust offline RL.

Theorem 1. Given an uncertainty level σ &gt; 0 , suppose that the penalty terms in Algorithm 2 are chosen as (22) for sufficiently large c b . With probability at least 1 -δ , the output ̂ π of Algorithm 2 obeys

<!-- formula-not-decoded -->

as long as the number of episodes K satisfies

<!-- formula-not-decoded -->

where c 0 and c 1 are some sufficiently large universal constants.

Our theorem is the first to characterize the sample complexities of robust offline RL under partial coverage , to the best of our knowledge (cf. Table 2). Theorem 1 shows that DRVI-LCB finds an ε -optimal robust policy as soon as the sample size T = KH is above the order of

<!-- formula-not-decoded -->

up to some logarithmic factor, where the burn-in cost is independent of the accuracy level ε . For sufficiently small accuracy level ε , this results in a sample complexity of

<!-- formula-not-decoded -->

Our theorem suggests that the sample efficiency of robust offline RL critically depends on the problem structure of the given RMDP (i.e. coverage of the optimal robust policy π ⋆ as measured by P ⋆ min ) as well as the quality of the history dataset (as measured by C ⋆ rob ). Given that C ⋆ rob can be as small as on the order of 1 /S , the sample complexity requirement can exhibit a much weaker dependency with the size of the state space S .

On the flip side, to assess the optimality of Theorem 1, we develop an information-theoretic lower bound for robust offline RL as provided in the following theorem.

Theorem 2. For any ( H,S,C,P ⋆ min , σ, ε ) obeying H ≥ 2 e 8 , C ≥ 4 /S , P ⋆ min ∈ (0 , 1 H ] , and ε ≤ H 384 e 6 log(1 /P ⋆ min ) , we can construct a collection of finite-horizon RMDPs {M θ | θ ∈ Θ } , an initial state distribution ρ , and a batch dataset with K independent sample trajectories each with length H satisfying 2 C ≤ C ⋆ rob ≤ 4 C , such that provided that

<!-- formula-not-decoded -->

Here, c 1 &gt; 0 is some universal constant, the infimum is taken over all estimators ̂ π , and P θ denotes the probability when the RMDP is M θ .

The messages of Theorem 2 are two-fold.

- When the uncertainty level σ ≲ 1 /H is relatively small, Theorem 2 shows that no algorithm can succeed in finding an ε -optimal robust policy when the sample complexity falls below the order of

<!-- formula-not-decoded -->

which is at least as large as the sample complexity requirement of non-robust offline RL (Li et al., 2022). Consequently, this leads to new insights regarding the statistical hardness of learning robust RMDPs with the KL uncertainty set: it can be at least as hard as the standard MDPs (which corresponds to σ = 0 ), for sufficiently small uncertainty levels.

- When the uncertainty level σ ≍ log(1 /P ⋆ min ) , Theorem 2 shows that no algorithm can succeed in finding an ε -optimal robust policy when the sample complexity falls below the order of

<!-- formula-not-decoded -->

which confirms the near-optimality of DRVI-LCB up to a factor of H 2 ignoring logarithmic factors. Therefore, DRVI-LCB is the first provable algorithm for robust offline RL with a near-optimal sample complexity without requiring the stringent full coverage assumption.

## 4 Robust offline RL for discounted infinite-horizon RMDPs

In this section, we turn to the studies of robust offline RL for discounted infinite-horizon MDPs.

## 4.1 Backgrounds on discounted infinite-horizon RMDPs

Similar to the finite-horizon setting, we consider the discounted infinite-horizon robust MDPs (RMDPs) represented by M rob = {S , A , γ, U σ ( P 0 ) , r } . Here, S = { 1 , 2 , · · · , S } is the state space, A = { 1 , 2 , · · · , A } is the action space, γ ∈ [0 , 1) is the discounted factor, and r : S × A → [0 , 1] is the intermediate reward function. Different from the standard MDPs, U σ ( P 0 ) denote the set of possible transition kernels within an uncertainty set centered around a nominal kernel P 0 : S × A → ∆( S ) using the distance measured in terms of the KL divergence. In particular, given an uncertainty level σ &gt; 0 , the uncertainty set around P 0 is specified as

<!-- formula-not-decoded -->

where we denote a vector of the transition kernel P or P 0 at ( s, a ) respectively as

<!-- formula-not-decoded -->

Note that at any time step, the adversary of the nature chooses a history-independent component within the fixed uncertainty set U σ ( P 0 s,a ) defined in (30), conditioned only on the current state-action pair ( s, a ) . This is to ensure the computation tractability of finding such adversary.

<!-- formula-not-decoded -->

Policy and robust value/Q functions. A (possibly random) stationary policy π : S → ∆( A ) represents the selection rule of the agent, namely, π ( a | s ) denote the probability of choosing a in state s . With some abuse of notation, let π ( s ) represent the action chosen by π when π is a deterministic policy. We define the robust value function V π,σ and robust Q-function Q π,σ respectively as

<!-- formula-not-decoded -->

where the value function V π,P and Q-function Q π,P w.r.t. policy π and transition kernel P are defined respectively by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the expectation is taken over the randomness of the trajectory. In words, the robust value/Q functions characterize the worst case over all the instances in the uncertainty set.

Optimal policy and robust Bellman equation. Similar to the finite-horizon RMDPs, it is well-known that there exists at least one deterministic policy that maximizes the robust value function and Q-function simultaneously in the infinite-horizon setting as well (Iyengar, 2005; Nilim and El Ghaoui, 2005). With this in mind, we denote the optimal policy as π ⋆ and the corresponding optimal robust value function (resp. optimal robust Q-function ) as V ⋆,σ (resp. Q ⋆,σ ), namely

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, we continue to admit the Bellman's optimality principle, resulting in the following robust Bellman consistency equation (resp. robust Bellman optimality equation ):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Occupancy distributions. To begin, let ρ be some initial state distribution. We denote d π,P ( s | ρ ) and d π,P ( s, a | ρ ) respectively as the state occupancy distribution and the state-action occupancy distribution induced by policy π , namely

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the occupancy distributions are conditioned on s 0 ∼ ρ and the sequence of actions and states are generated based on policy π and transition kernel P . Next, applying (38) with π = π ⋆ , we adopt the the following short-hand notation for the occupancy distributions associated with the optimal policy:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 4.2 Data collection and constructing the empirical MDP

Suppose that we observe a batch/history dataset D = { ( s i , a i , s ′ i ) } 1 ≤ i ≤ N consisting of N sample transitions. These transitions are independently generated, where the state-action pair is drawn from some behavior distribution d b ∈ ∆( S × A ) , followed by a next state drawn over the nominal transition kernel P 0 , i.e.,

<!-- formula-not-decoded -->

Armed with these, we are ready to introduce the goal in the infinite-horizon setting. Given the history dataset D obeying Assumption 2, for some target accuracy ε &gt; 0 , we aim to find a near-optimal robust policy ̂ π , which satisfies

<!-- formula-not-decoded -->

in a sample-efficient manner for some initial state distribution ρ .

Remark 2 . For simplicity, we limit ourselves to the case when the history dataset consists of independent sample transitions. It is not difficult to generalize to the Markovian data case, when we only have access to a single trajectory of data generated by following some behavior policy, by combining the two-fold subsampling trick in Li et al. (2022, Appendix D) with our analysis. We leave this extension to interested readers.

Similar to Assumption 1, we design the following robust single-policy clipped concentrability assumption tailored for infinite-horizon RMDPs to characterize the quality of the history dataset.

Assumption 2 (Robust single-policy clipped concentrability for infinite-horizon MDPs) . The behavior policy of the history dataset D satisfies

<!-- formula-not-decoded -->

for some finite quantity C ⋆ rob ∈ [ 1 S , ∞ ) . Following the convention 0 / 0 = 0 , we denote C ⋆ rob to be the smallest quantity satisfying (42) , and refer to it as the robust single-policy clipped concentrability coefficient.

Remark 3 . Similar to Remark 1, we can bound C ⋆ rob ≤ A when the batch dataset is generated using a simulator (Panaganti and Kalathil, 2022; Yang et al., 2022). By combining this bound of C ⋆ rob with the theoretical guarantees developed momentarily in Theorem 3, we obtain the comparison in Table 2.

Building an empirical nominal MDP Recalling that we have N independent samples in the dataset D = { ( s i , a i , s ′ i ) } 1 ≤ i ≤ N . First, we denote N ( s, a ) as the total number of sample transitions from any stateaction pair ( s, a ) as

<!-- formula-not-decoded -->

Armed with N ( s, a ) , we construct the empirical estimate ̂ P 0 of the nominal kernel P 0 by the visiting frequencies of state-action pairs as follows:

<!-- formula-not-decoded -->

for any ( s, a, s ′ ) ∈ S × A × S .

## 4.3 DRVI-LCB for discounted infinite-horizon RMDPs

With the estimate ̂ P 0 of the nominal transition kernel P 0 in hand, we are positioned to introduce our algorithm DRVI-LCB for infinite-horizon RMDPs, which bears some similarity with the finite-horizon version (cf. Algorithm 2), by taking the uncertainties of the value estimates into consideration throughout the value iterations. The procedure is summarized in Algorithm 3.

- Algorithm 3: Robust value iteration with LCB ( DRVI-LCB ) for infinite-horizon RMDPs. 1 input: a dataset D ; reward function r ; uncertainty level σ ; number of iterations M . 2 initialization: ̂ Q 0 ( s, a ) = 0 , ̂ V 0 ( s ) = 0 for all ( s, a ) ∈ S × A . 3 Compute the empirical nominal transition kernel ̂ P 0 according to (44); 4 Compute the penalty term b ( s, a ) according to (48); 5 for m = 1 , 2 , · · · , M do 6 for s ∈ S , a ∈ A do 7 Set ̂ Q m ( s, a ) according to (51); 8 for s ∈ S do 9 Set ̂ V m ( s ) = max a ̂ Q m ( s, a ) ; 10 output: ̂ π s.t. ̂ π ( s ) = arg max a ̂ Q M ( s, a ) for all s ∈ S .

The pessimistic robust Bellman operator. At the core of DRVI-LCB is a pessimistic variant of the classical robust Bellman operator in the infinite-horizon setting (Iyengar, 2005; Nilim and El Ghaoui, 2005; Zhou et al., 2021), denoted as T σ ( · ) : R SA → R SA , which we recall as follows:

<!-- formula-not-decoded -->

Encouragingly, the robust Bellman operator shares the nice γ -contraction property of the standard Bellman operator, ensuring fast convergence of robust value iteration by applying the robust Bellman operator (45) recursively. In the robust offline setting, instead of recursing using the population robust Bellman operator, we need to construct a pessimistic variant of the robust Bellman operator ̂ T σ pe ( · ) w.r.t. the empirical nominal kernel ̂ P 0 as follows:

<!-- formula-not-decoded -->

where b ( s, a ) denotes the penalty term that measures the data-dependent uncertainty of the value estimates. To specify the tailored penalty term b ( s, a ) in (46), we first introduce an additional term

<!-- formula-not-decoded -->

which in words represents the smallest positive transition probability of the estimated nominal kernel ̂ P 0 ( s ′ | s, a ) . Then for some δ ∈ (0 , 1) , some universal constant c b &gt; 0 , b ( s, a ) is defined as

<!-- formula-not-decoded -->

As shall be illuminated, our proposed pessimistic robust Bellman operator ̂ T σ pe ( · ) (cf. (46)) plays an important role in DRVI-LCB . Encouragingly, despite the additional data-driven penalty term b ( s, a ) , it still enjoys the celebrated γ -contractive property, which greatly facilitates the analysis. Before continuing, we summarize the γ -contraction property below, whose proof is postponed to Appendix C.1.

Lemma 2 ( γ -Contraction) . For any γ ∈ [0 , 1) , the operator ̂ T σ pe ( · ) (cf. (46) ) is a γ -contraction w.r.t. ∥·∥ ∞ . Namely, for any Q 1 , Q 2 ∈ R SA s.t. Q 1 ( s, a ) , Q 2 ( s, a ) ∈ [ 0 , 1 1 -γ ] for all ( s, a ) ∈ S × A , one has

<!-- formula-not-decoded -->

Additionally, there exists a unique fixed point ̂ Q ⋆,σ pe of the operator ̂ T σ pe ( · ) obeying 0 ≤ ̂ Q ⋆,σ pe ( s, a ) ≤ 1 1 -γ for all ( s, a ) ∈ S × A .

Our algorithm DRVI-LCB for infinite-horizon robust offline RL. Armed with the γ -contraction property of the pessimistic robust Bellman operator ̂ T σ pe ( · ) , we are positioned to introduce DRVI-LCB for infinite-horizon RMDPs, summarized in Algorithm 3. Specifically, DRVI-LCB can be seen as a value iteration algorithm w.r.t. ̂ T σ pe ( · ) (cf. (46)), whose update rule at the m -th iteration can be formulated as

<!-- formula-not-decoded -->

and ̂ V m ( s ) = max a ̂ Q m ( s, a ) for all m = 1 , 2 , · · · , M . In view of strong duality (Hu and Hong, 2013), the above convex problem can be translated into a dual formulation, leading to the following equivalent update rule:

<!-- formula-not-decoded -->

which can be solved efficiently (Iyengar, 2005; Panaganti and Kalathil, 2022; Yang et al., 2022) as a onedimensional optimization problem.

To finish the description, we initialize the estimates of Q-function ( ̂ Q 0 ) and value function ( ̂ V 0 ) to be zero and output the greedy policy of the final Q-estimates ( ̂ Q M ) as the final policy ̂ π , namely,

<!-- formula-not-decoded -->

It turns out that the iterates { ̂ Q m } m ≥ 0 of DRVI-LCB converge linearly to the fixed point ̂ Q ⋆,σ pe owing to the nice γ -contraction property outlined in Lemma 2. This fact is summarized in the following lemma, whose proof is postponed to Appendix C.2.

Lemma 3. Let ̂ Q 0 = 0 . The iterates of Algorithm 3 obey

<!-- formula-not-decoded -->

## 4.4 Performance guarantees

Before introducing the main theorems, we first define several essential metrics.

- d b min : the smallest positive entry of the distribution d b , i.e.,

<!-- formula-not-decoded -->

- P b min : the smallest positive state transition probability under the nominal kernel P 0 in the region covered by dataset D , i.e.,

<!-- formula-not-decoded -->

Note that P b min is determined only by the state-action pairs covered by the batch dataset D .

- P ⋆ min : the smallest positive state transition probability of the optimal robust policy π ⋆ under the nominal kernel P 0 , namely

<!-- formula-not-decoded -->

We also note that P ⋆ min is determined only by the state-action pairs covered by the optimal robust policy π ⋆ under the nominal model P 0 .

We are now positioned to introduce the sample complexity upper bound of DRVI-LCB , together with the minimax lower bound, for solving infinite-horizon RMDPs. First, we present the performance guarantees of DRVI-LCB for robust offline RL in the infinite-horizon case, with the proof deferred to Appendix C.3.

Theorem 3. Let c 0 and c 1 be some sufficiently large universal constants. Given an uncertainty level σ &gt; 0 , suppose that the penalty terms in Algorithm 3 are chosen as (48) for sufficiently large c b . With probability at least 1 -δ , the output ̂ π of Algorithm 3 obeys

<!-- formula-not-decoded -->

as long as the number of samples N satisfies

<!-- formula-not-decoded -->

The result directly indicates that DRVI-LCB can finds an ε -optimal policy as long as the sample size in dataset D exceeds the order of (ignoring logarithmic factors)

<!-- formula-not-decoded -->

Note that the burn-in cost is independent with the accuracy level ε , which tells us that the sample complexity is no more than

<!-- formula-not-decoded -->

as long as ε is small enough. The sample complexity of DRVI-LCB still dramatically outperforms prior works under full coverage, which has been compared in detail in Section 1.2 (cf. Table 2). In particular, our sample complexity produces an exponential improvement over Panaganti and Kalathil (2022); Zhou et al. (2021) in terms of the dependency with the effective horizon 1 1 -γ , which is especially significant for long-horizon problems. Compared with Yang et al. (2022), our sample complexity is better by at least a factor of S/P ⋆ min . To achieve the claimed bound, we resort to a delicate technique called the leave-one-out analysis (Agarwal et al., 2020; Li et al., 2022, 2020), by carefully designing an auxiliary set of RMDPs to decouple the statistical dependency introduced across the iterates of pessimistic robust value iteration. This is the first time that the leave-one-out analysis is applied to understanding the sample efficiency of model-based robust RL algorithms, which is of potential independent interest to tighten the sample complexity of other robust RL problems.

To complement the upper bound, we develop an information-theoretic lower bound for robust offline RL as provided in the following theorem whose proof can be found in Appendix C.4.

Theorem 4. For any ( S, P ⋆ min , C ⋆ rob , γ, σ, ε ) obeying 1 1 -γ ≥ 2 e 8 , P ⋆ min ∈ ( 0 , 1 -γ ] , S ≥ log ( 1 /P ⋆ min ) , C ⋆ rob ≥ 8 /S , and ε ≤ 1 384 e 6 (1 -γ ) log ( 1 /P ⋆ min ) , we can construct a collection of infinite-horizon RMDPs {M θ | θ ∈ Θ } , an initial state distribution ρ , and a batch dataset with N independent samples, such that

<!-- formula-not-decoded -->

provided that

<!-- formula-not-decoded -->

Here, c 1 &gt; 0 is some universal constant, the infimum is taken over all estimators ̂ π , and P θ denotes the probability when the RMDP is M θ .

Similar to the finite-horizon setting, the messages of Theorem 4 are two-fold.

- When the uncertainty level σ ≲ 1 -γ is relatively small, Theorem 4 shows that no algorithm can succeed in finding an ε -optimal robust policy when the sample complexity falls below the order of

<!-- formula-not-decoded -->

which is at least as large as the sample complexity requirement of non-robust offline RL (Li et al., 2022). Consequently, this again suggests that learning robust RMDPs with the KL uncertainty set can be at least as hard as the standard MDPs (which corresponds to σ = 0 ), for sufficiently small uncertainty levels.

- When the uncertainty level σ ≍ log(1 /P ⋆ min ) , Theorem 4 shows that no algorithm can succeed in finding an ε -optimal robust policy when the sample complexity falls below the order of

<!-- formula-not-decoded -->

which directly confirms that DRVI-LCB is near-optimal up to a polynomial factor of the effective horizon length 1 1 -γ (cf. (59)). To the best of our knowledge, DRVI-LCB is the first provable algorithm with near-optimal sample complexity for infinite-horizon robust offline RL. Moreover, the requirement imposed on the history dataset is also much weaker than prior literature on robust offline RL (Yang et al., 2022; Zhou et al., 2021), without the need of full coverage of the state-action space.

## 5 Numerical experiments

We conduct experiments on the gambler's problem (Sutton and Barto, 2018; Zhou et al., 2021) to evaluate the performance of the proposed algorithm DRVI-LCB , with comparisons to the robust value iteration algorithm DRVI without pessimism (Panaganti and Kalathil, 2022). Our code can be accessed at:

https://github.com/Laixishi/Robust-RL-with-KL-divergence .

Gambler's problem. In the gambler's game (Sutton and Barto, 2018; Zhou et al., 2021), a gambler bets on a sequence of coin flips, winning the stake with heads and losing with tails. Starting from some initial balance, the game ends when the gambler's balance either reaches 50 or 0 , or the total number of bets H is hit. This problem can be formulated as an episodic finite-horizon MDP, with a state space S = { 0 , 1 , · · · , 50 } and the associated possible actions a ∈ { 0 , 1 , · · · , min { s, 50 -s } } at state s . Here, we set the horizon length H = 100 . Moreover, the parameter of the transition kernel, which is the probability of heads for the coin flip, is fixed as p head and remains the same in all time steps h ∈ [ H ] . The reward is set as 1 when the state reaches s = 50 and 0 for all other cases. In addition, suppose the initial state (i.e., the gambler's initial balance) distribution ρ is taken uniformly at random within S .

The benefit of pessimism. We first utilize a history dataset with N independent samples per state-action pair and time step, generated from the nominal MDP with p 0 head = 0 . 6 . We evaluate the performance of the learned policy ̂ π using our proposed method DRVI-LCB with comparison to DRVI without pessimism, where we fix the uncertainty level σ = 0 . 1 for learning the robust optimal policy. The experiments are repeated 10 times with the average and standard deviations reported. To begin with, Figure 1(a) plots the sub-optimality value gap V ⋆,σ 1 ( s ) -V ̂ π,σ 1 ( s ) for every s ∈ S , when a sample size N = 100 is used to learn the robust policies. It is shown that DRVI-LCB outperform the baseline DRVI uniformly over the state space when the sample size is small, corroborating the benefit of pessimism in the sample-starved regime. Furthermore, Figure 1(b) shows the sub-optimality gap V ⋆,σ 1 ( ρ ) -V ̂ π,σ 1 ( ρ ) with varying sample sizes N = 100 , 300 , 1000 , 3000 , 5000 , where the initial test distribution ρ is generated randomly. 2 While the performance of DRVI-LCB and DRVI both improves with the increase of the sample size, the proposed algorithm DRVI-LCB achieves much better performance with fewer samples.

2 The probability distribution vector ρ ∈ ∆( S ) is generated as ρ ( s ) = u s / ∑ s ∈S u s , where u s is drawn independently from a uniform distribution.

Figure 1: Performance of the proposed algorithm DRVI-LCB using independent samples per state-action pair and time step, where it shows better sample efficiency than the baseline algorithm DRVI without pessimism, as well as better robustness in the learned policy compare to its non-robust counterpart.

<!-- image -->

The benefit of distributional robustness. To corroborate the benefit of distributional robustness, we evaluate the performance of the policy learned from N = 1000 samples using DRVI-LCB on perturbed environments with varying model parameters p head ∈ [0 . 25 , 0 . 75] . We measure the practical performance based on the ratio of winning (i.e., reaching the state s = 50 ) calculated from 3000 episodes. Figure 1(c) illustrates the ratio of winning against the test probability of heads for the policies learned from DRVI-LCB with σ = 0 . 01 and σ = 0 . 2 , which are benchmarked against the non-robust optimal policy of the nominal MDP using the exact model. It can be seen that the policies learned from DRVI-LCB deviate from the non-robust optimal policy as σ increases, which achieves better worst-case rates of winning across a wide range of perturbed environments. On the other end, while the non-robust policy maximizes the performance when the test environment is close to the history one used for training, its performance degenerates to be much worse than the robust policies when the probability of heads is mismatched significantly, especially when p head drops below, say around, 0 . 5 .

Impact of the number of states. We evaluate the performance of DRVI-LCB and DRVI when the number of states varies within [25 , 50 , 100 , 150 , 200 , 250] , using a fixed sample size N = 300 . Figure 1(d) show that DRVI-LCB performs consistently better than DRVI as the number of states S increases, where the value gap exhibits a linear scaling with respect to the number of states.

Performance using trajectory data. Instead of using independent samples, we now evaluate the proposed algorithm using a dataset consisting of K sample trajectories generated from a uniform random policy, for the same setting of Figure 1(b). Figure 2 shows the sub-optimality value gap with respect to the number of trajectories K , where the performance of both DRVI-LCB and DRVI improves as K increases, and

Figure 2: Performance of the proposed algorithm DRVI-LCB compared against DRVI using trajectory data.

<!-- image -->

DRVI-LCB achieves better performance especially when K is small, consistent with the observation under independent data.

## 6 Conclusion

To accommodate both model robustness and sample efficiency, this paper proposes a distributionally robust model-based algorithm for offline RL with the principle of pessimism. We study the finite-sample complexity of the proposed algorithm DRVI-LCB , and establish an information-theoretic lower bound to benchmark its near-optimality for a range of uncertainty levels. Numerical experiments are provided to demonstrate the efficacy of the proposed algorithm. To the best our knowledge, this provides the first provably near-optimal robust offline RL algorithm that learns under model perturbation and partial coverage. This work opens up several interesting directions.

- Tightening the gap between upper and lower bounds. Our upper and lower bounds still leave room for future improvements. For example, it is yet to establish the information-theoretic lower bound over the full range of the uncertainty set, and close the gap between the upper and lower bounds with respect to the horizon length.
- Model-free algorithms for robust offline RL. Can we design provably efficient model-free algorithms for robust offline RL with partial coverage? Recent works (Wang et al., 2023a,b) in understanding robust variants of Q-learning in the generative model might shed light on how to approach this question.
- Choice of uncertainty sets. Moreover, it is possible to extend our framework to handle uncertainty sets defined using other distances such as the chi-square distance and the total variation distance in a similar fashion. Shi et al. (2023) recently established near minimax-optimal sample complexities for the total variation distance and the chi-square distance in the generative model setting, paving ways to study these uncertainty sets in the offline setting.
- Adaptive tuning of the uncertainty set. In this work, we treat the radius of the uncertainty set as a fixed, a priori specified parameter, and study the sample complexity of learning a robust optimal policy with respect to the given uncertainty set modeling the sim-to-real gap. It is of great interest to incorporate the tuning of the uncertainty set (both its size and metric) to complete the pipeline of the algorithm design, which will require a different framework than the one adopted in the current paper.

We leave these questions to future investigations.

## Acknowledgements

This work is supported in part by the grants ONR N00014-19-1-2404, NSF CCF-2106778, DMS-2134080, and CNS-2148212. L. Shi is also gratefully supported by the Leo Finzi Memorial Fellowship, Wei Shen

and Xuehong Zhang Presidential Fellowship, and Liang Ji-Dian Graduate Fellowship at Carnegie Mellon University. The authors thank Gen Li, Zhengyuan Zhou, and Nian Si for helpful discussions.

## References

- Abdullah, M. A., Ren, H., Ammar, H. B., Milenkovic, V., Luo, R., Zhang, M., and Wang, J. (2019). Wasserstein robust reinforcement learning. arXiv preprint arXiv:1907.13196 .
- Agarwal, A., Kakade, S., and Yang, L. F. (2020). Model-based reinforcement learning with a generative model is minimax optimal. Conference on Learning Theory , pages 67-83.
- Badrinath, K. P. and Kalathil, D. (2021). Robust reinforcement learning using least squares policy iteration with provable performance guarantees. In International Conference on Machine Learning , pages 511-520. PMLR.
- Bertsimas, D., Gupta, V., and Kallus, N. (2018). Data-driven robust optimization. Mathematical Programming , 167(2):235-292.
- Blanchet, J. and Murthy, K. (2019). Quantifying distributional model risk via optimal transport. Mathematics of Operations Research , 44(2):565-600.
- Chen, J. and Jiang, N. (2019). Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning , pages 1042-1051. PMLR.
- Choi, J. J., Laibson, D., Madrian, B. C., and Metrick, A. (2009). Reinforcement learning and savings behavior. The Journal of finance , 64(6):2515-2534.
- Delage, E. and Ye, Y. (2010). Distributionally robust optimization under moment uncertainty with application to data-driven problems. Operations research , 58(3):595-612.
- Derman, E. and Mannor, S. (2020). Distributional robustness and regularization in reinforcement learning. arXiv preprint arXiv:2003.02894 .
- Ding, W., Shi, L., Chi, Y., and Zhao, D. (2023). Seeing is not believing: Robust reinforcement learning against spurious correlation. In Thirty-seventh Conference on Neural Information Processing Systems .
- Duchi, J. C. (2018). Introductory lectures on stochastic optimization. The Mathematics of Data , 25:99-186.
- Duchi, J. C. and Namkoong, H. (2021). Learning models with uniform performance via distributionally robust optimization. The Annals of Statistics , 49(3):1378-1406.
- Gao, R. (2022). Finite-sample guarantees for Wasserstein distributionally robust optimization: Breaking the curse of dimensionality. Operations Research .
- Garcıa, J. and Fernández, F. (2015). A comprehensive survey on safe reinforcement learning. Journal of Machine Learning Research , 16(1):1437-1480.
- Gilbert, E. N. (1952). A comparison of signalling alphabets. The Bell system technical journal , 31(3):504-522.
- Goyal, V. and Grand-Clement, J. (2022). Robust markov decision processes: Beyond rectangularity. Mathematics of Operations Research .
- Ho, C. P., Petrik, M., and Wiesemann, W. (2018). Fast Bellman updates for robust MDPs. In International Conference on Machine Learning , pages 1979-1988. PMLR.
- Ho, C. P., Petrik, M., and Wiesemann, W. (2021). Partial policy iteration for l1-robust markov decision processes. Journal of Machine Learning Research , 22(275):1-46.
- Hou, L., Pang, L., Hong, X., Lan, Y., Ma, Z., and Yin, D. (2020). Robust reinforcement learning with Wasserstein constraint. arXiv preprint arXiv:2006.00945 .

- Hu, Z. and Hong, L. J. (2013). Kullback-leibler divergence constrained distributionally robust optimization. Available at Optimization Online , pages 1695-1724.
- Iyengar, G. N. (2005). Robust dynamic programming. Mathematics of Operations Research , 30(2):257-280.
- Jin, Y., Yang, Z., and Wang, Z. (2021). Is pessimism provably efficient for offline RL? In International Conference on Machine Learning , pages 5084-5096.
- Kaufman, D. L. and Schaefer, A. J. (2013). Robust modified policy iteration. INFORMS Journal on Computing , 25(3):396-410.
- Kumar, A., Zhou, A., Tucker, G., and Levine, S. (2020). Conservative Q-learning for offline reinforcement learning. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F., and Lin, H., editors, Advances in Neural Information Processing Systems , volume 33, pages 1179-1191. Curran Associates, Inc.
- Levine, S., Kumar, A., Tucker, G., and Fu, J. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643 .
- Li, G., Shi, L., Chen, Y., Chi, Y., and Wei, Y. (2022). Settling the sample complexity of model-based offline reinforcement learning. arXiv preprint arXiv:2204.05275 .
- Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2020). Breaking the sample size barrier in model-based reinforcement learning with a generative model. In Advances in Neural Information Processing Systems , volume 33.
- Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2021). Sample complexity of asynchronous Q-learning: Sharper analysis and variance reduction. IEEE Transactions on Information Theory , 68(1):448-473.
- Munos, R. (2005). Error bounds for approximate value iteration. In Proceedings of the National Conference on Artificial Intelligence , volume 20, page 1006. Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999.
- Nilim, A. and El Ghaoui, L. (2005). Robust control of Markov decision processes with uncertain transition matrices. Operations Research , 53(5):780-798.
- Nilim, A. and Ghaoui, L. (2003). Robustness in markov decision problems with uncertain transition matrices. Advances in neural information processing systems , 16.
- Panaganti, K. and Kalathil, D. (2022). Sample complexity of robust reinforcement learning with a generative model. In International Conference on Artificial Intelligence and Statistics , pages 9582-9602. PMLR.
- Rahimian, H. and Mehrotra, S. (2019). Distributionally robust optimization: A review. arXiv preprint arXiv:1908.05659 .
- Rashidinejad, P., Zhu, B., Ma, C., Jiao, J., and Russell, S. (2021). Bridging offline reinforcement learning and imitation learning: A tale of pessimism. Neural Information Processing Systems (NeurIPS) .
- Roy, A., Xu, H., and Pokutta, S. (2017). Reinforcement learning under model mismatch. Advances in neural information processing systems , 30.
- Schulman, J., Ho, J., Lee, A. X., Awwal, I., Bradlow, H., and Abbeel, P. (2013). Finding locally optimal, collision-free trajectories with sequential convex optimization. In Robotics: science and systems , volume 9, pages 1-10. Citeseer.
- Shi, L., Li, G., Wei, Y., Chen, Y., and Chi, Y. (2022). Pessimistic Q-learning for offline reinforcement learning: Towards optimal sample complexity. In Proceedings of the 39th International Conference on Machine Learning , volume 162, pages 19967-20025. PMLR.
- Shi, L., Li, G., Wei, Y., Chen, Y., Geist, M., and Chi, Y. (2023). The curious price of distributional robustness in reinforcement learning with a generative model. arXiv preprint arXiv:2305.16589 .

- Sinha, A., Namkoong, H., and Duchi, J. (2018). Certifying some distributional robustness with principled adversarial training. In International Conference on Learning Representations .
- Smirnova, E., Dohmatob, E., and Mary, J. (2019). Distributionally robust reinforcement learning. arXiv preprint arXiv:1902.08708 .
- Song, J. and Zhao, C. (2020). Optimistic distributionally robust policy optimization. arXiv preprint arXiv:2006.07815 .
- Sutton, R. S. and Barto, A. G. (2018). Reinforcement learning: An introduction . MIT press.
- Tamar, A., Mannor, S., and Xu, H. (2014). Scaling up robust MDPs using function approximation. In International conference on machine learning , pages 181-189. PMLR.
- Topsøe, F. (2007). Some bounds for the logarithmic function. Inequality theory and applications , 4:137.
- Tsybakov, A. B. and Zaiats, V. (2009). Introduction to nonparametric estimation , volume 11. Springer.
- Vershynin, R. (2018). High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press.
- Wang, J., Gao, R., and Zha, H. (2022). Reliable off-policy evaluation for reinforcement learning. Operations Research .
- Wang, S., Si, N., Blanchet, J., and Zhou, Z. (2023a). A finite sample complexity bound for distributionally robust Q-learning. In International Conference on Artificial Intelligence and Statistics , pages 3370-3398. PMLR.
- Wang, S., Si, N., Blanchet, J., and Zhou, Z. (2023b). Sample complexity of variance-reduced distributionally robust Q-learning. arXiv preprint arXiv:2305.18420 .
- Wang, Y. and Zou, S. (2021). Online robust reinforcement learning with model uncertainty. Advances in Neural Information Processing Systems , 34.
- Wiesemann, W., Kuhn, D., and Rustem, B. (2013). Robust markov decision processes. Mathematics of Operations Research , 38(1):153-183.
- Wolff, E. M., Topcu, U., and Murray, R. M. (2012). Robust control of uncertain markov decision processes with temporal logic specifications. In 2012 IEEE 51st IEEE Conference on Decision and Control (CDC) , pages 3372-3379. IEEE.
- Xie, T., Jiang, N., Wang, H., Xiong, C., and Bai, Y. (2021). Policy finetuning: Bridging sample-efficient offline and online reinforcement learning. Advances in neural information processing systems , 34.
- Xu, H. and Mannor, S. (2012). Distributionally robust Markov decision processes. Mathematics of Operations Research , 37(2):288-300.
- Yan, Y., Li, G., Chen, Y., and Fan, J. (2023). The efficacy of pessimism in asynchronous Q-learning. IEEE Transactions on Information Theory .
- Yang, I. (2017). A convex optimization approach to distributionally robust markov decision processes with Wasserstein distance. IEEE Control Systems Letters , 1(1):164-169.
- Yang, W., Zhang, L., and Zhang, Z. (2022). Toward theoretical understandings of robust markov decision processes: Sample complexity and asymptotics. The Annals of Statistics , 50(6):3223-3248.
- Yin, M., Bai, Y., and Wang, Y.-X. (2021). Near-optimal offline reinforcement learning via double variance reduction. Advances in neural information processing systems , 34.
- Yin, M. and Wang, Y.-X. (2021). Optimal uniform OPE and model-based offline reinforcement learning in time-homogeneous, reward-free and task-agnostic settings. arXiv preprint arXiv:2105.06029 .

- Zhang, H., Chen, H., Xiao, C., Li, B., Liu, M., Boning, D., and Hsieh, C.-J. (2020). Robust deep reinforcement learning against adversarial perturbations on state observations. Advances in Neural Information Processing Systems , 33:21024-21037.
- Zhou, Z., Bai, Q., Zhou, Z., Qiu, L., Blanchet, J., and Glynn, P. (2021). Finite-sample regret bound for distributionally robust offline tabular reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 3331-3339. PMLR.

## A Preliminaries

Before starting, let us introduce some additional notation useful throughout the theoretical analysis. Let ess inf X denote the essential infimum of a function/variable X .

## A.1 Properties of the robust Bellman operator

To begin with, we introduce the following strong duality lemma which is widely used in distributionally robust optimization when the uncertainty set is defined with respect to the KL divergence.

Lemma 4 ((Hu and Hong, 2013), Theorem 1) . Suppose f ( x ) has a finite moment generating function in some neighborhood around x = 0 , then for any σ &gt; 0 and a nominal distribution P 0 , we have

<!-- formula-not-decoded -->

Armed with the above lemma, it is easily verified that for any positive constant M and a nominal distribution vector P 0 ∈ R 1 × S supported over the state space S , if X ( s ) ∈ [0 , M ] for all s ∈ S , then

<!-- formula-not-decoded -->

For convenience, we introduce the following lemma, paraphrased from Zhou et al. (2021, Lemma 4) and its proof, to further characterize several essential properties of the optimal dual value.

Lemma 5 ((Zhou et al., 2021)) . Let X ∼ P be a bounded random variable with X ∈ [0 , M ] . Let σ &gt; 0 be any uncertainty level and the corresponding optimal dual variable be

<!-- formula-not-decoded -->

Then the optimal value λ ⋆ obeys where λ ⋆ = 0 if and only if

Moreover, when λ ⋆ = 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.2 Concentration inequalities

In light of Lemma 5 (cf. 67), we are interested in comparing the values of essinf X when X is drawn from the population nominal distribution or its empirical estimate. This is supplied by the following lemma from Zhou et al. (2021).

Lemma 6 ((Zhou et al., 2021)) . Let X ∼ P be a discrete bounded random variable with X ∈ [0 , M ] . Let P n denote the empirical distribution constructed from n independent samples X 1 , X 2 , · · · , X n , and let ̂ X ∼ P n . Denote P min ,X as the smallest positive probability P min ,X := min { P ( X = x ) : x ∈ supp ( X ) } , where supp ( X ) is the support of X . Then for any δ ∈ (0 , 1) , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We next gather a few elementary facts about the Binomial distribution, which will be useful throughout the proof.

Lemma 7 (Chernoff's inequality) . Suppose N ∼ Binomial ( n, p ) , where n ≥ 1 and p ∈ [0 , 1) . For some universal constant c f &gt; 0 , we have

<!-- formula-not-decoded -->

Lemma 8 ((Shi et al., 2022, Lemma 8)) . Suppose N ∼ Binomial ( n, p ) , where n ≥ 1 and p ∈ [0 , 1] . For any δ ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hold with probability at least 1 -4 δ .

## A.3 Kullback-Leibler (KL) divergence

We next introduce some useful facts about the Kullback-Leibler (KL) divergence for two distributions P and Q , denoted as KL ( P ∥ Q ) . Denoting Ber ( p ) (resp. Ber ( q ) ) as the Bernoulli distribution with mean p (resp. q ), we introduce

<!-- formula-not-decoded -->

which represents the KL divergence from Ber ( p ) to Ber ( q ) . We now introduce the following lemma.

Lemma 9. For any p, q ∈ [ 1 2 , 1 ) and p &gt; q , it holds that

<!-- formula-not-decoded -->

Moreover, for any 0 ≤ x &lt; y &lt; q , it holds

<!-- formula-not-decoded -->

Proof. The first half of this lemma is proven in Li et al. (2022, Lemma 10). For the latter half, it follows from that the function

<!-- formula-not-decoded -->

is monotonically decreasing for all x ∈ (0 , q ] , since its derivative with respect to x satisfies ∂f ( x,q ) ∂x = log x q + log 1 -q 1 -x &lt; 0 .

as long as

## B Analysis: episodic finite-horizon RMDPs

## B.1 Proof of Theorem 1

Before starting, we introduce several additional notation that will be useful in the analysis. First, we denote the state-action space covered by the behavior policy π b in the nominal model P 0 as

<!-- formula-not-decoded -->

Moreover, we recall the definition in (23) and define a similar one based on the exact nominal model P 0 as

<!-- formula-not-decoded -->

Clearly, by comparing with the definitions (24) and (25), it holds that

<!-- formula-not-decoded -->

For any time step h ∈ [ H ] , we denote the set of possible state occupancy distributions associated with the optimal policy π ⋆ in a model within the uncertainty set P ∈ U σ ( P 0 ) as

<!-- formula-not-decoded -->

where the second equality is due to the fact that π ⋆ is chosen to be deterministic.

With these in place, the proof of Theorem 1 is separated into several key steps, as outlined below.

Step 1: establishing the pessimism property. To achieve this claim, we heavily count on the following lemma whose proof can be found in Appendix B.2.

Lemma 10. Instate the assumptions in Theorem 1. Then for all ( h, s, a ) ∈ [ H ] ×S×A , consider any vector V ∈ R S independent of ̂ P 0 h,s,a obeying ∥ V ∥ ∞ ≤ H . With probability at least 1 -δ , one has

<!-- formula-not-decoded -->

with b h ( s, a ) given in (22) . Moreover, for all ( h, s, a ) ∈ C b , with probability at least 1 -δ , one has

<!-- formula-not-decoded -->

Armed with the above lemma, with probability at least 1 -δ , we shall show the following relation holds

<!-- formula-not-decoded -->

which means that ̂ Q h (resp. ̂ V h ) is a pessimistic estimate of Q ̂ π,σ h (resp. V ̂ π,σ h ). Towards this, it is easily verified that the latter assertion concerning V ̂ π,σ h is implied by the former, since

<!-- formula-not-decoded -->

Therefore, the remainder of this step focuses on verifying the former assertion in (81) by induction.

- To begin, the claim (81) holds at the base case when h = H +1 , by invoking the trivial fact ̂ Q H +1 ( s, a ) = Q ̂ π,σ H +1 ( s, a ) = 0 .

- Then, suppose that ̂ Q h +1 ( s, a ) ≤ Q ̂ π,σ h +1 ( s, a ) holds for all ( s, a ) ∈ S × A at some time step h ∈ [ H ] , it boils down to show ̂ Q h ( s, a ) ≤ Q ̂ π,σ h ( s, a ) .

By the update rule of ̂ Q h ( s, a ) in Algorithm 2 (cf. line 7), the above relation holds immediately if ̂ Q h ( s, a ) = 0 since ̂ Q h ( s, a ) = 0 ≤ Q ̂ π,σ h ( s, a ) . Otherwise, ̂ Q h ( s, a ) is updated via

<!-- formula-not-decoded -->

where (i) rewrites the update rule back to its primal form (cf. (19)), (ii) holds by applying (79) with the condition (28) satisfied and the induction hypothesis ̂ V h +1 ≤ V ̂ π,σ h +1 , and lastly, (iii) follows by the robust Bellman consistency equation (8).

Putting them together, we have verified the claim (81) by induction.

Step 2: bounding V ⋆,σ h ( s ) -V ̂ π,σ h ( s ) . With the pessimism property (81) in place, we observe that the following relation holds

<!-- formula-not-decoded -->

where the last inequality follows from ̂ Q h ( s, π ⋆ h ( s ) ) ≤ max a ̂ Q h ( s, a ) = ̂ V h ( s ) . Then, by the robust Bellman optimality equation in (9) and the primal version of the update rule (cf. (19))

<!-- formula-not-decoded -->

we arrive at

<!-- formula-not-decoded -->

where (i) holds by applying Lemma 2 (cf. (79)) since ̂ V h +1 is independent of P 0 h,s,π ⋆ h ( s ) by construction, and (ii) arises from introducing the notation

<!-- formula-not-decoded -->

and consequently,

<!-- formula-not-decoded -->

To continue, let us introduce some additional notation for convenience. Define a sequence of matrices ̂ P inf h ∈ R S × S and vectors b ⋆ h ∈ R S for h ∈ [ H ] , where their s -th rows (resp. entries) are given by

<!-- formula-not-decoded -->

Applying (85) recursively over the time steps h, h +1 , · · · , H using the above notation gives

<!-- formula-not-decoded -->

where we let ( ∏ i -1 j = i ̂ P inf j ) = I for convenience.

For any d ⋆ h ∈ D ⋆ h (cf. (78)), taking inner product with (88) leads to

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

by the definition of D ⋆ i (cf. (78)) for all i = h +1 , · · · , H .

Step 3: controlling ⟨ d ⋆ i , b ⋆ i ⟩ using concentrability. Since ⟨ d ⋆ i , b ⋆ i ⟩ = ∑ s ∈S d ⋆ i ( s ) b ⋆ i ( s ) , we shall divide the discussion in two different cases.

- For s ∈ S where max P ∈U σ ( P 0 ) d ⋆,P i ( s, π ⋆ i ( s ) ) = max P ∈U σ ( P 0 ) d ⋆,P i ( s ) = 0 , it follows from the definition (cf. (78)) that for any d ⋆ i ∈ D ⋆ i , it satisfies that

<!-- formula-not-decoded -->

- For s ∈ S where max P ∈U σ ( P 0 ) d ⋆,P i ( s, π ⋆ i ( s ) ) = max P ∈U σ ( P 0 ) d ⋆,P i ( s ) &gt; 0 , by the assumption in (14)

<!-- formula-not-decoded -->

it implies that

<!-- formula-not-decoded -->

Lemma 1 tells that with probability at least 1 -8 δ ,

<!-- formula-not-decoded -->

where (i) holds due to

<!-- formula-not-decoded -->

for some sufficiently large c 1 , where the first inequality follows from Condition (28), the second inequality follows from

<!-- formula-not-decoded -->

and the last inequality follows from P b min ≤ 1 . In addition, (ii) follows from Assumption 1. With this in place, we observe that the pessimistic penalty (see (22)) obeys

<!-- formula-not-decoded -->

where (i) holds by applying (80) in view of the fact that ( i, s, π ⋆ i ( s ) ) ∈ C b by (92), and the last inequality holds by (93).

Combining the results in the above two cases leads to

<!-- formula-not-decoded -->

where (i) follows from the Cauchy-Schwarz inequality and the last inequality hold by the trivial fact

<!-- formula-not-decoded -->

Step 4: finishing up the proof. Then, inserting (97) back into (89) with h = 1 shows

<!-- formula-not-decoded -->

where the last inequality holds by plugging in the relation P ⋆ min ≤ P min ,i ( s, π ⋆ i ( s ) ) for i = 1 , . . . , H by the definition in (24) (see also (77)), and choosing c 2 to be large enough. The proof is completed.

## B.2 Proof of Lemma 10

To begin, we shall introduce the following fact that

<!-- formula-not-decoded -->

as long as Condition (28) holds. The proof is postponed to Appendix B.2.3. With this in mind, we shall first establish the simpler bound (80) and then move on to show (79).

## B.2.1 Proof of (80)

To begin, recall that (100) is satisfied for all ( h, s, a ) ∈ C b . By Lemma 8 and the union bound, it holds that with probability at least 1 -δ that for all ( h, s, a ) ∈ C b :

<!-- formula-not-decoded -->

To characterize the relation between P min ,h ( s, a ) and ̂ P min ,h ( s, a ) for any ( h, s, a ) ∈ C b , we suppose-without loss of generality-that P min ,h ( s, a ) = P 0 h ( s 1 | s, a ) and ̂ P min ,h ( s, a ) = ̂ P 0 h ( s 2 | s, a ) for some s 1 , s 2 ∈ S . Then, it follows that

<!-- formula-not-decoded -->

where (i) and (ii) follow from (101).

## B.2.2 Proof of (79)

The main goal of (79) is to control the gap between robust Bellman operations based on the nominal transition kernel P 0 h,s,a and the estimated kernel ̂ P 0 h,s,a by the constructed penalty term. Towards this, first consider ( h, s, a ) / ∈ C b , which corresponds to the state-action pairs ( s, a ) that haven't been visited at step h by the behavior policy. In other words, N h ( s, a ) = 0 . In this case, (79) can be easily verified that

<!-- formula-not-decoded -->

where (i) follows from the fact ̂ P 0 h,s,a = 0 when N h ( s, a ) = 0 (see (15)), (ii) arises from the assumption ∥ V ∥ ∞ ≤ H , and (iii) holds by the definition of b h ( s, a ) in (22). Therefore, the remainder of the proof will focus on verifying (79) for ( h, s, a ) ∈ C b . Rewriting the term of interest via duality (cf. Lemma 4) yields

<!-- formula-not-decoded -->

Denoting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 5 (cf. (65)) then gives that

<!-- formula-not-decoded -->

̸

due to ∥ V ∥ ∞ ≤ H . We shall control (103) in three different cases separately: (a) λ ⋆ h,s,a = 0 and ̂ λ ⋆ h,s,a = 0 ; (b) λ ⋆ h,s,a &gt; 0 and ̂ λ ⋆ h,s,a = 0 or λ ⋆ h,s,a = 0 and ̂ λ ⋆ h,s,a &gt; 0 ; and (c) λ ⋆ h,s,a = 0 or ̂ λ ⋆ h,s,a = 0 .

̸

Case (a): λ ⋆ h,s,a = 0 and ̂ λ ⋆ h,s,a = 0 . Applying Lemma 5 and Lemma 6 to (103) gives that, with probability at least 1 -δ KH ,

<!-- formula-not-decoded -->

where (i) holds by Lemma 5 (cf. (67)) and (ii) arises from Lemma 6 (cf. (68)) given (100).

Case (b): λ ⋆ h,s,a &gt; 0 and ̂ λ ⋆ h,s,a = 0 or λ ⋆ h,s,a = 0 and ̂ λ ⋆ h,s,a &gt; 0 . Towards this, note that two trivial facts are implied by the definition (104):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To continue, first, we consider a subcase when λ ⋆ h,s,a = 0 and ̂ λ ⋆ h,s,a &gt; 0 . With probability at least 1 -δ KH , it follows from Lemma 5 (cf. (67)) and Lemma 6 (cf. (68)) that

<!-- formula-not-decoded -->

leading to

<!-- formula-not-decoded -->

where (i) follows from the definition of ̂ λ ⋆ h,s,a in (104) and the fact in (107a).

We pause to claim that with probability at least 1 -δ , the following bound holds

<!-- formula-not-decoded -->

The proof is postponed to Appendix B.2.4. With (110) in place, we can further bound (109) (which is plugged into (103)) as

<!-- formula-not-decoded -->

where (i) follows from log(1 + x ) ≤ 2 | x | for any | x | ≤ 1 2 in view of (110), (ii) follows from (105) as well as (110), and the last line follows from (80) and choosing c b to be sufficiently large.

Moreover, note that it can be easily verified that

<!-- formula-not-decoded -->

due to the assumption ∥ V ∥ ∞ ≤ H . Plugging in the definition of b h ( s, a ) in (22), combined with the above bounds, we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

The other subcase when λ ⋆ h,s,a &gt; 0 and ̂ λ ⋆ h,s,a = 0 follows similarly from the bound

<!-- formula-not-decoded -->

and therefore, will be omitted for simplicity.

Case (c): λ ⋆ h,s,a &gt; 0 and ̂ λ ⋆ h,s,a &gt; 0 . It follows that

<!-- formula-not-decoded -->

where (i) can be verified by applying the facts in (107). Hence, the above term (114) can be controlled again in a similar manner as (109); we omit the details for simplicity.

Summing up. Combining the previous results in different cases by the union bound, with probability at least 1 -10 δ , it is satisfied that for all ( h, s, a ) ∈ C b :

<!-- formula-not-decoded -->

which concludes the proof.

## B.2.3 Proof of (100)

Observe that for all ( h, s, a ) ∈ C b :

<!-- formula-not-decoded -->

where (i) follows from Condition (28), (ii) follows from the definition that d b min ≤ d b h ( s, a ) for ( h, s, a ) ∈ C b , and (iii) comes from (77).

Lemma 1 then tells that with probability at least 1 -8 δ ,

<!-- formula-not-decoded -->

where the second line follows from the above relation as long as c 1 is sufficiently large. The last inequality of (100) then follows from

<!-- formula-not-decoded -->

since x ≤ -log(1 -x ) for all x ∈ [0 , 1] .

## B.2.4 Proof of (110)

Denoting

<!-- formula-not-decoded -->

as the support of P 0 h,s,a , we observe that

<!-- formula-not-decoded -->

where the second line follows from ∑ i a i = ∑ i b i a i b i ≤ (max i a i b i ) ∑ i b i for any positive sequences { a i , b i } i obeying a i , b i &gt; 0 .

To continue, note that for any ( h, s, a ) ∈ C b and s ′ ∈ supp ( P 0 h,s,a ) , N h ( s, a ) ̂ P 0 h ( s ′ | s, a ) follows the binomial distribution Binomial ( N h ( s, a ) , P 0 h ( s ′ | s, a ) ) . Thus, applying Lemma 7 with t = √ log ( KHS δ ) c f N h ( s,a ) P 0 h ( s ′ | s,a ) yields

<!-- formula-not-decoded -->

as soon as t ≤ 1 2 , which can be verified by the fact (100) and P min ,h ( s, a ) ≤ P 0 h ( s ′ | s, a ) (cf. (76)), namely,

<!-- formula-not-decoded -->

as long as c 1 is sufficiently large.

Applying (119) and taking the union bound over s ∈ supp ( P 0 h,s,a ) lead to that with probability at least 1 -δ KH ,

<!-- formula-not-decoded -->

where the last line uses again (120). Plugging this back into (118) and applying the union bound over ( h, s, a ) ∈ C b then completes the proof.

## B.3 Proof of Theorem 2

The proof of Theorem 2 is inspired by the construction in Li et al. (2022) for standard MDPs, but is considerably more involved to handle the uncertainty set unique in robust MDPs. In particular, we construct two different classes of hard instances for different range of the uncertainty level σ to achieve a tighter σ -dependent lower bound. In what follows, we start with the lower bound for the case when the uncertainty level is relatively small, by first constructing some hard instances and then characterizing the sample complexity requirements over these instances. We then move onto the case when the uncertainty level is relatively large, and carry out a similar argument.

## B.3.1 Construction of hard problem instances: small uncertainty level

Construction of a collection of hard MDPs To begin, let's consider a collection Θ ⊆ { 0 , 1 } H , consisting of vectors with H dimensions. The Gilbert-Varshamov lemma (Gilbert, 1952) tells that there exists a set Θ ⊆ { 0 , 1 } H such that:

̸

<!-- formula-not-decoded -->

Armed with Θ , we then generate a collection of MDPs

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The transition kernel P θ = { P θ h h } H h =1 of the MDP M θ is specified as follows:

<!-- formula-not-decoded -->

for any ( s, a, s ′ , h ) ∈ S × A × S × [ H ] . Here, p and q are set according to

<!-- formula-not-decoded -->

for c 1 = 1 / 8 and some c 2 that satisfies

<!-- formula-not-decoded -->

Clearly, it follows that for any ( s, a, h ) ∈ S × A × [ H ] .

Uncertainty set of the transition kernels. Denote the transition kernel vector as

<!-- formula-not-decoded -->

For any ( s, a, h ) ∈ S ×A× [ H ] , the perturbation of the transition kernels in M θ is restricted to the following uncertainty set

<!-- formula-not-decoded -->

with the uncertainty level σ satisfying and in particular, denote

for constant c 3 = 2 c 1 = 1 4 .

Proof. The proof is postponed to Appendix B.3.5.

<!-- formula-not-decoded -->

by construction. Furthermore, the MDP will stay in the state subset { 0 , 1 } if its initial state falls in { 0 , 1 } . The reward function of these MDPs is set as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Before continuing, we shall introduce some notation for convenience. For any P θ h h ( · | s, a ) in (123), we define the limit of the perturbed kernel transiting to the next state s ′ from the current state-action pair ( s, a ) by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Armed with the above definitions, we introduce the following lemma which implies some useful properties of the uncertainty set.

Lemma 11. The perturbed transition kernels obey

<!-- formula-not-decoded -->

Denoting p 1 := p ⋆ and q 1 := q ⋆ , we have that when the uncertainty level σ satisfies (130) ,

<!-- formula-not-decoded -->

Value functions and optimal policies. We take a moment to derive the corresponding value functions and identify the optimal policies. With some abuse of notation, for any MDP M θ , we denote π ⋆,θ = { π ⋆,θ h } H h =1 as the optimal policy, and let V π,σ,θ h (resp. V ⋆,σ,θ h ) represent the robust value function of policy π (resp. π ⋆,θ ) at step h with uncertainty radius σ . Armed with these notation, we introduce the following lemma which collects the properties concerning the value functions and optimal policies.

Lemma 12. Consider any θ ∈ Θ and any policy π . Then it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, for any h ∈ [ H ] and s ∈ S \ { 0 } , the optimal policies and the optimal value functions obey

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

provided that 0 &lt; c 1 ≤ 1 / 2 .

Proof.

<!-- formula-not-decoded -->

Construction of the history/batch dataset. In the nominal environment M θ , a batch dataset is generated consisting of K independent sample trajectories each of length H , where each trajectory is generated according to (10), based on the following initial state distribution ρ b and behavior policy π b = { π b h } H h =1 :

<!-- formula-not-decoded -->

Here, µ ( s ) is defined as the following state distribution supported on the state subset { 0 , 1 } :

<!-- formula-not-decoded -->

where 1 ( · ) is the indicator function, and C &gt; 0 is some constant that will determine the concentrability coefficient C ⋆ rob (as we shall detail momentarily) and obeys

<!-- formula-not-decoded -->

As it turns out, for any MDP M θ , the occupancy distributions of the above batch dataset are the same (due to symmetry) and admit the following simple characterization:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, we choose the following initial state distribution

<!-- formula-not-decoded -->

With this choice of ρ , the single-policy clipped concentrability coefficient C ⋆ rob and the quantity C are intimately connected as follows:

<!-- formula-not-decoded -->

The proof of the claim (141) and (143) are postponed to Appendix B.3.7.

for any h ∈ [ H ] , where

## B.3.2 Establishing the minimax lower bound: small uncertainty level

Towards this, we first make the following claim: for an arbitrary policy π obeying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We shall postpone the proof of this claim to Appendix B.3.8.

Armed with the above claim and following the same arguments in (Li et al., 2022, Section C.3.2), we complete the proof by observing: for some small enough constant c 4 , as long as the sample size is beneath

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂

where P θ denote the probability conditioned on that the MDP is M θ . We omit the details for brevity and complete the proof; interested readers can referred to (Li et al., 2022, Section C.3.2).

## B.3.3 Construction of hard problem instances: large uncertainty level

We now move onto the case when the uncertainty level is relatively large, we construct another class of hard instances, which is almost the same as the previous one except for the transition kernel.

Construction of a collection of hard MDPs. Let us introduce two MDPs

<!-- formula-not-decoded -->

where the state space is S = { 0 , 1 , . . . , S -1 } , and the action space is A = { 0 , 1 } . The transition kernel P ϕ of the constructed MDP M ϕ is defined as

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

In words, except at step h = 1 , the MDP always stays in the same state. Additionally, the MDP will always stay in the state subset { 0 , 1 } if the initial distribution is supported only on { 0 , 1 } , in view of (149). Here, p and q are set to be

<!-- formula-not-decoded -->

for some H ≥ 2 e 8 , α and ∆ obeying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

one has then we necessarily have

where β is set as

The assumption (151) immediately indicates the facts

<!-- formula-not-decoded -->

Moreover, for any ( h, s, a ) ∈ [ H ] ×S × A , the reward function is defined as

<!-- formula-not-decoded -->

Construction of the history/batch dataset. We utilize the same batch dataset described in Appendix B.3.1 and choose the same initial state distribution ρ in (142) As a result, for any MDP M ϕ , the occupancy distributions of the above batch dataset are the same (due to symmetry) and admit the following simple characterization:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of the claim (155) is postponed to Appendix B.3.9.

Uncertainty set of the transition kernels. Denote the transition kernel vector as

<!-- formula-not-decoded -->

For any ( s, a, h ) ∈ S ×A× [ H ] , the perturbation of the transition kernels in M ϕ is restricted to the following uncertainty set

<!-- formula-not-decoded -->

where the radius of the uncertainty set σ obeys:

<!-- formula-not-decoded -->

Before continuing, we shall introduce some notation for convenience. For any P ϕ h ( · | s, a ) in (149), we define the limit of the perturbed kernel transiting to the next state s ′ from the current state-action pair ( s, a ) by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Armed with the above definitions, we introduce the following lemma which implies some useful properties of the uncertainty set.

Lemma 13. When β satisfies (152) and the uncertainty level σ satisfies (158) , the perturbed transition kernels obey

<!-- formula-not-decoded -->

and in particular, denote

Proof. See Appendix B.3.10.

Value functions and optimal policies. Similar to Appendix B.3.1, for any MDP M ϕ , we denote π ⋆,ϕ = { π ⋆,ϕ h } H h =1 as the optimal policy, and let V π,σ,ϕ h (resp. V ⋆,σ,ϕ h ) represent the robust value function of policy π (resp. π ⋆,ϕ ) at step h with uncertainty radius σ . Then we introduce the following lemma which collects the properties concerning the value functions and optimal policies.

Lemma 14. For any ϕ = { 0 , 1 } and any policy π , defining

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, the optimal policies and the optimal value functions obey

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The robust single-policy clipped concentrability coefficient C ⋆ rob obeys

<!-- formula-not-decoded -->

it holds that

Proof. See Appendix B.3.11.

In view of Lemma 14, we note that the smallest positive state transition probability of the optimal policy π ⋆ under any MDP M ϕ with ϕ ∈ { 0 , 1 } thus can be given by

<!-- formula-not-decoded -->

which obeys according to (150) and (151).

## B.3.4 Establishing the minimax lower bound: large uncertainty level

We are now ready to establish the sample complexity lower bound. With the choice of the initial distribution ρ in (142), for any policy estimator ̂ π computed based on the batch dataset, we plan to control the quantity

<!-- formula-not-decoded -->

Step 1: converting the goal to estimate ϕ . We make the following claim which shall be verified in Appendix B.3.12: given ε ≤ H 384 e 6 log ( 1 α ) ≤ H 384 e 6 log ( 1 α +∆ ) , choosing

<!-- formula-not-decoded -->

which satisfies (151) with the aid of (158) and (150), it holds that for any policy ̂ π ,

<!-- formula-not-decoded -->

Armed with this relation between the policy ̂ π and its sub-optimality gap, we are positioned to construct an estimate of ϕ . We denote P ϕ as the probability distribution when the MDP is M ϕ , for any ϕ ∈ { 0 , 1 } .

Suppose for the moment that a policy estimate ̂ π achieves

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then in view of (168), we necessarily have ̂ π 1 ( ϕ | 0) ≥ 1 2 with probability at least 7 8 . With this in mind, we are motivated to construct the following estimate ̂ ϕ for ϕ ∈ { 0 , 1 } :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In what follows, we would like to show (171) cannot happen without enough samples, which would in turn contradict (168).

Step 2: probability of error in testing two hypotheses. Armed with the above preparation, we shall focus on differentiating the two hypotheses ϕ ∈ { 0 , 1 } . Towards this, consider the minimax probability of error defined as follows:

̸

<!-- formula-not-decoded -->

̸

where the infimum is taken over all possible tests ψ constructed from the batch dataset.

Let µ b ,ϕ (resp. µ b ,ϕ h ( s h ) ) be the distribution of a sample trajectory { s h , a h } H h =1 (resp. a sample ( a h , s h +1 ) conditional on s h ) for the MDP M ϕ . Following standard results from Tsybakov and Zaiats (2009, Theorem 2.2) and the additivity of the KL divergence (cf. Tsybakov and Zaiats (2009, Page 85)), we obtain

<!-- formula-not-decoded -->

where we also use the independence of the K trajectories in the batch dataset in the first line. Here, the second line arises from the chain rule of the KL divergence (Duchi, 2018, Lemma 5.2.8) and the Markov property of the sample trajectories (recall that d b ,P 0 h = d b ,P 1 h ) according to

<!-- formula-not-decoded -->

where the penultimate equality holds by the fact that P 0 h ( · | s, a ) and P 1 h ( · | s, a ) only differ when h = 1 and s = 0 , and the last equality follows from (155).

It remains to control the KL divergence terms in (173). Given p ≥ q ≥ 1 / 2 (cf. (153)), applying Lemma 9 (cf. (73)) yields

<!-- formula-not-decoded -->

where (i) follows from the definition (150), (ii) holds by plugging in the expression of ∆ in (167), (iii) arises from 1 -q ≤ 2(1 -p ) = 2 P ⋆ min (see (151) and (166)), p &gt; 1 2 , as long as c 1 is a large enough constant. It can which obeys

be shown that KL ( P 0 1 ( · | 0 , 1) ∥ P 1 1 ( · | 0 , 1) ) can be upper bounded in the same way. Substituting (174) back into (173) demonstrates that: if the sample size is chosen as

<!-- formula-not-decoded -->

then one necessarily has

<!-- formula-not-decoded -->

where (i) follows from (139) and (ii) holds by (165).

Step 3: putting things together. Finally, suppose that there exists an estimator ̂ π such that

<!-- formula-not-decoded -->

Then Step 1 tells us that the estimator ̂ ϕ defined in (170) must satisfy

̸

<!-- formula-not-decoded -->

̸

which cannot happen under the sample size condition in (175) to avoid contradition with (176). The proof is thus finished.

## B.3.5 Proof of Lemma 11

First, (133) can be easily verified by the definition of p h and q h in (132) and the transition P θ h h in (123).

Proof of the first inequality in (134) . It is observed that

<!-- formula-not-decoded -->

where (i) holds by log(1 + x ) ≥ x 2(1+ x ) when 0 ≤ x &lt; ∞ and log(1 + x ) ≥ x 2 2+ x 1+ x when -1 &lt; x ≤ 0 (Topsøe, 2007), and the penultimate inequality holds by 1 -3 c 1 2 H ≥ 1 -3 256 ≥ 63 64 . Here, (ii) can be verified by

<!-- formula-not-decoded -->

where (iii) holds by q ≥ 1 -3 c 1 2 H , and (iv) arises from 0 ≤ 1 -q ≤ 3 c 1 2 H .

With above fact and KL ( Ber ( q ⋆ ) ∥ Ber ( q ) ) = σ in mind, applying Lemma 9 leads to q ⋆ ≥ 1 -2 c 1 H .

Proof of the second inequality in (134) . First, observing the first claim in (134), combined with Lemma 9, we know that for any uncertainty level σ ≤ 1 20 H , there exists a unique q ⋆ obeying q ⋆ ≥ 1 -2 c 1 H &gt; 1 2 such that

<!-- formula-not-decoded -->

Then let us define the following function for 0 &lt; x ≤ 1 -q 3 (i.e., p = q + x )

<!-- formula-not-decoded -->

The first derivative ∇ x g ( x, q ) is

<!-- formula-not-decoded -->

where (i) holds by log(1 + x ) ≤ x 2 2+ x 1+ x when 0 ≤ x &lt; ∞ and log(1 + x ) ≥ x 2 2+ x 1+ x when -1 &lt; x ≤ 0 (Topsøe, 2007), and the last inequality always holds for any 0 &lt; x ≤ 1 -q 3 and q ⋆ ≥ 1 2 .

The above fact shows that g ( p -q, q ) ≥ g (0 , q ) = σ , and thus

<!-- formula-not-decoded -->

which complete the proof by observing that p ⋆ ≥ p -q + q ⋆ via applying Lemma 9 with (178).

## B.3.6 Proof of Lemma 12

Ordering the value function for different states. First, note that for any policy π at the final step H +1 , we have

<!-- formula-not-decoded -->

Then for any θ ∈ Θ and any policy π , it is easily verified that

<!-- formula-not-decoded -->

which directly indicates that

<!-- formula-not-decoded -->

Then we provide the following claim which will be proved momentarily using induction: For any θ ∈ Θ and any policy π , the following equation holds

<!-- formula-not-decoded -->

The above result leads to the following immediate fact for state s ∈ S \ { 0 } :

<!-- formula-not-decoded -->

since (185) holds for any π and h ∈ [ H ] . Therefore, for any state s ∈ S \ { 0 } , without loss of generality, we choose the optimal policy obeying

<!-- formula-not-decoded -->

Then the rest of the proof will focus on deriving the value function and optimal policy over state s = 0 . To begin with, recalling the value function in (192)

<!-- formula-not-decoded -->

and observing that the function V π,σ,θ h (0) is increasing in x π,θ h and that x π,θ h is increasing in π h ( θ h | 0) (due to the fact p ⋆ ≥ q ⋆ in (134)). As a result, the optimal policy obeys

<!-- formula-not-decoded -->

at state 0 . Plugging it back to (188) gives

<!-- formula-not-decoded -->

where (i) holds by x π ⋆ ,θ h = p ⋆ π ⋆,θ h ( θ h | 0) + q ⋆ π ⋆,θ h (1 -θ h | 0) = p ⋆ . Here, the last inequality holds since we observe that

<!-- formula-not-decoded -->

as long as c 3 ≤ 0 . 5 , which follows due to the elementary inequalities 1 -x ≤ exp( -x ) for any x ≥ 0 and exp( -x ) ≤ 1 -2 x/ 3 for any 0 ≤ x ≤ 1 / 2 .

Proof of claim in (185) . We shall (185) through induction. Towards this, assuming that at time step h +1 , the following holds

<!-- formula-not-decoded -->

Observing that the base case when h = H has already been confirmed in (184), now we move on to prove the same property for time step h .

To start with, the robust value function of state 0 at step h satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) uses the definition of the reward function in (127), (ii) uses the induction assumption in (191) so that the infimum is attained by picking the choice specified in (132) with a smallest probability mass imposed on the transition to state 0 . Finally, we plug in the definition (136) of x π,θ h in (iii), and the last line follows from (191).

<!-- formula-not-decoded -->

where the last inequality holds by the reward and transition function in (127) and (123) with the induction assumption (191).

Combining (192) and (193), we complete the proof:

<!-- formula-not-decoded -->

## B.3.7 Proof of claim (141) and (143)

Proof of the claim (141) . With the initial state distribution and behavior policy defined in (138), we have for any MDP M θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In view of (149a), the state occupancy distribution at any step h = 2 , 3 , · · · , H obeys

<!-- formula-not-decoded -->

where the last line makes use of the properties q ⋆ ≥ 1 -c 3 /H in Lemma 11 and

<!-- formula-not-decoded -->

provided that 0 &lt; c 3 = 1 / 4 &lt; 1 / 2 . In addition, as state 1 is an absorbing state and state 0 will only transfer to itself or state 1 at each time step, we directly achieve that

<!-- formula-not-decoded -->

For state 1 , as it is absorbing, we directly have

<!-- formula-not-decoded -->

which leads to

According to the assumption in (140), it is easily verified that

<!-- formula-not-decoded -->

Finally, combining (195), (196), (197), (198), the definitions of P θ h h ( · | s, a ) in (123) and the Markov property, we arrive at for any ( h, s ) ∈ [ H ] ×S ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of the claim (143) . Examining the definition of C ⋆ rob in (14), we make the following observations.

- For h = 1 , we have

<!-- formula-not-decoded -->

where (i) holds by d ⋆,P 1 ( s ) = ρ ( s ) = 0 for all s ∈ S \ { 0 } (see (142)) and π ⋆,θ h ( θ h | 0) = 1 for all h ∈ [ H ] , (ii) follows from the fact d ⋆,P 1 (0 , θ ) = 1 , (iii) is verified in (141), and the last equality arises from the definition in (139).

- Similarly, for h = 2 , 3 , · · · , H , we arrive at

<!-- formula-not-decoded -->

where (i) holds by the optimal policy in (137) and the trivial fact that d ⋆,P h ( s ) = 0 for all s ∈ S \ { 0 , 1 } (see (142) and (123)), (ii) arises from (141), and the last equality comes from (139).

Combining the above cases, we complete the proof by

<!-- formula-not-decoded -->

## B.3.8 Proof of the claim (145)

By virtue of (136) and (137), we see that x π ⋆,θ ,θ h = p ⋆ for all h ∈ [ H ] , which combined with (135) gives

<!-- formula-not-decoded -->

which directly leads to

<!-- formula-not-decoded -->

where (i) follows from the fact that x π h ≥ q ⋆ for any π and h ∈ [ H ] , and (iii) holds by the facts (137) and the choice (124) of ( p, q ) . Here, (ii) arises from

<!-- formula-not-decoded -->

where the first inequality holds by applying Lemma 11. With the fact of (203) in mind, combined with the fact q ⋆ ≥ 1 -c 3 H , following the same proof pipeline of (Li et al., 2022, (276) to (278)) leads to

<!-- formula-not-decoded -->

We omit the proof here for conciseness.

## B.3.9 Proof of (155)

With the initial state distribution and behavior policy defined in (138), we have for any MDP M ϕ with ϕ ∈ { 0 , 1 } ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In view of (149a), the state occupancy distribution at step h = 2 obeys

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

With the above result in mind and recalling the assumption in (153), we arrive at

<!-- formula-not-decoded -->

where (i) holds by applying (153) and (140) (which implies µ (0) ≤ µ (1) by the assumption in (140))

<!-- formula-not-decoded -->

Finally, from the definitions of P ϕ h ( · | s, a ) in (149b) and the Markov property, we arrive at for any ( h, s ) ∈ [ H ] ×S ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which leads to which directly leads to

## B.3.10 Proof of Lemma 13

Note that p ≥ q can be easily verified since p &gt; q , which indicates that the first assertion is true. So we will focus on the second assertion in (161). Towards this, invoking the definition in (72), let σ ′ be the KL divergence from Ber ( 1 β ) to Ber ( q ) , defined as follows

<!-- formula-not-decoded -->

where the second line uses the definition of q in (150). We claim that σ ′ satisfies the following relation with σ , which will be proven at the end of this proof:

<!-- formula-not-decoded -->

Recalling the definition of the transition kernel in (149a)

<!-- formula-not-decoded -->

the uncertainty set of the transition kernel with radius σ is thus given as

<!-- formula-not-decoded -->

Recalling the definition of q in (160), we can bound

<!-- formula-not-decoded -->

where (i) holds by σ ≤ σ ′ (cf. (211)) and the last equality follows from applying Lemma 9 (cf. (74)) and (210) to arrive at

<!-- formula-not-decoded -->

Proof of (211) . To control σ ′ , we plug in the assumptions in (153) and β ≥ 4 and arrive at the trivial facts

<!-- formula-not-decoded -->

The above facts directly lead to

Similarly, observing

<!-- formula-not-decoded -->

we arrive at

<!-- formula-not-decoded -->

as long as log ( 1 α +∆ ) ≥ β (cf. (152)). With (213) and (214) in hand, it is straightforward to see that the choice of the uncertainty radius σ in (158) obeys the advertised bound (211).

<!-- formula-not-decoded -->

## B.3.11 Proof of Lemma 14

For notational conciseness, we shall drop the superscript ϕ and use the shorthand V π,σ h = V π,σ,ϕ h and V ⋆,σ h = V ⋆,σ,ϕ h whenever it is clear from the context. We begin by deriving the robust value function for any policy π . Starting with state 1 , at any step h ∈ [ H ] , it obeys

<!-- formula-not-decoded -->

where the first equality follows from the robust Bellman consistency equation (cf. (8)), and the second equality follows from the observation that the distribution P ϕ h, 1 ,a is supported solely on state 1 in view of (149a), therefore U σ ( P ϕ h, 1 ,a ) = P ϕ h, 1 ,a . Leveraging the terminal condition V π,σ H +1 (1) = 0 , and recursively applying the previous relation, we have

<!-- formula-not-decoded -->

Similarly, turning to state 0 , at any step h &gt; 1 , the robust value function satisfies

<!-- formula-not-decoded -->

which again uses the fact that the distribution P ϕ h, 0 ,a is supported solely on state 0 in view of (149b), therefore U σ ( P ϕ h, 0 ,a ) = P ϕ h, 0 ,a . Leveraging the terminal condition V π,σ H +1 (0) = 0 , and recursively applying the previous relation, we have

<!-- formula-not-decoded -->

Taking (215) and (216) together, it follows that

<!-- formula-not-decoded -->

Consequently, the robust value function of state 0 at step h = 1 satisfies

<!-- formula-not-decoded -->

where (i) uses the definition of the reward function in (154), (ii) uses (217) so that the infimum is attained by picking the choice specified in (160) with a smallest probability mass imposed on the transition to state 0 . Finally, we plug in the definition (162) of z π ϕ in (iii), and the last line follows from (215).

Therefore, taking π = π ⋆,ϕ in the previous relation directly leads to

<!-- formula-not-decoded -->

where the second equality follows from (216). Observing that the function ( H -1) z is increasing in z and that z π ϕ is increasing in π 1 ( ϕ | 0) (due to the fact p ≥ q in (161)). As a result, the optimal policy obeys

<!-- formula-not-decoded -->

at state 0 , and plugging back to (219) gives

<!-- formula-not-decoded -->

where z π ⋆,ϕ ϕ = pπ ⋆,ϕ 1 ( ϕ | 0) + qπ ⋆,ϕ 1 (1 -ϕ | 0) = p . For the rest of the states, without loss of generality, we choose the optimal policy obeying

<!-- formula-not-decoded -->

Proof of claim (165) . Given that π ⋆,ϕ h ( ϕ | 0) = 1 for all h ∈ [ H ] and ρ (0) = 1 , for any P ∈ U σ ( P ϕ ) , we have

<!-- formula-not-decoded -->

which (i) holds by plugging in the definition (159), (ii) follows from the definition (160), and the final inequality arises from Lemma 13. Hence, for all 2 ≤ h ≤ H , by the Markov property and P ϕ h (0 | 0 , ϕ ) = 1 , we have

<!-- formula-not-decoded -->

Examining the definition of C ⋆ rob in (14), we make the following observations.

- For h = 1 , we have

<!-- formula-not-decoded -->

where (i) holds by d ⋆,P 1 ( s ) = ρ ( s ) = 0 for all s ∈ S \ { 0 } (see (142)) and π ⋆,ϕ h ( ϕ | 0) = 1 for all h ∈ [ H ] , (ii) follows from the fact d ⋆,P 1 (0 , ϕ ) = 1 , (iii) is verified in (155), and the last equality arises from the definition in (139).

- Similarly, for h = 2 , we arrive at

<!-- formula-not-decoded -->

where (i) holds by the optimal policy in (164) and the trivial fact that d ⋆,P 2 ( s ) = 0 for all s ∈ S \ { 0 , 1 } (see (142) and (149a)), (ii) arises from (155), and the last equality comes from (139).

- For all other steps h = 3 , . . . , H , observing from the deterministic transition kernels in (149b), it can be easily verified that

<!-- formula-not-decoded -->

Combining the above cases, we complete the proof by

<!-- formula-not-decoded -->

## B.3.12 Proof of the claim (168)

Recall that by virtue of (162) and (164), we arrive at

<!-- formula-not-decoded -->

Applying (163) yields

<!-- formula-not-decoded -->

where the last equality uses the definition (162). Therefore, it boils down to control p -q .

To continue, we define an auxiliary value function vector V ∈ R S × 1 obeying

<!-- formula-not-decoded -->

With this in hand, applying Lemma 4 gives

<!-- formula-not-decoded -->

where (i) follows from (see the definition of p in (160))

<!-- formula-not-decoded -->

Here, (ii) holds by letting

<!-- formula-not-decoded -->

The rest of the proof is then to control (229). We start with the observation that λ ⋆ &gt; 0 ; this is because in view of Lemma 5 (cf. (66)), it suffices to verify that

<!-- formula-not-decoded -->

where (i) holds by (158). We now claim the following bound for λ ⋆ holds, whose proof is postponed to the end:

<!-- formula-not-decoded -->

which immediately implies the following by taking exponential maps given λ ⋆ &gt; 0 :

<!-- formula-not-decoded -->

Moving to the second term of (229), it follows that

<!-- formula-not-decoded -->

where (i) follows from the definitions in (149) and (228), (ii) holds by log(1 + x ) &lt; x for x ∈ ( -1 , ∞ ) , (iii) can be verified by (233), β ≥ 4 , and (151):

<!-- formula-not-decoded -->

and the last line uses ( 1 α +∆ ) 3 β = ( 1 α +∆ ) 6 / log ( 1 α +∆ ) = e 6 by the definition of β in (152). Plugging (232) and (234) back into (229) and (227), we arrive at

<!-- formula-not-decoded -->

where (i) holds by (232) and the last equality follows directly from the choice of ∆ in (167).

Proof of inequality (232) . Applying (65) in Lemma 5 to λ ⋆ in (230) leads to the upper bound in (232):

<!-- formula-not-decoded -->

where the last inequality holds by (158). As a result, we shall focus on showing the lower bounds in (232) in the remainder of the proof.

Recalling the definition of q in (150), we can reparameterize 1 -q using two positive variables c q and λ q (whose choices will be made clearer soon) as follows:

<!-- formula-not-decoded -->

Deriving the first derivative of the function of interest f ( λ ) in (230) as follows:

<!-- formula-not-decoded -->

where (i) holds by the chosen transition kernels in (149) and the last line arises from basic calculus. To continue, when λ = λ q , the derivative of the function f ( λ ) can be expressed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) holds by (236), (ii) follows from the bound of σ in (158), (iii) arises from letting c q = β ≥ 4 and noting the fact 1 / 2 ≤ q &lt; 1 (see (153)), leading to

<!-- formula-not-decoded -->

Finally, the last line holds by 1 /β ≤ 1 4 and log ( 1 α +∆ ) = 2 β (see (152)).

To proceed, note that the function f ( λ ) is concave with respect to λ . Therefore, observing ∇ λ f ( λ ) | λ = λ q ≥ 0 with c q = β , we have λ q ≤ λ ⋆ , which implies (see (236))

<!-- formula-not-decoded -->

The above assertion directly gives

The proof is completed by noticing

<!-- formula-not-decoded -->

where (i) follows from (152), and the last inequality follows from (158) and the fact β ∈ [4 , ∞ ) .

## C Analysis: discounted infinite-horizon RMDPs

## C.1 Proof of Lemma 2

We shall first show that the operator ̂ T σ pe ( · ) (cf. (46)) is a γ -contraction, which will in turn imply the existence of the unique fixed point of ̂ T σ pe ( · ) . Before starting, suppose that the entries of Q 1 , Q 2 ∈ R SA are all bounded in [ 0 , 1 1 -γ ] for all ( s, a ) ∈ S × A . Denote that

<!-- formula-not-decoded -->

Proof of γ -contraction. We first show that ̂ T σ pe ( · ) is a γ -contraction. Towards this, instead of ̂ T σ pe ( · ) , we begin with a simpler operator ˜ T σ pe ( · ) , defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which consequently leads to

<!-- formula-not-decoded -->

It follows straightforwardly that

<!-- formula-not-decoded -->

and hence it suffices to establish the γ -contraction of ˜ T σ pe ( · ) . With this in mind, we observe that

<!-- formula-not-decoded -->

where the first equality holds by the definition of ˜ T σ pe ( · ) (cf. (243)), (i) follows from that the infimum operator is a 1 -contraction w.r.t. ∥ · ∥ ∞ and ∥P V 1 - P V 2 ∥ ∞ ≤ ∥ V 1 -V 2 ∥ ∞ for all P ∈ ∆( S ) , (ii) arises from the definitions in (242), and the last inequality is due to the maximum operator is also a 1 -contraction w.r.t. ∥ · ∥ ∞ . Combining the above two inequalities establish the desired statement.

Existence of the unique fixed point. To continue, we shall first claim that there exists at least one fixed point of ̂ T σ pe ( · ) . This is a standard argument, which we omit for brevity; interested readers are encouraged to refer to, e.g. Li et al. (2022), for details.

To prove the uniqueness of the fixed points of ̂ T σ pe ( · ) , suppose that there exist two fixed points Q ′ and Q ′′ obeying obeying Q ′ = ̂ T σ pe ( Q ′ ) and Q ′′ = ̂ T σ pe ( Q ′′ ) . Moreover, the definition of ̂ T σ pe ( · ) directly implies 0 ≤ Q ′ , Q ′′ ≤ 1 1 -γ , since for any 0 ≤ Q ≤ 1 1 -γ , it follows that 0 ≤ ̂ T σ pe ( Q ) ≤ 1 1 -γ . By the γ -contraction property, it follows that

<!-- formula-not-decoded -->

However, (247) can't happen given γ ∈ [ 1 2 , 1 ) , indicating the uniqueness of the fixed points of ̂ T σ pe ( · ) .

## C.2 Proof of Lemma 3

To begin with, considering any Q,Q ′ obeying Q ≤ Q ′ , and 0 ≤ Q,Q ′ ≤ 1 1 -γ . We observe that the operator ̂ T σ pe ( · ) (cf. (46)) has the monotone non-decreasing property, namely,

<!-- formula-not-decoded -->

where the last line uses Q ≤ Q ′ . Recalling the fixed point ̂ Q ⋆,σ pe of ̂ T σ pe ( · ) , armed with (248) and the initialization ̂ Q 0 = 0 , we arrive at

<!-- formula-not-decoded -->

where the inequality follows from ̂ Q 0 = 0 ≤ ̂ Q ⋆,σ pe . Implementing the above result recursively gives

<!-- formula-not-decoded -->

Applying the γ -contraction property in Lemma 2 thus yields that for any m ≥ 0 ,

<!-- formula-not-decoded -->

where the last inequality holds by the fact ∥ ̂ Q ⋆,σ pe ∥ ∞ ≤ 1 1 -γ (see Lemma 2).

## C.3 Proof of Theorem 3

To begin, we introduce some additional notation that will be useful throughout the analysis. We denote the state-action space covered by the batch dataset D as

<!-- formula-not-decoded -->

In addition, recalling the definition in (47), we define a similar one based on the true nominal model P 0 as

<!-- formula-not-decoded -->

which directly indicates that

<!-- formula-not-decoded -->

Next, we denote the set of possible state occupancy distributions associated with the optimal policy π ⋆ in a model within the uncertainty set P ∈ U σ ( P 0 ) as

<!-- formula-not-decoded -->

where the second equality is due to the fact that π ⋆ is chosen to be deterministic.

We are now ready to embark on the proof of Theorem 3. We first introduce a fact that is used throughout the proof; the proof is postponed to Appendix C.3.2:

<!-- formula-not-decoded -->

as long as (58) holds.

For notation simplicity, denote the output Q-function and value function from Algorithm 3 as ̂ Q = ̂ Q M and ̂ V = ̂ V M . Invoking Lemma 3 with M ≥ log σN 1 -γ log 1 γ directly leads to

<!-- formula-not-decoded -->

and therefore

<!-- formula-not-decoded -->

The proof of Theorem 3 is separated into several key steps as follows.

Step 1: controlling the uncertainty via leave-one-out analysis. Given access to only a finite number of samples for estimating the nominal transition kernel P 0 , we need to efficiently control

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

across the robust value iterations, where ̂ V is statistically dependent on ̂ P 0 s,a (since ̂ P 0 s,a will be reused in the update rule (cf. (51)) for all the iterations). A naive treatment via the standard covering arguments will unfortunately lead to rather loose bounds (Panaganti and Kalathil, 2022; Yang et al., 2022; Zhou et al., 2021). To overcome this challenge, we resort to the leave-one-out analysis-pioneered by Agarwal et al. (2020); Li et al. (2022, 2020) in the context of model-based RL-to decouple the statistical dependency. The results are summarized in the following lemma, with the proof provided in Appendix C.3.1.

Lemma 15. Instate the assumptions in Theorem 3. Then for all vector ˜ V obeying ∥ ∥˜ V -̂ V ⋆,σ pe ∥ ∥ ∞ ≤ 1 σN and ∥ ˜ V ∥ ∞ ≤ 1 1 -γ , with probability at least 1 -δ , one has

<!-- formula-not-decoded -->

for all ( s, a ) ∈ S × A . In addition, for all ( s, a ) ∈ C b , with probability at least 1 -δ , one has

<!-- formula-not-decoded -->

Step 2: establishing the pessimism property. Armed with Lemma 15, we aim to show the key property that

<!-- formula-not-decoded -->

Similar to the finite-horizon setting, it suffices to focus on verifying the former assertion in (258). Towards this, we first recall that the fixed point ̂ Q ⋆,σ pe of the pessimistic robust Bellman operator ̂ T σ pe ( · ) (cf. (46)) obeys

<!-- formula-not-decoded -->

If ̂ Q ⋆,σ pe ( s, a ) = 0 . Given the initialization ̂ Q 0 = 0 , invoking Lemma 3 gives

<!-- formula-not-decoded -->

As a result, Q ̂ π,σ ( s, a ) ≥ 0 = ̂ Q ( s, a ) as desired. Therefore, it boils down to examine the case when ̂ Q ⋆,σ pe ( s, a ) &gt; 0 . One has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) follows from (254), (ii) arises from (255) and the basic fact that infimum operator is 1 -contraction w.r.t ∥ · ∥ ∞ , and the last inequality holds by the definition of b ( s, a ) (cf. (48)) and Lemma 15. Putting the above inequality together with the robust Bellman equation (cf. (37a)) pertaining to Q ̂ π,σ ( s, a ) , we arrive at

<!-- formula-not-decoded -->

where (i) holds by setting ˜ P s,a = argmin P∈U σ ( P 0 s,a ) P V ̂ π,σ . Consequently, one has

<!-- formula-not-decoded -->

where (i) follows from ˜ P s,a ∈ ∆( S ) for all ( s, a ) ∈ S × A . Noting that 0 ≤ γ &lt; 1 , we conclude Q ̂ π,σ ( s, a ) -̂ Q ( s, a ) ≥ 0 for all ( s, a ) ∈ S × A . This establishes the claim (258).

Step 3: bounding V ⋆,σ ( ρ ) -V ̂ π,σ ( ρ ) . In view of the pessimistic property (cf. (258)), it follows that

<!-- formula-not-decoded -->

Towards this, note that

<!-- formula-not-decoded -->

where (i) follows from (254), (ii) holds by applying (259), (iii) arises from (255), and the basic fact that the infimum operator is a 1 -contraction w.r.t. ∥ · ∥ ∞ , and the final inequality holds by the definition of b ( s, a ) (see (48)) and Lemma 15.

To continue, invoking the robust Bellman optimality equation in (37b) gives

<!-- formula-not-decoded -->

Combining the above relation with (263), we arrive at

<!-- formula-not-decoded -->

where the final inequality holds evidently, by introducing

<!-- formula-not-decoded -->

Before continuing, for convenience, let us introduce a matrix ̂ P inf ∈ R S ×S and a vector b ⋆ ∈ R S , where their s -th rows (resp. entries) are defined as

<!-- formula-not-decoded -->

With these notation in hand, averaging (264) over the initial state distribution ρ leads to

<!-- formula-not-decoded -->

Applying the above result recursively gives

<!-- formula-not-decoded -->

where (i) holds by ∣ ∣ ρ ⊤ ( ̂ P inf ) i ( V ⋆,σ -̂ V ) ∣ ∣ ≤ 1 1 -γ for all i ≥ 0 , and that lim i →∞ γ i ρ ⊤ ( ̂ P inf ) i ( V ⋆,σ -̂ V ) = 0 since lim i →∞ γ i = 0 for all 0 ≤ γ &lt; 1 .

To further characterize the above performance gap, invoking the definition of d ⋆,P (cf. (38) and (39a)), we arrive at

<!-- formula-not-decoded -->

Plugging the above expression back into (268), and combining with(262), yields

<!-- formula-not-decoded -->

Step 4: controlling 〈 d ⋆, ̂ P inf , b ⋆ 〉 using concentrability. Note that ̂ P inf ∈ U σ ( P 0 ) (see (265) and (266)), which in words means ̂ P inf is some transition kernel inside U σ ( P 0 ) - the uncertainty set around the nominal kernel P 0 . Similar to the finite-horizon case, observing that we can express 〈 d ⋆, ̂ P inf , b ⋆ 〉 = ∑ s ∈S d ⋆, ̂ P inf ( s ) b ⋆ ( s ) , we divide the states into two cases and control them separately.

- Case 1: s ∈ S where max P ∈U σ ( P 0 ) d ⋆,P ( s, π ⋆ ( s ) ) = 0 . Since ̂ P inf ∈ U σ ( P 0 ) , one has

<!-- formula-not-decoded -->

which consequently indicates

<!-- formula-not-decoded -->

- Case 2: s ∈ S where max P ∈U σ ( P 0 ) d ⋆,P ( s, π ⋆ ( s ) ) &gt; 0 . For any such state s , we claim that

<!-- formula-not-decoded -->

This is due to Assumption 2, which requires C ⋆ rob to be finite given the numerator is positive:

<!-- formula-not-decoded -->

To continue, invoking the fact in (253) with ( s, π ⋆ ( s ) ) ∈ C b gives

<!-- formula-not-decoded -->

where (i) holds by Assumption 2, and the last inequality holds by ̂ P inf ∈ U σ ( P 0 ) . With this in mind, we can control the pessimistic penalty b ⋆ ( s ) (cf. (48)) by

<!-- formula-not-decoded -->

where (i) arises from (257), the penultimate inequality follows from (274), and the last inequality holds as long as c b is large enough.

Summing up the above two cases, we arrive at

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) arises from Cauchy-Schwarz inequality, and the last inequality holds since P min ( s, π ⋆ ( s ) ) ≥ P ⋆ min for all s ∈ S (see (251)) and the following fact (which has been established in (98)):

<!-- formula-not-decoded -->

Finally, inserting (275) back into (270), with probability at least 1 -2 δ , one has

<!-- formula-not-decoded -->

which concludes the proof.

## C.3.1 Proof of Lemma 15

We first note that the second assertion in (257) is the counterpart of (80), which can be verified following the same argument in Appendix B.2.1. For brevity, we omit its proof, and shall focus on verifying (256).

To begin with, we consider the situation when N ( s, a ) = 0 . In this case, (256) can be easily verified since

<!-- formula-not-decoded -->

where (i) follows from the fact ̂ P 0 s,a = 0 when N ( s, a ) = 0 (see (44)), and (ii) arises from the assumption ∥ V ∥ ∞ ≤ 1 1 -γ . Consequently, in the remainder of the proof, we focus on verifying (256) when N ( s, a ) &gt; 0 . Let us first introduce the counterpart of the claim (79) in Lemma 10 as follows.

Lemma 16. For all ( s, a ) ∈ S × A with N ( s, a ) &gt; 0 , consider any vector V ∈ R S independent of ̂ P 0 s,a obeying ∥ V ∥ ∞ ≤ 1 1 -γ . With probability at least 1 -δ , one has

<!-- formula-not-decoded -->

Proof. The proof follows from the same arguments in Appendix B.2.2, with small modifications to adapt to the infinite-horizon setting; we omit the details for conciseness.

Armed with the above point-wise concentration bound, we are now ready to derive the uniform concentration bound desired as in Lemma 15, counting on a leave-one-out argument divided into the following steps. The crux of the analysis is to construct a set of auxiliary RMDPs, each different from the empirical RMDP only at a single state but possessing crucial statistical independence that facilitates the concentration arguments, which can then be transferred back to the empirical RMDP via a simple triangle inequality.

Step 1: construction of auxiliary RMDPs with state-absorbing empirical nominal transitions. Denote the empirical infinite-horizon robust MDP with the nominal transition kernel ̂ P 0 as ̂ M rob . Then, for each state s and each scalar u ≥ 0 , we can construct an auxiliary robust MDP ̂ M s,u rob so that it is the same as ̂ M rob except the properties in state s . To be precise, let the nominal transition kernel and reward function of ̂ M s,u rob be P s,u and r s,u , which are given respectively as

̸

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

̸

Clearly, state s of the auxiliary ̂ M s,u rob is absorbing, meaning that the state stays at s once entering it. This removes the randomness of ̂ P 0 s,a for all a ∈ A in state s , a key property we will leverage later.

With the robust MDP ̂ M s,u rob in hand, we still need to complete the design by defining the corresponding penalty term for all ( ˜ s, a ) ∈ S × A , which is given as follows

<!-- formula-not-decoded -->

where P s,u min ( ˜ s, a ) is defined as the smallest positive state transition probability over the nominal kernel P s,u ( · | ˜ s, a ) :

<!-- formula-not-decoded -->

̸

In view of (278) and (47), it holds that P s,u min ( ˜ s, a ) = ̂ P min ( ˜ s, a ) , and therefore b s,u ( ˜ s, a ) = b ( ˜ s, a ) , when ˜ s = s for any u ≥ 0 . Armed with the above definitions, the pessimistic robust Bellman operator ̂ T σ s,u ( Q )( · ) of the RMDP ̂ M s,u rob is defined as

<!-- formula-not-decoded -->

Step 2: fixed-point equivalence between ̂ M rob and the auxiliary RMDP ̂ M s,u rob . Recall that ̂ Q ⋆,σ pe is the unique fixed point of ̂ T σ pe ( · ) with the corresponding value ̂ V ⋆,σ pe . We claim that there exists some choice of u such that the fixed point of ̂ T σ s,u ( Q )( · ) coincides with that of ̂ T σ pe ( · ) . In particular, given a state s , we show the following choice of u suffices:

<!-- formula-not-decoded -->

Towards this, we shall break our arguments in two different cases.

̸

- For state s ′ = s . In this case, for any a ∈ A , it can be verified that

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

where the second line follows from the definitions in (279) and (278) as well as b s,u ⋆ ( s ′ , a ) = b ( s ′ , a ) when s ′ = s , the last line arises from the definition of the pessimistic Bellman operator (46), and that ̂ Q ⋆,σ pe is the fixed point.

- For state s . In this case, for any u and a ∈ A , observing that P s,u ( s ′ | s, a ) has only one positive entry equal to 1 (cf. (278)), applying (281) yields

<!-- formula-not-decoded -->

Plugging the above fact into (280) leads to

<!-- formula-not-decoded -->

for all a ∈ A . As a result, we have for any a ∈ A :

<!-- formula-not-decoded -->

where the second line follows from the fact that P s,u ⋆ s,a is a singleton distribution at state s , and hence U σ ( P s,u ⋆ s,a ) = P s,u ⋆ s,a by the definition of the KL uncertainty set, and the second line follows from plugging in the definition of u ⋆ in (283) and b s,u ⋆ ( s, a ) in (286).

Summing up the above two cases, we establish that there exists a fixed point ̂ Q ⋆,σ s,u ⋆ of the operator ̂ T σ s,u ⋆ ( · ) if we let

̸

<!-- formula-not-decoded -->

Consequently, we confirm the existence of a fixed point of the operator ̂ T σ s,u ⋆ ( · ) . In addition, its corresponding value function ̂ V ⋆,σ s,u ⋆ also coincides with ̂ V ⋆,σ pe .

Step 3: building an ε -net for all reward values u . It is easily verified that the reward u ⋆ obeys

<!-- formula-not-decoded -->

As a result, we construct an ε -net (Vershynin, 2018) of the line segment within the range [ 0 , 2 σ + 2 1 -γ ] with ε = 1 σN as follows:

<!-- formula-not-decoded -->

Armed with this covering net U ε , we can construct an auxiliary robust MDP ̂ M s,u rob and its corresponding pessimistic robust Bellman operator for each u ∈ U ε (see Step 1). Following the same arguments in the proof of Lemma 2 (cf. Appendix C.1), for each u ∈ U ε , it can be verified that there exists a unique fixed point

̂ Q ⋆,σ s,u of the operator ̂ T σ s,u ( · ) , which satisfies 0 ≤ ̂ Q ⋆,σ s,u ≤ 1 1 -γ · 1 . In turn, the corresponding value function also satisfies ∥ ̂ V ⋆,σ s,u ∥ ∞ ≤ 1 1 -γ .

In view of the definitions in (278) and (279), for all u ∈ U ε , ̂ M s,u rob is statistically independent from ̂ P 0 s,a , which indicates the independence between ̂ V ⋆,σ s,u and ̂ P 0 s,a . This makes it possible to invoke Lemma 16, and taking the union bound over all samples N and u ∈ U ε give that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

hold simultaneously for all ( s, a, u ) ∈ S × A × U ε with N ( s, a ) &gt; 0 .

Step 4: a covering argument. Recalling that u ⋆ ∈ [ 0 , 2 σ + 2 1 -γ ] (see (289)), we can always find some ˜ u ∈ U ε such that | ˜ u -u ⋆ | ≤ 1 σN . Consequently, plugging in the operator in (282) yields

<!-- formula-not-decoded -->

̸

where (i) holds by b s, ˜ u ( s, a ) = b s,u ⋆ ( s, a ) for s (see (286)) and b s, ˜ u ( s ′ , a ) = b s,u ⋆ ( s ′ , a ) = b ( s ′ , a ) for all s ′ = s . With this in mind, we observe that the fixed points of ̂ T σ s, ˜ u ( · ) and ̂ T σ s,u ⋆ ( · ) obey

<!-- formula-not-decoded -->

which directly indicates that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Armed with the above facts, invoking the identity ̂ V ⋆,σ pe = ̂ V ⋆,σ s,u ⋆ established in Step 2 gives

<!-- formula-not-decoded -->

where (i) holds by applying the triangle inequality, (ii) arises from (295) and the basic fact that infimum operator is a 1 -contraction w.r.t. ∥ · ∥ ∞ , and the final inequality follows from (291).

and

Step 5: finishing up. Now we are positioned to finish up the proof. For all vector ˜ V obeying ∥ ∥˜ V -̂ V ⋆,σ pe ∥ ∥ ∞ ≤ 1 σN and ∥ ˜ V ∥ ∞ ≤ 1 1 -γ , we apply the triangle inequality and invoke (296) to reach

<!-- formula-not-decoded -->

Finally, we complete the proof by verifying that

<!-- formula-not-decoded -->

## C.3.2 Proof of (253)

For all ( s, a ) ∈ C b , one has

<!-- formula-not-decoded -->

where (i) follows from the condition (58), (ii) arises from the definition that d b min ≤ d b ( s, a ) for all ( s, a ) ∈ C b , and (iii) follows from the definition in (251). In particular, when c 1 is large enough, one has 2 3 log NS δ &lt; Nd b ( s,a ) 12 . To continue, we recall a key property of N ( s, a ) (cf. (43)) in the following lemma.

Lemma 17 ((Li et al., 2022, Lemma 7)) . Fix δ ∈ (0 , 1) . With probability at least 1 -δ , the quantities { N ( s, a ) } in (43) obey

<!-- formula-not-decoded -->

simultaneously for all ( s, a ) ∈ S × A .

Consequently, Lemma 17 tells us that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

as long as c 1 is large enough. Last but not least, taking the basic fact x ≤ -log(1 -x ) for all x ∈ [0 , 1] , the last inequality of (253) can be verified by

<!-- formula-not-decoded -->

## C.4 Proof of Theorem 4

Similar to the finite-horizon case, we shall develop the lower bounds for the two cases when the uncertainty levels σ vary separately.

## C.4.1 Construction of hard problem instances: small uncertainty level

We first construct some hard discounted infinite-horizon RMDP instances and then characterize the sample complexity requirements over these instances.

Construction of a collection of hard MDPs. Suppose there are two MDPs

<!-- formula-not-decoded -->

Here, S = { 0 , 1 , . . . , S -1 } , and A = { 0 , 1 } . The transition kernel P θ of the MDP M θ is specified as follows:

<!-- formula-not-decoded -->

for any ( s, a, s ′ ) ∈ S × A × S× , where p and q are set to be

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which satisfies for some c 1 ≤ 1 4 and small enough c 2 . In view of the assumptions (305), one has

<!-- formula-not-decoded -->

Finally, we define the reward function as

<!-- formula-not-decoded -->

Construction of the history/batch dataset. Define a useful state distribution (only supported on the state subset { 0 , 1 , 2 } ) as

<!-- formula-not-decoded -->

where C &gt; 0 is some constant that determines the robust concentrability coefficient C ⋆ rob (which will be made clear soon) and obeys

<!-- formula-not-decoded -->

A batch dataset-consists of N i.i.d samples { ( s i , a i , s ′ i ) } 1 ≤ i ≤ N -is generated over the nominal environment M θ according to (40), with the behavior distribution chosen to be:

<!-- formula-not-decoded -->

Additionally, we choose the following initial state distribution:

<!-- formula-not-decoded -->

Uncertainty set of the transition kernels. We next describe the radius σ of the uncertainty set in our construction of the robust MDPs, along with some useful properties, which are similar to the finite-horizon case. The perturbed transition kernels in M θ is limited to the following uncertainty set

<!-- formula-not-decoded -->

where P θ s,a := P θ ( · | s, a ) ∈ [0 , 1] 1 × S . Moreover, the radius of the uncertainty set σ obeys

<!-- formula-not-decoded -->

For any ( s, a, s ′ ) ∈ S × A × S , we denote the infimum entry of the perturbed transition kernel P s,a ∈ U σ ( P θ s,a ) moving to the next state s ′ as

<!-- formula-not-decoded -->

As shall be seen, the transition from state 0 to state 2 plays an important role in the analysis, for convenience, we denote

<!-- formula-not-decoded -->

With these definitions in place, we summarize some useful properties of the uncertainty set in the following lemma, which parallels Lemma 11 in the finite-horizon case.

Lemma 18. Suppose the uncertainty level σ satisfies (313) . The perturbed transition kernels obey

<!-- formula-not-decoded -->

for constant c 3 = 2 c 1 ≤ 1 4 .

Proof. The proof follows from the same arguments as the proof for Lemma 11 in Appendix B.3.5 by replacing H with 1 1 -γ ; we omit the details for brevity.

Value functions and optimal policies. Now we are positioned to derive the corresponding robust value functions and identify the optimal policies. For any MDP M θ with the above uncertainty set, denote π ⋆ θ as the optimal policy. In addition, we denote the robust value function of any policy π (resp. the optimal policy π ⋆ θ ) as V π,σ θ (resp. V ⋆,σ θ ). Then, we introduce the following lemma which describes some important properties of the robust value functions and optimal policies.

Lemma 19. For any θ = { 0 , 1 } and any policy π , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, the optimal value functions and the optimal policies obey

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, the robust single-policy clipped concentrability coefficient C ⋆ rob obeys

<!-- formula-not-decoded -->

where x π θ is defined as

Proof. See Appendix C.4.5.

## C.4.2 Establishing the minimax lower bound: small uncertainty level

Towards this, we first introduce the following lemma, which parallels the claim in (167)-(168) in the finitehorizon case.

Lemma 20. For any policy ̂ π ,

<!-- formula-not-decoded -->

Proof. This lemma can be directly verified by controlling V ⋆,σ θ (0) -V ̂ π,σ θ (0) with the help of Lemma 18 and Lemma 19; we omit the details for brevity.

Armed with this lemma, following the same arguments in Appendix B.3.4, we can complete the proof by observing that: let c 1 be some sufficient large constant, as long as the sample size is beneath

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where P θ denote the probability conditioned on that the MDP is M θ . We omit the details for brevity and complete the proof.

## C.4.3 Construction of hard problem instances: large uncertainty level

Construction of a collection of hard MDPs. Suppose there are two MDPs

<!-- formula-not-decoded -->

Here, γ is the discount parameter, S = { 0 , 1 , . . . , S -1 } is the state space, and A = { 0 , 1 } is the action space. The transition kernel P ϕ of either constructed MDP M ϕ is defined as

<!-- formula-not-decoded -->

where p and q are set as for some γ , α and ∆ obeying

then we necessarily have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, α and ∆ are some values that will be introduced later. Consequently, applying (324) directly leads to

<!-- formula-not-decoded -->

Note that state 1 and 2 are absorbing states. In addition, if the initial distribution is supported on states { 0 , 1 , 2 } , the MDP will always stay in the state { 1 , 2 } after the first transition.

Finally, we define the reward function as

<!-- formula-not-decoded -->

Construction of the history/batch dataset. Define a useful state distribution (only supported on the state subset { 0 , 1 , 2 } ) as

<!-- formula-not-decoded -->

where C &gt; 0 is some constant that determines the robust concentrability coefficient C ⋆ rob (which will be made clear soon) and obeys

<!-- formula-not-decoded -->

A batch dataset-consists of N i.i.d samples { ( s i , a i , s ′ i ) } 1 ≤ i ≤ N -is generated over the nominal environment M ϕ according to (40), with the behavior distribution chosen to be:

<!-- formula-not-decoded -->

Additionally, we choose the following initial state distribution:

<!-- formula-not-decoded -->

Uncertainty set of the transition kernels. We next describe the radius σ of the uncertainty set in our construction of the robust MDPs, along with some useful properties, which are similar to the finite-horizon case. To begin with, with slight abuse of notation, we introduce an important constant β defined as

<!-- formula-not-decoded -->

The perturbed transition kernels in M ϕ is limited to the following uncertainty set

<!-- formula-not-decoded -->

where P ϕ s,a := P ϕ ( · | s, a ) ∈ [0 , 1] 1 × S . Moreover, the radius of the uncertainty set σ obeys

<!-- formula-not-decoded -->

For any ( s, a, s ′ ) ∈ S × A × S , we denote the infimum entry of the perturbed transition kernel P s,a ∈ U σ ( P ϕ s,a ) moving to the next state s ′ as

<!-- formula-not-decoded -->

As shall be seen, the transition from state 0 to state 2 plays an important role in the analysis, for convenience, we denote

<!-- formula-not-decoded -->

With these definitions in place, we summarize some useful properties of the uncertainty set in the following lemma, which parallels Lemma 13 in the finite-horizon case.

Lemma 21. Suppose β satisfies (332) and the uncertainty level σ satisfies (334) . The perturbed transition kernels obey

<!-- formula-not-decoded -->

Proof. The proof follows from the same arguments as Appendix B.3.10 by replacing H with 1 1 -γ ; we omit the details for brevity.

Value functions and optimal policies. Now we are positioned to derive the corresponding robust value functions and identify the optimal policies. For any MDP M ϕ with the above uncertainty set, denote π ⋆ ϕ as the optimal policy. In addition, we denote the robust value function of any policy π (resp. the optimal policy π ⋆ ϕ ) as V π,σ ϕ (resp. V ⋆,σ ϕ ). Then, we introduce the following lemma which describes some important properties of the robust value functions and optimal policies.

Lemma 22. For any ϕ = { 0 , 1 } and any policy π , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, the optimal value functions and the optimal policies obey

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, choosing S ≥ 2 β , the robust single-policy clipped concentrability coefficient C ⋆ rob obeys

<!-- formula-not-decoded -->

where z π ϕ is defined as

Proof. See Appendix C.4.6.

## C.4.4 Establishing the minimax lower bound: large uncertainty level

Now we are positioned to provide the sample complexity lower bound. In view of Lemma 22, the smallest positive state transition probability of the optimal policy π ⋆ ϕ under any nominal transition kernel P ϕ with ϕ ∈ { 0 , 1 } satisfies:

<!-- formula-not-decoded -->

Our goal is to control the quantity w.r.t. any policy estimator ̂ π based on the batch dataset and the chosen initial distribution ρ in (331), which gives

<!-- formula-not-decoded -->

Towards this, we first introduce the following lemma, which parallels the claim in (167)-(168) in the finitehorizon case.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. This lemma follows from the same arguments as Appendix B.3.12 except replacing H with 1 1 -γ under the additional condition γ ≥ 1 2 ; we omit the details for brevity.

Armed with this lemma, following the same arguments in Appendix B.3.4, we can complete the proof by observing that: let c 1 be some sufficient large constant, as long as the sample size is beneath

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where P ϕ denote the probability conditioned on that the MDP is M ϕ . We omit the details for brevity and complete the proof.

<!-- formula-not-decoded -->

then we necessarily have

## C.4.5 Proof of Lemma 19

First, it is easily verified that for any policy π ,

<!-- formula-not-decoded -->

since the reward function r ( s, a ) = 0 for ( s, a ) ∈ S \ { 0 } × A in (307).

To continue, we observe that the robust value function of state 0 satisfies

<!-- formula-not-decoded -->

where (i) holds by the reward function defined in (307). To see (ii), note that (347) indicates V π,σ θ (0) ≥ 1 ≥ V π,σ θ (1) = 0 , so that the infimum is obtained by picking the smallest possible mass on the transition to state 2 , provided by the definition in (315), and (iii) follows by plugging in the definition of x π θ in (318).

Consequently, observing that the function 1 1 -γx is increasing in x and x π θ is also increasing in π ( θ | 0) (see the fact p ⋆ ≥ q ⋆ in (316)), the optimal policy in state 0 thus obeys

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which combined with (348) yields

Therefore,

<!-- formula-not-decoded -->

Regarding the optimal policy for the remaining states s &gt; 0 , since the action does not influence the state transition, without loss of generality, we choose the optimal policy to obey

<!-- formula-not-decoded -->

Proof of (320) . To begin with, for any MDP M θ with θ ∈ { 0 , 1 } , recall the definition of C ⋆ rob as

<!-- formula-not-decoded -->

Given π ⋆ θ ( θ | s ) = 1 for all s ∈ S and the initial distribution ρ (0) = 1 , for any P ∈ U σ ( P θ ) , we arrive at

<!-- formula-not-decoded -->

where (i) holds by (315) and (ii) follows from (316). In addition, we have

<!-- formula-not-decoded -->

since ρ (0) = 1 and state 0 and 1 are absorbing states for all policy and and all P ∈ U σ ( P θ ) . Armed with the above facts, we observe that

<!-- formula-not-decoded -->

which follows from the properties of the optimal policy in (319).

Consequently, we control C ⋆ rob in states separately:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) holds by (354) and S ≥ 2 , (ii) and (iii) follow from the definitions in (310) and (308), and (iv) and (v) and arise from the assumption in (309). Plugging the above results back into (356) directly completes the proof of

<!-- formula-not-decoded -->

## C.4.6 Proof of Lemma 22

For any M ϕ with ϕ ∈ { 0 , 1 } , we first characterize the robust value function for any policy π over different states. due to state absorbing, the uncertainty set becomes a singleton containing the nominal distribution at state s = 1 and s = 2 . It is easily observed that for any policy π , the robust value functions at state s = 1 and s = 2 obey

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since r (1 , a ) = 0 and r (2 , a ) = 1 . In addition, for state s &gt; 2 , the perturbed transition kernel is supported on itself and state 1 , both of which receive a reward of 0 by design (327), leading to

<!-- formula-not-decoded -->

Moving onto the remaining states, the robust value function of state 0 satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) holds by the reward function defined in (327). To see (ii), note that (358) indicates V π,σ ϕ (2) ≥ V π,σ ϕ (1) , so that the infimum is obtained by picking the smallest possible mass on the transition to state 2 , provided by the definition in (336). Last but not least, (iii) follows by plugging in the definition of z π ϕ in (339), and the last identity is due to (358). Consequently, taking π = π ⋆ ϕ , we directly arrive at

<!-- formula-not-decoded -->

Observing that the function z γ 1 -γ is increasing in z and z π ϕ is also increasing in π ( ϕ | 0) (see the fact p ≥ q in (337)), the optimal policy in state 0 thus obeys

<!-- formula-not-decoded -->

Finally, plugging the above fact back into (339) leads to

<!-- formula-not-decoded -->

which combined with (360) yields

<!-- formula-not-decoded -->

Regarding the optimal policy for the remaining states s &gt; 0 , since the action does not influence the state transition, without loss of generality, we choose the optimal policy to obey

<!-- formula-not-decoded -->

Proof of (341) . To begin with, for any MDP M ϕ with ϕ ∈ { 0 , 1 } , recall the definition of C ⋆ rob as

<!-- formula-not-decoded -->

Given π ⋆ ϕ ( ϕ | s ) = 1 for all s ∈ S and the initial distribution ρ (0) = 1 , for any P ∈ U σ ( P ϕ ) , we arrive at

<!-- formula-not-decoded -->

which holds due to that the agent transits from state 0 to other states at the first step and then will never go back to state 0 . In addition, one has for any P ∈ U σ ( P ϕ ) ,

<!-- formula-not-decoded -->

where (i) holds by (336) and the final inequality follows from (337) and γ ≥ 1 / 2 . Armed with the above facts, we observe that

<!-- formula-not-decoded -->

which follows from the properties of the optimal policy in (364) and consequently d ⋆,P ( s ) = d ⋆,P ( s, ϕ ) = 0 for all s &gt; 2 and all P ∈ U σ ( P ϕ ) .

To continue, we control the term in states { 0 , 1 , 2 } separately:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) holds by (367) and S ≥ 2 β , (ii), (iii) and (iv) follow from the definitions in (330) and (328), (v) and (vi) arise from the assumption in (329). Plugging the above results back into (368) directly completes the proof of

<!-- formula-not-decoded -->