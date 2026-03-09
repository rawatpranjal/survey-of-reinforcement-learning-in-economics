## Breaking the Sample Size Barrier in Model-Based Reinforcement Learning with a Generative Model

Gen Li ∗ UPenn

Yuting Wei ∗ UPenn

Yuejie Chi † CMU

Yuxin Chen ∗ UPenn

May 2020;

Revised: September 2022

## Abstract

This paper is concerned with the sample efficiency of reinforcement learning, assuming access to a generative model (or simulator). We first consider γ -discounted infinite-horizon Markov decision processes (MDPs) with state space S and action space A . Despite a number of prior works tackling this problem, a complete picture of the trade-offs between sample complexity and statistical accuracy is yet to be determined. In particular, all prior results suffer from a severe sample size barrier, in the sense that their claimed statistical guarantees hold only when the sample size exceeds at least |S||A| (1 -γ ) 2 . The current paper overcomes this barrier by certifying the minimax optimality of two algorithms - a perturbed modelbased algorithm and a conservative model-based algorithm - as soon as the sample size exceeds the order of |S||A| 1 -γ (modulo some log factor). Moving beyond infinite-horizon MDPs, we further study timeinhomogeneous finite-horizon MDPs, and prove that a plain model-based planning algorithm suffices to achieve minimax-optimal sample complexity given any target accuracy level. To the best of our knowledge, this work delivers the first minimax-optimal guarantees that accommodate the entire range of sample sizes (beyond which finding a meaningful policy is information theoretically infeasible).

Keywords: model-based reinforcement learning, minimaxity, policy evaluation, generative model

## Contents

| 1 Introduction   | 1 Introduction                                           | 1 Introduction                                               |   2 |
|------------------|----------------------------------------------------------|--------------------------------------------------------------|-----|
| 2                | Problem formulation                                      | Problem formulation                                          |   5 |
|                  | 2.1                                                      | Discounted infinite-horizon Markov decision processes        |   5 |
|                  | 2.2                                                      | Finite-horizon Markov decision processes . . . . . . . .     |   6 |
|                  | 2.3                                                      | Notation . . . . . . . . . . . . . . . . . . . . . . . . . . |   7 |
| 3                | Model-based planning in discounted infinite-horizon MDPs | Model-based planning in discounted infinite-horizon MDPs     |   7 |
|                  | 3.1                                                      | Model-based reinforcement learning: two algorithms .         |   7 |
|                  | 3.2                                                      | Theoretical guarantees . . . . . . . . . . . . . . . . . .   |   8 |
|                  | 3.3                                                      | Comparisons with prior works and implications . . . .        |   9 |
| 4                | Model-based planning in finite-horizon MDPs              | Model-based planning in finite-horizon MDPs                  |  10 |
|                  | 4.1                                                      | Algorithm: model-based planning . . . . . . . . . . . .      |  10 |
|                  | 4.2                                                      | Theoretical guarantees and implications . . . . . . . .      |  10 |

5

Other related works

11

∗ Department of Statistics and Data Science, Wharton School, University of Pennsylvania, Philadelphia, PA 19104, USA.

† Department of Electrical and Computer Engineering, Carnegie Mellon University, Pittsburgh, PA 15213, USA.

| 6 Analysis: infinite-horizon MDPs   | 6 Analysis: infinite-horizon MDPs                                                                     | 6 Analysis: infinite-horizon MDPs                                                                     | 6 Analysis: infinite-horizon MDPs                                                                     | 12   |
|-------------------------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|------|
|                                     | 6.1 Matrix notation and Bellman equations . . . . . . . . . . . . . . . . . . . . . . . . . .         | 6.1 Matrix notation and Bellman equations . . . . . . . . . . . . . . . . . . . . . . . . . .         | . . . .                                                                                               | 12   |
|                                     | 6.2 Analysis: model-based policy evaluation . . . . . . . . . . . . . . . . . . . .                   | 6.2 Analysis: model-based policy evaluation . . . . . . . . . . . . . . . . . . . .                   | . . . . . . . . . .                                                                                   | 13   |
|                                     | 6.3 Analysis: perturbed model-based planning . . . . .                                                | 6.3 Analysis: perturbed model-based planning . . . . .                                                | . . . . . . . . . . . . . . . . . . . . . . . .                                                       | 14   |
|                                     | 6.3.1                                                                                                 |                                                                                                       | Value function estimation for a policy obeying Bernstein-type conditions .                            | 15   |
|                                     | 6.3.2                                                                                                 | Decoupling statistical dependency via ( s,a ) -absorbing MDPs                                         | . . . . . . . . . . . . . . . . . . . . .                                                             | 15   |
|                                     | 6.3.3                                                                                                 | A tie-breaking . . .                                                                                  | argument . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                              | 17   |
|                                     | 6.3.4                                                                                                 | Proof of Theorem . . . . . . . .                                                                      | 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                           | 17   |
|                                     | 6.4 Analysis: conservative model-based planning . . . . . . . . . . . . . . . . . . . . . . . . . . . | 6.4 Analysis: conservative model-based planning . . . . . . . . . . . . . . . . . . . . . . . . . . . | 6.4 Analysis: conservative model-based planning . . . . . . . . . . . . . . . . . . . . . . . . . . . | 19   |
|                                     | Analysis:                                                                                             | finite-horizon MDPs                                                                                   | Analysis:                                                                                             | 20   |
|                                     | 7.1 Matrix notation and Bellman equations . . . . . . .                                               | 7.1 Matrix notation and Bellman equations . . . . . . .                                               | . . . . . . . . . . . . . . . . . . . . . . .                                                         | 20   |
|                                     | 7.2 An auxiliary value function sequence obeying Bernstein-type conditions . .                        | 7.2 An auxiliary value function sequence obeying Bernstein-type conditions . .                        | . . . . . . . . . .                                                                                   | 20   |
|                                     | 7.3 Proof of Theorem 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                      | 7.3 Proof of Theorem 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                      | . . . . . . . . . . . . .                                                                             | 21   |
|                                     | Discussion                                                                                            | Discussion                                                                                            | Discussion                                                                                            | 22   |
|                                     | Preliminary facts                                                                                     | Preliminary facts                                                                                     | Preliminary facts                                                                                     | 23   |
|                                     | B Proofs of auxiliary lemmas: infinite-horizon MDPs                                                   | B Proofs of auxiliary lemmas: infinite-horizon MDPs                                                   | B Proofs of auxiliary lemmas: infinite-horizon MDPs                                                   | 24   |
|                                     | B.1 Proofs of Lemma 1 and Lemma 2 .                                                                   | . . . .                                                                                               | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                             | 24   |
|                                     |                                                                                                       | B.1.1 Proof of Lemma 11 . . . . . . . . . . . . . . .                                                 | . . . . . . . . . . . . . . . . . . . . . . .                                                         | 27   |
|                                     | B.2                                                                                                   | Proof of Lemma 3 . . . . . . . . . . . . .                                                            | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                             | 28   |
|                                     | B.3                                                                                                   | Proof of Lemma 4 . . . . . . . . . . . . . . . . . . . . . . . . .                                    | . . . . . . . . . . . . . . . . .                                                                     | 29   |
|                                     | B.4                                                                                                   | Proof of Lemma 5 . . . . . . . . . . . . . . . . . . .                                                | . . . . . . . . . . . . . . . . . . . . . . .                                                         | 29   |
|                                     | B.5                                                                                                   | Proof of Lemma 6 . . . . . . . . . . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . . . . .                                                                   | 30   |
|                                     | B.6                                                                                                   | Proof of Lemma 7 . . . . . . . . . . . . . . .                                                        | . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                 | 33   |
|                                     | B.7                                                                                                   | Proof of Lemma 8 . . . . . . . . . . . . .                                                            | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                             | 33   |
|                                     | C Proofs of auxiliary lemmas: finite-horizon MDPs                                                     | C Proofs of auxiliary lemmas: finite-horizon MDPs                                                     | C Proofs of auxiliary lemmas: finite-horizon MDPs                                                     | 34   |
|                                     | C.1 . . . . . . . . . . . . . . . . . . . . . . . . . .                                               | C.1 . . . . . . . . . . . . . . . . . . . . . . . . . .                                               | . . . . . . . . . . . .                                                                               | 35   |
|                                     | Proof of Lemma 9 . . . .                                                                              | Proof of Lemma 9 . . . .                                                                              | Proof of Lemma 9 . . . .                                                                              |      |

C.2

Proof of Lemma 12 .

## 1 Introduction

Reinforcement learning (RL) (Sutton and Barto, 2018; Szepesvári, 2010), which is frequently modeled as learning and decision making in a Markov decision process (MDP), is garnering growing interest in recent years due to its remarkable success in practice. A core objective of RL is to search for a policy - based on a collection of noisy data samples - that approximately maximizes expected cumulative rewards in an MDP, without direct access to a precise description of the underlying model. 1 In contemporary applications, it is increasingly more common to encounter environments with prohibitively large state and action space, thus exacerbating the challenge of collecting enough samples to learn the model. To enable faithful policy learning in the sample-starved regime (i.e. the regime where the model complexity overwhelms the sample size), it is crucial to obtain a quantitative picture of the fundamental trade-off between sample complexity and statistical accuracy, and to design efficient algorithms that provably achieve the optimal trade-off.

Broadly speaking, there are at least two common algorithmic approaches: a model-based approach and a model-free one. The model-based approach decouples model estimation and policy learning tasks; more specifically, one first estimates the unknown model using the data samples in hand, and then leverages the fitted model to perform planning - a task that can be accomplished by resorting to Bellman's principle of optimality (Bellman, 1952). A notable advantage of model-based algorithms is their flexibility: the learned model can be adapted to perform new ad-hoc tasks without revisiting the data samples. In comparison, the model-free approach attempts to compute the optimal policy (and the optimal value function) without learning the model explicitly, which lends itself well to scenarios when a model is difficult to estimate or

1 Here and throughout, the 'model' refers to the transition kernel and the rewards of the MDP taken collectively.

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

37

changes on the fly. Characterizing the sample efficiency of both approaches has been the focal point of a large body of recent works, e.g. Agarwal et al. (2020); Azar et al. (2013); Jin et al. (2018); Kearns and Singh (1999); Li et al. (2023); Sidford et al. (2018a,b); Tu and Recht (2019); Wainwright (2019a,b).

In this paper, we pursue a comprehensive understanding of model-based RL, given access to a generative model - that is, a simulator that produces samples based on the transition kernel of the true MDP for each state-action pair (Kakade, 2003; Kearns and Singh, 1999). To allow for more precise discussions, we first look at an infinite-horizon discounted MDP with state space S , action space A and discount factor 0 &lt; γ &lt; 1 , and pay particular attention to the scenarios where the sizes of the state/action spaces and the effective horizon 1 1 -γ are all quite large. We obtain N samples per state-action pair by querying the generative model. For an arbitrary target accuracy level ε &gt; 0 , a desired model-based planning algorithm should return an ε -optimal policy with a minimal number of calls to the generative model. Particular emphasis is placed on the sub-linear sampling scenario, in which the total sample size is smaller than the total number |S| 2 |A| of model parameters (so that it is in general infeasible to estimate the model accurately).

Motivation: sample size barriers. Several prior works were dedicated to investigating model-based RL for γ -discounted infinite-horizon MDPs with a generative model, which uncovered the minimax optimality of this approach for an already wide regime (Agarwal et al., 2020; Azar et al., 2013). However, the results therein often suffered from a sample complexity barrier that prevents us from obtaining a complete trade-off curve between sample complexity and statistical accuracy. For instance, the state-of-the-art result Agarwal et al. (2020) required the total sample size to at least exceed |S||A| (1 -γ ) 2 (up to some log factor), thus restricting the validity of the theory for broader contexts. In truth, this is not merely an issue for model-based planning; the same barrier already showed up when analyzing the simpler task of model-based policy evaluation (Agarwal et al., 2020; Pananjady and Wainwright, 2019). Furthermore, an even more severe barrier emerged in prior theory for model-free methods; for instance, Sidford et al. (2018b); Wainwright (2019b) required the sample size to exceed |S||A| (1 -γ ) 3 modulo some log factor. In stark contrast, however, no lower bounds developed thus far preclude us from attaining reasonable statistical accuracy when going below the aforementioned sample complexity barriers, thus resulting in a gap between upper and lower bounds in this sample-starved regime. Noteworthily, such a sample size barrier is not only present for discounted infinite-horizon MDPs; the situation is similar for finite-horizon MDPs (Yin et al., 2021).

Our contributions. The current paper seeks to achieve optimal sample complexity even below the aforementioned sample size barrier. For γ -discounted infinite-horizon MDPs, we propose two model-based algorithms: (i) perturbed model-based planning : which performs planning based on an empirical MDP learned from samples with mild random reward perturbation ; and (ii) conservative model-based planning : which computes approximately optimal policies for the empirical MDP without reward perturbation. These two proposed algorithms provably find an ε -optimal policy with an order of |S||A| (1 -γ ) 3 ε 2 samples (up to log factor), thereby matching the minimax lower bound (Azar et al., 2013). Our result accommodates the full range of accuracy level ε (namely, ε ∈ (0 , 1 1 -γ ] ), thus unveiling the minimaxity of our algorithms as soon as the sample size exceeds |S||A| 1 -γ (modulo some log factor). Encouragingly, this covers the full range of sample sizes that enable one to find a policy strictly better than a random guess. See Table 1 for detailed comparisons with prior literature. Along the way, we also derive minimax-optimal statistical guarantees for policy evaluation, which strengthen state-of-the-art results by broadening the applicable sample size range.

Moving beyond discounted infinite-horizon MDPs, we further characterize the sample efficiency of modelbased planning for time-inhomogeneous finite-horizon MDPs, which provably achieves minimax-optimal sample complexity as well for the full range of target accuracy levels (Domingues et al., 2021). No reward perturbation or conservative action selection is needed for this finite-horizon scenario. See Table 2 for detailed comparisons with prior literature.

On the technical side, our theory for infinite-horizon MDPs is established upon a novel combination of several key ideas: (1) a high-order expansion of the estimation error for value functions, coupled with fine-grained analysis for each term in the expansion; (2) the construction of auxiliary leave-one-out type (state-action-absorbing) MDPs - motivated by Agarwal et al. (2020) - that help decouple the complicated statistical dependency between the empirically optimal policy (as opposed to value functions) and data samples; (3) a tie-breaking argument guaranteeing that the empirically optimal policy is sufficiently separated

from all other policies under reward perturbation. The case with finite-horizon MDPs is also established based on certain high-order expansion of the value estimation errors, in addition to careful variance control for the terms in the expansion.

| Algorithm                                                   | Sample size range            | Sample complexity     | ε -range                |
|-------------------------------------------------------------|------------------------------|-----------------------|-------------------------|
| Phased Q-learning Kearns and Singh (1999)                   | [ |S||A| (1 - γ ) 5 , ∞ )    | |S||A| (1 - γ ) 7 ε 2 | (0 , 1 1 - γ ]          |
| Empirical QVI Azar et al. (2013)                            | [ |S| 2 |A| (1 - γ ) 2 , ∞ ) | |S||A| (1 - γ ) 3 ε 2 | (0 , 1 √ (1 - γ ) |S| ] |
| Sublinear randomized value iteration Sidford et al. (2018b) | [ |S||A| (1 - γ ) 2 , ∞ )    | |S||A| (1 - γ ) 4 ε 2 | ( 0 , 1 1 - γ ]         |
| Variance-reduced QVI Sidford et al. (2018a)                 | [ |S||A| (1 - γ ) 3 , ∞ )    | |S||A| (1 - γ ) 3 ε 2 | (0 , 1]                 |
| Randomized primal-dual method Wang (2019)                   | [ |S||A| (1 - γ ) 2 , ∞ )    | |S||A| (1 - γ ) 4 ε 2 | (0 , 1 1 - γ ]          |
| Empirical MDP + planning Agarwal et al. (2020)              | [ |S||A| (1 - γ ) 2 , ∞ )    | |S||A| (1 - γ ) 3 ε 2 | (0 , 1 √ 1 - γ ]        |
| Perturbed empirical MDP + planning This paper               | [ |S||A| 1 - γ , ∞ )         | |S||A| (1 - γ ) 3 ε 2 | (0 , 1 1 - γ ]          |
| Empirical MDP + conservative planning This paper            | |S||A| 1 - γ , ∞ )           | |S||A| (1 - γ ) 3 ε 2 | (0 , 1 1 - γ ]          |

[

Table 1: Comparisons with prior results (up to log factors) regarding finding an ε -optimal policy in a γ -discounted infinite-horizon MDP with a generative model. The sample size range and the ε -range stand for the range of sample size and optimality gap (e.g. ε -accuracy) for the claimed sample complexity to hold. Note that the results in Kearns and Singh (1999); Wang (2019) only hold for a restricted family of MDPs satisfying certain ergodicity assumptions. In addition, Azar et al. (2013) (resp. Wainwright (2019b)) showed that empirical QVI (resp. variance-reduced Q-learning) finds an ε -optimal Q-function estimate with sample complexity |S||A| (1 -γ ) 3 ε 2 ( ε ∈ (0 , 1] ) in a sample size range [ |S||A| (1 -γ ) 3 , ∞ ) , which did not translate directly to an ε -optimal policy.

| Algorithm                                                   | Sample size range   | Sample complexity   | ε -range   |
|-------------------------------------------------------------|---------------------|---------------------|------------|
| Sublinear randomized value iteration Sidford et al. (2018b) | [ |S||A| H 3 , ∞ )  | |S||A| H 5 ε 2      | ( 0 ,H ]   |
| Variance-reduced QVI Sidford et al. (2018a)                 | [ |S||A| H 4 , ∞ )  | |S||A| H 4 ε 2      | (0 , 1]    |
| Empirical MDP + planning Yin et al. (2021)                  | [ |S||A| H 3 , ∞ )  | |S||A| H 4 ε 2      | (0 , √ H ] |
| Empirical MDP + planning This paper                         | |S||A| H 2 , ∞ )    | |S||A| H 4 ε 2      | (0 ,H ]    |

[

Table 2: Comparisons with prior results (up to log factors) regarding finding an ε -optimal policy in a timeinhomogeneous finite-horizon MDP with a generative model. The sample size range and the ε -range stand for the range of sample size and optimality gap (e.g. ε -accuracy) for the claimed sample complexity to hold. The results in Sidford et al. (2018a,b) were originally stated for the time-homogeneous case; we translate them into the time-inhomogeneous case with an additional factor of H . In addition, Li et al. (2021a) proved that Q-learning finds an ε -optimal Q-function estimate with sample complexity |S||A| H 4 ε 2 ( ε ∈ (0 , 1] ) in a sample size range [ |S||A| H 4 , ∞ ) , which did not translate directly to an ε -optimal policy.

## 2 Problem formulation

The current paper studies both discounted infinite-horizon MDPs and finite-horizon MDPs, which will be introduced separately in the sequel. Here and throughout, we adopt the standard notation [ H ] := { 1 , · · · , H } .

## 2.1 Discounted infinite-horizon Markov decision processes

Models and background. Consider a discounted infinite-horizon MDP represented by a quintuple M = ( S , A , P, r, γ ) , where S := { 1 , 2 , . . . , |S|} denotes a finite set of states, A := { 1 , 2 , . . . , |A|} is a finite set of actions, γ ∈ (0 , 1) stands for the discount factor, and r : S×A → [0 , 1] represents the reward function, namely, r ( s, a ) is the immediate reward received upon executing action a while in state s (here and throughout, we consider the normalized setting where the rewards lie within [0 , 1] ). In addition, P : S×A→ ∆( S ) represents the probability transition kernel of the MDP, where P ( s ′ | s, a ) denotes the probability of transiting from state s to state s ′ when action a is executed, and ∆( S ) denotes the probability simplex over S .

<!-- formula-not-decoded -->

A deterministic policy (or action selection rule) is a mapping π : S → A that maps a state to an action. The value function V π : S → R of a policy π is defined by which is the expected discounted total reward starting from the initial state s 0 = s ; here, the sample trajectory { ( s t , a t ) } t ≥ 0 is generated based on the transition kernel (namely, s t +1 ∼ P ( · | s t , a t ) ), with the actions taken according to policy π (namely, a t = π ( s t ) for all t ≥ 0 ). It is easily seen that 0 ≤ V π ( s ) ≤ 1 1 -γ . The corresponding action-value function (or Q-function) Q π : S × A → R of a policy π is defined by

where the actions are taken according to the policy π after the initial action (i.e. a t = π ( s t ) for all t ≥ 1 ). It is well-known that there exists an optimal policy, denoted by π /star , that simultaneously maximizes V π ( s ) (resp. Q π ( s, a ) ) for all states s ∈ S (resp. state-action pairs ( s, a ) ∈ ( S ×A ) ) (Sutton and Barto, 2018). The corresponding value function V /star := V π /star (resp. action-value function Q /star := Q π /star ) is called the optimal value function (resp. optimal action-value function).

<!-- formula-not-decoded -->

A generative model and an empirical MDP. The current paper focuses on a stylized generative model (also called a simulator) as studied in Kakade (2003); Kearns et al. (2002). Assuming access to this generative model, we collect N independent samples

<!-- formula-not-decoded -->

for each state-action pair ( s, a ) ∈ S × A , which allows us to construct an empirical transition kernel ̂ P as follows where 1 {·} is the indicator function. In words, ̂ P ( s ′ | s, a ) counts the empirical frequency of transitions from ( s, a ) to state s ′ . The total sample size should therefore be understood as N total := N |S||A| . This leads to an empirical MDP ̂ M = ( S , A , ̂ P,r, γ ) constructed from the data samples. We can define the value function and the action-value function of a policy π for ̂ M analogously, which we shall denote by ̂ V π and ̂ Q π , respectively. The optimal policy of ̂ M is denoted by ̂ π /star , with the optimal value function and Q-function denoted by ̂ V /star := ̂ V ̂ π /star and ̂ Q /star := ̂ Q ̂ π /star , respectively.

<!-- formula-not-decoded -->

Learning the optimal policy via model-based planning. Given a few data samples in hand, the task of policy learning seeks to identify a policy that (approximately) maximizes the expected discounted reward given the data samples. Specifically, for any target level ε &gt; 0 , the aim is to compute an ε -accurate policy π est obeying

<!-- formula-not-decoded -->

Naturally, one would hope to accomplish these tasks with as few samples as possible. Recall that for the normalized reward setting with 0 ≤ r ≤ 1 , the value function and Q-function fall within the range [0 , 1 1 -γ ] ; this means that the range of the target accuracy level ε should be set to ε ∈ [0 , 1 1 -γ ] . The model-based approach typically starts by constructing an empirical MDP ̂ M based on all collected samples, and then 'plugs in' this empirical model directly into the Bellman recursion to perform policy evaluation or planning, with prominent examples including Q-value iteration (QVI) and policy iteration (PI) (Bertsekas, 2017).

Aside: policy evaluation. A related task is policy evaluation, which aims to compute or approximate the value function V π under a given policy π . To be precise, for any target level ε &gt; 0 , the goal is to find an ε -accurate estimate V π est such that

## 2.2 Finite-horizon Markov decision processes

<!-- formula-not-decoded -->

Models and background. Another type of models considered in this paper is a finite-horizon MDP, which can be represented and denoted by M = ( S , A , { P h } H h =1 , { r h } H h =1 , H ) . Here, S and A denote respectively the state space and the action space as before, and H represents the horizon length of the MDP. For any 1 ≤ h ≤ H , we let P h : S × A → ∆( S ) denote the probability transition kernel at step h , that is, P h ( s ′ | s, a ) is the probability of transiting to s ′ from ( s, a ) at step h ; r h : S ×A → [0 , 1] indicates the reward function at step h , namely, r h ( s, a ) is the immediate reward gained at step h in response to ( s, a ) . As before, we assume normalized rewards throughout the paper, so that all the r h ( s, a ) 's reside within the interval [0 , 1] .

Let π = { π h } 1 ≤ h ≤ H represent a deterministic policy, such that for any 1 ≤ h ≤ H and any s ∈ S , π h ( s ) specifies the action selected at step h in state s . Note that π could be non-stationary, meaning that the π h 's might be different across different time steps h . The value function and the Q-function associated with policy π are defined respectively by for all s ∈ S and all 1 ≤ h ≤ H , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all ( s, a ) ∈ S × A and all 1 ≤ h ≤ H . As usual, the expectations are taken over the randomness of the MDP trajectory { ( s k , a k ) } 1 ≤ k ≤ H induced by the transition kernel { P h } H h =1 when policy π is adopted. With slight abuse of notation, we let Q π H +1 ( s, a ) = 0 for every ( s, a ) ∈ S × A and V π H +1 ( s ) = 0 for every s ∈ S . In view of the assumed range of the immediate rewards, it is easily seen that

<!-- formula-not-decoded -->

for any π , any state-action pair ( s, a ) , and any step h . Akin to the infinite-horizon counterpart, the optimal value functions { V /star h } 1 ≤ h ≤ H and optimal Q-functions { Q /star h } 1 ≤ h ≤ H are defined respectively by

<!-- formula-not-decoded -->

for any state-action pair ( s, a ) ∈ S × A and any 1 ≤ h ≤ H . It is well known that there exists at least one policy that allows one to simultaneously achieve the optimal value function and optimal Q-functions for all state-action pairs and all time steps. Throughout this paper, we shall denote by π /star = { π /star h } 1 ≤ h ≤ H an optimal policy.

A generative model and an empirical MDP. Similar to the infinite-horizon setting, we assume access to a generative model, which is able to generate N independent samples for each triple ( s, a, h ) ∈ S ×A× [ H ] as follows

The empirical transition kernel { P h } H h =1 is thus given by

<!-- formula-not-decoded -->

which records the empirical frequency of transitions from ( s, a ) to state s ′ at step h . This gives rise to a total sample size N total := NH |S||A| . We shall let ̂ M = ( S , A , { ̂ P h } H h =1 , { r h } H h =1 , H ) represent the empirical MDP constructed from the data samples. The value function and the Q-function of a policy π for ̂ M can be defined analogously, which shall be denoted by { ̂ V π h } H h =1 and { ̂ Q π h } H h =1 , respectively. We denote by ̂ π /star the optimal policy of ̂ M , and the resulting optimal value function and Q-function are denoted by ̂ V /star h := ̂ V ̂ π /star h and ̂ Q /star h := ̂ Q ̂ π /star h , respectively.

<!-- formula-not-decoded -->

Learning the optimal policy via model-based planning. Given the data samples in hand, the task of policy learning in the finite-horizon case can be defined similarly as the infinite-horizon counterpart. Specifically, for any target level ε &gt; 0 , the aim of policy learning is to compute an ε -accurate policy π est obeying

<!-- formula-not-decoded -->

With the normalized range of the reward function, it is easily seen that the value function and the Q-function reside within the interval [0 , H ] , thus implying that the range of the target accuracy level should be ε ∈ [0 , H ] .

## 2.3 Notation

Let X := ( |S| , |A| , 1 1 -γ , 1 ε ) . The notation f ( X ) = O ( g ( X )) means there exists a universal constant C 1 &gt; 0 such that f ≤ C 1 g , whereas the notation f ( X ) = Ω( g ( X )) means g ( X ) = O ( f ( X )) . In addition, the notation O ( · ) (resp. Ω( · ) ) is defined in the same way as O ( · ) (resp. Ω( · ) ) except that it ignores logarithmic factors.

˜ ˜ For any vector a = [ a i ] 1 ≤ i ≤ n ∈ R n , we overload the notation √ · and | · | in an entry-wise manner such that √ a := [ √ a i ] 1 ≤ i ≤ n and | a | := [ | a i | ] 1 ≤ i ≤ n . For any vectors a = [ a i ] 1 ≤ i ≤ n and b = [ b i ] 1 ≤ i ≤ n , the notation a ≥ b (resp. a ≤ b ) means a i ≥ b i (resp. a i ≤ b i ) for all 1 ≤ i ≤ n , and we let a ◦ b := [ a i b i ] 1 ≤ i ≤ n represent the Hadamard product. Additionally, we denote by 1 the all-one vector, and I the identity matrix. For any matrix A , we define the norm ‖ A ‖ 1 := max i ∑ j | A i,j | .

## 3 Model-based planning in discounted infinite-horizon MDPs

As summarized in Table 1, the theory of all prior works required the sample size per state-action pair to at least exceed N ≥ Ω ( 1 (1 -γ ) 2 ) . In order to break this sample size barrier, we develop two model-based algorithms that provably overcome such a sample size barrier.

## 3.1 Model-based reinforcement learning: two algorithms

Algorithm 1: perturbed model-based planning. The first algorithm applies model-based planning to an empirical MDP with randomly perturbed rewards . Specifically, for each state-action pair ( s, a ) ∈ S × A , we randomly perturb the immediate reward by

<!-- formula-not-decoded -->

where Unif (0 , ξ ) denotes the uniform distribution between 0 and some parameter ξ &gt; 0 (to be specified momentarily). 2 For any policy π , we denote by ̂ V π p the corresponding value function of the perturbed 2 Note that perturbation is only invoked when running the planning algorithms and does not require collecting new samples.

empirical MDP ̂ M p = ( S , A , ̂ P,r p , γ ) with the probability transition kernel ̂ P (cf. (3)) and the perturbed reward function r p . Let ̂ π /star p represent the optimal policy of M p , i.e.

Algorithm 2: conservative model-based planning. An alternative approach that eliminates the need of reward perturbation is to select approximately optimal actions for the empirical MDP instead of the absolute optimal actions. To be precise, denote by ̂ Q /star (resp. ̂ V /star ) the optimal action-value (resp. value) function of the empirical MDP ̂ M = ( S , A , ̂ P,r, γ ) with the probability transition kernel ̂ P (cf. (3)) and the original reward function r . By producing a random draw from ς ∼ Unif (0 , ξ ) (with ξ specified shortly), we can generate the following policy π c :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ ̂ ̂ Note that there is an index assigned to each action as A = { 1 , · · · , |A|} which induces a natural order for all actions. In words, this approach is more conservative and does not stick to the optimal actions w.r.t. the empirical MDP; instead, the policy ̂ π c picks out - for each state s ∈ S -the smallest indexed action that is within a gap of ς from optimal.

## 3.2 Theoretical guarantees

Indeed, both the above-mentioned approaches result in a value function (resp. Q-function) that well approximates the true optimal value function V /star (resp. optimal Q-function Q /star ). We start by presenting our results for the perturbed model-based approach.

Theorem 1 (Perturbed model-based planning) . There exist some universal constants c 0 , c 1 &gt; 0 such that: for any δ &gt; 0 and any 0 &lt; ε ≤ 1 1 -γ , the policy π /star p defined in (9) obeys

<!-- formula-not-decoded -->

with probability at least 1 -δ , provided that the perturbation size is ξ = c 1 (1 -γ ) ε |S| 5 |A| 5 and that the sample size per state-action pair exceeds

<!-- formula-not-decoded -->

In addition, both the empirical QVI and PI algorithms w.r.t. ̂ M p (cf. (Azar et al., 2013, Algorithms 1-2)) are able to recover π /star p perfectly within O ( 1 1 -γ log( |S||A| (1 -γ ) εδ ) ) iterations.

̂ Remark 1 . Theorem 1 holds unchanged if ξ is taken to be c 1 (1 -γ ) ε |S| α |A| α for any fixed constant α ≥ 1 . This paper picks the specific choice α = 5 merely to convey that a very small degree of perturbation suffices for our purpose.

Remark 2 . Perturbation brings a side benefit: one can recover the optimal policy ̂ π /star p of the perturbed empirical MDP ̂ M p exactly in a small number of iterations without incurring further optimization errors. To give a flavor of the overall computational complexity, let us take QVI for example Azar et al. (2013). Recall that each iteration of QVI takes time proportional to the time taken to read ̂ P (which is a matrix with at most N |S||A| nonzeros), hence the resulting computational complexity can be as low as O ( |S||A| (1 -γ ) 4 ε 2 log 2 ( |S||A| (1 -γ ) εδ ) ) .

Further, similar performance guarantees can be established for the conservative model-based approach without reward perturbation, as stated below.

Theorem 2 (Conservative model-based planning) . Under the same assumptions of Theorem 1 (including both the sample size and the choice of ξ ), the policy π c defined in (10) achieves

<!-- formula-not-decoded -->

with probability at least 1 -δ .

In a nutshell, the above theorems demonstrate that: both model-based algorithms we introduce succeed in finding an ε -optimal policy as soon as the total sample complexity exceeds the order of |S||A| (1 -γ ) 3 ε 2 (modulo some log factor). It is worth emphasizing that, compared to prior literature, our result imposes no restriction on the range of ε and, in particular, we allow the accuracy level ε to go all the way up to 1 1 -γ . Our result is particularly useful in the regime with small-to-moderate sample sizes, since its validity is guaranteed as long as

<!-- formula-not-decoded -->

Tackling the sample-limited regime (in particular, the scenario when N ∈ [ 1 1 -γ , 1 (1 -γ ) 2 ] ) requires us to develop new analysis frameworks beyond prior theory, which we shall discuss in detail momentarily.

We remark that the work Azar et al. (2013) established a minimax lower bound of the same order as (12) (up to some log factor) in the regime ε = O (1) . A closer inspection of their analysis, however, reveals that their argument and bound hold true as long as ε = O ( 1 1 -γ ) . This in turn corroborates the minimax optimality of our perturbed model-based approach for the full ε -range (which is previously unavailable), and demonstrates the information-theoretic infeasibility to learn a policy strictly better than a random guess if N ≤ O ( 1 1 γ ) . Put another way, the condition (14) contains the full range of 'meaningful' sample sizes.

˜ -Finally, we single out an intermediate result in the analysis of our theorems concerning model-based policy evaluation, which might be of interest on its own. Specifically, for any fixed policy π independent of the data, this task concerns value function estimation via the plug-in estimate ̂ V π (i.e. the value function of the empirical M under this policy). However simple as this might seem, existing theoretical underpinnings of this approach remain suboptimal, unless the sample size is already sufficiently large. Our result is the following, which does not require enforcing reward perturbation.

Theorem 3 (Model-based policy evaluation) . Fix any policy π . There exists some universal constant c 0 &gt; 0 such that: for any 0 &lt; δ &lt; 1 and any 0 &lt; ε ≤ 1 1 -γ , one has with probability at least 1 -δ , provided that the sample size per state-action pair exceeds

<!-- formula-not-decoded -->

In words, this theorem reveals that ̂ V π begins to outperform a random guess as soon as N ≥ ˜ Ω ( 1 1 -γ ) . The sample complexity bound (16) enjoys full coverage of the ε -range (0 , 1 1 -γ ] , and matches the minimax lower bound derived in (Pananjady and Wainwright, 2019, Theorem 2(b)) up to only a log log 1 1 -γ factor. In addition, a recent line of work investigated instance-dependent guarantees for policy evaluation (Khamaru et al. (2020); Pananjady and Wainwright (2019)). While this is not our focus, our analysis does uncover an instance-dependent bound with a broadened sample size range. See Lemma 1 and the discussion thereafter.

<!-- formula-not-decoded -->

## 3.3 Comparisons with prior works and implications

In order to discuss the novelty of our results in context, we take a moment to compare them with prior theory. See Table 1 for a more complete list of comparisons.

Prior bounds for planning and policy learning. None of the prior results with a generative model (including both model-based or model-free approaches) was capable of efficiently finding the desired policy while accommodating the full sample size range (14). For instance, the state-of-the-art analysis for the model-based approach Agarwal et al. (2020) required the sample size to at least exceed

<!-- formula-not-decoded -->

whereas the theory for the variance-reduced model-free approach Sidford et al. (2018a); Wainwright (2019b) imposed the sample size requirement

<!-- formula-not-decoded -->

In fact, it was previously unknown what is achievable in the sample size range N ∈ [ 1 1 -γ , 1 (1 -γ ) 2 ] . In contrast, our results confirm the minimax-optimal statistical performance of the model-based approach with full coverage of the ε -range and the sample size range.

Remark 3 . We briefly point out why the sample size barrier (17) appeared in the analysis of Agarwal et al. (2020). Take Agarwal et al. (2020) Section 4.3 for example: the contraction factor γ √ 8 log( |S||A| / (1 -γ ) δ ) N 1 1 -γ therein needs to be smaller than 1, thereby requiring N ≥ ˜ Ω ( (1 -γ ) -2 ) .

Prior bounds for policy evaluation. Regarding value function estimation for any fixed policy π , the prior results Agarwal et al. (2020); Azar et al. (2013); Pananjady and Wainwright (2019) for the plug-in approach all operated under the assumption that N ≥ ˜ Ω ( 1 (1 -γ ) 2 ) , which is more stringent than our result by a factor of at least 1 1 -γ . In addition, our sample complexity matches the state-of-the-art guarantees in the regime where ε ≤ 1 √ 1 -γ (Agarwal et al., 2020; Pananjady and Wainwright, 2019), while extending them to the range ε ∈ [ 1 √ 1 -γ , 1 1 -γ ] uncovered in these previous papers.

## 4 Model-based planning in finite-horizon MDPs

Moving beyond discounted infinite-horizon MDPs, our theoretical framework is also able to accommodate finite-horizon MDPs, which we detail in this section.

## 4.1 Algorithm: model-based planning

The algorithm considered in this section is model-based planning (without reward perturbation). Specifically, this model-based approach returns a policy π /star = { π /star h } 1 ≤ h ≤ H by means of the following two steps:

- ̂ 2) Run a classical dynamic programming algorithm (Bertsekas, 2017) to find an optimal policy ̂ π /star of the empirical MDP M .
- ̂ ̂ 1) Construct the empirical MDP ̂ M = ( S , A , { ̂ P h } 1 ≤ h ≤ H , { r h } 1 ≤ h ≤ H , H ) based on the data samples in hand (see (6) for the computation of the empirical transition kernel P h );

̂ Note that ̂ π /star h is an optimal policy of ̂ M at step h , computed by the dynamic programming algorithm calculated backward from h = H . Since ̂ π /star h is calculated solely based on what happens after step h , ̂ π /star h is independent of the empirical transitions { P j } 1 ≤ j&lt;h .

̂ It is noteworthy that, in contrast to the infinite-horizon counterpart in Section 3, we do not need to enforce random reward perturbation for this finite-horizon case.

## 4.2 Theoretical guarantees and implications

The model-based algorithm described above turns out to be nearly minimax optimal, as asserted by the following theorem.

Theorem 4 (Model-based planning) . There exist some universal constants c 0 , c 1 &gt; 0 such that: for any δ &gt; 0 and any 0 &lt; ε ≤ H , the aforementioned policy π /star returned by model-based planning obeys

<!-- formula-not-decoded -->

with probability at least 1 -δ , provided that the sample size for every triple ( s, a, h ) exceeds

<!-- formula-not-decoded -->

Akin to the discounted infinite-horizon scenario, the model-based approach manages to achieve ε -accuracy as long as the sample size per ( s, a, h ) exceeds the order of

<!-- formula-not-decoded -->

This result, which is valid for the full ε range (0 , H ] , is reminiscent of the bound (12), except that the effective horizon 1 1 -γ needs to be replaced by the horizon length H . Given that there are in total |S||A| H different combinations of ( s, a, h ) , the total sample complexity is on the order of O ( |S||A| H 4 ε 2 ) .

The quadruple scaling H 4 of this total sample complexity - as opposed to the cubic scaling in the discounted infinite-horizon case - is due to time inhomogeneity; that is, the P h 's might be different across h , resulting in an additional H factor. Again, our result kicks in as soon as the sample size satisfies improving upon the sample size requirement

˜

<!-- formula-not-decoded -->

in the state-of-the-art analysis for the model-based approach Yin et al. (2021).

## 5 Other related works

Classical analyses of reinforcement learning algorithms have largely focused on asymptotic performance (e.g. Jaakkola et al. (1994); Szepesvári (1998); Tsitsiklis and Van Roy (1997); Tsitsiklis (1994)). Leveraging the toolkit of concentration inequalities, a number of recent papers have shifted attention towards understanding the performance in the non-asymptotic and finite-time settings. A highly incomplete list includes Azar et al. (2017); Beck and Srikant (2012); Bhandari et al. (2018); Bradtke and Barto (1996); Cai et al. (2019); Chen et al. (2020); Dalal et al. (2018); Even-Dar and Mansour (2003); Fan et al. (2019); Gupta et al. (2019); Jin et al. (2018); Kaledin et al. (2020); Kearns and Singh (1999); Khamaru et al. (2020); Lakshminarayanan and Szepesvari (2018); Li et al. (2023, 2021c, 2022c); Mou et al. (2020); Qu and Wierman (2020); Shah and Xie (2018); Shi et al. (2022); Sidford et al. (2018a); Srikant and Ying (2019); Strehl et al. (2006); Wainwright (2019b); Xu and Gu (2020); Xu et al. (2019); Yan et al. (2022a), a large fraction of which is concerned with model-free algorithms.

The generative model (or simulator) adopted in this paper was first proposed in Kearns and Singh (1999), which has been invoked in Agarwal et al. (2020); Azar et al. (2012, 2013); Kakade (2003); Kearns et al. (2002); Kearns and Singh (1999); Khamaru et al. (2020); Lattimore and Hutter (2012); Li et al. (2022a); Pananjady and Wainwright (2019); Sidford et al. (2018a,b); Wainwright (2019b); Wang et al. (2021); Wang (2019); Yang and Wang (2019), to name just a few. In particular, Azar et al. (2013) developed the minimax lower bound on the sample complexity N = Ω ( |S||A| log( |S||A| ) (1 -γ ) 3 ε 2 ) necessary for finding an ε -optimal policy, and showed that, for any ε ∈ (0 , 1) , a model-based approach (e.g. applying QVI or PI to the empirical MDP) can estimate the optimal Q-function to within an ε -accuracy given near-minimal samples. Note, however, that directly translating this result to the policy guarantees leads to an additional factor of 1 1 -γ in estimation accuracy and of 1 (1 -γ ) 2 in sample complexity. In light of this, Azar et al. (2013) further showed that a nearoptimal sample complexity is possible for policy learning if the sample size is at least on the order of |S| 2 |A| (1 -γ ) 2 which, however, is no longer sub-linear in the model complexity. A recent breakthrough Agarwal et al. (2020) substantially improved the model-based guarantee with the aid of auxiliary state-absorbing MDPs, extending the range of sample complexity to [ |S||A| log( |S||A| ) (1 -γ ) 2 , ∞ ) . Our analysis is motivated in part by Agarwal et al. (2020), but also relies on several other novel techniques to complete the picture.

Finally, we remark that the construction of state-absorbing MDPs or state-action-absorbing MDPs falls under the category of 'leave-one-out' type analysis, which is particularly effective in decoupling complicated statistical dependency in various statistical estimation problems Agarwal et al. (2020); Chen et al. (2021, 2019a,b); El Karoui (2015); Ma et al. (2020); Pananjady and Wainwright (2019); Yan et al. (2021). The

<!-- formula-not-decoded -->

application of such an analysis framework to MDPs should be attributed to Agarwal et al. (2020). Other applications to Markov chains include Chen et al. (2019a); Pananjady and Wainwright (2019). More recently, several follow-up works have further generalized the leave-one-out analysis idea to accommodate broader RL settings including offline RL (Li et al., 2022b), RL with linear function approximation (Wang et al., 2021), and Markov games (Cui and Yang, 2021; Yan et al., 2022b), and so on.

## 6 Analysis: infinite-horizon MDPs

This section presents the key ideas for proving our main results, following an introduction of some convenient matrix notation.

## 6.1 Matrix notation and Bellman equations

It is convenient to present our proof based on some matrix notation for MDPs. Denoting by e 1 , · · · , e |S| ∈ R |S| the standard basis vectors, we can define:

- r ∈ R |S||A| : a vector representing the reward function r (so that r ( s,a ) = r ( s, a ) for all ( s, a ) ∈ S × A ).
- V π ∈ R |S| : a vector representing the value function V π (so that V π s = V π ( s ) for all s ∈ S ).
- Q π ∈ R |S||A| : a vector representing the Q-function Q π (so that Q π ( s,a ) = Q π ( s, a ) for all ( s, a ) ∈ S ×A ).
- V /star ∈ R |S| and Q /star ∈ R |S||A| : representing the optimal value function V /star and optimal Q-function Q /star .
- P ∈ R |S||A|×|S| : a matrix representing the probability transition kernel P , where the ( s, a ) -th row of P is a probability vector representing P ( ·| s, a ) . Denote P s,a as the ( s, a ) -th row of the transition matrix P .
- Π π ∈ { 0 , 1 } |S|×|S||A| : a projection matrix associated with a given policy π taking the following form

<!-- formula-not-decoded -->

- P π ∈ R |S||A|×|S||A| and P π ∈ R |S|×|S| : two square probability transition matrices induced by the policy π over the state-action pairs and the states respectively, defined by

<!-- formula-not-decoded -->

- r π ∈ R |S| : a reward vector restricted to the actions chosen by the policy π , namely, r π ( s ) = r ( s, π ( s )) for all s ∈ S (or simply, r π = Π π r ).

Armed with the above matrix notation, we can write, for any policy π , the Bellman consistency equation as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For a vector V = [ V i ] 1 ≤ i ≤|S| ∈ R |S| , we define the vector Var P ( V ) ∈ R |S||A| whose entries are given by

<!-- formula-not-decoded -->

which implies that

i.e. the variance of V w.r.t. P ( ·| s, a ) . This can be expressed using our matrix notation as follows

<!-- formula-not-decoded -->

Similarly, for any given policy π we define

<!-- formula-not-decoded -->

We shall also define ̂ V π , ̂ Q π , ̂ V /star , ̂ Q /star , ̂ P , ̂ P π , ̂ P π , Var ̂ P ( V ) , Var ̂ P π ( V ) w.r.t. the empirical MDP ̂ M in an analogous fashion.

## 6.2 Analysis: model-based policy evaluation

We start with the simpler task of policy evaluation, which also plays a crucial role in the analysis of planning. To establish our guarantees in Theorem 3, we aim to prove the following result. Here, we recall that the true value function under a policy π and the model-based empirical estimate are given respectively by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 1. Fix any policy π . Consider any 0 &lt; δ &lt; 1 , and suppose N ≥ 32 e 2 1 -γ log ( 4 |S| log( e 1 -γ ) δ ) . Then with probability at least 1 -δ , the vectors defined in (30) obey

Proof. The key proof idea is to resort to a high-order successive expansion of ̂ V π -V π , followed by finegrained analysis of each term up to a certain logarithmic order. See Appendix B.1.

Proof ideas. We now briefly and informally describe the key proof ideas. As a starting point, the elementary identities (30) allow us to obtain

Clearly, Theorem 3 is a straightforward consequence of Lemma 1. Further, we strengthen the result by providing an additional instance-dependent bound (see the first line of (31) that depends on the true instance P π , V π ), which is often tighter than the worst-case bound stated in the second line of (31). Our contribution can be better understood when compared with Pananjady and Wainwright (2019). Assuming that there is no noise in the rewards, our instance-dependent guarantee matches Pananjady and Wainwright (2019, Theorem 1(a)) up to some log log 1 1 -γ factor, while being capable of covering the full sample size range N ≥ ˜ Ω( 1 1 -γ ) . In contrast, Pananjady and Wainwright (2019, Theorem 1) is only valid when N ≥ ˜ Ω( 1 (1 -γ ) 2 ).

<!-- formula-not-decoded -->

Due to the complicated dependency between ( I -γ ̂ P π ) -1 and ( ̂ P π -P π ) V π , a natural strategy is to control these two terms separately and then to combine bounds; see Agarwal et al. (2020, Lemma 5) for an introduction. This simple approach, however, leads to sub-optimal statistical guarantees.

In order to refine the statistical analysis, we propose to further expand (32) in a similar way to deduce

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line holds due to the same reason as (32) (basically it can be seen by replacing r π with ( ̂ P π -P π ) V π in (32)). This can be viewed as a 'second-order' expansion, with (32) being a 'first-order' counterpart. The advantage is that: the first term in (33) becomes easier to cope with than its counterpart (32), owing to the independence between ( I -γ P π ) -1 and ( ̂ P π -P π ) V π . However, the second term in (33) remains difficult to control optimally. To remedy this issue, we shall continue to expand it to higher order (up to some logarithmic order), which eventually allows for optimal control of the estimation error.

Another crucial issue is that: in order to obtain fine-grained analyses on each term in the expansion (except for the first-order term), a common approach is to combine the Bernstein inequality with a classical entrywise bound on a quantity taking the form ( I -γ P π ) -1 √ Var P π ( V ) (which dates back to Azar et al. (2013)). Such a classical bound in prior literature, however, is not sufficiently tight for our purpose, which calls for refinement; see Lemma 11. Details are deferred to Appendix B.1.

## 6.3 Analysis: perturbed model-based planning

This subsection moves on to establishing our theory for model-based planning (cf. Theorem 1) and outlines the key ideas. In what follows, we shall start by analyzing the unperturbed version, which will elucidate the role of reward perturbation in our analysis.

We first make note of the following elementary decomposition:

<!-- formula-not-decoded -->

Step 1: bounding ‖ V π /star -̂ V π /star ‖ ∞ . Given that π /star is independent of the data, we can carry out this step using Lemma 1. Specifically, taking π = π /star in Lemma 1 yields that, with probability at least 1 -δ ,

̂ ̂ where the inequality follows from the optimality of ̂ π /star w.r.t. ̂ V (so that ̂ V π /star ≤ ̂ V ̂ π /star ) and the definition V /star = V π /star . This leaves us with two terms to control.

<!-- formula-not-decoded -->

Step 2: bounding ‖ ̂ V ̂ π /star -V ̂ π /star ‖ ∞ . Extending the result in Step 1 to ‖ ̂ V ̂ π /star -V ̂ π /star ‖ ∞ is considerably more challenging, primarily due to the complicated statistical dependency between ( V ̂ π /star , ̂ V ̂ π /star ) and the data matrix ̂ P . The recent work Agarwal et al. (2020) developed a clever 'leave-one-out' type argument by constructing some auxiliary state-absorbing MDPs to decouple the statistical dependency when ε &lt; 1 / √ 1 -γ . However, their argument falls short of accommodating the full range of ε . To address this challenge, our analysis consists of the following two steps, both of which require new ideas beyond Agarwal et al. (2020).

- Decoupling statistical dependency between ̂ π /star and ̂ P . Instead of attempting to decouple the statistical dependency between ̂ V ̂ π /star and ̂ P as in Agarwal et al. (2020), we focus on decoupling the statistical dependency between the policy ̂ π /star and ̂ P . If this can be achieved, then the proof strategy adopted in Step 1 for a fixed policy becomes applicable (see Section 6.3.1). A key ingredient of this step lies in the construction of a collection of auxiliary state-action-absorbing MDPs (motivated by Agarwal et al. (2020)), which allows us to get hold of ‖ V ̂ π /star -̂ V ̂ π /star ‖ ∞ . See Section 6.3.2 for details, with a formal bound delivered in Lemma 5.
- Tie-breaking via reward perturbation. A shortcoming of the above-mentioned approach, however, is that it relies crucially on the separability of ̂ π /star from other policies; in other words, the proof might fail if ̂ π /star is non-unique or not sufficiently distinguishable from others. Consequently, it remains to ensure that the optimal policy ̂ π /star stands out from all the rest for all MDPs of interest. As it turns out, this can be guaranteed with high probability by slightly perturbing the reward function so as to break the ties. See Section 6.3.3 for details.

In the sequel, we shall flesh out these key ideas.

## 6.3.1 Value function estimation for a policy obeying Bernstein-type conditions

Before discussing how to decouple statistical dependency, we record a useful result that plays an important role in the analysis. Specifically, Lemma 1 can be generalized beyond the family of fixed policies (namely, those independent of ̂ P ), as long as a certain Bernstein-type condition - to be formalized in (37) - is satisfied. To make it precise, we need to introduce a set of auxiliary vectors as follows

Our generalization of Lemma 1 is as follows, which does not require statistical independence between the policy π and the data ̂ P . Here, we remind the reader of the notation | z | := [ | z 1 | , · · · , | z n | ] /latticetop and √ z := [ √ z 1 , · · · , √ z n ] /latticetop for any vector z ∈ R n .

<!-- formula-not-decoded -->

Lemma 2. Suppose that there exists some quantity β 1 &gt; 0 such that { V ( l ) } (cf. (36) ) obeys

Suppose that N &gt; 16 e 2 1 -γ β 1 . Then the vectors V π = ( I -γ P π ) -1 r π and ̂ V π = ( I -γ ̂ P π ) -1 r π satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## ̂ 6.3.2 Decoupling statistical dependency via ( s, a ) -absorbing MDPs

While the Bernstein-type condition (37) clearly holds for some reasonably small β 1 if π is independent of ̂ P , it might remain valid if π exhibits fairly 'weak' statistical dependency on the data samples. This is a key step that paves the way for our subsequent analysis of π /star .

We are now positioned to demonstrate how to control ∥ ∥ ̂ V ̂ π /star -V ̂ π /star ∥ ∥ ∞ w.r.t. the optimal policy ̂ π /star to ̂ V . A crucial technical challenge lies in how to decouple the complicated statistical dependency between the optimal policy ̂ π /star and the ̂ V /star (which heavily relies on the data samples). Towards this, we resort to a leave-onerow-out argument built upon a collection of auxiliary MDPs, largely motivated by the novel construction in (Agarwal et al., 2020, Section 4.2). In comparison to Agarwal et al. (2020) that introduces state-absorbing MDPs (so that a state s is absorbing regardless of the subsequent actions chosen), our construction is a set of state-action-absorbing MDPs, in which a state s is absorbing only when a designated action a is always executed at the state s .

Construction of ( s, a ) -absorbing MDPs. For each state-action pair ( s, a ) and each scalar u with | u | ≤ 1 / (1 -γ ) , we construct an auxiliary MDP M s,a,u - it is identical to the original M except that it is absorbing in state s if we always choose action a in state s . More specifically, the probability transition kernel associated with M s,a,u (denoted by P M s,a,u ) can be specified by

/negationslash

<!-- formula-not-decoded -->

/negationslash where P M is the probability transition kernel w.r.t. the original M . Meanwhile, the instant reward received at ( s, a ) in M s,a,u is set to be u , while the rewards at all other state-action pairs stay unchanged. We can define ̂ M s,a,u analogously (so that its probability transition matrix is identical to ̂ P except that the ( s, a ) -th row becomes absorbing). The main advantage of this construction is that: for any fixed u , the MDP ̂ M s,a,u

is statistically independent of ̂ P s,a (the row of ̂ P corresponding to the state-action pair ( s, a ) , determined by the samples collected for the ( s, a ) pair).

̂ ̂ ̂ ̂ ̂ Remark 4 . The careful reader will remark that the instant reward u is constrained to reside within [ -1 1 -γ , 1 1 -γ ] rather than the usual range [0 , 1] . Fortunately, none of the subsequent steps that involve u requires u to lie within [0 , 1] .

To streamline notation, we let Q π s,a,u represent the Q-function of M s,a,u under a policy π , denote by π /star s,a,u the optimal policy associated with M s,a,u , and let Q /star s,a,u be the Q-function under this optimal policy π /star s,a,u . The notation V π s,a,u and V /star s,a,u regarding value functions, as well as their counterparts (i.e. ̂ Q π s,a,u , Q /star s,a,u , V π s,a,u , V /star s,a,u , π /star s,a,u ) in the empirical MDP M , can be defined in an analogous fashion.

Intimate connections between the auxiliary MDPs and the original MDP. In the following, we introduce a result that connects the Q-function and the value function of the absorbing MDP with those of the original MDP. The idea is motivated by Agarwal et al. (2020, Lemma 7) and its proof is deferred to Appendix B.2.

Lemma 3. PV

<!-- formula-not-decoded -->

Remark 5 . Lemma 3 does not rely on the particular form of P , and can be directly generalized to the empirical model P and the auxiliary MDPs built upon P .

̂ ̂ In words, by properly setting the instant reward u = u /star (which can be easily shown to reside within [ -1 1 -γ , 1 1 -γ ] ), one guarantees that the ( s, a ) -absorbing MDP and the original MDP have the same Q-function and value function under the respective optimal policies.

Representing ̂ π /star via a small set of policies independent of ̂ P s,a . With Lemma 3 in place, it is tempting to use ̂ M s,a, ̂ u /star with ̂ u /star := r ( s, a ) + γ ( ̂ P ̂ V /star ) s,a -γ ̂ V /star ( s ) to replace the original ̂ M . The rationale is simple: given that the probability transition matrix of ̂ M s,a, ̂ u /star does not rely upon ̂ P s,a , the statistical dependency between ̂ M s,a, ̂ u /star and ̂ P s,a is now fully embedded into a single parameter ̂ u /star . This motivates us to decouple the statistical dependency effectively by constructing an epsilon-net (see, e.g., Vershynin (2018)) w.r.t. this single parameter. The aim is to locate a point u 0 over a small fixed set such that (i) it is close to u /star , and (ii) its associated optimal policy is identical to the original optimal policy π /star .

̂ ̂ It turns out that this aim can be accomplished as long as the original Q-function ̂ Q /star satisfies a sort of separation condition (which indicates that there is no tie when it comes to the optimal policy). To make it precise, given any 0 &lt; ω &lt; 1 , our separation condition is characterized through the following event

/negationslash

Clearly, on the event B ω , the optimal policy ̂ π /star is unique, since for each s the action ̂ π /star ( s ) results in a strictly higher Q-value compared to any other action. With this separation condition in mind, our result is stated below. Here and throughout, we define an epsilon-net of the interval [ -1 1 -γ , 1 1 -γ ] as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which has cardinality at most 2 (1 -γ ) /epsilon1 .

Lemma 4. Consider any ω &gt; 0 , and suppose the event B ω (cf. (41) ) holds. Then for any pair ( s, a ) ∈ S×A , there exists a point u 0 ∈ N (1 -γ ) ω/ 4 , such that

Proof. See Appendix B.3.

<!-- formula-not-decoded -->

Deriving an optimal error bound under the separation condition. Armed with the above bounds, we are ready to derive the desired error bound by combining Lemma 2 and Lemma 4.

Lemma 5. Given 0 &lt; ω &lt; 1 and δ &gt; 0 , suppose that B ω (defined in (41) ) occurs with probability at least 1 -δ . Then with probability at least 1 -3 δ , provided that N ≥ c 0 log ( |S||A| (1 -γ ) δω ) 1 -γ for some sufficiently large constant c 0 &gt; 0 . Proof. See Appendix B.4.

<!-- formula-not-decoded -->

## 6.3.3 A tie-breaking argument

Unfortunately, the separation condition specified in B ω (cf. (41)) does not always hold. In order to accommodate all possible MDPs of interest without imposing such a special separation condition, we put forward a perturbation argument allowing one to generate a new MDP that (i) satisfies the separation condition, and that (ii) is sufficiently close to the original MDP.

Specifically, let us represent the proposed reward perturbation (8) in a vector form as follows

<!-- formula-not-decoded -->

where ζ = [ ζ ( s, a ) ] ( s,a ) ∈S×A is an |S||A| -dimensional vector composed of independent entries with each ζ ( s, a ) i . i . d . ∼ Unif (0 , ξ ) . We aim to show that: by randomly perturbing the reward function, we can 'break the tie' in the Q-function and ensure sufficient separation of Q-values associated with different actions.

To formalize our result, we find it convenient to introduce additional notation. Denote by π /star p the optimal policy of the MDP M p = ( S , A , P , r p , γ ) , and Q /star p its optimal state-action value function. We can define ̂ Q /star p and ̂ π /star p analogously for the MDP ̂ M p = ( S , A , ̂ P , r p , γ ) . Our result is phrased as follows. Lemma 6. Consider the perturbed reward vector defined in expression (45) . With probability at least 1 -δ ,

/negationslash

<!-- formula-not-decoded -->

This result holds unchanged if ( Q /star p , π /star p ) is replaced by ( ̂ Q /star p , ̂ π /star p ) . Proof. See Appendix B.5.

Lemma 6 reveals that at least a polynomially small degree of separation ( ω = ξδ (1 -γ ) 3 |S||A| 2 ) arises upon random perturbation (with size ξ ) of the reward function. As we shall see momentarily, this level of separation suffices for our purpose.

## 6.3.4 Proof of Theorem 1

Let us consider the randomly perturbed reward function as in (45). For any policy π , we denote by V π p (resp. ̂ V π p ) the corresponding value function vector in the MDP with probability transition matrix P (resp. ̂ P ) and reward vector r p . Note that π /star p (resp. ̂ π /star p ) denotes the optimal policy that maximizes V π p (resp. ̂ V π p ). In view of Lemma 6, with probability at least 1 -δ one has the separation

<!-- formula-not-decoded -->

/negationslash uniformly over all s and a = ̂ π /star p ( s ) . With this separation in place, taking ω := ξδ (1 -γ ) 3 |S||A| 2 in Lemma 5 yields

In addition, the value functions under any policy π obeys

<!-- formula-not-decoded -->

which taken collectively with the facts ‖ r -r p ‖ ∞ ≤ ξ and ‖ ( I -γ P π ) -1 ‖ 1 ≤ 1 1 -γ gives

<!-- formula-not-decoded -->

Specializing the above relation to π /star and π /star p gives

<!-- formula-not-decoded -->

Now let us consider the following decomposition

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step follows from the optimality of π /star p w.r.t. V p . Taking this collectively with the inequalities (48) and (49), one shows that with probability greater than 1 -3 δ ,

By taking ξ = (1 -γ ) ε 3 |S| 5 |A| 5 and N ≥ c 0 log ( |S||A| (1 -γ ) δε ) (1 -γ ) 3 ε 2 for some constant c 0 &gt; 0 large enough, we can ensure that 0 ≥ V ̂ π /star p -V /star ≥ -ε 1 as claimed. Regarding the Q-functions, the Bellman equation gives

Consequently, one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, we demonstrate that both the empirical QVI and PI w.r.t. ̂ M p are guaranteed to find ̂ π /star p in a few iterations. Suppose for the moment that we can obtain a policy π k obeying

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

where the last line results from (47) and (50). In other words, we can perfectly recover the policy ̂ π /star p from the estimate ̂ Q π k p , provided that (50) is satisfied. In addition, it has been shown that (Azar et al., 2013, Lemma 2) the greedy policy induced by k -th iteration of both algorithms - denoted by π k - satisfies ∥ ∥ ̂ Q π k p -̂ Q /star p ∥ ∥ ∞ ≤ 2 γ k +1 (1 -γ ) 2 . Taking ξ = c 1 (1 -γ ) ε |S| 5 |A| 5 and k = c 2 1 -γ log( |S||A| (1 -γ ) εδ ) for some constant c 2 &gt; 0 large enough, one guarantees that π k satisfies (50), which in turn ensures perfect recovery of ̂ π /star p .

<!-- formula-not-decoded -->

## 6.4 Analysis: conservative model-based planning

In view of the conservative model-based planning (10), we begin with the following decomposition:

V /star -V ̂ π c = ( V π /star -̂ V π /star ) + ( ̂ V π /star -̂ V ̂ π c ) + ( ̂ V ̂ π c -V ̂ π c ) . (51) In order to control the second term on the right-hand side of the above identity, we resort to the following lemma, whose proof is postponed to Section B.6.

Lemma 7. It holds that

Combining Lemma 7 with (51), we arrive at

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Clearly, the first term of (53) has already been controlled in Section 6.2, while the second term of (53) is extremely small when we take ξ = O ( (1 -γ ) ε |S| 5 |A| 5 ) . It thus suffices to bound the third term of (53), which again requires decoupling the statistical dependence between π c and ̂ P .

̂ Representing ̂ π c via a small set of policies independent of ̂ P s,a . Akin to our analysis for the perturbed model-based planning algorithm in Section 6.3, a key step lies in demonstrating the connection between ̂ π c and a reasonably small collection of leave-one-out auxiliary MDPs. Towards this, we are in need of the following lemma, which characterizes certain 'stability' of our conservative model-based strategy and lies at the core of our analysis. The proof is deferred to Section B.7.

Lemma 8. Consider any given Q-function ̂ Q : S × A → R and its associated value function ̂ V : S → R (i.e. ̂ V ( s ) = max a ̂ Q ( s, a ) for all s ). Generate an independent random variable ς ∼ Unif (0 , ξ ) . Then with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As an important implication of this lemma, the policy ̂ π c computed in (10) remains unchanged upon slight perturbation of the Q-function estimates. Armed with Lemma 8 and the leave-one-out auxiliary MDPs { M s,a,u } constructed in Section 6.3, our analysis proceeds as follows.

This is a fact that has already been established in the proof of Lemma 4; see (89).

- ̂ · For each ( s, a ) ∈ S × A , there exists a point u 0 ∈ N (1 -γ ) ω (cf. (42)) such that the optimal Q-function of ̂ Q /star s,a,u 0 of ̂ M s,a,u 0 obeys

<!-- formula-not-decoded -->

- Define a conservative policy for M s,a,u 0 as follows:

̂ where ̂ Q /star s,a,u 0 and ̂ V /star s,a,u 0 denote the optimal Q-function and optimal value function of ̂ M s,a,u 0 , respectively. Taking ω = ξδ 8 |S||A| and invoking Lemma 8 and (56), we arrive at

<!-- formula-not-decoded -->

̂ π c = ̂ π s,a,u 0 , c . (57) This result (57) parallels Lemma 4 for the perturbed model-based planning algorithm, revealing that ̂ π c is representable using a policy independent of the randomness associated with ( s, a ) . The remaining proof of Theorem 2 then follows from an identical argument as in the proof of Theorem 1, and is hence omitted here.

## 7 Analysis: finite-horizon MDPs

In this section, we outline the proof of Theorem 4. We shall start by introducing a set of convenient matrix notation before embarking on the main proof.

## 7.1 Matrix notation and Bellman equations

Akin to the infinite-horizon case, we introduce some matrix notation for finite-horizon MDPs. Analogous to Section 6.1, we introduce the following set of notation.

- r h ∈ R |S||A| : a vector representing the reward function r h at step h .
- V π h ∈ R |S| : a vector representing the value function V π h of π at step h .
- V /star h ∈ R |S| : a vector representing the optimal value function V /star h at step h .
- Q π h ∈ R |S||A| : a vector representing the Q-function Q π h of π at step h .
- Q /star h ∈ R |S||A| : a vector representing the optimal Q-function Q /star h at step h .
- P h ∈ R |S||A|×|S| : a matrix representing the probability transition kernel P h at step h .
- P h,π ∈ R |S|×|S| : a submatrix of P h , which consists of the rows with indices coming from { ( s, π h ( s )) | s ∈ S} .
- r π h ∈ R |S| : a subvector of r h , which consists of the rows with indices coming from { ( s, π h ( s )) | s ∈ S} .

Armed with the above notation, the Bellman equation here is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We shall also define ̂ V π h , ̂ Q π h , ̂ V /star h , ̂ Q /star h , ̂ P h , ̂ P h,π w.r.t. the empirical MDP ̂ M in an analogous fashion.

## 7.2 An auxiliary value function sequence obeying Bernstein-type conditions

Similar to the infinite-horizon case (in particular, Section 6.3.1), we find it convenient to introduce a collection of auxiliary vectors as follows. For any l ≥ 0 , define

<!-- formula-not-decoded -->

and for any 1 ≤ h ≤ H and any policy π , define the following sequences recursively:

<!-- formula-not-decoded -->

̂ ̂ As it turns out, if the above auxiliary sequence satisfies certain Bernstein-type conditions, then we can establish a useful upper bound on the entrywise difference between V (0) h and ̂ V (0) h , as stated below. The proof of this lemma is deferred to Section C.1.

As can be easily verified, { V (0) h } coincides with the value function of policy π in the true MDP M , while { V (0) h } corresponds to the value function of policy π in the empirical MDP M .

where we recall that for all s ∈ S

This also allows one to derive

Lemma 9. Suppose that there exists some quantity β 1 &gt; 0 such that the sequence { V ( l ) h +1 } constructed in (62) obeys

<!-- formula-not-decoded -->

for all 1 ≤ h ≤ H .

## 7.3 Proof of Theorem 4

Let us begin with the following elementary decomposition:

<!-- formula-not-decoded -->

We intend to bound both ̂ V ̂ π /star h -V ̂ π /star h and V π /star h -̂ V π /star h by means of Lemma 9. Towards this, we first note that for any policy π , the associated value functions of the MDP and the empirical MDP obey the following Bellman equations:

Here, the inequality follows from the definition V /star h = V π /star h , as well as the fact that ̂ V π /star h ≤ ̂ V ̂ π /star h (since ̂ π /star is the optimal policy of the empirical MDP). In light of (65), there are two terms that need to be controlled.

<!-- formula-not-decoded -->

- ̂ · Let us begin with the optimal policy π /star , which is fixed and statistically independent of the data samples. As a result, if we take π = π /star during the construction (62), then it is clearly seen that ̂ P h is statistically independent of V ( l ) h +1 . Applying the Bernstein inequality together with the union bound then guarantees that: with probability exceeding 1 -δ ,

along with the boundary conditions V π H +1 = ̂ V π H +1 = 0 . This indicates that the vector V (0) h (resp. ̂ V (0) h ) constructed in (62) is precisely the value function of policy π at step h in the true MDP (resp. empirical MDP). As a result, in order to invoke Lemma 9, it is sufficient to verify the Bernstein-type condition (63) w.r.t. policies π /star and π /star for some sufficiently small quantity β 1 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds uniformly over all 0 ≤ l ≤ log 2 H , 1 ≤ h ≤ H -1 , ( s, a ) ∈ S × A , where β 1 is given by

Armed with Condition (66), we can readily invoke Lemma 9 to reach

<!-- formula-not-decoded -->

with probability at least 1 -δ .

- Next, we move on to the policy ̂ π /star by taking π = ̂ π /star during the construction (62). Note that V ( l ) h +1 depends only on ̂ π /star j ( j ≥ h + 1) . In view of our assumption on ̂ π /star (i.e., it is computed backward via dynamic programming), ̂ π /star i is independent of any ̂ P j with j &lt; i , and hence V ( l ) h +1 is statistically

<!-- formula-not-decoded -->

independent from ̂ P h . Consequently, the preceding bounds (66) and (67) continue to hold. All of this immediately results in with probability exceeding 1 -δ .

<!-- formula-not-decoded -->

Substituting the above bounds into (65), we arrive at

<!-- formula-not-decoded -->

with probability greater than 1 -2 δ , provided that N ≥ 48 H log ( H |S||A| δ ) . By taking the right-hand side of (68) to be smaller than ε 1 , we immediately conclude the proof.

## 8 Discussion

This paper has demonstrated that (some variants of) model-based planning algorithms achieve the minimax sample complexity in the presence of a generative model, as soon as the sample size exceeds the order of |S||A| 1 -γ for γ -discounted infinite horizon MDPs and |S||A| H 2 for time-inhomogeneous finite-horizon MDPs (modulo some log factor). Compared to prior literature, our result has considerably broadened the sample size range, allowing us to pin down a complete trade-off curve between sample complexity and statistical accuracy.

The present work opens up several directions for future investigation, which we discuss in passing below.

- Is perturbation or conservative action selection necessary for infinite-horizon MDPs? The planning algorithm analyzed here for infinite-horizon MDPs is either applied to a perturbed variant of the empirical MDP (as in perturbed model-based planning) or run in a conservative manner (as in conservative model-based planning). This, however, gives rise to a natural question regarding the necessity of perturbation or conservative action selection: can we achieve optimal performance directly using plain model-based planning on the empirical MDP? While we conjecture that the answer is affirmative, settling this conjecture requires new techniques beyond the analysis framework of this paper.
- Improved analysis for model-free algorithms. As mentioned previously, a even more severe sample complexity barrier is present in all prior theory regarding model-free approaches (e.g. Li et al. (2022c); Sidford et al. (2018a); Wainwright (2019b)). Our analysis might shed light on how to overcome such barriers for model-free approaches.
- Time-homogeneous finite-horizon MDPs. When it comes to finite-horizon MDPs, the present work concentrates on time-inhomogeneous MDPs where the probability transition kernels may vary across time steps. Another important scenario is concerned with time-homogeneous MDPs, where P 1 = P 2 = · · · = P H . It remains unclear how to develop tight sample analysis for time-homogeneous MDPs due to the lack of statistical independence across time steps (namely, we shall use all samples to estimate the kernels across time steps as they are identical).
- Markovian sample trajectories. Going beyond the generative model, another common form of data samples takes the form of a Markovian sample trajectory, which is generated by taking actions according to a stationary behavior policy in the MDP. This is also referred to as the asynchronous setting in the context of Q-learning (Tsitsiklis, 1994). While the sample complexity of several RL algorithms under this data-generating mechanism has been studied in prior literature (e.g. Li et al. (2023, 2022c); Qu and Wierman (2020)), it remains unclear how to achieve minimax optimality for the full ε -range, due to the complicated statistical dependency across time. The recent work Li et al. (2022b) demonstrated the plausibility of converting a finite-horizon Markovian trajectory into independent samples via two-fold sample splitting in the context of offline RL. It would be interesting to investigate whether

one could employ a similar idea - in conjunction with a proper leave-one-out analysis framework to settle the sample complexity in the presence of Markovian samples.

- Online exploratory RL. In practice, there is no shortage of applications where the learner acquires data samples by executing the MDP in real time. This corresponds to an important setting, called online RL, that requires careful managing of the exploration-exploitation tradeoff (Bai et al., 2019; Jin et al., 2018; Li et al., 2021c). Interestingly, the model-based approach - with proper modification to implement optimism in the face of uncertainty - achieves minimax-optimal regret asymptotically (Azar et al., 2017), although its performance in the sample-starved regime remains largely unknown. It would be of great interest to see whether the analysis ideas developed herein could help characterize the sample efficiency of model-based online RL for the entire ε -range.
- Beyond the tabular setting. The current paper focuses on the tabular setting with finite state and action spaces. While we improve the sample size range, the sample complexities might still be prohibitively large when |S| and |A| are enormous. Therefore, it is desirable to further investigate settings where low-complexity function approximation is employed to improve the efficiency (e.g. Jin et al. (2020); Li et al. (2021b); Yang and Wang (2019)).

## Acknowledgements

Y. Wei is supported in part by the the NSF grants CCF-2106778, DMS-2147546/2015447 and CAREER award DMS-2143215. Y. Chi is supported in part by the ONR grants N00014-18-1-2142 and N00014-191-2404, the NSF grants CCF-1806154, CCF-2007911 and CCF-2106778. Y. Chen is supported in part by the Alfred P. Sloan Research Fellowship, the Google Research Scholar Award, the AFOSR grants FA955019-1-0030 and FA9550-22-1-0198, the ONR grant N00014-22-1-2354, and the NSF grants CCF-2221009, CCF-1907661, DMS-2014279, IIS-2218713 and IIS-2218773. We thank Qiwen Cui for pointing out an issue in Section B.5 in an early version of this paper, and thank Shicong Cen, Chen Cheng and Cong Ma for numerous discussions. Part of this work was done while G. Li, Y. Wei and Y. Chen were visiting the Simons Institute for the Theory of Computing.

## A Preliminary facts

We begin by recording a few elementary facts about P π and P π (see definitions in (24)). These are standard results and we omit the proofs for brevity.

Lemma 10. For any policy π , any probability transition matrix P ∈ R |S||A|×|S| and any 0 &lt; γ &lt; 1 , one has

- (b) All entries of the matrix ( I -γ P π ) -1 are non-negative;
- (a) ( I -γ P π ) -1 = ∑ ∞ i =0 ( γ P π ) i ;
- (c) ‖ ( I -γ P π ) -1 ‖ 1 ≤ 1 / (1 -γ ) ;
- (d) (1 -γ )( I -γ P π ) -1 1 = 1 ;
- (e) For any non-negative vectors 0 ≤ r 1 ≤ r 2 of compatible dimension, one has 0 ≤ ( I -γ P π ) -1 r 1 ≤ ( I -γ P π ) -1 r 2 .

The above results continue to hold if P π is replaced by P π .

## B Proofs of auxiliary lemmas: infinite-horizon MDPs

## B.1 Proofs of Lemma 1 and Lemma 2

Auxiliary notation and preliminaries. Before proceeding, we define several |S| -dimensional auxiliary vectors r ( i ) , V ( i ) , V ( i ) ( 1 ≤ i ≤ m ) recursively as follows where m will be specified momentarily.

<!-- formula-not-decoded -->

A crucial quantity that appears repeatedly in analyzing the above terms is ‖ ( I -γ P π ) -1 √ Var P π ( V ) ‖ ∞ , whose importance was already made apparent in the work Azar et al. (2013). A widely used upper bound on this quantity, originally due to (Azar et al., 2013, Lemma 8), is given by

<!-- formula-not-decoded -->

This bound turns out to be loose for our purpose, and we develop an improved bound as follows, whose proof is deferred to Appendix B.1.1.

Lemma 11. Consider any policy π and any probability transition matrix P ∈ R |S||A|×|S| . Let V be a vector obeying V = ( I -γ P π ) -1 r π for some |S| -dimensional vector r π ≥ 0 . For any 0 &lt; γ &lt; 1 , one has

Remark 6 . In comparison to the bound (70) derived in (Azar et al., 2013, Lemma 8), Lemma 11 offers an improved upper bound stated directly in terms of the properties of V rather than those of r .

<!-- formula-not-decoded -->

As it turns out, Lemma 11 allows us to obtain an entrywise bound for V ( l ) (1 ≤ l ≤ m ) . To begin with, the first term V (1) satisfies since r (1) = √ Var P π [ V (0) ] . Next, for any l &gt; 1 one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second identity results from the definition of r ( l ) , and the last inequality comes from Lemma 11. As a consequence, applying this inequality recursively gives

<!-- formula-not-decoded -->

Main proof. Equipped with the above facts, we are now in a position to prove the lemmas, for which we start with the more general one - Lemma 2. Consider any 0 ≤ l ≤ m . We first observe that

<!-- formula-not-decoded -->

where the second line follows since, by definition, ( I -γ P π ) V ( l ) = r ( l ) . Suppose that there exists some quantity β 1 &gt; 0 such that the following condition holds uniformly for all 0 ≤ l ≤ m . Then this combined with (74) gives

<!-- formula-not-decoded -->

Here, (i) follows since ( I -γ ̂ P π ) -1 is a non-negative matrix, (ii) comes from (75) and the triangle inequality. Recalling the definition of r ( l ) and V ( l ) and invoking Lemma 10(d), we can further bound the above as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above inequality (76) provides a recursive relation that in turn allows for an effective upper bound. Specifically, combining the inequalities (73) and (76) leads to

<!-- formula-not-decoded -->

∥ ∥ ̂ V ( l ) -V ( l ) ∥ ∥ ∞ ≤ γ √ β 1 N ∥ ∥̂ V ( l +1) -V ( l +1) ∥ ∥ ∞ + ( 4 √ β 1 (1 -γ ) N + γβ 1 (1 -γ ) N ) ( 4 γ √ 1 -γ ) l -1 ‖ V (1) ‖ ∞ =: b 1 ∥ ∥ ̂ V ( l +1) -V ( l +1) ∥ ∥ ∞ + b 2 b l -1 3 ‖ V (1) ‖ ∞ . Here for notational simplicity, we introduce and for l ≥ 1 ,

<!-- formula-not-decoded -->

Invoking the above recursive relation, we can arrange terms to reach

<!-- formula-not-decoded -->

Controlling the quantity α 2 . Now it suffices to control the two terms on the right-hand side of the inequality (77) separately, towards which we shall start with the quantity α 2 . Assuming that N ≥ 64 β 1 / (1 -γ ) , one can easily verify b 1 b 3 ≤ 1 / 2 . The summation of the geometric sequence thus gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step holds with the assumption N ≥ 64 β 1 1 -γ .

<!-- formula-not-decoded -->

Controlling the quantity α 1 . Next, we proceed to the quantity α 1 , which requires the control of ∥ ∥ ̂ V ( m ) -V ( m ) ∥ ∥ ∞ . In view of the identity (74), we obtain where the last inequality follows from the Bernstein-type condition (75) and the fact that ( I -γ ̂ P π ) -1 has non-negative entries. By virtue of the simple relation √ Var P π [ V ( m ) ] ≤ ∥ ∥ V ( m ) ∥ ∥ ∞ and the fact that ‖ ( I -γ ̂ P π ) -1 ‖ 1 ≤ 1 1 -γ (cf. Lemma 10(c)), it is further guaranteed that

which combined with the bound (73) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Putting the above bounds together yields

<!-- formula-not-decoded -->

where the last inequality holds provided that N &gt; 16 e 2 1 -γ β 1 .

Putting all this together. Combining the inequalities (77), (78) and (79) gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To finish up, set m = log( e 1 -γ ) and assume that N &gt; 16 e 2 1 -γ β 1 . Recognizing that V (0) = V π and ̂ V (0) = ̂ V π , we arrive at provided that N ≥ 16 e 2 β 1 1 -γ .

Proof of Lemma 2. Invoking the inequality (70) to bound the second term of (80), we reach

<!-- formula-not-decoded -->

where the last inequality uses the elementary fact that ∥ ∥ V π ∥ ∥ ∞ ≤ 1 1 -γ ‖ r ‖ ∞ and the assumption that N &gt; 16 e 2 1 -γ β 1 . We complete the proof of Lemma 2.

Proof of Lemma 1. Finally, to establish Lemma 1, we observe that: for any fixed policy π , the vector V ( l ) is independent of ̂ P π . The Bernstein inequality (e.g. (Agarwal et al., 2020, Lemma 6)) then reveals that with probability at least 1 -δ , holds uniformly for all 0 ≤ l ≤ m . This means that we can take β 1 := 2 log ( 4 m |S| δ ) with m = log( e 1 -γ ) for this case. Combining this with the inequality (80), we derive the advertised instance-dependent bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Further, this taken collectively with (70) and the crude bound ‖ V π ‖ ∞ ≤ 1 1 -γ ‖ r ‖ ∞ gives with the proviso that N ≥ 32 e 2 1 -γ log ( 4 |S| log e 1 -γ δ ) .

## B.1.1 Proof of Lemma 11

To begin with, we make the observation that

<!-- formula-not-decoded -->

where the identity (83) makes use of the relation V = r π + γ P π V . In addition, one can deduce that

<!-- formula-not-decoded -->

Here, (i) comes from Jensen's inequality (so that E [ √ v ] ≤ √ E [ v ] ) recognizing that each row of (1 -γ )( I -γ P π ) -1 is a probability distribution, and Lemma 10(d), (ii) is an elementary fact established in (Agarwal et al., 2020, Lemma 4). Combining Lemma 10(e) and the inequality (84) further yields

<!-- formula-not-decoded -->

Regarding the first term of (85), we observe that ‖ ( V ◦ V ) ‖ ∞ = ‖ V ‖ 2 ∞ . When it comes to the second term of (85), it is seen that where the last step arises from the triangle inequality and √ a + b ≤ √ a + √ b . This leaves us with two terms to deal with.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the first inequality holds true since ( I -γ 2 P π ) -1 = ∑ ∞ i =0 γ 2 i P i π ≤ ∑ ∞ i =0 γ i P i π = ( I -γ P π ) -1 , while the second line holds since V , r and ( I -γ P π ) -1 are all non-negative. Substitution into (85) thus yields as claimed.

## B.2 Proof of Lemma 3

To establish Lemma 3, it suffices to check that V /star and Q /star satisfy the Bellman optimality equations underlying M s,a,u /star . Towards this end, we study the absorbing state-action pair ( s, a ) and other pairs separately. For notational simplicity, we shall let P abs and r abs ( · , · ) denote respectively the probability transition matrix and the reward function associated with M s,a,u /star .

First, we observe that, by construction,

<!-- formula-not-decoded -->

Recall that V /star satisfies the Bellman optimality equation w.r.t. the original MDP, namely, Q /star ( s, a ) = r ( s, a )+ γ ( PV /star ) s,a . This together with our choice of u /star gives

<!-- formula-not-decoded -->

Putting the above identities together, we arrive at

<!-- formula-not-decoded -->

Next, consider any state-action pair ( s ′ , a ′ ) = ( s, a ) . Recalling again the properties of M s,a,u /star , we reach

/negationslash

<!-- formula-not-decoded -->

Here the last identity is due to the Bellman equation w.r.t. the original MDP. Combining (86) and (87) implies that V /star and Q /star satisfy Bellman's optimality equations in M s,a,u /star , thus concluding the proof.

## B.3 Proof of Lemma 4

Our first observation is that ̂ Q /star s,a,u satisfies Lipschitz continuity w.r.t. u in the sense that

The proof of this relation is identical to that of (Agarwal et al., 2020, Lemma 8); we omit here for brevity. In view of Lemma 3, if we set ̂ u /star := r ( s, a ) + γ ( P V /star ) s,a -γ V /star ( s ) , then one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, there exists a point u 0 in the epsilon-net N (1 -γ ) ω/ 4 such that | ̂ u /star -u 0 | ≤ (1 -γ ) ω/ 4 , which combined with the Lipschitz continuity property (88) gives

Additionally, for any s ′ ∈ S and any a 1 , a 2 ∈ A with a 1 = a 2 , we have the following decomposition

/negationslash

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

where the last inequality invokes the inequality (89). Moreover, our separation condition defined in (41) requires that: for any s ′ ∈ S and any a 2 = ̂ π /star ( s ′ ) , one has ̂ Q /star ( s ′ , ̂ π /star ( s ′ )) -̂ Q /star ( s ′ , a 2 ) ≥ ω , which together with (90) reveals that

Given that ̂ π /star s,a,u 0 ( s ′ ) := arg max a ′ ̂ Q /star s,a,u 0 ( s ′ , a ′ ) , it is seen from (91) that which holds true for all s ′ ∈ S . This concludes the proof.

## B.4 Proof of Lemma 5

For each state-action pair ( s, a ) , let us construct the epsilon-net N (1 -γ ) ω/ 4 as in the expression (42). For every u ∈ N (1 -γ ) ω/ 4 , recall that ̂ π /star s,a,u is defined as the optimal policy with respect to the ( s, a ) -absorbing MDP ̂ M s,a,u . By construction, the set of policies ̂ π /star s,a,u ( u ∈ N (1 -γ ) ω/ 4 ) is independent of ̂ P s,a . The Bernstein inequality (e.g. (Agarwal et al., 2020, Lemma 6)) taken together with the union bound thus guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

We start by bounding ∥ ∥̂ V ̂ π /star -V ̂ π /star ∥ ∥ ∞ . Recall the definition of the series { V ( l ) } in (69). Throughout this proof, we shall write V ( l ) π instead in order to make apparent the dependency on the policy π .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds uniformly over all 0 ≤ l ≤ log e 1 -γ , u ∈ N (1 -γ ) ω/ 4 , ( s, a ) ∈ S × A . Here, β 1 is given by

where we have used the fact |N (1 -γ ) ω/ 4 | ≤ 8 (1 -γ ) 2 ω . In addition, for any 0 &lt; ω &lt; 1 , Lemma 4 guarantees that for each state-action pair ( s, a ) ∈ S × A , there exists a point u 0 ∈ N (1 -γ ) ω/ 4 such that ̂ π /star = ̂ π /star s,a,u 0 . Invoking this important fact, we obtain

The above inequality further allows us to deduce that, with probability 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above derivation validates the assumption required for Lemma 2. As a result, if N &gt; 16 e 2 1 -γ β 1 and 0 &lt; ω &lt; 1 , then Lemma 2 leads to the advertised bound

<!-- formula-not-decoded -->

with probability at least 1 -δ . This together with (34) and (93) immediately establishes Lemma 5.

<!-- formula-not-decoded -->

## B.5 Proof of Lemma 6

The proofs for Q /star p and ̂ Q /star p are exactly the same; for the sake of conciseness, we shall only provide the proof for Q /star p . Here we aim to prove a more general result than Lemma 6, namely, with probability at least 1 -δ ,

/negationslash

Consider any state s and any actions a 1 = a 2 . In what follows, we allow r p ( s, a 1 ) = τ to vary, while freezing the values of all other rewards { r p (˜ s, a ) | (˜ s, a ) = ( s, a 1 ) } . To streamline notation, we define

/negationslash

<!-- formula-not-decoded -->

/negationslash

- r τ = [ r τ ( s, a )] ( s,a ) ∈S×A : the reward vector obeying

<!-- formula-not-decoded -->

/negationslash

- Q /star τ : the optimal Q-function when the reward vector is r τ ;
- V /star τ : the optimal value function when the reward vector is r τ ;
- π /star τ : the optimal policy when the reward vector is r τ .

Additionally, we claim for the moment that there exists a phase transition boundary τ th such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash

The proof of this claim is deferred to the end of this section. To establish Lemma 6, the idea is to control the size of the set

I 0 ,ω := { τ | ∣ ∣ Q /star τ ( s, a 1 ) -Q /star τ ( s, a 2 ) ∣ ∣ &lt; ω } (96) for some ω &gt; 0 to be specified shortly. As motivated by (95), we further break down this set into two parts I 0 ,ω = I 1 ,ω ∪ I 2 ,ω , where

In what follows, we first control the size of each set separately, and then demonstrate that the probability of these events happening is very small.

<!-- formula-not-decoded -->

Step 1. We begin with I 1 ,ω associated with the range τ &lt; τ th . In this case, the value function V /star τ does not vary with τ , since the reward r τ ( s, a 1 ) = τ is never active when calculating V /star τ (by virtue of (95a)). Thus, the Bellman equation allows us to write

<!-- formula-not-decoded -->

for some quantities B 1 and B 2 , where neither B 1 nor B 2 depends on the value of τ . Armed with this observation, we can easily show that: for any ω &gt; 0 , the interval I 1 ,ω (cf. (97a)) obeys

<!-- formula-not-decoded -->

and hence has length (or Lebesgue measure) at most 2 ω .

Step 2. We then move on to I 2 ,ω associated with the range τ &gt; τ th in which case π /star τ ( s ) = a 1 . Towards this, we first make some useful observations.

- To begin with, given the relation Q /star τ = r τ + γ PV /star τ , it is easily seen that for any τ 2 &gt; τ 1 &gt; τ th ,

<!-- formula-not-decoded -->

/negationslash

In addition, for any state-action pair (˜ s, a ) = ( s, a 1 ) , by construction we have r τ 2 (˜ s, a ) -r τ 1 (˜ s, a ) = 0 , which together with (98) indicates that

<!-- formula-not-decoded -->

/negationslash

- Next, observe that for any τ 2 &gt; τ 1 &gt; τ th ,

<!-- formula-not-decoded -->

and hence ‖ Q /star τ 2 -Q /star τ 1 ‖ ∞ ≥ ‖ V /star τ 2 -V /star τ 1 ‖ ∞ . This combined with (99) and the fact γ &lt; 1 implies that

<!-- formula-not-decoded -->

which together with the facts V /star τ 1 ( s ) = Q /star τ 1 ( s, a 1 ) and V /star τ 2 ( s ) = Q /star τ 2 ( s, a 1 ) (by virtue of (95b)) yields

<!-- formula-not-decoded -->

Invoke the Bellman equation to further derive

<!-- formula-not-decoded -->

where the last inequality holds since r τ 2 + γ PV /star τ 2 -r τ 1 -γ PV /star τ 1 ≥ r τ 2 -r τ 1 ≥ 0 (due to the monotonicity properties r τ 2 ≥ r τ 1 and V /star τ 2 ≥ V /star τ 1 ), and the last identity follows from the definition of r τ .

With the above two properties (99) and (102) in mind, we are ready to locate I 2 ,ω by showing that

Given that Q /star τ th ( s, a 1 ) ≥ Q /star τ th ( s, a 2 ) (in view of (95)), we have for any τ ≥ τ th and any a 2 = a 1 that

Here, the last inequality holds since

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) follows from (99) and (ii) is due to (101). As a result, for any τ &gt; τ th + ω 1 -γ , one can invoke (104) and (102) to see that

/negationslash which necessarily implies that such a τ does not lie within the interval I 2 ,ω as defined in (97b). This establishes the claimed relation (103).

<!-- formula-not-decoded -->

Step 3. Putting the results in the above two steps together, we see the set I 0 ,ω (cf. (96)) has total length (or Lebesgue measure) at most 3 ω 1 -γ . Given that r p ( s, a ) = r ( s, a ) + ζ ( s, a ) with ζ ( s, a ) ∼ Unif (0 , ξ ) , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By setting ω = δ (1 -γ ) ξ 3 |S||A| 2 , we arrive at

Finally, taking the union bound over all s, a 1 , a 2 , we conclude that

/negationslash thus establishing Lemma 6 as long as the claim (95) is valid.

<!-- formula-not-decoded -->

Proof of the claim (95) . To establish the claim, it suffices to take

<!-- formula-not-decoded -->

/negationslash

It thus suffices to verify (95b) for our choice (105). Towards this, suppose instead that there exist some τ 3 &lt; τ th ≤ τ 2 &lt; τ 1 such that

<!-- formula-not-decoded -->

/negationslash

/negationslash

It is straightforward to see that V /star τ 1 = V /star τ 3 , since in both cases, the reward r τ ( s, a 1 ) does not enter the calculation of the optimal value function (while the rewards in other state-action pairs are identical in both cases). In view of the monotonicity of the value function w.r.t. the reward vector, we have

<!-- formula-not-decoded -->

However, this contradicts our assumption that a 1 is the optimal action for state s at τ 2 but not at τ 3 , since enlarging τ 2 to τ 3 otherwise will enlarge the optimal value function V /star τ 3 . We have thus established (95).

## B.6 Proof of Lemma 7

To begin, we find it helpful to introduce a modified reward function r ∈ R |S||A| as follows

Armed with this new reward function, we subsequently define a vector ˜ Q = [ ˜ Q ( s, a )] as follows

<!-- formula-not-decoded -->

˜ Q := ˜ r + γ ̂ P ̂ V /star . (107) In view of the Bellman optimality equation ̂ Q /star = r + γ ̂ P ̂ V /star , we see that ˜ Q satisfies ˜ Q = ˜ r + ̂ Q /star -r , which combined with the construction (106) gives

As a consequence, it is easily seen that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This taken collectively with (107) demonstrates that ˜ Q and ̂ V /star are respectively the optimal Q-function and optimal value function of the MDP ˜ M = ( S , A , ˜ r, ̂ P,γ ) , since they satisfy the Bellman optimality condition w.r.t. ˜ M . In addition, if we let ˜ V ̂ π c represent the value function of the policy ̂ π c in ˜ M , then the preceding relation clearly implies that ˜ V ̂ π c = ̂ V /star . Using the above properties, one can deduce that

Here, (i) is due to the Bellman equation, (ii) relies on the fact ‖ ( I -γ ̂ P ̂ π c ) -1 ‖ 1 = 1 1 -γ , (iii) arises since ̂ V /star ( s ) -̂ Q /star ( s, ̂ π c ( s )) ≤ ς by construction of ̂ π c , whereas (iv) is valid since ς ∈ [0 , ξ ] . The lemma then follows by recognizing that ̂ V π /star ≤ ̂ V /star due to the optimality of ̂ V /star .

## B.7 Proof of Lemma 8

We first make the key observation that, with probability at least 1 -δ , the following event holds true:

<!-- formula-not-decoded -->

To justify this claim, note that its complementary event satisfies

<!-- formula-not-decoded -->

where the first inequality applies the union bound, and the last line follows since ς ∼ Unif (0 , ξ ) . Here, we abbreviate [ a ± b ] := [ a -b, a + b ] . Combining (108) with the assumption (55), we further reach

<!-- formula-not-decoded -->

for all ( s, a ) ∈ S × A , or equivalently,

<!-- formula-not-decoded -->

holds for all ( s, a ) ∈ S × A .

<!-- formula-not-decoded -->

We are now prepared to justify the claim of this lemma. To begin with, consider any action ̂ a ∈ { a ∈ A : Q ( s, a ) &gt; V ( s ) -ς } . Comparing this with the condition (109), we can easily see that the only possibility is

Therefore, by invoking a basic decomposition, we ensure that

<!-- formula-not-decoded -->

This essentially implies that

<!-- formula-not-decoded -->

Applying exactly the same argument for any action a ∈ { a ∈ A : Q ( s, a ) ≤ V ( s ) -ς } , we can also derive

These two set inequalities taken collectively establish the lemma.

<!-- formula-not-decoded -->

## C Proofs of auxiliary lemmas: finite-horizon MDPs

To begin, we find it helpful to control the entrywise magnitudes of { V ( l ) h } . This is accomplished via the following lemma, with the proof postponed to Section C.2.

Lemma 12. The vectors { V ( l ) h } defined in (62) obey for all 1 ≤ h ≤ H and all l ≥ 0 .

<!-- formula-not-decoded -->

## C.1 Proof of Lemma 9

Consider any l obeying 1 ≤ l ≤ m := log 2 H . By construction (cf. (62)), we see that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which connects V ( l ) h and ̂ V ( l ) h with V ( l ) h +1 and ̂ V ( l ) h +1 . Apply the above relation recursively and make use of the conditions (61) to arrive at where we adopt the convenient notation and let ∏ h -1 i = h ̂ P i,π = I . According to the triangle inequality, we can further deduce that

where the second line follows from the assumption (63), and the third line makes use of the definition (62) of r ( l ) h and the elementary identity ∏ j -1 i = h ̂ P i,π 1 = 1 (since each ̂ P i,π is a probability transition matrix). In view of the construction (62), we can also derive recursively that

<!-- formula-not-decoded -->

which combined with (113) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Further, the above inequality together with the triangle inequality immediately results in the following recursive relation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

thus revealing a useful connection between ‖ ̂ V ( l ) h -V ( l ) h ‖ ∞ and ‖ ̂ V ( l +1) h -V ( l +1) h ‖ ∞ . Applying this relation recursively with a little algebra leads to

<!-- formula-not-decoded -->

Additionally, it is easily seen from the definition (62) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which taken together with (113) and the elementary identity ∏ j -1 i = h ̂ P i,π 1 = 1 implies that

Substitution into (116) results in

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To finish up, it remains to control the terms { ‖ V ( l ) h ‖ ∞ } . Towards this, combining Lemma 12 with the above inequality (117) yields

Here, the last inequality holds true since

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

provided that √ 3 β 1 H N ≤ 1 / 2 . Invoking the assumption N ≥ 12 Hβ 1 again and taking m = log 2 H , we have ( √ 3 β 1 H N ) m ≤ 1 /H . This combined with (119) immediately leads to with the proviso that N ≥ 3 β 1 H . This concludes the proof.

## C.2 Proof of Lemma 12

To begin with, it is seen from the notation (28) that for all 1 ≤ j ≤ H , where the second identity makes use of the fact that V ( l ) j = r ( l ) j + P j,π V ( l ) j +1 (cf. (62)). Moreover, from the construction (62) we can easily derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above two results taken collectively give

<!-- formula-not-decoded -->

Here, (i) arises from Jensen's inequality; (ii) holds true due to the Cauchy-Schwarz inequality; (iii) follows from (120). By telescoping summation, one further arrives at

<!-- formula-not-decoded -->

∥ ∥ ∥ ∥ H -1 ∏ i = h P i,π ( V ( l ) H ◦ V ( l ) H ) ∥ ∥ ∥ ∥ ∞ ≤ ∥ ∥ ∥ ∥ H -1 ∏ i = h P i,π ∥ ∥ ∥ ∥ 1 ∥ ∥ ( V ( l ) H ◦ V ( l ) H )∥ ∥ ∞ ≤ max j ∥ ∥ V ( l ) j ∥ ∥ 2 ∞ . As a consequence, the above inequality allows one to deduce that

Here (iv) invokes the relation V ( l ) h = ∑ H -1 j = h ∏ j -1 i = h P i,π r ( l ) j (see (121)); and (v) holds true since

<!-- formula-not-decoded -->

and therefore, where the last inequality arises from the trivial upper bound max j ‖ V (0) j ‖ ∞ ≤ H .

## References

- Agarwal, A., Kakade, S., and Yang, L. F. (2020). Model-based reinforcement learning with a generative model is minimax optimal. Conference on Learning Theory (COLT) .
- Azar, M. G., Munos, R., and Kappen, B. (2012). On the sample complexity of reinforcement learning with a generative model. arXiv preprint arXiv:1206.6461 .
- Azar, M. G., Munos, R., and Kappen, H. J. (2013). Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3):325-349.
- Azar, M. G., Osband, I., and Munos, R. (2017). Minimax regret bounds for reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 , pages 263-272. JMLR. org.
- Bai, Y., Xie, T., Jiang, N., and Wang, Y.-X. (2019). Provably efficient Q-learning with low switching cost. Advances in Neural Information Processing Systems , 32.
- Beck, C. L. and Srikant, R. (2012). Error bounds for constant step-size Q-learning. Systems &amp; control letters , 61(12):1203-1208.
- Bellman, R. (1952). On the theory of dynamic programming. Proceedings of the National Academy of Sciences of the United States of America , 38(8):716.
- Bertsekas, D. P. (2017). Dynamic programming and optimal control (4th edition) . Athena Scientific.
- Bhandari, J., Russo, D., and Singal, R. (2018). A finite time analysis of temporal difference learning with linear function approximation. In Conference On Learning Theory , pages 1691-1692.
- Bradtke, S. J. and Barto, A. G. (1996). Linear least-squares algorithms for temporal difference learning. Machine learning , 22(1-3):33-57.
- Cai, Q., Yang, Z., Lee, J. D., and Wang, Z. (2019). Neural temporal-difference learning converges to global optima. In Advances in Neural Information Processing Systems , pages 11312-11322.
- Chen, Y., Chi, Y., Fan, J., and Ma, C. (2021). Spectral methods for data science: A statistical perspective. Foundations and Trends® in Machine Learning , 14(5):566-806.
- Chen, Y., Fan, J., Ma, C., and Wang, K. (2019a). Spectral method and regularized MLE are both optimal for top-K ranking. Annals of statistics , 47(4):2204.
- Chen, Y., Fan, J., Ma, C., and Yan, Y. (2019b). Inference and uncertainty quantification for noisy matrix completion. Proceedings of the National Academy of Sciences , 116(46):22931-22937.
- Chen, Z., Maguluri, S. T., Shakkottai, S., and Shanmugam, K. (2020). Finite-sample analysis of stochastic approximation using smooth convex envelopes. arXiv preprint arXiv:2002.00874 .
- Cui, Q. and Yang, L. F. (2021). Minimax sample complexity for turn-based stochastic game. In Uncertainty in Artificial Intelligence , pages 1496-1504. PMLR.
- Dalal, G., Szörényi, B., Thoppe, G., and Mannor, S. (2018). Finite sample analyses for TD(0) with function approximation. In Thirty-Second AAAI Conference on Artificial Intelligence .

<!-- formula-not-decoded -->

- Domingues, O. D., Ménard, P., Kaufmann, E., and Valko, M. (2021). Episodic reinforcement learning in finite MDPs: Minimax lower bounds revisited. In Algorithmic Learning Theory , pages 578-598.
- El Karoui, N. (2015). On the impact of predictor geometry on the performance on high-dimensional ridgeregularized generalized robust regression estimators. Probability Theory and Related Fields , pages 1-81.
- Even-Dar, E. and Mansour, Y. (2003). Learning rates for Q-learning. Journal of machine learning Research , 5(Dec):1-25.
- Fan, J., Wang, Z., Xie, Y., and Yang, Z. (2019). A theoretical analysis of deep Q-learning. arXiv preprint arXiv:1901.00137 .
- Gupta, H., Srikant, R., and Ying, L. (2019). Finite-time performance bounds and adaptive learning rate selection for two time-scale reinforcement learning. In Advances in Neural Information Processing Systems , pages 4706-4715.
- Jaakkola, T., Jordan, M. I., and Singh, S. P. (1994). Convergence of stochastic iterative dynamic programming algorithms. In Advances in neural information processing systems , pages 703-710.
- Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. (2018). Is Q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4863-4873.
- Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. (2020). Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pages 2137-2143. PMLR.
- Kakade, S. (2003). On the sample complexity of reinforcement learning . PhD thesis, University of London.
- Kaledin, M., Moulines, E., Naumov, A., Tadic, V., and Wai, H.-T. (2020). Finite time analysis of linear two-timescale stochastic approximation with Markovian noise. arXiv preprint arXiv:2002.01268 .
- Kearns, M., Mansour, Y., and Ng, A. Y. (2002). A sparse sampling algorithm for near-optimal planning in large Markov decision processes. Machine learning , 49(2-3):193-208.
- Kearns, M. J. and Singh, S. P. (1999). Finite-sample convergence rates for Q-learning and indirect algorithms. In Advances in neural information processing systems , pages 996-1002.
- Khamaru, K., Pananjady, A., Ruan, F., Wainwright, M. J., and Jordan, M. I. (2020). Is temporal difference learning optimal? an instance-dependent analysis. arXiv preprint arXiv:2003.07337 .
- Lakshminarayanan, C. and Szepesvari, C. (2018). Linear stochastic approximation: How far does constant step-size and iterate averaging go? In International Conference on Artificial Intelligence and Statistics , pages 1347-1355.
- Lattimore, T. and Hutter, M. (2012). PAC bounds for discounted MDPs. In International Conference on Algorithmic Learning Theory , pages 320-334. Springer.
- Li, G., Cai, C., Chen, Y., Gu, Y., Wei, Y., and Chi, Y. (2021a). Tightening the dependence on horizon in the sample complexity of Q-learning. In International Conference on Machine Learning , pages 6296-6306.
- Li, G., Cai, C., Chen, Y., Wei, Y., and Chi, Y. (2023). Is Q-learning minimax optimal? a tight sample complexity analysis. accepted to Operations Research .
- Li, G., Chen, Y., Chi, Y., Gu, Y., and Wei, Y. (2021b). Sample-efficient reinforcement learning is feasible for linearly realizable MDPs with limited revisiting. Advances in Neural Information Processing Systems , 34:16671-16685.
- Li, G., Chi, Y., Wei, Y., and Chen, Y. (2022a). Minimax-optimal multi-agent RL in zero-sum Markov games with a generative model. arXiv preprint arXiv:2208.10458 .
- Li, G., Shi, L., Chen, Y., Chi, Y., and Wei, Y. (2022b). Settling the sample complexity of model-based offline reinforcement learning. arXiv preprint arXiv:2204.05275 .

- Li, G., Shi, L., Chen, Y., Gu, Y., and Chi, Y. (2021c). Breaking the sample complexity barrier to regretoptimal model-free reinforcement learning. Advances in Neural Information Processing Systems , 34.
- Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2022c). Sample complexity of asynchronous Q-learning: Sharper analysis and variance reduction. IEEE Transactions on Information Theory , 68(1):448-473.
- Ma, C., Wang, K., Chi, Y., and Chen, Y. (2020). Implicit regularization in nonconvex statistical estimation: Gradient descent converges linearly for phase retrieval, matrix completion and blind deconvolution. Foundations of Computational Mathematics , 20(3):451-632.
- Mou, W., Li, C. J., Wainwright, M. J., Bartlett, P. L., and Jordan, M. I. (2020). On linear stochastic approximation: Fine-grained Polyak-Ruppert and non-asymptotic concentration. arXiv preprint arXiv:2004.04719 .
- Pananjady, A. and Wainwright, M. J. (2019). Value function estimation in Markov reward processes: Instance-dependent /lscript ∞ -bounds for policy evaluation. arXiv preprint arXiv:1909.08749 .
- Qu, G. and Wierman, A. (2020). Finite-time analysis of asynchronous stochastic approximation and Qlearning. In Conference on Learning Theory , pages 3185-3205.
- Shah, D. and Xie, Q. (2018). Q-learning with nearest neighbors. In Advances in Neural Information Processing Systems , pages 3111-3121.
- Shi, L., Li, G., Wei, Y., Chen, Y., and Chi, Y. (2022). Pessimistic Q-learning for offline reinforcement learning: Towards optimal sample complexity. International Conference on Machine Learning .
- Sidford, A., Wang, M., Wu, X., Yang, L., and Ye, Y. (2018a). Near-optimal time and sample complexities for solving Markov decision processes with a generative model. In Advances in Neural Information Processing Systems , pages 5186-5196.
- Sidford, A., Wang, M., Wu, X., and Ye, Y. (2018b). Variance reduced value iteration and faster algorithms for solving Markov decision processes. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 770-787.
- Srikant, R. and Ying, L. (2019). Finite-time error bounds for linear stochastic approximation and TD learning. In Conference on Learning Theory , pages 2803-2830.
- Strehl, A. L., Li, L., Wiewiora, E., Langford, J., and Littman, M. L. (2006). PAC model-free reinforcement learning. In International Conference on Machine Learning , pages 881-888.
- Sutton, R. S. and Barto, A. G. (2018). Reinforcement learning: An introduction . MIT press.
- Szepesvári, C. (1998). The asymptotic convergence-rate of Q-learning. In Advances in Neural Information Processing Systems , pages 1064-1070.
- Szepesvári, C. (2010). Algorithms for reinforcement learning. Synthesis lectures on artificial intelligence and machine learning , 4(1):1-103.
- Tsitsiklis, J. and Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. IEEE Transactions on Automatic Control , 42(5):674-690.
- Tsitsiklis, J. N. (1994). Asynchronous stochastic approximation and Q-learning. Machine learning , 16(3):185202.
- Tu, S. and Recht, B. (2019). The gap between model-based and model-free methods on the linear quadratic regulator: An asymptotic viewpoint. In Conference on Learning Theory , pages 3036-3083.
- Vershynin, R. (2018). High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press.

- Wainwright, M. J. (2019a). Stochastic approximation with cone-contractive operators: Sharp /lscript ∞ -bounds for Q-learning. arXiv preprint arXiv:1905.06265 .
- Wainwright, M. J. (2019b). Variance-reduced Q-learning is minimax optimal. arXiv preprint arXiv:1906.04697 .
- Wang, B., Yan, Y., and Fan, J. (2021). Sample-efficient reinforcement learning for linearly-parameterized MDPs with a generative model. Neural Information Processing Systems .
- Wang, M. (2019). Randomized linear programming solves the Markov decision problem in nearly linear (sometimes sublinear) time. Mathematics of Operations Research .
- Xu, P. and Gu, Q. (2020). A finite-time analysis of Q-learning with neural network function approximation. In International Conference on Machine Learning , pages 10555-10565.
- Xu, T., Zou, S., and Liang, Y. (2019). Two time-scale off-policy TD learning: Non-asymptotic analysis over Markovian samples. In Advances in Neural Information Processing Systems , pages 10633-10643.
- Yan, Y., Chen, Y., and Fan, J. (2021). Inference for heteroskedastic PCA with missing data. arXiv preprint arXiv:2107.12365 .
- Yan, Y., Li, G., Chen, Y., and Fan, J. (2022a). The efficacy of pessimism in asynchronous Q-learning. arXiv preprint arXiv:2203.07368 .
- Yan, Y., Li, G., Chen, Y., and Fan, J. (2022b). Model-based reinforcement learning is minimax-optimal for offline zero-sum Markov games. arXiv preprint arXiv:2206.04044 .
- Yang, L. and Wang, M. (2019). Sample-optimal parametric Q-learning using linearly additive features. In International Conference on Machine Learning , pages 6995-7004.
- Yin, M., Bai, Y., and Wang, Y.-X. (2021). Near-optimal provable uniform convergence in offline policy evaluation for reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 1567-1575.