## Near-Optimal Sample Complexity for Iterated CVaR Reinforcement Learning with a Generative Model

## Zilong Deng

ECEE, Arizona State University

## Simon Khan

Air Force Research Laboratory

## Abstract

In this work, we study the sample complexity problem of risk-sensitive Reinforcement Learning (RL) with a generative model, where we aim to maximize the Conditional Value at Risk (CVaR) with risk tolerance level τ at each step, named Iterated CVaR. We develop nearly matching upper and lower bounds on the sample complexity for this problem. Specifically, we first prove that a value iteration-based algorithm, ICVaR-VI, achieves an /epsilon1 -optimal policy with at most ˜ O ( SA (1 -γ ) 4 τ 2 /epsilon1 2 ) samples, where γ is the discount factor, and S, A are the sizes of the state and action spaces. Furthermore, if τ ≥ γ , then the sample complexity can be further improved to ˜ O ( SA (1 -γ ) 3 /epsilon1 2 ) . We further show a minimax lower bound of ˜ O ( (1 -γτ ) SA (1 -γ ) 4 τ/epsilon1 2 ) . For a constant risk level 0 &lt; τ ≤ 1, our upper and lower bounds match with each other, demonstrating the tightness and optimality of our analyses. We also investigate a limiting case with a small risk level τ , called Worst-Path RL, where the objective is to maximize the minimum possible cumulative reward. We develop matching upper and lower bounds of ˜ O ( SA p min ) , where p min denotes the minimum non-zero reaching probability of the transition kernel.

## 1 Introduction

Reinforcement learning (RL) (Sutton &amp; Barto, 2018) is a foundational framework for solving sequential decision-making problems, and it finds a wide

Proceedings of the 28 th International Conference on Artificial Intelligence and Statistics (AISTATS) 2025, Mai Khao,

Thailand. PMLR: Volume 258. Copyright 2025 by the author(s).

## Shaofeng Zou

ECEE, Arizona State University range of applications in e.g., large language models (Ouyang et al., 2022), robotics (Kober et al., 2013), and finance (Charpentier et al., 2021). Recently there has been a surge of interest in understanding its fundamental sample complexity, e.g., Bhandari et al. (2018); Srikant &amp; Ying (2019); Zou et al. (2019); Agarwal et al. (2021); Bhandari &amp; Russo (2024). However, the main focus was on the risk-neural setting, where the objective is to maximize the expected total rewards accumulated over time. As RL is increasingly applied to real-world sequential decision-making tasks, it often becomes essential to account for risk rather than simply optimizing for the total reward. This is particularly important when safety and worstcase avoidance considerations exist, e.g., in finance and investment, healthcare, autonomous systems, and process engineering. Nevertheless, a fundamental understanding of the sample complexity for risk-sensitive RL remains largely unexplored. In this paper, we focus on one of such problems and aim to understand the fundamental sample complexity for iterated CVaR RL with a generative model.

A widely used risk measure is called coherent risk measure, which satisfies the following four properties: (i) monotonicity;(ii) translation invariance; (iii) sub-additivity; and (iv) positive homogeneity (Artzner et al., 1999). Conditional Value at Risk (CVaR) is a popular coherent risk measure(Rockafellar et al., 2000). There are two types of CVaR RL objectives, Iterated (dynamic) CVaR RL (Hardy &amp; Wirch, 2004) and Static CVaR RL (Wang et al., 2023; Zhao et al., 2023). Iterated CVaR is a special case of Markov coherent risk (Tamar et al., 2015) where the coherent risk measure is CVaR. It has an iterative structure and focuses on the worst portion of the reward at each step. In Iterated CVaR RL problem, the agent aims to maximize the average of the worst portion at every step. Static CVaR RL (Wang et al., 2023; Zhao et al., 2023)) on the other hand, aims to maximize the CVaR of the total reward.

Static CVaR RL and Iterated CVaR RL are quite different. In Static CVaR RL, the optimal pol-

icy is history-dependent and not stationary. In B¨ auerle &amp; Ott (2011), it is shown that Static CVaR RL can be optimally solved by resolving to standard, risk-neutral RL in an augmented MDP. In Iterated CVaR RL, the optimal policy is Markovian and stationary if the environment is stationary. Broadly speaking, Static CVaR RL is more concerned with the overall risk of the total reward and may permit the agent to visit catastrophic states as long as the risk of the cumulative reward remains acceptable. In contrast, Iterated CVaR RL assesses risk at each step, offering a more cautious approach by preventing the agent from entering catastrophic states at any point in the trajectory.

In this paper, we focus on Iterated CVaR RL, and we aim to theoretically understand the sample complexity with access to a generator (Kearns &amp; Singh, 1998). With access to a generative model, one can draw samples from the transition kernel of a Markov decision process (MDP) conditioned on any arbitrary state-action pair. We then take a model-based approach, where we first construct a maximum likelihood estimate of the transition kernel and find the optimal policy for the estimated MDP model. In risk-neural RL, it was shown that such an approach achieves the minimax optimal sample complexity (Gheshlaghi Azar et al., 2013; Agarwal et al., 2020). In Iterated CVaR RL, the goal is to find the policy that maximizes the worst τ -portion of rewardto-go at each step. Clearly, this ensures that policy avoids getting into catastrophic states, prioritizing risk-sensitive behavior. Since the objective focuses on ignoring certain 'good' states and prioritizing the worst-case scenarios, we undoubtedly require more samples to learn a safe policy. This naturally raises the question: how many samples are needed to produce an optimal risk-sensitive policy?

In this paper, we make the connection between Iterated CVaR RL and distributionally robust RL (Iyengar, 2005; Nilim &amp; El Ghaoui, 2004) using the dual form of CVaR. We then decompose the error using approaches in robust RL (Shi et al., 2023) and develop novel analytical techniques to bound the change in CVaR resulting from approximation error. We prove that a value iteration-based algorithm, ICVaR-VI, achieves an /epsilon1 -optimal policy with at most ˜ O ( SA (1 -γ ) 4 τ 2 /epsilon1 2 ) samples, where γ is the discount factor, τ is the risk tolerance, and S, A are the sizes of the state and action spaces. Moreover, if τ ≥ γ , we further derive an improved sample complexity of ˜ O ( SA (1 -γ ) 3 /epsilon1 2 ) , which actually matches with the minimax optimal sample complexity for risk-neural RL (also see Table 1). We then develop a minimax lower bound that for any τ and γ , there always exists an

MDP, for which at least ˜ O ( (1 -γτ ) SA (1 -γ ) 4 τ/epsilon1 2 ) are needed. Comparing our upper and lower bounds, they match with each other in the order of sizes of state and action spaces S, A and effective horizon (1 -γ ) -1 when the risk level is a constant in (0 , 1].

Finally, we investigate a limiting case, named WorstPath RL (Du et al., 2022), where we consider a small risk level τ smaller than the minimum non-zero reaching probability of the transition kernel p min . In this case, the CVaR risk measure actually tries to find the worst-case state. This problem cannot be directly solved by taking the limit τ → 0, and the previous lower and upper bounds will also go to infinity as they depend on τ -1 . To tackle this problem, we design a new algorithm based on the reduced Bellman operator and develop matching upper and lower bounds of ˜ O ( SA p min ) , where p min denotes the minimum non-zero reaching probability of the transition kernel.

## 2 Related Work

Static CVaR RL: There is a long line of works focusing on static CVaR RL, which refers to the CVaR (i.e. the worst τ portion) of the accumulated total reward. Bastani et al. (2022) proved the first regret bound, and Wang et al. (2023) improved the results to be minimax optimal. Zhao et al. (2023) introduced function approximation to the MDP structure and studied static CVaR in low-rank MDPs. Additionally, Ni et al. (2024) developed the sample complexity of reward-free exploration in static CVaR. However, static CVaR is intrinsically different from the iterated CVaR RL studied in this paper. Iterated CVaR RL concerns the worst τ -percent of the reward-to-go at each step. Intuitively, static CVaR takes more cumulative reward into account and prefers actions that have better performance in general, while iterated CVaR prevents the agent from getting into catastrophic states (Du et al., 2022). Therefore, the algorithm designs and analysis techniques introduced above can not be applied to our problem.

Iterated CVaR RL: Hardy &amp; Wirch (2004) first introduced the Iterated CVaR and showed that it is a coherent risk measure. Osogami (2012), Chu &amp; Zhang (2014) and B¨ auerle &amp; Glauner (2022) studied the iterated coherent risk measures (iterated CVaR included) and proved the existence of a Markovian deterministic optimal policy. B¨ auerle &amp; Glauner (2022) also established a connection between iterated coherent risk measures and distributional robust MDPs. Chen et al. (2023) studied iterated CVaR with function approximation and human feedback.

For the more general iterated coherent risk measures

Table 1: Comparison of lower and upper bound for RL with Iterative Risk Measures. Some of the results are presented in terms of regret bound, and we converted them to sample complexity for the ease of comparison.

| Problem                                                                                                                | Lower bound                                                                                                 | Upper bound                                                                                                                                                                                |
|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Risk-neutral (Agarwal et al., 2020) Episodic, Iterated CVaR (Du et al., 2022) Episodic, Iterated OCE (Xu et al., 2023) | ˜ O ( SA (1 - γ ) 3 /epsilon1 2 ) ˜ O ( A /epsilon1 2 τ (1 - γ ) - 1 - 1 ) O ( SA (1 - γ ) 3 τ/epsilon1 2 ) | ˜ O ( SA (1 - γ ) 3 /epsilon1 2 ) ˜ O ( S 2 A (1 - γ ) 4 τ (1 - γ ) - 1 +1 /epsilon1 2 ) ( τ - 1 2 (1 - γ ) - 1 - 1 - (1 - γ ) - 1 ( τ - 1 / 2 - 1) ) 2 S 2 A (1 - √ τ ) 4 /epsilon1 2   |
| Infinite horizon, Iterated CVaR ( This work )                                                                          | ˜ O ( (1 - γτ ) SA (1 - γ ) 4 τ/epsilon1 2 )                                                                | ˜ O ( SA (1 - γ ) 4 τ 2 /epsilon1 2 ) ˜ O ( SA (1 - γ ) 3 /epsilon1 2 ) if τ ≥ γ                                                                                                           |

(Markov coherent risk), Tamar et al. (2015) derived the policy gradient algorithm for both static and dynamic (iterated) coherent risk measures. Huang et al. (2021) then proved that gradient dominant doesn't hold for iterated coherent risk measures and that stationary point is not guaranteed to be globally optimal.

To the best of our knowledge, Du et al. (2022) is the most related work to ours, where the Iterated CVaR objective in the episodic setting was studied. Their sample complexity results are listed in Table 1. In this paper, we obtain tighter upper and lower bounds, which are polynomial in the effective horizon (1 -γ ) -1 , and have a better dependence on the number of states S . Moreover, our bounds are minimax optimal for almost all choices of risk level. Xu et al. (2023) studied the recursive optimized certainty equivalent (OCE) problem in an episodic setting, where OCE is a more generalized risk measure, including CVaR. For Iterated CVaR RL, the upper and lower bounds in their paper are listed in Table 1. Compared to their results, our upper and lower bounds are much tighter (minimax optimal for almost all choices of risk level) and have a clear and easy-to-understand dependence on relevant factors.

Sample Complexity of Distributionally Robust RL with a Generative Model: The Iterative CVar RL problem can be equivalently written as a distributionally robust RL problem with a certain uncertainty set. The fundamental sample complexity for distributionally robust RL with a generative model has been studied in the literature for uncertainty sets defined by e.g., total variation (Shi et al., 2023; Panaganti &amp; Kalathil, 2022; Yang et al., 2022), χ 2 -divergence (Shi et al., 2023; Panaganti &amp; Kalathil, 2022; Yang et al., 2022), Kullback-Leibler (KL) divergence (Shi &amp; Chi, 2024).

## 3 Preliminaries and Problem Formulation

Notations. We denote by ∆( S ) the probability simplex over a set S . In this work, we use the standard O ( · ) notation to hide universal constant factors and use ˜ O ( · ) to hide logarithmic factors.

Conditional Value-at-Risk (CVaR). We begin by introducing two risk measures: value-at-risk (VaR) and conditional value-at-risk (CVaR). Let Z be a random variable with cumulative distribution function F Z ( z ) = P ( Z ≤ z ). The Value-at-risk at risk level τ ∈ (0 , 1] is defined as

<!-- formula-not-decoded -->

The conditional value-at-risk at risk level τ ∈ (0 , 1] is defined as where ( x ) + = max { x, 0 } for some x ∈ R . If F Z ( z ) is continuous at VaR τ ( Z ), then CVaR can also be equivalently written as (Shapiro et al., 2021):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From the above equation, CVaR can be viewed as the average of the worst τ -fraction of Z . When τ = 1, CVaR τ ( Z ) = E [ Z ], and when τ → 0, CVaR τ ( Z ) → ess inf( Z ).

When we need to specify the distribution of random variable Z , we write CVaR( Z ) as

<!-- formula-not-decoded -->

where P is the distribution of Z .

CVaR can be equivalently written in the dual formulation (Shapiro et al., 2021) using the risk envelope

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Standard Markov Decision Process. We denote an infinite horizon discounted Markov decision process (MDP) by a tuple M = ( S , A , γ, P, r ), where S and A are finite state and action spaces, γ ∈ [0 , 1) is the discount factor, P : S×A→ ∆( S ) denotes the transition kernel that maps a state-action pair to a probability distribution over S , and r : S × A → [0 , 1] is the deterministic reward function. Let S and A denote the sizes of the state and action spaces, respectively. A stationary policy is defined by π : S → ∆( A ). The value function of a policy π for state s is defined by

<!-- formula-not-decoded -->

where s t and a t denote the state and action at step t .

Iterated CVaR Objective. The risk-neutral objective defined in (7) fails to consider the risks arising from the stochastic nature of state transitions and the agent's policy decisions. Iterated coherent risk measures (Chu &amp; Zhang, 2014) have been introduced to model and evaluate these types of risks. For notational convenience, let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ρ s,a is a one-step coherent risk measure indexed by ( s, a ) ∈ S ×A and the distribution of s ′ follows the transition probability of P ( ·| s, a ). The objective of the risk-sensitive MDP is defined as follows:

<!-- formula-not-decoded -->

where V π ( s 0 ) = r ( s 0 , π ) + γρ s 0 ,π ( r ( s 1 , π ) + γρ s 1 ,π ( r ( s 2 , π ) + ... )). The trajectory { s 0 , s 1 , s 2 , ... } is generated from the MDP M and policy π . The objective V π is defined in a nested pattern rather than through a single static measure of the total discounted reward.

In this paper, we focus on a specific risk measure, the Conditional Value-at-Risk (CVaR), and refer to the objective in (10) as the iterated CVaR objective. For notational convenience, we denote

<!-- formula-not-decoded -->

The value function and Q-function are defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The choice of the risk-sensitive objective function in (12) guarantees the existence of an optimal policy and the optimal policy is Markovian (Chu &amp; Zhang, 2014). In contrast, the static CVaR objective, which only applies the risk measure to the total discounted reward once, does not have this property.

Optimal Risk-sensitive Policy and Bellman Operator. As shown in Chu &amp; Zhang (2014), there exists a deterministic stationary optimal policy π ∗ that maximizes the risk-sensitive value function simultaneously for all states:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The corresponding Bellman (optimality) equations are as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where s ′ ∼ P ( ·| s, a ). The Bellman operator is denoted by T τ : R SA → R SA and defined as follows:

<!-- formula-not-decoded -->

Since Q ∗ ( s, a ) is the unique fixed point of T τ , we can recover the optimal policy using a value iteration algorithm (see Algorithm 1). This converges rapidly due to the γ -contraction property of the T τ operator w.r.t. the l ∞ norm (Lemma 1).

Connection to Distributional Robust RL. Applying the dual form of CVaR, the Bellman equation can be re-written as

<!-- formula-not-decoded -->

This has the same form as the Bellman equation for distributional robust RL (Iyengar, 2005; Nilim &amp; El Ghaoui, 2004), and since E P [ ξ ] = 1: Pξ ∈ ∆( S ) is indeed a transition kernel. The uncertainty set of the transition kernel can be defined as follows:

<!-- formula-not-decoded -->

and the uncertainty set satisfied the ( s, a )-rectangularity (Iyengar, 2005). We can define the robust Bellman operator for CVaR as:

<!-- formula-not-decoded -->

Generative Model. Assume we have access to a generative model or a simulator, which can provide samples s ′ ∼ P ( ·| s, a ), for any ( s, a ). Suppose we call our generative model N times for each state-action pair. Let P be the empirical model defined as follows

<!-- formula-not-decoded -->

The total sample size is then NSA .

Goal. Given the collected samples, the goal is to learn the risk-sensitive optimal policy under risk level τ using as few samples as possible. Specifically, given a target accuracy tolerance /epsilon1 &gt; 0, the goal is to find an /epsilon1 -optimal risk-sensitive policy π s.t.

<!-- formula-not-decoded -->

## 4 Algorithm

In this section, we present a model-based approach, which first constructs an empirical nominal transition kernel based on the collected samples and then applies a value iteration-based algorithm ICVaR-VI to compute an optimal risk-sensitive policy in the approximated MDP.

Empirical Nominal Transition Kernel. The empirical nominal transition kernel ̂ P can be constructed as follows: ∀ ( s, a ) ∈ S × A

We define ̂ M to be the empirical MDP that is identical to the original M, except that it uses ̂ P instead of P for the transition kernel. We use ̂ V π and ̂ Q π to denote the value and action value functions of a policy π in ̂ M . And ̂ π ∗ , ̂ Q ∗ and ̂ V ∗ are the optimal policy and value functions in M .

<!-- formula-not-decoded -->

̂ Equipped with ̂ P 0 , we can define the empirical Bellman operator T τ as follows: ∀ ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

ICVaR-VI: Iterated CVaR Value Iteration. To find the fixed point of T τ , we introduce iterated CVaR

value iteration (ICVAR-VI)(Ruszczy´ nski, 2010), which is shown in Algorithm 1. The update rule can be written as:

<!-- formula-not-decoded -->

## ̂ where ̂ V t -1 = max a ̂ Q t -1 ( s, a ) for all s ∈ S . Algorithm 1 ICVaR-VI

Input: Empirical nominal transition kernel ̂ P ; reward function r ; risk level τ ; number of iterations T . 1: Initialization: ̂ Q 0 ( s, a ) = 0, ̂ V 0 ( s ) = 0 for all ( s, a ) ∈ S × A . 2: for t = 1 , 2 , . . . , T do 3: for s ∈ S , a ∈ A do 4: Update ̂ Q t ( s, a ) according to (24); 5: end for 6: for s ∈ S do 7: Set ̂ V t ( s ) = max a ̂ Q t ( s, a ); 8: end for 9: end for 10: Output: ̂ Q T , ̂ V T , and policy ̂ π ( s ) = arg max a ̂ Q T ( s, a ).

## 5 Theoretical Results

In this section, we present our main theoretical results. We start with the upper bound on the sample complexity for Iterated CVaR RL.

Theorem 1 (Sample Complexity Upper Bound) . For any risk level τ ∈ (0 , 1] , the number of samples needed by Algorithm ICVaR-VI to return an /epsilon1 -optimal policy with probability at least 1 -δ is at most ˜ O ( SA τ 2 (1 -γ ) 4 /epsilon1 2 ) . In addition, when τ ≥ γ , the sample complexity can be further improved to ˜ SA (1 γ ) 3 /epsilon1 2 .

Remark 1. In Theorem 1, the dependencies on S and A match with the result of Risk-neutral RL (Gheshlaghi Azar et al., 2013). Furthermore, our sample complexity matches the dependency on τ and 1 -γ with the result in Du et al. (2022) 1 , and our results improve the dependency on S by a factor of S (Du et al., 2022).

O ( -)

The sample complexity upper bound holds for any risk level τ ∈ (0 , 1]. In the special case when τ = 1,

1 In the finite-horizon episodic setting, the horizon H is analogous to (1 -γ ) -1 in the infinite-horizon setting. Du et al. (2022) focuses on the stationary finite-horizon episodic setting, where the transition kernel at each time step is the same, and therefore we incorporate an extra H into the sample complexity for a fair comparison with our results.

CVaR reduces to expectation, and Iterated CVaR RL reduces to risk-neutral RL. In this case, our sample complexity bound is ˜ O ( SA (1 -γ ) 3 /epsilon1 2 ) , which matches the result of the state-of-the-art sample complexity bound for standard risk-neutral RL (Agarwal et al., 2020; Gheshlaghi Azar et al., 2013). Furthermore, for an arbitrary constant risk level 0 &lt; τ &lt; γ , the sample complexity is increased by a factor of (1 -γ ) -1 , which, later in our lower bound analysis, is proved to be necessary.

In standard risk-neutral RL, the sample complexity of (1 -γ ) -3 can be obtained using Bernstein inequality in combination with the Bellman property of a policy's variance (Lemma 4 of Agarwal et al. (2020)), which could improve the sample complexity by a factor of (1 -γ ) -1 comparing to only using Hoeffding's inequality. However, using Bernstein inequality may not always improve the sample complexity for Iterated CVaR RL by a factor of (1 -γ ) -1 . In distributional robust RL (which is similar to our case since the dual form of Iterated CVaR can also be written as a distributionally robust optimization form), there is an extra term because the worst-case transition kernel is different from the nominal one, and whether Bernstein is superior to Hoeffding's inequality depends on the specific uncertainty set (Shi et al., 2023; Shi &amp; Chi, 2024;Panaganti &amp; Kalathil, 2022). In our setting, when τ &lt; γ , applying Bernstein's inequality to the uncertainty set for CVaR leads to the same sample complexity bound as Hoeffding's inequality. This phenomenon also appears in existing sample complexity analyses for χ 2 distributionally robust RL (Shi &amp; Chi, 2024). Still, for total-variation distance defined robust RL, Bernstein's inequality can improve the upper bound when τ ≥ γ , the total-variation distance of our uncertainty set is upper bounded by 1 -τ τ , and Bernstein's inequality can reduce the sample complexity by a factor of (1 -γ ) -1 .

The analysis of Iterated CVaR RL presents the following challenge: The objective is not the expected cumulative reward but rather the Iterated CVaR risk measure, which prevents us from decomposing the error as we do in risk-neutral cases. To address this challenge, we will establish a connection between Iterated CVaR RL and distributionally robust RL. This connection enables us to treat the risk-sensitive objective as the expected cumulative reward under the worstcase transition kernel. We then quantify the deviation between the empirical model and the true underlying model, using Hoeffding's inequality to derive bounds for τ &lt; γ . When τ ≥ γ , we introduce an alternative analytical approach using Bernstein inequality to further tighten the sample complexity bounds.

Below we present a proof sketch for Theorem 1 to high- light our major technical contributions (also see Appendix B for a complete proof).

Proof sketch of Theorem 1. With the connection between Iterated CVaR and distributionally robust RL, we can decompose the error in the following way. Let ̂ π ∗ denote the optimal risk-sensitive policy in the empirical model ̂ M , and let ̂ π represent the policy from Algorithm 1. Additionally, ̂ V represents the Iterated CVaR value function in the empirical model.

<!-- formula-not-decoded -->

The sub-optimality gap between π ∗ and ̂ π can be decomposed as where (i) holds by the optimality of ̂ π in ̂ M and γ -contraction property of risk-sensitive Bellman operator (Lemma 1). 1 ∈ R S is the all 1 vector.

<!-- formula-not-decoded -->

To bound || ̂ V π ∗ -V π ∗ || ∞ and || ̂ V ̂ π -V ̂ π || ∞ , we introduce a key inequality:

Applying the concentration lemma for CVaR (Lemma 3), we get that

<!-- formula-not-decoded -->

L stands for the log term of S , A , N and 1 δ . c 0 is a large enough constant. Finally, for a small enough /epsilon1 opt , we obtain the sample complexity upper bound.

For the special case when τ ≥ γ , we first introduce the total-variation distance bound for the CVaR uncertainty set U τ :

<!-- formula-not-decoded -->

This bound is useful when τ ≥ 1 2 . When the τ ≥ γ , the dependence on (1 -γ ) -1 of the extra term induced by worst-case transition mismatch ( C 2 in Shi et al. (2023)) is reduced when applying Berstein inequality:

<!-- formula-not-decoded -->

where Var P s,a ( V ) is the variance of V respect to distribution P s,a . In this case, we can reduce the order of (1 -γ ) -1 , which leads to the following bound:

where c 1 is some universal constant.

<!-- formula-not-decoded -->

The analysis of the above two cases concludes the proof of the sample complexity upper bound to get an /epsilon1 -optimal policy.

Additionally, in order to assess the tightness of Theorem 1, we further develop a minimax lower bound as follows, with the proof provided in Appendix C.2.

Theorem 2 (Sample Complexity Lower Bound) . Fix any τ ∈ ( 0 , 1 ] , γ ∈ ( 1 2 , 1 ) , there exist an MDP s.t. for any algorithm to obtain an /epsilon1 -optimal policy, the sample complexity is at least ˜ O ( (1 -γτ ) SA τ (1 -γ ) 4 /epsilon1 2 ) . In addition, when τ ≥ γ , the sample complexity of any algorithm is at least ˜ O ( SA (1 -γ ) 3 /epsilon1 2 ) .

Remark 2. Theorem 2 shows that when τ is small, there exists an MDP for which the sample complexity becomes unavoidably large. When the risk tolerance is low-indicating that the agent is highly sensitive to adverse states-more samples are needed for each stateaction pair to gather more accurate information about the environment, allowing the agent to develop a safer policy.

Lower Bound Analysis. In this following, we outline the proof idea for the lower bound in Theorem 2, with the full proof deferred to Appendix C.2. Our proof is inspired by the lower bound construction in distributional robust RL (Shi et al., 2023). We first construct two similar MDPs with close transition kernels that are hard to distinguish. For each MDP, there is an unknown optimal action. If an algorithm is capable of achieving an /epsilon1 -optimal policy in the Iterated CVaR RL problem, it must also be able to identify the optimal action and determine which MDP it is interacting with, with high probability. The challenge then becomes determining how many samples are needed to distinguish between two distributions. One notable difference in our problem is that, in our case, the two transition probabilities to the rewarding state are close to 1 -τ , rather than 1 -γ as in standard and distributional robust RL. Since CVaR computes the average over the worst τ -quantile of the reward-to-go, the transition probability to lower reward states must be smaller than τ for the worst-case transition kernel to differ. As a result, the final lower bound has a higher order in 1 -γ compared with risk-neutral settings when τ is relatively small.

Recall that the sufficient and necessary sample complexity for learning a standard risk-neutral MDP is ˜ O ( SA (1 -γ ) 3 /epsilon1 2 ) (Gheshlaghi Azar et al., 2013; Agarwal et al., 2020). Intuitively, the sample complexity for CVaR should include an additional 1 τ factor compared to the risk-neutral case, since CVaR only considers the worst τ -portion of outcomes and takes an average. Therefore, the number of samples needed should be ˜ O ( SA τ (1 -γ ) 3 /epsilon1 2 ) . This is true for static CVaR RL where Wang et al. (2023) provided a regret lower bound has an extra √ τ -1 term comparing to riskneutral regret lower bound ( √ τ -1 in regret is equivalent to τ -1 in PAC condition). But for Iterated CVaR RL, it is not merely a matter of averaging the worst τ -protion of trajectories. Du et al. (2022) in Section C.2 provides a detailed discussion of the differences between static and iterated CVaR in the episodic setting.

When τ ≥ γ , our sample complexity lower bound becomes ˜ O ( SA (1 -γ ) 3 /epsilon1 2 ) which matches the minimax optimal sample complexity for the risk-neutral case, in general, Iterated CVaR RL is harder to learn than standard RL. However, when τ is large, then the problem becomes closer to a risk-neutral one.

Nearly Tight Sample Complexity. By combining the upper bound from Theorem 1 with the minimax lower bound from Theorem 2, we confirm that the sample complexity is nearly optimal:

- When τ ∈ (0 , 1) is a constant independent of γ , our sample complexity upper bound ˜ O ( SA (1 -γ ) 4 /epsilon1 2 ) is tight and matches the minimax lower bound;
- When τ ≥ γ , our sample complexity upper bound is ˜ O ( SA (1 -γ ) 3 /epsilon1 2 ) , and it matches with the minimax lower bound;
- When τ ≤ 1 -γ , our sample complexity ˜ O ( SA τ 2 (1 -γ ) 4 /epsilon1 2 ) has a gap of 1 τ ≥ 1 1 -γ compared to the minimax lower bound. This case of small risk level τ will be further discussed in the next section.

## 6 Worst-Path RL

In this section, we investigate the problem with a fixed MDP and consider a limiting case where the risk level is small. This problem is referred to as worst-path RL (Du et al., 2022).

Specifically, consider an MDP, and denote by p min the minimum non-zero reaching probability from any state-action pair: ∀ ( s, a ) ∈ S × A , and ∀ s ′ ∈ supp( P ( ·| s, a )) , P ( s ′ | s, a ) ≥ p min . Consider small risk

level τ : τ ≤ p min . Here we use supp( P ) to denote the support of a distribution P .

This case is not covered by results in Theorems 1 and 2. Obviously, the sample complexity in Theorems 1 and 2 depends on 1 τ and goes to infinity as τ → 0. However, as will be shown later, in the case with τ ≤ p min , a sample complexity of ˜ O ( SA p min ) is minimax optimal, and it does not depend on /epsilon1 and 1 -γ .

Taking the minimax lower bound result in Theorem 2 as an example to explain the difference. To prove the minimax lower bound in Theorem 2, for any risk level τ , we construct two MDPs with transition probabilities to lower-reward states smaller than the risk level τ . However, such hard examples do not satisfy the problem setting in this section: τ ≤ p min .

As we have mentioned earlier, as τ → 0, CVaR τ ( Z ) → ess inf Z . In Du et al. (2022), regret bounds independent of the number of episodes were developed, where the bounds also depend on visitation probability to the worst state. Given access to a generative model, the problem is simpler since there is no need to explore. Then the transition to the worst states is simply the frequency of that state in the N samples generated. When τ is smaller than all possible non-zero transition probability, CVaR τ simply reduces to finding the worst state for the reward-to-go.

Bellman operator and Bellman equations . Since τ ≤ p min , the objective reduces to maximizing the accumulative reward along the worst-case trajectory. The Bellman equations can then be written as follows:

<!-- formula-not-decoded -->

where min s ′ ∈ supp( P ( ·| s,a )) considers the worst-case of all possible next states (with non-zero probability). The algorithm for this problem is similar to Algorithm 1 with a slightly different Bellman operator:

<!-- formula-not-decoded -->

where n ( s ′ , s, a ) denotes the number of samples with next state s ′ in the total N generated samples for state action pair ( s, a ). With that in mind, we substitute (32) to (23) in Algorithm 1, and we have the algorithm for the worst-path RL problem in this section.

Sample Complexity Analysis . Below we provide the sample complexity upper bound for the algorithm discussed above for the worst-path RL problem.

Theorem 3 (Worst-Path RL Upper Bound) . Consider a risk level τ ≤ p min . With probability at least 1 -δ , the number of samples needed to obtain an optimal policy is at most

<!-- formula-not-decoded -->

Remark 3. For any τ &lt; p min , CVaR risk measure reduces to the essential infimum, and therefore, the bound does not depend on τ anymore. More importantly, the upper bound now depends on 1 p min , which is strictly smaller than 1 τ 2 .

The key idea in the proof is to analyze the suboptimality gap using the occurrence of the worst-case state. If the worst-case state occurs with high probability, the Bellman operator behaves as it would under the true underlying model. Otherwise, the suboptimality gap becomes non-vanishing. This also explains why the sample complexity does not depend on /epsilon1 and 1 1 -γ here. This result aligns with our intuition: if we expect a state with probability p min to occur, we need 1 p min samples for each state-action pair.

To validate the optimality of our sample complexity upper bound, we also provide a minimax lower bound.

Theorem 4 (Worst-Path RL Lower Bound) . For a given risk level τ , there exists an MDP with P min ≥ τ such that for any algorithm to obtain an optimal policy at a risk level τ ≤ p min , the sample complexity is at least ˜ O ( SA p min ) .

The minimax lower bound matches with the upper bound in Theorem 3.

## 7 Acknowledgements

The work of Z. Deng and S. Zou was partially supported by NSF under Grants CCF-2438429 and ECCS2438392. This work is also partially supported by the AFRL VFRP program.

## 8 Conclusion

In this paper, we investigate Iterated CVaR RL problem in infinite horizon discounted MDP with access to a generative model. We introduce the algorithm ICVaR-VI and provide nearly matching sample complexity upper and lower bounds. Later we study the limit case with an arbitrarily small risk level τ , and provide tight upper and lower bounds. There are several interesting directions for future work, e.g., further closing the gap between the upper and lower sample complexity bounds and extending iterated CVaR to

other types of coherent risk measure or general Markov coherent risk.

## References

- Alekh Agarwal, Sham Kakade, and Lin F. Yang. Model-based reinforcement learning with a generative model is minimax optimal. In Proc. Annual Conference on Learning Theory (CoLT) , volume 125, pp. 67-83, 2020.
- Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research , 22(98):1-76, 2021.
- Philippe Artzner, Freddy Delbaen, Jean-Marc Eber, and David Heath. Coherent measures of risk. Mathematical finance , 9(3):203-228, 1999.
- Osbert Bastani, Jason Yecheng Ma, Estelle Shen, and Wanqiao Xu. Regret bounds for risk-sensitive reinforcement learning. Advances in Neural Information Processing Systems , 35:36259-36269, 2022.
- Nicole B¨ auerle and Alexander Glauner. Markov decision processes with recursive risk measures. European Journal of Operational Research , 296(3):953966, 2022.
- Nicole B¨ auerle and Jonathan Ott. Markov decision processes with Average-Value-at-Risk criteria. Mathematical Methods of Operations Research , 74: 361-379, 2011.
- Jalaj Bhandari and Daniel Russo. Global optimality guarantees for policy gradient methods. Operations Research , 2024.
- Jalaj Bhandari, Daniel Russo, and Raghav Singal. A finite time analysis of temporal difference learning with linear function approximation. In Proc. Annual Conference on Learning Theory (CoLT) , pp. 16911692. PMLR, 2018.
- Ga¨ elle Chagny. An introduction to nonparametric adaptive estimation. The Graduate Journal of Mathematics , 2016(2):105-120, 2016.
- Arthur Charpentier, Romuald Elie, and Carl Remlinger. Reinforcement learning in economics and finance. Computational Economics , pp. 1-38, 2021.
- Yu Chen, Yihan Du, Pihe Hu, Siwei Wang, Desheng Wu, and Longbo Huang. Provably efficient iterated CVaR reinforcement learning with function approximation. arXiv preprint arXiv:2307.02842 , 2023.
- Shanyun Chu and Yi Zhang. Markov decision processes with iterated coherent risk measures. International Journal of Control , 87(11):2286-2293, 2014.
- Christoph Dann, Tor Lattimore, and Emma Brunskill. Unifying pac and regret: Uniform pac bounds for episodic reinforcement learning. In Proc. Advances in Neural Information Processing Systems (NeurIPS) , volume 30, 2017.

- Yihan Du, Siwei Wang, and Longbo Huang. Provably efficient risk-sensitive reinforcement learning: Iterated CVaR and worst path. arXiv preprint arXiv:2206.02678 , 2022.
- Mohammad Gheshlaghi Azar, R´ emi Munos, and Hilbert J Kappen. Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine Learning , 91:325-349, 2013.
- Mary R Hardy and Julia L Wirch. The iterated CTE: a dynamic risk measure. North American Actuarial Journal , 8(4):62-75, 2004.
- Audrey Huang, Liu Leqi, Zachary C Lipton, and Kamyar Azizzadenesheli. On the convergence and optimality of policy gradient for markov coherent risk. arXiv preprint arXiv:2103.02827 , 2021.
- Garud N Iyengar. Robust dynamic programming. Mathematics of Operations Research , 30(2):257-280, 2005.
- Michael Kearns and Satinder Singh. Finite-Sample Convergence Rates for Q-Learning and Indirect Algorithms. In Proc. Advances in Neural Information Processing Systems (NeurIPS) , volume 11, 1998.
- Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research , 32(11):12381274, 2013.
- Xinyi Ni, Guanlin Liu, and Lifeng Lai. Risk-sensitive reward-free reinforcement learning with CVaR. In Proc. International Conference on Machine Learning (ICML) , volume 235, pp. 37999-38017, 2024.
- Arnab Nilim and Laurent El Ghaoui. Robustness in Markov decision problems with uncertain transition matrices. In Proc. Advances in Neural Information Processing Systems (NIPS) , pp. 839-846, 2004.
- Takayuki Osogami. Iterated risk measures for risksensitive markov decision processes with discounted cost. arXiv preprint arXiv:1202.3755 , 2012.
- Long Ouyang et al. Training language models to follow instructions with human feedback. In Proc. Advances in Neural Information Processing Systems (NeurIPS) , volume 35, pp. 27730-27744, 2022.
- Kishan Panaganti and Dileep Kalathil. Sample complexity of robust reinforcement learning with a generative model. In Proc. International Conference on Artifical Intelligence and Statistics (AISTATS) , pp. 9582-9602, 2022.
- R Tyrrell Rockafellar, Stanislav Uryasev, et al. Optimization of conditional value-at-risk. Journal of risk , 2:21-42, 2000.
- Andrzej Ruszczy´ nski. Risk-averse dynamic programming for Markov decision processes. Mathematical Programming , 125:235-261, 2010.
- Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczynski. Lectures on stochastic programming: modeling and theory . SIAM, 2021.
- Laixi Shi and Yuejie Chi. Distributionally robust model-based offline reinforcement learning with near-optimal sample complexity. Journal of Machine Learning Research , 25(200):1-91, 2024.
- Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, Matthieu Geist, and Yuejie Chi. The curious price of distributional robustness in reinforcement learning with a generative model. In Proc. Advances in Neural Information Processing Systems (NeurIPS) , 2023.
- R. Srikant and Lei Ying. Finite-time error bounds for linear stochastic approximation and TD learning. In Proc. Annual Conference on Learning Theory (CoLT) , pp. 2803-2830, 2019.
- Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction . The MIT Press, Cambridge, Massachusetts, 2018.
- Aviv Tamar, Yinlam Chow, Mohammad Ghavamzadeh, and Shie Mannor. Policy gradient for coherent risk measures. Advances in neural information processing systems , 28, 2015.
- Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science , volume 47. Cambridge University Press, 2018.
- Kaiwen Wang, Nathan Kallus, and Wen Sun. Nearminimax-optimal risk-sensitive reinforcement learning with CVaR. In Proc. International Conference on Machine Learning (ICML) , pp. 35864-35907, 2023.
- Wenhao Xu, Xuefeng Gao, and Xuedong He. Regret bounds for Markov decision processes with recursive Optimized Certainty Equivalents. In Proc. International Conference on Machine Learning (ICML) , pp. 38400-38427, 2023.
- Wenhao Yang, Liangyu Zhang, and Zhihua Zhang. Toward theoretical understandings of robust Markov decision processes: Sample complexity and asymptotics. The Annals of Statistics , 50(6):3223-3248, 2022.
- Yulai Zhao, Wenhao Zhan, Xiaoyan Hu, Ho-fung Leung, Farzan Farnia, Wen Sun, and Jason D Lee. Provably efficient CVaR RL in low-rank MDPs. arXiv preprint arXiv:2311.11965 , 2023.
- Shaofeng Zou, Tengyu Xu, and Yingbin Liang. Finitesample analysis for SARSA with linear function approximation. In Proc. Advances in Neural Informa-

tion Processing Systems (NeurIPS) , pp. 8665-8675, 2019.

## Checklist

1. For all models and algorithms presented, check if you include:
2. (a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. [Yes]
3. (b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. [Yes]
4. (c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. [Not Applicable]
2. For any theoretical claim, check if you include:
6. (a) Statements of the full set of assumptions of all theoretical results. [Yes]
7. (b) Complete proofs of all theoretical results. [Yes]
8. (c) Clear explanations of any assumptions. [Yes]
3. For all figures and tables that present empirical results, check if you include:
10. (a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). [Not Applicable]
11. (b) All the training details (e.g., data splits, hyperparameters, how they were chosen). [Not Applicable]
12. (c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). [Not Applicable]
13. (d) Adescription of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). [Not Applicable]
4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:
15. (a) Citations of the creator If your work uses existing assets. [Yes]
16. (b) The license information of the assets, if applicable. [Not Applicable]
17. (c) New assets either in the supplemental material or as a URL, if applicable. [Not Applicable]
18. (d) Information about consent from data providers/curators. [Not Applicable]
19. (e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. [Not Applicable]

5. If you used crowdsourcing or conducted research with human subjects, check if you include:
2. (a) The full text of instructions given to participants and screenshots. [Not Applicable]
3. (b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. [Not Applicable]
4. (c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. [Not Applicable]

## A Notations and useful lemmas

In the appendices, with a slight abuse of notations, we use P ∈ R SA × S to denote the transition matrix of the nominal transition kernel P , and let P s,a denote its ( s, a )-th row. Similarly, we could define the transition matrix ̂ P for the empirical nominal transition kernel ̂ P . We further define the following matrix/vector notations for the convenience of presentation. Let the state space S = { 0 , 1 , 2 , . . . , S -1 } and action space A = { 0 , 1 , 2 , . . . , A -1 } . A deterministic policy π is a mapping from the state space to the action space, i.e., π ( s ) is an action in A .

- r ∈ R SA : vector form of the reward function r .
- Π π ∈ { 0 , 1 } S × SA : projection matrix associated with a deterministic policy π :

where e /latticetop π (0) , e /latticetop π (1) , . . . , e /latticetop π ( S -1) ∈ R A are standard basis vectors and 0 ∈ R S is all zero vector.

<!-- formula-not-decoded -->

- r π ∈ R S : reward vector restricted to the actions chosen by the deterministic policy π , namely, r π ( s ) = r ( s, π ( s )) for all s ∈ S (or simply, r π = Π π r ).
- P V ∈ R SA × S , ̂ P V ∈ R SA × S : worst-case transition matrices for a vector V ∈ R S . We denote P V s,a (resp. ̂ P V s,a ) as the ( s, a )-th row. Specifically,

Furthermore, we make use of the following short-hand notation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The corresponding probability transition matrices are denoted by P π,V ∈ R SA × S , P π, ̂ V ∈ R SA × S , ̂ P π,V ∈ R SA × S and P π, ̂ V ∈ R SA × S , respectively.

<!-- formula-not-decoded -->

- ̂ · P π ∈ R S × S , ̂ P π ∈ R S × S , P π,V ∈ R S × S , P π, ̂ V ∈ R S × S , ̂ P π,V ∈ R S × S and ̂ P π, ̂ V ∈ R S × S : probability transition matrices w.r.t. policy π over the states:
- Var ¯ P ( V ) ∈ R SA : for any transition kernel ¯ P ∈ R SA × S and any vector V ∈ R S , the ( s, a )-th row of Var P ( V ) is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Var P s,a ( V ) := P s,a ( V 2 ) -( P s,a V ) 2 .

Lemma 1. (Ruszczy´ nski, 2010, Lemma 2). For any γ ∈ [0 , 1) , the robust Bellman operator T τ ( · ) is a γ -contraction w.r.t. ‖ · ‖ ∞ . Namely, for any Q 1 , Q 2 ∈ R SA s.t. Q 1 ( s, a ) , Q 2 ( s, a ) ∈ [0 , 1 1 -γ ] for all ( s, a ) ∈ S × A , one has

<!-- formula-not-decoded -->

Lemma 2 (Shi et al., 2023, Lemma 5) . Let ̂ Q 0 = 0 . The iterates { ̂ Q t } , { ̂ V t } of ICVaR-VI (Algorithm 1) obey

Furthermore, the output policy π obeys

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B Proof of Theorem 1: Sample Complexity Upper Bound

Step 1: Decomposing the error. The optimality gap can be decomposed as

<!-- formula-not-decoded -->

where (i) holds by the fact that ̂ π ∗ is the optimal policy under transition kernel ̂ P ; and (ii) follows from Lemma 2. The first and third terms in the sub-optimality gap in (39) can be bounded in the same way as follows

<!-- formula-not-decoded -->

where (i) holds by the fact that corresponds to the worst-case transition kernel in U τ ( P π ) for V π . By decomposing the error in a symmetric way, we can similarly obtain that

<!-- formula-not-decoded -->

Combining (40) and (41), we arrive at

<!-- formula-not-decoded -->

Step 2: Controlling || ̂ V π -V π || ∞ and || ̂ V ̂ π -V ̂ π || ∞ in (39). Lemma 3. For any δ ∈ (0 , 1) , with probability at least 1 -δ , one has that

<!-- formula-not-decoded -->

Proof. Note that V π ( s ) ∈ [ 0 , 1 1 -γ ] for any s ∈ S . Therefore, sup x ∈ R is equivalent to sup x ∈ [0 , 1 1 -γ ] . We first show that

<!-- formula-not-decoded -->

We define the function g as the difference in the expectation of V between the two transition kernel for a fix x and ( s, a ),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using Hoeffding's inequality, one has that with probability at least 1 -δ ,

It can be easily shown that g s,a ( s, V ) is 1-Lipschitz in x for any V such that ‖ V ‖ ∞ ≤ 1 1 -γ . To obtain the union bound, we construct an /epsilon1 1 -net N /epsilon1 1 over [0 , 1 1 -γ ], with size |N /epsilon1 1 | ≤ 3 (1 -γ ) /epsilon1 1 (Vershynin, 2018). By the union bound, we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where (i) follows from that the optimal x falls into an /epsilon1 1 -ball centered around some point in N /epsilon1 1 and g s,a ( x, V ) is 1-Lipschitz in x ; (ii) stems from applying the results in (46) and the union bound over S , A , and N /epsilon1 1 ; and (iii) follows if we let /epsilon1 1 = √ 2 log(6 SAN/δ ) N (1 -γ ) 2 and then |N /epsilon1 1 | ≤ 3 /epsilon1 1 (1 -γ ) ≤ 3 N . Substituting (47) back into (44), we have that with probability at least 1 -δ ,

From (48), it can be shown that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes the proof of Lemma 3.

Substituting (49) back into (42) we get that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where (i) holds by ( I -γ ̂ P π,V ) -1 = ∑ ∞ t =0 γ t ( ̂ P π,V ) t ≥ 0, (ii) follows from

<!-- formula-not-decoded -->

Finally take /epsilon1 opt ≤ √ 2 log(6 SAN/δ ) τ (1 -γ ) √ N , and plug (50) back to (39). We then have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

## B.1 Tighter bound using Berstein inequality when τ ≥ γ

Note that the bound in (52) applied to any τ and γ . In this subsection, we consider the scenario where τ ≥ γ . We show that a tighter bound can be achieved.

Then, for a fixed x that is independent with P s,a , using Bernstein inequality, one has that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

To derive the union bound, we can similarly construct a /epsilon1 1 -net over [0 , 1 1 -γ ] with size |N /epsilon1 1 | ≤ 3 /epsilon1 1 (1 -γ ) . By the union bound and (53), it holds with probability at least 1 -δ SA for all x ∈ N /epsilon1 1 ,

<!-- formula-not-decoded -->

From (44), we can show that with probability at least 1 -δ SA , where (i) follows from that the optimal x falls into an /epsilon1 1 -ball centered around some point in N /epsilon1 1 and g s,a is 1-Lipschitz in x . (ii) stems from taking /epsilon1 1 = log(2 SA |N /epsilon1 1 | /δ ) 3 N (1 -γ ) ; and ( iii ) is shown by |N /epsilon1 1 | ≤ 3 /epsilon1 1 (1 -γ ) ≤ 9 N .

<!-- formula-not-decoded -->

To bound the term on the right-hand side of (42), for any policy π , we show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) holds by ( I -γ ̂ P π,V ) -1 = ∑ ∞ t =0 γ t ( ̂ P π,V ) t ≥ 0; and (ii) holds with high probability by the bound in (55) and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying Lemma 4, then C 1 in (56) can be bounded as follows

Lemma 5. Consider a CVaR uncertainty set U τ defined in (19), then it holds that

<!-- formula-not-decoded -->

Then we have for all ( s, a ) ∈ S × A , and ¯ P s,a ∈ U τ ( P s,a ):

and (i) holds when τ ≥ γ . We then have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) follows from (61); and (ii) follows from (51).

In the following, we then bound C 3 . The following lemma plays an important role.

<!-- formula-not-decoded -->

Lemma 6 (Panaganti &amp; Kalathil, 2022, Lemma 6) . Consider any δ ∈ (0 , 1) . For any fixed policy π and fixed value function vector V ∈ R S , one has that with probability at least 1 -δ ,

Plugging (63) into (56), we can show that with high probability

<!-- formula-not-decoded -->

Finally, plugging the results of C 1 in (59), C 2 in (62) and C 3 in (64) back into (56), we have

<!-- formula-not-decoded -->

where (i) holds when τ ≥ γ ≥ 1 2 .

Plugging (65) back into (42), we have that

<!-- formula-not-decoded -->

Summing up the results for ̂ π and π ∗ and plugging back to (39) complete the proof as follows: taking /epsilon1 opt ≤ log( 18 SAN δ ) γ (1 -γ ) N and N ≥ log( 18 SAN δ ) (1 -γ ) 2 , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where the last inequality holds if N ≥ log( 18 SAN δ ) (1 -γ ) 2 , with probability at least 1 -δ .

## C Proof of Theorem 2: Sample Complexity Lower Bound

## C.1 Construction of hard problem instances

Construction of two hard MDPs. Suppose there are two standard MDPs defined as below:

<!-- formula-not-decoded -->

Here, γ is the discount factor, S = { 0 , 1 , ..., S -1 } is the state space. Given any state s ∈ { 2 , 3 , ..., S -1 } , the coresponding action spaces are A = { 0 , 1 , 2 , ..., A -1 } . While for state s = 0 and s = 1, the action space is only A ′ = { 0 , 1 } . For any φ ∈ { 0 , 1 } , the transition kernel P φ of the constructed MDP M φ is defined as where p and q are set to satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some p and ∆ &gt; 0. The above transition kernel P φ implies that State 1 is an absorbing state.

Then we define the reward function as:

<!-- formula-not-decoded -->

Additionally, we choose the following initial state distribution:

<!-- formula-not-decoded -->

Uncertainty set of the transition kernel. Recall that the uncertainty set of CVaR is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We define p and ∆:

where c ∈ (0 , 1). Consequently, applying (69) directly leads to

<!-- formula-not-decoded -->

For any ( s, a ) ∈ S × A , we denote the smallest transition probability of moving to the next state s ′ ∈ { 0 , 1 } in the uncertainty set as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equation can be verified by the definition of U τ ( P ). We further define the following notation

<!-- formula-not-decoded -->

which follows from the fact that p ≥ q ≥ 1 -τ in (72).

Robust value function and optimal robust policies. For any MDP M φ with the above uncertainty set, we denote π ∗ φ as the optimal policy, and the robust value function of any policy π as V π φ . Then we introduce the following lemma, which describes some important properties of the robust value function:

Lemma 7. For any φ = { 0 , 1 } and any policy π , the value function satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The optimal value function and the optimal policy satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.2 Establishing the minimax lower bound

Note that our goal is to quantify the sub-optimality gap between the policy estimator ̂ π and the optimal policy π ∗ on the initial state distribution ϕ , which is

<!-- formula-not-decoded -->

Step 1: Equivalence to estimating φ . With /epsilon1 ≤ c 16(1 -γ ) , let ∆ = 16(1 -γ ) 2 τ/epsilon1 . Applying (75) and (77a) we where z π φ is defined as

have that

̂ where (i) holds by the definition of p and q ; (ii) follows from z ̂ π φ ≤ p ; (iii) follows from the definition of p ; (iv) follows from γ ≥ 1 2 .

<!-- formula-not-decoded -->

With this connection between the sub-optimality gap and the policy ̂ π ( φ | 0), if the policy ̂ π is /epsilon1 -optimal with a high probability, i.e., then, we need ̂ π ( φ | 0) ≥ 1 2 with probabily at least 1 -δ . With this in mind, we can construct an estimator ̂ φ for the better action φ

which satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The problem now becomes to produce a correct estimator ̂ φ with high probability. Subsequently, the goal is to demonstrate that (81) cannot hold without a sufficient number of samples.

<!-- formula-not-decoded -->

Step 2: Probability of error in testing two hypotheses. Equipped with the aforementioned ground- work, we can now delve into differentiating between the two hypotheses φ ∈ { 0 , 1 } . To achieve this, we consider the concept of minimax probability of error, defined as follows:

<!-- formula-not-decoded -->

/negationslash

/negationslash

Here, the infimum is taken over all possible tests Ψ constructed from the samples generated from the nominal transition kernel P φ .

Moving forward, let us denote µ φ (resp. µ φ (s)) as the distribution of a sample tuple ( s i , a i , s ′ i ) under the nominal transition kernel P φ associated with M φ and the samples are generated independently. Applying standard results from Chagny (2016) and the additivity of the KL divergence, we obtain

<!-- formula-not-decoded -->

where the last inequality holds by observing that

<!-- formula-not-decoded -->

Now our focus shifts to bound the KL divergence in (83). Applying Lemma 2.7 in Iyengar (2005) gives

<!-- formula-not-decoded -->

where (i) stems from 1 -p = τ -cτ (1 -γ ) = τ (1 -c (1 -γ )) ≥ γτ , and p (1 -p ) ≥ min { τ (1 -τ ) , γτ (1 -γτ ) } = γτ (1 -γτ ); and (ii) follows from γ ≥ 1 2 . Note that KL ( P 0 ( s ′ || 0 , 0) || P 1 ( s ′ || 0 , 0) ) can be bounded in the same procedure. Substitute (85) back in to (83), if the sample size is selected as then one necessarily has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 3: putting the results together. Combined all the results above, suppose there is a policy π such that

<!-- formula-not-decoded -->

According to the discussion in Step 1, the estimated φ must satisfy

/negationslash

̂ ̂ However, this cannot be satisfied with the sample size being too small (86). Thus we have completed the proof of the lower bound.

<!-- formula-not-decoded -->

## D Proof for Worst Path RL

## D.1 Proof of Theorem 3

For any ( s, a ) ∈ S × A , n ( s ′ , s, a ) denotes the number of times when the next state is s ′ . Using Lemma F.4 in Dann et al. (2017), we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When N ≥ 2 p min ( 1 + 2 log ( SA δ )) , for any ( s, a ) ∈ S × A and s ′ ∈ supp( P ( ·| s, a )), we can show that

/negationslash

where (i) is derived from P ( s ′ | s, a ) ≥ p min for s ′ ∈ supp( P ( ·| s, a )). Combining with (90), we have that

<!-- formula-not-decoded -->

We decompose the sub-optimality gap in the same manner as in (39). We notice that with probability at least 1 -δ , for all ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

Plugging (93) back into (42), we have

Then the first and the third term in (39) then disappears, and we get that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When N ≥ 2 p min ( 1 + 2 log ( SA δ )) , for any /epsilon1 &gt; 0, if we take /epsilon1 opt ≤ /epsilon1 (1 -γ ) 2 γ , then with probability at least 1 -δ , we have that

<!-- formula-not-decoded -->

which concludes the proof.

## D.2 Proof of Theorem 4

In the following, we establish a sample complexity lower bound for Worst Path RL. We first construct two hard MDP instances similar to those in Section C.2. Suppose there are two standard MDPs defined below

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The transition kernel is defined as

The reward function is defined as

<!-- formula-not-decoded -->

The initial state distribution is also the same

<!-- formula-not-decoded -->

Since state s = 1 is an absorbing state and has reward 1, the value function at state 1 for any policy π is V π (1) = 1 1 -γ . At state s ∈ { 2 , 3 , ..., S -1 } , applying the Bellman operator we have V π ( s ) = γ 1 -γ . At state s = 0, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we want a policy π to be /epsilon1 -optimal with high probability:

<!-- formula-not-decoded -->

then if /epsilon1 ≤ γ 2(1 -γ ) , we necessarilly need π ( φ | 0) ≥ 1 2 with probability at least 1 -δ . Following the same procedure in the proof of the lower bound for Iterated CVaR RL, we constructed the estimator ̂ φ for the better action (80) that satisfies (81). We also notice that if n (0 , 0 , a ) ≥ 1 in the N samples, we can definitely tell the other action is superior. Otherwise, we cannot tell the difference between the two actions, and in this case, the probability of a correct guess is 1 2 . With this in mind, we have

<!-- formula-not-decoded -->

Inserting (103) into 81 we have that which further implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The overall sample complexity lower bound is then ˜ O ( SA p min ) .

## E Proof of Lemmas

## E.1 Proof of Lemma 5

For any ¯ P s,a ∈ U τ ( P s,a ) = { ¯ P s,a ∈ ∆( S ) , 0 ≤ ¯ P s,a ( s ′ ) P s,a ( s ′ ) ≤ 1 τ } , the total-variation between ¯ P s,a and P s,a is defined as:

<!-- formula-not-decoded -->

Here S ′ is a subset of state space S and P s,a ( S ′ ) = ∑ s ′ ∈S ′ P s,a ( s ′ ), ¯ P s,a ( S ′ ) = ∑ s ′ ∈S ′ ¯ P s,a ( s ′ ). By the definition of the uncertainty set (19), for any S ′ ∈ S , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This further implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any S ′ ∈ S ,

Using the second definition of total variation, we obtain that

<!-- formula-not-decoded -->

## E.2 Proof of Lemma 7

For any M φ ∈ { 0 , 1 } , we can easily find the value function at state s = 1 or any states s ∈ { 2 , 3 , · · · , S -1 }

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) and (ii) are according to the fact that the transitions defined over states s ≥ 1 in (68) give only one possible next state 1, and by the definition of the uncertainty set in (19), there exists only one transition kernel within the uncertainty set, which is the kernel itself.

The value function at state s = 0 with policy π satisfies

<!-- formula-not-decoded -->

where (i) follows from (73a), (73b) and (74). Note that V π φ (0) is increasing in z π φ and z π φ is upper bounded by p and reaches the upper bound when π ( φ | 0) = 1. Taking z π φ = p in (113), we get the optimal value function

<!-- formula-not-decoded -->