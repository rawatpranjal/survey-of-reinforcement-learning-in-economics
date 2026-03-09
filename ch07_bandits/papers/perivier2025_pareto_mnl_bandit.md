## On (Approximate) Pareto Optimality for the Multinomial Logistic Bandit

## Jierui Zuo

Department of Management Science and Engineering

Tsinghua University Beijing, China zuojr22@mails.tsinghua.edu.cn

## Hanzhang Qin

Department of Industrial Systems Engineering and Management National University of Singapore Singapore hzqin@nus.edu.sg

## Abstract

We provide a new online learning algorithm for tackling the Multinomial Logit Bandit (MNL-Bandit) problem. Despite the challenges posed by the combinatorial nature of the MNL model, we develop a novel Upper Confidence Bound (UCB)-based method that achieves Approximate Pareto Optimality by balancing regret minimization and estimation error of the assortment revenues and the MNL parameters. We develop theoretical guarantees characterizing the tradeoff between regret and estimation error for the MNL-Bandit problem through informationtheoretic bounds, and propose a modified UCB algorithm that incorporates forced exploration to improve parameter estimation accuracy while maintaining low regret. Our analysis sheds critical insights into how to optimally balance the collected revenues and the treatment estimation in dynamic assortment optimization.

## 1 Introduction

The Multinomial Logit Bandit (MNL-Bandit) problem is a dynamic framework for assortment optimization, where the goal is to iteratively learn consumer preferences while maximizing cumulative revenues over a finite horizon. This problem, rooted in online decision-making, bridges the exploration-exploitation tradeoff by dynamically offering subsets of items (assortments) to consumers whose choices follow the multinomial logit (MNL) model. Among the parametric family of modeling customer choice, the MNL model is celebrated for its analytical tractability and practical relevance in modeling consumer substitution behavior, with applications spanning retail, online advertising, and recommendation systems.

In classical assortment optimization, consumer preference parameters are estimated a priori, and static assortments are then deployed to maximize expected revenue. However, in fast-changing environments such as online retail, platforms must both optimize short-run revenue and learn accurate preference parameters. In practice, large platforms (e.g., Amazon, Netflix) place high value on precise utility estimates-especially for new or slow-moving items-and routinely measure the average treatment effect (ATE) of product, service, or algorithmic changes. Even when experiments temporarily depress sales, firms may proceed for long-run gains: as Jeff Bezos observed, 'sometimes we measure things and see that in the short term they actually hurt sales, and we do it anyway' Scheidies [2012]; similarly, Facebook's 2018 News Feed change accepted short-term metric declines to improve overall

platform health Constine [2018]. The MNL-Bandit framework captures this, but balancing accurate learning with low regret remains challenging; we formalize this trade-off and provide provable algorithms.

Recent advancements in multi-armed bandit (MAB) literature emphasize the tradeoff between exploration for accurate inference and exploitation for low regret. While classical MAB algorithms such as Upper Confidence Bound (UCB) and Thompson Sampling excel in minimizing regret, they typically fail to adequately account for parameter estimation accuracy, especially in structured settings like the MNL model. This underscores the need for a unified approach that achieves Pareto Optimality-a state where neither regret nor parameter estimation accuracy can be improved without compromising the other.

The concept of Pareto Optimality is increasingly recognized as a critical design principle in bandit frameworks involving multiple objectives. Pareto Optimal policies aim to operate on the Pareto frontier, where any improvement in one objective (e.g., lower regret) necessitates a tradeoff in the other (e.g., higher estimation error). This paradigm has been formalized in recent studies as a multi-objective optimization framework, providing theoretical and algorithmic insights into designing adaptive policies.

For the MNL-Bandit problem, the Pareto frontier is defined as the set of policies that optimally balance the regret of offering suboptimal assortments and the estimation error in learning the MNL parameters. Despite its relevance, achieving Pareto Optimality in the MNL-Bandit setting remains a significant challenge due to the non-linear and combinatorial nature of the MNL model.

This paper introduces a novel UCB-based algorithm tailored to the MNL-Bandit problem, which provably achieves Approximate Pareto Optimality. Our contributions can be summarized as follows:

- We establish theoretical guarantees for policies operating on the Approximate Pareto frontier of the MNL-Bandit problem. Specifically, we characterize the fundamental tradeoff between regret and estimation error through information-theoretic bounds by constructing hard instances.
- We propose a modified UCB algorithm that dynamically adjusts exploration and exploitation efforts to maintain Approximate Pareto Optimality. The algorithm incorporates mechanisms for forced exploration to improve parameter estimation accuracy without incurring excessive regret.
- We prove that our algorithm achieves sublinear regret and estimation error rates that asymptotically approach the Approximate Pareto frontier. By combining them with the derived lower bounds, we show that our algorithm achieves the best possible rate.

By addressing the dual objectives of regret minimization and preference estimation, this work advances the state-of-the-art in adaptive assortment optimization. It provides a rigorous framework for practitioners to design decision-making policies that are both efficient and statistically robust in complex, dynamic environments.

## 2 Related Literature and Contributions

Aparallel line of work in the MAB literature focuses on best-arm identification under fixed confidence or fixed budget settings (e.g., Gabillon et al. [2012]).Tan et al. [2021] investigate the inherent trade-off between regret minimization and best-arm identification, adopting Pareto optimality to characterize efficient policies. However, these studies are limited to classical K -armed bandits without combinatorial structure.

Our work builds on the MNL-Bandit literature, initiated by Agrawal et al. [2016, 2017], where each 'arm' corresponds to an assortment-a subset of items-and the bandit feedback is the chosen item from the offered set. The MNL-Bandit framework has broad applications in online revenue management, advertising, and recommendation systems Agrawal [2019].

Many recent papers study variants of the original MNL-Bandit model. To name a few, Oh and Iyengar [2019, 2021], Choi et al. [2024], Zhang and Luo [2024] studied the MNL-Bandit model with contextual information; Chen et al. [2020], Foussoul et al. [2023] focused on the MNL-Bandit problem with non-stationarity; Aznag et al. [2021], Chen et al. [2024] considered the MNL-Bandit

problem with knapsack constraints; Perivier and Goyal [2022] tackled the MNL-Bandit problem with dynamic pricing; Lee and Oh [2024], Zhang and Wang [2024] provided improved regret bounds for the MNL-Bandit.

Yet, our focus is different from most existing papers about MNL-Bandit that consider the regret of revenue maximization as the primary objective. Our work is instead motivated by Simchi-Levi and Wang [2023] who consider not only the revenue maximization objective but also the minimization of the estimation errors on the average treatment effects (ATEs). Simchi-Levi and Wang [2023] studied the Pareto frontiers of the K -armed bandit problem with ATE defined as the difference of expected reward of the distinct arms (i.e., the 'treatments'). Besides, several recent papers Zhao [2023], Xu et al. [2024], Wei et al. [2024], Qin and Russo [2024], Cook et al. [2024], Li et al. [2024] have investigated this fundamental trade-off in bandit learning by additional considerations related to fairness, best arm identification, diminishing marginal effects, and optimal statistical accuracy. As far as we know, none of these papers has touched upon the bandit learning problem with a combinatorial nature.

Naturally, a seemingly straightforward solution for achieving Pareto optimality for MNL-Bandits would be to generalize the algorithm by Simchi-Levi and Wang [2023] as simply exploring the assortments as independent arms. Nevertheless, this idea will not work since the number of assortments (i.e., the number of arms) is exponential in the number of items, so applying the simple generalization will result in a large estimation error.

The study of Pareto optimality in MNL-Bandits holds significant practical relevance, particularly in online recommendation systems such as those used by Netflix or Amazon. These platforms must simultaneously minimize regret and accurately estimate user preferences by adaptively selecting assortments. A policy on the Pareto frontier ensures that short-term gains in engagement or revenue do not come at the expense of long-term learning. By balancing exploration and exploitation, such policies enable effective and sustainable personalization strategies.

To this end, we propose a novel UCB-based algorithm that incorporates structured exploration of the MNL parameter space. The algorithm achieves low regret while maintaining a small estimation error that scales linearly with the number of items. In particular, it attains a lower error in estimating average treatment effects (ATE) between assortments, scaling as N 2 · √ N 1 -α , significantly improving upon prior approaches where estimation error scales exponentially with the size of the assortment space. Furthermore, we generalize the notion of Pareto Optimality to encompass both cumulative regret and estimation errors in expected revenues and attraction parameters. We also derive necessary and sufficient conditions for achieving Approximate Pareto Optimality in the MNL-Bandit setting, providing a principled foundation for designing multi-objective learning algorithms.

## 3 Model

## 3.1 The Basic MNL Model for Assortment Selection

At each time t , the seller offers an assortment S t ⊆ { 1 , . . . , N } . The customer chooses an item c t ∈ S t ∪ { 0 } , where 0 denotes the 'no-purchase' option. This choice is observed by the seller and used to refine future decisions.

Under the MNL model, the probability of selecting item i ∈ S t when offered S t = S is:

<!-- formula-not-decoded -->

where v i &gt; 0 is the unknown attraction parameter of item i , and v 0 = 1 is fixed for the no-purchase option. These parameters reflect item attractiveness and must be learned from customer choices.

Given v = ( v 1 , . . . , v N ) , the expected revenue of offering S is: R ( S, v ) = ∑ i ∈ S r i v i 1+ ∑ j ∈ S v j , where r i &gt; 0 is the known revenue of item i .

## 3.2 MNL-Bandit for Online Assortment Optimization

Given the basic MNL model, our objective is to design a history-dependent policy π that selects assortments ( S 1 , S 2 , . . . , S T ) over T decision periods to maximize the cumulative expected revenue:

E π ( ∑ T t =1 R ( S t , v ) ) , where R ( S, v ) is the expected revenue from offering assortment S . Direct optimization of the cumulative revenue is not tractable due to the unknown attraction parameters v . The parameters v i must be learned iteratively through consumer feedback, introducing the need to balance exploration (offering diverse assortments to learn v ) and exploitation (offering assortments that maximize revenue given the current knowledge of v ). A key performance metric is regret, defined as the cumulative revenue loss compared to the optimal policy with perfect knowledge of v : Reg ( T, v ) = ∑ T t =1 R ( S ∗ , v ) -E π [ ∑ T t =1 R ( S t , v ) ] , where S ∗ = arg max S ⊆{ 1 ,...,N } R ( S, v ) represents the optimal assortment under perfect knowledge. The regret measures the performance gap between the ideal revenue and the revenue achieved by the policy π . A well-designed policy aims to minimize regret over finite time steps T , balancing learning and revenue maximization.

## 3.3 Approximate Pareto Optimality in MNL-Bandit

In MNL-Bandit, balancing regret minimization and accurate estimation motivates the use of Pareto Optimality to characterize efficient policies.

Definition 3.1 (Pareto Optimality) . A policy ( π, ̂ ∆) , where π is the decision rule and ̂ ∆ is the estimator, is Pareto Optimal if no other admissible policy ( π ′ , ̂ ∆ ′ ) can strictly improve one objective without worsening the other. Formally, ( π, ̂ ∆) is Pareto Optimal if and only if there does not exist another policy ( π ′ , ̂ ∆ ′ ) such that:

<!-- formula-not-decoded -->

with at least one inequality strict.

Here, Reg π ( T, v ) denotes cumulative regret, and e ( ̂ ∆) = E [ | ̂ ∆ -∆ | ] , where ∆ in MNL-Bandit is either ∆ ( i,j ) R = R ( S τ i ) -R ( S τ j ) for i = j ∈ [ |S| ] or ∆ ( i,j ) v = v i -v j for i = j ∈ [ N ] . S π i is the i -th unique assortment under consideration, where i indexes the set of distinct assortments played by the policy.

̸

̸

While Definition 3.1 states the classical Pareto Optimality, this notion is often too brittle in stochastic bandit settings: constant-factor differences are statistically hard to distinguish, information-theoretic lower bounds and achievable guarantees are typically order-wise. We therefore adopt Approximate Pareto Optimality, which declares a pair optimal unless some alternative attains a strictly smaller rate in one objective (little-o) while remaining within a constant factor (big-O) in the other. This definition filters out immaterial constant effects and yields a stable, interpretable frontier. All results and comparisons that follow are stated under this approximate notion.

Definition 3.2 (Approximate Pareto Optimality) . A policy-estimator pair ( π, ˆ ∆) is approximately Pareto optimal if there exists no other admissible pair ( π ′ , ˆ ∆ ′ ) such that either

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In plain terms, no alternative can achieve a strictly smaller approximate regime (littleo ) in one objective while remaining within a constant factor (bigO ) in the other. Otherwise, if there exists other admissible pair ( π ′ , ˆ ∆ ′ ) that satisfies the condition mentioned above, it can be denoted as ( π ′ , ˆ ∆ ′ ) ≺ apo ( π, ˆ ∆) .

Definition 3.3 (Approximate Pareto Frontier) . The approximate Pareto frontier is the set of maximal elements under ≺ apo :

<!-- formula-not-decoded -->

or

Policies on P apo achieve rate-efficient trade-offs: no alternative improves one objective by a strictly better asymptotic rate (littleo ) while keeping the other within a constant factor (bigO ). Any policy off the frontier is approximately dominated.

Identifying frontier points can be posed as a rate-wise bi-objective problem under the preorder ⪯ apo :

<!-- formula-not-decoded -->

(minimization understood w.r.t. ⪯ apo ) .

1 In practice, we produce a family of solutions indexed by a trade-off parameter, tracing P apo and accommodating different preferences between regret and inference accuracy.

## 4 Algorithm and Analysis

To address the exploration-exploitation trade-off in MNL-Bandit, we propose a UCB-based algorithm that adaptively selects assortments and updates MNL parameter estimates. It is designed to both minimize regret and enable accurate inference under varying trade-off requirements.

## 4.1 Details of the Algorithm

We divide the time horizon T into epochs. In each epoch ℓ , a fixed assortment S ℓ is repeatedly offered until a no-purchase event, leading to an epoch length |E ℓ | ∼ Geom( p 0 ) , where p 0 = p 0 ( S ℓ ) denotes the probability of non-purchase given assortmrnt S ℓ . The total number of epochs L satisfies:

<!-- formula-not-decoded -->

The customer response c t ∈ S ℓ ∪ { 0 } provides feedback on item attractiveness. We define: ˆ v i,ℓ = ∑ t ∈E ℓ I ( c t = i ) , v i,ℓ = 1 T i ( ℓ ) ∑ ℓ ′ ∈T i ( ℓ ) ˆ v i,ℓ ′ , where T i ( ℓ ) is the set of prior epochs containing item i , and T i ( ℓ ) = |T i ( ℓ ) | . These estimates incorporate all historical interactions involving item i .

To balance optimism and statistical reliability, we define the UCB for item i as:

<!-- formula-not-decoded -->

This bound accounts for both variance and sample scarcity, shrinking with more observations.

The next assortment is chosen optimistically:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ensuring that assortment selection is guided by plausible upper bounds on reward.

To ensure sufficient exploration for inference, we introduce a randomization scheme: with probability α ℓ = 1 / (2 ℓ α ) , we offer the complement ( S ∗ ℓ ) c ; otherwise, we offer S ∗ ℓ . The parameter α ∈ [0 , 1 / 2] controls the decay of exploration. When α = 0 , both sets are offered equally-maximizing identifiability but incurring high regret. As α increases, the policy behaves closer to pure UCB, focusing on regret.

This randomized selection deviates from Agrawal et al. [2019], who always choose the optimistic set. Our added stochasticity improves inference by covering the full item set over time.

1 Here 'minimization' is taken with respect to the preorder ⪯ apo : a candidate is minimal if it is not approximately dominated by any other.

## Algorithm 1 MNLEXPERIMENTUCB

- 1: Input: Collection of assortments S , total time steps T , and exploration parameter α ∈ [0 , 1 2 ] .
- 2: Initialization: v UCB i, 0 = 1 , ̂ SumV 0 ( i ) = 0 , ∀ i ∈ [ N ];
- 3: t = 1 , ℓ = 1 keeps track of time steps and total epochs respectively and α ℓ = 1 2 ℓ α
- 4: while t &lt; T do
- 5: Compute S ∗ ℓ := arg max S ∈S ˜ R ℓ ( S ) ,

<!-- formula-not-decoded -->

where ( S ∗ ℓ ) c is the collection of items not in S ∗ ℓ .

- 6: Offer S ℓ and observe customer decision c t .
- 7: E ℓ ←E ℓ ∪ { t } keeps track of time steps in epoch ℓ ;
- 8: if c t = 0 then
- 9: compute ̂ v i,ℓ = ∑ t ∈E ℓ 1 ( c t = i ) , the number of consumers who chose i in epoch ℓ ;
- 10: update T i ( ℓ ) = { τ ≤ ℓ | i ∈ S τ } , T i ( ℓ ) = |T i ( ℓ ) | ;
- 11: update v i,ℓ = ( ∑ τ ∈T i ( ℓ ) ̂ v i,τ ) /T i ( ℓ ) , the sample mean of estimates;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 14: end if
- 15: t ← t +1
- 16: end while
- 17: Return: ̂ v i = ̂ SumV L ( i ) L .

To obtain unbiased long-term estimates, we define a weighted cumulative estimator:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, P ( i ∈ S ℓ ) adjusts for the randomness in selection. The estimator ̂ v i is unbiased since E [ ̂ SumV ℓ ( i )] = ℓ · v i . For a formal justification, we refer the reader to Lemma B.1 and Corollary B.2 in Appendix B. Using the estimated parameters, we compute the revenue for any assortment S τ i via: ̂ R ( S τ i ) = ∑ i ∈ Sτ i r i ̂ v i 1+ ∑ i ∈ Sτ i ̂ v i , enabling post-hoc evaluation and inference over the entire policy.

## 4.2 Analysis of the Algorithm

We make the following assumptions throughout the analysis.

Assumption 4.1 (MNL Parameters)

- (a). The MNL parameter corresponding to any item i ∈ { 1 , . . . , N } satisfies v i ≤ v 0 = 1 .
- (b). The family of assortments S is such that S ∈ S and Q ⊆ S imply that Q ∈ S .

The above assumptions about MNL parameters are widely assumed in the MNL-Bandit literature (see, e.g., Agrawal et al. [2019]). Assumption (a) reflects real-world settings (e.g., online retail) where no-purchase is typically the most likely outcome. Assumption (b) ensures structural closure under item removal and holds under many natural constraints, including cardinality and matroid constraints. Notably, Assumption (a) simplifies our analysis but is not essential for regret bounds. We include Algorithm 3 in Appendix D for relaxing (a).

## 4.2.1 Regret Upper Bound

In Agrawal et al. [2019], the regret is proved to satisfy Reg π ( T, v ) ≤ C 1 √ NT log NT + C 2 N log 2 NT , where C 1 and C 2 are absolute constants independent of problem parameters. In

MNL-Bandit, accurate estimation of the attraction parameters requires observing a sufficiently diverse set of choices. By enforcing the selection of suboptimal assortments, we introduce additional regret, but this helps improve the long-term statistical power of the estimation. Therefore, the inclusion of this extra regret is a necessary design choice to balance regret minimization with the accurate estimation of MNL parameters.

In our algorithm, for epoch ℓ , we set a carefully controlled probability α ℓ = 1 2 ℓ α for the supplement set of the optimistic assortment, i.e. P ( S ℓ = ( S ∗ ℓ ) c ) = α ℓ , which introduces extra regret to the regret term in Agrawal et al. [2019]. Define ∆ R ℓ := E [ |E ℓ | · [ R ( S ∗ , v ) -R ( S ℓ , v )] | S ℓ ] as the regret in epoch ℓ . Since we have shown the length of an epoch |E ℓ | conditioned on S ℓ is a geometric random variable with success probability being the probability of no purchase in S ℓ , i.e. 1 / (1 + ∑ i ∈ S ℓ v i ) , then we can derive an upper bound of ∆ R ℓ which is

<!-- formula-not-decoded -->

Thus we introduce ( N +1) · ∑ L ℓ =1 P ( S ℓ = ( S ∗ ℓ ) c ) ≤ CN · T 1 -α more in the cumulative regret. So we have Reg π ( T, v ) ≤ C 1 √ NT log NT + C 2 N log 2 NT + C 3 NT 1 -α . We provide the detailed proof of the following theorem in Appendix B.1.

Theorem 4.1. For any instance v = ( v 0 , . . . , v N ) of the MNL-Bandit problem with N items, r i ∈ [0 , 1] , and given the problem assumption, let Algorithm 1 run with α ∈ [0 , 1 2 ] the regret at any time T is

.

## 4.2.2 Inference for Attraction Parameters

Nowwefocus on estimating the attraction parameters. Since we have shown that E [ ̂ SumV ℓ ( i )] = ℓ · v i , we can define a set of martingales as M i ℓ = ̂ SumV ℓ ( i ) -ℓ · v i for i ∈ { 1 , · · · , N } . For any ℓ ∈ L , the martingale difference of M i ℓ is | M i ℓ -M i ℓ -1 | = ∣ ∣ ∣ ̂ v i,ℓ P ( i ∈ S ℓ | S ∗ ℓ ) · 1 ( i ∈ S ℓ ) -v i ∣ ∣ ∣ so that the variance of M i L can be written as

<!-- formula-not-decoded -->

and bounded by ∑ L ℓ =1 1 P ( i ∈ S ℓ | S ∗ ℓ ) · E [( ̂ v i,ℓ ) 2 | i ∈ S ℓ , H ℓ -1 ] . And we know ̂ v i,ℓ is a geometric random variable with parameter 1 1+ v i and P ( i ∈ S ℓ | S ∗ ℓ ) ≥ α ℓ . So we further bound the variance of M i L by 6( L +1) α +1 α +1 . By Bernstein's inequality, we can derive the following theorem.

Theorem 4.2. If Algorithm 1 runs with α ∈ [0 , 1 2 ] , with probability 1 -δ , for any i ∈ [ N ]

<!-- formula-not-decoded -->

Taking δ = 1 L 2 , then we cen get that E [ | ̂ v i -v i | ] = O (√ 1 ( L +1) 1 -α ) . Since ∑ L ℓ =1 |E ℓ | ≥ T and E [ |E ℓ | ] = 1 + ∑ i ∈ S ℓ v i , we can easily derive T L +1 ≤ N +1 which further implies that | ̂ v i -v i | ≤ 12 √ 2 ln ( 2 δ ) · √ ( N T ) 1 -α . So E [ | ̂ v i -v i | ] = O ( √ T α -1 ) . And according to triangle inequality we have: | ̂ ∆ ( i,j ) v -∆ ( i,j ) v | = | ( ̂ v i -̂ v j ) -( v i -v j ) | = | ( ̂ v i -v i ) -( ̂ v j -v j ) | ≤ | ̂ v i -v i | + | ̂ v j -v j | . So we can easily get the following corollary:

̸

Corollary 4.3. If Algorithm 1 runs with α ∈ [0 , 1 2 ] , the estimation error of parameter differences, i.e. ∆ ( i,j ) v = v i -v j for all i, j ∈ [ N ] , i = j is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 4.2.3 Inference for Expected Revenue

Since we have obtained the unbiased estimators for attraction parameters, a direct and useful idea is to use the estimates of v i to estimate the expected revenue, i.e. R ( S τ ) = ∑ i ∈ Sτ r i ̂ v i 1+ ∑ i ∈ Sτ ̂ v i . Then we can derive that | ̂ R ( S τ ) -R ( S τ ) | can be bounded by (2 N 2 + N ) | ̂ v i -v i | . And combined with Theorem 4.2, we have the following theorem:

Theorem 4.4. If Algorithm 1 runs with α ∈ [0 , 1 2 ] , with probability 1 -δ , for any τ ∈ [ |S| ]

<!-- formula-not-decoded -->

Thus we have: E [ | ̂ R ( S τ ) -R ( S τ ) | ] = O ( √ T α -1 ) . And similarly as above, by triangle inequality we have: | ̂ ∆ ( i,j ) R -∆ ( i,j ) R | = | ( ̂ R ( S τ i ) -̂ R ( S τ j )) -( R ( S τ i ) -R ( S τ j )) | = | ( ̂ R ( S τ i ) -R ( S τ i )) -( ̂ R ( S τ j ) -R ( S τ j )) | ≤ | ( ̂ R ( S τ i ) -R ( S τ i )) | + | ( ̂ R ( S τ j ) -R ( S τ j )) | Combined with Theorem 4.4, we can get that | ̂ ∆ ( i,j ) R -∆ ( i,j ) R | ≤ 72 ln ( 2 δ ) · N 2 √ ( N T ) 1 -α . Thus we get the following corollary:

̸

Corollary 4.5. If Algorithm 1 runs with α ∈ [0 , 1 2 ] , the estimation error of parameter differences, i.e. ∆ ( i,j ) R = R ( S τ i ) -R ( S τ j ) for all i, j ∈ [ |S| ] , i = j is | ̂ ∆ ( i,j ) R -∆ ( i,j ) R | = O ( √ T α -1 ) .

̸

As shown above, given a fixed total time steps T and confidence level δ , the estimation error of the difference between the expected revenue of assortment S τ i and S τ j , for any i = j ∈ [ |S| ] , scales as N 2 · √ N 1 -α in the number of items N. This indicates the effectiveness of our algorithm in addressing the complexities arising from the combinatorial nature of the MNL model.

## 4.2.4 On Approximate Pareto Optimality

Now we present the conditions of Approximate Pareto Optimality and verify that our algorithm is indeed Approximately Pareto Optimal. Note that when it comes to comparing regrets with errors, we will only focus on the order of T ignoring the universal constant, since T is usually relatively large.

(1) Regret and Estimation Error of ∆ R : In classic multi-armed bandit with K arms, Simchi-Levi and Wang [2023] proposed the necessary and sufficient condition of the Pareto Optimality of an admissible pair. All the same, by ignoring the MNL structure, we can directly see each assortment S τ as an arm with its only reward distribution of mean R ( S τ ) and then it follows the classic MAB games. Adding the Approximate natation doesn't change the sufficient and necessary condition and its proof except changing the RHS from ˜ O (1) to Θ(1) . We can easily get the following theorem whose rigorous proof is provided in Appendix A.

Theorem 4.6. In MNL-Bandit, an admissible pair ( π, ̂ ∆ R ) is Approximately Pareto Optimal if and only if it satisfies

<!-- formula-not-decoded -->

where φ is a MNL-Bandit instance, e φ ( T, ̂ ∆ ( i,j ) R ) is the estimation error of ATE between S τ i and S τ j , i.e. e φ ( T, ̂ ∆ ( i,j ) R ) = E π [∣ ∣ ∣ ̂ ∆ ( i,j ) R -∆ ( i,j ) R ∣ ∣ ∣ ] and Reg φ ( T, π ) is the cumulative regret within T time steps under policy π .

For Algorithm 1, by Theorem 4.1 and Theorem 4.4, we can get that max φ ∈E 0 [( max i&lt;j ≤|S| e φ ( T, ̂ ∆ ( i,j ) R ) ) √ Reg φ ( T, π ) ] = Θ(1) holds for Algorithm 1 when α ∈ [0 , 1 2 ) which implies that our algorithm is Approximately Pareto Optimal in terms of regret and estimation error of ∆ R .

(2) Regret and Estimation Error of ∆ v : Then we move on to analyze the Approximate Pareto Optimality between regret and ∆ v . Let us start with MNL-Bandit with only 2 items, which can later be extended to the general case N ≥ 2 . In the following theorem, we establish an minimax lower bound for e φ ( T, ̂ ∆ v ) √ Reg φ ( T, π ) .

Theorem 4.7. When N = 2 , for any admissible pair ( π, ̂ ∆ v ) , there always exists a hard instance φ ∈ E 0 such that e φ ( T, ̂ ∆ v ) √ Reg φ ( T, π ) is no less than a constant order, i.e.,

<!-- formula-not-decoded -->

In the above theorem, we have shown that no solution can perform better than a constant order in terms of e φ ( T, ̂ ∆ v ) √ Reg φ ( T, π ) in the worst case. The following theorem states that one policy is Approximately Pareto Optimal if it can achieve the constant order on e φ ( T, ̂ ∆ v ) √ Reg φ ( T, π ) .

Theorem 4.8. When N = 2 , an admissible pair ( π, ̂ ∆ v ) is Approximately Pareto Optimal if it satisfies

<!-- formula-not-decoded -->

Then we extend our results from N = 2 to the general case. According to Corollary 4.3, we can have max i&lt;j ≤ N e ( T, ∆ ( i,j ) v ) = O ( √ T α -1 ) . Then combined with Theorem 4.1, we can naturally derive that ( max i&lt;j ≤ N e φ ( T, ̂ ∆ ( i,j ) v ) ) √ Reg φ ( T, π ) = Θ(1) for all MNL-Bandit instance φ .

By such an observation, we can generalize Theorem 4.8 and get the sufficient condition for the general case: max φ ( max i&lt;j ≤ N e φ ( T, ̂ ∆ ( i,j ) v ) ) √ Reg φ ( T, π ) = Θ(1) . Therefore, Algorithm 1 is Approximately Pareto Optimal for all α ∈ [0 , 1 2 ] . Then combined the sufficient condition with the definition of Approximate Pareto Optimality, we can prove the following theorem by contradiction:

Theorem 4.9. In MNL-Bandit with N items, any Approximately Pareto Optimal ( π, ̂ ∆ v ) has

<!-- formula-not-decoded -->

Then we can conclude the following corollary:

Corollary 4.10. In MNL-Bandit, an admissible pair ( π, ̂ ∆ R ) is Approximately Pareto Optimal if and only if it satisfies max φ ∈E 0 ( max i&lt;j ≤ N e φ ( T, ̂ ∆ ( i,j ) v ) ) √ Reg φ ( T, π ) = Θ(1) .

Therefore, we conclude that the sufficient and necessary condition of Approximate Pareto Optimality is max φ ∈E 0 ( max i&lt;j e φ ( T, ̂ ∆ ( i,j ) ) ) √ Reg φ ( T, π ) = Θ(1) , where ̂ ∆ can be either ̂ ∆ R or ̂ ∆ v . As an immediate corollary, our algorithm is Approximately Pareto Optimal in both cases.

## 4.3 Discussion on the Assortment Size

There may be concerns that our algorithm implicitly requires the maximum assortment size K to satisfy K = max { | S ∗ | , | ( S ∗ ) c | } ≥ N 2 , since at each epoch we offer either the optimal assortment S ∗ or its complement ( S ∗ ) c = [ N ] \ S ∗ . This departure from the conventional assumption K ≪ N could restrict the applicability of our method in scenarios where only much smaller assortments are feasible.

In case of restriction on the maximum assortment size, i.e. K ≤ K ∗ , we can adjust the rule of choosing assortment in Algorithm 1 as below: α ℓ = 1 ⌈ N K ∗ ⌉· ℓ α and

<!-- formula-not-decoded -->

At the start of each epoch, we divide the products in ( S ∗ ℓ ) c into ⌈ N -| S ∗ ℓ | K ∗ ⌉ assortments, denoted as ( S ∗ ℓ ) c i , i ∈ { 1 , ..., ⌈ N -| S ∗ ℓ | K ∗ ⌉} , where each assortments contains at most K ∗ products and each product occurs in only one assortment.

By this adjustment, we can achieve similar result in MNL bandit and satisfy the restriction on the maximum assortment size without other modification to Algorithm 1. This shows the flexibility of our algorithm. The adjusted algorithm for this case is shown in Appendix C as Algorithm 2, which is also Approximately Pareto Optimal.

0.6

Cumulative Regre

150

0.31

100

0.21

MNLExperimentUC8 (a-0)

MN. Experiment 8(a=0.5

MNLexperimentoco (a».)

200

200

sensitivity or Moty to Exploraton Parameter a

Comoanison of Regret ys. Time Across Alconthms.

400

400

a=0.23

0-0.31

a=0.75

## 5 Numerical Experiments

We evaluate the practical performance of our UCB-with-complement exploration algorithm (MNLEXPERIMENTUCB) on a synthetic MNL-bandit task. We compare against two baselines: 1000

- MNLBanditEE : the standard exploration-exploitation UCB without complement-set sampling as in Agrawal et al. [2019].
- EXP3EG : an EXP3-style scheme as in Simchi-Levi and Wang [2023] with default parameters α = 0 . 5 , δ = 0 . 05 .

In each trial we draw v i ∼ U (0 . 1 , 1 . 0) , r i ∼ U (0 . 5 , 1 . 5) , i = 1 , . . . , 10 , and allow any nonempty subset of size at most K = 5 (total ∑ 5 k =1 ( 10 k ) = 637 assortments). Our method uses forced complement sampling with decay α ℓ = 1 2 ℓ α , α ∈ { 0 , 0 . 25 , 0 . 5 , 1 } . All experiments run for T = 1000 steps and are repeated over 20 independent trials for regret and time-series plots, and 20 trials for final estimation boxplots.

We use the following metrics to evaluate the algorithms:

- (1) Cumulative Regret: R ( T ) = ∑ T t =1 ( r ∗ -E [ R ( S t )]) , where r ∗ = max S E [ R ( S )] ;
- (2) MSE of attraction parameters: MSE v ( t ) = 1 N ∑ N i =1 (ˆ v ( t ) i -v i ) 2 ;

(3)

MSEof expected revenue estimates:

<!-- formula-not-decoded -->

Cumulative Regret: Figure 1 shows the mean cumulative regret over 20 runs. EXP3EG (brown) grows nearly linearly, while MNLBanditEE (blue) and our method (orange-purple) achieve much lower regret. As α increases, regret decreases in line with the theoretical O ( T 1 -α ) bound.

Figure 1: Comparison of Average cumulative regret.

![Image](data:image/png;base64,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)

Estimation of Attraction Parameters: Figure 2 plots the time-evolution of MSE v ( t ) (mean over 20 trials). All curves start high and rapidly decrease; small α produces a quicker initial drop but larger variance, whereas large α converges more slowly.

Figure 2: MSE v ( t ) vs. t .

![Image](data:image/png;base64,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)

Estimation of Expected Revenues: Figure 3 reports MSE R ( t ) . The trends mirror those of MSE v : moderate α strikes the best balance between under- and over-exploration, yielding the most accurate and stable revenue estimates.

MSE

R

(

t

) =

1

|S|

∑

R

(

t

)

(

S

)

-

R

(

S

))

2

, where

( ˆ

S

∈S

ˆ

R

(

t

)

(

S

) =

## 6 Conclusion

In this paper, we investigate Approximate Pareto Optimality for the MNL-Bandit model. We define Approximate Pareto Optimality as the trade-off between regret of the revenue minimization and the average estimation errors on assortment revenues or MNL parameters. We present sufficient and necessary conditions of Approximate Pareto Optimality and develop a novel algorithm that achieves Approximate Pareto Optimality for the MNL-Bandit. Future directions will include extending our result to more general dynamic assortment problems or studying Pareto Optimality with other forms of bandit feedback or other regret/ATE metrics.

## References

- Shipra Agrawal. Recent advances in multiarmed bandits for sequential decision making. Operations Research &amp; Management Science in the Age of Analytics , pages 167-188, 2019.
- Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. A near-optimal explorationexploitation approach for assortment selection. In Proceedings of the 2016 ACM Conference on Economics and Computation , pages 599-600, 2016.
- Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. Thompson sampling for the mnl-bandit. In Conference on learning theory , pages 76-78. PMLR, 2017.
- Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. Mnl-bandit: A dynamic learning approach to assortment selection. Operations Research , 67(5):1453-1485, 2019.
- Abdellah Aznag, Vineet Goyal, and Noemie Perivier. Mnl-bandit with knapsacks: a near-optimal algorithm. arXiv preprint arXiv:2106.01135 , 2021.
- Xi Chen, Yining Wang, and Yuan Zhou. Dynamic assortment optimization with changing contextual information. Journal of machine learning research , 21(216):1-44, 2020.
- Xi Chen, Mo Liu, Yining Wang, and Yuan Zhou. A re-solving heuristic for dynamic assortment optimization with knapsack constraints. arXiv preprint arXiv:2407.05564 , 2024.
- Hyun-jun Choi, Rajan Udwani, and Min-hwan Oh. Cascading contextual assortment bandits. Advances in Neural Information Processing Systems , 36, 2024.
- Josh Constine. Facebook feed change sacrifices time spent and news outlets for 'well-being', 2018. URL https://techcrunch.com/2018/01/11/facebook-time-well-spent/ .
- Thomas Cook, Alan Mishler, and Aaditya Ramdas. Semiparametric efficient inference in adaptive experiments. In Causal Learning and Reasoning , pages 1033-1064. PMLR, 2024.
- Ayoub Foussoul, Vineet Goyal, and Varun Gupta. Mnl-bandit in non-stationary environments. arXiv preprint arXiv:2303.02504 , 2023.
- Victor Gabillon, Mohammad Ghavamzadeh, and Alessandro Lazaric. Best arm identification: A unified approach to fixed budget and fixed confidence. Advances in Neural Information Processing Systems , 25, 2012.

Figure 3: MSE R ( t ) vs. t .

![Image](data:image/png;base64,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)

- Joongkyu Lee and Min-hwan Oh. Nearly minimax optimal regret for multinomial logistic bandit. arXiv preprint arXiv:2405.09831 , 2024.
- Jiachun Li, David Simchi-Levi, and Yunxiao Zhao. Optimal adaptive experimental design for estimating treatment effect. arXiv preprint arXiv:2410.05552 , 2024.
- Min-hwan Oh and Garud Iyengar. Thompson sampling for multinomial logit contextual bandits. Advances in Neural Information Processing Systems , 32, 2019.
- Min-hwan Oh and Garud Iyengar. Multinomial logit contextual bandits: Provable optimality and practicality. In Proceedings of the AAAI conference on artificial intelligence , volume 35, pages 9205-9213, 2021.
- Noemie Perivier and Vineet Goyal. Dynamic pricing and assortment under a contextual mnl demand. Advances in Neural Information Processing Systems , 35:3461-3474, 2022.
- Chao Qin and Daniel Russo. Optimizing adaptive experiments: A unified approach to regret minimization and best-arm identification. arXiv preprint arXiv:2402.10592 , 2024.
- Nick Scheidies. 15 business lessons from amazon's jeff bezos, 2012. URL https://www. incomediary.com/15-business-lessons-from-amazons-jeff-bezos/ .
- David Simchi-Levi and Chonghuan Wang. Multi-armed bandit experimental design: Online decisionmaking and adaptive inference. In International Conference on Artificial Intelligence and Statistics , pages 3086-3097. PMLR, 2023.
- Vincent Y. F. Tan, Zixin Zhong, and Wang Chi Cheung. On the pareto frontier of regret minimization and best arm identification in stochastic bandits. arXiv preprint arXiv:2110.08627 , 2021.
- Waverly Wei, Xinwei Ma, and Jingshen Wang. Fair adaptive experiments. Advances in Neural Information Processing Systems , 36, 2024.
- Jingxu Xu, Yuhang Wu, Yingfei Wang, Chu Wang, and Zeyu Zheng. A/b test and online experiment under diminishing marginal effects: Regret minimization and statistical inference. Available at SSRN 4640583 , 2024.
- Mengxiao Zhang and Haipeng Luo. Contextual multinomial logit bandits with general value functions. arXiv preprint arXiv:2402.08126 , 2024.
- Zhiheng Zhang and Zichen Wang. Online experimental design with estimation-regret trade-off under network interference. arXiv preprint arXiv:2412.03727 , 2024.
- Jinglong Zhao. Adaptive neyman allocation. arXiv preprint arXiv:2309.08808 , 2023.

## A Sufficient Conditions for Approximate Pareto Optimality

## A.1 Pareto Optimality between regret and ATE estimation

First, let us consider the Pareto optimality in classic multi-armed bandit with K arms introduced in Simchi-Levi and Wang [2023] which proposes the necessary and sufficient condition for Pareto optimality in MAB with general K arms. Specifically, an admissible pair ( π ∗ , ̂ ∆ ∗ ) is Pareto optimal if and only if

<!-- formula-not-decoded -->

In the MNL-Bandit setting, we can see each assortment as an arm, thus we have |S| arms and each arm has its own reward distribution. Speciafically, assortment S τ has a reward distribution with mean µ τ = ∑ i ∈ S τ r i v i 1+ ∑ i ∈ S τ v i . Therefore, it follows that there also exists Pareto optimality in the context of MNL-Bandit and the necessary and sufficient condition is

<!-- formula-not-decoded -->

where e φ ( T, ∆ ( i,j ) R ) is the estimation error of ATE between S τ i and S τ j ( ∆ ( i,j ) R = R ( S τ i ) -R ( S τ j ) ), i.e. e φ ( T, ∆ ( i,j ) R ) = E π [∣ ∣ ∣ ̂ ∆ ( i,j ) R -∆ ( i,j ) R ∣ ∣ ∣ ] and Reg φ ( T, π ) is the cumulative regret within T time steps under policy π .

But there may be concern on a fundamental rigor issue : the stated definitions of Pareto Optimality are not used consistently in the theorems; 'if and only if' claims rely on hidden logarithmic factors and a O(1) shorthand that makes the statements ill-posed. Therefore, in our paper, we proposed the definition of Approximate Pareto Optimality, which can fix the issue mentioned above. We also provide a formal proof of the sufficient condition of Approximate Pareto Optimality in Appendix A.3.

## A.2 Approximate Pareto Optimality between regret and parameter differences inference (when N = 2 )

Lemma A.1. When N = 2 , for any given online decision-making policy π , the error of any estimator of parameter difference can be lower bounded as follows, for any function f : n → [0 , 1 8 ] and any u ∈ E .

.

<!-- formula-not-decoded -->

Proof. First, we define distribution D as if X ∼ D ( a, b ) then X = 0 with probability a a + b and X = r with probability b a + b . Then we construct MNL model instance v = ( v 1 , v 2 ) and two MNLbandit instance φ 1 = ( D ( v 0 , v 1 ) , D ( v 0 , v 2 )) and φ 2 = ( D ( v 0 , v 1 ) , D ( v 0 , v 2 -2 f ( t )) . Without loss of generality we can assume v 1 ≥ v 2 and v 2 -2 f ( t ) ≥ 1 8 . Then we have ∆ φ 1 = v 1 -v 2 and ∆ φ 2 = v 1 -v 2 +2 f ( t ) .

We define the minimum distance test ψ ( ̂ ∆ v ) that is associated to ̂ ∆ v by

<!-- formula-not-decoded -->

If ψ ( ̂ ∆ v ) = 1 , we know that | ̂ ∆ v -∆ φ 1 | ≤ | ̂ ∆ v -∆ φ 2 | . By the triangle inequality, we can have, if ψ ( ̂ ∆ v ) = 1 ,

<!-- formula-not-decoded -->

which yields that

<!-- formula-not-decoded -->

Symmetrically, if ψ ( ̂ ∆ v ) = 2 , we can have

<!-- formula-not-decoded -->

Therefore, we can use this to show

<!-- formula-not-decoded -->

̸

where the last infimum is taken over all tests ψ based on H t that take values in { 1 , 2 } .

<!-- formula-not-decoded -->

where the equality holds due to Neyman-Pearson lemma and the second inequality holds due to Pinsker's inequality, and the third inequality holds due to the following:

<!-- formula-not-decoded -->

where we use

<!-- formula-not-decoded -->

and the last inequality holds because the history H t is generated by π and ∆ φ 1 E φ 1 [ T 2 ( T )] is just the expected regret of φ 1 , which is just the definition of regret. Thus we finish our proof.

̸

Theorem A.2. When N = 2 , for any admissible pair ( π, ̂ ∆ v ) , there always exists a hard instance φ ∈ E 0 such that e φ ( T, ̂ ∆ v ) √ Reg φ ( T, π ) is no less than a constant order, i.e.,

<!-- formula-not-decoded -->

Proof. Based on Lemma A.1, given policy π , and ̂ ∆ v , if f ( T ) ≤ √ | ∆ u | 64 Reg u ( T,π ) for some u ∈ E 0 ,

<!-- formula-not-decoded -->

where the second inequality holds due to Lemma A.1. We use φ π, ̂ ∆ v to denote arg max φ ∈E 0 E [ | ̂ ∆ v -∆ φ | ] given policy π and ̂ ∆ v , and thus e φ π, ̂ ∆ v ( T, ̂ ∆ v ) ≥ f ( T ) 4 . After taking f ( T ) = √ | ∆ φ π, ̂ ∆ v | 64 Reg φ π, ̂ ∆ v ( T,π ) , we retrieve for any given policy π and ̂ ∆ v ,

<!-- formula-not-decoded -->

where the last equation holds because we plug in f ( T ) and ∆ φ = Θ(1) for φ ∈ E 0 . Since the above inequalities hold for any policy π and ̂ ∆ v , we finish the proof.

Theorem A.3. When N = 2 , an admissible pair ( π, ̂ ∆ v ) is Approximately Pareto Optimal if it satisfies

<!-- formula-not-decoded -->

Proof. We conduct proof by contradiction. Assume that ( π 0 , ̂ ∆ 0 ) satisfies the above equality, but is not Approximately Pareto Optimal. This means that there exists a ( π 1 , ̂ ∆ 1 ) that Approximately Pareto dominates ( π 0 , ̂ ∆ 0 ) . The lower bound in Theorem A.2 guarantees that there must be a point at the front of ( π 1 , ̂ ∆ 1 ) , denoted by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the definition of Approximately Pareto dominance, there exists

<!-- formula-not-decoded -->

such that

Note that, as we have mentioned, the strict inequality in the above inequality is in the term of the dependence of T . It means that

<!-- formula-not-decoded -->

for some strictly positive p &gt; 0

, which contradicts with our assumption.

satisfying

## A.3 Approximate Pareto Optimality between regret and ATE estimation

The general idea is that in MNL-Bandit, we can see each assortment as an arm, thus we have |S| arms and each arm has its own reward distribution.

Lemma A.4. When N = 2 , for any given online decision-making policy π , the error of any estimator of parameter difference can be lower bounded as follows, for any function f : n → [0 , 1 8 ] and any u ∈ E .

<!-- formula-not-decoded -->

Theorem A.5. When N = 2 , for any admissible pair ( π, ̂ ∆ R ) , there always exists a hard instance φ ∈ E 0 such that e φ ( T, ̂ ∆ R ) √ Reg φ ( T, π ) is no less than a constant order, i.e.,

<!-- formula-not-decoded -->

The proof of Lemma A.4 and Theorem A.5 is the same as that in Simchi-Levi and Wang [2023].

Theorem A.6. When N = 2 , an admissible pair ( π, ̂ ∆ R ) is Approximately Pareto Optimal if it satisfies

<!-- formula-not-decoded -->

Proof. We conduct proof by contradiction. Assume that ( π 0 , ̂ ∆ 0 ) satisfies the above equality, but is not Approximately Pareto Optimal. This means that there exists a ( π 1 , ̂ ∆ 1 ) that Approximately Pareto dominates ( π 0 , ̂ ∆ 0 ) . The lower bound in Theorem A.2 guarantees that there must be a point at the front of ( π 1 , ̂ ∆ 1 ) , denoted by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the definition of Approximately Pareto dominance, there exists

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

such that

Note that, as we have mentioned, the strict inequality in the above inequality is in the term of the dependence of T . It means that

<!-- formula-not-decoded -->

for some strictly positive p &gt; 0 , which contradicts with our assumption.

In this way, we get a more rigorous statement on the case when N = 2 . And the generalization from N = 2 to general N is the same as that in Simchi-Levi and Wang [2023]. Thus we have that:

Theorem A.7. In MNL-Bandit, an admissible pair ( π, ̂ ∆ R ) is Approximately Pareto Optimal if and only if it satisfies

<!-- formula-not-decoded -->

where φ is a MNL-Bandit instance, e φ ( T, ̂ ∆ ( i,j ) R ) is the estimation error of ATE between S τ i and S τ j , i.e. e φ ( T, ̂ ∆ ( i,j ) R ) = E π [∣ ∣ ∣ ̂ ∆ ( i,j ) R -∆ ( i,j ) R ∣ ∣ ∣ ] and Reg φ ( T, π ) is the cumulative regret within T time steps under policy π .

satisfying

## B Analysis of Algorithm 1

## B.1 Regret Analysis

Lemma B.1. (Agrawal et al. [2019] Lemma A.1) The moment generating function of the estimate conditioned on S ℓ , ̂ v i , is given by:

<!-- formula-not-decoded -->

Proof. we have that the probability of a no-purchase event when assortment S ℓ is offered is given by

<!-- formula-not-decoded -->

Let n ℓ be the total number of offerings in epoch ℓ before a no-purchase occurred (i.e., n ℓ = |E ℓ | -1 ). Therefore, n ℓ is a geometric random variable with probability of success p 0 ( S ℓ ) . And given any fixed value of n ℓ , ϕ i,ℓ is a binomial random variable with n ℓ trials and a probability of success given by

<!-- formula-not-decoded -->

In the calculations below, for brevity, we use p 0 and q i to denote p 0 ( S ℓ ) and q i ( S ℓ ) , respectively. Hence, we have

<!-- formula-not-decoded -->

Because the moment-generating function for a binomial random variable with parameters n, p is ( pe θ +1 -p ) n , we have

<!-- formula-not-decoded -->

For any α , such that α (1 -p ) &lt; 1 , if n is a geometric random variable with parameter p , then we have

<!-- formula-not-decoded -->

Because n ℓ is a geometric random variable with parameter p 0 , and by the definition of q i and p 0 , we have q i (1 -p 0 ) = v i p 0 , it follows that for any θ &lt; log 1 + v i /v i , we have

<!-- formula-not-decoded -->

Then we can derive the following corollary from Lemma B.1.

Corollary B.2 (Unbiased Estimates) . We have the following results:

- (1) The estimates ̂ v i,ℓ , ℓ ≤ L , are i.i.d. geometrical random variables with parameter 1 1+ v i . Thus:

<!-- formula-not-decoded -->

- (2) ̂ v i,ℓ and v i ℓ are both unbiased estimates of V i for all i, t .

LemmaB.3. (Agrawal et al. [2019] Lemma A.2) If v i ≤ v 0 for all i , then for every epoch ℓ , according to our algorithm:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem B.4. (Agrawal et al. [2019] Lemma 4.1) For every ℓ = 1 , . . . , L :

- (1) v UCB i,ℓ ≥ v i with probability at least 1 -6 Nℓ for all i = 1 , . . . , N .
- (2) There exists constant C 1 , C 2 such that:

<!-- formula-not-decoded -->

with probability at least 1 -7 Nℓ .

Proof. By the design of Algorithm 1, we have

<!-- formula-not-decoded -->

Therefore, from Lemma B.3, we have

Thus,

<!-- formula-not-decoded -->

The first inequality in Theorem B.4 follows from (B.5). From the triangle inequality and (B.4), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Lemma B.3, we have

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Using the fact that √ a + √ b &lt; √ a + b , for any positive numbers a, b , we have

<!-- formula-not-decoded -->

From Lemma B.3, we have

<!-- formula-not-decoded -->

From (B.6), and applying the union bound on (B.7) and (B.8), we obtain

<!-- formula-not-decoded -->

Theorem B.4 follows from the above inequality and (B.5).

Lemma B.5. (Agrawal et al. [2019] Lemma A.3) Assume 0 ≤ w i ≤ v UCB i,ℓ for all i = 1 , . . . , N . Suppose S is the optimal assortment when the MNL parameters are given by w . Then:

<!-- formula-not-decoded -->

Proof. We prove the result by first showing that for any j ∈ S , we have R ( S, w j ) ≥ R ( S, w ) , where w j is vector w with the j th component increased to v UCB j (i.e., w j i = w i for all i = j and w j j = v UCB j ). We can use this result iteratively to argue that increasing each parameter of MNL to the highest possible value increases the value of R ( S, w ) to complete the proof.

̸

If there exists j ∈ S such that r j &lt; R ( S ) , then removing the product j from assortment S yields a higher expected revenue, contradicting the optimality of S . Therefore, we have

<!-- formula-not-decoded -->

Multiplying by ( v UCB j -w j )( ∑ i ∈ S \ j w i +1) on both sides of the above inequality and rearranging terms, we can show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem B.6. (Agrawal et al. [2019] Lemma 4.2) Suppose S ∗ ∈ S is the assortment with the highest expected revenue, and our algorithm offers S ℓ = arg max S ∈S ˜ R ℓ ( S ) in epoch ℓ . Then, for epoch ℓ , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -6 ℓ .

Lemma B.7. (Agrawal et al. [2019] Lemma A.4) If r i ∈ [0 , 1] and 0 ≤ v i ≤ v UCB i,ℓ for all i ∈ S ℓ , then:

<!-- formula-not-decoded -->

Proof. Because 1 + ∑ i ∈ S ℓ v UCB i,ℓ ≥ 1 + ∑ i ∈ S ℓ v i,ℓ , we have

<!-- formula-not-decoded -->

Theorem B.8. (Agrawal et al. [2019] Lemma 4.3) If r i ∈ [0 , 1] , there exist constants C 1 and C 2 such that for every ℓ = 1 , . . . , L , we have:

<!-- formula-not-decoded -->

with probability at least 1 -13 ℓ .

Proof. From Lemma B.7, we have

<!-- formula-not-decoded -->

From Lemma B.3, we have that, for each i = 1 , . . . , N and ℓ ,

<!-- formula-not-decoded -->

Therefore, from the union bound, it follows that

<!-- formula-not-decoded -->

Theorem B.8 follows from (B.11) and (B.12).

Theorem B.9. For any instance v = ( v 0 , . . . , v N ) of the MNL-Bandit problem with N items, r i ∈ [0 , 1] , and given the problem assumptions, let Algorithm 1 run with α ∈ [0 , 1 2 ] the regret at any time T is O ( √ NT log NT + N log 2 NT + NT 1 -α ) .

Proof. Now, we can put the lemmas together to analyze the regret:

<!-- formula-not-decoded -->

The probability of a no-purchase conditioned on S ℓ is given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

So,

Thus, by the formula of full probability, we have

<!-- formula-not-decoded -->

Then define ∆ R ℓ as:

<!-- formula-not-decoded -->

Define bad event:

<!-- formula-not-decoded -->

Then according to Theorem B.4 we have:

<!-- formula-not-decoded -->

Define event B ℓ as:

<!-- formula-not-decoded -->

Because both A ℓ and B C ℓ are 'low-probability' events, we can break down the regret in one epoch as follows:

<!-- formula-not-decoded -->

Using the fact that R ( S ∗ , v ) and R ( S ℓ , v ) are both bounded by 1 and V ( S ℓ ) ≤ N , we have ∆ R ℓ ≤ N +1 . Substituting the preceding inequality in the above equation, we obtain:

<!-- formula-not-decoded -->

By Theorem B.6, when event A c ℓ and B ℓ happens at the same time, we have ˜ R ℓ ( S ℓ ) ≥ ˜ R ℓ ( S ∗ ) ≥ R ( S ∗ , v ) , which implies that

<!-- formula-not-decoded -->

By Theorem B.8, we have

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

where C = max ( C 1 , C 2 ) . And it follows that

<!-- formula-not-decoded -->

Therefore, from the probability we have derived above:

̸

<!-- formula-not-decoded -->

Inequality (a) follows from the observation that L ≤ T , T i ≤ T ,

<!-- formula-not-decoded -->

Inequality (b) follows from Jensen's inequality.

Whereas inequality (c) follows from

<!-- formula-not-decoded -->

For any realization of L, E ℓ , T i , S ℓ , we have the following relation:

<!-- formula-not-decoded -->

Hence, we have E π ( ∑ L ℓ =1 n ℓ ) ≤ T . Let F denote the filtration corresponding to the offered assortments S 1 , . . . , S L ; then by the law of total expectation, we have:

<!-- formula-not-decoded -->

Therefore, it follows that:

<!-- formula-not-decoded -->

To get the worst-case upper bound, we maximize the bound subject to the above condition. Thus we have

̸

<!-- formula-not-decoded -->

Since we set α ∈ [0 , 1 2 ] , then we have

<!-- formula-not-decoded -->

## B.2 Inference Error of Attraction Parameter

Now, let's focus on the estimation error of attraction parameters, i.e.

<!-- formula-not-decoded -->

where φ is a MNL-Bandit instance and i, j ∈ [ N ]

.

First, define SumV ℓ ( i ) := ℓ · v i . Then we propose an IPW estimator of SumV ℓ ( i ) :

<!-- formula-not-decoded -->

where ̂ SumV 0 ( i ) = 0 for all i ∈ [ N ] and ̂ v i,ℓ is the estimation of v i in epoch ℓ that we have defined above. Then we can compute:

<!-- formula-not-decoded -->

So we can easily derive

Since

<!-- formula-not-decoded -->

So, M i ℓ is a martingale. And E [ M i ℓ ℓ ] = E [ ̂ v i -v i ] is the estimation error of v i .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which means that ̂ SumV ℓ ( i ) ℓ is an unbiased estimator of v i . Define:

<!-- formula-not-decoded -->

Then the variance of M i L can be written as

<!-- formula-not-decoded -->

where inequality (e) follows that ̂ v i,ℓ is a geometric random variable with parameter p i := 1 1+ v i which implies:

<!-- formula-not-decoded -->

inequality (f) follows that:

<!-- formula-not-decoded -->

and inequality (g) follows that:

<!-- formula-not-decoded -->

Then to apply Bernstein's Inequality, we further note that

<!-- formula-not-decoded -->

Therefore, by Bernstein's Inequality, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

both sides divided by L we have:

<!-- formula-not-decoded -->

taking δ = 1 L 2 we have:

<!-- formula-not-decoded -->

According to the algorithm, L is defined as the total number of epochs within time T, i.e. L is the minimum number for which ∑ L +1 ℓ =1 |E ℓ | ≥ T , so E π ( ∑ L +1 ℓ =1 |E ℓ | | L ) = ( L +1) · E π ( |E ℓ | ) ≥ T , which follows that T L +1 ≤ E π ( |E ℓ | ) = 1 + ∑ i ∈ S ℓ v i ≤ 1 + N . So

<!-- formula-not-decoded -->

## B.3 Inference Error of Expected Revenue

Here we use the estimates of attraction parameters to estimate the expected revenue. And the estimation error is defined as:

<!-- formula-not-decoded -->

where φ is a MNL-Bandit instance and τ ∈ [ |S| ] .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define:

then we have:

Since then we have:

<!-- formula-not-decoded -->

Since we have already proved E [ | ̂ v i -v i | ] = O ( √ T α -1 ) , thus

<!-- formula-not-decoded -->

## C Discussion on the Assortment Size

There may be concerns that our algorithm implicitly requires the maximum assortment size K to satisfy

<!-- formula-not-decoded -->

since at each epoch we offer either the optimal assortment S ∗ or its complement ( S ∗ ) c = [ N ] \ S ∗ . This departure from the conventional assumption K ≪ N could restrict the applicability of our method in scenarios where only much smaller assortments are feasible.

## Algorithm 2 MNLEXPERIMENTUCB WITH K ∗ CONSTRAINTS

- 1: Input: Collection of assortments S , total time steps T , and exploration parameter α ∈ [0 , 1 2 ] .
- 2: Initialization: v UCB i, 0 = 1 , ̂ SumV 0 ( i ) = 0 , ∀ i ∈ [ N ];
- 3: t = 1 , ℓ = 1 keeps track of time steps and total epochs respectively and α ℓ = 1 ⌈ N K ∗ ⌉· ℓ α
- 4: while t &lt; T do
- 5: Compute S ∗ ℓ := arg max S ∈S ˜ R ℓ ( S ) ,

<!-- formula-not-decoded -->

where ( S ∗ ℓ ) c is the collection of items not in S ∗ ℓ .

- 6: Offer S ℓ and observe customer decision c t .
- 7: E ℓ ←E ℓ ∪ { t } keeps track of time steps in epoch ℓ ;
- 8: if c t = 0 then
- 9: compute ̂ v i,ℓ = ∑ t ∈E ℓ 1 ( c t = i ) , the number of consumers who chose i in epoch ℓ ;
- 10: update T i ( ℓ ) = { τ ≤ ℓ | i ∈ S τ } , T i ( ℓ ) = |T i ( ℓ ) | ;
- 11: update v i,ℓ = ( ∑ τ ∈T i ( ℓ ) ̂ v i,τ ) /T i ( ℓ ) , the sample mean of estimates;
- 12: ̂ SumV ℓ ( i ) = ̂ SumV ℓ -1 ( i ) + ̂ v i,ℓ · 1 ( i ∈ S ℓ ) P ( i ∈ S ℓ | S ∗ ℓ ) ;

<!-- formula-not-decoded -->

- 14: end if
- 15: t ← t +1
- 16: end while

<!-- formula-not-decoded -->

In case of restriction on the maximum assortment size, i.e. K ≤ K ∗ , we can adjust the rule of choosing assortment in Algorithm 1 as below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At the start of each epoch, we divide the products in ( S ∗ ℓ ) c into ⌈ N -| S ∗ ℓ | K ∗ ⌉ assortments, denoted as ( S ∗ ℓ ) c i , i ∈ { 1 , ..., ⌈ N -| S ∗ ℓ | K ∗ ⌉} , where each assortments contains at most K ∗ products and each product occurs in only one assortment.

By this adjustment, we can achieve similar result in MNL bandit and satisfy the restriction on the maximum assortment size without other modification to Algorithm 1. This shows the flexibility of our algorithm. The adjusted algorithm for this case is shown below as Algorithm 2. And we can show that Algorithm 2 is also Pareto optimal.

Theorem C.1. For any instance v = ( v 0 , . . . , v N ) of the MNL-Bandit problem with N items, r i ∈ [0 , 1] , and given the problem assumption, let Algorithm 2 run with α ∈ [0 , 1 2 ] the regret at any time T is O ( √ NT log NT + N log 2 NT + NT 1 -α ) .

Proof.

<!-- formula-not-decoded -->

The probability of a no-purchase conditioned on S ℓ is given by:

<!-- formula-not-decoded -->

So,

<!-- formula-not-decoded -->

Thus, by the formula of full probability, we have

<!-- formula-not-decoded -->

Then define ∆ R ℓ as:

<!-- formula-not-decoded -->

Define bad event:

<!-- formula-not-decoded -->

Then according to Theorem B.4 we have:

<!-- formula-not-decoded -->

Define event B ℓ as:

<!-- formula-not-decoded -->

Because both A ℓ and B C ℓ are 'low-probability' events, we can break down the regret in one epoch as follows:

<!-- formula-not-decoded -->

Using the fact that R ( S ∗ , v ) and R ( S ℓ , v ) are both bounded by 1 and V ( S ℓ ) ≤ N , we have ∆ R ℓ ≤ N +1 . Substituting the preceding inequality in the above equation, we obtain:

<!-- formula-not-decoded -->

By Theorem B.6, when event A c ℓ and B ℓ happens at the same time, we have ˜ R ℓ ( S ℓ ) ≥ ˜ R ℓ ( S ∗ ) ≥ R ( S ∗ , v ) , which implies that

<!-- formula-not-decoded -->

By Theorem B.8, we have

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

where C ′ = max ( C 1 , C 2 ) . And it follows that

<!-- formula-not-decoded -->

Therefore, from the probability we have derived above:

̸

<!-- formula-not-decoded -->

Inequality (a) follows from the observation that L ≤ T , T i ≤ T ,

<!-- formula-not-decoded -->

Inequality (b) follows from Jensen's inequality.

Whereas inequality (c) follows from

<!-- formula-not-decoded -->

For any realization of L, E ℓ , T i , S ℓ , we have the following relation:

<!-- formula-not-decoded -->

Hence, we have E π ( ∑ L ℓ =1 n ℓ ) ≤ T . Let F denote the filtration corresponding to the offered assortments S 1 , . . . , S L ; then by the law of total expectation, we have:

<!-- formula-not-decoded -->

Therefore, it follows that:

<!-- formula-not-decoded -->

## Algorithm 3 MNLExperimentUCB with General Parameters

- 1: Input: Collection of assortments S , total time steps T , and exploration parameter α ∈ [0 , 1 2 ] .
- 2: Initialization: v UCB 2 i, 0 = 1 , ̂ SumV 0 ( i ) = 0 , ∀ i ∈ [ N ];
- 3: t = 1 , ℓ = 1 keeps track of time steps and total epochs respectively and α ℓ = 1 α
- 4: while t &lt; T do
- 5: Compute S ∗ ℓ := arg max S ∈S ˜ R ℓ ( S ) ,

<!-- formula-not-decoded -->

where ( S ∗ ℓ ) c is the collection of items not in S ∗ ℓ . √

- 7: Define ̂ S = { i | T i ( ℓ ) &lt; 48 log( Nℓ +1) } .
- 6: if T i ( ℓ ) &lt; 48 log( Nℓ +1) for some i ∈ S ℓ then √
- 8: Choose S ℓ ∈ S such that S ℓ ⊂ ̂ S .
- 9: end if
- 10: Offer S ℓ and observe customer decision c t .
- 11: E ℓ ←E ℓ ∪ { t } keeps track of time steps in epoch ℓ ;
- 12: if c t = 0 then
- 13: compute ̂ v i,ℓ = ∑ t ∈E ℓ 1 ( c t = i ) , the number of consumers who chose i in epoch ℓ ;
- 14: update T i ( ℓ ) = { τ ≤ ℓ | i ∈ S τ } , T i ( ℓ ) = |T i ( ℓ ) | , the number of epochs until ℓ that offered item i;
- 15: update v i,ℓ = ( ∑ τ ∈T i ( ℓ ) ̂ v i,τ ) /T i ( ℓ ) , the sample mean of estimates;

<!-- formula-not-decoded -->

update

17:

v

- 18: end if
- 19: t ← t +1

<!-- formula-not-decoded -->

To get the worst-case upper bound, we maximize the bound subject to the above condition. Thus we have

̸

<!-- formula-not-decoded -->

Since we set α ∈ [0 , 1 2 ] , then we have

<!-- formula-not-decoded -->

As shown above, the regret of Algorithm 2 (MNLExperimentUCB with K ∗ constraints) with α ∈ [0 , 1 2 ) at any time T is still O ( √ NT log NT + N log 2 NT + NT 1 -α ) . And since our adjustment does not change the estimators, the inference error won't change. Therefore, Algorithm 2 is Approximately Pareto Optimal.

## D Relaxing the No-Purchasing Assumption

In this section, we release the assumption v i ≤ v 0 , ∀ i ∈ [ N ] . We provide an algorithm based on Algorithm 1 for this setting to achieve Approximate Pareto Optimality and give rigorous proof of the regret upper bound. We first prove the initial exploratory phase is bounded.

UCB

i,ℓ

2

=

v

+max

{

√

v

, v

}

√

48 log(

T

√

i

Nℓ

(

ℓ

)

+1)

+

48 log(

T

√

i

Nℓ

(

ℓ

)

i,ℓ

i,ℓ

i,ℓ

- 2 ℓ

+1)

,

ℓ

=

ℓ

+1

Lemma D.1. Let L be the total number of epochs in our Algorithm, and let E ℓ denote the set of time steps in the exploratory epochs:

<!-- formula-not-decoded -->

where T i ( ℓ ) is the number of epochs item i has been offered before epoch ℓ . If S E L denote the time steps corresponding to epoch ℓ and v i ≤ Bv 0 for all i for some B ≥ 1 , then we have:

<!-- formula-not-decoded -->

where the expectation is over all possible outcomes of the algorithm.

Proof. Consider ℓ ∈ E L , |E ℓ | is a geometric random variable with parameter v 0 V ( S ℓ )+ v 0 .

Since v i ≤ Bv 0 for all i , we can assume W.L.O.G that v 0 = 1 , and thus |E ℓ | is a geometric random variable with parameter p ≥ v 0 B |S ℓ | + v 0 = 1 B |S ℓ | +1 .

Thus,

Combining (1) and (2), we have:

<!-- formula-not-decoded -->

Then we prove v UCB2 i,ℓ as an upper bound converging to v i has the following results:

Lemma D.2. For every epoch ℓ , if T i ( ℓ ) ≥ 48 log( √ Nℓ +1) for all i ∈ S ℓ , then:

1. v UCB2 i,ℓ ≥ v i with probability at least 1 -6 Nℓ for all i = 1 , . . . , N .
2. There exist constants C 1 and C 2 such that:

<!-- formula-not-decoded -->

with probability at least 1 -7 Nℓ .

Lemma D.3. If in epoch ℓ , T i ( ℓ ) ≥ 48 log( √ Nℓ +1) for all i ∈ S ℓ , then we have the following concentration bounds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to our algorithm setting, after every item has been offered in at least 48 log NT epochs, we do not have any exploratory epochs. Therefore:

<!-- formula-not-decoded -->

Lemma D.4. Suppose S ∗ ∈ S is the assortment with the highest expected revenue, and the algorithm offers S ℓ = S ∗ ( ℓ ) in epoch ℓ . Furthermore, if T i ( ℓ ) ≥ 48 log( √ Nℓ +1) for all i ∈ S ℓ , then we have:

<!-- formula-not-decoded -->

Lemma D.5. For every epoch ℓ , if r i ∈ [0 , 1] and T i ( ℓ ) ≥ 48 log( √ Nℓ +1) for all i ∈ S ℓ , then there exist constants C 1 and C 2 such that for every ℓ , we have:

with probability at least 1 -6 Nℓ .

<!-- formula-not-decoded -->

with probability at least 1 -13 Nℓ .

Theorem D.6. For any instance v = ( v 0 , . . . , v N ) of the MNL-Bandit problem with N items, r i ∈ [0 , 1] , and given the adjusted assumption , let Algorithm 3 run with α ∈ [0 , 1 2 ] . The regret at any time T is O ( CNB · log 2 ( NT ) + √ BNT log NT + NB · T 1 -α ) .

Proof. Putting it all together to prove the regret of Algorithm 2:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any S , R ( S, v ) ≤ R ( S ∗ , v ) ≤ 1 , so it follows that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the sake of brevity, we define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then:

Let T i denote the total number of epochs that offered an assortment containing item i . For all ℓ = 1 , . . . , L , define event B ℓ as (bad event):

<!-- formula-not-decoded -->

then

<!-- formula-not-decoded -->

Then define A ℓ for all ℓ = 1 , ..., L as:

then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we can break down the regret (in one epoch) as follows:

<!-- formula-not-decoded -->

where (g) follows that

<!-- formula-not-decoded -->

where C is a constant and C ≥ max { C 1 , C 2 } . Define φ = { i : v i ≥ 1 } , D = { i : v i &lt; 1 } . Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where inequality (h) follows that L, T i ≤ T , ∑ T i T i ( ℓ )=1 1 √ T i ( ℓ ) ≤ √ T i and ∑ T i T i ( ℓ )=1 1 T i ( ℓ ) ≤ log T i and inequality (i) follows Jensen's Inequality. And we have ∑ i v i E π ( T i ) ≤ T . Then we have:

̸

<!-- formula-not-decoded -->

where inequality (j) follows that the maximizing objective is concave so that we can use the Karush-Kuhn-Tucker conditions to derive the worst-case bound.

As shown above, when α ∈ [0 , 1 2 ) , the regret of Algorithm 2 is

<!-- formula-not-decoded -->

And the analysis of estimation error is the same as that in subsection B.2 and B.3. Therefore, we can derive that Algorithm 2 is also Approximately Pareto Optimal.

## E Technical Lemmas

Theorem E.1 (Bernstein's Inequality) . Let X 1 , X 2 , . . . be a martingale difference sequence, such that | X t | ≤ α t for a non-decreasing deterministic sequence α 1 , α 2 , . . . with probability 1. Let M t := ∑ t τ =1 X τ be a martingale. Let V 1 , V 2 , . . . be a deterministic upper bound on the variance V t := ∑ t τ =1 E [ X 2 τ | X 1 , . . . , X τ -1 ] of the martingale M t , such that V t -s satisfies √ ln ( 2 δ ) ( e -2) V t ≤ 1 α t . Then, with probability greater than 1 -δ for all t :

<!-- formula-not-decoded -->

Theorem E.2 (Neyman-Pearson Lemma) . Let P 0 and P 1 be two probability measures. Then for any test ψ , it holds

<!-- formula-not-decoded -->

Moreover, the equality holds for the Likelihood Ratio test ψ ⋆ = I ( p 1 ≥ p 0 ) .

## Corollary E.3.

<!-- formula-not-decoded -->

Proof. Denote that P 0 and P 1 are defined on the probability space ( X , A ) . By the definition of the total variation distance, we have

<!-- formula-not-decoded -->

where the last equality applies the Neyman-Pearson Lemma, and the fourth equality holds due to the fact that

<!-- formula-not-decoded -->

Theorem E.4 (Pinsker's Inequality) . Let P 1 and P 2 be two probability measures such that P 1 ≪ P 2 . Then,

<!-- formula-not-decoded -->

## F Source Code

The source code used in the numerical experiment is available at the following anonymous link for review purposes: https://anonymous.4open.science/r/MNL-61CD