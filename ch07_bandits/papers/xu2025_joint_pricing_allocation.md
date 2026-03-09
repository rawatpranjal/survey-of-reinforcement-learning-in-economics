## Joint Pricing and Resource Allocation: An Optimal Online-Learning Approach

Jianyu Xu 1,2 , Xuan Wang 2 , Yu-Xiang Wang 3 , and Jiashuo Jiang 2

1 Carnegie Mellon University, PA 2

Hong Kong University of Science and Technology, Hong Kong SAR 3 University of California San Diego, CA

May 23, 2025

## Abstract

We study an online learning problem on dynamic pricing and resource allocation, where we make joint pricing and inventory decisions to maximize the overall net profit. We consider the stochastic dependence of demands on the price, which complicates the resource allocation process and introduces significant non-convexity and non-smoothness to the problem. To solve this problem, we develop an efficient algorithm that utilizes a 'Lower-Confidence Bound (LCB)' meta-strategy over multiple OCO agents. Our algorithm achieves ˜ O ( √ Tmn ) regret (for m suppliers and n consumers), which is optimal with respect to the time horizon T . Our results illustrate an effective integration of statistical learning methodologies with complex operations research problems.

## 1 Introduction

The problem of dynamic pricing examines strategies of setting and adjusting prices in response to varying customer behaviors and market conditions. The mainstream of existing works on dynamic pricing, including Kleinberg and Leighton (2003); Broder and Rusmevichientong (2012); Cohen et al. (2020); Wang et al. (2021b), focuses on the estimation of demand curves while putting aside the decisions on the supply side. Another series of literature, including Besbes and Zeevi (2009); Chen et al. (2019, 2021a); Keskin et al. (2022), takes supply and inventories into account. However, these works simplify the supply cost as uniform and static, underestimating the difficulty of allocating products through sophisticated supply chains among multiple parties such as factories, warehouses, and retailers.

On the other hand, the problem of resource allocation - to serve different demand classes with various types of resources - presents a complex challenge within the field of operations research. Analogous to online dynamic pricing, the recent proliferation of e-platforms has magnified the importance of developing online allocation algorithms that efficiently manage supply and demand on the fly while maximizing cumulative utilities. However, traditional approaches in resource allocation are insufficient in depicting scenarios where the demand is stochastic and dependent on the price (Hwang et al., 2021). Therefore, it is critical to develop price-dependent online allocation models and methodologies that can simultaneously learn the demand curve and optimize the joint decisions on price, inventory, and allocation.

This work introduces a novel framework for tackling the online pricing and allocation problem with an emphasis on learning under uncertainty. More specifically, we consider a problem setting where both the price and inventory decisions are made at the beginning of each time period, followed by inventory allocation based on the realized price-dependent stochastic demands during that period. We summarize the proposed framework as follows:

Pricing and Allocation. For t = 1 , 2 , ..., T :

1. Determine the inventories of m suppliers as /vector I := [ I 1 , I 2 , . . . , I m ] /latticetop and incur an immediate inventory cost ∑ m i =1 γ i · I i . 2. Propose a price p t for all n consumers.
3. Based on the price p t , consumers generate their demands as /vector D := [ D 1 , D 2 , . . . , D n ] /latticetop .
4. We allocate inventories /vector I to satisfy demands /vector D . The allocation from Supplier i to Consumer j is denoted as X i,j . The total supplying cost is m i =1 n j =1 C i,j · X i,j .

We assume that the inventories are perishable and the leftover inventory cannot be carried over to the following period. Furthermore, we assume that the price is identical for all consumers. We formalize the above process as solving the following two-stage stochastic

- ∑ ∑ 5. We receive a payment from the consumers as ∑ m i =1 ∑ n j =1 p t · X i,j in total.

programming problem:

<!-- formula-not-decoded -->

Here g ( /vector I, p, /vector D ) is the minimum negative net profit (or loss ) under the best allocation scheme given inventories /vector I , price p , and realized demand /vector D (see Eq. (2) for its definition).

## 1.1 Summary of Results.

In this work, we establish a novel framework for solving the online pricing and allocation problem under demand uncertainties. Our main contributions are twofold:

1. Algorithmic Design against Non-Convexity . We propose an efficient onlinelearning algorithm for the (price,inventory) joint decision problem, which is highly non-convex . To navigate to the global optimal decisions among many sub-optimals, our algorithm incorporates an optimistic meta-algorithm to manage multiple online convex optimization (OCO) agents working locally.
2. Regret Analysis . We show that our algorithm achieves ˜ O ( √ Tmn ) regret, which is optimal with respect to T as it matches the information-theoretic lower bound in (Broder and Rusmevichientong, 2012).

## 1.2 Statement of Novelty.

To the best of our knowledge, we are the first to study dynamic pricing and inventory control under the framework of online resource allocation with uncertainty. Some existing works, such as Wang et al. (2021b); Chen and Gallego (2021), focus on related topics. However, their approaches are not capable of overcoming the significant challenges in our problem setting, including the local convexity, multiple suboptimals, and non-smoothness. As a consequence, we develop new techniques to solve the problem.

Confidence Bounds: Local (Vertical) and Interval (Horizontal) Perspectives Our setting poses two main challenges: bandit feedback and non-convexity with respect to price p . Existing dynamic pricing methods address these with continuum-armed bandit algorithms, but these typically yield O ( T 2 / 3 ) regret (Kleinberg, 2004), which is not optimal in this scenario. To improve upon this, we leverage the piecewise-convex structure of the expected loss: The domain [0 , p max ] can be divided into intervals, in each of which the function is locally convex. We assign a dedicated agent to each interval, applying online convex optimization (OCO) to approach its local optimal. We introduce twofold confidence bounds for each agent:

- (a) Vertical confidence bound : This quantifies the uncertainty in the agent's estimate of the local optimal value within its convex interval. It is updated at the end of each sub-epoch . (We provide the definitions of sub-epoch and epoch in Section 4).

- (b) Horizontal confidence bound : This describes the shrinking search space within the interval where the local optimum is likely to be found. It is updated at the end of each epoch .

As the algorithm proceeds, either the vertical confidence bound (value uncertainty) or the horizontal confidence bound (location uncertainty) for each agent becomes tighter, thereby improving both the accuracy of local optimization and the efficiency of interval selection. This hierarchical design enables us to achieve an improved regret bound of O ( √ T ) .

Lower-confidence-bound (LCB) strategy over agents. In order to distinguish among every local optimum, we develop an LCB meta-algorithm over these local OCO agents. With the vertical confidence bound of each agent, we obtain the least possible value in each interval and select the agent with the least lower bound to run in the following subepoch.

The remainder of this paper is organized as follows. In Section 2, we review the relevant literature, contrasting existing approaches with ours. Section 3 lays out the preliminaries, including notation and problem setup. In Section 4, we introduce our learning-based algorithm and highlight its novel structures. Section 5 provides the regret analysis and theoretical guarantees. We finally discuss potential extensions in Section 6 and conclude this paper in Section 7.

## 2 Related Works

In this section, we present a review of the pertinent literature on pricing, allocation, and online convex optimization (OCO) problems, aiming to position our work within the context of related studies.

Dynamic Pricing. Quantitative research on dynamic pricing dates back to Cournot (1897) and has attracted significant attention in the field of machine learning (Leme et al., 2021; Xu and Wang, 2021; Jia et al., 2022; Choi et al., 2023; Simchi-Levi and Wang, 2023). For single-product pricing problems, the crux is to learn the demand curve and approach the optimal price. Under the assumptions of bandit feedback and k th -smooth demand curves, Wang et al. (2021a) achieves an O ( T k +1 2 k +1 ) regret. However, their methodologies are not applicable to our setting: Our objective function is only Lipschitz continuous, leading to k = 1 and an O ( T 2 / 3 ) sub-optimal regret. In contrast, the piecewise convex property in our problem enables advanced methods to achieve a better regret. Another stream of works considers the heterogeneity of pricing processes, which includes item-wise features (Javanmard and Nazerzadeh, 2019; Cohen et al., 2020; Luo et al., 2021; Xu and Wang, 2022; Fan et al., 2021; Luo et al., 2022; Tullii et al., 2024), heteroscedasticity of valuations (Wang et al., 2021a; Ban and Keskin, 2021; Xu and Wang, 2024), time-

instationarity (Leme et al., 2021; Baby et al., 2023) and price discrimination (Chen et al., 2021c; Cohen et al., 2021; Eyster et al., 2021; Cohen et al., 2022; Xu et al., 2023; Karan et al., 2024). However, most of them are analyzing the differences (and their down-streaming effects) either in price or in demands, instead of the allocation process as we are concerned in this paper.

Pricing and Inventory Co-Decisions. The incorporation of inventory constraints into dynamic pricing problems began with the work of Besbes and Zeevi (2009) which assumed a fixed initial stock, and decisions of replenishment were later allowed in Chen et al. (2019). More recent studies, including Chen et al. (2020); Keskin et al. (2022), assumed perishable goods and took inventory costs into account. The stream of work by Chen et al. (2021a, 2023) further assumed the inventory-censoring effect on demands. However, none of them consider the heterogeneity of supply, nor the impact of prices on the allocation process. In our work, we not only model the inventory cost of each warehouse individually, but also depict the unit supplying cost from Warehouse i to Consumer j as a unique coefficient C i,j .

Resource Allocation. There is a broad literature on the study of resource allocation and various policies have been derived under various settings (e.g. Reiman and Wang 2008; Jasin and Kumar 2012; Ferreira et al. 2018; Asadpour et al. 2020; Bumpensanti and Wang 2020; Vera and Banerjee 2019; Jiang et al. 2022). Notably, the intersection of pricing and resource allocation has also been studied, for example, in Chen et al. (2021b) and Vera et al. (2021). However, previous works have primarily focused on either the allocation decision or the pricing decision separately. In contrast, in our paper, we consider a two-stage process where we first make the pricing decision which affects the demand, and then make the allocation decision. This feature distinguishes our paper from previous works on (pricebased) resource allocation.

Online Convex Optimization (OCO). OCO models a scenario where decisions are made iteratively, facing a series of convex loss functions, with the objective of minimizing cumulative regret over time (Shalev-Shwartz et al., 2012) or within certain budgets (Jenatton et al., 2016). In our work, we adopt zeroth-order methods in Agarwal et al. (2011) when we iterate within each local convexity interval. For a detailed review of classic and contemporary results on OCO, we kindly refer readers to Hazan (2016).

## 3 Preliminaries

In this section, we rigorously define the problem we are studying. We firstly formulate the offline version of the problem as a two-stage stochastic program in Section 3.1. We then develop the formulation of the online version in Section 3.2, where the demand parameters

are unknown. Finally, we present assumptions that are crucial to our online algorithm design by the end of this section.

## 3.1 Offline Problem Setting

We consider the following scenario where a retail company makes their decisions with the goal of maximizing their net profit. Suppose that the company has m warehouses and n retailers, producing and selling identical products. In general, this company faces the following three problems on inventory, pricing, and allocation:

1. What are the appropriate quantities each warehouse should load?
2. What is the optimal price that retailers should set?
3. How to allocate inventories from warehouses to stores under heterogeneous supply costs?

To address these questions, we model the problem as a two-stage stochastic program.

- (i) In Stage 1, the company makes inventory decisions /vector I = [ I 1 , I 2 , . . . , I m ] /latticetop , where I i is the inventory level of Warehouse i . Each unit of inventory at Warehouse i incurs an inventory cost γ i . In addition, the company decides a uniform price p for the products.
- (ii) In Stage 2, a stochastic demand /vector D = [ D 1 , D 2 , . . . , D n ] /latticetop is generated based on the price p , where D j represents the demand at Retailer j . Then the products are allocated from warehouses to stores in order to fulfill the realized demand. Each unit of supply from Warehouse i to Retailer j incurs an allocation cost C i,j , and each unit of fulfilled demand increases the total revenue by p .

The company aims to make the best (inventory, price) joint decisions that maximize their net profit, which can be formulated as the following optimization problem (where we equivalently minimize the negative net profit):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Here X i,j represents the quantity of inventories allocated from Warehouse i to Retailer j , and g ( /vector I, p, /vector D ) is the optimal objective value of the second-stage problem, which optimally allocates inventory /vector I to demand /vector D based on price p and cost parameters { C i,j } .

It is worth noting that the distribution of demand /vector D is dependent on the price p , and the minimization over p and /vector I takes this into account. However, the solution to g ( /vector I, p, /vector D ) is not relevant to this dependence, as it is solved after the realization of /vector D .

We denote an optimal solution to Eq. (1) as ( p ∗ , /vector I ∗ ) , and denote an optimal solution to Eq. (2) as X ∗ . Since Eq. (2) is a linear programming, we can solve it directly with any standard optimization tool. However, in order to solve Eq. (1), we have to know the distribution of /vector D and how it is dependent on the price p , both of which are not directly accessible from the seller's side as they do not have the full knowledge of the entire market. In the next subsection, we will discuss how we can 'learn' the demand distribution function under mild assumptions.

Now we propose a lemma that states the marginal convexities of g ( /vector I, p, /vector D ) .

Lemma 3.1. The function g ( /vector I, p, /vector D ) defined in Eq. (2) is marginally convex on /vector I and on /vector D .

The key to proving Lemma 3.1 is to show that g ( /vector I, p, /vector D ) is the piecewise maximum over a group of linear functions. Please refer to Appendix B.1 for details.

## 3.2 Online Problem Setting

Due to insufficient knowledge of the actual demand distribution, the company could propose pairs of ( p, /vector I ) that are suboptimal, leading to lower net profits compared to the optimal solution. However, the company has observations on the realized demand at each store, which enables them to estimate demand and subsequently improve their decision-making. In what follows, we study the online decision-making problem of setting prices and managing inventory.

Denote p t , /vector I t and /vector D t as the price, inventory and realized demand in each time period t = 1 , 2 , . . . , T , respectively. We make the following semi-parametric assumption on the demand model.

Assumption 3.2. Assume the realized demand is linear and noisy . Specifically, assume

<!-- formula-not-decoded -->

Here /vector a, /vector b, /vector N t ∈ R n are the base demand, the price sensitivity parameter, and the market noise of the retailers' demand, respectively. Assume /vector a, /vector b are fixed, and /vector N t are samples drawn from identical and independent distributions (i.i.d.) over time t , such that E [ /vector N t ] = /vector 0 .

## Algorithm 1 Sort tuples { ( i, j ) } i = m,j = n i =1 ,j =1 as follows:

- 1: Input: { ( i, j ) } tuples.
- 3: If C i,j &lt; C i ′ ,j ′ , then ( i, j ) ≺ ( i ′ , j ′ ) .
- 2: for each different pairs of tuples ( i, j ) and ( i ′ , j ′ ) do
- 4: If C i,j = C i ′ .j ′ and i &gt; i ′ , then ( i, j ) ≺ ( i ′ , j ′ ) .
- 6: end for
- 5: If C i,j = C i ′ ,j ′ , i = i ′ and j &gt; j ′ , then ( i, j ) ≺ ( i ′ , j ′ ) .
- 7: Output: { ( i k , j k ) } mn k =1 .

Given the linear-and-noisy demand model in (3), we define the cost function that we aim to minimize. Denote

<!-- formula-not-decoded -->

Since g ( /vector I, p, /vector D ) is marginally convex on /vector I according to Lemma 3.1, we know that Q ( /vector I, p ) and Q t ( /vector I, p ) are also marginally convex on /vector I . But what about their marginal behaviors on p ? We state in the following lemma:

Lemma 3.3. Sort { C i,j } m,n i =1 ,j =1 according to Algorithm 1. Denote C i 0 ,j 0 = 0 and C i mn +1 ,j mn +1 = p max . For any K ∈ { 0 } ∪ [ mn ] , function Q t ( /vector I, p ) is Lipschitz and marginally convex on p in range [ C i K ,j K , C i K +1 ,j K +1 ] .

We defer the proof of Lemma 3.3 to Appendix B.2. Furthermore, we have the following results.

Lemma 3.4. Define an optimistic cost function W ( p ) :

<!-- formula-not-decoded -->

We have W ( p ) is L W -Lipschitz where L W is a constant. Also, for any K ∈ { 0 } ∪ [ mn ] , the function W ( p ) is convex in the range [ C i K ,j K , C i K +1 ,j K +1 ] .

The proof of Lemma 3.4 is relegated to Appendix B.3. Finally, we define regret as the relative loss of net profit compared to that achieved by optimal decisions.

Definition 3.5 (Regret) . At each time t = 1 , 2 , . . . , T , denote /vector I t and p t as inventory and price decisions, respectively. Define

<!-- formula-not-decoded -->

as the regret of decision sequence { ( /vector I t , p t ) } T t =1 .

Before we conclude this section, we present some crucial assumptions for our algorithm design.

Assumption 3.6 (Boundedness) . We assume boundedness on the norms of /vector γ,/vector a, /vector b, /vector I and on price p . Specifically, there exist constants γ max , a max , b max , I max , p max such that ‖ /vector γ ‖ ∞ ≤ γ max , ‖ /vector a ‖ ∞ ≤ a max , ‖ /vector b ‖ ∞ ≤ b max , ‖ /vector I ‖ 1 ≤ I max and p ∈ [0 , p max ] . Without loss of generality, we assume γ max , a max , b max , I max , p max ≥ 1 .

Assumption 3.7 (Knowledge over parameters) . We have full knowledge on the problem parameters /vector γ, { C i,j } m,n i =1 ,j =1 and the boundedness parameters γ max , a max , b max , I max , p max before t = 0 . We do not know the model parameters /vector a, /vector b nor the distribution of /vector N t .

## 4 Algorithm

In this section, we present an online learning algorithm that proposes asymptotically optimal (inventory, price) decisions.

## 4.1 Algorithm Design Overview

We design a hierarchical algorithm to solve this problem. Algorithm 2 serves as the main algorithm. We firstly initialize ( mn +1) agents A K for K = 0 , 1 , . . . , mn by running each of them for 3 times. After initialization, every time we select the ˆ K -th agent A ˆ K whose Lower Confidence Bound LCB ˆ K is the minimum, and run A ˆ K for a period of time (depending on the stage in which it is). The status of each A K can be divided into three stages:

- (i) In Stage 1 (presented as Algorithm 3), we search for the local optimal decision within each A K 's domain.
- (ii) In Stage 2 (presented as Algorithm 4), we gather sufficient number of samples to ensure the confidence bound of A K is converging at a proper rate.
- (iii) In Stage 3 (presented as Algorithm 5), we do pure exploitation on the local optimal of A K , while also updating the confidence bound.

In the following subsections, we will introduce each component of our algorithm design (including the meta-algorithm and each OCO agent) in details.

## 4.2 Meta-Algorithm: a Lower-Confidence-Bound (LCB) Strategy

Since we have full-information feedback over any decision w.r.t. /vector I , we may always propose greedy inventories without causing bias. However, we only have bandit feedback w.r.t. the

## Algorithm 2 LCB Meta Algorithm

```
1: Input: m,n,T , sorted supply costs { C i k ,j k } mn +1 k =0 . Global parameters δ K , n 0 . 2: Initialization: ˆ W K := 0 , ∆ K := + ∞ , LCB K := -∞ , K = 0 , 1 , 2 , . . . , mn . 3: for K = 0 , 1 , 2 , . . . , mn do 4: Initialize local agent A K as follows: 5: Let L K, 1 = C i K ,j K , U K, 1 = C i K +1 ,j K +1 , and a K, 1 = 3 L K, 1 + U K, 1 4 , c K, 1 = L K, 1 + U K, 1 2 , b K, 1 = L K, 1 +3 U K, 1 4 . 6: for p = a K, 1 , b k, 1 , c k, 1 , do 7: Propose inventory /vector I 0 := [1 , 1 , . . . , 1] /latticetop and price p . 8: Record the marginal function of /vector I as Q 0 ( /vector I, p ) . 9: Find I K, 1 , 0 ( p ) := argmin /vector I Q 0 ( /vector I, p ) . 10: end for 11: Set the Stage Flag £ K ← 1 for Agent A K . 12: end for 13: while t ≤ T do 14: Let ˆ K := argmin K ∈ [ mn ] ∪ 0 LCB K . 15: if £ ˆ K == 1 then 16: Run Algorithm 3 (Stage 1 of A ˆ K ) for one sub-epoch . 17: t + = | the length of this sub-epoch | . 18: Update ˆ W ˆ K , ∆ ˆ K and £ ˆ K according to the statement of A ˆ K . 19: else if £ ˆ K == 2 then 20: Run Algorithm 4 (Stage 2 of A ˆ K ) until its completion . 21: t + = | the length of this Stage 2 | . 22: Update ˆ W ˆ K , ∆ ˆ K according to the statement of A ˆ K . 23: Update £ ˆ K ← 3 . 24: else if £ ˆ K == 3 then 25: Run Algorithm 5 (Stage 3 of A ˆ K ) for one single time period . 26: t + = 1 . 27: Update ˆ W ˆ K , ∆ ˆ K according to the statement of A ˆ K . 28: end if 29: for K ′ = 0 , 1 , 2 , . . . , mn. do 30: Update LCB K ′ ← ˆ W K ′ -34 · ∆ K ′ . 31: end for 32: end while
```

price p , as we have no direct feedback on the prices we are not proposing. Therefore, we conduct online learning on the optimistic cost function W ( p ) = min /vector I Q ( /vector I, p ) .

Due to the piecewise convexity of W ( p ) , we divide the price range [0 , p max ] into ( mn +1) intervals [ C i K ,j K , C i K +1 ,j K +1 ] , K = 0 , 1 , . . . , mn . Within each interval, we initialize an OCO agent A K that is responsible for converging to the local optimal. However, we cannot run multiple OCO agents simultaneously. Therefore, we require a meta-algorithm that serves as a manager over these agents and determine which A K to run at each time, so as to locate the optimal price with the least regret.

To achieve this, we develop a lower-confidence-bound (LCB) meta-algorithm as shown in Algorithm 2. We firstly ask each A K agent to maintain a confidence bound [ ˆ W K -34 ∆ K , ˆ W K + 34 ∆ K ] of its local optimal. Given this, the meta-algorithm then selects the agent K that minimizes the lower confidence bound. As we further show that ∆ K ≈ O ( √ 1 /T K ) where T K is the total time periods that A K has been running so far, we may upper bound the cumulative regret as O ( √ Tmn ) .

## 4.3 Agent A K : a Zeroth-Order Optimizer

As described in Section 4.2, we divide the price range [0 , p max ] into ( mn + 1) intervals [ C i K ,j K , C i K +1 ,j K +1 ] , K = 0 , 1 , 2 , . . . , mn , within each of which the objective function W ( p ) is convex. We then assign an agent A K to each interval, conducting online convex optimization (OCO) locally. We require the agent A K to learn and converge to the local optimal p ∗ K := argmin p ∈ [ C i K ,j K ,C i K +1 ,j K +1 ] W ( p ) over time, while also maintaining a valid error bar [ ˆ W K -34 ∆ K , ˆ W K + 34 ∆ K ] that contains W ( p ∗ K ) with high probability. To achieve the optimal regret, we rely on the following properties of A K :

- (a) The cumulative sub-regret of A K , i.e. performance suboptimality compared with W ( p ∗ K ) , is bounded by ˜ O ( √ T K ) as an optimal rate of OCO (if we have run A K for T K times so far).
- (b) The error bar ∆ K is bounded by ˜ O ( √ 1 T K ) as a requirement of the meta-algorithm.

We present a detailed introduction of each component of A K in Appendix A.

Technical Novelty We propose a unique methodology undergoing 'horizontal-and-vertical' convergence simultaneously, for the first time. In contrast, existing works adopt either 'vertical convergence' such as bandits algorithms (which allow non-convexity of objective functions but cannot achieve O ( √ T ) regret even with smoothness assumptions), or 'horizontal convergence' which is applicable to many online planning and optimization scenarios but requires global convexity assumptions.

## Algorithm 3 Agent A K Stage 1

```
1: Obtain ˆ W K , ∆ K and £ K from the Meta-Algorithm (Algorithm 2). 2: for Epoch τ = 1 , 2 , . . . , O (log T ) , do 3: Let a K,τ = 3 L K, 1 + U K, 1 4 , c K,τ = L K, 1 + U K, 1 2 , b K,τ = L K, 1 +3 U K, 1 4 . 4: for Sub-epoch s = 1 , 2 , . . . do 5: Let sub-epoch length n s := 2 s 6: Define a flag := 0 for error-bar update. 7: for ˆ p τ = a K,τ , b K,τ , c K,τ do 8: for t = 1 , 2 , . . . , n s do 9: Propose decisions ( /vector I t , p t ) = ( /vector I K,τ,s -1 (ˆ p τ ) , ˆ p τ ) . 10: Observe and record the marginal function Q t ( /vector I, ˆ p τ ) with respect to /vector I . 11: end for 12: Define an aggregated function Q K,τ,s ( /vector I, ˆ p τ ) := 1 n s · ∑ n s t =1 Q t ( /vector I, ˆ p τ ) . 13: Define the empirical optimal inventory I K,τ,s (ˆ p τ ) := argmin /vector I Q K,τ,s ( I, ˆ p τ ) . 14: Denote ˆ Q K,τ,s, ˆ p τ := Q k,τ,s ( I K,τ,s (ˆ p τ ) , ˆ p τ ) , and ∆ K,τ,s := δ K 2 √ n s . 15: end for 16: if ˆ Q K,τ,s,a K,τ > ˆ Q K,τ,s,b K,τ +4 ∆ K,τ,s then 17: Update L K,τ +1 ← a K,τ , U K,τ +1 ← U K,τ , flag ← 1 . 18: else if ˆ Q K,τ,s,a K,τ < ˆ Q K,τ,s,b K,τ -4 ∆ K,τ,s then 19: Update L K,τ +1 ← L K,τ , U K,τ +1 ← b K,τ , flag ← 1 . 20: else if ˆ Q K,τ,s,c K,τ < ˆ Q K,τ,s,a K,τ -4 ∆ K,τ,s then 21: Update L K,τ +1 ← a K,τ , U K,τ +1 ← U K,τ , flag ← 1 . 22: else if ˆ Q K,τ,s,c K,τ < ˆ Q K,τ,s,b K,τ -4 ∆ K,τ,s then 23: Update L K,τ +1 ← L K,τ , b K,τ +1 ← U K,τ , flag ← 1 . 24: end if 25: if flag == 1 then 26: if U K,τ +1 -L K,τ +1 > 1 T then 27: Continue to Epoch τ +1 (without updating ˆ W K or ∆ K ). 28: else 29: Set ˆ p ∗ K ← C K,τ , ˆ /vector I ∗ K, 0 ← /vector I K,τ,s ( C K,τ ) . 30: Update £ K ← 2 and Break (without updating ˆ W K or ∆ K ). 31: end if 32: else if ∆ K,τ,s -1 < ∆ K then 33: Update ∆ K ← ∆ K,τ,s -1 , ˆ W K ← min ˆ p τ ∈{ a K,τ ,c K,τ ,b K,τ } ˆ Q K,τ,s -1 , ˆ p τ . 34: end if 35: end for 36: end for
```

## Algorithm 4 Agent A K Stage 2 (a sub-epoch totally)

- 1: Obtain ˆ W K , ∆ K and £ K from the Meta-Algorithm (Algorithm 2).
- 2: Initialization : Set ˆ p ∗ K := C K,τ , ˆ /vector I ∗ K, 0 = /vector I K,τ,s ( C K,τ ) from Stage 1, and ˇ C K := log 4 / 3 p max +1 .
- 3: if ∆ K &lt; + ∞ then
- 5: else
- 4: Let N K, 2 := 4 δ 2 K ∆ 2 K
- 6: Let N K, 2 := n 0 = 6(log 4 / 3 T + ˇ C K )
- 7: end if
- 8: Let r K := log 2 ( N K, 2 )
- 9: for r = 1 , 2 , . . . , r K do
- 11: Propose decisions ( /vector I t , p t ) = ( ˆ /vector I ∗ K,r -1 , ˆ p ∗ K ) .
- 10: for t = 1 , 2 , . . . , m r := 2 r do
- 12: Observe and record the marginal function Q t ( /vector I, ˆ p ∗ K ) with respect to /vector I .
- 13: end for
- 16: Denote /vector I ∗ K := ˆ /vector I ∗ K,r K , ˆ Q ∗ K := ¯ Q K,r K ( /vector I ∗ K , ˆ p ∗ K ) .
- 14: Set ˆ /vector I ∗ K,r := argmin /vector I ¯ Q K,r ( /vector I, ˆ p ∗ K ) , where ¯ Q K,r ( /vector I, ˆ p ∗ K ) := 1 m r · ∑ /latticetop t =1 Q t ( /vector I, ˆ p ∗ K ) . 15: end for
- 17: Update ˆ W K ← ˆ Q ∗ K -L W · 1 T , ∆ K ← δ K √ N K, 2 and £ K ← 3 .

## Algorithm 5 Agent A K Stage 3 (each t as a sub-epoch)

- 1: Obtain ˆ W K , ∆ K and £ K from the Meta-Algorithm (Algorithm 2).
- 2: Initialization : Set ˆ p ∗ K , ˆ /vector I ∗ K,r K , ¯ Q K,r K ( /vector I, ˆ p ∗ K ) from Stage 2 (Algorithm 4).
- 4: Let N K, 3 ← N K, 2 as its initialization.
- 3: Denote /vector I ∗ K := ˆ /vector I ∗ K,r K , ˆ Q ∗ K := ¯ Q K,r K ( /vector I ∗ K , p ∗ K ) .
- 5: while t ≤ T do 6: Propose decisions ( /vector I t , p ′ t ) = ( /vector I ∗ K , ˆ p ∗ K ) .
- 8: Update ˆ W K ← ˆ Q ∗ K -L W · 1 T and ∆ K ← δ K √ N K, 3 .
- 7: Update ˆ Q ∗ K ← N K, 3 · ˆ Q ∗ K + Q t ( /vector I t ,p t ) N K, 3 +1 , and N K, 3 ← N K, 3 +1
- 9: end while

## 5 Analysis

In this section, we provide the theoretical analysis on the performance of Algorithm 2. We firstly propose our main theorem that upper bounds the cumulative regret.

Theorem 5.1 (Regret) . Let n 0 = 6(log 4 / 3 T + ˇ C K ) where ˇ C K := log 4 / 3 p max + 1 and δ K = √ 2log 48(2 mn +1) T /epsilon1 · max { p max , γ max } I max . Algorithm 2 guarantees an ˜ O ( √ Tmn + mn ) regret with probability at least 1 -/epsilon1 . Here ˜ O ( · ) omits the dependence on log 1 /epsilon1 and log T .

This regret rate is near-optimal with respect to T , as it matches the information-theoretic lower bound of Ω ( √ T ) (see Broder and Rusmevichientong, 2012, Theorem 3.1), which describes a special case as m = n = 1 and γ 1 = C 1 , 1 = 0 in our setting.

In the following, we present two lemmas that show the convergence of each agent A K . Specifically, Lemma 5.2 shows the 'horizontal convergence' of (inventory, price) decisions towards the local optimal. Lemma 5.3 shows the 'vertical convergence' of estimation error ∆ K such that we are maintaining and updating a valid lower-confidence bound for the majority of time.

Lemma 5.2 (Sub-regret of every A K ) . For agent A K (defined as Algorithms 3 to 5) that has been running for T K time periods so far, the cumulative sub-regret is bounded by:

<!-- formula-not-decoded -->

The proof of Lemma 5.2 is relegated to Appendix B.4.

Lemma 5.3 (Validity of ∆ K ) . For any agent A K has been running for T K time periods with T K ≥ 6(log 4 / 3 T + ˇ C K ) , we have ∆ K = ˜ O ( 1 √ T K ) .

The proof of Lemma 5.3 is in Appendix B.5. From Lemma 5.3, we directly get the following corollary.

Corollary 5.4. After at most N 0 := 6( mn + 1)(log 4 / 3 T + ˇ C K ) time periods, there does not exist any K ∈ [ mn ] ∪ { 0 } such that ∆ K = + ∞ .

Furthermore, due to the piecewise convexity of W ( p ) and the convergence rate of ∆ K , combining with Corollary 5.4, we have the following lemma.

Lemma 5.5. At any time t &gt; N 0 := 6( mn +1)(log 4 / 3 T + ˇ C K ) , we have

<!-- formula-not-decoded -->

Here L W is the Lipschitz coefficient of W ( p ) .

The proof details of Lemma 5.5 is displayed in Appendix B.6. With the help of all the lemmas above, we are now ready to provide an upper bound on the total regret.

<!-- formula-not-decoded -->

## 6 Discussion

Here we discuss some potential extensions of our work, serving as heuristics towards a broader field of research.

Generalization to non-linear demands. We assume the demand /vector D is a linear function of price p , which is a widely-used assumption (see LaFrance, 1985). Meanwhile, we still want to generalize our methodologies to a broader family of non-linear demands. Notice that the second-stage allocation problem defined by Eq. (2) does not involve the formulation of demand w.r.t. p . Therefore, we may still divide the price space into [ C i K ,j K , C i K +1 ,j K +1 ] intervals, and run an individual online optimization agent within each interval. With a similar analysis, we can achieve an ˜ O ( T α ( mn ) 1 -α ) regret, where α ≥ 1 / 2 is dependent on the demand family we assume. On the other hand, by selecting m = n = 1 and C i,j = 0 , we may have a lower bound at Ω ( T α ) .

Generalization to censored demands. In this work, we consider a warehouse-retailer setting where the demand orders are realized and informed to the suppliers before they are served. However, there exists another supply-demand relationship, such as groceries and wholesales, where the realized demands are revealed only after the resources are delivered

from the supply side to the demand side as a preparation. In that case, we should estimate the prospective demand and carefully balance the allocation among individuals in each side respectively, which goes much beyond a greedy allocation scheme as we solve Eq. (2). Besides, the realized demand might be censored when supply shortage occurs, making the problem more challenging. Therefore, we expect future investigations toward that new problem.

Pricing and service fairness. Our model maintains fairness in the pricing process by offering the same price to all consumers. However, while the greedy policy for resource allocation is reasonable, widely adopted, and analytically optimal, it leads to differentiated service levels among consumers. We anticipate future research focused on ensuring fairness in service levels during resource allocation.

## 7 Conclusion

In this paper, we study an online learning problem under the framework of pricing and allocation, where we make joint pricing and inventory decisions and allocate supplies to fulfill demands over time. To solve this non-convex problem, we propose a hierarchical algorithm which incorporates an LCB meta-algorithm over multiple local OCO agents. Our analysis shows that it guarantees an ˜ O ( √ Tmn + mn ) regret, which is optimal with respect to T as it matches the existing lower bound. Our work sheds light on the cross-disciplinary research of machine learning and operations research, especially from an online perspective.

## References

- Agarwal, A., Foster, D. P., Hsu, D. J., Kakade, S. M., and Rakhlin, A. (2011). Stochastic convex optimization with bandit feedback. Advances in Neural Information Processing Systems , 24.
- Asadpour, A., Wang, X., and Zhang, J. (2020). Online resource allocation with limited flexibility. Management Science , 66(2):642-666.
- Baby, D., Xu, J., and Wang, Y.-X. (2023). Non-stationary contextual pricing with safety constraints. Transactions on Machine Learning Research .
- Ban, G.-Y. and Keskin, N. B. (2021). Personalized dynamic pricing with machine learning: High-dimensional features and heterogeneous elasticity. Management Science , 67(9):55495568.
- Besbes, O. and Zeevi, A. (2009). Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. Operations Research , 57(6):1407-1420.

- Broder, J. and Rusmevichientong, P. (2012). Dynamic pricing under a general parametric choice model. Operations Research , 60(4):965-980.
- Bumpensanti, P. and Wang, H. (2020). A re-solving heuristic with uniformly bounded loss for network revenue management. Management Science , 66(7):2993-3009.
- Chen, B., Chao, X., and Ahn, H.-S. (2019). Coordinating pricing and inventory replenishment with nonparametric demand learning. Operations Research , 67(4):1035-1052.
- Chen, B., Chao, X., and Shi, C. (2021a). Nonparametric learning algorithms for joint pricing and inventory control with lost sales and censored demand. Mathematics of Operations Research , 46(2):726-756.
- Chen, B., Chao, X., and Wang, Y. (2020). Data-based dynamic pricing and inventory control with censored demand and limited price changes. Operations Research , 68(5):14451456.
- Chen, B., Wang, Y., and Zhou, Y. (2023). Optimal policies for dynamic pricing and inventory control with nonparametric censored demands. Management Science .
- Chen, N. and Gallego, G. (2021). Nonparametric pricing analytics with customer covariates. Operations Research .
- Chen, Q., Jasin, S., and Duenyas, I. (2021b). Joint learning and optimization of multiproduct pricing with finite resource capacity and unknown demand parameters. Operations Research , 69(2):560-573.
- Chen, X., Zhang, X., and Zhou, Y. (2021c). Fairness-aware online price discrimination with nonparametric demand models. arXiv preprint arXiv:2111.08221 .
- Choi, Y.-G., Kim, G.-S., Choi, Y., Cho, W., Paik, M. C., and Oh, M.-h. (2023). Semiparametric contextual pricing algorithm using cox proportional hazards model. In International Conference on Machine Learning , pages 5771-5786. PMLR.
- Cohen, M. C., Elmachtoub, A. N., and Lei, X. (2022). Price discrimination with fairness constraints. Management Science .
- Cohen, M. C., Lobel, I., and Paes Leme, R. (2020). Feature-based dynamic pricing. Management Science , 66(11):4921-4943.
- Cohen, M. C., Miao, S., and Wang, Y. (2021). Dynamic pricing with fairness constraints. Available at SSRN 3930622 .
- Cournot, A. A. (1897). Researches into the Mathematical Principles of the Theory of Wealth . Macmillan.

- Eyster, E., Madarász, K., and Michaillat, P. (2021). Pricing under fairness concerns. Journal of the European Economic Association , 19(3):1853-1898.
- Fan, J., Guo, Y., and Yu, M. (2021). Policy optimization using semiparametric models for dynamic pricing. arXiv preprint arXiv:2109.06368 .
- Ferreira, K. J., Simchi-Levi, D., and Wang, H. (2018). Online network revenue management using thompson sampling. Operations research , 66(6):1586-1602.
- Hazan, E. (2016). Introduction to online convex optimization. Foundations and Trends in Optimization , 2(3-4):157-325.
- Hwang, D., Jaillet, P., and Manshadi, V. (2021). Online resource allocation under partially predictable demand. Operations Research , 69(3):895-915.
- Jasin, S. and Kumar, S. (2012). A re-solving heuristic with bounded revenue loss for network revenue management with customer choice. Mathematics of Operations Research , 37(2):313-345.
- Javanmard, A. and Nazerzadeh, H. (2019). Dynamic pricing in high-dimensions. The Journal of Machine Learning Research , 20(1):315-363.
- Jenatton, R., Huang, J., and Archambeau, C. (2016). Adaptive algorithms for online convex optimization with long-term constraints. In International Conference on Machine Learning , pages 402-411. PMLR.
- Jia, H., Shi, C., and Shen, S. (2022). Online learning and pricing with reusable resources: Linear bandits with sub-exponential rewards. In International Conference on Machine Learning , pages 10135-10160. PMLR.
- Jiang, J., Ma, W., and Zhang, J. (2022). Degeneracy is ok: Logarithmic regret for network revenue management with indiscrete distributions. arXiv preprint arXiv:2210.07996 .
- Karan, A., Balepur, N., and Sundaram, H. (2024). Designing fair systems for consumers to exploit personalized pricing. arXiv preprint arXiv:2409.02777 .
- Keskin, N. B., Li, Y., and Song, J.-S. (2022). Data-driven dynamic pricing and ordering with perishable inventory in a changing environment. Management Science , 68(3):19381958.
- Kleinberg, R. (2004). Nearly tight bounds for the continuum-armed bandit problem. Advances in Neural Information Processing Systems , 17:697-704.
- Kleinberg, R. and Leighton, T. (2003). The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In IEEE Symposium on Foundations of Computer Science (FOCS-03) , pages 594-605. IEEE.

- LaFrance, J. T. (1985). Linear demand functions in theory and practice. Journal of Economic theory , 37(1):147-166.
- Leme, R. P., Sivan, B., Teng, Y., and Worah, P. (2021). Learning to price against a moving target. In International Conference on Machine Learning , pages 6223-6232. PMLR.
- Luo, Y., Sun, W. W., et al. (2021). Distribution-free contextual dynamic pricing. arXiv preprint arXiv:2109.07340 .
- Luo, Y., Sun, W. W., and Liu, Y. (2022). Contextual dynamic pricing with unknown noise: Explore-then-ucb strategy and improved regrets. In Advances in Neural Information Processing Systems .
- Reiman, M. I. and Wang, Q. (2008). An asymptotically optimal policy for a quantity-based network revenue management problem. Mathematics of Operations Research , 33(2):257282.
- Shalev-Shwartz, S. et al. (2012). Online learning and online convex optimization. Foundations and Trends® in Machine Learning , 4(2):107-194.
- Simchi-Levi, D. and Wang, C. (2023). Pricing experimental design: causal effect, expected revenue and tail risk. In International Conference on Machine Learning , pages 3178831799. PMLR.
- Tullii, M., Gaucher, S., Merlis, N., and Perchet, V. (2024). Improved algorithms for contextual dynamic pricing. arXiv preprint arXiv:2406.11316 .
- Vera, A. and Banerjee, S. (2019). The bayesian prophet: A low-regret framework for online decision making. ACM SIGMETRICS Performance Evaluation Review , 47(1):81-82.
- Vera, A., Banerjee, S., and Gurvich, I. (2021). Online allocation and pricing: Constant regret via bellman inequalities. Operations Research , 69(3):821-840.
- Wang, H., Talluri, K., and Li, X. (2021a). On dynamic pricing with covariates. arXiv preprint arXiv:2112.13254 .
- Wang, Y., Chen, B., and Simchi-Levi, D. (2021b). Multimodal dynamic pricing. Management Science .
- Xu, J., Qiao, D., and Wang, Y.-X. (2023). Doubly fair dynamic pricing. In International Conference on Artificial Intelligence and Statistics , pages 9941-9975. PMLR.
- Xu, J. and Wang, Y.-X. (2021). Logarithmic regret in feature-based dynamic pricing. Advances in Neural Information Processing Systems , 34.

- Xu, J. and Wang, Y.-X. (2022). Towards agnostic feature-based dynamic pricing: Linear policies vs linear valuation with unknown noise. International Conference on Artificial Intelligence and Statistics (AISTATS) .
- Xu, J. and Wang, Y.-X. (2024). Pricing with contextual elasticity and heteroscedastic valuation. In Forty-first International Conference on Machine Learning .

## A Algorithmic Components: a Detailed Instructions

In what follows, we elaborate each component of A K 's algorithmic design in detail.

Horizontal search space for p ∗ K . In the design of Stage 1 algorithm as presented in Algorithm 3, we adopt the framework of zeroth-order online convex optimization. Specifically, we establish an epoch-based update rule of the search space of local optimal p ∗ K . The search space (interval) for Epoch τ = 1 , 2 , . . . is denoted as [ L K,τ , U K,τ ] . Within each epoch, we divide the time horizon into a series of doubling sub-epochs to gather samples for W ( a ) , W ( b ) , W ( c ) where a, b, c are the three quarter points. By the end of each sub-epoch, we update the estimates and examine whether their estimation error bar is separable according to certain rules. As we keep doubling the size of sub-epochs, the estimation error bars are shrinking exponentially until they are separated. Then we reduce the search space by one quarter and proceed to Epoch τ + 1 Sub-Epoch 1 . When the search space is as sufficiently small as O (1 /T ) , we stop searching and proceed to Stage 2 .

Vertical uncertainty bound for W ( p ∗ K ) . In Stage 1 , we maintain an error bar ∆ K as the confidence bound of estimating each local optimal W ( p ∗ K ) . We show that the error bar has a size of ˜ O ( 1 √ T K ) if we have run A K for T K times so far. In addition to the statistical concentrations, another intuition of this fact comes from Lemma B.5: A not-distinguishable situation implies a comparable uncertainty bound for the optimal.

Complementary sampling to enhance ∆ K . It is worth noting that ∆ K &lt; + ∞ does not exist for granted even after Stage 1. This is because we cannot update ∆ K when the search space [ L K , U K ] is updated, i.e., no simultaneous 'horizontal converging' and 'vertical converging'. As a consequence, if we are very 'lucky' that we can always reduce the search space in the first sub-epoch of every epoch until U K,τ -L K,τ ≤ 1 /T , then we will have ∆ K = + ∞ until Stage 2 . We resolve this issue in two approaches: (1) We upper bound the time periods before any ∆ K &lt; + ∞ by O (log T ) based on the Pigeon-Hole Theorem. (2) We have Stage 2 as a complementary sampling stage without causing excessive regret. By the end of Stage 2 , we will have an ideal error bar for each agent A K .

Pure local exploitation contributing to global LCB. From Agent A K 's perspective, it runs pure exploitation in Stage 3 (Algorithm 5) without causing extra sub-regret. However, it still keeps updating the estimates of ˆ W K and ∆ K to facilitate the LCB metaalgorithm.

## B Proof Details

## B.1 Proof of Lemma 3.1

Notice that Eq. (2) defines a linear programming which contains a matrix variable with constraints on the (weighted) sum of each row and each column. Therefore, we may prove

a generalized version of Lemma 3.1, which is defined as follows.

Lemma B.1. Given parameters c ∈ R s , A ∈ R m × s + , B ∈ R n × s + , define the following optimization problem

<!-- formula-not-decoded -->

It holds that g ( /vector I, /vector D ) is convex w.r.t. [ /vector I ; /vector D ] .

Proof of Lemma B.1. Consider the Lagrangian of Eq. (10):

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

Here the second line is due to the strong duality of linear programming. Since the last line indicates that g ( /vector I, /vector D ) can be represented as the piecewise max of linear functions (which is convex), we know that g ( /vector I, /vector D ) is also convex w.r.t. [ /vector I ; /vector D ] jointly.

## B.2 Proof of Lemma 3.3

We denote ˜ C i,j ( p ) := -p + C i,j . For each fixed k ∈ [ mn ] , for any p ∈ ( C i k ,j k , C i k +1 ,j k +1 ) , the sign of any ˜ C i,j ( p ) is fixed. Therefore, when k is fixed, we may let X i,j = 0 for any ( i, j ) /followsequal ( i k +1 , j k +1 ) in the optimal solution of g ( /vector I, p, /vector D ) . Given this, there exists A ∈ R k × k , B ∈ R k × k , /vector C ∈ R k such that the optimization problem defined as Eq. (2) for

p ∈ ( C i k ,j k , C i k +1 ,j k +1 ) can be generalized to the following linear programming:

<!-- formula-not-decoded -->

Without loss of generality, in the following part of this proof of Lemma 3.3, we show that g ( /vector I, p, /vector D ) defined in Eq. (13) is convex for p ∈ ( C i k ,j k , C i k +1 ,j k +1 ) . The Lagrangian of this new g ( /vector I, p, /vector D ) is

<!-- formula-not-decoded -->

Since linear programming has strong duality, we further have

<!-- formula-not-decoded -->

As a consequence, the definition of Q ( /vector I, p ) is now generalized to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote

Notice that for any p 1 , p 2 s.t. C i k ,j k ≤ p 1 &lt; p 2 ≤ C i k +1 ,j k +1 , we have S ( p 1 ) ⊇ S ( p 2 ) , indicating that S ( p ) is a monotonically shrinking convex set as p increases. Given the definition of S ( p ) in Eq. (17), the definition of Q t ( /vector I, p ) is equivalent to

<!-- formula-not-decoded -->

In the following, we prove a more generalized lemma, from which we can immediately derive the convexity of Eq. (18).

Lemma B.2. Consider a family of functions F := { f ( x ; θ ) } , where f ( x ; θ ) is Lipschitz and convex on x ∈ [0 , 1] and is parametrized by θ . S ( x ) is a convex set that is monotonically not expanding w.r.t. x (i.e., S ( x 1 ) ⊇ S ( x 2 ) if 0 ≤ x 1 &lt; x 2 ≤ 1 ). If Q ( x ) := max θ ∈ S ( x ) f ( x ; θ ) is Lipschitz, then Q ( x ) is convex on [0 , 1] .

Proof of Lemma B.2. Denote the epigraph of a function g : R → R as

<!-- formula-not-decoded -->

Also, for set S ( x ) , x ∈ [0 , 1] , denote

<!-- formula-not-decoded -->

Since S ( x ) is monotonically not expanding, we know that S -1 ( θ ) is continuous. This is because θ ∈ S ( x ) is a sufficient condition of θ ∈ S ( y ) for any y ∈ [0 , x ] . Given these definitions, we have

<!-- formula-not-decoded -->

Since f ( x ; θ ) is a convex function with respect to x in [0 , 1] , we know that epi [ a,b ] f ( · ; θ ) is a convex domain for any 0 ≤ a ≤ b ≤ 1 . Therefore, their intersections over all θ ∈ S (0) is a convex domain. This shows that the epigraph of Q ( x ) is convex, and therefore Q ( x ) is convex.

With Lemma B.2, we know that Q t ( /vector I, p ) defined in Eq. (18) is convex with respect to p . This ends the proof of Lemma 3.3.

## B.3 Proof of Lemma 3.4

Denote /vector I ∗ ( p ) := argmin /vector I Q ( /vector I, p ) . According to the definition of W ( p ) given by Eq. (5), we have

<!-- formula-not-decoded -->

According to the definition of /vector I ∗ ( p ) , we have ∂Q ( /vector I,p ) ∂ /vector I = 0 at /vector I = /vector I ∗ ( p ) . Now we show that ∂Q ( /vector I ∗ ( p ) ,p ) ∂p is a monotonically increasing function of p . According to the definition of Q , we have

<!-- formula-not-decoded -->

Denote

<!-- formula-not-decoded -->

Due to the property that the limit operation preserves convexity, it is sufficient to show that ˇ Q ∆ ( /vector I ∗ , p ) is convex with respect to p . Also, since g ( /vector I, p, /vector D ) is the value to a linear programming(LP), the solution should be located on the vertex of its feasible space. Therefore, the non-smooth singularities of g ( /vector I ∗ ( p ) , p,/vector a -/vector b · p + /vector N t ) only exists when multiple vertices of the LP feasible space coincide. According to Eq. (2), the feasible space have at most ( mn + m + n mn ) vertices. Denote G ( m,n ) := 2 ( mn + m + n mn ) , and we know that there are at most

G ( m,n ) chances that at least two vertices can coincide with each other (hence causing a nonsmooth singularity). As a consequence, we have ˇ Q ∆ ( /vector I ∗ , p ) has at most E := ( 2 C ∆ ) n · G ( m,n ) non-smooth singularities. Without loss of generality, denote them as

<!-- formula-not-decoded -->

Also denote P 0 := C i K ,j K and P E +1 := C i K +1 ,j K +1 . Now we propose another two lemmas.

Lemma B.3. For p ∈ ( P e , P e +1 ) , e = 0 , 1 , 2 , . . . , E , we show that ∂ ˇ Q ( /vector I ∗ ,p ) ∂p is monotonically increasing on p .

Proof of Lemma B.3. Notice that

<!-- formula-not-decoded -->

Here ˇ /vector N t := [ -C 1 + l 1 ∆, -C 2 + l 2 ∆,... , -C n + l n ∆ ] /latticetop . Now we consider the monotonicity of ∂Q t ( /vector I ∗ ,p ) ∂p on each ( P e , P e +1 ) interval. Since there exist no singularities in this interval, we know that Q t /vector I ∗ , p ∈ C 2 in this range, and therefore we have

<!-- formula-not-decoded -->

Here the second line that we swap the sequence of derivatives is due to the smoothness within the ( P e , P e +1 ) smooth interval, and the last line is from Lemma 3.3 which shows the marginal convexity of Q t ( /vector I, p ) w.r.t. p . Therefore, we have proved the lemma.

Lemma B.4. At each P e for e = 0 , 1 , 2 , . . . , E , we have W ′ ( P -e ) ≤ W ′ ( P + e ) .

Proof of Lemma B.4. We firstly consider W ′ ( P -e ) . According to the proof of Lemma B.3, we have

<!-- formula-not-decoded -->

Here the fourth and the sixth lines are due to the Moore-Osgood theorem of exchanging limits, as Q t ( /vector I ∗ , p ) ∈ C 2 when p ∈ ( P e -1 , P e ) and p ∈ ( P e , P e +1 ) respectively. The fifth line comes from Lemma 3.3: as Q t ( /vector I, p ) is convex w.r.t. p , the left derivatives of ∂Q t ( /vector I,p ) ∂p should not exceed its right derivatives at any point p .

Applying Lemma B.3 on Eq. (22), we know that W ( p ) is convex within each smooth interval ( P e , P e +1 ) . Also, from Lemma B.4, we know that W ( p ) is convex at any singularity P e as its left derivatives does not exceed its right derivatives. Combining those two properties, we know that Q ( /vector I ∗ , p ) is convex w.r.t. p . This ends the proof.

## B.4 Proof of Lemma 5.2

The main idea of this proof originates from OCO with zeroth-order (bandit) feedback, as is displayed in Agarwal et al. (2011). Specifically, we conduct the proof in the following steps:

- (a) When an agent A K is in Stage 1, Epoch τ and Sub-Epoch s , then we sequentially show that

- (i) The aggregated function Q K,τ,s ( /vector I, p ) is concentrated to Q ( /vector I, p ) for the three proposed p = ˆ p τ ∈ { a K,τ , b K,τ , c K,τ } and for any /vector I , with O (1 / √ n s ) error.
- (ii) The ˆ Q K,τ,s,p , which takes the empirical optimal inventory decision, is concentrated to Q ( /vector I ∗ ( p ) , p ) = W ( p ) at those proposed prices p = ˆ p τ ∈ { a K,τ , b K,τ , c K,τ } , with O (1 / √ n s ) error.
- (iii) According to the convexity of W ( p ) , we upper bound the sub-regret per round W ( p ) -W ( p ∗ K ) by O (1 / √ n s ) , where p ∗ K is the local optimal price.
- (iv) We show that the total number of epochs in Stage 1 is O (log T ) . According to the doubling lengths of n s , the total sub-regret of A K is ˜ O ( √ T K ) by the time when A has proposed T K pairs of decisions ( /vector I t , p t ) .
- (b) When an agent A K reaches Stage 2 or 3, we know that the search space [ L K , U K ] is smaller than 1 /T . Given that the Q t functions (and therefore W ( p ) ) are Lipschitz, we may upper bound the sub-regret per step as O (1 /T ) and the total sub-regret as O (1) .

Before we get to proof details, we propose a lemma that generally holds for convex functions.

Lemma B.5. Suppose f : [ a, b ] → R is a L -Lipschitz convex function. Denote f ( x ∗ ) := min x ∈ [ a,b ] f ( x ) , x 1 = 3 a + b 4 , x 2 = a + b 2 , x 3 = a +3 b 4 . Assume there exists some fixed constants A and ∆ &gt; 0 such that f ( x i ) ∈ [ A -∆,A + ∆ ] , i = 1 , 2 , 3 , then we have

<!-- formula-not-decoded -->

Please kindly find the proof of Lemma B.5 in Appendix B.7.

Now we return to the main proof. We firstly propose the following concentration lemma

Lemma B.6. For Agent A K running in Epoch τ Sub-Epoch s , and ∀ /vector I ∈ R + , symbolic variable ˆ p τ ∈ { a K,τ , b K,τ , c K,τ } , we have

<!-- formula-not-decoded -->

with probability Pr ≥ 1 -ˆ /epsilon1 . Here C c := √ 2log 2 ˆ /epsilon1 · Q max and Q max := max { p max , γ max } I max .

We will specify the value of ˆ /epsilon1 as a function of /epsilon1 by the end of this proof. We defer the proof of Lemma B.6 to Appendix B.8. From Lemma B.6, we may get the following corollary.

Corollary B.7. For Agent A K running in Epoch τ Sub-Epoch s , and symbolic variable ˆ p τ ∈ { a K,τ , b K,τ , c K,τ } we have

<!-- formula-not-decoded -->

with probability Pr ≥ 1 -4 T ˆ /epsilon1 .

Proof of Corollary B.7. For each fixed tuple ( K,τ,s, ˆ p τ ) , according to Lemma B.6, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability Pr ≥ 1 -2ˆ /epsilon1 . Here the first inequality comes from the optimality of I K,τ,s (ˆ p τ ) over Q K,τ,s ( · , ˆ p τ ) as well as the concentration of Q K,τ,s ( /vector I ∗ (ˆ p τ ) , ˆ p τ ) towards Q ( /vector I ∗ (ˆ p τ ) , ˆ p τ ) . The second inequality comes from the concentration of Q K,τ,s ( I K,τ,s (ˆ p τ ) , ˆ p τ ) towards Q ( I K,τ,s (ˆ p τ ) , ˆ p τ ) as well as the optimality of /vector I ∗ (ˆ p τ ) over Q ( · , ˆ p τ ) .

Also, we have

<!-- formula-not-decoded -->

with probability Pr ≥ 1 -2ˆ /epsilon1 . Here the third line comes from the concentrations (the first and the third term) as well as the optimality of I K,τ,s (ˆ p τ ) over Q K,τ,s ( · , ˆ p τ ) . Besides, the other side that Q ( I K,τ,s (ˆ p τ ) , ˆ p τ ) -Q ( I ∗ (ˆ p τ ) , ˆ p τ ) is due to the optimality of /vector I ∗ (ˆ p τ ) over Q ( · , ˆ p τ ) .

Since the combination of ( K,τ,s, ˆ p τ ) is unique, and the total number of combinations is exactly T , we apply the union bound of probability and get that Eq. (30) holds for all ( K,τ,s, ˆ p τ ) tuples with probability Pr ≥ 1 -4 T ˆ /epsilon1 .

<!-- formula-not-decoded -->

Combining Corollary B.7 with Lemma B.5, we have the following corollary

Corollary B.8. Define a flag as shown in Algorithm 3, and define

<!-- formula-not-decoded -->

When flag == 0 by the end of Sub-Epoch s of Epoch τ , we have

<!-- formula-not-decoded -->

holds for ˆ p τ = a K,τ , b K,τ , c K,τ with probability Pr ≥ 1 -24 T ˆ /epsilon1 .

Proof of Corollary B.8. When flag == 0 , according to Algorithm 3, we know that

<!-- formula-not-decoded -->

Also, according to the convexity of W ( P ) in [ C i K ,j K , C i K +1 ,j K +1 ] , we know that W ( c K,τ ) ≤ W ( a K,τ )+ W ( b K,τ ) 2 ≤ max { W ( a K,τ ) , W ( b K,τ ) } . Without loss of generality, assume W ( a K,τ ) ≥ W ( b K,τ ) , and then we have

<!-- formula-not-decoded -->

Therefore we know that | ˆ Q K,τ,s,b K,τ -ˆ Q K,τ,s,c K,τ | ≤ 4 ∆ K,τ,s . Combining Corollary B.7, we have

<!-- formula-not-decoded -->

And also

<!-- formula-not-decoded -->

By applying Lemma B.5 with ∆ = 4 C c √ n s , we show that the lemma holds with Pr ≥ 1 -24 T ˆ /epsilon1 (since we have used Corollary B.7 for 6 times).

Finally, we show that upper bounds the total number of epochs in which A K is running, and we first denote this number as M K . In fact, from the design of Algorithm 3 , we know that by the end of each epoch, we have U K,τ +1 -L K,τ +1 = 3 4 ( U K,τ -L K,τ ) , i.e. the length of search space [ L K,τ , U K,τ ] reduces by 1 / 4 . Since L K, 1 = C i K ,j K , U K, 1 = C i K +1 ,j K +1 , we have

<!-- formula-not-decoded -->

Denote ˇ C K := log 4 / 3 p max +1 , and we have M K ≤ log 4 / 3 T + ˇ C K

With all properties above, we may derive the total sub-regret for A K . Firstly, the cumulative sub-regret in Epoch τ Sub-Epoch s is

<!-- formula-not-decoded -->

Secondly, denote the number of sub-epochs in Epoch τ as S τ and the length of Epoch τ as T τ (therefore we know that T τ = 3 · 2 S τ +1 -1 ), and the cumulative sub-regret in Epoch τ is bounded by

<!-- formula-not-decoded -->

Thirdly, we may calculate the total sub-regret of A K as

<!-- formula-not-decoded -->

This rate holds with Pr ≥ 1 -24 T ˆ /epsilon1 for each K ∈ [2 mn + 1] . Let ˆ /epsilon1 := /epsilon1 24 · (2 mn +1) T so that C c = δ K = √ 2log 48(2 mn +1) T /epsilon1 · max { p max , γ max } · I max , and we complete the proof of Lemma 5.2.

## B.5 Proof of Lemma 5.3

We analyze the behavior of ∆ K by considering the current stage of A K .

1. If A K is currently in Stage 1. Suppose A K has played for τ K epochs. Since each epoch reduces the price interval [ L K,τ , U K,τ ] to its 3/4, we know that τ K ≤ M k ≤ log 4 / 3 T + ˇ C K where C K := log 4 / 3 p max +1 (see Eq. (39)).

According to Pigeon-Hole Theorem, at least the longest epoch ˆ τ has been played for T K τ K ≥ 6(log 4 / 3 T + C K ) log 4 / 3 T + C K = 6 times. As a result, at least two sub-epochs have been reached in this epoch. Since we update ∆ K by the end of each sub-epoch (except for the last sub-epoch of each epoch), at least one of these two sub-epochs leads to an update on ∆ K . As a result, ∆ K &lt; + ∞ after this update, and after T K ≥ 6(log 4 / 3 T + C K ) time periods.

Denote the length of this epoch ˆ τ as H K, ˆ τ , and we know that H K, ˆ τ ≥ T K log 4 / 3 T + C K . Also, we denote the length of each sub-epoch of Epoch ˆ τ as H K, ˆ τ,s , s = 1 , 2 , . . . , S ˆ τ , where S ˆ τ is denoted as the number of sub-epochs of Epoch ˆ τ . Given those definitions,

we have

As a consequence, we have

<!-- formula-not-decoded -->

Since we still can update ∆ K by the end of Epoch ˆ τ Sub-Epoch S ˆ τ -1 , we may upper-bound ∆ K in the following approach

<!-- formula-not-decoded -->

2. If A K reaches Stage 2. Since we only run Stage 2 for once without stopping, updating ˆ W K , ∆ K or switching agents, we assume that T K reaches the end of Stage 2 without loss of generality. We firstly upper and lower bound the length of Stage 2. Denote T K, 1 as the time periods that A K spent on Stage 1, and T K, 2 := T K -T K, 1 as the time periods that A K has spent on Stage 2 so far. Remember that the purpose of conducting Stage 2 is to guarantee a ∆ K that is comparable to √ 1 T K , and at the end of Stage 2 we reduce ∆ K to its half comparing to the one we have by the end of Stage 1 (if not + ∞ ). Therefore, we have

<!-- formula-not-decoded -->

Here the first inequality represents that A K runs T K, 1 time periods in Stage 1, including H K, ˆ τ time periods in Stage 1 Epoch ˆ τ . The second and third inequalities hold because we get comparable ∆ K in Stage 1 and in Stage 2, and the best ∆ K we got in

<!-- formula-not-decoded -->

Stage 1 is on the longest sub-epoch, which is ˆ τ . The last inequality is from the proof shown in Case 1 (when A K reaches Stage 1).

Also, since Stage 2 applies a "Doubling Trick", we have

<!-- formula-not-decoded -->

As a result, we have

<!-- formula-not-decoded -->

3. If A K reaches Stage 3. Denote T K, 3 := T K -T K, 1 -T K, 2 as the time periods that A K has spent on Stage 3 so far. According to Algorithm 5, we know that ∆ K = δ K √ N K, 3
2. = δ K √ N K, 2 + T K, 3 = ˜ O ( 1 √ N K, 2 + T K, 3 ) . Also, since

<!-- formula-not-decoded -->

Therefore, we have ∆ K ≤ ˜ O ( 1 √ T K 2(log 4 / 3 T + C K ) ) = ˜ O ( 1 √ T K ) . This ends the proof of Lemma 5.3.

## B.6 Proof of Lemma 5.5

We consider each case where A K is in Stage 1,2,3, respectively.

1. If A K is in Stage 1. When updating ∆ K , we know that flag == 0 at Stage 1 Epoch τ Sub-Epoch s -1 according to Algorithm 3. Denote ˆ p := argmin ˆ p τ ∈{ a K,τ ,c K,τ ,b K,τ } ˆ Q K,τ,s -1 , ˆ p τ ,

and we have that

<!-- formula-not-decoded -->

Here the fourth line comes from Corollary B.7 and the last line is an application of C c = δ K and Corollary B.8.

On the other hand, the lower bound LCB K is not too faraway from W ( p ∗ K ) since we have:

<!-- formula-not-decoded -->

2. If A K is in Stage 2 and Stage 3, we may consider them altogether as the only update of LCB as well as ∆ K occurs by the end of Stage 2, which is also the 0 -th time period

of Stage 3. In this case, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the two cases listed above, we have proved this lemma.

## B.7 Proof of Lemma B.5

Proof. Denote f 1 := f ( x 1 ) , f 2 := f ( x 2 ) , f 3 := f ( x 3 ) . Then we prove this lemma by cases where x ∗ locates.

1. When x ∗ ∈ [ a, x 1 ] , we denote /epsilon1 := x 2 -x 1 x 2 -x ∗ . We know that /epsilon1 ∈ [ 1 2 , 1] , and then we have /epsilon1f ( x ∗ ) + (1 -/epsilon1 ) f ( x 2 ) ≥ f ( /epsilon1x ∗ +(1 -/epsilon1 ) x 2 ) = f ( x 1 ) due to the convexity of f ( x ) . As a result, we have:

<!-- formula-not-decoded -->

If f 1 ≥ f 2 , then we have f ( x ∗ ) ≥ f 2 ≥ A -∆ = A + ∆ -2 ∆ ≥ max { f 1 , f 2 , f 3 } -2 ∆ . Otherwise f 1 &lt; f 2 , then we have

<!-- formula-not-decoded -->

2. When x ∗ ∈ ( x 1 , x 2 ] , let /epsilon1 = x 3 -x 2 x 3 -x ∗ , and the proof goes the same way as in (1).
3. When x ∗ ∈ ( x 2 , x 3 ] , we let /epsilon1 = x 2 -x 1 x ∗ -x 1 and we know that /epsilon1 ∈ [ 1 2 , 1] . Since x 2 = /epsilon1 · x ∗ +(1 -/epsilon1 ) x 1 , we have /epsilon1f ( x ∗ ) + (1 -/epsilon1 ) f ( x 1 ) ≥ f ( x 2 ) according to the convexity of f ( x ) . Therefore, we have:

<!-- formula-not-decoded -->

If f 1 ≤ f 2 , then we have f ( x ∗ ) ≥ f 1 ≥ A -∆ = A + ∆ -2 ∆ ≥ max { f 1 , f 2 , f 3 } -2 ∆ . Otherwise f 1 &gt; f 2 , then we have

<!-- formula-not-decoded -->

4. When x ∗ ( x 3 , b ] , let /epsilon1 = x 3 -x 2 x ∗ -x 2 , and the proof goes the same way as (3).

## B.8 Proof of Lemma B.6

Proof. Notice that

<!-- formula-not-decoded -->

Denote Q max := max { p max , γ max } I max . By applying Hoeffding's Inequality to ∀ /vector I, ˆ p τ ∈ { a K,τ , b K,τ , c K,τ } , we have

<!-- formula-not-decoded -->

Let C c = Q max √ 2log 2 ˆ /epsilon1 and this completes the proof.