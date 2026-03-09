## Dynamic Pricing with Adversarially-Censored Demands

Jianyu Xu 1 , Yining Wang 2 , Xi Chen 3 , and Yu-Xiang Wang 4

1 Carnegie Mellon University

3 New York University

2 University of Texas at Dallas

4 University of California San Diego

Abstract. We study an online dynamic pricing problem where the potential demand at each time period t = 1 , 2 , . . . , T is stochastic and dependent on the price. However, a perishable inventory is imposed at the beginning of each time t , censoring the potential demand if it exceeds the inventory level. To address this problem, we introduce a pricing algorithm based on the optimistic estimates of derivatives. We show that our algorithm achieves ˜ O ( √ T ) optimal regret even with adversarial inventory series. Our findings advance the state-of-the-art in online decision-making problems with censored feedback, offering a theoretically optimal solution against adversarial observations.

Keywords: Dynamic pricing · Online learning · Censored feedback.

## 1 Introduction

The problem of dynamic pricing, where the seller proposes and adjusts their prices over time, has been studied since the seminal work of [24]. The crux to pricing is to balance the profit of sales per unit with the quantity of sales. Therefore, it is imperative for the seller to learn customers' demand as a function of price (commonly known as the demand curve ) on the fly. However, the demand can often be obfuscated by the observed quantity of sales, especially when censored by inventory stockouts. Such instances severely impede the seller from learning the underlying demand distributions, thereby hindering our pursuit of the optimal price.

Existing literature has devoted considerable effort to the intersection of pricing and inventory decisions. Such works often consider scenarios with indirectly observable lost demands [33], recoverable leftover demands [12], or controllable inventory level [15]. However, these assumptions do not always align with the realities faced in various common business environments. To illustrate, we present two pertinent examples:

Example 1 (Performance Tickets). Imagine that we manage a touring company that arranges a series of performances featuring a renowned artist across various cities. Each venue has a different seating capacity, which substantially affects how we set ticket prices. If the price is too high, it may deter attendance, leading to lower revenue. On the other hand, setting it too low could mean that tickets sell out quickly, leaving many potential attendees unable to purchase them. We do not know exactly how many people attempt to buy tickets and fail. Moreover, because the performances are unique, there is no assurance that those who miss out on one show will choose or be able to attend another. This variability in venue size across different locations requires us to continually adapt our pricing strategy. With more adaptive prices, we can maximize both attendance and revenue while accommodating unpredictable changes in seat availability.

Example 2 (Fruit Retails). Sweetsop ( Annona squamosa , or so-called 'sugar apple') is a particularly-perishable tropical fruit, typically lasting only 2 to 4 days [25]. Suppose we manage a local fruit shop and have partnered with a nearby farm for the supply of sweetsops during the harvest season. Due to their perishable nature, we receive sweetsops as soon as

they are ripe and picked from the farm every day. This irregular supply means that some days we might receive a large quantity while getting very few on other days. We must quickly sell these fruits before they spoil, yet managing the price becomes challenging. If we exhaust our inventory ahead of time, customers will turn to other fruit shops for purchase instead of waiting for our next restock.

Products in the two instances above have the following properties:

1. Inventory levels are determined by natural factors, and are arbitrarily given for different individual time periods.
2. Products are perishable and only salable within a single time period.

## 1.1 Problem Overview

In this work, we study a dynamic pricing problem where the products possess these properties. The problem model is defined as follows. At each time t = 1 , 2 , . . . , T , we firstly propose a price p t , and then a price-dependent potential demand occurs as d t . However, we might have no access to d t as it is censored by an adversarial inventory level γ t . Instead, we observe a censored demand D t = min { γ t , d t } and receive the revenue r t as a reward at t . Our goal is to approach the optimal price p ∗ t at every time t , thereby maximizing the cumulative revenue.

- Dynamic pricing with adversarial inventory constraint. For t = 1 , 2 , ..., T : 1. The seller (we) receives γ t identical products. 2. The seller proposes a price p t ≥ 0 . 3. The customers generate an invisible potential demand d t ≥ 0 , dependent on p t . 4. The market reveals an inventory-censored demand D t = min { γ t , d t } . 5. The seller gets a reward r t = p t · D t . 6. All unsold products perish before t +1 .

The notion of 'adversarial' inventory. We characterize this problem as having adversariallychosen inventory levels, though our setting differs from classical adversarial bandits or online convex optimization. Here, an adversary may pre-commit to an arbitrary inventory sequence { γ t } T t =1 , while the underlying demand noise remains stochastic-a hybrid model that combines adversarial contexts with stochastic outcomes, following the convention of 'adversarial features' in online learning [21,39]. Since inventory γ t is revealed at the beginning of round t , we can define per-round optimal prices p ∗ t . This leads to a stronger regret benchmark: we compete against this sequence of optimal actions { p ∗ t } T t =1 rather than the best fixed decision in hindsight typical of standard adversarial settings. This distinction is crucial as the inventory levels create a non-stationary optimization landscape where optimal prices vary dramatically across periods.

## 1.2 Summary of Contributions

We consider the problem setting shown above and assume that the potential demand d t = a -bp t + N t is linear and noisy . Here a, b ∈ R + are fixed unknown parameters and N t is an unknown and i.i.d. (independently and identically distributed) noise with zero mean. Under this premise, the key to obtaining the optimal price is to accurately learn the expected reward function r ( p ) , which is equivalent to learning the linear parameters [ a, b ] and the noise distribution. We are faced with three principal challenges:

1. The absence of unbiased observations of the potential demand or its derivatives with respect to p , which prevents us from estimating [ a, b ] directly.
2. The dependence of the optimal prices on the noise distribution, which is assumed to be unknown and partially censored.
3. The arbitrariness of the inventory levels, leading to non-stationary and highly-differentiated optimal prices { p ∗ t } over time.

In this paper, we introduce an algorithm that employs innovative techniques to resolve the aforementioned challenges. Firstly, we devise a pure-exploration phase that bypasses the censoring effect and obtains an unbiased estimator of 1 b (which leads to ˆ b and ˆ a as a consequence). Secondly, we maintain estimates of the noise CDF F ( x ) and ∫ F ( x ) dx over a series of discrete x 's, as well as the confidence bounds of each estimate. Thirdly, we design an optimistic strategy, C20CB as 'Closest-To-Zero Confidence Bound', that proposes the price p t whose reward derivative r ′ t ( p t ) is probably 0 or closest to 0 among a set of discretized prices. As we keep updating the estimates of r ′ t ( · ) with shrinking error bar, we asymptotically approach the optimal price p ∗ t since r ′ t ( p ∗ t ) = 0 for any t = 1 , 2 , . . . , T .

Novelty. To the best of our knowledge, we are the first to study the online dynamic pricing problem under adversarial inventory levels. Our C20CB algorithm attains an optimal ˜ O ( √ T ) regret guarantee with high probability. The methodologies we develop are crucial to our algorithmic design, and are potentially advancing a variety of online decision-making scenarios with censored feedback.

## 1.3 Paper Structure

The rest of this paper is organized as follows. We discuss related works in Section 2, and then describe the problem setting in Section 3. We propose our main algorithm C20CB in Section 4 and analyze its regret guarantee in Section 5. We further discuss potential extensions in Section 6, followed by a brief conclusion in Section 7.

## 2 Related Works

Here we discuss the closest related works on dynamic pricing, inventory constraints, and network revenue management. For a broader introduction of related literature, please refer to Section A.

Data-driven dynamic pricing. Dynamic pricing for identical products is a well-established research area, starting with [36] and continuing through seminal works by [7,10,55,57]. The standard approach involves learning a demand curve from price-sensitive demand arriving in real-time, aiming to approximate the optimal price. [36] provided algorithms with regret bounds of O ( T 2 3 ) and O ( √ T ) for arbitrary and infinitely smooth demand curves, respectively. [55] refined this further, offering an O ( T k +1 2 k +1 ) regret for k -times continuously differentiable demand curves. This line of inquiry is also intricately linked to the multi-armed bandit problems [4,37] and continuum-armed bandits [35], where each action taken reveals a reward without insight into the foregone rewards of other actions.

Pricing with inventory concerns. Dynamic pricing problems begin to incorporate inventory constraints with [7], which assumed a fixed initial stock available at the start of the selling period. They introduced near-optimal algorithms for both parametric and non-parametric demand distributions, operating under the assumption that the inventory is non-replenishable

and non-perishable. [57] adopted a comparable framework but allowed customers arrivals to follow a Poisson process. In these earlier works, the actual demand is fully disclosed until the inventory is depleted. Subsequent research allows inventory replenishment, with the seller's decisions encompassing both pricing and restocking at each time interval. [12] proposed a demand model subject to additive / multiple noise and developed a policy that achieved O ( √ T ) regret. More recent studies [14, 33] explored the pricing of perishable goods where the unsold inventory will expire. However, the uncensored demand is observable as assumed in both works. Specifically, [14] allowed recouping backlogged demand, albeit at a cost, and introduced an algorithm with optimal regret. [33] focused on the cases where both fulfilled demands and lost sales were observable.

[13] and their subsequent work, [15], are the closest works to ours as they adopt similar problem settings: In their works, the demand is censored by the inventory level and any leftover inventory or lost sales disappear at the end of each period. With the assumption of concave reward functions and the restriction of at most m price changes, [13] proposed MLE-based algorithms that attain a regret of ˜ O ( T 1 m +1 ) in the well-separated case and ˜ O ( T 1 2 + ϵ ) for some ϵ = o (1) as T → ∞ in the general case. Under similar assumptions (except infinite-order smoothness), [15] developed a reward-difference estimator, with which they not only enhanced the prior result for concave reward functions to ˜ O ( √ T ) but also obtained a general ˜ O ( T 2 / 3 ) regret for non-concave reward functions. Our problem model mirrors their difficulty, as we also lack access to both the uncensored demand and its gradient. However, they allowed the sellers to determine inventory levels with sufficient flexibility, hence better balancing the information revealed by the censored demand and the reward from (price, inventory) decisions. On the other hand, we assume that the inventory level at each time period is provided adversarially by nature, which could impede us from learning the optimal price in the worst-case scenarios. Furthermore, due to the non-stationarity of inventory levels in our setting, the optimal price p ∗ t deviates over time. Given this, the search-based methods adopted in [15] are no longer applicable to our problem.

Network Revenue Management (NRM). NRM [50] studies pricing and allocation of shared resources in a network. In the settings of [8] and subsequent works [44,47], marginal observations for each product can induce adversarial supply due to cross-product resource occupation. We also consider adversarial inventories, but focus on addressing demand censoring and reducing regret via online learning. [45] likewise used regret and considered censoring in demand data, but their minimax/maximin regret definitions differ from ours, and censoring is mainly used in empirical validation rather than for theoretical solutions. Representative NRM works include [28] on dynamic pricing with stochastic demand, establishing structural monotonicity and asymptotic optimality of simple policies; [49] showing bid-price controls are near-optimal in large capacities but not strictly optimal; and [42] addressing overbooking with product-specific no-shows via a randomized LP.

## 3 Problem Setup

We have defined the problem setting in Section 1.1. To further clarify the scope of our methodology, we make the following very first assumption before introducing further concepts.

Assumption 1 (Linear Demand) Assume the potential demand d t = d t ( p ) := a -bp + N t is linear and noisy . Denote d ( p ) := a -bp as the expected potential demand function . Denote D t ( p ) := min { γ t , d t ( p ) } as the censored demand function.

A linear demand model has been widely used in pricing literature, including [10,23,40,46]. We highlight two primary reasons: (1) According to [9], linear demand allows the prices

to converge to the true optimum even under mismatching scenarios. (2) As shown in [34], the same analysis can be applied to Generalized Linear Model (GLM), which has enhanced capability of capturing the real-world demands (see e.g. [11,54]).

## 3.1 Definitions

Here we define some key quantities that are involved in the algorithm design and analysis. Firstly, we define distributional functions of the noise N t .

Definition 1 (Distributional Functions). For N t as the demand noise, denote F ( x ) as its cumulative distribution function (CDF), x ∈ R . Also, denote the following G ( x ) as the integrated CDF :

<!-- formula-not-decoded -->

We will make more assumptions on the noise distribution later. Notice that we do not assume the existence of PDF for N t . However, if there exists its PDF in specific cases, we will adopt f ( x ) as a notation. Then, we define the revenue function and the regret.

Definition 2 (Revenue Function). Denote r t ( p ) as the expected revenue function of price p , satisfying

<!-- formula-not-decoded -->

Also, denote p ∗ t := argmax p r t ( p ) as the optimal price at time t .

Definition 3 (Regret). Denote

<!-- formula-not-decoded -->

as the cumulative regret (or regret ) of the price sequence { p t } T t =1 .

The definition of regret inherits from the tradition of online learning, capturing the performance difference between the algorithm-in-use and the best benchmark that an omniscient oracle can achieve (which knows everything except the realization of noise).

## 3.2 Assumptions

Firstly, we assume boundaries for parameters and price.

Assumption 2 (Boundedness) There exist known finite constants a max , b min , b max , γ min , and c &gt; 0 such that 0 &lt; a ≤ a max , 0 &lt; b min ≤ b ≤ b max , γ t ≥ γ min , N t ∈ [ -c, c ] . Also, we restrict the proposed price p t at any t = 1 , 2 , . . . , T satisfies 0 ≤ p t ≤ p max with a known finite constant p max &gt; 0 .

The assumption of boundedness on a, b and price is natural as it defines the scope of instances. We justify the assumption of noise with bounded support [ -c, c ] from two aspects: (1) The upper bound exists without loss of generality due to the existence of inventory-censoring effect. (2) The lower bound exists as we avoid negative demand. Secondly, we make assumptions on the noise distribution.

Assumption 3 (Noise Distribution) Each N t is drawn from an unknown independent and identical distribution (i.i.d.) satisfying E [ N t ] = 0 . The CDF F ( x ) is L F -Lipschitz . Also, according to Assumption 2, we have F ( -c ) = 0 , F ( c ) = 1 .

Thirdly, we make assumptions on the inequality relationships among parameters:

Assumption 4 (Inequalities of Parameters) The parameters and constants involved in the problem setting satisfy the following conditions:

|     | Assumption                      | Explanation                                                                                                                           |
|-----|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| (1) | a - c > γ t , ∀ t ∈ [ T ]       | The demand at p = 0 is completely censored by any inventory level, since customers will rush to buy until completely out-of-stock.    |
| (2) | γ t > 2 c, ∀ t ∈ [ T ]          | Inventory level should exceed the width of the noise support. Otherwise we can reshape the noise by capping N t at γ t - a + bp max . |
| (3) | a - bp max - c > 0              | Demands must be positive.                                                                                                             |
| (4) | γ min > a max - b min p max + c | Demands at p t = p max must be uncensored. We denote γ 0 := a max - b min p max + c for further use.                                  |
| (5) | p max ≥ a 2 b                   | Optimal price must be included in [0 , p max ] without loss of generality.                                                            |

Each assumption in Assumption 4 is justified by an explanation followed. We make Item (3) to avoid simultaneously upper-and lower-censoring effect, which is beyond the scope of this work and considered as an extension. It is worth mentioning that for those boundedness parameters (e.g. a max , b max , c, γ min ), we have to know their exact values, while we only require the existence of a, b without knowing them. Finally, for the benefit of regret analysis, we assume that the time horizon T is sufficiently large, such that its polynomial will not confound any constant or coefficient.

Assumption 5 (Large T ) For any constant r = O (1) , time horizon T is larger than any polynomial of parameters, i.e. T &gt; Ω ( ( a max b max cp max b min γ min ) r ) .

## 4 Algorithm Design

In this section, we present our core algorithm, C20CB (see Algorithm 1), which stands for a Closest-To-Zero Confidence Bound strategy that proposes asymptotically optimal prices over differentiated inventory levels and censoring effects.

## 4.1 Algorithm Design Overview

Our algorithm has two stages:

1. STAGE 1: Exploration : During the first τ = √ T rounds, the seller (we) proposes uniformly random prices in the range of [0 , p max ] . By the end of STAGE 1, we obtain ˆ a and ˆ b as plug-in estimators of a and b in the following stage.
2. STAGE 2: Optimistic Decision : We estimate the derivatives of the revenue function at discretized prices { p k,t } 's. For each p k,t , we not only estimate r ′ t ( p k,t ) but also maintain an error bar of that estimate. At each time t , we propose the price whose corresponding error bar covers 0 or closest to 0 if no covering exists.

Algorithm 1 exhibits several advantageous properties. It is suitable for processing streaming data as the constructions of ˆ a, ˆ b, ˆ r ′ t ( · ) are updated incrementally with each new observation (including e i,t , D t , 1 t ) without the need of revisiting any historical data. Additionally, it consumes ˜ O ( T 5 4 ) time complexity and O ( T 1 4 ) extra space, which are plausible for large T . A potential risk of computation might arise on the calculation of ˆ b , where ∑ τ t =1 e 1 ,t -e 2 ,t can be 0 with a small but nonzero probability. Although this event does not undermine the high-probability regret guarantee, it might still be harmful to the computational system for

## Algorithm 1 C20CB: Closest-To-Zero Confidence Bound (Main Algorithm)

- 1: Input : Parameters a max , b min , b max , p max , c and self-derived quantities τ, γ 0 , C a , C b , C F , C G , C N , C τ .
- 3: Estimate ( ˆ b, ˆ a ) ← PureExp ( τ, p max , γ 0 ) according to Algorithm 2 .
- 2: //STAGE 1: Pure Exploration for τ times.
- 4: //STAGE 2: Optimistic Acting
- 5: Define ∆ := ( C a + C b · p max ) · 1 √ τ , M := ⌊ c 2 ∆ ⌋ and w k := 2 k∆,k = -M, -M +1 , . . . , M .
- 6: Initialize ( F k , N k , G k , ∆ k ) ← ConfBoundInit ( τ, M, ˆ a, ˆ b, c, b max , p max , C b , C F , C G ) for k = -M,...,M , according to Algorithm 3 .

<!-- formula-not-decoded -->

1

- 9: Propose p t ← ˆ a 2 ˆ b and continue to t +1 (without recording feedback).
- 8: if γ t ≥ ˆ a + C a · √ τ 2 + c then
- 10: else

12:

Propose

- 11: Get ( p t , k t ) ← OptPrice ( M,γ t , { ( F k , N k , G k , ∆ k ) } k M = -M , c, γ 0 ) according to Algorithm 4 .

p

t

- 13: if k t &gt; -∞ then
- 14: Update

<!-- formula-not-decoded -->

- 15: Update N k t ← N k t +1 .
- end if
- 17: end if
- 16:
- 18: end for

## Algorithm 2 PureExp : Pure Exploration

- 1: for t = 1 , 2 , . . . , τ do
- 3: Observe demand D t , and indicators e i,t as defined in Eq. (7) for i = 1 , 2 , 3 .
- 2: Sample and propose a price p t ∼ U [0 , p max ] uniformly at random.
- 4: end for

<!-- formula-not-decoded -->

- 5: Estimate

numerical experiments. To mitigate this incident in practice, we may either extend STAGE 1 until one non-zero e 1 ,t -e 2 ,t = 1 is observed, or restart STAGE 1 at t = τ .

## 4.2 Pure-Exploration to Estimate Parameters from Biased Observations

As shown in Algorithm 2, we incorporate a uniform exploration phase to estimate a and b , bypassing the obstacle caused by demand censoring. This approach is supported by the following insight: When Y is a uniformly distributed random variable within a closed interval [ L, R ] , and X is another random variable, independent to Y and also distributed within [ L, R ] , we have:

<!-- formula-not-decoded -->

Here the second step uses the Law of Total Expectation. Eq. (6) indicates that we can derive an unbiased estimator of E [ X ] through 1 [ Y ≥ X ] even in the absence of any direct observation of X . Looking back to our algorithm, denote γ i,t := iγ t +(4 -i ) γ 0 4 , i = 1 , 2 , 3 as the three quarter points of γ t for t = 1 , 2 , . . . , T , and define

<!-- formula-not-decoded -->

D

t

1

t

:=

1

[

D

t

≤

γ

t

]

as the price, and observe and

.

## Algorithm 3 ConfBoundInit : Confidence Bound Initialization

```
for t = 1 , 2 , . . . , 2 M +1 do 2: Let k t := -M -1 + t and propose p t = w k t -( γ t -ˆ a ) ˆ b . 3: Observe D t and 1 t := 1 [ D t < γ t ] . 4: Initialize F k t ← 1 t , N k t ← 1 , G k t ← D t -γ t + c, ∆ k t ← C F · b max p max + C G + C b · 1 √ τ .
```

```
1: 5: end for
```

## Algorithm 4 OptPrice : Select Optimal Price

```
1: Initialize k t ← M as the index of arm to inspect, ρ t ← + ∞ as the smallest absolute value of the derivative estimates we have observed in Time Period t so far. 2: for k = M,M -1 , . . . , -M +1 , -M do 3: Denote p k,t := w k -( γ t -ˆ a ) ˆ b and ˆ r k,t := γ 0 -c + G k -ˆ b · p k,t · F k . 4: if ˆ r k,t -∆ k ≤ 0 ≤ ˆ r k,t + ∆ k then 5: Update k t ← k, ρ t ← 0 , and Break. 6: end if 7: Let ρ k,t := min {| ˆ r k,t -∆ k | , | ˆ r k,t + ∆ k |} as the smallest absolute derivative estimate of arm k . 8: if ρ k,t < ρ t then 9: Update ρ t ← ρ k,t and k t ← k . 10: end if 11: end for 12: if ˆ r k,t -∆ k > 0 , ∀ k = -M, -M +1 , . . . , M -1 , M then 13: Output p t ← ˆ a 2 ˆ b and k t ←-∞ . 14: else 15: Output p t ← p k t ,t and k t . 16: end if
```

When p t ∼ U [0 , p max ] , we have

<!-- formula-not-decoded -->

The last equality comes from E [ N t ] = 0 . By deploying different γ i,t at i = 1 , 2 , 3 , we can estimate a and b through the observations of e i,t according to Algorithm 2, effectively circumventing the censoring effect. A similar technique has been used by [26] to construct an unbiased estimator of valuation instead of the demand, as we are concerned. However, their application of uniform exploration might be sub-optimal as they adopt an exploration-then-exploitation design. In contrast, our algorithm uses this uniform exploration merely as a trigger of further learning. Our tight regret bound indicates that uniform exploration can still contribute to an optimal algorithm for a broad range of online learning instances.

## 4.3 Optimistic Strategy to Balance Derivatives Estimates v.s. Loss

With ˆ a and ˆ b established, we have an estimate of the underlying linear demand d t ( p ) = a -bp . However, we are still unaware of the noise distribution, which is crucial for the current optimal price, since the inventory level γ t partially censors the noise.

In order to balance the learning of noise distribution versus the loss of proposing suboptimal prices, we apply an optimistic strategy in STAGE 2. Usually, an 'optimistic strategy' chooses actions as if the best-case scenario within its current confidence region holds, thereby encouraging exploration of potentially rewarding options while still balancing risk.

STAGE 2 involves the following components:

- (i) We discretize the [ -c, c ] domain of noise CDF F ( · ) and its integration G ( · ) into small intervals of length 2 ∆ , with ∆ = O ( 1 T 1 / 4 ) . At the center of interval k (which is 2 k∆ ) for

Fig. 1: The price C20CB proposes based on confidence bounds of ˆ r k,t : (a) If there exist prices whose error bar contain 0 , then we propose the largest price among them. (b) If no error bar contains 0 but there does exist at least one below 0 , we propose the price whose corresponding error bar is closest to 0 . (c) If all error bars are above 0 , we propose p t = ˆ a 2 ˆ b .

![Image](data:image/png;base64,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)

each k = -M, -M +1 , . . . , M (with M := ⌊ c 2 ∆ ⌋ ), we maintain independent estimates of F and G , including their expectations and high-confidence error bars.

- (ii) At each time t , we construct a set of discrete prices { p k,t } M k = -M such that the quantity γ t -ˆ a + ˆ bp k,t matches the center of Interval k . Given this, we further construct an estimate of each r ′ t ( p k,t ) with plug-in estimators ˆ a, ˆ b and discrete estimators of F and G for the specific Interval k . The estimate includes its expectation and error bar (see Algorithm 3).
- (iii) Since the optimal price p ∗ t satisfies r ′ t ( p ∗ t ) = 0 , we identify the discrete price p k,t where the derivative estimate is 'possibly 0 ' or 'closest to 0 '. To make this, we design Algorithm 4 as a component of C20CB and illustrate the process in Figure 1, which includes the following three cases:
- (a) If there exists some p k,t such that the corresponding error bar (of its derivative estimate) contains 0 , we propose the largest price satisfying this condition.
- (b) If there is no p k,t whose corresponding error bar contains 0 (but there exists an error bar below 0 ), we propose the price whose error bound is closest to 0 .
- (c) If all error bars are above 0 , indicating that the reward function is monotonically increasing over the 'censoring area', we propose p t = ˆ a 2 ˆ b to exploit the noncensoring optimal price a 2 b . In this case, we do not need to record any observations nor to update any parameter/estimate.
- (iv) After proposing the price p k t ,t and observing feedback D t and 1 t , we update the estimates of F ( · ) and G ( · ) for Interval k t in which γ t -ˆ a + ˆ bp k t ,t exists.

In a nutshell, we maintain estimates and error bars of F ( · ) and G ( · ) at discrete points 2 k∆ , and map each 2 k∆ to a corresponding price p k,t once an inventory γ t occurs. Then we propose the price whose derivative estimate ˆ r k,t ± ∆ k is closest-to-zero among all k . Finally, we update the estimates with observations.

Here we provide an intuition of the optimality: On the one hand, the width of the interval can tolerate the error of mapping from 2 k∆ to p k,t , and the Lipschitzness of F and G ensures that our estimate within each small interval is roughly correct. On the other hand, we can show that the closest-to-zero derivative estimate implies a closest-to -p ∗ t price according to some locally strong convexity in the neighborhood of p ∗ t . As we have smoothness on the regret function, we suffer a quadratic loss at ( T · 1 T 1 / 4 ) 2 = O ( √ T ) cumulatively, which balances the loss of STAGE 1 that costs O ( τ ) = O ( √ T ) as well. For a rigorous regret analysis, we kindly refer the readers to Section 5.

Technical Highlights. This work is the first to introduce optimism on the derivatives and achieve optimal regret in an adversarial online learning problem. In contrast, existing works either develop optimistic algorithms on the reward (or loss) function as the original UCB strategy [37], or instead use unbiased stochastic gradients and conduct first-order methods for online optimization [30].

## 5 Regret Analysis

In this section, we analyze the cumulative regret of our algorithm and show a ˜ O ( √ T ) regret guarantee with high probability. Here we only display key lemmas and proof sketches, and we leave all of the proof details to Section B.

We first state our main theorem.

Theorem 1 (Regret). Let τ = √ T in Algorithm 1. For any adversarial { γ t } T t =1 input sequence, C20CB suffers at most ˜ O ( √ T · log T δ ) regret, with probability Pr ≥ 1 -δ .

Proof. In order to prove Theorem 1, we have to show the following three components:

1. The reward function r t ( p ) is unimodal. Also, r t ( p ) is smooth at p ∗ t , and is strongly concave on a neighborhood of p ∗ t .
2. The estimation error of a and b are bounded by O ( 1 T 1 / 4 ) at the end of STAGE 1.
3. The price whose derivative estimate has the closest-to-zero confidence bound is asymptotically close to p ∗ t .

In the following, we present each corresponding lemma regarding to the roadmap above.

Lemma 1 (Revenue Function r t ( p ) ). For the expected revenue function r t ( p ) defined in eq. (2) , the following properties hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2. There exists a constant L r &gt; 0 such that r ′ ( p ) is L r -Lipschitz .
3. r ′ ( p ) is monotonically non-increasing .
4. r t ( p ) is unimodal : There exists a unique p ∗ t ∈ [0 , a b ] such that r ′ t ( p ∗ t ) = 0 , and r t ( p ) monotonically increase in [0 , p ∗ t ] and decrease in [ p ∗ t , a b ] . Notice that a b &gt; p max according to Assumption 4.
5. r t ( p ) is smooth at p ∗ t : There exists a constant C s &gt; 0 such that r t ( p ∗ t ) -r t ( p ) ≤ C s ( p ∗ t -p ) 2 , ∀ p ∈ [0 , p max ] .

6. r t ( p ) is locally strongly concave : There exist ϵ t &gt; 0 and C ϵ &gt; 0 such that ∀ p 1 , p 2 ∈ [ p ∗ t -ϵ t , p ∗ t + ϵ t ] we have | r ′ t ( p 1 ) -r ′ t ( p 2 ) | ≥ C ϵ · | p 1 -p 2 | .
7. There exists a constant C v &gt; 0 such that for any t ∈ [ T ] and p ∈ ( p ∗ t -ϵ t , p ∗ t + ϵ t ) , we have | r t ( p ∗ t ) -r t ( p ) | ≤ C v · ( r ′ t ( p )) 2 .

Proof (sketch). Property 1 is from integration by parts. Property 2 is proved by the Lipschitzness of F ( x ) . Property 3 is from the monotonicity of F ( x ) . Property 4 can be proved by two steps: (4.1) The existence of p ∗ t ∈ [0 , a b ] by r ′ t (0) &gt; 0 and r ′ t ( a b ) &lt; 0 ; (4.2) The uniqueness of p ∗ t by contradiction. Properties 5 and 6 are mainly from the Lipschitzness of r ′ t ( p ) . Property 7 comes from r ′ t ( p ∗ t ) = 0 and the strong concavity (Property 6). Please kindly check Section B.1 as a detailed proof of Lemma 1. ⊓ ⊔

The properties of r t ( p ) and r ′ t ( p ) enable us to upper bound the cost of estimation error and decision bias. In the following, we propose a lemma that serves as a milestone of estimation error upper bounds.

Lemma 2 (Estimation Error of a and b ). For any η &gt; 0 , δ &gt; 0 , with probability Pr ≥ 1 -2 ηδ , we have

<!-- formula-not-decoded -->

where C a := p max ( C b + b max · √ 1 2 log 2 ηδ ) and C b := 8 b 2 max γ min -γ 0 · √ 1 2 log 2 ηδ .

Proof (sketch). The key observation to prove this lemma lies in the expectation of each e i,t that indicates whether the demand exceeds certain level under uniformly distributed prices. According to Law of Total Expectation, for any γ ∈ [ γ min , γ t ] at any time t ≤ τ in STAGE 1, we have

<!-- formula-not-decoded -->

With this property, we may construct method-of-moment estimates from e i,t in STAGE 1, which eliminate the influence of noise distribution and achieve an unbiased estimator of 1 b (and therefore ˆ b asymptotically). With ˆ b serving as a plug-in estimator, we later achieve ˆ a . We obtain the error bounds by applying Hoeffding's Inequalities. ⊔ ⊓

We defer the detailed proof of Lemma 2 to Section B.2. With the help of Lemma 2, we may upper bound the estimation error of r ′ t ( p ) at discrete prices. The error bound is displayed as the following lemma.

Lemma 3 (Estimation Error of r ′ t ( p k,t ) ). There exists constants C N &gt; 0 , C τ &gt; 0 such that for any t ∈ [ T ] , k ∈ {-M, -M +1 , . . . , M } , with probability Pr ≥ 1 -6 ηδ we have

<!-- formula-not-decoded -->

Here N k ( t ) and ∆ k ( t ) denotes the value of N k and ∆ k at the beginning of time period t .

Proof (sketch). Denote N k ( t ) , F k ( t ) and G k ( t ) as the value of N k , F k and G k at the beginning of time t . From Algorithm 1, we have | ˆ r k,t -r ′ t ( p k,t ) | ≤ | G k ( t ) -( G ( c ) -G ( γ t -a + bp k,t )) | + p k,t | ˆ bF k -bF ( γ t -a + bp k,t ) | , and we may separately upper bound each of these two differences.

1. For the term on G , we may split the quantity | G k ( t ) -( G ( c ) -G ( γ t -a + bp k,t )) | into three terms:
2. (i) | G k ( t ) -E [ G k ( t )] | as concentration, which is bounded by Hoeffding's inequality.
3. (ii) | E [ G k ( t )] -( G ( c ) -G (2 k∆ )) | as the G k deviation within each 'bin' of estimation (whose center is w k = 2 k∆ ). By definition of G k ( t ) , we have E [ G k ( t )] = 1 N k ( t ) ∑ t -1 s =1 1 [ k s == k ]( G ( c ) -G ( γ s -a + bp k s ,s )) , and therefore we have

<!-- formula-not-decoded -->

The last line is by Lipschitzness of G ( x ) .

- (iii) | G (2 k∆ ) -G ( γ t -a + bp k,t ) | , which is similar to the last step of (ii) above.
2. For the term on F , we may split the quantity | ˆ bF k -bF ( γ t -a + bp k,t ) | into four terms:
- (i) | ˆ bF k ( t ) -bF k ( t ) | ≤ | ˆ b -b | by the nature of F k ( t ) ≤ 1 , and further bounded by Lemma 2.
- (ii) | F k ( t ) -E [ F k ( t )] | , (iii) | E [ F k ( t )] -F (2 k∆ ) | , and (iv) | F (2 k∆ ) -F ( γ t -a + bp k,t ) | , are bounded in the same way presented above for G respectively.

By plugging in the estimation error from Lemma 2, we prove the present lemma. ⊔ ⊓

Please refer to Section B.3 as a rigorous proof of Lemma 3. Given this lemma, the derivatives of each discrete price p k,t is truthfully reflected by their corresponding error bound. Therefore, we intuitively see that the closest-to-zero confidence bound represents the closest-top ∗ t discrete price. We formulate this intuition as the following lemma:

Lemma 4 (Closest-To-Zero Confidence To Performance). Denote ∆ k ( t ) as the value of ∆ k at the beginning of period t . There exists two constants N 0 &gt; 0 , N 1 &gt; 0 such that for any t = 1 , 2 , . . . , T in STAGE 2, either of the following events occurs with high probability.

1. When ∃ k ∈ {-M, -M +1 , . . . , M } such that the Number k confidence bound satisfies ˆ r k,t -∆ k ( t ) ≤ 0 ≤ ˆ r k,t + ∆ k ( t ) , and also N k ( t ) &gt; N 0 , then we have p k,t ∈ [ p ∗ t -ϵ t , p ∗ t + ϵ t ] . Furthermore, there exists constant C in such that r t ( p ∗ t ) -r t ( p k,t ) ≤ C in ( 1 N k ( t ) + 1 τ ) .
2. When there exists no confidence bound that contains 0 , i.e. either ˆ r k,t -∆ k ( t ) &gt; 0 or ˆ r k,t + ∆ k ( t ) &lt; 0 , ∀ k ∈ {-M, -M +1 , . . . , M -1 , M } (happens at least for one k ), and also N k ( t ) &gt; N 1 , then we have

<!-- formula-not-decoded -->

(where C κ = L r ( C a + C b · p max ) 2 b min ) and p k,t ∈ [ p ∗ t -ϵ t , p ∗ t + ϵ t ] . Furthermore, there exists constant C out such that r t ( p ∗ t ) -r t ( p k,t ) ≤ C out ( 1 N k ( t ) + 1 τ ) .

Proof (sketch). The intuition to prove Lemma 4 is twofold:

1. When an error bar contains 0 , the true derivative of the corresponding price is close to 0 within the distance of its error bound. By applying Lemma 1 Property (7), we may upper bound the performance loss with the square of its derivatives, which is further upper bounded by the square of error bound.
2. When no error bar contains 0 , there exists an adjacent pair of prices whose error bars are separated by y = 0 . On the one hand, their derivatives difference is upper bounded due to the Lipschitzness of r ′ t ( p ) . On the other hand, the same derivatives difference is lower bounded by the closest-to-zero confidence bound. Therefore, the gap between y = 0 and the closest-to-zero confidence bound should be very small, and we still have a comparably small | r ′ t ( p t ) | if p t possesses that confidence bound. As a consequence, we have similar upper bound on the performance loss comparing with Case (1), up to constant coefficients. ⊔ ⊓

The detailed proof of Lemma 4 is presented in Section B.4. Finally, we have a lemma that upper bounds the regret of proposing p t = ˆ a 2 ˆ b under special conditions.

Lemma 5 (Proposing ˆ a 2 ˆ b ). When γ t &gt; ˆ a + C a · 1 √ τ 2 + c and when ˆ r k,t -∆ k ( t ) &gt; 0 , ∀ k = -M, -M +1 , . . . , M , we have p ∗ t = a 2 b and there exists a constant C non such that

<!-- formula-not-decoded -->

The intuition of Lemma 5 is that a 2 b is the optimal price without censoring, and we only need to show that either the optimal price or a 2 b is not censored (which are equivalent as the revenue function is unimodal). We defer its proof to Section B.5. This lemma serves as the last puzzle of the proof. With all lemmas above, we upper bound the overall regret as follows:

<!-- formula-not-decoded -->

Here the first two rows are a decomposition of STAGE 1 (for τ rounds), STAGE 2 Initialization (for 2 M +1 rounds), STAGE 2 Case (a) and (b) (proposing p t = p k t , Lemma 4) and STAGE 2 Case (c) (proposing p t = ˆ a 2 ˆ b , Lemma 5). The fourth row is by re-classification of 1 N k t ( t ) according to k , which leads to a summation over harmonic series (since each N k t ( t ) increases by 1 for the same class k t = k ). By applying a union bound, Eq. (16) holds with probability

<!-- formula-not-decoded -->

Here the first part comes from Lemma 2, and the second part comes from Lemma 3 for any t ∈ [ T ] and k ∈ {-M,...,M } . Let η := C a + C b · p max 10 c · T 5 / 4 , and we show that Theorem 1 holds. ⊓ ⊔

Remark 1. This ˜ O ( √ T ) regret upper bound is near-optimal up to log T factors, as it matches the Ω ( √ T ) information-theoretic lower bound proposed by [10] for a no-censoring problem setting with linear noisy demand.

## 6 Discussions

Here we provide some insights on the current limitations and potential extension of this work to a broader field of research.

Extensions to Non-linear Demand Curve. In this work, we adopt a linear-and-noisy model for the potential demands, which is standard in dynamic pricing literature. Also, we utilize the unimodal property brought by this linear demand model, even after the censoring effect is imposed. If we generalize our methodologies to nonlinear demand functions, we have to carefully distinguish between potential local optima and saddle points that may also cause r ′ t ( p ) = 0 for some sub-optimal p . We conjecture an Ω ( T m +1 2 m +1 ) lower bound in that case, where m is the order of smoothness. It is worth investigating whether the censoring effect will introduce new local optimals or swipe off existing ones in multimodal settings.

Generalization to Unbounded Noises. We assume the noise is bounded in a constant-width range. From the analysis in Section 5, we know that the threshold of learning the optimal price in our problem setting is still the estimation of parameters. Therefore, this boundedness assumption streamlines the pure-exploration phase, facilitates the estimation of the parameters b and a , and scales down the cumulative regret. While our methods and results can be extended to unbounded O ( 1 log T ) -subGaussian noises by simple truncation, challenges remain for handling generic unbounded noises. Moreover, the problem can be more sophisticated with dual-censoring , both from above by inventory-as we have discussed- and from below by 0 , especially when considering unbounded noises.

Extensions to Non-Lipschitz Noise CDF. In this work, we assume the noise CDF as a Lipschitz function as many pricing-related works did [26,52]. This assumption enables the local smoothness at p ∗ t and leads to a quadratic loss. However, this prevents us from applying our algorithm to non-Lipschitz settings, which even includes the noise-free setting. In fact, although we believe that a better regret rate exists for the noise-free setting, we have to state that the hardness of the problem is completely different with Lipschitz noises versus without it. Although a Lipschitz noise makes the observation 'more blur', it also makes the revenue curve 'more smooth'. We would like to present an analog example from the feature-based dynamic pricing problem: When the Gaussian noise N (0 , σ 2 ) is either negligible (with σ &lt; 1 T , see [21]) or super significant (with σ &gt; 1 , see [61]), the minimax regret is O (log T ) . However, existing works can only achieve O ( √ T ) regret when σ ∈ [ 1 T , 1] . We look forward to future research on our problem setting once getting rid of the Lipschitzness assumption.

Extensions to Contextual Pricing. In this work, we assume a and b are static, which may not hold in many real scenarios. Example 1 serves as a good instance, showcasing significant fluctuations in popularity across different performances. A reasonable extension of our work would be modeling a and b as contextual parameters. Similar modelings have been adopt by [54] and [6] in the realm of personalized pricing research.

Societal Impacts. Our research primarily addresses a non-contextual pricing model that does not incorporate personal or group-specific data, thereby adhering to conventional fairness standards relating to temporal, group, demand and utility discrepancies as outlined by [20] and [17]. However, the non-stationarity of inventory levels could result in varying fulfillment rate over time, i.e. the proportions of satisfied demands at { p ∗ t } 's might be different for t = 1 , 2 , . . . , T . This raises concern regarding unfairness in fulfillment rate [48], particularly for products of significant social and individual importance.

## 7 Conclusions

In this paper, we studied the online pricing problem with adversarial inventory constraints imposed over time series. We introduced an optimistic strategy and a C20CB algorithm that is capable of approaching the optimal prices from inventory-censored demands. Our algorithm achieves a regret guarantee of ˜ O ( √ T ) with high probability, which is informationtheoretically optimal. To the best of our knowledge, we are the first to address this adversarialinventory pricing problem, and our results indicate that the demand-censoring effect does not substantially increase the hardness of pricing in terms of minimax regret.

## Acknowledgement

Jianyu Xu started this work as a Ph.D. student at University of California, Santa Barbara. Xi Chen would like to thank the support from NSF via the Grant IIS-1845444.

## References

1. Agarwal, A., Hsu, D., Kale, S., Langford, J., Li, L., Schapire, R.: Taming the monster: A fast and simple algorithm for contextual bandits. In: International Conference on Machine Learning (ICML-14). pp. 1638-1646 (2014)
3. Auer, P., Cesa-Bianchi, N., Fischer, P.: Finite-time analysis of the multiarmed bandit problem. Machine learning 47 (2), 235-256 (2002)
2. Amin, K., Rostamizadeh, A., Syed, U.: Repeated contextual auctions with strategic buyers. In: Advances in Neural Information Processing Systems (NIPS-14). pp. 622-630 (2014)
4. Auer, P., Cesa-Bianchi, N., Freund, Y., Schapire, R.E.: The nonstochastic multiarmed bandit problem. SIAM Journal on Computing 32 (1), 48-77 (2002)
6. Ban, G.Y., Keskin, N.B.: Personalized dynamic pricing with machine learning: High-dimensional features and heterogeneous elasticity. Management Science 67 (9), 5549-5568 (2021)
5. Badanidiyuru, A., Kleinberg, R., Slivkins, A.: Bandits with knapsacks. In: Annual Symposium on Foundations of Computer Science (FOCS-13). pp. 207-216. IEEE (2013)
7. Besbes, O., Zeevi, A.: Dynamic pricing without knowing the demand function: Risk bounds and nearoptimal algorithms. Operations Research 57 (6), 1407-1420 (2009)
9. Besbes, O., Zeevi, A.: On the (surprising) sufficiency of linear models for dynamic pricing with demand learning. Management Science 61 (4), 723-739 (2015)
8. Besbes, O., Zeevi, A.: Blind network revenue management. Operations research 60 (6), 1537-1550 (2012)
10. Broder, J., Rusmevichientong, P.: Dynamic pricing under a general parametric choice model. Operations Research 60 (4), 965-980 (2012)
12. Chen, B., Chao, X., Ahn, H.S.: Coordinating pricing and inventory replenishment with nonparametric demand learning. Operations Research 67 (4), 1035-1052 (2019)
11. Bu, J., Simchi-Levi, D., Wang, C.: Context-based dynamic pricing with partially linear demand model. In: Advances in Neural Information Processing Systems (2022)
13. Chen, B., Chao, X., Shi, C.: Nonparametric learning algorithms for joint pricing and inventory control with lost sales and censored demand. Mathematics of Operations Research 46 (2), 726-756 (2021)
15. Chen, B., Wang, Y., Zhou, Y.: Optimal policies for dynamic pricing and inventory control with nonparametric censored demands. Management Science 70 (5), 3362-3380 (2024)
14. Chen, B., Chao, X., Wang, Y.: Data-based dynamic pricing and inventory control with censored demand and limited price changes. Operations Research 68 (5), 1445-1456 (2020)
16. Chen, Q., Jasin, S., Duenyas, I.: Joint learning and optimization of multi-product pricing with finite resource capacity and unknown demand parameters. Operations Research 69 (2), 560-573 (2021)

17. Chen, X., Simchi-Levi, D., Wang, Y.: Utility fairness in contextual dynamic pricing with demand learning. arXiv preprint arXiv:2311.16528 (2023)
19. Chu, W., Li, L., Reyzin, L., Schapire, R.: Contextual bandits with linear payoff functions. In: International Conference on Artificial Intelligence and Statistics (AISTATS-11). pp. 208-214 (2011)
18. Chen, X., Zhang, X., Zhou, Y.: Fairness-aware online price discrimination with nonparametric demand models. arXiv preprint arXiv:2111.08221 (2021)
20. Cohen, M.C., Elmachtoub, A.N., Lei, X.: Price discrimination with fairness constraints. Management Science (2022)
22. Cohen, M.C., Miao, S., Wang, Y.: Dynamic pricing with fairness constraints. Available at SSRN 3930622 (2021)
21. Cohen, M.C., Lobel, I., Paes Leme, R.: Feature-based dynamic pricing. Management Science 66 (11), 4921-4943 (2020)
23. Cohen, M.C., Perakis, G., Pindyck, R.S.: A simple rule for pricing with limited knowledge of demand. Management Science 67 (3), 1608-1621 (2021)
25. Crane, J.H., Balerdi, C.F., Maguire, I.: Sugar apple growing in the florida home landscape. Gainesville: University of Florida (2005)
24. Cournot, A.A.: Researches into the Mathematical Principles of the Theory of Wealth. Macmillan (1897)
26. Fan, J., Guo, Y., Yu, M.: Policy optimization using semiparametric models for dynamic pricing. Journal of the American Statistical Association 119 (545), 552-564 (2024)
28. Gallego, G., Van Ryzin, G.: Optimal dynamic pricing of inventories with stochastic demand over finite horizons. Management Science 40 (8), 999-1020 (1994)
27. Folland, G.B.: Real analysis: modern techniques and their applications, vol. 40. John Wiley &amp; Sons (1999)
29. Golrezaei, N., Jaillet, P., Liang, J.C.N.: Incentive-aware contextual pricing with non-parametric market noise. arXiv preprint arXiv:1911.03508 (2019)
31. Javanmard, A., Nazerzadeh, H.: Dynamic pricing in high-dimensions. The Journal of Machine Learning Research 20 (1), 315-363 (2019)
30. Hazan, E.: Introduction to online convex optimization. Foundations and Trends in Optimization 2 (3-4), 157-325 (2016)
32. Karan, A., Balepur, N., Sundaram, H.: Designing fair systems for consumers to exploit personalized pricing. arXiv preprint arXiv:2409.02777 (2024)
34. Keskin, N.B., Zeevi, A.: Dynamic pricing with an unknown demand model: Asymptotically optimal semi-myopic policies. Operations Research 62 (5), 1142-1167 (2014)
33. Keskin, N.B., Li, Y., Song, J.S.: Data-driven dynamic pricing and ordering with perishable inventory in a changing environment. Management Science 68 (3), 1938-1958 (2022)
35. Kleinberg, R.: Nearly tight bounds for the continuum-armed bandit problem. Advances in Neural Information Processing Systems 17 , 697-704 (2004)
37. Lai, T.L., Robbins, H.: Asymptotically efficient adaptive allocation rules. Advances in applied mathematics 6 (1), 4-22 (1985)
36. Kleinberg, R., Leighton, T.: The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In: IEEE Symposium on Foundations of Computer Science (FOCS-03). pp. 594-605. IEEE (2003)
38. Langford, J., Zhang, T.: The epoch-greedy algorithm for contextual multi-armed bandits. In: Advances in Neural Information Processing Systems (NIPS-07). pp. 817-824 (2007)
40. Lobo, M.S., Boyd, S.: Pricing and learning with uncertain demand. In: INFORMS Revenue Management Conference. Citeseer (2003)
39. Liu, A., Leme, R.P., Schneider, J.: Optimal contextual pricing and extensions. In: Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms (SODA-21). pp. 1059-1078. SIAM (2021)
41. Luo, Y., Sun, W.W., et al.: Distribution-free contextual dynamic pricing. arXiv preprint arXiv:2109.07340 (2021)
43. Miao, S., Chen, X., Chao, X., Liu, J., Zhang, Y.: Context-based dynamic pricing with online clustering. arXiv preprint arXiv:1902.06199 (2019)
42. Meissner, J., Strauss, A.: Network revenue management with inventory-sensitive bid prices and customer choice. European Journal of Operational Research 216 (2), 459-468 (2012)
44. Miao, S., Wang, Y.: Demand balancing in primal-dual optimization for blind network revenue management. arXiv preprint arXiv:2404.04467 (2024)
46. van Ryzin, G.J.: Models of demand. Oxford handbook of pricing management pp. 340-380 (2012)
45. Perakis, G., Roels, G.: Robust controls for network revenue management. Manufacturing &amp; Service Operations Management 12 (1), 56-76 (2010)
47. Simchi-Levi, D., Xu, Y., Zhao, J.: Blind network revenue management and bandits with knapsacks under limited switches. arXiv preprint arXiv:1911.01067 (2019)
48. Spiliotopoulou, E., Conte, A.: Fairness ideals in inventory allocation. Decision Sciences 53 (6), 985-1002 (2022)

49. Talluri, K., Van Ryzin, G.: An analysis of bid-price controls for network revenue management. Management science 44 (11-part-1), 1577-1593 (1998)
51. Thompson, W.R.: On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika 25 (3/4), 285-294 (1933)
50. Talluri, K.T., Van Ryzin, G.J.: The theory and practice of revenue management, vol. 68. Springer Science &amp; Business Media (2006)
52. Tullii, M., Gaucher, S., Merlis, N., Perchet, V.: Improved algorithms for contextual dynamic pricing. arXiv preprint arXiv:2406.11316 (2024)
54. Wang, H., Talluri, K., Li, X.: On dynamic pricing with covariates. Operations Research (2025)
53. Vera, A., Banerjee, S., Gurvich, I.: Online allocation and pricing: Constant regret via bellman inequalities. Operations Research 69 (3), 821-840 (2021)
55. Wang, Y., Chen, B., Simchi-Levi, D.: Multimodal dynamic pricing. Management Science 67 (10), 6136-6152 (2021)
57. Wang, Z., Deng, S., Ye, Y.: Close the gaps: A learning-while-doing algorithm for single-product revenue management problems. Operations Research 62 (2), 318-331 (2014)
56. Wang, Y., Du, S., Balakrishnan, S., Singh, A.: Stochastic zeroth-order optimization in high dimensions. In: International conference on artificial intelligence and statistics. pp. 1356-1365. PMLR (2018)
58. Xu, J., Jain, V., Wilder, B., Singh, A.: Online decision making with generative action sets. arXiv preprint arXiv:2509.25777 (2025)
60. Xu, J., Wang, X., Wang, Y.X., Jiang, J.: Joint pricing and resource allocation: An optimal online-learning approach. arXiv preprint arXiv:2501.18049 (2025)
59. Xu, J., Qiao, D., Wang, Y.X.: Doubly fair dynamic pricing. In: International Conference on Artificial Intelligence and Statistics. pp. 9941-9975. PMLR (2023)
61. Xu, J., Wang, Y.X.: Logarithmic regret in feature-based dynamic pricing. Advances in Neural Information Processing Systems 34 , 13898-13910 (2021)
63. Xu, J., Wang, Y.X.: Pricing with contextual elasticity and heteroscedastic valuation. In: Forty-first International Conference on Machine Learning (2024)
62. Xu, J., Wang, Y.X.: Towards agnostic feature-based dynamic pricing: Linear policies vs linear valuation with unknown noise. International Conference on Artificial Intelligence and Statistics (AISTATS) (2022)

## Appendix

## A More Related Works

Here we discuss more related works as a complement to Section 2.

Contextual pricing: Linear valuation and binary-censored demand. A surge of research has focused on feature-based (or contextual ) dynamic pricing [2,21,39,43]. These works considered situations where each pricing period is preceded by a context, influencing both the demand curve and noise distribution. Specifically, [21,31,61] explored a linear valuation framework with known distribution noise, leading to binary customer demand outcomes based on price comparisons to their valuations. Expanding on this, [26,29,41,62] examined similar models but with unknown noise distributions. In another vein, [6,54,63] investigated personalized pricing where demand is modeled as a generalized linear function sensitive to contextual price elasticity. Many of these works on valuation-based contextual pricing also assumed a censored demand: The seller only observes a binary feedback determined by a comparison of price with valuation, instead of observing the valuation directly. However, it is important to differentiate between the linear (potential) demand model we assumed and their linear valuation models, and there exists no inclusive relationship to each other.

Dynamic pricing under constraints. A variety of research works have been devoted to the field of dynamic pricing under specific concerns, which restricts the stages and outcomes of the market. This includes resource allocation [16,53,60] and price discrimination [18,20,22,32,59] as two typical instances. In this work, the constraints are twofold: First, it blocks our observations to the real potential demand, leading to biased estimates. Second, it restricts us from fulfilling the potential demand, leading to shifted targets (the optimal prices).

Multi-armed bandits. Multi-armed bandits (MAB) formalize sequential decision-making under uncertainty via the exploration-exploitation tradeoff, originating in the stochastic setting studied by [37]. Modern finite-time analyses led to widely used algorithms such as UCB [3], Thompson Sampling [51], and the EXP-3/4 family [4]. Contextual bandits extend MAB by conditioning rewards on observed features, enabling personalization and decision-making with side information. Representative algorithmic and theoretical foundations include Epoch greedy [38], LinUCB [19] and Taming-the-monster [1]. [5] introduces a 'Bandits with knapsacks' model that incorporates resource and budget limitations. A separated stream of 'Continuumarmed bandits', introduced by [35], generalize discrete action sets to continuous domains and exploit smoothness/metric structure. Their method is applicable to our problem setting but will lead to a sub-optimal O ( T 3 / 5 ) regret. A recent model of 'Generative bandits' [58] allows new arms to be generated on the fly (at a certain cost) for free future reuse. From a more general perspective, the stream of works in 'Zeroth-order optimization (ZOO)' [56] also belongs to bandits optimization.

## B Proof Details

## B.1 Proof of Lemma 1

Proof. We prove each property sequentially.

1. For r t ( p ) , we have:

<!-- formula-not-decoded -->

Here we adopt the notation f ( x ) as proximal derivatives of F ( x ) . According to Rademacher's Theorem (see Section 3.5 of [27]), given that F ( x ) is Lipschitz, the measure of x such that f ( x ) does not exist is zero, hence the integral holds. Here the eighth line comes from ∫ c -c f ( x ) dx = F ( c ) -F ( -c ) = 1 and ∫ c -c xf ( x ) dx = E [ x ] = 0 . Given the close form of r t ( p ) , we derive the form of r ′ t ( p ) .

2. As we have assumed, F ( x ) is L F -Lipschitz, and therefore F ( γ t -a + bp ) is b max L F -Lipschitz, and bpF ( γ t -a + bp ) is ( b max + b 2 max p max L F ) -Lipschitz. Also, we have dG ( γ t -a + bp ) dp = b · F ( γ t -a + bp ) ∈ [0 , b max ] . Let L r := (2 b max + b 2 max p max L F ) , and we know that r ′ t ( p ) is L r -Lipschitz.
3. On the one hand, we have

<!-- formula-not-decoded -->

On the other hand, for any ∆ p &gt; 0 , we have

<!-- formula-not-decoded -->

Since r ′ t ( p ) = γ t -c + G ( c ) -G ( γ t -a + bp ) -bpF ( γ t -a + bp ) , we know that both components are monotonically non-increasing.

4. We first show the existence of p ∗ t ∈ [0 , a b ] such that r ′ t ( p ∗ t ) = 0 . Recall that G ( c + x ) = G ( c ) + x for ∀ x &gt; 0 , and G ( c ) -G ( -c ) = ∫ c -c F ( ω ) dω ≥ 0 , and that γ t &gt; 2 c &gt; c as we

assumed. Given those, we have:

<!-- formula-not-decoded -->

Also, r ′ t ( p ) is Lipschitz as we proved above. Therefore, ∃ p ∗ t ∈ (0 , a b ) such that r ′ t ( p ∗ t ) = 0 . Now we show the uniqueness of p ∗ t . If there exists 0 &lt; p ∗ t &lt; q ∗ t &lt; a b such that r ′ t ( p ∗ t ) = r ′ t ( q ∗ t ) = 0 , then it leads to.

<!-- formula-not-decoded -->

This leads to contradictions that r ′ t ( p ∗ t ) = 0 . Therefore, p ∗ t is unique. Given this, we know that r t ( p ) is unimodal, which increases on (0 , p ∗ t ) and decreases on ( p ∗ t , a b ) .

5. Since p ∗ t is unique, and r ′ t ( p ) is L r -Lipschitz, we have:

<!-- formula-not-decoded -->

6. From the proof of part (4), we know that F ( γ t -a + bp ∗ t ) &gt; 0 (or otherwise r ′ t ( p ∗ t ) &gt; 0 leading to contradiction). Denote ϵ t := F ( γ t -a + bp ∗ t ) 2 L F b max , and we have:

<!-- formula-not-decoded -->

Let C ϵ := b min 2 · inf γ t ∈ [ γ min ,γ max ] F ( γ t -a + bp ∗ t ) . As [ γ min , γ max ] is a close set and F ( γ t -a + bp ∗ t ) holds for any γ t ∈ [ γ min , γ max ] , we know that C ϵ &gt; 0 is a universal constant. Given this coefficient, for any p 1 , p 2 ∈ [ p ∗ t -ϵ t , p ∗ t + ϵ t ] , p 1 &lt; p 2 , we have

<!-- formula-not-decoded -->

Here the third line is because 0 &lt; p 1 &lt; p 2 and therefore 0 &lt; F ( γ t -a + bp 1 ) ≤ F ( γ t -a + bp 2 ) .

7. According to part (6), for any p ∈ ( p ∗ t -ϵ t , p ∗ t + ϵ t ) , we have | r ′ t ( p ) | = | r ′ t ( p ) -r ′ t ( p ∗ t ) | ≥ C ϵ · | p -p ∗ t | . Therefore, we have

<!-- formula-not-decoded -->

Let C v := L r 2 C 2 ϵ and the property is proven.

## B.2 Proof of Lemma 2

Proof. Recall that γ 0 := a max -b min p max + c . Notice that

<!-- formula-not-decoded -->

Also, for any t = 1 , 2 , . . . , τ in STAGE 1, and any γ such that γ min ≤ γ t , we have

<!-- formula-not-decoded -->

Here the first row is due to Law of Total Expectation, and the last row is due to the zero-mean assumption of N t (see Assumption 3). Given this equation, we have:

<!-- formula-not-decoded -->

According to Hoeffding's Inequality, we have with Pr ≥ 1 -ηδ :

<!-- formula-not-decoded -->

Based on this concentration, we upper bound the estimation error between b and ˆ b by the end of STAGE 1:

<!-- formula-not-decoded -->

Here the fifth row requires 1 8 bp max ≥ 1 γ min -γ 0 √ 1 2 log 2 ηδ · 1 √ τ , which further requires T ≥ ( 8 bp max γ min -γ 0 ) 4 · 1 4 log 2 2 ηθ . According to Assumption 5, we know that this inequality holds. Denote C b := 4 b 2 max p max γ min -γ 0 · √ 1 2 log 2 ηδ and we have | ˆ b -b | ≤ C b · 1 √ τ with high probability.

Again, according to Hoeffding's Inequality, we have with Pr ≥ 1 -ηδ :

<!-- formula-not-decoded -->

Hence we have

<!-- formula-not-decoded -->

Denote C a := p max ( C b + b max √ 1 2 log 2 ηδ ) and the lemma is proven.

## B.3 Proof of Lemma 3

Proof. Here we consider the time periods before time t , and we use an index s to denote each time period s = 1 , 2 , . . . , t -1 , t . As a consequence, we have the notations D s , γ s , k s , 1 s corresponding to D t , γ t , k t , 1 t as we defined in Section 3 and Section 4. Also, we denote N k ( t ) , F k ( t ) and G k ( t ) as the value of N k , F k and G k at the beginning of time period t .

From Algorithm 1, we have

<!-- formula-not-decoded -->

Notice that G k ( t ) = 1 N k ( t ) · ∑ t -1 s =1 1 [ k s == k ] · ( D s -γ s + c ) . Also, for each D s on the price p k s ,s , we have

<!-- formula-not-decoded -->

Recall that W k = 2 k∆ . Hence we have

<!-- formula-not-decoded -->

Also, due to the fact that G ( x ) is 1 -Lipschitz, we have

<!-- formula-not-decoded -->

Therefore, we know that

<!-- formula-not-decoded -->

Also, since each D s -γ s + c &lt; a max + c , according to Hoeffding's Inequality, with Pr ≥ 1 -ηδ we have

<!-- formula-not-decoded -->

On the other hand, since

<!-- formula-not-decoded -->

Similar to Eq. (36), since F ( x ) is L F -Lipschitz, we have

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

Also, since 1 s = 1 [ D s &lt; γ s ] , according to Hoeffding's inequality, with Pr ≥ 1 -ηδ we have

<!-- formula-not-decoded -->

As a consequence, we have bounded the estimation error of ˆ r k,t from r ′ t ( p k,t ) :

<!-- formula-not-decoded -->

Here

<!-- formula-not-decoded -->

Finally, we apply the union bound on the probability, and know that Eq. (43) holds with probability Pr ≥ 1 -6 ηδ .

## B.4 Proof of Lemma 4

Proof. We first prove the lemma under Case 1 when some confidence bound contains 0 . Denote ρ t := min {| r ′ t ( p ∗ t -ϵ t ) | , | r ′ t ( p ∗ t + ϵ t ) |} , and we know that ρ t &gt; 0 due to the uniqueness of p ∗ t .

Now, let N 0 := 36 C 2 N ρ 2 t , where C N is the constant coefficient define in Lemma 3. Given this, when N k ( t ) ≥ N 0 , we have

<!-- formula-not-decoded -->

Also, since T is assumed as larger than any constant (see Assumption 5), we have T ≥ 1296 C 4 τ ρ 4 t , and therefore

<!-- formula-not-decoded -->

Given eq. (45) and eq. (46), we know that ∆ k ( t ) = C N · 1 √ N k ( t ) + C τ · 1 √ τ ≤ ρ t 6 + ρ t 6 = ρ t 3 . Now, if 0 ∈ [ˆ r k,t -∆ k ( t ) , ˆ r k,t + ∆ k ( t )] , we have | ˆ r k,t | ≤ ∆ k ( t ) and therefore

<!-- formula-not-decoded -->

Since r ′ t ( p ) is monotonically non-increasing, any p ∈ [0 , p max ] satisfying r ′ t ( p ) &lt; ρ t should satisfy p ∈ ( p ∗ t -ϵ t , p ∗ t + ϵ t ) . Therefore, we have p k,t ∈ ( p ∗ t -ϵ t , p ∗ t + ϵ t ) . According to Lemma 1 Property (7), we have:

<!-- formula-not-decoded -->

Let C in := 8 C v max { C 2 N , C 2 τ } and the first part of Lemma 4 holds. Now let us prove the lemma under Case 2 when no confidence bound contains 0. Formally stated, we have

<!-- formula-not-decoded -->

Denote θ t := inf k min {| ˆ r k,t + ∆ k ( t ) | , | ˆ r k,t -∆ k ( t ) |} , and we know that min {| ˆ r k t ,t + ∆ k ( t ) | , | ˆ r k t ,t -∆ k ( t ) |} θ t &gt; 0 , where k t is the k such that p k,t is proposed at time t . Therefore, we have | ˆ r k,t + ∆ k ( t ) | ≥ θ t and | ˆ r k,t -∆ k ( t ) | ≥ θ t , ∀ k . According to the prerequisite of Lemma 4 Part (2), there exists k 0 such that

<!-- formula-not-decoded -->

Also, since r ′ t ( p ) is L r -Lipschitz, we have

<!-- formula-not-decoded -->

As T is sufficiently large, we have L r b min ( C a + C b p max ) · 1 T 1 / 4 ≤ 1 6 · ρ t where ρ t := min {| r ′ t ( p ∗ t -ϵ t ) | , | r ′ t ( p ∗ t + ϵ t ) |} . Let N 1 := 144 C 2 N ρ 2 t . Similar to Eq. (45) and Eq. (46), we have

<!-- formula-not-decoded -->

Hence, we have

<!-- formula-not-decoded -->

Since r ′ t ( p ) is monotonically non-increasing, we know that p k t ,t ∈ ( p ∗ t -ϵ t , p ∗ t + ϵ t ) similar to the analysis in Case (1), and again we have | r t ( p ∗ t ) -r t ( p k t ,t ) | ≤ 8 C v ( C 2 N · 1 N k t ( t ) + C 2 τ · 1 τ ) . This completes the proof of Lemma 4 on both circumstances.

## B.5 Proof of Lemma 5

Proof. When γ t &gt; ˆ a + C a · 1 √ τ 2 + c &gt; a 2 + c , we know that the demand at p = a 2 b and its neighborhood is not censored. As a result, a 2 b is still a local optimal (and therefore global optimal due to the unimodality) of r t ( p ) . According to Lemma 2, we have:

<!-- formula-not-decoded -->

When ˆ r k,t -∆ k ( t ) &gt; 0 , ∀ k = -M, -M + 1 , . . . , M -1 , M , we know that r ′ t ( a + c -γ t b ) &gt; 0 according to Lemma 3. Since d t ( a + c -γ t b ) = a -b · a + c -γ t b + N t = γ t -c + N t ≤ γ t is not censored,

we know that the optimal price p ∗ t satisfies p ∗ t &gt; a + c -γ t b (since r ′ t ( p ∗ t ) = 0 &lt; r ′ t ( a + c -γ t b ) ) and therefore its demand is not censored. Therefore, the optimal price p ∗ t = a 2 b and we have its regret bounded by Eq. (54) identically. Let C non := 2 C s ( a 2 max C 2 b + b 2 min C 2 a ) 4 b 4 min and we have proven both cases.