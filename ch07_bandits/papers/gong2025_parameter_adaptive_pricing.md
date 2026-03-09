## Parameter-Adaptive Dynamic Pricing

Xueping Gong * and Jiheng Zhang †

* † Department of Industrial Engineering and Decision Analytics

* † The Hong Kong University of Science and Technology

## Abstract

Dynamic pricing is crucial in sectors like e-commerce and transportation, balancing exploration of demand patterns and exploitation of pricing strategies. Existing methods often require precise knowledge of the demand function, e.g., the H¨ older smoothness level and Lipschitz constant, limiting practical utility. This paper introduces an adaptive approach to address these challenges without prior parameter knowledge. By partitioning the demand function's domain and employing a linear bandit structure, we develop an algorithm that manages regret efficiently, enhancing flexibility and practicality. Our Parameter-Adaptive Dynamic Pricing (PADP) algorithm outperforms existing methods, offering improved regret bounds and extensions for contextual information. Numerical experiments validate our approach, demonstrating its superiority in handling unknown demand parameters.

## 1 Introduction

Dynamic pricing, a technique involving the real-time modulation of prices in response to changing market dynamics, has emerged as a crucial strategy in sectors like e-commerce and transportation. A successful dynamic pricing framework must strike a delicate balance between exploration-by learning demand patterns across different price points-and exploitation-by fine-tuning prices based on observed data on pricing and demand. This field has garnered considerable scholarly interest due to its practical implications and its pivotal role in revenue optimization. For an indepth exploration of the existing literature on dynamic pricing, readers are encouraged to consult [16]. Additionally, for valuable insights into specific applications of dynamic pricing strategies, Saharan et al. [35] offer a comprehensive perspective that can further enrich our understanding of this dynamic field.

An inherent challenge prevalent in existing methodologies lies in the requirement for precise knowledge of the H¨ older smoothness level β and the Lipschitz constant L governing the demand function. These assumptions often fail to align with the intricacies of real-world scenarios, thereby limiting the practical utility of these algorithms. In response to this limitation, our research distinguishes itself by venturing into unexplored terrain concerning adaptability within dynamic pricing systems. Specifically, we tackle the critical issue of adaptation when confronted with unknown values of the H¨ older smoothness level β and the Lipschitz constant L , a challenge that has remained largely unaddressed in prior works.

To address this challenge effectively, we introduce a novel adaptive methodology designed to navigate the inherent uncertainties surrounding key parameters. Initially, we partition the domain of the demand function into equidistant intervals, a foundational step in our approach. This discretization not only gives rise to a linear bandit structure but also facilitates the formulation of upper confidence bounds adept at accommodating biases arising from approximation errors. Building upon the unique characteristics of this linear bandit framework, we implement a stratified data partitioning technique to efficiently regulate regret in each iteration. The adaptive nature of this methodology obviates the necessity for a Lipschitz constant, thereby amplifying the flexibility and practicality of our strategy. In handling the smoothness parameter, we introduce an estimation method under a similarity condition, enabling seamless adaptation to unknown levels of smoothness. This method eliminates the need for the smoothness parameter, enhancing the flexibility and practicality of our approach. Lastly, through meticulous consideration, we determine an optimal number of discretization intervals to attain minimax optimality for this intricate problem.

Our contributions in this paper can be succinctly summarized as follows:

- Innovative adaptive algorithm: We introduce the Parameter-Adaptive Dynamic Pricing (PADP) algorithm, employing the layered data partitioning technique to eliminate the dependence on unknown parameters of the demand function. Moreover, our algorithm is designed to adapt to model misspecification, enhancing its robustness for real-world applications.
- Improved regret bound: Our proposed algorithm outperforms existing works such as [8] and [37] by achieving a superior regret upper bound. We address limitations present in the regret bound of [37], enhancing scalability in terms of price bounds and improving the order concerning the smoothness level β .
- Extension: We present additional extensions, including the incorporation of linear contextual information. By comparing with the established linear contextual effect model as detailed in [8], our method excels in enhancing the leading order with respect to contextual dimensions. Additionally, our method can recover the order concerning the Lipschitz constant in [8] when this parameter is provided in advance.
- Solid numerical experiments: Through a series of comprehensive numerical experiments, we underscore the necessity of understanding the parameters of the demand function. In the absence of this knowledge, previous methods either falter or exhibit degraded performance. This empirical evidence underscores the superiority of our method.

## 1.1 Related work

Nonparametric dynamic pricing. Dynamic pricing has been an active area of research, driven by advancements in data technology and the increasing availability of customer information. Initial research focused on non-contextual dynamic pricing without covariates [5, 10]. For example, Wang et al. [37] employed the UCB approach with local-bin approximations, achieving an ˜ O ( T ( m +1) / (2 m +1) ) regret for m -th smooth demand functions and establishing a matching lower bound. Bu et al. [8] extend this model into the one with additive linear contextual effect. Bu et al. [8] devise a different learning algorithm based on the biased linear contextual bandit borrowed from

Wang et al. [37] and their new idea of being more optimistic to chase the context-dependent optimal price. They establish an instance-dependent bounds and a matching lower bound.

Contextual dynamic pricing. The realm of context-based dynamic pricing has been extensively explored in the literature, as evidenced by studies such as [5], [27], [11], and [3]. Within this domain, the binary choice model, exemplified by works like [24], [21], [14], [38], [33], [32], and [18], assumes that every customer's purchase decision follows a Bernoulli distribution. However, our model exhibits a broader scope, accommodating non-binary demand scenarios and encompassing the binary case as a special instance. For a comprehensive overview of recent developments and related works, we direct the reader to [8] and [37]. Compared with these work, our approach reduces the order of smoothness level in the regret upper bound and obviates the necessity of a Lipschitz constant in the algorithmic implementation. Achieving such outcomes necessitates meticulous parameter selection within the algorithm and a sophisticated algorithmic design.

Contextual bandit. The most studied model in contextual bandit is the linear model (see, e.g., [25, 19, 1, 36, 2]), where the expected reward is a linear combination of contexts. The algorithms developed in these works are mostly built upon the celebrated idea of the OFU principle. Specifically, our approach relates to the concept of misspecified linear bandits [2, 36] within the context of dynamic pricing. Unlike traditional bandit algorithms, dynamic pricing requires consideration of both estimation variance and the approximation errors. Notably, we have discovered an intriguing finding that by leveraging the unique structure of the dynamic pricing problem, we can achieve a more precise and improved regret bound compared to directly applying complex existing algorithms designed for misspecified linear bandits. This improvement is mainly attributed to the fact that the bias encountered in each round is consistent across all prices (actions). This underscores the importance of leveraging the distinctive structure and attributes of the pricing context in order to achieve superior performance. There are also a substantial amount of literature considering non-parametric reward feedback under H¨ older continuous assumption (see, e.g., [37, 8]).

Notations. Throughout the paper, we use the following notations. For any positive integer n , we denote the set { 1 , 2 , · · · , n } as [ n ]. The cardinality of a set A is denoted by | A | . We use I { E } to represent the indicator function of the event E . Specifically, I { E } takes the value 1 if E happens, and 0 otherwise. For norms, we utilize the notation ‖ · ‖ p where 1 ≤ p ≤ ∞ to denote the /lscript p norm. Throughout the analysis, the notation ˜ O is employed to hide the dependence on absolute constants and logarithmic terms. It allows us to focus on the dominant behavior of the quantities involved.

## 2 Basic Setting

We consider a stylized dynamic pricing setting consisting of T selling time periods, conveniently denoted as t = 1 , 2 , · · · , T . At each time period t before a potential customer comes, a retailer sets a price p t ∈ [ p min , p max ], where [ p min , p max ] is a predetermined price range. The retailer then observes a randomized demand d t ∈ [0 , 1] and collects a revenue of r t = p t d t . We constrain the realized demand at every single selling period t to be at most one, which can be achieved by considering short arrival time periods such that at most one purchase will occur within each time period.

The randomly realized demands { d t } T t =1 given advertised prices { p t } T t =1 are governed by an underlying demand function f : [ p min , p max ] → [0 , 1] such that at each time t , d t ∈ [0 , 1] almost surely and E [ d t | p = p t ] = f ( p t ). That is, the demand function f ( p ) specifies the expected demand

under posted prices. In this paper, the demand function f is assumed to be unknown prior to the whole selling seasons and has to be learnt on the fly as the selling proceeds. The regret at time t is defined as the loss in reward resulting from setting the price p t compared to the optimal price p ∗ = arg max p ∈ [ p min ,p max ] pf ( p ). The cumulative regret over the horizon of T periods, denoted as Reg ( T ), is given by the expression:

<!-- formula-not-decoded -->

To enable learning of the underlying demand function f , we place the following smoothness condition on f :

To evaluate the performance, we consider the expected cumulative regret E [ Reg ( T )], which takes into account the randomness of the potential randomness of the pricing policy. The goal of our dynamic pricing policy is to decide the price p t at time t , by utilizing all historical data { ( p s , y s ) , s ∈ [ t -1] } , in order to minimize the expected cumulative regret.

Assumption 2.1 (Smoothness Condition) For some β &gt; 1 , and L &gt; 0 , the demand function f : [ p min , p max ] → [0 , 1] is H¨ older-continuous of exponent /pi1 ( k ) and Lipschitz constant L on domain [ p min , p max ] , meaning that f is /pi1 ( k ) -times differentiable on [ p min , p max ] , and furthermore

<!-- formula-not-decoded -->

It is essential to recognize that several widely accepted smoothness conditions in existing literature are included in Assumption 2.1 . For example, functions with β = 1 belong to the class of Lipschitz continuous functions, as outlined in [4]. When β = 2, this encompasses all functions with bounded second-order derivatives, consistent with the assumptions in [29]. For any general integer β , Assumption 2.1 aligns with Assumption 1 in [37].

It is important to note that, unlike various studies that assume concavity (e.g., [12]), we do not make this assumption here. Furthermore, this class of functions depends on the smoothness constant L , which intuitively reflects how closely f can be approximated by a polynomial. Since f is /pi1 ( k )-times differentiable on the bounded interval [ p min , p max ], we can easily demonstrate that

<!-- formula-not-decoded -->

Compared with [8] which relies on the knowledge of the upper bounds of derivatives, our algorithms can be more practical and eliminate the need of this prior knowledge.

## 3 Parameter Adaptive Dynamic Pricing

In this section, we present an adaptive policy designed in the presence of unknown L . We also derive minimax regret upper bounds for our proposed algorithm as well as additional special cases (very smooth demands) with further tailored regret bounds.

## 3.1 Algorithm design

In this subsection, our method is presented in Algorithm 1. The high-level idea of Algorithm 1 is the following. First, the algorithm partitions the price range [ p min , p max ] into N intervals. We regard each interval as an action in bandit and uses the UCB technique to control the exploration and exploitation. The UCB construction is inspired by linear contextual bandits, using local polynomial approximates of the demand function f .

Initialization. In each time period t , we begin by discrete the whole price range [ p min , p max ] into N sub-intervals with equal length, i.e., [ p min , p max ] = ⋃ N j =1 [ a j -1 , a j ], where a j = p min + j ( p max -p min ) N for j = 0 , 1 , · · · , N . We then can naturally construct the candidate action set A t, 1 = [ N ]. Such construction leads to linear bandit structure with bounded action set size.

Local polynomial regression. For each j ∈ [ N ], let θ j be a ( /pi1 ( β ) + 1)-dimensional column vector with its k -th element defined as f ( k -1) ( a j ) ( k -1)! . According to Lemma B.1, the demand function can be expressed as

<!-- formula-not-decoded -->

where φ j ( p ) = (1 , p -a j , · · · , ( p -a j ) /pi1 ( β ) ) /latticetop and | η j | ≤ L ( p max -p min ) β N β . This formulation allows us to focus on the estimator for each θ j .

By using the local polynomial regression, we can leverage the H¨ older smoothness condition to improve the approximation error at each small price interval. Suppose the length of a price interval is η . Under a constant approximation, the approximation error would be O ( η ) if the demand function f is Lipschitz. Conversely, if f satisfies Assumption 2.1 , we can bound the approximation error by O ( η β ) for β &gt; 1, thereby improving the original O ( η ) error.

After the initialization step, each dataset D j t,s consists of at least 1 + /pi1 ( β ) samples. Thanks to the polynomial map φ j , we can show that the first 1 + /pi1 ( β ) samples form a Vandermonde matrix, which implies Λ j t,s to be invertible.

Proposition 3.1 For each t &gt; N (1 + /pi1 ( β )) , s ∈ [ S ] and j ∈ [ N ] , the Gram matrix Λ j t,s is invertible.

From Proposition 3.1, we know that the estimator θ j t,s is unique. Based on the given estimators, we can construct the upper confidence bound for the optimal prices in each interval. Some studies [1, 15, 8] concerning linear contextual bandits consider the following form:

<!-- formula-not-decoded -->

However, this form is not applicable in our context, as it would cause the estimation error to depend on the upper bound of the linear parameter norm. In our scenario, it necessitates knowledge of the upper bounds of the derivatives of f , which can be unrealistic in practical applications.

Layered Data Partitioning. During the round t , we utilize the local polynomial regression estimator θ j t,s to drive a UCB procedure that strikes a balance between f -learning and revenue maximization. To explore potential optimal prices p ∈ [ p min , p max ] efficiently, we focus our attention on the f -values within the range of [ p min , p max ].

Instead of utilizing all the data collected prior to time t for an ordinary least squares estimator

## Algorithm 1 Parameter Adaptive Dynamic Pricing

```
Input: the discretization number N , the length T , the smoothness parameter β , the confidence parameter δ and the price bound [ p min , p max ] 1: Let a j = p min + j ( p max -p min ) N for j = 0 , 1 , · · · , N 2: Let φ j ( p ) = (1 , p -a j , · · · , ( p -a j ) /pi1 ( β ) ) /latticetop 3: Set S = /ceilingleft log 2 √ T /ceilingright , γ = » 1 2 ln(2 NST/δ ), Ψ s t = ∅ for s ∈ [ S ] // Initialization 4: for round t = 1 , 2 , · · · , N (1 + /pi1 ( β )) do 5: Let j = t mod N 6: Set a distinct price p t in [ a j -1 , a j ] and obtain the demand d t 7: Update the round collection Ψ σ t +1 = Ψ σ t ∪ { t } for all σ ∈ [ S ] 8: for round t = N (1 + /pi1 ( β )) + 1 , N (1 + /pi1 ( β )) + 2 , · · · , T do 9: Let s = 1 and A t, 1 = { j ∈ [ N ] | p τ ∈ [ a j -1 , a j ] , τ ∈ [ t -1] } 10: repeat 11: Compute D j t,s = { τ ∈ Ψ s t | p τ ∈ [ a j -1 , a j ] } for all j ∈ A t,s 12: For all j ∈ A t,s , compute the parameter θ j t,s = arg min θ ∑ τ ∈D j t,s ( d τ -θ /latticetop φ j ( p τ )) 2 13: For all j ∈ A t,s , compute the Gram matrix Λ j t,s = ∑ τ ∈D j t,s φ j ( p τ ) φ j ( p τ ) /latticetop 14: Compute the upper confidence bound U j t,s ( p ) = φ j ( p ) /latticetop θ j t,s + γ » φ j ( p ) /latticetop (Λ j t,s ) -1 φ j ( p ) 15: if sup p ∈ [ a j -1 ,a j ] p » φ j ( p ) /latticetop (Λ j t,s ) -1 φ j ( p ) ≤ p max / √ T for all j ∈ A t,s then 16: Choose the price p t = arg max j ∈A t,s sup p ∈ [ a j -1 ,a j ] pU j t,s ( p ) 17: Update the round collection Ψ σ t +1 = Ψ σ t for all σ ∈ [ S ] // step (a) 18: else if sup p ∈ [ a j -1 ,a j ] p » φ j ( p ) /latticetop (Λ j t,s ) -1 φ j ( p ) ≤ p max 2 -s for all j ∈ A t,s then 19: Let A t,s +1 = { j ∈ A t,s | sup p ∈ [ a j -1 ,a j ] pU j t,s ( p ) ≥ max j ′ ∈A t,s sup p ∈ [ a j -1 ,a j ] pU j ′ t,s ( p ) -p max 2 1 -s } 20: Let s ← s +1 // step (b) 21: else 22: Choose j t such that sup p ∈ [ a j t -1 ,a j t ] pγ » φ j ( p ) /latticetop (Λ j t,s ) -1 φ j ( p ) > p max 2 -s 23: Choose the price p t = arg max p ∈ [ a j t -1 ,a j t ] pγ » φ j ( p ) /latticetop (Λ j t,s ) -1 φ j ( p ) 24: Update Ψ s t +1 = Ψ s t ∪ { t } 25: until A price p t is found 26: Set the price p t and obtain the demand d t
```

for θ j , we divide the preceding time periods into disjoint layers indexed by s ∈ [ S ], such that ∪ s ∈ [ S ] Ψ s t = [ t -1]. At time t , we sequentially visit the layers s ∈ [ S ] and, within each layer, calculate the upper confidence bound for each θ j using data collected exclusively during the periods in Ψ s t .

The algorithm commences with s = 1, corresponding to the widest confidence bands, and incrementally increases s , while simultaneously eliminating sub-optimal prices using OLS estimates from Ψ s t . This procedure is meticulously designed to ensure that the 'stopping layer' s t determined at time t solely depends on { Ψ s t } s ≤ s t , effectively decoupling the statistical correlation in OLS estimates. The validity of this property has been formally stated and proven in [30, 2].

The UCB property described in Lemma 3.1 is highly intuitive. The first term captures the estimation, while the second term represents the bias stemming from the approximation error. As the learning progresses, the variance gradually diminishes. With an increasing number of samples of p t and d t , our constructed θ j t,s approach to θ j with the distance at most L ( p max -p min ) β √ (1+ /pi1 ( β )) ln t N β . It is important to note that our constructed UCB does not involve the bias term. This allows us to eliminate the need for knowledge of the Lipschitz constant. The reason behind this is that we compare the UCB values on both sides, and the largest bias terms are the same for all indices. Therefore, we can deduct the bias term at both sides in the step (a) of the algorithm and the Lipschitz term is implicitly absorbed in the adaptive term p max 2 1 -s . This advantage of our algorithm eliminates the need for explicit knowledge of the Lipschitz constant.

Lemma 3.1 Consider the round t and any layer s ∈ [ s t ] in the round t . Then for each j ∈ [ N ] , we have

<!-- formula-not-decoded -->

with probability at least 1 -δ NST .

The proof of Lemma 3.1 is also intuitive. Using polynomial regression, we implicitly construct a linear demand model given by ˜ d = φ j ( p ) /latticetop θ j + /epsilon1 . In each round, the linear demand model and the true model share the same price and noise, but they differ only in the demand. We then use the least squares estimator ˜ θ j t,s as a bridge to establish an upper bound between θ j and θ j t,s . Since ˜ θ j t,s is an unbiased estimator of θ j , the first term in the RHS of Lemma 3.1 arises from the estimation error. Because ˜ θ j t,s shares the same structure as θ j t,s , we can apply Lemma B.1 to bound the distance between these two estimators. Combining the two bounds, we obtain the results mentioned above.

Based on Lemma 3.1, we construct a high-probability event

<!-- formula-not-decoded -->

Then by the union bound, we have

<!-- formula-not-decoded -->

## 3.2 Regret analysis

In this subsection, we analyze the regret of our proposed Algorithm 1. We will show how to configure the hyperparameter N .

For simplicity, we introduce the revenue function

<!-- formula-not-decoded -->

We define the index of the best price within the interval [ a j -1 , a j ] at the candidate set A t,s as

<!-- formula-not-decoded -->

We also denote one of the best price as p ∗ ∈ arg max p ∈ [ p min ,p max ] rev ( p ).

One special case is the polynomial demand function. In this case, the approximation error is exactly zero and our algorithm ensures rev ( p ∗ ) = rev ( p j ∗ t,s t,s ) for each s , as the optimal action will not be eliminated when their constructed high-probability event holds. For general cases, due to the approximation error being proportional to L ( p max -p min ) β √ 2(1+ /pi1 ( β )) ln t N β when transitioning to a new layer, the regret of setting the price indexed by j ∗ t,s is naturally proportional to s -1. Combining the above discussion, we proceed to prove the following lemma.

Lemma 3.2 Given the event Γ , then for each round t and each layer s ∈ [ s t -1] , we have

<!-- formula-not-decoded -->

Now, we can effectively control the regret within layer s using j ∗ t,s as the benchmark. The upper bound in Lemma 3.3 can be divided into two components: the variance and the bias. As we increase the value of s , the variance decreases exponentially, while the bias only increases linearly. Therefore, having a larger value of s is advantageous for online learning.

Lemma 3.3 Given the event Γ , then for each round t , we have

<!-- formula-not-decoded -->

for all 2 ≤ s &lt; s t .

Based on Lemma 3.2 and Lemma 3.3, it is straightforward to derive the upper bound of the discrete-part regrets for prices in the layer s .

Lemma 3.4 Given the event Γ , for every round t , we have

<!-- formula-not-decoded -->

Therefore, the remaining tasks is to count the rounds in each D j t,s . From Lemma A.2, we know

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Based on the previous given materials, we are ready to prove the following theorem.

Theorem 3.1 Under Assumption 2.1 , the expected regret of Algorithm 1 with N = /ceilingleft ( p max -p min ) 2 β 2 β +1 T 1 2 β +1 /ceilingright satisfies

<!-- formula-not-decoded -->

with probability at least 1 -δ .

Lower bound. At a higher level, Theorem 2 in [37] shows that for a class of nontrivial problem instances, the optimal cumulative regret any admissible dynamic pricing policy could achieve is lower bounded by Ω( T β +1 2 β +1 ) if the underlying demand function f is assumed to satisfy Assumption 2.1 . This result shows that the regret upper bounds established in Theorem 3.1 is optimal in terms of dependency on T (up to logarithmic factors) and therefore cannot be improved.

We now discuss some special cases of β .

Very Smooth Demand. An interesting special case of Theorem 3.1 is when β = + ∞ , implying that the underlying demand function f is very smooth. In such a special case, it is recommended to select β as β = ln T , and the regret upper bound in Theorem 3.1 could be simplified to

<!-- formula-not-decoded -->

Strongly concave rewards. In various academic studies, researchers have examined the implications of strongly concave rewards. This concept implies that the demand function adheres to specific smoothness conditions, notably characterized by Assumption 2.1 with β = 2. Moreover, the second derivatives of the reward function pf ( p ) are negative.

The imposition of strong concavity on the reward function has been a prevalent assumption in previous works, such as those by [17] and [13]. This condition results in a uni-modal revenue function, which significantly simplifies the learning process. In contrast, our method is specifically designed to handle multi-modal demand functions, which are more representative of real-world applications.

From these observations, we can establish the following assumption, which has been discussed in numerous papers, including [37] and [17]. We will present this assumption directly to avoid unnecessary repetition in our discussion.

Assumption 3.1 The revenue function rev ( p ) = pf ( p ) satisfies

<!-- formula-not-decoded -->

for some positive constants c 1 and c 2 .

To achieve the optimal regret order, we can implement Algorithm 1 with β = 2 and N = Θ( T 1 4 ).

By modifying the proof of Lemma 3.4, we can establish that

<!-- formula-not-decoded -->

for any p ∈ [ a j -1 , a j ] with j ∈ A t,s . Next, we substitute the endpoints into Assumption 3.1 to derive that

<!-- formula-not-decoded -->

for j t,s ∈ A t,s - { j ∗ t,s } . It implies that | j t,s -j ∗ | ≤ O (log log T ), which further suggests that A t,s = O (log log T ).

Therefore, during the proof of Theorem 3.1, we have

<!-- formula-not-decoded -->

From the preceding discussion, we recognize that Assumption 3.1 is a strong condition that significantly reduces regret. Demand functions that satisfy this assumption exhibit performance characteristics similar to those of very smooth functions.

due to Lemma A.2. The regret due to approximation error can be bounded by O ( T/N 2 ) = O ( √ T ) . By combining these findings, we conclude that our method yields ˜ O ( √ T ) regret.

A notable case discussed in [17] examines the demand function 1 -F ( p ) with Bernoulli feedback. Their method incurs ˜ O ( √ T ) regret, which matches the order of our results. However, our approach demonstrates advantages by generalizing to accommodate a broader class of demand functions.

Comparison to existing works. If β ∈ N , a comparison can be made between our findings and those of [37]:

<!-- formula-not-decoded -->

Firstly, this bound requires p max -p min ≤ 1. Otherwise, the upper bound will exponentially increase when β goes to infty, i.e., the demand function is supper smooth. Secondly, we have enhanced the order of β from ˜ O ( β ( p max -p min ) β T β +1 2 β +1 ) to ˜ O ( √ β ( p max -p min ) β 2 β +1 T β +1 2 β +1 ). This improvement also eliminates the constraint of p max -p min ≤ 1. Additionally, our method does not require the knowledge L , compared with [37].

## 4 Linear Contextual Effect

In this section, we study the semi-parametric demand model with linear contextual effect, i.e., rev ( p t , x t ) = f ( p t ) + µ /latticetop x t . In this setting, x represents the context, and µ is a context-dependent parameter. Such model is also investigated by many scholars such as Bu et al. [8, 7]. We now introduce some assumptions on this model.

Assumption 4.1 1. The context x t is drawn i.i.d. from some unknown distribution with support in the α -dimensional unit ball.

2. The Gram matrix Σ = E [ xx /latticetop ] is invertible, i.e., λ min (Σ) &gt; 0 .

The first assumption is standard in the dynamic pricing literature, see, e.g., [3, 39]. The unit ball can be generalized into any compact sets. The second assumption can be found in linear demand models [8, 3] and it also applies to linear bandit setting [26, 22, 34].

To incorporate contextual information into our framework, we introduce an extended feature map:

<!-- formula-not-decoded -->

where φ ( p ) is the original polynomial map for the price p . The remaining steps in our approach are largely unchanged.

When context is not present, the optimal price remains fixed. In the absence of context, Wang et al. [37] propose the idea of embedding the dynamic pricing into a multi-armed bandit framework, treating each price segment as a distinct arm. The segment to which the fixed optimal price belongs is regarded as the 'best' arm. In contrast, in our context-based model, the optimal price varies depending on the context revealed in each period and thus changes over time. As a result, the 'best' arm shifts with each period, making the static multi-armed bandit approach described by Algorithm 1 inapplicable in the new setting. To address this challenge, we modify Algorithm 1 and present the new algorithm in Algorithm 2.

## Algorithm 2 Parameter Adaptive Contextual Dynamic Pricing

Require: the discretization number N , the length T , the smoothness parameter β , the confidence parameter δ and the price bound [ p min , p max ]

- 1: Let a j = p min + j ( p max -p min ) N for j = 0 , 1 , · · · , N
- 3: Set S = /ceilingleft log 2 √ T /ceilingright , γ = » 1 2 ln(2 NST/δ ), Ψ s t = ∅ for s ∈ [ S ], T = 0 4: for j = 1 , · · · , N do 5: Let Λ j = O and D j = ∅
- 2: Let ϕ j ( p, x ) = (1 , p -a j , · · · , ( p -a j ) /pi1 ( β ) , x /latticetop ) /latticetop ∈ R α + /pi1 ( β )+1
- 6: while Λ j is not invertible and T &lt; T do
- 8: Set a distinct price p T in [ a j -1 , a j ] and obtain the demand d T
- 7: Observe the context x T ∈ R α
- 9: Update the round collection Ψ σ T +1 = Ψ σ T ∪{T } for all σ ∈ [ S ]
- 11: Compute the Gram matrix
- 10: Update the dataset D j = D j ∪{T }

<!-- formula-not-decoded -->

- 12: Update the round collection Ψ σ t +1 = Ψ σ t for all σ ∈ [ S ]

// (continued)

<!-- formula-not-decoded -->

Unlike the initialization step in Algorithm 1, we use a stopping condition to guarantee the invertiblity of required Gram matrices. The following lemma indicates the expected stopping time T is of order O ( N log T ).

Lemma 4.1 In Algorithm 2, the following holds

<!-- formula-not-decoded -->

## Algorithm 3 Parameter Adaptive Contextual Dynamic Pricing (continued)

//

Continued from Algorithm 2

- 1: for round t = T +1 , 2 , · · · , T do

<!-- formula-not-decoded -->

- 2: Observe the context x t ∈ R α

4:

repeat

<!-- formula-not-decoded -->

- 6: For all j ∈ A t,s , compute the estimator

<!-- formula-not-decoded -->

- 7: For all j ∈ A t,s , compute the matrix

<!-- formula-not-decoded -->

- 8: Compute the upper confidence bound

<!-- formula-not-decoded -->

for some positive constant C 0 . The constant C 0 depends on α, β and λ min (Σ) .

Similar to Theorem 3.1, we can bound the regret of Algorithm 2, which is presented in Theorem 4.1.

Theorem 4.1 Under Assumption 2.1 and Assumption 4.1 , the expected regret of Algorithm 2 with N = /ceilingleft ( p max -p min ) 2 β 2 β +1 T 1 2 β +1 /ceilingright satisfies

<!-- formula-not-decoded -->

with probability at least 1 -δ .

The proof of Theorem 4.1 is almost identical to that of Theorem 3.1. The difference is to replace the feature map φ with the new one ϕ . Another difference lis in the initialization step. Since the contexts are i.i.d. generated, we use the stopping time analysis to guarantee the invertiblity. For completeness, we defer the proofs of Theorem 4.1 in Appendix.

Comparison to existing works. This improvement also shows advantageous in the linear contextual effect setting. As shown in [8], the regret is upper bounded by

<!-- formula-not-decoded -->

We have elevated the order of α + /pi1 ( β ) + 1 to √ α + /pi1 ( β ) + 1. Regarding the parameter L , the methodology in [8] mandates prior knowledge of L . By configuring the parameter:

<!-- formula-not-decoded -->

with knowledge of L , we can still attain the optimal order concerning L . However, in cases where L is unknown, our approach can adapt to this parameter, in contrast to the method in [8]. Besides, the method in [8] is not rate-optimal in terms of the dimension α .

Additionally, in the linear demand model d ( p, x ) = bp + µ /latticetop x , the upper bound transforms to ˜ O ( √ αT ), outperforming the optimal regret rate ˜ O ( α √ T ) in [3] in terms of the context dimension α . Xu and Wang [39] proves that the lower bound of this problem is Ω( √ αT ), which indicates that our method can achieve the minimax optimality in this simple demand function.

## 4.1 Adaptivity of smoothness parameter

When the smoothness parameter β is unknown, non-adaptive dynamic pricing algorithms cannot reach the optimal regret rate. It has been shown in [31] that no strategy can adapt optimally to the smoothness level of f for cumulative regret. However, adaptive strategies that are minimax optimal can exist if additional information about f is available. The self-similarity condition [20, 40, 9, 23] is often used to achieve adaptivity with beneficial properties.

To formalize this concept, we first define notation for function approximation over polynomial spaces. For any positive integer l , Poly ( l ) represents the set of all polynomials with degree up to l . For any function g ( · ), we use Γ U l g ( · ) to denote its L 2 -projection onto Poly ( l ) over some interval U ,

which can be obtained by solving the minimization problem:

<!-- formula-not-decoded -->

Definition 4.1 A function g : [ a, b ] → R is self-similar on [ a, b ] with parameters β &gt; 1 , /lscript ∈ Z + , M 1 ≥ 0 and M 2 &gt; 0 if for some positive integer c &gt; M 1 it holds that

<!-- formula-not-decoded -->

where we define

<!-- formula-not-decoded -->

for any positive integer c .

Unlike H¨ older smoothness, the self-similarity condition sets a global lower bound on approximation error using polynomial regression. This approach helps estimate the smoothness of demand functions by examining approximations across different scales. It has been used in nonparametric regression to create adaptive confidence intervals. In previous work, this condition was applied to estimate non-contextual demand functions [40], and for general contextual demand functions [9]. However, because our contextual terms are additive, we can reduce sample usage and achieve minimax optimality.

## Algorithm 4 Smoothness Parameter Estimation

Input: the length T , the smoothness parameter upper bound β max , the price bound [ p min , p max ]

- 1: Set local polynomial regression degree /pi1 ( β max )
- 3: for i = 1 , 2 do
- 2: Set k 1 = 1 2 β max +2 , k 2 = 1 4 β max +2 , K 1 = 2 /floorleft k 1 log 2 T /floorright , K 2 = 2 /floorleft k 2 log 2 T /floorright
- 4: Set trial time T i = T /floorleft 1 2 + k i /floorright
- 5: Set the price T i times from U [ p min , p max ] independently
- 6: Collect samples ( x j , p j , d j ) with size T i
- 7: for m = 1 , 2 , · · · , K i do
- 9: Construct the estimate ˆ f i ( p ) on [ p min + ( m -1)( p max -p min ) K i , p min + m ( p max -p min ) K i ] and corresponding ˆ µ i from the regression
- 8: Fit local polynomial regression on [ p min + ( m -1)( p max -p min ) K i , p min + m ( p max -p min ) K i ] with samples falling in the interval

<!-- formula-not-decoded -->

Our algorithm is based on a clear concept. We start by dividing the price range into small intervals, with each interval's length determined by β max . This enables us to use local polynomial regression to approximate the true demand function within each segment. As shown in Corollary 4.1 , our non-adaptive algorithm reaches optimal regret when the correct smoothness parameter is applied.

The rationale behind Algorithm 4 hinges on leveraging the H¨ older smoothness property of the demand function, which justifies the use of local polynomial regression for estimation. Nevertheless,

the self-similarity intrinsic to H¨ older smoothness functions complicates polynomial-based approximations, as it imposes recursive regularity constraints across scales. To mitigate this, the algorithm refines its estimation granularity-i.e., the number of subintervals partitioning the price domain-to enhance the resolution of the demand function's approximation. By iteratively adjusting the partition density, the method achieves a high-fidelity piecewise polynomial representation, balancing local accuracy with global structural constraints.

Lemma 4.2 Let O [ a,b ] = { ( x i , p i , d i ) } n i =1 be an i.i.d. sample set, where each p i ∈ [ a, b ] and ‖ x i ‖ ≤ 1 . Assume the sub-Gaussian parameter u 1 ≤ exp( u ′ 1 n v ) for some positive constants v and u ′ 1 , and that the polynomial degree /lscript ≥ α + /pi1 ( β ) . Then, there exist positive constants C 1 and C 2 such that, with probability at least 1 - O ( /lscripte -C 2 ln 2 n ) , for any p ∈ [ a, b ] , any ‖ x ‖ 2 ≤ 1 and n &gt; C 1 , the following inequality holds:

<!-- formula-not-decoded -->

where ˆ f and ˆ µ are estimated from the sample O [ a,b ] .

Lemma 4.2 shows that the estimation error in the linear contextual effect setting. If we apply the method from [9], the approximation error is O (( b -a ) α + β ). However, due to the additive structure of the demand function, we can avoid the extra α term in the error bound.

Theorem 4.2 With an upper bound β max of the smoothness parameter, under the assumptions in our setting, for some constant C &gt; 0 , with probability at least 1 -O ( e -C ln 2 T ) ,

<!-- formula-not-decoded -->

Theorem 4.2 highlights the effectiveness of our proposed Algorithm 4 in estimating the H¨ older smoothness parameter β . The adaptability and efficient smoothness parameter estimation enable it to construct tight confidence intervals for β and achieve a high convergence rate. These features contribute to the desired regret bound in dynamic pricing scenarios and open avenues for developing more robust and adaptive pricing algorithms.

The estimators of the demand function, denoted ˆ f 1 and ˆ f 2 , allow us to determine an estimate for the H¨ older smoothness parameter ˆ β . The estimator ˆ β is then fed into Algorithm 2 for the remaining time horizon T -T 1 -T 2 . We notice that for demand function f satisfying Assumption 2.1 with the parameter β , it also satisfies Assumption 2.1 with the parameter ˆ β ≤ β . Therefore, if Algorithm 1 is invoked with the estimator ˆ β ≤ β , then the regret is upper bounded by

<!-- formula-not-decoded -->

From Lemma B.3, we know that ˆ β converges to β with rate O ( ln ln T ln T ). We know that

<!-- formula-not-decoded -->

Therefore, we know that the regret caused by estimating β is at most

<!-- formula-not-decoded -->

For the adaptive part, note that T 1 , T 2 ≤ T β +1 2 β +1 , which means that the regret is upper bounded by O ( T β +1 2 β +1 ). We combing the above discussion we obtain the following corollary.

Corollary 4.1 If we run Algorithm 4 to obtain an estimator ˆ β and feed it to Algorithm 1, then the regret is

<!-- formula-not-decoded -->

Compared with Theorem 3.1, we pay extra (ln T ) 8( β max +1) term to obtain the estimator of β .

## 4.2 Application to pricing competitions

We now demonstrate how our method can be applied to the pricing competition models proposed in [28, 6]. These models consider a market with K sellers, each offering a single product with unlimited inventory, over a selling horizon of T rounds. At the beginning of each round, every seller simultaneously sets their price and then observes their private demand, which is influenced by the prices set by all other sellers. The demand follows a noisy and unknown linear model. Specifically, the demand faced by seller i is given by:

/negationslash

<!-- formula-not-decoded -->

where d i , p i , /epsilon1 i are the demand, price and sub-gaussian noise for the seller i , respectively. The competitors parameters µ ij capture the effect of the price set by competitor j on the demand faced by seller i . In [28], an exact solution is proposed for this linear demand scenario, with a policy that achieves a regret bound of O ( K √ T log T ) regret for each seller.

The pricing competition model above relies on several key assumptions that may not hold in real-world markets. First, it assumes that each seller can observe the prices of all other competitors, which is often unrealistic. A more reasonable assumption would be to allow each seller to observe only a small subset of the prices set by other sellers. Second, in highly competitive markets, the prices of many sellers tend to stabilize over time, forming an invariant distribution. Therefore, it is unrealistic to expect all competitors to engage in an exploration phase, especially when a new seller enters the market. Finally, the regret of Algorithm 2 for each seller scales linearly with the number of competitors, K , which becomes problematic when K is large. In practice, markets often consist of a very large number of sellers, making this result less satisfactory.

To address the limitations of the previous model, we consider the asymptotic regime, where the number of sellers K approaches infinity. In this setting, we aim to provide an approximate solution for cases where K is sufficiently large. We make the following assumptions:

Assumption 4.2 · The market is highly competitive and the market price distribution follows F .

- A single seller's price changes do not affect the overall price distribution F .

Now, consider the approximation of the demand function in equation (1). This leads to the following simplified model:

<!-- formula-not-decoded -->

where p o ∈ R α is a vector of prices observed by the seller, each component sampled from the joint market price distribution F . In this model, we allow the seller to observe α prices from the market, which can be interpreted as an α -dimensional context. Without loss of generality, we assume the components of p o are sorted, as this allows us to focus on the main effect of competition - namely, the values of the prices themselves - while ignoring the finer details of individual seller interactions. In this framework, we abstract away from the specific interactions between sellers and instead focus on the relationship between the seller and the overall market price distribution.

By making slight modifications to the model in [28], we arrive at a refined approximation. This model can be solved using the approach outlined in Algorithm 2. In the original model from [28], the demand function is linear in the proposed price, allowing us to apply the technique for very smooth demands and achieve a regret bound of O ( √ αT log T ) regret. Importantly, this regret is independent of the number of sellers, K . However, our approach extends to a broader range of demand functions, making it more flexible and applicable to a wider variety of market scenarios.

## 5 Discussion

In this section, we explore several extensions and improvements to the existing framework, aiming to enhance our understanding of the theoretical results and their applications.

Many products and customers. We consider K types of cases, such as different customer or product categories. The seller observes K distinct contexts to differentiate between these cases. Each case has its own demand function f i ( p ), and the seller's goal is to maximize the corresponding revenue pf i ( p ) when case i occurs. Let T i denote the number of rounds when the case i occurs. A straightforward maximization problem leads to the following inequality:

<!-- formula-not-decoded -->

Furthermore, the environment can equally divide the whole rounds into K equal length. Then from the lower bound of nonparametric dynamic pricing [37], we deduce a lower bound for K cases

<!-- formula-not-decoded -->

as the lower bound in the single case is Ω( T β +1 2 β +1 ) Notably, under the assumption of concavity (as specified in Assumption 3.1 ), we achieve a regret bound of ˜ O ( √ KT ). In [17], the authors utilize the demand function 1 -F i ( p ) for each product-customer pair. It is important to note that in [17], the total rounds can be viewed as LT , where L denotes the maximum load parameter in their analysis. If we let K represent the number of such product-customer pairs, we can recover the same order of regret as shown in their work.

Model misspecification. Our algorithms are robust to model misspecification. Suppose that the noise term has non-zero mean, and | /epsilon1 t | ≤ ε . If we apply the methods in [8, 37], we will require the knowledge of ε . However, our method is parameter-free, meaning that we do not require ε .

Lemma 5.1 Consider the round t and any layer s ∈ [ s t ] in the round t . Then for each j ∈ [ N ] , we have

<!-- formula-not-decoded -->

with probability at least 1 -δ NST .

The proof steps are similar to the well-specified cases, so we directly write down the following corollary. If the misspecification error is small, i.e., ε = O ( T -β 2 β +1 ), thus our method can still achieve the minimax optimality, indicating the robustness of our algorithm.

Corollary 5.1 If the demand function is misspecified and | /epsilon1 t | ≤ ε for all t , then the regret of Algorithm 1 is upper bounded by

O ((1 + L ) p max ( p max -p min ) β 2 β +1 ( α + /pi1 ( β ) + 1) 1 2 (log 2 T ) 4 β +3 4 β +2 T β +1 2 β +1 + /epsilon1T » ( α + /pi1 ( β ) + 1) log T ) with probability at least 1 -δ .

Unknown time horizon. We show that our algorithms can adapt to unknown time horizon T by doubling trick. Let the algorithms proceed in epochs with exponentially increasing in rounds. For n -th epoch, we run our algorithms with rounds 2 n and configure corresponding hyperparameter N with such epoch length. Then we can derive the following upper bound for Algorithm 1. From Holder's inequality, we have

<!-- formula-not-decoded -->

Therefore, we only need to pay extra (log T ) β 2 β +1 terms to make our algorithms to adapt to unknown horizon T .

## 6 Numerical Experiments

In this section, we conduct numerical experiments to study the empirical performance of Algorithm 1. We measure the performance of a learning algorithm by the simple regret defined as Reg ( T ) T . A similar metric is also utilized in [37] and [8].

In the first numerical experiment, we consider the following demand function

<!-- formula-not-decoded -->

with added noise N (0 , 0 . 01). We set the price range [ p min , p max ] = [2 . 6 , 3 . 8] and choose β = 2 . 5. The tight Lipschitz constant for this setting is L ∗ = max p ∈ [ p min ,p max ] | f ′ ( p ) | ≈ 2 . 66.

To demonstrate the advantages of our method, we compare it with the algorithm from [37], tested with L ∈ { L ∗ , 4 , 6 , 8 , 10 } . Both algorithms are evaluated over horizons

<!-- formula-not-decoded -->

for 50 runs.

From Figure 1, it is evident that larger Lipschitz constants lead to higher regret for the method in [37]. This underscores the importance of accurately knowing L , which can be unrealistic in real-world applications. In contrast, our method shows superior performance and practicality, as evidenced by its relative regret curve, outperforming the baseline algorithms in both effectiveness and ease of use.

![Image](data:image/png;base64,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)

·

10

Fig. 1: Relative regret curve with H¨ older smooth demand.

In the second numerical experiment, we consider the following demand function

<!-- formula-not-decoded -->

where Φ denotes the cumulative distribution function of the standard normal distribution. This demand function guarantees the unimodality of the revenue function. Additionally, because this demand function is significantly smoother compared to the previous one, we set β = ln T in both algorithms.

We observe that the relative regrets of both algorithms decrease as the smoothness level increases. Despite this, our method consistently outperforms the method in [37], demonstrating superior performance even without prior knowledge of the smoothness parameter.

## 7 Conclusion

In conclusion, we have introduced a novel algorithm that operates without prior knowledge of the parameters in the unknown demand function. Beyond its practical applicability, our method also achieves an improved regret upper bound. We have explored extensions of our approach, including

10

·

![Image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA4AAAAFBCAIAAACpfVY4AAC8HUlEQVR4nOzdB3iT1dcA8PNmJ03SNklXuvfem1VKobRAy56CqKCIojIEJyAILlRUBBVFBZG9V8vee5fSlu5duneTZn7Pbfz3Q5aM7p7f8z4+TZq+700N6cm5955DabVaQAghhBBCqK3Q2uxKCCGEEEIIYQCK2oJSqayurlYqlU/yYJVKVV1drVAoWn9cCCGEEGofjHa6LuoutFrtb7/9lpGR8cYbb9jZ2T34XYqi7r2npKTkxx9/FIvFs2bNotH++wNSUVFRY2Mjg8GQyWRisVgkErX0M+guKisrq6qqWCyWTCbT09MzMzNr7xF1VviabCn4mmxZeXl5Go3G2tr63nuqq6utrKyEQmHLXqu0tLSoqMjMzMzIyKhlz4y6DAxAUetSN5kyZYqdnV11dfWZM2fkcnleXl5YWJivr29eXt6SJUtMTEyYTCYA6OnpjRkz5s0339y8ebNSqWSz2f95/pKSkp9++qmsrGzs2LGBgYHP88e+oaEhJyfn0KFDY8eONTU1hW6mpqbmr7/+OnXq1AsvvODv7/88f+zT0tJu375dV1dXW1s7evRoiUQC3UkLvibXrVuXkpJiZWVVVVUVHh4eHBwM3UkLvibr6up2796t+3rQoEGGhobQ/Xz33Xdqtfq7775rvicjI+Orr76aPXt2//79n+2cZ86c2bJly8cff2xsbKzVahUKhe59u7S09LPPPgsLC5s+ffozD1ir1TY0NNy5c+fMmTOvvPIKn89/5lOhDgin4FGrozUBgO+//z4lJSUqKkosFr/xxhslJSX5+fl3794tKCjIzc29devWhQsXuFwuRVFPkvvU8fT0LCkpcXR0HDlypI2NzTMPUqPRnD59ev/+/Zs2baqrq4Pux9raWqvVcjicCRMmeHl5PfN5ysvLFy9ebGhoOGzYsMuXL3/yySdqtRq6k5Z6TWq12vT09FOnTh04cMDQ0NDDwwO6mZZ6TSoUioULF5aWlg4dOpTH43322WeNjY3QzZSUlGRlZZ05c6awsLD5zh49epiZmdXX1z/zafX09ExMTBgMkswqKSlpjvLd3NxcXV1ra2ufZ8wKheL48eO7d+/evn37Ey7iQp0IZkBRW9BoNAAgkUjUarVuKq28vLy+vp7FYv3yyy+6xEZ8fLyZmZlEIsnJyXnyM5eWlubl5b388stPHrM+FI1GGzhwoIODQ1xcHHRLKpXqxo0boaGhT5J4fgw6nS6RSJhMJp/P19fXz83N1Wg0dDoduo2Wek2qVCpvb+/p06d326nnlnpNZmZmHj169JdffuHz+f7+/itWrEhJSfH29obuJC0tbdCgQT/++OOVK1diY2N1d1IUxWAwmtdBqdVquVyup6fX2NioC/0BQCaTaTQaPT093f8RlUrFZrPVarUus+Dp6enr66tb63/u3LnU1FSZTMZkMhn/o2zC4/F051cqlSwWi6KohoYGLpdLo9HufcB92Gz2kCFDRCLRhQsX2va3hdoCBqCo7bzxxhtarba8vPzUqVMTJkywsrKytrbW/ZG+dOlSXV3dfX8StFrtr7/+KhaLR44c+ahzJiYmarVaNze3Rz2grq5OpVI9eD+bzeZyuffd2Z0/ZBcVFeXm5r7++uvP+Zs0MDBYvny5Wq1OSkoqLS2dOnWqbn1F99FSr0mKompqam7dupWdnQ0AAQEB3e032VKvybq6OoVCocvSsdnsmpqajIyMbhWAqlSq7OzsAQMGnD17dt++fTExMfctvgeA7Ozso0eP8vn8qqoqLpdbV1f38ssvnzp1qrKyksViyeXy6Ojo6urqH374wcLCwszM7Nq1a2+//fbJkydPnz69cOHC2tran376icVibd++PTg42NHRkUajFRcXnz17Njs7m6KocePG1dXVrVq1qqGhISoqqqKiIicnJywsrLKyMisri8lkjhs3jsViPXTwbfV7Qm0KA9DO6i7cLYVSCu5/E2lBWtCagqkRtOQS8qKiov3792dkZMyaNas5K1ZfX3/8+PHx48ff+0jd+2NFRYXuz8ajXLlyxcLCwtzc/KHflcvl27dvLywsvC8XpdFoXF1dY2JiWjczV1MAsgp44I2+JWk1ILQAbgtsc0lKSqLT6c7Ozi3ym7x169a2bdvEYnGvXr2ef2y5uVBVBc+XT/wPGg1YW4O+fgucqqVekzQaTSQScblcU1PTX3/9NTEx8dVXX33OsSnuVihKq1r5NallmYpYRgYd5zVpY2NjYWFRWloKAMVNGhoannNslQptgZxM7LQerRYMWZQFtwVe92VlZWq12traOjo6+osvvrh79+69aXWKorRa7RdffBEUFDRmzJhXX301ICBgxIgR8fHxJ0+e/Pzzz7lc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)

·

Fig. 2: Relative regret curve with the very smooth demand.

handling contextual information and very smooth demand functions.

For future work, establishing the lower bound with respect to β remains an open challenge. This would require the careful construction of complex instances to fully understand the limitations of our method.

·

## References

- [1] Y. Abbasi-Yadkori, D. P´ al, and C. Szepesv´ ari. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems , 24, 2011.
- [2] P. Auer. Using confidence bounds for exploitation-exploration trade-offs. Journal of Machine Learning Research , 3(Nov):397-422, 2002.
- [3] G.-Y. Ban and N. B. Keskin. Personalized dynamic pricing with machine learning: Highdimensional features and heterogeneous elasticity. Management Science , 67(9):5549-5568, 2021.
- [4] O. Besbes and A. Zeevi. Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. Operations research , 57(6):1407-1420, 2009.
- [5] O. Besbes and A. Zeevi. On the (surprising) sufficiency of linear models for dynamic pricing with demand learning. Management Science , 61(4):723-739, 2015.
- [6] J. R. Birge, H. Chen, N. B. Keskin, and A. Ward. To interfere or not to interfere: Information revelation and price-setting incentives in a multiagent learning environment. Operations Research , 2024.
- [7] J. Bu, D. Simchi-Levi, and C. Wang. Context-based dynamic pricing with separable demand models. Available at SSRN 4140550 , 2022.
- [8] J. Bu, D. Simchi-Levi, and C. Wang. Context-based dynamic pricing with partially linear demand model. Advances in Neural Information Processing Systems , 35:23780-23791, 2022.
- [9] C. Cai, T. T. Cai, and H. Li. Transfer learning for contextual multi-armed bandits. The Annals of Statistics , 52(1):207-232, 2024.
- [10] N. Cesa-Bianchi, T. Cesari, and V. Perchet. Dynamic pricing with finitely many unknown valuations. In Algorithmic Learning Theory , pages 247-273. PMLR, 2019.
- [11] E. Chen, X. Chen, L. Gao, and J. Li. Dynamic contextual pricing with doubly non-parametric random utility models. arXiv preprint arXiv:2405.06866 , 2024.
- [12] N. Chen and G. Gallego. Nonparametric pricing analytics with customer covariates. Operations Research , 69(3):974-984, 2021.
- [13] N. Chen and G. Gallego. A primal-dual learning algorithm for personalized dynamic pricing with an inventory constraint. Mathematics of Operations Research , 47(4):2585-2613, 2022.
- [14] Y.-G. Choi, G.-S. Kim, C. Yunseo, W. Cho, M. C. Paik, and M.-h. Oh. Semi-parametric contextual pricing algorithm using cox proportional hazards model. In International Conference on Machine Learning , pages 5771-5786. PMLR, 2023.
- [15] W. Chu, L. Li, L. Reyzin, and R. Schapire. Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics , pages 208-214. JMLR Workshop and Conference Proceedings, 2011.

- [16] A. V. Den Boer. Dynamic pricing and learning: historical origins, current research, and new directions. Surveys in operations research and management science , 20(1):1-18, 2015.
- [17] Y. E. Erginbas, T. Courtade, K. Ramchandran, and S. Phade. Online pricing for multi-user multi-item markets. Advances in Neural Information Processing Systems , 36:29718-29740, 2023.
- [18] J. Fan, Y. Guo, and M. Yu. Policy optimization using semiparametric models for dynamic pricing. Journal of the American Statistical Association , 119(545):552-564, 2024.
- [19] D. Foster and A. Rakhlin. Beyond ucb: Optimal and efficient contextual bandits with regression oracles. In International Conference on Machine Learning , pages 3199-3210. PMLR, 2020.
- [20] E. Gin´ e and R. Nickl. Confidence bands in density estimation. 2010.
- [21] N. Golrezaei, A. Javanmard, and V. Mirrokni. Dynamic incentive-aware learning: Robust pricing in contextual auctions. Advances in Neural Information Processing Systems , 32, 2019.
- [22] X. Gong and J. Zhang. Stochastic graph bandit learning with side-observations, 2023.
- [23] Y. Gur, A. Momeni, and S. Wager. Smoothness-adaptive contextual bandits. Operations Research , 70(6):3198-3216, 2022.
- [24] A. Javanmard and H. Nazerzadeh. Dynamic pricing in high-dimensions. Journal of Machine Learning Research , 20(9):1-49, 2019.
- [25] T. Lattimore and C. Szepesv´ ari. Bandit algorithms . Cambridge University Press, 2020.
- [26] L. Li, Y. Lu, and D. Zhou. Provably optimal algorithms for generalized linear contextual bandits. In International Conference on Machine Learning , pages 2071-2080. PMLR, 2017.
- [27] M. Li, D. Simchi-Levi, R. Tan, C. Wang, and M. X. Wu. Contextual offline demand learning and pricing with separable models. Available at SSRN 4619018 , 2023.
- [28] S. Li, C. Shi, and S. Mehrotra. Lego: Optimal online learning under sequential price competition. Available at SSRN 4803002 , 2024.
- [29] X. Li and Z. Zheng. Dynamic pricing with external information and inventory constraint. Management Science , 70(9):5985-6001, 2024.
- [30] Y. Li, Y. Wang, and Y. Zhou. Nearly minimax-optimal regret for linearly parameterized bandits. In Conference on Learning Theory , pages 2173-2174. PMLR, 2019.
- [31] A. Locatelli and A. Carpentier. Adaptivity to smoothness in x-armed bandits. In Conference on Learning Theory , pages 1463-1492. PMLR, 2018.
- [32] Y. Luo, W. W. Sun, and Y. Liu. Contextual dynamic pricing with unknown noise: Explorethen-ucb strategy and improved regrets. Advances in Neural Information Processing Systems , 35:37445-37457, 2022.

- [33] Y. Luo, W. W. Sun, and Y. Liu. Distribution-free contextual dynamic pricing. Mathematics of Operations Research , 49(1):599-618, 2024.
- [34] M.-h. Oh, G. Iyengar, and A. Zeevi. Sparsity-agnostic lasso bandit. In International Conference on Machine Learning , pages 8271-8280. PMLR, 2021.
- [35] S. Saharan, S. Bawa, and N. Kumar. Dynamic pricing techniques for intelligent transportation system in smart cities: A systematic review. Computer Communications , 150:603-625, 2020.
- [36] K. Takemura, S. Ito, D. Hatano, H. Sumita, T. Fukunaga, N. Kakimura, and K.-i. Kawarabayashi. A parameter-free algorithm for misspecified linear contextual bandits. In International Conference on Artificial Intelligence and Statistics , pages 3367-3375. PMLR, 2021.
- [37] Y. Wang, B. Chen, and D. Simchi-Levi. Multimodal dynamic pricing. Management Science , 67(10):6136-6152, 2021.
- [38] J. Xu and Y.-X. Wang. Towards agnostic feature-based dynamic pricing: Linear policies vs linear valuation with unknown noise. In International Conference on Artificial Intelligence and Statistics , pages 9643-9662. PMLR, 2022.
- [39] J. Xu and Y.-X. Wang. Pricing with contextual elasticity and heteroscedastic valuation. arXiv preprint arXiv:2312.15999 , 2023.
- [40] Z. Ye and H. Jiang. Smoothness-adaptive dynamic pricing with nonparametric demand learning. In International Conference on Artificial Intelligence and Statistics , pages 1675-1683. PMLR, 2024.

## A Proofs

## A.1 Proofs in non-contextual model

## Proof of Proposition 3.1.

Proof. To simplify notation, we denote the prices at rounds in D j t,s as p 1 , p 2 , · · · , p n where n ≥ /pi1 ( β ) + 1. We define the matrix

<!-- formula-not-decoded -->

By the algorithmic construction, the first /pi1 ( β ) + 1 prices are distinct. Consequently, A forms a Vandermonde matrix, which is known to be invertible since φ j is a polynomial map.

Now, considering all column vectors, we analyze the rank of the set:

<!-- formula-not-decoded -->

The rank of this set is /pi1 ( β ) + 1 because the first /pi1 ( β ) + 1 vectors are linearly independent and the degree of the polynomial map φ j is also /pi1 ( β ) + 1. Thus, we can express the rank of the Gram matrix Λ j t,s as follows:

<!-- formula-not-decoded -->

Since the Gram matrix Λ j t,s is a ( /pi1 ( β ) + 1) × ( /pi1 ( β ) + 1) matrix, it follows that Λ j t,s is invertible.

## Proof of Lemma 3.1.

Proof. Let H 0 denote the filtration generated by

<!-- formula-not-decoded -->

Now we consider the concentration for /epsilon1 τ , τ ∈ D j t,s . For notation brevity, denote H τ as the filtration generated by { /epsilon1 τ ′ : τ ′ ∈ D j t,s , τ ′ &lt; τ } ∪ H 0 . Then for τ = 0, the definition of H τ is consistent with H 0 . To apply Azuma's inequality, we need to check /epsilon1 τ is a martingale difference

  Based on the algorithmic construction, the selection of round τ into Ψ s t depends solely on H 0 . Therefore, condition on H 0 , { /epsilon1 τ , τ ∈ Ψ s t } are independent random variables. Considering the noise /epsilon1 τ with index τ in Ψ s t , we have E [ /epsilon1 τ |H 0 ] = 0 from the realizability condition.

/square

sequence adapted to filtration H τ . From the conditional independence of /epsilon1 τ , we have

<!-- formula-not-decoded -->

Let ˜ θ j t,s = (Λ j t,s ) -1 ∑ τ ∈D j t,s [( θ j ) /latticetop φ j ( p τ )+ /epsilon1 τ ] φ j ( p τ ). We fix arbitrary s ∈ [ S ] , t ∈ [ T ] and j ∈ A t,s . From the definition of ˜ θ j t,s , we obtain

<!-- formula-not-decoded -->

We further apply Azuma's inequality to /epsilon1 τ , τ ∈ D j t,s . By doing so, we can derive the following results:

<!-- formula-not-decoded -->

Therefore, we obtain

<!-- formula-not-decoded -->

Moreover, we have

It remains to bound

<!-- formula-not-decoded -->

where the inequality (a) is Cauchy-Schwarz inequality, the inequality (b) is from the condition of the step (a) in Algorithm 1, and the inequality (c) is due to Lemma A.2.

<!-- formula-not-decoded -->

By the triangular inequality, we find

<!-- formula-not-decoded -->

/square

## Proof of Lemma 3.2.

Proof. We prove this lemma by induction on s . For s = 1, the lemma holds naturally as A t, 1 = [ N ] and thus rev ( p ∗ ) = rev ( p j ∗ t, 1 t, 1 ). Assume that the bound holds in the layer s . It is sufficient to show that

If j ∗ t,s = j ∗ t,s +1 , the desired bound holds. Hence we assume that j ∗ t,s / ∈ A t,s +1 . Let ˆ j t,s := arg max j ∈A t,s sup p ∈ [ a j -1 ,a j ] pU j t,s ( p ) be the index with the highest UCB in A t,s . From the step (a) we know ˆ j t,s ∈ A t,s +1 . Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is due to Lemma B.1. From the definition of Γ, we know that

<!-- formula-not-decoded -->

From the algorithmic construction (step (b)), we know

<!-- formula-not-decoded -->

Since j ∗ t,s / ∈ A t,s by our assumption, the step (a) yields

<!-- formula-not-decoded -->

Combing all inequalities above, we obtain

<!-- formula-not-decoded -->

## Proof of Lemma 3.3.

Proof. For any layer s &lt; s t , the step (a) shows that

<!-- formula-not-decoded -->

where the last inequality is due to the definition of Γ.

The step (b) implies that

<!-- formula-not-decoded -->

as j t ∈ A t,s and s &lt; s t . Combing two inequalities we obtain

<!-- formula-not-decoded -->

Therefore, from the definition of Γ we have

3

p

max

2

1

-

s

<!-- formula-not-decoded -->

Rearranging all terms yields the desired inequality.

## Proof of Lemma 3.4.

/square

Proof. It follows from Lemma B.1 and Lemma 3.3 that

<!-- formula-not-decoded -->

/square

## Proof of Theorem 3.1.

Proof. Let Ψ 0 be the set of rounds for which an alternative is chosen when p t » φ j ( p t ) /latticetop (Λ j t, 1 ) -1 φ j ( p t ) ≤ 1 / √ T . Since 2 -S ≤ 1 / √ T , we have Ψ 0 ∪ ∪ s ∈ [ S ] Ψ s T +1 = [ T ]. Recall that p ∗ maximizes pf ( p ).

<!-- formula-not-decoded -->

where the last inequality is due to Lemma 3.4.

From Lemma A.2, we have which implies

As a corollary, we have

<!-- formula-not-decoded -->

Therefore, we have and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Lemma A.3, we know that

<!-- formula-not-decoded -->

For the regret caused in the initialization step, we directly bound the caused regret as

<!-- formula-not-decoded -->

Since N = /ceilingleft ( p max -p min ) 2 β 2 β +1 T 1 2 β +1 /ceilingright , we have

<!-- formula-not-decoded -->

Proof of Lemma 4.1. Proof. Consider the interval [ a j -1 , a j ]. We define the following stopping time

<!-- formula-not-decoded -->

where n denotes the sample size of D j . Consequently, we have

<!-- formula-not-decoded -->

Now we apply Lemma B.2 to establish an upper bound for T j . We have

<!-- formula-not-decoded -->

for δ &gt; 0. By setting δ = 1 /T we can prove E [ T j ] = O (log T ). Finally, we conclude the proof by summing over the N stopping times, yielding the desired result.

/square

## Proof of Lemma 4.2.

Proof. Let P n be a n × /lscript matrix with its j -th row ϕ ( p j , x j ) /latticetop for every m and d n = ( d 1 , d 2 , · · · , d n ) /latticetop . By the least square regression, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to least square estimation theory [40], we have

<!-- formula-not-decoded -->

with probability at least 1 -O ( /lscripte -C 2 ln 2 n ) for some constants C 2 depending on u 2 , α and β , and for n larger than some constants C 1 depending on u ′ 1 , u 2 , /lscript .

Note that

Define

<!-- formula-not-decoded -->

By the H¨ older assumption and Lemma B.1, there exists an /lscript + 1 dimensional vector θ 1 such that

<!-- formula-not-decoded -->

for ∀ p ∈ [ a, b ]. Hence, we have

<!-- formula-not-decoded -->

Therefore, we obtain that

<!-- formula-not-decoded -->

for some larger n &gt; C 1 depending on u ′ 1 , u 2 , /lscript .

Note that ϕ ( p, x ) = O (1) and ‖ ϕ ( p, x ) /latticetop ( ˆ θ -θ 1 ) ‖ = O ( ‖ ˆ θ -θ 1 ‖ ), ∀ p ∈ [ a, b ], n &gt; C 1 . We have

<!-- formula-not-decoded -->

We conclude the proof by noticing that E [ d p, x ] = f ( p ) + µ /latticetop x

. /square

Proof of Theorem 4.2. Proof. We first define an event A = {∃ i ∈ { 1 , 2 } , m ∈ { 1 , 2 , · · · , K i } , s.t., | O i,m | &lt;

|

T i 2 K i } . From the concentration inequality, we have

<!-- formula-not-decoded -->

Bt conditioning on A c , we can guarantee the number of samples in each interval.

We then consider the inequalities when the event A c holds. Define the interval I i,m = [ p min + ( m -1)( p max -p min ) K i , p min + m ( p max -p min ) K i ]. Invoking Lemma 4.2, with a probability of at least 1 -O ( /lscripte -C ln 2 T ), we have ∀ p ∈ I i,m ,

<!-- formula-not-decoded -->

for some sufficiently small constants v 1 , v 2 . Subsequently, we deduce the following upper bound

<!-- formula-not-decoded -->

for a small constant c &gt; 0.

From the proof of Lemma 4.2, we have

<!-- formula-not-decoded -->

Given the self-similarity condition of f f, we can establish

<!-- formula-not-decoded -->

Combining all inequalities above, a lower bound can be derived as

<!-- formula-not-decoded -->

From the algorithmic construction, we have

<!-- formula-not-decoded -->

with probability at least 1 -O (( α + /pi1 ( β max ) + 1) e -C ln 2 T ).

Lemma A.1 Assuming |D j t,s | ≥ 2 , then for arbitrary t and s , we have

<!-- formula-not-decoded -->

Proof. We have

<!-- formula-not-decoded -->

From the lemma 13 in [2] or the lemma 3 in [15], we know that

<!-- formula-not-decoded -->

## Lemma A.2 For all s , we have

<!-- formula-not-decoded -->

Proof. From the step (b), we know that

<!-- formula-not-decoded -->

By Lemma A.1, we obtain

<!-- formula-not-decoded -->

Therefore, combing above inequalities, we have

<!-- formula-not-decoded -->

/square

/square

Lemma A.3 Given the event Γ , for every round t ∈ Ψ 0 , we have

<!-- formula-not-decoded -->

Proof. Let j ∗ be the index such that p ∗ ∈ [ a j ∗ -1 , a j ∗ ].

<!-- formula-not-decoded -->

## A.2 Proofs in linear contextual effect model

Lemma A.4 Consider the round t and any layer s ∈ [ s t ] in the round t . Then for each j ∈ [ N ] , we have

<!-- formula-not-decoded -->

with probability at least 1 -δ NST .

Proof. In this proof, we modify the filtration H 0 as the following one generated by

<!-- formula-not-decoded -->

The remaining argument is the same as that in Lemma 3.1.

Let ˜ θ j t,s = (Λ j t,s ) -1 ∑ τ ∈D j t,s [( θ j ) /latticetop ϕ j ( p τ , x τ ) + /epsilon1 τ ] ϕ j ( p τ , x τ ). We fix arbitrary s ∈ [ S ] , t ∈ [ T ] and

/square

/square

j ∈ A t,s . From the definition of ˜ θ j t,s , we obtain

<!-- formula-not-decoded -->

We further apply Azuma's inequality to /epsilon1 t . By doing so, we can derive the following results:

<!-- formula-not-decoded -->

Therefore, we obtain

<!-- formula-not-decoded -->

Moreover, we have

It remains to bound where the inequality (a) is Cauchy-Schwarz inequality, the inequality (b) is from the condition of the step (a) in Algorithm 1, and the inequality (c) is due to Lemma A.9.

<!-- formula-not-decoded -->

By the triangular inequality, we find

<!-- formula-not-decoded -->

/square

Based on Lemma A.4, we construct a high-probability event which satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, we introduce the revenue function

<!-- formula-not-decoded -->

Given the context x t , we define the index of the best price within the interval [ a j -1 , a j ] at the candidate set A t,s as

<!-- formula-not-decoded -->

We also denote the context-dependent best price as p ∗ t ∈ arg max p ∈ [ p min ,p max ] rev ( p, x t ).

Lemma A.5 Given the event Γ , then for each round t and each layer s ∈ [ s t -1] , we have

<!-- formula-not-decoded -->

Proof. We prove this lemma by induction on s . For s = 1, the lemma holds naturally as A t, 1 = [ N ] and thus rev ( p ∗ t , x t ) = rev ( p j ∗ t, 1 t, 1 , x t ). Assume that the bound holds in the layer s . It is sufficient to show that

<!-- formula-not-decoded -->

If j ∗ t,s = j ∗ t,s +1 , the desired bound holds. Hence we assume that j ∗ t,s / ∈ A t,s +1 . Let ˆ j t,s :=

arg max j ∈A t,s sup p ∈ [ a j -1 ,a j ] pU j t,s ( p ) be the index with the highest UCB in A t,s . From the step (a) we know ˆ j t,s ∈ A t,s +1 . Then we have

<!-- formula-not-decoded -->

where the last inequality is due to Lemma B.1. From the definition of Γ, we know that

<!-- formula-not-decoded -->

From the algorithmic construction (step (b)), we know

<!-- formula-not-decoded -->

Since j ∗ t,s / ∈ A t,s by our assumption, the step (a) yields

<!-- formula-not-decoded -->

Combing all inequalities above, we obtain

<!-- formula-not-decoded -->

Lemma A.6 Given the event Γ , then for each round t , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. For any layer s &lt; s t , the step (a) shows that

<!-- formula-not-decoded -->

where the last inequality is due to the definition of Γ.

The step (b) implies that

<!-- formula-not-decoded -->

as j t ∈ A t,s and s &lt; s t . Combing two inequalities we obtain

<!-- formula-not-decoded -->

Therefore, from the definition of Γ we have

3 p max 2 1 -s

<!-- formula-not-decoded -->

Rearranging all terms yields the desired inequality.

/square

Lemma A.7 Given the event Γ , for every round t , we have

<!-- formula-not-decoded -->

Proof. It follows from Lemma B.1 and Lemma A.6 that

<!-- formula-not-decoded -->

Lemma A.8 Assuming |D j t,s | ≥ 2 , then for arbitrary t and s , we have

<!-- formula-not-decoded -->

Proof. We have

<!-- formula-not-decoded -->

From the lemma 13 in [2] or the lemma 3 in [15], we know that

<!-- formula-not-decoded -->

Lemma A.9 For all s , we have

<!-- formula-not-decoded -->

/square

Proof. From the step (b), we know that

<!-- formula-not-decoded -->

By Lemma A.1, we obtain

<!-- formula-not-decoded -->

Therefore, combing above inequalities, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now we have all materials to prove Theorem 4.1.

Proof. Let Ψ 0 be the set of rounds for which an alternative is chosen when p t » φ j ( p t , x t ) /latticetop (Λ j t, 1 ) -1 φ j ( p t , x t ) ≤ 1 / √ T . Since 2 -S ≤ 1 / √ T , we have Ψ 0 ∪ ∪ s ∈ [ S ] Ψ s T +1 = [ T ].

Recall that p ∗ t maximizes rev ( p, x t ).

<!-- formula-not-decoded -->

where the last inequality is due to Lemma A.7.

From Lemma A.9, we have which implies

As a corollary, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we have and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Lemma A.3, we know that

<!-- formula-not-decoded -->

For the regret caused in the initialization step, we directly bound the caused regret as p max T . Thanks to Lemma 4.1, we have

<!-- formula-not-decoded -->

Since N = /ceilingleft ( p max -p min ) 2 β 2 β +1 T 1 2 β +1 /ceilingright , we have

<!-- formula-not-decoded -->

## B Related Material

Lemma B.1 ([37]) Suppose f satisfies Assumption 2.1 and let [ a, b ] ⊂ [ p min , p max ] be an arbitrary interval. There exists a /pi1 ( β ) -degree polynomial P [ a,b ] ( x ) such that

<!-- formula-not-decoded -->

Lemma B.2 ([26]) Define V n = ∑ n t =1 x t x /latticetop t , where x t is drawn i.i.d. from some distribution with support in the d -dimensional unit ball. Futhermore, let Σ = E [ xx /latticetop ] be thr Gram matrix, and B,δ be two positive constants. Then, there exist positive, universal constants C 1 and C 2 such that

/square

λ min ( V n ) ≥ B with probability at least 1 -δ , as long as

Lemma B.3 With an upper bound β max of the smoothness parameter and Assumption 2.1 , then for some constant C &gt; 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -e -C ln 2 T .