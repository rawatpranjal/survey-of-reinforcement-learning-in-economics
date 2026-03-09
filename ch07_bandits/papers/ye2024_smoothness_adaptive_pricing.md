## Smoothness-Adaptive Dynamic Pricing with Nonparametric Demand Learning

Zeqi Ye /diamondmath and Hansheng Jiang †

/diamondmath

Nankai University /diamondmath zye@mail.nankai.edu.cn † University of Toronto hansheng.jiang@utoronto.ca

## Abstract

Westudy the dynamic pricing problem where the demand function is nonparametric and H¨ older smooth, and we focus on adaptivity to the unknown H¨ older smoothness parameter β of the demand function. Traditionally the optimal dynamic pricing algorithm heavily relies on the knowledge of β to achieve a minimax optimal regret of ˜ O ( T β +1 2 β +1 ). However, we highlight the challenge of adaptivity in this dynamic pricing problem by proving that no pricing policy can adaptively achieve this minimax optimal regret without knowledge of β . Motivated by the impossibility result, we propose a self-similarity condition to enable adaptivity. Importantly, we show that the self-similarity condition does not compromise the problem's inherent complexity since it preserves the regret lower bound Ω( T β +1 2 β +1 ). Furthermore, we develop a smoothness-adaptive dynamic pricing algorithm and theoretically prove that the algorithm achieves this minimax optimal regret bound without the prior knowledge β .

## 1 INTRODUCTION

Dynamic pricing, the practice of adjusting prices in real-time based on varying market demand, has become an integral strategy in domains like e-commerce and transportation. An effective dynamic pricing model needs to adequately balance the exploration by learning demand at various prices and the exploitation by optimizing prices based on observed price and demand data. We consider a canonical dynamic pricing problem with nonparametric demand learning. At the period t , the decision maker chooses a price p t and observes a noisy demand d t , where E [ d t | p t = p ] = f ( p ) for some unknown function f : P → R ≥ 0 mapping from price set P to demand. The goal of dynamic pricing is to maximize the total revenue collected over a finite time horizon. The performance of a dynamic pricing policy or algorithm is measured by the cumulative regret when compared with the maximal revenue in hindsight. More broadly framed as an online optimization problem, the dynamic pricing problem features nonparametric demand learning in that f can be of any functional form and continuous action space where price can be chosen at any value in a given price interval. Dynamic pricing problem has been an active topic for decades (Kleinberg and Leighton, 2003) and has found numerous applications in retailing, auctions, and advertising (Den Boer, 2015).

Without much regularity assumption on the demand function f , the optimal regret is shown to be ˜ O ( T 2 / 3 ). This regret rate can be improved to ˜ O ( T 1 / 2 ) if the uniqueness of the maximum and certain local concavity property of the revenue function r ( p ) = p · f ( p ) is imposed. However, such uniqueness assumption can be restrictive in practice and therefore other regularity assumptions, notably the smoothness condition, of the demand functions are considered. Nonetheless, a prevalent limitation in these methodologies is the presupposed exact knowledge of the H¨ older smoothness level β . In reality, such assumptions are frequently misaligned with the complexities of real-world applications, thus constraining the practical applicability of these algorithms. Against this backdrop, our work distinguishes itself by delving into the uncharted territories of adaptability in dynamic pricing. Specifically, we address the pressing challenge of how to adapt when the H¨ older smoothness level β is not known.

†

Facing the challenge of unknown smoothness parameter, it is natural to ask the following question:

Can we design a dynamic pricing strategy that does not require the prior knowledge of β while maintaining the optimal regret of O ( T β +1 2 β +1 ) ?

˜ Our answer to this question is two-fold: on the one hand, it is impossible to achieve adaptivity without imposing additional assumptions; on the other hand, we identify a novel condition that achieves adaptivity without reducing the original pricing problem's complexity. Our contributions in this paper can be summarized as follows:

- Characterizing Adaptivity Challenge: We formally characterize the challenge of adaptivity. In particular, we prove that without additional conditions, achieving the optimal regret for functions without knowing the H¨ older smoothness parameter is impossible. We show that one algorithm with optimal regret for a certain H¨ older smoothness parameter can have sub-optimal regret when directly applied to function class with lower H¨ older smoothness levels.
- Proposing a Self-Similarity Condition: To make adaptivity possible, we propose a self-similarity condition, which serves as a dual to the H¨ older smoothness assumption. Furthermore, our analysis reveals notable properties of the self-similarity condition, in particular regarding its practical applicability and sustenance of the dynamic pricing problem's complexity. We find that the self-similarity condition not only enables adaptivity but also does not decrease the intrinsic complexity of the original pricing problem in that the lower bound Ω( T β +1 2 β +1 ) does not change.
- Optimal Minimax Regret Rate: We design a Smoothness-Adaptive Dynamic Pricing ( SADP ) algorithm by incorporating a dedicated phase for the estimation of the smoothness parameter. Under the self-similarity condition, we establish a tight confidence interval for the estimated H¨ older smoothness parameter. We derive an optimal regret bound ˜ O ( T β +1 2 β +1 ) that matches the same optimal bound obtained by previous algorithms that require the knowledge of β .

Organization and Notation In Section 2, we introduce related literature on dynamic pricing, bandits, and statistics. In Section 3, we explicitly formulate the dynamic pricing problem under H¨ older smooth demand functions and introduce the adaptivity problem by first presenting the non-adaptive dynamic pricing algorithm. We discuss in-depth the adaptivity challenge in Section 4 and present two key favorable properties of the self-similarity condition. In Section 5, we present our smoothness adaptive dynamic pricing algorithm and give a detailed regret analysis. Lastly, we conclude the paper with discussions and future directions in Section 6.

/negationslash

Throughout the paper, the vectors are column vectors unless specified otherwise. The notation ‖ x ‖ denotes the L 2 norm of vector x , and given matrix A , the notation ‖ x ‖ A = ( x T Ax ) 1 / 2 denotes the A -norm of vector x . For matrix A , ‖ A ‖ = sup x =0 ‖ x T Ax ‖ / ‖ x ‖ denotes the L 2 operator norm of matrix A . We employ the notation O ( · ), Ω( · ), Θ( · ) to conceal constant factors, and ˜ O ( · ), ˜ Ω( · ), ˜ Θ( · ) are used to mask both constant and logarithmic factors.

## 2 Related Literature

Dynamic Pricing with Demand Learning Motivated by the applications in e-commerce and transportation, numerous works have studied dynamic pricing with continuous price space and demand learning (Kleinberg and Leighton, 2003; Besbes and Zeevi, 2009; Broder and Rusmevichientong, 2012; Besbes and Zeevi, 2012; Keskin and Zeevi, 2014; Chen and Gallego, 2022). The crux of non-contextual dynamic pricing lies in modeling and learning the unknown price and demand relationship. Earlier works mainly focus on parametric demand models with additional concavity property of the revenue function where a regret ˜ O ( √ T ) is typically shown to be optimal. For nonparametric demand models, ˜ O ( T k +1 2 k +1 ) regret can be achieved if the demand function is k times differentiable reward function for some integer k &gt; 0, and moreover a matching lower bound of Θ( T k +1 2 k +1 ) can be established (Wang et al., 2021). However, the smoothness level k needs to be known prior to the algorithmic design, and it is thus unclear if existing

algorithms are able to adapt to different smoothness levels. Our work improves upon Wang et al. (2021) by proposing a smoothness-adaptive dynamic pricing algorithm with the same minimax optimal regret rate and additionally, we extend the integer k to more generally β -smooth for any β ∈ R + .

In certain applications, consumer or product features, also known as contexts, are available and can be parametrized into the demand valuation (Qiang and Bayati, 2016; Javanmard, 2017; Cohen et al., 2020; Ban and Keskin, 2021; Xu and Wang, 2021). The landscape of regret analysis in contextual cases typically ranges from log( T ) to ˜ O ( √ T ) depending on different parametric or semiparametric assumptions on demand valuation and market noise. The smoothness level of both the demand function and the noise function may affect the regret bound, and theoretical results for adaptively learning the smoothness level are not known (Fan et al., 2022; Bu et al., 2022).

Continuum-Armed Bandit Problems Dynamic pricing is closely related to the continuum-armed bandit problem, where the actions are not discrete but rather lie in a continuous space as in the case of the continuous price space. Adaption to H¨ older smoothness level β while achieving the minimax regret rate has been considered in continuum-armed bandits as well. It is shown in Locatelli and Carpentier (2018) and Hadiji (2019) that adaptivity for free is generally impossible. Our non-adaptivity result for dynamic pricing shares the same spirit as in the continuum-armed bandit problem but requires different construction of function classes in the arguments. Liu et al. (2021) propose to use a general model approach for bandit problems, but the analysis only applies to the subcase of β ≤ 1. Due to non-adaptivity, additional assumptions are therefore necessary for establishing adaptivity. Specifically, the assumption of self-similarity emerges as a promising candidate because it has been demonstrated to maintain the minimax regret rates in both continuum-armed bandits (Cai and Pu, 2022) and contextual bandits (Gur et al., 2022) scenarios.

Adaptivity in Statistics More broadly, adaptive inference and adaptive estimators have been widely considered in statistics, but less is known if these techniques are suited for regret minimization. While several structural conditions have profound implications in nonparametric regression, such as monotonicity, concavity, as discussed in Cai et al. (2013), introducing any of these assumptions may either significantly diminish the problem's complexity or do not directly contribute to the learning of the smoothness parameter (Slivkins et al., 2019; Cai and Pu, 2022). Consequently, with any of these structural assumptions at play, the minimax regret operates at the parametric rate, making it agnostic to smoothness variations.

## 3 PRELIMINARIES

Problem Description We consider the dynamic pricing problem with demand learning over a finite time horizon of length T . At every time period t = 1 , . . . , T , the seller selects a price p t ∈ [ p min , 1], where 0 &lt; p min &lt; 1 is a predetermined price lower bound and the price upper bound is normalized to 1 without loss of generality. After the seller sets the price, the customers then arrive and a randomized demand d t ∈ [0 , d max ] is incurred. The randomized demand d t given price is determined by a demand function f : [ p min , 1] → [0 , d max ] and some random market noise, and the expectation of the randomized demand E [ d t | p t = p ] = f ( p ). The noise in demand d t -f ( p ) follows a sub-gaussian distribution with respect to some parameters. The revenue collected at time t is r t = p t · d t , and the expected revenue given p t is p t × f ( p t ).

As is common in previous literature on pricing (Wang et al., 2021; Bu et al., 2022), the H¨ older smoothness assumption is used to constrain the volatility of the demand function f in any given region. Throughout the paper, the demand function f is assumed to belong to the H¨ older smooth function class H ( β, L ) for certain β, L &gt; 0 that are defined as follows.

Definition 1 (H¨ older Smooth Function Class) . The H¨ older class of functions H 0 ( β, L ) is defined to be the set of w ( β ) times continuously differentiable functions g : [ p min , 1] → R such that for any p, p ′ ∈ [ p min , 1] ,

<!-- formula-not-decoded -->

where w ( β ) is the largest integer that is strictly smaller than β . We further define the function class H ( β, L ) as

<!-- formula-not-decoded -->

Policy and Regret An admissible dynamic pricing policy π over T selling periods is a sequence of T random functions π 1 , π 2 , · · · , π T such that π t : ( p 1 , d 1 , · · · , p t -1 , d t -1 ) ↦→ p t is a mapping function that maps the history prior to time t to a price p t . Since the demand function belongs to H ( β, L ) and thus continuous over [ p min , 1], there exists some optimal price p ∗ ∈ arg max p ∈ [ p min , 1] E [ r t | p t = p ]. Note that here we do not require the optimal price to be unique.

The performance of dynamic pricing policies is evaluated by the cumulative regret defined as follows. For an admissible dynamic pricing policy π over T selling periods, the regret R π ( T ) over a time horizon T is

<!-- formula-not-decoded -->

where the price sequence { p t } T t =1 is determined by the policy.

Non-Adaptive Pricing If the smoothness parameter β is known, non-adaptive dynamic pricing algorithms can achieve the optimal regret rate, which is called H¨ older-Smooth Dynamic Pricing ( HSDP ) algorithm and presented in Algorithm 1. The algorithm is designed based on the following idea. We first segment the price interval into many small intervals, and the length of each small bin depends on β , and then we can run local polynomial regression to approximate the true demand function in each small price interval separately. As we formally show later in Lemma 3, this non-adaptive algorithm achieves the optimal regret if it is run with the correct smoothness parameter.

## Algorithm 1 H¨ older-Smooth Dynamic Pricing ( HSDP )

Input: Time horizon T , H¨ older smoothness β , minimum price p min , maximum demand d max , number of bins N , parameter L &gt; 0, optional initial history D (0) ;

- 1: Set polynomial degree k = w ( β );
- 2: Partition [ p min , 1] into N segments of equal lengths, denoted as I j = [ a j , b j ] where a j = p min + ( j -1)(1 -p min ) N , b j = p min + j (1 -p min ) N for j = 1 , 2 , · · · , N , and let ∆ = L ( 1 -p min N ) ˆ β ;
- 5: Compute CI j := [∆ + (3 d max + L ) √ 2 √ n j ]( k +1)ln(2( k +1) T );
- 3: Initialize segment history, realized demands and trial numbers D j := { ( p t , d t ) : p t ∈ I j } , τ j := ∑ p t ∈D j p t d t , n j := |D j | where D j ⊂ D (0) for all 1 ≤ j ≤ N ; 4: for t = 1 , 2 , · · · , T do
- 6: Select j t := arg max 1 ≤ j ≤ N τ j n j + CI j ;
- 8: Do local polynomial regression on I j t with ridge type penalty and the estimator ˆ θ = arg min θ ∈ R k +1 ∑ ( p,d ) ∈D j t | d -〈 ˆ θ, φ ( k ) ( p ) 〉| 2 + ‖ θ ‖ 2 2 ;
- 7: Let δ = 1 T 2 , compute γ = L √ k +1+∆ √ |D j t | + d max √ 2( k +1)ln( 4( k +1) t δ )+2 and Λ = I ( k +1) × ( k +1) + ∑ ( p,d ) ∈D j t φ ( k ) ( p ) φ ( k ) ( p ) T ;
- 9: Set price p t = argmax p ∈ I j t p × min { d max , 〈 ˆ θ, φ ( k ) ( p ) 〉 + γ √ φ ( k ) ( p ) T Λ -1 φ ( k ) ( p ) + ∆ } ; 10: Observe realized demand d t ∈ [0 , d max ];
- 12: end for
- 11: Update τ j ← τ j + d t p t , n j ← n j +1 , D j ←D j ∪{ ( p t , d t ) } for j = j t ;

To help illustrate Algorithm 1, we introduce the concept of local polynomial regression, a crucial component of both Algorithm 1 and our smoothness-adaptive dynamic pricing algorithm that will be introduced later. Compared to conventional regression methods, the local polynomial regression approach incorporates a scaling process that offers several advantages in terms of flexibility, adaptability, and efficiency. By applying

the local polynomial regression with respect to a carefully chosen support set and focusing on specific small intervals, our method can effectively estimate the mean demand function with greater accuracy, making it suitable for a wide range of applications in dynamic pricing. The scaling process also allows for more efficient computation and model fitting, particularly in situations where data is limited or sparse.

Let t m ( p ) = ( 1 2 + p -a + b 2 b -a ) m and vector φ ( l ) ( p ) = ( t 0 ( p ) , t 1 ( p ) , ..., t l ( p )) T for some integer l. Define

Definition 2 (Local Polynomial Regression) . Let O = {( p (1) , d (1) ) , ... ( p ( m ) , d ( m ) )} be a sequence of observations, where p (1) has support ⊂ [ p min , 1] . Our goal is to estimate E [ d (1) | p (1) ] with these samples nonparametrically. Let I = [ a, b ] ⊂ [ p min , 1] , and let those observations such that p ( i ) ∈ I be { O I = ( p (1) , d (1) ) , ... ( p ( m 0 ) , d ( m 0 ) ) } . Then we can estimate E [ d (1) | p (1) ] by fitting a polynomial regression on [ a, b ] with samples in O I .

<!-- formula-not-decoded -->

For concreteness, if the minimizer is not unique we define ˆ θ = 0 . The local polynomial regression estimate on I is given by

<!-- formula-not-decoded -->

By using the local polynomial regression, we can leverage the H¨ older smoothness condition to improve the approximation error at each small price interval. Suppose the length of a price interval is /epsilon1 , a constant approximation would lead to approximation error O ( /epsilon1 ) if the demand function f is Lipschitz and generally no approximation guarantee without Lipschitz condition. On the other hand, if f ∈ H ( β, L ), we can bound the approximation error by O ( /epsilon1 β ), which improves O ( /epsilon1 ) error for β &gt; 1, and is also strictly better when β &lt; 1 and the Lipschitz condition fails to hold. Details of this approximation guarantee are presented in Lemma 12 in Appendix C.1.

Importantly, Algorithm 1 relies on the input of β to construct the number of small intervals N , which is decided as /ceilingleft T 1 2 β +1 /ceilingright . The parameter β affects the number of small intervals and therefore the length of each small interval. The underlying reason is that the approximation error of the local polynomial regression step crucially depends on the interval length, which plays an important role when establishing the optimal regret bound ˜ O ( T β +1 2 β +1 ). It is therefore highly nontrivial, if not impossible, to remove the dependence on β from the design of Algorithm 1.

## 4 ADAPTIVITY TO UNKNOWN SMOOTHNESS

## 4.1 Difficulty of Adaption

We show that it is impossible for any policy to achieve adaptivity without additional assumptions. This statement is formalized by establishing Theorem 1 where we consider two different smoothness levels. In Theorem 1, we show that a policy that achieves the optimal regret rate on a smoothness level α could not simultaneously do so on a smoothness level β &lt; α .

Theorem 1. It is impossible to achieve adaption without additional assumptions. Fix any two positive H¨ older smoothness parameters α &gt; β &gt; 0 , and parameters L ( α ) , L ( β ) &gt; 0 . Suppose that there is a policy π achieves the optimal regret ˜ O ( T α +1 2 α +1 ) over E [ d | p ] = f ( p ) ∈ H ( α, L ( α )) , then there exists a constant C &gt; 0 that is independent of π such that

<!-- formula-not-decoded -->

which means that it cannot achieve the optimal regret over E [ d | p ] = f ( p ) ∈ H ( β, L ( β )) .

The proof of Theorem 1 is accomplished by constructing a single basis function. From this basis function, we then generate demand functions with distinct H¨ older smoothness levels of α and β . By comparing the

Kullback-Leibler divergence between the resulting probability measures under these different conditions, we establish the existence of a regret gap. The comprehensive proof of Theorem 1 can be found in Appendix A.1.

The negative result in Theorem 1 highlights the difficulty of adaption and therefore necessitates the need for introducing additional conditions. A potential condition should ideally not only make the adaptivity possible for a wide range of functions as large as possible but also not trivialize the pricing problem's complexity .

## 4.2 Self-Similarity Condition

We identify the self-similarity condition to enable adaptivity with desirable properties. Before introducing the definition, we need some notation regarding function projections onto the space of polynomial functions. For any positive integer l , let Poly ( l ) denote the set of all polynomials of degree less than or equal to l . For any function g ( · ), we use Γ U l g ( · ) to denote the L 2 -projection of the function g ( · ) onto Poly ( l ) over some interval U , which can be computed by the following minimization

<!-- formula-not-decoded -->

Definition 3 (Self-Similarity Condition) . A function g : [ a, b ] → R , [ a, b ] ⊆ [0 , 1] is self-similar on [ a, b ] with parameters β, l ∈ Z + , M 1 ∈ R ≥ 0 , M 2 ∈ R + if for some positive integer c &gt; M 1 it holds that where we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any positive integer c . We denote the class of self-similar functions by S ( β, l, M 1 , M 2 ) .

In contrast to H¨ older smoothness, the self-similarity condition provides a global lower bound on the approximation error using polynomial regression. This dual nature facilitates the estimation of the smoothness of payoff functions by comparing the approximation on different scales. The self-similarity condition has previously appeared in the literature of nonparametric regression for constructing adaptive confidence intervals (Picard and Tribouley, 2000; Gin´ e and Nickl, 2010).

Example 1 (Example of Self-Similar Functions) . Let f be any function with continuous first-order derivative uniformly bounded by C 1 . We define the function class F as F ( f ) = { f : x ↦→ c 0 · x β + f : c 0 ∈ R , | c 0 | ≥ C 1 } , then all function in F ( f ) is self-similar with parameters β, l = 0 for some constants M 1 , M 2 depending on C 1 and C 2 .

To enable adaptivity, we assume that the self-similar condition holds for the demand function.

Assumption 1. The demand function f is self-similar with some parameters β, l, M 1 , M 2 .

## 4.3 Lower Bound

We emphasize that adding this self-similarity condition does not diminish the hardness of the dynamic pricing problem. We validate this statement by showing a worst-case lower bound of any policy Ω( T β +1 2 β +1 ) even in the presence of the self-similarity condition. Notably, this rate matches exactly with the lower bound given in the original problem without self-similarity condition (Wang et al., 2021).

Theorem 2 (Lower Bound) . Self-similarity does not change the minimax regret rate and therefore does not lower the problem difficulty for any admissible dynamic pricing policy π . Formally, for any positive parameters β, M 1 , L &gt; 0 , there exists a constant M 2 &gt; 0 satisfying that

<!-- formula-not-decoded -->

Theorem 2 says that there exists a class of non-trivial instances belonging to the self-similar function class such that the worst-case regret of any pricing policy is lower bounded by Ω( T β +1 2 β +1 ). A main challenge of proving Theorem 2 is therefore constructing such a class of demand functions to establish the lower bound. We now explain our constructions.

Let u ( · ) : [0 , 1] → R be a C ∞ function with u (0) = 1 , u (1) = 0 , u ( k ) (0) = 0 , u ( k ) (1) = 0 , ∀ k ∈ Z + . Let u 1 ( x ) = sin 2 ( π 2 · u (4 x -1) ) . ∀ x ∈ [0 , 1], let σ ( x ) = | x -1 2 | and define for some sufficiently small constant c 1 &gt; 0, then we have g ∈ H ( β, L ). On the other hand, we know that for some constant M 2 &gt; 0, g ∈ S ( β, w ( β ) , 0 , M 2 ) (Gur et al., 2022, Lemma 1.7). Also, g has a unique maximum point at x = 1 2 and for any x ∈ { 0 , 1 } , l ∈ { 0 , 1 , · · · , w ( β ) } it holds that g ( l ) ( x ) = 0.

<!-- formula-not-decoded -->

As a result, we have shown that for each L, β &gt; 0, there exists a function g : [0 , 1] → [0 , 1] satisfying that g ∈ H ( β, L ) ∩S ( β, w ( β ) , 0 , M 2 ) for some constant M 2 &gt; 0; and g has a unique maximizer at 1 2 ; and for any x ∈ { 0 , 1 } , k ∈ { 0 , 1 , · · · , w ( β ) } it holds that g ( k ) ( x ) = 0. The constructed function class, combined with a classical argument with the Kullback-Leibler divergence, leads to the lower bound stated in Theorem 2, and a complete proof of Theorem 2 can be found in Appendix A.4.

## 5 ALGORITHM AND REGRET ANALYSIS

In this section, we introduce our Smoothness-Adaptive Dynamic Pricing ( SADP ) algorithm and provide a detailed regret analysis.

## 5.1 Algorithm Description

We now present our SADP algorithm described in Algorithm 2, which incorporates an efficient smoothness parameter selection phase and is designed to adapt to the unknown H¨ older smoothness level.

Algorithm 2 Smoothness-Adaptive Dynamic Pricing ( SADP ) Input: Time horizon T , H¨ older smoothness range [ β min , β max ], minimum price p min , maximum demand d max , parameter L &gt; 0; 1: Set local polynomial regression degree l = w ( β max ); 2: Set k 1 = 1 2 β max +2 , k 2 = 1 4 β max +2 , K 1 = 2 /floorleft k 1 log 2 ( T ) /floorright , K 2 = 2 /floorleft k 2 log 2 ( T ) /floorright ; 3: for i = 1 , 2 do 4: Set trial time T i = T /floorleft 1 2 + k i /floorright ; 5: Pull arms T i times from U ( p min , 1) independently; 6: for m = 1 , 2 , · · · , K i do 7: Let the samples which fall in [ p min + ( m -1)(1 -p min ) K i , p min + m (1 -p min ) K i ] be O i,m = { ( p t , d t ) : p t ∈ [ p min + ( m -1)(1 -p min ) K i , p min + m (1 -p min ) K i ] } ; 8: Fit local polynomial regression on [ p min + ( m -1)(1 -p min ) K i , p min + m (1 -p min ) K i ] with O i,m , construct estimate ˆ f i ( p ) on the interval; 9: end for 10: end for 11: Let ˆ β = -ln(max ‖ ˆ f 2 -ˆ f 1 ‖ ∞ ) ln( T ) -ln(ln( T )) ln( T ) ; 12: Set N = /ceilingleft T 1 2 ˆ β +1 /ceilingright , D = ∪ i ∪ m O i,m ; 13: Call HSDP ( T -T 1 -T 2 , ˆ β, p min , d max , N, ∆ , L, D )

We provide some intuition behind the design of Algorithm 2. Harnessing the H¨ older smoothness assumption, we can employ local polynomial regression to estimate the demand function reasonably well. However, the

demand function is not easily approximated by polynomials, due to the inherent self-similarity condition presented by the dual nature of H¨ older smoothness. The estimation granularity refers to the number of intervals into which the domain of price is partitioned for better piecewise polynomial approximation.

In Algorithm 2, we employ two distinct levels of granularity to estimate the demand function, indexed by 1 and 2 respectively. For estimation i ∈ { 1 , 2 } , the price range is segmented into small intervals of size (1 -p min ) /K i , where K i is a quantity depending on T . The algorithm then allocates T i time periods to collect price and demand data points and fit with local polynomial regression. The constructed estimates of demand function f are ˆ f 1 and ˆ f 2 , which helps us to establish an estimate of the H¨ older smoothness parameter as defined in (1).

<!-- formula-not-decoded -->

The estimator ˆ β is then fed into the H¨ older-smooth dynamic pricing algorithm (Algorithm 1) for the remaining time horizon T -T 1 -T 2 . As evident from the algorithm design, the accuracy of ˆ β estimation is critical to the regret bound of our smoothness-adaptive dynamic pricing algorithm in Algorithm 2. In the next subsection, we provide a tight confidence interval for the estimator ˆ β , which plays a central role in our final regret analysis of the SADP algorithm.

## 5.2 Accuracy of Estimation

By employing two distinct levels of granularity to estimate the demand function and, in conjunction with the previously established upper and lower bounds of the approximation error, we can prove a confidence interval for the distance between the two estimations ‖ ˆ f 2 -ˆ f 1 ‖ ∞ . This distance is directly related to the H¨ older smoothness parameter β . Ultimately, we arrive at a reasonably narrow confidence interval for β , which converges rapidly as T increases. Formally, we have the following theorem, which plays a key role in establishing the regret bound.

Theorem 3. With an upper bound β max of the smoothness parameter, under the assumptions and settings in the Algorithm 2, for some constant C &gt; 0 , with probability at least 1 -O ( e -C ln 2 ( T ) ) ,

<!-- formula-not-decoded -->

Theorem 3 demonstrates the effectiveness of our proposed SADP algorithm in estimating the H¨ older smoothness parameter β without prior knowledge. This adaptability, along with the effective smoothness parameter selection phase, enables our algorithm to construct a tight confidence interval for the H¨ older smoothness parameter β and achieve a high convergence rate. These characteristics contribute to the desired regret bound in dynamic pricing scenarios, opening up possibilities for the development of more robust and adaptive dynamic pricing algorithms.

In order to prove Theorem 3, we firstly introduce a lemma to characterize the convergence on ˆ f .

Lemma 1. Let { p ( i ) , i = 1 , 2 , · · · , n } be an i.i.d. uniform sample in an interval I = [ a, b ] ⊂ [ p min , 1] , and O I = {( p (1) , d (1) ) , . . . , ( p ( n ) , d ( n ) )} . With the assumptions, suppose sub-gaussian parameter u 1 ≤ exp( u ′ 1 · n v ) for some positive constants ν, u ′ 1 , polynomial degree l ≥ w ( β ) . Let

∣ ∣ Then, there exist positive constants C 1 , C 2 such that with probability at least 1 -O ( e -C 2 ln 2 ( n ) ) , for any p ∈ I and n &gt; C 1 , the following inequality holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Also,

<!-- formula-not-decoded -->

With Lemma 1 in place, we can now proceed to prove Theorem 3. The proof for this theorem relies on the concentration result stated in Lemma 1 and the construction of the confidence interval based on the two-level granularity approach. By analyzing the relationship between the distance of the estimated demand functions and the H¨ older smoothness parameter β , we can show that the estimated ˆ β falls within the stated confidence interval with high probability.

We also present a lemma to mitigate the issue of insufficient sample points falling within certain intervals.

Lemma 2. Let { B i : i = 1 , 2 , · · · , n } be i.i.d random variables. Suppose B 1 ∼ Bernoulli ( 1 m ) . Let ¯ B = ∑ n i =1 B i n . Then

<!-- formula-not-decoded -->

Proof Sketch of Theorem 3. Our first objective is to ascertain an upper bound for the distance between ˆ f 1 and ˆ f 2 . Define the interval I i,m = [ p min + ( m -1)(1 -p min ) K i , p min + m (1 -p min ) K i ) . Invoking the first part of Lemma 1 and corroborated by Lemma 2, with a probability of at least 1 -O ( e -C ln 2 ( n ) ) , we have ∀ p ∈ I i,m , for some sufficiently small constants v 1 , v 2 .

<!-- formula-not-decoded -->

Subsequently, utilizing inequality (2), we deduce the following upper bound

<!-- formula-not-decoded -->

for a small constant c .

However, to prove the theorem, it's necessary to establish a lower bound for the distance between and ˆ f 1 and ˆ f 2 .

Firstly, using inequality (2), we can establish an upper bound for the distance between f and ˆ f 1 . Furthermore, by invoking the second part of Lemma 1 as well as Lemma 2, with probability at least 1 -O ( e -C ln 2 ( n ) ) , ∀ p ∈ I i,m , where v 1 , v 2 are sufficiently small.

<!-- formula-not-decoded -->

Given the self-similar properties of f , a lower bound for the distance between f and Γ l f is established:

<!-- formula-not-decoded -->

Subsequently, combining inequalities (2), (3), and (4), a lower bound can be derived as

<!-- formula-not-decoded -->

To advance the proof, we employ the probability union bound. For some constant C &gt; 0, with probability

at least 1 -O ( e -C ln 2 ( T ) ), the following holds:

<!-- formula-not-decoded -->

which completes the proof.

## 5.3 Regret Analysis

After obtaining an estimation of β , we can provide a more precise bound for the distance between the local polynomial projection and the demand function.

Theorem 4. Suppose f ∈ H ( β, L ) , SADP has an estimation of β with ˆ β , and is run with N ≥ /ceilingleft T 1 2 ˆ β +1 /ceilingright , ∆ ≤ L ( 1 -p min N ) ˆ β , then with probability 1 -O ( T -1 ) the cumulative regret of SADP is upper bounded by ˜ O ( T β +1 2 β +1 ) .

We introduce Lemma 3 to bound the regret rate of our non-adaptive algorithm HSDP , whose proof is in Appendix C.2.

Lemma 3. If HSDP is run with ˆ β ≤ β and other conditions stays the same as Theorem 4, then with probability 1 -O ( T -1 ) , the cumulative regret is upper bounded by O ( T ˆ β +1 2 ˆ β +1 ) .

Proof Sketch of Theorem 4. Considering the event A ∗ : { ˆ β ∈ [ β -4( β max +1)ln(ln( T )) ln( T ) , β ] } , and by Theorem 3, we know that P ( A ∗ ) ≥ 1 -O ( e -C ln 2 ( T ) ). Under event A ∗ , ˆ β converges to β with rate O ( ln(ln( T )) ln( T ) ), we have O ( T ˆ β +1 2 ˆ β +1 -T β +1 2 β +1 ) ≤ O ( T β +1 2 β +1 · T β -ˆ β (2 ˆ β +1)( β +1) ) ≤ ˜ O ( T β +1 2 β +1 ). Considering Lemma 3, we can derive the regret bound for HSDP under event A ∗ :

<!-- formula-not-decoded -->

For the adaptive part of SADP , note that T 1 , T 2 ≤ T β +1 2 β +1 , which means that the regret is bounded by O ( T β +1 2 β +1 ) . Applying the union bound with event A ∗ , we can derive that with probability 1 -O ( T -1 ) ,

<!-- formula-not-decoded -->

Theorem 4 highlights the effectiveness of the SADP algorithm in achieving the desired regret bound under the specified conditions. By estimating the H¨ older smoothness parameter β and generalizing the nonadaptive dynamic-pricing algorithm with non-integer H¨ older smoothness parameter, our algorithm is capable of maintaining a high level of performance in dynamic pricing scenarios.

## 6 CONCLUSION

Motivated by the challenge of unknown smoothness levels in applications, we develop a smoothness-adaptive dynamic pricing algorithm under self-similarity conditions. To make dynamic pricing algorithms, it is very

desirable to remove the parameter dependence of algorithms. Moving forward, it is promising to explore whether our approach can be generalized to other dynamic pricing problems such as feature-based dynamic pricing. To further improve adaptivity, it is of interest to consider other parameters that are implicitly used in the algorithm design.

## References

- Abbasi-Yadkori, Y., Pal, D., and Szepesvari, C. (2012). Online-to-confidence-set conversions and application to sparse stochastic bandits. In Artificial Intelligence and Statistics , pages 1-9. PMLR.
- Ban, G.-Y. and Keskin, N. B. (2021). Personalized dynamic pricing with machine learning: High-dimensional features and heterogeneous elasticity. Management Science , 67(9):5549-5568.
- Besbes, O. and Zeevi, A. (2009). Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. Operations research , 57(6):1407-1420.
- Besbes, O. and Zeevi, A. (2012). Blind network revenue management. Operations research , 60(6):1537-1550.
- Broder, J. and Rusmevichientong, P. (2012). Dynamic pricing under a general parametric choice model. Operations Research , 60(4):965-980.
- Bu, J., Simchi-Levi, D., and Wang, C. (2022). Context-based dynamic pricing with partially linear demand model. Advances in Neural Information Processing Systems , 35:23780-23791.
- Cai, T. T., Low, M. G., and Xia, Y. (2013). Adaptive confidence intervals for regression functions under shape constraints. The Annals of Statistics , 41(2):722-750.
- Cai, T. T. and Pu, H. (2022). Stochastic continuum-armed bandits with additive models: Minimax regrets and adaptive algorithm. The Annals of Statistics , 50(4):2179-2204.
- Chen, N. and Gallego, G. (2022). A primal-dual learning algorithm for personalized dynamic pricing with an inventory constraint. Mathematics of Operations Research , 47(4):2585-2613.
- Cohen, M. C., Lobel, I., and Paes Leme, R. (2020). Feature-based dynamic pricing. Management Science , 66(11):4921-4943.
- Den Boer, A. V. (2015). Dynamic pricing and learning: historical origins, current research, and new directions. Surveys in operations research and management science , 20(1):1-18.
- Fan, J., Guo, Y., and Yu, M. (2022). Policy optimization using semiparametric models for dynamic pricing. Journal of the American Statistical Association , pages 1-29.
- Gin´ e, E. and Nickl, R. (2010). Confidence bands in density estimation. Annals of statistics , 38(2):1122-1170.
- Gur, Y., Momeni, A., and Wager, S. (2022). Smoothness-adaptive contextual bandits. Operations Research , 70(6):3198-3216.
- Hadiji, H. (2019). Polynomial cost of adaptation for x-armed bandits. Advances in Neural Information Processing Systems , 32.
- Javanmard, A. (2017). Perishability of data: dynamic pricing under varying-coefficient models. The Journal of Machine Learning Research , 18(1):1714-1744.
- Keskin, N. B. and Zeevi, A. (2014). Dynamic pricing with an unknown demand model: Asymptotically optimal semi-myopic policies. Operations research , 62(5):1142-1167.
- Kleinberg, R. and Leighton, T. (2003). The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In 44th Annual IEEE Symposium on Foundations of Computer Science, 2003. Proceedings. , pages 594-605. IEEE.
- Liu, Y., Wang, Y., and Singh, A. (2021). Smooth bandit optimization: generalization to holder space. In International Conference on Artificial Intelligence and Statistics , pages 2206-2214. PMLR.
- Locatelli, A. and Carpentier, A. (2018). Adaptivity to smoothness in x-armed bandits. In Conference on Learning Theory , pages 1463-1492. PMLR.
- Picard, D. and Tribouley, K. (2000). Adaptive confidence interval for pointwise curve estimation. The Annals of Statistics , 28(1):298-335.

- Qiang, S. and Bayati, M. (2016). Dynamic pricing with demand covariates. arXiv preprint arXiv:1604.07463 . Slivkins, A. et al. (2019). Introduction to multi-armed bandits. Foundations and Trends ® in Machine Learning , 12(1-2):1-286.
- Stewart, G. W. (1977). On the perturbation of pseudo-inverses, projections and linear least squares problems. SIAM review , 19(4):634-662.
- Tropp, J. A. (2012). User-friendly tail bounds for sums of random matrices. Foundations of computational mathematics , 12:389-434.
- Wang, Y., Chen, B., and Simchi-Levi, D. (2021). Multimodal dynamic pricing. Management Science , 67(10):6136-6152.
- Xu, J. and Wang, Y.-X. (2021). Logarithmic regret in feature-based dynamic pricing. Advances in Neural Information Processing Systems , 34:13898-13910.

## Smoothness-Adaptive Dynamic Pricing with Nonparametric Demand Learning Supplementary Materials

## A PROOFS ON SELF-SIMILARITY

## A.1 Proof of Theorem 1

Proof of Theorem 1. Let d ∼ N ( f ( p ) , σ 2 ) be a normal random variable, σ 2 here is small enough such that N ( 0 , σ 2 ) is sub-Gaussian under different demand settings. Also let /epsilon1 1 = α -β 2(2 α +1)(2 β +1) and /epsilon1 2 = 2 /epsilon1 1 , /epsilon1 3 = /epsilon1 1 2 . Denote

<!-- formula-not-decoded -->

when c &lt; 1 2 is small enough, ψ ∈ H ( α, L ′ ( α )) ∩ H ( β, L ′ ( β )). Define a counting random variable Z k,m = ∑ T t =1 /BD { p t ∈ [ p min + m (1 -p min ) k , p min + ( m +1)(1 -p min ) k )} for any positive integer k. Let a ∝ T 1 2 α +1 -/epsilon1 2 α , b ∝ T 1 -/epsilon1 1 2 β +1 . Define index set S a,b = { 0 , 1 , · · · , b -1 } ∩ ( b a , + ∞ ) and let m 0 be the index in the index set such that E [ Z k,m 0 ] is the smallest. Define functions ψ a ( p ) = a -α ψ ( a ( p -p min )) and ψ b ( p ) = b -β ψ ( b ( p -p min ) -m 0 ). And let g 1 ( p ) = 1 p [ 1 2 + ψ a ( p ) ] , g 2 ( p ) = 1 p [ 1 2 + ψ a ( p ) + ψ b ( p ) ] , by Lemma 4, g 1 ∈ H ( α, L ( α )) , g 2 ∈ H ( β, L ( β )).

Denote the probability measure determined by A and f = g i by P i for i = 1 , 2. Let E i [ Z ] be the expectation of random variable Z ( p 1 , · · · , p T , d 1 , · · · , d T ) if the probability measure is P i .

If f = g 1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the conditions from the theorem, we have R T ( A ; g 1 ) ≤ O ( T α +1 2 α +1 + /epsilon1 3 ) . So we have

And by the definition of m 0 and notice that b &gt; a , we can derive that

<!-- formula-not-decoded -->

Then we can decompose the KL-divergence between P 1 , P 2 as

<!-- formula-not-decoded -->

By inequality (A.2) and Lemma 5 we can obtain

<!-- formula-not-decoded -->

Let A = { Z b,m 0 &gt; b -1 · a α · T α +1 2 α +1 + /epsilon1 3 · ln ( T ) } . Inequality A.2 implies that P 1 ( A ) = o (1). By Lemma 6, we have | P 1 ( A ) -P 2 ( A ) | = o (1), so we can derive that P 2 ( A ) = o (1).

Since

On A c , we have T -Z b,m 0 &gt; T 2 . Note that max p ∈ [ p min ,p min + 1 -p min a ) ψ a ( p ) /lessmuch max p ∈ [ p min + m 0 ( 1 -p min ) b ,p min + ( m 0 +1) ( 1 -p min ) b ) ψ b ( p ), then with the similar procedure in inequality A.1, if f = g 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which completes the proof.

## A.2 Technical Lemmas for Theorem 1

Lemma 4. Suppose r ( p ) ∈ H ( β, L ′ ) , p ∈ [ p min , 1] , 0 &lt; p min &lt; 1 , then f ( p ) = r ( p ) p ∈ H ( β, L ) for some constant L.

Proof of Lemma 4. This is a basic property of H¨ older class of functions, whose proof follows directly from Lemma 1.10 of Gur et al. (2022) and the fact that the function p ↦→ 1 /p is H¨ older smooth of any levels when restricted to the interval [ p min , 1].

Then we introduce three lemmas about KL-divergence which are standard results in the literature and therefore we omit the proofs.

Lemma 5. Let d 1 ∼ N ( µ 1 , σ 2 ) , d 2 ∼ N ( µ 2 , σ 2 ) , then the KL-divergence between d 1 and d 2 is ( µ 1 -µ 2 ) 2 2 σ 2 .

Lemma 6. Let P 1 , P 2 be two probability measures on the same σ -algebra, then for any event A on this σ -algebra, we have

<!-- formula-not-decoded -->

Lemma 7. For P 1 , P 2 defined as above, in terms of the total variation norm ‖·‖ TV , we have

<!-- formula-not-decoded -->

## A.3 Proof of Proposition 1 and Lemma 8 - 9

The results in this section will be used to prove Theorem 2.

## A.3.1 Proof of Proposition 1

Proposition 1. For each L, β &gt; 0 , there exists a function g : [0 , 1] → [0 , 1] satisfying that g ∈ H ( β, L ) ∩ S ( β, w ( β ) , 0 , M 2 ) for some constant M 2 &gt; 0 ; and g has a unique maximizer at 1 2 ; and for any x ∈ { 0 , 1 } , k ∈ { 0 , 1 , · · · , w ( β ) } it holds that g ( k ) ( x ) = 0 .

Proof. Let u ( · ) : [0 , 1] → R be a C ∞ function with u (0) = 1 , u (1) = 0 , u ( k ) (0) = 0 , u ( k ) (1) = 0 , ∀ k ∈ Z + . Let u 1 ( x ) = sin 2 ( π 2 · u (4 x -1) ) . ∀ x ∈ [0 , 1], let σ ( x ) = | x -1 2 | and define for some sufficiently small constant c 1 &gt; 0, then we have g ∈ H ( β, L ), and following the proof procedure as Lemma 1.7 of Gur et al. (2022), for some constant M 2 &gt; 0, g ∈ S ( β, w ( β ) , 0 , M 2 ). Also, g has a unique maximum point at x = 1 2 and for any x ∈ { 0 , 1 } , l ∈ { 0 , 1 , · · · , w ( β ) } it holds that g ( l ) ( x ) = 0.

<!-- formula-not-decoded -->

## A.3.2 Proof of Lemma 8

Lemma 8. The worst-case regret of algorithm A over time period T can be lower bounded as

<!-- formula-not-decoded -->

where f ∈ S ( β, w ( β ) , 0 , M 2 ) on [ p min , 1] for some constant M 2 .

Proof. For ∀ j ∈ { 1 , 2 , · · · , J } , which completes the proof.

<!-- formula-not-decoded -->

## A.3.3 Proof of Lemma 9

Lemma 9. For fixed j ∈ { 1 , 2 , · · · , J } , we have

<!-- formula-not-decoded -->

Proof. Because f 0 and f j only differs on I j , we have that

Then by Lemma 6, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and by Lemma 7, inequalities A.3,A.4 we have

Subsequently,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which completes the proof.

## A.4 Proof of Theorem 2

Proof of Theorem 2. Let β denote the true H¨ older smoothness parameter here.

Firstly, we introduce a proposition constructing the reward function we need.

Define number of intervals J = /ceilingleft /epsilon1 -1 T /ceilingright for /epsilon1 T defined as cT -1 2 β +1 where c is some sufficiently small constant depending only on β . Let I j = [ a j , b j ] , a j = p min + ( j -1)(1 -p min ) J , b j = p min + j (1 -p min ) J , for j = 1 , 2 , · · · J . Then define J different demand functions f 1 , f 2 , · · · , f J , let where f j ∈ H ( β, L ) ∩S ( β, w ( β ) , 0 , M 2 ). Define also f 0 ( p ) ≡ 1 2 p , p ∈ [ p min , 1].

<!-- formula-not-decoded -->

Denote the probability measure determined by algorithm A and f = f j by P j for j = 0 , 1 , · · · , J . Let E i [ Z ] be the expectation of random variable Z ( p 1 , · · · , p T , d 1 , · · · , d T ) if the probability measure is P j . Let demand d i ∼ N ( f ( p i ) , σ 2 ) , where σ 2 is small enough that N ( f ( p i ) , σ 2 ) is sub-Gaussian with parameters u 1 , u 2 .

Then we can upper bound the difference between E 0 [ T j ] and E j [ T j ] by the properties of KL-divergence.

Let j ∗ = arg min j ∈{ 1 , 2 , ··· ,J } E 0 [ T j ], it is obvious that E 0 [ T j ] ≤ T J ≤ T/epsilon1 T . Then by Lemma 9, for some sufficiently small c , we can obtain

Consequently, by Lemma 8

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which completes the proof.

## B PROOF OF CONSTRUCTED CONFIDENCE INTERVAL

## B.1 Proof of Lemma 1

To prove Lemma 1, we first introduce the following two lemmas.

Lemma 10. Suppose A,B are two n 0 × n 0 symmetric matrices. If ‖ A -B ‖ ≤ λ min ( B ) 2 , then

The proof of Lemma 1 directly follows Theorem 3.3 of Stewart (1977).

<!-- formula-not-decoded -->

Lemma 11. Suppose d (1) is a sub-Gaussian random variable with parameter u 1 , u 2 , then we can upper bound E [ | d (1) | ] with ( 2 √ ln(2 u 1 )+1 ) √ u 2 .

<!-- formula-not-decoded -->

Proof of Lemma 1. Without loss of generality, in this part, we do a translation for d (1) | p (1) to make its expectation 0. Let P n be a n × ( l +1) matrix with its m th row φ ( l ) ( p ( m ) ) T for every m and d n = ( d (1) , . . . , d ( n ) ) T . By least square regression, we obtain

Define

<!-- formula-not-decoded -->

The goal of this lemma is to obtain the convergence properties of θ , here we firstly prove the convergence of ( P T n P n n ) -1 . Let U 1 = φ ( l ) ( p (1) ) φ ( l ) ( p (1) ) T -E [ φ ( l ) ( p (1) ) φ ( l ) ( p (1) ) T ] , by Bernstein inequality(Tropp (2012), Theorem 1.6),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where R 1 = max { ∥ ∥ E [ U 1 · U T 1 ]∥ ∥ , ∥ ∥ E [ U T 1 · U 1 ]∥ ∥ } , R 2 = sup ‖ U 1 ‖ . For each m, we have t m ( p (1) ) ∈ [0 , 1] with probability 1, so R 1 , R 2 ≤ O (1). Let w = ln( n ) √ n , then we have

for some constant C &gt; 0.

<!-- formula-not-decoded -->

Let V ( p (1) ) = E [ φ ( l ) ( p (1) ) φ ( l ) ( p (1) ) T ] , we can prove there exists a constant M 0 which satisfies λ min ( V ( p (1) )) ≥ M 0 &gt; 0 where λ min denotes the least eigenvalue of the matrix. We have

Based on Lemma 10, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we prove the convergence of P T n d n n . Let 1 ( | d n | &gt; M ) be a n-dimensional vector whose m th element is 1 ( | d ( m ) | &gt; M ) , F M ( d n ) be a n-dimensional vector whose m th element is d ( m ) 1 ( | d ( m ) | ≤ M ) . Let U 2 = φ ( l ) ( p (1) ) d (1) · 1 ( | d (1) | ≤ M ) -E [ φ ( l ) ( p (1) ) d (1) · 1 ( | d (1) | ≤ M )] , and by Bernstein Inequality(Tropp (2012),Theorem 1.6), where R 3 = max { ∥ ∥ E [ U 2 · U T 2 ]∥ ∥ , ∥ ∥ E [ U T 2 · U 2 ]∥ ∥ } , R 4 = sup ‖ U 2 ‖ . Each element of U 2 is upper bounded by O ( M ) so we have R 3 ≤ O ( M 2 ) , R 4 ≤ O ( M ). Let w = M ln( n ) √ n , then we have

for some constant C &gt; 0. And by the sub-gaussian assumption, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking M = √ ln( u 1 n ) u 2 ln ( n ), we have and C is a constant that independent of u 1 , u 2 . And also by the we can deduce that

<!-- formula-not-decoded -->

which leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the three inequalities B.2, B.3 and B.4, we have the following where C, C 0 , C 2 are constants depending on u 2 and l .

<!-- formula-not-decoded -->

And by Lemma 11, we know that

<!-- formula-not-decoded -->

Then combining inequalities B.1, B.5 and B.6, recall the definition of θ 0 and ˆ θ , with probability at least 1 -O ( e -C 2 ln 2 ( n ) ) for some constants C 2 depending on u 2 and l , we have

<!-- formula-not-decoded -->

for n larger than some constant C 1 depending on u ′ 1 , u 2 , l .

In order to prove the first part of the lemma, we can show that | E [ d (1) | p (1) = p ] - 〈 t ( p (1) ) , θ 0 〉| = O ( ( b -a ) β ) . By the Holder assumption and taylor expansion, there exists an l + 1 dimensional vector θ 1 such that | E [ E [ d (1) | p (1) = p ]] -〈 φ ( l ) ( p (1) ) , θ 1 〉| = O ( ( b -a ) β ) , ∀ p ∈ I . So we have

Note that Γ I l { E [ d (1) | p (1) = p ] } = 〈 φ ( l ) ( p ) , θ 0 〉 and ˆ f ( p ; O , l, I ) = 〈 φ ( l ) ( p ) , ˆ θ 〉 , with φ ( l ) ( p (1) ) ≤ O (1), the second part of the lemma is proved.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then by inequality (B.7), we can deduce that with probability at least 1 -O ( e -C 2 ln 2 ( n ) )

Note that with φ ( l ) ( p (1) ) ≤ O (1), ∥ ∥ ∥ 〈 φ ( l ) ( p ) , ˆ θ 〉 - 〈 φ ( l ) ( p ) , θ 1 〉 ∥ ∥ ∥ ≤ O (∥ ∥ ∥ ˆ θ -θ 1 ∥ ∥ ∥ ) , ∀ p ∈ I , n &gt; C 1 , with probability at least 1 -O ( e -C 2 ln 2 ( n ) ) , the following inequality holds

Therefore the first part of the lemma is proved.

<!-- formula-not-decoded -->

## B.2 Proof of Theorem 3

Proof. We first define an event A = {∃ i ∈ { 1 , 2 } , m ∈ { 1 , 2 , . . . , K i } , s.t. | O i,m | &lt; T i 2 K i } , by Lemma 2, we have

<!-- formula-not-decoded -->

By conditioning on A c , we can guarantee the number of samples in each interval. Next, we aim to establish an upper bound for the distance between the distance between ˆ f 1 and ˆ f 2 . Let I i,m = [ p min + ( m -1)(1 -p min ) K i , p min + m (1 -p min ) K i ) . Invoking the first part of Lemma 1, with probability at least 1 -O ( e -C ln 2 ( n ) ) , ∀ p ∈ I i,m , the following inequality holds:

for some sufficiently small constants v 1 , v 2 .

Define the event

<!-- formula-not-decoded -->

Applying the union bound, we find that for some constant C &gt; 0.

<!-- formula-not-decoded -->

Then, conditioning on A c ∩ B c , we can, by inequality 2, derive an upper bound as follows:

However, to prove the theorem, it's necessary to establish a lower bound for the distance between and ˆ f 1 and ˆ f 2 . In the ensuing discussion, we aim to provide this lower bound.

<!-- formula-not-decoded -->

Firstly, using inequality 2, we can establish an upper bound for the distance between f and ˆ f 1 . This, in turn, aids in deducing an upper bound for the distance between Γ l f and ˆ f 2 .

Furthermore, by invoking the second part of Lemma 1 as well as Lemma 2, with probability at least 1 -O ( e -C ln 2 ( n ) ) , ∀ p ∈ I i,m , where v 1 , v 2 are sufficiently small.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define the event

<!-- formula-not-decoded -->

And applying the union bound we have

<!-- formula-not-decoded -->

Given the self-similar property of f , we can establish a lower bound for the distance between f and Γ l f .

<!-- formula-not-decoded -->

Subsequently, conditioning on A c ∩ B c ∩ D c , and using inequalities 2, 3, and (4), we can derive a lower bound as follows:

<!-- formula-not-decoded -->

Now, let's attempt to simplify the upper bound in inequality B.10:

<!-- formula-not-decoded -->

for some small constant c .

Similarly, we can simplify the lower bound in inequality B.13:

<!-- formula-not-decoded -->

Thus, on the event A c ∩ B c ∩ D c , we have

<!-- formula-not-decoded -->

Simultaneously we have for some constant C &gt; 0 which completes the proof.

<!-- formula-not-decoded -->

## C PROOF OF REGRET UPPER BOUNDS

## C.1 Proof of Lemma 12

Lemma 12. Suppose f ∈ H ( β, L ) and let I = [ a, b ] ⊂ [ p min , 1] be an arbitrary interval whose length is | I | . For estimation ˆ β ≤ β , there exists a polynomial with degree k = w ( ˆ β ) : P I ( p ) = Γ I k f ( p ) = ∑ k m =0 a m ( 1 2 + p -a + b 2 b -a ) m satisfying | a m | m ! ≤ L, ∀ m ≤ k , such that

<!-- formula-not-decoded -->

Proof. Firstly, let a m = f ( m ) ( a ) m ! ( b -a ) m

<!-- formula-not-decoded -->

By Taylor expansion with Lagrangian remainders, ∀ p ∈ I , ∃ ˜ p ∈ I such that

<!-- formula-not-decoded -->

With ˆ β ≤ β , we then have that

<!-- formula-not-decoded -->

## C.2 Proof of Lemma 3

Proof. We first state two important lemmas, Lemma 13 and Lemma 14, whose proofs will be included later in this section.

Lemma 13. Suppose f ∈ H ( β, L ) and let I = [ a, b ] ⊂ [ p min , 1] . If HSDP is invoked with ∆ ≥ L ( b -a ) ˆ β and outputs ˆ p , then with probability 1 -δ it holds that where γ = L √ k +1+∆ √ |D| + d max √ 2 ( k +1)ln ( 4( k +1) t δ ) +2 .

<!-- formula-not-decoded -->

Then we can use Lemma 13 to prove Lemma 14.

Lemma 14. Keep the same setting in Lemma 13 and let ˆ p 1 , · · · , ˆ p t be the output prices of t consecutive calls on I . Then with probability 1 -O ( T -1 ) it holds that

<!-- formula-not-decoded -->

Finally, with Lemma 14 proved, we can do the UCB analysis for the Theorem.

For all 1 ≤ j ≤ N , we define r ∗ ( I j ) = max p ∈ I j pf ( p ). Invoking Lemma 14, by concentration inequalities, with probability 1 -O ( T -1 ) , it holds uniformly for all j that

<!-- formula-not-decoded -->

Let T j be the total number of time periods that we invoke HSDP in the j th interval, and we invoke j t th interval at time t . Also, denote j ∗ = arg max j r ∗ ( I j ). Note that ˆ β is strictly less than β , then we can still derive a bound for the regret each round using UCB analysis

<!-- formula-not-decoded -->

And

<!-- formula-not-decoded -->

## C.3 Proof of Lemma 13 and 14

## C.3.1 Proof of Lemma 13

Proof. The ( p, d ) pairs in the history are labeled as { ( p i , d i ) } t i =1 in chronological order. And we can show that d i = f ( p i ) + ξ i = P I ( p i ) + ξ i + β i , where { ξ i } t i =1 are i.i.d sub-gaussian random variables with zero mean and | β i | ≤ ∆ with probability 1. Use vectors and matrices to denote them we have p = ( p i ) t i =1 , d = ( d i ) t i =1 , ξ = ( ξ i ) t i =1 , β = ( β i ) t i =1 and P = ( φ ( k ) ( p i ) T ) t i =1 ∈ R t × ( k +1) . And the ridge estimator ˆ θ can be written as ˆ θ = Λ -1 P T d = ( P T P + I ) -1 P T d , plug in d = P θ ∗ + ξ + β with θ ∗ is the real coefficient of the expansion, we have

<!-- formula-not-decoded -->

Multiplying ( ˆ θ -θ ∗ ) Λ on both sides and it leads to

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plug the above inequality into equation (C.1), then dividing √ ( ˆ θ -θ ∗ ) T Λ ( ˆ θ -θ ∗ ) from both sides, also noting that Λ /followsequal I which makes √ ( ˆ θ -θ ∗ ) T Λ ( ˆ θ -θ ∗ ) ≥ ∥ ∥ ∥ ˆ θ -θ ∗ ∥ ∥ ∥ 2 , we obtain where

Recall the definition of θ ∗ and the H¨ older class assumption, we have ‖ θ ∗ ‖ 2 ≤ L √ k +1. In order to bound G t ( z ), we introduce and prove Lemma 15:

<!-- formula-not-decoded -->

Lemma 15. Fix k, t and a probability δ , with probability 1 -δ it holds uniformly for all Λ defined above that

<!-- formula-not-decoded -->

Proof of Lemma 15. By definition we have ∥ ∥ φ ( k ) ( p i ) ∥ ∥ 2 ≤ √ k +1. Let /epsilon1 &gt; 0 be a small parameter. Denote ‖·‖ Λ = √ ( · ) Λ ( · ) as the Λ-norm of a vector, and B ( r, ‖·‖ ) = { z ∈ R k +1 : ‖ z ‖ ≤ r } as a ball. Let U ⊆ B (1 , ‖·‖ 2 ) be a /epsilon1 -covering of B (1 , ‖·‖ 2 ) which means that sup z ∈ B ( 1 , ‖·‖ 2 ) min z ′ ∈U ‖ z -z ′ ‖ 2 ≤ /epsilon1 . Fix arbitrary z ∈ U , for | ξ i | ≤ d max with probability 1, by Hoffeding's inequality we know that for any δ ∈ (0 , 1),

<!-- formula-not-decoded -->

Since Λ /followsequal I , we know that Φ Λ = B (1 , ‖·‖ Λ ) ⊆ B (1 , ‖·‖ 2 ) and therefore U is also a /epsilon1 -covering of Φ Λ , and it is easy to verify that there exists U with ln ( |U| ) ≤ ( k +1)ln ( 2 /epsilon1 ) . Applying union bound we have with probability 1 -δ ,

Considering the covering, by ∥ ∥ φ ( k ) ( p i ) ∥ ∥ 2 ≤ √ k +1, we have that ‖ Λ ‖ op ≤ 1 + ( k +1) t ≤ 2 ( k +1) t , then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By setting /epsilon1 = 1 ( k +1) t we complete the proof of Lemma 15.

Then back to the proof of Lemma 13. With inequality C.3, invoking Lemma 15, with probability 1 -δ we have

And ∀ p ∈ I , let ˆ f ( p ) = 〈 φ ( k ) ( p ) , ˆ θ 〉 , we can obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The upper bound ¯ f ( p ) = min { d max , 〈 φ ( k ) ( p ) , ˆ θ 〉 + γ √ φ ( k ) ( p ) T Λ -1 φ ( k ) ( p ) + ∆ } . We can infer from the above analysis that with probability 1 -δ , ¯ f ( p ) ≥ f ( p ) , ∀ p ∈ I . So max p ∈ I pf ( p ) -ˆ pf (ˆ p ) ≤ ˆ p | ¯ f (ˆ p ) -f ( p ) | ≤ 2 min { d max , γ √ φ ( k ) ( p ) T Λ -1 φ ( k ) ( p ) + ∆ } which completes the proof.

## C.3.2 Proof of Lemma 14

Proof. Invoke Lemma 13 with δ = 1 T 2 and let Λ i = I + ∑ i ′ &lt;i φ ( k ) (ˆ p i ′ ) φ ( k ) (ˆ p i ′ ) T denote the Λ matrix at the i th call. Denote γ max = max i ≤ t γ i , and we can easily verify γ max ≤ L √ k +1 + ∆ √ t + d max √ 6 ( k +1)ln(( k +1) T ). Recalling the right side of Lemma 13, and noting that γ max ≥ d max we have

Using the elliptical potential lemma (Abbasi-Yadkori et al. (2012), Lemma 11), we know that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Subsequently,

<!-- formula-not-decoded -->

So

<!-- formula-not-decoded -->