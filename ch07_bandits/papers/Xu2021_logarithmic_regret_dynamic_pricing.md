## Logarithmic Regret in Feature-based Dynamic Pricing

## Jianyu Xu

Department of Computer Science University of California, Santa Barbara Santa Barbara, CA 93106 xu\_jy15@ucsb.edu

## Yu-Xiang Wang

Department of Computer Science University of California, Santa Barbara Santa Barbara, CA 93106 yuxiangw@cs.ucsb.edu

## Abstract

Feature-based dynamic pricing is an increasingly popular model of setting prices for highly differentiated products with applications in digital marketing, online sales, real estate and so on. The problem was formally studied as an online learning problem [Javanmard and Nazerzadeh, 2019] where a seller needs to propose prices on the fly for a sequence of T products based on their features x while having a small regret relative to the best -'omniscient'- pricing strategy she could have come up with in hindsight. We revisit this problem and provide two algorithms (EMLP and ONSP) for stochastic and adversarial feature settings, respectively, and prove the optimal O ( d log T ) regret bounds for both. In comparison, the best existing results are O ( min { 1 λ 2 min log T, √ T }) and O ( T 2 / 3 ) respectively, with λ min being the smallest eigenvalue of E [ xx T ] that could be arbitrarily close to 0 . We also prove an Ω ( √ T ) information-theoretic lower bound for a slightly more general setting, which demonstrates that 'knowing-the-demand-curve' leads to an exponential improvement in feature-based dynamic pricing.

Keywords : dynamic pricing, online learning, adversarial features, optimal regret, affine invariant, distribution-free.

## Contents

| 1 Introduction    | 1 Introduction                         | 1 Introduction                                                                      | 3     |
|-------------------|----------------------------------------|-------------------------------------------------------------------------------------|-------|
| 2 Related Works   | 2 Related Works                        | 2 Related Works                                                                     | 4     |
| 3 Problem Setup   | 3 Problem Setup                        | 3 Problem Setup                                                                     | 5     |
| 4 Algorithms      | 4 Algorithms                           | 4 Algorithms                                                                        | 6     |
|                   | 4.1                                    | Pricing with Distribution-Free Stochastic Features                                  | 6     |
|                   | 4.2                                    | Pricing with Adversarial Features . . . . . . . .                                   | 7     |
| 5 Regret Analysis | 5 Regret Analysis                      | 5 Regret Analysis                                                                   | 7     |
|                   | 5.1                                    | O ( d log T ) Regret of EMLP . . . . . . . . . . .                                  | 7     |
|                   | 5.2                                    | O ( d log T ) Regret of ONSP . . . . . . . . . . .                                  | 9     |
|                   | 5.3                                    | Lower Bound for Unknown Distribution . . . . .                                      | 9     |
| 6                 | Numerical Result                       | Numerical Result                                                                    | 10    |
| 7                 | Discussion                             | Discussion                                                                          | 11    |
| 8                 | Conclusion                             | Conclusion                                                                          | 11    |
| A                 | Other related works                    | Other related works                                                                 | 14    |
| A.1               | History of Pricing . . . . .           | . . . . . . . . . . . .                                                             | 14    |
| A.2               | Dynamic Pricing as Bandits . . . . . . | . . . . . .                                                                         | 14    |
| A.3               | Structural Model . . . .               | . . . . . . . . . . . . .                                                           | 15    |
| B                 | Proofs                                 | Proofs                                                                              | 15    |
| B.1               | Proof of Lemma 2 . . .                 | . . . . . . . . . . . . . .                                                         | 15    |
| B.2               | Proofs in Section 5.1 . . . B.2.1      | . . . . . . . . . . . .                                                             | 15    |
|                   |                                        | Proof of Lemma 5 . . . . . . . . . . . .                                            | 15    |
|                   |                                        | B.2.2 Proof of Lemma 7 . . . . . . . . . . . .                                      | 16    |
|                   | B.2.3                                  | Proof of Lemma 9 . . . . . . . . . . . .                                            | 16    |
| B.3               | Proof of Lower More Discussions        | bound in Section 5.3 . . . . . . .                                                  | 20    |
| C                 |                                        |                                                                                     | 25    |
| C.1               | Dependence on B and Noise Variance     | . . . . . .                                                                         | 25    |
| C.2               | Algorithmic Design . . . . . .         | . . . . . . . . . . . . . . .                                                       | 28 28 |
|                   | C.2.1                                  | Probit and Logistic Regressions                                                     |       |
|                   | C.2.2 C.2.3                            | Advantages of EMLP over ONSP. . . . . Agnostic Dynamic Pricing: Explorations versus | 28 28 |
| C.3               | Problem Modeling                       | . . . . . . . . . . . . . . . .                                                     | 29    |
|                   | C.3.1                                  | Noise Distributions . . . . . . . . . . . .                                         | 29    |
|                   | C.3.2                                  | Linear Valuations on Features . . . . . . .                                         | 29    |
| C.4               | Ex Ante v.s. Ex Post Regrets           | . . . . . . . . . .                                                                 | 29    |
| C.5               |                                        | .                                                                                   |       |
|                   | Ethic Issues . . . . . . . .           | . . . . . . . . . . .                                                               | 30    |

## 1 Introduction

The problem of pricing - to find a high-and-acceptable price - has been studied since Cournot [1897]. In order to locate the optimal price that maximizes the revenue, a firm may adjust their prices of products frequently, which inspires the dynamic pricing problem. Existing works [Kleinberg and Leighton, 2003, Broder and Rusmevichientong, 2012, Chen and Farias, 2013, Besbes and Zeevi, 2015] primarily focus on pricing a single product, which usually will not work well in another setting when thousands of new products are being listed every day with no prior experience in selling them. Therefore, we seek methods that approach an acceptable-and-profitable price with only observations on this single product and some historical selling records of other products.

In this work, we consider a 'feature-based dynamic pricing' problem, which was studied by Amin et al. [2014], Cohen et al. [2020], Javanmard and Nazerzadeh [2019]. In this problem setting, a sales session (product, customer and other environmental variables) is described by a feature vector, and the customer's expected valuation is modeled as a linear function of this feature vector.

Feature-based dynamic pricing. For t = 1 , 2 , ..., T :

1. A feature vector x t ∈ R d is revealed that describes a sales session (product, customer and context).
2. The customer valuates the product as w t = x glyph[latticetop] t θ ∗ + N t .
3. The seller proposes a price v t &gt; 0 concurrently (according to x t and historical sales records).

4. The transaction is successful if v t ≤ w t , i.e., the seller gets a reward (payment) of r t = v t · ✶ ( v t ≤ w t ) . Here T is unknown to the seller (and thus can go to infinity), x t 's can be either stochastic (e.g., each sales session is drawn i.i.d.) or adversarial (e.g., the sessions arrive in a strategic sequence), θ ∗ ∈ R d is a fixed parameter for all time periods, N t is a zero-mean noise, and ✶ t = ✶ ( v t ≤ w t ) is an indicator that equals 1 if v t ≤ w t and 0 otherwise. In this online-fashioned setting, we only see and sell one product at each time. Also, the feedback is Boolean Censored , which means we can only observe ✶ t instead of knowing w t directly. The best pricing policy for this problem is the one that maximizes the expected reward, and the regret of a pricing policy is accordingly defined as the difference of expected rewards between this selected policy and the best policy.

## Summary of Results. Our contributions are threefold.

1. When x t 's are independently and identically distributed (i.i.d.) from an unknown distribution, we propose an 'Epoch-based Max-Likelihood Pricing (EMLP)' algorithm that guarantees a regret bound at O ( d log T ) . The design of EMLP is similar to that of the RMLP algorithm in Javanmard and Nazerzadeh [2019], but our new analysis improves their regret bound at O ( √ T ) when E [ xx glyph[latticetop] ] is near singular.
2. When x t 's are adversarial, we propose an 'Online-Newton-Step Pricing (ONSP)' algorithm that achieves O ( d log T ) regret on constant-level noises for the first time, which exponentially improves the best existing result of O ( T 2 / 3 ) [Cohen et al., 2020]. 1
3. Our methods that achieve logarithmic regret require knowing the exact distribution of N t in advance, as is also assumed in Javanmard and Nazerzadeh [2019]. We prove an Ω ( √ T ) lower bound on the regret if N t ∼ N (0 , σ 2 ) where σ is unknown , even with θ ∗ given and x t fixed for all t .

The O (log T ) regret of EMLP and ONSP meets the information-theoretical lower bound [Theorem 5, Javanmard and Nazerzadeh, 2019]. In fact, the bound is optimal even when w t is revealed to the learner [Mourtada, 2019]. From the perspective of characterizing the hardness of dynamic pricing problems, we generalize the classical results on 'The Value of Knowing a Demand Curve' [Kleinberg and Leighton, 2003] by further dividing the random-valuation class with an exponential separation of: (1) O (log T ) regret for knowing the demand curve exactly (even with adversarial features), and (2) Ω ( √ T ) regret for almost knowing the demand curves (up to a one-parameter parametric family).

1 Previous works [Cohen et al., 2020, Krishnamurthy et al., 2021] did achieve polylog regrets, but only for negligible noise with σ = O ( 1 T log T ) .

Table 1: Related Works and Regret Bounds w.r.t. T

| Algorithm          | Work                                            | Regret (upper) bound             | Feature     | Noise                                    |
|--------------------|-------------------------------------------------|----------------------------------|-------------|------------------------------------------|
| LEAP               | [Amin et al., 2014]                             | ˜ O ( T 2 3 )                    | i.i.d.      | Noise-free                               |
| EllipsoidPricing   | [Cohen et al., 2020]                            | O (log T )                       | adversarial | Noise-free                               |
| EllipsoidEXP4      | [Cohen et al., 2020]                            | ˜ O ( T 2 3 )                    | adversarial | Sub-Gaussian                             |
| PricingSearch      | [Leme and Schneider, 2018]                      | O (log log( T ))                 | adversarial | Noise-free                               |
| RMLP               | [Javanmard and Nazerzadeh, 2019]                | O (log T/C 2 min ) † O ( √ T ) √ | i.i.d.      | Log-concave, distribution-known          |
| RMLP-2             | [Javanmard and Nazerzadeh, 2019]                | O ( T )                          | i.i.d.      | Known parametric family of log-concave.  |
| ShallowPricing     | [Cohen et al., 2020]                            | O ( poly (log T ))               | adversarial | Sub-Gaussian, known σ = O ( 1 T log T )  |
| CorPV              | [Krishnamurthy et al., 2021] [Liu et al., 2021] | O ( poly (log T ))               | adversarial | Sub-Gaussian, known σ = O ( 1 T log T )  |
| Algorithm 2 (MSPP) |                                                 | O (log log( T ))                 | adversarial | Noise-free                               |
| EMLP               | This paper                                      | O (log T )                       | i.i.d.      | Strictly log-concave, distribution-known |
| ONSP               | This paper                                      | O (log T )                       | adversarial | Strictly log-concave, distribution-known |

## 2 Related Works

In this section, we discuss our results relative to existing works on feature-based dynamic pricing, and highlight the connections and differences to the related settings of contextual bandits and contextual search (for a broader discussion, see Appendix A).

Feature-based Dynamic Pricing. There is a growing body of work on dynamic pricing with linear features [Amin et al., 2014, Qiang and Bayati, 2016, Cohen et al., 2020, Javanmard and Nazerzadeh, 2019]. Table 1 summarizes the differences in the settings and results 2 . Among these work, our paper directly builds upon [Cohen et al., 2020] and [Javanmard and Nazerzadeh, 2019], as we share the same setting of online feature vectors, linear and noisy valuations and Boolean-censored feedback. Relative to the results in [Javanmard and Nazerzadeh, 2019], we obtain O ( d log T ) regret under weaker assumptions on the sequence of input features - in both distribution-free stochastic feature setting and the adversarial feature setting. It is to be noted that [Javanmard and Nazerzadeh, 2019] also covers the sparse high-dimensional setting, and handles a slightly broader class of demand curves. Relative to [Cohen et al., 2020], in which the adversarial feature-based dynamic pricing was first studied, our algorithm ONSP enjoys the optimal O ( d log T ) regret when the noise-level is a constant. In comparison, Cohen et al. [2020] reduces the problem to contextual bandits and applies the (computationally inefficient) 'EXP-4' algorithm [Auer et al., 2002] to achieve a ˜ O ( T 2 / 3 ) regret. The 'bisection' style-algorithm in both Cohen et al. [2020] and Krishnamurthy et al. [2021] could achieve ˜ O ( poly ( d ) poly log( T )) regrets but requires a small-variance subgaussian noise satisfying σ = O ( 1 T log T ) .

Lower Bounds. Most existing works focus on the lower regret bounds of non-feature-based models. Kleinberg and Leighton [2003] divides the problem setting as fixed, random, and adversarial valuations, and then proves each a Θ (log log T ) , Θ ( √ T ) , and Θ ( T 2 / 3 ) regret, respectively. Broder and Rusmevichientong [2012] further proves a Θ ( √ T ) regret in general parametric valuation models. In this work, we generalize the methods of Broder and Rusmevichientong [2012] to our feature-based setting and further narrow it down to a linear-feature Gaussian-noisy model. As a complement to Kleinberg and Leighton [2003], we further separate the exponential regret gap between: (1) O (log T ) of the hardest (adversarial feature) totally-parametric model, and (2) Ω ( √ T ) of the simplest (fixed known expectation) unknownσ Gaussian model.

Contextual Bandits. For readers familiar with the online learning literature, our problem can be reduced to a contextual bandits problem [Langford and Zhang, 2007, Agarwal et al., 2014] by discretizing the prices. But this reduction only results in O ( T 2 / 3 ) regret, as it does not capture the special structure of the feedback: an accepted price indicates the acceptance of all lower prices , and vise versa. Moreover, when comparing to linear bandits [Chu et al., 2011], it is the valuation instead of the expected reward that we assume to be linear.

Contextual Search. Feature-based dynamic pricing is also related to the contextual search problem [Lobel et al., 2018, Leme and Schneider, 2018, Liu et al., 2021, Krishnamurthy et al., 2021], which often involves learning from Boolean feedbacks, sometimes with a 'pricing loss' and 'noisy' feedback. These shared jargons make this problem appearing very similar to our problem. However, except for the noiseless cases [Lobel et al., 2018, Leme and Schneider, 2018], contextual search

2 We only concern the dependence on T since there are various different assumptions on d .

algorithms, even with 'pricing losses' and 'Noisy Boolean feedback' [e.g., Liu et al., 2021], do not imply meaningful regret bounds in our problem setup due to several subtle but important differences in the problem settings. Specifically, the noisy-boolean feedback model of [Liu et al., 2021] is about randomly toggling the 'purchase decision' determined by the noiseless valuation x glyph[latticetop] θ ∗ with probability 0 . 5 -glyph[epsilon1] . This is incompatible to our problem setting where the purchasing decision is determined by a noisy valuation x glyph[latticetop] θ ∗ + Noise. Ultimately, in the setting of [Liu et al., 2021], the optimal policy alway plays x glyph[latticetop] θ ∗ , but our problem is harder in that we need to exploit the noise and the optimal price could be very different from x glyph[latticetop] θ ∗ . 3 Krishnamurthy et al. [2021] also discussed this issue explicitly and considered the more natural noisy Boolean feedback model studied in this paper. Their result, similar to Cohen et al. [2020], only achieves a logarithmic regret when the noise on the valuation is vanishing in an ˜ O (1 /T ) rate.

## 3 Problem Setup

Symbols and Notations. Now we introduce the mathematical symbols and notations involved in the following pages. The game consists of T rounds. x t ∈ R d , v t ∈ R + and N t ∈ R denote the feature vector, the proposed price and the noise respectively at round t = 1 , 2 , ..., T . 4 We denote the product u t := x glyph[latticetop] t θ ∗ as an expected valuation . At each round, we receive a payoff (reward) r t = v t · ✶ t , where the binary variable ✶ t indicates whether the price is accepted or not, i.e., ✶ t = 1 ( v t ≤ w t ) . As we may estimate θ ∗ in our algorithms, we denote ˆ θ t as an estimator of θ ∗ , which we will formally define in the algorithms. Furthermore, we denote some functions that are related to noise distribution: F ( ω ) and f ( ω ) denote the cumulative distribution function (CDF) and probability density function (PDF) sequentially. We know that F ′ ( ω ) = f ( ω ) if we assume differentiability. To concisely denote all data observed up to round τ (i.e., feature, price and payoff of all past rounds), we define hist ( τ ) = { ( x t , v t , ✶ t ) for t = 1 , 2 , ..., τ } . hist ( τ ) represents the transcript of all observed random variables before round ( τ +1) . We define l t ( θ ) := -✶ t · log ( 1 -F ( v t -x glyph[latticetop] t θ ) ) -(1 -✶ t ) log ( F ( v t -x glyph[latticetop] t θ ) ) (1) as a negative log-likelihood function at round t . Also, we define an expected log-likelihood function L t ( θ ) :

<!-- formula-not-decoded -->

Notice that we will later define an ˆ L k ( θ ) which is, however, not an expectation.

Definitions of Key Quantities. We firstly define an expected reward function g ( v, u ) .

<!-- formula-not-decoded -->

This indicates that if the expected valuation is u and the proposed price is v , then the (conditionally) expected reward is g ( v, u ) . Now we formally define the regret of a policy (algorithm) A as is promised in Section 1.

Definition 1 (Regret) . Let A : R d × ( R d , R , { 0 , 1 } ) t -1 → R be a policy of pricing, i.e. A ( x t , hist ( t -1)) = v t . The regret of A is defined as follows.

<!-- formula-not-decoded -->

Here hist ( t -1) is the historical records until ( t -1) th round.

Summary of Assumptions. We specify the problem settings by proposing three assumptions.

Assumption 1 (Known, bounded, strictly log-concave distribution) . The noise N t is independently and identically sampled from a distribution whose CDF is F . Assume that F ∈ C 2 is strictly increasing and that F and (1 -F ) are strictly log-concave. Also assume that f and f ′ are bounded, and denote B f := sup ω ∈ R f ( ω ) , B f ′ := sup ω ∈ R | f ′ ( ω ) | as two constants.

3 As an explicit example, suppose the valuation x glyph[latticetop] θ ∗ = 0 , then the optimal price must be &gt; 0 in order to avoid zero return.

4 In an epoch-design situation, a subscript ( k, t ) indicates round t of epoch k .

Algorithm 1 Epoch-based max-likelihood pricing (EMLP)

```
Input: Convex and bounded set H Observe x 1 , randomly choose v 1 and get r 1 . Solve ˆ θ 1 = arg min θ ∈ H l 1 ( θ ) ; for k = 1 to glyph[floorleft] log 2 T glyph[floorright] +1 do Set τ k = 2 k -1 ; for t = 1 to τ k do Observe x k,t ; Set price v k,t = J ( x glyph[latticetop] k,t ˆ θ k ) ; Receive r k,t = v k t · ✶ t ; end for Solve: ˆ θ k +1 = arg min θ ∈ H ˆ L k ( θ ) , where ˆ L k ( θ ) = 1 τ k ∑ τ k t =1 l k,t ( θ ) . end for
```

Algorithm 2 Online Newton Step Pricing (ONSP)

```
Input: Convex and bounded set H , θ 1 , parameter γ, glyph[epsilon1] > 0 Set A 0 = glyph[epsilon1] · I d ; for t = 1 to T do Observe x t ; Set price v t = J ( x glyph[latticetop] t θ t ) ; Receive r t = v t · ✶ t ; Set surrogate loss function l t ( θ ) ; Calculate ∇ t = ∇ l t ( θ ) ; Rank-1 update: A t = A t -1 + ∇ t ∇ glyph[latticetop] t ; Newton step: ˆ θ t +1 = θ t -1 γ A -1 t ∇ t ; Projection: θ t +1 = ∏ A t H ( ˆ θ t +1 ) . end for
```

Assumption 2 (Bounded convex parameter space) . The true parameter θ ∗ ∈ H , where H ⊆ { θ : || θ || 2 ≤ B 1 } is a bounded convex set and B 1 is a constant. Assume H is known to us (but θ ∗ is not).

Assumption 3 (Bounded feature space) . Assume x t ∈ D ⊆ { x : || x || 2 ≤ B 2 } , ∀ t = 1 , 2 , . . . , T . Also, 0 ≤ x glyph[latticetop] θ ≤ B, ∀ x ∈ D, ∀ θ ∈ H , where B = B 1 · B 2 is a constant.

Assumption 2 and 3 are mild as we can choose B 1 and B 2 large enough. In Section 4.1, we may add further complement to Assumption 3 to form a stochastic setting. Assumption 1 is stronger since we might not know the exact CDF in practice, but it is still acceptable from an informationtheoretic perspective. There are at least three reasons that lead to this assumption: Primarily, this is necessary if we hope to achieve an O (log( T )) regret. We will prove in Section 5.3 that an Ω ( √ T ) is unavoidable if we cannot know one parameter exactly. Moreover, the pioneering work of Javanmard and Nazerzadeh [2019] also assumes a known noise distribution with log-concave CDF, and many common distributions are actually strictly log-concave, such as Gaussian and logistic. 5 Besides, although we did not present a method to precisely estimate σ in this work, it is a reasonable algorithm to replace with a plug-in estimator estimated using historical offline data. As we have shown, not knowing σ requires O ( √ T ) regret in general, but the lower bound does not rule out the plug-in approach achieving a smaller regret for interesting subclasses of problems in practice.

Finally, we state a lemma and define an argmax function helpful for our algorithm design.

Lemma 2 (Uniqueness) . For any u ≥ 0 , there exists a unique v ∗ ≥ 0 such that g ( v ∗ , u ) = max v ∈ R g ( v, u ) . Thus, we can define a greedily pricing function that maximizes the expected reward:

<!-- formula-not-decoded -->

v

Please see the proof of Lemma 2 in Appendix B.1.

## 4 Algorithms

In this section, we propose two dynamic pricing algorithms: EMLP and ONSP, for stochastic and adversarial features respectively.

## 4.1 Pricing with Distribution-Free Stochastic Features

Assumption 4 (Stochastic features) . Assume x t ∼ D ⊆ D are independently identically distributed (i.i.d.) from an unknown distribution, for any t = 1 , 2 , . . . , T .

The first algorithm, Epoch-based Max-Likelihood Pricing (EMLP) algorithm, is suitable for a stochastic setting defined by Assumption 4. EMLP proceeds in epochs with each stage doubling the length of the previous epoch. At the end of each epoch, we consolidate the observed data and solve a maximum likelihood estimation problem to learn θ . A max likelihood estimator (MLE) obtained by minimizing ˆ L k ( θ ) := 1 τ k ∑ τ k t =1 l k,t ( θ ) , which is then used in the next epoch as if it is the true parameter vector. In the equation, k, τ k denotes the index and length of epoch k . The estimator is

5 In fact, F and (1 -F ) are both log-concave if its PDF is log-concave, according to Prekopa's Inequality.

computed using data in hist ( k ) , which denotes the transcript for epoch 1 ∼ k . The pseudo-code of EMLP is summarized in Algorithm 1. In the remainder of this section, we discuss the computational efficiency and prove the upper regret bound of O ( d log T ) .

Computational Efficiency. The calculations in EMLP are straightforward except for arg min ˆ L k ( θ ) and J ( u ) . As g ( v, u ) is proved unimodal in Lemma 2, we may efficiently calculate J ( u ) by binary search. We will prove that l k,t is exp-concave (and thus also convex). Therefore, we may apply any off-the-shelf tools for solving convex optimization.

MLE and Probit Regression. A closer inspection reveals that this log-likelihood function corresponds to a probit [Aldrich et al., 1984] or a logit model [Wright, 1995] for Gaussian or logistic noises. See Appendix C.2.1.

Affine Invariance. Both optimization problems involved depend only on x glyph[latticetop] θ , so if we add any affine transformation to x into ˜ x = Ax , the agent can instead learn a new parameter of ˜ θ ∗ = ( A glyph[latticetop] ) -1 θ ∗ and achieve the same u t = x glyph[latticetop] t θ ∗ . Also, the regret bound is not affected as the upper bound B over x glyph[latticetop] θ does not change 6 . Therefore, it is only natural that the regret bound does not depend on the distribution x , nor the condition numbers of E [ xx glyph[latticetop] ] (i.e., the ratio of λ max /λ min ).

## 4.2 Pricing with Adversarial Features

In this part, we propose an 'Online Newton Step Pricing (ONSP)' algorithm that deals with adversarial { x t } series and guarantees O ( d log T ) regret. The pseudo-code of ONSP is shown as Algorithm 2. In each round, it uses the likelihood function as a surrogate loss and applies 'Online Newton Step'(ONS) method to update ˆ θ . In the next round, it adopts the updated ˆ θ and sets a price greedily. In the remainder of this section, we discuss some properties of ONSP and prove the regret bound.

The calculations of ONSP are straightforward. The time complexity of calculating the matrix inverse A -1 t is O ( d 3 ) , which is fair as d is small. In high-dimensional cases, we may use Woodbury matrix identity 7 to reduce it to O ( d 2 ) as we could get A -1 directly from the latest round.

## 5 Regret Analysis

In this section, we mainly prove the logarithmic regret bounds of EMLP and ONSP corresponding to stochastic and adversarial settings, respectively. Besides, we also prove an Ω ( √ T ) regret bound on fully parametric F with one parameter unknown.

## 5.1 O ( d log T ) Regret of EMLP

In this part, we present the regret analysis of Algorithm 1. First of all, we propose the following theorem as our main result on EMLP.

Theorem 3 (Overall regret) . With Assumptions 1, 2, 3 and 4, the expected regret of EMLP can be bounded by:

<!-- formula-not-decoded -->

where C s is a constant that depends only on F ( ω ) and is independent to D .

The proof of Theorem 3 is sophisticated. For the sake of clarity, we next present an inequality system as a roadmap toward the proof. After this, we formally illustrate each line of it with lemmas.

Since EMLP proposes J ( x glyph[latticetop] k,t ˆ θ k ) in every round of epoch k , we may denote the per-round regret as Reg t ( ˆ θ k ) , where:

<!-- formula-not-decoded -->

Therefore, it is sufficient to prove the following Theorem:

Theorem 4 (Expected per-round regret) . For the per-round regret defined in Equation (7) , we have:

<!-- formula-not-decoded -->

6 Here A is assumed invertible, otherwise the mapping from ˜ x t to u t does not necessarily exist.

7 ( A + xx glyph[latticetop] ) -1 = A -1 -1 1+ x glyph[latticetop] A -1 x A -1 x ( A -1 x ) glyph[latticetop] .

The proof roadmap of Theorem 4 can be written as the following inequality system.

<!-- formula-not-decoded -->

We explain Equation (8) in details. The first inequality comes from the following Lemma 5.

Lemma 5 (Quadratic regret bound) . We have:

<!-- formula-not-decoded -->

The intuition is that function g ( J ( u ) , u ) is 2 nd -order-smooth at ( J ( u ∗ ) , u ∗ ) . A detailed proof of Lemma 5 is in Appendix B.2.1. Note that C is highly dependent on the distribution F . After this, we propose Lemma 6 that contributes to the second inequality of Equation (8).

Lemma 6 (Quadratic likelihood bound) . For the expected likelihood function L t ( θ ) defined in Equation (2) , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Since the true parameter always maximizes the expected likelihood function [Murphy, 2012], by Taylor Expansion we have ∇ L ( θ ∗ ) = 0 , and hence L t ( θ ) -L t ( θ ∗ ) = 1 2 ( θ -θ ∗ ) glyph[latticetop] ∇ 2 L t ( ˜ θ )( θ -θ ∗ ) for some ˜ θ = αθ ∗ +(1 -α ) θ . Therefore, we only need to prove the following lemma:

Lemma7 (Strong convexity and Exponential Concavity) . Suppose l t ( θ ) is the negative log-likelihood function in epoch k at time t . For any θ ∈ H , x t ∼ D , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma 7 is in Appendix B.2.2. With this lemma, we see that Lemma 6 holds.

With Lemma 5 and Lemma 6, we can immediately get the following Lemma 8.

Lemma 8 (Surrogate Regret) . The relationship between Reg ( θ ) and likelihood function can be shown as follows:

<!-- formula-not-decoded -->

∀ θ ∈ H , ∀ x ∈ D , where C and C down are defined in Lemma 5 and 6 respectively.

Lemma 8 enables us to choose the negative log-likelihood function as a surrogate loss. This is not only an important insight of EMLP regret analysis, but also the foundation of ONSP design.

The last inequality of Equation (8) comes from this lemma:

Lemma 9 (Per-epoch surrogate regret bound) . Denoting ˆ θ k as the estimator coming from epoch ( k -1) and being used in epoch k , we have:

[ ˆ

L

( ˆ

θ

)

-

ˆ

L

(

θ

∗

)]

≤

C

C

exp down

τ

k

d

+1

Here C exp is defined in Equation 13, and E h [ · ] = E [ ·| hist ( k -1)] .

·

.

(15)

E

h

k

k

k

Proof of Lemma 9 is partly derived from the work Koren and Levy [2015], and here we give a proof sketch without specific derivations. A detailed proof lies in Appendix B.2.3.

Proof sketch of Lemma 9. We list the four main points that contribute to the proof:

- Notice that l k,t ( θ ) is strongly convex w.r.t. a seminorm x k,t x glyph[latticetop] k,t , we know ˆ L k ( θ ) is also strongly convex w.r.t. ∑ τ k t =1 x k,t x glyph[latticetop] k,t .
- For two strongly convex functions g 1 and g 2 , we can upper bound the distance between their arg-minimals (scaled by some norm || · || ) with the dual norm of ∇ ( g 1 -g 2 ) .
- Since a seminorm has no dual norm, we apply two methods to convert it into a norm: (1) separation of parameters and likelihood functions with a 'leave-one-out' method (to separately take expectations), and (2) separation of the spinning space and the null space.
- As the dual data-dependent norm offsets the sum of xx glyph[latticetop] to a constant, Lemma 9 holds.

We have so far proved Inequality (8) after proving Lemma 5, 6, 9. Therefore, Theorem 4 holds.

## 5.2 O ( d log T ) Regret of ONSP

Here we present the regret analysis of Algorithm 2 (ONSP). Firstly, we state the main theorem.

Theorem 10. With Assumptions 1, 2, 3, the regret of Algorithm 2 (ONSP) satisfies:

<!-- formula-not-decoded -->

where C a is a function only dependent on F .

Proof. Proof of Theorem 10 here is more concise than Section 5.1, because the important Lemma 7 and 8 have been proved there. From Lemma 8, we have:

<!-- formula-not-decoded -->

With Equation 17, we may reduce the regret of likelihood functions as a surrogate regret of pricing. From Lemma 7 we see that the log-likelihood function is C down C exp -exponentially concave 8 . This enables an application of Online Newton Step method to achieve a logarithmic regret. Therefore, by citing from the Online Convex Optimization [Hazan, 2016], we have the following Lemma.

Lemma 11 (Online Newton Step) . With parameters γ = 1 2 min { 1 4 GD , α } and glyph[epsilon1] = 1 γ 2 D 2 , and T &gt; 4 guarantees:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With Equation 17 and Lemma 11, we have:

<!-- formula-not-decoded -->

Therefore, we have proved Lemma 10.

## 5.3 Lower Bound for Unknown Distribution

In this part, we evaluate Assumption 1 and prove that an Ω ( √ T ) lower regret bound is unavoidable with even a slight relaxation: a Gaussian noise with unknown σ . Our proof is inspired by Broder and Rusmevichientong [2012] Theorem 3.1, while our lower bound relies on more specific assumptions (and thus applies to more general cases).

We firstly state Assumption 5 covering this part, and then state Theorem 12 as a lower bound:

8 A function f ( µ ) is α -exponentially concave iff ∇ 2 f ( µ ) glyph[followsequal] α ∇ f ( µ ) ∇ f ( µ ) glyph[latticetop] .

<!-- formula-not-decoded -->

Figure 1: The regret of EMLP, ONSP and EXP-4 on simulated examples (we only conduct EXP-4 up to T = 2 12 due to its exponential time consuming), with Figure a for stochastic features and Figure b for adversarial ones. The plots are in log-log scales with all regrets divided by a log( t ) factor to show the convergence. For EXP-4, we discretize the parameter space with T -1 3 -size grids, which would incur an ˜ O ( T 2 3 ) regret according to Cohen et al. [2020]. We also plot linear fits for some regret curves, where a slopeα line indicates an O ( T α ) regret. Besides, we draw error bars and bands with 0.95 coverage using Wald's test. The two diagrams reveal that (i) logarithmic regrets of EMLP and ONSP in the stochastic setting, (ii) a nearly-linear regret of EMLP in the adversarial setting, and (iii) O ( T 2 3 ) regrets of EXP-4 in both settings.

<!-- image -->

Assumption 5. The noise N t ∼ N (0 , σ 2 ) independently, where 0 &lt; σ ≤ 1 is fixed and unknown .

Theorem 12 (Lower bound with unknown σ ) . Under Assumption 2, 3, 4 and 5, for any policy (algorithm) Ψ : R d × ( R d , R , { 0 , 1 } ) t -1 → R + and any T &gt; 2 , there exists a Gaussian parameter σ ∈ R + , a distribution D of features and a fixed parameter θ ∗ , such that: Reg Ψ ≥ 1 24000 · √ T.

Remark: Here we assume x t to be i.i.d., which also implies the applicability on adversarial features. However, the minimax regret of the stochastic feature setting is Θ ( √ T ) [Javanmard and Nazerzadeh, 2019], while existing results have not yet closed the gap in adversarial feature settings.

Proof sketch of Theorem 12. Here we assume a fixed valuation, i.e. u ∗ = x glyph[latticetop] t θ ∗ , ∀ t = 1 , 2 , . . . . Equivalently, we assume a fixed feature. The main idea of proof is similar to that in Broder and Rusmevichientong [2012]: we assume σ 1 = 1 , σ 2 = 1 -T -1 4 , and we prove that: (1) it is costly for an algorithm to perform well in both cases if the σ 's are different by a lot, and (2) it is costly for an algorithm to distinguish the two cases if σ 's are close enough to each other. We put the detailed proof in Appendix B.3.

## 6 Numerical Result

In this section, we conduct numerical experiments to validate EMLP and ONSP. In comparison with the existing work, we implement a discretized EXP-4 [Auer et al., 2002] algorithm for pricing, as is introduced in Cohen et al. [2020] (in a slightly different setting). We will test these three algorithms in both stochastic and adversarial settings.

Basically, we assume d = 2 , B 1 = B 2 = B = 1 and N t ∼ N (0 , σ 2 ) with σ = 0 . 25 . In both settings, we conduct EMLP and ONSP for T = 2 16 rounds. For ONSP, we empirically select γ and glyph[epsilon1] that accelerates the convergence, instead of using the values specified in Lemma 11. Since EXP-4 consumes exponential time and requires the knowledge of T in advance to discretize the policy and valuation spaces, we execute EXP-4 for a series of T = 2 k , k = 1 , 2 , . . . , 12 . We repeat every experiment 5 times for each setting and then take an average.

Stochastic Setting. We implement and test EMLP, ONSP and EXP-4 with stochastic { x t } 's. The numerical results are shown in Figure 1a on a log-log diagram, with the regrets divided by log( t ) . It shows log( t ) -convergences on EMLP and ONSP, while EXP-4 is in a t α rate with α ≈ 0 . 699 .

Adversarial Setting. We implement the three algorithms and test them with an adversarial { x t } 's: for the k -th epoch, i.e. t = 2 k -1 , 2 k -1 +1 , . . . , 2 k -1 , we let x t = [1 , 0] glyph[latticetop] if k ≡ 1( mod 2) and x t = [0 , 1] glyph[latticetop] if k ≡ 0( mod 2) . The numerical results are shown in Figure 1b on a log-log diagram, with the regrets divided by log( t ) . The log-log plots of ONSP and EXP-4 are almost the same as those in Figure 1a. However, EMLP shows an almost linear ( t α rate with α ≈ 0 . 912 ) regret in this adversarial setting. This is because the adversarial series only trains one dimension of θ in each epoch, while the other is arbitrarily initialized and does not necessarily converge. However, in the next epoch, the incorrect dimension is exploited. Therefore, a linear regret originates.

## 7 Discussion

Here we discuss the coefficients on our regret bounds as a potential extension of future works. In Appendix C we will discuss more on algorithmic design, problem modeling, and ethic issues.

Coefficients on Regret Bounds. The exact regret bounds of both EMLP and ONSP contain a constant C exp C down that highly depends on the noise CDF F and could be large. A detailed analysis in Appendix C.1 shows that C exp C down is exponentially large w.r.t. B σ (see Equation 39 and Lemma 21) for Gaussian noise N (0 , σ 2 ) , which implies that a smaller noise variance would lead to a (much) larger regret bound. This is very counter-intuitive as a larger noise usually leads to a more sophisticated situation, but similar phenomenons also occur in existing algorithms that are suitable for constantvariance noise, such as RMLP in Javanmard and Nazerzadeh [2019] and OORMLP in Wang et al. [2020]. In fact, it is because a (constantly) large noise would help explore the unknown parameter θ ∗ and smoothen the expected regret. In this work, this can be addressed by increasing T since we mainly concern the asymptotic regrets as T →∞ with fixed noise distributions. However, we admit that it is indeed a nontrivial issue for finite T and small σ situations. There exists a 'ShallowPricing' method in Cohen et al. [2020] that can deal with a very-small-variance noise setting (when σ = ˜ O ( 1 T ) ) and achieve a logarithmic regret. Specifically, its regret bound would decrease as the noise variance σ decreases (but would still not reach O (log log T ) as the noise vanishes). We might also apply this method as a preprocess to cut the parameter domain and decrease B σ within logarithmic trials (see Cohen et al. [2020] Thm. 3), but it is still open whether a log( T ) regret is achievable when σ = Θ ( T -α ) for α ∈ (0 , 1) .

## 8 Conclusion

In this work, we studied the problem of online feature-based dynamic pricing with a noisy linear valuation in both stochastic and adversarial settings. We proposed a max-likelihood-estimate-based algorithm (EMLP) for stochastic features and an online-Newton-step-based algorithm (ONSP) for adversarial features. Both of them enjoy a regret guarantee of O ( d log T ) , which also attains the information-theoretic limit up to a constant factor. Compared with existing works, EMLP gets rid of strong assumptions on the distribution of the feature vectors in the stochastic setting, and ONSP improves the regret bound exponentially from O ( T 2 / 3 ) to O (log T ) in the adversarial setting. We also showed that knowing the noise distribution (or the demand curve) is required to obtain logarithmic regret, where we prove a lower bound of Ω ( √ T ) on the regret for the case when the noise is knowingly Gaussian but with an unknown σ . In addition, we conducted numerical experiments to empirically validate the scaling of our algorithms. Finally, we discussed the regret dependence on the noise variance, and proposed a subtle open problem for further study.

## Acknowledgments

The work is partially supported by the Adobe Data Science Award and a start-up grant from the UCSB Department of Computer Science. We appreciate the input from anonymous reviewers and AC as well as a discussion with Akshay Krishnamurthy for clarifying some details of Krishnamurthy et al. [2021].

## References

- A. Agarwal, D. Hsu, S. Kale, J. Langford, L. Li, and R. Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In International Conference on Machine Learning (ICML-14) , pages 1638-1646, 2014.

- J. H. Aldrich, F. D. Nelson, and E. S. Adler. Linear Probability, Logit, and Probit Models . Number 45. Sage, 1984.
- K. Amin, A. Rostamizadeh, and U. Syed. Repeated contextual auctions with strategic buyers. In Advances in Neural Information Processing Systems (NIPS-14) , pages 622-630, 2014.
- V. F. Araman and R. Caldentey. Dynamic pricing for nonperishable products with demand learning. Operations Research , 57(5):1169-1188, 2009.
- P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing , 32(1):48-77, 2002.
- G. Aydin and S. Ziya. Personalized dynamic pricing of limited inventories. Operations Research , 57 (6):1523-1531, 2009.
- D. Besanko, S. Gupta, and D. Jain. Logit demand estimation under competitive pricing behavior: An equilibrium framework. Management Science , 44(11-part-1):1533-1547, 1998.
- D. Besanko, J.-P. Dubé, and S. Gupta. Competitive price discrimination strategies in a vertical channel using aggregate retail data. Management Science , 49(9):1121-1138, 2003.
- O. Besbes and A. Zeevi. On the (surprising) sufficiency of linear models for dynamic pricing with demand learning. Management Science , 61(4):723-739, 2015.
- J. Broder and P. Rusmevichientong. Dynamic pricing under a general parametric choice model. Operations Research , 60(4):965-980, 2012.
- T. Chan, V. Kadiyali, and P. Xiao. Structural models of pricing. Handbook of pricing research in marketing , pages 108-131, 2009.
- N. Chen and G. Gallego. A primal-dual learning algorithm for personalized dynamic pricing with an inventory constraint. Mathematics of Operations Research , 2021.
- Y. Chen and V. F. Farias. Simple policies for dynamic pricing with imperfect forecasts. Operations Research , 61(3):612-624, 2013.
- W. Chu, L. Li, L. Reyzin, and R. Schapire. Contextual bandits with linear payoff functions. In International Conference on Artificial Intelligence and Statistics (AISTATS-11) , pages 208-214, 2011.
- M. C. Cohen, I. Lobel, and R. Paes Leme. Feature-based dynamic pricing. Management Science , 66 (11):4921-4943, 2020.
- A. A. Cournot. Researches into the Mathematical Principles of the Theory of Wealth . Macmillan, 1897.
- A. V. den Boer. Dynamic pricing and learning: historical origins, current research, and new directions. Surveys in Operations Research and Management Science , 20(1):1-18, 2015.
- M. Draganska and D. C. Jain. Consumer preferences and product-line pricing strategies: An empirical analysis. Marketing science , 25(2):164-174, 2006.
- G. C. Evans. The dynamics of monopoly. The American Mathematical Monthly , 31(2):77-83, 1924.
- E. Hazan. Introduction to online convex optimization. Foundations and Trends in Optimization , 2 (3-4):157-325, 2016.
- R. Iyengar, A. Ansari, and S. Gupta. A model of consumer learning for service quality and usage. Journal of Marketing Research , 44(4):529-544, 2007.
- A. Javanmard and H. Nazerzadeh. Dynamic pricing in high-dimensions. The Journal of Machine Learning Research , 20(1):315-363, 2019.
- P. L. Joskow and C. D. Wolfram. Dynamic pricing of electricity. American Economic Review , 102 (3):381-85, 2012.

- V. Kadiyali, N. J. Vilcassim, and P. K. Chintagunta. Empirical analysis of competitive product line pricing decisions: Lead, follow, or move together? Journal of Business , pages 459-487, 1996.
- N. B. Keskin and A. Zeevi. Dynamic pricing with an unknown demand model: Asymptotically optimal semi-myopic policies. Operations Research , 62(5):1142-1167, 2014.
- W. Kincaid and D. Darling. An inventory pricing problem. Journal of Mathematical Analysis and Applications , 7:183-208, 1963.
- R. Kleinberg and T. Leighton. The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In IEEE Symposium on Foundations of Computer Science (FOCS-03) , pages 594-605. IEEE, 2003.
- T. Koren and K. Levy. Fast rates for exp-concave empirical risk minimization. In Advances in Neural Information Processing Systems (NIPS-15) , pages 1477-1485, 2015.
- A. Krämer, M. Friesen, and T. Shelton. Are airline passengers ready for personalized dynamic pricing? a study of german consumers. Journal of Revenue and Pricing Management , 17(2): 115-120, 2018.
- A. Krishnamurthy, T. Lykouris, C. Podimata, and R. Schapire. Contextual search in the presence of irrational agents. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing (STOC-21) , pages 910-918, 2021.
- A. Lambrecht, K. Seim, and B. Skiera. Does uncertainty matter? consumer behavior under three-part tariffs. Marketing Science , 26(5):698-710, 2007.
- J. Langford and T. Zhang. The epoch-greedy algorithm for contextual multi-armed bandits. In Advances in Neural Information Processing Systems (NIPS-07) , pages 817-824, 2007.
- R. P. Leme and J. Schneider. Contextual search via intrinsic volumes. In 2018 IEEE 59th Annual Symposium on Foundations of Computer Science (FOCS-18) , pages 268-282. IEEE, 2018.
- A. Liu, R. P. Leme, and J. Schneider. Optimal contextual pricing and extensions. In Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms (SODA-21) , pages 1059-1078. SIAM, 2021.
- I. Lobel, R. P. Leme, and A. Vladu. Multidimensional binary search for contextual decision-making. Operations Research , 66(5):1346-1361, 2018.
- T. Mazumdar, S. P. Raj, and I. Sinha. Reference price research: Review and propositions. Journal of Marketing , 69(4):84-102, 2005.
- J. Mourtada. Exact minimax risk for linear least squares, and the lower tail of sample covariance matrices. arXiv preprint arXiv:1912.10754 , 2019.
- K. P. Murphy. Machine Learning: a Probabilistic Perspective . MIT press, 2012.
- S. Qiang and M. Bayati. Dynamic pricing with demand covariates. arXiv preprint arXiv:1604.07463 , 2016.
- H. Schultz et al. Theory and Measurement of Demand . The University of Chicago Press, 1938.
18. C.-H. Wang, Z. Wang, W. W. Sun, and G. Cheng. Online regularization for high-dimensional dynamic pricing algorithms. arXiv preprint arXiv:2007.02470 , 2020.
- P. Whittle. Multi-armed bandits and the gittins index. Journal of the Royal Statistical Society: Series B (Methodological) , 42(2):143-149, 1980.
- R. E. Wright. Logistic regression. 1995.

## APPENDIX

## A Other related works

Here we will briefly review the history and recent studies that are related to our work. For the historical introductions, we mainly refer to den Boer [2015] as a survey. For bandit approaches, we will review some works that apply bandit algorithms to settle pricing problems. For the structural models, we will introduce different modules based on the review in Chan et al. [2009]. Based on the existing works, we might have a better view of our problem setting and methodology.

## A.1 History of Pricing

It was the work of Cournot [1897] in 1897 that firstly applied mathematics to analyze the relationship between prices and demands. In that work, the price was denoted as p and the demand was defined as a demand function F ( p ) . Therefore, the revenue could be written as pF ( p ) . This was a straightforward interpretation of the general pricing problem, and the key to solving it was estimations of F ( p ) regarding different products. Later in 1938, the work Schultz et al. [1938] proposed price-demand measurements on exclusive kinds of products. It is worth mentioning that these problems are 'static pricing' ones, because F is totally determined by price p and we only need to insist on the optimal one to maximize our profits.

However, the static settings were qualified by the following two observations: on the one hand, a demand function may not only depends on the static value of p , but also be affected by the trend of p 's changing [Evans, 1924, Mazumdar et al., 2005]; on the other hand, even if F ( p ) is static, p itself might change over time according to other factors such as inventory level [Kincaid and Darling, 1963]. As a result, it is necessary to consider dynamics in both demand and price, which leads to a 'dynamic pricing' problem setting.

## A.2 Dynamic Pricing as Bandits

As is said in Section 2, the pricing problem can be viewed as a stochastic contextual bandits problem [see, e.g., Langford and Zhang, 2007, Agarwal et al., 2014]. Even though we may not know the form of the demand function, we can definitely see feedback of demands, i.e. how many products are sold out, which enables us to learn a better decision-making policy. Therefore, it can be studied in a bandit module. If the demand function is totally agnostic, i.e. the evaluations (the highest prices that customers would accept) come at random or even at adversary over time, then it can be modeled as a Multi-arm bandit (MAB) problem [Whittle, 1980] exactly. In our paper, instead, we focus on selling different products with a great variety of features. This can be characterized as a Contextual bandit (CB) problem [Auer et al., 2002, Langford and Zhang, 2007]. The work Cohen et al. [2020], which applies the 'EXP-4' algorithm from Auer et al. [2002], also mentions that 'the arms represent prices and the payoffs from the different arms are correlated since the measures of demand evaluated at different price points are correlated random variables'. A variety of existing works, including Kleinberg and Leighton [2003], Araman and Caldentey [2009], Chen and Farias [2013], Keskin and Zeevi [2014], Besbes and Zeevi [2015], has been approaching the demand function from a perspective of from either parameterized or non-parameterized bandits.

However, our problem setting is different from a contextual bandits setting in at least two perspectives: feedback and regret. The pricing problem has a specially structured feedback between full information and bandits setting. Specifically, r t &gt; 0 implies that all policies producing v &lt; v t will end up receiving r ′ t = v , and r t = 0 implies that all policies producing v &gt; v t will end up receiving r ′ t = 0 . However, the missing patterns are confounded with the rewards. Therefore it is non-trivial to leverage this structure to improve the importance sampling approach underlying the algorithm of Agarwal et al. [2014]. We instead consider the natural analog to the linear contextual bandits setting [Chu et al., 2011] 9 and demonstrate that in this case an exponential improvement in the regret is possible using the additional information from the censored feedback. As for regret, while in contextual bandits it refers to a comparison with the optimal policy, it is here referring to a comparison with the optimal action . In other words, though our approaches (both in EMLP and in ONSP) are finding the true parameter θ ∗ , the regret is defined as the 'revenue gap' between the optimal price and our proposed prices. These are actually equivalent in our fully-parametric setting (where we assume a

9 But do notice that our expected reward above is not linear, even if the valuation function is.

linear-valuation-known-noise model), but will differ a lot in partially parametric and totally agnostic settings.

## A.3 Structural Model

While a totally agnostic model guarantees the most generality, a structural model would help us better understand the mechanism behind the observation of prices and demands. The key to a structural pricing model is the behavior of agents in the market, including customers and/or firms. In other words, the behavior of each side can be described as a decision model. From the perspective of demand (customers), the work Kadiyali et al. [1996] adopts a linear model on laundry detergents market, Iyengar et al. [2007] and Lambrecht et al. [2007] study three-part-tariff pricing problems on wireless and internet services with mixed logit models. Besanko et al. assumed an aggregate logit model on customers in works Besanko et al. [1998] and Besanko et al. [2003] in order to study the competitive behavior of manufacturers in ketchup market. Meanwhile, the supply side is usually assumed to be more strategic, such as Bertrand-Nash behaviors [Kadiyali et al., 1996, Besanko et al., 1998, Draganska and Jain, 2006]. For more details, please see Chan et al. [2009].

## B Proofs

## B.1 Proof of Lemma 2

Proof. Since v ∗ = argmax g ( v, u ) , we have:

<!-- formula-not-decoded -->

Define ϕ ( ω ) = 1 -F ( ω ) f ( ω ) -ω , and we take derivatives:

<!-- formula-not-decoded -->

where the last equality comes from the strict log-concavity of (1 -F ( ω )) . Therefore, ϕ ( ω ) is decreasing and ϕ (+ ∞ ) = -∞ . Also, notice ϕ ( -∞ ) = + ∞ , we know that for any u ∈ R , there exists an ω such that ϕ ( ω ) = u . For u ≥ 0 , we know that g ( v, u ) ≥ 0 for v ≥ 0 and g ( v, u ) &lt; 0 for v &lt; 0 . Therefore, v ∗ ≥ 0 if u ≥ 0 .

## B.2 Proofs in Section 5.1

## B.2.1 Proof of Lemma 5

Proof. We again define ϕ ( ω ) = 1 -F ( ω ) f ( ω ) -ω as in Appendix B.1. According to Equation 5, we have:

<!-- formula-not-decoded -->

The last line of Equation 19 is due to the Implicit Function Derivatives Principle. From the result in Appendix B.1, we know that ϕ ′ ( ω ) &lt; -1 , ∀ ω ∈ R . Therefore, we have J ′ ( u ) ∈ (0 , 1) , u ∈ R , and hence 0 ≥ J ( u ) &lt; u + J (0) for u ≥ 0 . Since u ∈ [0 , B ] , we may assume that v ∈ [0 , B + J (0)] without losing generality. In the following part, we will frequently use this range.

Denote u := x glyph[latticetop] t θ, u ∗ = x glyph[latticetop] t θ ∗ . According to Equation 7, we know that:

<!-- formula-not-decoded -->

Here the first line is from the definition of g and Reg ( θ ) , the second line is due to Taylor's Expansion, the third line is from the fact that J ( u ∗ ) maximizes g ( v, u ∗ ) with respect to v , the fourth line is by calculus, the fifth line is from the assumption that 0 &lt; f ( ω ) ≤ B f , | f ′ ( ω ) | ≤ B f ′ and v ∈ [0 , B + J (0)] , the sixth line is because of J ′ ( u ) ∈ (0 , 1) , ∀ u ∈ R , and the seventh line is from the definition of u and u ∗ .

## B.2.2 Proof of Lemma 7

Proof. We take derivatives of l t ( θ ) , and we get:

<!-- formula-not-decoded -->

which directly proves the first inequality. For the second inequality, just notice that

<!-- formula-not-decoded -->

The only thing to point out is that f ( ω ) F ( ω ) and f ( ω ) 1 -F ( ω ) are all continuous for ω ∈ [ -B,B + J (0)] , as F ( ω ) is strictly increasing and thus 0 &lt; F ( ω ) &lt; 1 , ω ∈ R .

## B.2.3 Proof of Lemma 9

Proof. In the following part, we consider a situation that an epoch of n ≥ 2 rounds of pricing is conducted, generating l j ( θ ) as negative likelihood functions, j = 1 , 2 , . . . , n . Define a ' leave-one-

out 'negative log-likelihood function and let

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

Based on this definition, we know that ˜ θ i is independent to l i ( θ ) given historical data, and that ˜ θ i are identically distributed for all i = 1 , 2 , 3 , . . . , n .

In the following part, we will firstly propose and proof the following inequality:

<!-- formula-not-decoded -->

where ˆ θ is the short-hand notation of ˆ θ k as we do not specify the epoch k in this part. We now cite a lemma from Koren and Levy [2015]:

Lemma 13. Let g 1 , g 2 be 2 convex function defined over a closed and convex domain K ⊆ R d , and let x 1 = arg min x ∈K g 1 ( x ) and x 2 = arg min x ∈K g 2 ( x ) . Assume g 2 is locally δ -strongly-convex at x 1 with respect to a norm || · || . Then, for h = g 2 -g 1 we have

<!-- formula-not-decoded -->

Here || · || ∗ denotes a dual norm.

The following is a proof of this lemma.

Proof. (of Lemma 13) According to convexity of g 2 , we have:

<!-- formula-not-decoded -->

According to strong convexity of g 2 at x 1 , we have:

<!-- formula-not-decoded -->

Add Equation (23) and (24), and we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(25)

The first step is trivial. The second step is a sequence of g 2 = g 1 + h . The third step is derived by the following 2 first-order optimality conditions: ∇ g 1 ( x 1 ) glyph[latticetop] ( x 1 -x 2 ) ≤ 0 , and ∇ g 2 ( x 2 ) glyph[latticetop] ( x 2 -x 1 ) ≤ 0 . The fourth step is derived from Holder's Inequality:

<!-- formula-not-decoded -->

Therefore, the lemma holds.

In the following part, we will set up a strongly convex function of g 2 . Denote H = ∑ n t =1 x t x glyph[latticetop] t . From Lemma 7, we know that

<!-- formula-not-decoded -->

Here ˆ L ( θ ) is the short-hand notation of ˆ L k ( θ ) as we do not specify k in this part. Since we do not know if H is invertible, i.e. if a norm can be induced by H , we cannot let g 2 ( θ ) = ˆ L ( θ ) . Instead, we change the variable as follows:

We first apply singular value decomposition to H , i.e. H = UΣU glyph[latticetop] , where U ∈ R d × r , U glyph[latticetop] U = I r , Σ = diag { λ 1 , λ 2 , . . . , λ r } glyph[follows] 0 . After that, we introduce a new variable η := U glyph[latticetop] θ . Therefore, we have θ = Uη + V glyph[epsilon1] , where V ∈ R d × ( d -r ) , V glyph[latticetop] V = I d -r , V glyph[latticetop] U = 0 is the standard orthogonal bases of the null space of U , and glyph[epsilon1] ∈ R ( d -r ) . Similarly, we define ˜ η i = U glyph[latticetop] ˜ θ i and ˆ η = U glyph[latticetop] ˆ θ . According to these, we define the following functions:

<!-- formula-not-decoded -->

Now we prove that ˆ F ( η ) is locally-strongly-convex. Similar to the proof of Lemma 7, we have:

<!-- formula-not-decoded -->

That is to say, ˆ F ( η ) is locally C down n -strongly convex w.r.t Σ at η . Similarly, we can verify that ˜ F i ( η ) is convex (not necessarily strongly convex). Therefore, according to Lemma 13, let g 1 ( η ) = ˜ F i ( η ) , g 2 ( η ) = ˆ F ( η ) , and then x 1 = ˜ η i = U glyph[latticetop] ˜ θ i , x 2 = ˆ η = U glyph[latticetop] ˆ θ . Therefore, we have:

<!-- formula-not-decoded -->

Now let us show the validation of this theorem:

<!-- formula-not-decoded -->

And thus we have

<!-- formula-not-decoded -->

Thus the Inequality 22 is proved. After that, we have:

<!-- formula-not-decoded -->

Thus we has proved that E h [ L ( ˜ θ n )] -L ( θ ∗ ) ≤ C exp C down · d n . Notice that ˜ θ n is generated by optimizing the leave-one-out likelihood function ˜ L n ( θ ) = ∑ n -1 j =1 l j ( θ ) , which does not contain l n ( θ ) , and that the expected likelihood function L ( θ ) does not depend on any specific result occurring in this round. That is to say, every term of this inequality is not related to the last round ( x n , v n , ✶ n ) at all. In other words, this inequality is still valid if we only conduct this epoch from round 1 to ( n -1) .

Now let n = τ +1 , and then we know that θ τ +1 θ . Therefore, the theorem holds.

˜ = ˆ

## B.3 Proof of Lower bound in Section 5.3

Proof. We assume a fixed u ∗ such that x glyph[latticetop] θ ∗ = u ∗ , ∀ x ∈ D . In other words, we are considering a non-context setting. Therefore, we can define a policy as Ψ : { 0 , 1 } t → R + , t = 1 , 2 , . . . that does not observe x t at all. Before the proof begins, we firstly define a few notations: We denote Φ σ ( ω ) and p σ ( ω ) as the CDF and PDF of Gaussian distribution N (0 , σ 2 ) , and the corresponding J σ ( u ) = arg max v v (1 -Φ σ ( v -u )) as the pricing function.

Since we have proved that J ′ ( u ) ∈ (0 , 1) for u ∈ R in Appendix B.2.2, we have the following lemma:

Lemma 14. u -J σ ( u ) monotonically increases as u ∈ (0 , + ∞ ) , ∀ σ &gt; 0 . Also, we know that J σ (0) &gt; 0 , ∀ σ &gt; 0 .

Now consider the following cases: σ 1 = 1 , σ 2 = 1 -f ( T ) , where lim T →∞ f ( T ) = 0 , f ′ ( T ) &lt; 0 , 0 &lt; f ( T ) &lt; 1 2 . We will later determine the explicit form of f ( T ) .

Suppose u ∗ satisfies J σ 1 ( u ∗ ) = u ∗ . Solve it and get u ∗ = √ π 2 . Therefore, we have u ∈ (0 , u ∗ ) ⇔ J 1 ( u ) &gt; u , and u ∈ ( u ∗ , + ∞ ) ⇔ J 1 ( u ) &lt; u . As a result, we have the following lemma.

Lemma 15. For any σ ∈ ( 1 2 , 1) , we have:

<!-- formula-not-decoded -->

Proof. Firstly, we have:

<!-- formula-not-decoded -->

When σ ∈ ( 1 2 , 1) , we know u ∗ σ &gt; u ∗ . Since J 1 ( u ∗ ) = u ∗ and that u ∈ ( u ∗ , + ∞ ) ⇔ J 1 ( u ) &lt; u , we have u ∗ σ &gt; J 1 ( u ∗ σ ) . Hence

<!-- formula-not-decoded -->

Therefore, without losing generality, we assume that for the problem parameterized by σ 2 , the price v ∈ (0 , u ∗ ) . To be specific, suppose v ∗ ( σ ) = J σ ( u ∗ ) . Define Ψ t +1 : [0 , 1] t → (0 , u ∗ ) as any policy that proposes a price at time t +1 . Define Ψ = { Ψ 1 , Ψ 2 , . . . , Ψ T -1 , Ψ T } .

Define the sequence of price as V = { v 1 , v 2 , . . . , v T -1 , v T } , and the sequence of decisions as ✶ = { ✶ 1 , ✶ 2 , . . . , ✶ T -1 , ✶ T } . Denote V t = { v 1 , v 2 , . . . , v t , } . Define the probability (also the likelihood if we change u ∗ to other parameter u ):

Define a random variable Y t ∈ { 0 , 1 } t , Y t ∼ Q V t ,σ t and one possible assignment y t = { ✶ 1 , ✶ 2 , . . . , ✶ t -1 , ✶ t } . For any price v and any parameter σ , define the expected reward function as r ( v, σ ) := vΦ σ ( u ∗ -v ) . Based on this, we can further define the expected regret Regret( σ, T, Ψ ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now we have the following properties:

<!-- formula-not-decoded -->

Proof. 1. We have:

<!-- formula-not-decoded -->

Since v ∈ (0 , u ∗ ) , we have ( v 2 -u ∗ v -2 σ 2 ) &lt; -2 σ 2 . Also, since σ ∈ (1 / 2 , 1) , we have p σ ( u ∗ -v ) &gt; 1 √ 2 π · e -( u ∗ ) 2 2 · (1 / 2) 2 = 1 √ 2 πe π &gt; 0 . 017 . Therefore, we have

<!-- formula-not-decoded -->

;

As a result, we have:

<!-- formula-not-decoded -->

2. According to Equation 32, we know that:

<!-- formula-not-decoded -->

For u ∈ ( u ∗ , + ∞ ) , J 1 ( u ) &lt; u . According to Lemma 14, we have:

<!-- formula-not-decoded -->

Also, for u ∈ ( u ∗ , u ∗ σ ) , we have:

<!-- formula-not-decoded -->

Therefore, we have:

<!-- formula-not-decoded -->

## 3. This is because:

<!-- formula-not-decoded -->

In the following part, we will propose two theorems, which balance the cost of learning and that of uncertainty. This part is mostly similar to [BR12] Section 3, but we adopt a different family of demand curves here.

Theorem 17 (Learning is costly) . Let σ ∈ (1 / 2 , 1) and v t ∈ (0 , u ∗ ) , and we have: K ( Q V, 1 ; Q V,σ ) &lt; 9900(1 -σ ) 2 Regret(1 , T, Ψ ) . (36) Here v t = Ψ ( y t -1 ) , t = 1 , 2 , . . . , T .

Proof. First of all, we cite the following lemma that would facilitate the proof.

Lemma 18 (Corollary 3.1 in Taneja and Kumar, 2004) . Suppose B 1 and B 2 are distributions of Bernoulli random variables with parameters q 1 and q 2 , respectively, with q 1 , q 2 ∈ (0 , 1) . Then,

<!-- formula-not-decoded -->

According to the definition of KL-divergence, we have:

<!-- formula-not-decoded -->

For each term of the RHS, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here inequality (**) above is proved as follows: since v s ∈ (0 , u ∗ ) as is assumed, we have:

<!-- formula-not-decoded -->

As a result, we have 1 Φ σ ( u ∗ -v s )(1 -Φ σ ( u ∗ -v s )) ≤ 1 0 . 9939 × 0 . 0061 = 164 . 7988 ≤ 165 . Therefore, by summing up all s , we have:

<!-- formula-not-decoded -->

which concludes the proof.

Theorem 19 (Uncertainty is costly) . Let σ ≤ 1 -T -1 4 , and we have:

<!-- formula-not-decoded -->

Here v t = Ψ ( y t -1 ) , t = 1 , 2 , . . . , T .

Proof. First of all, we cite a lemma that would facilitate our proof:

Lemma20. Let Q 0 and Q 1 be two probability distributions on a finite space Y ; with Q 0 ( y ) , Q 1 ( y ) &gt; 0 , ∀ y ∈ Y . Then for any function F : Y → { 0 , 1 } ,

<!-- formula-not-decoded -->

where K ( Q 0 ; Q 1 ) denotes the KL-divergence of Q 0 and Q 1 .

Define two intervals of prices:

<!-- formula-not-decoded -->

Note that C 1 and C 2 are disjoint, since | u ∗ -J σ ( u ∗ ) | ≥ 2 5 | 1 -σ | = 2 5 T 1 / 2 according to Lemma 16 Property 2. Also, for v ∈ (0 , u ∗ ) \ C 2 , the regret is large according to Lemma 16 Property 1, because:

<!-- formula-not-decoded -->

Then, we have:

<!-- formula-not-decoded -->

According to Theorem 17 and Theorem 19, we can then prove Theorem 12. Let σ = 1 -T -1 4

<!-- formula-not-decoded -->

Thus Theorem 12 is proved valid.

## C More Discussions

## C.1 Dependence on B and Noise Variance

Here we use a concrete example to analyze the coefficients of regret bounds. Again, we assume that N t ∼ N (0 , σ 2 ) . Notice that both C s and C a have a component of C exp C down . In order to analyze C exp C down , we define a hazard function denoted as λ ( ω ) with ω ∈ R :

<!-- formula-not-decoded -->

where Φ 1 and p 1 are the CDF and PDF of standard Gaussian distribution. The concept of hazard function comes from the area of survival analysis . From Equation 11 and 13, we plug in Equation 38 and get:

<!-- formula-not-decoded -->

In Lemma 21, we will prove that λ ( ω ) is exponentially small as ω → + ∞ , and is asymptotically close to -ω as ω →-∞ . Therefore, C down is exponentially small and C exp is quadratically large with respect to B/σ . Although we assume that B and σ are constant, we should be alert that the scale of B/σ can be very large as σ goes to zero, i.e. as the noise is 'insignificant'. In practice (especially when T is finite), this may cause extremely large regret at the beginning. A 'Shallow Pricing' method introduced by Cohen et al. [2020] (as well as other domain-cutting methods in contextual searching) may serve as a good pre-process as it frequently conducts bisections to cut the feasible region of θ ∗ with high probability. According to Theorem 3 in Cohen et al. [2020], their Shallow Pricing algorithm will bisect the parameter set for at most logarithmic times to ensure that B σ has been small enough (i.e. upper-bounded by O ( poly log( T )) ). However, this does not necessarily means that we can use a O (log T ) -time pre-process to achieve the same effect, since they run the algorithm throughout the session while we only take it as a pre-process. Intuitively, at least under the adversarial feature assumption, we cannot totally rely on a few features occurring at the beginning (as they might be misleading) to cut the parameter set once and for all. A mixture approach of Shallow Pricing and EMLP/ONSP might work, as the algorithm can detect whether current B σ is larger than a threshold of bisection. However, this requires new regret analysis as the operations parameter domain are changing over time. Therefore, we claim in Section 7 that the regret bound is still open if σ = Θ ( T -α ) for α ∈ (0 , 1) .

Lemma 21 (Properties of λ ( ω ) ) . For λ ( ω ) := p 1 ( ω ) 1 -Φ 1 ( ω ) , we have:

<!-- formula-not-decoded -->

Proof. We prove the Lemma 21 sequentially:

1. We have:

<!-- formula-not-decoded -->

Therefore, it is equivalent to prove that p 1 ( -ω ) -ωΦ 1 ( -ω ) &gt; 0 .

Suppose f ( ω ) = p 1 ( ω ) + ωΦ 1 ( ω ) . We now take its derivatives as follows:

<!-- formula-not-decoded -->

Therefore, we know that f ( ω ) monotonically increases in R . Additionally, since we have:

<!-- formula-not-decoded -->

Therefore, we know that f ( ω ) &gt; 0 , ∀ ω ∈ R , and as a result, λ ′ ( ω ) &gt; 0 .

2. We have:
3. We only need to prove that

Actually, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

4.

<!-- formula-not-decoded -->

Thus the lemma holds.

## C.2 Algorithmic Design

## C.2.1 Probit and Logistic Regressions

A probit/logit model is described as follows: a Boolean random variable Y satisfies the following probabilistic distribution: P [ Y = 1 | X ] = F ( X glyph[latticetop] β ) , where X ∈ R is a random vector, β ∈ R is a parameter, and F is the cumulative distribution function (CDF) of a (standard) Gaussian/logistic distribution. In our problem, we may treat ✶ t as Y , [ x t glyph[latticetop] , v t ] glyph[latticetop] as X and [ θ ∗glyph[latticetop] , -1] glyph[latticetop] as β , which exactly fits this model if we assume the noise as Gaussian or logistic. Therefore, ˆ θ k = arg min θ ˆ L k ( θ ) can be solved via the highly efficient implementation of generalized linear models, e.g., GLMnet, rather than resorting to generic tools for convex programming. As a heuristic, we could leverage the vast body of statistical work on probit or logit models and adopt a fully Bayesian approach that jointly estimates θ and hyper-parameters of F . This would make the algorithm more practical by eliminating the need to choose the hyper-parameters when running this algorithm.

## C.2.2 Advantages of EMLP over ONSP.

For the stochastic setting, we specifically propose EMLP even though ONSP also works. This is because EMLP only 'switch' the pricing policy ˆ θ for log T times. This makes it appealing in many applications (especially for brick-and-mortar sales) where the number of policy updates is a bottleneck. In fact, the iterations within one epoch can be carried out entirely in parallel.

## C.2.3 Agnostic Dynamic Pricing: Explorations versus Exploitation

At the moment, the proposed algorithm relies on the assumption of a linear valuation function (see Appendix C.3 for more discussion on problem modeling). It will be interesting to investigate the settings of model-misspecified cases and the full agnostic settings. The key would be to exploit the structural feedback in model-free policy-evaluation methods such as importance sampling. The main reason why we do not explore lies in the noisy model: essentially we are implicitly exploring a higher (permitted) price using the naturally occurring noise in the data. In comparison, there is another problem setting named 'adversarial irrationality' where some of the customers will valuate the product adaptively and adversarially 10 . Existing work Krishnamurthy et al. [2021] adopts this setting and shows a linear regret dependence on the number of irrational customers, but they consider a different loss function (See Related Works Section).

10 An adaptive adversary may take actions adversarially in respond to the environmental changes. In comparison, what we allow for the 'adversarial features' is actually chosen by an oblivious adversary before the interactions start.

## C.3 Problem Modeling

## C.3.1 Noise Distributions

In this work, we have made four assumptions on the noise distribution: strict log-concavity, 2 nd -order smooth, known, and i.i.d.. Here we explain each of them specifically.

- The assumption of knowing the exact F is critical to the regret bound: If we have this knowledge, then we achieve O (log T ) even with adversarial features; otherwise, an Ω ( √ T ) regret is unavoidable even with stochastic features.
- The strictly log-concave distribution family includes Gaussian and logistic distributions as two common noises. In comparison, Javanmard and Nazerzadeh [2019] assumes log-concavity that further covers Laplacian, exponential and uniform distributions. Javanmard and Nazerzadeh [2019] also considers the cases when (1) the noise distribution is unknown but log-concave, and (2) the noise distribution is zero-mean and bounded by support of [ -δ, δ ] . For case (1), they propose an algorithm with regret O ( √ T ) and meanwhile prove the same lower bound. For case (2), they propose an algorithm with linear regret.
- The assumption that F is 2 nd -order smooth is also assumed by Javanmard and Nazerzadeh [2019] by taking derivatives f ′ ( v ) and applying its upper bound in the proof. Therefore, we are still unaware of the regret bound if the noise distribution is discrete, where a lower bound of Ω ( √ T ) can be directly applied from Kleinberg and Leighton [2003].
- We even assume that the noise is identically distributed. However, the noise would vary among different people. The same problem happens on the parameter θ ∗ : can we assume different people sharing the same evaluation parameter? We may interpret it in the following two ways, but there are still flaws: (1) the 'customer' can be the public, i.e. their performance is quite stable in general; or (2) the customer can be the same one over the whole time series. However, the former explanation cannot match the assumption that we just sell one product at each time, and the latter one would definitely undermine the independent assumption of the noise: people would do 'human learning' and might gradually reduce their noise of making decisions. To this extent, it is closer to the fact if we assume noises as martingales. This assumption has been stated in Qiang and Bayati [2016].

## C.3.2 Linear Valuations on Features

There exist many products whose prices are not linearly dependent on features. One famous instance is a diamond: a kilogram of diamond powder is very cheap because it can be produced artificially, but a single 5-carat (or 1 gram) diamond might cost more than $100,000. This is because of an intrinsic non-linear property of diamond: large ones are rare and cannot be (at least easily) compound from smaller ones. Another example lies in electricity pricing [Joskow and Wolfram, 2012], where the more you consume, the higher unit price you suffer. On the contrary, commodities tend to be cheaper than retail prices. These are both consequences of marginal costs: a large volume consuming of electricity may cause extra maintenance and increase the cost, and a large amount of purchasing would release the storage and thus reduce their costs. In a word, our problem setting might not be suitable for those large-enough features, and thus an upper bound of x glyph[latticetop] θ becomes a necessity.

## C.4 Ex Ante v.s. Ex Post Regrets

In this work, we considered the ex ante regret Reg ea = ∑ T t =1 max θ E [ v θ t · ✶ ( v θ t ≤ w t )] -E [ v t · ✶ ( v t ≤ w t )] , where v θ t = J ( x glyph[latticetop] t θ ) is the greedy price with parameter θ and w t = x glyph[latticetop] t θ ∗ + N t is the realized random valuation. The ex post definition of the cumulative regret, i.e., Reg ep = max θ ∑ T t =1 v θ t ✶ ( v θ t ≤ w t ) -v t ✶ ( v t ≤ w t ) makes sense, too. Note that we can decompose E [ Reg ep ] = Reg ea + E [max θ ∑ T t =1 v θ t ✶ ( v θ t ≤ w t ) -∑ T t =1 v θ ∗ t ✶ ( v θ ∗ t ≤ w t )] . While it might be the case that the second term is Ω ( √ dT ) as the reviewer pointed out, it is a constant independent of the algorithm. For this reason, we believe using Reg ea is without loss of generality, and it reveals more nuanced performance differences of different algorithms.

For an ex post dynamic regret, i.e., Reg d = ∑ T t =1 w t -v t · ✶ ( v t ≤ w t ) , it is argued in Cohen et al. [2020] that any policy must suffer an expected regret of Ω ( T ) (even if θ ∗ is known). We may also present a good example lies in N t ∼ N (0 , 1) , x glyph[latticetop] t θ ∗ = √ π 2 where the optimal price is √ π 2

as well but the probability of acceptance is only 1/2, and this leads to a constant per-step regret of 1 2 √ π 2 .

## C.5 Ethic Issues

A field of study lies in 'personalized dynamic pricing' [Aydin and Ziya, 2009, Chen and Gallego, 2021], where a firm makes use of information of individual customers and sets a unique price for each of them. This has been frequently applied in airline pricing [Krämer et al., 2018]. However, this causes first-order pricing discrimination. Even though this 'discrimination' is not necessarily immoral, it must be embarrassing if we are witted proposing the same product with different prices towards different customers. For example, if we know the coming customer is rich enough and is not as sensitive towards a price (e.g., he/she has a variance larger than other customers), then we are probably raising the price without being too risky. Or if the customer is used to purchase goods from ours, then he or she might have a higher expectation on our products (e.g., he/she has a θ = aθ ∗ , a &gt; 1 ), and we might take advantage and propose a higher price than others. These cases would not happen in an auction-based situation (such as a live sale), but might frequently happen in a more secret place, for instance, a customized travel plan.