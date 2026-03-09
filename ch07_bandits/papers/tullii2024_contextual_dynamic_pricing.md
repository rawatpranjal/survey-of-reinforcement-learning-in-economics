## Improved Algorithms for Contextual Dynamic Pricing

Matilde Tullii ∗

FairPlay Team, CREST, ENSAE

Nadav Merlis

FairPlay Team, CREST, ENSAE

Solenne Gaucher ∗ FairPlay Team, CREST, ENSAE

Vianney Perchet

FairPlay Team, CREST, ENSAE - Criteo AI Lab

## Abstract

In contextual dynamic pricing, a seller sequentially prices goods based on contextual information. Buyers will purchase products only if the prices are below their valuations. The goal of the seller is to design a pricing strategy that collects as much revenue as possible. We focus on two different valuation models. The first assumes that valuations linearly depend on the context and are further distorted by noise. Under minor regularity assumptions, our algorithm achieves an optimal regret bound of ˜ O ( T 2 / 3 ) , improving the existing results. The second model removes the linearity assumption, requiring only that the expected buyer valuation is β -Hölder in the context. For this model, our algorithm obtains a regret ˜ O ( T d +2 β/d +3 β ) , where d is the dimension of the context space.

## 1 Introduction

Setting a price and devising a strategy to dynamically adjust it poses a fundamental challenge in revenue management. This problem, known as dynamic pricing or online posted price auction, finds applications across various industries and has received significant attention from economists, operations researchers, statisticians, and machine learning communities. In this problem, a seller sequentially offers goods to arriving buyers by presenting a one-time offer at a specified price. If the offered price falls below the buyer's (unknown) valuation of the item, a transaction occurs, and the seller obtains the posted price as revenue. Conversely, if the price exceeds the buyer's valuation, the transaction fails, resulting in zero gain for the seller. Crucially, the seller solely receives binary feedback indicating whether the trade happened. Her objective is to learn from this limited feedback how to set prices that maximize her cumulative gains while ensuring that transactions take place. In this paper, we study the problem of designing an adaptive pricing strategy, when the seller can rely on contextual information, describing the product itself, the marketing environment, or the buyer.

While this problem has been extensively studied, previous results either rely on strong assumptions on the structure of the problem, greatly limiting the applicability of such approaches, or achieve sub-optimal regret bounds. In this work, we aim to improve both aspects-achieving better regret bounds while making minimal assumptions about the problem. Specifically, we study two different models for the valuation of buyers as a function of the context: 1) linear valuations , where the item valuation of buyers is an unknown noisy linear function of the context; and 2) non-parametric valuations , where the valuation is given by an unknown Hölder-continuous function of the contextual information, perturbed by noise.

∗ Equal contribution.

Table 1: Summary of existing regret bounds. g is the expected valuation function, F is the c.d.f. of the noise, and π ( x, p ) is the reward for price p and context x , defined in Section 2.1.

| Model           | Noise Assumption                                                             | Regret                                                                  |
|-----------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Linear          | F is known                                                                   | ˜ O ( T 2 / 3 ) [11]                                                    |
| Linear          | F is known or parametric, and log-concave                                    | ˜ O ( T 1 / 2 ) [15]                                                    |
| Linear          | F has m -th order derivatives                                                | ˜ O ( T 2 m +1 / 4 m - 1 ) [14]                                         |
| Linear          | F ′′ is bounded                                                              | ˜ O ( T 2 / 3 ) ∨ ∥ θ - ̂ θ ∥ 1 T [22]                                   |
| Linear          | F is Lipschitz                                                               | ˜ O ( T 3 / 4 ) [14, 21], ˜ O ( T 2 / 3 ) [this work] Ω( T 2 / 3 ) [31] |
| Linear          | Bounded noise                                                                | ˜ O ( T 3 / 4 ) , [31]                                                  |
| Non- parametric | π ( x, · ) is quadratic around its maximum for all x , F and g are Lipschitz | ˜ O ( T d +2 / d +4 ) [10] Ω( T d +2 / d +4 ) [10]                      |
| Non- parametric | F is Lipschitz and g is Hölder                                               | ˜ O ( T d +2 β / d +3 β ) [this work]                                   |

## 1.1 Related Work

Dynamic pricing has been extensively studied for half a century [19, 26], leading to rich research on both theoretical and empirical fronts. For comprehensive surveys on the topic, we refer the readers to [6, 12]. While earlier works assumed that the buyer's valuations are i.i.d. [18, 5, 16, 9], recent research has increasingly focused on feature-based (or contextual) pricing problems. In this scenario, product value and pricing strategy depend on covariates. Pioneering works considered valuations depending deterministically on the covariates. Linear valuations have been the most studied [3, 15, 11, 20], yet a few authors have also explored non-parametric valuations [24].

Recent works have extended these methods to random valuations, mainly assuming that valuations are given by a function of the covariate, distorted by an additive i.i.d. noise. As this poses more challenges, authors have mostly focused on the simplest case of linear valuation functions, under additional assumptions. Initial studies assumed knowledge of the noise distribution [11, 15, 30]. This assumption was later relaxed, albeit with additional regularity requirements on the cumulative distribution function (c.d.f.) of the noise and/or the reward function [14, 22], and then again by [31], that achieves a regret bound of ˜ O ( T 3 / 4 ) for linear valuations, while assuming only the boundedness of the noise. Closest to our work [14, 21] also focus on the case in which the only regularity required is the Lipschitzness of the CDF. Their approaches show some similarities with our work but still achieve suboptimal regret rates. A more detailed comparison between ours an their algorithms is presented later on in the paper. Other parametric models have been explored, with, for example, generalized linear regression models [28], though they also require strong assumptions, including quadratic behavior of the reward function around each optimal price. Few works have considered non-deterministic valuations with non-parametric valuation functions. Among those, [10] consider Lipschitz-continuous valuation functions of d -dimensional covariates. They achieve a regret of order ˜ O ( T d +2 / d +4 ) , assuming again quadratic behaviour around optimal prices. We refer to Table 1 for a comprehensive comparison between different previous works, their assumptions and regret bounds.

To improve on previous results, we design algorithms that share information on the noise distribution across different contexts. This idea relates to methods used in cross-learning , a research direction stemming from online bandit problems with graph feedback [23, 2]. In this framework, introduced by [4] and further studied in [27], when choosing to take action i in context x t , the agent observes the reward r i ( x t ) along with rewards r i ( x ′ t ) associated with other contexts x t ′ . Our algorithms leverage similar principles to learn information usable across different contexts. However, compared to the typical problems addressed by cross-learning methods (e.g., first-price auctions, sleeping bandits, multi-armed bandits with exogenous costs), the contextual dynamic problem is more complex due to the intricate dependence of the reward on the unknown valuation function.

## 1.2 Outline and Contributions

In this work, we tackle the problem of dynamic pricing with contextual information. We consider two models for the expected valuations of the buyer, assuming respectively that they are given by a

linear function, or by a non-parametric function. For both models, we present a general algorithmic scheme called VALUATION APPROXIMATION - PRICE ELIMINATION (VAPE), and provide bounds on its regret in both models:

- In the linear model, we obtain a regret of ˜ O ( T 2 / 3 ) , assuming only that the c.d.f. of the noise is Lipschitz. This concludes an extensive series of papers on the topic, as it establishes the minimax optimal regret rate and proves it is attainable under minimal assumptions.
- In the non-parametric model, we obtain a regret rate of ˜ O ( T d +2 β / d +3 β ) , assuming only the Lipschitz-continuity of the noise and the Hölder one of the valuation function. This result is the first of its kind under such minimal assumptions.

The rest of the paper is organized as follows. We begin by presenting the model and summarizing the notations used throughout the paper in Section 2.1. Section 2.2 outlines our assumptions and compares them with those in previous works. In Section 2.3, we discuss the main sources of difficulty of the problem and highlight the importance of information sharing in contextual dynamic pricing. In Section 3, we present our algorithmic scheme, VAPE, and provide an initial informal result bounding its regret. Then, in Section 4, we apply this algorithmic scheme to linear valuations and provide a bound on its regret. Finally, in Section 5, we extend this algorithm to non-parametric valuations.

## 2 Preliminaries

## 2.1 Model and Notations

The problem of dynamic pricing with contextual information is formalized as follows. At each step t ≤ T , a context x t ∈ R d , describing a sale session (product, customer, and context) is revealed. The customer assigns a hidden valuation y t to the product, and the seller proposes a price p t , based on x t and on historical sales records. If p t ≤ y t , the trade is successful, and the seller receives a reward y t ; otherwise the trade fails. The seller's only feedback is the binary outcome o t = 1 { p t ≤ y t } . We assume that the seller's valuation is given by

<!-- formula-not-decoded -->

where g : R d ↦→ R is the valuation function, and ξ t is a centered, bounded, i.i.d. noise term, independent of x t and of ( x s , p s , ξ s ) s&lt;t . In the present paper, we consider successively linear and non-parametric valuation functions g in Sections 4 and 5. The seller's objective is to maximize the sum of her cumulative earnings. We denote by π ( p, x t ) the expected reward of the seller if she posts a price p for a product described by covariate x t :

<!-- formula-not-decoded -->

Adopting the terminology of the literature on multi-armed bandits, we measure the performance of our algorithm and the difficulty of the problem through the regret R T , defined as

<!-- formula-not-decoded -->

Notations Throughout this paper, we make use of the following notation. We denote by ∥·∥ the Euclidean norm. For all A,B ∈ R , we denote by J A,B K the set { A,A + 1 , . . . , B } . R T ≲ B T (resp. R T = ˜ O ( B T ) ) means that there exists a (possibly problem-dependent) constant C such that R T ≤ CB T (resp. R T = O (log( T ) C B T ) ). Finally, f and F denote the p.d.f. and c.d.f. of the noise, respectively.

## 2.2 Assumptions

For both valuation models, we make the following assumptions on the context and noise distribution.

Assumption 1. Contexts and expected valuations are bounded: ∥ x t ∥ 2 ≤ B x and | g ( x t ) | ≤ B g a.s.

This assumption is classical in contextual dynamic pricing problems. We underline that contexts do not need to be random. In particular, they can be chosen by an adaptive adversary, aware of the

seller's strategy, and based on past realizations of ( x s , p s , ξ s ) s&lt;t . Assumption 1 is milder than the i.i.d. context assumption appearing in [14, 28, 10].

Dynamic pricing strategies mostly assume that the buyer's valuations are bounded. To enforce this, we assume that the noise is bounded; moreover, we assume that its c.d.f. Lipschitz continuous.

Assumption 2. The noise ξ t is bounded: | ξ t | ≤ B ξ a.s. Moreover, its c.d.f. F is L ξ -Lispchitz continuous: for all ( δ, δ ′ ) ∈ R d , | F ( δ ) -F ( δ ′ ) | ≤ L ξ | δ -δ ′ | .

Assumption 2 is weaker than most of the assumptions in related works. For example, [15] require both F and 1 -F to be log-concave. [14] assume that F has m -th derivative, and that δ -1 -F ( δ ) / F ′ ( δ ) is greater than some positive constant for all δ , achieving a regret of order ˜ O ( T 2 m +1 / 4 m -1 ) . In the case m = 1 , they propose a different algorithm, reaching a regret ˜ O ( T 3 / 4 ) . [22] consider Lipschitzcontinuous noise, under the additional assumption that, for every x , p ∗ ( x ) ∈ arg max p π ( x, p ) is unique, and that F ′′ is bounded. [10] assume quadratic behaviour around every maxima: for every x , p ∗ ( x ) ∈ arg max p π ( x, p ) , p ∗ ( x ) is unique, and for all p , C ( p ∗ ( x ) -p ) 2 ≤ π ( x, p ∗ ( x )) -π ( x, p ) ≤ C ′ ( p ∗ ( x ) -p ) 2 for some constants C, C ′ . The only work considering non-Lipschitz c.d.f. is [31]; however, they achieve a higher regret bound of ˜ O ( T 3 / 4 ) .

## 2.3 Information Sharing in Contextual Dynamic Pricing

For δ ∈ R , we denote D ( δ ) = P ( ξ t ≥ δ ) = 1 -F ( δ ) , the demand function associated with the noise ξ t . Note that, under Assumption 2, D is L ξ -Lipschitz continuous. Straightforward computations show that, for any price increment δ ∈ R , the expected reward corresponding to the price p = g ( x t ) + δ in the context x t is given by

<!-- formula-not-decoded -->

Equation (2) highlights the intricate roles played by the expected valuation g ( x t ) and the price increment δ = p -g ( x t ) in the reward. An immediate consequence is that the optimal price increment δ depends on the value of g ( x t ) . Intuitively, if g ( x t ) is large, the seller should choose δ to be small to ensure a high probability D ( δ ) to perform a trade. However, for smaller values of g ( x t ) , the seller might prefer a larger δ to ensure significant rewards when a trade occurs. Importantly, there is no explicit relationship between the optimal increments δ for different valuations g ( x t ) , so knowing the optimal price for a value g ( x t ) does not allow optimal pricing for a different value g ( x t ′ ) .

This reasoning suggests that the optimal price increment may span a wide range of values as the expected valuation g ( x t ) varies. Unfortunately, as is typical in bandit problems, it is necessary to estimate the reward function around the optimal price with high precision to ensure low regret. Consequently, solving the dynamic pricing problem may entail estimating the demand function precisely across a broad range of price increments. This marks a significant departure from noncontextual dynamic pricing and non-parametric bandit problems, where precise estimation of the reward function is often only necessary around its (single) maximum. Thus, the contextual dynamic pricing problem might be more challenging than its non-contextual counterpart, potentially leading to higher regret. This intuition is supported by the fact that straightforward application of basic bandit algorithms, even in the most simple linear model, leads to regret higher than the rate of order ˜ O ( T 2 / 3 ) encountered in non-contextual dynamic pricing problems, as we show in the following discussion.

Naïve bandit algorithms for contextual dynamic pricing. As a first attempt, one might apply a simple explore-then-commit algorithm. Such algorithms start with an exploration phase to obtain uniformly good estimates of both g and of the demand function D over a finite grid of price increments { δ k } k ∈K . Then, in a second exploitation phase, prices are set greedily to maximize the estimated reward. To bound the regret of this approach, note that uniform estimation of D over the grid { δ k } k ∈K with precision ϵ requires ϵ -2 |K| estimation rounds. Moreover, the Lipschitz continuity of the reward function implies a discretization error of order 1 / |K| . Classical arguments suggest that the regret would be at least T ( ϵ + 1 / |K| ) + |K| ϵ -2 , which is minimized for ϵ = 1 / |K| = T -1 / 4 . Thus, this approach would lead to a regret of order ˜ O ( T 3 / 4 ) .

Another approach, akin to that used in [10], involves partitioning the covariate space into bins and running independent algorithms for non-parametric bandits (such as CAB1 [17]) within each bin. Let us assume, for simplicity, contexts in [0 , 1] , and that we partition this segment into K bins. Then, the discretization error is 1 / K . Classical results show that the regret in one bin is ˜ O ( T 2 / 3 K ) , where

T K = T / K is the number of rounds in each bin. Consequently, the regret is ˜ O ( T / K + K × ( T / K ) 2 / 3 ) , which is minimized for K = T 1 / 4 , resulting in a regret ˜ O ( T 3 / 4 ) .

Thus, both approaches - using either independent bandit algorithms over binned contexts or common exploration rounds followed by an exploitation phase - suffer a regret of order T 3 / 4 in the linear model. This raises the question of whether this rate is optimal for the linear model, and if the contextual dynamic pricing problem is indeed more difficult than the non-contextual one. Strikingly, we show that this is not the case. We rely on an intermediate approach, based on regret-minimizing algorithms for each valuation level g ( x t ) that share information across different values of g ( x t ) . We show that it achieves an optimal regret rate of order ˜ O ( T 2 / 3 ) in the linear valuation model. Moreover, it achieves a rate of order ˜ O ( T d +2 β / d +3 β ) in the non-parametric valuation model under minimal assumptions.

## 3 Algorithmic Approach

In this section, we present the general algorithmic approach that we use to tackle dynamic pricing with covariates, called VALUATION APPROXIMATION - PRICE ELIMINATION (VAPE). Before presenting the full scheme, described in Algorithm 1, we start with some intuition that leads to its design. Then, we provide a first analysis of the regret of this algorithm.

## 3.1 Outline of the Algorithm

Equation (2) highlights how the reward is influenced by the expected valuation g ( x t ) and by the demand at the price increment δ = p t -g ( x t ) . To separate the effect of these terms, we estimate g and D independently. Hereafter, we assume that the valuations y t are bounded, in [ -B y , B y ] .

Estimation of g . To estimate g ( x t ) , we rely on the following observation: when prices p t are uniformly chosen from the interval [ -B y , B y ] , the random variable 2 B y ( o t -1 / 2) can serve as an unbiased estimate of g ( x t ) conditioned on x t . Given that 2 B y ( o t -1 / 2) is bounded, classical concentration results can be employed to bound the error of our estimates for g ( x t ) . Thus, in each round, we test whether our estimate of g ( x t ) is precise enough to ensure that the error g ( x t ) -̂ g ( x t ) is small. If this is not the case, we conduct a VALUATION APPROXIMATION round by setting a uniform price. In the next sections, we consider linear and non-parametric valuation functions, and we discuss how to ensure sufficient precision in a limited number of valuation approximation rounds.

Previous approaches for estimating valuation functions in the linear model include the regularized maximum-likelihood estimator [15, 30], which requires knowledge of the noise distribution. Another approach used in [22] relies on the relation between estimating a linear valuation function from binary feedback and the classical linear classification problem. The authors propose recovering the linear parameters θ through logistic regression; however, they do not provide an explicit estimation rate for θ . [20] use the EXP-4 algorithm to aggregate policies corresponding to different values of θ and F , thus circumventing the necessity to estimate them. In a similar vein, in the non-parametric valuation model, [10] avoid the need to estimate g ( x t ) by employing independent bandit algorithms for each (binned) value of x t . Closer to our method are the works of [14] and [21], who also set uniform prices to obtain unbiased estimates of the valuations. Nonetheless, their algorithms are significantly different from ours. First, they propose two-phased algorithms for which the phase length is set beforehand. Such an approach necessitates additional assumptions on how contexts are drawn; specifically, contexts are assumed to be i.i.d. from a distribution with a lower bound on the eigenvalues of the covariance matrix. This is needed to ensure that contexts observed in the first phase can represent the context distribution well. By contrast, our phases are adaptive, allowing our algorithm also to handle adversarial contexts and render these assumptions superfluous. Second, we obtain better regret rates by using piecewise-constant estimators, fitted in a regret-minimization sub-routine, as detailed in the next paragraph. On the other hand, [14] performs a phase of pure exploitation, relying on an estimate of the CDF F that is constructed using Kernel methods. [21], instead, re-frames the problem as a perturbed linear bandit, which exhibits a regret linear in the dimension. However, this dimension depends on the size of the discretization grid - which is horizon dependent - leading to worse rates.

Estimation of D . If the expected valuation g ( x t ) is known with sufficient precision, we can use it to estimate the demand function over a set of candidate price increments { δ k } k ∈K . More precisely, assume we set a price p t = ̂ g ( x t ) + δ k , and that | ̂ g ( x t ) -g ( x t ) | ≤ ϵ . Then, the observation o t can

| Algorithm 1 VALUATION APPROXIMATION - PRICE ELIMINATION (VAPE): General scheme   | Algorithm 1 VALUATION APPROXIMATION - PRICE ELIMINATION (VAPE): General scheme                                                                                   |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1:                                                                               | Input : Price increments { δ k } k ∈K , expected valuation precision err t ( x ) , reward confidence intervals [ LCB t ( k ) , UCB t ( k )] , parameters α , ϵ . |
| 2:                                                                               | while t ≤ T do                                                                                                                                                   |
| 3:                                                                               | if err t ( x t ) > ϵ then ▷ Valuation Approximation                                                                                                              |
| 4:                                                                               | Post a price p t ∼ U ([ - B y ,B y ])                                                                                                                            |
| 5:                                                                               | Use o t to improve the valuation estimator ̂ g ( x t )                                                                                                            |
| 6:                                                                               | else ▷ Price Elimination                                                                                                                                         |
| 7:                                                                               | A t ←{ k ∈ K : g t + δ k ∈ [0 ,B y ] }                                                                                                                           |
| 8:                                                                               | ̂ K t ←{ k ∈ A t : UCB t ( k ) ≥ max k ′ ∈A t LCB t ( k ′ ) }                                                                                                     |
| 9:                                                                               | Choose k t ∈ argmin k ∈K t N k t and post a price p t = ̂ g t + δ k t                                                                                             |
| 10:                                                                              | Update ̂ D k t t +1 , N k t t +1                                                                                                                                  |

be used as an almost unbiased estimate of the demand at level δ k , since

<!-- formula-not-decoded -->

Under Assumption 2, D is L ξ -Lipschitz, so the bias is of order L ξ ϵ . Then, relying on classical bandit techniques, we show that with high probability (for α small enough), | D ( δ k ) -̂ D k t | is of order L ξ ϵ + √ log(1 /α ) / N k t , where ̂ D k t is the average of the observations o t when setting a price p t = ̂ g ( x t ) + δ k , and N k t is the number of rounds in which we chose the price increment δ k up to round t . Importantly, to estimate ̂ D k t , we share information collected during all rounds we chose the increment δ k across all values of ̂ g ( x t ) ; this is necessary to obtain better regret rates. Then, using p t ̂ D k t as an estimate of the reward π ( x t , p t ) given the price p t = ̂ g ( x t ) + δ k , the error | π ( x t , p t ) -p t ̂ D k t | is of order B y ( L ξ ϵ + √ log(1 /α ) / N k t ) .

The PRICE ELIMINATION subroutine relies on the previous remark to select a price increment. For each increment δ k , we build a confidence bound [ LCB t ( δ k ) , UCB t ( δ k )] = [ p t ̂ D k t ± B y (2 L ξ ϵ + √ 2 log(1 /α ) / N k t )] for the reward of price p t = ̂ g ( x t ) + δ k . Then, we use a successive elimination algorithm [13, 25] to select a good increment. More precisely, we consider increments δ k such that UCB t ( δ k ) ≥ max l LCB t ( δ l ) , and we choose among these increments the increment δ k t that has been selected the least frequently. By doing so, we ensure to only select potentially optimal prices and gradually eliminate sub-optimal increments.

## 3.2 A First Bound on the Regret

Before discussing the application of the algorithmic scheme VAPE to linear and non-parametric valuation functions, we provide some intuition on regret bounds achievable through this scheme.

Claim 1. (Informal) Let δ k = kϵ for k ∈ K ≜ J ⌊ -B y -1 / ϵ ⌋ , ⌈ B y +1 / ϵ ⌉ K . Assume that, on a highprobability event, | ̂ g ( x t ) -g ( x t ) | ≤ ϵ for every round t where PRICE ELIMINATION is conducted. Then, on a high-probability event, the regret of VAPE verifies

<!-- formula-not-decoded -->

where T VA ( ϵ ) is a bound on the length of the VALUATION APPROXIMATION phase.

Claim 1 is proved in the Appendix by combining Equations (4) and (5), and Lemma 4. We provide a sketch of proof below. To bound on regret of VAPE using Claim 1, it will suffice to bound the length of the VALUATION APPROXIMATION phase, and prove high-probability error bounds on g ( x t ) .

Sketch of proof. Note that the regret in the VALUATION APPROXIMATION phase scales at most linearly with its length. Then, to prove Claim 1, it is enough to bound the regret during the PRICE ELIMINATION phase. We begin by bounding the sub-optimality gap of the price chosen at round t , showing that it is of order ϵ + √ log(1 /α ) / N k t t .

To do so, for p ∈ R , we define ∆ t ( x t , p ) = max p ′ π ( x t , p ′ ) -π ( x t , p ) the sub-optimality gap corresponding to price p . Recall that δ k t is the increment chosen at round t , i.e. that p t = ̂ g ( x t ) + δ k t .

Classical arguments from the bandit literature show that with high probability, for all k ∈ K , the upper and lower confidence bounds on π ( x t , ̂ g ( x t ) + δ k ) given by UCB t ( δ k ) and LCB t ( δ k ) are valid. Then, the optimal increment δ k ∗ t defined by k ∗ = arg max k ∈A t π ( x t , ̂ g ( x t ) + δ k ) belongs to the set of non-eliminated increments. Now, on the one hand, since UCB t ( δ k t ) ≥ LCB t ( δ k ∗ t ) , and since the confidence interval are valid, the gap π ( x t , ̂ g ( x t ) + δ k ∗ t ) -π ( x t , p t ) is of order ϵ + √ 2 log(1 /α ) / N k t t + √ 2 log(1 /α ) / N k ∗ t t . Our round-robin sampling scheme ensures that N k ∗ t t ≥ N k t t , so this bound is of order ϵ + √ log(1 /α ) / N k t t . On the other hand, our choice of grid { δ k } k ∈K , together with the Lipschitz-continuity of the reward in Assumption 2, imply that the cost ∆ t ( x t , ̂ g ( x t ) + δ k ∗ t ) of considering a discrete price grid is of order B y L ξ ϵ . Thus, at each round, the gap ∆ t ( x t , ̂ g ( x t )+ δ k t ) is at most of order ϵ + √ log(1 /α ) / N k t t (up to problem-dependent constants).

Now, let us decompose the regret of the PRICE ELIMINATION phase as follows:

<!-- formula-not-decoded -->

In order to bound ∑ t : k t = k ∆( x t , p t ) for k ∈ K , we begin by introducing further notations. Let us denote τ k 1 , . . . , τ k T the rounds in the PRICE ELIMINATION phase where we choose k t = k . We also define ∆ a = 2 -a and a such that ∆ a ≈ ϵ . For all a ≤ a , we also define t a such that the bound ϵ + √ log(1 /α ) / t a is of order ∆ a . Then, our previous reasoning implies that if i ≥ t a for some a ∈ { 1 , a } , it must be that ∆ t ( x t , p τ k i ) ≤ ∆ a . Moreover, for a ≥ 1 , each phase { t a , . . . , t a +1 } is of length approximately log(1 /α )(∆ -2 a +1 -∆ -2 a ) . Thus,

<!-- formula-not-decoded -->

Using the definitions of ∆ a and a , we find that this sum is of order log(1 /α ) /ϵ + ϵN k T . We conclude by summing over the values of k ∈ K , using ∑ k ∈K N k T ≤ T and the fact that |K| is of order ϵ -1 .

## 4 Linear Valuation Functions

In this section, we consider the linear valuation model, given by

<!-- formula-not-decoded -->

where θ ∈ R d is an unknown parameter. To ensure that the valuations are bounded, we assume the boundedness of the parameter θ .

Assumption 3. The parameter θ is bounded: ∥ θ ∥ ≤ B θ

Note that under Assumptions 1 and 3, the expected valuations g ( x t ) verify | g ( x t ) | ≤ B g for B g = B x × B θ . Moreover, the random valuations verify a.s. | y t | ≤ B y for B y = B g + B ξ .

We apply the VAPE algorithmic scheme to the problem of dynamic pricing with linear valuations. To estimate the valuation function, we use a ridge estimator for the parameter θ . Moreover, we distinguish between phases by setting ι t = 1 if t belongs to the VALUATION APPROXIMATION phase and ι t = 0 if t belongs to the PRICE ELIMINATION one. The details are presented in Algorithm 2.

Theorem 1. Assume that the valuations follow the model given by Equations (1) and (3) . Under Assumptions 1, 2, and 3, the regret of Algorithm VAPE for Linear Valuations with parameters

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability 1 -˜ O ( T -1 ) , where C B ξ ,B x ,B θ ,L ξ is a constant that polynomially depends on B ξ , B x , B θ , and L ξ .

2 The authors would like to thank Daniele Bracale for pointing out an incorrect choice of α in the previous version of the paper.

| Algorithm 2 VALUATION APPROXIMATION - PRICE ELIMINATION (VAPE) for Linear Valuations   | Algorithm 2 VALUATION APPROXIMATION - PRICE ELIMINATION (VAPE) for Linear Valuations                                                                             |
|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1:                                                                                     | Input : bounds B y and L ξ , parameters α , µ , ϵ . Initialize : ̂ θ 1 = 0 d , V 1 = I d , K = ⌈ ( B y +1) / ϵ ⌉ , K = - K,K , and for k ∈ K , N k 1 = ̂ D k 1 = . |
| 2: 3:                                                                                  | J K 0 while t ≤ T do                                                                                                                                             |
| 4:                                                                                     | if ∥ x t ∥ V - 1 t > µ then ▷ Valuation Approximation                                                                                                            |
| 5:                                                                                     | Post a price p t ∼ U ([ - B y ,B y ])                                                                                                                            |
| 6:                                                                                     | ι t ← 1 , V t +1 ← ∑ s ≤ t ι s x s x ⊤ s + I d , ̂ θ t +1 ← 2 B y V - 1 t +1 ∑ s ≤ t ι s ( o s - 1 2 ) x s                                                        |
| 7:                                                                                     | else ▷ Price Elimination                                                                                                                                         |
| 8:                                                                                     | ι t ← 0 , ̂ g t ← x ⊤ t ̂ θ t , A t ←{ k ∈ K : ̂ g t + kϵ ∈ [0 ,B y ] }                                                                                             |
| 9:                                                                                     | for k ∈ A t do                                                                                                                                                   |
| 10:                                                                                    | UCB t ( k ) ← ( ̂ g t + kϵ ) ( ̂ D k t + √ 2 log( 1 / α ) N k t +2 L ξ ϵ )                                                                                         |
| 11:                                                                                    | LCB t ( k ) ← ( ̂ g t + kϵ ) ( ̂ D k t - √ 2 log( 1 / α ) N k t - 2 L ξ ϵ )                                                                                        |
| 12:                                                                                    | K t ←{ k ∈ A t : UCB t ( k ) ≥ max k ′ ∈A t LCB t ( k ′ ) }                                                                                                      |
| 13:                                                                                    | Choose k t ∈ argmin k ∈K t N k t and post a price p t = ̂ g t + k t ϵ                                                                                             |
| 14:                                                                                    | Update ̂ D k t t +1 ← N k t t ̂ D k t t + o t N k t t +1 , N k t t +1 ← N k t t +1 .                                                                               |

Sketch of proof. [See Appendix B for the full proof] Using Claim 1, we see that it is enough to prove that the VALUATION APPROXIMATION phase allows to estimate g ( x t ) up to precision ϵ = ( d 2 log( T ) 2 / T ) 1 / 3 in at most O ( d 2 / 3 T 2 / 3 log( T ) 2 / 3 ) rounds.

To prove the first part of the claim, note that for all rounds in the PRICE ELIMINATION phase, ∥ x t ∥ V -1 t ≤ µ = ϵ / ( B y √ d log ( 1+ B 2 x T / α ) + B θ ) . Then,

<!-- formula-not-decoded -->

Classical result on ridge regression in bandit framework [1] show that on a large probability event, ∥ θ -̂ θ t ∥ V t ≤ ( B y √ d log ( 1+ B 2 x T / α ) + B θ ) , so | g ( x t ) -g ( x t ) | ≤ ϵ .

<!-- formula-not-decoded -->

To prove the second part of the claim, we rely on the elliptical potential lemma to bound the number of rounds where ∥ x t ∥ V -1 t ≥ µ . This Lemma states that ∑ |G| i =1 ∥ x t i ∥ V -1 t i -1 ≤ √ |G| d log ( |G| + d / d ) , where t i is the i -th round of the VALUATION APPROXIMATION phase, and |G| is its length. Using the fact that ∥ x t i ∥ V -1 t i -1 ≥ µ , we conclude that |G| ≤ d log( T + d / d ) µ 2 , which implies the result.

Theorem 1 provides a regret bound of order ˜ O ( T 2 / 3 ) , showing that VAPE for Linear Valuations is minimax optimal, possibly up to sub-logarithmic terms and to sub-linear dependence in the dimension. Indeed, it matches the T 2 / 3 lower bound established in [31] for linear valuation functions and Lipschitz-continuous demand functions. This result represents a clear improvement over the existing regret bounds for the same problem. Indeed, VAPE achieves the regret bound conjectured in [22] while at the same time removing their regularity assumption on the revenue function. On the other hand, we improve on the regret rate ˜ O ( T 3 / 4 ) achieved respectively in [31] under assumptions slightly milder than ours, and in [14] under stronger assumptions.

## 5 Non-Parametric Valuation Functions

In this Section, we consider the non-parametric valuation model. As usual in dynamic pricing, we assume that the valuation function g is bounded. Furthermore, we assume that it is ( L g , β )-Hölder continuous for some constants L g &gt; 0 and 0 &lt; β ≤ 1 .

Assumption 4. The valuation function g is ( L g , β )-Hölder: for all ( x, x ′ ) ∈ R d , | g ( x ) -g ( x ′ ) | ≤ L g ∥ x -x ′ ∥ β .

| Algorithm 3 VALUATION APPROXIMATION - PRICE ELIMINATION (VAPE) for Non-Parametric Valuations   | Algorithm 3 VALUATION APPROXIMATION - PRICE ELIMINATION (VAPE) for Non-Parametric Valuations                                                                                                                                                                                                 |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1:                                                                                             | Input : bounds B y and L ξ , finite set X ⊂ R d , parameters α , τ , ϵ . : G x = ∅ for all x ∈ X , K = ⌈ B y +1 / ϵ ⌉ , K = J - K,K K , and for k ∈ K , N k 1 = ̂ D k 1 = 0 t ≤ T do x t ← argmin x ′ ∈X ∥ x t - x ′ ∥ if &#124;G x t &#124; < τ then ▷ Price Post a price p ∼ U ([ - B ,B ]) |
| 2: 3:                                                                                          | Initialize while                                                                                                                                                                                                                                                                             |
| 4:                                                                                             | Elimination                                                                                                                                                                                                                                                                                  |
| 5:                                                                                             | t y y                                                                                                                                                                                                                                                                                        |
| 6:                                                                                             | G x t ←G x t ∪ { t } , ̂ g ( x t ) ← 2 B y &#124;G x t &#124; ∑ s ∈G x t ( o s - 1 2 )                                                                                                                                                                                                        |
| 7:                                                                                             | else ▷ Run Successive Elimination                                                                                                                                                                                                                                                            |
| 8:                                                                                             | ̂ g t ← ̂ g ( x t ) , A t ←{ k ∈ K : ̂ g t + kϵ ∈ [0 ,B y ] } for k ∈ A t do                                                                                                                                                                                                                    |
| 9:                                                                                             | UCB t ( k ) ← ( g t + kϵ ) ( D k + √ 2 log( 1 / α ) k +2                                                                                                                                                                                                                                     |
| 10:                                                                                            | ̂ ̂ t N t L ξ ϵ )                                                                                                                                                                                                                                                                              |
| 11:                                                                                            | LCB t ( k ) ← ( ̂ g t + kϵ ) ( ̂ D k t - √ 2 log( 1 / α ) N k t - 2 L ξ ϵ )                                                                                                                                                                                                                    |
| 12:                                                                                            | K t ←{ k ∈ A t : UCB t ( k ) ≥ max k ′ ∈A t LCB t ( k ′ ) }                                                                                                                                                                                                                                  |
| 13:                                                                                            | Choose k t ∈ argmin k ∈K t N k t and post a price p t = ̂ g t + k t ϵ                                                                                                                                                                                                                         |
| 14:                                                                                            | Update ̂ D k t t +1 ← N k t t ̂ D k t t + o t N k t +1 , N k t t +1 ← N k t t +1 .                                                                                                                                                                                                             |

Under Assumptions 1 and 2, the random valuations y t verify | y t | ≤ B y for B y = B ξ + B g .

Next, we apply the VAPE algorithmic scheme to the non-parametric valuation model. To estimate the function g, we use a finite grid of points, on which this function is evaluated. More precisely, we consider a minimal ( ϵ / 3 L g ) 1 / β -covering X of the ball of radius B x in R d , i.e. a finite set of points, of minimal cardinality, such that for any context x such that ∥ x ∥ ≤ B x , there exists a point in X at a distance at most ( ϵ / 3 L g ) 1 / β from x .

At each round, we round the context x t to the closest context x in X by setting x t = arg min x ′ ∈X ∥ x t -x ′ ∥ , and acting as if we observed the context x t . If this context has not been observed sufficiently, we conduct a round of VALUATION APPROXIMATION: we sample a price uniformly at random and use it to update our estimate of g ( x t ) ; otherwise, we proceed with the PRICE ELIMINATION phase. To distinguish between the VALUATION APPROXIMATION steps corresponding to contexts x ∈ X , we collect their indices in sets G x . The algorithm is presented in Algorithm 3.

Theorem 2. Assume that the valuations follow the model given by Equation (1) . Under Assumptions 1, 2, and 4, with probability 1 -˜ O ( T -1 ) the regret of Algorithm VAPE for non-parametric Valuations with parameters ϵ = ( T / log( T ) ) -β d +3 β , α = T -4 , τ = 18 B 2 y log( 2 |X| / α ) / ϵ 2 , and X a minimal ( ϵ / 3 L g ) 1 / β -covering of the ball of radius B x verifies

<!-- formula-not-decoded -->

where C B x ,B g ,B ξ ,L g ,L ξ ,d,β is a constant that polynomially depends on B x , B g , B ξ , L g , L ξ , d , and β .

Sketch of proof. [See Appendix C for the full proof] Using Claim 1, we only need to show that the length of the VALUATION APPROXIMATION phase is at most of order T d +2 β / d +3 β log( T ) β / d +3 β and that w.h.p., it allows estimating g uniformly on a ball of radius B x with precision ϵ =( T / log( T ) ) -β / d +3 β .

To prove the first part of the claim, we note that classical results imply that the size of a minimal covering of precision ϵ 1 / β of a ball in dimension d scales as ϵ -d / β . Then, the total length of the VALUATION APPROXIMATION phase is of order ϵ -d / β τ ≈ T d +2 β / d +3 β log( T ) β / d +3 β . To prove the second part of the lemma, note that the Hölder-continuity of g and the definition of the ( ϵ / 3 L g ) 1 / β -covering G ensure that | g ( x t ) -g ( x t ) | ≤ ϵ / 3 . Then, standard concentration arguments reveal that τ ≈ log( |X| /α ) /ϵ 2 samples are sufficient to estimate g ( x t ) with precision ϵ with high probability.

Theorem 2 shows that the Algorithm VALUATION APPROXIMATION - PRICE ELIMINATION for non-parametric valuations enjoys a ˜ O ( T d +2 β / d +3 β ) regret bound when the noise c.d.f. is Lipschitz

and the valuation function Hölder-continuous. This result is the first of its kind under such minimal assumptions. In particular, previous work by [10] assumes quadratic behavior around the optimal price for all values of g ( x ) - a very strong assumption. However, this rate is higher than the ˜ O ( T d + β / d +2 β ) rates that are usually encountered in β -Hölder non-parametric bandits [7]. Thus, the question of optimality of the VAPE algorithmic scheme in the non-parametric valuation problem remains open.

## 6 Conclusions

In this paper, we studied the problem of dynamic pricing with covariates. We first presented a novel algorithmic approach called VAPE, which adaptively alternates between improving the valuation approximation and learning to set prices through successive elimination. We then applied VAPE under two valuation models - when the buyer's valuation corresponds to a noisy linear function and when expected valuations follow a smooth non-parametric model. In the linear case, our regret bounds are order-optimal, while in the non-parametric setting, we improve existing results. All our results are proven under regularity assumptions that are either milder or match existing assumptions.

Our results on the linear valuation model are the first to match the existing lower bound rate of Ω ( T 2 / 3 ) under our assumptions. However, the optimal dependence of this rate on the dimension of the context remains unknown. Additionally, there are no similar lower bounds for non-parametric valuations. We conjecture that our results are also tight in this setting but leave this for future work. Future research directions also include exploring other valuation models, and further relaxing our assumptions, as Lipschitz-continuity of the noise (Assumption 2). Without this, even minor increases in the price could lead to a major drop in revenue, magnifying the impact of valuation approximation errors. Another limiting assumption is that the noise is independent and identically distributed, such that its distribution can be learned across different contexts. It is of great interest to study problems where the noise distribution can change between rounds, or depends on the context.

## Broader Impacts

As all pricing problems, dynamic pricing can have both positive and negative impacts - offering prices that are more suited to the buyers on the one hand, while increasing the seller's revenue at the expense of buyers on the other hand. In addition, as with many contextual problems, there might be biases and challenges involving fairness - one should make sure that similar customers are offered similar prices. While acknowledging these issues, our work was meant to focus only on the theoretical analysis of what is considered a well-established problem in literature, leaving the study of these related topics as future work.

## Acknowledgments

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101034255. Solenne Gaucher gratefully acknowledges funding from the Fondation Mathématique Jacques Hadamard. Vianney Perchet acknowledges support from the French National Research Agency (ANR) under grant number (ANR-19-CE23-0026 as well as the support grant, as well as from the grant 'Investissements d'Avenir' (LabEx Ecodec/ANR-11-LABX-0047). This research was supported in part by the French National Research Agency (ANR) in the framework of the PEPR IA FOUNDRY project (ANR-23-PEIA-0003) and through the grant DOOM ANR-23-CE23-0002. It was also funded by the European Union (ERC, Ocean, 101071601). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References

- [1] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems , 24, 2011.

- [2] Noga Alon, Nicolo Cesa-Bianchi, Ofer Dekel, and Tomer Koren. Online learning with feedback graphs: Beyond bandits. In Conference on Learning Theory , pages 23-35. PMLR, 2015.
- [3] Kareem Amin, Afshin Rostamizadeh, and Umar Syed. Repeated contextual auctions with strategic buyers. Advances in Neural Information Processing Systems , 27, 2014.
- [4] Santiago Balseiro, Negin Golrezaei, Mohammad Mahdian, Vahab Mirrokni, and Jon Schneider. Contextual bandits with cross-learning. Advances in Neural Information Processing Systems , 32, 2019.
- [5] Omar Besbes and Assaf Zeevi. Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. Operations research , 57(6):1407-1420, 2009.
- [6] Gabriel Bitran and René Caldentey. An overview of pricing models for revenue management. Manufacturing &amp; Service Operations Management , 5(3):203-229, 2003.
- [7] Sébastien Bubeck, Rémi Munos, Gilles Stoltz, and Csaba Szepesvári. &lt;i&gt;x&lt;/i&gt;-armed bandits. Journal of Machine Learning Research , 12(46):1655-1695, 2011. URL http://jmlr.org/ papers/v12/bubeck11a.html .
- [8] Alexandra Carpentier, Claire Vernade, and Yasin Abbasi-Yadkori. The elliptical potential lemma revisited. arXiv preprint arXiv:2010.10182 , 2020.
- [9] Nicolo Cesa-Bianchi, Tommaso Cesari, and Vianney Perchet. Dynamic pricing with finitely many unknown valuations. In Algorithmic Learning Theory , pages 247-273. PMLR, 2019.
- [10] Ningyuan Chen and Guillermo Gallego. Nonparametric pricing analytics with customer covariates. Operations Research , 69(3):974-984, 2021.
- [11] Maxime C Cohen, Ilan Lobel, and Renato Paes Leme. Feature-based dynamic pricing. Management Science , 66(11):4921-4943, 2020.
- [12] Arnoud V Den Boer. Dynamic pricing and learning: historical origins, current research, and new directions. Surveys in operations research and management science , 20(1):1-18, 2015.
- [13] Eyal Even-Dar, Shie Mannor, Yishay Mansour, and Sridhar Mahadevan. Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning problems. Journal of machine learning research , 7(6), 2006.
- [14] Jianqing Fan, Yongyi Guo, and Mengxin Yu. Policy optimization using semiparametric models for dynamic pricing. Journal of the American Statistical Association , 119(545):552-564, 2024.
- [15] Adel Javanmard and Hamid Nazerzadeh. Dynamic pricing in high-dimensions. Journal of Machine Learning Research , 20(9):1-49, 2019.
- [16] N Bora Keskin and Assaf Zeevi. Dynamic pricing with an unknown demand model: Asymptotically optimal semi-myopic policies. Operations research , 62(5):1142-1167, 2014.
- [17] Robert Kleinberg. Nearly tight bounds for the continuum-armed bandit problem. Advances in Neural Information Processing Systems , 17, 2004.
- [18] Robert Kleinberg and Tom Leighton. The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In 44th Annual IEEE Symposium on Foundations of Computer Science, 2003. Proceedings. , pages 594-605. IEEE, 2003.
- [19] Kenneth Littlewood. Forecasting and control of passenger bookings. The Airline Group of the International Federation of Operational Research Societies , 12:95-117, 1972.
- [20] Allen Liu, Renato Paes Leme, and Jon Schneider. Optimal contextual pricing and extensions. In Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 1059-1078. SIAM, 2021.
- [21] Yiyun Luo, Will Wei Sun, and Yufeng Liu. Contextual dynamic pricing with unknown noise: Explore-then-ucb strategy and improved regrets. Advances in Neural Information Processing Systems , 35:37445-37457, 2022.

- [22] Yiyun Luo, Will Wei Sun, and Yufeng Liu. Distribution-free contextual dynamic pricing. Mathematics of Operations Research , 49(1):599-618, 2024.
- [23] Shie Mannor and Ohad Shamir. From bandits to experts: On the value of side-observations. Advances in Neural Information Processing Systems , 24, 2011.
- [24] Jieming Mao, Renato Leme, and Jon Schneider. Contextual pricing for lipschitz buyers. Advances in Neural Information Processing Systems , 31, 2018.
- [25] Vianney Perchet and Philippe Rigollet. The multi-armed bandit problem with covariates. The Annals of Statistics , 41(2):693-721, 2013.
- [26] Marvin Rothstein. Hotel overbooking as a markovian sequential decision process. Decision Sciences , 5(3):389-404, 1974.
- [27] Jon Schneider and Julian Zimmert. Optimal cross-learning for contextual bandits with unknown context distributions. Advances in Neural Information Processing Systems , 36, 2024.
- [28] Virag Shah, Ramesh Johari, and Jose Blanchet. Semi-parametric dynamic contextual pricing. Advances in Neural Information Processing Systems , 32, 2019.
- [29] Roman Vershynin. High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press, 2018.
- [30] Chi-Hua Wang, Zhanyu Wang, Will Wei Sun, and Guang Cheng. Online regularization toward always-valid high-dimensional dynamic pricing. Journal of the American Statistical Association , pages 1-13, 2023.
- [31] Jianyu Xu and Yu-Xiang Wang. Towards agnostic feature-based dynamic pricing: Linear policies vs linear valuation with unknown noise. In International Conference on Artificial Intelligence and Statistics , pages 9643-9662. PMLR, 2022.

le5

1.41

1.2 -

1.0 -

. 0.8 1

log(regl

₴ 1031

0.4 -

0.2 -

0.0 -

1.0

Fan et al.

VAPE

1

1.5

Comparison - Stochastic Case

## A Simulations

2

2.0

2.5

4

5

3.0

Time horizon T

10'

106

log(reg(T))

log (reg(T))

105

103 -

104

![Image](data:image/png;base64,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)

3.5

4.0

Time horizon T

Figure 1: The plots here show the regrets rate of VAPE for linear evaluations, both in the standard and logarithmic scale (left and right respectively). The solid lines represent the average of the performance over 15 repetitions of the routine. The faded red area shows the standard error, while in the right subplot the dotted line corresponds to the theoretical regret bound.

Figure 2: The two subplots show a comparison between VAPE and the algorithm in [14] in the stochastic and adversarial case, where the time horizons used are T ∈ [1000 , 1700 , 3000 , 5000] (left subplot), and T ∈ [1000 , 1400 , 4200 , 9000] (right subplot). In both cases the solid lines represent the average of the regret rates across the 15 repetitions of the simulations, while the faded area the standard error. In the subplot on the right, due to the specificity of the setting, the variance across runs is minimal, hence the faded area results invisible. The regret graph is in both cases plotted in logaritmic scale.

![Image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAqQAAAD1CAIAAAAJaXCFAACX8ElEQVR4nOydB3xb1dn/zx3ayxq25b134sTZk+wEEkYgZDACYVOgbAodrL790xbaUgqUWTaUESCQvfd0nGk73ntLsrZ09/1/jq4xIWUkthyv83375hMU+epK1j2/e57xezBRFAECgUAgEIihC97fJ4BAIBAIBKJvQWKPQCAQCMQQh+zvExiU+P3+jRs3Hj16lOf5zMzMyy+/PCYmpo9eq62tbceOHddee61cLg/jYUtKSr755pvOzk69Xn/FFVfk5+fv2LFj/PjxERERv/iz69evDwQCS5cuPc/X4nl+8+bNl1xyiVarbWhoOHTo0DXXXEOSP/7d43n+wIED27Zt8/l80dHRCxcuHDFixAW+OQRiIELT9LZt2w4cOEDTdHJy8uWXX56cnNxHr2Wz2bZs2bJ06dLwrhsAgPLy8hdeeOG+++4bPXr02Y8fP3786NGjd9xxB+gbSktL6+vr586dK5PJzn7c6/W++eabN910U2Rk5NlP/uabb+x2u1arnTNnziWXXNJHZzWIQGJ/wbAs+/LLL1dVVd1www16vf706dNNTU0xMTFut7u1tVWv18fGxgIAOjo6ZDJZZ2enTqeLjIxsbGwEAMTFxeE43tnZKf2pUCji4uIwDPP5fG1tbYIgxMXFaTQalmXtdrtCoejs7NRqtVlZWSRJCoLQ2NgYCARMJlNUVBSGYe3t7U6nMyoqymQycRzncDhIkuzs7IyMjPx5ze7s7Hz++eenTJmyaNGi1tZWgiAaGxv/9re/3XnnnePGjUtMTKQoqrGxUaFQJCYmYhgGAHC5XK2trSqVKiEhoby83GazTZ48mWXZxMREgiAoimppaWFZ1mq1GgwG6X6opaVFFMW4uDibzfbnP/+Z5/mCggKNRpORkYHjMKRkt9ttNptGo4mLiyMIQjq3DRs2vPvuuzfccEN6enpFRUVdXV1ubq7NZnM6nSqVSjofURRbWlrcbrfBYLBarQRBuFyutra27g8fgRiAvPfee7t377755pujoqIk6UpOTvb5fM3NzWq1OiEhQbooCIKQvu0xMTHNzc0sy8bHx0uXtiiKHo+HIIj4+Hgcx/1+f1tbG8dxsbGxOp2O4zibzaZUKu12u06ny8rKIghCEISmpia/3280GqOionAc7+jokFYJs9l89rphsViMRuMvvou1IUwmU35+vnQhNzc30zRdXV199OjRFStW0DRtNpsxDPN6vTRNWyyWYDDY1NQkk8kSEhKkd8fzvNvt1odobGzkOC46OtpoNLrd7vb2dukNKhQKiqK8Xi/P816v12AwJCcnn/2u4+LitFotTdN79uxZsmRJt9gXFRX95S9/WbBgwcKFC9va2qqrqydNmuT3+202G0mS8fHx0g2QtP4olcr4+HiZTBYIBJqamhQKRXx8fPdyNJRAYn/BNDc379q167nnnhszZgwAYOzYsTzP19TUvPzyywRBeDyepUuXzps37/nnn2cYxmg0trW1TZo0qbGxsb6+Xvqn1157rbGxMTo6uqWlZcmSJZdddtnhw4ePHj3KsixJknfeeWcwGHz44YfT09MjIyNHjRq1cePG3Nzc9evXHzhwQK/XW63WVatWFRUVffjhhzqdLhgM3n///VFRUY888khSUhLP8xRFPfXUUyaT6afeAk3TDodDp9OlpKSMGjUKALB79+6WlpYtW7ZwHCeTyd544w2Xy0VR1LRp01auXHnq1Km33npLo9GoVKo777xTqVQeP378o48+qqurmzFjxnXXXXfy5Mk9e/awIW699VaDwfDvf//b4/GQJDlr1iyGYVpbWzdt2iTdzXz99dejRo0qLCx85513IiIi1Gr13XffLV2oHMd9+umnl19++ZIlSwAAo0aNEgTB6/WuW7fO4XC43e6CgoIlS5Zs3rx569atOp0uIiLipptucrlcr776qlwu9/l8V1999ezZsy/uNwKB+GX8fv/q1asff/zxuXPnAgAKCgp4nm9vb//73//OsqzP51u4cOHVV1/92muvtbS0WK3WlpaWsWPH2u322traSy+99Jprrvnggw9OnjyZnJzc1NR06aWXLlmy5OjRo4cPH2ZZVhTFu+66SxTFRx99NCkpyWw2jxkzZv369bm5uVu3bt29e7der4+MjLzllluKi4v/85//6PV6n8/3wAMPxMTE/OY3v4mNjRVFMRgMPvnkkxaL5Wfehd1uP3ny5KOPPrpjx46mpqbExMSdO3d+8MEHycnJVVVVFoulsrLyk08++eMf/6hWq9977z2j0bhw4cLXX3/d5XIxDDNu3Ljrr7/+v//97759+9LT06dOnVpeXt7U1KRUKkeNGnX55Zfv2bPnzJkzwWAwMjLy5ptvrqqqev755xMTE6OjozMyMiorK9PS0o4dO3bw4EGO4wAAt912myKEtCeR+OabbzIzM2+//fbuNYSm6Z07d1ZVVQWDwZiYmJtvvrm6uvr11183mUyiKN5+++1KpfLf//633++nKGrq1KlLly6V7mOGEkPt/VwE7HY7hmFxcXHdjxAE8emnnyoUij/84Q/XXnvtG2+80draWldXl5CQ8Oijj6pUqq1bt/7617+eNWvWmjVraJpuaWnRarUPPvjgwoULP//8c7vdnpaWlpKSEh0dffDgwZ07d/I8X1dXd8kll9x99916vb6hoSEQCGzZsiU5Ofm2225btmwZhmGvvfbalClTnn766eTk5Lffftvv99fU1EyaNOnxxx/3+Xz79+/vPj2O4yiKomm6u/MiOjp65cqVn3/++dKlSx9++OGKiooRI0ZkZGTccsstS5Ys2blzZ11d3e9///v77rvvs88+O3ny5DvvvJOZmfm73/3unnvuiYqK4jjOaDSuWrVq6dKla9euDQaDycnJKSkpVqv16NGj27Zta21tLSoqmjZt2j333DNx4sRx48alpaXdfvvtl19+Ocuyzc3NHo/nX//615QpU373u9/ddddd3XEImqYbGxvz8vK6Tx7Hca1Wm52dHRMTo9Vq33333fb29n379ikUilWrVq1cudJoNL766qsWi+U3v/nNrFmz3njjDa/XexG/DgjEeeFyufx+f1paWvcjBEF8+eWXfr//D3/4w2233fbOO+/U19c3NjaaTKYHH3wwLi5u/fr1t9122xVXXLFhwwZpyyuTye65556lS5d+/fXXzc3NaWlpycnJVqu1qKhIupmura2dNGnSvffea7FY6uvrg8Hgtm3brFbrrbfeet111xEE8dprr02cOPHpp5/Ozs5+4403AoFAdXX1hAkTHn/8cYqi9u3b1316HMcFg8Gz1w0AwNGjR0VRvPXWW5OTk7dv307T9Ntvv71gwYKHHnooJSXF7/enpqa2tbWdOnXK7XYXFhZmZGSsXbu2tbX1/vvvX758+caNG6W4II7jDzzwQEFBwZYtWwoKCu655565c+fK5fK0tLSEhASj0bhx48aTJ0/SNN3a2rpw4cK77rpL+jsAICkpSVotjx8/vnHjRixE9xmyLNve3p6SktL9CI7jCoUiIyMjISFBr9dv2LChtLS0pKSkpaVl+fLl9957r9Vq/fzzz30+30MPPbRkyZIvvviioaEBDDmQ2F8warVaugy6HxFFsa6ubuLEiRERERMmTJCC8DqdbvTo0TqdLjExMT093Wg0pqWlud1ulmUVCsXEiRNNJtO4ceMoimptbX3zzTcPHjwIAJDL5e3t7YIgGI3GyZMnK5VKkiRxHNdoNLfddltdXd1f//rXr7/+2m63OxyOiRMnarXaGTNmSPIZHx+fl5dnNBpjY2NtNlv36e3bt++xxx7717/+5XQ6pUdwHF++fPnHH3/8/PPPUxT15z//meM4uVyu0+lIkqyurh41alRUVNTIkSO1Wu2ZM2f8fv+kSZOkzYEU4MrKyrJarcnJySRJulyud999d/fu3RzHKZXK5ubmlJSUK664Yv369S+88MLhw4dlIXQ6HUEQOI6TJOlwOCiKmjx5sk6ni4qK6k7CEQSh1Wo7OzvP/sCLi4tfeeWVtrY2tVrt8/m8Xu8NN9wAAHjhhRc+/PDDjo6OioqKU6dO/eMf/9i9e7dWq/X7/Rfru4BAnC9KpRLH8XPuRCsrK8eNG2c2m8ePH4/jeFtbm1arLSgoMBgMiYmJGRkZ0dHRqampwWCQoiiZTDZ27NjIyMgxY8aIotje3v6f//xn7969giAolcq2tjae5/V6/fTp07vXDZVKtWrVqvb29ueff3716tVS4HrSpElarXbmzJktLS1OpzMuLm7EiBH/u27s37//N7/5zdnrBs/zGzZsEEXx+PHjcrl8+/btDQ0NPp9v+vTper1+4sSJoihGRETMnj177dq1R44c0ev1I0eOrK6uLikpeeONN7788kuSJAOBgEKhkN51ZGTkrbfeeuDAgf/3//7fnj172tvbX3311dLSUplMxrKs9LppaWl5eXkqlYogCJIkGYb5z3/+s2fPHp7nFQpFe3v7Od3jBEEolUqPx3P2g83Nza+99lppaamUD21ra5s5c2Z+fv6rr776yiuvNDY21tTUVFRUvPrqq2vXruU4jud5MORAYfwLJiEhISUl5dNPP73jjjtUKlV1dbVer09JSTly5MicOXOOHDkik8nMZrMYQroVkG48ux/hOO7w4cPz588vKipSqVQYhpWXlz/xxBPp6embN2+WnnP2vaooigzDpKSk/P73vz9+/PiLL76YnZ0dHR195MiR2NjY3bt3x8bGarVaQRCkn+3+i8To0aMTEhIUCoVer5ceCQQC9fX10kWen5+/a9cuQRB4nnc4HACA1NTUrVu32u32trY2j8eTk5Nz6NChAwcO5OTkUBTVfRDphaSCg8LCwoceemjUqFH79++XLpX58+dfeumlH3/88X/+85+srCxBEFwul/ReeJ43mUxKpfLAgQMxMTGBQMBgMEhZNKVSOX/+/C+++CInJycmJqaxsVEQhPLycplMdu+99x45cuSzzz7jeT4qKurhhx+urq5+6qmnCgoKcnNzCYK47rrrpMjez8chEYh+ISIiYuLEiZ988kl0dLRer6+rq1MoFGlpaUePHl28eHF5eTnP81artfviPWfdwDCM5/ljx451dnYeP34cx3GlUnnq1KmHHnpo5MiRe/bs6V43uq99URSlqprf/e53p0+ffuGFFzIzM2NiYg4dOpSQkLBr167Y2Fi9Xt+9Lv3vuiElzrsv+bq6uuLi4vHjxx8/ftxgMJw5c6alpUWv1+/fv3/BggUHDhwQBAEAcNlll9111111dXWLFi1Sq9WJiYkpKSmLFy9WKBQsy2ZnZ2/atEl6a4IgTJgwYebMmWvWrHnttdfMZrPNZnvsscc4jluzZo10NOmNdP/p9/vLysp+9atfjRkzZu/evWf/kwSO4zNnzvzkk09OnjyZlZXldrsbGhooivJ4PE888YTf71+/fr0gCHK5/K677goGg4899tjGjRuzsrIoirrmmmukQoEhWfqDxP6C0el0Dz744HvvvffEE0/IZDKtVrtq1aply5b9/e9/f+CBB1iWvf3226Ojo2UymbQJlofovuXEMIwkyba2tqeeeqqjo+OGG27IysqaNm3aiy++GBsbK5fLpfyTdBMgfXfVajXDMF999dWpU6d4np8wYUJOTs6qVavefffdnTt3EgTx8MMPn524UigUZxesRoQ4+y3QNL169erm5mYAQDAYvOmmm6Kjo8ePH//iiy+WlpYuWrSotLT04Ycf5jhu6dKl+fn5N91005tvvvnggw8ajcaHHnpIoVBI7wjHcblcHhERsXjx4ldffTUuLs7v92u12vb29rfeesvpdHIcd9lll5nN5nHjxv31r3+94oorUlNTVSqVXq+/55573nrrrb1795rN5ocffjgqKko6txtvvPHdd9/905/+JJfLcRy//PLLJ0+evGXLlgcffDAiIsJoNOI4vmXLlj179gAApOzDbbfd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)

In this section, we illustrate some numerical simulations that aim to show the empirical performance of our VAPE algorithm for Linear Valuations. Moreover we present a comparison with the algorithm proposed in [14] in the case in which the only regularity assumed on the CDF of the noise distribution is Lipschitzness. The code implemented for these simulations is publicly available in the repository: https://github.com/MatildeTulii1/ Improved-Algorithms-for-Contextual-Dynamic-Pricing

VAPE In order to test our algorithm, we built a dataset of 5 contexts belonging to R 3 generated by a canonical gaussian distribution and subsequently normalized. Throughout the run the contexts are chosen from this set uniformly at random, while the noise term is picked from a gaussian distribution truncated between -1 and 1 with mean 0 and variance 0 . 1 . Similarly, also the parameter θ is a normalized vector initially drawn from a gaussian distribution. The algorithm has been tested on time horizons T ∈ [1000 , 10000 , 50000 , 200000 , 500000 , 800000] , and the hyperparameters α, µ, ϵ are set as in the statement of Theorem 1. Figure 1 shows the results of this implementation. The empirical regret rates of VAPE respect the theoretical upper-bound expressed in the paper, moreover it shows optimal computational times that can handle big time horizons.

- Fan et al.

- VAPE|

Comparison - Adversarial Case

Comparison with [14] Next we compare our algorithm with the algorithm proposed in Appendix F of [14], in which they propose a routine to tackle the dynamic pricing problem with linear valuation in the case in which the CDF F is Lipschitz. The comparison is carried out in two different settings: a stochastic and an adversarial one, and to make it more fair both algorithm receive as input the time horizon T .

In the stochastic case, similarly as before we consider a set of possible contexts in R 3 drawn uniformly at random and then normalised. During the routine, at each time step one of these is randomly selected. This method of receiving contexts meets the assumption included in [14], making sure that no eigenvalue of the covariance matrix of the distribution of contexts is too small. As for the contexts, the parameter θ is selected ex-novo with every new run of the algorithm. We chose to implement a comparison with this specific algorithms since it was among the closest with our work, as discussed in the main paper, but its prohibitive computational costs make difficult to see the good behaviour of VAPE which, being based on a bandit approach, requires bigger time horizons to converge.

The adversarial case, instead is a toy example which is purposely designed to badly interfere with the algorithm proposed by [14]. In this case the set of contexts is made of only two samples of orthogonal vectors, specifically in the form [ x, 0 , z ] and [0 , 1 , 0] . To make sure that the effect of this choice of contexts is not invalidated by the parameter θ , this is considered to be fixed as [0 . 3 , 0 . 3 , 0 . 3] . The algorithm receives the first context during the exploration phase and the second during the exploitation one, such that the information gathered initially result meaningless in the latter subroutine. As before the computational costs of [14] limited the time horizons on which we were able to run this simulation, still it can be noted how VAPE, exposed to the same contexts in the same order, does not suffer from such choice, since the phases are defined adaptively, thus its regret rates remain consistent with the stochastic case. The results of this comparison are shown in Figure 2.

## B Proof of Theorem 1

We state several lemmas before proving Theorem 1. We begin by bounding the length of the exploration phase corresponding to lines 5 and 6 of Algorithm 2.

Lemma 1. Let G = { t ≤ T : ι t = 1 } . Almost surely, the length of exploration phase G is bounded as

<!-- formula-not-decoded -->

The following lemma bounds the error of our estimates for θ and D , for the values of µ prescribed in Theorem 1. Before stating the Lemma, we define the event

<!-- formula-not-decoded -->

Lemma 2. The event E happens with probability at least 1 -( α +2 T 2 |K| α ) .

Finally, we bound the number of times a sub-optimal price increment kϵ can be selected. For p ∈ R , x ∈ R d , we define

<!-- formula-not-decoded -->

Lemma 3. On the event E , for all t / ∈ G , if k t = k , then k must be such that

<!-- formula-not-decoded -->

We are now ready to bound the regret of Algorithm VAPE for Linear Valuations. We begin by rewriting the regret as

<!-- formula-not-decoded -->

Under Assumptions 1, 2, and 3, both the optimal price and p t are in [0 , B y ] , we know that the instantaneous regret is bounded by B y . Then,

<!-- formula-not-decoded -->

Using Lemma 1 together with the definition of µ , we find that

<!-- formula-not-decoded -->

We rely on the following Lemma to bound ∑ t/ ∈G ( max p ∈ [0 ,B y ] π ( x t , p ) -π ( x t , p t ) ) . Lemma 4. On the event E ,

<!-- formula-not-decoded -->

Combining Equations (4), (6), and Lemma 4, we find that

<!-- formula-not-decoded -->

Using the definition of K , ϵ and α allows us to conclude the proof.

## C Proof of Theorem 2

The proof of Theorem 2 follows closely the proof of Theorem 1. The following two Lemmas are analogues of Lemmas 1 and 2.

Lemma 5. Let X be an ( ϵ 3 L g ) 1 / β -covering of B B x ,d of minimal cardinality, and let G = ⋃ x ∈X G x . Almost surely, the length of exploration phase G is bounded as

<!-- formula-not-decoded -->

Recall that we defined the event E as

<!-- formula-not-decoded -->

The following lemma shows that E happens with large probability.

Lemma 6. The event E happens with probability at least 1 -( α +2 T 2 |K| α ) .

The rest of the proof holds follows the proof of Theorem 1. In particular, on the event E , we still have

<!-- formula-not-decoded -->

where we used the fact that the instantaneous regret is bounded by B y along with Lemma 4. Using Lemma 5, we obtain

<!-- formula-not-decoded -->

Using the definition of K , ϵ , τ and α allows us to conclude the proof.

## D Proof of Auxilliary Lemmas

## D.1 Proof of Lemma 1

We use the elliptical potential Lemma (see, e.g., Proposition 1 in [8]) to bound the total number of rounds used to estimate θ . Formally, denote the estimation indices G = { t 1 . . . , t |G| } and notice that ι t = 1 only for these indices. Thus, for all i ∈ [ |G| ] , we can write V t i = ∑ i k =1 x t k x ⊤ t k + I d and V t i -1 = V t i -1 . In particular, the elliptical potential lemma implies that

<!-- formula-not-decoded -->

Since for all t such that ι t = 1 , x ⊤ t V -1 t i -1 x t ≥ µ , this implies that

<!-- formula-not-decoded -->

Now, almost surely, |G| ≤ T . Using this bound and reorganizing the inequality leads to the desired result

<!-- formula-not-decoded -->

## D.2 Proof of Lemma 2

Lemma 2 is obtained by combining the following two results.

Lemma 7. Let us define the event

<!-- formula-not-decoded -->

Then, the event E 1 happens with probability at least 1 -α .

The remainder of the proof follows from the following lemma.

Lemma 8. Let us define the event

<!-- formula-not-decoded -->

Assume that event E 1 holds with probability 1 -α . Then, the event E happens with probability at least 1 -( α +2 T 2 |K| α ) .

## D.3 Proof of Lemma 3

We assume that t / ∈ G , that k t = k , and that N k t &gt; 0 (otherwise the statement is trivial). We begin by stating an auxiliary result, which follows immediately from Lemma 2.

Lemma 9. On the event E , we have that for all t / ∈ G , and all k ∈ A t ;

<!-- formula-not-decoded -->

Moreover, k ∗ t ∈ K t , where

<!-- formula-not-decoded -->

On the event E , Lemma 9 implies that

<!-- formula-not-decoded -->

Since k ∗ t ∈ A t , we have

<!-- formula-not-decoded -->

This implies

<!-- formula-not-decoded -->

Thus,

Now,

<!-- formula-not-decoded -->

since k ∈ A t . Moreover, since k t = k , and since k ∗ t ∈ K t by Lemma 9, we know that N k t ≤ N k ∗ t . This implies that

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Next, we bound the discretization error using the following Lemma.

Lemma 10. On the event E , we have that

<!-- formula-not-decoded -->

By Lemma 10, Equation (7) implies that on the event E ,

<!-- formula-not-decoded -->

## D.4 Proof of Lemma 4

Note that

<!-- formula-not-decoded -->

We bound this term on the high-probability event E . For k ∈ K , we define t k 1 &lt; · · · &lt; t k N k T +1 the rounds where t / ∈ G and k t = k . Note that t k i corresponds to the i -th time arm k is played, hence, according to our notation, N k t k i = i . We split these rounds into episodes as follows. We define a = ⌊-log 2 (18 L ξ ϵ ) ⌋ . For a ∈ J 1 , a K , we also define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With these notations, we have

<!-- formula-not-decoded -->

On the one hand, ∆( x t , p t ) ≤ B y for all t ≤ T , so

<!-- formula-not-decoded -->

On the other hand, using Lemma 3, we see that on the event E , if i ≥ t a and a ∈ J 1 , a K ,

<!-- formula-not-decoded -->

Since 2 -a ≥ 18 L ξ ϵ , this implies that

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

By definition of t a , this implies that

<!-- formula-not-decoded -->

where we used that 2 a ≤ 1 18 L ξ ϵ . Similarly,

<!-- formula-not-decoded -->

Combining these results, we find that

<!-- formula-not-decoded -->

We conclude the proof by summing over k ∈ K , and using the fact that ∑ k ∈K N k T +1 ≤ T .

## D.5 Proof of Lemma 5

We note that

<!-- formula-not-decoded -->

We conclude by using classical results on covering number of the ball (see, e.g., Corollary 4.2.13 in [29]), stating that there exists an ( ϵ 3 L g ) 1 / β -covering of the ball of radius B x in dimension d of

<!-- formula-not-decoded -->

## D.6 Proof of Lemma 6

The proof of Lemma 6 relies on the following Lemma.

Lemma 11. Let us define the event

<!-- formula-not-decoded -->

Then, the event E 1 happens with probability at least 1 -α .

Note that Lemma 8 still holds for non-parametric valuations. This concludes the proof of Lemma 6.

## D.7 Proof of Lemma 7

We introduce the variables

<!-- formula-not-decoded -->

and the σ -algebra F t = σ (( x s ) s ≤ t +1 , ( o s ) s ≤ t ) . Since V t -1 and x t are F t -1 -measurable, then so does ι t , and thus both ˜ x t +1 and ˜ y t are F t -measurable. Moreover, for any round where ι t = 1 , the price is chosen uniformly at random and we have

<!-- formula-not-decoded -->

where in the last equality we used that ∫ B ξ -B ξ ξf ( ξ ) d ξ = E [ ξ t ] = 0 . The same relation also trivially holds when ι t = 0 . Thus, conditionally on F t -1 , ˜ y t -˜ x ⊤ t θ is centered and in [ -B y , B y ] , which implies that it is B y -subgaussian. Now, for all t ≤ T , we have

<!-- formula-not-decoded -->

Using the fact that for all t ≥ 1 , ∥ ˜ x t ∥ ≤ B x , and that ∥ θ ∥ ≤ B θ , and applying Theorem 2 in [1], we find that for all t ≥ 0 , with probability 1 -α ,

<!-- formula-not-decoded -->

Note that our definitions of ˜ x t and ˜ y t ensure that ∥ ̂ θ t -θ ∥ ( ∑ s&lt;t ˜ x l ˜ x ⊤ l + I d ) = ∥ ̂ θ t -θ ∥ V t . Moreover, for all t ,

<!-- formula-not-decoded -->

In particular, if t / ∈ G , ∥ x ⊤ t ∥ ( V t ) -1 ≤ µ , so

<!-- formula-not-decoded -->

The conclusion follows from the choice ϵ = µ ( B y √ d log ( 1+ B 2 x T α ) + B θ ) , and the fact that ̂ g t = x ⊤ t ̂ θ t .

## D.8 Proof of Lemma 8

We rely on the following well-known result (we provide proof in the appendix for the sake of completeness).

Lemma 12. Let ( y t ) t ≥ 1 be a sequence of random variables adapted for a filtration F t , such that y t -E [ y t |F t -1 ] ∈ [ m,M ] . Assume that for t ∈ N ∗ , ι t ∈ { 0 , 1 } is F t -1 -measurable, and define N t = ∑ s ≤ t ι s , and ̂ µ t = ∑ s ≤ t ι s ( y s -E [ y s |F s -1 ]) N t if N t ≥ 1 . Then, for any t ∈ N ∗ and α ∈ (0 , 1) ,

<!-- formula-not-decoded -->

Moreover, for any l &gt; 0 and α ∈ (0 , 1) ,

<!-- formula-not-decoded -->

Note Lemma 8 holds trivially for all t such that N k t = 0 . Therefore we assume w.l.o.g. that N k t ≥ 1 (otherwise the statement is trivial). For any such given t ∈ [ T ] , we control the error | ̂ F k t -F ( kϵ ) | uniformly for k ∈ K . To do so, we rely on Lemma 12; we define ˜ ι t = 1 { ι t = 0 and k t = k } , and note that for F t = σ (( x 1 , . . . , x t +1 ) , ( o 1 , . . . , o t )) , ˜ ι t is F t -1 -measurable, and o t is F t adapted. Moreover,

<!-- formula-not-decoded -->

and directly by definition, it holds that ̂ D k t = ∑ s ≤ t ˜ ι t o t N t . Using Lemma 12, we find that with probability 1 -2 αt , N k t = 0 or

<!-- formula-not-decoded -->

Moreover, on the event E 1 , which happens w.p. at least 1 -α , for all t / ∈ G , | ̂ g t -g ( x t ) | ≤ ϵ . Using the fact that D is L ξ -Lipschitz, we find that for all t / ∈ G ,

<!-- formula-not-decoded -->

Thus, with probability 1 -2 αt ,

<!-- formula-not-decoded -->

Using a union bound over all k ∈ K and t ∈ [ T ] and then intersecting with E 1 using another union bound yields the desired result.

## D.9 Proof of Lemma 9

For any t / ∈ G , denoting p t ( k ) = ̂ g t + kϵ , we first rewrite

<!-- formula-not-decoded -->

Since the event E holds, the following hold for all t / ∈ G and k ∈ A t :

<!-- formula-not-decoded -->

In particular, we have that:

<!-- formula-not-decoded -->

Relation (1) holds since D is L ξ -Lipschitz and (2) is under the event E for all t / ∈ E . As the set A t is chosen such that ̂ g t + kϵ ≥ 0 for all k ∈ A t , it implies that

<!-- formula-not-decoded -->

Reorganizing, we get for all k ∈ A t and t / ∈ G

<!-- formula-not-decoded -->

which proves the first part of the statement.

Now let k ∗ t ∈ arg max k ∈A t π ( x t , ̂ g t + kϵ ) . By the first part of the claim, it holds that

<!-- formula-not-decoded -->

where relations ( ∗ ) are due to the first part of the lemma; this proves that k ∗ t ∈ K t .

## D.10 Proof of Lemma 10

The proof follows by noticing that, on the one hand, K ensures that for all p ∈ [0 , B y ] , there exists k ∈ K such that ̂ g t + kϵ ∈ [0 , B y ] and | ̂ g t + kϵ -p | ≤ ϵ . On the other hand, the prices considered are bounded by B y , and the demand function D is L ξ -Lipschitz, so the reward function π is B y L ξ -Lipschitz.

## D.11 Proof of Lemma 11

For x ∈ X , let us define recursively the variables ι x 1 = 1 { x 1 = x } , and for t &gt; 1 , ι x t = 1 { x t = x, and ∑ s&lt;t ι x s &lt; τ } , and define the variables

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the σ -algebra F t = σ (( x s ) s ≤ t +1 , ( o s ) s ≤ t ) . Note that ι x t is F t -1 -measurable, and thus both ˜ x t +1 and ˜ y t are F t -measurable. Moreover, for any round where ι x t = 1 , the price is chosen uniformly at random and we have

<!-- formula-not-decoded -->

where in the last equality we used that ∫ B ξ -B ξ ξf ( ξ ) d ξ = E [ ξ t ] = 0 . The same relation also trivially holds when ι x t = 0 . Thus, conditionally on F t -1 , ˜ y t -˜ g x t is centered and in [ -B y , B y ] . We denote N x t = ∑ s&lt;t ι x s , we note that if t / ∈ G x , then N x t = ⌈ τ ⌉ a.s. Using Lemma 12, we find that for all t / ∈ G x , a.s., N x t = ⌈ τ ⌉ . Then,

<!-- formula-not-decoded -->

Moreover, since g is ( L g , β )-Holder- continuous, and ∥ x t -x t ∥ ≤ ( ϵ 3 L g ) 1 / β a.s., we have

<!-- formula-not-decoded -->

Then, with probability at least 1 -α / |X| , for all t / ∈ G x ,

<!-- formula-not-decoded -->

where we used τ = 18 B 2 y log( |X| /α ) ϵ 2 . Using a union bound over X , we find that with probability at least 1 -α , for all t / ∈ G x ,

<!-- formula-not-decoded -->

Similarly, for all t / ∈ G , ∥ g ( x t ) -g ( x t ) ∥ ≤ L g ϵ 3 L g . Then, we have that with probability 1 -α , for all t / ∈ G x ,

<!-- formula-not-decoded -->

## D.12 Proof of Lemma 12

Let us define Z t = ∑ s ≤ t ι s ( y s -E [ y s |F s -1 ]) , and for x ∈ R , M t = exp ( xZ t -x 2 ( M -m ) 2 N t 8 ) . We begin by showing that M t is a super-martingale. Indeed, we have that

<!-- formula-not-decoded -->

where we use the fact that ( y t -E [ y t |F t -1 ]) is bounded in [ m,M ] together with the conditional version of Hoeffding's Lemma. Noticing that

<!-- formula-not-decoded -->

this proves that M t is a super-martingale, and so E [ M t ] ≤ E [ M 0 ] = 1 .

Now, for all ϵ &gt; 0 and all l ∈ N , and all x &gt; 0 , by a Markov-Chernoff argument,

<!-- formula-not-decoded -->

Using the previous result, we have that

<!-- formula-not-decoded -->

so

<!-- formula-not-decoded -->

In particular, for ϵ = ( M -m ) √ l · log(1 /α ) 2 and x = 4 ϵ l ( M -m ) 2 ,

<!-- formula-not-decoded -->

This proves the first part of the Lemma. Summing over the values of l from 1 to t , we find that

<!-- formula-not-decoded -->

Similar arguments can be used to prove that

<!-- formula-not-decoded -->

Noting that Z t = ˆ µ t N t and normalizing by N t (and since adding the case N t = 0 can only increase the probability) concludes the proof of the Lemma.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In the abstract and introduction, we claim to present a novel approach to dynamic pricing that enjoys improved regret bounds. We clearly state the approach and explain its novelty, while proving all stated bounds in the appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The main limitations of our papers are due to the assumptions on the setting. While we follow a minimal set of assumptions that is standard in the dynamic pricing literature, we discuss how to alleviate them in the conclusion section. The computational complexity of our algorithm is comparable to previous work on the topic: the complexity of the price elimination is polylog ( d, T ) , while the complexity of the valuation estimation depends on the valuation model. For linear valuations, it is polynomial, while for nonparametric ones, it is exponential - as standard in the non-parametric bandit literature.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best

judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We clearly state our assumptions and prove the results in the appendix.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The information needed to reproduce the experiment is detailed in Appendix A. Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The code to reproduce the experiments is publicly available in the repository https://github.com/MatildeTulii1/ Improved-Algorithms-for-Contextual-Dynamic-Pricing

## Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The details of the implementations of the simulations are detailed in Appendix A.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All the information relative to the statistical significance of the algorithm is contained in Appendix A.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: All the simulation can be (and were) run on a laptop without gpus.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper theoretically studies a well-established theoretical problem; as such, it does not have any direct ethical implications.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [Yes]

Justification: The paper theoretically studies a well-established theoretical problem and the broader impact of our work is only due to the potential impact of advancements in this problem. As all pricing problems, dynamic pricing can have both positive and negative impacts - offering prices that are more suited to the buyers on the one hand, while increasing the seller's revenue at the expense of buyers on the other hand. In addition, as with many contextual problems, there might be biases and challenges involving fairness - one should make sure that similar customers are offered similar prices. This study is orthogonal to ours, and we leave it for future work.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.