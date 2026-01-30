## Low-Rank Bandit Methods for High-Dimensional Dynamic Pricing

Jonas Mueller

MIT CSAIL

jonasmueller@csail.mit.edu

## Vasilis Syrgkanis

Microsoft Research vasy@microsoft.com

## Abstract

We consider dynamic pricing with many products under an evolving but lowdimensional demand model. Assuming the temporal variation in cross-elasticities exhibits low-rank structure based on fixed (latent) features of the products, we show that the revenue maximization problem reduces to an online bandit convex optimization with side information given by the observed demands. We design dynamic pricing algorithms whose revenue approaches that of the best fixed price vector in hindsight, at a rate that only depends on the intrinsic rank of the demand model and not the number of products. Our approach applies a bandit convex optimization algorithm in a projected low-dimensional space spanned by the latent product features, while simultaneously learning this span via online singular value decomposition of a carefully-crafted matrix containing the observed demands.

## 1 Introduction

In this work, we consider a seller offering N products, where N is large, and the pricing of certain products may influence the demand for others in unknown ways. We let p t P R N denote the vector of selected prices at which each product is sold during time period t P t 1 , . . . , T u , which results in total demands for the products over this period represented in the vector q t P R N . Note that q t represents a (noisy) evaluation of the aggregate demand curve at the chosen prices p t , but we never observe the counterfactual demand that would have resulted had we selected a different price-point. This is referred to as bandit feedback in the online optimization literature [Dani et al., 2007]. Our goal is find a setting of the prices for each time period to maximize the total revenue of the seller (over all rounds). This is equivalent to minimizing the negative revenue over time:

<!-- formula-not-decoded -->

Wecan alternatively maximize total profits instead of revenue by simply redefining p t as the difference between the product-prices and the cost of each product-unit. In practice, the seller can only consider prices within some constraint set S GLYPH&lt;128&gt; R N , which we assume is convex throughout. To find the optimal prices, we introduce the following linear model of the aggregate demands, which is allowed to change over time in a nonstationary fashion:

<!-- formula-not-decoded -->

Here, c t P R N denotes the baseline demand for each product in round t . B t P R N GLYPH&lt;2&gt; N is an asymmetric matrix of demand elasticities which represents how changing the price of one product may affect the demand of not only this product, but also demand for other products as well. By conventional economic wisdom, B t will have the largest entries along its diagonal because demand

Matt Taddy

Chicago Booth taddy@chicagobooth.edu

for a product is primarily driven by its price rather than the price of other possibly unrelated products. Since a price increase usually leads to falling demand, it is reasonable to assume all B t ' 0 are positive-definite (but not necessarily Hermitian), which implies that at each round: R t is a convex function of p t . The observed aggregate demands over each time period are additionally subject to random fluctuations driven by the noise term glyph[epsilon1] t P R N . Throughout, we suppose the noise in each round glyph[epsilon1] t is sampled i.i.d. from some mean-zero distribution with finite variance. The classic analysis of Houthakker and Taylor [1970] established that historical demand data often nicely fit a linear relationship. A wealth of past work on dynamic pricing has also posited linear demand models, although most prior research has not considered settings where the underlying model is changing over time [Keskin and Zeevi, 2014, Besbes and Zeevi, 2015, Cohen et al., 2016, Javanmard and Nazerzadeh, 2016, Javanmard, 2017].

Unlike standard statistical approaches to this problem which rely on stationarity, we suppose c t , B t may change every round and are possibly chosen adversarially. This consideration is particularly important in dynamic markets where the seller faces new competitors and consumers with ever-changing preferences who are actively seeking out the cheapest prices for products [Witt, 1986]. Our goal is to select prices p 1 , . . . , p T which minimize the expected regret E r R p p 1 , . . . , p T q GLYPH&lt;1&gt; R p p GLYPH&lt;6&gt; , . . . , p GLYPH&lt;6&gt; qs compared to always selecting the single best configuration of prices p GLYPH&lt;6&gt; GLYPH&lt;16&gt; argmin p P S E GLYPH&lt;176&gt; T t GLYPH&lt;16&gt; 1 R t p p q chosen in hindsight after the revenue functions R t have all been revealed.

Low regret algorithms ensure that in the case of a stationary underlying model, our chosen prices quickly converge to the optimal choice, and in nonstationary settings, our pricing procedure will naturally adapt to the intrinsic difficulty of the dynamic revenue-optimization problem [Shalev-Shwartz, 2011]. While low (i.e. o p T q ) regret is achievable using algorithms for online convex optimization with bandit feedback, the regret of existing methods is bounded below by Ω p ? N q , which is undesirable large when one is dealing with a vast number of products [Dani et al., 2007, Shalev-Shwartz, 2011, Flaxman et al., 2005]. To attain better bounds, we adopt a low-rank structural assumption that the variation in demands changes over time only due to d ! N underlying factors. Under this setting, we develop algorithms whose regret depends only on d rather than N by combining existing bandit methods with low-dimensional projections selected via online singular value decomposition. As far as we are aware, our main result (Theorem 3) is the first online bandit optimization algorithm whose regret provably does not scale with the action-space dimensionality.

Appendix D provides a glossary of notation used in this paper, and all proofs of our theorems are relegated to Appendix A. Throughout, C denotes a universal constant, whose value may change from line to line (but never depends on problem-specific constants such as T, d, r ).

## 2 Related Work

While bandit optimization has been successfully applied to dynamic pricing, research in this area has been primarily restricted to stationary settings [Kleinberg and Leighton, 2003, Besbes and Zeevi, 2009, den Boer and Bert, 2013, Keskin and Zeevi, 2014, Cohen et al., 2016, Misra et al., 2017]. Most similar to our work, Javanmard [2017] recently developed a bandit pricing strategy that presumes demand depends linearly on prices and product-specific features. High-dimensional dynamic pricing was also addressed by Javanmard and Nazerzadeh [2016] using sparse maximum likelihood. However, due to their reliance on stationarity, these approaches are less robust under evolving/adversarial environments compared with online optimization [Bubeck and Slivkins, 2012].

Beyond pricing, existing algorithms that combine bandits with subspace estimation [Gopalan et al., 2016, Djolonga et al., 2013, Sen et al., 2017] are solely designed for stationary ( stochastic ) settings rather than general online optimization (where the reward functions can vary adversarially over time). While the field of online bandit optimization has seen many advances since the pioneering work of Flaxman et al. [Flaxman et al., 2005], none of the recent improvements guarantees regret that is independent of the action-space dimension [Hazan and Levy, 2014, Bubeck et al., 2017]. To our knowledge, Hazan et al. [2016a] is the only prior work to present online optimization algorithms whose regret depends on an intrinsic low rank structure rather than the ambient dimensionality. However, their approach for online learning with experts is not suited for dynamic pricing since it is restricted to settings with: full-information (rather than bandit feedback), linear and noise-free (or stationary) reward functions, and actions that are specially constrained within the probability-simplex.

## 3 Low Rank Demand Model

We now introduce a special case of model (1) in which both c t and B t display low-rank changes over time. In practice, each product i may be described by some vector of features u i P R d (with d ! N ), which determine the similarity between products as well as their baseline demands. A natural method to gauge similarity between products i and j is via their inner product x u i , u j y V GLYPH&lt;16&gt; u T i Vu j under some linear transformation of the feature-space given by V ' 0 . For example, u i might be a binary vector indicating that product i falls into certain product-categories (where the number of categories d is far less than the number of products N ), and V might be a diagonal matrix specifying the cross-elasticity of demand within each product category. In this example, u T i Vu j GLYPH&lt;4&gt; p j would thus be the marginal effect on the demand for product i that results from selecting p j as the price for product j . Many recommender systems also assume products can be described using low-dimensional latent features that govern their desirability to consumers [Zhao et al., 2016, Sen et al., 2017].

By introducing time-varying metric transformations V t , our model allows these product-similarities to evolve over time. Encoding the features u i that represent each product as rows in a matrix U P R N GLYPH&lt;2&gt; d , we assume the following demand model, in which the temporal variation naturally exhibits low-rank structure:

<!-- formula-not-decoded -->

Here, the glyph[epsilon1] t P R N again reflect statistical noise in the observed demands, the z t P R d explain the variation in baseline demand over time, and the (asymmetric) matrices V t P R d GLYPH&lt;2&gt; d specify latent changes in the demand-price relationship over time. Under this model, the aggregate demand for product i at time t is governed by the prices of all products, weighted by their current feature-similarity to product i . To ensure our revenue-optimization remains convex, we restrict the adversary to choices that satisfy V t ' 0 for all t . Note that while the structural variation in our model is assumed to be low-rank, the noise in the observed demands may be intrinsically N -dimensional. In each round, p t and q t are the only quantities observed, while glyph[epsilon1] t , z t , V t all remain unknown (and we consider both cases where the product features U are known or unknown). In § 5.5, we verify that our low-rank assumption accurately describes real historical demand data.

## 4 Methods

Our basic dynamic pricing strategy is to employ the gradient-descent without a gradient (GDG) online bandit optimization technique of Flaxman et al. [2005]. While a naive application of this algorithm produces regret dependent on the number of products N , we ensure the updates of this method are only applied in the d -dimensional subspace spanned by U , which leads to regret bounds that depend only on d rather than N . When U is unknown, this subspace is simultaneously estimated online, in a somewhat similar fashion to the approach of Hazan et al. [2016a] for online learning with low-rank experts. If we define x GLYPH&lt;16&gt; U T p P R d , then under the low-rank model in (2) with E r glyph[epsilon1] t s GLYPH&lt;16&gt; 0 , the expected value of our revenue-objective in round t can be expressed as:

<!-- formula-not-decoded -->

As this problem's intrinsic dimensionality is only d , we can maximize expected revenues by merely considering a restricted set of d -dimensional actions x and functions f t over projected constraint set:

<!-- formula-not-decoded -->

## 4.1 Products with Known Features

In certain markets, it is clear how to featurize products [Cohen et al., 2016]. Under the low-rank model in (2) when U is given, we can apply the OPOK method (Algorithm 1) to select prices. This algorithm employs subroutines FINDPRICE and PROJECTION which both solve convex optimization problems in order to compute certain projections. Here, B d GLYPH&lt;16&gt; Unif pt x P R d : || x || 2 GLYPH&lt;16&gt; 1 uq denotes a uniform distribution over surface of the unit sphere in R d .

Intuitively, our algorithm adapts GDG to select low-dimensional actions x t P R d at each time point, and then seeks out a feasible price vector p t corresponding to the chosen x t . Note that when d ! N ,

| Algorithm 1 OPOK (Online Pricing Optimization with Known Features)                                                                                                                      | Algorithm 2 FINDPRICE( x ; U , S , p t GLYPH<1> 1 )                                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| R N GLYPH<2> d                                                                                                                                                                          | Input: x P R d , U P R N GLYPH<2> d , N N                                                                                                                                      |
| Input: η, δ,α ¡ 0 , U P , initial prices p 0 P S                                                                                                                                        | convex S GLYPH<128> R , p t GLYPH<1> 1 P R                                                                                                                                     |
| Output: Prices p 1 , . . . , p T to maximize revenue 1: Set prices to p 0 P S and observe q 0 p p 0 q ,R 0 p p 0 q 2: Define x 1 GLYPH<16> U T p 0 3: for t GLYPH<16> 1 , . . .,T : R d | Output: argmin p P S || p GLYPH<1> p t GLYPH<1> 1 || 2 subject to: U T p GLYPH<16> x                                                                                           |
|                                                                                                                                                                                         | Algorithm 3 PROJECTION( x , α , U , S )                                                                                                                                        |
| 4: ξ t GLYPH<18> Unif pt x P : || x || 2 GLYPH<16> 1 uq 5: r x t : GLYPH<16> x t GLYPH<0> δ ξ t 6: Set prices: p t GLYPH<16> FINDPRICE p r x t , U , S ,                                | Input: x P R d , α ¡ 0 , U P R N GLYPH<2> d , convex set S GLYPH<128> R N                                                                                                      |
| p t GLYPH<1> 1 q and observe q t p p t q ,R t p p t q                                                                                                                                   | Output: p 1 GLYPH<1> α q U T p p with p p : GLYPH<16> argmin p P S GLYPH<7> GLYPH<7> GLYPH<7> GLYPH<7> p 1 GLYPH<1> α q U T p GLYPH<1> x GLYPH<7> GLYPH<7> GLYPH<7> GLYPH<7> 2 |
| 7: x t GLYPH<0> 1 GLYPH<16> PROJECTION p x t GLYPH<1> ηR t p p t q ξ t , α , U , S q                                                                                                    |                                                                                                                                                                                |

there are potentially many price-vectors p P R N that map to the same low-dimensional vector x P R d via U T . Out of these, we select the one that is closest to our previously-chosen prices (via FINDPRICE), ensuring additional stability in our dynamic pricing procedure. In practice, the initial prices p 0 should be selected based on external knowledge or historical demand data.

Under mild conditions, Theorem 1 below states that the OPOK algorithm incurs O p T 3 { 4 ? d q regret when product features are a priori known. This result is derived from Lemma A.1 which shows that Step 7 of our algorithm corresponds (in expectation) to online projected gradient descent on a smoothed version of our objective defined as:

<!-- formula-not-decoded -->

where ζ is sampled uniformly from within the unit sphere in R d , and f t is defined in (3). We bound the regret of our pricing algorithm under the following assumptions (which ensure the revenue functions are bounded/smooth and the set of feasible prices is bounded/well-scaled):

- (A1) || z t || 2 / b for t GLYPH&lt;16&gt; 1 , . . . , T
- (A2) || V t || op / b for all t ( || GLYPH&lt;4&gt; || op denotes spectral norm)
- (A3) T ¡ 9 4 d 2
- (A4) U is an orthogonal matrix such that U T U GLYPH&lt;16&gt; I d GLYPH&lt;2&gt; d
- (A5) S GLYPH&lt;16&gt; t p P R N : || p || 2 / r u (with r ¥ 1 )

Requiring that the columns of U form an orthonormal basis for R d , condition (A4) can be easily enforced (when d GLYPH&lt;160&gt; N ) by first orthonormalizing the product features. Note that this orthogonality condition does not restrict the overall class of models specified in (2), and describes the case where the features used to encode each product are uncorrelated between products (i.e. a minimally-redundant encoding) and have been normalized across all products. To see why (A4) does not limit the allowed price-demand relationships, consider that we can re-express any (non-orthogonal) U GLYPH&lt;16&gt; OP in terms of orthogonal O P R N GLYPH&lt;2&gt; d . The demand model in (2) can then be equivalently expressed in terms of z 1 t GLYPH&lt;16&gt; Pz t , V 1 t GLYPH&lt;16&gt; PV t P T (after appropriately redefining the constant b in (A1)-(A2)), since: Uz t GLYPH&lt;1&gt; UV t U T p t GLYPH&lt;16&gt; Oz 1 t GLYPH&lt;1&gt; OV 1 t O T p t . To further simplify our analysis, we also from now adopt (A5) presuming the constraint set of feasible product-prices is a centered Euclidean ball (implying our p t , q t vectors now represent appropriately shifted/scaled prices and demands).

Theorem 1. Under assumptions (A1)-(A5), if we choose η GLYPH&lt;16&gt; 1 b p 1 GLYPH&lt;0&gt; d q ? T , δ GLYPH&lt;16&gt; T GLYPH&lt;1&gt; 1 { 4 b dr 2 p 1 GLYPH&lt;0&gt; r q 9 r GLYPH&lt;0&gt; 6 , α GLYPH&lt;16&gt; δ r , then there exists C ¡ 0 such that for any p P S :

<!-- formula-not-decoded -->

for the prices p 1 , . . . , p T selected by the OPOK algorithm.

Theorem A.2 shows the same O p T 3 { 4 ? d q regret bound holds for the OPOK algorithm under relaxed conditions solely based on the revenue functions and feasible prices rather than the specific properties of our low-rank structure assumed in (A1)-(A5).

## 4.2 Products with Latent Features

In many settings, it is not clear how to best represent products as feature-vectors. Once again adopting the low-rank demand model in (2), we now consider the case where U is unknown and must be estimated. We presume the orthogonality condition (A4) holds throughout this section (recall this does not restrict the class of allowed models), which implies U is both an isometry as well as the rightinverse of U T . Thus, given any low-dimensional action x P U T p S q , we can set the corresponding prices as p GLYPH&lt;16&gt; Ux such that U T p GLYPH&lt;16&gt; x . Lemma 1 shows that this price selection-method is feasible and corresponds to changing Step 6 in the OPOK algorithm to p t GLYPH&lt;16&gt; FINDPRICE p r x t , U , S , 0 q , where the next price is regularized toward the origin rather than the previous price p t GLYPH&lt;1&gt; 1 . Because prices p t are multiplied by the noise term glyph[epsilon1] t within each revenue-function R t , choosing minimum-norm prices can help reduce variance in the total revenue generated by our approach. As U is unknown, we instead employ an estimate p U P R N GLYPH&lt;2&gt; d , which is always restricted to be an orthogonal matrix.

Lemma 1. For any orthogonal matrix p U and any x P p U T p S q , define p p GLYPH&lt;16&gt; p Ux P R N . Under (A5): p p P S and p p GLYPH&lt;16&gt; FINDPRICE ( x , p U , S , 0 ).

Product Features with Known Span. In Theorem 2, we consider a minorly modified OPOK algorithm where price-selection in Step 6 is done using p t GLYPH&lt;16&gt; p U r x t rather than being regularized toward the previous price p t GLYPH&lt;1&gt; 1 . Even without knowing the true latent features, this result implies that the regret of our modified OPOK algorithm may still be bounded independently of the number of products N , as long as p U accurately estimates the column span of U .

Theorem 2. Suppose span p p U q GLYPH&lt;16&gt; span p U q , i.e. our orthogonal estimate has the same column-span as the underlying (rank d ) latent product-feature matrix. Let p 1 , . . . , p T P S denote the prices selected by our modified OPOK algorithm with p U used in place of the underlying U and η, δ, α chosen as in Theorem 1. Under conditions (A1)-(A5), there exists C ¡ 0 such that for any p P S :

<!-- formula-not-decoded -->

Features with Unknown Span and Noise-free Demands. In practice, span( U ) may be entirely unknown. If we assume the adversary is restricted to strictly positive-definite V t ¡ 0 for all t and there is no statistical noise in the observed demands (i.e. q t GLYPH&lt;16&gt; Uz t GLYPH&lt;1&gt; UV t U T p t in each round), then Lemma 2 below shows we can ensure span( U ) is revealed within the first d observed demand vectors by simply adding a minuscule random perturbation to all of our initial prices selected in the first d rounds. Thus, even without knowing the latent product feature subspace, an absence of noise in the observed demands enables us to realize a low regret pricing strategy via the same modified OPOK algorithm (applied after the first d rounds).

Lemma 2. Suppose that for t GLYPH&lt;16&gt; 1 , . . . , T : glyph[epsilon1] t GLYPH&lt;16&gt; 0 and V t ¡ 0 . If each p t is independently uniformly distributed within some (uncentered) Euclidean ball of strictly positive radius, then span p q 1 , . . . , q d q GLYPH&lt;16&gt; span p U q almost surely.

Features with Unknown Span and Noisy Demands. When the observed demands are noisy and span p U q is unknown, we select prices using the OPOL algorithm on the next page. The approach is similar to our previous OPOK algorithm, except we now additionally maintain a changing estimate of the latent product features' span. Our estimate is updated in an online fashion via an averaged singular value decomposition (SVD) of the previously observed demands.

Step 9 in our OPOL algorithm corresponds to online averaging of the currently observed demand vector q t with the historical observations stored in the j th column of matrix p Q . After computing the singular value decomposition of p Q GLYPH&lt;16&gt; r U r S r V T , Step 10 is performed by setting p U equal to the first d columns of r U (presumed to be the indices corresponding to the largest singular values in r S ). Since p Q is only minorly changed within each round, the update operation in Step 10 can be computed more efficiently by leveraging existing fast SVD-update procedures [Brand, 2006, Stange, 2008]. Note that by their definition as singular vectors, the columns of p U remain orthonormal throughout the execution of our algorithm.

| Algorithm 4 OPOL (Online Pricing Optimization with Latent Features)                  | Algorithm 4 OPOL (Online Pricing Optimization with Latent Features)                                                                                             |
|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input: η, δ,α ¡ 0 , rank d P r 1 ,N s , initial prices p 0 P S Output: p , . . . , p | Input: η, δ,α ¡ 0 , rank d P r 1 ,N s , initial prices p 0 P S Output: p , . . . , p                                                                            |
| 1:                                                                                   | Initialize p Q as N GLYPH<2> d matrix of zeros                                                                                                                  |
| 2:                                                                                   | Initialize p U as random N GLYPH<2> d orthogonal matrix                                                                                                         |
| 3:                                                                                   | Set prices to p 0 P S and observe q 0 p p 0 q ,R 0 p p 0 q                                                                                                      |
| 4:                                                                                   | Define x 1 GLYPH<16> p U T p 0                                                                                                                                  |
| 5:                                                                                   | for t GLYPH<16> 1 , . . .,T :                                                                                                                                   |
| 6:                                                                                   | r x t : GLYPH<16> x t GLYPH<0> δ ξ t , ξ t GLYPH<18> Unif pt x P R d : || x || 2 GLYPH<16> 1 uq                                                                 |
| 7:                                                                                   | Set prices: p t GLYPH<16> p U r x t and observe q t p p t q ,R t p p t q                                                                                        |
| 8:                                                                                   | x t GLYPH<0> 1 GLYPH<16> PROJECTION p x t GLYPH<1> ηR t p p t q ξ t , α , p U , S q                                                                             |
| 9:                                                                                   | With j GLYPH<16> 1 GLYPH<0> rp t GLYPH<1> 1 q mod d s , k GLYPH<16> floor p t { d q , update: p Q GLYPH<6> ,j - 1 k q t GLYPH<0> k GLYPH<1> 1 k p Q GLYPH<6> ,j |
| 10:                                                                                  | Set columns of p U as top d left singular vectors of p Q                                                                                                        |

To quantify the regret incurred by this algorithm, we assume the noise vectors glyph[epsilon1] t follow a subGaussian distribution for each t GLYPH&lt;16&gt; 1 , . . . , T . The assumption of sub-Gaussian noise is quite general, covering common settings where the noise is Gaussian, bounded, of strictly log-concave density, or any finite mixture of sub-Gaussian variables [Mueller et al., 2018]. Intuitively, the averaging in step 9 of our OPOL algorithm ensures statistical concentration of the noise in our observed demands, such that the true column span of the underlying U may be better revealed. More concretely, if we let s t GLYPH&lt;16&gt; z t GLYPH&lt;1&gt; V t U T p t and q GLYPH&lt;6&gt; t GLYPH&lt;16&gt; Us t , then the observed demands can be written as: q t GLYPH&lt;16&gt; q GLYPH&lt;6&gt; t GLYPH&lt;0&gt; glyph[epsilon1] t , where q GLYPH&lt;6&gt; t are the (unobserved) expected demands at our chosen prices. Thus, the j th column of p Q at round T is given by:

<!-- formula-not-decoded -->

where we assume for notational simplicity that T is divisible by d and define I j GLYPH&lt;16&gt; t j GLYPH&lt;0&gt; d p i GLYPH&lt;1&gt; 1 q : i GLYPH&lt;16&gt; 1 , . . . , T d u (so | I j | GLYPH&lt;16&gt; T d ). Because the average 1 | I j | GLYPH&lt;176&gt; i P I j glyph[epsilon1] i exhibits concentration of measure, results from random matrix theory imply that the span-estimator obtained from the first d singular vectors of p Q in Step 10 of our OPOL algorithm will rapidly converge to the column span of s Q GLYPH&lt;6&gt; P R N GLYPH&lt;2&gt; d , a matrix of averaged underlying expected demands. This is useful since s Q GLYPH&lt;6&gt; shares the same span as the underlying U .

Theorem 3 below shows that our OPOL algorithm achieves low-regret in the setting of unknown product features with noisy demands, and the regret again depends only on the intrinsic rank d (rather than the number of products N ).

Theorem 3. For unknown U , let p 1 , . . . , p T be the prices selected by the OPOL algorithm with η, δ, α set as in Theorem 1. Suppose glyph[epsilon1] t follows a sub-Gaussian p σ 2 q distribution and has statistically i.i.d. dimensions for each t . If (A1)-(A5) hold, then there exists C ¡ 0 such that for any p P S :

<!-- formula-not-decoded -->

Here, Q GLYPH&lt;16&gt; max ! 1 , σ 2 GLYPH&lt;1&gt; 2 σ 1 GLYPH&lt;0&gt; 1 σ 2 d GLYPH&lt;9&gt;) with σ 1 (and σ d ) defined as the largest (and smallest) nonzero singular values of the underlying rank d matrix s Q GLYPH&lt;6&gt; defined in (6).

Our proof of this result relies on standard random matrix concentration inequalities [Vershynin, 2012] and Theorem A.3, a useful variant of the Davis-Kahan theory introduced by Yu et al. [2015]. Intuitively, we show that span( U ) can be estimated to sufficient accuracy within sufficiently few rounds, and then follow similar reasoning to the proof of Theorem 2. Note that the regret in Theorem 3 depends on the constant Q whose value is determined by the noise-level σ and the extreme singular values of s Q GLYPH&lt;6&gt; defined in (6). In general, these quantities thus measure just how adversarial of an environment the seller is faced with. For example, when the underlying low-rank variation is of much smaller magnitude than the noise in our observations, it will be difficult to accurately estimate

the span of the latent product features. In control theory, a signal-to-noise expression similar to Q has also been recently proposed to quantify the intrinsic difficulty of system identification for the linear quadratic regulator [Dean et al., 2017]. A basic setting in which Q can be explicitly bounded is illustrated in Appendix B, where we suppose the underlying demand model parameters can only be imprecisely controlled by an adversary over time.

## 5 Experiments

We evaluate the performance of our methodology in settings where noisy demands are generated according to equation (2), and the underlying structural parameters of the demand curves are randomly sampled from Gaussian distributions (details in Appendix C.2). Throughout, p t and q t represent rescaled rather than absolute prices/demands, such that the feasible set S can be simply fixed as a centered sphere of radius r GLYPH&lt;16&gt; 20 . Noise in the (rescaled) demands for each individual product is always sampled as: glyph[epsilon1] t GLYPH&lt;18&gt; N p 0 , 10 q .

Our proposed algorithms are compared against the GDG online bandit algorithm of Flaxman et al. [2005], as well as a simple explore-then-exploit ( Explo re it ) technique. The latter method randomly samples p t during the first T 3 { 4 rounds (uniformly over S ) and for all remaining rounds, p t is fixed at the best price vector found during exploration. Explo re it reflects a standard pricing technique: initially experiment with prices and eventually settle on those that previously produced the most profit.

## 5.1 Stationary Demand Model

First, we consider a stationary setting where underlying structural parameters z t , GLYPH&lt;16&gt; z , V t GLYPH&lt;16&gt; V remain fixed. Before each experiment, we sample the entries of z , V independently as z ij GLYPH&lt;18&gt; N p 100 , 20 q , V ij GLYPH&lt;18&gt; N p 0 , 2 q , and U is fixed as a random sparse binary matrix that reflects which of d possible categories each product belongs to. Subsequently, we orthogonalize the columns of U and project V into V GLYPH&lt;16&gt; t V : V T GLYPH&lt;0&gt; V ' λ I u with λ GLYPH&lt;16&gt; 10 to ensure positive definite cross-product price elasticities. Here, z , V , λ are chosen to reflect properties of real-world demand curves: different products' baseline demands and elasticities should be highly diverse (wide range of z ), and prices should significantly influence demands such that price-increases severely decrease demand and affect demand for the same product more than other products (large value of λ , which in turn induces large values for certain entries of V ). We find the optimal price vector does not lie near the boundary of S ( || p GLYPH&lt;6&gt; || 2 GLYPH&lt;19&gt; 8 rather than 20), which shows that prices strongly influence demands under our setup.

Figures 1A and 1B show that our OPOK and OPOL algorithms are greatly superior to GDG when the dimensionality N exceeds the intrinsic rank d . When N GLYPH&lt;16&gt; d (no low-rank structure to exploit), our OPOK/OPOL algorithms closely match GDG (blue, green, and red curves overlap). Note that in this case: GDG and OPOK are nearly mathematically equivalent (same regret bound applies to both, but their empirical performance slightly differs in this case due to the internal stochasticity of each bandit algorithm), as are OPOL and OPOK (since d GLYPH&lt;16&gt; N implies p U is an orthogonal N GLYPH&lt;2&gt; N matrix and hence invertible). For small N , all online bandit optimization techniques outperform Explo re it , but GDG scales poorly to large N unlike our methods. Interestingly, OPOL (which must infer latent product features alongside the pricing strategy) performs slightly better than the OPOK approach, which has access to the ground-truth features. This is because in the presence of noise, our SVD-computed features can more robustly represent the subspace where projected pricing variation can maximally impact the overall observed demands. In contrast, the dimensionality-reduction in OPOK does not lead to any denoising.

## 5.2 Model with Demand Shocks

Next, we study a non-stationary setting where the underlying demand model changes drastically at times T { 3 and 2 T { 3 . At the start of each period r 0 , T { 3 s , r T { 3 , 2 T { 3 s , r 2 T { 3 , T s : we simply redraw the underlying structural parameters z t , V t from the same Gaussian distributions used for the stationary setting. Figures 1C and 1D show that our bandit techniques quickly adapt to the changes in

<!-- image -->

T

Figure 1: Average cumulative regret (over 10 repetitions with standard-deviations shaded) of various pricing strategies when underlying demand model is: (A)-(B) stationary over time, (C)-(D) : altered by structural shocks at times T { 3 and 2 T { 3 , (E)-(F) : drifting over time.

the underlying demand curves. The regret of the bandit algorithms decreases over time, indicating they begin to outperform the optimal fixed price chosen in hindsight (recall that our bandits may vary price over time, whereas regret is measured against the best fixed price-configuration which may fare much worse than a dynamic schedule in nonstationary environments). Once again, our low-rank methods achieve low regret for a large number of products unlike the existing approaches, while retaining the same strong performance as the GDG algorithm in the absence of low-rank structure.

## 5.3 Drifting Demand Model

Finally, we consider another non-stationary setting where underlying demand curves slowly change over time. Here, the underlying structural parameters z t , V t are initially drawn from the same previously used Gaussian distributions at t GLYPH&lt;16&gt; 0 , but then begin to stochastically drift over time according to: z t GLYPH&lt;0&gt; 1 GLYPH&lt;16&gt; z t GLYPH&lt;0&gt; w , V t GLYPH&lt;0&gt; 1 GLYPH&lt;16&gt; Π V p V t GLYPH&lt;0&gt; W q . Here, the entries of w and W are i.i.d. samples from N p 0 , 1 q and N p 0 , 0 . 1 q distributions, respectively, and Π V denotes the projection of a matrix into the strongly positive-definite set V we previously defined. Figures 1E and 1F illustrate how our bandit pricing approach can adapt to ever-changing demand curves. Again, our low-rank methods exhibit much stronger performance than GDG and Explo re it in the settings with many products.

## (A) Model (1) without temporal change

Figure 2: Regret of pricing strategies (for N GLYPH&lt;16&gt; 100 ) when underlying demand model has no low-rank structure (see Appendix C.1) and is: (A) stationary, (B) altered by shocks at T { 3 and 2 T { 3 as in § 5.2.

<!-- image -->

<!-- image -->

## 5.4 Misspecified Demand Model

Appendix C.1investigates the robustness of our algorithms in misspecified settings with full-rank or log-linear demands, where the assumptions of our demand model are explicitly violated. Even in the absence of explicit low-rank structure, running the OPOL algorithm with low values of d substantially outperforms other pricing strategies (Figure 2). These empirical results suggest that our OPOL algorithm is practically useful for various high-dimensional pricing problems, beyond those that exactly satisfy the low-rank/linearity assumptions in (2).

## 5.5 Rank of Historical Demand Data

While the aforementioned robustness analysisindicates our approach works well even when key assumptions are violated, it remains of interest whether our assumptions accurately describe actual demand variation for real products. One key implication of our assumptions in (2) is that the N GLYPH&lt;2&gt; T matrix Q GLYPH&lt;16&gt; r q 1 ; q 2 ; . . . ; q T s , whose columns contain the observed demands in each round, should be approximately low-rank when there is limited noise in the demand-price relationship. This is because under our assumptions, q 1 , . . . , q T only span a d -dimensional subspace in the absence of noise (see proof of Lemma 2).

Here, we study historical demand data 1 for 1,340 products sold at various prices over 7 weeks by the baking company Grupo Bimbo. Using this data, we form a matrix Q whose columns contain the total weekly demands for each product across all stores. The SVD of Q reveals the following percentages of variation in the observed demands are captured within the top k singular vectors: k GLYPH&lt;16&gt; 1 : 97 . 1% , k GLYPH&lt;16&gt; 2 : 99 . 1% , k GLYPH&lt;16&gt; 3 : 99 . 9% . This empirical analysis thus suggests that our low-rank assumption on the expected demand variation remains reasonable in practice.

## 6 Discussion

By exploiting a low-rank structural condition that naturally emerges in dynamic pricing problems, this work introduces an online bandit optimization algorithm whose regret provably depends only on the intrinsic rank of the problem rather than the ambient dimensionality of the action space. Our low-rank bandit approach to dynamic pricing scales to a large number of products with intercorrelated demand curves, even if the underlying demand model varies over time in an adversarial fashion. When applied to various high-dimensional dynamic pricing systems involving stationary, fluctuating, and misspecified demand curves, our approach empirically outperforms standard bandit methods. Future extensions of this work could include adaptations for predictable sequences in which future demands can be partially forecasted [Rakhlin and Sridharan, 2013], or generalizing our convex formulation and linear demand model to more general subspace structures [Hazan et al., 2016b].

1 Historical demand data obtained from: www.kaggle.com/c/grupo-bimbo-inventory-demand/

## References

- O. Besbes and A. Zeevi. Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. Operations Research , 57:1407-20, 2009.
- O. Besbes and A. Zeevi. On the surprising sufficiency of linear models for dynamic pricing with demand learning. Management Science , 61:723-39, 2015.
- M. Brand. Fast low-rank modifications of the thin singular value decomposition. Linear Algebra and its Applications , 415:20-30, 2006.
- S. Bubeck and A. Slivkins. The best of both worlds: Stochastic and adversarial bandits. Conference on Learning Theory , 2012.
- S. Bubeck, Y. T. Lee, and R. Eldan. Kernel-based methods for bandit convex optimization. Proceedings of 49th Annual ACM SIGACT Symposium on the Theory of Computing , 2017.
- M. Cohen, I Lobel, and R. P. Leme. Feature-based dynamic pricing. ACM Conference on Economics and Computation , 2016.
7. V Dani, T. P. Hayes, and S. M. Kakade. The price of bandit information for online optimization. Neural Information Processing Systems , 2007.
- S. Dean, H. Mania, N. Matni, B. Recht, and S. Tu. On the sample complexity of the linear quadratic regulator. arXiv:1710.01688 , 2017.
- A. V. den Boer and Z. Bert. Simultaneously learning and optimizing using controlled variance pricing. Management Science , 60:770-83, 2013.
- J. Djolonga, A. Krause, and V. Cevher. High-dimensional gaussian process bandits. Neural Information Processing Systems , 2013.
- A. D. Flaxman, A. T. Kalai, and H. B. McMahan. Online convex optimization in the bandit setting: Gradient descent without a gradient. Proceedings of the 16th Annual ACM-SIAM Symposium on Discrete Algorithms , 2005.
- A. Gopalan, O. Maillard, and M. Zaki. Low-rank bandits with latent mixtures. arXiv:1609.01508 , 2016.
- E. Hazan and K. Y. Levy. Bandit convex optimization: Towards tight bounds. Neural Information Processing Systems , 2014.
- E. Hazan, T. Koren, R. Livni, and Y. Mansour. Online learning with low rank experts. Conference on Learning Theory , 2016a.
- E. Hazan, K. Y. Levy, and S. Shalev-Shwartz. On graduated optimization for stochastic non-convex problems. International Conference on Machine Learning , 2016b.
- H. S. Houthakker and L. D. Taylor. Consumer demand in the United States . Harvard University Press, 1970.
- A. Javanmard. Perishability of data: Dynamic pricing under varying-coefficient models. Journal of Machine Learning Research , 18:1-31, 2017.
- A. Javanmard and H. Nazerzadeh. Dynamic pricing in high-dimensions. arXiv:arXiv:1609.07574 , 2016.
- N. B. Keskin and A. Zeevi. Dynamic pricing with an unknown demand model: asymptotically optimal semi-myopic policies. Operations Research , 62:1142-67, 2014.
- R. Kleinberg and T. Leighton. The value of knowing a demand curve: Bounds on regret for online posted-price auctions. Proceedings of the 44th Annual IEEE Symposium on Foundations of Computer Science , 2003.

- K. Misra, E. M. Schwartz, and J. Abernethy. Dynamic online pricing with incomplete information using multi-armed bandit experiments. Available at SSRN: http: // ssrn. com/ abstract= 2981814 , 2017.
- J. Mueller, T. Jaakkola, and D. Gifford. Modeling persistent trends in distributions. Journal of the American Statistical Association , 113:1296-1310, 2018.
- A. Rakhlin and K. Sridharan. Online learning with predictable sequences. Conference on Learning Theory , 2013.
- P. Rigollet. High dimensional statistics, 2015. MIT Opencourseware: ocw.mit.edu/courses/ mathematics/18-s997-high-dimensional-statistics-spring-2015/lecture-notes/ .
- M. Rudelson and R. Vershynin. The Littlewood-Offord problem and invertibility of random matrices. Advances in Mathematics , 218:600-33, 2008.
- R. Sen, K. Shanmugam, M. Kocaoglu, A. Dimakis, and S. Shakkottai. Contextual bandits with latent confounders: An NMF approach. Artificial Intelligence and Statistics , 2017.
7. Shai Shalev-Shwartz. Online learning and online convex optimization. Foundations and Trends in Machine Learning , 4:107-194, 2011.
- P. Stange. On the efficient update of the singular value decomposition. Proceedings in Applied Mathematics and Mechanics , 8:10827-28, 2008.
- R. Vershynin. Introduction to the non-asymptotic analysis of random matrices. In Y. Eldar and G. Kutyniok, editors, Compressed Sensing, Theory and Applications , pages 210-268. Cambridge University Press, 2012.
- U. Witt. How can complex economical behavior be investigated? The example of the ignorant monopolist revisited. Behavioral Science , 31:173-188, 1986.
- Y. Yu, T. Wang, and R. Samworth. A useful variant of the Davis-Kahan theorem for statisticians. Biometrika , 102:315-323, 2015.
- F. Zhao, M. Xiao, and Y. Guo. Predictive collaborative filtering with side information. International Joint Conference on Artificial Intelligence , 2016.

## Supplementary Material: Low-Rank Bandit Methods for High-Dimensional Dynamic Pricing

## Contents

| A Proofs and Auxiliary Theoretical Results   | A Proofs and Auxiliary Theoretical Results   | A Proofs and Auxiliary Theoretical Results   |   2 |
|----------------------------------------------|----------------------------------------------|----------------------------------------------|-----|
|                                              |                                              | Lemma A.1 . . . . . . . . . . . .            |   2 |
|                                              |                                              | Theorem A.1 . . . . . . . . . . .            |   2 |
|                                              | A.1                                          | Alternative OPOK Regret Bound .              |   2 |
|                                              |                                              | Theorem A.2 . . . . . . . . . . .            |   2 |
|                                              | A.2                                          | Proof of Theorem 1 . . . . . . . .           |   3 |
|                                              |                                              | Lemma A.2 . . . . . . . . . . . .            |   3 |
|                                              | A.3                                          | Proof of Lemma 1 . . . . . . . . .           |   3 |
|                                              | A.4                                          | Proof of Theorem 2 . . . . . . . .           |   4 |
|                                              | A.5                                          | Proof of Lemma 2 . . . . . . . . .           |   4 |
|                                              |                                              | Theorem A.3 . . . . . . . . . . .            |   5 |
|                                              | A.6                                          | Proof of Theorem 3 . . . . . . . .           |   5 |
| B                                            | Pricing against an Imprecise Adversary       | Pricing against an Imprecise Adversary       |   8 |
|                                              |                                              | Theorem B.4 . . . . . . . . . . .            |   8 |
| C                                            | Additional Experimental Results              | Additional Experimental Results              |  10 |
|                                              | C.1                                          | Misspecified Demand Models . .               |  10 |
|                                              | C.2                                          | Further Details about Experiments            |  11 |
| D                                            | Notation Glossary                            | Notation Glossary                            |  12 |

## A Proofs and Auxiliary Theoretical Results

Lemma A.1. For p P R N with U T p GLYPH&lt;16&gt; x GLYPH&lt;0&gt; δ ξ P R d , ξ GLYPH&lt;18&gt; Unif pt x P R d : || x || 2 GLYPH&lt;16&gt; 1 uq :

<!-- formula-not-decoded -->

Proof. Since we have: E glyph[epsilon1] r R t p p qs GLYPH&lt;16&gt; f t p x GLYPH&lt;0&gt; δ ξ q , this result follows directly from Lemma 2.1 in Flaxman et al. [2005].

Theorem A.1 (Flaxman et al., 2005) . Suppose for t GLYPH&lt;16&gt; 1 , . . . , T , each f t P rGLYPH&lt;1&gt; B,B s is a convex, L -Lipschitz function of x P R d , and the set of feasible actions U GLYPH&lt;128&gt; R d is convex, with Euclidean balls of radius r GLYPH&lt;210&gt; and r GLYPH&lt;211&gt; containing and contained-within U , respectively. Let x 1 , . . . , x T P R d denote the iterates of the GDG algorithm applied to f 1 , . . . , f T (i.e. online projected stochastic gradient descent applied to the p f t as defined in (5)). If we choose η, δ, α as in Theorem A.2, then:

<!-- formula-not-decoded -->

## A.1 Alternative OPOK Regret Bound

We provide another bound on the regret of our pricing algorithm that is similar to Theorem 1, but only relies on direct properties of the prices and revenue functions rather than properties of our assumed low-rank structure.

The following assumptions are adopted (revenue functions are bounded/smooth, and the set of feasible prices is bounded/well-scaled):

(A6) U T p S q contains a Euclidean ball of radius r GLYPH&lt;211&gt; and is contained within a ball of radius r GLYPH&lt;210&gt; ¥ r GLYPH&lt;211&gt;

- (A7) T ¡ GLYPH&lt;0&gt; 3 dr GLYPH&lt;210&gt; 2 r GLYPH&lt;211&gt; GLYPH&lt;8&gt; 2 (the number of pricing rounds is large)
- (A8) GLYPH&lt;7&gt; GLYPH&lt;7&gt; E r R t p p qs GLYPH&lt;7&gt; GLYPH&lt;7&gt; / B for all p P S , t GLYPH&lt;16&gt; 1 , . . . , T
- (A9) f t p x q is L -Lipschitz over x P U T p S q for t GLYPH&lt;16&gt; 1 , . . . , T

Theorem A.2. If conditions (A6)-(A9) are met and we choose η GLYPH&lt;16&gt; r GLYPH&lt;210&gt; B ? T , δ GLYPH&lt;16&gt; T GLYPH&lt;1&gt; 1 { 4 b Bdr GLYPH&lt;210&gt; r GLYPH&lt;211&gt; 3 p Lr GLYPH&lt;211&gt; GLYPH&lt;0&gt; B q , α GLYPH&lt;16&gt; δ r GLYPH&lt;211&gt; , then there exists C ¡ 0 such that for any p P S :

<!-- formula-not-decoded -->

for the prices p 1 , . . . , p T selected by the OPOK algorithm.

Proof. Condition (A8) implies the range of f t bounded by B over x P U T p S q . Recall that each f t is a convex function of x (as we required each V t ' 0 ) and for any p P S , we can define x GLYPH&lt;16&gt; U T p P U T p S q such that: E glyph[epsilon1] r R t p p qs GLYPH&lt;16&gt; f t p x q . Since convexity of S implies U T p S q is also convex, the proof of our result immediately follows from Theorem 3.3 in Flaxman et al. [2005], which is also restated here as Theorem A.1. Finally, we note that since both S and U T p S q are convex, our choice of η, δ, α ensures r x t P U T p S q and hence p t P S for all t .

## A.2 Proof of Theorem 1

Theorem 1. Under assumptions (A1)-(A5), if we choose η GLYPH&lt;16&gt; 1 b p 1 GLYPH&lt;0&gt; d q ? T , δ GLYPH&lt;16&gt; T GLYPH&lt;1&gt; 1 { 4 b dr 2 p 1 GLYPH&lt;0&gt; r q 9 r GLYPH&lt;0&gt; 6 , α GLYPH&lt;16&gt; δ r , then there exists C ¡ 0 such that for any p P S :

<!-- formula-not-decoded -->

for the prices p 1 , . . . , p T selected by the OPOK algorithm.

Proof. We show that (A1)-(A5) imply Theorem A.2 holds with r GLYPH&lt;210&gt; GLYPH&lt;16&gt; r GLYPH&lt;211&gt; GLYPH&lt;16&gt; r , B GLYPH&lt;16&gt; rb p 1 GLYPH&lt;0&gt; r q , and L GLYPH&lt;16&gt; p 2 r GLYPH&lt;0&gt; 1 q b . Bounding and simplifying the inequality then produces the desired result. Note that (A8) holds since:

<!-- formula-not-decoded -->

We also have Lipschitz continuity as required in (A9), since for all x P U T p S q :

<!-- formula-not-decoded -->

Finally, Lemma A.2 below implies (A6) holds with r GLYPH&lt;210&gt; GLYPH&lt;16&gt; r GLYPH&lt;211&gt; GLYPH&lt;16&gt; r .

Lemma A.2. For any orthogonal N GLYPH&lt;2&gt; d matrix U and p P S , condition (A5) implies:

<!-- formula-not-decoded -->

Proof. Consider the orthogonal extension of U , a matrix W GLYPH&lt;16&gt; r U , r U s P R N GLYPH&lt;2&gt; N formed by appending N GLYPH&lt;1&gt; d additional orthonormal columns to U that are also orthogonal to the columns of U . For any p P R N , we have:

<!-- formula-not-decoded -->

Combined with (A5), this implies UU T p p q P S and || x || 2 / r for any x P U T p S q . Now fix arbitrary x P R d which satisfies || x || 2 / r . By orthogonality of U :

<!-- formula-not-decoded -->

## A.3 Proof of Lemma 1

Lemma 1. For any orthogonal matrix p U and any x P p U T p S q , define p p GLYPH&lt;16&gt; p Ux P R N . Under condition (A5): p p P S and p p GLYPH&lt;16&gt; FINDPRICE ( x , p U , S , 0 ).

Proof. Given x P p U T p S q , there exists p P S with p U T p GLYPH&lt;16&gt; x . The proof of Lemma A.2 implies || p p || 2 / || p || 2 and p p GLYPH&lt;16&gt; p Ux GLYPH&lt;16&gt; p U p U T p P S when this set is a centered Euclidean ball. Finally, we note that p U T p p GLYPH&lt;16&gt; x since p U T p U GLYPH&lt;16&gt; I d GLYPH&lt;2&gt; d , so p p is the minimum-norm vector in S which is mapped to x by p U T .

## A.4 Proof of Theorem 2

Theorem 2. Suppose span p p U q GLYPH&lt;16&gt; span p U q , i.e. our orthogonal estimate has the same column-span as the underlying (rank d ) latent product-feature matrix. Let p 1 , . . . , p T P S denote the prices selected by our modified OPOK algorithm with p U used in place of the underlying U and η, δ, α chosen as in Theorem 1. Under conditions (A1)-(A5), there exists C ¡ 0 such that for any p P S :

<!-- formula-not-decoded -->

Proof. Define s p GLYPH&lt;16&gt; argmin p P S E glyph[epsilon1] T , t GLYPH&lt;16&gt; 1 R t p p q , p GLYPH&lt;6&gt; GLYPH&lt;16&gt; UU T s p . Note that E glyph[epsilon1] GLYPH&lt;17&gt; GLYPH&lt;176&gt; T t GLYPH&lt;16&gt; 1 R t p p GLYPH&lt;6&gt; q GLYPH&lt;25&gt; GLYPH&lt;16&gt;

E glyph[epsilon1] GLYPH&lt;17&gt; GLYPH&lt;176&gt; T t GLYPH&lt;16&gt; 1 R t p s p q GLYPH&lt;25&gt; and p GLYPH&lt;6&gt; P S by Lemma A.2, so p GLYPH&lt;6&gt; is an equivalently optimal setting of the product prices. Since U and p U share the same column-span, there exists low-dimensional action x GLYPH&lt;6&gt; P R k such that p GLYPH&lt;6&gt; GLYPH&lt;16&gt; p Ux GLYPH&lt;6&gt; . By orthogonality of p U : p U T r p GLYPH&lt;16&gt; p U T p Ux GLYPH&lt;6&gt; GLYPH&lt;16&gt; x GLYPH&lt;6&gt; , so x GLYPH&lt;6&gt; P p U T p S q is a feasible solution to our modified OPOK algorithm. For x P R d and p GLYPH&lt;16&gt; p Ux P R N , we re-express the expected revenue at this price vector by introducing f t, p U as a function of x parameterized by p U , as similarly done in (3):

<!-- formula-not-decoded -->

Convexity of R t in p implies f t, p U is convex in x for any p U . Note that our modified OPOK algorithm is (in expectation) running online projected gradient descent on a smoothed version of each f t, p U , defined similarly as in (5). Via the same argument employed in the previous section (based on Theorem A.1 and the proof of Theorem 1), we can show that for x GLYPH&lt;6&gt; P p U T p S q :

<!-- formula-not-decoded -->

where r x t are the low-dimensional actions chosen in Step 5 of our modified OPOK algorithm, such that p t GLYPH&lt;16&gt; p U r x t for the prices output by this method. To conclude the proof, we recall that for the OPOK-selected p t :

<!-- formula-not-decoded -->

## A.5 Proof of Lemma 2

Lemma 2. Suppose that for t GLYPH&lt;16&gt; 1 , . . . , T : glyph[epsilon1] t GLYPH&lt;16&gt; 0 and V t ¡ 0 . If each p t is independently uniformly distributed within some (uncentered) Euclidean ball of strictly positive radius, then span p q 1 , . . . , q d q GLYPH&lt;16&gt; span p U q almost surely.

Proof. In Lemma 2, we suppose that each p t GLYPH&lt;16&gt; r p t GLYPH&lt;0&gt; ζ t , where each ζ t is uniformly drawn from a centered Euclidean ball of nonzero radius in R N and z t , V t , r p t are fixed independently of the randomness in ζ t . Note that each q t GLYPH&lt;16&gt; Us t where s t GLYPH&lt;16&gt; z t GLYPH&lt;1&gt; V t U T p t P R d . Thus, span p q 1 , . . . , q d q GLYPH&lt;132&gt; span p U q and the two spans must be equal if s 1 , . . . , s d are linearly independent.

To show linear independence holds almost surely, we proceed inductively by proving Pr p s t P span p s 1 , . . . , s t GLYPH&lt;1&gt; 1 qq GLYPH&lt;16&gt; 0 for any 1 GLYPH&lt;160&gt; t / d . Wefirst note that s t GLYPH&lt;16&gt; z t GLYPH&lt;1&gt; V t U T r p t GLYPH&lt;1&gt; V t U T ζ t . Since V t ¡ 0 is invertible and U is orthogonal, V t U T ζ t is uniformly distributed over a nondegenerate ellipsoid E GLYPH&lt;128&gt; R d with nonzero variance under any projection in R d . Since this includes directions orthogonal to the p t GLYPH&lt;1&gt; 1 q -dimensional subspace spanned by s 1 GLYPH&lt;0&gt; V 1 U T r p 1 GLYPH&lt;1&gt; z 1 , . . . , s t GLYPH&lt;1&gt; 1 GLYPH&lt;0&gt; V t GLYPH&lt;1&gt; 1 U T r p t GLYPH&lt;1&gt; 1 GLYPH&lt;1&gt; z t GLYPH&lt;1&gt; 1 , this subspace has measure zero under the uniform distribution over E (for t / d ).

Theorem A.3 (Yu et al., 2015) . Let σ 1 ¡ GLYPH&lt;4&gt; GLYPH&lt;4&gt; GLYPH&lt;4&gt; ¡ σ d ¡ 0 denote the nonzero singular values of rank d matrix Q P R N GLYPH&lt;2&gt; d , whose left singular vectors are represented as columns in matrix U P R N GLYPH&lt;2&gt; d (such that Q has SVD: UΣV T ). If p U P R N GLYPH&lt;2&gt; d similarly contains the left singular vectors of some other N GLYPH&lt;2&gt; d matrix p Q , then there exists orthogonal matrix p O P R d GLYPH&lt;2&gt; d such that

<!-- formula-not-decoded -->

## A.6 Proof of Theorem 3

Theorem 3. For unknown U , let p 1 , . . . , p T be the prices selected by the OPOL algorithm with η, δ, α set as in Theorem 1. Suppose glyph[epsilon1] t follows a sub-Gaussian p σ 2 q distribution and has statistically independent dimensions for each t . If (A1)-(A5) hold, then there exists C ¡ 0 such that for any p P S :

<!-- formula-not-decoded -->

Here, Q GLYPH&lt;16&gt; max ! 1 , σ 2 GLYPH&lt;1&gt; 2 σ 1 GLYPH&lt;0&gt; 1 σ 2 d GLYPH&lt;9&gt;) with σ 1 (and σ d ) defined as the largest (and smallest) nonzero singular values of the underlying rank d matrix s Q GLYPH&lt;6&gt; defined in (6).

Proof. For notational convenience, suppose that T is divisible by d , T 3 { 4 ¥ d ¥ 3 , and the noisevariation parameter σ ¥ 1 throughout our proof. Throughout, the unknown U is orthogonal and rank d , and we let p GLYPH&lt;6&gt; GLYPH&lt;16&gt; argmin p P S E GLYPH&lt;17&gt; GLYPH&lt;176&gt; T t GLYPH&lt;16&gt; 1 R t p p q GLYPH&lt;25&gt; denote the optimal product pricing.

Recall from the proof of Theorem 2 that under our low-rank demand model, we can redefine p GLYPH&lt;6&gt; -UU T p GLYPH&lt;6&gt; P S and still ensure p GLYPH&lt;6&gt; GLYPH&lt;16&gt; argmin p P S E GLYPH&lt;17&gt; GLYPH&lt;176&gt; T t GLYPH&lt;16&gt; 1 R t p p q GLYPH&lt;25&gt; . Thus, we suppose without loss of generality that the optimal prices can be expressed as p GLYPH&lt;6&gt; GLYPH&lt;16&gt; Ux GLYPH&lt;6&gt; for some corresponding low-dimensional action x GLYPH&lt;6&gt; P U T p S q .

For additional clarity, we use p U t to denote the current N GLYPH&lt;2&gt; d estimate of the underlying product features obtained in Step 10 of our OPOL algorithm at round t . Note that the p U t are random variables which are determined by both the noise in the observed demands and the randomness employed within our pricing algorithm. Letting p t GLYPH&lt;16&gt; p U t x t denote the prices chosen by the OPOL algorithm in each round (and x t P p U T t p S q the corresponding low-dimensional actions), we have:

<!-- formula-not-decoded -->

The proof of Theorem 1 ensures both | f t, U | and | f t, p U t | (for any orthogonal p U t ) are bounded by rb p 1 GLYPH&lt;0&gt; r q over all x P U T p S q , so we can trivially bound the first summand in (8):

<!-- formula-not-decoded -->

To bound the second summand in (8), we first point out that U T p S q GLYPH&lt;16&gt; p U T t p S q by Lemma A.2 (since all p U t are restricted to be orthogonal). Thus, Algorithm 4 is essentially running the classic gradientfree bandit method of [Flaxman et al., 2005] to optimize the functions f t, p U t over the low-dimensional

action-space U T p S q , and the second term is exactly the regret of this method stated in Theorem 1:

<!-- formula-not-decoded -->

Finally, we complete the proof by bounding the third summand in (8). Defining O GLYPH&lt;128&gt; R d GLYPH&lt;2&gt; d as the set of orthogonal d GLYPH&lt;2&gt; d matrices, we have:

<!-- formula-not-decoded -->

where now choose p O P O as the orthogonal matrix such that E || p U t p O GLYPH&lt;1&gt; U || F satisfies the bound of Lemma A.3 for the t ¥ T 3 { 4 fixed above. Defining ∆ GLYPH&lt;16&gt; Ux GLYPH&lt;6&gt; GLYPH&lt;1&gt; p U t p Ox GLYPH&lt;6&gt; P R d , we plug in the definition of f t, p U from (7) and simplify to obtain the following bound:

<!-- formula-not-decoded -->

Combining our bounds for each of the three summands in (8) yields the following upper bound for the left-hand side, from which the inequality presented in Theorem 3 can be derived:

<!-- formula-not-decoded -->

Lemma A.3. For the p U produced in Step 10 of the OPOL algorithm after T rounds and any feasible low-dimensional action x P p U T p S q , there exists orthogonal d GLYPH&lt;2&gt; d matrix p O and universal constant C such that:

<!-- formula-not-decoded -->

where σ 1 and σ d denote the largest and smallest singular values of the underlying matrix s Q GLYPH&lt;6&gt; defined in (6).

Proof. Our proof relies on standard random matrix concentration results presented in Lemma A.4 and the variant of the Davis-Kahan theory proposed by Yu et al. [2015], which is restated here as Theorem A.3.

Lemma A.4 (variant of Lemma 4.2 in Rigollet [2015]) . Let E be a N GLYPH&lt;2&gt; d matrix (with N ¥ d ) of i.i.d. entries drawn from a sub-Gaussian p σ 2 q distribution. Then, with probability 1 GLYPH&lt;1&gt; δ :

<!-- formula-not-decoded -->

Recall that random variable X follows sub-Gaussian( σ 2 ) distribution if E r X s GLYPH&lt;16&gt; 0 and Pr p| X | ¡ x q / 2 exp pGLYPH&lt;1&gt; x 2 2 σ 2 q for all x ¡ 0 , and random vector w GLYPH&lt;18&gt; sub-Gaussian( σ 2 ) if E r w s GLYPH&lt;16&gt; 0 and u T w is a sub-Gaussian( σ 2 ) random variable for any unit vector u . Since the components of glyph[epsilon1] t are presumed statistically i.i.d., each value in s E GLYPH&lt;16&gt; p Q GLYPH&lt;1&gt; s Q GLYPH&lt;6&gt; must be the mean of T { d subGaussian p σ 2 { N q samples as a result of the averaging performed in Step 9 of our OPOL algorithm. Thus, the entries of s E are distributed as sub-Gaussian GLYPH&lt;1&gt; σ 2 d NT GLYPH&lt;9&gt; . Lemma A.4 implies:

<!-- formula-not-decoded -->

When T ¥ d, σ ¥ 1 , both E || s E || op and E || s E || 2 op are upper-bounded by 24 σ 2 a 6 πd { T . Combining Theorem A.3 with these concentration bounds implies that there exists d GLYPH&lt;2&gt; d orthogonal matrix p O such that:

<!-- formula-not-decoded -->

## B Pricing against an Imprecise Adversary

Theorem B.4 below illustrates a basic scenario under which an explicit high-probability bound for the constant Q from Theorem 3 can be obtained. Throughout our subsequent discussion, the largest and smallest nonzero singular values of a rankd matrix A will be denoted as σ 1 p A q and σ d p A q , respectively. We now assume that the adversary can only coarsely control the underlying baseline demand parameters z t in (2). More specifically, we suppose that in each round: z t GLYPH&lt;16&gt; z 1 t GLYPH&lt;0&gt; γ t , where only z 1 t (and V t ) may be adversarially selected and the γ t are purely stochastic terms outside of the adversary's control. In this scenario, we presume a random d GLYPH&lt;2&gt; d noise matrix Γ is drawn before the initial round such that:

(A10) Each entry Γ i,j is independently sampled with mean zero and magnitude bounded almost surely by b { 2 (i.e. E r Γ i,j s GLYPH&lt;16&gt; 0 , | Γ i,j | / b { 2 for all i, j ).

Recall that the constant b ¡ 0 upper bounds the magnitude of each z t as specified in (A1). Once the values of Γ have been sampled, we suppose that in round t : γ t GLYPH&lt;16&gt; Γ GLYPH&lt;6&gt; ,j is simply taken to be the j th column of this matrix with j GLYPH&lt;16&gt; 1 GLYPH&lt;0&gt; p t GLYPH&lt;1&gt; 1 q mod d (traversing the columns of Γ in order). Since boundedness of the values in Γ implies these entries follow a sub-Gaussian p b 2 { 4 q distribution, the following result applies:

Lemma B.5 (variant of Theorem 1.2 in Rudelson and Vershynin [2008]) . With probability at least 1 GLYPH&lt;1&gt; C b glyph[epsilon1] GLYPH&lt;1&gt; c b d : ?

<!-- formula-not-decoded -->

where C b ¡ 0 and c b P p 0 , 1 q are constants that depend (polynomially) only on b .

In selecting z 1 t , V t , we assume the imprecise adversary is additionally restricted to ensure:

<!-- formula-not-decoded -->

where constants c b , C b are given by Lemma B.5 (see Rudelson and Vershynin [2008] for details), and r ¥ 1 is still used to denote the radius of the set of feasible prices S . Note that these additional assumptions do not conflict with condition (A1) required in Theorem 4, since (A10), (A11) together ensure that || z t || 2 / b for z t GLYPH&lt;16&gt; z 1 t GLYPH&lt;0&gt; γ t . With these assumptions in place, we now provide an explicit bound for the constant Q defined in Theorem 3.

Theorem B.4. Under this setting of an imprecise adversary where conditions (A10) and (A11) are met, for any τ P p 1 2 C b sbd GLYPH&lt;0&gt; c b d , 1 q , Theorem 4 holds with:

<!-- formula-not-decoded -->

with probability ¥ 1 GLYPH&lt;1&gt; τ (over the initial random sampling of Γ ).

Proof. Recall that σ 1 (and σ d ) denote the largest (and smallest) nonzero singular values of the underlying rank d matrix s Q GLYPH&lt;6&gt; defined in (6). For suitable constants c 1 , c 2 : we show that σ 1 / c 1 and σ d ¥ c 2 with high probability, which then implies the upper bound: Q / max t 1 , σ c 2 p 2 c 1 GLYPH&lt;0&gt; 1 qu . We

2

first note that the orthogonality of U implies s Q GLYPH&lt;6&gt; GLYPH&lt;16&gt; U s S has the same nonzero singular values as the square matrix s S , whose j th column is given by:

<!-- formula-not-decoded -->

As s S has d columns, we have:

<!-- formula-not-decoded -->

where the latter inequality derives from the fact that (A5) and orthogonality of U imply:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Via similar reasoning, we also obtain the bound:

<!-- formula-not-decoded -->

Subsequently, we invoke Lemma B.5, which implies that with probability 1 GLYPH&lt;1&gt; τ :

<!-- formula-not-decoded -->

Combining (10) and (11), we obtain a high probability lower bound for σ d via the additive Weyl inequality (cf. Theorem 3.3.16 in Horn and Johnson [1991]):

<!-- formula-not-decoded -->

The proof is completed by defining c 1 GLYPH&lt;16&gt; b p 1 GLYPH&lt;0&gt; s q ? d 2 , c 2 GLYPH&lt;16&gt; τ GLYPH&lt;1&gt; c b d C b ? d GLYPH&lt;1&gt; sb ? d 2 , and subsequent simplification of the resulting bound using the fact that d ¥ 1 and s GLYPH&lt;160&gt; 1 .

## C Additional Experimental Results

## C.1 Misspecified Demand Models

Beyond evaluating our pricing strategies in settings where underlying demand curves adhere to our low-rank model in (2), we now consider different environments where our assumptions are purposefully violated, in order to investigate robustness and how well each approach generalizes to other types of demand behavior. As our interest lies in high-dimensional pricing applications, the number of products is fixed to N GLYPH&lt;16&gt; 100 throughout this section. Once again, p t and q t are presumed to represent suitably rescaled prices/aggregate-demands, such that the set of feasible prices S can always be fixed as a centered sphere of radius r GLYPH&lt;16&gt; 20 . Although none of the demand models considered here possesses explicit low-rank structure, we nevertheless apply our OPOL pricing algorithm with various choices of the rank parameter 1 / d / N GLYPH&lt;16&gt; 100 .

Linear full-rank model. We first study a scenario where underlying demands follow the basic linear relationship described in (1): q t GLYPH&lt;16&gt; c t GLYPH&lt;1&gt; B t p t GLYPH&lt;0&gt; glyph[epsilon1] t . Under this setting, the entries of c t , B t , and glyph[epsilon1] t are independently drawn from N p 100 , 20 q , N p 0 , 2 q , and N p 0 , 10 q distributions, respectively. Before demands are generated, B t is first projected onto the set of strongly positive-definite matrices t B : B T GLYPH&lt;0&gt; B ' λ I u with λ GLYPH&lt;16&gt; 10 as done in § 5. We consider both the stationary case where c t , B t are fixed over time as well as the case of demand shocks, in which these underlying parameters are re-sampled from their generating distributions at times T { 3 and 2 T { 3 . Note that the demands in this setting do not possess any explicit low-rank structure, nor are they governed by low-dimensional featurizations of the products.

Figure 2 depicts the performance of our pricing algorithms in this linear full-rank setting, showing the average cumulative regret (over 10 repetitions with standard-deviations shaded). Once again, the performance of the GDG approach and our OPOL algorithm with d GLYPH&lt;16&gt; N are essentially identical. In this setting, the standard bandit methods slightly outperform the Explo re it baseline, but they do not exhibit strong performance when optimizing over a 100-dimensional action space. Despite the lack of explicit low-rank structure in the underlying demand model, the OPOL algorithm produces greater revenues than the GDG and Explo re it baselines for all settings of d P r 10 , 90 s (but does fare worse than GDG if d ! 10 is chosen too small). In particular, when operating with relatively low values of d , the OPOL method very significantly outperforms the other pricing strategies. Similar phenomena in bandit algorithms over projected low-dimensional action subspaces have been documented by Wang et al. [2013], Li et al. [2016], Yu et al. [2017].

Log-linear model. While the linear demand model studied in this paper is one of the most popular methods for pricing products with varying elasticities, demands for products with constant elasticity are often better fit via a log-linear function of the prices [Maurice, 2010]. We also evaluate the performance of our bandit methods in such a setting, where demands are determined according to the following log-linear model:

<!-- formula-not-decoded -->

In our experiment under this setting, the entries of r c t , r B t , r glyph[epsilon1] t are independently drawn from N p 5 , 1 q , N p 0 , 0 . 1 q , and N p 0 , 1 q distributions, respectively. Before demands are generated, r B t is first projected onto the set of strongly positive-definite matrices t B : B T GLYPH&lt;0&gt; B ' λ I u with λ GLYPH&lt;16&gt; 0 . 1 . Again, two scenarios are considered: the stationary case where r c t , r B t are fixed over time, and the case of demand shocks, in which these underlying parameters are re-sampled from their generating distributions at times T { 3 and 2 T { 3 . Note that this log-linear model also does not possess any explicit low-rank properties.

Figure C.1 demonstrates that the same conclusions about our algorithm's behavior in the case of fullrank linear demands also hold for this log-linear setting. Even though it is now quite misspecified, the OPOLalgorithm with a small value of d performs remarkably well. Furthermore, the decreasing regret in Figure C.1B illustrates how bandit pricing methods can rapidly adapt to a changing marketplace, regardless whether the underlying demands are of varying or constant elasticities.

<!-- image -->

T

T

Figure C.1: Average cumulative regret (over 10 repetitions with standard-deviations shaded) of various pricing strategies (for N GLYPH&lt;16&gt; 100 ) when the underlying demand model is log-linear and: (A) stationary over time, (B) altered by structural shocks at T { 3 and 2 T { 3 .

## C.2 Further Details about Experiments

Our simulations always set the first prices used to initialize each method, p 0 , at the center of S . For each experiment in our paper, the bandit algorithm hyperparameters η, δ, α are set as specified in Theorem A.2, but without knowledge of the underlying demand model (as would need to be done in practicalapplications). Because the Lipschitz constant L and bound B are unknown in practice, these are crudely estimated prior to the initial round of our bandit pricing strategy from the observed (historical) revenues at a random collection of 100 minorly-varying prices. To compute regret, we identify the optimal fixed price with knowledge of the underlying demand curves at each time, performing the fixed-price optimization via Sequential Least Squares Programming [Kraft, 1988] which converges to the global optimum in our convex settings. In the Explo re it approach, transitioning from exploitation to exploration at time T 3 { 4 empirically outperformed the other choices we considered ( T 1 { 2 , T 2 { 3 , T { 10 , T { 3 ). Note that no matter how many experiments we run, the sensitive nature of pricing necessitates provable guarantees, which is a major strength of the adversarial regret bounds presented in this paper.

## D Notation Glossary

N ¡ 0 Number of products to price (assumed to be large)

d ¡ 0 Dimensionality of the product features (where d ! N )

t P t 1 , . . . , T u Index of each time period (i.e. round ) over which prices are fixed and demands aggregated

C ¡ 0 A universal constant that is problem-independent and does not depend on values like T, d, r

p t P R N Vector of prices for each product in period t (rescaled rather than absolute prices)

q t P R N Vector of demands for each product in period t (rescaled rather than absolute demands)

R t : R N GLYPH&lt;209&gt; R Negative total revenue produced by product pricing in period t (convex function)

S GLYPH&lt;128&gt; R N Convex set of feasible prices (taken to be ball of radius r throughout § 4.2)

glyph[epsilon1] t P R N Random noise in observed demands of period t (mean-zero with finite variance)

glyph[epsilon1] Represents the full set of random demand effects t glyph[epsilon1] 1 , . . . , glyph[epsilon1] T u

ξ t P R d Random noise variables drawn within each round of our bandit algorithms

ξ Represents the full set of random noise variables employed in our algorithms t ξ 1 , . . . , ξ T u

c t P R N Vector of baseline aggregate demands for each product in period t

B t P R N GLYPH&lt;2&gt; N Asymmetric positive-definite matrix of demand cross-elasticities in period t

U P R N GLYPH&lt;2&gt; d Matrix where i th row contains featurization of product i (presumed orthogonal in § 4.2)

p U P R N GLYPH&lt;2&gt; d Matrix whose column-span is used to estimate the column-span of U

z t P R d Vector which determines how product features affect the baseline demands in period t

V t P R d GLYPH&lt;2&gt; d Asymmetric positive-definite matrix that defines changing demand cross-elasticies in period t

|| x || 2 Euclidean norm of vector x

|| A || op Spectral norm of matrix A (magnitude of the largest singular value)

||

A

||

F

Frobenius norm of matrix

A

Unif p S q Uniform distribution over set S

p GLYPH&lt;6&gt; P R N Single best vector of prices chosen in hindsight: p GLYPH&lt;6&gt; GLYPH&lt;16&gt; argmin p P S E T , t GLYPH&lt;16&gt; 1 R t p p q

$$f t : R d GLYPH<209> R Function such that f t p x q GLYPH<16> E glyph[epsilon1] r R t p p qs for x GLYPH<16> U T p$$

$$f t, p U : R d GLYPH<209> R Function such that f t, p U p x q GLYPH<16> E glyph[epsilon1] r R t p p qs for x GLYPH<16> p U T p$$

η, δ, α ¡ 0 User specified hyperparameters of our bandit pricing algorithms

σ 2 ¡ 0 Sub-Gaussian parameter that specifies magnitude of noise effects in the observed demands

U T p S q d -dimensional actions that correspond to feasible prices: x P R d : x GLYPH&lt;16&gt; U T p for some p P S (

r GLYPH&lt;210&gt; , r GLYPH&lt;211&gt; ¡ 0 Radius of Euclidean balls containing/contained-within U T p S q , with r GLYPH&lt;210&gt; ¥ r GLYPH&lt;211&gt;

B ¡ 0 Upper bounds the magnitude of E r R t p p qs over all p P S , t GLYPH&lt;16&gt; 1 , . . . , T

L ¡ 0 Lipschitz constant of each f t p x q over all x P U T p S q , t GLYPH&lt;16&gt; 1 , . . . , T

b ¡ 0 Upper bounds the magnitude of z t , V t for t GLYPH&lt;16&gt; 1 , . . . , T ( || z t || 2 ⁄ b and || V t || op ⁄ b )

r ¥ 1 Radius of Euclidean ball adopted as the feasible set of (rescaled) prices throughout § 4.2

## Additional References for the Supplementary Material

- A. D. Flaxman, A. T. Kalai, and H. B. McMahan. Online convex optimization in the bandit setting: Gradient descent without a gradient. Proceedings of the 16th Annual ACM-SIAM Symposium on Discrete Algorithms , 2005.
- R. Horn and C. R. Johnson. Topics in Matrix Analysis . Cambridge Univ. Press, 1991.
- D. Kraft. A software package for sequential quadratic programming. Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace Center - Institute for Flight Mechanics, Koln, Germany , 1988.
- C. Li, K. Kandasamy, B. Poczos, and J. Schneider. High dimensional bayesian optimization via restricted projection pursuit models. Artificial Intelligence and Statistics , 2016.
- T. Maurice. Managerial Economics . McGraw-Hill Education, 2010.
- P. Rigollet. High dimensional statistics, 2015. MIT Opencourseware: ocw.mit.edu/courses/ mathematics/18-s997-high-dimensional-statistics-spring-2015/lecture-notes/ .
- M. Rudelson and R. Vershynin. The Littlewood-Offord problem and invertibility of random matrices. Advances in Mathematics , 218:600-33, 2008.
- Z. Wang, M. Zoghi, F. Hutter, D. Matheson, and N. de Freitas. Bayesian optimization in high dimensions via random embeddings. International Joint Conference on Artificial Intelligence , 2013.
- X. Yu, M. R. Lyu, and I. King. CBRAP: Contextual bandits with random projection. Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence , 2017.
- Y. Yu, T. Wang, and R. Samworth. A useful variant of the Davis-Kahan theorem for statisticians. Biometrika , 102:315-323, 2015.