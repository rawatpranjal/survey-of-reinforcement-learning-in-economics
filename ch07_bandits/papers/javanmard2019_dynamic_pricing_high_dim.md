## Dynamic Pricing in High-dimensions

Adel Javanmard ∗ Hamid Nazerzadeh ∗

January 3, 2018

## Abstract

We study the pricing problem faced by a firm that sells a large number of products, described via a wide range of features, to customers that arrive over time. Customers independently make purchasing decisions according to a general choice model that includes products features and customers' characteristics, encoded as d -dimensional numerical vectors, as well as the price offered. The parameters of the choice model are a priori unknown to the firm, but can be learned as the (binary-valued) sales data accrues over time. The firm's objective is to minimize the regret, i.e., the expected revenue loss against a clairvoyant policy that knows the parameters of the choice model in advance, and always offers the revenue-maximizing price. This setting is motivated in part by the prevalence of online marketplaces that allow for real-time pricing.

We assume a structured choice model, parameters of which depend on s 0 out of the d product features. We propose a dynamic policy, called Regularized Maximum Likelihood Pricing (RMLP) that leverages the (sparsity) structure of the high-dimensional model and obtains a logarithmic regret in T . More specifically, the regret of our algorithm is of O ( s 0 log d · log T ). Furthermore, we show that no policy can obtain regret better than O ( s 0 (log d +log T )).

## 1 Introduction

A central challenge in revenue management is determining the optimal pricing policy when there is uncertainty about customers' willingness to pay. Due to its importance, this problem has been studied extensively [KL03, BZ09, BKS13, WDY14, BR12, KZ14, dBZ14, CLPL16]. Most of these models are built around the following classic setting: customers arrive over time; the seller posts a price for each customer; if the customer's valuation is above the posted price, a sale occurs and the seller collects a revenue in the amount of the posted price; otherwise, no sale occurs and no revenue is generated. Based on this and the previous feedbacks, the seller updates the posted price. Therefore, the seller is involved in the realm of exploration-exploitation as he needs to choose between learning about the valuations and exploiting what has been learned so far to collect revenue.

In this work, we consider a setting with a large number of products which are defined via a wide range of features. The valuations are given by v ( θ, x ) with x being the (observable) feature vectors of products and θ 0 representing the customer's characteristics (true parameters of the choice model, which is initially unknown to the seller, cf. [ARS14, CLPL16].) . An important special case of this setting is the linear model in which

<!-- formula-not-decoded -->

∗ Department of Data Sciences and Operations, Marshall School of Business, University of Southern California Email: { ajavanma,nazerzad } @usc.edu

where z captures the idiosyncratic noise in valuations and α 0 is an unknown intercept.

Our setting is motivated in part by applications in online marketplaces. For instance, a company such as Airbnb recommends prices to hosts based on many features including the space (number of rooms, beds, bathrooms, etc.), amenities (AC, WiFi, washer, parking, etc.), the location (accessibility to public transportation, walk score of the neighborhood, etc.), house rules (pet-friendly, non-smoking, etc.), as well as the prediction of the demand which itself depends on many factors including the date, events in the area, availability and prices of near-by hotels, etc. [Air15]. Therefore, the vector describing each property can have hundreds of features. Another important application comes from online advertising. Online publishers set the (reserve) price of ads based on many features including user's demographic, browsing history, the context of the webpage, the size and location of the ad on the page, etc.

In this work, we propose Regularized Maximum Likelihood Pricing (RMLP) policy for dynamic pricing in high-dimensional environments. As suggested by its name, the policy uses maximum likelihood method to estimate the true parameters of the choice model. In addition, using an ( /lscript 1 -norm) regularizer, our policy exploits the structure of the optimal solution; namely, the performance of the RMLP policy significantly improves if the valuations are essentially determined by a small subset of features. More formally, the difference between the revenue obtained by our policy and the benchmark policy that knows in advance the true parameters of the choice model, µ 0 = ( θ 0 , α 0 ), is bounded by O ( s 0 log d · log T ) , where T , d , and s 0 respectively denote the length of the horizon, number of the features, and sparsity (i.e., number of non-zero elements of µ 0 ). We show that our results are tight up to a logarithmic factor. Namely, no policy can obtain regret better than O ( s 0 (log d +log T ) ) .

We point out that our results can be applied to applications where the features' dimensions are larger than the time horizon of interest. A powerful pricing policy for these applications should obtain regret that scales gracefully with the dimension. Note that in general, little can be learned about the model parameters µ 0 if T &lt; d , because the number of degrees of freedom d exceeds the number of observations T , and therefore, any estimator can be arbitrary erroneous. However, when there is prior knowledge about the structure of unknown parameter µ 0 , (e.g., sparsity), then accurate estimations are attainable even when T &lt; d .

## 1.1 Organization

The rest of the paper is organized as follows: In the remaining part of the introduction, we discuss how our work is positioned with respect to the literature and highlight our contributions. In Section 2, we formally present our model and discuss the technical assumptions and the benchmark policy. The RMLP policy is presented in Section 3, followed by its analysis in Section 4. We provide in Section 5, a bound on the performance of any dynamic pricing policy that does not know the choice model in advance. In Section 6, we generalize the RMLP policy to non-linear valuations functions. The proofs are relegated to the appendix.

## 1.2 Related Work

Our work contributes to literature on dynamic pricing as well as high dimensional statistics. In the following, we briefly overview the work closest to ours in these contexts.

Dynamic Pricing and Learning. The literature on dynamic pricing and learning has been growing over the past few years, motivated in part by the advances in big data technology

that allow firms to easily collect and utilize information. We briefly discuss some of the recent lines of research in this literature. We refer to [dB15] for an excellent survey on this topic.

- Parametric Approach. A natural approach to capture uncertainty about the customers' valuations is to model the uncertainty using a small number of parameters, and then estimate those parameters using classical statistical methods such as maximum likelihood [BR12, dBZ13, dBZ14] or least square estimation [GZ13, Kes14, BB16]. Our work is similar to this line of work, in that we assume a parametric model for customer's valuations and apply the maximum likelihood method using the randomness of the idiosyncratic noise in valuations. However, the parameter vector θ is high-dimensional, whose dimension d (that can even exceed the time horizon of interest T ). We use regularized maximum-likelihood in order to promote sparsity structure in the estimated parameter. Further, our pricing policy has an episodic theme which makes the posted prices p t in each episode independent of the idiosyncratic noise in valuations, z t , in that episode. This is in contrast to other policies based on maximum-likelihood, such as MLEGREEDY [BR12], or greedy iterative least square (GILS) [Kes14, dBZ14, QB16] that use the entire history of observations to update the estimate for the model parameters at each step.
- Bayesian Approach. One of the earliest work on Bayesian parametric approach in this context is by [Rot74] who consider a Bayesian framework where the firm can choose from two prices with unknown demand and show that (myopic) Bayesian policies may lead to 'incomplete learning.' However, carefully designed variations of the myopic policies can (optimally) learn the optimal price [HKZ12]; see also [KR99, AC09, FVR10, KZ14].
- Non-Parametric models. An early work in non-parametric setting is by [KL03]. They model the dynamic pricing problem as a multi-armed bandit (MAB) where each arm corresponds to a (discretized) posted price. They propose an O ( √ T )-algorithm where T is the length of the horizon. Similar results have been obtained in more general settings [BKS13, AD14] including setting with inventory constraints [BZ09, BDKS12, WDY14].
- Feature-based Models. Recent papers on dynamic pricing consider models with features/covariates. [ARS14], in a model similar to ours, present an algorithm that obtains regret O ( T 2 / 3 ); they also study dynamic incentive compatibility in repeated auctions. Another closely related work to ours is by [CLPL16]. Their model differs from ours in two main aspects: i ) their model is deterministic (no idiosyncratic noise) ii ) the arrivals (of features vectors) is modeled as adversarial. They propose a clever binary-search approach using the Ellipsoid method which obtains regret of O ( d 2 log( T/d )). [QB16] study a model where the seller can observe the demand itself, not a binary signal as in our setting. They show that a myopic policy based on least-square estimations can obtain a logarithmic regret. To the extent of our knowledge, ours is the first work that highlights the role of structure/sparsity in dynamic pricing.

[BB16] study a multi-armed bandit setting, with discrete arms, and high-dimensional covariates, generalizing results of [GZ13]. [BB16] present an algorithm, using a LASSO estimator, that obtains regret O ( K (log T +log d ) 2 ) where K denotes the number of arms. In contrast, our setting can be interpreted as a multi-armed bandit with continuous arms in a high dimensional space.

High Dimensional Statistics. There has been a great deal of work on regularized estimator under the high-dimensional scaling; see e.g. [VdG08]. Closer to the spirit of our work is the problem of 1-bit compressed sensing [PV13, BJ15]. In this problem, linear measurements are observed for an unknown parameter of interest but only the sign of these measurements are observed. Note that in our problem, seller is involved in both the learning task and also the policy design. Specifically, he should decide on the prices, which directly affect collected revenue and also indirectly influence the difficulty of the learning task. The market values are then compared with the posted prices, in contrast to 1-bit compressed sensing where the measurements are compared with zero (sign information). In addition, the pricing problem has an online nature while the 1-bit compressed sensing is mostly studied for offline setting. Finally, note that prices are set based on customer's purchase behavior, and hence introduce dependency among the collected information about the model parameters.

## 1.3 Notations

For a vector v , supp( v ) represents the positions of nonzero entries of v . Further, for a vector v and a subset J , v J is the restriction of v to indices in J . We write ‖ v ‖ p for the standard /lscript p norm of a vector v , i.e., ‖ v ‖ p = ( ∑ i | v i | p ) 1 /p and ‖ v ‖ 0 for the umber of nonzero entries of v . If the subscript p is omitted, it should be deemed as /lscript 2 norm. For two vectors a, b ∈ R d , the notation a · b = ∑ d i =1 a i b i represents the standard inner product. For two functions f ( n ) and g ( n ), the notation f ( n ) = O ( g ( n )) means that f is bounded above by g asymptotically, namely, f ( n ) ≤ Cg ( n ) for some fixed positive constant C &gt; 0. Throughout, φ ( x ) = e -x 2 / 2 / √ 2 π is the Gaussian density and Φ( x ) ≡ ∫ x -∞ φ ( u )d u is the Gaussian distribution.

## 2 Choice model

We consider a seller, who has a product for sale in each period t = 1 , 2 , · · · , T , where T denotes the length of the horizon and may be unknown the to the seller. Each product is represented by an observable vector of features (covariates) x t ∈ X ⊆ R d . Products may vary across periods and we assume that feature vectors x t are sampled independently from a fixed, but a priori unknown , distribution P X , supported on a bounded set X .

The product at time t has a market value v t = v ( x t ), which is not observed by the seller and function v is (a priori) unknown. At each period t , the seller posts a price p t . If p t ≤ v t , a sale occurs, and the seller collects revenue p t . If the price is set higher than the market value, p t &gt; v t , no sale occurs and no revenue is obtained. The goal of the seller is to design a pricing policy that maximizes the collected revenue.

We first assume that the market value of a product is a linear function of its covariates, namely

<!-- formula-not-decoded -->

where a · b denotes the inner product of vectors a and b . Here, { z t } t ≥ 1 are idiosyncratic shocks, referred to as noise, which are drawn independently and identically from a distribution with mean zero and cumulative function F , with density f ( x ) = F ′ ( x ), cf. [KZ14].The noise can account for the features that are not measured. We generalize our model to non-linear valuation functions in Section 6.

Parameter θ 0 is a prior unknown to seller. Therefore, the seller is involved in the realm of exploration-exploitation as he needs to choose between learning θ 0 and exploiting what has been learned so far to collect revenue.

Henceforth, we let µ 0 = ( θ 0 , α 0 ) ∈ R d +1 denote the true model parameters and also define the augmented feature vectors ˜ x t = ( x t , 1).

Let y t be the response variable that indicates whether a sale has occurred at period t :

<!-- formula-not-decoded -->

Note that the above model can be represented as the following probabilistic model:

/negationslash

## 2.1 Technical assumptions

<!-- formula-not-decoded -->

Our proposed algorithm exploits the structure (sparsity) of the feature space to improve its performance. To this aim, let s 0 denote the number of nonzero coordinates of θ 0 , i.e., s 0 = ‖ µ 0 ‖ 0 = ∑ d j =1 I ( µ 0 j = 0). We remark that s 0 is a priori unknown to the seller.

To simplify the presentation, we assume that ‖ x t ‖ ∞ ≤ 1, for all x t ∈ X , and ‖ µ 0 ‖ 1 ≤ W for a known constant W , where for a vector u = ( u 1 , . . . , u d ), ‖ u ‖ ∞ = max i ∈ [ d ] | u i | denotes the maximum absolute value of its entries and ‖ u ‖ 1 = ∑ d i =1 | u i | . We denote by Ω the set of feasible parameters, i.e.,

We also make the following assumption on the distribution of noise F .

<!-- formula-not-decoded -->

Assumption 2.1. The function F ( v ) is strictly increasing. Further, F ( v ) and 1 -F ( v ) are logconcave in v .

Log-concavity is a widely-used assumption in the economics literature [BB05]. Note that if the density f is symmetric and the distribution F is log-concave, then 1 -F is also log-concave. Assumption 2.1 is satisfied by several common probability distributions including normal, uniform, Laplace, exponential, and logistic. Note that the cumulative distribution function of all log-concave densities is also log-concave [BV04].

Our second assumption is on the product feature vectors.

Assumption 2.2. Product feature vectors are generated independently from a probability distribution P X with a bounded support X ∈ R d . We further assume that E ( x t ) is normalized to zero 1 and denoting by Σ = E ( x t x T t ) the covariance matrix of { x t } , we assume that Σ is a positive definite matrix. Namely, all of its singular values are bounded from below by a constant C min &gt; 0. We also denote the maximum eigenvalue of Σ by C max .

The above assumption holds for many common probability distributions, such as uniform, truncated normal, and in general truncated version of many more distributions. Generally, if P X is bounded below from zero on an open set around the origin, then it has a positive definite covariance matrix. Let us stress that we know neither the distribution P X , nor its covariance Σ.

/negationslash

1 This normalization does not imply any restriction because if E ( x t ) = 0, then it can be absorbed in the intercept term α 0 . More precisely, we consider model with intercept parameter ˜ α 0 = α 0 + θ 0 · E ( x t ).

## 2.2 Clairvoyant policy and performance metric

We evaluate the performance of our algorithm using the common notion of regret: the expected revenue loss compared with the optimal pricing policy that knows µ 0 in advance (but not the realizations of { z t } t ≥ 1 ). Let us first characterize this benchmark policy.

Using Eq. (1), the expected revenue from a posted price p is equal to

<!-- formula-not-decoded -->

Therefore, using first order conditions, for the optimal posted price, denoted by p ∗ , we have

<!-- formula-not-decoded -->

To simplify the presentation, let p ∗ t = p ∗ (˜ x t ) denote the optimal price at time t .

We now define ϕ ( v ) ≡ v -1 -F ( v ) f ( v ) corresponding to the virtual valuation function commonly used in mechanism design [Mye81]. By Assumption 2.1, ϕ is injective and hence we can define function g as follows

<!-- formula-not-decoded -->

It is easy to verify that g is non-negative. Note that by Eq. (4), for the optimal price we have

<!-- formula-not-decoded -->

Therefore, by rearranging the terms for the optimal price at time t we have

<!-- formula-not-decoded -->

We can now formally define the regret of a policy. Let π be the seller's policy that sets price p t at period t , and p t can depend on the history of events up to time t . The worst-case regret is defined as:

<!-- formula-not-decoded -->

where the expectation is with respect to the distributions of idiosyncratic noise, z t , and P X , the distribution of feature vectors. Moreover, Q ( X ) represents the set of probability distributions supported on a bounded set X .

Our algorithm uses the sparsity structure of µ 0 and learns the model with order of magnitude less data compared to a structure-ignorant algorithm. In Section 4, we show that our pricing scheme achieves a regret bound of O ( s 0 log T (log d +log T ) ) .

## 3 A Regularized Maximum Likelihood Pricing (RMLP) Policy

In this section, we present our dynamic pricing policy. Our policy runs in an episodic fashion. Episodes are indexed by k and time periods are indexed by t . The length of episode k is denoted by τ k . Throughout episode k , we set the prices equal to p t = g (˜ x t · ̂ µ k ) where ̂ µ k denotes the estimate 6

Input: (at time 0 ) function g , regularizations λ k , W (bound on ‖ µ 0 ‖ 1 ),

Output:

Input: (arrives over time) covariate vectors { ˜ x t } t ∈ N

- 1: τ 1 ← 1, p 1 ← 0, ̂ µ 1 ← 0 2: for each episode k = 2 , 3 , . . . do 3: Set the length of k -th episode: τ k ← 2 k -1 .

prices { p t } t ∈ N

- 4: Update the model parameter estimate ̂ µ k using the regularized ML estimator obtained from observations in the previous episode:

with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 5: For each period t during the k -th episode, set

Algorithm 1: RMLP policy for dynamic pricing

<!-- formula-not-decoded -->

of µ 0 which is obtained from the observations { ( x t , y t , p t ) } in the previous episode. Note that by Eq. (5), p t is the optimal posted price if µ k was the true underlying parameter of the model.

̂ We estimate µ 0 using a regularized maximum-likelihood estimator; see Eq. (25) where the (normalized) negative log-likelihood function for µ is given by Eq. (26). We note that as a consequence of the log concavity assumption on F and 1 -F , the optimization problem (25) is a convex problem. There is a large toolkit of various optimization methods (e.g., alternating direction method of multipliers (ADMM), fast iterative shrinkage-thresholding algorithm (FISTA), accelerated projected gradient descent, among many others) that can be used to solve this optimization problem. There are also recent developments on distributed solvers for /lscript 1 regularized cost function [BPC + 11].

Observe that by design, prices posted in the k -th episode are independent from the market value noises in this period, i.e., { z t } τ k +1 -1 t = τ k . This allows us to estimate µ 0 for each episode separately; see Proposition 8.1 in Section 8.1. Comparing to policies that use the entire data sale history in making decisions, some remarks are in order:

- Perishability of data: In practical applications, the unknown demand parameters will change over time, raising the concern of perishability of data. Namely, collected data becomes obsolete after a while and cannot be relied on for estimating the model parameters [KZ16, Jav17]. Common practical policies to mitigate this problem (discussed in [KZ16]) include moving windows and decaying weights which use only recent data to learn the model parameters. In contrast, methods that use the entire historical data suffers from this problem.
- Simplicity and efficiency: In RMLP policy, estimates of the model parameters are updated only at the first period of each episode (log T updates). Further, at each update, the policy

uses only the historical data from the previous episode. These two ideas together, not only allow for a neat analysis of the statistical dependency among samples but also decrease the computational cost. Scalability of the pricing policy is indispensable in practical applications as the sales data is collected at an unprecedented rate.

- Effect on regret: By using half of the historical data at each update, our policy loses at most a factor 2 in the total regret. (This becomes clear shortly when we discuss the estimation error rate in terms of number of samples.)

The lengths of episodes in our algorithm increase geometrically ( τ k = 2 k -1 ), allowing for more accurate estimate of µ 0 as the episode index grows. The algorithm terminates at the end of the horizon (period T ), but note that it does not need to know the length of the horizon in advance.

Regularization parameter λ k constrains the /lscript 1 norm of the estimator ̂ µ k . Selecting the value of λ k is of crucial importance as it effects the estimator error. We set it as λ k = O ( √ (log d ) /τ k -1 ) . More precisely, define

<!-- formula-not-decoded -->

where the derivatives are w.r.t. x . By the log-concavity property of F and 1 -F , we have

Hence, u 2 W captures the steepness of log F .

<!-- formula-not-decoded -->

In order to minimize the regret, we run the RMLP policy with

<!-- formula-not-decoded -->

Note that exploration and exploitation tasks are mixed in our algorithm. In the beginning of each episode, we use what is learned from previous episode to improve the estimation of θ 0 and then we exploit this estimate throughout the current episode to incur little regret. Meanwhile, the observations gathered in the current episode are used to update our estimate of θ 0 for the next episode. We analyze the performance of RMLP in the next section.

## 4 Regret analysis

Although the description of RMLP is oblivious to sparsity s 0 , its performance depends on the structure of the optimal solution. The following theorem bounds the regret of our dynamics pricing policy.

Theorem 4.1 (Regret Upper Bound) . Suppose Assumptions 2.1 and 2.2 hold. Then, the regret of the RMLP policy is of O ( s 0 log d · log T ) .

Below we provide an outline for the proof of Theorem 4.1 and defer its complete proof to Section 8.1.

1. In RMLP, the updates in the model parameter estimation only occurs at the beginning of each episode, with using only the samples collected in the previous episode. Therefore, the prices posted in each episode are independent from the market value noises in that episode. This observation also verifies that L ( µ ) given by (26), is indeed the negative log-likelihood of the samples collected in k -th episode. Note that this independence is not a mere serendipity, rather it holds because of the specific design of RMLP policy. Using this property, we use tools from high-dimensional statistics to bound the estimation error. To bound the error term ‖ µ k -µ 0 ‖ 2 , we compare the function values L ( µ k ) and L ( µ 0 ). The main challenge here is that L ( µ ) is not strictly convex in µ . 2 Hence, there can be, in principle, parameter vectors µ 1 and µ 2 that are close to each other and nevertheless the values of function L at these points are far from each other.

To cope with this challenge, we show that a so-called restricted eigenvalue condition holds for the feature products. This notion implies that L ( µ ) is strictly convex on the set of sparse vectors. 3 Using the restricted eigenvalue condition, we show the following /lscript 2 error for the regularized log-likelihood estimate in the k -th episode, µ k , holds true

As expected, the estimate gets more accurate as the episode's length increases; see Section 8.1 for more details.

<!-- formula-not-decoded -->

2. For any p ≥ 0, denote by r t ( p ) = p (1 -F ( p -˜ x t · µ 0 )), the expected revenue under price p . We bound R t in terms of r t ( p ∗ t ) -r t ( p t ). Since p ∗ t ∈ arg max { r t ( p ) } , we have r ′ t ( p ∗ t ) = 0, and by Taylor expansion of r t around p ∗ t , we obtain r t ( p ∗ t ) -r t ( p t ) = O (( p ∗ t -p t ) 2 ).
3. For t in the k -th episode, namely τ k -1 ≤ t ≤ τ k -1, we have

p ∗ t -p t = g ( µ 0 · ˜ x t ) -g ( ̂ µ k · ˜ x t ) ≤ | ( µ 0 -̂ µ k ) · ˜ x t | , which follows by showing that g is 1-Lipschitz. Further, by Assumption 2.2 (without loss of generality assume C max &gt; 1), we have

E [(( µ 0 -̂ µ k ) · ˜ x t ) 2 ] ≤ C max E [ ‖ ̂ µ k -µ 0 ‖ 2 2 ] , where the equality holds because x t is independent of ̂ µ k . The inequality holds because E ( x t ) = 0 and therefore

<!-- formula-not-decoded -->

from which we obtain that the maximum eigenvalue of E (˜ x t ˜ x T t ) is at most C max &gt; 1.

2 Note that ∇ 2 θ L = ( -1 /τ k -1 ) ∑ τ k -1 t = τk -1 ( ∂ 2 /∂ 2 u t L ) x t x T t , where u t = p t -θ · x t -α 0 . Therefore, ∇ 2 θ L is a d × d matrix of rank at most τ k -τ k -1 . Hence, L ( µ ) is strictly convex in µ only if τ k -τ k -1 ≥ d . However, since we are not updating our estimates in the middle of an episode, episodes of length d yield the regret to scale linearly in d , which is not desired.

3 It is strictly convex over the set of s 0 sparse vectors in d -dimension if the number of samples is above cs 0 log d for a suitable constant c &gt; 0.

Let R t be the regret occurred at step t . Combining the above bounds (step 2 and 3), we arrive at E [ R t ] = O ( s 0 (log d ) /τ k -1 ). Therefore, the cumulative expected regret in episode k works out at O ( s 0 log d ). Since the length of episodes increase geometrically, there are O (log T ) episodes by time T . This implies that the total expected regret by time T is O ( s 0 log d log T ).

## 4.1 Comparison with the 'common' regret of bound Ω( √ T )

There is an often-seen regret bound Ω( √ T ) in the literature of online decision making, which can be improved to a logarithmic regret bound if some type of 'separability assumption' holds true [DHK08, AYPS12]. Separability assumption posits that there is a positive constant gap between the rewards of the best and the second best actions. In our framework, the parameter µ belongs to a continuous set in R d +1 and therefore the separability assumption cannot be enforced as by choosing µ arbitrary close to µ 0 , one can obtain suboptimal (but arbitrary close to optimal) reward. However, our policy achieves O (log T ) regret. Here, we contrast our logarithmic lower bound with the folklore bound Ω( √ T ) to build further insight on our results.

Uninformative prices and Ω( √ T ) lower-bound. We focus on [BR12] which has a close framework to ours in that it considers a dynamic pricing policy from purchasing decisions and presents a pricing policy based on maximum likelihood estimation with regret O ( √ T ). Adopting their notation, it is assumed that market values v t are independent and identically distributed random variables coming from a distribution function that belongs to some family parametrized by z . Denote by d ( p ; z ) the demand curve. This curve determines the probability of a purchase at a given price, i.e., d ( p ; z ) = P z ( v t ≥ p ). [BR12] show that the worst-case regret of any pricing policy must be at least Ω( √ T ) (see Theorem 3.1 therein). The bound is proved by considering a specific family of demand curves d ( p ; z ), such that all demand curves in this family intersect at a common price. Further, the common price is the optimal price for a specific choice of parameter z 0 , i.e, p ∗ ( z 0 ). 4 Therefore, the price p ∗ ( z 0 ) is 'uninformative' since no policy can gain information about the demand parameter z , while pricing p ∗ ( z 0 ). The idea behind the derived lower bound for the worser-case regret is that for a policy to learn the underlying demand curve fast enough, it must necessarily choose prices that are away from (the uninformative) price p ∗ ( z 0 ) and this leads to a large regret when the true demand curve is indeed z 0 .

Intuition behind our results. In contrast to the previous case, for our framework there is no such uninformative price. First, note that the for a choice model with parameters µ 0 = ( θ 0 , α 0 ), the demand curve at time t is given by

<!-- formula-not-decoded -->

For n ≥ 1, we define the aggregate demand function up to time n as d n 1 = ( d 1 , d 2 , . . . , d n ). In the following, we argue that under our setting, there is no uninformative price. For any price p and

4 Specifically, they consider d ( p ; z ) = 0 . 5 + z -z p . Hence d (1; z ) = 1, for all z and it is shown that p ∗ ( z 0 ) = 1 for z 0 = 0 . 5.

any µ 1 , µ 2 , we have where ˜ X is the matrix with rows ˜ x /lscript , for 1 ≤ /lscript ≤ t . We also used the fact that f ( z ) ≥ c &gt; 0 for some constant c because F is strictly increasing by Assumption 2.1. As we show in Appendix A, for n ≥ c 0 s 0 log d (with c 0 a proper constant), ˜ X satisfy a so-called 'restricted eigenvalue', by which we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, for any fixed price p , if we vary the demand parameters µ 1 to some other value ̂ µ 1 , then the aggregate demand at price p also changes by an amount proportional to ‖ µ 1 -µ 2 ‖ 2 . Hence, any price in this setting is informative about the model parameters.

To build further insight, let us consider a more general choice model, where the utility of the customer from buying a product with feature vectors x t at price p is given by

<!-- formula-not-decoded -->

where θ 0 , α 0 , β 0 are unknown model parameters and z t is the noise term. The customer buys the product iff u ( x t ) ≥ 0. Note that the model we studied in this paper (see Equation (2)) is special case when the price sensitivity β 0 is known and hence can be normalized to 1. We next argue that in case of unknown β 0 , the uninformative prices do exist and hence the Ω( √ T ) is still in place.

To see this, fix arbitrary α ∗ , and let θ 0 = 0 and β 0 = g ( α ∗ ) -α ∗ + α 0 . Then, the demand curves will be unaltered over time and are given by

<!-- formula-not-decoded -->

It is easy to verify that p ∗ = 1 is the optimal price for the specific choice of α 0 = α ∗ . Further, all the demand curves intersect at p ∗ = 1 (they all have the value 1 -F ( g ( α ∗ ) -α ∗ ) at this price). Therefore, p ∗ is an uninformative price and no policy can gain information about α 0 by pricing at p ∗ . However, when α 0 = α ∗ , choosing prices that are away from this informative price leads to a large regret. Prices that are close to p ∗ does not have any information gain, and contrasting these two points, it can be shown that the worst case regret id of order Ω( √ T ). A formal proof follows the same lines ad the proof of [BR12, Theorem 3.1] and is omitted.

Finally, it is worth noting that the rate of learning demand parameter µ 0 is chiefly derived by three factors:

- Non-smoothness of distribution function F , as it controls the amount of information obtained about ˜ x t · µ 0 at each t . This is captured by quantity /lscript W defined by (34).
- The rate by which the feature vectors x t span the parameter space. This is controlled through the minimum eigenvalue of Σ, i.e., C min . If C min is small, the randomly generated features are relatively aligned and one requires larger sample size to estimate θ 0 within specified accuracy.
- Complexity of µ 0 . This is captured through the sparsity measure s 0 .

Contribution of these factors to the learning rate can be clearly seen in our derived learning bound (105).

## 4.2 Role of C min

In establishing our results, we relied on Assumption 2.2 which requires the population covariance of features to be positive definite. The lower bound on its eigenvalues, denoted by C min , appears in our regret bound as a factor 1 /C 2 min .

As evident from the proof of Proposition 8.1, Assumption 2.1 can be replaced by the weaker restricted eigenvalue condition [BvdG11, CT07], which is a common assumption in high-dimensional statistical learning. While assumption C min &gt; 0 allows for a fast learning rate of model parameters and a regret bound O (log T ), RMLP policy can still provably achieve regret O ( √ T ), even when C min = 0.

Theorem 4.2. Suppose that product feature vectors are generated independently from a probability distribution P X with a bounded support X ∈ R d . Under Assumption 2.1, the regret of RMLP policy is of O ( √ (log d ) T ) . Proof of Theorem 4.2 is given in Section 8.2.

## 5 Lower bound on regret

As discussed in Section 2.2, if the true parameter µ 0 is known, the optimal policy (in terms of expected revenue) is the one that sets prices as p t = g (˜ x t · µ 0 ). Let H t = { x 1 , x 2 , . . . , x t , z 1 , z 2 , . . . , z t } denote the history set up to time t , and recall that Ω denotes the set of feasible parameters, i.e., Ω = { µ ∈ R d +1 : ‖ µ ‖ 0 ≤ s 0 , ‖ µ ‖ 1 ≤ W } . We consider the following set of policies, Π:

Here π ( p t ) denotes the price posted by policy π at time t .

<!-- formula-not-decoded -->

We provide a lower bound on the achievable regret by any policy in set Π. Indeed this lower bound applies to an oracle who fully observes the market values after the price is either accepted or rejected. Compared to our setting, where the seller observes only the binary feedbacks (purchase/no purchase), this oracle appears exceedingly powerful at first sight but surprisingly, the derived lower bound matches the regret of our dynamic policy, up to a logarithmic factor.

Theorem 5.1. Consider linear model (1) with α 0 = 0 , where the market values v ( x t ) , 1 ≤ t ≤ T , are fully observed. We further assume that market value noises are generated as z t ∼ N (0 , σ 2 ) . Let Π be the set of policies given by (15) . Then, there exists constant C ′ &gt; 0 (depending on W and σ ), such that the following holds true for all T ∈ N .

<!-- formula-not-decoded -->

In the following we give an outline for the proof of Theorem 5.1, summarizing its main steps and defer the complete proof to Section 8.3.

1. We derive a lower bound for regret in terms of the minimax estimation error. Specifically, for t ∈ N , let

<!-- formula-not-decoded -->

be the regret at period t . Define Ω 0 = { θ ∈ R d : ( θ, 0) ∈ Ω } . We show that

<!-- formula-not-decoded -->

for some constants c, C &gt; 0.

2. Let θ T 1 = ( θ t ) T t =1 and define d ( θ T 1 , θ ) ≡ ∑ T t =1 min( ‖ θ t -θ ‖ 2 2 , C ). We use a standard argument (Le Cam's method) that relates the minimax /lscript 2 risk, min θ T 1 max θ 0 ∈ Ω 0 E d ( θ T 1 , θ 0 ), in terms of the error in multi-way hypothesis problem [Tsy08]. We first construct a maximal set of points in Ω 0 , such that minimum pairwise distances among them is at least δ . (Such set is usually referred to as a δ -packing in the literature). Here δ is a free parameter to be determined in the proof. We then use a standard reduction to show that any estimator with small minimax risk should necessarily solve a hypothesis testing problem over the packing set, with small error probability. More specifically, suppose that nature chooses one point from the packing set uniformly at random and conditional on nature's choice of the parameter vector, say θ 0 , the market value are generated according to 〈 x t , θ 0 〉 + z t with z t ∼ N (0 , σ 2 ). The problem is reduced to lower bounding the error probability in distinguishing θ 0 among the candidates in the packing set using the observed market values.
3. We apply Fano's inequality from information theory to lower bound the probability of error [Tsy08]. The Fano bound involves the logarithm of the cardinality of the δ -packing set as well as the mutual information between the observations (market values) and the random parameter vector θ 0 chosen uniformly at random from the packing set. Le Cam's method is used to derive minimal risk lower bound for an estimator ̂ θ , while here we have a sequence of estimators and need to adjust the Le Cam's method to get the lower bound for d ( θ T 1 , θ 0 ).

## 6 Nonlinear valuation function

In previous sections, we focused exclusively on linear valuation function given by Eq (1). Here, we extend our results and assume that the market valuations are modeled by a nonlinear function that depends on products' features and an independent noise term. Specifically, the market value of a product with feature vector x t is given by

<!-- formula-not-decoded -->

where the original features x t are transformed by a feature mapping φ : R d ↦→ R d , and function ψ : R ↦→ R is a general function that is log-concave and strictly increasing. Important examples of this model include log-log model ( ψ ( x ) = e x , φ ( x ) = ln( x )), semi-log model ( ψ ( x ) = e x , φ ( x ) = x ), and logistic model ( ψ ( x ) = e x / (1 + e x ), ψ ( x ) = x ).

Model (19) allows us to capture correlations and non-linear dependencies on the features. We next state our assumption on the feature mapping φ and then discuss our dynamic pricing policy and its regret bound for the general setting (19).

Assumption 6.1. Let p X be an (unknown) distribution from which the original features x t are sampled independently. Suppose that the feature mapping φ has continuous derivative and denote by Σ φ ≡ E ( φ ( x ) · φ ( x ) T ), the covariance of feature vector φ ( x ) under P X . We assume that there exist constants C min and C max such that for every eigenvalue σ of Σ φ , we have 0 &lt; C min ≤ σ &lt; C max &lt; ∞ .

Invoking Assumption 2.1, P X has a bounded support X and since φ has continuous derivative, it is Lipschitz on X and hence the image of X under φ remains bounded. Therefore, the new features φ ( x t ) are also sampled independently from a bounded set. The condition on Σ φ is analogous to that on Σ, as required by Assumption 2.2 for the linear setting.

Based on feature mapping φ , validity of Assumption 6.1 may depend on all moments of distribution P X . We provide an alternative to this assumption, which only depends on feature mapping φ and the second moment of P X . In stating the assumption, we use the notation D φ to denote the derivative matrix of a feature mapping φ . Precisely, for φ = ( φ 1 , . . . , φ d ), with φ i real-valued function defined on R d , we write D φ = ( ∂φ i /∂x j ) 1 ≤ i ≤ j ≤ d .

Assumption 6.2. Suppose that feature mapping φ has continuous derivative and its derivative D φ ( x ) is full-rank for almost all x . In addition, there exist constants C min and C max such that for every eigenvalue σ of covariance Σ, we have 0 &lt; C min ≤ σ &lt; C max &lt; ∞ .

Recall that the noise terms { z t } t ≥ 1 are drawn independently and identically from a distribution with cumulative function F and density f ( x ). Let λ ( v ) = f ( v ) / (1 -F ( v )) be the hazard rate function for distribution F . For a log-concave function ψ , we define

<!-- formula-not-decoded -->

- Right-hand side of (20) is strictly increasing and hence, g -1 ψ is well-defined.

Note that ψ ′ ( v ) /ψ ( v ) = log ′ ψ ( v ) and since ψ is log-concave, this term is decreasing. Further, since 1 -F is log-concave then its hazard rate λ is increasing (See proof of Lemma C.1.) Combining these observations, we have that -λ -1 ( ψ ′ ( v ) /ψ ( v )) is increasing. Consequently,

- We have ( g -1 ψ ) ′ ( v ) ≥ 1, for all v . This implies that 0 &lt; g ′ ψ ( v ) ≤ 1, for all v .

It is worth noting that for ψ ( v ) = v (linear model), we have g ψ = g , where g is defined by (5). Our pricing policy for the nonlinear model is conceptually similar to the linear setting: The policy runs in an episodic manner. During episode k , the prices are set as p t = ψ ( g ψ ( ̂ µ k · ˜ x t )), where ̂ µ k denotes the estimate of the true parameters ( θ 0 , α 0 ) using a regularized maximum-likelihood estimator applied to observations in the previous episode, and ˜ x t = ( φ ( x t ) , 1).

Theorem 6.3. Let ψ be log-concave and strictly increasing. Suppose that Assumptions 2.1 and 6.1 (or its alternative, Assumption 6.2) hold. Then, regret of the RMLP policy described as Algorithm 2 is of O ( s 0 log d · log T ) .

We describe our (modified) RMLP policy in Algorithm 2. There a few differences between Algorithm 2 and Algorithm 1: Firstly, the features x t are replaced by ˜ x t = ( φ ( x t ) , 1). Secondly, in the regularized estimator, prices p t are replaced by ψ -1 ( p t ). Thirdly, in the last step of algorithm prices are set as ψ ( g ψ ( ̂ µ k · ˜ x t )), with g ψ defined by Equation (20). Our next theorem bounds the regret of our pricing policy (Algorithm 2).

Proof of Theorem 6.3 is given in Appendix 8.4. Here, we summarize its key ingredients.

1. By increasing property of ψ , a sale occurs at period t when z t ≥ ψ -1 ( p t ) -µ 0 · ˜ x t . Hence, the log-likelihood estimator for this setting reads as (22). By virtue of Assumption 6.1 (or its alternative, Assumption 6.2) we get a similar estimation error for the regularized estimator to the one in Proposition 8.1.

Input: (at time 0 ) function g , regularizations λ k , W (bound on ‖ θ 0 ‖ 1 ),

Output:

Input: (arrives over time) covariate vectors { ˜ x t = ( φ ( x t ) , 1) } t ∈ N

- 1: τ 1 ← 1, p 1 ← 0, ̂ µ 1 ← 0 2: for each episode k = 2 , 3 , . . . do 3: Set the length of k -th episode: τ k ← 2 k -1 .

<!-- formula-not-decoded -->

prices { p t } t ∈ N

- 4: Update the model parameter estimate ̂ µ k using the regularized ML estimator obtained from observations in the previous episode:

where L ( µ ) is given by:

<!-- formula-not-decoded -->

- 5: For each period t during the k -th episode, set

## Algorithm 2: RMLP Policy for dynamic pricing under the nonlinear setting

<!-- formula-not-decoded -->

2. Similar to our derivation for linear setting, we show that the optimal pricing policy that knows µ 0 = ( θ 0 , α 0 ) in advance is given by p ∗ t = ψ ( g ψ ( θ 0 · ˜ x t )), where g ψ is defined based on Equation (20).
3. The difference between the posted price and the optimal price can be bounded as p t -p ∗ t = ψ ( g ψ ( ̂ µ k · ˜ x t )) -ψ ( g ψ ( µ 0 · ˜ x t )) ≤ L | ˜ x t · ( ̂ µ k -µ 0 ) | , for a constant L &gt; 0. This bound is similar to the corresponding bound for the linear setting, and following the same lines of our regret analysis for that case, we get R ( T ) = O ( s 0 log d · log T ).

## 7 Knowledge of market noise distribution

The proposed RMLP policy has assumed that the market noise distribution F is known to the seller. Knowledge of F has been used both in estimating the model parameters ( θ 0 , α 0 ) and in setting the prices p t . On the other hand, the benchmark policy is also assumed to have access to model parameters and the distribution F . Therefore, the regret bound established in Theorem 4.1 essentially measures how much the seller loses in revenue due to lack of knowledge of the underlying model parameters. In practice, however, the underlying distribution of valuations is not given and this rises the question of distribution-independent pricing policy.

It is worth mentioning that in some applications, although the underlying distribution of valuations is unknown, it belongs to a known class of distributions. For example, lognormal distributions

have proved to be a good fit for the distribution of valuations of advertisers in online advertising markets [EOS07, LP07, XYL09, BFMM14]. In Section 7.1, we consider a model where the underlying distribution belongs to a known class of log-concave distributions and propose a policy whose regret is O ( √ T ). We also argue that no policy can get a better regret bound.

Next, we pursue pricing policies under completely unknown distribution. Here, the regret is measured against an optimal clairvoyant policy that has full knowledge of the model parameters µ 0 and market noise realizations , { z t } t ≥ 1 , and thus extracts the customers' valuation at each step. Note that such a clairvoyant policy is much more powerful than the one considered in previous sections, as now it has access to noise realizations while before it only had knowledge of the noise distribution F .

## 7.1 Unknown distribution from a known class

Suppose that the maket noises are generated from a log-concave distribution F m,σ (e.g., Lognormal), with unknown mean m and unknown variance σ 2 . Without loss of generality, we can assume that m = 0; otherwise, in the valuation model (1), m can be absorbed in the intercept term α 0 . We next explain how the RMLP policy can be adapted to this case.

Define β 0 = 1 /σ and consider the transformation ˜ v t = β 0 v t , ˜ θ 0 = β 0 θ 0 , ˜ α 0 = β 0 α 0 , ˜ z t = β 0 z t . Then, the valuation model (1) can be written as

<!-- formula-not-decoded -->

where ˜ z t are drawn from F 0 , 1 . To lighten the notation, we use the shorthand F ≡ F 0 , 1 . We also let µ 0 = ( ˜ θ 0 , ˜ α 0 ). The response variables y t are then given by y t = I (˜ v t ≥ β 0 p t ).

<!-- formula-not-decoded -->

We propose a variant of RMLP policy, called RMLP-2 for this case. Similar to RMLP, it runs in an episodic manner but the length of episodes grows linearly. (Episode j is of length j periods.) At the first period of each episode, the price is chosen randomly and independently from the feature vectors. To be concrete, we set the price uniformly at random from [0 , 1]. At the other periods of the episode, the price is set optimally based on the current estimate of the model parameters. Specifically, for episode k , we set p t = (1 / ̂ β k ) g ( ̂ µ k · ˜ x t ), where the pricing function g is defined based on distribution F ≡ F 0 , 1 , given by (5), and the estimates ( ̂ µ k , ̂ β k ) are obtained via regularized loglikelihood. In forming the log-likelihood loss, we only consider the first period of each episode, where the prices are set randomly; for k ≥ 1, we denote by A k the set of first periods in episodes 1 , . . . , k , and write the log-likelihood based on the samples in A k :

A formal description of RMLP-2 is given in Algorithm 3. Note that in contrast to RMLP, in the RMLP-2 the length of episodes grows linearly rather than exponentially. This ways, we have |A k | = k , which provides enough samples to update the estimate ̂ θ k at a proper rate to get regret O ( √ T ). Our next result bounds the regret of RMLP-2.

Theorem 7.1. Consider the valuation model (1) , where noises z t are generated from a distribution F m,σ , with unknown mean m and variance σ 2 . Under Assumption 2.2 and assuming that distribution F m,σ satisfies Assumption 2.1, the regret of RMLP-2 policy is of O ( s 0 (log d ) √ T ) . Further, regret of any pricing policy in this case is Ω( √ T ) .

Input:

Pricing function g (corresponding to F 0 , 1 ), regularizations λ k , W (bound on ‖ µ 0 ‖ 1 )

Input: (arrives over time)

covariate vectors { ˜ x t = ( x t , 1) } t ∈ N

Output:

- 1: for each episode k = 1 , 2 , . . . do

prices { p t } t ∈ N

- 2: For the first period of the episode, offer the price uniformly at random from [0 , 1].
- 3: Denote by A k the set of first periods in episodes 1 , . . . , k .
- 4: Update the model parameter estimate µ k using the regularized ML estimator:

with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 5: For each period t during the k -th episode, set

## Algorithm 3: RMLP-2 policy for dynamic pricing

<!-- formula-not-decoded -->

We refer to Section 8.5 for the proof of Theorem 7.1. As discussed in the proof, the lower bound Ω( √ T ) applies to this case due to the existence of non-informative prices; See also Section 4.1.

## 7.2 A distribution-independent pricing policy

In this section, we propose a policy, called DIP (Distribution Independent Pricing), for the settings that the underlying valuation distribution is completely unknown. Before a detailed description of DIP, we provide the general intuition behind this policy.

Here, our focus is on applications where signal-to-noise ratio is large. Specifically, we assume that the customer's valuations are given by model (1) and the noise terms z t are drawn from an unknown distribution with bounded support. (The support of distribution is considered to be small compared to the nominal valuations ˜ x t · θ 0 + α 0 .) Therefore, valuations v t belong to a bounded interval [0 , K ]. Similar to RMLP, the DIP policy operates in episodes. Each episode consists of an exploration phase followed by an exploitation phase. All exploration phases are of length c , where c ≥ 1 is a constant. In these phases, the prices are set uniformly at random from the interval [0 , K ]. Following the exploration phase of episode k , there is an exploitation phase of k periods. In this phase, we offer the optimal prices based on the current estimates of the model parameters from the responses in the previous exploration phases. Therefore, the k -th episode consists of ( c + k ) periods. In early episodes, the ratio of exploration phase to exploitation phase is high, as we know very little about the model parameters and then it becomes lower in the later episodes as we have already obtained a good estimate of the underlying model parameters.

The formal description of the DIP policy is given in Algorithm 4. Our focus is on bounded noise, i.e, | z t | ≤ δ almost surely and hence we can take K = W + δ as the bound on customer's

Input:

exploration length ( c ), regularizations λ k , W (bound on ‖ µ 0 ‖ 1 ), noise bound δ

Input: (arrives over time) covariate vectors { ˜ x t = ( φ ( x t ) , 1) } t ∈ N

Output:

Output:

prices { p t } t ∈ N

- 1: K ← W + δ

- 2: for each cycle k = 1 , 2 , 3 , . . . do

- 3: Exploration episode ( c periods): Offer prices uniformly at random from [0 , K ].
- 4: Update the model parameter estimate ̂ µ k using the regularized ML estimator obtained from observations during the previous exploration episodes:

where L ( µ ) is given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and A k denotes the set of periods belonging to the first k exploration episodes. k

- 5: Exploitation episode ( k periods): Offer prices based on the current estimate µ as

## ̂ Algorithm 4: Distribution Independent Pricing (DIP) Policy

valuations.

We next prove a regret guarantee for DIP policy.

Theorem 7.2 (Regret Upper Bound) . Consider the valuation model (1) , where the noise terms { z t } t ≥ 1 are generated from an unknown zero-mean distribution with support [ -δ, δ ] . Further, suppose that the feature vectors satisfy Assumptions 2.2. Then, the regret of the DIP policy is O ( s 0 (log d ) √ T + δT ) . Here, the regret is against an optimal clairvoyant policy that knows the model parameters and the noise realizations { z t } t ≥ 1 .

In the following, we outline the main idea of the proof of Theorem 7.2. The proof minutiae are deferred to Section 8.6.

For a given time T , it is easy to verify that the number of cycles up to time T is O ( √ T ). Recall that in the exploration phases the prices are set randomly. The regret incurred in each period is O (1) since the valuations are bounded. Therefore, the cumulative regret in the exploration phases up to time t is O ( √ T ). Next, we bound the regret incurred during the exploitation phases. For each episode k , prices are posted as p t = ̂ µ k · ˜ x t -2 δ . Note that the term 2 δ is to ensure purchases occur with high probabilities. The regret is then due to the conservative term 2 δ and the estimation error ˜ x t · ( ̂ µ k -µ 0 ). The aggregate effect of these two factors results in a total regret of O ( δk + s 0 log d ) in episode k . Since there are O ( √ T ) cycles up to time T , the total regret incurred during the exploitation episodes is O ( δT + s 0 (log d ) √ T ).

<!-- formula-not-decoded -->

## 8 Proof of Theorems

## 8.1 Proof of Theorem 4.1

Following step 1 of the proof outline mentioned in Section 4, we consider the problem of estimating µ 0 based on observations from previous episode. Before we proceed, let us emphasize once again that the way RMLP is designed, posted prices at each episode are statistically independent from the market noises in that episode. This can be easily observed because p t = g ( x t · ̂ µ k ) for t belonging in the k -th episode, and µ k is estimated based on the samples in the ( k -1)-th episode.

Using probabilistic model (3), µ 0 is estimated by solving a regularized maximum likelihood (ML) optimization problem. The (normalized) negative log-likelihood function for µ reads as

̂ We fix k ≥ 1 and to lighten the notation, we use the indices 1 , 2 , . . . , n to correspond to periods in the k -the episode, i.e., t = τ k , τ k +1 , . . . , τ k +1 -1.

<!-- formula-not-decoded -->

Parameter µ is estimated as the solution of the following program:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define /lscript W as follows which corresponds to 'flatness' of function log F :

By Assumption 2.1, the log-concavity property of F and 1 -F , we have /lscript W &gt; 0.

The next theorem upper bounds the estimation error of the proposed regularized estimator.

Proposition 8.1 (Estimation Error) . Consider linear model (1) with µ 0 = ( θ 0 , α 0 ) ∈ Ω , under Assumptions 2.1 and 2.2. Let ̂ µ be the solution of optimization problem (33) with λ ≥ 4 u W √ (log d ) /n . Then, there exist positive constants c 0 and C such that, for n ≥ c 0 s 0 log( d ) , the following inequality holds with probability at least 1 -1 /d -2 e -n/ ( c 0 s 0 ) :

We refer to Appendix A for the proof of Proposition 8.1.

<!-- formula-not-decoded -->

As we see the /lscript 2 estimation error scales linearly with the sparsity level s 0 . As s 0 increases, the number of parameters to be estimated becomes larger and this makes the estimation problem harder, leading to worse /lscript 2 bound for a fixed number of samples, n . Further, choosing λ ∼ √ (log d ) /n (where ∼ indicates equality up to a constant factor), our /lscript 2 bound scales logarithmically in the dimension of the demand space, d . This allows to deal with high-dimensional applications and obtain a regret that scales logarithmically in d . Further, the estimation error shrinks as ∼ 1 /n ; getting more samples with fixed value of s 0 and d leads to better estimation accuracy. Finally, note that for small values of /lscript W , the log-likelihood function is very flat and there can be, in principle, vectors µ of log-likelihood value very close to the optimum and nevertheless far from the optimum. In other words, estimation task becomes harder as /lscript W gets smaller and this is clearly reflected in the derived estimation bound.

We next use Proposition 8.1 to bound the expected estimation error.

Corollary 8.2. Under assumptions of Proposition 8.1, the following holds true:

<!-- formula-not-decoded -->

Proof of Corollary 8.2 is straightforward and is omitted.

In the next proposition, we improve bound (36) for n ≥ c 1 d , for a constant c 1 &gt; 0. As we will see, the following result is useful to develop sharper upper bound for regret of RMLP policy.

Proposition 8.3. Under assumptions of Proposition 8.1, there exist constants c, c 1 &gt; 0 , such that for n ≥ c 1 d , the following holds true:

<!-- formula-not-decoded -->

Proposition 8.3 is proved in Appendix B.

We next establish some useful properties of the virtual valuation function ϕ and the price function g .

Lemma 8.4. If 1 -F is log-concave, then the virtual valuation function ϕ is strictly monotone increasing.

Lemma 8.5. If 1 -F is log-concave, then the price function g satisfies 0 &lt; g ′ ( v ) &lt; 1 , for all values of v ∈ R .

Proofs of Lemma 8.4 and 8.5 are given in Appendix C.1 and C.2, respectively. Given that ‖ ̂ µ k ‖ 1 ≤ W and | ˜ x t · µ k | ≤ W for all t, k ,

<!-- formula-not-decoded -->

We are now ready to bound the regret of our policy. For t ≥ 1, let

̂ ̂ where in the first inequality we used the fact that ϕ ( v ) is increasing as per Lemma 8.4 and hence g ( v ) = v + ϕ -1 ( -v ) ≤ v + | v | ≤ 2 | v | . Similarly, we have p ∗ t ≤ 2 W for all t .

<!-- formula-not-decoded -->

be the regret at period t . Further, let H t = { x 1 , x 2 , . . . , x t , z 1 , z 2 , . . . , z t } be the history set, up to time t (more precisely, H t is the filtration generated by { x 1 , x 2 , . . . , x t , z 1 , z 2 , . . . , z t } ). We also define ¯ H t = H t ∪ { x t +1 } as the filtration obtained after augmenting by the new feature x t +1 .

We write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define r t ( p ) ≡ p (1 -F ( p -˜ x t · µ 0 )) as the expected revenue under price p . Note that p ∗ t ∈ arg max r t ( p ) and thus r ′ t ( p ∗ t ) = 0. By Taylor expansion,

<!-- formula-not-decoded -->

for some p between p t and p ∗ t .

We next show that | r ′′ t ( p ) | ≤ C , with C = 2( B + WB ′ ), B = max v f ( v ), and B ′ = max v f ′ ( v ). To see this, we write

<!-- formula-not-decoded -->

where we use the fact that p t , p ∗ t ≤ 2 W and consequently p ≤ 2 W .

Combining Equations (41), (42), (43), along with 1-Lipschitz property of g gives

<!-- formula-not-decoded -->

Given that ˜ x t is independent of H t -1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ Σ = E (˜ x t ˜ x T t ). Using Equation (12),

Now, since the length of episodes grows exponentially, the number of episodes by period T is logarithmic in T . Specifically, T belongs to episode K = /floorleft log T /floorright +1. Hence,

<!-- formula-not-decoded -->

We bound the total regret over each episode by considering three separate cases:

- 2 k -2 ≤ c 0 s 0 log d : Here, c 0 is the constant in the statement of Proposition 8.1. In this case, episodes are not large enough to estimate µ 0 accurately enough, and thus we use a naive bound on regret. Clearly, by (38), we have E ( R t ) ≤ p ∗ t ≤ 2 W . Since the length of k th episode is 2 k -1 ≤ 2 c 0 s 0 log d , the total regret incurred during episode k is at most 4 c 0 Ws 0 log d .
- c 0 s 0 log d ≤ 2 k -2 ≤ c 1 d : Here, c 1 is the constant in the statement of Proposition 8.3. Continuing from Equation (46) and applying Corollary 8.2 to episode k , we obtain

<!-- formula-not-decoded -->

where in the last step we used τ k = 2 τ k -1 and τ k = 2 k -1 ≤ 2 c 1 d . Therefore, in this case

<!-- formula-not-decoded -->

where C ′ hides various constants in the right-hand side of (48).

- c 1 d &lt; 2 k -2 : Continuing from Equation (46) and applying Proposition 8.3 to episode k , we obtain

<!-- formula-not-decoded -->

Therefore, in this case

<!-- formula-not-decoded -->

where C ′ hides various constants in the right-hand side of (50). Combining the above three cases into Equation (47), we get

<!-- formula-not-decoded -->

which concludes the proof.

## 8.2 Proof of Theorem 4.2

By using Equation (78), we have with ˜ Σ = E (˜ x t ˜ x T t ).

<!-- formula-not-decoded -->

Therefore, letting K = /floorleft log T /floorright +1,

<!-- formula-not-decoded -->

We next bound the right-hand side of the above bound. Let X ( k ) ∈ R τ k × d be the matrix obtained by stacking feature vectors in episode k as rows. By applying bound (101) to samples in episode ( k -1), we get that with probability at least 1 -1 /d ,

Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For k ≥ 1, let S ( k ) ∈ R d × d be the empirical covariance of X ( k ) , and define E ( k ) = ˜ Σ -S ( k ) . Then,

<!-- formula-not-decoded -->

〈 ̂ µ k -µ 0 , ˜ Σ( ̂ µ k -µ 0 ) 〉 = 〈 ̂ µ k -µ 0 , S ( k -1) ( ̂ µ k -µ 0 ) 〉 + 〈 ̂ µ k -µ 0 , E ( k -1) ( ̂ µ k -µ 0 ) 〉 . (57) The first term is bounded using Equation (56) as follows:

with probability at least 1 -1 /d .

The second term can be bounded by virtue of the following lemma, whose proof if deferred to Appendix C.9

Lemma 8.6. For any k ≥ 1 and any vector v ∈ R d , we have with probability at least 1 -8 /d 2 .

By Lemma 8.6, we have

<!-- formula-not-decoded -->

Combining Equations (58) and (59), with probability at least 1 -9 /d we have with probability at least 1 -8 /d 2 .

for some constant C &gt; 0.

<!-- formula-not-decoded -->

Following a similar argument as in Section 8.1 (see Equation (47) and onwards) we have that the following holds for a suitable constant C &gt; 0:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 8.3 Proof of Theorem 5.1

The regret benchmark (7) is defined as the maximum gap between a policy and the oracle policy over different µ 0 ∈ Ω and p X ∈ Q ( X ). Without loss of generality, we assume X = [ -1 , 1] d . In order to obtain a lower bound on the regret, it suffices to consider a specific distribution in Q ( X ). We consider a distribution p X that selects coordinates x i , 1 ≤ i ≤ d , uniformly at random from {-1 , 1 } and independent of each other. We further assume that α 0 = 0 and θ 0 ∈ Ω 0 , where

<!-- formula-not-decoded -->

Fix an arbitrary policy π in family Π. Since the assumption α = 0 is known to the oracle, we have π ( p t ) = g ( x t · θ t ), for some θ t ∈ Ω 0 , which is H t -1 -measurable . Recalling our notation in the proof of Theorem 4.1, R t denotes the regret occurred at step t and by Equations (41), (42), we have

<!-- formula-not-decoded -->

for some p between p t and p ∗ t .

Our first lemma will be used in lower bounding E ( R t | ¯ H t -1 ).

Lemma 8.7. There exists a constant c 1 &gt; 0 (depending on W and σ ) such that, with probability one 5 , r ′′ t ( p ∗ t ) ≤ -c 1 , for all t ≥ 1 . Further, there exists constant δ &gt; 0 (depending on W and σ ) such that r ′′ t ( p ) ≤ -c 1 / 4 for p ∈ [ p ∗ t -δ, p ∗ t + δ ] , with probability one.

Proof of Lemma 8.7 is given in Appendix C.3.

Continuing from Equation (61), we consider two separate cases:

- | p t -p ∗ t | ≤ δ : We have p ∈ [ p ∗ t -δ, p ∗ t + δ ] and therefore by applying Lemma 8.7 we obtain

<!-- formula-not-decoded -->

- | p t -p ∗ t | &gt; δ : Since function r t has only one local maximum, namely p ∗ t , the function is increasing before p ∗ t and decreasing afterward. Therefore, if p t ≤ p ∗ t -δ then

<!-- formula-not-decoded -->

where p is some point in [ p ∗ t -δ, p ∗ t ] and we applied Lemma 8.7 in the last step. Similarly, for p t ≥ p ∗ t + δ we obtain

<!-- formula-not-decoded -->

where p ∈ [ p ∗ t -δ, p ∗ t ] this time. Combining these two inequalities, we get that r t ( p ∗ t ) -r t ( p t ) ≥ c 1 δ 2 / 8, if | p ∗ t -p t | ≥ δ .

5 The randomness comes from randomness in prices which in turn comes from randomness in features x t .

Writing the bounds in the two cases together, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used the fact that by Lemma 8.5, g ′ ( v ) &gt; c 2 over the bounded interval [ -W,W ], for some constant c 2 &gt; 0. We recall the definition of history set H t ≡ ¯ H t \{ x t +1 } = { x 1 , x 2 , . . . , x t , z 1 , z 2 , . . . , z t } . Since H t ⊆ ¯ H t , by iterated law of expectation, we get

Note that x t is independent of H t -1 and θ t -θ 0 is H t -1 -measurable.

We use the following lemma to lower bound the right-hand side of (67).

Lemma 8.8. Let x ∈ R d be a random vector such that its coordinates are chosen independently and uniformly at random from {-1 , 1 } . Further, suppose that v ∈ R d and δ &gt; 0 are deterministic. Then,

<!-- formula-not-decoded -->

Proof of Lemma 8.8 is given in Appendix C.4.

Applying Lemma 8.8 to bound (67), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, taking expectation from both sides with respect to H t -1 , we arrive at

Equation (70) lower bounds the expected regret at each step to the /lscript 2 estimation error.

We continue by establishing a minimax lower bound on /lscript 2 -risk of estimation.

Lemma 8.9. Consider linear model (1) , with α 0 = 0 , and assume that the market values v ( x t ) , 1 ≤ t ≤ T , are fully observed and the feature vectors are generated according to p X , described above. We further assume that the noise in market value is generated as z t ∼ N (0 , σ 2 ) . For a sequence of estimators θ t , we let θ t 1 = ( θ 1 , θ 2 , . . . , θ t ) . Then, conditional on feature vectors ( x 1 , . . . , x T ) , and for any fixed value C &gt; 0 , there exists a nonnegative constant C , depending on C , σ , W , such that

<!-- formula-not-decoded -->

Proof of Lemma 8.9 is given in Appendix C.5.

We are now ready to lower bound the regret of any policy in Π.

<!-- formula-not-decoded -->

where the last step follows from Lemma 8.9.

<!-- formula-not-decoded -->

## 8.4 Proof of Theorem 6.3

Let ˜ x t = ( φ ( x t ) , 1) denote the transformed features under the feature-map, augmented by the constant term 1. Also, let ˜ p t = ψ -1 ( p t ). We first show that Assumption 6.2 implies Assumption 6.1, and therefore it suffices to prove the theorem under Assumption 6.1.

Lemma 8.10. Suppose that Assumption 2.1 hold true. Then, Assumption 6.2 implies Assumption 6.1.

Proof of Lemma 8.10 is given in Appendix C.6.

By Assumption 2.1, the support of P X is abounded set X . Given that φ has a continuous derivative, it is Lipschitz on the bounded set X and ergo the image of X remains bounded under the feature-map φ . Putting differently, features ˜ x t are sampled from a bounded set in R d . Without loss of generality, we assume ‖ ˜ x t ‖ ∞ ≤ 1. Further, as per Assumption 6.1, the covariance of the underlying distribution Σ φ is positive definite with bounded eigenvalues.

On a different note, since ψ is strictly increasing, a sale occurs at period t when µ 0 · ˜ x t + z t ≥ ψ -1 ( p t ) = ˜ p t . Therefore the (negative) log-likelihood function for µ reads as

<!-- formula-not-decoded -->

The estimation bound (35) also holds for this setting and the proof goes along the same lines of the proof of Proposition 8.1, with slight modifications: ( i ) the features x t and prices p t should be replaced by ˜ x t and ˜ p t . ( ii ) Quantity u W and /lscript W in the statement of Propostion 8.1 should be set as M = (1 / 3) g ψ (0) + (2 / 3) W . This follows from the bounds below

Here, we used the facts that g ψ is 1-Lipschitz and increasing as explained below Equation (20).

We next characterize the optimal policy when the true parameter µ 0 = ( θ 0 , α 0 ) is known. The expected revenue from a poster price p works out at p (1 -F ( ψ -1 ( p ) -µ 0 · ˜ x t )). Writing this in terms of ˜ p = ψ -1 ( p ), the first order condition for the optimal price reads as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where λ denotes the hazard rate function. Equivalently

<!-- formula-not-decoded -->

By definition of function g ψ as per Equation (20), we get ˜ p ∗ = g ψ ( µ 0 · ˜ x t ) and thus p ∗ = ψ ( g ψ ( µ 0 · ˜ x t )). We are now ready to bound the regret of the algorithm. Similar to Equation (44), we have

<!-- formula-not-decoded -->

where L ≡ max | v |≤ ψ ( M ) | ψ ′ ( v ) | (since ψ is continuously differentiable, it attains a maximum over a bounded set.) In addition, we used the fact that g ′ ψ ( v ) ≤ 1 as explained below Equation (20). The inequalities above then follow from the mean-value theorem.

Given that ˜ x t is independent of H t -1 , we have

<!-- formula-not-decoded -->

where ˜ Σ φ = E (˜ x t ˜ x T t ). Using Assumption 6.1,

Rest of the proof is similar to proof of Theorem 8.1 (see after Equation (47)).

<!-- formula-not-decoded -->

## 8.5 Proof of Theorem 7.1

We consider representation (24) of the valuations and use the notation ˜ x t = ( x t , 1), µ 0 = ( ˜ θ 0 , ˜ α 0 ). Fixe k ≥ 1. Letting x ′ t = ( -˜ x t , p t ), we can write the log-likelihood loss as:

<!-- formula-not-decoded -->

Note that for t ∈ A k , prices are posted uniformly at random in [0 , 1] independently from the feature vector. Therefore, the population correlation works out at

<!-- formula-not-decoded -->

Given that Σ /followsequal C min I, we have Σ ′ /followsequal C ′ min I, with C ′ min ≡ min( C min , 1 / 12).Therefore, the augmented feature vectors x ′ t satisfy Assumption 2.2, with C ′ min &gt; 0. By applying Proposition 8.1, we get

‖ ( ̂ µ k , ̂ β k ) -( µ 0 , β 0 ) ‖ 2 2 ≤ Cs 0 λ 2 k //lscript 2 W . (81) with probability at least 1 -1 /d -2 e -k/ ( c 0 s 0 ) . We are now ready to bound the cumulative regret. Before proceeding, we need to figure out the clairvoyant policy.

Lemma 8.11. Let g be the pricing function corresponding to distribution F = F 0 , 1 , given by g ( v ) = v + ϕ -1 ( -v ) , where ϕ ( v ) = v -(1 -F ( v )) /f ( v ) is the virtual valuation function. Then, under model (24) , the clairvoyant optimal prices are given by

<!-- formula-not-decoded -->

with µ 0 = ( ˜ θ 0 , ˜ α 0 ) and ˜ x t = ( x t , 1) .

Proof of Lemma 8.11 is given in Appendix C.7.

In the first period that the price is set randomly, we use the following naive bound on the regret:

<!-- formula-not-decoded -->

where in the first inequality we used the fact that ϕ ( v ) is increasing for log-concave distribution and hence g ( v ) = v + ϕ -1 ( -v ) ≤ v + | v | ≤ 2 | v | . The last step holds because µ 0 /β 0 = ( θ 0 , α 0 ) and ‖ ( θ 0 , α 0 ) ‖ 1 ≤ W .

We next bound the regret at other periods of the episode. Let ¯ H t = { x ′ 1 , . . . , x ′ t , x ′ t +1 , z 1 , . . . , z t } be the history set up to time t . Similar to (44), we write

<!-- formula-not-decoded -->

In the last step, the first term is bounded using 1-Lipschitz property of g and the second term is bounded using the observation (1 / ̂ β k ) g (˜ x t · ̂ µ k ) ≤ 2 W , which can be derived similar to Equation (83). Recalling our notation H t = ¯ H t \{ x ′ t +1 } and applying the law of iterated expectations, we have with C 1 = max( CC max /β 2 0 , 4 C 1 W 2 /β 2 0 ).

Therefore, by applying bound (81) and following similar lines as in proof of Theorem 4.1 (see Equation (46) onwards), we bound the total regret during episode k as follows:

<!-- formula-not-decoded -->

Given that episode k is of length k ,

<!-- formula-not-decoded -->

for some constant C &gt; 0. Here, we use that fact that episode k is of length k .

We next argue that the number of episodes before time T is at most K 0 = √ 2 T 0 . To see this, it suffices to note that the total number of time periods after K 0 episodes is at least K 0 + ∑ K 0 c =1 c ≥ K 0 ( K 0 +1) / 2 ≥ T .

Therefore, by using bound (93), we get

<!-- formula-not-decoded -->

For the lower bound Ω( √ T ), note that under model (24) we can define the (scaled) customer's utility as

<!-- formula-not-decoded -->

Then, a purchase occurs if ˜ u ( x t ) &gt; 0. Following our discussion in Section 4.1 (see after Equation (14)), since β 0 is unknown, the uninformative prices do exist and therefore Ω( √ T ) applies to this case.

## 8.6 Proof of Theorem 7.2

We begin by stating a bound on the mean squared error of the estimator ̂ µ k given by optimization (29). Without loss of generality, we can assume that the noise distribution is zero-mean. Otherwise, the mean can be absorbed in the model intercept α 0 .

Proposition 8.12. Consider linear model (1) under Assumption 2.2, where the noise term z t are generated from an unknown distribution with mean zero and support in [ -δ, δ ] . Also, suppose that µ 0 ∈ Ω and let ̂ µ k be the solution of optimization problem (29) with K = W + δ and λ ≥ 8( K + W ) √ log d ck . Then, there exist positive constants c 0 and C such that, the following inequality holds with probability at least 1 -1 /d -2 e -ck/ ( c 0 s 0 ) :

The proof of Proposition 8.12 is given in Appendix C.8.

With Proposition 7.2 in place, we next bound the regret of DIP policy. First, we show that the regret incurred during the exploration phase of episode k is O (1). Since the noise is bounded, we have the following bound on the customer's valuation at each period

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, the regret against a clairvoyant that can extract the valuation at each period is also bounded by K , and the regret using the exploration phase of episode k is bounded by cK .

Next, we bound the regret incurred during the exploitation phase of episode c . During this phase, DIP policy offers prices p t = ̂ µ k · ˜ x t -2 δ . The revenue generated can be lower bounded as follows:

<!-- formula-not-decoded -->

where we used the fact that | z t | ≤ δ . Consequently, the regret at each period of this phase can be bounded as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ Furthermore, by Markov inequality, where in the last step, we have first computed the expectation with respect to ˜ x t and used the fact that ˜ x t is independent from the residual µ k -µ 0 .

<!-- formula-not-decoded -->

̂ Putting Equation (91) and (92) together, we obtain

Using the result of Proposition 8.12 and following a similar argument as in the proof of Theorem 8.1, we bound the total regret incurred in episode k . Given that episode k is of length k , we obtain

<!-- formula-not-decoded -->

for some constant C &gt; 0.

Now, we are ready to bound the cumulative regret incurred in the first T periods. Note that the way the cycles are defined in DIP policy, the number of cycles up to time T is at most √ 2 T . Hence,

<!-- formula-not-decoded -->

where C ′ = 64( K + W ) 2 KC max C/c .

## 9 Conclusion

In this work, we leverage tools from statistical learning to design a dynamic pricing policy for a setting wherein the products are described via high-dimensional features. Our policy is computationally efficient and by exploiting the structure of demand parameters, it obtains a regret that scales gracefully with the features dimension and the time horizon. Namely, the regret of our

algorithm scales linearly with the sparsity of the optimal solution and logarithmically with the dimension. We also show an O (log 2 T ) dependence of the regret on the length of the horizon. On the flip side, we provide a lower-bound of O (log T ) on the regret of any algorithm that does not know the true parameters of the model in advance.

A natural next step is providing a tight bound on the regret, closing the gap between the derived upper and lower bounds. Another step would be assuming that θ ∗ is not exactly sparse, but it can be well approximated by a sparse vector, i.e, ‖ θ 0 -θ s 0 ‖ 1 ≤ δ for some s 0 -sparse vector θ s 0 . An interesting question is to figure out how the regret scales with δ .

The choice model that proposed in this work assumes one product arrived at each period, and describes the customer's purchase behavior based on the product features and the posted price. A more general choice model would be the one that assumes multiple products at each period. More specifically, each customer has a 'consideration' set which includes products left after the customer has narrowed down her choices based on her own personal screening criteria, and then chooses the product from this set which brings maximum utility. (We model the no purchase option as an extra product). This generalization is the focus of a future work.

We also believe the ideas and techniques developed in this work can be be applied to other settings such as personalized pricing where information about the buyers can be used for price differentiation or optimizing reserve prices in online ad auctions. Another application would be assortment optimization and learning consumer choice models both in terms of the role of the structure [FJS13, KU16] as well as personalization [GNR14, COPSL15] in data-rich environments.

## A Proof of Proposition 8.1

We start by reviewing the notion of restricted eigenvalue (RE) which is commonplace in highdimensional statistical estimation.

Definition A.1. For a given matrix A ∈ R d × d and some integer s such that 1 ≤ s 0 ≤ d and a positive number c , we say that Restricted Eigenvalue (RE) condition is met if

/negationslash

<!-- formula-not-decoded -->

It is shown in [BvdG11] and [RZ13] that when two matrices A 0 , A 1 are close to each other (in the maximum element-wise norm) compared to sparsity s 0 , the RE condition for A 0 implies the RE condition for A 1 . This is particularly useful when A 0 is a population covariance matrix and A 1 is a corresponding empirical covariance matrix. To apply this result to our case, let ˜ X ∈ R n × d be the feature matrix with rows ˜ x t , corresponding to n products. Let ˜ Σ = E (˜ x t ˜ x T t ). Given that E ( x t ) = 0, we have

Further, by Assumption 2.2, we have Σ /followsequal C min I. Without loss of generality, we can assume C min ≤ 1, which implies ˜ Σ /followsequal C min I. Therefore, ˜ Σ satisfies RE condition with κ 2 ( ˜ Σ , s 0 , 3) ≥ C min . By using the following result, we conclude that ̂ Σ = ( ˜ X T ˜ X ) /n also satisfies RE condition with κ 2 ( Σ , s 0 , 3) ≥ C min / 2.

<!-- formula-not-decoded -->

̂ Proposition A.2. Let ̂ Σ = ( ˜ X T ˜ X ) /n and let S = supp( µ 0 ) be the support of µ 0 . Under Assumption 2.2, ̂ Σ satisfies the restricted eigenvalue condition with constant κ ( ̂ Σ , s 0 , 3) ≥ √ C min / 2 , with probability 1 -e -2 n/ ( c 0 s 0 ) and c 0 = 768 /C 2 min , provided that n ≥ c 0 s 0 log d ,

Proposition A.2 follows from the results established in [BvdG11] and [RZ13]. We outline the main steps of its proof in Appendix A.1 for the reader's convenience.

By the second-order Taylor's theorem, expanding around µ 0 we have for some ˜ µ on the line segment between µ 0 and µ . Invoking (32), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∇ and ∇ 2 represents the gradient and the hessian w.r.t θ . Further,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where u t ( µ ) = p t -〈 ˜ x t , µ 〉 , and log ′ F ( x ) and log ′′ F ( x ) represent first and second derivative w.r.t x , respectively.

By Equation (38), we have

We next introduce the set

<!-- formula-not-decoded -->

Further, recall that the sequences { p t } n t =1 and { x t } n t =1 are independent of { z t } n t =1 . Therefore, { u t ( µ 0 ) } T t =1 and { z t ( µ 0 ) } T t =1 are independent and by (3), we have E [ ξ t ( µ 0 )] = E [ E [ ξ t ( µ 0 ) | u t ( µ 0 )]] = 0. Further, by definition of u W , cf. Equation (11), we have | ξ t ( µ 0 ) | ≤ u W .

<!-- formula-not-decoded -->

By applying Azuma-Hoeffding inequality followed by union bounding over d coordinates of feature vectors, we obtain P ( F ) ≥ 1 -1 /d .

By optimality of ̂ µ , we write

On the other note, ‖ µ 0 ‖ 1 , ‖ ̂ µ ‖ 1 ≤ W and hence ‖ ˜ µ ‖ 1 ≤ W . This implies that | u t (˜ µ ) | ≤ 3 W . Therefore, by definition of /lscript W , cf. Equation (34), we have η t (˜ µ ) ≥ /lscript W . Recalling Equation (97), we get ∇ 2 L (˜ µ ) /followsequal /lscript W ( ˜ X T ˜ X/n ).

<!-- formula-not-decoded -->

̂ ̂ and by rearranging the terms and using (96), we arrive at

<!-- formula-not-decoded -->

Choosing λ ≥ 4 u W √ (log d ) /n , we have on F

Form now on, the analysis is exactly similar to the oracle inequality for Lasso estimator. We bring the analysis here for the reader's convenience.

Let S = supp( µ 0 ). On the left-hand side using triangle inequality, we have

<!-- formula-not-decoded -->

On the right-hand side, we have

<!-- formula-not-decoded -->

Using these two inequalities in (101), we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We next write

<!-- formula-not-decoded -->

where ( a ) follows from Equation (102); ( b ) holds by Cauchy-Shwarz inequality; ( c ) follows form the RE condition, which holds for ̂ Σ = ( ˜ X T ˜ X ) /n as stated by Proposition A.2, with κ ( ̂ Σ , s 0 , 3) ≥ √ C min / 2, and recalling the inequality ‖ ̂ µ S c -µ 0 ,S c ‖ 1 = ‖ ̂ µ S c ‖ 1 ≤ 3 ‖ ̂ µ S -µ 0 ,S ‖ as per Equation (102); Finally ( d ) follows from the inequality 2 √ ab ≤ a 2 + b 2 . Rearranging the terms, we obtain

Applying the RE condition again to the L.H.S of (103), we get

<!-- formula-not-decoded -->

and therefore,

The result follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.1 Proof of Proposition A.2

The proof follows by combining two lemmas from [BvdG11].

Definition A.3. A random variable ν is subgaussian if there exist constants L, σ 0 such that

We show the desired result holds for a more general case, namely for ˜ X with subgaussian entries. Before stating the proof, we recall a few definitions and notations.

<!-- formula-not-decoded -->

Note that bounded random variables are subgaussian. Specifically, if | ν | ≤ ν max , then ν is subgaussian with L = ν max and σ 0 = ν max √ e -1.

For a matrix A , we let ‖ A ‖ ∞ denote its (element wise) maximum norm, i.e., ‖ A ‖ ∞ = max i,j | A ij | . The next lemma shows that if two matrices are close enough in maximum norm and if the compatibility condition holds for one of them then it would also hold for the other one.

Lemma A.4. Suppose that the restricted eigenvalue (RE) condition holds for Σ 0 with constant κ (Σ 0 , s 0 , 3) &gt; 0 . If then the RE condition holds for Σ 1 with constant κ (Σ 1 , s 0 , 3) ≥ κ (Σ 0 , s 0 , 3) / √ 2 .

<!-- formula-not-decoded -->

Proof. Proof of Lemma A.4 We refer to Problem 6.10 of [BvdG11].

Lemma A.5. Consider ˜ X ∈ R n × p with i.i.d. rows generated from a distribution with covariance ˜ Σ ∈ R d × d . Let ̂ Σ = ˜ X T ˜ X/n be the corresponding empirical covariance. Further, suppose that the entries of X are uniformly subgaussian with parameters L, σ 0 . If n ≥ c 0 Ls 0 log d with c 0 = 768 L/κ 2 ( ˜ Σ , s 0 , 3) , then

Proof. Proof of Lemma A.5 The result follows readily from Problem 14.3 on page 535 of [BvdG11].

<!-- formula-not-decoded -->

Next we note that ˜ Σ satisfies the restricted eigenvalue condition with constant κ 2 ( ˜ Σ , s 0 , 3) ≥ C min because of Assumption 2.2. Further, since ‖ ˜ x t ‖ ∞ ≤ 1, we can apply the result of Lemma A.5 with L = 1, σ 0 = √ e -1. Proposition A.2 then follows from Lemma A.4.

## B Proof of Proposition 8.3

Define the event B n as follows:

Using concentration bounds on the spectrum of random matrices with subgaussian rows (see [Ver10, Equation (5.26)]), there exist constants c, c 1 &gt; 0 such that for n &gt; c 1 d , we have P ( B n ) ≥ 1 -e -cn 2 .

<!-- formula-not-decoded -->

For γ &gt; 0, we define the event F γ = {‖∇L ( µ 0 ) ‖ ∞ ≤ γ } . Using characterization (97), and by applying Azuma-Hoeffding inequality (similar to our argument after Equation (98)), we obtain

<!-- formula-not-decoded -->

We also let E 1 ,n ≡ B n ∩ F λ/ 2 , E 2 ,n ≡ B n ∩ F c λ/ 2 . To lighten the notation, we use the shorthand D ≡ ‖ ̂ µ -µ 0 ‖ 2 2 . We then have

<!-- formula-not-decoded -->

We treat each of the terms on the right-hand side separately.

- Term 1: We have

<!-- formula-not-decoded -->

- Term 2: Similar to proof of Proposition 8.1, on E 1 ,n , we have D ≤ 16 s 0 λ 2 / ( /lscript 2 W C 2 min ). Hence,

<!-- formula-not-decoded -->

- Term 3: To bound term 3, we first prove the following lemma.

Lemma B.1. On event E n ( γ ) ≡ B n ∩ F γ , with γ &gt; λ/ 2 , we have

<!-- formula-not-decoded -->

Lemma B.1 is proved in Section B.1.

We next bound term 3 as follows. Let L = 9 λ 2 d/ ( C 2 min /lscript 2 W ).

<!-- formula-not-decoded -->

For the first term on the right-hand side we write

<!-- formula-not-decoded -->

where the last step holds from Equation (108) with γ = λ/ 2.

<!-- formula-not-decoded -->

We next upper bound the second term. For arbitrary fixed c &gt; 1, let γ = √ Lc/ ( dκ ), with κ = 36 / ( /lscript 2 W C 2 min ). It is easy to verify that γ = √ cλ/ 2 &gt; λ/ 2. Further, by virtue of Lemma B.1, on E ( γ ) we have D ≤ κγ 2 d = Lc . Hence,

Further, applying Equation (108) and plugging for γ , we obtain

<!-- formula-not-decoded -->

Here, the second second step follows from definition of L and the last step holds because λ ≥ 4 u W √ (log d ) /n . Combining Equations (114) and (115), we have

<!-- formula-not-decoded -->

Using bounds (113) and (116) in Equation (112), we obtain

<!-- formula-not-decoded -->

The result follows by putting the upper bounds on the three terms together.

## B.1 Proof of Lemma B.1

We start by rewriting Equation (100), which follows from optimality of ̂ θ and log-concave property of the loss function.

On event E n ( γ ), Equation (118) implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the assumption γ &gt; λ/ 2 and our shorthand D ≡ ‖ µ -µ 0 ‖ 2 2 , we get

̂ ̂ Writing the above bound in terms of our shorthand D ≡ ‖ ̂ µ -µ 0 ‖ 2 2 , we obtain the desired result.

<!-- formula-not-decoded -->

## C Proof of Technical Lemmas

## C.1 Proof of Lemma 8.4

We write the virtual valuation function as ϕ ( v ) = v -1 /λ ( v ) where λ ( v ) = f ( v ) 1 -F ( v ) = -log ′ (1 -F ( v )) is the hazard rate function. Since 1 -F is log-concave, the hazard function λ ( v ) is increasing which implies that ϕ is strictly increasing. Indeed, by this argument ϕ ′ ( v ) &gt; 1.

## C.2 Proof of Lemma 8.5

Recalling the definition g ( v ) = v + ϕ -1 ( -v ), we have g ′ ( v ) = 1 -1 /ϕ ′ ( ϕ -1 ( -v )). Since ϕ is strictly increasing by Lemma 8.4, we have g ′ ( v ) &lt; 1. The claim g ′ ( v ) &gt; 0 follows if we show ϕ ′ ( ϕ -1 ( -v )) &gt; 1. For this we refer to the proof of Lemma 8.4, where we showed that ϕ ′ ( v ) &gt; 1 for all v .

## C.3 Proof of Lemma 8.7

Let φ ( v ) and Φ( v ) respectively denote the density and the distribution function of standard normal variable. Function h t and its derivatives read as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define ξ ≡ p ∗ t -x t · θ 0 = g ( x t · θ 0 ) -x t · θ 0 . Writing r ′′ t ( p ∗ t ) in term of ξ , we obtain

<!-- formula-not-decoded -->

By tail bound inequality for Gaussian distribution 1 -Φ( ξ/σ ) ≤ ( σ/ξ ) φ ( ξ/σ ) for ξ ≥ 0. Therefore,

<!-- formula-not-decoded -->

and the same bound obviously holds for ξ &lt; 0.

By definition of function g , | ξ | ≤ 3 W with ϕ being the virtual valuation fusion corresponding to the Gaussian distribution. Hence, φ ( ξ/σ ) ≥ φ (3 W/σ ). Putting this together with (124), we get r ′′ t ( p ∗ t ) ≤ -c 1 with c 1 = (1 /σ ) φ (3 W/σ ).

<!-- formula-not-decoded -->

For the second part of the Lemma statement, set δ ≤ min { 3 W,σ 2 / (18 W ) , σ 2 φ (3 W/σ ) } . For p ∈ [ p ∗ t -δ, p ∗ t + δ ], we have

Using Equation (124) we get p ( p -x t · θ 0 ) /σ 2 ≤ -1 / 2. Further,

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (125), (127) we obtain

<!-- formula-not-decoded -->

The result follows.

## C.4 Proof of Lemma 8.8

Let Z = x · v and ˜ Z = Z/ ‖ v ‖ 2 . Note that Var( ˜ Z ) = 1. Write the expectation in terms of the tail probability

<!-- formula-not-decoded -->

We consider two cases:

- δ ≤ ‖ v ‖ 2 : The right-hand side in (129) can be lower bounded as

<!-- formula-not-decoded -->

In the sequel, we provide two separate lower bounds for the right-hand side. Let ξ ≡ P ( | ˜ Z | ≥ 1). We have

We proceed to obtain another bound which utilizes the fact Var( ˜ Z ) = 1.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For t ≥ 1, we have P ( | ˜ Z | ≥ t ) ≤ ξ . Further, by applying Chernoff bound, we get

<!-- formula-not-decoded -->

Setting λ = t leads to P ( | ˜ Z | ≥ t ) ≤ 2 e -t 2 2 . Combining these bounds into (131), we obtain

<!-- formula-not-decoded -->

We summarize bounds (131) and (133) as in

<!-- formula-not-decoded -->

Turning back to Equation (130), in this case we have

<!-- formula-not-decoded -->

- δ ≥ ‖ v ‖ 2 : Similar to the previous case, the right-hand side in (129) can be lower bounded as

<!-- formula-not-decoded -->

The above two cases can be summarized as E (min( Z 2 , δ 2 )) ≥ 0 . 1min( ‖ v ‖ 2 2 , δ 2 ).

## C.5 Proof of Lemma 8.9

We use a standard argument that relates minimax /lscript 2 -risk in terms of the error in multi-way hypothesis testing problem; See e.g. [YB99, Yu97]. Let { ˜ θ 1 , . . . , ˜ θ m } be a δ -packing of set Ω, meaning that their pairwise distances are all at least δ . Parameter δ is free for now and its value will be determined later in the proof. We further let P j denote the induced probability on market values

( v ( x 1 ) , . . . , v ( x T )), conditional on ( x 1 , . . . , x T ) and for θ 0 = ˜ θ j . In other words, in defining distributions P j we treat feature vectors fixed. Let ν be random variable uniformly distributed on the hypothesis set { 1 , 2 , . . . , m } which indicates the index of the true parameter, i.e, ν = j means θ 0 = ˜ θ j .

/negationslash

Define d ( θ T 1 , θ ) ≡ ∑ T t =1 min( ‖ θ t -θ ‖ 2 2 , C ) and let µ be the value of j for which d ( θ T 1 , ˜ θ j ) is a minimum. Suppose that δ is chosen such that δ 2 ≤ C . If d ( θ T 1 , ˜ θ j ) &lt; δ 2 T/ 4 then µ = j , because assuming otherwise, we have µ = j ′ = j , and by triangle inequality

<!-- formula-not-decoded -->

for all t , where we used the inequality min( a + b, c ) ≤ min( a, c ) + min( b, c ) for a, b, c ≥ 0. Summing over t = 1 , 2 , . . . , T , we get

<!-- formula-not-decoded -->

Using Markov inequality, we can write where we used the assumption µ = j ′ . But this is a contradiction because ‖ ˜ θ j ′ -˜ θ j ‖ 2 ≥ δ (they form a δ -packing of Ω) and δ 2 ≤ C .

/negationslash

<!-- formula-not-decoded -->

/negationslash

We use Fano's inequality to lower bound the error probability on the right-hand side. We first construct a δ -packing of Ω similar to the one proposed in [RWY11, proof of Theorem 1].

Let s = s 0 / 2 ≤ d/ 2 and define

<!-- formula-not-decoded -->

As proved in [RWY11, Lemma 5], there exists a subset ˜ A ⊆ A of cardinality | ˜ A| ≥ exp( s 2 log d -s/ 2 s ) such that the Hamming distance between any two elements in ˜ A is at least s/ 2. Next, consider the set √ 2 s δ ˜ A for some δ ≤ W/ √ 2 s . whose exact value to be determined later. Then, for q in this set, ‖ q ‖ 1 = √ 2 sδ ≤ W and hence √ 2 s δ ˜ A ⊆ Ω 0 . Further, for q, q ′ ∈ √ 2 s δ ˜ A , we have the following bounds:

<!-- formula-not-decoded -->

By (139), the set √ 2 s δ ˜ A forms a δ -packing for Ω 0 with size | ˜ A| .

<!-- formula-not-decoded -->

We now turn back to bound (138). Left-hand side can be lower bounded using Fano's inequality. We omit the details here as it is a standard argument and instead we refer to [RWY11, proof of Theorem 1] for details. Using Fano's inequality and bound (140), we get

/negationslash

<!-- formula-not-decoded -->

with from which we obtain

/negationslash

Choosing δ 2 ≤ δ 2 1 ≡ σ 2 s 32 T log( d -s/ 2 s ), we obtain P ( µ = ν ) ≥ 1 / 4. Therefore, setting δ 2 = min( W 2 2 s , δ 2 1 , C ) and combining with bound (138), we conclude that

<!-- formula-not-decoded -->

Now since s = s 0 / 2 ≤ d/ 2, we have log(( d -s/ 2) /s ) ≥ c log( d/s ) with some constant c &gt; 0. Therefore, by using Equation (142) and substituting for s = s 0 / 2, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We next derive another separate lower bound for minimax risk, by assuming that an oracle gives us the true support of θ 0 . In this case, the least square estimator, applied to the observed features restricted to the true support S , achieves the optimal minimax /lscript 2 rate. This implies that ‖ θ t -θ 0 ‖ 2 2 ≥ cσ 2 s 0 /t , for t ≥ s 0 and a constant c &gt; 0. Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some constant c ′ &gt; 0, depending on σ and C .

Combining bounds in (144) and (146), we have

<!-- formula-not-decoded -->

## C.6 Proof of Lemma 8.10

for a constant ˜ C that depends on C,σ,W . The proof is complete.

We recall the notion of 0 -property established by [Pon87].

Definition C.1. A continuous function has the 0-property , if the pre-image of any set of probability zero is a set of probability zero.

As proved in [Pon87, Theorem 1], if a function φ : X ⊆ R d ↦→ R d is continuously differentiable, then it satisfies 0-property if and only if its derivative D φ is full rank for almost all x ∈ X . Therefore, we need to show that under Assumption 2.1, if φ has 0-property, then Assumption 6.1 holds true.

Supposing otherwise, there exists a nonzero v ∈ R d such that v T Σ φ v = 0. Therefore, E (( z · φ ( x )) 2 ) = 0 which implies that z · φ ( x ) = 0, almost surely. Define S ≡ { z ∈ R d : z · φ ( x ) = 0 } . Space S is ( d -1)-dimensional and all the points in φ ( X ) belong to S almost surely, i.e., P ( φ ( X ) ∩ S c ) = 0. However, since Σ is positive definite (with all of its eigenvalues target than C min , by Assumption 2.1), P X ( S ) = 0. Combining these observations, P ( φ ( X )) ≤ P ( S )+ P ( φ ( X ) ∩ S c ) = 0. Since φ has the 0-property, this implies that P X ( X ) = 0, which is a contradiction because X is the support of P X and thus P X ( X ) = 1. The result follows.

## C.7 Proof of Lemma 8.11

Under model (24), a purchase occurs at time t with the posted price price p if ˜ v t ≥ β 0 p . This is equivalent to ˜ z t ≥ β 0 p -˜ x t · µ 0 . Therefore, the expected revenue from a posted price p is given by

<!-- formula-not-decoded -->

By setting the first order conditions, the optimal price p ∗ t is given by the solution of the following equation:

<!-- formula-not-decoded -->

It is straightforward to verify that the solution p ∗ t of the above equation is given by p ∗ t = (1 /β 0 ) g ( µ 0 · ˜ x t ).

## C.8 Proof of Proposition 8.12

This proposition can be proved by following similar steps as in proof of Propostion 8.1. Indeed, in that proof most of the steps hold for any log-concave loss function and, in particular, for the quadratic loss, with u W = 2( K + W ) and /lscript W = 2. The only difference is that in Proposition 8.1, we had the negative log-likelihood loss function L and we used the observation that the expected loss vanishes at the true model parameters. Namely, we had ∇L ( µ ) = (1 /n ) ∑ n t =1 ξ t ( µ )˜ x t and we showed that E ( ξ t ( µ 0 )) = 0, from which we derived the high probability bound on ‖∇L ( µ 0 ) ‖ ∞ . (See definition of event F given by (98)).

<!-- formula-not-decoded -->

We show that a similar property holds for the quadratic loss function (30). To see this, recall that in the exploration phases the prices are drawn uniformly at random from the interval [0 , K ]. Therefore, E ( y t | v t ) = P ( v t ≥ p t | v t ) = v t /K . Letting ξ t ( µ ) = 2( Ky t -˜ x t · µ ), we have ∇L ( µ ) = 1 / ( ck ) ∑ n t ∈A k ξ t ( µ )˜ x t and where in the fist step, the inner expectation is with respect to price p t . Given that { (˜ x t , z t ) } t ≥ 1 are independent across t , by applying Azuma-Hoeffding inequality and a union bonding over d coordinates of features, we obtain that

<!-- formula-not-decoded -->

The rest of the proof is similar to the proof of Proposition 8.1 and is omitted.

## C.9 Proof of Lemma 8.6

Recall the notation ‖ E ( k ) ‖ ∞ = max i,j | E ( k ) ij | . Note that

<!-- formula-not-decoded -->

Therefore, we only need to bound ‖ E ( k ) ‖ ∞ . Fix 1 ≤ i, j ≤ d +1. We then have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let u ( ij ) /lscript = ˜ X /lscripti ˜ X /lscriptj . Then, | u ( ij ) /lscript | ≤ 1 because ‖ ˜ x /lscript ‖ ∞ ≤ 1. By applying Hoeffding's inequality,

Therefore, by union bonding over all indices 1 ≤ i, j ≤ d +1, we obtain that ‖ E ( k ) ‖ ∞ ≤ 3 √ (log d ) /τ k , with probability at least 1 -8 /d 2 . The claim follows from this result along with (152).

Acknowledgments Authors are thankful to Arnoud den Boer and Paat Rusmevichientong for their suggestions that improved this work. A. J. would also like to acknowledge the financial support of the Office of the Provost at the University of Southern California through the Zumberge Fund Individual Grant Program. Authors are supported in part by a Google Faculty Research Award.

## References

- [AC09] Victor F Araman and Ren´ e Caldentey, Dynamic pricing for nonperishable products with demand learning , Operations research 57 (2009), no. 5, 1169-1188. 3
- [AD14] Shipra Agrawal and Nikhil R. Devanur, Bandits with concave rewards and convex knapsacks , Proceedings of the Fifteenth ACM Conference on Economics and Computation, EC '14, 2014, pp. 989-1006. 3
- [Air15] Airbnb Documentation, Smart pricing: Set prices based on demand , https://www.airbnb.com/help/article/1168/smart-pricing--set-prices-based-on-demand , 2015. 2
- [ARS14] Kareem Amin, Afshin Rostamizadeh, and Umar Syed, Repeated contextual auctions with strategic buyers , Advances in Neural Information Processing Systems, 2014, pp. 622-630. 1, 3

- [AYPS12] Yasin Abbasi-Yadkori, David Pal, and Csaba Szepesvari, Online-to-confidence-set conversions and application to sparse stochastic bandits. , AISTATS, 2012, pp. 1-9. 10
- [BB05] Mark Bagnoli and Ted Bergstrom, Log-concave probability and its applications , Economic theory 26 (2005), no. 2, 445-469. 5
- [BB16] Hamsa Bastani and Mohsen Bayati, Online decision-making with high-dimensional covariates , Working Paper, 2016. 3
- [BDKS12] Moshe Babaioff, Shaddin Dughmi, Robert Kleinberg, and Aleksandrs Slivkins, Dynamic pricing with limited supply , Proceedings of the 13th ACM Conference on Electronic Commerce, EC '12, 2012, pp. 74-91. 3
- [BFMM14] Santiago R Balseiro, Jon Feldman, Vahab Mirrokni, and S Muthukrishnan, Yield optimization of display advertising with ad exchange , Management Science 60 (2014), no. 12, 2886-2907. 16
- [BJ15] Sonia A Bhaskar and Adel Javanmard, 1-bit matrix completion under exact low-rank constraint , Information Sciences and Systems (CISS), 2015 49th Annual Conference on, IEEE, 2015, pp. 1-6. 4
- [BKS13] Ashwinkumar Badanidiyuru, Robert Kleinberg, and Aleksandrs Slivkins, Bandits with knapsacks , Foundations of Computer Science (FOCS), 2013 IEEE 54th Annual Symposium on, IEEE, 2013, pp. 207-216. 1, 3
- [BPC + 11] Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein, Distributed optimization and statistical learning via the alternating direction method of multipliers , Foundations and Trends R © in Machine Learning 3 (2011), no. 1, 1-122. 7
- [BR12] Josef Broder and Paat Rusmevichientong, Dynamic pricing under a general parametric choice model , Operations Research 60 (2012), no. 4, 965-980. 1, 3, 10, 11
- [BV04] Stephen Boyd and Lieven Vandenberghe, Convex optimization , Cambridge university press, 2004. 5
- [BvdG11] Peter B¨ uhlmann and Sara van de Geer, Statistics for high-dimensional data , SpringerVerlag, 2011. 12, 32, 34, 35
- [BZ09] Omar Besbes and Assaf Zeevi, Dynamic pricing without knowing the demand function: risk bounds and near-optimal algorithms , Operations Research 57 (2009), 1407-1420. 1, 3
- [CLPL16] Maxime C Cohen, Ilan Lobel, and Renato Paes Leme, Feature-based dynamic pricing , ACM Conference on Economics and Computation (2016). 1, 3
- [COPSL15] Xi Chen, Zachary Owen, Clark Pixton, and David Simchi-Levi, A statistical learning approach to personalization in revenue management , Working Paper, 2015. 31
- [CT07] Emmanuel Candes and Terence Tao, The dantzig selector: Statistical estimation when p is much larger than n , The Annals of Statistics (2007), 2313-2351. 12

- [dB15] Arnoud V den Boer, Dynamic pricing and learning: historical origins, current research, and new directions , Surveys in operations research and management science 20 (2015), no. 1, 1-18. 3
- [dBZ13] Arnoud V den Boer and Bert Zwart, Simultaneously learning and optimizing using controlled variance pricing , Management Science 60 (2013), no. 3, 770-783. 3
- [dBZ14] A. V. den Boer and A. P. Zwart, Mean square convergence rates for maximum(quasi) likelihood estimation , Stochastic systems 4 (2014), 1 - 29. 1, 3
- [DHK08] Varsha Dani, Thomas P Hayes, and Sham M Kakade, Stochastic linear optimization under bandit feedback. , COLT, 2008, pp. 355-366. 10
- [EOS07] Benjamin Edelman, Michael Ostrovsky, and Michael Schwarz, Internet advertising and the generalized second-price auction: Selling billions of dollars worth of keywords , The American economic review 97 (2007), no. 1, 242-259. 16
- [FJS13] Vivek F Farias, Srikanth Jagabathula, and Devavrat Shah, A nonparametric approach to modeling choice with limited data , Management Science 59 (2013), no. 2, 305-322. 31
- [FVR10] Vivek F Farias and Benjamin Van Roy, Dynamic pricing with a prior on market response , Operations Research 58 (2010), no. 1, 16-29. 3
- [GNR14] Negin Golrezaei, Hamid Nazerzadeh, and Paat Rusmevichientong, Real-time optimization of personalized assortments , Management Science 60 (2014), no. 6, 1532-1551. 31
- [GZ13] Alexander Goldenshluger and Assaf Zeevi, A linear response bandit problem , Stochastic Systems 3 (2013), no. 1, 230-261. 3
- [HKZ12] J Michael Harrison, Bora Keskin, and Assaf Zeevi, Bayesian dynamic pricing policies: Learning and earning under a binary prior distribution , Management Science 58 (2012), no. 3, 570-586. 3
- [Jav17] Adel Javanmard, Perishability of data: Dynamic pricing under varying-coefficient models , Journal of Machine Learning Research 18 (2017), no. 53, 1-31. 7
- [Kes14] Bora Keskin, Optimal dynamic pricing with demand model uncertainty: A squaredcoefficient-of-variation rule for learning and earning , Working Paper, 2014. 3
- [KL03] Robert Kleinberg and Tom Leighton, The value of knowing a demand curve: Bounds on regret for online posted-price auctions , Proceedings of 44th Annual IEEE Symposium on Foundations of Computer Science, IEEE, 2003, pp. 594-605. 1, 3
- [KR99] Godfrey Keller and Sven Rady, Optimal experimentation in a changing environment , The review of economic studies 66 (1999), no. 3, 475-507. 3
- [KU16] Nathan Kallus and Madeleine Udell, Dynamic assortment personalization in high dimensions , Working Paper, 2016. 31

- [KZ14] Bora Keskin and Assaf Zeevi, Dynamic pricing with an unknown demand model: Asymptotically optimal semi-myopic policies , Operations Research 62 (2014), no. 5, 1142-1167. 1, 3, 4
- [KZ16] N Bora Keskin and Assaf Zeevi, Chasing demand: Learning and earning in a changing environment , Mathematics of Operations Research (2016). 7
- [LP07] S´ ebastien Lahaie and David M Pennock, Revenue analysis of a family of ranking rules for keyword auctions , Proceedings of the 8th ACM conference on Electronic commerce, ACM, 2007, pp. 50-56. 16
- [Mye81] Roger B. Myerson, Optimal auction design , Mathematics of Operations Research 6 (1981), no. 1, 58-73. 6
- [Pon87] Stanislav P Ponomarev, Submersions and preimages of sets of measure zero , Siberian Mathematical Journal 28 (1987), no. 1, 153-163. 41
- [PV13] Yaniv Plan and Roman Vershynin, One-bit compressed sensing by linear programming , Communications on Pure and Applied Mathematics 66 (2013), no. 8, 1275-1297. 4
- [QB16] Sheng Qiang and Mohsen Bayati, Dynamic pricing with demand covariates , Working Paper, 2016. 3
- [Rot74] Michael Rothschild, A two-armed bandit theory of market pricing , Journal of Economic Theory 9 (1974), no. 2, 185-202. 3
- [RWY11] Garvesh Raskutti, Martin J Wainwright, and Bin Yu, Minimax rates of estimation for high-dimensional linear regression over-balls , IEEE Transactions on Information Theory 57 (2011), no. 10, 6976-6994. 40
- [RZ13] Mark Rudelson and Shuheng Zhou, Reconstruction from anisotropic random measurements , IEEE Trans. on Inform. Theory 59 (2013), no. 6, 3434-3447. 32
- [Tsy08] A.B. Tsybakov, Introduction to nonparametric estimation , Springer Series in Statistics, Springer New York, 2008. 13
- [VdG08] Sara A Van de Geer, High-dimensional generalized linear models and the lasso , The Annals of Statistics (2008), 614-645. 4
- [Ver10] Roman Vershynin, Introduction to the non-asymptotic analysis of random matrices , arXiv preprint arXiv:1011.3027 (2010). 35
- [WDY14] Zizhuo Wang, Shiming Deng, and Yinyu Ye, Close the gaps: A learning-while-doing algorithm for single-product revenue management problems , Operations Research 62 (2014), no. 2, 318-331. 1, 3
- [XYL09] Baichun Xiao, Wei Yang, and Jun Li, Optimal reserve price for the generalized secondprice auction in sponsored search advertising , Journal of Electronic Commerce Research 10 (2009), no. 3, 114. 16

- [YB99] Yuhong Yang and Andrew Barron, Information-theoretic determination of minimax rates of convergence , Annals of Statistics (1999), 1564-1599. 39
- [Yu97] Bin Yu, Assouad, fano and le, cam , Research Papers in Probability and Statistics: Festschrift in Honor of Lucien Le Cam, Springer-Verlag, 1997, pp. 423-435. 39