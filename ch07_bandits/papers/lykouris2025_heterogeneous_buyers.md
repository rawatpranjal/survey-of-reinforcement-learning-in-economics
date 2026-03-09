## Contextual Dynamic Pricing with Heterogeneous Buyers

Thodoris Lykouris 1 , Sloan Nietert 2 , Princewill Okoroafor 3 , Chara Podimata 1 , and Julian Zimmert 4

1

MIT, {lykouris, podimata}@mit.edu 2 EPFL, sloan.nietert@epfl.ch 3 Harvard, pco9@cornell.edu 4 Google, zimmert@google.com

December 11, 2025

## Abstract

We initiate the study of contextual dynamic pricing with a heterogeneous population of buyers, where a seller repeatedly posts prices (over T rounds) that depend on the observable d -dimensional context and receives binary purchase feedback. Unlike prior work assuming homogeneous buyer types, in our setting the buyer's valuation type is drawn from an unknown distribution with finite support size K ⋆ . We develop a contextual pricing algorithm based on optimistic posterior sampling with regret ˜ O ( K ⋆ √ dT ) , which we prove to be tight in d and T up to logarithmic terms. Finally, we refine our analysis for the non-contextual pricing case, proposing a variance-aware zooming algorithm that achieves the optimal dependence on K ⋆ .

## 1 Introduction

In online learning for contextual pricing, a learner (aka seller) repeatedly sets prices for different products with the goal of maximizing revenue through interactions with agents (aka buyers or customers). Concretely, in each round t = 1 , . . . , T , nature selects a product with a d -dimensional feature representation u t (context) and the seller selects a price p t ≥ 0 . In the simplest variant, the linear valuation model , customers have a fixed intrinsic valuation model (type) that is unknown to the learner; this has a d -dimensional representation θ ⋆ whose coordinates reflect the valuation that each product feature adds, i.e., the customer's valuation is v t = ⟨ θ ⋆ , u t ⟩ + ε t where ε t is a noise term. The customer makes a purchase only when their valuation is higher than the price, i.e., v t ≥ p t . The learner's goal is to maximize revenue, i.e., the sum of the prices in rounds when purchases occur. An equivalent objective is to minimize regret , which is measured against a benchmark that always selects the customer's valuation as the price for the given round.

Before outlining our contributions, we highlight the unique challenge of online learning in contextual pricing. One key difficulty is that the learner faces both an infinite action space (i.e., all possible prices) and a discontinuous revenue function; indeed, even small price increases can deter a buyer from purchasing, hence causing sharp revenue loss for the learner. However, the problem offers a richer feedback structure than classical multi-armed bandits: a non-purchase indicates that all higher prices would also be rejected by the buyer, while a purchase confirms that all lower prices would be accepted too. The two primary approaches from the literature to tackle this problem

involve estimating the unknown parameter θ ⋆ through online regression or multi-dimensional binary search (see Section 1.2 for further discussion).

Acrucial limitation for both approaches is that they require all customers to behave homogeneously according to a single type θ ⋆ ; see the related work section for results robust to small deviations from this assumption. Moving beyond this homogeneity assumption, we pose the following question:

How can one design contextual pricing algorithms with a heterogeneous population of customers?

## 1.1 Our Contribution

Our setting (Section 2). To study contextual pricing with a heterogeneous buyer population, we assume that the type θ t in round t is drawn from a fixed, unknown distribution D ⋆ . When D ⋆ is supported on a single type θ ⋆ , we recover the homogeneous setting. In our setup, the number of distinct buyer types K ⋆ = | supp D ⋆ | reflects the degree of heterogeneity . We assume that K ⋆ &gt; 1 throughout the paper.

There are several obstacles to applying existing algorithms from the literature. First, canonical contextual pricing algorithms based on regression either compete against (simple) linear policies or assume context-independent and identically distributed (i.i.d.) valuation noise. In contrast, the optimal policy in our setting may best respond based on a context-dependent type rather than a fixed type, and the stochasticity due to heterogeneity is inherently context-dependent and thus non-i.i.d. Second, given that the buyer types are not observable , one cannot connect the observed feedback to shrinkage of type-dependent uncertainty sets; this rules out running canonical multi-dimensional binary search / contextual pricing algorithms for each buyer type in parallel. Third, since in our setting there is a continuum of actions , any canonical contextual bandits algorithm whose regret scales with the discretized action count (e.g., EXP4) will suffer suboptimal performance.

Our contextual pricing algorithm (Section 3). To tackle the above challenges, we employ recent advances in the contextual bandit literature that attain a better scaling with the number of actions, thus evading the shortcomings of EXP4 with naïve discretization. In particular, we build on the optimistic posterior sampling (OPS) approach (Zhang, 2022) which, in our setting, maintains a posterior µ t over all candidate type distributions. We call these candidate type distributions models and refer to their (possibly infinite) family as D . At a high level, in every round, OPS best responds to a model sampled from µ t . As typical in online learning, the posterior update penalizes models that disagree with the observed feedback ( model mismatch ) aiming to converge to the model D ⋆ . To encourage exploration in the absence of full information, this penalty is reduced by an optimism bias term that rewards models with the highest potential to positively contribute to the revenue. The OPS approach enables regret bounds of √ T · c · log |D| , scaling with a disagreement coefficient c that measures the per-context structural complexity of the reward functions and captures the tension between exploration and exploitation. This coefficient is always bounded by the number of actions but can be much smaller in general.

Our main technical contributions in adapting OPS to heterogeneous contextual pricing are twofold. First, to bound the disagreement coefficient c , we observe that, for any fixed context, the aggregate demand function induced by D ⋆ has at most K ⋆ 'jumps', 1 thus creating K ⋆ + 1 intervals. Over each interval, we bound the disagreement coefficient by a factor of 2 . Combining these arguments with a novel decomposition lemma for the disagreement coefficient of functions with K ⋆ breakpoints, we show that c ≤ 2( K ⋆ + 1) . When K ⋆ is known, we apply a variant of OPS over a finite covering of the class D containing all possible distributions over K ⋆ types, of log

1 Each 'jump' corresponds to a change of type from (say) type i to type i +1 .

cardinality dK ⋆ log T . Second, to extend our sublinear regret guarantee to the infinite model class D , we modify OPS to conservatively perturb its recommended prices (which cannot overly impact regret due to one-sided Lipschitzness of the revenue function). We then construct a coupling between the actual trajectory of OPS and one where D ⋆ belongs to the finite cover, allowing us to transfer regret bounds. Finally, we adapt to unknown K ⋆ by initializing OPS with a non-uniform prior over models. These technical contributions enable us to show a regret guarantee of ˜ O ( K ⋆ √ dT ) . Finally, we show that this guarantee is optimal (up to logarithmic terms) with respect to the dependence on both the contextual dimension d and the time horizon T , establishing a lower bound of Ω( √ K ⋆ dT ) for sufficiently large T = Ω( dK 3 ⋆ ) .

Non-contextual improvements (Section 4). The above upper and lower bounds raise a natural question on the optimal dependence of the regret on the number of buyer types K ⋆ ; we resolve this question in the non-contextual version of the problem ( d = 1 ) by providing an algorithm with an upper bound of ˜ O ( √ K ⋆ T ) . Our algorithm, ZoomV , combines zooming (i.e., adaptive discretization) methods from Lipschitz bandits (Kleinberg et al., 2008) with variance-aware confidence intervals (Audibert et al., 2009). Our analysis shows that the regret of ZoomV scales with a novel variance-aware zooming dimension that can be significantly smaller than the standard measure of complexity for Lipschitz bandits. For pricing, this variance adaptation unlocks our ˜ O (min { √ K ⋆ T, T 2 / 3 } ) bound (versus O ( T 2 / 3 ) , obtained via the standard zooming analysis).

The non-contextual version of pricing for heterogeneous buyers was previously studied by CesaBianchi et al. (2019), who establish a matching upper and lower bound if all types are 'well-separated' from each other. In independent and concurrent work, Bacchiocchi et al. (2025) remove this assumption and also achieve a regret bound of ˜ O ( √ K ⋆ T ) . Both of these algorithms employ binary search techniques that are specialized to the piecewise-linear revenue function. In contrast, ZoomV is a generic (one-sided) Lipschitz bandits algorithm, and the structure of the revenue function is only used within its analysis, to bound the variance-aware zooming dimension. One feature of this approach is that we also achieve T 2 / 3 regret when K ⋆ ≫ T 1 / 3 , without changing the algorithm.

Stronger type observability (Section 5). Finally, we consider contextual pricing under the assumption that the learner can identify each arriving type, i.e., where the learner observes ex-post information about the sampled type θ t . We analyze two observability models: one where the learner receives a discrete identifier z t ∈ [ K ⋆ ] - under which a computationally efficient pricing algorithm matches the ˜ O ( K ⋆ √ dT ) regret bound of OPS - and another where the full type vector θ t ∈ B d is observed - for which we reduce the dependence on K ⋆ and d to achieve regret ˜ O ( √ min { K ⋆ , d } T ) . These results demonstrate how richer feedback reduces complexity in dynamic pricing.

## 1.2 Related Work

Our work relates closely to three lines of work: (I) contextual pricing/search; (II) non-contextual pricing; and (III) Lipschitz bandits.

(I) Contextual pricing/search. The closest line to our work is contextual pricing/search . In contextual search, there is a repeated interaction between a learner and nature, where the learner is trying to learn a hidden vector θ ⋆ ∈ R d over time while receiving only single-bit feedback. Mathematically, at each round t ∈ [ T ] , the learner receives a (potentially adversarially chosen) context u t ∈ R d and decides to query y t ∈ R . The learner receives feedback σ t = sgn( ⟨ u t , θ ⋆ ⟩-y t ) ∈ {-1 , +1 } and incurs loss ℓ t ( y t , ⟨ u t , θ ⋆ ⟩ ) (Cohen et al., 2020). Notably, the learner does not observe ℓ t ( y t , ⟨ u t , θ ⋆ ⟩ ) ; only the binary feedback σ t . When the loss function ℓ t corresponds to the lost revenue as a result of posting price y t (i.e., the 'pricing loss'), this setting reduces to ours with a homogeneous buyer population, i.e., K ⋆ = 1 . The contextual search literature has also considered two other

loss functions: the symmetric/absolute loss ℓ t ( y t , ⟨ u t , θ ⋆ ⟩ ) := | y t - ⟨ u t , θ ⋆ ⟩| and the ε -ball loss ℓ t ( y t , ⟨ u t , θ ⋆ ⟩ ) = 1 {| y t -⟨ u t , θ ⋆ ⟩| &gt; ε } , which are motivated by settings other than pricing.

The second approach (e.g., Javanmard and Nazerzadeh, 2019; Javanmard, 2017; Fan et al., 2024; Luo et al., 2024) focuses exclusively on pricing settings. This approach uses regression-based algorithms for learning the correct price and needs to assume stochastic noise in the buyers' responses. There have also been works studying other aspects of contextual pricing (e.g., strategic agents (Amin et al., 2014) and unknown noise distribution (Xu and Wang, 2022)). Apart from the methodological differences with our work, both streams of literature focus on a homogeneous agent population and cannot be readily adapted for a heterogeneous population setting.

There have been two approaches in the literature for learning in contextual pricing and contextual search (for homogeneous agents/buyers). The first approach (e.g., Cohen et al., 2020; Lobel et al., 2018; Paes Leme and Schneider, 2022; Liu et al., 2021) employs a version of multidimensional binary search: specifically, the algorithms maintain a 'knowledge set' with all the possible values of θ ⋆ which are 'consistent' with the feedback that nature has given thus far. Similar to traditional binary search, the query point is chosen to be the point that (given the nature's feedback) will eliminate roughly half of the current knowledge set. As the knowledge set shrinks, the learner ends up with a small knowledge set for the possible values of θ ⋆ ; this is enough to guarantee sublinear regret. The series of works in (Cohen et al., 2020; Lobel et al., 2018; Paes Leme and Schneider, 2022; Liu et al., 2021) optimized regret bounds for the three different loss functions (i.e., symmetric, ε -ball, and pricing). The specific algorithms were different at each paper, but they all maintained a 'binary search' flavor. Most of the algorithms employing a multidimensional binary search approach can be 'robustified' to very little noise in the agents' responses; since the learner will irrevocably shrink the knowledge set according to the feedback received from nature, they can only afford very few mistakes.

Moving closer to the heterogeneous agents problem, Krishnamurthy et al. (2023) studied 'corruption-robust' contextual search, where the agent population is mostly homogeneous, except for C = o ( T ) corrupted agent responses. Their regret bounds were subsequently strengthened by Paes Leme et al. (2022), but the latter approach only works for contextual search with absolute and ε -ball loss and does not cover the pricing loss. This model has been also studied with a Lipschitz target function (Zuo, 2024). Learning with corruptions can be seen as a first step towards learning from heterogeneous agents, but the approaches above do not scale appropriately for truly heterogeneous agent populations. In contrast, we focus on fully heterogeneous settings, where we do not constrain the number or the size of the different buyer types.

(II) Non-contextual pricing. The special case of d = 1 , where there is no context to inform decision-making, was introduced by Kleinberg and Leighton (2003), who studied non-contextual dynamic pricing for a homogeneous, stochastic, and adversarial buyer population. For the adversarial buyer population, the authors assumed that there can be T different valuations and showed tight regret bounds of ˜ O ( T 2 / 3 ) . In contrast, in our setting, we assume that users are 'clustered' in K ⋆ types), and so the lower bound of Ω( T 2 / 3 ) of Kleinberg and Leighton (2003) does not apply.

The closest to our work is the work of Cesa-Bianchi et al. (2019), who consider pricing a heterogeneous agent population with an unknown number of types, but the types are still limited to be less than o ( T ) . Throughout the paper, we discuss how their bounds relate to ours for the special case of d = 1 . None of the aforementioned techniques readily generalize to contextual pricing settings.

(III) Lipschitz bandits. Our work is also related to the literature on Lipschitz bandits. Although the pricing loss is not fully Lipschitz, it has recently been observed that it satisfies a one-sided Lipschitzness . This allows us to leverage techniques from adaptive discretization (Kleinberg et al., 2008) to obtain improved bounds for d = 1 . Zooming had previously been applied to pricing (see, e.g., Podimata and Slivkins, 2021), but these algorithms are insufficient for the K ⋆ -types setting.

Indeed, their performance scales with a zooming dimension ZoomDim that is too large here. On the other hand, ZoomV uses variance-aware confidence intervals so that its performance scales with a smaller, variance-aware zooming dimension ZoomDimV . In particular, while ZoomDim can equal 1 for worst-case instances, we show that ZoomDimV is 0 (with a lower-order scaling constant at most K ⋆ ).

Finally, Krishnamurthy et al. (2020) consider contextual bandits with continuous action spaces, which encompasses the setting of this work. Their regret bounds cover the case where, for a fixed context, the expected reward is Lipschitz in the learner's action. Although their analysis can be adapted to the one-sided Lipschitz setting of pricing, their results either require stochastic contexts or incur large regret due to naïve discretization. Even in the stochastic case, their regret bound scales with a policy zooming coefficient that does not appear to admit a useful bound in terms of K ⋆ .

## 2 Setup and Preliminaries

Notation. Let ∥·∥ and ⟨· , ·⟩ denote the Euclidean norm and inner product on R d . Let S d -1 , B d ⊆ R d denote the unit sphere and ball, respectively. Let ∆( S ) denote the set of all probability measures on a measurable set S ⊆ R d , and let supp( D ) denote the support of D ∈ ∆( R d ) . We use ∆ k ( S ) for those D ∈ ∆( S ) with | supp( D ) | ≤ k . For a positive integer m , let [ m ] := { 1 , 2 , . . . , m } .

Problem setup. We consider T rounds of repeated interaction between a seller, a population of buyers, and an adversary. At each round t ∈ [ T ] , the seller posts a price p t ∈ [0 , 1] for an item to be sold and a buyer, sampled from the population, decides whether or not to buy the item based on their valuation v t ∈ [0 , 1] . We denote the indicator of their purchase by y t = 1 { v t ≥ p t } . The valuation of the buyer is determined by two factors: their type θ t , which encodes their intrinsic preferences, and an external context u t , which describes the current item to be sold and any relevant environmental factors. The learner does not know θ t , but they do know u t . We employ a linear valuation model, supposing that θ t and u t lie in d -dimensional spaces Θ ⊆ [0 , 1] d and U ⊆ S d -1 , respectively, and take v t = ⟨ θ t , u t ⟩ . We assume that ⟨ θ, u ⟩ ∈ [0 , 1] for all θ ∈ Θ and u ∈ U . We impose no further assumptions on the contexts, allowing them to be generated (potentially adaptively) by the adversary. On the other hand, we assume that each θ t is sampled independently from a fixed distribution D ⋆ ∈ ∆(Θ) that describes the buyer population, unknown to the seller. All together, the following occur at each round t ∈ [ T ] :

1. the adversary selects a context u t ∈ U ;
2. a buyer arrives with type θ t ∈ Θ sampled independently from D ⋆ , with valuation v t = ⟨ u t , θ t ⟩ ;
3. the seller observes u t and posts price p t ∈ [0 , 1] for the item;
4. the seller observes the purchase decision y t = 1 { v t ≥ p t } and receives revenue p t y t .

Benchmark. The seller's goal is to maximize their cumulative revenue compared to that which they could have achieved with knowledge of D ⋆ . To express this concisely, we introduce some additional notation. Each distribution Q over valuations in [0 , 1] induces the following:

- a demand function dem Q ( p ) := P v ∼ Q [ v ≥ p ] ,
- an expected revenue function rev Q ( p ) := p · dem Q ( p ) ,
- a revenue-maximizing best response br Q := arg max p ∈ [0 , 1] rev Q ( p ) (breaking ties arbitrarily),
- and a gap function gap Q ( p ) := rev Q ( br Q ) -rev Q ( p ) .

Once we restrict to a fixed context u ∈ U , each type θ ∈ Θ induces valuation v = ⟨ u, θ ⟩ . Thus, each type distribution D ∈ ∆(Θ) induces a projected valuation distribution Q = proj ( D,u ) ∈ ∆([0 , 1]) , defined as the law of ⟨ u, θ ⟩ when θ ∼ D . We then set dem D ( p, u ) := dem Q ( p ) , rev D ( p, u ) := rev Q ( p ) ,

br D ( u ) := br Q , and gap D ( p, u ) := gap Q ( p ) , accordingly. We abbreviate a subscript of D ⋆ by ' ⋆ ' alone, writing, e.g., dem ⋆ ( p, u ) and br ⋆ ( u ) . These details are summarized in Table 1.

A seller policy A is a (potentially randomized) map from a history { u τ , p τ , y τ } t -1 τ =1 and the current context u t to a posted price p t . An adversary policy B is a (potentially randomized) map from a history { u τ , θ τ , p τ , y τ } τ ∈ [ t -1] to the next context u t . We then define the seller's pricing regret by

<!-- formula-not-decoded -->

where { u t , p t } t ∈ [ T ] are selected according to A and B . We will omit the policies from the subscript when clear from context. We focus on controlling the pricing regret in expectation, and will say that A satisfies a regret bound f ( T ) if E [ R A , B ( T )] ≤ f ( T ) for all B .

Our guarantees will scale with context dimension d and the degree of heterogeneity , which we quantify via the support size K ⋆ := | supp( D ⋆ ) | (that may be infinite). We do not assume that K ⋆ is known to the seller. Designing an effective seller policy is challenging because D ⋆ , K ⋆ , and the realized buyer types are unknown to the seller, who must carefully balance exploration and exploitation given only the current context and the history of purchase outcomes.

Basic pricing facts. Finally, we provide some basic properties of the pricing problem, with proofs in Section A. Essential for this work is the one-sided Lipschitzness of the expected revenue function. This is a consequence of the monotonicity of demand functions, and it has previously been used to apply techniques from Lipschitz bandits to non-contextual pricing (Podimata and Slivkins, 2021).

Lemma 2.1 (One-sided Lipschitzness) . Fix any distribution Q ∈ ∆([0 , 1]) and let 0 ≤ p &lt; p ′ ≤ 1 . We then have rev Q ( p ′ ) -rev Q ( p ) ≤ dem Q ( p )( p ′ -p ) ≤ p ′ -p .

Throughout this work, we must handle distributional uncertainty over value distributions. To compare two distributions P, Q ∈ P ([0 , 1]) , we employ the Levy metric defined by

<!-- formula-not-decoded -->

This quantity is at most 1 and equals the side length of the largest square which can be inscribed between the graphs of dem P and dem Q (equivalently, the CDFs of P and Q ). For type distributions D,D ′ ∈ P (Θ) , we use the Levy distance between their projected value distributions, taking d L ( D,D ′ ) := sup u ∈ S d -1 d L ( proj ( D,u ) , proj ( D ′ , u )) . We use this metric because, if D and D ′ are close under d L , then there exists a policy which performs well on both of them; this motivates the use of the Lévy metric throughout the dynamic pricing literature (see, e.g., Paes Leme et al., 2023).

Table 1: Summary of main notation.

| Problem parameters (known)                                                                | Problem parameters (known)                                                                              | Instance parameters (unknown)                                      | Instance parameters (unknown)                                                          |
|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| context dimension d context/feature space U ⊆ S d - 1 type/preference space Θ ⊆ [0 , 1] d | context dimension d context/feature space U ⊆ S d - 1 type/preference space Θ ⊆ [0 , 1] d               | type distribution # types                                          | D ⋆ ∈ ∆(Θ) K ⋆ = &#124; supp( D ⋆ ) &#124;                                             |
| Non-contextual primitives w.r.t. valuation dist. Q ∈ ∆([0 , 1])                           | Non-contextual primitives w.r.t. valuation dist. Q ∈ ∆([0 , 1])                                         | Contextual primitives w.r.t. type dist. D ∈ ∆(Θ) and context u ∈ U | Contextual primitives w.r.t. type dist. D ∈ ∆(Θ) and context u ∈ U                     |
| demand revenue gap                                                                        | dem Q ( p ) = P v ∼ Q [ v ≥ p ] rev Q ( p ) = p · dem Q ( p ) gap Q ( p )=max rev Q ( x ) - rev Q ( p ) | projected value dist. rev./dem./gap rev./dem./gap for D ⋆          | proj ( D,u ) ∈ ∆([0 , 1]) f D ( p,u )= f proj ( D,u ) ( p ) f ⋆ ( p,u )= f D ⋆ ( p,u ) |

x

Lemma 2.2 (Pricing implication of Lévy metric bound) . Suppose that D,D ′ ∈ ∆(Θ) satisfy d L ( D,D ′ ) &lt; ε . Then the conservative best-response policy π ( u ) = max { br D ( u ) -ε, 0 } satisfies rev D ( π ( u ) , u ) ≥ rev D ( br D ( u ) , u ) -ε and rev D ′ ( π ( u ) , u ) ≥ rev D ′ ( br D ′ ( u ) , u ) -3 ε for all u ∈ U .

## 3 Contextual Algorithm with Optimal Dependence on d and T

We now develop statistically efficient (albeit computationally inefficient) algorithms for contextual pricing. In Section 3.1, we treat the simpler setting where D ⋆ belongs to a finite model class D and K ⋆ is known. In Section 3.2, we remove these two assumptions, achieving regret ˜ O ( K ⋆ √ dT ) . We also provide a regret lower bound of Ω( √ K ⋆ dT ) , even if K ⋆ is known, thus proving the optimality of our regret bound's dependence on d and T . Omitted proofs appear in Section B.

## 3.1 Warm-Up: Heterogeneous Contextual Pricing with a Finite Model Class

As a warm-up, we consider pricing when D ⋆ belongs to a known, finite model class.

Assumption 3.1. Assume that D ⋆ ∈ D , where D ⊆ ∆(Θ) is finite and known to the seller.

This realizability assumption simplifies our analysis, and the resulting algorithm extends naturally to infinite classes. We employ optimistic posterior sampling ( OPS ), originally studied for contextual bandits by Zhang (2022) under the name 'Feel-Good Thompson Sampling.' In our instantiation for contextual pricing, OPS (Algorithm 1) maintains a posterior distribution over models, initialized at prior µ 1 ∈ ∆( D ) . At round t with context u t ∈ U , we sample a model D t ∼ µ t , play the best-response price p t = br D t ( u t ) , and observe purchase feedback y t . Then, for each candidate model D ∈ D , we update its posterior weight µ t +1 ( D ) according to the loss ℓ λ ( proj ( D,u t ) , p t , y t ) , defined by

<!-- formula-not-decoded -->

The 'model mismatch' penalty captures the extent to which the observed demand y t differs from that predicted by D when p t is played. In particular, as a function of D ∈ D , the expected model mismatch E y t [( y t -dem D ( p t , u t )) 2 | p t ] is minimized only by those models D which make the same prediction as D ⋆ , i.e., those for which dem D ( p t , u t ) = dem ⋆ ( p t , u t ) . On the other hand, the 'optimism bias' reduces the loss for models which have the potential to provide large revenue, ensuring that we perform sufficient exploration.

## ALGORITHM 1: OPS : Contextual Pricing with a Finite Model Class

- 1 Input : finite model class D ⊆ ∆(Θ) , support size K ≥ 1 ; 2 initialize uniform prior µ 1 = Unif( D ) and optimism strength λ = √ log( |D| ) /KT ; 3 for each round t ∈ [ T ] do 4 observe context u t ; 5 sample model D t ∼ µ t ; 6 play p t = br D t ( u t ) and observe y t ; 7 update µ t +1 ( D ) ∝ µ t ( D ) exp ( -ℓ λ ( proj ( D,u t ) , p t , y t ) ) for each D ∈ D ;

Theorem 3.2. Under Assumption 3.1, OPS with K = K ⋆ achieves regret ˜ O ( √ K ⋆ T log |D| ) .

The requirement of known K ⋆ is imposed for simplicity and will be removed in Section 3.2. To prove Theorem 3.2, we employ a disagreement coefficient that controls the per-context complexity

of balancing exploration and exploitation. In general, for an arbitrary measurable space X and function class F : X → R , we define the disagreement coefficient of F by

<!-- formula-not-decoded -->

In our setting, each f will measure the discrepancy between the demand function predicted by some model D and that of the true model D ⋆ , for a fixed context (full details will appear shortly). While not the most primitive complexity measure, variants of this quantity have been successfully used to analyze a wide variety of structured bandits and active learning problems (see Remark 3.7). The δ 2 /ε 2 scaling was historically chosen so that dis ( F ) can be directly bounded by the domain size |X| . For our application, X = [0 , 1] is the (infinite) price set and each function f ∈ F , induced by a model D ∈ D and context u ∈ U , measures the discrepancy between the demand functions of D and D ⋆ after projection onto u , i.e., f ( p ) = dem D ( p, u ) -dem ⋆ ( p, u ) . In particular, we set

<!-- formula-not-decoded -->

By this definition, if dis ( D , D ⋆ ) is small and the seller plays price p ∼ ν when faced with context u , it is unlikely for a model D to disagree with D ⋆ at p if it is close to D ⋆ under the L 2 ( ν ) norm, i.e., if E q ∼ ν [( dem D ( q, u ) -dem ⋆ ( q, u )) 2 ] is small. In particular, playing p ∼ ν guarantees that

<!-- formula-not-decoded -->

with probability at least 1 -ε 2 δ 2 dis ( D , D ⋆ ) . Since OPS penalizes models with substantial least squares loss, while incentivizing exploration via its optimism bonus, we are able to show the following.

Lemma 3.3. Under Assumption 3.1 with optimism strength λ &gt; 0 , OPS (Algorithm 1) achieves regret ˜ O ( λ dis ( D , D ⋆ ) T +log( |D| ) /λ ) .

The proof in Section B.3 combines the OPS analysis of Zhang (2022) with a decoupling lemma due to Foster et al. (2021b). To control dis ( D , D ⋆ ) , we show that each function of the form dem D ( · , u ) -dem ⋆ ( · , u ) can be decomposed into K ⋆ +1 non-increasing pieces. In Section B.4, we prove the following disagreement coefficient bound for non-increasing functions.

Lemma 3.4. Let F : [0 , 1] → R be the set of nonincreasing functions. Then dis ( F ) ≤ 2 .

Next, we examine the useful notion of composite function classes. A function class G : Z → R is called an N -composite of F : X → R if there exists a disjoint partition Z = Z 1 ∪ · · · ∪ Z N and mappings { h i : Z i →X} i ∈ [ N ] such that each g ∈ G can be decomposed as g ( x ) = f i ( h i ( x )) for all x ∈ Z i and i ∈ [ N ] , for some choice of { f i : X → R } i ∈ [ N ] . We show the following in Section B.5.

Lemma 3.5. If G is an N -composite of F , then dis ( G ) ≤ N dis ( F ) .

With these results in hand, we bound dis ( D , D ⋆ ) = O ( K ⋆ ) and prove the theorem. Even though the action space is infinite, the disagreement coefficient matches that which would arise with K ⋆ actions.

Proof of Theorem 3.2. For each u ∈ U , the function dem ⋆ ( · , u ) is piecewise constant with K ⋆ +1 sections, since jumps can only occur at the projections of the K ⋆ types. For any D ∈ D , the demand dem D ( · , u ) is monotonic, since increasing price always reduces demand. Thus, dem D ( · , u ) -dem ⋆ ( · , u )

is non-increasing on each of the K ⋆ +1 sections, and so the function classes defining dis ( D , D ⋆ ) are ( K ⋆ +1) -composites of the non-increasing function class. Applying Lemmas 3.4 and 3.5 then implies that dis ( D , D ⋆ ) ≤ 2( K ⋆ +1) . The theorem then follows by the regret bound of Lemma 3.3 and our choice of λ .

Remark 3.6 (Comparison to Thompson sampling) . Standard Thompson sampling corresponds to the alternative choice of log losses: ℓ ( Q,p,y ) = log P z ∼ Ber( dem Q ( p )) [ z = y ] = y log dem Q ( p ) + (1 -y ) log(1 -dem Q ( p )) . In comparison, OPS uses the squared loss (this is not essential but simplifies analysis) and an optimism bias towards models under which the seller can attain large revenue. This is crucial for obtaining frequentist (rather than Bayesian) regret bounds, as outlined in Zhang (2022). One appealing aspect of the log loss is that it eliminates models which predict that the observed feedback would never occur, mirroring the elimination-based methods for contextual pricing when K ⋆ = 1 . Thus, one natural question (beyond the current scope) is whether OPS with log loss achieves regret scaling logarithmically in T when K ⋆ = 1 .

Remark 3.7 (Relation with existing results) . Variants of the disagreement coefficient and the related Alexander capacity are well-studied in the active learning and empirical process theory literature (Hanneke, 2014). The version above was first considered by Foster et al. (2021a). Foster et al. (2021b) proved a regret bound which translates to ˜ O ( √ dis ( D ) T log |D| ) in our setting, matching Theorem 3.2. However, the estimation-to-decisions (E2D) meta-algorithm which they employ is non-constructive, hence we apply OPS instead. In Zhang (2022), the regret of OPS is controlled by a distinct 'decoupling coefficient.' Our proof of Lemma 3.3 shows that a (slightly modified) decoupling coefficient is bounded by the disagreement coefficient.

## 3.2 The General Case

We now seek to eliminate the assumptions that D ⋆ belongs to a finite class D and that the support size K ⋆ is known to the seller. For the first point, we loosen the requirement that D ⋆ belongs to D and take D to be a large, finite cover of the full distribution space ∆(Θ) . Then, we replace the uniform prior µ 1 with a non-uniform prior that places less weight on models with large support sizes. Ultimately, this will enable a choice of optimism strength λ that is independent of K ⋆ , achieving our second goal. Unfortunately, if D ⋆ is close but not equal to a model in D , our analysis of OPS fails.

To remedy this, we employ perturbed OPS ( POPS , Algorithm 2), an OPS variant with conservatively perturbed and discretized prices. This modified algorithm and its analysis require some new notation. Given a value distribution Q ∈ ∆([0 , 1]) , define the ε -smoothed demand function dem ε Q by

<!-- formula-not-decoded -->

Similarly, we let rev ε Q ( p ) := p dem ε Q ( p ) . Define contextual extensions dem ε D ( p, u ) and dem ε D ( p, u ) as in the non-smoothed case. For discretization, write P ε := ε N ∩ [0 , 1] for prices which are multiples of ε and let br ε Q := arg max p ∈P ε rev ε ( p ) (lifting to br ε D ( u ) as in the non-smoothed case).

Now, at each round t ∈ [ T ] , POPS samples a model D t ∼ µ t from the current posterior µ t ∈ ∆( D ) and computes its (discretized) best response ˆ p t = br ε D t ( u t ) . Instead of posting price ˆ p t directly, POPS posts p t = max { ˆ p t -δ t , 0 } , where δ t ∼ Unif([0 , ε ]) is a small random perturbation. Due to this perturbation and discretization, we employ the modified loss ℓ ε λ ( proj ( D t , u t ) , ˆ p t , y t ) , where

<!-- formula-not-decoded -->

The perturbations allow us, in the analysis of POPS , to couple its trajectory when run with D ⋆ ̸∈ D to a trajectory where D ⋆ ∈ D . The discretization is needed to bound a modified disagreement coefficient which appears in the analysis. All together, we obtain the following.

## ALGORITHM 2: Perturbed OPS ( POPS ) for Contextual Pricing with Infinite Model Class

```
1 Input : discretization error ε ∈ [0 , 1) , finite model cover D ⊆ ∆(Θ) , model prior µ 1 ∈ ∆( D ) , optimism strength λ > 0 ; 2 for each round t ∈ [ T ] do 3 observe context u t ; 4 sample model D t ∼ µ t and perturbation strength δ t ∼ Unif([0 , ε ]) ; 5 play p t = max { ˆ p t -δ t , 0 } , where ˆ p t = br ε D t ( u t ) and observe y t ; 6 update µ t +1 ( D ) ∝ µ t ( D ) exp ( -ℓ ε λ ( proj ( D,u t ) , ˆ p t , y t ) ) for each D ∈ D ;
```

Theorem 3.8. With appropriately tuned parameters, POPS (Algorithm 2) achieves regret ˜ O ( K ⋆ √ dT ) without prior knowledge of K ⋆ . Moreover, even for known K ⋆ &gt; 1 and stochastic contexts, no contextual pricing policy can achieve expected regret o ( √ K ⋆ dT ) for all instances if T ≥ 8 dK 3 ⋆ log(2 d ) .

Our analysis views the perturbation at Step 5 as being performed by nature, rather than the seller. Treating the seller's action as ˆ p t , they then observe a purchase ( y t = 1 ) with probability

<!-- formula-not-decoded -->

justifying the definitions above. Through this lens, POPS can viewed as OPS for an alternative, smoothed demand model. To bound regret, we apply a three-step argument.

First, we show that POPS maintains our OPS regret bound when D ⋆ ∈ D . This requires bounding a modified decoupling coefficient and is the only step where discretization is used. A direct application of the previous OPS analysis provides a regret bound with respect to a smoothed and discretized benchmark. Fortunately, one-sided Lipschitzness of revenue (Lemma 2.1) ensures that this modified regret benchmark is within O ( εT ) of the original benchmark, as we prove in Section B.6.

Lemma 3.9. Under Assumption 3.1, using prior µ 1 ∈ ∆( D ) , discretization error ε ∈ [0 , 1) , and optimism strength λ &gt; 0 , POPS (Algorithm 2) achieves regret ˜ O ( λK ⋆ T +log(1 /µ 1 ( D ⋆ )) /λ + εT ) .

Next, we show that, if there exists D ∈ D whose smoothed demand function uniformly approximates that of D ⋆ , then the trajectory of POPS under D ⋆ can be coupled with that under D , such that the trajectories coincide with high probability. See Section B.7 for the proof.

Lemma 3.10. If there exists D ∈ D for which ∥ dem ε D -dem ε D ⋆ ∥ ∞ ≤ ε , then the trajectory { u t , ˆ p t , y t } T t =1 of POPS with type distribution D ⋆ can be coupled with that { u ′ t , ˆ p ′ t , y ′ t } T t =1 of POPS with type distribution D , such that the two trajectories are identical with probability 1 -εT .

Finally, we show that, to obtain a uniform ε -cover of the smoothed demand functions, it suffices to find a O ( ε 2 ) -cover of the type distributions under the Lévy metric d L , as defined in (1). Moreover, we show that the family of all type distributions with support size at most K , ∆ K (Θ) , admits an appropriately small Lévy cover. For notation, we write N ( X, d , τ ) for the size of the smallest subset X ′ ⊆ X which covers set X under metric d up to accuracy τ (i.e., for each x ∈ X , there exists x ′ ∈ X ′ such that d( x, x ′ ) ≤ τ ). A proof of the following appears in Section B.8.

Lemma 3.11. If D,D ′ ∈ ∆(Θ) satisfy d L ( D,D ′ ) ≤ ε 2 / 2 , then ∥ dem ε D -dem ε D ′ ∥ ∞ ≤ ε . Moreover, we have log N ( ∆ K (Θ) , d L , ε ) = ˜ O ( dK log 1 /ε ) .

In Section B.9, we combine these lemmas to prove the upper bound of Theorem 3.8. For the lower bound in Section B.10, we modify a construction from Cesa-Bianchi et al. (2019) for the non-contextual case, so that it can be tensored into d -dimensions without leaking information between orthogonal contexts.

Remark 3.12 (Bayesian analysis) . Consider the Bayesian setting where D ⋆ is sampled from a known prior µ ∈ ∆( D ) (keeping D finite for simplicity). Then, Lemma 3.9 with µ 1 = µ and ε = 0 implies that OPS , with µ 1 set to µ at Step 2, achieves Bayesian regret ˜ O ( λK ⋆ T + E D ⋆ ∼ µ [ -log µ ( D ⋆ )] /λ ) = ˜ O ( λK ⋆ T + H ( µ ) /λ ) , where H is Shannon entropy. For known K ⋆ , λ can be tuned to achieve regret ˜ O (√ K ⋆ TH ( µ ) ) , matching the (non-Bayesian) bound of Theorem 3.2 with H ( µ ) instead of log |D| .

Remark 3.13 (Misspecified/noisy setting) . We note that POPS is inherently robust to small misspecifications and noise. Indeed, if D ⋆ does not have support size K ⋆ but is within Lévy distance δ of the family ∆ K ⋆ (Θ) , then the proof of Theorem 3.8 still goes through if we choose ε ← T -2 + √ δ , incurring an additive regret overhead of √ δT 2 . The same overhead applies (up to logarithmic factors) in the noisy model where valuations are subject to mean-zero δ 2 -sub-Gaussian noise, since the associated convolution can only perturb demands by ˜ O ( δ ) under the Lévy metric. We do not attempt to optimize this overhead but note that one is better off using EXP4 (as below) when δ ≫ 1 / poly( T ) .

Remark 3.14 (Large K ⋆ ) . Our lower bound above can be restated as ˜ Ω(min { √ K ⋆ dT,d 1 / 3 T 2 / 3 } ) . When K ⋆ = ˜ Ω(( T/d ) 1 / 3 ) and the second term is active, POPS no longer has an advantage over EXP4 with naïvely discretized actions. In particular, one can run EXP4 with a policy cover of log cardinality ˜ O ( dε -2 ) and price set { ε, 2 ε, . . . , 1 } , incurring regret overhead εT due to discretization. This gives a regret bound of ˜ O ( √ Td 2 ε -3 + εT ) , which balances out to d 2 / 5 T 4 / 5 after tuning. Characterizing the optimal regret in this large K ⋆ regime is an interesting question beyond the current scope.

## 4 Non-Contextual Refinements via Zooming

Our results from Section 3 leave a key open question: what is the optimal regret dependence on K ⋆ ? We resolve this question for the non-contextual setting, where d = 1 and, without loss of generality, u t ≡ 1 for all t . To do so, we employ adaptive discretization (aka zooming ) methods from Lipschitz bandits (Kleinberg et al., 2008) with novel variance-aware confidence intervals and achieve a regret bound of ˜ O ( √ K ⋆ T ) . Throughout, we label the K ⋆ types supp( D ⋆ ) = { θ (1) &lt; θ (2) &lt; · · · &lt; θ ( K ⋆ ) } and set θ (0) = 0 .

Our algorithm, ZoomV , mirrors standard zooming (Kleinberg et al., 2008) with two key adjustments. First, since revenue is only one-sided Lipschitz, we use a dyadic price selection rule inspired by Podimata and Slivkins (2021). Second, our confidence intervals incorporate empirical variance, a method previously used for variance-aware K -armed bandits (Audibert et al., 2009). In more detail, ZoomV maintains a set S of active prices in [0 , 1] and a variance-aware confidence interval for the expected revenue at each p ∈ S . Each active price 'covers' an interval of neighboring, larger prices, with the width of this covering interval scaling proportionally to that of the confidence interval. The intuition is that a small increase in price can only marginally increase expected revenue, so it is not worth exploring such covered prices. Initially, every price in [0 , 1] is covered by some point in S . At each round, ZoomV optimistically chooses a price in S and updates its confidence and covering intervals. If after the update there exists an uncovered price, then we add a new point to S which covers it, maintaining the invariant that every price is covered.

Theorem 4.1. ZoomV (Algorithm 4) achieves regret ˜ O ( min { √ K ⋆ T, T 2 / 3 }) for non-contextual pricing, without knowledge of K ⋆ . This is minimax optimal up to logarithmic factors when K ⋆ &gt; 1 .

To bound regret, we employ a variance-aware zooming dimension which controls its performance. For comparison, we first recall the definition of the standard zooming dimension, which characterizes a certain complexity of the expected reward function rev ⋆ ( p ) . For each δ &gt; 0 , write X δ := { p ∈ [0 , 1] : gap ⋆ ( p ) ≤ δ } for the set of δ -approximate revenue maximizers. Write N ( X,δ ) := N ( X, | · | , δ ) for the smallest δ -covering of a set X ⊆ R . Then, for each c &gt; 0 , define the zooming dimension

<!-- formula-not-decoded -->

Standard zooming techniques imply that ZoomV achieves regret c 1 / (2+ z ) T 1 -1 / (2+ z ) when ZoomDim ( c ) ≤ z , even with confidence intervals that do not incorporate empirical variance. Since the price interval [0 , 1] is one-dimensional, one can bound ZoomDim ( c ) ≤ 1 for c = O (1) , giving regret ˜ O ( T 2 / 3 ) . Moreover, the set X δ of approximate revenue maximizers is contained in the union of K ⋆ intervals preceding the unknown types, where the interval corresponding to type θ ( i ) has width δ/ dem ⋆ ( θ ( i ) ) ≤ δ/ dem ⋆ ( θ ( K ⋆ ) ) . This implies ZoomDim ( c ) = 0 and gives regret ˜ O ( √ cT ) , but only for c = O ( K ⋆ / dem ⋆ ( θ ( K ⋆ ) )) , which may be arbitrarily large for worst-case instances.

To remedy this, we incorporate variance, writing σ 2 ( p ) = p 2 dem ⋆ ( p )(1 -dem ⋆ ( p )) for the revenue variance when p is played. For the problematic types above with low demand, variance is also low, and the confidence intervals of ZoomV are designed to adapt to this. Specifically, our proof in Section C.2 shows that the regret of ZoomV scales according to a variance-aware zooming dimension , defined as follows. First define the variance-weighted covering number N v ( X,δ ) := inf {∑ x ∈ X ′ σ 2 ( x ) : X ′ is a δ -cover of X } . Then, for each c &gt; 0 , we set

<!-- formula-not-decoded -->

Note that ZoomDimV ( c ) ≤ ZoomDim ( c ) , since σ 2 ( p ) ≤ 1 . Moreover, we show ZoomDim (10 K ⋆ ) = 0 , implying the desired ˜ O ( √ K ⋆ T ) regret. The lower bound follows from Theorem 3.8 with d = 1 .

Remark 4.2 (Comparison to Cesa-Bianchi et al. (2019)) . The non-contextual setting was previously studied by Cesa-Bianchi et al. (2019), whose Algorithm 1 achieves regret ˜ O ( √ K ⋆ T )+ V ( V +1) , where V = max i ∈ [ K ⋆ ] ( θ ( K ⋆ ) ) 4 ( θ ( i ) -θ ( i -1) ) -5 . They maintain a set of intervals which contain all types with substantial probability mass, gradually refining these intervals until they are all of width O ( T -1 / 2 ) , at which point they employ UCB over the intervals' left endpoints. Unfortunately, the instance-dependent term V ( V +1) can blow up to infinity for worst-case realizations of D ⋆ ∈ ∆ K ⋆ ([0 , 1]) , in contrast to our guarantee.

## 5 Improved Performance with Ex-Post Type Observability

We now study dynamic pricing with heterogeneous buyer types under the additional assumption that the learner can identify each arriving type. That is, after setting price p t , the learner observes the purchase feedback 1 {⟨ u t , θ t ⟩ ≥ p t } and some information about the sampled type θ t . We consider the two models of observability. In the first, the learner only observes an identifier z t ∈ [ K ⋆ ] that specifies which of the K ⋆ candidate types was drawn. In practice, the learner need not know K ⋆ a priori . Here, we design an algorithm that matches the ˜ O ( K ⋆ √ dT ) regret bound of POPS and can be implemented efficiently, using a contextual search algorithm for K ⋆ = 1 as a subroutine. In the second model, the learner observes the full type embedding θ t ∈ Θ . Here, we show that best-responding to a simple plug-in estimate for D ⋆ achieves an improved regret bound of ˜ O ( √ min { K ⋆ , d } T ) .

Observed type identifiers. Our algorithm for the model where the learner only observes the identifier uses a K ⋆ = 1 contextual search algorithm, ProjectedVolume (Lobel et al., 2018), as a

subroutine. We maintain a separate instance of this ProjectedVolume algorithm for each observed type and keep track of the empirical type frequencies along with the number of times we've explored each type. It then adaptively chooses which types to explore (or exploit) based on confidence estimates for the type distribution. We present the full algorithm and prove the following regret bound in Appendix D.1.

Theorem 5.1. Consider contextual dynamic pricing with ex-post type observability where the learner observes which type z t ∈ [ K ⋆ ] arrived. Then, Algorithm 5 achieves regret ˜ O ( K ⋆ √ dT ) and takes no more than time poly( K ⋆ , d, T ) per round.

Observed type vectors. If the full type vector θ t is revealed at the end of each round, we can achieve improved regret with a simpler algorithm. Indeed, writing ˆ D t = 1 t ∑ t τ =1 δ θ t for the empirical type distribution after round t , we take each p t as the best response to ˆ D t -1 along the current context.

Theorem 5.2. Consider contextual dynamic pricing with ex-post type observability where the learner observes θ t ∈ Θ at the end of each round. Then the algorithm which plays p 1 = 1 / 2 and best response p t = br ˆ D t -1 ( u t ) for remaining rounds achieves regret ˜ O ( √ min { K ⋆ , d } T ) . Each price can be computed in time poly( K ⋆ , d ) .

The proof in Section D.2 uses VC dimension bounds to show that the empirical revenue function rev ˆ D t converges uniformly in both arguments (price and context) to the true revenue function rev ⋆ .

## 6 Discussion

In this work, we have introduced contextual dynamic pricing with heterogeneous buyers. Our main algorithm achieves a regret bound of ˜ O ( K ⋆ √ dT ) , optimal up to a O ( √ K ⋆ ) factor and logarithmic terms. Our analysis bounds the disagreement coefficient by leveraging a novel decomposition lemma for aggregate demand functions with K ⋆ breakpoints, thereby ensuring an efficient exploration-exploitation tradeoff. Additionally, we propose a variance-aware zooming algorithm for the non-contextual pricing case, improving regret dependence on K ⋆ by incorporating adaptive discretization methods from the Lipschitz bandits literature. Finally, under stronger observability assumptions on the buyers' types, we develop efficient algorithms that significantly reduce regret to ˜ O ( √ min { K ⋆ , d } T ) , demonstrating the potential benefits of richer feedback in dynamic pricing settings.

There are several natural open questions, the first revolving around computation . The run time of POPS scales with the size of the discretized model class, which is exponential in K ⋆ and d . It would be interesting to see if there is a way to alleviate this exponential dependence, while achieving similar regret bounds. The second question is around the optimal dependence on K ⋆ for the general, contextual case. While in Section 4 we showed how to optimize the dependence of our bounds on K ⋆ it is unclear how to scale this approach for the contextual version of the problem. One starting point could be the results on Zooming techniques for contextual bandits (see e.g., Slivkins, 2011). Finally, it would be interesting to see if our results can be applied to more broad families of settings where a learner tries to learn from heterogeneous agents while obtaining only single-bit feedback: for example, it is unknown if the approach presented in this work generalizes to general contextual search settings (i.e., with ε -ball or symmetric loss) or if it generalizes for settings that share some core properties with pricing, but differ in the fundamental techniques used to address them (see e.g., Ho et al., 2014).

Acknowledgements. The authors thank Jason Gaitonde and Oliver Richardson for helpful discussions on high-dimensional probability and algorithm design. We are also grateful to the Simons

Institute for the Theory of Computing, as this work started during the Fall'22 semester-long program on Data-Driven Decision Processes.

## References

- Amin, K., Rostamizadeh, A., and Syed, U. (2014). Repeated contextual auctions with strategic buyers. In Advances in Neural Information Processing Systems .
- Audibert, J.-Y., Munos, R., and Szepesvári, C. (2009). Exploration-exploitation tradeoff using variance estimates in multi-armed bandits. Theoretical Computer Science , 410(19):1876-1902.
- Bacchiocchi, F., Castiglioni, M., Marchesi, A., and Gatti, N. (2025). Regret minimization for piecewise linear rewards: Contracts, auctions, and beyond. In ACM Conference on Economics and Computation (EC) .
- Boucheron, S., Lugosi, G., and Massart, P. (2013). Concentration Inequalities: A Nonasymptotic Theory of Independence . Oxford University Press.
- Cesa-Bianchi, N., Cesari, T., and Perchet, V. (2019). Dynamic pricing with finitely many unknown valuations. In Algorithmic Learning Theory .
- Cohen, M. C., Lobel, I., and Paes Leme, R. (2020). Feature-based dynamic pricing. Management Science , 66(11):4921-4943.
- Fan, J., Guo, Y., and Yu, M. (2024). Policy optimization using semiparametric models for dynamic pricing. Journal of the American Statistical Association , 119(545):552-564.
- Foster, D., Rakhlin, A., Simchi-Levi, D., and Xu, Y. (2021a). Instance-dependent complexity of contextual bandits and reinforcement learning: A disagreement-based perspective. In Conference on Learning Theory (COLT) .
- Foster, D. J., Kakade, S. M., Qian, J., and Rakhlin, A. (2021b). The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487 .
- Hanneke, S. (2014). Theory of disagreement-based active learning. Foundations and Trends® in Machine Learning , 7(2-3):131-309.
- Ho, C.-J., Slivkins, A., and Vaughan, J. W. (2014). Adaptive contract design for crowdsourcing markets: Bandit algorithms for repeated principal-agent problems. In ACM Conference on Economics and Computation (EC) .
- Javanmard, A. (2017). Perishability of data: dynamic pricing under varying-coefficient models. Journal of Machine Learning Research , 18(53):1-31.
- Javanmard, A. and Nazerzadeh, H. (2019). Dynamic pricing in high-dimensions. Journal of Machine Learning Research , 20(9):1-49.
- Kleinberg, R., Slivkins, A., and Upfal, E. (2008). Multi-armed bandits in metric spaces. In ACM Symposium on Theory of Computing (STOC) .
- Kleinberg, R. D. and Leighton, F. T. (2003). The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In Symposium on Foundations of Computer Science (FOCS) .

- Krishnamurthy, A., Langford, J., Slivkins, A., and Zhang, C. (2020). Contextual bandits with continuous actions: Smoothing, zooming, and adapting. Journal of Machine Learning Research , 21(137):1-45.
- Krishnamurthy, A., Lykouris, T., Podimata, C., and Schapire, R. (2023). Contextual search in the presence of adversarial corruptions. Operations Research , 71(4):1120-1135.
- Liu, A., Leme, R. P., and Schneider, J. (2021). Optimal contextual pricing and extensions. In Symposium on Discrete Algorithms (SODA) .
- Lobel, I., Paes Leme, R., and Vladu, A. (2018). Multidimensional binary search for contextual decision-making. Operations Research , 66(5):1346-1361.
- Luo, Y., Sun, W. W., and Liu, Y. (2024). Distribution-free contextual dynamic pricing. Mathematics of Operations Research , 49(1):599-618.
- Maurer, A. and Pontil, M. (2009). Empirical Bernstein bounds and sample variance penalization. In Conference on Learning Theory (COLT) .
- Paes Leme, R., Podimata, C., and Schneider, J. (2022). Corruption-robust contextual search through density updates. In Conference on Learning Theory (COLT) .
- Paes Leme, R. and Schneider, J. (2022). Contextual search via intrinsic volumes. SIAM Journal on Computing , 51(4):1096-1125.
- Paes Leme, R., Sivan, B., Teng, Y., and Worah, P. (2023). Pricing query complexity of revenue maximization. In ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 399-415.
- Podimata, C. and Slivkins, A. (2021). Adaptive discretization for adversarial Lipschitz bandits. In Conference on Learning Theory (COLT) .
- Shalev-Shwartz, S. and Ben-David, S. (2014). Understanding machine learning: From theory to algorithms . Cambridge University Press.
- Slivkins, A. (2011). Contextual bandits with similarity information. In Conference On Learning Theory (COLT) .
- Slivkins, A. et al. (2019). Introduction to multi-armed bandits. Foundations and Trends® in Machine Learning , 12(1-2):1-286.
- Xu, J. and Wang, Y.-X. (2022). Towards agnostic feature-based dynamic pricing: Linear policies vs linear valuation with unknown noise. In International Conference on Artificial Intelligence and Statistics (AISTATS) .
- Zhang, T. (2022). Feel-good Thompson sampling for contextual bandits and reinforcement learning. SIAM Journal on Mathematics of Data Science , 4(2):834-857.
- Zuo, S. (2024). Corruption-robust Lipschitz contextual search. In International Conference on Algorithmic Learning Theory (ALT) .

## A Proofs for Section 2

## A.1 One-Sided Lipschitzness of Revenue (Proof of Lemma 2.1)

We simply bound rev Q ( p ′ ) -rev Q ( p ) = p ′ dem Q ( p ′ ) -p dem Q ( p ) ≤ dem Q ( p )( p ′ -p ) ≤ p ′ -p , using monotonicity of demand functions.

## A.2 Pricing Implication of Lévy Metric Bound (Proof of Lemma 2.2)

By the definition of d L , it suffices to prove the lemma when d = 1 . The first revenue lower bound holds by Lemma 2.1. For the second, omitting the context u and writing [ x ] + = max { x, 0 } , we use the Lévy metric guarantee to bound

<!-- formula-not-decoded -->

as desired.

## B Proofs for Section 3

To unify analysis of Sections 3.1 and 3.2, we introduce a more general problem setup and algorithm.

## B.1 Generalized Problem Setup

To start, we replace the price set [0 , 1] with a subset P ⊆ [0 , 1] , which will remain [0 , 1] in Section 3.1 but will be restricted to a finite set for Section 3.2. Then, instead of selecting a type distribution D ⋆ , we have the adversary choose a demand function f ⋆ which maps price p ∈ P and context u ∈ U to a purchase probability f ⋆ ( p, u ) ∈ [0 , 1] . Then, at round t , if the adversary selects context u t ∈ U and the learner posts price p t ∈ P , purchase decision y t ∈ { 0 , 1 } is sampled independently from Ber( f ⋆ ( p t , u t )) . This abstracts away our previous notions of buyer types and values and will also model the smoothed environment of Section 3.2. We impose the corresponding notion of realizability.

Setting B.1 (realizability, general) . Under the setup described above, the demand function f ⋆ belongs to a known, finite class F of measurable functions from P × U to [0 , 1] .

Often, we shall fix a context and consider univariate (non-contextual) demand functions. Given a univariate demand function g : [0 , 1] → [0 , 1] , we define the corresponding revenue function

## Appendix

rev g ( p ) = p · g ( p ) , best-response br g = arg max p ∈P rev g (breaking ties arbitrarily), and gap gap g ( p ) = rev g ( br g ) -rev g ( p ) . For a contextual demand function f : [0 , 1] ×U → [0 , 1] and a context u ∈ U , write proj ( f, u ) for the induced univariate demand function p ↦→ f ( p, u ) . We define rev f ( p, u ) = rev proj ( f,u ) ( p ) , br f ( u ) = br proj ( f,u ) , gap f ( p, u ) = gap proj ( f,u ) ( p ) , and proj ( F , u ) := { proj ( f, u ) : f ∈ F} , along with rev ⋆ := rev f ⋆ , br ⋆ := br f ⋆ , and gap ⋆ := gap f ⋆ . Finally, we define

<!-- formula-not-decoded -->

generalizing the definition in Section 3.1. Regret is defined as the sum of gaps ∑ T t =1 gap ⋆ ( p t , u t ) .

## B.2 Generalized OPS and its Regret Guarantee

We now present a extension of OPS and POPS to the generalized setup of Section B.1. Both can be recovered for appropriate choices of F and P , which we will discuss later. First, for a univariate demand function g : P → [0 , 1] , price p , and purchase decision y , we define loss

<!-- formula-not-decoded -->

We now adapt OPS to this setting, introducing GOPS (Algorithm 3).

## ALGORITHM 3: GOPS : Generalized OPS for Contextual Pricing with Finite Model Class

- 1 Input : finite demand function class F , model prior µ 1 ∈ ∆( F ) , optimism strength λ &gt; 0 ; 2 for each round t ∈ [ T ] do
- 4 sample demand function f t ∼ µ t ;
- 3 observe context u t ;
- 5 play p t = br f t ( u t ) and observe y t ;
- 6 update µ t +1 ( f ) ∝ µ t ( f ) exp ( -ℓ λ ( proj ( f, u t ) , p t , y t ) ) for each f ∈ F ;

We prove the following regret bound.

Lemma B.2. Under Setting B.1, with model prior µ 1 ∈ ∆( F ) and optimism strength λ ≥ 4 /T , GOPS (Algorithm 3) achieves regret 25 λ ( dis ( F , f ⋆ ) ∨ 1) T log 2 ( T ) + log(1 /µ 1 ( f ⋆ )) /λ .

Our proof employs the following decoupling lemma.

Lemma B.3 (Foster et al., 2021b) . Let G be a finite family of univariate demand functions and fix g ⋆ ∈ G . Then, for any ν ∈ ∆( G ) and γ &gt; 0 , we have

<!-- formula-not-decoded -->

This is simply Lemma E.2 of Foster et al. (2021b) with function class { g -g ⋆ : g ∈ G} and ∆ → 0 . In our proof, G and g ⋆ will be the projections of F and f ⋆ onto a fixed context u ∈ U .

The remainder of our analysis is a slight modification to that of Zhang (2022), which we provide for completeness. For each round t ∈ [ T ] of OPS, we adopt the following notation:

- history up to round t : S t := { u τ , f τ , p τ , y τ } t τ =1 ,
- true univariate demand function: g ⋆ t := proj ( f ⋆ , u t ) ,
- univariate demand function posterior: ν t := proj ( µ t , u t ) := Law f ∼ µ t ( proj ( f, u t )) ,

- sampled univariate demand function: g t := proj ( f t , u t ) , so that p t = br g t ,
- independently sampled univariate demand function (for analysis): ˜ g t ∼ ν t ,
- regret: REG t := rev ⋆ ( br ⋆ ( u t ) , u t ) -rev ⋆ ( p t , u t ) = rev g ⋆ t ( p t ) -rev g ⋆ t ( p t ) ,
- least-squares errors: LS t ( g ) := ( g ( p t ) -g ⋆ ( p t )) 2 ,
- 'feel-good' (optimism) bonuses: FG t ( g ) := rev g ( br g ) -rev g ⋆ t ( br g ⋆ t ) ,
- loss discrepancies: ∆ L t ( g ) := ℓ λ ( g, p t , y t ) -ℓ λ ( g ⋆ t , p t , y t ) ,
- potential function: Z t := E S t log E f ∼ Unif( F ) exp ( -∑ t τ =1 ∆ L t ( proj ( f, u t )) ) .

Our proof requires several supporting lemmas. The first is a basic concentration result.

Lemma B.4. For c ≥ 0 and a random variable X supported on [0 , 1] , we have log E exp( -cX ) ≤ ( 1 2 c 2 -c ) E X . For X supported on [ a, b ] , we have log E exp( cX ) ≤ c E X + 1 8 ( b -a ) 2 c 2 .

Proof. For the first inequality, we bound

<!-- formula-not-decoded -->

The second inequality is exactly Hoeffding's lemma.

The next lemma mirrors Lemma 4 of Zhang (2022). This is a consequence of the definition of ∆ L t and the sub-Gaussianity of its components.

Lemma B.5. For round t of GOPS (Algorithm 3), we have

<!-- formula-not-decoded -->

We note that this lemma does not rely on how p t is selected.

Proof. Let g ∼ ν t and y ∼ Law( y t | u t , p t ) = Ber( g t ( p t )) be independent. Let ε = y -g ⋆ t ( p t ) denote the discrepancy between the observed and expected demand. Since demands lie in [0 , 1] , Lemma B.4 with X = ε gives

<!-- formula-not-decoded -->

Moreover, we have

<!-- formula-not-decoded -->

Combining with (4) gives

Therefore, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining, we have

We then bound

<!-- formula-not-decoded -->

where (a) uses Jensen's inequality and (b) uses Lemma B.4. Rearranging gives the lemma.

Now, we return to Lemma B.2, where we will finally incorporate our price selection rule and the decoupling lemma.

<!-- formula-not-decoded -->

where the last inequality follows by Hölder's inequality. For the first term, we use Lemma B.4 with X = LS t ( g ) and c = 3 4 to bound

<!-- formula-not-decoded -->

For the second term, we apply the lemma with X = FG t ( g ) and c = 3 λ to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

implying the lemma.

Our last helper lemma mirrors Lemma 5 of Zhang (2022).

Lemma B.6. For round t of GOPS (Algorithm 3), we have

<!-- formula-not-decoded -->

This lemma also does not rely on how prices are chosen.

Proof. Recall that µ 1 = Unif( F ) . Defining W t ( f | S t ) := exp ( -∑ t τ =1 ∆ L τ ( proj ( f, u τ )) ) , we have Z t = E S t log E f ∼ µ 1 W t ( f | S t ) . Note that

<!-- formula-not-decoded -->

Proof of Lemma B.2. For each round t ∈ [ T ] , we recall that p t = br g t and decompose

<!-- formula-not-decoded -->

Conditioning on S t -1 and u t , we apply Lemma B.3 with γ = 1 4 λ to obtain

<!-- formula-not-decoded -->

Taking expectations over S t -1 and u t , we bound

<!-- formula-not-decoded -->

using Lemma B.6. Summing over t ∈ [ T ] and noting that Z 0 = 0 , we bound

<!-- formula-not-decoded -->

Moreover, by realizability, we have

<!-- formula-not-decoded -->

Combining, we obtain

<!-- formula-not-decoded -->

as desired.

## B.3 Base Regret Bound for OPS (Proof of Lemma 3.3)

Under the general setup of Section B.1, we take F to be the class of demand functions induced by D , set P = [0 , 1] , and fix µ 1 = Unif( F ) . By these choices, GOPS coincides exactly with OPS , as does our notion of regret. Thus, Lemma B.2 gives the desired regret bound of

<!-- formula-not-decoded -->

## B.4 Disagreement Coefficient Bound for Non-increasing Functions (Proof of Lemma 3.4)

Fixing f ∈ F , ν ∈ ∆([0 , 1]) , and p ∈ [0 , 1] , suppose that E q ∼ ν [ f ( q ) 2 ] ≤ ε 2 and | f ( p ) | &gt; δ . If f ( p ) &gt; δ , then f ( q ) &gt; δ for all q ≤ p by monotonicity. Thus, P q ∼ ν ( q ≤ p ) δ 2 ≤ E q ∼ ν [ f ( q ) 2 ] ≤ ε 2 . Otherwise, if f ( p ) &lt; -δ , we analogously have P q ∼ ν ( q ≥ p ) δ 2 ≤ E q ∼ ν [ f ( q ) 2 ] ≤ ε 2 . Thus, for ν ∈ ∆( X ) , we have

<!-- formula-not-decoded -->

Plugging this into the definition of dis finishes the proof.

## B.5 Disagreement Coefficient Bound for Composite Classes (Proof of Lemma 3.5)

For any distribution ν ∈ ∆( Z ) , write ν i = h i ◦ ν | Z i for law of h i ( p ) when p ∼ ν , conditioned on p ∈ Z i , and let µ ν ( i ) = P p ∼ ν ( p ∈ Z i ) . We then bound

<!-- formula-not-decoded -->

as desired. Here, the first inequality uses that E q ∼ ν [ g ( q ) 2 ] ≥ µ ν ( i ) E q ∼ ν i [ f ( h i ( x )] for some f ∈ F , and the second uses that ν i ∈ ∆( X ) and the definition of dis .

## B.6 Base Regret Bound for POPS (Proof of Lemma 3.9)

POPS as generalized OPS. We observe that POPS is an instance of GOPS (Algorithm 3), with discretized price set P = P ε and smoothed demand function class F = { dem ε D : D ∈ D} (where each f ∈ F is viewed as a function on P ε ×U rather than [0 , 1] ×U ). Here, we view each ˆ p t as the posted price instead of p t . Indeed, taking f ⋆ = dem ε ⋆ , we have

<!-- formula-not-decoded -->

and, for value distribution Q ∈ ∆([0 , 1]) with smoothed demand function g = dem ε Q , we have

<!-- formula-not-decoded -->

where ℓ λ on the right hand size is defined in (3). We do note, however, that the regret benchmark with smoothed demands and discretized prices differs slightly from the original benchmark.

Fixing the regret benchmark. Applying Lemma B.2 for this choice of P and F , we have that

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

so the regret bound above measures cumulative revenue in line with our original regret definition. Although the benchmark does not match that in the original definition, we have | rev ε ⋆ ( br ε ⋆ ( u t ) , u t ) -rev ⋆ ( br ⋆ ( u t ) , u t ) | = O ( ε ) for all t ∈ [ T ] due to one-sided Lipschitzness (Lemma 2.1). Consequently, the regret of POPS is bounded by

<!-- formula-not-decoded -->

It remains to bound the disagreement coefficient by O ( K ⋆ ) , giving the lemma. Our argument below mirrors that in the proof of Theorem 3.2 but takes into account the smoothing and discretization.

Bounded disagreement coefficient. For each u ∈ U , the function dem ε ⋆ ( · , u ) with domain P ε is piecewise constant with O ( K ⋆ ) sections. Indeed, the unsmoothed demand function dem ⋆ ( · , u ) is piecewise constant with O ( K ⋆ ) sections, and smoothing can only introduce new sections at the O ( K ⋆ ) prices in P ε that are within distance ε of a previous section boundary. Moreover, smoothing preserves monotonicity of demand functions. Hence, the function classes defining dis ( F , f ⋆ ) are O ( K ⋆ ) -composites of the nonincreasing function class. Applying Lemmas 3.4 and 3.5 then implies that dis ( F , f ⋆ ) = O ( K ⋆ ) , giving the lemma.

## B.7 Trajectory Coupling (Proof of Lemma 3.10)

Fix any round t with context u t and best-response price ˆ p t . Since ∥ f ε D -f ε D ⋆ ∥ ∞ ≤ ε , feedback y t coincides with that which would have been obtained if D ⋆ = D with probability at least 1 -ε , conditioned on u t and ˆ p t . Since the update to µ t is only a function of u t , ˆ p t , and y t (notably, not the realized price p t ), we can iterate through all rounds and apply a union bound to obtain the lemma.

## B.8 Metric Entropy Bound (Proof of Lemma 3.11)

For part one, fix D,D ′ ∈ ∆(Θ) with d L ( D,D ′ ) ≤ ε 2 / 2 . Then, for all u ∈ U and ˆ p ∈ [0 , 1] , we have

<!-- formula-not-decoded -->

Note that the maximum is unneeded since proj ( D,u ) is supported on [0 , 1] and places no mass on negative values. Writing Q = proj ( D,u ) and Q ′ = proj ( D ′ , u ) , we then have

<!-- formula-not-decoded -->

Writing τ = ε 2 / 2 , we further have

<!-- formula-not-decoded -->

where the last step uses the fact that d L ( Q,Q ′ ) ≤ τ . A symmetric argument gives the reverse bound. Consequently, we have | f ε D (ˆ p, u ) -f ε D ′ (ˆ p, u ) | ≤ 1 ε · 2 τ = ε , as desired.

For part two, let C = { C 1 , C 2 , . . . , C n } denote the intersection of the standard partition of R d into cubes of side length ε/ √ d with Θ ⊆ [0 , 1] d . Denote the lexicographically smallest vertex of each C i by c i , and note that log( n ) = O ( d log( d/ε )) . Given any D ∈ ∆ K (Θ) , we define the initial discretization ˆ D 0 = ∑ n i =1 D ( C i ) δ c i . We obtain the final discretized measure ˆ D by rounding each weight to a neighboring multiple of ε/K (choice doesn't matter so long as we maintain unit mass, this is always possible). Then, for any context u ∈ S d -1 and price p ∈ [0 , 1] , we have

<!-- formula-not-decoded -->

using that each cube in C has diameter ε and that the mass in each cube was perturbed by at most ε/K (so with K cubes the probability of any event is shifted by at most ε ). Thus, d L ( D, ˆ D ) ≤ ε . Using balls and bins, we can thus bound

<!-- formula-not-decoded -->

## B.9 Proof of Theorem 3.8, Upper Bound

Upper bound. Fix ε = T -2 and λ = √ d/T . We now construct D and µ 1 . Write M = ⌈ log T ⌉ , and, for i = 1 , . . . , M , take D i to be a minimal ( ε 2 / 2) -cover of ∆ 2 i (Θ) under the Lévy metric. By Lemma 3.11, we have log |D i | = ˜ O (2 i d log 1 /ε ) . Now set D = D 1 ∪ · · · ∪ D M and take µ 1 ( D ) ∝ (2 i |D i | ) -1 for D ∈ D i . This ensures that log(1 /µ 1 ( D )) = ˜ O (2 i d log T ) for D ∈ D i .

Assume without loss of generality that M ≥ log K ⋆ ; otherwise, the regret bound is vacuous. Then, there exists ˆ D ∈ D ⌈ log K ⋆ ⌉ ⊆ D such that d L ( ˆ D,D ⋆ ) ≤ ε 2 / 2 and ∥ dem ε D -dem ε D ⋆ ∥ ∞ ≤ ε , again using Lemma 3.11. Thus, by Lemma 3.10, the realized trajectory of POPS { u t , ˆ p t , p t , y t } T t =1 can be coupled with an alternative trajectory { u ′ t , ˆ p ′ t , p ′ t , y ′ t } T t =1 of POPS with type distribution ˆ D , such that { u t , ˆ p t , y t } T t =1 = { u ′ t , ˆ p ′ t , y ′ t } T t =1 with probability at least 1 -εT . By Lemma 3.9, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At this point, we can use the coupling guarantee and the bound d L ( ˆ D,D ⋆ ) ≤ ε 2 to show that the left hand side above and the true expected regret differ by O ( ε 2 T ) = O (1) , giving the theorem.

Specifically, by the coupling guarantee, we have

<!-- formula-not-decoded -->

Since d L ( ˆ D,D ⋆ ) ≤ ε 2 , Lemma 2.2 implies that rev ⋆ ( br ⋆ ( u ′ t ) , u ′ t ) ≤ rev ˆ D ( br ˆ D ( u ′ t ) , u ′ t ) + O ( ε 2 ) for all rounds t . We further have

<!-- formula-not-decoded -->

Combining the above, we obtain

<!-- formula-not-decoded -->

as desired.

## B.10 Proof of Theorem 3.8, Lower Bound

Previously, Cesa-Bianchi et al. (2019) gave a lower bound of Ω( √ K ⋆ T ) for the non-contextual case. We now modify their one-dimensional construction so that it can be cleanly tensored into d dimensions, when K ⋆ ≥ 4 .

One-dimensional construction ( K ⋆ ≥ 4 ). Starting in 1D, we define valuations 1 2 = v 1 ≤ · · · ≤ v K ⋆ = 1 by v i := 1 2 + i -1 4 K ⋆ -2 i -2 . Define the base distribution Q 0 on { v 1 , . . . , v K ⋆ } by

<!-- formula-not-decoded -->

Observe that each valuation v i has the same expected revenue of 1 / 2 . Indeed, we compute

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Now, for j ∈ { 2 , . . . , K ⋆ -1 } , we define distribution Q j by slightly lowering the probability of v j -1 and increasing the probability of v j by the some small ε &gt; 0 , to be determined. That is, we define

<!-- formula-not-decoded -->

which is well-defined so long as ε ≤ 1 2 K ⋆ -2 . In contrast to the construction in Cesa-Bianchi et al. (2019), our Q j distributions share the same multiset of probability weights, differing only in the locations of the perturbed valuations. This allows us to tensor these problem instances into a d -dimensional instance without leaking information between instances.

Moreover, the hardness result of Cesa-Bianchi et al. (2019) is maintained, which applies even if K ⋆ is known to the learner. The proof is quite similar, so we defer it to Section B.11.

Lemma B.7 (Regret of one-dimensional family, K ⋆ ≥ 4 ) . Fix any number of types K ⋆ ≥ 4 and T ≥ K 3 ⋆ . Let A be any algorithm for non-contextual pricing. Then, for tuned ε and an instance Q ⋆ drawn uniformly at random from { Q j } j ∈{ 2 ,...,K ⋆ -1 } , A suffers expected regret at least Ω( √ K ⋆ T ) .

Tensoring one-dimensional instances. To extend these one-dimensional distributions into d dimensions, define the base distribution D 0 ∈ ∆([0 , 1] d ) as follows. For i ∈ [ K ⋆ ] , let θ i := [ v i , . . . , v i ] ∈ R d , w i := Q 0 ( v i ) , and take D 0 := ∑ K ⋆ i =1 w i δ θ i . That is, θ i ∈ [0 , 1] d has all entries equal to v i and w i is the probability D 0 places on θ i , taken to match Q 0 at v i . By design, the marginal distribution of D 0 along each coordinate is Q 0 . Now, for a selection j = ( j 1 , . . . , j d ) ∈ { 2 , . . . , K ⋆ -1 } d , define the perturbed instance D j by starting from D 0 and modifying it as follows:

- Adjust the probabilities w 1 and w 2 by w 1 ← w 1 -ε and w 2 ← w 2 + ε .
- For each dimension ℓ ∈ [ d ] , permute the ℓ th coordinates of the θ i vectors so that the marginal distribution of their ℓ th coordinates coincides with Q j ℓ .

Specifically, we define D j := ∑ K ⋆ i =1 ˜ w i δ ˜ θ j,i where

<!-- formula-not-decoded -->

and ˜ θ j,i [ ℓ ] := v σ ( i ) , for any permutation σ of [ K ⋆ ] such that σ ( j ℓ -1) = 1 , σ ( j ℓ ) = 2 , and σ ( K ⋆ ) = K ⋆ . Simply swapping j ℓ -1 and 1 and j ℓ with 2 works unless j ℓ = 3 , in which case one can send 2 → 1 , 3 → 2 , and 1 → 3 . As claimed above, this construction ensures that for each dimension ℓ ∈ [ d ] , the marginal distribution of D j is Q j ℓ . Indeed, for each i ∈ [ K ⋆ ] , we have

<!-- formula-not-decoded -->

Lower bounding the regret. For the contextual setting, we sample a selection j = ( j 1 , . . . , j d ) ∈ { 2 , . . . , K ⋆ -1 } d uniformly at random and set D ⋆ to the perturbed instance D j . We consider stochastic contexts, where each u t is the standard basis vector along coordinate ℓ t sampled uniformly at random from [ d ] . Now, fix any contextual pricing policy A for this this randomized environment. Our high-level intuition is that, for each coordinate ℓ ∈ [ d ] , the sub-environment during the roughly T/d rounds when ℓ t = ℓ mirrors that of Lemma B.7, and so we incur regret Ω( √ K ⋆ · T/d ) during such rounds. Summing over ℓ ∈ [ d ] then gives the lower bound.

̸

To formalize this, note that each coordinate is sampled at least T ′ = ⌊ T/ 2 d ⌋ times under an event E with probability at least 1 -1 /T . This follows by a Chernoff bound a union bound over coordinates since T ≥ 8 d log(2 d ) . Then, for each ℓ ∈ [ d ] , there is a natural policy A ℓ induced for the non-contextual setting of Lemma B.7 with time horizon T ′ . To start, A ℓ samples a valuation distribution ˜ Q ℓ ′ uniformly at random from { Q j } j ∈{ 2 ,...,K ⋆ -1 } for each ℓ ′ = ℓ . Further, it instantiates a simulated copy of A and a counter τ , initialized at 1 , tracking the round of this simulation. Then, for round t ′ = 1 , . . . , T ′ , A ℓ performs the following:

1. If τ &gt; T , play p t ′ = 1 for remaining rounds and terminate.
2. Otherwise, sample ℓ τ uniformly from [ d ] .
3. Submit the associated context as u τ to A and receive suggested price ˜ p τ .
4. If ℓ τ = ℓ , play price p t ′ = ˜ p τ , submit the purchase feedback ˜ y τ = y t ′ to A , increment τ ← τ +1 , and continue to the next round.
5. Otherwise, submit ˜ y τ ∼ Ber( x ) where x is the demand of ˜ Q ℓ τ at ˜ p τ , increment τ ← τ +1 , and return to Step 1.

By design, if A ℓ is run under the setting of Lemma B.7, its simulated copy of A experiences feedback indistinguishable from that described in our setting above. Thus, writing T ℓ for the rounds of our

initial setting where ℓ t = ℓ and conditioning on E (under which |T ℓ | ≥ T ′ ), we have

<!-- formula-not-decoded -->

since T ′ ≥ K 3 ⋆ and T ≥ 2 d . We then compute

<!-- formula-not-decoded -->

as desired.

Small K ⋆ . If K ⋆ = 2 , a simpler one-dimensional construction suffices. We set v 1 = 1 / 4 , v 2 = 1 / 2 , Q 0 ( v 1 ) = Q 0 ( v 2 ) = 1 / 2 , so that dem Q 0 ( v 1 ) = 1 , dem Q 0 ( v 1 ) = 1 / 2 , and rev Q 0 ( v 1 ) = rev Q 0 ( v 2 ) = 1 / 4 . We then define Q ± ( v 1 ) = 1 / 2 ∓ ε and Q ± ( v 2 ) = 1 / 2 ± ε , so that rev Q ± ( v 1 ) = 1 / 4 and rev Q ± ( v 2 ) = 1 / 4 ± ε/ 2 . Moreover, taking ε = 1 / √ T (valid so long as T ≥ 4 , which we assumed), we can simply employ the standard 2 -armed bandits lower bound (e.g., using the same techniques as in Section B.11) to show that no algorithm can achieve regret o ( √ T ) for both instances. Our argument above, tensoring one-dimensional instances and obtaining a contextual lower bound, still goes through, since Q + and Q -share the same set of probability weights, giving a lower bound of Ω( √ Td ) for a worst-case instance. For K ⋆ = 3 , we can easily tweak the K ⋆ = 2 construction to place negligible mass at v 3 = 0 , and the lower bound still holds.

## B.11 Non-contextual Lower Bound (Proof of Lemma B.7)

We first recall some basic information theory. Write KL( p ∥ q ) := E p [log(d p/ d q )] for the KullbackLeibler divergence between distributions p and q on the same domain X . When p and q are Bernoulli distributions with success probabilities a, b ∈ [0 , 1] , we write KL( a ∥ b ) = KL( p ∥ q ) .

Fact B.8 (Pinsker's inequality) . For p, q ∈ ∆( X ) and M ≥ 0 , we have

<!-- formula-not-decoded -->

Fact B.9 (Bernoulli KL bound) . For a, δ ∈ [0 , 1] such that a + δ ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

Fact B.10 (KL chain rule) . For distributions p, q over sequences X n =( X 1 , . . . , X n ) ∈X n ,

<!-- formula-not-decoded -->

where p ( X i | X 1 , . . . , X i -1 ) denotes the conditional distribution of X i under p given X 1 , . . . , X i -1 .

̸

Next, we observe that, if the buyer valuations are drawn from Q j , the price v j achieves expected revenue at least 1 / 2 + Ω( ε ) whereas all other v i = v j achieve revenue 1 / 2 . Indeed, the demand function of Q j is identical to Q 0 at all v i = v j , and at v j we have

̸

<!-- formula-not-decoded -->

Thus, gap Q j ( v i ) ≥ ε/ 2 · 1 { i = j } . Also, for i &gt; 1 , Eq. (5) implies that 1 / 2 ≤ dem Q 0 ( v i ) ≤ 5 / 6 . So, imposing that our perturbation size ε is less than 1 / 12 , we have that

<!-- formula-not-decoded -->

for all i, j ∈ { 2 , . . . , K ⋆ -1 } .

̸

Without loss of generality, we assume that the fixed pricing algorithm A is deterministic (since we may condition on any internal randomness of A ). Applying A to instance Q J induces a random sequence of prices P 1 , . . . , P T , where P t each is a function of the previous purchase decisions Y τ = 1 { V τ ≥ P τ } for τ ∈ [ t -1] . Without loss of generality, we may assume that each P t belongs to { v 2 , . . . , v K ⋆ -1 } , since rounding up to the nearest element of this set can only increase expected revenue. Thus, defining the empirical frequencies

Now, let J be drawn uniformly at random from { 2 , . . . , K ⋆ -1 } , so that Q J coincides with the random instance from the lemma statement. Conditioned on J , let the valuations V T = ( V 1 , . . . , V T ) be drawn i.i.d. from Q J , i.e., V T ∼ Q T J . Under this set up, v J is the unique optimal price, and playing v i for i = J incurs regret Ω( ε ) . We will be comparing to the alternative world where V T ∼ Q T 0 and all arms have equal expected revenue of 1 / 2 . All expectations under this alternative will be clearly denoted as such.

<!-- formula-not-decoded -->

we have ∑ K ⋆ -1 j =2 N j = T . Conditioned on J , the regret of A over the T rounds is

̸

<!-- formula-not-decoded -->

Thus, we have in expectation over J and V T that

<!-- formula-not-decoded -->

We next control E J,V T [ N J ] using the KL chain rule and Pinsker's inequality. Fixing j ∈ { 2 , . . . , K ⋆ -1 } , write q j ( · ) and q 0 ( · ) for the induced distributions on the entire purchase sequence Y T = ( Y 1 , . . . , Y T ) under Q j and Q 0 , respectively. By the chain rule, we compute

<!-- formula-not-decoded -->

In short, q 0 and q j are only distinguishable for rounds in which A selects price v j , and even in these rounds their divergence is bounded. Now, since N j is a deterministic function of Y T for fixed A ,

<!-- formula-not-decoded -->

Taking an expectation over J then gives

<!-- formula-not-decoded -->

Since ∑ K ⋆ -1 j =2 N j = T and J is uniform over { 2 , . . . , K ⋆ -1 } , K ⋆ ≥ 4 , it follows that

<!-- formula-not-decoded -->

Plugging this into (7) gives as desired.

<!-- formula-not-decoded -->

We now set ε = 1 3 √ 24 √ ( K ⋆ -2) /T . Our construction required that ε ≤ 1 / (2 K ⋆ -2) and ε ≤ 1 / 12 , which are satisfied under our assumption that T ≥ K 3 ⋆ . This yields a final lower bound of

<!-- formula-not-decoded -->

## ALGORITHM 4: ZoomV : Variance-Aware Zooming for Non-Contextual Pricing

```
1 Initialize : active price set S ←{ 2 i T : i = 0 , 1 , . . . , ⌊ log 2 T ⌋} ∪ { 1 } ; 2 for each round t ∈ [ T ] do 3 play p t ∈ arg max q ∈ S index t ( q ) ; 4 observe y t = 1 { θ t ≤ p t } and update n t +1 ( p t ) ; 5 if a price p > 1 /T becomes uncovered then 6 q ← min { q ′ ∈ S : q ′ > p t } ; 7 S ← S ∪ { ( p t + q ) / 2 }
```

## C Proofs for Section 4

To fully specify the algorithm, we introduce the following notation:

- Since our analysis exclusively reasons about the true type distribution D ⋆ , we abbreviate dem = dem ⋆ , rev = rev ⋆ , gap = gap ⋆ , and br = br ⋆ .
- For each price p ∈ [0 , 1] and round t ∈ [ T ] , we write
- T t ( p ) := { τ ∈ [ t -1] : p τ = p } for the set of previous rounds where p was played,
- n t ( p ) := |T t ( p ) | for the count of these rounds,
- µ t ( p ) := 1 n t ( p ) ∑ τ ∈T t ( p ) py t for the average revenue during these rounds,
- V t ( p ) := 1 n t ( p ) -1 ∑ τ ∈T t ( p ) ( py t -µ t ( p )) 2 for the sample variance, and
- σ 2 ( p ) := p 2 dem ( p )(1 -dem ( p )) for the population variance (unknown to the seller).

When n t ( p ) = 0 , we set µ t ( p ) = 0 = V t ( p ) = 0 . When n t ( p ) = 1 , take V t ( p ) = ∞ .

- Defining confidence radius r t ( p ) := √ 10 V t ( p ) log T n t ( p ) + 12 log( T ) n t ( p ) -1 (taken as + ∞ if n t ( p ) ≤ 1 ), a variant of Bernstein's inequality shows that | µ t ( p ) -rev ( p ) | ≤ r t ( p ) w.h.p. (see Lemma C.3).
- Write UCB t ( p ) := µ t ( p ) + r t ( p ) , so that rev ( p ) ≤ UCB t ( p ) w.h.p.
- We say a price p is covered by q ∈ S if p ∈ [ q, q + r t ( q )] and q is the largest active price no greater than p , i.e., q = max { q ′ ∈ S : q ′ ≤ p } . One-sided Lipschitzness of the revenue function (Lemma 2.1) implies that rev ( p ) -rev ( q ) ≤ r t ( q ) w.h.p.
- Define the index of a price q ∈ S as index t ( q ) := UCB t ( q ) + r t ( q ) . Each price p covered by some q ∈ S satisfies rev ( p ) ≤ index t ( q ) .

## C.1 Main Regret Bound for ZoomV (Proof of Theorem 4.1)

As mentioned, the lower bound follows by that in Theorem 3.8 when d = 1 . For the upper bound, we prove a generic regret bound in Section C.2 depending on the variance-aware zooming dimension.

Lemma C.1. For c &gt; 0 , ZoomV achieves regret ˜ O ( c 1 / (2+ z ) T 1 -1 / (2+ z ) ) , where z = ZoomDimV ( c ) .

Next, we prove the zooming dimension bounds claimed in Section 4.

Lemma C.2. We have ZoomDimV (10 K ⋆ ) = 0 and ZoomDimV (10) ≤ ZoomDim (10) ≤ 1 .

Proof. For each δ &gt; 0 and type i ∈ [ K ⋆ ] , let X ( i ) δ denote the set of activated arms p with gap ⋆ ( p ) ≤ δ that lie in the interval ( θ ( i -1) , θ ( i ) ] , to the left of type i . Since revenue is linearly increasing within

each such interval, with slope d i = dem ( θ ( i ) ) , the gap condition requires that each p ∈ X ( i ) δ also satisfies p ≥ θ ( i ) -δd -1 i . Moreover, for p ∈ X ( i ) δ , we have σ 2 ( p ) ≤ d i . Thus, we obtain

<!-- formula-not-decoded -->

implying the first bound. For the second, we note that N ( X δ , δ/ 20) ≤ N ([0 , 1] , δ/ 20) ≤ 20 δ -1 .

Combining the two lemmas gives the theorem. Indeed, the √ K ⋆ T bound follows by Lemma C.1 with c = 10 K ⋆ and z = 0 , using Lemma C.2. Similarly, the T 2 / 3 bound follows by taking c = 10 and z = 1 .

## C.2 Base Regret Bound for ZoomV (Proof of Lemma C.1)

We begin with a few helper lemmas. Throughout, we assume that T ≥ 3 (otherwise the regret bound holds trivially). Our proofs mirror those of similar lemmas in Slivkins et al. (2019), with small adjustments to handle the variance-adjusted confidence radii and the dyadic price selection rule.

Lemma C.3 (Concentration) . Write E clean for the event that

<!-- formula-not-decoded -->

for all t ∈ [ T ] and for all p ∈ [0 , 1] . Then P ( E clean ) ≥ 1 -8 T -2 .

Proof. For fixed p ∈ [0 , 1] and t ∈ [ T ] , Theorems 10 and 11 of Maurer and Pontil (2009) imply that

<!-- formula-not-decoded -->

with probability 1 -8 T -5 . We note that similar bounds appear in Audibert et al. (2009), which inspired our adjustments to the confidence intervals. Under this event, we further bound

<!-- formula-not-decoded -->

Taking a union bound over t , the above must hold for all t ∈ [ T ] with probability at least 1 -8 T -4 . Now, the same Chernoff bound argument used in Claim 4.13 of Slivkins et al. (2019) implies that | µ t ( p ) -rev ( p ) | ≤ r t ( p ) for all p ∈ [0 , 1] and t ∈ [ T ] with probability at least 1 -8 T -2 . One technical observation is that Claim 4.13 requires that the set of arms every played by the algorithm is finite. This holds for ZoomV due to our dyadic arm activation rule.

Lemma C.4 (Covering invariant) . At the beginning of each round, every price p ≥ 1 /T is covered by some active arm.

Proof. At round 1, r t ( q ) = ∞ for all active arms q ∈ S , and so all arms larger than 1 /T are covered by our choice of S . Now, suppose that the lemma holds up to round t , and that playing p t causes a price p ∈ R to become uncovered. Then, by the definition of covering, we must have

<!-- formula-not-decoded -->

where q is the nearest active price to the right of p t , selected at Step 6. First, verify that the added price, p ′ := ( p t + q ) / 2 , is less than p . Since r t +1 ( p t ) must be less than one, we must have n t +1 ( p ) -1 = n t ( p ) ≥ 12 . Thus, we can bound

<!-- formula-not-decoded -->

Consequently, one can show that

<!-- formula-not-decoded -->

Combining, we find that p ′ = ( p t + q ) / 2 ≤ p t + 1 2 r t ( p t ) &lt; p t + r t +1 ( p t ) &lt; p , as desired. Moreover, p ′ could not have already been active at round t ; otherwise, p t would not have been covering p . Finally, once p ′ is added, it covers p since r t +1 ( p ′ ) = ∞ .

Lemma C.5 (Gap bound) . Condition on E clean . Then gap ( p ) ≤ 5 r t ( p ) and n t ( p ) ≤ 252 σ 2 ( p ) log T gap ( p ) 2 + 504 log T gap ( p ) for all p ∈ [0 , 1] and t ∈ [ T ] .

Proof. Write ˆ p = max { br , 1 T } , and fix any price p ∈ [0 , 1] . Consider some round t at which p is played. By Lemma C.4, we know that ˆ p was covered at the beginning of round t by some price q ∈ S , and that p had a higher index than q . Hence,

<!-- formula-not-decoded -->

Moreover, index t ( p ) ≤ µ t ( p ) + 2 r t ( p ) ≤ rev ( p ) + 3 r t ( p ) . Thus,

<!-- formula-not-decoded -->

Now, if n t ( p ) ≤ 12 , gap ( p ) ≤ 1 &lt; r t +1 ( p ) . Otherwise, the bound above implies that gap ( p ) ≤ 5 r t +1 ( p ) . Since r t ( p ) only changes when p is played and gap ( p ) ≤ 5 r t ( p ) when t = 1 , this guarantee holds for all t . For the other bound, we apply concentration to obtain

<!-- formula-not-decoded -->

Rearranging and solving the quadratic inequality in n t ( p ) gives the stated result.

Compared to the standard gap bound lemma for zooming (see, e.g., Lemma 4.14 of Slivkins et al., 2019), Lemma C.5 is adapted to the variance of each price p . We can now prove the regret bound.

Lemma C.6 (Active arm separation) . Conditioned on E clean , consider any three consecutive active arms x &lt; y &lt; z which did not belong to S at initialization. Then z -x &gt; 1 10 min { gap ( x ) , gap ( y ) } .

Proof. If y was activated before z , then z must have been added as the midpoint of active arms y and y +2( z -y ) = 2 z -y at round τ z , when y must have not covered 2 z -y . Thus, by Lemma C.5, we would have 2( z -x ) &gt; 2( z -y ) = (2 z -y ) -y &gt; 1 5 gap ( y ) . On the other hand, if y was activated after z , then it must have been added as the midpoint of x and z at round τ y , when x must have not covered z . Again, by Lemma C.5, this would imply z -x &gt; 1 5 ∆( y ) .

Proof of Lemma C.1 We freely condition on E clean , since the complement has negligible probability O ( T -2 ) . For each δ &gt; 0 , let Y δ ⊆ X 2 δ denote the set of activated prices p with gap ( p ) ∈ [ δ, 2 δ ) . In what follows, we say that two prices are adjacent if they are neighboring within Y δ . Note that at most O (log T ) of the prices in Y δ were activated at initialization. Consider the set Y 0 δ which, for each such price, contains this price and up to two neighboring prices, such that the remaining prices Y δ \ Y 0 δ can be split into triples of neighboring prices. We then decompose

<!-- formula-not-decoded -->

where p i, 1 , p i, 2 , p i, 3 are neighboring for each i ∈ [ n ] . By Lemma C.6, we have p i, 3 -p i, 1 &gt; δ/ 10 for all i ∈ [ n ] , and so p i,k -p j,k &gt; δ/ 10 for all k ∈ [3] whenever i &lt; j -1 . Thus, we can partition Y δ \ Y 0 δ into at most 6 packings, each of which has separation at least δ/ 10 . Of course any ( δ/ 10) -packing of Y δ ⊆ X 2 δ is contained within a ( δ/ 5) -cover of X 2 δ . Consequently, we have

<!-- formula-not-decoded -->

Noting that a ( δ/ 10) -packing within [0 , 1] can have cardinality at most 10 δ -1 , we further bound | Y δ | = O (log T + δ -1 ) . Thus, by Lemma C.5, the regret incurred due to posting prices in Y δ is at most

<!-- formula-not-decoded -->

Now we sum over δ = 1 / 2 , 1 / 4 , . . . , α , where α will be tuned later, giving a total regret bound of

<!-- formula-not-decoded -->

## D Proofs for Section 5

We first recall some results from Vapnik-Chervonenkis (VC) theory. Given distributions D,D ′ over a finite domain X , define the total variation distance ∥ D -D ′ ∥ TV := sup A ⊆X | D ( A ) -D ′ ( A ) | .

Lemma D.1 (Section 28.1 of Shalev-Shwartz and Ben-David, 2014) . Fix a finite set X , a function family F ⊆ { 0 , 1 } X , a distribution D ∈ ∆( X ) , and δ ∈ (0 , 1) . Then, for X 1 , . . . , X n sampled i.i.d. from D , we have

<!-- formula-not-decoded -->

with probability at least 1 -δ , where V is the VC dimension of F .

Lemma D.2 (Theorem 9.3 of Shalev-Shwartz and Ben-David, 2014) . The family F = { 0 , 1 } X has VC dimension |X| , and the family of linear thresholds over R d has VC dimension d +1 . The former result implies that, under the setting above, we have

<!-- formula-not-decoded -->

with probability at least 1 -δ .

We also recall Bernstein's inequality for the case of i.i.d. Bernoulli random variables.

Lemma D.3 (Theorem 2.10 of Boucheron et al., 2013) . For i.i.d. X 1 , . . . , X n ∼ Ber( p ) and δ &gt; 0 ,

<!-- formula-not-decoded -->

with probability at least 1 -δ .

We now turn to the main proofs.

## D.1 Observed Type Identifiers (Proof of Theorem 5.1)

To state our result for the first setting, we introduce an ε -ball performance metric for contextual search. This is a slight strengthening of the standard ε -ball metric ∑ T t =1 1 {| p t -v t | &gt; ε } .

Definition D.4 (Strong ε -ball regret) . Let A be a contextual pricing policy which, at round t ∈ [ T ] with context u t , outputs a price p t ∈ [0 , 1] along with a confidence width w t ∈ [0 , 1] . For ε ∈ (0 , 1] , we say that A achieves strong ε -ball regret R ( T ) for contextual search if, when K ⋆ = 1 and D ⋆ = δ θ ⋆ , we have | p t -v t | ≤ w t for each round t and ∑ T t =1 1 { w t &gt; ε } ≤ R ( T ) , where v t = ⟨ u t , θ ⋆ ⟩ .

That is, A produces ε -accurate estimates for the true values, outside of up to R ( T ) rounds for exploration, and it can identify when these estimates are accurate. In practice, this tends to require that A maintain a confidence set around θ ⋆ whose width, when projected onto the current context, is greater than ε for at most R ( T ) rounds. Fortunately, there are existing efficient algorithms which achieve low ε -ball regret.

Lemma D.5 (Lobel et al., 2018) . For ε ∈ (0 , 1] , there exists a contextual search algorithm ProjectedVolume ( ε ) , based on the ellipsoid method, with strong ε -ball regret O ( d log( d/ε )) and running time poly( d, 1 /ε ) per round. 2

We now present our algorithm (Algorithm 5), which uses ProjectedVolume as a subroutine.

2 Although Lobel et al. (2018) state a slightly weaker guarantee, instead bounding ∑ T t =1 1 {| p t -v t | &gt; ε } = O ( d log( d/ε )) , this strengthened result is immediate from their proof.

## ALGORITHM 5: Contextual Pricing with Ex-Post Type Identification

```
1 initialize : observed types I = ∅ , ε = √ d log( T ) /T ; 2 for each round t ∈ [ T ] do 3 observe context u t ; 4 if exists i ∈ I such that width ( A i , u t ) > ε and m t ( i ) < εT then 5 play p t = price ( A i , u t ) and observe y t ; 6 observe type z t ∈ [ K ⋆ ] ; 7 update algorithm A i with y t if z t = i ; 8 increment m t ( i ) by 1; 9 else 10 let S = { i ∈ I : width ( A i , u t ) ≤ ε } ; 11 define F ( i ) = ∑ j ∈S n t ( j ) t -1 · 1 { price ( A j , u t ) ≥ price ( A i , u t ) } for each i ∈ S ; 12 set i ∗ = arg max i ∈S F ( i ) · price ( A i , u t ) ; 13 play p t = max { price ( A i ∗ , u t ) -ε, 0 } and observe y t ; 14 observe type z t ∈ [ K ⋆ ] ; 15 increment n t ( z t ) by 1; 16 if z t ̸∈ I then 17 initialize copy A z t of ProjectedVolume ( ε ) and set I ← I ∪ { z t } ;
```

Overview of Algorithm 5 We maintain a set I of observed types, initially empty, and an accuracy ε (tuned to minimize regret). We will initialize an independent copy A i of ProjectedVolume for each i added to I . Since these copies are simulated, we are free to query the price price ( A i , u t ) and confidence width width ( A i , u t ) for a context u t without updating A i . Moreover, for each i ∈ I , the algorithm maintains a frequency count n t ( i ) , recording the number of rounds which we have followed the recommended price of A i , along with an exploration count m t ( i ) , recording the number of rounds which we have played the price of A i due to its lack of confidence along the current context. At each round t , we perform the following:

- If there is an observed type i ∈ I such that width ( A i , u t ) &gt; ε and that its number of exploration plays m t ( i ) is below a threshold of εT/K ⋆ , the algorithm plays price ( A i , u t ) , observes the outcome and the realized type z t , and updates A i if z t = i . In addition, we increment n t ( i ) and m t ( i ) .
- Otherwise, it defines active set S = { i ∈ I : width ( A i , u t ) ≤ ε } , computes for each i ∈ S the score

<!-- formula-not-decoded -->

and plays p t = max { price ( A i ∗ , u t ) -ε, 0 } where i ∗ ∈ arg max i ∈S { F ( i ) · price ( A i , u t ) } . It then observes y t and z t and updates the frequency count n t +1 ( z t ) . Here, F is an estimate for the demand at the price suggested by A i , and so i ⋆ is an estimate for the revenue maximizing type. We pull back the price recommended by A i ⋆ by ε to avoid issues due to estimation error.

Bounding exploration regret. Write T 1 for the set of exploration rounds. By design, an exploration round is one in which some type i is used with width ( A i , u t ) &gt; ε and exploration counter satisfying m t ( i ) &lt; εT . Trivially, |T 1 | ≤ εT , so we can incur regret at most εT = ˜ O ( √ dT ) during exploration.

Bounding mass of types which saturate exploration threshold. Next, consider any type i that has been explored sufficiently so that m T ( i ) = εT after time T ; denote by S ′ the set of all such types. We will show that D ⋆ plays small mass on S ′ . Fix i ∈ S ′ and write T 1 ,i for the exploration rounds where we follow A i . Conditioned on T 1 ,i , we note that X t = 1 { z t = i } , t ∈ T 1 ,i , are i.i.d. Bernoulli random variables with Pr( X t = 1) = D ⋆ ( i ) . Defining

<!-- formula-not-decoded -->

our guarantee for ProjectedVolume (Lemma D.5) and the width condition for exploration imply that S i = O ( d log( d/ε )) . On the other hand, by Bernstein's inequality (Lemma D.3), we have

<!-- formula-not-decoded -->

with probability at least 1 -1 /T . Since m T ( i ) = εT = √ dT log( T ) , the dominant term is m T ( i ) D ⋆ ( i ) for T greater than a sufficiently large constant. Thus, we deduce that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or equivalently,

Summing over all types in S ′ (of which there are at most K ⋆ ) and taking a union bound, the total mass in S ′ is at most

<!-- formula-not-decoded -->

with probability at least 1 -K ⋆ /T . We condition on this bound holding for the remainder of the proof, since doing so contributes a negligible K ⋆ to the regret. We also condition on the event that, for each round t ∈ [ T ] the empirical frequencies of (all) types deviate from their true masses by at most O (√ K ⋆ log( T ) /t ) in total variation. This is permissible by Lemma D.2 and a union bound over rounds.

Bounding exploitation regret. Fix an exploitation round t , and recall the set of accurately estimated types

<!-- formula-not-decoded -->

Write v 1 , . . . , v K ⋆ ∈ [0 , 1] for the true values at round t . By our construction and the ε -ball guarantee for ProjectedVolume , A i returns a price that is an ε -accurate estimate of v i , for each i ∈ S . Moreover, by our analysis above, the mass on types outside of S is quite small. We thus bound

<!-- formula-not-decoded -->

as desired.

## D.2 Observed Type Vectors (Proof of Theorem 5.2)

We first show that rev ˆ D τ concentrates tightly around rev ⋆ , using a simple VC bound.

Lemma D.6. Fix D ∈ ∆ K (Θ) and let ˆ D t be the empirical measure of t i.i.d. samples from D . We then have

<!-- formula-not-decoded -->

with probability at least 1 -δ .

<!-- formula-not-decoded -->

Consequently, we bound rev ⋆ ( p t , u t ) + ˜ O ( K ⋆ √ d log( T ) /t ) from below by

<!-- formula-not-decoded -->

All together, we see that playing p t incurs regret at most ˜ O ( K ⋆ √ d log( T ) /t ) . Summing over exploitation rounds and adding the exploration regret gives a total bound of

<!-- formula-not-decoded -->

Proof. We compute

<!-- formula-not-decoded -->

where F is the space of linear threshold functions f p,u : supp( D ) → { 0 , 1 } given by f p,u ( θ ) = 1 {⟨ u, θ ⟩ ≥ p } . The result then follows by Lemma D.2.

Now, our best response policy ensures that, at each round t &gt; 1 , we have

<!-- formula-not-decoded -->

Consequently, regret is at most

<!-- formula-not-decoded -->

as desired.