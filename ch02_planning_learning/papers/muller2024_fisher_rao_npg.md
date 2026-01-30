## FISHER-RAO GRADIENT FLOWS OF LINEAR PROGRAMS AND STATE-ACTION NATURAL POLICY GRADIENTS

JOHANNES MÜLLER 1 , ♡ , SEMIH ÇAYCI 1 , AND GUIDO MONTÚFAR 2 , 3

Abstract. Kakade's natural policy gradient method has been studied extensively in the last years showing linear convergence with and without regularization. We study another natural gradient method which is based on the Fisher information matrix of the state-action distributions and has received little attention from the theoretical side. Here, the state-action distributions follow the Fisher-Rao gradient flow inside the state-action polytope with respect to a linear potential. Therefore, we study Fisher-Rao gradient flows of linear programs more generally and show linear convergence with a rate that depends on the geometry of the linear program. Equivalently, this yields an estimate on the error induced by entropic regularization of the linear program which improves existing results. We extend these results and show sublinear convergence for perturbed Fisher-Rao gradient flows and natural gradient flows up to an approximation error. In particular, these general results cover the case of state-action natural policy gradients.

Keywords: Fisher-Rao metric, linear program, entropic regularization, multi-player game, Markov decision process, natural policy gradient

MSC codes: 65K05, 90C05, 90C08, 90C40, 90C53

## 1. Introduction

Natural policy gradient (NPG) methods and their proximal and trust region formulations known as PPO and TRPO are among the most popular policy optimization techniques in modern reinforcement learning (RL). As such they serve as a cornerstone of many recent RL success stories including celebrated advancements in computer games [51, 52, 11] and the recent development of large language models like ChatGPT [1]. This has motivated a quickly growing body of work studying the theoretical aspects such as the convergence properties and statistical efficacy of natural policy gradient methods. Almost all of these works consider a specific model geometry where the Fisher-Rao metrics of the individual rows of the policy are mixed according to their state distribution or slight modifications of this [26, 9, 36, 28]. However, other choices for the model geometry are possible. In particular, the Fisher metric on the state-action distributions has been used to design a natural gradient method as well as actor-critic and a trust-region variant known as relative entropy search (REPS) [37, 38, 45]. This alternative natural policy gradient has been found to have the potential to reduce the severity of plateaus [37] and improve the performance of actor-critic methods [38]. Despite these findings, theoretical results remain scarce. Initial works on the convergence of state-action natural policy gradients show an exponential convergence guarantee [42], without quantifying the exponential rate and without addressing function approximation. In this article, we provide quantitative convergence results for state-action natural policy gradients both with and without function approximation.

For our theoretical analysis, we work in the space of state-action distributions which brings the benefit that the reward optimization problem becomes a linear program [27]. In particular, for rich enough parametric policy models, the state-action natural policy gradient methods can be described by the Fisher-Rao gradient flow of the state-action linear program [42]. This motivates

1 Department of Mathematics, RWTH Aachen University, Aachen, 52062, Germany

2 Departments of Mathematics and Statistics &amp; Data Science, University of California, Los Angeles, 90095, USA

3 Max Planck Institute for Mathematics in the Sciences, Leipzig, 04103, Germany

♡ Corresponding author

E-mail addresses : mueller@mathc.rwth-aachen.de, cayci@mathc.rwth-aachen.de, montufar@math.ucla.edu .

us to study Fisher-Rao gradient flows for general linear programs. These flows coincide with the solutions of entropy-regularized linear programs and thus by studying the convergence of the flow we also bound the error introduced by entropic regularization in linear programming.

- 1.1. Contributions. It is the goal of this article to provide insights into the convergence properties of state-action natural policy gradients with and without function approximation. To this end, we first provide an explicit convergence analysis of Fisher-Rao gradient flows of general linear programs. More precisely, our main contributions can be summarized as follows:
- We study Fisher-Rao flows of general linear programs. Leveraging a local generalized strong convexity condition we show linear convergence both in KL-divergence and function value with an exponential rate depending on the geometry of the linear program, see Theorem 3.2 and Theorem 3.13 for unique and non-unique optimizers.
- We obtain an estimate on the regularization error in entropy regularized linear programming improving known convergence rates, see Corollary 3.3.
- We study natural gradients for parametric measures and show sublinear convergence under inexact gradient evaluations up to an approximation error and a distribution mismatch measured in the χ 2 -divergence, see Corollary 4.7.
- In a multi-player game with a specific payoff structure, we show linear convergence of the natural gradient flow, see Theorem 4.9.
- In the context of Markov decision processes, we study state-action natural policy gradients and provide a sublinear convergence result for general policy parametrizations, see Corollary 5.4, and a linear convergence guarantee gradient for regular parametrizations, see Corollary 5.8. In particular, this covers tabular softmax, escort, and log-linear parameterizations.
- For non-unique optimizers, the asymptotic limit of Fisher-Rao gradient flows is known to be the information projection of the initial condition to the set of optimizers. We strengthen this by providing an exponential convergence rate, see Theorem 3.13, and by extending this result to state-action natural policy gradients. This shows that stateaction natural gradients converge to an optimal policy that achieves maximal entropy over states and actions, which characterizes its implicit bias , see Theorem 3.13 and Corollary 5.8.

1.2. Related works. State-action natural policy gradients were recently studied with and without state-action entropy regularization in [42]. For regularization strength λ &gt; 0 that work showed O ( e -λt ) convergence, but in the unregularized case, the precise exponential rate was not characterized.

A mirror descent variant of the state-action natural policy gradients was shown to achieve an optimal O ( √ T ) regret in an online setting in [66, 21, 44].

There has been a recent surge of works studying the natural policy gradient method proposed by Kakade. The initial results of [2] showed sublinear convergence rate O ( t -1 ) for unregularized problems. This was subsequently improved to a linear rate for step sizes found by exact line search [12] and constant step sizes [29, 3, 62]. For regularized problems, the method converges linearly for small step sizes, locally quadratically for Newton-like step sizes, and linearly with linear function approximation [17, 32]. The linear convergence of NPG has been extended to the function approximation regime and more general problem geometries, where these results either require geometrically increasing step sizes [61, 3, 62, 4] or entropy regularization [16, 30, 63, 32, 4]. However, these geometries do not cover the state-action geometries. Apart from the works on convergence rates for policy gradient methods for standard MDPs, a primal-dual NPG method with sublinear global convergence guarantees has been proposed for constrained MDPs [22, 23]. Where all of these results work in discrete time, the gradient flows corresponding to this type of natural policy gradient have been shown to converge linearly under entropy regularization for Polish state and action spaces [28].

Hessian geometries, which provide a rich generalization of the Fisher-Rao metric, have been studied in convex optimization both from a continuous time perspective and via a discrete-time

mirror descent analysis [5, 58]. In the context of linear programming, linear convergence of the Fisher-Rao gradient flow was shown in [5] albeit without a characterization of the convergence rate.

In the case of a linear program, the Fisher-Rao gradient flow parametrized by time corresponds to the trajectory of solutions of the entropy-regularized program parametrized by the inverse regularization strength, which has been studied in several works. An exponential convergence result was obtained in [19] and subsequently, the rate was characterized as O ( e -δt ) for a constant δ depending on the linear program [60, 54]. The results obtained in this article follow an alternative proof strategy and provide exponential convergence O ( e -∆ t ) , where ∆ ≥ δ , where we show that the improvement can be arbitrarily large, see Example 3.5. This improvement can be strict for the linear programs encountered in Markov decision processes under standard assumptions. Whereas existing works study convergence in function value, our results also cover convergence in the KL-divergence. Finally, the geometry of Fisher-Rao gradient flows or equivalently the entropic central path was recently described as the intersection of the feasible region with a toric variety [53].

1.3. Notation and terminology. For a finite set X , we denote the free vector space over X by R X = { µ : X → R } . Its elements can be identified with vectors ( µ x ) x ∈ X . Similarly, we denote the vectors with non-negative entries and positive entries by R X ≥ 0 and R X &gt; 0 , respectively. For two elements µ, ν ∈ R X we denote the Hadamard product , i.e., the entrywise product, between µ and ν by µ ⊙ ν ∈ R X , so that µ ⊙ ν ( x ) := µ ( x ) ν ( x ) . The total variation norm ∥·∥ TV : R X → R is given by ∥ µ ∥ TV := 1 2 ∑ x | µ x | . Finally, with 1 X ∈ R X we denote the all-one vector.

A polyhedron is a set P = { µ ∈ R X : ℓ i ( µ ) ≥ 0 for i = 1 , . . . , k } ⊆ R X , where ℓ i : R X → R are affine linear functions for i = 1 , . . . , k . A bounded (and thus compact) polyhedron is called a polytope . A polytope can be shown to be the convex hull of finitely many extreme points, which are called vertices and which we denote by Vert( P ) . Two vertices µ 1 , µ 2 ∈ Vert( P ) are called neighbors if the subspace { c ∈ R X : c ⊤ µ 1 = c ⊤ µ 2 = max µ ∈ P c ⊤ µ } has dimension | X | -1 . We denote the set of all neighbors of a vertex µ by N ( µ ) ⊆ Vert( P ) . The affine space aff span( P ) of a polytope P ⊆ R X is the smallest affine subspace of R X containing P . The relative interior int( P ) and boundary ∂P of P are the interior and boundary of P in its affine hull. Finally, the tangent space TP of P is given by the linear part of aff span( P ) .

We call ∆ X := { µ ∈ R X ≥ 0 : ∑ x µ x = 1 } the probability simplex . We say that µ ∈ ∆ X is absolutely continuous with respect to ν ∈ ∆ X if ν ( x ) = 0 implies µ ( x ) = 0 and write µ ≪ ν . We denote the expectation with respect to µ ∈ ∆ X by E µ and call χ 2 ( µ, ν ) := E ν [ ( µ ( x ) -ν ( x )) 2 ν ( x ) 2 ] the χ 2 -divergence between µ and ν . If Y is another finite set, we call the Cartesian product ∆ Y X = ∆ X · . . . · ∆ X the conditional probability polytope and associate its elements with stochastic matrices P ∈ R X ×Y ≥ 0 with ∑ x P ( x | y ) = 1 .

For a differentiable function f : Ω → R on an open subset Ω ⊆ R X we denote the Euclidean gradient and Hessian of f at µ ∈ R X by ∇ f ( µ ) ∈ R X and ∇ 2 f ( µ ) ∈ R X × X .

Finally, for a differentiable curve ( c t ) t ∈ I ⊆ M defined on an interval I ⊆ R mapping to a manifold M we denote its time derivative by ∂ t c t .

## 2. Preliminaries on Fisher-Rao Gradient Flows

To gain insight into natural gradient descent methods, we study their time-continuous version which is given by ∂ t θ t = -F ( θ t ) + ∇ f ( µ θ t ) , where µ θ is a parametrized measure model and f ( µ ) is an objective function and F ( θ ) is the Fisher-information matrix [6]. The objective function can be a log-likelihood in the case of maximum likelihood estimation or a linear function in the case of reinforcement learning as we will see in Section 5. The Fisher-information matrix is closely connected to a specific Riemannian geometry, the Fisher-Rao metric, on the space of probability measures, which we introduce and discuss here. As we study gradient-based optimizers, we put a special emphasis on gradient flows with respect to the Fisher-Rao metric and provide a self-contained review of the properties of Fisher-Rao gradient flows that we require later. The results in this section can be generalized to a large class of Hessian geometries and - apart from

the central path property - also to other objectives albeit with different proofs, for which we refer to [5, 39].

The Fisher-Rao metric is a Riemannian metric on the positive orthant given by

<!-- formula-not-decoded -->

where we denote the induced norm by ∥ v ∥ g FR µ := g FR µ ( v, v ) 1 2 . The Fisher-Rao metric was introduced in the seminal works of C. R. Rao [48, 49] to provide lower bounds on the statistical error in parameter estimation known as the Cramer-Rao bound. This geometric approach to statistical estimation has subsequently led to the development of the field of information geometry, where N. N. Čencov characterized the Fisher-Rao metric as the unique Riemannian metric (up to scaling) that is invariant under sufficient statistics [18, 7, 8]. Despite its central role in statistics, our main motivation for studying the Fisher-Rao metric is for its use in reinforcement learning, where it has been used to design natural gradient algorithms as well as trust region methods [6, 37, 45]. Further, it is very closely related to entropic regularization in linear programming, which enjoys immense popularity, particularly in computational optimal transport [47, 54], see also [60] for a detailed discussion of entropy regularized linear programming.

The Fisher-Rao metric is closely connected to the negative Shannon entropy

<!-- formula-not-decoded -->

as it is induced by the Hessian of the (negative) entropy, meaning that we have g FR µ ( v, w ) = v ⊤ ∇ 2 ϕ ( µ ) w for all v, w ∈ R X , µ ∈ R X &gt; 0 . As such, the Fisher-Rao metric falls into the class of Hessian metrics that have been studied in convex optimization; we refer to [5, 39] for general well-posedness and convergence results. An important concept in the analysis of Hessian gradient flows is the Bregman divergence induced by ϕ , which in the case of the negative entropy is given by the KL-divergence

<!-- formula-not-decoded -->

for µ, ν ∈ R X ≥ 0 with µ ≪ ν , where we use the common convention 0 log 0 0 := 0 .

Consider now a continuously differentiable function f : R X ≥ 0 → R that we assume to be differentiable on R X &gt; 0 that we want to optimize over a polytope P = R X ≥ 0 ∩ L , where L is a linear space. We denote the gradient of f : R X &gt; 0 → R at µ ∈ R X &gt; 0 with respect to the Fisher-Rao metric by ∇ FR f ( µ ) and call it the Fisher-Rao gradient . Further, we denote the Fisher-Rao gradient of f : int( P ) → R by ∇ FR P f ( µ ) ∈ TP , which is uniquely determined by

<!-- formula-not-decoded -->

Note that ∇ FR P f ( µ ) is the projection of ∇ FR f ( µ ) with respect to the Fisher-Rao metric onto TP . By examining the definition of the Fisher-Rao metric we see that this is equivalent to

<!-- formula-not-decoded -->

We say that ( µ t ) t ∈ [0 ,T ) ⊆ int( P ) solves the Fisher-Rao gradient flow if it solves the gradient flow with respect to the Fisher-Rao metric, i.e., if

<!-- formula-not-decoded -->

By using the characterization (2.5) of ∇ FR P f ( µ t ) , we see that ( µ t ) t ∈ [0 ,T ) ⊆ int( P ) solves the Fisher-Rao gradient flow (2.6) if and only if we have

<!-- formula-not-decoded -->

In the remainder, we study linear programs and work in the following setting.

Setting 2.1. We consider a finite set X and a linear program

<!-- formula-not-decoded -->

̸

with cost c ∈ R X and feasible region P = R X ≥ 0 ∩L with P ∩ R X &gt; 0 = ∅ , where L ⊆ R X is an affine space. By ( µ t ) t ∈ [0 ,T ) ⊆ int( P ) we denote a solution of the Fisher-Rao gradient flow (2.6) with initial condition µ 0 ∈ P ∩ R X &gt; 0 and potential f ( µ ) = c ⊤ µ , where T ∈ R ≥ 0 ∪ { + ∞} .

Fisher-Rao gradient flows are closely connected to the solutions of KL-regularized linear programs, c ⊤ µ -λD KL ( µ, µ 0 ) . The family of solutions of the regularized problems parametrized by the regularization strength λ is referred to as the (entropic) central path in optimization [15].

Proposition 2.2 (Central path property,[5]) . Consider Setting 2.1. Then µ t is uniquely characterized by

<!-- formula-not-decoded -->

Proof. Let ˆ µ t ∈ P denote the unique maximizer of g ( µ ) := c ⊤ µ -t -1 D KL ( µ, µ 0 ) over P for t &gt; 0 , then surely ˆ µ t ∈ int( P ) . Thus, ˆ µ t is uniquely determined by ⟨∇ g (ˆ µ t ) , v ⟩ = 0 for all v ∈ TP . Direct computation yields ∇ g ( µ ) = c -t -1 ( ∇ ϕ ( µ ) -∇ ϕ ( µ 0 )) and hence ˆ µ t is uniquely determined by

<!-- formula-not-decoded -->

On the other hand, for the gradient flow, we can use (2.7) and compute for v ∈ TP

<!-- formula-not-decoded -->

This shows µ t = ˆ µ t as claimed.

□

We can use the central path property to show O ( t -1 ) convergence. The following corollary can be generalized to arbitrary convex objectives [5].

Corollary 2.3 (Sublinear convergence rate,[5]) . Consider Setting 2.1 and assume that the linear program (2.8) admits a solution µ ⋆ ∈ P . Then for µ ∈ P it holds that

<!-- formula-not-decoded -->

Proof. We have c ⊤ µ t -t -1 D KL ( µ t , µ ) ≥ c ⊤ µ ⋆ -t -1 D KL ( µ ⋆ , µ ) by the central path property. Rearranging yields the result. □

One can use the central path property to show the long-time existence of Fisher-Rao gradient flows. Again, the following result can be generalized to a large class of Hessian geometries and potentials f , see [5, 39], albeit with more delicate proofs.

Theorem 2.4 (Well-posedness of FR GFs,[5]) . Consider Setting 2.1. Then there exists a unique global solution ( µ t ) t ≥ 0 ⊆ int( P ) of the Fisher-Rao gradient flow (2.6) .

Proof. The local existence and uniqueness follow from the Picard-Lindelöf theorem [56]. Hence, it suffices to show that the Fisher-Rao gradient flow does not hit the boundary ∂P in finite time. By the central path property, this is equivalent to the statement that the solutions of all KL-regularized problems (2.9) lie in the interior int( P ) of the polyhedron, which can be easily checked. □

Figure 1. Visualization of the suboptimality gap ∆ appearing in Theorem 3.2 associated to the linear program (3.1); note that ∆ deteriorates when c is almost orthogonal to a face of P .

<!-- image -->

## 3. Convergence of Fisher-Rao Gradient Flows

We have seen that Fisher-Rao gradient flows converge globally at a sublinear rate O ( t -1 ) . We now build on this analysis and show that once the gradient flow enters a vicinity of the optimizer, it converges at a quasi-linear rate O ( t κ e -∆ t ) , where ∆ &gt; 0 depends on the geometry of the linear program and κ &gt; 0 depends on the initial condition µ 0 . Note that this yields O ( e -ct ) convergence for all c &lt; ∆ and hence we also simply talk of a linear convergence rate. We consider linear programs of the following form.

Setting 3.1. We consider a finite set X and a linear program

<!-- formula-not-decoded -->

̸

with cost c ∈ R X and feasible region P = ∆ X ∩ L with P ∩ R X &gt; 0 = ∅ , where L ⊆ R X is an affine space. By ( µ t ) t ≥ 0 ⊆ int( P ) we denote the solution of the Fisher-Rao gradient flow (2.6) with initial condition µ 0 ∈ P ∩ R X &gt; 0 and the potential f ( µ ) = c ⊤ µ .

The following result is the main contribution of this article, where we defer the proof to Section 3.1. We first establish it under the assumption that the linear program (3.1) admits a unique solution and provide a generalization in Theorem 3.13.

Theorem 3.2 (Linear convergence of Fisher-Rao GFs of LPs) . Consider Setting 3.1 and assume that the linear program (3.1) admits a unique solution µ ⋆ ∈ P . Let

<!-- formula-not-decoded -->

where N ( µ ⋆ ) denotes the set of neighboring vertices of µ ⋆ and set

<!-- formula-not-decoded -->

Then for any t ≥ t 0 we have

<!-- formula-not-decoded -->

as well as

<!-- formula-not-decoded -->

The constant ∆ depends on the geometry of the linear program, see Figure 1. Indeed, the quotient c ⊤ µ ⋆ -c ⊤ µ ∥ µ ⋆ -µ ∥ TV is the slope of the objective along the edge µ ⋆ -µ . Consequently, ∆ decreases when the cost c is closer to orthogonal to a face of P .

Using the central path property of Fisher-Rao gradient flows and initializing at the maximum entropy distribution in P yields the following result.

Corollary 3.3 (Entropic regularization error) . Consider Setting 3.1 and assume that the linear program (3.1) admits a unique solution µ ⋆ ∈ P . For t &gt; 0 denote by µ ⋆ t the unique solution of the entropy-regularized linear program

<!-- formula-not-decoded -->

where H denotes the Shannon entropy. Then for any t ≥ t 0 we have

<!-- formula-not-decoded -->

as well as

<!-- formula-not-decoded -->

where R H := max µ ∈ P H ( µ ) -min µ ∈ P H ( µ ) ≤ log | X | denotes the entropic radius of P and ∆ &gt; 0 and t 0 ≥ 0 are defined in (3.2) and (3.3) , respectively.

Similar to the convergence result, here too one can remove the uniqueness assumption, see Remark 3.14.

Remark 3.4 (Comparison with existing results) . In [19] it was shown that the regularization error for entropy-regularized linear programs decays exponentially fast, without quantifying the convergence rate. The convergence rate of the error, as well as that of Fisher-Rao gradient flows, was subsequently studied in [60, 54] , establishing a rate O ( e -δt ) with

<!-- formula-not-decoded -->

For polytopes P ⊆ ∆ X that we consider here, we have

<!-- formula-not-decoded -->

showing that Theorem 3.2 offers an improvement of these previous results.

For the special case P = ∆ X , for which a matching lower bound was constructed in [60] , the two constants agree. More generally, it is easily checked that δ = ∆ if and only if there is a neighboring vertex µ ∈ N ( µ ⋆ ) which has minimal optimality gap c ⊤ µ ⋆ -c ⊤ µ and has disjoint support from µ ⋆ . To see this, note that for two probability vectors µ 1 , µ 2 ∈ ∆ X we have ∥ µ 1 -µ 2 ∥ TV = 1 2 ∥ µ 1 -µ 2 ∥ 1 ≤ 1 with ∥ µ 1 -µ 2 ∥ = 1 if and only if µ 1 and µ 2 have disjoint support, meaning

<!-- formula-not-decoded -->

Hence, for µ ∈ N ( µ ⋆ ) without disjoint support from µ ⋆ we have ∥ µ ⋆ -µ ∥ TV &lt; 1 . This implies that δ = ∆ if and only if there is a neighboring vertex µ ∈ N ( µ ⋆ ) which has minimal optimality gap c ⊤ µ ⋆ -c ⊤ µ and has disjoint support from µ ⋆ .

The constant ∆ depends on the slope of c along the outgoing edges and thus the local geometry of the feasible region around µ ⋆ , where δ is simply based on the suboptimality at the neighboring vertices. Because of this, the difference between δ and ∆ can be arbitrarily big as we show in Example 3.5. Further, for Markov decision processes the feasible region of the (dual) linear program is a strict subset D ⊊ ∆ S × A and under the standard exploratory Assumption 5.2 and more than one state we have δ &lt; ∆ , see Remark 5.11. In Section 5.2 we provide an explicit example of a Markov decision process where δ &lt; ∆ .

Further, for gradient flows with respect to a Riemannian metric of the form g σ µ ( v, w ) := ∑ x ∈ X v x w x µ σ x one can show O ( t -1 σ -1 ) convergence for σ ∈ (1 , 2) , see [42] . Note that this can be

extended to the case σ = 2 , corresponding to logarithmic barriers for which the central path converges at a O ( t -1 ) rate [15, Section 11.2] .

Example 3.5 (Arbitrarily large improvement) . We consider X = { 1 , 2 , 3 , 4 } and L = { µ ∈ R X : µ (1) = α } for α ∈ (0 , 1) . Then, the vertices of P = ∆ X ∩L are given by (1 -α ) δ 2 , (1 -α ) δ 3 and (1 -α ) δ 4 , where δ i denotes the Dirac at i . When choosing the cost c = δ 2 we have δ = 1 -α but ∆ = 1 . For α ↗ 1 the rate δ deteriorates towards 0 , whereas ∆ remains constant. The reason for this is that ∆ depends on the slope of c relative to the outgoing edges, whereas δ depends on the suboptimality of the neighboring vertices. Hence, δ can be smaller than ∆ by an arbitrarily large factor.

Remark 3.6 (Tightness) . For P = ∆ X we have µ t ( x ) ∼ e -tc x as can be seen from the first order stationarity conditions; hence, in this case, the bound is tight. For general P , in Section 5.2 we provide empirical evidence that our bound on the exponent is sometimes but not always tight depending on the specific c .

3.1. Convergence of Fisher-Rao Gradient Flows. At the heart of the proof lies the following result, which can easily be extended to general Hessian geometries. For this, one can follow the reasoning in [5, Proposition 4.9], which treats general Hessian geometries, but does not allow for time-dependent constants κ t and assumes the lower bound (3.10) in a neighborhood of µ ⋆ and not only along the trajectory.

Lemma 3.7. Consider Setting 2.1 and assume that there is an optimizer µ ⋆ ∈ P and κ t &gt; 0 for t &gt; t 0 ≥ 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we have as well as

For the proof of this result, we require the following identity.

Lemma 3.8 ([5]) . Consider Setting 2.1, whereby we allow f : R X &gt; 0 → R to be an arbitrary differentiable function, and fix µ ∈ P . Then for any t ≥ 0 , it holds that

<!-- formula-not-decoded -->

Proof. Denoting the negative Shannon entropy by ϕ , we compute

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma 3.7. Using (3.13) and (3.10) we find that for all t ≥ T it holds that ∂ t D KL ( µ ⋆ , µ t ) = c ⊤ µ t -c ⊤ µ ⋆ ≤ -κ t D KL ( µ ⋆ , µ t ) . Now Gronwall's inequality yields (3.11). By Corollary 2.3 we have for any h &gt; 0 that

<!-- formula-not-decoded -->

Taking the limit h → 0 yields (3.12).

<!-- formula-not-decoded -->

The lower bound (3.10) can be interpreted as a form of strong convexity under which the objective value controls the Bregman divergence, see also [33, 10] for a discussion of gradient domination and strong convexity conditions in Bregman divergence. To show that such a lower bound holds in the case of the linear program (3.1), we first lower bound the sub-optimality gap c ⊤ µ ⋆ -c ⊤ µ t in terms of an arbitrary norm, where we will later use the total variation distance.

Lemma 3.9. Consider a polytope P ⊆ R X and denote by F ⋆ the face of maximizers of the linear function µ ↦→ c ⊤ µ over P . Denote the set of neighboring vertices of a vertex µ by N ( µ ) and let ∥·∥ : R X → R ≥ 0 be an arbitrary semi-norm. Then either F ⋆ = P or with c 0 := + ∞ for c &gt; 0 , we have

<!-- formula-not-decoded -->

and further

<!-- formula-not-decoded -->

̸

Proof. If F ⋆ = P , then c ⊤ µ ⋆ -c ⊤ µ &gt; 0 for some vertex µ , which implies ∆ &gt; 0 . To simplify notation we denote the set E := { µ -µ ⋆ : µ ∈ N ( µ ⋆ ) \ F ⋆ , µ ⋆ ∈ vert( F ⋆ ) } of edges such that exactly one of the two endpoints is contained in F ⋆ . Then, the polytope P is contained in

<!-- formula-not-decoded -->

see Lemma A.1 and hence we can write µ ∈ P as µ = µ ⋆ + ∑ e α e e for some µ ⋆ ∈ F ⋆ . Using the triangle inequality we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 3.10. Consider a finite set X and a probability distribution µ ∈ ∆ X . Let c &gt; 1 and set δ := c -1 c +1 · min { µ x : µ x &gt; 0 } &gt; 0 . Then for all ν ∈ ∆ X satisfying ∥ µ -ν ∥ ∞ ≤ δ it holds that

<!-- formula-not-decoded -->

Proof. We bound the individual summands in the KL-divergence

<!-- formula-not-decoded -->

where X := { x ∈ X : µ x &gt; 0 } . If µ x , ν x &gt; 0 then

<!-- formula-not-decoded -->

where we used the convexity log( t + h ) ≤ log( t )+ h/t for t &gt; 0 , t + h &gt; 0 . We set ε := c -1 2 ∈ (0 , 1) , such that

If ∥ µ -ν ∥ ∞ ≤ δ then

<!-- formula-not-decoded -->

as well as

<!-- formula-not-decoded -->

and therefore 1 -ε ≤ µ x ν x ≤ 1 + ε . If µ x ≥ ν x then

<!-- formula-not-decoded -->

and if µ x &lt; ν x then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Together with (3.17) summing over x yields

<!-- formula-not-decoded -->

It remains to estimate the first part. Setting X c := X \ X , we have

<!-- formula-not-decoded -->

since µ x = 0 for x ∈ X c . Now we can estimate

<!-- formula-not-decoded -->

Combining (3.19) and (3.20) yields

<!-- formula-not-decoded -->

it holds that

□

Corollary 3.11 (Local KL-TV estimate) . Consider a finite set X and a probability distribution µ ∈ ∆ X . Then for all ν ∈ ∆ X satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. This is a direct consequence of Lemma 3.10. Indeed, for ε &gt; 0 small enough we have ∥ µ -ν ∥ ∞ ≤ 2 -ε 2+ ε · min { µ x : µ x &gt; 0 } and thus by Lemma 3.10 with c = 1+ ε we have D KL ( µ, ν ) ≤ (1 + ε ) ∥ µ -ν ∥ TV . Note that ε &gt; 0 was arbitrary. □

Now we prove our main result on the convergence of Fisher-Rao gradient flows.

Proof of Theorem 3.2. Setting δ := min { µ ⋆ x : µ ⋆ x &gt; 0 } and using Lemma 3.9 with ∥·∥ TV and Corollary 3.11 we have

<!-- formula-not-decoded -->

if ∥ µ ⋆ -µ t ∥ ∞ &lt; δ . By Corollary 2.3 we have

<!-- formula-not-decoded -->

Hence, for t &gt; t 0 we have ∥ µ ⋆ -µ t ∥ ∞ &lt; δ . In this case, we can estimate

<!-- formula-not-decoded -->

Thus for t &gt; t 0 we have c ⊤ µ ⋆ -c ⊤ µ t ≥ ∆ κ t D KL ( µ ⋆ , µ t ) , and Lemma 3.7 together with

<!-- formula-not-decoded -->

yield the result.

□

- 3.2. Estimating the regularization error. Using the central path property we can deduce an estimate on the regularization error from the convergence results for the Fisher-Rao gradient flow. If the uniform distribution is contained in P , µ Unif ∈ P , then the claim follows simply by setting µ 0 := µ Unif as

<!-- formula-not-decoded -->

If the uniform distribution is not contained in P , we can choose its information projection as an initial distribution µ 0 to the same effect. Indeed, recall that for

<!-- formula-not-decoded -->

we have by the Pythagorean theorem that

<!-- formula-not-decoded -->

for all µ ∈ P , see [8, Theorem 2.8]. Now we can estimate the regularization error.

Proof of Corollary 3.3. By the central path property the Fisher-Rao gradient flow ( µ t ) t ≥ 0 satisfies µ t = arg max { c ⊤ µ -t -1 D KL ( µ, µ 0 ) : µ ∈ P } . If we choose µ 0 as the information projection according to (3.24) the Pythagorean theorem yields

<!-- formula-not-decoded -->

This shows that µ t = arg max { c ⊤ µ + t -1 H ( µ ) : µ ∈ P } , i.e., that µ t is the solution of the entropy regularized linear program (3.6). Now the claim follows from Theorem 3.2 and D KL ( µ ⋆ , µ 0 ) = H ( µ 0 ) -H ( µ ⋆ ) ≤ R H . □

- 3.3. Non-unique maximizers. Both Theorem 3.2 and Corollary 3.3 are formulated under the assumption that the linear program (3.1) admits a unique solution. This is satisfied for almost all costs c ∈ R X , however, it can be generalized to all costs.

To proceed like in the proof with a unique maximizer, we need to identify the limit of µ t in F ⋆ . For linear objective functions the limit µ ⋆ is the information projection of µ 0 to F ⋆ , see [5, Corollary 4.8]. We include a proof here for the sake of completeness.

Corollary 3.12 (Implicit bias of Fisher-Rao GF) . Consider Setting 3.1 and denote the face of maximizers of the linear program (3.1) by F ⋆ . Then it holds that

<!-- formula-not-decoded -->

In words, the Fisher-Rao gradient flow converges to the information projection of µ 0 to F ⋆ , i.e., it selects the optimizer that has the minimum KL-divergence from µ 0 .

Proof. By compactness of P , the sequence ( µ t n ) n ∈ N has at least one accumulation point for any t n → + ∞ . Hence, we can assume without loss of generality that µ t n → ˆ µ and it remains to identify ˆ µ as the information projection µ ⋆ ∈ F ⋆ .

Surely, we have ˆ µ ∈ F ⋆ as c ⊤ ˆ µ = lim n →∞ c ⊤ µ t n = max µ ∈ P c ⊤ µ by Corollary 2.3. Further, by the central path property we have for any optimizer µ ′ ∈ F ⋆ that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and can conclude by minimizing over µ ′ ∈ F ⋆

̸

. □

Theorem 3.13. Consider Setting 3.1, assume that the linear program is non-trivial, i.e., that F ⋆ = P , where F ⋆ denotes the face of optimizers, and denote the information projection of µ 0 to F ⋆ by µ ⋆ ∈ F ⋆ and set

<!-- formula-not-decoded -->

and therefore

Hence, we have

Then for any κ ∈ (0 , ∆) there is t κ ∈ R ≥ 0 such that for any t ≥ t κ we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Corollary 3.12 shows that µ t → µ ⋆ . Let µ ⋆ t ∈ F ⋆ denote the ∥·∥ TV -projection of µ t onto F ⋆ , i.e., be such that

<!-- formula-not-decoded -->

as µ t → µ ⋆ ∈ F ⋆ . Now we have

<!-- formula-not-decoded -->

and hence µ ⋆ t → µ ⋆ . Note that µ ⋆ ∈ int( F ⋆ ) , i.e., has maximal support in F ⋆ and hence µ ⋆ t ≪ µ ⋆ , see Lemma A.2. Together with µ ⋆ t → µ ⋆ this yields

<!-- formula-not-decoded -->

Combining Corollary 3.11 and Lemma 3.9 yields

<!-- formula-not-decoded -->

where the right hand side converges to ∆ -1 ( c ⊤ µ ⋆ -c ⊤ µ ) for t → + ∞ . Hence, for κ &lt; ∆ and t large enough, we have

<!-- formula-not-decoded -->

where we used that µ ⋆ is the information projection of µ t to F ⋆ and µ ⋆ t ∈ F ⋆ , therefore establishing (3.10). Now we can conclude utilizing Lemma 3.7. □

A bound on the time t κ could be obtained through a refinement of Lemma 3.9 showing c ⊤ µ ⋆ -c ⊤ µ ≥ ∆ · ∥ µ ⋆ -µ ∥ TV for the information projection µ ⋆ of µ ∈ P to F ⋆ . Another approach to control t κ is to quantify the convergence of µ ⋆ t → µ ⋆ .

Remark 3.14 (Estimating the regularization error) . Just like before, we can estimate the regularization error with the same argument as in Corollary 3.3. In this case, the guarantee (3.28) holds with the entropic radius R H instead of D KL ( µ ⋆ , µ 0 ) .

## 4. Convergence of Natural Gradient Flows

In practice, it is often not feasible to perform optimization in the space of measures, and therefore one often resorts to parametric models. Natural gradients were introduced by S. Amari [6] and are designed to mimic the Fisher-Rao gradient flow by preconditioning the Euclidean gradient in parameter space with the Fisher information matrix. To study natural gradient methods, we work in the following setting.

̸

Setting 4.1. We consider a finite set X and a polytope P = ∆ X ∩L with P ∩ R X &gt; 0 = ∅ , where L ⊆ R X is an affine space. Further, we consider a differentiable parametrization R p → int( P ); θ ↦→ µ θ and a (possibly nonlinear) differentiable objective function f : R X &gt; 0 → R , and write f ( θ ) = f ( µ θ ) .

We work in continuous time and consider the following evolution of parameters.

Definition 4.2 (Natural gradient flow) . Consider Setting 4.1. We call

<!-- formula-not-decoded -->

the natural gradient flow , where F ( θ ) + denotes the pseudo-inverse of the Fisher information matrix with entries

<!-- formula-not-decoded -->

and

4.1. Compatible function approximation. In this subsection, and more precisely in Proposition 4.4, we describe the natural gradient direction as the minimizer of a linear least squares regression problem with features ϕ θ ( x ) = ∇ θ log µ θ ( x ) . This can be used to estimate the natural gradient from samples drawn from µ θ .

In the context of reinforcement learning similar techniques, albeit for a different notion of natural gradient, have been developed under the name compatible function approximation [55, 26, 2].

The measure µ t = µ θ t does not necessarily evolve according to the Fisher-Rao gradient flow on the polytope P (2.6) even if θ t satisfies the natural gradient flow in the parameter space (4.1). In the next lemma we describe the discrepancy between ∂ t µ t = ∂ t θ ⊤ t ∇ θ µ θ t and the Fisher-Rao gradient ∇ FR P f ( µ t ) .

Lemma 4.3. Consider Setting 4.1 and a parameter evolution ∂ t θ t = v t and write µ t = µ θ t . Then we have where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is an l 2 -regression error and C ( θ t ) := inf ν ∈ TP ∥ ∥ ∇ FR f ( µ t ) -ν ∥ ∥ 2 g FR µ θ t a projection error.

Proof. The Fisher-Rao gradient ∇ FR P f ( µ t ) of f : P → R is the Fisher-Rao projection of the Fisher-Rao gradient ∇ FR f ( µ t ) of f : R X &gt; 0 → R onto TP . Hence, by the Pythagorean theorem, we have

<!-- formula-not-decoded -->

Since ∇ FR P f ( µ t ) is the projection of ∇ FR f ( µ t ) to TP , we obtain

<!-- formula-not-decoded -->

Further, by the chain rule, we have ∂ t µ t = ∂ t θ ⊤ t ∇ θ µ θ t ( x ) = v ⊤ t ∇ θ µ θ t ( x ) . Using ∇ FR f ( µ ) = ∇ f ( µ ) ⊙ µ we conclude

<!-- formula-not-decoded -->

The distance between ∂ t µ t and the Fisher-Rao gradient ∇ FR P f ( µ t ) is up to a remainder term given by the least squares loss L ( v t , θ t ) , where v t = ∂ t θ t . The natural gradient is designed such that ∂ t µ t is close to ∇ FR P f ( µ t ) [6] and hence we can minimize the least squares loss L ( v, θ t ) with respect to v in order to approximate the natural gradient v t ≈ F ( θ t ) + ∇ f ( θ t ) . An important benefit of this formulation is that it can be used to estimate the natural gradient from data distributed according to µ θ t . We make this relation between the minimization of L and the natural gradient explicit.

Proposition 4.4 (Compatible function approximation) . Consider Setting 4.1, let F ( θ ) denote the Fisher-information matrix, and let L be defined as in (4.4) . Then v ∈ R p is a natural gradient at θ ∈ R p , i.e., satisfies F ( θ ) v = ∇ θ f ( θ ) , if and only if

<!-- formula-not-decoded -->

Proof. The objective function L ( w,θ ) is given, up to a constant, by

<!-- formula-not-decoded -->

The global minimizes are characterized by the normal equation F ( θ ) w = ∇ f ( θ ) . □

The term

<!-- formula-not-decoded -->

is can be interpreted as an approximation error . Note, however, that the precise nature of the least square loss L is different from the one well-known in reinforcement learning as we discuss in more detail in Remark 5.6. Examining the objective L ( w,θ ) and using Lemma 4.3 we see that the natural gradient flow minimizes the discrepancy between ∂ t µ t and the Fisher-Rao gradient ∇ FR P f ( µ t ) . In this case, the evolution ∂ t µ t is given by the orthogonal projection of the FisherRao gradient onto the tangent space of the parametrized model. A similar property holds for any natural gradient defined using a Riemannian metric on the polytope [7, 57, 43].

Corollary 4.5 (Projection property) . Consider a solution ( θ t ) t ∈ [0 ,T ) of the natural gradient flow (4.1) . We denote the projection with respect to the Fisher-Rao metric onto the generalized tangent space

<!-- formula-not-decoded -->

by P FR θ . Then it holds that

<!-- formula-not-decoded -->

In particular, if T θ t P = TP then ∂ t µ t = ∇ FR P f ( µ t ) .

Proof. By Proposition 4.4 the natural gradient direction v t is a minimizer of L ( · , θ t ) . By Lemma 4.3 this yields

<!-- formula-not-decoded -->

In particular, this shows that ∂ t µ t is the projection of ∇ FR P f ( µ t ) onto T θ P

. □

4.2. Convergence of natural gradient flows. We start with a generalization of Corollary 2.3 to cover cases where the evolution of µ t only approximately follows the Fisher-Rao gradient flow.

Proposition 4.6 (A perturbed convergence result) . Consider Setting 3.1, a differentiable curve µ : [0 , ∞ ) → int( P ) and a differentiable convex objective f : R X &gt; 0 . Assume that f admits a maximizer µ ⋆ over P with value f ⋆ . It holds that

<!-- formula-not-decoded -->

where δ 2 t := χ 2 ( µ ⋆ , µ t ) and ε 2 t := ∥ ∥ ∇ FR P f ( µ t ) -∂ t µ t ∥ ∥ 2 g FR d t .

Proof. We compute

<!-- formula-not-decoded -->

where we used Lemma 4.3 and Proposition 4.4 as well as ∥ µ t -µ ⋆ ∥ 2 g FR µ t = χ 2 ( µ ⋆ , µ t ) . Integration and rearranging now yields (4.8). □

If ( µ t ) t ≥ 0 solves the Fisher-Rao gradient flow, we have ε t = 0 and recover Corollary 2.3. For natural gradient flows, we obtain the following result.

Corollary 4.7. Consider Setting 4.1 and a solution ( θ t ) t ≥ 0 of the natural gradient flow (4.1) for a convex objective f and set µ t := µ θ t . Then (4.8) holds with

<!-- formula-not-decoded -->

Proof. Combine Proposition 4.6 with Lemma 4.3 and Proposition 4.4. □

Remark 4.8 (Baseline) . In reinforcement learning, baselines are often used when estimating the natural policy gradient from samples to reduce the variance of the estimates [59] . This amounts to projecting the gradient of the objective to the tangent space of the model. In our setting, this corresponds to projecting ∇ f ( µ ) ⊙ µ to the tangent space TP with respect to the Fisher-Rao metric g FR µ , see also [54, Subsection 4.1.1] . In the special case P = ∆ X the Fisher-Rao projection of ∇ f ( µ ) ⊙ µ is given by ∇ f ( µ ) ⊙ µ -κµ , where κ = ∑ x ∇ f ( µ )( x ) and the corresponding compatible function approximation objective is given by

<!-- formula-not-decoded -->

4.3. Global convergence for multi-player games. With function approximation Corollary 4.7 ensures sublinear convergence O ( 1 t ) up to a remainder compared to the linear rate global convergence guarantee of the Fisher-Rao gradient flow. With general function approximation, it is however not possible to guarantee global convergence [13] and also for other natural gradient methods the linear convergence guarantees are lost when working with function approximation [3, 16] unless one uses regularization. Here, we identify a scenario, where despite being in a function approximation setting, we can ensure global linear convergence.

For a rich enough parametrization Corollary 4.5 ensures that ( µ θ t ) t ≥ 0 follows the Fisher-Rao gradient flow in which case Theorem 3.2 implies the linear convergence of the natural gradient flow (4.1). A common example is the softmax parametrization µ θ ( x ) ∝ e θ ( x ) . For multi-player games with suitable payoff structure, the dynamics of the individual players decouple [14], which allows us to show global convergence for models with exponentially fewer parameters than the softmax parametrization.

Theorem 4.9. Consider a differentiable parametrization of conditional probabilities { m θ : θ ∈ R p } = int(∆ n X ) , where n ∈ N and X is a finite set, and suppose that span { ∂ θ i m θ : i = 1 , . . . , p } = T ∆ n X for all θ ∈ R p . Define a corresponding parametric independence model as

<!-- formula-not-decoded -->

Further, consider

<!-- formula-not-decoded -->

and the linear payoff f ( µ ) = c ⊤ µ and the natural gradient flow (4.1) . Then ( µ θ t ) t ≥ 0 solves the Fisher-Rao gradient flow in ∆ X n and hence, we have

<!-- formula-not-decoded -->

Proof. The Segre embedding ∆ n X → ∆ X n , ( µ i ) i =1 ,...,n ↦→ ⊗ n i =1 µ i is an isometry with respect to the product Fisher-Rao metric, i.e., the sum of the Fisher metrics over the individual factors, and the Fisher-Rao metric [36, 14]. In particular, this implies that ( µ θ t ) t ≥ 0 solves the Fisher-Rao gradient flow with respect to f restricted the independence model

<!-- formula-not-decoded -->

as ∂ t µ t = P T µ t I ∇ FR f ( µ t ) = ∇ FR f | I ( µ t ) , see [57]. Condition (4.11) implies that f factorizes along the marginalization map and hence the independence model I is invariant under the Fisher-Rao gradient flow [14]. Thus, ( µ θ t ) t ≥ 0 solves the Fisher-Rao gradient flow with potential f in ∆ X n , which can be solved explicitly [60]. □

Note that a model parametrizing ∆ n X only requires n ( | X | -1) parameters, whereas a model parametrizing the joint distributions ∆ X n requires | X | n -1 parameters. However, we require the cost vector c to lie in an n | X | -dimensional subspace of R X n .

## 5. Convergence of State-Action Natural Policy Gradients

Having studied general linear programs we now turn to the reward optimization problem in infinite-horizon discounted Markov decision processes. Reward optimization is well known to be equivalent to a linear program and the state-action natural policy gradient flow corresponds to the Fisher-Rao gradient flow inside the state-action polytope [27, 42]. We give a short overview of the required notions and refer to [24] for a thorough introduction to Markov decision processes.

In Markov decision processes (MDPs), we are concerned with controlling the state s ∈ S of some system through an action a ∈ A in order to achieve an optimal behavior over time. The evolution of the system is described by a Markov kernel P ∈ ∆ S × A S , where P ( s ′ | s, a ) denotes the probability of transitioning from state s to s ′ under action a . Here, we work with finite state and action spaces S and A . A (stochastic) policy is a Markov kernel π ∈ ∆ S A , where π ( a | s ) denotes the probability of selecting action a when in state s . For a fixed policy π ∈ ∆ S A and an initial distribution µ ∈ ∆ S we obtain a Markov process over S × A according to S 0 ∼ µ and

<!-- formula-not-decoded -->

and we denote its law by P π,µ . We consider a instantaneous reward vector r ∈ R S × A indicating how favorable a certain state and action combination is. As a criterion for the performance of a policy π we consider the infinite horizon discounted reward

<!-- formula-not-decoded -->

where the discount factor γ ∈ [0 , 1) is fixed and ensures convergence. The reward optimization problem is given by

<!-- formula-not-decoded -->

An important role in Markov decision processes play the state-action distributions d π ∈ ∆ S × A , which are given by

<!-- formula-not-decoded -->

They determine the reward as R ( π ) = ∑ s ∈ S ,a ∈ A r ( s, a ) d π ( s, a ) = r ⊤ d π . The set of state-action distributions has been characterized as a polytope, see [20].

Proposition 5.1 (State-action polytope) . The set D = { d π : π ∈ ∆ S A } ⊆ ∆ S × A of state-action distributions is a polytope given by

<!-- formula-not-decoded -->

where the defining linear equations are given by

<!-- formula-not-decoded -->

We refer to D as the state-action polytope . This leads to the linear programming formulation of Markov decision processes [27], given by 1

<!-- formula-not-decoded -->

1 Sometimes, this is referred to as the dual linear programming formulation of Markov decision processes, where the primal linear program has the optimal value function as its solution.

The state-action polytope D = ∆ S × A ∩L falls under the class of polytopes studied in Section 3. Given a state-action distribution d ∈ D , we can compute a corresponding policy π ∈ ∆ S A with d = d π by conditioning,

<!-- formula-not-decoded -->

if this is well-defined, see [41, 31], which leads us to the following assumption.

Assumption 5.2 (State exploration) . For any policy π ∈ ∆ S A the discounted state distribution is positive, i.e., ∑ a ∈ A d π ( s, a ) &gt; 0 for all s ∈ S .

This assumption is satisfied if µ ( s ) &gt; 0 for all s ∈ S as ∑ a ∈ A d π ( s, a ) ≥ (1 -γ ) µ ( s ) . This assumption is standard in linear programming approaches to Markov decision processes; policy gradient methods can fail to converge if it is violated [27, 35].

Policy optimization algorithms parameterize the policy π θ and optimize θ . As we study gradient-based approaches we work under the following assumption.

Assumption 5.3 (Differentiable parametrization) . We consider a differentiable policy parametrization R p → int(∆ S A ); θ ↦→ π θ .

We consider continuous-time natural policy gradient methods that optimize the parameters θ of a parametric policy π θ according to

<!-- formula-not-decoded -->

where we write R ( θ ) = R ( π θ ) . Here G ( θ ) denotes a Gramian matrix with entries G ( θ ) ij = g d θ ( ∂ θ i d θ , ∂ θ j d θ ) , where we write d θ = d π θ and g d denotes a Riemannian metric on the stateaction polytope D . In this context, the matrix G ( θ ) is referred to as a preconditioner. Various choices have been proposed for G ( θ ) , for example Kakade [26] suggested

<!-- formula-not-decoded -->

which is a weighted sum of Fisher-information matrices over the individual states [26, 9, 46]. This has been studied extensively in the literature, see for example [2, 17, 16, 29], and we refer to it as the Kakade NPG . We focus on the so-called state-action natural policy gradient given by the Fisher information matrix of the state-action distribution [37],

<!-- formula-not-decoded -->

This choice was observed to reduce the severity of plateaus, was used to design a natural actorcritic method [38], and is closely connected to the trust region method known as relative entropy policy search (REPS) [45].

5.1. Convergence guarantees. Now that we have built a convergence theory for general natural gradient flows we elaborate on the consequences for state-action natural policy gradients.

Corollary 5.4 (Sublinear convergence under function approximation) . Consider a finite discounted Markov decision process, suppose Assumption 5.2 and Assumption 5.3 hold, and consider a solution of the natural policy gradient flow (5.9) for G = G M and set R ⋆ := max π ∈ ∆ S A R ( π ) . Then it holds that

<!-- formula-not-decoded -->

where δ 2 t := χ 2 ( d ⋆ , d t ) and ε 2 t := min w ∈ R p ∥ ∥ ∇ FR D f ( µ t ) -w ⊤ ∇ θ d θ t ∥ ∥ 2 g FR d t and

<!-- formula-not-decoded -->

Proof. This is Corollary 4.7 for state-action natural policy gradients.

□

Remark 5.5 (Inexact gradient evaluations) . If the parameters follow the evolution ∂ t θ t = v t , then we can apply Proposition 4.6 to see that (5.12) remains valid with

<!-- formula-not-decoded -->

Remark 5.6 (Comparison to Kakade's natural policy gradient) . For Kakade's natural policy gradient in discrete time without entropy regularization in the function approximation regime, the value converges as O ( 1 t ) up to a remainder stemming from function approximation [4] . Compared to (5.12) , the O ( 1 t ) involves a conditional KL term corresponding to the Kakade geometry, which is induced by the conditional entropy rather than the entropy. More importantly, however, it comes with a multiplicative distribution mismatch coefficient, where it is unclear whether it remains bounded during optimization. However, it is unclear whether this is inherent to Kakade's natural policy gradient or an artifact of the proof. The remainder term in [4] again depends on the distribution mismatch and on a concentrability coefficient similar to χ 2 ( d ⋆ , d t ) . Another difference between Kakade's and state-action natural policy gradients is that the compatible function approximation regresses the (estimated) Q or advantage function instead of the reward vector r , therefore leading to a different approximation error ˜ ε t . Further, Kakade's natural policy gradient without entropy regularization in a function approximation setting has been shown to converge linearly when using geometrically increasing step sizes [61, 3, 62] .

Finally, entropy regularization with strength λ leads to O ( e -λt ) convergence up to a remainder term, where the same χ 2 -divergence appears in the remainder term albeit with a different approximation error term [16] .

We have studied general policy parameterizations and have seen that the corresponding stateaction distributions evolve according to the projection of the Fisher-Rao gradient flow. A particularly nice case is given by parameterizations that are rich enough to express all policies as in this case the state-action distributions exactly evolve according to the Fisher-Rao gradient flow. This is why we consider the following condition for policy parameterizations.

Definition 5.7 (Regular tabular parametrization) . We say that a differentiable parametrization R p → int(∆ S A ); θ ↦→ π θ is a regular tabular parametrization if it is surjective and satisfies

<!-- formula-not-decoded -->

Since π ↦→ d π is a diffeomorphism between int(∆ S A ) and int( D ) , see [42], we have

<!-- formula-not-decoded -->

for a regular policy parametrization.

Regular parametrizations include the following common examples:

- Expressive exponential families: For a feature map ϕ : S × A → R p and θ ∈ R p we consider the log-linear policy π θ ( a | s ) ∝ e θ ⊤ ϕ ( s,a ) . This provides a regular tabular parametrization if rank { ϕ ( s, a ) : s ∈ S , a ∈ A } = | S | · | A | , see [50, Remark 2.4]. In particular, this includes tabular softmax policies, where π θ ( a | s ) ∝ e θ s,a which is the arguably most commonly studied policy class.
- Escort transform: The so-called escort transform π θ ( a | s ) ∝ | θ s,a | p , for a parameter p ≥ 1 was introduced in [34] to reduce the plateaus of vanilla policy gradients when working with softmax policies.

For regular tabular parameterizations, the state-action distributions d t evolve according to the Fisher-Rao gradient flow inside the state-action polytope D . Hence, we can apply our general convergence theory to obtain the following result.

Corollary 5.8 (Linear convergence for tabular parametrizations) . Consider a finite discounted Markov decision process, suppose Assumption 5.2 and Assumption 5.3 hold, and consider a solution of the natural gradient flow (5.9) for a regular tabular parametrization and write µ t = d θ t . Then ( µ t ) t ≥ 0 solves the Fisher-Rao gradient flow of the linear program (5.7) and hence

Theorem 3.2 and Theorem 3.13 hold. This implies O ( e -∆ t + κ log t ) convergence for some κ ≥ 0 , where

<!-- formula-not-decoded -->

and R ⋆ = max π ∈ ∆ S A R ( π ) denotes the optimal reward.

Proof. The neighbors in D and ∆ S A correspond to each other [41]. Hence, d π is a neighbor of d ⋆ if π is deterministic and agrees with π ⋆ on all but one state. □

̸

Remark 5.9 (Comparison to Kakade's NPG) . Much like the state-action natural policy gradient, Kakade's natural policy gradient with exact gradient evaluations has been shown to converge linearly without the need for entropy regularized setting [29] . Here, the discrete-time setting is studied and NPG is interpreted as soft policy iteration. This is used to show a convergence rate of R ⋆ -R ( π k ) = O ( e -ck ) for any c ∈ (0 , ∆ K ) , where ∆ K := -(1 -γ ) -1 max { A ⋆ ( s, a ) : a = a ⋆ s } ≥ ∆ , where a ⋆ s denotes the optimal action in state s . Indeed, by the performance difference lemma, we have

<!-- formula-not-decoded -->

̸

Note that d ∈ N ( d ⋆ ) can be associated with a policy π that agrees with π ⋆ on all but one state, and we write π ( a 0 | s 0 ) = 1 for a 0 = a ⋆ s 0 for some s 0 and π ( a ⋆ s | s ) = 1 for s = s 0 . Since A ⋆ ( s, a ⋆ s ) = 0 we have d T A ⋆ = d ( s 0 ) A ⋆ ( s 0 , a 0 ) ≤ 0 and estimate

̸

̸

<!-- formula-not-decoded -->

̸

Overall, this yields d ⊤ A ⋆ ∥ d ⋆ -d ∥ TV ≥ A ⋆ ( s 0 , a 0 ) and therefore

<!-- formula-not-decoded -->

̸

Hence, the guaranteed convergence rate of Kakade's NPG is faster compared to the rate we provide for the state-action natural policy gradient. An essentially matching lower bound has been established for Kakade's NPG in [40] , whereas a matching lower bound is missing for state-action natural policy gradients. In our computational example, both converge at the same exponential rate O ( e -∆ K t ) even if ∆ &lt; ∆ K .

Remark 5.10 (Implicit bias) . In particular, Corollary 5.8 guarantees that in the case of multiple optimal policies, the gradient flows corresponding to state-action natural policy gradients converge exponentially fast towards the information projection d ⋆ of d π 0 to the set of maximizes D ⋆ = { d ∈ D : r ⊤ d = R ⋆ } ⊆ D . This shows that state-action natural gradients not only optimize the reward but produce the policy that induces a state-action distribution with maximal entropy with respect to the initial state-action distribution d π 0 . This characterizes the implicit bias of state-action natural policy gradients. Prior, the implicit bias of a natural actor-critic method has been analyzed by [25] , where they provided an O (log k ) bound on the optimal policy with maximal (weighted) entropy over the states. Note that as this bound grows with the number of iterations k it can't identify the limiting policy. Further, by using the reformulation as a Hessian gradient flow from [42] and results from convex optimization [5] convergence towards the (generalized) maximal entropy policy for Kakade's natural policy gradient has been established in [40] .

Remark 5.11 (Comparison to previous rates) . In Corollary 5.8 we provide linear convergence with exponent ∆ and have discussed in Remark 3.4 this exponent improves on previously established O ( e -δt ) guarantee in [60, 54] . To see this, we note that d π 1 , d π 2 ∈ D are neighboring vertices if and only if π 1 , π 2 ∈ ∆ S A are neighbors [41] . Two policies are neighboring if and only

if they are deterministic and agree on all but one state. Hence, if we consider an MDP with more than one state, there is at least one state s ∈ S such that π 1 ( a | s ) = π 2 ( a | s ) = 1 for some a ∈ A and therefore d π 1 ( s, a ) = d π 1 ( s ) &gt; 0 and d π 2 ( s, a ) = d π 2 ( s ) &gt; 0 . Hence, d π 1 , d π 2 ∈ D do not have disjoint support and as elaborated in Remark 3.4 this implies δ &lt; ∆ . Overall, this shows that for an exploratory MDP with more than one state, we have δ &lt; ∆ , meaning that our convergence rate improves upon [60, 54] .

5.2. Computational examples. We use an example from [26, 9, 37] of an MDP with two states s 1 , s 2 and two actions a 1 , a 2 , with the transitions and instantaneous rewards shown in Figure 2. Wemake our code available under https://github.com/muellerjohannes/fisher-rao-GFs-LPs .

Figure 2. Transition graph and reward of the MDP example.

<!-- image -->

We adopt the initial distribution µ ( s 1 ) = 0 . 8 , µ ( s 2 ) = 0 . 2 and work with a discount factor of γ = 0 . 9 . We can explicitly compute the rewards of the four deterministic policies to be R 1 = 0 . 98 , R 2 = 1 . 2 , R 3 = 1 . 84 and R 4 = 0 , and this way determine the optimal policy. Consequently, we can compute the exponent ∆ given in Corollary 5.8 to be ∆ = 0 . 8 . In contrast, the exponent δ given in [60, 54] is δ = 0 . 64 . In Remark 3.4 we observed that δ ≤ ∆ , and this now provides an explicit example where δ &lt; ∆ . Finally, we compute the constant ∆ K = 0 . 8 that describes the exponent in the convergence rate of Morimura's natural policy gradient [29].

To illustrate our theoretical findings, we run both state-action natural gradients as well as Kakade's natural policy gradient applied to a tabular soft-max parametrization for 30 random initializations. In order to prevent a blow-up of the parameters we use the update rule

<!-- formula-not-decoded -->

with stepsize η &gt; 0 , where we choose η = 10 -2 in our experiments. Intuitively, we expect θ k ≈ ˜ θ ηk if ( ˜ θ t ) t ≥ 0 solves the natural policy gradient flow.

5.2.1. A first example with tightness. Figure 3 plots the suboptimality gap R ⋆ -R ( θ k ) as well as the KL-divergence D KL ( d ⋆ , d θ k ) for the two different natural policy gradient methods and the same 30 random initializations. Additionally, the gray dashed line indicates the exponential decay rate O ( e -∆ ηk ) = O ( e -∆ K ηk ) guaranteed by Corollary 5.8 and by [29], respectively. We see that for all trajectories both the suboptimality gap R ⋆ -R ( θ k ) as well as the KL-divergence to the optimal state-action distribution D KL ( d ⋆ , d θ k ) decay at this guaranteed rate for both the state-action and Kakade's natural policy gradient method.

5.2.2. A second example and non-tightness. We complement our computational example by studying the same Markov decision process from Figure 2 but changing the reward vector according to r ( s 1 , a 2 ) = 3 . As above we can compute the three constants δ , ∆ and ∆ K , and obtain δ ≈ 0 . 5326 , ∆ ≈ 0 . 5789 and ∆ K = 1 . 1 . Here again δ &lt; ∆ as it is always guaranteed under the Assumption 5.2, see Remark 3.4. Further, in this example, we have ∆ &lt; ∆ K . We conduct the same experiment as before and report the findings in Figure 4. In the plots concerning the stateaction natural policy gradient, we plot both the guaranteed decay O ( e -∆ ηk ) (gray dashed line) and the decay O ( e -∆ K ηk ) guaranteed for Kakade's natural policy gradient (gray dotted line). We see that both methods exhibit the convergence rate O ( e -∆ K ηk ) . In particular, this indicates that our convergence analysis of the Fisher-Rao gradient, although improving on known results, is still not tight for general problems.

## 6. Conclusion and Outlook

We study Fisher-Rao gradient flows of linear programs and show they converge linearly with an exponent that depends on the geometry of the linear program. This yields an estimate on the error introduced by entropic regularization of the linear program, which improves existing

Figure 3. Shown are the suboptimality gap R ⋆ -R ( θ t ) (top row) and the KLdivergence D KL ( d ⋆ , d t ) (bottom row) for the state-action NPG (left column) and Kakade's NPG (right column) plotted in a logarithmic scale, along with the predicted exponential decay e -∆ ηk = e -∆ K ηk (dashed line), see Corollary 5.8 and [29] for state-action and Kakade's NPG, respectively.

<!-- image -->

Figure 4. Shown are the suboptimality R ⋆ -R ( θ k ) (top) and KL-divergence D KL ( d ⋆ , d θ k ) (bottom) for the state-action NPG (left) and Kakade's NPG (right); shown are also the guaranteed exponential decay rates e -∆ ηk for the state-action NPG (dashed line) and e -∆ K ηk for Kakade's NPG (dotted line). Although the guarantees are different, both methods exhibit the same fast decay rate.

<!-- image -->

guarantees. We extend this analysis to natural gradient flows for general parametrized measure models and show they converge at a sublinear rate O ( 1 t ) up to an approximation error and mismatch of the trajectory to the solution measure in the χ 2 -divergence. In particular, our results yield O ( 1 t ) convergence of state-action natural policy gradients without regularization under function approximation and linear convergence of state-action natural policy gradients for general tabular parametrizations. Finally, we provide computational examples illustrating our results.

Our results improve previous results, but some further improvements may be possible. In particular, we use the best global constant ∆ &gt; 0 for which the estimate (3.15) holds for all µ ∈ P . However, if one can improve this constant along the trajectory ( µ t ) t ≥ 0 this would directly imply an improvement of the convergence rate. A natural way to approach this is to characterize the direction from which the flow ( µ t ) t ≥ 0 is approaching the global optimizer µ ⋆ . Another interesting direction for future work is to study the statistical complexity of the stateaction natural policy gradients. Finally, it could be explored whether our convergence results can be used in order to modify the cost to achieve a faster convergence without changing the optimizer, which is known as reward shaping in the context of reinforcement learning.

## Acknowledgments

The project originated when JM was a PhD student at the International Max Planck Research School Mathematics in the Sciences at MPI MiS with additional support from the Evangelisches Studienwerk Villigst e.V. . SC and JM acknowledge funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under the project number 442047500 through the Collaborative Research Center Sparsity and Singular Structures (SFB 1481). GM has been supported in part by NSF CAREER 2145630, NSF 2212520, DFG SPP 2298 project 464109215, ERC 757983, and BMBF in DAAD project 57616814.

## Appendix A. Auxiliary results

Lemma A.1. Consider a polytope P ⊆ R X , a face F ⊆ P and consider the cone

<!-- formula-not-decoded -->

which is generated by the edges pointing out of F . Then we have P ⊆ F + C .

Proof. This is a generalization of [64, Lemma 3.6], which covers the case that F consists of a single vertex. We will show that

̸

<!-- formula-not-decoded -->

for which we pick an element u ∈ ˜ P . Consider a hyperplane H = { µ : a ⊤ µ = α } separating F and vert( P ) \ F and consider the face figure P/F := P ∩ H , which is a polytope. Now, we consider a translation ˜ H = { µ : a ⊤ µ = β } of H , such that u ∈ ˜ H . Now we have

<!-- formula-not-decoded -->

see [65, Proposition 2.30]. Hence, we can choose convex weights λ i such that

<!-- formula-not-decoded -->

where µ i ∈ vert( F ) , ν i ∈ N ( µ i ) \ F . □

Lemma A.2 (Information projections have maximal support) . Consider a polytope P = ∆ X ∩ L for an affine space L and a face F of P . Further, let ˆ µ ∈ F be the information projection of µ ∈ int( P ) to F , then ˆ µ ∈ int( F ) .

Proof. Note that ˆ µ ∈ F is characterized by D KL (ˆ µ, µ ) = min µ ′ ∈ F D KL ( µ ′ , µ ) . Assume that ˆ µ ∈ ∂F , then µ x 0 = 0 for some x 0 ∈ X . Consider now v ∈ R X such that ˆ µ + tv ∈ int( F ) for t &gt; 0 small enough, then surely v x 0 &gt; 0 . By convexity of the KL-divergence, we have D KL (ˆ µ, µ ) ≥ D KL (ˆ µ + tv, µ ) + t∂ t D KL (ˆ µ + tv, µ ) , where ∂ t D KL (ˆ µ + tv, µ ) →-∞ for t → 0 . This shows D KL (ˆ µ, µ ) &gt; D KL (ˆ µ + tv, µ ) for t small enough contradicting that ˆ µ is the information projection of µ . □

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774 , 2023.
- [2] Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. The Journal of Machine Learning Research , 22(1):4431-4506, 2021.
- [3] Carlo Alfano and Patrick Rebeschini. Linear convergence for natural policy gradient with log-linear policy parametrization. arXiv preprint arXiv:2209.15382 , 2022.
- [4] Carlo Alfano, Rui Yuan, and Patrick Rebeschini. A novel framework for policy mirror descent with general parameterization and linear convergence. Advances in Neural Information Processing Systems , 36, 2023.
- [5] Felipe Alvarez, Jérôme Bolte, and Olivier Brahic. Hessian Riemannian gradient flows in convex programming. SIAM journal on control and optimization , 43(2):477-501, 2004.
- [7] Shun-ichi Amari. Information geometry and its applications , volume 194. Springer, 2016.
- [6] Shun-ichi Amari. Natural gradient works efficiently in learning. Neural computation , 10(2):251-276, 1998.
- [8] Nihat Ay, Jürgen Jost, Hông Vân Lê, and Lorenz Schwachhöfer. Information geometry , volume 64. Springer, 2017.
- [9] J. Andrew Bagnell and Jeff Schneider. Covariant policy search. In Proceedings of the 18th International Joint Conference on Artificial Intelligence , IJCAI'03, page 1019-1024, San Francisco, CA, USA, 2003. Morgan Kaufmann Publishers Inc.
- [10] Heinz H Bauschke, Jérôme Bolte, Jiawei Chen, Marc Teboulle, and Xianfu Wang. On linear convergence of non-euclidean gradient methods without strong convexity and lipschitz gradient continuity. Journal of Optimization Theory and Applications , 182:1068-1087, 2019.
- [11] Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław Dębiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, et al. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680 , 2019.
- [12] Jalaj Bhandari and Daniel Russo. On the linear convergence of policy gradient methods for finite mdps. In International Conference on Artificial Intelligence and Statistics , pages 2386-2394. PMLR, 2021.
- [13] Jalaj Bhandari and Daniel Russo. Global optimality guarantees for policy gradient methods. Operations Research , 2024.
- [14] Bastian Boll, Jonas Cassel, Peter Albers, Stefania Petra, and Christoph Schnörr. A geometric embedding approach to multiple games and multiple populations. arXiv preprint arXiv:2401.05918 , 2024.
- [15] Stephen P Boyd and Lieven Vandenberghe. Convex optimization . Cambridge university press, 2004.
- [16] Semih Cayci, Niao He, and Rayadurgam Srikant. Convergence of entropy-regularized natural policy gradient with linear function approximation. SIAM Journal on Optimization , 34(3):2729-2755, 2024.
- [17] Shicong Cen, Chen Cheng, Yuxin Chen, Yuting Wei, and Yuejie Chi. Fast global convergence of natural policy gradient methods with entropy regularization. Operations Research , 2021.
- [18] NNČencov. Algebraic foundation of mathematical statistics. Statistics: A Journal of Theoretical and Applied Statistics , 9(2):267-276, 1978.
- [19] Roberto Cominetti and J San Martín. Asymptotic analysis of the exponential penalty trajectory in linear programming. Mathematical Programming , 67:169-187, 1994.
- [20] Cyrus Derman. Finite state Markovian decision processes . Academic Press, Inc., 1970.
- [21] Travis Dick, Andras Gyorgy, and Csaba Szepesvari. Online learning in markov decision processes with changing cost sequences. In International Conference on Machine Learning , pages 512-520. PMLR, 2014.
- [22] Dongsheng Ding, Kaiqing Zhang, Tamer Basar, and Mihailo Jovanovic. Natural policy gradient primal-dual method for constrained Markov decision processes. Advances in Neural Information Processing Systems , 33:8378-8390, 2020.
- [23] Dongsheng Ding, Kaiqing Zhang, Jiali Duan, Tamer Başar, and Mihailo R Jovanović. Convergence and sample complexity of natural policy gradient primal-dual methods for constrained MDPs. arXiv preprint arXiv:2206.02346 , 2022.
- [24] Onésimo Hernández-Lerma and Jean B Lasserre. Discrete-time Markov control processes: basic optimality criteria , volume 30. Springer Science &amp; Business Media, 2012.
- [25] Yuzheng Hu, Ziwei Ji, and Matus Telgarsky. Actor-critic is implicitly biased towards high entropy optimal policies. arXiv preprint arXiv:2110.11280 , 2021.
- [26] Sham M Kakade. A natural policy gradient. Advances in neural information processing systems , 14, 2001.

- [27] Lodewijk CM Kallenberg. Survey of linear programming for standard and nonstandard markovian control problems. part i: Theory. Zeitschrift für Operations Research , 40:1-42, 1994.
- [28] Bekzhan Kerimkulov, James-Michael Leahy, David Siska, Lukasz Szpruch, and Yufei Zhang. A FisherRao gradient flow for entropy-regularised Markov decision processes in Polish spaces. arXiv preprint arXiv:2310.02951 , 2023.
- [29] Sajad Khodadadian, Prakirt Raj Jhunjhunwala, Sushil Mahavir Varma, and Siva Theja Maguluri. On linear and super-linear convergence of natural policy gradient algorithm. Systems &amp; Control Letters , 164:105214, 2022.
- [30] Guanghui Lan. Policy mirror descent for reinforcement learning: Linear convergence, new sampling complexity, and generalized problem classes. Mathematical programming , pages 1-48, 2022.
- [31] Romain Laroche and Remi Tachet Des Combes. On the occupancy measure of non-markovian policies in continuous mdps. In International Conference on Machine Learning , pages 18548-18562. PMLR, 2023.
- [32] Haoya Li, Samarth Gupta, Hsiangfu Yu, Lexing Ying, and Inderjit Dhillon. Approximate newton policy gradient algorithms. SIAM Journal on Scientific Computing , 45(5):A2585-A2609, 2023.
- [33] Haihao Lu, Robert M Freund, and Yurii Nesterov. Relatively smooth convex optimization by first-order methods, and applications. SIAM Journal on Optimization , 28(1):333-354, 2018.
- [34] Jincheng Mei, Chenjun Xiao, Bo Dai, Lihong Li, Csaba Szepesvári, and Dale Schuurmans. Escaping the gravitational pull of softmax. Advances in Neural Information Processing Systems , 33:21130-21140, 2020.
- [35] Jincheng Mei, Chenjun Xiao, Csaba Szepesvari, and Dale Schuurmans. On the Global Convergence Rates of Softmax Policy Gradient Methods. In International Conference on Machine Learning , pages 6820-6829. PMLR, 2020.
- [36] Guido Montúfar, Johannes Rauh, and Nihat Ay. On the Fisher metric of conditional probability polytopes. Entropy , 16(6):3207-3233, 2014.
- [37] Tetsuro Morimura, Eiji Uchibe, Junichiro Yoshimoto, and Kenji Doya. A new natural policy gradient by stationary distribution metric. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2008, Antwerp, Belgium, September 15-19, 2008, Proceedings, Part II 19 , pages 82-97. Springer, 2008.
- [38] Tetsuro Morimura, Eiji Uchibe, Junichiro Yoshimoto, and Kenji Doya. A generalized natural actor-critic algorithm. Advances in neural information processing systems , 22, 2009.
- [39] Johannes Müller. Geometry of Optimization in Markov Decision Processes and Neural Network Based PDE Solvers . PhD thesis, University of Leipzig, 2023.
- [40] Johannes Müller and Semih Cayci. Essentially Sharp Estimates on the Entropy Regularization Error in Discrete Discounted Markov Decision Processes. arXiv preprint arXiv:2406.04163 , 2024.
- [41] Johannes Müller and Guido Montúfar. The Geometry of Memoryless Stochastic Policy Optimization in Infinite-Horizon POMDPs. In International Conference on Learning Representations , 2022.
- [42] Johannes Müller and Guido Montúfar. Geometry and convergence of natural policy gradient methods. Information Geometry , 7(1):485-523, 2024.
- [43] Johannes Müller and Marius Zeinhofer. Achieving High Accuracy with PINNs via Energy Natural Gradient Descent. In International Conference on Machine Learning , pages 25471-25485. PMLR, 2023.
- [44] Gergely Neu, Anders Jonsson, and Vicenç Gómez. A unified view of entropy-regularized Markov decision processes. arXiv preprint arXiv:1705.07798 , 2017.
- [46] Jan Peters and Stefan Schaal. Natural actor-critic. Neurocomputing , 71(7-9):1180-1190, 2008.
- [45] Jan Peters, Katharina Mulling, and Yasemin Altun. Relative entropy policy search. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 24, pages 1607-1612, 2010.
- [47] Gabriel Peyré, Marco Cuturi, et al. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning , 11(5-6):355-607, 2019.
- [49] C Radhakrishna Rao. Differential metrics in probability spaces. In Differential geometry in statistical inference , volume 10, pages 217-241. Institute of Mathematical Statistics, 1987.
- [48] C Radhakrishna Rao. Information and accuracy attainable in the estimation of statistical parameters. Bulletin of the Calcutta Mathematical Society , 37(3):81-91, 1945.
- [50] Johannes Rauh. Finding the Maximizers of the Information Divergence from an Exponential Family thesis, University of Leipzig, 2011.
25. . PhD
- [51] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning , pages 1889-1897. PMLR, 2015.
- [52] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- [53] Bernd Sturmfels, Simon Telen, François-Xavier Vialard, and Max von Renesse. Toric geometry of entropic regularization. Journal of Symbolic Computation , 120:102221, 2024.
- [54] Felipe Suárez Colmenares. Perspectives on Geometry and Optimization: from Measures to Neural Networks . PhD thesis, Massachusetts Institute of Technology, 2023.
- [55] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems , 12, 1999.

- [56] Gerald Teschl. Ordinary differential equations and dynamical systems , volume 140. American Mathematical Society, 2024.
- [57] Jesse van Oostrum, Johannes Müller, and Nihat Ay. Invariance properties of the natural gradient in overparametrised systems. Information geometry , 6(1):51-67, 2023.
- [59] Lex Weaver and Nigel Tao. The optimal reward baseline for gradient-based reinforcement learning. In Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence , UAI'01, page 538-545, San Francisco, CA, USA, 2001. Morgan Kaufmann Publishers Inc.
- [58] Li Wang and Ming Yan. Hessian informed mirror descent. Journal of Scientific Computing , 92(3):90, 2022.
- [60] Jonathan Weed. An explicit analysis of the entropic penalty in linear programming. In Conference On Learning Theory , pages 1841-1855. PMLR, 2018.
- [61] Lin Xiao. On the convergence rates of policy gradient methods. Journal of Machine Learning Research , 23(282):1-36, 2022.
- [62] Rui Yuan, Simon Shaolei Du, Robert M. Gower, Alessandro Lazaric, and Lin Xiao. Linear convergence of natural policy gradient methods with log-linear policies. In The Eleventh International Conference on Learning Representations , 2023.
- [63] Wenhao Zhan, Shicong Cen, Baihe Huang, Yuxin Chen, Jason D Lee, and Yuejie Chi. Policy mirror descent for regularized reinforcement learning: A generalized framework with linear convergence. SIAM Journal on Optimization , 33(2):1061-1091, 2023.
- [64] Günter M Ziegler. Lectures on polytopes , volume 152. Springer Science &amp; Business Media, 2012.
- [65] Günter M Ziegler. Lecture notes: Discrete Geometry I, 2013.
- [66] Alexander Zimin and Gergely Neu. Online learning in episodic markovian decision processes by relative entropy policy search. Advances in neural information processing systems , 26, 2013.