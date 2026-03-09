## Why Target Networks Stabilise Temporal Difference Methods

Mattie Fellows * 1 Matthew J.A. Smith * 1 Shimon Whiteson 1

## Abstract

Integral to many recent successes in deep reinforcement learning has been a class of temporal difference methods that use infrequently updated target values for policy evaluation in a Markov Decision Process. At the same time, a complete theoretical explanation for the effectiveness of target networks remains elusive. In this work, we provide an analysis of this popular class of algorithms, to finally answer the question: 'why do target networks stabilise TD learning'? To do so, we formalise the notion of a partially fitted policy evaluation method, which describes the use of target networks and bridges the gap between fitted methods and semigradient temporal difference algorithms. Using this framework we are able to uniquely characterise the so-called deadly triad -the use of TD updates with (nonlinear) function approximation and off-policy datawhich often leads to nonconvergent algorithms. This insight leads us to conclude that the use of target networks can mitigate the effects of poor conditioning in the Jacobian of the TD update. Furthermore, we show that under mild regularity conditions and a well tuned target network update frequency, convergence can be guaranteed even in the extremely challenging off-policy sampling and nonlinear function approximation setting.

## 1. Introduction

Since their introduction in deep Q -networks (DQN) a decade ago (Mnih et al., 2013; 2015), target networks have become a common feature of state-of-the-art deep reinforcement learning algorithms (Lillicrap et al., 2016; Haarnoja et al., 2017; 2018; Fujimoto et al., 2018). Theoretical analysis of target networks has been limited and there has been no satisfactory explanation for their empirical success in stabilising policy evaluation algorithms. Whilst recent analysis

* Equal contribution 1 Department of Computer Science, University of Oxford,Oxford, United Kingdom. Correspondence to: Mattie Fellows &lt;matthew.fellows@cs.ox.ac.uk&gt;.

Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute. Copyright 2023 by the authors.

has characterised the convergence properties of policy evaluation using target networks (Lee &amp; He, 2019; Fan et al., 2020; Zhang et al., 2021), existing approaches focus on asymptotic results, and usually make simplifying assumptions that neither hold in practice nor account for the true behaviour of target network-based updates. Our work finds that the use of target networks can guarantee that deep RL algorithms will not diverge, even in regimes where traditional RL algorithms fail. Additionally, we establish the first finite-time performance bounds for target networks and general function approximation-without strong simplifying assumptions. Moreover, we prove our key stability assumption can always be satisfied by augmenting our updates with simple ℓ 2 regularisation that does not change the TD fixed points. In doing so, we finally provide theoretical justification for the empirical success that has been observed in challenging, off-policy tasks.

To achieve this, we analyse the use of infrequently updated target value functions by characterising them as a family of methods that we refer to as partially fitted policy evaluation (PFPE). This variant bridges the gap between fitted policy evaluation (FPE) (Le et al., 2019)-which iteratively fit the Bellman backups onto the class of representable function approximators -and classic temporal difference (TD) algorithms (Sutton, 1988) by limiting the fitting phase to a fixed number of steps, precisely reflecting the periodically updated target network algorithms as used in practice.

To characterise the performance of PFPE, we express our algorithm-which has traditionally been viewed through the lens of two-timescale analysis-using a single update applied only to the target network parameters. We show that the stability of the algorithm is determined by analysing the eigenvalues of the Jacobian of this update. This formulation allows us to characterise both the limiting (asymptotic) and finite-time (non-asymptotic) convergence properties of PFPE. Furthermore, it suggests, counterintuitively, that target networks are actually the object being optimised rather than merely a means to stabilise conventional TD updates. This insight leads us to empirically investigate a novel target parameter update scheme that uses a momentum-style update (Polyak, 1964), setting the stage for future research of practical target-based algorithms.

Our bounds on the finite-time performance of PFPE apply to off-policy, nonlinear and partially fitted methods, which

have never been investigated previously. We develop key insights into the usefulness of target networks, which we find do not improve asymptotic performance when decaying step sizes are used. Instead, target networks improve the conditioning of TD and fitted methods when the step size does not tend to zero , as is often implemented in practice. Under non-decaying stepsizes, our Jacobian analysis shows how PFPE reconditions the TD Jacobian allowing us to prove convergence in regimes where classic TD methods are unstable, thereby breaking the so-called deadly triad that has plagued TD methods (Sutton &amp; Barto, 2018). Furthermore, our results do not depend on unwieldy assumptions or modifications of algorithms used in practice, such as projection, bounded state spaces, linear function approximation, or iterate averaging, as is done in previous analysis. In addition to our theoretical results, we experimentally evaluate our bounds on a toy domain, indicating that they are tight under relevant hyperparameter regimes. Taken together, our results lead to novel insight as to how exactly target networks affect optimisation, and when and why they are effective, leading to actionable results that can be used to further future research.

## 2. Preliminaries

Proofs for all theorems, propositions and corollaries can be found in Appendix B

We denote the set of all probability distributions on a set X as P ( X ) . We use ∥·∥ to denote the ℓ 2 -norm. For a matrix M , we denote the set of eigenvalues as λ ( M ) with the set of maximum normed eigenvalues as λ max ( M ) := arg sup λ ′ ∈ λ ( M ) | λ ′ | and λ min ( M ) := arg inf λ ′ ∈ λ ( M ) | λ ′ | . The ℓ 2 -norm (spectral norm) for matrix M is ∥ M ∥ = √ λ max ( M ⊤ M ) . Given a function f : X → R and a distribution µ ∈ P ( X ) , we denote the L 2 -norm as: ∥ f ∥ µ := √ E x ∼ µ [ f ( x ) 2 ] .

## 2.1. Reinforcement Learning

We consider the infinite horizon discounted RL setting. The agent interacts with an environment, formalised as a Markov Decision Process (MDP): M := ⟨S , A , P, P 0 , R, γ ⟩ with state space S , action space A , transition kernel P : S×A → P ( S ) , initial state distribution P 0 ∈ P ( S ) , bounded stochastic reward kernel R : S × A → P ([ -r max , r max ]) where r max ∈ R &lt; ∞ and scalar discount factor γ ∈ [0 , 1) . An agent in state s ∈ S taking action a ∈ A observes a reward r ∼ R ( s, a ) . The agent's behaviour is determined by a policy that maps a state to a distribution over actions: π : S → P ( A ) and the agent transitions to a new state s ′ ∼ P ( s, a ) . We denote the joint distribution of s ′ , a ′ , r conditioned on s, a for policy π as P π sar ( s, a ) . We seek to optimise (in the control case), or estimate (in the policy evaluation case) the expected discounted sum of future rewards starting from a given state s ∈ S . This quantity is given by the state value function, V π ( s ) = E a ∼ π ( s ) [ Q π ( s, a )] , with Q π : S × A → [ -r max / (1 -γ ) , r max / (1 -γ )] , the action value function, given recursively through the Bellman equation: Q π ( s, a ) = T π [ Q π ]( s, a ) , where the Bellman operator T π projects functions forwards by one step through the dynamics of the MDP:

<!-- formula-not-decoded -->

T π is a γ -contractive mapping and thus has a fixed point, which corresponds to the true value of π (Puterman, 2014). When estimating MDP values, we employ a value function approximation Q ω : S × A → R parametrised by ω ∈ Ω ⊆ R n .

Many RL algorithms employ TD learning for policy evaluation, which combines bootstrapping, state samples and sampled rewards to estimate the expectation in the Bellman operator (Sutton, 1988). In their simplest form, TD methods update the function approximation parameters according to:

<!-- formula-not-decoded -->

where s ∼ d, a ∼ µ ( s ) , s ′ , a ′ , r ∼ P π sar ( s, a ) , d ∈ P ( S ) is a sampling distribution, and µ is a sampling policy that may be different from the target policy π . For simplicity of notation and to accommodate the introduction of target networks in Section 3, we define the tuple ς := ( s, a, r, s ′ , a ′ ) with distribution P ς and the TD-error vector as:

<!-- formula-not-decoded -->

allowing us to write the TD parameter update as:

<!-- formula-not-decoded -->

We make the following i.i.d. assumption for clarity of exposition, but discuss other sampling regimes in Appendix D:

Assumption 1. Each s ∼ d is drawn i.i.d..

Typically, d is the steady-state distribution of an ergodic Markov chain. We denote the expected TD-error vector as: δ ( ω, ω ′ ) := E ς ∼ P ς [ δ ( ω, ω ′ , ς )] and define the set of TD fixed points as:

<!-- formula-not-decoded -->

If a TD algorithm converges, it converges to a TD fixed point. Convergence of TD methods can only be guaranteed for linear function approximators when sampling on-policy in an ergodic MDP, that is the agent sampling and target distributions are the same. We investigate the phenomenon further as part of our asymptotic analysis in Section 4.1.

## 3. Partially Fitted Policy Evaluation

Unfortunately, real-world applications of RL often demand the expressiveness of nonlinear function approximators like neural networks and/or the ability to use data that has been collected off-policy, i.e., by following a policy µ that differs from the target policy π for policy evaluation.

## 3.1. Fitted v Partially Fitted Policy Evaluation

Fitted methods improve on the sample efficiency and stability of TD methods by explicitly incorporating the limitations of the function approximation class through the use of a projection operator (Tsitsiklis &amp; Van Roy, 1997). These methods generally perform some variant of the iterate Q ¯ ω l +1 = Π d π T π Q ¯ ω l where Π d is the projection operator Π d Q = arg min Q ′ ∥ Q ′ -Q ∥ d,µ . These updates are known as fitted policy evaluation (PFE).

The projection step is needed to accommodate the fact that values generally cannot be exactly represented with function approximation. To obtain a practical way of carrying out the PFE updates, a separate set of target parameters can be introduced ¯ ω l ∈ Ω that parameterise the TD target and are updated every k timesteps:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The function approximator update in Equation (1) carries out k iterations of stochastic gradient descent (SGD) on the loss:

<!-- formula-not-decoded -->

before updating the target parameters. In the limit as k → ∞ , assuming convergence of SGD to a global minimum, fully fitted policy evaluation occurs by finding ω ∞ ∈ arg inf ω ∈ Ω L ( ω, ¯ ω l ) .

In practice k is finite and only partial policy evaluation occurs before updating the target parameters, a setting we call partially fitted policy evaluation (PFPE). Without loss of generality, we assume that ¯ ω 0 is deterministic with ∥ ¯ ω 0 ∥ &lt; ∞ and α i = α l for all kl ≤ i &lt; k ( l + 1) , that is stepsizes only change after updating target parameters. As the target parameters are updated to the approximator parameters every k timesteps in Equation (2), it suffices to consider the target parameter update in isolation when analysing PFPE. Our goal is thus to analyse a single update for the target parameters in the canonical form:

<!-- formula-not-decoded -->

where D := { ς i } k i =1 is a set of k samples from the environment with distribution P D and g k (¯ ω l , D l , α l ) reduces the k nested updates from Equation (1) into a single update for the target parameters.

## 3.2. Jacobian Analysis

In our analysis, we show that the stability of the expected PFPE update g k (¯ ω l , α l ) := E D∼ P D [ g k (¯ ω l , D , α l ) ] is determined by the conditioning of three Jacobians. We denote the Hessian of the loss as: H ( ω ; ¯ ω l ) := ∇ 2 ω L ( ω ; ¯ ω l ) , the Jacobian of the TD-error vector as: J δ ( ω ; ¯ ω l ) := ∇ ω ′ δ ( ω, ω ′ ) | ω ′ =¯ ω l and define the TD Jacobian as: J TD (¯ ω l ) := ∇ ω δ ( ω, ω ) | ω =¯ ω l . Observe that J TD (¯ ω l ) = J δ (¯ ω l , ¯ ω l ) -H (¯ ω l ; ¯ ω l ) . Without loss of generality, we assume that the Hessian matrix is diagonalisable because, if it is not, an arbitrarily small perturbation can make its eigenvalues distinct and therefore diagonalisable. So that these matrices exist, we require that the expected PFPE update is differentiable almost everywhere, a condition that is guaranteed by a Lipschitz assumption. We also require that the variance of the updates is bounded, motivating the following regularity assumption:

Assumption 2 (Function Approximator Regularity) . We assume that δ ( ω, ω ′ , ς ) is Lipschitz in ω, ω ′ with constant L : ∥ δ ( ω 1 , ω ′ 1 , ς ) -δ ( ω 2 , ω ′ 2 , ς ) ∥ ≤ L ( ∥ ω 1 -ω 2 ∥ + ∥ ω ′ 1 -ω ′ 2 ∥ ) and Ω is convex, V ς ∼ P ς [ δ ( ω, ω, ς )] := E ς ∼ P ς [ | δ ( ω, ω, ς ) -δ ( ω, ω ) ∥ 2 ] ≤ σ 2 δ for some σ 2 δ &lt; ∞ .

The bounded variance assumption can easily be achieved for unbounded function approximators by truncating the TD error vector, much like the commonly used gradient clipping in gradient descent. We now introduce the pathmean Jacobians, which are the principal element of our analysis:

<!-- formula-not-decoded -->

Intuitively, a path-mean Jacobian is the average of all of the Jacobians along the line joining ω to ω ⋆ . The convexity assumption in Assumption 2 ensures that the line integral joining any two points in Ω always exists. The Lipschitz assumption in Assumption 2 is only required for Section 4 and can be weakened to any condition that ensures the pathmean Jacobians exist for the remainder of the paper.

Our analysis in Section 4 proves that stability of TD and PFPE under decaying stepsizes is determined solely by the negative definiteness of the TD path-mean Jacobian ¯ J TD ( ω, ω ⋆ ) . In Section 5, we show for a non-diminishing stepsize regime that through suitable regularisation (which does not affect the TD fixed point), PFPE's stability can be determined only by α l and k , for which stable values exists. As ¯ H ( ω, ω ⋆ ; ¯ ω l ) is the path-mean Hessian of the loss, convergence can be guaranteed under the same mild

assumptions required to prove convergence of a stochastic gradient descent algorithm to minimise L ( ω ; ¯ ω l ) . This implies that PFPE can converge under regimes where TD will not as ¯ J TD ( ω, ω ⋆ ) is positive definite.

## 3.3. Analysis of PFE

We now showcase the power of our Jacobian analysis by writing the PFE updates exactly in terms of (¯ ω 0 -ω ⋆ ) :

Theorem 1. Under Assumption 2, the sequence of PFE updates ¯ ω ⋆ l +1 ∈ arg inf ω L ( ω, ¯ ω ⋆ l ) satisfy:

<!-- formula-not-decoded -->

We can use Theorem 1 to determine the stability of FPE updates. If sup ω,ω ′ ∈ Ω ∥ ∥ ¯ H ( ω ′ , ω ⋆ ; ω ) -1 ¯ J δ ( ω, ω ⋆ ; ω ⋆ ) ∥ ∥ &lt; 1 then the FPE updates are a contraction mapping and will converge to a fixed point under the Banach fixed-point theorem. We discuss the convergence of FPE under varying regularisation schemes in Section 5.1.

## 4. Asymptotic Analysis

We now study the behaviour of Equation (3) in the limit of l →∞ . We introduce the standard Robbins-Munro condition for the decaying stepsizes that is a necessary condition to ensure convergence to a fixed point:

Assumption 3 (Robbins-Munro) . Each α l is a positive scalar with ∑ ∞ l =0 α l = ∞ and ∑ ∞ l =0 α 2 l &lt; ∞ .

Now we introduce a core necessary assumption to prove stability of PFPE with diminishing stepsizes:

Assumption 4 (TD Stability) . There exists a region X TD ( ω ⋆ ) containing a fixed point ω ⋆ such that ¯ J TD ( ω, ω ⋆ ) has strictly negative eigenvalues for all ω ∈ X TD ( ω ⋆ ) .

The key insight from Assumption 4 is that the stability of PFPE under diminishing stepsizes is determined only by the eigenvalues of the single step path-mean Jacobian ¯ J TD ( ω, ω ⋆ ) , regardless of the value of k or α l . Indeed, stochastic approximation can be shown to be provably divergent if this condition cannot be satisfied (Pemantle, 1990). From this perspective, if TD diverges then so will PFPE under diminishing stepsizes , hence the asymptotic stability of PFPE is independent of k and α l , and, unlike updating under a two-timescale regime, introducing target parameters that are updated periodically every k timesteps does not improve asymptotic convergence properties under this analysis. Once Assumption 4 has been established, there are several approaches to prove convergence of the PFPE update under varying sampling conditions and projection assumptions. We follow the proof of (Vidyasagar, 2022), but discuss approaches that generalise our assumptions in Appendix D

Theorem 2. Let Assumptions 1 to 4 hold. If there exists some fixed point ω ⋆ with region of contraction X TD ( ω ⋆ ) and timestep t such that ¯ ω l ∈ X TD ( ω ⋆ ) for all l ≥ t the the sequence of target parameter updates in Equation (2) converge almost surely to ω ⋆ .

## 4.1. The Deadly Triad

We have established that it is not possible to prove convergence of PFPE under diminishing stepsizes if Assumption 4 does not hold. We now discuss how adherence to Assumption 4 formalises a phenomenon known as the deadly triad (Sutton &amp; Barto, 2018) where it has been established that TD cannot be proved to converge when using function approximators in the off-policy setting. To control for the effect of nonlinear function approximation, we first investigate linear function approximators of the form Q ω ( s, a ) = ϕ ( s, a ) ⊤ ω where ϕ : S × A → R n is a feature vector. Define the one-step lookahead distribution as: P µ := E s ∼ d,a ∼ µ ( s ) [ P ( s, a )] . Introducing the shorthand:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we can derive the TD Jacobian as:

<!-- formula-not-decoded -->

We now examine why the conditioning of ¯ J TD ( ω, ω ⋆ ) explains this phenomenon.

Linear Function Approximation For linear function approximators, we show in Appendix A.1 that γ ∥ Q ω ∥ P µ ,π &lt; ∥ Q ω ∥ d,µ for all ω is a sufficient condition for γ Φ ′ -Φ to have negative eigenvalues, thereby satisfying Assumption 4. This implies that the function approximator class remains non-expansive under the one-step lookahead distribution P µ , thereby preventing the function approximator diverging as the Markov chain is traversed. This condition has been introduced previously in the fitted Q -iteration literature (Wang et al., 2020; 2021) as a 'low distribution shift' assumption.

In the on-policy setting in an ergodic MDP, we can prove that there exists a stationary distribution d π induced by following the target policy π , that is µ = π . Moreover it is assumed that samples come from d π ; hence by the definition of ergodicity, the one-step lookahead distribution is the stationary distribution: P π = d π . It thus follows that γ ∥ Q ω ∥ P µ ,π = γ ∥ Q ω ∥ d π ,π &lt; ∥ Q ω ∥ d π ,π and hence Assumption 4 holds automatically for on-policy TD in an ergodic MDP, thereby establishing the convergence properties as a special case via Theorem 2.

For off-policy data, it is not possible to prove that γ ∥ Q ω ∥ P µ ,π &lt; ∥ Q ω ∥ d,µ holds without further assumptions on the sampling policy and MDP. In general, it is not possible to show that ¯ J TD ( ω, ω ⋆ ) is negative definite in the off-policy case as the distribution shift may be too high: there exist counterexample MDPs where off-policy algorithms such as Q -learning provably diverge under linear function approximation (Williams &amp; Baird, 1993; Baird, 1995a).

Nonlinear Function Approximation Even in an onpolicy regime, we cannot prove convergence of TD when nonlinear function approximators such as neural networks are used. In these cases, the path-mean Jacobian may not have a closed form solution. However, it can be bounded by the following norm (see Appendix A.2):

<!-- formula-not-decoded -->

Even making the same assumption as in Section 4.1 of sampling on-policy in an ergodic MDP to show that

<!-- formula-not-decoded -->

we cannot prove the negative definiteness of ¯ J TD ( ω, ω ⋆ ) required to satisfy Assumption 4. This is because the matrix E [ ( T π [ Q ω ] -Q ω ) ∇ 2 ω Q ω ] can be arbitrarily positive definite depending on the MDP and choice of function approximator. Indeed, there exist counterexample MDPs with provably divergent nonlinear function approximators when sampling on-policy (Tsitsiklis &amp; Van Roy, 1997).

## 5. Non-asymptotic Analysis

Our asymptotic analysis in Section 4 shows that increasing k or adjusting α l for PFPE does not affect the asymptotic strong convergence properties of the TD algorithm, implying that target networks do not stabilise TD if stepsizes tend to zero. We showed that the underlying reason for this was the deadly triad, which we formalised as adherence to Assumption 4. We now replace Assumption 4, that is ¯ J TD ( ω, ω ⋆ ) is negative definite, with the assumption that FPE is stable:

Assumption 5 (FPE Stability) . There exists a region X FPE ( ω ⋆ ) containing a fixed point ω ⋆ such that sup ω,ω ′ ∈X FPE ( ω ⋆ ) ∥ ∥ ¯ H ( ω ′ , ω ⋆ ; ω ) -1 ¯ J δ ( ω, ω ⋆ ; ω ⋆ ) ∥ ∥ &lt; 1 .

## 5.1. Stabilising FPE

We now prove that Assumption 5 can always be satisfied using regularisation schemes that do not affect the TD fixed points. We introduce the following regularised TD vector:

<!-- formula-not-decoded -->

where ρ ( ω, ω ′ ) is a regularisation term such that ρ ( ω, ω ) = 0 , thereby not changing the TD fixed point or TD update. As an example, ρ ( ω ′ , ω ′ ) can contain powers of regularisation terms M Reg ( ω -ω ′ ) in addition to combinations of δ ( ω ′ , ω ) and δ ( ω, ω ′ ) terms, where δ ( ω ′ , ω ) is a TD vector with target and Q -network parameters swapped. In this paper, we briefly study regularisation of the form:

<!-- formula-not-decoded -->

where µ mixes the TD updates and η controls the degree of regularisation. We emphasise that δ Reg (¯ ω l , ¯ ω l ) = δ (¯ ω l , ¯ ω l ) , leaving the TD update unchanged. In contrast, unless ω ⋆ is known a priori, introducing regularisation that modifies the TD update-as is done in (Zhang et al., 2021)-will affect the TD fixed points. We now prove that FPE can be stabilised by treating η and µ as hyperparameters to be tuned to the specific MDP.

Proposition 1. Using the regularised TD vector in Equation (5) , the path-mean Jacobians are:

<!-- formula-not-decoded -->

Assumption 5 is satisfied if:

<!-- formula-not-decoded -->

There exists finite η, µ such that Equation (6) holds.

The key insight from Proposition 1 is that regularisation stabilises FPE (and hence PFPE) without affecting existing TD fixed points, even when TD is unstable, motivating future research directions to develop sophisticated regularisation techniques.

## 5.2. Convergence Analysis

By carrying out a non-asymptotic analysis, we now investigate how the deadly triad can be broken by PFPE using Equation (4) when stepsizes do not tend to zero . This leads to a formal understanding of how target parameters stabilise TD under stepsize regimes that are actually used in practice when classic TD methods fail. The foundation of our analysis is a condition function that can be used to determine the stability of the updates:

Definition 1 (Condition Function) . For a subset X ( ω ⋆ ) ⊆ Ω with corresponding fixed point ω ⋆ ∈ X ( ω ⋆ ) such that ω i ∈ X ( ω ⋆ ) for all i ≥ 0 , let

<!-- formula-not-decoded -->

and define the condition function as:

<!-- formula-not-decoded -->

The condition function depends on the maximal eigenvectors of the Jacobians introduced in Section 3.2, and so can still be used to analyse general nonlinear function approximators for which the path-mean Jacobians have no analytic solution. Using the condition function, we decompose the error at a given timestep into the effect of the expected update plus the error induced by variance of the update:

Theorem 3. Define

<!-- formula-not-decoded -->

Let Assumptions 1 and 2 hold, then:

<!-- formula-not-decoded -->

The effect of the expected update (the first term in Equation (8)) is bounded by the condition function, which depends both on data conditioning but critically, on both k and α l as well and must diminish with increasing l to ensure convergence. Using this decomposition, we see convergence is guaranteed if the following assumption holds:

Assumption 6 (Contraction Region) . We assume that C ( α, k ) ≤ c &lt; 1 over X FPE ( ω ⋆ ) .

allowing us to prove convergence of PFPE for stepsizes that don't tend to zero provided that updates remain in a region of contraction:

Corollary 3.1. Let Assumptions 1, 2, 5 and 6 hold. For a fixed stepsize α l = α &gt; 0 ,

<!-- formula-not-decoded -->

Corollary 3.1 is a key result of this work. Our result demonstrates geometric decay of errors in l , to a ball of fixed radius

ασ k 1 -c . This is analogous to related work in stochastic gradient descent (Bottou et al., 2018), and matches the intuition that, without decaying stepsize, variance in the updates means that convergence to a fixed point does not occur. Note that the radius of the ball which we converge to can be made arbitrarily small by decreasing α .

This supports the use of a hybrid approach, wherein a fixed step size is used until iterates are no longer improving and then reducing step size and repeating to decrease the radius of the ball of convergence whilst maintaining k as small as possible. In the remainder of this section, we explore the properties of the condition function to ensure the existence of a region of contraction satisfying Assumption 6.

## 5.3. Properties of PFPE Condition Function

We now investigate key properties of Equation (7) to understand how target parameters can lead to convergence when classic TD methods fail. If ¯ J TD ( ω, ω ⋆ ) is positive definite, TD is provably divergent, however our analysis reveals that there are values of k and α l for which PFPE does converge.

<!-- formula-not-decoded -->

We first investigate the conditions for which our choice of function approximators can never be used to prove convergence. Our condition function implies that we cannot prove convergence for any λ ⋆ H ≤ 0 or λ ⋆ H ≥ 2 α l as repeated applications of | 1 -α l λ ⋆ H | 2 do not reduce the effect the ill-conditioning of ¯ J TD ( ω, ω ⋆ ) . We formalise this in the following regularity assumption:

Assumption 7 (Eigenvalue Regularity Assumption) . Given a region X ⊆ Ω , for all ω, ω ′ ∈ X there exists 0 &lt; λ min 1 and λ max 1 &lt; ∞ such that λ min ≤ λ ( ∇ 2 ω L ( ω ; ω ′ )) ≤ λ max .

We now propose two simple fixes to avoid this issue. Recall from Section 3.2 that λ ⋆ H is an eigenvalue of the Hessian of a loss. If λ ⋆ H was negative, this would imply that the Hessian is not positive semidefinite for all ω in the region of interest; hence we cannot prove convergence of stochastic gradient descent on the loss L ( ω ; ¯ ω l ) , let alone the full PFPE algorithm. To remedy this problem, the eigenvalues of the matrix can be increased using the regularisation introduced in Equation (4) without affecting the TD fixed point. However, if λ ⋆ H ≥ 2 α l , then the conditioning of the Hessian matrix is ill-suited to the chosen step-size, and an easy remedy is to decrease α l . Our bound shows that the condition function is lower bounded by ∥ J ⋆ FPE ∥ , and so if Assumption 5 does not hold, then convergence of PFPE is not provable.

Property 2: Monotonicity For | 1 -α l λ ⋆ H | &lt; 1 , C ( α l , k ) ≤ C ( α l , k ′ ) for k ≤ k ′ .

The monotonicity property ensures that | 1 -α l λ ⋆ H | &lt; 1 defines the interval of Hessian eigenvalues for which there is a regime in which we can increase k in order to ensure PFPE updates are a contraction mapping. This suggests that a key role of the target network is to help mitigate the effects of the ill-conditioning of the TD Jacobian when using fixed step sizes. We now investigate how decreasing stepsizes and increasing the number of PFPE steps affect the conditioning of PFPE, which validates this hypothesis.

<!-- formula-not-decoded -->

The first limit illustrates the effects of a diminishing stepsize sequence, confirming our bound is consistent with the results of the previous section that increasing k does not improve the convergence properties of PFPE if stepsizes tend to zero and PFPE only stabilises TD for 0 &lt; α l . By taking the limit k →∞ , we compliment our monotonicity result, obtaining a bound for how much we can improve on the stability of TD by increasing k . As expected, in the limit of k → ∞ , the condition function tends to ∥ J ⋆ FPE ∥ . Through this insight, we interpret PFPE as mixing FPE and TD updates according the coefficient | 1 -α l λ ⋆ H | k -1 : for k = 1 , PFPE uses only TD updates and in the limit k →∞ , PFPE recovers the FPE update.

## 5.4. Breaking the Deadly Triad

We now combine all properties presented in this section into our main result, proving that through suitable regularisation and choice of α l and k , PFPE breaks TD's deadly triad described in Section 4.1:

Theorem 4. Let Assumption 7 hold over X FPE ( ω ⋆ ) from Definition 1. For any 1 α l &gt; λ min 1 + λ max 1 2 such that α l &gt; 0 , any

<!-- formula-not-decoded -->

ensures that X FPE ( ω ⋆ ) is a region of contraction satisfying Assumption 6.

Theorem 4 demonstrates that appropriate values of α l and k can be found by treating them as hyperparameters, decreasing α l and increasing k until the algorithm is stable, reducing the conditions needed to prove convergence of PFPE to those of proving convergence of stochastic gradient descent on the loss L ( ω ; ¯ ω l ) . The key insight of Theorem 4 is that even when TD is unstable due to 1 &lt; ∥ I + α l ¯ J TD (¯ ω l , ω ⋆ ) ∥ , there exists a finite k such that C ( α l , k ) &lt; 1 and hence PFPE is stable. We illustrate this phenomenon with a sketch in Figure 1, demonstrating that increasing k ensures PFPE is provably convergent in regimes where TD cannot be proved to converge.

Figure 1: We plot C ( α = 0 . 1 , k ) for ∥ ¯ J ⋆ FPE ∥ = 0 . 85 and ∥ ¯ J ⋆ TD ∥ ≤ 1 . 5 with increasing k as a function of λ min .

<!-- image -->

The key insight of our analysis is that, unlike in TD where stability can only be proved if the matrix ¯ J δ ( ω, ω ⋆ ; ω ⋆ ) -¯ H ( ω, ω ⋆ ; ω ) is negative definite, with suitable regularisation, the stability of PFPE can be determined solely by tuning α l and k , regardless of the MDP, sampling regime, or function approximator, thereby breaking the deadly triad. The choice of α l and k thus becomes a trade-off between maintaining a fast rate of convergence and reducing the residual variance ( α l σ k ) 2 in Equation (8).

## 6. Related Work

Our work furthers the analysis of TD, FPE, and targetnetwork based methods. In this section we provide a brief overview of previous investigations of these algorithms.

Fitted Policy Evaluation FPE is a relatively well understood class of RL algorithms from a theoretical perspective. Nedi´ c &amp; Bertsekas (2003) analyse the convergence of the Least-Squares Policy Evaluation (LSPE) of Bertsekas &amp; Ioffe (1996) in an on-policy, linear function approximation setting. Analysis of LSPE shows that learning with constant step size leads to theoretical and empirical gains compared to TD and LSPE with decaying step sizes (Bertsekas et al., 2004), which mirrors our conclusions in Section 5.4.

In the context of fitted methods applied to off-policy and control problems, Munos &amp; Szepesvári (2008) prove generalisation properties of Fitted Q Iteration (Ernst et al., 2005) for general function classes under assumptions of low projection error and limited data distribution shift. Le et al. (2019) coin the term FPE, and formalise the algorithm for general function approximators, with theoretical results under similar assumptions to Munos &amp; Szepesvári (2008).

Theory of TD Previous results concerning convergence rates of classic TD methods largely argue that the Bellman operator is a contraction, and thus most focus on linear function approximation. Tsitsiklis &amp; Van Roy (1997) first proved convergence of linear, on-policy TD, arguing that the projected Bellman operator in this setting is a contraction. This corresponds to a special case of Assumption 4. Dalal et al. (2017) give the first finite time bounds for linear TD(0), under an i.i.d. data model similar to the one that we use here. Bhandari et al. (2018) provide bounds for linear TD in both the i.i.d. data setting and a correlated data setting, through analogy with SGD. Srikant &amp; Ying (2019) approach the problem from the perspective of Ordinary Differential Equations (ODE) analysis, bounding the divergence of a Lyapunov function from the limiting point of the ODE that arises from the TD update scheme.

Analysis of Target Networks Existing analysis of the theoretical properties of target networks are limited, usually involving algorithmic changes or restrictive assumptions. Yang et al. (2019) show convergence of a Q -learning approach using a target network that is updated using Polyak averaging with nonlinear function approximation. However their analysis-which makes use of two-timescale analysisrequires a projection step to limit the magnitude of parameters. Carvalho et al. (2020) show convergence of a related method using two-timescale analysis, though their target network update differs significantly from those used in practice. Zhang et al. (2021) analyse the use of target networks with linear function approximation, but require projection steps on both the target network and value parameters. Lee &amp; He (2019) provide finite-iteration bounds, but are limited to on-policy data, linear function approximation, and near-perfect fitting to the target network between updates. Fan et al. (2020) analyse the use of target networks for deep Q-learning (Mnih et al., 2015) with the simplifying assumption that they are performing some form of Fitted Q Iteration.

None of these efforts yield finite time bounds with target networks, nor do any match the policy evaluation methods used in practice as well as the PFPE analysis studied here. Furthermore, our use of a single target network update, rather than independent target and value updates leads to simpler bounds without the need for a two-timescale analysis.

GTD and TDC Methods While not directly related to PFPE or the use of target networks, GTD-style approaches (Sutton et al., 2008; 2009; Maei et al., 2009) also lead to convergent, TD-style algorithms, even with off-policy sampling or nonlinear function approximation. These methods maintain a second set of parameters which must be optimised at a faster timescale than the value parameters. However, these approaches are commonly found to be ineffective and not used in practice due to the difficulty in tuning the rate of second timescale (see, e.g. Fellows et al. (2021)), and potentially additional variance introduced by the second set of parameters (Ghiassian et al., 2020).

Improving Conditioning of TD Methods Previous work concerning conditioning of TD methods has been largely concerned with approximation of preconditioning approaches to iterative-methods (Saad, 2003). The first such approach was focused on preconditioning of on-policy, linear, least-squares forms of TD (Yao &amp; Liu, 2008). Chen et al. (2020); Romoff et al. (2020) adapt this approach for nonlinear function approximation, though their results are still on-policy. Our work, on the other hand, demonstrates that use of the target network, alongside fixed step sizes, changes the form of parameter iterates to ameliorate the poor conditioning that occurs when directly applying TD or fitted methods, even in off-policy settings.

## 7. Experiments

We proceed to empirical investigation of our bounds. First, we demonstrate that the use of an infrequently updated target network leads to convergence of off-policy evaluation on the Baird's notorious counterexample. Then, we evaluate the effect of a speculative modified update rule in the Cartpolev0 'gym' environment (Brockman et al., 2016). Additional implementation details for both experiments can be found in Appendix C.

## 7.1. Baird's Counterexample

In this experiment, we demonstrate the practicality of our core claim-that for sufficiently high k and low enough α , PFPE will not diverge, even under conditions that TD does. To do so, we evaluate the use of target networks with varying update frequencies on the well known off-policy counterexample due to Baird (1995b).

In this environment, depicted in Appendix C, rewards are zero everywhere, transitions are deterministic, and the true solution lies within the linear function approximation class that we make use of. The behaviour policy is set such that all states are sampled with uniform probability. The target policy, however, always transitions to a specific state, and remains there. Due to undersampling of this absorbing state, conventional TD policy evaluation diverges, demonstrating that even in simple environments, TD can be unstable when applied off policy with function approximation.

We report the stepwise (fitted) error in Figure 2 across different values of k , for fixed step size α = 0 . 01 , and fixed discount factor γ = 0 . 99 . We see that with k = 1 -which is equivalent to using TD with fixed step sizes-our parameters diverge. Likewise, if k is set to 5 or 10, we are unable to overcome the conditioning of the TD Jacobian and diverge,

10195

10166

10137

10108

1079

1050

1021

10-8

Baird's Counterexample Loss

Update Interval: 1

Update Interval: 10

Update Interval: 100

Update Interval: 500

Update Interval: 1000

albeit at a slower rate. Once we take k ≥ 500 , however, conditioning has improved enough to lead to convergence. This supports our theoretical conclusion: that PFPE can be used to improve the convergence conditions of TD.

10000 20000 30000 40000 50000

Figure 2: Experiment on Baird's counterexample. Decreasing the frequency of target network updates improves conditioning and leads to convergence of PFPE for suitable choices of hyperparameters.

<!-- image -->

## 7.2. Cartpole Experiment

One important insight of our analysis is that we can view the entire optimisation process as a sequence of updates to the target network only. This suggests investigation into alternative forms or acceleration of target network updates. Inspired by the use of optimisation methods with momentum in RL settings (Sarigül &amp; Avci, 2018; Haarnoja et al., 2018), we investigate the effects of a target network that is updated using momentum.

Unlike the standard periodic target network update in Equation (2), we postulate that there may be settings in which a periodic update with momentum may accelerate or stabilise convergence. This update works as follows:

<!-- formula-not-decoded -->

We investigate the effects of this momentum update on the Cartpole domain. For this experiment, we use control results in which the policy is continuously learned. This is because control problems are inherently off-policy, and induce additional instability, and thus benefit from faster and more stable convergence of values. We implement the standard DQN(Mnih et al., 2015) algorithm, with our modified target network update in order to examine its effect. The results are shown in Figure 3. Our proposed update indeed leads to improved learning and stability, at least for the hyperparameter ranges tested, suggesting that the momentum update has merit. As a result, we propose investigation of more

Figure 3: Cartpole Experiment. The agent with the momentum update is significantly more stable and able to consistently learn, while without the modified update, learning collapses.

<!-- image -->

sophisticated target network update schemes as an avenue for future research.

## 8. Conclusions

This work analysed the use of target networks through the formulation of a novel class of TD updates, which we refer to as PFPE. These updates generalise traditional TD(0) and fitted policy evaluation methods. Our analysis contributes asymptotic and finite time bounds without additional restrictive assumptions or significant changes to the algorithms used in practice. In our main result, we uncovered novel insight as to when and how target networks are useful: provided step-sizes don't tend to zero and FPE is stable, there always exists a finite number of update steps k and non-zero upper bound over stepsizes such that PFPE can improve conditioning to ensure learning is stable when classic TD methods fail. Our focus on the target network update as the object of concern in terms of optimisation suggests that novel, accelerated methods for updating target networks may help speed up and stabilise learning. Our initial experiments support this notion. Moreover, our analysis reveals that regularisation may be key to determining the stability of PFPE, opening a promising avenue for future research.

## Acknowledgements

Mattie Fellows is funded by a generous grant from Waymo. We would like to thank Valentin Thomas for providing a helpful discussion.

## References

Allasonniere, S., Kuhn, E., and Trouve, A. Construction of bayesian deformable models via a stochastic approximation algorithm: A convergence study. Bernoulli ,

- 16(3):641-678, 2010. ISSN 13507265. URL http: //www.jstor.org/stable/25735007 . D
- Andradottir, S. A projected stochastic approximation algorithm. In 1991 Winter Simulation Conference Proceedings. , pp. 954-957, 1991. doi: 10.1109/WSC.1991. 185710. D
- Andrieu, C., Moulines, E., and Priouret, P. Stability of stochastic approximation under verifiable conditions. SIAM Journal on Control and Optimization , 44(1):283-312, 2005. doi: 10.1137/ S0363012902417267. URL https://doi.org/10. 1137/S0363012902417267 . D
- Baird, L. Residual algorithms: Reinforcement learning with function approximation. In Prieditis, A. and Russell, S. J. (eds.), Proceedings of the Twelfth International Conference on Machine Learning (ICML 1995) , pp. 3037, San Francisco, CA, USA, 1995a. Morgan Kauffman. ISBN 1-55860-377-8. URL http://leemon.com/ papers/1995b.pdf . 4.1
- Baird, L. Residual algorithms: Reinforcement learning with function approximation. In Machine Learning Proceedings 1995 , pp. 30-37. Elsevier, 1995b. 7.1
- Bertsekas, D. P. and Ioffe, S. Temporal differences-based policy iteration and applications in neuro-dynamic programming. Lab. for Info. and Decision Systems Report LIDS-P-2349, MIT, Cambridge, MA , 14, 1996. 6
- Bertsekas, D. P., Borkar, V . S., and Nedic, A. Improved temporal difference methods with linear function approximation. Learning and Approximate Dynamic Programming , pp. 231-255, 2004. 6
- Bhandari, J., Russo, D., and Singal, R. A finite time analysis of temporal difference learning with linear function approximation, 2018. 6
- Borkar, V. S. and Meyn, S. P. The o.d. e. method for convergence of stochastic approximation and reinforcement learning. SIAM J. Control Optim. , 38(2): 447-469, jan 2000. ISSN 0363-0129. doi: 10.1137/ S0363012997331639. URL https://doi.org/10. 1137/S0363012997331639 . B.2, 2, 2
- Bottou, L., Curtis, F. E., and Nocedal, J. Optimization methods for large-scale machine learning. Siam Review , 60(2):223-311, 2018. 5.2
- Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., and Zaremba, W. Openai gym. arXiv preprint arXiv:1606.01540 , 2016. 7
- Brooms, A. C. Stochastic approximation and recursive algorithms with applications, 2nd edn by h. j. kushner and
- g. g. yin. Journal of the Royal Statistical Society: Series A (Statistics in Society) , 169(3):654-654, 2006. doi: https://doi.org/10.1111/j.1467-985X.2006.00430\_6.x. URL https://rss.onlinelibrary.wiley. com/doi/abs/10.1111/j.1467-985X.2006. 00430\_6.x . D
- Carvalho, D., Melo, F. S., and Santos, P. A new convergent variant of q-learning with linear function approximation. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems , volume 33, pp. 19412-19421. Curran Associates, Inc., 2020. URL https://proceedings. neurips.cc/paper/2020/file/ e1696007be4eefb81b1a1d39ce48681b-Paper. pdf . 6
- Chen, S., Devraj, A. M., Lu, F., Busic, A., and Meyn, S. Zap q-learning with nonlinear function approximation. Advances in Neural Information Processing Systems , 33: 16879-16890, 2020. 6
- Dalal, G., Szörényi, B., Thoppe, G., and Mannor, S. Finite sample analysis for td (0) with linear function approximation. arXiv preprint arXiv:1704.01161 , 2017. 6
- Debavelaere, V., Durrleman, S., and Allassonnière, S. On the convergence of stochastic approximations under a subgeometric ergodic Markov dynamic. Electronic Journal of Statistics , 15(1):1583 - 1609, 2021. doi: 10.1214/21-EJS1827. URL https://doi.org/10. 1214/21-EJS1827 . D
- Ernst, D., Geurts, P., and Wehenkel, L. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research , 6:503-556, 2005. 6
- Fan, J., Wang, Z., Xie, Y., and Yang, Z. A theoretical analysis of deep q-learning, 2020. 1, 6
- Fellows, M., Hartikainen, K., and Whiteson, S. Bayesian bellman operators. Advances in Neural Information Processing Systems , 34:13641-13656, 2021. 6
- Fujimoto, S., van Hoof, H., and Meger, D. Addressing function approximation error in actor-critic methods. In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pp. 1587-1596. PMLR, 10-15 Jul 2018. URL https://proceedings.mlr.press/v80/ fujimoto18a.html . 1
- Ghiassian, S., Patterson, A., Garg, S., Gupta, D., White, A., and White, M. Gradient temporal-difference learning with regularized corrections. In International Conference on Machine Learning , pp. 3524-3534. PMLR, 2020. 6

- Haarnoja, T., Tang, H., Abbeel, P., and Levine, S. Reinforcement learning with deep energy-based policies. In Precup, D. and Teh, Y. W. (eds.), Proceedings of the 34th International Conference on Machine Learning , volume 70 of Proceedings of Machine Learning Research , pp. 1352-1361. PMLR, 06-11 Aug 2017. URL https://proceedings.mlr.press/v70/ haarnoja17a.html . 1
- Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actorcritic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pp. 1861-1870. PMLR, 10-15 Jul 2018. URL https://proceedings.mlr. press/v80/haarnoja18b.html . 1, 7.2
- Le, H., Voloshin, C., and Yue, Y. Batch policy learning under constraints. In International Conference on Machine Learning , pp. 3703-3712. PMLR, 2019. 1, 6
- Lee, D. and He, N. Target-based temporal difference learning, 2019. 1, 6
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., and Wierstra, D. Continuous control with deep reinforcement learning. In Bengio, Y. and LeCun, Y. (eds.), ICLR , 2016. 1
- Maei, H., Szepesvari, C., Bhatnagar, S., Precup, D., Silver, D., and Sutton, R. S. Convergent temporal-difference learning with arbitrary smooth function approximation. Advances in neural information processing systems , 22, 2009. 6
- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., and Riedmiller, M. Playing atari with deep reinforcement learning. 2013. URL http://arxiv.org/abs/1312.5602 . cite arxiv:1312.5602Comment: NIPS Deep Learning Workshop 2013. 1
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. Human-level control through deep reinforcement learning. nature , 518(7540): 529-533, 2015. 1, 6, 7.2
- Munos, R. and Szepesvári, C. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research , 9 (5), 2008. 6
- Nedi´ c, A. and Bertsekas, D. P. Least squares policy evaluation algorithms with linear function approximation. Discrete Event Dynamic Systems , 13(1):79-110, 2003. 6
- Pemantle, R. Nonconvergence to Unstable Points in Urn Models and Stochastic Approximations. The Annals of Probability , 18(2):698 - 712, 1990. doi: 10.1214/aop/ 1176990853. URL https://doi.org/10.1214/ aop/1176990853 . 4
- Polyak, B. Some methods of speeding up the convergence of iteration methods. Ussr Computational Mathematics and Mathematical Physics , 4:1-17, 12 1964. doi: 10. 1016/0041-5553(64)90137-5. 1
- Puterman, M. L. Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp; Sons, 2014. 2.1
- Romoff, J., Henderson, P., Kanaa, D., Bengio, E., Touati, A., Bacon, P.-L., and Pineau, J. Tdprop: Does jacobi preconditioning help temporal difference learning? arXiv preprint arXiv:2007.02786 , 2020. 6
- Saad, Y. Iterative methods for sparse linear systems . SIAM, 2003. 6
- Sarigül, M. and Avci, M. Performance comparison of different momentum techniques on deep reinforcement learning. Journal of Information and Telecommunication , 2 (2):205-216, 2018. 7.2
- Srikant, R. and Ying, L. Finite-time error bounds for linear stochastic approximation and td learning. In Conference on Learning Theory , pp. 2803-2830. PMLR, 2019. 6
- Sutton, R. S. Learning to predict by the methods of temporal differences. Machine Learning , 3(1):9-44, Aug 1988. ISSN 1573-0565. doi: 10.1007/BF00115009. 1, 2.1
- Sutton, R. S. and Barto, A. G. Reinforcement Learning: An Introduction . The MIT Press, second edition, 2018. URL http://incompleteideas.net/ book/the-book-2nd.html . 1, 4.1
- Sutton, R. S., Szepesvári, C., and Maei, H. R. A convergent o (n) algorithm for off-policy temporal-difference learning with linear function approximation. Advances in neural information processing systems , 21(21):16091616, 2008. 6
- Sutton, R. S., Maei, H. R., Precup, D., Bhatnagar, S., Silver, D., Szepesvári, C., and Wiewiora, E. Fast gradientdescent methods for temporal-difference learning with linear function approximation. In Proceedings of the 26th annual international conference on machine learning , pp. 993-1000, 2009. 6
- Tsitsiklis, J. and Van Roy, B. An analysis of temporaldifference learning with function approximation. IEEE Transactions on Automatic Control , 42(5):674-690, 1997. doi: 10.1109/9.580874. 3.1, 4.1, 6

- Vidyasagar, M. Convergence of stochastic approximation via martingale and converse lyapunov methods, 2022. URL https://arxiv.org/abs/2205.01303 . 4, 2
- Wang, R., Foster, D. P., and Kakade, S. M. What are the statistical limits of offline rl with linear function approximation?, 2020. 4.1
- Wang, R., Wu, Y., Salakhutdinov, R., and Kakade, S. M. Instabilities of offline rl with pre-trained neural representation, 2021. 4.1
- Williams, R. J. and Baird, L. C. Analysis of some incremental variants of policy iteration: First steps toward understanding actor-cr. 1993. 4.1
- Yang, Z., Fu, Z., Zhang, K., and Wang, Z. Convergent reinforcement learning with function approximation: A bilevel optimization perspective, 2019. URL https:// openreview.net/forum?id=ryfcCo0ctQ . 6
- Yao, H. and Liu, Z.-Q. Preconditioned temporal difference learning. In Proceedings of the 25th international conference on Machine learning , pp. 1208-1215, 2008. 6
- Zhang, S., Yao, H., and Whiteson, S. Breaking the deadly triad with a target network, 2021. 1, 5.1, 6

## A. Derivations

## A.1. Derivation of Assumption 4 from low distributional shift

Starting from Assumption 4 and the definition of negative definiteness, we need to show:

<!-- formula-not-decoded -->

whenever γ ∥ Q ω ∥ P µ ,π &lt; ∥ Q ω ∥ d,µ , for all ω . Investigating the first term by expanding the expectations we see:

<!-- formula-not-decoded -->

This allows us to apply our assumption:

<!-- formula-not-decoded -->

## A.2. Nonlinear Jacobian Analysis

We start by bounding the maximum eigenvalue:

<!-- formula-not-decoded -->

We now substitute for the definition of the TD Jacobian, yielding:

<!-- formula-not-decoded -->

as required.

## B. Proofs

## B.1. FPE Analysis

Lemma 1. Under Assumption 2, the FPE update ¯ ω l +1 ∈ arg inf ω L ( ω, ¯ ω l ) satisfies:

<!-- formula-not-decoded -->

Proof. Given ¯ ω l , the FPE fixed point ¯ ω ⋆ l must be an element of the set:

<!-- formula-not-decoded -->

which we use to derive a stability condition for the projection operator:

<!-- formula-not-decoded -->

Let ℓ 1 ( t ) := ¯ ω ⋆ l -t (¯ ω ⋆ l -ω ⋆ ) and ℓ 2 ( t ) := ¯ ω l -t (¯ ω l -ω ⋆ ) . We introduce the notation:

<!-- formula-not-decoded -->

We observe that δ 1 (0 , ¯ ω l ) = δ (¯ ω ⋆ l , ¯ ω l ) and δ 1 (1 , ¯ ω l ) = δ ( ω ⋆ , ¯ ω l ) , and δ 2 (0 , ω ⋆ ) = δ ( ω ⋆ , ¯ ω l ) and δ 2 (1 , ω ⋆ ) = δ ( ω ⋆ , ω ⋆ ) . From the fundamental theorem of calculus and Assumption 2, it follows:

<!-- formula-not-decoded -->

as required.

Theorem 1. Under Assumption 2, the sequence of FPE updates ¯ ω ⋆ l +1 ∈ arg inf ω L ( ω, ¯ ω ⋆ l ) satisfy:

<!-- formula-not-decoded -->

Proof. From Equation (9) of Lemma 1, it follows:

<!-- formula-not-decoded -->

Recursively applying the result l times, our result follows immediately.

## B.2. Asymptotic Analysis

For this section, we define a Martingale difference sequence that captures the behaviour of our updates. Let { ω i } k i =0 denote the intermediate function approximation parameters between target parameter updates ¯ ω l +1 and ¯ ω l , with ω 0 = ¯ ω l and

ω k = ¯ ω l +1 . We start by writing our target parameter updates as:

<!-- formula-not-decoded -->

where we define h i (¯ ω l , D , α l ) recursively as:

<!-- formula-not-decoded -->

and remark that h 0 (¯ ω l , D , α l ) = 0 trivially. We write our target parameters updates as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and M l +1 defines the Martingale sequence:

<!-- formula-not-decoded -->

In this section, we demonstrate that the proof of Borkar &amp; Meyn (2000, Theorem 2.2) can be adapted to account for the additional term ε l +1 that arises due to the use of target networks in the updates. Lemma 2 demonstrates that as stepsizes tend to zero, the effect of ϵ l +1 becomes negligible, hence the inclusion of ε l +1 negligible to our analysis of the underlying ODE defined by the TD updates.

Lemma 2. Let ν n,n + m := ∑ m + n -1 l = n α l ϵ l +1 for m ≥ 1 . Under Assumptions 1 to 3, lim n →∞ sup m ∥ ν n,n + m ∥ = 0 almost surely.

Proof. We start by bounding each ∥ ϵ i +1 ∥ using the the Lipschitzness of δ from Assumption 2:

<!-- formula-not-decoded -->

where

To proceed, we recognise that each ∥ h i (¯ ω l , D l , α l ) ∥ ≤ c h &lt; ∞ almost surely where c h is a finite positive constant otherwise:

<!-- formula-not-decoded -->

for at least one i &gt; j , hence V ς ∼ P ς [ δ ( ω, ω ′ , ς )] = ∞ for some ω, ω ′ thereby violating Assumption 2. Using c h , we bound ∥ ϵ l +1 ∥ :

<!-- formula-not-decoded -->

almost surely. We use this result to bound ∥ ν n,n + m ∥ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hence by the bound established in Equation (10):

<!-- formula-not-decoded -->

Now, under Assumption 3, almost surely, as required.

Theorem 2. Under Assumptions 1- 4, the sequence of target parameter updates in Equation (2) converge almost surely to ω ⋆ .

Proof. Our update

<!-- formula-not-decoded -->

is identical to the update presented in Borkar &amp; Meyn (2000, Eq. 2.1.1) with an additional term ε l +1 . Proof of convergence to the ODE is given by Borkar &amp; Meyn (2000, Lemma 1), which is predicated on the convergence of:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for n ≥ 1 , that is lim n →∞ sup m ∥ ∆ n,n + m ∥ = 0 , almost surely. To adapt our updates so that Borkar &amp; Meyn (2000, Lemma 1) still applies, we recognise that the term ζ n is now replaced in our updates with:

<!-- formula-not-decoded -->

from Borkar &amp; Meyn (2000, Eq. 2.1.6) where

and hence ∆ n,n + m is replaced in our updates with:

<!-- formula-not-decoded -->

where ν n,n + m is defined as Lemma 2. All arguments of Borkar &amp; Meyn (2000, Lemma 1) remain unchanged, except Eq. 2.1.9, where we must now show that lim n →∞ sup m ∥ ¯ ∆ n,n + m ∥ = 0 :

<!-- formula-not-decoded -->

Applying Lemma 2 yields lim n →∞ sup m ∥ ν n,n + m ∥ = 0 almost surely, hence

<!-- formula-not-decoded -->

which is proved in Borkar &amp; Meyn (2000, Lemma 1). Convergence of our algorithm is thus only predicated on the convergence of the update:

<!-- formula-not-decoded -->

Borkar &amp; Meyn (2000, Theorem 2.2) proves convergence of Equation (11) almost surely to ω ⋆ given the following four conditions hold:

- I kδ ( ω, ω ) is Lipschitz in ω ,
- II Stepsizes α l satisfy Assumption 3,
- III The sequence {M l , F l } l ≥ 0 is a Martingale difference sequence with respect to the increasing family of σ -algebras: F l := σ ( { ¯ ω i , M i } i ∈{ 0: l } ) where E [ M l +1 |F l ] = 0 and E [ ∥M l +1 ∥ 2 |F l ] ≤ C (1 + ∥ ¯ ω l ∥ 2 ) for some positive C &lt; ∞ .
- IV The sequence of iterates remain bounded, that is sup l ∥ ¯ ω l ∥ &lt; ∞ almost surely.

Conditions I and II hold trivially.

For Condition III, we can take expectations of the Martingale difference:

<!-- formula-not-decoded -->

as required. We now show that the variance is bounded using Assumption 2:

<!-- formula-not-decoded -->

thereby satisfying Condition III.

Finally, we prove Condition IV using Vidyasagar (2022, Theorem 5), which states iterates remain bounded almost surely if:

- (a) Conditions I and III hold;
- (b) there exists some Lyapunov function V : Ω ↦→ R + such that a ∥ ω -ω ⋆ ∥ 2 ≤ V ( ω ) ≤ b ∥ ω -ω ⋆ ∥ 2 for constants a, b &gt; 0 and ∥∇ 2 ω V ( ω ) ∥ is bounded, and;
- (c) ∇ ω V ( ω ) ⊤ δ ( ω, ω ) &lt; 0 for all ω ∈ X TD ( ω ⋆ ) .

We propose V ( ω ) = 1 2 ∥ ω -ω ⋆ ∥ 2 as a candidate Lyapunov function, which trivially satisfies (b). We now show (c) holds by applying the fundamental theorem of calculus to δ ( ω, ω ) . Let ℓ ( t ) := ω -t ( ω -ω ⋆ ) . Like in Theorem 1, it follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all ω ∈ X TD ( ω ⋆ ) under Assumption 4, as required.

## B.3. Stabilising FPE

Proposition 1. Using the regularised TD vector in Equation (5) , the path-mean Jacobians are:

<!-- formula-not-decoded -->

Assumption 5 is satisfied if:

hence:

<!-- formula-not-decoded -->

There exists finite η, µ such that Equation (12) holds.

Proof. Taking derivatives of δ Reg ( ω, ω ′ ) :

<!-- formula-not-decoded -->

For clarity, we drop arguments of ω ′ , ω ⋆ and ω from our notation.

<!-- formula-not-decoded -->

We note that ( ¯ J δ -¯ H -Iη ) can always be made non-singular (and hence invertible) through an arbitrarily small change in η , allowing us to multiply the first term by ( ¯ J δ -¯ H -Iη ) -1 ( ¯ J δ -¯ H -Iη ) = I , yielding:

<!-- formula-not-decoded -->

We observe that:

<!-- formula-not-decoded -->

hence taking limits η →∞ yields:

for all η &gt; η ′ , as required.

<!-- formula-not-decoded -->

From the continuity of the norm, it follows:

<!-- formula-not-decoded -->

implying that lim η →∞ ∥ ¯ H Reg ( ω, ω ⋆ ; ¯ ω l ) ∥ &lt; 1 for any µ &gt; 2 , µ ∈ (0 , 2 3 ) , which it suffices assume for hereon out. From the definition of the limit, there exists some finite η ′ such that

<!-- formula-not-decoded -->

for all η &gt; η ′ for some small 0 &lt; ϵ &lt; 1 , and hence

<!-- formula-not-decoded -->

## B.4. Nonasymptotic Analysis

Lemma 3. Under Assumption 2, for i &gt; 0 the expected updates can be factored as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By the definition of the expected update ω i +1 :

<!-- formula-not-decoded -->

Like in Theorem 1, let ℓ ( t ) := ω i -t ( ω i -¯ ω ⋆ l ) define the line connecting ω i to ¯ ω ⋆ l . Using this notation we re-write the expected update as:

<!-- formula-not-decoded -->

Applying the fundamental theorem of calculus under Assumption 2 and the chain rule yields our desired result:

<!-- formula-not-decoded -->

Our second result follows immediately:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the definition of the expected update:

<!-- formula-not-decoded -->

Let ℓ ( t ) := ¯ ω l -t (¯ ω l -ω ⋆ ) define the line connecting ¯ ω l to ω ⋆ . Using this notation we re-write the expected update as:

<!-- formula-not-decoded -->

Applying the fundamental theorem of calculus under Assumption 2 and the chain rule yields our desired result:

<!-- formula-not-decoded -->

and for i = 0 :

For our final result:

Lemma 4. Under Assumption 2,

<!-- formula-not-decoded -->

Proof. We start by bounding the expected norm term using Jensen's inequality: E X [ √ X 2 ] ≤ √ E X [ X 2 ] :

<!-- formula-not-decoded -->

where we applied the triangle inequality to derive the final line. We bound the variance term by substituting ω i +1 = ω i + α l δ ( ω i , ¯ ω l , ς i ) :

<!-- formula-not-decoded -->

Applying Lemma 3 to the expectation and using the triangle inequality yields our desired result:

<!-- formula-not-decoded -->

Theorem 3. Define

Let Assumptions 1 and 2 hold, then:

<!-- formula-not-decoded -->

Proof. Let { ω i } k i =0 denote the intermediate function approximation parameters between target parameter updates ¯ ω l +1 and ¯ ω l , with ω 0 = ¯ ω l and ω k = ¯ ω l +1 . We define the set of samples up to i as: D i := { ς j } i j =0 with distribution P D i , with sample ς j having distribution P ς j . Under this notation, we must show:

<!-- formula-not-decoded -->

Applying Lemma 4 to the inner expectation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying Equation (13) from Lemma 4 to the inner expectation and applying Lemma 3 yields:

<!-- formula-not-decoded -->

Recursively applying Equation (16) to Equation (14) k -1 times yields:

<!-- formula-not-decoded -->

Now, applying Equation (13) and Lemma 3 to the expectation:

<!-- formula-not-decoded -->

Substituting into Equation (16):

<!-- formula-not-decoded -->

Finally, we apply Theorem 1 to yield our desired result:

<!-- formula-not-decoded -->

Corollary 3.1. Let Assumptions 1, 2, 5 and 6 hold. For a fixed stepsize α l = α &gt; 0 . For a fixed stepsize α l = α &gt; 0 ,

<!-- formula-not-decoded -->

Proof. We start by applying Theorem 3:

<!-- formula-not-decoded -->

As X FPE ( ω ⋆ ) is a region of contraction and ¯ ω l ∈ X FPE ( ω ⋆ ) for all l ≥ 0 , there exists a positive c &lt; 1 under Assumption 6 such that C ( α l , k ) ≤ c , hence:

<!-- formula-not-decoded -->

Now, for a fixed constant stepsize α l = α , we can apply Equation (17) l times, yielding:

<!-- formula-not-decoded -->

Now we apply the bound 1 -x ≤ exp( -x ) , yielding our desired result:

<!-- formula-not-decoded -->

## B.5. Breaking the Deadly Triad

Theorem 4. Let Assumption 7 hold over X FPE ( ω ⋆ ) from Definition 1. For any 1 α l &gt; λ min 1 + λ max 1 2 such that α l &gt; 0 , any

<!-- formula-not-decoded -->

ensures that X FPE ( ω ⋆ ) is a region of contraction satisfying Assumption 6.

Proof. Now, as | 1 -α l λ ′ | is a symmetric function of λ with a minima at λ = 1 α l and λ min 1 + λ max 1 2 is the mid point of λ min 1 and λ max 1 , it follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, hence

<!-- formula-not-decoded -->

Let ∥ ∥ ¯ J ⋆ FPE ∥ ∥ = 1 -ϵ where 0 &lt; ϵ &lt; 1 . From the definition of a limit, this implies that for ϵ there exists some finite k ′ such that whenever k &gt; k ′ :

<!-- formula-not-decoded -->

as required. To find the value of k for which C ( α l , k ) &lt; 1 , we set C ( α l , k ) = 1 and solve:

<!-- formula-not-decoded -->

## C. Additional Experiment Information

For both plots, each configuration was run over 5 random seeds, with the central tendency given by the mean, and the shaded errors representing the standard error of the mean. Hyperparameters that are not varied in the plots were optimised by grid search across either linear or logarithmic hyperparameter ranges, as is suitable. Parameters were chosen that led to the highest performance as averaged across random seeds, then relevant hyperparameters were varied, using the optimal fixed hyperparameters. Hyperparameters that were varied are denoted as lists in the tables below.

## C.1. Baird's Counterexample

Figure Figure 4 shows the counterexample. The behaviour policy chooses between the action represented by the wavy line with probability 6 / 7 , and the solid line with probability 1/7. The behaviour policy always chooses the solid line. The linear function approximation scheme is shown in terms of the value function weights. Sampling off policy in this way leads to divergence of TD, but PFPE converges, as seen in Figure 2.

## C.2. Cartpole Experiment

For the Cartpole experiment, we use a simple DQN-style setup with a small multilayer perceptron (MLP) representing the value function. A small adjustment is made from PFPE as characterised by the paper. Instead of updating value parameters on single data points, parameter updates are averaged across a small batch. This was found to increase stability of learning in both settings, with no no-

Figure 4: Baird's Counterexample. The solid (grey) action moves the agent to the lower state deterministically. The wavy (orange) action puts the agent into one of the upper states with equal probability

<!-- image -->

table effects when comparing across independent variables. This means that, in addition to our target network, we also make use of a replay buffer which stores observed transitions. As such, data used in updates was sampled uniformly from previous transitions. The policy was ϵ -greedy, with the estimated optimal action taken with probability 1 -ϵ . The environment is maintained by OpenAI as part of the gym suite, and falls under MIT licensing.

## D. Extensions

As discussed in Section 4, once we can establish Assumption 4 then there are several theoretical tools that become applicable from stochastic approximation to prove convergence under a range of assumptions. Brooms (2006) provide a comprehensive overview of classic methods. In particular, stochastic approximation has been shown to converge when sampling from an ergodic Markov chain under specific regularity assumptions (Allasonniere et al., 2010). Perhaps the easiest to verify in our context is those of Andrieu et al. (2005), who provides a series of assumptions that can be checked in practice. Moreover, this theory was recently extended to Markov chains that converge sub-geometrically to their station distributions by Debavelaere et al. (2021). Adherence of the updates to remain in a contractive region can be ensured by projection into an ever increasing subset of Ω until convergence occurs, which is detailed and analysed in Andradottir (1991).

Table 1: Relevant Parameters for Cartpole Experiment

| Parameter                                        | Value           |
|--------------------------------------------------|-----------------|
| Environment Parameters γ Architecture Parameters | 0.99 2 32 ReLU  |
| MLP Hidden Layers                                |                 |
| Hidden Layer Size                                |                 |
| Nonlinearity                                     |                 |
| ϵ                                                | 0.05            |
| Training Parameters                              |                 |
| Total Target Network Updates                     | 500             |
| Learning Rate                                    | [0.001, 0.0005] |
| Momentum ( µ )                                   | [0, 0.01]       |
| Batch Size                                       | 500             |
| Steps per Target Network Update ( k )            | 5               |
| Data Gathering Steps per Update                  | 5               |
| Replay Buffer Size                               | 2500            |