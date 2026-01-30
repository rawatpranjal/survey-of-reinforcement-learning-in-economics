## REGULARIZED Q-LEARNING WITH LINEAR FUNCTION APPROXIMATION

## Jiachen Xi

Department of Industrial and Systems Engineering Texas A&amp;M University College Station, TX 77843 jx3297@tamu.edu

## Alfredo Garcia

Department of Industrial and Systems Engineering Texas A&amp;M University College Station, TX 77843 alfredo.garcia@tamu.edu;

## Petar Momˇ cilovi´ c

Department of Industrial and Systems Engineering Texas A&amp;M University College Station, TX 77843 petar@tamu.edu

February 11, 2025

## ABSTRACT

Regularized Markov Decision Processes serve as models of sequential decision making under uncertainty wherein the decision maker has limited information processing capacity and/or aversion to model ambiguity. With functional approximation, the convergence properties of learning algorithms for regularized MDPs (e.g. soft Q-learning) are not well understood because the composition of the regularized Bellman operator and a projection onto the span of basis vectors is not a contraction with respect to any norm. In this paper, we consider a bi-level optimization formulation of regularized Q-learning with linear functional approximation. The lower level optimization problem aims to identify a value function approximation that satisfies Bellman's recursive optimality condition and the upper level aims to find the projection onto the span of basis vectors. This formulation motivates a single-loop algorithm with finite time convergence guarantees. The algorithm operates on two time-scales: updates to the projection of state-action values are 'slow' in that they are implemented with a step size that is smaller than the one used for 'faster' updates of approximate solutions to Bellman's recursive optimality equation. We show that, under certain assumptions, the proposed algorithm converges to a stationary point in the presence of Markovian noise. In addition, we provide a performance guarantee for the policies derived from the proposed algorithm.

## 1 Introduction

Recent literature has examined regularized Markov Decision Processes MDPs as models of sequential decision making under uncertainty that account for limited information processing capacity [1-3] and aversion to model misspecification or ambiguity [4]. In this literature, low entropy policies are not necessarily optimal because the agent has limited capacity to process new information and/or is averse to model uncertainty. A parallel strand of the literature in reinforcement learning algorithms [5-8] has also employed equivalent formulations with different motivations, namely that of ensuring learned policies are robust to perturbations in both the dynamics and the reward function [9]. For example, negative Shannon entropy regularization has been employed in [6, 10] to motivate 'soft' Q-learning and actor-critic algorithms with robust performance. Regularization has also been used to ensure learned policies are sparse (e.g., Tsallis entropy [8, 11, 12]). This approach maintains a multi-modal policy form but restricts the support to only optimal and near-optimal actions, thereby mitigating the impact of detrimental actions. Nonetheless, the convergence of regularized Q-learning with linear functional approximation is not guaranteed in general [13]. This is because the composition of a projection onto the span of predefined features with the regularized Bellman operator is not necessarily

a contraction with respect to any norm. Hence, a regularized Q-learning iteration process may diverge because the objectives of maximizing expected discounted reward and projecting onto the linear basis may 'push and pull' solution updates in significantly different directions. To our knowledge, there are no algorithms with finite-time guarantees to solve regularized MDPmodels with linear function approximation and general strongly convex and bounded regularizer.

To address this gap, we consider a bi-level optimization approach for regularized Q -learning that enables us to overcome this challenge. The lower level optimization problem aims to identify a value function approximation (referred to as the main solution) that satisfies Bellman's recursive optimality condition. The upper level aims to find a projection onto the span of basis vectors, i.e. a parametrized linear function (referred to as the target solution) that best approximates the main solution in the sense of minimizing mean square error. This formulation motivates a single-loop algorithm wherein at each iteration both the upper level and the lower level solutions are updated with different stepsizes: updates to the projection of state-action values are 'slow' in that they are implemented with a step size that is smaller than the one used for 'faster' updates of approximate solutions to Bellman's recursive optimality equation. We demonstrate that in the presence of Markovian noise the proposed algorithm converges to a stationary point at a rate O ( T -1 / 4 (log T ) 1 / 2 ) with T &gt; 0 denoting the total number of iterations. This result is cast in terms of the ℓ 2 norm rather than the squared ℓ 2 norm, because unlike other single-loop algorithms for reinforcement learning [14] the proposed algorithm does not require a projection for implementing upper level updates. Lastly, we establish finite-time performance guarantees for the learned policies. These guarantees reveal that, compared to the optimal policy, the expected performance gap of policies is O ( T -1 / 4 (log T ) 1 / 2 ) , augmented by the function approximation error and the error introduced by the smooth truncation operator.

## 1.1 Related Work

The stability of Q-learning and its related variants, particularly in the context of function approximation, has long been a focus of research [15, 16]. Numerous techniques have been proposed to address these stability concerns, including the target network [17], double estimator [18], fitted value iteration [19], gradient-based approaches [20], Zap Q-learning [21,22] and its variant, Zap Zero [23] among others. In this subsection, we briefly discuss some works that are most closely related to our work.

The single -loop algorithm proposed in this paper stands in contrast with nested -loop algorithms, e.g. [24,25] which require greater computational effort unless precise fine-tuning of the number of inner-loop iterations is available. Coupled Q-learning [26] is a single-loop approach that also relies on updates to the main and target solutions with decreasing step-sizes in a two-time scale manner as in [27]. However, as shown in [24] (see appendix E.3) the performance guarantee for Coupled Q-learning involves significant bias unless the discount factor is sufficiently small.

Our work is also related to approaches to Q-learning with ridge regularization [28,29]. However, ridge regularization results in a modified Bellman equation with an implicitly scaled down discount factor (see appendix E.2 in [24]). Therefore it is not clear how the identified solution approximates the solution of either the original MDP or its regularized version. In contrast, the finite-time guarantee provided in this paper ensures the identified solution approximately minimizes Mean Squared Projected Bellman Error (MSPBE), a measure of the extent to which the identified solution fails to satisfy Bellman's optimality equation.

Finally, the Greedy-GQ algorithm [20], inspired by the success of TDC in off-policy evaluation [30], is arguably the closest to our work. Greedy-GQ employs a two-timescale gradient-based approach to minimize the MSPBE, and its asymptotic convergence to a stationary point is established over i.i.d. sampled data. Recent work [31] provides a finite-time guarantee for the convergence of Greedy-GQ under Markovian noise with a rate of O ( T -1 / 3 log T ) , and it has been improved to O ( T -1 / 2 log T ) in [32]. Subsequent work [33] further improves the result to O ( T -1 / 2 ) with the use of a mini-batch gradient.

The performance guarantees studied in these Greedy-GQ papers concern the solution to a standard MDP problem (i.e. not regularized). In contrast, in this paper, we consider the solution of regularized MDPproblem wherein the regularizer belongs to a general class of strongly convex and bounded functions. This class includes negative Shannon and negative Tsallis entropy functions. Such regularized MDPs are used to model sequential decision making under uncertanity with limited information processing capacity [1-3] and/or aversion to model misspecification or ambiguity [4]. To our knowledge, there are no algorithms with finite-time guarantees to solve such regularized MDP models with linear function approximation. Our work contributes to fill this gap.

## 2 Preliminaries

In this section, we present the preliminaries for regularized Q-learning. We denote the dot product as ⟨· , ·⟩ . The ℓ 2 and ℓ ∞ norms are denoted by ∥ · ∥ and ∥ · ∥ ∞ , respectively.

## 2.1 Markov Decision Processes (MDPs)

We consider an infinite-horizon regularized Markov Decision Processes (MDPs) with finite action space. An MDP can be described as a tuple M ≜ ( S , A , P, µ 0 , R, γ ) , where S denotes the state space and A denotes the action space. P is the dynamics: P ( s ′ | s, a ) is the transition probability from state s to s ′ by taking action a . µ 0 is the initial distribution of the state. The reward function, R : S × A → R , is bounded in absolute value by R max ≥ 0 , i.e., -R max ≤ R ( s, a ) ≤ R max for all ( s, a ) ∈ S ×A . γ ∈ (0 , 1) is the discount factor. A policy π : S → ∆( A ) is defined as a mapping from the state space S to the probability distribution over the action space A . For discrete time t ≥ 0 , the trajectory, starting from an initial state s 0 ∼ µ 0 ( · ) , generated by the policy π in the MDP M can be represented as a set of transition tuples { s t , a t , s ′ t } t ≥ 0 , where a t ∼ π ( · | s t ) , s ′ t ∼ P ( · | s t , a t ) , and s t +1 = s ′ t , ∀ t ≥ 0 .

The state value and state-action value functions corresponding to a policy π are defined by the expected cumulative rewards obtained when initiating from state s ∈ S or state-action pair ( s, a ) ∈ S × A , and subsequently adhering to policy π , respectively:

<!-- formula-not-decoded -->

and Q π ( s, a ) = R ( s, a ) + γ E s ′ ∼ P ( ·| s,a ) [ V π ( s ′ )] . In general, the objective of an MDP is to identify the optimal policy π ∗ that maximizes the expected cumulative discounted reward. Mathematically, this can be expressed as:

<!-- formula-not-decoded -->

Let Q : S × A → R . The Bellman operator is defined as:

<!-- formula-not-decoded -->

Under certain assumptions, this operator is a contraction with modulus γ &lt; 1 and its unique fixed point Q ∗ characterizes the optimal value function as follows:

<!-- formula-not-decoded -->

## 2.2 Regularized Markov Decision Process

Let G : ∆( A ) → R be a strongly convex function, that is bounded, i.e. there exists B &gt; 0 such that -B ≤ G ( p ) ≤ B for all p ∈ ∆( A ) . For a given policy π , the regularized value function is defined as

<!-- formula-not-decoded -->

where G τ ≜ τ · G , and τ &gt; 0 is a positive coefficient that determines the degree of regularization applied. Let us also define Q π,G τ ( s, a ) ≜ R ( s, a ) + γ E s ′ ∼ P ( ·| s,a ) [ V π,G τ ( s ′ )] .

The regularized Bellman operator is defined as follows:

<!-- formula-not-decoded -->

Note that without regularization (i.e. τ = 0 ), definitions (1) and (2.2) are equivalent. The regularized Bellman operator can be re-written as:

<!-- formula-not-decoded -->

wherein G ∗ τ is the convex conjugate of G τ , i.e.:

<!-- formula-not-decoded -->

The regularized Bellman operator is a contraction in ℓ ∞ norm with modulus γ &lt; 1 (see Proposition 2 in [7]). Let V ∗ G τ ( s ) ≜ max π E s ∼ µ 0 [ V π,G τ ( s ) ] . According to Theorem 1 in [7], the unique fixed point of the regularized Bellman operator, say Q ∗ G τ , characterizes the optimal value in the following sense:

<!-- formula-not-decoded -->

Some Examples of Regularizers : A common choice for regularizer is negative Shannon entropy. As mentioned before, this choice has been used as a way to account for information processing costs [1-3] and to motivate the 'soft' Q-learning algorithm. In this case G τ ( π ( ·| s ′ )) = τ ∑ a ∈ A π ( a | s ′ ) log π ( a | s ′ ) and

<!-- formula-not-decoded -->

Another choice of regularizer is the negative Tsallis entropy, G ( π ( ·| s ′ )) = 1 2 ( ∥ π ( ·| s ′ ) ∥ 2 -1) . In [11], the authors show that this regularizer induces a sparse and multi-modal optimal policy. Some additional properties of regularized MDPs are stated in the following proposition (see Proposition 1 in [7] for proofs).

Proposition 2.1. Let G be a strongly convex function bounded by B &gt; 0 and τ &gt; 0 be the regularization coefficient associated with G . The following hold:

- i. The conjugate function is bounded:

<!-- formula-not-decoded -->

- ii. The gradient of the convex conjugate can be caharacterized as follows:

<!-- formula-not-decoded -->

- iii. (Lipschitz gradients) For some constant L G &gt; 0 .

<!-- formula-not-decoded -->

- iv. The optimal policy π ∗ τ ( ·| s ) = ∇ G ∗ τ ( Q ∗ τ ( s, · )) , for all s ∈ S .

## 2.3 Linear Function Approximation

Function approximation techniques are prevalent as they offer an effective solution to this problem by allowing agents to approximate the values using significantly fewer parameters. In this study, we focus on leveraging linear function approximation which approximates the value functions linearly in the given basis vectors.

The basis vectors ϕ i ∈ R |S||A| , i = 1 , 2 , . . . , d are, without loss of generality, linearly independent. We denote ϕ ( s, a ) = [ ϕ 1 ( s, a ) , ϕ 2 ( s, a ) , . . . , ϕ d ( s, a )] ⊤ ∈ R d for all ( s, a ) and let Φ ∈ R |S||A|× d defined by Φ = [ ϕ ( s 1 , a 1 ) , . . . , ϕ ( s |S| , a |A| ) ] ⊤ . We approximate state-action function as ˆ Q θ ≜ Φ θ , where θ ∈ R d is the parameter vector. We also use ˆ Q θ ( s, · ) = ϕ ( s, · ) ⊤ θ to denote the vector of the approximated state-action values of all actions given the state s ∈ S , where ϕ ( s, · ) = [ ϕ ( s, a 1 ) , . . . , ϕ ( s, a |A| ) ] . By Proposition 2.2 (iv), the optimal policy satisfies:

<!-- formula-not-decoded -->

Suppose µ is a distribution that exhibits full support across the state-action space, let D µ ∈ R |S||A|×|S||A| be a diagonal matrix with the distribution µ on its diagonal. Furthermore, we define the weighted ℓ 2 -norm with respect to the distribution µ as ∥ x ∥ D µ ≜ √ x ⊤ D µ x . Let Π D µ be the projection operator which maps a vector onto the span of the basis functions with respect to the norm ∥ · ∥ D µ ; it is given by Π D µ = Φ ( Φ ⊤ D µ Φ ) -1 Φ ⊤ D µ under the assumption that Φ ⊤ D µ Φ is a positive definite matrix.

Within the framework of linear function approximation, the primary objective of value function estimation is to determine an optimal parameter θ ∗ that satisfies the regularized projected Bellman equation: ˆ Q θ ∗ = Π D µ B τ ˆ Q θ ∗ , where µ is given. While the projection operator Π D µ is known to be a weighted ℓ 2 non-expansive mapping, it is important to note that the composite operator Π D µ B τ does not generally exhibit contraction properties with respect to any norm [24].

## 2.4 Standing Assumptions

We now state the standing assumptions for our analysis. Let π bhv be the behavioral policy used to collect data in regularized MDP M τ and we adopt the following assumption.

Assumption 2.2. The behavioral policy π bhv satisfies π bhv ( a | s ) &gt; 0 for all ( s, a ) , and the Markov chain induced by it is irreducible and aperiodic. Then there exist constants κ &gt; 0 and ρ ∈ (0 , 1) such that

<!-- formula-not-decoded -->

where d TV is the total-variation distance and µ bhv is the stationary distribution of state-action pairs.

Assumption 2.2 guarantees the ergodicity of the Markov chain, which in turn ensures that it converges to its stationary distribution at a geometric rate. Furthermore, Let D be the stationary distribution of the transition tuples induced by π bhv . Consequently, D = µ bhv ⊗ P , where ⊗ denotes the tensor product between two distributions. We use the term D to also refer to the marginal distribution of state-action pairs, i.e., D ( s, a ) = µ bhv ( s, a ) for all ( s, a ) ∈ S × A , with a slight abuse of notation. Finally we make the following standing assumption

<!-- formula-not-decoded -->

Note assumption 2.3 can be ensured by normalizing the features (see [31,34,35]). With a slight abuse of notation in the remainder of the paper we use Π to represent Π D µ bhv and ∥ · ∥ to refer to ∥ · ∥ D µ bhv .

## 3 Problem Formulation

## 3.1 Smooth Truncation Operator

Motivated by the stabilization of using the truncation operator within linear function approximation as highlighted in [24], we explore a smooth variant of this operator. This adaptation maintains the smoothness of Bellman backups in regularized MDPs and is particularly advantageous for gradient-based algorithms, which are our primary focus. For a threshold δ &gt; 0 , we define the smooth truncation operator K as

<!-- formula-not-decoded -->

where tanh is the hyperbolic tangent function; the particular choice of the operator is not essential - alternative choices are feasible. Consequently, K δ maps any value in R to the interval ( -δ, δ ) . We define ˆ G τ,δ as the composite operator formed by applying the smooth truncation operator K δ to the convex conjugate G ∗ τ of the regularizer, i.e.

<!-- formula-not-decoded -->

In addition, we let B τ,δ be the smooth truncated optimal regularized Bellman operator which is given as, for an arbitrary mapping Q : S × A → R :

<!-- formula-not-decoded -->

which preserves the property of γ -contraction in ℓ ∞ -norm.

In contrast to the hard truncation operator ⌈ x ⌉ δ ≜ max { min { x, δ } , -δ } , K δ is differentiable everywhere on R , with the gradient given by

<!-- formula-not-decoded -->

which equals 1 at x = 0 and approaches 0 as | x |→∞ . It is evident that |K δ ( x ) | ≤ |⌈ x ⌉ δ | ≤ | x | for all x ∈ R . This property can be beneficial in mitigating the overestimation issue. However, there is a trade-off involving the gap between K δ ( x ) and x when x is away from the origin. On the other hand, as the value of | x δ | decreases, K δ ( x ) approaches closer to x . Therefore, we have the inequality

<!-- formula-not-decoded -->

Further investigation of the threshold δ is presented in Section 6.

## 3.2 A Bi-level Formulation for Bellman Error Minimization

In what follows, we aim to minimize the Mean Squared Projected Bellman Error (MSPBE) defined as

<!-- formula-not-decoded -->

A bi-level optimization formulation for minimizing MSPBE is as follows:

<!-- formula-not-decoded -->

where the lower-level objective is

<!-- formula-not-decoded -->

Note that ˆ Q ω ∗ ( θ ) = Φ ω ∗ ( θ ) represents the projection of B τ,δ ˆ Q θ onto the space spanned by the basis functions with respect to the norm ∥ · ∥ D µ bhv , i.e. ˆ Q ω ∗ ( θ ) = Π B τ,δ ˆ Q θ . Hence, it can be readily seen that the upper level objective function is MSPBE, i.e. J ( θ ) = f ( θ, ω ∗ ( θ )) . In the above formulation, the parameters ω are referred to as the main solution (or network) whereas θ represents the parameters of the target solution (or network).

We now show the lower level objective function g ( θ, ω ) is strongly convex in ω :

Proposition 3.1. Under Assumption 2.2, there exists a constant λ g &gt; 0 that lower bounds the eigenvalue of Σ ≜ E D [ ϕ ( s, a ) ϕ ( s, a ) ⊤ ] . Consequently, g ( θ, ω ) is strongly convex in ω for any θ ∈ R d , and with modulus λ g / 2 . That is, for any ω 1 , ω 2 ∈ R d , θ ∈ R d ,

<!-- formula-not-decoded -->

See the proof in Appendix E.1. In addition, the solution ω ∗ ( θ ) of the lower level problem has the following closed-form:

<!-- formula-not-decoded -->

## 3.3 Gradient of Bellman Error

The gradient of J ( θ ) can be derived by using the chain rule as

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

Observe that the range of z is (0 , 1] . We make the following regularity assumption on the gradient:

Assumption 3.2. For θ ∈ R d , the matrix ˆ Σ τ,δ,θ is non-singular. Therefore, there exists a constant σ min &gt; 0 that lower bounds the smallest singular value of it for all θ ∈ R d .

Note that Assumption 3.2 is as strong as similar assumptions in [20, 22, 33, 36-39]. This is because z ( s ; τ, δ, θ ) → 1 as δ → + ∞ . Nonetheless, when δ has a finite value, Assumption 3.2 is weaker, especially for a small value of δ .

<!-- formula-not-decoded -->

## Algorithm 1 Single-loop Regularized Q-learning

- 1: Input: Constant T , step sizes α, β , initial parameters ω 0 , θ 0 , behavioral policy π bhv .
- 2: Sample Initial State: s 0 ∼ µ 0 ( · ) .
- 3: for t = 0 , . . . , T -1 do
- 4: Sample Action and Subsequent State: a t ∼ π bhv ( · | s t ) , s ′ t ∼ P ( · | s t , a t )
- .
- 5: Estimate Lower-Level Gradient: compute h t g by performing (10).
- 6: Update Lower-Level Parameter: perform (12a).
- 7: Estimate Upper-Level Gradient: compute h t f by performing (11).
- 8: Update Upper-Level Parameter: perform (12b).
- 9: Update State: s t +1 = s ′ t .
- 10: end for

## 4 A Single-Loop Algorithm

In this section, we introduce a single-loop algorithm for solving the bi-level problem (5) wherein at every iteration the upper level and the lower solutions are updated based upon samples from a surrogate gradient of the upper level objective and the gradient of the lower level objective. Specifically, the gradient of the lower level objective is:

<!-- formula-not-decoded -->

A surrogate gradient for the upper level objective:

<!-- formula-not-decoded -->

wherein ω ∗ ( θ ) is replaced by ω in (7). Samples for the upper-level surrogate gradient and lower-level gradient can be readily obtained as follows.

Let ( s t , a t , s ′ t ) denote the t -th transition on the Markov chain induced by π bhv on MDP M τ . The action a t is sampled from π bhv ( · | s t ) and the next state s t +1 = s ′ t , which is drawn from P ( · | s t , a t ) . The stochastic gradient of the lower-level problem and the stochastic surrogate gradient of the upper-level problem are represented by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

respectively, where z t := z ( s ′ t ; τ, δ, θ t ) . Recall the definition of π θ in (3), which represents the optimal policy derived from the state-action value estimation of the target network, ˆ Q θ . The parameters are updated with the step sizes α and β in the following sense:

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

where P denotes the projection onto ℓ 2 -balls with a radius of ( R max + γδ ) /λ g . This value ensures that the optimal point ω ∗ ( θ t ) of the lower-level problem lies within the specified ball. The algorithm for regularized Q-learning with adaptive stepsize is summarized in Algorithm 1.

Remark: Utilizing the projection P constrains the main network, preventing it from taking overly large steps in the incorrect direction [35, 39, 40]. The update rule for the upper-level problem in Algorithm 1 normalizes the stochastic gradient by dividing it by the difference between the weights of the main and target networks.

Lemma 4.1. (Asymptotic stability.) Suppose Assumptions 2.2, 2.3 and 3.2 hold. The approximated Q-value remains bounded almost surely for all a ∈ A :

<!-- formula-not-decoded -->

and

See the proof in Appendix C.1.

## 5 Finite-Time Guarantees

In this section, we present the finite-time convergence guarantee for Algorithm 1 and a finite-time performance bound of the derived policies. The approach we take is similar to that of [14, 41] wherein the two time scales refer to different constant step sizes for lower level and upper level updates, i.e. α &lt; β and the finite-time guarantee is expressed in terms of a target number of iterations T ≥ 1 . This differs from other approaches in which the step-sizes converge to zero at different rates as in [27, 42, 43].

## 5.1 Convergence Analysis

The upper level objective function in (5) is not smooth. We consider the following relaxation of smoothness:

Lemma 5.1. (Relaxation of smoothness.) Suppose Assumptions 2.2, 2.3 hold, for all θ 1 , θ 2 ∈ R d , then

<!-- formula-not-decoded -->

where L 0 = 4 /λ g and L 1 = L G |A| /τ +2 /δ . Moreover, the following holds:

<!-- formula-not-decoded -->

We refer the readers to Appendix C.2 for detailed proof. A similar property termed ( L 0 , L 1 ) -smoothness is incorporated into the optimization algorithm, enhancing the speed of resolution, as demonstrated in [44, 45]. By employing the smooth truncation operator to impose bounds and integrating the projection operation as illustrated in equation (12a), we ensure that the ℓ 2 -norm of the stochastic gradient of the lower-level problem is constrained.

Lemma 5.2. Under Assumption 2.3, we have

<!-- formula-not-decoded -->

The detailed proof can be found in Appendix C.3.

We present the finite-time convergence guarantees of Algorithm 1 in the following theorem.

Theorem 5.3. Suppose Assumptions 2.2, 2.3 and 3.2 hold. Let α = α 0 T -3 / 4 , β = β 0 T -1 / 2 , where α 0 , β 0 are positive constants, and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, the average ℓ 2 -norm of the gradient of the objective function satisfies

<!-- formula-not-decoded -->

The detailed proof can be found in Appendix B.1.

Observe that the bounds outlined in Theorem 5.3 are expressed in terms of ℓ 2 norms. This is different from the conventional use of squared ℓ 2 norms prevalent in the majority of two-timescale convergent algorithms, as demonstrated

Then, Algorithm 1 yields and

in [14, 25, 31-33, 39, 46]. The primary reason for this distinction lies in the normalization of the gradient employed in update (12b), which is common in the optimization techniques like normalized or clipped gradient-based algorithms. However, it can be inferred that the norm scaling at an order of T -1 / 4 (log T ) 1 / 2 is roughly equivalent to the squared norm scaling at an order of T -1 / 2 log T , albeit with a slight discrepancy.

Remark 5.4. The rates in (13a) and (14) have components that are inversely proportional to the square of λ g , the lower bound of the eigenvalue of Σ . This suggests that a larger λ g leads to faster convergence of the lower-level problem, thereby enhancing the convergence of the upper-level problem. Additionally, a high value of λ g , along with a large coefficient τ , augments the smoothness of the objective function, as established in Lemma 5.1. Furthermore, the rates in (13b) and (14) include components inversely proportional to σ min , the lower bound of the singular value of the matrix ˆ Σ τ,δ,θ . Increased smoothness in J , together with a large σ min , allows for a greater step size α . This, in practice, accelerates convergence. The effect of the threshold δ on the rates in Theorem 5.3 is more complex, and we defer the discussion to Section 6.

Remark 5.5. The convergence rate of Algorithm 1 is constrained by the convergence rate (13a) of its lower-level problem. In a nested loop algorithm framework, where the lower-level problem is optimally solved in each iteration, we have ω t +1 = ω ∗ ( θ t ) and ¯ ∇ θ f ( θ t , ω t +1 ) = ∇ J ( θ t ) . As a result, the bound defined in (7) can be achieved as O ( T -1 / 2 log T ) by choosing a step size α = α 0 T -1 / 2 , where T denotes the total number of iterations for updating the target network ˆ Q θ t .

We also demonstrate that the target network converges to the main network.

Corollary 5.6. The target network ˆ Q θ t converges to the main network ˆ Q ω t at a rate given by

<!-- formula-not-decoded -->

The detailed proof is in Appendix D.1.

## 5.2 Performance Analysis

In this subsection, we analyze the difference between the estimated state-action value function ˆ Q θ and the optimal value function Q ∗ τ in M τ .

The approximation error in our linear function approximation setting is defined as

<!-- formula-not-decoded -->

This error quantifies the representational power of the linear approximation architecture [34]. With additional information about the problems we are targeting, the space containing θ to be considered can be narrowed. Consequently, this leads to a further reduction in the approximation error. Also, we define

<!-- formula-not-decoded -->

which serves as an upper bound for all state value functions in M τ .

In the following theorem, we present the finite-time bound characterizing the difference between the estimated value function and the optimal value functions.

Theorem 5.7. Suppose Assumptions 2.2, 2.3 and 3.2 hold. Then Algorithm 1 yields

<!-- formula-not-decoded -->

The detailed proof can be found in Appendix B.2.

The first term in (16) can be bounded by leveraging Theorem 5.3. The term E approx is generally non-removable, a characteristic echoed in much of the literature on RL with linear function approximation [24, 26, 47]. The last term arises from the application of the smooth truncation operator K δ . We defer the discussion of this term to Section 6.

As the approximation error nears zero and the effect of the smooth truncation diminishes, the learned policy π θ t converges towards the optimal policy π ∗ τ . This convergence is realized under the condition of Bellman Completeness [48], provided that the threshold δ is also sufficiently large.

A corollary to Theorem 5.7 characterizes policy convergence by a straightforward application of Proposition 2.1(iii): Corollary 5.8. Suppose Assumptions 2.2, 2.3 and 3.2 hold. Then:

<!-- formula-not-decoded -->

The detailed proof is in Appendix D.2.

## 6 A Discussion on Threshold δ

The threshold δ utilized in [24] is set at R max / (1 -γ ) for the hard truncation operator. This value ensures that the true optimal value function Q ∗ is not excluded from the candidate function space under the Bellman completeness condition.

In our context, consider a function class

<!-- formula-not-decoded -->

We can ensure that Q ∗ τ ∈ Q δ , provided that Q ∗ τ is within the subspace of the linear span of the basis functions { ϕ i } d i =1 , by choosing δ ≥ δ 0 .

We assert that the projected smooth truncated optimal regularized Bellman backup of a state-action value function ˆ Q , expressed as Π B τ,δ ˆ Q , is also an element of Q δ for each ˆ Q ∈ Q δ , i.e., Π B τ,δ ˆ Q ∈ Q δ , ∀ ˆ Q ∈ Q δ . Given that Q δ is both convex and compact, the Brouwer fixed-point theorem ensures the existence of at least one fixed point ˆ Q ∗ ∈ Q δ such that ˆ Q ∗ = Π B τ,δ ˆ Q ∗ . It is noteworthy that this assurance is absent when dealing with the composed operator Π B τ where truncation is not present.

Unfortunately, Q ∗ τ is not a fixed point of Π B τ,δ even when Q ∗ τ ∈ Q δ , except for the case where V ∗ τ = 0 . Recall that the error induced by the smooth truncation operator is also accounted for in the bounds specified in Theorems 5.7. To address this challenge, choosing a larger value of δ is beneficial due to the following inequality:

<!-- formula-not-decoded -->

This suggests that by opting for a sufficiently large δ , the operator B τ,δ can be made to closely approximate B τ , reducing the bias in Theorem 5.7. For instance, opting for δ = δ 0 gives us the gap δ 0 -K δ ( δ 0 ) ≈ 0 . 2384 δ 0 . By selecting a threshold of δ = 10 δ 0 , we can significantly reduce this error, as demonstrated by δ 0 -K δ ( δ 0 ) ≈ 0 . 0033 δ 0 .

The drawbacks of a large threshold value for δ are two-fold. Firstly, a larger value of δ increases the ℓ 2 -norms of both the stochastic gradient h t g and the optimal point ω ∗ ( θ t ) in the lower-level problem. This leads to a higher variance in both the main network and the target network, affecting the convergence bounds as outlined in Theorem 5.3. More specifically, the bound in (13a) increases linearly, and the bound in (13b) increases quadratically with δ when δ is large. Secondly, an increase in δ correlates with potential higher approximation error, as defined in (15). This escalation adversely affects the accuracy of the estimated value functions, potentially undermining the performance of the resulting policies in the worst case.

A smaller δ , similar to the effect of the coefficient τ , reduces the smoothness of the function J , as detailed in Lemma 1. More importantly, the application of the smooth truncation operator K δ subtly alters the discount factor γ . This modification is different from the methods used in [28,29], which uniformly reduce the discount factor across all states. In contrast, the use of K δ results in a more significant impact on states s with values | G ∗ τ ( ˆ Q θ ( s, · )) | close to δ , for which z ( s ; τ, δ, θ ) is small, while states with values closed to 0 are affected to a lesser extent. However, a small δ affects more states than a large δ and consequently introduces a larger bias in the bounds stated in Theorem 5.7, as it effectively transforms the problem into one with a significantly smaller discount factor.

## 7 Numerical Illustration

In this section, we explore two environments: GridWorld and MountainCar-v0 . We begin with the GridWorld environment, where we elucidate the influences of the smooth truncation operator K δ on the fixed points. Graphical

0 -

0 -

0 -

0 -

1 -

1 -

1 -

1 -

1 -

2 -

2 -

2 -

2-

2-

2 -

3 1

3 -

3 -

3 -

3 -

3 -

4 -

4

4 -

4 -

4-

4 -

12.6

20.5

20.7

0.3

20.1

20.4

12.5

20.4

0.5

21.1

20.0

20.4

20.5

20.1

0.3

12.3

19.7

20.0

11.9

19.6

19.5

0.1

19.1

19.4

18.9

18.6

0.0

11.2

18.3

18.6

0

20.0

20.4

21.1

0.5

12.5

20.4

20.5

21.7

1.0

12.7

20.2

20.5

12.6

20.4

21.0

0.5

20.1

20.4

20.0

0.2

20.2

20.1

12.3

19.7

11.9

19.5

19.6

0.1

19.1

19.4

1

1

1

1

i

1

20.0

12.3

20.1

19.7

0.3

20.5

12.6

21.0

20.1

20.4

0.5

20.4

20.5

20.2

12.7

0.4

20.6

20.9

20.4

0.5

20.1

21.0

20.4

12.6

20.5

0.3

12.3

20.1

20.0

19.7

2

2

2

2

2

2

19.4

0.1

19.6

19.1

11.9

19.5

20.1

0.2

20.2

19.7

12.3

20.0

20.4

12.6

0.5

20.1

21.0

20.4

18.9

0.0

18.6

18.3

11.2

18.6

19.6

19.5

0.1

19.1

11.9

19.4

20.0

12.3

19.7

20.5

0.3

20.1

1.0

• 12

- 21

- 21

- 21

- 21

- 0.8

- 21

- 21.

- 21

• 12

- 21

• 12

- 20

• 20

- 20

- 20

- 0.6

• 12

- 20

- 20

- 20

- 20

- 11

-0.4

Figure 1: Rewards and Values Maps.

<!-- image -->

illustrations complement our discussion, offering a visual representation of the convergence of the estimation from the proposed algorithm and illustrating the reduction in MSPBE. Subsequently, we assess the policies derived from our algorithm within the MountainCar-v0 environment and compare their performance with policies stemming from existing algorithms.

## 7.1 GridWorld

We investigate the influence of the smooth truncation operator within a discrete GridWorld setting, characterized by a 5 × 5 state space and five possible actions: up, down, left, right, and stay. We consider the context of an entropyregularized Markov Decision Process (MDP), employing a regularizer G ( π ( · | s )) = ⟨ π ( · | s ) , log π ( · | s ) ⟩ with a coefficient τ = 1 , and a discount factor γ = 0 . 9 . The dynamics of agent movement are straightforward. The agent transitions between states as directed by its chosen action, except when attempting to cross the grid boundaries, where it remains at its current state. The initial distribution is set to be uniform across all states, a setup that extends to the behavior policy as the probability of each possible action is the same. Consequently, each state-action pair in the space S × A has an equal likelihood of occurrence, leading to a stationary distribution µ bhv ( s, a ) = 1 |S|×|A| = 1 125 for all ( s, a ) . The reward for each state is depicted in Fig. 1a.

We adopt second-order polynomial functions of the state as basis functions. Fig. 1b depicts the optimal regularized value function V τ . We conduct a comparison of the fixed point and the projected regularized Bellman equation, exploring scenarios both with and without the incorporation of a smooth truncation operator, all within the context of linear function approximation. The threshold of the smooth truncation K δ is defined as δ = cδ 0 = c (1+log 5) 1 -γ , with c taking values of 1 , 10 , and 30 . From the plots, it is evident that a small threshold ( c = 1) , as shown in Fig. 1d, introduces a significant bias into the estimation of the value function, in comparison to the fixed point obtained without applying a smooth truncation operator, depicted in Fig. 1c. Increasing the value of δ effectively mitigates this issue, as illustrated in Fig. 1e and Fig. 1f.

Next, we evaluate the convergence of Algorithm 1. Our focus is on evaluating the Mean-Square Projected Bellman Error (MSPBE) within the regularized MDP M τ , characterized by

<!-- formula-not-decoded -->

- 19.

- 19

- 19

- 19

1.0

Figure 2: MSPBE of the estimated state-action value functions. The graph shows the average MSPBE ( ± standard deviation) over 100 runs.

<!-- image -->

Table 1: Performance of Benchmark Algorithms over 20 Runs.

|   d | Q-learning           | CQL                  | Double QL            | Algorithm 1 , τ = 0 . 01   | Algorithm 1 , τ = 0 . 05   |
|-----|----------------------|----------------------|----------------------|----------------------------|----------------------------|
|  30 | - 177 . 28 ± 32 . 00 | - 199 . 37 ± 06 . 27 | - 177 . 51 ± 33 . 73 | - 144 . 36 ± 26 . 46       | - 177 . 83 ± 30 . 09       |
|  60 | - 143 . 08 ± 32 . 47 | - 200 . 00 ± 00 . 00 | - 132 . 76 ± 31 . 54 | - 121 . 01 ± 31 . 82       | - 123 . 84 ± 28 . 35       |
|  90 | - 175 . 40 ± 17 . 77 | - 200 . 00 ± 00 . 00 | - 146 . 78 ± 22 . 74 | - 121 . 37 ± 19 . 84       | - 120 . 31 ± 19 . 20       |
| 120 | - 138 . 74 ± 11 . 00 | - 199 . 12 ± 03 . 97 | - 146 . 23 ± 12 . 77 | - 176 . 17 ± 22 . 65       | - 145 . 15 ± 29 . 48       |
| 150 | - 141 . 04 ± 24 . 77 | - 200 . 00 ± 00 . 00 | - 155 . 60 ± 19 . 43 | - 139 . 58 ± 20 . 59       | - 120 . 17 ± 20 . 64       |
| 180 | - 135 . 62 ± 32 . 01 | - 196 . 32 ± 13 . 61 | - 110 . 66 ± 20 . 87 | - 120 . 88 ± 22 . 50       | - 122 . 64 ± 22 . 01       |

We apply Algorithm 1 to data gathered using π bhv , selecting parameters α = 0 . 05 and β = 0 . 5 . Fig. 2 demonstrates that Algorithm 1 achieves convergence even with a large threshold ( c = 30 ). This figure also elucidates the inherent trade-off between the convergence rate and the error induced by the smooth truncation operator. Specifically, a lower threshold facilitates quicker convergence but results in a higher MSPBE post-convergence. Conversely, increasing the threshold yields a reduction in MSPBE at the expense of a slightly slower convergence rate.

## 7.2 MountainCar-v0

The MountainCar-v0 environment, a standard control task available in OpenAI Gym [49], challenges an agent to maneuver a car placed between two hills to ascend the hill on the right. The car is characterized by a two-dimensional state space that encompasses its horizontal position and velocity. The agent can choose from three distinct actions at each timestep: accelerate to the left, accelerate to the right, or do nothing. For every timestep that elapses before the car reaches the summit, the agent incurs a reward of -1 . Consequently, the agent's objective is to reach the summit as fast as possible to maximize its cumulative reward. Given that the episode is capped at 200 timesteps, the agent's return is confined within the range of [ -200 , 0] .

We employ Radial Basis Functions (RBFs) as our basis functions and incorporate l Gaussian kernels, each with a width of 1 , randomly selected from the state space. Since each RBF corresponds to three distinct actions, the total number of features is d = 3 l . We set the stepsize parameters to α = 0 . 1 and β = 0 . 1 , and the threshold for Algorithm 1 is assigned a value of δ = 500 . Considering the environment's finite horizon, we adopt a discount factor of γ = 1 .

We evaluated the policies derived from Algorithm 1 with a negative entropy regularizer and compared them with those obtained through Q-learning [50], Coupled Q-learning (CQL) [26], Greedy-GQ [20], and Double Q-learning (Double QL) [18]. We selected the coefficient of the regularizer for Algorithm 1 from the set { 0 . 01 , 0 . 05 } . For Algorithm 1, we adopted the regularized policy π θ defined in 3 as the behavioral policy. In contrast, an ϵ -greedy policy, with ϵ = 0 . 1 , served as the behavioral policy for the other mentioned algorithms.

Table 1 summarizes the policy performances, with the exception of Greedy-GQ, which consistently registered a return of -200 across all testing episodes. Each of the 20 training phases, differentiated by random seeds, comprised 1 , 000 episodes. The table displays the average returns obtained during testing, averaged over 20 runs, each encompassing 10 episodes. The results demonstrate that the policies derived from Algorithm 1 outperform those obtained from other algorithms in most cases.

## References

- [1] N. Tishby and D. Polani, Information Theory of Decisions and Actions . New York, NY: Springer New York, 2011, pp. 601-636.
- [2] P. A. Ortega and D. A. Braun, 'Thermodynamics as a theory of decision-making with information-processing costs,' Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences , vol. 469, no. 2153, p. 20120683, 2013.
- [3] F. Matejka and A. McKay, 'Rational inattention to discrete choices: A new foundation for the multinomial logit model,' American Economic Review , vol. 105, no. 1, p. 272-98, January 2015.
- [4] L. P. Hansen and J. Miao, 'Aversion to ambiguity and model misspecification in dynamic stochastic environments,' Proceedings of the National Academy of Sciences , vol. 115, no. 37, pp. 9163-9168, 2018.
- [5] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, 'Trust region policy optimization,' in International conference on machine learning . PMLR, 2015, pp. 1889-1897.
- [6] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, 'Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,' in International conference on machine learning . PMLR, 2018, pp. 1861-1870.
- [7] M. Geist, B. Scherrer, and O. Pietquin, 'A theory of regularized markov decision processes,' in International Conference on Machine Learning . PMLR, 2019, pp. 2160-2169.
- [8] W. Yang, X. Li, and Z. Zhang, 'A regularized approach to sparse optimal policy in reinforcement learning,' Advances in Neural Information Processing Systems , vol. 32, 2019.
- [9] B. Eysenbach and S. Levine, 'Maximum entropy RL (provably) solves some robust RL problems,' arXiv preprint arXiv:2103.06257 , 2021.
- [10] T. Haarnoja, H. Tang, P. Abbeel, and S. Levine, 'Reinforcement learning with deep energy-based policies,' in Proceedings of the 34th International Conference on Machine Learning - Volume 70 , ser. ICML'17. JMLR.org, 2017, p. 1352-1361.
- [11] K. Lee, S. Choi, and S. Oh, 'Sparse Markov decision processes with causal sparse Tsallis entropy regularization for reinforcement learning,' IEEE Robotics and Automation Letters , vol. 3, no. 3, pp. 1466-1473, 2018.
- [12] A. Martins and R. Astudillo, 'From softmax to sparsemax: A sparse model of attention and multi-label classification,' in International conference on machine learning . PMLR, 2016, pp. 1614-1623.
- [13] L. Baird, 'Residual algorithms: Reinforcement learning with function approximation,' in Machine Learning Proceedings 1995 . Elsevier, 1995, pp. 30-37.
- [14] M. Hong, H.-T. Wai, Z. Wang, and Z. Yang, 'A two-timescale stochastic algorithm framework for bilevel optimization: Complexity analysis and application to actor-critic,' SIAM Journal on Optimization , vol. 33, no. 1, pp. 147-180, 2023.
- [15] S. Meyn, 'Stability of q-learning through design and optimism,' arXiv preprint arXiv:2307.02632 , 2023.
- [16] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction . MIT Press, 2018.
- [17] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al. , 'Human-level control through deep reinforcement learning,' Nature , vol. 518, no. 7540, pp. 529-533, 2015.
- [18] H. Hasselt, 'Double q-learning,' Advances in neural information processing systems , vol. 23, 2010.
- [19] D. Ernst, P. Geurts, and L. Wehenkel, 'Tree-based batch mode reinforcement learning,' Journal of Machine Learning Research , vol. 6, 2005.
- [20] H. R. Maei, C. Szepesvári, S. Bhatnagar, and R. S. Sutton, 'Toward off-policy learning control with function approximation.' in ICML , vol. 10, 2010, pp. 719-726.
- [21] S. Chen, A. M. Devraj, F. Lu, A. Busic, and S. Meyn, 'Zap q-learning with nonlinear function approximation,' Advances in Neural Information Processing Systems , vol. 33, pp. 16 879-16 890, 2020.

- [22] A. M. Devraj and S. Meyn, 'Zap q-learning,' Advances in Neural Information Processing Systems , vol. 30, 2017.
- [23] S. Meyn, Control systems and reinforcement learning . Cambridge University Press, 2022.
- [24] Z. Chen, J. P. Clarke, and S. T. Maguluri, 'Target network and truncation overcome the deadly triad in q -learning,' arXiv preprint arXiv:2203.02628 , 2022.
- [25] S. Ma, Z. Chen, Y. Zhou, and S. Zou, 'Greedy-GQ with variance reduction: Finite-time analysis and improved complexity,' arXiv preprint arXiv:2103.16377 , 2021.
- [26] D. Carvalho, F. S. Melo, and P. Santos, 'A new convergent variant of q-learning with linear function approximation,' Advances in Neural Information Processing Systems , vol. 33, pp. 19 412-19 421, 2020.
- [27] V. S. Borkar, 'Stochastic approximation with two time scales,' Systems &amp; Control Letters , vol. 29, no. 5, pp. 291-294, 1997.
- [28] H.-D. Lim, D. Lee et al. , 'RegQ: Convergent q-learning with linear function approximation using regularization,' 2023.
- [29] S. Zhang, H. Yao, and S. Whiteson, 'Breaking the deadly triad with a target network,' in International Conference on Machine Learning . PMLR, 2021, pp. 12 621-12 631.
- [30] R. S. Sutton, H. R. Maei, D. Precup, S. Bhatnagar, D. Silver, C. Szepesvári, and E. Wiewiora, 'Fast gradientdescent methods for temporal-difference learning with linear function approximation,' in Proceedings of the 26th annual international conference on machine learning , 2009, pp. 993-1000.
- [31] Y. Wang and S. Zou, 'Finite-sample analysis of Greedy-GQ with linear function approximation under Markovian noise,' in Conference on Uncertainty in Artificial Intelligence . PMLR, 2020, pp. 11-20.
- [32] Y. Wang, Y. Zhou, and S. Zou, 'Finite-time error bounds for Greedy-GQ,' arXiv preprint arXiv:2209.02555 , 2022.
- [33] T. Xu and Y. Liang, 'Sample complexity bounds for two timescale value-based reinforcement learning algorithms,' in International Conference on Artificial Intelligence and Statistics . PMLR, 2021, pp. 811-819.
- [34] J. Bhandari, D. Russo, and R. Singal, 'A finite time analysis of temporal difference learning with linear function approximation,' in Conference on learning theory . PMLR, 2018, pp. 1691-1692.
- [35] H. Shen, K. Zhang, M. Hong, and T. Chen, 'Asynchronous advantage actor critic: Non-asymptotic analysis and linear speedup,' 2020.
- [36] Z. Chen, S. Zhang, T. T. Doan, S. T. Maguluri, and J.-P. Clarke, 'Performance of q-learning with linear function approximation: Stability and finite-time analysis,' arXiv preprint arXiv:1905.11425 , p. 4, 2019.
- [37] D. Lee and N. He, 'A unified switching system perspective and ode analysis of q-learning algorithms,' arXiv preprint arXiv:1912.02270 , 2019.
- [38] F. S. Melo, S. P. Meyn, and M. I. Ribeiro, 'An analysis of reinforcement learning with function approximation,' in Proceedings of the 25th international conference on Machine learning , 2008, pp. 664-671.
- [39] T. Xu, S. Zou, and Y. Liang, 'Two time-scale off-policy TD learning: Non-asymptotic analysis over markovian samples,' Advances in Neural Information Processing Systems , vol. 32, 2019.
- [40] T. Xu, Z. Wang, and Y. Liang, 'Non-asymptotic convergence analysis of two time-scale (natural) actor-critic algorithms,' arXiv preprint arXiv:2005.03557 , 2020.
- [41] Y. Wu, W. Zhang, P. Xu, and Q. Gu, 'A finite-time analysis of two time-scale actor-critic methods,' in Proceedings of the 34th International Conference on Neural Information Processing Systems , ser. NIPS '20, 2020.
- [42] V. S. Borkar and S. Pattathil, 'Concentration bounds for two time scale stochastic approximation,' in 2018 56th Annual Allerton Conference on Communication, Control, and Computing (Allerton) . IEEE, 2018, pp. 504-511.
- [43] S. Bhatnagar and K. Lakshmanan, 'Multiscale q-learning with linear function approximation,' Discrete Event Dynamical Systems , vol. 26, pp. 477-509, 2015.
- [44] J. Zhang, T. He, S. Sra, and A. Jadbabaie, 'Why gradient clipping accelerates training: A theoretical justification for adaptivity,' arXiv preprint arXiv:1905.11881 , 2019.
- [45] B. Zhang, J. Jin, C. Fang, and L. Wang, 'Improved analysis of clipping algorithms for non-convex optimization,' Advances in Neural Information Processing Systems , vol. 33, pp. 15 511-15 521, 2020.
- [46] S. Zeng, C. Li, A. Garcia, and M. Hong, 'Maximum-likelihood inverse reinforcement learning with finite-time guarantees,' arXiv preprint arXiv:2210.01808 , 2022.
- [47] J. N. Tsitsiklis and B. Van Roy, 'An analysis of temporal-difference learning with function approximation,' IEEE TRANSACTIONS ON AUTOMATIC CONTROL , vol. 42, no. 5, 1997.

- [48] A. Zanette, 'When is realizability sufficient for off-policy reinforcement learning?' arXiv preprint arXiv:2211.05311 , 2022.
- [49] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, 'OpenAI gym,' arXiv preprint arXiv:1606.01540 , 2016.
- [50] C. J. Watkins and P. Dayan, 'Q-learning,' Machine learning , vol. 8, pp. 279-292, 1992.

## Appendix A Auxiliary Lemmas

Lemma A1. Under Assumptions 2.2 and 2.3, ω ∗ ( · ) is Lipschitz with respect to θ ∈ R d with constant γ/λ g .

Proof. The gradient of ω ∗ ( θ ) with respect to θ ∈ R d is

<!-- formula-not-decoded -->

The ℓ 2 -norm of the gradient of ω ∗ ( θ ) is bounded as ∥∇ θ ω ∗ ( θ ) ∥ ≤ 1 /λ g . According to the mean value theorem, we have

<!-- formula-not-decoded -->

Lemma A2. Under Assumption 2.3, the policy π θ ( · | s ) is Lipschitz in θ ∈ R d with constant L G √ |A| /τ for s ∈ S .

Proof. By the properties of the regularizer and Assumption 2.3, for s ∈ S and θ 1 , θ 2 ∈ R d , we have

<!-- formula-not-decoded -->

Lemma A3. Under Assumption 2.3, the matrix ˆ Σ τ,δ,θ is Lipschitz in θ ∈ R d with constant ( L G |A| /τ +2 /δ ) .

Proof. We first derive the Lipschitz property for ˆ G τ,δ ( ˆ Q θ ( s ′ , · )) 2 with respect to θ . The derivative of the expression with respect to θ is given by

<!-- formula-not-decoded -->

The ℓ 2 -norm of this derivative is bounded by 2 δ . Thus, by the mean value theorem, for s ∈ S and θ 1 , θ 2 ∈ R d , we obtain

<!-- formula-not-decoded -->

Next, we establish the Lipschitz property for ˆ Σ τ,δ,θ . Given the definition of ˆ Σ τ,δ,θ in ( ?? ), for θ 1 , θ 2 ∈ R d , we have

<!-- formula-not-decoded -->

where the last inequality is due to Lemma A2 and (18).

Lemma A4. Under Assumption 2.3, two consecutive iterates θ t and θ t +1 in Algorithm 1 for t = 0 , 1 , . . . , T -1 satisfy ∥ θ t +1 -θ t ∥ ≤ 2 α .

̸

Proof. The proof for the case of ω t +1 = θ t is trivial, since θ t +1 = θ t . On the other hand, when ω t +1 = θ t ,

<!-- formula-not-decoded -->

Lemma A5. Under Assumption 2.3, for ω, θ ∈ R d , the difference between the gradient and the surrogate gradient of J ( · ) is bounded: ∥ ∥ ∇ J ( θ ) -¯ ∇ θ f ( θ, ω ) ∥ ∥ ≤ 2 ∥ ω ∗ ( θ ) -ω ∥ .

Proof. Given the definitions of the gradient ∇ J ( θ ) and the surrogate gradient ¯ ∇ θ f ( θ, ω ) in (7) and (11), respectively, we have

<!-- formula-not-decoded -->

## Appendix B Proofs of Theorems

## B.1 Proof of Theorem 5.3

Let F t := σ { ω 0 , θ 0 , . . . , ω t , θ t } and F ′ t := σ { ω 0 , θ 0 , . . . , ω t , θ t , ω t +1 } be the filtration of the random variables up to iteration t , where σ {·} denotes the σ -algebra generated by the random variables. Under Assumption 2.2, we define T α as the number of iterations needed to ensure the bias incurred by the Markovian sampling is sufficiently small:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first start with the error in the lower-level problem. The following lemma demonstrates the tracking error of the estimated main network ˆ Q ω t .

Lemma B1. (Tracking error). Given the conditions stated in Theorem 5.3,

<!-- formula-not-decoded -->

See Appendix C.4 for a detailed proof. As a result, by Jensen's inequality, we can derive

<!-- formula-not-decoded -->

Now, we provide the proof for the second bound in Theorem 5.3. Toward this end, we use the following lemma.

Lemma B2. Given the conditions stated in Theorem 5.3,

<!-- formula-not-decoded -->

See Appendix C.5 for a detailed proof.

Finally, it is straightforward to derive (14). By the triangle inequality and Lemma A5, we have

<!-- formula-not-decoded -->

The proof is concluded.

We define T β in a similar way:

## B.2 Proof of Theorem 5.7

We start with the first bound in Theorem 5.7. For t &gt; 0 , the triangle inequality yields

<!-- formula-not-decoded -->

For the first term in (22), due to regularized Bellman equation and the γ -contraction property of the smooth truncated optimal regularized Bellman operator B τ,δ in ℓ ∞ -norm, we obtain

<!-- formula-not-decoded -->

where the last inequality follows from the fact ∥ V π,τ ∥ ∞ ≤ R max + τB 1 -γ = δ 0 for any policy π and (4). Next, we can bound the second term in (22) by the definition of approximation error in (15):

<!-- formula-not-decoded -->

Finally, we bound the last term in (22). Under Assumption 3.2 and given the expression of ∇ J ( · ) , we have

<!-- formula-not-decoded -->

Then, provided the equation ˆ Q ω ∗ ( θ ) = Π B τ,δ ˆ Q θ , we obtain

<!-- formula-not-decoded -->

Substituting (23), (24) and (25) into (22) leads to

<!-- formula-not-decoded -->

Taking full expectation and summing from t = 0 to t = T -1 for the above inequality completes the proof of (16). For the second bound in Theorem 5.7, given the definition of V t τ , we have

<!-- formula-not-decoded -->

For the first term in (26), we utilize the properties of the convex conjugate G ∗ of the regularizer:

<!-- formula-not-decoded -->

where the first inequality follows from Taylor's theorem and the second inequality is due to Proposition 2.1. For the second term in (26), the bound can be developed similarly. Given definition of π in (3), for s ∈ S

θ , we have

<!-- formula-not-decoded -->

Therefore, by the definition of V π,τ and Q π,τ , we can derive the bound

<!-- formula-not-decoded -->

Here, our focus is on providing a bound for the first term, as bounds for the second and third terms have already been established in (24) and (25). Given the regularized Bellman equation, we can derive

<!-- formula-not-decoded -->

where the last inequality follows from the 1 -Lipschitz property of K δ . Substituting (29), (24) and (25) into (28) leads to

<!-- formula-not-decoded -->

Combining the above inequality with (27) and using the result in (16) complete the proof of Theorem 5.7.

## Appendix C Proofs of Other Lemmas

## C.1 Proof of Lemma 4.1

With δ = δ 0 , we now show sup t max a ∈A { | ϕ ( s t +1 , a ) ⊤ θ t | } &lt; ∞ w.p. 1. By contradiction, we consider two cases separately:

<!-- formula-not-decoded -->

Let us consider case (a), i.e. there is a T &lt; ∞ such that max a ∈A { | ϕ ( s t +1 , a ) ⊤ θ t | } &gt; δ 0 + τB for all t ≥ T . By Proposition 2.1.(i), we have | G ∗ τ ( ˆ Q θ t ( s t +1 , · )) | ≥ δ 0 , hence, truncation is implemented for all t ≥ T and:

<!-- formula-not-decoded -->

for all t ≥ T . By ergodicity assumption, ω t → ω ∗ and θ t → ω ∗ where

<!-- formula-not-decoded -->

However, | ˆ G τ,δ ( ˆ Q ω ∗ ( s t +1 , · )) ) | &lt; δ 0 . A contradiction. For case (b), let { θ t ( n ) } be a subsequence such that max a ∈A { | ϕ ( s t +1 , a ) ⊤ θ t ( n ) | } →∞ . Let τ ( n ) be defined as

<!-- formula-not-decoded -->

By Lemma A.4 in the paper, for all t &gt; 0 , ∥ θ t +1 -θ t ∥ ≤ 2 α . Hence,

<!-- formula-not-decoded -->

By hypothesis max a ∈A { | ϕ ( s t +1 , a ) ⊤ θ t ( n ) | } →∞ , hence t ( n ) -τ ( n ) →∞ . As in the proof of case (a),

<!-- formula-not-decoded -->

and lim n ω t ( n ) = ω ∗ and lim n θ t ( n ) = ω ∗ . A contradiction.

## C.2 Proof of Lemma 5.1

We first prove the first inequality in Lemma 5.1. By the expression of ∇ J in (7), for θ 1 , θ 2 ∈ R d , we have ∥∇ J ( θ 1 ) -∇ J ( θ 2 ) ∥

<!-- formula-not-decoded -->

where the third inequality is due to Lemma A1 and Lemma A3, and the last inequality follows from the definition of ˆ Σ τ,δ,θ in ( ?? ).

For the second inequality in Lemma 5.1, we utilize the integral remainder term from the Taylor expansion and the preceding inequality:

<!-- formula-not-decoded -->

This completes the proof.

## C.3 Proof of Lemma 5.2

By the definition of h t g in (10), we have

<!-- formula-not-decoded -->

where the last inequality is due to the projection operator in the update (12a) and the fact λ g ≤ 1 .

## C.4 Proof of Lemma B1

To start, we define and

<!-- formula-not-decoded -->

ξ t g has the following properties.

Lemma C1. For t ≥ 0 , ξ t g ( θ, ω ) is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of Lemma C1 is in Appendix C.6.

For t ≥ T α , following the update of ω t +1 in (12a) and taking the expectation yields

<!-- formula-not-decoded -->

where the first inequality follows from the property of the projection P and the last inequality is due to Proposition E.1. We first analyze the first term in (30). If β ≥ 1 /λ g , this term can be upper bounded by 0 and the subsequent results are easy to derive. If otherwise, meaning β &lt; 1 /λ g , we have

<!-- formula-not-decoded -->

where the first inequality is due to Young's inequality (and holds for any c &gt; 0 ), the second one follows from combining Lemma A1 and Lemma A4, and the last one is obtained by setting c = 2(1 -βλ g ) βλ g .

For the second term in (30), we first define a random variable ˜ ξ g ( θ, ω ) :

<!-- formula-not-decoded -->

where (˜ s, ˜ a, ˜ s ′ ) ∼ D . Given t ≥ T α , we have

<!-- formula-not-decoded -->

where the first inequality is due to Lemma C1, Assumption 2.2, and Lemma 1 in [35].

The last term in (30) can be bounded by Lemma 5.2 as

<!-- formula-not-decoded -->

Taking the full expectation of (32) and substituting it with (31), (33) into (30) yields

<!-- formula-not-decoded -->

Summing it from t = T α +1 to t = T and rearranging the it gives

<!-- formula-not-decoded -->

Given the definition of T α in (19) and the choice of α , we obtain T α = O (log T ) . Thus, we can derive

<!-- formula-not-decoded -->

Combining (35) and (36) yields

<!-- formula-not-decoded -->

where the last equality is given by α = αT -3 / 4 and β = βT -1 / 2 .

## C.5 Proof of Lemma B2

Define and

<!-- formula-not-decoded -->

ξ t g has the following properties.

Lemma C2. For t ≥ 0 ,

1. ξ t f ( θ, ω ) is 24 -Lipschitz in ω for ω ∈ Ω ;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof for Lemma C2 is in Appendix C.7.

̸

We first consider the case of ω t +1 = θ t . By Lemma 5.1, for t ≥ T α , we have

<!-- formula-not-decoded -->

where the second inequality follows from Lemma A4. Under Assumption 3.2, the ℓ 2 -norm of the surrogate gradient is lower bounded as

<!-- formula-not-decoded -->

Thus we can bound the first term on the right-hand side of (37) as

<!-- formula-not-decoded -->

For the second term on the right-hand side of (37),

<!-- formula-not-decoded -->

where the last inequality is due to Lemma A5.

Next, we focus on the third term on the right-hand side of (37). We define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last term in (37) can be bounded as

<!-- formula-not-decoded -->

where the last inequality follows from (38).

Taking full expectation in (37) and plugging (39), (40), (41) and the preceding inequality yield

<!-- formula-not-decoded -->

Observe that the above inequality also holds for the case of ω t +1 = θ t . Given the conditions

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

Summing this inequality from t = T β +1 to t = T -1 gives:

<!-- formula-not-decoded -->

To continue, we consider the first term in (42). By the triangle inequality and Lemma A4, we have

<!-- formula-not-decoded -->

Given the definition of the objective function J in (5), we obtain

<!-- formula-not-decoded -->

where the last two inequalities follow from Young's inequality.

For the second term in (42), we bound it by using (21):

<!-- formula-not-decoded -->

Substituting (43) and (44) into (42) and dividing ασ min / 4 on both sides yield that

<!-- formula-not-decoded -->

For t ≤ T β , we can bound the ℓ 2 -norm of the surrogate gradient as

<!-- formula-not-decoded -->

where the last inequality follows from Lemma A4. As a result, we have

<!-- formula-not-decoded -->

Combining (45) and (46), and dividing T lead to

<!-- formula-not-decoded -->

where the last equality is given by α = αT -3 / 4 and β = βT -1 / 2 .

## C.6 Proof of Lemma C1

We start with the first property. For ω 1 , ω 2 ∈ Ω , θ ∈ R d , it has

<!-- formula-not-decoded -->

Next, we prove the second property. For ω ∈ Ω , θ 1 , θ 2 ∈ R d , it follows that

<!-- formula-not-decoded -->

where the last inequality follows from

<!-- formula-not-decoded -->

and Lemma A1.

## C.7 Proof of Lemma C2

We start with the first property. For ω 1 , ω 2 ∈ Ω , θ ∈ R d , it has

<!-- formula-not-decoded -->

The second term on the right-hand side of the above inequality can be bounded as follows, with a similar derivation applying to the third term.

<!-- formula-not-decoded -->

Consequently, the proof is completed as

<!-- formula-not-decoded -->

Regarding the second property of ξ t f , we derive the following inequality first. For θ 1 , θ 2 ∈ R d , ω ∈ Ω ,

<!-- formula-not-decoded -->

where the third inequality follows from Lemma A3 and the last inequality is due to Assumption 3.2. Consequently, for θ 1 , θ 2 ∈ R d , ω ∈ Ω , it has

<!-- formula-not-decoded -->

## Appendix D Proofs of Corollaries

## D.1 Proof of Corollary 5.6

The proof of the result in Corollary 5.6 is straightforward. Given the triangle inequality and Lemma 5.2, we have

<!-- formula-not-decoded -->

As a result, under Assumption 3.2, we can derive

<!-- formula-not-decoded -->

where the last equality is from the Theorem 5.3.

## D.2 Proof of Corollary 5.8

Proof. For t ≥ 0 , we have

<!-- formula-not-decoded -->

Summing this inequality over t = 0 to t = T -1 and taking the expectation, we obtain

<!-- formula-not-decoded -->

The proof follows by applying Theorem 5.7.

## Appendix E Proofs of Propositions

## E.1 Proof of Proposition

Under Assumption 2.2, and with the linear independency of the basis vectors, the matrix Σ is positive-definite. Consequently, there exists a positive constant λ g that lower-bounds the eigenvalues of Σ . For ω 1 , ω 2 ∈ Ω , by Taylor theorem, we have

<!-- formula-not-decoded -->

The proof is complete.