# 3.6: Aggregation

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 427-448
**Topics:** aggregation, representative states, discretization, POMDP discretization, error bounds, feature aggregation, biased aggregation, distributed aggregation

---

Cost function difference

12

10

8

6

4

2

-0.8

-0.6

-0.4

-0.2

Linear policy parameter

Without the Newton Step

Figure 3.4.5 Illustration of the performance enhancement obtained when the Newton step/rollout is used in conjunction with an o ff -line trained (suboptimal) base policy for the linear quadratic problem; cf. Example 3.4.1. The figure shows the quadratic cost coe ffi cient di ff erences K L -K ∗ and K ˜ L -K ∗ as a function of L , where K L is the quadratic cost coe ffi cient of θ (without one-step lookahead/Newton step) and K L is the quadratic cost coe ffi cient of ˜ θ (with one-step lookahead/Newton step). The optimal performance is obtained for L ∗ ≈ -0 glyph[triangleright] 4.

<!-- image -->

ning. In other words, when the problem parameters change, as in adaptive control, there may not be enough time to retrain the parametric policy to deal with the changes . From our point of view in this book, there is also another important reason why approximation in value space is needed on top of approximation in policy space: the o ff -line trained policy may not perform nearly as well as the corresponding one-step or multistep lookahead/rollout policy , because it lacks the extra power of the associated exact Newton step (cf. our discussion of AlphaZero and TD-Gammon in Section 1.1, and linear quadratic problems in Section 1.5).

## Example 3.4.1 (A Linear-Quadratic Example)

Let us illustrate the benefit of the Newton step associated with one-step lookahead in an adaptive control setting. We consider a one-dimensional linear-quadratic problem, and we compare the performance of a linear policy with its corresponding one-step lookahead/rollout policy; cf. Fig. 3.4.5. In this example the system equation has the form

<!-- formula-not-decoded -->

and the quadratic cost function parameters are q = 1, r = 0 glyph[triangleright] 5. The optimal policy can be calculated to be

<!-- formula-not-decoded -->

with L ∗ ≈ -0 glyph[triangleright] 4, and the optimal cost function is

<!-- formula-not-decoded -->

where K ∗ ≈ 1 glyph[triangleright] 1. We want to to explore what happens when we use a policy of the form

<!-- formula-not-decoded -->

where L = L ∗ (e.g., a policy that is optimal for another system equation or cost function parameters). The cost function of θ L has the form

<!-- formula-not-decoded -->

where K L is obtained by using the formulas given in Section 1.5. When J θ is used as cost function approximation in a one-step lookahead/Newton step scheme, the cost function obtained is K ˜ L x 2 , where ˜ θ ( x ) = ˜ Lx is the corresponding one-step lookahead policy. As the figure shows, when the change L -L ∗ is substantial, the performance degradation K ˜ L -K ∗ is negligible, while the performance degradation K L -K ∗ is substantial.

## 3.5 POLICY GRADIENT AND RELATED METHODS

In this section we focus on infinite horizon problems and we discuss an alternative training approach for approximation in policy space, which is based on controller parameter optimization: we parametrize the policies by a vector r , and we optimize the corresponding expected cost over r . Thus, in contrast with the preceding section, we directly aim to approximate an optimal policy, rather than approximate a fixed policy.

For the most part in this section, we will assume no constraints on the parameter r , although we will occasionally comment regarding the presence of constraints. In particular, we determine r through the minimization

<!-- formula-not-decoded -->

where J ˜ θ ( r ) ( i 0 ) is the cost of the policy ˜ θ ( r ) starting from the initial state i 0 , and the expected value above is taken with respect to a suitable probability distribution of the initial state i 0 (cf. Fig. 3.5.1). This is to be contrasted with the classification approach of the preceding section, whereby we aim to learn a fixed policy, possibly as part of a policy iteration process. Here, we instead aim directly for an (approximately) optimal policy.

Note that in the case where the initial state i 0 is known and fixed, the method involves just minimization of J ˜ θ ( r ) ( i 0 ) over r . This simplifies a great deal the minimization, particularly when the problem is deterministic.

Similar to the preceding section, we focus on o ff -line training of policies . The on-line training of a policy involves a controller that interacts with the environment, applies controls, observes in real-time the e ff ects of these controls, and updates the policy parameters based on the observations. This challenging context is of considerable interest in practice, particularly in robotics, but is beyond our scope in this book.

/negationslash

Uncertainty System Environment Cost Control Current State certainty System Environment Cost Control Current State

Uncertainty System Environment Cost Control Current State

Uncertainty System Environment Cost Control Current State

Uncertainty System Environment Cost Control Current State

<!-- image -->

Uncertainty System Environment Cost Control Current State

Figure 3.5.1 Illustration of the policy optimization framework for an infinite horizon problem. Policies are parametrized with a parameter vector r and denoted by ˜ θ ( r ), with components ˜ θ ( i↪ r ), i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n . Each parameter value r determines a policy ˜ θ ( r ), and a cost J ˜ θ ( r ) ( i 0 ) for each initial state i 0 , as indicated in the figure. The optimization determines r through the minimization

<!-- formula-not-decoded -->

where the expected value above is taken with respect to a suitable probability distribution of i 0 .

Before delving into the details, it is worth reminding the reader that using an o ff -line trained policy without combining it with approximation in value space has the fundamental shortcoming, which we noted frequently in this book: the o ff -line trained policy will not perform nearly as well as a scheme that uses this policy in conjunction with lookahead minimization , because it lacks the extra power of the associated exact Newton step.

## 3.5.1 Gradient Methods for Parametric Cost Optimization

Let us first consider methods that perform the minimization (3.39) by using a gradient method, and for simplicity let us assume that the initial condition i 0 is known [otherwise an expected value over i 0 must be introduced in the cost function; cf. Eq. (3.39)]. Thus the aim is to minimize J ˜ θ ( r ) ( i 0 ) over r by using the gradient method

<!-- formula-not-decoded -->

assuming that J ˜ θ ( r ) ( i 0 ) is di ff erentiable with respect to r . Here γ k is a positive stepsize parameter, and ∇ ( · ) denotes gradient with respect to r evaluated at the current iterate r k .

An important concern in this method is that the gradients ∇ J ˜ θ ( r k ) ( i 0 ) may not be explicitly available. In this case, the gradients can be approximated by finite di ff erences of cost function values J ˜ θ ( r k ) ( i 0 ). Unfortunately,

when the problem is stochastic, the cost function values may be computable only through Monte Carlo simulation. This may introduce a large amount of noise, so it is likely that we will need to average many samples in order to obtain su ffi ciently accurate gradients, thereby making the method ine ffi -cient. On the other hand, when the problem is deterministic, this di ffi culty does not arise, and the use of the gradient method (3.40) or other methods that do not rely on the use of gradients (such as coordinate descent) is facilitated.

In what follows in this section we will focus on alternative and typically more e ffi cient gradient-like methods for stochastic problems, which are based on sampling, rather than the exact calculation of the gradient, as in Eq. (3.40). Before doing so, we address briefly the issue of nondi ff erentiabilities and constraints in gradient-based optimization

## Constraints and Nondi ff erentiablities - Soft-Max Approximation

The gradient method (3.40) assumes that the cost function J ˜ θ ( r ) is di ff erentiable with respect to r . Frequently, however, this is not so, and to deal with nondi ff erentiabilities, a common practice has been to use the so-called soft-max approximation to smooth the di ff erentiabilities. As an example of this approach, nondi ff erentiable terms of the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c is a positive parameter, and λ i are scalars satisfying

<!-- formula-not-decoded -->

The parameter c controls the quality of the approximation (as c increases the approximation becomes better, but also the smoothed problem becomes more ill-conditioned and harder to solve by gradient methods). The parameters λ i can be interpreted as Lagrange multipliers, which when chosen properly, improve the quality of the approximation of the smoothed function in the neighborhood of an optimal solution; see the book [Ber82a] for a detailed discussion.

The soft-max approximation is discussed within the context of a broad class of smoothing schemes in the author's book [Ber82a], Chapters 3 and 5. It was

can be approximated by

Another di ffi culty associated with the gradient method (3.40) is that there may be constraints on the parameter vector r . Gradient methods can be modified to deal with constraints, particularly in simple cases, such as nonnegativity or simplex constraints; see the discussion of Section 3.5.3 on gradient projection and proximal algorithms. However, constraints can also be dealt with by reparametrization of the policies in a way that eliminates the constraints. An important special case arises when the policies of a finite state and control spaces problem are replaced by randomized policies, which are then reparametrized to eliminate the associated probability distribution constraints by using the soft-max operation.

In particular, suppose that the set of policies is enlarged to include all randomized policies. Then, for a problem with states i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , controls u = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , and constraints u ∈ U ( i ), an ordinary policy θ , which applies control θ ( i ) ∈ U ( i ), i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , is replaced by a randomized policy that applies at state i the control u i with probability r u ( i ). This randomized policy consists of a probability distribution { r u ( i ) ♣ u ∈ U ( i ) } for each state i , where the probabilities/parameters r u ( i ) must satisfy

<!-- formula-not-decoded -->

The policy optimization aims to find an optimal probability distribution for each state i .

The important point here is that the set of randomized policies is continuous and thus far better suited for the application of the gradient optimization (3.40). On the other hand the parameters r u ( i ) must satisfy the simplex constraints (3.41), thus requiring a gradient method that can deal with constraints. The corresponding policy optimization problem can be solved by a gradient-like method that can deal with the probability constraints (3.41). Such methods will be briefly discussed in the next section. However, it is also possible to reparametrize the space of probability distributions with a transformation that approximates probabilities as follows:

<!-- formula-not-decoded -->

first proposed in the author's papers [Ber75], [Ber77]; see also the paper by Poljak [Pol79], and the paper by Nesterov [Nes05], who used the smoothing approach for the complexity analysis of gradient methods. The soft-max approximation also bears a close connection to the exponential method of multipliers, first proposed by Kort and Bertsekas [KoB72]; see the books [Ber82a], Chapter 5, and [Ber17], Chapter 6, which also discuss a duality connection with the entropy minimization algorithm, a proximal minimization algorithm that is based on an exponential penalty function.

where θ u ( i ) are the transformed parameters and φ ( i ) are some scalars that depend on i . In this way the policy optimization problem becomes unconstrained. We note, however, that there are some serious di ffi culties associated with this reparametrization. In particular, policy controls r u ( i ) that are applied with probability 1, are mapped to parameters θ u ( i ) that are equal to ∞ in the reparametrized problem; see Exercise 3.2(b). Still the reparametrization r u ( i ) ↦→ θ u ( i ) of Eq. (3.42) is used widely in practice.

## 3.5.2 Policy Gradient-Like Methods

To get a sense of the general principle underlying the incremental gradient approach that uses randomization and sampling, let us digress from the DP context of this chapter, and consider the generic optimization problem

<!-- formula-not-decoded -->

where Z is some constraint set, and F is some real-valued function. Note that F may be nondi ff erentiable, and indeed the constraint set Z may be discrete (e.g., Z may be the finite set of stationary policies in a finitespaces Markovian decision problem; see Exercise 3.2). Hence gradient methods would seem like an unlikely candidate for solution. However, we will transform the problem to one that is amenable to the use of the gradient methodology through an unusual procedure that we describe next.

We will convert the problem (3.43) to the stochastic problem

<!-- formula-not-decoded -->

where z is viewed as a random variable, P Z is the set of probability distributions over Z , p denotes the generic distribution in P Z , and E p ¶·♦ denotes expected value with respect to p . Of course this enlarges the search space from Z to P Z , but it enhances the use of randomization schemes, simulation-based methods, even if the original problem is deterministic. Moreover, the cost function of the stochastic problem (3.44) may have some nice di ff erentiability properties that are lacking in the original deterministic version (3.43). This is true in general, although under special assumptions on F , such as convexity, these di ff erentiability properties are enhanced (see the paper [Ber73]). The reader may be able to appreciate the striking di ff erences between the deterministic problem (3.43) and the stochastic problem (3.44) by working out simple examples; see Exercise 3.2.

At this point it is not clear how the stochastic optimization problem (3.44) relates to our stochastic DP context of this chapter. We will return to this question later, but for the purpose of orientation, we note that to obtain a problem of the form (3.44), we must enlarge the set of policies to include randomized policies , mapping a state i into a probability distribution over the set of controls U ( i ).

Suppose now that we restrict attention to a subset ˜ P Z ⊂ P Z of probability distributions p ( z ; r ) that are parametrized by some continuous parameter r , e.g., a vector in some Euclidean space. In other words, we approximate the stochastic optimization problem (3.44) with the restricted problem

<!-- formula-not-decoded -->

Then we may use a gradient method for solving this problem, such as

<!-- formula-not-decoded -->

where ∇ ( · ) denotes gradient with respect to r of the function in parentheses, evaluated at the current iterate r k .

## Likelihood-Ratio Policy Gradient Methods

We will first consider an incremental version of the gradient method (3.46). This method requires that p ( z ; r ) is di ff erentiable with respect to r . It relies on a convenient gradient formula, sometimes referred to as the log-likelihood trick , which involves the natural logarithm of the sampling distribution.

This formula is obtained by the following calculation, which is based on interchanging gradient and expected value, and using the gradient formula ∇ (log p ) = ∇ pglyph[triangleleft]p . We have

<!-- formula-not-decoded -->

and finally

<!-- formula-not-decoded -->

The parametrization p ( z ; r ) must be di ff erentiable with respect to r in order for the gradient to exist. One possibility, noted earlier, is to use a soft-max parametrization such as p ( z ; r ) = e r ′ φ ( z ) ∑ y ∈ Z e r ′ φ ( y ) for all z ∈ Z , where φ ( z ) denotes a vector that suitably depends on z . This may involve some pitfalls, which are

illustrated in Exercise 3.2(b).

where for any given z , ∇ ( log ( p ( z ; r ) ) ) is the gradient with respect to r of the function log p ( z ; · ) , evaluated at r (the gradient is assumed to exist).

( ) The preceding formula suggests an incremental implementation of the gradient iteration (3.46) that approximates the expected value in the right side in Eq. (3.47) with a single sample (cf. Section 3.1.3). The typical iteration of this method is as follows.

## Sample-Based Gradient Method for Parametric Approximation of min z ∈ Z F ( z )

Let r k be the current parameter vector.

- (a) Obtain a sample z k according to the distribution p ( z ; r k ).
- (b) Compute the sampled gradient ∇ ( log ( p ( z k ; r k ) ) ) glyph[triangleright]
- (c) Iterate according to

<!-- formula-not-decoded -->

The advantage of the preceding sample-based method is its simplicity and generality. It allows the use of parametric approximation for any minimization problem (well beyond DP), as long as the logarithm of the sampling distribution p ( z ; r ) can be conveniently di ff erentiated with respect to r , and samples of z can be obtained using the distribution p ( z ; r ).

Note that in iteration (3.48) r is adjusted along a random direction. This direction does not involve at all the gradient of F (even if it exists), only the gradient of the logarithm of the sampling distribution! As a result the iteration has a model-free character : we don't need to know the mathematical form of the function F as long as we have a simulator that produces the cost function value F ( z ) for any given z . This is also a major advantage o ff ered by many random search methods.

An important issue is the e ffi cient computation of the sampled gradient ∇ ( log ( p ( z k ; r k ) )) glyph[triangleright] In the context of DP, including the SSP and discounted problems that we have been dealing with, there are some specialized procedures and corresponding parametrizations to approximate this gradient conveniently. The following is an example.

## Example 3.5.1 (Policy Gradient Method for Discounted DP)

Consider the α -discounted problem and denote by z the infinite horizon statecontrol trajectory:

<!-- formula-not-decoded -->

We consider a parametrization of randomized policies with parameter r , so the control at state i is generated according to a distribution p ( u ♣ i ; r ) over U ( i ). Then for a given r , the state-control trajectory z is a random vector with probability distribution denoted p ( z ; r ). The cost corresponding to the trajectory z is

<!-- formula-not-decoded -->

and the problem is to minimize over r

<!-- formula-not-decoded -->

To apply the sample-based gradient method (3.48), given the current iterate r k , we must generate the sample state-control trajectory

<!-- formula-not-decoded -->

according to the distribution p ( z ; r k ), we compute the corresponding cost F ( z k ), and also calculate the sampled gradient

<!-- formula-not-decoded -->

Let us assume that the logarithm of the randomized policy distribution p ( u ♣ i ; r ) is di ff erentiable with respect to r (a soft-min policy parametrization is often recommended for this purpose). Then the logarithm that is di ff erentiated in Eq. (3.50) can be written as

<!-- formula-not-decoded -->

and its gradient (3.50), which is needed in the iteration (3.48), is given by

<!-- formula-not-decoded -->

This gradient involves the current randomized policy, but does not involve the system's transition probabilities and the costs per stage.

The policy gradient method (3.48) can now be implemented with a finite horizon approximation whereby r k is changed after a finite number N of time steps [so the infinite cost and gradient sums (3.49) and (3.51) are replaced by finite sums]. The method takes the form

<!-- formula-not-decoded -->

where z k N = ( i k 0 ↪ u k 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ i k N -1 ↪ u k N -1 ) is the generated N -step trajectory, F N ( z k N ) is the corresponding cost, and γ k is the stepsize. The initial state i k 0 of the trajectory is chosen randomly, with due regard to exploration issues.

Policy gradient methods for other types of DP problems can be similarly developed, as well as variations involving a combination of policy and cost function parametrizations [e.g., replacing F ( z ) of Eq. (3.49) by a parametrized estimate that has smaller variance, and possibly subtracting a suitable baseline, cf. the following discussion]. This leads to a class of actor-critic methods that di ff er from the PI-type methods that we discussed earlier in this chapter; we refer to the end-of-chapter literature for a variety of specific schemes. The policy gradient methods that we discussed in this section are sometimes called actor-only methods, as they involve only policy parametrizations.

## Implementation Issues

There are several additional issues to consider in the implementation of the sample-based gradient method (3.48). The first of these is that the problem solved is a randomized version of the original. If the method produces a parameter r in the limit and the distribution p ( z ; r ) is not atomic (i.e., it is not concentrated at a single point), then a solution z ∈ Z must be extracted from p ( z ; r ). In the SSP and discounted problems that we have been dealing with, the subset ˜ P Z of parametric distributions typically contains the atomic distributions, while it can be shown that minimization over the set of all distributions P Z produces the same optimal value as minimization over Z (the use of randomized policies does not improve the optimal cost of the problem), so this di ffi culty does not arise.

Another issue is how to improve sampling e ffi ciency. To this end, let us note a simple generalization of the gradient method (3.48), which can often improve its performance. It is based on the gradient formula

<!-- formula-not-decoded -->

where b is any scalar. This formula generalizes Eq. (3.47), where b = 0, and holds in view of the following calculation, which shows that the term multiplying b in Eq. (3.52) is equal to 0:

<!-- formula-not-decoded -->

where the last equality holds because ∑ z ∈ Z p ( z ; r ) is identically equal to 1 and hence does not depend on r .

Based on the gradient formula (3.52), we can modify the sampled gradient iteration (3.48) to read as follows:

<!-- formula-not-decoded -->

where b is some fixed scalar, called the baseline . Whereas the choice of b does not a ff ect the gradient ∇ ( E p ( z ; r ) { F ( z ) } ) [cf. Eq. (3.52)], it a ff ects the incremental gradient

<!-- formula-not-decoded -->

which is used in the iteration (3.53). Thus, by optimizing the baseline b , empirically or through a calculation (see e.g., [DNP11]), we can improve the performance of the algorithm. Moreover, in the context of discounted and SSP problems, state-dependent baseline functions have been used. Ideas of cost shaping are useful within this context; we refer to the RL textbook [Ber19a] and specialized literature for further discussion.

## 3.5.3 Scaling and Proximal Policy Optimization Methods

Gradient-like methods are notorious for their slow convergence, which is exacerbated by the detrimental e ff ects of noise (when applied in a stochastic environment). A principal technique to deal with this di ffi culty is to scale the gradient by multiplication with a positive definite matrix.

In particular, scaled gradient methods for solving general nonlinear unconstrained optimization problems of the form

<!-- formula-not-decoded -->

where f is twice continuously di ff erentiable objective function, have the form

<!-- formula-not-decoded -->

where D k is a positive definite symmetric m × m matrix (see nonlinear programming books, such as [Ber16]). Two extreme cases are:

- (a) Select D k to be the Hessian matrix of f evaluated at x k , in which case we obtain Newton's method . This method converges superlinearly when started su ffi ciently close to a strong local minimum (one where the Hessian is positive definite). An approximation to Newton's method for least squares problems where f ( x ) = ∑ N i =1 ‖ g i ( x ) ‖ 2 , is the Gauss-Newton method , which neglects the second derivatives of g i (see e.g., [Ber16] for discussions of the Gauss-Newton method).

- (b) Select D k to be a diagonal approximation to the Hessian matrix of f evaluated at x k . This method converges to a strong local minimum under appropriate conditions on D k (it should not be too 'small'). Its convergence rate is linear, and typically much slower than the superlinear convergence rate of Newton's method. On the other hand, a gradient method with diagonal scaling is much simpler than Newton's method, and more readily admits an incremental implementation of the type that we have discussed. Moreover, incremental versions of diagonally scaled gradient methods have worked well for the training of neural networks and other approximation architectures, particularly when supplemented with e ff ective heuristics; cf. the discussion of Section 3.1.3 in connection with the training of neural networks and the ADAM algorithm.

The scaled gradient iteration (3.54) can also be written in an alternative and equivalent form in terms of a quadratic optimization:

<!-- formula-not-decoded -->

This is easily verified by setting to 0 the derivative of the quadratic cost above, thereby obtaining

<!-- formula-not-decoded -->

which is equivalent to Eq. (3.55).

The form (3.55) of the scaled gradient iteration is in turn related to two other important optimization methods (see nonlinear programming books such as [Ber16]). The first of these is the gradient projection method , which applies to constrained optimization problems of the form

<!-- formula-not-decoded -->

where C is some closed convex set. It takes the form

<!-- formula-not-decoded -->

This iteration can be equivalently written as

<!-- formula-not-decoded -->

where P D k ( · ) denotes projection onto C with respect to a norm that is defined by the scaling matrix D k (hence the name 'gradient projection'). We can see this by writing

<!-- formula-not-decoded -->

and expanding the quadratic form on the right-hand side, to verify that the iterations of Eqs. (3.57) and (3.58) are equivalent. When the problem is unconstrained, i.e., C = /Rfractur m , the gradient projection iteration (3.57) is equivalent to the scaled gradient iterations (3.54) and (3.55).

Let us also mention the proximal minimization algorithm , which applies to the constrained optimization problem (3.56), and has the form

<!-- formula-not-decoded -->

It can be seen that the gradient projection method (3.57) and the proximal algorithm (3.59) have strong similarities, and that they coincide when f is linear. Note also that the two algorithms can be suitably combined in ways that fit the structure of specific problems, and admit incremental and randomized/simulation-based implementations (see the author's survey paper [Ber10d]).

A second way to describe the gradient iteration (3.55) involves the 'trust region' optimization

<!-- formula-not-decoded -->

where β k is a positive scalar. It can be shown that the methods (3.55) and (3.60) are equivalent for an appropriate value of β k (see[Ber16]). The role of β k is to regulate the size of the constraint and ensure that the increment r k +1 -r k is appropriately small. This guards against oscillations of the iterates r k , which may lead to divergence.

## Natural Gradient Methods

While the scaling techniques discussed above have worked well for the general nonlinear optimization problem min r ∈/Rfractur m f ( r ), alternative scaling techniques have been suggested for stochastic gradient methods that involve optimization over parametrized probability distributions. In place of an inverse Hessian matrix, these methods use the Fisher Information Matrix (FIM), which accounts for the curvature of the cost function induced by the parametrization.

For the case of the parametric optimization problem

<!-- formula-not-decoded -->

that we have discussed earlier in this section [cf. Eq. (3.45)], the FIM is given by

<!-- formula-not-decoded -->

where the gradient ∇ ( log ( p ( z ; r ) ) (a column vector) is computed with the formulas given earlier (cf. Example 3.5.1, for the case of discounted DP problems). The scaled version of the corresponding gradient method of Eq. (3.46) is given by

<!-- formula-not-decoded -->

with the gradient in the preceding iteration given by the earlier derived formula (3.47):

<!-- formula-not-decoded -->

The scaling induced by the FIM (3.62) in the scaled gradient method (3.63) is conceptually related to the Newton and the Gauss-Newton scaling noted earlier in connection to the scaled gradient method (3.54); see the relevant literature for more details, including the scholarly and comprehensive book by Amari [Ama16]. The scaled gradient

<!-- formula-not-decoded -->

is also known as the natural gradient for the problem (3.61). Incremental/sampled versions of the scaled gradient method (3.63) are possible along the lines discussed earlier; cf. Eq. (3.48). In particular, for a discounted DP problem of Example 3.5.1, we can use the FIM formula (3.62) (or an approximation thereof), where

<!-- formula-not-decoded -->

cf. Eq. (3.51), and z k is the infinite horizon k th sample state-control trajectory

<!-- formula-not-decoded -->

The motivation for using the FIM in parametric optimization problems of the form (3.61), is that it defines an alternative norm in the space of probability distributions, which captures the sensitivity of the distribution p ( · ; r ) with respect to the parameter r more e ff ectively than the norm defined the inverse Hessian approximations D k discussed earlier. In this regard, it should be noted that the FIM is the Hessian matrix of the Kullback-Leibler (KL) divergence, an important metric for measuring distance between two probability distributions. We refer to the papers by Amari [Ama98] and Kakade [Kak02] for further discussion. For more recent works that provide many additional references, see Deisenroth, Neumann,

and Peters [DNP11], Grondman et al. [GBL12], Fazel et al. [FGK18], Cen et al. [CCC21], Bhandari and Russo [BhR24], and Muller and Montufar [MuM24].

Finally let us note that the trust region analog of the scaled iteration (3.63) takes the form

<!-- formula-not-decoded -->

where β k is a positive scalar, in analogy with Eq. (3.60). Implementations of the policy gradient method with FIM scaling and with forms of trust region are provided by the proximal policy optimization (PPO) method, suggested by Shulman et al. [SWD17], which has been used widely, including for the training of large language models (see Section 3.5.3).

## Constrained Natural Gradient Methods

Natural gradient methods can be extended to constrained optimization problems of the form

<!-- formula-not-decoded -->

where C ⊂ /Rfractur m is a closed convex constraint set, in analogy with problem (3.56). Important special cases of constrained problems arise often in practice, such as when C consists of nonnegativity constraints:

<!-- formula-not-decoded -->

or has some other simple form, such as upper and lower bounds on the components of r , a simplex constraint, or a Cartesian product of simplexes. In what follows, we will provide e ffi cient variants of natural gradient methods for simple constraint sets, focusing primarily on the orthant constraint (3.66).

To simplify notation, let us denote by r 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r m the m components of r , and let us use an abbreviated notation for the gradient and partial derivatives of the cost function:

<!-- formula-not-decoded -->

With this notation, the unconstrained natural gradient method is given by

<!-- formula-not-decoded -->

and its projected analog [cf. Eq. (3.57)] is given by

<!-- formula-not-decoded -->

This is a valid method with strong convergence guarantees, which follow from well-established analysis of gradient projection methods for general constrained nonlinear programming problems (see e.g., [Ber16]). Unfortunately, however, the practical implementation of this method runs into some serious di ffi culties. Chief among these is that the quadratic minimization in Eq. (3.68) may be much harder than the matrix inversion of Eq. (3.67). This is true even if C is the simple orthant constraint (3.66).

To address this issue we may consider a method that involves a simpler projection. For the case of the orthant constraint (3.66), one possibility that comes to mind is

<!-- formula-not-decoded -->

where [ · ] + is the 'clipping' operation, whereby the i th component [ r ] + i of [ r ] + is obtained from the i th component r i of r according to

<!-- formula-not-decoded -->

and ˜ M ( r k ) is either the FIM M ( r k ) or a suitable approximation thereof.

Unfortunately when ˜ M ( r k ) is taken to be equal to the FIM M ( r k ), the algorithm (3.69) does not work. The reason is illustrated in Fig. 3.5.2, and has to do with the fact that when r k lies at the boundary of the orthant, the iteration may not lead to cost reduction regardless of how small the stepsize γ k is . In particular, it is possible that the unprojected iterate r k -γ k ˜ M ( r k ) -1 g ( r k ) improves the cost function for g k su ffi ciently small, but after projection onto the orthant it does not [see Fig. 3.5.2(a)]. In fact it is possible that r k is optimal, but the iteration (3.69) moves r k away from the optimal [see Fig. 3.5.2(b)].

It turns out, however, that there is a class of approximate FIM matrices ˜ M ( r k ) for which cost reduction can be guaranteed. This class is su ffi ciently wide to allow fast convergence when ˜ M ( r k ) properly embodies the essential content of the true FIM M ( r k ).

We say that a symmetric m × m matrix D with elements d ij is diagonal with respect to a subset of indices I ⊂ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ , if

/negationslash

<!-- formula-not-decoded -->

Let us denote for all r ≥ 0

<!-- formula-not-decoded -->

∣ see Fig. 3.5.2 for an example. The following is a key proposition, due to the author's works [Ber82b], [Ber82c], which shows how to select an approximation to the FIM in order to guarantee the validity of the projected iteration (3.69).

Figure 3.5.2. Two-dimensional examples where I + ( r k ) = ¶ 2 ♦ [we have r k 1 &gt; 0, while r k 2 = 0 and g 2 ( r k ) &gt; 0]. (a) A case where the iteration

<!-- image -->

<!-- formula-not-decoded -->

fails to make progress when ˜ M ( r k ) is not properly diagonalized. (b) A case where the iteration fails to stop at an optimal solution r ∗ when ˜ M ( r k ) is not properly diagonalized.

Proposition 3.5.1: Let r ≥ 0 and let ˜ M ( r ) be a positive definite symmetric matrix that is diagonal with respect to I + ( r ). Denote by r ( γ ) the iterate of the projected natural gradient iteration (3.69), starting from r , as a function of the stepsize γ :

<!-- formula-not-decoded -->

Consider a vector r that is nonstationary, in the sense that it violates the first order optimality condition

<!-- formula-not-decoded -->

Then there exists a scalar γ &gt; 0 such that the following cost improvement property holds:

<!-- formula-not-decoded -->

Based on Prop. 3.5.1 we conclude that to guarantee cost improvement, the matrix ˜ M ( r k ) in the iteration (3.69) should be chosen to be diagonal

with respect to a subset of indices that contains

<!-- formula-not-decoded -->

∣ and the stepsize g k should be chosen su ffi ciently small to guarantee cost improvement. Actually, it turns out that to ensure convergence one should implement the iteration more carefully. The reason is that the set I + ( r k ) exhibits an undesirable discontinuity at the boundary of the constraint set, whereby given a sequence ¶ r k ♦ of interior points that converges to a boundary point r the set I + ( r k ) may be strictly smaller than the set I + ( r ). This causes di ffi culties in proving convergence of the algorithm and may have an adverse e ff ect on its rate of convergence. To bypass these di ffi culties one may add to the set I + ( x k ) the indices of those variables x k i that satisfy g i ( r k ) &gt; 0 and are 'near' zero (i.e., 0 ≤ r k i ≤ /epsilon1 , where /epsilon1 is a small fixed scalar).

The theoretical properties of the projection method (3.69), with the preceding modification and a properly defined stepsize rule are solid. Moreover, extensive computational practice has established its superior performance over alternative methods of the gradient type. We refer to the author's constrained optimization book [Ber82b] (Section 1.5) and the paper [Ber82c], where the method (3.69) was first proposed under the name 'two-metric projection method;' see also Gafni and Bertsekas [GaB84]. These references, together with the book [Ber16] (Section 3.4), also discuss extensions to broader classes of problems involving linear constraints, such as upper and lower bounds on the parameters r i , and simplex constraints. For more recent works that are focused on machine learning applications, see Schmidt, Kim, and Sra [SKS12], Xie and Wright [XiW21], [XiW24], and Wu and Xie [WuX24].

An important property of the projected natural gradient method (3.69) is that it can preserve the beneficial scaling properties of the Fisher information matrix. The reason is that the method tends to quickly identify the subspace of active constraints (the ones for which r i = 0 at an optimal solution) and then reduces to the unconstrained natural gradient method on that subspace. For the same reason, the method is also well-suited for an incremental implementation.

## 3.5.4 Random Direction Methods

In this section, we will consider an alternative class of incremental versions of the policy gradient method (3.46), repeated here for convenience:

<!-- formula-not-decoded -->

These methods are based on the use of a random search direction and only two sample function values per iteration . They are generally faster than

z r r

Figure 3.5.3 The distribution p ( z ; r ) used in the gradient iteration (3.72).

<!-- image -->

methods that use a finite di ff erence approximation of the entire cost function gradient; see the book by Spall [Spa03] for a detailed discussion, and the paper by Nesterov and Spokoiny [NeS17] for a more theoretical view. Moreover, these methods do not require the derivative of the sampling distribution or its logarithm. On the other hand, these methods have not been applied widely to the training of policies in the RL field.

For simplicity we first consider the case where z and r are scalars, and later discuss the multidimensional case. In particular, we assume that p ( z ; r ) is symmetric and is concentrated with probabilities p i at the points r + /epsilon1 i and r -/epsilon1 i , where /epsilon1 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ /epsilon1 m are some small positive scalars. Thus we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

The gradient iteration (3.71) becomes

<!-- formula-not-decoded -->

Let us now consider approximation of the gradient by finite di ff erences:

<!-- formula-not-decoded -->

We approximate the gradient iteration (3.72) by

<!-- formula-not-decoded -->

One possible sample-based/incremental version of this iteration is

<!-- formula-not-decoded -->

where i k is an index generated with probabilities that are proportional to p i k . This algorithm uses one out of the m terms of the gradient in Eq. (3.73).

The extension to the case, where z and r are multidimensional, is straightforward. Here p ( z ; r ) is a probability distribution, whereby z takes values of the form r + /epsilon1 d , where d is a random vector that lies on the surface of the unit sphere, and /epsilon1 (independently of d ) takes scalar values according to a distribution that is symmetric around 0. The idea is that at r k , we first choose randomly a direction d k on the surface of the unit sphere, and then change r k along d k or along -d k , depending on the sign of the corresponding directional derivative. For a finite di ff erence approximation of this iteration, we sample z k along the line ¶ r k + /epsilon1 d k ♣ /epsilon1 ∈ /Rfractur ♦ , and similar to the iteration (3.74), we set

<!-- formula-not-decoded -->

where /epsilon1 k is the sampled value of /epsilon1 .

Let us also discuss the case where p ( z ; r ) is a discrete but nonsymmetric distribution, i.e., z takes values of the form r + /epsilon1 d , where d is a random vector that lies on the surface of the unit sphere, and /epsilon1 is a zero mean scalar. Then the analog of iteration (3.75) is

<!-- formula-not-decoded -->

where /epsilon1 k is the sampled value of /epsilon1 . Thus in this case, we still require two function values per iteration. Generally, for a symmetric sampling distribution, iteration (3.75) tends to be more accurate than iteration (3.76), and is often preferred.

Algorithms of the form (3.75) and (3.76) are known as random direction methods . They use only two cost function values per iteration, and a direction d k that need not be related to the gradient of F in any way. There is some freedom in selecting d k , which could potentially be exploited in specific schemes. Selection of the stepsize γ k and the sampling distribution for /epsilon1 can be tricky, particularly when the values of F are noisy.

## 3.5.5 Random Search and Cross-Entropy Methods

The main drawback of the policy gradient methods that we have considered so far is the risk of unreliability due to the stochastic uncertainty corrupting the calculation of the gradients, the slow convergence that is typical of

Figure 3.5.4 Schematic illustration of the cross-entropy method. At the current iterate r k , we construct an ellipsoid E k centered at r k . We generate a number of random samples within E k , and we 'accept' a subset of the samples that have 'low' cost. We then choose r k +1 to be the sample mean of the accepted samples, and construct a sample 'covariance' matrix of the accepted samples. We then form the new ellipsoid E k +1 using this matrix and a suitably enlarged radius, and continue. Notice the resemblance with a policy gradient method: we move from r k to r k +1 in a direction of cost improvement.

<!-- image -->

gradient methods in many settings, and the presence of local minima. For this reason, alternative methods based on random search are potentially more reliable alternatives. Viewed from a high level, random search methods are similar to policy gradient methods in that they aim at iterative cost improvement through sampling. However, they need not involve randomized policies, they are not subject to cost di ff erentiability restrictions, and they o ff er some global convergence guarantees, so in principle they are not a ff ected much by local minima.

Let us consider a parametric policy optimization approach based on solving the problem

<!-- formula-not-decoded -->

cf. Eq. (3.39). Random search methods for this problem explore the space of the parameter vector r in some randomized but intelligent fashion. There are several types of such methods for general optimization, and some of them have been suggested for approximate DP. We will briefly describe the cross-entropy method , which has gained considerable attention.

The method, when adapted to the approximate DP context, bears resemblance to policy gradient methods, in that it generates a parameter sequence ¶ r k ♦ by changing r k to r k +1 along a direction of 'improvement.'

This direction is obtained by using the policy ˜ θ ( r k ) to generate randomly cost samples corresponding to a set of sample parameter values that are concentrated around r k . The current set of sample parameters are then screened: some are accepted and the rest are rejected, based on a cost improvement criterion. Then r k +1 is determined as a 'central point' or as the 'sample mean' in the set of accepted sample parameters, some more samples are generated randomly around r k +1 , and the process is repeated; see Fig. 3.5.4. Thus successive iterates r k are 'central points' of successively better groups of samples, so in some broad sense, the random sample generation process is guided by cost improvement. This is a general idea that is shared with other popular classes of random search methods.

The cross-entropy method is very simple to implement, does not suffer from the fragility of gradient-based optimization, does not involve randomized policies, and relies on some supportive theory. Importantly, the method does not require the calculation of gradients, and it does not require di ff erentiability of the cost function. Moreover, it does not need a model to compute the required costs of di ff erent policies; a simulator is su ffi cient.

Like all random search methods, the convergence rate guarantees of the cross-entropy method are limited, and its success depends on domainspecific insights and the skilled use of heuristics. However, the method relies on solid ideas and has gained a favorable reputation. In particular, it was used with impressive success in the context of the game of tetris; see Szita and Lorinz [SzL06], and Thiery and Scherrer [ThS09]. There have also been reports of domain-specific successes with related random search methods; see Salimans et al. [SHC17]. We refer to the end-of-chapter literature for details and examples of implementation.

## 3.5.6 Refining and Retraining Parametric Policies

Suppose that we already have a pre-trained parametric policy, such as one implemented through a neural network, and we want to refine it. A possible motivation for this may be to correct some deficiencies of the policy, such as a bias towards undesirable behaviors. For example the policy may produce poor or unsafe responses at some sensitive states. We note that refining a pre-trained parametric policy has become the subject of intensive interest following the availability of highly capable pre-trained transformer-based software such as ChatGPT and DeepSeek.

A common way to retrain a policy is to modify the cost function of the problem in a way that penalizes the undesirable behaviors. To this end we need a special purpose/domain-specific dataset consisting of input-output pairs that are generated using human or machine feedback. The dataset may be supplemented with specific instructions on how to choose controls in special or critical situations.