## Performance Bounds for λ Policy Iteration and Application to the Game of Tetris

## Bruno Scherrer

MAIA Project-Team, INRIA Lorraine 615 rue du Jardin Botanique 54600 Villers-l` es-Nancy

FRANCE

Editor:

Shie Mannor

BRUNO.SCHERRER@INRIA.FR

## Abstract

We consider the discrete-time infinite-horizon optimal control problem formalized by Markov decision processes (Puterman, 1994; Bertsekas and Tsitsiklis, 1996). We revisit the work of Bertsekas and Ioffe (1996), that introduced λ policy iteration-a family of algorithms parametrized by a parameter λ -that generalizes the standard algorithms value and policy iteration, and has some deep connections with the temporal-difference algorithms described by Sutton and Barto (1998). We deepen the original theory developed by the authors by providing convergence rate bounds which generalize standard bounds for value iteration described for instance by Puterman (1994). Then, the main contribution of this paper is to develop the theory of this algorithm when it is used in an approximate form. We extend and unify the separate analyzes developed by Munos for approximate value iteration (Munos, 2007) and approximate policy iteration (Munos, 2003), and provide performance bounds in the discounted and the undiscounted situations. Finally, we revisit the use of this algorithm in the training of a Tetris playing controller as originally done by Bertsekas and Ioffe (1996). Our empirical results are different from those of Bertsekas and Ioffe (which were originally qualified as 'paradoxical' and 'intriguing'). We track down the reason to be a minor implementation error of the algorithm, which suggests that, in practice, λ policy iteration may be more stable than previously thought.

Keywords: stochastic optimal control, reinforcement learning, Markov decision processes, analysis of algorithms

## 1. Introduction

We consider the discrete-time infinite-horizon optimal control problem formalized by Markov decision processes (Puterman, 1994; Bertsekas and Tsitsiklis, 1996). We revisit the λ policy iteration algorithm introduced by Bertsekas and Ioffe (1996), also published in the reference textbook of Bertsekas and Tsitsiklis (1996), 1 that (as stated by the authors) 'is primarily motivated by the case of large and complex problems where the use of approximation is essential' . It is a family of algorithms parametrized by a parameter λ that generalizes the standard dynamic-programming algorithms value iteration (which corresponds to the case λ = 0) and policy iteration (case λ = 1), and has some deep connections with the temporal-difference algorithms that are well known to the reinforcement-learning community (Sutton and Barto, 1998; Bertsekas and Tsitsiklis, 1996).

1. The work of Bertsekas and Ioffe (1996) being historically anterior to the textbook of Bertsekas and Tsitsiklis (1996), we only refer to the former in the rest of the paper.

In their original paper, Bertsekas and Ioffe (1996) show the convergence of λ policy iteration for its exact version and provide its asymptotic convergence rate. The authors also describe a case study involving an instance of approximate λ policy iteration, but neither their paper nor (to the best of our knowledge) any subsequent work show that this makes sense: two important issues are whether approximations can be controlled throughout the iterations and checking that the approach does not break when considering an undiscounted problem like Tetris. In this paper, we extend the theory on this algorithm in several ways. We derive its non-asymptotic convergence rate for its exact version. More importantly, we develop the theory of λ policy iteration for its main purpose, that is-recall the above quote-when it is run in an approximate form. We show that the performance loss due to using the greedy policy with respect to the current value estimate instead of the optimal policy can be made arbitrarily small by controlling the error along the iterations. Last but not least, we show that our analysis can be extended to the undiscounted case.

The rest of the paper is organized as follows. In Section 2, we introduce the framework of Markov decision processes, describe the two standard algorithms, value and policy iteration. Section 3 describes λ policy iteration in an original way that makes its connection with these standard algorithms obvious. We discuss there the close connection with TD( λ ) (Sutton and Barto, 1998) and recall the main results obtained by Bertsekas and Ioffe (1996): convergence and asymptotic rate of convergence of the exact algorithm. Our main results are stated in Section 4. We first argue that the analysis of λ policy iteration is more involved than that of value and policy iteration since neither contraction nor monotonicity arguments, that analysis of these two algorithms rely on, hold for λ policy iteration. We provide a non-asymptotic analysis of λ policy iteration and several asymptotic performance bounds for its approximate version. We close this section by presenting performance bounds of approximate λ policy iteration that also apply to the undiscounted case. We discuss in Section 5 the relations between our results and those previously obtained for approximate value and policy iteration by Munos (2003, 2007). Last but not least, Section 6 revisits the empirical part of the work of Bertsekas and Ioffe (1996), where an approximate version of λ policy iteration is used for training a Tetris controller.

## 2. Framework And Standard Algorithms

We begin by describing the framework of Markov decision processes we consider throughout the paper. We go on by describing the two main algorithms of the literature, value and policy iteration, for solving the related problem.

We consider a discrete-time dynamic system whose state transition depends on a control. We assume that there is a state space X of finite 2 size N . When at state i ∈{ 1 , .., N } , an action is chosen from a finite action space A . The action a ∈ A specifies the transition probability pij ( a ) to the next state j . At each transition, the system is given a reward r ( i , a , j ) where r is the instantaneous reward function . In this context, we look for a stationary deterministic policy (a function π : X → A that maps states into actions 3 ) that maximizes the expected discounted sum of rewards from any state i ,

2. We restrict our attention to finite state space problems for simplicity. The extension of our study to infinite/continuous state spaces is straightforward.

3. Restricting our attention to stationary deterministic policies is not a limitation. Indeed, for the optimality criterion to be defined soon, it can be shown that there exists at least one stationary deterministic policy that is optimal (Puterman, 1994).

called the value of policy π at state i :

<!-- formula-not-decoded -->

∣ where E π denotes the expectation conditional on the fact that the actions are selected with the policy π , and 0 &lt; γ &lt; 1 is a discount factor. 4 The tuple 〈 X , A , p , r , γ 〉 is called a Markov decision process (MDP) (Puterman, 1994; Bertsekas and Tsitsiklis, 1996).

The optimal value starting from state i is defined as

<!-- formula-not-decoded -->

We write P π for the N × N stochastic matrix whose elements are pij ( π ( i )) and r π the vector whose components are ∑ j pij ( π ( i )) r ( i , π ( i ) , j ) . The value functions v π and v ∗ can be seen as vectors on X . It is well known that v π is a solution of the following Bellman equation:

<!-- formula-not-decoded -->

The value function v π is thus a fixed point of the linear operator T π v : = r π + γ P π v . As P π is a stochastic matrix, its eigenvalues cannot be greater than 1, and consequently I -γ P π is invertible. This implies that

<!-- formula-not-decoded -->

It is also well known that the optimal value v ∗ satisfies the following Bellman equation:

<!-- formula-not-decoded -->

where the max operator is component-wise. In other words, v ∗ is a fixed point of the nonlinear operator Tv : = max π T π v . For any value vector v , we call a greedy policy with respect to the value v a policy π that satisfies:

<!-- formula-not-decoded -->

or equivalently T π v = Tv . We write, with some abuse of notation 5 greedy( v ) any policy that is greedy with respect to v . The notions of optimal value function and greedy policies are fundamental to optimal control because of the following property: any policy π ∗ that is greedy with respect to the optimal value is an optimal policy and its value v π ∗ is equal to v ∗ .

The operators T π and T are γ -contraction mappings with respect to the max norm ‖ . ‖ ∞ (Puterman, 1994) defined as follows for all vector u :

<!-- formula-not-decoded -->

In what follows, we only describe what this means for T but the same holds for T π . Being a γ -contraction mapping for the max norm means that for all pairs of vectors ( v , w ) ,

<!-- formula-not-decoded -->

4. We will consider the undiscounted situation ( γ = 1) in Section 4.4, and introduce appropriate related assumptions there.

5. There might be several policies that are greedy with respect to some value v .

## SCHERRER

| Algorithm 1 Value iteration                                                                                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input: An MDP, an initial value v 0 Output: An (approximately) optimal k ← 0 repeat v k + 1 ← Tv k // Update the value k ← k + 1 until some stopping criterion Return greedy( v k ) |
| policy                                                                                                                                                                              |

This ensures that the fixed point v ∗ of T exists and is unique. Furthermore, for any initial vector v 0,

<!-- formula-not-decoded -->

Given an MDP, standard algorithmic solutions for computing an optimal value-policy pair are value and policy iteration (Puterman, 1994). The rest of this section describes both algorithms with some of the relevant properties for the subject of this paper.

The value iteration algorithm for computing the value of a policy π and the value of the optimal policy π ∗ rely on Equation 2. Algorithm 1 provides a description of value iteration for computing an optimal policy (replace T by T π in it and one gets value iteration for computing the value of some policy π ). The contraction property induces some interesting properties for value iteration. Not only does it ensure convergence, but it also implies a linear rate of convergence of the value vk to v ∗ : for all k ≥ 0,

<!-- formula-not-decoded -->

It is possible to derive a performance bound, that is a bound on the difference between the real value of a policy produced by the algorithm and the value of the optimal policy π ∗ by using the following well-known property (Puterman, 1994): For all v , if π = greedy ( v ) then

<!-- formula-not-decoded -->

Let π k denote the policy that is greedy with respect to vk -1. Then,

<!-- formula-not-decoded -->

Policy iteration is an alternative method for computing an optimal policy for an infinite-horizon discounted Markov decision process. This algorithm is based on the following property: if π is some policy, then any policy π ′ that is greedy with respect to the value of π , that is any π ′ satisfying π ′ = greedy ( v π ) , is better than π in the sense that v π ′ ≥ v π . Policy iteration exploits this property in order to generate a sequence of policies with increasing values. It is described in Algorithm 2. Note that we use the analytical form of the value of a policy given by Equation 1. When the state space and the action space are finite, policy iteration converges to an optimal policy π ∗ in a finite number of iterations (Puterman, 1994; Bertsekas and Tsitsiklis, 1996). In infinite state spaces, if the function v ↦→ P greedy ( v ) is Lipschitz, then it can be shown that policy iteration has a quadratic convergence rate (Puterman, 1994).

Algorithm 2 Policy iteration

Input:

An MDP, an initial policy π 0

Output:

An (approximately) optimal policy

k ← 0

repeat

$$vk ← ( I - γ P k ) - 1 r k π k + 1 ← greedy ( vk )$$

$$π π // Estimate the value of π k // Update the policy 1$$

k ← k +

until some stopping criterion

Return π k

## 3. The λ Policy Iteration Algorithm

In this section, we describe the family of algorithms that is the main topic of this paper, ' λ policy iteration,' 6 originally introduced by Bertsekas and Ioffe (1996). λ policy iteration is parametrized by a coefficient λ ∈ ( 0 , 1 ) and generalizes value and policy iteration. When λ = 0, λ policy iteration reduces to value iteration while it reduces to policy iteration when λ = 1. We also recall the fact discussed by Bertsekas and Ioffe (1996) that λ policy iteration draws some connections with temporal-difference algorithms (Sutton and Barto, 1998).

Webegin by giving some intuition about how one can make a connection between value and policy iteration. At first sight, value iteration builds a sequence of value functions and policy iteration a sequence of policies. In fact, both algorithms can be seen as updating a sequence of value-policy pairs. With some little rewriting-by decomposing the (nonlinear) Bellman operator T into (i) the maximization step and (ii) the application of the (linear) Bellman operator-it can be seen that each iterate of value iteration is equivalent to the two following updates:

<!-- formula-not-decoded -->

The left hand side of the above equation uses the operator T π k + 1 while the right hand side uses its definition. Similarly-by inverting in Algorithm 2 the order of (i) the estimation of the value of the current policy and (ii) the update of the policy, and by using the fact that the value of the policy π k + 1 is the fixed point of T π k + 1 (Equation 2)-it can be argued that every iteration of policy iteration does the following:

<!-- formula-not-decoded -->

This rewriting makes both algorithms look close to each other. Both can be seen as having an estimate vk of the value of policy π k , from which they deduce a potentially better policy π k + 1. The corresponding value v π k + 1 of this better policy may be regarded as a target which is tracked by the next estimate vk + 1. The difference is in the update that enables to go from vk to vk + 1: while policy iteration directly jumps to the value of π k + 1 (by applying the Bellman operator T π k + 1 an infinite number of times), value iteration only makes one step towards it (by applying T π k + 1 only once).

6. It was also called 'temporal-difference based policy iteration' in the original paper, but we take the name λ policy iteration, as it was the name picked by most subsequent works.

From this common view of value iteration, it is natural to introduce the well-known modified policy iteration algorithm (Puterman and Shin, 1978) which makes n steps at each update:

<!-- formula-not-decoded -->

Figure 1: Visualizing λ policy iteration in the greedy partition. Following Bertsekas and Tsitsiklis (1996, p. 226), one can decompose the value space as a collection of polyhedra, such that each polyhedron corresponds to a region where one policy is greedy. This is called the greedy partition . In the above example, there are only 3 policies, π 1, π 2 and π ∗ . vk is the initial value. greedy ( vk ) = π 2, greedy ( v π 2 ) = π 1, and greedy ( v π 1 ) = π ∗ . Therefore policy iteration (or '1 policy iteration') generates the sequence (( π 2 , v π 2 ) , ( π 1 , v π 1 ) , ( π ∗ , v π ∗ )) . Value iteration (or '0 policy iteration') starts by slowly updating vk towards v π 2 until it crosses the boundary π 1 / π 2, after which it tracks alternatively v π 1 and v π 2 , until it reaches the π ∗ part. In other words, value iteration makes small steps. λ policy iteration is doing something intermediate: it makes steps of which the length is controlled by λ .

<!-- image -->

of λ policy iteration. λ policy iteration is doing a λ -adjustable step towards the value of π k + 1:

<!-- formula-not-decoded -->

The equivalence between the left and the right representation of λ policy iteration needs here to be proved. For all k ≥ 0 and all function v , Bertsekas and Ioffe (1996) introduce the following operator 7

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

7. The equivalence between Equations 4 and 5 follows trivially from the definition of T π k + 1 .

## Algorithm 3 λ policy iteration

Input:

An MDP, λ ∈ ( 0 , 1 ) , an initial value v 0

Output:

An (approximately) optimal policy

k ← 0

## repeat

<!-- formula-not-decoded -->

until some convergence criterion

Return greedy( vk )

## and prove that

- Mk is a contraction mapping of modulus λγ for the max norm ;
- The next iterate vk + 1 of λ policy iteration is the (unique) fixed point of Mk .

The left representation of λ policy iteration is obtained by 'unrolling' Equation 4 an infinite number of times, while the right one is obtained by using Equation 5 and solving the linear system vk + 1 = Mkvk + 1.

As illustrated in figure 1, the parameter λ (or n in the case of modified policy iteration) can informally be seen as adjusting the size of the step for tracking the target v π k + 1 : the bigger the value, the longer the step. Formally, λ policy iteration (consider the above left hand side) consists in doing a geometric average of parameter λ of the terms ( T π k + 1 ) j vk for all values of j . The right hand side is here interesting because it clearly shows that λ policy iteration generalizes value iteration (when λ = 0) and policy iteration (when λ = 1). The operator Mk gives some insight on how one may concretely implement one iteration of λ policy iteration: it can for instance be done through a valueiteration like algorithm which applies Mk iteratively. Then, the fact that its contraction factor is λγ is interesting: when λ &lt; 1, finding the corresponding fixed point can generally be done in fewer iterations than that of T π k + 1 , which is only γ -contracting.

In order to fully describe the λ policy iteration algorithm, we introduce an operator that corresponds to the computation of the fixed point of Mk . For any value v and any policy π , define:

<!-- formula-not-decoded -->

where the different equalities are due to basic algebra and the fact that T π v = r π + γ P π v .

λ policy iteration is formally described in Algorithm 3. Our description includes a potential error term ε k when updating the value, which stands for several possible sources of error at each iteration: this error might be the computer round off, the fact that we use an approximate architecture for representing v , a stochastic approximation of P π k , etc... or a combination of these. It is

## SCHERRER

Figure 2: λ policy iteration, a fundamental algorithm for reinforcement learning. We represent a picture of the family of algorithms corresponding to λ policy iteration. The vertical axis corresponds to whether one does full backup (exact computation of the expectations) or stochastic approximation (estimation through samples). The horizontal axis corresponds to the depth of the backups, and is controlled by the parameter λ . This drawing is reminiscent of the picture that appears in chapter 10.1 of the textbook by Sutton and Barto (1998) that represents 'two of the most important dimensions' of reinforcement-learning methods along the same dimensions. In that drawing, from top to bottom and left to right, the authors labeled the corners 'Dynamic Programming', 'Exhaustive search', 'TemporalDifference learning' and 'Monte-Carlo'. It is interesting to notice that Sutton and Barto (1998) comment their drawing as follows: 'At three of the four corners of the space are the three primary methods for estimating values: DP, TD, and Monte Carlo' . They do not recognize the fourth corner as one of the reinforcement-learning primary methods . Our representation of λ policy iteration actually suggests that in place of 'Exhaustive search', policy iteration, which consists in computing the value of the current policy, is the deepest backup method , and can be considered as the batch version of Monte Carlo.

<!-- image -->

straightforward to see that the λ policy iteration reduces to value iteration (Algorithm 1) when λ = 0 and to policy iteration 8 (Algorithm 2) when λ = 1.

The definition of the operator T π λ given by Equation 7 is the form we have used for the introduction of λ policy iteration as an intermediate algorithm between value and policy iteration. The equivalent form given by Equation 6 can be used to make a connection with the TD( λ ) algorithm 9

8. Policy iteration starts with an initial policy while λ policy iteration starts with some initial value. To be precise, '1 policy iteration' starting with v 0 is equivalent to policy iteration starting with the greedy policy with respect to v 0 .

9. TD stands for temporal difference. As we have mentioned in Footnote 6, λ policy iteration was originally also called 'temporal-difference based policy iteration' and the presentation of Bertsekas and Ioffe (1996) starts from the formulation of Equation 6 (which is close to TD( λ )), and afterwards makes the connection with value and policy iteration.

(Sutton and Barto, 1998). Indeed, through Equation 6, the evaluation phase of λ policy iteration can be seen as an incremental additive procedure:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is zero if and only if the value vk is equal to the optimal value v ∗ . It can be shown (Bertsekas and Ioffe, 1996) that the vector ∆ k has components given by:

with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

being the temporal difference associated to transition i → j , as defined by Sutton and Barto (1998). When one uses a stochastic approximation of λ policy iteration, that is when the expectation E π t + 1 is approximated by sampling, λ policy iteration reduces to the algorithm TD( λ ) which is described in chapter 7 of Sutton and Barto (1998). In particular, when λ = 1, the terms in the above sum collapse and become the exact discounted return:

<!-- formula-not-decoded -->

and the stochastic approximation matches the Monte-Carlo method. Also, Bertsekas and Ioffe (1996) show that approximate TD( λ ) with a linear feature architecture, as described in chapter 8.2 of Sutton and Barto (1998), corresponds to a natural approximate version of λ policy iteration where the value is updated by least squares fitting using a gradient-type iteration after each sample. Last but not least, as illustrated in figure 2, the reader might notice that the 'unified view' of reinforcement-learning algorithms which is depicted in chapter 10.1 of Sutton and Barto (1998) is in fact a picture of λ policy iteration.

To our knowledge, little has been done concerning the analysis of λ policy iteration: the only results available concern the exact case (when ε k = 0). Define the following factor

<!-- formula-not-decoded -->

We have 0 ≤ β ≤ γ &lt; 1. If λ = 0 (value iteration) then β = γ , and if λ = 1 (policy iteration) then β = 0. In the original article introducing λ policy iteration, Bertsekas and Ioffe (1996) show the convergence and provide the following asymptotic rate of convergence.

## Proposition 1 (Convergence of λ PI, Bertsekas and Ioffe, 1996)

The sequence vk converges to v ∗ . Furthermore, after some index k ∗ , the rate of convergence is linear in β as defined in Equation 9, that is

<!-- formula-not-decoded -->

where

Figure 3: This simple deterministic MDP is used to show that λ policy iteration cannot be analyzed in terms of contraction (see text for details).

<!-- image -->

By making λ close to 1, β can be arbitrarily close to 0 so the above rate of convergence might look overly impressive. This needs to be put into perspective: the index k ∗ is the index after which the policy π k does not change anymore (and is equal to the optimal policy π ∗ ). As we said when we introduced the algorithm, λ controls the speed at which one wants vk to 'track the target' v π k + 1 ; when λ = 1, this is done in one step (and if π k + 1 = π ∗ then vk + 1 = v ∗ ).

## 4. Analysis Of λ Policy Iteration

λ policy iteration is conceptually nice since it generalizes the two most well-known algorithms for solving Markov decision processes. In the literature, lines of analysis are different for value and policy iteration. Analyzes of value iteration are based on the fact that it computes the fixed point of the Bellman operator which is a γ -contraction mapping in max norm (Bertsekas and Tsitsiklis, 1996). Unfortunately, it can be shown that the operator by which policy iteration updates the value from one iteration to the next is in general not a contraction in max norm. In fact, this observation can be drawn for λ policy iteration as soon as it does not reduce to value iteration:

Proposition 2 If λ &gt; 0 , there exists no norm for which the operator v ↦→ T greedy ( v ) λ v by which λ policy iteration updates the value from one iteration to the next is a contraction.

Proof To see this, consider the deterministic MDP (shown in figure 3) with two states { 1 , 2 } and two actions { change , stay } . The instantaneous rewards of being in state 1 and 2 are respectively r 1 = 0 and r 2 = 1 (they do not depend on the action nor the resulting state), and the transitions are characterized as follows: Pchange ( 2 | 1 ) = Pchange ( 1 | 2 ) = Pstay ( 1 | 1 ) = Pstay ( 2 | 2 ) = 1. Consider the following two value functions v = ( ε , 0 ) and v ′ = ( 0 , ε ) with ε &gt; 0. Their corresponding greedy policies are π =( stay , change ) and π ′ =( change , stay ) . Then, we can compute the next iterates of v

and v ′ (using Equation 7):

Then while

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As ε can be arbitrarily small, the norm of T π λ v -T π ′ λ v ′ can be arbitrarily larger than that of v -v ′ when λ &gt; 0.

Analyzes of policy iteration usually rely on the fact that the sequence of values generated is nondecreasing (Bertsekas and Tsitsiklis, 1996; Munos, 2003). Unfortunately, it can easily be seen that as soon as λ is smaller than 1, the value functions may decrease (it suffices to take a very high initial value). For non trivial values of λ , λ policy iteration is neither contracting nor non-decreasing, so we need a new proof technique.

## 4.1 Main Proof Ideas

The rest of this section provides an overview of our analysis. We show how to compute an upper bound of the loss for λ policy iteration in the general (possibly approximate) case. It is the basis for the derivation of component-wise bounds for exact λ policy iteration (Section 4.2) and approximate λ policy iteration (Section 4.3). Consider λ policy iteration as described in Algorithm 3, and the sequence of value-policy-error triplets ( vk , π k , ε k ) it generates.

Our goal is to provide a bound of the loss of using policy π k instead of the optimal policy:

<!-- formula-not-decoded -->

Our analysis amounts to decompose the loss as follows:

<!-- formula-not-decoded -->

where wk is the value of the k th before the approximation ε k is incurred:

<!-- formula-not-decoded -->

We shall call the term dk = v ∗-wk the distance as it is a measure of distance between the optimal value and the k th value wk . Similarly, we shall call the term sk = wk -v π k the shift as it shows the shift between the k th value wk and the value of the k th policy (as mentioned before, the former can indeed be understood as tracking the latter). As it will appear shortly, we will be able to bound both quantities, and thus deduce a bound on the loss. Actual bounds on dk and sk will be based on a bound on the Bellman residual of the k th value:

<!-- formula-not-decoded -->

To lighten the notations, from now on we write: Pk : = P π k , Tk : = T π k , P ∗ : = P π ∗ . We refer to the factor β as introduced by Bertsekas and Ioffe (Equation 9 page 1189). Also, the following stochastic matrix plays a recurrent role in our analysis: 10

<!-- formula-not-decoded -->

For a vector u , we use the notation u for an upper bound of u and u for a lower bound.

Our analysis relies on a series of lemmas that we now state (for clarity, all the proofs are deferred to appendix B).

Lemma 3 The shift is related to the Bellman residual as follows:

<!-- formula-not-decoded -->

Lemma 4 The Bellman residual at iteration k + 1 cannot be much lower than that at iteration k:

<!-- formula-not-decoded -->

where xk : =( γ Pk -I ) ε k only depends on the approximation error.

As a consequence, a lower bound of the Bellman residual is: 11

<!-- formula-not-decoded -->

Using Lemma 3, the bound on the Bellman residual also provides an upper bound on the shift: 12

<!-- formula-not-decoded -->

Lemma 5 The distance at iteration k + 1 cannot be much greater than that at iteration k:

<!-- formula-not-decoded -->

where yk : = λγ 1 -λγ Ak + 1 ( -bk ) -γ P ∗ ε k depends on the lower bound of the Bellman residual and the approximation error.

10. The fact that this is indeed a stochastic matrix is explained at the beginning of the appendices.

11. We use the property here that if some vectors satisfy the component-wise inequality x ≤ y , and if P is a stochastic matrix, then the component-wise inequality Px ≤ Py holds.

12. We use the fact that ( 1 -γ )( I -γ Pk ) -1 is a stochastic matrix (see Footnote 10) and Footnote 11.

Then, an upper bound of the distance is: 13

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the upper bounds on the distance and the shift enable us to derive the upper bound on the loss.

The above derivation is a generalization of that of Munos (2003) for approximate policy iteration. Note however that it is not a trivial generalization: when λ = 1, that is when both proofs coincide, β = 0 and Lemmas 3 and 4 have the following particularly simple form: sk = 0 and bk + 1 ≥ xk + 1.

The next two subsections contain our main results, which take the form of performance bounds when using λ policy iteration. Section 4.2 gathers the results concerning exact λ policy iteration, while Section 4.3 presents those concerning approximate λ policy iteration.

## 4.2 Performance Bounds For Exact λ Policy Iteration

Consider exact λ policy iteration for which we have ε k = 0 for all k . By exploiting the recursive relations we have described in the previous section (this process is detailed in appendix C), we can derive the following component-wise bounds for the loss.

## Lemma 6 (Component-Wise rate of convergence of exact λ PI)

For all k &gt; 0 , the following matrices

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

are stochastic and the performance of the policies generated by λ policy iteration satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where e is the vector of which all components are 1 .

In order to derive (more interpretable) max norm bounds from the above component-wise bound, we rely on the following lemma, which for clarity of exposition is proved in appendix G.

13. See Footnote 11.

Eventually, as

Lemma 7 If for some non-negative vectors x and y, some constant K ≥ 0 , and some stochastic matrices X and X ′ we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With this, the component-wise bounds of Lemma 6 become:

## Proposition 8 (Non-asymptotic bounds for exact λ policy iteration)

For any k &gt; 0 , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These non-asymptotic bounds supplement the asymptotic bound of Proposition 1 from Bertsekas and Ioffe (1996). Remarkably, these max-norm bounds show no dependence on the value λ . The bound of Equation 13 is expressed in terms of the initial distance between the value function and the optimal value function, and constitutes a generalization of the rate of convergence of value iteration by Puterman (1994) that we described in Equation 3 page 1184. The second inequality, Equation 14, is expressed in terms of the initial Bellman residual and is also well-known for value iteration (Puterman, 1994). The last inequality described in Equation 15 relies on the distance between the value function and the optimal value function and the value difference between the optimal policy and the first greedy policy; compared to the others, it has the advantage of not containing a 1 1 -γ factor. To our knowledge, this bound is even new for the specific cases of value and policy iteration.

## 4.3 Performance Bounds For Approximate λ Policy Iteration

We now turn to the (slightly more involved) results on approximate λ policy iteration. We provide component-wise bounds of the loss l k = v ∗-v π k ≥ 0 of using policy π k instead of using the optimal policy, with respect to the approximation error ε k , the policy Bellman residual Tkvk -vk and the Bellman residual Tvk -vk = Tk + 1 vk -vk . Note the subtle difference between the two Bellman residuals: the policy Bellman residual says how much vk differs from the value of π k while the Bellman residual says how much vk differs from the value of the policies π k + 1 and π ∗ .

The core of our analysis, and the main contribution of this article, is described in the following lemma.

## Lemma 9 (Component-Wise performance bounds for app. λ policy iteration)

For all k ≥ j ≥ 0 , the following matrices

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

are stochastic and for all k,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first relation (Equation 16) involves the errors ( ε k ), is based on Lemmas 3-5 (presented in Section 4.1) and is proved in appendix D. The two other inequalities (the asymptotic performance of approximate λ policy iteration with respect to the Bellman residuals in Equations 17 and 18) are somewhat simpler and are proved independently in appendix E.

By taking the max norm in the above component-wise performance bounds, we obtain, for all k ,

<!-- formula-not-decoded -->

In the specific context of value and policy iteration, Munos (2003, 2007) has argued that most supervised learning algorithms (such as least squares regression) that are used in practice for approximating each iterate control the errors ( ε k ) for some weighted Lp norm ‖·‖ p , µ , defined for some distribution µ on the state space X as follows:

<!-- formula-not-decoded -->

As a consequence, Munos (2007, 2003) explained how to derive an analogue of the above result where the approximation error ε k is expressed in terms of this Lp norm. Based on Munos' works, we provide below a useful technical lemma (proved in appendix G) that shows how the performance of approximate λ policy iteration can be translated into Lp norm bounds.

Lemma 10 Let xk, yk be vectors and Xkj, X ′ k j stochastic matrices satisfying for all k

<!-- formula-not-decoded -->

where ( ξ i ) i ≥ 1 is a sequence of non-negative weights satisfying

<!-- formula-not-decoded -->

Then, for all distribution µ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

are distributions and

Thus, using this lemma and the fact that ∑ ∞ i = 1 γ i = γ 1 -γ , Lemma 9 can be turned into the following proposition.

## Proposition 11 ( Lp norm performance of approximate λ PI)

With the notations of Lemma 9, for all p, k ≥ j ≥ 0 and all distribution µ,

<!-- formula-not-decoded -->

are distributions and the performance of the policies generated by λ policy iteration satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition 11 means that in order to control the performance loss (the left hand side) for some µ -weighted Lp norm, one needs to control the errors ε j , the policy Bellman residual Tj j k -vj or the Bellman residual Tvk -1 -vk -1 (the right hand sides) respectively for the norms µkj , µ ′ k j and µ ′′ k .

Unfortunately, these distributions depend on unknown quantities (such as the stochastic matrix of the optimal policy, see the definitions in Lemma 9) and cannot be used in practice by the algorithm. To go round this issue, we follow Munos (2003, 2007) and introduce some assumption on the stochasticity of the MDP in terms of a so-called concentrability coefficient . Assume there exists a distribution ν and a real number C ( ν ) such that

<!-- formula-not-decoded -->

For instance, if one chooses the uniform law ν , then there always exists such a C ( ν ) ∈ ( 1 , N ) where N is the size of the state space. More generally, a small value of C ( ν ) requires that the underlying MDPhas a significant amount of stochasticity; see (Munos, 2003, 2007) for more discussion on this coefficient. Given this definition, we have the following property.

Lemma 12 Let X be a convex combination of products of stochastic matrices of the MDP. For any distribution µ, vector y, and p,

<!-- formula-not-decoded -->

Proof It can be seen from the definition of the concentrability coefficient C ( ν ) that µ T X ≤ C ( ν ) ν T . Thus,

<!-- formula-not-decoded -->

Using this lemma, and the fact that for any p , ‖ x ‖ ∞ = max µ ‖ x ‖ p , µ , the Lp bounds of Proposition 11 lead to the following proposition.

## Proposition 13 ( L ∞ / Lp norm performance of approximate λ PI)

Let C ( ν ) be the concentrability coefficient defined in Equation 20. For all p,

<!-- formula-not-decoded -->

It is, once again, remarkable that these bounds do not explicitly depend on the value of λ . However, it should be clear that, with respect to the previous bounds, the influence of λ is now hidden in the concentrability coefficient C ( ν ) . Furthermore, as it is the case in TD( λ ) methods, and as will be

illustrated in the case study in Section 6, the value of λ will directly influence the errors ∥ ∥ ε j ∥ ∥ p , ν and the Bellman residual terms ‖ Tkvk -vk ‖ p , ν and ‖ Tvk -1 -vk -1 ‖ p , ν .

In general, one cannot give the guarantee that approximate λ policy iteration will converge. However, the performance bounds with respect to the approximation error can be improved if we observe empirically that the value or the policy converges. Note that the former condition implies the latter (while the opposite is not true: the policy may converge while the value still oscillates). Indeed, we have the following corollary (proved in appendix F).

## Corollary 14 ( L ∞ / Lp norm performance of app. λ PI in case of convergence)

If the value converges to some v, then the approximation error converges to some ε , and the corresponding greedy policy π satisfies

<!-- formula-not-decoded -->

If the policy converges to some π , then

<!-- formula-not-decoded -->

It is interesting to notice that in the latter weaker situation where only the policy converges, the constant decreases from 1 ( 1 -γ ) 2 to 1 1 -γ when λ varies from 0 to 1; in other words, the closer to policy iteration, the better the bound in that situation.

## 4.4 Extension To The Undiscounted Case

The results we have described so far only apply to the situation where the discount factor γ is smaller than 1. Indeed, all our bounds involve terms of the form 1 1 -γ that diverge to infinity as γ tends to 1. In this last subsection, we show how the component-wise analysis of Lemma 9 can be exploited to also cover the case where we have an undiscounted MDP ( γ = 1), as for instance in the the case study on the Tetris domain presented in Section 6.

In undiscounted infinite horizon control problems, it is generally assumed that there exists a N + 1 th termination absorbing state 0. Once the system reaches this state, it remains there forever with no further reward, that is formally:

<!-- formula-not-decoded -->

In order to derive our results, we will introduce conditions that ensure that termination is guaranteed in finite time with probability 1 under any sequence of actions. Formally, we will assume that there exists an integer n 0 ≤ N and a real number α &lt; 1 such that for all initial distributions µ , all actions a 0 , a 1 , ..., an 0 -1, the following relation

/negationslash

<!-- formula-not-decoded -->

holds. 14 We can think of the MDP as only defined on the N non-terminal states, that is on { 1 , ... N } . Then, for any policy π , the matrix P π is sub-stochastic , and the above assumption implies that for

14. In the literature, a stationary policy that reaches the terminal state in finite time with probability 1 is said to be proper . The usual assumptions in undiscounted infinite horizon control problems are: (i) there exists at least one proper policy and (ii) for every improper policy π , the corresponding value equals -∞ for at least one state. The situation we consider here is simpler, since we assume that all (non-necessarily stationary nor deterministic) policies are proper.

all set of n 0 policies π 1 , π 2 , · · · , π n 0 ,

∥ ∥ The component-wise analysis of λ policy iteration is here identical to what we have done before, except that we have 15 γ = 1 and β = 1. The matrix Ak that appeared recurrently in our analysis has the following special form:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and is a sub-stochastic matrix. The first bound of the component-wise analysis of λ policy iteration (Lemma 9 page 1194) can be generalized as follows (see appendix H for details).

## Lemma 15 (Component-Wise bounds in the undiscounted case)

Assume that there exist n 0 and α such that Equation 21 holds. Write η : = 1 -λ n 0 1 -λ n 0 α . For all i, write

<!-- formula-not-decoded -->

For all j &lt; k, the following matrices

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

are sub-stochastic and the performance of the policies generated by λ policy iteration satisfies

<!-- formula-not-decoded -->

By observing that η ∈ ( 0 , 1 ) , and that for all x ∈ ( 0 , 1 ) , 0 ≤ 1 -x n 0 1 -x ≤ n 0, it can be seen that the coefficients δ i are finite for all i . Furthermore, when n 0 = 1 (which matches the discounted case with α = γ ), one can observe that δ i = γ i 1 -γ and that one recovers the result of Lemma 9.

This lemma can then be exploited to show that λ policy iteration enjoys an Lp norm guarantee. Indeed, an analogue of Proposition 11 (whose proof is detailed in appendix H) is the following proposition.

## Proposition 16 ( Lp norm bound in the undiscounted case)

Let C ( ν ) be the concentrability coefficient defined in Equation 20 page 1197. Let the notations and conditions of Lemma 15 hold. For all distribution µ on ( 1 , · · · , N ) and k ≥ j ≥ 0 ,

<!-- formula-not-decoded -->

15. For simplicity in our discussion, we consider λ &lt; 1 to avoid the special case λ = 1 for which β may be indefinite (see the definition of β in Equation 9 page 1189). The interested reader may however check that the results that we state are continuous in the neighborhood of λ = 1.

are non-negative vectors and

∥ ∥ are distributions on ( 1 , · · · , N ) . Then for all p, the loss of the policies generated by λ policy iteration satisfies where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

There are two main differences with respect to the results we have presented for the discounted case:

1. The fact that we considered the model (and thus the algorithm) only on the non-terminal states ( 1 , · · · , N ) means that we made the assumption that there is no error incurred in the terminal state 0. Note, however, that this is not a strong assumption since the value of the terminal state is necessarily 0.
2. The constant K ( λ , n 0 ) is dependent on λ . More precisely, it can be observed that:

<!-- formula-not-decoded -->

and that this is the minimal value of λ ↦→ K ( λ , n 0 ) . Although we took particular care in deriving this bound, we leave for future work the question whether one could prove a similar result with the constant n 0 2 ( 1 -α ) 2 -n 0 1 -α for all λ ∈ ( 0 , 1 ) . When n 0 = 1 (which matches the discounted case with α = γ ), K ( λ , 1 ) does not depend anymore on λ and we recover, without surprise, the bound of Proposition 11 since

<!-- formula-not-decoded -->

## 5. Related Work

The study of approximate versions of value and policy iteration has been the topic of a rich literature (Bertsekas and Tsitsiklis, 1996), in particular in the discounted case on which we focus in what follows. The most well-known results, due to Bertsekas and Tsitsiklis (1996, pp. 332-333 for value iteration and Prop. 6.2 p. 276 for policy iteration), states that the performance loss due to using the policies π k instead of the optimal policy π ∗ satisfies:

<!-- formula-not-decoded -->

As mentioned earlier (after Equation 19 page 1195), Munos (2003, 2007) has argued that the above bound does not directly apply to practical implementations that usually control some Lp norm of the

errors. Munos extended the error analysis of Bertsekas and Tsitsiklis (1996) to this situation. His analysis begins by the following error propagation for value iteration-taken from (Munos, 2007, Lemma 4.1)-and for policy iteration-adapted from 16 (Munos, 2003, Lemma 4).

## Lemma 17 (Asymptotic component-wise performance of AVI and API)

For all k &gt; j ≥ 0 , the following matrices

<!-- formula-not-decoded -->

are stochastic. The asymptotic performance of the policies generated by approximate value iteration satisfies

<!-- formula-not-decoded -->

The asymptotic performance of the policies generated by approximate policy iteration satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, introducing the concentrability coefficient C ( ν ) (Equation 20 page 1197) and using the techniques that we described through Lemmas 10 and 12, Munos (2003, 2007) turned these componentwise bounds into L ∞ / Lp norm bounds that match those of our (more general) Proposition 13. In particular he obtains the following bound for both value and policy iteration,

<!-- formula-not-decoded -->

that generalizes that of Bertsekas and Tsitsiklis (1996) (Equation 23) since [ C ( ν )] 1 / p tends to 1 when p tends to infinity. Munos also provides some improved bounds when value iteration converges to some value (Munos, 2007, sections 5.2 and 5.3), or when policy iteration converges to some policy (Munos, 2003, Remark 4); similarly, these are special cases of our Corollary 14 page 1198.

At a somewhat more technical level, our key result on approximations, stated in Lemma 9 page 1194, gives a component-wise analysis for the whole family of algorithms λ policy iteration. It is thus natural to look at the relations between our bounds for general λ and the bounds derived separately by Munos for value iteration (Equation 24) and policy iteration (Equations 25 and 26).

16. We provide here a correction of the result stated by Munos (2003, Theorem 1) that is obtained by an inappropriate exchange of an expectation and a sup operator (Munos, 2003, Proofs of Corollaries 1 and 2). Note, however that the concentrability coefficient based results (Munos, 2003, Theorems 2 and 3) are not affected.

In the case where λ = 0 (and thus when λ policy iteration reduces to value iteration), consider the bound we gave in Equation 16. Since λ = 0, we have β = γ , Ak = Pk and

<!-- formula-not-decoded -->

Our bound thus implies that

<!-- formula-not-decoded -->

The bound derived by Munos for approximate value iteration (Equation 24) is

<!-- formula-not-decoded -->

The above bounds are very close to each other: we can go from Equation 27 to Equation 28 by replacing Pk -1 ... Pj by ( P ∗ ) k -j . Now, when λ = 1 (when λ policy iteration reduces to policy iteration), we have β = 0, Ak =( 1 -γ )( I -γ Pk ) -1 Pk and it is straightforward to see that Bkj = Rkj and B ′ k j = R ′ k j , and the bound given in Equation 16 matches that of Munos in Equation 25. Finally, it can easily be observed that the stochastic matrices involved in Equation 26 (with the policy Bellman residual) match those of the one we gave in Equation 17: formally, we have R ′′ k j = Ckj and Rkj = C ′ k j . Thus, up to some little details, our component-wise analysis unifies those of Munos. It is not a surprise that we fall back on the result of Munos for approximate policy iteration because, as already mentioned at the end of Section 4.1, our proof is a generalization of his. If we do not exactly recover the component-wise analysis of Munos for approximate value iteration, this is not really fundamental as we saw that it does not affect the results once stated in terms of concentrability coefficients.

All our Lp norm bounds involve the use of some simple concentrability coefficient C ( ν ) (defined in Equation 20 page 1197). Munos (2007) introduced some concentrability coefficients that are finer than C ( ν ) . In the same spirit, Farahmand et al. (2010) recently revisited the error propagation of Munos (2007, 2003) and improved (among other things) the constant in the bound related to these concentrability coefficients. In (Scherrer et al., 2012), we have further enhanced this constant by providing even finer coefficients, and provided a practical lemma (Scherrer et al., 2012, Lemma 3) to convert any component-wise bound into an Lp norm bound. Thus, rewriting our results for λ policy iteration with these refined coefficients is straightforward, and is not pursued here.

Possible actions

Possible next states

/

Figure 4: Modeling the Tetris game as an MDP

<!-- image -->

## 6. Application Of λ Policy Iteration To The Game Of Tetris

In the final part of this paper, we consider (and describe for the sake of keeping this paper selfcontained) exactly the same application (Tetris) and implementation as Bertsekas and Ioffe (1996). Our main motivation here comes from the fact that we obtain empirical results that are different (and much less intriguing) than those of the original study. This gives us the opportunity to describe what we think are the reasons for such a difference. But before doing so, we begin by describing the Tetris domain.

## 6.1 The Game of Tetris And Its Model As An MDP

Tetris is a popular video game created in 1985 by Alexey Pajitnov. The game is played on a 10 × 20 grid where pieces of different shapes fall from the top. The player has to choose where each piece is added: he can move it horizontally and rotate it. When a row is filled, it is removed and all cells above it move one row downwards. The goal is to remove as many lines as possible before the game is over, that is when there is not enough space remaining on the top of the pile to put the current new piece.

Instead of mimicking the original game, precisely described by Fahey (2003), Bertsekas and Ioffe (1996) have focused on the main problem, that is choosing where and in which orientation to

drop each coming piece. The corresponding MDP model, illustrated in figure 4, is straightforward: the state consists of the wall configuration and the shape of the current piece. An action is the horizontal translation and the rotation which are applied to the piece before it is dropped on the wall. The reward is the number of lines which are removed after we have dropped the piece. As one considers the maximization of the score (the total number of lines removed during a game), the natural choice for the discount factor is γ = 1, that is we model Tetris as an undiscounted MDP, of which the terminal state corresponds to 'game over'.

In a bit more details, the dynamics of Tetris is made of two components: the place where one drops the current piece and the choice of a new piece. As the latter component is uncontrollable (a new piece is chosen with uniform probability), the value functions does not need to be computed for all wall-piece pairs configurations but only for all wall configurations (Bertsekas and Ioffe, 1996). Also considering that the first component of the dynamics is deterministic, the optimal value function satisfies the following Bellman equation:

<!-- formula-not-decoded -->

where S is the set of wall configurations, P is the set of pieces, A ( p ) is the set of translation-rotation pairs that can be applied to a piece p , r ( s , p , a ) and succ ( s , p , a ) are respectively the number of lines removed and the (deterministic) next wall configuration if one puts a piece p on the wall s in translation-orientation a . The only function that satisfies the above Bellman equation gives, for each wall configuration s , the average best score that can be achieved from s . If we know this function, a one step look-ahead strategy (that is a greedy policy) performs optimally.

## 6.2 An Instance Of Approximate λ Policy Iteration

For large scale problems, many approximate dynamic-programming algorithms are based on two complementary tricks:

- one uses samples to approximate the expectations such as that of Equation 8;
- one only looks for a linear approximation of the optimal value function:

<!-- formula-not-decoded -->

where θ =( θ ( 0 ) . . . θ ( K )) is the parameter vector and Φ k ( s ) are some predefined feature functions on the state space. Thus, each value of θ characterizes a value function v θ over the entire state space.

The instance of approximate λ policy iteration of Bertsekas and Ioffe (1996) follows these ideas. More specifically, this algorithm is devoted to MDPs which have a termination state, that has 0 reward and is absorbing. For this algorithm to be run, one must further assume that all policies are proper, which means that all policies reach the termination state with probability one in finite time. 17

17. Bertsekas and Ioffe (1996) consider a weaker assumption for exact λ policy iteration and its analysis, namely that there exists at least one proper policy. However, this assumption is not sufficient for their approximate algorithm, because this builds sample trajectories that need to reach a termination state. If the terminal state were not reachable in finite time, this algorithm may not terminate in finite time.

This condition holds in the case of Tetris; in fact, Burgiel (1997) has shown that, whatever the strategy, some sequence of pieces (which necessarily occurs in finite time with probability 1) leads to game-over whatever the decisions taken. In particular, this implies that the condition required for our analysis (Equation 21 page 1198) holds.

Similarly to exact λ policy iteration, this approximate λ policy iteration maintains a compact value-policy pair ( θ t , π t ) . Given θ t , π t + 1 is the greedy policy with respect to v θ t , and can easily be computed exactly in any given state as the argmax in Equation 29. This policy π t + 1 is used to simulate a batch of M trajectories: for each trajectory m , ( sm , 0 , sm , 1 , . . . , sm , Nm -1 , sm , Nm ) denotes the sequence of states of the m th trajectory, with sm , Nm being the termination state. Then, for approximating the temporal-difference equation (Equation 8 page 1189), a reasonable choice for θ t + 1 is one that satisfies:

<!-- formula-not-decoded -->

for all trajectories m , where

<!-- formula-not-decoded -->

and for all j &lt; Nm -1,

<!-- formula-not-decoded -->

are the temporal differences. Note that Equations 30 and 31 correspond to the terminal states after which there is no subsequent reward. A standard and efficient solution to this problem consists in minimizing the least-squares error, that is to choose θ t + 1 as follows:

<!-- formula-not-decoded -->

This approximate version of λ policy iteration generalizes well-known algorithms. When λ = 0, the generic term becomes a sample of [ T π k + 1 v ]( sm , k ) :

<!-- formula-not-decoded -->

When λ = 1, the generic term becomes the sampled discounted return from sm , k until the end of the trajectory:

<!-- formula-not-decoded -->

In other words, for these limit values of λ , the algorithms correspond to approximate versions of value and policy iteration as described by Bertsekas and Tsitsiklis (1996). Also, as explained by Bertsekas and Ioffe (1996) and already mentioned in the introduction, the TD( λ ) algorithm with linear features described by Sutton and Barto (1998, chapter 8.2) matches the algorithm we have just described when the above fitting problem is approximated using gradient iterations after each sample.

We follow the same protocol as originally proposed by Bertsekas and Ioffe (1996). Let w = 10 be the width of the board. We consider approximating the value function as a linear combination of 2 w + 2 = 22 feature functions:

<!-- formula-not-decoded -->

where

- for all k ∈ { 1 , 2 , · · · , w } , hk is the height of the k th column of the wall;
- for all k ∈ { 1 , 2 , · · · , w -1 } , ∆ hk is the height difference | hk -hk + 1 | between columns k and k + 1;
- H is the maximum wall height , that is max k hk ;
- L is the number of holes (the number of empty cells covered by at least one full cell).

Westarted our experiments with the initial following vector: θ ( 2 w ) = -10, θ ( 2 w + 1 ) = -1 and θ ( k ) = 0 for all k &lt; 2 w , so that the initial greedy policy scores in the low tens (Bertsekas and Ioffe, 1996). We used M = 100 training games for each policy update. As this implementation of λ policy iteration is stochastic, we ran each experiment 10 times. figure 5 displays the learning curves. The left graph shows the 10 runs (each point is the average score computed with the M = 100 games) and the corresponding point-wise average for a single value of λ , while the right graph shows such point-wise average curves for different values of λ : 0.0, 0.3, 0.5, 0.7 and 0.9. We chose to display on the left graph the runs corresponding to the value of λ = 0 . 9 that seemed to be the best on the right graph.

We can make the following observations.

- Although we initialized with not so bad a policy (the first value is around 30), the performance first drops to 0 and it really starts improving after a few iterations (typically around ten). This is due to the fact that the initial value function is really bad: with the given parameters, the initial value is negative whereas it is clear that the optimal value function (the average best score) is positive. Further experiments showed that the overall behavior of the algorithm was not affected by the weight initialization.

Figure 5: Average Score versus the number of iterations. Left : 10 runs of λ policy iteration with λ = 0 . 9. Each point of each run is the average score computed with M = 100 games. The dark curve is a point-wise average of the 10 runs. Right : Point-wise average of 10 runs of λ policy iteration for different values of λ ; the curve which appears to be the best ( λ = 0 . 9) is the same as the bold curve of the left graph.

<!-- image -->

- The rise of performance globally happens sooner for larger values of λ , that is for values that make the algorithm closer to policy iteration. This is not surprising as it complies with the fact that λ modulates the speed at which the value estimate tracks the real value of the current policy. However, the performance did not rise for λ = 1 (when it is equivalent to approximate policy iteration). We believe this is due to the fact that the variance of the value update is too high.
- Quantitatively, the scores reach an overall level of 4000 lines per games for a big range of values of λ .

The empirical results we have just described qualitatively and quantitatively differ from those of Bertsekas and Ioffe (1996), even though it is the exact same experimental setup. About their results, the authors wrote: ' An interesting and somewhat paradoxical observation is that a high performance is achieved after relatively few policy iterations, but the performance gradually drops significantly. We have no explanation for this intriguing phenomenon, which occurred with all of the successful methods that we tried '. As we explain now, we believe that the 'intriguing' character of the results of Bertsekas and Ioffe (1996) might be related to a subtle implementation difference. Indeed, we can reproduce learning curves that are similar to those of Bertsekas and Ioffe (1996) with a little modification in our implementation of λ policy iteration, that removes the special treatments for the terminal states done through Equations 30 and 31. More precisely, if we replace them by the following equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

that is if we replace the terminal value 0 by the value v θ t ( sm , Nm ) which is computed through the features of the terminal wall configuration sm , Nm , then we get the performance shown in figure 6.

Figure 6: Average score versus the number of iterations of λ policy iteration, modified so that it resembles the results of Bertsekas and Ioffe (1996) (see text for details).

<!-- image -->

We observe that the performance evolution qualitatively matches the performance curves published in Bertsekas and Ioffe (1996) and illustrates the above quotation describing the 'intriguing phenomenon.' 18

In such a modified form, the approximate λ policy iteration algorithm makes much less sense. In particular, it is not true anymore that it reduces to approximate value iteration and approximate policy iteration when λ = 0 and λ = 1 respectively: Equations 34 and 35 induce a bias so that we cannot recover the identities of Equations 32 and 33. A closer examination of these experiments showed that the weights ( θ k ) were diverging. This is not a surprise, since the use of Equations 34 and 35 violates the condition (expressed at the end of Section 4.4) that there should be no error in the terminal state.

## 7. Conclusion And Future Work

We have considered the λ policy iteration algorithm introduced by Bertsekas and Ioffe (1996) that generalizes the standard algorithms value and policy iteration. We have extended the preliminary analysis of this algorithm provided by Bertsekas and Ioffe (1996) in various ways:

1. We have derived non-asymptotic convergence rates for its exact version. In particular, one such rate (Equation 13 page 1194) generalizes that for value iteration by Puterman (1994), and another one (Equation 15) is to our knowledge new even when λ policy iteration reduces to value or policy iteration.

18. A watchful reader may have noticed that the performance that we obtain is about twice that of Bertsekas and Ioffe (1996). A close inspection of the Tetris domain description given by Bertsekas and Ioffe (1996) shows that the authors consider the game of Tetris on a 10 × 19 board instead of our 10 × 20 setting, and as argued in a recent review on Tetris (Thi´ ery and Scherrer, 2009), this small difference is sufficient for explaining such a big performance difference.

2. We have provided asymptotic performance bounds when the algorithm is run with approximation, that generalize those made separately for value iteration (Munos, 2007) and policy iteration (Munos, 2003).
3. Furthermore, under assumptions ensuring that a terminal is reached in finite time with probability 1, we have extended our bounds to the undiscounted situation.

More generally, we believe that an important contribution of this paper is of conceptual nature: we have provided a unified view on some of the main approximate dynamic programming algorithms. Though the usual contraction or monotonicity arguments do not apply anymore, we explained in Section 4.1 how series of component-wise inequalities on objects we called the value , the distance , the shift and the Bellman residuals could lead to bounds on the performance loss. This line of analysis has recently been reused in variations of λ policy iteration. In (Scherrer et al., 2012), this has allowed us to provide an Lp norm performance bound for the modified policy iteration family of algorithms (Puterman and Shin, 1978). In (Thi´ ery and Scherrer, 2010; Scherrer and Thi´ ery, 2010), we have given L ∞ norm performance bounds 19 of an algorithm, named optimistic policy iteration, that makes any convex combination of the modified policy iteration possible updates, and thus generalizes both λ policy iteration and modified policy iteration. We hope that this original line of analysis will be useful for the study of other dynamic-programming/reinforcement-learning algorithms in the future.

Regarding λ policy iteration, an important research direction would be to study the implications of the choice of the parameter λ , as for instance is done by Singh and Dayan (1998) for the value estimation problem. On this matter, the original analysis by Bertsekas and Ioffe (1996) shows how one can concretely implement λ policy iteration. Each iteration requires the computation of the fixed point of the β -contracting operator Mk (see Equation 5 page 1186). We plan to study the trade-off between the ease for computing this fixed point (the smaller β , the faster) and the time for λ policy iteration to converge to the optimal policy (the bigger β , the faster). Although the reader might have noticed that most of our bounds have no explicit dependence on λ , the algorithm implicitly depends on λ through the stochastic matrices that are involved along the iterations, and the variance of the error terms. Understanding better the influence of this main parameter constitutes interesting future work.

Last but not least, we should insist on the fact that the implementation that we have described in Section 6.2, and which is borrowed from Bertsekas and Ioffe (1996), is just one possible instance of λ policy iteration. In the case of linear approximation architectures, Thi´ ery and Scherrer (2010) have proposed an implementation of λ policy iteration that is based on LSPI (Lagoudakis and Parr, 2003), in which the fixed point of Mk is approximated using LSTD(0) (Bradtke and Barto, 1996). Recently, Bertsekas (2011) proposed to compute this very fixed point with a variation of LSPE( λ ′ ) (Bertsekas and Ioffe, 1996; Nedi´ c and Bertsekas, 2003) for some λ ′ potentially different from λ . Because of their very close structure, any existing implementation of approximate policy iteration may probably be turned into some implementation of λ policy iteration. Proposing such implementations and assessing their relative merits constitutes interesting future research. This may in particular be done through some finite sample analysis, as recently done for approximate value and policy iteration implementations (Antos et al., 2007, 2008; Munos and Szepesv´ ari, 2008; Lazaric et al., 2010).

19. The extension to Lp norm is straightforward.

## Acknowledgments

The author would like to thank Christophe Thi´ ery for contributing to the code and the illustration for Tetris, and the two anonymous reviewers for providing many comments that helped improve the presentation of the paper. Preliminary versions of this paper appeared as technical reports on http://hal.inria.fr/inria-00185271/en .

## Appendix A.

The following appendices contain all the proofs concerning the analysis of λ policy iteration. We write Pk = P π k for the stochastic matrix corresponding to the policy π k which is greedy with respect to vk -1, P ∗ for the stochastic matrix corresponding to the optimal policy π ∗ . Similarly we write Tk and T for the associated Bellman operators.

The proof techniques we have developed are inspired by those of Munos (2003, 2007). Most of the inequalities appear from the definition of the greedy operator:

<!-- formula-not-decoded -->

We often use the property that a convex combination of stochastic matrices is also a stochastic matrix. A recurrent instance of this property is: if P is some stochastic matrix, then the geometric average

<!-- formula-not-decoded -->

with 0 ≤ α &lt; 1 is also a stochastic matrix. We use the property that if some vectors x and y are such that x ≤ y , then Px ≤ Py for any stochastic matrix P . Eventually, we will use the following equivalent forms of the operator T π λ (three of them were introduced in page 1187): for any value v and any policy π , we have

<!-- formula-not-decoded -->

## Appendix B. Proofs Of Lemmas 3-5 (Core Lemmas Of The Error Propagation)

In this section, we prove the series of Lemmas that are at the heart of our analysis of the error propagation of λ policy iteration.

## B.1 Proof Of Lemma 3 (A Relation Between The Shift And the Bellman Residual)

Using the definition of wk = T π k λ vk -1 and the formulation of Equation 37, we can see that we have:

<!-- formula-not-decoded -->

Therefore with

with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Suppose that we have a lower bound of the Bellman residual: bk -1 ≥ bk -1 (we shall derive one soon). Since ( I -γ Pk ) -1 Ak only has non-negative elements then

<!-- formula-not-decoded -->

## B.2 Proof Of Lemma 4 (A Lower Bound On The Bellman Residual)

From the definition of the algorithm, and using the fact that Tkv π k = v π k , we see that:

<!-- formula-not-decoded -->

where we eventually used the relation between sk and bk (Lemma 3). In other words:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since Ak is a stochastic matrix and β ≥ 0, we get by induction:

<!-- formula-not-decoded -->

## B.3 Proof Of Lemma 5 (An Upper Bound On The Distance)

Given that T ∗ v ∗ = v ∗ , we have

<!-- formula-not-decoded -->

Using the definition of wk + 1 = T π k + 1 λ vk and the formulation of Equation 36, one can see that the distance satisfies:

<!-- formula-not-decoded -->

Since π k + 1 is greedy with respect to vk , we have Tk + 1 vk ≥ T ∗ vk and therefore:

<!-- formula-not-decoded -->

As a consequence, the distance satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Noticing that:

we get:

where

Since P ∗ is a stochastic matrix and γ ≥ 0, we have by induction:

<!-- formula-not-decoded -->

## Appendix C. Proof Of Lemma 6 (Performance Of Exact λ Policy Iteration

We here derive the convergence rate bounds for exact λ policy iteration (as expressed in Lemma 6 page 1193). We rely on the loss bound analysis of appendix B with ε k = 0. In this specific case, we know that the loss l k ≤ dk + sk where

<!-- formula-not-decoded -->

Introducing the following stochastic matrices:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have and

Therefore the loss satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with

To end the proof, we simply need to prove the following lemma:

Lemma 18 E ′ k is a stochastic matrix.

Proof Using the facts that λγ γ -β = 1 1 -β and ( 1 -β )( 1 -λγ ) = 1 -γ , one can observe that

<!-- formula-not-decoded -->

and deduce that E ′ k is a stochastic matrix, since it is a convex combination of stochastic matrices.

## C.1 Proof Of Equation 11 (A Bound With Respect To The Bellman Residual)

We first need the following lemma:

Lemma 19 The bias and the distance are related as follows:

<!-- formula-not-decoded -->

Proof Since π k + 1 is greedy with respect to vk , Tk + 1 vk ≥ T ∗ vk and

<!-- formula-not-decoded -->

We thus have:

<!-- formula-not-decoded -->

Then Equation 38 becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where:

is a stochastic matrix.

## C.2 Proof Of Equation 10 (A Bound With Respect To The Distance)

From Lemma 19, we know that

Then, Equation 38 becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where is a stochastic matrix.

<!-- formula-not-decoded -->

## C.3 Proof Of Equation 12 (A Bound With Respect To The Distance And The Loss Of The Greedy Policy)

Define ˆ v 0 : = v 0 -Ke where K is some constant and e denotes the vector of which all components are 1. The following statements are equivalent:

<!-- formula-not-decoded -->

The minimal K for which ˆ b 0 ≥ 0 is thus K : = max s [ v 0 ( s ) -v π 1 ( s )] . As ˆ v 0 and v 0 only differ by a constant vector, they generate the same sequence of policies π 1 , π 2 ... Then, as ˆ b 0 ≥ 0, Equation 38 implies that

<!-- formula-not-decoded -->

The result is obtained by noticing that

<!-- formula-not-decoded -->

## Appendix D. Proof Of Equation 16 In Lemma 9 (Component-Wise Bounds On The Error Propagation)

We here use the loss bound analysis of appendix B to derive an asymptotic analysis of approximate λ policy iteration with respect to the approximation error. The results stated here constitute a proof of the first inequality of Lemma 9 page 1194.

## D.1 Proof Of Equation 16

Since the loss satisfies

<!-- formula-not-decoded -->

an upper bound of the loss can be derived from the upper bound of the distance and the shift.

Let us first concentrate on the bound dk of the distance. Lemmas 4 and 5 imply that:

<!-- formula-not-decoded -->

Writing

<!-- formula-not-decoded -->

and putting all things together, we see that:

<!-- formula-not-decoded -->

where between the first two lines, we used the fact that:

<!-- formula-not-decoded -->

using the identities λγ = γ -β 1 -β and 1 -γλ = 1 -γ 1 -β .

Let us now consider the bound sk of the shift. From Lemma 3 and the bound on bk in Equation 40, we have

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Eventually, from Equations 39, 41 and 42 we get:

<!-- formula-not-decoded -->

Introduce the following matrices:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 20 Bkj and B ′ k j are stochastic matrices.

Proof Using the identities: λγ = γ -β 1 -β and ( 1 -β )( 1 -γλ ) = 1 -γ , one can see that

<!-- formula-not-decoded -->

and deduce that Bkj is a stochastic, since it is a convex combination of stochastic matrices. Then it is also clear that B ′ k j is a stochastic matrix.

Thus, Equation 43 can be rewritten as follows:

<!-- formula-not-decoded -->

## Appendix E. Proofs Of Equations 17-18 In Lemma 9 (Component-Wise Bounds With Respect To The Bellman Residuals)

In this section, we study the loss

<!-- formula-not-decoded -->

with respect to the two following Bellman residuals :

<!-- formula-not-decoded -->

The term b ′ k says how much vk differs from the value of π k while bk says how much vk differs from the value of the policies π k + 1 and π ∗ . The results stated here prove the last two inequalities of Lemma 9 page 1194.

## E.1 Proof Of Equation 17 (Bounds With Respect To The Policy Bellman Residual)

Our analysis relies on the following lemma

Lemma 21 Suppose that we have a policy π , a function v that is an approximation of the value v π of π in the sense that its residual b ′ : = T π v -v is small. Taking the greedy policy π ′ with respect to v reduces the loss as follows:

where P and P ′ are the stochastic matrices which correspond to π and π ′ .

<!-- formula-not-decoded -->

Proof We have:

<!-- formula-not-decoded -->

where we used the fact that T ∗ v ≤ T π ′ v . One can see that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used the fact that T π v ≤ T π ′ v . We get the result by putting back Equations 45 and 46 into Equation 44.

To derive a bound for λ policy iteration, we simply apply the above lemma to π = π k , v = vk and π ′ = π k + 1. We thus get:

<!-- formula-not-decoded -->

By induction, we obtain for all k ,

<!-- formula-not-decoded -->

where we have defined the following stochastic matrices:

<!-- formula-not-decoded -->

## E.2 Proof Of Equation 18 (Bounds With Respect To The Bellman Residual)

We rely on the following lemma, that is for instance proved by Munos (2007).

Lemma 22 Suppose that we have a function v. Let π be the greedy policy with respect to v. Then

<!-- formula-not-decoded -->

and that

We provide a proof for the sake of completeness: Proof Using the fact that T ∗ v ≤ T π v , we see that

<!-- formula-not-decoded -->

Using Equation 45 we see that:

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

To derive a bound for λ policy iteration, we simply apply the above lemma to v = vk -1 and π = π k . We thus get:

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

are stochastic matrices.

## Appendix F. Proofs Of Corollary 14

This section provides a proof of Corollary 14 page 1198, in which we refine the bounds when the value or the policy converges.

## F.1 Proof Of The First Inequality Of Corollary 14 (When The Value Converges)

Suppose that λ policy iteration converges to some value v . Let policy π be the corresponding greedy policy, with stochastic matrix P . Let b be the Bellman residual of v . It is also clear that the approximation error also converges to some ε . Indeed from Algorithm 3 and Equation 6, we get:

<!-- formula-not-decoded -->

From the bound with respect to the Bellman residual (Equation 47 page 1219), we can see that:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Lemma 23 Bv and D are stochastic matrices.

Proof It is clear that D is a stochastic matrix. For Bv , we simply observe that

<!-- formula-not-decoded -->

and deduce that Bv is a stochastic matrix, as a convex combination of stochastic matrices. Then, the first bound of Corollary 14 follows from the application of Lemmas 10 and 12.

## F.2 Proof Of The Second Inequality Of Corollary 14 (When The Policy Converges)

Suppose that λ policy iteration converges to some policy π . Write P the corresponding stochastic matrix and

<!-- formula-not-decoded -->

Then for some big enough k 0, we have:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is a stochastic matrix (for the same reasons why Bkj is a stochastic matrix in Lemma 20). Noticing that

<!-- formula-not-decoded -->

we can deduce that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 24 B π k j and B ′ π k j are stochastic matrices.

Proof It is clear that B π k j is a stochastic matrix. Also, since

<!-- formula-not-decoded -->

B ′ π k j is a convex combination of stochastic matrices, and thus a stochastic matrix. Then, the second bound of Corollary 14 follows from the application of Lemmas 10 and 12.

## Appendix G. Proofs Of Lemmas 7 And 10 (From Component-Wise Bounds To Lp Norm Bounds)

This section contains the proofs of Lemmas 7 (page 1194) and 10 (page 1196) that enable us to derive Lp norm performance bounds from component-wise bounds. It is easy to see that Lemma 7 is a special case of Lemma 10, so we only prove the latter.

Consider the notations of Lemma 10. We have for all k ,

<!-- formula-not-decoded -->

By taking the absolute value and using the fact that Xkj and X ′ k j are stochastic matrices, we get for all k ,

<!-- formula-not-decoded -->

It can then be seen that

<!-- formula-not-decoded -->

By using Jensen's inequality (with the convex function x ↦→ x p ), we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used ∑ k -1 j = 0 ξ k -j ≤ K ′ . Wecan apply the exact same analysis to any starting index l (instead of 0) and since the function l ↦→ sup k ′ ≥ j ′ ≥ l ∥ ∥ y j ′ ∥ ∥ p , µ k ′ j ′ is non-decreasing, we deduce that:

and the result follows.

## Appendix H. Proofs Of Lemma 15 And Proposition 16 (Analysis Of The Undiscounted Case)

This last section contains the proofs of Lemma 15 and Proposition 16 that provide the analysis of an undiscounted problem.

## H.1 Proof Of Lemma 15 (Component-Wise Bound)

First of all, we recall the relation expressed in Equation 22 page 1199 between the loss and the stochastic matrices:

<!-- formula-not-decoded -->

It is obtained by simply rewriting the first inequality of Lemma 9 with γ = 1 and β = 1 (note in particular that the terms δ k -j collapse through the definition of Gkj and G ′ k j ).

To complete the proof of the lemma, we need to show that the matrices Gkj and G ′ k j are substochastic matrices. By construction, these matrices are sum of non-negative matrices so we only need to show that their max norm is smaller than or equal to 1.

For all n , write M n the set of matrices that is defined as follows:

- for all sets of n policies ( π 1 , π 2 , · · · , π n ) , P π 1 P π 2 · · · P π n ∈ M n ;
- for all η ∈ ( 0 , 1 ) , and ( P , Q ) ∈ M n × M n , η P +( 1 -η ) Q ∈ M n .

The motivation for introducing this set is that we have the following properties: For all n , P ∈ M n is a sub-stochastic matrix such that ‖ P ‖ ∞ ≤ α ⌊ n n 0 ⌋ . We use the somewhat abusive notation Π n for denoting any element of M n . For instance, for some matrix P , writing P = a Π i + b Π j Π k = a Π i + b Π j + k should be read as follows: there exist P 1 ∈ M i , P 2 ∈ M j , P 3 ∈ M k and P 4 ∈ M k + j such that P = aP 1 + bP 2 P 3 = aP 1 + bP 4.

Recall the definition of the sub-stochastic matrix

<!-- formula-not-decoded -->

Let i ≤ j &lt; k . It can be seen that

<!-- formula-not-decoded -->

Now, observe that

<!-- formula-not-decoded -->

As a consequence, writing η : = 1 -λ n 0 1 -λ n 0 α , we see from Equation 48 that

<!-- formula-not-decoded -->

∥ ∥ Similarly, by using Equation 49 and noticing that 1 -λ n 0 1 -λ λ → 1 -→ n 0, it can be seen that

<!-- formula-not-decoded -->

We are ready to bound the norm of the matrix Gkj :

<!-- formula-not-decoded -->

where we used the definition of η . Therefore Gkj is a sub-stochastic matrix. It trivially follows that G ′ k j is also a sub-stochastic matrix.

## H.2 Proof Of Proposition 16 ( Lp Norm Bound)

In order to prove the Lp norm bound of Proposition 16, we rely on the following variation of Lemma 10.

Lemma 25 If x k and yk are sequences of vectors and Xkj, X ′ k j sequences of sub-stochastic matrices satisfying

<!-- formula-not-decoded -->

where ( ξ i ) i ≥ 1 is a sequence of non-negative weights satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is a non-negative vector and ˜ µkj : = µkj ‖ µkj ‖ 1 is a distribution, and

<!-- formula-not-decoded -->

∥ ∥ Proposition 16 is obtained by applying this Lemma and an analogue of Lemma 12 for Lp norm on the component-wise bound (Lemma 15, see previous subsection). The only remaining thing that needs to be checked is that ∑ ∞ i = 1 δ i has the right value. This is what we do now.

Proof The proof follows the lines of that of Lemma 10 in appendix G. The only difference is that in order to express the bound in terms of the distributions ˜ µkj , we use the fact that µkj ≤ ˜ µkj which derives from ∥ µkj ∥ 1 ≤ 1 since Xkj and X ′ k j are sub-stochastic matrices.

Similarly to Equation 49, one can see that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then, for all distribution µ, and

As a consequence,

<!-- formula-not-decoded -->

where for all x , f ( x ) : = ( 1 -x n 0 ) ( 1 -x )( 1 -x n 0 α ) and f ( 1 ) = n 0 1 -α by continuity. Now, we can conclude by noticing that and δ 0 = n 0 1 -α = f ( 1 ) .

## References

- A. Antos, Cs. Szepesv´ ari, and R. Munos. Value-iteration based fitted policy iteration: Learning with a single trajectory. In ADPRL , pages 330-337. IEEE, 2007.
- A. Antos, Cs. Szepesv´ ari, and R. Munos. Learning near-optimal policies with Bellman-residual minimization based fitted policy iteration and a single sample path. Machine Learning Journal , 71:89-129, 2008.
- D. Bertsekas. Lambda policy iteration: A review and a new implementation. Technical Report LIDS-2874, MIT, 2011.
- D. Bertsekas and S. Ioffe. Temporal differences-based policy iteration and applications in neurodynamic programming. Technical Report LIDS-P-2349, MIT, 1996.
5. D.P. Bertsekas and J.N. Tsitsiklis. Neurodynamic Programming . Athena Scientific, 1996.
6. S.J. Bradtke and A.G. Barto. Linear least-squares algorithms for temporal difference learning. Machine Learning , 22(1-3):33-57, 1996.
- H. Burgiel. How to Lose at Tetris. Mathematical Gazette , 81:194-200, 1997.
- C. P. Fahey. Tetris AI, computer plays tetris. http://colinfahey.com/tetris/tetris\_en. html , 2003.
9. A.M. Farahmand, R. Munos, and Cs. Szepesv´ ari. Error propagation for approximate policy and value iteration. In Advances in Neural Information Processing Systems 23 (NIPS 2010) , 2010.
10. M.G. Lagoudakis and R. Parr. Least-squares policy iteration. Journal of Machine Learning Research , 4:1107-1149, 2003.
- A. Lazaric, M. Ghavamzadeh, and R. Munos. Analysis of a classification-based policy iteration algorithm. In International Conference on Machine Learning (ICML 2010) , pages 607-614, 2010.
- R. Munos. Error bounds for approximate policy iteration. In International Conference on Machine Learning (ICML 2003) , pages 560-567, 2003.
- R. Munos. Performance bounds in Lp -norm for approximate value iteration. SIAM Journal on Control and Optimization , 46(2):541-561, 2007.
- R. Munos and Cs. Szepesv´ ari. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research , 9:815-857, 2008.

<!-- formula-not-decoded -->

- A. Nedi´ c and D. P. Bertsekas. Least squares policy evaluation algorithms with linear function approximation. Discrete Event Dynamic Systems , 13:79-110, 2003.
- M. Puterman. Markov Decision Processes . Wiley, New York, 1994.
- M. Puterman and M. Shin. Modified policy iteration algorithms for discounted markov decision problems. Management Science , 24(11), 1978.
- B. Scherrer and C. Thi´ ery. Performance bound for approximate optimistic policy iteration. Technical report, INRIA, 2010.
- B. Scherrer, M. Ghavamzadeh, V. Gabillon, and M. Geist. Approximate modified policy iteration. In International Conference on Machine Learning (ICML 2012) , Edinburgh, Scotland, 2012.
- S. Singh and P. Dayan. Analytical mean squared error curves for temporal difference learning. Machine Learning Journal , 32(1):5-40, 1998.
7. R.S. Sutton and A.G. Barto. Reinforcement Learning, An introduction . BradFord Book. The MIT Press, 1998.
- C. Thi´ ery and B. Scherrer. Improvements on learning Tetris with cross entropy. International Computer Games Association Journal , 32, 2009.
- C. Thi´ ery and B. Scherrer. Least-squares λ policy iteration: Bias-variance trade-off in control problems. In International Conference on Machine Learning (ICML 2010) , Haifa, Isra¨ el, 2010.