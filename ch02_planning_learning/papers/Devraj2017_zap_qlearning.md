## Fastest Convergence for Q-Learning

Adithya M. Devraj and Sean P. Meyn

March 23, 2018

## Abstract

The Zap Q-learning algorithm introduced in this paper is an improvement of Watkins' original algorithm and recent competitors in several respects. It is a matrix-gain algorithm designed so that its asymptotic variance is optimal. Moreover, an ODE analysis suggests that the transient behavior is a close match to a deterministic Newton-Raphson implementation. This is made possible by a two time-scale update equation for the matrix gain sequence.

The analysis suggests that the approach will lead to stable and efficient computation even for non-ideal parameterized settings. Numerical experiments confirm the quick convergence, even in such non-ideal cases. The comparison plot on this first page, taken from Fig. 9 of this paper, is an illustration of the amazing acceleration in convergence using the new algorithm.

A secondary goal of this paper is tutorial. The first half of the paper contains a survey on reinforcement learning algorithms, with a focus on minimum variance algorithms.

Keywords: Reinforcement learning, Q-learning, Stochastic optimal control 2000 AMS Subject Classification: 93E20, 93E35

<!-- image -->

x10

∗ Research supported by the National Science Foundation under grants CPS-0931416 and EPCN-1609131

† A.D. and S.M. are with the Department of Electrical and Computer Engg. at the University of Florida, Gainesville.

A part of the research was conducted when the authors were visitors at the Simons Institute for the Theory of Computing at University of California, Berkeley.

∗ †

## 1 Introduction

It is recognized that algorithms for reinforcement learning such as TD- and Q-learning can be slow to converge. The poor performance of Watkins' Q-learning algorithm was first quantified in [31], and since then many papers have appeared with proposed improvements, such as [10, 1].

An emphasis in much of the literature is computation of finite-time PAC (probably almost correct) bounds as a metric for performance. Explicit bounds were obtained in [31] for Watkins' algorithm, and in [1] for the 'speedy' Q-learning algorithm that was introduced by these authors. A general theory is presented in [21] for stochastic approximation algorithms.

In each of the models considered in prior work, the update equation for the parameter estimates can be expressed

<!-- formula-not-decoded -->

in which { α n } is a positive gain sequence, and { ∆ n } is a martingale difference sequence. This representation is critical in analysis, but unfortunately is not typical in reinforcement learning applications outside of these versions of Q-learning. For Markovian models, the usual transformation used to obtain a representation similar to (1) results in an error sequence { ∆ n } that is the sum of a martingale difference sequence and a telescoping sequence [16]. It is the telescoping sequence that prevents easy analysis of Markovian models.

This gap in the research literature carries over to the general theory of Markov chains. Examples of concentration bounds for i.i.d. sequences or martingale-difference sequences include the finitetime bounds of Hoeffding and Bennett. Extensions to Markovian models either offer very crude bounds [19], or restrictive assumptions [15, 11]; this remains an active area of research [23].

In contrast, asymptotic theory for stochastic approximation (as well as general state space Markov chains) is mature. Large Deviations or Central Limit Theorem (CLT) limits hold under very general assumptions [3, 14, 5].

The CLT will be a guide to algorithm design in the present paper. For a typical stochastic approximation algorithm, this takes the following form: denoting { ˜ θ n := θ n -θ ∗ : n ≥ 0 } to be the error sequence, under general conditions the scaled sequence { √ n ˜ θ n : n ≥ 1 } converges in distribution to a Gaussian distribution, N (0 , Σ θ ). Typically, the scaled covariance is also convergent:

<!-- formula-not-decoded -->

The limit is known as the asymptotic covariance .

An asymptotic bound such as (2) may not be satisfying for practitioners of stochastic optimization or reinforcement learning, given the success of finiten performance bounds in prior research. There are however good reasons to apply this asymptotic theory in algorithm design:

- (i) The asymptotic covariance Σ θ has a simple representation as the solution to a Lyapunov equation. It is easily improved or optimized by design.
- (ii) As shown in examples in this paper, the asymptotic covariance is often a good predictor of finite-time performance, since the CLT approximation is accurate for reasonable values of n .

Two approaches are known for optimizing the asymptotic covariance. First is the remarkable averaging technique of Polyak and Juditsky [24, 25] and Ruppert [27] ([12] provides an accessible treatment in a simplified setting). Second is what we will call Stochastic Newton-Raphson , based on a special choice of matrix gain for the algorithm. The second approach underlies the analysis of the averaging approach.

We are not aware of theory that distinguishes the performance of Polyak-Ruppert averaging as compared to the Stochastic Newton-Raphson method. It is noted in [21] that the averaging

approach often leads to very large transients, so that the algorithm should be modified (such as through projection of parameter updates). This may explain why averaging is not very popular in practice. In our own numerical experiments it is observed that the rate of convergence of CLT in this case is slow when compared to matrix gain methods.

In addition to accelerating the convergence rate of standard algorithms for reinforcement learning, it is hoped that this paper will lead to entirely new algorithms. In particular, there is little theory to support Q-learning in non-ideal settings in which the optimal ' Q -function' does not lie in the parameterized function class. Convergence results have been obtained for a class of optimal stopping problems [37], and for deterministic models [17]. There is now intense practical interest, despite an incomplete theory. A stronger supporting theory will surely lead to more efficient algorithms.

Contributions A new class of algorithms is proposed, designed to more accurately mimic the classical Newton-Raphson algorithm. It is based on a two time-scale stochastic approximation algorithm, constructed so that the matrix gain tracks the gain that would be used in a deterministic Newton-Raphson method.

The application of this approach to reinforcement learning results in the new Zap Q-learning algorithms. A full analysis is presented for the special case of a complete parameterization (similar to the setting of Watkins' original algorithm). It is found that the associated ODE has a remarkable and simple representation, which implies consistency under suitable assumptions. Extensions to non-ideal parameterized settings are also proposed, and numerical experiments show dramatic variance reductions. Moreover, results obtained from finiten experiments show close solidarity with asymptotic theory.

The potential complexity introduced by the matrix gain is not of great concern in many cases, because of the dramatically acceleration in the rate of convergence. Moreover, the main contribution of this paper is not a single algorithm but a class of algorithms, wherein the computational complexity can be dealt with separately. For example, in a parameterized setting, the basis functions can be intelligently pruned via random projection [2].

The remainder of the paper is organized as follows. Background on computing and optimizing the asymptotic covariance is contained in Section 2. Application to Q-learning, and theory surrounding the new Zap Q-learning algorithm is developed in Section 3. Numerical results are surveyed in Section 4, and conclusions are contained in Section 5. The proofs of the main results are contained in the Appendix; the final page contains Table 2 containing a list of notation.

## 2 Stochastic Newton Raphson and TD-Learning

This first section is largely a tutorial on reinforcement learning. It is shown that the LSTD( λ ) learning algorithm of [8, 7, 22] is an instance of the 'SNR algorithm', in which there is only one time-scale for the parameter and matrix-gain updates. The original motivation for the LSTD( λ ) algorithm had no connection with asymptotic variance. It was shown later in [13] that the LSTD ( λ ) algorithm is the minimum asymptotic variance version of the TD ( λ ) algorithm of [30].

The focus is on fixed point equations associated with an uncontrolled Markov chain, denoted X = { X n : n = 0 , 1 , . . . } , on a measurable state space ( X , B ( X )). It is assumed to be ψ -irreducible and aperiodic [20]. In Section 3 we specialize to a finite state space.

In control applications and analysis of learning algorithms, it is necessary to construct a Markov chain Φ , of which X is a component. Other components may be an input process, or a sequence of

'eligibility vectors' that arise in TD-learning. It will be assumed throughout that there is a unique stationary realization of Φ , with unique marginal distribution denoted glyph[pi1] .

## 2.1 Motivation from SA &amp; ODE fundamentals

The goal of stochastic approximation is to compute the solution f ( θ ∗ ) = 0 for a function f : R d → R d . If the function is easily evaluated, then successive approximation can be used, and under stronger conditions the Newton-Raphson algorithm:

<!-- formula-not-decoded -->

Under general conditions the convergence rate of (3) is quadratic (much faster than geometric), which is not generally true of successive approximation.

Stochastic approximation is itself an approximation of successive approximation. It is assumed that f ( θ ) = E [ f ( θ, Φ)], where f : R d × R m → R d and Φ is a random variable with distribution glyph[pi1] . The standard stochastic approximation algorithm is defined by

<!-- formula-not-decoded -->

For simplicity it is assumed that Φ is the stationary realization of the Markov chain. It is always assumed that the scalar gain sequence { α n } is non-negative, and satisfies:

<!-- formula-not-decoded -->

While convergent under general conditions, the rate of convergence of (4) can often be improved dramatically through the introduction of a matrix gain. This is explained first in a simple linear setting.

## 2.2 Optimal covariance for linear stochastic approximation

In many applications of reinforcement learning we arrive at a linear recursion of the form

<!-- formula-not-decoded -->

where A n +1 = A (Φ n +1 ) is a d × d matrix and b n +1 = b (Φ n +1 ) is a d × 1 vector, n ≥ 0. Let A,b denote the respective steady-state means:

<!-- formula-not-decoded -->

It is assumed throughout this section that A is Hurwitz: the real part of each eigenvalue is negative. Under this assumption, and subject to mild conditions on Φ , it is known that { θ n } converges with probability one to θ ∗ = A -1 b [3, 14, 5].

Convergence of the recursion (6) will be assumed henceforth. It is also assumed that the gain sequence is given by α n = 1 /n , n ≥ 1.

Under general conditions, the asymptotic covariance Σ θ defined in (2) is the non-negative semidefinite solution to the Lyapunov equation:

<!-- formula-not-decoded -->

A solution is guaranteed only if each eigenvalue of A has real part that is strictly less than -1 / 2. If there exists an eigenvalue which does not satisfy this property, then under general conditions

the asymptotic covariance is infinity (see Thm. 2.1). Hence the Hurwitz assumption must be strengthened to ensure that the asymptotic covariance is finite.

The matrix Σ ∆ is obtained as follows: based on (6), the error sequence { ˜ θ n = θ n -θ ∗ } evolves according to a deterministic linear system driven by 'noise':

<!-- formula-not-decoded -->

in which ∆ is the sum of three terms:

<!-- formula-not-decoded -->

with ˜ A n +1 = A n +1 -A , ˜ b n +1 = b n +1 -b . The third term vanishes with probability one. The 'noise covariance matrix' Σ ∆ has the following two equivalent forms:

<!-- formula-not-decoded -->

in which S T = ∑ T n =1 ∆ n , and

<!-- formula-not-decoded -->

where the expectation is in steady-state. It is assumed that the CLT holds for sample-averages of the noise sequence:

<!-- formula-not-decoded -->

where the limit is in distribution. This is a mild requirement when Φ is Markovian [20].

A finite asymptotic covariance can be guaranteed by increasing the gain: choose α n = g/n in (6), with g &gt; 0 sufficiently large so that the eigenvalues of gA satisfy the required bound. More generally, a matrix gain can be introduced:

<!-- formula-not-decoded -->

in which G is a d × d matrix. Provided the matrix GA satisfies the eigenvalue bound, the corresponding asymptotic covariance Σ G θ is finite and solves a modified Lyapunov equation:

<!-- formula-not-decoded -->

The choice G ∗ = -A -1 is analogous to the gain used in the Newton-Raphson algorithm (3). With this choice, the asymptotic covariance is finite and given by

<!-- formula-not-decoded -->

It is a remarkable fact that this choice is optimal in the strongest possible statistical sense: For any other gain G , the two asymptotic covariance matrices satisfy

<!-- formula-not-decoded -->

That is, the difference Σ G θ -Σ ∗ is positive semi-definite [3, 14, 5].

The following theorem summarizes the results on the asymptotic covariance for the matrix-gain recursion (12). The proof is contained in Section A.1 of the Appendix.

Theorem 2.1. Suppose that the eigenvalues of GA lie in the strict left half plane, and that the noise sequence satisfies the CLT (11) with finite covariance Σ ∆ . Then, the stochastic approximation recursion defined in (12) is convergent, and the following also hold:

- (i) Suppose that ( λ, v ) is an eigenvalue-eigenvector pair satisfying

<!-- formula-not-decoded -->

where v † denotes the conjugate transpose of the vector v . Then

<!-- formula-not-decoded -->

and consequently, the asymptotic covariance Σ G θ is not finite.

- (ii) If all the eigenvalues of GA satisfy Re ( λ ) &lt; -1 / 2 , then the corresponding asymptotic covariance Σ G θ is finite, and can be obtained as the solution to the Lyapunov equation (13)
- (iii) For any matrix gain G the asymptotic covariance admits the lower bound

<!-- formula-not-decoded -->

This lower bound is achieved using G ∗ := -A -1 .

glyph[intersectionsq]

glyph[unionsq]

Thm. 2.1 inspires improved algorithms in many settings. The first, which is essentially known, e.g. [27, 14, p. 331], will be called stochastic Newton-Raphson (SNR).

Stochastic Newton-Raphson This algorithm is obtained by estimating the mean A simultaneously with the estimation of θ ∗ : recursively define

<!-- formula-not-decoded -->

where θ 0 and ̂ A 1 are initial conditions.

If the steady-state mean A (defined in (7)) is invertible, then ̂ A n is invertible for all n sufficiently large.

The sequence { n ̂ A n θ n : n ≥ 0 } admits a simple recursive representation that implies the following alternative representation of the SNR parameter estimates:

Proposition 2.2. Suppose ̂ A n is invertible for each n ≥ 1 . Then, the sequence of estimates { θ n } obtained using (15) are identical to the direct estimates:

<!-- formula-not-decoded -->

Based on the proposition, it is obvious that the SNR algorithm is consistent whenever the Law of Large Numbers holds for the sequence { A n , b n } . Under the assumptions of Thm. 2.1, the resulting asymptotic covariance is identical to what would be obtained with the constant matrix gain G ∗ = -A -1 .

Algorithm design in this linear setting is simplified in part because f is an affine function of θ , so that the gain G n appearing in the standard Newton-Raphson algorithm (3) does not depend upon the parameter estimates { θ k } . However, an ODE analysis of the SNR algorithm suggests that even in this linear setting, the dynamics are very different from its deterministic counterpart:

<!-- formula-not-decoded -->

While evidently A t converges to A exponentially fast in the linear model, with a poor initial condition we might expect poor transient behavior.

In extending the SNR algorithm to a nonlinear stochastic approximation algorithm, an ODE approximation of the form (16) will be possible under general conditions, but the matrix A will depend on θ . In addition to poor transient behavior, the coupled equations may be difficult to analyze. And, just as in the linear model, the continuous time system looks very different from the deterministic Newton-Raphson recursion (3).

The next class of algorithms are designed so that the associated ODE more closely matches the deterministic recursion.

## 2.3 Zap Stochastic Newton-Raphson

This is a two time-scales algorithm with a higher step-size for the matrix recursion. In the linear setting of this section, it is defined by the variant of (15):

<!-- formula-not-decoded -->

It is different from the original Stochastic Newton-Raphson algorithm because of the two time-scale construction: The second step-size sequence { γ n +1 } is non-negative, satisfies (5), and also

<!-- formula-not-decoded -->

The asymptotic covariance is again optimal. The ODE associated with the sequence { θ n } is far simpler, and exactly matches the usual Newton-Raphson dynamics:

<!-- formula-not-decoded -->

This simplicity is also revealed in application to Q-learning, in which A depends on the parameter. A key point to note here is that the Zap version of the SNR algorithm plays a significant role in analysis as well as in performance improvement of general non-linear function approximation problems. We briefly discuss these in the following.

## 2.3.1 Zap SNR for non-linear stochastic approximation

Consider a stochastic approximation algorithm of the form (4) with f ( θ ) = E [ f ( θ, Φ)], a nonlinear function of the parameter vector θ . The ODE of the two algorithms: SNR and Zap-SNR look significantly different in this case; it is found that this difference is reflected in the rate of

We again take α n = 1 /n , n ≥ 1.

convergence of the stochastic recursion (as we will see in the case of Q-learning). The SNR algorithm is essentially the same as (15):

<!-- formula-not-decoded -->

Note that the function ∇ f ( θ n , φ n +1 ) may or may not be readily accessible, and this is application specific. In the case of Q-learning with linear function approximation, though the function f is iteslf non-linear in θ , ∇ f is readily computable.

The ODE for the pair of recursions (20) once again will be similar to (16):

<!-- formula-not-decoded -->

The Zap-SNR algorithm is a generalization of (17):

<!-- formula-not-decoded -->

where once again the step-size sequence { γ n } satisfies (5), and (18). Similar to (19), the ODE of this algorithm is identical to the deterministic Newton-Raphson dynamics:

<!-- formula-not-decoded -->

The general convergence and stability analysis of both (20) and (22) is open. In Section 3 we show that when applied to Q-learning, the algorithms do converge under certain technical conditions. However, the assumptions under which the single time-scale algorithm (20) converges is far more restrictive than the assumptions under which the the two-time-scale algorithm (22) converges.

## 2.3.2 Dealing with complexity: An O ( d ) Zap-SNR algorithm

It is common to discard the idea of second order methods because of their computational complexity. Before we move on to the specific applications in Reinforcement Learning, we propose an enhancement of the SNR algorithms that will result in complexity that is comparable to first order methods.

We believe that we have convinced the readers that the two-timescale Zap-SNR algorithm (22) is of more interest to us (we will make this more precise in Section 3), and hence restrict to extensions of this algorithm here.

It is assumed that there is no complexity in 'calculating' the gradient function ∇ f ( · , · ), and that it is readily available. This is not be true in all applications, but holds in the applications of interest in this paper. Under these assumptions, computational complexity arises from the operations that are performed in manipulating these quantities.

The per-iteration complexity of the first order algorithm (1) is O ( d ), since θ ∈ R d . If the algorithm is run for T iterations (assuming we have a data sequence of length T ), the total complexity is O ( dT ). The per iteration complexity in the case of the Zap-SNR algorithm (22) is O ( d 2 ), because it involves the product of a matrix inverse (of dimension d × d ) and a vector (of dimension d × 1). The total complexity of the algorithm after running for T iterations is O ( Td 2 ).

The essential idea behind the O ( d ) Zap-SNR algorithm is to perform the O ( d 2 ) complexity steps only once every N ≥ d iterations, so that the total computational complexity for a data sequence of length T is O ( Td 2 N ); essentially resulting in the complexity of the first order method if N = d . This is done by 'batching' the data sequence into mini-sequences of length N , and applying recursions (22) for each batch as follows: For i ≥ 0

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first two definitions in (25) are straightforward; the expression for ˆ γ i +1 ,N is obtained in such a way that the recursions in (24) very closely resemble the recursions in (22) 1 .

A remarkable (but almost obvious) property of the O ( d ) Zap-SNR algorithm (24) is that it has the same asymptotic properties (specifically, the asymptotic covariance) as that of the original Zap-SNR algorithm (22). This once again is made more precise in a future version of the paper. The specific application of this algorithm to Q-learning is discussed in Section 3.7.

## 2.4 Application to temporal-difference algorithms

The general theory is illustrated here, through application to TD( λ )-learning algorithms.

Let { P n } denote the transition semigroup for the Markov chain X : For each n ≥ 0, x ∈ X , and A ∈ B ( X ),

<!-- formula-not-decoded -->

The standard operator-theoretic notation is used for conditional expectation: for any measurable function f : X → R ,

In a finite state space setting, P n is the n -step transition probability matrix of the Markov chain, and the conditional expectation appears as matrix-vector multiplication:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let c : X → R + denote a cost function, and β ∈ (0 , 1) a discount factor. The discounted-cost value function is defined as h = ∑ ∞ n =0 β n P n c , which is the unique solution to the Bellman equation

<!-- formula-not-decoded -->

1 This deserves more explanation and we plan to provide one in a future version of the paper.

where,

TD-learning algorithms are designed to obtain approximations of h within a finite-dimensional parameterized class.

Consider the case of a d -dimensional linear parameterization. A function ψ : X → R d is chosen, which is viewed as a collection of d basis functions. Each vector θ ∈ R d is associated with the approximate value function h θ = ∑ i θ i ψ i . There are two standard criteria for defining optimality of the parameter. Most natural is the minimum norm approach:

<!-- formula-not-decoded -->

in which the choice of norm is part of the design of the algorithm. Most common is

<!-- formula-not-decoded -->

where the expectation is in steady-state.

In the Galerkin approach, a d -dimensional stationary stochastic process ζ is constructed that is adapted to a stationary realization of X . An algorithm is designed to obtain the vector θ ∗ ∈ R d that satisfies

<!-- formula-not-decoded -->

in which the expectation is again in steady state. The d -dimensional stochastic process ζ is called the sequence of eligibility vectors .

The motivation for the first criterion (27) is clear, but algorithms that solve this problem often suffer from high variance. The Galerkin approach is used because it is simple and generally applicable. Also, if the basis functions are chosen such that h = h θ · for some θ · ∈ R d , and if the solution to (29) is unique, then the Galerkin approach will yield the exact solution h .

The goal of the TD( λ ) learning algorithm is to solve the Galerkin relaxation (29) in which the eligibility vectors are obtained by passing { ψ ( X n ) } through the corresponding first-order low-pass filter: ζ n +1 = λβζ n + ψ ( X n +1 ), n ≥ 0. It is always assumed that λ ∈ [0 , 1]. It is shown in [33] that the solutions to the Galerkin fixed point equation (29) and the minimum norm problem (27) coincide if λ = 1, with the norm defined by (28).

TD( λ ) algorithm: For initialization θ 0 , ζ 0 ∈ R d , the sequence of estimates are defined recursively:

<!-- formula-not-decoded -->

The recursion (30) can be placed in the form (6) in which Φ n = ( X n , X n -1 , ζ n -1 ), and

<!-- formula-not-decoded -->

Based on this representation, it can be shown that the TD( λ ) algorithm is consistent provided the basis vectors are linearly independent, in the sense that E glyph[pi1] [ ψ ( X n ) ψ ( X n ) T ] &gt; 0.

It is also easy to construct an example for which the asymptotic covariance is infinite: Take any consistent example, and scale the basis vectors by a small constant ε . Using the basis εψ , the resulting matrix A is scaled by ε 2 . Hence, for sufficiently small ε &gt; 0, each eigenvalue of A will have real part that is strictly greater than -1 / 2 .

An application of the SNR matrix gain algorithm (15) results in an algorithm with optimal asymptotic covariance. This results in the coupled recursions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α n ≡ 1 /n , for n ≥ 1.

<!-- formula-not-decoded -->

The following proposition follows directly from Prop. 2.2:

Proposition 2.3. Suppose that ̂ A n is invertible for all n ≥ 1 . Then, the sequence of parameters obtained using the SNR-TD( λ ) algorithm (32,33) coincides with the direct estimates:

<!-- formula-not-decoded -->

where E 1 = ̂ A IC -A 1 , ̂ A IC denoting the matrix ̂ A 1 in (32,33), and the sequence of vectors { ζ n } are again defined by ζ n +1 = λβζ n + ψ ( X n +1 ) . glyph[intersectionsq] glyph[unionsq]

It is a remarkable fact that this algorithm is essentially equivalent to the LSTD( λ ) algorithm of [8, 7, 22]: The LSTD( λ ) algorithm is defined to be (34) with E 1 = 0.

## 3 Q-Learning

The class of algorithms considered next is designed for a controlled Markov model, whose input process is denoted U . It is assumed that the state space X and the action space U on which U evolves are both finite. Denote glyph[lscript] = | X | and glyph[lscript] u = | U | .

## 3.1 Notation and assumptions

It is convenient to maintain the operator-theoretic notation used in the uncontrolled setting. There is now a controlled transition matrix that acts on functions h : X → R via

<!-- formula-not-decoded -->

For any non-anticipative input sequence U we have P u h ( x ) = E [ h ( X t +1 ) | X t 0 , U t 0 ] on the event X t = x and U t = u .

There is a finite number of deterministic stationary policies that are enumerated as { φ ( i ) : 1 ≤ i ≤ glyph[lscript] φ } , with glyph[lscript] φ = ( glyph[lscript] u ) glyph[lscript] . A randomized stationary policy is defined by a pmf µ on the integers { 1 ≤ i ≤ glyph[lscript] φ } and such that for each t ,

<!-- formula-not-decoded -->

where { ι ( t ) } is an i.i.d. sequence on { 0 , 1 } glyph[lscript] φ satisfying ∑ k ι k ( t ) = 1, and P { ι k ( t ) = 1 | X t 0 } = µ ( k ) for all k and t .

For any deterministic stationary policy φ , let S φ denote the substitution operator, defined for any function q : X × U → R by S φ q ( x ) = q ( x, φ ( x )). If the policy φ is randomized, of the form (35), then we denote

<!-- formula-not-decoded -->

With P viewed as a single matrix with glyph[lscript] · glyph[lscript] u rows and glyph[lscript] columns, and S φ viewed as a matrix with glyph[lscript] rows and glyph[lscript] · glyph[lscript] u columns, the following interpretations hold:

Lemma 3.1. Suppose that U is defined using a stationary policy φ (possibly randomized). Then, both X and the pair process ( X , U ) are Markovian, and

- (i) P φ := S φ P is the transition matrix for X .
- (ii) PS φ is the transition matrix for ( X , U ) . glyph[intersectionsq] glyph[unionsq]

A cost function c : X × U → R is given together with a discount factor β ∈ (0 , 1). For any (possibly randomized) stationary policy φ , the resulting value function is denoted

<!-- formula-not-decoded -->

The minimal value function is denoted h ∗ , which is the unique solution to the discounted-cost optimality equation (DCOE):

<!-- formula-not-decoded -->

The minimizer defines a stationary policy φ ∗ : X → U that is optimal over all input sequences [4].

The associated 'Q-function' is defined to be the term within the brackets, Q ∗ ( x, u ) := c ( x, u ) + βP u h ∗ ( x ). The DCOE implies a similar fixed point equation for the Q-function:

<!-- formula-not-decoded -->

in which Q ( x ) := min u Q ( x, u ) for any function Q : X × U → R .

For any function q : X × U → R , let φ q : X → U denote an associated policy satisfying

<!-- formula-not-decoded -->

for each x ∈ X . It is assumed to be specified uniquely as follows:

<!-- formula-not-decoded -->

The fixed point equation (38) becomes

<!-- formula-not-decoded -->

In the analysis that follows it is necessary to consider the Q-function associated with all possible cost functions simultaneously: given any function ς : X × U → R , let Q ( ς ) denote the corresponding solution to the fixed point equation (38), with c replaced by ς . That is, the function q = Q ( ς ) is the solution to the fixed point equation,

<!-- formula-not-decoded -->

For a pmf µ defined on the set of policy indices { 1 ≤ i ≤ glyph[lscript] φ } , denote

<!-- formula-not-decoded -->

so that ∂ Q µ ς is the ' Q -function' obtained with the cost function ς , and the randomized stationary policy defined by µ (see also discussion of the SARSA algorithm following the proof of Lemma 3.6). It follows that the functional Q can be expressed as the minimum over all pmfs µ :

<!-- formula-not-decoded -->

There is a single degenerate pmf that attains the minimum for each ( x, u ) (the optimal stationary policy is deterministic) [4].

Lemma 3.2. The mapping Q is a bijection on the set of real-valued functions on X × U . It is also piecewise linear, concave and monotone.

Proof. The fixed point equation (42) defines the Q-function with respect to the cost function ς . Concavity and monotonicity hold because q = Q ( ς ) as defined in (44) is the minimum of linear, monotone functions. The existence of an inverse q ↦→ ς follows from (42). glyph[intersectionsq] glyph[unionsq]

AGalerkin approach to approximating Q ∗ is formulated as follows: Consider a linear parameterization Q θ ( x, u ) = θ T ψ ( x, u ), with θ ∈ R d and ψ : X × U → R d , and denote Q θ ( x ) = min u Q θ ( x, u ). Obtain a d -dimensional stationary stochastic process ζ that is adapted to ( X , U ), and define θ ∗ to be a solution to

<!-- formula-not-decoded -->

where the expectation is in steady-state.

Similar to TD( λ )-learning, a possible approach to estimate θ ∗ is the following:

Q( λ ) algorithm: For initialization θ 0 , ζ 0 ∈ R d , the sequence of estimates are defined recursively:

<!-- formula-not-decoded -->

The success of this approach has been demonstrated in a few restricted settings, such as optimal stopping problems [37], deterministic models [17], and variations of Watkins algorithm that are discussed next.

## 3.2 Watkins algorithm

The basic Q-learning algorithm of [36, 35] is a particular instance of the Galerkin approach with λ = 0 in (46). The basis functions are taken to be indicator functions:

<!-- formula-not-decoded -->

where { ( x k , u k ) : 1 ≤ k ≤ d } is an enumeration of all state-input pairs. The goal of this approach is to compute the function Q ∗ exactly.

The parameter θ is identified with the estimate Q θ , and hence θ ∈ R d with d = glyph[lscript] · glyph[lscript] u . The basic stochastic approximation algorithm to solve (45) coincides with Watkins algorithm:

<!-- formula-not-decoded -->

Only one entry of the approximation is updated at each time point, corresponding to the previous state-input pair ( X n , U n ) observed.

Assumption Q1: The input is defined by a randomized stationary policy of the form (35). The joint process ( X , U ) is an irreducible Markov chain. That is, it has a unique invariant pmf glyph[pi1] satisfying glyph[pi1] ( x, u ) &gt; 0 for each x, u . glyph[intersectionsq] glyph[unionsq]

Assumption Q2: The optimal policy φ ∗ is unique.

The ODE for stability analysis takes on the following simple form:

glyph[intersectionsq]

glyph[unionsq]

<!-- formula-not-decoded -->

in which q t ( x ) = min u q t ( x, u ) as defined below (38). This ODE is stable under Assumption Q1, which then implies that the parameter estimates converge to Q ∗ a.s. [6].

Under Assumption Q2 there exists ε &gt; 0 such that

<!-- formula-not-decoded -->

This justifies a linearization of the ODE (49), in which q t is replaced by S φ ∗ q t .

Although the algorithm is consistent, it should be clear that the asymptotic covariance of this algorithm is typically infinite.

Theorem 3.3. Suppose that Assumptions Q1 and Q2 hold. Then, the sequence of parameters { θ n } obtained using the Q-learning algorithm (48) converges to Q ∗ a.s.. Suppose moreover that the conditional variance of h ∗ ( X t ) is positive:

<!-- formula-not-decoded -->

and (1 -β ) max x,u glyph[pi1] ( x, u ) ≤ 1 2 . Then, in the case α n ≡ 1 /n ,

<!-- formula-not-decoded -->

The assumption (1 -β ) max x,u glyph[pi1] ( x, u ) ≤ 1 2 is satisfied whenever β ≥ 1 2 .

The proof of convergence can be found in [36, 35]. The proof of infinite asymptotic covariance is given in Section A.2 of the Appendix. An eigenvector for A is constructed with strictly positive entries, and with real eigenvalue satisfying λ ≥ -1 / 2. Interpreted as a function v : X × U → C , this eigenvector satisfies

<!-- formula-not-decoded -->

Assumption (50) ensures that the right hand side is strictly positive, as required in Thm. 2.1 (i). The recursion (48) for the Q-learning algorithm can be written in the form (6) in which

<!-- formula-not-decoded -->

This motivates the introduction of stochastic Newton-Raphson algorithms that are considered next.

## 3.3 SNR and Zap Q-Learning

For a sequence of d × d matrices G = { G n } and λ ∈ [0 , 1], the matrix-gain Q( λ ) algorithm is described as follows:

G -Q( λ ) algorithm: For initialization θ 0 , ζ 0 ∈ R d , the sequence of estimates are defined recursively:

<!-- formula-not-decoded -->

The special case based on stochastic Newton-Raphson (17) is called the Zap-Q( λ ) algorithm:

## Algorithm 1 Zap-Q( λ ) algorithm

| Input: Initial θ 0 ∈ R d , ζ 0 = ψ ( X 0 ,U 0 ), ̂ A 0 ∈ R d × d , n = 0, T ∈ Z   | Input: Initial θ 0 ∈ R d , ζ 0 = ψ ( X 0 ,U 0 ), ̂ A 0 ∈ R d × d , n = 0, T ∈ Z   | glyph[triangleright] Initialization                 |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|-----------------------------------------------------|
| 1: repeat                                                                        | 1: repeat                                                                        | 1: repeat                                           |
| 2:                                                                               | φ X n +1 n := argmin u Q θ n ( X n +1 ,u );                                      |                                                     |
| 3:                                                                               | d n +1 := c ( X n ,U n )+ βQ θ n ( X n +1 ,φ X n +1 n ) - Q θ n ( X n ,U n );    | glyph[triangleright] Temporal difference term       |
| 4:                                                                               | A n +1 := ζ n [ βψ ( X n +1 ,φ X n +1 n ) - ψ ( X n ,U n ) ] T ;                 |                                                     |
| 5:                                                                               | ̂ A n +1 = ̂ A n + γ n +1 [ A n +1 - ̂ A n ] ;                                      | glyph[triangleright] Matrix gain update rule        |
| 6:                                                                               | θ n +1 = θ n - α n +1 ̂ A - 1 n +1 ζ n d n +1 ;                                   | glyph[triangleright] Zap-Q update rule              |
| 7:                                                                               | ζ n +1 := λβζ n + ψ ( X n +1 ,U n +1 );                                          | glyph[triangleright] Eligibility vector update rule |
| 8:                                                                               | n = n +1                                                                         |                                                     |
| 9:                                                                               | until n T                                                                        |                                                     |

≥

It is assumed that a projection is employed to ensure that { ̂ A -1 n } is a bounded sequence - this is most easily achieved using the Matrix Inversion Lemma.

The analysis that follows is specialized to λ = 0 and the basis (47) that is used in Watkins' algorithm. The resulting Zap-Q algorithm is defined as follows, after identifying Q θ and θ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ̂ G ∗ n = -[ ̂ A n ] -1 , and [ · ] denotes a projection, chosen so that { ̂ G ∗ n } is a bounded sequence. In Thm. 3.4 it is established that the projection is required only for a finite number of iterations: { ̂ A -1 n : n ≥ n · } is a bounded sequence, where n · &lt; ∞ a.s..

An equivalent representation for the parameter recursion (53) is

<!-- formula-not-decoded -->

in which c and θ n are treated as d -dimensional vectors rather than functions on X × U , and

<!-- formula-not-decoded -->

It would seem that the analysis is complicated by the fact that the sequence { A n } depends upon { θ n } through the policy sequence { φ n } . Part of the analysis is simplified by obtaining a recursion for the following d -dimensional sequence:

<!-- formula-not-decoded -->

where Π is the d × d diagonal matrix with entries Π( k, k ) := glyph[pi1] ( x k , u k ). This admits a very simple recursion in the special case γ ≡ α . In the other case considered, wherein the step-size sequence γ satisfies (18), the recursion for ̂ C is more complex, but the ODE analysis is simplified.

## 3.4 Main results

Conditions for convergence of the Zap-Q algorithm (53,54) are summarized in Thm. 3.4. The following assumption is used to address the discontinuity in the recursion for { ̂ A n } resulting from the dependence of A n +1 on φ n .

Assumption Q3: The sequence of policies { φ n } satisfies:

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[intersectionsq] glyph[unionsq]

Theorem 3.4. Suppose that Assumptions Q1-Q3 hold, with the gain sequences α and γ satisfying

<!-- formula-not-decoded -->

for some fixed ρ ∈ ( 1 2 , 1) . Then,

- (i) The parameter sequence { θ n } obtained using the Zap-Q algorithm (53,54) converges to Q ∗ a.s..
- (ii) The asymptotic covariance (2) is minimized over all G -Q( 0 ) matrix gain versions of Watkins' Q-learning algorithm.
- (iii) An ODE approximation holds for the sequence { θ n , ̂ C n } , by continuous functions ( q , c ) satisfying

<!-- formula-not-decoded -->

This ODE approximation is exponentially asymptotically stable, with lim t →∞ q t = Q ∗ . glyph[intersectionsq] glyph[unionsq]

See Section 3.6.2 and standard references such as [5] for the precise meaning of the ODE approximation (60).

Proof of Thm. 3.4. Boundedness of the sequences { θ n , ̂ A n : n ≥ 0 } and { ̂ A -1 n : n ≥ n · } is established in Lemmas A.3 and A.6, where n · &lt; ∞ a.s.. The ODE approximation is established in Prop. A.7. These two results combined with standard arguments establishes (i) [5].

Result (ii) follows from convergence of the algorithm, just as in the case of TD-learning. Uniqueness of the optimal policy is needed so that the recursion for { θ n } admits a linearization around Q ∗ . glyph[intersectionsq] glyph[unionsq]

In the case γ ≡ α , the three consequences hold under a stronger assumption than Q3:

Proposition 3.5. Suppose that Assumptions Q1-Q2 hold, γ ≡ α , and the sequence of policies { φ n } is convergent. Then, the parameter sequence { θ n } obtained using the Zap-Q algorithm (53,54) converges to Q ∗ a.s..

glyph[intersectionsq] glyph[unionsq]

The convergence assumption in Prop. 3.5 is far stronger than Q3: Recall that the policies { φ n } evolve in a finite set { φ ( i ) : 1 ≤ i ≤ glyph[lscript] φ } . Convergence means that φ n = φ ( ˜ k ) for some integer-valued random variable ˜ k , and all n sufficiently large.

The proof of Prop. 3.5 is based on a simple inverse dynamic programming argument: it is easily shown that ̂ C n is convergent to c in the case γ ≡ α , and it is also easily established that lim n →∞ θ n - Q ( ̂ C n ) = 0 in this case. The proof of Thm. 3.4 is more delicate, and is based on extensions of ODE arguments in [6].

The simplicity of the proof of Prop. 3.5 suggests that this case would be preferred. However, when γ n ≡ α n = 1 /n we do not know how to relax the assumption that { φ n } is convergent. Analysis is complicated by the fact that ̂ A n is obtained as a uniform average of { A n } .

The ODE analysis in the proof of Thm. 3.4 suggests that the dynamics of the two time-scale algorithm closely matches the Newton-Raphson ideal. Moreover, the two time-scale algorithm has the best performance in all of the numerical experiments surveyed in Section 4.

## 3.5 ODE and Policy Iteration

Recall the definition of ∂ Q µ in (43). The ODE approximation (60) can be expressed

<!-- formula-not-decoded -->

where µ t is any pmf satisfying ∂ Q µ t c t = q t , and the derivative exists for a.e. t (see Lemma A.10 for full justification). This has an interesting geometric interpretation. Without loss of generality, assume that the cost function is non-negative, so that q evolves in the positive orthant R d + whenever its initial condition lies in this domain.

Figure 1: ODE for SNR2 Q-Learning. The light arrows show typical vectors in the vector field that defines the ODE (61). The solution starting at q 0 ∈ Θ 3 initially moves in a straight line towards Q 3 .

<!-- image -->

A typical solution to the ODE is shown in Fig. 1: the trajectory is piecewise linear, with changes in direction corresponding to changes in the policy φ q t . Each set Θ k shown in the figure corresponds to a deterministic policy:

<!-- formula-not-decoded -->

Lemma 3.6. For each k the set Θ k is a convex polyhedron, and also a positive cone. When q t ∈ interior (Θ k ) then

<!-- formula-not-decoded -->

Proof. The power series expansion holds:

<!-- formula-not-decoded -->

For each n ≥ 1 we have [ PS φ ] n = PP n -1 φ S φ , which together with (36) implies the desired result. glyph[intersectionsq] glyph[unionsq]

The function Q k is the fixed-policy Q-function considered in the SARSA algorithm [18, 26, 32]. While q t evolves in the interior of the set Θ k , it moves in a straight line towards the function Q k . On reaching the boundary, it then moves in a straight line to the next Q-function. This is something like a policy iteration recursion, since the policy φ q t is obtained as the argmin over u of q t ( · , u ).

Of course, it is far easier to establish stability of the equivalent ODE (60).

## 3.6 Overview of proofs

This final subsection is dedicated to the proof of Prop. 3.5, and the main ideas in the proof of Thm. 3.4. It is assumed throughout the remainder of this section that Assumptions Q1-Q3 hold. Proofs of technical lemmas are contained in Appendix A.3.

We require the usual probabilistic foundations: There is a probability space (Ω , F , P ) that supports all random variables under consideration. The probability measure P may depend on an initialization of the Markov chain. All stochastic processes under consideration are assumed adapted to a filtration denoted {F n : n ≥ 0 } .

We begin with the proof of the simpler Prop. 3.5.

## 3.6.1 Inverse Dynamic Programming Analysis

Prop. 3.5 is a quick consequence of the following extension of Prop. 2.2:

Proposition 3.7. Suppose that Assumptions Q1-Q3 hold. Suppose moreover that each of the matrices { ̂ A n : n ≥ n · } is invertible for some n · ≥ 1 that is a.s. finite. Then, the following recursion holds for n ≥ n · :

<!-- formula-not-decoded -->

glyph[intersectionsq] glyph[unionsq]

where Ψ n is defined in (56) .

Proof of Prop. 3.5. The assumption that the sequence of policies { φ n } converges to a (possibly random) limit φ ∞ has the following consequences: First, this implies that ̂ A n defined in (54) converges:

<!-- formula-not-decoded -->

Second, for all n sufficiently large the following identities hold, by applying the definitions of φ n and Q -1 :

<!-- formula-not-decoded -->

From (63), since the limit on the right hand side is invertible and the set of all invertible matrices is open, it follows that there is an integer n · that is finite a.s., and such that ̂ A n is invertible for n ≥ n · .

Now applying Prop. 3.7, the recursion (62) is reduced to the following in the case that γ ≡ α :

<!-- formula-not-decoded -->

This is essentially a Monte-Carlo average of { Π -1 Ψ n c : n ≥ 0 } . Since the steady state expectation of Ψ n is equal to Π, convergence follows from the Law of Large Numbers:

<!-- formula-not-decoded -->

Combining equations (63), (64), and (65) implies

<!-- formula-not-decoded -->

Lemma 3.2 completes the proof: lim n →∞ θ n = Q ( c ) = Q

## 3.6.2 ODE Analysis

The remainder of this section is devoted to a high-level view of the proof of the ODE approximation for the two time-scale algorithm, with α and γ defined in (59).

The construction of an approximating ODE involves first defining a continuous time process. Denote

<!-- formula-not-decoded -->

and define ¯ q t n = θ n for these values, with the definition extended to R + via linear interpolation. We say that the ODE approximation d dt q = f ( q ) holds if we have the approximation,

<!-- formula-not-decoded -->

where the error process satisfies, for each T &gt; 0,

<!-- formula-not-decoded -->

Such approximations will be represented using the more compact notation:

<!-- formula-not-decoded -->

∗ . glyph[intersectionsq] glyph[unionsq]

An ODE approximation holds for Watkins algorithm, with f ( q t ) defined by the right hand side of (49), or in more compact notation:

<!-- formula-not-decoded -->

The significance of this representation is that q t is the Q-function associated with the 'cost function' c t : q t = Q ( c t ).

To construct an ODE, it is convenient first to obtain an alternative and suggestive representation for the pair of equations (53,54). A vector-valued sequence of random variables {E k } will be called ODE-friendly if it admits the decomposition,

The same notation will be used in the following treatment of Zap Q-learning. Along with the piecewise linear continuous-time process { ¯ q t : t ≥ 0 } , denote by { ¯ A t : t ≥ 0 } the piecewise linear continuous-time process defined similarly, with ¯ A t n = ̂ A n , n ≥ 1, and ¯ c t = Q -1 (¯ q t ) for t ≥ 0.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

in which { ∆ k : k ≥ 1 } is a martingale-difference sequence satisfying E [ ‖ ∆ k +1 ‖ 2 | F k ] ≤ ¯ σ 2 ∆ a.s. for some finite ¯ σ 2 ∆ and all k , {T k : k ≥ 1 } is a bounded sequence, and the final sequence is bounded and satisfies

Lemma 3.8. The pair of equations (53,54) can be expressed,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

in which the sequence {E q n : n ≥ 1 } is ODE-friendly. The sequence {E A n } is ODE-friendly provided Assumption Q3 holds. glyph[intersectionsq] glyph[unionsq]

<!-- formula-not-decoded -->

The assertion that {E q n , E A n } are ODE-friendly follows from standard arguments based on solutions to Poisson's equation for zero-mean functions of the Markov chain ( X , U ) [29]. The proof of Lemma 3.9 is based on an extension of this technique to the present setting.

Lemma 3.9. For each n ≥ 0 , where { ∆ Ψ k } is a martingale difference sequence with uniformly bounded second moment, and the sequences {T k , ε k : k ≥ 0 } are also bounded. If Assumption Q3 holds then { ε k } satisfies (70) . glyph[intersectionsq] glyph[unionsq]

<!-- formula-not-decoded -->

The representation in Lemma 3.8 appears similar to an Euler approximation of the solution to an ODE:

<!-- formula-not-decoded -->

It is discontinuity of the function f A that presents the most significant challenge in analysis of the algorithm - this violates standard conditions for existence and uniqueness of solutions to the ODE without disturbance.

Fortunately there is special structure that will allow the construction of an ODE approximation. Some of this structure is highlighted in the lemma that follows. These approximations are taken from Lemmas A.3 and A.6.

Lemma 3.10. For each t, T 0 ≥ 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where g t := γ n /α n when t = t n for some n , and extended to all t ∈ R + by linear interpolation. glyph[intersectionsq] glyph[unionsq]

The 'gain' g t appearing in (73) converges to infinity rapidly as t →∞ : Based on the definitions in (59), it follows from (66) that t n ≈ log( n ) for large n , and consequently g t ≈ exp((1 -ρ ) t ) for large t . This suggests that the integrand Π[ I -βPS φ ¯ q t ] + ¯ A t should converge to zero rapidly with t . This intuition is made precise in the Appendix. Through several subsequent transformations, these integral equations are shown to imply the ODE approximation in Thm. 3.4.

## 3.7 An O ( d ) Zap-Q learning algorithm

In this subsection, we introduce an O ( d ) Zap-Q learning algorithm, which is basically the O ( d ) Zap-SNR algorithm described in Section 2.3.2 specialized to Q-learning.

Based on the equations (24), (25), (53), and (54), the algorithm is defined as follows: Fix N = d , and for i ≥ 0, where,

<!-- formula-not-decoded -->

̂ G ∗ ( i +1) N = -[ ̂ A ( i +1) N ] -1 , with [ · ] denoting a projection, chosen so that { ̂ G ∗ ( i +1) N } is a bounded sequence. For a given parameter vector θ , the policy φ θ is defined in (40).

Once again, we claim that the asymptotic properties of the above defined O ( d ) Zap-Q learning algorithm is the same as that of the Zap-Q learning algorithm defined in (53) and (54), with justification postponed to a future version of the paper.

## 4 Numerical Results

Results from numerical experiments are surveyed here to illustrate the performance of the Zap Q-learning algorithm (53,54). Comparisons are made with several existing algorithms, including Watkins Q-learning (48), Watkins Q-learning with Ruppert-Polyak-Juditsky (RPJ) averaging [28, 24, 25], Watkins Q-learning with a 'polynomial learning rate' [10], and the more recent Speedy Q-learning algorithm [1].

<!-- formula-not-decoded -->

In addition, the Watkins algorithm with a scalar gain g is considered, with g chosen so that the algorithm has finite asymptotic covariance. When the value of g is optimized and numerical conditions are favorable (e.g., the condition number of A is not too large) it is found that the performance is nearly as good as the Zap-Q algorithm. However, there is no free lunch:

- (i) Design of the scalar gain g depends on approximation of A , and hence θ ∗ . While it is possible to estimate A via Monte-Carlo in Zap Q-learning, it is not known how to efficiently update approximations for an optimal scalar gain.
- (ii) A reasonable asymptotic covariance required a large value of g . Consequently, the scalar gain algorithm had massive transients, resulting in a poor performance in practice.
- (iii) Transient behavior could be tamed through projection to a bounded set. However, this again requires prior knowledge of the region in the parameter space to which θ ∗ belongs.

Projection of parameters was also necessary for RPJ averaging.

The following batch mean method was used to estimate the asymptotic covariance.

Batch Mean Method At stage n of the algorithm we will be interested in the distribution of a vector-valued random variable of the form f n ( θ n ), where f n : R d → R m is possibly dependent on n . The batch mean method is used to estimate its statistics: For each algorithm, N parallel simulations are run with θ 0 initialized i.i.d. according to some distribution. Denoting θ i n to be the vector θ n corresponding to the i th simulation, the distribution of the random variable f n ( θ n ) is estimated based on the histogram of the independent samples { f n ( θ i n ) : 1 ≤ i ≤ N } .

An important special case in this paper is f n ( θ n ) = W n := √ n ( θ n -θ ∗ ). However, since the limit θ ∗ is not available, the empirical mean is substituted:

<!-- formula-not-decoded -->

The estimate of the covariance of f n ( θ n ) is then obtained as the sample covariance of { W i n , 1 ≤ i ≤ N } . This corresponds to the estimate of the asymptotic covariance Σ θ defined in (2).

The value N = 10 3 is used in all of the experiments surveyed here.

## 4.1 Finite state-action MDP

Figure 2: Graph for MDP

<!-- image -->

Consider first a simple stochastic-shortest-path problem. The state space X = { 1 , . . . , 6 } coincides with the six nodes on the un-directed graph shown in Fig. 2. The action space U = { e x,x ′ } , x, x ′ ∈ X , consists of all feasible edges along which an agent can travel, including each 'self-loop', u = e x,x . The number of state-action pairs for this example coincides with the number of nodes plus twice the number of edges: d = 18.

The controlled transition matrix is defined as follows: If X n = x ∈ X , and U n = e x,x ′ ∈ U , then X n +1 = x ′ with probability 0 . 8, and with probability 0 . 2, the next state is randomly chosen between all neighboring nodes. The goal is to reach the state x ∗ = 6 and maximize the time spent there. This is modeled through a discounted-reward optimality criterion with discount factor β ∈ (0 , 1). The one-step reward is defined

as follows:

glyph[negationslash]

<!-- formula-not-decoded -->

The solution to the discounted-cost optimal control problem can be computed numerically for this model; the optimal policy is unique and independent of β .

Six different variants of Q-learning were tested:

1. Watkins' algorithm with scalar gain g , so that α n ≡ g/n
2. Watkins' algorithm using RPJ averaging, with γ n ≡ ( α n ) 0 . 6 ≡ n -0 . 6
3. Watkins' algorithm with the polynomial learning rate α n ≡ n -0 . 6
4. Speedy Q-learning
5. Zap Q-learning with α ≡ γ
6. Zap Q-learning with γ n ≡ ( α n ) 0 . 85 ≡ n -0 . 85

The basis was taken to be the same as in Watkins Q-learning algorithm. In each case, the randomized policy was taken to be uniform: feasible transitions were sampled uniformly at each time.

Discount factors β = 0 . 8 and β = 0 . 99 were considered. In each case, the unique optimal parameter θ ∗ = Q ∗ was obtained numerically.

Asymptotic Covariance Speedy Q-learning cannot be represented as a standard stochastic approximation, so standard theory cannot be applied to obtain its asymptotic covariance. The Watkins' algorithm with polynomial learning rate has infinite asymptotic covariance.

For the other four algorithms, the asymptotic covariance Σ θ was computed by solving the Lyapunov equation (13) based on the matrix gain G that is particular to each algorithm. Recall that G = -A -1 in the case of either of the Zap-Q algorithms.

The matrices A and Σ ∆ appearing in (13) are defined with respect to Watkins' Q-learning algorithm with α n = 1 /n . The first matrix is A = -Π[ I -βPS φ ∗ ] under the standing assumption that the optimal policy is unique. The proof that this is a linearization comes first from the representation of the ODE approximation (49) in vector form:

<!-- formula-not-decoded -->

Uniqueness of the optimal policy implies that f is locally linear: there exists ε &gt; 0 such that

<!-- formula-not-decoded -->

The matrix Σ ∆ was also obtained numerically, without resorting to simulation.

The eigenvalues of the 18 × 18 matrix A are real in this example, as shown in Fig. 3 for both values of β . To ensure that the eigenvalues of gA are all strictly less than -1 / 2 in a scalar gain algorithm requires the (approximate) lower bounds g &gt; 45 for β = 0 . 8, and g &gt; 900 for β = 0 . 99. Thm. 2.1 implies that the asymptotic covariance Σ θ ( g ) is finite for this range of g in the Watkins algorithm with α n ≡ g/n . Fig. 4 shows the normalized trace of the asymptotic covariance as a function of g &gt; 0, and the significance of g ≈ 45 and g ≈ 900.

Figure 3: Eigenvalues of the matrix A for the 6-state example

<!-- image -->

Figure 4: The normalized trace of the asymptotic covariance for the scaled Watkins algorithm with different scalar gains g , for the 6-state example: σ 2 ( g ) = trace (Σ θ ( g )) and σ 2 ( G ∗ ) = trace (Σ ∗ ).

<!-- image -->

Based on this analysis or on Thm. 3.3, it follows that the asymptotic covariance is not finite for the standard Watkins' algorithm with α n ≡ 1 /n . In simulations it was found that the parameter estimates are not close to θ ∗ even after many millions of samples. This is illustrated for the case β = 0 . 8 in Fig. 5, which shows a histogram of 10 3 estimates of θ n (15) with n = 10 6 (other entries showed similar behavior).

It was found that the algorithm performed very poorly in practice for any scalar gain algorithm. For example, more than half of the 10 3 experiments using β = 0 . 8 and g = 70 resulted in values of θ n (15) exceeding θ ∗ (15) by 10 4 (with θ ∗ (15) ≈ 500), even with n = 10 6 . The algorithm performed well with the introduction of projection in the case β = 0 . 8. With β = 0 . 99, the performance was unacceptable for any scalar gain, even with projection.

The results presented next used a gain of g = 70 in the case β = 0 . 8, and projection of each entry of the estimates to the interval ( -∞ , 1000]. Fig. 6 shows normalized histograms of { W i n ( k ) : 1 ≤ i ≤ N } , as defined in (76), with k = 10 , 18.

The Central Limit Theorem holds: W n is expected to be approximately normally distributed: N (0 , Σ θ ( g )), when n is large. Of the d = 18 entries of the vector W n , with n ≥ 10 4 , it was found that the asymptotic variance matched the histogram nearly perfectly for k = 10, while k = 18 showed the worst fit.

These experiments were repeated for each of the Zap-Q algorithms, for which the asymptotic variance Σ ∗ is obtained using the formula (14). Plots are shown only for Case 2: the two time-scale algorithm, with γ n = ( α n ) 0 . 85 . Histograms in the case of β = 0 . 8 are shown in Fig. 7, and Fig. 8

Figure 5: Histogram of 10 3 estimates of θ n (15), with n = 10 6 for the Watkins algorithm applied to the 6-state example with discount factor β = 0 . 8

<!-- image -->

Figure 6: Comparison of theoretical and empirical asymptotic variance for the scaled Watkins' algorithm, with gain g = 70, applied to the 6-state example with discount factor β = 0 . 8

<!-- image -->

for β = 0 . 99. The covariance estimates and the Gaussian approximations match the theoretical predictions remarkably well for n ≥ 10 4 .

Figure 7: Comparison of theoretical and empirical asymptotic variance of the two time-scale Zap-Q algorithm applied to the 6-state example; β = 0 . 8

<!-- image -->

Bellman Error The Bellman error at iteration n is denoted:

<!-- formula-not-decoded -->

This is identically zero if and only if θ n = Q ∗ . If { θ n } converges to Q ∗ then B n = ˜ θ n -βPS φ ∗ ˜ θ n for all sufficiently large n , and the CLT holds for {B n } whenever it holds for { θ n } . Moreover, on

Figure 8: Comparison of theoretical and empirical asymptotic variance of the Zap-Q-learning algorithm applied to the 6-state example; β = 0 . 99

<!-- image -->

Figure 9: Maximum Bellman error {B n : n ≥ 0 } for the six Q-learning algorithms

<!-- image -->

denoting the maximal error the sequence { √ n B n } also converges in distribution as n →∞ . Fig. 9 contains plots of {B n } for the six different Q-learning algorithms.

<!-- formula-not-decoded -->

For large n , the two versions of Zap Q-learning exhibit similar behavior since ̂ A n converges to A in both algorithms. Though all six algorithms perform reasonably well when β = 0 . 8, Zap Q-learning is the only one that achieves near zero Bellman error within n = 10 6 iterations in the case β = 0 . 99. Moreover, the performance of the two time-scale algorithm is clearly superior to the one time-scale algorithm.

Fig. 9 shows only the typical behavior - repeated trails were run to investigate the range of possible outcomes. For each algorithm, the outcomes of N = 1000 independent simulations resulted in samples {B i n , 1 ≤ i ≤ N } , with θ 0 uniformly distributed on the interval [ -10 3 , 10 3 ] for β = 0 . 8 and [ -10 4 , 10 4 ] for β = 0 . 99.

Fig. 12 and Fig. 13 shows histograms of {B i n , 1 ≤ i ≤ N } , n = 10 6 , for all the six algorithms; this corresponds to the data shown in Fig. 10 and Fig. 11 at n = 10 6 .

The batch means method was used to obtain estimates of the mean and variance of B n for a range of values of n . Plots of the mean and 2 σ confidence intervals are shown in Fig. 10 for the case β = 0 . 8, and plots for β = 0 . 99 are shown in Fig. 11.

<!-- image -->

Figure 10: Simulation-based 2 σ confidence intervals for the six Q-learning algorithms with discount factor β = 0 . 8.

Figure 11: Simulation-based 2 σ confidence intervals for the six Q-learning algorithms with discount factor β = 0 . 99.

<!-- image -->

Figure 12: Histogram of the maximal Bellman error when discount factor β = 0 . 8 and number of iterations n = 10 6 .

<!-- image -->

Figure 13: Histogram of the maximal Bellman error when discount factor β = 0 . 99 and number of iterations n = 10 6 .

<!-- image -->

## 4.1.1 Performance of the O ( d ) Zap-Q learning algorithm

In this subsection, we test the performance of the O ( d ) Zap-Q learning algorithm that was defined in equations (74) and (75) of Section 3.7 by applying it to the stochastic shortest path problem. We restrict to the comparison of the Bellman errors (defined in (78)) of the different algorithms, and we consider the case β = 0 . 99.

Fig. 14 contains plots of {B n } for the different Q-learning algorithms. For the O ( d ) Zap-Q learning algorithms, the batch size was set to N = 100 ( d = 18 in this problem). We notice in the figure that the O ( d ) algorithm performs nearly as well as the O ( d 2 ) algorithm when the step-sizes (ˆ γ i ) are chosen appropriately. Furthermore, the naive batching technique applied to a single-time-scale Stochastic Newton-Raphson algorithm ( γ n ≡ α n and therefore ˆ γ i ≈ α i ) performs extremely poorly.

<!-- image -->

10

Figure 14: Maximum Bellman error {B n : n ≥ 0 } for different Q-learning algorithms

## 4.2 Finance model

The next example is taken from [34, 9]. The reader is referred to these references for complete details of the problem set-up and the reinforcement learning architecture used in this prior work. The example is of interest because it shows how the Zap Q-learning algorithm can be used with a more general basis, and also how the technique can be extended to optimal stopping time problems.

The Markovian state process considered in [34, 9] is the vector of ratios:

<!-- formula-not-decoded -->

in which { ˜ p t : t ∈ R } is a geometric Brownian motion (derived from an exogenous price-process). This uncontrolled Markov chain is positive Harris recurrent on the state space X ≡ R 100 [20].

The 'time to exercise' is modeled as a stopping time τ ∈ Z + . The associated expected reward is defined as E [ β τ r ( X τ )], with r ( X n ) := X n (100) = ˜ p n / ˜ p n -100 and β ∈ (0 , 1) fixed. The objective of finding a policy that maximizes the expected reward is modeled as an optimal stopping time problem.

The value function is defined to be the supremum over all stopping times:

<!-- formula-not-decoded -->

This solves the Bellman equation:

<!-- formula-not-decoded -->

The associated Q-function is denoted Q ∗ ( x ) := β E [ h ∗ ( X n +1 ) | X n = x ], which solves a similar fixed point equation:

<!-- formula-not-decoded -->

A stationary policy φ : X →{ 0 , 1 } assigns an action for each state x ∈ X as

<!-- formula-not-decoded -->

Each policy φ defines a stopping time and associated average reward, denoted

<!-- formula-not-decoded -->

The optimal policy is expressed as

<!-- formula-not-decoded -->

The corresponding optimal stopping time that solves the supremum in (79) is achieved using this policy: τ ∗ = min { n : φ ∗ ( X n ) = 1 } [34].

The objective here is to find an approximation for Q ∗ in a parameterized class { Q θ := θ T ψ : θ ∈ R d } , where ψ : X → R d is a vector of basis functions. For a fixed parameter vector θ , the associated value function is denoted

<!-- formula-not-decoded -->

The function h φ θ was estimated using Monte-Carlo in the numerical experiments surveyed below.

Approximations to the Optimal Stopping Time Problem To obtain the optimal parameter vector θ ∗ , in [34] the authors apply the Q(0)-learning algorithm:

<!-- formula-not-decoded -->

This is one of the few parameterized Q-learning settings for which convergence is guaranteed [34]. In [9] the authors attempt to improve the performance of the Q(0) algorithm through the use of the sequence of matrix gains and a special choice for the { α n } :

<!-- formula-not-decoded -->

where g and b are positive constants. The resulting recursion is the G -Q(0) algorithm:

<!-- formula-not-decoded -->

Through trial and error the authors find that g = 10 2 , b = 10 4 gives good performance. These values were also used in the experiments described in the following.

The limiting matrix gain is given by

<!-- formula-not-decoded -->

where the expectation is in steady-state. The asymptotic covariance Σ G θ is the unique positive semidefinite solution to the Lyapunov equation (13), provided all eigenvalues of GA satisfy Re( λ ) &lt; -1 2 . The Zap Q-learning algorithm for this example is defined by the following recursion:

<!-- formula-not-decoded -->

It is conjectured that the asymptotic covariance Σ ∗ is obtained using (14), where the matrix A is the limit of ̂ A n :

<!-- formula-not-decoded -->

Figure 15: Eigenvalues of A and GA for the finance example

<!-- image -->

Experimental Results The experimental setting of [34, 9] is used to define the set of basis functions and other parameters. The dimension of the parameter vector d was chosen to be 10, with the basis functions defined in [9]. The objective here is to compare the performances of G -Q(0) and the Zap-Q algorithms in terms of both parameter convergence, and with respect to the resulting average reward (81).

The asymptotic covariance matrices Σ ∗ and Σ G θ were estimated through the following steps: The matrices A and G were estimated via Monte-Carlo. Estimation of A requires an estimate of θ ∗ ; this was taken to be θ n , with n = 2 × 10 6 , obtained using the Zap-Q two timescale algorithm with α n ≡ 1 /n and γ n ≡ α 0 . 85 n . This estimate of θ ∗ was also used to estimate the covariance matrix Σ ∆ defined in (10) using the batch means method. The matrices Σ G θ and Σ ∗ were then obtained using (13) and (14), respectively.

It was found that the trace of Σ G θ was about 15 times greater than that of Σ ∗ .

High performance despite ill-conditioned matrix gain The real part of the eigenvalues of A are shown on a logarithmic scale on the left-hand side of Fig. 15. The eigenvalues of the matrix A have a wide spread: The condition-number is of the order 10 4 . This presents a challenge in applying any method. In particular, it was found that the performance of any scalar-gain algorithm was extremely poor, even with projection of parameter estimates.

This is a consequence of the fact that the basis functions { ψ i } are nearly linearly dependent. A better basis should be considered in future work, but the main objective here is to test the new methods in a challenging setting, and to compare with prior approaches.

<!-- image -->

×

×

×

×

Figure 16: Theoretical and empirical variance for the finance example

In applying the Zap Q-learning algorithm it was found that the estimates { ̂ A n } in (84) are nearly singular. Despite the unfavorable setting for this approach, the performance of the algorithm was much better than any alternative that was tested. The upper row of Fig. 16 contains normalized histograms of { W i n ( k ) = √ n ( θ i n ( k ) -θ n ( k )) : 1 ≤ i ≤ N } for the Zap-Q algorithm. The variance for finite n is close to the theoretical predictions based on the asymptotic covariance Σ ∗ . The histograms were generated for two values of n , and k = 1 , 7. Of the d = 10 possibilities, the histogram for k = 1 had the worst match with theoretical predictions, and k = 7 was the closest.

The eigenvalues corresponding to the matrix GA are shown on the right hand side of Fig. 15.

It is found that one of these eigenvalues is very close to -0 . 5, and the sufficient condition for trace (Σ G θ ) &lt; ∞ is barely satisfied. It is worth stressing that the finite asymptotic covariance was not a design goal in this prior work. It is only now on revisiting this paper that we find that the sufficient condition λ &lt; -1 2 is satisfied.

The lower row of Fig. 16 contains the normalized histograms of { W i n ( k ) = √ n ( θ i n ( k ) -θ n ( k )) : 1 ≤ i ≤ N } for the G -Q(0) algorithm for n = 2 × 10 4 and 2 × 10 6 , and k = 1 , 7, along with the theoretical predictions based on the asymptotic covariance Σ G θ .

Figure 17: Histograms of the average reward obtained using the G -Q(0) learning and the Zap-Q-learning, γ n ≡ α -ρ n ≡ n -ρ

<!-- image -->

Asymptotic variance of the discounted reward Denote h n = h φ , with φ = φ θ n . Histograms of the average reward h n ( x ) were obtained for x ( i ) = 1, 1 ≤ i ≤ 100, and various values of n , based on N = 1000 independent simulations. The plots shown in Fig. 17 are based on n = 2 × 10 k , for k = 4 , 5 , 6. Omitted in this figure are outliers : values of the reward in the interval [0 , 1). Table 1 lists the number of outliers for each n and each algorithm.

Recall that the asymptotic covariance of the G -Q(0) algorithm was not far from optimal (its trace was about 15 times larger than obtained using Zap Q-learning). However, it is observed that this algorithm suffers from much larger outliers. It can also be seen that doubling the scalar gain g (causing the largest eigenvalue of GA to be ≈ -1) results in slightly better performance.

## 5 Conclusions

Watkins' Q-learning algorithm is elegant, but subject to two common and valid constraints: it can be very slow to converge, and it is not obvious how to extend this approach to obtain a stable algorithm in non-trivial parameterized settings. This paper addresses both concerns with the new Zap Q( λ ) algorithms that are motivated by asymptotic theory of stochastic approximation.

There are many avenues for future research. It would be valuable to find an alternative to Assumption Q3 that is readily verified. Based on the ODE analysis, it seems likely that the conclusions of Thm. 3.4 hold without this additional assumption. No theory has been presented here for non-ideal parameterized settings. It is conjectured that conditions for stability of Zap Q( λ )-learning will hold under general conditions. Consistency is a more challenging problem and is a focus of current research.

In terms of algorithm design, it is remarkable to see how well the scalar-gain algorithms perform, provided projection is employed and the condition number of A is not too large. It is possible to estimate the optimal scalar gain based on estimates of the matrix A that is central to this paper. How to do so without introducing high complexity is an open question.

On the other hand, the performance of RPJ averaging is unpredictable. In many experiments it is found that the asymptotic covariance is a poor indicator of finiten performance when using this

Table 1: Outliers observed in N = 1000 runs. Each table represents the number of runs which resulted in an average reward below a certain value

| n                 | 2 × 10 4          | 2 × 10 5          | 2 × 10 6          | n                      | 2 × 10 4               | 2 × 10 5               | 2 × 10 6               |
|-------------------|-------------------|-------------------|-------------------|------------------------|------------------------|------------------------|------------------------|
| G -Q(0) g = 100   | 827               | 775               | 680               | G -Q(0) g = 100        | 811                    | 755                    | 654                    |
| G -Q(0) g = 200   | 824               | 725               | 559               | G -Q(0) g = 200        | 806                    | 706                    | 537                    |
| Zap-Q ρ = 1       | 820               | 541               | 625               | Zap-Q ρ = 1            | 55                     | 0                      | 0                      |
| Zap-Q ρ = 0 . 8   | 236               | 737               | 61                | Zap-Q ρ = 0 . 8        | 0                      | 0                      | 0                      |
| Zap-Q ρ = 0 . 85  | 386               | 516               | 74                | Zap-Q ρ = 0 . 85       | 0                      | 0                      | 0                      |
| (a) h n ( x ) < 1 | (a) h n ( x ) < 1 | (a) h n ( x ) < 1 | (a) h n ( x ) < 1 | (b) h n ( x ) < 0 . 95 | (b) h n ( x ) < 0 . 95 | (b) h n ( x ) < 0 . 95 | (b) h n ( x ) < 0 . 95 |
| n                 | 2 × 10 4          | 2 × 10 5          | 2 × 10 6          | n                      | 2 × 10 4               | 2 × 10 5               | 2 × 10 6               |
| G -Q(0) g = 100   | 774               | 727               | 628               | G -Q(0) g = 100        | 545                    | 497                    | 395                    |
| G -Q(0) g = 200   | 789               | 688               | 525               | G -Q(0) g = 200        | 641                    | 518                    | 390                    |
| Zap-Q ρ = 1       | 4                 | 0                 | 0                 | Zap-Q ρ = 1            | 0                      | 0                      | 0                      |
| Zap-Q ρ = 0 . 8   | 0                 | 0                 | 0                 | Zap-Q ρ = 0 . 8        | 0                      | 0                      | 0                      |
| Zap-Q ρ = 0 . 85  | 0                 | 0                 | 0                 | Zap-Q ρ = 0 . 85       | 0                      | 0                      | 0                      |

approach. There are many suggestions in the literature for improving this technique (see discussion after Theorem 3 of [21]) .

The results in this paper suggest new approaches that we hope will simultaneously

- (i) Reduce complexity and potential numerical instability of matrix inversion,
- (ii) Improve transient performance, and
- (iii) Maintain optimality of the asymptotic covariance

## References

- [1] M. G. Azar, R. Munos, M. Ghavamzadeh, and H. Kappen. Speedy Q-learning. In Advances in Neural Information Processing Systems , 2011.
- [2] K. Barman and V. S. Borkar. A note on linear function approximation using random projections. Systems &amp; Control Letters , 57(9):784-786, 2008.
- [3] A. Benveniste, M. M´ etivier, and P. Priouret. Adaptive algorithms and stochastic approximations , volume 22 of Applications of Mathematics (New York) . Springer-Verlag, Berlin, 1990. Translated from the French by Stephen S. Wilson.
- [4] D. P. Bertsekas. Dynamic Programming and Optimal Control , volume 2. Athena Scientific, 4th edition, 2012.
- [5] V. S. Borkar. Stochastic Approximation: A Dynamical Systems Viewpoint . Hindustan Book Agency and Cambridge University Press (jointly), Delhi, India and Cambridge, UK, 2008.

- [6] V. S. Borkar and S. P. Meyn. The ODE method for convergence of stochastic approximation and reinforcement learning. SIAM J. Control Optim. , 38(2):447-469, 2000. (also presented at the IEEE CDC , December, 1998).
- [7] J. A. Boyan. Technical update: Least-squares temporal difference learning. Mach. Learn. , 49(2-3):233-246, 2002.
- [8] S. J. Bradtke and A. G. Barto. Linear least-squares algorithms for temporal difference learning. Mach. Learn. , 22(1-3):33-57, 1996.
- [9] D. Choi and B. Van Roy. A generalized Kalman filter for fixed point approximation and efficient temporal-difference learning. Discrete Event Dynamic Systems: Theory and Applications , 16(2):207-239, 2006.
- [10] E. Even-Dar and Y. Mansour. Learning rates for Q-learning. Journal of Machine Learning Research , 5(Dec):1-25, 2003.
- [11] P. W. Glynn and D. Ormoneit. Hoeffding's inequality for uniformly ergodic Markov chains. Statistics and Probability Letters , 56:143-146, 2002.
- [12] V. R. Konda and J. N. Tsitsiklis. Convergence rate of linear two-time-scale stochastic approximation. Ann. Appl. Probab. , 14(2):796-819, 2004.
- [13] V. V. G. Konda. Actor-critic algorithms . PhD thesis, Massachusetts Institute of Technology, 2002.
- [14] H. J. Kushner and G. G. Yin. Stochastic approximation algorithms and applications , volume 35 of Applications of Mathematics (New York) . Springer-Verlag, New York, 1997.
- [15] R. B. Lund, S. P. Meyn, and R. L. Tweedie. Computable exponential convergence rates for stochastically ordered Markov processes. Ann. Appl. Probab. , 6(1):218-237, 1996.
- [16] D.-J. Ma, A. M. Makowski, and A. Shwartz. Stochastic approximations for finite-state Markov chains. Stochastic Process. Appl. , 35(1):27-45, 1990.
- [17] P. G. Mehta and S. P. Meyn. Q-learning and Pontryagin's minimum principle. In IEEE Conference on Decision and Control , pages 3598-3605, Dec. 2009.
- [18] S. P. Meyn and A. Surana. TD-learning with exploration. In 50th IEEE Conference on Decision and Control, and European Control Conference , pages 148-155, Dec 2011.
- [19] S. P. Meyn and R. L. Tweedie. Computable bounds for convergence rates of Markov chains. Ann. Appl. Probab. , 4:981-1011, 1994.
- [20] S. P. Meyn and R. L. Tweedie. Markov chains and stochastic stability . Cambridge University Press, Cambridge, second edition, 2009. Published in the Cambridge Mathematical Library. 1993 edition online.
- [21] E. Moulines and F. R. Bach. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. In Advances in Neural Information Processing Systems 24 , pages 451459. Curran Associates, Inc., 2011.

- [22] A. Nedic and D. Bertsekas. Least squares policy evaluation algorithms with linear function approximation. Discrete Event Dynamic Systems: Theory and Applications , 13(1-2):79-110, 2003.
- [23] D. Paulin. Concentration inequalities for Markov chains by Marton couplings and spectral methods. Electron. J. Probab. , 20:32 pp., 2015.
- [24] B. T. Polyak. A new method of stochastic approximation type. Avtomatika i telemekhanika (in Russian). translated in Automat. Remote Control, 51 (1991) , pages 98-107, 1990.
- [25] B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM J. Control Optim. , 30(4):838-855, 1992.
- [26] G. A. Rummery and M. Niranjan. On-line Q-learning using connectionist systems. Technical report 166, Cambridge Univ., Dept. Eng., Cambridge, U.K. CUED/F-INENG/, 1994.
- [27] D. Ruppert. A Newton-Raphson version of the multivariate Robbins-Monro procedure. The Annals of Statistics , 13(1):236-245, 1985.
- [28] D. Ruppert. Efficient estimators from a slowly convergent Robbins-Monro processes. Technical Report Tech. Rept. No. 781, Cornell University, School of Operations Research and Industrial Engineering, Ithaca, NY, 1988.
- [29] A. Shwartz and A. Makowski. On the Poisson equation for Markov chains: existence of solutions and parameter dependence. Technical Report, Technion-Israel Institute of Technology, Haifa 32000, Israel., 1991.
- [30] R. S. Sutton. Learning to predict by the methods of temporal differences. Mach. Learn. , 3(1):9-44, 1988.
- [31] C. Szepesv´ ari. The asymptotic convergence-rate of Q-learning. In Proceedings of the 10th International Conference on Neural Information Processing Systems , pages 1064-1070. MIT Press, 1997.
- [32] C. Szepesv´ ari. Algorithms for Reinforcement Learning . Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan &amp; Claypool Publishers, 2010.
- [33] J. N. Tsitsiklis and B. Van Roy. An analysis of temporal-difference learning with function approximation. IEEE Trans. Automat. Control , 42(5):674-690, 1997.
- [34] J. N. Tsitsiklis and B. Van Roy. Optimal stopping of Markov processes: Hilbert space theory, approximation algorithms, and an application to pricing high-dimensional financial derivatives. IEEE Trans. Automat. Control , 44(10):1840-1851, 1999.
- [35] C. J. C. H. Watkins. Learning from Delayed Rewards . PhD thesis, King's College, Cambridge, Cambridge, UK, 1989.
- [36] C. J. C. H. Watkins and P. Dayan. Q -learning. Machine Learning , 8(3-4):279-292, 1992.
- [37] H. Yu and D. P. Bertsekas. Q-learning and policy iteration algorithms for stochastic shortest path problems. Annals of Operations Research , 208(1):95-132, 2013.

## A Appendices

## A.1 Asymptotic covariance for Markov chains

As an illustration of the Lyapunov equation (8) that is solved by the asymptotic covariance, consider the error recursion for a one-dimensional version of (6) in which A n ≡ 1, and the algorithm is scaled by a gain parameter g &gt; 0:

<!-- formula-not-decoded -->

When g = 1 this is a standard Monte-Carlo average. For general g &gt; 1 2 the Lyapunov equation admits the solution:

This grows without bound as g →∞ or g ↓ 1 2 , as illustrated below:

<!-- formula-not-decoded -->

<!-- image -->

Conditions to ensure that the covariance is infinite are presented in the following:

Proposition A.1. Consider the linear recursion

<!-- formula-not-decoded -->

in which { ∆ n } is a martingale difference sequence satisfying Σ ∆ n = Cov(∆ n ) → Σ ∆ as n →∞ .

- (i) Suppose that ( λ, v ) is an eigenvalue-eigenvector pair satisfying GAv = λv , Re ( λ ) ≥ -1 / 2 , and v † G Σ ∆ G T v &gt; 0 . Then,

<!-- formula-not-decoded -->

- (ii) Suppose that all the eigenvalues of GA satisfy Re ( λ ) &lt; -1 / 2 , then the asymptotic covariance Σ G θ is finite, and is obtained as a solution to the Lyapunov equation (13) .

Proof. Define Z n := √ n ˜ θ n and Σ n := E [ Z n Z T n ]. A standard Taylor-series approximation of Z n results in the following recursive definition of Σ n :

<!-- formula-not-decoded -->

The assumptions of part (i) of the proposition implies:

<!-- formula-not-decoded -->

and therefore σ 2 n ≥ v † Σ ∆ v log n + O (1), implying the result in part (i) of the proposition. Under the assumptions of part (ii), (86) implies that Σ n → Σ G θ , where Σ G θ is obtained as a solution to the Lyapunov equation (13). glyph[intersectionsq] glyph[unionsq]

## A.2 Proof of Thm. 3.3

Recall that Π denotes the d × d diagonal matrix with entries Π( k, k ) = glyph[pi1] ( x k , u k ), and the matrix A is a function of q :

<!-- formula-not-decoded -->

In a neighborhood of θ ∗ = Q ∗ , the operator S φ q coincides with S φ ∗ , and we denote A ∞ = A ( θ ∗ ).

Recall from Lemma 3.1 that PS φ ∗ is a d × d transition matrix, and so is the following

To prove the result we construct an eigenvector v ∈ R d for A ∞ whose entries are strictly positive. Next, under the assumptions of the proposition, we show that the corresponding eigenvalue satisfies λ = Re( λ ) ≥ -1 / 2, and the result then follows from Prop. A.1 combined with (51).

<!-- formula-not-decoded -->

The construction of an eigenvector is via the representation -A -1 ∞ = (1 -β ) -1 T Π -1 . Since this is a positive and irreducible matrix, we can apply Perron-Frobenius theory to conclude that there is a maximal eigenvalue λ PF &gt; 0 and an everywhere positive eigenvector v satisfying

<!-- formula-not-decoded -->

The Perron-Frobenius eigenvalue coincides with the spectral radius of A -1 ∞ :

<!-- formula-not-decoded -->

The vector v is also an eigenvector for A ∞ with associated eigenvalue

<!-- formula-not-decoded -->

Thus, λ ≥ -1 2 under the assumptions of the proposition.

## A.3 Proof of Thm. 3.4

The remainder of the Appendix is devoted to the proof of Thm. 3.4.

Lemma 3.9 is used to establish the 'ODE friendly' property for the error sequences appearing in the ODE approximations.

Proof of Lemma 3.9. Let H : X × U → R d × d solve Poisson's equation:

<!-- formula-not-decoded -->

with F n := σ ( X k , U k : k ≤ n ). The following representation is immediate:

<!-- formula-not-decoded -->

glyph[intersectionsq] glyph[unionsq]

where and

<!-- formula-not-decoded -->

which satisfies (70) under Assumption Q3.

glyph[intersectionsq]

glyph[unionsq]

Recall the ' o (1)' notation used in (67) is interpreted in a functional sense. It is a function of two variables: o (1) = E T 0 ,T 0 + t , satisfying for each T &gt; 0,

<!-- formula-not-decoded -->

The notation E T 0 ,T 0 + t = o ( v T 0 ,t ) for a vector-valued function of ( T 0 , t ) has an analogous interpretation:

<!-- formula-not-decoded -->

This notation will also be used in the standard setting in which the variable t is absent. In particular, E T 0 = o (1) simply means that E T 0 → 0 as T 0 →∞ .

Recall the definitions of α n , γ n , and t n in (59) and (66); g t was defined to be a piecewise linear function with g t = γ n /α n when t = t n for some n , and extended to all t ∈ R + by linear interpolation. Recall that g t ≈ exp((1 -ρ ) t ) for large t under (59). For fixed T 0 &gt; 0, let G T 0 , ( · ) denote the cumulative distribution function on the interval [0 , T 0 ]:

<!-- formula-not-decoded -->

This CDF defines a probability measure on the interval [0 , T 0 ] with the following properties:

Lemma A.2. The probability measure associated with G T 0 , ( · ) has a density on (0 , T 0 ] and a single point mass at zero. Its total mass is concentrated near T 0 : For any κ &gt; 0 ,

<!-- formula-not-decoded -->

An associated pmf on the set of policy indices is defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the final term is the martingale difference sequence:

<!-- formula-not-decoded -->

The telescoping sequence is thus,

<!-- formula-not-decoded -->

glyph[intersectionsq]

glyph[unionsq]

Lemma A.3. The linear systems representation (73) holds:

<!-- formula-not-decoded -->

Furthermore,

- (i) There exists T · ≥ 0 satisfying T · &lt; ∞ a.s., and for which the processes { ¯ A t : t ≥ T · } and { ¯ A -1 t : t ≥ T · } are bounded:

<!-- formula-not-decoded -->

- (ii) For each t, T 0 ≥ 0 ,

<!-- formula-not-decoded -->

- (iii) With ∂ Q µ defined in (43) , and µ t defined in (88) , the following representation holds:

<!-- formula-not-decoded -->

- (iv) The following approximations hold:

<!-- formula-not-decoded -->

Proof. The representation (73) directly follows from Lemmas 3.8 and 3.9. For T 0 &gt; 0, the solution to this linear time varying system is the sum of three terms:

<!-- formula-not-decoded -->

in which E A T 0 ,T 0 + τ = o (1), T 0 →∞ and

<!-- formula-not-decoded -->

and for each t ≥ 0, k t is the integer satisfying φ ( k t ) = φ ¯ q t .

<!-- formula-not-decoded -->

We begin with a proof of boundedness of { ¯ A t : t ≥ 0 } , considering the three terms in (92) separately. The first term I A T 0 ( t ) vanishes as t →∞ for each fixed T 0 . The second term admits the uniform bound:

in which b A := max k ‖ Π[ I -βPS φ ( k ) ] ‖ is an upper bound on ‖ Π ∂ Q -1 k τ ‖ . It is shown next that the final term I E T 0 ( t ) converges to zero as T 0 →∞ . Applying integration by parts:

<!-- formula-not-decoded -->

where the second equation used G T 0 + t,T 0 + t = 1 and E A T 0 ,T 0 = 0. Using this identity and the same arguments used to bound I B T 0 ( t ) then gives

<!-- formula-not-decoded -->

Using (93) and (94) in (92) establishes boundedness of { ¯ A t } :

<!-- formula-not-decoded -->

and hence also boundedness of the sequence { ̂ A n : n ≥ 0 } .

The representation (92) along with (94) also implies the evolution equation in part (ii) of the lemma. It is shown next that (91) holds. This will imply that { ¯ A -1 t : t ≥ T · } is bounded, which will complete the proof of the lemma.

The approximation (92) was obtained for T 0 &gt; 0. Lemma A.2 implies that we can let T 0 ↓ 0 in this bound to obtain,

<!-- formula-not-decoded -->

Next, based on the definition of µ t in (88), the representation (90) for ∂ Q -1 µ t is obtained:

<!-- formula-not-decoded -->

Combining the above result with (95), (91) is obtained. This also implies that { ¯ A -1 t : t ≥ T · } is bounded. glyph[intersectionsq] glyph[unionsq]

Lemma A.4. The linear systems representation holds:

<!-- formula-not-decoded -->

Furthermore, for a constant b q &lt; ∞ , and each t, T 0 ≥ 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Boundedness of { ̂ A -1 n : n ≥ n · } in Lemma A.3 (i) implies that ̂ G ∗ n = -̂ A -1 n for n ≥ n · in recursion (71). Representation (96) then follows from Lemmas 3.8 and 3.9. The evolution equation (96) implies:

<!-- formula-not-decoded -->

Denoting

<!-- formula-not-decoded -->

and applying Lemma A.3 (i),

<!-- formula-not-decoded -->

The factor 1 before the integral in the second inequality comes from choosing T 0 large enough, so that o ( ¯ q ) ¯ q . Equation (97a) is then obtained by applying the Gr¨ onwall lemma [5].

Applying the triangle inequality to (97a), and choosing T 0 such that ‖ o (1) ‖ ≤ 1 for all T 0 ≥ T 0 gives

‖ T 0 + τ ‖ ≤ ‖ T 0 + τ ‖

<!-- formula-not-decoded -->

In particular, the above inequality is true for T 0 = T 0 . Furthermore, the following holds under the assumption that the sequence { ̂ A n } is projected prior to inversion: For a constant b T 0 &lt; ∞ ,

<!-- formula-not-decoded -->

The bound (97b) is obtained on combining (99) and (100).

Lemma A.5. For each t ≥ 0 , the following holds:

<!-- formula-not-decoded -->

glyph[intersectionsq]

glyph[unionsq]

<!-- formula-not-decoded -->

Proof. Equation (90) of Lemma A.3 implies the following representation:

<!-- formula-not-decoded -->

Subtracting c t from each side and taking norms gives the bound,

<!-- formula-not-decoded -->

Lemma 3.2 implies that the mappings Q and Q -1 are Lipschitz: for a constant b Q ,

<!-- formula-not-decoded -->

Substituting into (102) gives

<!-- formula-not-decoded -->

where the second inequality uses equation (97a) of Lemma A.4, and the last approximation follows from Lemma A.2.

Next, applying (97b) of Lemma A.4 and Lemma A.2, (101a) is obtained:

<!-- formula-not-decoded -->

Using (101a) and (91), (101b) is obtained:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma A.6. (i) sup t ≥ 0 ‖ ¯ q t ‖ &lt; ∞ , a.s. .

- (ii) For t ≥ 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Equation (96) of Lemma A.4 implies,

<!-- formula-not-decoded -->

Lemma A.5 along with the fact that { ¯ A -1 t : t ≥ T · } is bounded implies

<!-- formula-not-decoded -->

Boundedness of { ¯ q t } is established by applying the Gr¨ onwall lemma [5], and (103a) immediately follows.

Next apply the approximation (91): substituting -¯ A -1 T Π = ∂ Q µ T + o (1) in (104), and using the fact that { ¯ q t } is bounded, (103b) is obtained. Equation (103c) then follows from (101b) of Lemma A.5. glyph[intersectionsq] glyph[unionsq]

With these results established, the ODE approximation will quickly follow. For a fixed but arbitrary time-horizon T &gt; 0, define a family of uniformly bounded and uniformly Lipschitz continuous functions { Γ T 0 : T 0 ≥ 0 } , where Γ T 0 : [0 , T ] → R m for each T 0 ≥ 0 and some integer m . The family of functions is constructed from the following familiar components: for t ∈ [0 , T ],

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

More precisely, Γ T 0 is a function of two variables, t ∈ [0 , T ] and ω ∈ Ω. To say that { Γ T 0 } is uniformly bounded and uniformly Lipschitz continuous means that there exists Ω · ∈ F with measure one such that for each ω ∈ Ω · , the family of functions { Γ T 0 ( ω, · ) : T 0 ≥ 0 } , is uniformly bounded and Lipschitz. The bound and the Lipschitz constant may depend on ω , but are independent of T 0 .

Any sub-sequential limit of { Γ T 0 : T 0 ≥ 0 } will be denoted Γ. To maintain consistency with the notation in Thm. 3.4, the first two components are denoted

<!-- formula-not-decoded -->

The ODE limit is recast as follows:

Proposition A.7. Any sub-sequential limit Γ of { Γ T 0 : T 0 ≥ 0 } must have the following form: for t ∈ [0 , T ] ,

- (i) q t := Γ 1 ( t ) = Q ( c t ) .
- (ii) d dt c t = -c t + c .
- (iii) For a.e. t ∈ [0 , T ] , there is a pmf µ t such that

<!-- formula-not-decoded -->

The first relation, that Γ 1 ( t ) = Q (Γ 2 ( t )), is obvious because the mapping Q is continuous. The proofs of (ii) and (iii) are similar: prior results and a few results that follow are reinterpreted as properties of { Γ T 0 } that are preserved in any sub-sequential limit. For example, (103c) admits the representation in terms of Γ T 0 :

<!-- formula-not-decoded -->

The following result establishes that the left hand side represents a continuous functional of Γ T 0 :

Lemma A.8. For fixed T &gt; 0 , l &gt; 0 , and b &gt; 0 , let H l,b denote the set of all functions h : [0 , T ] → R satisfying ‖ h ‖ ≤ b , and are Lipschitz continuous with Lipschitz constant l :

<!-- formula-not-decoded -->

The set H l,b is compact as a subset of C ([0 , T ] , R ) . Moreover, the following real-valued functional is Lipschitz continuous on H l,b ×H l,b :

<!-- formula-not-decoded -->

Proof. Since C T is bilinear, it is sufficient to obtain Lipschitz constants in either variable.

For a fixed function g ∈ H l,b , and any two functions f 1 , f 2 ∈ H l,b ,

<!-- formula-not-decoded -->

which implies Lipschitz continuity of C T in its first variable, with Lipschitz constant Tl .

A similar result is obtained for a fixed f ∈ H l,b . Using integration by parts:

<!-- formula-not-decoded -->

For any two functions g 1 , g 2 ∈ H l,b ,

<!-- formula-not-decoded -->

This proves Lipschitz continuity of C T in its second variable, with Lipschitz constant (2 b + Tl ). glyph[intersectionsq] glyph[unionsq]

The next result implies another continuous relationship for Γ T 0 3 :

Lemma A.9. For each T 0 ≥ 0 and t &gt; 0 , there exists a pmf µ such that

<!-- formula-not-decoded -->

with µ τ defined in (88) . That is, the matrix t -1 Γ T 0 3 ( t ) lies in the compact set { ∂ Q -1 ν : ν is a pmf } . glyph[intersectionsq] glyph[unionsq]

Lemma A.10. For any sub-sequential limit Γ , let t 0 denote a point at which both q t and c t are differentiable. Let µ be any pmf satisfying ∂ Q µ c t 0 = Q ( c t 0 ) = q t 0 . Then,

<!-- formula-not-decoded -->

Proof. This is an instance of the chain rule. A proof is provided since Q is not smooth.

These two functions are approximated by a line at this time point:

<!-- formula-not-decoded -->

where v q , v c are the respective derivatives. The lemma asserts that v q = ∂ Q µ v c .

Denote L q t = q t 0 +( t -t 0 ) v q , L c t = c t 0 +( t -t 0 ) v c , t ∈ R . Applying (44) we have for each t :

<!-- formula-not-decoded -->

Differentiability of Q ( c t ) at t 0 then implies the desired conclusion: v q = d dt Q ( L c t ) ∣ ∣ t = t 0 = ∂ Q µ v c . glyph[intersectionsq] glyph[unionsq]

<!-- formula-not-decoded -->

Proof of Prop. A.7. Recall that (i) has been established. Result (iii) is established next, which will quickly lead to (ii).

Lemma A.9 implies the following relationship for Γ T 0 3 : For each T 0 ≥ 0 and t &gt; 0, there exists a pmf µ such that

<!-- formula-not-decoded -->

It follows that the same is true for any sub-sequential limit Γ: There is a parameterized family of pmfs { µ τ } such that

<!-- formula-not-decoded -->

In the pre-limit we have d dt Γ T 0 3 ( t ) × d dt Γ T 0 4 ( t ) = I for each t and T 0 . It can be shown using Laplace transform arguments that the same must be true in the limit for a.e. t , giving the first half of (iii).

Next, we prove the second half of (iii): ∂ Q -1 µ t q t = c t for a.e. t . From equation (101a) of Lemma A.5,

In Γ T 0 notation:

<!-- formula-not-decoded -->

Lemma A.8 asserts that the left hand side of the above equation defines a continuous functional of Γ T 0 , and therefore the relationship also holds in the limit:

<!-- formula-not-decoded -->

This establishes the second half of part (iii) of the lemma.

Part (ii) is obtained using similar arguments: it is established that the left hand side of (105) is a continuous mapping of Γ T 0 , so the relation is true for any sub-sequential limit Γ:

<!-- formula-not-decoded -->

Combining Lemma A.10 with (iii) gives ∂ Q µ t d dt c t = d dt q t at points of differentiability, and thence (107) implies that for a.e. t

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Table 2: Notation

| Symbol           | Type                     | Description                                                                         |
|------------------|--------------------------|-------------------------------------------------------------------------------------|
| X                | set; x is a component    | state space of the Markov chain                                                     |
| U                | set; u is a component    | action space of the Markov chain                                                    |
| c                | function : X × U → R     | cost function                                                                       |
| β                | scalar ∈ (0 , 1)         | discount factor                                                                     |
| θ                | vector; ∈ R d            | parameter vector                                                                    |
|                  | function : X → R d       | basis functions for TD-learning                                                     |
| ψ                | function : X × U → R d   | basis functions for Q-learning                                                      |
| α n              | scalar; ∈ (0 , 1]        | step-size sequence                                                                  |
| γ n              | scalar; ∈ (0 , 1]        | step-size sequence                                                                  |
| h                | function : X → R         | value function                                                                      |
| h φ θ            | function : X → R         | a linear approximation to h : h φ θ = θ T ψ                                         |
| Q                | function : X × U → R     | SARSA Q-function for an uncontrolled Markov chain                                   |
| Q θ              | function : X × U → R     | a linear approximation to Q : Q θ = θ T ψ                                           |
| θ ∗              | vector; ∈ R d            | optimal parameter vector satisfying: h = h θ ∗ or Q = Q θ ∗                         |
| ˜ θ              | vector; ∈ R d            | error in the parameter vector: ˜ θ = θ - θ ∗                                        |
| q                | operator                 | given q : X × U → R , q ( x ) = min u q ( x,u )                                     |
|                  | function : X → R d       | eligibility vector for TD-learning                                                  |
| ζ                | function : X × U → R d   | eligibility vector for Q-learning                                                   |
| Φ                | sequence { Φ n : n ≥ 0 } | Markov chain of interest when applying stochastic approximation                     |
| glyph[pi1]       | function : X × U → R     | steady state distribution / probability mass function of Φ                          |
| ∆                | sequence { ∆ n : n ≥ 0 } | error sequence                                                                      |
| φ                | function : X → U         | policy                                                                              |
| glyph[lscript]   | scalar                   | number of elements in X : glyph[lscript] = | X |                                    |
| glyph[lscript] u | scalar                   | number of elements in U : glyph[lscript] u = | U |                                  |
| glyph[lscript] φ | scalar                   | number of possible policies: glyph[lscript] φ = ( glyph[lscript] u ) glyph[lscript] |
| P u              | operator                 | tr. kernel (controlled): P u f ( x ) = E [ f ( X n +1 ) | X n = x, U n = u ]        |
| P φ              | operator                 | tr. kernel (uncontrolled): P φ f ( x ) = E [ f ( X n +1 ) | X n = x, U n = φ ( x )] |
| S φ              | operator                 | given q : X × U → R , S φ q ( x ) = q ( x,φ ( x ))                                  |
| h φ              | function : X → R         | value function for a given policy φ                                                 |
| φ q              | function : X → U         | q -optimal policy: φ q ( x ) ∈ argmin u q ( x,u )                                   |
| h φ              | function : X → R         | value function for an uncontrolled Markov chain with policy φ                       |
| h ∗              | function : X → R         | optimal value function                                                              |
| Q                | operator                 | given c : X × U → R , Q ( c ) is the optimal Q -function for cost c                 |
| Q ∗              | function : X × U → R     | optimal Q -function for cost c                                                      |
| B                | function : X × U → R     | Bellman error for the Q -function approximation                                     |
| Π                | matrix ∈ R d × d         | diagonal matrix: Π( k,k ) = glyph[pi1] ( x ( k ) ,u ( k ) )                         |
| Ψ                | matrix ∈ R d × d         | outer product of the basis functions: Ψ = ψ × ψ T                                   |
| G                | sequence { G n : n ≥ 0 } | arbitrary sequence of matrix gains                                                  |
| G                | matrix ∈ R d × d         | steady state mean of G                                                              |
| A n ; A          | matrix ∈ R d × d         | linearization in stochastic approximation; A s.s. mean                              |
| b n ; b          | vector ∈ R d             | vector sequence; b s.s. mean                                                        |
| ̂ A n             | matrix ∈ R d × d         | n -step Monte-Carlo estimate of A                                                   |
| ̂ b n             | vector ∈ R d             | n -step Monte-Carlo estimate of b                                                   |