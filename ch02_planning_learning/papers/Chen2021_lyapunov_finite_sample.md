## A Lyapunov Theory for Finite-Sample Guarantees of Asynchronous Q-Learning and TD-Learning Variants

Zaiwei Chen 1 , ∗ , Siva Theja Maguluri 1 , † , Sanjay Shakkottai 2 , and Karthikeyan Shanmugam 3

1 Georgia Institute of Technology, ∗ zchen458@caltech.edu , † siva.theja@gatech.edu

2 The University of Texas at Austin, 2 sanjay.shakkottai@utexas.edu

3 IBM Research AI group, 3 KarthikeyanShanmugam88@gmail.com

## Abstract

This paper develops an unified framework to study finite-sample convergenceguarantees of a large class of value-based asynchronous reinforcement learning (RL) algorithms. We do this by first reformulating the RL algorithms as Markovian Stochastic Approximation (SA) algorithms to solve fixed-point equations. We then develop a Lyapunov analysis and derive mean-square error bounds on the convergence of the Markovian SA. Based on this result, we establish finite-sample mean-square convergence bounds for asynchronous RL algorithms such as Q -learning, n -step TD, TD ( λ ) , and off-policy TD algorithms including V-trace. As a by-product, by analyzing the convergence bounds of n -step TD and TD ( λ ) , we provide theoretical insights into the bias-variance trade-off, i.e., efficiency of bootstrapping in RL. This was first posed as an open problem in [41].

## 1 Introduction

Reinforcement learning (RL) is a promising approach to solve sequential decision making problems in complex and stochastic systems [42]. RL has seen remarkable successes in solving many practical problems, such as the game of Go [38], health care [13], and robotics [25]. Despite such empirical successes, the convergence properties of many RL algorithms are not well understood.

Most of the value-based RL algorithms can be viewed as stochastic approximation (SA) algorithms for solving suitable Bellman equations. Due to the nature of sampling in RL, many such algorithms inevitably perform the so-called asynchronous update. That is, in each iteration, only a subset of the components of the vector-valued iterate is updated. Moreover, the components being updated are usually selected in a stochastic manner along a single trajectory based on an underlying Markov chain. Handling such asynchronous updates is one of the main challenges in analyzing the behavior of RL algorithms. In this paper, we study such asynchronous RL algorithms through the lens of Markovian SA algorithms, and develop a unified Lyapunov approach to establish finite-sample bounds on the mean-square error. The results enable us to tackle the long-standing problem about the efficiency of bootstrapping in RL [41].

## 1.1 Main Contributions

We next summarize our main contributions in the following.

Finite-Sample Bounds for Markovian SA. We establish finite-sample convergence guarantees (under various choices of stepsizes) of a stochastic approximation algorithm, which involves a contraction mapping, and is driven by both Markovian and martingale difference noise. Specifically, when using constant stepsize α , the convergence rate is geometric, with asymptotic accuracy approximately O ( α log(1 /α )) . When using diminishing stepsizes of the form α/ ( k + h ) ξ (where ξ ∈ (0 , 1] ), the convergence rate is O (log( k ) /k ξ ) , provided that α and h are appropriately chosen.

Finite-Sample Bounds for Q -Learning. We establish finite-sample convergence bounds of the asynchronous Q -learning algorithm. In the constant stepsize regime, our result implies a sample complexity of

<!-- formula-not-decoded -->

where N min is the minimal component of the stationary distribution on the state-action space induced by the behavior policy. Our result improves the state-of-the-art mean square bound of asynchronous Q -learning [3] by a factor of at least |S||A| . See Section 3.1.3 for a detailed comparison with related literature.

Finite-Sample Bounds for V-trace. We establish for the first time finite-sample convergence bounds of the V-trace algorithm when performing asynchronous update [17]. The V-trace algorithm can be viewed as an off-policy variant of the n -step TD-learning algorithm, and uses two truncation levels ¯ c and ¯ ρ in the importance sampling ratios to control the bias and variance in the algorithm. It was discussed in [17] that qualitatively, ¯ ρ mainly determines the limit point of V-trace, while ¯ c mainly controls the variance in the estimate. Our finite-sample analysis quantitatively justifies this observation by showing a sample complexity bound proportional to ¯ ρ 2 (∑ n i =0 ( γ ¯ c ) i ) 2 . Based on this result, we see that ¯ c is the main reason for variance reduction, and we need to aggressively choose the truncation level ¯ c ≤ 1 /γ to avoid an exponential factor in the sample complexity.

Finite-Sample Bounds for n -Step TD. We establish finite-sample convergence guarantees of the onpolicy n -step TD-learning algorithm. In n -step TD, the parameter n adjusts the degree of bootstrapping in the algorithm. In particular, n = 1 corresponds to extreme bootstrapping (TD (0) ), while n = ∞ corresponds to no bootstrapping (Monte Carlo method). Despite empirical observations [42], the choice of n that leads to the optimal performance of the algorithm is not theoretically understood. Based on our finite-sample analysis, we show that the parameter n appears as n/ (1 -γ n ) 2 in the sample complexity result, therefore demonstrate an explicit trade-off between bootstrapping (small n ) and Monte Carto method (large n ). In addition, based on the sample complexity bound, we show that in order to achieve the optimal performance of the n -step TD-learning algorithm, the parameter n should be chosen approximately as min(1 , 1 / log(1 /γ )) .

Finite-Sample Bounds for TD ( λ ) . We establish finite-sample convergence bounds of the TD ( λ ) algorithm for any λ in the interval (0 , 1) . The TD ( λ ) update can be viewed as a convex combination of all n -step TD-learning update. Similar to n -step TD, the parameter λ is used to adjust the degree of bootstrapping in TD ( λ ) , and there is a long-standing open problem about the efficiency of bootstrapping [41]. By deriving explicit finite-sample performance bounds of the TD ( λ ) algorithm as a function of λ , we provide theoretical insight into the bias-variance trade-off in choosing λ . Specifically, in the constant-stepsize TD ( λ ) algorithm, after the k -th iteration, the 'bias" is of the size (1 -Θ(1 / (1 -βλ ))) k , which is in favor of large λ (more Monte Carlo), while the 'variance" is of the size Θ(1 / [(1 -βλ ) log(1 / ( βλ ))]) , and is in favor of small λ (more bootstrapping).

## 1.2 Motivation and Technical Approach

In this Section, we illustrate our approach of dealing with asynchronous RL algorithms using the Q -learning algorithm as a motivating example.

## 1.2.1 Illustration via Q-Learning

The Q -learning algorithm is a recursive approach for finding the optimal policy corresponding to a Markov decision process (MDP) (see Section 3.1 for details). At time step k , the algorithm updates a vector (of dimension state-space size × action-space size) Q k , which is an estimate of the optimal Q -function, using noisy samples collected along a single trajectory (aka. sample-path). After a sufficient number of iterations, the vector Q k is a close approximation of the true Q -function, which (after some straightforward computations) delivers the optimal policy for the MDP. Concretely, let { ( S k , A k ) } be a sample trajectory of state-action pairs collected by applying some behavior policy to the model. The Q -learning algorithm performs a scalar update to the (vector-valued) iterate Q k based on:

when ( s, a ) = ( S , A ) , and Q ( s, a ) = Q ( s, a ) otherwise. Further,

<!-- formula-not-decoded -->

is a function representing the temporal difference in the Q -function iterate.

At a high-level, this recursion approximates the fixed-point of the Bellman equation through samples along a single trajectory. There are, however, two sources of noise in this approximation: (1) asynchronous update where only one of the components in the vector Q k is updated (component corresponding to the state-action pair ( S k , A k ) encountered at time k ), and other components in the vector Q k are left unchanged, and (2) stochastic noise due to the expectation in the Bellman operator being replaced by a single sample estimate Γ 1 ( · ) at time step k.

## 1.2.2 Reformulation through Markovian SA

To overcome the challenge of asynchronism (aka. scalar update of the vector Q k ), our first step is to reformulate asynchronous Q -learning as a Markovian SA algorithm [7] by introducing an operator that captures asynchronous updates along a trajectory. A Markovian SA algorithm is an iterative approach to solve fixedpoint equations (see Section 2), and leads to recursions of the form:

<!-- formula-not-decoded -->

Here x k is the main iterate, α k is the stepsize, F ( · ) is an operator that is (in an appropriate expected sense) contractive with respect to a suitable norm, Y k is noise derived from the evolution of a Markov chain, and w k is additive noise (see Section 2 for details). To cast Q -learning as a Markovian SA, let F : R |S||A| ×S×A× S ↦→ R |S||A| be an operator defined by [ F ( Q,s 0 , a 0 , s 1 )]( s, a ) = /BD { ( s 0 ,a 0 )=( s,a ) } Γ 1 ( Q,s 0 , a 0 , s 1 ) + Q ( s, a ) for all ( s, a ) . Then the Q -learning algorithm (1) can be rewritten as:

<!-- formula-not-decoded -->

which is of the form of (2) with x k replaced by Q k , w k = 0 , and Y k = ( S k , A k , S k +1 ) . The key takeaway is that in (3), the various noise terms (both due to performing asynchronous update and due to samples replacing an expectation in the Bellman equation) are encoded through introducing the operator F ( · ) and the associated evolution of the Markovian noise { Y k } .

## 1.2.3 Analyzing the Markovian SA

To study the Markovian SA algorithm (3), let ¯ F ( · ) be the expectation of F ( · , S k , A k , S k +1 ) , where the expectation is taken with respect to the stationary distribution of the Markov chain { ( S k , A k , S k +1 ) } . Under

mild conditions, we show that ¯ F ( Q ) = N H ( Q )+( I -N ) Q . Here H ( · ) is the Bellman's optimality operator for the Q -function [5]. The matrix N is a diagonal matrix with { p ( s, a ) } ( s,a ) ∈S×A sitting on its diagonal, where p ( s, a ) is the stationary visitation probability of the state-action pair ( s, a ) .

Thus, by recentering the iteration in (3) about ¯ F ( · ) , we have:

An important insight about the operator ¯ F ( · ) is that it can be viewed as an asynchronous variant of the Bellman operator H ( · ) . To see this, consider a state-action pair ( s, a ) . The value of [ ¯ F ( Q )]( s, a ) can be interpreted as the expectation of a random variable, which takes [ H ( Q )]( s, a ) w.p. p ( s, a ) , and takes Q ( s, a ) w.p. 1 -p ( s, a ) . This precisely captures the asynchronous update in the Q -learning algorithm (1) in that, at steady-state, Q k ( s, a ) is updated w.p. p ( s, a ) , and remains unchanged otherwise. Moreover, since it is well-known that H ( · ) is a contraction mapping with respect to ‖·‖ ∞ , we also show that ¯ F ( · ) is a contraction mapping with respect to ‖ · ‖ ∞ , with the optimal Q -function being its unique fixed-point.

<!-- formula-not-decoded -->

In summary, we have recast asynchronous Q -learning as an iterative update that decomposes the Q k update into an expected update (averaged over the stationary distribution of the noise Markov chain) and a 'residual update" due to the Markovian noise. As will see in Section 2, this update equation has the interpretation of solving the fixed-point equation ¯ F ( Q ) = Q , with Markovian noise in the update.

## 1.2.4 Finite-Sample Bounds for Markovian SA

Weuse a unified Lyapunov approach for deriving finite-sample bounds on the update in (4). Specifically, the Lyapunov approach handles both (1) non-smooth ‖ · ‖ ∞ -contraction of the averaged operator ¯ F ( · ) , and (2) Markovian noise that depends on the state-action trajectory. To handle ‖·‖ ∞ -contraction, or more generally arbitrary norm contraction, inspired by [11], we use the Generalized Moreau Envelope as the Lyapunov function. To handle the Markovian noise, we use the conditioning argument along with the geometric mixing of the underlying Markov chain [5, 40]. Finally, for recursions beyond Q -learning, we deal with additional extraneous martingale difference noise through the tower property of the conditional expectation.

As we later discuss, beyond Q -learning, TD-learning variants such as off-policy V-trace, n -step TD, and TD ( λ ) can all be modeled by Markovian SA algorithms involving a contraction mapping (possibly with respect to different norm), and Markovian noise. Therefore, our approach unifies the finite-sample analysis of value-based RL algorithms.

## 1.3 Related Literature

In this section, we discuss related literature on SA algorithms. We defer the discussion on related literature on finite-sample bounds of RL algorithms (such as Q -learning, V-trace, n -step TD, and TD ( λ ) ) to the corresponding sections where we introduce these results.

Stochastic approximation method was first introduced in [36] for iteratively solving systems of equations. Since then, SA method is widely used in the context of optimization and machine learning. For example, in optimization, a special case of SA known as stochastic gradient descent (SGD) is a popular approach for iteratively finding the stationary points of some objective function [9]. In reinforcement learning, SA method is commonly used to solving the Bellman equation [5], as will be studied in this paper.

Early literature on SA focuses on the asymptotic convergence [4, 26, 27]. A popular approach there is to view the SA algorithm as a stochastic and discrete counterpart of an ordinary differential equation (ODE), and show that SA algorithms converges asymptotically as long as the ODE is stable. See [7, 8] for more details about such ODE approach. Beyond convergence, the asymptotic convergence rate of SA algorithms are studied in [10, 15].

More recently, finite-sample convergence guarantees of SA algorithms have seen a lot of attention. For SA algorithms with i.i.d. or martingale difference noise, finite-sample analysis was performed in [47] under a cone-contraction assumption, and in [11] under a contraction assumption. The Lyapunov function we use in this paper is indeed inspired by [11]. However, [11] studies SA under martingale difference noise while we have both martingale and Markovian noise.

For SA algorithms with Markovian noise, finite-sample convergence bounds were established in [6, 40] for linear SA. For nonlinear SA with Markovian noise, [12] established the convergence bounds under a strong monotone assumption. In this paper, the operator we work with is neither linear nor strongly monotone.

In the context of optimization, convergence rates of SGD algorithms have been studied thoroughly in the literature. See [9, 28] and the references therein for more details. In SGD algorithm, the update involves the gradient of some objective function, while in our setting, we do not have such gradient. Therefore, the SA algorithm we study in this paper is different from SGD (except when minimizing a smooth and strongly convex function, in which case there is contractive operator with respect to the Euclidean norm [37]).

## 2 Markovian Stochastic Approximation

In this section, we present finite-sample convergence bounds for a general stochastic approximation algorithm, which serves as a universal model for the RL algorithms we are going to study in Section 3.

## 2.1 Problem Setting

Suppose we want to solve for x ∗ ∈ R d in the equation

<!-- formula-not-decoded -->

where Y ∈ Y is a random variable with distribution µ , and F : R d ×Y ↦→ R d is a general nonlinear operator. We assume the set Y is finite, and denote ¯ F ( x ) = E Y ∼ µ [ F ( x, Y )] as the expected operator.

In the case where ¯ F ( · ) is known, Eq. (5) can be solved using the simple fixed-point iteration x k +1 = ¯ F ( x k ) , which is guaranteed to convergence when ¯ F ( · ) is a contraction operator. When the distribution µ of the random variable Y is unknown, and hence ¯ F ( · ) is unknown, we consider solving Eq. (5) using the stochastic approximation method described in the following.

Let { Y k } be Markov chain with stationary distribution µ . Then the SA algorithm iteratively updates the estimate x k by:

<!-- formula-not-decoded -->

where { α k } is a sequence of stepsizes, and { w k } is a random process representing the additive extraneous noise. To establish finite-sample convergence bounds of Algorithm (6), we next formally state our assumptions. Most of them are naturally satisfied in the RL algorithms we are going to study in Section 3. Let ‖·‖ c be some arbitrary norm in R d .

Assumption 2.1. There exist A 1 , B 1 &gt; 0 such that

<!-- formula-not-decoded -->

- (2) ‖ F ( 0 , y ) ‖ c ≤ B 1 for any y ∈ Y .

Assumption 2.2. The operator ¯ F ( · ) is a contraction mapping with respect to ‖ · ‖ c , with contraction factor β ∈ (0 , 1) . That is, it holds for any x 1 , x 2 ∈ R d that ‖ ¯ F ( x 1 ) -¯ F ( x 2 ) ‖ c ≤ β ‖ x 1 -x 2 ‖ c .

Remark. By applying Banach fixed-point theorem [1], Assumption 2.2 guarantees that the target equation (5) has a unique solution, which we have denoted by x ∗ .

Assumption 2.3. The Markov chain M = { Y k } has a unique stationary distribution µ ∈ ∆ |Y| , and there exist constants C &gt; 0 and σ ∈ (0 , 1) such that max y ∈Y ‖ P k ( y, · ) -µ ( · ) ‖ TV ≤ Cσ k for all k ≥ 0 , where ‖ · ‖ TV stands for the total variantion distance [29].

Remark. Since the state-space Y of the Markov chain { Y k } is finite, Assumption 2.3 is satisfied when { Y k } is irreducible and aperiodic [29].

Under Assumption 2.3, we next introduce the notion of Markov chain mixing, which will be frequently used in our derivation.

Definition 2.1. For any δ &gt; 0 , the mixing time t δ ( M ) of the Markov chain M = { Y k } with precision δ is defined by t δ ( M ) = min { k ≥ 0 : max y ∈Y ‖ P k ( y, · ) -µ ( · ) ‖ TV ≤ δ } .

Remark. Note that under Assumption 2.3, we have t δ ≤ log( C/σ )+log(1 /α ) log(1 /σ ) for any δ &gt; 0 , which implies lim δ → 0 δt δ = 0 . This property is important in our analysis for controlling the Markovian noise { Y k } .

For simplicity of notation, in this section, we will just write t δ for t δ ( M ) , and further use t k for t α k , where α k is the stepsize used in the k -th iteration of Algorithm (6). To state our last assumption regarding the additive noise { w k } . let F k is the Sigma-algebra generated by { ( x i , Y i , w i ) } 0 ≤ i ≤ k -1 ∪ { x k } .

Assumption 2.4. The random process { w k } satisfies

- (1) E [ w k |F k ] = 0 for all k ≥ 0 .

(2) ‖ w k ‖ c ≤ A 2 ‖ x k ‖ c + B 2 for all k ≥ 0 , where A 2 , B 2 &gt; 0 are numerical constants.

Remark. Assumption 2.4 states that { w k } is a martingale difference sequence with respect to the filtration F k , and it can grow at most linear with respect to the iterate x k .

Finally, we specify the requirements for choosing the stepsize sequence { α k } . We will consider using stepsizes of the form α k = α ( k + h ) ξ , where α, h &gt; 0 and ξ ∈ [0 , 1] . The constants ¯ α and ¯ h used in stating the following condition are specified in Appendix A.2.

Condition 2.1. (1) Constant Stepsize. When ξ = 0 , there exists a threshold ¯ α ∈ (0 , 1) such that the stepsize α is chosen to be in (0 , ¯ α ) . (2) Linear Stepsize. When ξ = 1 , for each α &gt; 0 , there exists a threshold ¯ h &gt; 0 such that h is chosen to be at least ¯ h . (3) Polynomial Stepsize. For any ξ ∈ (0 , 1) and α &gt; 0 , there exists a threshold ¯ h &gt; 0 such that h is chosen to be at least ¯ h .

## 2.2 Finite-Sample Convergence Guarantees

Under the assumptions stated above, we next present the finite-sample bounds of Algorithm (6).

For simplicity of notation, let A = A 1 + A 2 +1 , which can be viewed as the combined effective Lipschitz constant, and let B = B 1 + B 2 . Let c 1 = ( ‖ x 0 -x ∗ ‖ c + ‖ x 0 ‖ c + B/A ) 2 , and c 2 = ( A ‖ x ∗ ‖ c + B ) 2 . The constants { ϕ i } 1 ≤ i ≤ 3 we are going to use are defined explicitly in Appendix A, and depend only on the contraction norm ‖ · ‖ c and the contraction factor γ . We will revisit them later in Lemma 2.1. Define K = min { k ≥ 0 : k ≥ t k } , which is well-defined under Assumption 2.3. We now state the finite-sample convergence bounds of Algorithm (6) in the following.

Theorem 2.1. Consider { x k } of Algorithm (6). Suppose that Assumptions 2.1, 2.2, 2.3 and 2.4 are satisfied. Then we have the following results.

- (1) When k ∈ [0 , K -1] , we have ‖ x k -x ∗ ‖ 2 c ≤ c 1 almost surely.
- (2) When k ≥ K , we have the following finite-sample convergence bounds.
3. (a) Under Condition 2.1 (1), we have:

<!-- formula-not-decoded -->

- (b) Under Condition 2.1 (2), we have:
- (i) when α &lt; 1 /ϕ 2 :

<!-- formula-not-decoded -->

- (ii) when α = 1 /ϕ 2 :
- (iii) when α &gt; 1 /ϕ 2 :

<!-- formula-not-decoded -->

- (c) Under Condition 2.1 (3), we have:

<!-- formula-not-decoded -->

Remark. Recall that t δ ≤ log( C/σ )+log(1 /α ) log(1 /σ ) under Assumption 2.3. Therefore, we have t k ≤ ξ log( k + h )+log( C/ ( ασ )) log(1 /σ ) , which introduces an additional logarithmic factor in the bound.

In all cases of Theorem 2.1, we state the results as a combination of two terms. The first term is usually viewed as the 'bias", and it involves the error in the initial estimate x 0 (through the constant c 1 ), and the geometric decay term (for constant stepsize case). The second term is usually understood as the 'variance", and hence involves the constant c 2 , which represents the noise variance at x ∗ . This form of convergence bounds is common in related literature. See for example in [9] about the results for the popular SGD algorithm.

From Theorem 2.1, we see that constant stepsize is very efficient in driving the bias the zero, but cannot eliminate the variance. When using linear stepsize, the convergence bounds crucially depend on the value of α . In order to balance the bias and the variance terms to achieve the optimal convergence rate, we need to choose α &gt; 1 /ϕ 2 , and the resulting optimal convergence rate is roughly O (log( k ) /k ) . When using polynomial stepsize, although the convergence rate is the sub-optimal O (log( k ) /k ξ ) , it is more robust in the sense that it does not depend on α .

Switching focus, we now revisit the constants { ϕ i } 1 ≤ i ≤ 3 in Theorem 2.1, which as mentioned earlier, depend only on the contraction norm ‖·‖ c and the contraction factor β . In the following lemma, we consider two cases where ‖ · ‖ c = ‖ · ‖ 2 and ‖ · ‖ c = ‖ · ‖ ∞ . Both of them will be useful when we study convergence bounds of RL algorithms. The proof of the following result is presented in Appendix A.3.1.

Lemma 2.1. The following bounds hold regarding the constants { ϕ i } 1 ≤ i ≤ 3 .

- (1) When ‖ · ‖ c = ‖ · ‖ 2 , we have ϕ 1 ≤ 1 , ϕ 2 ≥ 1 -β , and ϕ 3 ≤ 228 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that when compared to ‖·‖ 2 -contraction, where the constant ϕ 3 is bounded by a numerical constant, the upper bound for ϕ 3 has an additional log( d ) 1 -β factor under the ‖ · ‖ ∞ -contraction. It was argued in [11] that in general such log( d ) factor is unimprovable.

## 2.3 Outline of the Proof

In this Section, we present the key ideas in proving Theorem 2.1. The detailed proof is presented in Appendix A. At a high level, we use a Lyapunov approach. That is, we find a function M : R d ↦→ R such that the following one-step contractive inequality holds:

<!-- formula-not-decoded -->

which then can be repeatedly used to derive finite-sample bounds of the SA algorithm (6).

## 2.3.1 Generalized Moreau Envelope as a Lyapunov Function

Inspired by [11], we will use M ( x ) = min u ∈ R d { 1 2 ‖ u ‖ 2 c + 1 2 θ ‖ x -u ‖ 2 p } as the Lyapunov function, where θ &gt; 0 and p ≥ 2 are tunable parameters. The function M ( · ) is called the Generalized Moreau Envelope, which is known to be a smooth approximation of the function 1 2 ‖ x ‖ 2 c , with smoothness parameter p -1 θ . See [11] for more details about using the Generalized Moreau Envelope as a Lyapunov function.

Using the smoothness property of M ( · ) and the update equation (6), we have for all k ≥ 0 :

<!-- formula-not-decoded -->

What remains to do is to bound the terms T 1 to T 4 . The term T 1 represents the expected update. We show that it is negative and is of the order O ( α k ) , hence giving the negative drift term in the target one-step contractive inequality (7). Using the assumption that { w k } is a martingale difference sequence and the tower property of conditional expectation, we show that the error term T 3 is indeed zero. Also, we show that the error term T 4 is of the size O ( α 2 k ) = o ( α k ) . The main challenge here is to control the error term T 2 , which arises due to the Markovian noise { Y k } .

## 2.3.2 Handling the Markovian Noise

To control the term T 2 , we need to carefully use a conditioning argument along with the geometric mixing of { Y k } . Specifically, we first show that the error is small when we replace x k by x k -t k in the term T 2 , where we recall that t k is the mixing time of the Markov chain { Y k } with precision α k . Now, consider the resulting term

<!-- formula-not-decoded -->

First taking expectation conditioning on x k -t k and Y k -t k , then we have

<!-- formula-not-decoded -->

Using the mixing time (cf. Definition 2.1) of { Y k } , we see that the difference between E [ F ( x k -t k , Y k ) | x k -t k , Y k -t k ] and ¯ F ( x k -t k ) (which can written as E µ [ F ( x, Y )] evaluated at x = x k -t k ) is of the size o (1) , hence concluding that ˜ T 2 = o ( α k ) by the tower property of conditional expectation.

This type of conditioning argument was first introduced in [5] [Section 4.4.1 The Case of Markov Noise] to establish the asymptotic convergence of linear SA with Markovian noise. Later, it was used more explicitly in [40] to study finite-sample bounds of linear SA, and in [12] to study nonlinear SA under a strong monotone condition. In this paper, we study nonlinear SA under arbitrary norm contraction, which is fundamentally different from [12, 40].

Using the upper bounds we have for the terms T 1 to T 4 in Eq. (8), we obtain the desired one-step contractive inequality (7). The rest of the proof follows by repeatedly using this inequality and evaluating the final expression for using different stepsize sequence { α k } .

In summary, we have stated finite-sample convergence bounds of a general stochastic approximation algorithm, and highlighted the key ideas in the proof. Next, we use Theorem 2.1 as a universal tool to study the convergence bounds of reinforcement learning algorithms.

## 3 Finite-Sample Guarantees of Reinforcement Learning Algorithms

We begin by introducing the underlying model for the RL problem. The RL problem is usually modeled by an MDP where the transition dynamics are unknown. In this work we consider an MDP consisting of a finite set of states S , a finite set of actions A , a set of unknown transition probability matrices that are indexed by actions { P a ∈ R |S|×|S| | a ∈ A} , a reward function R : S × A ↦→ R , and a discount factor γ ∈ (0 , 1) . We assume without loss of generality that the range of the reward function is [0 , 1] .

The goal in RL is to find an optimal policy π ∗ so that the cumulative reward received by using π ∗ is maximized. More formally, given a policy π , define its state-value function V π : S ↦→ R by

∣ for all s , where E π [ · ] means that the actions are selected according to the policy π . Then, a policy π ∗ is said to be optimal if V π ∗ ( s ) ≥ V π ( s ) for any state s and policy π . Under mild conditions, it was shown that such an optimal policy always exists [34].

<!-- formula-not-decoded -->

In RL, the problem of finding an optimal policy is called the control problem, which is solved with popular algorithms such as Q -learning [48]. A sub-problem is to find the value function of a given policy, which is called the prediction problem. This is solved with TD-learning and its variants such as TD ( λ ) , n -step TD [42], and the off-policy V-trace [17]. We next show that our SA results can be used to establish finite-sample convergence bounds of all the RL algorithms listed above, hence unifies the finite-sample analysis of value-based RL algorithms with asynchronous update.

## 3.1 Off-Policy Control: Q-Learning

Wefirst introduce the Q -learning algorithm proposed in [48]. Define the Q -function associated with a policy π by

<!-- formula-not-decoded -->

∣ for all ( s, a ) . Denote Q ∗ as the Q -function associated with an optimal policy π ∗ . (all optimal policies share the same optimal Q -function). The motivation of the Q -learning algorithm is based on the following result [5, 42]:

<!-- formula-not-decoded -->

The above result implies that knowing the optimal Q -function alone is enough to compute an optimal policy. The Q -learning algorithm is an iterative method to estimate the optimal Q -function. First, a sample trajectory { ( S k , A k ) } is collected using a suitable behavior policy π b . Then, initialize Q 0 ∈ R |S||A| . For each k ≥ 0 and state-action pair ( s, a ) , the iterate Q k ( s, a ) is updated by

<!-- formula-not-decoded -->

when ( s, a ) = ( S k , A k ) , and Q k +1 ( s, a ) = Q k ( s, a ) otherwise. Here Γ 1 ( Q k , S k , A k , S k +1 ) = R ( S k , A k )+ γ max a ′ ∈A Q k ( S k +1 , a ′ ) -Q k ( S k , A k ) is the temporal difference. To establish the finite-sample bounds of the Q -learning algorithm, we make the following assumption.

Assumption 3.1. The behavior policy π b satisfies π b ( a | s ) &gt; 0 for all ( s, a ) , and the Markov chain M S = { S k } induced by π b is irreducible and aperiodic.

The requirement that π b ( a | s ) &gt; 0 for all ( s, a ) is necessary even for the asymptotic convergence of Q -learning [44]. The irreducibility and aperiodicity assumption is also standard in related work [45, 46]. Since we work with finite-state MDPs, Assumption 3.1 on M S implies that M S has a unique stationary distribution, denoted by κ b ∈ ∆ |S| , and M S mixes at a geometric rate [29].

## 3.1.1 Properties of the Q-Learning Algorithm

To derive finite-sample guarantees of the Q -learning algorithm, we will follow the road map described in Section 1.2. We begin by formally remodeling the Q -learning algorithm. Let Y k = ( S k , A k , S k +1 ) for all k ≥ 0 . Note that the random process M Y = { Y k } is also a Markov chain, whose state-space is denoted by Y , and is finite. Define an operator F : R |S||A| ×Y ↦→ R |S||A| by

<!-- formula-not-decoded -->

/BD for all ( s, a ) . Then Q -learning algorithm (9) can be written by

<!-- formula-not-decoded -->

which is in the same form of the SA algorithm (6) with w k being identically equal to zero. Next, we establish the properties of the operator F ( · , · ) and the Markov chain { Y k } in the following proposition, which guarantees that Assumptions 2.1 - 2.3 are satisfied in the context of Q -learning.

Let N ∈ R |S||A|×|S||A| be the diagonal matrix with { κ b ( s ) π b ( a | s ) } ( s,a ) ∈S×A sitting on its diagonal. Let N min = min ( s,a ) κ b ( s ) π b ( a | s ) , which is positive under Assumption 3.1. The proof of the following proposition is presented in Appendix B.1.

Proposition 3.1. Suppose that Assumption 3.1 is satisfied, Then we have the following results.

- (1) The operator F ( · , · ) satisfies ‖ F ( Q 1 , y ) -F ( Q 2 , y ) ‖ ∞ ≤ 2 ‖ Q 1 -Q 2 ‖ ∞ and ‖ F ( 0 , y ) ‖ ∞ ≤ 1 for any Q 1 , Q 2 ∈ R |S||A| , and y ∈ Y .
- (2) The Markov chain M Y = { Y k } has a unique stationary distribution µ , and there exist C 1 &gt; 0 and σ 1 ∈ (0 , 1) such that max y ∈Y ‖ P k +1 ( y, · ) -µ ( · ) ‖ TV ≤ C 1 σ k 1 for any k ≥ 0 .
- (3) Define the expected operator ¯ F : R |S||A| ↦→ R |S||A| of F ( · , · ) by ¯ F ( Q ) = E Y ∼ µ [ F ( Q,Y )] . Then
4. (a) ¯ F ( · ) is explicitly given by ¯ F ( Q ) = N H ( Q ) + ( I -N ) Q , where H ( · ) is the Bellman operator for the Q -function.

(b) ¯ F ( · ) is a contraction mapping with respect to ‖·‖ ∞ , with contraction factor β 1 := 1 -N min (1 -γ ) .

(c) ¯ F ( · ) has a unique fixed-point Q ∗ .

As we see, the ( s, a ) -th entry of ¯ F ( Q ) is given by

<!-- formula-not-decoded -->

which captures the nature of performing asynchronous update as illustrated in Section 1.2. We shall refer to ¯ F ( · ) as the asynchronous Bellman operator in the following.

## 3.1.2 Finite-Sample Bounds of Q-Learning

Proposition 3.1 enables us to apply Theorem 2.1 and Lemma 2.1 (2) to the Q -learning algorithm. For ease of exposition, we only present the result of using constant stepsize, whose proof and the result for using diminishing stepsizes are presented in Appendix B.2.

Theorem 3.1. Consider { Q k } of Algorithm (9). Suppose that Assumption 3.1 is satisfied, and α k = α for all k ≥ 0 , where α is chosen such that αt α ( M Y ) ≤ c Q, 0 (1 -β 1 ) 2 log( |S||A| ) ( c Q, 0 is a numerical constant). Then we have for all k ≥ t α ( M Y ) :

<!-- formula-not-decoded -->

where c Q, 1 = 3( ‖ Q 0 -Q ∗ ‖ ∞ + ‖ Q 0 ‖ ∞ +1) 2 and c Q, 2 = 912 e (3 ‖ Q ∗ ‖ ∞ +1) 2 .

Remark. Recall that t α ( M Y ) is the mixing time of the Markov chain { Y k } with precision α . Using Proposition 3.1 (2), we see that t α ( M Y ) produces an additional log(1 /α ) factor in the bound.

Similar to Theorem 2.1, we view the first term on the RHS of the convergence bound as the the bias, and the second term as the variance. Since we are using constant stepsize, the bias term goes to zero geometrically fast while the variance is of the size O ( α log(1 /α )) .

Based on Theorem 3.1, we next derive the sample complexity of Q -learning. The proof of the following result is presented in Appendix B.4.

Corollary 3.2. In order to make E [ ‖ Q k -Q ∗ ‖ ∞ ] ≤ /epsilon1 , where /epsilon1 &gt; 0 is a given accuracy, the total number of samples required is of the size

<!-- formula-not-decoded -->

From Corollary 3.2, we see that the dependence on the accuracy /epsilon1 is O ( /epsilon1 -2 log 2 (1 //epsilon1 )) , and the dependence on the effective horizon is ˜ O ((1 -γ ) -5 ) . These two results match with known results in the literature [3]. The parameter N min is defined to be min s,a κ b ( s ) π b ( a | s ) , hence captures the quality of exploration of the behavior policy π b . Since N min ≥ 1 / |S||A| , we see that the best possible dependence on the size of the state-action space is ˜ O ( |S| 3 |A| 3 ) .

Remark. In the ˜ O ( · ) notation, we ignore all the polylogarithmic terms. Moreover, we upper bound ‖ Q ∗ ‖ ∞ by 1 / (1 -γ ) in deriving the sample complexity result.

## 3.1.3 Related Literature on Q-learning

The Q -learning algorithm [48] is perhaps one of the most well-known algorithms in the RL literature. The asymptotic convergence of Q -learning was established in [8, 21, 44], and the asymptotic convergence rate in [16, 43]. Beyond asymptotic behavior, finite-sample analysis of Q -learning was also thoroughly studied in the literature [3, 18, 22, 31, 35]. The state-of-the-art sample complexity for asynchronous Q -learning goes to [31] 1 , which has a better dependence on the size of the state-action space compared to this work. In addition to being a contractive SA, Q -learning has many other properties, such as the update equation being asynchronous, the iterates being uniformly bounded by a constant [19], which are used in [31] for their analysis. While our SA framework did not exploit these properties of Q -learning (which results in a sub-optimal sample complexity), it is a more general framework that enables us to study a wide variety of algorithms beyond Q -learning. A typical example is the V-trace algorithm studied in the previous section. Due to off-policy sampling, the iterates of V-trace do not admit a uniform upper bound.

## 3.2 Off-Policy Prediction: V-Trace

Wenext switch our focus to solving the prediction problem using TD-learning variants. Specifically, we first consider the V-trace algorithm for off-policy TD-learning [17]. Let π b be a behavior policy used to collect samples, π be the target policy (i.e., we want to evaluate V π ), and n be a positive integer. Let

<!-- formula-not-decoded -->

be the truncated importance sampling ratios at ( s, a ) , where ¯ ρ ≥ ¯ c ≥ 1 are the two truncation levels. Suppose a sequence of state-action pairs { ( S k , A k ) } is collected under the behavior policy π b . Then, with initialization V 0 ∈ R |S| , for each k ≥ 0 and s ∈ S , the V-trace algorithm updates the estimate V k ( s ) by

<!-- formula-not-decoded -->

when s = S k , and V k +1 ( s ) = V k ( s ) otherwise. Here Γ 2 ( V k , S i , A i , S i +1 ) = R ( S i , A i ) + γV k ( S i +1 ) -V k ( S i ) is the temporal difference. Note that when π b = π , and ¯ c = ¯ ρ = 1 , Eq. (10) reduces to the update equation for the on-policy n -step TD [42]. To establish finite-sample convergence bounds of Algorithm (10), we make the following assumption.

Assumption 3.2. The behavior policy π b satisfies for all s ∈ S : { a ∈ A | π ( a | s ) &gt; 0 } ⊆ { a ∈ A | π b ( a | s ) &gt; 0 } , and the Markov chain M S = { S k } induced by π b is irreducible and aperiodic.

The first part of Assumption 3.2 is call the coverage assumption, which states that, for any state, if it is possible to explore a specific action under the target policy π , then it is also possible to explore such an action under the behavior policy π b . This requirement is necessary for off-policy RL. The second part of Assumption 3.2 implies that { S k } has a unique stationary distribution, denoted by κ b ∈ ∆ |S| . Moreover, the Markov chain { S k } mixes at a geometric rate [29].

## 3.2.1 Properties of the V-Trace Algorithm

To establish the convergence bounds of the V-trace algorithm, similar to Q -learning, we first model the V-trace algorithm in the form of SA algorithm (6). For any k ≥ 0 , let Y k =

1 While the results in [31] are stated in terms of high probability bounds, due to the boundedness of Q -learning, their concentration bounds can be translated into a mean-square bound with the same sample complexity. See [30] for a proof.

( S k , A k , ..., S k + n -1 , A k + n -1 , S k + n ) . It is clear that { Y k } is also a Markov chain, whose state space is denoted by Y . Define an operator F : R |S| ×Y ↦→ R |S| by for all s ∈ S . Then the V-trace update equation (10) can be equivalently written by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under Assumptions 3.2, we next establish the properties of the operator F ( · ) and the Markov chain { Y k } , which allow us to call for our main results in Section 2. Before that, we need to introduce more notation in the following.

Notation. For any policy π , let P π ∈ R |S|×|S| be the transition probability matrix under policy π , i.e., P π ( s, s ′ ) = ∑ a ∈A π ( a | s ) P a ( s, s ′ ) . Also, we let R π ∈ R |S| be such that R π ( s ) = ∑ a ∈A π ( a | s ) R ( s, a ) . Let C,D ∈ R |S|×|S| be diagonal matrices such that

Let C min = min s ∈S C ( s ) and D min = min s ∈S D ( s ) . Note that we have 0 &lt; C min ≤ D min ≤ 1 under Assumption 3.2. Let K ∈ R |S|×|S| be a diagonal matrix with diagonal entries { κ b ( s ) } s ∈S , and let K min = min s ∈S κ b ( s ) . Define two policies π ¯ c and π ¯ ρ by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition 3.2. Under Assumptions 3.2, the V-trace algorithm (11) has the following properties:

- (1) The operator F ( · ) satisfies:
2. (a) ‖ F ( V 1 , y ) -F ( V 2 , y ) ‖ ∞ ≤ (2¯ ρ +1) η ( γ, ¯ c ) ‖ V 1 -V 2 ‖ ∞ for all V 1 , V 2 ∈ R |S| and y ∈ Y , where η ( γ, ¯ c ) = 1 -( γ ¯ c ) n 1 -γ ¯ c when γ ¯ c = 1 , and η ( γ, ¯ c ) = n when γ ¯ c = 1 .
3. (b) ‖ F ( 0 , y ) ‖ ∞ ≤ ¯ ρη ( γ, ¯ c ) for all y ∈ Y .

/negationslash

- (2) The Markov chain { Y k } has a unique stationary distribution, denoted by µ . Moreover, there exists C 2 &gt; 0 and σ 2 ∈ (0 , 1) such that max y ∈Y ‖ P k + n ( y, · ) -µ ( · ) ‖ TV ≤ C 2 σ k 2 for all k ≥ 0 .
- (3) Define the expected operator ¯ F : R |S| ↦→ R |S| of F ( · ) by ¯ F ( V ) = E Y ∼ µ [ F ( V, Y )] for all V ∈ R |S| . Then
3. (a) ¯ F ( · ) is explicitly given by
4. (b) ¯ F ( · ) is a contraction mapping with respect to ‖ · ‖ ∞ , with contraction factor

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (c) ¯ F ( · ) has a unique fixed-point V π ¯ ρ , which is the value function of the policy π ¯ ρ .

The proof of Proposition 3.2 is presented in Appendix C. Observe from Proposition 3.2 (3) that the asynchronous Bellman operator ¯ F ( · ) associated with the V-trace algorithm is a β 2 -contraction with respect to ‖ · ‖ ∞ . A similar contraction property for synchronous V-trace was shown in [17] and [11], but with a different contraction factor.

## 3.2.2 Finite-Sample Bounds of V-Trace

We here present the convergence bounds of V-trace for using constant stepsize, whose proof and the result for using diminishing stepsize are presented in Appendix C.

Theorem 3.3. Consider { V k } of Algorithm (10). Suppose that Assumption 3.2 is satisfied, and α k = α for all k ≥ 0 , where α is chosen such that α ( t α ( M S ) + n ) ≤ c V, 0 (1 -β 2 ) 2 (¯ ρ +1) 2 η 2 ( γ, ¯ c ) log( |S| ) ( c V, 0 is a numerical constant). Then we have for all k ≥ t α ( M S ) + n :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark. Similarly as in Q -learning, under Assumption 3.2, we can further bound the mixing time t α ( M S ) by L (log(1 /α ) + 1) , where L &gt; 0 is a constant that solely depends on the underlying Markov chain { ( S k , A k ) } .

The rate of convergence (geometric convergence with accuracy O ( α log(1 /α )) ) is similar to that of Q -learning. The truncation level ¯ ρ determines the limit point V π ¯ ρ . The truncation level ¯ c mainly controls the variance term. These observations agree with results in [11], where synchronous V-trace is studied. To formally characterize how the parameters of V-trace impact the convergence rate, we next derive the sample complexity bound.

When ¯ ρ = 1 / min s,a π b ( a | s ) ≥ max s,a π ( a | s ) /π b ( a | s ) , the bias due to introducing the truncation level ¯ ρ is eliminated and hence we have V π ¯ ρ = V π and also D min = 1 . In this case, based on Theorem 3.3, we have the following sample complexity bound, whose proof is identical to that of Corollary 3.2 and is omitted.

Corollary 3.4. When ¯ ρ = 1 / min s,a π b ( a | s ) , in order to make E [ ‖ V k -V π ‖ ∞ ] ≤ /epsilon1 , the number of samples required for the V-trace algorithm (10) is of the size

<!-- formula-not-decoded -->

Another term that arises in the sample complexity of V-trace is ˜ O ( n ¯ ρ 2 η ( γ, ¯ c ) 2 (1 -γC min ) 3 D 3 min (1 -( γC min ) n ) 3 ) , which is a consequence of performing n -step off-policy TD with truncated importance sampling ratios. The impact of the parameter n will be analyzed in detail in Section 3.3, where we study on-policy n -step TD and the efficiency of bootstrapping. We here focus on the two truncation levels ¯ c and ¯ ρ . Note that ¯ ρ = 1 / min s,a π b ( a | s ) ≥ 1 / |A| , hence the side effect of ensuring V π ¯ ρ = V π by choosing large enough ¯ ρ is to introduce at least a factor of |A| -2 in the sample complexity. This can also be viewed as a measure of the quality of exploration through the behavior policy. Therefore, the total dependence on the size of the state-action space is at least ˜ O ( |S| 3 |A| 2 ) . We want to point out that this lowest possible value may not be achievable since K min = 1 / |S| and min s,a π b ( a | s ) = 1 / |A| may not hold simultaneously.

We use the upper bound 1 / (1 -γ ) for ‖ V π ¯ ρ ‖ ∞ when deriving the sample complexity result. From Corollary 3.4, we see that the dependence on the accuracy /epsilon1 , the effective horizon 1 / (1 -γ ) , and the parameter K min that captures the quality of exploration are the same as Q -learning.

The dependence of the sample complexity on the truncation level ¯ c is through the term η ( γ, ¯ c ) . In view of the expression of the function η ( γ, ¯ c ) given in Proposition 3.2 (1), we see that to avoid an exponential factor of n we need to aggressively truncate the importance sampling ratios by choosing ¯ c &lt; 1 /γ .

## 3.2.3 Related Literature on V-trace

The V-trace algorithm was first proposed in [17] as an off-policy variant of n -step TD-learning. The key novelty in V-trace is that the two truncation levels ¯ c and ¯ ρ are introduced in the importance sampling ratios to separately control the bias and the variance. The asymptotic convergence of V-trace in the case where n = ∞ was established in [17]. As for finite-sample guarantees, [11] studies n -step V-trace with synchronous update. The main difference between the sample complexity of the asynchronous V-trace studied in this paper and the synchronous V-trace studied in [11] is that there is an additional factor of K -3 min in our bound (cf. Corollary 3.4), which captures the quality of exploration and is the key feature of asynchronous RL algorithms. Other algorithms that are closely related to V-trace are the off-Policy Q π ( λ ) [20], Tree-backup TB( λ ) [33], Retrace ( λ ) [32], and Q -trace [24].

## 3.3 On-Policy Prediction: n -Step TD

In this section, we study the convergence bounds of the on-policy n -step TD-learning algorithm, which can be viewed as a special case of the V-trace algorithm with π b = π and ¯ c = ¯ ρ = 1 . Therefore, one can directly apply Theorem 3.3 to this setting and obtain finite-sample bounds for n -step TD. However, we will show that due to on-policy sampling there are better properties (i.e., ‖·‖ 2 -contraction) of the n -step TD algorithm we can exploit, which enables us to obtain tighter bounds. Observe that in the case of on-policy n -step TD, the update equation (10) simplifies to:

<!-- formula-not-decoded -->

when s = S k , and V k +1 ( s ) = V k ( s ) otherwise, where

<!-- formula-not-decoded -->

is the n -step temporal difference.

An important idea in the n -step TD is to use the parameter n to adjust the bootstrapping effect. When n = 0 , Eq. (12) is the standard TD (0) update, which corresponds to extreme bootstrapping. When n = ∞ , Eq. (12) is the Monte Carlo method for estimating V π , which corresponds to no bootstrapping. A longstanding question in RL is about the efficiency of bootstrapping, i.e., the choice of n that leads to the optimal performance of the algorithm [42].

In the following sections, we will establish finite-sample convergence bounds of the n -step TD-learning algorithm. By evaluating the resulting sample complexity bound as a function of n , we provide theoretical insight into the bias-variance trade-off in terms of n , as well as an estimate of the optimal value of n . Similarly as in the previous sections, we make the following assumption.

Assumption 3.3. The Markov chain M S = { S k } induced by the target policy π is irreducible and aperiodic.

Since we are using on-policy sampling in n -step TD, the target policy must be explorative. Assumption 3.3 ensures this property, and also implies that { S k } has a unique stationary distribution (denoted by κ ∈ ∆ |S| ), and the geometric mixing property [29].

## 3.3.1 Properties of the n -Step TD-Learning Algorithm

To apply Theorem 2.1, we begin by rewriting the update equation (12) in the form of the SA algorithm studied in Section 2. Let a sequence { Y k } be defined by Y k = ( S k , A k , ..., S k + n -1 , A k + n -1 , S k + n ) for all

k ≥ 0 . It is clear that { Y k } is a Markov chain, whose state-space is denoted by Y and is finite. Define an operator F : R |S| ×Y ↦→ R |S| by

<!-- formula-not-decoded -->

/BD Then the n -step TD algorithm (12) can be equivalently written by

<!-- formula-not-decoded -->

We next establish the properties of the n -step TD algorithm in the following proposition, whose proof is presented in Appendix D. Let K ∈ R |S|×|S| be a diagonal matrix with diagonal entries { κ ( s ) } s ∈S , and let K min = min s ∈S κ ( s ) .

Proposition 3.3. Under Assumption 3.3, the n -step TD-learning algorithm (12) has the following properties.

- (1) The operator F ( · ) satisfies for all V 1 , V 2 ∈ R |S| and y ∈ Y :
2. (a) ‖ F ( V 1 , y ) -F ( V 2 , y ) ‖ 2 ≤ 2 ‖ V 1 -V 2 ‖ 2 .
3. (b) ‖ F ( 0 , y ) ‖ 2 ≤ 1 1 -γ .
- (2) The Markov chain { Y k } has a unique stationary distribution, denoted by µ . Moreover, there exists C 3 &gt; 0 and σ 3 ∈ (0 , 1) such that max y ∈Y ‖ P k + n ( y, · ) -µ ( · ) ‖ TV ≤ C 3 σ k 3 for all k ≥ 0 .
- (3) Define the expected operator ¯ F : R |S| ↦→ R |S| of F ( · ) by ¯ F ( V ) = E Y ∼ µ [ F ( V, Y )] for all V ∈ R |S| . Then
6. (a) ¯ F ( · ) is explicitly given by

<!-- formula-not-decoded -->

- (b) ¯ F ( · ) is a contraction mapping with respect to the /lscript p -norm ‖·‖ p for any p ∈ [1 , ∞ ] , with a common contraction factor

<!-- formula-not-decoded -->

- (c) ¯ F ( · ) has a unique fixed-point V π .

From Proposition 3.3, we see that the asynchronous Bellman operator ¯ F ( · ) associated with the on-policy n -step TD-learning algorithm is a β 3 -contraction with respect to ‖ · ‖ p for any p ∈ [1 , ∞ ] , which is a major difference compared to its off-policy variant V-trace. In particular, this implies that ¯ F ( · ) is a contraction with respect to the standard Euclidean norm ‖ · ‖ 2 . This is the property we are going to exploit in establishing finite-sample bounds of n -step TD in the next section.

To intuitively understand the ‖·‖ 2 -contraction property, recall a 'less known" property from [5, 45] that the n -step Bellman operator T n π ( · ) is a contraction operator with respect to the weighted /lscript 2 -norm ‖ · ‖ κ , with weights being the stationary distribution κ . Similar to Q -learning, the asynchronous Bellman operator ¯ F ( · ) is a convex combination of the identity operator I and the n -step Bellman operator T n π ( · ) , using the stationary distribution κ as weights. Therefore, due to this 'normalization", the asynchronous Bellman operator is a contraction mapping with respect to the unweighted /lscript 2 -norm.

## 3.3.2 Finite-Sample Bounds of n -Step TD

In this section, we use the ‖ · ‖ 2 -contraction property from Proposition 3.3 to derive finite-sample convergence bounds of Algorithm (12). Note that Lemma 2.1 (1) is applicable in this case. The proof of the following result is presented in Appendix D.

Theorem 3.5. Consider { V k } of Algorithm (12). Suppose that Assumption 3.3 is satisfied, and α k ≡ α with α chosen such that α ( t α ( M S ) + n ) ≤ ˆ c 0 (1 -β 3 ) ( ˆ c 0 is a numerical constant). Then we have for all k ≥ t α ( M S ) + n :

<!-- formula-not-decoded -->

where ˆ c 1 = ( ‖ V 0 -V π ‖ 2 + ‖ V 0 ‖ 2 +4) 2 and ˆ c 2 = 228(4(1 -γ ) ‖ V π ‖ 2 +1) 2 .

To analyze the impact of the parameter n , we begin by rewriting the convergence bounds in Theorem 3.5 focusing only on n -dependent terms. Using the explicit expression of the contraction factor β 3 , in the k -th iteration, the bias term is of the size (1 -Θ(1 -γ n )) k . Since the mixing time t α ( M S ) of the original Markov chain { S k } does not depend on n , the variance term is of the size O ( n/ (1 -γ n )) . Now we can clearly see that as n increases to infinity, the bias goes down while the variance goes up, thereby demonstrating a bias-variance trade-off in the n -step TD-learning algorithm.

To formally characterize how the parameters of the n -step TD algorithm impact its convergence rate and computing an estimate of the optimal choice of n , we next derive the sample complexity of n -step TD based on Theorem 3.5. The proof of the following result is identical to that of Corollary 3.2 and is omitted.

Corollary 3.6. In order to make E ][ ‖ V k -V π ‖ 2 ] ≤ /epsilon1 , the number of samples required for the n -step TDlearning algorithm (12) is of the size

<!-- formula-not-decoded -->

In light of the dependence on the parameter n , ˜ O ( n (1 -γ n ) -2 ) , the optimal choice of n can be estimated by minimizing the function n (1 -γ n ) -2 over all positive integers. By doing that, we obtain the following estimate:

Note that we use ‖ V π ‖ 2 ≤ |S| 1 / 2 / (1 -γ ) in deriving the sample complexity. Although the norms we used in the mean square distance are different for n -step TD and V-trace, since ‖ x ‖ ∞ ≤ ‖ x ‖ 2 for any x , we clearly see that on-policy n -step TD has a better sample complexity over off-policy V-trace. First of all, it enjoys a better dependency on the effective horizon (set n = 1 to see such dependence), which is ˜ O ((1 -γ ) -4 ) . In addition, since K min ≤ 1 / |S| , the dependency on K min is at most K -2 . 5 min for n -step TD while V-trace has K -3 min (cf. Corollary 3.4). The main reason for such an improvement in sample complexity is that we are able to exploit the ‖·‖ 2 -contraction of the corresponding asynchronous Bellman operator ¯ F ( · ) in n -step TD.

<!-- formula-not-decoded -->

where /floorleft x /ceilingright stands for the integer closest to x . This result implies that when the discount factor γ is small (specifically γ ≤ 1 /e ), there is not much improvement in using multi-step TD-learning over using single step TD-learning, and when the discount factor is large, using n -step TD-learning with n ∼ /floorleft 1 / log(1 /γ ) /ceilingright has provable improvement.

## 3.3.3 Related Literature on n -Step TD

The notion of using multi-step returns instead of only one-step return was introduced in [49]. See [42] [Chapter 7] for more details about n -step TD. The asymptotic convergence of n -step TD can be established using the general stochastic approximation algorithm under contraction assumption [5]. Regarding the choice of n , it was observed in empirical experiments that n -step TD (with a suitable choice of n ) usually outperforms TD (0) and Monte Carlo method [39, 42]. However, theoretical understanding to this phenomenon is not well established in the literature. We derive finite-sample convergence bounds of the n -step TD-learning algorithm as an explicit function of n . This requires us to compute the exact expression of the contraction factor β 3 of the asynchronous Bellman operator (Proposition 3.3 (3)), and the mixing time (Proposition 3.3 (2)).

## 3.4 On-Policy Prediction: TD ( λ )

We next consider the on-policy TD ( λ ) algorithm, which effectively uses a convex combination of all the multi-step temporal differences at each update. We begin by describing the TD ( λ ) algorithm for estimating the value function V π of a policy π . Suppose that we have collected a sample trajectory { ( S k , A k ) } using the policy π . Then, with initialization V 0 ∈ R |S| , for any λ ∈ (0 , 1) , the estimate V k is iteratively updated according to

<!-- formula-not-decoded -->

for all s ∈ S , where Γ 4 ( V k , S k , A k , S k +1 ) = R ( S k , A k ) + γV k ( S k +1 ) -V k ( S k ) is the temporal difference, and z k ( s ) = k i =0 ( γλ ) k -i { S i = s } is the eligibility trace [5, 42].

∑ /BD A key idea in the TD ( λ ) algorithm is to use the parameter λ to adjust the bootstrapping effect. When λ = 0 , Algorithm (13) becomes the standard TD (0) update, which is pure bootstrapping. Another extreme case is when λ = 1 . This corresponds to using pure Monte Carlo method. Theoretical understanding of the efficiency of bootstrapping is a long-standing open problem in RL [41].

In the following Section, we establish finite-sample convergence bounds of the TD ( λ ) algorithm. By evaluating the resulting bound as a function of λ , we provide theoretical insight into the bias-variance tradeoff in choosing λ . Similar to n -step TD, we make the following assumption.

Assumption 3.4. The Markov chain M S = { S k } induced by the target policy π is irreducible and aperiodic.

As a result of Assumption 3.4, the Markov chain { S k } has a unique stationary distribution, denoted by κ ∈ ∆ |S| , and the geometric mixing property [29].

## 3.4.1 Properties of the TD ( λ ) Algorithm

Unlike the previous algorithms we studied, the TD ( λ ) algorithm cannot be viewed as a direct variant of the SA algorithm (6). This is because of the geometric averaging induced by the eligibility trace in TD( λ ), which creates dependencies over the entire past trajectory. We overcome this difficulty by using an additional truncation argument, and separately handle the residual error due to truncation. For ease of exposition, we consider only using constant stepsize in the TD ( λ ) algorithm, i.e., α k = α for all k ≥ 0 .

For any k ≥ 0 , let Y k = ( S 0 , ..., S k , A k , S k +1 ) (which takes value in Y k := S k +2 × A ), and define a time-varying operator F k : R |S| ×Y k ↦→ R |S| by

<!-- formula-not-decoded -->

for all s ∈ S . Note that the sequence { Y k } is not a Markov chain since it has a time-varying state-space. Using the notations of { Y k } and F k ( · , · ) , we can rewrite the update equation of the TD ( λ ) algorithm by

<!-- formula-not-decoded -->

Although Eq. (14) is similar to the update equation for SA algorithm (6), since the sequence { Y k } is not a Markov chain and the operator F k ( · , · ) is time-varying, our Theorem 2.1 is not directly applicable.

To overcome this difficulty, let us carefully look at the operator F k ( · , · ) . Although F k ( V k , Y k ) depends on the whole trajectory of states visited before (through the term ∑ k i =0 ( γλ ) k -i /BD { S i = s } ), due to the geometric factor ( γλ ) k -i , the states visited during the early stage of the iteration are not important. Inspired by this observation, we define the truncated sequence { Y τ k } of { Y k } by Y τ k = ( S k -τ , ..., S k , A k , S k +1 ) for all k ≥ τ , where τ is a fixed non-negative integer. Note that the random process M Y = { Y τ k } is now a Markov chain, whose state-space is denoted by Y τ and is finite. Similarly, we define the truncated operator F τ k : R |S| ×Y τ ↦→ R |S| of F k ( · , · ) by

<!-- formula-not-decoded -->

for all s ∈ S . Using the above notation, we can further rewrite the update equation (14) by

<!-- formula-not-decoded -->

Now, we argue that when the truncation level τ is large enough, the last term on the RHS of the previous equation is negligible compared to the other two terms. In fact, we have the following result. See Appendix E for its proof.

Lemma 3.1. For all k ≥ 0 and τ ∈ [0 , k ] , denote y = ( s 0 , ..., s k , a k , s k +1 ) and y τ = ( s k -τ , ..., s k , a k , s k +1 ) . Then the following inequality holds for all V ∈ R |S| : ‖ F τ k ( V, y τ ) -F k ( V, y ) ‖ 2 ≤ ( γλ ) τ +1 1 -γλ (1 + 2 ‖ V ‖ 2 ) .

Lemma 3.1 indicates that the error term in Eq. (15) is indeed geometrically small. Suppose we ignore that error term. Then the update equation becomes V k +1 ≈ V k + α k ( F τ k ( V k , Y τ k ) -V k ) . Since the random process M Y = { Y τ k } is a Markov chain, once we establish the required properties for the truncated operator F τ k ( · , · ) , our SA results become applicable.

From now on, we will choose τ = min { k ≥ 0 : ( γλ ) k +1 ≤ α } ≤ log(1 /α ) log(1 / ( γλ )) , where α is the constant stepsize we use. This implies that the error term in Eq. (15) is of the size O ( α 2 ) . Under this choice of τ , we next investigate the properties of the operator F τ k ( · , · ) and the random process { Y τ k } in the following Proposition (See Appendix E for its proof). Let K ∈ R |S|×|S| be a diagonal matrix with diagonal entries { κ ( s ) } s ∈S , and let K min = min s ∈S κ ( s ) .

Proposition 3.4. Suppose Assumption 3.3 is satisfied. Then we have the following results.

- (1) For any k ≥ τ , the operator F τ k ( · , · ) satisfies ‖ F τ k ( V 1 , y ) -F τ k ( V 2 , y ) ‖ 2 ≤ 3 1 -γλ ‖ V 1 -V 2 ‖ 2 , and ‖ F τ k ( 0 , y ) ‖ 2 ≤ 1 1 -γλ for any V 1 , V 2 ∈ R |S| and y ∈ Y τ .
- (2) The Markov chain { Y τ k } k ≥ τ has a unique stationary distribution, denoted by µ . Moreover, there exists C 4 &gt; 0 and σ 4 ∈ (0 , 1) such that max y ∈Y τ ‖ P k + τ +1 ( y, · ) -µ ( · ) ‖ TV ≤ C 4 σ k 4 for all k ≥ 0 .
- (3) For any k ≥ τ , define the expected operator ¯ F τ k : R |S| ↦→ R |S| by ¯ F τ k ( V ) = E Y ∼ µ [ F τ k ( V, Y )] . Then

(a) ¯ F τ k ( · ) is explicitly given by

<!-- formula-not-decoded -->

- (b) ¯ F τ k ( · ) is a contraction mapping with respect to ‖·‖ p for any p ∈ [1 , ∞ ] , with a common contraction factor

<!-- formula-not-decoded -->

- (c) ¯ F τ k ( · ) has a unique fixed-point V π .

Similar to n -step TD, the truncated asynchronous Bellman operator ¯ F ( · ) associated with the TD ( λ ) algorithm is a contraction with respect to the /lscript p -norm ‖·‖ p for any 1 ≤ p ≤ ∞ , with a common contraction factor β 4 . This enables us to use our SA results along with Lemma 2.1 (1).

## 3.4.2 Finite-Sample Bounds of TD ( λ )

We now present the finite-sample convergence bound of the TD ( λ ) algorithm for using constant stepsize, where we exploit only the ‖ · ‖ 2 -contraction property from Proposition 3.4. The proof is presented in Appendix E.3.

Theorem 3.7. Consider { V k } of Algorithm (13). Suppose that Assumption 3.4 is satisfied and α k ≡ α with α chosen such that α ( t α ( M S ) + 2 τ + 1) ≤ ˜ c 0 (1 -β 4 )(1 -γλ ) 2 ( ˜ c 0 is a numerical constant). Then the following inequality holds for all k ≥ t α ( M S ) + 2 τ +1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark. Under Assumption 3.4, the mixing time t α ( M S ) is at most an affine function of log(1 /α ) . More importantly, it does not depend on the parameter λ .

The convergence rate of TD ( λ ) is similar to that of n -step TD. We here focus on the impact of the parameter λ . We begin by rewriting both the bias term and the variance term in the resulting convergence bound of Theorem 3.7 focusing only on λ -dependent terms. Then, the bias term is of the size (1 -Θ(1 / (1 -γλ ))) k while the variance term is between Θ(1 / (1 -γλ ) log(1 / ( γλ ))) and Θ(1 / (1 -γλ )) . Now observe that the bias term is in favor of large λ (i.e., less bootstrapping, more Monte Carlo) while the variance term is in favor of small λ (i.e., more bootstrapping, less Monte Carlo). This observation agrees with empirical results in the literature [23, 42]. Therefore, we demonstrate a bias-variance trade-off in choosing λ , which addresses one of the open problems in [41] on the efficiency of bootstrapping in RL.

## 3.4.3 Related Literature on TD ( λ )

The idea of using λ -return and eligibility traces was introduced and developed in [21, 49]. See [42] [Chapter 12] for more details. The convergence of TD ( λ ) was established in [14].

Regarding the parameter λ , empirical observations indicate that a properly chosen intermediate value of λ usually outperforms both TD (0) and TD (1) [39]. Theoretical justification of this observation is, to some extend, provided in [23], where they study a variant of the TD ( λ ) algorithm called phased TD. The

TD ( λ ) algorithm is often used along with function approximation in practice. The asymptotic convergence of TD ( λ ) with linear function approximation was established in [45]. More recently, [6, 40] established the finite-sample bounds of TD ( λ ) with linear function approximation by modeling the algorithm as a linear stochastic approximation with Markovian noise. The result of [6] indicates that TD ( λ ) in general outperforms TD (0) . However, [6] does not provide explicit trade-offs between the convergence bias and variance in choosing λ . Similarly, [40] does not have an explicit bound, and thus do not study bias-variance trade-off, which is what we did in this paper. To achieve that, we need to carefully characterize the contraction factor β 4 of the truncated Bellman operator ¯ F τ k ( · ) , as well as the mixing time of the truncated Markov chain { Y τ k } .

## 4 Conclusion

In this work, we provide a unified framework for establishing finite-sample convergence bounds of valuebased Reinforcement Learning algorithms. The key idea is to first remodel the RL algorithm as a Markovian SA associated with a contractive asynchronous Bellman operator, and then derive the convergence bounds of such SA algorithm using a Lyapunov-drift argument. Based on the universal result on Markovian SA, we derive finite-sample convergence guarantees of Q -learning for solving the control problem, and various TD-learning algorithms (e.g. off-policy V-trace, n -step TD, and TD ( λ ) ) for solving the prediction problem, where we also provide theoretical insight into the long-standing question about the efficiency of bootstrapping in RL.

## References

- [1] Banach, S. (1922). Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales. Fund. math , 3(1):133-181.
- [2] Beck, A. (2017). First-order methods in optimization , volume 25. SIAM.
- [3] Beck, C. L. and Srikant, R. (2013). Improved upper bounds on the expected error in constant step-size Q -learning. In 2013 American Control Conference , pages 1926-1931. IEEE.
- [4] Benveniste, A., Métivier, M., and Priouret, P. (2012). Adaptive algorithms and stochastic approximations , volume 22. Springer Science &amp; Business Media.
- [5] Bertsekas, D. P. and Tsitsiklis, J. N. (1996). Neuro-dynamic programming . Athena Scientific.
- [6] Bhandari, J., Russo, D., and Singal, R. (2018). A Finite Time Analysis of Temporal Difference Learning With Linear Function Approximation. In Conference On Learning Theory , pages 1691-1692.
- [7] Borkar, V. S. (2009). Stochastic approximation: a dynamical systems viewpoint , volume 48. Springer.
- [8] Borkar, V. S. and Meyn, S. P. (2000). The ODE method for convergence of stochastic approximation and reinforcement learning. SIAM Journal on Control and Optimization , 38(2):447-469.
- [9] Bottou, L., Curtis, F. E., and Nocedal, J. (2018). Optimization methods for large-scale machine learning. Siam Review , 60(2):223-311.
- [10] Chen, S., Devraj, A., Bernstein, A., and Meyn, S. (2020a). Accelerating Optimization and Reinforcement Learning with Quasi-Stochastic Approximation. Preprint arXiv:2009.14431 .

- [11] Chen, Z., Maguluri, S. T., Shakkottai, S., and Shanmugam, K. (2020b). Finite-Sample Analysis of Contractive Stochastic Approximation Using Smooth Convex Envelopes. Advances in Neural Information Processing Systems , 33.
- [12] Chen, Z., Zhang, S., Doan, T. T., Clarke, J.-P., and Maguluri, S. T. (2019). Finite-Sample Analysis of Nonlinear Stochastic Approximation with Applications in Reinforcement Learning. Preprint arXiv:1905.11425 .
- [13] Dann, C., Li, L., Wei, W., and Brunskill, E. (2019). Policy certificates: Towards accountable reinforcement learning. In International Conference on Machine Learning , pages 1507-1516. PMLR.
- [14] Dayan, P. and Sejnowski, T. J. (1994). TD( λ ) converges with probability 1 . Machine Learning , 14(3):295-301.
- [15] Devraj, A. M., Bušic, A., and Meyn, S. (2018). Zap meets momentum: Stochastic approximation algorithms with optimal convergence rate. Preprint arXiv:1809.06277 .
- [16] Devraj, A. M. and Meyn, S. (2017). Zap Q -learning. In Advances in Neural Information Processing Systems , pages 2235-2244.
- [17] Espeholt, L., Soyer, H., Munos, R., Simonyan, K., Mnih, V., Ward, T., Doron, Y., Firoiu, V., Harley, T., Dunning, I., et al. (2018). IMPALA: Scalable Distributed Deep-RL with Importance Weighted ActorLearner Architectures. In International Conference on Machine Learning , pages 1407-1416.
- [18] Even-Dar, E. and Mansour, Y. (2003). Learning rates for Q -learning. Journal of Machine Learning Research , 5(Dec):1-25.
- [19] Gosavi, A. (2006). Boundedness of iterates in Q -learning. Systems &amp; control letters , 55(4):347-349.
- [20] Harutyunyan, A., Bellemare, M. G., Stepleton, T., and Munos, R. (2016). Q( λ ) with Off-Policy Corrections. In International Conference on Algorithmic Learning Theory , pages 305-320. Springer.
- [21] Jaakkola, T., Jordan, M. I., and Singh, S. P. (1994). Convergence of stochastic iterative dynamic programming algorithms. In Advances in neural information processing systems , pages 703-710.
- [22] Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. (2018). Is Q -learning provably efficient? In Proceedings of the 32nd International Conference on Neural Information Processing Systems , pages 4868-4878.
- [23] Kearns, M. J. and Singh, S. P. (2000). Bias-Variance Error Bounds for Temporal Difference Updates. In COLT , pages 142-147. Citeseer.
- [24] Khodadadian, S., Chen, Z., and Maguluri, S. T. (2021). Finite-Sample Analysis of Off-Policy Natural Actor-Critic Algorithm. arXiv preprint arXiv:2102.09318 .
- [25] Kober, J., Bagnell, J. A., and Peters, J. (2013). Reinforcement learning in robotics: A survey. The International Journal of Robotics Research , 32(11):1238-1274.
- [26] Kushner, H. (2010). Stochastic approximation: a survey. Wiley Interdisciplinary Reviews: Computational Statistics , 2(1):87-96.
- [27] Kushner, H. J. and Clark, D. S. (2012). Stochastic approximation methods for constrained and unconstrained systems , volume 26. Springer Science &amp; Business Media.

- [28] Lan, G. (2020). First-order and Stochastic Optimization Methods for Machine Learning . Springer.
- [29] Levin, D. A. and Peres, Y. (2017). Markov chains and mixing times , volume 107. American Mathematical Soc.
- [30] Li, G., Cai, C., Chen, Y., Wei, Y., and Chi, Y. (2023). Is Q -learning minimax optimal? a tight sample complexity analysis. Operations Research .
- [31] Li, G., Wei, Y., Chi, Y., Gu, Y ., and Chen, Y . (2020). Sample complexity of asynchronous Q -learning: Sharper analysis and variance reduction. Preprint arXiv:2006.03041 .
- [32] Munos, R., Stepleton, T., Harutyunyan, A., and Bellemare, M. G. (2016). Safe and efficient off-policy reinforcement learning. In Proceedings of the 30th International Conference on Neural Information Processing Systems , pages 1054-1062.
- [33] Precup, D., Sutton, R. S., and Singh, S. P. (2000). Eligibility Traces for Off-Policy Policy Evaluation. In Proceedings of the Seventeenth International Conference on Machine Learning , pages 759-766.
- [34] Puterman, M. L. (1995). Markov decision processes: Discrete stochastic dynamic programming. Journal of the Operational Research Society , 46(6):792-792.
- [35] Qu, G. and Wierman, A. (2020). Finite-Time Analysis of Asynchronous Stochastic Approximation and Q -Learning. In Conference on Learning Theory , pages 3185-3205. PMLR.
- [36] Robbins, H. and Monro, S. (1951). A stochastic approximation method. The Annals of Mathematical Statistics , pages 400-407.
- [37] Ryu, E. K. and Boyd, S. (2016). Primer on monotone operator methods. Appl. Comput. Math , 15(1):343.
- [38] Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. (2017). Mastering the game of go without human knowledge. Nature , 550(7676):354.
- [39] Singh, S. P. and Sutton, R. S. (1996). Reinforcement learning with replacing eligibility traces. Machine learning , 22(1):123-158.
- [40] Srikant, R. and Ying, L. (2019). Finite-time error bounds for linear stochastic approximation and TD learning. In Conference on Learning Theory , pages 2803-2830.
- [41] Sutton, R. S. (1999). Open theoretical questions in reinforcement learning. In European Conference on Computational Learning Theory , pages 11-17. Springer.
- [42] Sutton, R. S. and Barto, A. G. (2018). Reinforcement learning: An introduction . MIT press.
- [43] Szepesvári, C. et al. (1997). The asymptotic convergence-rate of Q -learning. In NIPS , volume 10, pages 1064-1070. Citeseer.
- [44] Tsitsiklis, J. N. (1994). Asynchronous stochastic approximation and Q -learning. Machine learning , 16(3):185-202.
- [45] Tsitsiklis, J. N. and Van Roy, B. (1997). Analysis of temporal-difference learning with function approximation. In Advances in neural information processing systems , pages 1075-1081.

- [46] Tsitsiklis, J. N. and Van Roy, B. (1999). Average cost temporal-difference learning. Automatica , 35(11):1799-1808.
- [47] Wainwright, M. J. (2019). Stochastic approximation with cone-contractive operators: Sharp /lscript ∞ -bounds for Q -learning. Preprint arXiv:1905.06265 .
- [48] Watkins, C. J. and Dayan, P. (1992). Q -learning. Machine learning , 8(3-4):279-292.
- [49] Watkins, C. J. C. H. (1989). Learning from delayed rewards.

## Appendices

## A Proof of Theorem 2.1

We will state and prove a more general version of Theorem 2.1. To do that, we need to introduce more notation and explicitly specify the requirement for choosing the stepsize sequence { α k } .

Notation. Let g ( x ) = 1 2 ‖ x ‖ 2 s , where the norm ‖ · ‖ s is properly chosen so that the function g ( · ) is a smooth function with respect to the norm ‖ · ‖ s . That is, the function g ( · ) is convex, differentiable, and there exists L &gt; 0 such that g ( x 2 ) ≤ g ( x 1 ) + 〈∇ g ( x 1 ) , x 2 -x 1 〉 + L 2 ‖ x 1 -x 2 ‖ 2 s for any x 1 , x 2 ∈ R d . For example, /lscript p -norm with p ∈ [2 , ∞ ) works with L = p -1 [2]. Since we work with finite-dimensional space R d , there exist /lscript cs , u cs &gt; 0 such that /lscript cs ‖ · ‖ c ≤ ‖ · ‖ c ≤ u cs ‖ · ‖ s . Let θ &gt; 0 be chosen such that β 2 &lt; 1+ θ/lscript 2 cs 1+ θu 2 cs , which is always possible since β ∈ (0 , 1) . Denote

<!-- formula-not-decoded -->

which are the constants we used to state Theorem 2.1. Note that ϕ 2 ∈ (0 , 1) under our choice of θ .

Now we state the requirement in choosing the stepsizes { α k } . For simplicity, we use α i,j for ∑ j k = i α k . Condition A.1. The sequence { α k } is non-increasing and satisfies α k -t k ,k -1 ≤ min( ϕ 2 ϕ 3 A 2 , 1 4 A ) for all k ≥ t k .

We next state a more general version of Theorem 2.1. Recall that K = min { k : k ≥ t k } , which is well-defined under Assumption 2.3.

Theorem A.1. Consider { x k } generated by Algorithm (6). Suppose that Assumptions 2.1, 2.2, 2.3 and 2.4 are satisfied, and the stepsize sequence { α k } satisfies Condition A.1. Then we have the following results.

- (1) For any k ∈ [0 , K -1] , we have: ‖ x k -x ∗ ‖ 2 c ≤ c 1 almost surely.
- (2) For any k ≥ K , we have

<!-- formula-not-decoded -->

where c 1 = ( ‖ x 0 -x ∗ ‖ c + ‖ x 0 ‖ c + B/A ) 2 and c 2 = ( A ‖ x ∗ ‖ c + B ) 2 .

Once we have Theorem A.1, we can evaluate the bound when α k = α ( k + h ) ξ to get Theorem 2.1. This is presented in Appendix A.2. In Appendix A.2, we also show how Condition 2.1 is obtained from Condition A.1 and the explicit requirements on the thresholds ¯ c and ¯ h . We next present the proof of Theorem A.1.

## A.1 Proof of Theorem A.1

## A.1.1 Step One: Constructing a Valid Lyapunov Function

Let f ( x ) = 1 2 ‖ x ‖ 2 c . We will use the Generalized Moreau Envelope of f ( · ) with respect to g ( · ) :

<!-- formula-not-decoded -->

as the Lyapunov function to study Algorithm (6). We first summarize the properties of M θ,g f ( · ) in the following proposition, which was established in [11]. For simplicity, we will just write M ( · ) for M θ,g f ( · ) in the following unless we want to emphasize the dependence on the choices of θ and g ( · ) .

Proposition A.1. The function M ( x ) has the following properties.

- (1) M ( x ) is convex, and L θ -smooth with respect to ‖ · ‖ s . That is, M ( y ) ≤ M ( x ) + 〈∇ M ( x ) , y -x 〉 + L 2 θ ‖ x -y ‖ 2 s for all x, y ∈ R d .
- (2) There exists a norm, denoted by ‖ · ‖ m , such that M ( x ) = 1 2 ‖ x ‖ 2 m .
- (3) Let /lscript cm = (1 + θ/lscript 2 cs ) 1 / 2 and u cm = (1 + θu 2 cs ) 1 / 2 . Then it holds that /lscript cm ‖ · ‖ m ≤ ‖ · ‖ c ≤ u cm ‖ · ‖ m . Using Proposition A.1 and the update equation (6), we have for any k ≥ 0 :

<!-- formula-not-decoded -->

The term T 1 represents the expected update of the stochastic iterative algorithm (6), and is bounded in the following lemma, whose proof can be found in [11].

<!-- formula-not-decoded -->

As we have seen in Lemma A.1, the term T 1 provides us the desired negative drift, i.e., the -O ( α k ) term in the target one-step contractive inequality (7). What remains to do is to control all the error terms T 2 to T 4 in Eq. (17).

## A.1.2 Step Two: Bounding the Error Terms

We begin with the term T 2 . Since { w k } is a martingale difference sequence with respect to the filtration F k (cf. Assumption 2.4), while x k is measurable with respect to F k , we have by the tower property of conditional expectation that

<!-- formula-not-decoded -->

Next we analyze the error term T 3 , which is due to the Markovian noise { Y k } . We first decompose T 3 in the following way:

<!-- formula-not-decoded -->

To proceed, we need the following lemma, which allows us to control the difference between x k 1 and x k 2 when | k 1 -k 2 | is relatively small. The proof can be found in Appendix A.3.2.

Lemma A.2. Given non-negative integers k 1 ≤ k 2 satisfying α k 1 ,k 2 -1 ≤ 1 4 A , we have for all k ∈ [ k 1 , k 2 ] :

<!-- formula-not-decoded -->

Using the assumption that α k 1 ,k 2 -1 ≤ 1 4 A in the resulting inequality of Lemma A.2, we have the following corollary, which will also be frequently used in the derivation.

Corollary A.2. Under same conditions given in Lemma A.2, we have for all k ∈ [ k 1 , k 2 ] :

<!-- formula-not-decoded -->

Recall that we require α k -t k ,k -1 ≤ 1 4 A for all k ≥ t k in Condition A.1. Therefore, Lemma A.2 is applicable when k 1 = k -t k and k 2 = k -1 for any k ≥ t k .

Now we are ready to control the terms T 31 , T 32 , and T 33 in the following lemma. The terms T 31 and T 32 are controlled mainly by constantly applying Lemma A.2 and the Lipschitz property of the operator F ( · ) (cf. Assumptions 2.1). Bounding the term T 33 requires using the geometric mixing of the Markov chain { Y k } (cf. Assumption 2.3). The proof is presented in Appendix A.3.3.

Lemma A.3. The following inequalities hold for all k ≥ t k :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now that Lemma A.3 provides upper bounds on the terms T 31 , T 32 , and T 33 , using them in Eq. (18) and we have the following result.

Lemma A.4. The following inequality holds for all k ≥ t k :

<!-- formula-not-decoded -->

Lastly, we bound the error term T 4 in the following lemma, whose proof is provided in Appendix A.3.4.

<!-- formula-not-decoded -->

Nowwehave control on all the error terms T 1 to T 4 . Using them in Eq. (17), and we obtain the following result. The proof is presented in Appendix A.3.5

Lemma A.6. The following inequality holds for all k ≥ t k :

<!-- formula-not-decoded -->

Note that Lemma A.6 provides the desired one-step contractive inequality. We next repeatedly use Lemma A.6 to derive finite-sample convergence bounds of Algorithm (6). Using the constants { ϕ i } 1 ≤ i ≤ 3 defined in Eq. (16), then Lemma A.6 reads:

<!-- formula-not-decoded -->

Since α k -t k ,k -1 ≤ ϕ 2 / ( ϕ 3 A 2 ) for all k ≥ K (cf. Condition A.1), we have by the previous inequality that

<!-- formula-not-decoded -->

for all k ≥ K . Recursively using the previous inequality and we have for any k ≥ K :

<!-- formula-not-decoded -->

According to Condition A.1, we also have α 0 ,k -1 ≤ 1 / (4 A ) for any k ∈ [0 , K ] . Using Corollary A.2 one more time and we have for any k ∈ [0 , K ] :

<!-- formula-not-decoded -->

This proves Theorem A.1 (1). Since the previous inequality implies E [ ‖ x K -x ∗ ‖ 2 c ] ≤ c 1 , we obtain for all k ≥ K :

<!-- formula-not-decoded -->

This proves Theorem A.1 (2).

## A.2 Finite-Sample Convergence Bounds for Using Various Stepsizes

We next proceed to prove Theorem 2.1 by evaluating the convergence bounds in Theorem A.1 when the stepsize sequence is chosen by α k = α ( k + h ) ξ , where α, h &gt; 0 and ξ ∈ (0 , 1) . We begin by restating Theorem 2.1 in full details.

Theorem A.3. Consider { x k } of Algorithm (6). Suppose that Assumptions 2.1, 2.2, 2.3 and 2.4 are satisfied. Then we have the following results.

- (1) When k ∈ [0 , K -1] , we have ‖ x k -x ∗ ‖ 2 c ≤ c 1 almost surely.
- (2) When k ≥ K , we have the following finite-sample convergence bounds.
3. (a) Let ¯ α ∈ (0 , 1) be chosen such that αt α ≤ min( ϕ 2 ϕ 3 A 2 , 1 4 A ) for all α ∈ (0 , ¯ α ) . Then when α k ≡ α ∈ (0 , ¯ α ) , we have for all k ≥ t α :

<!-- formula-not-decoded -->

- (b) When α k = α k + h , for any α &gt; 0 , let ¯ h be chosen such that α 0 ,K -1 ≤ min( ϕ 2 ϕ 3 A 2 , 1 4 A ) for all h ≥ ¯ h . Then
- (i) When α &lt; 1 /ϕ 2 , we have for all k ≥ K :

<!-- formula-not-decoded -->

- (ii) When α = 1 /ϕ 2 , we have for all k ≥ K :

<!-- formula-not-decoded -->

- (iii) When α &gt; 1 /ϕ 2 , we have for all k ≥ K :

<!-- formula-not-decoded -->

- (c) When α k = α ( k + h ) ξ , for any ξ ∈ (0 , 1) and α &gt; 0 , let ¯ h be chosen such that ¯ h ≥ [2 ξ/ ( ϕ 2 α )] 1 / (1 -ξ ) and α 0 ,K -1 ≤ min( ϕ 2 ϕ 3 A 2 , 1 4 A ) for any h ≥ ¯ h . Then we have for all k ≥ K :

<!-- formula-not-decoded -->

Proof of Theorem A.3. (1) Theorem A.3 (1) directly follows from Theorem A.1 (1).

- (2) (a) When using constant stepsize α , it is clear that Condition A.1 is satisfied when αt α ≤ min( ϕ 2 ϕ 3 A 2 , 1 4 A ) . We next verify the existence of such threshold ¯ α . Note that we have by definition of t α and Assumption 2.3 that

<!-- formula-not-decoded -->

It follows that lim α → 0 αt α = 0 . Hence there exists ¯ α ∈ (0 , 1) such that Condition A.1 is satisfied for all α ∈ (0 , ¯ α ) , which is stated in Condition 2.1 (1). We next evaluate Eq. (19). When α k ≡ α , we have for all k ≥ t α :

<!-- formula-not-decoded -->

This proves Theorem A.3 (2) (a).

- (b) Consider the case where α k = α k + h . We first verify the existence of the threshold ¯ h . We begin by comparing α k -t k with α k . Using Assumption 2.3 and we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, there exists ¯ h 1 &gt; 0 such that α k -t k ≤ 2 α k holds for any k ≥ t k when h ≥ ¯ h 1 . Now consider the requirement stated in Condition A.1. Using the fact that { α k } is non-increasing, we have

<!-- formula-not-decoded -->

Hence there exists ¯ h 2 &gt; 0 such that α k -t k ,k -1 ≤ min( ϕ 2 ϕ 3 A 2 , 1 4 A ) holds for any k ≥ t k when h ≥ ¯ h 2 . Now choosing ¯ h = max( ¯ h 1 , ¯ h 2 ) , Condition A.1 is satisfied. This is stated in Condition 2.1 (2). Furthermore, by construction we have α k -t k ≤ 2 α k for any k ≥ t k . We next evaluate the RHS of Eq. (19) in the following lemma, whose proof is presented in Appendix A.3.6.

Lemma A.7. The following inequality hold for all k ≥ K :

<!-- formula-not-decoded -->

This proves Theorem A.3 (2) (b).

- (c) Now we consider using α k = α ( k + h ) ξ , where ξ ∈ (0 , 1) and α, h &gt; 0 . Using the same line of proof as in the previous section, one can show that for any ξ ∈ (0 , 1) and α &gt; 0 , there exists ¯ h &gt; 0 such that Condition A.1 is satisfied for all h ≥ ¯ h . Furthermore, we assume without loss of generality that α k -t k ≤ 2 α k for all k ≥ t k and ¯ h ≥ [2 ξ/ ( ϕ 2 α )] 1 / (1 -ξ ) . We next evaluate the RHS of Eq. (19) in the following lemma, whose proof is presented in Appendix A.3.7.

It follows that

Lemma A.8. The following inequality hold for all k ≥ K :

<!-- formula-not-decoded -->

This proves Theorem A.3 (2) (c).

## A.3 Proof of Technical Lemmas

## A.3.1 Proof of Lemma 2.1

- (1) When ‖ · ‖ c = ‖ · ‖ 2 , we choose θ = 1 and g ( x ) = 1 2 ‖ x ‖ 2 2 . It follows that L = 1 and u cs = /lscript cs = 1 . Therefore, we have by definition (16) that ϕ 1 = 1 , ϕ 2 = 1 -β , and ϕ 3 = 228 .
- (2) Recall the definition of { ϕ i } 1 ≤ i ≤ 3 in Eq. (16). When ‖ · ‖ c = ‖ · ‖ ∞ , we choose θ = ( 1+ β 2 β ) 2 -1 and g ( x ) = 1 2 ‖ x ‖ 2 p with p = 2log( d ) , where d is the dimension of the iterates x k . It follows that L = p -1 ≤ 2log( d ) [2], u cs = 1 , and /lscript cs = 1 /d 1 /p = 1 / √ e . Therefore, we have

<!-- formula-not-decoded -->

## A.3.2 Proof of Lemma A.2

We first show that under Assumption 2.1, the size of ‖ F ( x, y ) ‖ c and ‖ ¯ F ( x ) ‖ c can grow at most affinely in terms of ‖ x ‖ c . Using Triangle inequality, we have

<!-- formula-not-decoded -->

where the last inequality follows from Assumption 2.1. It follows that

<!-- formula-not-decoded -->

Furthermore, we have by Jensen's inequality and the convexity of norms that

<!-- formula-not-decoded -->

The previous two inequalities will be frequently used in the derivation here after. Now we proceed to prove Lemma A.2. For any k ∈ [ k 1 , k 2 -1] , using Triangle inequality, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the previous inequality is equivalent to

<!-- formula-not-decoded -->

which implies for all k ∈ [ k 1 , k 2 ] :

<!-- formula-not-decoded -->

Using the fact that 1 + x ≤ e x ≤ 1 + 2 x for all x ∈ [0 , 1 / 2] , we have when α k 1 ,k 2 -1 ≤ 1 4 A :

<!-- formula-not-decoded -->

It follows that for all k ∈ [ k 1 , k 2 ] :

<!-- formula-not-decoded -->

Using the previous inequality in Eq. (20) and we have for any k ∈ [ k 1 , k 2 -1] :

<!-- formula-not-decoded -->

Hence, we have for any k ∈ [ k 1 , k 2 ] :

<!-- formula-not-decoded -->

Since α k 1 ,k -1 ≤ α k 1 ,k 2 -1 when k ∈ [ k 1 , k 2 ] , we obtain the first claimed inequality:

<!-- formula-not-decoded -->

Now for the second claimed inequality, since

<!-- formula-not-decoded -->

we have ‖ x k 2 -x k 1 ‖ c ≤ 4 α k 1 ,k 2 -1 ( A ‖ x k 2 ‖ c + B ) . Therefore, we have for any k ∈ [ k 1 , k 2 ] :

<!-- formula-not-decoded -->

which is the second claimed inequality.

## A.3.3 Proof of Lemma A.3

- (1) For the term T 31 , using Hölder's inequality and we have

<!-- formula-not-decoded -->

where ‖·‖ ∗ s denotes the dual norm of ‖·‖ s . Wefirst control the term ‖∇ M ( x k -x ∗ ) -∇ M ( x k -t k -x ∗ ) ‖ ∗ s . Recall that an equivalent definition of a convex function h ( x ) been L - smooth with respect to norm ‖ · ‖ is that

<!-- formula-not-decoded -->

where ‖ · ‖ ∗ is the dual norm of ‖ · ‖ [2]. Therefore, since M ( x ) is L θ -smooth with respect to ‖ · ‖ s , we have

<!-- formula-not-decoded -->

where the last line follows from Lemma A.2 and Triangle inequality.

Wenext control the term ‖ F ( x k , Y k ) -¯ F ( x k ) ‖ c . Using Assumptions 2.1, 2.2, and the fact that ¯ F ( x ∗ ) = x ∗ , we have

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

- (2) Consider the term T 32 . Using Hölder's inequality and we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the term ‖∇ M ( x k -t k -x ∗ ) ‖ ∗ s , we have

<!-- formula-not-decoded -->

where the last line follow from Corollary A.2. For the term ‖ F ( x k , Y k ) -F ( x k -t k , Y k ) + ¯ F ( x k -t k ) -¯ F ( x k ) ‖ c , using Assumptions 2.1 and 2.2 and we obtain

<!-- formula-not-decoded -->

where in the last line we used Lemma A.2. It follows that

<!-- formula-not-decoded -->

- (3) Consider the term T 33 . We first take expectation conditioning on x k -t k and Y k -t k to obtain

<!-- formula-not-decoded -->

For the term ‖∇ M ( x k -t k -x ∗ ) ‖ ∗ s , we have from Eq. (22) that

<!-- formula-not-decoded -->

For the term ‖ E [ F ( x k -t k , Y k ) | x k -t k , Y k -t k ] -¯ F ( x k -t k ) ‖ c , using the geometric mixing of the Markov chain { Y k } (cf. Assumption 2.3), we have

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

Taking the total expectation on both sides of the previous inequality yields the desired result.

## A.3.4 Proof of Lemma A.5

Using Proposition A.1 (2), Assumption 2.1, and Assumption 2.4 (2), we have

<!-- formula-not-decoded -->

## A.3.5 Proof of Lemma A.6

Using the constants { ϕ i } 1 ≤ i ≤ (cf. Eq. (16)) and Lemmas A.1, A.4, and A.5 in Eq. (17) and we have for all k ≥ t k :

<!-- formula-not-decoded -->

## A.3.6 Proof of Lemma A.7

We first simplify the RHS of Eq. (19) using α k = α k + h . Since we have chosen h such that α k -t k ,k -1 ≤ 2 α k for any k ≥ t k , Eq. (19) implies

For the term E 1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now consider the term E 2 . Similarly we have

The result then follows from using the upper bounds we obtained for the terms E 1 and E 2 in inequality (23).

<!-- formula-not-decoded -->

## A.3.7 Proof of Lemma A.8

When α k = α ( k + h ) ξ , similarly we have from Eq. (19) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As for the term E 2 , we will show by induction that E 2 ≤ 2 α ϕ 2 1 ( k + h ) ξ for all k ≥ 0 . Consider a sequence { u k } k ≥ 0 (with u 0 = 0 ) defined by

<!-- formula-not-decoded -->

It can be easily verified that u k = E 2 . Since u 0 = 0 ≤ 2 α ϕ 2 1 h ξ , we have the base case. Now suppose u k ≤ 2 α ϕ 2 1 ( k + h ) ξ for some k &gt; 0 . Consider u k +1 , and we have

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

where we used (1 + 1 x ) x &lt; e for all x &gt; 0 and e x ≥ 1 + x for all x ∈ R . Therefore, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line follows from h ≥ ¯ h ≥ [2 ξ/ ( ϕ 2 α )] 1 / (1 -ξ ) . The induction is now complete, and we have E 2 ≤ 2 α ϕ 2 1 ( k + h ) ξ for all k ≥ 0 . Using the upper bounds we obtained for the terms E 1 and E 2 in inequality (24) and we have the desired result.

## B Q-Learning

## B.1 Proof of Proposition 3.1

- (1) For any Q 1 , Q 2 ∈ R |S||A| and y ∈ Y , we have

<!-- formula-not-decoded -->

Similarly, for any y ∈ Y , we have

- (2) It is clear from Assumption 3.1 that { Y k } has a unique stationary distribution, denoted by µ . Moreover, we have µ ( s, a, s ′ ) = κ b ( s ) π b ( a | s ) P a ( s, s ′ ) for any ( s, a, s ′ ) ∈ Y . Consider the second claim. Using the definition of total variation distance, we have for all k ≥ 0 :

/negationslash

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C 1 &gt; 0 and σ 1 ∈ (0 , 1) are constants. Note that the last line of the previous inequality follows from Assumption 3.1.

- (3) (a) Using the Markov property, we have for any Q ∈ R |S||A| and ( s, a ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where H : R |S||A| ↦→ R |S||A| is the Bellman's optimality operator defined by

<!-- formula-not-decoded -->

for any ( s, a ) . Now use the definition of the matrix N and we have ¯ F ( Q ) = N H ( Q ) + ( I -N ) Q .

- (b) Since it is well-known that the Bellman's optimality operator H ( · ) is a γ -contraction with respect to ‖ · ‖ ∞ , we have for any Q 1 , Q 2 ∈ R |S||A| :

<!-- formula-not-decoded -->

Therefore, the operator ¯ F ( · ) is a contraction mapping with respect to ‖·‖ ∞ , with contraction factor β 1 = 1 -N min (1 -γ ) .

- (c) It is enough to show that Q ∗ is a fixed-point of ¯ F ( · ) , the uniqueness part follows from ¯ F ( · ) being a contraction [1]. Using the fact that H ( Q ∗ ) = Q ∗ , we have

<!-- formula-not-decoded -->

## B.2 Proof of Theorem 3.1

Since the contraction norm is ‖ · ‖ ∞ , Lemma 2.1 (2) is applicable. To apply Theorem 2.1, we first identify the corresponding constants using Proposition 3.1 in the following:

<!-- formula-not-decoded -->

Now we apply Theorem 2.1 (2) (a). When α k ≡ α with α chosen such that

<!-- formula-not-decoded -->

we have for all k ≥ t α ( M Y ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c Q, 1 = 3( ‖ Q 0 -Q ∗ ‖ ∞ + ‖ Q 0 ‖ ∞ +1) 2 and c Q, 2 = 912 e (3 ‖ Q ∗ ‖ ∞ +1) 2 .

## B.3 Q-Learning with Diminishing Stepsizes

We next present the finite-sample bounds for Q -learning with diminishing stepsizes, whose proof follows by directly applying Theorem 2.1 (2) (b) and (2) (c) and hence is omitted.

Theorem B.1. Consider { Q k } of Algorithm (9). Suppose that Assumption 3.1 is satisfied, then we have the following results.

- (1) (a) When α k = α k + h with α = 1 1 -β 1 and properly chosen h , there exists K ′ 1 &gt; 0 such that the following inequality holds for all k ≥ K ′ 1 :

<!-- formula-not-decoded -->

where c ′ Q, 1 = 3( ‖ Q 0 -Q ∗ ‖ ∞ + ‖ Q 0 ‖ ∞ +1) 2 and c ′ Q, 2 = 3648 e (3 ‖ Q ∗ ‖ ∞ +1) 2

- (b) When α k = α k + h with α = 2 1 -β 1 and properly chosen h , there exists K ′ 1 &gt; 0 such that the following inequality holds for all k ≥ K ′ 1 :

<!-- formula-not-decoded -->

- (c) When α k = α k + h with α = 4 1 -β 1 and properly chosen h , there exists K ′ 1 &gt; 0 such that the following inequality holds for all k ≥ K ′ 1 :

<!-- formula-not-decoded -->

- (2) When α k = α ( k + h ) ξ with ξ ∈ (0 , 1) , α &gt; 0 , and properly chosen h , there exists K ′ 1 &gt; 0 such that the following inequality holds for all k ≥ K ′ 1 :

<!-- formula-not-decoded -->

## B.4 Proof of Corollary 3.2

Wewill derive a more general result, which implies Corollary 3.2. Suppose we have a non-negative sequence { z k } and the following bound:

<!-- formula-not-decoded -->

where τ 1 ∈ (0 , 1) , τ 2 &gt; 0 , and t α ≤ L (log(1 /α ) + 1) for some L &gt; 0 . Then, in order for z k ≤ /epsilon1 2 , in view of the term τ 2 αt α , we need

<!-- formula-not-decoded -->

Using the bound of α in the term (1 -τ 1 α ) k z 0 , we have

<!-- formula-not-decoded -->

Using this result along with Jensen's inequality in the finite-sample bound of Q -learning proves Corollary 3.2.

## C V-Trace

## C.1 Proof of Proposition 3.2

- (1) Using the definition of F ( V, y ) , we have for any V 1 , V 2 ∈ R |S| , y ∈ Y , and s ∈ S :

/negationslash

It follows that ‖ F ( V 1 , y ) -F ( V 2 , y ) ‖ ∞ ≤ (2¯ ρ +1) η ( γ, ¯ c ) ‖ V 1 -V 2 ‖ ∞ . For any y ∈ Y and s ∈ S , we have

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

It follows that ‖ F ( 0 , y ) ‖ ∞ ≤ η ( γ, ¯ c ) .

- (2) The proof is identical to that of Proposition 3.1 (2).
- (3) (a) Using the definition of ¯ F ( · ) , we have for any V ∈ R |S| and s ∈ S :

<!-- formula-not-decoded -->

For any 0 ≤ i ≤ n -1 , we have by the Markov property and the tower property of conditional expectation that

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

(b) For any V 1 , V 2 ∈ R |S| , we have

<!-- formula-not-decoded -->

For simplicity of notation, denote G = I -K ∑ n -1 i =0 ( γCP π ¯ c ) i D ( I -γP π ¯ ρ ) . To evaluate the /lscript ∞ -norm of G , we first show that G has non-negative entries. Note that G can be equivalently written by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In view of Eq. (26), it remains to show that the matrix DP π ¯ ρ -CP π ¯ c D has non-negative entries. For any s, s ′ ∈ S , we have

<!-- formula-not-decoded -->

where the last line follows from ¯ c ≤ ¯ ρ .

Now since the matrix G has non-negative entries, we have

<!-- formula-not-decoded -->

- It follows that ∥ ∥ ¯ F ( V 1 ) -¯ F ( V 2 ) ∥ ∥ ∞ ≤ ‖ G ‖ ∞ ‖ V 1 -V 2 ‖ ∞ ≤ β 2 ‖ V 1 -V 2 ‖ ∞ . (c) It is enough to show that V π ¯ ρ is a fixed-point of ¯ F , the uniqueness follows from ¯ F being a contraction operator. Using the Bellman equation V π ¯ ρ = R π ¯ ρ + γP π ¯ ρ V π ¯ ρ , we have by Eq. (25) that

<!-- formula-not-decoded -->

## C.2 Proof of Theorem 3.3

We will apply Theorem 2.1 and Lemma 2.1 (2) to the V-trace algorithm. We begin by identifying the constants:

<!-- formula-not-decoded -->

Now we apply Theorem 2.1 (2) (a). When α k = α for all k ≥ 0 , where α is chosen such that

<!-- formula-not-decoded -->

we have for all k ≥ t α + n :

<!-- formula-not-decoded -->

where c V, 1 = 3( ‖ V 0 -V π ¯ ρ ‖ ∞ + ‖ V 0 ‖ ∞ +1) 2 and c V, 2 = 3648 e ( ‖ V π ¯ ρ ‖ ∞ +1) 2 .

## C.3 V-trace with Diminishing Stepsizes

We here only present using linear stepsize that achieves the optimal convergence rate (Theorem 2.1 (2) (b) (iii)).

Theorem C.1. Consider { V k } of Algorithm (10). Suppose Assumption 3.2 is satisfied and α k = α k + h with α = 4 1 -β 2 and properly chosen h . Then there exists K ′ 2 &gt; 0 such that the following inequality holds for all k ≥ K ′ 2 :

<!-- formula-not-decoded -->

where c ′ V, 1 = 3 ‖ V 0 -V π ¯ ρ ‖ ∞ + ‖ V 0 ‖ ∞ +1) 2 and c ′ V, 2 = 233472 e 2 ( ‖ V π ¯ ρ ‖ ∞ +1) 2 .

Proof of Theorem C.1. The corresponding constants have been identified in the proof of Theorem 3.3. Now apply Theorem 2.1 (2) (c). When α k = α k + h with α = 4 1 -β 2 and properly chosen h , there exists K ′ 2 &gt; 0 such that we have for all k ≥ K ′ 2 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D n -step TD

## D.1 Proof of Proposition 3.3

- (1) (a) For any V 1 , V 2 ∈ R |S| and y ∈ Y , we have

<!-- formula-not-decoded -->

- (b) For any y ∈ Y , we have

It follows that ‖ ‖ ≤ 1 -γ

<!-- formula-not-decoded -->

- (2) The proof is identical to that of Proposition 3.1 (2).
- (3) (a) Since n -step TD is a special case of V-trace, we can directly apply Proposition 3.2 (3) (a) here. Observe that when π = π b and ¯ c = ¯ ρ = 1 , we have C = D = I and P π ¯ c = P π ¯ ρ = P π . Hence we have

<!-- formula-not-decoded -->

- (b) For any V 1 , V 2 ∈ R |S| and p ≥ 1 , we have

∥ ∥ For simplicity of notation, we denote G = I -K ∑ n -1 i =0 ( γP π ) i ( I -γP π ) . Since G has non-negative entries (established in the proof of Proposition 3.2 (3) (b)), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∥ ∥ Moreover, using the fact that κ is the stationary distribution of P π (i.e., κ /latticetop P π = κ /latticetop ), we have

To proceed, we need the following lemma.

<!-- formula-not-decoded -->

LemmaD.1. Let G ∈ R d × d be a matrix with non-negative entries. Then we have for all p ∈ [1 , ∞ ] :

<!-- formula-not-decoded -->

Proof of Lemma D.1. The result clearly holds when p = 1 or p = ∞ . Now consider p ∈ (1 , ∞ ) . Using the definition of induced matrix norm, we have for any x = 0 :

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

It follows that ‖ G ‖ p ≤ ‖ G ‖ 1 /p 1 ‖ G ‖ 1 -1 /p ∞ .

Using Lemma D.1 and we have

<!-- formula-not-decoded -->

Therefore, we have ‖ ¯ F ( V 1 ) -¯ F ( V 2 ) ‖ 2 ≤ β 3 ‖ V 1 -V 2 ‖ 2 . Hence the operator ¯ F ( · ) is a contraction mapping with respect to ‖ · ‖ 2 , with contraction factor β 3 .

(c) The proof is identical to that of Proposition 3.2 (3) (c).

## D.2 Proof of Theorem 3.5

We will apply Theorem and Lemma 2.1 (1) to the n -step TD algorithm. We begin by identifying the constants:

<!-- formula-not-decoded -->

Now apply Theorem 2.1 (2) (a). When α k = α for all k ≥ 0 , where α is chosen such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have for all k ≥ t α ( M S ) + n :

<!-- formula-not-decoded -->

where ˆ c 1 = ( ‖ V 0 -V π ‖ 2 + ‖ V 0 ‖ 2 +4) 2 and ˆ c 2 = 228(4(1 -γ ) ‖ V π ‖ 2 +1) 2 .

## D.3 n -Step TD with Diminishing Stepsizes

For n -step TD with diminishing stepsize, we here only present the result for using linear diminishing stepsize that achieves the optimal convergence rate (Theorem 2.1 (2) (b) (iii)).

Theorem D.1. Consider { V k } of Algorithm (12). Suppose that Assumption 3.3 is satisfied and α k = α k + h with α = 2 1 -β 3 and properly chosen h . Then there exists K ′ 3 &gt; 0 such that the following inequality holds for all k ≥ K ′ 3 :

<!-- formula-not-decoded -->

where ˆ c ′ 1 = ( ‖ V 0 -V π ‖ 2 + ‖ V 0 ‖ 2 +4) 2 and ˆ c ′ 2 = 7296 e (4(1 -γ ) ‖ V π ‖ 2 +1) 2 .

Proof of Theorem D.1. The constants are already identified in the proof of Theorem 3.5. Apply Theorem 2.1) (2) (b) (iii), when α k = α k + h with α = 2 1 -β 3 and properly chosen h , there exists K ′ 3 &gt; 0 such that we have for all k ≥ K ′ 3 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E TD ( λ )

The following lemma is useful when proving Lemma 3.1 and Proposition 3.4.

Lemma E.1. Let I be a finite set. For any k ≥ 0 , define two sequences { i t } 0 ≤ t ≤ k and { a t } 0 ≤ t ≤ k be such that i t ∈ I and a t ≥ 0 for all t = 0 , 1 , ..., k . Let x ∈ R |I| be defined by x i = ∑ k t =0 a t /BD { i t = i } for all i ∈ I . Then we have

<!-- formula-not-decoded -->

Proof of Lemma E.1. Using the definition of ‖ · ‖ 2 , we have

<!-- formula-not-decoded -->

The result follows by taking square root on both sides of the previous inequality.

## E.1 Proof of Lemma 3.1

For any V ∈ R |S| and ( s 0 , ..., s k , a k , s k +1 ) , we have by definition of the operators F τ k ( · , · ) and F k ( · , · ) that

<!-- formula-not-decoded -->

The result follows by taking the square root on both sides of the previous inequality.

## E.2 Proof of Proposition 3.4

- (1) For any V 1 , V 2 ∈ R |S| and y ∈ Y τ , we have by Triangle inequality that

<!-- formula-not-decoded -->

(Lemma E.1)

<!-- formula-not-decoded -->

Similarly, for any y ∈ Y τ , we have

<!-- formula-not-decoded -->

It follows that ‖ F τ k ( 0 , y ) ‖ 2 ≤ 1 1 -γλ .

- (2) The proof is identical to that of Propositon 3.1 (2).
- (3) (a) For any V ∈ R |S| and s ∈ S , we have

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

- (b) For any V 1 , V 2 ∈ R |S| and p ∈ [1 , ∞ ] , we have

Denote G = I -K ∑ τ i =0 ( γλP π ) i ( I -γP π ) . It remains to provide an upper bound on ‖ G ‖ p . Since

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the matrix G λ,τ has non-negative entries. Therefore, we have and

<!-- formula-not-decoded -->

‖ G λ,τ ‖ 1 = ‖ 1 /latticetop G λ,τ ‖ ∞ = ∥ ∥ ∥ ∥ 1 /latticetop -κ /latticetop (1 -γ )(1 -( γλ ) τ +1 ) 1 -γλ ∥ ∥ ∥ ∥ ∞ = 1 -K min (1 -γ )(1 -( γλ ) τ +1 ) 1 -γλ . It then follows from Lemma D.1 that

<!-- formula-not-decoded -->

Hence the operator F τ k ( · , · ) is a contraction with respect to ‖·‖ p , with a common contraction factor β 4 = 1 -K min (1 -γ )(1 -( γλ ) τ +1 ) 1 -γλ .

- (c) It is enough to show that V π is a fixed-point of ¯ F τ k ( · ) , the uniqueness follows from ¯ F τ k ( · ) being a contraction. Using the Bellman equation R π + γP π V π -V π = 0 , we have

<!-- formula-not-decoded -->

## E.3 Proof of Theorem 3.7

Wewill exploit the ‖·‖ 2 -contraction property of the operator ¯ F τ k ( · ) provided in Proposition 3.4. Let M ( x ) = ‖ x ‖ 2 2 be our Lyapunov function. Using the update equation (15), and we have for all k ≥ 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The terms 1 , 2 , and 3 correspond to the terms T 1 , T 3 , and T 4 in Eq. (17), and hence can be controlled in the exact same way as provided in Lemmas A.1, A.4, and A.5. The upper bounds of 1 , 2 , and 3 are summarized in the following lemma, whose proof is omitted.

Lemma E.2. The following inequalities hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As for the terms 3 , 4 , and 5 , we can easily use Lemma E.3 along with the Cauchy-Schwarz inequality to bound them, which gives the following result.

Lemma E.3. The following inequalities hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma E.3. (1) For all k ≥ τ , we have

<!-- formula-not-decoded -->

## (2) For all k ≥ τ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(3) For all k ≥ τ , we have

<!-- formula-not-decoded -->

The rest of the proof is to use the upper bounds we derived for the terms 1 to 6 in Eq. (28) to obtain the one-step contractive inequality. Repeatedly using such one-step inequality and we get the finite-sample bounds stated in Theorem 3.7.