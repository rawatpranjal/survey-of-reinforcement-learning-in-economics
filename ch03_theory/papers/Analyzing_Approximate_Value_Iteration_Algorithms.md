## Analyzing Approximate Value Iteration Algorithms

Arunselvan Ramaswamy ∗ †

Shalabh Bhatnagar

June 1, 2021

## Abstract

In this paper, we consider the stochastic iterative counterpart of the value iteration scheme wherein only noisy and possibly biased approximations of the Bellman operator are available. We call this counterpart as the approximate value iteration (AVI) scheme. Neural networks are often used as function approximators, in order to counter Bellman's curse of dimensionality. In this paper, they are used to approximate the Bellman operator. Since neural networks are typically trained using sample data, errors and biases may be introduced. The design of AVI accounts for implementations with biased approximations of the Bellman operator and sampling errors. We present verifiable sufficient conditions under which AVI is stable (almost surely bounded) and converges to a fixed point of the approximate Bellman operator. To ensure the stability of AVI, we present three different yet related sets of sufficient conditions that are based on the existence of an appropriate Lyapunov function. These Lyapunov function based conditions are easily verifiable and new to the literature. The verifiability is enhanced by the fact that a recipe for the construction of the necessary Lyapunov function is also provided. We also show that the stability analysis of AVI can be readily extended to the general case of set-valued stochastic approximations. Finally, we show that AVI can also be used in more general circumstances, i.e., for finding fixed points of contractive set-valued maps.

## 1 Introduction

Reinforcement Learning (RL) is a machine learning paradigm which consists of one or more agents that interact with an environment by taking actions. These actions have consequences for the agent(s) and affect the state of the environment, which is modelled as a Markov decision process (MDP). The typical goal

∗ Heinz Nixdorf Institute and the Department of Computer Science, Paderborn University, 33102 Paderborn, Germany arunr@mail.upb.de

† Dept. of Computer Science and Automation and the Robert Bosch Center for CyberPhysical Systems, Indian Institute of Science, Bengaluru - 560012, India. shalabh@iisc.ac.in

of a reinforcement learning problem is to solve a given sequential decision making problem. Popular RL algorithms include Approximate Value Iteration, QLearning, Temporal Difference Learning and Actor-Critic algorithms. Although based on the dynamic programming principle, they differ from traditional dynamic programming algorithms as they do not assume complete knowledge of the MDP. RL algorithms are versatile as they use sample-based methods that work with noisy observations of objective functions such as Value function and Q-factors.

Traditional RL algorithms are not readily applicable to decision making problems that involve large state and action spaces, since they suffer from Bellman's curse of dimensionality. This curse is lifted when these algorithms are integrated with artificial neural networks (ANN). Traditional RL algorithms have seen a major resurgence in recent years, as they are used in combination with ANN to solve problems arising in wide ranging applications including transportation, health-care and finance. Such algorithms are popularly referred to as Deep Reinforcement Learning (Deep RL) algorithms. Their popularity was triggered when 'better than' human performance was demonstrated, in playing Atari and Go, using RL algorithms, [13] [18]. The popular roles of ANN within the framework of RL include the approximation of the objective function (Value function, Q-Value Function, etc.) and extraction of relevant features. Deep RL is highly effective owing to recent advances in the field of RL, as well as the benefits resulting from the use of neural network architectures made possible due to significantly enhanced computational capacities. While Deep RL algorithms are empirical miracles, there is very little theory to back them up. In this paper we aim to address this issue.

The main caveat with using neural networks for function approximation is that their architecture needs to be chosen without complete knowledge of the function to be approximated. In fact, there need not exist network weights (hyper-parameters) such that the corresponding network output equals the objective function, everywhere. Further, since the neural networks are trained in an online manner, starting from a random initialization, the transient approximation errors may be large. From the previous argument, it is clear that one cannot expect that these errors would vanish asymptotically. In other words, to understand the long-term behavior of Deep RL algorithms, it is important to consider the effect of these persistent errors. Further, non-diminishing errors have a strong bearing on the stability (almost sure boundedness) of the algorithm at hand. It is well known in the literature that RL algorithms with unbounded approximation errors are not stable [7]. Hence, the weakest condition under which Deep RL algorithms can be expected to be stable is when the errors are asymptotically bounded. In this paper, we prove stability under the aforementioned weak condition for Value Iteration algorithms that use the approximate Bellman operator.

Traditional dynamic programming algorithms require taking expectations, to calculate the value function or the Q-factors. However, modern Deep RL algorithms are applied to scenarios wherein calculating expectations is impossible. Instead, one needs to work with 'data samples'. While the algorithms that can use samples are typically easy to implement, sampling errors often affect stability and convergence properties. In this paper, we characterize the limiting set of the approximate value iteration algorithm wherein the additive errors that arise from sampling are biased . We believe that ours is one of the first results

## 1.1 Approximate Value Iteration Algorithm

Value iteration methods are an important class of RL algorithms that are easy to implement. However, for many important applications, these methods suffer from the Bellman's curse of dimensionality . In this paper, we present Approximate Value Iteration (AVI) as a means to address the Bellman's curse of dimensionality by introducing an approximate Bellman operator within a classical value iteration framework. We assume that the Bellman operator is approximated in an online manner. Hence, the approximation errors vary over time, albeit in an asymptotically bounded manner. As stated earlier, if the approximation errors are allowed to be unbounded then the algorithm may not converge, see [7] for details. AVIs with bounded approximation errors have been previously studied in [7, 12, 14]. Bertsekas and Tsitsiklis [7] studied scenarios wherein the approximation errors are uniformly bounded over all states. Munos [14] extended the analysis of [7], allowing for approximation errors that are bounded in the weighted p-norm sense, for the infinite horizon discounted cost problem. In addition to a convergence analysis, [14] also provides a rate of convergence (finite time) analysis, under the assumption that the transition probabilities or future state distributions be 'smooth', among others.

In this paper, we consider the following AVI algorithm:

<!-- formula-not-decoded -->

where J n is the current estimate of the optimal cost-to-go vector; { a ( n ) } n ≥ 0 is the step-size sequence; T is the Bellman operator; /epsilon1 n is the approximation error at time n ; M n +1 is a square integrable Martingale difference sequence to account for sampling errors. In Section 3 we show that the structure of (1) encompasses two important variants that are relevant to the deep RL paradigm. The first variant allows for the use of a (neural network based) function approximation of the Bellman operator, say AT , such that the approximation errors are possibly biased. Within the context of deep RL, the neural network is trained in an online manner using a time-varying loss function such as [ ATJ n ( s n ) -( r ( s n , a ) + γJ n ( s n +1 ))] 2 at time n . As in Deep Q-Learning, it may be wise to sample mini-batches from an experience replay [13]. The second variant allows for the use of sampling, instead of taking expectations. As before, the sampling errors are allowed to be stochastic and biased. For a further discussion on these variants, the reader is referred to Section 3.

An important contribution of this paper is in the weakening of assumptions involved in the analysis of (1). For e.g., we do not require the previously mentioned restriction on transition probabilities of future distributions (cf. Section 3). As a consequence, we only present an asymptotic analysis and not a stronger finite sample analysis. However, our analysis encompasses both the stochastic shortest path and the discounted cost infinite horizon problems. With regards to the stability of (1), it is not immediately clear whether the approximation operator or the sampling errors influence it in a negative way. Thus, another important contribution of this paper is in proving the stability of AVI under standard assumptions from literature. It should also be noted that we characterise the optimality of the limiting cost-to-go vector found by

AVI. Specifically, we show that it is a fixed point of the 'perturbed Bellman operator' and belongs to a small neighborhood of J ∗ . We further relate the size of this neighborhood to the asymptotic norm-bound on the approximation errors.

## 1.2 Stochastic Approximation Algorithms

To develop the sufficient conditions for the convergence of AVI, and for its analysis, we build on tools from the fields of stochastic approximation algorithms (SAs) and viability theory. SAs are an important class of model-free iterative algorithms that are used to develop and analyze algorithms for stochastic control and optimization. There is a long and rich history to research in the field of SAs, see [17] [3] [4] [9] [11] [5]. SAs encompass both the algorithmic and theoretical aspects. Viability theory plays a major role in the latter. For more details on viability theory the reader is referred to [2]. While the main focus of this paper is to understand the long-term behavior of AVI, we deviate from this theme a little and present Lyapunov function based stability conditions for set-valued SAs. In other words, we believe that our stability analysis is readily applicable, verbatim, to set-valued SAs . The analyses presented herein build on the works of [1] and [16].

## 1.3 Fixed point finding algorithms for contractive set-valued maps

The ideas used to analyze AVI are later used to develop and analyze a SA for finding fixed points of set-valued maps. Fixed point theory is an active area of research due to its applications in a multitude of disciplines. Recently, the principle of dynamic programming (DP) was generalized by Bertsekas [6] to solve problems which, previously, could not be solved using classical DP. This extension involved a new abstract definition of the Bellman operator. The theory thus developed is called Abstract Dynamic Programming. An integral component of this new theory involves showing that the solution to the abstract Bellman operator is its fixed point. We believe that the results of Section 7 are helpful in solving problems that can be formulated as an Abstract Dynamic Program. Our contribution on this front is in the development and analysis of a SA for finding fixed points of contractive set-valued maps, see Section 7 for details. As mentioned before, we show that such algorithms are bounded almost surely and that they converge to a sample path dependent fixed point of the set-valued map under consideration. To the best of our knowledge ours is the first SA, complete with analysis, for finding fixed points of set-valued maps .

## 2 Definitions and Notations

Key definitions and notations encountered in this paper are listed in this section.

[Upper-semicontinuous map] We say that H is upper-semicontinuous, if given sequences { x n } n ≥ 1 (in R d 1 ) and { y n } n ≥ 1 (in R d 2 ) with x n → x , y n → y and y n ∈ H ( x n ), n ≥ 1, then y ∈ H ( x ).

[Marchaud Map] A set-valued map H : R d 1 → { subsets of R d 2 } is called Marchaud if it satisfies the following properties: (i) for each x ∈ R d 1 , H ( x ) is convex and compact; (ii) (point-wise boundedness) for each x ∈ R d 1 , sup w ∈ H ( x ) ‖ w ‖ &lt; K (1 + ‖ x ‖ ) for some K &gt; 0; (iii) H is upper-semicontinuous .

Let H be a Marchaud map on R d . The differential inclusion (DI) given by

<!-- formula-not-decoded -->

Φ t ( x ) = { x ( t ) | x ∈ ∑ , x (0) = x } . Let B × M ⊂ [0 , + ∞ ) × R d and define is then guaranteed to have at least one solution that is absolutely continuous. The reader is referred to [2] for more details. We say that x ∈ ∑ if x is an absolutely continuous map that satisfies (2). The set-valued semiflow Φ associated with (2) is defined on [0 , + ∞ ) × R d as:

<!-- formula-not-decoded -->

[Limit set of a solution &amp; ω -limit-set] The limit set of a solution x with x (0) = x is given by L ( x ) = ⋂ t ≥ 0 x ([ t, + ∞ )). Let M ⊆ R d , the ω -limit-set be defined by ω Φ ( M ) = ⋂ t ≥ 0 Φ [ t, + ∞ ) ( M ) .

[Invariant set] M ⊆ R d is invariant if for every x ∈ M there exists a trajectory, x ∈ ∑ , entirely in M with x (0) = x , x ( t ) ∈ M , for all t ≥ 0. Note that the definition of invariant set used in this paper, is the same as that of positive invariant set used in [5] and [10].

[Open and closed neighborhoods of a set] Let x ∈ R d and A ⊆ R d , then d ( x, A ) := inf {‖ x -y ‖ | y ∈ A } . We define the δ -open neighborhood of A by N δ ( A ) := { x | d ( x, A ) &lt; δ } . The δ -closed neighborhood of A is defined by N δ ( A ) := { x | d ( x, A ) ≤ δ } . The open ball of radius r around the origin is represented by B r (0), while the closed ball is represented by B r (0).

[Internally chain transitive set] M ⊂ R d is said to be internally chain transitive if M is compact invariant and for every x, y ∈ M , /epsilon1 &gt; 0 and T &gt; 0 we have the following: There exists n and Φ 1 , . . . , Φ n that are n solutions to the differential inclusion ˙ x ( t ) ∈ H ( x ( t )), points x 1 (= x ) , . . . , x n +1 (= y ) ∈ M and n real numbers t 1 , t 2 , . . . , t n greater than T such that: Φ i t i ( x i ) ∈ N /epsilon1 ( x i +1 ) and Φ i [0 ,t i ] ( x i ) ⊂ M for 1 ≤ i ≤ n . The sequence ( x 1 (= x ) , . . . , x n +1 (= y )) is called an ( /epsilon1,T ) chain in M from x to y .

[Attracting set &amp; fundamental neighborhood] A ⊆ R d is attracting if it is compact and there exists a neighborhood U such that for any /epsilon1 &gt; 0, ∃ T ( /epsilon1 ) ≥ 0 with Φ [ T ( /epsilon1 ) , + ∞ ) ( U ) ⊂ N /epsilon1 ( A ). Such a U is called the fundamental neighborhood of A . In addition to being compact if the attracting set is also invariant then it is called an attractor . The basin of attraction of A is given by B ( A ) = { x | ω Φ ( x ) ⊂ A } .

[Global attractor] If the basin of a given attractor is R d , then the attractor is called global attractor.

[Globally asymptotically stable equilibrium point] A point x 0 is an equilibrium point for the DI (2), if 0 ∈ H ( x 0 ). Further, it is globally asymptotically stable if it is a global attractor. This notion is readily extensible to sets.

[Lyapunov stable] The above set A is Lyapunov stable if for all δ &gt; 0, ∃ /epsilon1 &gt; 0 such that Φ [0 , + ∞ ) ( N /epsilon1 ( A )) ⊆ N δ ( A ).

## 3 Approximate value iteration methods

Most value iteration methods are based on fixed point finding algorithms. This is because the optimal cost-to-go vector is a fixed point of the Bellman operator. Since the Bellman operator is contractive with respect to some weighted maxnorm ‖· ‖ ν , it follows from fixed point theory that there is a unique fixed point for it. Further, this fixed point is the required optimal cost-to-go vector. Suppose T is the Bellman operator, the aim of value iteration methods is to find J ∗ such that J ∗ = TJ ∗ . Given a cost-to-go vector J = ( J ( s ) , s ∈ S ) T , let

<!-- formula-not-decoded -->

where A is the action space and S is the state space with s, s ′ ∈ S ; r ( s, a ) is the single-stage cost when action a is chosen in state s ; P is the (unknown) transition probability law with P ( s, a, s ′ ) being the probability of transition from state s to s ′ when action a is taken and 0 &lt; γ ≤ 1 is the discount factor. When 0 &lt; γ &lt; 1 we are in the infinite horizon discounted cost problem setting, and γ = 1 corresponds to the setting of the infinite horizon stochastic shortest path problem. In this paper we do not distinguish between the two, we shall implicitly work with the appropriate definition of T .

Suppose T can be exactly calculated, then the recursion J n +1 ← TJ n converges to J ∗ starting from any J 0 . However, for an exact calculation of T , one requires complete knowledge of the transition probability law P and the reward function r ( · , · ). In many applications this could be a hard requirement to satisfy. In today's deep learning age, it is common to work with approximations of the Bellman operator . These approximations are noisy (stochastic) and biased. Below, we present Approximate Value Iteration (AVI) , a stochastic iterative counterpart of traditional value iteration, designed to operate in the presence of noise and approximations:

<!-- formula-not-decoded -->

where J n ∈ R d for all n ≥ 0; A is the approximation operator; T is the Bellman operator, see (3); { a ( n ) } n ≥ 0 is the given step-size sequence and { M n +1 } n ≥ 0 is the noise sequence. Note that J n = ( J n (1) , . . . , J n ( d )) T for n ≥ 0 and the state space is given by S = { 1 , . . . , d } . Let us rewrite (4) as the following:

<!-- formula-not-decoded -->

where /epsilon1 n := ATJ n -TJ n is the approximation error at stage n ; { M n } n ≥ 1 is a square integrable Martingale difference noise sequence that is adapted to the filtration {F n } , defined by F n := σ 〈 J m , /epsilon1 m | m ≤ n 〉 , n ≥ 0.

## 3.1 Asssumptions to analyze AVI

Before listing the assumptions required to analyze (4)/(5), let us define the weighted max-norm ‖· ‖ ν . Given ν = ( ν 1 , . . . , ν d ) T such that ν i &gt; 0 for 1 ≤ i ≤ d , let ‖ x ‖ ν = max { | x i | ν i | 1 ≤ i ≤ d } , where x = ( x 1 , x 2 , . . . , x d ) T ∈ R d .

- (AV1) The Bellman operator T is contractive with respect to some weighted max-norm, ‖· ‖ ν , i.e., ‖ Tx -Ty ‖ ν ≤ α ‖ x -y ‖ ν for some 0 &lt; α &lt; 1.
- (AV2) T has a unique fixed point J ∗ and J ∗ is the unique globally asymptotically stable equilibrium point of ˙ J ( t ) = TJ ( t ) -J ( t ).
- (AV3) Almost surely lim sup n →∞ ‖ /epsilon1 n ‖ ν ≤ /epsilon1 , for some fixed /epsilon1 &gt; 0.
- (AV4) The step-size sequence { a ( n ) } n ≥ 0 is such that ∀ n, a ( n ) ≥ 0, ∑ n ≥ 0 a ( n ) = ∞ and ∑ n ≥ 0 a ( n ) 2 &lt; ∞ .

<!-- formula-not-decoded -->

- (AV5) ( M n , F n ) n ≥ 1 is a square integrable Martingale difference sequence ( E [ M n +1 |

K (1 + ‖ J n ‖ 2 ) , where n ≥ 0 and K &gt; 0.. The filtration {F n } n ≥ 0 is as defined above.

Now, we briefly discuss the above listed requirements. Assumption ( AV 1) is standard in literature and is readily satisfied in many applications, see Section 2.2 of Bertsekas and Tsitsiklis [7] for details. In Section 5, we discuss how (AV2) ensures the stability of AVI. ( AV 3) requires that the stochastic approximation errors are asymptotically bounded in an almost sure sense . This asymptotic bound is with respect to the weighted max-norm. Later, in Section 5.1, we show that the analysis of (5) is unaltered when the approximation errors are more generally bounded in the weighted p-norm sense (weighted Euclidean norms). Let us say that ( AV 3) is violated. This implies that ‖ /epsilon1 n ( m ) ‖ ↑ ∞ along a sequence { n ( m ) } m ≥ 0 ⊆ N . In words, there is a massive failure in approximating the Bellman operator, and there are points wherein the approximate Bellman operator differs from the true one by large amounts. Therefore, ∑ n ≥ 0 a ( n ) ‖ /epsilon1 n ‖ may equal ∞ as a consequence, and we can never guar- antee stability. Assumption ( AV 3) essentially requires that such circumstances be avoided.

On the surface ( AV 5) concerns the additive noise terms that are modelled as a square integrable martingale difference sequence. However, it also serves the dual role of analyzing the 'sample-based' variant of AVI . To illustrate this, we assume that the single-stage reward function r ( · , · ) is a given deterministic function, and that the algorithm can sample from the transition probability law P ( s, a, · ). If we use the notation ˆ T to represent sample-Bellman operator, then

<!-- formula-not-decoded -->

where ψ ( s, a ) is a random variable that takes values in the state-space S , and is distributed according to P ( s, a, · ). This is the setting of deep learning, and

the sample-based variant of AVI, is given by

<!-- formula-not-decoded -->

If we condition on the current state and action, it is fair to assume that the ψ ( s, a ) samples required to calculate the RHS of (6), are independent across time. Specifically, if we codify all the samples taken at stage n as ψ n , it follows that { ψ n } n ≥ 0 is an independent sequence. First, we define the filtration as: F 0 = σ 〈 J 0 , /epsilon1 0 〉 and F n = σ 〈 J m , /epsilon1 m , ψ k | m ≤ n, k &lt; n 〉 , n ≥ 1. Next, we define a zero-mean square integrable martingale difference sequence as: M n +1 := ( ˆ TJ n -J n -E [ ˆ TJ n -J n | F n ]) , n ≥ 0. It follows from the definition of the filtration, that E [ ˆ TJ n -J n | F n ] = TJ n -J n for all n ≥ 0, where T is the Bellman operator as defined in (3). Hence, the sample-based AVI (6) can be written as (5), with M n +1 as above. In other words, the AVI algorithm given by (5) has a general architecture, and can be used to solve problems involving biased function approximations and noisy samples .

Since M n +1 = ( ˆ TJ n -J n ) -( TJ n -J n ), its component along the dimension associated with state s is given by ˆ TJ n ( s ) -TJ n ( s ). It follows from the definitions of ˆ T and T that:

<!-- formula-not-decoded -->

∣ ∣ Let us use ‖· ‖ ∞ to represent the max-norm. There exists K 1 &gt; 0 such that the above inequality becomes:

<!-- formula-not-decoded -->

∣ ∣ For non-negative a, b ∈ R , we have that ( a + b ) 2 ≤ 2( a 2 + b 2 ) (using AM-GM inequality). Hence

<!-- formula-not-decoded -->

∣ ∣ As the max-norm is bounded above by the Euclidian norm ( ‖· ‖ 2 ∞ ≤ ‖· ‖ 2 ), we conclude from the above inequality that:

<!-- formula-not-decoded -->

≥ variation process associated with the Martingale sequence is a.s. bounded. It then follows from the Martingale Convergence Theorem that n ∑ m =0 a ( m ) M m , n ≥ 0, converges almost surely. If we interpret the martingale difference term M n +1 as the sampling error at stage n , then we may conclude that the sampling errors vanish over time .

In other words, the sample-based AVI satisfies ( AV 5). If we are able to show that sup n 0 ‖ J n ‖ &lt; ∞ a.s., then it follows from ( AV 4) that the quadratic

As stated in the Abstract and the Introduction, we are interested in the stability and convergence analysis of AVI. Before we present this analysis, we take a detour and consider general set-valued SAs, in order to present the

required (Lyapunov function based) stability assumptions . Clearly the AVI given by (5) can be viewed as appropriate set-valued SA. We present the stability assumptions for the general setting of set-valued SAs, since they can be additionally applied to analyze other algorithms arising in Deep RL and stochastic approximation settings. Following the short detour, we shall return to the analysis of AVI.

## 4 Lyapunov stability assumptions and general set-valued stochastic approximations

Before presenting the Lyapunov stability conditions, we present the general structure of set-valued SAs. Consider the following iteration in R d :

<!-- formula-not-decoded -->

where y n ∈ H ( x n ) for all n ≥ 0 with H : R d → { subsets of R d } , { a ( n ) } n ≥ 0 is the given step-size sequence and { M n +1 } n ≥ 0 is the given noise sequence. In addition to the required stability assumptions, we list a few that are required to present the stability analysis of (7) 1 .

- (A1) H : R d →{ subsets of R d } is a Marchaud map.
- (A2) (i) For all n ≥ 0, ‖ M n +1 ‖ ≤ D , where D &gt; 0 is some constant.

Note that the assumption on the Martingale noise terms, ( A 2), is stronger than the corresponding one in Section 3, i.e., ( AV 5). We consider this stronger assumption for the sake of clarity in presenting the stability analysis of (7). Later, in Section 6.2, we show that the aforementioned stability analysis carries forward even under the more general ( AV 5) instead of ( A 2). Below, we present a key assumption required to prove the stability of (5) and (7). Additionally, we present two different yet related variants of this key assumption, both of which lead to identical analyses. The verification of these assumptions involves the construction of an associated Lyapunov function. A recipe for its construction is discussed in Remark 1. These Lyapunov stability assumptions provide an alternative to the ones in [16]. Further, in lieu of Remark 1, we believe that these assumptions are readily verifiable.

<!-- formula-not-decoded -->

## Lyapunov function based stability assumptions

We begin by recalling the definition of the set-valued semiflow Φ from Section 2. Given a solution x of the DI ˙ x ( t ) ∈ H ( x ( t )), Φ t ( x ) := { x ( t ) | x ∈ ∑ , x (0) = x } , and ∑ is the set of solutions.

1 ( A 3 a ) is the required Lyapunov-based stability condition. Two additional alternative stability conditions, viz., ( A 3 b ) and ( A 3 c ) will also be presented.

- (A3a) Associated with the differential inclusion (DI) ˙ x ( t ) ∈ H ( x ( t )) is a compact set Λ, a bounded open neighborhood U ( Λ ⊆ U ⊆ R d ) and a function V : U → R + such that

- ( i ) ∀ t ≥ 0, Φ t ( U ) ⊆ U i.e., U is strongly positively invariant.
- ( ii ) V -1 (0) = Λ.
- ( iii ) V is a continuous function such that for all x ∈ U \ Λ and y ∈ Φ t ( x ) we have V ( x ) &gt; V ( y ), for any t &gt; 0.

It follows from ( A 3 a ) and Proposition 3.25 of Bena¨ ım, Hofbauer and Sorin [5] that Λ contains a Lyapunov stable attracting set. Further there exists an attractor contained in Λ whose basin of attraction contains U . Let us define V s := { x | V ( x ) &lt; s } and V r := { x | V ( x ) ≤ r } , for every s &gt; 0 and r ≥ 0. Then, using the compactness of Λ and ( A 3 a )( ii ) we can show that ⋂ r&gt; 0 V r = Λ.

Since U is open, ∃ 0 &lt; R ( a ) &lt; sup x ∈U V ( x ), such that V r ⊂ U , for r ≤ R ( a ). Now, we define open sets B and C such that the following conditions are satisfied:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We use the notations B a and C a to indicate that the constructed sets are associated with (A3a). Let B a := V r ( a ) and C a := V s ( a ) such that 0 &lt; r ( a ) &lt; s ( a ) &lt; R ( a ), where R ( a ) is such that V q ⊂ U for q ≤ R ( a ). Since V is continuous, we have that the closures of V r ( a ) and V s ( a ) satisfy V r ( a ) = V r ( a ) and V s ( a ) = V s ( a ) , respectively. Hence, conditions (Ca) and (Cb) are readily satisfied.

The purpose of defining such sets will be clarified shortly. For now, we proceed with the first variant of ( A 3 a ), we call it ( A 3 b ):

- (A3b) Associated with ˙ x ( t ) ∈ H ( x ( t )) is a compact set Λ, a bounded open neighborhood U and a function V : U → R + such that
- (i) ∀ t ≥ 0, Φ t ( U ) ⊆ U i.e., U is strongly positively invariant.
- (ii) V -1 (0) = Λ.
- (iii) V is an upper semicontinuous function such that for all x ∈ U \ Λ and y ∈ Φ t ( x ) we have V ( x ) &gt; V ( y ), where t &gt; 0.
- (iv) V r := { x | V ( x ) ≤ r } is closed for each r ≥ 0.

Since ( A 3 a )( iii ) = ⇒ ( A 3 b )( iii ), one may view ( A 3 b ) as a weakening of ( A 3 a ). Again, using Proposition 3.25 of Bena¨ ım, Hofbauer and Sorin [5] we get that Λ contains an attractor set such that U belongs to its basin of attraction. As in the case of ( A 3 a ), we wish to define open sets B b and C b satisfying the previously mentioned conditions (Ca) and (Cb). We begin by claiming that V r is open for r &gt; 0. We prove this claim by showing that V c r = { x | V ( x ) ≥ r } is closed. For this, we consider { x n } n ≥ 0 ⊆ V c r such that lim n →∞ x n = x . From the upper semicontinuity of V , we get V ( x ) ≥ lim sup n V ( x n ) ≥ r , hence, x ∈ V c r .

<!-- formula-not-decoded -->

→∞ Thus we get that V r is open and V r is closed (from (A3b)(iv)) for r &gt; 0. Finally, note that Λ = ⋂ r&gt; 0 V r as a consequence of (A3b)(ii). Hence, as in the case of (A3a), 0 &lt; R ( b ) &lt; sup V ( x ) such that r for r R ( b ).

We are now ready to define B b and C b satisfying conditions (Ca) and (Cb). As before, we let B b := V r ( b ) and C b := V s ( b ) , where 0 &lt; r ( b ) &lt; s ( b ) &lt; R ( b ). From

(A3b)(iv), we get that V r ( b ) ⊂ V r ( b ) and V s ( b ) ⊂ V s ( b ) . Using this observation, we can easily conclude that conditions (Ca) and (Cb) are satisfied.

Remark 1. If we can associate ˙ x ( t ) ∈ H ( x ( t )) with an attractor set A and a strongly positive invariant neighborhood U , of A , then we can define an uppersemicontinuous Lyapunov function V , as found in Remark 3.26, Section 3.8 of [5]. In particular, we can define a local Lyapunov function V : U → R + such that V ( x ) := max { d ( y, A ) g ( t ) | y ∈ Φ t ( x ) , t ≥ 0 } , where g is an increasing function with 0 &lt; c &lt; g ( t ) &lt; d for all t ≥ 0 .

follows from the upper semicontinuity of V that sup u ∈U V ( u ) &lt; ∞ i.e., (A3b)(iv)

We claim that V , as defined above, satisfies ( A 3 b ) . To see this, we begin by noting that ( A 3 b )( i ) is trivially satisfied. If we let Λ := A , then it follows from the definition of V that ( A 3 b )( ii ) is satisfied. Since U is strongly positive invariant and V ( x ) ≤ sup u ∈U d ( u, A ) × d for x ∈ U , sup u ∈U V ( u ) &lt; ∞ . It now is satisfied. To show that ( A 3 b )( iii ) is also satisfied, we first fix x ∈ U and t &gt; 0 . It follows from the definition of a semi-flow that Φ s ( y ) ⊆ Φ t + s ( x ) for any y ∈ Φ t ( x ) , where s &gt; 0 . Further,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The RHS of the above equation is V ( y ) i.e., V ( x ) &gt; V ( y ) .

It is left to show that V r is closed for r ≥ 0 . For this, we present a proof by contradiction, by assuming that { x n } n ≥ 0 ⊂ V r , lim n x n = x and x / ∈ V r .

→∞ We have V ( x ) = r + c for some c &gt; 0 . From the definition of V , we get that ∃ t ( x ) ≥ 0 and y ( x ) ∈ Φ t ( x ) ( x ) such that d ( y ( x ) , A ) g ( t ( x )) &gt; r + c/ 2 . We may use Corollary 2 (Approximate Selection Theorem) from Chapter2, Section 2 of [2] to construct { y n } n ≥ 0 such that y n ∈ Φ t ( x ) ( x n ) for each n ≥ 0 and lim inf n →∞ y n = y ( x ) . Hence ∃ N such that d ( y N , A ) g ( t ( x )) &gt; r + c/ 4 . This yields a contradiction, as x N ∈ V r implying that d ( y N , A ) g ( t ( x )) ≤ r .

Below is the final variant of ( A 3 a ):

- (A3c) (i) A is the global attractor of ˙ x ( t ) ∈ H ( x ( t )).
- (iii) V ( x ) ≥ V ( y ) for all x ∈ A , y ∈ Φ t ( x ) and t &gt; 0.
- (ii) V : R d → R + is an upper semicontinuous function such that V ( x ) &gt; V ( y ) for all x ∈ R d \ A , y ∈ Φ t ( x ) and t &gt; 0.
- (iv) V r := { x | V ( x ) ≤ r } is closed for each r ≥ 0.

Since A is the global attractor, it is compact and for every x / ∈ A , we have that V ( x ) ≥ sup y ∈A V ( y ) and A = ⋂ r&gt; sup y ∈A V ( y ) V r . Further, ∃ R ( c ) with sup y ∈A V ( y ) &lt;

R ( c ) &lt; ∞ such that V r is a bounded open set for r ≤ R ( c ). Again, we define open sets B c and C c satisfying conditions (Ca) and (Cb) (see below the statement of ( A 3 a )), as B c := V r ( c ) and C c := V s ( c ) , 0 &lt; r ( c ) &lt; s ( c ) &lt; R ( c ). The steps involved in showing that these are indeed the required sets follow similar arguments as for the sets associated with (A3b).

Remark 2. In Remark 1, we explicitly constructed a local Lyapunov function satisfying ( A 3 b ) . We can similarly construct a global Lyapunov function, ˆ V : R d → R + , satisfying ( A 3 c ) as follows: ˆ V ( x ) := max { d ( y, A ) g ( t ) | y ∈ Φ t ( x ) , t ≥ 0 } , with g ( · ) defined as in Remark 1. To see that ˆ V satisfies ( A 3 c ) , one may emulate the arguments following Remark 1.

Let us say that we are given bounded open sets B and C such that B ⊂ C . Also, ˙ x ( t ) ∈ H ( x ( t )) can be associated with an attractor such that B is its fundamental neighborhood. Then a classical way to ensure stability of (7) is by projecting the iterate at every stage n , i.e., project x n onto B whenever x n / ∈ C . This associated projective scheme is given by:

<!-- formula-not-decoded -->

where ˆ x 0 = x 0 and /intersectionsq B , C : R d → { subsets of R d } is the projection operator that projects onto the boundary of B , when the operand escapes from set C , i.e.,

<!-- formula-not-decoded -->

.

The advantage of using (8) as opposed to (7) is that stability is trivially ensured since B is bounded. The main drawback, however, is that B and C cannot be easily constructed. Further, if B and C are not carefully chosen, then the algorithm may converge to an undesirable limit. Recall that we constructed bounded sets B a/b/c and C a/b/c , assuming ( A 3 a/b/c ) and satisfying conditions (Ca) and (Cb). The tuple ( B a/b/c , C a/b/c ) can be used to obtain a hypothetical realization of (8), which is the primary tool in the stability analysis of (7). A key step in this analysis involves 'comparing' the right-hand sides of the iterations (7) and (8). To facilitate such a comparison, we make the natural assumption that the realizations of the martingale noise are identical in both the iteration and its projective counterpart. Being that (8) is hypothetical, this assumption is fair.

We are now ready to present our final assumption which specifies the relationship between a set-valued SA and its projective counterpart.

- (A4) Almost surely, there exists N &lt; ∞ such that sup n ≥ N ‖ x n +1 -˜ x n +1 ‖ &lt; ∞ , where ˜ x n +1 = (ˆ x n + a ( n )[ y n + M n +1 ]) is the value (8) at time n +1 before projection. The sequences { x n } and { ˆ x n } are generated by (7), and (8) using sets B a/b/c and C a/b/c , respectively.

In Section 5, we show that the AVI given by (5) satisfies (A3c) and (A4). The verifiability of (A4) for general problems involving set-valued mean-fields, is discussed in Section 6.3. Before proceeding, we define the following:

Inward directing set: Given a differential inclusion ˙ x ( t ) ∈ H ( x ( t )) , an open set O is said to be an inward directing set with respect to the aforementioned

differential inclusion, if Φ t ( x ) ⊆ O , t &gt; 0 , whenever x ∈ ∂ O , where ∂ O denotes the boundary of set O . In words, any solution starting at the boundary of O is 'directed inwards', into O .

Proposition 1. C i is an inward directing set with respect to ˙ x ( t ) ∈ H ( x ( t )) , and a fundamental neighborhood of attractor A , where i ∈ { a, b, c } .

Proof. PROOF: Fix i ∈ { a, b, c } . Recall that C i = V r ( i ) for an appropriately chosen r ( i ) &gt; 0. Further, recall that C i ⊆ V r ( i ) , where V r ( i ) is a compact (closed and bounded) subset of the basin of attraction for attractor A . Hence V r ( i ) , and consequently C i , is a fundamental neighborhood of A . Recall that V r ( i ) = { x | V ( x ) &lt; r ( i ) } , and that V r ( i ) = { x | V ( x ) ≤ r ( i ) } . Since C i ⊆ V r ( i ) , we have that V ( x ) = r ( i ) whenever x ∈ ∂ C i . In other words, the Lyapunov function V evaluated at any point on the boundary of C i equals r ( i ). If follows from the assumption (A3i) that V ( x ) &gt; V ( y ) for y ∈ Φ t ( x ), x ∈ U \ Λ and t &gt; 0. Hence, for x ∈ ∂ C i and t &gt; 0, we have that V (Φ t ( x )) &lt; r ( i ) and Φ t ( x ) ⊆ V r ( i ) , i.e., C i is inward directing. /squaresolid

To show that (7) is stable, we compare it with the projective counterpart (8). To enable a hypothetical realization of (8), we need to ensure the existence of an inward directing set with respect to the associated DI . This inward directing set could be C a , C b or C c , depending on whether ( A 3 a ), ( A 3 b ) or ( A 3 c ) is verified. Formally, we will prove the following stability theorem:

<!-- formula-not-decoded -->

Before we can arrive at the above statement, we need to prove a few auxiliary lemmata (Lemmas 3 - 7). However, we defer these and the proof of Theorem 1 to Section 6. For now, we assume the truth of the Theorem 1, and return to the analysis of AVI.

## 5 Analyzing the Approximate Value Iteration Algorithm

Let us analyze AVI, assuming ( AV 1) -( AV 5). Before we show that AVI converges to a fixed point of the perturbed Bellman operator, we need to show that it is stable. For this, we show that ( AV 1) -( AV 5) together imply that ( A 1) , ( A 3) and ( A 4) are satisfied, then invoke Theorem 1. Before proceeding with the analysis, let us recall the AVI recursion:

<!-- formula-not-decoded -->

where, from (AV3), lim sup n →∞ ‖ /epsilon1 n ‖ ν ≤ /epsilon1 a.s. For more details on the notations, the reader is referred to Section 3. It follows from ( AV 3), that there exists N &lt; ∞ , possibly sample path dependent, such that sup n ≥ N ‖ /epsilon1 n ‖ ν ≤ /epsilon1 . Since we are only interested in the asymptotic behavior of the recursion, without loss of generality, we may assume that sup /epsilon1 n ν /epsilon1 a.s. We begin our analysis with n ≥ 0 ‖ ‖ ≤

a couple of technical lemmas. First, let us define ν max := max { ν i | 1 ≤ i ≤ d } and ν min := min { ν i | 1 ≤ i ≤ d } , then for z ∈ R d we have:

<!-- formula-not-decoded -->

Lemma 1. B /epsilon1 := { y | ‖ y ‖ ν ≤ /epsilon1 } is a convex compact subset of R d , where /epsilon1 &gt; 0 .

Proof. PROOF: First we show that B /epsilon1 is convex. Given y 1 , y 2 ∈ B /epsilon1 and y = λy 1 +(1 -λ ) y 2 , where λ ∈ (0 , 1), we need show that y ∈ B /epsilon1 . This is a direct consequence of: ‖ y ‖ ν ≤ λ ‖ y 1 ‖ ν +(1 -λ ) ‖ y 2 ‖ ν ≤ λ/epsilon1 +(1 -λ ) /epsilon1 .

Now we show that B /epsilon1 is compact. Since ‖ y ‖ ∞ ν max ≤ ‖ y ‖ ν , it follows that B /epsilon1 is a bounded set. It is left to show that B /epsilon1 is closed. Let y n → y and y n ∈ B /epsilon1 for every n . Since lim inf n →∞ ‖ y n ‖ ν ≥ ‖ y ‖ ν , it follows that y ∈ B /epsilon1 . /squaresolid

Lemma 2. The set-valued map ˜ T given by ˜ Tx ↦→ Tx + B /epsilon1 is a Marchaud map.

Proof. PROOF: Since B /epsilon1 is a compact convex set, it follows that ˜ Tx is compact and convex. Since T is a contraction map, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Given z ∈ R d , we have:

where ‖ z ‖ = √ z 2 1 + . . . + z 2 d is the standard Euclidean norm, and d ≥ 1. Hence we have

√

<!-- formula-not-decoded -->

z ∈ Tx where K ′ := ( d ν max ν min ) ∨ ( d ν max ‖ T 0 ‖ ν ). We have that sup z ∈ ˜ Tx ‖ z ‖ ≤ ‖ Tx ‖ + sup z ∈ B /epsilon1 ‖ z ‖ . It follows from Lemma 1 that K := K ′ ∨ sup z ∈ B /epsilon1 ‖ z ‖ is finite, hence sup ˜ ‖ z ‖ ≤ K (1 + ‖ x ‖ ).

We now show that ˜ T is upper semicontinuous. Let x n → x , y n → y and y n ∈ ˜ Tx n for n ≥ 0. Since T is continuous, we have that Tx n → Tx . Hence, ( y n -Tx n ) → ( y -Tx ) and /epsilon1 ≥ lim inf n →∞ ‖ y n -Tx n ‖ ν ≥ ‖ y -Tx ‖ ν . In other words, y ∈ ˜ Tx . /squaresolid

Let us define the Hausdorff metric with respect to the weighted max-norm H ν .

Definition: Let us suppose that we are given A,B ⊆ R d . The Hausdorff metric with respect to the weighted max-norm ‖· ‖ ν , is given by H ν ( A,B ) := sup x ∈ A d ν ( x, B ) ∨ sup y ∈ B d ν ( y, A ) , where d ν ( x, B ) := inf {‖ x -y ‖ ν | y ∈ B } and

d ν ( y, A ) := inf {‖ x -y ‖ ν | x ∈ A } . The Hausdorff metric can be more generally defined with respect to any metric ρ as H ρ ( A,B ) := sup x ∈ A ρ ( x, B ) ∨ sup y ∈ B ρ ( y, A ) ,

Claim 1: Given x, y ∈ R d , there exist x ∗ ∈ ˜ Tx and y ∗ ∈ ˜ Ty such that ‖ x ∗ -y ∗ ‖ ν = H ν ( ˜ Tx, ˜ Ty ) .

where ρ ( y, A ) := inf { ρ ( x, y ) | x ∈ A } and ρ ( x, B ) := inf { ρ ( x, y ) | y ∈ B } . Now, we state a couple of simple claims without proofs:

The above claim directly follows from the observation that for any x 0 ∈ ˜ Tx and y 0 ∈ ˜ Ty we have

<!-- formula-not-decoded -->

Claim 2: For any z ∈ R d we can show that

<!-- formula-not-decoded -->

Given a set-valued map H : R d →{ subsets of R d } , x ∈ R d is an equilibrium point of H iff the origin belongs to H ( x ). Since J ∗ is the unique fixed point of the Bellman operator T , J is a fixed point of the perturbed Bellman operator ˜ T (defined in the statement of Lemma 2), or an equilibrium point of the ˜ TJ -J operator, iff ‖ TJ -J ‖ ν ≤ /epsilon1 . In other words, the equilibrium set of ˜ TJ -J is given by A = { J | ‖ TJ -J ‖ ν ≤ /epsilon1 } . Since J ∗ ∈ A and T is continuous, A constitutes a closed neighborhood of J ∗ . Controlling the norm-bounds, /epsilon1 , on the approximation errors is one way to control the size of A .

/negationslash

Assumption ( AV 2) dictates that J ∗ is the global attractor (global asymptotic stable equilibrium point) of ˙ J ( t ) = TJ ( t ) -J ( t ). The following is a consequence of the upper-semicontinuity of J ∗ : given a neighborhood N of J ∗ , ∃ /epsilon1 ( N ) &gt; 0 such that ˙ J ( t ) ∈ ˜ TJ ( t ) -J ( t ) has a global attractor A ′ ⊆ N , provided /epsilon1 ≤ /epsilon1 ( N ) a.s. The norm-bound on the approximation errors, /epsilon1 ( N ) , is therefore a function of the neighborhood N . We show that the asymptotic behavior of AVI, (5), is identical to that of a solution to ˙ J ( t ) ∈ ˜ TJ ( t ) -J ( t ), i.e., it converges to A ′ . However, it is desirable for AVI to converge to an equilibrium point in A . In what follows, we will in fact show that (5) converges to A∩A ′ and that A∩A ′ = φ .

the reader is referred to the discussion around (8) in Section 4. Using these sets,

Typically, a certain degree of accuracy is expected from the AVI. This accuracy is specified through the specification of a neighborhood N , of J ∗ . Once N is fixed, the above discussion provides /epsilon1 ( N ), the asymptotic normbound on the approximation errors, and A ′ , the global attractor associated with ˙ J ( t ) ∈ ˜ TJ ( t ) -J ( t ). Note that the asymptotic error bound is ensured by effectively training a parameterized function (for e.g., neural networks) to approximate the Bellman operator. As stated earlier, the stability analysis of AVI involves verifying that ( A 1) -( A 4) are satisfied. Among the three variants of ( A 3), we choose to verify ( A 3 c ). Recall that verifying ( A 3 c ), of Section 4, involves the construction of a global Lyapunov function, and this function can be constructed using the recipe presented in Remark 2. The main ingredients of this recipe are a global attractor set and an associated differential inclusion. For AVI, we construct the global Lyapunov function using A ′ and ˙ J ( t ) ∈ ˜ TJ ( t ) -J ( t ). Note that once ( A 3 c ) is verified, we construct bounded open sets B and C such that A ′ ⊆ B and B ⊆ C . For details on this construction,

we can associate a projective counterpart to AVI:

<!-- formula-not-decoded -->

Let us call this projective approximate value iteration . It is worth noting that the noise sequences in (5) and (12) are identical and that ˆ /epsilon1 n ≤ /epsilon1 for all n . Following the analysis in Section 6.1, we can conclude that ˆ J n → A ′ . We are now ready to present the main theorem of this paper.

Theorem 2. Under ( AV 1) -( AV 5) , the AVI recursion given by (5) is stable and converges to an equilibrium point of ˜ TJ -J , i.e., to some point in { J | ‖ TJ -J ‖ ν ≤ /epsilon1 } , where lim sup n →∞ ‖ /epsilon1 n ‖ ν ≤ /epsilon1 a.s.

Proof. PROOF: To prove that AVI is stable, we begin by showing that it satisfies ( A 1) , ( A 3) and ( A 4). Then, we invoke Theorem 1 to infer the stability of AVI. Once we have stability, we proceed with the convergence analysis. From the discussion presented before the statement of this theorem, it is clear that ( A 3 c ) is satisfied . It follows from ( AV 1), ( AV 3) and (11) that ( A 1) is satisfied. Hence, we shall obtain the stability of AVI if we are able to show that ( A 4) is satisfied. Recall that we use J n to denote the original AVI iterate, and ˆ J n to denote its associated projective counterpart. As mentioned before, it follows from the analysis in Section 6.1 that ˆ J n →A ′ , where A ′ is as defined in this section. Further, this analysis does not require that ( A 4) is satisfied.

We are now ready to show that ( A 4) is also satisfied by AVI. Since ˆ J n →A ′ , there exists N , possibly sample path dependent, such that ˆ J n ∈ B for all n ≥ N . For k ≥ 0 and n ≥ N ,

‖ J n + k +1 -ˆ J n + k +1 ‖ ν ≤ ∥ ∥ ∥ J n + k -ˆ J n + k + a ( n + k ) ( ( TJ n + k + /epsilon1 n + k ) -( T ˆ J n + k +ˆ /epsilon1 n + k ) -( J n + k -ˆ J n + k ) )∥ ∥ ∥ ν . Grouping terms of interest in the above inequality we get:

<!-- formula-not-decoded -->

As a consequence of (10) the above equation becomes

<!-- formula-not-decoded -->

We now consider the following two cases:

In this case (13) becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Simplifying the above equation, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Simplifying the above equation, we get

<!-- formula-not-decoded -->

We may thus conclude the following:

<!-- formula-not-decoded -->

Applying the above set of arguments to ‖ J n + k -ˆ J n + k ‖ ν and proceeding recursively to ‖ J N -ˆ J N ‖ ν we may conclude that for any n ≥ N ,

<!-- formula-not-decoded -->

If we couple the above equation with (11), we get that ( A 4) is satisfied. It now follows from Theorem 1 that sup n ≥ 0 ‖ J n ‖ &lt; ∞ a.s. (stability of AVI).

/negationslash

To analyze the convergence properties of AVI, we use Theorem 3.6 and Lemma 3.8 of Bena¨ ım [5]. Specifically, it follows from Theorem 3.6 [5] that AVI converges to a closed connected internally chain transitive invariant set, S , of ˙ J ( t ) ∈ ˜ TJ ( t ) -J ( t ). Further, since A ′ is a global attractor of ˙ J ( t ) ∈ ˜ TJ ( t ) -J ( t ), it follows that S ⊆ A ′ . Hence J n →A ′ . But we need to show that J n →A , the set of equilibrium points of the set-valued operator ˜ TJ ( t ) -J ( t ). We achieve this by showing that the J n sequence, in fact, converges to A∩A ′ and that A∩A ′ = φ . For this we need Theorem 2 from Chapter 6 of Aubin and Cellina [2] which we reproduce below.

[Theorem 2, Chapter 6 [2]] Let F be an upper semicontinuous map from a closed subset K ⊂ X to X with compact convex values and x ( · ) be a solution trajectory of ˙ x ( t ) ∈ F ( x ( t )) that converges to some x ∗ in K . Then x ∗ is an equilibrium of F .

Since sup n ≥ 0 ‖ J n ‖ &lt; ∞ a.s., there exists a large compact convex set K ⊆ R d , possibly sample path dependent, such that J n ∈ K for all n ≥ 0. Further, K can be chosen such that the 'tracking solution' of ˙ x ( t ) ∈ ˜ TJ ( t ) -J ( t ) is also inside K . It now follows that the conditions of the above stated theorem are satisfied. Hence every limit point of (5) is an equilibrium point of ˜ TJ -J . In other words, J n →A∩A ′ , as required. /squaresolid

Remark 3. We can show that { J | ‖ TJ -J ‖ ν ≤ /epsilon1 } ↓ { J ∗ } as /epsilon1 ↓ 0 . In other words, as the norm-bound on the asymptotic errors of the AVI decreases, the limiting set asymptotically diminishes and converges to the point J ∗ .

Given a cost-to-go vector J , let J ( s ), its s th component, denote the costto-go value associated with state s , where 1 ≤ s ≤ d and d is the number of states in the system. Suppose J ∞ is the limit of AVI, then it follows from Theorem 2, that | TJ ∞ ( s ) -J ∞ ( s ) | ν s ≤ /epsilon1 . For the discounted cost problem, we know that the Bellman operator is contractive with respect to the /lscript ∞ norm, i.e., ν s = 1 for 1 ≤ s ≤ d . Then, | TJ ∞ ( s ) -J ∞ ( s ) | ≤ /epsilon1 for all states s . Given state s and an asymptotic bound of /epsilon1 on the approximation errors, in the weighted max-norm sense, Theorem 2 states that the limit of AVI J ∞ , satisfies | TJ ∞ ( s ) -J ∞ ( s ) | ≤ ν s /epsilon1 . Hence, lower is the weight ν s associated with state s , the closer J ∞ ( s ) is to the optimal cost-to-go value J ∗ ( s ).

## 5.1 A note on the norm used to bound the approximation errors

Recall that the approximation errors are asymptotically bounded in the weighted max-norm sense. These errors are consequences of Bellman operator approximations, used to counter Bellman's curse of dimensionality . In a model-free setting, typically one is given data of the form ( x n , v n ), where v n is an unbiased estimate of the objective function at x n . The online training of an approximation operator can be emulated by a supervised learning algorithm. This algorithm would return a good fit g , of the Bellman operator, from within a class of possible functions T . The objective for these algorithms would be to minimize the empirical approximation errors. Previously, we considered approximation operators that minimize errors in the weighted max-norm sense. This means, ensuring that the approximation errors are uniformly bounded across all states, something that may be hard in large-scale applications. Here, the errors may be minimized in the weighted p-norm sense.

In many applications the approximation operators work by minimizing the errors in the /lscript 1 and /lscript 2 norms, see Munos [14] for details. Let us consider the general case of approximation errors being bounded in the weighted p-norm sense. Specifically, we consider (5) with ‖ /epsilon1 n ‖ ω,p ≤ /epsilon1 for some fixed /epsilon1 &gt; 0. Recall the definition of the weighted p-norm of a given z ∈ R d :

<!-- formula-not-decoded -->

where ω = ( ω 1 , . . . , ω d ) is such that ω i &gt; 0, 1 ≤ i ≤ d and p ≥ 1.

We observe the following relation between weighted p-norm and the weighted max-norm. For z ∈ R d , we have ‖ z ‖ ∞ ν max ≤ ‖ z ‖ ν ≤ ‖ z ‖ ∞ ν min , where ‖· ‖ ∞ denotes the (unweighted) max-norm, ν max = max { ν 1 , . . . , ν d } , ν min = min { ν 1 , . . . , ν d } , and ‖ z ‖ ν = max { | z i | ν i | 1 ≤ i ≤ d } . Then,

<!-- formula-not-decoded -->

where ω min := min { ω 1 , . . . , ω d } . Let us consider the following stochastic iterative AVI scheme:

<!-- formula-not-decoded -->

where ˜ /epsilon1 n = TJ n -TJ n , T is the approximate Bellman operator when the approximation errors are bounded in the weighted p-norm sense, and ‖ ˜ /epsilon1 n ‖ ω,p ≤ /epsilon1 for all n ≥ 0. Using the previously discussed relation between the weighted pnorm and the weighted max-norm we may arrive at a similar result for (16). Before stating the theorem, we recall that the Bellman operator T is contractive with respect to the weighted max-norm ‖· ‖ ν .

Theorem 3. Under (AV1,2,4,5), the AVI given by (16) is stable and converges to some point in { J | ‖ TJ -J ‖ ω,p ≤ /epsilon1 } , provided lim sup n →∞ ‖ ˜ /epsilon1 ‖ ω,p ≤ /epsilon1 and ω min &gt; 0 .

Proof. PROOF: First, we show that (16) is stable, i.e., sup n ≥ 0 ‖ J n ‖ ω,p &lt; ∞ a.s. To do this we emulate the proof of Theorem 2, in particular, the part of the proof that shows the stability of (5). The main difference between (5) and (16) is that the approximation errors in the former are bounded in the weighted max-norm sense, while in the latter they are bounded in the weighted p-norm sense. To emulate the proof, it is sufficient to show that:

<!-- formula-not-decoded -->

This is because, (AV3) is now valid with /epsilon1 ν min ω 1 /p min replacing /epsilon1 in its statement. Given that lim sup n →∞ ‖ ˜ /epsilon1 n ‖ ω,p ≤ /epsilon1 , from previous discussion on norms, we get:

<!-- formula-not-decoded -->

Hence, sup n ≥ 0 ‖ J n ‖ ν &lt; ∞ a.s. Again, using the previously discussed norm relations and ω min &gt; 0, we get the required sup n ≥ 0 ‖ J n ‖ ω,p &lt; ∞ a.s.

<!-- formula-not-decoded -->

It follows from Theorem 3.6 of [5] that (16) tracks a solution to ˙ J ( t ) ∈ TJ ( t ) -J ( t ), where TJ = TJ + ˜ B /epsilon1 and ˜ B /epsilon1 := { δ | ‖ δ ‖ ω,p ≤ /epsilon1 } . By 'tracking', we mean that the asymptotic properties of (16) and ˙ J ( t ) ∈ TJ ( t ) -J ( t ) are identical. Now, as in the proof of Theorem 2, we invoke Theorem 2 from Chapter 6 of [2] to deduce that J n → an equilibrium point of TJ -J , as n →∞ . Hence, 0 ∈ TJ ∞ -J ∞ , 0 ∈ TJ ∞ -J ∞ + ˜ B /epsilon1 , and ‖ TJ ∞ -J ∞ ‖ ω,p ≤ /epsilon1 , as required.

In [14], a supervised learning algorithm is described, to approximate the Bellman operator. An important step in the algorithm is the sampling of states. When a state is sampled well, the approximate Bellman operator evaluated at that state has a high accuracy. Further, the weight ω s , from the above discussed weighted p-norm, is directly proportional to sampling rate of state s , where 1 ≤ s ≤ d . A larger ω s therefore corresponds to a better approximation at s . In the proof of Theorem 3, we required ω min &gt; 0 to establish the stability of (16). Let us suppose we use the approximation algorithm from [14] within our AVI routine. Then, it follows from the previous discussion that in order to guarantee

the stability of AVI, one must ensure that the approximation algorithm samples every state (possibly unequally). On the other hand, when state s is never sampled, ω s and (hence) ω min equal 0. We expect the approximation errors to be high for s . In this case, stability of AVI cannot be guaranteed as it may be that J n ( s ) → ∞ with positive probability, primarily due to the unchecked accumulation of approximation errors over time.

Given a state s with ω s &gt; 0, it follows from Theorem 3 that | TJ ∞ ( s ) -J ∞ ( s ) | ≤ /epsilon1 / ω 1 /p s . Hence, larger the weight value ω s , the closer J ∞ ( s ) is to the optimal costto-go value J ∗ ( s ). Suppose s is associated with a very small weight value, i.e., ω s ≈ 0, | TJ ∞ ( s ) -J ∞ ( s ) | might be very large. This is hardly surprising, as the approximate Bellman operator evaluated at s is expected to have very high approximation errors. To summarize, if one were to use the supervised learning routine from [14], then AVI is stable when all the states are sampled a large number of times . With respect to the optimality of the limit, it varies over the state space. In particular, the limit of AVI, J ∞ , evaluated at states that are very well sampled will be close to their optimal cost-to-go values.

## 5.2 Relevance to literature

In solving large-scale sequential decision making problems AVI methods are a popular choice, due to their versatility and simplicity. Further, they facilitate in finding close-to-optimal solutions despite approximation and sampling errors. Ensuring the stablitiy of such methods is a challenge in many reinforcement learning applications. A major contribution of this paper, not addressed in previous literature, is the development of easily verifiable sufficient conditions for the almost sure boundedness of AVI methods involving set-valued dynamics.

An important contribution to the understanding of AVI methods has been due to Munos [14]. Here, the infinite horizon discounted cost problem is considered and solved using AVI. The analysis is performed when the approximation errors are bounded in the weighted p-norm sense, a significant improvement over [7] that only considered max norms. In particular, a strong rate of convergence result is presented. However, the basic procedure considered is a numerical AVI scheme where complete knowledge of the 'system model', i.e., the transition probabilities, is assumed. In addition, smoothness restrictions are imposed on the transition kernel. Such requirements are hard to verify within RL settings. For the asymptotic analysis presented herein, we do not require such assumptions.

We believe that the structure of iteration (5) is generic and is observed in many deep learning algorithms. For example, it occurs in the Q-learning procedure where the Bellman operator T is in fact the Q-Bellman operator. Here, no information is known about the system transition probabilities, i.e., we are in the 'model-free' setting. Our analysis is readily applicable to this scenario. Finally, note that our analysis works for both stochastic shortest path and infinite horizon discounted cost problems.

To summarize, we analyze AVI-like schemes for which (a) information on the transition probabilities is unknown, (b) there is a measurement error (albeit asymptotically bounded) that may arise for instance from the use of function approximation, and (c) the basic framework involves either the stochastic shortest path or the discounted cost setting.

## 6 Stability analysis of general set-valued stochastic approximations

Now that we have completed the analysis of AVI, we present a stability analysis of general set-valued stochastic approximations, under the Lyapunov stability assumptions presented in Section 4. Before we begin, we recall (7), the setvalued stochastic iterate:

<!-- formula-not-decoded -->

where y n ∈ H ( x n ) for all n ≥ 0 with H : R d → { subsets of R d } , { a ( n ) } n ≥ 0 is the given step-size sequence and { M n +1 } n ≥ 0 is the given noise sequence. Let us also recall the projective counterpart of (7), given by (8) for easy reference:

<!-- formula-not-decoded -->

where ˆ x 0 = x 0 and /intersectionsq B , C : R d → { subsets of R d } is the projection operator that projects onto set B , when the operand escapes from set C , i.e.,

<!-- formula-not-decoded -->

.

## 6.1 Analysis of the associated projective scheme

We study the stability properties of (7) by analyzing its hypothetical projective counterpart (8). Recall that the main purpose of ( A 3 a/b/c ) is to ensure the existence of an inward directing set. From Proposition 1, we get that C a/b/c is the required inward directing set. Further, from previous discussions we have that B a/b/c ⊂ C a/b/c . Since the roles of the variants ( A 3 a/b/c ), B a/b/c and C a/b/c are indistinguishable, with a slight abuse of notation, we generically refer to them using ( A 3), B and C , respectively.

Before proceeding, we make a quick note on notation. Previously, the projective iterates were represented by ˆ x n and the normal iterates by x n . Since we only consider the projective iterates in this section, for the sake of aesthetics, we abuse the notation slightly and simply use x n and omit the 'hat' notation. Hence, the projective scheme written using the new notation is given by:

<!-- formula-not-decoded -->

where y n ∈ H ( x n ) and x 0 ∈ /intersectionsq B , C (˜ x 0 ), with ˜ x 0 ∈ R d . Note that the initial point ˜ x 0 is first projected before starting the projective scheme. The above equation can be rewritten as

<!-- formula-not-decoded -->

Integral to our analysis is the construction of a linearly interpolated trajectory that has identical asymptotic behavior as the projective counterpart (19). We begin by dividing [0 , ∞ ) into diminishing intervals using the step-size sequence. Let t 0 := 0 and t n := n -1 ∑ m =0 a ( m ) for n ≥ 1. The linearly interpolated trajectory X l ( t ) is defined as follows:

<!-- formula-not-decoded -->

The above constructed trajectory is right continuous with left-hand limits, i.e., X l ( t ) = lim s ↓ t X l ( s ) and lim s ↑ t X l ( s ) exist. Further the jumps occur exactly at those t n 's for which the corresponding g n -1 's are non-zero. We also define three piece-wise constant trajectories X c ( · ), Y c ( · ) and G c ( · ) as follows: X c ( t ) := x n , Y c ( t ) := y n and G c ( t ) := n -1 ∑ m =0 g m for t ∈ [ t n , t n +1 ). The trajectories X c ( · ), Y c ( · ) and G c ( · ) are also right continuous with left-hand limits. We define a linearly interpolated trajectory associated with { M n +1 } n ≥ 0 as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We define a few 'left-shifted trajectories' using the above constructed trajectories. For t ≥ 0,

Lemma 3. X n l ( t ) = X n l (0) + t ∫ 0 Y n c ( τ ) dτ + W n l ( t ) + G n c ( t ) for t ≥ 0 .

Proof. PROOF: Fix s ∈ [ t m , t m +1 ) for some m ≥ 0. We have the following:

<!-- formula-not-decoded -->

Let us express X n l ( t ) in the form of the above equation. Note that t n + t ∈ [ t n + k , t n + k +1 ) for some k ≥ 0. Then we have the following:

<!-- formula-not-decoded -->

Unfolding x n + k , in the above equation till x n ( i.e., X n l (0)), yields:

<!-- formula-not-decoded -->

We make the following observations:

G n c ( t ) = n + k -1 ∑ l = n g l , W n l ( t n + k -t n ) = n + k -1 ∑ l = n a ( l ) M l +1 , W n l ( t ) = W n l ( t n + k -t n ) + ( t n + t -t n + k ) M n + k +1 and t ∫ 0 Y n c ( τ ) dτ = n + k -1 ∑ l = n a ( l ) y l +( t n + t -t n + k ) y n + k . As a consequence of the above observations, (21) becomes:

<!-- formula-not-decoded -->

Fix T &gt; 0. If { X n l ([0 , T ]) | n ≥ 0 } and { G n c ([0 , T ]) | n ≥ 0 } are viewed as subsets of D ([0 , T ] , R d ) equipped with the Skorohod topology, then we may use the Arzela-Ascoli theorem for D ([0 , T ] , R d ) to show that they are relatively compact, see Billingsley [8] for details. The Arzela-Ascoli theorem for D ([0 , T ] , R d ) states the following: A set S ⊆ D ([0 , T ] , R d ), is relatively compact if and only if the following conditions are satisfied:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If { X n l ([0 , T ]) | n ≥ 0 } and { G n c ([0 , T ]) | n ≥ 0 } are point-wise bounded and any two of their discontinuities are separated by at least ∆, for some fixed ∆ &gt; 0, then the above four conditions will be satisfied, see [8] for details.

Lemma 4. { X n l ([0 , T ]) | n ≥ 0 } and { G n c ([0 , T ]) | n ≥ 0 } are relatively compact in D ([0 , T ] , R d ) equipped with the Skorohod topology.

Proof. PROOF: Recall from ( A 2)( i ) that ‖ M n +1 ‖ ≤ D , n ≥ 0. Since H is Marchaud, it follows that sup x ∈C , y ∈ H ( x ) ‖ y ‖ ≤ C 1 for some C 1 &gt; 0 and that sup n ≥ 0 ‖ ˜ x n +1 - x n ‖ ≤ ( sup m ≥ 0 a ( m ) ) ( C 1 + D ). Further, ‖ g n ‖ ≤ ‖ ˜ x n +1 -x n ‖ + d ( x n , ∂ B ) ≤ C 2 for some constant C 2 that is independent of n . In other words, we have that

the sequences { X n l ([0 , T ]) | n ≥ 0 } and { G n c ([0 , T ]) | n ≥ 0 } are point-wise bounded. It remains to show that any two discontinuities are separated. Let ˜ d := min x ∈ ∂ C d ( x, B ) and D 1 := D + sup x ∈C sup y ∈ H ( x ) ‖ y ‖ . Clearly ˜ d &gt; 0. Define

<!-- formula-not-decoded -->

If there is a jump at t n , then x n ∈ ∂ B . It follows from the definition of m ( n ) that x k ∈ C for n ≤ k ≤ m ( n ). In other words, there are no discontinuities in the interval [ t n , t n + m ( n ) ) and t n + m ( n ) -t n ≥ ˜ d 2 D 1 . If we fix ∆ := ˜ d 2 D 1 &gt; 0, then any two discontinuities are separated by at least ∆. /squaresolid

Since T is arbitrary, it follows that { X n l ([0 , ∞ )) | n ≥ 0 } and { G n c ([0 , ∞ )) | n ≥ 0 } are relatively compact in D ([0 , ∞ ) , R d ). Since { W n l ([0 , T ]) | n ≥ 0 } is pointwise bounded (assumption ( A 2)( i )) and continuous, it is relatively compact in D ([0 , T ] , R d ). It follows from ( A 2)( ii ) that any limit of { W n l ([0 , T ]) | n ≥ 0 } , in D ([0 , T ] , R d ), is the constant 0 function. Suppose we consider sub-sequences of

{ X n l ([0 , T ]) | n ≥ 0 } and { X n l (0) + T ∫ 0 Y n c ( τ ) dτ + G n c ( T ) | n ≥ 0 } along which the aforementioned noise trajectories converge, then their limits are identical.

Consider { m ( n ) } n ≥ 0 ⊆ N along which { G m ( n ) c ([0 , T ]) } n ≥ 0 and { X m ( n ) l ([0 , T ]) } n ≥ 0 converge in D ([0 , T ] , R d ). Further, let g m ( n ) -1 = 0 for all n ≥ 0. Suppose the limit of { G m ( n ) c ([0 , T ]) } n ≥ 0 is the constant 0 function, then it can be shown that the limit of { X m ( n ) l ([0 , T ]) } n ≥ 0 is

<!-- formula-not-decoded -->

where X (0) ∈ C and Y ( t ) ∈ H ( X ( t )) for t ∈ [0 , T ]. The proof of this is along the lines of the proof of Theorem 2 in Chapter 5.2 of Borkar [10]. Suppose every limit of { G m ( n ) c ([0 , T ]) } n ≥ 0 is the constant 0 function whenever g m ( n ) -1 = 0 for all n ≥ 0, then every limit of { X m ( n ) l ([0 , T ]) } n ≥ 0 is a solution to ˙ X ( t ) ∈ H ( X ( t )). Suppose we show that the aforementioned statement is true for every T &gt; 0. Then, along with ( A 4) the stability of (7) is implied. Note that the set K := { n | g n = 0 } has infinite cardinality since any two discontinuities are at least ∆ &gt; 0 apart.

Lemma 5. Let K = { n | g n = 0 } . Without loss of generality let { X n l ([0 , T ]) } n ∈ K and { G n c ([0 , T ]) } n ∈ K be convergent, as n →∞ , in D ([0 , T ] , R d ) . Then X n l ( t ) →

Proof. PROOF: We begin by making the following observations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (b) Any two discontinuities of X ( · ) are at least ∆ apart.
- (c) G (0) = 0.
- (d) Solutions to ˙ x ( t ) ∈ H ( x ( t )) with starting points in C will not hit the boundary, ∂ C , later, i.e., they remain in the interior of C . This observation is a consequence of Proposition 1.

/negationslash

It follows from the above observations that ( i ) X ( t ) ∈ C for small values of t , ( ii ) τ := inf { t | t &gt; 0 , X ( t + ) = X ( t -) } and τ &gt; 0. It follows from the nature of convergence that ∃ τ ′ n &gt; τ &gt; τ n , n ≥ 0 such that

<!-- formula-not-decoded -->

For large values of n , X n l ( · ) has exactly one jump (point of discontinuity) at a point ˆ τ n ∈ [ τ n , τ ′ n ]. Let δ := ‖ X ( τ + ) -X ( τ -) ‖ &gt; 0, then for large values of n we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Also, X n l (ˆ τ -n ) is not in C and X n l (ˆ τ + n ) is in ∂ B . Further, since ˆ τ -n -τ n → 0, as n →∞ , it follows that

Hence, X ( τ -) / ∈ C . Similarly, we have that X ( τ + ) ∈ ∂ B . Observe that X ([0 , τ )) is a solution to ˙ x ( · ) ∈ H ( x ( · )) such that X (0) ∈ C , since G ( t ) = 0 for t ∈ [0 , τ ). Further, since C is inward directing , we have that X ( t ) ∈ C for t ∈ [0 , τ ). Since X ( t ) ∈ C for t &lt; τ and X ( τ -) / ∈ C we have X ( τ -) ∈ ∂ C .

Since X (0) ∈ C , we have that V ( X (0)) ≤ R , for some 0 &lt; R &lt; ∞ . As a consequence of our choice of C ( C is C a / C b / C c within the context of ( A 3 a )/( A 3 b )/( A 3 c )) we have V ( x ) = V ( y ) for any x, y ∈ ∂C , hence we may fix R := V ( x ) for any x ∈ ∂C . Fix τ 0 ∈ (0 , τ ), it follows from Proposition 1 that V ( X ( τ 0 )) &lt; R . Let t n ↑ τ such that t n ∈ ( τ 0 , τ ) for n ≥ 1. Without loss of generality, X ( t n ) → X ( τ -) and V ( X ( t n )) → V ( X ( τ -)), as t n → τ (else we may choose a subsequence of { t n } n ≥ 0 along which V ( X ( t n )) is convergent). Thus, ∃ N such that V ( X ( t n )) &gt; V ( X ( τ 0 )) for n ≥ N . Since X ([ τ 0 , t n ]) is a solution to ˙ x ( t ) ∈ H ( x ( t )) with starting point X ( τ 0 ), the aforementioned conclusion contradicts ( A 3 a )( iii )/( A 3 b )( iii )/( A 3 c )( iii ). In other words, X ( τ -) ∈ C and / ∈ ∂ C . Thus we have shown that there is no jump at τ , i.e., X ( τ + ) = X ( τ -). /squaresolid

Suppose ( A 3 a ) / ( A 3 b ) holds, then it follows from Proposition 3.25 of Bena¨ ım, Hofbauer and Sorin [5] that there is an attractor set A ⊆ Λ such that C a / C b is within the basin of attraction. Suppose ( A 3 c ) holds, then A is the global attractor of ˙ x ( t ) ∈ H ( x ( t )).

Lemma 6. The projective stochastic approximation scheme given by (18) converges to the attractor A .

Proof. PROOF: We begin by noting that T of Lemma 5 is arbitrary. Since G ≡ 0, after a certain number of iterations of (18), there are no projections, i.e., ˜ x n = x n for n ≥ N . Here N could be sample path dependent. Further, it follows from Lemma 5 that the projective scheme given by (18) tracks a solution to ˙ x ( t ) ∈ H ( x ( t )). In other words, the projective scheme given by (18) converges to a limit point of the DI , ˙ x ( t ) ∈ H ( x ( t )).

The iterates given by (18) are within C after sometime and they track a solution to ˙ x ( t ) ∈ H ( x ( t )). Since C is within the basin of attraction of A , the iterates converge to A . /squaresolid

The Lemmas proven in this section yield Theorem 4, the main result concerning the stability of (7), stated below. Then, it follows from Theorem 3.6 and Lemma 3.8 of Bena¨ ım [5] that (7) converges to a closed connected internally chain transitive invariant set associated with ˙ x ( t ) ∈ H ( x ( t )).

Theorem 4. Under ( A 1) -( A 4) and ( AV 4) , the set-valued SA given by (7) is stable (bounded almost surely).

Proof. PROOF: Let { ˆ x n } n ≥ 0 be the iterates generated by the projective scheme and { x n } n ≥ 0 the iterates generated by (7). It follows from Lemma 6 that ˆ x n → A such that A ⊂ B . In other words there exists N , possibly sample path dependent, such that ˆ x n ∈ B for all n ≥ N . Without loss of generality, this is the same N from ( A 4), else one can use the maximum of the two. Since B is a bounded set, we have that sup n ≥ N ‖ ˆ x n ‖ ≤ sup y ∈B ‖ y ‖ &lt; ∞ . It now follows from

<!-- formula-not-decoded -->

## 6.2 Relaxing assumption (A2)

The above stated Theorem 4 does not guarantee the stability of AVI, (5). This is because, the analysis involved in proving the aforementioned theorem requires that the Martingale noise satisfy ( A 2), instead of the weaker ( AV 5). Since only ( AV 5) is guaranteed for AVI, we need to prove Theorem 1, stated at the end of Section 4. It is a generalization of Theorem 4, in that it requires the less restrictive ( AV 5). For the benefit of the reader we recall ( AV 5) below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the above, F 0 := σ 〈 x 0 〉 and F n := σ 〈 x 0 , x 1 , . . . , x n , M 1 , . . . , M n 〉 for n ≥ 1. Further, without loss of generality, we may assume that K in ( AV 5) and ( A 1) are equal (otherwise we can use maximum of the two constants).

|

In what follows, we try to understand the role of ( A 2) in the analysis presented in Section 6.1. Then, we show that it can be readily modified to require the less restrictive ( AV 5). In analyzing the projective scheme given by (18), assumption ( A 2) is used in Lemma 4. Specifically, ( A 2)( i ) is used to show that any two discontinuities of { X l n ([0 , T ]) } n ≥ 0 and { G n c ([0 , T ]) } n ≥ 0 are separated by at least ∆ &gt; 0. We show that the aforementioned property holds when ( A 2) is replaced by ( AV 5). First, we prove an auxiliary result.

Lemma 7. Let { t m ( n ) , t l ( n ) } n ≥ 0 be such that t l ( n ) &gt; t m ( n ) , t m ( n +1) &gt; t l ( n ) and lim n →∞ ( t l ( n ) -t m ( n ) ) = 0 . Fix an arbitrary c &gt; 0 and consider the following:

<!-- formula-not-decoded -->

∥ ∥ Then P ( { ψ n &gt; c } i.o. ) = 0 within the context of the projective scheme given by (18).

Proof. PROOF: We shall show that ∑ n ≥ 0 P ( ψ n &gt; c ) &lt; ∞ . It follows from Chebyshev's inequality that

<!-- formula-not-decoded -->

Since { M n +1 } n ≥ 0 is a martingale difference sequence, we get:

<!-- formula-not-decoded -->

Within the context of the projective scheme given by (18), almost surely ∀ n, x n ∈ C , i.e., sup n ≥ 0 ‖ x n ‖ ≤ C 1 &lt; ∞ a.s. It follows from ( AV 5) that E [ ‖ M n +1 ‖ 2 ] ≤

<!-- formula-not-decoded -->

K ( 1 + E ‖ x n ‖ 2 ) . Hence, E [ ‖ M n +1 ‖ 2 ] ≤ K ( 1 + C 2 1 ) . Equation (22) becomes

Since t l ( n ) &gt; t m ( n ) and t m ( n +1) &gt; t l ( n ) , we have ∑ n ≥ 0 l ( n ) -1 ∑ i = m ( n ) a ( i ) 2 ≤ ∑ n ≥ 0 a ( n ) 2 . Finally we get,

<!-- formula-not-decoded -->

The claim now follows from the Borel-Cantelli lemma.

/squaresolid

Let us consider the scenario in which we cannot find ∆, the least separation between any two points of discontinuity. In other words, there exists { t ( m ( n )) , t ( l ( n )) } n ≥ 0 such that t l ( n ) &gt; t m ( n ) , t m ( n +1) &gt; t l ( n ) and lim n →∞ ( t l ( n ) -t m ( n ) ) = 0. Since we have assumed that there are no jumps between t ( m ( n )) and t ( l ( n )), we have X l ( t + m ( n ) ) ∈ ∂ B and X l ( t -l ( n ) ) / ∈ C for all n ≥ 0. The reader may note that every jump-point corresponds to a point in time, when the algorithm escapes C . We have

<!-- formula-not-decoded -->

We have that sup n ≥ 0 ‖ y n ‖ ≤ D ′ for some 0 &lt; D ′ &lt; ∞ , and ˜ d = min x ∈ ∂ C d ( x, B ). The above equation becomes

<!-- formula-not-decoded -->

∥ ∥ Since ( t l ( n ) -t m ( n ) ) → 0, for large n , ∥ ∥ ∥ ∥ ∥ l ( n ) -1 ∑ i = m ( n ) a ( i ) M i +1 ∥ ∥ ∥ ∥ ∥ &gt; d/ 2. This directly contradicts Lemma 7. Hence we can always find ∆ &gt; 0 separating any two points of discontinuity.

<!-- formula-not-decoded -->

In Lemma 6, ( A 2) is used to ensure the convergence of (18) to the attractor A . In Theorem 4, ( A 2) is used to ensure the convergence of (7) to a closed connected internally chain transitive invariant set of the associated DI . Specifically, it is ( A 2)( ii ) that ensures these convergences. Let us define ζ n := n -1 ∑ k =0 a ( k ) M k +1 , n ≥ 1. If { ζ n } n ≥ 1 converges, then it trivially follows that the martingale noise sequence satisfies ( A 2)( ii ). To show convergence, it is enough to show that the corresponding quadratic variation process converges almost surely. In other words, we need to show that ∑ n ≥ 0 a ( n ) 2 E ( ‖ M n +1 ‖ 2 |F n ) &lt; ∞ a.s or

<!-- formula-not-decoded -->

∑ n ≥ 0 E ( a ( n ) 2 ‖ M n +1 ‖ 2 ) &lt; ∞ . Consider the following:

Convergence of the quadratic variation process in the context of Lemma 6 follows from (23) and the fact that E ‖ x n ‖ 2 ≤ sup x ∈C ‖ x ‖ 2 . In other words,

<!-- formula-not-decoded -->

Similarly, for convergence in Theorem 4, it follows from (23) and stability of the iterates (sup n ≥ 0 ‖ x n ‖ &lt; ∞ a.s.) that

<!-- formula-not-decoded -->

In other words, both in Lemma 6 and Theorem 4, assumption ( A 2)( ii ) is satisfied. This gives us Theorem 1, a generalization of Theorem 4, stated at the end of Section 4.

## 6.3 Verifiability of ( A 4)

The verifiable sufficient conditions for ( A 4), along with the required proof are presented in the form of Lemma 8, below. The statement of this lemma is presented for a slightly more general from of a set-valued iteration, given by x n +1 ∈ G n ( x n , ξ n ) , n ≥ 0. If we define ξ n := M n +1 , x n := J n and G n ( J n , M n +1 ) := J n + a ( n ) ( TJ n -J n + B /epsilon1 n (0) + M n +1 ) , then it is easy to see that AVI, (5), has the aforementioned set-valued iterative form.

Lemma 8. Let B and C be open bounded subsets of R d such that B ⊂ C . Consider the algorithm

<!-- formula-not-decoded -->

We make the following assumptions:

1. { ξ n } n ≥ 0 is a random sequence that constitutes noise.
2. G n is an almost-surely-bounded diameter and contractive (in the first coordinate with second co-ordinate fixed) set-valued map, with respect to some metric ρ , for n ≥ 0 . In other words, H ρ ( G n ( x, ξ ) , G n ( y, ξ )) ≤ αρ ( x, y ) for some 0 &lt; α &lt; 1 and sup u,v ∈ G n ( x,ξ ) ρ ( u, v ) ≤ ˆ D , where 0 &lt;

ˆ D &lt; ∞ and x ∈ R d .

3. The projective sequence { ˆ x n } n ≥ 0 generated by

<!-- formula-not-decoded -->

converges to some vector x ∗ ∈ B .

Then, almost surely, ∃ N &lt; ∞ such that sup n ≥ N ρ (ˆ x n , x n ) &lt; ∞ .

Proof. PROOF: Since ˆ x n → x ∗ as n →∞ , there exists N such that ˆ x n ∈ B for all n ≥ N . For k ≥ 0 we have the following:

<!-- formula-not-decoded -->

The arguments used to obtain the above inequality are identical to the ones used to obtain (10) in the proof of Lemma 1 in Section 5. Unfolding the right hand side down to stage n we get the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 7 Finding fixed points of set-valued maps and Abstract Dynamic Programming

In Section 5 we showed that the AVI algorithm given by (5) is stable, and converges to a small neighborhood of the optimal cost-to-go vector J ∗ . For this, we started by observing that the fixed points of the perturbed Bellman operator belong to a small neighborhood of J ∗ as a consequence of the upper semicontinuity of attractor sets. Then we showed that (5) converges to a fixed point of the perturbed Bellman operator, thereby showing that (5) converges to a small neighborhood of J ∗ . In this section, we generalize the ideas of Sections 3 and 5 to develop and analyze an iterative algorithm for finding fixed points of general contractive set-valued maps.

To motivate the requirement of such a fixed point finding algorithm, we consider the field of Abstract Dynamic Programming which is an important extension of classical Dynamic Programming (DP), wherein the Bellman operator is defined in a more abstract manner, see [6]. As in classical DP, we are given: state space X , action space A , set of cost funtions R ( X ) := { J | J : X → R } and a set of valid policies M := { µ | µ : X → A} . Instead of a single-stage cost function which is then used to define the Bellman operator, in Abstract DP, one is given a function H : X ×A× R ( X ) → R . The Bellman operators are defined as follows:

<!-- formula-not-decoded -->

In [6], Bertsekas has presented sufficient conditions for the existence of a fixed point of the above Bellman operator and for its optimality. Algorithms for finding fixed points have an important role to play in Abstract DP. If we allow H to be set-valued or if H can only be estimated with non-zero bias, then the algorithm presented in this section can be helpful.

Suppose that we are given a set-valued map T : R d →{ subsets of R d } (need not be a 'set-valued counterpart of the Bellman operator'). We present sufficient conditions under which the following stochastic approximation algorithm is bounded a.s. and converges to a fixed point of T :

<!-- formula-not-decoded -->

where

- (i) y n ∈ Tx n -x n for all n ≥ 0.
- (iii) { M n +1 } n ≥ 0 is the martingale difference noise sequence satisfying ( AV 5).
- (ii) { a ( n ) } n ≥ 0 is the given step-size sequence satisfying ( AV 4).

## Definitions:

1. Given a metric space ( R d , ρ ), recall the Hausdorff metric with respect to ρ as follows:

<!-- formula-not-decoded -->

where A,B ⊂ R d and ρ ( u, C ) := min { ρ ( u, v ) | v ∈ C } for any u ∈ R d and C ⊆ R d .

2. We call a set-valued map T as contractive if and only if H ρ ( Tx,Ty ) ≤ αρ ( x, y ), where x, y ∈ R d and 0 &lt; α &lt; 1.
3. We say that T is of bounded diameter if and only if diam ( Tx ) ≤ D , ∀ x ∈ R d and given 0 &lt; D &lt; ∞ . Here diam ( A ) := sup { ρ ( z 1 , z 2 ) | z 1 , z 2 ∈ A } for any A ⊂ R d .

We impose the following restrictions on (24):

- (AF1) T is a Marchaud map that is of bounded diameter and contractive with respect to some metric ρ .
- (AF2) The metric ρ is such that ‖ x -y ‖ ≤ C ρ ( x, y ) for x, y ∈ R d , C &gt; 0.
- (AF3) Let F := { x | x ∈ Tx } denote the set of fixed points of T . There exists a compact subset F ′ ⊆ F along with a strongly positive invariant bounded open neighborhood.

OR

F is the unique global attractor of ˙ x ( t ) ∈ Tx ( t ) -x ( t ).

Since T is assumed to be contractive with respect to ρ , it follows from Theorem 5 of Nadler [15] that T has at least one fixed point. Assumption ( AF 2) is readily satisfied by the popular metric norms such as the weighted p-norms and the weighted max-norms among others. Assumption ( AF 3) is imposed to ensure that (24) satisfies ( A 3 b ) or ( A 3 c ). Specifically, ( AF 3) is imposed to ensure the existence of an inward directing set associated with ˙ x ( t ) ∈ Tx ( t ) -x ( t ), see Proposition 1 for details. In other words, we can find bounded open sets C F and B F such that C F is inward directing and B F ⊂ C F .

As in Section 3, we compare (24) with it's projective counterpart given by:

<!-- formula-not-decoded -->

where y n ∈ T ˆ x n -ˆ x n , { M n +1 } n ≥ 0 is identical for both (24) and (25) and /intersectionsq B F , C F ( · ) is the projection operator defined at the beginning of Section 6.1. The analysis

of the above projective scheme proceeds in an identical manner as in Section 6.1. Specifically, we may show that every limit point of the projective scheme (25) belongs to B F . The following theorem is immediate.

Theorem 5. Under ( AF 1) -( AF 3) and ( AV 5) , the iterates given by (24) are bounded almost surely. Further, any limit point of (24) (as n → ∞ ) is a fixed point of the set-valued map T .

Proof. PROOF: The proof of this theorem proceeds in a similar manner to that of Theorem 2. We only provide an outline here to avoid repetition. We begin by showing that (24) is bounded almost surely (stable) by comparing it to (25). Since the limit points of (25) belong to B F , there exists N , possibly sample path dependent, such that ˆ x n ∈ C F for all n ≥ N . For k ≥ 0, we have the following set of inequalities:

<!-- formula-not-decoded -->

where diam ( Tx ) ≤ D for every x ∈ R d . Recall that 0 &lt; α &lt; 1 is the contraction parameter of the set-valued map T . We consider two possible cases.

Case 1. 2 D ≤ (1 -α ) ρ ( x n + k , ˆ x n + k ) : In this case, it can be shown that

<!-- formula-not-decoded -->

Case 2. 2 D &gt; (1 -α ) ρ ( x n + k , ˆ x n + k ) : In this case, it can be shown that

<!-- formula-not-decoded -->

We conclude the following:

<!-- formula-not-decoded -->

It follows from the above inequality and ( AF 2) that (24) satisfies assumption ( A 4). Hence, we get that { x n } n ≥ 0 is bounded almost surely (stable).

Since the iterates are stable, it follows from [Theorem 2, Chapter 6, [2]] that every limit point of (24) is an equilibrium point of the set-valued map x ↦→ Tx -x . In other words, if x ∗ is a limit point of (24), then 0 ∈ Tx ∗ -x ∗ , i.e., x ∗ ∈ Tx ∗ . Hence we have shown that every limit point of (24) is a fixed point of the set-valued map T . /squaresolid

Remark 4. It is assumed that T is of bounded diameter, see ( AF 1) . This assumption is primarily required to show the almost sure boundedness of (24). Specifically, it is used to show that ( A 4) is satisfied. Depending on the problem at hand, one may wish to do away with this 'bounded diameter' assumption. For example, if we have sup n ≥ 0 diam ( Tx n ) &lt; ∞ a.s. instead, the bounded diameter assumption can be dispensed with.

Since T is Marchaud, it is point-wise bounded, i.e., sup z ∈ Tx ‖ z ‖ ≤ K (1 + ‖ x ‖ ) , where K &gt; 0 . In other words, diam ( Tx ) ≤ 2 K (1+ ‖ x ‖ ) . In principle, the pointwise boundedness of T does allow for unbounded diameters, i.e., diam ( Tx ) ↑ ∞ as ‖ x ‖ ↑ ∞ . Our bounded diameter assumption prevents this scenario from happening. In applications that use 'approximation operators', it is often reasonable to assume that the errors (due to approximations) are bounded. Then the 'associated set-valued map' is naturally of bounded diameter. The reader is referred to Section 3 for an example of this setting.

## 8 Conclusions

We analyzed the stability and convergence behaviors of the Approximate Value Iteration (AVI) method from Reinforcement Learning. Such approaches utilize an approximation of the Bellman operator within a fixed point finding iteration. We modelled the approximation errors as an additive non-diminishing random process that is asymptotically bounded. We were motivated by the use of neural networks as function approximators, usually trained in an online manner to minimize these errors. Although it is improbable that the approximation errors completely vanish, it is fair to expect that they remain bounded. This is because, unbounded errors mean that the Bellman operator is approximated very poorly at some points, and that the difference between the true and the approximate operator is infinite. Evaluating the Bellman operator requires taking expectations. Within the framework of deep RL, expectations are replaced by samples. Our analysis accounts for the sampling errors by modelling them as an additive martingale difference term, which is shown to vanish asymptotically.

An important contribution of our work is providing the set of Lyapunov function based stability conditions. In addition to using them to show the stability of AVI, we presented a stability analysis of general set-valued SAs based on the aforementioned assumptions. Regarding convergence of AVI, we showed that the limit is a fixed point of the perturbed Bellman operator. Further, it belongs to a small neighborhood of the optimal cost-to-go vector J ∗ .

In the future we would like to extend the ideas in this paper to consider on-policy reinforcement learning algorithms with function approximations. Additionally, we believe that our ideas can be extended to develop and analyze algorithms that solve constrained Markov decision processes. Finally, it would be interesting to see if the general Lyapunov function based stability conditions can be extended to stochastic approximations with set-valued maps and Markovian noise.

## References

- [1] J. Abounadi, D.P. Bertsekas, and V. Borkar. Stochastic approximation for nonexpansive maps: Application to q-learning algorithms. SIAM Journal on Control and Optimization , 41(1):1-22, 2002.
- [2] J. Aubin and A. Cellina. Differential Inclusions: Set-Valued Maps and Viability Theory . Springer, 1984.

- [3] M. Bena¨ ım. A dynamical system approach to stochastic approximations. SIAM J. Control Optim. , 34(2):437-472, 1996.
- [4] M. Bena¨ ım and M. W. Hirsch. Asymptotic pseudotrajectories and chain recurrent flows, with applications. J. Dynam. Differential Equations , 8:141176, 1996.
- [5] M. Bena¨ ım, J. Hofbauer, and S. Sorin. Stochastic approximations and differential inclusions. SIAM Journal on Control and Optimization , pages 328-348, 2005.
- [6] D.P. Bertsekas. Abstract dynamic programming . Athena Scientific Belmont, MA, 2013.
- [7] D.P. Bertsekas and J.N. Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1st edition, 1996.
- [8] P. Billingsley. Convergence of Probability Measures . John Wiley &amp; Sons, 2013.
- [9] V. S. Borkar. Stochastic approximation with two time scales. Syst. Control Lett. , 29(5):291-294, 1997.
- [10] V. S. Borkar. Stochastic Approximation: A Dynamical Systems Viewpoint . Cambridge University Press, 2008.
- [11] V. S. Borkar and S.P. Meyn. The O.D.E. method for convergence of stochastic approximation and reinforcement learning. SIAM J. Control Optim , 38:447-469, 1999.
- [12] D.P. De Farias and B. Van Roy. On the existence of fixed points for approximate value iteration and temporal-difference learning. Journal of Optimization theory and Applications , 105(3):589-608, 2000.
- [13] V. Mnih, K. Kavukcuoglu, D. Silver, A.A. Rusu, J. Veness, M.G. Bellemare, A. Graves, M. Riedmiller, A.K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. Nature , 518(7540):529, 2015.
- [14] R. Munos. Error bounds for approximate value iteration. In Proceedings of the National Conference on Artificial Intelligence , volume 20, page 1006, 2005.
- [15] S. Nadler. Multi-valued contraction mappings. Pacific Journal of Mathematics , 30(2):475-488, 1969.
- [16] A. Ramaswamy and S. Bhatnagar. A generalization of the Borkar-Meyn theorem for stochastic recursive inclusions. Mathematics of Operations Research , 42(3):648-661, 2017.
- [17] Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical statistics , pages 400-407, 1951.
- [18] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, et al. Mastering the game of go without human knowledge. Nature , 550(7676):354, 2017.