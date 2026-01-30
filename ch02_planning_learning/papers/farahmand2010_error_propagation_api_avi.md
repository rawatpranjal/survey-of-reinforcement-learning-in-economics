## Error Propagation for Approximate Policy and Value Iteration

## Amir massoud Farahmand

Department of Computing Science University of Alberta Edmonton, Canada, T6G 2E8

R´ emi Munos

Sequel Project, INRIA Lille

Lille, France remi.munos@inria.fr

amirf@ualberta.ca

Csaba Szepesv´ ari ∗

Department of Computing Science University of Alberta Edmonton, Canada, T6G 2E8

szepesva@ualberta.ca

## Abstract

We address the question of how the approximation error/Bellman residual at each iteration of the Approximate Policy/Value Iteration algorithms influences the quality of the resulted policy. We quantify the performance loss as the L p norm of the approximation error/Bellman residual at each iteration. Moreover, we show that the performance loss depends on the expectation of the squared Radon-Nikodym derivative of a certain distribution rather than its supremum - as opposed to what has been suggested by the previous results. Also our results indicate that the contribution of the approximation/Bellman error to the performance loss is more prominent in the later iterations of API/A VI, and the effect of an error term in the earlier iterations decays exponentially fast.

## 1 Introduction

The exact solution for the reinforcement learning (RL) and planning problems with large state space is difficult or impossible to obtain, so one usually has to aim for approximate solutions. Approximate Policy Iteration (API) and Approximate Value Iteration (A VI) are two classes of iterative algorithms to solve RL/Planning problems with large state spaces. They try to approximately find the fixedpoint solution of the Bellman optimality operator.

AVI starts from an initial value function V 0 (or Q 0 ), and iteratively applies an approximation of T ∗ , the Bellman optimality operator, (or T π for the policy evaluation problem) to the previous estimate, i.e., V k +1 ≈ T ∗ V k . In general, V k +1 is not equal to T ∗ V k because (1) we do not have direct access to the Bellman operator but only some samples from it, and (2) the function space in which V belongs is not representative enough. Thus there would be an approximation error ε k = T ∗ V k -V k +1 between the result of the exact VI and A VI.

Some examples of AVI-based approaches are tree-based Fitted Q-Iteration of Ernst et al. [1], multilayer perceptron-based Fitted Q-Iteration of Riedmiller [2], and regularized Fitted Q-Iteration of Farahmand et al. [3]. See the work of Munos and Szepesv´ ari [4] for more information about A VI.

∗ Csaba Szepesv´ ari is on leave from MTA SZTAKI. We would like to acknowledge the insightful comments by the reviewers. This work was partly supported by AICML, AITF, NSERC, and PASCAL2 under n o 216886.

API is another iterative algorithm to find an approximate solution to the fixed point of the Bellman optimality operator. It starts from a policy π 0 , and then approximately evaluates that policy π 0 , i.e., it finds a Q 0 that satisfies T π 0 Q 0 ≈ Q 0 . Afterwards, it performs a policy improvement step, which is to calculate the greedy policy with respect to (w.r.t.) the most recent action-value function, to get a new policy π 1 , i.e., π 1 ( · ) = arg max a ∈A Q 0 ( · , a ) . The policy iteration algorithm continues by approximately evaluating the newly obtained policy π 1 to get Q 1 and repeating the whole process again, generating a sequence of policies and their corresponding approximate action-value functions Q 0 → π 1 → Q 1 → π 2 → ··· . Same as AVI, we may encounter a difference between the approximate solution Q k ( T π k Q k ≈ Q k ) and the true value of the policy Q π k , which is the solution of the fixed-point equation T π k Q π k = Q π k . Two convenient ways to describe this error is either by the Bellman residual of Q k ( ε k = Q k -T π k Q k ) or the policy evaluation approximation error ( ε k = Q k -Q π k ).

API is a popular approach in RL literature. One well-known algorithm is LSPI of Lagoudakis and Parr [5] that combines Least-Squares Temporal Difference (LSTD) algorithm (Bradtke and Barto [6]) with a policy improvement step. Another API method is to use the Bellman Residual Minimization (BRM) and its variants for policy evaluation and iteratively apply the policy improvement step (Antos et al. [7], Maillard et al. [8]). Both LSPI and BRM have many extensions: Farahmand et al. [9] introduced a nonparametric extension of LSPI and BRM and formulated them as an optimization problem in a reproducing kernel Hilbert space and analyzed its statistical behavior. Kolter and Ng [10] formulated an l 1 regularization extension of LSTD. See Xu et al. [11] and Jung and Polani [12] for other examples of kernel-based extension of LSTD/LSPI, and Taylor and Parr [13] for a unified framework. Also see the proto-value function-based approach of Mahadevan and Maggioni [14] and iLSTD of Geramifard et al. [15].

A crucial question in the applicability of API/A VI, which is the main topic of this work, is to understand how either the approximation error or the Bellman residual at each iteration of API or A VI affects the quality of the resulted policy. Suppose we run API/A VI for K iterations to obtain a policy π K . Does the knowledge that all ε k s are small (maybe because we have had a lot of samples and used powerful function approximators) imply that V π K is close to the optimal value function V ∗ too? If so, how does the errors occurred at a certain iteration k propagate through iterations of API/AVI and affect the final performance loss?

There have already been some results that partially address this question. As an example, Proposition 6.2 of Bertsekas and Tsitsiklis [16] shows that for API applied to a finite MDP, we have lim sup k →∞ ‖ V ∗ -V π k ‖ ∞ ≤ 2 γ (1 -γ ) 2 lim sup k →∞ ‖ V π k -V k ‖ ∞ where γ is the discount facto. Similarly for AVI, if the approximation errors are uniformly bounded ( ‖ T ∗ V k -V k +1 ‖ ∞ ≤ ε ), we have lim sup k →∞ ‖ V ∗ -V π k ‖ ∞ ≤ 2 γ (1 -γ ) 2 ε (Munos [17]).

Nevertheless, most of these results are pessimistic in several ways. One reason is that they are expressed as the supremum norm of the approximation errors ‖ V π k -V k ‖ ∞ or the Bellman error ‖ Q k -T π k Q k ‖ ∞ . Compared to L p norms, the supremum norm is conservative. It is quite possible that the result of a learning algorithm has a small L p norm but a very large L ∞ norm. Therefore, it is desirable to have a result expressed in L p norm of the approximation/Bellman residual ε k .

In the past couple of years, there have been attempts to extend L ∞ norm results to L p ones [18, 17, 7]. As a typical example, we quote the following from Antos et al. [7]:

Proposition 1 (Error Propagation for API - [7]) . Let p ≥ 1 be a real and K be a positive integer. Then, for any sequence of functions { Q ( k ) } ⊂ B ( X × A ; Q max )(0 ≤ k &lt; K ) , the space of Q maxbounded measurable functions, and their corresponding Bellman residuals ε k = Q k -T π Q k , the following inequalities hold:

<!-- formula-not-decoded -->

where R max is an upper bound on the magnitude of the expected reward function and

<!-- formula-not-decoded -->

This result indeed uses L p norm of the Bellman residuals and is an improvement over results like Bertsekas and Tsitsiklis [16, Proposition 6.2], but still is pessimistic in some ways and does

not answer several important questions. For instance, this result implies that the uniform-over-alliterations upper bound max 0 ≤ k&lt;K ‖ ε k ‖ p,ν is the quantity that determines the performance loss. One may wonder if this condition is really necessary, and ask whether it is better to put more emphasis on earlier/later iterations? Or another question is whether the appearance of terms in the form of || d ( ρP π 1 ··· P πm ) dν || ∞ is intrinsic to the difficulty of the problem or can be relaxed.

The goal of this work is to answer these questions and to provide tighter upper bounds on the performance loss of API/AVI algorithms. These bounds help one understand what factors contribute to the difficulty of a learning problem. We base our analysis on the work of Munos [17], Antos et al. [7], Munos [18] and provide upper bounds on the performance loss in the form of ‖ V ∗ -V π k ‖ 1 ,ρ (the expected loss weighted according to the evaluation probability distribution ρ - this is defined in Section 2) for API (Section 3) and A VI (Section 4). This performance loss depends on a certain function of ν -weighted L 2 norms of ε k s, in which ν is the data sampling distribution, and C ρ,ν ( K ) that depends on the MDP, two probability distributions ρ and ν , and the number of iterations K .

In addition to relating the performance loss to L p norm of the Bellman residual/approximation error, this work has three main contributions that to our knowledge have not been considered before: (1) We show that the performance loss depends on the expectation of the squared Radon-Nikodym derivative of a certain distribution, to be specified in Section 3, rather than its supremum. The difference between this expectation and the supremum can be considerable. For instance, for a finite state space with N states, the ratio can be of order O ( N 1 / 2 ) . (2) The contribution of the Bellman/approximation error to the performance loss is more prominent in later iterations of API/AVI. and the effect of an error term in early iterations decays exponentially fast. (3) There are certain structures in the definition of concentrability coefficients that have not been explored before. We thoroughly discuss these qualitative/structural improvements in Section 5.

## 2 Background

In this section, we provide a very brief summary of some of the concepts and definitions from the theory of Markov Decision Processes (MDP) and reinforcement learning (RL) and a few other notations. For further information about MDPs and RL the reader is referred to [19, 16, 20, 21].

A finite-action discounted MDP is a 5-tuple ( X , A , P, R , γ ) , where X is a measurable state space, A is a finite set of actions, P is the probability transition kernel, R is the reward kernel, and 0 ≤ γ &lt; 1 is the discount factor. The transition kernel P is a mapping with domain X × A evaluated at ( x, a ) ∈ X × A that gives a distribution over X , which we shall denote by P ( ·| x, a ) . Likewise, R is a mapping with domain X × A that gives a distribution of immediate reward over R , which is denoted by R ( ·| x, a ) . We denote r ( x, a ) = E [ R ( ·| x, a )] , and assume that its absolute value is bounded by R max.

A mapping π : X → A is called a deterministic Markov stationary policy, or just a policy in short. Following a policy π in an MDP means that at each time step A t = π ( X t ) . Upon taking action A t at X t , we receive reward R t ∼ R ( ·| x, a ) , and the Markov chain evolves according to X t +1 ∼ P ( ·| X t , A t ) . We denote the probability transition kernel of following a policy π by P π , i.e., P π ( dy | x ) = P ( dy | x, π ( x )) .

The value function V π for a policy π is defined as V π ( x ) glyph[defines] E [ ∑ ∞ t =0 γ t R t ∣ ∣ ∣ X 0 = x ] and the action-value function is defined as Q π ( x, a ) glyph[defines] E [ ∑ ∞ t =0 γ t R t ∣ ∣ ∣ X 0 = x, A 0 = a ] . For a discounted MDP, we define the optimal value and action-value functions by V ∗ ( x ) = sup π V π ( x ) ( ∀ x ∈ X ) and Q ∗ ( x, a ) = sup π Q π ( x, a ) ( ∀ x ∈ X , ∀ a ∈ A ) . We say that a policy π ∗ is optimal if it achieves the best values in every state, i.e., if V π ∗ = V ∗ . We say that a policy π is greedy w.r.t. an action-value function Q and write π = ˆ π ( · ; Q ) , if π ( x ) ∈ arg max a ∈A Q ( x, a ) holds for all x ∈ X . Similarly, the policy π is greedy w.r.t. V , if for all x ∈ X , π ( x ) ∈ argmax a ∈A ∫ P ( dx ′ | x, a )[ r ( x, a ) + γV ( x ′ )] (If there exist multiple maximizers, some maximizer is chosen in an arbitrary deterministic manner). Greedy policies are important because a greedy policy w.r.t. Q ∗ (or V ∗ ) is an optimal policy. Hence, knowing Q ∗ is sufficient for behaving optimally (cf. Proposition 4.3 of [19]).

We define the Bellman operator for a policy π as ( T π V )( x ) glyph[defines] r ( x, π ( x )) + γ ∫ V π ( x ′ ) P ( dx ′ | x, a ) and ( T π Q )( x, a ) glyph[defines] r ( x, a ) + γ ∫ Q ( x ′ , π ( x ′ )) P ( dx ′ | x, a ) . Similarly, the Bellman optimality operator is defined as ( T ∗ V )( x ) glyph[defines] max a { r ( x, a ) + γ ∫ V ( x ′ ) P ( dx ′ | x, a ) } and ( T ∗ Q )( x, a ) glyph[defines] r ( x, a ) + γ ∫ max a ′ Q ( x ′ , a ′ ) P ( dx ′ | x, a ) .

For a measurable space X , with a σ -algebra σ X , we define M ( X ) as the set of all probability measures over σ X . For a probability measure ρ ∈ M ( X ) and the transition kernel P π , we define ρP π ( dx ′ ) = ∫ P ( dx ′ | x, π ( x )) dρ ( x ) . In words, ρ ( P π ) m ∈ M ( X ) is an m -step-ahead probability distribution of states if the starting state distribution is ρ and we follow P π for m steps. In what follows we shall use ‖ V ‖ p,ν to denote the L p ( ν ) -norm of a measurable function V : X → R : ‖ V ‖ p p,ν glyph[defines] ν | V | p glyph[defines] ∫ X | V ( x ) | p dν ( x ) . For a function Q : X × A ↦→ R , we define ‖ Q ‖ p p,ν glyph[defines] 1 |A| ∑ a ∈A ∫ X | Q ( x, a ) | p dν ( x ) .

## 3 Approximate Policy Iteration

Consider the API procedure and the sequence Q 0 → π 1 → Q 1 → π 2 → ··· → Q K -1 → π K , where π k is the greedy policy w.r.t. Q k -1 and Q k is the approximate action-value function for policy π k . For the sequence { Q k } K -1 k =0 , denote the B ellman R esidual (BR) and policy A pproximation E rror (AE) at each iteration by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The goal of this section is to study the effect of ν -weighted L 2 p norm of the Bellman residual sequence { ε BR k } K -1 k =0 or the policy evaluation approximation error sequence { ε AE k } K -1 k =0 on the performance loss ‖ Q ∗ -Q π K ‖ p,ρ of the outcome policy π K .

The choice of ρ and ν is arbitrary, however, a natural choice for ν is the sampling distribution of the data, which is used by the policy evaluation module. On the other hand, the probability distribution ρ reflects the importance of various regions of the state space and is selected by the practitioner. One common choice, though not necessarily the best, is the stationary distribution of the optimal policy.

Because of the dynamical nature of MDP, the performance loss ‖ Q ∗ -Q π K ‖ p,ρ depends on the difference between the sampling distribution ν and the future-state distribution in the form of ρP π 1 P π 2 · · · . The precise form of this dependence will be formalized in Theorems 3 and 4. Before stating the results, we require to define the following concentrability coefficients.

Definition 2 (Expected Concentrability of the Future-State Distribution) . Given ρ, ν ∈ M ( X ) , ν glyph[lessmuch] λ 1 ( λ is the Lebesgue measure), m ≥ 0 , and an arbitrary sequence of stationary policies { π m } m ≥ 1 , let ρP π 1 P π 2 . . . P π m ∈ M ( X ) denote the future-state distribution obtained when the first state is distributed according to ρ and then we follow the sequence of policies { π k } m k =1 .

Define the following concentrability coefficients that is used in API analysis:

<!-- formula-not-decoded -->

1 For two measures ν 1 and ν 2 on the same measurable space, we say that ν 1 is absolutely continuous with respect to ν 2 (or ν 2 dominates ν 1 ) and denote ν 1 glyph[lessmuch] ν 2 iff ν 2 ( A ) = 0 ⇒ ν 1 ( A ) = 0 .

with the understanding that if the future-state distribution ρ ( P π ∗ ) m 1 ( P π ) m 2 (or ρ ( P π ∗ ) m 1 ( P π 1 ) m 2 P π 2 or ρP π ∗ ) is not absolutely continuous w.r.t. ν , then we take c PI 1 ,ρ,ν ( m 1 , m 2 ; π ) = ∞ (similar for others).

Also define the following concentrability coefficient that is used in AVI analysis:

<!-- formula-not-decoded -->

with the understanding that if the future-state distribution ρ ( P π ∗ ) m 1 ( P π ) m 2 is not absolutely continuous w.r.t. ν , then we take c VI ,ρ,ν ( m 1 , m 2 ; π ) = ∞ .

In order to compactly present our results, we define the following notation:

<!-- formula-not-decoded -->

Theorem 3 (Error Propagation for API) . Let p ≥ 1 be a real number, K be a positive integer, and Q max ≤ R max 1 -γ . Then for any sequence { Q k } K -1 k =0 ⊂ B ( X × A , Q max ) (space of Q max-bounded measurable functions defined on X × A ) and the corresponding sequence { ε k } K -1 k =0 defined in (1) or (2) , we have

<!-- formula-not-decoded -->

. where E ( ε 0 , . . . , ε K -1 ; r ) = ∑ K -1 k =0 α 2 r k ‖ ε k ‖ 2 p 2 p,ν . (a) If ε k = ε BR for all 0 ≤ k &lt; K , we have

<!-- formula-not-decoded -->

(b) If ε k = ε AE for all 0 ≤ k &lt; K , we have

<!-- formula-not-decoded -->

## 4 Approximate Value Iteration

Consider the AVI procedure and the sequence V 0 → V 1 → ··· → V K -1 , in which V k +1 is the result of approximately applying the Bellman optimality operator on the previous estimate V k , i.e., V k +1 ≈ T ∗ V k . Denote the approximation error caused at each iteration by

<!-- formula-not-decoded -->

The goal of this section is to analyze A VI procedure and to relate the approximation error sequence { ε k } K -1 k =0 to the performance loss ‖ V ∗ -V π K ‖ p,ρ of the obtained policy π K , which is the greedy policy w.r.t. V K -1 .

Theorem 4 (Error Propagation for AVI) . Let p ≥ 1 be a real number, K be a positive integer, and V max ≤ R max 1 -γ . Then for any sequence { V k } K -1 k =0 ⊂ B ( X , V max ) , and the corresponding sequence { ε k } K -1 k =0 defined in (3) , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 5 Discussion

In this section, we discuss significant improvements of Theorems 3 and 4 over previous results such as [16, 18, 17, 7].

## 5.1 L p norm instead of L ∞ norm

As opposed to most error upper bounds, Theorems 3 and 4 relate ‖ V ∗ -V π K ‖ p,ρ to the L p norm of the approximation or Bellman errors ‖ ε k ‖ 2 p,ν of iterations in API/AVI. This should be contrasted with the traditional, and more conservative, results such as lim sup k →∞ ‖ V ∗ -V π k ‖ ∞ ≤ 2 γ (1 -γ ) 2 lim sup k →∞ ‖ V π k -V k ‖ ∞ for API (Proposition 6.2 of Bertsekas and Tsitsiklis [16]). The use of L p norm not only is a huge improvement over conservative supremum norm, but also allows us to benefit from the vast literature on supervised learning techniques, which usually provides error upper bounds in the form of L p norms, in the context of RL/Planning problems. This is especially interesting for the case of p = 1 as the performance loss ‖ V ∗ -V π K ‖ 1 ,ρ is the difference between the expected return of the optimal policy and the resulted policy π K when the initial state distribution is ρ . Convenient enough, the errors appearing in the upper bound are in the form of ‖ ε k ‖ 2 ,ν which is very common in the supervised learning literature. This type of improvement, however, has been done in the past couple of years [18, 17, 7] - see Proposition 1 in Section 1.

## 5.2 Expected versus supremum concentrability of the future-state distribution

The concentrability coefficients (Definition 2) reflect the effect of future-state distribution on the performance loss ‖ V ∗ -V π K ‖ p,ρ . Previously it was thought that the key contributing factor to the performance loss is the supremum of the Radon-Nikodym derivative of these two distributions. This is evident in the definition of C ρ,ν in Proposition 1 where we have terms in the form of || d ( ρ ( P π ) m ) dν || ∞

<!-- formula-not-decoded -->

Nevertheless, it turns out that the key contributing factor that determines the performance loss is the expectation of the squared Radon-Nikodym derivative instead of its supremum. Intuitively this implies that even if for some subset of X ′ ⊂ X the ratio d ( ρ ( P π ) m ) dν is large but the probability ν ( X ′ ) is very small, performance loss due to it is still small. This phenomenon has not been suggested by previous results.

As an illustration of this difference, consider a Chain Walk with 1000 states with a single policy that drifts toward state 1 of the chain. We start with ρ ( x ) = 1 201 for x ∈ [400 , 600] and zero everywhere else. Then we evaluate both || d ( ρ ( P π ) m ) dν || ∞ and ( E X ∼ ν [ | d ( ρ ( P π ) m ) dν | 2 ] ) 1 2 for m = 1 , 2 , . . . when ν is the uniform distribution. The result is shown in Figure 1a. One sees that the ratio is constant in the beginning, but increases when the distribution ρ ( P π ) m concentrates around state 1 , until it reaches steady-state. The growth and the final value of the expectation-based concentrability coefficient is much smaller than that of supremum-based.

Figure 1: (a) Comparison of E X ∼ ν [ | d ( ρ ( P π ) m ) dν | 2 ] 1 2 and ∥ ∥ ∥ d ( ρ ( P π ) m ) dν ∥ ∥ ∥ ∞ (b) Comparison of ‖ Q ∗ -Q k ‖ 1 for uniform and exponential data sampling schedule. The total number of samples is the same. [The Y -scale of both plots is logarithmic.]

<!-- image -->

It is easy to show that if the Chain Walk has N states and the policy has the same concentrating behavior and ν is uniform, then || d ( ρ ( P π ) m ) dν || ∞ → N , while ( E X ∼ ν [ | d ( ρ ( P π ) m ) dν | 2 ] ) 1 2 → √ N when m → ∞ . The ratio, therefore, would be of order Θ( √ N ) . This clearly shows the improvement of this new analysis in a simple problem. One may anticipate that this sharper behavior happens in many other problems too.

More generally, consider C ∞ = || dµ dν || ∞ and C L 2 = ( E X ∼ ν [ | dµ dν | 2 ] ) 1 2 . For a finite state space with N states and ν is the uniform distribution, C ∞ ≤ N but C L 2 ≤ √ N . Neglecting all other differences between our results and the previous ones, we get a performance upper bound in the form of ‖ Q ∗ -Q π K ‖ 1 ,ρ ≤ c 1 ( γ ) O ( N 1 / 4 ) sup k ‖ ε k ‖ 2 ,ν , while Proposition 1 implies that ‖ Q ∗ -Q π K ‖ 1 ,ρ ≤ c 2 ( γ ) O ( N 1 / 2 ) sup k || glyph[epsilon1] k || 2 ,ν . This difference between O ( N 1 / 4 ) and O ( N 1 / 2 ) shows a significant improvement.

## 5.3 Error decaying property

Theorems 3 and 4 show that the dependence of performance loss ‖ V ∗ -V π K ‖ p,ρ (or ‖ Q ∗ -Q π K ‖ p,ρ ) on { ε k } K -1 k =0 is in the form of E ( ε 0 , . . . , ε K -1 ; r ) = ∑ K -1 k =0 α 2 r k ‖ ε k ‖ 2 p 2 p,ν . This has a very special structure in that the approximation errors at later iterations have more contribution to the final performance loss. This behavior is obscure in previous results such as [17, 7] that the dependence of the final performance loss is expressed as E ( ε 0 , . . . , ε K -1 ; r ) = max k =0 ,...,K -1 ‖ ε k ‖ p,ν (see Proposition 1).

This property has practical and algorithmic implications too. It says that it is better to put more effort on having a lower Bellman or approximation error at later iterations of API/AVI. This, for instance, can be done by gradually increasing the number of samples throughout iterations, or to use more powerful, and possibly computationally more expensive, function approximators for the later iterations of API/A VI.

To illustrate this property, we compare two different sampling schedules on a simple MDP. The MDP is a 100 -state, 2 -action chain similar to Chain Walk problem in the work of Lagoudakis and Parr [5]. We use AVI with a lookup-table function representation. In the first sampling schedule, every 20 iterations we generate a fixed number of fresh samples by following a uniformly random walk on the chain (this means that we throw away old samples). This is the fixed strategy. In the exponential strategy, we again generate new samples every 20 iterations but the number of samples at the k th iteration is ck γ . The constant c is tuned such that the total number of both sampling strategy is almost the same (we give a slight margin of about 0 . 1% of samples in favor of the fixed strategy). What we compare is ‖ Q ∗ -Q k ‖ 1 ,ν when ν is the uniform distribution. The result can be seen in Figure 1b. The improvement of the exponential sampling schedule is evident. Of course, one

may think of more sophisticated sampling schedules but this simple illustration should be sufficient to attract the attention of practitioners to this phenomenon.

## 5.4 Restricted search over policy space

One interesting feature of our results is that it puts more structure and restriction on the way policies may be selected. Comparing C PI ,ρ,ν ( K ; r ) (Theorem 3) and C VI ,ρ,ν ( K ; r ) (Theorem 4) with C ρ,ν (Proposition 1) we see that:

(1) Each concentrability coefficient in the definition of C PI ,ρ,ν ( K ; r ) depends only on a single or two policies (e.g., π ′ k in c PI 1 ,ρ,ν ( K -k, m ; π ′ k ) ). The same is true for C VI ,ρ,ν ( K ; r ) . In contrast, the m th term in C ρ,ν has π 1 , . . . , π m as degrees of freedom, and this number is growing as m →∞ .

- (2) The operator sup in C PI ,ρ,ν and C VI ,ρ,ν appears outside the summation. Because of that, we only have K + 1 degrees of freedom π ′ 0 , . . . , π ′ K to choose from in API and remarkably only a single degree of freedom in AVI. On the other other hand, sup appears inside the summation in the definition of C ρ,ν . One may construct an MDP that this difference in the ordering of sup leads to an arbitrarily large ratio of two different ways of defining the concentrability coefficients.

(3) In API, the definitions of concentrability coefficients c PI 1 ,ρ,ν , c PI 2 ,ρ,ν , and c PI 3 ,ρ,ν (Definition 2) imply that if ρ = ρ ∗ , the stationary distribution induced by an optimal policy π ∗ , then

<!-- formula-not-decoded -->

coefficients). This special structure is hidden in the definition of C ρ,ν in Proposition 1, and instead we have an extra m 1 degrees of flexibility.

Remark 1. For general MDPs, the computation of concentrability coefficients in Definition 2 is difficult, as it is for similar coefficients defined in [18, 17, 7].

## 6 Conclusion

To analyze an API/AVI algorithm and to study its statistical properties such as consistency or convergence rate, we require to (1) analyze the statistical properties of the algorithm running at each iteration, and (2) study the way the policy approximation/Bellman errors propagate and influence the quality of the resulted policy.

The analysis in the first step heavily uses tools from the Statistical Learning Theory (SLT) literature, e.g., Gy¨ orfi et al. [22]. In some cases, such as A VI, the problem can be cast as a standard regression with the twist that extra care should be taken to the temporal dependency of data in RL scenario. The situation is a bit more complicated for API methods that directly aim for the fixed-point solution (such as LSTD and its variants), but still the same kind of tools from SLT can be used too - see Antos et al. [7], Maillard et al. [8].

The analysis for the second step is what this work has been about. In our Theorems 3 and 4, we have provided upper bounds that relate the errors at each iteration of API/A VI to the performance loss of the whole procedure. These bounds are qualitatively tighter than the previous results such as those reported by [18, 17, 7], and provide a better understanding of what factors contribute to the difficulty of the problem. In Section 5, we discussed the significance of these new results and the way they improve previous ones.

Finally, we should note that there are still some unaddressed issues. Perhaps the most important one is to study the behavior of concentrability coefficients c PI 1 ,ρ,ν ( m 1 , m 2 ; π ) , c PI 2 ,ρ,ν ( m 1 , m 2 ; π 1 , π 2 ) , and c VI ,ρ,ν ( m 1 , m 2 ; π ) as a function of m 1 , m 2 , and of course the transition kernel P of MDP. A better understanding of this question alongside a good understanding of the way each term ε k in E ( ε 0 , . . . , ε K -1 ; r ) behaves, help us gain more insight about the error convergence behavior of the RL/Planning algorithms.

## References

- [1] Damien Ernst, Pierre Geurts, and Louis Wehenkel. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research , 6:503-556, 2005.

- [2] Martin Riedmiller. Neural fitted Q iteration - first experiences with a data efficient neural reinforcement learning method. In 16th European Conference on Machine Learning , pages 317-328, 2005.
- [3] Amir-massoud Farahmand, Mohammad Ghavamzadeh, Csaba Szepesv´ ari, and Shie Mannor. Regularized fitted Q-iteration for planning in continuous-space markovian decision problems. In Proceedings of American Control Conference (ACC) , pages 725-730, June 2009.
- [4] R´ emi Munos and Csaba Szepesv´ ari. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research , 9:815-857, 2008.
- [5] Michail G. Lagoudakis and Ronald Parr. Least-squares policy iteration. Journal of Machine Learning Research , 4:1107-1149, 2003.
- [6] Steven J. Bradtke and Andrew G. Barto. Linear least-squares algorithms for temporal difference learning. Machine Learning , 22:33-57, 1996.
- [7] Andr´ as Antos, Csaba Szepesv´ ari, and R´ emi Munos. Learning near-optimal policies with Bellman-residual minimization based fitted policy iteration and a single sample path. Machine Learning , 71:89-129, 2008.
- [8] Odalric Maillard, R´ emi Munos, Alessandro Lazaric, and Mohammad Ghavamzadeh. Finitesample analysis of bellman residual minimization. In Proceedings of the Second Asian Conference on Machine Learning (ACML) , 2010.
- [9] Amir-massoud Farahmand, Mohammad Ghavamzadeh, Csaba Szepesv´ ari, and Shie Mannor. Regularized policy iteration. In D. Koller, D. Schuurmans, Y. Bengio, and L. Bottou, editors, Advances in Neural Information Processing Systems 21 , pages 441-448. MIT Press, 2009.
- [10] J. Zico Kolter and Andrew Y. Ng. Regularization and feature selection in least-squares temporal difference learning. In ICML '09: Proceedings of the 26th Annual International Conference on Machine Learning , pages 521-528, New York, NY, USA, 2009. ACM.
- [11] Xin Xu, Dewen Hu, and Xicheng Lu. Kernel-based least squares policy iteration for reinforcement learning. IEEE Trans. on Neural Networks , 18:973-992, 2007.
- [12] Tobias Jung and Daniel Polani. Least squares SVM for least squares TD learning. In In Proc. 17th European Conference on Artificial Intelligence , pages 499-503, 2006.
- [13] Gavin Taylor and Ronald Parr. Kernelized value function approximation for reinforcement learning. In ICML '09: Proceedings of the 26th Annual International Conference on Machine Learning , pages 1017-1024, New York, NY, USA, 2009. ACM.
- [14] Sridhar Mahadevan and Mauro Maggioni. Proto-value functions: A Laplacian framework for learning representation and control in markov decision processes. Journal of Machine Learning Research , 8:2169-2231, 2007.
- [15] Alborz Geramifard, Michael Bowling, Michael Zinkevich, and Richard S. Sutton. iLSTD: Eligibility traces and convergence analysis. In B. Sch¨ olkopf, J. Platt, and T. Hoffman, editors, Advances in Neural Information Processing Systems 19 , pages 441-448. MIT Press, Cambridge, MA, 2007.
- [16] Dimitri P. Bertsekas and John N. Tsitsiklis. Neuro-Dynamic Programming (Optimization and Neural Computation Series, 3) . Athena Scientific, 1996.
- [17] R´ emi Munos. Performance bounds in l p norm for approximate value iteration. SIAM Journal on Control and Optimization , 2007.
- [18] R´ emi Munos. Error bounds for approximate policy iteration. In ICML 2003: Proceedings of the 20th Annual International Conference on Machine Learning , 2003.
- [19] Dimitri P. Bertsekas and Steven E. Shreve. Stochastic Optimal Control: The Discrete-Time Case . Academic Press, 1978.
- [20] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction (Adaptive Computation and Machine Learning) . The MIT Press, 1998.
- [21] Csaba Szepesv´ ari. Algorithms for Reinforcement Learning . Morgan Claypool Publishers, 2010.
- [22] L´ aszl´ o Gy¨ orfi, Michael Kohler, Adam Krzy˙ zak, and Harro Walk. A Distribution-Free Theory of Nonparametric Regression . Springer Verlag, New York, 2002.