## Bruno Scherrer

Inria, Villers-l` es-Nancy, F-54600, France Universit´ e de Lorraine, LORIA, UMR 7503, Vandœuvre-l` es-Nancy, F-54506, France

## Abstract

We consider the infinite-horizon discounted optimal control problem formalized by Markov Decision Processes. We focus on several approximate variations of the Policy Iteration algorithm: Approximate Policy Iteration (API) (Bertsekas &amp; Tsitsiklis, 1996), Conservative Policy Iteration (CPI) (Kakade &amp; Langford, 2002), a natural adaptation of the Policy Search by Dynamic Programming algorithm (Bagnell et al., 2003) to the infinite-horizon case (PSDP ∞ ), and the recently proposed Non-Stationary Policy Iteration (NSPI( m )) (Scherrer &amp; Lesner, 2012). For all algorithms, we describe performance bounds with respect the per-iteration error glyph[epsilon1] , and make a comparison by paying a particular attention to the concentrability constants involved, the number of iterations and the memory required. Our analysis highlights the following points: 1) The performance guarantee of CPI can be arbitrarily better than that of API, but this comes at the cost of a relative-exponential in 1 glyph[epsilon1] -increase of the number of iterations. 2) PSDP ∞ enjoys the best of both worlds: its performance guarantee is similar to that of CPI, but within a number of iterations similar to that of API. 3) Contrary to API that requires a constant memory, the memory needed by CPI and PSDP ∞ is proportional to their number of iterations, which may be problematic when the discount factor γ is close to 1 or the approximation error glyph[epsilon1] is close to 0 ; we show that the NSPI( m ) algorithm allows to make an overall trade-off between memory and performance. Simulations with these schemes confirm our analysis.

Proceedings of the 31 st International Conference on Machine Learning , Beijing, China, 2014. JMLR: W&amp;CP volume 32. Copyright 2014 by the author(s).

## 1. Introduction

We consider an infinite-horizon discounted Markov Decision Process (MDP) (Puterman, 1994; Bertsekas &amp; Tsitsiklis, 1996) ( S , A , P, r, γ ) , where S is a possibly infinite state space, A is a finite action space, P ( ds ′ | s, a ) , for all ( s, a ) , is a probability kernel on S , r : S → [ -R max , R max ] is a reward function bounded by R max , and γ ∈ (0 , 1) is a discount factor. A stationary deterministic policy π : S → A maps states to actions. We write P π ( ds ′ | s ) = P ( ds ′ | s, π ( s )) for the stochastic kernel associated to policy π . The value v π of a policy π is a function mapping states to the expected discounted sum of rewards received when following π from these states: for all s ∈ S ,

<!-- formula-not-decoded -->

The value v π is clearly bounded by V max = R max / (1 -γ ) . It is well-known that v π can be characterized as the unique fixed point of the linear Bellman operator associated to a policy π : T π : v ↦→ r + γP π v . Similarly, the Bellman optimality operator T : v ↦→ max π T π v has as unique fixed point the optimal value v ∗ = max π v π . A policy π is greedy w.r.t. a value function v if T π v = Tv , the set of such greedy policies is written G v . Finally, a policy π ∗ is optimal, with value v π ∗ = v ∗ , iff π ∗ ∈ G v ∗ , or equivalently T π ∗ v ∗ = v ∗ .

The goal of this paper is to study and compare several approximate Policy Iteration schemes. In the literature, such schemes can be seen as implementing an approximate greedy operator, G glyph[epsilon1] , that takes as input a distribution ν and a function v : S → R and returns a policy π that is ( glyph[epsilon1], ν ) -approximately greedy with respect to v in the sense that:

<!-- formula-not-decoded -->

where for all x , νx denotes E s ∼ ν [ x ( s )] . In practice, this approximation of the greedy operator can be achieved through a glyph[lscript] p -regression of the so-called Q-function -the stateaction value function-(a direct regression is suggested by Kakade &amp; Langford (2002), a fixed-point LSTD approach is used by Lagoudakis &amp; Parr (2003b)) or through a

## Approximate Policy Iteration Schemes: A Comparison

BRUNO.SCHERRER@INRIA.FR

(cost-sensitive) classification problem (Lagoudakis &amp; Parr, 2003a; Lazaric et al., 2010). With this operator in hand, we shall describe several Policy Iteration schemes in Section 2. Then Section 3 will provide a detailed comparative analysis of their performance guarantees, time complexities, and memory requirements. Section 4 will go on by providing experiments that will illustrate their behavior, and confirm our analysis. Finally, Section 5 will conclude and present future work.

## 2. Algorithms

API We begin by describing the standard Approximate Policy Iteration (API) (Bertsekas &amp; Tsitsiklis, 1996). At each iteration k , the algorithm switches to the policy that is approximately greedy with respect to the value of the previous policy for some distribution ν :

<!-- formula-not-decoded -->

If there is no error ( glyph[epsilon1] k = 0 ) and ν assigns a positive weights to every state, it can easily be seen that this algorithm generates the same sequence of policies as exact Policy Iterations since from Equation (1) the policies are exactly greedy.

CPI/CPI( α )/API( α ) We now turn to the description of Conservative Policy Iteration (CPI) proposed by (Kakade &amp;Langford, 2002). At iteration k , CPI (described in Equation (3)) uses the distribution d π k ,ν = (1 -γ ) ν ( I -γP π k ) -1 -the discounted cumulative occupancy measure induced by π k when starting from ν -for calling the approximate greedy operator, and uses a stepsize α k to generate a stochastic mixture of all the policies that are returned by the successive calls to the approximate greedy operator, which explains the adjective 'conservative':

<!-- formula-not-decoded -->

The stepsize α k +1 can be chosen in such a way that the above step leads to an improvement of the expected value of the policy given that the process is initialized according to the distribution ν (Kakade &amp; Langford, 2002). The original article also describes a criterion for deciding whether to stop or to continue. Though the adaptive stepsize and the stopping condition allows to derive a nice analysis, they are in practice conservative: the stepsize α k should be implemented with a line-search mechanism, or be fixed to some small value α . We will refer to this latter variation of CPI as CPI( α ).

It is natural to also consider the algorithm API( α ) (mentioned by Lagoudakis &amp; Parr (2003a)), a variation of API that is conservative like CPI( α ) in the sense that it mixes the new policy with the previous ones with weights α and

1 -α , but that directly uses the distribution ν in the approximate greedy step:

<!-- formula-not-decoded -->

Because it uses ν instead of d π k ,ν , API( α ) is simpler to implement than CPI( α ) 1 .

PSDP ∞ We are now going to describe an algorithm that has a flavour similar to API-in the sense that at each step it does a full step towards a new deterministic policybut also has a conservative flavour like CPI-in the sense that the policies considered evolve more and more slowly. This algorithm is a natural variation of the Policy Search by Dynamic Programming algorithm (PSDP) of Bagnell et al. (2003), originally proposed to tackle finite-horizon problems, to the infinite-horizon case; we thus refer to it as PSDP ∞ . To the best of our knowledge however, this variation has never been used in an infinite-horizon context.

The algorithm is based on finite-horizon non-stationary policies. Given a sequence of stationary deterministic policies ( π k ) that the algorithm will generate, we will write σ k = π k π k -1 . . . π 1 the k -horizon policy that makes the first action according to π k , then the second action according to π k -1 , etc. Its value is v σ k = T π k T π k -1 . . . T π 1 r . We will write ∅ the 'empty' non-stationary policy. Note that v ∅ = r and that any infinite-horizon policy that begins with σ k = π k π k -1 . . . π 1 , which we will (somewhat abusively) denote ' σ k . . . ' has a value v σ k ... ≥ v σ k -γ k V max . Starting from σ 0 = ∅ , the algorithm implicitely builds a sequence of non-stationary policies ( σ k ) by iteratively concatenating the policies that are returned by the approximate greedy operator:

<!-- formula-not-decoded -->

While the standard PSDP algorithm of Bagnell et al. (2003) considers a horizon T and makes T iterations, the algorithm we consider here has an indefinite number of iterations. The algorithm can be stopped at any step k . The theory that we are about to describe suggests that one may return any policy that starts by the non-stationary policy σ k . Since σ k is an approximately good finite-horizon policy, and as we consider an infinite-horizon problem, a natural output that one may want to use in practice is the infinitehorizon policy that loops over σ k , that we shall denote ( σ k ) ∞ .

1 In practice, controlling the greedy step with respect to d π k ,ν requires to generate samples from this very distribution. As explained by Kakade &amp; Langford (2002), one such sample can be done by running one trajectory starting from ν and following π k , stopping at each step with probability 1 -γ . In particular, one sample from d π k ,ν requires on average 1 1 -γ samples from the underlying MDP. With this respect, API( α ) is much simpler to implement.

From a practical point of view, PSDP ∞ and CPI need to store all the (stationary deterministic) policies generated from the start. The memory required by the algorithmic scheme is thus proportional to the number of iterations, which may be prohibitive. The aim of the next paragraph, that presents the last algorithm of this article, is to describe a solution to this potential memory issue.

NSPI( m ) We originally devised the algorithmic scheme of Equation (5) (PSDP ∞ ) as a simplified variation of the Non-Stationary PI algorithm with a growing period algorithm (NSPI-growing) (Scherrer &amp; Lesner, 2012) 2 . With respect to Equation (5), the only difference of NSPIgrowing resides in the fact that the approximate greedy step is done with respect to the value v ( σ k ) ∞ of the policy that loops infinitely over σ k (formally the algorithm does π k +1 ←G glyph[epsilon1] k +1 ( ν, v ( σ k ) ∞ ) ) instead of the value v σ k of only the first k steps here. Following the intuition that when k is big, these two values will be close to each other, we ended up considering PSDP ∞ because it is simpler. NSPIgrowing suffers from the same memory drawback as CPI and PSDP ∞ . Interestingly, the work of Scherrer &amp; Lesner (2012) contains another algorithm, Non-Stationary PI with a fixed period (NSPI( m )), that has a parameter that directly controls the number of policies stored in memory.

Similarly to PSDP ∞ , NSPI( m ) is based on non-stationary policies. It takes as an input a parameter m . It requires a set of m initial deterministic stationary policies π m -1 , π m -2 , . . . , π 0 and iteratively generates new policies π 1 , π 2 , . . . . For any k ≥ 0 , we shall denote σ m k the m -horizon non-stationary policy that runs in reverse order the last m policies, which one may write formally: σ m k = π k π k -1 . . . π k -m +1 . Also, we shall denote ( σ m k ) ∞ the m -periodic infinite-horizon nonstationary policy that loops over σ m k . Starting from σ m 0 = π 0 π 1 . . . π m -1 , the algorithm iterates as follows:

<!-- formula-not-decoded -->

Each iteration requires to compute an approximate greedy policy π k +1 with respect to the value v ( σ k m ) ∞ of ( σ m k ) ∞ , that is the fixed point of the compound operator 3 :

<!-- formula-not-decoded -->

When one goes from iterations k to k +1 , the process consists in adding π k +1 at the front of the ( m -1) -horizon policy π k π k -1 . . . π k -m +2 , thus forming a new m -horizon

2 We later realized that it was in fact a very natural variation of PSDP. To 'give Caesar his due and God his', we kept as the main reference the older work and gave the name PSDP ∞ .

3 Implementing this algorithm in practice can trivially be done through cost-sensitive classification in a way similar to Lazaric et al. (2010). It could also be done with a straight-forward extension of LSTD( λ ) to non-stationary policies.

policy σ m k +1 . Doing so, we forget about the oldest policy π k -m +1 of σ m k and keep a constant memory of size m . At any step k , the algorithm can be stopped, and the output is the policy π k,m = ( σ m k ) ∞ that loops on σ m k . It is easy to see that NSPI( m ) reduces to API when m = 1 . Furthermore, if we assume that the reward function is positive, add 'stop actions' in every state of the model that lead to a terminal absorbing state with a null reward, and initialize with an infinite sequence of policies that only take this 'stop action', then NSPI( m ) with m = ∞ reduces to PSDP ∞ .

## 3. Analysis

For all considered algorithms, we are going to describe bounds on the expected loss E s ∼ µ [ v π ∗ ( s ) -v π ( s )] = µ ( v π ∗ -v π ) of using the (possibly stochastic or nonstationary) policy π ouput by the algorithms instead of the optimal policy π ∗ from some initial distribution µ of interest as a function of an upper bound glyph[epsilon1] on all errors ( glyph[epsilon1] k ) . In order to derive these theoretical guarantees, we will first need to introduce a few concentrability coefficients that relate the distribution µ with which one wants to have a guarantee, and the distribution ν used by the algorithms 4 .

Definition 1. Let c (1) , c (2) , . . . be the smallest coefficients in [1 , ∞ ) ∪{∞} such that for all i and all sets of deterministic stationary policies π 1 , π 2 , . . . , π i , µP π 1 P π 2 . . . P π i ≤ c ( i ) ν . For all m,k , we define the following coefficients in [1 , ∞ ) ∪ {∞} :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, let c π ∗ (1) , c π ∗ (2) , . . . be the smallest coefficients in [1 , ∞ ) ∪{∞} such that for all i , µ ( P π ∗ ) i ≤ c π ∗ ( i ) ν . We define:

<!-- formula-not-decoded -->

Finally let C π ∗ be the smallest coefficient in [1 , ∞ ) ∪{∞} such that d π ∗ ,µ = (1 -γ ) µ ( I -γP π ∗ ) -1 ≤ C π ∗ ν .

With these notations in hand, our first contribution is to provide a thorough comparison of all the algorithms. This is done in Table 1. For each algorithm, we describe some performance bounds and the required number of iterations and memory. To make things clear, we only display the dependence with respect to the concentrability constants, the

4 The expected loss corresponds to some weighted glyph[lscript] 1 -norm of the loss v π ∗ -v π . Relaxing the goal to controlling the weighted glyph[lscript] p -norm for some p ≥ 2 allows to introduce some finer coefficients (Farahmand et al., 2010; Scherrer et al., 2012). Due to lack of space, we do not consider this here.

Table 1. Upper bounds on the performance guarantees for the algorithms. Except when references are given, the bounds are to our knowledge new. A comparison of API and CPI based on the two known bounds was done by Ghavamzadeh &amp; Lazaric (2012). The first bound of NSPI( m ) can be seen as an adaptation of that provided by Scherrer &amp; Lesner (2012) for the more restrictive glyph[lscript] ∞ -norm setting.

| Algorithm                                         | Performance Bound                                                                               | Performance Bound                                                 | Performance Bound                                                                                           | # Iter.                                                                                                                 | Memory                                      | Reference                |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------|--------------------------|
| API (Eq. (2)) ( = NSPI(1))                        | C (2 , 1 , 0) C (1 , 0)                                                                         | 1 (1 - γ ) 2 1 (1 - γ ) 2                                         | glyph[epsilon1] glyph[epsilon1] log 1 glyph[epsilon1]                                                       | 1 1 - γ log 1 glyph[epsilon1]                                                                                           | 1                                           | (Lazaric et al., 2010)   |
| API( α ) (Eq. (4)                                 | C (1 , 0)                                                                                       | 1 (1 - γ ) 2                                                      | glyph[epsilon1]                                                                                             | 1 α (1 - γ ) log                                                                                                        | 1 glyph[epsilon1]                           |                          |
| CPI( α )                                          | C (1 , 0)                                                                                       | 1 (1 - γ ) 3                                                      | glyph[epsilon1]                                                                                             | 1 α (1 - γ )                                                                                                            | log 1 glyph[epsilon1]                       |                          |
| CPI (Eq. (3))                                     | C (1 , 0) C π ∗                                                                                 | 1 (1 - γ ) 3 1 (1 - γ ) 2                                         | glyph[epsilon1] log 1 glyph[epsilon1] glyph[epsilon1]                                                       | 1 1 - γ 1 glyph[epsilon1] γ glyph[epsilon1] 2                                                                           | log 1 glyph[epsilon1]                       | (Kakade &Langford, 2002) |
| PSDP ∞ (Eq. (5)) ( glyph[similarequal] NSPI( ∞ )) | C π ∗ C (1) π ∗                                                                                 | 1 (1 - γ ) 2 1 1 - γ                                              | glyph[epsilon1] log 1 glyph[epsilon1] glyph[epsilon1]                                                       | 1 1 - γ 1 1 - γ                                                                                                         | log 1 glyph[epsilon1] log 1 glyph[epsilon1] |                          |
| NSPI( m ) (Eq. (6))                               | C (2 ,m, 0) C (1 , 0) m C (1) π ∗ + γ mC (2 ,m,m ) 1 - γ m C π ∗ + γ m C (2 ,m, 0) m (1 - γ m ) | 1 (1 - γ )(1 - γ m ) 1 (1 - γ ) 2 (1 - γ m ) 1 1 - γ 1 (1 - γ ) 2 | glyph[epsilon1] glyph[epsilon1] log 1 glyph[epsilon1] glyph[epsilon1] glyph[epsilon1] log 1 glyph[epsilon1] | 1 1 - γ log 1 glyph[epsilon1] 1 1 - γ log 1 glyph[epsilon1] 1 1 - γ log 1 glyph[epsilon1] 1 1 - γ log 1 glyph[epsilon1] | m                                           |                          |

Figure 1. Hierarchy of the concentrability constants. A constant A is better than a constant B -see the text for details-if A is a parent of B on the above graph. The best constant is C π ∗ .

<!-- image -->

discount factor γ , the quality glyph[epsilon1] of the approximate greedy operator, and-if applicable-the main parameters α / m of the algorithms. For API( α ), CPI( α ), CPI and PSDP ∞ , the required memory matches the number of iterations. All but two bounds are to our knowledge original. The derivation of the new results are given in Appendix A.

Our second contribution, that is complementary with the comparative list of bounds, is that we can show that there exists a hierarchy among the constants that appear in all the bounds of Table 1. In the directed graph of Figure 1, a constant B is a descendent of A if and only if the implication { B &lt; ∞ ⇒ A &lt; ∞} holds 5 . The 'if and only if' is important here: it means that if A is a parent of B , and B is not a parent of A , then there exists an MDP for which A

5 Dotted arrows are used to underline the fact that the comparison of coefficients is restricted to the case where the parameter m is finite.

is finite while B is infinite; in other words, an algorithm that has a guarantee with respect to A has a guarantee that can be arbitrarily better than that with constant B . Thus, the overall best concentrability constant is C π ∗ , while the worst are C (2 , 1 , 0) and C (2 ,m, 0) . To make the picture complete, we should add that for any MDP and any distribution µ , it is possible to find an input distribution ν for the algorithm (recall that the concentrability coefficients depend on ν and µ ) such that C π ∗ is finite, though it is not the case for C (1) π ∗ (and as a consequence all the other coefficients). The derivation of this order relations is done in Appendix B.

The standard API algorithm has guarantees expressed in terms of C (2 , 1 , 0) and C (1 , 0) only. Since CPI's analysis can be done with respect to C π ∗ , it has a performance guarantee that can be arbitrarily better than that of API, though the opposite is not true. This, however, comes at the cost of an exponential increase of time complexity since CPI may require a number of iterations that scales in O ( 1 glyph[epsilon1] 2 ) , while the guarantee of API only requires O ( log 1 glyph[epsilon1] ) iterations. When the analysis of CPI is relaxed so that the performance guarantee is expressed in terms of the (worse) coefficient C (1 , 0) (obtained also for API), we can slightly improve the rateto ˜ O ( 1 glyph[epsilon1] ) -, though it is still exponentially slower than that of API. This second result for CPI was proved with a technique that was also used for CPI( α ) and API( α ). We conjecture that it can be improved for CPI( α ), that should be as good as CPI when α is sufficiently small.

PSDP ∞ enjoys two guarantees that have a fast rate like those of API. One bound has a better dependency with respect to 1 1 -γ , but is expressed in terms of the worse coefficient C (1) π ∗ . The second guarantee is almost as good as that

of CPI since it only contains an extra log 1 glyph[epsilon1] term, but it has the nice property that it holds quickly with respect to glyph[epsilon1] : in time O (log 1 glyph[epsilon1] ) instead of O ( 1 glyph[epsilon1] 2 ) , that is exponentially faster. PSDP ∞ is thus theoretically better than both CPI (as good but faster) and API (better and as fast).

Now, from a practical point of view, PSDP ∞ and CPI need to store all the policies generated from the start. The memory required by these algorithms is thus proportional to the number of iterations. Even if PSDP ∞ may require much fewer iterations than CPI, the corresponding memory requirement may still be prohibitive in situations where glyph[epsilon1] is small or γ is close to 1 . We explained that NSPI( m ) can be seen as making a bridge between API and PSDP ∞ . Since (i) both have a nice time complexity, (ii) API has the best memory requirement, and (iii) NSPI( m ) has the best performance guarantee, NSPI( m ) is a good candidate for making a standard performance/memory trade-off. If the first two bounds of NSPI( m ) in Table 1 extends those of API, the other two are made of two terms: the left terms are identical to those obtained for PSDP ∞ , while the two possible right terms are new, but are controlled by γ m , which can thus be made arbitrarily small by increasing the memory parameter m . Our analysis thus confirms our intuition that NSPI( m ) allows to make a performance/memory trade-off in between API (small memory) and PSDP ∞ (best performance). In other words, as soon as memory becomes a constraint, NSPI( m ) is the natural alternative to PSDP ∞ .

## 4. Experiments

In this section, we present some experiments in order to illustrate the empirical behavior of the different algorithms discussed in the paper. We considered the standard API as a baseline. CPI, as it is described by Kakade &amp; Langford (2002), is very slow (in one sample experiment on a 100 state problem, it made very slow progress and took several millions of iterations before it stopped) and we did not evaluate it further. Instead, we considered two variations: CPI+ that is identical to CPI except that it chooses the step α k at each iteration by doing a line-search towards the policy output by the greedy operator 6 , and CPI( α ) with α = 0 . 1 , that makes 'relatively but not too small' steps at each iteration. To assess the utility for CPI to use the distribution d ν,π for the approximate greedy step, we also considered API( α ) with α = 0 . 1 , the variation of API described in Equation (4) that makes small steps, and that only differs from CPI( α ) by the fact that the approximate greedy step uses the distribution ν instead of d π k ,ν . In addition to these algorithms, we considered PSDP ∞ and NSPI( m ) for the values m ∈ { 5 , 10 , 30 } .

6 We implemented a crude line-search mechanism, that looks on the set 2 i α where α is the minimal step estimated by CPI to ensure improvement.

In order to assess their quality, we consider finite problems where the exact value function can be computed. More precisely, we consider Garnet problems first introduced by Archibald et al. (1995), which are a class of randomly constructed finite MDPs. They do not correspond to any specific application, but remain representative of the kind of MDP that might be encountered in practice. In brief, we consider Garnet problems with |S| ∈ { 50 , 100 , 200 } , |A| ∈ { 2 , 5 , 10 } and branching factors in { 1 , 2 , 10 } . The greedy step used by all algorithms is approximated by an exact greedy operator applied to a noisy orthogonal projection on a linear space of dimension |S| 10 with respect to the quadratic norm weighted by ν or d ν,π (for CPI+ and CPI( α )) where ν is uniform.

For each of these 3 3 = 27 parameter instances, we generated 30 i.i.d. Garnet MDPs ( M i ) 1 ≤ i ≤ 30 . For each such MDP M i , we ran API, API(0.1), CPI+, CPI(0.1), NSPI( m ) for m ∈ { 5 , 10 , 30 } and PSDP ∞ 30 times. For each run j and algorithm, we compute for all iterations k ∈ (1 , 100) the performance, i.e. the loss L j,k = µ ( v π ∗ -v π k ) with respect to the optimal policy. Figure 2 displays statistics about these random variables. For each algorithm, we display a learning curve with confidence regions that account for the variability across runs and problems. The supplementary material contains statistics that are respectively conditioned on the values of n S , n A and b , which gives some insight on the influence of these parameters.

From these experiments and statistics, we can make a series of observations. The standard API scheme is much more variable than the other algorithms and tends to provide the worst performance on average. CPI+ and CPI( α ) display about the same asymptotic performance on average. If CPI( α ) has slightly less variability, it is much slower than CPI+, that always converges in very few iterations (most of the time less than 10, and always less than 20). API( α )-the naive conservative variation of API that is also simpler than CPI( α )-is empirically close to CPI( α ), while being on average slightly worse. CPI+, CPI( α ) and PSDP ∞ have a similar average performance, but the variability of PSDP ∞ is significantly smaller. PSDP ∞ is the algorithm that overall gives the best results. NSPI( m ) does indeed provide a bridge between API and PSDP ∞ . By increasing m , the behavior gets closer to that of PSDP ∞ . With m = 30 , NSPI( m ) is overall better than API( α ), CPI+, and CPI( α ), and close to PSDP ∞ . The above relative observations are stable with respect to the number of states n S and actions n A . Interestingly, the differences between the algorithms tend to vanish when the dynamics of the problem gets more and more stochastic (when the branching factor increases). This complies with our analysis based on concentrability coefficients: there are all finite when the dynamics mixes a lot, and their relative difference are the biggest in deterministic instances.

Figure 2. Statistics for all instances. The MDPs ( M i ) 1 ≤ i ≤ 30 are i.i.d. with the same distribution as M 1 . Conditioned on some MDP M i and some algorithm, the error measures at all iteration k are i.i.d. with the same distribution as L 1 ,k . The central line of the learning curves gives the empirical estimate of the overall average error ( E [ L 1 ,k ]) k . The three grey regions (from dark to light grey) are estimates of respectively the variability (across MDPs) of the average error ( Std [ E [ L 1 ,k | M 1 ]]) k , the average (across MDPs) of the standard deviation of the error ( E [ Std [ L 1 ,k | M 1 ]]) k , and the variability (across MDPs) of the standard deviation of the error ( Std [ Std [ L 1 ,k | M 1 ]]) k . For ease of comparison, all curves are displayed with the same x and y range.

<!-- image -->

## 5. Discussion, Summary and Future Work

We have considered several variations of the Policy Iteration schemes for infinite-horizon problems: API, CPI, NSPI( m ), API( α ) and PSDP ∞ 7 . We have in particular explained the fact-to our knowledge so far unknownthat the recently introduced NSPI( m ) algorithm generalizes API (that is obtained when m =1) and PSDP ∞ (that is very similar when m = ∞ ). Figure 1 synthesized the theoretical guarantees about these algorithms. Most of the bounds are to our knowledge new.

One of the first important message of our work is that what is usually hidden in the constants of the performance bounds does matter. The constants involved in the bounds for API, CPI, PSDP ∞ and for the main (left) terms of NSPI( m ) can be sorted from the worst to the best as follows: C (2 , 1 , 0) , C (1 , 0) , C (1) π ∗ , C π ∗ . A detailed hierarchy of all constants was depicted in Figure 1. This is to our knowledge the first time that such an in-depth comparison of the bounds is done, and our hierarchy of constants has interesting implications that go beyond the Policy Iteration schemes we have been focusing on in this paper. As a matter of fact, several other dynamic programming algorithms, namely AVI (Munos, 2007), λ PI (Scherrer, 2013), AMPI (Scherrer et al., 2012), come with guarantees involv-

7 We recall that to our knowledge, the use of PSDP ∞ (PSDP in an infinite-horizon context) is not documented in the literature.

ing the worst constant C (2 , 1 , 0) , which suggests that they should not be competitive with the best algorithms we have described here.

At the purely technical level, several of our bounds come in pair; this is due to the fact that we have introduced a new proof technique. This led to a new bound for API, that improves the state of the art in the sense that it involves the constant C (1 , 0) instead of C (2 , 1 , 0) . It also enabled us to derive new bounds for CPI (and its natural algorithmic variant CPI( α )) that is worse in terms of guarantee but has a better time complexity ( ˜ O ( 1 glyph[epsilon1] ) instead of O ( 1 glyph[epsilon1] 2 ) ). We believe this new technique may be helpful in the future for the analysis of other MDP algorithms.

Let us sum up the main insights of our analysis. 1) The guarantee for CPI can be arbitrarily stronger than that of API/API( α ), because it is expressed with respect to the best concentrability constant C π ∗ , but this comes at the cost of a relative-exponential in 1 glyph[epsilon1] -increase of the number of iterations. 2) PSDP ∞ enjoys the best of both worlds: its performance guarantee is similar to that of CPI, but within a number of iterations similar to that of API. 3) Contrary to API that requires a constant memory, the memory needed by CPI and PSDP ∞ is proportional to their number of iterations, which may be problematic in particular when the discount factor γ is close to 1 or the approximation error glyph[epsilon1] is close to 0 ; we showed that the NSPI( m ) algorithm allows to make an overall trade-off between memory and perfor-

mance.

The main assumption of this work is that all algorithms have at disposal an glyph[epsilon1] -approximate greedy operator. It may be unreasonable to compare all algorithms on this basis, since the underlying optimization problems may have different complexities: for instance, methods like CPI look in a space of stochastic policies while API moves in a space of deterministic policies. Digging and understanding in more depth what is potentially hidden in the term glyph[epsilon1] -as we have done here for the concentrability constants-constitutes a very natural research direction.

Last but not least, we have run numerical experiments that support our worst-case analysis. On simulations on about 800 Garnet MDPs with various characteristics, CPI( α ), CPI+ (CPI with a crude line-search mechanism), PSDP ∞ and NSPI( m ) were shown to always perform significantly better than the standard API. CPI+, CPI( α ) and PSDP ∞ performed similarly on average, but PSDP ∞ showed much less variability and is thus the best algorithm in terms of overall performance. Finally, NSPI( m ) allows to make a bridge between API and PSDP ∞ , reaching an overall performance close to that of PSDP ∞ with a controlled memory. Implementing other instances of these algorithmic schemes, running and analyzing experiments on bigger domains constitutes interesting future work.

## A. Proofs for Table 1

PSDP ∞ : For all k , we have

<!-- formula-not-decoded -->

where we defined e k = max π ′ T π ′ v σ k -1 -T π k v σ k -1 . As P π ∗ is non negative, we deduce by induction:

<!-- formula-not-decoded -->

By multiplying both sides by µ , using the definition of the coefficients c π ∗ ( i ) and the fact that νe j ≤ glyph[epsilon1] j ≤ glyph[epsilon1] , we get:

<!-- formula-not-decoded -->

The bound with respect to C (1) π ∗ is obtained by using the fact that v σ k ... ≥ v σ k -γ k V max and taking k ≥ ⌈ log 2 V max glyph[epsilon1] 1 -γ ⌉ .

Starting back in Equation (7) and using the definition of C π ∗ (in particular the fact that for all i , µ ( γP π ∗ ) i ≤ 1 1 -γ d π ∗ ,µ ≤ C π ∗ 1 -γ ν ) and the fact that νe j ≤ glyph[epsilon1] j , we get:

<!-- formula-not-decoded -->

and the other bound is obtained by using the fact that v σ k ... ≥ v σ k -γ k V max , ∑ k i =1 glyph[epsilon1] i ≤ kglyph[epsilon1] , and considering the number of iterations k = ⌈ log 2 V max glyph[epsilon1] 1 -γ ⌉ .

API/NSPI( m ): API is identical to NSPI(1), and its bounds are particular cases of the first two bounds for NSPI( m ), so we only consider NSPI( m ). By following the proof technique of Scherrer &amp; Lesner (2012), writing Γ k,m = ( γP π k )( γP π k -1 ) · · · ( γP π k -m +1 ) and e k +1 = max π ′ T π ′ v π k,m -T π k +1 v π k,m , one can show that:

<!-- formula-not-decoded -->

Multiplying both sides by µ (and observing that e k ≥ 0 ) and the fact that νe j ≤ glyph[epsilon1] j ≤ glyph[epsilon1] , we obtain:

<!-- formula-not-decoded -->

which leads to the first bound by taking k ≥ ⌈ log 2 V max glyph[epsilon1] 1 -γ ⌉ . Starting back on Equation (9), assuming for simplicity that glyph[epsilon1] -k = 0 for all k ≥ 0 , we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which leads to the second bound by taking k = ⌈ log 2 V max glyph[epsilon1] 1 -γ ⌉ . Last but not least, starting back on Equation (8), and using the fact that ( I -Γ k -i,m ) -1 = I + Γ k -i,m ( I -Γ k -i,m ) -1 we see that:

<!-- formula-not-decoded -->

The first term of the r.h.s. can be bounded exactly as for PSDP ∞ . For the second term, we have:

<!-- formula-not-decoded -->

and we follow the same lines as above (from Equation (9) to Equations (10) and (11)) to conclude.

CPI, CPI( α ), API( α ): Conservative steps are addressed by a tedious generalization of the proof for API by Munos (2003). Due to lack of space, the proof is deferred to the Supplementary Material.

## B. Proofs for Figure 1

We here provide details on the order relation for the concentrability coefficients.

<!-- formula-not-decoded -->

and C π ∗ is the smallest coefficient C satisfying d π ∗ ,µ ≤ Cν . (ii) We may have C π ∗ &lt; ∞ and C (1) π ∗ = ∞ by design- ing a MDP on N where π ∗ induces a deterministic transition from state i to state i +1 .

C (1) π ∗ → C (1 , 0) : (i) We have C (1) π ∗ ≤ C (1 , 0) because for all i , c π ∗ ( i ) ≤ c ( i ) . (ii) It is easy to obtain C (1) π ∗ &lt; ∞ and C (1 , 0) = ∞ since C (1) π ∗ only depends on one policy while C (1) π ∗ depends on all policies.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(ii) One may have C (1 ,m ) &lt; ∞ and C (2 ,m,m ) = ∞ when c ( i ) = Θ( 1 i 2 γ i ) , since the generic term of C (1 ,m ) is Θ( 1 i 2 ) (the sum converges) while that of C (2 ,m,m ) is Θ( 1 i ) (the sum diverges). The reasoning is similar for the other relation.

<!-- formula-not-decoded -->

C (2 , 1 , 0) ↔ C (2 ,m, 0) : (i) We clearly have C (2 ,m, 0) ≤ 1 -γ m 1 -γ C (2 , 1 , 0) . (ii) C (2 ,m, 0) can be rewritten as follows:

<!-- formula-not-decoded -->

Then, using the fact that 1 + ⌊ i m ⌋ ≥ max ( 1 , i m ) , we have

<!-- formula-not-decoded -->

Thus, when m is finite, C (2 ,m, 0) &lt; ∞⇒ C (2 , 1 , 0) &lt; ∞ .

## References

- Archibald, T., McKinnon, K., and Thomas, L. On the Generation of Markov Decision Processes. Journal of the Operational Research Society , 46:354-361, 1995.
- Bagnell, J.A., Kakade, S.M., Ng, A., and Schneider, J. Policy search by dynamic programming. In NIPS , 2003.
- Bertsekas, D.P. and Tsitsiklis, J.N. Neuro-Dynamic Programming . Athena Scientific, 1996.
- Farahmand, A.M., Munos, R., and Szepesv´ ari, Cs. Error propagation for approximate policy and value iteration (extended version). In NIPS , 2010.
- Ghavamzadeh, M. and Lazaric, A. Conservative and Greedy Approaches to Classification-based Policy Iteration. In AAAI , 2012.
- Kakade, Sham and Langford, John. Approximately optimal approximate reinforcement learning. In ICML , 2002.
- Lagoudakis, M. and Parr, R. Reinforcement Learning as Classification: Leveraging Modern Classifiers. In ICML , 2003a.
- Lagoudakis, M.G. and Parr, R. Least-squares policy iteration. Journal of Machine Learning Research (JMLR) , 4: 1107-1149, 2003b.
- Lazaric, A., Ghavamzadeh, M., and Munos, R. Analysis of a Classification-based Policy Iteration Algorithm. In ICML , 2010.
- Munos, R. Error Bounds for Approximate Policy Iteration. In ICML , 2003.
- Munos, R. Performance Bounds in Lp norm for Approximate Value Iteration. SIAM J. Control and Optimization , 2007.
- Puterman, M. Markov Decision Processes . Wiley, New York, 1994.
- Scherrer, B. Performance Bounds for Lambda Policy Iteration and Application to the Game of Tetris. Journal of Machine Learning Research , 14:1175-1221, 2013.
- Scherrer, B. and Lesner, B. On the Use of Non-Stationary Policies for Stationary Infinite-Horizon Markov Decision Processes. In NIPS , 2012.
- Scherrer, Bruno, Ghavamzadeh, Mohammad, Gabillon, Victor, and Geist, Matthieu. Approximate Modified Policy Iteration. In ICML , 2012.

## C. Proof for CPI, CPI( α ), API( α )

We begin by proving the following result:

Theorem 1. At each iteration k &lt; k ∗ of CPI (Equation (3) ), the expected loss satisfies:

<!-- formula-not-decoded -->

Proof. Using the facts that T π k +1 v π k = (1 -α k +1 ) v π k + α k +1 T π k +1 v π k and the notation e k +1 = max π ′ T π ′ v π k -T π ′ k +1 v π k , we have:

<!-- formula-not-decoded -->

Using the fact that v π k +1 = ( I -γP π k +1 ) -1 r , and the fact that ( I -γP π k +1 ) -1 is non-negative, we can see that

<!-- formula-not-decoded -->

Putting this back in Equation (12), we obtain:

<!-- formula-not-decoded -->

Define the matrix Q k = [(1 -α k ) I + α k γP π ∗ ] , the set N i,k = { j ; k -i + 1 ≤ j ≤ k } (this set contains exactly i elements), the matrix R i,k = ∏ j ∈N i,k Q j , and the coefficients β k = 1 -α k (1 -γ ) and δ k = ∏ k i =1 β k . By repeatedly using the fact that the matrices Q k are non-negative, we get by induction

<!-- formula-not-decoded -->

Let P j ( N i,k ) be the set of subsets of N i,k of size j . With this notation we have

<!-- formula-not-decoded -->

where for all subset I of N i,k , we wrote

<!-- formula-not-decoded -->

Therefore, by multiplying Equation (13) by µ , using the definition of the coefficients c ( i ) , and the facts that ν ≤ (1 -

## Supplementary Material

γ ) d ν,π k +1 , we obtain:

<!-- formula-not-decoded -->

Now, using the fact that for x ∈ (0 , 1) , log(1 -x ) ≤ -x , we can observe that

<!-- formula-not-decoded -->

As a consequence, we get δ k ≤ e -(1 -γ ) ∑ k i =1 α i .

In the analysis of CPI, Kakade &amp; Langford (2002) show that the learning steps that ensure the nice performance guarantee of CPI satisfy α k ≥ (1 -γ ) glyph[epsilon1] 12 γV max , the right term e { (1 -γ ) ∑ k i =1 α i } above tends 0 exponentially fast, and we get the following corollary that shows that CPI has a performance bound with the coefficient C (1 , 0) of API in a number of iterations O ( log 1 glyph[epsilon1] glyph[epsilon1] ) .

Corollary 1. The smallest (random) iteration k † such that log V max glyph[epsilon1] 1 -γ ≤ ∑ k † i =1 α i ≤ log V max glyph[epsilon1] 1 -γ + 1 is such that k † ≤ 12 γV max log V max glyph[epsilon1] glyph[epsilon1] (1 -γ ) 2 and the policy π k † satisfies:

<!-- formula-not-decoded -->

Since the proof is based on a generalization of the analysis of API and thus does not use any of the specific properties of CPI, it turns out that the results we have just given can straightforwardly be specialized to CPI( α ).

Corollary 2. Assume we run CPI( α ) for some α ∈ (0 , 1) , that is CPI (Equation (3) ) with α k = α for all k .

<!-- formula-not-decoded -->

The above bound for CPI( α ) involves the factor 1 (1 -γ ) 3 . Aprecise examination of the proof shows that this amplification is due to the fact that the approximate greedy operator uses the distribution d π k ,ν ≥ (1 -γ ) ν instead of ν (for API). In fact, using a very similar proof, it is easy to show that API( α ) satisfies the following result.

Corollary 3. Assume API( α ) is run for some α ∈ (0 , 1) .

<!-- formula-not-decoded -->

## D. More details on the Numerical Simulations

Domain and Approximations In our experiments, a Garnet is parameterized by 4 parameters and is written G ( n S , n A , b, p ) : n S is the number of states, n A is the number of actions, b is a branching factor specifying how many possible next states are possible for each state-action pair ( b states are chosen uniformly at random and transition probabilities are set by sampling uniform random b -1 cut points between 0 and 1) and p is the number of features (for linear function approximation). The reward is state-dependent: for a given randomly generated Garnet problem, the reward for each state is uniformly sampled between 0 and 1. Features are chosen randomly: Φ is a n S × p feature matrix of which each component is randomly and uniformly sampled between 0 and 1. The discount factor γ is set to 0 . 99 in all experiments.

All the algorithms we have discussed in the paper need to repeatedly compute G glyph[epsilon1] ( ρ, v ) for some distribution ρ = ν or ρ = d π,ν . In other words, they must be able to make calls to an approximate greedy operator applied to the value v of some policy for some distribution ρ . To implement this operator, we compute a noisy estimate of the value v with a uniform white noise u ( ι ) of amplitude ι , then projects this estimate onto the space spanned by Φ with respect to the ρ -quadratic norm (projection that we write Π Φ ,ρ ), and then applies the (exact) greedy operator on this projected estimate. In a nutshell, one call to the approximate greedy operator G glyph[epsilon1] ( ρ, v ) amounts to compute G Π Φ ,ρ ( v + u ( ι )) .

Simulations We have run series of experiments, in which we callibrated the perturbations (noise, approximations) so that the algorithm are significantly perturbed but no too much (we do not want their behavior to become too erratic). After trial and error, we ended up considering the following setting. We used Garnet problems G ( n S , n A , b, p ) with the number of states n S ∈ { 50 , 100 , 200 } , the number of actions n A ∈ { 2 , 5 , 10 } , the branching factor b ∈ { 1 , 2 , 10 }} ( b = 1 corresponds to deterministic problems), the number of features to approximate the value p = n S 10 , and the noise level ι = 0 . 1 ( 10% ).

In addition to Figure 2 that shows the statistics overall for the all the parameter instances, Figure 3, 4 and 5 display statistics that are respectively conditioned on the values of n S , n A and b , which gives some insight on the influence of these parameters.

Figure 3. Statistics conditioned on the number of states. Top: n S = 50 . Middle: n S = 100 . Bottom n S = 200 .

<!-- image -->

Figure 4. Statistics conditioned on the number of actions. Top: n A = 2 . Middle: n A = 5 . Bottom n a = 10 .

<!-- image -->

Figure 5. Statistics conditioned on the branching factor. Top: b = 1 (deterministic). Middle: b = 2 . Bottom b = 10 .

<!-- image -->