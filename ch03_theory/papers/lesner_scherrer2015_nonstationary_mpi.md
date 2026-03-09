enters.

public

Autorisation

HAL

T

HAL

open science

<!-- image -->

## Non-stationary approximate modified policy iteration

Boris Lesner, Bruno Scherrer

## To cite this version:

Boris Lesner, Bruno Scherrer. Non-stationary approximate modified policy iteration. ICML 2015, Jul 2015, Lille, France. ￿hal-01186664￿

## HAL Id: hal-01186664 https://inria.hal.science/hal-01186664v1

Submitted on 25 Aug 2015

HAL is a multi-disciplinary open access archive for the deposit and dissemination of scientific research documents, whether they are published or not. The documents may come from teaching and research institutions in France or abroad, or from public or private research centers.

L'archive ouverte pluridisciplinaire HAL , est destinée au dépôt et à la diffusion de documents scientifiques de niveau recherche, publiés ou non, émanant des établissements d'enseignement et de recherche français ou étrangers, des laboratoires publics ou privés.

<!-- image -->

Boris Lesner Bruno Scherrer

## Non-Stationary Approximate Modified Policy Iteration

BORIS.LESNER.DATEXIM@GMAIL.COM

BRUNO.SCHERRER@INRIA.FR

Inria, Villers-l` es-Nancy, F-54600, France Universit´ e de Lorraine, LORIA, UMR 7503, Vandœuvre-l` es-Nancy, F-54506, France

## Abstract

We consider the infinite-horizon γ -discounted optimal control problem formalized by Markov Decision Processes. Running any instance of Modified Policy Iteration-a family of algorithms that can interpolate between Value and Policy Iteration-with an error glyph[epsilon1] at each iteration is known to lead to stationary policies that are at least 2 γglyph[epsilon1] (1 -γ ) 2 -optimal. Variations of Value and Policy Iteration, that build glyph[lscript] -periodic nonstationary policies, have recently been shown to display a better 2 γglyph[epsilon1] (1 -γ )(1 -γ glyph[lscript] ) -optimality guarantee. We describe a new algorithmic scheme, Non-Stationary Modified Policy Iteration, a family of algorithms parameterized by two integers m ≥ 0 and glyph[lscript] ≥ 1 that generalizes all the above mentionned algorithms. While m allows one to interpolate between Value-Iteration-style and Policy-Iteration-style updates, glyph[lscript] specifies the period of the non-stationary policy that is output. We show that this new family of algorithms also enjoys the improved 2 γglyph[epsilon1] (1 -γ )(1 -γ glyph[lscript] ) -optimality guarantee. Perhaps more importantly, we show, by exhibiting an original problem instance, that this guarantee is tight for all m and glyph[lscript] ; this tightness was to our knowledge only known in two specific cases, Value Iteration ( m = 0 , glyph[lscript] = 1) and Policy Iteration ( m = ∞ , glyph[lscript] = 1) .

## 1. Introduction

Dynamic Programming (DP) is an elegant approach for addressing γ -discounted infinite-horizon optimal control problems formalized as Markov Decision Processes (MDP) (Puterman, 1994). The two most well-known DP algorithms in this framework are Value Iteration (VI) and

Proceedings of the 32 nd International Conference on Machine Learning , Lille, France, 2015. JMLR: W&amp;CP volume 37. Copyright 2015 by the author(s).

Policy Iteration (PI). While the former has typically lighter iterations, the latter usually converges much faster. Modified Policy Iteration (MPI), that interporlates between the two, was introduced to improve the convergence rate of VI while remaining lighter than PI (Puterman &amp; Shin, 1978).

When the optimal control problem one considers is large, an option is to consider approximate versions of these DP algorithms, where each iteration may be corrupted with some noise glyph[epsilon1] . An important question is the sensitivity of such an approach to the noise. Bertsekas &amp; Tsitsiklis (1996) gather several results regarding approximate versions of VI and PI (thereater named AVI and API). It is known that the policy output by such procedures is guaranteed to be 2 γglyph[epsilon1] (1 -γ ) 2 -optimal. In particular, when the perturbation glyph[epsilon1] tends to 0 , one recovers an optimal solution. This analysis was recently generalized to an approximate implementation of MPI (AMPI) independently by Canbolat &amp; Rothblum (2012) and Scherrer et al. (2012). The better guarantee, obtained by the latter2 γglyph[epsilon1] (1 -γ ) 2 -optimalityexactly matches that of AVI and API. The algorithmic scheme AMPI can be implemented in various ways, reducing the original control problem to a series of (more standard) regression and classification problems (Scherrer et al., 2012), and lead to state-of-the-art results on large benchmark problems, in particular on the Tetris domain (Gabillon et al., 2013).

An apparent weakness of these sensitivity analyses is that the dependence with respect to the discount factor γ is bad: since γ is typically close to 1 , the denominator of the constant 2 γ (1 -γ ) 2 often makes the guarantee uninformative in practice. Unfortunately, it turns out that it is not so much a weakness of the analyses but a weakness of the very algorithmic approach since Bertsekas &amp; Tsitsiklis (1996) and Scherrer &amp; Lesner (2012) showed that the bound 2 γglyph[epsilon1] (1 -γ ) 2 is tight respectively for API and AVI and thus cannot be improved in general. Interestingly, the authors of the latter article described a trick for modifying AVI and API so as to improve the guarantee: even though one knows that there exists a stationary policy that is optimal, Scherrer &amp; Lesner (2012) showed that variations of A VI and API

that compute glyph[lscript] -periodic non-stationary policies (thereafter named NS-AVI and NS-API) lead to an improved bound of 2 γglyph[epsilon1] (1 -γ )(1 -γ glyph[lscript] ) . For values of glyph[lscript] of the order of 1 log 1 γ -that is equivalent to 1 1 -γ when γ is close to 1-the guarantee is improved by a significant factor (of order 1 1 -γ ). With respect to the standard A VI and API schemes, the only extra algorithmic price to pay is memory that is then O ( glyph[lscript] ) instead of O (1) . As often in computer science, one gets a clear trade-off between quality and memory.

To the best of our knowledge, it is not known whether the non-stationary trick also applies to a modified algorithm that would interpolate between NS-AVI and NS-API. Perhaps more importantly, it is not known whether the improved bound 2 γglyph[epsilon1] (1 -γ )(1 -γ glyph[lscript] ) is tight for NS-A VI or NS-API, and even whether the standard 2 γglyph[epsilon1] (1 -γ ) 2 bound is tight for AMPI. In this article, we fill the missing parts of this topic in the literature. We shall describe NS-AMPI, a new nonstationary MPI algorithm that generalizes all previously mentioned algorithms-AVI, API, AMPI, NS-AVI and NSAPI-and prove that it returns a policy that is 2 γglyph[epsilon1] (1 -γ )(1 -γ glyph[lscript] ) -optimal. Furthermore, we will show that for any value of the period glyph[lscript] and any degree of interpolation between NSAVI and NS-API, such a bound is tight. Thus, our analysis not only unifies all previous works, but it provides a complete picture of the sensitivity analysis for this large class of algorithms.

The paper is organized as follows. In Section 2 we describe the optimal control problem. Section 3 describes the state-of-the-art algorithms AMPI, NS-AVI and NS-API along with their known sensitivity analysis. In Section 4, we describe the new algorithm, NS-AMPI, and our main results: a performance guarantee (Theorem 3) and a matching lower bound (Theorem 4). Section 5 follows by providing the proof skteches of both results. Section 6 describes a small numerical illustration of our new algorithm, which gives some insight on the choice of its parameters. Section 7 concludes and mentions potential future research directions.

## 2. Problem Setting

We consider a discrete-time dynamic system whose state transition depends on a control. Let X be a state space. When at some state, an action is chosen from a finite action space A . The current state x ∈ X and action a ∈ A characterize through a homogeneous probability kernel P ( dx | x, a ) the distribution of the next state x ′ . At each transition, the system is given a reward r ( x, a, x ′ ) ∈ R where r : X × A × X → R is the instantaneous reward function. In this context, the goal is to determine a sequence of actions ( a t ) adapted to the past of the process until time t that maximizes the expected discounted sum of rewards from any starting state x :

<!-- formula-not-decoded -->

where 0 &lt; γ &lt; 1 is a discount factor. The tuple 〈 X,A,P,r,γ 〉 is called a Markov Decision Process (MDP) and the associated optimization problem infinite-horizon stationary discounted optimal control (Puterman, 1994; Bertsekas &amp; Tsitsiklis, 1996) .

An important result of this setting is that there exists at least one stationary deterministic policy, that is a function π : X → A that maps states into actions, that is optimal (Puterman, 1994). As a consequence, the problem is usually recast as looking for the stationary deterministic policy π that maximizes for every state x the quantity

<!-- formula-not-decoded -->

called the value of policy π at state x . The notation E π means that we condition on trajectories such that x t +1 ∼ P π ( ·| x t ) , where P π ( dx | x ) is the stochastic kernel P ( dx | x, π ( x )) that chooses actions according to policy π . We shall similarly write r π : X → R for the function giving the immediate reward while following policy π :

<!-- formula-not-decoded -->

Two linear operators are associated with the stochastic kernel P π : a left operator on functions f ∈ R X

<!-- formula-not-decoded -->

and a right operator on distributions µ

<!-- formula-not-decoded -->

In words, ( P π f )( x ) is the expected value of f after following policy π for a single time-step starting from x , and µP π is the distribution of states after a single time-step starting from µ .

Given a policy π , it is well known that the value v π is the unique solution of the following Bellman equation:

<!-- formula-not-decoded -->

In other words, v π is the fixed point of the affine operator T π v := r π + γP π v . The optimal value starting from state x is defined as

<!-- formula-not-decoded -->

It is also well known that v ∗ is characterized by the following Bellman equation:

<!-- formula-not-decoded -->

where the max operator is componentwise. In other words, v ∗ is the fixed point of the nonlinear operator Tv := max π T π v . Finally, for any function v ∈ R X , we say that a policy π is greedy with respect to v if it satisfies:

<!-- formula-not-decoded -->

or equivalently T π v = Tv . We write, with some abuse of notation 1 G ( v ) any policy that is greedy with respect to v . The notions of optimal value function and greedy policies are fundamental to optimal control because of the following standard property: any policy π ∗ that is greedy with respect to the optimal value is an optimal policy and its value v π ∗ is equal to v ∗ . Thus, the main problem amounts to computing the optimal value function v ∗ . The next section descibes algorithmic approaches from the literature.

## 3. State-of-the-Art Algorithms

We begin by describing the Approximate Modified Policy Iteration (AMPI) algorithmic scheme (Scherrer et al., 2012). Starting from an arbitrary value function v 0 , AMPI generates a sequence of value-policy pairs

<!-- formula-not-decoded -->

where m ≥ 0 is a free parameter. At each iteration k , the term glyph[epsilon1] k accounts for a possible approximation in the evaluation step. AMPI generalizes the well-known approximate DP algorithms Value Iteration (AVI) and Policy Iteration (API) for values m = 0 and m = ∞ , respectively. In the exact case ( glyph[epsilon1] k = 0 ), MPI requires less computation per iteration than PI (in a way similar to VI) and enjoys the faster convergence (in terms of number of iterations) of PI (Puterman &amp; Shin, 1978; Puterman, 1994).

It was recently shown that controlling the errors glyph[epsilon1] k when running AMPI is sufficient to ensure some performance guarantee (Scherrer et al., 2012; Canbolat &amp; Rothblum, 2012). For instance, we have the following performance bound, that is remarkably independent of the parameter m . 2

1 There might be several policies that are greedy with respect to v .

2 Note that in practice, the term glyph[epsilon1] k will generally depend on m . The exact dependence may strongly depend on the precise implementation and we refer the reader to (Scherrer et al., 2012) for examples of such analyses. In this paper, we only consider the situation of a uniform error bound on the errors, all the more that extensions to more complicated errors is straightforward.

Theorem 1 (Scherrer et al. (2012, Remark 2)) . Consider AMPI with any parameter m ≥ 0 . Assume there exists an glyph[epsilon1] &gt; 0 such that the errors satisfy ‖ glyph[epsilon1] k ‖ ∞ &lt; glyph[epsilon1] for all k . Then, the loss due to running policy π k instead of the optimal policy π ∗ satisfies

<!-- formula-not-decoded -->

In the specific case corresponding to A VI ( m = 0 ) and API ( m = ∞ ), this bound matches performance guarantees that have been known for a long time (Singh &amp; Yee, 1994; Bertsekas &amp; Tsitsiklis, 1996). The asymptotic constant 2 γ (1 -γ ) 2 can be very big, in particular when γ is close to 1 . Unfortunately, it cannot be improved: Bertsekas &amp; Tsitsiklis (1996, Example 6.4) showed that the bound is tight for PI, Scherrer &amp;Lesner (2012) proved that it is tight for VI, 3 and we will prove in this article 4 the-to our knowledge unknownfact that it is also tight for AMPI. In other words, improving the performance bound requires to change the algorithms.

Even though the theory of optimal control states that there exists a stationary policy that is optimal, Scherrer &amp; Lesner (2012) recently showed that the performance bound of Theorem 1 could be improved in the specific cases m = 0 and m = ∞ by considering variations of AVI and API that build periodic non-stationary policies (instead of stationary policies). Surprisingly, the Non-Stationary AVI (NS-AVI) algorithm proposed there works almost exactly like AVI: it builds the exact same sequence of value-policy pairs from any initialization v 0 (compare with AMPI with m = 0 ):

<!-- formula-not-decoded -->

The only difference is in what is output: while A VI would return the last policy, say π k after k iterations, NS-AVI returns the periodic non-stationary policy π k,glyph[lscript] that loops in reverse order on the last glyph[lscript] generated policies:

<!-- formula-not-decoded -->

Following the policy π k,glyph[lscript] means that the first action is selected by π k , the second one by π k -1 , until the glyph[lscript] th one by π k -glyph[lscript] +1 , then the policy loops and the next actions are selected by π k , π k -1 , so on and so forth. Note that when glyph[lscript] = 1 , we recover the output of A VI: the last policy π k that is used for all actions.

3 Though the MDP instance used to show the tightness of the bound for VI is the same as that for PI (Bertsekas &amp; Tsitsiklis, 1996, Example 6.4), Scherrer &amp; Lesner (2012) seem to be the first to argue about it in the literature.

4 Theorem 4 page 4 with glyph[lscript] = 1 .

To describe the other algorithm proposed by Scherrer &amp; Lesner (2012), Non-Stationary API (NS-API), we shall introduce the linear Bellman operator T π k,glyph[lscript] associated with π k,glyph[lscript] :

<!-- formula-not-decoded -->

It is indeed straightforward to show that the value v π k,glyph[lscript] obtained by following π k,glyph[lscript] is the unique fixed point of T π k,glyph[lscript] . Then, from any initial set of glyph[lscript] policies ( π 0 , π -1 , . . . , π -glyph[lscript] +1 ) , NS-API generates the following sequence of value-policy pairs:

<!-- formula-not-decoded -->

While computing the value v k requires (approximately) solving the fixed point equation v π k,glyph[lscript] = T π k,glyph[lscript] v π k,glyph[lscript] of the non-stationary policy π k,glyph[lscript] made of the last glyph[lscript] computed policies, the new policy π k +1 that is computed in the greedy step is (as usual) a simple stationary policy. After k iterations, similarly to NS-A VI, the algorithm returns the periodic non-stationary policy π k,glyph[lscript] . Here again, setting glyph[lscript] = 1 provides the standard API algorithm.

On the one hand, using these non-stationary variants may require more memory since one must store glyph[lscript] policies instead of one. On the other hand, the following result shows that this extra memory allows us to improve the performance guarantee.

Theorem 2 (Scherrer &amp; Lesner (2012, Theorems 2 and 4)) . Consider NS-AVI or NS-API with any parameter l ≥ 0 . Assume there exists an glyph[epsilon1] &gt; 0 such that the errors satisfy ‖ glyph[epsilon1] k ‖ ∞ &lt; glyph[epsilon1] for all k . Then, the loss due to running the non-stationary policy π k,glyph[lscript] instead of the optimal policy π ∗ satisfies

<!-- formula-not-decoded -->

where g 0 = 2 1 -γ glyph[lscript] ‖ v ∗ -v 0 ‖ ∞ for NS-AVI or g 0 = ∥ ∥ v ∗ -v π 0 ,glyph[lscript] ∥ ∥ ∞ for NS-API.

For any glyph[lscript] ≥ 1 , it is a factor 1 -γ 1 -γ glyph[lscript] better than in Theorem 1. Using glyph[lscript] = ⌈ 1 1 -γ ⌉ yields 5 an asymptotic performance bound of 3 . 164 γ 1 -γ glyph[epsilon1] . which constitutes an improvement of order O ( 1 1 -γ ) , which is significant in typical situations where γ is close to 1.

5 Using the facts that 1 -γ ≤ -log γ and log γ ≤ 0 , we have log γ glyph[lscript] ≤ log γ 1 1 -γ ≤ 1 -log γ log γ = -1 hence γ glyph[lscript] ≤ e -1 . Therefore 2 1 -γ glyph[lscript] ≤ 2 1 -e -1 &lt; 3 . 164 .

## 4. Main results

We are now ready to present the first contribution of this paper. We shall introduce a new algorithm, Non-Stationary AMPI (NS-AMPI), that generalizes NS-AVI and NS-API (in the same way the standard AMPI algorithm generalizes standard AVI and API) and AMPI (in the same way NS-VI and NS-PI respectively generalize AVI and API). Given some free parameters m ≥ 0 and glyph[lscript] ≥ 1 , an arbitrary value function v 0 and an arbitrary set of glyph[lscript] -1 policies π 0 , π -1 , π -glyph[lscript] +2 , consider the algorithm that builds a sequence of value-policy pairs as follows:

<!-- formula-not-decoded -->

While the greedy step is identical to that of all algorithms, the evaluation step involves the non-stationary Bellman operator T π k +1 ,glyph[lscript] (composed with itself m times) that we introduced in the previous section, composed with the standard Bellman operator T π k +1 . As in NS-AVI and NS-API, after k iterations, the output of the algorithm is the periodic nonstationary policy π k,glyph[lscript] . For the values m = 0 and m = ∞ , it is easy to see that one respectively recovers NS-A VI and NS-API. When glyph[lscript] = 1 , one recovers AMPI (that itself generalizes the standard A VI and API algorithms, obtained if we further set respectively m = 0 and m = ∞ ).

At this point, a natural question is whether the previous sensitivity results extend to this more general setting. As the following original result states, the answer is yes.

Theorem 3. Consider NS-AMPI with any parameters m ≥ 0 and glyph[lscript] ≥ 1 . Assume there exists an glyph[epsilon1] &gt; 0 such that the errors satisfy ‖ glyph[epsilon1] k ‖ ∞ &lt; glyph[epsilon1] for all k . Then, the loss due to running policy π k,glyph[lscript] instead of the optimal policy π ∗ satisfies

<!-- formula-not-decoded -->

Theorem 3 asymptotically generalizes both Theorem 1 for glyph[lscript] &gt; 1 (the bounds match when glyph[lscript] = 1 ) and Theorem 2 for m &gt; 0 (the bounds are very close when m = 0 or m = ∞ ). As already observed for AMPI, it is remarkable that this performance bound is independent of m .

The second main result of this article is that the bound of Theorem 3 is tight, in the precise sense formalized by the following theorem.

Theorem 4. For all parameter values m ≥ 0 and glyph[lscript] ≥ 1 , for all discount factor γ , for all glyph[epsilon1] &gt; 0 , there exists an MDP instance, an initial value function v 0 , a set of initial policies π 0 , π -1 , . . . , π -glyph[lscript] +2 and a sequence of error terms ( glyph[epsilon1] k ) k ≥ 1 satisfying ‖ glyph[epsilon1] k ‖ ∞ ≤ glyph[epsilon1] , such that for all iterations k , the bound of Theorem 3 is satisfied with equality.

This theorem generalizes the (separate) tightness results for PI ( m = ∞ , glyph[lscript] = 1 ) (Bertsekas &amp; Tsitsiklis, 1996) and for VI ( m = 0 , glyph[lscript] = 1 ) (Scherrer &amp; Lesner, 2012), which are the only results we are aware of. To our knowledge, this result is new even for the standard AMPI algorithm ( m arbitrary but glyph[lscript] = 1 ), and for the non-trivial instances of NSVI ( m = 0 , glyph[lscript] &gt; 1 ) and NS-PI ( m = ∞ , glyph[lscript] &gt; 1 ) proposed by Scherrer &amp; Lesner (2012).

Since it is well known that there exists an optimal policy that is stationary, our result-as well as those of Scherrer &amp; Lesner (2012)-suggesting to consider non-stationary policies may appear strange. There exists, however, a very simple approximation scheme of discounted infinitehorizon control problems-that has to our knowledge never been documented in the literature-that sheds some light on the deep reason why non-stationary policies may be an interesting option. Given an infinite-horizon problem, consider approximating it by a finite-horizon discounted control problem by 'cutting the horizon' after some sufficiently big instant T (that is assume there is no reward after time T ). Contrary to the original infinite-horizon problem, the resulting finite-horizon problem is non-stationary, and has therefore naturally a non-stationary solution that is built by dynamic programming in reverse order. Moreover, it can be shown (Kakade, 2003, by adapting the proof of Theorem 2.5.1) that solving this finite-horizon with VI with a potential error of glyph[epsilon1] at each iteration, will induce at most a performance error of 2 ∑ T -1 i =0 γ t glyph[epsilon1] = 2(1 -γ T ) 1 -γ glyph[epsilon1] . If we add the error due to truncating the horizon ( γ T max s,a | r ( s,a ) | 1 -γ ), we get an overall error of order O ( 1 1 -γ glyph[epsilon1] ) for a memory T of the order of 6 ˜ O ( 1 1 -γ ) . Though this approximation scheme may require a significant amount of memory (when γ is close to 1 ), it achieves the same O ( 1 1 -γ ) improvement over the performance bound of AVI/API/AMPI as NS-AVI/NS-API/NS-AMPI do. In comparison, the nonstationary algorithms with a fixed period glyph[lscript] can be seen as a more flexible way to make the trade-off between the memory and the quality.

## 5. Proof sketches

We begin by considering Theorem 3. While the performance guarantee was obtained through three independent proofs for NS-VI, NS-PI and AMPI, the more general setting that we consider here involves a totally unified proof, which we describe in the remaining of this section.

We write P k (resp. P ∗ ) for the transition kernel P π k (resp.

<!-- formula-not-decoded -->

P π ∗ ) induced by the stationary policy π k (resp. π ∗ ). We will write T k (resp. T ∗ ) for the associated Bellman operator. Similarly, we will write P k,glyph[lscript] for the transition kernel associated with the non-stationary policy π k,glyph[lscript] and T k,glyph[lscript] for its associated Bellman operator. For k ≥ 0 we define the following quantities: b k = T k +1 v k -T k +1 ,glyph[lscript] T k +1 v k , s k = v k -v π k,glyph[lscript] -glyph[epsilon1] k , d k = v ∗ -v k + glyph[epsilon1] k , and l k = v ∗ -v π k,glyph[lscript] . The last quantity, the loss l k of using policy π k,glyph[lscript] instead of π ∗ is the quantitiy we want to ultimately upper bound.

The core of the proof consists in deriving the following recursive relations.

Lemma 1. The quantities b k , s k and d k satisfy:

<!-- formula-not-decoded -->

Since glyph[epsilon1] is a uniform upper-bound on the pointwise absolute value of the errors | glyph[epsilon1] k | , the first inequality implies that b k ≤ O ( glyph[epsilon1] ) , and as a result, the second and third inequalities gives us d k ≤ O ( glyph[epsilon1] ) and s k ≤ O ( glyph[epsilon1] ) . This means that the loss l k = d k + s k will also satisfy l k ≤ O ( glyph[epsilon1] ) and the result is obtained by taking the norm ‖ · ‖ ∞ . The actual bound given in the theorem requires a careful expansion of these three inequalities where we make precise what we have just hidden in the O -notations. The details of this expansion are tedious and deferred to Appendix B of the Supplementary Material. We thus now concentrate on the proof of these relations.

Proof of Lemma 1. We will repeatedly use the fact that since policy π k +1 is greedy with respect to v k , we have

<!-- formula-not-decoded -->

For a non-stationary policy π k,glyph[lscript] , the induced glyph[lscript] -step transition kernel is P k,glyph[lscript] = P k P k -1 · · · P k -glyph[lscript] +1 . As a consequence, for any function f : S → R , the operator T k,glyph[lscript] may be expressed as: T k,glyph[lscript] f = r k + γP k, 1 r k -1 + γ 2 P k, 2 r k -2 + · · · + γ glyph[lscript] P k,glyph[lscript] f and, for any function g : S → R , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Let us now bound b k . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now turn to the bound of d k :

<!-- formula-not-decoded -->

Finally, we prove the relation for s k :

<!-- formula-not-decoded -->

Figure 1. The deterministic MDP matching the bound of Theorem 3.

<!-- image -->

We now turn to the tightness results given in Theorem 4. The proof considers a generalization of the MDP instance used to prove the tightness of the bound for VI (Scherrer &amp;

Lesner, 2012) and PI (Bertsekas &amp; Tsitsiklis, 1996, Example 6.4). Precisely, this MDP consists of states { 1 , 2 , . . . } , two actions: left ( ← ) and right ( → ); the reward function r and transition kernel P are characterized as follows for any state i ≥ 2 :

<!-- formula-not-decoded -->

and r (1) = 0 and P (1 | 1)1 for state 1 (all the other transitions having zero probability mass). As a shortcut, we will use the notation r i for the non-zero reward r ( i, → ) in state i . Figure 1 depicts the general structure of this MDP. It is easily seen that the optimal policy π ∗ is to take ← in all states i ≥ 2 , as doing otherwise would incur a negative reward. Therefore, the optimal value v ∗ ( i ) is 0 in all states i . The proof of the above theorem considers that we run AMPI with v 0 = v ∗ = 0 , π 0 = π -1 = · · · = π glyph[lscript] +2 = π ∗ , and the following sequence of error terms:

<!-- formula-not-decoded -->

In such a case, one can prove that the sequence of policies π 1 , π 2 , . . . , π k that are generated up to iteration k is such that for all i ≤ k , the policy π i takes ← in all states but i , where it takes → . As a consequence, a nonstationary policy π k,glyph[lscript] built from this sequence takes → in k (as dictated by π k ), which transfers the system into state k + glyph[lscript] -1 incurring a reward of r k . Then the policies π k -1 , π k -2 , . . . , π k -glyph[lscript] +1 are followed, each indicating to take ← with 0 reward. After glyph[lscript] steps, the system is again is state k and, by the periodicity of the policy, must again use the action π k ( k ) = → . The system is thus stuck in a loop, where every glyph[lscript] steps a negative reward of r k is received. Consequently, the value of this policy from state k is:

<!-- formula-not-decoded -->

As a consequence, we get the following lower bound,

<!-- formula-not-decoded -->

which exactly matches the upper bound of Theorem 3 (since v 0 = v ∗ = 0 ). The proof of this result involves computing the values v k ( i ) for all states i , steps k of the algorithm, and values m and glyph[lscript] of the parameters, and proving that the policies π k +1 that are greedy with respect to these values satisfy what we have described above. Due to lack of space, the complete proof is deferred to Appendix B of the Supplementary Material; in Lemma 7 and the associated Figures 4 and 5 there, note the quite complex shape of the value function that is induced by the cyclic nature of the MDP and the NS-AMPI algorithm.

## 6. Empirical Illustration

In this section, we describe an empirical illustration of the new algorithm NS-AMPI. Note that the goal here is not to convince the reader that the new degrees of freedom for approximate dynamic programming may be interesting in difficult real control problems-we leave this important question to future work-but rather to give some insight, on small and artificial well-controlled problems, on the effect of the main parameters m and glyph[lscript] .

The problem we consider, the dynamic location problem from Bertsekas &amp; Yu (2012), involves a repairman moving between n sites according to some transition probabilities. As to allow him do his work, a trailer containing supplies for the repair jobs can be relocated to any of the sites at each decision epoch. The problem consists in finding a relocation policy for the trailer according the repairman's and trailer's positions which maximizes the discounted expectation of a reward function.

Given n sites, the state space has n 2 states comprising the locations of both the repairman and the trailer. There are n actions, each one corresponds to a possible destination of the trailer. Given an action a = 1 , . . . , n , and a state s = ( s r , s t ) , where the repairman and the trailer are at locations s r and s t , respectively, we define the reward as r ( s, a ) = -| s r -s t | - | s t -a | / 2 . At any time-step the repairman moves from its location s r &lt; n with uniform probability to any location s r ≤ s ′ r ≤ n ; when s r = n , he moves to site 1 with probability 0 . 75 or otherwise stays. Since the trailer moves are deterministic, the transition kernel is

<!-- formula-not-decoded -->

and 0 everywhere else.

We evaluated the empirical performance gain of using nonstationary policies by implementing the algorithm using random error vectors glyph[epsilon1] k , with each component being uniformly random between 0 and some user-supplied value glyph[epsilon1] . The adjustable size (with n ) of the state and actions spaces allowed to compute an optimal policy to compare with the approximate ones generated by NS-AMPI for all combinations of parameters glyph[lscript] ∈ { 1 , 2 , 5 , 10 } and m ∈ { 1 , 2 , 5 , 10 , 25 , ∞} . Recall that the cases m = 1 and m = ∞ correspond respectively to the NS-VI and NS-PI, while the case glyph[lscript] = 1 corresponds to AMPI. We used n = 8 locations, γ = 0 . 98 and glyph[epsilon1] = 4 in all experiments.

Figure 2 shows the average value of the error v ∗ -v π k,glyph[lscript] per iteration for the different values of parameters m and glyph[lscript] . For each parameter combination, the results are obtained by averaging over 250 runs. While higher values of glyph[lscript] impacts computational efficiency (by a factor O ( glyph[lscript] ) ) it always re-

Figure 2. Average error of policy π k,glyph[lscript] per iteration k of NSAMPI. Red lines for glyph[lscript] = 1 , yellow for glyph[lscript] = 2 , green for glyph[lscript] = 5 and blue for glyph[lscript] = 10 .

<!-- image -->

sults with better asymptotic performance. Especially with the lower values of m , a higher glyph[lscript] allows for faster convergence. While increasing m , this trend fades to be finally reversed in favor of faster convergence for small glyph[lscript] . However, while small glyph[lscript] converges faster, it is with greater error than with higher glyph[lscript] after convergence. It can be seen that convergence is attained shortly after the glyph[lscript] th iteration which can be explained by the fact that the first policies (involving π 0 , π -1 , . . . , π -glyph[lscript] +2 ), are of poor quality and the algorithm must perform at least glyph[lscript] iterations to 'push them out' of π k,glyph[lscript] .

We conducted a second experiment to study the relative influence of the parameters glyph[lscript] and m . From the observation that, in the very setting we are considering, the time complexity of an iteration of NS-AMPI can be roughly summarized by the number glyph[lscript]m +1 of applications of a stationary policy's Bellman operator, we ran the algorithm for fixed values of the product glyph[lscript]m and measured the asymptotic policy error for varying values of glyph[lscript] after 150 iterations. These results are depicted on Figure 3. This setting gives insight on how to set both parameters for a given 'time budget' glyph[lscript]m . While runs with a lower glyph[lscript] are slightly faster to converge, higher values always give the best policies after a sufficient number of iterations, and greatly reduces the variance across all runs, showing that non-stationarity adds robustness to the approximation noise. Regarding asymptotic quality, it thus appears that the best setting is to favor glyph[lscript] instead of m .

Overall, both experiments confirm our theoretical analysis that the main parameter for asymptotic quality is glyph[lscript] . Regarding the rate of convergence, the first experiments sug-

<!-- image -->

glyph[lscript]

glyph[lscript]

Figure 3. Policy error and standard deviation after 150 iterations for different different values of glyph[lscript] . Each plot represents a fixed value of the product glyph[lscript]m . Data is collected over 250 runs with n = 8 .

gests that too big values of glyph[lscript] may be harmful. In practice, a schedule where glyph[lscript] progressively grows while m decreases may provide the best compromise. Confirming this, as well as studying approximate implementations designed for real problems constitutes a matter for future investigation.

## 7. Conclusion

We have described a new dynamic-programming scheme, NS-AMPI, that extends and unifies several state-of-the-art algorithms of the literature: AVI, API, AMPI, NS-VI, and NS-PI. NS-AMPI has two integer parameters: m ≥ 0 that allows to move from a VI-style update to a PI-style update, and glyph[lscript] ≥ 1 that characterizes the period of the nonstationary policy that it builds. In Theorem 3, we have provided a performance guarantee for this algorithm that is independent of m and that improves when glyph[lscript] increases; since glyph[lscript] directly controls the memory of the process, this allows to make a trade-off between memory and quality. In the literature, similar upper bounds were only known for AMPI (Scherrer et al., 2012)glyph[lscript] = 1 and m arbitraryand NS-AVI/NS-API (Scherrer &amp; Lesner, 2012)glyph[lscript] arbitrary but m ∈ { 0 , ∞} . For most settingsglyph[lscript] &gt; 1 and 1 ≤ m &lt; ∞ -the result is new. By exhibiting a specially designed MDP, we argued (Theorem 4) that our analysis is tight. Similar lower bounds were only known for AVI and APIglyph[lscript] = 1 and m ∈ { 0 , ∞} . In other words, we have generalized the scarce existing bounds in a unified setting and closed the gap between the upper and lower bounds for all values of m ≥ 0 and glyph[lscript] ≥ 1 .

A practical limitation of Theorem 3 is that it assumes that the errors glyph[epsilon1] k are controlled in max norm. In practice, the evaluation step of dynamic programming algorithm is usually done through some regression scheme-see for instance (Bertsekas &amp; Tsitsiklis, 1996; Antos et al., 2007a;b; Scherrer et al., 2012)-and thus controlled through the L 2 ,µ norm, defined as ‖ f ‖ 2 ,µ = √ ∫ f ( x ) µ ( dx ) . Munos (2003; 2007) originally developed such analyzes for AVI and API. Farahmand et al. (2010) and Scherrer et al. (2012) later improved them. Using a technical lemma due to Scherrer et al. (2012, Lemma 3), one can easily deduce 7 from our analysis (developed in Appendix A of the Supplementary Material) the following performance bound.

Corollary 1. Consider AMPI with any parameters m ≥ 0 and glyph[lscript] ≥ 1 . Assume there exists an glyph[epsilon1] &gt; 0 such that the errors satisfy ‖ glyph[epsilon1] k ‖ 2 ,µ &lt; glyph[epsilon1] for all k . Then, the expected (with respect to some initial measure ρ ) loss due to running policy π k,glyph[lscript] instead of the optimal policy π ∗ satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is a convex combination of concentrability coefficients based on Radon-Nikodym derivatives c ( j ) = max π 1 , ··· ,π j ∥ ∥ ∥ d ( ρP π 1 P π 2 ··· P π j ) dµ ∥ ∥ ∥ 2 ,µ .

With respect to the previous bound in norm ‖ · · · ‖ ∞ , this bound involves extra constants C j,k,l ≥ 1 . Each such coefficient C j,k,l is a convex combination of terms c ( i ) , that each quantifies the difference between 1) the distribution µ used to control the errors and 2) the distribution obtained by starting from ρ and making k steps with arbitrary sequences of policies. Overall, this extra constant can be seen as a measure of stochastic smoothness of the MDP (the smoother, the smaller). Further details on these coefficients can be found in (Munos, 2003; 2007; Farahmand et al., 2010).

We have shown on a small numerical study the significant influence of the parameter glyph[lscript] on the asymptotic quality of approximately optimal controllers, and suggested that optimizing the speed of convergence may require a fine schedule between glyph[lscript] and m . Instantiating and analyzing specific implementations of NS-AMPI as was done recently for AMPI (Scherrer et al., 2012), and applying them on large domains constitutes interesting future work.

7 Precisely, Lemma 3 of (Scherrer et al., 2012) should be applied to Equation (8) page 15 in Appendix A of the Supplementary Material.

## References

- Antos, A., Munos, R., and Szepesv´ ari, C. Fitted Q-iteration in continuous action-space MDPs. In NIPS , 2007a.
- Antos, A., Szepesv´ ari, C., and Munos, R. Value-iteration based fitted policy iteration: learning with a single trajectory. In Approximate Dynamic Programming and Reinforcement Learning, 2007. ADPRL 2007 , pp. 330-337. IEEE, 2007b.
- Bertsekas, D.P. and Tsitsiklis, J.N. Neuro-dynamic programming . Athena Scientific, 1996.
- Bertsekas, D.P. and Yu, H. Q-learning and enhanced policy iteration in discounted dynamic programming. Mathematics of Operations Research , 37(1):66-94, 2012.
- Canbolat, P. and Rothblum, U. (Approximate) iterated successive approximations algorithm for sequential decision processes. Annals of Operations Research , pp. 1-12, 2012. ISSN 0254-5330.
- Farahmand, A.M., Munos, R., and Szepesv´ ari, Cs. Error propagation for approximate policy and value iteration (extended version). In NIPS , 2010.
- Gabillon, Victor, Ghavamzadeh, Mohammad, and Scherrer, Bruno. Approximate Dynamic Programming Finally Performs Well in the Game of Tetris. In Neural Information Processing Systems (NIPS) 2013 , South Lake Tahoe, United States, December 2013.
- Kakade, S.M. On the Sample Complexity of Reinforcement Learning . PhD thesis, University College London, 2003.
- Munos, R. Error bounds for approximate policy iteration. In International Conference on Machine Learning (ICML) , pp. 560-567, 2003.
- Munos, R. Performance bounds in L p -norm for approximate value iteration. SIAM Journal on Control and Optimization , 46(2):541-561, 2007.
- Puterman, M.L. Markov decision processes: Discrete stochastic dynamic programming . John Wiley &amp; Sons, Inc., 1994.
- Puterman, M.L. and Shin, M.C. Modified policy iteration algorithms for discounted Markov decision problems. Management Science , 24(11):1127-1137, 1978.
- Scherrer, B. and Lesner, B. On the Use of Non-Stationary Policies for Stationary Infinite-Horizon Markov Decision Processes. In Advances in Neural Information Processing Systems 25 , pp. 1835-1843, 2012.
- Scherrer, B., Ghavamzadeh, M., Gabillon, V., and Geist, M. Approximate Modified Policy Iteration. In Proceedings of the 29th International Conference on Machine Learning (ICML-12) , pp. 1207-1214, July 2012.
- Singh, S. and Yee, R. An upper bound on the loss from approximate optimal-value functions. Machine Learning , 16-3:227-233, 1994.

## Supplementary Material for Non-Stationary Approximate Modified Policy Iteration

## A. Proof of Theorem 3

For clarity, we here provide a detailed and complete proof. Throughout this proof we will write P k (resp. P ∗ ) for the transition kernel P π k (resp. P π ∗ ) induced by the stationary policy π k (resp. π ∗ ). We will write T k (resp. T ∗ ) for the associated Bellman operator. Similarly, we will write P k,glyph[lscript] for the transition kernel associated with the non-stationary policy π k,glyph[lscript] and T k,glyph[lscript] for its associated Bellman operator.

For k ≥ 0 we define the following quantities:

- b k = T k +1 v k -T k +1 ,glyph[lscript] T k +1 v k . This quantity which we will call the residual may be viewed as a non-stationary analogue of the Bellman residual v k -T k +1 v k .
- s k = v k -v π k,glyph[lscript] -glyph[epsilon1] k . We will call it shift , as it measures the shift between the value v π k,glyph[lscript] and the estimate v k before incurring the error.
- d k = v ∗ -v k + glyph[epsilon1] k . This quantity, called distance thereafter, provides the distance between the k th value function (before the error is added) and the optimal value function.
- l k = v ∗ -v π k,glyph[lscript] . This is the loss of the policy v π k,glyph[lscript] . The loss is always non-negative since no policy can have a value greater than or equal to v ∗ .

The proof is outlined as follows. We first provide a bound on b k which will be used to express both the bounds on s k and d k . Then, observing that l k = s k + d k will allow to express the bound of ‖ l k ‖ ∞ stated by Theorem 3. Our arguments extend those made by Scherrer et al. (2012) in the specific case glyph[lscript] = 1 .

We will repeatedly use the fact that since policy π k +1 is greedy with respect to v k , we have

<!-- formula-not-decoded -->

For a non-stationary policy π k,glyph[lscript] , the induced glyph[lscript] -step transition kernel is

<!-- formula-not-decoded -->

As a consequence, for any function f : S → R , the operator T k,glyph[lscript] may be expressed as:

<!-- formula-not-decoded -->

then, for any function g : S → R , we have and

The following notation will be useful.

Definition 1 (Scherrer et al. (2012)) . For a positive integer n , we define P n as the set of discounted transition kernels that are defined as follows:

1. for any set of n policies { π 1 , . . . , π n } , ( γP π 1 )( γP π 2 ) · · · ( γP π n ) ∈ P n ,
2. for any α ∈ (0 , 1) and P 1 , P 2 ∈ P n , αP 1 +(1 -α ) P 2 ∈ P n

With some abuse of notation, we write Γ n for denoting any element of P n .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Example 1 ( Γ n notation) . If we write a transition kernel P as P = α 1 Γ i + α 2 Γ j Γ k = α 1 Γ i + α 2 Γ j + k , it should be read as: 'There exists P 1 ∈ P i , P 2 ∈ P j , P 3 ∈ P k and P 4 ∈ P j + k such that P = α 1 P 1 + α 2 P 2 P 3 = α 1 P 1 + α 2 P 4 . '.

We first provide three lemmas bounding the residual, the shift and the distance, respectively.

Lemma 2 (residual bound) . The residual b k satisfies the following bound:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Proof. We have:

<!-- formula-not-decoded -->

Which can be written as

<!-- formula-not-decoded -->

Then, by induction:

<!-- formula-not-decoded -->

Lemma 3 (distance bound) . The distance d k satisfies the following bound:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where and

Proof. First expand d k :

<!-- formula-not-decoded -->

Then, by induction

<!-- formula-not-decoded -->

Using the bound on b k from Lemma 2 we get:

<!-- formula-not-decoded -->

First we have:

<!-- formula-not-decoded -->

Second we have:

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

Lemma 4 (shift bound) . The shift s k is bounded by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Proof. Expanding s k we obtain:

<!-- formula-not-decoded -->

Plugging the bound on b k of Lemma 2 we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 5 (loss bound) . The loss l k is bounded by:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Proof. Using Lemmas 3 and 4, we have:

<!-- formula-not-decoded -->

Plugging back the values of x k and y k and using the fact that glyph[epsilon1] 0 = 0 we obtain:

<!-- formula-not-decoded -->

We now provide a bound of η k in terms of d 0 :

Lemma 6.

Proof. First recall that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In order to bound η k in terms of d 0 only, we express b 0 in terms of d 0 :

<!-- formula-not-decoded -->

Consequently, we have:

<!-- formula-not-decoded -->

We now conclude the proof of Theorem 3. Taking the absolute value in Lemma 6 we obtain:

<!-- formula-not-decoded -->

Since l k is non-negative, from Lemma 5 we have:

<!-- formula-not-decoded -->

Since ‖ v ‖ ∞ = max | v | , d 0 = v ∗ -v 0 and l k = v ∗ -v π k,glyph[lscript] , we can take the maximum in (8) and conclude that:

<!-- formula-not-decoded -->

## B. Proof of Theorem 4

We shall prove the following result.

Lemma 7. Consider NS-AMPI with parameters m ≥ 0 and glyph[lscript] ≥ 1 applied on the problem of Figure 1, starting from v 0 = 0 and all initial policies π 0 , π -1 , . . . , π -glyph[lscript] +2 equal to π ∗ . Assume that at each iteration k , the following error terms are applied, for some glyph[epsilon1] ≥ 0 :

<!-- formula-not-decoded -->

Then NS-AMPI can 8 generate a sequence of value-policy pairs that is described below.

For all iterations k ≥ 1 , the policy π k takes the optimal action in all states but k , that is

<!-- formula-not-decoded -->

For all iterations k ≥ 1 , the value function v k satisfies the following equations:

- For all i &lt; k :
- For all i &gt; k +(( k -1) m +1) glyph[lscript]

<!-- formula-not-decoded -->

- For all i such that k ≤ i ≤ k +(( k -1) m +1) glyph[lscript] :
- -For i = k +( qm + p +1) glyph[lscript] with q ≥ 0 and 0 ≤ p &lt; m (i.e. i = k + nglyph[lscript] , n ≥ 1 ):

<!-- formula-not-decoded -->

- -For i = k :

<!-- formula-not-decoded -->

- -For i = k + qglyph[lscript] + p with 0 ≤ q ≤ ( k -1) m -1 and 1 ≤ p &lt; glyph[lscript] :

<!-- formula-not-decoded -->

- -Otherwise, i.e. when i = k +( k -1) mglyph[lscript] + p with 1 ≤ p &lt; glyph[lscript] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The relative complexity of the different expressions of v k in Lemma 7 is due to the presence of nested periodic patterns in the shape of the value function along the state space and the horizon. Figures 4 and 5 give the shape of the value function for different values of glyph[lscript] and m , exhibiting the periodic patterns. The proof of Lemma 7 is done by recurrence on k .

## B.1. Base case k = 1

Since v 0 = 0 , π 1 is the optimal policy that takes ← in all states as desired. Hence, ( T 1 ,glyph[lscript] ) m T 1 v 0 = 0 in all states. Accounting for the errors glyph[epsilon1] 1 we have v 1 = ( T 1 ,glyph[lscript] ) m T 1 v 0 + glyph[epsilon1] 1 = glyph[epsilon1] 1 . As can be seen on Figures 4 and 5, when k = 1 we only need to consider equations (10.b), (10.c), (10.e) and (10.f) since the others apply to an empty set of states.

First, we have

<!-- formula-not-decoded -->

8 We write here 'can' since at each iteration, several policies will be greedy with respect to the current value.

Figure 4. Shape of the value function with glyph[lscript] = 2 and m = 3 .

<!-- image -->

i

(state)

Figure 5. Shape of the value function with glyph[lscript] = 3 and m = 2 .

<!-- image -->

10

10

(10.d)

(10.b)

(10.d)

(10.b)

(10.b)

(1

(10.

which is (10.b) when q = ( k -1) = 0 and p = 0 .

Second, we have which corresponds to (10.c).

Third, for 1 ≤ p &lt; glyph[lscript] we have corresponding to (10.e).

Finally, for all the remaining states i &gt; 1 + glyph[lscript] , we have

<!-- formula-not-decoded -->

corresponding to (10.f).

The base case is now proved.

## B.2. Induction Step

We assume that Lemma 7 holds for some fixed k ≥ 1 , we now show that it also holds for k +1 .

<!-- formula-not-decoded -->

We begin by showing that the policy π k +1 is greedy with respect to v k . Since there is no choice in state 1 is → , we turn our attention to the other states. There are many cases to consider, each one of them corresponding to one or more states. These cases, labelled from A through F, are summarized as follows, depending on the state i :

- (A) 1 &lt; i &lt; k +1
- (B) i = k +1
- (C) i = k +1+ qglyph[lscript] + p with 1 ≤ p &lt; glyph[lscript] and 0 ≤ q ≤ ( k -1) m

<!-- formula-not-decoded -->

- (E) i = k +1+(( k -1) m +1) glyph[lscript]
- (F) i &gt; k +1+(( k -1) m +1) glyph[lscript]

Figure 6 depicts how those cases cover the whole state space.

<!-- formula-not-decoded -->

Figure 6. Policy cases, each state is represented by a letter corresponding to a case of the policy π k +1 . Starting from 1 , state number increase from left to right.

For all states i &gt; 1 in each of the above cases, we consider the action-value functions q → k +1 ( i ) (resp. q ← k +1 ( i ) ) of action → (resp. ← ) defined as:

<!-- formula-not-decoded -->

In case i = k + 1 (B) we will show that q → k +1 ( i ) = q ← k +1 ( i ) meaning that a policy π k +1 greedy for v k may be either π k +1 ( k +1) = → or π k +1 ( k +1) = ← . In all other cases we show that q → k +1 ( i ) &lt; q ← k +1 ( i ) which implies that for those i = k +1 , π k +1 ( i ) = ← , as required by Lemma 7.

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A: In states 1 &lt; i &lt; k +1 We have q → k +1 ( i ) = r i + γv k ( i + glyph[lscript] -1) and q ← k +1 ( i ) = γv k ( i -1) , depending on the value of i + glyph[lscript] -1 , which is reached by taking the → action, we need to consider two cases:

glyph[negationslash]

- Case 1: i + glyph[lscript] -1 = k . In this case v k ( i + glyph[lscript] -1) is described by either (10.a) or (10.d) when i + glyph[lscript] -1 is less than, or greater than k , respectively. In either case we have v k ( i + glyph[lscript] -1) = -γ ( k -1)( glyph[lscript]m +1) glyph[epsilon1] = v k ( i -1) and hence:

<!-- formula-not-decoded -->

which gives π k +1 ( i ) = ← as desired.

- Case 2: i + glyph[lscript] -1 = k .

<!-- formula-not-decoded -->

giving π k +1 ( i ) = ← as desired.

- B: In state k +1 Looking at the action value function q ← k +1 in state k +1 , we observe that:

<!-- formula-not-decoded -->

This means that the algorithm can take π k +1 ( k +1) = → so as to satisfy Lemma 7.

- C: In states i = k +1+ qglyph[lscript] + p We restrict ourselves to the cases when 1 ≤ p &lt; glyph[lscript] and 0 ≤ q ≤ ( k -1) m . Three cases for the value of q need to be considered:
- Case 1: 0 ≤ q &lt; ( k -1) m -1 . We have:

<!-- formula-not-decoded -->

- Case 2: q = ( k -1) m -

<!-- formula-not-decoded -->

- Case 3: q = ( k -1) m

<!-- formula-not-decoded -->

D: In states i = k +1+( qm + p +1) glyph[lscript] In these states, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As for the right-hand side of (11) we need to consider two cases:

- Case 1: p +1 &lt; m :

In the following, define

Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, observe that

<!-- formula-not-decoded -->

Plugging this back into (12), we get:

<!-- formula-not-decoded -->

- Case 2: p +1 = m :

Using the fact that p +1 = m implies γ glyph[lscript] ( p +1) -γ glyph[lscript] ( m +1) 1 -γ glyph[lscript] = γ glyph[lscript]m we have:

<!-- formula-not-decoded -->

where we concluded by observing that this is the same result as (12).

<!-- formula-not-decoded -->

F: In states i &gt; k +(( k -1) m +1) glyph[lscript] +1 Following (10.f) we have v k ( i -1) = v k ( i + glyph[lscript] -1) = 0 and hence

<!-- formula-not-decoded -->

## B.2.2. THE VALUE FUNCTION v k +1

In the following we will show that the value function v k +1 satisfies Lemma 7. To that end we consider the value of (( T k +1 ,glyph[lscript] ) m T k +1 v k )( s 0 ) by analysing the trajectories obtained by first following m times π k,glyph[lscript] then π k +1 from various starting states s 0 .

Given a starting state s 0 and a non stationary policy π k +1 ,glyph[lscript] , we will represent the trajectories as a sequence of triples ( s i , a i , r ( s i , a i )) i =0 ,...,glyph[lscript]m arranged in a 'trajectory matrix' of glyph[lscript] columns and m rows. Each column corresponds to one of the policies π k +1 , π k , . . . , π k +2 -glyph[lscript] . In a column labeled by policy π j the entries are of the form ( s i , π j ( s i ) , r ( s i , π j ( s i )) ; this layout makes clear which stationary policy is used to select the action in any particular step in the trajectory. Indeed, in column π j , we have ( s i , → , r j ) if and only if s i = j , otherwise each entry is of the form ( s i , ← , 0) . Such a matrix accounts for the first m applications of the operator T k +1 ,glyph[lscript] . One addional row of only one triple ( s i , π k +1 ( s i ) , r π k +1 ( s i )) represents the final application of T k +1 . After this triple comes the end state of the trajectory s glyph[lscript]m +1 .

Figure 7. The trajectory matrix of policy π 4 ,glyph[lscript] starting from state 10 with m = 4 and glyph[lscript] = 3 .

| m = 4 times   | glyph[lscript] = 3 steps   | glyph[lscript] = 3 steps   | glyph[lscript] = 3 steps   |
|---------------|----------------------------|----------------------------|----------------------------|
| m = 4 times   | π 4                        | π 3                        | π 2                        |
| m = 4 times   | (10 , ← , 0)               | (9 , ← , 0)                | (8 , ← , 0)                |
| m = 4 times   | (7 , ← , 0)                | (6 , ← , 0)                | (5 , ← , 0)                |
| m = 4 times   | (4 , → , r 4 )             | (6 , ← , 0)                | (5 , ← , 0)                |
| m = 4 times   | (4 , → , r 4 )             | (6 , ← , 0)                | (5 , ← , 0)                |
| m = 4 times   | (4 , → , r 4 )             | 6                          |                            |

Example 2. Figure 7 depicts the trajectory matrix of policy π 4 ,glyph[lscript] = π 4 π 3 π 2 with m = 4 and glyph[lscript] = 3 . The trajectory starts from state s 0 = 10 and ends in state s glyph[lscript]m +1 = 6 . The ← action is always taken with reward 0 except when in state 4 under the policy π 4 . From this matrix we can deduce that, for any value function v :

<!-- formula-not-decoded -->

With this in hand, we are going to prove each case of Lemma 7 for v k +1 .

In states i &lt; k +1 Following m times π k +1 ,glyph[lscript] and then π k +1 starting from these states consists in taking the ← action glyph[lscript]m +1 times to eventually finish either in state 1 if i ≤ glyph[lscript]m +2 with value

<!-- formula-not-decoded -->

or otherwise in state i -glyph[lscript]m -1 &lt; k with value

<!-- formula-not-decoded -->

This matches Equation (10.a) in both cases.

In states i = k + 1 + ( qm + p + 1) glyph[lscript] Consider the states i = k + 1 + ( qm + p + 1) glyph[lscript] with q ≥ 0 and 0 ≤ p &lt; m . Following m times π k +1 ,glyph[lscript] and then π k +1 starting from state i gives the following trajectories:

- when q = 0 , ( i.e. i = k +1+( p +1) glyph[lscript] ):

<!-- formula-not-decoded -->

Using (10.b) with q = p = 0 as our induction hypothesis, this gives

<!-- formula-not-decoded -->

Accounting for the error term and the fact that i = k +1+ glyph[lscript] ⇐⇒ p = q = 0 , we get

<!-- formula-not-decoded -->

which is (10.b) for k +1 and q = 0 as desired.

- when 1 ≤ q ≤ k :

In this case we have i -( glyph[lscript]m + 1) ≥ k + 1 , meaning that k + 1 , the first state where the → action would be available is unreachable (in the sense that the tractory could end in k +1 , but no action will be taken there). Consequently the ← action is taken glyph[lscript]m +1 times and the system ends in state i -glyph[lscript]m -1 = k +(( q -1) m + p +1) glyph[lscript] . Therefore, using (10.b) as induction hypothesis and the fact that i glyph[negationslash]∈ { k +1 , k + glyph[lscript] +1 } = ⇒ glyph[epsilon1] k +1 ( i ) = 0 , we have:

<!-- formula-not-decoded -->

which statisfies (10.b) for k +1 .

In state k +1 Following m times π k +1 ,glyph[lscript] and then π k +1 starting from k +1 gives the following trajectory:

<!-- formula-not-decoded -->

As a consequence, with (10.c) as induction hypothesis we have:

<!-- formula-not-decoded -->

Hence, which matches (10.c).

In states i = k +1 + qglyph[lscript] + p For states i = k +1 + qglyph[lscript] + p with 0 ≤ q ≤ km -1 and 1 ≤ p &lt; glyph[lscript] , the policy π k +1 ,glyph[lscript] always takes the ← action with either one of the following trajectories

- when q ≥ m :

<!-- formula-not-decoded -->

As a consequence, with (10.d) as induction hypothesis we have:

<!-- formula-not-decoded -->

which satisfies (10.d) in this case.

- when q &lt; m :

<!-- formula-not-decoded -->

Assuming that negative states correspond to state 1 , where the action is irrelevant, we have the following trajectory:

|             | glyph[lscript] steps                                 | glyph[lscript] steps   | glyph[lscript] steps                                     |
|-------------|------------------------------------------------------|------------------------|----------------------------------------------------------|
|             | π k +1                                               | . . .                  | π k - glyph[lscript] +2                                  |
| q times     | ( k +1+ qglyph[lscript] + p, ← , 0) . . .            | . . . . . .            | ( k +( q - 1) glyph[lscript] + p +2 , ← , 0) . . .       |
|             | ( k +1+ glyph[lscript] + p, ← , 0) ( k +1+ p, ← , 0) | . . . . . .            | ( k + p +2 , ← , 0) ( k - glyph[lscript] + p +2 , ← , 0) |
| m - q times | . . . +1 - ( m - q - 1) glyph[lscript] + p, ← , 0)   | . . . . . .            | . . . k - ( m - q ) glyph[lscript] + p +2 , ← , 0)       |
|             | ( k                                                  | .                      | ( k +( q - m ) glyph[lscript] + p                        |
|             | ( k +1 - glyph[lscript] + p, ← , 0)                  |                        | ( k - 2 glyph[lscript] + p +2 , ← , 0)                   |
|             | ( k +1 - ( m - q ) glyph[lscript] + p, ← ,           | . .                    |                                                          |
|             | 0)                                                   |                        |                                                          |

In the above trajectory, one can see that only the ← action is taken (ignoring state 1 ). Indeed, since we follow the policies π k +1 π k , . . . , π k -glyph[lscript] +2 the → action may only be taken in states k +1 , k, . . . , k -glyph[lscript] +2 . When state k +1 is reached, the selected action is π k -p +1 ( k +1) which is ← since p ≥ 1 . The same reasonning applies in the next states k, . . . , k -glyph[lscript] +1 , where p ≥ 1 prevents to use a policy that would select the → action in those states.

Since p -glyph[lscript] &lt; 0 the trajectory always terminates in a state j &lt; k with value v k ( j ) = -γ ( k -1)( glyph[lscript]m -1) glyph[epsilon1] as for the q ≥ m case, which allows to conclude that (10.d) also holds in this case.

In states i = k +1+ kmglyph[lscript] + p Observe that following m times π k +1 ,glyph[lscript] and then π k +1 once amounts to always take ← actions. Thus, one eventually finishes in state k +( k -1) mglyph[lscript] + p ≥ k +1 , which, since glyph[epsilon1] k ( i ) = 0 , gives

<!-- formula-not-decoded -->

satisfiying (10.e).

In states i &gt; k +1+( km +1) glyph[lscript] In these states, the action ← is taken glyph[lscript]m +1 times ending up in state j &gt; k +(( k -1) m +1) glyph[lscript] , with value v k ( j ) = 0 , from which v k +1 ( i ) = 0 follows as required by (10.f).