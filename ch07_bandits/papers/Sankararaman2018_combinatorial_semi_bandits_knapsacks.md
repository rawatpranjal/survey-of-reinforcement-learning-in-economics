## Combinatorial Semi-Bandits with Knapsacks ∗

Karthik A. Sankararaman †

University of Maryland, College Park

Aleksandrs Slivkins ‡

Microsoft Research NYC

## Abstract

We unify two prominent lines of work on multi-armed bandits: bandits with knapsacks and combinatorial semi-bandits . The former concerns limited 'resources' consumed by the algorithm, e.g., limited supply in dynamic pricing. The latter allows a huge number of actions but assumes combinatorial structure and additional feedback to make the problem tractable. We define a common generalization, support it with several motivating examples, and design an algorithm for it. Our regret bounds are comparable with those for BwK and combinatorial semi-bandits.

## 1 Introduction

Multi-armed bandits ( MAB ) is an elegant model for studying the tradeoff between acquisition and usage of information, a.k.a. explore-exploit tradeoff [Robbins, 1952, Thompson, 1933]. In each round an algorithm sequentially chooses from a fixed set of alternatives (sometimes known as actions or arms ), and receives reward for the chosen action. Crucially, the algorithm does not have enough information to answer all 'counterfactual' questions about what would have happened if a different action was chosen in this round. MABproblems have been studied steadily since 1930-ies, with a huge surge of interest in the last decade.

This paper combines two lines of work related to bandits: on bandits with knapsacks ( BwK ) [Badanidiyuru et al., 2013a] and on combinatorial semi-bandits [Gy¨ orgy et al., 2007]. BwK concern scenarios with limited 'resources' consumed by the algorithm, e.g., limited inventory in a dynamic pricing problem. In combinatorial semi-bandits, actions correspond to subsets of some 'ground set', rewards are additive across the elements of this ground set ( atoms ), and the reward for each chosen atom is revealed ( semi-bandit feedback ). A paradigmatic example is an online routing problem, where atoms are edges in a graph, and actions are paths. Both lines of work have received much recent attention and are supported by numerous examples.

Our contributions. We define a common generalization of combinatorial semi-bandits and BwK , termed Combinatorial Semi-Bandits with Knapsacks ( SemiBwK ). Following all prior work on BwK , we focus on an i.i.d. environment: in each round, the 'outcome' is drawn independently from a fixed distribution over the

∗ Extended abstract appears in the 21st International Conference on Artificial Intelligence and Statistics ( AIStats 2018 ).

† Email: kabinav@cs.umd.edu Supported in part by NSF Awards CNS 1010789 and CCF 1422569.

‡ Email: slivkins@microsoft.com

possible outcomes. Here the 'outcome' of a round is the matrix of reward and resource consumption for all atoms. 1 We design an algorithm for SemiBwK , achieving regret rates that are comparable with those for BwK and combinatorial semi-bandits.

Specifics are as follows. As usual, we assume 'bounded outcomes': for each atom and each round, rewards and consumption of each resource is non-negative and at most 1 . Regret is relative to the expected total reward of the best all-knowing policy, denoted OPT . For BwK problems, this is known to be a much stronger benchmark than the traditional best-fixed-arm benchmark. We upper-bound the regret in terms of the relevant parameters: time horizon T , (smallest) budget B , number of atoms n , and OPT itself (which may be as large as nT ). We obtain

<!-- formula-not-decoded -->

√

The 'shape' of the regret bound is consistent with prior work: the OPT / B additive term appears in the optimal regret bound for BwK , and the √ T and √ OPT additive terms are very common in regret bounds for MAB. The per-round running time is polynomial in n , and near-linear in n for some important special cases.

Our regret bound is optimal up to polylog factors for paradigmatic special cases. BwK is a special case when actions are atoms. For OPT &gt; Ω( T ) , the regret bound becomes ˜ O ( T √ n/B + √ nT ) , where n is the number of actions, which coincides with the lower bound from [Badanidiyuru et al., 2013a]. Combinatorial semi-bandits is a special case with B = nT . If all feasible subsets contain at most k atoms, we have OPT ≤ kT , and the regret bound becomes ˜ O ( √ knT ) . This coincides with the Ω( √ knT ) lower bound from [Kveton et al., 2014].

Our main result assumes that the action set, i.e., the family of feasible subsets of atoms, is described by a matroid constraint . 2 This is a rather general scenario which includes many paradigmatic special cases of combinatorial semi-bandits such as cardinality constraints, partition matroid constraints, and spanning tree constraints. We also assume that B &gt; ˜ Ω( n + √ nT ) .

Our model captures several application scenarios, incl. dynamic pricing, dynamic assortment, repeated auctions, and repeated bidding. We work out these applications, and explain how our regret bounds improve over prior work.

Challenges and techniques. BwK problems are challenging compared to traditional MAB problems with i.i.d. rewards because it no longer suffices to look for the best action and/or optimize expected per-round rewards; instead, one essentially needs to look for a distribution over actions with optimal expected total reward across all rounds. Generic challenges in combinatorial semi-bandits concern handling exponentially many actions (both in terms of regret and in terms of the running time), and taking advantage of the additional feedback. And in SemiBwK , one needs to deal with distributions over subsets of atoms, rather than 'just' with distributions over actions.

Our algorithm connects a technique from BwK and a randomized rounding technique from combinatorial optimization. (With five existing BwK algorithms and a wealth of approaches for combinatorial optimization, choosing the techniques is a part of the challenge.)

We build on a BwK algorithm from Agrawal and Devanur [2014a], which combines linear relaxations and a well-known 'optimism-under-uncertainty' paradigm. A generalization of this algorithm to SemiBwK results in a fractional solution x , a vector over atoms. Randomized rounding converts x into a distribution over feasible subsets of atoms that equals x in expectation. It is crucial (and challenging) to ensure that this

1 Our model allows arbitrary correlations within a given round, both across rewards and consumption for the same atom and across multiple atoms. Such correlations are essential in applications such as dynamic pricing and dynamic assortment. E.g., customers' valuations can be correlated across products, and algorithm earns only if it sells; see Section 5 for details.

2 Matroid is a standard notion in combinatorial optimization which abstracts and generalizes linear independence.

distribution contains enough randomness so as to admit concentration bounds not only across rounds, but also across atoms. Our analysis 'opens up' a fairly technical proof from prior work and intertwines it with a new argument based on negative correlation.

We present our algorithm and analysis so as to 'plug in' any suitable randomized rounding technique. This makes our presentation more lucid, and also leads to faster running times for important special cases.

Solving SemiBwK using prior work. Solving SemiBwK using an algorithm for BwK would result in a regret bound like (1.1) with n replaced with the number of actions. The latter could be on the order of n k if each action can consist of at most k atoms, or perhaps even exponential in n .

SemiBwK can be solved as a special case of a much more general linear-contextual extension of BwK from Agrawal and Devanur [2014a, 2016]. In their model, an algorithm takes advantage of the combinatorial structure of actions, yet it ignores the additional feedback from the atoms. Their regret bounds have a worse dependence on the parameters, and apply for a much more limited range of parameters. Further, their per-round running time is linear in the number of actions, which is often prohibitively large.

To compare the regret bounds, let us focus on instances of SemiBwK in which at most one unit of each resource is consumed in each round. (This is the case in all our motivating applications, except repeated bidding.) Then Agrawal and Devanur [2014a, 2016] assume B &gt; √ nT 3 / 4 , and achieve regret ˜ O ( n √ T OPT B + n 2 √ T ) . 3 It is easy to see that we improve upon the range and upon both summands. In particular, we improve both summands by the factor of n √ n in a lucid special case when B &gt; Ω( T ) and OPT &lt; O ( T ) . 4

Werun simulations to compare our algorithm against prior work on BwK and combinatorial semi-bandits.

Related work. Multi-armed bandits have been studied since Thompson [1933] in Operations Research, Economics, and several branches of Computer Science, see [Gittins et al., 2011, Bubeck and Cesa-Bianchi, 2012] for background. Among broad directions in MAB, most relevant is MAB with i.i.d. rewards, starting from [Lai and Robbins, 1985, Auer et al., 2002].

Bandits with Knapsacks ( BwK ) were first introduced by Badanidiyuru et al. [2013a] as a common generalization of several models from prior work and many other motivating examples. Subsequent papers extended BwK to 'smoother' resource constraints and introduced several new algorithms [Agrawal and Devanur, 2014a], and generalized BwK to contextual bandits [Badanidiyuru et al., 2014, Agrawal et al., 2016, Agrawal and Devanur, 2016]. All prior work on BwK and special cases thereof assumed i.i.d. outcomes.

Special cases of BwK include dynamic pricing with limited supply [Babaioff et al., 2015, Besbes and Zeevi, 2009, 2012, Wang et al., 2014], dynamic procurement on a budget [Badanidiyuru et al., 2012, Singla and Krause, 2013, Slivkins and Vaughan, 2013], dynamic ad allocation with advertiser budgets [Slivkins, 2013], and bandits with a single deterministic resource [Guha and Munagala, 2007, Gupta et al., 2011, Tran-Thanh et al., 2010, 2012]. Some special cases admit instance-dependent logarithmic regret bounds [Xia et al., 2016b,a, Combes et al., 2015a, Slivkins, 2013] when there is only one bounded resource and unbounded time, or when resource constraints do not bind across arms.

Combinatorial semi-bandits were studied by Gy¨ orgy et al. [2007], in the adversarial setting. In the i.i.d. setting, in a series of works by [Anantharam et al., 1987, Gai et al., 2010, 2012, Chen et al., 2013, Kveton et al., 2015b, Combes et al., 2015b], an optimal algorithm was achieved. This result was then extended

√

√

4 In prior work on combinatorial bandits (without constraints), semi-bandit feedback improves regret bound by a factor of n , see the discussion in Kveton et al. [2015b].

3 Agrawal and Devanur [2014a, 2016] state regret bound with term + n T rather than + n 2 T , but they assume that per-round rewards lie in [0 , 1] . Since per-round rewards can be as large as n in our setting, we need to scale down all rewards by a factor of n , apply their regret bound, and then scale back, which results in the regret bound with + n 2 √ T . When per-round consumption can be as large as n , regret bound from Agrawal and Devanur [2014a, 2016] becomes ˜ O ( n 2 OPT √ T/B + n 2 √ T ) due to rescaling. √

to atoms with linear rewards by Wen et al. [2015]. Kveton et al. [2014] obtained improved results for the special case when action set is described by a matroid. Some other works studied a closely related 'cascade model', where the ordering of atoms matters [Kveton et al., 2015a, Katariya et al., 2016, Zong et al., 2016]. Contextual semi-bandits have been studied in [Wen et al., 2015, Krishnamurthy et al., 2016].

Randomized rounding schemes (RRS) come from the literature on approximation algorithms in combinatorial optimization (see Williamson and Shmoys [2011], Papadimitriou and Steiglitz [1982] for background). RRS were introduced in Raghavan and Tompson [1987]. Subsequent work [Gandhi et al., 2006, Asadpour et al., 2010, Chekuri et al., 2010, 2011] developed RRS which correlate the rounded random variables so as to guarantee sharp concentration bounds.

Discussion. The basic model of multi-armed bandits can be extended in many distinct directions: what auxiliary information, if any, is revealed to the algorithm before it needs to make a decision, which feedback is revealed afterwards, which 'process' are the rewards coming from, do they have some known structure that can be leveraged, are there global constraints on the algorithm, etc. While many real-life scenarios combine several directions, most existing work proceeds along only one or two. We believe it is important (and often quite challenging) to unify these lines of work. For example, an important recent result of Syrgkanis et al. [2016], Rakhlin and Sridharan [2016] combined 'contextual' and 'adversarial' bandits.

## 2 Our model and preliminaries

Our model, called Semi-Bandits with Knapsacks ( SemiBwK ) is a generalization of multi-armed bandits (henceforth, MAB ) with i.i.d. rewards. As such, in each round t = 1 , . . . , T , an algorithm chooses an action S t from a fixed set of actions F , and receives a reward µ t ( S t ) for this action which is drawn independently from a fixed distribution that depends only on the chosen action. The number of rounds T , a.k.a. the time horizon , is known.

There are d resources being consumed by the algorithm. The algorithm starts out with budget B j ≥ 0 of each resource j . All budgets are known to the algorithm. If in round t action S ∈ F is chosen, the outcome of this round is not only the reward µ t ( S ) but the consumption C t ( S, j ) of each resource j ∈ [ d ] . We refer to C t ( S ) = ( C t ( S, j ) : j ∈ [ d ]) as the consumption vector . 5 Following prior work on BwK , we assume that all budgets are the same: B j = B for all resources j . 6 Algorithm stops as soon as any one of the resources goes strictly below 0. The round in which this happens is called the stopping time and denoted τ stop . The reward collected in this last round does not count; so the total reward of the algorithm is rew = ∑ t&lt;τ stop µ t ( S t ) .

Actions correspond to subsets of a finite ground set A , with n = |A| ; we refer to elements of A as atoms . Thus, the set F of actions corresponds to the family of 'feasible subsets' of A . The rewards and resource consumption is additive over the atoms: for each round t and each atom a there is a reward µ t ( a ) ∈ [0 , 1] and consumption vector C t ( a ) ∈ [0 , 1] d such that for each action S ⊂ F it holds that µ t ( S ) = ∑ a ∈ S µ t ( a ) and C t ( S ) = ∑ a ∈ S C t ( a ) .

We assume the i.i.d. property across rounds, but allow arbitrary correlations within the same round. Formally, for a given round t we consider the n × ( d +1) 'outcome matrix' ( µ t ( a ) , C t ( a ) : a ∈ A ) , which specifies rewards and resource consumption for all resources and all atoms. We assume that the outcome matrix is chosen independently from a fixed distribution D M over such matrices. The distribution D M is not revealed to the algorithm. The mean rewards and mean consumption is denoted µ ( a ) := E [ µ t ( a )] and

5 We use bold font to indicate vectors and matrices.

6 This is w.l.o.g. because we can divide all consumption of each resource j by B j / min j ′ ∈ [ d ] B j ′ . Effectively, B is the smallest budget in the original problem instance.

C ( a ) := E [ C t ( a )] . We extend the notation to actions, i.e., to subsets of atoms: µ ( S ) := ∑ a ∈ S µ ( a ) and C ( S ) := ∑ a ∈ S C ( a ) .

An instance of SemiBwK consists of the action set F ⊂ 2 [ n ] , the budgets B = ( B j : j ∈ [ d ]) , and the distribution D M . The F and B are known to the algorithm, and D M is not. As explained in the introduction, SemiBwK subsumes Bandits with Knapsacks ( BwK ) and semi-bandits. BwK is the special case when F consists of singletons, and semi-bandits is the special case when all budgets are equal to B j = nT (so that the resource consumption is irrelevant).

Following the prior work on BwK , we compete against the 'optimal all-knowing algorithm': an algorithm that optimizes the expected total reward for a given problem instance; its expected total reward is denoted by OPT . As observed in Badanidiyuru et al. [2013a], OPT can be much larger ( e.g., factor of 2 larger) than the expected cumulative reward of the best action, for a variety of important special cases of BwK . Our goal is to minimize regret , defined as OPT minus algorithm's total reward.

Combinatorial constraints. Action set F is given by a combinatorial constraint , i.e., a family of subsets. Treating subsets of atoms as n -dimensional binary vectors, F corresponds to a finite set of points in R n . We assume that the convex hull of F forms a polytope in R n . In other words, there exists a set of linear constraints over R n whose set of feasible integral solutions is F . We call such F linearizable ; the convex hull is called the polytope induced by F .

Our main result is for matroid constraints , a family of linearizable combinatorial constraints which subsumes several important special cases such as cardinality constraints, partition matroid constraints, spanning tree constraints and transversal constraints. Formally, F is a matroid if it contains the empty set, and satisfies two properties: (i) if F contains a subset S , then it also contains every subset of S , and (ii) for any two subsets S, S ′ ∈ F with | S | &gt; | S ′ | it holds that S ′ ∪{ a } ∈ F for each atom a ∈ S \ S ′ . See Appendix B for more background and examples.

We incorporate prior work on randomized rounding for linear programs. Consider a linearizable action set F with induced polytope P ⊂ [0 , 1] n . The randomized rounding scheme (henceforth, RRS ) for F is an algorithm which inputs a feasible fractional solution x ∈ P and the linear equations describing P , and produces a random vector Y over F . Weconsider RRS 's such that E [ Y ] = x and Y is negatively correlated (see below for definition); we call such RRS 's negatively correlated . Several such RRS are known: e.g., for cardinality constraints and bipartite matching [Gandhi et al., 2006], for spanning trees [Asadpour et al., 2010], and for matroids [Chekuri et al., 2010].

Negative correlation. Let X = ( X 1 , X 2 , . . . , X m ) denote a family of random variables which take values in [0 , 1] . Let X := 1 m ∑ m i =1 X i be the average, and µ := E [ X ] .

Family X is called negatively correlated if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Independent random variables satisfy both properties with equality. For intuition: if X 1 , X 2 are Bernoulli and (2.1) is strict, then X 1 is more likely to be 0 if X 2 = 1 .

Negative correlation is a generalization of independence that allows for similar concentration bounds , i.e., high-probability upper bounds on | X -µ | . However, our analysis does not invoke them directly. Instead,

we use a concentration bound given a closely related property:

<!-- formula-not-decoded -->

Theorem 2.1. If (2.3) , then for some absolute constant c ,

<!-- formula-not-decoded -->

This theorem easily follows from [Impagliazzo and Kabanets, 2010], see Appendix A.

Confidence radius. We bound deviations | X -µ | in a way that gets sharper when µ is small, without knowing µ in advance. (We use the notation X , X, µ as above.) To this end, we use the notion of confidence radius from [Kleinberg et al., 2015, Babaioff et al., 2015, Badanidiyuru et al., 2013a, Agrawal and Devanur, 2014b] 7 :

<!-- formula-not-decoded -->

If random variables X are independent, then event

<!-- formula-not-decoded -->

happens with probability at least 1 -O ( e -Ω( α ) ) , for any given α &gt; 0 . We use this notion to define upper/lower confidence bounds on the mean rewards and mean resource consumption. Fix round t , atom a , and resource j . Let ˆ µ t ( a ) and ˆ C t ( a, j ) denote the empirical average of the rewards and resourcej consumption, resp., between rounds 1 and t -1 . Let N t ( a ) be the number of times atom a has been chosen in these rounds ( i.e., included in the chosen actions). The confidence bounds are defined as

<!-- formula-not-decoded -->

where proj ( x ) := argmin y ∈ [0 , 1] | y -x | denotes the projection into [0 , 1] . Wealways use the same parameter α = c conf log( ndT ) , for an appropriately chosen absolute constant c conf . We suppress α and c conf from the notation. We use a vector notation µ ± t and C ± t ( j ) to denote the corresponding n -dimensional vectors over all atoms a .

By (2.6), with probability 1 -O ( e -Ω( α ) ) the following hold.

<!-- formula-not-decoded -->

## 3 Main algorithm

Let us define our main algorithm, called SemiBwK -RRS . The algorithm builds on an arbitrary RRS for the action set F . It is parameterized by this RRS , the polytope P induced by F (represented as a collection of

7 For instance Theorem 2.1 in [Badanidiyuru et al., 2013b]

linear constraints), and a number glyph[epsilon1] &gt; 0 . In each round t , it recomputes the upper/lower confidence bounds, as defined in (2.7), and solves the following linear program:

<!-- formula-not-decoded -->

This linear program defines a linear relaxation of the original problem which is 'optimistic' in the sense that it uses upper confidence bounds for rewards and lower confidence bounds for consumption. The linear relaxation is also 'conservative' in the sense that it rescales the budget by 1 -glyph[epsilon1] . Essentially, this is to ensure that the algorithm does not run out of budget with high probability. Parameter glyph[epsilon1] will be fixed throughout. For ease of notation, we will denote B glyph[epsilon1] := (1 -glyph[epsilon1] ) B henceforth. The LP solution x can be seen as a probability vector over the atoms. Finally, the algorithm uses the RRS to convert the LP solution into a feasible action. The pseudocode is given as Algorithm 1.

## Algorithm 1: SemiBwK -RRS

input: an RRS for action set F , induced polytope P (as a set of linear constraints), glyph[epsilon1] &gt; 0 .

<!-- formula-not-decoded -->

1. Recompute Confidence Bounds as in (2.7)
2. Obtain fractional solution x t ∈ [0 , 1] n by solving the linear program LP ALG .
3. Obtain a feasible action S t ∈ F by invoking the RRS on vector x t .
4. Semi-bandit Feedback : observe the rewards/consumption for all atoms a ∈ S t .

If action set F is described by a matroid constraint, we can use the negatively correlated RRS from Chekuri et al. [2010]. In particular, we obtain a complete algorithm for several combinatorial constraints commonly used in the literature on semi-bandits, such as partition matroid constraints, spanning trees. More background on matroid constraints can be found in the Appendix B.

Theorem 3.1. Consider the SemiBwK problem with a linearizable action set F that admits a negatively correlated RRS . Then algorithm SemiBwK -RRS with this RRS achieves expected regret bound at most

<!-- formula-not-decoded -->

Here T is the time horizon, n is the number of atoms, and B is the budget. We require B &gt; 3( αn + √ αnT ) , where α = Θ(log( ndT )) is the parameter in confidence radius. Parameter glyph[epsilon1] in the algorithm is set to √ αn B + αn B + √ αnT B .

Corollary 3.2. Consider the setting in Theorem 3.1 and assume that the action set F is defined by a matroid on the set of atoms. Then, using the negatively correlated RRS from [Chekuri et al., 2010], we obtain regret bound (3.1) .

Running time of the algorithm. The algorithm does two computationally intensive steps in each round: solves the linear program ( LP ALG ) and runs the RRS . For matroid constraints, the RRS from Chekuri et al.

[2010] has O ( n 2 ) running time. Hence, in the general case the computational bottleneck is solving the LP, which has n variables and O (2 n ) constraints. Matroids are known to admit a polynomial-time seperation oracle [ e.g., see Schrijver, 2002]. It follows that the entire set of constraints in LP ALG admits a polynomialtime separation oracle, and therefore we can use the Ellipsoid algorithm to solve LP ALG in polynomial time. For some classes of matroid constraints the LP is much smaller: e.g., for cardinality constraints (just d +1 constraints) and for traversal matroids on bipartite graphs (just 2 n + d constraints). Then near-linear-time algorithms can be used.

Our algorithm works under any negatively correlated RRS. We can use this flexibility to improve the per-round running time for some special cases. (Making decisions extremely fast is often critical in practical applications of bandits [ e.g., see Agarwal et al., 2016].) We obtain near-linear per-round running times for cardinality constraints and partition matroid constraints. Indeed, LP ALG can be solved in near-linear time, as mentioned above, and we can use a negatively correlated RRS from [Gandhi et al., 2006] which runs in linear time. These classes of matroid constraints are important in our applications (see Section 5).

## 4 Proof of Theorem 3.1

Proof overview. First, we argue that LP ALG provides a good benchmark that we can use instead of OPT . Specifically, at any given round, the optimal value for LP ALG in each round is at least 1 T (1 -glyph[epsilon1] ) OPT with high probability. We prove this by constructing a series of LPs, starting with a generic linear relaxation for BwK and ending with LP ALG , and showing that the optimal value does not decrease along the series.

Next we define an event that occur with high probability, henceforth called clean event . This event concerns total rewards, and compares our algorithm against LP ALG :

<!-- formula-not-decoded -->

We prove that it is indeed a high-probability event in three steps. First, we relate the algorithm's reward ∑ t r t to its expected reward ∑ t µ · S t , where we interpret the chosen action S t , a subset of atoms, as a binary vector over the atoms. Then we relate ∑ t µ · S t to ∑ t µ + t · S t , replacing expected rewards with the upper confidence bounds. Finally, we relate ∑ t µ + t · S t to ∑ t µ + t · x t , replacing the output of the RRS with the corresponding expectations. Putting it together, we relate algorithm's reward to ∑ t µ + t · x t , as needed. It is essential to bound the deviations in the sharpest way possible; in particular, the naive ˜ O ( √ T ) bounds are not good enough. To this end, we use several tools: the confidence radius from (2.5), the negative correlation property of the RRS, and another concentration bound from prior work.

A similar 'clean event' (with a similar proof) concerns the total resource consumption of the algorithm. We condition on both clean events, and perform the rest of the analysis via a 'deterministic' argument not involving probabilities. In particular, we use the second 'clean event' to guarantee that the algorithm never runs out of resources.

We use negative correlation via a rather delicate argument. We extend the concentration bound in Theorem 2.1 to a random process that evolves over time, and only assumes that property (2.3) holds within each round conditional on the history. For a given round, we start with a negative correlation property of S t and construct another family of random variables that conditionally satisfies (2.3). The extended concentration bound is then applied to this family. The net result is a concentration bound for ∑ t µ + t · S t as if we had n × T independent random variables there.

The rest of the section contains the full proof.

## 4.1 Linear programs

We argue that LP ALG provides a good benchmark that we can use instead of OPT . Fix round t and let OPT ALG , t denote the optimal value for LP ALG in this round. Then:

<!-- formula-not-decoded -->

We will prove this by constructing a series of LP's, starting with a generic linear relaxation for BwK and ending with LP ALG . We show that along the series the optimal value does not decrease with high probability.

The first LP, adapted from Badanidiyuru et al. [2013a], has one decision variable for each action, and applies generically to any BwK problem.

<!-- formula-not-decoded -->

Let OPT BwK ( B ) denote the optimal value of this LP with a given budget B . Then:

<!-- formula-not-decoded -->

Proof. The second inequality in Claim 4.2 follows from [Lemma 3.1 in Badanidiyuru et al., 2013a]. We will prove the first inequality as follows. Let x ∗ denote an optimal solution to LP BwK (B). Consider (1 -glyph[epsilon1] ) x ∗ ; this is feasible to LP BwK ( B glyph[epsilon1] ), since for every S ,

<!-- formula-not-decoded -->

Hence, this is a feasible solution. Now, consider the objective function. Let y denote an optimal solution to LP BwK ( B glyph[epsilon1] ). We have that

<!-- formula-not-decoded -->

Now consider a simpler LP where the decision variables correspond to atoms. As before, P denotes the polytope induced by action set F .

<!-- formula-not-decoded -->

Here C = ( C ( a, j ) : a ∈ A,j ∈ d ) is the n × d matrix of expected consumption, and C † denotes its transpose. The notation glyph[precedesorcurly] means that the inequality ≤ holds for for each coordinate.

Leting OPT atoms denote the optimal value for LP ATOMS , we have:

Claim 4.3. With probability at least 1 -δ we have, OPT ALG , t ≥ OPT atoms ≥ OPT BwK ( B glyph[epsilon1] ) .

Proof. We will first prove the second inequality.

Consider the optimal solution vector x to LP BwK ( B glyph[epsilon1] ). Define S ∗ := { S : x ( S ) = 0 } .

glyph[negationslash]

We will now map this to a feasible solution to LP ATOMS and show that the objective value does not decrease. This will then complete the claim. Consider the following solution y defined as follows.

<!-- formula-not-decoded -->

We will now show that y is a feasible solution to the polytope P . From the definition of y , we can write it as y = ∑ S ∈ S ∗ x ( S ) × I [ S ] . Here, I [ S ] is a binary vector, such that it has 1 at position a if and only if atom a is present in set S . Hence, y is a point in the polytope since it can be written as convex combination of its vertices.

Now, we will show that, y also satisfies the resource consumption constraint.

<!-- formula-not-decoded -->

The last inequality is because in the optimal solution, the x value corresponding to subset S ∗ is 1 while rest all are 0. We will now show that y produces an objective value at least as large as x .

<!-- formula-not-decoded -->

Now we will prove the first inequality. We will assume the 'clean event' that µ + t ≥ µ and C -t ≤ C t for all t . Hence, the inequality holds with probability at least 1 -δ .

Consider a time t . Given an optimal solution x ∗ to LP ATOMS we will show that this is feasible to LP ALG ,t . Note that, x ∗ satisfies the constraint set x ∈ P since that is same for both LP ALG ,t and LP ATOMS . Now consider the constraint C -t ( j ) · x ≤ B glyph[epsilon1] T . Note that C -t ( a, j ) ≤ C ( a, j ) . Hence, we have that C -t ( j ) · x ∗ ≤ C ( j ) · x ∗ ≤ B glyph[epsilon1] T . The last inequality is because x ∗ is a feasible solution to LP ATOMS .

Now consider the objective function. Let y ∗ denote the optimal solution to LP ALG ,t .

<!-- formula-not-decoded -->

Hence, combining Claim 4.2 and Claim 4.3, we obtain Lemma 4.1.

## 4.2 Negative correlation and concentration bounds

Our analysis relies on several facts about negative correlation and concentration bounds. First, we argue that property (2.1) in the definition of negative correlation is preserved under a specific linear transformation:

Claim 4.4. Suppose ( X 1 , X 2 , . . . , X m ) is a family of negatively correlated random variables with support [0 , 1] . Fix numbers λ 1 , λ 2 , . . . , λ m ∈ [0 , 1] . Consider two families of random variables:

<!-- formula-not-decoded -->

Then both families satisfy property (2.1) .

Proof. Let us focus on family F + ; the proof for family F -is very similar.

Denote µ i = E [ X i ] and Y i := (1 + λ i ( X i -µ i )) / 2 and z i := (1 -λ i µ i ) / 2 for all i ∈ [ m ] . Note that Y i = λ i X i / 2 + z i and z i ≥ 0 , X i ≥ 0 . Fix a subset S ⊆ [ m ] . We have,

<!-- formula-not-decoded -->

Second, we extend Theorem 2.1 to a random process that evolves over time, and only assumes that property (2.3) holds within each round conditional on the history.

Theorem 4.5. Let Z T = { ζ t,a : a ∈ A , t ∈ [ T ] } be a family of random variables taking values in [0 , 1] . Assume random variables { ζ t,a : a ∈ A} satisfy property (2.1) given Z t -1 and have expectation 1 2 given Z t -1 , for each round t . Let Z = 1 nT ∑ a ∈A ,t ∈ [ T ] ζ t,a be the average. Then for some absolute constant c ,

<!-- formula-not-decoded -->

Proof. Weprove that family Z t satisfies property (2.3), and then invoke Theorem 2.1. Let us restate property (2.3) for the sake of completeness:

<!-- formula-not-decoded -->

Fix subset S ⊂ Z T . Partition S into subsets S t = { ζ t,a ∈ Z T ∩ S } , for each round t . Fix round τ and denote

<!-- formula-not-decoded -->

We will now prove the following statement by induction on τ :

<!-- formula-not-decoded -->

The base case is when τ = 1 . Note that G τ is just the product of elements in set ζ 1 and they are negatively correlated from the premise. Therefore we are done. Now for the inductive case of τ ≥ 2 ,

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

This completes the proof of Eq. 4.4. We obtain Eq. 4.3 for τ = T .

Third, we invoke Eq. 2.6 for rewards and resource consumptions:

Lemma 4.6. With probability at least 1 -e -Ω( α ) , we have the following:

<!-- formula-not-decoded -->

Fourth, we use a concentration bound from prior work which gets sharper when the expected sum is very small, and does not rely on independent random variables:

Theorem 4.7 (Babaioff et al. [2015]) . Let X 1 , X 2 , . . . , X m denote a set of { 0 , 1 } random variables. For each t , let α t denote the multiplier determined by random variables X 1 , X 2 , . . . , X t -1 . Let M = ∑ m t =1 M t where M t = E [ X t | X 1 , X 2 , . . . , X t -1 ] . Then for any b ≥ 1 , we have the following with probability at least 1 -m -Ω( b ) :

<!-- formula-not-decoded -->

## 4.3 Analysis of the 'clean event'

Let us set up several events, henceforth called clean events , and prove that they hold with high probability. Then the remainder of the analysis can proceed conditional on the intersection of these events. The clean events are similar to the ones in Agrawal and Devanur [2014b], but are somewhat 'stronger', essentially because our algorithm has access to per-atom feedback and our analysis can use the negative correlation property of the RRS .

In what follows, it is convenient to consider a version of SemiBwK in which the algorithm does not stop, so that we can argue about what happens w.h.p. if our algorithm runs for the full T rounds. Then we show that our algorithm does indeed run for the full T rounds w.h.p.

Recall that x t be the optimal fractional solution obtained by solving the LP in round t . Let Y t ∈ { 0 , 1 } n be the random binary vector obtained by invoking the RRS (so that the chosen action S t ∈ F corresponds to a particular realization of Y t , interpreted as a subset). Let G t := { Y t ′ : ∀ t ′ ≤ t } denote the family of RRS realizations up to round t .

## 4.3.1 'Clean event' for rewards

For brevity, for each round t let µ t = ( µ t ( a ) : a ∈ A ) be the vector of realized rewards, and let r t := µ t ( S t ) = µ t · Y t be the algorithm's reward at this round.

Lemma 4.8. Consider SemiBwK without stopping. Then with probability at least 1 -nT e -Ω( α ) :

<!-- formula-not-decoded -->

Proof. We prove the Lemma by proving the following three high-probability inequalities.

With probability at least 1 -nT e -Ω( α ) : the following holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will use the properties of RRS to prove Eq. 4.10. Proof of Eq. 4.9 is similar to Agrawal and Devanur [2014b], while proof of Eq. 4.8 follows immediately from the setup of the model. Using the parts (4.8) and (4.10) we can now find an appropriate upper bound on √ ∑ t ∈ [ T ] µ + t · x t and using this upper bound, we prove Lemma 4.8.

Proof of Eq. 4.8. Recall that r t = µ t Y t . Note that, E [ µ t Y t ] = µY t when the expectation is taken over just the independent samples of µ . By Theorem 4.7, with probability 1 -e -Ω( α ) we have:

<!-- formula-not-decoded -->

The last inequality is because Y t is a feasible solution to LP ALG .

Proof of Eq. 4.9. For this part, the arguments similar to Agrawal and Devanur [2014b] follow with some minor adaptations. For sake of completeness we describe the full proof. Note that we have,

<!-- formula-not-decoded -->

Now, using Lemma 4.6 in Appendix, we have that with probability 1 -nTe -Ω( α )

<!-- formula-not-decoded -->

Hence, we have

<!-- formula-not-decoded -->

The last inequality is from the definition of Rad function and using the Cauchy-Swartz inequality. Note that µN T = ∑ t ≤ T µ · Y t . Also, since we have with probability 1 -e -Ω( α ) , µ ( a ) ≤ µ + t ( a ) , we have,

<!-- formula-not-decoded -->

Finally note that Y t is a feasible solution to the semi-bandit polytope P . Hence, we have that

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Proof of Eq. 4.10: Recall that for each round t , the UCB vector µ + t is determined by the random variables G t -1 = { Y t ′ : ∀ t ′ &lt; t } . Further, conditional on a realization of G t -1 , the random variables { Y t ( a ) : a ∈ A} are negatively correlated from the property of RRS. Let ˜ ζ t ( a ) := µ + t ( a ) Y t ( a ) , a ∈ A . Note that we have E [ ˜ ζ t ( a ) |G t -1 ] = µ + t ( a ) x t ( a ) . Define

<!-- formula-not-decoded -->

From Claim 4.4, we have that { ζ t ( a ) : a ∈ A} conditioned on G t -1 satisfy (2.1). Further, E [ ζ t ( a ) |G t -1 ] = 1 2 . Therefore, the family { ζ t ( a ) : t ∈ [ T ] , a ∈ A} satisfies the assumptions in Theorem 4.5 and hence satisfies Eq. 4.2 for some absolute constant c . Plugging back the ˜ ζ t ( a ) 's, we obtain an upper-tail concentration bound:

<!-- formula-not-decoded -->

To obtain a corresponding concentration bound for the lower tail, we apply a similar argument to

<!-- formula-not-decoded -->

Once again from Claim 4.4, we have that { ζ ′ t ( a ) : a ∈ A} conditioned on G t -1 satisfy (2.1). The family { ζ ′ t ( a ) : t ∈ [ T ] , a ∈ A} satisfies the assumptions in Theorem 4.5 and hence satisfies Eq. 4.2. Plugging back the ˜ ζ t ( a ) 's, we obtain a lower-tail concentration bound:

<!-- formula-not-decoded -->

Combining these two we have,

<!-- formula-not-decoded -->

Hence setting η = √ α nT , we obtain Eq. 4.10 with probability at least 1 -e -Ω( α ) .

Combining Eq. (4.8) , (4.9) and (4.10) Let us denote H := √ ∑ t ∈ [ T ] µ + t · x t . Adding the three equations we get √

<!-- formula-not-decoded -->

Rearranging and solving for H , we have

<!-- formula-not-decoded -->

Plugging this back into Eq. 4.12, we get Lemma 4.8.

## 4.3.2 'Clean event' for resource consumption

We define a similar 'clean event' for consumption of each resource j . By a slight abuse of notation, for each round t let C t ( j ) = ( C t ( a, j ) : a ∈ A ) be the vector of realized consumption of resource j . Let χ t ( j ) denote algorithm's consumption for resource j at round t ( i.e., χ t ( j ) = C t ( j ) · Y t ).

Lemma 4.9. Consider SemiBwK without stopping. Then with probability at least 1 -nT e -Ω( α ) :

<!-- formula-not-decoded -->

Proof. The proof is similar to Lemma 4.8. We will split the proof into following three equations. Fix an arbitrary resource j ∈ [ d ] . With probability at least 1 -nTe -Ω( α ) the following holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the parts 4.13, 4.14 and 4.15 we can find an upper bound on √∑ t ≤ T C t ( j ) · Y t . Hence, combining Lemmas 4.13, 4.14 and 4.15 with this bound and taking an Union Bound over all the resources, we get Lemma 4.9.

Proof of Eq. 4.13. We have that { C t ( a, j ) : a ∈ A} is a set of independent random variables over a probability spacee C Ω . Note that, E C Ω C t ( a, j ) Y t ( a ) = C ( a, j ) Y t ( a ) . Hence, we can invoke Theorem 4.7 on independent random variables to get with probability 1 -nTe -Ω( α )

<!-- formula-not-decoded -->

Proof of Eq. 4.14. This is very similar to proof of 4.9 and we will skip the repetitive parts. Hence, we have with probability 1 -nTe -Ω( α )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Eq. 4.15. Recall that for each round t and each resource j , the LCB vector C -t ( j ) is determined by the random variables G t -1 = { Y t ′ : ∀ t ′ &lt; t } . Similar to the proof of Eq. 4.10, random variables { Y t ( a ) : a ∈ A} obtained from the RRS are negatively correlated given G t -1 . As before define ˜ ζ t ( a ) = C -t ( a ) Y t ( a ) , a ∈ A . We have that E [ ζ t ( a ) | G t -1 ] = C -t ( a ) x t ( a ) .

By Claim 4.4, random variables

<!-- formula-not-decoded -->

satisfy (2.1), given G t -1 . We conclude that family { ζ t ( a ) : t ∈ [ T ] , a ∈ A} satisfies the assumptions in Theorem 4.5, and therefore satisfies Eq. 4.2 for some absolute constant c . Therefore, we obtain an upper-tail concentration bound for ˜ ζ t ( a ) 's:

<!-- formula-not-decoded -->

To obtain a corresponding concentration bound for the lower tail, we apply a similar argument to

<!-- formula-not-decoded -->

Once again, invoking Claim 4.4 we have that { ζ ′ t ( a ) : a ∈ A} conditioned on G t -1 satisfy (2.1). Thus, family { ζ t ( a ) : t ∈ [ T ] , a ∈ A} satisfies the assumptions in Theorem 4.5, and therefore satisfies Eq. 4.2. We obtain:

<!-- formula-not-decoded -->

Combing the two tails we have,

<!-- formula-not-decoded -->

Once again, setting η = √ α nT , we obtain Eq. 4.15 with probability at least 1 -e -Ω( α ) .

Proof of Lemma 4.9. Denote G = √∑ t ≤ T C ( j ) · Y t . From Equation 4.13, 4.14 and 4.15, we have that G 2 -2Ω( √ αn ) G ≤ ∑ t ≤ T C -t ( j ) · x t + O ( αn ) + √ αnT . Note that ∑ t ≤ T C -t ( j ) · x t ≤ B glyph[epsilon1] . Hence, G 2 -2Ω( √ αn ) G ≤ B glyph[epsilon1] + O ( αn ) + √ αnT . Hence, re-arranging this gives us G ≤ √ B glyph[epsilon1] + O ( √ αn ) + ( αnT ) 1 / 4 . Plugging this back in Equations 4.13, 4.14 and 4.15, we get Lemma 4.9.

## 4.4 Putting it all together

Similar to Agrawal and Devanur [2014b], we will handle the hard constraint on budget, by choosing an appropriate value of glyph[epsilon1] . We then combine the above Lemma on 'rewards' clean event to compare the reward of the algorithm with that of the optimal value of LP to obtain the regret bound in Theorem 3.1. Additionally, we use the Lemma on 'consumption' clean event to argue that the algorithm doesn't exhaust the resource budget before round T . Formally, consider the following.

Recall that from Lemma 4.1, we have OPT ALG , ≥ 1 T (1 -glyph[epsilon1] ) OPT . Let us define the performance of the algorithm as ALG = ∑ t ≤ T r t . From Lemma 4.8, that with probability at least 1 -ndT e -Ω( α )

<!-- formula-not-decoded -->

Choosing glyph[epsilon1] = √ αn B + αn B + √ αnT B and using the assumption that B &gt; 3( αn + √ αnT ) , we derive Eq. 3.1. For any given δ , we set α = Ω(log( ndT δ )) to obtain a success probability of at least 1 -δ .

Now we will argue that the algorithm does not exhaust the resource budget before round T with probability at least 1 -ndT e -Ω( α ) . Note that for every resource j ∈ [ d ] ,

<!-- formula-not-decoded -->

Hence, combining this with Lemma 4.9, we have ∑ t ≤ T C t ( j ) Y t ≤ (1 -glyph[epsilon1] ) B + glyph[epsilon1]B ≤ B.

## 5 Applications and special cases

Let us discuss some notable examples of SemiBwK (which generalize some of the numerous applications listed in Badanidiyuru et al. [2013a]). Our results for these examples improve exponentially over a naive application of the BwK framework. Compared to what can be derived from [Agrawal and Devanur, 2014a, 2016], our results feature a substantially better dependence on parameters, a much better per-round running time, and apply to a wider range of parameters. However, we leave open the possibility that the regret bounds can be improved for some special cases.

Dynamic pricing. The dynamic pricing application is as follows. The algorithm has d products on sale with limited supply: for simplicity, B units of each. Following Besbes and Zeevi [2012], we allow supply constraints across products, e.g., a 'gadget' that goes into multiple products. In each round t , an agent arrives (who can buy any subset of the products), the algorithm chooses a vector of prices p t ∈ [0 , 1] d to offer the agent, and the agent chooses what to buy at these prices. For simplicity, the agent is interested in buying (or is only allowed to buy) at most one item of each product. The agent has a valuation vector over products, so that the agent buys a given product if and only if her valuation for this product is at least as high as the offered price. The entire valuation vector is drawn as an independent sample from a fixed and unknown distribution (but valuations may be correlated across products). The algorithm maximizes the total revenue from sales.

To side-step discretization issues, we assume that prices are restricted to a known finite subset S ⊂ [0 , 1] . Achieving general regret bounds without such restriction appears beyond reach of the current techniques for BwK . 8

To model it as a SemiBwK problem, the set of atoms is all price-product pairs. The combinatorial constraint is that at most one price is chosen for each product. (If an action does not specify a price for some product, the default price is used.) This is a 'partition matroid' constraint, see Appendix B. Rewards correspond to revenue from sales, and resources correspond to inventory constraints. √

For comparison, results of [Agrawal and Devanur, 2014a, 2016] apply only when B &gt; nT 3 / 4 , and yield regret bound of ˜ O ( d 3 | S | 2 √ T ) . 9 Thus, our regret bounds feature a better dependence on the number of allowed prices | S | (which can be very large) and the number of products d . Further, our regret bounds hold in a meaningful way for the much larger range of values for budget B .

We obtain regret ˜ O ( d √ dB | S | + √ T | S | ) using Corollary 3.2, whenever B &gt; ˜ Ω( n + nT ) . This is because OPT ≤ dB , since that is the maximum number of products available, and the number of atoms is n = d | S | . √

For a naive application of the BwK framework, arms correspond to every possible realization of prices for the d products. Thus, we have | S | d arms, with a corresponding exponential blow-up in regret.

Dynamic assortment. The dynamic assortment problem is similar to dynamic pricing in that the algorithm is selling d products to an agent, with a limited inventory B of each product, and is interested in maximizing the total revenue from sales. As before, agents can have arbitrary valuation vectors, drawn from a fixed but unknown distribution. However, the algorithm chooses which products to offer, whereas all prices are fixed externally. There is a large number of products to choose from, and any subset of k glyph[lessmuch] d of them can be offered in any given round.

8 Prior work on dynamic pricing with limited supply [ e.g., Besbes and Zeevi, 2009, Babaioff et al., 2015, Badanidiyuru et al., 2013a] achieves regret bounds without restricting itself to a particular finite set of prices, but only for a simple special case of (essentially) a single product.

9 We obtain this by plugging in OPT ≤ dB and n = d | S | into their regret bound. For dynamic pricing the total per-resource consumption is bounded by 1 , so we can apply their results without rescaling the consumption.

To model this as SemiBwK , atoms correspond to products, and actions correspond to subsets of at most k atoms. The combinatorial constraint forms a matroid (see Appendix B). Rewards correspond to sales, and resources correspond to products, as in dynamic pricing. Since OPT ≤ min( dB,kT ) , Corollary 3.2 yields regret ˜ O ( k √ dT ) when B &gt; Ω( T ) , and regret ˜ O ( d √ dB + √ dT ) in general.

In a naive application of BwK , arms are subsets of k products. Hence, we have O ( d k ) arms. The other parameters of the problem remain the same. This leads to regret bound ˜ O ( d √ Bd k ) , with an exponential dependence on k .

Repeated auctions. Consider a repeated auction with adjustable parameters, e.g., repeated second-price auction with reserve price that can be adjusted from one round to another. While prior work [Cesa-Bianchi et al., 2013, Badanidiyuru et al., 2013a] concerned running one repeated auction, we generalize this scenario to multiple repeated auctions with shared inventory ( e.g., the same inventory may be sold via multiple channels to different audiences).

More formally, the auctioneer is running r simultaneous repeated auctions to sell a shared inventory of d products, with limited supply B of each product ( e.g., different auctions can cater to different audiences). Each auction has a parameter which the algorithm can adjust over time. We assume that this parameter comes from a finite domain S ⊂ [0 , 1] . For simplicity, assume the auctions are synchronized with one another. As in prior work, we assume that in every round of each auction a fresh set of participants arrives, sampled independently from a fixed joint distribution, and only a minimal feedback is observed: the products sold and the combined revenue.

Following prior work [Cesa-Bianchi et al., 2013, Badanidiyuru et al., 2013a], we only assume minimal feedback: for each auction, what were the products sold and what was the combined revenue from this auction. In particular, we do not assume that the algorithm has access to participants' bids. Not using participants' bids is desirable for privacy considerations, and in order to reduce the participants' incentives to game the learning algorithm.

To model this problem as SemiBwK , atoms are all auction-parameter pairs. The combinatorial constraint is that an action must specify at most one parameter value for each auction. This corresponds to partition matroid constraints, see Appendix B. There is a 'default parameter' for each auction, in case an action does not specify the parameter. We have a resource for each product being auctioned. For simplicity, each product has supply of B . Note that OPT ≤ dB and number of atoms is n = r | S | . Hence, our main result yields regret ˜ O ( d √ r | S | B + √ r | S | T ) .

Anaive application of the BwK framework would have arms that correspond to all possible combinations of parameters, for the total of O ( | S | r ) arms. Again, we have an exponential blow-up in regret. Alternatively, one may try running r seperate instances of BwK, one for each auction, but that may result result in budgets being violated since the items are shared across the auctions and it is unclear a priori how much of each item will be sold in each auction.

One can also consider a 'flipped' version of the previous example, where the algorithm is a bidder rather than the auction maker. The bidder participates in r repeated auctions, e.g., ad auctions for different keywords. We assume a stationary environment: bidder's utility from a given bid in a given round of a given auction is an independent sample from a fixed but unknown distribution. The only limited resource here is the bidder's budget B . Bids are constrained to lie in a finite subset S .

To model this as SemiBwK , atoms correspond to the auction-bid pairs. The combinatorial constraint is that each action must specify at most one bid for each auction. (There is a 'default bid' for each auction in case an action does not specify the bid for this auction.) There is exactly one resource, which is money and the total budget is B . Note that the number of atoms is n = r | S | . Hence, our main result yields regret ˜ O (OPT √ r | S | /B + √ r | S | T ) .

Total Reward

3000

2250

1500

750

0

1000

Dynamic Assortment (n=26, B=T/2, d=1, K=2)

Total Reward

4000

3000

Figure 1: Dynamic Assortment (left) and Dynamic Pricing (right) experiments for n = 26 .

<!-- image -->

Anaive application of BwK would have arms that correspond to all possible combinations of bids, for the total of O ( | S | r ) arms; so we have an exponential blow-up in regret.

## 6 Numerical Simulations

We ran some experiments on simulated datasets in order to compare our algorithm, SemiBwK -RRS , with some prior work that can be used to solve SemiBwK :

- the primal-dual algorithm for BwK from Badanidiyuru et al. [2013a], denoted pdBwK .
- an algorithm for combinatorial semi-bandits with a matroid constraint: 'Optimistic Matroid Maximization' from Kveton et al. [2014], denoted OMM .
- the linear-contextual BwK algorithm from Agrawal and Devanur [2016], discussed in the Introduction, denoted linCBwK .

To speed up the computation in linCBwK , we used a heuristic modification suggested by the authors in a private communication. This modification did not substantially affect average rewards in our preliminary experiments. We also made a heuristic improvement to our algorithm, setting glyph[epsilon1] = 0 and α = 5 . We use the same value of α for the pdBwK algorithm as well.

Problem instances. We did not attempt to comprehensively cover the huge variety of problem instances in SemiBwK . Instead, we focus on two representative applications from Section 5.

The first experiment is on dynamic assortment. We have n products, and for each product i there is an atom i and a resource i . The (fixed) price for each product is generated as an independent sample from U [0 , 1] , a uniform distribution on [0 , 1] . At each round, we sample the buyers's valuation from U [0 , 1] , independently for each product. If the valuation for a given product is greater than the price, one item of this product is sold (and then the reward for atom i is the price, and consumption of resource i is 1 ). Else, we set reward for atom i and consumption for resource i to be 0 .

The second experiment is on dynamic pricing with two products. We have n/ 2 allowed prices, uniformly spaced in the [0 , 1] interval. Recall that atoms correspond to price-product pairs, for the total of n atoms.

Dynamic Pricing (n=26, B=T/2, d=1, K=2)

7ofomparision of various algorithms as T varies (n=6 , K=2, d=1, B=T/2)

ОРТ

SemiBwK-RRS

OMM

PD-BwK

linCBwK-sample

2000

6000

5000

₫ 4000

&amp; 3000

2000

1000

1800

3000

4000

Value of T

5000

6000

2568mparision of various algorithms as T varies (n=26, K=2, d=1, B=T/2)

OPT

SemiBwK-RRS

OMM

PD-BwK

linCBwK-sample

2000

<!-- image -->

4000

≥ 3000

2 2000

1000

1600

soomparision of various algorithms as T varies (n=26 , K=2, d=1, B=T/2)

ОРТ

SemiBwK-RRS

OMM

PD-BwK

linCBwK-sample

2000

3000

4000

Value of T

5000

6000

2560mparision of various algorithms as T varies (n=26, K=2, d=1, B=T/2)

ОРТ

SemiBwK-RRS

OMM

PD-BwK

linCBwK-sample

2000

3000

4000

Value of T

5000

6000

Figure 2: Experimental Results for Uniform matroid (left plots) and Partition matroid (right plots) on independent (upper) and correlated (lower) instances for n = 26 .

In each round t , the valuation v t,i for each product i is chosen independently from a normal distribution N ( v 0 i , 1) truncated on [0 , 1] . The mean valuation v 0 i is drawn (once for all rounds) from U [0 , 1] . If v t,i is greater than the offered price p , one item of this product is sold. Then reward for the corresponding atom ( p, i ) is the price p , and consumption of product i is 1 . If there is no sale for this product, the reward and consumption for each atom ( p, i ) is set to 0 .

The third experiment is a modification of the dynamic assortment example, in which we ensure that even non-action (e.g., no sale) exhausts resources other than time. As in dynamic assortment, we have n products, and for each product i there is an atom i and a resource i . The (fixed) price for each product is generated as an independent sample from U [0 , 1] , a uniform distribution on [0 , 1] . At each round, we sample the buyers's valuation from U [0 , 1] , independently for each product. If the valuation for a given product is greater than the price, one item of this product is sold (and then the reward for atom i is the price, and consumption of resource i is 1 ). Else, we do something different from dynamic assortment: we set reward for atom i and consumption for resource i to be the buyer's valuation.

The fourth experiment is a similar modification of the dynamic pricing example. We have n/ 2 allowed prices, uniformly spaced in the [0 , 1] interval. Recall that atoms correspond to price-product pairs, for the total of n atoms. In each round t , the valuation v t,i for each product i is chosen independently from a normal distribution N ( v 0 i , 1) truncated on [0 , 1] . The mean valuation v 0 i is drawn (once for all rounds) from U [0 , 1] . If the valuation for a given product i is greater than the offered price p , one item of this product is sold (and then reward for the corresponding atom ( p, i ) is the price, and consumption of product i is 1 ). If there is no sale for this product, we do something different from dynamic pricing. For each atom ( p, i ) , if p &lt; v t,i then the reward for atom ( p, i ) is drawn independently from U [0 , 1] and resource consumption is 1 ; else, reward

2000

500

1800

...•

....

2000

&amp; 1000

...•

7ofomparision of various algorithms as T varies (n=6 , K=2, d=1, B=T/2)

ОРТ

SemiBwK-RRS

Matroid Bandits

PD-BwK

linCBwK-sample

2000

6000

5000

₫ 4000

&amp; 3000

2000

1000

1600

206omparision of various algorithms as T varies (n=6, K=2, d=1, B=T/2)

OPT

SemiBwK-RRS

OMM

PD-BwK

linCBwK-sample

2000

5000-

4000

3000

Figure 3: Experimental Results for Uniform matroid (left plots) and partition matroid (right plots) on independent (upper) and correlated (lower) instances for n = 6 .

<!-- image -->

is 0 and consumption is . 3 . While dynamic assortment is modeled with a uniform matroid, and dynamic pricing is modeled with a partition matroid, we tried both matroids on each family.

Experimental setup and results. We choose various values of n , B and T and run our algorithms on the above two datasets assuming both a uniform matroid constraint and a partition matroid constraint. We choose n ∈ { 6 , 26 } , T ∈ { 1000 , 2000 , 3000 , 4000 , 5000 , 6000 } and B = T/ 2 . The maximum number of atoms in any action is set to K = 2 . For a given algorithm, dataset and configuration of n and T , we simulate each algorithm for 20 independent runs and take the average. We calculate the total reward obtained by the algorithm at the end of T time-steps.

Figure 1 shows results for the first two experiments. Figures 2 and 3 show the results on the third and fourth experiments. Our algorithm achieves the best regret among the competitors. As a benchmark, we included the performance of the fractional allocation in LP OPT , denoted OPT .

Additional experiment. linCBwK and pdBwK have running times proportional to the number of actions. We ran an additional experiment which compared per-step running times. We first calculate the average running time for every 10 steps and take the median of 50 such runs. For both Uniform matroid and Partition matroid, we run the faster RRS due to Gandhi et al. [2006]. See Figure 4 for results.

Details of heuristic implementation of linCBwK . We now briefly describe the heuristic we use to simulate the linCBwK algorithm. Note that even though the per-time-step running time of linCBwK is reasonable, it takes a significant time when we want to perform simulations for many time-steps. The time-consuming step in the linCBwK algorithm is the solution to a convex program for computing the optimistic estimates (namely ˜ µ t and ˜ W t ). Hence, the heuristic gives a faster way to obtain this estimate. We sample multiple

1500

500

1800

...•

606omparision of various algorithms as T varies (n=6, K=2, d=1, B=T/2)

ОРТ

SemiBwK-RRS

OMM

PD-BwK

linCBwK-sample

log(micro seconds per step)

13

10

7

3

0

10

Comparison of Running Times

8

20

<!-- image -->

40

Value of n

Figure 4: Variation of per-step running times as n increases for the various algorithms.

times from a multi-variate Gaussian with mean ˆ µ and covariance M t (to obtain estimate ˜ µ t ) and with mean ˆ w tj and covariance M t (to obtain estimate ˜ w tj for each resource j ). We use these samples to compute the objective to choose the action at time-step t . For each sample, we compute the best action based on the objective in linCBwK . We finally choose the action that occurs majority number of times in these samples. The number of samples we choose is set to 30.

Language Details of algorithms. All algorithms except linCBwK were implemented in Python. The linCBwK algorithm was implemented in MATLAB. This difference is crucial when we compare running times since language construct can speed-up or slow down algorithms in practice. However, it is known that 10 for matrix operations commonly encountered in engineering and statistics, MATLAB implementations runs several orders of magnitude faster than the corresponding python implementation. Since linCBwK is the slowest of the four algorithms, our comparison of running times across languages is justified.

Acknowledgements. Karthik would like to thank Aravind Srinivasan for some useful discussions.

## References

- A. Agarwal, S. Bird, M. Cozowicz, M. Dudik, J. Langford, L. Li, L. Hoang, D. Melamed, S. Sen, R. Schapire, and A. Slivkins. Multiworld testing: A system for experimentation, learning, and decisionmaking, 2016. A white paper, available at https://github.com/Microsoft/mwt-ds/raw/ master/images/MWT-WhitePaper.pdf .
- S. Agrawal and N. R. Devanur. Bandits with concave rewards and convex knapsacks. In 15th ACM Conf. on Economics and Computation (ACM EC) , 2014a.
- S. Agrawal and N. R. Devanur. Bandits with concave rewards and convex knapsacks. In Proceedings of the fifteenth ACM conference on Economics and computation , pages 989-1006. ACM, 2014b.
- S. Agrawal and N. R. Devanur. Linear contextual bandits with knapsacks. In 29th Advances in Neural Information Processing Systems (NIPS) , 2016.

10 https://www.mathworks.com/products/matlab/matlab-vs-python.html

30

50

75

- S. Agrawal, N. R. Devanur, and L. Li. An efficient algorithm for contextual bandits with knapsacks, and an extension to concave objectives. In 29th Conf. on Learning Theory (COLT) , 2016.
- V. Anantharam, P. Varaiya, and J. Walrand. Asymptotically efficient allocation rules for the multiarmed bandit problem with multiple plays-part i: Iid rewards. IEEE Transactions on Automatic Control , 32(11): 968-976, 1987.
- A. Asadpour, M. X. Goemans, A. Madry, S. O. Gharan, and A. Saberi. An o (log n/log log n)-approximation algorithm for the asymmetric traveling salesman problem. In SODA , volume 10, pages 379-389. SIAM, 2010.
- P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning , 47(2-3):235-256, 2002.
- M. Babaioff, S. Dughmi, R. D. Kleinberg, and A. Slivkins. Dynamic pricing with limited supply. ACM Trans. on Economics and Computation , 3(1):4, 2015. Special issue for 13th ACM EC , 2012.
- A. Badanidiyuru, R. Kleinberg, and Y. Singer. Learning on a budget: posted price mechanisms for online procurement. In 13th ACM Conf. on Electronic Commerce (EC) , pages 128-145, 2012.
- A. Badanidiyuru, R. Kleinberg, and A. Slivkins. Bandits with knapsacks. In 54th IEEE Symp. on Foundations of Computer Science (FOCS) , 2013a.
- A. Badanidiyuru, R. Kleinberg, and A. Slivkins. Bandits with knapsacks. A technical report on arxiv.org ., May 2013b.
- A. Badanidiyuru, J. Langford, and A. Slivkins. Resourceful contextual bandits. In 27th Conf. on Learning Theory (COLT) , 2014.
- O. Besbes and A. Zeevi. Dynamic pricing without knowing the demand function: Risk bounds and nearoptimal algorithms. Operations Research , 57:1407-1420, 2009.
- O. Besbes and A. J. Zeevi. Blind network revenue management. Operations Research , 60(6):1537-1550, 2012.
- S. Bubeck and N. Cesa-Bianchi. Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems. Foundations and Trends in Machine Learning , 5(1), 2012.
- N. Cesa-Bianchi, C. Gentile, and Y. Mansour. Regret minimization for reserve prices in second-price auctions. In ACM-SIAM Symp. on Discrete Algorithms (SODA) , 2013.
- C. Chekuri, J. Vondrak, and R. Zenklusen. Dependent randomized rounding via exchange properties of combinatorial structures. In Foundations of Computer Science (FOCS), 2010 51st Annual IEEE Symposium on , pages 575-584. IEEE, 2010.
- C. Chekuri, J. Vondr´ ak, and R. Zenklusen. Multi-budgeted matchings and matroid intersection via dependent rounding. In Proceedings of the twenty-second annual ACM-SIAM symposium on Discrete Algorithms , pages 1080-1097. SIAM, 2011.
- W. Chen, Y. Wang, and Y. Yuan. Combinatorial multi-armed bandit: General framework and applications. In S. Dasgupta and D. Mcallester, editors, Proceedings of the 30th International Conference on Machine Learning (ICML-13) , volume 28, pages 151-159. JMLR Workshop and Conference Proceedings, 2013.

- R. Combes, C. Jiang, and R. Srikant. Bandits with budgets: Regret lower bounds and optimal algorithms. ACM SIGMETRICS Performance Evaluation Review , 43(1):245-257, 2015a.
- R. Combes, M. S. T. M. Shahi, A. Proutiere, et al. Combinatorial bandits revisited. In Advances in Neural Information Processing Systems , pages 2116-2124, 2015b.
- Y. Gai, B. Krishnamachari, and R. Jain. Learning multiuser channel allocations in cognitive radio networks: A combinatorial multi-armed bandit formulation. In New Frontiers in Dynamic Spectrum, 2010 IEEE Symposium on , pages 1-9. IEEE, 2010.
- Y. Gai, B. Krishnamachari, and R. Jain. Combinatorial network optimization with unknown variables: Multi-armed bandits with linear rewards and individual observations, Oct. 2012.
- R. Gandhi, S. Khuller, S. Parthasarathy, and A. Srinivasan. Dependent rounding and its applications to approximation algorithms. Journal of the ACM (JACM) , 53(3):324-360, 2006.
- J. Gittins, K. Glazebrook, and R. Weber. Multi-Armed Bandit Allocation Indices . John Wiley &amp; Sons, 2011.
- S. Guha and K. Munagala. Multi-armed Bandits with Metric Switching Costs. In 36th Intl. Colloquium on Automata, Languages and Programming (ICALP) , pages 496-507, 2007.
- A. Gupta, R. Krishnaswamy, M. Molinaro, and R. Ravi. Approximation algorithms for correlated knapsacks and non-martingale bandits. In 52nd IEEE Symp. on Foundations of Computer Science (FOCS) , pages 827-836, 2011.
- A. Gy¨ orgy, T. Linder, G. Lugosi, and G. Ottucs´ ak. The on-line shortest path problem under partial monitoring. J. of Machine Learning Research (JMLR) , 8:2369-2403, 2007.
- R. Impagliazzo and V. Kabanets. Constructive proofs of concentration bounds. In Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques , pages 617-631. Springer, 2010.
- S. Katariya, B. Kveton, C. Szepesv´ ari, and Z. Wen. DCM bandits: Learning to rank with multiple clicks. In Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016 , pages 1215-1224, 2016.
- R. Kleinberg, A. Slivkins, and E. Upfal. Bandits and experts in metric spaces. Working paper, published at http://arxiv.org/abs/1312.1277 , 2015. Merged and revised version of conference papers in ACM STOC 2008 and ACM-SIAM SODA 2010 .
- A. Krishnamurthy, A. Agarwal, and M. Dud´ ık. Contextual semibandits via supervised learning oracles. In 29th Advances in Neural Information Processing Systems (NIPS) , 2016.
- B. Kveton, Z. Wen, A. Ashkan, H. Eydgahi, and B. Eriksson. Matroid bandits: Fast combinatorial optimization with learning. In N. L. Zhang and J. Tian, editors, UAI , pages 420-429. AUAI Press, 2014.
- B. Kveton, C. Szepesvari, Z. Wen, and A. Ashkan. Cascading bandits: Learning to rank in the cascade model. In D. Blei and F. Bach, editors, Proceedings of the 32nd International Conference on Machine Learning (ICML-15) , pages 767-776. JMLR Workshop and Conference Proceedings, 2015a.
- B. Kveton, Z. Wen, A. Ashkan, and C. Szepesvri. Tight regret bounds for stochastic combinatorial semibandits. In G. Lebanon and S. V. N. Vishwanathan, editors, AISTATS , JMLR Workshop and Conference Proceedings. JMLR.org, 2015b.

- T. L. Lai and H. Robbins. Asymptotically efficient Adaptive Allocation Rules. Advances in Applied Mathematics , 6:4-22, 1985.
- C. H. Papadimitriou and K. Steiglitz. Combinatorial optimization: algorithms and complexity . Courier Corporation, 1982.
- P. Raghavan and C. D. Tompson. Randomized rounding: a technique for provably good algorithms and algorithmic proofs. Combinatorica , 7(4):365-374, 1987.
- A. Rakhlin and K. Sridharan. BISTRO: an efficient relaxation-based method for contextual bandits. In 33nd Intl. Conf. on Machine Learning (ICML) , 2016.
- H. Robbins. Some Aspects of the Sequential Design of Experiments. Bull. Amer. Math. Soc. , 58:527-535, 1952.
- A. Schrijver. Combinatorial optimization: polyhedra and efficiency , volume 24. Springer Science &amp; Business Media, 2002.
- A. Singla and A. Krause. Truthful incentives in crowdsourcing tasks using regret minimization mechanisms. In 22nd Intl. World Wide Web Conf. (WWW) , pages 1167-1178, 2013.
- A. Slivkins. Dynamic ad allocation: Bandits with budgets. A technical report on arxiv.org/abs/1306.0155 , June 2013.
- A. Slivkins and J. W. Vaughan. Online decision making in crowdsourcing markets: Theoretical challenges. SIGecom Exchanges , 12(2), December 2013.
- V. Syrgkanis, A. Krishnamurthy, and R. E. Schapire. Efficient algorithms for adversarial contextual learning. In 33nd Intl. Conf. on Machine Learning (ICML) , 2016.
- W. R. Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika , 25(3-4):285294, 1933.
- L. Tran-Thanh, A. Chapman, E. M. de Cote, A. Rogers, and N. R. Jennings. glyph[epsilon1] -first policies for budget-limited multi-armed bandits. In 24th AAAI Conference on Artificial Intelligence (AAAI) , pages 1211-1216, 2010.
- L. Tran-Thanh, A. Chapman, A. Rogers, and N. R. Jennings. Knapsack based optimal policies for budgetlimited multi-armed bandits. In 26th AAAI Conference on Artificial Intelligence (AAAI) , pages 11341140, 2012.
- Z. Wang, S. Deng, and Y. Ye. Close the gaps: A learning-while-doing algorithm for single-product revenue management problems. Operations Research , 62(2):318-331, 2014.
- Z. Wen, B. Kveton, and A. Ashkan. Efficient learning in large-scale combinatorial semi-bandits. In F. R. Bach and D. M. Blei, editors, ICML , JMLR Workshop and Conference Proceedings, pages 1113-1122. JMLR.org, 2015.
- D. P. Williamson and D. B. Shmoys. The design of approximation algorithms . Cambridge university press, 2011.
- Y. Xia, W. Ding, X.-D. Zhang, N. Yu, and T. Qin. Budgeted bandit problems with continuous random costs. In Asian Conference on Machine Learning , pages 317-332, 2016a.

- Y. Xia, T. Qin, W. Ma, N. Yu, and T.-Y. Liu. Budgeted multi-armed bandits with multiple plays. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence , IJCAI'16, pages 22102216. AAAI Press, 2016b. ISBN 978-1-57735-770-4. URL http://dl.acm.org/citation. cfm?id=3060832.3060930 .
- S. Zong, H. Ni, K. Sung, N. R. Ke, Z. Wen, and B. Kveton. Cascading bandits for large-scale recommendation problems. 2016.

## A Proof of Theorem in Preliminaries

Theorem 2.1 follows easily from Theorem 3.3 in Impagliazzo and Kabanets [2010].

Theorem (Theorem 2.1) . Let X = ( X 1 , X 2 , . . . , X m ) denote a collection of random variables which take values in [0 , 1] , and let X := 1 m ∑ m i =1 X i be their average. Suppose X satisfies (2.3) , i.e., E [ ∏ i ∈ S X i ] ≤ ( 1 2 ) | S | for every S ⊆ [ m ] . Then for some absolute constant c ,

<!-- formula-not-decoded -->

Proof. Fix η &gt; 0 . From Theorem 3.3 in Impagliazzo and Kabanets [2010], we have that

<!-- formula-not-decoded -->

where D KL ( · ‖ · ) denotes KL-divergence, so that

<!-- formula-not-decoded -->

From Pinsker's inequality we have, D KL (1 / 2 + η ‖ 1 / 2) ≥ 2 η 2 , which implies (A.1).

## B Matroid constraints

To make this paper more self-contained, we provide more background on matroid constraints and special cases thereof.

Recall that in SemiBwK , we have a finite ground set whose elements are called 'atoms', and a family F of 'feasible subsets' of the ground set which are the actions. To be consistent with the literature on matroids, the ground set will be denoted E . Family F of subsets of E is called a matroid if it satisfies the following properties:

- Empty set : The empty set φ is present in F
- Hereditary property : For two subsets X,Y ⊆ E such that X ⊆ Y , we have that

<!-- formula-not-decoded -->

- Exchange property : For X,Y ∈ F and | X | &gt; | Y | , we have that

<!-- formula-not-decoded -->

Matroids are linearizable , i.e., the convex hull of F forms a polytope in R E . (Here subsets of F are intepreted as binary vectors in R E .) In other words, there exists a set of linear constraints whose set of feasible integral solutions is F . In fact, the convex hull of F , a.k.a. the matroid polytope , can be represented via the following linear system:

<!-- formula-not-decoded -->

Here x ( S ) := ∑ e ∈ S x e , and rank ( S ) = max {| Y | : Y ⊆ S, Y ∈ F} is the 'rank function' for F .

F is indeed the set of all feasible integral solutions of the above system. This is a standard fact in combinatorial optimization, e.g., see Theorem 40.2 and its corollaries in Schrijver [2002].

We will now describe some well-studied special cases of matroids. That they indeed are special cases of matroids is well-known, we will not present the corresponding proofs here.

In all LPs presented below, we have variables x e for each arom e ∈ E , and we use shorthand x ( S ) := ∑ e ∈ S x e for S ⊂ E .

Cardinality constraints. Cardinality constraint is defined as follows: a subset S of atoms belongs to F if and only if | S | ≤ K for some fixed K . This is perhaps the simplest constraint that our results are applicable to. In the context of SemiBwK , each action selects at most K atoms.

The corresponding induced polytope is as follows:

<!-- formula-not-decoded -->

Partition matroid constraints. A generalization of cardinality constraints, called partition matroid constraints, is defined as follows. Suppose we have a collection B 1 , . . . , B k of disjoint subsets of E , and numbers d 1 , . . . , d k . A subset S of atoms belongs to F if and only if | S ∩ B i | ≤ d i for every i . Partition matroid constraints appear in several applications of SemiBwK such as dynamic pricing, adjusting repeated auctions, and repeated bidding. In these applications, each action selects one price/bid for each offered product. Also, partition matroid constraints can model clusters of mutually exclusive products in dynamic assortment application.

The induced polytope is as follows:

<!-- formula-not-decoded -->

Spanning tree constraints. Spanning tree constraints describe spanning trees in a given undirected graph G = ( V, E ) , where the atoms correspond to edges in the graph. A spanning tree in G is a subset E ′ ⊂ E of edges such that ( V, E ′ ) is a tree. Action set F consists of all spanning trees of G .

The induced polytope is as follows:

<!-- formula-not-decoded -->

Here, E S denotes the edge set in subgraph induced by node set S ⊂ V .