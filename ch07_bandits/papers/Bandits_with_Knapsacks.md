## Bandits with Knapsacks ∗

Ashwinkumar Badanidiyuru †

Robert Kleinberg ‡

May 2013

This revision: September 2017

## Abstract

Multi-armed bandit problems are the predominant theoretical model of exploration-exploitationtradeoffs in learning, and they have countless applications ranging from medical trials, to communication networks, to Web search and advertising. In many of these application domains the learner may be constrained by one or more supply (or budget) limits, in addition to the customary limitation on the time horizon. The literature lacks a general model encompassing these sorts of problems. We introduce such a model, called bandits with knapsacks , that combines bandit learning with aspects of stochastic integer programming. In particular, a bandit algorithm needs to solve a stochastic version of the well-known knapsack problem , which is concerned with packing items into a limited-size knapsack. A distinctive feature of our problem, in comparison to the existing regret-minimization literature, is that the optimal policy for a given latent distribution may significantly outperform the policy that plays the optimal fixed arm. Consequently, achieving sublinear regret in the bandits-with-knapsacks problem is significantly more challenging than in conventional bandit problems.

We present two algorithms whose reward is close to the information-theoretic optimum: one is based on a novel 'balanced exploration' paradigm, while the other is a primal-dual algorithm that uses multiplicative updates. Further, we prove that the regret achieved by both algorithms is optimal up to polylogarithmic factors. We illustrate the generality of the problem by presenting applications in a number of different domains, including electronic commerce, routing, and scheduling. As one example of a concrete application, we consider the problem of dynamic posted pricing with limited supply and obtain the first algorithm whose regret, with respect to the optimal dynamic policy, is sublinear in the supply.

∗ An extended abstract of this paper [Badanidiyuru et al., 2013] was published in IEEE FOCS 2013 .

This paper has undergone several rounds of revision since the original 'full version' has been published on arxiv.org in May 2013. Presentation has been thoroughly revised throughout, from low-level edits to technical intuition and details to high-level restructuring of the paper. Some of the results have been improved: a stronger regret bound in one of the main results (Theorem 4.1), and a more general example of preadjusted discretization for dynamic pricing (Theorem 7.7). Introduction discusses a significant amount of follow-up work, and is up-to-date regarding open questions.

Parts of this research have been done while A. Badanidiyuru was a research intern at Microsoft Research and a graduate student at Cornell University, and while R. Kleinberg was a Consulting Researcher at Microsoft Research.

A. Badanidiyuru was partially supported by NSF grant IIS-0905467. R. Kleinberg was partially supported by NSF grants CCF0643934, IIS-0905467 and AF-0910940, a Microsoft Research New Faculty Fellowship, and a Google Research Grant.

† Google Research, Mountain View CA, USA. Email: ashwinkumarbv@gmail.com .

‡ Department of Computer Science, Cornell University, Ithaca NY, USA. Email: rdk@cs.cornell.edu .

§ Microsoft Research, New York NY, USA. Email: slivkins@microsoft.com .

Aleksandrs Slivkins §

## Contents

| 1 Introduction   | 1 Introduction                                                | 1 Introduction                                                       | 3     |
|------------------|---------------------------------------------------------------|----------------------------------------------------------------------|-------|
|                  | 1.1                                                           | Our model: bandits with knapsacks ( BwK ) . . . . . . .              | 3     |
|                  | 1.2                                                           | Main results . . . . . . . . . . . . . . . . . . . . . . .           | 5     |
|                  | 1.3                                                           | Challenges and techniques . . . . . . . . . . . . . . .              | 6     |
|                  | 1.4                                                           | Follow-up work and open questions . . . . . . . . . .                | 7     |
|                  | 1.5                                                           | Related work . . . . . . . . . . . . . . . . . . . . . .             | 9     |
| 2                | Preliminaries                                                 | Preliminaries                                                        | 10    |
| 3                | LP relaxation for policy value                                | LP relaxation for policy value                                       | 12    |
| 4                | Algorithm BalancedExploration                                 | Algorithm BalancedExploration                                        | 14    |
|                  | 4.1 Deterministic properties of BalancedExploration           | .                                                                    | 15    |
|                  |                                                               | High-probability events . . . . . . . . . . . . . . . . .            | 16    |
|                  | 4.2 4.3                                                       | Clean execution analysis . . . . . . . . . . . . . . . .             | 17    |
| 5                |                                                               | Algorithm PrimalDualBwK                                              | 22    |
|                  | 5.1 Warm-up: The deterministic case . . . . . . . . . . . . . | 5.1 Warm-up: The deterministic case . . . . . . . . . . . . .        | 24    |
|                  | 5.2                                                           | Analysis modulo error terms . . . . . . . . . . . . . .              | 26    |
|                  | 5.3                                                           | Error analysis . . . . . . . . . . . . . . . . . . . . . .           | 28    |
| 6                |                                                               | Bound                                                                | 30    |
|                  | Lower 6.1                                                     | The new lower-bounding example: proof of Claim 6.2(b)                | 31    |
|                  | 6.2                                                           | Background on KL-divergence (for the proof of Lemma                  | 33    |
|                  | 6.3                                                           | The KL-divergence argument: proof of Lemma 6.6 . .                   | 34    |
| 7                |                                                               | with preadjusted discretization                                      | 36    |
|                  | BwK                                                           | BwK                                                                  | 36    |
|                  | 7.1                                                           | Preadjusted discretization as a general technique . . . .            | 38    |
|                  | 7.2                                                           | A general bound on discretization error . . . . . . . .              | 39    |
|                  | 7.3                                                           | Preadjusted discretization for dynamic pricing . . . . . procurement |       |
|                  | 7.4                                                           | Preadjusted discretization for dynamic . .                           | 40    |
| 8                | Applications and corollaries                                  | Applications and corollaries                                         | 42    |
|                  | 8.1 Dynamic pricing with limited supply                       | . . . . . . . . . .                                                  | 43    |
|                  |                                                               | Dynamic procurement and crowdsourcing markets                        |       |
|                  | 8.2 8.3                                                       | . . Other applications to Electronic Markets . . . . . . . .         | 44 46 |
|                  | 8.4                                                           | Application to network routing and scheduling . . . . .              | 46    |
|                  | Bibliography                                                  | Bibliography                                                         | 47    |
| A                | The optimal dynamic policy beats the best fixed arm           | The optimal dynamic policy beats the best fixed arm                  | 51    |
| B                | BalancedExploration beats PrimalDualBwK sometimes             | BalancedExploration beats PrimalDualBwK sometimes                    | 52    |
| C                | Analysis of the Hedge Algorithm                               | Analysis of the Hedge Algorithm                                      | 52    |
|                  | D Facts for the proof of the lower bound                      | D Facts for the proof of the lower bound                             | 54    |

## 1 Introduction

For more than fifty years, the multi-armed bandit problem (henceforth, MAB ) has been the predominant theoretical model for sequential decision problems that embody the tension between exploration and exploitation, 'the conflict between taking actions which yield immediate reward and taking actions whose benefit ( e.g., acquiring information or preparing the ground) will come only later,' to quote Whittle's apt summary [Whittle, 1980]. Owing to the universal nature of this conflict, it is not surprising that MAB algorithms have found diverse applications ranging from medical trials, to communication networks, to Web search and advertising.

A common feature in many of these application domains is the presence of one or more limited-supply resources that are consumed during the decision process. For example, scientists experimenting with alternative medical treatments may be limited not only by the number of patients participating in the study but also by the cost of materials used in the treatments. A website experimenting with displaying advertisements is constrained not only by the number of users who visit the site but by the advertisers' budgets. A retailer engaging in price experimentation faces inventory limits along with a limited number of consumers. The literature on MAB problems lacks a general model that encompasses these sorts of decision problems with supply limits. Our paper contributes such a model, called bandits with knapsacks (henceforth BwK ), in which a bandit algorithm needs to solve a stochastic, multi-dimensional version of the well-known knapsack problem . We present algorithms whose regret (normalized by the payoff of the optimal policy) converges to zero as the resource budget and the optimal payoff tend to infinity. In fact, we prove that this convergence takes place at the information-theoretically optimal rate.

## 1.1 Our model: bandits with knapsacks ( BwK )

Problem definition. A learner has a fixed set of potential actions, a.k.a. arms , denoted by X and called action space . (In our main results, X will be finite, but we will also consider extensions with an infinite set of arms, see Section 8 and Section 7.) There are d resources being consumed by the learner. Over a sequence of time steps, the learner chooses an arm and observes two things: a reward and a resource consumption vector . Rewards are scalar-valued, whereas resource consumption vectors are d -dimensional: the i -th component represents consumption of resource i . For each resource i there is a pre-specified budget B i representing the maximum amount that may be consumed, in total. The process stops at the first time τ when the total consumption of some resource exceeds its budget. The objective is to maximize the total reward received before time τ .

We assume that the environment does not change over time. Formally, the observations for a fixed arm x in each time step ( i.e., the reward and resource consumption vector) are independent samples from a fixed joint distribution on [0 , 1] × [0 , 1] d , called the latent distribution for arm x .

There is a known, finite time horizon T . We model it as one of the resources, one unit of which is deterministically consumed in each decision period, and the budget is T .

Notable examples. The conventional MAB problem, with a finite time horizon T , naturally fits into this framework. A more interesting example is the dynamic pricing problem faced by a retailer selling B items to a population of T unit-demand consumers who arrive sequentially. Modeling this as a BwK problem, rounds correspond to consumers, and arms correspond to the possible prices which may be offered to a consumer. Reward is the revenue from a sale, if any. Resource consumption vectors express the number of items sold and consumers seen, respectively. Thus, if a price p is offered and accepted, the reward is p and the resource consumption is [ 1 1 ] . If the offer is declined, the reward is 0 and the resource consumption is [ 0 1 ] .

Adual problem of dynamic pricing is dynamic procurement , where the algorithm is dynamically buying rather than dynamically selling. The reward refers to the number of bought items, and the budget constraint

B now applies to the amount spent (which is why the two problems are not merely identical up to sign reversal). If a price p is offered and accepted, the reward is 1 and the resource consumption is [ p 1 ] . If the offer is declined, the reward is 0 and the resource consumption is [ 0 1 ] . This problem is also relevant to the domain of crowdsourcing: the items bought then correspond to microtasks ordered on a crowdsourcing platform such as Amazon Mechanical Turk.

Another simple example concerns dynamic ad allocation for pay-per-click ads with unknown click probabilities. There is one advertiser with several ads and budget B across all ads, and T users to show the ads to. The ad platform allocates one ad to a new user in each round. Whenever a given ad x is chosen and clicked on, the advertiser pays a known amount π x . To model this as a BwK problem, arms correspond to ads, rewards are the advertiser's payments, and resource consumption refers to the amount spent by the advertiser and the number of users seen. Thus, if ad x is chosen and clicked, the reward is π x and the resource consumption is [ π x 1 ] ; otherwise, the reward is 0 and the resource consumption is [ 0 1 ] .

All three examples can be easily generalized to multiple resource constraints: resp., to selling multiple products, procuring different types of goods, and allocating ads from multiple advertisers.

Benchmark and regret. The performance of an algorithm will be measured by its regret : the worst case, over all possible tuples of latent distributions, of the difference between OPT and the algorithm's expected total reward. Here OPT is the expected total reward of the benchmark: an optimal dynamic policy , an algorithm that maximizes expected total reward given foreknowledge of the latent distributions.

In a conventional MAB problem, the optimal dynamic policy is to play a fixed arm, namely the one with the highest expected reward. In the BwK problem, the optimal dynamic policy is more complex, as the choice of an arm in a given round depends on the remaining supply of each resource. In fact, we doubt there is a polynomial-time algorithm to compute the optimal dynamic policy given the latent distributions; similar problems in optimal control have long been known to be PSPACE-hard [Papadimitriou and Tsitsiklis, 1999].

It is easy to see that the optimal dynamic policy may significantly out-perform the best fixed arm. To take a simple example, consider a problem instance with d resources and d arms such that pulling arm i deterministically produces a reward of 1 , consumes one unit of resource i , and does not consume any other resources. We are given an initial endowment of B units of each resource. Any policy that plays a fixed arm i in each round is limited to a total reward of B before running out of its budget of resource i . Whereas an algorithm that alternates arms in a round-robin fashion achieves reward dB : d times larger. Similar, but somewhat more involved examples can be found for application domains of interest, see Appendix A. Interestingly, in all these examples it suffices to consider a time-invariant mixture of arms, i.e., a policy that samples in each period from a fixed probability distribution over arms regardless of the remaining resource supplies. In particular, in the simple example above it suffices to consider a uniform distribution.

Alternative definitions. More generally we could model the budget constraints as a downward-closed polytope P ⊂ R d + such that the process stops when the sum of resource consumption vectors is no longer in P . However, our assumption that P is a box constraint is virtually without loss of generality. If P is instead specified by a system of inequalities { Ax /precedesequal b } , we can redefine the resource consumption vectors to be Ax instead of x and then the budget constraint is the box constraint defined by the vector b . The only potential downside of this transformation is that it increases the dimension of the resource vector space, when the constraint matrix A has more rows than columns. However, one of our algorithms has regret depending only logarithmically on d , so this increase typically has only a mild effect on regret.

Our stopping condition halts the algorithm as soon as any budget is exceeded. Alternatively, we could restrict the algorithm to actions that cannot possibly violate any constraint if chosen in the current round, and stop if there is no such action. This alternative is essentially equivalent to the original version: each budget constraint changes by at most one, which does not affect our regret bounds in any significant way.

## 1.2 Main results

We seek regret bounds that are sublinear in OPT , whereas in analyzing MAB algorithms one typically expresses regret bounds as a sublinear function of the time horizon T . This is because a regret guarantee of the form o ( T ) may be unacceptably weak for the BwK problem because supply limits prevent the optimal dynamic policy from achieving a reward close to T . An illustrative example is the dynamic pricing problem with supply B /lessmuch T : the seller can only sell B items, each at a price of at most 1, so bounding the regret by any number greater than B is worthless. To achieve sublinear regret, the algorithm must be able to explore each arm a significant number of times without exhausting its resource budgets. Accordingly, we parameterize our regret bound by B = min i B i , the smallest budget constraint.

Algorithms. We present an algorithm, called PrimalDualBwK , whose regret is sublinear in OPT as both OPT and B tend to infinity. More precisely, denoting the number of arms by m , our algorithm's regret is

<!-- formula-not-decoded -->

We also present another algorithm, called BalancedExploration , whose regret bound is the same up to logarithmic factors for d = O (1) . The regret bounds for the two algorithms are incomparable: while PrimalDualBwK achieves a better dependence on d , BalancedExploration performs better in some special cases, see Appendix B for a simple example. While PrimalDualBwK is very computationally efficient, the specification of BalancedExploration involves a mathematically well-defined optimization step for which we do not provide a specific implementation, see Remark 4.2 fur further discussion.

where the ˜ O () notation hides logarithmic factors. Note that without resource constraints, i.e., setting B = T , we recover regret ˜ O ( √ m OPT ) , which is optimal up to log factors [Auer et al., 2002b]. In fact, we prove a slightly stronger regret bound which has an optimal scaling property: if all budget constraints, including the time horizon, are increased by the factor of α , then the regret bound scales as √ α . 1 The algorithm is computationally efficient, in a strong sense: with machine word size of log T bits or more, the per-round running time is O ( md ) . Moreover, if each arm j consumes only d j resources that are known in advance, then the per-round running time is O ( m + d + ∑ j d j ) .

Lower bound. We provide a matching lower bound: we prove that the regret bound (1) is optimal up to polylogarithmic factors; moreover, this holds for any given tuple of parameters . Specifically, we show that for any given tuple ( m,B, OPT ) , any algorithm for BwK must incur regret

<!-- formula-not-decoded -->

in the worst-case over all instances of BwK with these ( m,B, OPT ) . We also show that this dependence on the smallest budget constraint is inevitable in the worst case.

Applications and special cases. We derive corollaries for the three examples outlined in Section 1.1:

- We obtain regret ˜ O ( B 2 / 3 ) for the basic version of dynamic pricing. This is optimal for each ( B,T ) pair [Babaioff et al., 2015]. Prior work [Babaioff et al., 2015, Wang et al., 2014] achieved ˜ O ( B 2 / 3 ) regret w.r.t. the best fixed price, and ˜ O ( √ B ) regret assuming 'regularity'. 2 The former result is much weaker than ours, see Appendix A for a simple example, and the latter result is incomparable.
- Weobtain regret ˜ O ( T/B 1 / 4 ) for the basic version of dynamic procurement. Prior work [Badanidiyuru et al., 2012] achieves a constant-factor approximation to the optimum with a prohibitively large constant (at least in the tens of thousands), so our result is a big improvement unless OPT /greatermuch T/B 1 / 4 .

1 The square-root scaling is optimal even for the basic MAB problem, as proved in Auer et al. [2002b].

2 'Regularity' is a standard (but limiting) condition which states that the mapping from prices to expected rewards is concave.

- We obtain regret ˜ O ( √ B ) for the basic version of dynamic ad allocation. This is optimal when B = T ( i.e., when the budget constraint is void), by the basic √ T lower bound for MAB.

Our model admits numerous generalizations of these three examples, as well as applications to several other domains. To emphasize the generality of our contributions, we systematically discuss applications and corollaries in Section 8. Pointers to prior work on special cases of BwK can be found in Section 1.5.

## 1.3 Challenges and techniques

Challenges. As with all MAB problems, a central issue in BwK is the tradeoff between exploration and exploitation. A na¨ ıve way to resolve this tradeoff is to separate exploration and exploitation: before the algorithm starts, the rounds are partitioned into 'exploration rounds' and 'exploitation rounds', so that the arms chosen in the former does not depend on the feedback, and the feedback from the latter is discarded. 3 For example, an algorithm may pick an arm uniformly at random for a pre-defined number of rounds, then choose the best arm given the observations so far, and stick to this arm from then on. However, it tends to be much more efficient to combine exploration and exploitation by adapting the exploration schedule to observations. Typically in such algorithms all but the first few rounds serve both exploration and exploitation. Thus, one immediate challenge is to implement this approach in the context of BwK .

The BwK problem is significantly more difficult to solve than conventional MAB problems for the following three reasons. First, in order to estimate the performance of a given time-invariant policy, one needs to estimate the expected total reward of this policy, rather than the per-round expected reward (because the latter does not account for resource constraints). Second, since exploration consumes resources other than time, the negative effect of exploration is not limited to the rounds in which it is performed. Since resource consumption is stochastic, this negative effect is not known in advance, and can only be estimated over time. Finally, and perhaps most importantly, the optimal dynamic policy can significantly outperform the best fixed arm, as mentioned above. In order to compete with the optimal dynamic policy, an algorithm needs, essentially, to search over mixtures of arms rather than over arms themselves, which is a much larger search space. In particular, our algorithms improve over the performance of the best fixed arm, whereas algorithms for explore-exploit learning problems typically do not. 4

Our algorithms. Algorithm BalancedExploration explicitly optimizes over mixtures of arms, based on a simple idea: balanced exploration inside confidence bounds. The design principle underlying many confidence-bound based algorithms for stochastic MAB, including the famous UCB1 algorithm [Auer et al., 2002a] and our algorithm PrimalDualBwK , is generally, 'Exploit as much as possible, but use confidence bounds that are wide enough to encourage some exploration.' The design principle in BalancedExploration , in contrast, could be summarized as, 'Explore as much as possible, but use confidence bounds that are narrow enough to eliminate obviously suboptimal alternatives.' Our algorithm balances exploration across arms, exploring each arm as much as possible given the confidence bounds. More specifically, there are designated rounds when the algorithm picks a mixture that approximately maximizes the probability of choosing this arm, among the mixtures that are not obviously suboptimal given the current confidence bounds.

Algorithm PrimalDualBwK is a primal-dual algorithm based on the multiplicative weights update method. It maintains a vector of 'resource costs' that is adjusted using multiplicative updates. In every period it estimates each arm's expected reward and expected resource consumption, using upper confidence bounds for the former and lower confidence bounds for the latter; then it plays the most 'cost-effective' arm,

3 While the intuition behind this definition has been well-known for some time, the precise definition is due to Babaioff et al. [2014], Devanur and Kakade [2009].

4 A few notable exceptions are in [Auer et al., 2002b, Abraham et al., 2013, Besbes and Zeevi, 2012, Badanidiyuru et al., 2012]. Of these, Besbes and Zeevi [2012] and Badanidiyuru et al. [2012] are on special cases of BwK , and are discussed later.

namely the one with the highest ratio of estimated resource consumption to estimated resource cost, using the current cost vector. Although confidence bounds and multiplicative updates are the bread and butter of online learning theory, we consider this way of combining the two techniques to be quite novel. In particular, previous multiplicative-update algorithms in online learning theory - such as the Exp3 algorithm for MAB [Auer et al., 2002b] or the weighted majority [Littlestone and Warmuth, 1994] and Hedge [Freund and Schapire, 1997] algorithms for learning from expert advice - applied multiplicative updates to the probabilities of choosing different arms (or experts). Our application of multiplicative updates to the dual variables of the LP relaxation of BwK is conceptually quite a different usage of this technique.

Having alternative techniques to solve the same problem is generally useful in a rich problem space such as MAB. Indeed, one often needs to apply techniques beyond the original models for which they were designed, perhaps combining them with techniques that handle other facets of the problem. When pursuing such extensions, some alternatives may be more suitable than others, in particular because they are more compatible with the other techniques. We already see examples of that in the follow-up work:

Agrawal and Devanur [2014] and Badanidiyuru et al. [2014] use some of the techniques from BalancedExploration and PrimalDualBwK , resp., see Section 1.4 for more details.

LP-relaxation. In order to compare our algorithms to OPT , we compare both to a more tractable benchmark given by time-invariant mixtures of arms. More precisely, we define a linear programming relaxation for the expected total reward achieved by a time-invariant mixture of arms, and prove that the optimal value OPT LP achieved by this LP-relaxation is an upper bound for OPT . Therefore it suffices to relate our algorithms to the time-invariant mixture of arms that achieves OPT LP , and bound their regret with respect to OPT LP .

Lower bounds. The lower bound (2) is based on a simple example in which all arms have reward 1 and 0-1 consumption of a single resource, and one arm has slightly smaller expected resource consumption than the rest. To analyze this example, we apply the KL-divergence technique from the MAB lower bound in Auer et al. [2002b]. Some technical difficulties arise, compared to the derivation in Auer et al. [2002b], because the arms are different in terms of the expected consumption rather than expected reward, and because we need to match the desired value for OPT .

Discretization. In some applications, such as dynamic pricing and dynamic procurement, the action space X is very large or infinite, so our main algorithmic result is not immediately applicable. However, the action space has some structure that our algorithms can leverage: e.g., a price is just a number in some fixed interval. To handle such applications, we discretize the action space: we apply a BwK algorithm with a restricted, finite action space S ⊂ X , where S is chosen in advance. Immediately, we obtain a bound on regret with respect to the optimal dynamic policy restricted to S . Further, we select S so as to balance the tradeoff between | S | and the discretization error : the decrease in the performance benchmark due to restricting the action space to S . We call this approach preadjusted discretization . While it has been used in prior work, the key step of bounding the discretization error is now considerably more difficult, as one needs to take into account resource constraints and argue about mixtures of arms rather than individual arms.

We bound discretization error for subset S which satisfies certain axioms, and apply this result to handle dynamic pricing with a single product and dynamic procurement with a single budget constraint. While the former application is straightforward, the latter takes some work and uses a non-standard mesh of prices. Bounding the discretization error for more than one resource constraints (other than time) appears to be much more challenging; we only achieve this for a special case.

## 1.4 Follow-up work and open questions

Since the BwK problem provides a novel general problem formulation in online learning, it lends itself to a rich set of research questions in a similar way as the stochastic MAB problem did following Lai and Robbins

[1985] and Auer et al. [2002a]. Some of these questions were researched in the follow-up work.

Follow-up work. Following the conference publication of this paper [Badanidiyuru et al., 2013], there have been several developments directly inspired by BwK .

Agrawal and Devanur [2014] extend BwK from hard resource constraints and additive rewards to a more general model that allows penalties and diminishing returns. In particular, the time-averaged outcome vector ¯ v is constrained to lie in an arbitrary given convex set, and the total reward can be an arbitrary concave, Lipschitz-continuous function of ¯ v . They provide several algorithms for this model whose regret scales optimally as a function of the time horizon. Remarkably, these algorithms specialize to three new algorithms for BwK , based on different ideas. One of these new BwK algorithms follows the 'optimism under uncertainty' approach from [Auer et al., 2002a] (with an additional trick of rescaling the resource constraints). Despite the apparent simplicity, it is shown to satisfy our main regret bound (1).

Badanidiyuru et al. [2014] extend BwK to contextual bandits : a bandit model where in each round the 'context' is revealed ( e.g., a user profile), then the algorithm selects an arm, and the resulting outcome (in our case, reward and resource consumption) depends on both the chosen arm and the context. Badanidiyuru et al. [2014] merge BwK and contextual bandits with policy sets [Langford and Zhang, 2007], a well-established, very general model for contextual bandits. They achieve regret that scales optimally in terms of the time horizon and the number of policies (resp., square-root and logarithmic). Akin to BalancedExploration , their algorithm is not computationally efficient.

Both Agrawal and Devanur [2014] and Badanidiyuru et al. [2014] take advantage of various techniques developed in this paper. First, both papers use (a generalization of) linear relaxations from Section 3. In fact, the two claims in Section 3 are directly used in Badanidiyuru et al. [2014] to derive the corresponding statements for the contextual version. Second, Badanidiyuru et al. [2014] build on the design and analysis of BalancedExploration , and merging them with a technique from prior work on contextual bandits [Dud´ ıik et al., 2011]. Third, the analysis of one of the algorithms in Agrawal and Devanur [2014] relies on the bound on error terms (Lemma 5.6) from our analysis of PrimalDualBwK . Fourth, the analysis of discretization errors in Badanidiyuru et al. [2014] uses a technique from Section 7.

Two recent developments, Agrawal et al. [2016] and Agrawal and Devanur [2016], concern the contextual version of BwK . Agrawal et al. [2016] consider a common generalization of the extended BwK model in [Agrawal and Devanur, 2014] and the contextual BwK model in [Badanidiyuru et al., 2014]. In particular, for the latter model they achieve the same regret as Badanidiyuru et al. [2014], but with a computationally efficient algorithm, resolving the main open question in that paper. On a technical level, their work combines ideas from [Agrawal and Devanur, 2014] and a recent break-through in contextual bandits [Agarwal et al., 2014]. Agrawal and Devanur [2016] extend the model in [Agrawal and Devanur, 2014] to contextual bandits with a linear dependence on contexts ( e.g., see Chu et al. [2011]), achieving an algorithm with optimal dependence on the time horizon and the dimensionality of contexts. 5

Open questions (current status). While the general regret bound in Equation (1) is optimal up to logarithmic factors, better algorithms may be possible for various special cases. To rule out a domain-specific result that improves upon the general regret bound, one would need to prove a lower bound which, unlike the one in Equation (2), is specific to that domain. Currently domain-specific lower bounds are known only for the basic K -armed bandit problem and for dynamic pricing.

For problems with infinite multi-dimensional action spaces, such as dynamic pricing with multiple products and dynamic procurement with multiple budgets, we are limited by the lack of a general approach to upper-bound the discretization error and choose the preadjusted discretization in a principled way. A similar issue arises in the contextual extension of BwK studied in Badanidiyuru et al. [2014] and Agrawal et al.

5 Agrawal and Devanur [2014] prove a similar result for a special case when contexts do not change over time. They also claimed an extension to time-varying contexts, which has subsequently been retracted (see Footnote 1 in Agrawal and Devanur [2016]).

[2016], even for a single resource constraint. To obtain regret bounds that do not depend on a specific choice of preadjusted discretization, one may need to go beyond preadjusted discretization.

The study of multi-armed bandit problems with large strategy sets has been a very fruitful line of investigation. It seems likely that some of the techniques introduced here could be wedded with the techniques from that literature. In particular, it would be intriguing to try combining our primal-dual algorithm PrimalDualBwK with confidence-ellipsoid algorithms for stochastic linear optimization ( e.g., see Dani et al. [2008]), or enhancing the BalancedExploration algorithm with the technique of adaptively refined discretization, as in the zooming algorithm of Kleinberg et al. [2008].

It is tempting to ask about a version of BwK in which the rewards and resource consumptions are chosen by an adversary. Achieving sublinear regret bounds for this version appears hopeless even for the fixed-arm benchmark. In order to make progress in the positive direction, one may require a more subtle notion of benchmark and/or restrictions on the power of the adversary.

## 1.5 Related work

The study of prior-free algorithms for stochastic MAB problems was initiated by Lai and Robbins [1985] and Auer et al. [2002a]. Subsequent work supplied algorithms for stochastic MAB problems in which the set of arms can be infinite and the payoff function is linear, concave, or Lipschitz-continuous; see a recent survey [Bubeck and Cesa-Bianchi, 2012] for more background. Confidence bound techniques have been an integral part of this line of work, and they remain integral to ours.

As explained earlier, stochastic MAB problems constitute a very special case of bandits with knapsacks, in which there is only one type of resource and it is consumed deterministically at rate 1. Several papers have considered the natural generalization in which there is a single resource (other than time), with deterministic consumption, but different arms consume the resource at different rates. Guha and Munagala [2007] gave a constant-factor approximation algorithm for the Bayesian case of this problem, which was later generalized by Gupta et al. [2011] to settings in which the arms' reward processes need not be martingales. Tran-Thanh et al. [2010, 2012] presented prior-free algorithms for this problem; the best such algorithm achieves a regret guarantee qualitatively similar to that of the UCB1 algorithm.

Several recent papers study models that, in hindsight, can be cast as special cases of BwK :

- The two papers [Tran-Thanh et al., 2010, 2012] mentioned above and Ding et al. [2013] consider models with a single resource and unlimited time.
- Dynamic pricing with limited supply has been studied in [Besbes and Zeevi, 2009, Babaioff et al., 2015, Besbes and Zeevi, 2012, Wang et al., 2014]. 6
- The basic version of dynamic procurement (as per Section 1.1) has been studied in [Badanidiyuru et al., 2012, Singla and Krause, 2013]. 7 More background on the connection to crowdsourcing can be found in the survey Slivkins and Vaughan [2013].
- Dynamic ad allocation (without budget constraints) and various extensions thereof that incorporate user/webpage context have received a considerable attention, starting with [Pandey et al., 2007a,b, Langford and Zhang, 2007]. In fact, the connection to pay-per-click advertising has been one of the main drivers for the recent surge of interest in MAB.

6 The earlier papers [Blum et al., 2003, Kleinberg and Leighton, 2003] focus on the special case of unlimited supply. While we only cited papers that pursue regret-minimizing formulation of dynamic pricing, Bayesian and parametric formulations versions have a rich literature in Operations Research and Economics, see Boer [2015] for a literature review.

7 The regret bound in [Singla and Krause, 2013] is against the best-fixed-price benchmark, which may be much smaller than OPT , see Appendix A for a simple example. Benchmarks aside, one cannot directly compare our regret bound and theirs, because they do not derive a worst-case regret bound. [Singla and Krause, 2013] is simultaneous work w.r.t. our conference publication.

- [Amin et al., 2012, Tran-Thanh et al., 2014] study repeated bidding on a budget, and Cesa-Bianchi et al. [2013] study adjusting a repeated auction (albeit without inventory constraints); see Section 8 for more details on these special cases.
- Perhaps the earliest paper on resource consumption in MAB is Gy¨ orgy et al. [2007]. They consider a contextual bandit model where the only resource is time, consumed at different rate depending on the context and the chosen arm. The restriction to a single context is a special case of BwK .

Preadjusted discretization has been used in prior work on MAB on metric spaces ( e.g., [Kleinberg, 2004, Hazan and Megiddo, 2007, Kleinberg et al., 2008, Lu et al., 2010]) and dynamic pricing ( e.g., [Kleinberg and Leighton, 2003, Blum et al., 2003, Besbes and Zeevi, 2009, Babaioff et al., 2015]). However, bounding the discretization error in BwK is much more difficult.

Our BalancedExploration algorithm extends the 'active arms elimination' algorithm [Even-Dar et al., 2002] for the stochastic MAB problem, where one iterates over arms that are not obviously suboptimal given the current confidence bounds . The novelty is that our algorithm chooses over mixtures of arms, and the choice is 'balanced' across arms. 'Policy elimination' algorithm of Dud´ ıik et al. [2011] extends 'active arms elimination' in a different direction: to contextual bandits. Like BalancedExploration , policy elimination algorithm makes a 'balanced' choice among objects that are more complicated than arms, and this choice is not computationally efficient; however, the technical details are very different.

While BwK is primarily an online learning problem, it also has elements of a stochastic packing problem. The literature on prior-free algorithms for stochastic packing has flourished in recent years, starting with prior-free algorithms for the stochastic AdWords problem [Devanur and Hayes, 2009], and continuing with a series of papers extending these results from AdWords to more general stochastic packing integer programs while also achieving stronger performance guarantees [Agrawal et al., 2014, Devanur et al., 2011, Feldman et al., 2010, Molinaro and Ravi, 2012]. A running theme of these papers (and also of the primal-dual algorithm in this paper) is the idea of estimating of an optimal dual vector from samples, then using this dual to guide subsequent primal decisions. Particularly relevant to our work is the algorithm of Devanur et al. [2011], in which the dual vector is adjusted using multiplicative updates, as we do in our algorithm. However, unlike the BwK problem, the stochastic packing problems considered in prior work are not learning problems: they are full information problems in which the costs and rewards of decisions in the past and present are fully known. The only uncertainty is about the future.) As such, designing algorithms for BwK requires a substantial departure from past work on stochastic packing. Our primal-dual algorithm depends upon a hybrid of confidence-bound techniques from online learning and primal-dual techniques from the literature on solving packing LPs; combining them requires entirely new techniques for bounding the magnitude of the error terms that arise in the analysis. Moreover, our BalancedExploration algorithm manages to achieve strong regret guarantees without even computing a dual solution.

## 2 Preliminaries

BwK : problem formulation. There is a fixed and known, finite set of m arms (possible actions), denoted X . There are d resources being consumed. The time proceeds in T rounds, where T is a finite, known time horizon. In each round t , an algorithm picks an arm x t ∈ X , receives reward r t ∈ [0 , 1] , and consumes some amount c t,i ∈ [0 , 1] of each resource i . The values r t and c t,i are revealed to the algorithm after the round. There is a hard constraint B i ∈ R + on the consumption of each resource i ; we call it a budget for resource i . The algorithm stops at the earliest time τ when one or more budget constraint is violated; its total reward is equal to the sum of the rewards in all rounds strictly preceding τ . The goal of the algorithm is to maximize the expected total reward.

The vector ( r t ; c t, 1 , c t, 2 , . . . , c t,d ) ∈ [0 , 1] d +1 is called the outcome vector for round t . We assume stochastic outcomes : if an algorithm picks arm x , the outcome vector is chosen independently from some fixed distribution π x over [0 , 1] d +1 . The distributions π x , x ∈ X are not known to the algorithm. The tuple ( π x : x ∈ X ) comprises all latent information in the problem instance. A particular BwK setting (such as 'dynamic pricing with limited supply') is defined by the set of all feasible tuples ( π x : x ∈ X ) . This set, called the BwK domain , is known to the algorithm.

We compare the performance of our algorithms to the expected total reward of the optimal dynamic policy given all the latent information, which we denote by OPT . (Note that OPT depends on the latent information, and therefore is a latent quantity itself.) Regret is defined as OPT minus the expected total reward of the algorithm.

W.l.o.g. assumptions. For technical convenience, we make several assumptions that are w.l.o.g.

We express the time horizon as a resource constraint: we model time as a specific resource, say resource 1 , such that every arm deterministically consumes B 1 /T units of this resource whenever it is picked. W.l.o.g., B i ≤ T for every resource i .

We assume there exists an arm, called the null arm which yields no reward and no consumption of any resource other than time. Equivalently, an algorithm is allowed to spend a unit of time without doing anything. Any algorithm ALG that uses the null arm can be transformed, without loss in expected total reward, to an algorithm ALG ′ that does not use the null arm. Indeed, in each round ALG ′ runs ALG until it selects a non-null arm x or halts. In the former case, ALG ′ selects x and returns the observe feedback to ALG . After ALG halts, ALG ′ selects arms arbitrarily.

We say that the budgets are uniform if B i = B for each resource i . Any BwK instance can be reduced to one with uniform budgets by dividing all consumption values for every resource i by B i /B , where B = min i B i . (That is tantamount to changing the units in which we measure consumption of resource i .) Our technical results are for BwK with uniform budgets. We will assume uniform budgets B from here on.

Useful notation. Let µ x = E [ π x ] ∈ [0 , 1] d +1 be the expected outcome vector for each arm x , and denote µ = ( µ x : x ∈ X ) . We call µ the latent structure of a problem instance. The BwK domain induces a set of feasible latent structures, which we denote M feas .

If D is a distribution over arms, let r ( D , µ ) = ∑ x ∈ X D ( x ) r ( x, µ ) and c ( D , µ ) = ∑ x ∈ X D ( x ) c ( x, µ ) be, respectively, the expected reward and expected resource consumption in a single round if an arm is sampled from distribution D . Let REW ( D , µ ) denote the expected total reward of the time-invariant policy that uses distribution D .

For notational convenience, we will write µ x = ( r ( x, µ ); c 1 ( x, µ ) , . . . , c d ( x, µ ) ) . Also, we will write the expected consumption as a vector c ( x, µ ) = ( c 1 ( x, µ ) , . . . , c d ( x, µ ) ) .

High-probability events. We will use the following expression, which we call the confidence radius .

<!-- formula-not-decoded -->

Here C rad = Θ(log( dT | X | )) is a parameter which we will fix later; we will keep it implicit in the notation. The meaning of Equation (3) and C rad is explained by the following tail inequality from [Kleinberg et al., 2008, Babaioff et al., 2015]. 8

Theorem 2.1 (Kleinberg et al. [2008], Babaioff et al. [2015]) . Consider some distribution with values in [0 , 1] and expectation ν . Let ν be the average of N independent samples from this distribution. Then

̂ ̂ 8 Specifically, this follows from Lemma 4.9 in the full version of Kleinberg et al. [2008], and Theorem 4.8 and Theorem 4.10 in the full version of Babaioff et al. [2015] (both full versions can be found on arxiv.org).

<!-- formula-not-decoded -->

More generally, Equation (4) holds if X 1 , . . . , X N ∈ [0 , 1] are random variables, ̂ ν = 1 N ∑ N t =1 X t is the sample average, and ν = 1 N ∑ N t =1 E [ X t | X 1 , . . . , X t -1 ] . If the expectation ν is a latent quantity, Equation (4) allows us to estimate ν by a high-confidence interval

ν ∈ [ ̂ ν -rad ( ̂ ν, N ) , ̂ ν + rad ( ̂ ν, N )] , (5) whose endpoints are observable (known to the algorithm). This estimate is on par with the one provided by Azuma-Hoeffding inequality (up to constant factors), but is much sharper for small ν . 9

It is sometimes useful to argue about any ν which lies in the high-confidence interval (5), not just the latent ν = E [ ν ] . We use the following claim which is implicit in Kleinberg et al. [2008].

## 3 LP relaxation for policy value

̂ Claim 2.2 (Kleinberg et al. [2008]) . For any ν, ̂ ν ∈ [0 , 1] , Equation (5) implies that rad ( ̂ ν, N ) ≤ 3 rad ( ν, N ) .

OPT -the expected reward of the optimal dynamic policy given foreknowledge of the distribution of outcome vectors - is typically difficult to characterize exactly. In fact, even for a time-invariant policy, it is difficult to give an exact expression for the expected reward due to the dependence of the reward on the random stopping time when the resource budget is exhausted. To approximate these quantities, we consider the fractional relaxation of BwK in which the number of rounds in which a given arm is selected (and also the total number of rounds) can be fractional, and the reward and resource consumption per unit time are deterministically equal to the corresponding expected values in the original instance of BwK .

The following linear program constitutes our fractional relaxation of the optimal dynamic policy.

<!-- formula-not-decoded -->

The variables ξ x represent the fractional relaxation for the number of rounds in which a given arm x is selected. This is a bounded LP (because ∑ x ξ x r ( x, µ ) ≤ ∑ x ξ x ≤ T ). The optimal value of this LP is denoted by OPT LP . We will also use the dual LP, shown below.

<!-- formula-not-decoded -->

The dual variables η i can be interpreted as a unit cost for the corresponding resource i .

Lemma 3.1. OPT LP is an upper bound on the value of the optimal dynamic policy: OPT LP ≥ OPT .

One way to prove this lemma is to define ξ x to be the expected number of times arm x is played by the optimal dynamic policy, and argue that the vector ( ξ x , x ∈ X ) is primal-feasible and that ∑ x ξ x r ( x, µ ) is the expected reward of the optimal dynamic policy. We instead present a simpler proof using ( LP-dual ) and a martingale argument. A similar lemma (but for a technically different setting of online stochastic packing problems) was proved in Devanur et al. [2011].

9 Essentially, Azuma-Hoeffding inequality states that | ν -̂ ν | ≤ O ( √ C rad /N ) , whereas by Theorem 2.1 for small ν it holds with high probability that rad ( ̂ ν, N ) ∼ C rad /N .

Proof of Lemma 3.1. Let η ∗ = ( η ∗ 1 , . . . , η ∗ d ) denote an optimal solution to ( LP-dual ). Interpret each η ∗ i as a unit cost for the corresponding resource i . By strong LP duality, we have B ∑ i η ∗ i = OPT LP . Dual feasibility implies that for each arm x , the expected cost of resources consumed when x is pulled exceeds the expected reward produced. Thus, if we let Z t denote the sum of rewards gained in rounds 1 , . . . , t of the optimal dynamic policy, plus the cost of the remaining resource endowment after round t , then the stochastic process Z 0 , Z 1 , . . . , Z T is a supermartingale. Let τ be the stopping time of the algorithm, i.e. the total number of rounds. Note that Z 0 = B ∑ i η ∗ i = OPT LP , and Z τ -1 equals the algorithm's total payoff, plus the cost of the remaining (non-negative) resource supply at the start of round τ . By Doob's optional stopping theorem, Z 0 ≥ E [ Z τ -1 ] and the lemma is proved.

Remark 3.2 . Implicit in this proof is a simple, but powerful observation that for any algorithm,

<!-- formula-not-decoded -->

Each summand on the right-hand side is non-negative, and equals 0 if and only if the arm x t lies in the support of the primal solution. We use this observation to motivate the design of our primal-dual algorithm. Remark 3.3 . For each of the two main algorithms, we prove a regret bound of the form

<!-- formula-not-decoded -->

where REW is the expected total reward of the algorithm, and f () depends only on parameters ( B,m,d ) . This regret bound has an optimal scaling property, highlighted in the Introduction: if all budget constraints, including the time horizon, are increased by the factor of α , then the regret bound f ( OPT LP ) scales as √ α .

Regret bound (6) implies the claimed regret bounds relative to OPT because

<!-- formula-not-decoded -->

where the second inequality follows trivially because g ( x ) = max( x -f ( x ) , 0) is a non-decreasing function of x for x ≥ 0 , and OPT LP ≥ OPT .

Let us apply a similar LP-relaxation to a time-invariant policy that uses distribution D over arms. We approximate the expected total reward of this policy in a similar way: we define a linear program in which the only variable t represents the expected stopping time of the algorithm.

<!-- formula-not-decoded -->

The optimal value to ( LP-distr ), which we call the LP-value of D , is

<!-- formula-not-decoded -->

Observe that t is feasible for ( LP-distr ) if and only if ξ = t D is feasible for ( LP-primal ). Therefore

<!-- formula-not-decoded -->

This supremum is attained by any distribution D ∗ = ξ/ ‖ ξ ‖ 1 such that ξ = ( ξ x : x ∈ X ) is an optimal solution to ( LP-primal ). A distribution D ∗ ∈ argmax D LP ( D , µ ) is called LP-optimal for µ .

Claim 3.4. For any latent structure µ , there exists a distribution D over arms which is LP-optimal for µ and moreover satisfies the following three properties:

- (a) c i ( D , µ ) ≤ B/T for each resource i .
- (c) If D has a support of size exactly 2 then for some resource i we have c i ( D , µ ) = B/T . (Such distribution D will be called LP-perfect for µ .)
- (b) D has a support of size at most d .

Proof. Fix the latent structure µ . It is a well-known fact that for any linear program there exists an optimal solution whose support has size that is exactly equal to the number of constraints that are tight for this solution. Take any such optimal solution ξ = ( ξ x : x ∈ X ) for ( LP-primal ), and take the corresponding LP-optimal distribution D = ξ/ ‖ ξ ‖ 1 . Since there are d constraints in ( LP-primal ), distribution D has support of size at most d . If it satisfies (a), then it also satisfies (c) (else it is not optimal), and we are done.

Suppose property (a) does not hold for D . Then there exists a resource i such that c i ( D , µ ) &gt; B/T . Since the i -th constraint in ( LP-primal ) can be restated as ‖ ξ ‖ 1 c i ( D , µ ) ≤ B , it follows that ‖ ξ ‖ 1 &lt; T . Therefore the constraint in ( LP-primal ) that expresses the time horizon is not tight. Consequently, at most d -1 constraints in ( LP-primal ) are tight for ξ , so the support of D has size at most d -1 .

Note that c j ( D ′ , µ ) = αc j ( D , µ ) ≤ B/T for each resource j , with equality for j = i . Hence, D ′ satisfies properties (a) and (c). Also, r ( D ′ , µ ) = αr ( D , µ ) , and so

Let us modify D to obtain another LP-optimal distribution D ′ which satisfies properties (a-c). W.l.o.g., pick i to maximize c i ( D , µ ) and let α = B T /c i ( D , µ ) . Define D ′ ( x ) = α D ( x ) for each non-null arm x and place the remaining probability in D ′ on the null arm. This completes the definition of D ′ .

<!-- formula-not-decoded -->

Therefore D ′ is LP-optimal. It satisfies property (b) because it adds at most one to the support of D .

## 4 Algorithm BalancedExploration

This section presents and analyzes BalancedExploration , one of the two main algorithms. The design principle behind BalancedExploration is to explore as much as possible while avoiding obviously suboptimal strategies. On a high level, the algorithm is very simple. The goal is to converge on an LP-perfect distribution. The time is divided into phases of | X | rounds each. In the beginning of each phase p , the algorithm prunes away all distributions D over arms that with high confidence are not LP-perfect given the observations so far. The remaining distributions over arms are called potentially perfect . Throughout the phase, the algorithm chooses among the potentially perfect distributions. Specifically, for each arm x , the algorithm chooses a potentially perfect distribution D p,x which approximately maximizes D p,x ( x ) , and 'pulls' an arm sampled independently from this distribution. This choice of D p,x is crucial; we call it the balancing step . The algorithm halts as soon as the time horizon is met, or any of the constraints is exhausted. The pseudocode is given in Algorithm 1.

## Algorithm 1 BalancedExploration

- 1: For each phase p = 0 , 1 , 2 , . . . do
- 2: Recompute the set ∆ p of potentially perfect distributions D over arms.
- 4: pick any distribution D = D p,x ∈ ∆ p such that D ( x ) ≥ 1 2 max D ′ ∈ ∆ p D ′ ( x ) .
- 3: Over the next | X | rounds, for each x ∈ X :
- 5: choose an arm to 'pull' as an independent sample from D .
- 6: halt if time horizon is met or one of the resources is exhausted.

We believe that BalancedExploration , like UCB1 [Auer et al., 2002a], is a very general design principle and has the potential to be a meta-algorithm for solving stochastic online learning problems.

Theorem 4.1. Consider an instance of BwK with d resources, m = | X | arms, and the smallest budget B = min i B i . Algorithm BalancedExploration achieves regret

Moreover, Equation (7) holds with f ( OPT LP ) equal to the right-hand side of Equation (9).

<!-- formula-not-decoded -->

Remark 4.2 . The specification of BalancedExploration involves a mathematically well-defined step approximate optimization over potentially perfect distributions - for which we do not provide a specific implementation. Yet, BalancedExploration is a bandit algorithm in the sense that it is a well-defined mapping from histories to actions. We prove an 'information-theoretic' statement: there is an algorithm with the claimed regret. Such results are not uncommon in the literature, e.g., [Kleinberg et al., 2008, Kleinberg and Slivkins, 2010, Agarwal et al., 2014], typically as first solutions for new, broad problem formulations, and are meaningful as proof-of-concept for the corresponding regret bounds and techniques.

Remaining details of the specification. In the beginning of each phase p , the algorithm recomputes a 'confidence interval' I p for the latent structure µ , so that (informally) µ ∈ I p with high probability. Then the algorithm determines which distributions D over arms can potentially be LP-perfect given that µ ∈ I p . Specifically, let ∆ p be set of all distributions D that are LP-perfect for some latent structure µ ′ ∈ I p ; such distributions are called potentially perfect (for phase p ).

It remains to define the confidence intervals I p . For phase p = 0 , the confidence interval I 0 is simply M feas , the set of all feasible latent structures. For each subsequent phase p ≥ 1 , the confidence interval I p is defined as follows. For each arm x , consider all rounds before phase p in which this arm has been chosen. Let N p ( x ) be the number of such rounds, let ̂ r p ( x ) be the time-averaged reward in these rounds, and let ̂ c p,i ( x ) be the time-averaged consumption of resource i in these rounds. We use these averages to estimate r ( x, µ ) and c i ( x, µ ) as follows:

̂ ̂ The confidence interval I p is the set of all latent structures µ ′ ∈ I p -1 that are consistent with these estimates. This completes the specification of BalancedExploration .

<!-- formula-not-decoded -->

For each phase of BalancedExploration , the round in which an arm is sampled from distribution D p,x will be called designated to arm x . We need to use approximate maximization to choose D p,x , rather than exact maximization, because an exact maximizer argmax D∈ ∆ p D ( x ) is not guaranteed to exist.

Proof overview. We start with some properties of the algorithm that follow immediately from the specification and hold deterministically (with probability 1). Then we identify several properties that the algorithm satisfies with very high probability. The rest of the analysis focuses on a 'clean execution' of the algorithm: an execution in which all these properties hold. We analyze the 'error terms' that arise due to the uncertainty on the latent structure, and use the resulting 'error bounds' to argue about the algorithm's performance.

## 4.1 Deterministic properties of BalancedExploration

First, we show that any two latent structures in the confidence interval I p correspond to similar consumptions and rewards, for each arm x . This follows deterministically from the specification of I p .

Claim 4.3. Fix any phase p , any two latent structures µ ′ , µ ′′ ∈ I p , an arm x , and a resource i . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We prove Equation (12); Equation (13) is proved similarly.

Let N = N p ( x ) . By specification of BalancedExploration , any µ ′ ∈ I p is consistent with estimate (11):

It follows that

<!-- formula-not-decoded -->

Finally, we observe that by Claim 2.2,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ For each phase p and arm x , let ¯ D p,x = 1 p ∑ q&lt;p D q,x ( x ) be the average of probabilities for arm x among the distributions in the preceding phases that are designated to arm x . Because of the balancing step in BalancedExploration , we can compare this quantity to D ( x ) , for any D ∈ ∆ p . (Here we also use the fact that the confidence intervals I p are non-increasing from one phase to another.)

<!-- formula-not-decoded -->

Proof. Fix arm x . Recall that ¯ D p,x = 1 p ∑ q&lt;p D q,x ( x ) , where D q,x is the distribution chosen in the round in phase q that is designated to arm x . Fix any phase q &lt; p . Because of the balancing step, D q,x ( x ) ≥ 1 2 D ′ ( x ) for any distribution D ′ ∈ ∆ q . Since the confidence intervals I q are non-increasing from one phase to another, we have I p ⊂ I q for any q ≤ p , which implies that ∆ p ⊂ ∆ q . Consequently, D q,x ( x ) ≥ 1 2 D ( x ) for each q &lt; p , and the claim follows.

## 4.2 High-probability events

We keep track of several quantities: the averages ̂ r p ( x ) and ̂ c p,i ( x ) defined above, as well as several other quantities that we define below.

Further, consider all rounds in phases q &lt; p . There are N = p | X | such rounds. The average distribution chosen by the algorithm in these rounds is ¯ D p = 1 N ∑ q&lt;p,x ∈ X D q,x . We are interested in the corresponding quantities r ( ¯ D p , µ ) and c i ( ¯ D p , µ ) , We compare these quantities to ̂ r p = 1 N ∑ N t =1 r t and ̂ c p,i = 1 N ∑ N t =1 c t,i , the average reward and the average resourcei consumption in phases q &lt; p .

Fix phase p and arm x . Recall that N p ( x ) is the number of rounds before phase p in which arm x is chosen. Now, let us consider all rounds before phase p that are designated to arm x . Let n p ( x ) denote the number of times arm x has been chosen in these rounds. Let ̂ D p,x = n t ( x ) /p be the corresponding empirical probability of choosing x . We compare this to ¯ D p,x .

We consider several high-probability events which follow from applying Theorem 2.1 to the various quantities defined above. All these events have a common shape: some quantities ν, ̂ ν satisfy Equation (5) for some N . If this is the case, we that ν is an N -strong estimator for ν .

̂ Lemma 4.5. For each phase p , arm x , and resource i , with probability e -Ω( C rad ) it holds that:

- (a) r p ( x ) is an N p ( x ) -strong estimator for r ( x, µ ) , and ̂ c p,i ( x ) is an N p ( x ) -strong estimator for c i ( x, µ ) .

(c) r ( ¯ D p , µ ) is an ( p | X | ) -strong estimator for ̂ r p , and c i ( ¯ D p , µ ) is an ( p | X | ) -strong estimator for ̂ c p,i . We rely on several properties of the confidence radius rad () , which we summarize below. (We omit the easy proofs.)

- ̂ (b) ¯ D p,x is an p -strong estimator for ̂ D p,x .

Claim 4.6. The confidence radius rad ( ν, N ) , defined in Equation (3), satisfies the following properties:

- (a) monotonicity: rad ( ν, N ) is non-decreasing in ν and non-increasing in N .
- (b) concavity: rad ( ν, N ) is concave in ν , for any fixed N .
- (c) max(0 , ν -rad ( ν, N )) is non-decreasing in ν .
- (e) rad ( ν, N ) ≤ 3 C rad N whenever ν ≤ 4 C rad N .
- (d) ν -rad ( ν, N ) ≥ 1 4 ν whenever 4 C rad N ≤ ν ≤ 1 .
- (f) rad ( ν, αN ) = 1 α rad ( αν, N ) , for any α ∈ (0 , 1] .

## 4.3 Clean execution analysis

- (g) 1 N ∑ N /lscript =1 rad ( ν, /lscript ) ≤ O (log N ) rad ( ν, N ) .

It is convenient to focus on a clean execution of the algorithm: an execution in which all events in Lemma 4.5 hold. We assume a clean execution in what follows. Also, we fix an arbitrary phase p in such execution.

Clean execution analysis falls into two parts. First, we analyze the 'error terms': we look at the LP-value (resp., expected reward, or expected resource consumption) of a given distribution, and upper-bound the difference in this quantity between different latent structures µ, µ ′ in the confidence interval I p , or between different potentially perfect distributions D ′ , D ′′ ∈ ∆ p . The culmination is Lemma 4.12, which upperbounds the difference | LP ( D ′ , µ ′ ) -LP ( D ′′ , µ ′′ ) | in terms of parameters p d , B , T , and OPT LP . Second, we apply these error bounds to reason about the algorithm itself. The key quantities of interest are LP-values of the chosen distributions, average reward/consumption, and the stopping time.

## 4.3.1 Bounding the error terms

Since a clean execution satisfies the event in Claim 4.5(a), it immediately follows that:

Claim 4.7. The confidence interval I p contains the (actual) latent structure µ . Therefore, D ∗ ∈ ∆ p for any distribution D ∗ that is LP-perfect for µ .

,

<!-- formula-not-decoded -->

Claim 4.8. Fix any latent structures µ ′ , µ ′′ ∈ I p and any distribution D ∈ ∆ p . Then for each resource i

Proof. We prove Equation (14); Equation (15) is proved similarly. Let us first prove the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Intuitively, in order to argue that we have good estimates on quantities related to arm x , it helps to prove that this arm has been chosen sufficiently often. Using the definition of clean execution and Claim 4.4, we accomplish this as follows:

<!-- formula-not-decoded -->

Consider two cases depending on D ( x ) . For the first case, assume D ( x ) ≥ 8 C rad p . Using Claim 4.6(d) and the previous equation, it follows that N p ( x ) ≥ 1 8 p D ( x ) . Therefore:

<!-- formula-not-decoded -->

The second case is that D ( x ) &lt; 8 C rad p . Then Equation (16) follows simply because C rad p ≤ rad ( · , p ) .

<!-- formula-not-decoded -->

We have proved Equation (16). We complete the proof of Equation (14) using concavity of rad ( · , p ) and the fact that, by the specification of BalancedExploration , D has support of size at most d .

In what follows, we will denote M p = max D∈ ∆ p , µ ∈ I p LP ( D , µ ) .

Claim 4.9. Fix any latent structures µ ′ , µ ′′ ∈ I p and any distribution D ∈ ∆ p . Then

Proof. Since D ∈ ∆ p , it is LP-perfect for some latent structure µ . Then LP ( D , µ ) = T r ( D , µ ) . Therefore:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We need a little more work to bound the difference in the LP values in the other direction.

Consider t 0 = LP ( D , µ ′ ) /r ( D , µ ′ ) ; this is the value of the variable t in the optimal solution to the linear program ( LP-distr ). Let us obtain a lower bound on this quantity. Assume t 0 &lt; T . Then one of the budget constraints in ( LP-distr ) must be tight, i.e. t 0 c i ( D , µ ′ ) = B for some resource i .

Let Ψ = rad ( B T , p d ) . It follows that t 0 = B/c i ( D , µ ′ ) ≥ T (1 -O ( T B Ψ)) . Therefore:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using Equation (18) and noting that r ( D , µ ) = LP ( D , µ ) /T ≤ M p /T , we conclude that

We obtain the same upper bound on | LP ( D , µ ) -LP ( D , µ ′′ ) | , and the claim follows.

We will use Φ p ( M p ) to denote the right-hand side of Equation (17) as a function of M p .

## Claim 4.10.

- (a) Fix any latent structure µ ∗ ∈ I p , and any distributions D ′ , D ′′ ∈ ∆ p . Then

<!-- formula-not-decoded -->

- (b) Fix any latent structure µ ′ , µ ′′ ∈ I p , and any distributions D ′ , D ′′ ∈ ∆ p . Then

<!-- formula-not-decoded -->

Proof. (a). Since D ′ , D ′′ ∈ ∆ p , it holds that D ′ and D ′′ are LP-perfect for some latent structures µ ′ and µ ′′ . Further, pick a distribution D ∗ that is LP-perfect for µ ∗ . Then:

<!-- formula-not-decoded -->

- (b). Follows easily from part (a) and Lemma 4.9.

The following claim will allow us to replace Φ p ( M p ) by Φ p ( OPT LP ) .

<!-- formula-not-decoded -->

Proof. Consider the two summands in Φ p ( M p ) :

<!-- formula-not-decoded -->

We consider the following three cases. The first case is that S 1 ( M p ) ≥ M p / 12 . Solving for M p , we obtain M p ≤ O ( TdC rad p ) , which implies that

<!-- formula-not-decoded -->

The second case is that S 2 ( M p ) ≥ M p / 12 . Then

<!-- formula-not-decoded -->

In remaining case, Φ p ( M p ) ≤ M p 6 . Then from Claim 4.10(b) we get that M p ≤ 2 OPT LP . Noting that Φ p ( M ) is a non-decreasing function of M , we obtain

<!-- formula-not-decoded -->

Claim 4.11 and Claim 4.10 imply our main bound on the error terms:

Lemma 4.12. Fix any latent structure µ ′ , µ ′′ ∈ I p , and any distributions D ′ , D ′′ ∈ ∆ p . Then

<!-- formula-not-decoded -->

## 4.3.2 Performance of the algorithm

The remainder of the analysis deals with rewards and resource consumption of the algorithm. We start with lower-bounding the LP-value for the chosen distributions.

Claim 4.13. For each distribution D p,x chosen by the algorithm in phase p ,

<!-- formula-not-decoded -->

Proof. The claim follows easily from Lemma 4.12, noting that D p,x ∈ ∆ p .

The following corollary lower-bounds the average reward; once we have it, it essentially remains to lower-bound the stopping time of the algorithm.

Corollary 4.14. ̂ r p ≥ 1 T ( OPT LP -O (log p ) Φ p ( OPT LP )) . Proof. Throughout this proof, denote Φ p /defines Φ p ( OPT LP ) . By Claim 4.13, for each distribution D q,x chosen by the algorithm in phase q &lt; p it holds that

<!-- formula-not-decoded -->

Averaging the above equation over all rounds in phases q &lt; p , we obtain

<!-- formula-not-decoded -->

For the last inequality, we used Claim 4.6(fg) to average the confidence radii in Φ q . Using the high-probability event in Claim 4.5(c):

<!-- formula-not-decoded -->

Now using the monotonicity of ν -rad ( ν, N ) (Claim 4.6(c)) we obtain

<!-- formula-not-decoded -->

For the last equation, we use the fact that Φ p /T ≥ Ω( rad ( OPT LP /T, p d )) ≥ Ω( rad ( OPT LP /T, p | X | )) .

The following two claims help us to lower-bound the stopping time of the algorithm.

Claim 4.15. c i ( D p,x , µ ) ≤ B T + O (1) rad ( B T , p d ) for each resource i .

Proof. By the algorithm's specification, D p,x ∈ ∆ p , and moreover there exists a latent structure µ ′ ∈ I p such that D p,x is LP-perfect for µ ′ . Apply Claim 4.8, noting that c i ( D p,x , µ ′ ) ≤ B T by LP-perfectness.

Proof. Using a property of the clean execution, namely the event in Claim 4.5(c), we have

<!-- formula-not-decoded -->

Consider all rounds preceding phase p .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the last inequality, we used Claim 4.6(fg) to average the confidence radii.

Using the upper bound on c i ( ¯ D , µ ) that we derived above,

Using a general property of the confidence radius that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we conclude that

We obtain the claim by plugging the upper bounds (20) and (21) into (19).

We are ready to put the pieces together and derive the performance guarantee for a clean execution of BalancedExploration .

Lemma 4.17. Consider a clean execution of BalancedExploration . Then the total reward

<!-- formula-not-decoded -->

We can use Corollary 4.14 to bound REW from below:

Proof. Throughout this proof, denote Φ p /defines Φ p ( OPT LP ) . Let p be the last phase in the execution of the algorithm, and let T 0 be the stopping time. Letting m = | X | , note that pm &lt; T 0 ≤ ( p +1) m .

REW = T 0 ̂ r p +1 &gt; pm ̂ r p +1 ≥ p m T ( OPT LP -O (Φ p log p )) . (22) Let us bound pm T from below. The algorithm stops either when it runs out of time or if it runs out of resources during phase p . In the former case, p = /floorleft T/m /floorright . In the latter case, B = T 0 ̂ c p +1 , i for some resource i , so B ≤ m ( p +1) ̂ c p +1 , i . Using Corollary 4.16, we obtain the following lower bound on p :

<!-- formula-not-decoded -->

Plugging this into Equation (22), we conclude:

<!-- formula-not-decoded -->

To complete the proof, we observe that ( p Φ p log p ) is increasing in p (by definition of Φ p ), and plug in a trivial upper bound p ≤ T/m .

To finish the proof of Theorem 4.1, we write down the definition of Φ T/m ( OPT LP ) , m = | X | , and plug in the definition of the confidence radius (3):

<!-- formula-not-decoded -->

## 5 Algorithm PrimalDualBwK

This section develops an algorithm, called PrimalDualBwK , that solves the BwK problem using a very natural and intuitive idea: greedily select arms with the greatest estimated 'bang per buck,' i.e. reward per unit of resource consumption. One of the main difficulties with this idea is that there is no such thing as a known 'unit of resource consumption': there are d different resources, and it is unclear how to trade off consumption of one resource versus another. The dual LP in Section 3 gives some insight into how to quantify this trade-off: an optimal dual solution η ∗ can be interpreted as a vector of unit costs for resources, such that for every arm the expected reward is less than or equal to the expected cost of resources consumed. Then the bang-per-buck ratio for a given arm x can be defined as r ( x, µ ) / ( η ∗ · c ( x, µ )) , where the denominator represents the expected cost of pulling this arm. The arms in the support of the optimal distribution ξ ∗ are precisely the arms with a maximal bang-per-buck ratio (by complimentary slackness), and pulling any other arm necessarily increases regret relative to OPT LP (by Remark 3.2).

To estimate the bang-per-buck ratios, our algorithm will try to learn an optimal dual vector η ∗ in tandem with learning the latent structure µ . Borrowing an idea from [Plotkin et al., 1995, Garg and K¨ onemann, 2007, Arora et al., 2012], we use the multiplicative weights update method to learn the optimal dual vector. This method raises the cost of a resource exponentially as it is consumed, which ensures that heavily demanded resources become costly, and thereby promotes balanced resource consumption. Meanwhile, we still have to ensure (as with any multi-armed bandit problem) that our algorithm explores the different arms frequently enough to gain adequately accurate estimates of the latent structure. We do this by estimating rewards and resource consumption as optimistically as possible, i.e. using upper confidence bound (UCB) estimates for rewards and lower confidence bound (LCB) estimates for resource consumption. Although both of these techniques - multiplicative weights and confidence bounds - have been successfully applied in previous online learning algorithms, it is far from obvious that this particular hybrid of the two methods should be effective. In particular, the use of multiplicative updates on dual variables, rather than primal ones, distinguishes our algorithm from other bandit algorithms that use multiplicative weights (e.g. the Exp3 algorithm [Auer et al., 2002b]) and brings it closer in spirit to the literature on stochastic packing algorithms, especially [Devanur et al., 2011].

The pseudocode is presented as Algorithm 2. When we refer to the UCB or LCB for a latent parameter (the reward of an arm, or the amount of some resource that it utilizes), these are computed as follows. Letting ˆ ν denote the empirical average of the observations of that random variable 10 and letting N denote the number of times the random variable has been observed, the lower confidence bound (LCB) and upper confidence bound (UCB) are the left and right endpoints, respectively, of the confidence interval [0 , 1] ∩ [ˆ ν -rad (ˆ ν, N ) , ˆ ν + rad (ˆ ν, N )] . The UCB or LCB for a vector or matrix are defined componentwise.

The algorithm is fast: with machine word size of log T bits or more, the per-round running time is O ( md ) . Moreover, if each arm x consumes only d x resources that are known in advance, then L t,x can be implemented as a d x -dimensional vector, and EstCost x can be computed in O ( d x ) time. Then the per-round running time is O ( m + d + ∑ x d x ) .

Discussion 5.1 . The cost update in step 13 requires some explanation. Let us interpret this step as a separate algorithm which solves a particular problem. The problem is to optimize the total expected payoff when in each round t &gt; m , one chooses a distribution y t = v t / ‖ v t ‖ 1 over resources, and receives expected payoff y t · L t,x t . This is the well-known 'best-expert' problem in which actions correspond to resources, and each action i is assigned payoff L t,x t ( i ) . Step 13 implements a multiplicative-weights algorithm for solving this problem. In fact, we could have used any other algorithm for this problem with a similar performance guarantee, as in Proposition 5.4.

10 Note that we initialize the algorithm by pulling each arm once, so empirical averages are always well-defined.

## Algorithm 2 PrimalDualBwK with parameter /epsilon1 ∈ (0 , 1)

<!-- formula-not-decoded -->

But why does solving this particular best-experts problem make sense for PrimalDualBwK ? Particularly, why does it make sense to maximize this notion of expected payoffs? Let us view distribution y t as a vector of normalized costs of resources. Consider the total expected normalized cost consumed by the algorithm after round m , denote it W . Then W = ∑ τ t = m +1 y t c ( x t , µ ) . A lower confidence bound on this quantity is W LCB = ∑ τ t = m +1 y t · L t,x t , which is precisely the total expected payoff in the best-experts problem. In the analysis, we relate W LCB and the upper confidence bound on the total expected reward in the same rounds, REW UCB = ∑ τ t = m +1 u t,x t Specifically, we prove that for any implementation of step 13, we have

<!-- formula-not-decoded -->

(This follows from Equation (31).) Thus, maximizing W LCB is a reasonable goal for the cost update rule.

Step 13 can also be seen as a variant of the Garg-K¨ onemann width reduction technique [Garg and K¨ onemann, 2007]. The ratio u t,x / EstCost x that we optimize in step 12 may be unboundedly large, so in the multiplicative update in step 13 we rescale this value to L t,x ( i ) , which is guaranteed to be at most 1; this rescaling is mirrored in the analysis of the algorithm. Interestingly, unlike the Garg-K¨ onemann algorithm which applies multiplicative updates to the dual vectors and weighted averaging to the primal ones, in our algorithm the multiplicative updates and weighted averaging are both applied to the dual vectors.

Discussion 5.2 . From the primal-dual point of view, we could distinguish a 'primal' problem in which one chooses among arms, and a 'dual' problem in which one updates the cost vector. In the primal problem, the choice of costs is deemed adversarial, and the goal is to ensure Equation (23). In the dual problem, the choice of arms is deemed adversarial, and the goal is to maximize W LCB so as to obtain Proposition 5.4. In both problems, one is agnostic as to how the upper/lower confidence bounds u t and L t,x are updated over time. As mentioned above, the dual problem falls under a standard setting of the 'best-expert' problem, and is solved via a standard algorithm for this problem. Meanwhile the primal problem is solved via bang-per-buck ratios and an ad-hoc application of the 'optimism under uncertainty' principle.

When the rewards and consumptions are deterministic, 11 the analysis is completely modular: it works no matter which algorithm is used to solve the primal (resp., dual) problem. In the general case, the primal

11 Then the dual problem maximizes W rather than W LCB , and the primal problem ensures (25) rather than (23), see Section 5.1.

algorithm also needs to ensure that the 'error terms' come out suitably small.

The following theorem expresses the regret guarantee for PrimalDualBwK .

Theorem 5.3. Consider an instance of BwK with d resources, m = | X | arms, and the smallest budget B = min i B i . The regret of algorithm PrimalDualBwK with parameter /epsilon1 = √ ln( d ) /B satisfies

Moreover, Equation (7) holds with f ( OPT LP ) equal to the right-hand side of Equation (24).

<!-- formula-not-decoded -->

The rest of the section proves this theorem. Throughout, it will be useful to represent the latent values as matrices and vectors. For this purpose, we will number the arms as X = { 1 , . . . , m } and let r ∈ R m denote the vector whose x -th component is r ( x, µ ) , the expected reward, for each arm x ∈ X . Similarly we will let C ∈ R d × m denote the matrix whose ( i, x ) entry is c i ( x, µ ) , the expected resource consumption, for each resource i and each arm x . Let e d j ∈ { 0 , 1 } d denote the d -dimensional j -th coordinate vector.

While PrimalDualBwK uses multiplicative weights update as a general technique, we make use of a specific performance guarantee in our analysis. To this end, let us recall algorithm Hedge [Freund and Schapire, 1997] from online learning theory, also known as the multiplicative weights algorithm. It is an online algorithm for maintaining a d -dimensional probability vector y while observing a sequence of d -dimensional payoff vectors π 1 , . . . , π τ . The version presented below, along with the following performance guarantee, is adapted from Kleinberg [2007]; a self-contained proof appears in Appendix C.

## Algorithm 3 Hedge with parameter /epsilon1 ∈ (0 , 1)

- 1: v 1 = 1 { v t ∈ R d + for each round t . } 2: for each round t = 1 , 2 , 3 , . . . do 3: Output distribution y t = v t / ‖ v t ‖ 1 . 4: Input payoff vector π t ∈ [0 , 1] d . 5: for each resource i do 6: v t +1 ( i ) = v t ( i ) (1 + /epsilon1 ) /lscript , /lscript = π t ( i ) .

Proposition 5.4. Fix any parameter /epsilon1 ∈ (0 , 1) and any stopping time τ . For any sequence of payoff vectors π 1 , . . . , π τ ∈ [0 , 1] d , we have

<!-- formula-not-decoded -->

## 5.1 Warm-up: The deterministic case

To present the application of Hedge to BwK in its purest form, we first consider the 'deterministic case' in which the rewards of the various arms are deterministically equal to the components of a vector r ∈ R m , and the resource consumption vectors are deterministically equal to the columns of a matrix C ∈ R d × m . Then there is no need to use upper/lower confidence bounds, so the algorithm can be simplified considerably, see Algorithm 4. In the remainder of this subsection we discuss this algorithm and analyze its regret.

Algorithm 4 is an instance of the multiplicative-weights update method for solving packing linear programs. Interpreting it through the lens of online learning, as in the survey by Arora et al. [2012], it is updating a vector y t = v t / ‖ v t ‖ 1 using the Hedge algorithm, where the payoff vector in any round t &gt; m

## Algorithm 4 Algorithm PrimalDualBwK for deterministic outcomes, with parameter /epsilon1 ∈ (0 , 1)

## 1: Initialization

- 2: In the first m rounds, pull each arm once.
- 3: For each arm x ∈ X , let r x ∈ [0 , 1] and C x ∈ [0 , 1] d
- 4: denote the reward and the resource consumption vector revealed in Step 2.
- 5: v 1 = 1 ∈ [0 , 1] d .

v

- 6: { v t ∈ [0 , 1] d is the roundt estimate of the optimal solution η ∗ to ( LP-dual ) in Section 3. }

t

(

i

)

as an estimate of the (fictional) unit cost of resource

- 9: for rounds t = m +1 , . . . , τ (i.e., until resource budget exhausted) do
- 10: For each arm x ∈ X ,
- 7: { We interpret 8: Set /epsilon1 = √ ln( d ) /B .
- 11: Expected cost for one pull of arm x is estimated by EstCost x = C x · v t .
- 13: Update estimated unit cost for each resource i :
- 12: Pull arm x = x t ∈ X that maximizes r x / EstCost x , the bang-per-buck ratio.

<!-- formula-not-decoded -->

is given by π t = C x t and the goal is to optimize the total (expected) payoff W = ∑ τ t = m +1 y t · C x t . Note that W is also the total cost consumed by Algorithm 4.

<!-- formula-not-decoded -->

To see why W is worth maximizing, let us relate it to the total reward collected by the algorithm in rounds t &gt; m ; denote this quantity by REW = ∑ τ -1 t = m +1 r t . We will prove that

For this reason, maximizing W also helps maximize REW . Proving it is a major step in the analysis.

Let ξ ∗ denote an optimal solution of the primal linear program ( LP-primal ) from Section 3, and let OPT LP = r ᵀ ξ ∗ denote the optimal value of that LP.

For each round t , let z t = e m x t denote the x t -th coordinate vector. We claim that

<!-- formula-not-decoded -->

In words: z t maximizes the 'bang-per-buck ratio' among all distributions z over arms. Indeed, the argmax in Equation (26) is well-defined as that of a continuous function on a compact set. Say it is attained by some distribution z over arms, and let ρ ∈ R be the corresponding max . By maximality of ρ , the linear inequality ρy ᵀ t Cz ≥ r ᵀ z also holds at some extremal point of the probability simplex ∆ [ X ] , i.e. at some point-mass distribution. For any such point-mass distribution, the corresponding arm maximizes the bang-per-buck ratio in the algorithm. Claim proved.

Proof of Equation (25). It follows that

<!-- formula-not-decoded -->

, for each

i

.

i

}

Here the sums are over rounds t with m &lt; t &lt; τ . Now, letting ¯ y = 1 REW ∑ t r t y t ∈ [0 , 1] d be the rewardsweighted average of distributions y m +1 , . . . , y τ , it follows that

<!-- formula-not-decoded -->

The last inequality follows because all components of Cξ ∗ are at most B by the primal feasibility of ξ ∗ .

Now, combining Equation (25) and the regret bound for Hedge , we obtain

<!-- formula-not-decoded -->

To continue this argument, we need to choose an appropriate vector y to make the right-hand side large. Recall that π t = Cz t , so ∑ m&lt;t&lt;τ π t is simply the total consumption vector in all rounds m &lt; t &lt; τ . We know some resource i must be exhausted by the time the algorithm stops, so the consumption of this resource is at least B . In a formula: ∑ τ t =1 y π t ≥ B , where y = e d i is the identity vector for resource i . Plugging in this y into Equation (27), we obtain:

<!-- formula-not-decoded -->

This completes regret analysis for the deterministic case.

## 5.2 Analysis modulo error terms

We now commence the analysis of Algorithm PrimalDualBwK . In this subsection we show how to reduce the problem of bounding the algorithm's regret to a problem of estimating two error terms that reflect the difference between the algorithm's confidence-bound estimates of its own reward and resource consumption with the empirical values of these random variables. The error terms will be treated in Section 5.3.

Recall that the algorithm computes LCBs on expected resource consumption L t,x ∈ [0 , 1] d and UCBs on expected rewards u t,x ∈ [0 , 1] , for each round t and each arm x . We also represent the LCBs as a matrix L t ∈ [0 , 1] d × m whose x -th column equals L t,x , for each arm x . We also represent the UCBs as a vector u t ∈ [0 , 1] m over arms whose x -th component equals u t,x . Let C t be the resource-consumption matrix for round t . That is, C t ∈ [0 , 1] d × m denotes the matrix whose ( i, x ) entry is the actual consumption of resource i in round t if arm x were chosen in this round.

As in the previous subsection, let z t = e m x t denote the x t -th coordinate vector, and let y t = v t / ‖ v t ‖ 1 be the vector of normalized costs. Similar to Equation (26), z t maximizes the 'bang-per-buck ratio' among all distributions z over arms:

<!-- formula-not-decoded -->

By Theorem 2.1 and our choice of C rad , it holds with probability at least 1 -T -1 that the confidence interval for every latent parameter, in every round of execution, contains the true value of that latent parameter. We call this high-probability event a clean execution of PrimalDualBwK . Our regret guarantee will

hold deterministically assuming that a clean execution takes place. The regret can be at most T when a clean execution does not take place, and since this event has probability at most T -1 it contributes only O (1) to the regret. We will henceforth assume a clean execution of PrimalDualBwK .

Claim 5.5. In a clean execution of Algorithm PrimalDualBwK with parameter /epsilon1 = √ ln( d ) /B , the algorithm's total reward satisfies the bound where E t = C t -L t and δ t = u t -r t for each round t &gt; m .

<!-- formula-not-decoded -->

Proof. The claim is proven by mimicking the analysis of Algorithm 4 in the preceding section, incorporating error terms that reflect the differences between observable values and latent ones. As before, let ξ ∗ denote an optimal solution of the primal linear program ( LP-primal ), and let OPT LP = r ᵀ ξ ∗ denote the optimal value of that LP. Let REW UCB = ∑ m&lt;t&lt;τ u ᵀ t z t denote the total payoff the algorithm would have obtained, after its initialization phase, if the actual payoff at time t were replaced with the upper confidence bound. Let y = e d i , where i is a resource exhausted by the algorithm when it stops; then y ᵀ ( ∑ τ t =1 C t z t ) ≥ B . As before,

Finally let

Assuming a clean execution, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The algorithm's actual payoff, REW = ∑ τ t =1 r ᵀ t z t , satisfies the inequality

Combining this with (32), and plugging in /epsilon1 = √ ln( d ) /B , we obtain the bound (29), as claimed.

<!-- formula-not-decoded -->

## 5.3 Error analysis

We complete the proof of Theorem 5.3 by proving upper bounds on the terms ∥ ∥ ∑ m&lt;t&lt;τ E t z t ∥ ∥ ∞ and ∣ ∣ ∑ m&lt;t&lt;τ δ t z t ∣ ∣ that appear on the right side of (29). Both bounds follow from a more general lemma which we present below.

∣ Lemma 5.6. Consider two sequences of vectors a 1 , . . . , a τ and b 1 , . . . , b τ , in [0 , 1] m , and a vector a 0 ∈ [0 , 1] m . For each arm x and each round t &gt; m , let a t,x ∈ [0 , 1] be the average observed outcome up to round t , i.e., the average outcome a s,x over all rounds s ≤ t in which arm x has been chosen by the algorithm; let N t,x be the number of such rounds. Assume that for each arm x and all rounds t with m&lt;t&lt;τ we have

The general lemma considers a sequence of vectors a 1 , . . . , a τ in [0 , 1] m and another vector a 0 ∈ [0 , 1] m . Here a t,x ∈ [0 , 1] represents a numerical outcome ( i.e., a reward or a consumption of a given resource) if arm x is pulled in round t , and a 0 ,x represents the corresponding expected outcome. Further, for each round t &gt; m we have an estimate b t ∈ [0 , 1] m for the outcome vector a t . We only assume a clean execution of the algorithm, and we derive an upper bound on ∣ ∣ ∑ m&lt;t&lt;τ ( b t -a t ) ᵀ z t ∣ .

<!-- formula-not-decoded -->

Let A = ∑ τ -1 t =1 a t,x t be the total outcome collected by the algorithm. Then

Before proving the lemma, we need to establish a simple fact about confidence radii.

<!-- formula-not-decoded -->

Claim 5.7. For any two vectors a, M ∈ R m + , we have

Proof. The definition of rad ( · , · ) implies that rad ( a x , M x ) M x ≤ √ C rad a x M x + C rad . Summing these inequalities and applying Cauchy-Schwarz,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma 5.6. For convenience, denote N t = ( N t, 1 , . . . , N t,m ) , and observe that

<!-- formula-not-decoded -->

We decompose the left side of (33) as a sum of three terms,

<!-- formula-not-decoded -->

then bound the three terms separately. The first sum is clearly bounded above by m . We next work on bounding the third sum. Let s = τ -1 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We would like to replace the expression a ᵀ 0 N s on the last line with the expression a s ᵀ N s = A . To do so, recall Equation (36) and apply the following calculation:

<!-- formula-not-decoded -->

Plugging this into Equation (37), we bound the middle sum in (35) as

Summing up the upper bounds for the three terms on the right side of (35), we obtain (33).

<!-- formula-not-decoded -->

Corollary 5.8. In a clean execution of PrimalDualBwK , and

<!-- formula-not-decoded -->

Proof. The first inequality is obtained by applying Lemma 5.6 with vector sequences a t = r t and b t = u t , and vector a 0 = r . In other words, a 0 is the vector of expected rewards across all arms.

The second inequality is obtained by applying the same lemma separately for each resource i , with vector sequences a t = ( e d i ) ᵀ C t and b t = e d i L t , and vector a 0 being the i -th row of matrix C . In other words, a 0 is the vector of expected consumption of resource i across all arms.

Proof of Theorem 5.3: If m ≥ B/ log( dT ) , then the regret bound in Theorem 5.3 is trivial. Therefore we can assume without loss of generality that m ≤ B/ log( dT ) . Therefore, recalling Equation (29), we observe that

The term m + 1 on the right side of Equation (29) is bounded above by m log( dmT ) . Finally, using Corollary 5.8 we see that the sum of the final two terms on the right side of (29) is bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The theorem follows by plugging in C rad = Θ(log( dmT )) = O (log( dT )) (because m ≤ B ≤ T ).

## 6 Lower Bound

We prove that regret (1) obtained by algorithm PrimalDualBwK is optimal up to polylog factors. Specifically, we prove that any algorithm for BwK must, in the worst case, incur regret

<!-- formula-not-decoded -->

where m = | X | is the number of arms and B = min i B i is the smallest budget.

Theorem 6.1. Fix any m ≥ 2 , d ≥ 1 , OPT ≥ m , and ( B 1 , . . . , B d ) ∈ [2 , ∞ ) . Let G be the family of all BwK problem instances with m arms, d resources, budgets ( B 1 , . . . , B d ) and optimal reward OPT . Then any algorithm for BwK must incur regret (39) in the worst case over G .

We treat the two summands in Equation (39) separately:

Claim 6.2. Consider the family G from Theorem 6.1, and let ALG be some algorithm for BwK .

Theorem 6.1 follows from Claim 6.2(ab). For part (a), we use a standard lower-bounding example for MAB. For part (b), we construct a new example, specific to BwK , and analyze it using KL-divergence.

- (a) ALG incurs regret Ω ( min ( OPT , √ m OPT )) in the worst case over G . (b) ALG incurs regret Ω ( min ( OPT , OPT √ m B )) in the worst case over G .

Proof of Claim 6.2(a). Fix m ≥ 2 and OPT ≥ m . Let G 0 be the family of all MAB problem instances with m arms and time horizon T = /floorleft 2 OPT /floorright , where the 'best arm' has expected reward µ ∗ = OPT /T and all other arms have reward µ ∗ -/epsilon1 with /epsilon1 = 1 4 √ m/T . Note that µ ∗ ∈ [ 1 2 , 3 4 ] and /epsilon1 ≤ 1 4 . It is well-known [Auer et al., 2002b] that any MAB algorithm incurs regret Ω( √ m OPT ) in the worst case over G 0 .

To ensure that G 0 ⊂ G , let us treat each MAB instance in G 0 as a BwK instance with d resources, budgets ( B 1 , . . . , B d ) , and no resource consumption.

## 6.1 The new lower-bounding example: proof of Claim 6.2(b)

Our lower-bounding example is very simple. There are m arms. Each arm gives reward 1 deterministically. There is a single resource with budget B . 12 The resource consumption, for each arm and each round, is either 0 or 1 . The expected resource consumption is p -/epsilon1 for the 'best arm' and p for all other arms, where 0 &lt; /epsilon1 &lt; p &lt; 1 . There is time horizon T &lt; ∞ . Let G ( p,/epsilon1 ) denote the family of all such problem instances, for fixed parameters ( p, /epsilon1 ) . We analyze this family in the rest of this section.

We rely on the following fact about stopping times of random sums. For the sake of completeness, we provide a proof in Section D.

Fact 6.3. Let S t be the sum of t i.i.d. 0-1 variables with expectation q . Let τ ∗ be the first time this sum reaches a given number B ∈ N . Then E [ τ ∗ ] = B/q . Moreover, for each T &gt; E [ τ ∗ ] it holds that

Infinite time horizon. It is convenient to consider the family of problem instances which is the same as G ( p,/epsilon1 ) except that it has the infinite time horizon; denote it G ∞ ( p,/epsilon1 ) . We will first prove the desired lower bound for this family, then extend it to G ( p,/epsilon1 ) .

<!-- formula-not-decoded -->

The two crucial quantities that describe algorithm's performance on an instance in G ∞ ( p,/epsilon1 ) is the stopping time and the total number of plays of the best arm. (Note that the total reward is equal to the stopping time minus 1.) The following claim connects these two quantities.

Claim 6.4 (Stopping time) . Fix an algorithm ALG for BwK and a problem instance in G ∞ ( p,/epsilon1 ) . Consider an execution of ALG on this problem instance. Let τ be the stopping time of ALG . For each round t , let N t be the number of rounds s ≤ t in which the best arm is selected. Then

<!-- formula-not-decoded -->

Proof. Let C t be the total resource consumption after round t . Note that E [ C t ] = pt -/epsilon1N t . We claim that

<!-- formula-not-decoded -->

Indeed, let Z t = C t -( pt -/epsilon1N t ) . It is easy to see that Z t is a martingale with bounded increments, and moreover that Pr[ τ &lt; ∞ ] = 1 . Therefore the Optional Stopping Theorem applies to Z t and τ , so that E [ Z τ ] = E [ Z 0 ] = 0 . Therefore we obtain Equation (40).

To complete the proof, it remains to show that C τ = /floorleft B +1 /floorright . Recall that ALG stops if and only if C t &gt; B . Since resource consumption in any round is either 0 or 1 , it follows that C τ = /floorleft B +1 /floorright .

Corollary 6.5. Consider the setting in Claim 6.4. Then:

- (a) If ALG always chooses the best arm then E [ τ ] = /floorleft B +1 /floorright / ( p -/epsilon1 ) .
- (c) p E [ τ ] -/epsilon1 E [ N τ ] = ( p -/epsilon1 ) (1 + OPT ) .
- (b) OPT = /floorleft B +1 /floorright / ( p -/epsilon1 ) -1 for any problem instance in G ∞ ( p,/epsilon1 ) .

Proof. For part(b), note that we have E [ τ ] ≤ /floorleft B +1 /floorright / ( p -/epsilon1 ) , so OPT ≤ /floorleft B +1 /floorright / ( p -/epsilon1 ) -1 . By part (a), the equality is achieved by the policy that always selects the best arm.

The heart of the proof is a KL-divergence argument which bounds the number of plays of the best arm. This argument is encapsulated in the following claim, whose proof is deferred to Section 6.3.

12 More formally, other resources in the setting of Theorem 6.1 are not consumed. For simplicity, we leave them out.

Lemma 6.6 (best arm) . Assume p ≤ 1 2 and /epsilon1 p ≤ 1 16 √ m B . Then for any BwK algorithm there exists a problem instance in G ∞ ( p,/epsilon1 ) such that the best arm is chosen at most 3 4 OPT times in expectation.

Armed with this bound and Corollary 6.5(c), it is easy to lower-bound regret over G ∞ ( p,/epsilon1 ) .

Proof. Fix any algorithm ALG for BwK . Consider the problem instance whose existence is guaranteed by Lemma 6.6. Let τ be the stopping time of ALG , and let N t be the number of rounds s ≤ t in which the best arm is selected. By Lemma 6.6 we have E [ N τ ] ≤ 3 4 OPT . Plugging this into Corollary 6.5(c) and rearranging the terms, we obtain E [ τ ] ≤ (1 + OPT )(1 -/epsilon1 4 p ) . Therefore, regret of ALG is OPT -( E [ τ ] -1) ≥ /epsilon1 4 p OPT .

Claim 6.7 (regret) . If p ≤ 1 2 and /epsilon1 p ≤ 1 16 √ m B then any BwK algorithm incurs regret /epsilon1 4 p OPT over G ∞ ( p,/epsilon1 ) .

Thus, we have proved the lower bound for the infinite time horizon.

Finite time horizon. Let us 'translate' a regret bound for G ∞ ( p,/epsilon1 ) into a regret bound for G ( p,/epsilon1 ) .

Wewill need a more nuanced notation for OPT . Consider the family of problem instances in G ( p,/epsilon1 ) ∪G ∞ ( p,/epsilon1 ) with a particular time horizon T ≤ ∞ . Let OPT ( p,/epsilon1,T ) be the optimal expected total reward for this family (by symmetry, this quantity does not depend on which arm is the best arm). We will write OPT T = OPT ( p,/epsilon1,T ) when parameters ( p, /epsilon1 ) are clear from the context.

Claim 6.8. For any fixed ( p, /epsilon1 ) and any T &gt; OPT ∞ it holds that OPT T ≥ OPT ∞ -OPT 2 ∞ /T .

Proof. Let τ ∗ be the stopping time of a policy that always plays the best arm on a problem instance in G ∞ ( p,/epsilon1 ) .

<!-- formula-not-decoded -->

The inequality is due to Fact 6.3.

Claim 6.9. Fix ( p, /epsilon1 ) and fix algorithm ALG . Let REG T be the regret of ALG over the problem instances in G ( p,/epsilon1 ) ∪ G ∞ ( p,/epsilon1 ) with a given time horizon T ≤ ∞ . Then REG T ≥ REG ∞ -OPT 2 ∞ /T.

Proof. For each problem instance I ∈ G ∞ ( p,/epsilon1 ) , let REW T ( I ) be the expected total reward of ALG on I , if the time horizon is T ≤ ∞ . Clearly, REW ∞ ( I ) ≥ REW T ( I ) . Therefore, using Claim 6.8, we have:

<!-- formula-not-decoded -->

Proof. By Claim 6.7, ALG incurs regret at least /epsilon1 4 p OPT ∞ for some problem instance in G ∞ ( p,/epsilon1 ) . By Claim 6.9, ALG incurs regret at least /epsilon1 8 p OPT ∞ for the same problem instance in G ( p,/epsilon1 ) with time horizon T . Since OPT ∞ ≥ OPT T , this regret is at least /epsilon1 8 p OPT T = Ω( OPT T ) min(1 , √ m/B ) .

Lemma 6.10 (regret: finite time horizon) . Fix p ≤ 1 2 and /epsilon1 = p 16 min(1 , √ m/B ) . Then for any time horizon T &gt; 8 p /epsilon1 OPT ∞ and any BwK algorithm ALG there exists a problem instance in G ( p,/epsilon1 ) with time horizon T for which ALG incurs regret Ω( OPT T ) min(1 , √ m/B ) .

Let us complete the proof of Claim 6.2(b). Recall that Claim 6.2(b) specifies the values for ( m,B, OPT ) that our problem instance must have. Since we have already proved Claim 6.2(a) and OPT √ m B ≤ O ( √ m OPT ) for OPT &lt; 3 B , it suffices to assume OPT ≥ 3 B .

Recall from Corollary 6.5(b) that OPT ( p,/epsilon1, ∞ ) = Γ p -1 , where

Let /epsilon1 ( p ) = p 16 min(1 , √ m/B ) , as prescribed by Lemma 6.10. Then taking /epsilon1 = /epsilon1 ( p ) we obtain regret Ω( OPT T ) min(1 , √ m/B ) for any parameter p ≤ 1 2 and any time horizon T &gt; 8 p /epsilon1 OPT ( p,/epsilon1, ∞ ) . It remains to pick such p and T so as to ensure that f ( p, T ) = OPT , where f ( p, T ) = OPT ( p,/epsilon1 ( p ) ,T ) .

<!-- formula-not-decoded -->

is a 'constant' for the purposes of this argument, in the sense that it does not depend on p or T . So we can state the sufficient condition for proving Claim 6.2(b) as follows:

<!-- formula-not-decoded -->

Recall that OPT ( p,/epsilon1, ∞ ) ≥ OPT ( p,/epsilon1,T ) for any T , and OPT ( p,/epsilon1,T ) ≥ 1 2 OPT ( p,/epsilon1, ∞ ) for any T &gt; 2 OPT ( p,/epsilon1, ∞ ) by Claim 6.8. We summarize this as follows: for any T &gt; 2( Γ p -1) ,

<!-- formula-not-decoded -->

Define p 0 = Γ / OPT . Since OPT ≥ 3 B , Γ ≤ 16 15 ( B +1) and B ≥ 4 , it follows that p 0 ≥ 1 2 . Let T = 8Γ /epsilon1 ( p 0 ) . Then Equation (42) holds for all p ∈ [ p 0 / 4 , 1 2 ] . In particular,

<!-- formula-not-decoded -->

Since f ( p, T ) is continuous in p , there exists p ∈ [ p 0 / 4 , p 0 ] such that f ( p, T ) = OPT . Since p ≤ p 0 , we have T ≥ 8Γ /epsilon1 ( p ) , satisfying all requirements in Equation (41). This completes the proof of Claim 6.2(b), and therefore the proof of Theorem 6.1.

## 6.2 Background on KL-divergence (for the proof of Lemma 6.6)

The proof of Lemma 6.6 relies on the concept of KL-divergence. Let us provide some background to make on KL-divergence to make this proof self-contained. We use a somewhat non-standard notation that is tailored to the needs of our analysis.

The KL-divergence (a.k.a. relative entropy ) is defined as follows. Consider two distributions µ, ν on the same finite universe Ω . 13 Assume µ /lessmuch ν (in words, µ is absolutely continuous with respect to ν ), meaning that ν ( w ) = 0 ⇒ µ ( w ) = 0 for all w ∈ Ω . Then KL-divergence of µ given ν is

<!-- formula-not-decoded -->

In this formula we adopt a convention that 0 0 = 1 . We will use the fact that

<!-- formula-not-decoded -->

Henceforth, let µ, ν be distributions on the universe Ω ∞ , where Ω is a finite set. For /vector w = ( w 1 , w 2 , . . . ) ∈ Ω ∞ and t ∈ N , let us use the notation /vector w t = ( w 1 , . . . , w t ) ∈ Ω t . Let µ t be a restriction of µ to Ω t : that is, a distribution on Ω t given by

<!-- formula-not-decoded -->

13 We use µ, ν to denote distributions throughout this section, whereas µ denotes the latent structure elsewhere in the paper.

The next-round conditional distribution of µ given /vector w t , t &lt; T is defined by

<!-- formula-not-decoded -->

Note that µ ( · | /vector w t ) is a distribution on Ω for every fixed /vector w t .

The conditional KL-divergence at round t +1 is defined as

<!-- formula-not-decoded -->

In words, this is the KL-divergence between the next-round conditional distributions µ ( · | /vector w t ) and ν ( · | /vector w t ) , in expectation over the random choice of /vector w t according to distribution µ t .

We will use the following fact, known as the chain rule for KL-divergence:

<!-- formula-not-decoded -->

Here for notational convenience we define KL 1 ( µ ‖ ν ) /defines KL ( µ 1 ‖ ν 1 ) .

## 6.3 The KL-divergence argument: proof of Lemma 6.6

Fix some BwK algorithm ALG and fix parameters ( p, /epsilon1 ) . Let I x be the problem instance in G ∞ ( p,/epsilon1 ) in which the best arm is x . For the analysis, we also consider an instance I 0 which coincides with I x but has no best arm: that is, all arms have expected resource consumption p . Let τ ( I ) be the stopping time of ALG for a given problem instance I , and let N x ( I ) be the expected number of times a given arm x is chosen by ALG on this problem instance.

Consider problem instance I 0 . Since all arms are the same, we can apply Corollary 6.5(a) (suitably modified to the non-best arm) and obtain E [ τ ( I 0 )] = /floorleft B +1 /floorright /p . We focus on an arm x with the smallest N x ( I 0 ) . For this arm it holds that

In what follows, we use this inequality to upper-bound N x ( I x ) . Informally, if arm x is not played sufficiently often in I 0 , ALG cannot tell apart I 0 and I x .

<!-- formula-not-decoded -->

The transcript of ALG on a given problem instance I is a sequence of pairs { ( x t , c t ) } t ∈ N , where for each round t ≤ τ ( I ) it holds that x t is the arm chosen by ALG and c t is the realized resource consumption in that round. For all t &gt; τ ( I ) , we define ( x t , c t ) = ( null , 0) . To map this to the setup in Section 6.2, denote Ω = ( X ∪ { null } ) ×{ 0 , 1 } . Then the set of all possible transcripts is a subset of Ω ∞ .

Every given problem instance I induces a distribution over Ω ∞ . Let µ, ν be the distributions over Ω ∞ that are induced by I 0 and I x , respectively. We will use the following shorthand:

<!-- formula-not-decoded -->

For any T ∈ N (which we will fix later), we can write

<!-- formula-not-decoded -->

We will bound diff [1 , T ] and diff [ T +1 , ∞ ] separately.

Upper bound on diff [1 , T ] . This is where we use KL-divergence. Namely, by Equation (43) we have

<!-- formula-not-decoded -->

Now, by the chain rule (Equation (44)), we can focus on upper-bounding the conditional KL-divergence KL t ( µ ‖ ν ) at each round t ≤ T .

Claim 6.11. For each round t ≤ T it holds that

Proof. The main difficulty here is to carefully 'unwrap' the definition of KL t ( µ ‖ ν ) .

<!-- formula-not-decoded -->

Fix t ≤ T and let /vector w t ∈ Ω t be the partial transcript up to and including round t . For each arm y , let f ( y | /vector w t ) be the probability that ALG chooses arm y in round t , given the partial transcript /vector w t . Let c ( y |I ) be the expected resource consumption for arm y under a problem instance I . The transcript for round t +1 is a pair w t +1 = ( x t +1 , c t +1 ) , where x t +1 is the arm chosen by ALG in round t +1 , and c t +1 ∈ { 0 , 1 } is the resource consumption in that round. Therefore if c t +1 = 1 then

<!-- formula-not-decoded -->

Similarly, if c t +1 = 0 then

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

Taking expectations over w t +1 = ( x t , c t ) ∼ µ ( · | /vector w t ) , we obtain

Taking expectations over /vector w t ∼ µ t , we obtain the conditional KL-divergence KL t ( µ ‖ ν ) . Equation (48) follows because

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will use the following fact about logarithms, which is proved using standard quadratic approximations for the logarithm. The proof is in Section D.

Fact 6.12. Assume /epsilon1 p ≤ 1 2 and p ≤ 1 2 . Then

<!-- formula-not-decoded -->

Now we can put everything together and derive an upper bound on diff [1 , T ] .

<!-- formula-not-decoded -->

Proof. By Claim 6.11 and Fact 6.12, for each round t ≤ T we have

<!-- formula-not-decoded -->

By the chain rule (Equation (44)), we have

<!-- formula-not-decoded -->

The last inequality is the place where we use our choice of x , as expressed by Equation (45).

<!-- formula-not-decoded -->

Upper bound on diff [ T, ∞ ] . Consider the problem instance I x , and consider the policy that always chooses the best arm. Let ν ∗ be the corresponding distribution over transcripts Ω ∞ , and let τ be the corresponding stopping time. Note that ν ∗ ( x t = x ) if and only if τ &gt; t . Therefore:

<!-- formula-not-decoded -->

The second inequality can be proved using a simple 'coupling argument'. The last inequality follows from Fact 6.3, observing that E [ τ ] = OPT .

<!-- formula-not-decoded -->

Putting the pieces together. Assume p ≤ 1 2 and /epsilon1 p ≤ 1 2 . Denote γ = /epsilon1 p √ B +1 m . Using the upper bounds on diff [1 , T ] and diff [ T +1 , ∞ ] and plugging them into Equation (46), we obtain for T = OPT / √ γ . Recall that N x ( I 0 ) &lt; OPT /m . Thus, we obtain

<!-- formula-not-decoded -->

Recall that we need to conclude that N x ( I x ) ≤ 3 4 OPT . For that, it suffices to have γ ≤ 1 16 .

## 7 BwK with preadjusted discretization

In this section we develop a general technique for preadjusted discretization, and apply it to dynamic pricing with a single product and dynamic procurement with a single budget. For both applications, our regret bounds significantly improve over prior work. While the dynamic pricing application is fairly straightforward given the general result, the dynamic procurement application takes some work and uses a non-standard mesh of prices. We also obtain an initial result for dynamic pricing with multiple products. The main technical challenge is to upper-bound the discretization error; we can accomplish this whenever the expected resource to expected consumption ratio of each arm can be expressed in a particularly simple way.

## 7.1 Preadjusted discretization as a general technique

The high-level idea behind preadjusted discretization is to apply an existing BwK algorithm with a restricted, finite action space S ⊂ X that is chosen in advance. Typically S is, in some sense, 'uniformly spaced' in X , and its 'granularity' is tuned in advance so as to minimize regret.

Consider a problem instance with action space restricted to S . Let REW ( S ) be the algorithm's reward on this problem instance, and let OPT LP ( S ) be the corresponding value of OPT LP , as defined in Section 3. OPT ( X ) and OPT LP ( X ) will refer to the corresponding quantities for the original action space X . The key two quantities in our analysis of preadjusted discretization are

<!-- formula-not-decoded -->

Note that algorithm's regret can be expressed as

<!-- formula-not-decoded -->

Now, suppose S is parameterized by /epsilon1 &gt; 0 which controls its 'granularity'. Adjusting the /epsilon1 involves balancing R ( S ) and Err ( S | X ) : indeed, decreasing /epsilon1 tends to increase R ( S ) but decrease Err ( S | X ) . We upper-bound the S -regret via our main algorithmic result; 14 the challenge is to upper-bound Err ( S | X ) .

While in practice the action set X is usually finite (although possibly very large), it is mathematically more elegant to consider infinite X . For example, we prefer to allow arbitrary fractional prices, even though in practice they may have to be rounded to whole cents. However, recall that OPT LP in Section 3 is only defined for a finite action space X . To handle infinite X , we define

A typical scenario where one would want to apply preadjusted discretization is when an algorithm chooses among prices. More formally, each arm includes a real-valued vector of prices in [0 , 1] (and perhaps other things, such as the maximal number of items for sale). The restricted action set S consists of all arms such that all prices belong to a suitably chosen mesh M ⊂ [0 , 1] with granularity /epsilon1 . There are several types of meshes one could consider, depending on the particular BwK domain. The most natural ones are the /epsilon1 -additive mesh , with prices that are integer multiples of /epsilon1 , and /epsilon1 -multiplicative mesh mesh , with prices of the form (1 -/epsilon1 ) /lscript , /lscript ∈ N . Both have been used in the prior work on MAB in metric spaces [Kleinberg, 2004, Hazan and Megiddo, 2007, Kleinberg et al., 2008, Lu et al., 2010]) and dynamic pricing (e.g., [Kleinberg and Leighton, 2003, Blum et al., 2003, Besbes and Zeevi, 2009, Babaioff et al., 2015]). Somewhat surprisingly, for dynamic procurement we find it optimal to use a very different mesh, called /epsilon1 -hyperbolic mesh , in which the prices are of the form 1 1+ /epsilon1/lscript , /lscript ∈ N .

<!-- formula-not-decoded -->

In line with Lemma 3.1, let us argue that OPT LP ( X ) ≥ OPT ( X ) even when X is infinite. Specifically, we prove this for all versions of dynamic pricing and dynamic procurement, and more generally for any BwK domain such that for each arm there are only finitely many possible outcome vectors.

Lemma 7.1. Consider a BwK domain with infinite action space X , such that for each arm there are only finitely many possible outcome vectors. Then OPT LP ( X ) ≥ OPT ( X ) .

Proof. Fix a problem instance, and consider an optimal dynamic policy for this instance. W.l.o.g. this policy is deterministic. 15 For each round, this policy defines a deterministic mapping from histories to arms to be played in this round. Since there are only finitely many possible histories, the policy can only use a finite subset of arms, call it X ′ ⊂ X . By Lemma 3.1, we have

<!-- formula-not-decoded -->

14 We need to use the regret bound in terms of the best known upper bound on OPT , rather than OPT itself, because the latter is not known to the algorithm. For example, for dynamic pricing one can use OPT ≤ B .

15 A randomized policy can be seen as a distribution over deterministic policies, so one of these deterministic policies must have same or better expected total reward.

## 7.2 A general bound on discretization error

We develop a general bound on discretization error Err ( S | X ) , as defined in Equation (49). To this end, we consider the expected reward to expected consumption ratios of arms (and the differences between them), whereas in the work on MAB in metric spaces it suffices to consider the difference in expected rewards.

To simplify notation, we suppress µ , the (actual) latent structure: e.g., we will write c i ( D ) = c i ( D , µ ) , r ( D ) = r ( D , µ ) , and LP ( D , µ ) = LP ( D ) for distributions D and resources i .

Definition 7.2. We say that arm x /epsilon1 -covers arm y if the following two properties are satisfied for each resource i such that c i ( x ) + c i ( y ) &gt; 0 :

- (i) r ( x ) /c i ( x ) ≥ r ( y ) /c i ( y ) -/epsilon1 .
- (ii) c i ( x ) ≥ c i ( y ) .

A subset S ⊂ X of arms is called an /epsilon1 -discretization of X if each arm in X is /epsilon1 -covered by some arm in S .

Theorem 7.3 (preadjusted discretization) . Fix a BwK domain with action space X . Let S ⊂ X be an /epsilon1 -discretization of X , for some /epsilon1 ≥ 0 . Then the discretization error Err ( S | X ) is at most /epsilon1dB . Consequently, for any algorithm with S -regret R ( S ) we have OPT LP ( X ) -REW ( S ) = R ( S ) + /epsilon1dB .

Proof. We need to prove that Err ( S | X ) ≤ /epsilon1dB . If X is infinite, then (by Equation (50)) it suffices to prove Err ( S | X ′ ) ≤ /epsilon1dB for any finite subset of X ′ ⊂ X . Let D be the distribution over arms in X ′ which maximizes LP ( D , µ ) . We use D to construct a distribution D S over S which is nearly as good.

We define D S as follows. Since S is an /epsilon1 -discretization of X , there exists a family of subsets ( cov ( x ) ⊂ X : x ∈ S ) so that each arm x ∈ S /epsilon1 -covers all arms in cov ( x ) , the subsets are disjoint, and their union is X . Fix one such family of subsets, and define

<!-- formula-not-decoded -->

To argue that LP ( D S , µ ) is large, we upper-bound the resource consumption c i ( D S ) , for each resource i , and lower-bound the reward r ( D S ) .

Note that ∑ x ∈ S D S ( S ) ≤ 1 by Definition 7.2(ii). With the remaining probability, the null arm is chosen (i.e., the algorithm skips a given round).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(Note that the above argument did not use the property (i) in Definition 7.2.)

In what follows, for each arm x define I x = { i : c i ( x ) &gt; 0 } .

Let τ ( D ) = min i B c i ( D ) be the stopping time in the linear relaxation, so that LP ( D ) = τ ( D ) r ( D ) . By Equation (51) we have τ ( D S ) ≥ τ ( D ) . We are ready for the final computation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 7.3 Preadjusted discretization for dynamic pricing

We apply the machinery developed above to handle the basic version of dynamic pricing, as defined in Section 1.1. In fact, our technique easily generalizes to multiple products, in a particular scenario which we call dynamic bundle-pricing . We present the more general result directly.

The dynamic bundle-pricing problem is defined as follows. There are d products, with limited supply of each, and T rounds. In each round, a new buyer arrives, an algorithm chooses a bundle of products and a price, and offers this bundle for this price. The offer is either accepted or rejected. The bundle is a vector ( b 1 , . . . , b d ) , so that b i ∈ N units of each product i are offered. We assume that the bundle must belong to a fixed collection F of allowed bundles. Buyers' valuations over bundles can be arbitrary (in particular, not necessarily additive); they are drawn independently from a fixed distribution over valuations. For normalization, we assume that each buyer's valuation for any bundle of /lscript units lies in the interval [0 , /lscript ] ; accordingly, the offered price for such bundle can w.l.o.g. be restricted to the same interval.

Theorem 7.4. Consider the dynamic bundle-pricing problem such that there are d products, each with supply B . Assume each allowed bundle consists of at most /lscript items, and prices are in [0 , /lscript ] . Algorithm PrimalDualBwK with an /epsilon1 -additive mesh, for some /epsilon1 = /epsilon1 ( B, |F| , /lscript ) , has regret O ( dB 2 / 3 ( |F| /lscript ) 1 / 3 ) .

˜ Corollary 7.5. Consider the dynamic pricing problem, as defined Section 1.1. Algorithm PrimalDualBwK with an /epsilon1 -additive mesh, for a suitably chosen /epsilon1 = /epsilon1 ( B ) , has regret ˜ O ( B 2 / 3 ) .

˜ The basic version from Section 1.1 is a special case with a single product and a single allowed bundle which consists of one unit of this product. Taking d = /lscript = |F| = 1 in Theorem 7.4, we obtain regret O ( B 2 / 3 ) . This regret bound is optimal for any pair ( B,T ) , as proved in Babaioff et al. [2015].

Proof of Theorem 7.4. First, let us cast this problem as a BwK domain. To ensure that per-round rewards and per-round resource consumptions lie in [0 , 1] , we scale them down by the factor of /lscript . Accordingly, the rescaled supply constraint is B ′ = B//lscript . In what follows, consider the scaled-down problem instance.

An arm is a pair x = ( b, p ) , where b ∈ F is a bundle and p ∈ [0 , 1] is the offered price. Let F ( x ) be the probability of a sale for this arm, divided by /lscript ; this probability is non-increasing in p for a fixed bundle b . Then expected per-round reward is r ( x ) = p F ( x ) , and expected per-round consumption of product i is c i ( x ) = b i F ( x ) . Therefore,

<!-- formula-not-decoded -->

This is a crucial domain-specific property that enables preadjusted discretization. It follows that for any arm x = ( b, p ) , this arm /epsilon1 -covers any arm x ′ = ( b, p ′ ) such that p ′ -/epsilon1 ≤ p ≤ p ′ . Therefore an /epsilon1 -additive mesh S is an /epsilon1 -discretization, for any /epsilon1 &gt; 0 .

Consider algorithm PrimalDualBwK with action space S . Using Theorem 5.3 and observing that OPT LP ≤ dB ′ , we obtain S -regret

By Theorem 7.3 discretization error is Err ( S | X ) ≤ /epsilon1dB ′ . So, regret relative to OPT LP is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for a suitably chosen /epsilon1 = ( B//lscript ) -1 / 3 |F| 1 / 3 . Recall that this is regret for the rescaled problem instance. For the original problem instance, rewards are scaled up by the factor of /lscript , so regret is scaled up by /lscript , too.

One can easily extend Theorem 7.4 to a setting where in each round an algorithm offers several copies of the same bundle for the same per-bundle price, and an agent can choose how many copies to buy (if any). More precisely, in each round an algorithm chooses two things: a bundle from F and the number of copies of this bundle. The latter is restricted to be at most Λ , where Λ is a known parameter. We call this setting dynamic bundle-pricing with multiplicity Λ . The algorithm and analysis is essentially the same.

Theorem 7.6. Consider dynamic bundle-pricing with multiplicity Λ . Assume that each product has supply B , and each allowed bundle consists of at most /lscript items. Algorithm PrimalDualBwK with an /epsilon1 -additive mesh, for a suitably chosen /epsilon1 = /epsilon1 ( B, |F| , /lscript, Λ) , has regret ˜ O ( d ( B Λ) 2 / 3 ( |F| /lscript ) 1 / 3 ) .

## 7.4 Preadjusted discretization for dynamic procurement

Application to dynamic procurement takes a little more work and results in a weaker regret bound, compared to the application to dynamic pricing. The main reason is that the natural mesh for dynamic procurement is /epsilon1 -hyperbolic (rather than /epsilon1 -additive). One needs to bound this mesh from below to make it finite, which increases the mesh size and the discretization error.

While our main goal here is to handle the basic version of dynamic procurement, as defined in Section 1.1, the same technique easily extends to a generalization where the algorithm can buy multiple items in each round. The generalization is defined as follows. In each round t , the algorithm offers to buy up to Λ units at price p t per unit, where p t ∈ [0 , 1] is chosen by the algorithm. The outcome is summarized by the number k t of items bought, where k t is an independent sample from some fixed (but unknown) distribution parameterized by p t and Λ . The algorithm is constrained by the time horizon T , budget B , and per-round supply constraint Λ . We prove the following:

Theorem 7.7. Consider dynamic procurement with up to Λ items bought per round. Algorithm PrimalDualBwK with a suitably chosen action space S yields regret ˜ O (Λ 5 / 4 T/B 1 / 4 ) . Specifically, S = [ p 0 , 1] ∩ M , where M is the /epsilon1 -hyperbolic mesh, for some parameters /epsilon1,p 0 ∈ (0 , 1) that depend only on B , T and Λ .

Let us model this problem as a BwK domain. The action space is X = [0 , 1] : the arms correspond to all possible prices. (The zero price corresponds to the 'null arm'.) To ensure that rewards and consumptions lie in [0 , 1] , we scale them down by a factor of Λ , as in Section 7.3, so that reward in any round t is k t / Λ , and budget consumption is p t k t / Λ . Accordingly, budget is rescaled to B ′ = B/ Λ . Henceforth, consider the scaled-down problem instance, unless specified otherwise.

Let F ( p ) be the expected per-round number of items sold for a given price p , divided by Λ ; note that it is non-decreasing in p . Then expected budget consumption is c ( p ) = p F ( p ) , and expected reward is simply r ( p ) = F ( p ) . It follows that

<!-- formula-not-decoded -->

Like Equation (53), this is a crucial domain-specific property that enables preadjusted discretization.

By Definition 7.2 price p /epsilon1 -covers price arm q if and only if q &lt; p and 1 p ≥ 1 q -/epsilon1 . This makes the hyperbolic mesh a natural mesh for this problem, rather than additive or multiplicative ones. It is easy to see that the /epsilon1 -hyperbolic mesh S on X is an /epsilon1 -discretization of X : namely, each price q is /epsilon1 -covered by the smallest price p ≥ q that lies in S .

Unfortunately, this mesh has infinitely many points. In fact, it is easy to see that any /epsilon1 -discretization on X must be infinite, even for Λ = 1 . To obtain a finite /epsilon1 -discretization, we only consider prices p ≥ p 0 , for some parameter p 0 to be tuned later. Below we argue that this restriction is not too damaging:

Claim 7.8. Consider dynamic procurement with non-unit supply. Then for any p 0 ∈ (0 , 1) it holds that

<!-- formula-not-decoded -->

Proof. When p 0 &gt; B ′ /T the bound is trivial and for the rest of the proof we assume that p 0 ≤ B ′ /T .

<!-- formula-not-decoded -->

By Equation (50) it suffices to replace OPT LP ([0 , 1]) in the claim with OPT LP ( X 0 ) , for any given finite subset X 0 ⊂ [0 , 1] . Let D be an LP-perfect distribution for the problem instance restricted to X 0 ; such D exists by Claim 3.4. Thus, LP ( D ) = OPT LP ( X ) and c ( D ) ≤ B ′ T . Furthermore, D has a support of size at most 2 ; denote it as arms p 1 , p 2 ∈ [0 , 1] , p 1 ≤ p 2 , where the null arm would correspond to p 1 = 0 . If p 1 ≥ p 0 then D has support in the interval [ p 0 , 1] , and we are done; so from here on we assume p 1 &lt; p 0 . Note that

To prove the desired lower bound on OPT LP ([ p 0 , 1]) , we construct a distribution D ′ with support in { 0 } ∪ [ p 0 , 1] and a sufficiently large LP-value. (Here the zero price corresponds to the null arm.) Suppose p B ′ . Define ′ by putting probability mass on price B ′ . Since c ( B ′ ) B ′ , we have

2 ≤ T D T T ≤ T

<!-- formula-not-decoded -->

and we are done. From here on, assume p 2 &gt; B ′ T .

Now consider the main case: p 1 ≤ p 0 ≤ B ′ T &lt; p 2 . Define distribution D ′ as follows:

<!-- formula-not-decoded -->

We claim that c ( D ′ ) ≤ B ′ T . If D ′ ( p 2 ) = 0 then c ( D ′ ) = c ( p 0 ) ≤ p 0 ≤ B ′ T . If D ′ ( p 2 ) &gt; 0 then D ′ ( p 2 ) = D ( p 2 ) -p 0 /p 2 , and therefore,

<!-- formula-not-decoded -->

Then c ( D ′ ) ≤ c ( D ) ≤ B ′ T . Claim proved.

Therefore, LP ( D ′ ) = T r ( D ′ ) . To complete the proof:

<!-- formula-not-decoded -->

Suppose algorithm PrimalDualBwK is applied to a problem instance with a finite action space S . Then by Theorem 5.3 the S -regret is

R ( S ) = ˜ O ( √ mT + T √ m/B ′ ) , m = | S | . Let S = [ p 0 , 1] ∩ M , where M is the /epsilon1 -hyperbolic mesh, for some /epsilon1,p 0 ∈ (0 , 1) . Then m = | S | ≤ 1 /epsilon1p 0 . Moreover, S is an /epsilon1 -discretization for action space X ′ = [ p 0 , 1] , for the same reason that M is an /epsilon1 -discretization for the original action space X = [0 , 1] . Therefore:

<!-- formula-not-decoded -->

Optimizing the choice of /epsilon1 and p 0 , we obtain the final regret bound of ˜ O ( T ( B ′ ) -1 / 4 ) . Recall that this is the regret bound for the rescaled problem instance. Going back to the original problem instance, regret is multiplied by a factor of Λ . This completes the proof of Theorem 7.7.

## 8 Applications and corollaries

We systematically overview various applications of BwK and corresponding corollaries. This section can be read independently of the technical material in the rest of the paper.

Some technicalities. In applications with very large or infinite action space X we apply a BwK algorithm with a restricted, finite action space S ⊂ X , where S is chosen in advance. Immediately, we obtain a bound on the S -regret : regret with respect to the value of OPT LP on the restricted action space (such bound depends on | S | ). Instantiating such regret bounds is typically straightforward once one precisely defines the setting. In some applications we can choose S using preadjusted discretization, as discussed in Section 7.

In some of the applications, per-round reward and resource consumption may be larger than 1 . Then one needs to scale them down to fit the definition of BwK and apply our regret bounds, and scale them back up to obtain regret for the original (non-rescaled) version. We encapsulate this argument as follows:

Lemma 8.1. Consider a version of BwK with finite action set S , in which per-round rewards are upperbounded by r 0 , and per-round consumption of each resource is at most c 0 . Then one can achieve regret by applying algorithm PrimalDualBwK with suitably rescaled rewards, resource consumption, and budgets.

<!-- formula-not-decoded -->

Proof. Denote R ( OPT , B ) = √ | S | OPT + OPT √ | S | /B , as in the main regret bound. To cast this problem as an instance of BwK , consider a rescaled problem instance in which all rewards are divided by r 0 , and all consumptions and budgets are divided by c 0 . Now we can apply regret bound Equation (1) for the scaled-down problem instance; we obtain regret ˜ O ( R ( OPT /r 0 , B/c 0 )) . Multiply this regret bound by r 0 to obtain a regret bound for the original problem instance.

## 8.1 Dynamic pricing with limited supply

In dynamic pricing, the algorithm is a monopolist seller that interacts with T agents (potential buyers) arriving one by one. In each round, a new agent arrives, the algorithm makes an offer, the agent chooses among the offered alternatives, and leaves. The offer specifies which goods are offered for sale at which prices. The agent has valuations over the offered bundles of goods, and chooses an alternative which maximizes her utility: value of the bundle minus the price. An agent is characterized by her valuation function : function from all possible bundles of goods that can be offered to their respective valuations. For each arriving agent, the valuation function is private : not known to the algorithm. It is assumed to be drawn from a fixed (but unknown) distribution over the possible valuation functions, called the demand distribution . Algorithm's objective is to maximize the total revenue; there is no bonus for left-over inventory.

Basic version. In the basic version from Section 1.1, the algorithm has B identical items for sale. In each round, the algorithm chooses a price p t and offers one item for sale at this price, and an agent either buys or leaves. The agent has a fixed private value v t ∈ [0 , 1] for an item, and buys if and only if p t ≥ v t . Recall from Corollary 7.5 that we obtain regret O ( B 2 / 3 ) , which is optimal according to [Babaioff et al., 2015].

˜ Extension: multiple products. When multiple products are offered for sale, it often makes sense to price them jointly. Formally, the algorithm has d products for sale, with B i units of each product i . (To simplify regret bounds, let us assume B i = B .) In each round t , the algorithm chooses a vector of prices ( p t, 1 , . . . , p t,d ) ∈ [0 , 1] d and offers at most one unit of each product i at price p t,i . The agent then chooses the subset of products to buy. We allow arbitrary demand distributions; we do not restrict correlations between valuations of different products and/or subsets of products.

˜ Extension: non-unit demands. Agents may be interested in buying more than one unit of the product, and may have valuations that are non-linear in the number of products bought. Accordingly, let us consider an extension where an algorithm can offer each agent multiple units. More specifically: in each round t , the algorithm offers up to λ t units at a fixed price p t per unit, where the pair ( p t , λ t ) is chosen by the algorithm, and the agent then chooses how many units to buy, if any. We restrict λ t ≤ Λ , where Λ is a fixed parameter. We obtain regret ˜ O ( B Λ) 2 / 3 by Theorem 7.6 (considering a special case when there is a single product and a single allowed bundle with one unit of this product). One can also consider a version with λ t = Λ , so that the algorithm only chooses prices; then a very similar argument gives regret O ( B 2 / 3 Λ 1 / 3 ) .

Given a finite set S of allowed price vectors, such as an /epsilon1 -additive mesh for some specific /epsilon1 &gt; 0 , we obtain S -regret ˜ O ( d √ B | S | ) . This follows from Lemma 8.1, observing that per-round rewards are at most r 0 = d , per-round consumption of each resource is at most c 0 = 1 , and the optimal value is OPT ≤ dB .

There may also be a fixed collection of subsets that agents are allowed to buy, e.g., agents may be restricted to buying at most three items in total. This does not affect our analysis and the regret bound.

Joint pricing is not needed in the special case when each agent can buy an arbitrary subset I of products, and her valuations are additive: v ( I ) = ∑ i ∈ I v ( i ) . Then she buys each product i if and only if the offered price for this product exceeds v ( i ) . Therefore the problem is equivalent to a collection of d separate perproduct problems, and one can run a separate BwK algorithm for each product. Using Corollary 7.5 separately for each product, one obtains regret O ( dB 2 / 3 ) .

˜ Extension: network revenue management. More generally, an algorithm may have d products for sale which may be produced on demand from limited primitive resources , so that each unit of each product i consumes a fixed and known amount c ij ∈ [0 , 1] of each primitive resource j . This generalization is known as network revenue management problem (see Besbes and Zeevi [2012] and references therein). All other details are the same as above; for simplicity, let us focus on a version in which each agent buys at most one item. Given a finite set S of allowed price vectors, we obtain S -regret given by (54) with r 0 = c 0 = d .

In particular, if all resource constraints (including the time horizon) are scaled up by factor γ , regret scales as √ γ . This improves over the main result in Besbes and Zeevi [2012], where (essentially) regret is stated in terms of γ and scales as γ 2 / 3 .

Extension: bundling and volume pricing. When selling to agents with non-unit demands, an algorithm may use discounts and/or surcharges for buying multiple units of a product (the latter may make sense for high-valued products such as tickets to events at the Olympics). More generally, an algorithm can may use discounts and/or surcharges for some bundles of products, where each bundle can include multiple units of multiple products, e.g., two beers and one snack. In full generality, there is a collection F of allowed bundles. In each round an algorithm offers a menu of options which consists of a price for every allowed bundle in F (and the 'none' option), and the agent chooses one option from this menu. Thus, in each round the algorithm needs to choose a price vector over the allowed bundles.

For a formal result, assume there is a finite set S of allowed price vectors, each bundle in F can contain at most /lscript units total, and the per-bundle prices are restricted to lie in the range [0 , /lscript ] . Then we obtain S -regret ˜ O ( d/lscript √ /lscriptB | S | ) . This follows from Lemma 8.1, observing that per-round rewards are at most r 0 = /lscript , per-round consumption of each resource is at most c 0 = /lscript , and the optimal value is OPT ≤ d/lscriptB .

The action space here is |F| -dimensional, which may result in a prohibitively large number of allowed price vectors. One can reduce the 'dimensionality' of the action space by restricting how the bundles may be priced. For example, each bundle may be priced at a volume discount x % compared to buying each unit separately, where x depends only on the number of items in the bundle.

Moreover, we can analyze preadjusted discretization for a version where in each round the algorithm chooses only one bundle to offer. By Theorem 7.4, we obtain regret O ( B 2 / 3 ( |F| /lscript ) 1 / 3 ) .

˜ Extension: buyer targeting. Suppose there are /lscript different types of buyers (say men and women ), and the demand distribution of a buyer depends on her type. The buyer type is modeled as a sample from a fixed but unknown distribution. In each round the seller observes the type of the current buyer (e.g., using a cookie or a user profile), and can choose the price depending on this type.

This can be modeled as a BwK domain where arms correspond to functions from buyer types to prices. For example, with /lscript buyer types and a single product, the (full) action space is X = [0 , 1] /lscript . Assuming we are given a restricted action space S ⊂ X , we obtain S -regret ˜ O ( √ B | S | ) .

## 8.2 Dynamic procurement and crowdsourcing markets

A 'dual' problem to dynamic pricing is dynamic procurement , where the algorithm is buying rather than selling. In the basic version, the algorithm has a budget B to spend, and is facing T agents (potential sellers) that are arriving sequentially. In each round t , a new agent arrives, the algorithm chooses a price p t ∈ [1] and offers to buy one item at this price. The agent has private value v t ∈ [0 , 1] for an item (unknown

to the algorithm), and sells if and only if p t ≥ v t . The value is an independent sample from some fixed (but unknown) distribution. Algorithm's goal is to maximize the number of items bought. Recall from Theorem 7.7 that we obtain regret O ( T/B 1 / 4 ) for this version.

˜ Application to crowdsourcing markets. The problem is particularly relevant to the emerging domain of crowdsourcing , where agents correspond to the (relatively inexpensive) workers on a crowdsourcing platform such as Amazon Mechanical Turk, and 'items' bought/sold correspond to simple jobs ('microtasks') that can be performed by these workers. The algorithm corresponds to the 'requester': an entity that submits jobs and benefits from them being completed. The (basic) dynamic procurement model captures an important issue in crowdsourcing that a requester interacts with multiple users with unknown values-per-item, and can adjust its behavior (such as the posted price) over time as it learns the distribution of users. While this basic model ignores some realistic features of crowdsourcing environments (see a survey Slivkins and Vaughan [2013] for background and discussion), some of these limitations are addressed by the generalizations which we present below.

Extension: non-unit supply. We consider an extension where agents may be interested in more than one item, and their valuations may be non-linear. For example, a worker may be interested in performing several jobs. In each round t , the algorithm offers to buy up to Λ units at a fixed price p t per unit, where the price p t is chosen by the algorithm and Λ is a fixed parameter. The t -th agent then chooses how many units to sell. Recall from Theorem 7.7 that we obtain regret ˜ O (Λ 5 / 4 T/B 1 / 4 ) for this extension.

Extension: multiple types of jobs . Wecan handle an extension in which there are d types of jobs requested on the crowdsourcing platform, with a separate budget B i for each type. Each agent t has a private cost v t,i ∈ [0 , 1] for each type i ; the vector of private costs comes from a fixed but unknown distribution over d -dimensional vectors (note that arbitrary correlations are allowed). The algorithm derives reward u i ∈ [0 , 1] from each job of type i . In each round t , the algorithm offers a vector of prices ( p t, 1 , . . . , p t,d ) , where p t,i is the price for one job of type i . For each type i , the agent performs one job of this type if and only if p t,i ≥ v t,i , and receives payment p t,i from the algorithm.

Here arms correspond to the d -dimensional vectors of prices, so that the action space is X = [0 , 1] d . Given the restricted action space S ⊂ X , we obtain S -regret ˜ O ( d )( √ T | S | + T √ d | S | /B ) , where B is the smallest budget. This follows from Lemma 8.1, observing that per-round rewards are at most r 0 = d , per-round consumption of each budget is at most c 0 = 1 , and the optimal value is OPT ≤ dT .

Extension: additional features. We can also model more complicated 'menus' so that each agent can perform several jobs of the same type. Then in each round, for each type i , the algorithm specifies the maximal offered number of jobs of this type and the price per one such job. We can also incorporate constraints on the maximal number of jobs of each type that is needed by the requester, and/or the maximal amount of money spend on each type.

Extension: competitive environment. There may be other requesters in the system, each offering its own vector of prices in each round. (This is a realistic scenario in crowdsourcing, for example.) Each seller / worker chooses the requester and the price that maximize her utility. One standard way to model such a competitive environment is to assume that the 'best offer' from the competitors is a vector of prices which comes from a fixed but unknown distribution. This can be modeled as a BwK instance with a different distribution over outcomes which reflects the combined effects of the demand distribution of agents and the 'best offer' distribution of the environment.

## 8.3 Other applications to Electronic Markets

Ad allocation with unknown click probabilities. Consider pay-per-click (PPC) advertising on the web (in particular, this is a prevalent model in sponsored search auctions). The central premise in PPC advertising is that an advertiser derives value from her ad only when the user clicks on this ad. The ad platform allocates ads to users that arrive over time.

Consider the following simple (albeit highly idealized) model for PPC ad allocation. Users arrive over time, and the ad platform needs to allocate an ad to each arriving user. There is a set X of available ads. Each ad x is characterized by the payment-per-click π x and click probability µ x ; the former quantity is known to the algorithm, whereas the latter is not. If an ad x is chosen, it is clicked on with probability µ x , in which case payment π x is received. The goal is to maximize the total payment. This setting and various extensions thereof that incorporate user/webpage context have received a considerable attention in the past several years (starting with [Pandey et al., 2007a,b, Langford and Zhang, 2007]). In fact, the connection to PPC advertising has been one of the main motivations for the recent surge of interest in MAB.

We enrich the above setting by incorporating advertisers' budgets . In the most basic version, for each ad x there is a budget B x -the maximal amount of money that can be spent on this ad. More generally, an advertiser can have an ad campaign which consists of a subset S of ads, so that there is a per-campaign budget S . Even more generally, an advertiser can have a more complicated budget structure: a family of overlapping subsets S ⊂ X and a separate budget B S for each S . For example, BestBuy can have a total budget for the ad campaign, and also separate budgets for ads about TVs and ads about computers. Finally, in addition to budgets (i.e., constraints on the number of times ads are clicked), an advertiser may wish to have similar constraints on the number of times ads are shown. BwK allows us to express all these constraints.

Adjusting a repeated auction. An auction is held in every round, with a fresh set of participants. The number of participants and a vector of their types come from a fixed but unknown distribution. The auction is adjustable : it has some parameter that the auctioneer adjust over time so as to optimize revenue. For example, Cesa-Bianchi et al. [2013] studies a repeated second price auction with an adjustable reserve price, with unlimited inventory of a single product. BwK framework allows to incorporate limited inventory of items to be sold at the auction, possibly with multiple products.

Repeated bidding. A bidder participates in a repeated auction, such as a sponsored search auction. In each round t , the bidder can adjust her bid b t based on the past performance. The outcome for this bidder is a vector ( p t , u t ) , where p t is the payment and u t is the utility received. We assume that this vector comes from a fixed but unknown distribution. The bidder has a fixed budget. Similar setting have been studied in [Amin et al., 2012, Tran-Thanh et al., 2014], for example.

We model this as a BwK problem where arms correspond to the possible bids, and the single resource is money. Note that (the basic version of) dynamic procurement corresponds to this setting with two possible outcome vectors ( p t , u t ) : (0 , 0) and ( b t , 1) .

The BwK setting also allows to incorporate more complicated constraints. For example, an action can result in several different types of outcomes that are useful for the bidder (e.g., an ad shown to a male or an ad shown to a female ), but the bidder is only interested in a limited quantity of each outcome.

## 8.4 Application to network routing and scheduling

In addition to applications to Electronic Markets, we describe two applications to network routing and scheduling. In both applications an algorithm chooses between different feasible policies to handle arriving 'service requests', such as connection requests in network routing and jobs in scheduling.

Adjusting a routing protocol. Consider the following stylized application to routing in a communication

network. Connection requests arrive one by one. A connection request consists of a pair of terminals; assume the pair comes from a fixed but unknown distribution. The system needs to choose a routing protocol for each connection, out of several possible routing protocols. The routing protocol defines a path that connects the terminals; abstractly, each protocol is simply a mapping from terminal pairs to paths. Once the path is chosen, a connection between the terminals is established. Connections persist for a significant amount of time. Each connection uses some amount of bandwidth. For simplicity, we can assume that this amount is fixed over time for every connection, and comes from a fixed but unknown distribution (although even a deterministic version is interesting). Each edge in the network (or perhaps each node) has a limited capacity: the total bandwidth of all connections that pass though this edge or node cannot exceed some value. A connection which violates any capacity constraint is terminated. The goal is to satisfy a maximal number of connections.

We model this problem as BwK as follows: arms correspond to the feasible routing protocols, each edge/node is a limited resource, each satisfied connection is a unit reward.

Further, if the time horizon is partitioned in epochs, we can model different bandwidth utilization in each phase; then a resource in BwK is a pair (edge,epoch).

Adjusting a scheduling policy. An application with a similar flavor arises in the domain of scheduling long-running jobs to machines. Suppose jobs arrive over time. Each job must be assigned to one of the machines (or dropped); once assigned, a job stays in the system forever (or for some number of 'epochs'), and consumes some resources. Jobs have multiple 'types' that can be observed by the scheduler. For each type, the resource utilization comes from a fixed but unknown distribution. Note that there may be multiple resources being consumed on each machine: for example, jobs in a datacenter can consume CPU, RAM, disk space, and network bandwidth. Each satisfied job of type i brings utility u i . The goal of the scheduler is to maximize utility given the constrained resources.

The mapping of this setting to BwK is straightforward. The only slightly subtle point is how to define the arms: in BwK terms, arms correspond to all possible mappings from job types to machines.

One can also consider an alternative formulation where there are several allowed scheduling policies (mappings from types and current resource utilizations to machines), and in every round the scheduler can choose to use one of these policies. Then the arms in BwK correspond to the allowed policies.

## Acknowledgements

The authors wish to thank Moshe Babaioff, Peter Frazier, Luyi Gui, Chien-Ju Ho and Jennifer Wortman Vaughan for helpful discussions related to this work. In particular, the application to routing protocols generalizes a network routing problem that was communicated to us by Luyi Gui. The application of dynamic procurement to crowdsourcing have been suggested to us by Chien-Ju Ho and Jennifer Wortman Vaughan. We are grateful to anonymous JACM referees for their thorough and insightful feedback.

## References

- Ittai Abraham, Omar Alonso, Vasilis Kandylas, and Aleksandrs Slivkins. Adaptive crowdsourcing algorithms for the bandit survey problem. In 26th Conf. on Learning Theory (COLT) , 2013.
- Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In 31st Intl. Conf. on Machine Learning (ICML) , 2014.

- Shipra Agrawal and Nikhil R. Devanur. Bandits with concave rewards and convex knapsacks. In 15th ACM Conf. on Economics and Computation (ACM EC) , 2014.
- Shipra Agrawal and Nikhil R. Devanur. Linear contextual bandits with knapsacks. In 29th Advances in Neural Information Processing Systems (NIPS) , 2016.
- Shipra Agrawal, Zizhuo Wang, and Yinyu Ye. A dynamic near-optimal algorithm for online linear programming. Operations Research , 62(4):876-890, 2014.
- Shipra Agrawal, Nikhil R. Devanur, and Lihong Li. An efficient algorithm for contextual bandits with knapsacks, and an extension to concave objectives. In 29th Conf. on Learning Theory (COLT) , 2016.
- Kareem Amin, Michael Kearns, Peter Key, and Anton Schwaighofer. Budget optimization for sponsored search: Censored learning in mdps. In 28th Conf. on Uncertainty in Artificial Intelligence (UAI) , 2012.
- Sanjeev Arora, Elad Hazan, and Satyen Kale. The multiplicative weights update method: a meta-algorithm and applications. Theory of Computing , 8(1):121-164, 2012.
- Peter Auer, Nicol` o Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning , 47(2-3):235-256, 2002a.
- Peter Auer, Nicol` o Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM J. Comput. , 32(1):48-77, 2002b. Preliminary version in 36th IEEE FOCS , 1995.
- Moshe Babaioff, Yogeshwer Sharma, and Aleksandrs Slivkins. Characterizing truthful multi-armed bandit mechanisms. SIAM J. on Computing (SICOMP) , 43(1):194-230, 2014. Preliminary version in 10th ACM EC , 2009.
- Moshe Babaioff, Shaddin Dughmi, Robert D. Kleinberg, and Aleksandrs Slivkins. Dynamic pricing with limited supply. ACM Trans. on Economics and Computation , 3(1):4, 2015. Special issue for 13th ACM EC , 2012.
- Ashwinkumar Badanidiyuru, Robert Kleinberg, and Yaron Singer. Learning on a budget: posted price mechanisms for online procurement. In 13th ACM Conf. on Electronic Commerce (EC) , pages 128-145, 2012.
- Ashwinkumar Badanidiyuru, Robert Kleinberg, and Aleksandrs Slivkins. Bandits with knapsacks. In 54th IEEE Symp. on Foundations of Computer Science (FOCS) , 2013.
- Ashwinkumar Badanidiyuru, John Langford, and Aleksandrs Slivkins. Resourceful contextual bandits. In 27th Conf. on Learning Theory (COLT) , 2014.
- Omar Besbes and Assaf Zeevi. Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. Operations Research , 57:1407-1420, 2009.
- Omar Besbes and Assaf J. Zeevi. Blind network revenue management. Operations Research , 60(6):15371550, 2012.
- Avrim Blum, Vijay Kumar, Atri Rudra, and Felix Wu. Online learning in online auctions. In 14th ACMSIAM Symp. on Discrete Algorithms (SODA) , pages 202-204, 2003.
- Arnoud V. Den Boer. Dynamic pricing and learning: Historical origins, current research, and new directions. Surveys in Operations Research and Management Science , 20(1), June 2015.

- S´ ebastien Bubeck and Nicolo Cesa-Bianchi. Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems. Foundations and Trends in Machine Learning , 5(1), 2012.
- Nicol´ o Cesa-Bianchi, Claudio Gentile, and Yishay Mansour. Regret minimization for reserve prices in second-price auctions. In ACM-SIAM Symp. on Discrete Algorithms (SODA) , 2013.
- Wei Chu, Lihong Li, Lev Reyzin, and Robert E. Schapire. Contextual Bandits with Linear Payoff Functions. In 14th Intl. Conf. on Artificial Intelligence and Statistics (AISTATS) , 2011.
- Varsha Dani, Thomas P. Hayes, and Sham Kakade. Stochastic Linear Optimization under Bandit Feedback. In 21th Conf. on Learning Theory (COLT) , pages 355-366, 2008.
- Nikhil Devanur and Sham M. Kakade. The price of truthfulness for pay-per-click auctions. In 10th ACM Conf. on Electronic Commerce (EC) , pages 99-106, 2009.
- Nikhil R. Devanur and Thomas P. Hayes. The AdWords problem: Online keyword matching with budgeted bidders under random permutations. In 10th ACM Conf. on Electronic Commerce (EC) , pages 71-78, 2009.
- Nikhil R. Devanur, Kamal Jain, Balasubramanian Sivan, and Christopher A. Wilkens. Near optimal online algorithms and fast approximation algorithms for resource allocation problems. In 12th ACM Conf. on Electronic Commerce (EC) , pages 29-38, 2011.
- Wenkui Ding, Tao Qin, Xu-Dong Zhang, and Tie-Yan Liu. Multi-armed bandit with budget constraint and variable costs. In 27th AAAI Conference on Artificial Intelligence (AAAI) , 2013.
- Miroslav Dud´ ıik, Daniel Hsu, Satyen Kale, Nikos Karampatziakis, John Langford, Lev Reyzin, and Tong Zhang. Efficient optimal leanring for contextual bandits. In 27th Conf. on Uncertainty in Artificial Intelligence (UAI) , 2011.
- Eyal Even-Dar, Shie Mannor, and Yishay Mansour. PAC bounds for multi-armed bandit and Markov decision processes. In 15th Conf. on Learning Theory (COLT) , pages 255-270, 2002.
- Jon Feldman, Monika Henzinger, Nitish Korula, Vahab S. Mirrokni, and Clifford Stein. Online stochastic packing applied to display ad allocation. In 18th Annual European Symp. on Algorithms (ESA) , pages 182-194, 2010.
- Yoav Freund and Robert E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences , 55(1):119-139, 1997.
- Naveen Garg and Jochen K¨ onemann. Faster and simpler algorithms for multicommodity flow and other fractional packing problems. SIAM J. Computing , 37(2):630-652, 2007.
- Sudipta Guha and Kamesh Munagala. Multi-armed Bandits with Metric Switching Costs. In 36th Intl. Colloquium on Automata, Languages and Programming (ICALP) , pages 496-507, 2007.
- Anupam Gupta, Ravishankar Krishnaswamy, Marco Molinaro, and R. Ravi. Approximation algorithms for correlated knapsacks and non-martingale bandits. In 52nd IEEE Symp. on Foundations of Computer Science (FOCS) , pages 827-836, 2011.
- Andr´ as Gy¨ orgy, Levente Kocsis, Ivett Szab´ o, and Csaba Szepesv´ ari. Continuous time associative bandit problems. In 20th Intl. Joint Conf. on Artificial Intelligence (IJCAI) , pages 830-835, 2007.

- Elad Hazan and Nimrod Megiddo. Online Learning with Prior Information. In 20th Conf. on Learning Theory (COLT) , pages 499-513, 2007.
- Robert Kleinberg. Nearly tight bounds for the continuum-armed bandit problem. In 18th Advances in Neural Information Processing Systems (NIPS) , 2004.
- Robert Kleinberg. Lecture notes for CS 683 (week 2), Cornell University, 2007. http://www.cs.cornell.edu/courses/cs683/2007sp/lecnotes/week2.pdf .
- Robert Kleinberg and Tom Leighton. The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In 44th IEEE Symp. on Foundations of Computer Science (FOCS) , pages 594-605, 2003.
- Robert Kleinberg and Aleksandrs Slivkins. Sharp dichotomies for regret minimization in metric spaces. In 21st ACM-SIAM Symp. on Discrete Algorithms (SODA) , 2010.
- Robert Kleinberg, Aleksandrs Slivkins, and Eli Upfal. Multi-armed bandits in metric spaces. In 40th ACM Symp. on Theory of Computing (STOC) , pages 681-690, 2008.
- Tze Leung Lai and Herbert Robbins. Asymptotically efficient Adaptive Allocation Rules. Advances in Applied Mathematics , 6:4-22, 1985.
- John Langford and Tong Zhang. The Epoch-Greedy Algorithm for Contextual Multi-armed Bandits. In 21st Advances in Neural Information Processing Systems (NIPS) , 2007.
- Nick Littlestone and Manfred K. Warmuth. The weighted majority algorithm. Information and Computation , 108(2):212-260, 1994.
- Tyler Lu, D´ avid P´ al, and Martin P´ al. Showing Relevant Ads via Lipschitz Context Multi-Armed Bandits. In 14th Intl. Conf. on Artificial Intelligence and Statistics (AISTATS) , 2010.
- Marco Molinaro and R. Ravi. Geometry of online packing linear programs. In 39th Intl. Colloquium on Automata, Languages and Programming (ICALP) , pages 701-713, 2012.
- Sandeep Pandey, Deepak Agarwal, Deepayan Chakrabarti, and Vanja Josifovski. Bandits for Taxonomies: A Model-based Approach. In SIAM Intl. Conf. on Data Mining (SDM) , 2007a.
- Sandeep Pandey, Deepayan Chakrabarti, and Deepak Agarwal. Multi-armed Bandit Problems with Dependent Arms. In 24th Intl. Conf. on Machine Learning (ICML) , 2007b.
- Christos H. Papadimitriou and John N. Tsitsiklis. The complexity of optimal queuing network control. Math. Oper. Res. , 24(2):293-305, 1999.
- Serge A. Plotkin, David B. Shmoys, and Eva Tardos. Fast approximation algorithms for fractional packing and covering problems. Mathematics of Operations Research , 20:257-301, 1995.
- Adish Singla and Andreas Krause. Truthful incentives in crowdsourcing tasks using regret minimization mechanisms. In 22nd Intl. World Wide Web Conf. (WWW) , pages 1167-1178, 2013.
- Aleksandrs Slivkins and Jennifer Wortman Vaughan. Online decision making in crowdsourcing markets: Theoretical challenges. SIGecom Exchanges , 12(2), December 2013.

- Long Tran-Thanh, Archie Chapman, Enrique Munoz de Cote, Alex Rogers, and Nicholas R. Jennings. /epsilon1 -first policies for budget-limited multi-armed bandits. In 24th AAAI Conference on Artificial Intelligence (AAAI) , pages 1211-1216, 2010.
- Long Tran-Thanh, Archie Chapman, Alex Rogers, and Nicholas R. Jennings. Knapsack based optimal policies for budget-limited multi-armed bandits. In 26th AAAI Conference on Artificial Intelligence (AAAI) , pages 1134-1140, 2012.
- Long Tran-Thanh, Lampros C. Stavrogiannis, Victor Naroditskiy, Valentin Robu, Nicholas R. Jennings, and Peter Key. Efficient regret bounds for online bid optimisation in budget-limited sponsored search auctions. In 30th Conf. on Uncertainty in Artificial Intelligence (UAI) , 2014.
- Zizhuo Wang, Shiming Deng, and Yinyu Ye. Close the gaps: A learning-while-doing algorithm for singleproduct revenue management problems. Operations Research , 62(2):318-331, 2014.
- Peter Whittle. Multi-armed bandits and the Gittins index. J. Royal Statistical Society, Series B , 42(2): 143-149, 1980.

## A The optimal dynamic policy beats the best fixed arm

Let us provide additional examples of BwK problem instances in which the optimal dynamic policy (in fact, the best fixed distribution over arms) beats the best fixed arm.

Dynamic pricing. Consider the basic setting of 'dynamic pricing with limited supply': in each round a potential buyer arrives, and the seller offers him one item at a price; there are k items and n &gt; k potential buyers. One can easily construct distributions for which offering a mixture of two prices is strictly superior to offering any fixed price. In fact this situation arises whenever the 'revenue curve' (the mapping from prices to expected revenue) is non-concave and its value at the quantile k/n lies below its concave hull.

Consider a simple example: fix /epsilon1 = k δ -1 / 2 with δ ∈ (0 , 1 2 ) , and assume that the buyer's value for an item is v = 1 with probability /epsilon1 k n and v = /epsilon1 with the remaining probability, for some fixed /epsilon1 ∈ (0 , 1) .

To analyze this example, let REW ( D ) be the expected total reward (i.e., the expected total revenue) from using a fixed distribution D over prices in each round; let REW ( p ) be the same quantity when D deterministically picks a given price p .

- Clearly, if one offers a fixed price in all rounds, it only makes sense to offer prices p = /epsilon1 and p = 1 . It is easy to see that REW ( /epsilon1 ) = /epsilon1k and REW (1) ≤ n · Pr[ sale at price 1 ] = /epsilon1k .
- Now consider a distribution D which picks price /epsilon1 with probability (1 -/epsilon1 ) k n , and picks price 1 with the remaining probability. It is easy to show that REW ( D ) ≥ /epsilon1k (2 -o (1)) .

So, REW ( D ) is essentially twice as large compared to the total expected revenue of the best fixed arm.

Dynamic procurement. A similar example can be constructed in the domain of dynamic procurement. Consider the basic setting thereof: in each round a potential seller arrives, and the buyer offers to buy one item at a price; there are T sellers and the buyer is constrained to spend at most budget B . The buyer has no value for left-over budget and each sellers value for the item is drawn i.i.d from an unknown distribution. Then a mixture of two prices is strictly superior to offering any fixed price whenever the 'sales curve' (the mapping from prices to probability of selling) is non-concave and its value at the quantile B/T lies below its concave hull.

Let us provide a specific example. Fix any constant δ &gt; 0 , and let /epsilon1 = B 1 / 2+ δ . Each seller has the following two-point demand distribution: the seller's value for item is v = 0 with probability B T , and v = 1 with the remaining probability. We use the notation REW ( D ) and REW ( p ) as defined above.

- Clearly, if one offers a fixed price in all rounds, it only makes sense to offer prices p = 0 and p = 1 . It is easy to see that REW (0) ≤ T · Pr[ selling at price 0] = B and REW (1) = B .
- Now consider a distribution D which picks price 0 with probability 1 -B -/epsilon1 T , and picks price 1 with the remaining probability. It is easy to show that REW ( D ) ≥ (2 -o (1)) B .

Again, REW ( D ) is essentially twice as large compared to the total expected sales of the best fixed arm.

## B BalancedExploration beats PrimalDualBwK sometimes

We provide a simple example in which BalancedExploration achieves much better regret than (what we can prove for) PrimalDualBwK . The reason is that BalancedExploration is aware of the BwK domain, whereas PrimalDualBwK is not. More precisely, BalancedExploration is parameterized by M feas , the set of all latent structures that are feasible for the BwK domain.

The example is a version of the deterministic example from Section 1.1. There is a time horizon T and two other resources, both with budget B &lt; T/ 2 . There are m arms, partitioned into two same-size subsets, X 1 and X 2 . Per-round rewards and per-round resource consumptions are deterministic for all arms. All arms get per-round reward 1 . For each resource i , each arm in X i only consumes this resource. Letting c i ( x ) = c i ( x, µ ) denote the (expected) per-round consumption of resource i by arm x , one of the following holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Analysis. Note that an optimal dynamic policy alternates the two arms in proportion, 1 : 2 or 2 : 1 , depending on the case, and an LP-optimal distribution over arms samples them in the same proportion.

The key argument is that, informally, BalancedExploration can tell (i) from (ii) after the initial O ( m log T ) rounds. In the specification of BalancedExploration , consider the confidence radius for the per-round consumption of resource i , as defined in Equation (11). After any one arm x is played at least C log T rounds, for a sufficiently large absolute constant C , this confidence radius goes below 1 / 4 . Therefore any two latent structures µ , µ ′ in the confidence interval satisfy | c i ( x, µ ) -c i ( x, µ ′ ) | &lt; 1 2 . It follows that the confidence interval consists of a single latent structure, either the one corresponding to (i) or the one corresponding to (ii), which is the correct latent structure for this problem instance. Accordingly, the chosen distribution over arms, being 'potentially perfect' by design, is LP-optimal. Thus, BalancedExploration uses the LP-optimal distribution over arms after the initial O ( m log T ) rounds.

The resulting regret is ˜ O ( m + √ B ) , where the √ B term arises because the empirical frequencies of the two arms can deviate by O ( √ B ) from the optimal values. Whereas with algorithm PrimalDualBwK we can only guarantee regret ˜ O ( √ mB ) .

## C Analysis of the Hedge Algorithm

We provide a self-contained proof of Proposition 5.4, the performance guarantee for the Hedge algorithm from Freund and Schapire [1997]. The presentation is adapted from Kleinberg [2007].

For the sake of convenience, we restate the algorithm and the proposition. It is an online algorithm for maintaining a d -dimensional probability vector y while observing a sequence of d -dimensional payoff vectors π 1 , . . . , π τ . The algorithm is initialized with a parameter /epsilon1 ∈ (0 , 1) .

## Algorithm Hedge ( /epsilon1 )

- 1: v 1 = 1
- 2: for t = 1 , 2 , . . . , τ do
- 3: y t = v t / ( 1 ᵀ v t ) .
- 4: v t +1 = Diag { (1 + /epsilon1 ) π ti } v t .

The performance guarantee of the algorithm is expressed by the following proposition.

Proposition (Proposition 5.4, restated) . For any 0 &lt; /epsilon1 &lt; 1 and any sequence of payoff vectors π 1 , . . . , π τ ∈ [0 , 1] d , we have

Proof. The analysis uses the potential function Φ t = 1 ᵀ v t . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the third line, we have used the inequality (1 + /epsilon1 ) x ≤ 1 + /epsilon1x which is valid for 0 ≤ x ≤ 1 . Now, summing over t = 1 , . . . , τ we obtain

<!-- formula-not-decoded -->

The maximum of y ᵀ ( ∑ τ t =1 π t ) over y ∈ ∆ [ d ] must be attained at one of the extreme points of ∆ [ d ] , which are simply the standard basis vectors of R d . Say that the maximum is attained at e i . Then we have

<!-- formula-not-decoded -->

The last line follows from two observations. First, our choice of i ensures that ∑ τ t =1 π ti ≥ ∑ τ t =1 y ᵀ π t for every y ∈ ∆ [ d ] . Second, the inequality ln(1 + /epsilon1 ) &gt; /epsilon1 -/epsilon1 2 holds for every /epsilon1 &gt; 0 . In fact,

<!-- formula-not-decoded -->

## D Facts for the proof of the lower bound

For the sake of completeness, we provide self-contained proofs for the two facts used in Section 6.

Fact (Fact 6.3, restated) . Let S t be the sum of t i.i.d. 0-1 variables with expectation q . Let τ be the first time this sum reaches a given number B ∈ N . Then E [ τ ] = B/q . Moreover, for each T &gt; E [ τ ] it holds that

Proof. E [ τ ] = B/q follows from the martingale argument presented in the proof of Claim 6.4. Formally, take q = p -/epsilon1 and N τ = τ .

<!-- formula-not-decoded -->

Assume T &gt; E [ τ ] . The proof of Equation (55) uses two properties, one being that a geometric random variable is memoryless and other being Markov's inequality. Let us first bound the random variable τ -T conditional on the event that τ &gt; T .

<!-- formula-not-decoded -->

By Markov's inequality we have Pr[ τ ≥ T ] ≤ E [ τ ] T . Combining the two inequalities, we have

<!-- formula-not-decoded -->

Fact (Fact 6.12, restated) . Assume /epsilon1 p ≤ 1 2 and p ≤ 1 2 . Then

<!-- formula-not-decoded -->

Proof. To prove the inequality we use the following standard inequalities:

<!-- formula-not-decoded -->

It follows that:

<!-- formula-not-decoded -->