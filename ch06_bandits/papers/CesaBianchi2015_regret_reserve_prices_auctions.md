## Regret Minimization for Reserve Prices in Second-Price Auctions

Nicolò Cesa-Bianchi, Claudio Gentile, and Yishay Mansour

Abstract -Weshow a regret minimization algorithm for setting the reserve price in a sequence of second-price auctions, under the assumption that all bids are independently drawn from the same unknown and arbitrary distribution. Our algorithm is computationally efficient, and achieves a regret of ˜ O ( √ T ) in a sequence of T auctions. This holds even when the number of bidders is stochastic with a known distribution.

Index Terms -Prediction theory, sequential analysis, statistical learning, semi-supervised learning.

## I. INTRODUCTION

C ONSIDER a merchant selling items through e-Bay auctions. The sell price in each auction is the secondhighest bid, and the merchant knows the price at which the item was sold, but not the individual bids from the bidders that participated in the auction. How can the merchant set a reserve price in order to optimize revenues? Similarly, consider a publisher selling advertisement space through Ad Exchange (such as AdX) or Supply Side Platform (such as Adsense), where advertisers bid for the advertisement slot and the price is the second-highest bid. With no access to the number of bidders that participate in the auction, and knowing only the actual price that was charged, how can the publisher set an optimal reserve price?

We abstract this scenario by considering the following problem: A seller is faced with repeated auctions, where each auction has a (different) set of bidders, and each bidder draws bids from some fixed unknown distribution which is the same for all bidders. It is important to remark that we need not assume that the bidders indeed bid their private value. Our assumption on the bidders' behavior, a priori, implies that if they bid using the same strategy, their bid distribution is identical. 1 The sell price is the second-highest bid, and the

Manuscript received July 11, 2013; revised August 8, 2014; accepted October 16, 2014. Date of publication October 29, 2014; date of current version December 22, 2014. N. Cesa-Bianchi was supported in part by the MIUR through the ARS TechnoMedia Project within PRIN 2010-2011 under Grant 2010N5K7EB 003. Y. Mansour was supported in part by the Israel Science Foundation, in part by the U.S.-Israel Binational Science Foundation, in part by the Ministry of Science and Technology, Israel, and in part by the Israeli Centers of Research Excellence Program under Grant 4/11.

N. Cesa-Bianchi is with the Dipartimento di Informatica, Università degli Studi di Milano, Milan 20122, Italy (e-mail: nicolo.cesa-bianchi@unimi.it).

C. Gentile is with the Dipartimento di Scienze Teoriche ed Applicate, Università dell'Insubria, Como 22100, Italy (e-mail: claudio.gentile@uninsubria.it).

Y. Mansour is with the School of Computer Science, Tel Aviv University, Tel Aviv 6997801, Israel (e-mail: mansour@tau.ac.il).

Communicated by V. Borkar, Associate Editor for Communication Networks.

Digital Object Identifier 10.1109/TIT.2014.2365772

1 For example, if we had considered a first-price auction, then assuming that bidders use the same strategy to map their private value to a bid would result in the same bid distribution.

seller's goal is to maximize the revenue by only relying on information regarding revenues on past auctions.

The issue of revenue maximization in second-price auctions has received a significant attention in the economics literature. The Revenue Equivalence theorem shows that truthful mechanisms 2 that allocate identically have identical revenue (see [15]). Myerson [14], for the case of monotone hazard rate distributions, characterized the optimal revenue maximization truthful mechanism as a second-price auction with a seller's reserve price , i.e., with a minimum price disqualifying any bid below it.

In addition to their theoretical relevance, reserve prices are to a large extent the main mechanism through which a seller can directly influence the auction revenue in today's electronic markets. The examples of e-Bay, AdX and Adsense are just a few in a large collection of such settings. The practical significance of optimizing reserve prices in sponsored search was reported in [16], where optimization produced a significant impact on Yahoo!'s revenue.

We stress that unlike much of the mechanism design literature (see [15]), we are not searching for the optimal revenue maximization truthful mechanism. Rather, our goal is to maximize the seller's revenue in a given, yet very popular, mechanism of second-price auction with a reserve price. In our model, the seller has only information about the auction price (and possibly about the number of bidders that participated in the auction). We assume all buyers have the same unknown bid distribution, but we make no assumptions about this distribution, only that the bids are from a bounded domain. In particular, we do not assume that the distribution has a monotone hazard rate, a traditional assumption in the economics literature. The main modeling assumption we rely upon is that buyers draw their value independently from the same distribution (i.e., bids are independent and identically distributed). This is a reasonable assumption when the auction is open to a wide audience of potential buyers. In this case, it is plausible that the seller's strategy of choosing reserve prices has no influence on the distribution of bids.

## A. Our Results

The focus of our work is on setting the reserve price in a second-price auction, in order to maximize the seller's revenue. Our main result is an online algorithm that optimizes the seller's reserve price based only on the observation of the seller's actual revenue at each step. We show that after T steps

2 A mechanism is truthful if it is a dominant action for the bidders to bid their private value.

( T repetitions of the auction) our algorithm has a regret of only ˜ O ( √ T ) . Namely, using our online algorithm the seller has an average revenue per auction that differs from that of the optimal reserve price by at most ˜ O ( 1 / √ T ) , assuming the value of any bid lies in a bounded range interval.

Our algorithm is rather easy to explain and motivate at a high level. Let us start with a simple O ( T 2 / 3 ) regret minimization algorithm, similar to [12]. The algorithm discretizes the range of reserve prices to /Theta1( T 1 / 3 ) price bins, and uses some efficient multi-armed bandit algorithm (see [5]) over the bins. It is easy to see that lowering the optimal reserve price by /epsilon1 will result in an average loss of at most /epsilon1 . 3 This already shows that vanishing average regret is achievable, specifically, a regret of O ( T 2 / 3 ) . Our main objective is to improve over this basic algorithm and achieve a regret of ˜ O ( √ T ) .

An important observation to understand our algorithm is that by setting the reserve price low (say, zero) we observe the second-highest bid, since this will be the price in the auction. Hence, with enough observations, we can reconstruct the distribution of the second-highest bid. Given the assumption that the bidders' bid distributions are identical, we can recover the bid distribution of an individual bidder, and the distribution of the highest bid. Clearly, a good approximation to this distribution results in a good approximation to the optimal reserve price. Unfortunately, this simple method does not improve the regret, since a good approximation of the second-highest bid distribution incurs a significant loss in the exploration, and results in a regret of O ( T 2 / 3 ) , similar to the regret of the discretization approach.

Our main solution is to perform only a rough estimate of the second-highest bid distribution. Using this rough estimate, we can set a better reserve price. In order to facilitate future exploration, it is important to set the new reserve price to the lowest potentially optimal reserve price. The main benefit is that our new reserve price has a lower regret with respect to the optimal reserve price, and we can bound this improved regret. We continue in this process, getting improved approximations to the optimal reserve price, and accumulating lower regret (per time step) in each successive iteration, resulting in a total regret of ˜ O ( √ T ) for T time steps.

Our ability to reconstruct the bid distribution depends on our knowledge about the number of participating bidders in the auction. Our simpler case involves a known number of bidders (Section II). We later extend the algorithm and analysis to the case where there is stochasticity in the number of bidders through a known distribution (Section III). In both cases we prove a regret bound of ˜ O ( √ T ) . This bound is optimal up to logarithmic factors. In fact, simple choices of the bid distribution exist that force any algorithm to have order √ T regret, even when there are only two bidders whose bids are revealed to the algorithm at the end of each auction.

Finally, in Section IV we present two extensions. One is for the case when the regret analysis refers to the stronger notion of realized regret (Section IV-A), the other extension

3 Note that the setting is not symmetric, and increasing by /epsilon1 might lower the revenue significantly, by disqualifying many attractive bids.

is a standard twist that removes any prior knowledge on the time horizon T (Section IV-B).

## B. Related Work

There is a vast literature in Algorithmic Game Theory on second price auctions, with sponsored search as a motivating application. An important thread of research concerns the design of truthful mechanisms to maximize the revenue in the worst case, and the derivation of competitive ratio bounds, see [10]. A recent related work [8] discusses revenue maximization in a Bayesian setting. Their main result is a mechanism that achieves a constant approximation ratio with respect to any prior distribution using a single sample. They also show that with additional samples, the approximation ratio improves, and in some settings they even achieve a 1 -/epsilon1 approximation. In contrast, we assume a fixed but unknown prior distribution, and consider the rate at which we can approximate the optimal reserve price. In our setting, as we mentioned before, achieving a 1 -/epsilon1 approximation, even for /epsilon1 = T -1 / 3 , is straightforward, and the main focus of this paper is to show that a rate of /epsilon1 = T -1 / 2 is attainable.

Item pricing, which is related to regret minimization under partial observation [5], has also received significant attention. A specific related work is [12], where the effect of knowing the demand curve is studied. (The demand curve is equivalent to the bid distribution.) The mechanism discussed in [12] is a posted price mechanism, and the regret is computed in both stochastic and adversarial settings. In the stochastic setting they assume that the expected revenue function is strictly concave, and use the UCB algorithm of [3] over discretized bid values to derive their strategy. Again, we do not make such assumptions in our work.

The question of the identification of the buyers' utilities given the auction outcome has been studied in the economics literature. The main goal is to recover in the limit the buyers' private value distribution (i.e., the buyers' utility function), given access to the resulting auction price (i.e., the auction outcome) and assuming that bidders utilities are independent and identically distributed [1], [9]. It is well known in the economics literature that given a bid distribution that has a monotone hazard rate, there is a unique reserve price maximizing the expected revenue in a second-price auction, and this optimal price is independent of the number of bidders [14]. As we do not make the monotone hazard rate assumption, in our case the optimal price for each auction might depend on the actual (varying) number of bidders. Because the seller does not observe the number of bidders before setting the reserve price (Section III), we prove our results using the regret to the best reserve price, with respect to a known prior over the number of bidders. As we just argued, depending on the bid distribution, this best reserve price need not be the same as the optimal reserve price one could set when knowing the actual number of bidders in advance.

There have been some works [7], [11], [20] on optimizing the reserve price, concentrating on more involved issues that arise in practice, such as discrete bids, nonstationary behavior, hidden bids, and more. While we are definitely not the first

Fig. 1. The revenue function R for m = 5 bids of value B ( 1 ) = 0 . 7 , B ( 2 ) = 0 . 5 , B ( 3 ) = 0 . 35 , B ( 4 ) = 0 . 24 , B ( 5 ) = 0 . 05. For p ∈ [ 0 , B ( 2 ) ] the revenue is constant, R ( p ) = B ( 2 ) . For p ∈ [ B ( 2 ) , B ( 1 ) ] the revenue grows linearly, R ( p ) = p , For p ∈ [ B ( 1 ) , 1 ] the revenue is null, R ( p ) = 0.

<!-- image -->

ones to consider approximating optimal reserve prices in a second-price auction, to the best of our knowledge this is the first work that derives formal and concrete convergence rates.

Finally, note that any algorithm for one-dimensional stochastic bandit optimization could potentially be applied to solve our revenue maximization problem. Indeed, whenever a certain reserve price is chosen, the algorithm observes a realization of the associated stochastic revenue. While many algorithms exist that guarantee low regret in this setting, they all rely on specific assumptions on the function to optimize (in our case, the expected revenue function). See [6] obtains a regret of order √ T under smoothness and strong concavity. The authors of [2] achieve a regret worse only by logarithmic factors without concavity, but assuming other conditions on the derivatives. The work [21] shows a bound of the same order just assuming unimodality. The work [4] also obtains the same asymptotics ˜ O ( √ T ) on the regret using a local Lipschitz condition. The approach developed in this paper avoids making any assumption on the expected revenue function, such as Lipschitzness or bounded number of maxima. Instead, it exploits the specific feedback model provided by the secondprice auction in order gain information about the optimum.

## II. KNOWN NUMBER OF BIDDERS

We first show our results for the case where the number of bidders m is known and fixed. In Section III we will remove this assumption, and extend the results to the case when the number of bidders is a random variable with a known distribution. Fortunately, most of the ideas of the algorithm can be explained and nicely analyzed in the simpler case.

## A. Preliminaries

The auctioneer organizes an auction about an item to be sold. He collects m ≥ 2 bids B 1 , B 2 , . . . , Bm which are i.i.d. bounded random variables (for definiteness, we let Bi ∈ [ 0 , 1 ] for i = 1 , . . . , m ) whose common cumulative distribution function F is arbitrary and unknown. We let

Fig. 2. At the beginning of Stage i + 1, Algorithm 1 has at its disposal an estimate ̂ µ i (here represented by the piecewise constant solid line) of the actual expected revenue function µ (the thick solid line). The upper horizontal dashed line indicates the estimate ̂ µ i ( ̂ p ∗ i ) of the actual maximum µ( p ∗ ) (recall that ̂ p ∗ i is a maximizer of ̂ µ i ( · ) ). The lower horizontal dashed line indicates the lower end of the confidence interval for ̂ µ i ( ̂ p ∗ i ) . This defines the next set Pi + 1 of candidate optimal reserve prices, here marked by the thick solid line on the price axis, and the next reserve price ̂ p i + 1 , which is the lowest price in Pi + 1 . In this figure, ̂ p i + 1 = 0. Also, for simplicity, we have disregarded the further constraint ̂ F 2 , i ( p ) ≤ 1 -α .

<!-- image -->

B ( 1 ) , B ( 2 ) , . . . , B ( m ) denote the corresponding order statistics B ( 1 ) ≥ B ( 2 ) ≥ · · · ≥ B ( m ) .

In this simplified setting, we consider a protocol in which a learning algorithm (or a 'mechanism') is setting a reserve price (i.e., a minimal price) p ∈ [ 0 , 1 ] for the auction. The algorithm then observes a revenue R ( p ) = R ( p ; B 1 , . . . , Bm ) defined as follows: 

<!-- formula-not-decoded -->

In words, if the reserve price p is higher than the highest bid B ( 1 ) , the item is not sold, and the auctioneer's revenue is zero; if p is lower than B ( 1 ) but higher than the second-highest bid B ( 2 ) then we sell at the reserve price p (i.e., the revenue is p ); finally, if p is lower than B ( 2 ) we sell the item to the bidder who issued the highest bid B ( 1 ) at the price of the secondhighest bid B ( 2 ) (hence the revenue is B ( 2 ) ). Figure 1 gives a pictorial illustration of the revenue function R ( p ) .

The expected revenue µ( p ) = E [ R ( p ) ] is the expected value of the revenue gathered by the auctioneer when the algorithm plays price p , the expectation being over the bids B 1 , B 2 , . . . Bm . Let

<!-- formula-not-decoded -->

be the optimal price for the bid distribution F . We also write F 2 to denote the cumulative distribution function of B ( 2 ) . We can write the expected revenue as [ ] [ ∣ ]

<!-- formula-not-decoded -->

where the first term is the baseline, the revenue of a secondprice auction with no reserve price. The second term is the

gain due to the reserve price (increasing the revenue beyond the second-highest bid). The third term is the loss due to the possibility that we will not sell (when the reserve price is higher than the highest bid). The following fact streamlines the computation of µ( p ) . All proofs are given in the appendices.

Fact 1: With the notation introduced so far, we have

∫

<!-- formula-not-decoded -->

where the expectation E [ · ] is over the m bids B 1 , B 2 , . . . , Bm. The algorithm interacts with its environment (the bidders) in a sequential fashion. At each time step t = 1 , 2 , . . . the algorithm sets a price pt and receives revenue Rt ( pt ) = R ( pt ; Bt , 1 , . . . , Bt , m ) which is a function of the random bids Bt , 1 , . . . , Bt , m at time t . The price pt depends on past revenues Rs ( ps ) for s &lt; t , and therefore on past bids. Given a sequence of reserve prices p 1 , . . . , pT , we define the (cumulative) expected regret as

<!-- formula-not-decoded -->

where the expectation E t = E t [ · | p 1 , . . . , pt -1 ] is over the random bids at time t , conditioned on all past prices p 1 , . . . , pt -1 (i.e., conditioned on the past history of the bidding process). This implies that the expected regret (1) is indeed a random variable, as each pt depends on the past random revenues. Our goal is to devise an algorithm whose regret after T steps is ˜ O ( √ T ) with high probability, and with as few assumptions as possible on F . We see in the sequel that, when T is large, this goal can actually be achieved with no assumptions whatsoever on the underlying distribution F . Moreover, in Section IV-A we use a uniform convergence argument to show that the same regret bound ˜ O ( √ T ) holds with high probability for the realized regret

<!-- formula-not-decoded -->

Note that here the realized revenue of the seller is compared against the best reserve price on each sequence of bid realizations. Therefore, the realized regret is a much stronger notion of regret than the expected regret (1).

It is well known that from the distribution of any order statistics one can reconstruct the underlying distribution. Unfortunately, we do not have access to the true distribution of order statistics, but only to an approximation thereof. We first need to show that a small deviation in our approximation will have a small effect on our final result. The following preliminary lemma will be of great importance in our approximations. It shows that if we have a small error in the approximation of F 2 ( p ) we can recover µ( p ) with a small error. The function β( · ) therein maps ( F ( · )) m to F 2 ( · ) . In fact, since the bids are independent with the same distribution, we have F 2 ( p ) = m ( F ( p )) m -1 ( 1 -F ( p )) + ( F ( p )) m = β ( ( F ( p )) m ) . The main technical difficulty arises from the fact that the function β -1 ( · ) we use in reconstructing ( F ( · ) ) m from F 2 ( · ) -see pseudocode in Algorithm 1, is not a Lipschitz function.

Lemma 1: Fix an integer m ≥ 2 and consider the function

Then β -1 ( · ) exists in [ 0 , 1 ] . Moreover, if a ∈ ( 0 , 1 ) and x ∈ [ 0 , 1 ] are such that a -/epsilon1 ≤ β( x ) ≤ a + /epsilon1 for some /epsilon1 ≥ 0 , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In a nutshell, this lemma shows how approximations in the value of β( · ) turn into approximations in the value of β -1 ( · ) . Because the derivative of β -1 is infinite at 1, we cannot hope to get a good approximation unless a is bounded away from 1. For this very reason, we need to make sure that our function approximations are only applied to cases where the arguments are not too close to 1. The approximation parameter α in the pseudocode of Algorithm 1 serves this purpose.

## B. The Algorithm

Our algorithm works in stages , where the same price is consistently played during each stage. 4 Stage 1 lasts T 1 steps, during which the algorithm plays pt = ̂ p 1 for all t = 1 , . . . , T 1. Stage 2 lasts T 2 steps, during which the algorithm plays pt = ̂ p 2 for all t = T 1 + 1 , . . . , T 1 + T 2, and so on, up to S stages. Overall, the regret suffered by this algorithm can be written as

<!-- formula-not-decoded -->

where the sum is over the S stages. The length Ti of each stage will be set later on, as a function of the total number of steps T . The reserve prices ̂ p 1 , ̂ p 2 , . . . are set such that 0 = ̂ p 1 ≤ ̂ p 2 ≤ · · · ≤ 1. At the end of each stage i , the algorithm computes a new estimate ̂ µ i of the expected revenue function µ in the interval [ ̂ pi , 1 ] , where p ∗ is likely to lie. This estimate depends on the empirical cumulative distribution function ̂ F 2 , i of F 2 computed during stage i in the interval [ ̂ pi , 1 ] . The algorithm's pseudocode is given in Algorithm 1. The quantity C δ, i ( p ) therein is defined as

√

<!-- formula-not-decoded -->

C δ, i ( p ) is a confidence interval (at confidence level 1 -δ/( 3 S ) ) for the point estimate ̂ µ i ( p ) in stage i , where S = S ( T ) is either the total number of stages or an upper bound thereof.

Stage 1 is a seed stage, where the algorithm computes a first approximation ̂ µ 1 of µ . Since the algorithm plays ̂ p 1 = 0, and R ( 0 ) = B ( 2 ) , during this stage T 1 independent realizations of the second-bid variable B ( 2 ) are observed. Hence the empirical distribution ̂ F 2 , 1 in Algorithm 1 is a standard cumulative empirical distribution function based on i.i.d. realizations of B ( 2 ) . The approximation ̂ µ 1 is based on the corresponding expected revenue µ contained in Fact 1, where β( · ) is the function defined in Lemma 1, mapping ( F ( p )) m to F 2 ( p ) .

4 For simplicity, we have disregarded rounding effects in the computation of the integer stage lengths Ti .

Т.

For Stage 1 = 2,3, ...

Algorithm 1 Regret Minimizer

• For all + = 1 + [= Tj,..., Ej=, Tj, play Pr = Pi, and observe revenues Ri(Pi),...., RT, (Pi),

- Compute maximizer

## Algorithm 1 Regret Minimizer

argmax

Mi-1 (p).

Input: Confidence level δ ∈ ( 0 , 1 ] , approximation parameter α ∈ ( 0 , 1 ] , time horizon T ; Let stage lengths Ti = T 1 -2 -i for i = 1 , 2 , . . .

## Stage 1:

- For all t = 1 , . . . , T 1, play pt = ̂ p 1 = 0 and observe revenues R 1 ( 0 ), . . . , RT 1 ( 0 ) .

∣

<!-- formula-not-decoded -->

- Compute, for x ∈ [ 0 , 1 ] , empirical distribution
- Compute, for p ∈ [ 0 , 1 ] , approximation

∫

<!-- formula-not-decoded -->

For Stage i = 2 , 3 , . . .

∑

∑

- For all t = 1 + i -1 j = 1 Tj , . . . , i j = 1 Tj , play pt = ̂ pi , and observe revenues R 1 ( ̂ pi ), . . . , RTi ( ̂ pi ) , where ̂ pi is computed as follows:
- -Compute maximizer

{

<!-- formula-not-decoded -->

}

- -Let Pi = p ∈ [ ̂ pi -1 , 1 ] : ̂ µ i -1 ( p ) ≥ ̂ µ i -1 ( ̂ p ∗ i -1 ) -2 C δ, i -1 ( ̂ p ∗ i -1 ) -2 C δ, i -1 ( p ) . -Set ̂ pi = min Pi ⋂{ p : ̂ F 2 , i -1 ( p ) ≤ 1 -α } .

∣

{

}∣

<!-- formula-not-decoded -->

- Compute, for x ∈ [ ̂ pi , 1 ] , empirical distribution
- Compute, for p ∈ [ ̂ pi , 1 ] , approximation

∫

∫

<!-- formula-not-decoded -->

Note that if β -1 is available, maximizing the above function (done in Stage 2) can easily be computed from the data. The presence of the unknown constant E [ B ( 2 ) ] is not a problem for this computation. 5 In Stage 2 (encompassing trials t = T 1 + 1 , . . . , T 1 + T 2) the algorithm calculates the empirical maximizer

<!-- formula-not-decoded -->

then computes the set of candidate optimal reserve prices

{

}

<!-- formula-not-decoded -->

and sets the reserve price ̂ p 2 to be the lowest one in P 2, subject to the additional constraint that 6 ̂ F 2 , 1 ( p ) ≤ 1 -α . Price ̂ p 2 is played during all trials within Stage 2. The corresponding revenues Rt ( ̂ p 2 ) , for t = 1 , . . . , T 2, are gathered and used to construct an empirical cumulative distribution ̂ F 2 , 2 and an approximate expected revenue function ̂ µ 2 to be used only in

5 Note that in the algorithm (subsequent Stage 2) we either take the difference of two values ̂ µ 1 ( p 1 ) -̂ µ 1 ( p 2 ) , in which case the constant cancels, or maximize over ̂ µ 1 ( p ) , in which case the constant does not change the outcome.

6 Note that the intersection is not empty, since ̂ p ∗ 1 is in the intersection.

the subinterval 7 [ ̂ p 2 , 1 ] .

{

In order to see why ̂ F 2 , 2 and ̂ µ 2 are useful only on [ ̂ p 2 , 1 ] , observe that

̂

R

(

p

)

=

p

̂

2

or 0

if

2

if

B

(

2

)

B

B

(

(

2

2

) &lt;

)

≥

p

̂

2

̂

2

p

Thus, for any x ≥ ̂ p 2 we have that

<!-- formula-not-decoded -->

Hence, if we denote by R 1 ( ̂ p 2 ), . . . , RT 2 ( ̂ p 2 ) the revenues observed by the algorithm during Stage 2, the empirical distribution function

∣

<!-- formula-not-decoded -->

}∣

approximates F 2 ( x ) only for x ∈ [ ̂ p 2 , 1 ] .

All other stages i &gt; 2 proceed similarly, each stage i relying on the existence of empirical estimates ̂ F 2 , i -1, ̂ µ i -1, and ̂ pi -1 delivered by the previous stage i -1. Figure 2 gives a pictorial explaination of the way the algorithm works.

7 Once again, computing the argmax of ̂ µ 2 over [ ̂ p 2 , 1 ] as well as the set of candidates P 3 (done in the subsequent Stage 3) is not prevented by the presence of the unknown constants E [ B ( 2 ) ] and ∫ ̂ p 2 0 F 2 ( t ) dt therein.

} ∣

.

## C. Regret Analysis

We start by showing that for all stages i the term 1 -̂ F 2 , i ( p ) in the denominator of C δ, i ( p ) can be controlled for all p such that µ( p ) is bounded away from zero. Recall that S = S ( T ) denotes (an upper bound on) the total number of stages.

Lemma 2: With the notation introduced so far, for any fixed stage i,

√

<!-- formula-not-decoded -->

holds with probability at least 1 -δ/( 3 S ) , uniformly over p ∈ [ ̂ pi , 1 ] , conditioned on all past stages.

In the sequel, we use Lemma 2 with p = p ∗ and assume that 1 -̂ F 2 , i ( p ∗ ) ≥ α holds for each stage i with probability at least 1 -δ/( 3 S ) , where the approximation parameter α is defined as

√

<!-- formula-not-decoded -->

provided p ∗ ∈ [ ̂ pi , 1 ] . In order to ensure that α &gt; 0, it suffices to have µ( p ∗ ) &gt; 0 and T large enough -see Theorem 1 below. Recall that it is important to guarantee that ̂ F 2 , i ( p ) be bounded away from 1 for all arguments p which we happen to evaluate ̂ F 2 , i at. This is because the function β -1 has an infinite derivative at 1.

The following lemma is crucial to control the regret of Algorithm 1. It states that the approximation in stage i is accurate. In addition, it bounds the empirical regret in stage i , provided our current reserve price is lower than the optimal reserve price. The proof is a probabilistic induction over stages.

Lemma 3: The event

∣

∣

<!-- formula-not-decoded -->

holds with probability at least 1 -δ/ 3 simultaneously in all stages i = 1 , . . . , S. Moreover, the events

<!-- formula-not-decoded -->

both hold with probability at least 1 -δ simultaneously in all stages i = 1 , . . . , S.

The next theorem proves our regret bound under the assumption that µ( p ∗ ) is nonzero. Note that µ( p ∗ ) = 0 corresponds to the degenerate case µ( p ) = 0 for all p ∈ [ 0 , 1 ] . Under the above assumption, the theorem states that when the horizon T is sufficiently large, then with high probability the regret of Algorithm 1 is O ( √ T log log log T ( log log T ) ) = ˜ O ( √ T ) . It is important to remark that in this bound there is no explicit dependence on the number m of bidders.

Theorem 1: For any distribution F of the bids and any m ≥ 2 such that µ( p ∗ ) &gt; 0 , we have that Algorithm 1 operating on any time horizon T such that

(

)

<!-- formula-not-decoded -->

using approximation parameter α ≥ µ( p ∗ ) 2 / 12 has regret (√ )

<!-- formula-not-decoded -->

with probability at least 1 -δ .

The proof of this theorem follows by applying at each stage i the uniform approximation delivered by Lemma 3 on the accuracy of empirical to true regret. This would bound √

the regret in stage i by 8 1 α Ti -1 ln 6 S δ -see the proof in Appendix V. We then set the length Ti of stage i as Ti = T 1 -2 -i , i.e., T 1 = √ T , T 2 = T 3 / 4 , T 3 = T 7 / 8 , . . . , which implies that the total number of stages S is O ( log log T ) . Finally, we sum the regret over the stages to derive the theorem.

Two remarks are in order at this point. First, the reader should observe that the bound in Theorem 1 does not explicitly contain a dependence on m ; this is mostly due to the fact that the approximation result in Lemma 1 is in turn independent of m . The number of bidders m shows up implicitly only through µ( p ∗ ) -see also the discussion at the end of Section V. Second, the way we presented it makes Algorithm 1 depending on the time horizon T , though this prior knowledge is not strictly required: In Section IV-B we show a standard 'doubling trick' for making Algorithm 1 independent of the time horizon T .

## D. Lower Bounds

The next result shows that the √ T dependence of the regret on the time horizon T is not a consequence of our partial information setting. Indeed, this dependence cannot be removed even if the mechanism is allowed to observe the actual bids after setting the reserve price in each repetition of the auction.

Theorem 2: There exists a distribution of bids such that any deterministic algorithm operating with m = 2 bidders is forced to have expected regret

<!-- formula-not-decoded -->

= Although the result is proven for deterministic algorithms, it can easily be extended to randomized algorithms through a standard argument.

## III. RANDOM NUMBER OF BIDDERS

We now consider the case when the number of bidders m in each trial is a random variable M distributed according to a known discrete distribution Q over { 2 , 3 , 4 , . . . } . The assumption that Q is known is realistic: one can think of estimating it from historical data that might be provided by the auctioneer. On each trial, the value M = m is randomly generated according to Q , and the auctioneer collects m bids B 1 , B 2 , . . . , Bm . For given m , these bids are i.i.d. bounded random variables B ∈ [ 0 , 1 ] with unknown cumulative distribution F , which is the setting considered in Section II.

For simplicity, we assume that M is independent of the random variables Bi . For fixed M = m , we denote by B ( 1 ) m ≥ B ( 2 ) m ≥ · · · ≥ B ( m ) m the corresponding order statistics.

Our learning algorithm is the same as before: In each time step, the algorithm is requested to set reserve price p ∈ [ 0 , 1 ] and, for the given realization of M = m , only observes the value of the revenue function R m ( p ) = R ( p ; B 1 , B 2 , . . . , Bm ) defined as



<!-- formula-not-decoded -->

without knowing the specific value of m that generated this revenue. Namely, after playing price p the algorithm is observing an independent realization of the random variable R M ( p ) . The expected revenue µ( p ) is now

<!-- formula-not-decoded -->

where the inner expectation E [ · | M = m ] is over the random bids B 1 , B 2 , . . . , Bm .

Again, we want to minimize the expected regret with respect to the optimal reserve price

<!-- formula-not-decoded -->

for the bid distribution F , averaged over the distribution Q over the number of bidders M , where the expected regret over T time steps is

<!-- formula-not-decoded -->

and pt is the price set by the algorithm at time t . In Section IV-A we show that the same regret bound holds for the realized regret

<!-- formula-not-decoded -->

where Mt is the number of bidders at time t .

<!-- formula-not-decoded -->

Let F 2 , m denote the cumulative distribution function of B ( 2 ) m . We use E M [ F 2 , M ] ( x ) to denote the mixture distribution ∑ ∞ m = 2 Q ( m ) F 2 , m ( x ) . Likewise,

Relying on Fact 1, one can easily see that

∫

<!-- formula-not-decoded -->

As in Section II, our goal is to devise an online algorithm whose expected regret is of the order √ T , with as few assumptions as possible on F and Q .

We first extend Lemma 1 to handle this more general setting. 8

Lemma 4: Let T be the probability generating function of M,

<!-- formula-not-decoded -->

and define the auxiliary function

<!-- formula-not-decoded -->

where, for both functions, we let the argument x range in [0,1]. Then T and A are bijective mappings from [0,1] onto [0,1] and both T -1 and A -1 exist in [0,1]. Moreover, letting a ∈ ( 0 , 1 ) , and 0 ≤ /epsilon1 &lt; 1 -a, if x is such that then

<!-- formula-not-decoded -->

In addition, if 9

<!-- formula-not-decoded -->

holds for all x ∈ [ 0 , 1 ] then, for any a ∈ ( 0 , 1 ) and /epsilon1 ≥ 0 ,

--Observe that T ( · ) and A ( · ) in this lemma have been defined in such a way that 10

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(

) )

<!-- formula-not-decoded -->

and

Hence, E M [ F M ] ( p ) in (5) satisfies (

In particular, when P ( M = m ) = 1 as in Section II, we obtain T ( x ) = x m and A ( x ) = m x m -1 -( m -1 ) x m . Thus, in this case A ( T -1 ( · ) ) is the function β( · ) defined in Lemma 1, and the reconstruction function β -1 ( · ) we used throughout Section II is T ( A -1 ( · ) ) . Because this is a more general setting then the one in Section II, we do still have the technical issue of insuring that the argument of this recostruction function is not too close to 1.

As in the fixed m case, the algorithm proceeds in stages. In each stage i the algorithm samples the function E M [ F 2 , M ]

8 More precisely, in dealing with a more general setting we only obtain a slightly looser result than Lemma 1.

9 Condition (7) is a bit hard to interpret: It is equivalent to the convexity of the function T ( A -1 ( x )) for x ∈ [ 0 , 1 ] (see the proof of Lemma 4 in Appendix V), and it can be shown to be satisfied by many standard parametric families of discrete distributions Q , e.g., Uniform, Binomial, Poisson, Geometric. There are, however, examples where this condition does not hold. For instance, the distribution Q , where Q ( 2 ) = 0 . 4, Q ( 8 ) = 0 . 6, and Q ( m ) = 0 for any m /negationslash= 2 , 8 does not satisfy (7) for x = 0 . 6, i.e., it yields a function T ( A -1 ( x )) which is not convex on x = 0 . 6.

10 Recall from Section II-A that, for any fixed M = m , we have F 2 , m ( p ) = m ( F ( p )) m -1 ( 1 -F ( p )) + ( F ( p )) m .

<!-- formula-not-decoded -->

by sampling R M ( p ) at appropriate values of p . This allows it to build an empirical distribution ̂ F 2 , i and to reconstruct the two unknown functions E M [ F 2 , M ] and E M [ F M ] occurring in (5) over an interval of reserve prices that is likely to contain p ∗ . Whereas E M [ F 2 , M ] is handled directly, the reconstruction of E M [ F M ] requires us to step through the functions T and A according to the following scheme:

<!-- formula-not-decoded -->

Namely, in stage i we sample E M [ F 2 , M ] to obtain the empirical distribution ̂ F 2 , i , and then estimate E M [ F M ] in (5) through T ( A -1 ( ̂ F 2 , i ( · ))) .

With this notation in hand, the detailed description of the algorithm becomes very similar to the one in Section II-B. Hence, in what follows we only emphasize the differences, which are essentially due to the modified confidence interval delivered by Lemma 4, as compared to Lemma 1.

In order to emphasize that the role played by the composite function A ( T -1 ( · )) here is the very same as the function β( · ) in Section II, we overload the notation and define in this section β( x ) = A ( T -1 ( x )) , where T and A are given in Lemma 4. Moreover, we define for brevity ¯ F 2 ( x ) = E M [ F 2 , M ] ( x ) .

In particular, if we rely on (6), the new confidence interval size for Stage i depends on the empirical distribution ̂ F 2 , i through the quantity (we again overload the notation)

√

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Similarly, if we rely on (8), we have instead

√

<!-- formula-not-decoded -->

The resulting pseudocode is the same as in Algorithm 1, where the observations Rt ( ̂ pi ) therein have to be interpreted as distributed i.i.d. as R M ( ̂ pi ) , and E [ B ( 2 ) ] and F 2 in ̂ µ i are replaced by their M -average counterparts E M E [ B ( 2 ) M ] and ¯ F 2. We call the resulting algorithm the Generalized Algorithm 1.

As for the analysis, Lemma 2 is replaced by the following (because of notation overloading, the statement is the same as that of Lemma 2, but the involved quantities are different, and so is the proof in the appendix).

Lemma 5: With the notation introduced at the beginning of this section, if S = S ( T ) is (an upper bound on) the total number of stages, we have that, for any fixed stage i,

√

<!-- formula-not-decoded -->

holds with probability at least 1 -δ/( 3 S ) , uniformly over p ∈ [ ̂ pi , 1 ] , conditioned on all past stages.

Then an easy adaptation of Lemma 3 leads to the following expected regret bound. The proof is very similar to the proof of Theorem 1, and is therefore omitted.

Theorem 3: With the notation introduced at the beginning of this section, for any pair of distributions F and Q such that µ( p ∗ ) &gt; 0 we have that the Generalized Algorithm 1, operating on any time horizon T satisfying

(

)

<!-- formula-not-decoded -->

)

and with approximation parameter α ≥ µ( p ∗ ) 2 / 12 , has regret ( √

<!-- formula-not-decoded -->

with probability at least 1 -δ , where A = E [ M ] if (6) holds and A = 1 if (8) holds.

## IV. EXTENSIONS

This section further extends the results contained in the previous two sections. First, we show how to bound with high probability the realized regret (Section IV-A). Second, we show how to turn Algorithm 1 into an algorithm that does not rely on prior knowledge of the time horizon T (Section IV-B).

## A. Bounding the Realized Regret

In this section, we show how to bound in probability the realized regret

<!-- formula-not-decoded -->

suffered by the Generalized Algorithm 1. As a special case, this clearly applies to Algorithm 1, too.

We need the following definitions and results from empirical process theory-see [19]. Let F be a set of [ 0 , 1 ] -valued functions defined on a common domain X . We say that F shatters a sequence x 1 , . . . , xn ∈ X if there exists r 1 , . . . , rn ∈ R such that for each ( a 1 , . . . , an ) ∈ { 0 , 1 } n there exists f ∈ F for which f ( xi ) &gt; ri iff ai = 1 for all i = 1 , . . . , n . The pseudo-dimension [17] of F , which is defined as the length of the longest sequence shattered by F , controls the rate of uniform convergence of means to expectations in F . This is established by the following known lemma, which combines Dudley's entropy bound with a bound on the metric entropy of F in terms of the pseudo-dimension-see [18], [19].

Lemma 6: Let X 1 , X 2 , . . . be i.i.d. random variables defined on a common probability space and taking values in X . There exists a universal constant C &gt; 0 such that, for any fixed T and δ ,

∣

∣

√

<!-- formula-not-decoded -->

hm to

-=

For Block r = 0, 1, 2.

(JT).

VT log -

Algorithm 2 Anytime Regret Minimizer

Run Algorithm 1 with input parameters

(T log log T

)+'

VT log log T

log T

## Algorithm 2 Anytime Regret Minimizer

Input: Confidence level δ ( 0 , 1 , approximation parameter α ( 0 , 1 ;

For Block r = 0 , 1 , 2

∈ ] ∈ ] , . . .

Run Algorithm 1 with input parameters

<!-- formula-not-decoded -->

with probability at least 1 -δ , where d is the pseudo-dimension of F . [ ]

Recall that E M E R M ( p ) = µ( p ) for all p ∈ [ 0 , 1 ] . Let R = { R M ( p ) : p ∈ [ 0 , 1 ] } be the class of revenue functions indexed by reserve prices p ∈ [ 0 , 1 ] . Hence, for each p , R M ( p ) is a [ 0 , 1 ] -valued function of the number M of bidders and the bids B 1 , . . . , BM . In the appendix we prove the following bound.

Lemma 7: The pseudo-dimension of the class R is 2 .

As announced, the following is the main result of this section, whose proof combines Lemma 6, Lemma 7, together with a standard martingale argument.

Theorem 4: Under the assumptions of Theorem 3 (Section III), the actual regret of Generalized Algorithm 1 satisfies

(√

)

<!-- formula-not-decoded -->

with probability at least 1 -δ , where A = E [ M ] if (6) holds and A = 1 if (8) holds.

## B. The Case of Unknown Time Horizon T

We use a standard 'doubling trick' argumentsee [5, Sec. 2.3] applied to Algorithm 1 (the same argument applies to the Generalized Algorithm 1). The idea is to partition the sequence 1 , 2 , . . . of time steps into blocks of geometrically growing length, where each block r = 0 , 1 , . . . starts at time 2 r and ends at time 2 r + 1 -1. At the beginning of each new block r , we restart Algorithm 1 from scratch, using Tr = 2 r + 1 -1 -2 r + 1 = 2 r as new horizon parameter, and setting the current confidence level as δ r = δ / ( r + 1 )( r + 2 ) , where δ is the desired confinence level. The algorithm's pseudocode is given in Algorithm 2.

Using the standard analysis for the doubling trick, Algorithm 2 is easily see to achieve the following bound on the expected regret.

√

)

Theorem 5: For any distribution F of the bids and any m ≥ 2 such that µ( p ∗ ) &gt; 0 , we have that Algorithm 2 using approximation parameter α ≥ µ( p ∗ ) 2 / 12 has regret O ( 1 µ( p ∗ ) 8 log 2 T log log T δ + √ T log log T µ( p ∗ ) log log T δ

(

)

with probability at least 1 -δ simultaneously over all T .

Clearly enough, combining with Theorem 4 a similar statement can be given for the realized regret as well.

## V. CONCLUSIONS AND DISCUSSION

Optimizing the reserve price in a second-price auction is an important theoretical and practical concern. We introduced a regret minimization algorithm to optimize the reserve price incurring a regret of only ˜ O ( √ T ) . We showed the result both for the case where the number of bidders is known, and for the case where the number of bidders is drawn from a known distribution. The former assumption, of known fixed number of bidders, is applicable when the number of bidders is given as the outcome of the auction. The assumption that the distribution over the number of bidders is known is rather realistic, even in the case where the number of participating bidders is not given explicitly. For example, one can hope to estimate such data from historical data that might be made available from the auctioneer.

Our optimization of the reserve prices depends only on observable outcomes of the auction. Specifically, we need only observe the seller's actual revenue at each step. This is important in many applications, such as e-Bay, AdX or AdSense, where the auctioneer is a different entity from the seller, and provides the seller with only a limited amount of information regarding the actual auction. It is also important that we make no assumptions about the distribution of the bidder's bid (or its relationship to the bidder's valuation) since many such assumptions are violated in reality. The only assumption that we do make is that the distributions of the bidders are identical. This assumption is a fairly good approximation of reality in many cases where the seller conducts a large number of auctions and bidders rarely participate in a large number of them.

The resulting algorithm is very simple at a high level, and potentially attractive to implement in practice. Conceptually, we would like to estimate the optimal reserve price. The main issue is that if we simply exploit the current best estimate, we might miss essential exploration. This is why, instead of playing the current best estimate, the algorithm plays a minimal /epsilon1 -optimal reserve price, where /epsilon1 shrinks over time. The importance of playing the minimal near-optimal reserve price is that it allows for efficient exploration of the prices, due to the specific feedback model provided by the second-price auction setting.

An interesting direction for extending our results is the generalized second price auction model, when multiple items of different quality are sold at each step. Here the problem of estimating the expected revenue function becomes more involved due to the presence of terms that depend on the correlation of order statistics.

A different open issue, of more technical nature, is whether the inverse dependence on µ( p ∗ ) in Theorem 1 (and on µ( p ∗ ) 2 in Theorem 3) can somehow be removed. Indeed, these factors do not seem to be inherent to the problem itself, but only to the kind of algorithms we use.

In a similar vein, because the number of bidders (Section II) or the distribution on the number of bidders (Section III under assumption (8)) does not explicitly show up in the regret bounds, one may wonder whether our algorithm really needs to know this information. Unfortunately, the answer seems to be affirmative, as our algorithm hinges on reconstructing the underlying bid distribution F ( · ) from the distribution of the second-highest bid F 2 ( · ) , and we are currently unaware of how this could be done without knowing m (or its distribution). A simple attempt to remove the dependence on m from Algorithm 1 is to let m → ∞ in the reconstruction function β( · ) . The resulting β( x ) would still be well defined, since lim m →∞ β( x ) = x -x log x , uniformly over x ∈ [ 0 , 1 ] , and Lemma 1 would still hold since its statement is independent of m . However, we would no longer be optimizing the 'right' function ̂ µ i ( p ) but an approximation thereof, the error in this approximation propagating across time steps in an additive manner, so that at the end of T steps we would obtain a linear regret bound of the form ˜ O ( dm T + √ T ) , where dm is constant with T , and goes to zero as the true underlying m goes to infinity. The question whether it is possible to refine this simple argument so as to achieve a nontrivial (i.e., sublinear in T ) cumulative regret bound without knowing anything about m remains open.

## APPENDIX MAIN PROOFS

Proof of Fact 1: By definition of R ( p ) we can write

∫

<!-- formula-not-decoded -->

∫

By applying the identity E [ X ] = P ( X &gt; x ) dx to the nonnegative random variable B ( 2 ) I { B ( 2 ) &gt; p } we obtain

∫

<!-- formula-not-decoded -->

Moreover, and

(

)

<!-- formula-not-decoded -->

(

)

(

)

<!-- formula-not-decoded -->

Substituting the above into (9) and simplifying concludes the proof. /square

Proof of Lemma 1: A simple derivative argument shows that the function β( · ) is a strictly increasing and concave mapping from [ 0 , 1 ] onto [ 0 , 1 ] . Hence its inverse β -1 ( · ) exists and is strictly increasing and convex on [ 0 , 1 ] . From our assumptions we immediately have: (i) x ≤ β -1 ( a + /epsilon1) for any /epsilon1 ∈ [ 0 , 1 -a ] , and (ii) β -1 ( a -/epsilon1) ≤ x for any /epsilon1 ∈ [ 0 , a ] .

<!-- formula-not-decoded -->

In turn, because of the convexity of β -1 ( · ) , we have

Similarly, by the convexity and the monotonicity of β -1 ( · ) we can write

∣

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At this point, we need the following technical claim. Claim 1:

Proof of Claim: Note that the case a ∈ [ 0 , 3 / 4 ) holds trivially since 2 √ 1 -a ≥ 1, and therefore we need to consider only a ∈ [ 3 / 4 , 1 ] . Introduce the auxiliary function f ( a ) = 1 -2 √ 1 -a . The claim is proven by showing that β( f ( a )) ≤ a for all a ∈ [ 3 / 4 , 1 ] . We prove the claim by showing that β( f ( a )) is a concave function of a ∈ [ 3 / 4 , 1 ] , ∣

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, using L'Hopital's rule,

<!-- formula-not-decoded -->

(

)

<!-- formula-not-decoded -->

since m ≥ 2. Moreover, which is nonpositive if and only if m (( f ( a )) -1 / m -1 ) ≤ ( 1 -f ( a )) ( f ( a )) -m + 1 m holds for any a ∈ [ 3 / 4 , 1 ] . Since f ( a ) ranges in [ 0 , 1 ] when a ∈ [ 3 / 4 , 1 ] , after some simplifications, one can see that the above inequality is equivalent to

<!-- formula-not-decoded -->

In turn, this inequality can be seen to hold by showing via a simple derivative argument that the function g ( x ) = m x m + 1 m + 1 is convex and increasing for x ∈ [ 0 , 1 ] , while g ( 0 ) = 1 &gt; 0 and g ′ ( 1 ) = m + 1. /square

The claim together with (10) and (11) allows us to conclude the proof of Lemma 1. Specifically, the second inequality in (2) is obtained by (10) and extended to any /epsilon1 ≥ 0 just by observing that, by the claim, for /epsilon1 &gt; 1 -a the rightmost side of (2) is larger than 1. Moreover, the first inequality in (2) is obtained by (11) and extended to any /epsilon1 ≥ 0 by observing that for /epsilon1 &gt; a the left-most side of (2) is smaller than β -1 ( a ) -2 a √ 1 -a ≤ a -2 a √ 1 -a ≤ 0 for any a ∈ [ 0 , 1 ] , where we have used the fact that β -1 ( a ) ≤ a . /square

(

Proof of Lemma 2: Let B ( 1 ) k and B ( 2 ) k denote the maximum and the second-maximum of k i.i.d. bids B 1 , . . . , Bk . Set for brevity A = P ( B ( 1 ) m &gt; p ) . Then we have

(

)

<!-- formula-not-decoded -->

and

)

(

)

(

)

<!-- formula-not-decoded -->

Hence

(

)

<!-- formula-not-decoded -->

In turn, A ≥ µ( p ) , since each time all the bids are less than p the revenue is zero. Therefore we have obtained that

<!-- formula-not-decoded -->

holds for all p ∈ [ 0 , 1 ] . Finally, since ̂ F 2 , i is the empirical version of F 2 based on the observed revenues during stage i (see Section II-C), the classical Dvoretzky-KieferWolfowitz (DKW) inequality [13] implies that with probability at least 1 -δ/ 3 S , conditioned on all past stages, √

<!-- formula-not-decoded -->

Proof of Lemma 3: We start by proving (3). Fix any stage i and write ∣ ∫ ∫ ∣

<!-- formula-not-decoded -->

The DKW inequality implies that √

<!-- formula-not-decoded -->

holds with probability at least 1 -δ/( 3 S ) . As for the second term in (12) we apply again the DKW inequality in combination with Lemma 1 with x = ( F ( p )) m = β -1 ( F 2 ( p ) ) , a ̂ F 2 , i ( p ) , and /epsilon1 √ 1 ln 6 S . This yields

) ∣

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with the same probability of at least 1 -δ/( 3 S ) . Putting together and using the union bound over the S stages gives (3).

We prove (4) by induction on i = 1 , . . . , S . We first show that the base case i = 1 holds with probability at least 1 -δ/ S .

Then we show that if (4) holds for i -1, then it holds for i with probability at least 1 -δ/ S over all random events in stage i . Therefore, using a union bound over i = 1 , . . . , S we get that (4) holds simultaneously for all i with probability at least 1 -δ .

For the base case i = 1 note that ̂ µ 1 ( p ∗ ) ≤ ̂ µ 1 ( ̂ p ∗ 1 ) holds with probability at least 1 -δ/( 3 S ) because we are assuming (Lemma 2) that ̂ F 2 ( p ∗ ) ≤ 1 -α holds with the same probability, and so ̂ p ∗ 1 maximizes ̂ µ 1 over a range that with probability at least 1 -δ/( 3 S ) contains p ∗ . Moreover, using (3) we obtain

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

as required. Finally, p ∗ ≥ ̂ p 1 trivially holds because ̂ p 1 = 0.

̂ µ 1 ( ̂ p ∗ 1 ) -µ( ̂ p ∗ 1 ) ≤ 2 C δ, 1 ( ̂ p ∗ 1 ). Since µ( ̂ p ∗ ) -µ( p ∗ ) ≤ 0 by definition of p ∗ ≤ 1 -≤ 1 +

We now prove (4) for i &gt; 1 using the inductive assumption p ∗ ≥ ̂ pi -1 and

The inductive assumption and ̂ F 2 , i ( p ∗ ) ≤ 1 -α directly imply p ∗ ∈ Pi ⋂{ p : ̂ F 2 , i -1 ( p ) ≤ 1 -α } (recall the definition of the set of candidate prices Pi given in Algorithm 1). Thus we have p ∗ ≥ ̂ pi and ̂ µ i ( ̂ p ∗ i ) ≥ ̂ µ i ( p ∗ ) , because ̂ p ∗ i maximizes ̂ µ i over a range that contains p ∗ . The rest of the proof closely follows that of (4) for the base case i = 1. /square

<!-- formula-not-decoded -->

Proof of Theorem 1: If S = S ( T ) is the total number of stages, then the regret of our algorithm is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For all stages i &gt; 1 the following chain on inequalities jointly hold with probability at least 1 -δ uniformly over i = 2 , . . . , S , where in the last step we used the fact that ̂ F 2 , i -1 ( p ∗ ) ≤ 1 -α holds by Lemma 2, and that ̂ F 2 , i -1 ( p ) ≤ 1 -α for p = ̂ pi and

EMI|

EMI

€

IS

dI (A-'(x))

alA '(x))

. More that 1

€

EEMI

al(A "(x))

I' (У)I (у)

dI(A (x))

1- 1(A'(a))

(x).1.

alA '(x))

1 -2a.

dr

VT a

ST.

1 \_(adc)

72 In(65 /б)

€

+.

€

+

+

1—0.

dr

1\_(a+e

1 \_ 7-1(a)

dia=

VTS

6S

dr2

dr

&lt;\_1-16010

In

VT =

и (n*)4

VT (log log log T + log 1/8) (log log T)

(1 \_ v)2 (THI(v))3°

de??

dr

1 — a a T.

- In

=

( — W)

1 \_ T-1(adc)

и (n*)

p = ̂ p ∗ i -1 by the very definitions of ̂ pi and ̂ p ∗ i -1 , respectively. Substituting back into (14) we see that with probability at least 1 -δ the regret of our algorithm is at most

√

<!-- formula-not-decoded -->

Our setting Ti = T 1 -2 -i for i = 1 , 2 , . . . implies that S is upper bounded by the minimum integer n such that

<!-- formula-not-decoded -->

Since i ≥ log 2 log 2 T makes Ti ≥ T 2 , then S ≤ /ceilingleft 2 log 2 log 2 T /ceilingright = O ( log log T ) . Moreover, observe that Ti = T 1 -2 -i is equivalent to T 1 = √ T and Ti √ Ti -1 = √ T , for i &gt; 1. We therefore have the upper bound

√

<!-- formula-not-decoded -->

If µ( p ∗ ) &gt; 0 and

<!-- formula-not-decoded -->

then α ≥ µ( p ∗ ) 2 / 12, and the above is of order √

<!-- formula-not-decoded -->

as claimed.

Proof of Lemma 4: We start by observing that T ( 0 ) = A ( 0 ) = 0, T ( 1 ) = A ( 1 ) = 1, T ′ ( x ) ≥ 0 for x ∈ [ 0 , 1 ] , and A ′ ( x ) = ( 1 -x ) T ′′ ( x ) ≥ 0 when x ∈ [ 0 , 1 ] . Hence both T ( x ) and A ( x ) are strictly increasing mappings from [0,1] onto [0,1], and so are T -1 ( x ) , A -1 ( x ) and A ( T -1 ( x )) . Hence our assumptions on x can be rewritten as

/square

<!-- formula-not-decoded -->

∣

<!-- formula-not-decoded -->

Moreover, since T ( · ) and A ( · ) are both C ∞ ( 0 , 1 ) , so is T ( A -1 ( · )) . Let /epsilon1 &lt; 1 -a . We can write for some ξ ∈ ( a , a + /epsilon1) , where

<!-- formula-not-decoded -->

∑

<!-- formula-not-decoded -->

and we set for brevity y = A -1 ( x ) ∈ [ 0 , 1 ] . Now, for any y ∈ [ 0 , 1 ] ,

As a consequence, since A -1 is a nondecreasing function, we can write

∣

<!-- formula-not-decoded -->

1 (У)

the last inequality deriving from 11 A ( x ) ≥ T ( x ) for all x ∈ [ 0 , 1 ] . Finally, from the convexity of T we have T ( x ) ≥ T ( 1 ) + ( x -1 ) T ′ ( 1 ) = 1 + ( x -1 ) E [ M ] . Thus T -1 ( x ) ≤ 1 -1 -x E [ M ] , x ∈ [ 0 , 1 ] , which we plug back into (16) to see that ∣

<!-- formula-not-decoded -->

Replacing backwards, this yields the second inequality of (6).

∣

<!-- formula-not-decoded -->

To prove the first inequality of (6), we start off showing it to hold for /epsilon1 &lt; min { a , 1 -a } , and then extend it to /epsilon1 &lt; 1 -a . Set /epsilon1 &lt; a . Then proceeding as above we can see that, for some ξ ∈ ( a -/epsilon1 , a ) , the last inequality requiring also /epsilon1 &lt; 1 -a . If now /epsilon1 satisfies a ≤ /epsilon1 &lt; 1 -a (assuming a &lt; 1 / 2) then the first inequality of (6) is trivially fulfilled. In fact,

<!-- formula-not-decoded -->

since E [ M ] ≥ 2. This concludes the proof of (6).

In order to prove (8), we set for brevity y = A -1 ( x ) , and using the rules of differentiating inverse functions, we see that

<!-- formula-not-decoded -->

Thus d 2 dx 2 T ( A -1 ( x )) ≥ 0 for x ∈ [ 0 , 1 ] is equivalent to

<!-- formula-not-decoded -->

Since y ranges over [0,1] when x does, (7) is actually equivalent to the convexity of T ( A -1 ( x )) on x ∈ [ 0 , 1 ] . Under the above convexity assumption, we can write, for /epsilon1 ≤ 1 -a ,

On the other hand, if /epsilon1 &gt; 1 -a the above inequality vacuously holds, since the right-hand side is larger than one, while T ( A -1 ( x )) ≤ 1 for any x ∈ [ 0 , 1 ] . This proves the

<!-- formula-not-decoded -->

11 Whereas the function A ( · ) is, in general, neither convex nor concave, T ( · ) is a convex lower bound on A ( · ) .

+

((x),.

1- 1(A (a))

a

+

и (₴)

+

" (2)

=()»

+

1)&amp;

12d

AIl!

a

1

1

4

+7=17=

second inequality in (8). Similarly, by the convexity and the monotonicity of T ( A -1 ( · )) we can write, for all /epsilon1 ∈ [ 0 , a ] ,

∣

<!-- formula-not-decoded -->

which gives the first inequality in (8). We extend the above to any /epsilon1 ≥ 0 by simply observing that /epsilon1 &gt; a implies that T ( A -1 ( a )) -/epsilon1 1 -a &lt; a -a 1 -a &lt; 0, where T ( A -1 ( a )) ≤ a follows from the convexity of T ( A -1 ( · )) . This makes (8) trivially fulfilled. /square

Proof of Lemma 5: Let B ( 1 ) m and B ( 2 ) m denote the highest and the second-highest of m i.i.d. bids B 1 , . . . , Bm . Recall from the proof of Lemma 2 that, for any m ≥ 2

(

)

(

)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

[

i

nit

NI-

- T-а

0, 7),

12).

NIn

UI-

a

and

Moreover,

) ]

(

∣

<!-- formula-not-decoded -->

the second-last inequality being Jensen's, and the last one deriving from I { B ( 1 ) m &gt; p } ≥ R m ( p ) for all m ≥ 2 and p ∈ [ 0 , 1 ] . We then conclude as in the proof of Lemma 2 by applying DKW on the uniform convergence of ̂ F 2 , i to ¯ F 2. /square

Proof of Theorem 2: Consider a setting with two bidders ( m = 2) where both bids B 1 , B 2 are revealed at the end of each auction, irrespective of the chosen reserve price. Note that a lower bound in this setting implies a lower bound in the harder setting of Theorem 1, in which only the revenue is revealed.

Consider bid distributions of the form

(

)

(

)

<!-- formula-not-decoded -->

{

}

Since the bid distribution is supported on 1 2 , 3 4 , the expected revenue of a reserve price 0 &lt; p &lt; 1 2 is never greater than that of p = 1 2 . Similarly, the expected revenue of a reserve price 1 2 &lt; p &lt; 3 4 is never greater than that of p = 3 4 . Therefore, without loss of generality we may restrict our attention to strategies that select their prices from { 1 2 , 3 4 } .

(

We now compute the expected revenue of p = 1 2 and p = 3 4 , ( ) ( )

(

)

<!-- formula-not-decoded -->

Therefore

<!-- formula-not-decoded -->

)

(

)

For any fixed /epsilon1 ∈ 0 , 1 4 , consider now the random variable Z ∈ {-1 , + 1 } such that P ( Z = + 1 ) = P ( Z = -1 ) = 1 2 and let µ P be the expected revenue function of the bid distribution

{

<!-- formula-not-decoded -->

We now prove that no deterministic mechanism can have small regret on both conditional bid distributions P ( · | Z = + 1 ) and

<!-- formula-not-decoded -->

(

)

(

)

<!-- formula-not-decoded -->

(

)

(

)

<!-- formula-not-decoded -->

Moreover, switching /epsilon1 to -/epsilon1 gives

Since /epsilon1 is chosen of the order of T -1 / 2 , in the rest of the proof we may ignore the term /epsilon1 2 appearing in the expected revenue function µ P . This adds a constant to the regret, which is taken into account by the asymptotic notation. Now let p ∗ = p ∗ ( Z ) be the optimal reserve price for the conditional bid distribution. That is, p ∗ = 1 2 if Z = + 1 and p ∗ = 3 4 if Z = -1.

Fix any deterministic algorithm choosing reserve prices p 1 , p 2 , . . . Let T 1 / 2 and T 3 / 4 be the number of times pt = 1 2 and pt = 3 4 , respectively. Finally, let T ∗ be the number of times pt = p ∗ . Because the regret increases by /epsilon1 2 every time pt /negationslash= p ∗ (recall that we are ignoring the /epsilon1 2 term in µ P ), the expected regret of the algorithm with respect to the worst-case choice of Z is

<!-- formula-not-decoded -->

In this simplified setting, where the mechanism can observe the individual bids, each pt is determined by the independent bid pairs ( B ( 1 ) 1 , B ( 2 ) 1 ) , . . . , ( B ( 1 ) T , B ( 2 ) T ) . Let P + T and P -T be the joint distributions of the bid pairs when Z = + 1 and

(

)

Z = -1, respectively. Finally, let PT = 1 2 P + T + P -T . Then, Pinsker's inequality implies √

<!-- formula-not-decoded -->

where we used the convexity of the relative entropy KL · ‖ P + T in the last step.

2 + 2 -

)

We now recognize that P + T and P -T are product distributions of T pairs of (shifted and scaled) independent Bernoulli random variables B 1 /epsilon1 and B 1 /epsilon1 , with parameters 1 2 + /epsilon1 and

1 2 -/epsilon1 , respectively. Therefore, taking the scaling factor into account and using the chain rule for relative entropy gives ( )

(

)

(

)

<!-- formula-not-decoded -->

the last inequality holding if /epsilon1 ∈ 0 , 0 . 47 . Hence √

<!-- formula-not-decoded -->

Similarly,

<!-- formula-not-decoded -->

Therefore

(

)

<!-- formula-not-decoded -->

This implies

<!-- formula-not-decoded -->

Proof of Lemma 7: Since the revenue R M ( p ) is determined by B ( 1 ) M and B ( 2 ) M only, we use the notation Rp ( b 1 , b 2 ) to denote the revenue R M ( p ) when B ( 1 ) M = b 1 and B ( 2 ) M = b 2. Since b 1 ≥ b 2, in order to compute the pseudo-dimension of F we have to determine the largest number of points shattered in the region S = { ( b 1 , b 2 ) : 0 ≤ b 2 ≤ b 1 ≤ 1 } ⊂ R 2 where the functions Rp are defined as 

Choosing /epsilon1 = /Theta1 T -1 / 2 concludes the proof of the theorem. /square

<!-- formula-not-decoded -->

Note that each function Rp defines an axis-parallel rectangle with corners ( p , p ) , ( p , 0 ) , ( 1 , p ) and ( 1 , 0 ) . Inside the rectangle Rp = p , to the left of the rectangle Rp = 0, and points ( b 1 , b 2 ) ∈ S above it satisfy Rp ( b 1 , b 2 ) = b 2.

We now show that F shatters any two points x 1 = ( b 1 , b 2 ) and x 2 = ( b 1 + /epsilon1 , b 2 + /epsilon1) in the region S such that /epsilon1 &gt; 0 and b 2 + /epsilon1 &lt; b 1. This is shown by the following case analysis where we use the values r 1 = b 2 and r 2 = b 2 + /epsilon1 .

- b 1 &lt; p &lt; b 1 + /epsilon1 , this realizes the pattern ( 0 , 1 ) because Rp ( x 1 ) = 0 ≤ b 2 and Rp ( x 2 ) = p &gt; b 2 + /epsilon1 ;
- p &gt; b 1 + /epsilon1 , this realizes the pattern ( 0 , 0 ) because Rp ( x 1 ) = 0 ≤ b 2 and Rp ( x 2 ) = 0 ≤ b 2 + /epsilon1 ;
- b 2 + /epsilon1 &lt; p &lt; b 1, this realizes the pattern ( 1 , 1 ) because Rp ( x 1 ) = p &gt; b 2 and Rp ( x 2 ) = p &gt; b 2 + /epsilon1 ;

In order to prove that F can not shatter any three points in S , arrange on the real line the six coordinate values of these three points. These six numbers define seven intervals. When p ranges within any such interval, the value of Rp must remain constant on all the three points. This is because the value of Rp ( b 1 , b 2 ) changes only when p crosses b 1 or b 2. But then, F can only realize at most seven of the eight patterns needed to shatter the three points. /square

- b 2 &lt; p &lt; b 2 + /epsilon1 , this realizes the pattern ( 1 , 0 ) because Rp ( x 1 ) = p &gt; b 2 and Rp ( x 2 ) = b 2 + /epsilon1 .

Proof of Theorem 4: For the sake of brevity, let Rt ( p ) denote R Mt t ( p ) . Also, let E t [·] be the conditional expectation E t [ · | p 1 , . . . , pt -1 ] , i.e., the expectation of the random variable at argument conditioned on all past bids and all past number of bidders. Let p ∗ T be the random variable defined as

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In order to bound (18) we combine Lemma 6 with Lemma 7. This gives

(

)

√

<!-- formula-not-decoded -->

with probability at least 1 -δ , where C is the constant mentioned in Lemma 6. [ ]

In order to bound (19), note that Zt = E t Rt ( pt ) -Rt ( pt ) for t = 1 , 2 , . . . is a martingale difference sequence with

&lt;

12 In (65rr/Orr)

1

6Srr

(log log T

72 In (65, /б,)

1 8*

u (n*)

/log log T + log 1/6)

log log T

и (n*)

/og log 7 + 10g 1/8)

65,

VT, ≥

и (n*)4

1

u (n*) 4

In v2T In -

VT log log T

In

S.

In

65,

S.

1 In

8 Sr

In

Mog log 7 + log 1/6)

bounded increments, E t [ Zt ] = 0 with Zt ∈ [-1 , 1 ] for each t . Therefore, the Hoeffding-Azuma inequality for martingales establishes that

√

<!-- formula-not-decoded -->

Finally, term (20) is bounded via Theorem 3 after observing that µ( p ∗ T ) ≤ µ( p ∗ ) , where p ∗ = argmax p ∈[ 0 , 1 ] µ( p ) is the maximizer of the expected revenue. This concludes the proof. /square with probability at least 1 -δ .

Proof of Theorem 5: Following (15), the regret accumulated in block r is upper bounded by

(

√

)

<!-- formula-not-decoded -->

with probability at least 1 -δ r , where Sr ≤ /ceilingleft 2 log 2 log 2 Tr /ceilingright = O ( log r ) , provided µ( p ∗ ) &gt; 0 and √

<!-- formula-not-decoded -->

On the other hand, if (21) is false, then we can simply upper bound the cumulative regret in block r by its length Tr .

Because the algorithm is restarted at the beginning of each block, these cumulative regrets have to be summed over blocks. Let rT be the index of the final block. Clearly, if the total number of steps is T then TrT ≤ 2 T , and rT ≤ 1 + log 2 T . Moreover, denote by ¯ r the larger r such that (21) is false. Taking a union bound over blocks (and noting that ∑ r δ r ≤ δ ), we have that with probability at least 1 -δ the cumulative regret of Algorithm 1 is upper bounded by

(

√

)

<!-- formula-not-decoded -->

We upper bound the two sums in the above expression separately. We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

But by its very definition ¯ r also satisfies so that the first sum can be overapproximated as

(

(

))

<!-- formula-not-decoded -->

As for the second sum, we can write

(

√

)

<!-- formula-not-decoded -->

(

√

)

<!-- formula-not-decoded -->

Putting together proves the claimed bound.

## REFERENCES

- [1] S. Athey and P. A. Haile, 'Identification of standard auction models,' Econometrica , vol. 70, no. 6, pp. 2107-2140, 2002.
- [2] P. Auer, R. Ortner, and C. SzepesvÆri, 'Improved rates for the stochastic continuum-armed bandit problem,' in Proc. 20th Annu. Conf. Learn. Theory , 2007, pp. 454-468.
- [3] P. Auer, N. Cesa-Bianchi, and P. Fischer, 'Finite-time analysis of the multiarmed bandit problem,' Mach. Learn. , vol. 47, nos. 2-3, pp. 235-256, 2002.
- [4] S. Bubeck, R. Munos, G. Stoltz, and C. SzepesvÆri, ' X -armed bandits,' J. Mach. Learn. Res. , vol. 12, pp. 1587-1627, Feb. 2011.
- [5] N. Cesa-Bianchi and G. Lugosi, Prediction, Learning, and Games . Cambridge, U.K.: Cambridge Univ. Press, 2006.
- [6] E. W. Cope, 'Regret and convergence bounds for a class of continuumarmed bandit problems,' IEEE Trans. Autom. Control , vol. 54, no. 6, pp. 1243-1253, Jun. 2009.
- [7] E. David, A. Rogers, N. R. Jennings, J. Schiff, S. Kraus, and M. H. Rothkopf, 'Optimal design of English auctions with discrete bid levels,' ACM Trans. Internet Technol. , vol. 7, no. 2, 2007, Art. ID 12.
- [8] P. Dhangwotnotai, T. Roughgarden, and Q. Yan, 'Revenue maximization with a single sample,' in Proc. 11th ACM Conf. Electron. Commerce , 2010, pp. 129-138.
- [9] P. A. Haile and E. Tamer, 'Inference with an incomplete model of English auctions,' J. Political Economy , vol. 111, no. 1, pp. 1-51, 2003.
- [10] J. D. Hartline and A. R. Karlin, 'Profit maximization in machanism design,' in Algorithmic Game Theory , N. Nisan, T. Roughgarden, E. Tardos, and V. V. Vazirani, Eds. Cambridge, U.K.: Cambridge Univ. Press, 2007.
- [11] A. X. Jiang and K. Leyton-Brown, 'Bidding agents for online auctions with hidden bids,' Mach. Learn. , vol. 67, nos. 1-2, pp. 117-143, 2007.
- [12] R. Kleinberg and T. Leighton, 'The value of knowing a demand curve: Bounds on regret for online posted-price auctions,' in Proc. 44th Annu. IEEE Symp. Found. Comput. Sci. , Oct. 2003, pp. 594-605.
- [13] P. Massart, 'The tight constant in the Dvoretzky-Kiefer-Wolfowitz inequality,' Ann. Probab. , vol. 18, no. 3, pp. 1269-1283, 1990.
- [14] R. B. Myerson, 'Optimal auction design,' Math. Oper. Res. , vol. 6, no. 1, pp. 58-73, 1981.
- [15] N. Nisan, 'Introduction to mechanism design (for computer scientists),' in Algorithmic Game Theory , N. Nisan, T. Roughgarden, E. Tardos, and V. V. Vazirani, Eds. Cambridge, U.K.: Cambridge Univ. Press, 2007.
- [16] M. Ostrovsky and M. Schwarz, 'Reserve prices in internet advertising auctions: A field experiment,' in Proc. 12th ACM Conf. Electron. Commerce , 2011, pp. 59-60.
- [17] D. Pollard, Empirical Processes: Theory and Applications (Probability and Statistics), vol. 2. Hayward, CA, USA: Institute of Mathematical Statistics, 1990.
- [18] A. W. van der Vaart and J. A. Wellner, Stochastic Convergence and Empirical Processes , 2nd ed. New York, NY, USA: Springer-Verlag, 2011.
- [19] A. W. van der Vaart and J. A. Wellner, Weak Convergence and Empirical Processes . New York, NY, USA: Springer-Verlag, 1996.
- [20] W. E. Walsh, D. C. Parkes, T. Sandholm, and C. Boutilier, 'Computing reserve prices and identifying the value distribution in real-world auctions with market disruptions,' in Proc. 23rd AAAI Conf. Artif. Intell. , 2008, pp. 1499-1502.
- [21] J. Y. Yu and S. Mannor, 'Unimodal bandits,' in Proc. 28th Int. Conf. Mach. Learn. , 2011, pp. 41-48.

/square

Nicolò Cesa-Bianchi is professor of Computer Science at the University of Milano, Italy, where he is currently director the Computer Science programs. He was President of the Association for Computational Learning and member of the steering committee of the EC-funded Network of Excellence PASCAL2. He served as action editor for the Machine Learning Journal, for IEEE TRANSACTIONSON INFORMATIONTHEORY, and for the Journal of Machine Learning Research . He is currently associate editor for the Journal of Information and Inference and member of the Board of Directors of the Association for Computational Learning. He was program chair of the 13th Annual Conference on Computational Learning Theory and of the 13th International Conference on Algorithmic Learning Theory. He held visiting positions with UC Santa Cruz, Graz Technical University, Ecole Normale Superieure in Paris, Google, and Microsoft Research. His main research interests include statistical learning theory, online learning, sequential prediction. He is coauthor of the monographs Prediction, Learning, and Games and Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems . He is recipient of a Google Research Award and of a Xerox Foundation UAC Award.

Claudio Gentile is currently associate professor in Computer Science at the University of Insubria, Italy, where he is also coordinating the PhD program in Computer Science and Computational Mathematics. His current research interests are in Online Learning, Big Data analysis, Machine Learning on networked data, and applications thereof. He is member of the editorial board of the Machine Learning Journal from 2006, member of the editorial board of the Journal of Machine Learning Research from 2009, steering committee member of EU-funded Networks of Excellence projects PASCAL and PASCAL2, PC chair, Area Chair or Senior PC member of several international conferences in the field of Machine Learning/Artificial Intelligence/Data Mining (e.g., COLT, NIPS, IJCAI, ECML/PKDD). He was visiting researcher at UC Santa Cruz, Microsoft Research, Technion, University College London, Tel-Aviv University, INRIA Lille, Telefonica, Amazon.

Yishay Mansour got his PhD from MIT in 1990, following it he was a postdoctoral fellow in Harvard and a Research Staff Member in IBM T.J. Watson Research Center. Since 1992 he is at Tel-Aviv University, where he is currently a Professor of Computer Science and has serves as the first head of the Blavatnik School of Computer Science during 2000-2002. He was the director of the Israeli Center of Research Excellence in Algorithms. Prof. Mansour joined MicroSoft Research in Israel in 2014. Before that he held visiting positions with MicroSoft, Bell Labs, AT&amp;T research Labs, IBM Research, and Google Research. He has mentored start-ups as Riverhead, which was acquired by Cisco, Ghoonet and Verix. Prof. Mansour has published numerous journal and proceeding papers in various areas of computer science with special emphasis on communication networks, machine learning, and algorithmic game theory, and has supervised over a dozen graduate students in those areas. Prof. Mansour is currently an associate editor in a multiple distinguished journals and has been on numerous conference program committees. He was the program chair of COLT (1998) and serves on the COLT steering committee.