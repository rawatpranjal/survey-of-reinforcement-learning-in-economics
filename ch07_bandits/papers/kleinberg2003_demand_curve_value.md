## The Value of Knowing a Demand Curve: Bounds on Regret for On-line Posted-Price Auctions

Robert Kleinberg

∗ Tom Leighton † April 17, 2007

## Abstract

We consider the revenue-maximization problem for a seller with an unlimited supply of identical goods, interacting sequentially with a population of n buyers through an on-line posted-price auction mechanism, a paradigm which is frequently available to vendors selling goods over the Internet. For each buyer, the seller names a price between 0 and 1; the buyer decides whether or not to buy the item at the specified price, based on her privately-held valuation. The price offered is allowed to vary as the auction proceeds, as the seller gains information from interactions with the earlier buyers.

The additive regret of a pricing strategy is defined to be the difference between the strategy's expected revenue and the revenue derived from the optimal fixed-price strategy. In the case where buyers' valuations are independent samples from a fixed probability distribution (usually specified by a demand curve), one can interpret the regret as specifying how much the seller should be willing to pay for knowledge of the demand curve from which buyers' valuations are sampled.

The answer to the problem depends on what assumptions one makes about the buyers' valuations. We consider three such assumptions: that the valuations are all equal to some unknown number p , that they are independent samples from an unknown probabilility distribution, or that they are chosen by an oblivious adversary. In each case, we derive upper and lower bounds on regret which match within a factor of log n ; the bounds match up to a constant factor in the case of identical valuations.

∗ Department of Mathematics, MIT, Cambridge MA 02139, and Akamai Technologies, 8 Cambridge Center, Cambridge, MA 02142. Email: rdk@math.mit.edu. Supported by a Fannie and John Hertz Foundation Fellowship.

† Department of Mathematics, MIT, Cambridge MA 02139, and Akamai Technologies, 8 Cambridge Center, Cambridge, MA 02142. Email: ftl@math.mit.edu.

## 1 Introduction

The rising popularity of Internet commerce has spurred much recent research on market mechanisms which were either unavailable or impractical in traditional markets, because of the amount of communication or computation required. We consider one such mechanism, the on-line posted-price auction, in which a seller with an unlimited supply of identical goods interacts sequentially with a population of n buyers. For each buyer, the seller names a price between 0 and 1; the buyer then decides whether or not to buy the item at the specified price, based on her privately-held valuation for the good. This transaction model is dictated by the following considerations:

- Following earlier authors [5], [7], [8], [13], we are interested in auction mechanisms which are strategyproof , meaning that buyers weakly maximize their utility by truthfully revealing their preferences. As shown in [5], this requirement in the on-line auction setting is equivalent to requiring that the seller charge buyer i a price which depends only on the valuations of previous buyers.
- Given that the price offered to buyer i does not depend on any input from that buyer, it is natural for the seller to announce this price before the buyer reveals any preference information. In fact, for reasons of trust, the buyers may not want to reveal their preferences before an offer price is quoted [6].
- For privacy reasons, the buyers generally do not wish to reveal any preference information after the price is quoted either, apart from their decision whether or not to purchase the good. Also, buyers are thus spared the effort of precisely determining their valuation, since the mechanism only requires them to determine whether it is greater or less than the quoted price.

The seller's pricing strategy will tend to converge to optimality over time, as she gains information about how the buyers' valuations are distributed. A natural question which arises is: what is the cost of not knowing the distribution of the buyers' valuations in advance? In other words, assume our seller pursues a pricing strategy S which maximizes her expected revenue ρ ( S ). As is customary in competitive analysis of auctions, we compare ρ ( S ) with the revenue ρ ( S opt ) obtained by a seller who knows the buyers' valuations in advance but is constrained to charge the same price to all buyers ([5],[6],[7],[8]). While previous authors have analyzed auctions in terms of their competitive ratio (the ratio between ρ ( S ) and ρ ( S opt )), we instead analyze the additive regret , i.e. the difference ρ ( S ) -ρ ( S opt ). This is a natural parameter to study for two reasons. First, it roughly corresponds to the amount the seller should be willing to pay to gain knowledge of the buyers' valuations, e.g. by doing market research. Second, it was shown by Blum et al in [6] that there are randomized pricing strategies achieving competitive ratio 1 + ε for any ε &gt; 0; thus it is natural to start investigating the lower-order terms, i.e. the o (1) term in the ratio ρ ( S ) /ρ ( S opt ) for the optimal pricing strategy S .

One can envision several variants of this problem, depending on what assumptions are made about the buyers' valuations. We will study three valuation models.

- Identical: All buyers' valuations are equal to a single price p ∈ [0 , 1]. This price is unknown to the seller.
- Random: Buyers' valuations are independent random samples from a fixed probability distribution on [0 , 1]. The probability distribution is not known to the seller.
- Worst-case: The model makes no assumptions about the buyers' valuations. They are chosen by an adversary who is oblivious to the algorithm's random choices.

Our results are summarized in the following three theorems. In all of them, the term 'pricing strategy' refers to a randomized on-line algorithm for choosing offer prices, unless noted otherwise.

Theorem 1.1. Assuming identical valuations, there is a deterministic pricing strategy achieving regret O (log log n ) . No pricing strategy can achieve regret o (log log n ) .

Theorem 1.2. Assuming random valuations, there is a pricing strategy achieving regret O ( √ n log n ) , under the hypothesis that the function

<!-- formula-not-decoded -->

has a unique global maximum x ∗ in the interior of [0 , 1] , and that f ′′ ( x ∗ ) &lt; 0 . No pricing strategy can achieve regret o ( √ n ) , even under the same hypothesis on the distribution of valuations.

Theorem 1.3. Assuming worst-case valuations, there is a pricing strategy achieving regret O (( n 2 / 3 (log n ) 1 / 3 ) . No pricing strategy can achieve regret o ( n 2 / 3 ) .

The lower bound in the random-valuation model is the most difficult of the results stated above, and we believe it is this paper's main contribution. No such bound was known previously, and our proof introduces novel techniques which we believe may be applicable elsewhere. Moreover, our lower bound does not rely on constructing a contrived demand curve to defeat a given pricing strategy. Rather, we will show that for any family D of demand curves satisfying some reasonably generic axioms, and for any randomized pricing strategy, the probability of achieving expected regret o ( √ n ) when the demand curve is chosen randomly from D is zero. Note the order of quantification here, which differs from √ n lower bounds which have appeared in the literature on the closelyrelated multi-armed bandit problem. In those theorems it was shown that, given foreknowledge of n , one could construct a random sequence of payoffs forcing any strategy to have expected regret Ω( √ n ). In our theorem, the demand curve is chosen randomly without foreknowledge of n or of the pricing strategy, and it is still the case that the probability of the strategy achieving regret o ( √ n ) is zero.

## 1.1 Related work

There has been much recent activity applying notions from the theory of algorithms to the analysis of auction mechanisms. While much of this work focuses on combinatorial auctions - a subject not touched on here - there has also been a considerable amount of work on auction mechanisms for selling identical individual items, the setting considered in this paper. In [7], [8], the authors consider mechanisms for off-line auctions, i.e. those in which all buyers reveal their valuations before any goods are sold. The authors characterize mechanisms which are truthful (a term synonymous with 'strategyproof', defined above), and show that no such mechanism can be constant-competitive with respect to the optimal single-price auction, assuming worst-case valuations. In contrast, they present several randomized off-line auction mechanisms which are truthful and constant-competitive with respect to the optimal auction which is constrained to set a single price and to sell at least two copies of the good.

On-line auctions were considered in [5], [6], in the posted-price setting considered here as well as the setting where buyers reveal their valuations but are charged a price which depends only on the information revealed by prior buyers. In the latter paper, techniques from machine learning theory are applied to yield a (1 + ε )-competitive on-line mechanism (for any ε &gt; 0) under the hypothesis that the optimal single-price auction achieves revenue Ω( h log h log log h ), where [1 , h ] is the interval to which the buyers' valuations belong. In Section 4.1, we use their algorithm (with a very minor technical modification) to achieve expected regret O ( n 2 / 3 (log n ) 1 / 3 ) assuming worst-case valuations.

An interesting hybrid of the off-line and on-line settings is considered by Hartline in [9]. In that paper, the mechanism interacts with the set of buyers in two rounds, with prices in the second round influenced by the preferences revealed by buyers in the first round. Assuming the set of buyers participating in the first round is a uniform random subset of the pool of n buyers, the paper exhibits a posted-price mechanism which is 4-competitive against the optimal single-price auction.

On-line multi-unit auctions (in which buyers may bid for multiple copies of the item) are considered in [4], which presents a randomized algorithm achieving competitive ratio O (log B ) where B is the ratio between the highest and lowest per-unit prices offered. This result is sharpened in [10], where the optimal competitive ratio (as a function of B ) is determined exactly.

The preceding papers have all adopted the worst-case model for buyers' valuations, as is customary in the computer science literature. The traditional approach in the economics literature (e.g. [12]) is to assume that buyers' valuations are i.i.d. samples from a known probability distribution. Our random-valuations model occupies a middle ground between these two sets of assumptions, in that the i.i.d. hypothesis is preserved but the probability distribution (i.e. the demand curve) is unknown to the seller. The same set of hypotheses is made by Ilya Segal in [13], a paper which considers strategyproof off-line multi-unit auction mechanisms. (A multi-unit auction is one in which buyers may purchase more than one copy of the good.) Segal compares the expected regret of the optimal strategyproof off-line mechanism with that of the optimal on-line posted-price mechanism (which he calls the 'optimal experimentation mechanism') under three assumptions on the space D of possible demand curves:

- D is a finite set ('Hypothesis testing');
- D is parametrized by a finite-dimensional Euclidean space ('Parametric estimation');
- D is arbitrary ('Non-parametric estimation').

In this terminology, our paper is concerned with bounding the expected regret of the optimal experimentation mechanism in the non-parametric case. Segal explicitly refrains from addressing this case, writing, 'The optimal experimentation mechanism would be very difficult to characterize in [the non-parametric] setting. Intuitively, it appears that its convergence rate may be slower [than that of the optimal off-line mechanism] because the early purchases at prices that are far from p ∗ will prove useless for fine-tuning the price around p ∗ .' This intuition is confirmed by the lower bound we prove in Section 3.2.

Our work is also closely tied to the literature on the so-called 'multi-armed bandit problem,' in which a gambler in a casino with K slot machines must decide which machine to play in a sequence of n trials, basing his decisions on the payoffs observed in prior trials. As in our auction problem, the regret is defined as the difference between the gambler's expected net payoff and the net payoff obtained from the best single action (slot machine) over the sequence of n trials.

In their pioneering work on the multi-armed bandit problem, Lai and Robbins [11] assumed that for each action, the payoffs on each of the n trials are i.i.d. random variables, but the distribution of the payoff variable varies from one action to another. Under this hypothesis, they exhibited an algorithm achieving expected regret O (log n ) as n →∞ , and proved that this is the optimal regret, up to constant factors. Auer et al [2] sharpened this analysis to obtain explicit regret bounds which hold for finite n .

If we view each potential offer price in [0 , 1] as a slot machine with random payoff, then our on-line posted-price auction problem (in the random-valuations model) becomes a special case of the 'continuum-armed bandit problem', i.e. the variant of the multi-armed bandit problem in which there is an uncountable set of slot machines indexed by a real parameter t , with the expected reward depending continuously on t . This problem is considered by Agrawal in [1], who describes an algorithm achieving regret O ( n 3 / 4+ ε ) in the case that the expected reward is a differentiable function of t . (The paper also gives a regret bound under a weaker continuity hypothesis on the expected reward, but it is more difficult to state.) Our upper bound of O ( √ n log n ) in the random-valuations model is better than the one obtained by applying Agrawal's algorithm, as might be expected because our auction problem is a highly-specific special case of the continuum-armed bandit problem. However, our lower bound of O ( √ n ) for the random-valuations model directly implies the same lower bound for Agrawal's continuum-armed bandit problem. This answers a question left open at the end of [1], when Agrawal writes, 'We do not have any tighter bounds on the learning loss other than those available for the finite case,' referring to the Ω(log n ) lower bound proved by Lai and Robbins. Our paper is thus the first demonstration of an exponential separation between the expected regret when the set of possible actions is finite and when it is infinite.

In [3], the multi-armed bandit problem is studied from an adversarial perspective, parallel to our worst-case valuation model. The authors present an algorithm achieving expected regret O ( √ nK log K ) and a nearly-matching lower bound of Ω( √ nK ) for this problem. Their algorithm forms the basis for the online posted-price auction algorithms in [6] and in Section 4.1 of this paper, and our lower-bound proof in the worst-case model (Theorem 4.3) is an adaptation of their lower-bound proof.

We would like to elucidate the difference between the lower bound for regret appearing in [3] and the lower bound presented here in the random-valuations model (Theorem 3.9). In both constructions, the rewards are i.i.d. random variables whose distribution is also random, and the theorem establishes that for any strategy the expected regret is Ω( √ n ), where the expectation is over the random choices of both the distribution and of the rewards themselves. However, in our random-valuations lower bound we define regret as the difference in expected payoff between the on-line pricing strategy and the optimal strategy which has foreknowledge of the demand curve but not of the individual buyers' valuations . This definition of regret accords with the definition adopted in papers studying the multi-armed bandit problem with i.i.d. random rewards (e.g. [1], [11]) and differs from the definition of regret adopted in [3], namely the difference in expected payoff between the on-line strategy and the ex post optimal single action. Because we choose to measure regret relative to the ex ante rather than the ex post optimal strategy, a subtler analysis is required. In [3], the lower bound comes from constructing a counterexample in which the reward distribution is so close to uniform that it is information-theoretically impossible, in n trials, to learn which of the possible reward distributions is generating the payoffs. More precisely, there is no sequence of n experiments such that an observer, after seeing the outcomes of the n experiments and being asked to guess which reward distribution is generating the payoffs, could outperform a random guesser by more than a constant factor. In our random-valuations model, it may be possible in some cases to gain rather precise information on which reward distribution is generating the payoffs in the course of n trials, but the process of gaining such knowledge will generally require offering prices which are too far from the optimal price, thereby incurring a cost of Ω( √ n ) in the process of learning this information. (This is a nice illustration of the trade-off between exploration and exploitation.) The difference between the two lower bounds is most clearly illustrated by considering the case in which the algorithm is allowed to engage in an initial n trials using 'play money,' and is then judged in terms of the expected regret incurred in the course of the subsequent n trials. The example which furnishes the lower bound in [3] has the property that no strategy could take advantage of the n free trials to achieve o ( √ n ) expected regret on the following n trials. In contrast, the example furnishing our √ n lower bound in Section 3.2 has the property that there exist pricing strategies which can take advantage of the n free trials and achieve O (1) expected regret on the following n trials, yet it is impossible to achieve o ( √ n ) regret on the first n trials.

## 2 Identical valuations

## 2.1 Upper bound

When all buyers have the same valuation p ∈ [0 , 1], the situation is completely different from the scenarios considered above, because there is no randomness in the buyer's response. Every response gives the seller perfect information about a lower or upper bound on p , depending on whether the buyer's response was to accept or to reject the price offered.

A pricing strategy S which achieves regret O (log log n ) may be described as follows. The strategy keeps track of a feasible interval [ a, b ], initialized to [0 , 1], and a precision parameter ε , initialized to 1 / 2. In a given phase of the algorithm, the seller offers the prices a, a + ε, a + 2 ε, . . . until one of them is rejected. If a + kε was the last offer accepted in this phase, then [ a + kε, a +( k +1) ε ] becomes the new feasible interval, and the new precision parameter is ε 2 . This process continues until the length of the feasible interval is less than 1 /n ; then the seller offers a price of a to all remaining buyers.

Theorem 2.1. Strategy S achieves regret O (log log n ) .

Proof. The number of phases is equal to the number of iterations of repeated squaring necessary to get from 1 / 2 to 1 /n , i.e. O (log log n ). Let p denote the valuation shared by all buyers. The seller accrues regret for two reasons:

- Items are sold at a price q &lt; p , accruing regret p -q .
- Buyers decline items, accruing regret p .

At most one item is declined per phase, incurring p &lt; 1 units of regret, so the declined offers contribute O (log log n ) to the total regret.

In each phase except the first and the last, the length b -a of the feasible interval is √ ε (i.e. it is the value of ε from the previous phase), and the set of offer prices carves up the feasible interval into subintervals of length ε . There are 1 / √ ε such subintervals, so there are at most 1 / √ ε offers made during this phase. Each time one of them is accepted, this contributes at most b -a = √ ε to the total regret. Thus, the total regret contribution from accepted offers in this phase is less than or equal to (1 / √ ε ) · √ ε = 1. There are O (log log n )

phases, so the total regret contribution from accepted offers in these phases is also O (log log n ).

In the final phase, the length of the feasible interval is less than 1 /n , and each offer is accepted. There are at most n such offers, so they contribute at most 1 to the total regret.

## 2.2 Lower bound

Theorem 2.2. If S is any randomized pricing strategy, and p is randomly sampled from the uniform distribution on [0 , 1] , the expected regret of S when the buyers' valuations are p is Ω(log log n ) .

Proof. It suffices to prove the lower bound for a deterministic pricing strategy S , since any randomized pricing strategy is a probability distribution over deterministic ones. At any stage of the game, let a denote the highest price that has yet been accepted, and b the lowest price that has yet been declined; thus p ∈ [ a, b ]. As before, we will refer to this interval as the feasible interval . It is counterproductive to offer a price less than a or greater than b , so we may assume that the pricing strategy works as follows: it offers an ascending sequence of prices until one of them is declined; it then limits its search to the new feasible interval, offering an ascending sequence of prices in this interval until one of them is declined, and so forth.

Divide the pool of buyers into phases (starting with phase 0) as follows: phase k begins immediately after the end of phase k -1, and ends after an addtional 2 2 k -1 buyers, or after the first rejected offer following phase k -1, whichever comes earlier. The number of phases is Ω(log log n ), so it suffices to prove that the expected regret in each phase is Ω(1).

Claim 2.3. Let I k denote the set of possible feasible intervals at the start of phase k . The cardinality of I k is at most 2 2 k .

Proof. The proof is by induction on k . The base case k = 0 is trivial. Now assume the claim is true for a particular value of k , and let Let I k = [ a k , b k ] be the feasible interval at the start of phase k . Let x 1 ≤ x 2 ≤ · · · ≤ x j denote the ascending sequence of prices that S will offer during phase k if all offers are accepted. (Here j = 2 2 k -1.) Then the feasible interval at the start of phase k + 1 will be one of the subintervals [ a k , x 1 ] , [ x 1 , x 2 ] , [ x 2 , x 3 ] , . . . , [ x j -1 , x j ] , [ x j , b k ]. There are at most j = 2 2 k such subintervals, and at most 2 2 k possible choices for I k (by the induction hypothesis), hence there are at most 2 2 k +1 elements of I k +1 .

Claim 2.4. Let | I | denote the length of an interval I . With probability at least 3/4, | I k | ≥ 1 4 · 2 -2 k .

Proof. The expectation of 1 / | I k | may be computed as follows:

<!-- formula-not-decoded -->

where the last inequality follows from Claim 2.3. Now use Markov's Inequality:

<!-- formula-not-decoded -->

Claim 2.5. The expected regret in phase k is at least 1 64 .

Proof. Let E k denote the event that p ≥ 1 / 4 and | I k | ≥ 1 4 · 2 -2 k . This is the intersection of two events, each having probability ≥ 3 / 4, so Pr( E k ) ≥ 1 / 2. It suffices to show that the expected regret in phase k , conditional on E k , is at least 1/32. So from now on, assume that p ≥ 1 / 4 and | I k | ≥ 1 4 · 2 -2 k .

Let m denote the midpoint of I k . As before, let j = 2 2 k -1 and let x 1 ≤ x 2 ≤ · · · ≤ x j denote the ascending sequence of prices which S would offer in phase k if no offers were rejected. We distinguish two cases:

Case 1: x j ≥ m . With probability at least 1/2, p &lt; m and the phase ends in a rejected offer, incurring a regret of p , which is at least 1/4. Thus the expected regret in this case is at least 1/8.

Case 2: x j &lt; m . The event { p &gt; m } occurs with probability 1/2, and conditional on this event the expectation of p -m is | I k | / 4 ≥ 2 -2 k / 16. Thus with probability at least 1/2, there will be 2 2 k -1 accepted offers, each contributing 2 -2 k / 16 to the expected regret, for a total of (2 2 k -1)(2 -2 k ) / 16 ≥ 1 / 32.

Thus there are Ω(log log n ) phases, each contributing Ω(1) to the expected regret of S , which establishes the theorem.

## 3 Random valuations

## 3.1 Preliminaries

In this section we will consider the case each buyer's valuation v is an independent random sample from a fixed but unknown probability distribution on [0 , 1]. It is customary to describe this probability distribution in terms of its demand curve

<!-- formula-not-decoded -->

Given foreknowledge of the demand curve, but not of the individual buyers' valuations, it is easy to see what the optimal pricing strategy would be. The expected revenue obtained from setting price x is xD ( x ). Since buyers' valuations are independent and the demand curve is known, the individual buyers'

responses provide no useful information about future buyers' valuations. The best strategy is thus to compute

<!-- formula-not-decoded -->

and to offer this price to every buyer. We denote this strategy by S ∗ , and its expected revenue by ρ ( S ∗ ). Clearly, for any on-line pricing strategy S , we have

<!-- formula-not-decoded -->

and it may be argued that in the context of random valuations it makes the most sense to compare ρ ( S ) with ρ ( S ∗ ) rather than ρ ( S opt ). We address this issue by proving a lower bound on ρ ( S ∗ ) -ρ ( S ) and an upper bound on ρ ( S opt ) -ρ ( S ).

A deterministic pricing strategy can be specified by a sequence of rooted planar binary trees T 1 , T 2 , . . . , where the n -th tree specifies the decision tree to be applied by the seller when interacting with a population of n buyers. (Thus T n is a complete binary tree of depth n .) We will use a to denote a generic internal node of such a decision tree, and glyph[lscript] to denote a generic leaf. The relation a ≺ b will denote that b is a descendant of a ; here b may be a leaf or another internal node. If e is an edge of T , we will also use a ≺ e (resp. e ≺ a ) to denote that e is below (resp. above) a in T , i.e. at least one endpoint of e is a descendant (resp. ancestor) of a . The left subtree rooted at a will be denoted by T l ( a ), the right subtree by T r ( a ). Note that T l ( a ) (resp. T r ( a )) includes the edge leading from a to its left (resp. right) child.

The internal nodes of the tree are labeled with numbers x a ∈ [0 , 1] denoting the price offered by the seller at node a , and random variables v a ∈ [0 , 1] denoting the valuation of the buyer with whom the seller interacts at that node. The buyer's choice is represented by a random variable

<!-- formula-not-decoded -->

.

In other words, χ a is 1 if the buyer accepts the price offered, 0 otherwise.

The tree T n specifies a pricing strategy as follows. The seller starts at the root r of the tree and offers the first buyer price x r . The seller moves from this node to its left or right child depending on whether the buyer declines or accepts the offer, and repeats this process until reaching a leaf which represents the outcome of the auction.

A strategy as defined above is called a non-uniform deterministic pricing strategy. Auniform deterministic pricing strategy is one in which there is a single infinite tree T whose first n levels comprise T n for each n . (This corresponds to a pricing strategy which is not informed of the value of n at the outset of the auction.) A randomized pricing strategy is a probability distribution over deterministic pricing strategies.

As mentioned above, the outcome of the auction may be represented by a leaf glyph[lscript] ∈ T n , i.e. the unique leaf such that for all ancestors a ≺ glyph[lscript] , glyph[lscript] ∈ T r ( a ) ⇔ χ a = 1. A probability distribution on the buyers' valuations v a induces a probability distribution on outcomes glyph[lscript] . We will use p D ( glyph[lscript] ) to denote the probability assigned to glyph[lscript] under the valuation distribution represented by demand curve D . For an internal node a , p D ( a ) denotes the probability that the outcome leaf is a descendant of a . We define p D ( e ) similarly for edges e ∈ T .

## 3.2 Lower bound

## 3.2.1 A family of random demand curves

The demand curves D appearing in our lower bound will be random samples from a space D of possible demand curves. In this section we single out a particular random demand-curve model, and we enumerate the properties which will be relevant in establishing the lower bound. The choice of a particular random demand-curve model is done here for ease of exposition, and not because of a lack of generality in the lower bound itself. In Section 3.2.6 we will indicate that Theorem 3.9 applies to much broader classes D of demand curves. In particular we believe that it encompasses random demand-curve models which are realistic enough to be of interest in actual economics and e-commerce applications.

For now, however, D denotes the one-parameter family of demand curves { D t : 0 . 3 ≤ t ≤ 0 . 4 } defined as follows. Let

<!-- formula-not-decoded -->

In other words, the graph of ˜ D t consists of three line segments: the middle segment is tangent to the curve xy = 1 / 7 at the point ( t, 1 / 7 t ), while the left and right segments belong to lines which lie below that curve and are independent of t . Now we obtain D t by smoothing ˜ D t . Specifically, let b ( x ) be a nonnegative, even C ∞ function supported on the interval [ -0 . 01 , 0 . 01] and satisfying ∫ 0 . 01 -0 . 01 b ( x ) dx = 1. Define D t by convolving ˜ D t with b , i.e.

<!-- formula-not-decoded -->

We will equip D = { D t : 0 . 3 ≤ t ≤ 0 . 4 } with a probability measure by specifying that t is uniformly distributed in [0 . 3 , 0 . 4].

Let x ∗ t = arg max x ∈ [0 , 1] xD t ( x ). It is an exercise to compute that x ∗ t = t . (With ˜ D t in place of D t this would be trivial. Now D t ( x ) = ˜ D t ( x ) unless x is within 0.01 of one of the two points where ˜ D ′ t is discontinuous, and these two points are far from maximizing x ˜ D t ( x ), so xD t ( x ) is also maximized at x = t .)

The specifics of the construction of D are not important, except insofar as they enable us to prove the properties specified in the following lemma.

Lemma 3.1. There exist constants α, β &gt; 0 and γ &lt; ∞ such that for all D = D t 0 ∈ D and x ∈ [0 , 1] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here x ∗ denotes x ∗ t 0 , D ( k ) ( x ) denotes the k -th t -derivative of D t ( x ) at t = t 0 , and ˙ D ( x ) denotes D (1) ( x ) .

The proof of the lemma is elementary but tedious, so it is deferred to Appendix A.

## 3.2.2 High-level description of the proof

The proof of the lower bound on regret is based on the following intuition. If there is uncertainty about the demand curve, then no single price can achieve a low expected regret for all demand curves. The family of demand curves exhibited above is parametrized by a single parameter t , and we will see that if the uncertainty about t is on the order of ε then the regret per buyer is Ω( ε 2 ). (This statement will be made precise in Lemma 3.7 below.) So to avoid accumulating Ω( √ n ) regret on the last Ω( n ) buyers, the pricing strategy must ensure that it reduces the uncertainty to O ( n -1 / 4 ) during its interactions with the initial O ( n ) buyers. However - and this is the crux of the proof - we will show that offering prices far from x ∗ is much more informative than offering prices near x ∗ , so there is a quantifiable cost to reducing the uncertainty in t . In particular, reducing the uncertainty to O ( n -1 / 4 ) costs Ω( √ n ) in terms of expected regret.

To make these ideas precise, we will introduce a notion of 'knowledge' which quantifies the seller's ability to distinguish the actual demand curve from nearby ones based on the information obtained from past transactions, and a notion of 'conditional regret' whose expectation is a lower bound on the pricing strategy's expected regret. We will show that the ratio of conditional regret to knowledge is bounded below, so that the strategy cannot accumulate Ω( √ n ) knowledge without accumulating Ω( √ n ) regret. Finally, we will show that when the expected knowledge is less than a small constant multiple of √ n , there is so much uncertainty about the true demand curve that the expected regret is Ω( √ n ) with high probability (taken over the probability measure on demand curves).

## 3.2.3 Definition of knowledge

In the following definitions, log denotes the natural logarithm function. T denotes a finite planar binary tree, labeled with a pricing strategy as explained in Section 3.1. When f is a function defined on leaves of T , we will use the notation E D f to denote the expectation of f with respect to the probability distribution p D on leaves, i.e.

<!-- formula-not-decoded -->

For a given demand curve D = D t 0 , we define the infinitesimal relative entropy of a leaf glyph[lscript] ∈ T by

<!-- formula-not-decoded -->

and we define the knowledge of glyph[lscript] as the square of the infinitesimal relative entropy:

<!-- formula-not-decoded -->

Those familiar with information theory may recognize IRE D ( glyph[lscript] ) as the t -derivative of glyph[lscript] 's contribution to the weighted sum defining the relative entropy RE ( D ‖ D t ), and K D ( glyph[lscript] ) as a random variable whose expected value is a generalization of the notion of Fisher information .

An important feature of IRE D ( glyph[lscript] ) is that it may be expressed as a sum of terms coming from the edges of T leading from the root to glyph[lscript] . For an edge e = ( a, b ) ∈ T , let

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

## 3.2.4 Definition of conditional regret

For a given D , the conditional regret R D ( glyph[lscript] ) may be informally defined as follows. At the end of the auction, if the demand curve D were revealed to the seller and then she were required to repeat the same sequence of offered prices { x a : a ≺ glyph[lscript] } to a new, independent random population of buyers whose valuations are distributed according to D , then R D ( glyph[lscript] ) is the expected regret incurred by the seller during this second round of selling. Formally, R D ( glyph[lscript] ) is defined as follows. Let

<!-- formula-not-decoded -->

where x ∗ = arg max x ∈ [0 , 1] { xD ( x ) } as always. Note that if two different sellers offer prices x ∗ , x , respectively, to a buyer whose valuation is distributed according to D , then r D ( x ) is the difference in their expected revenues. Now let

<!-- formula-not-decoded -->

Although R D ( glyph[lscript] ) is not equal to the seller's actual regret conditional on outcome glyph[lscript] , it is a useful invariant because E D R D ( glyph[lscript] ) is equal to the actual expected regret of S relative to S ∗ . (It is also therefore a lower bound on the expected regret of S relative to S opt .) This fact is far from obvious, because the distribution of the actual buyers' valuations, conditioned on their responses to the prices they were offered, is very different from the distribution of n new independent buyers. In general the expected revenue of S or S ∗ on the hypothetical independent population of n buyers will not equal the expected revenue obtained from the actual population of n buyers, conditioned on those buyers' responses. Yet the expected difference between the two random variables, i.e. the regret, is the same for both populations of buyers. This fact is proved in the following lemma.

Lemma 3.2. Let S be a strategy with decision tree T , and let S ∗ be the fixedprice strategy which offers x ∗ to each buyer. If the buyers' valuations are independent random samples from the distribution specified by D , then the expected revenue of S ∗ exceeds that of S by exactly E D R D ( glyph[lscript] ) .

Proof. Let

<!-- formula-not-decoded -->

At a given point of the sample space, let glyph[lscript] denote the outcome leaf, and let a 1 , a 2 , . . . , a n be the ancestors of glyph[lscript] . Then the revenue of S ∗ is ∑ n i =1 χ ∗ a i x ∗ , and the revenue of S is ∑ n i =1 χ a i x a i . It follows that the expected difference between the two is

<!-- formula-not-decoded -->

## 3.2.5 Proof of the lower bound

In stating the upcoming lemmas, we will introduce constants c 1 , c 2 , . . . . When we introduce such a constant we are implicitly asserting that there exists a constant 0 &lt; c i &lt; ∞ depending only on the demand curve family D , and satisfying the property specified in the statement of the corresponding lemma.

We begin with a series of lemmas which establish that E D K D is bounded above by a constant multiple of E D R D . Assume for now that D is fixed, so x ∗ is also fixed, and put

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Recall from Lemma 3.1 that

<!-- formula-not-decoded -->

hence

Now we see that

<!-- formula-not-decoded -->

so the lemma holds with c 1 = β .

<!-- formula-not-decoded -->

Proof. As in the preceding lemma, the idea is to rewrite the sum over leaves as a sum over internal nodes and then bound the sum term-by-term. (In this case, actually it is a sum over internal edges of T .) A complication arises from the fact that the natural expression for E D K D ( glyph[lscript] ) involves summing over pairs of ancestors of a leaf; however, we will see that all of the cross-terms cancel, leaving us with a manageable expression.

<!-- formula-not-decoded -->

For any e ∈ T , the sum ∑ e ′ glyph[follows] e p D ( e ′ ) ire D ( e ′ ) vanishes because the terms may be grouped into pairs p D ( e ′ ) ire D ( e ′ )+ p D ( e ′′ ) ire D ( e ′′ ) where e ′ , e ′′ are the edges

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

joining a node a ∈ T to its right and left children, respectively, and we have

<!-- formula-not-decoded -->

Thus

<!-- formula-not-decoded -->

so the lemma holds with c 2 = 2 γ 2 .

Corollary 3.5. E D K D ( glyph[lscript] ) ≤ c 3 E D R D ( glyph[lscript] ) .

The relevance of Corollary 3.5 is that it means that when E D R D is small, then p D t ( glyph[lscript] ) cannot shrink very rapidly as a function of t , for most leaves glyph[lscript] . This is made precise by the following Lemma. Here and throughout the rest of this section, D refers to a demand curve D t 0 ∈ D .

Lemma 3.6. For all sufficiently large n , if E D R D &lt; √ n then there exists a set S of leaves such that p D ( S ) ≥ 1 / 2 , and p D t ( glyph[lscript] ) &gt; c 4 p D ( glyph[lscript] ) for all glyph[lscript] ∈ S and all t ∈ [ t 0 , t 0 + n -1 / 4 ] .

The proof is quite elaborate, so we have deferred it to Appendix B.

We will also need a lemma establishing the growth rate of R D t ( glyph[lscript] ) for a fixed leaf glyph[lscript] , as t varies.

Lemma 3.7. R D ( glyph[lscript] ) + R D t ( glyph[lscript] ) &gt; c 5 ( t -t 0 ) 2 n for all leaves glyph[lscript] ∈ T n and for all D t ∈ D .

Proof. We know that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so it suffices to prove that r D ( x ) + r D t ( x ) &gt; c 4 ( t -t 0 ) 2 for all x ∈ [0 , 1]. Assume without loss of generality that t -t 0 &gt; 0. (Otherwise, we may reverse the roles of D and D t .) Let x ∗ and x ∗ t denote the optimal prices for D,D t , respectively. (Recall that D = D t 0 .) Note that x ∗ t -x ∗ &gt; α ( t -t 0 ), by property 1 of Lemma 3.1.

Let h = x -x ∗ , h t = x -x ∗ t , and note that | h | + | h t | &gt; α ( t -t 0 ). Now

<!-- formula-not-decoded -->

so the lemma holds with c 5 = 1 2 c 1 α 2 .

We now exploit Lemmas 3.6 and 3.7 to prove that if E D R D is less than some small constant multiple of √ n when D = D t 0 , then E D t R D t = Ω( √ n ) on a large fraction of the interval [ t 0 , t 0 + n -1 / 4 ]. The idea behind the proof is that Lemma 3.6 tells us there is a large set S of leaves whose measure does not vary by more than a constant factor as we move t across this interval, while Lemma 3.7 tells us that the regret contribution from leaves in S is Ω( √ n ) for a large fraction of the t -values in this interval. In the following proposition, c ( M ) denotes the function min { 1 , 1 2 c 4 c 5 (1 + c 4 ) -1 M -2 } .

Proof. If E D R D &lt; c ( M ) √ n , we may apply Lemma 3.6 to produce a set S of leaves such that p D ( S ) ≥ 1 / 2 and p D t ( glyph[lscript] ) &gt; c 4 p D ( glyph[lscript] ) for all glyph[lscript] ∈ S and all t ∈ [ t 0 , t 0 + n -1 / 4 ]. Now,

Proposition 3.8. For all M and all sufficiently large n , if E D R D &lt; c ( M ) √ n , then E D t R D t &gt; c ( M ) √ n for all t ∈ [ t 0 +(1 /M ) n -1 / 4 , t 0 + n -1 / 4 ] .

and, for all t ∈

<!-- formula-not-decoded -->

where the fourth line is derived from the third by applying the inequality R D t ( glyph[lscript] ) &gt; c 4 ( t -t 0 ) 2 n -R D ( glyph[lscript] ) coming from Lemma 3.7.

Theorem 3.9. Let S be any randomized non-uniform strategy, and let R D ( S , n ) denote the expected ex ante regret of S on a population of n buyers whose valuations are independent random samples from the probability distribution specified by the demand curve D . Then

<!-- formula-not-decoded -->

In other words, if D is drawn at random from D , then almost surely R D ( S , n ) is not o ( √ n ) .

Proof. It suffices to prove the theorem for a deterministic strategy S , since any randomized strategy is a probability distribution over such strategies. Now assume, to the contrary, that

<!-- formula-not-decoded -->

and choose M large enough that the left side of (2) is greater than 1 /M . Recall from Lemma 3.3 that E D R D = R D ( S , n ). We know that for every D = D t 0 ∈ D such that E D R D &lt; c ( M ) √ n ,

<!-- formula-not-decoded -->

Now choose N large enough that the set

<!-- formula-not-decoded -->

has measure greater than 1 /M . Replacing X N if necessary with a proper subset still having measure greater than 1 /M , we may assume that { t : D t ∈ X N } is disjoint from [0 . 4 -ε, 0 . 4] for some ε &gt; 0. Choosing n large enough that n &gt; N and n -1 / 4 &lt; ε , equation (3) ensures that the sets

<!-- formula-not-decoded -->

are disjoint for k = 0 , 1 , . . . , M -1 . But each of the sets X k N , being a translate of X N , has measure greater than 1 /M . Thus their total measure is greater than 1, contradicting the fact that D has measure 1.

## 3.2.6 General demand-curve models

The methods of the preceding section extend to much more general families of demand curves. Here we will merely sketch the ideas underlying the extension. Suppose that D is a compact subset of the space C 4 ([0 , 1]) of functions on [0 , 1] with continuous fourth derivative, and that the demand curves D ∈ D satisfy the following two additional hypotheses:

- (Unique global max) The function f ( x ) = xD ( x ) has a unique global maximum x ∗ ∈ [0 , 1], and it lies in the interior of the interval.
- (Non-degeneracy) The second derivative of f is strictly negative at x ∗ .

Suppose D is also endowed with a probability measure, denoted µ . The proof of the lower bound relied heavily on the notion of being able to make a 'oneparameter family of perturbations' to a demand curve. This notion may be encapsulated using a flow φ ( D,t ) mapping an open set U ⊆ D× R into D , such that ( D ×{ 0 } ) ∩ U has measure 1, and φ ( D, 0) = D when defined. We will use the shorthand D t for φ ( D,t ). The flow must satisfy the following properties:

- (Additivity) φ ( D,s + t ) = φ ( φ ( D,s ) , t ).
- (Measure-preservation) If X ⊆ D and φ ( D,t ) is defined for all D ∈ X , then µ ( φ ( X,t )) = µ ( X ).
- (Smoothness) The function g ( t, x ) = D t ( x ) is a C 4 function of t and x .
- (Profit-preservation) If x ∗ t denotes the point at which the function xD t ( x ) achieves its global maximum, then x ∗ t D t ( x ∗ t ) = x ∗ 0 D 0 ( x ∗ 0 ) for all t such that D t is defined.

glyph[negationslash]

- (Non-degeneracy) d dt ( x ∗ t ) = 0.
- (Rate dampening at 0 and 1) For k = 1 , 2 , 3 , 4 , the functions ∣ ∣ ∣ D ( k ) D ∣ ∣ ∣ and ∣ ∣ ∣ D ( k ) 1 -D ∣ ∣ ∣ are uniformly bounded above, where D ( k ) denotes the k -th derivative of D with respect to t .

Provided that these axioms are satisfied, it is possible to establish all of the properties specified in Lemma 3.1. Property 1 follows from compactness of D and non-degeneracy of φ , property 2 follows from the compactness of D together with the non-degeneracy and 'unique global max' axioms for D , and property 4 is the rate-dampening axiom. Property 3 is the subtlest: it follows from the smoothness, profit-preservation, and rate-dampening properties of φ . The key observation is that profit-preservation implies that

<!-- formula-not-decoded -->

so that x ∗ D t ( x ∗ ), as a function of t , is maximized at t = 0. This, coupled with smoothness of φ , proves that ˙ D ( x ∗ ) = 0. Another application of smoothness yields the desired bounds.

The final steps of Theorem 1 used the translation-invariance of Lebesgue measure on the interval [0 . 3 , 0 . 4] to produce M sets whose disjointness yielded the desired contradiction. This argument generalizes, with the flow φ playing the role of the group of translations. It is for this reason that we require φ to satisfy the additivity and measure-preservation axioms.

## 3.3 Upper bound

The upper bound on regret in the random-valuation model is based on applying techniques from the literature on the multi-armed bandit problem, specifically [2]. To do so, we discretize the set of possible actions by limiting the seller to strategies which only offer prices belonging to the set { 1 /K, 2 /K,... , 1 -1 /K, 1 } , for suitably-chosen K . (It will turn out that K = θ (( n/ log n ) 1 / 4 ) is the best choice.)

We are now in a setting where the seller must choose one of K possible actions on each of n trials, where each action yields a reward which is a random variable taking values in [0 , 1], whose distribution depends on the action chosen, but the rewards for a given action are i.i.d. across the n trials. This is the scenario studied in [2]. They define µ i to be the expected reward of action i , µ ∗ = max { µ 1 , . . . , µ K } , and

<!-- formula-not-decoded -->

Having made these definitions, the following theorem is proven.

Theorem 3.10 ([2], Theorem 1.) . There exists a strategy ucb1 such that, for all K &gt; 1 , if ucb1 is run on a set of K actions having arbitrary reward distributions P 1 , . . . , P K with support in [0 , 1] , then its expected regret after any number n of plays is at most

<!-- formula-not-decoded -->

To be precise, the model in [2] assumes that the reward variables for different actions are independent, an assumption which does not hold in our scenario. However, this assumption is not used in their proof of Theorem 3.10, so we may still apply the theorem.

To apply this theorem, we need to know something about the values of ∆ 1 , . . . , ∆ K in the special case of interest to us. When the buyer's valuation is v , the payoff of action i/K is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall that we are making the following hypothesis on the demand curve D : the function f ( x ) = xD ( x ) has a unique global maximum at x ∗ ∈ (0 , 1), and f ′′ ( x ∗ ) is defined and strictly negative. This hypothesis is useful because it enables us to establish the following lemma, which translates directly into bounds on ∆ i .

Lemma 3.11. There exist constants C 1 , C 2 such that C 1 ( x ∗ -x ) 2 &lt; f ( x ∗ ) -f ( x ) &lt; C 2 ( x ∗ -x ) 2 for all x ∈ [0 , 1] .

Proof. The existence and strict negativity of f ′′ ( x ∗ ) guarantee that there are constants A 1 , A 2 , ε &gt; 0 such that A 1 ( x ∗ -x ) 2 &lt; f ( x ∗ ) -f ( x ) &lt; A 2 ( x ∗ -x ) 2 for all x ∈ ( x ∗ -ε, x ∗ + ε ). The compactness of X = { x ∈ [0 , 1] : | x ∗ -x | ≥ ε } , together with the fact that f ( x ∗ ) -f ( x ) is strictly positive for all x ∈ X , guarantees that

Hence there are constants B 1 , B 2 such that B 1 ( x ∗ -x ) 2 &lt; f ( x ∗ ) -f ( x ) &lt; B 2 ( x ∗ -x ) 2 for all x ∈ X . Now put C 1 = min { A 1 , B 1 } and C 2 = max { A 2 , B 2 } to obtain the lemma.

Corollary 3.12. ∆ i ≥ C 1 ( x ∗ -i/K ) 2 for all i . If ˜ ∆ 0 ≤ ˜ ∆ 1 ≤ . . . ≤ ˜ ∆ K -1 are the elements of the set { ∆ 1 , . . . , ∆ k } sorted in ascending order, then ˜ ∆ j ≥ C 1 ( j/ 2 K ) 2 .

Proof. The inequality ∆ i ≥ C 1 ( x ∗ -i/K ) 2 is a restatement of the lemma using the formulae for ∆ i , µ i given above in (4),(6). The lower bound on ˜ ∆ j follows upon observing that at most j elements of the set { 1 /K, 2 /K,... , 1 } lie within a distance j/ 2 K of x ∗ .

<!-- formula-not-decoded -->

Proof. At least one of the numbers { 1 /K, 2 /K,... , 1 } lies within 1 /K of x ∗ ; now apply the upper bound on f ( x ∗ ) -f ( x ) stated in the lemma.

Putting all of this together, we have derived the following upper bound.

Theorem 3.14. Assuming that the function f ( x ) = xD ( x ) has a unique global maximum x ∗ ∈ (0 , 1) , and that f ′′ ( x ∗ ) is defined and strictly negative, the strategy ucb1 with K = glyph[ceilingleft] ( n/ log n ) 1 / 4 glyph[ceilingright] achieves expected regret O ( √ n log n ) .

Proof. Consider the following four strategies:

- ucb1 , the strategy defined in [2].
- S opt , the optimal fixed-price strategy.
- S ∗ , the fixed-price strategy which offers x ∗ to every buyer.
- S ∗ K , the fixed-price strategy which offers i ∗ /K to every buyer, where i ∗ /K is the element of { 1 /K, 2 /K,... , 1 } closest to x ∗ .

As usual, we will use ρ ( · ) to denote the expected revenue obtained by a strategy. We will prove a O ( √ n log n ) upper bound on each of ρ ( S ∗ K ) -ρ ( ucb1 ), ρ ( S ∗ ) -ρ ( S ∗ K ), and ρ ( S opt ) -ρ ( S ∗ ), from which the theorem follows immediately. √

We first show, using 3.10, that ρ ( S ∗ K ) -ρ ( ucb1 ) = O ( n log n ). By Corollary 3.12,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging these estimates into the regret bound in Theorem 3.10, we see that the regret of ucb1 relative to S ∗ K is O (( n log n ) 1 / 2 ), as claimed.

Next we bound the difference ρ ( S ∗ ) -ρ ( S ∗ K ). The expected revenues of S ∗ and S ∗ K are nx ∗ D ( x ∗ ) and nµ ∗ , respectively. Applying Corollary 3.13, the regret of S ∗ K relative to S ∗ is bounded above by

<!-- formula-not-decoded -->

Finally, we must bound ρ ( S opt ) -ρ ( S ∗ ). For any x ∈ [0 , 1], let ρ ( x ) denote the revenue obtained by the fixed-price strategy which offers price x , and let x opt = arg max x ∈ [0 , 1] ρ ( x ). We begin by observing that for all x &lt; x opt ,

<!-- formula-not-decoded -->

This is simply because every buyer that accepts price x opt would also accept x , and the amount of revenue lost by setting the lower price is x opt -x per buyer. Now

<!-- formula-not-decoded -->

so a bound on Pr( ρ ( x ) -ρ ( x ∗ ) &gt; λ ) for fixed x translates into a bound on Pr( ρ ( x opt ) -ρ ( x ∗ ) &gt; λ ). But for fixed x , the probability in question is the probability that a sum of n i.i.d. random variables, each supported in [ -1 , 1] and with negative expectation, exceeds λ . The Chernoff-Hoeffding bound tells us that so

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally,

<!-- formula-not-decoded -->

Remark 3.15. If the seller does not have foreknowledge of n , it is still possible to achieve regret O ( √ n log n ) by maintaining an estimate n est of n , initialized to 1. When 2 k buyers have been seen, the seller sets n est to 2 k +1 and reinitializes ucb1 using this new value of n est .

## 4 Worst-case valuations

In the worst-case valuation model, we assume that the buyers' valuations are chosen by an adversary who has knowledge of n and of the pricing strategy, but is oblivious to the algorithm's random choices. A similar adversarial model for the multi-armed bandit problem was considered by Auer et al in [3]. In that paper the authors present an algorithm Exp3 achieving regret O ( √ nK log K ), where K is the number of possible actions, and they exhibit a lower bound of Ω( √ nK ) on regret. The algorithm, which is based on the weighted-majority learning algorithm of Littlestone and Warmuth, was applied in the setting of on-line auctions by Blum et al [6], who normalize the buyers' valuations to lie in an interval [1 , h ] and then prove the following theorem:

Theorem 4.1 ([6], Theorem 5.) . There exists a pricing strategy Exp3 and a constant c ( ε ) such that for all valuation sequences, if the optimal fixed-price revenue ρ ( S opt ) satisfies ρ ( S opt ) &gt; c ( ε ) h log h log log h , then Exp3 is (1 + ε ) -competitive relative to ρ ( S opt ) .

Our upper and lower bounds for regret in the worst-case valuation model are based on the techniques employed in these two papers. The upper bound (Theorem 4.2) is virtually a restatement of Blum et al's theorem, though the change in emphasis from competitive ratio to additive regret necessitates a minor change in technical details. Our worst-case lower bound (Theorem 4.3) is influenced by Auer et al's proof of the corresponding lower bound for the adversarial multi-armed bandit problem in [3]. While it is possible to prove our result entirely using the techniques from their paper, we will instead present a proof using the techniques developed in Section 3.2, partly in the interest of making the paper more self-contained and partly to illustrate the power of those techniques.

## 4.1 Upper bound

Following [6], as well as the technique used in Section 3.3 above, we specify a finite set of offer prices X = { 1 /K, 2 /K,... , 1 } and constrain the seller to select prices from this set only. This reduces the posted-price auction problem to an instance of the multi-armed bandit problem, to which the algorithm Exp3 of [3] may be applied. Denote this pricing strategy by S . The relevant theorem about Exp3 is the following.

Theorem 4.2 ([3], Corollary 4.2.) . If one runs the algorithm Exp3 with a set of K actions, over n steps, with the rewards for each action in each step belonging to [0 , 1] , then the expected regret of Exp3 relative to the best fixed action is at most 2 √ e -1 √ nK log K .

Thus, if S opt X denotes the fixed-price strategy which chooses the best offer price i ∗ /K from X , and S opt denotes the fixed-price strategy which chooses the best offer price x ∗ from [0 , 1], we have the following inequalities:

<!-- formula-not-decoded -->

where the second inequality follows from the fact that S opt K is no worse than the strategy which offers 1 K glyph[floorleft] Kx ∗ glyph[floorright] to each buyer. √

If we pick K = glyph[ceilingleft] n/ log n glyph[ceilingright] 1 / 3 , then both nK log K ) and n/K are O ( n 2 / 3 (log n ) 1 / 3 ). We have thus expressed the regret of Exp3 as a sum of two terms, each of which is O ( n 2 / 3 (log n ) 1 / 3 ), establishing the upper bound asserted in Theorem 1.3.

Readers familiar with [6] will recognize that the only difference between this argument and their Theorem 5 is that they choose the prices in X to form a geometric progress (so as to optimize the competitive ratio) while we choose them to form an arithmetic progression (so as to optimize the additive regret).

## 4.2 Lower bound

In [3], the authors present a lower bound of √ nK for the multi-armed bandit problem with payoffs selected by an oblivious adversary. Ironically, the power of the adversary in this lower bound comes not from adapting to the on-line algorithm A , but from adapting to the number of trials n . In fact, the authors define a model of random payoffs (depending on n but not the algorithm) such that the expected regret of any algorithm on a random sample from this distribution is Ω( √ nK ). The idea is select one of the K actions uniformly at random and designate it as the 'good' action. For all other actions, the payoff in each round is a uniform random sample from { 0 , 1 } , but for the good action the payoff is a biased sample from { 0 , 1 } , which is 1 with probability 1 / 2+ ε , where ε = θ ( √ K/n ). A strategy which knows the good action will achieve expected payoff (1 / 2 + ε ) n = 1 / 2 + θ ( √ nK ). It can be shown, for information-theoretic reasons, that no strategy can learn the good action rapidly and reliably enough to play it more than n/K + θ ( ε √ n 3 /K ) times in expectation, from which the lower bound on regret follows.

A similar counterexample can be constructed in the context of our on-line posted-price auction problem, i.e. a random distribution of buyers' valuations (depending on n but not the algorithm) such that the expected regret of any algorithm on a random sample from this distribution is Ω( n 2 / 3 ). The idea is roughly the same as above: one randomly chooses a subinterval of [0 , 1] of length 1 /K to be the interval of 'good prices', and chooses the distribution of buyers' valuations so that the expected revenue per buyer is a constant independent of the offer price outside the interval of good prices, and is ε higher than this constant inside the interval of good prices. As above, there is a trade-off between choosing ε too large (which makes it too easy for strategies to learn which prices belong to the good interval) or too small (which leads to a negligible difference in revenue between the best strategy and all others), and the optimal trade-off is achieved when ε = θ ( √ K/n ). However, in our setting there is an additional constraint that ε ≤ 1 /K , since the expected payoff can grow by no more than 1 /K on an interval of length 1 /K . This leads to the values K = θ ( n 1 / 3 ) , ε = θ ( n -1 / 3 ) and yields the stated lower bound of Ω( n 2 / 3 ).

There are two complications that come up along the way. One is that the seller's algorithm has a continuum of alternatives at every step, rather than a finite set of K alternatives as in the example from [3]. This can be dealt with by restricting the buyers' valuations to lie in a finite set V = { v 1 , v 2 , . . . , v K } . Then there is no incentive for the seller to offer a price which lies outside of V , so we may assume the seller is constrained to offer prices in V and prove lower bounds for this restricted class of strategies.

The second complication that arises is that the adversary in [3] was more powerful: he could specify the reward for each action independently, whereas our adversary can only set a valuation v , and this v determines the rewards for all actions simultaneously. While this entails choosing a more complicated reward distribution, the complication only makes the computations messier without introducing any new ideas into the proof.

Theorem 4.3. For any given n , there exists a finite family P = { p n j } K j =1 of probability distributions on [0 , 1] , such that if p n j is chosen uniformly at random from P and then buyers' valuations are sampled independently at random according to p n j , no pricing strategy can achieve expected regret o ( n 2 / 3 ) , where the expectation is over both the random choice of D and the randomly-sampled valuations.

Proof sketch. For simplicity, assume n = 8 K 3 , and put ε = 1 / 2 K . The valuations will be independent random samples from the set V = { 1 2 , 12 + ε, 12 + 2 ε, . . . , 1 -ε, 1 } . A 'baseline probability distribution' p base on V is defined so that

<!-- formula-not-decoded -->

A random sample from p n j ∈ P is generated by sampling v ∈ V at random from the distribution p base , and then adding ε to it with probability 1/10 if and only if v = 1 -jε .

For any random variable X depending on a sequence of samples from V , we'll write E base ( X ), E j ( X ) to denote the expectation of X with respect to the distributions p base , p n j , respectively. We let r t denote the Boolean random variable which is 1 if and only if the t -th buyer accepted the price offered. As in [3], let r t denote the vector ( r 1 , . . . , r t ), and let r = r T . Assume that the seller's pricing strategy S only offers prices in V -since it is counterproductive to offer a price outside V when the buyers' valuations always belong to V -and let N i denote the random variable specifying the number of times price 1 -iε is offered.

Lemma 4.4. Let f : { 0 , 1 } T → [0 , M ] be any function defined on sequences r . Then for any action i ,

<!-- formula-not-decoded -->

The proof is exactly the same as the proof of Lemma B.1 in [3], except for the estimate of the relative entropy KL ( p base { r t | r t -1 }‖ p n i { r t | r t -1 } ) . The term KL ( 1 2 ‖ 1 2 + ε ) must be replaced by

<!-- formula-not-decoded -->

for some v ∈ { 3 / 4 , 3 / 4 + ε, . . . , 1 } . A tedious computation verifies that this is bounded above by 0 . 36 ε 2 , and one finishes up as in their proof of Lemma B.1.

For a deterministic on-line pricing strategy S , the random variable N i is a function of r , so we may apply the above lemma to conclude that

<!-- formula-not-decoded -->

hence

<!-- formula-not-decoded -->

In other words, the expected number of times S chooses the 'good' offer price is at most (1 / 3) εn √ n/K . Recalling that K = 1 2 n 1 / 3 and ε = 2 n -1 / 3 , we see that S makes the right choice at most (2 √ 2 / 3) n times in expectation, so it makes the wrong choice Ω( n ) times in expectation. Each time it does so, it incurs an expected regret of ε/ 10 = Ω( n -1 / 3 ). Thus the total expected regret of S is Ω( n 2 / 3 ), as claimed.

## 5 Acknowledgements

We would like to thank Jason Hartline, Jon Kleinberg, and Leah Brooks for helpful discussions relating to this work. We would also like to thank Denny Denker and Sean Moriarity for helpful discussions that led us to investigate these issues.

## References

- [1] R. Agrawal. The continuum-armed bandit problem. SIAM J. Control and Optimization , 33:1926-1951, 1995.
- [2] P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multi-armed bandit problem. Machine Learning , 47:235-256, 2002.
- [3] P. Auer, N. Cesa-Bianchi, Y. Freund, and R. Schapire. Gambling in a rigged casino: The adversarial multi-armed bandit problem. In Proceedings of the 36th Annual IEEE Symposium on Foundations of Computer Science , 1995.
- [4] A. Bagchi, A. Chaudhary, R. Garg, M. Goodrich, and V. Kumar. Seller-focused algorithms for online auctioning. In Proc. 7th Internation Workshop on Algorithms and Data Structures (WADS 2001) , vol. 2125. Springer Verlag LNCS, 2001.
- [5] Z. Bar-Yossef, K. Hildrum, and F. Wu. Incentive-compatible online auctions for digital goods. In Proc. 13th Symp. on Discrete Alg. 964-970, 2002.
- [6] A. Blum, V. Kumar, A. Rudra, and F. Wu Online learning in online auctions. In Proc. 14th Symp. on Discrete Alg. 202-204, 2003.
- [7] A. Fiat, A. Goldberg, J. Hartline, and A. Wright. Competitive generalized auctions. In Proc. 34th ACM Symposium on the Theory of Computing. ACM Press, New York, 2002.
- [8] A. Goldberg, J. Hartline, and A. Wright. Competitive auctions and digital goods. In Proc. 12th Symp. on Discrete Alg. , 735-744, 2001.
- [9] J. Hartline. Dynamic posted price mechanisms. Manuscript, 2001.
- [10] R. Lavi and N. Nisan. Competitive analysis of incentive compatible online auctions. In Proceedings of the 2nd ACM Conference on Electronic Commerce (EC-00) . 233-241, 2000.
- [11] T. L. Lai and H. Robbins. Asymptotically efficient adaptive allocations rules. Adv. in Appl. Math. , 6:4-22, 1985.
- [12] R. Myerson. Optimal auction design. Mathematics of Operations Research , 6:58-73, 1981.
- [13] I. Segal. Optimal pricing mechanisms with unknown demand. Manuscript, 2002.

## A Proof of Lemma 3.1

In this section we restate and prove Lemma 3.1.

Lemma A.1. There exist constants α, β &gt; 0 and γ &lt; ∞ such that for all D = D t 0 ∈ D and x ∈ [0 , 1] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here x ∗ denotes x ∗ t 0 , D ( k ) ( x ) denotes the k -th t -derivative of D t ( x ) at t = t 0 , and ˙ D ( x ) denotes D (1) ( x ) .

Proof. We begin with some useful observations about the relation between ˜ D t and D t . The function ˜ D t is piecewise-linear, and linear functions are preserved under convolution with an even function whose integral is 1. Recall that the bump function b is an even function supported in [ -0 . 01 , 0 . 01] and satisfying ∫ 0 . 01 -0 . 01 b ( x ) dx = 1; hence D t ( x ) = ˜ D t ( x ) unless x is within 0 . 01 of one of the two points where the derivative of ˜ D t is discontinuous. The x -coordinates of these two points are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For t in the range [0 . 3 , 0 . 4] this means that x 0 ∈ (0 . 115 , 0 . 259) , x 1 ∈ (0 . 416 , 0 . 546) . Recalling that b is a C ∞ function, we find that t ↦→ ˜ D t is a continuous mapping from [0 . 3 , 0 . 4] to C ∞ ([0 , 1]). Hence { ˜ D t : 0 . 3 ≤ t ≤ 0 . 4 } is a compact subset of C ∞ ([0 , 1]), and consequently for 1 ≤ k &lt; ∞ , the k -th derivative of ˜ D t is bounded uniformly in t .

We now proceed to prove each of the properties stated in the Lemma.

1. First we verify that x ∗ t = t , as stated in Section 3.2. If x lies in the interval I t = [ x 0 + 0 . 01 , x 1 -0 . 01] where D t ( x ) = 2 / 7 t -x/ 7 t 2 , then xD t ( x ) = 2 x/ 7 t -x 2 / 7 t 2 = 1 7 [1 -(1 -x/t ) 2 ], which is uniquely maximized when x = t and xD t ( x ) = 1 / 7. Note that the estimates given above for x 0 and x 1 ensure that [ x 0 +0 . 01 , x 1 -0 . 01] always contains [0 . 3 , 0 . 4], so t always lies in this interval. If x lies in the interval where D t ( x ) = 1 -2 x or D t ( x ) = (1 -x ) / 2, then xD t ( x ) is equal to 2(1 / 16 -( x -1 / 4) 2 ) or (1 / 2)(1 / 4 -( x -1 / 2) 2 ), and in either case xD t ( x ) can not exceed 1 / 8. It is straightforward but tedious to verify that xD t ( x ) is bounded away from 1 / 7 when | x -x 0 | ≤ 0 . 01 or | x -x 1 | ≤ 0 . 01; this confirms that x ∗ t = t is the unique global maximum of the function xD t ( x ). Having verified this fact, it follows immediately that d/dt ( x ∗ t ) | t = t 0 = 1.
2. On the interval I t where D t ( x ) = 2 / 7 t -x/ 7 t 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We have seen that for t ∈ [0 . 3 , 0 , 4], xD t ( x ) attains its maximum value of 1 / 7 at a point x ∗ t ∈ I t and is strictly less than 1 / 7 at all other points of [0 , 1]. By compactness it follows that there exist ε, δ &gt; 0 such that

<!-- formula-not-decoded -->

for all x ∈ [0 , 1] , t ∈ [0 . 3 , 0 . 4]. Combining (8), which holds when x ∈ I t , with (9), which holds when x glyph[negationslash]∈ ( x ∗ t -δ, x ∗ t + δ ) , we obtain

<!-- formula-not-decoded -->

for all x ∈ [0 , 1] .

3. If x &lt; 0 . 1 or x &gt; 0 . 6, then D t ( x ) is independent of t , so ˙ D ( x ) = 0, which establishes the desired inequality. If x ∈ [0 . 1 , 0 . 6], then D ( x ) and 1 -D ( x ) are both bounded below by 0 . 2, so it remains to verify that sup {| ˙ D t ( x ) / ( x ∗ t -x ) |} &lt; ∞ . The function | ˙ D t ( x ) | is a continuous function of t and x , so by compactness it is bounded above by a constant. It follows that for any constant ε &gt; 0, sup {| ˙ D t ( x ) / ( x ∗ t -x ) | : ε &lt; | x ∗ t -x |} &lt; ∞ . Choose ε small enough that [ x ∗ t -ε, x ∗ t + ε ] is contained in the interval I t where D t ( x ) = 2 / 7 t -x/ 7 t 2 for all t ∈ [0 . 3 , 0 . 4]. Then for | x ∗ t -x | ≤ ε ,

<!-- formula-not-decoded -->

so sup {| ˙ D t ( x ) / ( x ∗ t -x ) |} &lt; ∞ as claimed.

4. As before, if x glyph[negationslash]∈ [0 . 1 , 0 . 6] then D ( k )( x ) = 0 so there is nothing to prove. If x ∈ [0 . 1 , 0 . 6] then D ( x ) and 1 -D ( x ) are both bounded below by 0 . 2, and | D ( k )( x ) | is uniformly bounded above, by compactness.

## B Proof of Lemma 3.6

In this section we restate and prove Lemma 3.6.

Lemma B.1. For all sufficiently large n , if E D R D &lt; √ n then there exists a set S of leaves such that p D ( S ) ≥ 1 / 2 , and p D t ( glyph[lscript] ) &gt; c 4 p D ( glyph[lscript] ) for all glyph[lscript] ∈ S and all t ∈ [ t 0 , t 0 + n -1 / 4 ] .

Proof. It suffices to prove that there exists a set S of leaves such that p D ( S ) ≥ 1 / 2 and | log( p D t ( glyph[lscript] ) /p D ( glyph[lscript] )) | is bounded above by a constant for glyph[lscript] ∈ S . Let F ( t, glyph[lscript] ) = log( p D t ( glyph[lscript] )). By Taylor's Theorem, we have

<!-- formula-not-decoded -->

for some t 1 ∈ [ t 0 , t ]. (Here F ′ , F ′′ , F ′′′ , F ′′′′ refer to the t -derivatives of F . Throughout this section, we will adopt the same notational convention when referring to the t -derivatives of other functions, in contrast to the 'dot' notation used in other sections of this paper.) This means that

<!-- formula-not-decoded -->

We will prove that, when glyph[lscript] is randomly sampled according to p D , the expected value of each term on the right side of (10) is bounded above by a constant. By Markov's Inequality, it will follow that right side is bounded above by a constant for a set S of leaves satisfying p D ( S ) ≥ 1 / 2, thus finishing the proof of the Lemma.

Unfortunately, bounding the expected value of the right side of (10) requires a separate computation for each of the four terms. For the first term, we observe that | F ′ ( t 0 , glyph[lscript] ) | 2 is precisely K D ( glyph[lscript] ), so E D ( | F ′ ( t 0 , glyph[lscript] ) | 2 ) ≤ c 3 √ n by Corollary 3.5. It follow, using the Cauchy-Schwarz Inequality, that E D ( | F ′ ( t 0 , glyph[lscript] ) | n -1 / 4 ) ≤ √ c 3 .

To bound the remaining three terms, let a 0 , a 1 , . . . , a n = glyph[lscript] be the nodes on the path in T from the root a 0 down to the leaf glyph[lscript] . Let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To prove that E D ( | F ′′ ( t 0 , glyph[lscript] ) | ) = O ( √ n ), we use the fact that the random variable F ′′ ( t 0 , glyph[lscript] ) is a sum of two random variables ∑ n -1 i =0 q ′′ ( a i ) q ( a i ) and -∑ n -1 i =0 ( q ′ ( a i ) q ( a i ) ) 2 . We bound the expected absolute value of each of these two terms separately.

We have so

For the second term, we use the fact that | q ′ ( a i ) /q ( a i ) | = O ( h a i ), which is property 3 from Lemma 3.1. Thus

<!-- formula-not-decoded -->

and the right side is O ( √ n ) using Lemma 3.3 and our hypothesis that E D R D ≤ √ n . To bound the first term, ∑ n -1 i =0 q ′′ ( a i ) q ( a i ) , we start by observing that, conditional on the value of a i , the random variable q ′′ ( a i ) q ( a i ) has mean zero and variance O (1). The bound on the conditional variance follows from property 4 in Lemma 3.1. The mean-zero assertion follows from the computation

<!-- formula-not-decoded -->

This means that the random variables q ′′ ( a i ) /q ( a i ) form a martingale difference sequence, hence

<!-- formula-not-decoded -->

The bound E D (∣ ∣ ∣ ∑ n -1 i =0 q ′′ ( a i ) q ( a i ) ∣ ∣ ∣ ) = O ( √ n ) follows using the Cauchy-Schwarz Inequality, as before.

We turn now to proving that E D ( | F ′′′ ( t 0 , glyph[lscript] ) | ) = O ( n 3 / 4 ). As before, the first step is to use (14) to express F ′′′ ( t 0 , glyph[lscript] ) as a sum of three terms

<!-- formula-not-decoded -->

and then to bound the expected absolute value of each of these terms separately. Exactly as above, one proves that the random variables q ′′′ ( a i ) /q ( a i ) form a martingale difference sequence and have bounded variance, and consequently E D ( | X | ) = O ( √ n ). Recalling that | q ′ ( a i ) /q ( a i ) | = O ( h a i ) and | q ′′ ( a i ) /q ( a i ) | = O (1) (properties 3 and 4 from Lemma 3.1, respectively) we find that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line follows from Lemma 3.3. Finally, we have

<!-- formula-not-decoded -->

Combining the estimates for E D ( | X | ) , E D ( | Y | ) , E D ( | Z | ), we obtain the bound E D ( | F ′′′ ( t 0 , glyph[lscript] ) | ) = O ( n 3 / 4 ) as desired.

Finally, to prove | F ′′′′ ( t 1 , glyph[lscript] ) | = O ( n ), we use the formula

<!-- formula-not-decoded -->

Each of the random variables q ( k )( a i ) /q ( a i ) for k = 1 , 2 , 3 , 4 is O (1), hence each summand on the right side of (15) is O (1). Summing all n terms, we obtain | F ′′′′ ( t 1 , glyph[lscript] ) | = O ( n ) as desired.