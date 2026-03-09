## OPTIMAL AUCTION DESIGN*t

## ROGER B. MYERSON

Northwestern University

This paper considers the problem faced by a seller who has a single object to sell to several possible buyers, when the seller has imperfect information about how much the might be willing to pay for the object. The seller's problem is to design an auction gam has a Nash equilibrium giving him the highest possible expected utility. Optimal auctio derived in this paper for a wide class of auction design problems.

1. Introduction. Consider the problem faced by someone who has an object to sell, and who does not know how much his prospective buyers might be willing to pay for the object. This seller would like to find some auction procedure which can give him the highest expected revenue or utility among all the different kinds of auctions known (progressive auctions, Dutch auctions, sealed bid auctions, discriminatory auctions, etc.). In this paper, we will construct such optimal auctions for a wide class of sellers' auction design problems. Although these auctions generally sell the object at a discount below what the highest bidder is willing to pay, and sometimes they do not even sell to highest bidder, we shall prove that no other auction mechanism can give higher expected utility to the seller.

To analyze the potential performance of different kinds of auctions, we follow Vickrey [11] and study the auctions as noncooperative games with imperfect information. (See Harsanyi [3] for more on this subject.) Noncooperative equilibria of specific auctions have been studied in several papers, such as Griesmer, Levitan, and Shubik [1], Ortega-Reichert [7], Wilson [12], [13]. Wilson [14] and Milgrom [5] have shown asymptotic optimality properties for sealed-bid auctions as the number of bidders goes to infinity. Harris and Raviv [2] have found optimal auctions for a class of symmetric two-bidder auction problems. Independent work on optimal auctions has also been done by Riley and Samuelson [8] and Maskin and Riley [4]. A general bibliography of the literature on competitive bidding has been collected by Rothkopf and Stark [10].

The general plan of this paper is as follows. ?2 presents the basic assumptions and notation needed to describe the class of auction design problems which we will study. In ?3, we characterize the set of feasible auction mechanisms and show how to formulate the auction design problem as a mathematical optimization problem. T lemmas, needed to analyze and solve the auction design problem, are presented in ?5 describes a class of optimal auctions for auction design problems satisfying a regulatory condition. This solution is then extended to the general case in ?6. In ?7, an example is presented to show the kinds of counter-intuitive auctions which may be optimal when bidders' value estimates are not stochastically independent. A few concluding comments about implementation are put forth in ?8.

* Received January 29, 1979; revised October 15, 1979.

AMS 1980 subject classification. Primary 90D45. Secondary 90C10.

IA OR 1973 subject classification. Main: Games.

OR/MS Index 1978 subject classification. Primary: 236 games/group decisions/noncooperative.

Key words. Auctions, expected revenue, direct revelation mechanisms.

tThe author gratefully acknowledges helpful conversations with Paul Milgrom, Michael Rothkopf, and especially Robert Wilson, who suggested this problem. This paper was written while the author was a visitor at the Zentrum fur interdisziplinare Forschung, Bielefeld, Germany.

0364-765X/81/0601 /0058$01.25

Copyright ? 1981, The Institute of Management Sciences

2. Basic definitions and assumptions. To begin, we must develop tions and assumptions, to describe the class of auction design pr paper will consider. We assume that there is one seller who has a si He faces n bidders, or potential buyers, numbered 1,2, . . ., n. We l set of bidders, so that

<!-- formula-not-decoded -->

We will use i and j to represent typi

The seller's problem derives from t various bidders are willing to pay fo some quantity ti which is i's value e maximum amount which i would be information about it.

We shall assume that the seller's uncertainty about the value estimate of bidder i can be described by a continuous probability distribution over a finite interval. Specifically, we let ai represent the lowest possible value which i might assign to the object; we let bi represent the highest possible value which i might assign to the object; and we let f :[ai,bi] - R be the probability density function for i's value estimate ti. We assume that: - oo &lt; ai &lt; bi &lt; + oo; fi(ti) &gt; 0, Vti E [ai, bi]; and fi(.) is a continuous function on [ai, bj. Fi: [ai, bi] - [0, 1] will denote the cumulative distribution function corresponding to the density fi(-), so that

<!-- formula-not-decoded -->

Thus Fi(t1) is the seller's assessment of the probability that bidd estimate of ti or less.

We will let T denote the set of all possible combinations of bidder that is,

<!-- formula-not-decoded -->

For any bidder i, we let T\_i denote the set of all possible co estimates which might be held by bidders other than i, so that

<!-- formula-not-decoded -->

Until ?7, we will assume that the value estimates of th independent random variables. Thus, the joint density t = (tl, . . . , t) of individual value estimates is

<!-- formula-not-decoded -->

Of course, bidder i considers his own value estimate to be a know random variable. However, we assume that bidder i assesses the pr tions for the other bidders' value estimates in the same ways as the sel both the seller and bidder i assess the joint density function on T t-i = (tl, ., ti\_l, ti+, ..., tn) of values for all bidders other than i

<!-- formula-not-decoded -->

The seller's personal value estimate for the object, if he were to kee to any of the n bidders, will be denoted by t0. We assume that the se information about the object, so that to is known to all the bidders.

There are two general reasons why one bidder's value estimates may the seller and the other bidders. First, the bidder's personal preferen unknown to the other agents (for example, if the object is a painting, the not know how much he really enjoys looking at the painting). Second, might have some special information about the intrinsic quality of the ob know if the painting is an old master or a copy). We may refer to these t preference uncertainty and quality uncertainty.' This distinction is ve there are only preference uncertainties, then informing bidder i abou estimate should not cause i to revise his valuation. (This does not mean not revise his bidding strategy in an auction if he knewj's value estim only that i's honest preferences for having money versus having the obje change.) However, if there are quality uncertainties, then bidder i might his valuation of the object after learning about other bidders' value es if i learned that tj was very low, suggesting thatj had received discour tion about the quality of the object, then i might honestly revise downward his assessment of how much he should be willing to pay for the object.

In much of the literature on auctions (see [11], for example), only the special case of pure preference uncertainty is considered. In this paper, we shall consider a more general class of problems, allowing for certain forms of quality uncertain as well. Specifically, we shall assume that there exist n revision effect functions ej: [ai, bi] -&gt; R such that, if another bidder i learned that tj wasj's value estimate for the object, then i would revise his own valuation by e.(tj). Thus, if bidder i learned that t = (t,, .. ., tn) was the vector of value estimates initially held by the n bidders, then i would revise his own valuation of the object to

<!-- formula-not-decoded -->

Similarly, we shall assume that the seller would reassess his object to

<!-- formula-not-decoded -->

if he learned that t was the vector of value estimates initially held by the bidd the case of pure preference uncertainty, we would simply have ej(t) = 0.

(To justify our interpretation of ti as i's initial estimate of the value of the object should assume that these revision effects have expected-value zero, so that

<!-- formula-not-decoded -->

However, this assumption is not actually necessary without it, only the interpretation of the ti would

3. Feasible auction mechanisms. Given the dens effect functions ei and vi as above, the seller's problem is to select an auction mechanism to maximize his own expected utility. We must now develop the notation to describe the auction mechanisms which he might select. To begin, we shall restrict our attention to a special class of auction mechanisms: the direct revelation mechanisms.

In a direct revelation mechanism, the bidders simultaneously and confidentially announce their value estimates to the seller; and the seller then determines who gets

'I am indebted to Paul Milgrom for pointing out this distinction.

the object and how much each bidder must pay, as some functi announced value estimates t = (t, . . ., tn). Thus, a direct revelation mechanism is described by a pair of outcome functions (p,x) (of the form p: T-&gt; R" and x: T Rn) such that, if t is the vector of announced value estimates then pi(t) is the probability that i gets the object and xi(t) is the expected amount of money which bidder i must pay to the seller. (Notice that we allow for the possibility that a bidder might have to pay something even if he does not get the object.)

We shall assume throughout this paper that the seller and the bidders are risk neutral and have additively separable utility functions for money and the object being sold. Thus, if bidder i knows that his value estimate is ti, then his expected utility from an auction mechanism described by (p, x) is

<!-- formula-not-decoded -->

where dt\_i = dt . .. dti\_ dti,+ . . . dt,.

Similarly, the expected utility for the seller from this auction mechanism is

<!-- formula-not-decoded -->

where dt = dtI ... dt,.

Not every pair of functions (p, x) represents a feasible auction mechanism, howev There are three types of constraints which must be imposed on (p, x).

First, since there is only one object to be allocated, the function p must satis following probability conditions:

<!-- formula-not-decoded -->

Second, we assume that the seller cannot force a bidder to parti which offers him less expected utility then he could get on his own. If he did not participate in the auction, the bidder could not get the object, but also would not pay any money, so his utility payoff would be zero. Thus, to guarantee that the bidders will participate in the auction, the following individual-rationality conditions must be satisfied:

<!-- formula-not-decoded -->

Third, we assume that the seller could not prevent any bidder from lying about value estimate, if the bidder expected to gain from lying. Thus the revelation me nism can be implemented only if no bidder ever expects to gain from lying. That honest responses must form a Nash equilibrium in the auction game. If bidder i claimed that si was his value estimate when ti was his true value estimate, then his expected utility would be

<!-- formula-not-decoded -->

where (t \_,si) = (tl, . . . , ti, s,+, .I . , t). Thus, to guarantee that no bidder has any incentive to lie about his value estimate, the following incentive-compatibility conditions must be satisfied:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We say that (p,x) is feasible (or that (p,x) represents a feasible auctio iff (3.3), (3.4), and (3.5) are all satisfied. That is, if the seller plans to allocat according to p and to demand monetary payments from bidders accord the scheme can be implemented, with all bidders willing to participate hone only if (3.3)-(3.5) are satisfied.

Thus far, we have only considered direct revelation mechanisms, in bidders are supposed to honestly reveal their value estimates. However, the design other kinds of auction games. In a general auction game, each bi set of strategy options ei; and there are outcome functions

<!-- formula-not-decoded -->

which described how the allocation of the object and the bidders' fees bidders' strategies. (That is, if 0 = (01, . . . , 9n) were the vector of strategi bidder in the auction game, then Ai() would be the probability of i get and xi(0) would be the expected payment from i to the seller.)

An auction mechanism is any such auction game together with a desc strategic plans which the bidders are expected to use in playing the ga strategic plan can be represented by a function i: [ai, bi] -&gt; i, such tha strategy which i is expected to use in the auction game if his value estimat general notation, our direct revelation mechanisms are simply those a nisms in which 3, = [ai, bi] and M(ti) \_ ti.

In this general framework, a feasible auction mechanism must satisf which generalize (3.3)-(3.5). Since there is only one object, the probabilit be nonnegative and sum to one or less, for any 0. The auction mechanis nonnegative expected utility to each bidder, given any possible value es he would not participate in the auction. The strategic plans must equilibrium in the auction game, or else some bidder would revise his p

It might seem that problem of optimal auction design must be quite because there is no bound on the size or complexity of the strategy spaces seller may use in constructing the auction game. The basic insight whic solve auction design problems is that there is really no loss of generality in only direct revelation mechanisms. This follows from the following fact.

LEMMA 1. (THE REVELATION PRINCIPLE.) Given any feasible auction mec exists an equivalent feasible direct revelation mechanism which gives to the bidders the same expected utilities as in the given mechanism.

This revelation principle has been proven in the more general contex collective choice problems, as Theorem 2 in [6]. To see why it is true, s are given a feasible auction mechanism with arbitrary strategy spaces Oi, w functions p and x, and with strategic plans Hi, as above. Then consid revelation mechanism represented by the functions p: T- Rn and x: T-

<!-- formula-not-decoded -->

That is, in the direct revelation mechanism (p,x), the seller first asks announce his type, and then computes the strategy which the bidder w according to the strategic plans in the given auction mechanism, and f ments the outcomes prescribed in the given auction game for these strategi direct revelation mechanism (p,x) always yields the same outcome auction mechanism, so all agents get the same expected utilities in bot

And (p,x) must satisfy the incentive-compatibility constraints (3.5

strategic plans formed an equilibrium in the given feasible mechan could gain by lying to the seller in the revelation game, then he c "lying to himself" or revising his strategic plan in the given mechanis feasible.

Using the revelation principle, we may assume, without loss of g seller only considers auction mechanisms in the class of feasible di mechanisms. That is, we may henceforth identify the set of feasi nisms with the set of all outcome functions (p,x) which satisfy t through (3.5). The seller's auction design problem is to choose th

-&gt; R" and x: T--&gt; R so as to maximize Uo(p,x) subject to (3.3)-(3.5)

Notice that we have not used (2.7) or (2.8) anywhere in this secti characterize the set of all feasible auction mechanisms even when t their revised valuations vi(t) using functions vi: T-- R, which are additive form (2.7). However, in the next three sections, to derive to the problem of optimal auction design, we shall have to restrict our class of problems in which (2.7) and (2.8) hold.

4. Analysis of the problem. Given an auction mechanism (p, x)

<!-- formula-not-decoded -->

for any bidder i and any value estimate ti. So Qi(p, ti) is the c that bidder i will get the object from the auction mechanism (p, x estimate is ti.

Our first result is a simplified characterization of the feasible auction mechanisms.

LEMMA 2. (p, x) is feasible if and only if the following conditions hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

PROOF. Using (2.8), our special assumption about the form of vi(t), we get

<!-- formula-not-decoded -->

Thus, the incentive-compatibility constraint (3.5) is equivalent to

<!-- formula-not-decoded -->

Thus (p, x) is feasible if and only if (3.3), (3.4), and (4.6) hold. We will now sh that (3.4) and (4.6) imply (4.2)-(4.4).

Using (4.6) twice (once with the roles of si and ti switched), we get

<!-- formula-not-decoded -->

Then (4.2) follows, when si &lt; ti.

These inequalities can be rewritten for any 8 &gt; 0

<!-- formula-not-decoded -->

Since Qi(p, si) is increasing in si, it is Riemann integrable. So:

<!-- formula-not-decoded -->

which gives us (4.3).

Of course, (4.4) follows directly from (3.4), so all the conditions in Lemma 2 fol from feasibility.

Now we must show that the conditions in Lemma 2 also imply (3.4) and (4.6).

Since Q,(p,s,) &gt; 0 by (3.3), (3.4) follows from (4.3) and (4.4).

To show (4.6), suppose si &lt; ti; then (4.2) and (4.3) give us:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, if s, &gt; ti then

<!-- formula-not-decoded -->

Thus (4.6) follows from (4.2) and (4.3). So the conditions in Lemma 2 also imply feasibility. This proves the lemma.

So (p, x) represents an optimal auction if and only if it maximizes U0(p, x) subje to (4.2)-(4.4) and (3.3). Our next lemma offers some simpler conditions for optimalit

LEMMA 3. Suppose that p: T- Rn' maximizes

<!-- formula-not-decoded -->

subject to the constraints (4.2) and (3.3). Suppose also that

<!-- formula-not-decoded -->

Then (p, x) represents an optimal auction.

PROOF. Recalling (3.2), we may write the seller's objective function as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

But, using Lemma 2, we know that for any feasible (p,x):

<!-- formula-not-decoded -->

From (2.7) and (2.8) we get

<!-- formula-not-decoded -->

Substituting (4.10) and (4.11) into (4.9) gives us:

<!-- formula-not-decoded -->

So the seller's problem is to maximize (4.12) subject to the constr (4.4), and (3.3) from Lemma 2. In this formulation, x appears only in the objective function and in the constraints (4.3) and (4.4). These two co be rewritten as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If the seller chooses x according to (4.8), then he satisfies both (4.3)

gets

<!-- formula-not-decoded -->

which is the best possible value for this term in (4.12).

Thus using (4.8), we can drop x from the seller's problem entirely second term on the right side of (4.12) is a constant, indepen objective function can be simplified to (4.7), and (4.2) and constraints left to be satisfied. This completes the proof of the lem

Equation (4.12) also has an important implication which is worth stating as a theorem in its own right.

COROLLARY (THE REVENUE-EQUIVALENCE THEOREM). The seller's expected utility from a feasible auction mechanism is completely determined by the probability function p and the numbers Ui(p, x, ai) for all i.

That is, once we know who gets the object in each possible situation (as specified by p)

and how much expected utility each bidder would get if his value estimate were at its

lowest possible level ai, then the seller's expected utility from the auction on the payment function x. Thus, for example, the seller must get th utility from any two auction mechanisms which have the properties that always goes to the bidder with the highest value estimate above to and (2) would expect zero utility if his value estimate were at its lowest possi bidders are symmetric and all ei = 0 and ai = 0, then the Dutch auctions a auctions studied in [11] both have these two properties, so Vickrey's equiv may be viewed as a corollary of our equation (4.12). However, we shall see auctions are not in general optimal for the seller.

5. Optimal auctions in the regular case. With a simple regularity a can compute optimal auction mechanisms directly from Lemma 3.

We may say that our problem is regular if the function

<!-- formula-not-decoded -->

is a monotone strictly increasing function of ti, for every i in regular if ci(si) &lt; ci(ti) whenever ai &lt; si &lt; ti &lt; bi. (Recall that we are assuming fi (ti) &gt; 0 for all ti in [ai, bi, so that ci(ti) is always well defined and continuous.)

Now consider an auction mechanism in which the seller keeps the object if to &gt; maxiEN (ci(ti)), and he gives it to the bidder with the highest ci(ti) otherwise. I ci(ti) = cj(tj) = maxkEN (ck(tk)) &gt; to, then the seller may break the tie by giving to the lower-numbered player, or by some other arbitrary rule. (Ties will only happen wit probability zero in the regular case.) Thus, for this auction mechanism,

<!-- formula-not-decoded -->

For all t in T, this mechanism maximizes the sum

<!-- formula-not-decoded -->

subject to the constraints that

<!-- formula-not-decoded -->

Thus p maximizes (4.7) subject to the probability condition (3.3). To check tha satisfies (4.2) we need to use regularity. Suppose si &lt; ti. Then ci(si) &lt; c(ti), a whenever bidder i could win the object by submitting a value estimate of si, also win if he changed to ti. That is pi(t\_i,si) &lt; pi(t\_i,ti), for all t\_i. So Qi(p, probability of i winning the object given that ti is his value estimate, is in increasing function of ti, as (4.2) requires. Thus p satisfies all the conditions of Le 3.

To complete the construction of our optimal auction, we let x be as in (4.8):

<!-- formula-not-decoded -->

This formula may be rewritten more intuitively, as follows. For estimates from bidders other than i, let

<!-- formula-not-decoded -->

Then zi(t\_ ) is the infimum of all winning bids for i against t\_i; so

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, (4.8) becomes

<!-- formula-not-decoded -->

That is, bidder i must pay only when he gets the object, a zi(t\_i)), the amount which the object would have been submitted his lowest possible winning bid.

If all the revision effect functions are identically zero (tha bidders are symmetric (ai = aj, bi = b, f(.) = fj(. )) and regul

<!-- formula-not-decoded -->

That is, our optimal auction becomes a modified Vickrey auction [11] seller himself submits a bid equal to ci- l (to) (notice that all ci = cj i case, and regularity guarantees that ci is invertible) and then sells t highest bidder at the second highest price. This conclusion only hold the bidders are symmetric and the c,(-) functions are strictly increasing

For example, suppose to = 0, each ai = 0, bi = 100, ei(ti) = 0, and every i and every ti between 0 and 100. Then straightforward com ci(ti) = 2ti - 100, which is increasing in ti. So the seller should sell to the at the second highest price, except that he himself should submit a bid 100/2 = 50. By announcing a reservation price of 50, the seller ris (1/2)n of keeping the object even though some bidder is willing to pay it; but the seller also increases his expected revenue, because he can c price when the object is sold.

Thus the optimal auction may not be expost efficient. To see more can happen, consider the example in the above paragraph, for the c Then the seller has value estimate to = 0, and the one bidder has a value from a uniform distribution on [0, 100]. Ex post efficiency would r bidder must always get the object, as long as his value estimate is positiv bidder would never admit to more than an infinitesimal value esti positive bid would win the object. So the seller would have to expect zero never kept the object. In fact, the seller's optimal policy is to refuse for less than 50, which gives him expected revenue 25.

More generally, when the bidders are asymmetric, the optimal auc times even sell to a bidder whose value estimate is not the highest. F ei(ti) = 0 and fi(ti) = 1/(bi - ai) for all ti between ai and bi (the gen distribution case with no revision effects) we get

<!-- formula-not-decoded -->

This gives us

which is increasing in ti. So in the optimal auction, the bidder with the will get the object. If bi &lt; bj, then i may win the object even if ti &lt; t 2ti - bi &gt; 2tj - b. In effect, the optimal auction discriminates against whom the upper bounds on the value estimates are higher. This discri discourages such bidders from under-representing value estimates close to t bounds.

6. Optimal auctions in the general case. Without regularity, the auction mechanism proposed in the preceding section would not be feasible, since it would violate (4.2). To extend our solution to the general case, we need some carefully chosen definitions.

The cumulative distribution function Fi [ai, bi]--&gt;[0, 1] for bidder i is continuous and strictly increasing, since we assume that the density function f, is always strictly positive. Thus F,(.) has an inverse F,1i- [0, ]-&gt; [ai, bi], which is also continuous and strictly increasing.

For each bidder i, we now define four functions which have the unit interval [0, 1] as their domain. First, for any q in [0, 1], let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next let Gi:[0, 1]-&gt;R be the convex hull of the function Hi(-); in th Rockafellar ([9, p. 36])

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

That is, G/(-) is the highest convex function on [0, 1] such that Gi(q) q.

As a convex function, G, is continuously differentiable except at countably many points, and its derivative is monotone increasing. We define g : [0, 1]-&gt; R so that

<!-- formula-not-decoded -->

whenever this derivative is defined, and continuity.

We define C: [ai, bi] -&gt; R so that and let

<!-- formula-not-decoded -->

(It is straightforward to check that, in the regular case w Gi = Hi, gi = hi, and ci = ci.)

Finally, for any vector of value estimates t, let M(t) be t c,(ti) is maximal among all bidders and is higher than t

<!-- formula-not-decoded -->

We can now state our main result: that in an optimal auction, th always be sold to the bidder with the highest ci(ti), provided this i Thus, we may think of ci(ti) as the priority level for bidder i when hi ti, in the seller's optimal auction.

THEOREM. Let p: T--&gt; R and x: T-&gt; R" satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all i in N and t in T. Then (p, x) represents an optimal auction m

PROOF. First, using integration by parts, we derive the following

<!-- formula-not-decoded -->

But Gi is the convex hull of Hi on [0, 1] and Hi is conti Gi(l) = Hi(l). Thus the endpoint terms in the last expressi

Now, recall the maximand (4.7) in Lemma 3. Using (6.9)

<!-- formula-not-decoded -->

Now consider (p,x) as defined in the theorem. Observe that p always puts all probability on bidders for whom (ei(ti)- to) is nonnegative and maximal. Thus, f any p satisfying (3.3):

<!-- formula-not-decoded -->

Of course p itself does satisfy the probability condition (3.3).

and

For any p which satisfies (4.2) (that is, for which Qi(p, ti) is an increasin t,), we must have

<!-- formula-not-decoded -->

since Hi &gt; Gi.

To see that p satisfies (4.2), observe first that ci(ti) is an increasing function of ti, because Fi and gi are both increasing functions. Thus pi(t) is increasing as a function of ti, for any fixed t\_i, and so Qi(p, ti) is also an increasing function of ti. Sop satisfies (4.2).

Since G is the convex hull of H, we know that G must be flat whenever G &lt; H; that is, if Gi(r) &lt; Hi(r) then g;(r) = Gi"(r) = 0. So if Hi(Fi(ti)) - Gi(Fi(ti)) &gt; 0 then ci(ti)

and Qi(p, ti) are constant in some neighborhood of ti. This implies that

<!-- formula-not-decoded -->

Substituting (6.11), (6.12), and (6.13) back into (6.10), we can see that p maximizes (4.7) subject to (4.2) and (3.3). This fact, together with Lemma 3, proves the theorem.

To get some practical interpretation for these important ci functions, consider the special case of n = 1; that is, suppose there is only one bidder. Then our optimal auction becomes:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

That is, the seller should offer to sell the object at the price

<!-- formula-not-decoded -->

and he should keep the object if the bidder is unwilling to pay this price.

Thus, if bidder i were the only bidder, then the seller would sell the object to i if an only if ci(ti) were greater than or equal to to. In other words, ci(ti) is the highest level to, the seller's personal value estimate, such that the seller would sell the object to i at a price of ti or lower, if all other bidders were removed.

7. The independence assumption. Throughout this paper we have assumed that the bidders' value estimates are stochastically independent. Independence is a strong assumption, so we now consider an example to show what optimal actions may look like when value estimates are not independent.

For simplicity, we consider a discrete example. Suppose there are two bidders, each of whom may have a value estimate of ti = 10 or ti = 100 for the object. Let us assume that the joint probability distribution for value estimates (tl, t2) is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Obviously the two value estimates are not independent. Let us also assume that there are no revision effects (e* = 0), and to = 0.

Now consider the following auction mechanism. If both bidders have high value estimates (t, = t2 = 100), then sell the object to one of them for price 100, randomizing equally to determine which bidder buys the object. If one bidder has a high value estimate (100) and the other has a low value estimate (10), then sell the object to the

<!-- formula-not-decoded -->

high bidder for 100, and charge the low bidder 30 (but give him bidders have low value estimates (10), then give 15 units of money to o give 5 units of money and the object to the other, again choosing t object at random.

The outcome functions (p,x) of this auction mechanism are:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This may seem like a very strange auction, but in fact it is optimal. It is straightforward to check that honesty is a Nash equilibrium in this auction game, in that neither bidder has any incentive to misrepresent his value estimate if he expects the other bidder to be honest. Furthermore, the object is always delivered to a bidder who values it most highly; and yet each bidders' expected utility from this auction mechanism is zero, whether his value is high or low. So this auction mechanism is feasible and it allows the seller to exploit the entire value of the object from the bidders. Thus this is an optimal auction mechanism, and it gives the seller expected revenue

<!-- formula-not-decoded -->

To see why this auction mechanism works so well, observe that the seller is really doing two things. First, he is selling the object to one of the highest bidders at the highest bidders' value estimate. Second, if a bidder says his value estimate is equal to 10, then that bidder is forced to accept a side-bet of the following form: "pay 30 if the other bidder's value is 100, get 15 if the other bidder's value is 10." This side-bet has expected value 0 to a bidder whose value estimate is truly 10, since then the conditional probability is 1/3 that the other has value 100 and 2/3 that the other has value 10. But if a bidder were to lie and claim to have value estimate 10, when 100 was his true value estimate, then this side-bet would have expected value 2 (- 30) + 3 (10) =- 5 for him (since he would now assess conditional probabilities 2 and 1 3 3 3 re ne te

Th sid co th co sel ex an

Of Fo th wh me

2Er

gua

change x to:

<!-- formula-not-decoded -->

keeping p as above.)

One might ask whether there are any optimal auctions for our example which do not have this strange property of sometimes telling the seller to pay money to the bidders. The answer is No; if we add the constraint that the seller should never pay money to the bidders (that is, all xi(t) &gt; 0), then no feasible auction mechanism gives the seller expected utility higher than 662 . To prove this fact, observe that the auction design problem is a linear programming problem when the number of possible value estimates is finite, as in this example. The objective function in the problem is Uo(p, x), which is linear in p and x. As in ?2, the feasibility constraints are of three types: probability constraints (p,(t) &gt; 0, ipi(t) &lt; 1), individual-rationality constraints (Ui(p,x,ti) &gt; 0), and incentive-compatibility constraints (that Ui(p, x, ti) must be greater than or equal to the utility which i would expect from acting as if si were his value estimate when ti was true). All of these constraints are linear in p and x. So we get a linear programming problem, and for our example its optimal value is 70, with the optimal solution shown above. But if we add the constraints xi(t) &gt; 0 for all i and t, then the optimal value drops to 66 2, for this example. To attain this "second-best" value of 662 with nonnegative x, the seller should keep the object if t1 = t2 = 10, and otherwise the seller should sell the object to a high bidder for 100.

8. Implementation. A few remarks about the implementability of our optimal auctions should now be made. Once the f and ei functions have been specified, the only computations necessary to implement our optimal auction are to compute the ci functions and to evaluate (6.8). But these are all straightforward one-dimensional problems. The equilibrium strategies for the bidders are also easy to compute in our optimal auction, since each bidder's optimal strategy is to simply reveal his true value estimate.

In terms of sensitivity analysis, notice that (6.8) guarantees that our auction mechanism (p, x) will be feasible, and yet the densities f do not appear in (6.8). So our optimal auction will satisfy the individual-rationality and incentive-compatibility constraints ((3.4) and (3.5)) even if the density functions are misspecified from the point of view of the bidders. However the revision-effect functions ei do appear in (6.8) (through vi), so if there are errors in specifying the ei functions then bidders may have incentive to bid dishonestly in the auction we compute.

In general, we must recognize that an auction design problem must be treated like any problem of decision-making under uncertainty. No auction mechanism can guarantee to the seller the full realization of his object's value under all circumstances. Thus, the seller must make his best assessment of the probabilities and choose the auction design which offers him the highest expected utility, on average. The usual "garbage-in, garbage-out" warning must apply here, as in all operations research, but careful use of models and sensitivity analysis should enable a seller to improve his average revenues with optimally designed auctions.

## References

[1] Griesmer, J. H., Levitan, R. E. and Shubik, M. (1967). Towards a Study of Bidding Processes, Part Four: Games with Unknown Costs. Naval Res. Logist. Quart. 14 415-433.

[2] Harris, M. and Raviv, A. (1978). Allocation Mechanism and the Design of Auction. Working Paper, Graduate School of Industrial Administration, Carnegie-Mellon University, Pittsburgh, PA.

- [3] Harsanyi, J. C. (1967-1968). Games with Incomplete Information Played by " Management Sci. 14 159-189, 320-334, 486-502.
- [4] Maskin, E. and Riley, J. G. (1980). Auctioning an Indivisible Object. Discussion Paper No. 87D, Kennedy School of Government, HIarvard University.
- [5] Milgrom, P. R. (1979). A Convergence Theorem for Competitive Bidding with Differential Information. Econometrica. 47 679-688.
- [6] Myerson, R. B. (1979). Incentive Compatibility and the Bargaining Problem. Econometrica. 47 61-73.
- [7] Ortega-Reichert, A. (1968). Models for Competitive Bidding under Uncertainty. Technical Report 8, Department of Operations Research, Stanford University.
- [8] Riley, J. G. and Samuelson, W. F. (to appear). Optimal Auctions. American Economic Review.
- [9] Rockafellar, R. T. (1970). Convex Analysis. Princeton University Press, Princeton.
- [10] Rothkopf, M. H. and Stark, R. M. (1979). Competitive Bidding: a Comprehensive Bibliography. O 27 364-390.
- [11] Vickrey, W. (1961). Counterspeculation, Auctions and Competitive Sealed Tenders. Journal of Fi 16 8-37.
- [12] Wilson, R. B. (1967). Competitive Bidding with Asymmetrical Information. Management Sci. 13 A816-A820.
- [13] . (1969). Competitive Bidding with Disparate Information. Management Sci. 15 446-448.
- [14] . (1977). A Bidding Model of Perfect Competition. Review of Economic Studies 44 511

GRADUATE SCHOOL OF MANAGEMENT, NORTHWESTERN UNIVERSITY, 2001 SHERIDAN ROAD, EVANSTON, ILLINOIS 60201