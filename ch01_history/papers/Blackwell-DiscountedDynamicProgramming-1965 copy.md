JSTOR

e

<!-- image -->

## Discounted Dynamic Programming

## Author(s): David Blackwell

Source: The Annals of Mathematical Statistics , Feb., 1965, Vol. 36, No. 1 (Feb., 1965), pp. 226-235

Published by: Institute of Mathematical Statistics

Stable URL: https://www.jstor.org/stable/2238089

## REFERENCES

Linked references are available on JSTOR for this article: https://www.jstor.org/stable/2238089?seq=1&amp;cid=pdfreference#references\_tab\_contents You may need to log in to JSTOR to access the linked references.

JSTOR is a not-for-profit service that helps scholars, researchers, and students discover, use, and build upon a wide range of content in a trusted digital archive. We use information technology and tools to increase productivity and facilitate new forms of scholarship. For more information about JSTOR, please contact support@jstor.org.

Your use of the JSTOR archive indicates your acceptance of the Terms &amp; Conditions of Use, available at https://about.jstor.org/terms

<!-- image -->

Institute of Mathematical Statistics  is collaborating with JSTOR to digitize, preserve and extend access to The Annals of Mathematical Statistics

## DISCOUNTED DYNAMIC PROGRAMMING

BY DAVID BLACKWELL'

University of California, Berkeley

1. Introduction. Soon after the appearance of Wald's work in sequenitial analysis, Richard Bellman recognized the broad applicability of the methods of sequential analysis, named this body of methods dynamic programming, and applied the methods to many problems (see [1] and papers cited there). The first developm--ent of a general theory underlying these methods is due to Karlin [6], and a rather complete analysis of the finite case was given by Howard [5]. Dubins and Savage [3] have recently developed a general theory of gambling; the relation of gambling to dynamic programming is not completely clear, but it is certainly close.

Our formulation of a dynamic programming problem is somewhat narrower than Bellman's. For us, a dynamic programming problem is specified by four objects S, A, q, r, where S, A are any non-empty Borel sets, q associates with each pair (s, a) - S X A a probability distribution q( s I S, a) on S, and r is a bounded Baire function on S X A X S. We think of S as the set of possible states of somne system, and A as the set of acts available to you. Periodically, say once a day, you observe the current state s of the system, then choose an act a E A. Theni the system moves to a new state s' (which will be the state you observe tomorrow), selected according to q( - I s, a), and you receive a reward r(s, a, s'). Your problem is, given the initial state of the system, to maxilmize your total expected reward over the infinite future.

This total expected reward may well be infinite, for example, if r 1. Or it may well be unidefined. For example, if S has two elements 0, 1, A has only a single elemenit, q is deterministic with 0 -4 1, 1 -* 0, and the transition 0 1 yields 81, while 1 -&gt; 0 costs $1, the series of rewards, starting in state 0, is 1- 1 + 1 - 1 + * . We shall avoid this problem by introducing a discount factor 3, 0&lt; ? &lt; 1, so that unit reward on the nth day is worth only f3r`1 and shall try to maxinmize the total discounted expected reward.

A plan iX specifies for each n ? 1 what act to choose on the nth day as a Borel measurable function of the history h = (si , a1, * * , s.) of the system to date or, more genierally, it specifies for each h a probability distribution over A. Associated with each it is a bounded function i(-r) on S, the total expected discounted reward from ii, as a function of the initial state of the system. We shall be especially interested in the (non-randonized) stationary plans r. A stationary 7r is defined by a single function f mapping S into A: whenever the system is in state s, you choose act f(s).

Received 24 September 1964.

Prepared with the partial support of the National Science Foundation, Grant GP-2593.

Our main results are

- (1) There need not exist an e-optimal 7r, i.e. we give an example in which there is an E &gt; 0 such that for every -r there is a 7r' such that

<!-- formula-not-decoded -->

- (2) There always exists a (p, E) -optimal stationary 7r*, i.e. for any probability distribution p on S and any e &gt; 0, there is a stationary 7r* such that, for every 7r,

<!-- formula-not-decoded -->

- (3) Not every 7r need be dominated within E by a stationary 7r*, i.e. we g example of a -r and an E &gt; 0 such that, for every stationary 7r*,

<!-- formula-not-decoded -->

- (4) If A is countable, there is an e-optimal stationary 7r*, i.e. for every E &gt; 0, there is a stationary 7r* such that, for every ir,

<!-- formula-not-decoded -->

- (5) If A is finite there is an optimal stationary 7r*, i.e. there is a stationary 7r such that, for every -r,

<!-- formula-not-decoded -->

- (6) If there is an optimal 7r, there is one which is stationary. (Theorem 6(c)).
2. Probabilistic definitions and notation. By a Borel set we mean a Borel subset of some complete separable metric space. A probability on a non-empty Borel set X is a probability measure defined over the Borel subsets of X; the set of all probabilities on X is denoted by P(X). For any non-empty Borel sets X, Y, a conditional probability on Y given X is a function q( . ) such that for each x e X, q( . I x) is a probability on Y and for each Borel set B c Y, q(B I ) is a Baire function on X. The set of all conditional probabilities on Y given X is denoted by Q(Y I X). The product space of X and Y will be denoted by XY. The set of bounded Baire functions on X is denoted by M(X). For any u 8 M(XY) and any q 8 Q( Y X), qu denotes the element of M(X) whose value at xo E X is qu(xo) = J u(xo, y) dq(y I xo). For any p 8 P(X) and any u 8 M(X), pu is the integral of u with respect to p. For any p e P(X), q e Q( Y I X), pq is the probability on XY such that, for every u 8 M(XY), pq (u) = p(qu). Every probability m on' XY has a factorization m = pq; p is unique and is just the marginal distribution of the first coordinate variable with respect to m; q is not quite unique; it is a version of the conditional distribution of the second coordinate variable given the first. These facts and all others in this section, except the Lemma at the end, are in [7].

We extend the above notation in an obvious way to a finite or countable sequence of non-empty Borel sets X1, X2, * X X . If qn E Q(X?+1 I X, * E X,) for n &gt; 1 and p E P(X1), pql ... qn is a probability on X1X2 ... Xn+1, pqlq2 ... is

a probability on the iiifinite product space X1X2 ' **, q2q3 E Q(X3X4 X1X2) for any U E lI(XlX2 ... Xnf1) n &gt; 1 and any n, 1 ? m ? n, qm * qnu e M(X1 ... Xm), etc.

To avoid further complicating an already involved notation, we introduce an ambiguity as follows: for any function u on Y, we shall use the same symbol u to denote the function v on XY such that v(x, y) = u(y) for all y. Thus, for example, for arnyq - Q( Y I X), u - M( Y), qu - M(X); anyq - Q( Y I X) will also denote the element q' of Q(Y I ZX) defined by q'( I z, *) = q( . ), etc.

A p E P(X) is degenerate if it is concentrated at some one point x ? X; a q - Q( Y X) is degenerate if each q( * I x) is degenerate. The degenerate q are exactly those for which there is a Baire function f mapping X into Y for which q({f(x)} I x) = 1 for all x - X. Any such f will also denote its associated degenerate q, so that, for any u - MI(XY), fu(x) = u(x, f(x) ) for all x - X.

We shall use the following.

LEMMA [2]. For any q e Q(Y IX), u - M(XY), E &gt; 0 there is a degenerate f -Q(Y IX) such that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

The Leiimma asserts that, in the situation where we observe x E X then choose y E Y, receiving an income u(x, y), any randomized plan q can be replaced by a non-randomized plan f such that (1) our expected income for each x is at least as large as it was before and (2) with probability 1, for each x, the actual income under q does not exceed the actual income under f by as much as E.

3. Dynamic programming definitions and notation. A dynamic programming problem is defined by S, A, q, r, fi, where S, A are any non-empty Borel sets, q - Q(S I SA), r - M(SAS), andO ?&lt; f &lt; 1. A planr is a sequence (7rw, 7r2, - * *), where w,n - Q(A I H.) and H. = SA ... S(2n - 1 factors) is the set of possible histories of the system when the nth act must be chosen. A plan r is (non-randomized) Markov if each wr is a degenerate elemeent of Q(A IS), i.e. r = (fi, f2 ), where each fn is a Baire function from S into A, and is (nonrandonmized) stationary if there is a Baire function f mapping S into A such that 7rn = f for all n. The stationary plan defined by f is denoted by f(?).

Any plan 7r, together with the law of Ii-otion q of the system, defines for each initial state s a conditional distribution on the set Q = ASAS ... of futures of the system, i.e. it defines an element of Q(Q I S), namely e,, = 7rl7r2q * . - . Denote the coordinate functions on SO by o, a,, 02 X a2 , ... so that our reward on the nth day, as a function of the history of the system, is r(-n, a., X n+I), and our total discounted reward is u = Z E3n-'r(o-n I an 2 0n+I). The expected total discounted reward from r, as a function of the initial state, is then

<!-- formula-not-decoded -->

(Note the use of ambiguous notation.)

For ainy p E P(S), and any e &gt; 0, wr* will be called (p, e)-optimal if p{I(7r) &gt; I(wr*) + e} = 0 for every7r. 7r* will be called e-optimal if it is optimal for every p, or, equivalently, if I(7r) ? I(wr*) + e for all 7r, s, and will be called optimal if it is e-optimal for every e &gt; 0 or, equivalently, if I(7r) ? I(7r*) for all 7r, s. (p, E)-optimal stationary plans always exist, but p-optimal, i.e. (p, e) -optimal for every E &gt; 0, E-optimal plans (stationary or not) may not exist, as the following examples show.

EXAMPLE 1. (There are no p-optimal plans). S has a single element, say 0, and A has countably mnany elements, say 1, 2, 3, *.. . We take r(O, a, 0) = (a - 1)/a. There is nor with I(7r) = 1/(1 - 3), but sup7r I(r) = 1/(1 - i).

EXAMPLE 2. (There are no E-optimal plans). We take S = A = unit interval [0, 1]. The state of the system remains fixed: (s, a) -* s, and the reward r is 1 or O according as (s, a) is in a given Borel subset B c SA or not. For any r = (711, 712, ** ) {s: 7rwqr &gt; O} is a Borel subset of the projection D of B on S. For B chosen so that D is not a Borel set there is an So E D for which rir = 0, so that I(X) (so) ? , + 023+ * = (7 0-). Since there isa w* with I(7*) (so) = 1/(1 - d), wr is not e-optimal for any e &lt; 1.

## 4. Existence of (p, E)-optimal 7r.

THEOREM 1. For any p E P(S) and any e &gt; 0 there is a (p, E)-optimal plan.

The proof of Theorem 1 is simple but, regrettably, non-constructive. We associate with each r the number pI(r), the expected return from wr when the initial state has distribution p, deiiote by v the upper bound over all 7r of the numbers pI(7w), choose a sequence 7r 72) . . . of policies with pI(w(r)- v, and set u = sup, I(w n).

Let Sn consist of all s for which n is the smallest k with I(w7r(k)) \_ u -, and let 7r* be the plan which uses 7(n) for all initial states s E S, X i.e.

<!-- formula-not-decoded -->

Then I(7r*) = I(7rw(n) on Sn , and I(7r*) &gt; u - e everywhere. We show that, for every 7r, p{I(r) &lt; u} = 1, which will show that 7r* is (p, E)-optimal. For, take any w and any y &gt; 0. The construction above, applied to the sequence w, 1(2)** yields a 7r** with I(7r**) ? max (u, I(7r)) -y everywhere. But pI(w**) &lt; v &lt; pu, while pI(w**) ? p max (u, I(7r)) -y, so that p max (u, I(7r)) ? pu + 'y. Since y is any positive number, p max (u, I(7r)) &lt; pu, and p{I(7r) &lt; u} = 1.

## 5. (p, E)-domination by Markov 7r.

THEOREM 2. For any p E P(S), e &gt; 0, 7r, there is a Mfarkov 7r* which (p, E)domiinates 7r, i.e. p{I(7r*) ? I(7r) - e} = 1.

PROOF. We may suppose that 7r is already M\arkov from some point on, say for n &gt; N, since any two policies 7r, 7r' which agree for the first N days have III(7r') - I(7r) 11 &lt;-3NIIrII/(1 - () where, for any u E M(S), liull = sup8 lu(s) 1. We now show that, if wr = (71i, * , 7rN X fN+l X - - * ) is Markov for n &gt; N, for any -y &gt; 0 there is an fN mapping S into A with p{I(7r') ? I(7r) - y} = 1,

where 7r' = (7r, 7 **, X fN , fN -y = E/N, will produce a Markov ir*

To find fN, we write I(7r) = 7rlq ... 7rNlq(u + fO N7rNqv), where u(si, a, ,, SN) = sN1 k1 r(Sk ak , Skil) and V(SN, aN , SN+i) = r(SN , aN , SN+1) + (Z'=i fN+lq ... fN+kqr) (SN , aN SN+1). It suff fN for which

<!-- formula-not-decoded -->

Consider the probability n = p7rlq ... 7rN on SA ... SA (2N factors), and denote the coordinate variables by ci , ai, ... , cN U aN . For any fN , x = 7r,q ... 7rN-1qfNW(c1) is a version of E(w(o-N , fN (-N)) I c-1) and y = 7riql ... 7rN-1lrNW(-1i) is a version of E(w(cN , aN) I c,). If we choose fN so that W(cN Xf(cN)) \_ W(0-N c aN) - y with probability 1, we shall have x &gt;? y - y with probability 1, which is equivalent to (3). That such anfN exists follows at once from the Lemma of Section 2 with X = S, Y = A, q a version of the conditional distribution of aN given cN U =w, and E = -y. This completes the proof.

COROLLARY. For any p E P(S), E &gt; 0, there is a (p, E)-optimal Markov 7r*.

PROOF. From Theorem 1 there is a (p, E/2)-optimal 7r and from Theorem 2

there is a Markov r* which (p, e/2) -dominates ir. This 7r* is (p, e) -optimal.

In Theorem 2 we cannot replace (p, E)-domination by E-domination. Here is an example.

ExAMPLE 3. (A plan 7r which cannot be E-dominated by a Markov plan). We take S = B u X, where B is a Borel subset of the unit square XY whose projection D on X is not a Borel set. A is the unit interval. The law of motion q is degenerate and independent of a: (x, y) -* x, x -* x. r(x, a, x) = 1 if (x, a) ? B, r = 0 otherwise. Any plan 7r* such that ir*( f sl, aU, ..., s n) is degenerate at y whenever si = (x, y) has I(7r*) = A/ (1 -A on B. For any 7r = (X1 , 7r2 , ... *) for which T2 Q(A I S), i.e. does not depend on the initial state, the set of x ? X for which 7r2qr &gt; 0 is a Borel subset of D, so there is an xo E D for which r2qr = 0. For any yo with (xo, yo) e B, we have

<!-- formula-not-decoded -->

so I(7r) &lt; I(7r*) - for some s.

6. Stationary plans and operators. Associated with each Baire function f mapping S into A is a corresponding operator T, mapping M(S) into M(S), defined as follows. For u e M(S), Tu = fq(r + /3u), where the u on the right, considered as a function on SAS, depends on the last coordinate only. Tu is our expected income, as a function of the initial state, if we start using f(?) but are terminated at the beginning of the second day with a final reward u(s'), where s' is the state at termination. Tnu has a similar interpretation, replacing "second'" by "n + 1st". The following properties ot T, formulated as a theorem, are immediate.

THEOREm 3. (a) T is monotone, i.e. u &lt; v for all s implies Tu &lt; Tv for all s.

- (b) For any constant c, T(u + c) = Tu + f3c.
- (c) For arty Markov r = (fi , f2, * ), TI (7r) = I (f, 7r), where (f, 7r) denotes the Markov plan (f, fi, f2, * ).

For any Mlarkov 7r = (fi , f2, * ) we shall say that f mapping S into A is 7r-generated if there is a partition of S into Borel sets Si, S2, .. such that f = fn on Sn; we say that a Markov 7r' = (q1, 92, *** ) is 7r-generated if each gn is 7r-generated. We associate with each Markov 7r the operator U, mapping M( S) into -11(S), defined by Uu = supn Tnu, where Tn is the operator associated with fn . The following interpretation of U will be justified later. U'u is our optimal expected return, over all 7r-generated Markov 7r, as a function of the initial state, if we start using 7r' but are terminated at the beginning of the n + 1st day with a final reward u (s'), where s' is the state at termination. Here are some basic properties of U.

THEOREM 4. (a) UT is mnonotone.

- (b) For any constant c, U(u + c) = Uu + 3c.
- (c) For any T' associated with a 7r-generated f, Tu &lt; Uu.
- (d) For any u E M1 ( S) and any e &gt; 0, there is a 7r-generated f whose associate T satisfies Tu ? Uu - e.

PROOF. (a), (c) are immediate. For (b) we have

<!-- formula-not-decoded -->

so that U(u + c) &lt; Uu + ,Bc. This inequality, with u replaced by u + c, c by -c, yields Uu &lt; U(u + c) - fc, establishing (b). For (d), let Sn consist of all s for which

<!-- formula-not-decoded -->

and set f = fn for s E S. . Then, for any v, Tv = Tnv on S. , where T is ass withf. In particular, Tu = Tnu &gt; Uu - E on Sn , so Tu &gt; Uu - e everywhere.

To justify our informal interpretation of U, note that, for any Markov 7r' = (1 92 *... ), the total income from 7r' with termination on the n + 1st day with final payment u is

<!-- formula-not-decoded -->

where Ti' is the operator associated with gi. If 7r' is 7r-generated, Tiv \_ Uv fo all i, so that In(7r', u) ? U'u. To find 7r with In(7r', u) &gt; UnU - E choose any positive numbers -i, and choose gi 7r-generated so that

<!-- formula-not-decoded -->

By induction downward on i, starting at i = n, we obtain

<!-- formula-not-decoded -->

where d, = El + fEj+i + + /3En. For i = 1 we obtain In(r', u) &gt; U'u - di, and the Ei can be chosen so that di ? E.

THEOREM 5. If U is any operator with properties (a) and (b), U is a contraction with modulus f, i.e. IIUu - Uvfj ? I 3ju - vf l, so that, from the Banach fixed-point theorem, U has a unique fixed point u*, and 11 Unu - u*ii &lt; 3'llu - u*1i for all n.

PROOF. V ? u + i|u - vii yields

<!-- formula-not-decoded -->

using (a)-, (b). Interchange u and v to obtain Uu ? Uv + llJu - vll, completin the proof.

The principal general results on optimal plans are contained in the following theorem. Related results are given by Dubins and Savage [3], as indicated.

THEOREM 6. (a) For any Markov r = (f' , f2, * * * ), denoting by Tn the operator associated with fn and by U = sup T, the operator associated with ir, the fixed point u* of U is the optimal return among ir-generated plans: I(r') &lt; u* for every 7rgenerated r', and for every E &gt; 0 there is a ir-generated f such that I(f (X) ? u*

- E. Any f with Tu* \_ u* - E(1- ) satisfies this inequality.
- (b) For any p PF(S), E &gt; 0, there is a (p, E)-optimal plan which is stationary.
- (c) For any e &gt; 0, if there is an E-optimal r* = (7r , 7r2, *** ), there is an E/ (1 - 8)-optimal plan which is stationary ([3], Theorem 3.9.6).
- (d) Denote for each a E A by Ta the operator associated with f a. Any u with T.u ? u for all a is an upper bound on incomes: I (ir) ? u for all 7r ([3], Theorems 2.12.1, 3.3.1).
- (e) If for every E &gt; 0 there is an E-optimal plan, then the optimal return u* is Baire function and it satisfies the optimality equation u* supa Tau* ( [3], Theore 3.3.1).
- (f) A ir is optimal if and only if its return I(ir) satisfies the optimality equation. PROOF. (a) For any 7r-generated 7r' = (gl , g2, * * ), we have I(ir') = T,' ... Tnr'u, where un = I(gn+1, gn+2, ) and Ti' is the operator associated with g2,. Since each Ti' is a contraction with modulus ,

<!-- formula-not-decoded -->

Thus T1' ... Tnu* I(Qr') as n -+ oo. But T,' ... Tn u* ? Uru&amp; = u* so that I(r') &lt; u*. From Theorem 4(d), there is a ir-generated f for which Tu* &gt; Uu* - = u* - E, where E' = E(1 - f). We verify inductively that

<!-- formula-not-decoded -->

Since T nu* I (f(??)), we conclude that

<!-- formula-not-decoded -->

- (b) From the Corollary to Theorem 2, there is a (p, e/2) -optimal Markov = (fi, f2, ***) From (a), there is a stationary f(") with I(f(')) &gt; u*(e/2) ? IQ(r) - (e/2), where u* is the fixed point of the U associated with 1r. Thisf(") is (p, E)-optimal.

- (c) For any 7r*= (7rl , 7r2, ***) f(X*) = w(s, a, s') = IQ(s,a) (s') and xlsa denotes the p with the second day, when the first state and act are s, a, i.e. 1sa= (i1, 2, * * where

<!-- formula-not-decoded -->

If lr* is e-optimal, w(s, a, s') &lt; I(ir*) (s') + e for all s', so that I(7r*) &lt; irlq(r + f3I(lr*) + A3e) = ir1h, say. From the Lemma of Section 2, there is an f for which fh ? ir1h -for all s, so that, for the T corresponding to f, I(x*) ? T(I(ix*)) + f3E. By induction on n we obtain TnI(ax*) &gt; I(ax*) - E(3 + + f3n) . Letting n -* oo yields I(f *)) ? I(r*) - 03e/(1 -,). Since 7r* is e-optimal, f(') is e + [E/(1 - (3)] = e/(1 - 3)-optimal.

(d) For any so E S and any e &gt; 0, there is a stationary f(?) such that

<!-- formula-not-decoded -->

just choose f(X) (p, e) -optimal, where p is concentrated on so. Tau ? u for all a implies Tu ? u for all T and in particular for the T associated with f. Thus Tnu decreases to I(f(")) and I(f (?)) ? u. Then I(ir)(so) ? u(so) + e. Letting e -&gt; 0 completes the proof.

- (e) From (c), the hypothesis implies that there is a 1/n-optimal stationary plan fn(?) say. With r = (fJ, f2 X - * ), the fixed point u* of the U associa with ir is, from (a), the optimal return among ir-generated policies. In particular u* ? I(fn(?) so that u* &gt; I (r) for all ir, and u* is the optimal return. We have supa Tau* &gt; Uu* = u*. On the other hand, for any a E A,

<!-- formula-not-decoded -->

where (a, f(?)) is the Mlarkov policy (g, f, f, f, ** ) with g = a. Letting n -o yields Tau* ? u*. Thus u* satisfies the optimality equation.

- (f) If I(ir*) satisfies the optimality equation, we obtain fronm (d), with u = I(,r*), that 7r* is optimal. Conversely, if r* is optimal, the hypothesis of (e) is satisfied, so that u*, the optimal return, does satisfy the optimality equation.
- REMARKS. (d) is extremely useful in proving optimality; if u is known to be the return from a policy ir and ut satisfies Tau &lt; u for all u, (d) implies that iX is optimal. The criterion for optimality in (e), (f) was stated in general, without proof, by Bellman [1]. We do not know whether the optimal return always satisfies the optiniality equation or whether, evein under the hypothesis of (e), the (bounded) solution is unique.
7. Further results. If A is countable, with elemeints a1 , a2, , every Markov plan is r*-generated, where 7r* = (gl, g2 , - - *) and gn- an. Conversely, for any pure M\/Iarkov r = (f' , f2, *), the study of ir-generated plans can be reduced to the countable A case by interpreting act i in state s as the selection of fn (s). We prefer to keep the original A, aiid introduce the concept of essential countability as follows. Two acts a aild b will be called equivalent at state s if

<!-- formula-not-decoded -->

i.e. if Tau(s) = Tbu(s) for all u E M(S). For any Markov ix = (fi, f2 * A will be called essentially countable by ir if for every (s, a) there is an n for f.(s) is equivalent to a at s. A will be called essentially finite by ir if there is a partition of S into Borel sets Si, S2, *- * such that for every (s, a) with s e S, at least one of the acts fi(s), ... , fn(s) is equivalent to a at s.

THEOREM 7. (a) If A is essentially countable by r = (fi , f2 * * * ), the fixed point u* of the operator U associated with r is the optimal return. U is identical with the operator SUpa Ta , so that u* is the unique (bounded) solution of the optimality equation. For every E &gt; 0 there is an E-optimnal stationary plan.

- (b) If A is essentially finite by ir = (f' , f2 ... ),there is an optinal stationary plan.
- PROOF. (a) For any u, s, Tnu(8) - Tau(s), where a = fn(s). Thus Uu &lt; SUpa Tau. But for any a - A, Tau(s) = Tnu(S) for some n, so that Tau(s) &lt; Uu(s), and supa Tau(s) &lt; Uu(s). Thus the operators supa Ta , U are identical. Theorem 6 (d) then implies I(r) &lt; u* for all r. From Theorem 6 (a) there is a stationary f(?) with I(f(?)) &gt; u*-E. This f(?) is E-optimal.
- (b) If A is essentially finite, define Bn as the set of all s for which n is the smallest i with Tiu* (s) = supn Tnu* (s) (the sequence {TnU* (s) } contains only finitely nmany different numbers). Define f = fn on Bn , sO that Tu* = Uu*u*, where T is associated with f. Then u*, as the fixed point of T, is the return from f(?), and f(-) is optimal.

We conclude with the extension of the improvement routines given by Howard [5] and Eaton and Zadeh [4] for the case of finite S, A.

THEOREM 8. (a) (Howard improvement). If I(g, 7r) \_ I(r), then I(g(X)) &gt; I (g , ') &gt;\_ I(7r)

- (b) (Eaton-Zadeh improvement). For any f, g mapping S into A, define h f on I(f(o)) &gt; I(g(?)), h = g on I(g(?)) &gt; I(f(*)). Then I(h(-)) &gt; max (I(f(00)), I(g -)))
- PROOF. (a) If T is associated with g, we have TI(r) = I(g, ir) ? I(7r), so that

<!-- formula-not-decoded -->

- (b) (Proof by Ashok lMlaitra). If T1, T2, T are associated withf, g, h, we have,
- for any u,

<!-- formula-not-decoded -->

With u = max (I(f (X), I(g (?)) , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus Tu &gt; u so that I(h W?) &gt; u.

## REFERENCES

- [11 BELLMAN, RICHARD (1957). Dynamic Programming. Princeton Univ. Press.
- [21 BLACKWELL, ID. (1964). Memoryless strategies in finite stage dynamic programming. Ann. Math. Statist. 35 863-865.
- [3] DUBINS, L. E. and SAVAGE, L. J. (1963). How to gamble if you must (dittoed draft).
- [41 EATON, J. H. and ZADERt, L. A. (1961). Optimal pursuit strategies in discrete state probabilistic systems. J. Basic Engineering Ser. D 84 23-29.
- [51 HOWARD, RONALD A. (1960). Dynamic Programming and Markov Processes. Wiley, New York.
- [6] KARLIN, S. (1955). The structure of dynamic programming models. Naval Res. Logist. Quart. 2 285-294.
- [7] LOEVE, M. (1960). Probability Theory. Van Nostrand, Princeton.