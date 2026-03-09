## David Blackwell and Dynamic Programming

## Bill Sudderth University of Minnesota

## Statistics 169, Dynamic Programming

Blackwell taught a course on dynamic programming at U.C. Berkeley in the 1960's. It was taken by engineers, operations researchers, statisticians, and mathematicians among others. I took the course in 1965. It was a great course!

The course met once a week for about two hours.

<!-- image -->

## An example: Spend-or-Save

You begin with s 1 dollars, choose a 1 ∈ [0 , s 1 ] to spend on consumption, and save s 1 -a 1 .

You receive u ( a 1 ) in utility, and begin the next stage with cash

<!-- formula-not-decoded -->

Here Y 1 is your random income and has a given distribution. You then choose a 2 ∈ [0 , s 2 ], and so on.

Future stages are discounted at rate β ∈ (0 , 1), and you want to maximize the expectation of

<!-- formula-not-decoded -->

## Dynamic Programming (Markov Decision Theory)

Five ingredients:

S, A, r, q, β .

Begin at state s 1 ∈ S , select an action a 1 ∈ A , receive a reward r ( s 1 , a 1 ).

Move to a new state s 2 with distribution q ( ·| s 1 , a 1 ). Select a 2 ∈ A , receive β · r ( s 2 , a 2 ).

Move to s 3 with distribution q ( ·| s 2 , a 2 ), select a 3 ∈ A , receive β 2 · r ( s 3 , a 3 ). And so on.

Your total reward is the expected value of

<!-- formula-not-decoded -->

<!-- image -->

Three Conditions to make ∑ ∞ n =1 β n -1 r ( sn, an ) well-defined

Discounted Problems: r bounded, 0 ≤ β &lt; 1. Blackwell (1962,1965)

Positive Problems: r

≥ 0, β =1. Blackwell (1967)

Negative Problems: r ≤ 0, β =1. Strauch (1966)

## Plans and Rewards

A plan π selects each action an as a function of the history ( s 1 , a 1 , . . . , an -1 , sn ). The reward from π at the initial state s 1 = s is

<!-- formula-not-decoded -->

The optimal reward at s is

<!-- formula-not-decoded -->

Basic problems : Calculate the optimal reward function V ∗ ( · ) and find optimal or nearly optimal plans.

## Stationary Plans

A stationary plan is one that ignores the past when selecting an action.

Formally, a plan π is stationary if there is a function f : S ↦→ A such that π ( s 1 , a 1 , . . . , an -1 , sn ) = f ( sn ) for all ( s 1 , a 1 , . . . , an -1 , sn ).

Notation :

π = f ∞ .

Fundamental Question : Do optimal or nearly optimal stationary plans exist?

## Discrete Discounted Dynamic Programming

Theorem 1 (Blackwell, 1962) If S and A are finite and 0 ≤ β &lt; 1 , then there is an optimal stationary plan. Indeed, there is a stationary plan that is optimal for all β sufficiently close to 1.

A plan satisfying the final phrase is now called Blackwell optimal . (Hordijk and Yushkevich (2002))

## Blackwell Operators for Discounted Problems

Assume : S, A countable, r bounded, 0 ≤ β &lt; 1.

Let B be the Banach space of bounded functions x : S ↦→ R equipped with the supremum norm.

Let f : S ↦→ A and π = f ∞ . Define operators Tf and U for x ∈ B :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 2 The operators Tf and U are β -contractions on B . The fixed point of Tf is the reward function V ( π )( · ) for π = f ∞ , the fixed point of U is the optimal reward function V ∗ ( · ) .

## The Bellman Equation

For s ∈ S , V ∗ ( s ) = UV ∗ ( s ), or

<!-- formula-not-decoded -->

This equality is known as the Bellman equation or the optimality equation .

Let glyph[epsilon1] &gt; 0. For each s ∈ S , we can select f ( s ) ∈ A so that

<!-- formula-not-decoded -->

Blackwell showed that the reward function V ( π )( · ) for the stationary plan π = f ∞ satisfies:

<!-- formula-not-decoded -->

So good stationary plans exist.

## Measurable Dynamic Programming

The first formulation of dynamic programming in a general measure theoretic setting was given by Blackwell (1965). He assumed:

1. S and A are Borel subsets of some nice measurable space (say, a Euclidean space).
2. The reward function r ( s, a ) is Borel measurable.
3. The law of motion q ( ·| s, a ) is a regular conditional distribution.

Plans are required to select actions in a Borel measurable way.

## Measurability Problems

In his 1965 paper, Blackwell showed by example that for a Borel measurable dynamic programming problem:

## The optimal reward function V ∗ ( · ) need not be Borel measurable and good Borel measurable plans need not exist.

This led to work by a number of mathematicians including R. Strauch, D. Freedman, M. Orkin, D. Bertsekas, S. Shreve, and Blackwell himself. It follows from their work that for a Borel problem:

## The optimal reward function V ∗ ( · ) is universally measurable and that there do exist good universally measurable plans.

## Blackwell's (1965) Example

Let S = A = [0 , 1]. The state of the system remains fixed: q ( s | s, a ) = 1 for all s, a . The reward function is

<!-- formula-not-decoded -->

where B is a Borel subset of S × A such that the projection

<!-- formula-not-decoded -->

is not Borel. The optimal reward at s is

<!-- formula-not-decoded -->

The optimal reward is not Borel measurable and there are no good Borel measurable plans.

## Positive Dynamic Programming

Assume : β =1, r ( s, a ) ≥ 0 for all ( s, a ), and the optimal reward function V ∗ ( s ) &lt; ∞ for all s .

Theorem 3 (Blackwell 1967). For 0 &lt; glyph[epsilon1] &lt; 1 and P a probability measure on S such that ∫ V ∗ dP &lt; ∞ , there exists a a stationary plan π such that

<!-- formula-not-decoded -->

Blackwell showed by example that there need not exist a stationary π such that V ( π )( s ) ≥ V ∗ ( s ) -glyph[epsilon1] for all s .

Theorem 4 (Ornstein 1969, Frid 1972) Given 0 &lt; glyph[epsilon1] &lt; 1 and a probability measure P on S , there exists a stationary plan π with payoff V ( π )( s ) at s such that

<!-- formula-not-decoded -->

Question: Is there a stationary plan π

such that

<!-- formula-not-decoded -->

Answer: Not in general. (Blackwell and Ramakrishnan (1988))

## Negative Dynamic Programming

$$Assume:$$

$$β =1 , r ( s, a ) ≤ 0 for all ( s, a ).$$

A simple example of Dubins and Savage (1965) shows there need not exist good stationary plans even when S has only three elements and A is countable.

The fundamental paper is by Strauch (1966), based on his PhD thesis under Blackwell. There do exist optimal stationary plans if A is finite.

## Question: Optimal Plan ⇒ Stationary Optimal Plan?

Yes for discounted or negative problems . If π is optimal, then so is f ∞ where f ( s ) is the first action for π when the initial state is s .

Theorem 5 (Ornstein 1969, Blackwell 1970, Orkin 1974) If there is an optimal plan for a positive problem, then, for each probability P on S , there exists a stationary plan π which is optimal with P - probability one.

Open question:

Can the set of probability zero be eliminated?

## Convergent Dynamic Programming

Assume:

β =1 and that

<!-- formula-not-decoded -->

for all s ∈ S .

Many results, such as the Bellman equation, still hold in this general setting (Feinberg, 2002). For A compact, Schal (1983) proved that good stationary strategies exist.

## Applications

Blackwell's fundamental work on dynamic programming led to applications in many areas including statistics,finance, economics, communication networks, water resources management, and even mathematics itself.

For information about applications and recent developments, see the Handbook of Markov Decision Processes (2002) edited by E. Feinberg and A. Shwartz.

## Blackwell's papers on dynamic programming

On the functional equation of dynamic programming (1961). J. Math. Anal. &amp; Appl. 2 273-276.

Discrete dynamic programming (1962). Ann. Math. Statist. 33 719-726.

Probability bounds via dynamic programming (1964). AMS Proc. Symp. Appl. Math. v. XVI 277-280.

Memoryless strategies in finite-stage dynamic programming (1964). Ann. Math. Statist. 35 863-865.

Discounted dynamic programming (1965). Ann. Math. Statist. 36 226-235.

Positive dynamic programming (1967). Proc. 5th Berkeley Symp. 415-418.

On stationary policies (1970). J. Royal Stat. Soc, A 133 33-37.

The optimal reward operator in dynamic programming (1974). Ann. Prob. 2 926-941 (with D. Freedman and M. Orkin).

The stochastic processes of Borel gambling and dynamic programming (1976). Ann. Statist. 4 370-374.

Stationary plans need not be uniformly adequate for leavable, Borel gambling problems (1988). Proc. AMS 102 1024-1027 (with S. Ramakrishnan).