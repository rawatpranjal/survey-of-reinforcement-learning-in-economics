## The Complexity of Computing a Nash Equilibrium

Constantinos Daskalakis ∗

Paul W. Goldberg † June 4, 2008

## Abstract

In 1951, John F. Nash proved that every game has a Nash equilibrium [43]. His proof is non-constructive, relying on Brouwer's fixed point theorem, thus leaving open the questions: Is there a polynomial-time algorithm for computing Nash equilibria? And is this reliance on Brouwer inherent? Many algorithms have since been proposed for finding Nash equilibria, but none known to run in polynomial time. In 1991 the complexity class PPAD, for which Brouwer's problem is complete, was introduced [48], motivated largely by the classification problem for Nash equilibria; but whether the Nash problem is complete for this class remained open. In this paper we resolve these questions: We show that finding a Nash equilibrium in three-player games is indeed PPAD-complete; and we do so by a reduction from Brouwer's problem, thus establishing that the two problems are computationally equivalent. Our reduction simulates a (stylized) Brouwer function by a graphical game [33], relying on 'gadgets,' graphical games performing various arithmetic and logical operations. We then show how to simulate this graphical game by a three-player game, where each of the three players is essentially a color class in a coloring of the underlying graph. Subsequent work [8] established, by improving our construction, that even two-player games are PPAD-complete; here we show that this result follows easily from our proof.

## Categories and Subject Descriptors

F. 2. 0 [Analysis of Algorithms and Problem Complexity]: General

## General Terms

Theory, Algorithms, Economics

## Keywords

Complexity, Nash Equilibrium, PPAD-Completeness, Game Theory

∗ Computer Science Division, University of California at Berkeley. Research supported by NSF ITR Grants CCR0121555 and CCF-0515259 and a grant from Microsoft Research. email: costis@cs.berkeley.edu

† Department of Computer Science, University of Liverpool. Research supported by the EPSRC grant GR/T07343/01 'Algorithmics of Network-sharing Games'. This work was begun while the author was visiting UC Berkeley. email: P.W.Goldberg@liverpool.ac.uk

‡ Computer Science Division, University of California at Berkeley. Research supported by NSF ITR grants CCR0121555 and CCF-0515259 and a grant from Microsoft Research. email: christos@cs.berkeley.edu

Christos H. Papadimitriou ‡

## Contents

| 1   | Introduction                          | Introduction                                                |   3 |
|-----|---------------------------------------|-------------------------------------------------------------|-----|
| 2   | Background                            | Background                                                  |   7 |
|     | 2.1                                   | Basic Definitions from Game Theory . . . . . . . . . .      |   7 |
|     | 2.2                                   | Related Work on Computing Equilibria . . . . . . . .        |   8 |
| 3   | The Class PPAD                        | The Class PPAD                                              |  10 |
|     | 3.1                                   | Total Search Problems . . . . . . . . . . . . . . . . . .   |  10 |
|     | 3.2                                   | Computing a Nash Equilibrium is in PPAD . . . . . .         |  11 |
|     | 3.3                                   | The Brouwer Problem . . . . . . . . . . . . . . . . .       |  18 |
| 4   | Reductions Among Equilibrium Problems | Reductions Among Equilibrium Problems                       |  23 |
|     | 4.1                                   | Preliminaries: Game Gadgets . . . . . . . . . . . . . .     |  23 |
|     | 4.2                                   | Reducing Graphical Games to Normal Form Games .             |  27 |
|     | 4.3                                   | Reducing Normal Form Games to Graphical Games .             |  31 |
|     | 4.4                                   | Combining the Reductions . . . . . . . . . . . . . . . .    |  36 |
|     | 4.5                                   | Reducing to Three Players . . . . . . . . . . . . . . . .   |  37 |
|     | 4.6                                   | Preservation of Approximate equilibria . . . . . . . . .    |  42 |
|     | 4.7                                   | Reductions Between Different Notions of Approximation       |  51 |
| 5   | The Main Reduction                    | The Main Reduction                                          |  54 |
| 6   | Further Results and Open Problems     | Further Results and Open Problems                           |  62 |
|     | 6.1                                   | Two Players . . . . . . . . . . . . . . . . . . . . . . . . |  62 |
|     | 6.2                                   | Approximate Nash Equilibria . . . . . . . . . . . . . .     |  64 |
|     | 6.3                                   | Nash Equilibria in Graphical Games . . . . . . . . . .      |  65 |
|     | 6.4                                   | Special Cases . . . . . . . . . . . . . . . . . . . . . . . |  65 |
|     | 6.5                                   | Further Applications of our Techniques . . . . . . . . .    |  65 |

## 1 Introduction

Game Theory is one of the most important and vibrant mathematical fields established during the 20th century. In 1928, John von Neumann, extending work by Borel, showed that any two-person zero-sum game has an equilibrium - in fact, a min-max pair of randomized strategies [44]. Two decades later it was understood that this is tantamount to Linear Programming duality [14], and thus (as it was established another three decades hence [34]) computationally tractable. However, it became clear with the publication of the seminal book [45] by von Neumann and Morgenstern that this two-player, zero-sum case is too specialized; for the more general and important non-zero sum and multi-player games no existence theorem was known.

In 1951, Nash showed that every game has an equilibrium in mixed strategies, hence called Nash equilibrium [43]. His argument for proving this powerful and momentous result relies on another famous and consequential result of the early 20th century, Brouwer's fixed point theorem [35]. The original proof of that result is notoriously nonconstructive (Brouwer's preoccupation with constructive Mathematics and Intuitionism notwithstanding); its modern combinatorial proof (based on Sperner's Lemma, see, e.g., [48]) does suggest an algorithm for the problem of finding an approximate Brouwer fixed point (and therefore for finding a Nash equilibrium) - albeit one of exponential complexity. In fact, it can be shown that any 'natural' algorithm for Brouwer's problem (roughly, treating the Brouwer function as a black box, a property shared by all known algorithms for the problem) must be exponential [31]. Over the past half century there has been a great variety of other algorithmic approaches to the problem of finding a Nash equilibrium (see Section 2.2); unfortunately, none of these algorithms is known to run in polynomial time. Whether a Nash equilibrium in a given game can be found in polynomial time had remained an important open question.

Such an efficient algorithm would have many practical applications; however, the true importance of this question is conceptual. The Nash equilibrium is a proposed model and prediction of social behavior, and Nash's theorem greatly enhances its plausibility. This credibility, however, is seriously undermined by the absence of an efficient algorithm. It is doubtful that groups of rational players are more powerful than computers - and it would be remarkable, and potentially very useful, if they were. To put it bluntly, 'if your laptop can't find it, then, probably, neither can the market.' Hence, whether an efficient algorithm for finding Nash equilibria exists is an important question in Game Theory, the field for which the Nash equilibrium is perhaps the most central concept.

Besides Game Theory, the 20th century saw the development of another great mathematical field, which also captured the century's zeitgeist and has had tremendous growth and impact: Computational Complexity. However, the mainstream concepts and techniques developed by complexity theorists for classifying computational problems according to their difficulty - chief among them NP-completeness - are not directly applicable for fathoming the complexity of the problem of finding Nash equilibria, exactly because of Nash's Theorem: Since a Nash equilibrium is always guaranteed to exist, NP-completeness does not seem useful in exploring the complexity of finding one. NP-complete problems seem to draw much of their difficulty from the possibility that a solution may not exist. How would a reduction from satisfiability to Nash (the problem of finding a Nash equilibrium) look like? Any attempt to define such a reduction quickly leads to NP = coNP.

Motivated mainly by this open question regarding Nash equilibria, Meggido and Papadimitriou [42] defined in the 1980s the complexity class TFNP (for 'NP total functions'), consisting exactly of all search problems in NP for which every instance is guaranteed to have a solution. Nash of course belongs there, and so do many other important and natural problems, finitary versions of Brouwer's problem included. But here there is a difficulty of a different sort: TFNP is a 'semantic

class' [47], meaning that there is no easy way of recognizing nondeterministic Turing machines which define problems in TFNP -in fact the problem is undecidable; such classes are known to be devoid of complete problems.

To capture the complexity of Nash , and other important problems in TFNP, another step is needed: One has to group together into subclasses of TFNP total functions whose proofs of totality are similar. Most of these proofs work by essentially constructing an exponentially large graph on the solution space (with edges that are computed by some algorithm), and then applying a simple graph-theoretic lemma establishing the existence of a particular kind of node. The node whose existence is guaranteed by the lemma is the sought solution of the given instance. Interestingly, essentially all known problems in TFNP can be shown total by one of the following arguments:

- In any dag there must be a sink. The corresponding class, PLS for 'polynomial local search' had already been defined in [32], and contains many important complete problems.
- In any directed graph with outdegree one, and with one node with indegree zero, there must be a node with indegree at least two. The corresponding class is PPP (for 'polynomial pigeonhole principle').
- In any undirected graph with one odd-degree node, there must be another odd-degree node. This defines a class called PPA for 'polynomial parity argument' [48], containing many important combinatorial problems (unfortunately none of them are known to be complete).
- In any directed graph with one unbalanced node (node with outdegree different from its indegree), there must be another unbalanced node. The corresponding class is called PPAD for 'polynomial parity argument for directed graphs,' and it contains Nash , Brouwer , and Borsuk-Ulam (finding approximate fixed points of the kind guaranteed by Brouwer's Theorem and the Borsuk-Ulam Theorem, respectively, see [48]). The latter two were among the problems proven PPAD-complete in [48]. Unfortunately, Nash -the one problem which had motivated this line of research - was not shown PPAD-complete; it was conjectured that it is.

In this paper we show that Nash is PPAD-complete, thus answering the open questions discussed above. We show that this holds even for games with three players. In another result (which is a crucial component of our proof) we show that the same is true for graphical games . Thus, a polynomial-time algorithm for these problems would imply a polynomial algorithm for, e.g., computing Brouwer fixed points, despite the exponential lower bounds for large classes of algorithms [31], and the relativizations in [2] - oracles for which PPAD has no polynomial-time algorithm.

Our proof gives an affirmative answer to another important question arising from Nash's Theorem, namely, whether the reliance of its proof on Brouwer's fixed point theorem is inherent. Our proof is essentially a reduction in the opposite direction to Nash's: An appropriately discretized and stylized PPAD-complete version of Brouwer's fixed point problem in 3 dimensions is reduced to Nash.

The structure of the reduction is the following: We represent a point in the three-dimensional unit cube by three players each of which has two strategies. Thus, every combination of mixed strategies for these players corresponds naturally to a point in the cube. Now, suppose that we are given a function from the cube to itself represented as a circuit. We construct a graphical game in which the best responses of the three players representing a point in the cube implement the given function, so that the Nash equilibria of the game must correspond to Brouwer fixed points. This is done by decoding the coordinates of the point in order to find their binary representation

(inputs to the circuit), and then simulating the circuit that represents the Brouwer function by a graphical game - an important alternative form of games defined in [33], see Section 2.1. This part of the construction relies on certain 'gadgets,' small graphical games acting as arithmetical gates and comparators. The graphical game thus 'computes' (in the sense of a mixed strategy over two strategies representing a real number) the value of the circuit at the point represented by the mixed strategies of the original three players, and then induces the three players to add appropriate increments to their mixed strategy. This establishes a one-to-one correspondence between Brouwer fixed points of the given function and Nash equilibria of the graphical game and shows that Nash for graphical games is PPAD-complete.

One difficulty in this part of the reduction is related to brittle comparators. Our comparator gadget sets its output to 0 if the input players play mixed strategies x , y that satisfy x &lt; y , to 1 if x &gt; y , and to anything if x = y ; moreover, it is not hard to see that no 'robust' comparator gadget is possible, one that outputs a specific fixed value if the input is x = y . This in turn implies that no robust decoder from real to binary can be constructed; decoding will always be flaky for a non-empty subset of the unit cube and, at that set, arbitrary values can be output by the decoder. On the other hand, real to binary decoding would be very handy since the circuit representing the given Brouwer function should be simulated in binary arithmetic. We take care of this difficulty by computing the Brouwer function on a 'microlattice' around the point of interest and averaging the results, thus smoothing out any effects from boundaries of measure zero.

To continue to our main result for three-player normal form games, we establish certain reductions between equilibrium problems. In particular, we show by reductions that the following three problems are equivalent:

- Nash for r -player (normal form) games, for any r &gt; 3.
- Nash for three-player games.
- Nash for graphical games with two strategies per player and maximum degree three (that is, of the exact type used in the simulation of Brouwer functions).

Thus, all these problems and their generalizations are PPAD-complete (since the third one was already shown to be PPAD-complete).

Our results leave open the question of Nash for two-player games. This case had been thought to be a little easier, since linear programming-like techniques come into play and solutions consisting of rational numbers are guaranteed to exist [38]; on the contrary, as exhibited in Nash's original paper, there are three-player games with only irrational equilibria. In the precursors of the current paper [30, 16, 19], it was conjectured that there is a polynomial algorithm for two-player Nash . Surprisingly, a few months after our proof was circulated, Chen and Deng [8] came up with a proof establishing that this problem is PPAD-complete as well. In the last section of the present paper we show how this result can be obtained by a simple modification of our proof.

The structure of the paper is as follows. In Section 2, we provide some background on game theory and survey previous work regarding the computation of equilibria. In Section 3, we review the complexity theory of total functions, we define the class PPAD which is central in our paper, and we describe a canonical version of the Brouwer Fixed Point computation problem which is PPADcomplete and will be the starting point for our main result. In Section 4, we present the game-gadget machinery needed for our proof of the main result and establish the computational equivalence of different Nash equilibrium computation problems; in particular, we describe a polynomial reduction from the problem of computing a Nash equilibrium in a normal form game of any constant number of players or a graphical game of any constant degree to that of computing a Nash equilibrium of a

three player normal form game. Finally, in Section 5 we present our main result that computing a Nash equilibrium of a 3-player normal form game is PPAD-hard. Section 6 contains some discussion of the result and future research directions.

## 2 Background

## 2.1 Basic Definitions from Game Theory

A game in normal form has r ≥ 2 players, 1 , . . . , r , and for each player p ≤ r a finite set S p of pure strategies. The set S of pure strategy profiles is the Cartesian product of the S p 's. We denote the set of pure strategy profiles of all players other than p by S -p . Also, for a subset T of the players we denote by S T the set of pure strategy profiles of the players in T . Finally, for each p and s ∈ S we have a payoff or utility u p s ≥ 0 - also occasionally denoted u p js for j ∈ S p and s ∈ S -p . We refer to the set { u p s } s ∈ S as the payoff table of player p . Also, for notational convenience and unless otherwise specified, we will denote by [ t ] the set { 1 , . . . , t } , for all t ∈ N .

A mixed strategy for player p is a distribution on S p , that is, real numbers x p j ≥ 0 for each strategy j ∈ S p such that ∑ j ∈ S p x p j = 1. A set of r mixed strategies { x p j } j ∈ S p , p ∈ [ r ], is called a (mixed) Nash equilibrium if, for each p , ∑ s ∈ S u p s x s is maximized over all mixed strategies of p -where for a strategy profile s = ( s 1 , . . . , s r ) ∈ S , we denote by x s the product x 1 s 1 · x 2 s 2 · · · x r s r . That is, a Nash equilibrium is a set of mixed strategies from which no player has a unilateral incentive to deviate. It is well-known (see, e.g., [46]) that the following is an equivalent condition for a set of mixed strategies to be a Nash equilibrium:

<!-- formula-not-decoded -->

/negationslash

The summation ∑ s ∈ S -p u p js x s in the above equation is the expected utility of player p if p plays pure strategy j ∈ S p and the other players use the mixed strategies { x q j } j ∈ S q , q = p . Nash's theorem [43] asserts that every normal form game has a Nash equilibrium .

We next turn to approximate notions of equilibrium. We say that a set of mixed strategies x is an /epsilon1 -approximately well supported Nash equilibrium , or /epsilon1 -Nash equilibrium for short, if the following holds:

<!-- formula-not-decoded -->

Condition (2) relaxes (1) in that it allows a strategy to have positive probability in the presence of another strategy whose expected payoff is better by at most /epsilon1 .

This is the notion of approximate Nash equilibrium that we use in this paper. There is an alternative, and arguably more natural, notion, called /epsilon1 -approximate Nash equilibrium [40], in which the expected utility of each player is required to be within /epsilon1 of the optimum response to the other players' strategies. This notion is less restrictive than that of an approximately well supported one. More precisely, for any /epsilon1 , an /epsilon1 -Nash equilibrium is also an /epsilon1 -approximate Nash equilibrium, whereas the opposite need not be true. Nevertheless, the following lemma, proved in Section 4.7, establishes that the two concepts are computationally related (a weaker version of this fact was pointed out in [9]).

Lemma 1 Given an /epsilon1 -approximate Nash equilibrium { x p j } j,p of a game G we can compute in polynomial time a √ /epsilon1 · ( √ /epsilon1 +1+4( r -1) u max ) -approximately well supported Nash equilibrium { ˆ x p j } j,p , where r is the number of players and u max is the maximum entry in the payoff tables of G .

In the sequel we shall focus on the notion of approximately well-supported Nash equilibrium, but all our results will also hold for the notion of approximate Nash equilibrium. Notice that Nash's theorem ensures the existence of an /epsilon1 -Nash equilibrium -and hence of an /epsilon1 -approximate

Nash equilibrium- for every /epsilon1 ≥ 0; in particular, for every /epsilon1 there exists an /epsilon1 -Nash equilibrium whose probabilities are integer multiples of /epsilon1/ (2 r × u maxsum), where u maxsum is the maximum, over all players p , of the sum of all entries in the payoff table of p . This can be established by rounding a Nash equilibrium { x p j } j,p to a nearby (in total variation distance) set of mixed strategies { ˆ x p j } j,p all the entries of which are integer multiples of /epsilon1/ (2 r × u maxsum). Note, however, that a /epsilon1 -Nash equilibrium may not be close to an exact Nash equilibrium; see [25] for much more on this important distinction.

A game in normal form requires r | S | numbers for its description, an amount of information that is exponential in the number of players. A graphical game [33] is defined in terms of an undirected graph G = ( V, E ) together with a set of strategies S v for each v ∈ V . We denote by N ( v ) the set consisting of v and v 's neighbors in G , and by S N ( v ) the set of all |N ( v ) | -tuples of strategies, one from each vertex in N ( v ). In a graphical game, the utility of a vertex v ∈ V only depends on the strategies of the vertices in N ( v ) so it can be represented by just | S N ( v ) | numbers. In other words, a graphical game is a succinct representation of a multiplayer game, advantageous when it so happens that the utility of each player only depends on a few other players. A generalization of graphical games are the directed graphical games , where G is directed and N ( v ) consists of v and the predecessors of v . The two notions are almost identical; of course, the directed graphical games are more general than the undirected ones, but any directed graphical game can be represented, albeit less concisely, as an undirected game whose graph is the same except with no direction on the edges. In the remaining of the paper, we will not be very careful in distinguishing the two notions; our results will apply to both. The following is a useful definition.

Definition 1 Suppose that GG is a graphical game with underlying graph G = ( V, E ) . The affectsgraph G ′ = ( V, E ′ ) of GG is a directed graph with edge ( v 1 , v 2 ) ∈ E ′ if the payoff to v 2 depends on the action of v 1 , that is, the payoff to v 2 is a non-constant function of the action of v 1 .

In the above definition, an edge ( v 1 , v 2 ) in G ′ represents the relationship ' v 1 affects v 2 '. Notice that if ( v 1 , v 2 ) ∈ E ′ then { v 1 , v 2 } ∈ E , but the opposite need not be true -it could very well be that some vertex v 2 is affected by another vertex v 1 , but vertex v 1 is not affected by v 2 .

Since graphical games are representations of multi-player games, it follows by Nash's theorem that every graphical game has a mixed Nash equilibrium. It can be checked that a set of mixed strategies { x v j } j ∈ S v , v ∈ V , is a mixed Nash equilibrium if and only if

<!-- formula-not-decoded -->

Similarly the condition for an approximately well supported Nash equilibrium can be derived.

## 2.2 Related Work on Computing Equilibria

Many papers in the economic, optimization, and computer science literature over the past 50 years study the computation of Nash equilibria. A celebrated algorithm for computing equilibria in 2player games, which appears to be efficient in practice, is the Lemke-Howson algorithm [38]. The algorithm can be generalized to multi-player games, see, e.g., the work of Rosenm¨ uller [51] and Wilson [57], albeit with some loss of efficiency. It was recently shown to be exponential in the worst case [53]. Other algorithms are based on computing approximate fixed points, most notably algorithms that walk on simplicial subdivisions of the space where the equilibria lie [54, 27, 36, 37, 23]. None of these algorithms is known to be polynomial-time.

Lipton and Markakis [39] study the algebraic properties of Nash equilibria, and point out that standard quantifier elimination algorithms can be used to solve them, but these are not polynomialtime in general. Papadimitriou and Roughgarden [50] show that, in the case of symmetric games, quantifier elimination results in polynomial algorithms for a broad range of parameters. Lipton, Markakis and Mehta [40] show that, if we only require an /epsilon1 -approximate Nash equilibrium, then a subexponential algorithm is possible. If the Nash equilibria sought are required to have any special properties, for example optimize total utility, the problem typically becomes NP-complete [29, 13]. In addition to our work, as communicated in [30, 16, 19], other researchers (see, e.g., [5, 1, 11, 55]) have explored reductions between alternative types of games.

In particular, the reductions by Bubelis [5] in the 1970s comprise a remarkable early precursor of our work; it is astonishing that these important results had not been pursued for three decades. Bubelis established that the Nash equilibrium problem for 3 players captures the computational complexity of the same problem with any number of players. In Section 4 we show the same result in an indirect way, via the Nash equilibrium problem for graphical games - a connection that is crucial for our PPAD-completeness reduction. Bubelis also demonstrated in [5] that any algebraic number can be the basis of a Nash equilibrium, something that follows easily from our results (Theorem 14).

Etessami and Yannakakis studied in [25] the problem of computing a Nash equilibrium exactly (a problem that is well-motivated in the context of stochastic games) and came up with an interesting characterization of its complexity (considerably higher than PPAD), along with that of several other problems. In Section 6.5, we mention certain interesting results at the interface of [25]'s approach with ours.

## 3 The Class PPAD

## 3.1 Total Search Problems

A search problem S is a set of inputs I S ⊆ Σ ∗ on some alphabet Σ such that for each x ∈ I S there is an associated set of solutions S x ⊆ Σ | x | k for some integer k , such that for each x ∈ I S and y ∈ Σ | x | k whether y ∈ S x is decidable in polynomial time. Notice that this is precisely NP with an added emphasis on finding a witness.

For example, let us define r -Nash to be the search problem S in which each x ∈ I S is an r -player game in normal form together with a binary integer A (the accuracy specification ), and S x is the set of 1 A -Nash equilibria of the game (where the probabilities are rational numbers of bounded size as discussed). Similarly, d -graphical Nash is the search problem with inputs the set of all graphical games with degree at most d , plus an accuracy specification A , and solutions the set of all 1 A -Nash equilibria. (For r &gt; 2 it is important to specify the problem in terms of a search for approximate Nash equilibrium - exact solutions may need to be high-degree algebraic numbers, raising the question of how to represent them as bit strings.)

/negationslash

A search problem is total if S x = ∅ for all x ∈ I S . For example, Nash's 1951 theorem [43] implies that r -Nash is total. Obviously, the same is true for d -graphical Nash . The set of all total search problems is denoted TFNP. A polynomial-time reduction from total search problem S to total search problem T is a pair f, g of polynomial-time computable functions such that, for every input x of S , f ( x ) is an input of T , and furthermore for every y ∈ T f ( x ) , g ( y ) ∈ S x .

TFNP is what in Complexity is sometimes called a 'semantic' class [47], i.e., it has no generic complete problem. Therefore, the complexity of total functions is typically explored via 'syntactic' subclasses of TFNP, such as PLS [32], PPP, PPA and PPAD [48]. In this paper we focus on PPAD.

PPAD can be defined in many ways. As mentioned in the introduction, it is, informally, the set of all total functions whose totality is established by invoking the following simple lemma on a graph whose vertex set is the solution space of the instance:

In any directed graph with one unbalanced node (node with outdegree different from its indegree), there is another unbalanced node.

This general principle can be specialized, without loss of generality or computational power, to the case in which every node has both indegree and outdegree at most one. In this case the lemma becomes:

In any directed graph in which all vertices have indegree and outdegree at most one, if there is a source (a node with indegree zero), then there must be a sink (a node with outdegree zero).

Formally, we shall define PPAD as the class of all total search problems polynomial-time reducible to the following problem:

end of the line: Given two circuits S and P , each with n input bits and n output bits, such that P (0 n ) = 0 n = S (0 n ) , find an input x ∈ { 0 , 1 } n such that P ( S ( x )) = x or S ( P ( x )) = x = 0 n .

/negationslash

/negationslash

/negationslash

/negationslash

Intuitively, end of the line creates a directed graph G S,P with vertex set { 0 , 1 } n and an edge from x to y whenever both y = S ( x ) and x = P ( y ); S and P stand for 'successor candidate' and 'predecessor candidate'. All vertices in G S,P have indegree and outdegree at most one, and there is at least one source, namely 0 n , so there must be a sink. We seek either a sink, or a source other than 0 n . Notice that in this problem a sink or a source other than 0 n is sought; if we insist on a sink, another complexity class called PPADS, apparently larger than PPAD, results.

The other important classes PLS, PPP and PPA, and others, are defined in a similar fashion based on other elementary properties of finite graphs. These classes are of no relevance to our analysis so their definition will be skipped; the interested reader is referred to [48].

A search problem S in PPAD is called PPAD -complete if all problems in PPAD reduce to it. Obviously, end of the line is PPAD-complete; furthermore, it was shown in [48] that several problems related to topological fixed points and their combinatorial underpinnings are PPAD-complete: Brouwer, Sperner, Borsuk-Ulam , Tucker. Our main result in this paper (Theorem 12) states that so are the problems 3Nash and 3graphical Nash .

## 3.2 Computing a Nash Equilibrium is in PPAD

We establish that computing an approximate Nash equilibrium in an r -player game is in PPAD. The r = 2 case was shown in [48].

Theorem 1 r -Nash is in PPAD , for r ≥ 2 .

Proof. We reduce r -Nash to end of the line . Note that Nash's original proof [43] utilizes Brouwer's fixed point theorem - it is essentially a reduction from the problem of finding a Nash equilibrium to that of finding a Brouwer fixed point of a continuous function; the latter problem can be reduced, under certain continuity conditions, to end of the line , and is therefore in PPAD. The, rather elaborate, proof below makes this simple intuition precise.

/negationslash

Let ∆ n = { x ∈ R n + | ∑ n k =1 x k = 1 } be the ( n -1)-dimensional unit simplex. Then the space of mixed strategy profiles of the game is ∆ r n := × r p =1 ∆ n . For notational convenience we embed ∆ r n in R n · r and we represent elements of ∆ r n as vectors in R n · r . That is, if ( x 1 , x 2 , . . . , x r ) ∈ ∆ r n is a mixed strategy profile of the game, we identify this strategy profile with a vector x = ( x 1 ; x 2 ; . . . ; x r ) ∈ R n · r resulting from the concatenation of the mixed strategies. For p ∈ [ r ] and j ∈ [ n ] we denote by x ( p, j ) the (( p -1) n + j )-th coordinate of x , that is x ( p, j ) := x ( p -1) n + j .

Let G be a normal form game with r players, 1 , . . . , r , and strategy sets S p = [ n ], for all p ∈ [ r ], and let { u p s : p ∈ [ r ] , s ∈ S } be the utilities of the players. Also let /epsilon1 &lt; 1. In time polynomial in |G| + log(1 //epsilon1 ), we will specify two circuits S and P each with N = poly ( |G| , log(1 //epsilon1 )) input and output bits and P (0 N ) = 0 N = S (0 N ), so that, given any solution to end of the line on input S , P , one can construct in polynomial time an /epsilon1 -approximate Nash equilibrium of G . This is enough for reducing r -Nash to end of the line by virtue of Lemma 1. Our construction of S , P builds heavily upon the simplicial approximation algorithm of Laan and Talman [37] for computing fixed points of continuous functions from the product space of unit simplices to itself.

We are about to describe our reduction from finding an /epsilon1 -approximate Nash equilibrium to end of the line . The nodes of the end of the line graph will correspond to the simplices of a triangulation of ∆ r n which we describe next.

Triangulation of the Product Space of Unit Simplices. For some d , to be specified later, we describe the triangulation of ∆ r n induced by the regular grid of size d . For this purpose, let us denote by ∆ n ( d ) the set of points of ∆ n induced by the grid of size d , i.e.

<!-- formula-not-decoded -->

and similarly define ∆ r n ( d ) = × r p =1 ∆ n ( d ). Moreover, let us define the block diagonal matrix Q by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where, for all p ∈ [ r ], Q p is the n × n matrix defined by

Let us denote by q ( p, j ) the (( p -1) n + j )-th column of Q . It is clear that adding q ( p, j ) T /d to a mixed strategy profile corresponds to shifting probability mass of 1 /d from strategy j of player p to strategy ( j mod n ) + 1 of player p .

For all p ∈ [ r ] and k ∈ [ n ], let us define a set of indices I p,k as I p,k := { ( p, j ) } j ≤ k . Also, let us define a collection T of sets of indices as follows

Suppose, now, that q 0 is a mixed strategy profile in which every player plays strategy 1 with probability 1, that is q 0 ( p, 1) = 1, for all p ∈ [ r ], and for T ∈ T define the set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Defining T ∗ := ∪ p ∈ [ r ] I p,n -1 , it is not hard to verify that

<!-- formula-not-decoded -->

Moreover, if, for T ∈ T , we define B ( T ) := A ( T ) \ ∪ T ′ ∈T ,T ′ ⊂ T A ( T ′ ), the collection { B ( T ) } T ∈T partitions the set ∆ r n .

To define the triangulation of ∆ r n let us fix some set T ∈ T , some permutation π : [ | T | ] → T of the elements of T , and some x 0 ∈ A ( T ) ∩ ∆ r n ( d ). Let us then denote by σ ( x 0 , π ) the | T | -simplex which is the convex hull of the points x 0 , . . . , x | T | defined as follows

<!-- formula-not-decoded -->

The following lemmas, whose proof can be found in [37], describe the triangulation of ∆ r n . We define A ( T, d ) := A ( T ) ∩ ∆ r n ( d ), we denote by P T the set of all permutations π : [ | T | ] → T , and we set

<!-- formula-not-decoded -->

Lemma 2 ([37]) For all T ∈ T , the collection of | T | -simplices Σ T triangulates A ( T ) .

Corollary 1 ([37]) ∆ r n is triangulated by the collection of simplices Σ T ∗ .

The Vertices of the end of the line Graph. The vertices of the graph in our construction will correspond to the elements of the set

<!-- formula-not-decoded -->

Let us encode the elements of Σ with strings { 0 , 1 } N ; choosing N polynomial in |G| , the description size of G , and log d is sufficient.

We proceed to define the edges of the end of the line graph in terms of a labeling of the points of the set ∆ r n ( d ), which we describe next.

/negationslash

Labeling Rule. Recall the function f : ∆ r n → ∆ r n defined by Nash to establish the existence of an equilibrium [43]. To describe f , let U p j ( x ) := ∑ s ∈ S -p u p js x s be the expected utility of player p , if p plays pure strategy j ∈ [ n ] and the other players use the mixed strategies { x q j } j ∈ [ n ] , q = p ; let also U p ( x ) := ∑ s ∈ S u p s x s be the expected utility of player p if every player q ∈ [ r ] uses mixed strategy { x q j } j ∈ [ n ] . Then, the function f is described as follows:

<!-- formula-not-decoded -->

where, for each p ∈ [ r ], j ∈ [ n ],

<!-- formula-not-decoded -->

It is not hard to see that f is continuous, and that f ( x ) can be computed in time polynomial in the binary encoding size of x and G . Moreover, it can be verified that any point x ∈ ∆ r n such that f ( x ) = x is a Nash equilibrium [43]. The following lemma establishes that f is λ -Lipschitz for λ := [1 + 2 U max rn ( n +1)], where U max is the maximum entry in the payoff tables of the game.

Lemma 3 For all x, x ′ ∈ ∆ r n ⊆ R n · r such that || x -x ′ || ∞ ≤ δ ,

<!-- formula-not-decoded -->

Proof. We use the following bound shown in Section 4.6, Lemma 14.

Lemma 4 For any game G , for all p ≤ r , j ∈ S p ,

<!-- formula-not-decoded -->

/negationslash

It follows that for all p ∈ [ r ], j ∈ [ n ],

<!-- formula-not-decoded -->

Denoting B p j ( x ) := max (0 , U p j ( x ) -U p ( x )), for all p ∈ [ r ], j ∈ [ n ], the above bounds imply that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the above bounds we get that, for all p ∈ [ r ], j ∈ [ n ], where we made use of the following lemma:

Lemma 5 For any x, x ′ , y, y ′ , z, z ′ ≥ 0 such that x + y 1+ z ≤ 1 ,

Proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We describe a labeling of the points of the set ∆ r n ( d ) in terms of the function f . The labels that we are going to use are the elements of the set L := ∪ p ∈ [ r ] I p,n . In particular,

We assign to a point x ∈ ∆ r n the label ( p, j ) iff ( p, j ) is the lexicographically least index such that x p j &gt; 0 and f ( x ) p j -x p j ≤ f ( x ) q k -x q k , for all q ∈ [ r ] , k ∈ [ n ] .

This labeling rule satisfies the following properties:

- Completeness : Every point x is assigned a label; hence, we can define a labeling function /lscript : ∆ r n →L .
- Properness : x p j = 0 implies /lscript ( x ) = ( p, j ).

/negationslash

- Efficiency : /lscript ( x ) is computable in time polynomial in the binary encoding size of x and G .

A simplex σ ∈ Σ is called completely labeled if all its vertices have different labels; a simplex σ ∈ Σ is called p -stopping if it is completely labeled and, moreover, for all j ∈ [ n ], there exists a vertex of σ with label ( p, j ). Our labeling satisfies the following important property.

Theorem 2 ([37]) Suppose a simplex σ ∈ Σ is p -stopping for some p ∈ [ r ] . Then all points x ∈ σ ⊆ R n · r satisfy

<!-- formula-not-decoded -->

Proof. It is not hard to verify that, for any simplex σ ∈ Σ and for all pairs of points x, x ′ ∈ σ ,

<!-- formula-not-decoded -->

Suppose now that a simplex σ ∈ Σ is p -stopping, for some p ∈ [ r ], and that, for all j ∈ [ n ], z ( j ) is the vertex of σ with label ( p, j ). Since, for any x , ∑ i ∈ [ n ] x p i = 1 = ∑ i ∈ [ n ] f ( x ) p i , it follows from the labeling rule that

Hence, for all x ∈ σ , j ∈ [ n ],

<!-- formula-not-decoded -->

where we used the fact that the diameter of σ is 1 d (in the infinity norm) and the function f is λ -Lipschitz. Hence, in the opposite direction, for all x ∈ σ , j ∈ [ n ], we have

<!-- formula-not-decoded -->

Now, by the definition of the labeling rule, we have, for all x ∈ σ , q ∈ [ r ], j ∈ [ n ],

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the above, it follows that, for all x ∈ σ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

whereas

## The Approximation Guarantee. By virtue of Theorem 2, if we choose

<!-- formula-not-decoded -->

then a p -stopping simplex σ ∈ Σ, for any p ∈ [ r ], satisfies that, for all x ∈ σ ,

<!-- formula-not-decoded -->

which by Lemma 6 below implies that x is a n √ /epsilon1 ′ (1 + nU max ) ( 1 + √ /epsilon1 ′ (1 + nU max ) ) max { U max , 1 } -approximate Nash equilibrium. Choosing

<!-- formula-not-decoded -->

implies that x is an /epsilon1 -approximate Nash equilibrium.

Lemma 6 If a vector x = ( x 1 ; x 2 ; . . . ; x r ) ∈ R n · r satisfies

<!-- formula-not-decoded -->

Proof. Let us fix some player p ∈ [ r ], and assume, without loss of generality, that then x is a n √ /epsilon1 ′ (1 + nU max ) ( 1 + √ /epsilon1 ′ (1 + nU max ) ) max { U max , 1 } -approximate Nash equilibrium.

<!-- formula-not-decoded -->

For all j ∈ [ n ], observe that | f ( x ) p j -x p j | ≤ /epsilon1 ′ implies

Setting /epsilon1 ′′ := /epsilon1 ′ (1 + nU max ), the above inequality implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us define t := x p k +1 + x p k +2 + . . . + x p n , and let us distinguish the following cases

- If t ≥ √ /epsilon1 ′′ U max , then summing Equation (3) for j = k +1 , . . . , n implies

which gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- If t ≤ √ /epsilon1 ′′ U max , then multiplying Equation (3) by x p j and summing over j = 1 , . . . , n gives

<!-- formula-not-decoded -->

Now observe that for any setting of the probabilities x p j , j ∈ [ n ], it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, observe that, since U p ( x ) = ∑ j ∈ [ n ] x p j U p j ( x ), it follows that which implies that

<!-- formula-not-decoded -->

Plugging this into (5) implies

<!-- formula-not-decoded -->

Further, using (6) gives

<!-- formula-not-decoded -->

which implies

The last inequality then implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (4) and (7), we have the following uniform bound

<!-- formula-not-decoded -->

Since B p 1 ( x ) = U p 1 ( x ) -U ( x ), it follows that player p cannot improve her payoff by more that /epsilon1 ′′′ by changing her strategy. This is true for every player, hence x is a /epsilon1 ′′′ -approximate Nash equilibrium.

The Edges of the end of the line Graph. Laan and Talman [37] describe a pivoting algorithm which operates on the set Σ, by specifying the following:

- a simplex σ 0 ∈ Σ, which is the starting simplex ; σ 0 contains the point q 0 and is uniquely determined by the labeling rule;

- a partial one-to-one function h : Σ → Σ, mapping a simplex to a neighboring simplex, which defines a pivoting rule ; h has the following properties 1 :
- -σ 0 has no pre-image;
- -any simplex σ ∈ Σ that has no image is a p -stopping simplex for some p ; and, any simplex σ ∈ Σ \ { σ 0 } that has no pre-image is a p -stopping simplex for some p ;
- -both h ( σ ) and h -1 ( σ ) are computable in time polynomial in the binary encoding size of σ , that is N , and G -given that the labeling function /lscript is efficiently computable;

The algorithm of Laan and Talman starts off with the simplex σ 0 and employs the pivoting rule h until a simplex σ with no image is encountered. By the properties of h , σ must be p -stopping for some p ∈ [ r ] and, by the discussion above, any point x ∈ σ is an /epsilon1 -approximate Nash equilibrium.

In our construction, the edges of the end of the line graph are defined in terms of the function h : if h ( σ ) = σ ′ , then there is a directed edge from σ to σ ′ . Moreover, the string 0 N is identified with the simplex σ 0 . Any solution to the end of the line problem thus defined corresponds by the above discussion to a simplex σ such that any point x ∈ σ is an /epsilon1 -approximate Nash equilibrium of G . This concludes the construction.

## 3.3 The Brouwer Problem

In the proof of our main result we use a problem we call Brouwer , which is a discrete and simplified version of the search problem associated with Brouwer's fixed point theorem. We are given a continuous function φ from the 3-dimensional unit cube to itself, defined in terms of its values at the centers of 2 3 n cubelets with side 2 -n , for some n ≥ 0 2 . At the center c ijk of the cubelet K ijk defined as

<!-- formula-not-decoded -->

where i, j, k are integers in { 0 , 1 , . . . , 2 n -1 } , the value of φ is φ ( c ijk ) = c ijk + δ ijk , where δ ijk is one of the following four vectors (also referred to as colors):

- δ 1 = ( α, 0 , 0)
- δ 2 = (0 , α, 0)
- δ 3 = (0 , 0 , α )
- δ 0 = ( -α, -α, -α )

Here α &gt; 0 is much smaller than the cubelet side, say 2 -2 n .

Thus, to compute φ at the center of the cubelet K ijk we only need to know which of the four displacements to add. This is computed by a circuit C (which is the only input to the problem) with 3 n input bits and 2 output bits; C ( i, j, k ) is the index r such that, if c is the center of cubelet K ijk , φ ( c ) = c + δ r . C is such that C (0 , j, k ) = 1, C ( i, 0 , k ) = 2, C ( i, j, 0) = 3, and

1 More precisely, the pivoting rule h of Laan and Talman is defined on a subset Σ ′ of Σ. For our purposes, let us extend their pivoting rule h to the set Σ by setting h ( σ ) = σ for all σ ∈ Σ \ Σ ′ .

2 The value of the function near the boundaries of the cubelets could be determined by interpolation -there are many simple ways to do this, and the precise method is of no importance to our discussion.

C (2 n -1 , j, k ) = C ( i, 2 n -1 , k ) = C ( i, j, 2 n -1) = 0 (with conflicts resolved arbitrarily), so that the function φ maps the boundary to the interior of the cube. A vertex of a cubelet is called panchromatic if among the cubelets adjacent to it there are four that have all four displacements δ 0 , δ 1 , δ 2 , δ 3 . Sperner's Lemma guarantees that, for any circuit C satisfying the above properties, a panchromatic vertex exists, see, e.g., [48]. An alternative proof of this fact follows as a consequence of Theorem 3 below.

Brouwer is thus the following total problem: Given a circuit C as described above, find a panchromatic vertex. The relationship with Brouwer fixed points is that fixed points of φ only ever occur in the vicinity of a panchromatic vertex . We next show:

## Theorem 3 Brouwer is PPAD -complete.

Proof. That Brouwer is in PPAD follows from the main result of this paper (Theorem 12), which is a reduction from Brouwer to r -Nash , which has been shown to be in PPAD in Theorem 1.

To show hardness, we shall reduce end of the line to Brouwer . Given circuits S and P with n inputs and outputs, as prescribed in that problem, we shall construct an 'equivalent' instance of Brouwer , that is, another circuit C with 3 m = 3( n +4) inputs and two outputs that computes the color of each cubelet of side 2 -m , that is to say, the index i such that δ i is the correct displacement of the Brouwer function at the center of the cubelet encoded into the 3 m bits of the input. We shall first describe the Brouwer function φ explicitly, and then argue that it can be computed by a circuit.

Our description of φ proceeds as follows: We shall first describe a 1-dimensional subset L of the 3-dimensional unit cube, intuitively an embedding of the path-like directed graph G S,P implicitly given by S and P . Then we shall describe the 4-coloring of the 2 3 m cubelets based on the description of L . Finally, we shall argue that colors are easy to compute locally, and that panchromatic vertices correspond to endpoints other than the standard source 0 n of G S,P .

We assume that the graph G S,P is such that for each edge ( u, v ), one of the vertices is even (ends in 0) and the other is odd; this is easy to guarantee by duplicating the vertices of G S,P .

L will be orthonormal, that is, each of its segments will be parallel to one of the axes; all coordinates of endpoints of segments are integer multiples of 2 -m , a factor that we omit in the discussion below. Let u ∈ { 0 , 1 } n be a vertex of G S,P . By 〈 u 〉 we denote the integer between 0 and 2 n -1 whose binary representation is u . Associated with u there are two line segments of length 4 of L . The first, called the principal segment of u , has endpoints u 1 = (8 〈 u 〉 +2 , 3 , 3) and u ′ 1 = (8 〈 u 〉 + 6 , 3 , 3). The other auxiliary segment has endpoints u 2 = (3 , 8 〈 u 〉 + 6 , 2 m -3) and u ′ 2 = (3 , 8 〈 u 〉 +10 , 2 m -3). Informally, these segments form two dashed lines (each segment being a dash) that run along two edges of the cube and slightly in its interior (see Figure 1).

Now, for every vertex u of G S,P , we connect u ′ 1 to u 2 by a line with three straight segments, with joints u 3 = (8 〈 u 〉 +6 , 8 〈 u 〉 +6 , 3) and u 4 = (8 〈 u 〉 +6 , 8 〈 u 〉 +6 , 2 m -3). Finally, if there is an edge ( u, v ) in G S,P , we connect u ′ 2 to v 1 by a jointed line with breakpoints u 5 = (8 〈 v 〉 +2 , 8 〈 u 〉 +10 , 2 m -3) and u 6 = (8 〈 v 〉 +2 , 8 〈 u 〉 +10 , 3). This completes the description of the line L if we do the following perturbation: exceptionally, the principal segment of u = 0 n has endpoints 0 1 = (2 , 2 , 2) and 0 ′ 1 = (6 , 2 , 2) and the corresponding joint is 0 3 = (6 , 6 , 2).

It is easy to see that L traverses the interior of the cube without ever 'nearly crossing itself'; that is, two points p, p ′ of L are closer than 3 · 2 -m in Euclidean distance only if they are connected by a part of L that has length 8 · 2 -m or less. (This is important in order for the coloring described below of the cubelets surrounding L to be well-defined.) To check this, just notice that segments of different types (e.g., [ u 3 , u 4 ] and [ u ′ 2 , u 5 ]) come closer than 3 · 2 -m only if they share an endpoint; segments of the same type on the z = 3 or the z = 2 m -3 plane are parallel and at least 4 apart; and segments parallel to the z axis differ by at least 4 in either their x or y coordinates.

Figure 1: The orthonormal path connecting vertices (u,v); the arrows indicate the orientation of colors surrounding the path.

<!-- image -->

We now describe the coloring of the 2 3 m cubelets by four colors corresponding to the four displacements. Consistent with the requirements for a Brouwer circuit, we color any cubelet K ijk where any one of i, j, k is 2 m -1, with 0. Given that, any other cubelet with i = 0 gets color 1; with this fixed, any other cubelet with j = 0 gets color 2, while the remaining cubelets with k = 0 get color 3. Having colored the boundaries, we now have to color the interior cubelets. An interior cubelet is always colored 0 unless one of its vertices is a point of the interior of line L , in which case it is colored by one of the three other colors in a manner to be explained shortly. Intuitively, at each point of the line L , starting from (2 , 2 , 2) (the beginning of the principle segment of the string u = 0 n ) the line L is 'protected' from color 0 from all 4 sides. As a result, the only place where the four colors can meet is vertex u ′ 2 or u 1 , u = 0 n , where u is an end of the line . . .

In particular, near the beginning of L at (2 , 2 , 2) the 27 cubelets K ijk with i, j, k ≤ 2 are colored as shown in Figure 2. From then on, for any length-1 segment of L of the form [( x, y, z ) , ( x ′ , y ′ , z ′ )] consider the four cubelets containing this segment. Two of these cubelets are colored 3, and the other two are colored 1 and 2, in this order clockwise (from the point of view of an observer at ( x, y, z )). The remaining cubelets touching L are the ones at the joints where L turns. Each of these cubelets, a total of two per turn, takes the color of the two other cubelets adjacent to L with which it shares a face.

/negationslash

Now it only remains to describe, for each line segment [ a, b ] of L , the direction d in which the

Figure 2: The 27 cubelets around the beginning of line L .

<!-- image -->

two cubelets that are colored 3 lie. The rules are these (in Figure 1 the directions d are shown as arrows):

- If [ a, b ] = [ u 1 , u ′ 1 ] then d = (0 , 0 , -1) if u is even and d = (0 , 0 , 1) if u is odd.
- If [ a, b ] = [ u ′ 1 , u 3 ] then d = (0 , 0 , -1) if u is even and d = (0 , 0 , 1) if u is odd.
- If [ a, b ] = [ u 3 , u 4 ] then d = (0 , 1 , 0) if u is even and d = (0 , -1 , 0) if u is odd.
- If [ a, b ] = [ u 4 , u 2 ] then d = (0 , 1 , 0) if u is even and d = (0 , -1 , 0) if u is odd.
- If [ a, b ] = [ u 2 , u ′ 2 ] then d = (1 , 0 , 0) if u is even and d = ( -1 , 0 , 0) if u is odd.
- If [ a, b ] = [ u ′ 2 , u 5 ] then d = (0 , -1 , 0) if u is even and d = (0 , 1 , 0) if u is odd.
- If [ a, b ] = [ u 5 , u 6 ] then d = (0 , -1 , 0) if u is even and d = (0 , 1 , 0) if u is odd.
- If [ a, b ] = [ u 6 , v 1 ] then d = (0 , 0 , 1) if u is even and d = (0 , 0 , -1) if u is odd.

This completes the description of the construction. Notice that, for this to work, we need our assumption that edges in G S,P go between odd and even vertices. Regarding the alternating orientation of colored cubelets around L , note that we could not simply introduce 'twists' to make them always point in (say) direction d = (0 , 0 , -1) for all [ u 1 , u ′ 1 ]. That would create a panchromatic vertex at the location of a twist.

The result now follows from the following two claims:

1. A point in the cube is panchromatic in the described coloring if and only if it is
2. (a) an endpoint u ′ 2 of a sink vertex u of G S,P , or
3. (b) an endpoint u 1 of a source vertex u = 0 n of G S,P

/negationslash

2. A circuit C can be constructed in time polynomial in | S | + | P | , which computes, for each triple of binary integers i, j, k &lt; 2 m , the color of cubelet K ijk .

Regarding the first claim, the endpoint u ′ 2 of a sink vertex u , or the endpoint u 1 of a source vertex u other than 0 n , will be a point where L meets color 0, hence a panchromatic vertex. There is no alternative way that L can meet color 0 and no other way a panchromatic vertex can occur.

Regarding the second claim, circuit C is doing the following. C (0 , j, k ) = 1, for j, k &lt; 2 m -1, C ( i, 0 , k ) = 2 for i &gt; 0, i, k &lt; 2 m -1, C ( i, j, 0) = 3 for i, j &gt; 0, i, j &lt; 2 m -1. Then by default, C ( i, j, k ) = 0. However the following tests yield alternative values for C ( i, j, k ), for cubelets adjacent to L . LSB ( x ) denotes the least significant bit of x , equal to 1 if x is odd, 0 if x is even, and undefined if x is not an integer. For example, a [ u ′ 1 , u 3 ] , u = 0 n segment is given by (letting x = 〈 u 〉 ):

/negationslash

1. If k = 2 and i = 8 x +5 and LSB ( x ) = 1 and j ∈ { 3 , . . . , 8 x +6 } then C ( i, j, k ) = 2.
2. If k = 2 and i = 8 x +6 and LSB ( x ) = 1 and j ∈ { 2 , . . . , 8 x +6 } then C ( i, j, k ) = 1.
3. If k = 3 and ( i = 8 x + 5 or i = 8 x + 6) and LSB ( x ) = 1 and j ∈ { 2 , . . . , 8 x + 5 } then C ( i, j, k ) = 3.
4. If k = 2 and ( i = 8 x + 5 or i = 8 x + 6) and LSB ( x ) = 0 and j ∈ { 2 , . . . , 8 x + 6 } then C ( i, j, k ) = 3.
5. If k = 3 and i = 8 x +5 and LSB ( x ) = 0 and j ∈ { 3 , . . . , 8 x +5 } then C ( i, j, k ) = 1.
6. If k = 3 and i = 8 x +6 and LSB ( x ) = 0 and j ∈ { 2 , . . . , 8 x +5 } then C ( i, j, k ) = 2.

A [ u ′ 2 , u 5 ] segment uses the circuits P and S , and, in the case LSB ( x ) = 1, x = 〈 u 〉 , is given by:

1. If ( k = 2 m -3 or k = 2 m -4) and j = 8 x + 10 and S ( x ) = x ′ and P ( x ′ ) = x and i ∈ { 2 , . . . , 8 x ′ +2 } then C ( i, j, k ) = 3.
2. If k = 2 m -3 and and j = 8 x +9 and S ( x ) = x ′ and P ( x ′ ) = x and i ∈ { 3 , . . . , 8 x ′ +2 } then C ( i, j, k ) = 1.
3. If k = 2 m -4 and j = 8 x +9 and S ( x ) = x ′ and P ( x ′ ) = x and i ∈ { 3 , . . . , 8 x ′ + 1 } then C ( i, j, k ) = 2.

The other segments are done in a similar way, and so the second claim follows. This completes the proof of hardness.

## 4 Reductions Among Equilibrium Problems

In the next section we show that r -Nash is PPAD-hard by reducing Brouwer to it. Rather than r -Nash , it will be more convenient to first reduce Brouwer to d -graphical Nash , the problem of computing a Nash equilibrium in graphical games of degree d . Therefore, we need to show that the latter reduces to r -Nash . This will be the purpose of the current section; in fact, we will establish something stronger, namely that

Theorem 4 For every fixed d, r ≥ 3 ,

- Every r -player normal form game and every graphical game of degree d can be mapped in polynomial time to (a) a 3 -player normal form game and (b) a graphical game with degree 3 and 2 strategies per player, such that there is a polynomial-time computable surjective mapping from the set of Nash equilibria of the latter to the set of Nash equilibria of the former.
- There are polynomial-time reductions from r -Nash and d -graphical Nash to both 3 -Nash and 3 -graphical Nash .

Note that the first part of the theorem establishes mappings of exact equilibrium points between different games, whereas the second asserts that computing approximate equilibrium points in all these games is polynomial-time equivalent. The proof, which is quite involved, is presented in the following subsections. In Subsection 4.1, we present some useful ideas that enable the reductions described in Theorem 4, as well as prepare the necessary machinery for the reduction from Brouwer to d -graphical Nash in Section 5. Subsections 4.2 through 4.6 provide the proof of the theorem. In Subsection 4.7, we establish a polynomial-time reduction from the problem of computing an approximately well supported Nash equilibrium to the problem of computing an approximate Nash equilibrium. A mapping from r -player games to 3-player games was already known by Bubelis [5].

## 4.1 Preliminaries: Game Gadgets

We describe the building blocks of our constructions. As we have observed earlier, if a player v has two pure strategies, say 0 and 1, then every mixed strategy of that player corresponds to a real number p [ v ] ∈ [0 , 1] which is precisely the probability that the player plays strategy 1. Identifying players with these numbers, we are interested in constructing games that perform simple arithmetical operations on mixed strategies; for example, we are interested in constructing a game with two 'input' players v 1 and v 2 and another 'output' player v 3 so that in any Nash equilibrium the latter plays the sum of the former, i.e., p [ v 3 ] = min { p [ v 1 ] + p [ v 2 ] , 1 } . Such constructions are considered below.

Notation: We use x = y ± /epsilon1 to denote y -/epsilon1 ≤ x ≤ y + /epsilon1 .

Proposition 1 Let α be a non-negative real number. Let v 1 , v 2 , w be players in a graphical game GG with two strategies per player, and suppose that the payoffs to v 2 and w are as follows.

<!-- formula-not-decoded -->

Figure 3: G × α , G =

<!-- image -->

Payoffs to w :

| w plays 0   | v 1 plays 0 v plays 1   | v 2 plays 0 v 2 plays 1 0 0 α α   |
|-------------|-------------------------|-----------------------------------|
| w plays 0   | 1                       |                                   |
| w plays 0   |                         | v 2 plays 0 v 2 plays 1           |
| w plays 1   | v 1 plays 0 v plays 1   | 0 1 1                             |
| w plays 1   | 1                       | 0                                 |

Then, for /epsilon1 &lt; 1 , in every /epsilon1 -Nash equilibrium of game GG , p [ v 2 ] = min( α p [ v 1 ] , 1) ± /epsilon1 . In particular, in every Nash equilibrium of game GG , p [ v 2 ] = min( α p [ v 1 ] , 1) .

Proof. If w plays 1, then the expected payoff to w is p [ v 2 ], and, if w plays 0, the expected payoff to w is α p [ v 1 ]. Therefore, in an /epsilon1 -Nash equilibrium of GG , if p [ v 2 ] &gt; α p [ v 1 ] + /epsilon1 then p [ w ] = 1. However, note also that if p [ w ] = 1 then p [ v 2 ] = 0. (Payoffs to v 2 make it prefer to disagree with w .) Consequently, p [ v 2 ] cannot be larger than α p [ v 1 ] + /epsilon1 , so it cannot be larger than min( α p [ v 1 ] , 1)+ /epsilon1 . Similarly, if p [ v 2 ] &lt; min( α p [ v 1 ] , 1) -/epsilon1 , then p [ v 2 ] &lt; α p [ v 1 ] -/epsilon1 , so p [ w ] = 0, which implies -again since v 2 has the biggest payoff by disagreeing with w - that p [ v 2 ] = 1 ≥ 1 -/epsilon1 , a contradiction to p [ v 2 ] &lt; min( α p [ v 1 ] , 1) -/epsilon1 . Hence p [ v 2 ] cannot be less than min( α p [ v 1 ] , 1) -/epsilon1 .

We will denote by G × α the (directed) graphical game shown in Figure 3, where the payoffs to players v 2 and w are specified as in Proposition 1 and the payoff of player v 1 is completely unconstrained: v 1 could have any dependence on other players of a larger graphical game GG that contains G × α or even depend on the strategies of v 2 and w ; as long as the payoffs of v 2 and w are specified as above the conclusion of the proposition will be true. Note in particular that using the above construction with α = 1, v 2 becomes a 'copy' of v 1 ; we denote the corresponding graphical game by G = . These graphical games will be used as building blocks in our constructions; the way to incorporate them into some larger graphical game is to make player v 1 depend (incoming edges) on other players of the game and make v 2 affect (outgoing edges) other players of the game. For example, we can make a sequence of copies of any vertex, which form a path in the graph. The copies then will alternate with distinct w vertices.

Proposition 2 Let α , β , γ be non-negative real numbers. Let v 1 , v 2 , v 3 , w be players in a graphical game GG with two strategies per player, and suppose that the payoffs to v 3 and w are as follows.

<!-- formula-not-decoded -->

Figure 4: G α

Figure 5: G + , G ∗ , G -

Payoffs to w :

<!-- formula-not-decoded -->

Then, for /epsilon1 &lt; 1 , in any /epsilon1 -Nash equilibrium of GG , p [ v 3 ] = min( α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] , 1) ± /epsilon1 . In particular, in every Nash equilibrium of GG , p [ v 3 ] = min( α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] , 1) .

Proof. If w plays 1, then the expected payoff to w is p [ v 3 ], and if w plays 0 then the expected payoff to w is α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ]. Therefore, in an /epsilon1 -Nash equilibrium of GG , if p [ v 3 ] &gt; α p [ v 1 ]+ β p [ v 2 ]+ γ p [ v 1 ] p [ v 2 ]+ /epsilon1 then p [ w ] = 1. However, note from the payoffs to v 3 that if p [ w ] = 1 then p [ v 3 ] = 0. Consequently, p [ v 3 ] cannot be strictly larger than α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] + /epsilon1 . Similarly, if p [ v 3 ] &lt; min( α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] , 1) -/epsilon1 , then p [ v 3 ] &lt; α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] -/epsilon1 and, due to the payoffs to w , p [ w ] = 0. This in turn implies -since v 3 has the biggest payoff by disagreeing with w - that p [ v 3 ] = 1 ≥ 1 -/epsilon1 , a contradiction to p [ v 3 ] &lt; min( α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] , 1) -/epsilon1 . Hence p [ v 3 ] cannot be less than min( α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] , 1) -/epsilon1 .

Remark 1 It is not hard to verify that, if v 1 , v 2 , v 3 , w are players of a graphical game GG and the payoffs to v 3 , w are specified as in Proposition 2 with α = 1 , β = -1 and γ = 0 , then, in every /epsilon1 -Nash equilibrium of the game GG , p [ v 3 ] = max(0 , p [ v 1 ] -p [ v 2 ]) ± /epsilon1 ; in particular, in every Nash equilibrium, p [ v 3 ] = max(0 , p [ v 1 ] -p [ v 2 ]) .

Let us denote by G + and G ∗ the (directed) graphical game shown in Figure 5, where the payoffs to players v 3 and w are specified as in Proposition 2 taking ( α, β, γ ) equal to (1 , 1 , 0) (addition) and (0 , 0 , 1) (multiplication) respectively. Also, let G -be the game when the payoffs of v 3 and w are specified as in Remark 1.

Proposition 3 Let v 1 , v 2 , v 3 , v 4 , v 5 , v 6 , w 1 , w 2 , w 3 , w 4 be vertices in a graphical game GG with two strategies per player, and suppose that the payoffs to vertices other than v 1 and v 2 are as follows.

Payoffs to w 1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Payoffs to w 2 and v 3 are chosen using Proposition 2 to ensure p [ v 3 ] = p [ v 1 ](1 -p [ v 5 ]) ± /epsilon1 3 , in every /epsilon1 -Nash equilibrium of game GG .

3 We can use Proposition 2 to multiply by (1 -p [ v 5 ]) in a similar way to multiplication by p [ v 5 ]; the payoffs to w 2 have v 5 's strategies reversed.

which implies and, therefore,

Figure 6: G max

<!-- image -->

Payoffs to w 3 and v 4 are chosen using Proposition 2 to ensure p [ v 4 ] = p [ v 2 ] p [ v 5 ] ± /epsilon1 , in every /epsilon1 -Nash equilibrium of game GG .

Payoffs to w 4 and v 6 are chosen using Proposition 2 to ensure p [ v 6 ] = min(1 , p [ v 3 ]+ p [ v 4 ]) ± /epsilon1 , in every /epsilon1 -Nash equilibrium of game GG .

Then, for /epsilon1 &lt; 1 , in every /epsilon1 -Nash equilibrium of game GG , p [ v 6 ] = max( p [ v 1 ] , p [ v 2 ]) ± 4 /epsilon1 . In particular, in every Nash equilibrium, p [ v 6 ] = max( p [ v 1 ] , p [ v 2 ]) .

The graph of the game looks as in Figure 6. It is actually possible to 'merge' w 1 and v 5 , but we prefer to keep the game as is in order to maintain the bipartite structure of the graph in which one side of the partition contains all the vertices corresponding to arithmetic expressions (the v i vertices) and the other side all the intermediate w i vertices.

Proof. If, in an /epsilon1 -Nash equilibrium, we have p [ v 1 ] &lt; p [ v 2 ] -/epsilon1 , then it follows from w 1 's payoffs that p [ w 1 ] = 1. It then follows that p [ v 5 ] = 1 since v 5 's payoffs induce it to imitate w 1 . Hence, p [ v 3 ] = ± /epsilon1 and p [ v 4 ] = p [ v 2 ] ± /epsilon1 , and, consequently, p [ v 3 ] + p [ v 4 ] = p [ v 2 ] ± 2 /epsilon1 . This implies p [ v 6 ] = p [ v 2 ] ± 3 /epsilon1 , as required. A similar argument shows that, if p [ v 1 ] &gt; p [ v 2 ] + /epsilon1 , then p [ v 6 ] = p [ v 1 ] ± 3 /epsilon1 .

If | p [ v 1 ] -p [ v 2 ] | ≤ /epsilon1 , then p [ w 1 ] and, consequently, p [ v 5 ] may take any value. Assuming, without loss of generality that p [ v 1 ] ≥ p [ v 2 ], we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We conclude the section with the simple construction of a graphical game G α , depicted in Figure 4, which performs the assignment of some fixed value α ≥ 0 to a player. The proof is similar in spirit to our proof of Propositions 1 and 2 and will be skipped.

Proposition 4 Let α be a non-negative real number. Let w , v 1 be players in a graphical game GG with two strategies per player and let the payoffs to w , v 1 be specified as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, for /epsilon1 &lt; 1 , in every /epsilon1 -Nash equilibrium of game GG , p [ v 1 ] = min( α, 1) ± /epsilon1 . In particular, in every Nash equilibrium of GG , p [ v 1 ] = min( α, 1) .

Before concluding the section we give a useful definition.

Definition 2 Let v 1 , v 2 , . . . , v k , v be players of a graphical game G f such that, in every Nash equilibrium, it holds that p [ v ] = f ( p [ v 1 ] , . . . , p [ v k ]) , where f is some function with k arguments and range [0 , 1] . We say that the game G f has error amplification at most c if, in every /epsilon1 -Nash equilibrium, it holds that p [ v ] = f ( p [ v 1 ] , . . . , p [ v k ]) ± c/epsilon1 .

In particular, the games G = , G + , G -, G ∗ , G α described above have error amplifications at most 1, whereas the game G max has error amplification at most 4.

## 4.2 Reducing Graphical Games to Normal Form Games

We establish a mapping from graphical games to normal form games as specified by the following theorem.

Theorem 5 For every d &gt; 1 , a graphical game (directed or undirected) GG of maximum degree d can be mapped in polynomial time to a ( d 2 +1) -player normal form game G so that there is a polynomial-time computable surjective mapping g from the Nash equilibria of the latter to the Nash equilibria of the former.

## Proof. Overview:

Figure 7 shows the construction of G = f ( GG ). We will explain the construction in detail as well as show that it can be computed in polynomial time. We will also establish that there is a surjective mapping from the Nash equilibria of G to the Nash equilibria of GG . In the following discussion we will refer to the players of the graphical game as 'vertices' to distinguish them from the players of the normal form game.

We first rescale all payoffs so that they are nonnegative and at most 1 (Step 1); it is easy to see that the set of Nash equilibria is preserved under this transformation. Also, without loss of generality, we assume that all vertices v ∈ V have the same number of strategies, | S v | = t . We color the vertices of G , where G = ( V, E ) is the affects graph of GG , so that any two adjacent vertices have different colors, but also any two vertices with a common successor have different colors (Step 3). Since this type of coloring will be important for our discussion we will define it formally.

/negationslash

/negationslash

/negationslash

To get such coloring, it is sufficient to color the union of the underlying undirected graph G ′ with its square (with self-loops removed) so that no adjacent vertices have the same color; this can be done with at most d 2 colors -see, e.g., [6]- since G ′ has degree d by assumption; we are going to use r = d 2 or r = d 2 + 1 colors, whichever is even, for reasons to become clear shortly. We assume for simplicity that each color class has the same number of vertices, adding dummy vertices

Definition 3 Let GG be a graphical game with affects graph G = ( V, E ) . We say that GG can be legally colored with k colors if there exists a mapping c : V → { 1 , 2 , . . . , k } such that, for all e = ( v, u ) ∈ E , c ( v ) = c ( u ) and, moreover, for all e 1 = ( v, w ) , e 2 = ( u, w ) ∈ E with v = u , c ( v ) = c ( u ) . We call such coloring a legal k -coloring of GG .

Input: Degree d graphical game GG : vertices V , | V | = n ′ , | S v | = t for all v ∈ V . Output: .

Normal-form game G

1. If needed, rescale the entries in the payoff tables of GG so that they lie in the range [0 , 1]. One way to do so is to divide all payoff entries by max { u } , where max { u } is the largest entry in the payoff tables of GG .
2. Let r = d 2 or r = d 2 +1; r chosen to be even.
3. Let c : V -→ { 1 , . . . , r } be a r -coloring of GG such that no two adjacent vertices have the same color, and, furthermore, no two vertices having a common successor -in the affects graph of the game- have the same color. Assume that each color is assigned to the same number of vertices, adding to V extra isolated vertices to make up any shortfall; extend mapping c to these vertices. Let { v ( i ) 1 , . . . , v ( i ) n/r } denote { v : c ( v ) = i } , where n ≥ n ′ .
4. For each p ∈ [ r ], game G will have a player, labeled p , with strategy set S p ; S p will be the union (assumed disjoint) of all S v with c ( v ) = p , i.e.,

<!-- formula-not-decoded -->

5. Taking S to be the cartesian product of the S p 's, let s ∈ S be a strategy profile of game G . For p ∈ [ r ], u p s is defined as follows:
2. (a) Initially, all utilities are 0.
3. (b) For v 0 ∈ V having predecessors v 1 , . . . , v d ′ in the affects graph of GG , if c ( v 0 ) = p (that is, v 0 = v ( p ) j for some j ) and, for i = 0 , . . . , d ′ , s contains ( v i , a i ), then u p s = u v 0 s ′ for s ′ a strategy profile of GG in which v i plays a i for i = 0 , . . . , d ′ .
4. (c) Let M &gt; 2 n r .
5. (d) For odd number p &lt; r , if player p plays ( v ( p ) i , a ) and p +1 plays ( v ( p +1) i , a ′ ), for any i , a , a ′ , then add M to u p s and subtract M from u p +1 s .

Figure 7: Reduction from graphical game GG to normal form game G

if needed to satisfy this property. Henceforth, we assume that n is an integer multiple of r so that every color class has n r vertices.

We construct a normal form game G with r ≤ d 2 +1 players. Each of them corresponds to a color and has t n r strategies, the t strategies of each of the n r vertices in its color class (Step 4). Since r is even, we can divide the r players into pairs and make each pair play a generalized Matching Pennies game (see Definition 4 below) at very high stakes, so as to ensure that all players will randomize uniformly over the vertices assigned to them 4 . Within the set of strategies associated with each vertex, the Matching Pennies game expresses no preference, and payoffs are augmented to correspond to the payoffs that would arise in the original graphical game GG (see Step 5 for the exact specification of the payoffs).

Definition 4 The (2-player) game Generalized Matching Pennies is defined as follows. Call the 2 players the pursuer and the evader , and let [ n ] denote their strategies. If for any i ∈ [ n ] both players play i , then the pursuer receives a positive payoff u &gt; 0 and the evader receives a payoff

4 A similar trick is used in Theorem 7.3 of [55], a hardness result for a class of circuit games.

of -u . Otherwise both players receive 0 . It is not hard to check that the game has a unique Nash equilibrium in which both players use the uniform distribution.

## Polynomial size of G = f ( GG ) :

## Construction of the mapping g :

The input size is |GG| = Θ( n ′ · t d +1 · q ), where n ′ is the number of vertices in GG and q the size of the values in the payoff matrices in the logarithmic cost model. The normal form game G has r ∈ { d 2 , d 2 +1 } players, each having tn/r strategies, where n ≤ rn ′ is the number of vertices in GG after the possible addition of dummy vertices to make sure that all color classes have the same number of vertices. Hence, there are r · ( tn/r ) r ≤ ( ( d 2 +1) ( tn ′ ) d 2 +1 ) payoff entries in G . This is polynomial in |G G | so long as d is constant. Moreover, each payoff entry will be of polynomial size since M is of polynomial size and each payoff entry of the game G is the sum of 0 or M and a payoff entry of GG .

Given a Nash equilibrium N G = { x p ( v,a ) } p,v,a of G = f ( GG ), we claim that we can recover a Nash equilibrium { x v a } v,a of GG , N GG = g ( N G ), as follows:

<!-- formula-not-decoded -->

Clearly g is computable in polynomial time.

## Proof that g maps Nash equilibria of G to Nash equilibria of GG :

For v ∈ V , c ( v ) = p , let ' p plays v ' denote the event that p plays ( v, a ) for some a ∈ S v . We show that in a Nash equilibrium N G of game G , for every player p and every v ∈ V with c ( v ) = p , Pr( p plays v ) ∈ [ λ -1 M , λ + 1 M ], where λ = ( n r ) -1 . Note that the 'fair share' for v is λ .

Call GG ′ the graphical game resulting from GG by rescaling the utilities so that they lie in the range [0 , 1]. It is easy to see that any Nash equilibrium of game GG is, also, a Nash equilibrium of game GG ′ and vice versa. Therefore, it is enough to establish that the mapping g maps every Nash equilibrium of game G to a Nash equilibrium of game GG ′ .

Lemma 7 For all v ∈ V , in a Nash equilibrium of G , Pr( c ( v ) plays v ) ∈ [ λ -1 M , λ + 1 M ] .

If p is odd (a pursuer) then p + 1 (the evader) will have utility of at least -λM + 1 for playing any strategy ( v ( p +1) i , a ) , a ∈ S v ( p +1) i , whereas utility of at most -λM -λ +1 for playing any strategy ( v ( p +1) j , a ), a ∈ S v ( p +1) j . Since -λM + 1 &gt; -λM -λ + 1, in a Nash equilibrium, Pr ( p +1 plays v ( p +1) j ) = 0. Therefore, there exists some k such that Pr ( p +1 plays v ( p +1) k ) &gt; λ . Now the payoff of p for playing any strategy ( v ( p ) j , a ) , a ∈ S v ( p ) j , is at most 1, whereas the payoff for playing any strategy ( v ( p ) k , a ) , a ∈ S v ( p ) k is at least λM . Thus, in a Nash equilibrium, player p should not include any strategy ( v ( p ) j , a ) , a ∈ S v ( p ) j , in her support; hence Pr ( p plays v ( p ) j ) = 0, a contradiction.

Proof. Suppose, for a contradiction, that in a Nash equilibrium of G , Pr ( p plays v ( p ) i ) &lt; λ -1 M for some i , p . Then there exists some j such that Pr ( p plays v ( p ) j ) &gt; λ + 1 M λ .

If p is even, then p -1 will have utility of at most ( λ -1 M ) M + 1 for playing any strategy ( v ( p -1) i , a ) , a ∈ S v ( p -1) i , whereas utility of at least ( λ + 1 M λ ) M for playing any strategy ( v ( p -1) j , a ),

a ∈ S v ( p -1) j . Hence, in a Nash equilibrium Pr ( p -1 plays v ( p -1) i ) = 0, which implies that there exists some k such that Pr ( p -1 plays v ( p -1) k ) &gt; λ . But, then, p will have utility of at least 0 for playing any strategy ( v ( p ) i , a ) , a ∈ S v ( p ) i , whereas utility of at most -λM +1 for playing any strategy ( v ( p ) k , a ), a ∈ S v ( p ) k . Since 0 &gt; -λM + 1, in a Nash equilibrium, Pr ( p plays v ( p ) k ) = 0. Therefore, there exists some k ′ such that Pr ( p plays v ( p ) k ′ ) &gt; λ . Now the payoff of p -1 for playing any strategy ( v ( p -1) k , a ) , a ∈ S v ( p -1) k , is at most 1, whereas the payoff for playing any strategy ( v ( p -1) k ′ , a ) , a ∈ S v ( p -1) k ′ is at least λM . Thus, in a Nash equilibrium, player p -1 should not include any strategy ( v ( p -1) k , a ) , a ∈ S v ( p -1) k , in her support; hence Pr ( p -1 plays v ( p -1) k ) = 0, a contradiction.

From the above discussion, it follows that every vertex is chosen with probability at least λ -1 M by the player that represents its color class. A similar argument shows that no vertex is chosen with probability greater than λ + 1 M . Indeed, suppose, for a contradiction, that in a Nash equilibrium of G , Pr ( p plays v ( p ) j ) &gt; λ + 1 M for some j , p ; then there exists some i such that Pr ( p plays v ( p ) i ) &lt; λ -1 M λ ; now, distinguish two cases depending on whether p is even or odd and proceed in the same fashion as in the argument used above to show that no vertex is chosen with probability smaller than λ -1 /M .

To see that { x v a } v,a , defined by (9), corresponds to a Nash equilibrium of GG ′ note that, for any player p and vertex v ∈ V such that c ( v ) = p , the division of Pr( p plays v ) into Pr( p plays ( v, a )), for various values of a ∈ S v , is driven entirely by the same payoffs as in GG ′ ; moreover, note that there is some positive probability p ( v ) ≥ ( λ -1 M ) d &gt; 0 that the predecessors of v are chosen by the other players of G and the additional expected payoff to p resulting from choosing ( v, a ), for some a ∈ S v , is p ( v ) times the expected payoff of v in GG ′ if v chooses action a and all other vertices play as specified by (9). More formally, suppose that p = c ( v ) for some vertex v of the graphical game GG ′ and, without loss of generality, assume that p is odd (pursuer) and that v is the vertex v ( p ) i in the notation of Figure 7. Then, in a Nash equilibrium of the game G , we have, by the definition of a Nash equilibrium, that for all strategies a, a ′ ∈ S v of vertex v :

<!-- formula-not-decoded -->

But

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and similarly for a ′ . Therefore, (10) implies

<!-- formula-not-decoded -->

Dividing by ∏ u ∈N ( v ) \{ v } ∑ j ∈ S u x c ( u ) ( u,j ) = ∏ u ∈N ( v ) \{ v } Pr( c ( u ) plays u ) = p ( v ) and invoking (9) gives

<!-- formula-not-decoded -->

where we used that p ( v ) ≥ ( λ -1 M ) d &gt; 0, which follows by Lemma 7.

Mapping g is surjective on the Nash equilibria of GG ′ and, therefore, GG : We will show that, for every Nash equilibrium N GG ′ = { x v a } v,a of GG ′ , there exists a Nash equilibrium N G = { x p ( v,a ) } p,v,a of G such that (9) holds. The existence can be easily established via the existence of a Nash equilibrium in a game G ′ defined as follows. Suppose that, in N GG ′ , every vertex v ∈ V receives an expected payoff of u v from every strategy in the support of { x v a } a . Define the following game G ′ whose structure results from G by merging the strategies { ( v, a ) } a of player p = c ( v ) into one strategy s p v , for every v such that c ( v ) = p . So the strategy set of player p in G ′ will be { s p v | c ( v ) = p } also denoted as { s ( p ) 1 , . . . , s ( p ) n/r } for ease of notation. Define now the payoffs to the players as follows. Initialize the payoff matrices with all entries equal to 0. For every strategy profile s ,

- for v 0 ∈ V having predecessors v 1 , . . . , v d ′ in the affects graph of GG ′ , if, for i = 0 , . . . , d ′ , s contains s c ( v i ) v i , then add u v 0 to u c ( v 0 ) s .
- for odd number p &lt; r if player p plays strategy s ( p ) i and player p + 1 plays strategy s ( p +1) i then add M to u p s and subtract M from u p +1 s (Generalized Matching Pennies).

Note the similarity in the definitions of the payoff matrices of G and G ′ . From Nash's theorem, game G ′ has a Nash equilibrium { y p s p v } p,v and it is not hard to verify that { x p ( v,a ) } p,v,a is a Nash equilibrium of game G , where x p ( v,a ) := y p s p v · x v a , for all p , v ∈ V such that c ( v ) = p , and a ∈ S v .

## 4.3 Reducing Normal Form Games to Graphical Games

We establish the following mapping from normal form games to graphical games.

Theorem 6 For every r &gt; 1 , a normal form game with r players can be mapped in polynomial time to an undirected graphical game of maximum degree 3 and two strategies per player so that there is a polynomial-time computable surjective mapping g from the Nash equilibria of the latter to the Nash equilibria of the former.

Given a normal form game G having r players, 1 , . . . , r , and n strategies per player, say S p = [ n ] for all p ∈ [ r ], we will construct a graphical game GG , with a bipartite graph of maximum degree 3, and 2 strategies per player, say { 0 , 1 } , with description length polynomial in the description length of G , so that from every Nash equilibrium of GG we can recover a Nash equilibrium of G . In the following discussion we will refer to the players of the graphical game as 'vertices' to distinguish them from the players of the normal form game. It will be easy to check that the graph of GG is bipartite and has degree 3; this graph will be denoted G = ( V ∪ W,E ), where W and V are disjoint, and each edge in E goes between V and W . For every vertex v of the graphical game, we will denote by p [ v ] the probability that v plays pure strategy 1.

Recall that G is specified by the quantities { u p s : p ∈ [ r ] , s ∈ S } . A mixed strategy profile of G is given by probabilities { x p j : p ∈ [ r ] , j ∈ S p } . GG will contain a vertex v ( x p j ) ∈ V for each player p and strategy j ∈ S p , and the construction of GG will ensure that in any Nash equilibrium of GG , the quantities { p [ v ( x p j )] : p ∈ [ r ] , j ∈ S p } , if interpreted as values { x p j } p,j , will constitute a Nash equilibrium of G . Extending this notation, for various arithmetic expressions A involving any x p j and u p s , vertex v ( A ) ∈ V will be used, and be constructed such that in any Nash equilibrium of GG , p [ v ( A )] is equal to A evaluated at the given values of u p s and with x p j equal to p [ v ( x p j )]. Elements of

W are used to mediate between elements of V , so that the latter ones obey the intended arithmetic relationships.

We use Propositions (1-4) as building blocks of GG , starting with r subgraphs that represent mixed strategies for the players of G . In the following, we construct a graphical game containing vertices { v ( x p j ) } j ∈ [ n ] , whose probabilities sum to 1, and internal vertices v p j , which control the distribution of the one unit of probability mass among the vertices v ( x p j ). See Figure 8 for an illustration.

## Proposition 5 Consider a graphical game that contains

- for j ∈ [ n ] a vertex v ( x p j )
- for j ∈ [ n -1] a vertex v p j
- for j ∈ [ n -1] a vertex w j ( p ) used to ensure p [ v ( ∑ j i =1 x p i )] = p [ v ( ∑ j +1 i =1 x p i )](1 -p [ v p j ])
- for j ∈ [ n ] a vertex v ( ∑ j i =1 x p i )
- for j ∈ [ n -1] a vertex w ′ j ( p ) used to ensure p [ v ( x p j +1 )] = p [ v ( ∑ j +1 i =1 x p i )] p [ v p j ]

Also, let v ( ∑ n i =1 x p i ) have payoff of 1 when it plays 1 and 0 otherwise. Then, in any Nash equilibrium of the graphical game, ∑ n i =1 p [ v ( x p i )] = 1 and moreover p [ v ( ∑ j i =1 x p i )] = ∑ j i =1 p [ v ( x p i )] , and the graph is bipartite and of degree 3.

- a vertex w ′ 0 ( p ) used to ensure p [ v ( x p 1 )] = p [ v ( ∑ 1 i =1 x p i )]

Proof. It is not hard to verify that the graph has degree 3. Most of the degree 3 vertices are the w vertices used in Propositions 1 and 2 to connect the pairs or triples of graph players whose probabilities are supposed to obey an arithmetic relationship. In a Nash equilibrium, v ( ∑ n i =1 x p i ) plays 1. The vertices v p j split the probability p [ v ( ∑ j +1 i =1 x p i )] between p [ v ( ∑ j i =1 x p i )] and p [ v ( x p j +1 )].

Comment. The values p [ v p j ] control the distribution of probability (summing to 1) amongst the n vertices v ( x p j ). These vertices can set to zero any proper subset of the probabilities p [ v ( x p j )].

Notation. For s ∈ S -p let x s = x 1 s 1 · x 2 s 2 · · · x p -1 s p -1 · x p +1 s p +1 · · · x r s r . Also, let U p j = ∑ s ∈ S -p u p js x s be the utility to p for playing j in the context of a given mixed profile { x s } s ∈ S -p .

Lemma 8 Suppose all utilities u p s (of G ) lie in the range [0 , 1] for some p ∈ [ r ] . We can construct a degree 3 bipartite graph having a total of O ( rn r ) vertices, including vertices v ( x p j ) , v ( U p j ) , v ( U p ≤ j ) , for all j ∈ [ n ] , such that in any Nash equilibrium,

/negationslash

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

Figure 8: Diagram of Proposition 5

<!-- image -->

The general idea is to note that the expressions for p [ v ( U p j )] and p [ v ( U p ≤ j )] are constructed from arithmetic subexpressions using the operations of addition, multiplication and maximization. If each subexpression A has a vertex v ( A ), then using Propositions 1 through 4 we can assemble them into a graphical game such that in any Nash equilibrium, p [ v ( A )] is equal to the value of A with input p [ v ( x p j )], p ∈ [ r ], j ∈ [ n ]. We just need to limit our usage to O ( rn r ) subexpressions and ensure that their values all lie in [0 , 1].

## Proof. Note that

<!-- formula-not-decoded -->

Let S -p = { S -p (1) , . . . , S -p ( n r -1 ) } , so that

<!-- formula-not-decoded -->

/negationslash

For each partial sum ∑ z /lscript =1 u p jS -p ( /lscript ) x S -p ( /lscript ) , 1 ≤ z ≤ n r -1 , include vertex v ( ∑ z /lscript =1 u p jS -p ( /lscript ) x S -p ( /lscript ) ). Similarly, for each partial product of the summands u p js ∏ p = q ≤ z x q s q , 0 ≤ z ≤ r , include vertex v ( u p js ∏ p = q ≤ z x q s q ). So, for each strategy j ∈ S p , there are n r -1 partial sums and r + 1 partial products for each summand. Then, there are n partial sequences over which we have to maximize. Note that, since all utilities are assumed to lie in the set [0 , 1], all partial sums and products must also lie in [0 , 1], so the truncation at 1 in the computations of Propositions 1, 2, 3 and 4 is not a problem. So using a vertex for each of the 2 n + ( r + 1) n r arithmetic subexpressions, a Nash equilibrium will compute the desired quantities.

/negationslash

We repeat the construction specified by Lemma 8 for all p ∈ [ r ]. Note that, to avoid large degrees in the resulting graphical game, each time we need to make use of a value x q s q we create a new copy of the vertex v ( x q s q ) using the gadget G = and, then, use the new copy for the computation of the desired partial product; an easy calculation shows that we have to make ( r -1) n r -1 copies of v ( x q s q ), for all q ≤ r , s q ∈ S q . To limit the degree of each vertex to 3 we create a binary tree of copies of v ( x q s q ) with ( r -1) n r -1 leaves and use each leaf once.

Proof of Theorem 6: Let G be a r -player normal-form game with n strategies per player and construct GG = f ( G ) as shown in Figure 9. The graph of GG has degree 3, by the graph structure of our gadgets from Propositions 1 through 4 and the fact that we use separate copies of the v ( x p j ) vertices to influence different v ( U p j ) vertices (see Step 4 and discussion after Lemma 8).

## Polynomial size of GG = f ( G ) :

The size of GG is polynomial in the description length r · n r q of G , where q is the size of the values in the payoff tables in the logarithmic cost model.

## Construction of g ( N GG ) (where N GG denotes a Nash equilibrium of GG ):

Given a Nash equilibrium g ( N GG ) of GG , we claim that we can recover a Nash equilibrium { x p j } p,j of G by taking x p j = p [ v ( x p j )]. This is clearly computable in polynomial-time.

## Proof that the reduction preserves Nash equilibria:

Call G ′ the game resulting from G by rescaling the utilities so that they lie in the range [0 , 1]. It is easy to see that any Nash equilibrium of game G is, also, a Nash equilibrium of game G ′ and vice versa. Therefore, it is enough to establish that the mapping g ( · ) maps every Nash equilibrium

Input: Normal form game G with r players, n strategies per player, utilities { u p s : p ∈ [ r ] , s ∈ S } . Output: Graphical game GG with bipartite graph ( V ∪ W,E ).

1. If needed, rescale the utilities u p s so that they lie in the range [0 , 1]. One way to do so is to divide all utilities by max { u p s } .
2. For each player/strategy pair ( p, j ) let v ( x p j ) ∈ V be a vertex in GG .
3. For each p ∈ [ r ] construct a subgraph as described in Proposition 5 so that in a Nash equilibrium of GG , we have ∑ j p [ v ( x p j )] = 1.
4. Use the construction of Proposition 1 with α = 1 to make ( r -1) n r -1 copies of the v ( x p j ) vertices (which are added to V ). More precisely, create a binary tree with copies of v ( x p j ) which has ( r -1) n r -1 leaves.
5. Use the construction of Lemma 8 to introduce (add to V ) vertices v ( U p j ), v ( U p ≤ j ), for all p ∈ [ r ], j ∈ [ n ]. Each v ( U p j ) uses its own set of copies of the vertices v ( x p j ). For p ∈ [ r ], j ∈ [ n ] introduce (add to W ) w ( U p j ) with
6. (a) If w ( U p j ) plays 0 then w ( U p j ) gets payoff 1 whenever v ( U p ≤ j ) plays 1, else 0.
7. (b) If w ( U p j ) plays 1 then w ( U p j ) gets payoff 1 whenever v ( U p j +1 ) plays 1, else 0.
6. Give the following payoffs to the vertices v p j (the additional vertices used in Proposition 5 whose payoffs were not specified).
9. (a) If v p j plays 0 then v p j has a payoff of 1 whenever w ( U p j ) plays 0, otherwise 0.
10. (b) If v p j plays 1 then v p j has a payoff of 1 whenever w ( U p j ) plays 1, otherwise 0.
7. Return the underlying undirected graphical game GG .

Figure 9: Reduction from normal form game G to graphical game GG

of game GG to a Nash equilibrium of game G ′ . By Proposition 5, we have that ∑ j x p j = 1, for all p ∈ [ r ]. It remains to show that, for all p , j , j ′ ,

<!-- formula-not-decoded -->

We distinguish the cases:

- If there exists some j ′′ &lt; j ′ such that ∑ s ∈ S -p u p j ′′ s x s &gt; ∑ s ∈ S -p u p j ′ s x s , then, by Lemma 8, p [ v ( U p ≤ j ′ -1 )] &gt; p [ v ( U p j ′ )]. Thus, p [ v p j ′ -1 ] = 0 and, consequently, v ( x p j ′ ) plays 0 as required, since
- The case j &lt; j ′ reduces trivially to the previous case.

<!-- formula-not-decoded -->

- It remains to deal with the case j &gt; j ′ , under the assumption that, for all j ′′ &lt; j ′ , ∑ s ∈ S -p u p j ′′ s x s ≤

∑ s ∈ S -p u p j ′ s x s , or, equivalently, which in turn implies that

It follows that there exists some k , j ′ + 1 ≤ k ≤ j , such that p [ v ( U p k )] &gt; p [ v ( U p ≤ k -1 )]. Otherwise, p [ v ( U p ≤ j ′ )] ≥ p [ v ( U p ≤ j ′ +1 )] ≥ . . . ≥ p [ v ( U p ≤ j )] ≥ p [ v ( U p j )] &gt; p [ v ( U p j ′ )], which is a contradiction to p [ v ( U p ≤ j ′ )] ≤ p [ v ( U p j ′ )]. Since p [ v ( U p k )] &gt; p [ v ( U p ≤ k -1 )], it follows that p [ w ( U p k -1 )] = 1 ⇒ p [ v p k -1 ] = 1 and, therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Mapping g is surjective on the Nash equilibria of G ′ and, therefore, G : We will show that given a Nash equilibrium N G ′ of G ′ there is a Nash equilibrium N GG of GG such that g ( N GG ) = N G ′ . Let N G ′ = { x p j : p ≤ r, j ∈ S p } . In N GG , let p [ v ( x p j )] = x p j . Lemma 8 shows that the values p [ v ( U p j )] are the expected utilities to player p for playing strategy j , given that all other players use the mixed strategy { x p j : p ≤ r, j ∈ S p } . We identify values for p [ v p j ] that complete a Nash equilibrium for GG .

Based on the payoffs to v p j described in Figure 9 we have

- If p [ v ( U p ≤ j )] &gt; p [ v ( U p j +1 )] then p [ w ( U p j )] = 0; p [ v p j ] = 0;
- If p [ v ( U p ≤ j )] &lt; p [ v ( U p j +1 )] then p [ w ( U p j )] = 1; p [ v p j ] = 1;
- If p [ v ( U p ≤ j )] = p [ v ( U p j +1 )] then choose p [ w ( U p j )] = 1 2 ; p [ v p j ] is arbitrary (we may assign it any value)

Given the above constraints on the values p [ v p j ] we must check that we can choose them (and there is a unique choice) so as to make them consistent with the probabilities p [ v ( x p j )]. We use the fact the values x p j form a Nash equilibrium of G . In particular, we know that p [ v ( x p j )] = 0 if there exists j ′ with U p j ′ &gt; U p j . We claim that for j satisfying p [ v ( U p ≤ j )] = p [ v ( U p j +1 )], if we choose

<!-- formula-not-decoded -->

then the values p [ v ( x p j )] are consistent. ✷

## 4.4 Combining the Reductions

Suppose that we take either a graphical or a normal-form game, and apply to it both of the reductions described in the previous sections. Then we obtain a game of the same type and a surjective mapping from the Nash equilibria of the latter to the Nash equilibria of the former.

Corollary 2 For any fixed d , a (directed or undirected) graphical game of maximum degree d can be mapped in polynomial time to an undirected graphical game of maximum degree 3 so that there is a polynomial-time computable surjective mapping g from the Nash equilibria of the latter to the Nash equilibria of the former.

The following also follows directly from Theorems 6 and 5, but is not as strong as Theorem 7 below.

Corollary 3 For any fixed r &gt; 1 , a r -player normal form game can be mapped in polynomial time to a 10 -player normal form game so that there is a polynomial-time computable surjective mapping g from the Nash equilibria of the latter to the Nash equilibria of the former.

Proof. Theorem 6 converts a r -player game G into a graphical game GG based on a graph of degree 3. Theorem 5 converts GG to a 10-player game G ′ , whose Nash equilibria encode the Nash equilibria of GG and hence of G . (Note that for d an odd number, the proof of Theorem 5 implies a reduction to a ( d 2 +1)-player normal form game.)

We next prove a stronger result, by exploiting in more detail the structure of the graphical games GG constructed in the proof of Theorem 6. The technique used here will be used in Section 4.5 to strengthen the result even further.

Theorem 7 For any fixed r &gt; 1 , a r -player normal form game can be mapped in polynomial time to a 4 -player normal form game so that there is a polynomial-time computable surjective mapping g from the Nash equilibria of the latter to the Nash equilibria of the former.

Proof. Construct G ′ from G as shown in Figure 10.

By Theorem 6, GG (as constructed in Figure 10) is of polynomial size. The size of GG ′ is at most 3 times the size of GG since we do not need to apply Step 3 to any edges that are themselves constructed by an earlier iteration of Step 3. Finally, the size of G ′ is polynomial in the size of GG ′ from Theorem 5.

## Polynomial size of G ′ = f ( G ) .

## Construction of g ( N G ′ ) (for N G ′ a Nash equilibrium of G ′ ).

Let g 1 be a surjective mapping from the Nash equilibria of GG to the Nash equilibria of G , which is guaranteed to exist by Theorem 6. It is trivial to construct a surjective mapping g 2 from the Nash equilibria of GG ′ to the Nash equilibria of GG . By Theorem 5, there exists a surjective mapping g 3 from the Nash equilibria of G ′ to the Nash equilibria of GG ′ . Therefore, g 3 ◦ g 2 ◦ g 1 is a surjective mapping from the Nash equilibria of G ′ to the Nash equilibria of G .

## 4.5 Reducing to Three Players

We will strengthen Theorem 7 to reduce a r -player normal form game to a 3-player normal form game. The following theorem together with Theorems 5 and 6 imply the first part of Theorem 4.

Theorem 8 For any fixed r &gt; 1 , a r -player normal form game can be mapped in polynomial time to a 3 -player normal form game so that there is a polynomial-time computable surjective mapping g from the Nash equilibria of the latter to the Nash equilibria of the former.

Proof. The bottleneck of the construction of Figure 10 in terms of the number k of players of the resulting normal form game G ′ lies entirely on the ability or lack thereof to color the vertices of the affects graphs of GG with k colors so that, for every vertex v , its neighborhood N ( v ) in the

Input: Normal form game G with r players, n strategies per player, utilities { u p s : p ≤ r, s ∈ S } . Output: 4-player Normal form game G ′ .

1. Let GG be the graphical game constructed from G according to Figure 9. Recall that the affects graph G = ( V ∪ W,E ) of GG has the following properties:
- Every edge e ∈ E is from a vertex of set V to a vertex of set W or vice versa.
- Every vertex of set W has indegree at most 3 and outdegree at most 1 and every vertex of set V has indegree at most 1 and outdegree at most 2.
2. Color the graph ( V ∪ W,E ) of GG as follows: let c ( w ) = 1 for all W -vertices w and c ( v ) = 2 for all V -vertices v .
3. Construct a new graphical game GG ′ from GG as follows. While there exist v 1 , v 2 ∈ V , w ∈ W , ( v 1 , w ) , ( v 2 , w ) ∈ E with c ( v 1 ) = c ( v 2 ):
6. (a) Every W -vertex has at most 1 outgoing edge, so assume ( w,v 1 ) /negationslash∈ E .
7. (b) Add v ( v 1 ) to V , add w ( v 1 ) to W .
8. (c) Replace ( v 1 , w ) with ( v 1 , w ( v 1 )), ( w ( v 1 ) , v ( v 1 )), ( v ( v 1 ) , w ( v 1 )), ( v ( v 1 ) , w ). Let c ( w ( v 1 )) = 1, choose c ( v ( v 1 )) ∈ { 2 , 3 , 4 } /negationslash = c ( v ′ ) for any v ′ with ( v ′ , w ) ∈ E . Payoffs for w ( v 1 ) and v ( v 1 ) are chosen using Proposition 1 with α = 1 such that in any Nash equilibrium, p [ v ( v 1 )] = p [ v 1 ].
4. The coloring c : V ∪ W →{ 1 , 2 , 3 , 4 } has the property that, for every vertex v of GG ′ , its neighborhood N ( v ) in the affects graph of the game -recall it consists of v and all its predecessors- is colored with |N ( v ) | distinct colors. Rescale all utilities of GG ′ to [0,1] and map game GG ′ to a 4-player normal form game G ′ following the steps 3 through 5 of figure 7.

Figure 10: Reduction from normal form game G to 4-player game G ′

affects graph is colored with |N ( v ) | distinct colors, i.e. on whether there exists a legal k coloring. In Figure 10, we show how to design a graphical game GG ′ which is equivalent to GG -in the sense that there exists a surjective mapping from the Nash equilibria of the former to the Nash equilibria of the latter- and can be legally colored using 4 colors. However, this cannot be improved to 3 colors since the addition game G + and the multiplication game G ∗ , which are essential building blocks of GG , have vertices with indegree 3 (see Figure 5) and, therefore, need at least 4 colors to be legally colored. Therefore, to improve our result we need to redesign addition and multiplication games which can be legally colored using 3 colors.

Notation: In the following,

- x = y ± /epsilon1 denotes y -/epsilon1 ≤ x ≤ y + /epsilon1
- v : s denotes 'player v plays strategy s '

Proposition 6 Let α, β, γ be non-negative integers such that α + β + γ ≤ 3 . There is a graphical game G + , ∗ with two 'input players' v 1 and v 2 , one 'output player' v 3 and several intermediate players, with the following properties:

Payoffs to w 1 :

<!-- formula-not-decoded -->

2. Game played by players v 2 ′ , w 3 , v 3 :

<!-- formula-not-decoded -->

Payoffs to w 3 :

<!-- formula-not-decoded -->

Figure 11: The new addition/multiplication game and its legal 3-coloring.

<!-- image -->

- the graph of the game can be legally colored using 3 colors
- for any /epsilon1 ∈ [0 , 0 . 01] , at any /epsilon1 -Nash equilibrium of game G + , ∗ it holds that p [ v 3 ] = min { 1 , α p [ v 1 ]+ β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] } ± 81 /epsilon1 ; in particular at any Nash equilibrium p [ v 3 ] = min { 1 , α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] } .

Proof. The graph of the game and the labeling of the vertices is shown in Figure 11. All players of G + , ∗ have strategy set { 0 , 1 } except for player v ′ 2 who has three strategies { 0 , 1 , ∗} . Below we give the payoff tables of all the players of the game. For ease of understanding we partition the game G + , ∗ into four subgames:

1. Game played by players v 1 , w 1 , v ′ 1 :

<!-- formula-not-decoded -->

3. Game played by players v 2 , w 2 , v ′ 2 :

Payoffs to w 2 :

<!-- formula-not-decoded -->

Payoffs to v ′ 2 :

<!-- formula-not-decoded -->

4. Game played by players v ′ 1 , v ′ 2 , w, u :

Payoffs to w :

<!-- formula-not-decoded -->

Payoffs to u :

<!-- formula-not-decoded -->

Claim 1 At any /epsilon1 -Nash equilibrium of G + , ∗ : p [ v ′ 1 ] = 1 8 p [ v 1 ] ± /epsilon1 .

Proof. If w 1 plays 0, then the expected payoff to w 1 is 1 8 p [ v 1 ], whereas if w 1 plays 1, the expected payoff to w 1 is p [ v ′ 1 ]. Therefore, in an /epsilon1 -Nash equilibrium, if 1 8 p [ v 1 ] &gt; p [ v ′ 1 ] + /epsilon1 , then p [ w 1 ] = 0. However, note also that if p [ w 1 ] = 0 then p [ v ′ 1 ] = 1, which is a contradiction to 1 8 p [ v 1 ] &gt; p [ v ′ 1 ] + /epsilon1 . Consequently, 1 8 p [ v 1 ] cannot be strictly larger than p [ v ′ 1 ]+ /epsilon1 . On the other hand, if p [ v ′ 1 ] &gt; 1 8 p [ v 1 ]+ /epsilon1 , then p [ w 1 ] = 1 and consequently p [ v ′ 1 ] = 0, a contradiction. The claim follows from the above observations.

Claim 2 At any /epsilon1 -Nash equilibrium of G + , ∗ : p [ v ′ 2 : 1] = 1 8 p [ v 2 ] ± /epsilon1 .

Proof. If w 2 plays 0, then the expected payoff to w 2 is 1 8 p [ v 2 ], whereas, if w 2 plays 1, the expected payoff to w 2 is p [ v ′ 2 : 1].

If, in an /epsilon1 -Nash equilibrium, 1 8 p [ v 2 ] &gt; p [ v ′ 2 : 1] + /epsilon1 , then p [ w 2 ] = 0. In this regime, the payoff to player v ′ 2 is 0 if v ′ 2 plays 0, 1 if v ′ 2 plays 1 and 0 if v ′ 2 plays ∗ . Therefore, p [ v ′ 2 : 1] = 1 and this contradicts the hypothesis that 1 8 p [ v 2 ] &gt; p [ v ′ 2 : 1] + /epsilon1 .

On the other hand, if, in an /epsilon1 -Nash equilibrium, p [ v ′ 2 : 1] &gt; 1 8 p [ v 2 ] + /epsilon1 , then p [ w 2 ] = 1. In this regime, the payoff to player v ′ 2 is p [ u : 0] if v ′ 2 plays 0, 0 if v ′ 2 plays 1 and p [ u : 1] if v ′ 2 plays ∗ . Since p [ u : 0] + p [ u : 1] = 1, it follows that p [ v ′ 2 : 1] = 0 because at least one of p [ u : 0], p [ u : 1] will be greater than /epsilon1 . This contradicts the hypothesis that p [ v ′ 2 : 1] &gt; 1 8 p [ v 2 ] + /epsilon1 and the claim follows from the above observations.

Claim 3 At any /epsilon1 -Nash equilibrium of G + , ∗ : p [ v ′ 2 : ∗ ] = α 8 p [ v 1 ] + β 8 p [ v 2 ] + γ 8 p [ v 1 ] p [ v 2 ] ± 10 /epsilon1 .

Proof. If w plays 0, then the expected payoff to w is α p [ v ′ 1 ] + (1 + β ) p [ v ′ 2 : 1] + 8 γ p [ v ′ 1 ] p [ v ′ 2 : 1], whereas, if w plays 1, the expected payoff to w is p [ v ′ 2 : 1] + p [ v ′ 2 : ∗ ].

On the other hand, if, in a /epsilon1 -Nash equilibrium, p [ v ′ 2 : 1] + p [ v ′ 2 : ∗ ] &gt; α p [ v ′ 1 ] + (1 + β ) p [ v ′ 2 : 1]+8 γ p [ v ′ 1 ] p [ v ′ 2 : 1]+ /epsilon1 , then p [ w ] = 1 and consequently p [ u ] = 0. In this regime, the payoff to player v ′ 2 is p [ w 2 : 1] if v ′ 2 plays 0, p [ w 2 : 0] if v ′ 2 plays 1 and 0 if v ′ 2 plays ∗ . Since p [ w 2 : 0] + p [ w 2 : 1] = 1, it follows that p [ v ′ 2 : ∗ ] = 0. So the hypothesis can be rewritten as 0 &gt; α p [ v ′ 1 ] + β p [ v ′ 2 : 1] + 8 γ p [ v ′ 1 ] p [ v ′ 2 : 1] + /epsilon1 which is a contradiction.

If, in a /epsilon1 -Nash equilibrium, α p [ v ′ 1 ] + (1 + β ) p [ v ′ 2 : 1] + 8 γ p [ v ′ 1 ] p [ v ′ 2 : 1] &gt; p [ v ′ 2 : 1] + p [ v ′ 2 : ∗ ] + /epsilon1 , then p [ w ] = 0 and, consequently, p [ u ] = 1. In this regime, the payoff to player v ′ 2 is 0 if v ′ 2 plays 0, p [ w 2 : 0] if v ′ 2 plays 1 and p [ w 2 : 1] if v ′ 2 plays ∗ . Since p [ w 2 : 0] + p [ w 2 : 1] = 1, it follows that at least one of p [ w 2 : 0], p [ w 2 : 1] will be larger than /epsilon1 so that p [ v ′ 2 : 0] = 0 or, equivalently, that p [ v ′ 2 : 1] + p [ v ′ 2 : ∗ ] = 1. So the hypothesis can be rewritten as α p [ v ′ 1 ] + (1 + β ) p [ v ′ 2 : 1] + 8 γ p [ v ′ 1 ] p [ v ′ 2 : 1] &gt; 1 + /epsilon1 . Using Claims 1 and 2 and the fact that /epsilon1 ≤ 0 . 01 this inequality implies α 8 p [ v 1 ] + 1+ β 8 p [ v 2 ] + γ 8 p [ v 1 ] p [ v 2 ] + ( α +1 + β + 3 γ ) /epsilon1 &gt; 1 + /epsilon1 and further that α +1+ β + γ 8 +( α +1+ β +3 γ ) /epsilon1 &gt; 1 + /epsilon1 . We supposed α + β + γ ≤ 3 therefore the previous inequality implies 1 2 +10 /epsilon1 &gt; 1 + /epsilon1 , a contradiction since we assumed /epsilon1 ≤ 0 . 01.

Therefore, in any /epsilon1 -Nash equilibrium, p [ v ′ 2 : 1]+ p [ v ′ 2 : ∗ ] = α p [ v ′ 1 ]+(1+ β ) p [ v ′ 2 : 1]+8 γ p [ v ′ 1 ] p [ v ′ 2 : 1] ± /epsilon1 , or, equivalently, p [ v ′ 2 : ∗ ] = α p [ v ′ 1 ] + β p [ v ′ 2 : 1] + 8 γ p [ v ′ 1 ] p [ v ′ 2 : 1] ± /epsilon1 . Using claims 1 and 2 this can be restated as p [ v ′ 2 : ∗ ] = α 8 p [ v 1 ] + β 8 p [ v 2 ] + γ 8 p [ v 1 ] p [ v 2 ] ± 10 /epsilon1

Claim 4 At any /epsilon1 -Nash equilibrium of G + , ∗ : p [ v 3 ] = min { 1 , α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] } ± 81 /epsilon1 .

Proof. If w 3 plays 0, the expected payoff to w 3 is 8 p [ v ′ 2 : ∗ ], whereas, if w 3 plays 1, the expected payoff to w 3 is p [ v 3 ]. Therefore, in a /epsilon1 -Nash equilibrium, if p [ v 3 ] &gt; 8 p [ v ′ 2 : ∗ ] + /epsilon1 , then p [ w 3 ] = 1 and, consequently, p [ v 3 ] = 0, which is a contradiction to p [ v 3 ] &gt; 8 p [ v ′ 2 : ∗ ] + /epsilon1 .

From the above observations it follows that p [ v 3 ] = min { 1 , 8 p [ v ′ 2 : ∗ ] } ± /epsilon1 and, using claim 3, p [ v 3 ] = min { 1 , α p [ v 1 ] + β p [ v 2 ] + γ p [ v 1 ] p [ v 2 ] } ± 81 /epsilon1 .

On the other hand, if 8 p [ v ′ 2 : ∗ ] &gt; p [ v 3 ] + /epsilon1 , then p [ w 3 ] = 0 and consequently p [ v 3 ] = 1. Hence, p [ v 3 ] cannot be less than min { 1 , 8 p [ v ′ 2 : ∗ ] -/epsilon1 } .

It remains to show that the graph of the game can be legally colored using 3 colors. The coloring is shown in Figure 11.

Now that we have our hands on the game G + , ∗ of Proposition 6, we can reduce r -player games to 3-player games, for any fixed r , using the algorithm of Figure 10 with the following tweak: in the construction of game GG at Step 1 of the algorithm, instead of using the addition and multiplication gadgets G + , G ∗ of Section 4.1, we use our more elaborate G + , ∗ gadget. Let us call the resulting game GG . We will show that we can construct a graphical game GG ′ which is equivalent to GG in the sense that there is a surjective mapping from the Nash equilibria of GG ′ to the Nash equilibria of GG and which, moreover, can be legally colored using three colors. Then we can proceed as in Step 4 of Figure 10 to get the desired 3-player normal form game G ′ .

<!-- image -->

G

Figure 12: The interposition of two G = games between gadgets G 1 and G 2 does not change the game.

The construction of GG ′ and its coloring can be done as follows: Recall that all our gadgets have some distinguished vertices which are the inputs and one distinguished vertex which is the output . The gadgets are put together to construct GG by identifying the output vertices of some gadgets as the input vertices of other gadgets. It is easy to see that we get a graphical game with the same functionality if, instead of identifying the output vertex of some gadget with the input of another gadget, we interpose a sequence of two G = games between the two gadgets to be connected, as shown in Figure 12. If we 'glue' our gadgets in this way then the resulting graphical game GG ′ can be legally colored using 3 colors:

- i. (stage 1) legally color the vertices inside the 'initial gadgets' using 3 colors
- ii. (stage 2) extend the coloring to the vertices that serve as 'connections' between gadgets; any 3-coloring of the initial gadgets can be extended to a 3-coloring of GG ′ because, for any pair of gadgets G 1 , G 2 which are connected (Figure 12) and for any colors assigned to the output vertex a of gadget G 1 and the input vertex e of gadget G 2 , the intermediate vertices b , c and d can be also colored legally. For example, if vertex a gets color 1 and vertex e color 2 at stage 1, then, at stage 2, b can be colored 2, c can be colored 3 and d can be colored 1.

This completes the proof of the theorem.

## 4.6 Preservation of Approximate equilibria

Our reductions so far map exact equilibrium points. In this section we generalize to approximate equilibria and prove the second part of Theorem 4. We claim that the reductions of the previous sections translate the problem of finding an /epsilon1 -Nash equilibrium of a game to the problem of finding an /epsilon1 ′ -Nash equilibrium of its image, for /epsilon1 ′ polynomial in /epsilon1 and inverse polynomial in the size of the game. As a consequence, we obtain polynomial-time equivalence results for the problems r -Nash and d -graphical-Nash . To prove the second part of Theorem 4, we extend Theorems 5, 6 and 8 of the previous sections.

Theorem 9 For every fixed d &gt; 1 , there is a polynomial-time reduction from d -graphical-Nash to ( d 2 +1) -Nash .

Proof. Let ˜ GG be a graphical game of maximum degree d and GG the resulting graphical game after rescaling all utilities by 1 / max { ˜ u } , where max { ˜ u } is the largest entry in the utility tables of game ˜ GG , so that they lie in the set [0 , 1], as in the first step of Figure 7. Assume that /epsilon1 &lt; 1.

In time polynomial in |GG| +log(1 //epsilon1 ), we will specify a normal form game G and an accuracy /epsilon1 ′ with the property that, given an /epsilon1 ′ -Nash equilibrium of G , one can recover in polynomial time an /epsilon1 -Nash equilibrium of GG . This will be enough, since an /epsilon1 -Nash equilibrium of GG is trivially an /epsilon1 · max { ˜ u } -Nash equilibrium of game GG and, moreover, |GG| is polynomial in | GG| .

Suppose that p = c ( v ) for some vertex v of the graphical game GG . As in the proof of Theorem 5, Lemma 7, it can be shown that in any /epsilon1 ′ -Nash equilibrium of the game G ,

˜ ˜ We construct G using the algorithm of Figure 7; recall that M ≥ 2 n r , where r is the number of color classes specified in Figure 7 and n is the number of vertices in GG after the possible addition of dummy vertices to make sure that all color classes have the same number of vertices (as in Step 3 of Figure 7). Let us choose /epsilon1 ′ ≤ /epsilon1 ( r n -1 M ) d ; we will argue that from any /epsilon1 ′ -Nash equilibrium of game G one can construct in polynomial time an /epsilon1 -Nash equilibrium of game GG .

<!-- formula-not-decoded -->

Now, without loss of generality, assume that p is odd (pursuer) and suppose that v is vertex v ( p ) i in the notation of Figure 7. Then, in an /epsilon1 ′ -Nash equilibrium of the game G , we have, by the definition of a Nash equilibrium, that for all strategies a, a ′ ∈ S v of vertex v :

But

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and similarly for a ′ . Therefore, the previous inequality implies

<!-- formula-not-decoded -->

So letting

<!-- formula-not-decoded -->

as we did in the proof of Theorem 5, we get that, for all v ∈ V , a, a ′ ∈ S v ,

<!-- formula-not-decoded -->

where T = ∏ u ∈N ( v ) \{ v } ∑ j ∈ S u x c ( u ) ( u,j ) = ∏ u ∈N ( v ) \{ v } Pr[ c ( u ) plays u ] ≥ ( r n -1 M ) d . By the definition of /epsilon1 ′ it follows that /epsilon1 ′ / T ≤ /epsilon1 . Hence, from (13) it follows that { x v a } v,a is an /epsilon1 -Nash equilibrium of the game GG .

We have the following extension of Theorem 6.

Theorem 10 For every fixed r &gt; 1 , there is a polynomial-time reduction from r -Nash to 3 -graphical Nash with two strategies per vertex.

Proof. Let ˜ G be a normal form game with r players, 1 , 2 , . . . , r , and strategy sets S p = [ n ], for all p ∈ [ r ], and let { ˜ u p s : p ∈ [ r ] , s ∈ S } be the utilities of the players. Denote by G the game constructed at the first step of Figure 9 which results from ˜ G after rescaling all utilities by 1 / max { ˜ u p s } so that they lie in [0 , 1]; let { u p s : p ∈ [ r ] , s ∈ S } be the utilities of the players in game G . Also, let /epsilon1 &lt; 1. In time polynomial in |G| +log(1 //epsilon1 ), we will specify a graphical game GG and an accuracy /epsilon1 ′ with the property that, given an /epsilon1 ′ -Nash equilibrium of GG , one can recover in polynomial time an /epsilon1 -Nash equilibrium of G . This will be enough, since an /epsilon1 -Nash equilibrium of G is trivially an /epsilon1 · max { ˜ u p s } -Nash equilibrium of game ˜ G and, moreover, |G| is polynomial in | ˜ G| . In our reduction, the graphical game GG will be the same as the one described in the proof of Theorem 6 (Figure 9), while the accuracy specification will be of the form /epsilon1 ′ = /epsilon1/p ( |G| ), where p ( · ) is a polynomial that will be be specified later. We will use the same labels for the vertices of the game GG that we used in the proof Theorem 6.

Suppose N GG is some /epsilon1 ′ -Nash equilibrium of the game GG and let { p [ v ( x p j )] } j,p denote the probabilities with which the vertices v ( x p j ) of GG play strategy 1. In the proof of Theorem 6 we considered the following mapping from the Nash equilibria of game GG to the Nash equilibria of game G :

<!-- formula-not-decoded -->

Although (14) succeeds in mapping exact equilibrium points, it fails for approximate equilibria, as specified by the following remark -its justification follows from the proof of Lemma 9.

/negationslash

Remark 2 For any /epsilon1 ′ &gt; 0 , there exists an /epsilon1 ′ -Nash equilibrium of game GG such that ∑ j p [ v ( x p j )] = 1 , for some player p ≤ r , and, moreover, p [ v ( U p j )] &gt; p [ v ( U p j ′ )] + /epsilon1 ′ , for some p ≤ r , j and j ′ , and, yet, p [ v ( x p j ′ )] &gt; 0 .

/negationslash

Recall from Section 4.3, that, for all p , j , the probability p [ v ( U p j )] represents the utility of player p for playing pure strategy j , when the other players play according to { x q j := p [ v ( x q j )] } j,q = p 5 . Therefore, not only the { x p j := p [ v ( x p j )] } j do not necessarily constitute a distribution -this could be easily fixed by rescaling- but, also, the defining property of an approximate equilibrium (2) is in question. The following lemma bounds the deviation from the approximate equilibrium conditions.

Lemma 9 In any /epsilon1 ′ -Nash equilibrium of the game GG ,

- (ii) for all p ∈ [ r ] , j, j ′ ∈ [ n ] , p [ v ( U p j )] &gt; p [ v ( U p j ′ )] + 5 cn/epsilon1 ′ ⇒ p [ v ( x p j ′ )] ∈ [0 , cn/epsilon1 ′ ] ,
- (i) for all p ∈ [ r ] , | ∑ j p [ v ( x p j )] -1 | ≤ 2 cn/epsilon1 ′ , and,

where c ≥ 1 is the maximum error amplification of the gadgets used in the construction of GG .

Proof. Note that at an /epsilon1 ′ -Nash equilibrium of game GG the following properties are satisfied for all p ∈ [ r ] by the vertices of game GG , since the error amplification of the gadgets is at most c :

5 Note, however, that, since we are considering an /epsilon1 ′ -Nash equilibrium of game GG , the Equation (11) of Section 4.3 will be only satisfied approximately as specified by Lemma 11.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of (i): By successive applications of (16) and (17), we deduce

<!-- formula-not-decoded -->

Proof of (ii): Let us first observe the behavior of vertices w ( U p j ) and v p j in an /epsilon1 ′ -Nash equilibrium.

- Behavior of w ( U p j ) vertices: The utility of vertex w ( U p j ) for playing strategy 0 is p [ v ( U p ≤ j )], whereas for playing 1 it is p [ v ( U p j +1 )]. Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Behavior of v p j vertices: The utility of vertex v p j for playing strategy 0 is 1 -p [ w ( U p j )], whereas for playing 1 it is p [ w ( U p j )]. Therefore,

<!-- formula-not-decoded -->

Note that, since the error amplification of the gadget G max is at most c and computing p [ v ( U p ≤ j )], for all j , requires j applications of G max ,

<!-- formula-not-decoded -->

To establish the second part of the claim, we need to show that, for all p , j , j ′ ,

<!-- formula-not-decoded -->

1. Note that, if there exists some j ′′ &lt; j ′ such that p [ v ( U p j ′′ )] &gt; p [ v ( U p j ′ )] + c/epsilon1 ′ n , then

<!-- formula-not-decoded -->

Then, because p [ v ( U p ≤ j ′ -1 )] &gt; p [ v ( U p j ′ )] + /epsilon1 ′ , it follows that p [ w ( U p j ′ -1 )] = 0 and p [ v p j ′ -1 ] = 0. Therefore,

<!-- formula-not-decoded -->

2. The case j &lt; j ′ reduces to the previous for j ′′ = j .
3. It remains to deal with the case j &gt; j ′ , under the assumption that, for all j ′′ &lt; j ′ ,

<!-- formula-not-decoded -->

which, in turn, implies

<!-- formula-not-decoded -->

Let us further distinguish the following subcases

- (a) If there exists some k , j ′ +1 ≤ k ≤ j , such that p [ v ( U p k )] &gt; p [ v ( U p ≤ k -1 )] + /epsilon1 ′ , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (b) If, for all k , j ′ +1 ≤ k ≤ j , it holds that p [ v ( U p k )] ≤ p [ v ( U p ≤ k -1 )] + /epsilon1 ′ , we will show a contradiction; hence, only the previous case can hold. Towards a contradiction,we argue first that

To show this, we distinguish the cases j = j ′ +1, j &gt; j ′ +1.

- In the case j = j ′ +1, we have

<!-- formula-not-decoded -->

- In the case j &gt; j ′ +1, we have for all k , j ′ +2 ≤ k ≤ j ,

<!-- formula-not-decoded -->

where the last inequality holds since the game G max has error amplification at most c . Summing these inequalities for j ′ +2 ≤ k ≤ j , we deduce that

<!-- formula-not-decoded -->

It follows that

But,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and recall that

<!-- formula-not-decoded -->

We can deduce that

<!-- formula-not-decoded -->

which combined with the above implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Lemma 9, it follows that the extraction of an /epsilon1 -Nash equilibrium of game G from an /epsilon1 ′ -Nash equilibrium of game GG cannot be done by just interpreting the values { x p j := p [ v ( x p j )] } j as the mixed strategy of player p . What we show next is that, for the right choice of /epsilon1 ′ , a trim and renormalize transformation succeeds in deriving an /epsilon1 -Nash equilibrium of game G from an /epsilon1 ′ -Nash equilibrium of game GG . Indeed, for all p ≤ r , suppose that { ˆ x p j } j are the values derived from { x p j } j by setting and then renormalizing the resulting values { ˆ x p j } j so that ∑ j ˆ x p j = 1.

<!-- formula-not-decoded -->

Lemma 10 There exists a polynomial p ( · ) such that, if {{ x p j } j } p is an /epsilon1/p ( |G| ) -Nash equilibrium of game GG , then the trimmed and renormalized values {{ ˆ x p j } j } p constitute an /epsilon1 -Nash equilibrium of game G .

Proof. We first establish the following useful lemma

Lemma 11 At an /epsilon1 ′ -Nash equilibrium of game GG , for all p , j , it holds that

<!-- formula-not-decoded -->

where c is the maximum error amplification of the gadgets used in the construction of GG , ζ r = c/epsilon1 ′ +((1 + ζ ) r -1)( c/epsilon1 ′ +1) , ζ = 2 r log n c/epsilon1 ′ .

Proof. Using the same notation as in Section 4.3, let S -p = { S -p (1) , . . . , S -p ( n r -1 ) } , so that

<!-- formula-not-decoded -->

/negationslash

Recall that in GG , for each partial sum ∑ z /lscript =1 u p jS -p ( /lscript ) x S -p ( /lscript ) , 1 ≤ z ≤ n r -1 , we have included vertex v ( ∑ z /lscript =1 u p jS -p ( /lscript ) x S -p ( /lscript ) ). Similarly, for each partial product of the summands u p js ∏ p = q ≤ z x q s q , 0 ≤ z ≤ r , we have included vertex v ( u p js ∏ p = q ≤ z x q s q ). Note that, since we have rescaled the utilities to the set [0 , 1], all partial sums and products must also lie in [0 , 1]. Note, moreover, that, to avoid large degrees in the resulting graphical game, each time we need to make use of a value x q s q we create a new copy of the vertex v ( x q s q ) using the gadget G = and, then, use the new copy for the computation of the desired partial product; an easy calculation shows that we have to make ( r -1) n r -1 copies of v ( x q s q ), for all q ≤ r , s q ∈ S q . To limit the degree of each vertex to 3 we create a binary tree of copies of v ( x q s q ) with ( r -1) n r -1 leaves and use each leaf once. Then, because of the error amplification of G = , this already induces an error of ±/ceilingleft log ( r -1) n r -1 /ceilingright c/epsilon1 ′ to each of the factors of the partial products. The following lemma characterizes the error that results from the error amplification of our gadgets in the computation of the partial products and can be proved easily by induction.

Lemma 12 For all p ≤ r , j ∈ S p , s ∈ S -p and z ≤ r ,

/negationslash where ζ z = c/epsilon1 ′ +((1 + ζ ) z -1)( c/epsilon1 ′ +1) , ζ = 2 r log n c/epsilon1 ′ .

<!-- formula-not-decoded -->

The following lemma characterizes the error in the computation of the partial sums and can be proved by induction using the previous lemma for the base case.

Lemma 13 For all p ≤ r , j ∈ S p and z ≤ n r -1 ,

<!-- formula-not-decoded -->

where ζ r is defined as in Lemma 12.

From Lemma 13 we can deduce, in particular, that for all p ≤ r , j ∈ S p ,

<!-- formula-not-decoded -->

/negationslash

/negationslash

Lemma 14 For all p ≤ r , j ∈ S p ,

Proof. We have

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

By the coupling lemma, we have that

/negationslash

<!-- formula-not-decoded -->

/negationslash

/negationslash

/negationslash

/negationslash

/negationslash

/negationslash for any coupling of ( X q ) q = p and ( Y q ) q = p . Applying a union bound to the right hand side of the above implies

/negationslash

<!-- formula-not-decoded -->

/negationslash

/negationslash

<!-- formula-not-decoded -->

/negationslash

/negationslash

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

/negationslash so that from (22), (23) we get

Now, note that, for all q ,

<!-- formula-not-decoded -->

Hence, (25) implies

<!-- formula-not-decoded -->

/negationslash

/negationslash

/negationslash

Now let us fix a coupling between ( X q ) q = p and ( Y q ) q = p so that, for all q = p ,

/negationslash

Such a coupling exists by the coupling lemma for each q = p individually, and for the whole vectors ( X q ) q = p and ( Y q ) q = p it exists because also the X q 's are independent and so are the Y q 's. Then (24) implies that

/negationslash

/negationslash

/negationslash

/negationslash

/negationslash

∣ ∣ Let us denote by X q the random variable, ranging over the set S q , which represents the mixed strategy { x q i } i ∈ S q , q ≤ r . Similarly define the random variable Y q from the mixed strategy { y q i } i ∈ S q , q ≤ r . Note, then, that 1 2 ∑ s ∈ S -p | x s -y s | is precisely the total variation distance between the vector random variable ( X q ) q = p and the vector random variable ( Y q ) q = p . That is,

/negationslash

/negationslash

/negationslash

We can conclude the proof of Lemma 10, by invoking Lemmas 11 and 14. Indeed, by the definition of the { ˆ x p j } , it follows that for all p , j ∈ S p , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where X {·} is the indicator function. Therefore, which implies

So,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which by Lemma 14 implies that

<!-- formula-not-decoded -->

where the second inequality follows from the fact that we have rescaled the utilities so that they lie in [0 , 1].

Choosing /epsilon1 ′ = /epsilon1 40 cr 2 n r +1 , we will argue that the conditions of an /epsilon1 -Nash equilibrium are satisfied by the mixed strategies { ˆ x p j } p,j . First, note that:

which implies that

<!-- formula-not-decoded -->

Also, note that

<!-- formula-not-decoded -->

which gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, for all p ≤ r , j, j ′ ∈ S p , we have that

<!-- formula-not-decoded -->

Therefore, { ˆ x p j } is indeed an /epsilon1 -Nash equilibrium of game G , which concludes the proof of the lemma.

We have the following extension of Theorem 8.

Theorem 11 For every fixed r &gt; 1 , there is a polynomial-time reduction from r -Nash to 3 -Nash .

Proof. The proof follows immediately from the proofs of Theorems 9 and 10. Indeed, observe that the reduction of Theorem 10 still holds when we use the gadget G + , ∗ of Section 4.5 for the construction our graphical games, since the gadget G + , ∗ has constant error amplification. Therefore, the problem of computing an /epsilon1 -Nash equilibrium of a r -player normal form game G can be polynomially reduced to computing an /epsilon1 ′ -Nash equilibrium of a graphical game GG ′ which can be legally colored with 3 colors (after performing the 'glueing' step described in the end of the proof of Theorem 8 and appropriately adjusting the /epsilon1 ′ specified in the proof of Theorem 10). Observe, further, that the reduction of Theorem 9 can be used to map the latter to computing an /epsilon1 ′′ -Nash equilibrium of a 3-player normal form game G ′′ , since the number of players that are required for G ′′ is equal to the minimum number of colors needed for a legal coloring of GG ′ . The claim follows by combining the reductions.

## 4.7 Reductions Between Different Notions of Approximation

We establish a polynomial time reduction from the problem of computing an approximately well supported Nash equilibrium to the problem of computing an approximate Nash equilibrium. As pointed out in Section 2, the reduction in the opposite direction is trivial, since an /epsilon1 -approximately well supported Nash equilibrium is also an /epsilon1 -approximate Nash equilibrium.

Lemma 15 Given an /epsilon1 -approximate Nash equilibrium { x p j } j,p of a game G we can compute in polynomial time a √ /epsilon1 · ( √ /epsilon1 +1+4( r -1) max { u } ) -approximately well supported Nash equilibrium { ˆ x p j } j,p , where r is the number of players in G and max { u } is the maximum entry in the payoff tables of G .

Proof. Since { x p j } j,p is an /epsilon1 -approximate Nash equilibrium, it follows that for every player p ≤ r and every mixed strategy { y p j } j for that player

Equivalently,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For all p ≤ r , denote U p j = ∑ s -p ∈ S -p u p js -p x s -p , for all j ∈ S p , and U p max = max j U p j . Then, if we choose { y p j } j to be some pure strategy from the set arg max j U p j , (27) implies

Now, let us fix some player p ≤ r . We want to upper bound the probability mass that the distribution { x p j } j assigns to pure strategies j ∈ S p which give expected utility U p j more than an additive /epsilon1k smaller than U p max , for some k to be specified later. The following bound is easy to derive using (28).

Claim 5 For all p , set

<!-- formula-not-decoded -->

where X A is the characteristic function of the event A . Then

<!-- formula-not-decoded -->

Let us consider then the strategy profile { ˆ x p j } j,p defined as follows

We establish the following bound on the L 1 distance between the strategy profiles { x p j } j and { ˆ x p j } j .

<!-- formula-not-decoded -->

Claim 6 For all p , ∑ j ∈ S p | x p j -ˆ x p j | ≤ 2 k -1 .

<!-- formula-not-decoded -->

Proof. Denote S p, 1 := { j | j ∈ S p , U p j ≥ U p max -/epsilon1k } and S p, 2 := S p \ S p, 1 . Then

<!-- formula-not-decoded -->

Now, for all players p , let ˆ U p j and ˆ U p max be defined similarly to U p j and U p max . Recall Lemma 14 from Section 4.6.

Lemma 16 For all p , j ∈ S p ,

<!-- formula-not-decoded -->

/negationslash

Let us then take ∆ 2 = 2 r -1 k -1 max p,j ∈ S p ,s ∈ S -p { u p js } . Claim 6 and Lemma 16 imply that the strategy profile { ˆ x p j } j,p satisfies

<!-- formula-not-decoded -->

We will establish that { ˆ x p j } j,p is a ( /epsilon1k + 2∆ 2 )-Nash equilibrium. Equivalently, we shall establish that

Indeed,

Taking k = 1+ 1 √ /epsilon1 , it follows that { ˆ x p j } j,p is a √ /epsilon1 · ( √ /epsilon1 +1+4( r -1) max { u p js } )-Nash equilibrium.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 5 The Main Reduction

We prove our main result, namely

Theorem 12 Both 3-Nash and 3 -graphical Nash are PPAD -complete.

Proof. That 3-Nash is in PPAD follows from Theorem 1. That 3graphical Nash is in PPAD follows by reducing it to 3-Nash , by Theorem 4, and then invoking Theorem 1. We hence focus on establishing the PPAD -hardness of the problems.

The reduction is from the problem Brouwer defined in Section 3.3. Given an instance of Brouwer , that is a circuit C with 3 n input bits and 2 output bits describing a Brouwer function as specified in Section 3.3, we construct a graphical game G , with maximum degree three, that simulates the circuit C , and specify an accuracy /epsilon1 , so that, given an /epsilon1 -Nash equilibrium of G , one can find in polynomial time a panchromatic vertex of the Brouwer instance. Then, since, by Theorem 4, 3graphical Nash reduces to 3-Nash , this completes the proof.

The graphical game G that we construct will be binary , in that each vertex v in it will have two strategies, and thus, at equilibrium, will represent a real number in [0 , 1], denoted p [ v ]. (Letting 0 and 1 denote the strategies, p [ v ] is the probability that v plays 1.) There will be three distinguished vertices v x , v y , and v z which will represent the coordinates of a point in the three dimensional cube and the construction will guarantee that in any Nash equilibrium of game G this point will be close to a panchromatic vertex of the given Brouwer instance.

The building blocks of G will be the game-gadgets G α , G × α , G = , G + , G -, G ∗ that we constructed in Section 4.1 plus a few new gadgets. Recall from Propositions 1, 2 and 4, Figures 4, 3 and 5, that

Lemma 17 There exist binary graphical games G α , where α is any rational in [0 , 1] , G × α , where α is any non-negative rational, G = , G + , G -, G ∗ , with at most four players a , b , c , d each, such that, in all games, the payoffs of a and b do not depend on the choices of the other vertices c, d , and, for /epsilon1 &lt; 1 ,

1. in every /epsilon1 -Nash equilibrium of game G α , we have p [ d ] = α ± /epsilon1 ;
2. in every /epsilon1 -Nash equilibrium of game G × α , we have p [ d ] = min(1 , α p [ a ]) ± /epsilon1 ;
3. in every /epsilon1 -Nash equilibrium of game G = , we have p [ d ] = p [ a ] ± /epsilon1 ;
4. in every /epsilon1 -Nash equilibrium of game G + , we have p [ d ] = min { 1 , p [ a ] + p [ b ] } ± /epsilon1 ;
5. in every /epsilon1 -Nash equilibrium of game G -, we have p [ d ] = max { 0 , p [ a ] -p [ b ] } ± /epsilon1 ;
6. in every /epsilon1 -Nash equilibrium of game , we have p [ d ] = p [ a ] p [ b ] /epsilon1 ;

where by x = y ± /epsilon1 we denote y -/epsilon1 ≤ x ≤ y + /epsilon1

- G ∗ · ± .

Let us, further, define a comparator game G &lt; .

Lemma 18 There exists a binary graphical game G &lt; with three players a , b and d such that the payoffs of a and b do not depend on the choices of d and, in every /epsilon1 -Nash equilibrium of the game, with /epsilon1 &lt; 1 , it holds that p [ d ] = 1 , if p [ a ] &lt; p [ b ] -/epsilon1 , and p [ d ] = 0 , if p [ a ] &gt; p [ b ] + /epsilon1 .

Proof. Let us define the payoff table of player d as follows: d receives a payoff of 1 if d plays 0 and a plays 1, and d receives a payoff of 1 if d plays 1 and b plays 1, otherwise d receives a payoff of 0. Equivalently, d receives an expected payoff of p [ a ], if d plays 0, and an expected payoff of p [ b ], if d plays 1. It immediately follows that, if in an /epsilon1 -Nash equilibrium p [ a ] &lt; p [ b ] -/epsilon1 , then p [ d ] = 1, whereas, if p [ a ] &gt; p [ b ] + /epsilon1 , p [ d ] = 0.

Figure 13: Brittleness of Comparator Games.

<!-- image -->

Notice that, in G &lt; , p [ d ] is arbitrary if | p [ a ] -p [ b ] | ≤ /epsilon1 ; hence we call it the brittle comparator . As an aside, it is not hard to see that a robust comparator, one in which d is guaranteed, in an exact Nash equilibrium, to be, say, 0 if p [ a ] = p [ b ], cannot exist, since it could be used to produce a simple graphical game with no Nash equilibrium, contradicting Nash's theorem. For completeness we present such a game in Figure 13, where vertices e and b constitute a G 1 game so that, in any Nash equilibrium, p [ b ] = 1, vertices d , f , a constitute a G = game so that, in any Nash equilibrium, p [ a ] = p [ d ] and vertices a , b , d constitute a comparator game with the hypothetical behavior that p [ d ] = 1, if p [ a ] &lt; p [ b ] and p [ d ] = 0, if p [ a ] ≥ p [ b ]. Then it is not hard to argue that the game of Figure 13 does not have a Nash equilibrium contrary to Nash's theorem: indeed if, in a Nash equilibrium, p [ a ] = 1, then p [ d ] = 0, since p [ a ] = 1 = p [ b ], and so p [ a ] = p [ d ] = 0, by G = , a contradiction; on the other hand, if, in a Nash equilibrium, p [ a ] &lt; 1, then p [ d ] = 1, since p [ a ] &lt; 1 = p [ b ], and so p [ a ] = p [ d ] = 1, by G = , again a contradiction.

To continue with our reduction from Brouwer to 3 -graphical nash , we include the following vertices to the graphical game G .

- the three coordinate vertices v x , v y , v z ,
- for i ∈ { 1 , 2 , . . . , n } , vertices v b i ( x ) , v b i ( y ) and v b i ( z ) , whose p -values correspond to the i -th most significant bit of p [ v x ], p [ v y ], p [ v z ],
- for i ∈ { 1 , 2 , . . . , n } , vertices v x i , v y i and v z i , whose p -values correspond to the fractional number resulting from subtracting from p [ v x ], p [ v y ], p [ v z ] the fractional numbers corresponding to the i -1 most significant bits of p [ v x ], p [ v y ], p [ v z ] respectively.

We can extract these values by computing the binary representation of /floorleft p [ v x ]2 n /floorright and similarly for v y and v z , that is, the binary representations of the integers i, j, k such that ( x, y, z ) = ( p [ v x ] , p [ v y ] , p [ v z ]) lies in the cubelet K ijk . This is done by a graphical game that simulates, using the arithmetical gadgets of Lemmas 17 and 18, the following algorithm ( &lt; ( a, b ) is 1 if a ≤ b and 0 if a &gt; b ):

```
x 1 = x ; for i = 1 , . . . , n do: { b i ( x ) := < (2 -i , x i ); x i +1 := x i -b i ( x ) · 2 -i } ;
```

similarly for y and z ;

This is accomplished in G by connecting these vertices as prescribed by Lemmas 17 and 18, so that p [ v x i ] , p [ v b i ( x ) ], etc. approximate the value of x i , b i ( x ) etc. as computed by the above algorithm. The following lemma (when applied with m = n ) shows that this device properly decodes the first

n bits of the binary expansion of x = p [ v x ], as long as x is not too close to a multiple of 2 -n (suppose /epsilon1 &lt;&lt; 2 -n to be fixed later).

Lemma 19 For m ≤ n , if ∑ m i =1 b i 2 -i + 3 m/epsilon1 &lt; p [ v x ] &lt; ∑ m i =1 b i 2 -i + 2 -m -3 m/epsilon1 for some b 1 , . . . , b m ∈ { 0 , 1 } , then, in every /epsilon1 -Nash equilibrium of G , p [ v b j ( x ) ] = b j , and p [ v x j +1 ] = p [ v x ] -∑ j i =1 b i 2 -i ± 3 j/epsilon1 , for all j ≤ m .

<!-- formula-not-decoded -->

Proof. The proof is by induction on j . For j = 1, the hypothesis ∑ m i =1 b i 2 -i + 3 m/epsilon1 &lt; p [ v x ] &lt; ∑ m i =1 b i 2 -i +2 -m -3 m/epsilon1 implies, in particular, that and, since p [ v x 1 ] = p [ v x ] ± /epsilon1 , it follows that

<!-- formula-not-decoded -->

By Lemma 18, this implies that p [ v b 1 ( x ) ] = b 1 ; note that the preparation of the constant 1 2 -against which a comparator game compares the value p [ v x 1 ]- is done via a G 1 2 game which introduces an error of ± /epsilon1 . For the computation of p [ v x 2 ], the multiplication of p [ v b 1 ( x ) ] by 1 2 and the subtraction of the product from p [ v x 1 ] introduce an error of ± /epsilon1 each and, therefore, p [ v x 2 ] = p [ v x 1 ] -b 1 1 2 ± 2 /epsilon1 . And, since p [ v x 1 ] = p [ v x ] ± /epsilon1 , it follows that p [ v x 2 ] = p [ v x ] -b 1 1 2 ± 3 /epsilon1 , as required.

<!-- formula-not-decoded -->

Supposing that the claim holds up to j -1 ≤ m -1, we will show that it holds for j . By the induction hypothesis, we have that p [ v x j ] = p [ v x ] -∑ j -1 i =1 b i 2 -i ± 3( j -1) /epsilon1 . Combining this with ∑ m i =1 b i 2 -i +3 m/epsilon1 &lt; p [ v x ] &lt; ∑ m i =1 b i 2 -i +2 -m -3 m/epsilon1 , it follows that which implies

Continue as in the base case.

Assuming that x = p [ v x ], y = p [ v y ], z = p [ v z ] are all at distance greater than 3 n/epsilon1 from any multiple of 2 -n , the part of G that implements the above algorithm computes i , j , k such that the point ( x, y, z ) lies in the cubelet K ijk ; that is, there are 3 n vertices of the game G whose p values are equal to the n bits of the binary representation of i , j , k . Once we have the binary representations of i , j , k , we can feed them into another part of G that simulates the circuit C . We could simulate the circuit by having vertices that represent gates, using addition (with ceiling 1) to simulate or , multiplication for and , and 1 -x for negation. However, there is a simpler way, one that avoids the complications related to accuracy, to simulate Boolean functions under the assumption that the inputs are 0 or 1:

Lemma 20 There are binary graphical games G ∨ , G ∧ , G ¬ with two input players a, b (one input player a for G ¬ ) and an output player c such that the payoffs of a and b do not depend on the choices of c , and, at any /epsilon1 -Nash equilibrium with /epsilon1 &lt; 1 / 4 in which p [ a ] , p [ b ] ∈ { 0 , 1 } , p [ c ] is also in { 0 , 1 } , and is in fact the result of applying the corresponding Boolean function to the inputs.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. These games are in the same spirit as G &lt; . In G ∨ , for example, the payoff to c is 1 / 2 if it plays 0; if c plays 1 its payoff is 1 if at least one of a, b plays 1, and it is 0 if they both play 0. Similarly for G ∧ and G ¬ .

It would seem that all we have to do now is to close the loop as follows: in addition to the part of G that computes the bits of i, j, k , we could have a part that simulates circuit C in the neighborhood of K ijk and decides whether among the vertices of the cubelet K ijk there is a panchromatic one; if not, the vertices v x , v y and v z could be incentivized to change their p values, say in the direction δ C ( i,j,k ) , otherwise stay put. To simulate a circuit evaluation in G we could have one vertex for each gate of the circuit so that, in any /epsilon1 -Nash equilibrium in which all the p [ v b i ( x ) ]'s are 0 -1, the vertices corresponding to the outputs of the circuit also play pure strategies, and, furthermore, these strategies correspond correctly to the outputs of the circuit.

But, as we mentioned above, there is a problem: Because of the brittle comparators, at the boundaries of the cubelets the vertices that should represent the values of the bits of i , j , k hold in fact arbitrary reals and, therefore, so do the vertices that represent the outputs of the circuit, and this noise in the calculation can create spurious Nash equilibria. Suppose for example that ( x, y, z ) lies on the boundary between two cubelets that have color 1, i.e. their centers are assigned vector δ 1 by C , and none of these cubelets has a panchromatic vertex. Then there ought not to be a Nash equilibrium with p [ v x ] = x , p [ v y ] = y , p [ v z ] = z . We would want that, when p [ v x ] = x , p [ v y ] = y , p [ v z ] = z , the vertices v x , v y , v z have the incentive to shift their p values in direction δ 1 , so that v x prefers to increase p [ v x ]. However, on a boundary between two cubelets, some of the 'bit values' that get loaded into the vertices v b i ( x ) , could be other than 0 and 1, and then there is nothing we can say about the output of the circuit that processes these values.

To overcome this difficulty, we resort to the following averaging maneuver : We repeat the above computation not just for the point ( x, y, z ), but also for all M = (2 m + 1) 3 points of the form ( x + p · α, y + q · α, z + s · α ), for -m ≤ p, q, s ≤ m , where m is a large enough constant to be fixed later (we show below that m = 20 is sufficient). The vertices v x , v y , v z are then incentivized to update their values according to the consensus of the results of these computations, most of which are reliable, as we shall show next.

Let us first describe this averaging in more detail. It will be convenient to assume that the output of C is a little more explicit than 3 bits: let us say that C computes six bits ∆ x + , ∆ x -, ∆ y + , ∆ y -, ∆ z + , ∆ z -, such that at most one of ∆ x + , ∆ x -is 1, at most one of ∆ y + , ∆ y -is 1, and similarly for z , and the increment of the Brouwer function at the center of K ijk is α · (∆ x + -∆ x -, ∆ y + -∆ y -, ∆ z + -∆ z -), equal to one of the vectors δ 0 , δ 1 , δ 2 , δ 3 specified in the definition of Brouwer , where recall α = 2 -2 n .

The game G has the following structure: Starting from ( x, y, z ), some part of the game is devoted to calculating the points ( x + p · α, y + q · α, z + s · α ), -m ≤ p, q, s ≤ m . Then, another part evaluates the circuit C on the binary representation of each of these points yielding 6 M output bits, ∆ x + 1 , . . . , ∆ z -M . A final part calculates the following averages

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which correspond to the average positive, respectively negative, shift of all M points.

We have already described above how to implement the bit extraction and the evaluation of a circuit using the gadgets of Lemmas 17, 18 and 20. The computation of points ( x + p · α, y + q · α, z + s · α ), for all -m ≤ p, q, s ≤ m , is also easy to implement by preparing the values α | p | , α | q | , α | s | , using gadgets G α | p | , G α | q | , G α | s | , and then adding or subtracting the results to x , y and z respectively, depending on whether p is positive or not and similarly for q and s . Of course, these computations are subject to truncations at 0 and 1 (see Lemma 17).

To implement the averaging of Equations (29) and (30) we must be careful on the order of operations. Specifically, we first have to multiply the 6 outputs, ∆ x + t , ∆ x -t , ∆ y + t , ∆ y -t , ∆ z + t , ∆ z -t , of each circuit evaluation by α M using the G × α M gadget and, having done so, we then implement the additions (29) and (30). Since α will be a very small constant, by doing so we avoid undesired truncations at 0 and 1.

We can now close the loop by inserting equality, addition and subtraction gadgets, G = , G + , G -, that force, at equilibrium, x to be equal to ( x ′ + δx + ) -δx -, where x ′ is a copy of x created using G = , and similarly for y and z . Note that in G we respect the order of operations when implementing ( x ′ + δx + ) -δx -to avoid undesired truncations at 0 or 1 as we shall see next. This concludes the reduction; it is clear that it can be carried out in polynomial time.

Our proof is concluded by the following claim. For the following lemma we choose /epsilon1 = α 2 . Recall from our definition of Brouwer that α = 2 -2 n .

Lemma 21 In any /epsilon1 -Nash equilibrium of the game G , one of the vertices of the cubelet(s) that contain ( p [ v x ] , p [ v y ] , p [ v z ]) is panchromatic.

Proof. We start by pointing out a simple property of the increments δ 0 , . . . , δ 3 :

Lemma 22 Suppose that for nonnegative integers k 0 , . . . , k 3 all three coordinates of ∑ 3 i =0 k i δ i are smaller in absolute value than αK 5 where K = ∑ 3 i =0 k i . Then all four k i are positive.

Let us denote by v δx + , { v ∆ x + t } 1 ≤ t ≤ M the vertices of G that represent the values δx + , { ∆ x + t } 1 ≤ t ≤ M . To implement the averaging

Proof. For the sake of contradiction, suppose that k 1 = 0. It follows that k 0 &lt; K/ 5 (otherwise the negative x coordinate of ∑ 3 i =0 k i δ i would be too large), and thus one of k 2 , k 3 is larger than 2 K/ 5, which makes the corresponding coordinate of ∑ 3 i =0 k i δ i too large, a contradiction. Similarly if k 2 = 0 or k 3 = 0. Finally, if k 0 = 0 then one of k 1 , k 2 , k 3 is at least K/ 3 and the associated coordinate of ∑ 3 i =0 k i δ i at least αK/ 3, again a contradiction.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

inside G , we first multiply each p [ v ∆ x + t ] by α M using a G α M gadget and we then sum the results by a sequence of addition gadgets. Since each of these operations induces an error of ± /epsilon1 and there are 2 M -1 operations it follows that

Similarly, denoting by v δx -, { v ∆ x -t } 1 ≤ t ≤ M the vertices of G that represent the values δx -, { ∆ x -t } 1 ≤ t ≤ M , it follows that

<!-- formula-not-decoded -->

and similarly for directions y and z .

Wecontinue the proof by distinguishing two subcases for the location of ( x, y, z ) = ( p [ v x ] , p [ v y ] , p [ v z ])

- (a) the point ( p [ v x ] , p [ v y ] , p [ v z ]) is further than ( m +1) α from every face of the cube [0 , 1] 3 ,
- (b) the point ( p [ v x ] , p [ v y ] , p [ v z ]) is at distance at most ( m +1) α from some face of the cube [0 , 1] 3 .

Case (a): Denoting by v x + p · α the player of G that represents x + p · α , the small value of /epsilon1 relative to α implies that at most one of the values p [ v x + p · α ], -m ≤ p ≤ m , can be 3 n/epsilon1 -close to a multiple of 2 -n , and similarly for the directions y and z . Indeed, recall that x + p · α is computed from x by first preparing the value | p | α via a G | p | α gadget and then adding or subtracting the result to x -depending on whether p is positive or not- using G + or G -. It follows that

<!-- formula-not-decoded -->

since each gadget introduces an error of ± /epsilon1 , where note that there are no truncations at 0 or 1, because, by assumption, ( m +1) α &lt; p [ v x ] &lt; 1 -( m +1) α . Consequently, for p &gt; p ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since m is a constant, α = 2 -2 n , /epsilon1 = α 2 , and n is assumed to be large enough. Hence, from among the M = (2 m +1) 3 circuit evaluations, all but at most 3(2 m +1) 2 , or at least K = (2 m -2)(2 m +1) 2 , compute legitimate, i.e. binary, ∆ x + etc. values.

Let us denote by K ⊆ {m,... , m } 3 , |K| ≥ K , the set of values ( p, q, r ) for which the bit extraction from ( p [ v x + p · α ] , p [ v y + q · α ] , p [ v z + r · α ]) results in binary outputs and, consequently, so does the circuit evaluation. Let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall that we have inserted gadgets G + , G -and G = in G to enforce that in a Nash equilibrium x = x ′ + δx + -δx -, where x ′ is a copy of x . Because of the defection of the gadgets this will not be exactly tight in an /epsilon1 -Nash equilibrium. More precisely, denoting by v x ′ the player of G corresponding to x ′ , the following are true in an /epsilon1 -Nash equilibrium

<!-- formula-not-decoded -->

where for the second observe that both p [ v δx + ] and p [ v δx -] are bounded above by α +(2 M -1) /epsilon1 so there will be no truncations at 0 or 1 when adding p [ v δx + ] to p [ v ′ x ] and then subtracting p [ v δx -]. By combining the above we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and, moreover, and, similarly, for y and z

the other cases follow similarly.

First, we show that, for all -m ≤ p ≤ m , the bit extraction from p [ v x + p · α ] results in binary outputs. From the proof of Lemma 19 it follows that, to show this, it is enough to establish that p [ v x + pα ] &lt; 2 -n -3 n/epsilon1 , for all p . Indeed, for p ≥ 0, Equation (33) applies because there are no truncations at 1 at the addition gadget. So for p ≥ 0 we get

<!-- formula-not-decoded -->

On the other hand, for p &lt; 0, there might be a truncation at 0 when we subtract the value | p | α from p [ v x ]. Nevertheless, we have that

<!-- formula-not-decoded -->

Therefore, for all -m ≤ p ≤ m , the bit extraction from p [ v x + p · α ] is successful, i.e. results in binary outputs.

<!-- formula-not-decoded -->

Now, if we use (31), (32), (35), (36) we derive

<!-- formula-not-decoded -->

where S K /lscript , S K c /lscript is the /lscript coordinate of S K , S K c . Moreover, since |K| ≥ K , the summation S K c /lscript has at most M -K summands and because each of them is at most α M in absolute value it follows that | S K c /lscript | ≤ α M ( M -K ), for all /lscript = x, y, z . Therefore, we have that

<!-- formula-not-decoded -->

Finally, note by the definition of the set K that, for all ( p, q, r ) ∈ K , the bit extraction from ( p [ v x + p · α ] , p [ v y + q · α ] , p [ v z + r · α ]) and the following circuit evaluation result in binary outputs. Therefore, S K = 1 M ∑ 3 i =0 k i δ i for some nonnegative integers k 0 , . . . , k 3 adding up to |K| . From the above we get that

<!-- formula-not-decoded -->

By choosing m = 20, the bound becomes less than αK/ 5, and so Lemma 22 applies. It follows that, among the results of the |K| circuit computations, all four δ 0 , . . . , δ 3 appeared. And, since every point on which the circuit C is evaluated is within /lscript 1 distance at most 3 mα +6 /epsilon1 &lt;&lt; 2 -n from the point ( x, y, z ), as Equation (33) dictates, this implies that among the corners of the cubelet(s) containing ( x, y, z ) there must be one panchromatic corner, completing the proof of Lemma 21 for case (a).

Case (b): We will show that there is no /epsilon1 -Nash equilibrium in which ( p [ v x ] , p [ v y ] , p [ v z ]) is within distance ( m +1) α from a face of [0 , 1] 3 . We will argue so only for the case

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the directions y and z the picture is exactly the same as in case (a) and, therefore, there exists at most one q, -m ≤ q ≤ m , and at most one r , -m ≤ r ≤ m , for which the bit extraction from p [ v y + q · α ] and p [ v z + r · α ] fails. Therefore, from among the M = (2 m +1) 3 points of the form ( p [ v x + p · α ] , p [ v y + q · α ] , p [ v z + r · α ]) the bit extraction succeeds in all but at most 2(2 m +1) 2 of them.

Therefore, at least K ′ = (2 m -1)(2 m + 1) 2 circuit evaluations are successful, i.e. in binary arithmetic, and, moreover, they correspond to points inside cubelets of the form K ijk with i = 0. In particular, from Equation (34) and the analogous equations for the y and z coordinates, it follows that the successful circuit evaluations correspond to points inside at most 4 neighboring cubelets of the form K 0 jk . Since these cubelets are adjacent to the x = 0 face of the cube, from the properties of the circuit C in the definition of the problem Brouwer , it follows that, among the outputs of these evaluations, one of the vectors δ 0 , δ 1 , δ 2 , δ 3 is missing. Without loss of generality, let us assume that δ 0 is missing. Then, since there are K ′ successful evaluations, one of δ 1 , δ 2 , δ 3 appears at least K ′ / 3 times.

If this is vector δ 1 (similar argument applies for the cases δ 2 , δ 3 ), then denoting by v x ′ + δx + the player corresponding to x ′ + δx + , the following should be true in an /epsilon1 -Nash equilibrium.

<!-- formula-not-decoded -->

in the second inequality of the third line above, we used that p [ v x ] ≤ ( m + 1) α . Combining the above we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or equivalently that which implies

which is not satisfied by our selection of parameters.

To conclude the proof of Theorem 12, if we find any /epsilon1 -Nash equilibrium of G , Lemma 21 has shown that by reading off the first n binary digits of p [ v x ], p [ v y ] and p [ v z ] we obtain a solution to the corresponding instance of Brouwer .

## 6 Further Results and Open Problems

## 6.1 Two Players

Soon after our proof became available, Chen and Deng [7] showed that our PPAD-completeness result can be extended to the important two-player case. Here we present a rather simple modification of our proof from the previous section establishing this result.

Theorem 13 ([7]) 2-Nash is PPAD -complete.

Proof. Let us define d -additive graphical Nash to be the problem d -graphical Nash restricted to bipartite graphical games with additive utility functions defined next.

Definition 5 Let GG be a graphical game with underlying graph G = ( V, E ) . We call GG a bipartite graphical game with additive utility functions if G is a bipartite graph and, moreover, for each vertex v ∈ V and for every pure strategy s v ∈ S v of that player, the expected payoff of v for playing the pure strategy s v is a linear function of the mixed strategies of the vertices in N v \ { v } with rational coefficients; that is, there exist rational numbers { α s v u,s u } u ∈N v \{ v } ,s u ∈ S u , α s v u,s u ∈ [0 , 1] for all u ∈ N ( v ) \ { v } , s u ∈ S u , such that the expected payoff to vertex v for playing pure strategy s v is

<!-- formula-not-decoded -->

where p [ u : s u ] denotes the probability that vertex u plays pure strategy s u .

The proof is based on the following lemmas.

Lemma 23 Brouwer is poly-time reducible to 3 -additive graphical Nash .

Lemma 24 3 -additive graphical Nash is poly-time reducible to 2-Nash .

Proof of Lemma 23: The reduction is almost identical to the one in the proof of Theorem 12. Recall that given an instance of Brouwer a graphical game was constructed using the gadgets G α , G × α , G = , G + , G -, G ∗ , G ∨ , G ∧ , G ¬ , and G &gt; . In fact, gadget G ∗ is not required, since only multiplication by a constant is needed which can be accomplished via the use of gadget G × α . Moreover, it is not hard to see by looking at the payoff tables of the gadgets defined in Section 4.1 and Lemma 18 that, in gadgets G α , G × α , G = , G + , G -, and G &gt; , the non-input vertices have the additive utility functions property of Definition 5. Let us further modify the games G ∨ , G ∧ , G ¬ so that their output vertices have the additive utility functions property.

Lemma 25 There are binary graphical games G ∨ , G ∧ , G ¬ with two input players a, b (one input player a for G ¬ ) and an output player c such that the payoffs of a and b do not depend on the choices of c , c 's payoff satisfies the additive utility functions property, and, in any /epsilon1 -Nash equilibrium with /epsilon1 &lt; 1 / 4 in which p [ a ] , p [ b ] ∈ { 0 , 1 } , p [ c ] is also in { 0 , 1 } , and is in fact the result of applying the corresponding Boolean function to the inputs.

Proof. For G ∨ , the payoff of player c is 0 . 5 p [ a ] + 0 . 5 p [ b ] for playing 1 and 1 4 for playing 0. For G ∧ , the payoff of player c is 0 . 5 p [ a ] + 0 . 5 p [ b ] for playing 1 and 3 4 for playing 0. For G ¬ , the payoff of player c is p [ a ] for playing 0 and p [ a : 0] for playing 1.

If the modified gadgets G ∨ , G ∧ , G ¬ specified by Lemma 25 are used in the construction of Theorem 12, all vertices of the resulting graphical game satisfy the additive utility functions property of Definition 5. To make sure that the graphical game is also bipartite we modify the gadgets G ∨ , G ∧ , G ¬ , and G &gt; with the insertion of an extra output vertex. The modification is the same for all 4 gadgets: let c be the output vertex of any of these gadgets; we introduce a new output vertex e , whose payoff only depends on the strategy of c , but c 's payoff does not depend on the strategy of e , and such that the payoff of e is p [ c ] for playing 1 and p [ c : 0] for playing 0 (i.e. e 'copies' c , if c 's strategy is pure). It is not hard to see that, for every gadget, the new output vertex has the same behavior with regards to the strategies of the input vertices as the old output vertex, as specified by Lemmas 18 and 25. Moreover, it is not hard to verify that the graphical game resulting from the construction of Theorem 12 with the use of the modified gadgets G ∨ , G ∧ , G ¬ , and G &gt; is bipartite; indeed, it is sufficient to color blue the input and output vertices of all G × α , G = , G + , G -, G ∨ , G ∧ , G ¬ , and G &gt; gadgets used in the construction, blue the output vertices of all G α gadgets used, and red the remaining vertices. ✷

˜ The construction of G from GG is almost identical to the one described in Figure 7. Let V = V 1 /unionsq V 2 be the bipartition of the vertices of set V so that all edges are between a vertex in V 1 and a vertex in V 2 . Let us define c : V →{ 1 , 2 } as c ( v ) = 1 iff v ∈ V 1 and let us assume, without loss of generality, that | v : c ( v ) = 1 | = | v : c ( v ) = 2 | ; otherwise, we can add to GG isolated vertices to make up any shortfall. Suppose that n is the number of vertices in GG (after the possible addition of isolated vertices) and t the cardinality of the strategy sets of the vertices in V , and let /epsilon1 ′ = /epsilon1/n . Let us then employ the Steps 4 and 5 of the algorithm in Figure 7 to construct the normal form game G from the graphical game GG ; however, we choose M = 6 tn /epsilon1 and we modify Step 5b to read as follows

Proof of Lemma 24: Let ˜ GG be a bipartite graphical game of maximum degree 3 with additive utility functions and GG the graphical game resulting after rescaling all utilities to the set [0 , 1], e.g. by dividing all utilities by max { ˜ u } , where max { ˜ u } is the largest entry in the payoff tables of game ˜ GG . Also, let /epsilon1 &lt; 1. In time polynomial in |GG| +log(1 //epsilon1 ), we will specify a 2-player normal form game G and an accuracy /epsilon1 ′ with the property that, given an /epsilon1 ′ -Nash equilibrium of G , one can recover in polynomial time an /epsilon1 -Nash equilibrium of GG . This will be enough, since an /epsilon1 -Nash equilibrium of GG is trivially an /epsilon1 · max { ˜ u } -Nash equilibrium of game ˜ GG and, moreover, |GG| is polynomial in | GG| .

- (b)' for v ∈ V and s v ∈ S v , if c ( v ) = p and s contains ( v, s v ) and ( u, s u ) for some u ∈ N ( v ) \ { v } , s u ∈ S u , then u p s = α s v u,s u ,

where we used the notation from Definition 5.

<!-- formula-not-decoded -->

We argue next that, given an /epsilon1 ′ -Nash equilibrium { x p ( v,a ) } p,v,a of G , { x v a } v,a is an /epsilon1 -Nash equilibrium of GG , where

Suppose that p = c ( v ) for some vertex v of the graphical game GG . As in the proof of Theorem 5, Lemma 7, it can be shown that in any /epsilon1 ′ -Nash equilibrium of the game G ,

<!-- formula-not-decoded -->

Now, without loss of generality assume that p = 1 (the pursuer) and suppose v is vertex v ( p ) i , in the notation of Figure 7. Then, in an /epsilon1 ′ -Nash equilibrium of the game G , we have, by the definition

of a Nash equilibrium, that for all strategies s v , s ′ v ∈ S v of vertex v :

But

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and similarly for s ′ v . Therefore, (37) implies

<!-- formula-not-decoded -->

Lemma 26 For all v , a ∈ S v ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By (38) and Lemma 26, we get that, for all v ∈ V , s v , s ′ v ∈ S v ,

<!-- formula-not-decoded -->

Since n 2 /epsilon1 ′ + |N v \{ v }| t n M ≤ /epsilon1 , it follows that { x v a } v,a is an /epsilon1 -Nash equilibrium of the game GG . ✷

## 6.2 Approximate Nash Equilibria

Our proof establishes that it is PPAD-hard to find an approximate Nash equilibrium when the desired additive approximation /epsilon1 is an inverse exponential in the size of the instance. What happens for larger /epsilon1 's? Chen, Deng and Teng show that, for any /epsilon1 which is inverse polynomial in n , computing an /epsilon1 -Nash equilibrium in 2-player games with n strategies per player remains PPAD-complete [9]; this is done by a modification of our reduction in which the starting Brouwer problem is defined not on the 3-dimensional cube, but in the n -dimensional hypercube. Intuitively, the difference is this: In order to create the exponentially many cells needed to embed the 'line,' our construction had to resort to exponentially small cell size; in contrast, the n -dimensional hypercube contains exponentially many cells, all of reasonably large size.

The result of [9] implies that there is no fully polynomial time approximation scheme (a family of approximation algorithms that are polynomial in both the input size and 1 /epsilon1 ). But is there a polynomial time approximation scheme (family of polynomial algorithms with 1 /epsilon1 in the exponent)? This is a major open question that is left open.

And how about finitely large /epsilon1 's? Since the establishment of PPAD-completeness of Nash , we have seen a sequence of polynomial algorithms for finding /epsilon1 -approximate Nash equilibria with /epsilon1 = . 5 [17], . 39 [18], . 37 [4]; the best known /epsilon1 at the time of writing is . 34 [56].

## 6.3 Nash Equilibria in Graphical Games

Besides normal-form games, our work settles the complexity of computing a Nash equilibrium in graphical games of degree at most 3, again in the negative direction. Elkind, Goldberg and Goldberg show that a Nash equilibrium of graphical games with maximum degree 2 and 2 strategies per player can be computed in polynomial time [24]. Daskalakis and Papadimitriou describe a polynomial time approximation scheme for graphical games with a constant number of strategies per player, bounded degree and treewidth at most logarithmic in the number of players [20]. Can approximate Nash equilibria in general graphical games be computed efficiently?

## 6.4 Special Cases

Are there important and broad classes of games for which the Nash equilibrium problem can be solved efficiently? It has been shown that finding Nash equilibria in normal form games with all utilities either 1 or -1 (the so-called win-lose games ) remains PPAD-complete [1, 10]. Rather surprisingly, it was also recently shown that, essentially, it is PPAD-complete to play even repeated games [3] (the so-called 'Folk Theorem for repeated games' [52] notwithstanding).

On the positive side, Daskalakis and Papadimitriou [21, 22] develop a polynomial-time approximation scheme for anonymous games (games in which the utility of each player depends on her own strategy and the number of other players playing various strategies, but not the identities of these players), when the number of strategies per player is bounded. Although their algorithm is too inefficient to have a direct effect in practice, it does remove the intractability obstacle for a very large class of multiplayer games. Note that finding a Nash equilibrium in anonymous games is not known to be PPAD-complete.

## 6.5 Further Applications of our Techniques

What is the complexity of the Nash Equilibrium problem in other classes of succinctly representable games with many players (besides the graphical problems resolved in this paper)? For example, are these problems even in PPAD? (It is typically easy to see that they cannot be easier than the normal-form problem.) Daskalakis, Fabrikant and Papadimitriou give a general sufficient condition, satisfied by all known succinct representations of games, for membership of the Nash equilibrium problem in the class PPAD [15]. The basic idea is using the 'arithmetical' gadgets in our present proof to simulate the calculation of utilities in these succinct games. However, whether computing a sequential equilibrium [46] in an extensive-form game is in PPAD is left open.

Our technique can be used to treat two other open problems in complexity. One is that of the complexity of simple stochastic games defined in [12], heretofore known to be in TFNP, but not in any of the more specialized classes like PPAD or PLS . Now, it is known that this problem is equivalent to evaluating combinational circuits with max , min , and average gates. Since all three kinds of gates can be implemented by the graphical games in our construction, it follows that solving simple stochastic games is in PPAD. 6

Similarly, by an explicit construction we can show the following.

Theorem 14 Let p : [0 , 1] → R be any polynomial function such that p (0) &lt; 0 and p (1) &gt; 0 . Then there exists a graphical game in which all vertices have two strategies, 0 and 1 , and in which the mixed Nash equilibria correspond to a particular vertex v playing strategy 1 with probability equal to the roots of p ( x ) between 0 and 1 .

6 One has to pay some attention to the approximation; see [25] for details.

Proof Sketch. Let p be described by its coefficients α 0 , α 1 , . . . , α n , so that

<!-- formula-not-decoded -->

Taking A := ( ∑ n i =0 | α i | ) -1 , it is easy to see that the range of the polynomial q ( x ) := 1 2 Ap ( x ) + 1 2 is [0 , 1], that q (0) &lt; 1 2 , q (1) &gt; 1 / 2, and that every point r ∈ [0 , 1] such that q ( r ) = 1 2 is a root of p . We define next a graphical game GG in which all vertices have two strategies, 0 and 1, and a designated vertex v of GG satisfies the following

- (i) in any mixed Nash equilibrium of GG the probability x v 1 by which v plays strategy 1 satisfies q ( x v 1 ) = 1 / 2;
- (ii) for any root r of p in [0 , 1], there exists a mixed Nash equilibrium of GG in which x v 1 = r ;

The graphical game has the following structure:

- there is a component graphical game GG q with an 'input vertex' v and an 'output vertex' u such that, in any Nash equilibrium of GG , the mixed strategies of u and v satisfy x u 1 = q ( x v 1 ); a graphical game which progressively performs the computations required for the evaluation of q ( · ) on x v 1 can be easily constructed using our game-gadgets; note that the computations can be arranged in such an order that no truncations at 0 or 1 happen (recall the rescaling by 1 2 A and the shifting around 1 / 2 done above);
- a comparator game G &gt; (see Lemma 18) compares the mixed strategy of u with the value 1 2 , prepared by a G 1 / 2 gadget (see Section 4.1), so that the output vertex of the comparator game plays 0 if x u 1 &gt; 1 2 , 1 if x u 1 &lt; 1 2 , and anything if x u 1 = 1 2 ;
- we identify the output player of G &gt; with player v ;

It is not hard to see that GG satisfies Properties (i) and (ii).

As a corollary of Theorem 14, it follows that fixed points of polynomials can be computed by computing (exact) Nash equilibria of graphical games. Computing fixed points of polynomials via exact Nash equilibria in graphical games can be extended to the multi-variate case again via the use of game gadgets to evaluate the polynomial and the use of a series of G = gadgets to set the output equal to the input.

Both this result and the result about simple stochastic games noted above were shown independently by [25], while Theorem 14 was already shown by Bubelis [5].

## References

- [1] T. G. Abbott, D. Kane and P. Valiant. 'On the Complexity of Two-Player Win-Lose Games,' In the 46th Annual IEEE Symposium on Foundations of Computer Science, FOCS 2005.
- [2] P. Beame, S. Cook, J. Edmonds, R. Impagliazzo and T. Pitassi. 'The Relative Complexity of NP Search Problems,' Journal of Computer and System Sciences , 57(1):13-19, 1998.
- [3] C. Borgs, J. Chayes, N. Immorlica, A. T. Kalai, V. Mirrokni and C. H. Papadimitriou. 'The Myth of the Folk Theorem,' In the 40th ACM Symposium on Theory of Computing, STOC 2008.
- [4] H. Bosse, J. Byrka and E. Markakis. 'New Algorithms for Approximate Nash Equilibria in Bimatrix Games,' In the 3rd International Workshop on Internet and Network Economics, WINE 2007.
- [5] V. Bubelis. 'On Equilibria in Finite Games,' International Journal of Game Theory , 8(2):6579, 1979.
- [6] G. J. Chang, W. Ke, D. Kuo, D. D. Liu and R. K. Yeh. 'On L(d, 1)-Labelings of Graphs,' Discrete Mathematics , 220(1-3): 57-66, 2000.
- [7] X. Chen and X. Deng. '3-NASH is PPAD-Complete,' Electronic Colloquium in Computational Complexity , TR05-134, 2005.
- [8] X. Chen and X. Deng. 'Settling the Complexity of 2-Player Nash-Equilibrium,' In the 47th Annual IEEE Symposium on Foundations of Computer Science, FOCS 2006.
- [9] X. Chen, X. Deng and S. Teng. 'Computing Nash Equilibria: Approximation and Smoothed Complexity,' In the 47th Annual IEEE Symposium on Foundations of Computer Science, FOCS 2006.
- [10] X. Chen, S. Teng and P. Valiant. 'The Approximation Complexity of Win-Lose Games,' In the 18th Annual ACM-SIAM Symposium On Discrete Algorithms, SODA 2007.
- [11] B. Codenotti, A. Saberi, K. Varadarajan and Y. Ye. 'Leontief Economies Encode Nonzero Sum Two-Player Games,' In the 17th Annual ACM-SIAM Symposium On Discrete Algorithms, SODA 2006.
- [12] A. Condon. 'The Complexity of Stochastic Games,' Information and Computation, 96(2): 203224, 1992.
- [13] V. Conitzer and T. Sandholm. 'Complexity Results about Nash Equilibria,' In the 18th International Joint Conference on Artificial Intelligence, IJCAI 2003.
- [14] G. B. Dantzig. Linear Programming and Extensions , Princeton University Press, 1963.
- [15] C. Daskalakis, A. Fabrikant and C. H. Papadimitriou. 'The Game World is Flat: The Complexity of Nash Equilibria in Succinct Games,' In the 33rd International Colloquium on Automata, Languages and Programming, ICALP 2006.
- [16] C. Daskalakis, P. W. Goldberg and C. H. Papadimitriou. 'The Complexity of Computing a Nash Equilibrium,' In the 38th ACM Symposium on Theory of Computing, STOC 2006.

- [17] C. Daskalakis, A. Mehta and C. H. Papadimitriou. 'A Note on Approximate Nash Equilibria,' In the 2nd international Workshop on Internet and Network Economics, WINE 2006.
- [18] C. Daskalakis, A. Mehta and C. H. Papadimitriou. 'Progress in Approximate Nash Equilibria,' In the 8th ACM Conference on Electronic Commerce, EC 2007.
- [19] C. Daskalakis and C. H. Papadimitriou. 'Three-Player Games Are Hard,' Electronic Colloquium in Computational Complexity , TR05-139, 2005.
- [20] C. Daskalakis and C. H. Papadimitriou. 'Computing Pure Nash Equilibria via Markov Random Fields,' In the 7th ACM Conference on Electronic Commerce, EC 2006.
- [21] C. Daskalakis and C. H. Papadimitriou. 'Computing Equilibria in Anonymous Games,' In the 48th Annual IEEE Symposium on Foundations of Computer Science, FOCS 2007.
- [22] C. Daskalakis and C. H. Papadimitriou. 'Discretized Multinomial Distributions, Covers, and Nash Equilibria in Anonymous Games,' ArXiv, 2008.
- [23] B. C. Eaves. 'Homotopies for Computation of Fixed Points,' Mathematical Programming , 3: 1-22, 1972.
- [24] E. Elkind, L. A. Goldberg and P. W. Goldberg. 'Nash Equilibria in Graphical Games on Trees Revisited,' In the 7th ACM Conference on Electronic Commerce, EC 2006.
- [25] K. Etessami and M. Yannakakis. 'On the Complexity of Nash Equilibria and Other Fixed Points,' In the 48th Annual IEEE Symposium on Foundations of Computer Science, FOCS 2007.
- [26] A. Fabrikant, C.H. Papadimitriou and K. Talwar. 'The Complexity of Pure Nash Equilibria,' In the 36th ACM Symposium on Theory of Computing, STOC 2004.
- [27] C. B. Garcia, C. E. Lemke and H. J. Luthi. 'Simplicial Approximation of an Equilibrium Point of Noncooperative N-Person Games,' Mathematical Programming , 4: 227-260, 1973.
- [28] J. Geanakoplos. 'Nash and Walras Equilibrium via Brouwer,' Economic Theory , 21, 2003.
- [29] I. Gilboa and E. Zemel. 'Nash and Correlated Equilibria: Some Complexity Considerations,' Games and Economic Behavior , 1(1): 80-93, 1989.
- [30] P. W. Goldberg and C. H. Papadimitriou. 'Reducibility Among Equilibrium Problems,' In the 38th ACM Symposium on Theory of Computing, STOC 2006.
- [31] M. Hirsch, C. H. Papadimitriou and S. Vavasis. 'Exponential Lower Bounds for Finding Brouwer Fixpoints,' Journal of Complexity , 5(4): 379-416, 1989.
- [32] D. S. Johnson, C. H. Papadimitriou and M. Yannakakis. 'How Easy is Local Search?,' Journal of Computer and System Sciences , 37(1): 79-100, 1988.
- [33] M. Kearns, M. Littman and S. Singh. 'Graphical Models for Game Theory,' In the 17th Conference in Uncertainty in Artificial Intelligence, UAI 2001.
- [34] L. G. Khachiyan. 'A Polynomial Algorithm in Linear Programming,' Soviet Mathematics Doklady , 20(1): 191-194, 1979.

- [35] B. Knaster, C. Kuratowski and S. Mazurkiewicz. 'Ein Beweis des Fixpunktsatzes f¨ ur ndimensionale Simplexe,' Fundamenta Mathematicae, 14: 132-137, 1929.
- [36] G. van der Laan and A. J. J. Talman. 'A Restart Algorithm for Computing Fixed Points Without an Extra Dimension,' Mathematical Programming , 17: 74-84, 1979.
- [37] G. van der Laan and A. J. J. Talman. 'On the Computation of Fixed Points in the Product Space of Unit Simplices and an Application to Noncooperative N Person Games,' Mathematics of Operations Research , 7(1): 1-13, 1982.
- [38] C. E. Lemke and J. T. Howson, Jr. 'Equilibrium Points of Bimatrix Games,' SIAM Journal of Applied Mathematics , 12: 413-423, 1964.
- [39] R. Lipton and E. Markakis. 'Nash Equilibria via Polynomial Equations,' In the 6th Latin American Symposium, LATIN 2004.
- [40] R. Lipton, E. Markakis and A. Mehta. 'Playing Large Games Using Simple Strategies,' In the 4th ACM Conference on Electronic Commerce, EC 2003.
- [41] M. Littman, M. Kearns and S. Singh. 'An Efficient, Exact Algorithm for Single Connected Graphical Games,' In the 15th Annual Conference on Neural Information Processing Systems , NIPS 2001.
- [42] N. Megiddo and C. H. Papadimitriou. 'On Total Functions, Existence Theorems and Computational Complexity,' Theoretical Computer Science , 81(2): 317-324, 1991.
- [43] J. Nash. 'Non-cooperative Games,' Annals of Mathematics , 54: 289-295, 1951.
- [44] J. von Neumann. 'Zur Theorie der Gesellshaftsspiele,' Mathematische Annalen , 100: 295-320, 1928.
- [45] J. von Neumann and O. Morgenstern. Theory of Games and Economic Behavior , Princeton University Press, 1944.
- [46] M.J. Osborne and A. Rubinstein. A Course in Game Theory , MIT Press, 1994.
- [47] C. H. Papadimitriou. Computational Complexity , Addison Wesley, 1994.
- [48] C. H. Papadimitriou. 'On the Complexity of the Parity Argument and Other Inefficient Proofs of Existence,' Journal of Computer and System Sciences , 48(3): 498-532, 1994.
- [49] C. H. Papadimitriou. 'Computing Correlated Equilibria in Multiplayer Games,' In the 37th ACM Symposium on Theory of Computing, STOC 2005.
- [50] C. H. Papadimitriou and T. Roughgarden. 'Computing Equilibria in Multi-Player Games,' In the 16th Annual ACM-SIAM Symposium On Discrete Algorithms, SODA 2005.
- [51] J. Rosenm¨ uller. 'On a Generalization of the Lemke-Howson Algorithm to Noncooperative N-Person Games,' SIAM Journal of Applied Mathematics , 21(1): 73-79, 1971.
- [52] A. Rubinstein. 'Equilibrium in Supergames with the Overtaking Criterion,' Journal of Economic Theory, 21: 1-9, 1979.

- [53] R. Savani and B. von Stengel. 'Exponentially Many Steps for Finding a Nash Equilibrium in a Bimatrix Game,' In the 45th Annual IEEE Symposium on Foundations of Computer Science, FOCS 2004.
- [54] H. E. Scarf. 'The Approximation of Fixed Points of a Continuous Mapping,' SIAM Journal of Applied Mathematics , 15(5): 1328-1343, 1967.
- [55] G. Schoenebeck and S. Vadhan. 'The Computational Complexity of Nash Equilibria in Concisely Represented Games,' In the 7th ACM Conference on Electronic Commerce, EC 2006.
- [56] H. Tsaknakis and P. G. Spirakis. 'An Optimization Approach for Approximate Nash Equilibria,' In the 3rd International Workshop on Internet and Network Economics, WINE 2007.
- [57] R. Wilson. 'Computing Equilibria of N-Person Games,' SIAM Journal of Applied Mathematics , 21: 80-87, 1971.