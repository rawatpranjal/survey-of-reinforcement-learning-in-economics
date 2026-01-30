## Progress on the union-closed conjecture and offsprings in winter 2022-2023

Stijn Cambie ∗

June 22, 2023

Mathematicians had little idea whether the easy-to-state union-closed conjecture was true or false even after 40 years. However, last winter saw a surge of interest in the conjecture and its variants, initiated by the contribution of a researcher at Google. Justin Gilmer made a significant breakthrough by discovering a first constant lower bound for the proportion of the most common element in a unionclosed family.

## 1 Introduction of the Union-Closed conjecture

The union-closed conjecture is due to Peter Frankl 1 , who constructed the elegant statement in 1979 after observing many implications of the statement. Before fully stating it, we need to define crucial concepts from set theory.

The ground set is generally denoted with [ n ] = { 1 , 2 , . . . , n } , where n ∈ N is a finite number. A subset A ⊆ [ n ] is nothing more than a set containing integers between 1 and n , e.g., A = { 2 , 4 , 6 } ⊂ [7].

A family F ⊆ 2 [ n ] is a collection of subsets of [ n ]. Here 2 [ n ] contains all 2 n possible subsets of [ n ], which includes the empty set ∅ as well.

A family F is called union-closed if for every A,B ∈ F , the union A ∪ B belongs to F . This can be written as F = F ∪ F , where the latter equals exactly { A ∪ B | A,B ∈ F} . An example of such a family is presented in Figure 1. An other example, for every m ∈ N , is the family F m = { A | A ⊆ [ m ] ∨ A = [ k ] for some m +1 ≤ k ≤ m 2 } which consists of the 2 m subsets of [ m ], as well as m 2 -m intervals consisting of the first k natural numbers.

Figure 1: Example of union-closed family

<!-- image -->

∗ Extremal Combinatorics and Probability Group (ECOPRO), Institute for Basic Science (IBS), Daejeon, South Korea, supported by the Institute for Basic Science (IBS-R029-C4), E-mail: stijn.cambie@hotmail.com

1 See also https://en.wikipedia.org/wiki/P%C3%A9ter\_Frankl and https://www.nrc.nl/nieuws/2023/01/20/na-wiskundige-opwinding-o

The Union-closed conjecture can now be formally stated as follows.

Conjecture 1 (Union-closed conjecture) . If F /negationslash = {∅} is a union-closed family with ground set [ n ] , then there exists an element i ∈ [ n ] such that at least half of the sets in F contain i .

Considering our previous example F m for large m , one can verify that it might be that only a small fraction of the elements of the ground set are abundant (belong to at least half of the sets) and their average proportion of sets to which they belong can tend to zero. Note that this conjecture would be (arguably) false when taking an infinite ground set N , e.g. by considering the (union-closed) family of finite subsets of N .

This conjecture can also be formulated in many different ways. For example, one can consider bitstrings in { 0 , 1 } n with the element-wise OR -operation. For instance, when n = 4 and F = { 0011 , 1100 , 1111 } , we note that 0011 + 1100 = 1111. This family is closed under the OR -operation, which corresponds to being union-closed in the initial formulation.

Taking the complements of the set, one obtains the Intersection-closed sets conjecture, which states that an intersection-closed family has an element in its ground set appearing in at most half of the sets. In [3, Sec. 3], one can also find a lattice-, graph-, and Salzborn-formulation.

On November 17, 2022, Justin Gilmer [10], a researcher at Google working in machine learning, made a breakthrough by proving a first constant fraction for Conjecture 1. Soon thereafter, as fast as a few days, his result made others put improvements and related results on the preprint server Arxiv. In this note, we summarize the contributions and progress that was made in the winter of 2022-2023. We explain the main ideas of Gilmer's approach (Section 2), mention the forthcoming extensions of his method (Sections 3 and 4), as well as an unsuccessful attempt (Section 5) and discuss other work related to the Union-closed conjecture (Section 6).

## 2 The observations and key elements in the proof by Gilmer

A first elementary observation by Gilmer is that one can always prove a statement by proving the contrapositive of that statement. Since the statement of the union-closed conjecture is that simple already, it might be no one considered that before. The contraposition of Conjecture 1 can be stated as follows. If a non-empty family F has no element appearing in at least half of the sets of F , then F is not a union-closed family. By remarking that A ∪ A = A for every set A , one knows that F ⊆ F ∪F , and thus |F ∪ F| &gt; |F| whenever F is not a union-closed family. While posing related questions and studying counterexamples to variants of Conjecture 1 similar to the ones in [8], Gilmer noted that the entropy of a family might play a role. 2 The entropy H ( X ) of a discrete random variable X equals the Shannon entropy of its probability distribution. The latter can be purely presented with a formula. If each possible outcome x belongs to a (finite) set A , and has probability p x , then

<!-- formula-not-decoded -->

When sampling uniformly at random from F , the entropy will equal log 2 |F| and no higher entropy is possible. If one can sample from F ∪ F in such a way that the entropy is larger than log 2 |F| , then one can conclude that |F ∪ F| &gt; |F| . This is exactly the core of Gilmer's approach.

More precisely, he proved the following statement.

Theorem 2. Let A and B denote independent and identically distributed random variables that sample from a common distribution over subsets of [ n ] . Assume that for all i ∈ [ n ] , P [ i ∈ A ] ≤ 0 . 01 . Then H ( A ∪ B ) ≥ 1 . 26 H ( A ) .

As a corollary, by taking the uniform distribution over the subsets of [ n ], one knows that if F ⊂ 2 [ n ] is a family for which every element is contained in no more than 1% of the sets, then |F ∪F| ≥ |F| 1 . 26 . 3

2 More details on his journey/ thought process can be found in 3 As a corollary of later work by Sawin, this is at least |Φ| 1 . 74

https://www.youtube.com/watch?v=AZaP0EwjR\_I&amp;t

This implies that whenever |F| ≥ 2, either |F ∪ F| &gt; |F| (and so the family is not union-closed) or there is an element appearing in at least a 0 . 01 fraction of the sets in F . From this, one can conclude that Conjecture 1 is true for a half replaced by 0 . 01 .

example 3. Let F = {{ 1 } , { 2 }} and thus F ∪ F = {{ 1 } , { 2 } , { 1 , 2 }} . Let A and B be i.i.d. random variables that output a set of F uniformly at random. Then P ( A = { 1 } ) = P ( A = { 2 } ) and analogously for B , which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now H ( A ) = 2 · 1 2 log 2 2 = 1 and H ( A ∪ B ) = 2 · 1 4 log 2 4 + 1 2 log 2 = 3 2 ( &lt; log 2 3) . Since log 2 (2) &lt; H ( A ∪ B ) , we conclude that it is impossible that A ∪ B takes values in a family with only 2 elements and thus |F ∪ F| &gt; |F| , i.e. Gilmer's method verifies that F is not union-closed.

example 4. Let F = ( [3] ≤ 2 ) and thus F ∪ F = 2 [3] . Note that |F| = 7 and every 1 ≤ i ≤ 3 appears in exactly 3 sets and thus in a 3 7 fraction. Let A,B be i.i.d. random variables that output a set of F uniformly at random. Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and thus H ( A ) &gt; H ( A ∪ B ). We conclude that this is an example for which Gilmer's method does not provide evidence that the family is not union-closed, even while the maximum fraction of occurence of an element is 3 7 .

Note: Analogously, when F = ( [5] ≤ 3 ) , one can verify that H ( A ) = log 2 (26) ∼ 4 . 7 and H ( A ∪ B ) ∼ 4 . 54 . Every element appears in a 11 26 fraction in this case.

## 3 Quick refinement of Gilmer's idea

The binary entropy function h ( p ) = -( p log 2 p +(1 -p ) log 2 (1 -p )) plays a role in the computations in the work of Gilmer. Noting that h ( p ) ≤ h (2 p -p 2 ) whenever p ≤ ψ := 3 - √ 5 2 , Gilmer claimed that his ideas could be extended to prove a fraction equal to ψ. The authors of [1, 5, 18, 15] quickly implemented this approach. All four of these papers essentially reduced Conjecture 1 for the constant ψ to the following key lemma, an inequality in one variable.

Now

<!-- formula-not-decoded -->

The validity of this lemma was established in two different ways by [1] and Sawin [18]. The former used accurate computer calculations and applied interval arithmetic on three intervals, while the latter utilized a purely calculus-based approach. Thanks to some communication between the authors of [1] and [5], in [5] a reference to the formal proof of [1] was added. In [15] the lemma was split in two parts without formal proof, but both can be verified easily.

A short and more elegant proof for Lemma 5 was given later by Boppana [2], even while the proof itself would originate from 1989. This proof relies on the following extension of the classical Rolle's theorem, which follows from observations in e.g. [12].

Theorem 6. Let f be a differentiable function on a interval I . Let m ( f ) be the sum of multiplicities of the roots of f in I . Then m ( f ′ ) ≥ m ( f ) -1 .

By iterating the theorem three times, one finds m ( f ) ≤ m ( f ′′′ ) + 3. Applying this result on the function f ( x ) = h ( x 2 ) -φxh ( x ) and counting the multiplicities of the roots 0 , 1 φ and 1 of f , the conclusion that f is nonnegative on [0 , 1] follows quickly. Once Lemma 5 is derived, the proof for Conjecture 1 for constant ψ (instead of 0 . 5) is rather short in each of the papers [1, 5, 15, 18], indicated e.g. by the total length of the paper by Chase and Lovett [5]. Their work has three steps. First, they extended the analytic claim (Lemma 5) to the two-variate function f ( x, y ) := h ( xy ) h ( x ) y + h ( y ) x . Next they prove a strengthened inequality between the entropy of A ∪ B and the one of A and B , for random variables A and B (not necessarily identical) on { 0 , 1 } n for which every bit is 1 with a bounded probability. Finally, they finish the proof of their slightly more general statement that holds for approximate union-closed families. The latter being families for which the union of two random drawn sets belong to the family with a high probability.

One example which certifies the sharpness of their proof can be derived from F 1 + F 2 = { A | A ∈ F 1 ∨ A ∈ F 2 } where F 1 = ( [ n ] ψn + n 2 / 3 ) and F 2 = ( [ n ] ≥ (1 -ψ ) n ) . For this, one need to note that |F 1 | &gt;&gt; |F 2 | and that the union of two (iid uniform sampled) random sets from F 1 belongs with very high probability to F 2 . The expected size of the union is slightly larger (with an additional term of the order n 2 / 3 , i.e. Θ( n 2 / 3 )) than n -(1 -ψ ) 2 n = (1 -ψ ) n , and since the variance on the size is O ( n 1 / 2 ), the union almost surely belongs to F 2 as well. The conclusion is still valid when replacing the term n 2 / 3 by any function g ( n ) for which n &gt;&gt; g ( n ) &gt;&gt; n 1 / 2 .

Figure 2: An approximate union-closed family whose elements appear in at most a ψ + o (1) fraction.

<!-- image -->

In a different direction, in his paper, Gilmer included some ideas for a full resolution of Conjecture 1, but some of these directions were immediately proven not to hold by Sawin and Ellis [18, 7].

## 4 Further refinements and extensions related to Gilmer's work

Sawin [18] gave a suggestion to improve the bound further, which given the sharpness of the form for union-closed families may be considered surprising. Hereby the essence is in a question purely stated in terms of probability distributions. His suggestion was worked out by Yu [20] and Cambie [4]. Yu [20] considered the approach in a slightly more general form initially and made a lower bound computable by restricting to the suggestion of Sawin and applying [1, Lem. 5] and the Krein-Milman theorem [13] to bound the support (number of values with nonzero probability) of a joint distribution by 4. A numerical computation then yield a bound equal to (roughly) 0 . 38234 . In parallel, Cambie [4] found an upper bound for Sawin's approach which indicates that the improvement is way smaller than expected and one would hope for. The construction is a discrete probability distribution with only two values having nonzero probability, with the values determined by a system of equations involving the entropy function. Additionally he proved that this value is sharp, by first reducing the support to 3 elements, where one of the elements equals 1 . Finally, the conclusion is derived from the combination of 3-dimensional plots, a numerical minimization problem and a more precise solution for the case where the support has exactly two elements, one of which equals 1.

Finally, building upon the work of [5], Yuster [21] considered families that are almost k -unionclosed, meaning that the union of k independent uniform random sets from F belongs to F with high probability. He conjectured a tight version for the minimum frequency (the proportion of sets containing the element) of some element in such families, with the threshold for this frequency being the unique real root in [0 , 1] of (1 -x ) k = x , denoted by ψ k . To understand the sharpness of his conjecture and the intuition behind the choice of ψ k , consider the union of F 1 = ( [ n ] ψ k n + n 2 / 3 ) and F 2 = ( [ n ] ≥ (1 -ψ k ) n ) . If at least one set from F 2 is included among the k sets drawn, the union is guaranteed to belong to F 2 . If all k sets belong to F 1 , the expected size of the union is n -(1 -ψ k ) k n +Θ( n 2 / 3 ), and since the variance is O ( n 1 / 2 ), the union almost surely belongs to F 2 as well. The conjecture is proven to be true for k ≤ 4, while for larger values of k a weaker bound is established.

## 5 The final Eureka moment, not yet

When Scandone [19] uploaded a preprint claiming the full resolution of the union-closed conjecture, there arose initially excitement. However, upon closer examination it became clear that Scandone's proposed solution had several issues, including a significant flaw that requires revising the underlying construction. This was communicated to Scandone by Terence Tao, and the details of this issue are briefly explained later in this section.

Nevertheless, Scandone's underlying idea holds potential and is worth mentioning for the valuable intuition it provides for Gilmer's approach. Let F be a family which is not union-closed, so F∪F /negationslash = F . A random variable taking values in F has entropy at most log 2 |F| and equality occurs only for uniform sampling from F . By considering various examples, e.g. F = {{ 1 } , { 2 }} , the reader can verify that there is no strategy to choose two random variables A,B which sample sets from F , such that A ∪ B samples uniformly random from F ∪ F . On the other hand, if for every set A ∈ F the probability of obtaining it is almost equal to the original probability and a few other sets from ( F ∪ F ) \F happen with a small probability, the entropy can increase. The reason for this is that the derivative of h (plotted in Figure 3) is a continuously decreasing function on the interval (0 , 1), with h ′ (0) = + ∞ . To provide a more explicit explanation of Scandone's idea, we describe his proposed construction in detail.

Let A,B be independent random variables that take any set of F uniformly at random. Define a P ([ n ])-valued random variable A δ (depending on δ ) through the relation

<!-- formula-not-decoded -->

For every X ∈ F , Pr[ A δ = X ] ≥ (1 -δ ) Pr[ A = X ] and thus for δ sufficiently small, we have

h (Pr[ A δ = X ]) -h (Pr[ A = X ]) /greaterorsimilar δ/ |F| h ′ (1 / |F| ) . 4 On the other hand, for X ∈ ( F ∪ F ) \F , let the probability p := Pr[ A ∪ B = X ]. We have that h ( δp ) ∼ -δp (log δ +log p -1). By choosing δ to be sufficiently small such that -log δ is much greater than 1 p h ′ (1 / |F| ), we can ensure that H ( A δ ) &gt; H ( A ) holds.

Figure 3: Plot of the binary entropy function h

<!-- image -->

Equivalently, the variable A δ can be obtained by considering, in addition to A and B , a Bernoulli random variable of parameter δ , Z δ , which determines whether we take A ∪ B or only A . The flaw in the argument is that, in the process of revealing all the digits of A δ (computed using the chain rule for the entropy), the indeterminacy provided by Z δ (and the consequent improvement of the bounds) is lost after the first step. More precisely, there is step in the computations in which a conditional probability distribution has been erroneously replaced by its expected value, and this produces the aforementioned flaw in the argument. The comment of Tao can be rephrased as follows, 'the idea of modifying the union operation by Gilmer is promising, but a single global bit Z δ is not sufficient to do the job, and a more involved construction is needed'.

## 6 A better understanding by progress in a different direction

In this final section, we conclude with the essence of a recent paper and two preprints on the unionclosed conjecture, which consider different aspects and angles of attack on Conjecture 1.

While Frankl's conjecture is about the existence of one abundant element (element that appears in at least half of the sets) in the family, it is also natural to wonder if there are more abundant elements, assuming that all sets in the family are sufficiently large. The following conjecture by Cui and Hu [6] would imply Conjecture 1.

Conjecture 7. If F is a finite union-closed family of sets whose smallest set is of size at least 2 , then there are at least two elements such that each belong to more than half of the sets of F .

At the end of 2022, the three authors of [11] considered this different direction and proved that Conjecture [6] is not true when replacing 2 by a larger integer. They proved (among other results) that there are families all of whose sets have size at least k , where k can be arbitrary large, which do only have 2 abundant elements. The main construction is the family P 12 4 . The family P 12 4 consists of all subsets S of { 0 , 1 , . . . , 11 } of size at least 4 such that either { 0 , 1 } ⊂ S , or 0 ∈ S and S ⊆ { 0 , 2 , . . . , 10 } , or 1 ∈ S and S ⊆ { 1 , 3 , . . . , 11 } . The reader can verify that |P 12 4 | = (2 10 -11) + 2 · 16 = 1045, while every element 2 ≤ i ≤ 11 only appears 2 9 -1+11 = 522 times. One way to increase the size of sets in families with non-abundant elements is to duplicate an element within the sets. However, this creates blocks of size at least 2. A block is defined by Poonen [16] as a maximum set of elements that all belong to the exact same sets of a family. Poonen also noted that to prove Conjecture 1, it is sufficient to focus on families for which no block is a singleton. Due to this, it is interesting to note that the

4 To be precise, we assume |Φ| ≥ 3 and 2 |Φ| + δ &lt; 1 .

construction of the family P 12 4 in [11] can be extended to such families.Let k ≥ 3 be a fixed integer and let n be a sufficiently large even integer as a function of k ( n ≥ 10 k works). Let E n = { i ∈ [ n ] | i ≡ 0 (mod 2) } and O n = { i ∈ [ n ] | i ≡ 1 (mod 2) } be the set of even and odd integers in [ n ] respectively. Consider the family P n k consisting of subsets S of [ n ] of size at least k , such that either

- { 1 , 2 } ⊂ S ,
- S ⊂ E n and 2 ∈ S , or
- S ⊂ O n and 1 ∈ S .

It is clear that 1 and 2 are abundant elements. Now the other elements appear all equally often (by symmetry) and by a small bijection and counting argument, we conclude that these elements are not abundant whenever

<!-- formula-not-decoded -->

Since this is the case for n sufficiently large, the conclusion is clear.

Another result related with union-closed families and the smallest set size, was published early 2023. Ellis, Ivan and Leader [9] proved that for every k ∈ N , there exists a union-closed family in which the (unique) smallest set has size k , but where each element of this set has frequency (1 + o (1)) log k 2 k . As such, proving that focusing on the smallest set cannot work in the strongest possible sense. They also proposed the problem of verifying the union-closed conjecture for a family for which they were unable to verify the statement. The latter was verified by Pulaj and Wood [17]. They also proved new bounds on the least number m (given k and n ) such that every union-closed family F containing any A ⊆ [ n ] k with |A| = m as a subfamily, satisfies Conjecture 1.

Note added: In June 2023, Liu [14] improved the constant slightly with a different method of coupling.

( ) We can conclude that despite the progress that originates from the breakthrough of Justin Gilmer, the exact version of Conjecture 1 is still not proven. Mathematicians are still thinking about other directions or modifications of the strategy and hope to resolve Conjecture 1 in the future. Taking into account that the improvement by taking combinations suggested by Sawin [18] turned out to be tinier than expected and hoped for, as illustrated by the example in [4], it seems that the focus should go towards essential new ideas. In particular, the union-closed conjecture might be a distraction of a more general behaviour that |F ∪ F| &gt; |F| c for some c ( ε ) &gt; 1 when every element of [ n ] appears in less than a 1 2 -ε fraction of the sets in F . 5

## Acknowledgements

We thank Zachary Chase, Justin Gilmer, Raffaele Scandone and Lei Yu for internal communication while writing this manuscript.

## References

- [1] R. Alweiss, B. Huang, and M. Sellke. Improved Lower Bound for the Union-Closed Sets Conjecture. arXiv e-prints , page arXiv:2211.11731, Nov. 2022.
- [2] R. B. Boppana. A Useful Inequality for the Binary Entropy Function. arXiv e-prints , page arXiv:2301.09664, Jan. 2023.
- [3] H. Bruhn and O. Schaudt. The journey of the union-closed sets conjecture. Graphs Combin. , 31(6):2043-2074, 2015.

5 communicated by Zachary Chase

- [4] S. Cambie. Better bounds for the union-closed sets conjecture using the entropy approach. arXiv e-prints , page arXiv:2212.12500, Dec. 2022.
- [5] Z. Chase and S. Lovett. Approximate union closed conjecture. arXiv e-prints , page arXiv:2211.11689, Nov. 2022.
- [6] Z. Cui and Z. Hu. Two stronger versions of the union-closed sets conjecture. Adv. Math. (China) , 50(6):829-851, 2021.
- [7] D. Ellis. Note: a counterexample to a conjecture of Gilmer which would imply the union-closed conjecture. arXiv e-prints , page arXiv:2211.12401, Nov. 2022.
- [8] D. Ellis. Union-closed families with small average overlap densities. Electron. J. Combin. , 29(1):Paper No. 1.11, 5, 2022.
- [9] D. Ellis, I. Leader, and M.-R. Ivan. Small Sets in Union-Closed Families. Electron. J. Combin. , 30(1):Paper No. 1.8-, 2023.
- [10] J. Gilmer. A constant lower bound for the union-closed sets conjecture. arXiv e-prints , page arXiv:2211.09055, Nov. 2022.
- [11] A. Kabela, M. Polák, and J. Teska. The number of abundant elements in union-closed families without small sets. arXiv e-prints , page arXiv:2212.09279, Dec. 2022.
- [12] V. P. Kostov. On arrangements of real roots of a real polynomial and its derivatives. Serdica Math. J. , 29(1):65-74, 2003.
- [13] M. Krein and D. Milman. On extreme points of regular convex sets. Studia Math. , 9:133-138, 1940.
- [14] J. Liu. Improving the Lower Bound for the Union-closed Sets Conjecture via Conditionally IID Coupling. arXiv e-prints , page arXiv:2306.08824, June 2023.
- [15] L. Pebody. Extension of a Method of Gilmer. arXiv e-prints , page arXiv:2211.13139, Nov. 2022.
- [16] B. Poonen. Union-closed families. J. Combin. Theory Ser. A , 59(2):253-268, 1992.
- [17] J. Pulaj and K. Wood. Local Configurations in Union-Closed Families. arXiv e-prints , page arXiv:2301.01331, Jan. 2023.
- [18] W. Sawin. An improved lower bound for the union-closed set conjecture. arXiv e-prints , page arXiv:2211.11504, Nov. 2022.
- [19] R. Scandone. A proof of the union-closed sets conjecture. arXiv e-prints , page arXiv:2302.03484, Feb. 2023.
- [20] L. Yu. Dimension-Free Bounds for the Union-Closed Sets Conjecture. arXiv e-prints , page arXiv:2212.00658, Dec. 2022.
- [21] R. Yuster. Almost k -union closed set systems. arXiv e-prints , page arXiv:2302.12276, Feb. 2023.