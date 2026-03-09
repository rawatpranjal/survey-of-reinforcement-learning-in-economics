## How to compute multi-dimensional stable and unstable manifolds of piecewise-linear maps.

D.J.W. Simpson

School of Mathematical and Computational Sciences Massey University Palmerston North, 4410 New Zealand

October 17, 2023

## Abstract

For piecewise-linear maps the stable and unstable manifolds of hyperbolic periodic solutions are themselves piecewise-linear. Hence compact subsets of these manifolds can be represented using polytopes (i.e. polygons, in the case of two-dimensional manifolds). Such representations are efficient and exact so for computational purposes are superior to representations that use a large number of points on some mesh (as is usually done in the smooth setting). We introduce a method for computing convex polytope representations of stable and unstable manifolds. For an unstable manifold we iterate a suitably small subset of the local unstable manifold and prior to each iteration subdivide polytopes where they intersect the switching manifold of the map. We prove the output converges to the (entire) unstable manifold and use it to visualise attractors and bifurcations of the three-dimensional border-collision normal form: we identify a heterodimensional-cycle, a two-dimensional unstable manifold whose closure appears to be a unique attractor, and a piecewise-linear analogue of a first homoclinic tangency where an attractor appears to be destroyed.

## 1 Introduction

For nonlinear dynamical systems it is often extremely helpful to understand the global behaviour of the stable and unstable manifolds of saddle-type invariant sets. This is because stable manifolds typically form boundaries for basins of attraction and under parameter variation chaos can be generated and attractors can be destroyed when stable and unstable manifolds first intersect [9, 16].

One-dimensional manifolds are relatively easy to compute. To compute a one-dimensional unstable manifold of an equilibrium x ∗ of a system of ordinary differential equations, one just

f3(U)

f(U)

fª(U)

needs to evolve two points x ∗ ± εv , where v is the associated unstable eigenvector and ε &gt; 0 is suitably small. For a map (system of difference equations) one iterates a large number of points distributed across a fundamental domain [12]. Computations of two-dimensional manifolds require more effort and many methods have been developed for doing this [10]. However, these are designed for smooth dynamical systems for which the manifolds are curved surfaces. To represent these computationally it is necessary to use a large number of points on some two-dimensional mesh.

Here we present a new method for computing multi-dimensional stable and unstable manifolds of piecewise-linear maps. For simplicity we only treat maps that are continuous and have a single switching manifold, but the same approach should be effective for discontinuous maps and maps with several switching manifolds. The method works by repeatedly iterating a suitable initial local approximation U , and a typical output is shown in Fig. 1. At each step the method generates a compact subset of the manifold and represents it as a union of convex polytopes. This approach is based on the fact that the image under the map of a convex polytope is either another convex polytope or a union of two convex polytopes. Computationally each polytope is represented by the convex hull of its vertices. In this way relatively large subsets of the manifold can be characterised using relatively few points.

Below we prove that the computation limits to the entire manifold as the number of steps tends to infinity and use the method to compute various two-dimensional manifolds. Higher dimensional manifolds are not attempted here as they are significantly more difficult to visualise, plus require algorithms in computational geometry to encode the polytopes and split them at the switching manifold [18]. Such algorithms are not needed for two-dimensional

Figure 1: A numerically computed unstable manifold of a saddle fixed point (green dot) with two unstable directions. This figure is for the three-dimensional border-collision normal form (2.1) with (5.1) using the parameter values (5.2). Each plot shows the switching manifold Σ, where the map is continuous but non-differentiable. The computation starts with a quadrilateral U (a subset of the local unstable manifold), and after 12 iterations the computed manifold consists of 533 polygons (bottom right).

<!-- image -->

manifolds because the vertices of polygons admit a natural cyclical ordering.

This work is motivated by a need to better understand how dynamics changes at bordercollision bifurcations where a fixed point of a piecewise-smooth map collides with a switching manifold and the local dynamics is captured by a piecewise-linear approximation [20]. Bordercollision bifurcations have been heavily studied for one-dimensional maps [15, 22] and twodimensional maps [1, 14] where stable and unstable manifolds of saddles are at most onedimensional. Recent studies have revealed novel dynamics in three dimensions [13]. To understand these further detailed computations of two-dimensional manifolds should prove useful. Since the method computes the manifolds extremely accurately (with linearity no approximations are needed), their computations (even of manifolds that are more than two dimensional) could be used in computer-assisted proofs [7].

We start in § 2 by clarifying the class of maps under consideration and how periodic solutions can be encoded symbolically. In § 3 we review elementary properties of convex polytopes that underpin the computations. In § 4 we explain the method in more detail and prove that the computed subsets converge to the (full) manifold in the limit of infinitely many iterations. In § 5 we discuss some details for how the method can be implemented for two-dimensional manifolds and illustrate this with the three-dimensional border-collision normal form. Finally § 6 provides a brief discussion.

## 2 Periodic solutions and symbolic itineraries

Here we discuss the basic aspects of periodic solutions of piecewise-linear maps that will be needed below. Further details on this topic can be found in [20].

We consider maps on R n of the form

<!-- formula-not-decoded -->

where A L and A R are n × n matrices and b, c ∈ R n . The map is assumed to be continuous on the switching manifold c T x = 0, thus A L and A R differ by a rank-one matrix, specifically A R = A L + ac T for some a ∈ R n . We assume c is not the zero vector so that the switching manifold, call it Σ, is a codimension-one manifold (in fact a hyperplane). If det( A L ) det( A R ) &gt; 0 the map is invertible, meaning every x ∈ R n has a unique preimage f -1 ( x ).

To describe orbits symbolically we use the following definition. For any x ∈ R n not belonging to Σ, define

<!-- formula-not-decoded -->

In this paper we will not need to assign a symbol to points on Σ.

Now suppose (2.1) has a periodp solution γ . For any y ∈ γ we can express γ as the ordered set { y, f ( y ) , . . . , f p -1 ( y ) } . These points are distinct and f p ( y ) = y . Assuming γ has no points on Σ, we can use y to define the word

<!-- formula-not-decoded -->

That is, X = X 0 X 1 · · · X p -1 where X i = σ ( f i ( y )) for each i = 0 , 1 , . . . , p -1. We then refer to γ as an X -cycle. Different points in γ generate different words, but these words will all be cyclic permutations of one another. Thus any periodp solution with no points on Σ is an X -cycle for a word X of length p that is unique up to cyclic permutation.

Stable and unstable manifolds of periodic solutions are defined as follows.

## Definition 2.1. The stable manifold of γ is

and the unstable manifold of γ is

<!-- formula-not-decoded -->

W u ( γ ) = { x ∈ R n ∣ ∣ x has a sequence of preimages converging to γ } .

Continuing to assume γ has no points on Σ, each point of γ has a neighbourhood in which f p is smooth. Thus we can use classical dynamical systems theory to help us understand the nature of its stable and unstable manifolds.

In fact in this neighbourhood f p is affine and can be expressed explicitly as follows. By composing the pieces of f in the order specified by the word X , in a neighbourhood of y ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In this neighbourhood D f p ( x ) = M X , thus the stability multipliers of γ are the eigenvalues of M X . Notice the same eigenvalues result using any other point in γ because these eigenvalues are independent of the cyclical ordering of the p matrices in (2.4).

If none of the eigenvalues has unit modulus, as is generically the case, then γ is hyperbolic. In this case we can apply the local stable manifold theorem [19]. Any point in γ , say y , is a fixed point of f p , and its stable and unstable subspaces are the invariant subspaces of x ↦→ M X x that are aligned with all stable and unstable directions, respectively. The local stable manifold theorem ensures y has local stable and unstable manifolds of the same dimensions as the corresponding subspaces and tangent to these subspaces at y . Further, W u ( γ ) can be constructed by iterating the local unstable manifold under f , and if f is invertible W s ( γ ) can be constructed by iterating the local stable manifold under f -1 .

But in our piecewise-linear setting the local stable and unstable manifolds coincide with the appropriately translated stable and unstable subspaces. For this reason the small set U that we will use to initialise our computation is not an approximation to a small part of the stable or unstable manifold, it is part of the manifold.

where

aff(P)

P

## 3 Fundamentals of convex polytopes

Here we clarify the definition of a convex polytope and the concepts of dimension and relative interior that will be needed below. For a more gradual and detailed introduction to these topics refer to the first three sections of Brøndsted [3].

A set C ⊂ R n is convex if (1 -s ) x + sy ∈ C for all x, y ∈ C and 0 ≤ s ≤ 1. The convex hull of a set S ⊂ R n is the smallest convex set containing S .

Definition 3.1. A set P ⊂ R n is a convex polytope if it is the convex hull of a non-empty finite set.

Fig. 2 shows an convex polytope in R 3 . Here P is the convex hull of four points (its vertices). These points lie on a plane, so P is two-dimensional. Formally the dimension of a convex polytope can be defined as follows.

A set V ⊂ R n is an affine subspace it is a translate of a (linear) subspace. The affine hull of a set S ⊂ R n is the smallest affine subspace containing S .

Definition 3.2. The dimension of a convex polytope is the dimension of its affine hull (which is the dimension of its linear translate).

Below we use the concept of relative interior instead of interior because if the dimension of a convex polytope P ⊂ R n is less than n , then its interior (with respect to R n ) is the empty set.

Definition 3.3. The relative interior of a convex polytope is its interior taken with respect to its affine hull.

In the case of polygons (two-dimensional polytopes), a point in the polygon belongs to its relative interior if and only if it is not a vertex or lies on an edge of the polygon, see again Fig. 2.

The following result is readily proved by direct means [3].

Lemma 3.1. Let P be the convex hull of x 1 , . . . , x K ∈ R n and let g : R n → R n be an affine map. Then g ( P ) is the convex hull of g ( x 1 ) , . . . , g ( x K ) .

Figure 2: A sketch of a convex polygon P ⊂ R 3 , its affine hull aff( P ), and a point x that belongs to the relative interior of P .

<!-- image -->

Next we consider polytopes in the phase space of a piecewise-linear map (2.1). Recall, Σ denotes the switching manifold c T x = 0.

Definition 3.4. A connected set S ⊂ R n crosses Σ if there exist x 1 , x 2 ∈ S with c T x 1 &lt; 0 and c T x 2 &gt; 0.

If a convex polytope crosses Σ, then Σ divides the polytope into two sets, Fig. 3. These sets are themselves convex polytopes. This is a trivial consequence of the exterior representation of a convex polytope: any convex polytope can be expressed as the convex hull of its vertices (the interior representation) or as the intersection of a finite set of half-spaces (the exterior representation). Thus Σ merely introduces an additional half-space; also the dimension is unchanged:

Lemma 3.2. A d -dimensional convex polytope that crosses Σ is the union of two d -dimensional convex polytopes that do not cross Σ .

Figure 3: A sketch of the intersection of a convex polygon P ⊂ R 3 with the switching manifold Σ.

<!-- image -->

## 4 Computing unstable manifolds

Here we explain our method for computing multi-dimensional unstable manifolds of a piecewiselinear map f (2.1). If the map is invertible then stable manifolds can be computed in the same way by using f -1 in place of f .

Let γ be a periodp solution with no points on Σ. Assume γ is hyperbolic, let y ∈ γ , and let X be the word (2.2). Let H be the affine subspace that contains y , has the same dimension as the unstable subspace of M X , and is tangent to this subspace. As discussed in § 2, the branch of W u ( γ ) that emanates from y does so coincident with H .

Let d be the dimension of H (this is the unstable index of γ ), and assume 1 ≤ d ≤ n -1. Let U ⊂ H be a d -dimensional convex polytope that contains y in its relative interior and obeys the following admissibility condition: for every vertex v of U there exists a sequence of preimages { f -i ( v ) } ∞ i =1 satisfying

<!-- formula-not-decoded -->

This condition ensures that U is not too big. It says that each vertex of U has preimages repeatedly the following the symbols in X (in reverse order). Since H contains only unstable directions, these preimages converge to γ . The convexity of U then gives the following result (proved in Appendix A).

Lemma 4.1. With the above assumptions, every point in U belongs to W u ( γ ) .

Now let Q (0) = P (0) 1 = U , and for all i ≥ 1 define

<!-- formula-not-decoded -->

Computationally we represent each Q ( i ) as a union of convex polytopes:

<!-- formula-not-decoded -->

To do this we use the images of the polygons in Q ( i -1) to form Q ( i ) : by Lemmas 3.1 and 3.2 the image under f of any convex polytope is either another convex polytope or the union of two convex polytopes.

Observe Q ( i ) belongs to the branch of W u ( γ ) that contains f i mod p ( y ) ∈ γ . Thus

<!-- formula-not-decoded -->

where r ≥ 0, includes exactly one part of each branch of W u ( γ ). The following result justifies using Z ( r ) with a large value of r to approximate W u ( γ ).

Theorem 4.2. For any x ∈ R n the following are equivalent:

- i) x ∈ W u ( γ ) ;
2. ii) there exists N ≥ 0 such that x ∈ Z ( r ) for all r ≥ N .

Proof. In a neighbourhood N of y where f p is affine, H is locally invariant under f p . Since U ⊂ H has y in its relative interior and H contains only unstable directions, no point on the boundary of U ∩ N (taken with respect to H ) converges to y in N under f p . Thus there exists a d -dimensional subset V ⊂ U with y in its relative interior such that V ⊂ f /lscriptp ( U ) for all /lscript ≥ 0. Hence V ⊂ Q ( /lscriptp ) for all /lscript ≥ 0, so V ⊂ Z ( r ) for all r ≥ 0.

Now suppose x ∈ W u ( γ ). Then x has a sequence of preimages under f that converges to γ . Since V ⊂ H has y in its relative interior and contains all unstable directions, this sequence must contain a point w ∈ V . Let N ≥ 0 be such that x = f N ( w ). Then w ∈ Z ( r ) for all r ≥ 0, thus x ∈ Z ( r ) for all r ≥ N .

Conversely suppose there exists N ≥ 0 such that x ∈ Z ( r ) for all r ≥ N . Let i ∈ { 0 , 1 , . . . , p -1 } be such that x ∈ Q ( N + i ) , and w ∈ U be such that f N + i ( w ) = x . Then w ∈ W u ( γ ) by Lemma 4.1, thus x ∈ W u ( γ ) as required.

## 5 Two-dimensional implementation and examples

In this section we first give details of the implementation of the method for two-dimensional unstable manifolds, then describe three examples.

To initialise the method we need a suitable set U . To construct U we first evaluate y = ( I -M X ) -1 P X b (the unique fixed point of (2.3)), and use the eigenvectors of M X to identify a basis { u 1 , u 2 } of the unstable subspace. Any point in H can then be written as y + k 1 u 1 + k 2 u 2 , for some k 1 , k 2 ∈ R .

Recall H is invariant under (2.3). Thus the restriction of (2.3) to H can be expressed as an invertible two-dimensional map on the values of k 1 and k 2 . This map, call it g , will be needed in a moment.

Let K ≥ 3 be the number of vertices we want U to have. To ensure y lies in the relative interior of U , we define K pairs of points ( k 1 , k 2 ) equispaced on a circle in R 2 centred at the origin, and use these to construct K points y + k 1 u 1 + k 2 u 2 to be the vertices of U . These points will satisfy the admissibility condition (4.1) if the radius of the circle is sufficiently small. Numerically (4.1) can be verified for all i = 1 , 2 , . . . , 1000, say (in which case it is almost certainly true for all i ≥ 1), by explicitly computing preimages. To compute the preimages it is important to use g -1 because γ is a saddle thus numerical (round-off) error will mean that preimages of f will not converge to γ .

Once an appropriate set U has been identified, we store P (0) 1 = U as a list of vertices, and also call this Q (0) . By (4.2), each Q ( i ) is a union of m i convex polygons P ( i ) j , each of which is stored as a list of vertices. To compute each Q ( i +1) we perform the following steps to each P ( i ) j . First we evaluate c T v at every vertex v of P ( i ) j . If no two of these values have different signs then P ( i ) j does not cross Σ by convexity. Otherwise P ( i ) j does cross Σ, in which case its edges intersect Σ at exactly two points. It is a simple coding exercise to compute these points then use them and the vertices of P ( i ) j to form two convex polygons that do not cross Σ and whose union is P ( i ) j . In either case we then map every vertex under f to create one or two polygons in Q ( i +1) (this step is justified by Lemma 3.1).

We now show computed manifolds of the three-dimensional border-collision normal form. This is the map (2.1) on R 3 with

<!-- formula-not-decoded -->

and is a normal form in the sense that any three-dimensional map of the form (2.1) that satisfies a certain genericity condition (observability) can be transformed to the normal form under an affine change of coordinates [5]. Studies of bifurcations in the three-dimensional normal form include [4, 13, 17, 21].

First we consider the normal form with parameter values

<!-- formula-not-decoded -->

Х3

0

-1 -

1 -.

C

fixed point

These values were obtained by starting with Figure 1 of [6] for an instance of the twodimensional normal form (which corresponds to δ L = δ R = 0 in (5.1)) for which the map has a two-dimensional attractor, and altering the parameter values slightly to create fully three-dimensional dynamics. Numerical simulations (not shown) suggest that with (5.2) the normal form has a chaotic attractor that is equal to the closure of the unstable manifold of a fixed point. This manifold is two-dimensional and its computation is shown in Fig. 4 using i = 12 iterations of the quadrilateral U shown in Fig. 1. The computation gives f 12 ( U ) as the union of 533 polygons. These are plotted as semi-transparent surfaces and provides some tangible impression of the presumably fractal geometry of the attractor.

Next we use the values

<!-- formula-not-decoded -->

obtained by altering values used in Figure 8 of [7]. With (5.3) the normal form has a saddle fixed point with a one-dimensional unstable manifold and a two-dimensional stable manifold. From their numerical computation, Fig. 5, we find that these manifolds nearly intersect at points far from the fixed point. For instance, near the middle of the figure the stable manifold (blue) has 'spikes' close to the unstable manifold (red). This suggests that the values (5.3) are close to where the stable and unstable manifolds first intersect (indeed with instead τ L = 1 . 51 the right branch of the unstable manifold diverges). Such an intersection is a piecewise-linear analogue of a first homoclinic tangency. Numerical simulations suggest that an attractor is destroyed when the parameter values are perturbed to create such an intersection.

Figure 4: Aplot of the phase space of the three-dimensional border-collsion normal form with parameter values (5.2) showing a two-dimensional unstable manifold whose closure appears to be a chaotic attractor. This is an enlarged view of the bottom right plot of Fig. 1.

<!-- image -->

Х3

X3

0.2.

2

0.1

1 -

0.

0

-0.1 .

-1

-1.2

-2 -

-2

-1

Finally we consider the values fixed point

-0.8

-0.6

X2

<!-- formula-not-decoded -->

S

-0.4

X2

-0.2

<!-- image -->

0.2

-0.4

Figure 5: The two-dimensional stable (blue) and one-dimensional unstable (red) manifolds of a fixed point of the three-dimensional border-collision normal form with parameter values (5.3). The stable manifold appears to form the boundary of the basin of attraction of a chaotic attractor that, when parameters are varied (e.g. τ L is increased slightly), is destroyed due to the stable and unstable manifolds developing non-trivial intersections.

Figure 6: A plot of the phase space of the three-dimensional border-collsion normal form with parameter values (5.4) showing the stable (blue) and unstable (red) manifolds of a fixed point and LLR -cycle. Their respective stable and unstable manifolds intersect at q and along S , hence these sets have a heterodimensional-cycle.

<!-- image -->

used for Figure 6 of [8]. Here the normal form has a saddle fixed point (the green point in Fig. 6) with unstable index two, and a saddle LLR -cycle (the yellow points in Fig. 6) with unstable index one. The value of τ L was chosen in [8] (accurate to ten decimal places) so that the one-dimensional stable manifold of the fixed point intersects the one-dimensional unstable manifold of the LLR -cycle (e.g. at the point q ). The two-dimensional manifolds were not computed in [8]; they have been computed here and found to intersect (e.g. along the line segment S ). For clarity the manifolds have only been grown as far as needed to identify these intersections.

Together the intersections show that there exists a heteroclinic connection from the fixed point to the LLR -cycle, and another connection from the LLR -cycle back to the fixed point. Significantly, the fixed point and LLR -cycle have different unstable indices. Consequently the connections form a heterodimensional cycle ; such cycles are well known for generating non-hyperbolic dynamics [2].

## 6 Discussion

This paper has introduced a method for efficiently computing multi-dimensional stable and unstable manifolds of piecewise-linear maps. The method simply iterates polytopes, and it remains to see if variations on this approach can produce improved performance. For instance, the method does not explicitly manage cases where the dynamics on the manifold expands relatively strongly in one direction that would cause the computation to produce a subset of the manifold that is stretched greatly in one direction. For stable and unstable manifolds of equilibria of ordinary differential equations this is a common difficulty that can be dealt with, for instance, by evolving the boundary of the computed subset so that at each step all points on the boundary have the same geodesic distance from the equilibrium [11]. For polytopes we could impose a similar constraint, e.g. that the geodesic distance of all vertices on the boundary of the subset varies by at most, say, 50%. Also it remains to modify the method to compute stable manifolds when (2.1) is non-invertible by computing all preimages of the subset at each step.

## Acknowledgements

The author thanks the organisers of ICDEA 2023 where many of the ideas were developed, and encouragement from Soumitro Banerjee. This work was supported by Marsden Fund contract MAU2209 managed by Royal Society Te Ap¯ arangi.

## A Proof of Lemma 4.1

Let v (1) , v (2) , . . . v ( K ) denote the vertices of U . For each k ∈ { 1 , 2 , . . . , K } let v ( k ) 0 = v ( k ) and let { v ( k ) i } ∞ i =1 be a sequence of preimages of v ( k ) satisfying (4.1). That is, for all i ≥ 1,

<!-- formula-not-decoded -->

where A ( i ) = A X -i mod p .

Choose any x ∈ U and form the convex combination x = ∑ K k =1 λ k v ( k ) . Let x 0 = x and for each i ≥ 1 define

Notice x i → γ because v ( k ) i → γ for each k . To complete the proof it remains to show that { x i } ∞ i =1 is a sequence of preimages of x , i.e. f ( x i ) = x i -1 for all i ≥ 1.

<!-- formula-not-decoded -->

Choose any i ≥ 1. By convexity, c T x i = ∑ K k =1 λ k c T v ( k ) i has same sign as each c T v ( k ) i . Thus as required.

## References

- [1] S. Banerjee and C. Grebogi. Border collision bifurcations in two-dimensional piecewise smooth maps. Phys. Rev. E , 59(4):4052-4061, 1999.
- [2] C. Bonatti, L.J. D´ ıaz, and M. Viana. Dynamics Beyond Uniform Hyperbolicity. Springer, New York, 2005.
- [3] A. Brøndsted. An Introduction to Convex Polytopes. Springer-Verlag, New York, 1983.
- [4] S. De, P.S. Dutta, S. Banerjee, and A.R. Roy. Local and global bifurcations in threedimensional, continuous, piecewise-smooth maps. Int. J. Bifurcation Chaos , 21(6):16171636, 2011.
- [5] M. di Bernardo. Normal forms of border collision in high dimensional non-smooth maps. In Proceedings IEEE ISCAS, Bangkok, Thailand , volume 3, pages 76-79, 2003.
- [6] P. Glendinning. Bifurcation from stable fixed point to 2D attractor in the border collision normal form. IMA J. Appl. Math. , 81(4):699-710, 2016.
- [7] P. Glendinning and D.J.W. Simpson. Chaos in the border-collision normal form: A computer-assisted proof using induced maps and invariant expanding cones. Appl. Math. Comput. , 434:127357, 2022.
- [8] P. Glendinning and D.J.W. Simpson. Unstable dimension variability and heterodimensional cycles in the border-collision normal form. Phys. Rev. E , 108(2):L022202, 2023.

<!-- formula-not-decoded -->

- [9] C. Grebogi, E. Ott, and J.A. Yorke. Crises, sudden changes in chaotic attractors, and transient chaos. Phys. D , 7:181-200, 1983.
- [10] B. Krauskopf, H. Osinga, E.J. Doedel, M.E. Henderson, J. Guckenheimer, A. Vladimirsky, M. Dellnitz, and O. Junge. A survey of methods for computing (un)stable manifolds of vector fields. Int. J. Bifurcation Chaos , 15(3):763-791, 2005.
- [11] B. Krauskopf and H.M. Osinga. Computing geodesic level sets on global (un)stable manifolds of vector fields. SIAM J. Appl. Dyn. Syst. , 2(4):546-569, 2003.
- [12] Yu.A. Kuznetsov. Elements of Bifurcation Theory. , volume 112 of Appl. Math. Sci. Springer-Verlag, New York, 3rd edition, 2004.
- [13] S.S. Muni and S. Banerjee. Bifurcations of mode-locked periodic orbits in threedimensional maps. Submitted to Chaos , 2023.
- [14] H.E. Nusse and J.A. Yorke. Border-collision bifurcations including 'period two to period three' for piecewise smooth systems. Phys. D , 57:39-57, 1992.
- [15] H.E. Nusse and J.A. Yorke. Border-collision bifurcations for piecewise-smooth onedimensional maps. Int. J. Bifurcation Chaos. , 5(1):189-207, 1995.
- [16] J. Palis and F. Takens. Hyperbolicity and Sensitive Chaotic Dynamics at Homoclinic Bifurcations. Cambridge University Press, New York, 1993.
- [17] M. Patra. Multiple attractor bifurcation in three-dimensional piecewise linear maps. Int. J. Bifurcation Chaos , 28(10):1830032, 2018.
- [18] F.P. Preparata and M.I. Shamos. Computational Geometry: An Introduction. SpringerVerlag, New York, 1985.
- [19] C. Robinson. Dynamical Systems. Stability, Symbolic Dynamics, and Chaos. CRC Press, Boca Raton, FL, 2nd edition, 1999.
- [20] D.J.W. Simpson. Border-collision bifurcations in R n . SIAM Rev. , 58(2):177-226, 2016.
- [21] D.J.W. Simpson. The structure of mode-locking regions of piecewise-linear continuous maps: I. Nearby mode-locking regions and shrinking points. Nonlinearity , 30(1):382-444, 2017.
- [22] I. Sushko, V. Avrutin, and L. Gardini. Bifurcation structure in the skew tent map and its application as a border collision normal form. J. Diff. Eq. Appl. , 22(8):1040-1087, 2016.