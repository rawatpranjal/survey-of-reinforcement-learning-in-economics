## On a new relation between entanglement and geometry from M(atrix) theory

Vatche Sahakian 1

Harvey Mudd College Physics Department, 241 Platt Blvd. Claremont CA 91711 USA

## Abstract

In the context of Matrix/light-cone gauge M-theory, we develop a new approach for computing quantum entanglement between a probe gravitating in the vicinity of a source mass and the source mass. We demonstrate that this entanglement is related to the gravitational potential energy between the two objects. We then show that the Von Neumann entropy is a function of two derivatives of the gravitational potential. We conjecture a relation between the entropy and the local Riemann tensor sampled by the probe, establishing a general scheme to relate entropy to local geometric data. This relation connects the rate of change, rotation, and twist of a small volume element at the probe's location to the quantum entanglement of the probe with the source.

1 sahakian@hmc.edu

## 1 Introduction and highlights

Various relations between quantum information and spacetime geometry seem to hint at the need for a fundamental rethinking of gravity. In this program, the general theme appears to be that gravity is an emergent phenomenon; and that underlying microscopic quantum degrees of freedom weave - through quantum entanglement - a fabric that we effectively perceive as space. In this note, we want to analyze these ideas in the context of Matrix theory, a non-perturbative formulation of string theory and quantum gravity [1]. We will consider a simple setup where a massive source pulls gravitationally on a probe; and where it is well-known that the effective quantum potential that arises from Matrix theory matches exactly with the expected gravitational potential that the probe experiences in light-cone gauge M-theory [2]. This effective potential arises from integrating out fast off-diagonal matrix modes that correspond to strings stretched between the two objects. In this work, we add the slower diagonal excitations and derive their quantum effective potential. We then demonstrate that the quantum vacuum of these modes is an entangled state in such a way that the entanglement entropy between source and probe is generally a function of derivatives of their gravitational potential. We compute the Von Neumann entropy and, based on the result we obtain, we conjecture a relation between the entropy and the local Riemann tensor sampled by the probe. Essentially, this entanglement entropy is shown to be directly related to local tidal forces. This connects the entropy to the rate of change, rotation, and twist of a small volume element at the location of the probe. The setup is reminiscent of entropy-area relations, except the statement we obtain is local.

In the first section, we describe the setup and outline the computation of the entanglement entropy. In the second section, we present a conjecture relating this entropy to local geometry. The Conclusion section discusses the more general implications of these results and future directions.

## 2 Quantum entanglement and gravity

Matrix theory is 0+1 dimensional U ( N ) Super Yang-Mills (SYM) theory that is purported to be dual to light-cone gauge M-theory. The rank of the gauge group N maps onto light-cone momentum in M-theory. Our starting point is the Matrix theory action in the background

field gauge 2

<!-- formula-not-decoded -->

All fields are in the adjoint of U ( N ), and the spinor fields Ψ α are 10 dimensional MajoranaWeyl. The last term in the first line is a gauge fixing term for the condition

<!-- formula-not-decoded -->

and G is a matrix of Faddeev-Popov ghosts. The Yang-Mills coupling is given by g 2 YM = 2 R where R is the radius of the M-theory light-cone circle. We work in string units, glyph[lscript] s = 1. We take the background as

<!-- formula-not-decoded -->

with all other fields vanishing. This is a block diagonal configuration with X i 1 being an N 1 × N 1 matrix, and X i 2 being an N 2 × N 2 matrix; we have N = N 1 + N 2 . In M-theory language, X i 1 is to represent an object that carries N 1 units of light-cone momentum - such as a spherical mass or a graviton; while X i 2 represents another object with N 2 units of light-cone momentum. We then want to write down an effective action by perturbing this background by

<!-- formula-not-decoded -->

The centers of mass of the two background objects are given by

<!-- formula-not-decoded -->

while the size of each object might naturally be represented by the second moments

<!-- formula-not-decoded -->

We assume that the two background objects are widely separated from each other so that their gravitational potential energy is small compared to their kinetic energies. We also

2 We will try to follow, as much as possible, the notation and conventions used in [2] and [3].

assume that their sizes are much smaller than the distance between them. In this regime, the off-diagonal perturbations in (4) are heavy or high frequency modes. One can then integrate them out and discovers that, for large N 1 , 2 and while setting all diagonal perturbations to zero, the resulting effective potential for the background variables X i 1 and X i 2 agrees with the Newtonian gravitational potential between the two objects in light-cone gauge M-theory [2]. This is a remarkable result in support of the Matrix theory-M theory correspondence.

Our task is to add to this computation the lighter, slower perturbations on the diagonal: the x 1 , 2 's, a 1 , 2 's, and ψ 1 , 2 's. We then want to write the effective potential for the x 1 , 2 and ψ 1 , 2 after the fast modes are integrated out. We write the effective potential, after integrating out the heavy off-diagonal modes, as

<!-- formula-not-decoded -->

where the first term S 0 comes from the part of the action that does not involve the offdiagonal perturbations and takes the form

<!-- formula-not-decoded -->

An important observation here is that there are no x 1 -x 2 couplings in S 0 ; hence, the coupling between the two objects, and thus any entanglement between them, can come only from S V . Furthermore, there are no ψ 1 -ψ 2 coupling terms in S 0 ; nor will there be any in S V : to leading order in small perturbations, given the action's quadratic form in the fermions, there is no entanglement to be considered between the fermionic diagonal modes.

The second piece of (7), S V , involves the off-diagonal perturbations that can be integrated out in the regime of interest. The computation of S V proceeds as in [2] where the diagonal perturbations were set to zero, except now X i 1 and X i 2 are now shifted by x i 1 and x i 2 ; we get from the ground state energy of the oscillators [2]

<!-- formula-not-decoded -->

where we define the 'mass matrices' along [2]: from the bosonic sector involving x and a , we have

<!-- formula-not-decoded -->

from the fermionic sector involving ψ , we have

<!-- formula-not-decoded -->

and from the ghost sector, we get

<!-- formula-not-decoded -->

In these expressions, we have defined the matrix

<!-- formula-not-decoded -->

In (9), Tr' corresponds to tracing over both group and Lorentz spaces. Throughout, we are assuming, as in [2], that the background satisfies the equations of motion, and hence all terms linear in the perturbations should be dropped. Hence, it is implicit in (9) that we drop linear terms in x i 1 and x i 2 once the expression is expanded further. Given the similarities between (9) and the result in [2], with the only modification coming from the shifts by x i 1 and x i 2 in (13), the computations proceed along similar steps: we write the square root of the matrices using a Dyson perturbation series in M 1 b and M 1 f , where M 1 b and M 1 f are smaller than M 0 b and M 0 f . The zeroth order corresponds to zero point energy and cancels by supersymmetry once we include the contribution from the ghosts ( M g only contributes to zeroth order); the cancellations carry over to linear, quadratic, and third order in M 1 . The first non-zero contribution arises at fourth order and we get

<!-- formula-not-decoded -->

where we defined

<!-- formula-not-decoded -->

Tr L involves tracing over Lorentz space, while Tr refers to tracing over group space as usual. Let us then write

<!-- formula-not-decoded -->

where we define

<!-- formula-not-decoded -->

so that all diagonal perturbations are in the ∆ K i matrix. To proceed further, we will focus onto a subsector of diagonal perturbations that perturb the location of the centers of masses of the two objects. We write

<!-- formula-not-decoded -->

where ε i 1 and ε i 2 are now the small perturbations associated with blocks 1 and 2 respectively. Beyond being a physically natural choice, these perturbations also decouple from other perturbations as they drop out of the commutators appearing in (1). This means that truncating to this sector of perturbations is mathematically consistent. The first part of the action given by S 0 in (8) then becomes

<!-- formula-not-decoded -->

We also have

We then get

<!-- formula-not-decoded -->

Assuming that the size of each object R 1 and R 2 is much smaller than the separation distance between them, the eigenvalues of K i 2 scale as r 2 where we define the relative position vector between the centers of mass of the two objects as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

More generally, we expect that

<!-- formula-not-decoded -->

where κ i is a matrix whose entries scale at most as R 2 1 and R 2 2 , the characteristic sizes of the two objects - independent of the distance r separating them. As long as R 1 , 2 glyph[lessmuch] r , we can then approximately write

<!-- formula-not-decoded -->

which is large, scaling as r 2 with large r . Looking back at (14), focus first on the exponential factor in the integrand. Whether for bosons or fermions, we have a structure of the form

<!-- formula-not-decoded -->

For large r , this implies that the predominant contribution to the integral in (14) comes from the region where the τ 's are zero. As a result, we can approximately write, as in [2],

<!-- formula-not-decoded -->

This leads to a very similar expression to the effective Newtonian potential computed in [2], now given by

<!-- formula-not-decoded -->

where we define

And we also have

<!-- formula-not-decoded -->

Notice that, given that the center of mass perturbations commute with all matrices, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These time derivatives of ε 1 , 2 are sub-leading to the kinetic terms of the perturbations arising in (19) as they will be multiplied by ∼ r -7 . The terms involving ∂ t ε 1 , 2 can then be dropped as long as the distance between the two objects is large. We then get

<!-- formula-not-decoded -->

Note next that the F ij and F 0 i are independent of r i , the separation vector between the two objects. To see this, we have from (23)

<!-- formula-not-decoded -->

where the matrix entries of κ i scale as the size of each object, independent of r i . As for F 0 i , we have from (23)

<!-- formula-not-decoded -->

demonstrating that F 0 i is also r i independent - but of course it depends on ∂ t r i . Putting things together, we can then write

<!-- formula-not-decoded -->

where a and b sum over 1 and 2, and where V is the potential from [2]

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Note that, as promised, we dropped terms linear in ε . In [2], it was shown that V matches precisely (including numerical coefficient) with the expected Newtonian gravitational potential averaged over the light-cone direction between the two objects as long as N 1 , 2 are large

<!-- formula-not-decoded -->

where p 1 and p 2 are the eleven dimensional momenta of the two objects.

Combining this result with the rest of the action from (19), we then have the effective action for ε 1 and ε 2 - that represent diagonal perturbations of the two objects

<!-- formula-not-decoded -->

where we used the fact that, in the regime of large distance r between the object, the potential V depends only on r i = x i 1 -x i 2 .

To proceed further, we need to setup a particular scenario where one of the two objects is treated as the heavy source and the other is a light probe. This sets the stage to interpreting the soon to be computed quantum entanglement as a measure of the local curved geometry experienced by the probe due to the source. Let us take object 1 to be the massive 'star' whose geometry object 2 is probing; for example, we might write

<!-- formula-not-decoded -->

where the J i are the angular momentum matrices, satisfying the SU (2) algebra and the Casimir relation

<!-- formula-not-decoded -->

where we assumed that N 1 glyph[greatermuch] 1. Similarly, we can take object 2 to be a spherical 'planet' with N 2 units of light-cone momentum that is much lighter and smaller. Each object has a nonzero size R 1 , 2 which is, at the least, the radius of the corresponding black hole. However, spatially localized configurations like the one given by (39) do not solve the equations of motion without an additional infrared cutoff i.e. we may not assume that the background is on shell as we have done so. If object 1 were to be a black hole, we expect that the chaotic nature of Matrix theory admits a metastable spherical configuration that is long-lived as it evaporates away slowly via Hawking radiation [4]. It has been shown that this stochastic short timescale dynamics can be effectively modeled by adding by hand a quadratic mass term to the action. Alternatively, one can imagine a background flux that stabilizes the configuration like in the case of the giant gravitons of the Berenstein-Maldacena-Nastase (BMN) Matrix model [5]. In either scenario, object 1 maintains a finite size due to some additional terms in the action, either due to effective stochastic physics or due to a non-flat background that essentially puts the system in a box. Here, we account for this by adding by hand a generic stabilizing term, the simplest of which would be

<!-- formula-not-decoded -->

where α 1 , 2 are positive constants that are tuned to assure a given stable sizes R 1 , 2 for objects 1 and 2 3 . The important general observation is that α 1 and α 2 must be positive to assure stability, and they are larger for larger objects. To see this, for the configuration given by (39), we can check that the size of object 1 is R 1 = r 1 , and its mass scales as M ∼ r 2 1 ∼ α 1 N 2 1 (the area of the spherical membrane). For fixed light-cone momentum N 1 , large α 1 corresponds to larger energy. Treating object 2 as the light probe, we henceforth assume that α 2 glyph[lessmuch] α 1 . In fact, as we shall see, it does not matter which one of the two objects is the lighter probe - the entanglement entropy of either one is the same as the other's, as expected from the fact that the combined system of diagonal perturbations is in a pure state.

The result of this is that one ends up adding an additional terms to the effective action (38) of the form -N 1 α 1 ( ε i 1 ) 2 and -N 2 α 2 ( ε i 2 ) 2 which dominate the corresponding ( ε i 1 ) 2 and ( ε i 2 ) 2 terms in (38). We then have the modified effective action

<!-- formula-not-decoded -->

We rescale the perturbations so as to canonically normalize the kinetic terms

<!-- formula-not-decoded -->

3 For example, it is easy to check that, for a spherical configuration of radius R 1 given by (39), one needs α 1 = 8 R 2 1 /N 2 1 .

We end up with the final effective action for the perturbations 4

<!-- formula-not-decoded -->

We write ∂ i = ∂/∂z i 2 , derivatives with respect to the probe's location. This is the effective action that describes the diagonal perturbations, to leading order R 1 /r and R 2 /r , between blocks 1 and 2 of the matrices - in a regime where object 2 is a light probe under the influence of a massive object 1 that curves the spacetime around it. We next compute the quantum entanglement in the vacuum of the z 1 -z 2 system arising from the z 1 z 2 coupling term in this effective action.

We have a system with two degrees of freedom with a Hamiltonian

<!-- formula-not-decoded -->

where a, b sum over 1 , 2. Following [6], we define the matrix ω as

<!-- formula-not-decoded -->

in 2 × 2 block diagonal form. The density matrix for the vacuum state takes the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have evaluated the square root of the matrix in the regime where (a) the offdiagonal entries of W are much smaller than the diagonal ones; and where (b) we have α 2 glyph[lessmuch] α 1 since object 2 is the probe.

We are interested in computing the entanglement entropy of object 2 with object 1 by tracing over the Hilbert space of object 1 and computing the Von Neumann entropy of the resulting reduced density matrix. Following [6], we then define

<!-- formula-not-decoded -->

4 One can also consider the probe to be a graviton. As a result, α 2 → 0 and we must keep the ε 2 2 term from (38). The subsequent computation is then slightly modified and the general pattern persists as long as the probe is much lighter than the source.

In our case, we get

where ω 11 is the sub-block of the matrix on the diagonal referring to object 1, ω 12 is the sub-block between objects 1 and 2, etc... Λ is then a 9 × 9 matrix in the tangent space of the probe's location - parameterized by z i 2 . For our case, we have

<!-- formula-not-decoded -->

We have defined γ to absorb all constants that refer to information about the individual objects, such as their sizes, masses, and equations of state. Note also that the eigenvalues of Λ are much smaller than one in the regime we have been working in.

The Von Neumann entropy of interest is then given by [6]

<!-- formula-not-decoded -->

where the simpler form on the second line is valid when the eigenvalues of Λ are much smaller than one, as is the case for us. Note than all U ( N ) matrix structure has disappeared and the relevant object lives in the tangent space of the probe's position - the vector space over which the expression traces. We have hence computed the entanglement entropy of the two objects in the quantum vacuum of perturbations of their centers of mass, and have shown how this entropy is a function of the gravitational potential that the probe experiences due to the presence of the source.

## 3 A new entropy-geometry relation

The gravitational potential V encodes information about the curvature of the spacetime at the probe's location. This means that we must be able to relate the entanglement entropy of the probe to local spacetime geometry. We start on the general relativity side with the light-cone gauge M-theory probe evolving along a timelike geodesic with tangent denoted by u µ , where µ = 0 , 1 , · · · , 10. Indices 1 , · · · , 9 are the transverse directions to the light-cone, mapping onto the Matrix theory target space indices; while the theory is boosted in lightcone direction x 10 . Let z µ i be nine spacelike vectors tangent to u µ , so we have i, j = 1 , · · · , 9. One can project on this sub-space using

<!-- formula-not-decoded -->

We can then relate the Newtonian gravitational potential V of the probe to the local Riemann tensor that it samples by [7]

<!-- formula-not-decoded -->

Looking back at (50), we see that the entropy is expressed as a function of the double derivatives of the potential, instead of covariant derivatives. This is natural in the context of Matrix theory as the Matrix theory formulation is background dependent, built up on top of a flat Minkowski background. This suggests that the probe coordinates on the Matrix theory side of the correspondence cannot map onto general coordinates that the dual Mtheory geometry might be written in. We then conjecture that one is required to interpret the Matrix theory coordinates as locally flat coordinates at the location of the probe on the M-theory side 5 . Matrix theory would then build up geometry locally through probe tidal acceleration that the Matrix effective potential can naturally determine. In locally flat coordinates at the location of the probe, the Christoffel symbols vanish and we have

<!-- formula-not-decoded -->

We also have at the location of the probe η µν z µ i u ν = 0 for i = 1 , · · · , 9. It is easy to check that we can write

<!-- formula-not-decoded -->

where we use the light-cone metric such that -2 u + u -+ ( u i ) 2 = -1. Note that the lightcone momentum p + = N 2 /R , and the light-cone energy is p -= ( m 2 +( p i ) 2 ) / (2 p + ). We also have ∂ -V = 0 from the fact that V is averaged over x -since no longitudinal momentum is exchanged between source and probe. We then can write

<!-- formula-not-decoded -->

defining the new quantity R ij built out of the local Riemann tensor, or equivalently tidal forces.

5 Note that (53) does not map onto the desired form involving simple derivatives at asymptotic infinity where curvatures are weak and where our computation is designed to hold. Hence, there is no alternative to locally flat coordinates, where the Christoffel symbols vanish. Note also that this is a more general coordinate system than Riemann normal or Fermi normal coordinates, and there is still infinite freedom globally in fixing locally flat coordinates. At the location of the probe, the freedom consists of local rotations SO (9), a subgroup of the gauge group of eleven dimensioanl gravity given the resriction to light-cone gauge. As required, this is also the symmetry group on the Matrix theory side.

Putting things together, we write

<!-- formula-not-decoded -->

This is a local relation between the curvature sampled by the probe and the quantum entanglement between the center of mass degrees of freedom of source and probe. Note that this entropy is finite, not surprisingly given that we are working in a UV complete theory of quantum gravity. In the limit where the curvature vanishes, so that this expression for entropy. Next, we consider the expression

<!-- formula-not-decoded -->

which is a measure of deformations of the shape and orientation of a small sphere at the probe along its trajectory. We then have a version of Raychaudhuri's equation

<!-- formula-not-decoded -->

where we have dropped higher order terms that are smaller than the leading contribution at weak curvatures. Using locally flat coordinates, and projecting onto the nine dimension subspace using the z µ i 's, we have

<!-- formula-not-decoded -->

In particular dθ/dτ , where θ is the trace of θ ij using the metric h ij , is the rate of change of a volume element along the probe's geodesic. Hence, equation (57) establishes a relation between the source-probe entanglement entropy and the rate at which a small volume of space shrinks, rotates, and twists along the geodesic of the probe. If we were to choose the probe to be massless, one can easily show that one obtains a similar relation but now involving an area transverse to a congruence of null geodesics associated with the probe. All this is somewhat reminiscent of the entropy-area relations we encounter in other settings [8, 9] with one significant difference being that our relation is a local statement.

## 4 Conclusion

We have demonstrated that Von Neumann entanglement entropy between two blocks of matrices in Matrix theory - that represent a probe gravitating near a source - can quite generically be written as a function of derivatives of their mutual gravitational potential.

We also presented arguments and a conjecture for expressing this relation as a map between entanglement entropy and local spacetime geometry as sampled by the probe in the background of the source.

We considered a particular scenario and worked consistently only to leading order in weak gravitational potential energy. Yet, the analysis introduces a new general way to develop maps between quantum information and spacetime geometry in Matrix theory. This involves looking at diagonal matrix fluctuations and focusing on the ground state density matrix of these degrees of freedom. As a result, when one focuses on a sub-block of a matrix, the resulting reduced density matrix and quantum entanglement will be related to the effective Matrix potential between the two matrix sub-blocks arising from integrating out fast offdiagonal modes. This mechanism appears general and might hint to why, at least to leading order in weak gravity, one expects a relation between entanglement entropy and spacetime geometry.

Entanglement entropy by nature is multi-faceted. It depends on how one slices parts of a larger system, and on what quantum state the entire system lives in. These freedoms are very much reflected in the analysis, where we made a series of choices to set a computationally accessible setup. There are many more settings to explore, and a catalogue of case studies can help develop intuition on the general pattern of expected relations between entropy and local geometry. We end by pointing out a couple of particularly interesting cases: the case involving massless probes, where one has the promise to connect with ideas from holography and entropy of light-sheets developed from different perspectives [10, 11, 12]; and the case where the approach is used in BMN theory that admits stable giant gravitons and hence the need to add stablizing terms to the action is avoided [13].

## 5 Acknowledgments

This work was supported by NSF grant number PHY-0968726.

## References

- [1] T. Banks, W. Fischler, S. H. Shenker, and L. Susskind, 'M theory as a matrix model: A conjecture,' Phys. Rev. D55 (1997) 5112-5128, hep-th/9610043 .
- [2] D. Kabat and I. Taylor, Washington, 'Spherical membranes in matrix theory,' Adv. Theor. Math. Phys. 2 (1998) 181-206, hep-th/9711078 .
- [3] K. Sugiyama and K. Yoshida, 'Giant graviton and quantum stability in matrix model on pp wave background,' Phys. Rev. D66 (2002) 085022, hep-th/0207190 .
- [4] H. Du and V. Sahakian, 'Emergent geometry from stochastic dynamics, or Hawking evaporation in M(atrix) theory,' 1812.05020 .
- [5] D. E. Berenstein, J. M. Maldacena, and H. S. Nastase, 'Strings in flat space and pp waves from N=4 superYang-Mills,' JHEP 0204 (2002) 013, hep-th/0202021 .
- [6] H. Casini and M. Huerta, 'Entanglement entropy in free quantum field theory,' J. Phys. A42 (2009) 504007, 0905.2562 .
- [7] S. W. Hawking and G. F. R. Ellis, The Large Scale Structure of Space-Time . Cambridge Monographs on Mathematical Physics. Cambridge University Press, 2011.
- [8] S. Ryu and T. Takayanagi, 'Holographic derivation of entanglement entropy from ads/cft,' Phys. Rev. Lett. 96 (2006) 181602, hep-th/0603001 .
- [9] T. Nishioka, S. Ryu, and T. Takayanagi, 'Holographic Entanglement Entropy: An Overview,' J. Phys. A42 (2009) 504008, 0905.0932 .
- [10] R. Bousso, 'A covariant entropy conjecture,' JHEP 07 (1999) 004, hep-th/9905177 .
- [11] V. Sahakian, 'Holography, a covariant c-function and the geometry of the renormalization group,' Phys. Rev. D62 (2000) 126011, hep-th/9910099 .
- [12] V. Sahakian, work in progress.
- [13] A. Busis, V. Sahakian, to appear.