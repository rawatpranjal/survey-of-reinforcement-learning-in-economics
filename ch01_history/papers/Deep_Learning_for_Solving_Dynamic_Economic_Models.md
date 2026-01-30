## A note on electric-magnetic duality and soft charges

Marc Henneaux, a,b, 1 Cédric Troessaert c

a

Université Libre de Bruxelles and International Solvay Institutes, Physique Mathématique des Interactions Fondamentales,

Campus Plaine-CP 231, Bruxelles B-1050, Belgium

b Collège de France, 11 place Marcelin Berthelot, 75005 Paris, France

c Haute-Ecole Robert Schuman, Rue Fontaine aux Mûres, 13b, B-6800, Belgium

E-mail: henneaux@ulb.ac.be , cedric.troessaert@hers.be

Abstract: We derive the asymptotic symmetries of the manifestly duality invariant formulation of electromagnetism in Minkoswki space. We show that the action is invariant under two algebras of angle-dependent u (1) transformations, one electric and the other magnetic. As in the standard electric formulation, Lorentz invariance requires the addition of additional boundary degrees of freedom at infinity, found here to be of both electric and magnetic types. A notable feature of this duality symmetric formulation, which we comment upon, is that the on-shell values of the zero modes of the gauge generators are equal to only half of the electric and magnetic fluxes (the other half is brought in by Diracstring type contributions). Another notable feature is the absence of central extension in the angle-dependent u (1) 2 -algebra.

1 ORCID: 0000-0002-3558-9025

## Contents

|   1 | Introduction                                              |   1 |
|-----|-----------------------------------------------------------|-----|
| 2   | Starting point                                            |   3 |
| 2.1 | Action and presymplectic form                             |   3 |
| 2.2 | Hamiltonian vector fields                                 |   4 |
| 2.3 | Boundary conditions                                       |   4 |
| 2.4 | Improper gauge symmetries                                 |   5 |
| 2.5 | Equations of motion                                       |   7 |
| 3   | Poincaré invariance                                       |   8 |
| 3.1 | Boundary degrees of freedom                               |   8 |
| 3.2 | More improper gauge transformations                       |   9 |
| 3.3 | Poincaré transformations                                  |  10 |
| 3.4 | Poincaré transformations of the improper gauge generators |  12 |
| 3.5 | SO (2) duality generator                                  |  12 |
| 4   | Sources                                                   |  13 |
| 5   | Conclusions                                               |  13 |

## 1 Introduction

The asymptotic structure of electromagnetism in Minkowski space has been a subject of great interest in the last years, with the discovery that soft photon theorems could be viewed as Ward identities of the corresponding asymptotic symmetries [1], triggering a lot of insightful activity [2-6] reviewed in [7]. (Earlier work on the asymptotic symmetries of electromagnetism at null infinity involves [8, 9].)

While this work was originally focused on null infinity, the structure of the asymptotic symmetry algebra, which is given by arbitrary functions on the 2 -sphere ('angle-dependent u (1) transformations') was also explored at spatial infinity [10-12] and equivalence between the two formulations demonstrated. In particular the antipodal matching conditions of the null infinity approaches, relating fields at the past of I + to fields at the future of I -could be justified on a dynamical basis [12]. The proof of equivalence involves an interesting change of basis in the algebra based on a parity decomposition.

The above formulations are 'purely electric' and exhibit only one angle-dependent u (1) symmetry. As shown in [13], there exists a second angle-dependent u (1) symmetry. It corresponds to 'large' gauge transformations acting on the dual potential and can be exibited in the magnetic formulation.

There exists a formulation of electromagnetism in which electric-magnetic duality, which is always a symmetry of the action [14], is manifest. This formulation is first-order and involves two vector potentials, which are not only duality-conjugate, but also canonically conjugate. Duality-symmetry is then a bona fide Noether symmetry of standard type [14]. This symmetry extends to a sp ( n ) symmetry of the action - and not just of the field equations - when scalar field couplings of appropriate form are included [15]. Although the formulation of [14] involves two vector potentials, we stress that it is equivalent to the standard one-potential formulation of Maxwell theory. The equations of motion for the two vector potentials are of first order, so that the amount of physical, free initial data is unchanged. In particular, there is only one photon, and only two physical degrees of freedom per space point.

In the manifestly duality-invariant formulation, each vector potential enjoys a separate u (1) gauge symmetry. The gauge transformations can be either 'proper' or 'improper' [16], depending on their behaviour at infinity. The purpose of this note is to show that the two angle-dependent u (1) symmetries separately displayed in either the pure electric or the pure magnetic formulations, are actually both simultaneously present as standard improper gauge symmetries in the duality-invariant formulation. None of these groups of transformations needs a special treatment and both follow from the application of standard rules. Their generators, computed through canonical methods, are given by non-vanishing surface integrals, which can easily be written down in terms of the variables of the dualitysymmetric formulation.

As explained in [17], the coupling to sources involves both minimal coupling terms and Dirac-type coupling terms with Dirac strings [18]. Each type of couplings contributes half of the total coupling in the duality-symmetric formulation. It follows that the generators of improper gauge transformations, associated with minimal couplings, only gives half of the total electric or magnetic fluxes. This factor of one-half is not the result of an incorrect symplectic structure but is built in the construction. Also built in the approach of [17] is the fact that the electric and magnetic charges are non dynamical c -numbers. They have therefore zero Poisson bracket with any quantity. This precludes the appearance of a central charge in the algebra of the angle-dependent u (1) generators, found in other, different treatments [19, 20].

Our paper is organized as follows. In Section 2, we briefly recall the duality invariant formulation of [14, 17] (see also [21, 22]) and discuss the specific features due to the degeneracy of the pre-symplectic form that follows from the action. We give boundary conditions on the vector potentials, which involve parity conditions with a twist given by a gradient, extending the work of [12] to the double-potential formulation. We then derive (some of) the improper gauge transformations. Section 3 turns to Poincaré invariance. We introduce surface degrees of freedom at infinity along the lines of our previous treatment [12], which are necessary to make the boosts preserve the pre-symplectic structure. These surface degrees of freedom, somewhat reminiscent of those introduced in [23], bring in their own improper gauge transformations, which are written. All the improper gauge transformations combine to form an angle-dependent u (1) 2 symmetry, which is the same as the one found in null infinity analyses. The Poisson bracket algebra of all the improper gauge

transformations is shown not to involve a central extension. The so (2) duality rotation is also worked out and contains, in addition to the Chern-Simons term found in [14], an extra contribution involving the new surface degrees of freedom. Section 4 briefly comments on the introduction of sources and the need to introduce parity-symmetric Dirac strings, as in [24]. Finally, we close our paper with comments on the dressings of physical states (Section 5).

## 2 Starting point

## 2.1 Action and presymplectic form

We start with the first-order manifestly duality invariant action [14, 17],

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with 1

Here, ( A a i ) ≡ ( A i , Z i ) and annihilate Ω ,

<!-- formula-not-decoded -->

are the 'magnetic fields' ( B i 1 is actually the standard electric field, while B i 2 is the standard magnetic field [14, 17] - up to signs that depend on conventions).

The presymplectic 2 -form following from the action,

<!-- formula-not-decoded -->

does not take the standard ' d V p ∧ d V q ' canonical form. How to deal with such 2 -forms is well known and recalled, for instance, in appendix A of [12], of which we take the notations.

The notable feature of Ω is that it is degenerate (hence the 'pre' in 'presymplectic'). Indeed, vector fields of the following form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These correspond precisely to proper gauge transformations (see Subsection 2.4 below). These are the only vector field with this property 2 : setting X = ( α a ) , one finds

i

<!-- formula-not-decoded -->

1 Latin indices from the beginning of the alphabet ( a, b, . . . ) are 'internal' indices taking the values 1 and 2 . The internal Levi-Civita epsilon tensor is glyph[epsilon1] ab , with glyph[epsilon1] 12 = 1 . Latin indices from the middle of the alphabet ( i, j, k, . . . ) run from 1 to 3 ( glyph[epsilon1] 123 = 1 ).

2 In order to prove this statement, we assume that the potentials A a i satisfy the asymptotic behaviour given in Subsection 2.3 below.

One must solve the equation i X Ω = 0 for the α a i 's. Now, the d V A b k are independent 1 -forms, and so, the bulk term vanishes if and only if the coefficient ∂ [ i α a j ] vanishes. This implies α a i = ∂ i α a . In order to abide by the asymptotic behaviour of A a i , the function α a should tend to a O (1) -function that can depend on the angles as r → ∞ , i.e., α a = α a ( θ, ϕ ) + O ( r -1 ) for some function α a on the 2 -sphere. The surface term can then be rewritten as -(1 / 2) GLYPH&lt;11&gt; S ∞ d 2 S i glyph[epsilon1] ab glyph[epsilon1] ijk d V ( ∂ j A a k ) α b , which vanishes only if the α b 's tend to a constant at infinity, i.e., if the functions α a on the 2 -sphere reduce to their 0 -th spherical harmonic, α a ( θ, ϕ ) = α a 0 .

The zero vector fields of Ω (i.e., the phase space vector fields that annihilate Ω , i X Ω = 0 ) are thus precisely the vector fields generating proper gauge transformations (Subsection 2.4).

## 2.2 Hamiltonian vector fields

One says that a phase space vector field X (representing an infinitesimal transformation through δz α = X α , where z α are the phase space variables) is Hamiltonian (and that the transformation is canonical) if there exists a phase space function F such that

<!-- formula-not-decoded -->

The function F is called the generator of the transformation. This is equivalent to the condition L X Ω = 0 , i.e., the transformation generated by X preserves the presymplectic form (we assume trivial topology, more precisely, that every closed 2 -form is exact).

Given a Hamiltonian vector field, the function F is, as usual, determined up to a constant. Because Ω is degenerate, however, two new features arise. First, a function F can be associated with a Hamiltonian vector field only if i X a d V F = L X a F = 0 for the zero vector fields X a of Ω , i.e., if it is invariant under the flow generated by X a . In our case, this means that F has to be gauge invariant (under proper gauge transformations). Second, given a function F fulfilling this condition, the corresponding X is determined up to a combination of the X a 's. In our case, this means that the transformation generated by F is determined up to a proper gauge transformation. In particular, the zero phase space function ( F ≡ 0 ) is associated with the Hamiltonian vector fields defining the proper gauge transformations.

These properties are physically sensible. It turns out sometimes, however, to be more convenient to deal with a true symplectic form, i.e., an invertible Ω . There are different ways to get one. One way, which preserves spacetime locality, enlarges the phase space and is described below.

## 2.3 Boundary conditions

Before proceeding further, we shall specify the boundary conditions on the fields. These are adapted from [12, 25] and read:

<!-- formula-not-decoded -->

where n is the unit vector to the radial spheres and stands therefore for coordinates on the unit sphere. Instead of standard polar coordinates ( θ, ϕ ) which behave as θ → π -θ

and ϕ → ϕ + π under the antipodal map, we shall find it convenient to use coordinates x A which transform instead as x A →-x A . (Of course, neither ( θ, φ ) nor ( x A ) provide a single global chart.)

We impose in addition the following 'twisted parity condition' on the leading term of the vector potentials,

<!-- formula-not-decoded -->

for some λ a ( n ) that may assumed to be even. So, the leading terms are even up to a gradient. Differently put, the even part of A a i is unrestricted, but the odd part must be a gradient to leading order.

This implies that the fields are odd to leading orders,

<!-- formula-not-decoded -->

In spherical coordinates, the twisted parity conditions read

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

With these boundary and parity conditions the presymplectic term in the action is finite. The logarithmic divergence, potentially present without parity conditions, is actually absent even if one allows a gradient in A a i because the coefficient of ˙ A a i in the kinetic term is identically transverse.

## 2.4 Improper gauge symmetries

The boundary conditions are invariant under gauge transformations

<!-- formula-not-decoded -->

glyph[negationslash]

where there is no parity condition on ε a . When ε a ( x A ) = 0 , the gauge transformation is generically improper and defines a non trivial symmetry [16]. More precisely, the generator G [ ε a ] , found through the equation i X Ω = -d V G , is given by

<!-- formula-not-decoded -->

as the above computation indicates. Since both d 2 S i = n i d 2 S and B i a are odd, only the even part of the gauge parameters ε a contributes to the charges,

<!-- formula-not-decoded -->

Furthermore, because we consider source-free electromagnetism, the electromagnetic fluxes corresponding to constant ε a 's in (2.15) are zero as can be seen by converting the surface integral to a volume integral through Gauss formula. This is in agreement with the observation we made in section 2.1 that gauge parameters ε a that tend to a constant at infinity generate proper gauge transformations. The other angle-dependent u (1) transformations have generically non-vanishing charges, however.

The improper gauge symmetries are abelian and form the infinite dimensional algebra u (1) even ⊕ u (1) even . By the notation u (1) even we mean the infinite-dimensional algebra of gauge transformations parametrized by arbitrary even functions on the 2 -sphere (and with no zero mode). We shall discuss below how the odd functions on the sphere enlarge further the algebra.

It is worth stressing that the generator of improper gauge symmetries is a pure surface term, without bulk piece. This is because, with the degenerate presymplectic structure, the proper gauge transformations (which can be viewed as the bulk part of the improper ones) have through the equation i X Ω = -d V G a 'generator' G that identically vanishes. The surface term by itself is a well defined generator.

Another interesting feature that is exhibited by formula (2.15) is that both the electric and magnetic fluxes come with a factor 1 / 2 . In hindsight this was to be expected if one recalls that in the manifestly duality invariant formulation, both electric and magnetic sources are minimally coupled to their respective electromagnetic potentials, but only with half of the strength of their charges (see [17] and Section 4 below). Gauss law for the electromagnetic fields yield therefore half of the fluxes. Electric and magnetic sources have also Dirac string type couplings, again with half of the strength of the charges. The Dirac strings carry the other half of the fluxes.

Finally, we note that the presymplectic form (2.4) evidently belongs to a family of presymplectic forms that differ by boundary terms and yield the same equations of motion,

<!-- formula-not-decoded -->

with σ ∈ [ -1 , 1] . The value σ = 0 , considered in [17], reproduces the duality invariant presymplectic form considered here, while the extreme values σ = -1 and σ = 1 correspond to the electric or magnetic formulations, respectively (e.g., σ = -1 yields Ω -1 = -GLYPH&lt;1&gt; d 3 xglyph[epsilon1] ijk ∂ i d V A 2 j ∧ d V A 1 k which is the standard presymplectic form GLYPH&lt;1&gt; d 3 xd V π k ∧ d V A k of the usual electric formulation with the electric potential A k ≡ A 1 k , if one recalls that the electric field E k = glyph[epsilon1] ijk ∂ i Z j is minus the momentum conjugate to A k ( A 2 k ≡ Z k in the notations of [14])).

Even though the difference between the presymplectic structures Ω σ is a mere surface term, these are physically inequivalent because the surface term in question does not vanish. This leads to two important distinct features. First the forms of the electric and magnetic couplings are different. Both couplings are allowed, but they must be included differently dependong on the value of σ . An electric source minimally couples with strength 1 2 (1 -σ ) to the electric vector potential, the remaining of the coupling ( 1 2 (1 + σ ) ) being accounted for by Dirac string type terms [17]. Similarly, a magnetic source minimally couples with

strength 1 2 (1+ σ ) to the magnetic vector potential, the remaining of the coupling ( 1 2 (1 -σ ) ) being accounted for by Dirac string type terms. The symmetric case ( σ = 0 ) has both types of sources and of couplings on the same footing.

Second, the respective weights of the physically relevant improper gauge transformations are different. For all values of σ ∈ ( -1 , 1) , the improper gauge symmetries form the algebra u (1) even ⊕ u (1) even , and the generators take the above form but with weights 1 2 (1 + σ ) and 1 2 (1 -σ ) , respectively. In the limiting cases σ = ± 1 , one of the two u (1) even 's becomes proper because the corresponding generators vanish for all configurations. The improper gauge transformations reduce to a single algebra of angle-dependent (even) u (1) transformations. So, for σ = -1 , the magnetic gauge transformations are all proper. In that case, the electric flux is entirely carried by the electric field (there is no Dirac string for electric sources) while the magnetic flux of magnetic monopoles is entirely carried by the Dirac string. Conversely, for σ = +1 , the electric gauge transformations are all proper and the coupling of electric sources is entirely of Dirac string type.

## 2.5 Equations of motion

Stationarity of the action (2.1), δS = 0 , implies

<!-- formula-not-decoded -->

where we have dropped terms at the initial and final time boundaries and where we have used the boundary conditions as r →∞ to infer that B c m δA a j ∼ O (1 /r 3 ) decays too fast to contribute to the surface integral at spatial infinity. Dropping boundary terms at t = t i and t = t f is legitimate provided one includes the appropriate surface terms there, along the lines discussed for instance in [27]. We will not dwelve on this well understood issue here, since we want to focus on the difficulties raised by the behaviour of the fields at spatial infinity.

The vanishing of the bulk term in δS implies

<!-- formula-not-decoded -->

for some arbitrary functions A b 0 which are only restricted at this stage to be such that ∂ k A 0 is of order O ( 1 r ) , in order to preserve in time the asymptotic behaviour of A b k .

The vanishing of the surface term in δS puts constraints on the leading term of the coefficient glyph[epsilon1] ijk n i ˙ A b k ∼ glyph[epsilon1] ijk n i ∂ k A b 0 of δA a j in the surface integral. Indeed, since the leading even part of the variation δA a j is arbitrary, the leading even part of its coefficient should be equal to zero, or equivalently, the leading odd part of ∂ k A b 0 (coming from the leading even part of A 0 ) ) must vanish. This implies that the ambiguity in the time evolution, captured by the A 0 -term in (2.18), is a a proper gauge transformation , as it should. There is no improper gauge transformation involved in the time evolution ambiguity once the Hamiltonian is given. Without loss of generality, one may asymptotically fix the gauge and assume that there is no O (1) -piece in A 0 ,

<!-- formula-not-decoded -->

It is easy to verify that there is no additional constraint coming from the leading odd part of the variation δA a j so that the equations of motion are completely equivalent to (2.18) and (2.19) (under the above partial gauge fixing).

An interesting consequence of the equations of motion is

<!-- formula-not-decoded -->

as one can see by expanding (2.18) with k = r in powers of r -1 .

## 3 Poincaré invariance

## 3.1 Boundary degrees of freedom

In order to implement Poincaré invariance, one needs to introduce a surface degree of freedom at infinity, Ψ , which is conjugate to the gauge-invariant asymptotic value A r of the radial component of the electric vector potential. This was explained in [12], where it was also shown that this new degree of freedom can be interpreted as the O (1 /r ) -term in the asymptotic expansion of the temporal component A 0 . In the duality-symmetric formulation, one needs a so (2) electric-magnetic doublet Ψ a .

Following [12], we thus add to the action the term

<!-- formula-not-decoded -->

leading to

<!-- formula-not-decoded -->

The new symplectic structure is thus

<!-- formula-not-decoded -->

The factor of one half present in the new boundary term matches the factor of one half in the symplectic form Ω , which is itself a consequence of the boundary term difference between the usual 'electric' symplectic structure and the duality invariant one that we have pointed out.

Because A a r is odd, only the odd part of Ψ a appears in the action. The even part of Ψ a is pure gauge. One may either fix that gauge and assume e.g. that it is zero, so that Ψ a ( -x A ) = -Ψ a ( x A ) , or one can chose not to fix that gauge and keep the even part of Ψ a arbitrary. It turns out that this second approach is more convenient. Thus we have

<!-- formula-not-decoded -->

where the odd part of Ψ a corresponds to physical degrees of freedom. Given the extra degeneracy of the extended symplectic form, a Hamiltonian function must be invariant under the corresponding gauge symmetry, i.e., be independent of ( Ψ a ) even .

The new degree of freedom and the new term in the action bring in additional equations, which are

<!-- formula-not-decoded -->

The first one follows from extremization with respect to Ψ a and is a consequence of the other equations of motion. The second one follows from extremization with respect to A a r .

## 3.2 More improper gauge transformations

The action is also invariant under arbitrary (time-independent) shifts of Ψ a by an odd function,

<!-- formula-not-decoded -->

These are canonical transformations with generators

<!-- formula-not-decoded -->

These transformations are improper gauge transformations since their generators generically do not vanish.

The total set of soft charges is thus given by

<!-- formula-not-decoded -->

with ε a = ε a even and µ a = µ a odd . The generators reduce to surface integrals at infinity, without bulk term, because Gauss' law has been solved for and so is identically satisfied. As we explained, this does not prevent the surface integrals to be well defined canonical generators. In the pure electric formulation where Gauss' law is not solved for, it is natural not only to extend the boundary degree of freedom Ψ a in the bulk (as we shall actually do below), but also to introduce its bulk conjugate momentum, which is constrained to vanish, preserving a symmetric treatment of the two gauge symmetries. There is no motivation for performing this second step here, since there is no bulk constraint associated with the standard gauge invariance to begin with.

The improper gauge transformations commute and the algebra of their charges does not acquire a central extension. This is is because the generators are invariant under both proper and improper gauge symmetries. It must be contrasted with the approach of [19, 20]. Note that the zero modes of the generators identically vanish (no source) and so a central charge extension mixing electric and magnetic generators of the form

<!-- formula-not-decoded -->

(say, or of similar undifferentiated type), is not possible since the right-hand side must identically vanish when ε a = constant.

## 3.3 Poincaré transformations

One can now easily verify Poincaré invariance. The steps are the same as in the electric formulation [12]. We denote the normal and tangential components of the Poincaré transformations by ξ and ξ i , respectively. One has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and T = constant, as well as ξ i = b i j x j + W i with b ij = -b ji and W i = constant. Here, D A is the covariant derivative on the unit sphere with standard round metric γ AB . In spherical coordinates, the tangential components read

<!-- formula-not-decoded -->

where Y A and W are functions on the sphere such that

<!-- formula-not-decoded -->

(the Y A 's define Killing vectors on the sphere).

For normal deformations we assume that the fields transform as

<!-- formula-not-decoded -->

with

(with B a i ≡ B j b δ ab δ ij ) and

<!-- formula-not-decoded -->

The function Ψ a appearing in (3.13) is an extension inside the bulk of the boundary variable Ψ a , in the following sense,

<!-- formula-not-decoded -->

glyph[negationslash]

This matching condition is the only condition on Ψ a , since two different extensions will then differ by a proper gauge transformation. Note that for non zero boosts ( b = 0 ), the linear growth of b compensates the 1 r decay of Ψ a so that the term ξ Ψ a tends to the O (1) even function b Ψ a at infinity and induces therefore a nontrivial improper gauge transformation in δ ξ, 0 A a i .

Similarly, the transformation (3.14) involves both an improper gauge part, namely, δ ξ, 0 ( Ψ a ) odd and a proper gauge one, namely δ ξ, 0 ( Ψ a ) even . That second one is completely arbitrary. The gauge choice made in (3.14) is convenient as it can be related to the Lorentz gauge (see below).

For convenience we also provide the leading contribution of (3.13) in spherical coordinates

<!-- formula-not-decoded -->

with √ γe AB ≡ glyph[epsilon1] rAB .

One can motivate the form of δ ξ, 0 A a i as follows. If we want to view ( A a µ ) as the components of 4 -vectors, their Poincaré transformations should coincide with their Lie derivatives on-shell and up to gauge transformations. Now, with ( ξ µ ) = ( ξ, 0) ,

<!-- formula-not-decoded -->

upon use of the equations of motion. Thus, if we fix further the freedom in A a 0 such that A a 0 = Ψ a , which is permissible and in fact already considered in our earlier work [12, 25], one finds

<!-- formula-not-decoded -->

which is precisely (3.13) (recall that g = 1 in Cartesian coordinates).

When the condition A a 0 = Ψ a holds, the Lorenz gauge ∂ µ A µ = 0 holds asymptotically, since ∂ µ A µ = ∂ 0 A 0 + O (1 /r 2 ) = ∂ 0 Ψ a r + O (1 /r 2 ) = O (1 /r 2 ) on account of ˙ Ψ a = 0 . This provides a motivation for the transformation rule for Ψ a - in addition to the fact that the complete transformation must be canonically generated. Indeed, when the Lorentz gauge holds, the transformation of A a 0 is L ξ A a 0 = ξ µ ∂ µ A a 0 + ∂ 0 ξ µ A a µ . Now, with ξ µ ∂ ∂x µ = x k ∂ ∂x 0 + x 0 ∂ ∂x k (boost along the k -th direction), one has ( ξ µ ) = ( x k , 0) and ( ∂ 0 ξ µ ) = (0 , δ m k ) on the slice x 0 = 0 (translating the slice in time will only generate additional translation terms that affect only the subleading terms of the fields), and so L ξ A a 0 = x k ∂ 0 A a 0 + A a k = x k ∂ m A a m + A a k = ∂ m ( x k A a m ) , i.e.,

<!-- formula-not-decoded -->

with ξ ≡ x k ≡ br with b the function of the angles corresponding to x k . Expanding this relation in polar coordinates and keeping the leading term, one gets exactly (3.14).

The associated generator, which is invariant under proper gauge transformations and is therefore indeed an acceptable canonical generator, is given by

<!-- formula-not-decoded -->

Note that only the odd part of Ψ a appears in this expression as requested by gauge invariance, since √ γglyph[epsilon1] ab e AB ∂ A A b B = lim r →∞ B r a is even.

Because the (pre-)symplectic form is degenerate, the transformation generated by P full ξ, 0 is defined up to proper gauge transformations, which can be chosen as one pleases. By contrast the improper gauge transformations entering the transformations is not arbitrary. In particular, the asymptotic term ∂ A ( b Ψ a ) in the transformation of A a A is an improper gauge transformation which is determined by the generator. Two different bulk extensions of that improper gauge transformation differ by a proper gauge transformation. This the ambiguity in the bulk field Ψ a .

For the spatial component ξ i we assume that Ψ a transforms as a scalar under rotations. We can then write the full generator for the spatial translations and rotations as

<!-- formula-not-decoded -->

which is again easily seen to be invariant under proper gauge transformations.

With these generators at hand one can show that the system is indeed invariant under Poincaré transformations and fulfill the Poincaré algebra.

## 3.4 Poincaré transformations of the improper gauge generators

The generators of the Poincaré and gauge transformations span a semidirect sum, with the gauge transformations being an abelian ideal. The action of the Lorentz algebra controlling the semi-direct sum is given by

<!-- formula-not-decoded -->

At this point we have two pairs of functions of definite parity, the odd µ a and the even glyph[epsilon1] a . In order to compare our results with the null infinity analysis, it remains to argue that these functions combine to form functions on the sphere with no definite parity. To that end, one can adapt [28] or Appendix C in [12], where this was explicitly shown for the 'common' formulation of electrodynamics and which generalizes to the case at hand.

## 3.5 SO (2) duality generator

One key feature of the double potential formulation of electrodynamics is the manifest SO (2) duality invariance that acts locally on the canonical variables [14]. This duality transformation also rotates the asymptotic fields Ψ a . Explicitly, the rotation

<!-- formula-not-decoded -->

leaves the action invariant. Through the formula i X Ω = -d V Q , one finds that it is generated by

<!-- formula-not-decoded -->

The first term is the Chern-Simons term of [14]. The second appears beause of the extra surface degrees of freedom Ψ a .

Without the parity conditions of Section 2.3 this expression would be logarithmically divergent. However, since we impose that the leading term in A a i is even up to a gradient (which contributes a manifestly finite surface term since the magnetic field is identically transverse), the coefficient of the logarithmically divergent term is equal to zero and we get a well defined symmetry generator.

The algebra of the so (2) -duality generator with the generators of the improper gauge transformations can easily be worked out,

<!-- formula-not-decoded -->

## 4 Sources

The inclusion of sources follows the pattern of [17]. In the duality-symmetric formulation, half of the coupling is of standard minimal type, while the other half follows the Dirac procedure and involves Dirac strings. To comply with the asymptotic parity conditions, these strings will be chosen symmetrically in the asymptotic region, i.e., half of the flux carried by the string (a quarter of the total flux for that matter) will be brought from one direction, while the other half (quarter) will be brought from the antipodal direction (asymptotically). This is the same set up as for gravity [24].

The detailed form of the action with sources included is given in [17]. What replaces the identity

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where A runs over the various sources, located at glyph[vector] z ( A ) . It follows that

<!-- formula-not-decoded -->

is a 'c-number', having zero bracket with anything.

The action is invariant under the same set of gauge transformations δA a i = ∂ i ε a , which must be supplemented by the phase transformation δp ( A ) i = -q ( A ) a ∂ε a ∂z i ( A ) for the momenta conjugate to the position of the charged particles. The generators of the gauge transformations take the same form as before, with improper gauge transformations characterized by non-vanishing surface integrals at infinity. Because the zero mode of the magnetic field is a c -number, there is again no central charge in the abelian algebra of the improper gauge generators, which is unchanged.

In addition to the gauge transformations, the theory is also invariant under arbitrary shifts of the Dirac strings (provided they remain attached to the sources). Non vanishing displacement of the Dirac strings at infinity is a pure gauge transformation, with zero generator. It is useful not to fix this gauge and to allow non zero displacements at infinity, as this enables one to control more easily Poincaré invariance. Indeed, under Lorentz boosts, the strings will naturally change their orientation and it is better not to impose a condition that it should always run, say, along the z -axis.

We close this section by noting that a different way to include sources was developed in [29]. It would be of interest to define consistent asymptotic conditions in that approach too.

## 5 Conclusions

One of the main motivations for studying the asymptotic behaviour of the fields at spatial infinity, on Cauchy hypersurfaces, is to get better tools to understand the structure of is the identity with sources

the space of physical states, a tricky issue in gauge theories. States are indeed naturally defined on Cauchy hypersurfaces. Physical states involves dressings [30]. Both electric and magnetic dressings are necessary in the presence of magnetic poles [31]. Most dressings are usually chosen not to fulfill any particular parity conditions [32, 33], leading to divergences in the boost generators, which must be regulated. Given the ambiguity in the dressing of physical states, one might consider dressings that fulfill our (twisted) parity conditions. This has the advantage of avoiding divergences in Lorentz boosts altogether, as well as logarithmic behaviour at null infinity of some components of the fields which might lead to problems [12, 25]. There is to our knowledge no indication that imposing such parity conditions is a physical limitation.

Note added: after this paper was completed, we became aware of the interesting paper [34] where the duality-symmetric formulation is also considered, but at null infinity, with almost no overlap with our construction.

## Acknowledgments

We thank Glenn Barnich, Oscar Fuentealba, Victor Lekeu, Sucheta Majumdar, Javier Matulich, Stefan Prohazka and Friedrich Schäller for insightful discussions.

The research of MH is supported by the ERC Advanced Grant 'High-Spin-Grav' and by FNRS-Belgium (Convention FRFC PDR T.1025.14 and Convention IISN 4.4503.15). During part of this work, CT was supported by the Max Planck Institute for Gravitational Physics (Albert Einstein Institute) in Potsdam.

## References

- [1] T. He, P. Mitra, A. P. Porfyriadis and A. Strominger, 'New Symmetries of Massless QED,' JHEP 1410 (2014) 112 [arXiv:1407.3789 [hep-th]].
- [2] V. Lysov, S. Pasterski and A. Strominger, 'Low's Subleading Soft Theorem as a Symmetry of QED,' Phys. Rev. Lett. 113 (2014) no.11, 111601 [arXiv:1407.3814 [hep-th]].
- [3] D. Kapec, V. Lysov and A. Strominger, 'Asymptotic Symmetries of Massless QED in Even Dimensions,' arXiv:1412.2763 [hep-th].
- [4] D. Kapec, M. Pate and A. Strominger, 'New Symmetries of QED,' arXiv:1506.02906 [hep-th].
- [5] M. Campiglia and A. Laddha, 'Subleading soft photons and large gauge transformations,' JHEP 1611 (2016) 012 [arXiv:1605.09677 [hep-th]].
- [6] E. Conde and P. Mao, 'Remarks on asymptotic symmetries and the subleading soft photon theorem,' Phys. Rev. D 95 (2017) no.2, 021701 [arXiv:1605.09731 [hep-th]].
- [7] A. Strominger, 'Lectures on the Infrared Structure of Gravity and Gauge Theory,' arXiv:1703.05448 [hep-th].
- [8] A. Strominger, 'Asymptotic Symmetries of Yang-Mills Theory,' JHEP 1407 (2014) 151 [arXiv:1308.0589 [hep-th]].

- [9] G. Barnich and P. H. Lambert, 'Einstein-Yang-Mills theory: Asymptotic symmetries,' Phys. Rev. D 88 (2013) 103006 [arXiv:1310.2698 [hep-th]].
- [10] A. Balachandran and S. Vaidya, 'Spontaneous Lorentz Violation in Gauge Theories,' Eur. Phys. J. Plus 128 (2013), 118 [arXiv:1302.3406 [hep-th]].
- [11] M. Campiglia and R. Eyheralde, 'Asymptotic U (1) charges at spatial infinity,' JHEP 1711 (2017) 168 [arXiv:1703.07884 [hep-th]].
- [12] M. Henneaux and C. Troessaert, 'Asymptotic symmetries of electromagnetism at spatial infinity,' JHEP 1805 (2018) 137 [arXiv:1803.10194 [hep-th]].
- [13] A. Strominger, 'Magnetic Corrections to the Soft Photon Theorem,' Phys. Rev. Lett. 116 (2016) no.3, 031602 [arXiv:1509.00543 [hep-th]].
- [14] S. Deser and C. Teitelboim, 'Duality Transformations of Abelian and Nonabelian Gauge Fields,' Phys. Rev. D 13 (1976) 1592.
- [15] C. Bunster and M. Henneaux, 'Sp(2n,R) electric-magnetic duality as off-shell symmetry of interacting electromagnetic and scalar fields,' PoS HRMS 2010 (2010) 028 [arXiv:1101.6064 [hep-th]].
- [16] R. Benguria, P. Cordero and C. Teitelboim, 'Aspects of the Hamiltonian Dynamics of Interacting Gravitational Gauge and Higgs Fields with Applications to Spherical Symmetry,' Nucl. Phys. B 122 (1977) 61.
- [17] S. Deser, A. Gomberoff, M. Henneaux and C. Teitelboim, 'Duality, selfduality, sources and charge quantization in Abelian N form theories,' Phys. Lett. B 400 (1997) 80 [hep-th/9702184].
- [18] P. A. M. Dirac, 'The Theory of magnetic poles,' Phys. Rev. 74 (1948) 817.
- [19] V. Hosseinzadeh, A. Seraj and M. M. Sheikh-Jabbari, 'Soft Charges and Electric-Magnetic Duality,' JHEP 1808 (2018) 102 [arXiv:1806.01901 [hep-th]].
- [20] L. Freidel and D. Pranzetti, 'Electromagnetic duality and central charge,' Phys. Rev. D 98 (2018) no.11, 116008 [arXiv:1806.03161 [hep-th]].
- [21] M. Henneaux and C. Teitelboim, 'Dynamics of Chiral (Selfdual) p Forms,' Phys. Lett. B 206 (1988) 650.
- [22] J. H. Schwarz and A. Sen, 'Duality symmetric actions,' Nucl. Phys. B 411 (1994) 35 [hep-th/9304154].
- [23] J. Gervais, B. Sakita and S. Wadia, 'The Surface Term in Gauge Theories,' Phys. Lett. B 63 (1976), 55.
- [24] C. W. Bunster, S. Cnockaert, M. Henneaux and R. Portugues, 'Monopoles for gravitation and for higher spin fields,' Phys. Rev. D 73 (2006) 105014 [hep-th/0601222].
- [25] M. Henneaux and C. Troessaert, 'Hamiltonian structure and asymptotic symmetries of the Einstein-Maxwell system at spatial infinity,' JHEP 1807 (2018) 171 [arXiv:1805.11288 [gr-qc]].
- [26] M. Henneaux and C. Teitelboim, 'Quantization of gauge systems,' Princeton, USA: Univ. Pr. (1992) 520 p
- [27] M. Henneaux and C. Teitelboim, 'Consistent quantum mechanics of chiral p forms,' in 2nd Meeting on Quantum Mechanics of Fundamental Systems (CECS) Santiago, Chile, December 17-20, 1987 , pp. 79-112. 1987.

- [28] C. Troessaert, 'The BMS4 algebra at spatial infinity,' Class. Quant. Grav. 35 (2018) no.7, 074003 [arXiv:1704.06223 [hep-th]].
- [29] G. Barnich and A. Gomberoff, 'Dyons with potentials: Duality and black hole thermodynamics,' Phys. Rev. D 78 (2008) 025025 [arXiv:0705.0632 [hep-th]].
- [30] P. A. M. Dirac, 'Gauge invariant formulation of quantum electrodynamics,' Can. J. Phys. 33 (1955) 650.
- [31] Z. Antunovic and P. Senjanovic, 'Coherent States And The Solution Of The Infrared Problem Of The Quantum Field Theory Of Electric And Magnetic Charge,' Phys. Lett. 136B (1984) 423.
- [32] S. B. Giddings, 'Generalized asymptotics for gauge fields,' JHEP 1910 (2019) 066 [arXiv:1907.06644 [hep-th]].
- [33] S. Choi and R. Akhoury, 'Magnetic Soft Charges, Dual Supertranslations and 't Hooft Line Dressings,' arXiv:1912.02224 [hep-th].
- [34] A. Bhattacharyya, L. Y. Hung and Y. Jiang, 'Null hypersurface quantization, electromagnetic duality and asymptotic symmetries of Maxwell theory,' JHEP 1803 (2018) 027 [arXiv:1708.05606 [hep-th]].