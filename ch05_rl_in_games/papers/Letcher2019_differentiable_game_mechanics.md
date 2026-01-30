## Time decay estimate with diffusion wave property and smoothing effect for solutions to the compressible Navier-Stokes-Korteweg system

Takayuki KOBAYASHI and Kazuyuki TSUDA Osaka University,

1-3, Machikaneyamacho, Toyonakashi, 560-8531, JAPAN e-mail: jtsuda@sigmath.es.osaka-u.ac.jp

## Abstract

Time decay estimate of solutions to the compressible Navier-Stokes-Korteweg system is studied. Concerning the linearized problem, the decay estimate with diffusion wave property for an initial data is derived. As an application, the time decay estimate of solutions to the nonlinear problem is given. In contrast to the compressible Navier-Stokes system, for linear system regularities of the initial data are lower and independent of the order of derivative of solutions owing to smoothing effect from the Korteweg tensor. Furthermore, for the nonlinear system diffusion wave property is obtained with an initial data having lower regularity than that of study of the compressible Navier-Stokes system.

Key Words and Phrases. compressible Navier-Stokes-Korteweg system, time decay estimate, diffusion wave property, smoothing effect.

2010 Mathematics Subject Classification Numbers. 35Q30, 76N10

## 1 Introduction

We study time decay estimate for solutions to the following compressible Navier-StokesKorteweg system in R n ( n ≥ 2):

<!-- formula-not-decoded -->

 Here ρ = ρ ( x, t ) and M = ( M 1 ( x, t ) , · · · , M n ( x, t )) denote the unknown density and momentum respectively, at time t ∈ R + and position x ∈ R n ; ρ 0 = ρ 0 ( x ) and M 0 = M 0 ( x )

denote given initial datas; S and K denote the viscous stress tensor and the Korteweg stress tensor that are given by

<!-- formula-not-decoded -->

where d ij ( M ρ ) = 1 2 ( ∂ ∂x i ( M ρ ) j + ∂ ∂x j ( M ρ ) i ) ; µ and µ ′ are the viscosity coefficients that are assumed to be constants satisfying

<!-- formula-not-decoded -->

κ denotes the capillary constant that is assumed to be a positive constant. Note that if κ = 0 in the Korteweg tensor, the usual compressible Navier-Stokes equation (the abbreviation is used by 'CNS' below) is obtained; P = P ( ρ ) is the pressure that is assumed to be a smooth function of ρ satisfying P ′ ( ρ ∗ ) &gt; 0 , where ρ ∗ is a given positive constant and ( ρ ∗ , 0) denotes a given constant state. We consider solutions to (1.1) around the constant state.

The system (1.1) describes two phase flow with phase transition between liquid and vapor in a compressible fluid as a diffuse interface model. In the diffuse interface model, the phase boundary is regarded as a narrow transition layer and fluid state is described by change of the density. Hence it is enough to consider one set of equations and a single spatial domain in contrast to the classical sharp interface model. It is well known that the phase field method use the idea of diffuse interface effectively for numerical simulation.

Concerning derivation of (1.1), Van der Waals [22] observes that a phase transition boundary can be regarded as a thin transition zone, i.e, diffuse interface caused by a steep gradient of the density. Based on his idea, Korteweg [14] suggests the stress tensor including the term ∇ ρ ⊗ ∇ ρ of the Navier-Stokes equation. Then Dunn and Serrin [2] generalize the Korteweg's work and provide the system (1 . 1) with (1.2). In recent works, Heida and M´ alek [7] derive (1.1) by the entropy production method in difference from [2]. Freist¨ uhler and Kotschote [3] derive the Navier-Stokes-Allen-Cahn system and the Navier-Stokes-Chan-Hilliard system which describe two phase flow of mixture materials from some model of Korteweg type. Gorban and Karlin [4] derive the Korteweg tensor from the Bolzmann equation.

We first study time decay estimate for solutions to linearized problem of (1.1). We shall show that the leading part of solutions consists of ρ ∗ and a divergence free momentum field decaying in the same order as a n -dimensional heat kernel as t goes to infinity in L p ( p ≥ 2). Solutions are decomposed into low frequency part and high frequency part. We also shall show that the low frequency part of density decays faster than the n -dimensional heat kernel in L ∞ norm and the decay order is obtained. On the other hand, we show that solutions may grow in L 1 norm as t goes to infinity and the growth order is obtained. These properties are called as 'diffusion wave property' which occurs from terms in the Green matrix given by the convolution of the Green functions of the diffusion equation and the wave equation. The diffusion wave property is studied for CNS by Hoff and

Zumbrun [8, 9], and Kobayashi and Shibata [13]. We also give L p -L q (1 ≤ q ≤ 2 ≤ p ) estimate of solutions for the low frequency part. Concerning the high frequency part, it is shown that solutions have exponential decay as t →∞ similarly to CNS. It differs from CNS that smoothing effect of solutions appears in the estimate of the high frequency part.

To show the decay estimate of solutions to the linearized problem, we use the Fourier transform method as in [13] for CNS and Shibata [17] for the viscoelastic equation. In contrast to [13], due to the Korteweg tensor, the smoothing effect of the heat kernel appears in every components of the Green matrix of (1.1). Therefore we do not assume any regularity for the initial value in the estimate of the high frequency part. (See Theorem 3.4 below. ) On the other hand, though the Korteweg tensor is added, the order of roots for characteristic equations of the linearized system coincides with that of CNS on the low frequency part in the Fourier space. Hence, the estimates of the low frequency part are derived similarly to the proofs of [13].

Furthermore, we derive time decay estimates for solutions to the nonlinear problem (1.1). Concerning (1.1), Danchin and Desjardins [1] show global existence of a solution around the motionless state ( ρ ∗ , 0) with a small initial data u 0 ∈ ( B n 2 2 , 1 ∩ B n 2 -1 2 , 1 ) × B n 2 -1 2 , 1 , where B n 2 2 , 1 denotes the usual homogeneous Besov space. Hattori and Li [5, 6] show the global existence of a H N +1 × H N solution with a small u 0 ∈ H N +1 × H N , where N is an integer satisfying that N ≥ [ n/ 2] + 2 and [ n/ 2] denotes the integer part of n/ 2. For three dimensional case Tan and R. Zhang, X. Zhang and Tan [20, 24] study the global existence for a small u 0 ∈ H 4 × H 3 with large time behavior of solutions. Tan, Wang and Xu [19] state the global existence of a solution which has lower regularity, that is, C ([0 , ∞ ); H 2 × H 1 ) class with a small u 0 ∈ H 2 × H 1 . Wang and Tan [23] show convergence rates of L p (2 ≤ p ) norms of the solution for 3 dimensional case under a small initial value; Let the velocity field v be defined by v = M/ρ . If ‖ ( ρ 0 , v 0 ) ‖ H s +1 × H s &lt;&lt; 1 ( s ≥ 3), then we have that for t &gt; 0

<!-- formula-not-decoded -->

where ( ρ 0 , v 0 ) denotes a given initial data and H s denotes the usual L 2 Sobolev space. If in addition s ≥ 4 we have that

<!-- formula-not-decoded -->

Okita [16] show that for n dimensional case L 2 time decay rate of solutions to CNS around some stationary solution is obtained with a small H s initial data, where s is an integer satisfying that s ≥ [ n/ 2] + 1. We shall show that for (1.1) in n dimensional case nonlinear parts in the Duhamel formula have faster decay than linear parts and L ∞ and L 1 estimates of solutions are stated as t goes to infinity with the decay and growth rates. We also obtain L 2 time decay estimate of solutions with the decay rate similar to [23] and CNS [16]. It differs from [16] and [23] that in our results the estimates reflect the diffusion wave property of the system. In both [16] and [23] the diffusion wave property of solutions to the nonlinear problem is not studied. In addition, by decomposition method

for solution (cf. [16] for CNS), the initial data is assumed in lower regularity class than that of [8, 23]. Concerning L ∞ estimate of solutions in the high frequency part, since that of linear problem has singularity at t = 0, we apply L 2 energy method instead of using the estimates of linear problem in the high frequency part. Concerning L 1 estimate of solutions we use another type of estimate which has lower singularity at t = 0 in the high frequency part. Note that by the Korteweg tensor ρ has higher regularity than that of velocity.

To obtain Theorem 3.6 (i) below, the conservation form has an important role similarly to the proof of [8] for CNS. Note that owing to the smoothing effect of ρ from the Korteweg tensor, even if we consider the conservation form (1.1) no derivative loss occurs in the energy method for the high frequency part. The energy estimate is one of key points in the method of [16]. Therefore, we have the estimates of solutions to (1.1) including Theorem 3.6 (i) in lower regularity. As for CNS, due to the derivative loss of the density, the energy estimate is incompatible with the conservation form. Hence we can not obtain a similar property to Theorem 3.6 (i) in lower regularity by using the decomposition method. Indeed, [8] shows a similar property to Theorem 3.6 (i) with an initial data in H [ n/ 2]+3 class for CNS.

This paper is organized as follows. In section 2 notations and lemmas are described which shall be used in this paper. In section 3, main results are stated for the linearized and nonlinear problems of (1.1) respectively. In section 4, the proof of the time decay estimate of solutions to the linearized problem of (1.1) is given. In section 4, that to nonlinear problem is given.

@

## 2 Preliminaries

In this section we introduce notations which will be used throughout this paper. Furthermore, we introduce some lemmas which will be useful in the proof of the main results.

We denote the norm on X by ‖ · ‖ X for a given Banach space X .

Let 1 /lessdblequal p /lessdblequal ∞ . L p denotes the usual L p space on R n . Let k be a nonnegative integer. W k,p and H k denotes the usual L p and L 2 Sobolev space of order k respectively. (As usual, we define that H 0 := L 2 .)

For simplicity, L p denotes the set of all vector fields w = /latticetop ( w 1 , · · · , w n ) on R n with w j ∈ L p ( j = 1 , · · · , n ) and ‖ · ‖ L p denotes the norm ‖ · ‖ ( L p ) n if no confusion will occur. Similarly, a function space X denotes the set of all vector fields w = /latticetop ( w 1 , · · · , w n ) on R n with w j ∈ L p ( j = 1 , · · · , n ) and the norm ‖ · ‖ X n is denoted by ‖ · ‖ X if no confusion will occur.

We take u = /latticetop ( φ, m ) with φ ∈ H k and m = /latticetop ( m 1 , · · · , m n ) ∈ H j . Then the norm ‖ u ‖ H k × H j denotes the norm of u on H k × H j , that is, we define

<!-- formula-not-decoded -->

When j = k , for simplicity we denote H k × ( H k ) n by H k . The norm ‖ u ‖ H k denotes the

norm ‖ u ‖ H k × ( H k ) n i.e., we define that

<!-- formula-not-decoded -->

Similarly, for u = /latticetop ( φ, m ) ∈ X × Y with m = /latticetop ( m 1 , · · · , m n ) , the norm ‖ u ‖ X × Y denotes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If Y = X n , the symbol X denotes X × X n for simplicity, and we define its norm ‖ u ‖ X × X n by ‖ u ‖ X ;

The symbols ˆ f and F [ f ] denote the Fourier transform of f for the space variables;

<!-- formula-not-decoded -->

In addition, the inverse Fourier transform of f is denoted by F -1 [ f ];

<!-- formula-not-decoded -->

The function space H k ( ∞ ) denotes the set of all u ∈ H k satisfying supp ˆ u ⊂ {| ξ | ≥ r ∞ } , where r ∞ denotes a positive constant.

For operators L 1 and L 2 , we denote by [ L 1 , L 2 ] the commutator of L 1 and L 2 , i.e.,

<!-- formula-not-decoded -->

For a nonnegative number s , [ s ] denotes the integer part of s .

The symbol ' ∗ ' denotes the spatial convolution.

We next state some lemmas which will be used in the proof of the main results.

The following lemma is the well-known Sobolev type inequality.

Lemma 2.1. Let s satisfy s &gt; n/ 2 . Then there holds the inequality

<!-- formula-not-decoded -->

for f ∈ H s .

The following inequalities are stated which are concerned with composite functions.

Lemma 2.2. Let s be an integer satisfying s ≥ [ n/ 2] + 1 . Let s j and µ ( j ) ( j = 1 , · · · , /lscript ) be nonnegative integers and multiindices satisfying 0 ≤ | µ ( j ) | ≤ s j ≤ s + | µ ( j ) | , µ = µ (1) + · · · + µ ( /lscript ) , s = s 1 + · · · + s /lscript ≥ ( /lscript -1) s + | µ | , respectively. Then there holds

<!-- formula-not-decoded -->

See, e.g., [11] for the proof of Lemma 2 . 2.

Lemma 2.3. Let s be an integer satisfying s ≥ [ n/ 2] + 1 . Suppose that F is a smooth function on I , where I is a compact interval of R . Then for a multi-index α with 1 ≤ | α | ≤ s , there hold the estimates

<!-- formula-not-decoded -->

for f 1 ∈ H s with f 1 ( x ) ∈ I for all x ∈ R n and f 2 ∈ H | α | ; and

<!-- formula-not-decoded -->

for f 1 ∈ H s +1 with f 1 ( x ) ∈ I for all x ∈ R n and f 2 ∈ H | α |-1 .

See, e.g., [10] for the proof of Lemma 2 . 3.

## 3 Main results

In this section, main results are stated for (1 . 1). (1.1) is reformulated as follows. Hereafter we assume that ρ ∗ = 1 without loss of generality. We set φ = ρ -1 and m = M γ where γ = √ P ′ (1). Substituting φ and m into (1.1), then we obtain

 where u = /latticetop ( φ, m ), ν = µ , ˜ ν = µ + µ ′ , κ 0 = κ γ , φ 0 = ρ 0 -1, m 0 = M 0 γ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first consider the time decay estimate of solutions to linearized problem for (1 . 1). (1 . 1) is linearized as follows.

<!-- formula-not-decoded -->

By taking the Fourier transform of (3.2) with respect to the space variable x , we obtain the following ordinary differential equation with a parameter ξ .

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

 Therefore, the solutions of (3.2) are given by the following formulas. Hereafter we define that A = ν + ˜ ν 2 , B = 2 γ ν + ˜ ν , K = 2 √ κ 0 γ ν + ˜ ν . If | ξ | = 0 , B/ √ 1 -K 2 when 0 &lt; K &lt; 1 and | ξ | /negationslash = 0 when K ≥ 1, then the Fourier transforms of φ and m are given by where

denote roots of the characteristic equation of (3.3). Note that when | ξ | /negationslash = 0 and K ≥ 1 it holds that λ + ( ξ ) -λ -( ξ ) = 0. In addition, due to the term in (3.2) from the Korteweg tensor, in contrast to [13], the higher order term with respect to ξ , i.e.,

<!-- formula-not-decoded -->

appears in (3.4). On the otherhand, if 0 &lt; K &lt; 1 and min 1 2 , 2 ξ

2 B √ 1 K 2 , then ˆ φ and ˆ m are given by

{ B 2 √ 1 -K } ≤ | | ≤

<!-- formula-not-decoded -->

where Γ stands for a closed pass including λ ± and included in the set { z ∈ C | Re z ≤ -c 0 } and c 0 stands for a positive number satisfying that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash

We define a cut-off function ϕ 1 in C ∞ ( R n ) as follows. We assume that K = 1.

/negationslash

<!-- formula-not-decoded -->

Furthermore, we define cut-off functions ϕ ∞ and ϕ M in C ∞ ( R n ) by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If K = 1, ϕ 1 and ϕ ∞ are defined as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We define solution operators on low frequency part E 1 and that on high frequency part E ∞ of (3.2) as follows.

E

1

(

t

) = (

E

1

,φ

(

t

)

, E

1

,m

(

t

))

,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that as in [21] E 1 is continuous for t ≥ 0 on L p (1 ≤ p ≤ ∞ ) and E ∞ is continuous for t ≥ 0 on H s +1 × H s ( s ≥ [ n/ 2] + 1). If an initial time is t 0 &gt; 0 but not 0 in (3.2), then we write the solution operators by E j ( t, t 0 ) ( j = 1 , ∞ ) respectively.

Concerning the solution ( φ, m ) to (3.2) and the solution operators on the low frequency part, (3.9) and (3.10), we obtain the following estimates.

Theorem 3.1. (i) It holds that for the solution ( φ, m ) to (3.2) and t &gt; 0

<!-- formula-not-decoded -->

∥ ∥ where K ν = K ν ( t, x ) denotes the standard heat kernel and m 0 ,in denotes a divergence-free part of m 0 that are respectively given by

<!-- formula-not-decoded -->

(ii) ( L ∞ estimate for E 1 ) For t &gt; 0 it holds that

<!-- formula-not-decoded -->

(iii) ( L 1 estimate for E 1 ) When the space dimension n ≥ 3 and n is an odd number, then for any t &gt; 0 we have the following estimate.

<!-- formula-not-decoded -->

(iv) ( L p -L q estimate for E 1 ) For t &gt; 0 and 1 ≤ q ≤ 2 ≤ p ≤ ∞ we have the following estimate.

<!-- formula-not-decoded -->

Remark 3.2. Concerning (iii), so far we do not obtain the similar estimate when the space dimension n ≥ 2 and n is an even number. The key point to obtain (iii) is pointwise estimate of the Green function as mentioned in the proof of Theorem 3.1 below. In the pointwise estimate, we need a great deal of cancellation to overcome the similar difficulty related to the Riesz kernel to that of [13]. As in [9] it seems that the pointwise estimates of the Green function are different between odd dimensional case and even dimensional case by the Huygens principle and the estimate with even dimensional case is more complicated than that of odd dimensional case. Hence, more delicate analysis is needed to obtain the similar estimate to (iii) in even dimensional case. Note that diffusion wave property, especially the retardation of the parabolic decay in L p ( p &lt; 2) occurs for multi dimensional case as in [8].

Remark 3.3. We discuss the optimality of the decay exponents in Theorem 3.1 below. Concerning (i) and (iv), the first approximation of solutions is K ν ∗ m 0 ,in , that is, the Stokes flow part and optimality of decay exponents of solutions to the Stokes equation is well known. Concerning (ii) and (iii), Hoff and Zumbrun [9] consider some linear artificial viscous equation whose solutions approximate behavior of solutions to the linearized compressible Navier-Stokes equation. They give not only upper bounds but also lower ones of the Green function and verify that decay exponents of L p bounds are sharp at least up to a logarithmic term. The decay exponents coincide with those of (ii) and (iii). Therefore we think that the decay rates of (ii) and (iii) are optimal.

Furthermore, the following estimate holds for the solution operator on the high frequency part (3.8).

Theorem 3.4. ( L p -L p estimate for E ∞ ) Let 1 ≤ p ≤ ∞ . Then it holds that

<!-- formula-not-decoded -->

for t &gt; 0 , k ≥ 0 and | α | ≥ 0 , where ( δ 1 , δ 2 ) = (1 / 2 , 1) , (1 , 3 / 2) for K = 1 and K = 1 respectively. In addition, when K = 1 and 0 &lt; t ≤ 1 , we have the following estimate.

/negationslash

<!-- formula-not-decoded -->

for 1 ≤ p ≤ ∞ , where σ 0 is any positive number satisfying 0 &lt; σ 0 &lt; 1 / 2 .

Remark 3.5. Theorem 3.4 implies smoothing effect of solutions to the linearized problem in the high frequency part. The estimate (3.14) has lower singularity at t = 0 than (3.13) with K = 1.

We next consider time decay estimates of solutions to the nonlinear problem (1.1). The following L ∞ , L 2 and L 1 estimates are stated for the solution u = /latticetop ( φ, m ) to the system (3.1).

Theorem 3.6. Let u = /latticetop ( φ, m ) be the solution to (3.1) .

(i) Let E = E ( t ) be the solution operator for the linearized problem (3.2) defined by E = E 1 + E ∞ . We assume that u 0 = /latticetop ( φ 0 , m 0 ) ∈ ( H s +1 × H s ) ∩ L 1 , where s denotes a nonnegative integer satisfying s ≥ [ n/ 2] + 1 . We define the norm ||| u 0 ||| s by

<!-- formula-not-decoded -->

There exists a constant /epsilon1 1 &gt; 0 such that if ||| u 0 ||| s ≤ /epsilon1 1 , then for t ≥ 0 we have the estimate

<!-- formula-not-decoded -->

where δ 1 ( t ) = 1 for n ≥ 3 and δ 1 ( t ) = log(1 + t ) for n = 2 .

(ii) Under the assumption of (i) it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where q ( n ) = -3 n -1 4 for n = 2 , 3 and q ( n ) = -n 2 -1 2 for n ≥ 4 . Furthermore, it also holds that for t ≥ 0 and k = 0 , 1

<!-- formula-not-decoded -->

(iii) Let n be an odd number satisfying n ≥ 3 . There exists a constant /epsilon1 2 &gt; 0 such that if ||| u 0 ||| s ≤ /epsilon1 2 , then the following estimate is true for t ≥ 1 .

<!-- formula-not-decoded -->

Remark 3.7. Concerning the first approximation of u , i.e., K ν ∗ m 0 ,in , we have the following estimate.

<!-- formula-not-decoded -->

Remark 3.8. In Theorem 3.6, the diffusion wave property of the solution appears in L ∞ and L 1 estimates.

## 4 Proof of the estimates for solution to the linear problem

In this section, we give the proofs of Theorem 3.1 and Theorem 3.4. To prove Theorem 3.1, we put

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

for j = 1 , ∞ . We see from (3.4) that

<!-- formula-not-decoded -->

for j = 1 , ∞ . We first show Theorem 3.1 (ii). Concerning the proof of Theorem 3.1 (ii), it is enough to show the following proposition by the same reason as that in [13, Theorem 2.1 (1)] based on the Young inequality.

## Proposition 4.1. We set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ψ = ψ ( ω ) ∈ C ∞ ( S n -1 ) , S n -1 = { ξ ∈ R n ; | ξ | = 1 } and ψ ( ξ ) = ψ ( ξ/ | ξ | ) . Then it holds that for n ≥ 2 and t &gt; 0 .

Proposition 4.1 is yielded as follows; As for the estimate near a light cone, that is, for { ( t, x ); | x | ≥ R 0 t, t ≥ max(1 , R/R 0 ) 4 } , using the stationary phase method as that in [13] we directly obtain Proposition 4.1, where R and R 0 are some suitable positive constants used in the argument. In the case such that t ≥ 1 and | x | ≤ R 0 t , we set

<!-- formula-not-decoded -->

∣ If ( ν + ˜ ν ) 2 &lt; 4 κ 0 γ , we define f 1 and g 1 by

<!-- formula-not-decoded -->

If ( ν + ˜ ν ) 2 ≥ 4 κ 0 γ , we define f 2 and g 2 by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In contrast to [13], new terms κ 0 ∇ ∆ φ and (3.5) appear in the linearized problem and the solution formula respectively. However, f j and g j ( j = 1 , 2) have the same order of | ξ | as f and g used in the proof of [13, Theorem 2.1 (1)] as | ξ | goes to 0. Therefore a similar manner to the proof of [13, Theorem 2.1 (1)] based on (4.2), the well-known formulas for fundamental solution to wave equation and Shimizu and Shibata [18, Theorem 2.3] shows Proposition 4.1 in the case such that t ≥ 1 and | x | ≤ R 0 t . Since direct calculation shows that

<!-- formula-not-decoded -->

we get Proposition 4.1. Using f j and g j ( j = 1 , 2), Theorem 3.1 (iii) and Theorem 3.1 (iv) are directly verified by a similar proof to that of [13, Theorem 2.1 (2), Theorem 2.3].

Theorem 3.1 (i) easily follows from Proposition 4.1, definitions of L i,j and the Young inequality.

We next show Theorem 3.4. Similarly to [13], we define K -, ∞ ( t ) and K -,/lscript ( t, x ) by

<!-- formula-not-decoded -->

and for /lscript ≥ 0. Note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We put L ± ( t ), M ± ,β ( t ), K + , ∞ ( t ) and K 1 , ∞ ( t ) by the same forms as those used in [13], i.e., we put

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that L ab, ∞ ( a = 1 , 2 , b = 1 , 2) and K a, ∞ ( a = 1 , 2 , 3) are linear combinations by L ± ( t ), M ± ,β ( t ), ∆ M ± ,β ( t ), K ± , ∞ ( t ) and K 1 , ∞ ( t ). We shall show that

<!-- formula-not-decoded -->

for K = 1, t &gt; 0, /lscript ≥ 0, | α | ≥ 0 and 1 ≤ p ≤ ∞ . When K = 1, we also show that

<!-- formula-not-decoded -->

/negationslash

Indeed, if K = 1, since κ 0 &gt; 0, we have that

/negationslash

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If K = 1, we have that

Hence, if ν + ˜ ν &gt; 2 √ κ 0 γ , we can rewrite λ -to

<!-- formula-not-decoded -->

where and note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash

∣ ∣ Furthermore, if ν +˜ ν ≤ 2 √ κ 0 γ we see that Re λ -= -A | ξ | 2 . Hence we see from (4.5) and (4.6) that for | ξ | ≥ 2 B √ | 1 -K 2 | with K = 1 or | ξ | ≥ 1 with K = 1 and | β | ≥ 0 there exist positive constants c 1 and c 2 such that

In addition, we see that

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

Therefore, using and the integration by parts n +1 times, we obtain that for K = 1

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

On the other hand, we also obtain from (4.5) that

<!-- formula-not-decoded -->

It follows from (4.8) and (4.9) that

<!-- formula-not-decoded -->

(4.10) and the Young inequality derive (4.3). When K = 1, (4.4) is verified by a similar argument to the proof of (4.3). Furthermore, when K = 1, we see that for 0 &lt; t ≤ 1 and 0 &lt; σ 0 &lt; 1 / 2

<!-- formula-not-decoded -->

∣ ∣ (4.11) and the strong L p multiplier theorem ([8, Proposition 4.2]) show that

<!-- formula-not-decoded -->

for 1 ≤ p ≤ ∞ and 0 &lt; t ≤ 1. L ± ( t ), M ± ,β ( t ), K + , ∞ ( t ) and K 1 , ∞ ( t ) are estimated similarly to (4.3), (4.4) and (4.12). In addition, we see from the estimate of M ± ,β ( t ) and (4.7) that

<!-- formula-not-decoded -->

where δ 2 is the same one as that in Theorem 3.4. Concerning the estimate in the middle frequency part the desired estimate directly follows from the solution formulas (3.6) and (3.7) and the same manner as that in [13], i.e., we have the estimate

<!-- formula-not-decoded -->

for 1 ≤ p ≤ ∞ . This together with estimates L ± ( t ), M ± ,β ( t ), K ± , ∞ ( t ) and K 1 , ∞ ( t ) shows Theorem 3.4. This completes the proof. /square

## 5 Proof of the estimates for solution to the nonlinear problem

In this section, we give the proof of Theorem 3.6. Concerning (3.1), set

<!-- formula-not-decoded -->

Then (3.1) is rewritten as

<!-- formula-not-decoded -->

where F ( u ) = /latticetop (0 , f ( u )). Note that we already have existence of global H s +1 × H s solutions in the introduction. Hence our task is to discuss a priori estimates of the solutions to (5.1).

We consider decomposition of solutions to (5.1) into low frequency part and high frequency part as in [16]. Operators P 1 and P ∞ on L 2 are defined as follows for the decomposition;

<!-- formula-not-decoded -->

We have the following properties for P j ( j = 1 , ∞ ).

Lemma 5.1. [16, Lemma 4.2] Let k be a nonnegative integer. Then P 1 is a bounded linear operator from L 2 to H k . In fact, it holds that

<!-- formula-not-decoded -->

As a result, for any 2 ≤ p ≤ ∞ , P 1 is bounded from L 2 to L p .

Lemma 5.2. [16, Lemma 4.2], [12, Lemma 4.4] (i) Let k be a nonnegative integer. Then P ∞ is a bounded linear operator on H k .

- (ii) There hold the inequalities

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let u = /latticetop ( φ, m ) be a solution to (5.1). Based on the Duhamel principle, similarly to [16, Proposition 4.3], we decompose the solution to (5.1) into low frequency part and high frequency part as

<!-- formula-not-decoded -->

where u j = P j u and u 0 j = P j u 0 ( j = 1 , ∞ ).

We prove Theorem 3.6 (ii) before we prove (i). We set

<!-- formula-not-decoded -->

where φ 1 = P 1 φ and m 1 = P 1 m . Furthermore, we also set

<!-- formula-not-decoded -->

for the space dimension n = 2 , 3 and

<!-- formula-not-decoded -->

for the space dimension n ≥ 4. By Theorem 3.1 (ii) and (5.2), it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Concerning estimate of the second term of right hand side in (5.3), we obtain that

<!-- formula-not-decoded -->

where we define that M ( t ) = M 1 ( t )+ M ∞ ( t ). In fact, as for the convection term P 1 div ( m ⊗ m ) in the nonlinearity P 1 F ( u 1 + u ∞ ), we see from Lemma 5.1 that

<!-- formula-not-decoded -->

and then it follow from Theorem 3.1 (ii) that

<!-- formula-not-decoded -->

Since other nonlinear terms are estimated similarly, we have (5.6). We next show that

<!-- formula-not-decoded -->

where δ 1 ( t ) is the same one in Theorem 3.6 (i). Indeed, we define I 1 and I 2 by and

<!-- formula-not-decoded -->

I 1 is estimated by Theorem 3.1 (ii) as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

I 2 can be estimated directly. Then we see from the estimates I 1 and I 2 that

<!-- formula-not-decoded -->

∥ ∥ Since other nonlinear term can be estimated similarly to (5.8), we have (5.7). (5.7) together with (5.3), (5.4), (5.5) and (5.6) imply that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, we also obtain the following L 2 type estimate similarly based on Theorem 3.1 (iv).

<!-- formula-not-decoded -->

for k = 0 , 1.

Concerning the L ∞ type estimate for u ∞ , since the estimate in Theorem 3.4 has singularity at t = 0 and the estimate of derivative of solutions has stronger singularity, we use L 2 energy estimate instead of using Theorem 3.4. By Lemma 2.1 we see that

<!-- formula-not-decoded -->

Hence it is enough to obtain L 2 type estimate of u ∞ . As for the L 2 type estimate of u ∞ , using L 2 energy estimate stated in [21, Proposition 5.4], the following proposition is obtained for the high frequency part.

Proposition 5.3. Let s be a nonnegative integer satisfying s ≥ [ n/ 2] + 1 . Assume that

<!-- formula-not-decoded -->

for all T ′ &gt; 0 . Assume also that u ∞ = /latticetop ( φ ∞ , m ∞ ) satisfies

<!-- formula-not-decoded -->

for all T ′ &gt; 0 . Then there exists an energy functional E [ u ∞ ] such that there holds the estimate

<!-- formula-not-decoded -->

on (0 , T ′ ) for all T ′ &gt; 0 . Here d is a positive constant; C is a positive constant independent of T ′ ; E [ u ∞ ] is equivalent to ‖ u ∞ ‖ 2 H s +1 × H s , i.e,

<!-- formula-not-decoded -->

and E [ u ∞ ]( t ) is absolutely continuous in t ∈ [0 , T ′ ] for all T ′ &gt; 0 .

To apply the energy method, concerning estimate of the nonlinearity P ∞ F ( u ) in H s × H s -1 , the following estimate is verified by direct computation based on Lemmas 2.1-2.3.

## Lemma 5.4. It holds that

<!-- formula-not-decoded -->

We set D [ u ∞ ] = ‖∇ φ ∞ ( t ) ‖ 2 H s +1 + ‖∇ m ∞ ( t ) ‖ 2 H s . By (5.13) and Lemma 5.4, it is obtained that there exists a positive constant c 3 such that

<!-- formula-not-decoded -->

D [ u ∞ ] and ˜ E [ u ∞ ] are defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for n = 2 , 3 and

for n ≥ 4. We see from (5.14) that

<!-- formula-not-decoded -->

Combining (5.9), (5.10), (5.11) with (5.15), it is derived that

<!-- formula-not-decoded -->

Especially, we get

<!-- formula-not-decoded -->

Since M ( t ) is continuous in t , there exists a time t 1 &gt; 0 such that

<!-- formula-not-decoded -->

for t ∈ [0 , t 1 ]. This together with (5.16) derives that if /epsilon1 1 in Theorem (ii) is sufficient small we have that there exists a positive constant C 2 such that

M ( t ) ≤ C 2 uniformly for all t.

Consequently, Theorem 3.6 (ii) is verified.

Theorem 3.6 (i) directly follows from (5.9) and (5.10)

The proof of Theorem 3.6 (iii) is given as follows. By (5.2) we derive that

<!-- formula-not-decoded -->

for j = 1 , ∞ . We set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Concerning the estimates of J 1 and J 2 , we derive the following estimates by direct computations based on L 1 type estimates in Theorems 3.1 and 3.4.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By (5.17), (5.18), Theorems 3.1 and 3.4, it holds that

<!-- formula-not-decoded -->

for j = 1 , ∞ . Since u = u 1 + u ∞ and M ( t ) is bounded by u 0 in Theorem 3.6 (ii), we obtain Theorem 3.6 (iii). This completes the proof. /square

Acknowledgements. The first author is partly supported by Grants-in-Aid for Scientific Research with the Grant number: 16H03945. The second author is partly supported by Grant-in-Aid for JSPS Fellows with the Grant number: A17J047780.

## References

- [1] R. Danchin and B. Desjardins, Existence of solutions for compressible fluid models of Korteweg type, Ann. Inst. Henri Poincar´ e Anal. nonlinear 18 (2001) pp.97-133.
- [2] J.E. Dunn and J. Serrin, On the thermomechanics of interstitial working, Arch. Rational Mech. Anal., 88 (1985), pp. 95-133.
- [3] H. Freist¨ uhler and M. Kotschote, Phase-field and Korteweg-type models for the timedependent flow of compressible two-phase fluids, Arch. Ration. Mech. Anal., 224 (2017), pp.1-20.
- [4] A. N. Gorban and I. V. Karlin, Beyond Navier-Stokes equations: Capillarity of ideal gas, Contemporary physics, 58 (2017) pp. 70-90.
- [5] H. Hattori and D. N. Li, Solutions for Two-Dimensional System for Materials of Korteweg Type, SIAM J. Math. Anal., 25 (1994), pp. 85-98.
- [6] H. Hattori and D. N. Li, Global Solutions of a High Dimensional System for Korteweg Materials, J. Math. Anal. Appl., 198 (1998), pp. 84-97.
- [7] M. Heida and J. M´ alek, On compressible Korteweg fluid-like materials, Internat. J. Engrg. Sci., 48 (2010), pp. 1313-1324.
- [8] D. Hoff and K. Zumbrun, Multi-dimensional diffusion waves for the Navier-Stokes equations of compressible flow, Indiana Univ. Math. J., 44 (1995), pp.603-676.
- [9] D. Hoff and K. Zumbrun, Pointwise decay estimates for multidimensional NavierStokes diffusion waves, Z. Angew. Math. Phys., 48 (1997), pp.597-614.
- [10] Y. Kagei and S. Kawashima, Stability of planar stationary solutions to the compressible Navier-Stokes equation on the half space, Commun. Math. Phys., 266 (2006), pp. 401-430.

- [11] Y. Kagei and T. Kobayashi, Asymptotic Behavior of Solutions of the Compressible Navier-Stokes Equation on the Half Space, Arch. Rational Mech. Anal., 177 (2005), pp. 231-330.
- [12] Y. Kagei and K. Tsuda, Existence and stability of time periodic solution to the compressible Navier-Stokes equation for time periodic external force with symmetry, J. Differential Equations, 258 (2015), pp.399-444.
- [13] T. Kobayashi and Y. Shibata, Remark on the rate of decay of solutions to linearized compressible Navier-Stokes equations, Pacific Journal of Mathematics, 207 (2002), pp. 199-234.
- [14] D.J. Korteweg, Sur la forme que prennent les ´ equations du mouvement des fluides si lfon tient compte des forces capillaires caus´ ees par des variations de densit´ e consid´ erables mais continues et sur la th´ eorie de la capillarite dans lfhypoth` ese dfune variation continue de la densit´ e, Archives N´ eerlandaises des sciences exactes et naturelles, Ser 2 (6) (1901), pp. 1-24.
- [15] M. Kotschote, Strong solutions for a compressible fluid model of Korteweg type, Annales de l'Institut Henri Poincar´ e, 25 (2008), pp. 679-696.
- [16] M. Okita, On the convergence rates for the compressible Navier- Stokes equations with potential force, Kyushu J. Math. 68 (2014), pp. 261-286.
- [17] Y. Shibata, On the rate of decay of solutions to linear viscoelastic equation, Math. Methods Appl. Sci., 23 (2000), pp.203-226.
- [18] Y. Shibata and S. Shimizu, A decay property of the Fourier transform and its application to the Stokes problem, J. Math. Fluid Mech, 3 (2001), pp. 213 - 230.
- [19] Z. Tan, H. Wang and J. Xu, Global existence and optimal L 2 decay rate for the strong solutions to the compressible fluid models of Korteweg type, J. Math. Anal. Appl., 390 (2012), pp.181-187.
- [20] Z. Tan and R. Zhang, Optimal decay rates of the compressible fluid models of Korteweg type, Z. Angew. Math. Phys., 65 (2014), pp.279-300.
- [21] K. Tsuda, Existence and stability of time periodic solution to the compressible Navier-Stokes-Korteweg system on R 3 , J. Math. Fluid Mech., 18 (2016), pp.157185.
- [22] J.D. Van der Waals, Th´ eorie thermodynamique de la capillarit´ e, dans lfhypoth` ese dfune variation continue de la densit´ e, Archives N´ eerlandaises des sciences exactes et naturelles XXVIII (1893), pp. 121-209.
- [23] Y. Wang and Z. Tan, Optimal decay rates for the compressible fluid models of Korteweg type, J. Math. Anal. Appl., 379 (2011), pp. 256-271.

- [24] X. Zhang and Z. Tan, Decay estimates of the non-isentropic compressible fluid models of Korteweg type in R 3 , Commun. Math. Sci., 12 (2014), pp.1437-1456.