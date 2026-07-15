                                         Journal of Machine Learning Research 20 (2019) 1-40                                Submitted 1/19; Published 2/19




                                                                     Differentiable Game Mechanics

                                         Alistair Letcher∗                                                                  ahp.letcher@gmail.com
                                         University of Oxford
                                         David Balduzzi∗                                                                     dbalduzzi@google.com
                                         DeepMind
                                         Sébastien Racanière                                                             sracaniere@google.com




arXiv:1905.04926v1 [cs.LG] 13 May 2019
                                         DeepMind
                                         James Martens                                                                 jamesmartens@google.com
                                         DeepMind
                                         Jakob Foerster                                                                 jakobfoerster@gmail.com
                                         University of Oxford
                                         Karl Tuyls                                                                         karltuyls@google.com
                                         DeepMind
                                         Thore Graepel                                                                            thore@google.com
                                         DeepMind


                                         Editor: Kilian Weinberger


                                                                                           Abstract
                                              Deep learning is built on the foundational guarantee that gradient descent on an objective
                                              function converges to local minima. Unfortunately, this guarantee fails in settings, such
                                              as generative adversarial nets, that exhibit multiple interacting losses. The behavior of
                                              gradient-based methods in games is not well understood – and is becoming increasingly
                                              important as adversarial and multi-objective architectures proliferate. In this paper, we
                                              develop new tools to understand and control the dynamics in n-player differentiable games.
                                                  The key result is to decompose the game Jacobian into two components. The first,
                                              symmetric component, is related to potential games, which reduce to gradient descent on
                                              an implicit function. The second, antisymmetric component, relates to Hamiltonian games,
                                              a new class of games that obey a conservation law akin to conservation laws in classical
                                              mechanical systems. The decomposition motivates Symplectic Gradient Adjustment (SGA),
                                              a new algorithm for finding stable fixed points in differentiable games. Basic experiments
                                              show SGA is competitive with recently proposed algorithms for finding stable fixed points
                                              in GANs – while at the same time being applicable to, and having guarantees in, much
                                              more general cases.
                                              Keywords: Game Theory, Generative Adversarial Networks, Deep Learning, Classical
                                              Mechanics, Hamiltonian Mechanics, Gradient Descent, Dynamical Systems


                                         1. Introduction
                                         A significant fraction of recent progress in machine learning has been based on applying
                                         gradient descent to optimize the parameters of neural networks with respect to an objective
                                         function. The objective functions are carefully designed to encode particular tasks such as

                                         c 2019 Alistair Letcher, David Balduzzi, Sébastien Racanière, James Martens, Jakob Foerster, Karl Tuyls, Thore
                                         Graepel.
                                         License: CC-BY 4.0, see https://creativecommons.org/licenses/by/4.0/. Attribution requirements are provided
                                         at http://jmlr.org/papers/v20/19-008.html.
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




supervised learning. A basic result is that gradient descent converges to a local minimum of
the objective function under a broad range of conditions (Lee et al., 2017). However, there
is a growing set of algorithms that do not optimize a single objective function, including:
generative adversarial networks (Goodfellow et al., 2014; Zhu et al., 2017), proximal gradient
TD learning (Liu et al., 2016), multi-level optimization (Pfau and Vinyals, 2016), synthetic
gradients (Jaderberg et al., 2017), hierarchical reinforcement learning (Wayne and Abbott,
2014; Vezhnevets et al., 2017), intrinsic curiosity (Pathak et al., 2017; Burda et al., 2019),
and imaginative agents (Racanière et al., 2017). In effect, the models are trained via games
played by cooperating and competing modules.
    The time-average of iterates of gradient descent, and other more general no-regret
algorithms, are guaranteed to converge to coarse correlated equilibria in games (Stoltz and
Lugosi, 2007). However, the dynamics do not converge to Nash equilibria – and do not
even stabilize in general (Mertikopoulos et al., 2018; Papadimitriou and Piliouras, 2018).
Concretely, cyclic behaviors emerge even in simple cases, see example 1.
    This paper presents an analysis of the second-order structure of game dynamics that
allows to identify two classes of games, potential and Hamiltonian, that are easy to solve
separately. We then derive symplectic gradient adjustment 1 (SGA), a method for finding
stable fixed points in games. SGA’s performance is evaluated in basic experiments.

1.1 Background and problem description
Tractable algorithms that converge to Nash equilibria have been found for restricted classes of
games: potential games, two-player zero-sum games, and a few others (Hart and Mas-Colell,
2013). Finding Nash equilibria can be reformulated as a nonlinear complementarity problem,
but these are ‘hopelessly impractical to solve’ in general (Shoham and Leyton-Brown, 2008)
because the problem is PPAD hard (Daskalakis et al., 2009).
     Players are primarily neural nets in our setting. For computational reasons we restrict
to gradient-based methods, even though game-theorists have considered a much broader
range of techniques. Losses are not necessarily convex in any of their parameters, so Nash
equilibria are not guaranteed to exist. Even leaving existence aside, finding Nash equilibria
in nonconvex games is analogous to, but much harder than, finding global minima in neural
nets – which is not realistic with gradient-based methods.
     There are at least three problems with gradient-based methods in games. Firstly, the
potential existence of cycles (recurrent dynamics) implies there are no convergence guarantees,
see example 1 below and Mertikopoulos et al. (2018). Secondly, even when gradient descent
converges, the rate of convergence may be too slow in practice because ‘rotational forces’
necessitate extremely small learning rates, see figure 4. Finally, since there is no single
objective, there is no way to measure progress. Concretely, the losses obtained by the
generator and the discriminator in GANs are not useful guides to the quality of the images
generated. Application-specific proxies have been proposed, for example the inception score
for GANs (Salimans et al., 2016), but these are of little help during training. The inception
score is domain specific and is no substitute for looking at samples. This paper tackles the
first two problems.

1. Source code is available at https://github.com/deepmind/symplectic-gradient-adjustment.


                                               2
                               Differentiable Game Mechanics




1.2 Outline and summary of main contributions
The infinitesimal structure of games. We start with the basic case of a zero-sum
bimatrix game: example 1. It turns out that the dynamics under simultaneous gradient
descent can be reformulated in terms of Hamilton’s equations. The cyclic behavior arises
because the dynamics live on the level sets of the Hamiltonian. More directly useful, gradient
descent on the Hamiltonian converges to a Nash equilibrium.
    Lemma 1 shows that the Jacobian of any game decomposes into symmetric and antisym-
metric components. There are thus two ‘pure’ cases corresponding to when the Jacobian is
symmetric and anti-symmetric. The first case, known as potential games (Monderer and
Shapley, 1996), have been intensively studied in the game-theory literature because they are
exactly the games where gradient descent does converge.
    The second case, Hamiltonian2 games, were not studied previously, probably because
they coincide with zero-sum games in the bimatrix case (or constant-sum, depending on the
constraints). Zero-sum and Hamiltonian games differ when the losses are not bilinear or
when there are more than two players. Hamiltonian games are important because (i) they are
easy to solve and (ii) general games combine potential-like and Hamiltonian-like dynamics.
Unfortunately, the concept of a zero-sum game is too loose to be useful when there are many
players: any
          Pnn-player game can be reformulated as a zero-sum (n + 1)-player game where
`n+1 = − i=1 `i . In this respect, zero-sum games are as complicated as general-sum games.
In contrast, Hamiltonian games are much simpler than general-sum games. Theorem 4
shows that Hamiltonian games obey a conservation law – which also provides the key to
solving them, by gradient descent on the conserved quantity.

Algorithms. The general case, neither potential nor Hamiltonian, is more difficult and is
therefore the focus of the remainder of the paper. Section 3 proposes symplectic gradient
adjustment (SGA), a gradient-based method for finding stable fixed points in general games.
Appendix A contains TensorFlow code to compute the adjustment. The algorithm computes
two Jacobian-vector products, at a cost of two iterations of backprop. SGA satisfies a few
natural desiderata explained in section 3.1: (D1) it is compatible with the original dynamics;
and it is guaranteed to find stable equilibria in (D2) potential and (D3) Hamiltonian games.
    For general games, correctly picking the sign of the adjustment (whether to add or
subtract) is critical since it determines the behavior near stable and unstable equilibria.
Section 2.4 defines stable equilibria and contrasts them with local Nash equilibria. Theorem
10 proves that SGA converges locally to stable fixed points for sufficiently small parameters
(which we quantify via the notion of an additive condition number). While strong, this may
be impractical or slow down convergence significantly. Accordingly, lemma 11 shows how to
set the sign so as to be attracted towards stable equilibria and repelled from unstable ones.
Correctly aligning SGA allows higher learning rates and faster, more robust convergence,
see theorem 15. Finally, theorem 17 tackles the remaining class of saddle fixed points by
proving that SGA locally avoids strict saddles for appropriate parameters.

Experiments. We investigate the empirical performance of SGA in four basic experiments.
The first experiment shows how increasing alignment allows higher learning rates and faster
convergence, figure 4. The second set of experiments compares SGA with optimistic mirror
 2. Lu (1992) defined an unrelated notion of Hamiltonian game.


                                                   3
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




descent on two-player and four-player games. We find that SGA converges over a much
wider range of learning rates.
    The last two sets of experiments investigate mode collapse, mode hopping and the related,
less well-known problem of boundary distortion identified in Santurkar et al. (2018). Mode
collapse and mode hopping are investigated in a setup involving a two-dimensional mixture
of 16 Gaussians that is somewhat more challenging than the original problem introduced in
Metz et al. (2017). Whereas simultaneous gradient descent completely fails, our symplectic
adjustment leads to rapid convergence – slightly improved by correctly choosing the sign of
the adjustment.
    Finally, boundary distortion is studied using a 75-dimensional spherical Gaussian. Mode
collapse is not an issue since there the data distribution is unimodal. However, as shown in
figure 10, a vanilla GAN with RMSProp learns only one of the eigenvalues in the spectrum
of the covariance matrix, whereas SGA approximately learns all of them.
   The appendix provides some background information on differential and symplectic
geometry, which motivated the developments in the paper. The appendix also explores what
happens when the analogy with classical mechanics is pushed further than perhaps seems
reasonable. We experiment with assigning units (in the sense of masses and velocities) to
quantities in games, and find that type-consistency yields unexpected benefits.

1.3 Related work

Nash (1950) was only concerned with existence of equilibria. Convergence in two-player
games was studied in Singh et al. (2000). WoLF (Win or Learn Fast) converges to Nash
equilibria in two-player two-action games (Bowling and Veloso, 2002). Extensions include
weighted policy learning (Abdallah and Lesser, 2008) and GIGA-WoLF (Bowling, 2004).
Infinitesimal Gradient Ascent (IGA) is a gradient-based approach that is shown to converge
to pure Nash equilibria in two-player two-action games. Cyclic behaviour may occur in case
of mixed equilibria. Zinkevich (2003) generalised the algorithm to n-action games called
GIGA. Optimistic mirror descent approximately converges in two-player bilinear zero-sum
games (Daskalakis et al., 2018), a special case of Hamiltonian games. In more general
settings it converges to coarse correlated equilibria.
    Convergence has also been studied in various n-player settings, see Rosen (1965); Scutari
et al. (2010); Facchinei and Kanzow (2010); Mertikopoulos and Zhou (2016). However, the
recent success of GANs, where the players are neural networks, has focused attention on a
much larger class of nonconvex games where comparatively little is known, especially in the
n-player case. Heusel et al. (2017) propose a two-time scale methods to find Nash equilibria.
However, it likely scales badly with the number of players. Nagarajan and Kolter (2017)
prove convergence for some algorithms, but under very strong assumptions (Mescheder et al.,
2018). Consensus optimization (Mescheder et al., 2017) is closely related to our proposad
algorithm, and is extensively discussed in section 3. A variety of game-theoretically or
minimax motivated modifications to vanilla gradient descent have been investigated in the
context of GANs, see Mertikopoulos et al. (2019); Gidel et al. (2018).
    Learning with opponent-learning awareness (LOLA) infinitesimally modifies the objectives
of players to take into account their opponents’ goals (Foerster et al., 2018). However, Letcher

                                               4
                             Differentiable Game Mechanics




          A                                          B




                                        symplectic
                                         rotation

Figure 1: A minimal example of Hamiltonian mechanics. Consider a game where `1 (x, y) =
          xy, `2 (x, y) = −xy, and the dynamics are given by ξ(x, y) = (y, −x). The game
          is a special case of example 1. (A) The dynamics ξ cycle around the origin since
          they live on the level sets of the Hamiltonian H(x, y) = 12 (x2 + y 2 ). (B) Gradient
          descent on the Hamiltonian H converges to the Nash equilibrium of the game, at
          the origin (0, 0). Note that A| ξ = (x, y) = ∇H.



et al. (2019) recently showed that LOLA modifies fixed points and thus fails to find stable
equilibria in general games.
    Symplectic gradient adjustment was independently discovered by Gemp and Mahadevan
(2018), who refer to it as “crossing-the-curl”. Their analysis draws on powerful techniques
from variational inequalities and monotone optimization that are complementary to those
developed here – see for example Gemp and Mahadevan (2016, 2017); Gidel et al. (2019).
Using techniques from monotone optimization, Gemp and Mahadevan (2018) obtained more
detailed and stronger results than ours, in the more particular case of Wasserstein LQ-GANs,
where the generator is linear and the discriminator is quadratic (Feizi et al., 2017; Nagarajan
and Kolter, 2017).
    Network zero-sum games are shown to be Hamiltonian systems in Bailey and Piliouras
(2019). The implications of the existence of invariant functions for games is just beginning
to be understood and explored.
Notation. Dot products are written as v| w or hv, wi. The angle between two vectors is
θ(v, w). Positive definiteness is denoted S  0.

2. The Infinitesimal Structure of Games
In contrast to the classical formulation of games, we do not constrain the parameter sets
to the probability simplex or require losses to be convex in the corresponding players’
parameters. Our motivation is that we are primarily interested in use cases where players
are interacting neural nets such as GANs (Goodfellow et al., 2014), a situation in which
results from classical game theory do not straightforwardly apply.

Definition 1 (differentiable game)
A differentiable game consists in a set of players [n] = {1, . . . , n} and corresponding twice

                                               5
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




                                      d       n . Parameters are w = (w , . . . , w ) ∈ Rd
      Pn differentiable losses {`i : R → R}
continuously
                                          d
                                              i=1                         1        n
where i=1 di = d. Player i controls wi ∈ R i , and aims to minimize its loss.

It is sometimes convenient to write w = (wi , w−i ) where w−i concatenates the parameters
of all the players other than the ith , which is placed out of order by abuse of notation.
    The simultaneous gradient is the gradient of the losses with respect to the parameters of
the respective players:
                             ξ(w) = (∇w1 `1 , . . . , ∇wn `n ) ∈ Rd .

By the dynamics of the game, we mean following the negative of the vector field, −ξ, with
infinitesimal steps. There is no reason to expect ξ to be the gradient of a single function in
general, and therefore no reason to expect the dynamics to converge to a fixed point.

2.1 Hamiltonian Mechanics
Hamiltonian mechanics is a formalism for describing the dynamics in classical physical
systems, see Arnold (1989); Guillemin and Sternberg (1990). The system is described via
canonical coordinates (q, p). For example, q often refers to position and p to momentum of
a particle or particles.
    The Hamiltonian of the system H(q, p) is a function that specifies the total energy as a
function of the generalized coordinates. For example, in a closed system the Hamiltonian is
given by the sum of the potential and kinetic energies of the particles. The time evolution
of the system is given by Hamilton’s equations:
                               dq   ∂H            dp    ∂H
                                  =         and      =−    .
                               dt   ∂p            dt    ∂q
An importance consequence of the Hamiltonian formalism is that the dynamics of the
physical system – that is, the trajectories followed by the particles in phase space – live on
the level sets of the Hamiltonian. In other words, the total energy is conserved.

2.2 Hamiltonian Mechanics in Games
The next example illustrates the essential problem with gradients in games and the key
insight motivating our approach.

Example 1 (Conservation
                     Pn of energy in a zero-sum unconstrained bimatrix game)
Zero-sum games, where i=1 `i ≡ 0, are well-studied. The zero-sum game

                        `1 (x, y) = x| Ay   and   `2 (x, y) = −x| Ay

has a Nash equilibrium at (x, y) = (0, 0). The simultaneous gradient ξ(x, y) = (Ay, −A| x)
rotates around the Nash, see figure 1.
    The matrix A admits singular value decomposition (SVD) A = U| DV. Changing
                      1                   1
to coordinates u = D 2 Ux and v = D 2 Vy gives `1 (u, v) = u| v and `2 (u, v) = −u| v.
Introduce the Hamiltonian
                            1               1
                H(u, v) =     kuk22 + kvk22 = (x| U| DUx + y| V| DVy) .
                            2                2

                                              6
                             Differentiable Game Mechanics




Remarkably, the dynamics can be reformulated via Hamilton’s equations in the coordinates
given by the SVD of A:                                 
                                               ∂H ∂H
                                  ξ(u, v) =       ,−      .
                                               ∂v    ∂u
The vector field ξ cycles around the equilibrium because ξ conserves the Hamiltonian’s level
sets (i.e. hξ, ∇Hi = 0). However, gradient descent on the Hamiltonian converges
to the Nash equilibrium. The remainder of the paper explores the implications and
limitations of this insight.
    Papadimitriou and Piliouras (2016) recently analyzed the dynamics of Matching Pennies
(essentially, the above example) and showed that the cyclic behavior covers the entire
parameter space. The Hamiltonian reformulation directly explains the cyclic behavior via a
conservation law.

2.3 The Generalized Helmholtz Decomposition
The Jacobian of a game with dynamics ξ is the (d × d)-matrix of second-derivatives
                             d
J(w) := ∇w · ξ(w)| = ∂ξ∂w
                        α (w)
                           β
                                 , where ξ α (w) is the αth entry of the d-dimensional
                                  α,β=1
vector ξ(w). Concretely, the Jacobian can be written as
                               ∇2w1 `1   ∇2w1 ,w2 `1 · · ·      ∇2w1 ,wn `1
                                                                           
                            ∇  2          ∇2w2 `2   ···        ∇2w2 ,wn `2 
                             w2 ,w1 `2
                   J(w) = 
                                                                            
                                  ..                                 ..     
                                  .                                  .     
                                ∇2wn ,w1 `n ∇2wn ,w2 `n · · ·    ∇2wn `n

where ∇2wi ,wj `k is the (di × dj )-block of 2nd -order derivatives. The Jacobian of a game is a
square matrix, but not necessarily symmetric. Note: Greek indices α, β run over d parameter
dimensions whereas Roman indices i, j run over n players.
Lemma 1 (generalized Helmholtz decomposition)
The Jacobian of any vector field decomposes uniquely into two components J(w) = S(w) +
A(w) where S ≡ S| is symmetric and A + A| ≡ 0 is antisymmetric.
Proof Any matrix decomposes uniquely as M = S + A where S = 12 (M + M| ) and
A = 12 (M−M| ) are symmetric and antisymmetric. The decomposition is preserved by orthog-
onal change-of-coordinates: given orthogonal matrix P, we have P| MP = P| SP + P| AP
since the terms remain symmetric and antisymmetric. Applying the decomposition to the
Jacobian yields the result.


   The connection to the classical Helmholtz decomposition in calculus is sketched in
appendix B. Two natural classes of games arise from the decomposition:
Definition 2 A game is a potential game if the Jacobian is symmetric, i.e. if A(w) ≡ 0.
It is a Hamiltonian game if the Jacobian is antisymmetric, i.e. if S(w) ≡ 0.
Potential games are well-studied and easy to solve. Hamiltonian games are a new class of
games that are also easy to solve. The general case is more difficult, see section 3.

                                               7
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




2.4 Stable Fixed Points (SFPs) vs Local Nash Equilibria (LNEs)
There are (at least) two possible solution concepts in general differentiable games: stable
fixed points and local Nash equilibria.

Definition 3 A point w is a local Nash equilibrium if, for all i, there exists a neighbor-
hood Ui of wi such that `i (wi0 , w−i ) ≥ `i (wi , w−i ) for wi0 ∈ Ui .

    We introduce local Nash equilibria because finding global Nash equilibria is unrealistic in
games involving neural nets. Gradient-based methods can reliably find local – but not global
– optima of nonconvex objective functions (Lee et al., 2016, 2017). Similarly, gradient-based
methods cannot be expected to find global Nash equilibria in nonconvex games.

Definition 4 A fixed point w∗ with ξ(w∗ ) = 0 is stable if J(w∗ )  0 and J(w∗ ) is
invertible, unstable if J(w∗ ) ≺ 0 and a strict saddle if J(w∗ ) has an eigenvalue with
negative real part. Strict saddles are a subset of unstable fixed points.

     The definition is adapted from Letcher et al. (2019), where conditions on the Jacobian
hold at the fixed point; in contrast, Balduzzi et al. (2018a) imposed conditions on the
Jacobian in a neighborhood of the fixed point. We motivate this concept as follows.
     Positive semidefiniteness, J(w∗ )  0, is a minimal condition for any reasonable notion of
stable fixed point. In the case of a single loss `, the Jacobian of ξ = ∇` is the Hessian of `,
i.e. J = ∇2 `. Local convergence of gradient descent on single functions cannot be guaranteed
if J(w∗ )  0, since such points are strict saddles. These are almost always avoided by Lee
et al. (2017), so this semidefinite condition must hold.
     Another viewpoint is that invertibility and positive semidefiniteness of the Hessian
together imply positive definiteness, and the notion of stable fixed point specializes, in a
one-player game, to local minima that are detected by the second partial derivative test.
These minima are precisely those which gradient-like methods provably converge to. Stable
fixed points are defined by analogy, though note that invertibility and semidefiniteness do
not imply positive definiteness in n-player games since J may not be symmetric.
     Finally, it is important to impose only positive semi-definiteness to keep the class as
large as possible. Imposing strict positivity would imply that the origin is not an SFP in
the cyclic game `1 = xy = −`2 from Example 1, while clearly deserving of being so.

Remark 1 The conditions J(w∗ )  0 and J(w∗ ) ≺ 0 are equivalent to the conditions on
the symmetric component S(w∗ )  0 and S(w∗ ) ≺ 0 respectively, since

                               u| Ju = u| Su + u| Au = u| Su

for all u, by antisymmetry of A. This equivalence will be used throughout.

   Stable fixed points and local Nash equilibria are both appealing solution concepts, one
from the viewpoint of optimisation by analogy with single objectives, and the other from
game theory. Unfortunately, neither is a subset of the other:

                                               8
                             Differentiable Game Mechanics




Example 2 (stable 6=⇒ local Nash)
Let `1 (x, y) = x3 + xy and `2 (x, y) = −xy. Then
                                 2                             
                                  3x + y                     6x 1
                      ξ(x, y) =              and J(x, y) =          .
                                     −x                      −1 0

There is a stable fixed point with invertible Hessian at (x, y) = (0, 0), since ξ(0, 0) = 0 and
J(0, 0)  0 invertible. However any neighbourhood of x = 0 contains some small  > 0 for
which `1 (−, 0) = −3 < 0 = `1 (0, 0), so the origin is not a local Nash equilibrium.

Example 3 (local Nash 6=⇒ stable)
Let `1 (x, y) = `2 (x, y) = xy. Then
                                                                
                                       y                       0 1
                            ξ(x, y) =     and       J(x, y) =        .
                                       x                       1 0

There is a fixed point at (x, y) = (0, 0) which is a local (in fact, global) Nash equilibrium
since `1 (0, y) = 0 ≥ `1 (0, 0) and `2 (x, 0) = 0 ≥ `2 (0, 0) for all x, y ∈ R. However J = S has
eigenvalues λ1 = 1 and λ2 = −1 < 0, so (0, 0) is not a stable fixed point.

    In Example 3, the Nash equilibrium is a saddle point of the common loss ` = xy. Any
algorithm that converges to Nash equilibria will thus converge to an undesirable saddle point.
This rules out local Nash equilibrium as a solution concept for our purposes. Conversely,
Example 2 emphasises the better notion of stability whereby player 1 may have a local
incentive to deviate from the origin immediately, but would later be punished for doing so
since the game is locally dominated by the ±xy terms, whose only ‘resolution’ or ‘stable
minimum’ is the origin (see Example 1).

2.5 Potential Games
Potential games were introduced by Monderer and Shapley (1996). It turns out that our
definition of potential game above coincides with a special case of the potential games of
Monderer and Shapley (1996), which they refer to as exact potential games.

Definition 5 (classical definition of potential game)
A game is a potential game if there is a single potential function φ : Rd → R and positive
numbers {αi > 0}ni=1 such that
                                                                                     
               φ(wi0 , w−i ) − φ(wi00 , w−i ) = αi `i (wi0 , w−i ) − `i (wi00 , w−i )

for all i and all wi0 , wi00 , w−i , see Monderer and Shapley (1996).

Lemma 2 A game is a potential game iff αi ∇wi `i = ∇wi φ for all i, which is equivalent to
                                                             |
                αi ∇2wi wj `i = αj ∇2wi wj `j = αj ∇2wj wi `j    ∀i, j.                (1)

Proof See Monderer and Shapley (1996).



                                                9
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




Corollary 3 If αi = 1 for all i then equation (1) is equivalent to requiring that the Jacobian
of the game is symmetric.

Proof In an exact potential game, the Jacobian coincides with the Hessian of the potential
function φ, which is necessarily symmetric.


    Monderer and Shapley (1996) refer to the special case where αi = 1 for all i as an exact
potential game. We use the shorthand ‘potential game’ to refer to exact potential games
in what follows.
    Potential games have been extensively studied since they are one of the few classes of
games for which Nash equilibria can be computed (Rosenthal, 1973). For our purposes, they
are games where simultaneous gradient descent on the losses corresponds to gradient descent
on a single function. It follows that descent on ξ converges to a fixed point that is a local
minimum of φ or a saddle.

2.6 Hamiltonian Games
Hamiltonian games, where the Jacobian is antisymmetric, are a new class games. They are
related to the harmonic games introduced in Candogan et al. (2011), see section B.4. An
example from Balduzzi et al. (2018b) may help develop intuition for antisymmetric matrices:

Example 4 (antisymmetric structure of tournaments)
Suppose n competitors play one-on-one and that the probability of player i beating player
j is pij . Then, assuming there
                             are no draws, the probabilities satisfy pij + pji = 1 and
                                     p       n
pii = 12 . The matrix A =      log 1−pijij           of logits is then antisymmetric. Intuitively,
                                             i,j=1
antisymmetry reflects a hyperadversarial setting where all pairwise interactions between
players are zero-sum.

   Hamiltonian games are closely related to zero-sum games.

Example 5 (an unconstrained bimatrix game is zero-sum iff it is Hamiltonian)
Consider bimatrix game with `1 (x, y) = x| Py and `2 (x, y) = x| Qy, but where the parame-
ters are not constrained to the probability simplex. Then ξ = (Py, Q| x) and the Jacobian
components have block structure
                                                                         
                  1       0      P−Q                   1       0      P+Q
             A=                               and S =
                  2 (Q − P)|        0                  2 (P + Q)|        0
The game is Hamiltonian iff S = 0 iff P + Q = 0 iff `1 + `2 = 0.

   However, in general there are Hamiltonian games that are not zero-sum and vice versa.

Example 6 (Hamiltonian game that is not zero-sum)
Fix constants a and b and suppose players 1 and 2 minimize losses

                       `1 (x, y) = x(y − b) and `2 (x, y) = −(x − a)y

with respect to x and y respectively.

                                                 10
                            Differentiable Game Mechanics




Example 7 (zero-sum game that is not Hamiltonian)
Players 1 and 2 minimize

                        `1 (x, y) = x2 + y 2    `2 (x, y) = −(x2 + y 2 ).

The game actually has potential function φ(x, y) = x2 − y 2 .

Hamiltonian games are quite different from potential games. In a Hamiltonian game there is a
Hamiltonian function H that specifies a conserved quantity. In potential games the dynamics
equal ∇φ; in Hamiltonian games the dynamics are orthogonal to ∇H. The orthogonality
implies the conservation law that underlies the cyclic behavior in example 1.

Theorem 4 (conservation law for Hamiltonian games)
Let H(w) := 12 kξ(w)k22 . If the game is Hamiltonian then

  i) ∇H = A| ξ and

 ii) ξ preserves the level sets of H since hξ, ∇Hi = 0.

iii) If the Jacobian is invertible and limkwk→∞ H(w) = ∞ then gradient descent on H
     converges to a stable fixed point.

Proof Direct computation shows ∇H = J| ξ for any game. The first statement follows
since J = A in Hamiltonian games.
     For the second statement, the directional derivative is Dξ H = hξ, ∇Hi = ξ | A| ξ where
ξ A ξ = (ξ | A| ξ)| = ξ | Aξ = −(ξ | A| ξ) since A = −A| by anti-symmetry. It follows that
  | |

ξ | A| ξ = 0.
     For the third statement, gradient descent on H will converge to a point where ∇H =
  |
J ξ(w) = 0. If the Jacobian is invertible then clearly ξ(w) = 0. The fixed-point is stable
since 0 ≡ S  0 in a Hamiltonian game, recall remark 1.


    In fact, H is a Hamiltonian function for the game dynamics, see appendix B for a concise
explanation. We use the notation H(w) = 12 kξ(w)k2 throughout the paper. However, H
can only be interpreted as a Hamiltonian function for ξ when the game is Hamiltonian.
    There is a precise mapping from Hamiltonian games to symplectic geometry, see ap-
pendix B. Symplectic geometry is the modern formulation of classical mechanics (Arnold,
1989; Guillemin and Sternberg, 1990). Recall that periodic behaviors (e.g. orbits) often arise
in classical mechanics. The orbits lie on the level sets of the Hamiltonian, which expresses
the total energy of the system.

3. Algorithms
We have seen that fixed points of potential and Hamiltonian games can be found by descent
on ξ and ∇H respectively. This section tackles finding stable fixed points in general games.

                                               11
          Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




3.1 Finding Stable Fixed Points
There are two classes of games where we know how to find stable fixed points: potential
games where ξ converges to a local minimum and Hamiltonian games where ∇H, which is
orthogonal to ξ, finds stable fixed points.
   In the general case, the following desiderata provide a set of reasonable properties for an
adjustment ξ λ of the game dynamics. Recall that θ(u, v) is the angle between the vectors u
and v.
Desiderata.       To find stable fixed points, an adjustment ξ λ to the game dynamics should
satisfy

D1. compatible3 with game dynamics: hξ λ , ξi = α1 · kξk2 ;

D2. compatible with potential dynamics:
    if the game is a potential game then hξ λ , ∇φi = α2 · k∇φk2 ;

D3. compatible with Hamiltonian dynamics:
    If the game is Hamiltonian then hξ λ , ∇Hi = α3 · k∇Hk2 ;

D4. attracted to stable equilibria:
    in neighborhoods where S  0, require θ(ξ λ , ∇H) ≤ θ(ξ, ∇H);

D5. repelled by unstable equilibria:
    in neighborhoods where S ≺ 0, require θ(ξ λ , ∇H) ≥ θ(ξ, ∇H).

for some α1 , α2 , α3 > 0.

    Desideratum D1 does not guarantee that players act in their own self-interest – this
requires a stronger positivity condition on dot-products with subvectors of ξ, see Balduzzi
(2017). Desiderata D2 and D3 imply that the adjustment behaves correctly in potential and
Hamiltonian games respectively.
    To understand desiderata D4 and D5, observe that gradient descent on H = 12 kξk2
will find local minima that are fixed points of the dynamics. However, we specifically wish
to converge to stable fixed points. Desideratum D4 and D5 require that the adjustment
improves the rate of convergence to stable fixed points (by finding a steeper angle of descent),
and avoids unstable fixed points.
    More concretely, desiderata D4 can be interpreted as follows. If ξ points at a stable
equilibrium then we require that ξ λ points more towards the equilibrium (i.e. has smaller
angle). Conversely, desiderata D5 requires that if ξ points away then the adjustment should
point further away.
    The unadjusted dynamics ξ satisfies all the desiderata except D3.

3.2 Consensus Optimization
Since gradient descent on the function H(w) = 12 kξk2 finds stable fixed points in Hamiltonian
games, it is natural to ask how it performs in general games. If the Jacobian J(w) is invertible,
then ∇H = J| ξ = 0 iff ξ = 0. Thus, gradient descent on H converges to fixed points of ξ.
 3. Two nonzero vectors are compatible if they have positive inner product.


                                                   12
                             Differentiable Game Mechanics




    However, there is no guarantee that descent on H will find a stable fixed point. Mescheder
et al. (2017) propose consensus optimization, a gradient adjustment of the form

                                  ξ + λ · J| ξ = ξ + λ · ∇H.

Unfortunately, consensus optimization can converge to unstable fixed points even in simple
cases where the ‘game’ is to minimize a single function:

Example 8 (consensus optimization can converge to a global maximum)
Consider a potential game with losses `1 (x, y) = `2 (x, y) = − κ2 (x2 + y 2 ) with κ  0. Then
                                                      
                                       x             κ 0
                             ξ = −κ ·     and J = −
                                       y              0 κ

Note that kξk2 = κ2 (x2 + y 2 ) and
                                                          
                                        |                 x
                               ξ + λ · J ξ = κ(λκ − 1) ·     .
                                                          y

Descent on ξ + λ · J| ξ converges to the global maximum (x, y) = (0, 0) unless λ < κ1 .

Although consensus optimization works well in two-player zero-sum, it cannot be considered
a candidate algorithm for finding stable fixed points in general games since it fails in the
basic case of potential games. Consensus optimization only satisfies desiderata D3 and D4.

3.3 Symplectic Gradient Adjustment
The problem with consensus optimization is that it can perform worse than gradient descent
on potential games. Intuitively, it makes bad use of the symmetric component of the Jacobian.
Motivated by the analysis in section 2, we propose symplectic gradient adjustment, which
takes care to only use the antisymmetric component of the Jacobian when adjusting the
dynamics.

Proposition 5 The symplectic gradient adjustment (SGA)

                                      ξ λ := ξ + λ · A| ξ.

satisfies D1–D3 for λ > 0, with α1 = 1 = α2 and α3 = λ.

Proof First claim: λ · ξ | A| ξ = 0 by anti-symmetry of A. Second claim: A ≡ 0 in
a potential game, so ξ λ = ξ = ∇φ. Third claim: hξ λ , ∇Hi = hξ λ , J| ξi = hξ λ , A| ξi =
λ · ξ | AA| ξ = λ · k∇Hk2 since J = A by assumption.
Note that desiderata D1 and D2 are true even when λ < 0. This will prove useful, since
example 9 shows that it may be necessary to pick negative λ near S ≺ 0. Section 3.5 shows
how to also satisfy desiderata D4 and D5.

                                              13
           Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




3.4 Convergence
We begin by analysing convergence of SGA near stable equilibria. The following lemma
highlights that the interaction between the symmetric and antisymmetric components
is important for convergence. Recall that two matrices A and S commute iff [A, S] :=
AS − SA = 0. That is, A and S commute iff AS = SA. Intuitively, two matrices commute
if they have the same preferred coordinate system.

Lemma 6 If S  0 is symmetric positive semidefinite and S commutes with A then ξ λ
points towards stable fixed points for non-negative λ:

                                      hξ λ , ∇Hi ≥ 0 for all λ ≥ 0.

Proof First observe that ξ | ASξ = ξ | S| A| ξ = −ξ | SAξ, where the first equality holds
since the expression is a scalar, and the second holds since S = S| and A = −A| . It follows
that ξ | ASξ = 0 if SA = AS. Finally rewrite the inequality as

                  hξ λ , ∇Hi = hξ + λ · A| ξ, Sξ + A| ξi = ξ | Sξ + λξ | AA| ξ ≥ 0

since ξ | ASξ = 0 and by positivity of S, λ and AA| .


    The lemma suggests that in general the failure of A and S to commute should be
important for understanding the dynamics of ξ λ . We therefore introduce the additive
condition number κ to upper-bound the worst-case noncommutativity of S, which allows
to quantify the relationship between ξ λ and ∇H. If κ = 0, then S = σ · I commutes with all
matrices. The larger the additive condition number κ, the larger the potential failure of S
to commute with other matrices.

Theorem 7 Let S be a symmetric matrix with eigenvalues σmax ≥ · · · ≥ σmin . The additive
condition number4 of S is κ := σmax − σmin . If S  0 is positive semidefinite with additive
condition number κ then λ ∈ (0, κ4 ) implies

                                              hξ λ , ∇Hi ≥ 0.

If S is negative semidefinite, then λ ∈ (0, κ4 ) implies

                                             hξ −λ , ∇Hi ≤ 0.

The inequalities are strict if J is invertible.

Proof We prove the case S  0; the case S  0 is similar. Rewrite the inequality as

                        hξ + λ · A| ξ, ∇Hi = (ξ + λ · A| ξ)| · (S + A| )ξ
                                               = ξ | Sξ + λξ | ASξ + λξ | AA| ξ

 4. The condition number of a positive definite matrix is σσmax
                                                            min
                                                                .


                                                     14
                                  Differentiable Game Mechanics




Let β = kA| ξk and S̃ = S − σmin · I, where I is the identity matrix. Then

                        ξ | Sξ + λξ | ASξ + λ · β 2 ≥ ξ | S̃ξ + λξ | AS̃ξ + λ · β 2

since ξ | Sξ ≥ ξ | S̃ξ by construction and ξ | AS̃ξ = ξ | ASξ − σmin ξ | Aξ = ξ | ASξ because
ξ | Aξ = 0 by the anti-symmetry of A. It therefore suffices to show that the inequality holds
when σmin = 0 and κ = σmax .
     Since S is positive semidefinite, there exists an upper-triangular square-root matrix T
such that T| T = S and so ξ | Sξ = kTξk2 . Further,
                                                     √
                      |ξ | ASξ| ≤ kA| ξk · kT| Tξk ≤ σmax · kA| ξk · kTξk.
                 √
since kTk2 =         σmax . Putting the observations together obtains

          kTξk2 + λ(kAξk2 − hAξ, Sξi) ≥ kTξk2 + λ(kAξk2 − kAξk kSξk
                                               ≥ kTξk2 + λkAξk(kAξk − kSξk)
                                                                      √
                                               ≥ kTξk2 + λkAξk(kAξk − σmax kTξk)
          √               √
Set α =       λ and η =       σmax . We can continue the above computation

  kTξk2 + λ(kAξk2 − hAξ, Sξi) ≥ kTξk2 + α2 kAξk(kAξk − ηkTξk)
                                       = kTξk2 + α2 kAξk2 − α2 kAξkηkTξk
                                       = (kTξk − αkAξk)2 + 2αkAξk kTξk − α2 ηkAξk kTξk
                                       = (kTξk − αkAξk)2 + kAξk kTξk(2α − α2 η)

Finally, 2α − α2 η > 0 for any α in the range (0, η2 ), which is to say, for any 0 < λ < σmax
                                                                                          4
                                                                                              .
The kernel of S and the kernel of T coincide. If ξ is in the kernel of A, resp. T, it cannot
be in the kernel of T, resp. A and the term (kTξk − αkAξk)2 is positive. Otherwise, the
term kAξkkTξk is positive.


    The theorem above guarantees that SGA always points in the direction of stable fixed
points for λ sufficiently small. This does not technically guarantee convergence; we use
Ostrowski’s theorem to strengthen this formally. Applying Ostrowski’s theorem will require
taking a more abstract perspective by encoding the adjusted dynamics into a differentiable
map F : Ω → Rd of the form F (w) = w − αξ λ (w).

Theorem 8 (Ostrowski) Let F : Ω → Rd be a continuously differentiable map on an open
subset Ω ⊆ Rd , and assume w∗ ∈ Ω is a fixed point. If all eigenvalues of ∇F (w∗ ) are strictly
in the unit circle of C, then there is an open neighbourhood U of w∗ such that for all w0 ∈ U ,
the sequence F k (w0 ) of iterates of F converges to w∗ . Moreover, the rate of convergence is
at least linear in k.

Proof This is a standard result on fixed-point iterations, adapted from Ortega and Rhein-
boldt (2000, 10.1.3).



                                                    15
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




Corollary 9 A matrix M is called positive stable if all its eigenvalues have positive real
part. Assume w∗ is a fixed point of a differentiable game such that (I + λA| )J(w∗ ) is
positive stable for λ in some set Λ. Then SGA converges locally to w∗ for λ ∈ Λ and α > 0
sufficiently small.

Proof Let X = (I + λA| ). By definition of fixed points, ξ(w∗ ) = 0 and so

                 ∇[Xξ](w∗ ) = ∇X(w∗ )ξ(w∗ ) + X(w∗ )∇ξ(w∗ ) = XJ(w∗ )

is positive stable by assumption, namely has eigenvalues ak + ibk with ak > 0. Writing
F (w) = w − αXξ(w) for the iterative procedure given by SGA, it follows that

                                 ∇F (w∗ ) = I − α∇[Xξ](w∗ )

has eigenvalues 1 − αak − iαbk , which are in the unit circle for small α. More precisely,

                |1 − αak − iαbk |2 < 1    ⇐⇒        1 − 2αak + α2 a2k + α2 b2k < 1
                                                              2ak
                                          ⇐⇒        0<α< 2
                                                            ak + b2k

which is always possible for ak > 0. Hence ∇F (w∗ ) has eigenvalues in the unit circle for
0 < α < mink 2ak /(a2k + b2k ), and we are done by Ostrowski’s Theorem since w∗ is a fixed
point of F .



Theorem 10 Let w∗ be a stable fixed point and κ the additive condition number of S(w∗ ).
Then SGA converges locally to w∗ for all λ ∈ (0, κ4 ) and α > 0 sufficiently small.

Proof By Theorem 5 and the assumption that w∗ is a stable fixed point with invertible
Jacobian, we know that
                       hξ λ , ∇Hi = h(I + λA| )ξ, J| ξi > 0
for λ ∈ (0, κ4 ). The proof does not rely on any particular property of ξ, and can trivially be
extended to the claim that
                                     h(I + λA| )u, J| ui > 0
for all non-zero vectors u. In particular this can be rewritten as

                                     u| J(I + λA| )ui > 0 ,

which implies positive definiteness of J(I + λA| ). A positive definite matrix is positive stable,
and any matrices AB and BA have identical spectrum. This implies also that (I + λA| )J is
positive stable, and we are done by the corollary above.


   We conclude that SGA converges to an SFP if λ is small enough, where ‘small enough’
depends on the additive condition number.

                                               16
                             Differentiable Game Mechanics




3.5 Picking sign(λ)
This section explains desiderata D4–D5 and shows how to pick sign(λ) to speed up conver-
gence towards stable and away from unstable fixed points. In the example below, almost
any choice of positive λ results in convergence to an unstable equilibrium. The problem
arises from the combination of a weak repellor with a strong rotational force.

Example 9 (failure case for λ > 0)
Suppose  > 0 is small and
                                                          
                    `1 (x, y) = − x2 − xy and `2 (x, y) = − y 2 + xy
                                 2                         2
with an unstable equilibrium at (0, 0). The dynamics are
                                                          
                               −x        −y               0 −1
                       ξ =·         +          with A =
                               −y         x               1 0
and                                             
                                      |    x      −y
                                     A ξ=     +
                                           y      x
Finally observe that
                                                               
                                 |              x                −y
                       ξ + λ · A ξ = (λ − ) ·     + (1 + λ) ·
                                                y                x
which converges to the unstable equilibrium if λ > .

    We now show how to pick the sign of λ to avoid unstable equilibria. First, observe that
hξ, ∇Hi = ξ | (S + A)| ξ = ξ | Sξ. It follows that for ξ 6= 0:
                               (
                                 if S  0 then hξ, ∇Hi ≥ 0;
                                                                                        (2)
                                 if S ≺ 0 then hξ, ∇Hi < 0.

A criterion to probe the positive/negative definiteness of S is thus to check the sign of hξ, ∇Hi.
The dot product can take any value if S is neither positive nor negative (semi-)definite. The
behavior near saddle points will be explored in Section 3.7.
    Recall that desiderata D4 requires that, if ξ points at a stable equilibrium then we
require that ξ λ points more towards the equilibrium (i.e. has smaller angle). Conversely,
desiderata D5 requires that, if ξ points away then the adjustment should point further away.
More formally,

Definition 6 Let u and v be two vectors. The infinitesimal alignment of ξ λ := u + λ · v
with a third vector w is
                                       d  2
                    align(ξ λ , w) :=     cos θλ |λ=0 for θλ := θ(ξ λ , w).
                                      dλ
If u and w point the same way, u| w > 0, then align > 0 when v bends u further toward w,
see figure 2A. Otherwise align > 0 when v bends u away from w, see figure 2B.
    The following lemma allows us to rewrite the infinitesimal alignment in terms of known
(computable) quantities, from which we can deduce the correct choice of λ.

                                               17
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




 A                  u                       B                            u
                        v’                                          v’
                   v                                                     v
               w                                         w




Figure 2: Infinitesimal alignment between u + λv and w is positive (cyan) when small
          positive λ either: (A) pulls u toward w, if w and u have angle < 90◦ ; or (B)
          pushes u away from w if their angle is > 90◦ . Conversely, the infinitesimal
          alignment is negative (red) when small positive λ either: (A) pushes u away from
          w when their angle is acute or (B) pulls u toward w when their angle is obtuse.


Algorithm 1 Symplectic Gradient Adjustment
                         n                       n
        losses L = {`i }i=1 , weights W = {wi }i=1
  Input:
  ξ ← gradient(`i , wi ) for (`i , wi ) ∈ (L, W)
  A| ξ ← get sym adj(L, W)              // appendix A
  if align then
                      1      2 , w) for w ∈ W)
                                               
     ∇H ← gradient(
                     2  kξk             
    λ ← sign d1 hξ, ∇HihA| ξ, ∇Hi +             1
                                          //  = 10
  else
    λ←1
  end if
  Output: ξ + λ · A| ξ         // plug into any optimizer


Lemma 11 When ξ λ is the symplectic gradient adjustment,
                                                          
             sign align(ξ λ , ∇H) = sign hξ, ∇Hi · hA| ξ, ∇Hi .

Proof Observe that
                                 2
                                      hξ, ∇Hi + 2λhξ, ∇HihA| ξ, ∇Hi + O(λ2 )
                
          2         hξ λ , ∇Hi
       cos θλ =                     =                       
                   kξ λ k · k∇Hk               kξk2 + O(λ2 ) · k∇Hk2

where the denominator has no linear term in λ because ξ ⊥ A| ξ. It follows that the sign of
the infinitesimal alignment is
                                      
                               d    2
                                              n                   o
                       sign      cos θλ = sign hξ, ∇HihA| ξ, ∇Hi
                              dλ

as required.

                                            18
                            Differentiable Game Mechanics




     A                                             B



                    w                                                w




Figure 3: Alignment and learning rates. The larger cos θ, the larger the learning rate η that
          can be applied to unit vector ξ without w + η · ξ leaving the unit circle.




    Intuitively, computing the sign of hξ, ∇Hi provides a check for stable and unstable
fixed points. Computing the sign of hA| ξ, ∇Hi checks whether the adjustment term points
towards or away from the nearby fixed point. Putting the two checks together yields a
prescription for the sign of λ, as follows.

Proposition 12 Desiderata D4–D5 are satisfied for λ such that λ·hξ, ∇Hi·hA| ξ, ∇Hi ≥ 0.

Proof If we are in a neighborhood
                                    of a stable 
                                                  fixed point then
                                                                  hξ, ∇Hi ≥ 0. It follows
                                                      |
by lemma 11 that sign align(ξ λ ), ∇H) = sign hA ξ, ∇Hi and so choosing sign(λ) =
              
sign hA| ξ, ∇Hi leads to the angle between ξ λ and ∇H being smaller than the angle
between ξ and ∇H, satisfying desideratum D4. The proof for the unstable case is similar.


Alignment and convergence rates. Gradient descent is also known as the method
of steepest descent. In general games, however, ξ does not follow the steepest path to
fixed points due to the ‘rotational force’, which forces lower learning rates and slows down
convergence.
    The following lemma provides some intuition about alignment. The idea is that, the
smaller the cosine between the ‘correct direction’ w and the ‘update direction’ ξ, the smaller
the learning rate needs to be for the update to stay in a unit ball, see figure 3.

Lemma 13 (alignment lemma)
If w and ξ are unit vectors with 0 < w| ξ then kw−η·ξk ≤ 1 for 0 ≤ η ≤ 2w| ξ = 2 cos θ(w, ξ).
In other words, ensuring that w − ηξ is closer to the origin than w requires smaller learning
rates η as the angle between w and ξ gets larger.

Proof Check kw − η · ξk2 = 1 + η 2 − 2η · w| ξ ≤ 1 iff η 2 ≤ 2η · w| ξ. The result follows.


   The next lemma is a standard technical result from the convex optimization literature.

                                             19
          Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




Lemma 14 Let f : Rd → R be a convex Lipschitz smooth function satisfying k∇f (y) −
∇f (x)k ≤ L · ky − xk for all x, y ∈ Rd . Then

                                                                L
                      |f (y) − f (x) − h∇f (x), y − xi| ≤         · ky − xk2
                                                                2
for all x, y ∈ Rd .

Proof See Nesterov (2004).


   Finally, we show that increasing alignment helps speed convergence:

Theorem 15 Suppose f is convex and Lipschitz smooth with k∇f (x)−∇f (y)k ≤ L·kx−yk.
Let wt+1 = wt − η · v where kvk = k∇f (wt )k. Then the optimal step size is η ∗ = cos θ
                                                                                   L where
θ := θ(∇f (wt ), v), with

                                                    cos2 θ
                            f (wt+1 ) ≤ f (wt ) −          · k∇f (wt )k2 .
                                                     2L
    The proof of Theorem 15 adapts lemma 14 to handle the angle arising from the ‘rotational
force’.
Proof By the lemma 14,
                                                                L
                          f (y) ≤ f (x) + h∇f (x), y − xi +       ky − xk2
                                                                2
                                                          L
                               = f (x) − η · h∇f, ξi + η 2  · kξk2
                                                          2
                                                          L
                               = f (x) − η · h∇f, ξi + η 2 · k∇f k2
                                                          2
                                                η
                               = f (x) − η(α − L) · k∇f k2
                                                2
where α := cos θ. Solve                       n      η o
                                min ∆(η) = min −η(α − L)
                                 η          η        2
                                     2
to obtain η ∗ = L
                α
                  and ∆(η ∗ ) = − α2 L as required.


    Increasing the cosine with the steepest direction improves convergence. The alignment
computation in algorithm 1 chooses λ to be positive or negative such that ξ λ is bent towards
stable (increasing the cosine) and away from unstable fixed points. Adding a small  > 0 to
the computation introduces a weak bias towards stable fixed points.

3.6 Aligned Consensus Optimization
The stability criterion in (2) also provides a simple way to prevent consensus optimization
from converging to unstable equilibria. Aligned consensus optimization is
                                                      
                                 ξ + |λ| · sign hξ, ∇Hi · J| ξ,                         (3)

                                                20
                                        Differentiable Game Mechanics




                       learning rate   0.01                  0.032                   0.1




    GRADIENT DESCENT




    SGA




Figure 4: SGA allows faster and more robust convergence to stable fixed points than vanilla
          gradient descent in the presence of ‘rotational forces’, by bending the direction
          of descent towards the fixed point. Note the gradient descent diverges extremely
          rapidly in the top-right panel, which has a different scale from the other panels.



where in practice we set λ = 1. Aligned consensus optimization satisfies desiderata D3–D5.
However, it behaves strangely in potential games. Multiplying by the Jacobian is the
‘inverse’ of Newton’s method since for potential games the Jacobian of ξ is the Hessian of
the potential function. Multiplying by the Hessian increases the gap between small and
large eigenvalues, increasing the (usual, multiplicative) condition number and slows down
convergence. Nevertheless, consensus optimization works well in GANs (Mescheder et al.,
2017), and aligned consensus may improve performance, see experiments below.
    Dropping the first term ξ from (3) yields a simpler update that also satisfies D3–D5.
However, the resulting algorithm performs poorly in experiments (not shown), perhaps
because it is attracted to saddles.

3.7 Avoiding Strict Saddles
How does SGA behave near saddles? We show that Symplectic Gradient Adjustment locally
avoids strict saddles, provided that λ and α are small and parameters are initialized with
(arbitrarily small) noise. More precisely, let F(w) = w−αξ λ (w) be the iterative optimization
procedure given by SGA. Then every strict saddle w∗ has a neighbourhood U such that
{w ∈ U | Fn (w) → w∗ as n → ∞} has measure zero for small α > 0 and λ.
    Intuitively, the Taylor expansion around a strict saddle w∗ is locally dominated by
the Jacobian at w∗ , which has a negative eigenvalue. This prevents convergence to w∗

                                                     21
                         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




                                  OMD                                                OMD




     STEPS TO CONVERGE                                        LOSS AFTER 250 STEPS
                                  SGA                                                SGA




                                  LEARNING RATE                                        LEARNING RATE


Figure 5: Comparison of SGA with optimistic mirror descent. The plots sweep over learning
          rates in range [0.01, 1.75], with λ = 1 throughout for SGA. (Left): iterations to
          convergence, with maximum value of 250 after which the run was interrupted.
          (Right): average absolute value of losses over the last 10 iterations, 240-250, with
          a cutoff at 5.



for random initializations of w near w∗ . The argument is made rigorous using the Stable
Manifold Theorem following Lee et al. (2017).

Theorem 16 (Stable Manifold Theorem)
Let w∗ be a fixed point for the C 1 local diffeomorphism F : U → Rd , where U is a neighbour-
hood of w∗ in Rd . Let E s ⊕ E u be the generalized eigenspaces of ∇F (w∗ ) corresponding to
eigenvalues with |σ| ≤ 1 and |σ| > 1 respectively. Then there exists a local stable center man-
ifold W with tangent space E s at w∗ and a neighbourhood B of w∗ such that F (W ) ∩ B ⊂ W
and ∩∞n=0 F
            −n (B) ⊂ W .


Proof See Shub (2000).


    It follows that if ∇F (w∗ ) has at least one eigenvalue |σ| > 1 then E u has dimension at
least 1. Since W has tangent space E s at w∗ with codimension at least one, we conclude
that W has measure zero. This is central to proving that the set of nearby initial points
which converge to a given strict saddle w∗ has measure zero. Since w is initialized randomly,
the following theorem is obtained.

Theorem 17 SGA locally avoids strict saddles almost surely, for α > 0 and λ small.

Proof Let w∗ a strict saddle and recall that SGA is given by

                                             F (w) = w − α(I − αJ)ξ(w) .

All terms involved are continuously differentiable and we have

                                            ∇F (w∗ ) = I − α(I − αJ)J(w∗ )

                                                         22
                             Differentiable Game Mechanics




by assumption that ξ(w∗ ) = 0. Since all terms except I are of order at least α, ∇F (w∗ )
is invertible for all α sufficiently small. By the inverse function theorem, there exists a
neighbourhood U of w∗ such that F is has a continuously differentiable inverse on U . Hence
F restricted to U is a C 1 diffeomorphism with fixed point w∗ .
    By definition of strict saddles, J(w∗ ) has an eigenvalue with negative real part. It follows
by continuity that (I − αJ)J(w∗ ) also has an eigenvalue a + ib with a < 0 for α sufficiently
small. Finally,
                                ∇F (w∗ ) = I − α(I − αJ)J(w∗ )
has an eigenvalue σ = 1 − αa − iαb with

                         |σ| = 1 − 2αa + α2 (a2 + b2 ) ≥ 1 − 2αa > 1 .

It follows that E s has codimension at least one, implying in turn that the local stable set W
has measure zero. We can now prove that

                              Z = {w ∈ U | lim F n (w) = w∗ }
                                             n→∞

has measure zero, or in other words, that local convergence to w∗ occurs with zero probability.
Let B the neighbourhood guaranteed by the Stable Manifold Theorem, and take any w ∈ Z.
By definition of convergence there exists N ∈ N such that F N +n (w) ∈ B for all n ∈ N, so
that
                                F N (w) ∈ ∩∞ n∈N F
                                                   −n
                                                      (B) ⊂ W
by the Stable Manifold Theorem. This implies that w ∈ F −N (W ), and by extension
w ∈ ∪n∈N F −n (W ). Since w was arbitrary, we obtain the inclusion

                                     Z ⊆ ∪n∈N F −n (W ) .

Now F −1 is C 1 , hence locally Lipschitz and thus preserves sets of measure zero, so that
F −n (W ) has measure zero for each n. Countable unions of measure zero sets are still
measure zero, so we conclude that Z also has measure zero. In other words, SGA converges
to w∗ with zero probability upon random initialization of w in U .


   Unlike stable and unstable fixed points, it is unclear how to avoid strict saddles using
only alignment, that is, independently from the size of λ.

4. Experiments
We compare SGA with simultaneous gradient descent, optimistic mirror descent (Daskalakis
et al., 2018) and consensus optimization (Mescheder et al., 2017) in basic settings.

4.1 Learning rates and alignment
We investigate the effect of SGA when a weak attractor is coupled to a strong rotational
force:
                              1                           1
                   `1 (x, y) = x2 + 10xy and `2 (x, y) = y 2 − 10xy
                              2                           2

                                               23
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




                                      SYMPLECTIC GRADIENT ADJUSTMENT
                                             learning rate 1.0




                                      OPTIMISTIC MIRROR DESCENT
                                              learning rate 1.0




                                      SYMPLECTIC GRADIENT ADJUSTMENT
                                             learning rate 0.4




                                      OPTIMISTIC MIRROR DESCENT
                                              learning rate 0.4




                                      SYMPLECTIC GRADIENT ADJUSTMENT
                                             learning rate 1.16




                                      OPTIMISTIC MIRROR DESCENT
                                              learning rate 1.16




          Figure 6: Individual runs on zero-sum bimatrix game in section 4.2.


Gradient descent is extremely sensitive to the choice of learning rate η, top row of figure 4. As
η increases through {0.01, 0.032, 0.1} gradient descent goes from converging extremely slowly,
to diverging slowly, to diverging rapidly. SGA yields faster, more robust convergence. SGA

                                                    24
                            Differentiable Game Mechanics




converges faster with learning rates η = 0.01 and η = 0.032, and only starts overshooting
the fixed point for η = 0.1.

4.2 Basic adversarial games
Optimistic mirror descent is a family of algorithms that has nice convergence properties in
games (Rakhlin and Sridharan, 2013; Syrgkanis et al., 2015). In the special case of optimistic
gradient descent the updates are
                            wt+1 ← wt − η · ξ t − η · (ξ t − ξ t−1 ).
Figure 5 compares SGA with optimistic gradient descent (OMD) on a zero-sum bimatrix
game with `1/2 (w1 , w2 ) = ±w1| w2 . The example is modified from Daskalakis et al. (2018)
who also consider a linear offset that makes no difference. A run is taken to have converged
if the average absolute value of losses on the last 10 iterations is < 0.01; we end each
experiment after 250 steps.
    The left panel shows the number of steps to convergence (when convergence occurs) over
a range of learning rates. OMD’s peak performance is better than SGA, where the red curve
dips below the blue. Howwever, we find that SGA converges – and does so faster – for a
much wider range of learning rates. OMD diverges for learning rates not in the range [0.3,
1.2]. Simultaneous gradient descent oscillates without converging (not shown). The right
panel shows the average performance of OMD and SGA on the last 10 steps. Once again,
here SGA consistently performs better over a wider range of learning rates. Individual runs
are shown in figure 6.
OMD and SGA on a four-player game. Figure 7 shows time to convergence (using
the same convergence criterion as above) for optimistic mirror descent and SGA. The games
are constructed with four players, each of which controls one parameter. The losses are
                                            
                          `1 (w, x, y, z) = w2 + wx + wy + wz
                                            2
                                                  
                          `2 (w, x, y, z) = −wx + x2 + xy + xz
                                                  2
                                                        
                           `3 (w, x, y, z) = −wy − xy + y 2 + yz
                                                        2
                                                              
                           `4 (w, x, y, z) = −wz − xz − yz + z 2 ,
                                                              2
           1
where  = 100 in the left panel and  = 0 in the right panel. The antisymmetric component
of the game Jacobian is                                  
                                            0  1    1 1
                                         −1 0      1 1
                                  A=    −1 −1 0 1
                                                          

                                           −1 −1 −1 0
and the symmetric component is
                                                         
                                         1       0   0   0
                                       0        1   0   0
                                   S=·
                                       0
                                                          .
                                                 0   1   0
                                         0       0   0   1

                                               25
                         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




                                              OMD                                                OMD




     STEPS TO CONVERGE                                       STEPS TO CONVERGE
                                              SGA                                                SGA

                                    WEAK                                                NO
                                  ATTRACTOR                                          ATTRACTOR




                                  LEARNING RATE                                  LEARNING RATE


Figure 7: Time to convergence of OMD and SGA on two 4-player games. Times are cutoff
                                                                                  1
          after 5000 iterations. Left panel: Weakly positive definite S with  = 100 . Right
          panel: Symmetric component is identically zero.




OMD converges considerably slower than SGA across the full range of learning rates. It also
diverges for learning rates > 0.22. In contrast, SGA converges more quickly and robustly.


4.3 Learning a two-dimensional mixture of Gaussians

We apply SGA to a basic Generative Adversarial Network setup adapted from Metz et al.
(2017). Data is sampled from a highly multimodal distribution designed to probe the
tendency of GANs to collapse onto a subset of modes during training. The distribution is a
mixture of 16 Gaussians arranged in a 4 × 4 grid. Figure 8 shows the probability distribution
that is sampled to train the generator and discriminator. The generator and discriminator
networks both have 6 ReLU layers of 384 neurons. The generator has two output neurons;
the discriminator has one.

    Figure 9 shows results after {2000, 4000, 6000, 8000} iterations. The networks are trained
under RMSProp. Learning rates were chosen by visual inspection of grid search results at
iteration 8000. More precisely, grid search was over learning rates {1e-5, 2e-5,5e-5, 8e-5, 1e-4,
2e-4, 5e-4} and then a more refined linear search over [8e-5, 2e-4]. Simultaneous gradient
descent and SGA are shown in the figure.

    The last two rows of figure 9 show the performance of consensus optimization without
and with alignment. Introducing alignment slightly improves speed of convergence (second
column) and final result (fourth column), although intermediate results in third column are
ambiguous.

    Simultaneous gradient descent exhibits mode collapse followed by mode hopping in later
iterations (not shown). Mode hopping is analogous to the cycles in example 1. Unaligned SGA
converges to the correct distribution; alignment speeds up convergence slightly. Consensus
optimization performs similarly in this GAN example. However, consensus optimization can
converge to local maxima even in potential games, recall example 8.

                                                        26
                            Differentiable Game Mechanics




Figure 8: Ground truth for GAN experiments on a two-dimensional mixture of 16 Gaussians.



4.4 Learning a high-dimensional unimodal Gaussian
Mode collapse is a well-known phenomenon in GANs. A more subtle phenomenon, termed
boundary distortion, was identified in Santurkar et al. (2018). Boundary distortion is a form
of covariate shift where the generator fails to model the true data distribution.
    Santurkar et al demonstrate boundary distortion using data sampled from a 75-dimensional
unimodal Gaussian with spherical covariate matrix. Mode collapse is not a problem in this
setting because the data distribution is unimodal. Nevertheless, they show that vanilla
GANs fail to learn most of the spectrum of the covariate matrix.
    Figure 10 reproduces their result. Panel A shows the ground truth: all 75 eigenvalues
are equal to 1.0. Panel B shows the spectrum of the covariance matrix of the data generated
by a GAN trained with RMSProp. The GAN concentrates on a single eigenvalue and
essentially ignores the remaining 74 eigenvalues. This is similar to, but more extreme than,
the empirical results obtained in Santurkar et al. (2018). We emphasize that the problem is
not mode collapse, since the data is unimodal (although, it’s worth noting that most of the
mass of a high-dimensional Gaussian lies on the “shell”).
    Finally, panel C shows the spectrum of the covariance matrix of the data sampled from
a GAN trained via SGA. The GAN approximately learns all the eigenvalues, with values
ranging between 0.6 and 1.5.

5. Discussion
Modern deep learning treats differentiable modules like plug-and-play lego blocks. For this
to work, at the very least, we need to know that gradient descent will find local minima.
Unfortunately, gradient descent does not necessarily find local minima when optimizing
multiple interacting objectives. With the recent proliferation of algorithms that optimize
more than one loss, it is becoming increasingly urgent to understand and control the dynamics
of interacting losses. Although there is interesting recent work on two-player adversarial
games such as GANs, there is essentially no work on finding stable fixed points in more
general games played by interacting neural nets.
    The generalized Helmholtz decomposition provides a powerful new perspective on game
dynamics. A key feature is that the analysis is indifferent to the number of players. Instead,

                                             27
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




          GRADIENT DESCENT




                               learning rate 1e-4


          SGA without ALIGNMENT




                               learning rate 9e-5


          SGA with ALIGNMENT




                               learning rate 9e-5


          CONSENSUS OPTIMIZATION




                               learning rate 9e-5


          CONSENSUS OPTIMIZATION with ALIGNMENT




                               learning rate 9.25e-5




         Iteration:    2000                            4000        6000      8000


Figure 9: First row: Simultaneous gradient descent suffers from mode collapse and in later
          iterations (not shown) mode hopping. Second and third rows: vanilla SGA
          converges smoothly to the ground truth (figure 8). SGA with alignment converges
          slightly faster. Fourth and fifth rows: Consensus optimization without and
          with alignment.



it is the interplay between the simultaneous gradient ξ on the losses and the symmetric and
antisymmetric matrices of second-order terms that guides algorithm design and governs the
dynamics under gradient adjustments.

                                                              28
                                   Differentiable Game Mechanics




      A       Ground truth             B          RMSProp          C            SGA




Eigenvalue



             Eigenvalue number               Eigenvalue number           Eigenvalue number



Figure 10: Panel A: The ground truth is a 75 dimensional spherical Gaussian whose
           covariance matrix has all eigenvalues equal to 1.0. Panel B: A vanilla GAN
           trained with RMSProp approximately learns the first eigenvalue, but essentially
           ignores all the rest. Panel C: Applying SGA results in the GAN approximately
           learning all 75 eigenvalues, although the range varies from 0.6 to 1.5.



    Symplectic gradient adjustment is a straightforward application of the generalized
Helmholtz decomposition. It is unlikely that SGA is the best approach to finding stable fixed
points. A deeper understanding of the interaction between the potential and Hamiltonian
components will lead to more effective algorithms. Reinforcement learning algorithms that
optimize multiple objectives are increasingly common, and second-order terms are difficult
to estimate in practice. Thus, first-order methods that do not use Jacobian-vector products
are of particular interest.

Gamification. Finally, it is worth raising a philosophical point. In this paper we are
concerned with finding stable fixed points (because, for example, they yield pleasing samples
in GANs). We are not concerned with the losses of the players per se. The gradient
adjustments may lead to a player acting against its own self-interest by increasing its loss.
We consider this acceptable insofar as it encourages convergence to a stable fixed point. The
players are but a means to an end.
   We have argued that stable fixed points are a more useful solution concept than local
Nash equilibria for our purposes. However, neither is entirely satisfactory, and the question
“What is the right solution concept for neural games?” remains open. In fact, it likely has
many answers. The intrinsic curiosity module introduced by Pathak et al. (2017) plays two
objectives against one another to drive agents to search for novel experiences. In this case,
converging to a fixed point is precisely what is to be avoided.
    It is remarkable – to give a few examples sampled from many – that curiosity, generating
photorealistic images, and image-to-image translation (Zhu et al., 2017) can be formulated
as games. What else can games do?

Acknowledgements.                We thank Guillaume Desjardins and Csaba Szepesvari for useful
comments.

                                                  29
        Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




References
Sherief Abdallah and Victor R. Lesser. A multiagent reinforcement learning algorithm with
  non-linear dynamics. J. Artif. Intell. Res., 33:521–549, 2008.

Vladimir Arnold. Mathematical Methods of Classical Mechanics. Springer, 1989.

James Bailey and Georgios Piliouras. Multiagent Learning in Network Zero-Sum Games is a
  Hamiltonian System. In AAMAS, 2019.

D Balduzzi. Strongly-Typed Agents are Guaranteed to Interact Safely. In ICML, 2017.

D Balduzzi, S Racanière, J Martens, J Foerster, K Tuyls, and T Graepel. The mechanics of
  n-player differentiable games. In ICML, 2018a.

David Balduzzi, Karl Tuyls, Julien Perolat, and Thore Graepel. Re-evaluating Evaluation.
 In NeurIPS, 2018b.

Raoul Bott and Loring Tu. Differential Forms in Algebraic Topology. Springer, 1995.

Francesco Bottacin. A Marsden-Weinstein Reduction Theorem for Presymplectic Manifolds.
  2005.

Michael Bowling and Manuela Veloso. Multiagent learning using a variable learning rate.
 Artificial Intelligence, 136:215–250, 2002.

Michael H. Bowling. Convergence and no-regret in multiagent learning. In NeurIPS, pages
 209–216, 2004.

Yuri Burda, Harri Edwards, Deepak Pathak, Amos Storkey, Trevor Darrell, and Alexei A.
  Efros. Large-Scale Study of Curiosity-Driven Learning. In ICLR, 2019.

Ozan Candogan, Ishai Menache, Asuman Ozdaglar, and Pablo A Parrilo. Flows and
 decompositions of games: Harmonic and potential games. Mathematics of Operations
 Research, 36(3):474–503, 2011.

C Daskalakis, P W Goldberg, and C Papadimitriou. The Complexity of Computing a Nash
  Equilibrium. SIAM J. Computing, 39(1):195–259, 2009.

C Daskalakis, A Ilyas, V Syrgkanis, and H Zeng. Training GANs with Optimism. In ICLR,
  2018.

Francisco Facchinei and Christian Kanzow. Generalized Nash Equilibrium Problems. Annals
  of Operations Research, 175(1):177–211, 2010.

S Feizi, C Suh, F Xia, and D Tse.        Understanding GANs: the LQG setting.         In
  arXiv:1710.10793, 2017.

J N Foerster, R Y Chen, M Al-Shedivat, S Whiteson, P Abbeel, and I Mordatch. Learning
  with Opponent-Learning Awareness. In AAMAS, 2018.

                                           30
                           Differentiable Game Mechanics




Ian Gemp and Sridhar Mahadevan. Online Monotone Optimization. In arXiv:1608.07888,
  2016.

Ian Gemp and Sridhar Mahadevan. Online Monotone Games. In arXiv:1710.07328, 2017.

Ian Gemp and Sridhar Mahadevan. Global Convergence to the Equilibrium of GANs using
  Variational Inequalities. In arXiv:1808.01531, 2018.

Gauthier Gidel, Reyhane Askari Hemmat, Mohammad Pezeshki, Remi Lepriol, Gabriel
 Huang, Simon Lacoste-Julien, and Ioannis Mitliagkas. Negative Momentum for Improved
 Game Dynamics. In arXiv:1807.04740, 2018.

Gauthier Gidel, Hugo Berard, Gaëtan Vignoud, Pascal Vincent, and Simon Lacoste-Julien.
 A Variational Inequality Perspective on Generative Adversarial Networks. In ICLR, 2019.

I J Goodfellow, J Pouget-Abadie, M Mirza, B Xu, D Warde-Farley, S Ozair, A Courville,
   and Y Bengio. Generative Adversarial Nets. In NeurIPS, 2014.

Victor Guillemin and Shlomo Sternberg. Symplectic Techniques in Physics. Cambridge
  University Press, 1990.

Sergiu Hart and Andreu Mas-Colell. Simple Adaptive Strategies: From Regret-Matching to
  Uncoupled Dynamics. World Scientific, 2013.

M Heusel, H Ramsauer, T Unterthiner, B Nessler, G Klambauer, and S Hochreiter. GANs
 Trained by a Two Time-Scale Update Rule Converge to a Nash Equilibrium. In NeurIPS,
 2017.

M Jaderberg, W M Czarnecki, S Osindero, O Vinyals, A Graves, and K Kavukcuoglu.
 Decoupled Neural Interfaces using Synthetic Gradients. In ICML, 2017.

Xiaoye Jiang, Lek-Heng Lim, Yuan Yao, and Yinyu Ye. Statistical ranking and combinatorial
  Hodge theory. Math. Program., Ser. B, 127:203–244, 2011.

Jason D Lee, Max Simchowitz, Michael I Jordan, and Benjamin Recht. Gradient Descent
  Converges to Minimizers. In COLT, 2016.

JD Lee, I Panageas, G Piliouras, M Simchowitz, MI Jordan, and B Recht. First-order
  Methods Almost Always Avoid Saddle Points. In arXiv:1710.07406, 2017.

Alistair Letcher, Jakob Foerster, David Balduzzi, Tim Rocktäschel, and Shimon Whiteson.
  Stable Opponent Shaping in Differentiable Games. In ICLR, 2019.

Bo Liu, Ji Liu, Mohammad Ghavamzadeh, Sridhar Mahadevan, and Marek Petrik. Proximal
  Gradient Temporal Difference Learning Algorithms. In IJCAI, 2016.

Xiaoyun Lu. Hamiltonian games. Journal of Combinatorial Theory, Series B, 55:18–32,
  1992.

Panayotis Mertikopoulos and Zhengyuan Zhou. Learning in games with continuous action
  sets and unknown payoff functions. In arXiv:1608.07310, 2016.

                                           31
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




Panayotis Mertikopoulos, Christos Papadimitriou, and Georgios Piliouras. Cycles in adver-
  sarial regularized learning. In SODA, 2018.

Panayotis Mertikopoulos, Houssam Zenati, Bruno Lecouat, Chuan-Sheng Foo, Vijay Chan-
  drasekhar, and Georgios Piliouras. Mirror descent in saddle-point problems: Going the
  extra (gradient) mile. In ICLR, 2019.

Lars Mescheder, Sebastian Nowozin, and Andreas Geiger. The Numerics of GANs. In
  NeurIPS. 2017.

Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which Training Methods for
  GANs do actually Converge? In ICML, 2018.

L Metz, B Poole, D Pfau, and J Sohl-Dickstein. Unrolled generative adversarial networks.
  In ICLR, 2017.

Dov Monderer and Lloyd S Shapley. Potential Games. Games and Economic Behavior, 14:
 124–143, 1996.

Vaishnavh Nagarajan and J Zico Kolter. Gradient descent GAN optimization is locally
  stable. In NeurIPS, 2017.

John Nash. Equilibrium points in n-person games. PNAS, 36(1):48–49, 1950.

Yurii Nesterov. Introductory Lectures on Convex Optimization: A Basic Course. Kluwer,
  2004.

J. Ortega and W. Rheinboldt. Iterative Solution of Nonlinear Equations in Several Variables.
   Society for Industrial and Applied Mathematics, 2000.

Christos Papadimitriou and Georgios Piliouras. From Nash Equilibria to Chain Recurrent
 Sets: Solution Concepts and Topology. In ITCS, 2016.

Christos Papadimitriou and Georgios Piliouras. From Nash Equilibria to Chain Recurrent
 Sets: An Algorithmic Solution Concept for Game Theory. Entropy, 20, 2018.

D Pathak, P Agrawal, A A Efros, and T Darrell. Curiosity-driven Exploration by Self-
  supervised Prediction. In ICML, 2017.

David Pfau and Oriol Vinyals. Connecting Generative Adversarial Networks and Actor-Critic
  Methods. In arXiv:1610.01945, 2016.

S Racanière, T Weber, D P Reichert, L Buesing, A Guez, D J Rezende, A P Badia,
  O Vinyals, N Heess, Y Li, R Pascanu, P Battaglia, D Hassabis, D Silver, and D Wierstra.
  Imagination-Augmented Agents for Deep Reinforcement Learning. In NeurIPS, 2017.

Sasha Rakhlin and Karthik Sridharan. Optimization, Learning, and Games with Predictable
  Sequences. In NeurIPS, 2013.

J B Rosen. Existence and Uniqueness of Equilibrium Points for Concave N -Person Games.
  Econometrica, 33(3):520–534, 1965.

                                            32
                           Differentiable Game Mechanics




R W Rosenthal. A Class of Games Possessing Pure-Strategy Nash Equilibria. Int J Game
 Theory, 2:65–67, 1973.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen.
  Improved Techniques for Training GANs. In NeurIPS, 2016.

Shibani Santurkar, Ludwig Schmidt, and Aleksander Madry. A Classification-Based Study
  of Covariate Shift in GAN Distributions. In ICML, 2018.

G Scutari, D P Palomar, F Facchinei, and Jong-Shi Pang. Convex Optimization, Game
 Theory, and Variational Inequality Theory. IEEE Signal Processing Magazine, pages
 35–49, 2010.

Yoav Shoham and Kevin Leyton-Brown. Multiagent Systems: Algorithmic, Game-Theoretic,
  and Logical Foundations. Cambridge University Press, 2008.

Michael Shub. Global Stability of Dynamical Systems. Springer, 2000.

S Singh, M Kearns, and Y Mansour. Nash Convergence of Gradient Dynamics in General-Sum
  Games. In UAI, 2000.

G Stoltz and G Lugosi. Learning correlated equilibria in games with compact sets of
 strategies. Games and Economic Behavior, 59:187–208, 2007.

Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E. Schapire. Fast Convergence
  of Regularized Learning in Games. In NeurIPS, 2015.

A Vezhnevets, S Osindero, T Schaul, N Heess, M Jaderberg, D Silver, and K Kavukcuoglu.
  FeUdal Networks for Hierarchical Reinforcement Learning. In ICML, 2017.

G Wayne and L F Abbott. Hierarchical Control Using Networks Trained with Higher-Level
  Forward Models. Neural Computation, (26), 2014.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired Image-to-Image
  Translation using Cycle-Consistent Adversarial Networks. In CVPR, 2017.

Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent.
 In ICML, pages 928–936, 2003.

APPENDIX

A. TensorFlow code to compute SGA
Source code is available at https://github.com/deepmind/symplectic-gradient-adjustment.
Since computing the symplectic adjustment is quite simple, we include an explicit description
here for completeness.
   The code requires a list of n losses, Ls, and a list of variables for the n players, xs.
The function fwd gradients which implements forward mode auto-differentiation is in the
module tf.contrib.kfac.utils.


                                           33
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




% compute Jacobian-vector product Jv
def jac vec(ys, xs, vs) :
    return fwd gradients(ys, xs, grad xs = vs, stop gradients = xs)

% compute Jacobian| -vector product J| v
def jac tran vec(ys, xs, vs) :
  dydxs = tf.gradients(ys, xs, grad ys = vs, stop gradients = xs)
  return [tf.zeros like(x) if dydx is None
          else dydx for (x, dydx) in zip(xs, dydxs)]

% compute Symplectic Gradient Adjustment A| ξ
def get sym adj(Ls, xs) :
  % compute game dynamics ξ
  xi = [tf.gradients(`, x)[0] for (`, x) in zip(Ls, xs)]
  J xi = jac vec(xi, xs, xi)
  Jt xi = jac tran vec(xi, xs, xi)
  % compute A| ξ = 21 (J| ξ − Jξ)
  At xi = [ jt−j
              2  for (j, jt) in zip(J xi, Jt xi)]
  return At xi

B. Helmholtz, Hamilton, Hodge, and Harmonic games
This section explains the mathematical connections with the Helmholtz decomposition,
symplectic geometry and the Hodge decomposition. The discussion is not necessary to
understand the main text. It is also not self-contained. The details can be found in textbooks
covering differential and symplectic geometry (Arnold, 1989; Guillemin and Sternberg, 1990;
Bott and Tu, 1995).

B.1 The Helmholtz Decomposition
The classical Helmholtz decomposition states that any vector field ξ in 3-dimensions is the
sum of curl-free (gradient) and divergence-free (infinitesimal rotation) components:
                                                                   h                     i
              ξ=         ∇φ             +        curl(B)               curl(•) := ∇ × (•)
                         |{z}                    | {z }
                   gradient component       rotational component


We explain the link between curl and the antisymmetric component of the game Jacobian.
Recall that gradients of functions are actually differential 1-forms, not vector fields. Differ-
ential 1-forms and vector fields on a manifold are canonically isomorphic once a Riemannian
metric has been chosen. In our case, we are implicitly using the Euclidean metric. The
antisymmetric matrix A is the differential 2-form obtained by applying the exterior derivative
d to the 1-form ξ.
    In 3-dimensions, the Hodge star operator is an isormorphism from differential 2-forms to
vector fields, and the curl can be reformulated as curl(•) = ∗d(•). In claiming A is analogous
to curl, we are simply dropping the Hodge-star operator.

                                                     34
                              Differentiable Game Mechanics




    Finally, recall that the Lie algebra of infinitesimal rotations in d-dimensions is given
by antisymmetric matrices. When d = 3, the Lie algebra can be represented as vectors
(three numbers specify a 3 × 3 antisymmetric matrix) with the ×-product as Lie bracket. In
general, the antisymmetric matrix A captures the infinitesimal tendency of ξ to rotate at
each point in the parameter space.

B.2 Hamiltonian Mechanics
A symplectic form ω is a closed nondegenerate differential 2-form. Given a manifold with
a symplectic form, a vector field ξ is Hamiltonian vector field if there exists a function
H : M → R satisfying
                                    ω(ξ, •) = dH(•) = h∇H, •i.                               (4)

The function is then referred to as the Hamiltonian function of the vector field. In our
case, the antisymmetric matrix A is a closed 2-form because A = dξ and d ◦ d = 0. It may
however be degenerate. It is therefore a presymplectic form (Bottacin, 2005).
    Setting ω = A, equation (4) can be rewritten in our notation as

                                           ω(ξ, •) = dH(•),
                                           | {z } | {z }
                                             A| ξ         ∇H


justifying the terminology ‘Hamiltonian’.

B.3 The Hodge Decomposition
The exterior derivative dk : Ωk (M ) → Ωk+1 (M ) is a linear operator that takes differential
k-forms on a manifold M , Ωk (M ), to differential k + 1-forms, Ωk+1 (M ). In the case k = 0,
the exterior derivative is the gradient, which takes 0-forms (that is, functions) to 1-forms.
Given a Riemannian metric, the adjoint of the exterior derivative δ goes in the opposite
direction. Hodge’s theorem states that k-forms on a compact manifold decompose into a
direct sum over three types:

                   Ωk (M ) = dΩk−1 (M ) ⊕ Harmonick (M ) ⊕ δΩk+1 (M ).

Setting k = 1, we recover a decomposition that closely resembles the generalized Helmholtz
decomposition:

              Ω1 (M ) =        dΩ0 (M )            ⊕Harmk (M ) ⊕         δΩ2 (M )
              | {z }           | {z }                                    | {z }
              1-forms     gradients of functions                   antisymmetric component


The harmonic component is isomorphic to the de Rham cohomology of the manifold – which
is zero when k = 1 and M = Rn .
    Unfortunately, the Hodge decomposition does not straightforwardly apply to the case
when M = Rn , since Rn is not compact. It is thus unclear how to relate the generalized
Helmholtz decomposition to the Hodge decomposition.

                                                     35
         Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




B.4 Harmonic and Potential Games
Candogan et al. (2011) derive a Hodge decomposition for games that is closely related in
spirit to our generalized Helmholtz decomposition – although the details are quite different.
Candogan et al. (2011) work with classical games (probability distributions on finite strategy
sets). Their losses are multilinear, which is easier than our setting, but they have constrained
solution sets, which is harder in many ways. Their approach is based on combinatorial
Hodge theory (Jiang et al., 2011) rather than differential and symplectic geometry. Finding
a best-of-both-worlds approach that encompasses both settings is an open problem.

C. Type Consistency
The next two sections carefully work through the units in classical mechanics and two-player
games respectively. The third section briefly describes a use-case for type consistency.

C.1 Units in Classical Mechanics
Consider the well-known Hamiltonian
                                                                               
                                           1                     1
                                 H(p, q) =                κ · q + · p2
                                                                2
                                           2                     µ

where q is position, p = µ · q̇ is momentum, µ is mass, κ is surface tension and H measures
energy. The units (denoted by τ ) are

                                      τ (q) = m            τ (p) = kg·m
                                                                     s


                                      τ (κ) = kg
                                              s2
                                                               τ (µ) = kg

where m is meters, kg is kilograms and s is seconds. Energy is measured in joules, and
                                             2
indeed it is easy to check that τ (H) = kg·m
                                          s2
                                               .
                                                       ∂        1
   Note that the units for differentation by x are τ ( ∂x ) = τ (x) . For example, differentiating
                                                                                  ∂H
by time has units 1s . Hamilton’s equations state that q̇ = ∂H    1
                                                             ∂p = µ · p and ṗ = − ∂q = −κ · q
where
                                 τ (q̇) = m
                                          s      τ (ṗ) = kg·m
                                                           s2
                                                                     
                                        ∂         1                ∂            s
                                τ       ∂q       =m        τ       ∂p       = kg·m

The resulting flow describing the dynamics of the system is

                                      ∂         ∂   1   ∂        ∂
                           ξ = q̇ ·      + ṗ ·    = p·   − κq ·
                                      ∂q        ∂p  µ ∂q         ∂p

with units τ (ξ) = 1s . Hamilton’s equations can be reformulated more abstractly via symplectic
geometry. Introduce the symplectic form

                                                                                kg · m2
                          ω = dq ∧ dp with units                    τ (ω) =             .
                                                                                   s

                                                      36
                              Differentiable Game Mechanics




Observe that contracting the flow with the Hamiltonian obtains

                                                            ∂H        ∂H
                         ιξ ω = ω(ξ, •) = dH =                 · dq +    · dp
                                                            ∂q        ∂p
                                       2
with units τ (dH) = τ (H) = kg·m
                              s2
                                 .
Losses in classical mechanics. Although there is no notion of “loss” in classical me-
chanics, it is useful (for the next section) to keep pushing the formal analogy. Define the
“losses”
                                      1
                           `1 (q, p) = · qp and `2 (q, p) = −κ · qp                     (5)
                                      µ
                     2                        2   2
with units τ (`1 ) = ms and τ (`2 ) = kg s·m
                                          3   . The Hamiltonian dynamics can then be recovered
game-theoretically by differentiating `1 and `2 with respect to q and p respectively. It is
easy to check that
                                ∂H ∂        ∂H ∂     ∂`1 ∂     ∂`2 ∂
                          ξ=             −         =        +        .
                                 ∂p ∂q       ∂q ∂p    ∂q ∂q    ∂p ∂p
The duality between vector fields and differential forms. Finally recall that the
symplectic form in games was not “pulled out of thin air” as ω = dq ∧ dp, but rather derived
as ω = dξ [ , where ξ [ is the differential form corresponding to the vector field ξ under the
musical isomorphism [ : T M → T ∗ M .
     It is instructive to compute ξ [ in the case of a classical mechanical system and see
                                                                               [
                                                                                 ∂
what happens. Naively, we would guess that the musical isomorphism is ∂q             = dq and
 [
  ∂
  ∂p    = dp. However, applying the naive musical isomorphism to ξ to get

                                                  ∂`1        ∂`2
                                           ξ[ =       · dq +     · dp
                                                  ∂q         ∂p

results in a type violation because

                                                          m2
                                            
                                     ∂`1
                                 τ       · dq = τ (`1 ) =
                                     ∂q                   s

whereas
                                                               kg 2 · m2
                                                 
                                          ∂`2
                                  τ           · dp = τ (`2 ) =
                                          ∂p                       s3
and we cannot add objects with different types.
   To correct the type inconsistency, define the musical isomorphism as
                                 [                                 [
                             ∂          µ                        ∂              1
                                       = · dq         and                  =      · dp
                             ∂q         2                        ∂p            2κ

with inverse
                                           2 ∂                                  ∂
                          (dq)] =           ·         and    (dp)] = 2κ ·          .
                                           µ ∂q                                 ∂p

                                                       37
          Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




The correction terms in the direction [ : T M → T ∗ M invert the coupling terms κ and µ1
that were originally introduced into the Hamiltonian for physical reasons. Applying the
corrected musical isomorphism to ξ yields
                             µ ∂f         1 ∂g       1
                      ξ[ =    ·   · dq +    ·  · dp = (p · dq − q · dp) .
                             2 ∂q        2κ ∂p       2

The two terms of ξ [ then have coherent types
                                                                     2
                            τ ∂`∂q
                                   1
                                      ·  µ · dq   =m s · kg · m = kg·m
                                                                    s
                                                                        2
                                                         s2 kg·m
                         τ ∂` 2
                             ∂p κ·  1
                                        · dp   = kg·m
                                                   s2
                                                       · kg · s = kg·m  s

as required. The associated two form is

                                   ∂2f   1 ∂2g
                                              
                        [
                ω := dξ = − µ ·         − ·      dq ∧ dp = −dq ∧ dp
                                   ∂q∂p κ ∂q∂p
                                                                              2
which recovers the symplectic form (up to sign) with units τ (ω) = kg·m   s as required. Finally,
observe that
                                                                        
                           [     1 p ∂              ∂
                      hξ, ξ i =        ·     − κq ·    , p · dq − q · dp
                                 2 µ ∂q             ∂p
                                                   
                                 1            1
                              =     κ · q 2 + · p2 = H(p, q)
                                 2            µ

recovering the Hamiltonian.

C.2 Units in Two-Player Games
Without loss of generality let w = (x; y) where we refer to x as position and y as momentum
so that τ (x) = m and τ (y) = kg·ms . The aim of this section is to check type-consistency
under these, rather arbitrarily assigned, units. Since we are considering a game, we do not
require that x and y have the same dimension – even though this would necessarily be the
case for a physical system. The goal is to verify that units can be consistently assigned to
games.
    Consider a quadratic two player game of the form
                                                                  
                             1 | |  A11 A12          x        |  |
                                                                     b1
                   `1 (w) =     x y                       + x y
                             2           A21 A22      y                b2

and                                                      
                              1 | |  C11 C12   x     |  |
                                                            d1
                     `2 (w) =   x y                + x y
                              2       C21 C22   y            d2
We restrict to quadratic games since our methods only involve first and second derivatives.
We assume the matrices A and C are symmetric without loss of generality so that, for
example, A12 = A|21 . Adding constant terms to `1 and `2 makes no difference to the analysis
so they are omitted.

                                                  38
                              Differentiable Game Mechanics



                                                            2                   2
    By (5), the units for `1 and `2 should be ms and kg·m
                                                        s3
                                                            respectively. We can therefore
derive the correct units for each of the components of the quadratic losses as
                                1      1                    m
                                   s   kg     m                 s
                      m kg·m                      + m kg·m  
                              
                                         
                            s                               s
                                     1         s        kg·m                                m
                                    kg        kg 2        s                                 kg
                       |                 {z                         }       |       {z            }
                                    w| Aw                                           w| b

for `1 and                                                                                    
                                 kg 2        kg                                        kg 2 ·m
                                                              
                                s3         s2 
                                                       m                              s3 
                       m kg·m
                           s                                + m kg·m 
                                                                      s                           
                                   kg         1        kg·m                                kg·m
                                   s2         s          s                                  s2
                   |                {z                          }       |           {z                }
                                   w| Cw                                            w| d

for `2 . It follows from a straightforward computation that the vector field ξ = ∂`1 ∂   ∂`2 ∂
                                                                                 ∂x ∂x + ∂y ∂y
has type τ (ξ) = 1s as required.
    The presymplectic form ω = dξ [ makes use of the musical isomorphism [ : T M → T ∗ M .
                                            ∂ [             ∂ [
As in section C.1, if we naively define ( ∂x  ) = dx and ( ∂y ) = dy then

                                                  ∂`1        ∂`2
                                    ξ[ =              · dx +     · dy
                                                  ∂x         ∂y
                                                                    2  kg ·m                              2   2
                                                            ∂`2
which is type inconsistent because τ ( ∂`         m
                                       ∂x · dx) = s and τ ( ∂y · dy) =
                                          1
                                                                         s3
                                                                             .
Type-consistency via SVD. It is necessary, as in section C.1, to correct the naive
musical isomorphism by taking into account the coupling constants for the mixed position-
momentum terms. In the classical setup the coupling constants were the scalars µ1 and κ,
whereas in a game they are the off-diagonal blocks A12 and C12 .
   Apply singular value decomposition to factorize

                           A12 = U|A DA VA             and C12 = U|C DC VC

where the entries of the diagonal matrices have types τ (DA ) = kg   1
                                                                       and τ (DC ) = kg
                                                                                      s2
                                                                                         , and
the types of the orthogonal matrices U and V are pure scalars. The diagonal matrices DA
and DC have the same types as µ1 and κ in the classical system since they play the same
coupling role.
    Extending the procedure adopted in the section C.1, fix the type-inconsistency by defining
the musical isomorphisms as
                                 [
                                   ∂
                                        = U|A D−1A UA · dx
                                  ∂x
and                                         [
                                         ∂            |
                                                   = VC D−1
                                                         C VC · dy.
                                        ∂y
                                                                            √
Alternatively, the isomorphisms
                      √         can be computed by noting that U|A D−1
                                                                    A UA = ( A12 A21 )
                                                                                       −1
      |   −1                   −1
and VC DC VC = ( C21 C12 ) .

                                                       39
          Letcher, Balduzzi, Racanière, Martens, Foerster, Tuyls, Graepel




     The dual isomorphism ] : T ∗ M → T M is then
                                           ∂                          ∂
                   (dx)] = U|A DA UA ·                    |
                                             and (dy)] = VC DC V C ·
                                          ∂x                         ∂y
If                                                    
                                           A12 y + b1
                                       ξ=
                                            C21 x + d2
then it follows that               | −1
                                   UA DA UA b1 + U|A UA y
                                                          
                               [
                              ξ =   |              |
                                   VC D−1
                                       C VC d2 + VC VC x
with associated closed two form

                         ωτ = dξ [ = − U|A VA − U|C VC dx ∧ dy.
                                                      


where the notation ωτ emphasizes that the two-form is type-consistent.

C.3 What Does Type-Consistency Buy?
Example 10 Consider the loss functions

                              f (x, y) = xy     and    g(x, y) = 2xy,

with ξ = (y, 2x). There is no function φ : R2 → R such that ∇φ = ξ. However, there is a
family of functions φα (x, y) = α · xy which satisfies

                         ξ, ∇φα = α · (x2 + 2y 2 ) ≥ 0       for all α > 0.

Although ξ is not a potential field, there is a family of functions on which ξ performs gradient
descent – albeit with coordinate-wise learning rates that may not be optimal. The vector
field ξ arguably does not require adjustment. This kind of situation often arises when the
learning rates of different parameters are set adaptively during training of neural nets, by
rescaling them by positive numbers.
    The vanilla and type-consistent 1-forms corresponding to ξ are, respectively,

                       ξ [ = y · dx + 2x · dy   and ξ [τ = y · dx + x · dy

with
                         ω = dξ [non = dx ∧ dy        and ωτ = dξ [τ = 0.
 It follows that the type-consistent symplectic gradient adjustment is zero. Type-consistency
‘detects’ that no gradient adjustment is needed in example 10.




                                                 40
