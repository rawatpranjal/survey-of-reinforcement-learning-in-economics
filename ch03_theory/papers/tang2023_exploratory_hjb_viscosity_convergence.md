## EXPLORATORY HJB EQUATIONS AND THEIR CONVERGENCE

WENPIN TANG, YUMING PAUL ZHANG, AND XUN YU ZHOU

Abstract. We study the exploratory Hamilton-Jacobi-Bellman (HJB) equation arising from the entropy-regularized exploratory control problem, which was formulated by Wang, Zariphopoulou and Zhou (J. Mach. Learn. Res., 21, 2020) in the context of reinforcement learning in continuous time and space. We establish the well-posedness and regularity of the viscosity solution to the equation, as well as the convergence of the exploratory control problem to the classical stochastic control problem when the level of exploration decays to zero. We then apply the general results to the exploratory temperature control problem, which was introduced by Gao, Xu and Zhou (arXiv:2005.04057, 2020) to design an endogenous temperature schedule for simulated annealing (SA) in the context of non-convex optimization. We derive an explicit rate of convergence for this problem as exploration diminishes to zero, and find that the steady state of the optimally controlled process exists, which is however neither a Dirac mass on the global optimum nor a Gibbs measure.

Key words: HJB equations, stochastic control, partial differential equations, reinforcement learning, exploratory control, entropy regularization, simulated annealing, overdamped Langevin equation.

## 1. Introduction

Reinforcement learning (RL) is an active subarea of machine learning. The RL research has predominantly focused on Markov Decision Processes (MDPs) in discrete time and space; see Sutton and Barto (2018) for a systematic account of the theory and applications, as well as a detailed description of bibliographical and historical development of the field. Wang et al. (2020) are probably the first to formulate and develop an entropy-regularized, exploratory control framework for RL in continuous time with continuous feature (state) and action (control) spaces. In this framework, stochastic relaxed control, a measure-valued process, is employed to represent exploration through randomization, capturing the notion of 'trial and error' which is the core of RL. Entropy of the control is incorporated explicitly as a regularization term in the objective function to encourage exploration, with a weight parameter λ &gt; 0 on the entropy to gauge the tradeoff between exploitation (optimization) and exploration (randomization). This exploratory formulation has been extended to other settings and used to solve applied problems; see e.g. Guo et al. (2020) and Firoozi and Jaimungal (2020) to mean-field games, and Wang and Zhou (2020) to Markowitz mean-variance portfolio optimization. Gao et al. (2020) apply the same formulation to temperature control of Langevin diffusions arising from simulated annealing for non-convex optimization. The problem itself is not directly related to RL; however the authors take the same idea of 'exploration through randomization' and invoke exploratory controls to smooth out the highly unstable yet theoretically optimal bang-bang control. For more literature review on the exploratory control, see Zhou (2021).

Date : September 22, 2021.

Wang et al. (2020) derive the following Hamilton-Jacobi-Bellman (HJB) partial differential equation (PDE) associated with the exploratory control problem, parameterized by the weight parameter λ &gt; 0:

<!-- formula-not-decoded -->

This equation, called the exploratory HJB equation , appears to be characteristically different from the HJB equation corresponding to a classical stochastic control problem. Among other things, (1) does not involve the supremum operator in the control variable typically appearing in a classical HJB equation. This is because the supremum is replaced by a distribution among controls in the exploratory formulation. Wang et al. (2020) do not study this general equation in terms of its well-posedness (existence and uniqueness of the viscosity solution), regularity, stability in λ , and the convergence when λ → 0 + . They do, however, solve the important linear-quadratic (LQ) case where the exploratory HJB equation can be solved explicitly, leading to the optimal distribution for exploration being a Gaussian distribution. Wang and Zhou (2020) apply this result to a continuous-time Markowitz portfolio selection problem which is inherently LQ.

The goal of this paper is to study the general exploratory HJB equations beyond the LQ setting. We first analyze a class of elliptic PDEs under fairly general assumptions on the coefficients (Theorems 8 and 9). The application of the general results obtained to the exploratory HJB equations allows us to identify the assumptions needed, to derive the wellposedness of viscosity solutions and their regularity, and to establish a connection between the exploratory control problem and the classical stochastic control problem (Theorems 10 and 11). More specifically to the last point, we show that as the exploration weight decays to zero, the value function of the former converges to that of the latter. This result, which extends Wang et al. (2020) to the general setting, is important for RL especially in terms of finding the regret bound (or the cost of exploration as termed in Wang et al. (2020)). As a passing note, our analysis for the general class of fully nonlinear elliptic PDEs may be of independent interest to the PDE community.

In the second part of this paper, we focus on a special exploratory HJB equation resulting from the exploratory temperature control problem of the Langevin diffusions. The latter problem was introduced by Gao et al. (2020) aiming at designing a state-dependent temperature schedule for simulated annealing (SA). To provide a brief background (see Gao et al. (2020) for more details), one of the central problems in continuous optimization is to escape from saddle points and local minima, and to find a global minimum of a non-convex function f : R d → R . Applying the SA technique to the gradient descent algorithm consists of adding a sequence of independent Gaussian noises, scaled by 'temperature' parameters controlling the level of noises. The continuous version of the SA algorithm is governed by the following stochastic differential equation (SDE):

<!-- formula-not-decoded -->

where ( B t , t ≥ 0) is a d -dimensional Brownian motion, and the temperature schedule ( β t , t ≥ 0) is a stochastic process. If β t ≡ β is constant in time, then (2) is the well-known overdamped Langevin equation whose stationary distribution is the Gibbs measure G β ( dx ) ∝ exp( -f ( x ) /β ) dx ( f is called the landscape, and β the temperature).

When allowing ( β t , t ≥ 0) to be a stochastic process, we have naturally a stochastic control problem in which one controls the dynamics (2) through this temperature process in order to achieve the highest efficiency in optimizing f . Gao et al. (2020) find that the optimal control of this problem is of bang-bang type: the temperature process switches between two extremum points in the search interval. Such a bang-bang solution is almost unusable in practice since it is highly sensitive to errors. Moreover, in the present paper we discover that the optimal state process under the bang-bang control may even not be wellposed in dimensions d ≥ 3 (Section 4.1). These observations support the entropy-regularized exploratory formulation of temperature control proposed by Gao et al. (2020), not so much from a learning perspective, but from a desire of smoothing out the bang-bang control.

The results for the general exploratory HJB equations apply readily to the temperature control setting in terms of the well-posedness, regularity and convergence (Corollaries 12 and 16). Moreover, due to the special structure of the controlled dynamics (2), we are able to derive an explicit convergence rate of λ ln(1 /λ ) for the exploratory temperature control problem as λ tends to zero (Theorem 15). Finally, we consider the long time behavior of the associated optimally controlled process and show that it will not converge to the global minimum of f nor any Gibbs measure with landscape f (Theorem 18). The first property is indeed preferred from an exploration point of view because exploration is meant to involve as many states as possible instead of focusing only on the single state of the minimizer. The second property hints the possibility of a more variety of target measures other than Gibbs measures for SA.

The remainder of the paper is organized as follows. In Section 2, we provide some background on the exploratory control framework and present the corresponding exploratory HJB equation. In Section 3, we investigate the exploratory HJB equation and establish general results in terms of its well-posedness, regularity and convergence. We also identify the value function of the exploratory control problem as the unique solution to the exploratory HJB equation. In Section 4, we apply the general results to the exploratory temperature control problem, derive an explicit convergence rate, and study the long time behavior of the associated optimal state process. While the main focus of the paper is on problems in the infinite time horizon, in Section 5 we discuss the case of a finite time horizon. Finally, Section 6 concludes with a few open questions suggested.

## 2. Background and problem formulation

In this section, we provide some background on the exploratory control problem that is put forth in Wang et al. (2020).

Below we collect some notations that will be used throughout this paper.

- For a square matrix X = ( X ij ) ∈ R d × d , X T denotes its transpose, Tr( X ) its trace, | X | its spectral norm, and | X | max = max 1 ≤ i,j ≤ d | X ij | its max norm. Moreover, S d = { X ∈ R d × d : X T = X } denotes the set of d × d symmetric matrices with the spectral norm.
- For x, y ∈ R d , x · y denotes the inner product between x and y , | x | = √ ∑ d i =1 x 2 i denotes the Euclidean norm of x , B R = { x : | x | ≤ R } denotes the Euclidean ball of radius R centered at 0, and | x | max = max 1 ≤ i ≤ d | x i | denotes the max norm of x .

- Let O ⊆ R d be open. For a function f : O → R , ∇ f , ∇ 2 f and ∆ f = Tr( ∇ 2 f ) denote respectively its gradient, Hessian and Laplacian.
- For a bounded function f : O → R , || f || L ∞ ( O ) = sup x ∈O | f ( x ) | denotes the sup norm of f .
- A function f ∈ C k ( O ), or simply f ∈ C k , if it is k -time continuously differentiable. The C k norm is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- A function f ∈ C k,α ( O ), or simply f ∈ C k,α (0 &lt; α &lt; 1), if it is k -time continuously differentiable and its k th derivatives of f are α -H¨ older continuous. The C k,α norm is

/negationslash

<!-- formula-not-decoded -->

- For two probability measures P and Q , || P -Q || TV = sup A | P ( A ) -Q ( A ) | denotes the total variation distance between P and Q .
- 2.1. Classical control problem. Let (Ω , F , P , {F t } t ≥ 0 ) be a filtered probability space on which we define a d -dimensional F t -adapted Brownian motion ( B t , t ≥ 0). Let U be a generic action/control space, and u = ( u t , t ≥ 0) be a control which is an F t -adapted process taking values in U .

The classical stochastic control problem is to control the state variable X t ∈ R d , whose dynamics is governed by the SDE:

<!-- formula-not-decoded -->

where b : R d × U → R d is the drift, and σ : R d × U → R d × d is the covariance matrix of the state variable. Here the superscript ' u ' in X u t emphasizes the dependence of the state variable on the control u . The goal of the control problem is to maximize the total discounted reward, leading to the (optimal) value function:

∣ where h : R d ×U → R is a reward function, ρ &gt; 0 is the discount factor, and A 0 ( x ) denotes the set of admissible controls which may depend on the initial state value X u 0 = x .

<!-- formula-not-decoded -->

By a standard dynamic programming argument, the HJB equation associated with the problem (4) is

<!-- formula-not-decoded -->

In the classical stochastic control setting, the functional forms of h, b, σ are given and known. It is known that a suitably smooth solution to the HJB equation (5) gives the value function (4). Further, the optimal control is represented as a deterministic mapping from the current state to the action/control space: u ∗ t = u ∗ ( X ∗ t ). The mapping u ∗ is called an optimal feedback control, which is derived offline from the 'sup u ∈U ' term in (5). This procedure of obtaining

the optimal feedback control is called the verification theorem. The corresponding optimally controlled process ( X ∗ t , t ≥ 0) is governed by the SDE:

<!-- formula-not-decoded -->

provided that it is well-posed (i.e. it has a unique weak solution). See e.g. Yong and Zhou (1999); Fleming and Soner (2006) for detailed accounts of the classical stochastic control theory.

2.2. Exploratory control problem. In the RL setting, the model parameters are unknown, i.e. the functions h, b, σ are not known. Thus, one needs to explore and learn the optimal controls through repeated trials and errors. Inspired by this, Wang et al. (2020) model exploration by a probability distribution of controls π = ( π t ( · ) , t ≥ 0) over the control space U from which each trial is sampled. The exploratory state dynamics is

˜ where the coefficients b ( · , · ) and σ ( · , · ) are defined by

<!-- formula-not-decoded -->

with P ( U ) being the set of absolutely continuous probability density functions on U . The distributional control π = ( π t ( · ) , t ≥ 0) is also known as the relaxed control, and a classical control u = ( u t , t ≥ 0) is a special relaxed control when π t ( · ) is taken as the Dirac mass at u t .

<!-- formula-not-decoded -->

The exploratory control problem is an optimization problem similar to (4) but under relaxed controls. Moreover, to encourage exploration, Shannon's entropy is added to the objective function as a regularization term:

where λ &gt; 0 is a weight parameter controlling the level of exploration (also called the temperature parameter), and A ( x ) is the set of admissible distributional controls specified by the following definition.

<!-- formula-not-decoded -->

Definition 1. We say a density-function-valued stochastic process π = ( π t ( · ) , t ≥ 0) , defined on a filtered probability space (Ω , F , P , {F t } t ≥ 0 ) along with a d -dimensional F t -adapted Brownian motion ( B t , t ≥ 0) , is an admissible distributional (or exploratory) control, denoted by π ∈ A ( x ) , if

- ( ii ) For any Borel set A ⊂ U , the process ( t, ω ) → ∫ A π t ( u, ω ) du is F t -progressively measurable;
- ( i ) For each t ≥ 0 , π t ( · ) ∈ P ( U ) a.s.;
- ( iii ) The SDE (7) has solutions on the same filtered probability space whose distributions are all identical.

Now we quickly review a formal derivation of the solution to the exploratory control problem (7)-(9), following Wang et al. (2020). Again, by a dynamic programming argument, the

HJB equation to (7)-(9) is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, through the same verification theorem argument, the optimal distributional control is obtained by solving the maximization problem in (10) with the constraints ∫ U π ( u ) du = 1 and π ( u ) ≥ 0 a.e. on U . This yields the optimal feedback control:

<!-- formula-not-decoded -->

By injecting (11) into (10), we get the nonlinear elliptic PDE (1), or the exploratory HJB equation. Note that this equation is parameterized by the weight parameter λ &gt; 0.

Applying the feedback control (11) to the state dynamics (7), we obtain the optimally controlled dynamics:

dX λ, ∗ t = ˜ b ( X λ, ∗ t , π ∗ ( · , X λ, ∗ t )) dt + ˜ σ ( X λ, ∗ t , π ∗ ( · , X λ, ∗ t )) dB t , (12) provided that it is well-posed, i.e. it has a weak solution which is unique in distribution. This condition is satisfied if b ( · , · ) and σ ( · , · ) are measurable and bounded, x → σ ( x, · ) is continuous, and σ ( · , · ) is strictly elliptic in the sense that σ ( · , · ) σ ( · , · ) T ≥ Λ I ; see e.g. Stroock and Varadhan (1979) for discussions on the well-posedness of SDEs. The optimal distributional control is then π λ, ∗ t ( · ) = π ∗ ( · , X λ, ∗ t ), t ≥ 0.

The exploratory HJB equation (1) is a new type of PDE in control theory, which begs a number of questions. The first question is, naturally, its well-posedness (existence and uniqueness) in certain sense. The second question is its dependence and convergence in λ &gt; 0. In practice, this parameter is often set to be small. Thus, we are interested in the limit of the solution to (1) as λ → 0 + , along with its convergence rate. We will answer these questions in the following two sections.

## 3. Analysis of the exploratory HJB equation

In this section, we study the exploratory HJB equation (1) under some general assumptions on the functions h ( · , · ) , b ( · , · ) , σ ( · , · ). For a concise analysis it is advantageous to analyze the general fully nonlinear elliptic PDEs of the form

<!-- formula-not-decoded -->

In Section 3.1 we recall a few results on general elliptic PDEs in bounded domains, and prove a comparison principle for viscosity solutions of sub-quadratic growth in R d . We show in Section 3.2 that, under some continuity and growth conditions on the operator F , (13) has a unique smooth solution among functions that have sub-quadratic growth in R d . In Section 3.3, we consider a sequence of operators F λ that converge locally uniformly to F ,

and derive a convergence rate of the corresponding solutions v λ as λ → 0 + to v . The rate of convergence for not necessarily bounded solutions with general operators (in particular with possibly unbounded coefficients) is novel. Finally in Section 3.4, we specify the general PDE results to the exploratory HJB equation (1), and prove a convergence result for the exploratory control problem (7)-(9) as λ → 0 + .

- 3.1. General results on second order elliptic equations. The standard references for second order elliptic PDEs are Gilbarg and Trudinger (1983); Caffarelli and Cabr´ e (1995). Here we recall some definitions and useful results.

Consider the general fully nonlinear equations (13). We make the following assumptions on the operator F : S d × R d × R × R d → R .

- ( a ) F is continuous in all its variables, and for each r ≥ 1 there exist γ r , γ r &gt; 0 such that for any x, y ∈ B r and ( X,p,q,s ) ∈ S d × R 2 d × R ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- ( b ) There exist Λ 2 &gt; Λ 1 &gt; 0 such that for any P ∈ S d positive semi-definite, and any ( X,p,s,x ) ∈ S d × R d × R × R d ,

<!-- formula-not-decoded -->

- ( c ) There exists ρ &gt; 0 such that for all ( X,p,x ) ∈ S d × R d × R d and t ≥ s ,

<!-- formula-not-decoded -->

These assumptions are standard (see Ishii and Lions (1990); Crandall et al. (1992)), and guarantee the existence and uniqueness of the viscosity solution to the equation (13) in a bounded domain with a Dirichlet boundary condition. The proof is given by Perron's method and the comparison principle. Note that there exist weaker conditions than the ones stated above to ensure the well-posedness of (13) in bounded domains; however, assumptions ( a )-( c ) are simpler and sufficient for our purpose.

Now we recall the definition of viscosity solutions to (13).

Definition 2. Let Ω be an open set in R d .

- (i) We say an upper semicontinuous (resp. lower semicontinuous) function v : Ω → R is a subsolution (resp. supersolution) to (13) if the following holds: for any smooth function φ in Ω such that v -φ has a local maximum (resp. minimum) at x 0 ∈ Ω , we have

<!-- formula-not-decoded -->

- (ii) We say a continuous function v : Ω → R is a (viscosity) solution to (13) if it is both a subsolution and a supersolution.

Throughout this paper, by a solution of a PDE we mean a viscosity solution unless otherwise stated.

Assume that there are a set of functions defined on in Ω: { v ε ( x ) , ε &gt; 0 } . Recall the definition of half-relaxed limits:

<!-- formula-not-decoded -->

Clearly, v ∗ is upper semicontinuous and v ∗ is lower semicontinuous. It is known that sub and supersolutions are stable under the half-relaxed limit operations; see Crandall et al. (1992).

Lemma 3. Let Ω ⊆ R d be open, { F λ , λ &gt; 0 } be a set of operators satisfying the assumptions (a)-(c) with the same constants. Suppose that F λ converges locally uniformly in all its variables to an operator ¯ F as λ → 0 + . Then

<!-- formula-not-decoded -->

- (i) if v λ is a sequence of bounded subsolutions to F λ ( ∇ 2 v λ , ∇ v λ , v λ , · ) ≤ 0 in Ω for some λ → 0 + , then their upper half-relaxed limit v ∗ is a subsolution to
- (ii) if v λ is a sequence of bounded supersolutions to F λ ( ∇ 2 v λ , ∇ v λ , v λ , · ) ≥ 0 in Ω for some λ → 0 + , then their lower half-relaxed limit v ∗ is a supersolution to

<!-- formula-not-decoded -->

Next we consider the regularity of solutions to (13). We need the following additional assumption on the operator F .

Definition 4. We say that an operator F = F ( X,p,s,x ) is concave in X , if for any M,N ∈ S d , p, x ∈ R d , and s ∈ R we have

<!-- formula-not-decoded -->

where the derivative and the inequality are in the sense of distribution.

The following result concerns higher regularity of bounded solutions to concave operators; see e.g. Caffarelli and Cabr´ e (1995) and Lian et al. (2020). As a consequence, viscosity solutions to concave operators are classical solutions.

Lemma 5 (Theorems 2.1 and 2.6, Lian et al. (2020)) . Assume that F = F ( X,p,s,x ) satisfies (a)-(c), and let R 2 &gt; R 1 &gt; 0 . If v is a bounded viscosity solution to F ( ∇ 2 v, ∇ v, v, x ) = 0 in B R 2 , then v is C 1 ,α in B R 1 . Moreover if F is concave in X , then v is C 2 ,α in B R 1 . The upper bounds for || v || C 1 ,α ( B R 1 ) or || v || C 2 ,α ( B R 1 ) depend only on the constants in assumptions (a)-(c), R 1 , R 2 , and ‖ v ‖ L ∞ ( B R 2 ) .

Finally, we prove a comparison principle for solutions to (13), where the operator F is assumed to have a certain sub-quadratic growth in x in the whole domain R d . This comparison principle will be used to prove the uniqueness of the solution to the exploratory HJB equation (1) under some assumptions on h ( · , · ) , b ( · , · ) , σ ( · , · ).

Lemma 6 (Comparison principle in R d ) . Assume that F satisfies (a)-(c) with γ r &gt; 0 such that

<!-- formula-not-decoded -->

Let v 1 and v 2 be locally uniformly bounded and be, respectively, a subsolution and a supersolution to (13) in R d such that

<!-- formula-not-decoded -->

Then v 1 ≤ v 2 in R d .

Note that in this lemma, there is no requirement on γ r . A proof of Lemma 6 relies on the following classical comparison principle for elliptic PDEs in a bounded domain.

Lemma 7 (Comparison principle, Theorem III.1, Ishii and Lions (1990)) . Let Ω ⊆ R d be a bounded open set, and assume (a)-(c) hold. Let u (resp. v ) be a bounded subsolution (resp. supersolution) to (13) in Ω such that

<!-- formula-not-decoded -->

Then u ≤ v in Ω .

Proof of Lemma 6. It follows from (15) that there exists C &gt; 0 such that for all r ≥ 0,

<!-- formula-not-decoded -->

Set C ′ := C +2 d Λ 2 ρ -1 , and for any small ε &gt; 0, define

<!-- formula-not-decoded -->

We claim that v ε is a supersolution to (13) in R d . Indeed, assume that there is ϕ ∈ C ∞ ( R d ) such that v ε -ϕ has a local minimum at x 0 ∈ R d . Then v -ϕ ε with ϕ ε := ϕ -ε ( C ′ + | x | 2 ) has a local minimum at x 0 . Using the facts that v 2 is a supersolution and F satisfies (a)-(c), we get by (17) that

<!-- formula-not-decoded -->

Hence v ε is a supersolution.

Next, due to (16), there exists R ε &gt; 0 such that v ε ( x ) ≥ v 1 ( x ) for all | x | ≥ R ε . Therefore applying Lemma 7 to v 1 , v ε with Ω = B R ε yields

<!-- formula-not-decoded -->

Taking ε → 0 leads to v 2 ≥ v 1 in R .

<!-- formula-not-decoded -->

The above proof of Lemma 6 follows rather standard lines. The comparison principle (and the well-posedness) for unbounded solutions to nonlinear elliptic equations in unbounded domains do exist in the literature; see e.g. Crandall et al. (1992); Capuzzo-Dolcetta et al. (2005); Koike and Ley (2011); Armstrong and Tran (2015). However, those results do not apply to the problem in which we are interested. In particular, none of these results covers the cases of unbounded b ( · , · ) and/or F being inhomogeneous in X , inherent in the exploratory control problem.

- 3.2. Well-posedness and stability. In this subsection we prove the well-posedness of solutions of sub-quadratic growth to (13). We need some assumptions on γ r , γ r . Let γ : (0 , ∞ ) → (0 , ∞ ) be C 2 . Setting γ r := γ ( r ) , γ ′ r := γ ′ ( r ) , γ ′′ r := γ ′′ ( r ), we assume that

<!-- formula-not-decoded -->

This γ r represents a rate of sub-quadratic growth. For instance, we can take γ r = C (1 + r a ) or C (1 + r a ln(1 + r )) with a ∈ [0 , 2) , C &gt; 0.

Theorem 8. The following hold:

- (i) Assume that (a)-(c) hold with γ r satisfying (18) and γ r satisfying

<!-- formula-not-decoded -->

Then there exists a unique solution v of sub-quadratic growth to (13) , and v is locally uniformly C 1 ,α . Moreover, there exists C &gt; 0 such that for all r ≥ 1 ,

<!-- formula-not-decoded -->

- (ii) Assume that there are operators F λ satisfying (a)-(c) uniformly with the above γ r , γ r for λ ∈ (0 , 1) , such that F λ → F as λ → 0 + locally uniformly in all the variables. Then the unique solution v λ to

<!-- formula-not-decoded -->

- (iii) If F (or F λ ) is concave in X , then v (or v λ ) is locally uniformly C 2 ,α .

is C 1 ,α , satisfies (20) , and v λ → v locally uniformly as λ → 0 + .

- Proof. (i) With the comparison principle (Lemma 6), we only need to produce a supersolution
- and a subsolution that have sub-quadratic growth at infinity, and invoke Perron's method. By ( a )-( c ) and (19), there exists a constant C &gt; 0 such that for any x ∈ R d , ( X,p ) ∈ d R d
- S × , and s ≥ 0, if r := | x | ≥ 1, then

and if r ∈ [0 , 1), then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let φ ∈ C 2 ([0 , ∞ )) be a regularization of r → γ r such that

Define ¯ v ( x ) := C 1 + C 2 φ ( | x | ) for some C 1 , C 2 &gt; 0 to be determined. For simplicity, below we drop ( x ) and ( | x | ) from the notations of ¯ v ( x ), φ ( | x | ), φ ′ ( | x | ) and φ ′′ ( | x | ). For | x | ≥ 1, we have from (22),

<!-- formula-not-decoded -->

It follows from (18) that

Therefore by picking C 2 and then C 1 to be sufficiently large, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This inequality holds the same when | x | &lt; 1 by (23) and (24); Similarly, one can show that v := -¯ v is a subsolution. Clearly both ¯ v and v have at most sub-quadratic growth. Thus by Perron's method and Lemma 6 (note that by (18), γ r satisfies (15)), we obtain the unique solution v to (13), and v ≤ v ≤ ¯ v yields (20). Finally v ∈ C 1 ,α follows from Lemma 5.

- (iii) This follows readily from Lemma 5. /square
- (ii) The above argument also yields the unique solution v λ to (21), with v λ ∈ C 1 ,α satisfying (20) for each λ ∈ (0 , 1). Let v ∗ , v ∗ be defined as in Lemma 3. Since F λ → F locally uniformly, Lemma 3 yields that v ∗ and v ∗ are, respectively, a supersolution and a subsolution to (13). As v ∗ and v ∗ have at most sub-quadratic growth, applying Lemma 6 yields v ∗ ≥ v ∗ in R d . The other direction of the inequality holds trivially by definition; hence v ∗ = v ∗ which then equals the unique solution v to (13). This shows v λ → v locally uniformly as λ → 0 + .
- 3.3. Rate of convergence. Recall that | X | denotes the spectral norm for X ∈ S d . We make the following assumption on the difference between F and F λ :
- ( d ) There exists a continuous function ω 0 : [0 , ∞ ) 4 → [0 , ∞ ) such that for each λ ≥ 0, ω 0 ( λ, · , · , · ) is non-decreasing in all its variables, ω 0 (0 , · , · , · ) ≡ 0, and for each ( X,p,s,x ) ∈ S × R d × R × R d we have

<!-- formula-not-decoded -->

In the remainder of this subsection, we derive a convergence rate of v λ → v as λ → 0 + , assuming that the Lipschitz norms of v λ and v are not too large at x → ∞ . To our best knowledge, this error estimate result in the general setting with possibly unbounded solutions in R d is new.

Theorem 9. Let C 0 ≥ 1 , η ∈ [0 , 2) , F, F λ satisfy (a)-(d) with γ r = C 0 (1 + r η ) , γ r = C 0 (1+ r η -1 ) , and v and v λ be, respectively, the solutions to (13) and (21) . Suppose for some α ≥ 0 , we have for each r ≥ 1 ,

<!-- formula-not-decoded -->

Then there exist A,C &gt; 0 such that for all λ ∈ (0 , 1) and r ≥ 1 , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We will only show that v cannot be too much larger than v λ for λ ∈ (0 , 1) in B r ; the proof for the other direction is almost identical. From the assumption and Theorem 8, there is C 1 ≥ C 0 such that for all r ≥ 1, we have γ r ≤ C 1 r η , γ r ≤ C 1 (1 + r η -1 ), and

<!-- formula-not-decoded -->

Then after writing

<!-- formula-not-decoded -->

for some r ≥ 1, (28) yields δ r ≤ C 1 r η .

Let R 1 := Ar for some A ≥ 1, and R 2 := R 1+ ε 1 with ε = 2 -η 2 ∈ (0 , 1]. We consider a radially symmetric, and radially non-decreasing function φ : R d → [0 , ∞ ) such that and for some C = C ( d ),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Due to (28) and (30), there exists x 0 ∈ B R 2 such that for all x ∈ B R 2 . A regularization of the map x → exp (max { 0 , x -r } /R 1 ) -1 will do if A is large enough depending only on η, C 1 . With one fixed A , below we prove a finer bound of δ r for all r large enough and λ ∈ (0 , 1).

<!-- formula-not-decoded -->

Similarly, for any β ≥ 1, we can find x 1 , y 1 ∈ B R 2 such that

If φ ( x 1 ) ≤ φ ( y 1 ), noting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

in view of (26), we conclude from (32) and (33) that which yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This estimate still holds if φ ( x 1 ) ≥ φ ( y 1 ) by the same argument. Let us write C φ := φ ( x 1 ) + φ ( y 1 ). It follows from (33) that

<!-- formula-not-decoded -->

Now we proceed by making use of (33). Since v, v λ are solutions to (13) and (21) respectively, the Crandall-Ishii lemma (Crandall et al., 1992, Theorem 3.2) yields that there are matrices X , Y ∈ S d satisfying the following:

Since (1 + ε ) η ≤ 2, (35) and (28) yield C φ ≤ C 1 R η 2 ≤ C 1 R 2 1 .

where and

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using (c) and (37) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Writing Y ′ := Y -∇ 2 φ ( y 1 ) and Z := X -Y + ∇ 2 φ ( x 1 )+ ∇ 2 φ ( y 1 ), we conclude from (a),(b),(d), and x 1 , y 1 ∈ B R 2 that

Then we apply (28), (31), (34), and C φ ≤ C 1 R 2 1 to obtain where c 4 := 1 + min { (1 -η )(1 + ε ) , 0 } ∈ (0 , 1] by ε = 2 -η 2 , and C = C ( C 0 , C 1 ) &gt; 0.

<!-- formula-not-decoded -->

Notice that X ≤ Y , and -6 βI ≤ Y ≤ 6 βI by (36). Therefore (31) implies for some C = C (Λ 2 ) &gt; 0,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, it follows from β ≥ 1, R 1 = Ar and C φ ≤ CR 2 1 that for some C = C ( A ) &gt; 0,

Plugging the above estimates into (38) shows

<!-- formula-not-decoded -->

Notice that by Ishii and Lions (1990, Lemma 3.1) and (36), there is C = C ( d ) &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore if CR α + η 2 ≤ Λ 1 β , we obtain

Now we pick β := R c 1 1 with c 1 := (1 + ε )(2 α + η ) + c 4 . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds when r ≥ 1 ( R 1 = Ar ) is large enough. By (39), there exist C,C ′ &gt; 0 depending only on C 0 , C 1 and η such that

Recall (35). Upon further assuming Ar = R 1 ≥ ( C ′ /ρ ) 1 /c 4 , we have

<!-- formula-not-decoded -->

This leads to the desired conclusion with A replaced by CA , where A,C &gt; 0 depend only on d, η , C 0 , C 1 , ρ . /square

- 3.4. Exploratory HJB equations: well-posedness and convergence. Now we apply the general PDE results established in the previous subsections to study the well-posedness of the exploratory HJB equation (1) for fixed λ &gt; 0, as well as the convergence of the solution as λ → 0 + .

We assume that the control space U is a non-empty open subset of some Euclidian space R l , and let ρ &gt; 0. Consider the operator associated with the exploratory HJB equation (1):

<!-- formula-not-decoded -->

and the operator associated with the classical HJB equation (5):

<!-- formula-not-decoded -->

We also make the following assumptions on the functions h ( · , · ) , b ( · , · ) , σ ( · , · ).

Assumption 1. There are positive γ r , γ r ∈ C 2 (0 , ∞ ) satisfying (18) and (19) such that the following hold:

- (i) For each r ≥ 1 , | h ( · , · ) | is bounded by γ r in B r ×U , and | b ( · , · ) | is bounded by γ r in B r ×U .
- (iii) There exist Λ 2 &gt; Λ 1 &gt; 0 such that Λ 1 I ≤ σ ( · , · ) σ ( · , · ) T ≤ Λ 2 I in R d ×U .
- (ii) For each r ≥ 1 and all u ∈ U , h ( · , u ) , b ( · , u ) and σ ( · , u ) are uniformly Lipschitz continuous with Lipschitz bound γ r in B r .
- (iv) h ( · , · ) , b ( · , · ) , σ ( · , · ) are locally uniformly continuous in R d ×U .

<!-- formula-not-decoded -->

- (v) We have

and the following holds locally uniformly in ( X,p,x ) ∈ S d × R d × R d :

The condition (42) is to ensure that F λ with λ ∈ (0 , 1) are well-defined, whereas the condition (43) is to guarantee F λ → F locally uniformly as λ → 0 + which is a reasonable requirement. If U is a bounded set, then assumption ( v ) holds trivially. Note that Assumption 1 rules out the LQ case (i.e. b ( · , · ) , σ ( · , · ) are linear and h ( · , · ) quadratic); but the corresponding exploratory and classical HJB equations for LQ can both be solved explicitly and the solutions are quadratic functions; see Wang et al. (2020). In other words, the LQ case can be solved separately and specially and hence is not our concern here.

<!-- formula-not-decoded -->

We have the following result by specializing the results in Subsections 3.2-3.3 to the operators F λ , F defined by (40)-(41).

Theorem 10. Let F λ , F be defined by (40) -(41) and Assumption 1 hold. Then the assumptions (a)-(d) hold uniformly for F λ , F for all λ ∈ (0 , 1) , with

<!-- formula-not-decoded -->

and F λ , F are concave in X . Consequently, the equation F λ ( ∇ 2 v λ , ∇ v λ , v λ , x ) = 0 (resp. F ( ∇ 2 v, ∇ v, v, x ) = 0 ) has a unique solution v λ (resp. v ) of sub-quadratic growth. Moreover,

- (ii) There exists C &gt; 0 such that sup B r | v ( x ) | + | v λ ( x ) | ≤ Cγ r for each r ≥ 1 .
- (i) v λ , v are locally C 2 ,α for some α ∈ (0 , 1) .
- (iii) v λ → v locally uniformly as λ → 0 + .

Proof. It is direct to check that Assumption 1 implies assumptions (a)-(c). To see (d), note that if U is a bounded set, F λ ( X,p,s,x ) → F ( X,p,s,x ) locally uniformly in X,p,s,x as λ → 0 + since h ( x, u ) , b ( x, u ) , σ ( x, u ) are locally uniformly continuous in u and uniformly continuous in x . If U is unbounded, we use (43) to get the convergence.

Clearly the operator F is concave in X according to Definition 4. Now we show that F λ is also concave in X . Let us write, for any fixed p, x ,

<!-- formula-not-decoded -->

and G = G ( X,u ) := exp( λ -1 g ( X,u )). Then

<!-- formula-not-decoded -->

Direct computation yields that for any N = ( N ij ) ∈ S d , where the last inequality is due to H¨ older's inequality and G &gt; 0. Therefore F λ is concave in X . All the conclusions now follow from Theorem 8. /square

<!-- formula-not-decoded -->

One can derive a convergence rate for v λ → v as λ → 0 + in the spirit of Theorem 9, but we chose not to present it in the above theorem because its expression would be overly complex for the general case. In the next section, we will derive a simple, explicit rate for a special application case - the temperature control problem.

So far we have focused our attention on the HJB equations. The connection to the control problems is stipulated in the following theorem.

Theorem 11. Consider the exploratory control problem (7) -(9) with the value function v λ . Let Assumption 1 hold, and assume that the SDE (12) is well-posed. Then v λ is the unique solution of sub-quadratic growth to the exploratory HJB equation (1) . Moreover, v λ is locally C 2 ,α for some α ∈ (0 , 1) , and

<!-- formula-not-decoded -->

where v is the value function of the classical control problem (3) -(4) and the unique solution of sub-quadratic growth to the classical HJB equation (5) .

Proof. Under Assumption 1, let v ′ λ be the unique solution to (1). According to Theorem 10 (ii), v ′ λ has polynomial growth. By a standard verification argument, we have v λ ( x ) ≤ v ′ λ ( x ) for all x ∈ R d . Since (12) is well-posed, the equality is achieved by the relaxed control π ∗ t ( · ) = π ∗ ( · , X λ, ∗ t ), namely, v λ ≡ v ′ λ . The remaining of the theorem follows readily from Theorem 10. /square

Theorem 11 indicates that the exploratory control problem (7)-(9) converges to the classical stochastic control problem (3)-(4) as the weight parameter λ → 0 + . The technical assumption needed is that the optimally controlled process ( X λ, ∗ t , t ≥ 0) defined by the SDE (12) is well-posed. If γ r = C (1 + r ) for some C &gt; 0 in Assumption 1, then it is easy to see that x → ˜ b ( x, π ∗ ( · , x )) is bounded and measurable, and x → ˜ σ ( π ∗ ( · , x )) is bounded, continuous and strictly elliptic. Classical theory of Stroock and Varadhan (1979) then implies that (12) is well-posed. However, if η ∈ (1 , 2), then b ( · , · ) and x → ˜ b ( x, π ∗ ( · , x )) are unbounded. Now, if it is true that the solution v λ to (1) is locally C 3 (under additional assumptions on h ( · , · ) , b ( · , · ) , σ ( · , · )), then we have that the functions x → ˜ b ( x, π ∗ ( · , x )) and x → ˜ σ ( π ∗ ( · , x )) are locally Lipschitz. In this case, (12) has a unique strong solution, hence well-posed, up to the explosion time τ ∞ := lim k →∞ inf { t ≥ 0 : | X λ, ∗ t | &gt; k } . Further non-explosion conditions (see e.g. Meyn and Tweedie (1993b)) ensure that τ ∞ = ∞ almost surely, leading to the well-posedness of (12).

## 4. Exploratory temperature control problem

In this section we apply the general results obtained in the previous section to the exploratory temperature control problem. This problem was formulated by Gao et al. (2020) for temperature control in the context of SA. In Section 4.1, we provide a brief background on this problem. A detailed analysis of the associated exploratory HJB equation is given in Section 4.2. There we derive an explicit convergence rate for the value function as the weight parameter tends to zero. Finally in Section 4.3, we study the steady state of the optimally controlled process of the problem.

4.1. Exploratory temperature control problem. To design an endogenous temperature control for SA, Gao et al. (2020) first consider the following stochastic control problem:

<!-- formula-not-decoded -->

Here, the temperature process ( β t , t ≥ 0) is taken as the control. Following Gao et al. (2020), we take the control space U = [ a, 1] for a fixed a ∈ (0 , 1) throughout this section. Note that the upper bound of U can be replaced by any positive number, while we require that the lower bound of U be away from 0 to guarantee a minimal effort for exploration.

By setting U = [ a, 1], h ( x, u ) = f ( x ), b ( x, u ) = -∇ f ( x ), σ ( x, u ) = √ 2 u , and substituting 'sup' with 'inf' in (5), we obtain the classical HJB equation of the temperature control

problem (44):

<!-- formula-not-decoded -->

It is then easily seen from the verification theorem that an optimal feedback control has the bang-bang form: β ∗ = 1 if Tr( ∇ 2 v ( x )) &lt; 0, and β ∗ = a if Tr( ∇ 2 v ( x )) ≥ 0. Using this temperature control scheme, one should switch between the highest temperature and the lowest one, depending on the sign of Tr( ∇ 2 v ( x )). As mentioned in the introduction, there are two disadvantages, one in theory and the other in application, of this bang-bang strategy:

- (1) Although theoretically optimal, this strategy is practically too rigid to achieve good performance as it only has two actions: a → 1 and 1 → a . It is too sensitive to errors which are inevitable in any real world application.
- (2) The corresponding optimally controlled dynamics is governed by the SDE:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

There is a subtle issue regarding the well-posedness of the SDE (46). Note that g is bounded and strictly elliptic. If ∇ f is assumed to be bounded, it follows from Exercise 12.4.3 in Stroock and Varadhan (1979) that (46) has a weak solution for all dimension d . However, the uniqueness in distribution may fail since g is discontinuous (see e.g. Safonov (1999) for an example). According to Exercises 7.3.3 and 7.3.4 in Stroock and Varadhan (1979), the uniqueness holds for d = 1 , 2. But it remains unknown whether the uniqueness in distribution is still valid for d ≥ 3.

There has been some literature on the uniqueness in distribution of SDEs with discontinuous diffusion coefficients via the martingale problem; see e.g. Bass and Pardoux (1987); Stramer and Tweedie (1997); Krylov (2004). In these works, it is assumed that the set of discontinuity has some special geometric structure. However, for the diffusion coefficient (47), the set of discontinuity is determined by the sign of Tr( ∇ 2 v ( x )), which is much more complex. By Theorem 8 below, ∇ 2 v is continuous so the set { Tr( ∇ 2 v ) &gt; 0 } (resp. { Tr( ∇ 2 v ) &lt; 0 } ) is open; but this condition alone cannot guarantee the uniqueness in distribution of (46).

To address the first disadvantage above, Gao et al. (2020) introduce the exploratory version of (44) in order to smooth out the temperature process. This way, a classical control ( β t , t ≥ 0) is replaced by a relaxed control π = ( π t ( · ) , t ≥ 0) over the control space U = [ a, 1], rendering the following exploratory dynamics:

<!-- formula-not-decoded -->

The exploratory temperature control problem is to solve

<!-- formula-not-decoded -->

where A ( x ) is the set of admissible controls specified by Definition 1.

By setting h ( x, u ) = f ( x ), b ( x, u ) = -∇ f ( x ) and σ ( x, u ) = √ 2 u , we get the corresponding exploratory HJB equation:

The corresponding optimal feedback control is

<!-- formula-not-decoded -->

which yields the optimally controlled process governed by the SDE:

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the diffusion coefficient, g λ , is now continuous , and √ 2 a ≤ g λ ( · ) ≤ 2. If ∇ f is assumed to be bounded, it follows from the classical theory of Stroock and Varadhan (1979) that (52) is well-posed. This is in stark contrast with the controlled dynamics (46) which is not necessarily well-posed. In summary, the optimal temperature control scheme of this exploratory formulation allows any level of temperature and renders a well-posed state process, thereby remedying simultaneously the two aforementioned disadvantages of the classical formulation.

To study the equation (50) and the process governed by (52), we make the following assumptions on the function f .

Assumption 2. The function f ∈ C 2 satisfies

- ( i ) There exists a constant C &gt; 0 such that

<!-- formula-not-decoded -->

- ( ii ) There exist χ &gt; 0 and R &gt; 0 such that

<!-- formula-not-decoded -->

Note that a combination of ( i ) and ( ii ) yields a linear growth of f . These conditions, in fact, guarantee that both the value function v λ and the optimal state process X λ, ∗ have good properties. We will see that ( i ) alone is sufficient for identifying the value function v λ as the solution to the HJB equation, and ( ii ) is essentially a Lyapunov/Poincar´ e condition which ensures the convergence of X λ, ∗ as λ → 0 + .

4.2. Analysis of the exploratory HJB equation. In this subsection, we apply the results in Section 3 to study (50). The corresponding operators are and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Specializing Assumption 1 to U = [ a, 1], h ( x, u ) = f ( x ), b ( x, u ) = -∇ f ( x ) and σ ( x, u ) = √ 2 u leads to the following assumption on f .

Assumption 3. Assume that f ∈ C 2 ( R d ) , and there are positive γ r , γ r ∈ C 2 (0 , ∞ ) such that for each r ≥ 1 ,

<!-- formula-not-decoded -->

where γ r , γ r satisfy (18) and (19) .

Assumption 3 basically demands a sub-quadratic growth on f and a sub-linear growth on |∇ f | . It is more general than Assumption 2-( i ). In particular, it recovers Assumption 2 -( i ) when γ r = C (1 + r ).

The following result is an easy corollary of Theorem 10.

Corollary 12. Let F, F λ be defined by (54) -(55) , and Assumption 3 hold. Then

- (i) There exists a unique solution v of sub-quadratic growth to the equation F ( ∇ 2 v, ∇ v, v, x ) = 0 , and v is locally uniformly C 2 ,α .
- (iii) There exists C ≥ 1 such that for all r ≥ 1 ,
- (ii) For each λ &gt; 0 , there exists a unique solution v λ of sub-quadratic growth to the equation F λ ( ∇ 2 v λ , ∇ v λ , v λ , x ) = 0 , and v λ is locally uniformly C 2 ,α .

<!-- formula-not-decoded -->

and, moreover, v λ → v locally uniformly as λ → 0 + .

Next we apply Theorem 9 to derive an explicit rate of convergence for v λ → v as λ → 0 + , by assuming that Assumption 3 holds with the choice of γ r = C (1 + r η ) for some η ∈ [0 , 2).

- (i) F and F λ satisfy the assumptions ( a ) -( c ) with γ r = C (1 + r η ) , and γ r = C (1 + r η -1 ) .

Lemma 13. Let Assumption 3 hold with γ r = C (1 + r η ) for some η ∈ [0 , 2) . Then

- (ii) The assumption ( d ) holds with

<!-- formula-not-decoded -->

where d is the dimesion of the state space.

Proof. The proof of ( i ) is the same as the one of Theorem 10, in which the expression of γ r follows from (19). The proof of ( ii ) follows from direct computations, and we will prove (57) for the case when z := Tr X/λ &gt; 0 the other case being similar. Notice that

If z ≥ 1 we have

<!-- formula-not-decoded -->

and if z ∈ (0 , 1) we have

<!-- formula-not-decoded -->

Therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the conclusion follows since d | X | ≥ | Tr X | .

In the following lemma, we present a point-wise bound of |∇ v | and |∇ v λ | .

Lemma 14. Let Assumption 3 hold with γ r = C (1 + r η ) for some η ∈ [0 , 2) . Then there exists C &gt; 0 such that for any r ≥ 1 we have

<!-- formula-not-decoded -->

Proof. We will only prove for v , and that for v λ is identical because F λ , λ &gt; 0, have uniformly elliptic second order terms, while the lower order terms are the same as F .

Fix r ≥ 1, and let u ( x ) := r -η v ( r -γ x ) with γ := max { η -1 , 0 } . According to Corollary 12, u is uniformly bounded in B 2 r 1+ γ , and it satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Thus, by the assumption of the lemma and γ ≥ η -1, we have for some C &gt; 0,

<!-- formula-not-decoded -->

This allows us to apply Theorem 2.1 in Lian et al. (2020) (see also Theorem 2.1 in ´ Swiech (1997)) to conclude that sup x ∈ B r 1+ γ |∇ u ( x ) | ≤ C for some C independent of r , completing the proof. /square

Finally, we state the convergence rate result, the proof of which follows from Theorem 9, Lemma 13 and Lemma 14.

Theorem 15. Let F, F λ be defined by (54) -(55) , and Assumption 3 hold with γ r = C (1+ r η ) for some η ∈ [0 , 2) . Also let v λ (resp. v ) be the unique solution of sub-quadratic growth to the equation F λ ( ∇ 2 v λ , ∇ v λ , v λ , x ) = 0 (resp. F ( ∇ 2 v, ∇ v, v, x ) = 0 ). Then there exists C &gt; 0 such that for all λ ∈ (0 , 1) and r ≥ 1 we have

<!-- formula-not-decoded -->

with c := 1 + min { (1 -η )(4 -η ) / 2 , 0 } .

Combining Theorem 11 and Theorem 15, we get the following result characterizing the value function of the exploratory temperature control problem and its convergence.

Corollary 16. Consider the exploratory temperature control problem (48) -(49) with value function v λ . Let Assumption 2(i) hold. Then v λ is the unique solution of sub-quadratic growth to the exploratory HJB equation (50) . Moreover, v λ is locally C 2 ,α for some α ∈ (0 , 1) , and there exists C &gt; 0 such that for all λ ∈ (0 , 1) and r ≥ 1 ,

<!-- formula-not-decoded -->

where v is the unique solution of sub-quadratic growth to the classical HJB equation (45) .

Because the constant C &gt; 0 in (58) is independent of λ ∈ (0 , 1) and r ≥ 1, we can minimize the right hand side of (58) with respect to r to get r min = λ -1 &gt; 1. With r min , (58) reduces to

<!-- formula-not-decoded -->

Note that for many real-world optimization problems, one can (and probably should ) restrict herself to a bounded set - however large it might be - containing all the 'important' states. Thus when λ is sufficiently small, the ball of radius 1 /λ contains these states of interest, and the leading term on the right hand side of (59) is λ ln(1 /λ ). Therefore, the estimate (59) essentially stipulates that v λ converges to v at the rate of λ ln(1 /λ ) as λ → 0 + .

4.3. Optimally controlled state process. In this subsection we consider the long time behavior of the optimal state process (52) of the exploratory temperature control problem.

We start by recalling some basics in stochastic stability. Consider the general diffusion process X = ( X t , t ≥ 0) in R d of form:

<!-- formula-not-decoded -->

where b : R d → R d is the drift, and σ : R d → R d × d is the diffusion (or covariance) matrix. Assuming that (60) is well-posed, let L be the infinitesimal generator of the diffusion process X defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and L ∗ be the corresponding adjoint operator given by where ψ : R d → R is a suitably smooth test function. The probability density ρ t ( · ) of the process X at time t then satisfies the Fokker-Planck equation:

<!-- formula-not-decoded -->

It is not always true that ρ t ( · ) converges as t → ∞ to a probability measure. But if b and σ satisfy some growth conditions, it can be shown that as t → ∞ , ρ t ( · ) converges in total variation distance to ρ ( · ) which is the stationary distribution (or steady state) of X . It is then easily deduced from (63) that ρ is characterized by the equation L ∗ ρ = 0. For instance, the overdamped Langevin equation with b ( x ) = -∇ f ( x ) and σ ( x ) = √ 2 β I is time-reversible, and the stationary distribution, under some growth condition on f , is the Gibbs measure where Z β := ∫ R d exp( -f ( x ) /β ) dx is the normalizing constant. However, for general b and σ , the stationary distribution ρ ( · ) may not have a closed-form expression. The standard references for stability of diffusion processes are Ethier and Kurtz (1986); Meyn and Tweedie (1993a,b). We record a result on the ergodicity of diffusion processes.

<!-- formula-not-decoded -->

Lemma 17. Assume that b : R d → R d is bounded, and σ : R d → R d × d is bounded and strictly elliptic, and that there exists 0 &lt; α ≤ 1 such that b, σ are locally uniformly α -H¨ older continuous, i.e. for each R &gt; 0 there is a constant C R &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then (60) is well-posed, i.e. it has a weak solution which is unique in distribution. Assume further that there exist M 1 &gt; 0 , M 2 &lt; ∞ , a compact set C ⊂ R d , and a function V : R d → [1 , ∞ ) with V ( x ) →∞ as | x |→∞ such that

Then the (unique) distribution of the solution to (60) converges in total variation distance to its unique stationary distribution as t →∞ .

Proof. The fact that the diffusion process (60) is well-posed follows from Theorem 6.2 in Stroock and Varadhan (1979). Recall that a Borel set C ⊂ R d is called petite if there exist a distribution q on R + and a nonzero Borel measure ν on R d such that ∫ ∞ 0 P x ( X t ∈ A ) q ( dt ) ≥ ν ( A ) for all x ∈ C and all Borel sets A ⊂ R d . Under the condition (66) with a petite set C , Theorems 2.1 and 2.2 in Tang (2019) imply that the diffusion process is positive Harris recurrent, and converges in total variation distance to its unique stationary distribution. Further by Theorem 2.1 in Stramer and Tweedie (1997), the diffusion process is a Lebesgue irreducible (and T -) process. However, according to Theorem 4.1 in Meyn and Tweedie (1993a), each compact set is petite, which concludes the proof. /square

The following theorem describes the long time behavior of the optimal state process (52) of the exploratory temperature control problem (48)-(49). Recall that || · || TV denotes the total variation distance between probability measures.

Theorem 18. Let Assumption 2 hold. Then we have:

- (i) For each λ &gt; 0 , the process ( X λ, ∗ t , t ≥ 0) converges in total variation distance to its unique stationary distribution as t →∞ .
- (ii) For each λ &gt; 0 , let ρ λ be the stationary distribution of the process ( X λ, ∗ t , t ≥ 0) . Fix θ &gt; 0 and δ &gt; 0 . Then there exists c &gt; 0 such that

<!-- formula-not-decoded -->

Consequently, ( X λ, ∗ t , t ≥ 0) does not converge in probability to any θ ∈ R d (and in particular to argmin f ( x ) ).

/negationslash

- (iii) Let G β , β &gt; 0 , be the Gibbs measure of the form (64) . Then for each λ &gt; 0 , ρ λ = G β for any β &gt; 0 . Moreover, there exists c &gt; 0 such that

<!-- formula-not-decoded -->

Proof. ( i ) Note that X λ, ∗ is a diffusion process with b ( x ) = -∇ f ( x ) and σ ( x ) = g λ ( x ) I . It is clear that b is bounded, and σ is bounded and strictly elliptic. By Assumption 2-( ii ), |∇ 2 f | is bounded, and thus b = -∇ f satisfies the H¨ older condition (65). By Corollary 12, v λ is locally C 2 . It follows that g λ is locally H¨ older continuous, and so is σ = g λ I . It is easy to see that

<!-- formula-not-decoded -->

By Assumption 2-( ii ), the condition (66) is satisfied with M 1 = χ and M 2 = sup x ∈ B R L f ( x ). It suffices to apply Lemma 17 to conclude.

( ii ) This follows from the fact that g λ is bounded away from 0. We argue by contradiction that inf λ&gt; 0 ρ λ ( { x : | x -θ | &gt; δ } ) = 0. Then for ε &gt; 0, there exists λ &gt; 0 such that ρ λ ( { x : | x -θ | &gt; δ } ) &lt; ε . By part ( i ), ( X λ, ∗ t , t ≥ 0) converges in total variation distance to ρ λ . So for t sufficiently large, we have

<!-- formula-not-decoded -->

On the other hand, b = -∇ f and σ = g λ I are H¨ older continuous, and σσ T ≥ 2 aI with 2 a independent of λ . By Aronson's comparison theorem (see Aronson (1967)),

<!-- formula-not-decoded -->

where c, C &gt; 0 are constants independent of t and λ . By taking ε &gt; 0 to be arbitrarily small, the estimates (67) and (68) lead to a contradiction.

/negationslash

- ( iii ) We first prove that ρ λ = G β for any β &gt; 0. We argue by contradiction that ρ λ = G β for some β &gt; 0. Recall from (62) that the adjoint operator of the optimal controlled process

L ρ λ = 0, we get

<!-- formula-not-decoded -->

On the other hand, ρ λ = G β is the stationary distribution of the overdamped Langevin equation dX t = -∇ f ( X t ) dt + √ 2 βdB t ; so it satisifies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Comparing (69) and (70) yields i.e. g λ ρ λ -βρ λ is a harmonic function. By Assumption 2-( ii ), f ( x ) → + ∞ as | x | → ∞ . Thus, g λ ρ λ -βρ λ → 0 as | x |→∞ . According to Liouville's theorem, any bounded harmonic function is constant (see e.g. Theorem 8, Chapter 2 in Evans (2010)). So g λ ρ λ -βρ λ ≡ 0, and hence g λ ≡ β . Injecting to (53), we see that v λ only depends on a , β and λ . This contradicts the HJB equation (50) where v λ also depends on f .

/negationslash

Now we prove that ρ λ is bounded away from any Gibbs measure G β . We argue by contradiction that inf β&gt; 0 || ρ λ - G β || TV = 0. Then there exists a sequence { β n } n ≥ 1 such that || ρ λ -G β n || TV → 0 as n →∞ . This is impossible if lim n →∞ β n = ∞ , since G β does not converge to a probability measure as β → ∞ . Thus, we can extract a convergent subsequence { β ′ n } n ≥ 1 from { β n } n ≥ 1 . If lim n →∞ β ′ n = β ′ &gt; 0, this implies that ρ λ = G β ′ which contradicts the fact that ρ λ = G β for any β &gt; 0. If lim n →∞ β ′ n = 0, then ρ λ is concentrated on argmin f , whose validity is ruled out by part ( ii ). /square

Theorem 18 indicates that, with a fixed level of exploration, the optimally controlled process ( X λ, ∗ t , t ≥ 0) does have a stationary distribution. This provides a theoretical justification to the SA algorithm devised by Gao et al. (2020) based on discretizing (52). The result that this

stationary distribution is not a Dirac mass on the minimizer of f is expected theoretically because (52) is a genuine diffusion process due to its strict ellipticity. It is indeed preferred from an exploration point of view because the essence of exploration is to involve as many states as possible instead of just focusing on the single state of the minimizer, in the same spirit of the classical overdamped Langevin diffusion that converges to the Gibbs measure instead of the Dirac one. The fact that the stationary distribution of (52) is not a Gibbs measure is the most intriguing one; it suggests the possibility of a more variety of target measures - beyond Gibbs measures - when it comes to SA for non-convex optimization.

Theorem 18 does not provide a convergence rate for ( X λ, ∗ t , t ≥ 0) to converge to its stationary distribution. This is due to the assumption that |∇ f | is bounded, which is a sufficient condition for the well-posedness of the process ( X λ, ∗ t , t ≥ 0) in the verification argument. If we can relax this condition while the process ( X λ, ∗ t , t ≥ 0) is still well-posed, then more can be said about the convergence. For instance, assume that |∇ 2 f | is bounded, g λ is locally Lipschitz, and

<!-- formula-not-decoded -->

where φ is a strictly concave function increasing to infinity (e.g. φ ( s ) = s α for some 0 &lt; α &lt; 1). In this case, let H φ ( s ) = ∫ s 1 ds φ ( s ) . Then there exists C &gt; 0 such that (Bakry et al., 2008)

<!-- formula-not-decoded -->

So ( X λ, ∗ t , t ≥ 0) converges to its stationary distribution with a sub-exponential rate. If instead of (71) we assume that

<!-- formula-not-decoded -->

for some M &gt; 0, then there exist c &gt; 0 and C &gt; 0 such that

<!-- formula-not-decoded -->

That is, ( X λ, ∗ t , t ≥ 0) converges exponentially to its stationary distribution. See e.g. Bakry et al. (2008) for further discussions on the convergence rate of diffusion processes. This means that if we can relax the well-posedness condition (e.g. removing the boundedness assumption on |∇ f | so that either (71) or (72) is satisfied), then we can derive a convergence rate for the optimally controlled process ( X λ, ∗ t , t ≥ 0) of the exploratory temperature control problem as t →∞ .

To conclude this subsection, we study the stability of stationary distributions of ( X λ, ∗ t , t ≥ 0) with different λ 's. For a general analysis on the stability of stationary distributions of diffusion processes with different drift and covariance coefficients, see Bogachev et al. (2014, 2017, 2018). The idea is to bound the total variation distance between stationary distributions in terms of diffusion parameters. We recall a lemma which is due to Bogachev et al. (2018).

Lemma 19. Let ( b 1 , σ 1 ) and ( b 2 , σ 2 ) be pairs of drift and covariance coefficients associated with the diffusion process (60) . For each k = 1 , 2 , assume that b k is bounded and measurable, and σ k is bounded, strictly elliptic and globally Lipschitz. Then the diffusion process associated

with ( b k , σ k ) has a unique stationary distribution ρ k ( dx ) = ρ k ( x ) dx . For 1 ≤ i ≤ d , let

<!-- formula-not-decoded -->

and Φ := ( σ 1 σ T 1 -σ 2 σ T 2 ) ∇ ρ 2 ρ 2 -( φ 1 -φ 2 ) . Assume further that there exist κ &gt; 0 , M &gt; 0 and R &gt; 0 such that

<!-- formula-not-decoded -->

Then there exists C &gt; 0 such that

<!-- formula-not-decoded -->

Theorem 20. Let Assumption 2 hold, and assume further that there exist κ &gt; 0 , M &gt; 0 and R &gt; 0 such that

<!-- formula-not-decoded -->

and that the solution v λ to (50) is C 3 with bounded third derivatives. For each λ &gt; 0 , let ρ λ ( dx ) be the stationary distribution of the optimal state process governed by (52) . Then

<!-- formula-not-decoded -->

Proof. We apply Lemma 19 with b 1 ( x ) = b 2 ( x ) = -∇ f ( x ), and σ 1 ( x ) = g λ ′ ( x ) I , σ 2 ( x ) = g λ ( x ) I . In this case,

<!-- formula-not-decoded -->

It is easy to see that Φ( x ) → 0 as λ ′ → λ . Since v λ has bounded third derivatives, we have g λ is globally Lipschitz. Because b 2 = -∇ f is bounded and σ 2 = g λ I is bounded, Lipschitz and strict elliptic, it follows from Theorem 3.1.2 in Bogachev et al. (2015) that

<!-- formula-not-decoded -->

By the dominated convergence theorem, we get ∫ R d | Φ( x ) | ρ λ ( dx ) → 0 as λ ′ → λ . It suffices to apply Lemma 19 to conclude. /square

The assumption (73) is a version of the dissipative condition, which is standard in Langevin sampling and optimization. The assumption that |∇ f | is bounded restricts the range of the dissipative exponent κ to (0 , 1]. The only technical assumption in Theorem 20 is that the solution v λ to the exploratory HJB equation (50) is three times continuously differentiable with bounded third derivatives. It implies that ∇ 2 v λ is continuously differentiable and is globally Lipschitz, which is stronger than the result of Theorem 8 that ∇ 2 v is locally H¨ older continuous. It is interesting to know whether Assumption 2 (possibly with some additional conditions on f ) implies the boundedness of third derivatives of the solution to (50).

## 5. Finite Time Horizon

The exploratory control problem (7)-(9) is a relaxed control problem in an infinite time horizon, and the associated exploratory HJB equation is, therefore, elliptic. Nevertheless, the arguments in the paper can be adapted, to the extent they can, to the finite time setting where the HJB equation is parabolic.

We follow the formulation in Zhou (2021). Fix T &gt; 0, and consider the stochastic control problem whose value function is

∣ where h 1 : [0 , T ] × R d ×U → R and h 2 : R d → R are reward functions, and A 0 ( t, x ) is the set of admissible classical controls with respect to X u t = x . The state dynamics is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note here b, σ, h 1 depend on t explicitly.

Denote by ∂ t the partial derivative in t , and by ∇ x and ∇ 2 x the gradient and Hessian in x respectively. The classical HJB equation associated with the problem (75)- (76) is

(77)

<!-- formula-not-decoded -->

It is known that a smooth solution to the HJB equation (77) gives the value function (75). The optimal control at time t is u ∗ t = u ∗ ( t, X ∗ t ), where u ∗ : [0 , T ] × R d →U is a deterministic mapping obtained by solving the 'sup u ∈U ' term in (77), and the optimally controlled process is governed by

<!-- formula-not-decoded -->

provided that it is well-posed.

The exploratory control problem with finite time horizon is to solve an entropy-regularized relaxed control problem whose value function is

<!-- formula-not-decoded -->

∣ where A ( t, x ) is the set of distributional control processes defined similarly to the infinite horizon setting, and the exploratory dynamics is with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the exploratory HJB equation is the following nonlinear parabolic PDE:

<!-- formula-not-decoded -->

and the optimal state process is governed by provided that it is well-posed.

To identify the value function (79) (resp. (75)) as the solution to the HJB equation (83) (resp. (77)), the verification theorem requires that these solutions to be C 1 , 2 t,x . However, when U = [ 1 2 , 1], h 1 ( x, u ) = 0, b ( x, u ) = 0 and σ ( x, u ) = √ 2 uI , a result of Caffarelli and Stefanelli (2008) shows that the solutions to (77) are not C 1 , 2 t,x . For general fully nonlinear parabolic PDEs, the solution is only known to be C α, 1+ α t,x for some α ∈ (0 , 1). We record this fact in the following theorem.

<!-- formula-not-decoded -->

Theorem 21. Let Assumption 1 hold for h 1 ( · , · , · ) , b ( · , · , · ) , σ ( · , · , · ) , and assume that h 2 ( · ) satisfies

<!-- formula-not-decoded -->

- (i) v λ , v are C α, 1+ α t,x locally uniformly in [0 , T ) × R d for some α ∈ (0 , 1) .

Then the HJB equation (83) (resp. (77) ) has a unique solution v λ (resp. v ) of sub-quadratic growth for t ∈ [0 , T ] . Moreover,

- (ii) There exists C &gt; 0 such that sup x ∈ B r ,t ∈ [0 ,T ] ( | v ( t, x ) | + | v λ ( t, x ) | ) ≤ Cγ r for each r ≥ 1 .
- (iii) v λ → v locally uniformly as λ → 0 + .

We refer to Wang (1992a,b) and Crandall et al. (2000) for the interior point-wise regularity estimate for fully nonlinear parabolic PDEs. To close the gap between what the verification theorem requires for the regularity of HJB equations and what is known for the regularity of general fully nonlinear parabolic PDEs, it remains a significant open question to find proper assumptions on h 1 ( · , · , · ) , b ( · , · , · ) , σ ( · , · , · ) under which the HJB equations (83) and (77) have unique C 1 , 2 t,x solutions?

Onthe other hand, under a further assumption that σ does not depend on u , Gozzi and Russo (2006a,b) showed that the verification theorem only requires the solution to the HJB equation to be C 0 , 1 t,x . Combining this result with Theorem 21, we obtain the following analog of Theorem 11 for the exploratory control problem with a finite time horizon.

Theorem 22. Consider the exploratory control problem (79) -(80) whose value function is v λ . Let Assumption 1 hold for h 1 ( · , · , · ) , b ( · , · , · ) , σ ( · , · , · ) with σ ( · , · , · ) ≡ σ being constant, and assume that h 2 ( · ) satisfies

Further assume that the SDE (84) is well-posed. Then v λ is the unique solution of subquadratic growth to the exploratory HJB equation (83) . Moreover, v λ is locally C α, 1+ α t,x for some α ∈ (0 , 1) , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where v is the value function of the classical control problem (75) -(76) and the unique solution of sub-quadratic growth to the classical HJB equation (77) .

## 6. Conclusions

In this paper, we study the exploratory HJB equation arising from a continuous-time reinforcement learning framework - that of the exploratory control - put forth by Wang et al. (2020). We establish the well-posedness and regularity of its solution under general assumptions on the system dynamics parameters. This allows for identifying the value function of the exploratory control problem in general cases, which goes beyond the LQ setting. We also establish a connection between the exploratory control problem and the classical stochastic control problem by showing that the value function of the former converges to that of the latter as the weight parameter for exploration tends to zero. We then apply our general theory to a special example - the exploratory temperature control problem originally introduced by Gao et al. (2020) as a variant of SA. We provide a detailed analysis of the problem, with an explicit rate of convergence derived as the weight parameter vanishes. We also consider the long time behavior of the associated optimally controlled process, and study properties of its stationary distribution. The tools that we develop in this paper encompass stochastic control theory, partial differential equations and probability theory.

There are many important open, if technical, questions. First, we have proved in Theorem 10 that the exploratory HJB equation (1) has a unique smooth solution if Assumption 1 holds. In particular, the drift b is allowed to have sub-linear growth in x . However, in order to identify the value function of the exploratory control problem as the solution to (1), one needs the SDE (12) to be well-posed. This is satisfied if b is bounded. The question now is what assumptions, in addition to Assumption 1 and in particular the sub-linear growth of b ( · , u ), are required to ensure the well-posedness of (12). A related question is whether we have the well-posedness of (1) and (5) for b ( · , u ) beyond sub-linear growth (e.g. of linear growth or polynomial growth). If the answer to the first question is positive, then we will have a complete characterization of the exploratory control problem for sub-linear b 's. In the case of the exploratory temperature control problem, we will then no longer need to impose a bounded restriction on b = |∇ f | . As discussed after Theorem 18, we may then specify a convergence rate for the optimally controlled process with a Lyapunov condition.

Second, in the study of stability of stationary distributions of the optimal state processes with different λ 's (Theorem 20), we make a technical assumption that v λ has bounded third derivatives. It is challenging, yet interesting, to know under what conditions on f this assumption holds. It can be shown using the arguments in Section 3 that for bounded f with all bounded derivatives, the solution to the exploratory HJB equation (50) has such a solution. To completely solve the stability problem, a first step is to prove/disprove whether such a solution exists for f of linear growth. More generally, one can ask under what conditions on h ( · , · , · ) , b ( · , · , · ) , σ ( · , · , · ) does (1) have a unique (viscosity) solution which is C 3 with bounded third derivatives. As discussed after Theorem 11, if (1) has a unique C 3 solution, then the SDE (12) is well-posed under additional non-explosion conditions.

Acknowledgement: Tang gratefully acknowledges financial support through an NSF grant DMS-2113779 and through a start-up grant at Columbia University. Zhou gratefully acknowledges financial supports through a start-up grant at Columbia University and through the Nie Center for Intelligent Asset Management.

## References

- S. N. Armstrong and H. V. Tran. Viscosity solutions of general viscous Hamilton-Jacobi equations. Math. Ann. , 361(3-4):647-687, 2015.
- D. G. Aronson. Bounds for the fundamental solution of a parabolic equation. Bull. Amer. Math. Soc. , 73:890-896, 1967.
- D. Bakry, P. Cattiaux, and A. Guillin. Rate of convergence for ergodic continuous Markov processes: Lyapunov versus Poincar´ e. J. Funct. Anal. , 254(3):727-759, 2008.
- R. F. Bass and E. Pardoux. Uniqueness for diffusions with piecewise constant coefficients. Probab. Theory Related Fields , 76(4):557-572, 1987.
- V. I. Bogachev, A. I. Kirillov, and S. V. Shaposhnikov. The Kantorovich and variation distances between invariant measures of diffusions and nonlinear stationary Fokker-PlanckKolmogorov equations. Math. Notes , 96(5-6):855-863, 2014.
- V. I. Bogachev, N. V. Krylov, M. R¨ ockner, and S. V. Shaposhnikov. Fokker-PlanckKolmogorov equations , volume 207 of Mathematical Surveys and Monographs . American Mathematical Society, Providence, RI, 2015.
- V. I. Bogachev, A. I. Kirillov, and S. V. Shaposhnikov. Distances between stationary distributions of diffusions and the solvability of nonlinear Fokker-Planck-Kolmogorov equations. Teor. Veroyatn. Primen. , 62(1):16-43, 2017.
- V. I. Bogachev, M. R¨ ockner, and S. V. Shaposhnikov. The Poisson equation and estimates for distances between stationary distributions of diffusions. J. Math. Sci. (N.Y.) , 232(3, Problems in mathematical analysis. No. 92 (Russian)):254-282, 2018.
- L. A. Caffarelli and X. Cabr´ e. Fully nonlinear elliptic equations , volume 43 of American Mathematical Society Colloquium Publications . American Mathematical Society, Providence, RI, 1995.
- L. A. Caffarelli and U. Stefanelli. A counterexample to C 2 , 1 regularity for parabolic fully nonlinear equations. Comm. Partial Differential Equations , 33(7-9):1216-1234, 2008.
- I. Capuzzo-Dolcetta, F. Leoni, and A. Vitolo. The Alexandrov-Bakelman-Pucci weak maximum principle for fully nonlinear equations in unbounded domains. Comm. Partial Differential Equations , 30(10-12):1863-1881, 2005.
- M. G. Crandall, H. Ishii, and P.-L. Lions. User's guide to viscosity solutions of second order partial differential equations. Bull. Amer. Math. Soc. (N.S.) , 27(1):1-67, 1992.
- M. G. Crandall, M. Kocan, and A. ´ Swiech. L p -theory for fully nonlinear uniformly parabolic equations. Comm. Partial Differential Equations , 25(11-12):1997-2053, 2000.
- S. N. Ethier and T. G. Kurtz. Markov processes: Characterization and Convergence . Wiley Series in Probability and Mathematical Statistics. John Wiley &amp; Sons, Inc., 1986.
- L. C. Evans. Partial differential equations , volume 19 of Graduate Studies in Mathematics . American Mathematical Society, Providence, RI, second edition, 2010.
- D. Firoozi and S. Jaimungal. Exploratory LQG mean field games with entropy regularization. 2020. arXiv:2011.12946.

- W. H. Fleming and H. M. Soner. Controlled Markov processes and viscosity solutions , volume 25 of Stochastic Modelling and Applied Probability . Springer, New York, second edition, 2006.
- X. Gao, Z. Q. Xu, and X. Y. Zhou. State-dependent temperature control for Langevin diffusions. 2020. arXiv:2005.04507.
- D. Gilbarg and N. S. Trudinger. Elliptic partial differential equations of second order , volume 224 of Grundlehren der Mathematischen Wissenschaften [Fundamental Principles of Mathematical Sciences] . Springer-Verlag, Berlin, second edition, 1983.
- F. Gozzi and F. Russo. Verification theorems for stochastic optimal control problems via a time dependent Fukushima-Dirichlet decomposition. Stochastic Process. Appl. , 116(11): 1530-1562, 2006a.
- F. Gozzi and F. Russo. Weak Dirichlet processes with a stochastic control perspective. Stochastic Process. Appl. , 116(11):1563-1583, 2006b.
- X. Guo, R. Xu, and T. Zariphopoulou. Entropy regularization for mean field games with learning. 2020. arXiv:2010.00145.
- H. Ishii and P.-L. Lions. Viscosity solutions of fully nonlinear second-order elliptic partial differential equations. J. Differential Equations , 83(1):26-78, 1990.
- S. Koike and O. Ley. Comparison principle for unbounded viscosity solutions of degenerate elliptic PDEs with gradient superlinear terms. J. Math. Anal. Appl. , 381(1):110-120, 2011.
- N. V. Krylov. On weak uniqueness for some diffusions with discontinuous coefficients. Stochastic Process. Appl. , 113(1):37-64, 2004.
- Y. Lian, L. Wang, and K. Zhang. Pointwise regularity for fully nonlinear elliptic equations in general forms. 2020. arXiv:2012.00324.
- S. P. Meyn and R. L. Tweedie. Stability of Markovian processes. III. Foster-Lyapunov criteria for continuous-time processes. Adv. in Appl. Probab. , 25(3):518-548, 1993a.
- S. P. Meyn and R. L. Tweedie. Stability of Markovian processes. II. Continuous-time processes and sampled chains. Adv. in Appl. Probab. , 25(3):487-517, 1993b.
- M. V. Safonov. Nonuniqueness for second-order elliptic equations with measurable coefficients. SIAM J. Math. Anal. , 30(4):879-895, 1999.
- O. Stramer and R. L. Tweedie. Existence and stability of weak solutions to stochastic differential equations with non-smooth coefficients. Statist. Sinica , 7(3):577-593, 1997.
- D. W. Stroock and S. R. S. Varadhan. Multidimensional diffusion processes , volume 233 of Grundlehren der Mathematischen Wissenschaften [Fundamental Principles of Mathematical Sciences] . Springer-Verlag, Berlin-New York, 1979.
- R. S. Sutton and A. G. Barto. Reinforcement learning: an introduction . Adaptive Computation and Machine Learning. MIT Press, Cambridge, MA, second edition, 2018.
- A. ´ Swiech. W 1 ,p -interior estimates for solutions of fully nonlinear, uniformly elliptic equations. Adv. Differential Equations , 2(6):1005-1027, 1997.
- W. Tang. Exponential ergodicity and convergence for generalized reflected Brownian motion. Queueing Syst. , 92(1-2):83-101, 2019.
- H. Wang and X. Y. Zhou. Continuous-time mean-variance portfolio selection: a reinforcement learning framework. Math. Finance , 30(4):1273-1308, 2020.
- H. Wang, T. Zariphopoulou, and X. Y. Zhou. Reinforcement learning in continuous time and space: A stochastic control approach. J. Mach. Learn. Res. , 21:1-34, 2020.
- L. Wang. On the regularity theory of fully nonlinear parabolic equations. I. Comm. Pure Appl. Math. , 45(1):27-76, 1992a.

- L. Wang. On the regularity theory of fully nonlinear parabolic equations. II. Comm. Pure Appl. Math. , 45(2):141-178, 1992b.
- J. Yong and X. Y. Zhou. Stochastic controls - Hamiltonian systems and HJB equations , volume 43 of Applications of Mathematics (New York) . Springer-Verlag, New York, 1999.

X. Y. Zhou. Curse of optimality, and how do we break it. 2021. SSRN:3845462.

Department of Industrial Engineering and Operations Research, Columbia University.

Email address :

wt2319@columbia.edu

Department of Mathematics, University of California, San Diego.

Email address :

yzhangpaul@ucsd.edu

Department of Industrial Engineering and Operations Research, Columbia University.

Email address :

xz2574@columbia.edu