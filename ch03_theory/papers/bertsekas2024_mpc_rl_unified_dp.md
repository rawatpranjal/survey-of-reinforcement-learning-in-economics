## A Unified Framework Based on Dynamic

## Model Predictive Control and Reinforcement Learning: Programming

Dimitri P. Bertsekas ∗

∗ Arizona State University, Tempe, AZ USA

Abstract: In this paper we describe a new conceptual framework that connects approximate Dynamic Programming (DP), Model Predictive Control (MPC), and Reinforcement Learning (RL). This framework centers around two algorithms, which are designed largely independently of each other and operate in synergy through the powerful mechanism of Newton's method. We call them the off-line training and the on-line play algorithms. The names are borrowed from some of the major successes of RL involving games; primary examples are the recent (2017) AlphaZero program (which plays chess, [SHS17], [SSS17]), and the similarly structured and earlier (1990s) TD-Gammon program (which plays backgammon, [Tes94], [Tes95], [TeG96]). In these game contexts, the off-line training algorithm is the method used to teach the program how to evaluate positions and to generate good moves at any given position, while the on-line play algorithm is the method used to play in real time against human or computer opponents. Significantly, the synergy between off-line training and on-line play also underlies MPC (as well as other major classes of sequential decision problems), and indeed the MPC design architecture is very similar to the one of AlphaZero and TD-Gammon. This conceptual insight provides a vehicle for bridging the cultural gap between RL and MPC, and sheds new light on some fundamental issues in MPC. These include the enhancement of stability properties through rollout, the treatment of uncertainty through the use of certainty equivalence, the resilience of MPC in adaptive control settings that involve changing system parameters, and the insights provided by the superlinear performance bounds implied by Newton's method.

To be published in Proc. of IFAC NMPC, Kyoto, August 2024

Keywords: Model Predictive Control, Adaptive Control, Dynamic Programming, Reinforcement Learning, Newton's Method

## 1. INTRODUCTION

We will describe a conceptual framework for approximate DP, RL, and their connections to MPC, which was first presented in the author's recent books [Ber20] and [Ber22a]. The present paper borrows heavily from these books, the course textbook [Ber23], the overview papers [Ber21a], [Ber22c], as well as recent research by the author and his collaborators. 1

Our framework is very broadly applicable thanks to the generality of the DP methodology on which it rests. This generality allows arbitrary state and control spaces, thus facilitating a free movement between continuous-space infinite-horizon formulations (such as those arising in control system design and MPC), discrete-space finite-horizon problem formulations (such as those arising in games and integer programming), and mixtures thereof that involve both continuous and discrete decision variables.

1 Special thanks are due to Yuchao Li for extensive helpful interactions relating to many of the topics discussed in this paper. Early discussions on MPC with Moritz Diehl were greatly appreciated. The suggestions of Manfred Morari and James Rawlings, as well as those of the reviewers, were also very much appreciated.

To present our framework, we will first focus on a class of deterministic discrete-time optimal control problems, which underlie typical MPC formulations. In subsequent sections, we will indicate how the principal conceptual components of our framework apply to problems that involve stochastic as well as set membership uncertainty, and how they impact the effectiveness of MPC for indirect adaptive control.

## 1.1 An MPC Problem Formulation

The theory and applications of MPC has undergone extensive development, since the early days of optimal control, thanks to research efforts from several scientific communities. 2 The early papers by Clarke, Mohtadi, and Tuffs

2 The idea underlying MPC is on-line optimization with a truncated rolling horizon and a terminal cost function approximation. This idea has arisen in several contexts, motivated by different types of applications. It has been part of the folklore of the optimal control and operations research literature, dating to the 1960s and 1970s. Simultaneously, it was used in important chemical process control applications, where the name 'model predictive control' (or 'modelbased predictive control') and the related name 'dynamic matrix control' were introduced. The term 'predictive' arises often in this

[CMT87a], [CMT87b], Keerthi and Gilbert [KeG88], and Mayne and Michalska [MaM88], attracted significant attention. Surveys, which give many of the early references, were given by Morari and Lee [MoL99], Mayne et al. [MRR00], Findeisen et al. [FIA03], and Mayne [May14]. Textbooks such as Maciejowski [Mac02], Goodwin, Seron, and De Dona [GSD06], Camacho and Bordons [CaB07], Kouvaritakis and Cannon [KoC16], Borrelli, Bemporad, and Morari [BBM17], Rawlings, Mayne, and Diehl [RMD17], and Rakovic and Levine [RaL18], collectively provide a comprehensive view of the MPC methodology.

More recent works have aimed to integrate 'learning' into MPC, similar to the practices of the RL and AI communities. This line of research is very active at present; for some representative papers, see [CLD19], [GrZ19], [Rec19], [CFM20], [HWM20], [MGQ20], [BeP21], [KRW21], [BGH22], [CWA22], [GrZ22], [MDT22], [MJR22], [SKG22], and [DuM23].

To provide an overview of the main ideas of our framework, let us consider a deterministic stationary discrete-time system of the form

<!-- formula-not-decoded -->

where x k and u k are the state and control at time k , taking values in some spaces X and U . We consider stationary feedback policies µ , whereby at a state x we apply control u = µ ( x ), subject to the constraint that µ ( x ) must belong to a given set U ( x ) for each x .

1

The cost function of µ , starting from an initial state x 0 is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α ∈ (0 , 1] is a discount factor, and

∈

x

f

X,

+

J

J

X

˜

/lscript

-

g

‖

-

‖

∈

)}

(

∗

αJ

) +

(

x

U

u

)

(

min

/lscript min

∈

) =

∈

u

min

k

x

∗

(

J

˜

∈

U

{

g

(

-

m

x

(

)

f

(

, u x, u

k

)

+

,

) +

Approximation Error

Approximation Error

µ

u

k

,...,u

+

/lscript

-

1

∑

x, u

1

m

α

=0

)

x

(

g

,

x, u

)

) +

α

J

f

(

x, u arg

{

<!-- image -->

<!-- formula-not-decoded -->

J

(

˜

˜

x

k

+

m

m

α

/lscript

˜

J

(

x

k

(

x, u

)

)}

(

)

-

‖

∗

;

1

,

) +

x, u

(

x, u

Input (Control) Output (Function of the State) Changing Fi Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) min u ∈ U ( x ) { g ( x, u ) u k At x ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, min u ∈ U ( x ) { g ( x u k At x Input (Control) Output (Function of the State) Cha Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Stability of policies is of paramount importance in MPC. In particular, the issue of stability was addressed theoretically by Keerthi and Gilbert [KeG88], and stability issues have been discussed in detail in the overview paper by Mayne et al. [MRR00]. A stability analysis with discrete constraint sets was given by Rawlings and Risbeck [RaR17]. The paper by Krener [Kre19] considers methods to estimate the optimal cost function for use as terminal cost function, aiming to achieve stabilization with MPC lookahead that is as small as possible.

{

u

## J ∗ ( x ) = min u ∈ U ( x ) { g J ∗ ( x ) = min u ∈ U ( x Fig. 1. Illustration of approximation in value space with one-step lookahead.

Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . Input (Control) Output (Functi Input (Control) Output (Fu Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . In the context of the present paper, however, because X and U can be arbitrary sets, it is necessary to use a more general line of analysis and a nontraditional definition of stability. In particular, we say that a policy µ is stable if

Transformer He

Transforme

1

-

2

d

<!-- formula-not-decoded -->

-

d

d θ x l d θ

, . . . , u

1

S

, . . . ,

-

+1

N

+

1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 -d 1 1 d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ u k +1 Cost Function Approximation k Region of convergence (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 Time k +1 Time k + m i j Time 0 Time k Region of convergence (˜ u 0 , . . . , ˜ u k -1 , u k , u k Time k +1 Time k 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ u k +1 Cost Function Approximation For problems where α = 1, this definition of stability is qualitatively similar to traditional definitions of stability in control theory/MPC contexts, including linear-quadratic problems (to be used later for visualization purposes). Our subsequent discussion of stability implicitly assumes such a context, and may not be meaningful in other contexts, such as games, discrete optimization, cases where α &lt; 1, etc. Note that J ∗ ( x ) is finite for all x if there exists at least one stable policy, which we will assume in this paper.

/lscript

0 1

1

α

α

m i j

‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 2 α /lscript 1 -b 0 b 1 b m -2 b m -1 1 b 0 1 b 1 d b 0 b 1 b m -2 b m 1 b 0 1 b 1 ‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 2 α 1 -1.2 Approximation in Value Space - MPC and RL It is known that J ∗ satisfies the Bellman equation

We also assume that there is a cost-free and absorbing termination state t [i.e., g ( t, u ) = 0 and f ( t, u ) = t for all u ∈ U ( t )]; e.g., the origin in typical optimal regulation settings in control. The optimal cost function is defined by until all tasks are performed Optimal cost and policy J ∗ ( x ) µ ∗ ( x ) J ∗ ( x ) µ ∗ ( x 1 ) ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ( u 0 , . . . , u k , u k , ˜ u k +1 and that if µ ∗ ( x ) attains the minimum above for all x , then µ ∗ is an optimal policy. Moreover for a policy µ , we have

<!-- formula-not-decoded -->

where M is the set of all admissible policies, and our objective is to find an optimal policy µ ∗ , i.e., one that satisfies J µ ∗ ( x ) = J ∗ ( x ) for all x ∈ X .

This is a typical MPC problem formulation, and it includes the classical linear-quadratic problems where X = ℜ n , U = ℜ m , f is linear, g is positive definite quadratic, and the termination state t is the origin of ℜ n . Note that our formulation makes no assumptions on the nature of the state and control spaces X and U ; they can be arbitrary. However, the problem and its computational solution have been analyzed at the level of generality used here in the author's paper [Ber17b], which can serve as a foundation for mathematical results and analysis that we will use somewhat casually in this paper.

path breaking literature, and generally refers to taking into account the system's future, while applying control in the present. Related ideas appeared independently in the computer science literature, in contexts of search ( A ∗ and related), planning, and game playing.

m

2

d

1

d

2

˜

-

-

<!-- formula-not-decoded -->

N

-

1

)

d

u

d

0 1

, . . . ,

˜

u

N

<!-- formula-not-decoded -->

˜

∗

µ

J

J

˜

µ

˜

-

‖

∗

J

-

‖

-

‖

˜

J

-

+1 Tru

1 ‖ J ‖ J Time 0 Time k Time k Time 0 Time k Time These are results that are generally accepted in the optimal control literature. Their rigorous mathematical proofs at the level of generality considered here are given in the paper [Ber17b], which relies on the general theory of abstract DP problems with nonnegative cost, developed in the paper [Ber77] and extensively discussed in the books [BeS78], [Ber22b]; see also Ch. 3 of the thesis [Li23] for a related discussion.

A major RL approach, which we call approximation in value space , is to replace J ∗ with an approximating realvalued function ˜ J , and obtain a suboptimal policy ˜ µ with the minimization see Fig. 1. We assume that the minimum above is attained for all x ∈ X , and refer to ˜ µ as the one-step lookahead policy .

<!-- formula-not-decoded -->

There is also an ℓ -step lookahead version of the preceding approach, which involves the solution of an ℓ -step DP

k

+1

d

1

{

)}

(

g

(

x

-

Approximation Error

‖

‖

-

∗

J

J

˜

for Riccati Equation Next Control

1st Step Future 1st

/lscript

Steps

1st Step Future 1st

Approximation Error

-

‖

‖

u

∗

µ

˜

Performance Error

J

J

k

‖

˜

J

-

{

∗

J

∗

Performance Error

Current State Control

J

˜

µ

J

∗

x

k

)}

∗

u

k

/lscript

-Step Lookahead Minimization

Current State Control

u

U

x

)

∈

(

Current State Control

J

x

min

) =

(

Current State Control ent State Control

urrent Position ol

u

Current State Control

k

/lscript

-Step Lookahead Minimization

FF-LINE TRAINING

u

k

(

∈

u

/lscript

u

k

/lscript

-Step Lookahead Minimization

u

-Step Lookahead Minimization

,

x

)

αJ

f

(

‖

x, u

X,

Off-Line Obtained Terminal Cost Approximation

Performance Error

Approximation Error

Line Obtained Terminal Cost Approximation

1 3 2 9 5 8 7 10

for Riccati Equation erminal Cost Approximation

iccati Equation Next Control layer Corrected

n Next Control

F

, r

i

(

)

k

u

µ

or

(

u

J

˜

k

k

ost

J

At

∗

x

k

g

F

(

1

L

-

(

d θ x l

α

µ

ce

p

. . . ,

+1

˜

u

0

(˜

u

k

1

, . . . , u

p

′

e

u

(˜

-

N

k

+

m i j

b

=

+

0

m

b

d

-

b

2

1

d

2

b

1

m

|

J

d

0

b

m

{

d

J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} ∈ ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; min u k ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 -d 1 1 -d 2 1 -d m -1 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ u k +1 Cost Function Approximation ‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 2 α /lscript 1 -α 1 J ∗ ( x ) = min u ∈ g ( x, u ) + αJ ∗ f ( x, u ) , x ∈ X, ˜ µ ( x ) ∈ arg u ∈ x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 -d 1 1 -d 2 1 -d m -1 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ u k +1 Cost Function Approximation Time 0 Time k At x At x k Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 -d 1 1 -d 2 1 -d m -1 m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ u k +1 Cost Function Approximation ‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 2 α /lscript 1 -α 1 ( ) : Feature-based parametric architecture State Vector of weights Original States Aggregate States osition 'value' Move 'probabilities' Simplify E {·} hoose the Aggregation and Disaggregation Probabilities se a Neural Network or Other Scheme Form the Aggregate States se a Neural Scheme or Other Scheme ossibly Include 'Handcrafted' Features enerate Features F ( i ) of Formulate Aggregate Problem enerate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem ame algorithm learned multiple games (Go, Shogi) ggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) pproximation in a space of basis functions Plays much better than s programs i, u, j ) Transition probabilities p ij ( u ) W p ontrolled Markov Chain Evaluate Approximate Cost ˜ J µ of valuate Approximate Cost ˜ J µ ( F ( i ) ) of ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ( F ( i ) ) : Feature-based architecture Final Features ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture 1 , . . . , r s : Scalar weights) : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π J ≥ J + , J ( t ) = 0 } 1 1st Step Future 1st /lscript Steps J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; min u k ,u k +1 ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x At x k F L ( K ) for L Corresponding to an Unstable and a Stable Policy Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 Approximation Error ‖ J -J ‖ Performance Error ‖ J ˜ µ -J ‖ x k 1st Step Future 1st /lscript Steps J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; min u k ,u k +1 ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x At x k K ) for L Corresponding to an Unstable and a Stable Policy Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 Approximation Error ‖ ˜ J -J ∗ ‖ Performance Error ‖ J ˜ µ -J ∗ ‖ x k 1st Step Future 1st /lscript Steps J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; min u k ,u k +1 ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} At x k F L ( K ) for L Corresponding to an Unstable and a Stable Policy Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 Approximation Error ‖ J -J ∗ ‖ Performance Error ‖ J ˜ µ -J ∗ ‖ x 1st Step Future 1st /lscript Steps J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; min u k ,u k +1 ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x At x k F L ( K ) for L Corresponding to an Unstable and a Stable Policy Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 k ‖ -‖ Performance Error ‖ J ˜ µ -J ∗ ‖ x k ps ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, n x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } in ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} ponding to an Unstable and a Stable Policy tput (Function of the State) Changing Fixed . . . nsformer Heuristic Stage N u = ( u 0 , . . . , u N -1 ) -1 ) -1 d m d 1 m m -1 . . . 1 roximation Error ‖ ˜ J -J ∗ ‖ Performance Error ‖ J ˜ µ -J ∗ ‖ x k uture 1st /lscript Steps J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; in ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} t x ) for L Corresponding to an Unstable and a Stable Policy t (Control) Output (Function of the State) Changing Fixed . . . e 0 Time k Transformer Heuristic ion of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) , u k , u k +1 , . . . , u N -1 ) +1 Time k + m i j -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( ) )} , x ∈ X ; min u k ,u k +1 ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x At x k F L ( K ) for L Corresponding to an Unstable and a Stable Policy Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; min u k ,u k +1 ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x At x k F L ( K ) for L Corresponding to an Unstable and a Stable Policy Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 1st Step Future 1st /lscript Steps J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; min u k ,u k +1 ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x At x k F L ( K ) for L Corresponding to an Unstable and a Stable Policy Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 1st Step Future 1st /lscript Steps J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; min u k ,u k +1 ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x At x k F L ( K ) for L Corresponding to an Unstable and a Stable Policy Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 min u k ,u k +1 ,...,u k + /lscript -1 m =0 α m g ( x k + m , u k + m ) + α /lscript J ( x k min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x At x k F L ( K ) for L Corresponding to an Unstable and a Stable Policy Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 Fig. 2. Illustration of approximation in value space with ℓ -step lookahead. The ℓ -step minimization at x k yields a sequence ˜ u k , ˜ u k +1 , . . . , ˜ u k + ℓ -1 . The control ˜ u k is applied at x k , and defines the ℓ -step lookahead policy ˜ µ via ˜ µ ( x k ) = ˜ u k . The controls ˜ u k +1 , . . . , ˜ u k + ℓ -1 are discarded. This is similar to mainstream MPC schemes. problem, where ℓ is a positive integer, with a terminal cost function approximation ˜ J . Here at a state x k we minimize the cost of the first ℓ stages with the future costs approximated by ˜ J (see Fig. 2). If this minimization yields a control sequence ˜ u k , ˜ u k +1 , . . . , ˜ u k + ℓ -1 , we apply the control ˜ u k at x k , and discard the controls ˜ u k +1 , . . . , ˜ u k + ℓ -1 . This defines a policy ˜ µ via ˜ µ ( x k ) = ˜ u k . Actually, we may view ℓ -step lookahead minimization as the special case of its one-step counterpart where the lookahead function is the optimal cost function of an ( ℓ -1)-stage DP problem that starts at x k +1 and has a terminal cost α ℓ ˜ J ( x k + ℓ ) after ℓ -1 stages. Note that the multistep scheme depicted in Fig. 2 can be recognized as the most common MPC architecture design (usually α = 1 is chosen in MPC). When the ℓ -step lookahead minimization problem involves continuous control variables, this minimization can often be done by nonlinear programming algorithms, such as sequential quadratic programming and related methods; for some representative papers, see [ABQ99], [WaB10], [OSB13], [BBM17], [RMD17], [LHK18], [Wri19], [FXB22]. However, when discrete/integer variables are involved, time consuming mixed integer programming computations or space and control discretization methods may be required [BeM99], [BBM17]. 1.3 Rollout with a Stable Policy An important cost function approximation approach is rollout, where ˜ J is the cost function J µ of a stable policy µ , i.e., one for which J µ ( x ) &lt; ∞ for all x ∈ X . We discuss this approach in this section, together with associated stability issues. In the MPC context it is often critical that the policy ˜ µ obtained by one-step and ℓ -step lookahead is stable. It can be shown that ˜ µ is stable if ˜ J satisfies the following version of a Lyapunov condition: ˜ J ( x ) ≥ min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , ∀ x ∈ X ; see [Ber17b], [Ber20]. In particular, if ˜ J = J µ for some stable policy µ , then J µ is real-valued and satisfies the preceding Lyapunov condition. 3 To see this, note that from Bellman's equation we have, J µ ( x ) = g ( x, µ ( x ) ) + αJ µ ( f ( x, µ ( x ) ) ) , so that J µ ( x ) ≥ min u ∈ U ( x ) { g ( x, u ) + αJ µ ( f ( x, u ) )} , for all x ∈ X . Thus J µ satisfies the Lyapunov condition, implying that ˜ µ is stable when ˜ J = J µ . In this case we call µ the base policy , and we call ˜ µ the rollout policy that is based on µ .

) +

x, u

(

g

/lscript

Steps

<!-- image -->

In MPC problems that involve state constraints, it may also be necessary to modify the state space X to ensure that the ℓ -step lookahead minimization has a feasible solution (i.e., that the control can keep the state within X ). This leads to the problem of reachability of a target tube , which was first formulated and analyzed in the author's PhD thesis [Ber71] and papers [BeR71], [Ber72], and subsequently discussed and adapted more broadly in the control and MPC literature, e.g., [KoG98], [Bla99], [Ker00], [RKM06], [GFA11], [May14], [CLL23], and [XDS23]. In the context of the off-line training/on-line play conceptual framework of the present paper, reachability issues are ordinarily dealt with off-line, as they tend to involve substantial preliminary target tube calculations. An alternative and simpler possibility is to replace the state constraints with penalty or barrier functions as part of the cost per stage.

Performance Error

,

k

+1

u

k

+1

, . . . , u

l

, . . . , u

+

/lscript

1

‖ J ˜ µ -J ∗ ‖ x k ‖ ˜ J -J ∗ ‖ Performance Error ‖ J ˜ µ -J ∗ ‖ x k , . . . , u l + /lscript -1 x k /lscript -1 l + /lscript -1 l + /lscript -1 k ˜ J u k Performance Error ‖ J ˜ µ -J ∗ ‖ x k ˜ J ‖ J ˜ µ -J ∗ ‖ x k u k +1 , . . . , u l + /lscript -1 ˜ J ‖ J ˜ µ -J ∗ ‖ x k l + /lscript -1 k X, ∈ X ; + /lscript ) Several RL methods are available for computing suitable terminal cost approximations ˜ J by using some form of learning from data, thus circumventing the solution of Bellman's equation. The approximation in value space approach has also received a lot of attention in the MPC literature, but in the early days of MPC there was little consideration of learning that involves training of neural networks and other approximation architectures, as practiced by the RL community.

U

(

x

)

)

min

U

x, u

+

(

/lscript

‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 2 α /lscript 1 -α Time k +1 Truncated Horizon 'Rollout' Future 1 Rollout is a major RL approach, which is simple and very reliable, based on extensive computational experience. It is closely connected to the MPC design philosophy, as has been discussed in the author's early overview paper [Ber05a] and recent books. An important conceptual point is that rollout consists of a single iteration of the fundamental DP method of policy iteration, whose connection with Newton's method in the context of linear-quadratic problems [BeK65], [Kle68], and other Markov decision problems [PoA67], [PuB78], [PuB79] is well known.

The main difficulty with rollout is that computing the required values of J µ ( f ( x, µ ( x )) ) on-line may require time consuming simulation. This is an even greater difficulty for the ℓ -step lookahead version of rollout, where the required number of base policy values increases exponentially with ℓ . In this case, approximate versions of rollout may be used, such as simplified rollout , truncated rollout , and multiagent rollout ; see the books [Ber19], [Ber20], [Ber22a], [Ber23], and the subsequent discussion.

3 Note that if µ is unstable, then J µ is not real-valued and does not qualify for use as ˜ J in the one-step lookahead scheme.

l

+

-

)}

(

X,

{

x

k

‖

-

‖

x

/lscript

-

1

## 1.4 Off-Line Training and On-line Play

Current Position

Implicit in approximation in value space is a conceptual separation between two algorithms: States x k +2

W

p

p

W

: Functions

p

′

′

+

∈

(

x

u

U

m

∈

1

u

arg

=0

U

u

)

x

(

u

k

,u

k

At

+1

,...,u

x

At

k

+

x

k

J

˜

α

/lscript

-

∈

) +

x, u

g

1

/lscript min

(

˜

˜

α

J

f

(

x, u

)

˜

)

x

U

u

x

m

)

)

g

min

u

∈

µ

(

x

)

Input (Control) Output (Function of the Stat

u

k

u

k

,u min

+1

{

g

(

)}

(

x

g

(

-

(

(

(

x, u x, u

) +

k

+

m

) +

αJ

k

, u

+

J

m

) +

α

J

/lscript

f

∗

∈

∈

U

α

(

U

x

)

∗

(

,...,u

k

˜

(

{

m

=0

At

k

x

At

x

x

+

)

{

(

u

min min

{

) =

/lscript

∈

1

˜

Transformer Heuristic

U

(

x

)

(

1

arg

/lscript

∑

-

k

Time 0 Time

k

Input (Control) Output (Function of the Stat

µ

(

x

min

∈

min

-

Initial State 15 1 5 18 4 19 9 21 25 8 12 13

u

(

c

k

(

c

(0)

)

c

)

)}

x, u

g

(

g

) +

(

x, u

α

) +

J

α

f

(

f

(

x, u

f

x, u

)

(

(

x, u

)

)

˜

u

k

+1)

c

α

(

x

)

U

m

{

,u

(

{

)

-

k

+1

,...,u

∈

At

(

N

g

(

x

J

x, u

f

(

k

+

m

, u

k

+

m

m

Region of convergence

x

At

x

k

+

k

-

{

Time 0 Time

k

/lscript

1

=0

/lscript

∑

) +

α

/lscript

1) Parking Space d θ x l

N u

Stage

Transformer Heuristic

-

1

m

α

J

˜

) +

)

˜

J

f

,

)}

= (

(

u

/lscript

α

x, u

f

(

Input (Control) Output (Function of the Stat

m

k

+

, u

) +

N

)

Stage

-

1

Transformer Heuristic

N u

(

Observations

)

J

˜

f

= (

x, u

α

J

˜

(

)

u

)

(

f

Input (Control) Output (Function of the Stat

Off-Line Obtained Player Off-Line Obtained Cost Approximation

: Functions

: Functions

The development of the AlphaZero program by DeepMind Inc, as described in the papers [SHS17], [SSS17], is perhaps the most impressive success story in reinforcement learning (RL) to date. AlphaZero plays Chess, Go, and other games, and is an improvement in terms of performance and generality over the earlier AlphaGo program [SHM16], which plays the game of Go only. AlphaZero plays chess and Go as well or better than all competitor computer programs, and much better than all humans. W { | ≥ W ≥

) +

N

1

-

)

N u

Stage

m

-

m

-

m

(

= (

d

1

)

m m

1

N u

= (

m

1

d

1

d

-

1

m m

d

1

-

-

d

u

u

m m

-

u

N u

J

-

µ

1

+1

k

m

1

= (

J

(

i

)

µ

i

(

m

)

1

d

d

N u

m

u

u

+1

k

-

= (

1

m m

-

J

2

α

∈ J |

1

/lscript

J

+

u

1

m

+1

k

d

-

-

1

d

1

2

α

m m

/lscript

α

π

-stable

-

-

(a) The off-line training algorithm, which designs the cost function approximation ˜ J , and possibly other problem components (such as for example a base policy for rollout, or a target/safety tube of states where the system must stay at all times). (b) The on-line play algorithm, which implements the policy ˜ µ in real-time via one-step or ℓ -step lookahead minimization, cf. Fig. 2. An important point is that the off-line training and online play algorithms can often be designed independently of each other. In particular, approximations used in the on-line lookahead minimization need not relate to the methods used for construction of the terminal cost approximation ˜ J . Moreover, ˜ J can be simple and primitive, particularly in the case of multistep lookahead, or it may be based on sophisticated off-line training methods involving neural networks. Alternatively, ˜ J may be computed off-line with a problem approximation approach , as the optimal or nearly optimal cost function of a simplified optimization problem, which is more convenient for computation (e.g., a linear-quadratic problem approximation, following linearization of nonlinear dynamics of the original problem). Problem simplifications may include exploiting decomposable structure, reducing the size of the state space, neglecting some of the constraints, and ignoring various types of uncertainties. 4 We note that the off-line training/on-line play separation does not explicitly appear in early MPC frameworks, but it is often used in more recent MPC proposals, noted earlier, where ˜ J may involve the training of neural networks with data. On the other hand, the off-line training/on-line play division is common in RL schemes, as well as game programs such as computer chess and backgammon, which we discuss in the next section. 1.5 AlphaZero and TD-Gammon s i 1 i m -1 i m . . . j 1 j 2 j 3 j 4 p ( j 1 ) p ( j 2 ) p ( j 3 ) p ( j 4 ) Neighbors of i m Projections of Neighbors of i m State x Feature Vector φ ( x ) Approximator φ ( x ) ′ r /lscript Stages Riccati Equation Iterates P P 0 P 1 P 2 γ 2 -1 γ 2 P P +1 Cost of Period k Stock Ordered at Period k Inventory System r ( u k ) + cu k x k +1 = x k + u + k -w k Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD C AB C AD C DA C CD C BD C DB C AB Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . . p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play 1st Game / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 System x k +1 = f k ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 3 5 2 4 6 2 1 s i 1 i m -1 i m . . . j 1 j 2 j 3 j 4 p ( j 1 ) p ( j 2 ) p ( j 3 ) p ( j 4 ) Neighbors of i m Projections of Neighbors of i m State x Feature Vector φ ( x ) Approximator φ ( x ) ′ r /lscript Stages Riccati Equation Iterates P P 0 P 1 P 2 γ 2 -1 γ 2 P P +1 Cost of Period k Stock Ordered at Period k Inventory System r ( u k ) + cu k x k +1 = x k + u + k -w k Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD C AB C AD C DA C CD C BD C DB C AB Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . . p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play 1st Game / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 System x k +1 = f k ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 3 5 2 4 6 2 1 Belief State p k Controller µ k Control u k = µ k ( p k ) x 0 x 1 x k x N u k x k +1 Initial State x 0 s Terminal State t Length = 1 x 0 a 0 1 2 t b C Destination J ( x k ) → 0 for all p -stable π from x 0 with x 0 ∈ X and π ∈ P p,x 0 W p + = { within W p + Prob. u Prob. 1 -u Cost 1 Cost 1 - √ u J (1) = min { c, a + J (2) } J (2) = b + J (1) J ∗ J µ J µ ′ J µ ′′ J µ 0 J µ 1 J µ 2 J µ 3 J µ 0 f ( x ; θ k ) f ( x ; θ k +1 ) x k F ( x k ) F ( x ) x k +1 F ( x k +1 ) x k +2 x ∗ = F ( x ∗ ) F µ k ( x ) F µ k Improper policy µ Proper policy µ 1 Current Position x k ON-LINE PLAY Off-Line Obtained Player Off-Line Obtained Cost Approximation OFF-LINE TRAINING 6 1 3 2 9 5 8 7 10 Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 Current Position x k ON-LINE PLAY Off-Line Obtained Player Off-Line Obtained Cost Approximation OFF-LINE TRAINING 6 1 3 2 9 5 8 7 10 Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 Current Position x k ON-LINE PLAY Off-Line Obtained Player Off-Line Obtained Cost Approximation OFF-LINE TRAINING 6 1 3 2 9 5 8 7 10 Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 Current Position x k ON-LINE PLAY Off-Line Obtained Player Off-Line Obtained Cost Approximation OFF-LINE TRAINING 6 1 3 2 9 5 8 7 10 Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 Current Position x k ON-LINE PLAY Off-Line Obtained Player Off-Line Obtained Cost Approximation OFF-LINE TRAINING 6 1 3 2 9 5 8 7 10 Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π ) = 0 } 1 OFF-LINE TRAINING 6 1 3 2 9 5 8 7 10 Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π 1 Current Position x k ON-LINE PLAY Lookahead Tree States x k +1 States x k +2 Off-Line Obtained Player Off-Line Obtained Cost Approximation OFF-LINE TRAINING 6 1 3 2 9 5 8 7 10 Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π J ( x k ) → 0 for all p ′ -stable π 1 Current Position x k ON-LINE PLAY Lookahead Tree Stat States x k +2 Position Evaluation Off-Line Obtained Player Off-Line Obtained Cost Approxima OFF-LINE TRAINING 6 1 3 2 9 5 8 7 10 Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregat I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Pr Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much bett all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based archi ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ 1 Fixed Base Policy Adaptive Reoptimization Position Evaluator Linear policy parameter Optimal /lscript = 3 /lscript = 2 m = 4 Model With the Newton Step Adaptive Rollout Without the Newton Step Base Player J µ k = T µ k J µ k J µ k +1 = T µ k +1 J µ k +1 Reoptimization Current Position x k ON-LINE PLAY Lookahead Tree States x k +1 States x k +2 Position Evaluation Policy ˜ µ with T ˜ µ ˜ J = T ˜ J Corresponds to One-Step Lookahead Tree States x k +1 States x k +2 Policy Evaluation for µ k and for µ k +1 Evaluation Policy ˜ µ with T ˜ µ ˜ J = T ˜ J Position Evaluation Policy ˜ µ with T ˜ µ ˜ J = T ˜ J Cost of µ k Cost of µ k +1 NOTE: J is a function (an n -vector for n states) The figure is a one-dimensional 'slice' of the graph of J OFF-LINE TRAINING Off-Line Training of Value and/or Policy Result of Approximations Result of 6 1 3 2 9 5 8 7 10 ON-LINE PLAY Linearized Bellman Eq. at J µ k Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q NEWTON STEP INTERPRETATION min u ∈ U ( x ) E { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) ) } Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem 1 b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ Cost Function Approximation ‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 1 Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ Cost Function Approximation ‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 1 -α 1 (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ Cost Function Approximation ‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 2 α /lscript 1 -α 1 Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ u k +1 Cost Function Approximation ‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 2 α /lscript 1 -α 1 Input (Control) Output (Function of the Stat Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ u k +1 Cost Function Approximation ‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 2 α /lscript 1 -α 1 Input (Control) Output (Function of the Stat Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ u k +1 Cost Function Approximation ‖ J ˜ µ -J ∗ ‖ ‖ ˜ J -J ∗ ‖ ≤ 2 α /lscript 1 -α 1 Fig. 3. Illustration of the architecture of AlphaZero chess. It uses a very long lookahead minimization involving moves and countermoves of the two players followed by a terminal position evaluator, which is designed through extensive off-line training using a deep neural network. There are many implementation details that we will not discuss here; for example the lookahead is selective, because some lookahead paths are pruned, by using a form of Monte Carlo tree search. Also a primitive form of rollout is used at the end of the lookahead minimization to resolve dynamic terminal positions. Note that the off-line-trained neural network of AlphaZero produces both a position evaluator and a playing policy. However, the neural networktrained policy is not used directly for on-line play. iteration, adapted to off-line training with self-generated data. Moreover, AlphaZero learned how to play chess very quickly; within hours, it played better than all humans and computer programs (with the help of awesome parallel computation power, it must be said). The architecture of AlphaZero is described in Fig. 3. A comparison with Fig. 2 shows that the architectures of AlphaZero and MPC are very similar. They both involve optimization over a truncated rolling horizon with a terminal cost approximation. 5 In AlphaZero, the cost function approximation takes the form of a position evaluator, which uses a deep neural network, trained off-line with an immense amount of chess data. The neural network training process also yields a player that can select a move 'instantly' at any given chess position, and can be used to assist the on-line lookahead process.

=

J

J

J

J

J

ˆ

+

p

′

, J

with

The AlphaZero program is remarkable in several other ways. In particular, it has learned how to play without human instruction, just data generated by playing against itself. In RL this is called self-learning , and can be viewed as a form of the classical DP method of policy

4 Two successful applications of problem approximation exploiting decomposable structures, where the author was personally involved, are described in the papers [KGB82] and [MLW24]. Another type of problem approximation, involving the use of some type of certainty equivalence, will be discussed in Section 3.

k

<!-- image -->

The success of the AlphaZero design framework was replicated by other chess programs such as LeelaChess and Stockfish. It is presently believed that the principal contributor to their success is long lookahead, which uses an efficient on-line play algorithm that involves various forms of tree pruning. The off-line trained position evaluator and player have also contributed to success, although likely to a lesser extent.

The principles of the AlphaZero design have much in common with the earlier TD-Gammon programs of Tesauro

5 Note that AlphaZero is trained to select moves assuming that it plays against an adversarial opponent. Its design philosophy would be more closely aligned to MPC, if it were to play against a known and fixed opponent, whose moves can be perfectly predicted at any given position.

u

{

(

t

x

k

u

u

k

k

m

π

ggregate States

States

J

k

+1

+1

x

(

K

k

) =

= 0

= Optimized

b

b

= Optimized

b

x

2

2

k

x

=

F

)

K

(

b

Transition Cost

Transition Cost ij

d

k

u

2

i

k

= 0

u

q

q

-

1

u

q

. . . b

q

= 0

-Component Control

-

m

u

= (

u

1

m

u

u

, . . . , u

)

1

= (

u

m

j

S f

j

(

u

)

k

S f u u

p

(

)

u

1

= 0

k

u u

u

1

2

u

u

ij

. . . b

b

∗

b

∗

olicy Improvement by Rollout Policy Space Approximation of Rollout Policy at state

∈

∗

m

∗

-Component Control

Bellman Operator Value Iterations Largest Invariant Set

, . . . , u

i

Bellman Operator Value Iterations Largest Invariant Set provement by Rollout Policy Space Approximation of Rollout Policy at state

Initial State 15 1 5 18 4 19 9 21 25 8 12 13

One-step Lookahead with

Lookahead with

(

j

) =

J

(

z

r

) 0

;

z r r

+

/epsilon1

r r

+

/epsilon1

1

r

+

ent Position

2

Corrected ed

2

/epsilon1

x

V

Solution of the Aggregate Problem Transition Cost

V

Line Obtained Player Off-Line Obtained Cost Approximation t Position

x

k

tart End Plus Terminal Cost Approximation

-LINE TRAINING

k

3 2 9 5 8 7 10

n

x

e Obtained Player Off-Line Obtained Cost Approximation

ON-LINE PLAY

ed Player Off-Line Obtained Cost Approximation

Plus Terminal Cost Approximation er Corrected

s i

J

isaggregation Probabilities

INE TRAINING

˜

tion Probabilities

INING

(

ggregation Probabilities

2 9 5 8 7 10

7 10

1

j

Corrected

i

)

, r n Probabilities

˜

J

d

ector of weights Original States Aggregate States ax

˜

)

*

˜

u

State

x

k

p

(

tion 'value' Move 'probabilities' Simplify

J

J J

Cost

e

k

x

(

k

µ

Policy ˜

x

ON-LINE PLAY Lookahead Tree States ose the Aggregation and Disaggregation Probabilities

, r

)

: Feature-based parametric architecture State

i

k

enerate 'Improved' Policy ˜

a Neural Network or Other Scheme Form the Aggregate States

Improved' Policy ˜

ture-based parametric architecture State

Player Off-Line Obtained Cost Approximation tor of weights Original States Aggregate States

)

ghts Original States Aggregate States

1

Prob.

u

m

Prob. 1

-

2

u

{·}

Cost 1 Cost 1

′

- √

u

E

{·}

n 'value' Move 'probabilities' Simplify i y

tate

Ay

)

i

(

)

K K L

-

c

i

1) Parking Spaces

c

(0)

(

k

)

c

(

k

+1)

c

(

N

Bellman Equation on Space of Quadratic Functions

J

(

x

) =

Tube Constraint Cannot be Satisfied for all if

J

(

x

) =

Kx

2

K

x 0 ∈ X if a &gt; 1 F ( K ) 45 u k x β β Kx 2 K S F ( K ) 45 position evaluations, effectively expands the length of the lookahead.

= 2

x

k

+

a &gt;

1

## Rollout Policy Network µ Thus in summary:

µ

- m of i ≈ J µ ( i ) J µ ( i ) Feature ( i ) J µ ( i ) Feature i ) J µ ( i ) Feature ON-LINE PLAY Lookahead Tree States x k +1 ˜ J J * Cost ˜ J µ F ( i ) , r of i ≈ J µ ( i ) J µ ( i ) Feature = 2 m = 4 Model µ F ( K ) = min L ∈/Rfractur F L ( K ) T 2 Cost 28 Cost 27 Cost 13 Lookahead Controller ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J L ∈/Rfractur F L ( K ) ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J (a) The on-line player of AlphaZero plays much better than its extensively trained off-line player. This is due to the beneficial effect of policy improvement with long lookahead minimization, which corrects for the inevitable imperfections of the neural network-trained off-line player, and position evaluator/terminal cost approximation.

∈ J |

: Feature-based parametric architecture State

-

∈A

y

y

-

∗

y

jy

) =

(

j

˜

J

y

φ

r

r

˜

<!-- image -->

˜

K

a

State

Position 'value' Move 'probabilities' Simplify

ING

Ay

F

(

1

i

(

) +

i

. . , F

(

)

b φ

Functions

J

i

)

:

i

(

s

)

:

: Functions

J

ate Cost

J

)

)

=

˜

)

, r

i

(

s

µ

s

)

(

F

(

i

/lscript

=1

/lscript

, r

:

:

)

i

(

s

ar weights)

∑

)

s

/lscript

i

F

(

)

r

/lscript

/lscript unctions

=1

J

ˆ

eights)

J

p

unctions

≥

ˆ

with

p

J

p

J

J

ˆ

J

′

J

≥

J

+

ˆ

J

p

J

′

with

, J

(

J

(

x

J

t

J

(

1

|

≥

{

eterministic Transition x k +1 = f k ( x k , u k ) ggregate Problem Cost Vector r ∗ ˜ J 1 = Corrected V Enlarged State Space Aggregate States Cost ˜ J 0 Cost ˜ J 1 Cost r ∗ *Best Score* epresentative States Controls u are associated with states i Optimal Aggregate Costs r ∗ x y 1 y 2 y 3 1 1 ( i, v ) φ m ( i, v ) φ 2 ( i, v ) ˆ J ( i, v ) = r ′ φ ( i, v ) tic Transition x k +1 = f k ( x k , u k ) Problem Cost Vector r ∗ ˜ J 1 = Corrected V Enlarged State Space States Cost ˜ J 0 Cost ˜ J 1 Cost r ∗ *Best Score* tive States Controls u are associated with states i Optimal Aggregate Costs r ∗ x y 1 y 2 y 3 1 /lscript Stages Riccati Equation Iterates P P 0 P 1 P 2 γ 2 -1 γ 2 P P +1 Cost of Period k Stock Ordered at Period k Inventory System r ( u k ) + cu k x k +1 = x k + u + k -w k Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD C AB C AD C DA C CD C BD C DB C AB Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . . p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play 1st Game / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 System x k +1 = f k ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 3 5 2 4 6 2 1 /lscript Stages Riccati Equation Iterates P P 0 P 1 P 2 γ 2 -1 γ 2 P P +1 Cost of Period k Stock Ordered at Period k Inventory System r ( u k ) + cu k x k +1 = x k + u + k -w k Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD C AB C AD C DA C CD C BD C DB C AB Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . . p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play 1st Game / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 System x k +1 = f k ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 3 5 2 4 6 2 1 J (1) = min { c, a + J (2) } J (2) = b + J (1) J ∗ J µ J µ ′ J µ ′′ J µ 0 J µ 1 J µ 2 J µ 3 J µ 0 f ( x ; θ k ) f ( x ; θ k +1 ) x k F ( x k ) F ( x ) x k +1 F ( x k +1 ) x k +2 x ∗ = F ( x ∗ ) F µ k ( x ) F µ k +1 ( x ) Improper policy µ Proper policy µ 1 Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 e the Aggregation and Disaggregation Probabilities Neural Network or Other Scheme Form the Aggregate States Neural Scheme or Other Scheme ly Include 'Handcrafted' Features te Features F ( i ) of Formulate Aggregate Problem te 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem lgorithm learned multiple games (Go, Shogi) ate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) ximation in a space of basis functions Plays much better than grams k g ( i, u, j ) Transition probabilities p ij ( u ) W p olled Markov Chain Evaluate Approximate Cost ˜ J µ of te Approximate Cost ˜ J µ ( F ( i ) ) of , . . . , F s ( i ) ) : Vector of Features of i Feature-based architecture Final Features ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture Scalar weights) J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π + , J ( t ) = 0 } 1 Move 'probabilities' Simplify E {·} regation and Disaggregation Probabilities etwork or Other Scheme Form the Aggregate States heme or Other Scheme 'Handcrafted' Features res F ( i ) of Formulate Aggregate Problem ved' Policy ˆ µ by 'Solving' the Aggregate Problem learned multiple games (Go, Shogi) r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) in a space of basis functions Plays much better than ) Transition probabilities p ij ( u ) W p rkov Chain Evaluate Approximate Cost ˜ J µ of ximate Cost ˜ J µ ( F ( i ) ) of Vector of Features of i re-based architecture Final Features ) r /lscript it is a linear feature-based architecture with J ( x k ) → 0 for all p -stable π with J ( x k ) → 0 for all p ′ -stable π ) = 0 } 1 a Neural Scheme or Other Scheme ibly Include 'Handcrafted' Features erate Features F ( i ) of Formulate Aggregate Problem erate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem e algorithm learned multiple games (Go, Shogi) regate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) roximation in a space of basis functions Plays much better than rograms α k g ( i, u, j ) Transition probabilities p ij ( u ) W p trolled Markov Chain Evaluate Approximate Cost ˜ J µ of uate Approximate Cost ˜ J µ ( F ( i ) ) of = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ( i ) ) : Feature-based architecture Final Features ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture . . . , r s : Scalar weights) ≥ ˆ J p with J ( x k ) → 0 for all p -stable π ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π 1 0 ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature e-based parametric architecture State s Original States Aggregate States ove 'probabilities' Simplify E {·} ation and Disaggregation Probabilities ork or Other Scheme Form the Aggregate States e or Other Scheme andcrafted' Features F ( i ) of Formulate Aggregate Problem ' Policy ˆ µ by 'Solving' the Aggregate Problem rned multiple games (Go, Shogi) Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) space of basis functions Plays much better than ansition probabilities p ij ( u ) W p Chain Evaluate Approximate Cost ˜ J µ of F ( i ) ) of Vector of Features of i based architecture Final Features it is a linear feature-based architecture k ) → 0 for all p -stable π x k ) → 0 for all p ′ -stable π Position 'value' Move 'probabilities' Simplify {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π 1 States x k +2 Position Evaluation Policy ˜ µ with T ˜ µ ˜ J = T ˜ J Corresponds to One-Step Lookahead Tree States x k +1 States x k +2 Policy Evaluation for µ k and for µ k +1 Evaluation Policy ˜ µ with T ˜ µ ˜ J = T ˜ J Position Evaluation Policy ˜ µ with T ˜ µ ˜ J = T ˜ J Cost of µ k Cost of µ k +1 NOTE: J is a function (an n -vector for n states) The figure is a one-dimensional 'slice' of the graph of J OFF-LINE TRAINING Off-Line Training of Value and/or Policy Result of Approximations Result of 6 1 3 2 9 5 8 7 10 ON-LINE PLAY Linearized Bellman Eq. at J µ k Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q NEWTON STEP INTERPRETATION min u ∈ U ( x ) E { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) ) } Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem 1 ˜ L = -ab ˜ K r + b 2 ˜ K ˜ L = -abK 1 r + b 2 K 1 Slope = 1 1 ˜ L = -ab ˜ K r + b 2 ˜ K ˜ L = -abK 1 r + b 2 K 1 Slope = 1 1 Fig. 4. Illustration of the architecture of TD-Gammon with truncated rollout [TeG96]. It uses a relatively short lookahead minimization followed by rollout and terminal position evaluation (i.e., game simulation with the one-step lookahead player; the game is truncated after a number of moves, with a position evaluation at the end). Note that backgammon involves stochastic uncertainty, and its state is the pair of current board position and dice roll. [Tes94], [Tes95], [TeG96] that play backgammon (a game of substantial computational and strategical complexity, which involves a large state space, as well as stochastic uncertainty due to the rolling of dice); see Fig. 4. TD-Gammon also uses an off-line neural network-trained terminal position evaluator, and importantly, in its 1996 version, it also extends its on-line lookahead by rollout. Tesauro's programs stimulated much interest in RL in the middle 1990s, and the last of these programs [TeG96] exhibits different and better play than humans. The rollout algorithm, based on Monte-Carlo simulation, has been a principal contributor to this achievement. 6 A striking empirical observation is that while the neural network used in AlphaZero was trained extensively, the player that it obtained off-line is not used directly during on-line play (it is too inaccurate due to approximation errors that are inherent in off-line neural network training). Instead a separate on-line player is used to select moves, based on multistep lookahead minimization, a limited form of rollout, and a terminal position evaluator that was trained using experience with the off-line player (cf. Fig. 3). The on-line player performs a form of policy improvement, which is not degraded by neural network approximations. As a result, it greatly improves the performance of the off-line player. An important lesson from AlphaZero and TD-Gammon is that the performance of an off-line trained policy can be greatly improved by on-line approximation in value space, with long lookahead (involving minimization or rollout with the off-line policy, or both), and terminal cost approximation that is obtained off-line. This performance enhancement is often dramatic and is due to a simple fact, which is couched on algorithmic mathematics and is a focal point of the present paper: (a) Approximation in value space with one-step lookahead minimization amounts to a step of Newton's method for solving Bellman's equation . (b) The starting point for the Newton step is based on the results of off-line training, and can be enhanced by longer lookahead minimization and on-line rollout . Indeed the major determinant of the quality of the on-line policy is the Newton step that is performed on-line, while off-line training plays a secondary role by comparison. 1.6 An Overview of our Framework In the next section, we will aim to illustrate the principal ideas of our framework. These are the following: (a) One-step lookahead is equivalent to a step of Newton's method for solving the Bellman equation. (b) ℓ -step lookahead is equivalent to a step of a Newton/SOR method, whereby the Newton step is preceded by ℓ -1 SOR steps (a form of DP/value iterations; SOR stands for successive over-relaxation in numerical analysis terminology).

E

k

x

k

+1

Similarly, TD-Gammon performs on-line a policy improvement step using one-step or two-step lookahead minimization, which is not degraded by neural network approximations. Note that the lookahead minimization in computer backgammon is short, because its lookahead tree of moves and countermoves expands very quickly to take into account the stochastic dice rolls. However, rollout with a base policy, aided by a trained neural network that provides

6 The name 'rollout' was coined by Tesauro [TeG96] in the context of backgammon. It refers to simulating/'rolling out' and averaging the scores of many backgammon games, starting from the current position and using the one-step lookahead player that is based on the position evaluator.

∈A

∗

φ

jy

J

=

+ ≤ J } W p + from -ab ˜ K r + b 2 ˜ K a -a (b) The TD-Gammon player that uses long rollout with a policy plays much better than TD-Gammon without rollout. This is due to the beneficial effect of the rollout, which serves as a substitute for long lookahead minimization.
- (c) There is a superlinear relation between the approximation error ∥ ˜ J -J ∗ ∥ and the performance error ∥ J ˜ µ -J ∗ ∥ , owing to the preceding Newton step interpretation. As a result, within the region of convergence of Newton's method, the performance error ∥ J ˜ µ -J ∗ ∥ is small and often negligible. In particular, the MPC policy ˜ µ is very close to optimal if ˜ J lies within the region of superlinear convergence of Newton's method.
- (d) The region of convergence of Newton's method expands as the length ℓ of the lookahead minimization increases. Thus the performance of the MPC policy ˜ µ improves as ℓ increases, and is essentially optimal if ℓ is sufficiently large regardless of the quality of the

(

i

Off-Line Obtained Player Off-Line Obtained Cost Approximation

r

k

∑

∑

1

u

1

∈

m

)

u

1

u

m

S

terminal cost approximation ˜ J . Indeed, for finite state and control spaces, discount factor α &lt; 1, and a long enough lookahead, it can be shown that ˜ µ is an optimal policy, regardless of the size of the approximation error ∥ ˜ J -J ∗ ∥ ; see Appendix A.4 of the book [Ber22a] and Prop. 2.3.1 of the book [Ber22b].

- (e) The region of stability, i.e., the set of ˜ J for which ˜ µ is stable in the sense that J ˜ µ ( x ) &lt; ∞ for all x ∈ X, is closely connected to the region of convergence of Newton's method.
- (f) The region of stability is also enlarged by increasing the length of the rollout horizon, as long as the base policy that is used for rollout is stable.
- (g) Rollout with a stable policy µ (i.e., ˜ J = J µ ) guarantees that the lookahead policy ˜ µ is also stable, regardless of the length ℓ of lookahead.

In the next section, we will illustrate the preceding points through the use of a simple one-dimensional linearquadratic problem, for which the Bellman equation can be defined through a one-dimensional Riccati equation. We note, however, that all the insights obtained through the Riccati equation survive intact to far more general problems, involving abstract Bellman equations where cost functions are defined over an arbitrary state space. 7

In Section 3, we will briefly discuss stochastic extensions, where the system equation involves stochastic disturbances w k :

<!-- formula-not-decoded -->

The primary difficulty with stochastic problems is the increase of the computation required for both off-line training and on-line play, which may now involve Monte-Carlo simulation of w k . This computation can be effectively mitigated with the use of certainty equivalence , i.e., by replacing the stochastic disturbances w k with typical values w k (such as for example the expected values E { w k } ). However, it is essential that when performing the ℓ -step lookahead minimization, we use certainty equivalence only for the time steps k +1 , . . . , k + ℓ -1 , after the first step . This is necessary in order to maintain the Newton step character of the on-line play process.

In Section 4, we will comment on connections of the MPC/AlphaZero framework with adaptive control. An additional benefit of on-line policy generation by approximation in value space, not observed in the context of games (which have stable rules and environment), is that it works well with changing problem parameters and online replanning. Mathematically, what happens is that the Bellman equation is perturbed due to the parameter

7 In this more general setting, the Bellman equation does not have the differentiability properties required to define the classical form of Newton's method. However, Newton's method has been extended to nondifferentiable operator equations through the work of many researchers starting in the late 70s, and in a form that is perfectly adequate to support theoretically the DP/RL/MPC setting; see Josephy [Jos79], Robinson [Rob80], [Rob88], [Rob11], Kojima and Shindo [KoS86], Kummer [Kum88], [Kum00], Pang [Pan90], Qi and Sun [Qi93], [QiS93], Facchinei and Pang [FaP03], Ito and Kunisch [ItK03], Bolte, Daniilidis, and Lewis [BDL09]. A convergence analysis of the nondifferentiable form of Newton's method, together with a discussion of superlinear performance bounds that relate to MPC, is given in Appendix A of the book [Ber22a].

changes, but approximation in value space still operates as a Newton step. An essential requirement within this context is that a system model is estimated on-line through some identification method, and is used during the onestep or multistep lookahead minimization process, similar to what is done in indirect adaptive control. Within this context, we propose a simplified/faster version of indirect adaptive control, which uses rollout in place of policy reoptimization.

## 2. OFF-LINE TRAINING AND ON-LINE PLAY SYNERGY THROUGH NEWTON'S METHOD

We will now aim to understand the character of approximation in value space as it relates to the Bellman equation, and to the principal algorithms for its solution. To this end we will focus on the one-dimensional version of the classical linear-quadratic problem, where the system has the form

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Here the state x k and the control u k are scalars, and the coefficients a and b are also scalars, with b = 0. The cost function has the form where q and r are positive scalars, and we assume no discounting ( α = 1).

This one-dimensional case admits a convenient and visually insightful analysis of the algorithmic issues that are central for our purposes. However, the analysis fully generalizes to multidimensional linear-quadratic problems. It also extends to general DP problems, including those involving arbitrary state and control spaces, stochastic or set membership uncertainty, as well as multiplicative/risksensitive cost functions. At this level of generality, the analysis requires a more demanding mathematical treatment that is based on the machinery of abstract DP; see the books [Ber20], [Ber22b].

## 2.1 The Riccati Equation

Let us summarize the main analytical and computational results that we will need (all of these are well known and can be found in many sources, including nearly all textbooks on MPC and optimal control). The optimal cost function is quadratic of the form

<!-- formula-not-decoded -->

where the scalar K ∗ is the unique positive solution of Riccati equation

<!-- formula-not-decoded -->

Moreover, the optimal policy is linear of the form

<!-- formula-not-decoded -->

where L ∗ is the scalar given by

<!-- formula-not-decoded -->

The Riccati equation is simply the Bellman equation restricted to quadratic functions J ( x ) = Kx 2 with K ≥ 0. Both the Riccati and the Bellman equations can be viewed as fixed point equations, and can be graphically interpreted and solved graphically as indicated in Fig. 5.

U11|

tron arK

arK

&lt;*

is als arK

•loop :

K*

F(K):

+9.

unction is

(K) meet at a k+1

also c

) = La

Ko ≥ 0, it

Jk are also

-he VI

TJ

rates

- 1

ар-

•111

g in the conte modestin o

T

2

stable L

+ 6L|&gt; 1

5* KL

u

k

w

k

P

k

w

k

Q

0

x

TJ

a

b

k

P

k

+1

x

k

+1

J

P

Q

0

P

Period eriod

ACD

CD

C

AB

C

AB

AD

DA

C

At

k

+

+1

/lscript

y

/lscript

1

-

m

1

=

) Region of Attraction of

m

=0

0

k

+

/lscript

y

x

k

+

bL

1

)

T

(

L

y

) +

α

m

u

J

˜

y T

Lx

(

f

(

, u

∈/Rfractur

= min

k

+

/lscript

1

)

k

m

H

(

k

0

α

=0

min

+1

Multistep Lookahead Policy Cost l

K

+

/lscript

-

or

k

˜

= min

J

˜

J T

˜

J

˜

µ

T

µ

J

F

L

(

K

) = (

a

+

bL

)

+

J J

Region where Sequential Improvement Holds

Multistep Lookahead Policy Cost

u

k

,u or

T

2

= min

µ

= min

T

Cost of Truncated Rollout Policy ˜

2

y

∗

+

) =

1

y

g

(

x

T

Multistep Lookahead Policy Cost

+

µ

˜

q

m

TJ

u

2

, u

y

+

˜

F

2

˜

µ

J

µ

rL

=

≤

∈

˜

2

k

Given quadratic cost approximation

)

(

a

∈/Rfractur

(

K

bL

+

)

x

-

J

-

TJ

L

µ

J

µ

q

+

Multistep Lookahead bL

+

K

a

(

+

)

x

˜

q

bL

-

(

x

˜

+

˜

˜

˜

˜

J

˜

L

T

J

(

K

˜

µ

F

) = (

+

a

Given quadratic cost approximation

L

J T

min

g

(

min x, u

) +

x

(

) =

J

(

J

˜

x

∈/Rfractur

{

f

(

x, u

U

(

TJ

)

F

s

U

u

Kx

2

)

+

, we find

α

J

f

x, u

)

˜

x

(

q

One-Step Lookahead Policy Cost l

Multistep Lookahead Policy Cost

µ

T

K

F

bL

) = min

(

K

α

= min

µ

g

(

K

L

µ

x, u

) +

u

At

x

k

= min

)

+

)

K

F

K

2

∈

L

rL

2

L

F

(

K

˜

J J

=

)

,

with

L

K

(

L

{

{

J

F

(

˜

Region where Sequential Improvement Holds

T

2

˜

J T

J

˜

J

˜

J T

J

Instability Region Stability Region 0

Interval I Interval II Interval III Interval IV

x

= min

TJ

2

Instability Region Stability Region 0

¯

˜

) for

µ

T

K

1

= 0

) +

α

˜

≤

K

) =

y

H

J

(

y

(

y

µ

˜

J

˜

K

L

¯

J

˜

µ

n

∈

T

)

˜

µ

Given quadratic cost approximation

˜

K

2

T

˜

K

µ

˜

J

µ

˜

= 0

p

T

xy

Possible States

x

K

)

Assuming

1

(

F

K

L

˜

)

with

= argmin

) = (

(

L

K

F

}

}

˜

µ

= argmin

) = min

µ

L

) = (

F

a

L

+

(

a

bL

K

+

)

F

( ˜

K

)

K

+

)}

2

(

J T

) = (

bL

L

At

x

L

k

µ

L

˜

Multistep Lookahead Policy Cost l

µ

˜

J

F

L

( ˜

K

)

H

(

y

) =

)

L

2

a

K

+

bL

2

)}

T

Kx

H

+

˜

J J

˜

L

1

2

T

J

(

)

,

TJ

∈/Rfractur

Region where Sequential Improvement Holds

q

bL

(

y

)

+

µ

)

q

=

µ

TJ

x

) = ˜

(

≤

Corresponding to an Unstable and

, we find

Effective Cost Approximation Value S

= min

≤

Lx

Region where Sequential Im bL

T

TJ

¯

J T

K

s

K

y

|

˜

K

µ

˜

&lt;

µ

α

J

(

)

(0

≤

µ

1

,

1]

T

)

Input (Control) Output (Function of the State)

˜

L

-

(

y

K

y

=1

˜

Input (Control) Output (Function of the State) Changing Fixed

∑

=

˜

X

J J

µ

˜

X

=

Multiagent Q-factor minimization

K

1

˜

T

TJ

u

Instability Region Stability Region

= 0

K

∈

(

y

)

y

n

min

)

J J

Possible States

x

(

K

˜

˜

)

= 0

p

xy

K

¯

(

µ

˜

µ

=

X

Multi

∑

˜

=1

y

˜

X

to construct the one-step lookahead policy ˜

T

L

µ

k

Multistep lookahead moves the starting point of

Time 0 Time

Termination State Constraint Set

k

w

using an Corresponds to One-Step Lookahead Policy ˜

k

2-State/2-Control Example

∗

µ

T

˜

Unstable

K

TJ

k

k

where

J

L

˜

µ

Region of convergence

Current Partial Folding Moving Obstacle using an Corresponds to One-Step Lookahead Policy ˜

P

Effective Cost Approximation Value Space Approximation State 1

P

k

µ

T

J

˜

<!-- image -->

and its cost function

, . . . ,

u

˜

-

k

1

Complete Folding Corresponding to Open

) =

k Q

arK

=1

AD

µ

, u

Yields Truncated Rollout Policy ˜

2

0

k

, u

a

b

Multiagent Q-factor minimi

N u

= (

, .

One-Step Lookahead

J J

˜

µ

˜

u

0

=

µ

L

K

1

=

Termination State Constrai

µ

T

J

=

-

Multistep Lookahead

K K

K

∗

.

2-State/2-Control Example

k

k

k

+

using an Corresponds to One-Step Lo

Current Partial Folding Mo

{

K

TJ

K

K

µ

µ

1

<!-- image -->

Effective Cost Approximation Value Space Appr

=

TJ

, . . . ,

˜

u

) for all ˜

min

u

q

(

U

x

2

)

¯

+

) +

α

k

+1

y

˜

TJ

K

∗

)(2) =

rL

p

;

J

=1

(

y

)

xy

J

(

K

∗

∈

≤

(

J

= 0

µ

= min

)(1)

Instability Region Stability Regio

0

k +1 Initial State A C AB AC CA CD ABC CAB CAD CDA C AC C CA C CD C BC C CB C CD DA C CD C BD C DB C AB Expert Rollout with Base Policy m -Step Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) r + b K L T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Defined by Newton step from ˜ J for solving J = TJ Line Stability Region F ( K ) = arK r + b 2 K + q F ˜ L ( K 1 ) ˆ K = a 2 -1 T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Defined by Newton step from ˜ J for solving J = TJ Newton step from T /lscript -1 ˜ J for solving J = TJ ( TJ )(1) Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation k +1 Initial State A C AB AC CA CD ABC CAB CAD CDA C AC C CA C CD C BC C CB C CD C CD C BD C DB C AB ˜ J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ -r b 2 Generic stable policy µ T µ J Generic unstable policy µ ′ T µ ′ J Cost of Truncated Rollout Policy ˜ µ 1 of the graph of T J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J (2) Fig. 5. Graphical solution of the Riccati equation. The optimal cost function is J ∗ ( x ) = K ∗ x 2 . The scalar K ∗ solves the fixed point equation K = F ( K ). It can be found graphically as the positive value of K that corresponds to the point where the graphs of the functions K and F ( K ) meet. A similar interpretation can be given for the solution of the general Bellman equation, which however cannot be visually depicted for problems involving more than one or two states; see the books [Ber20], [Ber22a], and [Ber22b]. { } Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD C AB C C DA C CD C BD C DB C AB Complete Folding Correspo Expert Rollout with Base Policy m Approximation of E {·} : n ∑ Expert Rollout with Base Policy m -Step Approximation of E {·} : Approximate minimiza min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ T ˜ µ T µ m ˜ J = TT µ m ˜ J Yields Truncated Rollout Policy ˜ µ Define It follows that K L is the unique solution of the linear equation K = F L ( K ) , where F L ( K Line Stability Region F ( K ) = arK r + b 2 K T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Newton step from ˜ J for solving J = T using an Corresponds to One-Step Lookahead Policy ˜ µ Line Stability Region F ( K ) = arK r + b 2 K + q F ˜ L ( K 1 ) ˆ K = a 2 -1 T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Define Newton step from ˜ J for solving J State 2 ( TJ )(1) ˜ J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ Generic stable policy µ T µ J Generic unstable poli Cost of Truncated Rollout Policy ˜ µ 1 of the graph Multistep lookahead moves the starting point of the Newton step The longer the lookahead the better The start of the Newton step must be within the region of stabili L Multistep lookahead moves the starting point of the Newton step closer to K ∗ The longer the lookahead the better The start of the Newton step must be within the region of stability The start of the Newton step must be within th Longer lookahead promotes stability of the multi Value Policy The start of the Newton step must be within the region of Longer lookahead promotes stability of the multistep looka Value Policy Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 -d 1 1 -d 2 1 -d m -1 1 -d m ( u 0 , . . . , u Fig. 6. Graphical solution of the Riccati equation for a linear policy µ ( x ) = Lx . When µ is stable, its cost function is J µ ( x ) = K L x 2 , where K L corresponds to the point where the graphs of the functions K and F L ( K ) meet. min n p xy min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) Multiagent Q-factor minimization x k Possible States TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ Multistep Lookahead Policy Cost T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds Interval I Interval II Interval III Interval IV 1 Multistep Lookahead Policy Cost F L ( K ) = ( a + bL ) 2 K + q T 2 ˜ J T ˜ J ˜ J Region where Sequential I F L ( K ) = ( a + bL ) 2 K + q + rL 2 F ˜ L ( K ) T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ TJ Instability Region Stability Region 0 T µ m ˜ J 2 2 2

D

C

epair pair

p

2(

e

/

e

/

0

¯

K

Termination State Infinite Horizon Approximatio

.

)

(0

,

Termination State Infinite Horizon Approximation Subspace

T

for the

τ

One-Step Lookahead Policy Cost

.

˜

J

n

k

+1

k

(

y

.

Longer lookahead promotes stability of the multistep lookahead policy

µ

Q-factor approximation

¯

Possible Stat

, u

.

˜

K

, R

for all

˜

2

x

)

˜

K

= 0

-to- y J

xy

(

p

K

=1

y

∗

)

K

R

K

(1) =

˜

(

x

=

X

X

k

= 0

Mu

Multistep Lookahead Policy Cost

)

/negationslash

)

T

k

1

The preceding one-dimensional problem is well suited for geometric i

/

Complete Folding Corresponding to Open

1

k

,

2

r

Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . 1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn / Bold Play x k , u k u k x k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . . p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T We can also characterize graphically the cost function of a policy µ that is linear of the form µ ( x ) = Lx , and is also stable, in the sense that the scalar L satisfies | a + bL | &lt; 1, so that the corresponding closed-loop system x k +1 = ( a + bL ) x k Do not Repair Repair 1 2 n -1 n p 11 p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p u 1 ˆ u 1 10 11 12 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 x u ∗ x ∗ u ∗ x ∗ | a + bL | &gt; TJ = min TJ = min Value Policy Value Policy Controls u ∈ x y αJ (2) (2 αr , Controls u ∈ U ( x ) x y Shortest N 1 x k L k u k w k x k +1 = A k x k + B k u k + w k min u ∈ U ( x ) n ∑ y =1 p xy ( min u ∈ U ( x ) n ∑ y =1 p xy Termination State Constraint Set X X = r b 2 +1 1 -r b 2 K K ∗ K k K k +1 αKr r + αKb 2 +1 Current Partial Folding Moving Obstacle TJ r b 2 + q q F ( K ) = arK r + b 2 ar b 2 ˜ L = -ab r + ab

(

y

J

∗

∑

(2) = 0

+1

K

1

Multiagent Q-factor minimi

x

1

k

+1

u

˜

Rollo gave earlier in this section, because approximation in value space, and t

u

˜

x

x

˜

∗

u

2

x

x

k

x

∗

3

+1

Termination State Infinite Horizon Approximation Subspace Bellman Eq:

is a function of

3

n

2(

-

n

1)

-

/ Timid Play 2nd Game Timid Play 1st Game / Bold Play p d 1 p d p w 1 p x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 1 High Cost Transition Chosen by Heuristic at Capacity=1 Optimal Solution 2.4.2, 2.4.3 Timid Play 2nd Game / Bold Play Multistep Lookahead Policy Cost J is stable. Its cost function has the form J µ ( x ) = K L x 2 , 2nd Game Controls F ( P ) ˜ P P k P Expert

1st Game

∗

1

+1

involve quadratic cost functions

k

/ --w High Cost Transition Chosen by Heuristic at x Permanent Trajectory Tentative Trajectory Optimal Trajectory ChoTimid Play 1st Game / Bold Play p d 1 p d p w 1 p w F L ( K ) = ( a + bL ) 2 K + q + rL 2 F ˜ L ( K ) where K L solves the equation 8 tions of just the number K u ∈ U ( x ) r b 2 +1 1 r b 2 K K ∗ K k k k r b 2 +1 1 r b 2 K K ∗ K k K Rollout with Base Policy m -Step Line Stability Region J ∗ (1) = 0 J (1) ( TJ

-

x

k

0

0

+1

+1

4

6

-

0

-

=

1

k

=

6

2

f

2

8

9

3

3

8

emperature perature

=

∗

2.4.5

Controls

u

/

˜

1

∗

˜

1

x

˜

k

2

+

k

is a function

Termination State Constrai

=

µ

T

J

Multiagent

x

x

L

µ

u

1

, which can be represented b

J

(1)

2

2

3

2.4.5

∗

0

x

˜

x

u

˜

Multiagent

∗

∗

+

rL

F

˜

(

Permanent Trajectory Tentative Trajectory Optimal T

High Cost Transition Chose

L

1

. In particular, Bellman's equation can be repla

K

k

k

k

+

2-State/2-Control Exa

(2) = 0 Exact using an Corresponds to One-Step Lo

(2) = 0 Exact VI:

k

+1

J

(1

Capacity=1 Optimal Soluti using an Corresponds to One-Step Lookahead Policy ˜

Similarly, approximation in value space with one-step and multistep loo

Permanent Trajectory Tenta

K

(

P

)

<!-- image -->

2

µ m

k

k

x

u

J

2

2

Steps

1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 f k ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 9 6 1 2 x 0 u 0 u 1 x 1 Oven 1 Oven 2 Final Temperature A k y k + ξ k y k +1 C k w k ic Problems 1 Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost Approximation in Policy Space Heuristic Cost Approximation for for Stages Beyond Truncation y k Feature States y k +1 Cost g k ( x k , u k ) Approximate Q-Factor ˜ Q ( x, u ) At x Approximation ˆ J min u ∈ U ( x ) E w { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) ) } Truncated Rollout Policy µ m Steps 1 1 sen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost Approximation in Policy Space Heuristic Cost Approximation for for Stages Beyond Truncation y k Feature States y k +1 Cost g k ( x k , u k ) Approximate Q-Factor ˜ Q ( x, u ) At x Approximation ˆ J min u ∈ U ( x ) E w { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) ) } Steps 1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 6 1 2 x 0 u 0 u 1 x 1 Oven 1 Oven 2 Final Temperature A k y k + ξ k y k +1 C k w k Problems ate Info Ch. 3 1 T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ J T ˜ µ J TJ Instability Region Stability Region 0 T µ m ˜ J a 2 r b 2 a 2 r b 2 + q q F ( K ) = a 2 rK r + b 2 K + q K ∗ = 0 ˜ K = 0 ¯ K ¯ K = 0 K ˜ L ˜ L = -ab ˜ K r + ab 2 ˜ K K 1 ˜ L = -abK 1 r + b 2 K 1 F ( K ) = a rK r + b 2 K J ∗ (1) = 0 J (1) ( TJ )(1) = min { J (1) , 1 } 1 K = F L ( K ) = ( a + bL ) 2 K + q + rL 2 . The graphical solution of this equation is illustrated in Fig. 6. The function F L ( K ) is linear, with slope ( a + bL ) 2 that is strictly less than 1. In particular, K L corresponds to the point where the graphs of the functions K and F L ( K ) meet. If µ ( x ) = Lx is unstable, in the sense that the scalar L satisfies | a + bL | &gt; 1, then its cost function is given by J µ ( x ) = ∞ for all x = 0 and J µ (0) = 0. In this case the graphs of the functions K and F L ( K ) meet at a negative value of K , which has no meaning in the context of the linear-quadratic problem. 2.2 Iterative Solution by Value and Policy Iteration The classical DP algorithm of Value Iteration (VI for short) produces a sequence of cost functions { J k } by applying the Bellman equation operator repeatedly, starting from an initial nonnegative function J 0 . For our linearquadratic problem it takes the form J k +1 ( x ) = min u ∈ℜ { qx 2 + ru 2 + J k ( ax + bu ) } . 8 Sometimes this equation is called the 'Lyapunov equation' in the control theory literature. In this paper, we will refer to it as the 'Riccati equation for linear policies.' 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -System x k +1 = f k ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 3 5 2 4 6 2 10 5 7 8 3 9 6 1 2 Initial Temperature x 0 u 0 u 1 x 1 Oven 1 Oven 2 Final Temperature x 2 ξ k y k +1 = A k y k + ξ k y k +1 C k w k Stochastic Problems 1 sen by Base Heuristic at x 0 Initia Base Policy Rollout Policy n -2 One-Step or Multistep Look Approximation in Policy Sp for Stages Beyond Truncatio Approximate Q-Factor u ∈ U ( x ) w { Truncated Rollout Policy µ sen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value n -2 One-Step or Multistep Lookahead for stages Possible Approximation in Policy Space Heuristic Cost Approx for Stages Beyond Truncation y k Feature States y k +1 Approximate Q-Factor ˜ Q ( x, u ) At x Approximation ˆ J min u ∈ U ( x ) E w { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) Truncated Rollout Policy 1 1 stability Figs. 3.5-3.6, and the rollout and PI Figs. 3.8-3.9 can be repres We will next present these graphs and obtain corresponding geometrical in obtain similar insights about what happens in exceptional cases where we Bellman's Equation and Value Iteration Approximation in Value Space with One-Step and Multistep Lookahead Region of Stability Rollout and Policy Iteration 31 T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement TJ Instability Region Stability Region 0 T µ m ˜ J a 2 r b 2 a 2 r b 2 + q q F ( K ) = a 2 rK r + b 2 K + q K ∗ = 0 ˜ K = 0 ˜ L = -ab ˜ K r + ab ˜ K K 1 ˜ L = -abK 1 2 F ( K ) = a 2 rK r + b 2 K J ∗ (1) = 0 J (1) ( TJ { } 1 x y Shortest N -Stage Distance x -toαJ k (2) (2 αr k , 2 αr k ) Terminal Position Evaluation 1 x y Shortest N -Stage Distance x -toy J ∗ (1) = αJ k (2) (2 αr k , 2 αr k ) Terminal Position Evaluation 1 1 1 ˜ F ( P ) k Q 0 P -R E { B 2 } 45 ◦ Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD C AB C AD C DA C CD C BD C DB C AB Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play 1st Game / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 Current Partial Folding Mo Complete Folding Correspo Expert Rollout with Base Policy m Approximation of E {·} : min u ∈ U ( x ) n ∑ y =1 x 1 k Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k ( x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 High Cost Transition Chose Capacity=1 Optimal Soluti Permanent Trajectory Tenta Expert Rollout with Base Policy m -Step Approximation of E {·} : Approximate minim min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k + x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Ro Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optima sen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Val Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choic Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajector Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Cho Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajecto Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ x ∗ u ∗ x ∗ u ∗ x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 n -One-Step or Multistep Lookahead for stages Possible Terminal Cost T ˜ µ T µ m ˜ J = TT µ m ˜ J Yields Truncated Rollout Policy ˜ Effective Cost Approximation Value Space State 2 ( TJ )(1) ˜ J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of Generic stable policy µ T µ J Generic unstable Cost of Truncated Rollout Policy ˜ µ 1 of the g J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ TJ = min µ T µ J One-Step Lookahead Policy C TJ = min µ T µ J Multistep Lookahead Policy C Multistep Lookahead Policy Cost J is a functi F L ( K ) = ( a + bL ) 2 K + q + rL 2 F T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvem Line Stability Region F ( K ) = r + b 2 K T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncate Newton step from ˜ J for solving J = T 1 Line Stability Region F ( K ) = arK r + b 2 K + q F ˜ L ( K 1 ) ˆ K = a 2 T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ De Newton step from ˜ J for solving J = TJ Newton step from T /lscript -1 ˜ J for solving J = TJ ( TJ )(1) 1 Fig. 7. Graphical illustration of VI. It has the form K k +1 = F ( K k ), where F is the Riccati operator, F ( K ) = a 2 rK r + b 2 K + q. The algorithm converges to K ∗ starting from any K 0 ≥ 0. When J 0 is quadratic of the form J 0 ( x ) = K 0 x 2 with K 0 ≥ 0, it can be seen that the VI iterates J k are also quadratic of the form J k ( x ) = K k x 2 , where K k = F ( K k -1 ) . Then the VI algorithm becomes a fixed point iteration that uses the Riccati operator F . The algorithm is illustrated in Fig. 7. As can be seen from the figure, when starting from any K 0 ≥ 0, the algorithm generates a sequence { K k } of nonnegative scalars that converges to K ∗ . Another major algorithm is Policy Iteration (PI for short). It produces a sequence of stable policies { µ k } , starting with some stable policy µ 0 . Each policy has improved cost function over the preceding one, i.e., J µ k +1 ( x ) ≤ J µ k ( x ) for all k and x , and the sequence of policies { µ k } converges to the optimal. Policy iteration is of major importance in RL, since most of the successful algorithmic RL schemes

µ m

2

Approximation in Policy Space Heuristic Cost Approximation for

k

System

-

1

x

x

1

2

x

1

sen by Base Heuristic at sen by Base Heuristic at

u

k

k

Feature States

Instability Region Stability Region 0

k

w

)(1) = min

, u

k

1

d

(1)

k

min

,

τ

1

.

De

µ

arK

µ

r

+

b

K

)

p

1

xy x,

Q

(

(

˜

)(2) =

) }

E

x, u

g

(

sen by Base Heuristic at

k

y

+1

1

0

0

Initial

x

)

µ

, w

)

=

µ

(

x

Initial

x

x

k

x

0

Initia

k

k

k

k

k

2

=

n

k

f

2

(

x

, u

-

k

2

k

Base Policy Rollout Policy Approximation in Value Space

Base Policy Rollout Policy Approximation in Value Space

One-Step or Multistep Lookahead for stages Possib for Stages Beyond Truncation

+1

Approximate Q-Factor

n

6

n

-

-

3

5

2

4

2

Cost

g

(

x

k

, u

k

)

T

µ m

Base Policy Rollout Policy

˜

n

J

y

TJ

) At

a

2

n

ˆ

2

Approximation in Policy Space Heuristic Cost App

2

2

Approximation

x

r

J

(

x, u

r

b

a

b

a

r

2

+

rK

b

y

K

k

˜

One-Step or Multistep Look

2

for Stages Beyond Truncation

-

One-Step or Multistep Lookahead for stages Possible Termin

2

One-Step or Multistep Lookahead for stages Possible Terminal

2

+

q q F

(

K

) =

2

+

q K

= 0

K

Feature States

y

=

k

∗

˜

Q

0

Timid Play

x

˜

x

Rollout with Base Policy

Line Stability Region

m

-Step

1

1

2

F

2

F

˜

(

2

-

-

˜

˜

x

k

At

˜

µ m

∈/Rfractur

-

∗

(

) =

u

-

=

bL| &gt; 1, t

-

̸

Truncated Rollout Policy

1

0

k

,...,u

k

2

1

min

∈/Rfractur

L

+

q

= min

F

(

F

0

y

=

y

L

y

)

) = min

K

(

K

rL

2

F

F

)

,

with

+

K

(

K

L

L

H

bL

a

(

+

) = min

)

(

K

)

x

2

2

∈/Rfractur

F

,

with

K

(

L

L

)

,

with

F

L

(

K

F

)

) = (

(

K

a

L

bL

+

K

G

2

∈/Rfractur

a

) = (

F

(

bL

+

)

K

2

)

y G

y

y

(

(

(

y

) =

L

-

˜

2

K

K

2

2

2

Policy Improvement by Rollout Policy Space Approximation of Rollout Policy at state

L

= min

= min

) = (

a

+

q

+

(2)

= arg min

L

+

One-step Lookahead with

L

rL

2

L

K

= arg min

L

F

( ˜

)

L

F

H

(

y

) =

+

bL

L

+

rL

(

ax

(

ax

˜

2

2

(

j

J

2

2

(

(

)

c

m

1)

c

c

+1)

m

(

(

c

)

u

2

u

K

=

=

Lx

Lx

q

One-step Lookahead with

{

{

c

-

+

+

}

) =

( ˜

(

G

(

K

y

y

L

) =

bu bu

2

2

˜

J

)

)

(

M

)

)

∈A

)

H

-

(

y

y G

) =

(

y

2

jy

y

∗

r

u

G

)

= min

F

2

φ

K

(

2

ru

-

)

(

y

)

2

x

+

+

qx

= min

F

2

+

y

)

2

qx

(

+

ax min

y G

= min ru

(

u

+

K

{

x

∑

2

bu

∈/Rfractur

2

1) Linear Stable Policy Quadratic Cost Approximation

{

2

2

)

K

+

= min ru

qx

∈/Rfractur

u

∈/Rfractur

2

y

∗

r

2

M

2

2

y

}

}

j

∈A

K

)

x

a

(

K

+

)

bL

x

+

+

x

K

bL

a

)

+

(

φ

jy

)

1) Linear Stable Policy Quadratic Cost Approximation min

min

(2)

c

(

qx qx

{

F

(

-

∈/Rfractur

∈/Rfractur

= min

= min

q

q

+

+

rL

rL

m

2

L

L

1)

c

(

K

) Region of Attraction of

1

(

y

G

y G

) Region of Attraction of

(

∈/Rfractur

-

L

or or

(

+

q

+

rL

2

c

y

Given quadratic cost approximation

J

dratic cost approximation

∗

y

∗

y

) Region of Attraction of roximation

˜

J

˜

L

(2)

L

1)

= arg min

c

k

x

(

m

c

Position

x

K

µ

-

to construct the one-step lookahead policy ˜

x

qx

(

F

K

(

∑

)

x

2

= min

∈/Rfractur

∈/Rfractur

m

)

c

(

m

+1)

c

(

M

)

c

(

M

-

y

c

(

m

to construct the one-step lookahead policy ˜

∗

˜

˜

) =

) = ˜

2

/epsilon1

r

+

1

2

1

/epsilon1

z r r

+

+

) 0

r

/epsilon1

2

r

+

/epsilon1

m

r

;

(

r

z

p

p

(

z

;

r

) 0

with ru

Lx

-

= min

= min

/epsilon1

{

L

-

∈/Rfractur

2

2

L

r

+

F

/epsilon1

∈/Rfractur

m

K

(

K

min

u

{

u

=

2

m

Lx

+

/epsilon1

ax

x

)

2

(

=

K

ax

(

+

qx bu

= min

L

· · ·

)

)

bu

˜

J

(

bu

(

ax

2

K

ru

+

+

qx

(

Lx

˜

ax

+

=

u

˜

p

2

. (e.g., a NN) Data (

}

bu

(

F

ax

+

K

(

+

ru

+

K

(

.

2

2

2

ru

+

+

qx

)

p

p

2

∈/Rfractur

2

x

(

}

J

2

.

2

) =

Kx

1

2

2

m

2

2

· · ·

L

min

= min

F

2

= min

x

)

K

qx

(

+

.

∈/Rfractur

Lx

u

= min

.

ru

-

q

}

2

x

bu

+

) =

= min

x

ax

)

K

2

+

bu

)

u

2

K

+

2

bL

s

bu

+

(

a

+

)

K

(

)

x

ax

+

)

s

s

+

qx

2

+

K

(

-

µ

/epsilon1

}

u

{

1

2

/epsilon1

+

/epsilon1

r

r

+

+

m

(

x

or

m

K

p

rL

{

+

2

K

+

(

q

ax

2

= min

+

∈/Rfractur

}

rL

+

ru ax

2

2

p

1

2

u

p

. (e.g., a NN) Data (

∈/Rfractur

{

}

(

a

+

2

2

+

bu bL

)

, c

x

{

2

x

2

2

F

+

(

/epsilon1

/epsilon1

1

r

∈/Rfractur

Lx

) = ˜

= min

r

u

rL

+

∈/Rfractur

r

=

{

2

2

min bu

K

)

)

2

L

qx ru

+

-

Tangent Riccati Operator at min

-

˜

2

2

z r r

F

L

or

K

(

F

) = (

bL

+

a

a

K

(

)

) = (

+

bL

K

K

)

2

2

with

2

(

+

K

(

L

∈/Rfractur

2

)

(

x

, c

2

x

= min

2

2

= min qx

+

qx

+

ru

K

ru ax

+

)

2

2

K

L

∈/Rfractur

q

a

= min

)

(

Lx

u

2

2

x

∈/Rfractur

u

2

qx

+

ru

= min

=

{

+

Lx

{

bL

=

+

rL

F

+

(

{

K

bL

+

F

2

{

Region of Attraction of rL

rL

L

+

+

q

q

+

+

K

or

) = min

a

(

K

∈/Rfractur

L

)

Lx

{

= min

x

F

∈/Rfractur

L

)

min

q

}

}

Transition Cost

+

+

L

}

i

{

)

2

˜

Kx qx

2

+

2

)

s

min

2

}

(

ax

)

u

(

bu

K

+

}

{

+

ax

(

K

ax

2

bu

+

2

K

+

K

+

rL

= min

}

}

∈/Rfractur

= min

)

bu

+

J

2

2

2

2

+

K

min

(

a

bL

2

qx

+

ru with

2

2

L

u

=

Lx

x

∗

+

)

qx

)

∈/Rfractur

,

(

x

L

Solution of the Aggregate Problem Transition Cost

2

= min

V

Corrected

Solution of the Aggregate Problem Transition Cost

Region of Attraction of

L

q

= min

) = min

F

(

K

)

,

) = min

F

L

(

K

)

,

Tangent Riccati Operator at

˜

K

Start End Plus Terminal Cost Approximation

{

Start End Plus Terminal Cost Approximation

L

K

2

u

∈/Rfractur

{

NEWTON STEP for Bellman Eq. 2-Step Lookahead Minimization

y

y

y

NEWTON STEP for Bellman Eq. 2-Step Lookahead Minimization

∗

∗

0

y

1

H

F

(

K

Disaggregation Probabilities

Enhancements to the Starting Point of Newton Step Value Iterations

˜

H

y

(

, we find

, we find

1) Linear Stable Policy Quadratic Cost approximation

-

o construct the one-step lookahead policy ˜

ON-LINE PLAY

k

)

c

(

m

+1)

c

(

M

c

(

M

)

one-step lookahead policy ˜

E TRAINING

Player Off-Line Obtained Cost Approximation s i

-

ahead policy ˜

Tangent Riccati Operator at iccati Operator at

NING

1

9 5 8 7 10

j

10

r at

˜

K

˜

J

orrected

J

r

˜

Region of Attraction of

Unstable

*

˜

x

p

(

j

bL

Cost

ON-LINE PLAY Lookahead Tree States

J J

k

a

+

&gt;

Map

: Feature-based parametric architecture State

|

Stable

L

a

+

and its cost function

|

t function

J

|

re-based parametric architecture State

˜

µ

d Player Off-Line Obtained Cost Approximation of weights Original States Aggregate States

)

ts Original States Aggregate States

State

x

'value' Move 'probabilities' Simplify

INING

) =

10

M M

K

˜

L

x

2

ove 'probabilities' Simplify

˜

he Aggregation and Disaggregation Probabilities

V

F

(

K

L

F

K

, we find

L

Kx

J

(

) =

(

x

, we find

x

˜

˜

<!-- image -->

Disaggregation Probabilities

0

) =

y

1

G

(

y

)

Given quadratic cost approximation

Enhancements to the Starting Point of Newton Step Value Iterations

) =

Kx

Aggregation Probabilities

x

k

-

Given quadratic cost approximation

Current Position

x

k

ON-LINE PLAY

ON-LINE PLAY

) =

) =

G

G

(

(

y

y

Aggregation Probabilities

)

)

State

Off-Line Obtained Player Off-Line Obtained Cost Approximation

Off-Line Obtained Player Off-Line Obtained Cost Approximation

L

a

1 Stable

|

|

bL

k

x

Policy ˜

+

L

c

s i

a

k

Off-Line Obtained Player Off-Line Obtained Cost Approximation

Off-Line Obtained Player Off-Line Obtained Cost Approximation

µ

(

|

1) Linear Stable Policy Quadratic Cost Approximation

1)

OFF-LINE TRAINING

OFF-LINE TRAINING

(

m

and its cost function

Generate 'Improved' Policy ˜

1) Linear Stable Policy Quadratic Cost Approximation

) =

-

6 1 3 2 9 5 8 7 10

Generate 'Improved' Policy ˜

j

1

˜

)

c

(

m

+1)

m

-

1)

c

(

K

to construct the one-step lookahead policy ˜

6 1 3 2 9 5 8 7 10

L

x

2

to construct the one-step lookahead policy ˜

ru qx

2

2

2

)

bu

+

Lx

2

=

q

∈/Rfractur

{

+

+

K

ru

+

}

+

2

q

rL

ax

+

rL

2

(

x

2

F

L

rL

(

2

ax

+

+

K

(

K

bu

2

)

a

+

2

}

(

K

2

2

)

+

K

b

(

bL

) = (

+

a

2

)

a

q

+

rL

2

2

y

∗

, we find

1) Linear Stable Policy Quadratic Cost approximation

J

(

x

) =

Kx

2

y

H

(

(

G

) =

˜

˜

y

)

-

x

˜

Kx

2

) =

1) Linear Stable Policy Qua

(

y

)

y G

ON-LINE PLAY

1) Linear Stable Policy to construct the one-step lookahe

Off-Line Obtained Player Off-Line Obt

˜

(

to construct the one-step lookahead policy ˜

˜

p

J J

J

˜

Multistep lookahead moves the starting point of the Newton step closer to

J

Player Corrected

(

(

)

i

i y

Ay

) +

b φ

Tangent Riccati Operator at

1

(

Cost

µ

*

˜

Multistep lookahead moves the starting point of the Newton step closer to

ON-LINE PLAY Lookahead Tree States

˜

1) Linear Stable Policy Quadratic Cost Approximation

Cost

J J

Deterministic Transition

J

J

˜

Off-Line Obtained Player Off-Line Obtained Cost

1) Linear Stable Policy Quadratic Cost Appr

Map

Tangent Riccati Operator at

1) Linear Stable Policy Quadratic Cost Approximation

Deterministic Transition

Tangent Riccati Operator at

The longer the lookahead the better

: Feature-based parametric architecture State

The longer the lookahead the better

Off-Line Obtained Player Off-Line Obtained Cost Approximation

: Feature-based parametric architecture State

Aggregate Problem Cost Vector

State

ON-LINE PLAY Lookahead Tree Stat

Vector of weights Original States Aggregate States

Unstable

: Feature-based parametric architecture State

Vector of weights Original States Aggregate States

a

NEWTON STEP for Bellman Eq.

Position 'value' Move 'probabilities' Simplify

Aggregate Problem Cost Vector

Region of Attraction of

Vector of weights Original States Aggregate States

Vector of weights Original States Aggregate States and its cost function

Aggregate States Cost

Position 'value' Move 'probabilities' Simplify lookahead moves the starting point of the Newton step closer to

J

*

cy

µ

Cost

J

˜

Off-Line Obtained Player Off-Line Obtained Cost Approxim

The start of the Newton step must be within the region of stability

˜

J

+

bL

&gt;

Choose the Aggregation and Disaggregation Probabilities

γ

P

2

(

)

j

F

(

≈

)

)

J

J

µ

i

)

J

(

E

by 'Solvi of

J

K

µ

)

, r

) or

K

}

+

K

)

+

2

ru

+

{

u

<!-- image -->

i

≈

K

k

∗

∗

+1

i

(

)

,

µ

)

/lscript Stages Riccati Equation Iterates P P 0 P 1 P 2 γ -1 P +1 Cost of Period k Stock Ordered at Period k Inventory System r ( u k ) + cu k x k +1 = x k + u + k -w k Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) ural Network or Other Scheme Form the Aggregate States ural Scheme or Other Scheme Include 'Handcrafted' Features Features F ( i ) of Formulate Aggregate Problem 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem orithm learned multiple games (Go, Shogi) gation and Disaggregation Probabilities work or Other Scheme Form the Aggregate States me or Other Scheme Handcrafted' Features F ( i ) of Formulate Aggregate Problem d' Policy ˆ µ by 'Solving' the Aggregate Problem Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem by 'Solving' the Aggregate Problem Cost ˜ J µ ( F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature ture-based parametric architecture State hts Original States Aggregate States Move 'probabilities' Simplify E {·} egation and Disaggregation Probabilities twork or Other Scheme Form the Aggregate States eme or Other Scheme The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy the lookahead the better of the Newton step must be within the region of stability kahead promotes stability of the multistep lookahead policy es the starting point of the Newton step closer to K ∗ d the better step must be within the region of stability tes stability of the multistep lookahead policy Approximation Error ‖ ˜ J -J ∗ ‖ Performance Error ‖ J ˜ µ -J ∗ ‖ x k 1st Step Future 1st /lscript Steps J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; Enhancements to the Starting Point of Newton Step Unstable L | a + bL | &gt; 1 Stable L | a + bL | &lt; 1 and its cost function J ˜ µ ( x ) = K ˜ L x 2 M M -1 m ˜ µ ( x ) = arg min µ ( T µ ˜ J )( x ) or Multistep lookahead moves the starting point of the Newton step closer to K ∗ Enhancements to the Starting Point of Newton Step Unstable L | a + bL | &gt; 1 Stable L | a + bL | &lt; 1 and its cost function J ˜ µ ( x ) = K ˜ L x 2 M M -1 m ˜ µ ( x ) = arg min µ ( T µ ˜ J )( x ) or Multistep lookahead moves the starting point of the Newton step closer to K ∗ Fig. 8. Illustration of the interpretation of approximation in value space with one-step lookahead as a Newton step that maps ˜ J to the cost function J ˜ µ of the onestep lookahead policy. use explicitly or implicitly some form of approximate PI. We will discuss PI and its relation with rollout later, and we will provide visual interpretations based on their connection with Newton's method. Representative States Controls u are associated with states i Optimal Aggregate Costs r ∗ x 1 Aggregate States Cost ˜ J 0 Cost ˜ J 1 Cost r ∗ *Best Score* Representative States Controls u are associated with states i Optimal Aggregate Costs r ∗ x y 1 y 2 y 3 1 /lscript Stages Riccati Equation Iterates P P 0 P 1 P 2 γ 2 -1 γ 2 P P +1 Cost of Period k Stock Ordered at Period k Inventory System r ( u k ) + cu k x k +1 = x k + u + k -w k Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD C AB C AD C DA C CD C BD C DB C AB Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) ˜ J µ ( F ( i ) , r ) : Feature-based parametric r : Vector of weights Original States Ag Position 'value' Move 'probabilities' S Choose the Aggregation and Disaggreg Use a Neural Network or Other Schem I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Feature Generate Features F ( i ) of Formulate A Generate 'Impoved' Policy ˆ ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture r : Vector of weights Original States Aggregate Stat Position 'value' Move 'probabilities' Simplify {· Choose the Aggregation and Disaggregation Proba Use a Neural Network or Other Scheme Form the I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Pr Player Corrected F ( i ) , r ) of i ≈ J µ ( i ) J µ ( i ) Feature Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregate States I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Generate Features F ( i ) of Formulate Aggregate Problem 6 1 3 2 9 5 8 7 10 Player Corrected ˜ J ˜ J J * Cost ˜ J µ ( F ( Map ˜ J µ ( F ( i ) , r ) : Feature-based parametric architecture State r : Vector of weights Original States Aggregate States Position 'value' Move 'probabilities' Simplify E {·} Choose the Aggregation and Disaggregation Probabilities Use a Neural Network or Other Scheme Form the Aggregat I 1 I q Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features and its cost function J ˜ µ ( x ) = K ˜ L x 2 M M -1 m ˜ µ ( x ) = arg min µ ( T µ Multistep lookahead moves the starting point of the Newton step closer to The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Multistep lookahead moves the starting point of the Newton step closer to The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy L Multistep lookahead moves the starting point of the Newton step closer to K ∗ The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy and its cost function J ˜ µ ( x ) = K ˜ L x 2 M M -1 m ˜ µ ( x ) = arg min µ ( T µ ˜ J )( x ) or Multistep lookahead moves the starting point of the Newton step closer to K ∗ The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy NEWTON STEP Enhancements to the Starting Point of Newton Step Unstable L | a + bL | &gt; 1 Stable L | a + bL | &lt; 1 and its cost function J ˜ µ ( x ) = K ˜ L x 2 M M -1 m ˜ µ ( x ) = arg min µ ( T µ ˜ J )( x ) or Multistep lookahead moves the starting point of the Newton step closer to K ∗ NEWTON STEP Enhancements to the Starting Point of Newton Step Unstable L | a + bL | &gt; 1 Stable L | a + bL | &lt; 1 and its cost function J ˜ µ ( x ) = K ˜ L x 2 M M -1 m ˜ µ ( x ) = arg min µ ( T µ ˜ J )( x ) or Multistep lookahead moves the starting point of the Newton step closer to ∗ Enhancements to the Starting Point of Newton Step Unstable L | a + bL | &gt; 1 Stable L | a + bL | &lt; 1 and its cost function J ˜ µ ( x ) = K ˜ L x 2 M M -1 m ˜ µ ( x ) = arg mi Multistep lookahead moves the starting point of the Newton st The longer the lookahead the better NEWTON STEP for Bellman Eq. Enhancements to the Starting Point of Newton Step Unstable L | a + bL | &gt; 1 Stable L | a + bL | &lt; 1 and its cost function J ˜ µ ( x ) = K ˜ L x 2 M M -1 m ˜ µ ( x ) = arg min µ ( T µ ˜ J )( x ) or Multistep lookahead moves the starting point of the Newton step closer to K ∗ Enhancements to the Starting Point of Newton Step Value Iterations Unstable L | a + bL | &gt; 1 Stable L | a + bL | &lt; 1 and its cost function J ˜ µ ( x ) = K ˜ L x 2 M M -1 m ˜ µ ( x ) = arg min µ ( T µ ˜ J )( x ) or Multistep lookahead moves the starting point of the Newton step closer to K ∗ The longer the lookahead the better Longer lookahead promotes stability of the multistep lookahead policy Value Policy Off-Line Training On-Line Play 1 The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy Off-Line Training On-Line Play 1 for Riccati Equation Approximation Error ‖ ˜ J -J ∗ ‖ Performance Error ‖ J ˜ µ -J ∗ ‖ x k 1st Step Future 1st /lscript Steps J ∗ ( x ) = min u ∈ U ( x ) { g ( x, u ) + αJ ∗ ( f ( x, u ) )} , x ∈ X, ˜ µ ( x ) ∈ arg min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} , x ∈ X ; Fig. 9. Illustration of the interpretation of approximation in value space with multistep lookahead and truncated rollout as a Newton step, which maps the result of multiple VI iterations starting with the terminal cost function approximation ˜ J to the cost function J ˜ µ of the multistep lookahead policy.

∗

J

{·}

2

) Cost function

Generate 'Impoved' Policy ˆ

˜

1

Aggregate costs

r

∗

{·}

Cost function

J

˜

0

(

i

ation in a space of basis functions Plays much better than

/lscript es

F

˜

J J

(

ˆ

J

p

i

r weights)

ctions

≥

≥

J

≥

ˆ

p

ˆ

J

′

with

J

p

with

|

≥

J

+

ˆ

, J

(

≥

J

+

with

- 1 1 W { | ≥ 1 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ u Cost Function Approximation (b) As a result of (a), the quadratic cost coefficients ˜ K and K ˜ L satisfy the quadratic convergence relation | K ˜ L -K ∗ | | ˜ K -K ∗ | 2 &lt; ∞ .

C AB C AD C DA C CD C BD C DB C AB Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . . p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play 1st Game / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 System x k +1 = f k ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 3 5 2 4 6 2 1 /lscript Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 e costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) ams ( i, u, j ) Transition probabilities p ij ( u ) W p led Markov Chain Evaluate Approximate Cost ˜ J µ of Approximate Cost ˜ J µ ( F ( i ) ) of F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i : Feature-based architecture Final Features i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture s : Scalar weights) ctions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π , J ( t ) = 0 } arned multiple games (Go, Shogi) Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) a space of basis functions Plays much better than ransition probabilities p ij ( u ) W p v Chain Evaluate Approximate Cost ˜ J µ of mate Cost ˜ J µ ( F ( i ) ) of , F s ( i ) ) : Vector of Features of i -based architecture Final Features s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture weights) with J ( x k ) → 0 for all p -stable π J ( x k ) → 0 for all p ′ -stable π } Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J , J ( t ) = 0 } Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with x k ) → 0 for all p ′ -stable π + = J J J + , J ( t ) = 0 'Handcrafted' Features ) of Formulate Aggregate Problem ed' Policy ˆ µ by 'Solving' the Aggregate Problem learned multiple games (Go, Shogi) r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) n a space of basis functions Plays much better than Transition probabilities p ij ( u ) W p kov Chain Evaluate Approximate Cost ˜ J µ of imate Cost ˜ J µ ( F ( i ) ) of . , F s ( i ) ) : Vector of Features of i re-based architecture Final Features ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture J ( x k ) → 0 for all p -stable π J ( x k ) → 0 for all p ′ -stable π Off-Line Training On-Line Play Parked/Termination State Infinite Horizon Approximation Subspace Bellman Eq: J (1) = αJ (2) , J (2) = J (2) Controls u ∈ U ( x ) x y Shortest N -Stage Distance x -toy J ∗ (1) = J ∗ (2) = 0 Exact VI: J k +1 (1) = αJ k (2) , J k +1 (2) = J k (2) (2 αr k , 2 αr k ) 1 raining On-Line Play rmination State Infinite Horizon Approximation Subspace Bellman Eq: J (1) = αJ (2) , J (2) = u ∈ U ( x ) st N -Stage Distance x -toy J ∗ (1) = J ∗ (2) = 0 Exact VI: J k +1 (1) = αJ k (2) , J k +1 (2) = r k ) 1 e Play te Infinite Horizon Approximation Subspace Bellman Eq: J (1) = αJ (2) , J (2) = istance x -toy J ∗ (1) = J ∗ (2) = 0 Exact VI: J k +1 (1) = αJ k (2) , J k +1 (2) = 1 min u k ,u k +1 ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x At x k Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 -d 1 1 -1 -d m -1 The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy Off-Line Training On-Line Play 1 The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy Off-Line Training On-Line Play 1 2.3 Visualizing Approximation in Value Space The use of Riccati equations allows insightful visualization of approximation in value space. This visualization, although specialized to linear-quadratic problems, is consistent with related visualizations for more general infinite horizon problems. In particular, in the books [Ber20] and [Ber22a], Bellman operators, which define the Bellman equations, are used in place of Riccati operators, which define the Riccati equations. We will first show that approximation in value space with one-step lookahead can be viewed as a Newton step for solving the Riccati equation; see Fig. 8. In particular, let us consider a quadratic cost function approximation of the form ˜ J ( x ) = Kx 2 , where K ≥ 0. We will show that: (a) An iteration of Newton's method for solving the Riccati equation K = F ( K ), starting from a value ˜ K yields the quadratic cost coefficient K ˜ L of the cost function J ˜ µ of the one-step lookahead policy ˜ µ , which is linear of the form ˜ µ ( x ) = ˜ Lx and has cost function J ˜ µ ( x ) = K ˜ L x 2 . Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . . p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play 1st Game / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 System x k +1 = f k ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 3 5 2 4 6 2 1 Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 Approximation in a space of basis functions Plays much better than all chess programs Cost α g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π W + = { J | J ≥ J + , J ( t ) = 0 } 1 Same algorithm learned multiple games Aggregate costs r ∗ /lscript Cost function Approximation in a space of basis func all chess programs Cost α k g ( i, u, j ) Transition probabilitie Controlled Markov Chain Evaluate Ap Evaluate Approximate Cost ˜ J µ ( F ( i ) ) o F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Fea ˜ J µ ( F ( i ) ) : Feature-based architecture F If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a line ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 W p ′ : Functions J ≥ ˆ J p ′ with W + = { J | J ≥ J + , J ( t ) = 0 Generate 'Impoved' Policy ˆ µ by 'Solving' the Agg Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost functi Approximation in a space of basis functions Plays all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Feature If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-b ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -sta W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → W + = { J | J ≥ J + , J ( t ) = 0 } 1 Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Problem Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function ˜ J 1 ( j ) Approximation in a space of basis functions Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based architecture ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p ′ -stable π 1 Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate P Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function ˜ J 0 ( i ) Cost function Approximation in a space of basis functions Plays much bet all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cost ˜ J µ of Evaluate Approximate Cost ˜ J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Final Features If ˜ J µ ( F ( i ) , r ) = ∑ s /lscript =1 F /lscript ( i ) r /lscript it is a linear feature-based arch ( r 1 , . . . , r s : Scalar weights) W p : Functions J ≥ ˆ J p with J ( x k ) → 0 for all p -stable π W p ′ : Functions J ≥ ˆ J p ′ with J ( x k ) → 0 for all p 1 Value Policy Off-Line Training On-Line Play Parked/Termination State Infinite Horizon Approximation Subspace Bellman Eq: αJ (2) Controls u ∈ U ( x ) x y Shortest N -Stage Distance x -toy J ∗ (1) = J ∗ (2) = 0 Exact VI: J k +1 (1) = αJ k (2) , J αJ k (2) (2 αr k , 2 αr k ) 1 Off-Line Training On-Line Play Parked/Termination State Infinite Horizon Approximation Subspace Bellman Eq: αJ (2) Controls u ∈ U ( x ) x y Shortest N -Stage Distance x -toy J ∗ (1) = J ∗ (2) = 0 Exact VI: αJ k (2) (2 αr k , 2 αr k ) 1 Off-Line Training On-Line Play Parked/Termination State Infinite Horizon Approximation Subspace Bellman Eq: J (1) = αJ (2) αJ (2) Controls u ∈ U ( x ) x y Shortest N -Stage Distance x -toy J ∗ (1) = J ∗ (2) = 0 Exact VI: J k +1 (1) = αJ k (2) , J k +1 (2) = αJ k (2) (2 αr k , 2 αr k ) 1 Value Policy Off-Line Training On-Line Play Parked/Termination State Infinite Horizon Approximation Subspace Bellman Eq: J (1) = αJ (2) , J (2) = αJ (2) Controls u ∈ U ( x ) x y Shortest N -Stage Distance x -toy J ∗ (1) = J ∗ (2) = 0 Exact VI: J k +1 (1) = αJ k (2) , J k +1 (2) = αJ k (2) (2 αr k , 2 αr k ) 1 The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy Off-Line Training On-Line Play 1 The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy Off-Line Training On-Line Play 1 The start of the Newton step must be within the region of sta Longer lookahead promotes stability of the multistep lookahea Value Policy Off-Line Training On-Line Play 1 The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy Off-Line Training On-Line Play 1 The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy Off-Line Training On-Line Play 1 min u k ,u k +1 ,...,u k + /lscript -1 { /lscript -1 ∑ m =0 α m g ( x k + m , u k + m ) + α /lscript ˜ J ( f ( x k + /lscript -1 , u k + /lscript -1 ) ) } min u ∈ U ( x ) { g ( x, u ) + α ˜ J ( f ( x, u ) )} u k At x At x k Input (Control) Output (Function of the State) Changing Fixed . . . Time 0 Time k Transformer Heuristic Region of convergence d θ x l Stage N u = ( u 0 , . . . , u N -1 ) (˜ u 0 , . . . , ˜ u k -1 , u k , u k +1 , . . . , u N -1 ) Time k +1 Time k + m i j b 0 b 1 b m -2 b m -1 0 1 d 1 d 2 d m -1 d m d 1 m m -1 . . . 1 -b 0 1 -b 1 -d 1 1 -b m -2 1 -b m -1 -d m -1 1 -d 1 1 -d 2 1 -d m -1 1 -d m ( u 0 , . . . , u k , u k , ˜ u k +1 , . . . , ˜ u N -1 ) for all ˜ u k +1 Cost Function Approximation set-membership uncertainty. The reason for this generality is the universal character of the corresponding mathematical proof arguments, which rely on the theory of abstract DP. For the case of multistep lookahead minimization, which typically underlies the MPC architecture, we will also show that the Newton step property holds. Indeed, this property is enhanced, because the region of convergence of Newton's method is enlarged by longer lookahead , as we will argue graphically later. The extension of the Newton step interpretation is not surprising because, as noted earlier, we may view ℓ -step lookahead as a one-step lookahead where the cost function approximation is the optimal cost function of an ( ℓ -1)-stage DP problem with a terminal cost ˜ J ( x k + ℓ ) on the state x k + ℓ obtained after ℓ -1 stages; see Fig. 9. Indeed, let us first consider one-step lookahead minimization with any terminal cost function approximation of the form ˜ J ( x ) = Kx 2 , where K ≥ 0. The one-step lookahead control at state x , which we denote by ˜ µ ( x ), is obtained by minimizing the right side of Bellman's equation when J ( x ) = Kx 2 :

k

k

+1

- 1 (c) As a result of (b), for ˜ K within the region of convergence of Newton's method, the one-step lookahead policy cost function J ˜ µ tends to be closer to J ∗ than ˜ J , and for ˜ J close to J ∗ , the policy ˜ µ is very close to optimal.

These facts admit a simple proof for the linear-quadratic case, but qualitatively hold in great generality, i.e., for arbitrary state and control spaces, for finite and infinite horizon problems, and in the presence of stochastic and

|

˜

J

µ

∗

Cost

µ

/lscript

≥

(

x

J

(

+

d

2

1

}

) = 0

J

J

0

(

)

J

i

(1) =

α

˜

, J

(2) =

˜

J

(1) =

′

J

1

(

j

αJ

)

(1) =

αJ

k

-stable

(2)

π

k

x

)

J

(

0 for all

}

1

We can break this minimization into a sequence of two minimizations as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the function F L ( K ) is the Riccati equation operator for the generic linear policy µ ( x ) = Lx . Figure 10 illustrates the two-step minimization of the preceding equation, and shows how the graph of the Riccati operator

,

0

p

′

-st

→

K

+

µ

(

i

i

(

˜

J J

*

K

∗

+

rL

b

2

L

∈/Rfractur

∈/Rfractur

F

L

(

L

bL

)

∈/Rfractur

}

) = (

a

}

{

}

}

{

{

}

}

{

{

2

Kx

+

p

′

t

+

+

˜

ru ru

+

+

(

3

One-Step Lookahead Policy Cost l

µ

= min

TJ

0

y

-Тл + 6

n of appr

1) =a+

+6+ 4.

aon here

1

T

2

H

(

y

) =

G

(

)

1

-

TJ

T

µ

J

Cost 28 Cost 27 Cost 13

y G

(

y

) Region of Attraction of ne tho

e of Quadratic Functions

Given quadratic cost approximation

One-Step Lookahead Policy Cost l

= min

J J

µ

˜

=

T

µ

˜

J

µ

˜

T

µ

˜

J

One-Step Lookahead Policy Cost l

TJ

Bellman Operator Value Iterations Largest Invariant Set n quadratic cost approximation

˜

∗

J

x

(

) =

= min

= min

µ

Value Space Approximation

˜

y

J

Kx

˜

2

, we find

µ

T

µ

J

Multistep Lookahead Policy Cost l

) =

˜

Kx

2

˜

, we find

TJ

= min

TJ

T

µ

J

(

x

J

(

x

) =

L

˜

Kx

˜

2

= arg min

F

L

J J

µ

˜

=

T

µ

˜

J

µ

˜

T

µ

˜

J

L

J J

y

L

K

= arg min

˜

µ

˜

µ

T

TJ

)

G

) =

(

( ˜

G

˜

µ

µ

T

µ

µ

J

One-Step Lookahead Policy Cost l

One-Step Lookahead Policy Cost l

= min

T

J

=

T

µ

˜

J

µ

˜

Bellman Equation on Space of Quadratic Functions

TJ

= min

Multistep Lookahead Policy Cost

T

µ

˜

J

µ

J

-

One-Step Lookahead Policy Cost l

c

(2)

1)

c

(

F

L

m

1)

c

(

m

(

M

)

(

(

m

c

)

c

n

µ

˜

J J

˜

m

+1)

c

ent Riccati Operator at y Cost l

=

˜

J J

able

L

a

+

L

|

k

m

K

k

k

u

w

its cost function

)

P

˜

P

k

P

Improvement Holds

)

Newton Step also

0

k Q

-

B

E

{

)

x

0

k

2

=

J

k

˜

J J

J J

˜

µ

=

T

J

T

J J

˜

˜

µ

µ

˜

Value Space Approximation

µ

˜

J

˜

µ

˜

µ

Bellman Operator Value Iterations

=

T

µ

˜

J

µ

˜

˜

T

J

J J

=

Interval I Interval II Interval III Interval IV

Multistep Lookahead Policy Cost l

= min

0

µ

µ

˜

T

J

2

µ

J

1

TJ

)

y G

(

= min

Multistep Lookahead Policy Cost l

Multistep Lookahead Policy Cost l

-

=

T

˜

µ

K

µ

µ

˜

˜

µ

˜

J J

J J

K

˜

s

(0

∈

J

=

∗

˜

µ

=

T

J

T

)

J

˜

µ

+(1

J

µ

) =

=

Kx

(

y

H

c

Bellman Equation on Space of Qu

= min

T

µ

Multistep Lookahead Policy Cost l

T

µ

) =

M

(

y

)

H

y

y G

(

T

µ

=

y

)

J

µ

˜

J

(

x

J

1) to construct the one-step lookahead policy ˜

J J

˜

)

-

µ

˜

T

µ

J

µ

˜

µ

˜

J

TJ

µ

(

y

TJ

T

µ

J

m

+1)

c

( ˜

(

K

M

)

)

L

(

Tube Constraint Cannot be Satisfied for all

1) to construct the one-step lookahead policy ˜

+

) = ˜

if a &gt;

1

F

(

(

x

-

K

K

µ

µ

µ

µ

˜

S

) = ˜

T

µ

J

Lx

) 45

One-Step Lookahead Policy C

µ

˜

˜

J TJ

˜

2

˜

µ

=

Multistep Lookahead Policy Cost

x

Multistep Lookahead Policy Cost

α

0

µ

(

X

x

x β

Multistep Lookahead Policy Cost

K

J

u

µ

˜

k

(

∈

s

y

(0

∈

,

K

)

1]

T

Rollout Policy Network

)

a

+

bL

TJ

≤

Possible States

X

˜

=

Region where Sequential Improvement Holds

X

J T

˜

µ

J

µ

˜

T

) = arg min

˜

˜

k

x

µ

J

J

˜

K µ

˜

µ

J

2

(

T

k

T

˜

µ

+1

J

J

-

Cost 28 Cost 27 Cost 13 Lookahead Controller

µ

µ

˜

J J

=

Instability Region Stability Region 0

Instability Region Stability Region 0

TJ

T

=

(Sudden death)

˜

µ

arK

2-State/2-Control Example

µ

k

+1

P

P

Effective Cost Approximation Value Space Approximation State 1

J

µ

˜

˜

µ

=

,

1]

K

2

˜

Newton iterate starting from

α

J

˜

= min

(

µ

y

)

∈

)

≤

(0

s

T

T

˜

µ

µ

J

K

,

T

J J

J T

J

Tube Constraint Cannot be Satisfi

TJ

J T

∗

˜

µ

Region where Sequential Im

1]

T

+1

x

k

Possible States

n

p

xy min

(

T

µ

J

=

µ

(

x

)

˜

Instability Region Stabilit

˜

X

X

Multi

y

=1

-

J T

µ

J

TJ

˜

J

R

Value Space Approximatio

∑

≤

Multiagent Q-factor minimi

µ

J

=

Effective Cost Approximation Value Space Appro

Termination State Constrai

T

One-Step Lookahead Polic

-

µ

using an Corresponds to One-

˜

µ

J

˜

µ

=

k

∗

K K

K

k

k

+

2-State/2-Control Example

Effective Cost Approximation Value Space Appr

Cost of ˜

µ

Current Partial Folding Mo

Newton iterate starting fr

Line Stable Policies Unstable

Cost 28 Cost 27 Cos

=

Define

µ

˜

J J

-

<!-- image -->

Region where Sequentia

Complete Folding Correspo

Generic unstable polic

˜

µ

(

y

)

E

m

{·}

)

(2)

:

J

(

=1

K K

Riccati

=

T

˜

µ

Multistep Lookahead Policy Cost l

Newton iterate starting from

∗

2

∑

x

3

˜

+

q

rL

F

+

(

K

)

µ

µ

J

T

Region where Sequential Improvement H

∗

˜

µ

J

T

Effective Cost Approximation Value Space Approximation

2

˜

µ

b

AD

µ m

(

T

=

<!-- image -->

)

c

c

(

(

M

-

◦

<!-- image -->

K

+

q

One-Step Lookahead Policy Cost l

k

J J

=

2

ar

˜

µ

b

˜

µ

k at Period k +1 Initial State A C AB AC CA CD ABC B ACD CAB CAD CDA S B C AB C AC C CA C CD C BC C CB C CD C AD C DA C CD C BD C DB C AB ot Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . . p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn Game / Timid Play 2nd Game / Bold Play ame / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w Line Stability Region F ( K ) = arK r + b 2 K + q T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Defined by Newton step from ˜ J for solving J = TJ State 2 ( TJ )(1) ˜ J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ -r b 2 Generic stable policy µ T µ J Generic unstable policy µ ′ T µ ′ J Cost of Truncated Rollout Policy ˜ µ 1 of the graph of T J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J ∗ (2) TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J Multistep Lookahead Policy Cost J is a function of x F L ( K ) = ( a + bL ) 2 K + q + rL 2 F ˜ L ( K ) ˜ J Region where Sequential Improvement Holds TJ ≤ J T ˜ µ J ˜ K TJ Instability Region Stability Region using an Corresponds to One-Step Lookahead Policy ˜ µ Line ch Win Probability 1 0 p w (Sudden death) Iteration: K k +1 = F ( K k ) quation: K = F ( K ) k ( x ) or K k +1 = F ( K k ) from ( x ) or K k +1 = F ( K k ) from ne-Step Lookahead Policy ˜ µ ble Policy Optimal Policy e of Newton's Method Riccati Equation Optimal Policy Riccati Equation: K = F ( K ) J ( x ) = Kx 2 = F ( K ) x 2 = J k ( x ) or K k +1 = F ( K k ) from J k +1 ( x ) = K k +1 x 2 = F ( K k ) x 2 = J k ( x ) or K k +1 = F ( K k ) from using an Corresponds to One-Step Lookahead Policy ˜ µ Line Stable Policies Unstable Policy Optimal Policy Region of stability Also Region of Convergence of Newton's Method Riccati Equation Cost of rollout policy ˜ µ Cost of base policy µ The longer the lookahead the better The start of the Newton step must be within the region of stability Longer lookahead promotes stability of the multistep lookahead policy Value Policy Off-Line Training On-Line Play Parked/Termination State Infinite Horizon Approximation Subspace Bellman Eq: J (1) = αJ (2) , J (2) = αJ (2) Controls u ∈ U ( x ) x y Shortest N -Stage Distance x -toy J ∗ (1) = J ∗ (2) = 0 Exact VI: J k +1 (1) = αJ k (2) , J k +1 (2) = αJ k (2) (2 αr k , 2 αr k ) Terminal Position Evaluation longer the lookahead the better start of the Newton step must be within the region of stability er lookahead promotes stability of the multistep lookahead policy e Policy ine Training On-Line Play ed/Termination State Infinite Horizon Approximation Subspace Bellman Eq: J (1) = αJ (2) , J trols u ∈ U ( x ) Shortest N -Stage Distance x -toy J ∗ (1) = J ∗ (2) = 0 Exact VI: J k +1 (1) = αJ k (2) , J k +1 (2) = αr k , 2 αr k ) Rollout with Base Policy m -Step Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Newton iterate starting from K K L ˜ K = -ab ˜ K r + b 2 ˜ K a -a ˜ L = -ab ˜ K r + b 2 ˜ K ˜ L = -abK 1 r + b 2 K 1 Slope = 1 ˜ J Region where Sequential Improvement Holds TJ ≤ J T ˜ µ J ˜ K µ K 1 Fig. 10. Illustration of how the graph of the Riccati operator F can be obtained as the lower envelope of the linear operators F L ( K ) = ( a + bL ) 2 K + q + bL, as L ranges over ℜ , i.e. F ( K ) = min L ∈ℜ F L ( K ) . Moreover, for any fixed ˜ K , the scalar ˜ L that attains the minimum is given by ˜ L = -ab ˜ K r + b 2 ˜ K , and is such that the line corresponding to the graph of F ˜ L is tangent to the graph of F at ˜ K , as shown in the figure. F can be obtained as the lower envelope of the linear operators F L , as L ranges over the real numbers. Let us now fix the terminal cost function approximation to some ˜ Kx 2 , where ˜ K ≥ 0, and consider the corresponding one-step lookahead policy ˜ µ . Figure 11 illustrates the corresponding linear cost function F ˜ L of ˜ µ , and shows that its graph is a tangent line to the graph of F at the point ˜ K (cf. Fig. 10). ˜ F ( P ) k Q 0 P -R E { B 2 } 45 ◦ Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD C AB C AD C DA C CD C BD C DB C AB Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play Expert Rollout with Base Policy Approximation of min u ∈ U ( x ) n y p xy x k , u k u k x k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k ( x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 Expert Rollout with Base Policy m -Step Approximation of E {·} : Approximate minimiz min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Roll Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 T ˜ µ T µ m ˜ J = TT µ m ˜ J using an Corresponds to One-Step Lookahead Policy ˜ Line Stability Region F ( K ) = arK r + b 2 K + q T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Defined by Newton step from ˜ J for solving J = TJ ˜ L = -r + ab 2 ˜ K ab ˜ K using an Corresponds to One-Step Lookahead Policy ˜ µ Line Stability Region F ( K ) = arK r + b 2 K + q T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Defined by Newton step from ˜ J for solving J = TJ ˜ L = -r + ab 2 ˜ K ab ˜ K using an Corresponds to One-Step Lookahead Policy ˜ µ Line Stability Region F ( K ) = arK r + b 2 K + q T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Newton step from ˜ J for solving J = TJ Generic stable policy µ T µ J Cost of Truncated Rollout Policy ˜ µ 1 of the graph J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J ∗ TJ = min µ T µ J One-Step Lookahead Policy Cost l Multistep Lookahead Policy Cost J is a function of F L ( K ) = ( a + bL ) 2 K + q + rL 2 F ˜ L ( K ) State 2 ( TJ )(1) ˜ J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ Generic stable policy µ T µ J Generic unstable poli Cost of Truncated Rollout Policy ˜ µ 1 of the graph J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = TJ = min µ T µ J One-Step Lookahead Policy Cost l TJ = min µ T µ J Multistep Lookahead Policy Cost l Multistep Lookahead Policy Cost J is a function Region of stability Also Region of Convergence of ˜ J TJ Instability Region Mat Stability Region Slope=1 also Newton Step Value Optimal Policy J ( x ) = Kx 2 = F ( K ) x 2 = J k +1 ( x ) = K k +1 x 2 = F ( K using an Corresponds to Fig. 11. Illustration of approximation in value space with one-step lookahead. Given a terminal cost approximation ˜ J = ˜ Kx 2 , we compute the corresponding linear policy ˜ µ ( x ) = ˜ Lx , where ˜ L = -ab ˜ K r + b 2 ˜ K , and the corresponding cost function K ˜ L x 2 , using the Newton step shown. min u ∈ U ( x ) n ∑ y =1 p xy Effective Cost Approximation Value Space Approximation ˜ J J ˜ µ J ˜ µ T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ Cost of Truncated Rollout Policy ˜ µ TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J Multistep Lookahead Policy Cost F L ( K ) = ( a + bL ) 2 K + q + rL 2 F ˜ L ( K ) T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ J T ˜ µ J T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ Cost of Truncated Rollout Policy ˜ µ TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ Multistep Lookahead Policy Cost F L ( K ) = ( a + bL ) 2 K + q + rL 2 F ˜ L ( K ) T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ ˜ µ ˜ µ Cost of Truncated Rollout Policy ˜ µ TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T J TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J Multistep Lookahead Policy Cost F L ( K ) = ( a 2 2 T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ J T ˜ µ J TJ Instability Region Stability Region 0 T µ m ˜ J Value Space Approximation One-Step Lookahead Policy C ˜ J Region where Sequential Im TJ Instability Region Stabilit Effective Cost Approximation Value Space Approximation T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ Cost of Truncated Rollout Policy ˜ µ TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J Multistep Lookahead Policy Cost F L ( K ) = ( a + bL ) 2 K + q + rL 2 T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ J T J R 0 R 1 R 2 T 2 Value Space Approximation ˜ J J ˜ µ One-Step Lookahead Policy Cost l Newton iterate starting from ˜ L = -ab ˜ K r + b 2 ˜ K ˜ L = -abK 1 r + b 2 K 1 Slope ˜ J Region where Sequential Improv F ( K ) Interval I Interval II Interval III Interval IV K s K ∗ J 0 J µ = -1 µ T µ J = -µ +(1 -µ 2 ) J TJ = min µ ∈ (0 , 1] T µ J ˜ L = -ab ˜ K r + b 2 ˜ K

rL

2

∗

Define

J

∗

˜

u

0

˜

Value Space Approximati

(2

˜

x

1

Multiagent Q-factor minim

F

(

K

1

High Cost Transition Chose

Line Stable Policies Unsta

L

µ

T

J

=

µ

One-Step Lookahead Poli using an Corresponds to One-

Capacity=1 Optimal Soluti

Termination State Constrai

µ m

T

J

˜

-

(1)

˜

,

+

r

b

2

Defined by

{·}

n

p

1

K

1

˜

1 of the graph of

}

Q

x

(

)

Valu xy

) }

=1

Riccati

(

x,

Defin

(2)

˜

+

bL

)

K

= min

TJ

1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 em x k +1 = f k ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 2 4 6 2 5 7 8 3 9 6 1 2 al Temperature x 0 u 0 u 1 x 1 Oven 1 Oven 2 Final Temperature +1 = A k y k + ξ k y k +1 C k w k hastic Problems 1 1 T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ J T ˜ µ J TJ Instability Region Stability Region 0 T µ m ˜ J a 2 r b 2 a 2 r b 2 + q q F ( K ) = a 2 rK r + b 2 K + q K ∗ = 0 ˜ K = 0 ¯ K ¯ K = 0 K ˜ L ˜ L = -ab ˜ K r + ab 2 ˜ K K 1 ˜ L = -abK 1 r + b 2 K 1 F ( K ) = a 2 rK r + b 2 K J ∗ (1) = 0 J (1) ( TJ )(1) = min { J (1) , 1 } 1 st of base policy µ 1 1 1 inal Position Evaluation 1 Base Policy Rollout Policy Approximation in Value Space n n -1 n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost Approximation in Policy Space Heuristic Cost Approximation for for Stages Beyond Truncation y k Feature States y k +1 Cost g k ( x k , u k ) Approximate Q-Factor ˜ Q ( x, u ) At x Approximation ˆ J min u ∈ U ( x ) E w { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) ) } Truncated Rollout Policy µ m Steps 1 Thus the function F ˜ L can be viewed as a linearization of F at the point ˜ K , and defines a linearized problem: to find a solution of the equation K = F ˜ L ( K ) = q + b ˜ L 2 + K ( a + b ˜ L ) 2 . The important point now is that the solution of this equation, denoted K ˜ L , is the same as the one obtained from a single iteration of Newton's method for solving the Riccati equation, starting from the point ˜ K . This is illustrated in Fig. 11. To elaborate, let us note that the classical form of Newton's method for solving a fixed point problem of the form y = T ( y ), where y is an n -dimensional vector, operates as follows: At the current iterate y k , we linearize T and find the solution y k +1 of the corresponding linear fixed point problem. Assuming T is differentiable, the linearization is obtained by using a first order Taylor expansion: y k +1 = T ( y k ) + ∂T ( y k ) ∂y ( y k +1 -y k ) , 1st Game / Timid Play 1st Game / 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 ---System x k +1 = f k ( x k , u k , w k ) u k = µ k ( x k ) µ k w k x k 3 5 2 4 6 2 10 5 7 8 3 9 6 1 2 Initial Temperature x 0 u 0 u 1 x 1 Oven 1 Oven 2 Final Temperature x 2 ξ k y k k y k + ξ k y k +1 C k w k Permanent Trajectory Tent sen by Base Heuristic at x 0 Initi Base Policy Rollout Policy n -2 One-Step or Multistep Loo Approximation in Policy S for Stages Beyond Truncati Approximate Q-Factor min u ∈ U ( x ) E w { sen by Base Heuristic at n -2 One-Step or Multistep Lookahead for stages Possible Approximation in Policy Space Heuristic Cost Appro for Stages Beyond Truncation y k Feature States y k +1 Approximate Q-Factor ˜ Q ( x, u ) At x Approximation J min u ∈ U ( x ) E w { g ( x, u, w ) + α ( Truncated Rollout Policy µ m Steps 1 1 1 1 1 T 2 ˜ J T ˜ J ˜ J TJ a 2 r b 2 a 2 r b 2 + q q F ( K ) = a 2 rK r + b 2 K + q K ∗ = 0 ˜ K = 0 ¯ K ˜ L = -ab ˜ K r + ab 2 ˜ K K 1 ˜ L = -abK 1 r + ab 2 K 1 F ( K ) = a 2 rK r + b 2 K J ∗ (1) = 0 J (1) ( TJ )(1) = min { J 1 T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement TJ Instability Region Stability Region 0 T µ m ˜ J a 2 r b 2 a 2 r b 2 + q q F ( K ) = a 2 rK r + b 2 K + q K ∗ = 0 ˜ K = 0 ¯ ˜ L = -ab ˜ K r + ab 2 ˜ K K 1 ˜ L = -abK 1 F ( K ) = a 2 rK r + b 2 K J ∗ (1) = 0 J (1) ( TJ )(1) = min { J (1) , 1 } Also Region of Convergen Cost of rollout policy ˜ µ C r b 2 +1 1 -r b 2 K K ∗ K k k k Current Partial Folding Mo Complete Folding Correspo Expert Rollout with Base Policy Approximation of E : min u ∈ U ( x ) ∑ y x 1 , u 1 u 2 x 2 d τ TJ Instability Region Stability Region 0 T µ m ˜ J ar b 2 + q q F ( K ) = arK r + b 2 K + q ˜ K ˜ L = -r + ab 2 ˜ K ab ˜ K using an Corresponds to One-Step Lookahead Policy ˜ µ Line Stability Region F ( K ) = arK r + b 2 K + q T ˜ µ ( T µ m ˜ J ) = T ˜ ) Yields Truncated Rollout Policy ˜ µ Defined by TJ ar b 2 + q q F ( K ) = arK r + b 2 K + q ˜ K K ˜ L ˜ L = -r + ab 2 ˜ K ab ˜ K using an Corresponds to One-Step Lookahead Policy ˜ µ Line Stability Region F ( K ) = arK r + b 2 K + q T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ ar b 2 + q q F ( K ) = arK r + b 2 K + q ˜ K = 0 ˜ L = -r + ab 2 ˜ K ab ˜ K K 1 ˜ L = -abK 1 using an Corresponds to One-Step Lookahead Policy ˜ µ Line Stability Region F ( K ) = arK r + b 2 K + q F ˜ L ( K 1 ) T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Newton step from ˜ J for solving J = TJ Line Stable Policies Unstable Region of stability Also Region of Convergence of TJ Instability Region Stability Region 0 ar b 2 + q q F ( K ) = arK r + b 2 K + q ˜ K using an Corresponds to One-Step Lookahead Policy ˜ µ Line Stability Region F ( K ) = arK r + b 2 K + q T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Defined by Newton step from ˜ J for solving J = TJ x k L k u k w k x k +1 = A k x k + B k u k + w k F ( P ) ˜ P P k P k +1 P ∗ Q 0 ˜ P R A 2 ˜ F ( P ) k Q 0 P -R E { B 2 } 45 ◦ Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD C AB C C DA C CD C BD C DB C AB . Newton iterate starting f ˜ J Region where Sequenti TJ Instability Region Ma Stability Region Slope=1 also Newton Step Optimal Policy State 1 State 2 K ∗ ¯ K 2-State/2-Control Example Effective Cost Approximation Value Space Approx State 2 ( TJ )(1) ˜ J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ -r b 2 Generic stable policy µ T µ J Generic unstable policy Cost of Truncated Rollout Policy ˜ µ J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J Fig. 12. Illustration of approximation in value space with two-step lookahead. Starting with a terminal cost approximation ˜ J = ˜ Kx 2 , we obtain K 1 using a single value iteration. We then compute the corresponding linear policy ˜ µ ( x ) = ˜ Lx , where ˜ L = -abK 1 r + b 2 K 1 and the corresponding cost function K ˜ L x 2 , using the Newton step shown. The figure shows that for any K ≥ 0, the corresponding ℓ -step lookahead policy will be stable for all ℓ larger than some threshold.

=

˜

J

A

g

∗

1 where ∂T ( y k ) /∂y is the n × n Jacobian matrix of T evaluated at the vector y k . For the linear quadratic problem, T is equal to the Riccati operator F , and is differentiable. However, there are extensions of Newton's method that are based on solving a linearized system at the current iterate, but relax the differentiability requirement to piecewise differentiability, and/or component concavity (here the role of the Jacobian matrix is played by subgradient operators). The quadratic or similarly fast superlinear con+1 Stochastic Problems 1 Truncated Rollout Policy µ 1 k k k k k Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 High Cost Transition Chos Newton step from ˜ J for solving J = TJ Newton step from ˜ J for solving J = TJ Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play 1st Game / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w J ( x ) = Kx 2 = F ( K ) x 2 J k +1 ( x ) = K k +1 x 2 = F ( using an Corresponds to Line Stable Policies Unst TJ = min µ T µ J One-Step Lookahead Policy Cost l J TJ = min µ T µ J Multistep Lookahead Policy Cost l J Multistep Lookahead Policy Cost J is a function of F L ( K ) = ( a + bL ) 2 K + q + rL 2 F ˜ L ( K ) vergence property is maintained in these extended forms of Newton's method; see the monograph [Ber22a] (Appendix A) and the paper [Ber22c], which provide a convergence analysis and discussion related to the DP/MPC context. The preceding argument can be extended to ℓ -step lookahead minimization to show that a similar Newton step interpretation is possible (Fig. 12 depicts the case ℓ = 2). Indeed in this case, instead of linearizing F at ˜ K , we

0

-

1

0

System

-

x

k

0

+1

0

-

=

f

1

k

(

1

.

x

k

-

5

, u

k

0

5

.

, w

k

1

-

T

u

2

k

1

˜

J T

=

˜

.

5

˜

1

J

µ

J

k

(

-

(

Capacity=1 Optimal Soluti

Region where Sequential Improvement Ho

x

k

1

)

5

µ

-

Region of stability

0

2

Permanent Trajectory Tent sen by Base Heuristic at

k

w

k

x

k

x

m

0

Initi

Also Region of Converge

0

.

x, u, w

J

J

f

(

J

T

˜

˜

µ

J TJ

T

T

= min

(

K

=

F

(

K

)

+

q q F

) =

K

k

+1

T

˜

µ

˜

µ

-

T

= min

=

T

J

˜

µ

J J

-

2

}

Value Iteration:

)

ewton iterate starting from

Painl

K

s the lenath. of

Policy Improv head.

One-Step Lookahead Policy Cost l

Step Lookahead Policy Cost l

Policy Cost l

Region where Sequential Improvement Holds

˜

µ

˜

=

T

µ

˜

J

Cost-to-go approximation Expected value approximation

˜

µ

T

µ

˜

J

Optimal cost

J

J

µ

1

(

x

)

/J

µ

0

(

x

) =

K

1

/K

J J

Cost-to-go approximation Expected value approximation

T

µ

˜

TJ

J

˜

J T

Optimal cost

J

∗

µ

˜

J J

˜

µ

˜

=

T

µ

˜

J

µ

˜

T

µ

˜

J

J J

µ

˜

=

T

µ

˜

J

µ

˜

Newton iterate starting from on iterate starting from

g from

K

K

≤

J

K µ

˜

T

µ

Tangent Line of

Tangent Line of

K

˜

T

using an Corresponds to One-Step Lookahead Policy ˜

K

J J

µ

=

T

µ

J

∗

µ

J

˜

µ

=

T

˜

µ

Cost of rollout policy ˜

J

˜

µ

0

L

0

r

Cost of base policy

Optimal Base Rollout

µ

Cost of base policy using an Corresponds to One-Step Lookahead Policy ˜

Instability Region Stability Region

µ

J

µ

µ

J

µ

˜

=

T

µ

˜

J

µ

˜

µ

J

policy ˜

µ

Simplified minimization

J

µ

=

T

Simplified minimization

Region where Sequential Improvement Holds gion where Sequential Improvement Holds

K

K µ

˜

µ

µ

Value iterations

Line Stable Policy Unstable Policy Region of stability

s

K

n

p

1]

,

(0

Optimal cost ntial Improvement Holds

Instability Region Stability Region Slope=1

nstability Region Stability Region Slope=1

Stability Region Slope=1

(

)

∈

min

y

xy

=1

x

T

∑

µ

J J

µ

Multiagent Q-factor minimization

Cost of rollout policy ˜

ne Stable Policies Unstable Policy Optimal Policy

Termination State Constraint Set

µ

˜

T

using an Corresponds to One-Step Lookahead Policy ˜

k

˜

µ

=

Simplified minimization

K K K

∗

K

Current Partial Folding Moving Obstacle egion of stability

to One-Step Lookahead Policy ˜

x

+1

=

A

Line Stable Policies Unstable Policy Optimal Policy

Stable Policies Unstable Policy Optimal Policy

k

∗

+1

P

P

nstable Policy Optimal Policy ions

J

(

x

) =

R

on of stability

E

{

B

J J

µ

T

J

˜

Complete Folding Corresponding to Open

k

x

+1

=

µ

˜

=

Steps

Cost Function Approximation Position Evaluator

2-State/2-Control Example

P

Multiagent Q-factor minimization

+1

Rollout with Base Policy

E

∗

k

P

Termination State Constraint Set

Effective Cost Approximation Value Space Approximation State 1

P

Also Region of Convergence of Newton's Method

-

Region where Sequential Improvement Holds

Region of Convergence of Newton's Method

<!-- image -->

˜

C

CD

{·}

1 Lookahead Minimization Steps

AC

Robust Base Policy Figure

/epsilon1

m

Steps of Rollout

∗

µ

Cost of rollout

Value Space Approximation

˜

J

One-Step Lookahead Policy Cos yy

(

u

)

p

(

u

)

Newton iterate starting from

p

-

K

µ

p

K

yt yt

(

)

(

u

u

1

2

x y

)

µ

x y

+1

(

)

αp

-

Region where Sequential Impr yx

u

)

α

Instability Region Stability

n

min

x

)

(

) 1

p

+(1

∑

-

-

α

α

y

=1

µ

2

xy

)

(

u

J

ˆ

Cost of base polic

-

+1

x

k

m

+

+1

Multiagent Q-factor minimiz using an Corresponds to One-St

Termination State Constraint

Multiagent

Line Stable Policies Unstable Po

K

<!-- image -->

r

b

2

k

n

P

ABC

CA

AB

CD

u

-

J J

Current Partial Folding Moving Obstacle

˜

µ

using an Corresponds to One-Step Lookahead Policy ˜

Cost of Truncated Rollout Policy ˜

J

Interval I Interval II Interval III Interval IV

C

0

CD

C

u

x

µ

1

0

bL

ˆ

u

u

1

∗

10 11 12

x

+(1

0

∗

1

u

C

AB

C

System

+

µ

1

x

µ

q

k

2

2

+

2

K

)

)

J TJ

2

3

k

x

+1

=

0

f

rL

k

2

(

u

∗

n

x

k

b

µ

0

k

2

3

2

1

K K

∗

K

k

k

k

+1

Current Partial Folding Movi

µ

J

Cost of ˜

µ

µ

˜

{

T

µ

˜

r

2

+

αb

J TJ

= min

}

µ

T

{

.

n

.

k

k

k

k

r

2

k

, u

x

u

d

2

1

1

2

}

n

y

=1

µ

∑

=

T

τ

x

Multistep Lookahead Policy Cost l

1

Rollout Choice

Q-factor approximation

1

x

∗

Optimal Cost Terminal States Cost Approximation Cost

Optimal Cost Terminal States Cost Approximation Cost

1

(

g

1

1

-

µ

1 of the graph of

˜

J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J We can similarly describe the policy iteration (PI) algorithm. It is simply the repeated application of rollout, and generates a sequence of stable linear policies { µ k } . By replicating our earlier calculations, we see that these policies have the form

+1 Initial State A C

k

CAB

C

T

AC

˜

µ

J

DA

C

˜

K

18

Repair

1)

2(

p

n

TJ

J T

mid Play

≤

0

(Sudden death)

w

p

id Play

K

-

CAD CDA CA CD C BD C DB C AB 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . . -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Bold Play J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ -r b 2 Generic stable policy µ T µ J Generic unstable policy µ ′ T µ ′ J 1 of the graph of T J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J ∗ (2) TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J Multistep Lookahead Policy Cost J is a function of x F ( K ) = ( a F ˜ L ( K ) TJ Instability Region Stability Region µ Line 1 ˜ µ J K µ K = K 1 -K 2 Approximately Reoptimized Rollout Policy Exactly Reoptimized Policy Stable Policies Unstable Policy K s K ∗ K µ K -1 2 -µ -1 J µ = -1 µ T µ J = --= min µ ∈ (0 , 1] T µ J ˜ L = -ab ˜ K r + b 2 ˜ K Region of Instability Region of Stability T µ J = -µ +(1 -µ 2 ) J ˆ K y ∗ U ∗ = { 1 , 2 } H ( y ) = min { H 1 ( y ) , H 2 ( y ) , H 3 ( y ) } U ( k ) = { 1 , 3 } U ( k ) = { 2 } /lscript = 2 , m = 4 TG Fig. 13. Illustration of the region of stability, i.e., the set of K ≥ 0 such that the one-step lookahead policy is stable. This is also the set of initial conditions for which Newton's method converges to K ∗ asymptotically. linearize at K ℓ -1 = F ℓ -1 ( ˜ K ) , i.e., at the result of ℓ -1 successive applications of F starting with ˜ K . Each application of F corresponds to a value iteration. Thus the effective starting point for the Newton step is F ℓ -1 ( ˜ K ). 2.4 Region of Stability of Approximation in Value Space It is useful to define the region of stability of approximation in value space as the set of K ≥ 0 such that | a + bL K | &lt; 1 , ∈ U ( x ) ∑ y =1 p xy ( u ) ( g ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C ∗ x ∗ u ∗ x ∗ ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost Approximation in Policy Space Heuristic Cost Approximation for 1 Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Cho sen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n -2 +1 1 -r αb 2 Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 u 2 k x 2 k d τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 min u ∈ U ( x ) ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ x ∗ u ∗ x ∗ u ∗ x ∗ ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 -2 1 Optimal cost Cost of rollout policy ˜ Cost E { g ( x, u, y ) } Cost E { g ( i, u, j ) } Value Network Current Policy Network Approximate Policy Approximate Policy Evaluation Approximately Improved Policy Evaluation Approximate Policy Evaluation Approximate Policy Improvement 0 1 2 3 4 5 6 Deterministic Stochastic Rollout Continuous MPC Constrained Discrete MCTS Variance Reduction Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3.3.3 2.4.3 2.4.2 3.3 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Cost E { g ( x, u, y ) } Cost E { g ( i, u, j ) } Value Network Current Policy Network Approximate Policy Approximate Policy Evaluation Approximately Improved Policy Evaluatio Approximate Policy Evaluation Approximate Policy Improvement 0 1 2 3 4 5 6 Deterministic Stochastic Rollout Continuous MPC Constrained Discrete MCTS Variance Reduction Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3.3.3 2.4.3 2.4.2 3. 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Complete Folding Correspon Expert Rollout with Base Policy m -Approximation of E {·} : min u ∈ U ( x ) p xy ( u u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u High Cost Transition Chosen Capacity=1 Optimal Solutio Permanent Trajectory Tentat sen by Base Heuristic at Base Policy Rollout Policy n -2 Value Network Current Policy Network Approximate P Approximate Policy Evaluation Approximately Improve Approximate Policy Evaluation Approximate Policy Im 0 1 2 3 4 5 6 Deterministic Stochastic Rollout Continuous MPC Co MCTS Variance Reduction Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Monte Carlo Tree Search ' -b Generic stable policy µ T µ J Generic unstable policy µ ′ T µ ′ J Cost of Truncated Rollout Policy ˜ µ 1 of the graph of T J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J ∗ (2) TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ TJ = min µ T µ J Multistep Lookahead Policy Cost J is a function of x F L ( K ) = ( a + bL ) 2 K + q + rL 2 F ˜ L ( K ) T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ Instability Region Stability Region 0 T µ m ˜ J a 2 r b 2 a 2 r b 2 + q q F ( K ) = a 2 rK r + b 2 K + q K ∗ = 0 ˜ K = 0 ¯ K ¯ K = 0 ab ˜ K abK Region of stability Also Region of Convergence of E { B 2 } 45 Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD AD C DA C CD C BD C DB C AB Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play 1st Game / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 , u k , w k ) u k = µ k ( x k ) µ k w k x k Fig. 14. Illustration of the rollout algorithm. Starting from a linear stable base policy µ , it generates a stable rollout policy ˜ µ . The quadratic cost coefficient of ˜ µ is obtained from the quadratic cost coefficient of µ with a Newton step for solving the Riccati equation. 3.3, contains a broader discussion of the region of stability and the role of multistep lookahead in enlarging it. 2.5 Rollout and Policy Iteration Let us return to the linear quadratic problem and the rollout algorithm starting from a stable linear base policy µ . It obtains the rollout policy ˜ µ by using a policy improvement operation, which by definition, yields the one-step lookahead policy that corresponds to terminal cost approximation J µ . Figure 14 illustrates the rollout algorithm. It can be seen from the figure that the rollout policy is in fact an improved policy, in the sense that J ˜ µ ( x ) ≤ J µ ( x ) for all x , something that is true in general (not just for linear-quadratic problems). Among others, this implies that the rollout policy is stable.

k

)

1

.

1

f

k

(

x

k

k

) from

9

=

6

F

(

K

olicy ˜

µ

ature

x

olicy k + ξ k y k +1 C k w k blems 1 1 1 d Riccati Equation TJ = min µ T µ J One-Step Lookahead Policy Cost l TJ = min µ T µ J Multistep Lookahead Policy Cost l Multistep Lookahead Policy Cost /lscript -Step Lookahead In this connection, it is interesting to note that with increased lookahead, the effective starting point F ℓ -1 ( ˜ K ) is pushed more and more within the region of stability, and approaches K ∗ as ℓ increases. In particular, it can be seen that for any given ˜ K ≥ 0 , the corresponding ℓ -step lookahead policy will be stable for all ℓ larger than some threshold ; see Fig. 12. The book [Ber22a], Section

F

L

(

K

) = (

a

+

bL

)

2

K

1

+

1st Game / Bold Play p d 1 -p d p w 1 -p w 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 , u k , w k ) u k = µ k ( x k ) µ k w k x k 1 2 0 u 0 u 1 x 1 Oven 1 Oven 2 Final Temperature L T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ J T ˜ µ J TJ Instability Region Stability Region 0 T µ m ˜ J a 2 r b 2 a 2 r b 2 + q q F ( K ) = a 2 rK r + b 2 K + q K ∗ = 0 ˜ K = 0 ¯ K ¯ K = 0 K ˜ L ˜ L = -ab ˜ K r + ab 2 ˜ K K 1 ˜ L = -abK r + 2 K 1 F ( K ) = a 2 rK r + b 2 K J ∗ (1) = 0 J (1) ( TJ )(1) = min { J (1) , 1 } 1 1 1 1 1 k ) from 0 y H 1 ( y ) H 2 ( y ) H 3 ( y ) U ( k ) = { 1 } State 1 State 2 K ∗ K ∗ = 0 ¯ K ˆ K 2-State/2-Control Example y ∗ Effective Cost Approximation Value Space Approximation State 1 State 2 ( TJ )(1) Base Policy ˜ J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ -r b 2 Generic stable policy µ T µ J Generic unstable policy ′ T µ ′ J Cost of Truncated Rollout Policy ˜ T J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J ∗ (2) where L K is the linear coefficient of the one-step lookahead policy corresponding to K . It can be seen that the region of stability is also closely related to the region of convergence of Newton's method : the set of points K starting from which Newton's method, applied to the Riccati equation K = F ( K ), converges to K ∗ asymptotically. Note that for our one-dimensional linear-quadratic problem, the region of stability is the interval ( K S , ∞ ) that is characterized by the single point K S where F has derivative equal to 1; see Fig. 13. For multidimensional problems, the region of stability may not be characterized as easily. Still, however, it is generally true that the region of stability is enlarged as the length of the lookahead increases . Moreover, substantial subsets of the region of stability may be conveniently obtained. Results of this type are known within the MPC framework under various conditions (see the papers by Mayne at al. [MRR00], Magni et al. [MDM01], and the MPC book [RMD17]). One-Step or Multistep Lookahead for stages Possible Terminal Cost 1 n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost 1 Monte Carlo Tree Search ' min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x i ) , w i ) + ˜ J k 'Future' Monte Carlo Tree Search ' min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x i ) , w i ) + ˜ J 'Future' One-Step or Multistep Looka Approximation in Policy Spa for Stages Beyond Truncation Approximate Q-Factor ˜ Q ( x, min u ∈ U ( x ) E w { g ( x, u, Truncated Rollout Policy µ min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( Optimal Cost Terminal States Cost Approximation Co 'Future' ˜ L = -r + ab 2 ˜ K K 1 ˜ L = -1 r + b 2 K 1 F ( K ) = a 2 rK r + b 2 K J ∗ (1) = 0 J (1) ( TJ )(1) = min { J (1) , 1 } 1 1 3 5 2 4 6 2 10 5 7 8 3 9 6 1 2 Initial Temperature x 0 u 0 u 1 x 1 Oven 1 Oven 2 Final Temperature x 2 ξ k y k +1 = A k y k + ξ k y k +1 C k w k Stochastic Problems Since the rollout policy is a one-step lookahead policy, it can also be described using the formulas that we developed earlier in this section. In particular, let the base policy have the form µ 0 ( x ) = L 0 x, where L 0 is a scalar. We require that µ 0 is stable, i.e., | a + bL 0 | &lt; 1. From our earlier calculations, we have that the cost function of µ 0 is J µ 0 ( x ) = K 0 x 2 , where K 0 = q + rL 2 0 1 -( a + bL 0 ) 2 . Moreover, the rollout policy µ 1 has the form µ 1 ( x ) = L 1 x, where L 1 = -abK 0 r + b 2 K 0 .

<!-- formula-not-decoded -->

q

+

rL

2

F

˜

L

(

K

)

J J

Q-factor approximation

=

T

TJ

˜

J

µ

˜

µ

J

˜

T

˜

µ

T

≤

J T

x

0

Initial

K

˜

L

n n

-

i, u, j

1

) Policy i, u, j

) Policy

˜

µ

˜

1)

.

1

k

)

+1

n

˜

µ

=

˜

µ

CB

BC

C

min

J

(

(

x, u, y

) +

α

˜

y

J

T

J

-

˜

≤

d

C

˜

+

µ

g

(

where L k is generated by the iteration

<!-- formula-not-decoded -->

with K k given by

<!-- formula-not-decoded -->

The corresponding cost functions have the form

<!-- formula-not-decoded -->

A favorable characteristic that enhances the performance of rollout and PI is that the graph of F ( K ) is relatively 'flat' for K &gt; K ∗ . This is due to the concavity of the Riccati operator. As a result, the cost improvement due to the Newton step is even more pronounced, and is relatively insensitive to the choice of base policy. This feature generalizes to multidimensional problems with or without constraints; see the computational study [LKL23]. Cost E { g ( T ˜ µ J ˜ µ T ˜ µ J TJ TJ = min T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ Instability Region Stability Region 0 T µ m ˜ J ar b 2 + q q F ( K ) = arK r + b 2 K + q ˜ K using an Corresponds to One-Step Lookahead Policy ˜ µ State 1 State 2 K ∗ State 2 ( TJ )(1) ˜ J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ 0 State 2 ( e.g., On-Line Player Performance /lscript Steps µ ( x ) = Lx

Cost Function Approximation Position Evaluator

Region of Stability e.g.,

/lscript

-

1 Lookahead Minimization Steps

Robust Base Policy Figure

m

=

/epsilon1

=

K

1

K

2

˜

µ

˜

µ

J J

J

-

˜

Approximately Reoptimized Rollout Policy

Exactly Reoptimized Policy Stable Policies Unsta

˜

µ

Value Space Approximation

˜

J J

One-Step Lookahead Policy Cost yy

)

p

u

K

K

s

= min

(

u

µ

)

p

∈

(0

,

Newton iterate starting from

T

1]

K

T

µ

K

˜

-

µ

1

2

-

Region where Sequential Impro

K

yx

αp

u

)

(

T

αp

=

y

Instability Region Stability R

µ

J

n

, H

2

p

xy

-

x

∑

(

u

)

(

y

)

, H

3

(

y

)

y

J

)

K

µ

+(1

ˆ

2

=1

)

U

Cost of base policy

}

(

/lscript

(

-

Multiagent Q-factor minimizat using an Corresponds to One-Ste

m

k

) =

2

Effective Cost Approximation Value Space Approximati

{

}

Steps Slo

Termination State Constraint

On-Line Player Perform

Cost Function Approximation Position Evaluator

J J

2-State/2-Contr

µ

˜

J

µ

K

˜

=

k

Effective Cost Approximation Value Space App

Line Stable Policies Unstable Poli

∗

k

k

+1

Cost Function Approxim

Effective Cost Approximation Value Space Approximation

T

µ

˜

µ

µ

= min

J

J TJ

T

µ

Cost of ˜

Cost of Truncated Rollout Policy ˜

TJ

µ

µ

J

= min

T

TJ

µ

µ

J

T

= min

Multistep Lookahead Policy Cost

F

<!-- image -->

L

Region of Stability

K

-

µ

r

Current Partial Folding Movin

Robust Base Policy Figure

˜

m

Step

Complete Folding Correspondi

L

b

1

/epsilon1

=

Value Network Current Policy Network Approximate Poli

One-Step Lookahead Policy Cost l

T

Region of Stability

2

= min

2

K

K

2

Part of the classical linear-quadratic theory is that J µ k converges to the optimal cost function J ∗ , while the generated sequence of linear policies { µ k } , where µ k ( x ) = L k x , converges to the optimal policy, assuming that the initial policy is linear and stable. The convergence rate of the sequence { K k } is quadratic, as is typical of Newton's method. This result was proved by Kleinman [Kle68] for the continuous-time multidimensional version of the linear quadratic problem, and it was extended later to more general problems. In particular, the corresponding discrete-time result was given by Hewer [Hew71], and followup analysis, which relates to policy iteration with approximations, was given by Feitzinger, Hylla, and Sachs [FHS09], and Hylla [Hyl11]. Kleinman gives credit to Bellman and Kalaba [BeK65] for the one-dimensional version of his results. Applications of approximate PI in the context of MPC have been discussed in Rosolia and Borrelli [RoB18], and Li et al. [LJM21], among others. Approximate Policy Evaluation Approximately Improved Approximate Policy Evaluation Approximate Policy Impr 0 1 2 3 4 5 6 Deterministic Stochastic Rollout Continuous MPC Cons MCTS Variance Reduction TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ Multistep Lookahead Policy Cost T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ J TJ Instability Region Stability Region 0 using an Corresponds to One-Step Lookahead Policy ˜ µ TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J Multistep Lookahead Policy Cost T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ J T ˜ µ J TJ Instability Region Stability Region 0 using an Corresponds to One-Step Lookahead Policy ˜ µ Expert Rollout with Base Policy m -St Approximation of E {·} : Ap min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 Line Stability Region F ( K ) = arK r + b 2 K + q T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Defined by Newton step from ˜ J for solving J = TJ Generic stable policy µ T µ J Generic unstable policy µ ′ T µ ′ J Cost of Truncated Rollout Policy ˜ µ 1 of the graph of T J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J ∗ (2) TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T Multistep Lookahead Policy Cost J is a function of x F L ( K ) = ( a + bL ) 2 K + q + rL 2 F ˜ L ( K ) Also Region of Convergence of Ne Generic stable policy µ T µ J Generic unstable poli Cost of Truncated Rollout Policy ˜ µ 1 of the grap J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J ∗ ( TJ = min µ T µ J One-Step Lookahead Policy Cost TJ = min µ T µ J Multistep Lookahead Policy Cost Multistep Lookahead Policy Cost /lscript -Step Lookahe Approximately Reoptimized Rollout Policy Exactly Reoptimized Policy Stable Policies Unsta F L ( K ) Interval I Interval II Interval III Interval IV K s K J 0 J µ = -1 µ T µ J = -µ +(1 -µ 2 ) J TJ = min µ ∈ (0 , 1] T ˜ L = -ab ˜ K r + b 2 ˜ K Region of Instability Region of Stability T µ J = -y U = 1 , 2 H ( y ) = min H ( y ) , H ( y ) , H ( y ) L e.g., /lscript -1 Lookahead Minimization Steps m Steps of Rollout Robust Base Policy Figure /epsilon1 = K 1 -K 2 Approximately Reoptimized Rollout Policy Exactly Reoptimized Policy Stable Policies Unstable Policy F L ( K ) Interval I Interval II Interval III Interval IV K s K ∗ K µ K -1 2 -µ -1 J 0 J µ = -1 µ T µ J = -µ +(1 -µ 2 ) J TJ = min µ ∈ (0 , 1] T µ J ˜ L = -ab ˜ K r + b 2 ˜ K e.g., /lscript -1 Lookahead Mi Robust Base Policy Figu Approximately Reoptimi Exactly Reoptimized Po F L ( K ) Interval I Interval II Inte J 0 J µ = -1 µ T µ J = -µ +(1 Fig. 15. Illustration of the m -step truncated rollout algorithm with one-step lookahead. Starting with a linear stable base policy µ ( x ) = Lx , it generates a rollout policy ˜ µ . The quadratic cost coefficient of ˜ µ is obtained with a Newton step, after approximating of the quadratic cost coefficient K L of µ with m = 4 value iterations that start from ˜ K . Compare with the nontruncated rollout Fig. 14. robot and vehicle routing problems with imperfect state information, among others; see [BKB20], [GPG22], and [WGP23].

It is important to note that rollout, like policy iteration, can be applied universally, well beyond the linearquadratic/MPC context that we have discussed here. In fact, the main idea of rollout algorithms, obtaining an improved policy starting from some other suboptimal policy, has appeared in several DP contexts, including games; see e.g., Abramson [Abr90], and Tesauro and Galperin [TeG96]. The adaptation of rollout to discrete deterministic optimization problems and the principal results relating to cost improvement were given in the paper by Bertsekas, Tsitsiklis, and Wu [BTW97], and were also discussed in the neuro-dynamic programming book [BeT96]. Rollout algorithms for stochastic problems were further formalized in the papers by Bertsekas [Ber97], and Bertsekas and Casta˜ non [BeC99]. Extensions to constrained rollout were given in the author's papers [Ber05a], [Ber05b]. Rollout algorithms were also proposed in nontruncated form within the MPC framework; see De Nicolao, Magni, and Scattolini [DMS98], [MaS04], and followup works.

A noteworthy extension, highly relevant to MPC as well as other contexts, is multiagent rollout , which deals successfully with the acute computational difficulties arising from the large (Cartesian product) control spaces that are typical of multiagent problems. The author's book [Ber20] and paper [Ber21a] discuss this research, and give references to supportive computational studies in multi-

µ

µ

T

J

µ

˜

J J

Base Policy

J

˜

µ

=

T

˜

µ

J

˜

µ

˜

J TJ

-

J J

˜

µ

˜

µ

T

µ

K

Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3. 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Monte Carlo Tree Search ' Line Stability Region Line Stability Region x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 High Cost Transition Chosen b Capacity=1 Optimal Solution Permanent Trajectory Tentativ sen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Ap n -2 One-Step or Multistep Lookah 1 T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ J T TJ Instability Region Stability Region 0 T µ m ˜ J a 2 r b 2 a 2 r b 2 + q q F ( K ) = a 2 rK r + b 2 K + q K ∗ = 0 ˜ K = 0 ¯ K ¯ K = 0 K ˜ L ˜ L = -ab ˜ K r + ab 2 ˜ K K 1 ˜ L = -abK 1 r + b 2 K 1 F L ( K ) = ( a + bL ) K + q + rL F ˜ L ( 1 ∗ ∗ { } { 1 2 3 0 y H 1 ( y ) H 2 ( y ) H 3 ( y ) U ( k ) = { 1 } U ( k ) = { 2 } /lscript State 1 State 2 K ∗ K ∗ = 0 ¯ K ˆ K 2-State/2-Contr Effective Cost Approximation Value Space App State 2 ( TJ )(1) Base Policy ˜ J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ Generic stable policy µ T µ J Generic unstable pol Region of Instability Region of Stability T µ J = -µ +(1 -µ 2 ) J ˆ K y ∗ U ∗ = { 1 , 2 } H ( y ) = min { H 1 ( y ) , H 2 ( y ) , H 3 ( y ) } U ( k ) = { 1 , 3 } 0 y H 1 ( y ) H 2 ( y ) H 3 ( y ) U ( k ) = { 1 } U ( k ) = { 2 } /lscript = 2 , m = 4 TG State 1 State 2 K ∗ K ∗ = 0 ¯ K ˆ K 2-State/2-Control Example y ∗ Effective Cost Approximation Value Space Approximation State 1 State 2 ( TJ )(1) Region of Instability Re y ∗ U ∗ = { 1 , 2 } H ( y ) = 0 y H 1 ( y ) H 2 ( y ) H 3 ( y ) State 1 State 2 K ∗ K ∗ Effective Cost Approxi State 2 ( TJ )(1) Finally, we note that the author's books [Ber20], [Ber22a], [Ber23] provide extensive references to the journal literature, which includes a large number of computational studies. These studies discuss variants and problem-specific adaptations of rollout algorithms and consistently report favorable computational experience. The size of the cost improvement over the base policy is often impressive, evidently owing to the fast convergence rate of Newton's method that underlies rollout.

## 2.6 Truncated Rollout

rK

a

2

k

+

/lscript

1

Approximation in Policy Space for Stages Beyond Truncation

2

b

{

K

g

k

(

x

k

, u

k

, w

k

) +

µ

1 of the grap

Base Policy

˜

-

˜

µ

J

˜

µ

i

(

x

i

J J

=

k k +1 k + /lscript -1 i = k +1 Optimal Cost Terminal States Cost Approximation Cost 'Future' 1 Approximate Q-Factor ˜ Q ( x, u ) min u ∈ U ( x ) E w { g ( x, u, w Truncated Rollout Policy µ m J ∗ (1) = 0 J (1) ( TJ )(1) = min { J (1) , 1 } 1 1 J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J ∗ ( TJ = min µ T µ J One-Step Lookahead Policy Cost TJ = min µ T µ J Multistep Lookahead Policy Cost Base Policy J J ˜ µ J ˜ µ = T ˜ µ -Generic stable policy µ T µ J Generic unstable policy µ ′ T µ ′ J Cost of Truncated Rollout Policy ˜ µ 1 of the graph of T J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J ∗ (2) Generic stable policy µ Cost of Truncated Rollo J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) An m -step truncated rollout scheme with a stable linear base policy µ ( x ) = Lx , one-step lookahead minimization, and terminal cost approximation ˜ J ( x ) = ˜ Kx 2 is obtained by starting at ˜ K , executing m VI steps using µ , followed by a one-step lookahead minimization/Newton step. It is visually interpreted as in Fig. 15, where m = 4.

∑

g

Cost of Truncated Rollout Policy ˜

-Step Lookahe

J

One-Ste rL

˜

(

F

2

J

µ

r

b

u

) =

K

µ

,...,µ

F

,µ

µ

µ

˜

µ

˜

J

J TJ

= min

J

T

T

Cost of ˜

+

r

E

min

˜

1 1 Multistep Lookahead Policy Cost /lscript F L ( K ) = ( a + bL ) 2 K + q + 1 TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J 1 TJ = min µ T µ TJ = min µ T µ Thus the difference with (nontruncated) rollout is that we use m VI steps starting from ˜ K to approximate the cost function K L x 2 of the base policy. Truncated rollout makes little sense in linear-quadratic problems where K L can be easily computed by solving the Riccati equation. However, it is useful in more general problem settings, as it may save significantly in computation, relative to obtaining exactly J µ (which requires an infinite number of VI steps).

L

Some interesting points regarding truncated rollout schemes are the following:

- (a) Lookahead by truncated rollout may be an economic substitute for lookahead by minimization, in the sense that it may achieve a similar performance at significantly reduced computational cost; see e.g., [LiB24].

Multiste

2

xt

T

˜

µ

J

˜

µ

(

Step

## 2.7 Double Rollout

)

Cost-to-go approximation Expected value approximation

Optimal cost

µ

µ

1

x

) =

/J

K

J

)

J

x

/K

(

(

0

1

Cost-to-go approximation Expected value approximation

Optimal cost

T

∗

T

µ

µ

µ

µ

˜

µ

µ

˜

J J

T

=

J

T

J

=

0

0

L

r

J

˜

µ

Cost of base policy

Effective Cost Approximation Value Space Approximation

T

˜

µ

=

= min

µ

T

T

˜

µ

µ

J

J

Cost of rollout policy ˜

˜

µ

Cost of base policy

µ

Optimal Base Rollout

µ

˜

J J

˜

µ

J

˜

µ

Cost of rollout

Cost of ˜

µ

µ

J

T

˜

µ

J

J

µ

˜

µ

=

˜

µ

T

J

∗

J TJ

µ

J

µ

J

- (b) Lookahead by m -step truncated rollout with a stable policy has an increasingly beneficial effect on the stability properties of the lookahead policy, as m increases. policy ˜ µ Simplified minimization min n p xy ( u ) g ( x, u, y ) + Cost of Truncated Rollout Policy ˜ µ TJ = min µ T µ J

∑

Simplified minimization

Optimal cost

x

(

)

Policy Improvement with Base Policy

x

u

w

L

These statements are difficult to establish analytically in some generality. However, they can be intuitively understood in the context with our one-dimensional linear quadratic problem, using geometrical constructions like the one of Fig. 15. They are also consistent with the results of computational experimentation. We refer to the monograph [Ber22a] for further discussion. ∈ y =1 Multiagent Q-factor minimization x k Termination State Constraint Set X X = X r b 2 +1 1 -r αb 2 ˜ K K K ∗ K k K k +1 F ( K ) = r + Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open T µ J J µ T µ J J µ = T µ J µ J ˜ µ = T ˜ µ J ˜ µ Cost of rollout policy ˜ µ Simplified minimization min n p xy u ∈ U r b 2 +1 1 -r αb 2 ˜ TJ = min µ T µ J Multistep Lookahead Policy Cost F L ( K ) = ( a + bL ) 2 K T 2 ˜ J T ˜ J ˜ J J 0 J

k

x

k

+1

=

u

U

k

k

Instability Region Stability Region 0

P

P

p

yt yt

(

u

∗

K

µ

J

min

) 1

x

(

)

-

(

)

u

)

µ

K

n

α

y

-

α

x y x y

-

p

xy

=1

Input (Control) Output (Function of the State) Changin

∑

Multiagent Q-factor minimiz

µ

+(1

µ

2

+1

m

+

+1

k

x

Termination State Constraint

-

1

)

µ

(

x

1

)

Expert

k

TJ

P

F

(

P

˜

P

Multiagent Q-factor minimization

Rollout with Base Policy

We noted that rollout with a base policy µ amounts to a single policy iteration starting with µ , to produce the (improved) rollout policy ˜ µ . The process can now be continued to apply a second policy iteration. This results in a double rollout policy, i.e., a second rollout policy that uses the first rollout policy ˜ µ as a base policy. For deterministic problems, the needed rollout policy costs can be computed recursively on-line, with computation that may be tractable, thanks to rollout truncation or special simplifications that take advantage of the deterministic character of the problem. Parallel computation, for which rollout is well suited, can also be very helpful in this respect. Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice r b 2 +1 1 -r αb 2 ˜ K K K ∗ K k K k +1 F ( K ) = αrK r + αb 2 K +1 Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation Termination State Constraint Set X X = X ˜ X Multiagent r b 2 +1 1 -r αb 2 ˜ K K K ∗ K k K k +1 F ( K ) = αrK r + αb 2 K +1 Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Optimal cost Cost of rollout policy ˜ µ Cost of base policy µ Cost E { g ( x, u, y ) } Cost E { g ( i, u, j ) } Value Network Current Policy Network Approximate Policy Approximate Policy Evaluation Approximately Improved Policy Evaluation Optimal cost Cost of rollout policy ˜ µ Cost of base policy µ Cost E { g ( x, u, y ) } Cost E { g ( i, u, j ) } Value Network Current Policy Network Approximate Policy Approximate Policy Evaluation Approximately Improved Policy Evaluatio r b 2 +1 1 -r b 2 K K ∗ K k k k +1 Current Partial Folding Movi Complete Folding Correspon Expert Rollout with Base Policy m -Approximation of E {·} : min u ∈ U ( x ) n ∑ y =1 p xy ( u x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation ˜ F ( P ) k Q 0 P -R E { B 2 } 45 ◦ Stock at Period k +1 Initial State A C AB AC CA CD ABC ACB ACD CAB CAD CDA S A S B C AB C AC C CA C CD C BC C CB C CD C AB C AD C DA C CD C BD C DB C AB Do not Repair Repair 1 2 n -1 n p 11 p 12 p 1 n p 1( n -1) p 2( n -1) . . . ar b 2 + q q F ( K ) = arK r + b 2 K + q ˜ K using an Corresponds to One-Step Lookahead Policy ˜ µ Line Stability Region F ( K ) = arK r + b 2 K + q T ˜ µ ( T µ m ˜ J ) = T ( T µ m ˜ J ) Yields Truncated Rollout Policy ˜ µ Defined by Newton step from ˜ J for solving J = TJ State 1 State 2 K ∗ ¯ K 2-State/2-Control Example Effective Cost Approximation Value Space Approximation State 2 ( TJ )(1) ˜ J J ˜ µ J ˜ µ = T ˜ µ J ˜ µ T ˜ µ J TJ = min µ T µ J Cost of ˜ µ -r b 2 Generic stable policy µ T µ J Generic unstable policy µ ′ T µ ′ J Cost of Truncated Rollout Policy ˜ µ 1 of the graph of T J ∗ J ∗ (1) J ∗ (2) ( TJ ∗ )(1) = J ∗ (1) ( TJ ∗ )(2) = J ∗ (2) TJ = min µ T µ J One-Step Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ J ∗ 2 ( x 2 ) µ ∗ 2 ( x 2 ) J ∗ N -1 ( x N -1 ) µ ∗ N -1 ( x N -1 ) J ∗ N ( x N ) µ ∗ N ( x N u k : guess word selected at time k X 0 X 1 X 2 X N -1 X N State x k Parameter θ b k u k = µ ( x k , b k ) Belief Estimator for solving the Bellman Eq. Kx 2 = F ( K ) x 2 or K = F ( K J k +1 ( x ) = K k +1 x 2 = F ( K k ) x 2 m -Component Control u = ( u 1 , . . . , u m ) u 1 u m Fig. 16. Illustration of a double Newton step. Starting from a quadratic cost coefficient ˜ K that defines the policy µ , it uses µ as a base policy to implement a rollout policy ˜ µ . followed by a Newton step (cf. Fig. 12). For this statement to be correct, ˜ K should lie within the region of stability. Such ˜ K may obtained by using multiple value iterations, as in the case where a multistep lookahead minimization is performed, i.e. ℓ &gt; 1 (cf. Fig. 12).

+1

∗

Value iterations

<!-- image -->

k

k

Capacity=1 Optimal Solution 2.4.2, 2.4.3

k

∗

1

Multiagent

Approximate Policy Evaluation Approximate Policy Improvement

Approximate Policy Evaluation Approximate Policy Improvement

k

0

1

x

2.4.5

k

sen by Base Heuristic at

Initial sen by Base Heuristic at

x

0

Instability Region Stability Region 0

E

, w

w

)

Initial

, w

i

k

J

J

+ ˜

+ ˜

µ

k

k

k

k

k

k

Base Policy Rollout Policy Approximation in Value Space n n -n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost 1 sen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 One-Step or Multistep Lookahead for stages Possible Terminal Cost 1 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Monte Carlo Tree Search ' min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x i ) ) 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Monte Carlo Tree Search ' min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x i ) Base Policy Rollout Policy n -2 One-Step or Multistep Looka Approximation in Policy Spa for Stages Beyond Truncation Approximate Q-Factor ˜ Q ( x, min u ∈ U ( x ) { g ( x, u, Truncated Rollout Policy 6 1 2 x 0 u 0 u 1 x 1 Oven 1 Oven 2 Final Temperature y k +1 C k w k 1 a 2 r b 2 a 2 r b 2 + q q F ( K ) = a 2 rK r + b 2 K + q K ∗ = 0 ˜ K = 0 ¯ K ¯ K = 0 K ˜ L ˜ L = -ab ˜ K r + ab 2 ˜ K K 1 ˜ L = -abK 1 r + b 2 K 1 F ( K ) = a 2 rK r + b 2 K J ∗ (1) = 0 J (1) ( TJ )(1) = min { J (1) , 1 } 1 Each if the m steps of the engine may involves a multi-m to evaluate the corresponding starting from a leaf position of the lookahead tree Base policy is the evaluation of a chess engine Terminal Cost Approximation ˜ J Rollout Policy Network Value Network µ F ( K ) = min L ∈/Rfractur F L ( K ) The Newton step interpretation of approximation in value space leads to an important insight into the special character of the initial step in ℓ -step lookahead implementations. In particular, it is only the first step that acts as the Newton step , and needs to be implemented with precision; cf. Fig. 9. The subsequent ℓ -1 steps consist of a sequence of value iterations with starting point α ℓ ˜ J , and only serve to enhance the quality of the starting point of the Newton step. As a result, their precise implementation is not critical ; this is a major point in the narrative of the author's book [Ber22a].

Triple and higher order rollout, which amount to multiple successive policy iterations, are possible. However, the online computational costs quickly become overwhelming, despite the potential use of truncation and other simplifications, or parallel computation. For further discussion of double rollout, see Section 2.3.5 of the book [Ber20] and Section 6.5 of the book [Ber22], and for computational experimentation results, see the recent paper by Li and Bertsekas [LiB24], which deals with special inference contexts in hidden Markov models. Policy iteration/double rollout has also been discussed by Yan et al. [YDR04] in the context of the game of solitaire, and by Silver and Barreto [SiB22] in the context of a broader class of search methods. Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost Approximation in Policy Space Heuristic Cost Approximation for u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Cho Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chon -2 Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 n -2 1 0 1 2 3 4 5 6 Deterministic Stochastic Rollout Continuous MPC Constrained Discrete MCTS Variance Reduction Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3.3.3 2.4.3 2.4.2 3.3 0 1 2 3 4 5 6 Deterministic Stochastic Rollout Continuous MPC Constrained Discrete MCTS Variance Reduction Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3.3.3 2.4.3 2.4.2 3. u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u High Cost Transition Chosen Capacity=1 Optimal Solutio Permanent Trajectory Tentat p 22 p 2 n p 2( n -1) p 2( n -1) p ( n -1)( n -1) p ( n -1) n p nn 2nd Game / Timid Play 2nd Game / Bold Play 1st Game / Timid Play 1st Game / Bold Play p d 1 -p d p w 1 -p w 0 -0 1 -0 0 -1 1 . 5 -0 . 5 1 -1 0 . 5 -1 . 5 0 -2 System x k +1 = f k ( x , u , w ) u = µ ( x ) µ w x 3 5 2 4 6 2 10 5 7 8 3 9 TJ = min µ T µ J Multistep Lookahead Policy Cost l ˜ J J ˜ µ = T ˜ µ Multistep Lookahead Policy Cost J is a function of x F L ( K ) = ( a + bL ) 2 K + q + rL 2 F ˜ L ( K ) T 2 ˜ J T ˜ J ˜ J Region where Sequential Improvement Holds TJ ≤ Bellman Operator Value Iterations Largest Invariant Set Bellman Equation on Space of Quadratic Functions J ( x ) Tube Constraint Cannot be Satisfied for all x 0 ∈ X if a 20 40 18 2 6 22 Unstable System x k +1 = 2 x k + u k x β -β m -step truncated rollout greedy policy with respect to t Note that it is also possible to consider variants of rollout on top of approximation in value space, such as truncated and simplified versions. An important example of the truncated version is the TD-Gammon architecture, where the terminal cost function approximation is constructed off-line by using a neural network. 2.9 The Importance of the First Step of Lookahead

## 2.8 Double Newton Step - Rollout on Top of Approximation in Value Space Initial Temperature x 2

ξ

Given a quadratic cost coefficient ˜ K that defines the policy µ ( x ) = L ˜ K x , it is natural and convenient to consider rollout that uses µ as a base policy. This can be viewed as rollout that is built on top of approximation in value space. We call this algorithm double Newton step , because it consists of two Newton steps: a first step that maps ˜ Kx 2 to J µ ( x ) and a second step that maps J µ ( x ) to the cost function J ˜ µ ( x ) of the rollout policy ˜ µ that is produced when the base policy is µ ; see Fig. 16. k k +1 k k k Stochastic Problems

y

=

A

y

+

ξ

The double Newton step is much more powerful than the algorithm that performs approximation in value space with two-step lookahead starting from ˜ K . In particular, both algorithms involve multiple steps for solving the Riccati equation starting from ˜ K . However, the former algorithm amounts to a Newton step followed by a Newton step, while the later algorithm amounts to a value iteration

Optimal Cost Terminal States Cost Approximation Cost g ( i, u, j ) Policy 'Future' 1 Optimal Cost Terminal States Cost Approximation Cost g ( i, u, j ) Policy 'Future' 1 1 R 0 R 1 R 2 T 2 Cost 28 Cost 27 Cost 13 Lookahead Contro 1 This idea suggests that we can simplify (within reason) the lookahead steps after the first with small (if any) performance loss for the multistep lookahead policy. An important example of such a simplification is the use of certainty equivalence, which will be discussed in the next section. Other possibilities include various ways of 'pruning' the lookahead tree; see [Ber23], Section 2.4. On the other hand, pruning the lookahead tree at the first stage of lookahead, as is often done in Monte Carlo Tree Search, can have a serious detrimental effect on the quality of the MPC policy.

In practical terms, simplifications after the first step of the multistep lookahead can save a lot of on-line computation, which can be fruitfully invested in extending the length

i

TJ

T

µ m

˜

J

1

2

(

u

µ

=

of the lookahead. This insight is supported by substantial computational experimentation, starting with the paper by Bertsekas and Casta˜ non [BeC98], which verified the beneficial effect of certainty equivalence (after the first step) as a rollout simplification device for stochastic problems. On the other hand, implementing imprecisely the minimization of the first step can adversely impact the performance of the multistep lookahead policy. This point is often missed in the design of approximate lookahead minimization schemes, such as Monte Carlo Tree Search.

2.10 Newton Step Interpretation of Approximation in Value Space in General Infinite Horizon Problems

The interpretation of approximation in value space as a Newton step, and related notions of stability that we have discussed in this section admit a broad generalization. The key fact in this respect is that our DP problem formulation allows arbitrary state and control spaces, both discrete and continuous, and can be extended even further to general abstract models with a DP structure; see the abstract DP book [Ber22b].

Within this more general context, the Riccati operator is replaced by an abstract Bellman operator and the quadratic terminal cost function ˜ Kx 2 is replaced by a general cost function ˜ J . Valuable insight can be obtained from graphical interpretations of the Bellman equation, the VI and PI algorithms, one-step and multistep approximation in value space, the region of stability, and exceptional behavior; see the book [Ber22a], and Section 1.6.7 of the book [Ber23] for a discussion of the MPC context. Naturally, the graphical interpretations and visualizations are limited to one dimension. However, the visualizations provide insight, and motivate conjectures and mathematical proof analysis, much of which is given in the books [Ber20] and [Ber22a].

2.11 How Approximation in Value Space Can Fail and What to Do About It

Practice has shown that MPC is a reliable methodology that can be made to work, assuming (as we have in this section) that a system model is available in either analytical form or in simulator form, and that this model is not changing over time. Still, however, even under these favorable circumstances, failure is possible, in the sense that the ℓ -step lookahead MPC policy is performing poorly. Typically the reason for failure is that the terminal cost approximation ˜ J lies outside the region of convergence of the Newton step. This region depends on ℓ (see the discussion near the end of Section 2.4), as well as the truncated rollout scheme, which effectively modifies the starting point of the Newton step (see the discussion of Section 2.6). 9

For an example of broad interest, let us assume that ˜ J is obtained by training with data a neural network (e.g., as in AlphaZero and TD-Gammon). Let us also focus on the

9 In the case of the linear quadratic problem with terminal cost approximation ˜ J ( x ) = ˜ Kx 2 , ℓ -step lookahead minimization, and m -step truncated rollout with stable policy µ ( x ) = Lx , the region of stability is the set of all ˜ K such that F ℓ -1 ( F m L ( ˜ K ) ) belongs to the set of K such that | a + bL K | &lt; 1; see Section 2.4 and Fig. 13.

case of one-step lookahead with no truncated rollout. In this case there are three components that determine the approximation error ˜ J -J ∗ :

- (a) The power of the neural network architecture , which roughly speaking is a measure of the error that would be obtained if infinite data were available and used optimally to obtain ˜ J by training the given neural network.
- (b) The additional error degradation due to limited availability of training data .
- (c) The additional error degradation due to imperfections in the training methodology .

Thus if the architecture is not powerful enough to bring ˜ J -J ∗ within the region of convergence of Newton's method, approximation in value space with one-step lookahead will likely fail, no matter how much data is collected and how effective the associated training method is .

In this case, there are two potential practical remedies:

- (1) Use a more powerful architecture/neural network for representing ˜ J , so it can be brought closer to J ∗ .
- (2) Extend the combined length of the lookahead minimization and truncated rollout in order to bring the effective value of ˜ J within the region of convergence of Newton's method.

The first remedy typically requires a deep neural network or transformer, which uses more weights and requires more expensive training. 10 The second remedy requires longer on-line computation and/or simulation, which may run to difficulties because of some practical implementation limits. Parallel computation and sophisticated multistep lookahead methods may help to mitigate these requirements (see the corresponding discussions in the books [Ber22a] and [Ber23]).

## 3. THE TREATMENT OF STOCHASTIC UNCERTAINTY THROUGH CERTAINTY EQUIVALENCE

The main ideas of our framework extend to the case of a stochastic system of the form 11

<!-- formula-not-decoded -->

where w k is random with given probability distribution that depends only on the current state x k and control u k ,

10 For a recent example of implementation of a grandmaster-level chess program with one-step lookahead and a huge-size (270M parameters) neural network position evaluator, see Ruoss et al. [RDM24].

11 In this section we restrict ourselves to stochastic uncertainty. For a parallel development relating to set-membership uncertainty and a minimax viewpoint, we refer to the books [Ber22a], Section 6.8, [Ber22b], Chapter 5, and [Ber23], Section 2.12. The paper [Ber21b] addresses the challenging issue of convergence of Newton's method, applied to the Bellman equation of sequential zero-sum Markov games and minimax control problems. The zero-sum game structure differs in a fundamental way from its one-player optimization counterpart: its Bellman equation mapping need not be concave, and this complicates the convergence properties of Newton's method. The paper [Ber21b] proposes new PI algorithms for discounted infinite horizon Markov games and minimax control, which are globally convergent, admit distributed asynchronous implementations, and lend themselves to the use of rollout and other RL methods.

and not on earlier states and controls. The cost per stage also depends on w k and is g ( x k , u k , w k ).

The cost function of µ , starting from an initial state x 0 is where E {·} denotes expected value. The optimal cost function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

again satisfies the Bellman equation, which now takes the form

Furthermore, if µ ∗ ( x ) attains the minimum above for all x , then µ ∗ is an optimal policy.

Similar to the deterministic case, approximation in value space with one-step lookahead replaces J ∗ with an approximating function ˜ J , and obtains a suboptimal policy ˜ µ with the minimization

It is also possible to use ℓ -step lookahead, with the aim to improve the performance of the policy obtained through approximation in value space. This, however, can be computationally expensive, because the lookahead graph expands fast as ℓ increases, due to the stochastic character of the problem. Using certainty equivalence (CE for short) is an important approximation approach for dealing with this difficulty, as it reduces the search space of the ℓ -step lookahead minimization. Moreover, CE mitigates the excessive simulation because it reduces the stochastic variance of the lookahead calculations at each stage.

<!-- formula-not-decoded -->

In the pure but somewhat flawed version of the CE approach, when solving the ℓ -step lookahead minimization problem, we simply replace all of the uncertain quantities w k , w k +1 , . . . , w k + ℓ -1 , . . . , w N -1 by some fixed nominal values, thus making that problem fully deterministic. Unfortunately, this affects significantly the character of the approximation: when w k is replaced by a deterministic quantity, the Newton step interpretation of the underlying approximation in value space scheme is lost to a great extent.

Still, we may largely correct this difficulty, while retaining substantial simplification, by using CE after the first stage of the ℓ -step lookahead. We can do this with a CE scheme whereby at state x k , we replace only the uncertain quantities w k +1 , . . . , w N -1 by deterministic values, while we treat the first, i.e., w k , as a stochastic quantity. 12

This type of CE approach, first proposed and tested in the paper by Bertsekas and Casta˜ non [BeC99], has an important property: it maintains the Newton step character of the approximation in value space scheme . In particular, the cost function J ˜ µ of the ℓ -step lookahead

12 Variants of the CE approach, based on less drastic simplifications of the probability distributions of the uncertain quantities, which involve multiple representative scenarios, are given in the author's books [Ber17a], Section 6.2.2, and [Ber19a], Section 2.3.2. Related ideas have also been suggested in MPC contexts; see e.g., Lucia, Finkler, and Engell [LFE13].

policy ˜ µ is generated by a Newton step, applied to the function obtained by the last ℓ -1 minimization steps (modified by CE, and applied to the terminal cost function approximation); see the monograph [Ber20] and Sections 1.6.7, 2.7.2, 2.8.3, of the textbook [Ber23] for a discussion. Thus the benefit of the fast convergence of Newton's method is restored. In fact based on insights derived from this Newton step interpretation, it appears that the performance penalty for the CE approximation is often small. At the same time the ℓ -step lookahead minimization involves only one stochastic step, the first one, and hence potentially a much 'thinner' lookahead graph, than an ℓ -step minimization that does not involve any CE-type approximations.

## 4. MPC AND ADAPTIVE CONTROL

Our discussion so far dealt with problems with a known mathematical model, i.e., one where the system equation, cost function, control constraints, and probability distributions of disturbances are perfectly known. The mathematical model may be available through explicit formulas and assumptions, or through a computer program that can emulate all of the mathematical operations involved in the model, including Monte Carlo simulation for the calculation of expected values. 13 In practice, however, it is common that the system parameters are either not known exactly or can change over time, and this introduces potentially enormous complications. 14

Let us also note that unknown problem environments are an integral part of the artificial intelligence view of RL. In particular, to quote from the popular book by Sutton and Barto [SuB18], 'learning from interaction with the environment is a foundational idea underlying nearly all theories of learning and intelligence' while RL is described as 'a computational approach to learning from interaction with the environment.' The idea of learning from interaction with the environment is often connected with the idea of exploring the environment to identify its characteristics.

In control theory this is often viewed as part of the system identification methodology, which aims to construct mathematical models of dynamic systems. The system identification process is often combined with the control process to deal with unknown or changing problem parameters, in a framework that is sometimes called dual control . 15 This

13 The term 'model-free' is often used to describe the latter situation, but in reality there is a mathematical model that is hidden inside the simulator, so the ideas of present section apply in principle. 14 The difficulties introduced by a changing environment complicate the balance between off-line training and on-line play. It is worth keeping in mind that as much as learning to play high quality chess is a great challenge, the rules of play are stable; they do not change unpredictably in the middle of a game! Problems with changing system parameters can be far more challenging!

15 The dual control framework was introduced in a series of papers by Feldbaum, starting in 1960 with [Fel60]. These papers emphasized the division of effort between system estimation and control, now more commonly referred to as the exploration-exploitation tradeoff . In the last paper of the series [Fel63], Feldbaum prophetically concluded as follows: 'At the present time, the most important problem for the immediate future is the development of approximate solution methods for dual control theory problems, the formulation of sub-optimal strategies, the determination of the numerical value

## 4.1 Robust Control

Given a controller design that has been obtained assuming a nominal DP problem model, one possibility is to simply ignore changes in problem parameters. We may then try to investigate the performance of the current design for a suitable range of problem parameter values, and ensure that it is adequate for the entire range. This is sometimes called a robust controller design . Deterministic MCTS Variance Reduction 2.4.3, 2.4.4 2.4.2 3.3, 3.4 0 1 2 3 4 5 6 Deterministic MCTS Variance Reduction 0 1 2 3 4 5 6 Deterministic MCTS Variance Reduction

'Future'

In PID control, no attempt is made to maintain a mathematical model and to track unknown model parameters as they change. A more ambitious form of suboptimal control is to separate the control process into two phases, a system yt

p

)

xt

u

yy

p

)

(

u

αp xy

xx

(

u

αp

)

xx xy

yx

p

xy

(

yx

(

)

u

Initial State 15 1 5 18 4 19 9 21 25 8 12 13

yx xy

αp

)

αp xt

αp

(

u

yy

(

p

(

)

)

u

αp

c

yx

αp

)

αp

)

(

u

yy yx

)

(

(

u

yy

αp

(0)

(

(

p

y

u

+1)

) 1

)

c

(

u

c

)

k

αp

-

) 1

c

α

k

(

yx yy

x y xx

xy

αp

(

(

u

u

)

u

u

)

(

u

)

N

(

(

u

) 1

) System Data Control Parameter Estimati

) System Data Control Parameter Estimation

-

+1

c

k k

1)

N

(

) System State Data Control Parameter Es

k

Cost of base policy

Cost of base policy

µ

k

k

k

x

(

x

, u

, w

)

Cost of base policy

k

Obs

µ

µ

Value Network Current Policy Network Approximate Policy

Value Network Current Policy Network Approximate Policy

x 0 a 0 1 2 t b C Destination J ( x k ) → 0 for all p -stable π from x 0 with x 0 ∈ X and π ∈ P p,x 0 within W p + Approximate Policy Evaluation Approximately Improved Policy Evalu Approximate Policy Evaluation Approximately Improved Polic Approximate Policy Evaluation Approximate Policy Improvem Value Network Current Policy Network Approximate Policy Approximate Policy Evaluation Approximately Improved Polic g ( i, u, j ) } µ Cost of base policy µ E g ( i, u, j ) µ Cost of base policy µ Fig. 17. Schematic illustration of on-line replanning: the concurrent parameter estimation and system control. The system parameters are estimated on-line and the estimates are periodically passed on to the controller.

<!-- image -->

In this section, we will briefly review some of the most commonly used approaches for dealing with unknown parameters, such as robust control, PID control, and indirect adaptive control. We will also suggest a simplified version of indirect adaptive control that uses rollout (possibly truncated and supplemented with terminal cost approximation) in place of policy reoptimization. Value Network Current Policy Network Approximate Policy xx

xy

)

)

u

x p

u

(

(

p

p

(

u

)

p

yy

(

u

)

p

xt

(

u

xx

)

p

x p

αp xx

u

(

yt

(

u

)

is one of the most challenging areas of stochastic optimal and suboptimal control, and has been studied intensively since the early 1960s, with several textbooks and research monographs written: Astr¨ om and Wittenmark [AsW94], Astr¨ om and Hagglund [AsH06], Bodson [Bod20], Goodwin and Sin [GoS84], Ioannou and Sun [IoS96], Jiang and Jiang [JiJ17], Krstic, Kanellakopoulos, and Kokotovic [KKK95], Kumar and Varaiya [KuV86], Liu, et al. [LWW17], Lavretsky and Wise [LaW13], Narendra and Annaswamy [NaA12], Sastry and Bodson [SaB11], Slotine and Li [SlL91], and Vrabie, Vamvoudakis, and Lewis [VVL13]. These books describe a vast array of methods spanning 60 years, and ranging from adaptive and modelfree approaches, to self-tuning regulators, to simultaneous or sequential control and identification, to time series models, to extremum-seeking methods, to simulation-based RL techniques, etc. Stage 1 Stage 2 Stage 3 Stage N N -Heuristic Cost 'Future' System x k Belief State p k Controller µ k Initial State x 0 s Terminal State t Cost 0 Cost g ( x, u, y Optimal cost Cost of rollout policy ˜ µ Cost E { g ( x, u, y ) } Cost E { g ( i, u, j ) } Cost 0 Cost g ( x, u, y Optimal cost Cost of rollout policy ˜ µ Cost E { g ( x, u, y ) } Cost E { g ( i, u, j ) Cost 0 Cost g ( x, u, y Optimal cost Cost of rollout policy ˜ µ Cost E { g ( x, u, y ) } Cost E { g ( i, u, j ) } αp xx ( u ) αp xy ( u ) αp yx ( u ) αp yy ( u ) 1 -α Cost 0 Cost g ( x, u, y ) System State Data Control Parameter Estimation Optimal cost Cost of rollout policy ˜ µ Cost of base policy µ Cost E { g ( x, u, y ) } Cost E { Value Network Current Policy Network Approximate Policy x p xx ( u ) p xy ( u ) p yx ( u ) p yy ( u ) p xt ( u ) p yt ( u ) x y αp xx ( u ) αp xy ( u ) αp yx ( u ) αp yy ( u ) 1 -α Cost 0 Cost g ( x, u, y ) System State Data Control Parameter Estimation Optimal cost Cost of rollout policy ˜ Cost E { g ( x, u, y ) } Cost x p xx ( u ) p xy ( u ) p yx ( u ) p yy ( u ) p xt ( u ) p yt ( u ) x y αp xx ( u ) αp xy ( u ) αp yx ( u ) αp yy ( u ) 1 -α Cost 0 Cost g ( x, u, y ) System State Data Control Parameter Estimation Optimal cost Cost of rollout policy ˜ Cost E { g ( x, u, y ) } Cost E { g ( i, u, j

0 1 2 3 4 5 6

Prob. u Prob. 1 -u Cost 1 Cost 1 - √ u J (1) = min { c, a + J (2) } J (2) = b + J (1) J ∗ J µ J µ ′ J µ ′′ J µ 0 J µ 1 J µ 2 J µ 3 J µ 0 f ( x ; θ k ) f ( x ; θ k +1 ) x k F ( x k ) F ( x ) x k +1 F ( x k +1 ) Improper policy µ Proper policy µ Approximate Policy Evaluation Approximate Policy Improvement 0 1 2 3 4 5 6 Deterministic Stochastic Rollout Continuous MPC Constrained Disc MCTS Variance Reduction Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3.3.3 2.4.3 2.4. 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Monte Carlo Tree Search ' 0 1 2 3 4 5 6 Deterministic Stochastic Rollout Continuous MPC Constrain MCTS Variance Reduction Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3.3.3 2. 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Monte Carlo Tree Search ' Approximate Policy Evaluation Approximate Policy Improveme 0 1 2 3 4 5 6 Deterministic Stochastic Rollout Continuous MPC Constraine MCTS Variance Reduction Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3.3.3 2. 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Monte Carlo Tree Search ' Approximate Policy Evaluation Approximately Improved Policy Evaluation Approximate Policy Evaluation Approximate Policy Improvement Stochastic Rollout Continuous MPC Constrained Discrete Combinatorial Multiagent Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3.3.3 2.4.3 2.4.2 3.3, 3.4 { } Value Network Current Policy Network Approximate Policy Approximate Policy Evaluation Approximately Improved Policy Evaluation Approximate Policy Evaluation Approximate Policy Improvement Stochastic Rollout Continuous MPC Constrained Discrete Combinatorial Multiage ) } Approximate Policy Evaluation Approximately Improved Policy Evaluation Approximate Policy Evaluation Approximate Policy Improvement Stochastic Rollout Continuous MPC Constrained Discrete Combinatorial Multiagent identification phase and a control phase . In the first phase the unknown parameters are estimated, while the control takes no account of the interim results of estimation. The final parameter estimates from the first phase are then used to implement an optimal or suboptimal policy in the second phase. This alternation of estimation and control phases may be repeated several times during any system run in order to take into account subsequent changes of the parameters. Moreover, it is not necessary to introduce a hard separation between the identification and the control phases. They may be going on simultaneously, with new parameter estimates being introduced into the control process, whenever this is thought to be desirable; see Fig. 17. This approach is often called on-line replanning and is generally known as indirect adaptive control in the adaptive control literature, see e.g., Astr¨ om and Wittenmark [AsW94].

+

x

) Policy

∗

x

x

k

+2

State Space First Stage

A simple time-honored robust/adaptive control approach for continuous-state problems is PID (Proportional-IntegralDerivative) control . 16 The control theory and practice literature contains extensive accounts. In particular, PID control aims to maintain the output of a single-input single-output dynamic system around a set point or to follow a given trajectory, as the system parameters change within a relatively broad range. In its simplest form, the PID controller is parametrized by three scalar parameters, which may be determined by a variety of methods, some of them manual/heuristic. PID control is used widely and with success, although its range of application is mainly restricted to single-input, single-output continuous-state control systems. 4.2 Dealing with Unknown Parameters Through System Identification and Reoptimization - On-Line Replanning 1 min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x i ) , w i ) Optimal Cost Terminal States Cost Approximation Cost g ( i, u, j ) Po 'Future' 1 min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x Optimal Cost Terminal States Cost Approximation Cost g ( i, 'Future' 1 min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x Optimal Cost Terminal States Cost Approximation Cost g ( i, 'Future' 1 Monte Carlo Tree Search ' min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x i ) , w i ) + ˜ J k + /lscript ( x k + /lscript ) } Optimal Cost Terminal States Cost Approximation Cost g ( i, u, j 'Future' 1 Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3.3.3 2.4.3 2.4.2 3.3, 3.4 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Monte Carlo Tree Search ' min u k ,µ k +1 ,...,µ k /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x i ) , w i ) + ˜ J k + /lscript ( x k + /lscript ) } Optimal Cost Terminal States Cost Approximation Cost g ( i, u, j ) Policy µ State Space First Sta 'Future' 1 Section 2.3 Section 2.4 Sections 2.5, 3.1 3.3 3.4 3.2, 3.3, 3.3.3 2.4.3 2.4.2 3.3, 3.4 2.4.3, 2.4.4 2.4.2 3.3, 3.4 Monte Carlo Tree Search ' min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x i ) , w i ) + ˜ J } Optimal Cost Terminal States Cost Approximation Cost g ( i, u, j ) Policy µ State Space First Stage 1 Unfortunately, there is still another difficulty with this type of on-line replanning: it may be hard to recompute an optimal or near-optimal policy on-line, using a newly identified system model. In particular, it may be impossible to use time-consuming methods that involve for example the training of a neural network or discrete/integer control constraints. A simpler possibility is to use approximation in value space that uses rollout with some kind of robust base policy. We discuss this approach next. 17 4.3 Adaptive Control by Rollout We will now consider dealing with unknown or changing parameters by means of an approximate form of on-line replanning that is based on rollout. Let us assume that some problem parameters change and the current controller becomes aware of the change 'instantly' (i.e., very quickly, before the next control needs to be applied). The method by which the problem parameters are recalculated or become known is immaterial for the purposes of the

of risk in quasi-optimal systems and its comparison with the value of risk in existing systems.'

16 According to Wikipedia, 'a formal control law for what we now call PID or three-term control was first developed using theoretical analysis, by Russian American engineer Nicolas Minorsky' in 1922 [Min22].

µ

k

+

/lscript

(

x

k

+

/lscript

)

17 Still another possibility is to deal with this difficulty by precomputation. In particular, assume that the set of problem parameters may take a known finite set of values (for example each set of parameter values may correspond to a distinct maneuver of a vehicle, motion of a robotic arm, flying regime of an aircraft, etc). Then we may precompute a separate controller for each of these values. Once the control scheme detects a change in problem parameters, it switches to the corresponding predesigned current controller. This is sometimes called a multiple model control design or gain scheduling , and has been applied with success in various settings over the years.

e mu rossible plates

follo

Xk+1

be trunc

------+.

Fig. 1

Consider th infinite hor

18. Schematic illustration of adap

Cost-to-go approximation Expected value approximation

Cost-to-go approximation Expected value approximation

Cost-to-go approximation Expected value approximation o-go approximation Expected value approximation

1

Optimal cost

)

(

x

L

) =

/J

µ

0

(

x

J

K

1

/K

0

L

0

r

J

∗

/K

J

∗

J

Cost-to-go approximation Expected value approximation

T

J

µ

˜

Cost of base policy

µ

1

0

) =

µ

(

x

)

/J

(

x

) =

K

1

/K

0

L

0

r

µ

1

L

(

x

)

r

Cost-to-go approximation Expected value approximation

1

0

-go approximation Expected value approximation l cost

cost

=

J

J

T

J

J

∗

µ

Optimal cost

(

(

J

x

/J

)

x

(

) =

x

T

µ

Optimal cost

µ

=

µ

µ

˜

µ

µ

1

T

T

J

)

/J

µ

T

J J

µ

=

T

µ

µ

µ

∗

µ

rollout policy ˜

J

J

˜

µ

=

T

˜

µ

J

ollout policy ˜

Cost of rollout policy ˜

fied minimization Changing System, Cost, and Con- straint Parameters

µ

ameters

µ

Optimal cost

J

∗

J

(

x

)

/J

(

x

K

/K

L

r

µ

J J

µ

/J

r

µ

=

T

µ

1

(

x

) =

K

0

Optimal cost

0

T µ J J µ = T µ J µ J ˜ µ = T ˜ µ J ˜ µ Cost of base policy µ Cost of rollout policy ˜ µ Optimal Base Rollout Simplified minimization µ 1 µ 0 1 0 0 T µ J J µ = T µ J µ J ˜ µ = T ˜ µ J ˜ µ Cost of base policy µ Cost of rollout policy ˜ µ Optimal Base Rollout Simplified minimization Cost of rollout policy ˜ µ Optimal Base Rollout Simplified minimization Changing System, Cost, and Con-˜ µ J ˜ µ Cost of base policy µ Optimal Base Rollout 0 µ K 1 /K 0 L 0 r µ r µ tation should be fast enough to be performed between successive control applications. Note, however, that accelerated/truncated versions of rollout, as well as parallel computation, can be used to meet this time constraint.

µ

J

µ

J

µ

˜

=

T

µ

˜

Changing System, Cost, and Constraint Parameters

Yields Rollout Policy ˜

Simplified minimization

Changing System, Cost, and Constraint Parameters ed minimization

ing System, Cost, and Constraint Parameters

Linearized Bellman Eq. at g System, Cost, and Constraint Parameters

zed Bellman Eq. at

Through d Bellman Eq. at

h

˜

T

µ

J

µ

=

TJ

µ

Through

T

˜

µ

J

˜

T

µ

J

µ

=

TJ

µ

terations

Value iterations

n

rations

u

)

x

(

U

min xy

p

Rollout with Base Policy

y

∈

=1

with Base Policy

Policy Improvement with Base Policy

∑

with Base Policy

Improvement with Base Policy evaluations for

provement with Base Policy

n

Q-factor minimization

µ

Policy evaluations for n State Constraint Set

aluations for

µ

min xy

(

p

u

)

y

=1

(

g

)

(

x

U

u

)

) +

x, u, y

(

y

rtial Folding

∈

(

∑

n

n

min

u

(

rtial Folding Moving Obstacle

u

∈

U

(

x

)

min

p

y

actor minimization

y

=1

U

(

x

)

mization

∈

=1

k

Possible States

x

∑

Multiagent Q-factor minimization olding Corresponding to Open

aint Set

J

K

∗

J

/K

µ

<!-- image -->

∑

gent Q-factor minimization

X

X X

X

Multiagent

=

k

nt Q-factor minimization

x

˜

ate Constraint Set

X X

=

1

r

b

2

2

-

-

˜

r

αb

αb

˜

2

ation of

n

u

∈

min

m

Expert

-Step

y

)

x

U

(

ase Policy on of

2

{·}

k

k

d

τ

min

y

∈

)

(

x

U

(

u

)

12

x

∗

∑

y

E

n

g

p

=1

(

τ

∈

1

2

∗

u

+1

k

(

R

(

u

min

u

U

u

∈

x

y

min

∗

2

k

U

(

(

x

x

)

∗

1

ximation

, u

3

1

k

1

k

2

k

k

2

∗

x

∗

2

x

x

R

2

(

u

y

2

3

τ

∗

k

τ

u

∗

2

d

d

k

k

x

˜

x

)

+1

0

1

x

T

0

0

3

2

˜

u

1

ristic at

∗

1

u

∗

u

∗

1

∗

1

1

u

∗

x

1

1

u

˜

x

x

∗

x

∗

2

x

∗

1

∗

0

2

ic at

0

u

∗

2

x

∗

u

x

x

2

0

u

2

∗

x

2

∗

x

u

∗

∗

2

Initial

u

)

Multiagent

b

αb

Yields Rollout Policy ˜

Yields Rollout Policy ˜

Lookahead Minimization

Policy Improvement with Base Policy

k

Possible States

Termination State Constraint Set

Lookahead Minimization µ µ and ˜ µ p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x Possible States x x µ µ ) + α ˜ J ( y ) ) µ α ˜ J ( y ) ) We will now present a one-dimensional linear-quadratic example of on-line replanning involving the use of rollout. The purpose of the example is to illustrate how rollout with a policy that is optimal for a nominal set of problem parameters works well when the parameters change from their nominal values. This property is not practically useful in linear-quadratic problems because when the parameters change, it is possible to calculate the new optimal policy in closed form, but it is indicative of the performance robustness of rollout in other contexts.

Possible States

X X

=

X

˜

X

K

X

X

th Base Policy m -Step E {·} : Approximate minimization: ∑ =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) proximation ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 Transition Chosen by Heuristic at x ∗ 1 Rollout Choice 1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Trajectory Tentative Trajectory Optimal Trajectory ChoInitial y Rollout Policy Approximation in Value Space n n -1 r Multistep Lookahead for stages Possible Terminal Cost l Folding l Folding Moving Obstacle ing Corresponding to Open m -Step : Approximate minimization: xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 u ∗ x ∗ ˜ u ˜ x 1 ˜ u 1 ˜ x 1 nsition Chosen by Heuristic at x ∗ 1 Rollout Choice ptimal Solution 2.4.2, 2.4.3 2.4.5 jectory Tentative Trajectory Optimal Trajectory Chooving Obstacle ponding to Open : Approximate minimization: x, u, y ) + α ˜ J ( y ) ) k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C ˜ x 2 ˜ u 2 ˜ x 3 sen by Heuristic at x ∗ 1 Rollout Choice tion 2.4.2, 2.4.3 2.4.5 ntative Trajectory Optimal Trajectory ChoCurrent Partial Folding Moving Obstacle Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at Capacity=1 Optimal Solution 2.4.2, 2.4.3 1 b 2 αb 2 k k +1 r + αb Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 1 r b 2 +1 1 -r αb 2 ˜ K K K ∗ K k K k +1 F ( K ) = αrK r + αb 2 K +1 Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 1 Termination State Constraint Set X X = X ˜ X Multiagent r b 2 +1 1 -r αb 2 ˜ K K K ∗ K k K k +1 F ( K ) = αrK r + αb 2 K +1 Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 1 ation State Constraint Set X X = X ˜ X Multiagent K K K ∗ K k K k +1 F ( K ) = αrK r + αb 2 K +1 t Partial Folding Moving Obstacle te Folding Corresponding to Open with Base Policy m -Step Value Network Policy Network ximation of E {·} : Approximate minimization: x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) r approximation 0 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 1 Termination State Constraint Set X X = X ˜ X Multiagent r +1 1 -r αb 2 ˜ K K K ∗ K k K k +1 F ( K ) = αrK r + αb 2 K +1 Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x u ∗ x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice 1 k k +1 k + m +1 tion State Constraint Set X X = X ˜ X Multiagent K K K ∗ K k K k +1 F ( K ) = αrK r + αb 2 K +1 Partial Folding Moving Obstacle e Folding Corresponding to Open with Base Policy m -Step Value Network Policy Network imation of E {·} : Approximate minimization: n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) approximation 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 st Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Fig. 18. Schematic illustration of adaptive control by rollout. One-step lookahead is followed by simulation with the base policy, which stays fixed. The system, cost, and constraint parameters are changing over time, and the most recent values are incorporated into the lookahead minimization and rollout operations. For the discussion in this section, we may assume that all the changing parameter information is provided by some computation and sensor 'cloud' that is beyond our control. The base policy may also be revised based on various criteria. Moreover the lookahead minimization may involve multiple steps, while the rollout may be truncated. following discussion. It may involve a limited form of parameter estimation, whereby the unknown parameters are 'tracked' by data collection over a few time stages; or it may involve new features of the control environment, such as a changing number of servers and/or tasks in a service system. x k +1 = x k + bu k , and the quadratic cost function lim N →∞ N -1 ∑ k =0 ( x 2 k + ru 2 k ) . The optimal cost function is given by J ∗ ( x ) = K ∗ x 2 , where K ∗ is solves the Riccati equation K = rK r + b 2 K +1 . The optimal policy has the form µ ∗ ( x ) = L ∗ x, where L ∗ = -bK ∗ r + b 2 K ∗ .

min tion

∈

w

E

y

U

x

(

k

(

˜

) At

x

(

-Factor

Q

x, u in

{

, u, w

U

α

) +

x

g

(

E

w

)

(

lout Policy

Steps

µ m

1

tion in Policy Space Heuristic Cost Approximation for Beyond Truncation y k Feature States y k +1 Cost g k ( x k , u k ) te Q-Factor ˜ Q ( x, u ) At x Approximation ˆ J { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) ) } Rollout Policy µ m Steps 1 0 ollout Policy Approximation in Value Space n n -1 ultistep Lookahead for stages Possible Terminal Cost in Policy Space Heuristic Cost Approximation for nd Truncation y k Feature States y k +1 Cost g k ( x k , u k ) x, u ) At x Approximation ˆ J x, u, w ) + α ˜ J ( f ( x, u, w ) ) } µ m Steps 1 tial cy Approximation in Value Space n n -1 okahead for stages Possible Terminal Cost Space Heuristic Cost Approximation for Feature States y k +1 Cost g k ( x k , u k ) Approximation ˆ J ˜ J ( f ( x, u, w ) ) } 1 We thus assume away/ignore issues of parameter estimation, and focus on revising the controller by on-line replanning based on the newly obtained parameters. This revision may be based on any suboptimal method, but rollout with the current policy used as the base policy is particularly attractive. Here the advantage of rollout is that it is simple and reliable. In particular, it does not require a complicated training procedure to revise the current policy, based for example on the use of neural networks or other approximation architectures, so no new policy is explicitly computed in response to the parameter changes . Instead the current policy is used as the base policy for (possibly truncated) rollout, and the available controls at the current state are compared by a one-step or mutistep minimization, with cost function approximation provided by the base policy (cf. Fig. 18).

Note that over time the base policy may also be revised (on the basis of an unspecified rationale). In this case the rollout policy will be adjusted both in response to the changed current policy and in response to the changing parameters. This is necessary in particular when the constraints of the problem change.

The principal requirement for using rollout in an adaptive control context is that the rollout control compu-

<!-- formula-not-decoded -->

x

X

F

K

K

(

X X = X ˜ X Multiagent k +1 F ( K ) = αrK r + αb 2 K +1 x k +1 x k + m +1 ˜ X Multiagent αrK 2 +1 x k +1 x k + m +1 Multiagent Consider the deterministic one-dimensional undiscounted infinite horizon linear-quadratic problem involving the linear system x, u, y

y

)

)

) =

(

y

k

∗

Rollout Choice

Rollout Choice

1

∗

2.4.5

We will consider the nominal problem parameters b = 2 and r = 0 . 5. We can then verify that for these parameters, the corresponding optimal cost and optimal policy coefficients are

We will now consider changes of the values of b and r while keeping L constant, and we will compare the quadratic cost coefficient of the following cost functions as b and r vary:

- (a) The optimal cost function K ∗ x 2 .

<!-- formula-not-decoded -->

- (b) The cost function K L x 2 that corresponds to the base policy µ L ( x ) = Lx. From our earlier discussion, we have
- (c) The cost function ˜ K L x 2 that corresponds to the rollout policy ˜ µ L ( x ) = ˜ Lx, obtained by using the policy µ L as base policy. Using the formulas given earlier, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

) =

-

Possible States

x

k

+1

x

k

+

m

+1

x

Possible States

µ

k

+1

k

+

m

r

x

+1 1

-

r

x

˜

K K K

∗

K

K

1

and

+1

Cost-to-go approximation Expected value approximation

Optimal cost

J

T

J J

J

µ

µ

T

=

µ

Cost of rollout policy ˜

Optimal cost

Simplified minimization

µ

µ

µ

J J

T

T

=

Policy Improvement

Cost-to-go approximation Expected value approximation

Optimal cost

Cost of rollout policy ˜

J

T

Simplified minimization

µ

µ

Policy Improvement

µ

µ

J J

=

T

J

u

Cost of rollout policy ˜

Simplified minimization

Policy Improvement

Multiagent Q-factor minimization

Termination State Constraint Set

r

r

b

2

Multiagent Q-factor minimization

αb

+1 1

2

u

-

Termination State Constraint Set

Current Partial Folding Moving Obstacle

r

r

b

T

µ

αb

+1 1

Complete Folding Corresponding to Open

2

Expert

Multiagent Q-factor minimization

-

Termination State Constraint Set

r

Current Partial Folding Moving Obstacle

Rollout with Base Policy

Complete Folding Corresponding to Open

2

2

+1 1

r

b

αb

Cost-to-go approximation Expected value approximation

Approximation of

-

Expert

Rollout with Base Policy

Current Partial Folding Moving Obstacle

Value iterations Policy evaluations

Complete Folding Corresponding to Open

Expert

Approximation of

u

Rollout with Base Policy

k

x

d

x

, u

u

k

k

k

1

2

2

1

Approximation of

<!-- image -->

Termination State Constraint Set

{·}

Q-factor approximation

u

u

ˆ

0

0

1

x

0

x

x

1

u

0

∗

0

Base Policy Rollout Policy Approximation in Value Space

0

u

x

∗

-

µ

Value iterations Policy evaluations

x

Possible States

k

n n

˜

1

Permanent Trajectory Tentative Trajectory Optimal Trajectory Cho- sen by Base Heuristic at

+1

k

k

+

m

+1

x

x

Multiagent Q-factor minimization

˜

min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Choen by Base Heuristic at x 0 Initial x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x u ∗ x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Termination State Constraint Set X X = X ˜ X Multiagent r b 2 +1 1 -r αb 2 ˜ K K K ∗ K k K k +1 F ( K ) = αrK r + αb 2 K +1 Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u U ( x ) n p xy ( u ) g ( x, u, y min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) Multiagent Q-factor minimization x k Possible States x k +1 Termination State Constraint Set X X = X ˜ X Multiagent r b 2 +1 1 -r αb 2 ˜ K K K ∗ K k K k +1 F ( K ) = αrK r + αb 2 K +1 Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open Expert Rollout with Base Policy m r b 2 +1 1 -r αb 2 ˜ K K K ∗ K k K k +1 F ( K ) = αrK r + αb 2 K +1 Current Partial Folding Moving Obstacle Complete Folding Corresponding to Open Expert Rollout with Base Policy m -Step Value Network Policy Network Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) Fig. 19. Illustration of adaptive control by rollout under changing problem parameters. The quadratic cost coefficients K ∗ (optimal, denoted by solid line), K L (base policy, denoted by circles), and ˜ K L (rollout policy, denoted by asterisks) for the two separate cases where r = 0 . 5 and b varies, and b = 2 and r varies. The value of L is fixed at the value that is optimal for b = 2 and r = 0 . 5

-

2

2

-Step Value Network Policy Network

) +

˜

C

∈

)

+1

)

)

y

(

J

α

-

, u

k

y

(

, R

k

y

1 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost 1 u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost High Cost Transition Chosen by Heuristic at x ∗ 1 Rollout Choice Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Chosen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost 1 The difference K L -K ∗ is indicative of the robustness of the policy µ L , i.e., the performance loss incurred by ignoring the values of b and r , and continuing to use the policy µ L , which is optimal for the nominal values b = 2 and r = 0 . 5, but suboptimal for other values of b and r . The difference ˜ K L -K ∗ is indicative of the performance loss due to using on-line replanning by rollout rather than using optimal replanning. Finally, the difference K L -˜ K L is indicative of the performance improvement due to online replanning using rollout rather than keeping the policy µ L unchanged.

=1

y

∈

Initial

x

Capacity=1 Optimal Solution 2.4.2, 2.4.3 2.4.5 Permanent Trajectory Tentative Trajectory Optimal Trajectory Choen by Base Heuristic at x 0 Initial Base Policy Rollout Policy Approximation in Value Space n n -1 One-Step or Multistep Lookahead for stages Possible Terminal Cost One-Step or Multistep Lookahead for stages Possible Terminal Cost 1 Base Policy Rollout Policy Approximation in Value Space n n -1 n -2 One-Step or Multistep Lookahead for stages Possible Terminal Cost 1 x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( k ) x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 Approximation of E {·} : Approximate minimization: min u ∈ U ( x ) n ∑ y =1 p xy ( u ) ( g ( x, u, y ) + α ˜ J ( y ) ) x 1 k , u 1 k u 2 k x 2 k d k τ x 1 k , u 1 k u 2 k x 2 k d k τ Q-factor approximation u 1 ˆ u 1 10 11 12 R ( y k +1 ) T k (˜ y k , u k ) = ( ˜ y k , u k , R ( y k +1 ) ) ∈ C x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 1 ˜ x 2 ˜ u 2 ˜ x 3 x 0 u ∗ 0 x ∗ 1 u ∗ 1 x ∗ 2 u ∗ 2 x ∗ 3 ˜ u 0 ˜ x 1 ˜ u 1 ˜ x 1 Figure 19 shows the coefficients K ∗ , K L , and ˜ K L for a range of values of r and b . As predicted by the cost improvement property of rollout, we have K ∗ ≤ ˜ K L ≤ K L .

## 5. CONCLUDING REMARKS

We have argued that the connections between the MPC and RL fields are strong, and that the most successful design architectures of the two fields share important characteristics, which relate to Newton's method. Indeed, in the author's view, a principal theoretical reason for the successes of the two fields is the off-line training/online play synergism that rests upon the mathematical foundations of Newton's method.

Still the cultures of MPC and RL have different starting points and have grown in different directions. One of the primary reasons is the preference for continuous state and control spaces in MPC, which stems from the classical control theory tradition. Thus stability and safety/reachability issues have been of paramount importance in MPC, but they are hardly ever considered in RL. The main reason is that stability does not arise mathematically or practically in the discrete state and control contexts of games, Markovian decision problems, and more recently large language models that are favored in RL. At the same time, the ideas of learning from data are not part of the control theory tradition, and they have only been addressed relatively recently in a systematic way.

Multiagent x k + m +1 The framework that we have presented in this paper also aims to support a trend of increased use of machine learning methods in MPC. The fact that at their foundation, MPCand RL share important principles suggests that this trend will continue and accelerate in the future.

## 6. REFERENCES

[ABQ99] Allgower, F., Badgwell, T. A., Qin, J. S., Rawlings, J. B., and Wright, S. J., 1999. 'Nonlinear Predictive Control and Moving Horizon Estimation - An Introductory Overview,'Advances in Control: Highlights of ECC'99, pp. 391-449.

[AGH19] Andersson, J. A., Gillis, J., Horn, G., Rawlings, J. B., and Diehl, M., 2019. 'CasADi: A Software Framework for Nonlinear Optimization and Optimal Control,' Math. Programming Computation, Vol. 11, pp. 1-36.

1 [Abr90] Abramson, B., 1990. 'Expected-Outcome: A General Model of Static Evaluation,' IEEE Trans. on Pattern Analysis and Machine Intelligence, Vol. 12, pp. 182-193.

n n -1 [AsH06] Astr¨ om, K. J., and Hagglund, T., 2006. Advanced PID Control, Instrument Society of America, Research Triangle Park, NC.

1 It can be seen that the rollout policy performance is very close to the one of the exactly reoptimized policy, while the base policy yields much worse performance. This is a consequence of the superlinear convergence rate of Newton's method that underlies rollout:

<!-- formula-not-decoded -->

where for a given initial state x , ˜ J ( x ) is the rollout cost, J ∗ ( x ) is the optimal cost, and J ( x ) is the base policy cost.

[AsW94] Astr¨ om, K. J., and Wittenmark, B., 1994. Adaptive Control, 2nd Ed., Prentice-Hall, Englewood Cliffs, NJ.

[BBB22] Bhambri, S., Bhattacharjee, A., and Bertsekas, D. P., 2022. 'Reinforcement Learning Methods for Wordle: A POMDP/Adaptive Control Approach,' arXiv:2211.10298.

[BBM17] Borrelli, F., Bemporad, A., and Morari, M., 2017. Predictive Control for Linear and Hybrid Systems, Cambridge Univ. Press, Cambridge, UK.

[BDL09] Bolte, J., Daniilidis, A., and Lewis, A., 2009. 'Tame Functions are Semismooth,' Math. Programming, Vol. 117, pp. 5-19.

X

(

X X

∑

Q-factor approximation

0

(

)

=

X

[BGH22] Brunke, L., Greeff, M., Hall, A. W., Yuan, Z., Zhou, S., Panerati, J., and Schoellig, A. P., 2022. 'Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning,' Annual Review of Control, Robotics, and Autonomous Systems, Vol. 5, pp. 411-444.

[BKB20] Bhattacharya, S., Kailas, S., Badyal, S., Gil, S., and Bertsekas, D. P., 2020. 'Multiagent Rollout and Policy Iteration for POMDP with Application to MultiRobot Repair Problems,' in Proc. of Conference on Robot Learning (CoRL); also arXiv preprint, arXiv:2011.04222.

[BTW97] Bertsekas, D. P., Tsitsiklis, J. N., and Wu, C., 1997. 'Rollout Algorithms for Combinatorial Optimization,' Heuristics, Vol. 3, pp. 245-262.

[BeC99] Bertsekas, D. P., and Casta˜ non, D. A., 1999. 'Rollout Algorithms for Stochastic Scheduling Problems,' Heuristics, Vol. 5, pp. 89-108.

[BeK65] Bellman, R., and Kalaba, R. E., 1965. Quasilinearization and Nonlinear Boundary-Value Problems, Elsevier, NY.

[BeM99] Bemporad, A., and Morari, M., 1999. 'Control of Systems Integrating Logic, Dynamics, and Constraints,' Automatica, Vol. 35, pp. 407-427.

[BeP21] Bemporad, A., and Piga, D., 2021. 'Global Optimization Based on Active Preference Learning with Radial Basis Functions,' Machine Learning, Vol. 110, pp. 417-448.

[BeR71] Bertsekas, D. P., and Rhodes, I. B., 1971. 'On the Minimax Reachability of Target Sets and Target Tubes,' Automatica, Vol. 7, pp. 233-247.

[BeS78] Bertsekas, D. P., and Shreve, S. E., 1978. Stochastic Optimal Control: The Discrete Time Case, Academic Press, NY.; republished by Athena Scientific, Belmont, MA, 1996 (can be downloaded from the author's website).

[BeT96] Bertsekas, D. P., and Tsitsiklis, J. N., 1996. Neuro-Dynamic Programming, Athena Scientific, Belmont, MA.

[Ber71] Bertsekas, D. P., 1971. 'Control of Uncertain Systems With a Set-Membership Description of the Uncertainty,' Ph.D. Dissertation, Massachusetts Institute of Technology, Cambridge, MA (can be downloaded from the author's website).

[Ber72] Bertsekas, D. P., 1972. 'Infinite Time Reachability of State Space Regions by Using Feedback Control,' IEEE Trans. Aut. Control, Vol. AC-17, pp. 604-613.

[Ber77] Bertsekas, D. P., 1977. 'Monotone Mappings with Application in Dynamic Programming,' SIAM J. on Control and Opt., Vol. 15, pp. 438-464.

[Ber97] Bertsekas, D. P., 1997. 'Differential Training of Rollout Policies,' Proc. of the 35th Allerton Conference on Communication, Control, and Computing, Allerton, Ill.

[Ber05a] Bertsekas, D. P., 2005. 'Dynamic Programming and Suboptimal Control: A Survey from ADP to MPC,' European J. of Control, Vol. 11, pp. 310-334.

[Ber05b] Bertsekas, D. P., 2005. 'Rollout Algorithms for Constrained Dynamic Programming,' Lab. for Information and Decision Systems Report LIDS-P-2646, MIT.

[Ber17a] Bertsekas, D. P., 2017. Dynamic Programming and Optimal Control, Vol. I, Athena Scientific, Belmont, MA.

[Ber17b] Bertsekas, D. P., 2017. 'Value and Policy Iteration in Deterministic Optimal Control and Adaptive Dynamic Programming,' IEEE Transactions on Neural Networks and Learning Systems, Vol. 28, pp. 500-509.

[Ber19] Bertsekas, D. P., 2019. Reinforcement Learning and Optimal Control, Athena Scientific, Belmont, MA.

[Ber20] Bertsekas, D. P., 2020. Rollout, Policy Iteration, and Distributed Reinforcement Learning, Athena Scientific, Belmont, MA.

[Ber21a] Bertsekas, D. P., 2021. 'Multiagent Reinforcement Learning: Rollout and Policy Iteration,' IEEE/CAA Journal of Automatica Sinica, Vol. 8, pp. 249-271.

[Ber21b] Bertsekas, D. P., 2021. 'Distributed Asynchronous Policy Iteration for Sequential Zero-Sum Games and Minimax Control,' arXiv:2107.10406

[Ber22a] Bertsekas, D. P., 2022. 'Lessons from AlphaZero for Optimal, Model Predictive, and Adaptive Control,' Athena Scientific, Belmont, MA (can be downloaded from the author's website).

[Ber22b] Bertsekas, D. P., 2022. Abstract Dynamic Programming, 3rd Ed., Athena Scientific, Belmont, MA (can be downloaded from the author's website).

[Ber22c] Bertsekas, D. P., 2022. 'Newton's Method for Reinforcement Learning and Model Predictive Control,' Results in Control and Optimization, Vol. 7, pp. 100-121.

[Ber23] Bertsekas, D. P., 2023. 'A Course in Reinforcement Learning,' Athena Scientific, 2023 (can be downloaded from the author's website).

[Bla99] Blanchini, F., 1999. 'Set Invariance in Control A Survey,' Automatica, Vol. 35, pp. 1747-1768.

[Bod20] Bodson, M., 2020. Adaptive Estimation and Control, Independently Published.

[CFM20] Chen, S., Fazlyab, M., Morari, M., Pappas, G. J., and Preciado, V. M., 2020. 'Learning Lyapunov Functions for Piecewise Affine Systems with Neural Network Controllers,' arXiv preprint arXiv:2008.06546.

[CLD19] Coulson, J., Lygeros, J., and Dorfler, F., 2019. 'Data-Enabled Predictive Control: In the Shallows of the DeePC,' 18th European Control Conference, pp. 307-312.

[CLL23] Choi, J. J., Lee, D., Li, B., How, J. P., Sreenath, K., Herbert, S. L., and Tomlin, C. J., 2023. 'A Forward Reachability Perspective on Robust Control Invariance and Discount Factors in Reachability Analysis,' arXiv preprint arXiv:2310.17180.

[CMT87a] Clarke, D. W., Mohtadi, C., and Tuffs, P. S., 1987. 'Generalized Predictive Control - Part I. The Basic Algorithm,' Automatica, Vol. 23, pp. 137-148.

[CMT87b] Clarke, D. W., Mohtadi, C., and Tuffs, P. S., 1987. 'Generalized Predictive Control Part II,' Automatica, Vol. 23, pp. 149-160.

[CWA22] Chen, S. W., Wang, T., Atanasov, N., Kumar, V., and Morari, M., 2022. 'Large Scale Model Predictive Control with Neural Networks and Primal Active Sets,' Automatica, Vol. 135.

[CaB07] Camacho, E. F., and Bordons, C., 2007. Model Predictive Control, 2nd Ed., Springer, New York, NY.

[DFH09] Diehl, M., Ferreau, H. J., and Haverbeke, N., 2009. 'Efficient Numerical Methods for Nonlinear MPC and Moving Horizon Estimation,' in Nonlinear Model Predictive Control: Towards New Challenging Applications, by L. Magni, D. M. Raimondo, F. Allgower (eds.), Springer, pp. 391-417.

[DMS98] De Nicolao, G., Magni, L., and Scattolini, R., 1998. 'Stabilizing Receding-Horizon Control of Nonlinear Time-Varying Systems,' IEEE Transactions on Aut. Control, Vol. 43, pp. 1030-1036.

[DuM23] Duan, Y., and Wainwright, M.J., 2023. 'A Finite-Sample Analysis of Multi-Step Temporal Difference Estimates,' in Learning for Dynamics and Control Conference, N. Matni, M. Morari, G. J. Pappas (eds.), Proc. of Machine Learning Research, pp. 612-624.

[FHS09] Feitzinger, F., Hylla, T., and Sachs, E. W., 2009. 'Inexact Kleinman-Newton Method for Riccati Equations,' SIAM Journal on Matrix Analysis and Applications, Vol. 3, pp. 272-288.

[FXB22] Fu, A., Xing, L., and Boyd, S., 2022. 'Operator Splitting for Adaptive Radiation Therapy with Nonlinear Health Dynamics,' Optimization Methods and Software, Vol. 37, pp. 2300-2323.

[FIA03] Findeisen, R., Imsland, L., Allgower, F., and Foss, B.A., 2003. 'State and Output Feedback Nonlinear Model Predictive Control: An Overview,' European Journal of Control, Vol. 9, pp. 190-206.

[FaP03] Facchinei, F., and Pang, J.-S., 2003. FiniteDimensional Variational Inequalities and Complementarity Problems, Vols I and II, Springer, NY.

[Fel60] Feldbaum, A. A., 1960. 'Dual Control Theory,' Automation and Remote Control, Vol. 21, pp. 874-1039.

[Fel63] Feldbaum, A. A., 1963. 'Dual Control Theory Problems,' IFAC Proceedings, pp. 541-550.

[GFA11] Gonzalez, R., Fiacchini, M., Alamo, T., Guzman, J. L., and Rodriguez, F., 2011. 'Online Robust TubeBased MPC for Time-Varying Systems: A Practical Approach,' International Journal of Control, Vol. 84, pp. 1157-1170.

[GPG22] Garces, D., Bhattacharya, S., Gil, G., and Bertsekas, D., 2022. 'Multiagent Reinforcement Learning for Autonomous Routing and Pickup Problem with Adaptation to Variable Demand,' arXiv preprint arXiv:2211.14983.

[GSD06] Goodwin, G., Seron, M. M., and De Dona, J. A., 2006. Constrained Control and Estimation: An Optimisation Approach, Springer, NY.

[GoS84] Goodwin, G. C., and Sin, K. S. S., 1984. Adaptive Filtering, Prediction, and Control, Prentice-Hall, Englewood Cliffs, N. J.

[GrZ19] Gros, S., and Zanon, M., 2019. 'Data-Driven Economic NMPC Using Reinforcement Learning,' IEEE Trans. on Aut. Control, Vol. 65, pp. 636-648.

[GrZ22] Gros, S., and Zanon, M., 2022. 'Learning for MPC with Stability and Safety Guarantees,' Automatica, Vol. 146, pp. 110598.

[HWM20] Hewing, L., Wabersich, K. P., Menner, M., and Zeilinger, M. N., 2020. 'Learning-Based Model Predictive Control: Toward Safe Learning in Control,' Annual Review of Control, Robotics, and Autonomous Systems, Vol. 3, pp. 269-296.

[Hew71] Hewer, G., 1971. 'An Iterative Technique for the Computation of the Steady State Gains for the Discrete Optimal Regulator,' IEEE Trans. on Aut. Control, Vol. 16, pp. 382-384.

[Hyl11] Hylla, T., 2011. Extension of Inexact KleinmanNewton Methods to a General Monotonicity Preserving Convergence Theory, PhD Thesis, Univ. of Trier.

[IoS96] Ioannou, P. A., and Sun, J., 1996. Robust Adaptive Control, Prentice-Hall, Englewood Cliffs, N. J.

[ItK03] Ito, K., and Kunisch, K., 2003. 'Semi-Smooth Newton Methods for Variational Inequalities of the First Kind,' Mathematical Modelling and Numerical Analysis, Vol. 37, pp. 41-62.

[JiJ17] Jiang, Y., and Jiang, Z. P., 2017. Robust Adaptive Dynamic Programming, J. Wiley, NY.

[Jos79] Josephy, N. H., 1979. 'Newton's Method for Generalized Equations,' Wisconsin Univ-Madison, Mathematics Research Center Report No. 1965.

[KGB82] Kimemia, J., Gershwin, S. B., and Bertsekas, D. P., 1982. 'Computation of Production Control Policies by a Dynamic Programming Technique,' in Analysis and Optimization of Systems, A. Bensoussan and J. L. Lions (eds.), Springer, N. Y., pp. 243-269.

[KKK95] Krstic, M., Kanellakopoulos, I., Kokotovic, P., 1995. Nonlinear and Adaptive Control Design, J. Wiley, NY.

[KRW21] Kumar, P., Rawlings, J. B., and Wright, S. J., 2021. 'Industrial, Large-Scale Model Predictive Control with Structured Neural Networks,' Computers and Chemical Engineering, Vol. 150.

[KeG88] Keerthi, S. S., and Gilbert, E. G., 1988. 'Optimal, Infinite Horizon Feedback Laws for a General Class of Constrained Discrete Time Systems: Stability and MovingHorizon Approximations,' J. Optimization Theory Appl., Vo. 57, pp. 265-293.

[Ker00] Kerrigan, E. C., 2000. Robust Constraint Satisfaction: Invariant Sets and Predictive Control, PhD. Thesis, University of London.

[Kle68] Kleinman, D. L., 1968. 'On an Iterative Technique for Riccati Equation Computations,' IEEE Trans. Aut. Control, Vol. AC-13, pp. 114-115.

[KoC16] Kouvaritakis, B., and Cannon, M., 2016. Model Predictive Control: Classical, Robust and Stochastic, Springer, NY.

[KoG98] Kolmanovsky, I., and Gilbert, E. G., 1998. 'Theory and Computation of Disturbance Invariant Sets for Discrete-Time Linear Systems,' Mathematical Problems in Engineering, Vol. 4, pp. 317-367.

[KoS86] Kojima, M., and Shindo, S., 1986. 'Extension of Newton and Quasi-Newton Methods to Systems of PC 1 Equations,' J. of the Operations Res. Society of Japan, Vol. 29, pp. 352-375.

[Kre19] Krener, A. J., 2019. 'Adaptive Horizon Model Predictive Control and Al'brekht's Method,' arXiv preprint arXiv:1904.00053.

[KuV86] Kumar, P. R., and Varaiya, P. P., 1986. Stochastic Systems: Estimation, Identification, and Adaptive Control, Prentice-Hall, Englewood Cliffs, N. J.

[Kum88] Kummer, B., 1988. 'Newton's Method for NonDifferentiable Functions,' Mathematical Research, Vol. 45, pp. 114-125.

[Kum00] Kummer, B., 2000. 'Generalized Newton and NCP-methods: Convergence, Regularity, Actions,' Discussiones Mathematicae, Differential Inclusions, Control and Optimization, Vol. 2, pp. 209-244.

[LFE13] Lucia, S., Finkler, T., and Engell, S., 2013. 'Multi-Stage Nonlinear Model Predictive Control Applied to a Semi-Batch Polymerization Reactor Under Uncertainty,' Journal of Process Control, Vol. 23, pp. 1306-1319.

[LHK18] Liao-McPherson, D., Huang, M., and Kolmanovsky, I., 2018. 'A Regularized and Smoothed Fischer?Burmeister Method for Quadratic Programming with Applications to Model Predictive Control,' IEEE Trans. on Automatic Control, Vol. 64, pp. 2937-2944.

[LJM21] Li, Y., Johansson, K. H., Martensson, J., and Bertsekas, D. P., 2021. 'Data-Driven Rollout for Deterministic Optimal Control,' arXiv preprint arXiv:2105.03116.

[LKL23] Li, Y., Karapetyan, A., Lygeros, J., Johansson, K. H., and Martensson, J., 2023. 'Performance Bounds of Model Predictive Control for Unconstrained and Constrained Linear Quadratic Problems and Beyond,' IFACPapers On Line, Vol. 56, pp. 8464-8469.

[LWW17] Liu, D., Wei, Q., Wang, D., Yang, X., and Li, H., 2017. Adaptive Dynamic Programming with Applications in Optimal Control, Springer, Berlin.

[LaW13] Lavretsky, E., and Wise, K., 2013. Robust and Adaptive Control with Aerospace Applications, Springer.

[Li23] Li, Y., 2023. Approximate Methods of Optimal Control via Dynamic Programming Models, PhD Thesis, Royal Institute of Technology, Stockholm.

[LiB24] Li, Y., and Bertsekas, D., 2024. 'Most Likely Sequence Generation for n -Grams, Transformers, HMMs, and Markov Chains, by Using Rollout Algorithms,' arXiv:2403.15465.

[MBS23] Moreno-Mora, F., Beckenbach, L., and Streif, S., 2023. 'Predictive Control with Learning-Based Terminal Costs Using Approximate Value Iteration,' IFAC-Papers On Line, Vol. 56, pp. 3874-3879.

[MDM01] Magni, L., De Nicolao, G., Magnani, L., and Scattolini, R., 2001. 'A Stabilizing Model-Based Predictive Control Algorithm for Nonlinear Systems,' Automatica, Vol. 37, pp. 1351-1362.

[MGQ20] Mittal, M., Gallieri, M., Quaglino, A., Salehian, S., and Koutnik, J., 2020. 'Neural Lyapunov Model Predictive Control: Learning Safe Global Controllers from Suboptimal Examples,' arXiv preprint arXiv:2002.10451.

[MDT22] Mukherjee, S., Drgona, J., Tuor, A., Halappanavar, M., and Vrabie, D., 2022. Neural Lyapunov Differentiable Predictive Control,' 2022 IEEE 61st Conference on Decision and Control, pp. 2097-2104.

[MJR22] Mania, H., Jordan, M. I., and Recht, B., 2022. 'Active Learning for Nonlinear System Identification with Guarantees,' J. of Machine Learning Research, Vol. 23, pp. 1-30.

[MLW24] Musunuru, P., Li, Y., Weber, J., and Bertsekas, D., 'An Approximate Dynamic Programming Framework for Occlusion-Robust Multi-Object Tracking,' ArXiv Preprint arXiv:2405.15137, May 2024.

[MRR00] Mayne, D., Rawlings, J. B., Rao, C. V., and Scokaert, P. O. M., 2000. 'Constrained Model Predictive Control: Stability and Optimality,' Automatica, Vol. 36, pp. 789-814.

[MaM88] Mayne, D. Q., and Michalska, H., 1988. 'Receding Horizon Control of Nonlinear Systems,' Proc. of the 27th IEEE Conf. on Decision and Control, pp. 464-465.

[MaS04] Magni, L., and Scattolini, R., 2004. 'Stabilizing Model Predictive Control of Nonlinear Continuous Time Systems,' Annual Reviews in Control, Vol. 28, pp. 1-11.

[May14] Mayne, D. Q., 2014. 'Model Predictive Control: Recent Developments and Future Promise,' Automatica, Vol. 50, pp. 2967-2986.

[Min22] Minorsky, N., 1922. 'Directional Stability of Automatically Steered Bodies,' J. Amer. Soc. Naval Eng.,Vol. 34, pp. 280-309.

[MoL99] Morari, M., and Lee, J. H., 1999. 'Model Predictive Control: Past, Present, and Future,' Computers and Chemical Engineering, Vol. 23, pp. 667-682.

[NaA12] Narendra, K. S., and Annaswamy, A. M., 2012. Stable Adaptive Systems, Courier Corp.

[OSB13] O'Donoghue, B., Stathopoulos, G., and Boyd, S., 2013. 'A Splitting Method for Optimal Control,' IEEE Trans. on Control Systems Technology, Vol. 21, pp. 24322442.

[Pan90]

Pang,

J.

S.,

1990.

'Newton's

Method for

B-

Differentiable Equations,' Math. of Operations Res., Vol.

15, pp. 311-341.

[PoA69] Pollatschek, M. A. and Avi-Itzhak, B., 1969. 'Algorithms for Stochastic Games with Geometrical Interpretation,' Management Science, Vol. 15, pp. 399-415.

[PuB78] Puterman, M. L., and Brumelle, S. L., 1978. 'The Analytic Theory of Policy Iteration,' in Dynamic Programming and Its Applications, M. L. Puterman (ed.), Academic Press, NY.

[PuB79] Puterman, M. L., and Brumelle, S. L., 1979. 'On the Convergence of Policy Iteration in Stationary Dynamic Programming,' Math. of Operations Res., Vol. 4, pp. 6069.

[Qi93] Qi, L., 1993. 'Convergence Analysis of Some Algorithms for Solving Nonsmooth Equations,' Math. of Operations Res., Vol. 18, pp. 227-244.

[QiS93] Qi, L., and Sun, J., 1993. 'A Nonsmooth Version of Newton's Method,' Math. Programming, Vol. 58, pp. 353-367.

[RDM24] Ruoss, A., Del´ etang, G., Medapati, S., GrauMoya, J., Wenliang, L. K., Catt, E., Reid, J., and Genewein, T., 2024. 'Grandmaster-Level Chess Without Search,' arXiv:2402.04494.

[RKM06] Rakovic, S. V., Kerrigan, E. C., Mayne, D. Q., and Lygeros, J., 2006. 'Reachability Analysis of DiscreteTime Systems with Disturbances,' IEEE Trans. on Aut. Control, Vol. 51, pp. 546-561.

[RMD17] Rawlings, J. B., Mayne, D. Q., and Diehl, M. M., 2017. Model Predictive Control: Theory, Computation, and Design, 2nd Ed., Nob Hill Publishing.

[RaL18] Rakovic, S. V., and Levine, W. S., eds., 2018. Handbook of Model Predictive Control, Springer.

[RaR17] Rawlings, J. B., and Risbeck, M. J., 2017. 'Model Predictive Control with Discrete Actuators: Theory and Application,' Automatica, Vol. 78, pp. 258-265.

[Rec19] Recht, B., 2019. 'A Tour of Reinforcement Learning: The View from Continuous Control,' Annual Review of Control, Robotics, and Autonomous Systems, Vol. 2, pp. 253-279.

[RoB17] Rosolia, U., and Borrelli, F., 2017. 'Learning Model Predictive Control for Iterative Tasks. A DataDriven Control Framework,' IEEE Trans. on Aut. Control, Vol. 63, pp. 1883-1896.

[Rob80] Robinson, S. M., 1980. 'Strongly Regular Generalized Equations,' Math. of Operations Res., Vol. 5, pp. 43-62.

[Rob88] Robinson, S. M., 1988. 'Newton's Method for a Class of Nonsmooth Functions,' Industrial Engineering Working Paper, University of Wisconsin; also in SetValued Analysis Vol. 2, 1994, pp. 291-305.

[Rob11] Robinson, S. M., 2011. 'A Point-of-Attraction Result for Newton's Method with Point-Based Approximations,' Optimization, Vol. 60, pp. 89-99.

[SHM16] Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., and Dieleman, S., 2016. 'Mastering the Game of Go with Deep Neural Networks and Tree Search,' Nature, Vol. 529, pp. 484-489.

[SHS17] Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., and Lillicrap, T., 2017. 'Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm,' arXiv:1712.01815.

[SKG22] Seel, K., Kordabad, A. B., Gros, S., and Gravdahl, J. T., 2022. 'Convex Neural Network-Based Cost Modifications for Learning Model Predictive Control,' IEEE Open Journal of Control Systems, Vol. 1, pp. 366379.

[SSS17] Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A. and Chen, Y., 2017. 'Mastering the Game of Go Without Human Knowledge,' Nature, Vol. 550, pp. 354-359.

[SaB11] Sastry, S., and Bodson, M., 2011. Adaptive Control: Stability, Convergence and Robustness, Courier Corp.

[SiB22] Silver, D., and Barreto, A., 2022. 'SimulationBased Search,' in Proc. Int. Cong. Math, Vol. 6, pp. 48004819.

[SlL91] Slotine, J.-J. E., and Li, W., Applied Nonlinear Control, Prentice-Hall, Englewood Cliffs, N. J.

[TeG96] Tesauro, G., and Galperin, G. R., 1996. 'On-Line Policy Improvement Using Monte Carlo Search,' NIPS, Denver, CO.

[Tes94] Tesauro, G. J., 1994. 'TD-Gammon, a SelfTeaching Backgammon Program, Achieves Master-Level Play,' Neural Computation, Vol. 6, pp. 215-219.

[Tes95] Tesauro, G. J., 1995. 'Temporal Difference Learning and TD-Gammon,' Communications of the ACM, Vol. 38, pp. 58-68.

[VVL13] Vrabie, D., Vamvoudakis, K. G., and Lewis, F. L., 2013. Optimal Adaptive Control and Differential Games by Reinforcement Learning Principles, The Institution of Engineering and Technology, London.

[XDS23] Xie, H., Dai, L., Sun, Z., and Xia, Y., 2023. 'Maximal Admissible Disturbance Constraint Set for TubeBased Model Predictive Control,' IEEE Trans. on Automatic Control, Vol. 68, pp. 6773-6780.

[WGP23] Weber, J., Giriyan, D., Parkar, D., Richa, A., and Bertsekas, D., 2023. 'Distributed Online Rollout for Multivehicle Routing in Unmapped Environments,' arXiv preprint arXiv:2305.11596v1.

[WaB10] Wang, Y., and Boyd, S., 2010. 'Fast Model Predictive Control Using Online Optimization,' IEEE Trans. on Control Systems Tech., Vol. 18, pp. 267-278.

[Wri19] Wright, S. J., 2019. 'Efficient Convex Optimization for Linear MPC,' Handbook of Model Predictive Control, pp. 287-303.

[YDR04] Yan, X., Diaconis, P., Rusmevichientong, P., and Van Roy, B., 2004. 'Solitaire: Man Versus Machine,' Advances in Neural Information Processing Systems, Vol. 17, pp. 1553-1560.