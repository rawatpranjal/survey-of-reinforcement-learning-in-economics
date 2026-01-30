## Rollout, Policy Iteration, and Distributed Reinforcement Learning

Current Course at ASU (Research monograph to appear; partial draft at my website)

Dimitri P . Bertsekas

February 2020

- 1 Approximate Policy Iteration
- 2 Approximate Policy Iteration with Value and Policy Networks
- 3 Multiagent Rollout - Simplifying and Parallelizing the One-Step Lookahead
- 4 Multiprocessor Parallelization

## Outline

Tail problem approximation

u

k

u

k

u

k

u

k

Position 'values' Move 'probabilities'

## AlphaGo (2016) and AlphaZero (2017) U U 1 U 2

<!-- image -->

Choose the Aggregation and Disaggregation Probabilities

## AlphaZero (Google-Deep Mi AlphaZero (Google-Deep Mind) Plays m Use a Neural Network or Other Scheme Form the Aggre

At State x k Current state Plays different! Tail problem approximation u 1 k u 2 k u 3 k u 4 k u 5 k Self-Learnin Plays much better than all chess programs Cost α k g ( i, u, j ) Transition probabilities p ij ( u ) W p Controlled Markov Chain Evaluate Approximate Cos Use a Neural Scheme or Other Scheme Possibly Include 'Handcrafted' Features Alphazero has discovered a new way to play!

˜

Empty schedule LOOKAHE At State x k Current state x 0 ... MCTS Empty schedule LOOKAHEAD MINIMI Learned from scratch ... with 4 hours of training! AlphaZero (Google-Deep Mind) Plays much better than Evaluate Approximate Cost J µ ( F ( i ) ) of F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) ) : Vector of Features of i ˜ J µ ( F ( i ) ) : Feature-based architecture Features If ˜ J µ F ( i ) = s F /lscript ( i ) r /lscript Generate Features F ( i ) of Formulate Aggregate Problem Generate 'Impoved' Policy ˆ µ by 'Solving' the Aggregate Same algorithm learned multiple games (Go, Shogi) Aggregate costs r ∗ /lscript Cost function

J

˜

˜

it is a linear feature-based

0

(

i

) Cost function

J

1

Approximation in a space of basis functions Plays much

/lscript

=1

Scalar weights)

q

## ( ) ∑ ( r , . . . , r : all chess programs The AlphaZero methodology is based on several ideas:

k

- min u ,µ ,...,µ 1 s Cost α k g ( i, u, j ) Transition probabilities The fundamental DP idea of policy iteration/improvement.

+1

k

Evaluate Approximate Cost

J

u

˜

k

{

W

J

µ

(

F

k

(

+

k

/lscript

+1

i

)

)

Plays different! Approximate Value Function Player Feat ij

E

p

(

1

,...,µ

,µ

→

k

of

g

k

(

x

k

, u

k

, w

k

)

u

)

p

- min E -At State x k Current state x 0 ... MCTS Lookahead Min W p : Functions J ≥ ˆ J p with J ( x k ) 0 for all p -stable Controlled Markov Chain Evaluate Approximate Cost ˜ J µ Approximations with value and policy neural net approximations.
- Massive parallel computation.

ˆ

+

/lscript

-

1

Feature-based architecture Final Features

- Subspace Empty schedule LOOKAHEAD MINIMIZATION ROLL W p ′ : Functions J ≥ J p ′ with J ( x k ) → 0 for all p ′ -stabl F ( i ) = ( F 1 ( i ) , . . . , F s ( i ) : Vector of Features of i ˜ Lookahead approximations: Monte Carlo Tree Search.

J

J

µ

If

F

˜

J

|

W

µ

≥

J

+

F

{

=

,µ

x

(

+

i

)

F

=

(

:

i

)

)

, J

(

Rollout:

min E g ( x , u , w ) + k + VI converges to J from within W ( ) ∑ ( r 1 , . . . , r s : Scalar weights) We will aim to: Develop a methodology that relates to AlphaZero, but applies far more generally.

k

k

+1

s

/lscript

+

=1

/lscript

i

,...,µ

, u

)

}

k

+

)

-

(

r

1

t

/lscript

0 VI converges to

) = 0

it is a linear feature-based arc

Simulation with fixed policy Par

/lscript

-

1

from within

+1

T

k

=

J

+

/lscript

k

k

k

p

m

g

(

(

{

(

)

k

k

ˆ

k

x

(

m

ration Index

## Recall the α -Discounted Markovian Decision Problem k PI index k J µ k J ∗ 0 1 2 . . . Error Zone Width ( /epsilon1 +2 αδ ) / (1 -α ) 2 x = f ( x , u , w ) α k g

k

Tentative trajectory

, u

N

k

x

- Φ r = Π ( T ( λ ) µ (Φ r ) ) Π( J µ ) µ ( i ) ∈ arg min u ∈ U ( i ) Q µ ( i, u, r ) Subspace M = { Φ r | r ∈ /Rfractur m } Based on ˜ J µ ( i, r ) J µ k n ˜ ˜ Φ r = Π ( T µ (Φ r ) ) Π( J µ ) µ ( i ) ∈ arg min u ∈ U ( i ) Q µ ( i, u, r ) Subspace M = { Φ r | r ∈ /Rfractur m } Based on ˜ J µ ( i, r ) J µ k n ˜ ˜ J : µ ) µ ( i ) ∈ arg min u ∈ U ( i ) Q µ ( i, u, r ) ∈ /Rfractur m } Based on ˜ J µ ( i, r ) J µ k ˜ ˜ Variable Length Rollout Selective Depth Rollout Policy Limited Rollout Selective Depth Adaptive Simulation Po ntrol v ( j, v ) Cost = 0 State-Control Pairs Transitions under policy µ Evaluate Cost Function Aggregate Problem Approximation J µ ( i ) ˜ J µ ( i ) u 1 k u 2 k u k straint Relaxation Termination State straint Relaxation Infinite number of stages, and stationary system and cost System xk + 1 = f ( xk , uk , wk ) with state, control, and random disturbance.

)

-

J

µ

3

3

(

x

k

, u

k

µ

, w

k

k

+1

k

k

k

<!-- image -->

1

u

2

Terminal Cost Fun

- Good approximation Poor Approximation σ ( ξ ) = ln(1 + e ) max { 0 , ξ } ˜ J ( x ) Good approximation Poor Approximation max { 0 , ξ } ˜ J ( x ) Poor Approximation σ ( ξ ) = ln(1 + e ξ ) u ˜ Q k ( x k , u ) Q k ( x k , u ) u k ˜ u k Q k ( x k , u ) mited Rollout Selective Depth Adaptive Simulation Policy Tail problem approximation u 1 u 2 u 3 u Aggregate Problem Approximation J µ ( i ) J µ ( i ) u k u k u k u straint Relaxation φ j/lscript Cost of stage k : α k g ( xk , µ ( xk ) , wk ) ; α ∈ ( 0 , 1 ] is the discount factor. Cost of policy µ

T

˜

(

λ

)

˜

˜

- min u ∈ U ( i ) ∑ j =1 p ij ( u ) ( g ( i, u, j ) + J ( j ) ) Computation of J : ξ min u ∈ U ( i ) ∑ j =1 p ij ( u ) ( g ( i, u, j ) + J ( j ) ) Computation of σ ( ξ g ( i, u, j ) + J ( j ) ) Computation of J : riable Length Rollout Selective Depth Rollout Policy µ Adaptive Simulation ˜ 1 2 3 4 u 5 Tail problem approximation u Policies µ that map states to controls, with µ ( x ) ∈ U ( x ) for all x and k .

µ

States

1

k

k

u

k

k

Self-Learni

Approximation

k

k

k

i

k

Q

u

˜

) = ln(1 +

5

k

k

(

x

e

ξ

)

Self-Learni

k

, u

)

k

4

-

J

k

˜

with 4 hours of training! Curren

States

i

+2

k

+1

m

+2

) (

m

<!-- formula-not-decoded -->

AlphaZero (Google-Deep Mind) Play

u

)

c

+2)-Solut

1

) Neural Network

u

u

2

,

1

˜

) (

, . . . ,

m

+2)-Solution

u

˜

, u

m

+1

, u

m

<!-- formula-not-decoded -->

age 1 Stage 2 Stage 3 Stage

˜

u

1

f States (˜

2

,

u

Set of States (˜

) Neural Network

. . , u

Set of States

N

) Current

1

c

Heuristic Cost

) Neural Network

N N

u

) Set of States (˜

1

u

,

u

2

˜

Heuristic 'Future' System

(

, u

N

2

x

k

+1

, u

=

3

1)

f

)

k

k

(

x

) Set of States (˜

1

Set of States (˜

u

1

c

(

N

)

N

(

) Current

N

1)

k k

+1

-Solution (˜

N

u

) Current

1

, . . . ,

At State

m

m

)

x

Set of States

m

˜

u

, . . . ,

Plays different! Approximate Value Function Player Feat

k

0

AlphaZero (Google-Deep Mind) Plays much better than all computer progr

u

u

1

, . . . , u

-Solution (˜

1

u

= (

, . . . ,

Current state

x

1

˜

u

-Solution (˜

u

-

-

k

way to play! Base Policy Evaluation ment

## Policy Iteration Algorithm Alphazero has discovered a new way t Alphazero has discovered a new way to play! Base Poli One-Step Lookahead Policy Improvement Alphazero has discovered a n Alphazero has discovered a new way to play! Base Alphazero has discovered a new way to play! Base Policy

One-Step Lookahead Policy Improvement

One-Step Lookahead Policy Impro zero has discovered a new way to play! Base Policy Evaluation

Alphazero has discovered a new way to play! Base Policy Evaluatio

Alphazero has discovered a new way to play! Base Policy E

One-Step Lookahead Policy Improvement ookahead Policy Improvement

One-Step Lookahead Policy Improvement red a new way to play! Base Policy Evaluation

Improvement

One-Step Lookahead Policy Improvement ent Rollout Policy ˜

(

) Current State x, r

Rollout Policy ˜

tate x µ

Rollout Policy ˜

One-Step Lookahead Policy Improvement

Rollout Policy ˜

µ

Rando

Policy Evaluation Policy Improvement Rollout Policy ne-Step Lookahead Policy Improvement

x µ

Policy Evaluation Policy Improve

Base Policy

Base

Base P

µ

µ

Rollout

µ

) Current State

Randomized

Rollout Policy ˜

µ

Randomiz

µ

x

Randomize

µ

*

R

Approximate Policy Evaluation Approximate Policy I

valuation Approximate Policy Improvement olicy Evaluation Policy Improvement Rollout Policy ˜

J

<!-- image -->

*

*

Approximate Policy Evaluation Appr

Value Network Policy Network

J

˜

Bellman Eq. with

Approximate Policy Evaluati

Bellman Eq

State-Control Pairs instead of

Bellman Eq. with

J

Approximate Policy Evaluation Approximate Poli

J

St

µ

instead of

J

## y Network ˜ J State-Control Pairs Data-Trained Classifier Approximate Policy Improvement u = ˜ µ ( x, r ) Current State x µ Rollout Policy ˜ µ Randomized Policy Evaluation Policy Improvement Rollout Policy ˜ µ Fundamental policy improvement property

k

˜

J

J

µ

u

instead of

J

J

*

˜

Base Policy

µ

Approximate Policy Evaluation Approximate Policy Improvement x, r

Approximate Policy Evaluation Approximate Policy Impro

µ

Approximate Policy Evaluation Approximate Policy Impr

Bellman Eq. with

= ˜

µ

(

Value Network Policy Network

Value Network Policy Network

Value Network Policy Netw

Classifier

˜

State-Control Pairs Data-Traine

J

˜

State-Control Pairs Dat

<!-- formula-not-decoded -->

State-Control Pairs Data-Trained

Classifier

Value Network Policy Network

Value Network Policy Network

Classifier

J

Classifier

Classifier

Bellman Eq. with

J

˜

Approximate Policy Evaluation Approximate Policy Improvement

µ

instead of

*

## There are many variants of policy iteration

Value Network Policy Network ˜ J State-Control Pairs Data-Trained assifier Approximate Policy Evaluation Approximate Policy Improvement Optimistic, multistep, Q-learning versions, etc.

Value Network Policy Network

Classifier

˜

J

State-Control Pairs Data-Trained

OUR FOCUS: APPROXIMATE VERSIONS

J

J

State-Control

State-Control Pairs D

J

ith

˜

µ

(

J

µ

instead of

J

*

hazero has discovered a new way to play! Base Policy Evaluation

u

## Approximate Policy Iteration u = ˜ µ ( x, r ) Current State x µ Rollout Policy ˜ µ u = ˜ µ ( x, r ) Current State u = ˜ µ ( x, r ) Current State x µ u = ˜ µ ( x, r ) Current State Bellman Eq. with J µ instead of J * Bellman Eq. with

Lookahead Policy Improvement

Approximate Policy Evaluation Approximate Policy Improvem

Rollout Policy ˜

) Current State

u

= ˜

µ

x µ

(

x, r

) Current State

µ

x µ

Rollout Pol x µ

Rollout Policy ˜

µ

Approximate Policy Evaluation Approximate Poli

Approximate Policy Evaluation Value Network

µ

(

x, r

Approximate Policy Evaluation Approximate Policy Im

= ˜

Alphazero has discovered a new way to play! Base Policy E

Approximate Policy Evaluation Value Network Approximate Policy

Value Network Policy Network lassifier

x, r

) Current State

One-Step Lookahead Policy Improvement

u

= ˜

µ

(

Approximate Policy Evaluation Approxim

State-Control Pairs Data-Trained

µ

Approximate Policy Evaluation Value Network Approxima

Rollout Policy ˜

x µ

Rollout Policy ˜

<!-- image -->

α

x

) }

)

f

x, u, w

(

J

˜

## System: x k +1 = 2 x k + u k Control constraint: | u k | ≤ 1 Approximate Q-Factor ˜ Q ( x, u ) At x Methodological issues to deal with for challenging large-scale problems

## min u ∈ U ( x ) E w { g ( x, u, w ) + ( Approximate Q-Factor ˜ Q ( x, u ) At min u ∈ U ( x ) E w { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) ) } Improvement Policy Network Policy improvement property holds approximately

- Cost per stage: x 2 k + u 2 k System: x k +1 = 2 x k + u k Control constraint: | u k | ≤ 1 Cost per stage: x 2 k + u 2 k Theoretical issues: Error bounds, convergence guarantees, sampling efficiency, etc.
- Target Tube 0 k Sample Q-Factors ( /lscript -1)-Stages State x k + /lscript = 0 Complete Tours Current Partial Tour Next Cities Next States 0 k -Target Tube 0 k Sample Q-Factors ( /lscript -1)-Stages State x k + /lscript = 0 No guarantee of success: We just try different schemes based on theoretical understanding, intuition, experience ... hopefully something will work.
- { X 0 , X 1 , . . . , X N } must be reachable Largest reachable tube x 0 Control u k ( /lscript -1)-Stages Base Heuristic Minimization { X 0 , X 1 , . . . , X N } must be reachable Largest reachable tube x Control u ( /lscript 1)-Stages Base Heuristic Minimization Implementation choices: What to approximate, how to sample, how to train, on-line vs off-line, model-free vs model-based, etc.

Complete Tours Current Partial Tour Next Cities Next States

1

,n

1

,n

2

,n

2

,n

3

,n

3

,n

+1

Stages

Stage

+

+

R

+

R

Q

Q

R

Q

, . . . , k

k

k

-

Base Heuristic Minimization Possible Path Q 1 ,n + R 1 ,n Q 2 ,n + R 2 ,n Q 3 ,n + R 3 ,n Stage k Stages k +1 , . . . , k + /lscript OUR FOCUS: DISTRIBUTED (ASYNCHRONOUS) COMPUTATION

1

+

/lscript

-

µ

µ

1

## About this Talk

## We will focus on two types of distributed computation schemes

- Multiagent parallelization: Deal with large control spaces, e.g., controls with multiple components

<!-- formula-not-decoded -->

- Multiprocessor parallelization: Deal with large state spaces through partitioning, and distributed training of multiple value and policy networks (one per set of the state space partition).

## References

- Distributed asynchronous value iteration papers (DPB, 1982-83), Parallel and Distributed Computation book (DPB and Tsitsiklis, 1989).
- Distributed asynchronous policy iteration and Q-learning papers (Williams and Baird, 1993, DPB and Yu, 2010-14).
- Multiagent rollout paper (DPB, 2019).
- Partitioned rollout and policy iteration for POMDP paper (Bhattacharya, Badyal, Wheeler, Gil, DPB, 2020).

(

## Approximation in Value Space: From Values ˜ J ( x ) to a Policy ˜ µ ( x ) min u ∈ U ( x ) E w g ( x, u, w ) + α ˜ J f ( x, u, w

in rminal Cost Approximation Policy

Cost Approximation Policy

E

{

x

)

w

g

-Factor

(

x, u, w

˜

ut Approximate

Q

(

x, u

Approximate Q-Factor oximate

k

k

+

x

2

u

Min Approximation

System:

k

g Software Critic Software

x

N

/lscript

1

Target Tube 0

| ≤

|

)

<!-- image -->

are Critic Software

2

k

}

k

u

Control constraint:

must be reachable Largest reacha

1)-Stages Base Heuristic Minimiz

1)-Stages Base Heuristic Minimization

k

{·}

Control

x

k

+

k

u

(

/lscript

Control constraint:

u

k

x

Cost per stage:

u

+

-

k

1)-Stages Base Heuristic Minimization

+

u

2

x

0

Target Tube 0 k Sample Q-Factors ( /lscript -1)-Stages State x k + /lscript = 0 Complete Tours Current Partial Tour Next Cities Next States System: x k +1 = 2 | | ≤ Cost per stage: x 2 k + u 2 k { X 0 , X 1 , . . . , X N } must be reachable Largest reachable tube must be reachable Largest reachable tube Complete Tours Current Partial Tour Next Cities N Truncated Rollout Approximate Truncated R ties Corresponding to Open x ij i j z j = 0 or 1 Open Close Null rresponding to Open x ij i j z j = 0 or 1 Open Close Null At state x , use ˜ J (in place of J ∗ ) in Bellman's Eq. to obtain a control ˜ u = ˜ µ ( x ) .

k

+

/lscript

1

k

+

/lscript

1

u

)

k

Q

,n

k

2

,n

1

+

R

Sample Q-Factors (

1

+

R

/lscript

Q

3

,n

+

+ ˜

-

1)-Stages St

Stage

2

,n

R

3

,n

k

Stages

k

Partial Folding Software Critic Software

1)-Stages Base Heuristic Minimization

1

2

,n

Q

k

Stages

+1

, . . . , k

+

/lscript

1

Base Heuristic Minimization Possible Path

1)-Stages State

x

}

1)-Stages State

Clients

x

k

-

+

/lscript

= 0

}

/lscript

Facilities Corresponding to Ope

Simulation Nearest Neighbor Heuristic Move to th

## Q 1 ,n + R 1 ,n Q 2 ,n + R 2 ,n Q 3 ,n + R 3 ,n Stage k Base Heuristic Minimization Possible Path { X 0 , X 1 , . . . , X N } must be reachable Largest reachable tube x 0 Control u k ( /lscript -1)-Stages Base Heuristic Minimization Target Tube 0 k Sample Q-Factors ( /lscript x 0 Control u k ( /lscript -Target Tube 0 k Sample Q-Factors ( /lscript -k Sample Q-Factors ( /lscript -1)-Stages State x k + /lscript = 0 s Current Partial Tour Next Cities Next States min ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + -∑ i = k +1 g i ( x i , µ i ( x i ) , w i + ˜ J k + /lscript ( x k + /lscript ) -1 E { g k ( x k , u k , w k ) + -∑ i = k +1 g i ( x i , µ i ( x i ) , w i ) J k + /lscript ( x k + /lscript ) THE THREE APPROXIMATIONS: How to construct ˜ J .

Complete Tours Current Partial Tour Next Cities Next States

k

n

s

+2

h

-

Simulation Nearest Neighbor Heuristic Move to the Right Possible

k

+

= 0

- Path Complete Tours Current Partial Tour Next Cities Next States R 2 ,n Q 3 ,n + R 3 ,n Stage k Stages k +1 , . . . , k + /lscript 1 Path k +1 b k +2 Policy µ m Steps 1 2 3 Policy µ m Steps 1 2 3 How to simplify E {·} operation.

,n

˜ J k +1 ( x k +1 ) = min u k +1 ∈ U k +1 ( x k +1 ) E { g k +1 ( x k +1 , u k +1 , w k +1 ) Q 1 ,n + R 1 ,n Q 2 ,n + R 2 ,n Q 3 ,n + R 3 ,n -Base Heuristic Minimization Possible Path Simulation Nearest Neighbor Heuristic Move to the Right Possible Q 1 ,n + R 1 ,n Q 2 ,n + R 2 Base Heuristic Minimization Possible Path Simulation Nearest Neighbor Heuristic Move to the Right Possible Path Minimization Possible Path rest Neighbor Heuristic Move to the Right Possible ˜ J k +1 ( x k +1 ) = min u k +1 ∈ U k +1 ( x k +1 ) E { g k +1 ( x k +1 , u k +1 , w k min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k Approximation Truncated Rollout Policy µ m Steps Φ r ∗ λ rtificial Terminal to Terminal Cost g N ( x N ) i k b k i k +1 b k +1 i k +2 u k u k +1 u k +2 imation Truncated Rollout Policy µ m Steps Φ r ∗ λ l Terminal to Terminal Cost g N ( x N ) i k b k i k +1 b k +1 i k +2 u k u k +1 u k +2 How to simplify min operation. Each of the three approximations can be designed almost independently of the others, leading to a large variety of methods.

z

Stage

k

+1

m Observer Controller Belief Estimator erver Controller Belief Estimator Bertsekas

k

Stages

,n

3

Stage

R

+

-

+ ˜

k

b

k

+1

, . . . , k

k

Stages

+

/lscript

Belief States

1

, u

f

x

(

g

b

k

k

+1

+2

+1

+1

k

k

k

N

N

,n

J

with Cost

g

+1

(

x

)

+1

, w

)

b

k

, . . . , k

+2

k

+

Policy

+1

+2

f

k

+1

/lscript

-

µ m

) +

+

Step

(

x

+1

k

J

k

)

,

+ ˜

(

x

3

k

Q

b

1

-

+

2

+1

/lscript

-

Optimal trajectory

## Parametric Approximation in Policy Space One-Step Lookahead Policy Improvement ˜ µ

Policy Evaluation Policy Improvement Rollout Policy ˜

new way to play! Base Policy Evaluation ovement ˜

µ

Base

Uncertainty System Environment Cost C

Alphazero has discovered a new way t

µ

ironment Cost Control Current State ement Rollout Policy ˜

φ

β

2

, u

-

β

)

x

(

,γ

s

)

max

Rollout Policy ˜

3

<!-- image -->

) +

4

(

,β

β

c

y

β

3

β

Randomiz

0

, ξ

(a

4

{

}

Li

Uncertainty System Environment Cost C

β

φ

(a) (b)

3

'Deceptive' Lo

1

,β

x

Linear Unit

,β

4

,γ

(

)

)

Linear Unit Rectifier

,γ

0

max

,γ

b

{

2

)

Approximate Policy Evaluation Approximate Policy Impr

i

-

-

u

+1 Stages Optimal trajectory

(

{

φ

x γ

s

µ

'Deceptive' Low Cost ry

x

-∗ i ∗ i High Cost Suboptimal u ′ 'Deceptive' Low Cost u O /lscript +1 Stages Optimal trajectory Slope γ β High Cost Suboptimal u ′ 'Deceptive' Low Cost u Optimal traj i y ∗ i x γ ( x -β 3 ) γ ( x -β 4 ) + -max { 0 , ξ } Slope γ β (Assigns x to u ) . TRUNCATED ROLLOUT with BASE Optimization and training over a parametric family of policies µ ( x , r ) , where r is a parameter (e.g., a neural net).

)

γ β

(

c

γ

i

R

x

min

i

R

}

, ξ

β

min

i

y

β

3

max

High Cost Suboptimal

β

˜

2

Value Network Policy Network

y

(

R

) =

max

k

y

i

+1

min

k

φ

(

Li

β

i

y

'Deceptive' Lo

-

u

(

′

k

b

(

y

min

2

k

i

+1

+1

k

i

)

J

+1 Stages Optimal trajectory tion Approximate Policy Improvement

)

E g

k

c

i

+1

y

(

x

k

b

i

+1

)

2

, u

˜

k

+1

x

-

/lscript

, w

k

i

y

i

+1

∗

)

y

max

i

(

c

i

y

∗

i

b

i

∗

i

)

2

u

R

i

∈

min

∗

U

i

+1

y

∗

i

x

+1

max

)

i

2

E

,γ

{

y

i

∗

+1

β

x

+1

g

k

+ ˜

J

k

(

+2

4

k

f

k

-

(

x

Slope

β

1

1

## From Value Approx. ˜ J ( x ) to Policy µ ( x , r ) r ) Current State x µ Rollout Policy ˜ µ pproximate Policy Evaluation Approximate Policy Improvement ˜ µ ( x, r ) Current State x µ Rollout Policy ˜ µ ( x, r ) Current State x µ Rollout Policy ˜ µ One-Step Lookahead Policy Improvement ˜ µ Evaluation Policy Improvement Rollout Policy ˜ µ Base Policy µ Alphazero has discovered a new way to play! Base Policy Evaluation One-Step Lookahead Policy Improvement ˜ µ One-Step Lookahead Policy Improvemen

signs

(

x, r oximate Policy Evaluation Approximate Policy Improvement

Policy Evaluation Policy Improvement Rollout Policy ˜

µ

Policy Evaluation Policy Improvement proximate Policy Evaluation Approximate Policy Improvement

x

to

µ

alue Network Policy Network e Network Policy Network

Assigns

Rollout Policy Approximation in Value Space ate Policy Evaluation Approximate Policy Improvement

= ˜

) Current State

Approximation in Value Space

µ

(

x, r etwork Policy Network ˜

Multistep Lookahead alue Network Policy Network

kahead ion in Policy Space

*

u

<!-- image -->

ace instead of

J

µ

e Q-Factor

, u

= ˜

µ

(

x, r

) Current State x µ

Rollout Policy ˜

Bellman Eq. TRUNCATED ROLLOUT with BASE

<!-- image -->

˜

Q

J

(

µ

Randomized

u

=

µ

(

x, r

Base Policy

µ

) Pairs (

) Current State

x

s

, u

s

)

x µ

Roll

Bellman Eq. TRU

x, u ) At x µ instead of J * Bellman Eq. TRUNCATED ROLLOUT with BASE u = ˜ µ ( x, r ) Current State x µ Rollout Policy ˜ µ Randomized J µ instead of J * x is classified as type u ⇐⇒ at state x we apply control u

u

## min ∈ U ( x ) E w { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) ) } ) + ˜ POLICY µ proximate Policy Evaluation Approximate Policy Improvement J µ instead of J * Bellman Eq. TRUNCATED ROLLOUT with BASE POLICY µ POLICY µ Training the rollout policy as a classifier:

) At

x

- e Q-Factor ˜ Q ( x, u ) At x u, w α J ( f ( x, u, w ) ) } , u ) At x Approximate Policy Evaluation Approximate Policy Improvement ssigns x to u ) Approximate Policy Evaluation Approximate Policy Improvement Approximate Policy Evaluation Ap We generate a training set of sample pairs ( x s , u s ) , s = 1 , . . . , q , by one-step lookahead, i.e.,

1

alue Network Policy Network

= 2

x

+

u

ontrol constraint:

k

k

k

ge:

x

2

+

u

2

k

State-Control Pairs Data-Trained Classifier with

µ

<!-- formula-not-decoded -->

N

must be reachable Largest reachable tube

, X

}

- 0 k Sample Q-Factors ( /lscript -1)-Stages State x k + /lscript = 0 ours Current Partial Tour Next Cities Next States -Factors ( /lscript -1)-Stages State x k + /lscript = 0 Initial State Current State Approximation Truncated Rollout Using a Local Policy Network Policy Network ˜ J State-Control Pairs Data-Trained Classifier with µ Initial State Current State Approx a Local Policy Network Example: Introduce a parametric family of policies µ ( x , r ) of some form (e.g., a neural net). Then estimate r by least squares fit
- u k ( /lscript -1)-Stages Base Heuristic Minimization reachable Largest reachable tube s Base Heuristic Minimization ˜ J State-Control Pairs Data-Trained Classifier with µ itial State Current State Approximation Truncated Rollout Using Value Network Policy Network ˜ J State-Control Pairs Data-Traine Approximate the one-step lookahead policy using the training set.

artial Tour Next Cities Next States ate Space Partition

Q

n

+

R

R

2

,n

2

3

,n

+

ch Set Has a Local Value Network and a Local Policy Network tic Minimization Possible Path

n Possible Path

<!-- formula-not-decoded -->

State Space Partition

Nearest Neighbor Heuristic Move to the Right Possible

## From Policy µ to Value Approx. ˜ J Classifier with µ Value Network Policy Network

Initial State Rollout Using Local Policy

State-Control Pairs Data-Trained Classifier with

Terminal Cost Supplied bu Local Value

Initial State Truncated Rollout Using Local Policy

Truncated Rollout Approximate Base Po

Optimal Cost Terminal Cost Approximation

Each Set Has a Local Value Network and a Local P

Complete Folding Current

Terminal Cost Supplied by Local Value Network T

Truncated Rollo

= 0

j

<!-- image -->

st Terminal Cost Approximation Policy

Clients or

1

Open Clo

Facilities Corresponding to Open

Truncated Rollout Approximate

u

k

,µ

k

## olding Software Critic Software How to approximate J µ ( x ) ?

Truncated Rollout Approxima

x

ij i j z

g

i

x

## Partial Folding Software Critic Software Partial Folding Software Critic Software { k + /lscript -1 Rollout Approximate Truncated Rollout Approximate Base Policy Cost Policy µ defines a cost approximation ˜ J ≈ J µ through truncated simulation

min

+1

,...,µ

k

+

/lscript

1

Clients

Co

Complete Foldin

i

i

x

(

)

Facilities Corresponding to Open

Complete Folding Current Partial Folding

-

Facilities Corresponding to Open

(

i

=

∑

k

+1

- { b k Belief States b k +1 b k +2 Policy µ m Steps 1 2 3 For deterministic problems: Run µ from x once and accumulate stage costs.

u

k

,µ

B

(

b, u, z

)

h

(

Clients

u

b

)

k

i

=

k

+1

Artificial Terminal to Terminal Cost

Belief States

x

, w ij

(

x

N

i j z

)

)

i

k

+ ˜

j

b

k

J

= 0

x

i

k

+

(

x

k

ij or

k

i

k

+

/lscript

1

-

+1

b

1

k

- min u k ,µ k +1 ,...,µ k + /lscript -1 E g k ( x k , u k , w k ) + ∑ i = k +1 b k Belief States b k +1 b k +2 Policy µ m Steps 1 2 min u k ,µ k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ i = k +1 g i ( x i , µ i ( x i ) , w i k +1 k Optimal Cost Approximation Truncated Rollout Policy µ m Steps Φ r ∗ λ acilities Corresponding to Open x ij i j z j = 0 or 1 Open Close Null min k +1 ,...,µ k + /lscript -1 E { g k ( x k , u k , w k ) + k + /lscript -1 ∑ g i ( x i , µ i ( x i ) , w i ) + ˜ J k + /lscript ( x k + /lscript ) } For stochastic problems: Run µ from x many times and Monte Carlo average. Use truncation: Simulate µ for a limited number of stages, and neglect the costs of the remaining stages or add some cost approximation at the end to compensate. Bertsekas Reinforcement Learning 13 / 28

Steps 1 2 3

b

b

Policy

µ m

g

+2

N

/lscript

i

E

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

, µ

azero has discovered a new way to play! Base Policy Evaluation imate Policy Evaluation Approximate Policy Improvement

Approximate Policy Evaluation Approximate Policy Improvem

(

x, u, w

Approximate Policy Evaluation Approxi

= 2

)

Approximate Q-Factor

<!-- image -->

min

g

g

x, u, w

(

J

α

min

Cost per stage:

˜

Approximate Policy Evaluation Approximate Policy Im

Approximate Policy Evaluation Approximate Po

Lookahead Policy Improvement olicy Rollout Policy Approximation in Value Space

s

ep or Multistep Lookahead

µ

˜

(

) Current State x, r

x

to

u

)

One-Step Lookahead Policy Improvement

Approximate Q-Factor roximate Policy Evaluation Approximate Policy Improvement

imation in Policy Space

Network Policy Network Value Data imate Q-Factor

e-Control Pairs Data-Trained Classifier with e Network Policy Network

˜

x, u, w

f

(

) +

x, u, w

α

J

˜

)

f

(

x, u, w

) }

)

k

Cont

f

(

x, u, w

)

x

+

u

k

- u ∈ U ( x ) E w { ( ) } ˜ Q ( x, u ) At x = 2 x k + u k Control constraint: min u ∈ U ( x ) E w { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) ) } ˜ Q ( x, u ) At x u ∈ U ( x ) w { ( ) } Approximate Q-Factor ˜ Q ( x, u ) At x System: u Control constraint: | u k | ≤ 1 u ∈ U ( x ) w { ( ) } Approximate Q-Factor ˜ Q ( x, u ) At x System: x k +1 = 2 x k + u k Control constraint: min u ∈ U ( x ) E w { g ( x, u, w ) + α ˜ J ( f ( x, u, w ) ) } imate Q-Factor System: x k +1 = 2 x k + u k Control constraint: | u k | ≤ Cost per stage: x 2 k + u 2 k { X 0 , X 1 , . . . , X N } must be reachable Largest reachabl x 0 Control u k ( /lscript -Target Tube 0 k Sample Q-Factors ( u = ˜ µ ( x, r Approximate Policy Evaluation Approximate Policy Improvement Value Network Policy Network State Current State Approximation Truncated Rollout Using cy Network Policy improvement property: In the idealized case (no approximations), J ˜ µ ( x ) ≤ J µ ( x ) , for all x

:

) }

˜

Q

(

x, u

) At

x

+

u

k

1)-Stages Base Heuristic Minimizat

u

1)-Stages Stat

k

1

|

| ≤

/lscript

(

min

E

) +

Approximate Q-Factor

= 2

x

x

k

x, u

k

+1

) At

System:

x

Approximate Q-Factor

k

k

+1

- 0 k -Target Tube 0 k Sample Q-Factors ( /lscript 1)-Stages State x x 0 Control u k ( /lscript -1)-Stages Base Heuristic Minimization Target Tube 0 k Sample Q-Factors ( /lscript -1)-Stages State x k + /lscript = 0 Target Tube 0 k Sample Q-Factors ( /lscript -1)-Stages State x k + /lscript = trol u k ( /lscript -1)-Stages Base Heuristic Minimization Complete Tours Current Partial Tour Next Cities N Base Heuristic Minimization Possible Path nal Cost Supplied by Local Value Network Terminal State Most RL algorithms, including Alphazero, use variants of the above scheme.

1

,n

Q

- | | ≤ Cost per stage: x 2 k + u 2 k { X 0 , X 1 , . . . , X N } must be reachable Largest reachable tube x Control u ( /lscript 1)-Stages Base Heuristic Minimization System: x k +1 = 2 x k + u k Control constraint: | u k | ≤ 1 Cost per stage: x 2 k + u 2 k { X 0 , X 1 , . . . , X N } must be reachable Largest reachable tube Cost per stage: x 2 k + u 2 k { X 0 , X 1 , . . . , X N } must be reachable Largest reachable tube x 0 Control u k ( /lscript -1)-Stages Base Heuristic Minimization Cost per stage: x 2 k + u 2 k { X 0 , X 1 , . . . , X N } must be reachable Largest reachable tube x 0 Control u k ( /lscript -1)-Stages Base Heuristic Minimization x k +1 = 2 x k + u k Control constraint: | u k | ≤ 1 er stage: x 2 k + u 2 k 1 , . . . , X N } must be reachable Largest reachable tube { X 0 , X 1 , . . . , X N } must be reachable Largest reacha x 0 Control u k ( /lscript -1)-Stages Base Heuristic Minimiz Target Tube 0 k Sample Q-Factors ( /lscript -1)-Stages St -Complete Tours Current Partial Tour Next Cities Nex Q 1 ,n + R 1 ,n Q 2 ,n + R 2 ,n Q 3 ,n + R 3 ,n Stage k Stages k + pace Partition et Has a Local Value Network and a Local Policy Network With approximations, policy improvement is approximate (within an error bound). There are many variants of this scheme: Optimistic policy iteration, Q-learning, temporal differences, etc.

k

+

Simulation Nearest Neighbor Heuristic Move to the

, . . . , k

3

,n

3

R

+

u

+

k

/lscript

,n

Stage

+1

U

1

k

+1

k

x

k

Stages

+1

)

k

-

+1

, . . . , k

+

/lscript

-

1

k

Simulation Nearest Neighbor Heuristic Move to the Right Possible

k

Stage

k

Stages

+1

,n

Q

1

,n

3

Q

,n

2

,n

+

3

R

,n

2

Simulation Nearest Neighbor Heuristic Move to the Right Pos

(

∈

+1

)

- -k + /lscript = 0 Complete Tours Current Partial Tour Next Cities Next States Target Tube 0 k Sample Q-Factors ( /lscript -1)-Stages State x k + /lscript = 0 Complete Tours Current Partial Tour Next Cities Next States Complete Tours Current Partial Tour Next Cities Next States Q 1 ,n + R Q + R Q + R Stage k Stages k +1 , . . . , k + /lscript 1 Complete Tours Current Partial Tour Next Cities Next States Q 1 ,n + R 1 ,n Q 2 ,n + R 2 ,n Q 3 ,n + R 3 ,n Stage k Stages k +1 , . . . , k Tube 0 k Sample Q-Factors ( /lscript -1)-Stages State x k + /lscript = 0 ete Tours Current Partial Tour Next Cities Next States Q 1 ,n + R 1 ,n Q 2 ,n + R 2 ,n Q 3 ,n + R 3 ,n Stage k Stages Base Heuristic Minimization Possible Path Simulation Nearest Neighbor Heuristic Move to the Path Some variants are highly optimistic, i.e., use very little data between value updates and policy updates.

1

,n

3

,n

3

,n

2

,n

Q

1

,n

+

2

,n

R

+

u

k

1

(

x, u, w

) +

α

J

R

x

Q

+

R

f

(

g

(

x, u, w

E

)

x

k

˜

Base Heuristic Minimization Possible Path Q 1 ,n + R 1 ,n Q 2 ,n + R 2 ,n Q 3 ,n + R 3 ,n Stage k Stages k +1 , . . . , k + /lscript -1 Base Heuristic Minimization Possible Path Base Heuristic Minimization Possible Path Simulation Nearest Neighbor Heuristic Move to the Right Possible Base Heuristic Minimization Possible Path Q 2 ,n + R 2 ,n -euristic Minimization Possible Path Path ˜ J k +1 ( x k +1 ) = min E { g k +1 ( x k +1 , u k +1 , w HOW DO WE USE PARALELLIZATION IN ROLLOUT AND APPROXIMATE PI?

## Four Possible Types of Parallelization

<!-- image -->

## A Spiders-and-Fly Example (or Search-and-Rescue)

<!-- image -->

15 spiders move along 4 directions ( ≤ 1 unit) w. perfect observation; fly moves randomly

- Objective is to catch the fly in minimum time.
- One-step lookahead and rollout are impossible: ≈ 5 15 Q-factors.
- We reformulate one-step lookahead but maintain the cost improvement property:
- glyph[trianglerightsld] Spiders move one-at-a-time with knowledge of other spiders' and fly's positions.
- glyph[trianglerightsld] The control is broken down into a sequence of 15 spider moves (5 · 15 = 75 Q-factors).

e con

Transit om Cost

lIng ol aclion

co nf

Ck+1)

k, Uk, Wk)

tory

~kT.

## Trading off Control and State Complexity (NDP book, 1996) Policy Q-Factor Evaluation Iteration Index k PI index k J µ J ∗ 0 1 2 . . . Error Zone Width ( /epsilon1 +2

P

k

1

, u

k

x

, u

x

k

, u

1

k

(

x

˜

x

k

k

k

, u

N

)

x

k

N

Π(

r

)

Π(

J

αδ

=

, u

E

(

of Current policy

f

x

k

k

k

)

Tentative trajecto

, w

k

x

N

x

N

k

) Cost = 0 State-C

, w

k

)

k

µ

)

µ

k

<!-- image -->

Control

v

(

j, v

Stage

)

Q

˜

µ

(

i, u, r

)

k

Future Stges

Φ

r

= Π

T

Termination State

)

(

i

) Cost = 0 State-Control Pairs Transitions under policy

Variable Length Rollout Selective

µ

)

arg min

u

∈

Based on

x

′

∈

U

˜

µ

J

µ

k

k

(Φ

Aggregate Problem Approximatio

N

µ

(

Approxim i, r

)

J

k

k

k

x

µ

) Monte Carlo tree sea

J

µ

x

)

µ

(

i

)

arg min

u

∈

U

(

i

## r | r ∈ /Rfractur m } Based on ˜ J µ ( i, r ) J µ k ˜ ˜ Subspace M = { Φ r | r n Control u k Cost g k ( x k , u k ) x k x k +1 straint Relaxation An equivalent reformulation - 'Unfolding" the control action

ij u, j

min

m

N

(

- ( u ) ( g ( i, u, j ) + J ( j ) ) Computation of J : ation Poor Approximation σ ( ξ ) = ln(1 + e ξ ) ) Monte Carlo tree search First Step 'Future' min u ∈ U ( i ) ∑ j =1 p ij ( u ) ( g ( i, u, j ) + ˜ J ( j ) ) Co Good approximation Poor Approxima max { 0 , ξ } ˜ J ( x ) Cost 0 Cost g ( i, u, j Limited Rollout Selective Depth u ˜ Q k ( x k , u ) Q k ( x k , u ) u ˜ u Q ( x 0 x k x 1 k +1 x 2 k +1 x 3 k Variable Length Rollout Selective Depth Rollout Policy µ Adaptive Simula Limited Rollout Selective Depth Adaptive Simulation Policy ˜ u k u k ˜ x k +1 x k +1 ˜ x N x N x ′ N Φ r = Π ( T ( λ ) µ (Φ r ) ) Π( J µ ) µ ( i ) ∈ arg min u ∈ U ( i ) ˜ Q µ ( i, u, r ) Subspace M = { Φ r | r ∈ /Rfractur m } Based on ˜ J µ ( n ˜ Tail problem approximation u 1 u φ j/lscript Aggregate Problem Approximation J µ ( i ) ˜ J µ ( i ) u 1 k u straint Relaxation The control space is simplified at the expense of m -1 additional layers of states, and corresponding m -1 cost functions J 1 ( xk , u 1 k ) , J 2 ( xk , u 1 k , u 2 k ) , . . . , J m -1 ( xk , u 1 k , . . . , u m -1 k )

i, u, j

c

) +

(0)

c

(

J

k

x

∈ /Rfractur

N

(

)

)

j

c

(

k

+1

4

Computation

+1)

c

+1

k

State

(

N

1) P

- N Aggr. States Stage 1 Stage 2 Stage 3 Stage N -1 -Solutions (˜ u 1 , . . . , ˜ u m , u m +1 , u m +2 ) ( m +2)-Solution Set of States ( u , u ) Set of States ( u , u , u ) Feature Extraction Node Subset S 1 S N Aggr. States Stage 1 Sta Candidate ( m +2)-Solutions (˜ u 1 , . . . , ˜ u m , u m Set of States ( u 1 ) Set of States ( u 1 , u 2 ) Set Initial State 15 1 5 18 4 19 9 21 2 Stage 1 Stage 2 Stage 3 Stage N u ˜ Q k ( x k , u ) Q k ( x k , u ) u k ˜ u k Q k ( x k , u ) -˜ Q k ( x k , u ) x 0 x k x 1 k +1 x 2 k +1 x 3 k +1 x 4 k +1 States x N Base Heuristic i k States i k +1 Stat ∈ ∑ ( ) Good approximation Poor Approximation σ ( ξ ) = l max { 0 , ξ } ˜ J ( x ) Cost 0 Cost g ( i, u, j ) Monte Carlo tree search First St Learned from scratch ... with 4 h Tail problem approximation u 1 k u 2 k u 3 k u 4 k u 5 k Self-Le φ j/lscript Multiagent (one-component-at-a-time) rollout is just standard rollout for the reformulated problem. The increase in size of the state space does not adversely affect rollout.

Run the Heuristics From Each Candidate (

u

U

(

i

)

j

=1

p

ij

(

u

)

g

(

AlphaZero (Google-Deep Mind)

with 4 hours of training! C

- 1 2 1 2 3 s From Each Candidate ( m +2)-Solution (˜ u , . . . , ˜ u , u ) Initial State 15 1 5 18 4 19 9 21 25 8 12 13 Feature Extraction Learned from scratch ... The cost improvement property is maintained.

(

- 1 m m +1 Set of States (˜ u 1 , ˜ u 2 ) Neural Network u , . . . , u N ) Current m -Solution (˜ u , . . . , ˜ u ) Set of States (˜ u 1 ) Set of States (˜ u 1 , ˜ u 2 ) Neu Set of States u = ( u , . . . , u ) Current m -S Heuristic Cost Heuristic 'Futur Stage 1 Stage 2 Stage 3 Stage N N -1 c ( N ) c ( N -1) k k +1 Node Subset S 1 S N Aggr. States Stage 1 Stage 2 Stage 3 Candidate ( m +2)-Solutions (˜ u 1 , . . . , ˜ u m , u m +1 , u m +2 ) ( m Plays different! Approximate Val AlphaZero (Google-Deep Mind) Plays much better t Complexity reduction: The one-step lookahead branching factor is reduced from n m to nm , where n is the number of possible choices for each component u i k .

tic

1

N

-Solutions

u

1

, . . . , u

N

-

1

)

1

m

Set of States (

Heuristic Cost

Cost

G

(

u

m

-

N

) Heuristic

1

1

N

u

) Set of States (

) Set of States (

Belief State

, u

p

Controller

u

,

, u

m

µ

Con

-Solutions

2

u

= (

u

1

, . .

u

}

x

∈

1

Shortest path

## Time to catch the flies

- Base policy (each spider follows the shortest path): 85
- All-at-once rollout (125 move choices): 34
- One-at-a-time rollout (15 move choices): 34

## Demo 1

All at once

One at a time

## Multiagent Parallelization and Coordination Issues

<!-- image -->

- One-at-a-time rollout and all-at-once rollout produce different rollout policies. One may be better than the other.
- Exact policy iteration issues. One-at-a-time rollout used repeatedly (as in policy iteration) may stop short of the optimal.
- We speculate that in approximate policy iteration, one-at-a-time rollout will often perform about as well as all-at-once rollout.
- We can try to induce agent parallelization and asynchronism: Divide agents in 'weakly coupled groups" ... Require little or no coordination among groups.

## Group Coordination Issues

<!-- image -->

## Several interesting theoretical and algorithmic issues remain to be resolved

- How do we form groups? Use feature-based groupings?
- Frequency of communication?
- Aggregated coordination between groups?
- Distributed info processing?

cial nj

u

)

(

te

i

jn

States

p

(

Feature Extraction Mapping Feature Vector ator

r

)

i

(

φ

Approximator

′

ute to Queue 2

ute to Queue 2

u

(

p

(

u

n t p

)

)

Aggregate States Features

Special

Aggregate States Features

)

p

u

(

α

N

C

n t p nn

(

≤ · · · ≤

-

in ni

L

≤

1

0

) Linear Cost

C

L

α

k

k

2

(

p

α

∗

k

k

k

States

≤

α

N

-

3

≤

α

N

-

2

<!-- image -->

Route to Queue 2

(

u

· · · ≥

1

k

k

β

L

L

N

0

0

in

3

u

)

-

φ

p

≤

(

L

+

-

k

3

1

L

) L

i

≤

1

n ) λ ∗ λ µ λ h µ,λ ( n ) = ( λ µ -λ ) N µ ( n ) 1 -( n -1) Cost = 1 Cost = 2 u = 2 Cost = -10 µ ∗ ( i +1) µ µ p , p jk ( u ) ν k ( u ) , p ki ( u ) J ∗ ( p ) µ 1 µ 2 h λ ( n ) λ ∗ λ µ λ h µ,λ ( n ) = ( λ µ -λ ) N µ ( n ) n -1 -( n -1) Cost = 1 Cost = 2 u = 2 Cost = -10 µ ∗ ( i Nullspace Component Sequence Orthogonal Component x = ( C ′ Σ -1 C + β I ) -1 ( C ′ Σ -1 d + β ¯ x ) Nullspace Component Sequence Orthogona x = ( C ′ Σ -1 C + β I ) -1 ( C ′ Σ -1 1 -1 C I + A N -1 ( p ) α N -1 1 -1 C I + A N -1 ( p ) α N -1 0 I 1 p C J N -1 ( p ) L 0 L 0 + L 1 Partition the state space into several subsets and construct a separate policy and value approximation in each subset.

p C

C

+

k

1

)

µ

J

u

)

J

k

k

k

ki

µ

N

A

k

1

N

p

)

˜

-

3

N

2

µ

T

N

u

)

(

Markov Chain

- ulation error Solution of ˜ J µ = WT µ T µ 0 J J µ 0 = T µ 0 J µ 0 T 0 J J α N -2 α N -3 0 I 1 Use features to generate the partition.

)

, p

(1

) Bias Π

µ

( ˜

-

, µ

J

0

J

I

µ

2

Markov Chain

-

J

-

p

Slope

P

)

I

, . . .

α

N

-

1

C

p C

(

1

+

A

p C

-

=

3

(

- Simulation error Solution of ˜ J µ = WT µ ( ˜ J µ ) Bias Π Φ r µ µ µ 0 µ 0 µ 0 (1 -p ) C I + A N -1 ( p How do we implement truncated rollout and policy iteration with partitioning?

0

m

P

)

β

C

J

1

N

m

)

µ

(

M q nsition diagram and costs under policy

=

p

µ

jk

-

(

p

)

C

ν

I

(

u

)

+

, p

A

N

1

(

)

Transition diagram and costs under policy

J

∗

0

1

0

{

′

′

= arg min

L R R

Cost

1

t

= 2

1

N

t

0

-

(1

}

t

-

c

-

τ,t

1

}

Delay Actuator Estimator Syst

-

p

+1

k

s

= arg min

t

=1

pf

)

L

τ

=0

(

z

)

0

, µ

′

τ,t

)

′

r

t

=1

, . . .

τ

=0

ment

∈/Rfractur pL

µ

1

φ

(

i

r

′

L R R

s

∈/Rfractur

α/epsilon1 r

r

+1

z

)

α/epsilon1 r

k

(

pL

pf

= 2

L

)

Cost

p

-

(1

µ

S

N

α

M

t

D

(

r

φ

k

(

i

{

c

1 0

+

ν

E

j

ment nn

## An old and fairly obvious training idea:

- Assign one processor to each subset of the partition.
- Each processor uses a local value and a local policy approximation, and maintains asynchronous communication to other processors.
- Update values locally on each subset (policy evaluation by value iteration).
- Update policies locally on each subset (policy improvement, possibly using multiagent parallelization).
- Communicate asynchronously local values and policies to other processors.

## However:

- The obvious algorithm fails (for the lookup table representation case - a counterexample by Williams and Baird, 1993).
- The DPB-HJY algorithm, 2010, corrects this difficulty and proves convergence (assuming a lookup table representation for policies and cost functions).
- Admits extension to neural net approximations (some error bounds available).

ut Using Local Policy Network instead of

ut Using Local Policy Network

3

N

µ

J

α

OLICY

-

≤

ion

α

µ

-

≤ · · · ≤

## Value Network Policy Network J S Classifier with µ POLICY µ POLICY µ J µ instead of J * Bellman Eq. TRUNCATED RO POLICY µ J µ instead of J * Bellman Eq. TRUNCATED ROLLOUT with B u = ˜ µ ( x, r ) Current State x µ Rollout Policy ˜ µ Randomized Approximate Policy Evaluation Approximate Policy Improvement Value Network Policy Network ˜ J ( x, r ) = ∑ m /lscript =1 r /lscript φ /lscript ( x ) x S 1 S /lscript S m . . . r 1 r /lscript r m cal Value Network and a Local Policy Network Each Set Has a Local Value Network and a Local Policy Network cal Value Network and a Local Policy Network or 1 k Using the Base Policy a Local Value Network

ion

β

N

(

p

-

)

1

N

β

-

1

-

0

2

I

1

J

Base Policy

Randomized

Bellman Eq. TRU

## Approximate Policy Iteration with Local Value and Policy Networks Approximate Policy Evaluation App u = ˜ µ ( x, r ) Current State x µ Rollout Policy ˜ µ Randomized J µ instead of J * Bellman Eq. TRUNCATED ROLLOUT with B u = ˜ µ ( x, r ) Current State x µ Roll J µ instead of * u = ˜ µ ( x, r ) Current State x µ Rollout Policy ˜ µ u = ˜ µ ( x, r ) Current State x µ Rollout Policy ˜ µ Policy Evaluation Policy Improvement Rollout Policy ˜ µ J * Bellman Eq. TRUNCATED ROLLOUT with BASE POLICY µ Approximate Policy Evaluation Approximate P State Space Partition 1 -L 0

µ

˜

Approximate Policy Evaluation Approximate Policy Improveme

Value Network Policy Network pplied by Local Value Network

pplied by Local Value Network

≤

˜

J

α

Initial State Rollout Using Local Pol

Approximate Policy Evaluation A

Approximate Policy Evaluation Approximate Po

Bellman Eq. TRUNCATED ROLLOUT with BASE

C

L

State-Control Pairs Data-Trained Classifier

1

State-Control Pairs Data-Tr

Value Network Policy Network

Approximate Policy Evaluation Approximate Policy Improveme

˜

J

Terminal Cost Supplied bu Local V

-

0

J

˜

State-Control

State-Control Pairs Data-Tra

Initial State Truncated Rollout Using Local Pol

N

Initial State Rollout Using Local Policy Network p C

-

State Space Partition

1

(

p

R R

t

2

f

t r

z

k

<!-- image -->

)

1

C

α

t

Continue Terminate Instruction Accept

C

Initial State Rollout Using Local

0

Initial State Rollout Using Local Policy Networ

L

L

L

L

+

1

L

Terminal Cost Supplied bu Local

1

0

1

Terminal Cost Supplied by Local Value Networ

Each Set Has a Local Value Network and a Loc

Terminal Cost Supplied by Local Value Networ

-

1

+

(

p

A

)

N

L

-

- Each set has a Local Value Network and a Local Policy Network (1 -p ) C I + A N -1 ( p ) α N -1 0 I 1 p C Start with some base policy and a value network for each set.
- Terminal Cost Supplied by Local Value Network (1 -p ) L 0 pL 1 L R R 1 t Delay Actuator Estimator Sy ment Partitioning may be a good way to deal with adequate state space exploration.
- 1 -r µ k E { x k | I k } Φ k -1 Obtain a policy and a value network for the truncated rollout policy. Repeat.

u

k

1

µ

k

k

I

k

Φ

k

1

3

L

0

+

(

p

0

L

)

N

+

N

-

A

1

N

0

3

N

2

1

7000

6800

6600

6400

6200

## Distributed RL for POMDP (BBWGB paper, 2020)

pAPI-T (6 policy nets, 3 value nets)

pAPI-NT (6 policy nets)

API (1 policy net)

<!-- image -->

- 20 potentially damaged locations along a pipeline. 6000
- Damage of each location is imperfectly known; evolves according to a Markov chain (5 levels of damage). Number of states: ≈ 10 15
- Repair robot moves left or right, visits and repairs locations. May want to give preference to 'urgent" repairs.
- Belief space partitioning with 6 policy networks and 3 value networks.

<!-- image -->

- RL is a VERY computationally intensive methodology.
- Parallel asynchronous computation is an obvious answer.
- It is important to identify methods that are amenable to distributed computation.
- One-time rollout with a base policy, multiagent parallelization, and/or local value and policy networks is well-suited. Often easy to implement, typically reliable.
- Repeated rollout (i.e, approximate policy iteration) with partitioned architecture and multiagent parallelization, and/or local value and policy networks is well-suited, but is more complicated and more ambitious.
- Rollout has close connections to model predictive control.
- Rollout has many applications to discrete/combinatorial optimization problems.
- There are many interesting analytical and implementation challenges.

## Thank you!