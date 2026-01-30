## Exact and Approximate Dynamic Programming

| Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Contents   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| 1.1. AlphaZero, O ff -Line Training, and On-Line Play . . . . p. 4 1.2. Deterministic Dynamic Programming . . . . . . . . . p. 10 1.2.1. Finite Horizon Problem Formulation . . . . . . . p. 10 1.2.2. The Dynamic Programming Algorithm . . . . . . p. 14 1.2.3. Approximation in Value Space and Rollout . . . . p. 22 1.3. Stochastic Exact and Approximate Dynamic Programming p. 28 1.3.1. Finite Horizon Problems . . . . . . . . . . . . p. 28 1.3.2. Approximation in Value Space for Stochastic DP . p. 34 1.3.3. Approximation in Policy Space . . . . . . . . . p. 38 1.3.4. Training of Cost Function and Policy . . . . . . . . Approximations . . . . . . . . . . . . . . . . p. 41 1.4. Infinite Horizon Problems - An Overview . . . . . . . p. 42 1.4.1. Infinite Horizon Methodology . . . . . . . . . . p. 46 1.4.2. Approximation in Value Space - Infinite Horizon . p. 49 1.4.3. Understanding Approximation in Value Space . . . p. 55 1.5. Newton's Method - Linear Quadratic Problems . . . . . p. 56 1.5.1. Visualizing Approximation in Value Space - Region of . Stability . . . . . . . . . . . . . . . . . . . p. 62 1.5.2. Rollout and Policy Iteration . . . . . . . . . . p. 70 |            |

| 1.6.                                                                    | Examples, Reformulations, and Simplifications . . . . . p. 79 1.6.1. A Few Words About Modeling . . . . . . .   |
|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
|                                                                         | . . p. 79 Problems with a Termination State . . . . . . . p.                                                    |
| 1.6.2.                                                                  | 83                                                                                                              |
| 1.6.3. General Discrete Optimization Problems                           | . . . . . p. 85                                                                                                 |
| 1.6.4. General Finite to Infinite Horizon Reformulation                 | . p. 89                                                                                                         |
| 1.6.5. State Augmentation, Time Delays, Forecasts, Uncontrollable State | and . . . Components . . . . . . . . p. 91                                                                      |
| 1.6.6. Partial State Information and Belief States                      | . . . . p. 97                                                                                                   |
| 1.6.7. Multiagent Problems and Multiagent                               | p. 102                                                                                                          |
| 1.6.8. Problems with Unknown or Changing                                | Rollout . . Model - . . . .                                                                                     |
| Adaptive Control . . 1.6.9. Model Predictive Control .                  | . . . . . . . . . . . . p. 107                                                                                  |
|                                                                         | . . . . . . . . . . p. 117                                                                                      |
| 1.7. Reinforcement Learning and Decision/Control                        | . . . . p. 131                                                                                                  |
| 1.7.1. Di ff erences in Terminology                                     | . . . . . . . . . . p. 131 p. 133                                                                               |
| 1.7.2. Di ff erences in Notation . .                                    | . . . . . . . . . .                                                                                             |
| 1.7.3. Relations Between DP and RL                                      | . . . . . . . . p. 134                                                                                          |
| 1.7.4. Synergy Between Large Language Models                            | and RL p. 135                                                                                                   |
| 1.7.5. Machine Learning and Optimization                                | p. 136                                                                                                          |
| . . . . . 1.7.6. Mathematics in Machine Learning and Optimization p.    | 139                                                                                                             |
| Sources, and Exercises . . . . . . . . . .                              |                                                                                                                 |
| 1.8. Notes,                                                             | . . p. 140                                                                                                      |

This chapter has multiple purposes:

- (a) To provide an overview of the exact dynamic programming (DP) methodology, with a focus on suboptimal solution methods . We will first discuss finite horizon problems, which involve a finite sequence of successive decisions, and are thus conceptually and analytically simpler. We will consider separately deterministic and stochastic finite horizon problems (Sections 1.2 and 1.3, respectively). The reason is that deterministic problems are simpler and have some favorable characteristics, which allow the application of a broader variety of methods. Significantly they include challenging discrete and combinatorial optimization problems, which can be fruitfully addressed with some of the reinforcement learning (RL) methods that are the main subject of the book. We will also discuss somewhat briefly the more intricate infinite horizon methodology (Section 1.4), and refer to the author's DP textbooks [Ber12], [Ber17a], the RL books [Ber19a], [Ber20a], and the neuro-dynamic programming monograph [BeT96] for a fuller presentation.
- (b) To summarize the principal RL methodologies, with primary emphasis on approximation in value space . This is the approximate DP-based architecture that underlies the AlphaZero, AlphaGo, TD-Gammon and other related programs, as well as the Model Predictive Control (MPC) methodology, one of the principal control system design methods. We will also argue later (Chapter 2) that approximation in value space provides the entry point for the use of RL methods for solving discrete optimization and integer programming problems.
- (c) To explain the major principles of approximation in value space, and its division into the o ff -line training and the on-line play algorithms . A key idea here is the connection of these two algorithms through the algorithmic methodology of Newton's method for solving the problem's Bellman equation. This viewpoint, recently developed in the author's 'Rollout and Policy Iteration ...' book [Ber20a] and the visually oriented 'Lessons from AlphaZero ...' monograph [Ber22a], underlies the entire course and is discussed for the simple, intuitive, and important class of linear quadratic problems in Section 1.5.
- (d) To overview the range of problem types where our RL methods apply, and to explain some of their major algorithmic ideas (Section 1.6) . Included here are partial state observation problems (POMDP), multiagent problems, and problems with unknown model parameters, which can be addressed with adaptive control methods.

This chapter will also discuss selectively some major algorithms for approximate DP and RL, including rollout and policy iteration. A broader discussion of DP/RL may be found in the RL books [Ber19a], [Ber20a], the DP textbooks [Ber12], [Ber17a], the neuro-dynamic programming mono-

graph [BeT96], alongside other textbooks referenced in Section 1.8.

The book reflects the author's decision/control and operations research orientation, which has in turn guided the choices of terminology, notation, and mathematical style throughout. On the other hand, RL methods have been developed within the artificial intelligence community, as well as the decision/control and operations research communities. Despite similarities in the mathematical structures of the problems that these communities address, there are notable di ff erences in terminology, notation, and culture, which can be confusing to researchers entering the field. We have thus provided in Section 1.7 a glossary and an orientation to assist the reader in navigating the full range of the DP/RL literature.

## 1.1 ALPHAZERO, OFF-LINE TRAINING, AND ON-LINE PLAY

One of the most exciting recent success stories in RL is the development of the AlphaGo and AlphaZero programs by DeepMind Inc; see [SHM16], [SHS17], [SSS17]. AlphaZero plays Chess, Go, and other games, and is an improvement in terms of performance and generality over AlphaGo, which plays the game of Go only. Both programs play better than all competitor computer programs available in 2022, and much better than all humans. These programs are remarkable in several other ways. In particular, they have learned how to play without human instruction, just data generated by playing against themselves. Moreover, they learned how to play very quickly. In fact, AlphaZero learned how to play chess better than all humans and computer programs within hours (with the help of awesome parallel computation power, it must be said).

Perhaps the most impressive aspect of AlphaZero/chess is that its play is not just better, but it is also very di ff erent than human play in terms of long term strategic vision. Remarkably, AlphaZero has discovered new ways to play a game that has been studied intensively by humans for hundreds of years. Still, for all of its impressive success and brilliant implementation, AlphaZero is couched on well established theory and methodology, which is the subject of the present book, and is portable to far broader realms of engineering, economics, and other fields. This is the methodology of DP, policy iteration, limited lookahead, rollout, and approximation in value space.

It is also worth noting that the principles of the AlphaZero design have much in common with the work of Tesauro [Tes94], [Tes95], [TeG96] on computer backgammon. Tesauro's programs stimulated much interest in RL in the middle 1990s, and exhibit similarly di ff erent and better play than human backgammon players. A related impressive program for the (one-player) game of Tetris, also based on the method of policy iteration, is described by Scherrer et al. [SGG15], who relied on several earlier works, including those of Tsitsiklis and Van Roy [TsV96], and Bertsekas and Io ff e

[BeI96]. For a better understanding of the connections of AlphaZero and AlphaGo Zero with Tesauro's programs and the concepts developed here, the 'Methods' section of the paper [SSS17] is recommended.

To understand the overall structure of AlphaZero and its connection to our DP/RL methodology, it is useful to divide its design into two parts: o ff -line training , which is an algorithm that learns how to evaluate chess positions, and how to steer itself towards good positions with a default/base chess player, and on-line play , which is an algorithm that generates good moves in real time against a human or computer opponent, using the training it went through o ff -line. We will next briefly describe these algorithms, and relate them to DP concepts and principles.

## O ff -Line Training and Policy Iteration

This is the part of the program that learns how to play through o ff -line self-training, and is illustrated in Fig. 1.1.1. The algorithm generates a sequence of chess players and position evaluators . A chess player assigns'probabilities' to all legal moves in a given position, representing the likelihood of each move being chosen. A position evaluator assigns a numerical score to a given position, predicting the player's chances of winning from that position. The chess player and the position evaluator are represented by two neural networks, a policy network and a value network , which accept a chess position and generate a set of move probabilities and a position evaluation, respectively.

In more traditional DP terminology, a position is the state of the game, a position evaluator is a cost function that gives (an estimate of) the optimal cost-to-go at a given state, and the chess player is a randomized policy for selecting actions/controls at a given state. ‡

The overall training algorithm is a form of policy iteration , a classical DP algorithm that will be of primary interest to us in this book. Starting from a given player, it repeatedly generates (approximately) improved players, and settles on a final player that is judged empirically to be 'best'

Here the neural networks play the role of function approximators ; see Chapter 3. By viewing a player as a function that assigns move probabilities to a position, and a position evaluator as a function that assigns a numerical score to a position, the policy and value networks provide approximations to these functions based on training with data (training algorithms for neural networks and other approximation architectures are also discussed in the RL books [Ber19a], [Ber20a], and the neuro-dynamic programming book [BeT96]).

‡ One more complication is that chess and Go are two-player games, while most of our development will involve single-player optimization. However, DP theory extends to two-player games, although we will not focus on this extension. Alternately, we can consider training a game program to play against a known fixed opponent; this is a one-player setting, discussed further in Section 2.12.

atch ...

with 4 hours of training! Current 'Improv

! Approximate Value Function Player Features Mappin

<!-- image -->

erent! Approximate Value Function Player Features Mappin

Figure 1.1.1 Illustration of the AlphaZero training algorithm. It generates a sequence of position evaluators and chess players. The position evaluator and the chess player are represented by two neural networks, a value network and a policy network, which accept a chess position and generate a position evaluation and a set of move probabilities, respectively.

out of all the players generated. Policy iteration may be separated conceptually in two stages (see Fig. 1.1.1).

- (a) Policy evaluation : Given the current player and a chess position, the outcome of a game played out from the position provides a single data point. Many data points are collected and used to train a value network, which then serves as the position evaluator for the player.
- (b) Policy improvement : Given the current player and its position evaluator, trial move sequences are selected and evaluated for the rest of the game starting from many positions. An improved player is then generated by adjusting the move probabilities of the current player towards the trial moves that have yielded the best results. In AlphaZero this is done with a complicated algorithm called Monte Carlo Tree Search . However, policy improvement can also be done more simply. For example, all possible move sequences from a given position could be tried, extending forward a few moves, with the terminal

Quoting from the paper [SSS17]: 'The AlphaGo Zero selfplay algorithm can similarly be understood as an approximate policy iteration scheme in which MCTS is used for both policy improvement and policy evaluation. Policy improvement starts with a neural network policy, executes an MCTS based on that policy's recommendations, and then projects the (much stronger) search policy back into the function space of the neural network. Policy evaluation is applied to the (much stronger) search policy: the outcomes of selfplay games are also projected back into the function space of the neural network. These projection steps are achieved by training the neural network parameters to match the search probabilities and selfplay game outcome respectively.' Note, however, that a twoperson game player, trained through selfplay, may fail against a particular human or computer player that can exploit training vulnerabilities. This is a theoretical but rare possibility; see our discussion in Section 2.12.

position evaluated by the player's position evaluator. The move evaluations obtained in this way are used to nudge the move probabilities of the current player towards more successful moves, thereby obtaining data that is used to train a policy network that represents the new player.

Tesauro's TD-Gammon algorithm [Tes94] is similarly based on approximate policy iteration, but uses a di ff erent methodology for approximate policy evaluation, based on the TD( λ ) algorithm. Unlike AlphaZero, TD-Gammon does not employ a policy network or MCTS. Instead, it relies solely on a value network, to replicate the functionality of a policy network by generating moves on-line via a one-step or two-step lookahead minimization. For a detailed description, see Section 8.6 of the neuro-dynamic programming book [BeT96].

## On-Line Play and Approximation in Value Space - Rollout

Suppose that a 'final' player has been obtained through the AlphaZero o ff -line training process just described. This player could, in principle, play chess against any human or computer opponent by generating move probabilities using its o ff -line-trained policy network. At any position, the player would simply select the move with the highest probability from the policy network. While this approach would allow the player to make decisions quickly, it would not be strong enough to defeat highly skilled human opponents. AlphaZero's extraordinary strength arises only when the o ff -line-trained player and position evaluator are embedded into another algorithm, which we refer to as the on-line player (see Fig. 1.1.2). At a given position, it generates a lookahead tree of all possible multiple move and countermove sequences, up to a given depth. It then evaluates the e ff ect of the remaining moves by using the position evaluator of the o ff -line obtained value network.

The architecture of the final version of Tesauro's TD-Gammon program [TeG96] is similar to the one of AlphaZero, and uses an o ff -line neural network-trained terminal position evaluator; see Fig. 1.1.3. However, it also includes a middle portion, called 'truncated rollout,' which involves running a player for a few moves, before using the position evaluator. In e ff ect, rollout can be viewed as an economical way to extend the length of the lookahead. In the published version of AlphaZero/chess [SHS17], there is also a rollout portion, but it is is rather rudimentary; the first portion (multistep lookahead) is quite long and e ffi ciently implemented, so that a sophisticated rollout portion is not essential. Rollout plays a significant role in AlphaGo [SHM16], and is critically important in Tesauro's backgammon program, where long multistep lookahead is infeasible due to the rapid expansion of the lookahead tree.

Architectures that are similar to the ones of AlphaZero and TDGammon will be generically referred to as approximation in value space

ON-LINE

PLAY

XO

Lookahead Tree

Current

Position

(X K)

States X+1

ON-LINE PLAY

ON-LINE PLAY Lookahead Tree States

Figure 1.1.2 Illustration of the on-line player of AlphaZero and many other computer chess programs. At a given position, it generates a lookahead tree of multiple moves up to some depth, and evaluates the e ff ect of the remaining moves by using the position evaluator of the o ff -line player. There are many implementation details that we do not discuss here; for example the lookahead is selective, because some lookahead paths are pruned, by using a form of Monte Carlo tree search. Note that the o ff -line-trained neural network of AlphaZero produces both a position evaluator and a playing policy. However, the neural network-trained policy is not used directly for on-line play.

<!-- image -->

in this book. Architectures of this type are also known as approximate dynamic programming , or neuro-dynamic programming , and will be central for our purposes.

Among other settings, approximation in value space is used in control system design, particularly in model predictive control (MPC), which is described in Section 1.6.9. There, the number of steps in lookahead minimization is called the control interval , while the total number of steps in

The names 'approximate dynamic programming' and 'neuro-dynamic programming' are often used as synonyms to RL. However, RL is generally thought to also subsume the methodology of approximation in policy space, which involves search for optimal parameters within a parametrized set of policies. The search is done with methods that are largely unrelated to DP, such as for example stochastic gradient or random search methods. Approximation in policy space may be used o ff -line to design a policy that can be used for on-line rollout. It will be discussed in Sections 3.4 and 3.5. A more detailed discussion, consistent with the terminology used here, can be found in Chapter 5 of the RL book [Ber19a].

-Line Obtained Cost Approximation

ON-LINE

PLAY

Lookahead Tree

Current

Position

Xk)

States k+1

ON-LINE PLAY

ON-LINE PLAY Lookahead Tree States

Figure 1.1.3 Illustration of the architecture of TD-Gammon with truncated rollout [TeG96]. It uses a relatively short lookahead minimization followed by rollout and terminal position evaluation (i.e., game simulation with the one-step lookahead player; the game is truncated after a number of moves, with a position evaluation at the end). Note that backgammon involves stochastic uncertainty, and its state is the pair of current board position and dice roll. For this reason, rollout requires Monte-Carlo simulation, and is quite time-consuming. As a result, backgammon programs that play under restrictive time controls, use limited or highly truncated forms of rollout.

<!-- image -->

lookahead minimization and truncated rollout is called the prediction interval ; see e.g., Magni et al. [MDM01]. The benefit of truncated rollout in providing an economical substitute for longer lookahead minimization is well known in MPC.

It should be noted that the preceding description of AlphaZero and related approximation in value space architectures is somewhat simplified. We will be discussing refinements and details as the book progresses. However, DP ideas with cost function approximations, similar to the on-line player illustrated in Fig. 1.1.2, will be central to the book. Moreover, the algorithmic division between o ff -line training and on-line policy implementation will be conceptually very important for our discussions.

We finally note that in approximation in value space the o ff -line training and the on-line play algorithms can often be decoupled and independently designed. For example the o ff -line training portion may be simple, such as using a known policy for rollout without truncation, or without

The Matlab toolbox for MPC design explicitly allows the user to set these two intervals.

Truncated

-Line Obtained Cost Approximation

Figure 1.2.1 Illustration of a deterministic N -stage optimal control problem. Starting from state x k , the next state under control u k is generated nonrandomly, according to

<!-- image -->

<!-- formula-not-decoded -->

and a stage cost g k ( x k ↪ u k ) is incurred.

terminal cost approximation. Conversely, a sophisticated process may be used for o ff -line training of a terminal cost function approximation, which is used immediately following one-step or multistep lookahead in a value space approximation scheme.

## 1.2 DETERMINISTIC DYNAMIC PROGRAMMING

In all DP problems, the central object is a discrete-time dynamic system that generates a sequence of states under the influence of actions or controls. The system may evolve deterministically or randomly (under the additional influence of a random disturbance).

## 1.2.1 Finite Horizon Problem Formulation

In finite horizon problems the system evolves over a finite number N of time steps (also called stages). The state and control at time k of the system will be generally denoted by x k and u k , respectively. In deterministic systems, x k +1 is generated nonrandomly, i.e., it is determined solely by x k and u k ; see Fig. 1.2.1. Thus, a deterministic DP problem involves a system of the form

<!-- formula-not-decoded -->

where k is the time index,

x k is the state of the system, an element of some space, u k is the control or decision variable, to be selected at time k from some given set U k ( x k ) that depends on x k ,

f k is a function of ( x k ↪ u k ) that describes the mechanism by which the state is updated from time k to time k +1,

N is the horizon, i.e., the number of times control is applied.

In the case of a finite number of states, the system function f k may be represented by a table that gives the next state x k +1 for each possible value of the pair ( x k ↪ u k ). Otherwise a mathematical expression or a computer implementation is necessary to represent f k .

The set of all possible x k is called the state space at time k . It can be any set and may depend on k . Similarly, the set of all possible u k is called the control space at time k . Again it can be any set and may depend on k . Similarly the system function f k can be arbitrary and may depend on k .

The problem also involves a cost function that is additive in the sense that the cost incurred at time k , denoted by g k ( x k ↪ u k ), accumulates over time. Formally, g k is a function of ( x k ↪ u k ) that takes scalar values, and may depend on k . For a given initial state x 0 , the total cost of a control sequence ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ is

<!-- formula-not-decoded -->

where g N ( x N ) is a terminal cost incurred at the end of the process. This is a well-defined scalar, since the control sequence ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ together with x 0 determines exactly the state sequence ¶ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♦ via the system equation (1.1). We want to minimize the cost (1.2) over all sequences ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ that satisfy the control constraints, thereby obtaining the optimal value as a function of x 0 : ‡

<!-- formula-not-decoded -->

This generality is one of the great strengths of the DP methodology and guides the exposition style of this book, and the author's other DP works. By allowing general state and control spaces (discrete, continuous, or mixtures thereof), and a k -dependent choice of these spaces, we can focus attention on the truly essential algorithmic aspects of the DP approach, exclude extraneous assumptions and constraints from our model, and avoid duplication of analysis.

The generality of our DP model is also partly responsible for our choice of notation. In the artificial intelligence and operations research communities, finite state models, often referred to as Markovian Decision Problems (MDP), are common and use a transition probability notation (see Section 1.7.2). Unfortunately, this notation is not well suited for deterministic models, and also for continuous spaces models, both of which are important for the purposes of this book. For the latter models, it involves transition probability distributions over continuous spaces, and leads to mathematics that are far more complex as well as less intuitive than those based on the use of the system function (1.1).

‡ Here and later we write 'min' (rather than 'inf') even if we are not sure that the minimum is attained; similarly we write 'max' (rather than 'sup') even if we are not sure that the maximum is attained.

aY auu tult ol tro ouent +1 †

State Transition

Cost 91(x1, U1)

X2 = f1 (X1, U1)

X2

C1

U1

Stage 1

X1

Stage 2

XN-1

UN-

••• Stage N -1

X2

State Space Partition Initial States

Initial State

Initial States

Stage 0

Terminal Arcs

Cost gN (IN)

Artificial Terminal

<!-- image -->

Current Position

Current Position

Figure 1.2.2 Transition graph for a deterministic finite-state system. Nodes correspond to states x k . Arcs correspond to state-control pairs ( x k ↪ u k ). An arc ( x k ↪ u k ) has start and end nodes x k and x k +1 = f k ( x k ↪ u k ) ↪ respectively. We view the cost g k ( x k ↪ u k ) of the transition as the length of this arc. The problem is equivalent to finding a shortest path from initial nodes of stage 0 to the terminal node t .

## Discrete Optimal Control Problems

There are many situations where the state and control spaces are naturally discrete and consist of a finite number of elements. Such problems are often conveniently described with an acyclic graph specifying for each state x k the possible transitions to next states x k +1 . The nodes of the graph correspond to states x k and the arcs of the graph correspond to state-control pairs ( x k ↪ u k ). Each arc with start node x k corresponds to a choice of a single control u k ∈ U k ( x k ) and has as end node the next state f k ( x k ↪ u k ). The cost of an arc ( x k ↪ u k ) is defined as g k ( x k ↪ u k ); see Fig. 1.2.2. To handle the final stage, an artificial terminal node t is added. Each state x N at stage N is connected to the terminal node t with an arc having cost g N ( x N ).

Note that control sequences ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ correspond to paths originating at the initial state (a node at stage 0) and terminating at one of the nodes corresponding to the final stage N . If we view the cost of an arc as its length, we see that a deterministic finite-state finite-horizon problem is equivalent to finding a minimum-length (or shortest) path from the initial nodes of the graph (stage 0) to the terminal node t . Here, by the length of a path we mean the sum of the lengths of its arcs.

Generally, combinatorial optimization problems can be formulated as deterministic finite-state finite-horizon optimal control problems. The idea

It turns out also that any shortest path problem (with a possibly nonacyclic graph) can be reformulated as a finite-state deterministic optimal control problem. See [Ber17a], Section 2.1, and [Ber91], [Ber98] for extensive accounts of shortest path methods, which connect with our discussion here.

with Cost Equal to Ter-

Terminal Arcs with Cost Equal

to Ter-

+1 Initial State A C

+1 Initial State A C

Figure 1.2.3 The transition graph of the deterministic scheduling problem of Example 1.2.1. Each arc of the graph corresponds to a decision leading from some state (the start node of the arc) to some other state (the end node of the arc). The corresponding cost is shown next to the arc. The cost of the last operation is shown as a terminal cost next to the terminal nodes of the graph.

<!-- image -->

is to break down the solution into components, which can be computed sequentially. The following is an illustrative example.

## Example 1.2.1 (A Deterministic Scheduling Problem)

Suppose that to produce a certain product, four operations must be performed on a given machine. The operations are denoted by A, B, C, and D. We assume that operation B can be performed only after operation A has been performed, and operation D can be performed only after operation C has been performed. (Thus the sequence CDAB is allowable but the sequence CDBA is not.) The setup cost C mn for passing from any operation m to any other operation n is given. There is also an initial startup cost S A or S C for starting with operation A or C, respectively (cf. Fig. 1.2.3). The cost of a sequence is the sum of the setup costs associated with it; for example, the operation sequence ACDB has cost S A + C AC + C CD + C DB glyph[triangleright]

We can view this problem as a sequence of three decisions, namely the choice of the first three operations to be performed (the last operation is determined from the preceding three). It is appropriate to consider as state the set of operations already performed, the initial state being an artificial state corresponding to the beginning of the decision process. The possible state transitions corresponding to the possible states and decisions for this problem are shown in Fig. 1.2.3. Here the problem is deterministic, i.e., at a given state, each choice of control leads to a uniquely determined state. For example, at state AC the decision to perform operation D leads to state ACD

CAB

CAD

CDA

CAD

CDA

CDA

with certainty, and has cost C CD . Thus the problem can be conveniently represented with the transition graph of Fig. 1.2.3 (which in turn is a special case of the graph of Fig. 1.2.2). The optimal solution corresponds to the path that starts at the initial state and ends at some state at the terminal time and has minimum sum of arc costs plus the terminal cost.

## 1.2.2 The Dynamic Programming Algorithm

In this section we will state the DP algorithm and formally justify it. Generally, DP is used to solve a problem of sequential decision making over N stages, by breaking it down to a sequence of simpler single-stage problems.

In particular, the algorithm aims to find a sequence of optimal controls u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 by generating a corresponding sequence of optimal cost functions J ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ J ∗ N -1 . It starts with J ∗ N equal to the terminal cost function g N and computes the next function J ∗ N -1 , by solving a single stage decision problem whose optimization variable is u N -1 . It then uses J ∗ N -1 to compute J ∗ N -2 , and proceeds similarly to compute all the remaining cost functions J ∗ N -3 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ J ∗ 0 .

The algorithm rests on a simple idea, the principle of optimality , which roughly states the following; see Fig. 1.2.4.

## Principle of Optimality

Let ¶ u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ be an optimal control sequence, which together with x 0 determines the corresponding state sequence ¶ x ∗ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x ∗ N ♦ via the system equation (1.1). Consider the subproblem whereby we start at x ∗ k at time k and wish to minimize the 'cost-to-go' from time k to time N ,

<!-- formula-not-decoded -->

over ¶ u k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ with u m ∈ U m ( x m ), m = k↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1. Then the truncated optimal control sequence ¶ u ∗ k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ is optimal for this subproblem.

The subproblem referred to above is called the tail subproblem that starts at x ∗ k . Stated succinctly, the principle of optimality says that the tail of an optimal sequence is optimal for the tail subproblem . Its intuitive justification is simple. If the truncated control sequence ¶ u ∗ k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ were not optimal as stated, we would be able to reduce the cost further by switching to an optimal sequence for the subproblem once we reach x ∗ k (since the preceding choices of controls, u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ k -1 , do not restrict our future choices).

For an auto travel analogy, suppose that the fastest route from Phoenix to Boston passes through St Louis. The principle of optimality translates

Tail subproblem Time

Future Stages Terminal Cost

Cost 0 Cost

Optimal control sequence

Figure 1.2.4 Schematic illustration of the principle of optimality. The tail ¶ u ∗ k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ of an optimal sequence ¶ u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ is optimal for the tail subproblem that starts at the state x ∗ k of the optimal state trajectory.

<!-- image -->

to the obvious fact that the St Louis to Boston portion of the route is also the fastest route for a trip that starts from St Louis and ends in Boston.

The principle of optimality suggests that the optimal cost function can be constructed in piecemeal fashion going backwards: first compute the optimal cost function for the 'tail subproblem' involving the last stage, then solve the 'tail subproblem' involving the last two stages, and continue in this manner until the optimal cost function for the entire problem is constructed.

The DP algorithm is based on this idea: it proceeds sequentially by solving all the tail subproblems of a given time length, using the solution of the tail subproblems of shorter time length . We illustrate the algorithm with the scheduling problem of Example 1.2.1. The calculations are simple but tedious, and may be skipped without loss of continuity. However, they may be worth going over by a reader that has no prior experience in the use of DP.

## Example 1.2.1 (Scheduling Problem - Continued)

Let us consider the scheduling Example 1.2.1, and let us apply the principle of optimality to calculate the optimal schedule. We have to schedule optimally the four operations A, B, C, and D. There is a cost for a transition between two operations, and the numerical values of the transition costs are shown in Fig. 1.2.5 next to the corresponding arcs.

According to the principle of optimality, the 'tail' portion of an optimal schedule must be optimal. For example, suppose that the optimal schedule is CABD. Then, having scheduled first C and then A, it must be optimal to complete the schedule with BD rather than with DB. With this in mind, we solve all possible tail subproblems of length two, then all tail subproblems of length three, and finally the original problem that has length four (the subproblems of length one are of course trivial because there is only one operation

In the words of Bellman [Bel57]: 'An optimal trajectory has the property that at an intermediate point, no matter how it was reached, the rest of the trajectory must coincide with an optimal trajectory as computed from this intermediate point as the starting point.'

+1 Initial State A C

6 1 3 2 9 5 8 7 10

+1 Initial State A C

6 1 3 2 9 5 8 7 10

ABC

6 1 3 2 9 5 8 7 10

ACD

CAB

CAB

CAD

CAD

CDA

6 1 3 2 9 5 8 7 10

6 1 3 2 9 5 8 7 10

CAD

CDA

6 1 3 2 9 5 8 7 10

CDA

6 1 3 2 9 5 8 7 10

Figure 1.2.5 Transition graph of the deterministic scheduling problem, with the cost of each decision shown next to the corresponding arc. Next to each node/state we show the cost to optimally complete the schedule starting from that state. This is the optimal cost of the corresponding tail subproblem (cf. the principle of optimality). The optimal cost for the original problem is equal to 10, as shown next to the initial state. The optimal schedule corresponds to the thick-line arcs.

<!-- image -->

that is as yet unscheduled). As we will see shortly, the tail subproblems of length k + 1 are easily solved once we have solved the tail subproblems of length k , and this is the essence of the DP technique.

Tail Subproblems of Length 2 : These subproblems are the ones that involve two unscheduled operations and correspond to the states AB, AC, CA, and CD (see Fig. 1.2.5).

State AB : Here it is only possible to schedule operation C as the next operation, so the optimal cost of this subproblem is 9 (the cost of scheduling C after B, which is 3, plus the cost of scheduling D after C, which is 6).

State AC : Here the possibilities are to (a) schedule operation B and then D, which has cost 5, or (b) schedule operation D and then B, which has cost 9. The first possibility is optimal, and the corresponding cost of the tail subproblem is 5, as shown next to node AC in Fig. 1.2.5.

State CA : Here the possibilities are to (a) schedule operation B and then D, which has cost 3, or (b) schedule operation D and then B, which has cost 7. The first possibility is optimal, and the corresponding cost of the tail subproblem is 3, as shown next to node CA in Fig. 1.2.5.

State CD : Here it is only possible to schedule operation A as the next operation, so the optimal cost of this subproblem is 5.

CDA

Tail Subproblems of Length 3 : These subproblems can now be solved using the optimal costs of the subproblems of length 2.

State A : Here the possibilities are to (a) schedule next operation B (cost 2) and then solve optimally the corresponding subproblem of length 2 (cost 9, as computed earlier), a total cost of 11, or (b) schedule next operation C (cost 3) and then solve optimally the corresponding subproblem of length 2 (cost 5, as computed earlier), a total cost of 8. The second possibility is optimal, and the corresponding cost of the tail subproblem is 8, as shown next to node A in Fig. 1.2.5.

State C : Here the possibilities are to (a) schedule next operation A (cost 4) and then solve optimally the corresponding subproblem of length 2 (cost 3, as computed earlier), a total cost of 7, or (b) schedule next operation D (cost 6) and then solve optimally the corresponding subproblem of length 2 (cost 5, as computed earlier), a total cost of 11. The first possibility is optimal, and the corresponding cost of the tail subproblem is 7, as shown next to node C in Fig. 1.2.5.

Original Problem of Length 4 : The possibilities here are (a) start with operation A (cost 5) and then solve optimally the corresponding subproblem of length 3 (cost 8, as computed earlier), a total cost of 13, or (b) start with operation C (cost 3) and then solve optimally the corresponding subproblem of length 3 (cost 7, as computed earlier), a total cost of 10. The second possibility is optimal, and the corresponding optimal cost is 10, as shown next to the initial state node in Fig. 1.2.5.

Note that having computed the optimal cost of the original problem through the solution of all the tail subproblems, we can construct the optimal schedule: we begin at the initial node and proceed forward, each time choosing the optimal operation, i.e., the one that starts the optimal schedule for the corresponding tail subproblem. In this way, by inspection of the graph and the computational results of Fig. 1.2.5, we determine that CABD is the optimal schedule.

## Finding an Optimal Control Sequence by DP

We now state the DP algorithm for deterministic finite horizon problems by translating into mathematical terms the heuristic argument underlying the principle of optimality. The algorithm constructs functions

<!-- formula-not-decoded -->

sequentially, starting from J * N , and proceeding backwards to J * N -1 ↪ J * N -2 ↪ etc. We will show that the value J * k ( x k ) represents the optimal cost of the tail subproblem that starts at state x k at time k .

Cost 0 Cost

Future Stages Terminal Cost

Future Stages Terminal Cost

<!-- image -->

Cost 0 Cost

Figure 1.2.6 Illustration of the DP algorithm. The tail subproblem that starts at x k at time k minimizes over ¶ u k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ the 'cost-to-go' from k to N ,

<!-- formula-not-decoded -->

To solve it, we choose u k to minimize the (1st stage cost + Optimal tail problem cost) or

<!-- formula-not-decoded -->

## DP Algorithm for Deterministic Finite Horizon Problems

Start with

<!-- formula-not-decoded -->

and for k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, let

<!-- formula-not-decoded -->

The DP algorithm together with the construction of the functions J * k ( x k ) are illustrated in Fig. 1.2.6. Note that at stage k , the calculation in Eq. (1.5) must be done for all states x k before proceeding to stage k -1. The key fact about the DP algorithm is that for every initial state x 0 , the number J * 0 ( x 0 ) obtained at the last step, is equal to the optimal cost J * ( x 0 ). Indeed, a more general fact can be shown, namely that for all

k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, and all states x k at time k , we have

<!-- formula-not-decoded -->

where J ( x k ; u k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ) is the cost generated by starting at x k and using subsequent controls u k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 :

<!-- formula-not-decoded -->

Thus, J * k ( x k ) is the optimal cost for an ( N -k )-stage tail subproblem that starts at state x k and time k , and ends at time N . Based on the interpretation (1.6) of J ∗ k ( x k ), we call it the optimal cost-to-go from state x k at stage k , and refer to J ∗ k as the optimal cost-to-go function or optimal cost function at time k . In maximization problems the DP algorithm (1.5) is written with maximization in place of minimization, and then J ∗ k is referred to as the optimal value function at time k .

Once the functions J * 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ J * N have been obtained, we can use a forward algorithm to construct an optimal control sequence ¶ u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ and corresponding state trajectory ¶ x ∗ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x ∗ N ♦ for the given initial state x 0 .

We can prove this by induction. The assertion holds for k = N in view of the initial condition

<!-- formula-not-decoded -->

To show that it holds for all k , we use Eqs. (1.6) and (1.7) to write

<!-- formula-not-decoded -->

where for the last equality we use the induction hypothesis. A subtle mathematical point here is that, through the minimization operation, the cost-to-go functions J ∗ k may take the value -∞ for some x k . Still the preceding induction argument is valid even if this is so.

<!-- formula-not-decoded -->

The same algorithm can be used to find an optimal control sequence for any tail subproblem. Figure 1.2.5 traces the calculations of the DP algorithm for the scheduling Example 1.2.1. The numbers next to the nodes, give the corresponding cost-to-go values, and the thick-line arcs give the construction of the optimal control sequence using the preceding algorithm.

The following example deals with the classical traveling salesman problem involving N cities. Here, the number of states grows exponentially with N , and so does the corresponding amount of computation for exact DP. We will show later that with rollout, we can solve the problem approximately with computation that grows polynomially with N .

## Example 1.2.2 (Traveling Salesman Problem)

Here we are given N cities and the travel time between each pair of cities. We wish to find a minimum time travel that visits each of the cities exactly once and returns to the start city. To convert this problem to a DP problem, we form a graph whose nodes are the sequences of k distinct cities, where k = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N . The k -city sequences correspond to the states of the k th stage. The initial state x 0 consists of some city, taken as the start (city A in the example of Fig. 1.2.7). A k -city node/state leads to a ( k +1)-city node/state by adding a new city at a cost equal to the travel time between the last two of the k + 1 cities; see Fig. 1.2.7. Each sequence of N cities is connected to an artificial terminal node t with an arc of cost equal to the travel time from the last city of the sequence to the starting city, thus completing the transformation to a DP problem.

The optimal costs-to-go from each node to the terminal state can be obtained by the DP algorithm and are shown next to the nodes. Note, however, that the number of nodes grows exponentially with the number of cities N . This makes the DP solution intractable for large N . As a result, large travel-

15 1 5 18 4 19 9 21 25 8 12

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

A AB AC AD ABC ABD ACB ACD ADB ADC

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Matrix of Intercity Travel Costs

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

<!-- image -->

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Figure 1.2.7 The DP formulation of the traveling salesman problem of Example 1.2.2. The travel times between the four cities A, B, C, and D are shown in the matrix at the bottom. We form a graph whose nodes are the k -city sequences and correspond to the states of the k th stage, assuming that A is the starting city. The transition costs/travel times are shown next to the arcs. The optimal costs-to-go are generated by DP starting from the terminal state and going backwards towards the initial state, and are shown next to the nodes. There is a unique optimal sequence here (ABDCA), and it is marked with thick lines. The optimal sequence can be obtained by forward minimization [cf. Eq. (1.8)], starting from the initial state x 0 .

ing salesman and related scheduling problems are typically not addressed with exact DP, but rather with approximation methods. Some of these methods are based on DP and will be discussed later.

## Q-Factors and Q-Learning

An alternative (and equivalent) form of the DP algorithm (1.5), uses the optimal cost-to-go functions J * k indirectly. In particular, it generates the

optimal Q-factors , defined for all pairs ( x k ↪ u k ) and k by

<!-- formula-not-decoded -->

Thus the optimal Q-factors are simply the expressions that are minimized in the right-hand side of the DP equation (1.5).

Note that the optimal cost function J * k can be recovered from the optimal Q-factor Q * k by means of the minimization

<!-- formula-not-decoded -->

Moreover, the DP algorithm (1.5) can be written in an essentially equivalent form that involves Q-factors only [cf. Eqs. (1.9)-(1.10)]:

<!-- formula-not-decoded -->

Exact and approximate forms of this and other related algorithms, including counterparts for stochastic optimal control problems, comprise an important class of RL methods known as Q-learning .

## 1.2.3 Approximation in Value Space and Rollout

The forward optimal control sequence construction of Eq. (1.8) is possible only after we have computed J * k ( x k ) by DP for all x k and k . Unfortunately, in practice this is often prohibitively time-consuming, because the number of possible x k and k can be very large. However, a similar forward algorithmic process can be used if the optimal cost-to-go functions J * k are replaced by some approximations ˜ J k . This is the basis for an idea that is central in RL: approximation in value space . ‡ It constructs a suboptimal solution ¶ ˜ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u N -1 ♦ in place of the optimal ¶ u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ , based on using ˜ J k in place of J * k in the DP algorithm (1.8).

The term 'Q-factor' has been used in the books [BeT96], [Ber19a], [Ber20a] and is adopted here as well. Another term used is 'action value' (at a given state). The terms 'state-action value' and 'Q-value' are also common in the literature. The name 'Q-factor' originated in reference to the notation used in an influential Ph.D. thesis [Wat89] that proposed the use of Q-factors in RL.

‡ Approximation in value space (sometimes called 'search' or 'tree search' in the AI literature) is a simple idea that has been used quite extensively for deterministic problems, well before the development of the modern RL methodology. For example it conceptually underlies the widely used A ∗ method for computing approximate solutions to large scale shortest path problems. For a view of A ∗ that is consistent with our approximate DP framework, the reader may consult the author's DP book [Ber17a].

## Approximation in Value Space - Use of ˜ J k in Place of J *

Start with and set

<!-- formula-not-decoded -->

Sequentially, going forward, for k = 1 ↪ 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, set

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In approximation in value space the calculation of the suboptimal sequence ¶ ˜ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u N -1 ♦ is done by going forward (no backward calculation is needed once the approximate cost-to-go functions ˜ J k are available). This is similar to the calculation of the optimal sequence ¶ u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ♦ , and is independent of how the functions ˜ J k are computed. The motivation for approximation in value space for stochastic DP problems is vastly reduced computation relative to the exact DP algorithm (once ˜ J k have been obtained): the minimization (1.11) needs to be performed only for the N states x 0 ↪ ˜ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ x N -1 that are encountered during the on-line control of the system, and not for every state within the potentially enormous state space, as is the case for exact DP.

The algorithm (1.11) is said to involve a one-step lookahead minimization , since it solves a one-stage DP problem for each k . In what follows we will also discuss the possibility of multistep lookahead , which involves the solution of an /lscript -step DP problem, where /lscript is an integer, 1 &lt; /lscript &lt; N -k , with a terminal cost function approximation ˜ J k + /lscript . Multistep lookahead typically (but not always) provides better performance over one-step lookahead in RL approximation schemes. For example in AlphaZero chess, long multistep lookahead is critical for good on-line performance. The intuitive reason is that with /lscript stages being treated 'exactly' (by optimization), the e ff ect of the approximation error

<!-- formula-not-decoded -->

tends to become less significant as /lscript increases. However, the solution of the multistep lookahead optimization problem, instead of the one-step lookahead counterpart of Eq. (1.11), becomes more time consuming.

k

## Rollout with a Base Heuristic for Deterministic Problems

A major issue in value space approximation is the construction of suitable approximate cost-to-go functions ˜ J k . This can be done in many di ff erent ways, giving rise to some of the principal RL methods. For example, ˜ J k may be constructed with a sophisticated o ff -line training method, as discussed in Section 1.1. Alternatively, ˜ J k may be obtained on-line with rollout , which will be discussed in detail in this book. In rollout, the approximate values ˜ J k ( x k ) are obtained when needed by running a heuristic control scheme, called base heuristic or base policy , for a suitably large number of stages, starting from the state x k , and accumulating the costs incurred at these stages.

The major theoretical property of rollout is cost improvement : the cost obtained by rollout using some base heuristic is less or equal to the corresponding cost of the base heuristic. This is true for any starting state, provided the base heuristic satisfies some simple conditions, which will be discussed in Chapter 2.

There are also several variants of rollout, including versions involving multiple heuristics, combinations with other forms of approximation in value space methods, multistep lookahead, and stochastic uncertainty. We will discuss such variants later. For the moment we will focus on a deterministic DP problem with a finite number of controls. Given a state x k at time k , this algorithm considers all the tail subproblems that start at every possible next state x k +1 , and solves them suboptimally by using some algorithm, referred to as base heuristic.

Thus when at x k , rollout generates on-line the next states x k +1 that correspond to all u k ∈ U k ( x k ), and uses the base heuristic to compute the sequence of states ¶ x k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♦ and controls ¶ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ such that

<!-- formula-not-decoded -->

and the corresponding cost

<!-- formula-not-decoded -->

The rollout algorithm then applies the control that minimizes over u k ∈ U k ( x k ) the tail cost expression for stages k to N :

<!-- formula-not-decoded -->

For an intuitive justification of the cost improvement mechanism, note that the rollout control ˜ u k is calculated from Eq. (1.11) to attain the minimum over u k over the sum of two terms: the first stage cost g k (˜ x k ↪ u k ) plus the cost of the remaining stages ( k +1 to N ) using the heuristic controls. Thus rollout involves a first stage optimization (rather than just using the base heuristic), which accounts for the cost improvement. This reasoning also explains why multistep lookahead tends to provide better performance than one-step lookahead in rollout schemes.

XO

X1

Current State

•••

Xk

Next States

Xk+1

Xk+1

XN

Figure 1.2.8 Schematic illustration of rollout with one-step lookahead for a deterministic problem. At state x k , for every pair ( x k ↪ u k ), u k ∈ U k ( x k ), the base heuristic generates an approximate Q-factor

<!-- image -->

<!-- formula-not-decoded -->

and selects the control ˜ θ k ( x k ) with minimal Q-factor.

Equivalently, and more succinctly, the rollout algorithm applies at state x k the control ˜ θ k ( x k ) given by the minimization

<!-- formula-not-decoded -->

where ˜ Q k ( x k ↪ u k ) is the approximate Q-factor defined by see Fig. 1.2.8.

<!-- formula-not-decoded -->

Note that the rollout algorithm requires running the base heuristic for a number of times that is bounded by Nn , where n is an upper bound on the number of control choices available at each state. Thus if n is small relative to N , it requires computation equal to a small multiple of N times the computation time for a single application of the base heuristic. Similarly, if n is bounded by a polynomial in N , the ratio of the rollout algorithm computation time to the base heuristic computation time is a polynomial in N .

## Example 1.2.3 (Traveling Salesman Problem - Continued)

Let us consider the traveling salesman problem of Example 1.2.2, whereby a salesman wants to find a minimum cost tour that visits each of N given cities c = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 exactly once and returns to the city he started from. With each pair of distinct cities c , c ′ , we associate a traversal cost g ( c↪ c ′ ). Note

Heuristic

Initial City

Xo

X1

Next Partial

Tours

Next Cities

Current

Partial Tour

Xk

X k+1

7k+1

X'N

Figure 1.2.9 Rollout with the nearest neighbor heuristic for the traveling salesman problem of Example 1.2.3. The initial state x 0 consists of a single city. The final state x N is a complete tour of N cities, containing each city exactly once.

<!-- image -->

that we assume that we can go directly from every city to every other city. There is no loss of generality in doing so because we can assign a very high cost g ( c↪ c ′ ) to any pair of cities ( c↪ c ′ ) that is precluded from participation in the solution. The problem is to find a visit order that goes through each city exactly once and whose sum of costs is minimum.

/negationslash

There are many heuristic approaches for solving the traveling salesman problem. For illustration purposes, let us focus on the simple nearest neighbor heuristic, which starts with a partial tour, i.e., an ordered collection of distinct cities, and constructs a sequence of partial tours, adding to the each partial tour a new city that does not close a cycle and minimizes the cost of the enlargement. In particular, given a sequence ¶ c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ♦ (with k &lt; N -1) consisting of distinct cities, the nearest neighbor heuristic adds a city c k +1 that minimizes g ( c k ↪ c k +1 ) over all cities c k +1 = c 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k , thereby forming the sequence ¶ c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ↪ c k +1 ♦ . Continuing in this manner, the heuristic eventually forms a sequence of N cities, ¶ c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c N -1 ♦ , thus yielding a complete tour with cost

<!-- formula-not-decoded -->

/negationslash

We can formulate the traveling salesman problem as a DP problem as we discussed in Example 1.2.2. We choose a starting city, say c 0 , as the initial state x 0 . Each state x k corresponds to a partial tour ( c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ) consisting of distinct cities. The states x k +1 , next to x k , are sequences of the form ( c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ↪ c k +1 ) that correspond to adding one more unvisited city c k +1 = c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k (thus the unvisited cities are the feasible controls at a given partial tour/state). The terminal states x N are the complete tours of the form ( c 0 ↪ c 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c N -1 ↪ c 0 ), and the cost of the corresponding sequence of city choices is the cost of the corresponding complete tour given by Eq. (1.14). Note that the number of states at stage k increases exponentially with k , and so does the computation required to solve the problem by exact DP.

Let us now use as a base heuristic the nearest neighbor method. The corresponding rollout algorithm operates as follows: After k &lt; N -1 iterations, we have a state x k , i.e., a sequence ¶ c 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ♦ consisting of distinct cities. At the next iteration, we add one more city by running the

Xk+1

Complete Tours

Nearest Neighbor

XN

Heuristic

A AB AC AD ABC ABD ACB ACD ADB ADC

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

Matrix of Intercity Travel Costs

Figure 1.2.10 Rollout with the nearest neighbor base heuristic, applied to a traveling salesman problem. At city A, the nearest neighbor heuristic generates the tour ACDBA (labelled T 0 ). At city A, the rollout algorithm compares the tours ABCDA, ACDBA, and ADCBA, finds ABCDA (labelled T 1 ) to have the least cost, and moves to city B. At AB, the rollout algorithm compares the tours ABCDA and ABDCA, finds ABDCA (labelled T 2 ) to have the least cost, and moves to city D. The rollout algorithm then moves to cities C and A (it has no other choice). The final tour T 2 generated by rollout turns out to be optimal in this example, while the tour T 0 generated by the base heuristic is suboptimal. This is suggestive of a general result: the rollout algorithm for deterministic problems generates a sequence of solutions of decreasing cost under some conditions on the base heuristic that we will discuss in Chapter 2, and which are satisfied by the nearest neighbor heuristic.

<!-- image -->

/negationslash nearest neighbor heuristic starting from each of the sequences of the form ¶ c 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k ↪ c ♦ where c = c 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ c k . We then select as next city c k +1 the city c that yielded the minimum cost tour under the nearest neighbor heuristic; see Fig. 1.2.9. The overall computation for the rollout solution is bounded by a polynomial in N , and is much smaller than the exact DP computation. Figure 1.2.10 provides an example where the nearest neighbor heuristic and the corresponding rollout algorithm are compared; see also Exercise 1.1.

A AB AC AD ABC ABD ACB ACD ADB ADC

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20

## 1.3 STOCHASTIC EXACT AND APPROXIMATE DYNAMIC PROGRAMMING

We will now extend the DP algorithm and our discussion of approximation in value space to problems that involve stochastic uncertainty in their system equation and cost function. We will first discuss the finite horizon case, and the extension of the ideas underlying the principle of optimality and approximation in value space schemes. We will then consider the infinite horizon version of the problem, and provide an overview of the underlying theory and algorithmic methodology.

## 1.3.1 Finite Horizon Problems

The stochastic optimal control problem di ff ers from its deterministic counterpart primarily in the nature of the discrete-time dynamic system that governs the evolution of the state x k . This system includes a random 'disturbance' w k with a probability distribution P k ( · ♣ x k ↪ u k ) that may depend explicitly on x k and u k , but not on values of prior disturbances w k -1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w 0 . The system has the form

<!-- formula-not-decoded -->

where as earlier x k is an element of some state space, the control u k is an element of some control space. The cost per stage is denoted by g k ( x k ↪ u k ↪ w k ) and also depends on the random disturbance w k ; see Fig. 1.3.1. The control u k is constrained to take values in a given subset U k ( x k ), which depends on the current state x k .

Given an initial state x 0 and a policy π = ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ , the future states x k and disturbances w k are random variables with distributions defined through the system equation

<!-- formula-not-decoded -->

The discrete equation format and corresponding x -u -w notation is standard in the optimal control literature. For finite-state stochastic problems, also called Markovian Decision Problems (MDP), the system is often represented conveniently in terms of control-dependent transition probabilities. A common notation in the RL literature is p ( s↪ a↪ s ′ ) for transition probability from s to s ′ under action a . This type of notation is not well suited for deterministic problems, which involve no probabilistic structure at all and are of major interest in this book. The transition probability notation is also cumbersome for problems with a continuous state space; see Sections 1.7.1 and 1.7.2 for further discussion. The reader should note, however, that mathematically the system equation and transition probabilities are equivalent, and any analysis that can be done in one notational system can be translated to the other notational system.

Figure 1.3.1 Illustration of an N -stage stochastic optimal control problem. Starting from state x k , the next state under control u k is generated randomly, according to x k +1 = f k ( x k ↪ u k ↪ w k ) ↪ where w k is the random disturbance, and a random stage cost g k ( x k ↪ u k ↪ w k ) is incurred.

<!-- image -->

and the given distributions P k ( · ♣ x k ↪ u k ). Thus, for given functions g k , k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N , the expected cost of π starting at x 0 is

<!-- formula-not-decoded -->

where the expected value operation E ¶·♦ is taken with respect to the joint distribution of all the random variables w k and x k . An optimal policy π ∗ is one that minimizes this cost; i.e.,

<!-- formula-not-decoded -->

where Π is the set of all policies.

An important di ff erence from the deterministic case is that we optimize not over control sequences ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ [cf. Eq. (1.3)], but rather over policies (also called closed-loop control laws , or feedback policies ) that consist of a sequence of functions

<!-- formula-not-decoded -->

where θ k maps states x k into controls u k = θ k ( x k ), and satisfies the control constraints, i.e., is such that θ k ( x k ) ∈ U k ( x k ) for all x k . Policies are more general objects than control sequences, and in the presence of stochastic uncertainty, they can result in improved cost, since they allow choices of controls u k that incorporate knowledge of the state x k . Without this knowledge, the controller cannot adapt appropriately to unexpected values of the state, and as a result the cost can be adversely a ff ected. This is a fundamental distinction between deterministic and stochastic optimal control problems.

We assume an introductory probability background on the part of the reader. For an account that is consistent with our use of probability in this book, see the textbook by Bertsekas and Tsitsiklis [BeT08].

The optimal cost depends on x 0 and is denoted by J * ( x 0 ); i.e.,

<!-- formula-not-decoded -->

We view J * as a function that assigns to each initial state x 0 the optimal cost J * ( x 0 ), and call it the optimal cost function or optimal value function .

## Stochastic Dynamic Programming

The DP algorithm for the stochastic finite horizon optimal control problem has a similar form to its deterministic version, and shares several of its major characteristics:

- (a) Using tail subproblems to break down the minimization over multiple stages to single stage minimizations.
- (b) Generating backwards for all k and x k the values J * k ( x k ), which give the optimal cost-to-go starting from state x k at stage k .
- (c) Obtaining an optimal policy by minimization in the DP equations.
- (d) A structure that is suitable for approximation in value space, whereby we replace J * k by approximations ˜ J k , and obtain a suboptimal policy by the corresponding minimization.

## DP Algorithm for Stochastic Finite Horizon Problems

Start with

<!-- formula-not-decoded -->

and for k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, let

<!-- formula-not-decoded -->

For each x k and k , define θ ∗ k ( x k ) = u ∗ k where u ∗ k attains the minimum in the right side of this equation. Then, the policy π ∗ = ¶ θ ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ ∗ N -1 ♦ is optimal.

The key fact is that starting from any initial state x 0 , the optimal cost is equal to the number J * 0 ( x 0 ), obtained at the last step of the above DP algorithm. This can be proved by induction similar to the deterministic case; we will omit the proof (which incidentally involves some mathematical fine points; see the discussion of Section 1.3 in the textbook [Ber17a]).

Simultaneously with the o ff -line computation of the optimal costto-go functions J * 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ J * N , we can compute and store an optimal policy π ∗ = ¶ θ ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ ∗ N -1 ♦ by minimization in Eq. (1.15). We can then use this

policy on-line to retrieve from memory and apply the control θ ∗ k ( x k ) once we reach state x k . The alternative is to forego the storage of the policy π ∗ and to calculate the control θ ∗ k ( x k ) by executing the minimization (1.15) on-line.

There are a few favorable cases where the optimal cost-to-go functions J * k and the optimal policies θ ∗ k can be computed analytically using the stochastic DP algorithm. A prominent such case involves a linear system and a quadratic cost function, which is a fundamental problem in control theory. We illustrate the scalar version of this problem next. The analysis can be generalized to multidimensional systems (see optimal control textbooks such as [Ber17a]).

## Example 1.3.1 (Linear Quadratic Optimal Control)

Here the system is linear,

<!-- formula-not-decoded -->

and the state and control are scalars. Moreover, the disturbance w k has zero mean and given variance σ 2 . The cost is quadratic of the form:

<!-- formula-not-decoded -->

where q and r are known positive weighting parameters. We assume no constraints on x k and u k (in reality such problems include constraints, but it is common to neglect the constraints initially, and check whether they are seriously violated later).

As an illustration, consider a vehicle that moves on a straight-line road under the influence of a force u k and without friction. Our objective is to maintain the vehicle's velocity at a constant level ¯ v (as in an oversimplified cruise control system). The velocity v k at time k , after time discretization of its Newtonian dynamics and addition of stochastic noise, evolves according to

<!-- formula-not-decoded -->

where w k is a stochastic disturbance. By introducing x k = v k -¯ v , the deviation between the vehicle's velocity v k at time k from the desired level ¯ v , we obtain the system equation

<!-- formula-not-decoded -->

Here the coe ffi cient b relates to a number of problem characteristics including the weight of the vehicle and the road conditions. The cost function expresses our desire to keep x k near zero with relatively little force.

We will apply the DP algorithm, and derive the optimal cost-to-go functions J ∗ k and optimal policy. We have

<!-- formula-not-decoded -->

and by applying Eq. (1.15), we obtain

<!-- formula-not-decoded -->

and finally, using the assumptions E ¶ w N -1 ♦ = 0, E ¶ w 2 N -1 ♦ = σ 2 , and bringing out of the minimization the terms that do not depend on u N -1 ,

<!-- formula-not-decoded -->

The expression minimized over u N -1 in the preceding equation is convex quadratic in u N -1 , so by setting to zero its derivative with respect to u N -1 ,

<!-- formula-not-decoded -->

we obtain the optimal policy for the last stage:

<!-- formula-not-decoded -->

Substituting this expression into Eq. (1.17), we obtain with a straightforward calculation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

We can now continue the DP algorithm to obtain J ∗ N -2 from J ∗ N -1 . An important observation is that J ∗ N -1 is quadratic (plus an inconsequential constant term), so with a similar calculation we can derive θ ∗ N -2 and J ∗ N -2 in closed form, as a linear and a quadratic (plus constant) function of x N -2 , respectively. This process can be continued going backwards, and it can be verified by induction that for all k , we obtain the optimal policy and optimal cost-to-go function in the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

and the sequence ¶ K k ♦ is generated backwards by the equation

<!-- formula-not-decoded -->

starting from the terminal condition K N = qglyph[triangleright]

The process by which we obtained an analytical solution in this example is noteworthy. A little thought while tracing the steps of the algorithm will convince the reader that what simplifies the solution is the quadratic nature of the cost and the linearity of the system equation. Indeed, it can be shown in generality that when the system is linear and the cost is quadratic, the optimal policy and cost-to-go function are given by closed-form expressions, even for multi-dimensional linear systems (see [Ber17a], Section 3.1). The optimal policy is a linear function of the state, and the optimal cost function is a quadratic in the state plus a constant.

Another remarkable feature of this example, which can also be extended to multi-dimensional systems, is that the optimal policy does not depend on the variance of w k , and remains una ff ected when w k is replaced by its mean (which is zero in our example). This is known as certainty equivalence , and occurs in several types of problems involving a linear system and a quadratic cost; see [Ber17a], Sections 3.1 and 4.2. For example it holds even when w k has nonzero mean. For other problems, certainty equivalence can be used as a basis for problem approximation, e.g., assume that certainty equivalence holds (i.e., replace stochastic quantities by some typical values, such as their expected values) and apply exact DP to the resulting deterministic optimal control problem. This is an important part of the RL methodology, which we will discuss later in this chapter, and in more detail in Chapter 2.

Note that the linear quadratic type of problem illustrated in the preceding example is exceptional in that it admits an elegant analytical solution. Most DP problems encountered in practice require a computational solution.

## Q-Factors and Q-Learning for Stochastic Problems

Similar to the case of deterministic problems [cf. Eq. (1.9)], we can define optimal Q-factors for a stochastic problem, as the expressions that are minimized in the right-hand side of the stochastic DP equation (1.15). They are given by

<!-- formula-not-decoded -->

The optimal cost-to-go functions J * k can be recovered from the optimal Q-factors Q * k by means of

<!-- formula-not-decoded -->

and the DP algorithm can be written in terms of Q-factors as

<!-- formula-not-decoded -->

We will later be interested in approximate Q-factors, where J * k +1 in Eq. (1.20) is replaced by an approximation ˜ J k +1 . Again, the Q-factor corresponding to a state-control pair ( x k ↪ u k ) is the sum of the expected first stage cost using ( x k ↪ u k ), plus the expected cost of the remaining stages starting from the next state as estimated by the function ˜ J k +1 .

## 1.3.2 Approximation in Value Space for Stochastic DP

Generally the computation of the optimal cost-to-go functions J * k can be very time-consuming or impossible. One of the principal RL methods to deal with this di ffi culty is approximation in value space. Here approximations ˜ J k are used in place of J * k , similar to the deterministic case; cf. Eqs. (1.8) and (1.11).

## Approximation in Value Space - Use of ˜ J k in Place of J * k

At any state x k encountered at stage k , set

<!-- formula-not-decoded -->

The one-step lookahead minimization (1.21) needs to be performed only for the N states x 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N -1 that are encountered during the on-line control of the system. By contrast, exact DP requires that this type of minimization be done for every state and stage.

## The Three Approximations

When designing approximation in value space schemes, one may consider several interesting simplification ideas, which are aimed at alleviating the computational overhead. Aside from cost function approximation (use ˜ J k +1 in place of J * k +1 ), there are other possibilities. One of them is to simplify the lookahead minimization over u k ∈ U k ( x k ) [cf. Eq. (1.15)] by replacing U k ( x k ) with a suitably chosen subset of controls that are viewed as most promising based on some heuristic criterion.

In Section 1.6.7, we will discuss a related idea for control space simplification for the multiagent case where the control consists of multiple

components, u k = ( u 1 k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m k ). Then, a sequence of m single component minimizations can be used instead, with potentially enormous computational savings resulting.

Another type of simplification relates to approximations in the computation of the expected value in Eq. (1.21) by using limited Monte Carlo simulation. The Monte Carlo Tree Search method, which will be discussed in Chapter 2, Section 2.7.5, is one possibility of this type.

Still another type of expected value simplification is based on the certainty equivalence approach , to be discussed in more detail in Chapter 2, Section 2.7.2. In this approach, at stage k , we replace the future random variables w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w k + m by some deterministic values w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w k + m , such as their expected values. We may also view this as a form of problem approximation, whereby for the purpose of computing ˜ J k +1 ( x k +1 ), we 'pretend' that the problem is deterministic, with the future random quantities replaced by deterministic typical values. This is one of the most e ff ective techniques to make approximation in value space for stochastic problems computationally tractable, particularly when it is combined with multistep lookahead minimization, as we will discuss later.

Figure 1.3.2 illustrates the three approximations involved in approximation in value space for stochastic problems: cost-to-go approximation, simplified minimization, and expected value approximation . They may be designed largely independently of each other, and may be implemented with a variety of methods. Much of the discussion in this book will revolve around di ff erent ways to organize these three approximations for both cases of one-step and multistep lookahead.

As indicated in Fig. 1.3.2, an important approach for cost-to-go approximation is problem approximation , whereby the functions ˜ J k +1 in Eq. (1.21) are obtained as the optimal or nearly optimal cost functions of a simplified optimization problem, which is more convenient for computation. Simplifications may include exploiting decomposable structure, ignoring various types of uncertainties, and reducing the size of the state space. Several types of problem approximation approaches are discussed in the author's RL book [Ber19a]. A major approach is aggregation , which will be discussed in Section 3.6. In this book, problem approximation will not receive much attention, despite the fact that it can often be combined very e ff ectively with the approximation in value space methodology that is our main focus.

Another important approach for on-line cost-to-go approximation is rollout, which we discuss next. This is similar to the rollout approach for deterministic problems, discussed in Section 1.2.

## Rollout for Stochastic Problems - Truncated Rollout

In the rollout approach, we select ˜ J k +1 in Eq. (1.21) to be the cost function of a suitable base policy (perhaps with some approximation). Note that

Parametric approximation Neural nets Discretization

Multiagent Q-factor minimization

Approximate Min Approximate

Simple choices Parametric approximation Problem approximation

Certainty equivalence Monte Carlo tree search

Figure 1.3.2 Schematic illustration of approximation in value space for stochastic problems, and the three approximations involved in its design. Typically the approximations can be designed independently of each other, and with a variety of approaches. There are also multistep lookahead versions of approximation in value space, which will be discussed later.

<!-- image -->

any policy can be used on-line as base policy, including policies obtained by a sophisticated o ff -line procedure, using for example neural networks and training data. The rollout algorithm has a cost improvement property, whereby it yields an improved cost relative to its underlying base policy. We will discuss this property and some conditions under which it is guaranteed to hold in Chapter 2.

A major variant of rollout is truncated rollout , which combines the use of one-step optimization, simulation of the base policy for a certain number of steps m , and then adds an approximate cost ˜ J k + m +1 ( x k + m +1 ) to the cost of the simulation, which depends on the state x k + m +1 obtained at the end of the rollout. Note that if one foregoes the use of a base policy (i.e., m = 0), one recovers as a special case the general approximation in value space scheme (1.21); see Fig. 1.3.3. Thus rollout provides an extra layer of lookahead to the one-step minimization, but this lookahead need not extend to the end of the horizon.

Note also that versions of truncated rollout with multistep lookahead minimization are possible. They will be discussed later. The terminal cost approximation is necessary in infinite horizon problems, since an infinite number of stages of the base policy rollout is impossible. However, even for finite horizon problems it may be necessary and/or beneficial to artificially truncate the rollout horizon. Generally, a large combined number of multistep lookahead minimization and rollout steps is likely to be beneficial.

## Cost Versus Q-Factor Approximations - Robustness and OnLine Replanning

Similar to the deterministic case, Q-learning involves the calculation of either the optimal Q-factors (1.20) or approximations ˜ Q k ( x k ↪ u k ). The

Possible States

Multiagent Q-factor minimization for Stages Beyond Truncation

Figure 1.3.3 Schematic illustration of truncated rollout. One-step lookahead is followed by simulation of the base policy for m steps, and an approximate cost ˜ J k + m +1 ( x k + m +1 ) is added to the cost of the simulation, which depends on the state x k + m +1 obtained at the end of the rollout. If the base policy simulation is omitted (i.e., m = 0), one recovers the general approximation in value space scheme (1.21). Truncated rollout with multistep lookahead is also possible and is discussed in some detail in Chapter 2.

<!-- image -->

approximate Q-factors may be obtained using approximation in value space schemes, and can be used to obtain approximately optimal policies through the Q-factor minimization

<!-- formula-not-decoded -->

Since it is possible to implement approximation in value space by using cost function approximations [cf. Eq. (1.21)] or by using Q-factor approximations [cf. Eq. (1.22)], the question arises which one to use in a given practical situation. One important consideration is the facility of obtaining suitable cost or Q-factor approximations. This depends largely on the problem and also on the availability of data on which the approximations can be based. However, there are some other major considerations.

In particular, the cost function approximation scheme

<!-- formula-not-decoded -->

has an important disadvantage: the expected value above needs to be computed on-line for all u k ∈ U k ( x k ) , and this may involve substantial computation . It also has an important advantage in situations where the system function f k , the cost per stage g k , or the control constraint set U k ( x k ) can change as the system is operating. Assuming that the new f k , g k , or U k ( x k ) become known to the controller at time k , on-line replanning may be used, and this may improve substantially the robustness of the approximation in

value space scheme . By comparison, the Q-factor function approximation scheme (1.22) does not allow for on-line replanning. On the other hand, for problems where there is no need for on-line replanning, the Q-factor approximation scheme may not require the on-line computation of expected values and may allow a much faster on-line computation of the minimizing control ˜ θ k ( x k ) via Eq. (1.22).

One more disadvantage of using Q-factors will emerge later, as we discuss the synergy between o ff -line training and on-line play based on Newton's method; see Section 1.5. In particular, we will interpret the cost function of the lookahead minimization policy ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ as the result of one step of Newton's method for solving the Bellman equation that underlies the DP problem, starting from the terminal cost function approximations ¶ ˜ J 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ J N ♦ . This synergy tends to be negatively a ff ected when Q-factor (rather than cost) approximations are used.

## 1.3.3 Approximation in Policy Space

The major alternative to approximation in value space is approximation in policy space , whereby we select the policy from a suitably restricted class of policies, usually a parametric class of some form. In particular, we can introduce a parametric family of policies (or approximation architecture, as we will call it in Chapter 3),

<!-- formula-not-decoded -->

where r k is a parameter, and then estimate the parameters r k using some type of training process or optimization; cf. Fig. 1.3.4.

In this section and throughout this book, we focus on selecting a policy o ff -line , possibly through training with o ff -line-collected data. There are algorithms that aim to improve parametric policies by using data that is collected on-line, but this subject is beyond our scope (see also the relevant discussion on policy gradient methods in Chapter 3).

Neural networks, described in Chapter 3, are often used to generate the parametric class of policies, in which case r k is the vector of weights/parameters of the neural network. In Chapter 3, we will also discuss methods for obtaining the training data required for obtaining the parameters r k , and we will consider several other classes of approximation architectures.

A general scheme for parametric approximation in policy space is to somehow obtain a training set, consisting of a large number of sample state-control pairs

<!-- formula-not-decoded -->

such that for each s , u s k is a 'good' control at state x s k . We can then choose the parameter r k by solving the least squares/regression problem

<!-- formula-not-decoded -->

Uncertainty System Environment Cost Control Current State

Uncertainty System Environment Cost Control Current State

(

) Approximate Q-Factor

Figure 1.3.4 Schematic illustration of parametric approximation in policy space.

<!-- image -->

A policy

<!-- formula-not-decoded -->

from a parametric class is computed o ff -line based on data, and it is used to generate the control u k = ˜ θ k ( x k ↪ r k ) on-line, when at state x k .

(possibly modified to add regularization). In particular, we may determine u s k using a human or a software 'expert' that can choose 'near-optimal' controls at given states, so ˜ θ k is trained to match the behavior of the expert. Methods of this type are commonly referred to as supervised learning in artificial intelligence.

An important approach for generating the training set ( x s k ↪ u s k ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , for the least squares training problem (1.24) is based on approximation in value space. In particular, we may use a one-step lookahead minimization of the form

<!-- formula-not-decoded -->

Here ‖ · ‖ denotes the standard quadratic Euclidean norm. It is implicitly assumed here (and in similar situations later) that the controls are members of a Euclidean space (i.e., the space of finite dimensional vectors with real-valued components) so that the distance between two controls can be measured by their normed di ff erence (randomized controls, i.e., probabilities that a particular action will be used, fall in this category). Regression problems of this type arise in the training of parametric classifiers based on data, including the use of neural networks (see Section 3.4). Assuming a finite control space, the classifier is trained using the data ( x s k ↪ u s k ) , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q↪ which are viewed as state-category pairs, and then a state x k is classified as being of 'category' ˜ θ k ( x k ↪ r k ). Parametric approximation architectures, and their training through the use of classification and regression techniques are described in Chapter 3. An important modification is to use regularized regression where a quadratic regularization term is added to the least squares objective. This term is a positive multiple of the squared deviation ‖ r -ˆ r ‖ 2 of r from some initial guess ˆ r .

where ˜ J k +1 is a suitable (separately obtained) approximation in value space. Alternatively, we may use an approximate Q-factor based minimization

<!-- formula-not-decoded -->

where ˜ Q k is a (separately obtained) Q-factor approximation. We may view this as approximation in policy space built on top of approximation in value space .

There is a significant advantage of the least squares training procedure of Eq. (1.24), and more generally approximation in policy space: once the parametrized policy ˜ θ k is obtained, the computation of controls

<!-- formula-not-decoded -->

during on-line operation of the system is often much easier compared with the lookahead minimization (1.23). For this reason, one of the major uses of approximation in policy space is to provide an approximate implementation of a known policy (no matter how obtained) for the purpose of convenient on-line use. On the negative side, such an implementation is less well suited for on-line replanning.

## Model-Free Approximation in Policy Space

There are also alternative optimization-based approaches for policy space approximation. The main idea is that once we use a vector ( r 0 ↪ r 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r N -1 ) to parametrize the policies π , the expected cost J π ( x 0 ) is parametrized as well, and can be viewed as a function of ( r 0 ↪ r 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r N -1 ). We can then optimize this cost by using a gradient-like or random search method. This is a widely used approach for optimization in policy space, which will be discussed somewhat briefly in this book (see Section 3.5, and the RL book [Ber19a], Section 5.7).

An interesting feature of this approach is that in principle it does not require a mathematical model of the system and the cost function; a computer simulator (or availability of the real system for experimentation) su ffi ces instead. This is sometimes called a model-free implementation . The advisability of implementations of this type, particularly when they rely exclusively on simulation (i.e., without the use of any prior mathematical model knowledge), is a hotly debated and much contested issue; see for example the review paper by Alamir [Ala22].

The term 'model-free' can be confusing. In reality, there is always a model in DP/RL problem formulations . It is just a question of whether it is a mathematical model (i.e., based on equations), or a computer model (i.e., based on computer simulation or a trained neural network), or a hybrid model, (i.e., one that relies both on mathematical equations and computer software).

Target Cost Function

Figure 1.3.5 The general structure for parametric cost approximation. We approximate the target cost function J ( x ) with a member from a parametric class ˜ J ( x↪ r ) that depend on a parameter vector r . We use training data ( x s ↪ J ( x s ) ) , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , and a form of optimization that aims to find a parameter ˆ r that 'minimizes' the size of the errors J ( x s ) -˜ J ( x s ↪ ˆ r ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q .

<!-- image -->

We finally note an important conceptual di ff erence between approximation in value space and approximation in policy space. The former is primarily an on-line method (with o ff -line training used optionally to construct cost function approximations for one-step or multistep lookahead). The latter is primarily an o ff -line training method (which may be used without modification for on-line play or optionally to provide a policy for on-line rollout).

## 1.3.4 Training of Cost Function and Policy Approximations

When it comes to o ff -line constructed approximations, a major approach is based on the use of parametric approximation. Feature-based architectures and neural networks are very useful within our RL context, and will be discussed in Chapter 3, together with methods that can be used for training them.

A general structure for parametric cost function approximation is illustrated in Fig. 1.3.5. We have a target function J ( x ) that we want to

The principal role of neural networks within the context of this book is to provide the means for approximating various target functions from input-output data. This includes cost functions and Q-factors of given policies, and optimal cost-to-go functions and Q-factors; in this case the neural network is referred to as a value network (sometimes the alternative term critic network is also used). In other cases the neural network represents a policy viewed as a function from state to control, in which case it is called a policy network (the alternative term actor network is also used). The training methods for constructing the cost function, Q-factor, and policy approximations from data are mostly based on optimization and regression, and will be reviewed in Chapter 3. Further DPoriented discussions are found in many sources, including the RL books [Ber19a], [Ber20a], and the neuro-dynamic programming book [BeT96]. Machine learning books, including those describing at length neural network architectures and training are also recommended; see e.g., the recent book by Bishop and Bishop [BiB24], and the references quoted therein.

Approximating Function

approximate with a member of a parametric class of functions ˜ J ( x↪ r ) that depend on a parameter vector r (to simplify, we drop the time index, using J in place of J k ). To this end, we collect training data ( x s ↪ J ( x s ) ) , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , which we use to determine a parameter ˆ r that leads to a good 'fit' between the data J ( x s ) and the predictions ˜ J ( x s ↪ ˆ r ) of the parametrized function. This is usually done through some form of optimization that aims to minimize in some sense the size of the errors J ( x s ) -˜ J ( x s ↪ ˆ r ).

The methodological ideas for parametric cost approximation can also be used for approximation of a target policy θ with a policy from a parametric class ˜ θ ( x↪ r ). The training data may be obtained, for example, from rollout control calculations, thus enabling the construction of both value and policy networks that can be combined for use in a perpetual rollout scheme. However, there is an important di ff erence: the approximate cost values ˜ J ( x↪ r ) are real numbers, whereas the approximate policy values ˜ θ ( x↪ r ) are elements of a control space U . Thus if U consists of m dimensional vectors, ˜ θ ( x↪ r ) consists of m numerical components. In this case the parametric approximation problems for cost functions and for policies are fairly similar, and both involve continuous space approximations.

On the other hand, the case where the control space is finite, U = ¶ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ♦ . is markedly di ff erent. In this case, for any x , ˜ θ ( x↪ r ) consists of one of the m possible controls u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m . This ushers a connection with traditional classification schemes, whereby objects x are classified as belonging to one of the categories u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m , so that θ ( x ) defines the category of x , and can be viewed as a classifier . Some of the most prominent classification schemes actually produce randomized outcomes, i.e., x is associated with a probability distribution

<!-- formula-not-decoded -->

which is a randomized policy in our policy approximation context; see Fig. 1.3.6. This is done usually for reasons of algorithmic convenience, since many optimization methods, including least squares regression, require that the optimization variables are continuous. In this case, the randomized policy (1.25) can be converted to a nonrandomized policy using a maximization operation: associate x with the control of maximum probability (cf. Fig. 1.3.6),

<!-- formula-not-decoded -->

The use of classification methods for approximation in policy space will be discussed in Chapter 3 (Section 3.4).

## 1.4 INFINITE HORIZON PROBLEMS - AN OVERVIEW

We will now provide an outline of infinite horizon stochastic DP with an emphasis on its aspects that relate to our RL/approximation methods. We

Figure 1.3.6 A general structure for parametric policy approximation for the case where the control space is finite, U = ¶ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ♦ , and its relation to a classification scheme. It produces a randomized policy of the form (1.25), which is converted to a nonrandomized policy through the maximization operation (1.26).

<!-- image -->

will deal primarily with infinite horizon stochastic problems, where we aim to minimize the total cost over an infinite number of stages, given by

<!-- formula-not-decoded -->

see Fig. 1.4.1. Here, J π ( x 0 ) denotes the cost associated with an initial state x 0 and a policy π = ¶ θ 0 ↪ θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ , and α is a scalar in the interval (0 ↪ 1]. The functions g and f that define the cost per stage and the system equation

<!-- formula-not-decoded -->

do not change from one stage to the next. The stochastic disturbances, w 0 ↪ w 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] , have a common probability distribution P ( · ♣ x k ↪ u k ).

When α is strictly less that 1, it has the meaning of a discount factor , and its e ff ect is that future costs matter to us less than the same costs incurred at the present time. Among others, a discount factor guarantees that the limit defining J π ( x 0 ) exists and is finite (assuming that the range of values of the stage cost g is bounded). This is a nice mathematical property that makes discounted problems analytically and algorithmically tractable.

Thus, by definition, the infinite horizon cost of a policy is the limit of its finite horizon costs as the horizon tends to infinity. The three types of problems that we will focus on are:

- (a) Stochastic shortest path problems (SSP for short). Here, α = 1 but there is a special cost-free termination state; once the system reaches that state it remains there at no further cost. In some types of problems, the termination state may represent a goal state that we are trying to reach at minimum cost, while in others it may be a state that we are trying to avoid for as long as possible. We will mostly assume a problem structure such that termination is inevitable under all policies. Thus the horizon is in e ff ect finite, but its length is random and may be a ff ected by the policy being used. A significantly

Figure 1.4.1 Illustration of an infinite horizon problem. The system and cost per stage are stationary, except for the use of a discount factor α . If α = 1, there is typically a special cost-free termination state that we aim to reach.

<!-- image -->

more complicated type of SSP problems, which we will discuss selectively, arises when termination can be guaranteed only for a subset of policies, which includes all optimal policies. Some common types of SSP belong to this category, including deterministic shortest path problems that involve graphs with cycles.

- (b) Discounted problems . Here, α &lt; 1 and there need not be a termination state. However, we will see that a discounted problem with a finite number of states can be readily converted to an SSP problem. This can be done by introducing an artificial termination state to which the system moves with probability 1 -α at every state and stage, thus making termination inevitable. As a result, algorithms and analysis for SSP problems can be easily adapted to discounted problems; the DP textbook [Ber17a] provides a detailed account of this conversion, and an accessible introduction to discounted and SSP problems with a finite number of states.
- (c) Deterministic nonnegative cost problems . Here, the disturbance w k takes a single known value. Equivalently, there is no disturbance in the system equation and the cost expression, which now take the form

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

We assume further that there is a cost-free and absorbing termination state t , and that we have

<!-- formula-not-decoded -->

/negationslash and g ( t↪ u ) = 0 for all u ∈ U ( t ). This type of structure expresses the objective to reach or approach t at minimum cost, a classical control problem. An extensive analysis of the undiscounted version of this problem was given in the author's paper [Ber17b].

Discounted stochastic problems with a finite number of states [also referred to as discounted MDP (abbreviation for Markovian Decision Problem) ] are very common in the DP/RL literature, particularly because of

their benign analytical and computational nature. Moreover, there is a widespread belief that discounted MDP can be used as a universal model, i.e., that in practice any other kind of problem (e.g., undiscounted problems with a termination state and/or a continuous state space) can be painlessly converted to a discounted MDP with a discount factor that is close enough to 1. This is questionable, however, for a number of reasons:

- (a) Deterministic models are common as well as natural in many practical contexts (including discrete optimization/integer programming problems), so to convert them to MDP does not make sense.
- (b) The conversion of a continuous-state problem to a finite-state problem through some kind of discretization involves mathematical subtleties that can lead to serious practical/algorithmic complications. In particular, the character of the optimal solution may be seriously distorted by converting to a discounted MDP through some form of discretization, regardless of how fine the discretization is.
- (c) For some practical shortest path contexts it is essential that the termination state is ultimately reached. However, when a discount factor α is introduced in such a problem, the character of the problem may be fundamentally altered. In particular, the threshold for an appropriate value of α may be very close to 1 and may be unknown in practice. For a simple example consider a shortest path problem with states 1 and 2 plus a termination state t . From state 1 we can go to state 2 at cost 0, from state 2 we can go to either state 1 at a small cost /epsilon1 &gt; 0 or to the termination state at a substantial cost C &gt; 0. The optimal policy over an infinite horizon is to go from 1 to 2 and from 2 to t . Suppose now that we approximate the problem by introducing a discount factor α ∈ (0 ↪ 1). Then it can be shown that if α &lt; 1 -/epsilon1 glyph[triangleleft]C , it is optimal to move indefinitely around the cycle 1 → 2 → 1 → 2 and never reach t , while for α &gt; 1 -/epsilon1 glyph[triangleleft]C the shortest path 2 → 1 → t will be obtained. Thus the solution of the discounted problem varies discontinuously with α : it changes radically at some threshold, which in general may be unknown.

An important class of problems that we will consider in some detail in this book is finite-state deterministic problems with a large number of states. Finite horizon versions of these problems include challenging discrete optimization problems, whose exact solution is practically impossible. An important fact to keep in mind is that we can transform such problems to infinite horizon SSP problems with a termination state at the end of the horizon, so that the conceptual framework of the present section applies. The approximate solution of discrete optimization problems by RL methods, and particularly by rollout, will be considered in Chapter 2, and has been discussed at length in the books [Ber19a] and [Ber20a].

## 1.4.1 Infinite Horizon Methodology

There are several analytical and computational issues regarding our infinite horizon problems. Many of them revolve around the relation between the optimal cost function J * of the infinite horizon problem and the optimal cost functions of the corresponding N -stage problems.

In particular, let J N ( x ) denote the optimal cost of the problem involving N stages, initial state x , cost per stage g ( x↪ u↪ w ), and zero terminal cost. This cost is generated after N iterations of the algorithm

<!-- formula-not-decoded -->

starting from J 0 ( x ) ≡ 0. The algorithm (1.31) is known as the value iteration algorithm (VI for short). Since the infinite horizon cost of a given policy is, by definition, the limit of the corresponding N -stage costs as N →∞ , it is natural to speculate that:

- (a) The optimal infinite horizon cost is the limit of the corresponding N -stage optimal costs as N →∞ ; i.e.,

<!-- formula-not-decoded -->

for all states x .

- (b) The following equation should hold for all states x ,

<!-- formula-not-decoded -->

This is obtained by taking the limit as N →∞ in the VI algorithm (1.31) using Eq. (1.32). The preceding equation, called Bellman's equation , is really a system of equations (one equation per state x ), which has as solution the optimal costs-to-go of all the states.

- (c) If θ ( x ) attains the minimum in the right-hand side of the Bellman equation (1.33) for each x , then the policy ¶ θ↪ θ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ should be optimal. This type of policy is called stationary , and for simplicity it is denoted by θ .

This is just the finite horizon DP algorithm of Section 1.3.1, except that we have reversed the time indexing to suit our infinite horizon context. In particular, consider the N -stages problem and let V N -k ( x ) be the optimal cost-to-go starting at x with k stages to go, and with terminal cost equal to 0. Applying DP, we have for all x ,

<!-- formula-not-decoded -->

By defining J k ( x ) = V N -k ( x ) glyph[triangleleft] α N -k , we obtain the VI algorithm (1.31).

- (d) The cost function J θ of a stationary policy θ satisfies

<!-- formula-not-decoded -->

We can view this as just the Bellman equation (1.33) for a di ff erent problem, where for each x , the control constraint set U ( x ) consists of just one control, namely θ ( x ). Moreover, we expect that J θ is obtained in the limit by the VI algorithm:

<!-- formula-not-decoded -->

where J θ↪N is the N -stage cost function of θ generated by

<!-- formula-not-decoded -->

starting from J θ↪ 0 ( x ) ≡ 0 or some other initial condition; cf. Eqs. (1.31)-(1.32).

All four of the preceding results can be shown to hold for finitestate discounted problems, and also for finite-state SSP problems under reasonable assumptions. The results also hold for infinite-state discounted problems, provided the cost per stage function g is bounded over the set of possible values of ( x↪ u↪ w ), in which case we additionally can show that J * is the unique solution of Bellman's equation. The VI algorithm is also valid under these conditions, in the sense that J k → J * , even if the initial function J 0 is nonzero. The motivation for a di ff erent choice of J 0 is faster convergence to J * ; generally the convergence is faster as J 0 is chosen closer to J * . The associated mathematical proofs can be found in several sources, e.g., [Ber12], Chapter 1, or [Ber19a], Chapter 4.

It is important to note that for infinite horizon problems, there are additional important algorithms that are amenable to approximation in value space. Approximate policy iteration, Q-learning, temporal di ff erence methods, linear programming, and their variants are some of these; see the RL books [Ber19a], [Ber20a]. For this reason, in the infinite horizon case, there is a richer set of algorithmic options for approximation in value space, despite the fact that the associated mathematical theory is more complex. In this book, we will only discuss approximate forms and variations of the policy iteration algorithm, which we describe next.

For undiscounted problems and discounted problems with unbounded cost per stage, we may still adopt the four preceding results as a working hypothesis. However, we should also be aware that exceptional behavior is possible under unfavorable circumstances, including nonuniqueness of solution of Bellman's equation, and nonconvergence of the VI algorithm to J ∗ from some initial conditions; see the books [Ber12], [Ber22b].

Policy Evaluation Policy Improvement Rollout Policy ˜

<!-- image -->

Policy Evaluation Policy Improvement Rollout Policy ˜

Figure 1.4.2 Schematic illustration of PI as repeated rollout. It generates a sequence of policies, with each policy θ in the sequence being the base policy that generates the next policy ˜ θ in the sequence as the corresponding rollout policy. This rollout policy is used as the base policy in the subsequent iteration.

## Policy Iteration

A major infinite horizon algorithm is policy iteration (PI for short). We will argue that PI, together with its variations, forms the foundation for self-learning in RL, i.e., learning from data that is self-generated (from the system itself as it operates) rather than from data supplied from an external source. Figure 1.4.2 describes the method as repeated rollout, and indicates that each of its iterations consists of two phases:

- (a) Policy evaluation , which computes the cost function J θ of the current (or base) policy θ . One possibility is to solve the corresponding Bellman equation

<!-- formula-not-decoded -->

cf. Eq. (1.34). However, the value J θ ( x ) for any x can also be computed by Monte Carlo simulation, by averaging over many randomly generated trajectories the cost of the policy starting from x .

- (b) Policy improvement , which computes the 'improved' (or rollout) policy ˜ θ using the one-step lookahead minimization

<!-- formula-not-decoded -->

We call ˜ θ 'improved policy' because we can generally prove that

<!-- formula-not-decoded -->

This cost improvement property will be shown in Chapter 2, Section 2.7, and can be used to show that PI produces an optimal policy in a finite number of iterations under favorable conditions (for example for finitestate discounted problems; see the DP books [Ber12], [Ber17a], or the RL book [Ber19a]).

The rollout algorithm in its pure form is just a single iteration of the PI algorithm . It starts from a given base policy θ and produces the rollout policy ˜ θ . It may be viewed as approximation in value space with one-step lookahead that uses J θ as terminal cost function approximation. It has the advantage that it can be applied on-line by computing the needed values of J θ ( x ) by simulation. By contrast, approximate forms of PI for challenging problems, involving for example neural network training, can only be implemented o ff -line.

## 1.4.2 Approximation in Value Space - Infinite Horizon

The approximation in value space approach that we discussed in connection with finite horizon problems can be extended in a natural way to infinite horizon problems. Here in place of J * , we use an approximation ˜ J , and generate at any state x , a control ˜ θ ( x ) by the one-step lookahead minimization

<!-- formula-not-decoded -->

This minimization yields a stationary policy ¶ ˜ θ↪ ˜ θ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ , with cost function denoted J ˜ θ [i.e., J ˜ θ ( x ) is the total infinite horizon discounted cost obtained when using ˜ θ starting at state x ]; see Fig. 1.4.3. Note that when ˜ J = J * , the one-step lookahead policy attains the minimum in the Bellman equation (1.33) and is expected to be optimal. This suggests that one should try to use ˜ J as close as possible to J * , which is generally true as we will argue later.

Naturally an important goal to strive for is that J ˜ θ is close to J * in some sense. However, for classical control problems, which involve steering and maintaining the state near a desired reference state (e.g., problems with a cost-free and absorbing terminal state, and positive cost for all other states), stability of ˜ θ may be a principal objective . In this book, we will discuss stability issues primarily for this one class of problems, and we will consider the policy ˜ θ to be stable if J ˜ θ is real-valued , i.e.,

<!-- formula-not-decoded -->

Selecting ˜ J so that ˜ θ is stable is a question of major interest for some application contexts, such as model predictive and adaptive control, and will be discussed in the next section within the limited context of linear quadratic problems.

## /lscript -Step Lookahead

An important extension of one-step lookahead minimization is /lscript -step lookahead , whereby at a state x k we minimize the cost of the first /lscript &gt; 1 stages

¿ An lins olor citl multiaton 1od

At x

First Step

"Future"

minuEU(x) E{g(x, u, w) + aJ (f(x, 2, w)) }

One-Step Lookahead

First l Steps

Multistep Lookahead

"Future"

Figure 1.4.3 Schematic illustration of approximation in value space with one-step and /lscript -step lookahead minimization for infinite horizon problems. In the former case, the minimization yields at state x a control ˜ u , which defines the one-step lookahead policy ˜ θ via

<!-- image -->

<!-- formula-not-decoded -->

In the latter case, the minimization yields a control ˜ u k policies ˜ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ k + /lscript -1 . The control ˜ u k is applied at x k while the remaining sequence ˜ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ k + /lscript -1 is discarded. The control ˜ u k defines the /lscript -step lookahead policy ˜ θ .

with the future costs approximated by a function ˜ J (see the bottom half of Fig. 1.4.3). This minimization yields a control ˜ u k and a sequence ˜ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ k + /lscript -1 . The control ˜ u k is applied at x k , and defines the /lscript -step lookahead policy ˜ θ via ˜ θ ( x k ) = ˜ u k , while ˜ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ k + /lscript -1 are discarded. Actually, we may view /lscript -step lookahead minimization as the special case of its one-step counterpart where the lookahead function is the optimal cost function of an ( /lscript -1)-stage DP problem with a terminal cost ˜ J ( x k + /lscript ) on the state x k + /lscript obtained after /lscript -1 stages.

The motivation for /lscript -step lookahead minimization is that by increasing the value of /lscript , we may require a less accurate approximation ˜ J to obtain good performance . Otherwise expressed, for the same quality of cost function approximation, better performance may be obtained as /lscript becomes larger. This will be explained visually later, using the formalism of Newton's method in Section 1.5. In particular, for AlphaZero chess, long multistep lookahead is critical for good on-line performance. Another motivation for multistep lookahead is to enhance the stability properties of the gener-

On-line play with multistep lookahead minimization (and possibly truncated rollout) is referred to by a number of di ff erent names in the RL literature, such as on-line search , predictive learning , learning from prediction , etc; in the model predictive control literature the combined interval of lookahead minimization and truncated rollout is referred as the prediction interval .

At Xk min

Uk, Hk+1,., Hk+e-1

Min Approximation

Figure 1.4.4 Approximation in value space with one-step lookahead for infinite horizon problems. There are three potential areas of approximation, which can be considered independently of each other: optimal cost approximation, expected value approximation, and minimization approximation.

<!-- image -->

ated on-line policy , as we will discuss later in Section 1.5. On the other hand, solving the multistep lookahead minimization problem, instead of the one-step lookahead counterpart of Eq. (1.36), is more time consuming.

## The Three Approximations: Optimal Cost, Expected Value, and Lookahead Minimization Approximations

There are three potential areas of approximation for infinite horizon problems: optimal cost approximation, expected value approximation, and minimization approximation; cf. Fig. 1.4.4. They are similar to their finite horizon counterparts that we discussed in Section 1.3.2. In particular, we have potentially:

- (a) A terminal cost approximation ˜ J of the optimal cost function J * : A major advantage of the infinite horizon context is that only one approximate cost function ˜ J is needed, rather than the N functions ˜ J 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ J N of the N -step horizon case.
- (b) An approximation of the expected value operation : This operation can be very time consuming. It may be simplified in various ways. For example some of the random quantities w k ↪ w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w k + /lscript -1 appearing in the /lscript -step lookahead minimization may be replaced by deterministic quantities; this is another example of the certainty equivalence approach , which we discussed in Section 1.3.2.
- (c) A simplification of the minimization operation : For example in multiagent problems the control consists of multiple components,

<!-- formula-not-decoded -->

with each component u i chosen by a di ff erent agent/decision maker. In this case the size of the control space can be enormous, but it

can be simplified in ways that will be discussed later (e.g., choosing components sequentially, one-agent-at-a-time). This will form the core of our approach to multiagent problems; see Section 1.6.7 and Chapter 2, Section 2.9.

We will next describe briefly various approaches for selecting the terminal cost function approximation.

## Constructing Terminal Cost Approximations for On-Line Play

A major issue in value space approximation is the construction of a suitable approximate cost function ˜ J . This can be done in many di ff erent ways, giving rise to some of the principal RL methods.

For example, ˜ J may be constructed with sophisticated o ff -line training methods. Alternatively, the approximate values ˜ J ( x ) may be obtained online as needed with truncated rollout, by running an o ff -line obtained policy for a suitably large number of steps, starting from x , and supplementing it with a suitable, perhaps primitive, terminal cost approximation.

For orientation purposes, let us describe briefly four broad types of approximation. We will return to these approaches later, and we also refer to the RL and approximate DP literature for more detailed discussions.

- (a) O ff -line problem approximation : Here the function ˜ J is computed o ff -line as the optimal or nearly optimal cost function of a simplified optimization problem, which is more convenient for computation. Simplifications may include exploiting decomposable structure, reducing the size of the state space, neglecting some of the constraints, and ignoring various types of uncertainties. For example we may consider using as ˜ J the cost function of a related deterministic problem, obtained through some form of certainty equivalence approximation, thus allowing computation of ˜ J by gradient-based optimal control methods or shortest path-type methods.

A major type of problem approximation method is aggregation , described in Section 3.6, and in the books [Ber12], [Ber19a] and papers [Ber18a], [Ber18b]. Aggregation provides a systematic procedure to simplify a given problem. A principal example is to group states together into a relatively small number of subsets, called aggregate states. The optimal cost function of the simpler aggregate problem is computed by exact DP methods, possibly involving the use of simulation. This cost function is then used to provide an approximation ˜ J to the optimal cost function J * of the original problem, using some form of interpolation.

- (b) On-line simulation : This possibility arises in rollout algorithms for stochastic problems, where we use Monte-Carlo simulation and some suboptimal policy θ (the base policy) to compute (whenever needed) values ˜ J ( x ) that are exactly or approximately equal to J θ ( x ). The

policy θ may be obtained by any method, e.g., one based on heuristic reasoning (such as in the case of the traveling salesman Example 1.2.3), or o ff -line training based on a more principled approach, such as approximate policy iteration or approximation in policy space. Note that while simulation is time-consuming, it is uniquely wellsuited for the use of parallel computation. Moreover, it can be simplified through the use of certainty equivalence approximations.

- (c) On-line approximate optimization . This approach involves the solution of a suitably constructed shorter horizon version of the problem, with a simple terminal cost approximation. It can be viewed as either approximation in value space with multistep lookahead, or as a form of rollout algorithm. It is often used in model predictive control (MPC).
- (d) Parametric cost approximation , where ˜ J is obtained from a given parametric class of functions J ( x↪ r ), where r is a parameter vector, selected by a suitable algorithm. The parametric class typically involves prominent characteristics of x called features , which can be obtained either through insight into the problem at hand, or by using training data and some form of neural network (see Chapter 3).

Such methods include approximate forms of PI, as discussed in Section 1.1 in connection with chess and backgammon. The policy evaluation portion of the PI algorithm can be done by approximating the cost function of the current policy using an approximation architecture such as a neural network (see Chapter 3). It can also be done with stochastic iterative algorithms such as TD( λ ), LSPE( λ ), and LSTD( λ ), which are described in the DP book [Ber12] and the RL book [Ber19a]. These methods are somewhat peripheral to our course, and will not be discussed at any length. We note, however, that approximate PI methods do not just yield a parametric approximate cost function J ( x↪ r ), but also a suboptimal policy, which can be improved on-line by using (possibly truncated) rollout.

Aside from approximate PI, parametric approximate cost functions J ( x↪ r ) may be obtained o ff -line with methods such as Q-learning, linear programming, and aggregation methods, which are also discussed in the books [Ber12] and [Ber19a].

Let us also mention that for problems with special structure, ˜ J may be chosen so that the one-step lookahead minimization (1.36) is facilitated. In fact, under favorable circumstances, the lookahead minimization may be carried out in closed form. An example is when the system is nonlinear, but the control enters linearly in the system equation and quadratically in the cost function, while the terminal cost approximation is quadratic. Then the one-step lookahead minimization can be carried out analytically, because it involves a function that is quadratic in u .

## From O ff -Line Training to On-Line Play

Generally o ff -line training will produce either just a cost approximation (as in the case of TD-Gammon), or just a policy (as for example by some approximation in policy space/policy gradient approach), or both (as in the case of AlphaZero). We have already discussed in this section one-step lookahead and multistep lookahead schemes to implement on-line approximation in value space using ˜ J ; cf. Fig. 1.4.3. Let us now consider some additional possibilities, which involve the use of a policy θ that has been obtained o ff -line (possibly in addition to a terminal cost approximation). Here are some of the main possibilities:

- (a) Given a policy θ that has been obtained o ff -line, we may use as terminal cost approximation ˜ J the cost function J θ of the policy. For the case of one-step lookahead, this requires a policy evaluation operation, and can be done on-line, by computing (possibly by simulation) just the values of

<!-- formula-not-decoded -->

that are needed [cf. Eq. (1.36)]. For the case of /lscript -step lookahead, the values

<!-- formula-not-decoded -->

for all states x k + /lscript that are reachable in /lscript steps starting from x k are needed. This is the simplest form of rollout, and only requires the o ff -line construction of the policy θ .

- (b) Given a terminal cost approximation ˜ J that has been obtained o ff -line, we may use it on-line to compute fast when needed the controls of a corresponding one-step or multistep lookahead policy ˜ θ . The policy ˜ θ can in turn be used for rollout as in (a) above. In a truncated variation of this scheme, we may also use ˜ J to approximate the tail end of the rollout process (an example of this is the rollout-based TD-Gammon algorithm).
- (c) Given a policy θ and a terminal cost approximation ˜ J , we may use them together in a truncated rollout scheme, whereby the tail end of the rollout with θ is approximated using the cost approximation ˜ J . This is similar to the truncated rollout scheme noted in (b) above, except that the policy θ is computed o ff -line rather than on-line using ˜ J and one-step or multistep lookahead.

The preceding three possibilities are the principal ones for using the results of o ff -line training within on-line play schemes. Naturally, there are variations where additional information is computed o ff -line to facilitate and/or expedite the on-line play algorithm. As an example, in MPC, in addition to a terminal cost approximation, a target tube may need to be computed o ff -line in order to guarantee that some state constraints can

be satisfied on-line; see the discussion of MPC in Section 1.6.9. Other examples of this type will be noted in the context of specific applications.

Finally, let us note that while we have emphasized approximation in value space with cost function approximation, our discussion applies to Q-factor approximation, involving functions

<!-- formula-not-decoded -->

The corresponding one-step lookahead scheme has the form

<!-- formula-not-decoded -->

cf. Eq. (1.36). The second term on the right in the above equation represents the cost function approximation

<!-- formula-not-decoded -->

The use of Q-factors is common in the 'model-free' case where a computer simulator is used to generate samples of w , and corresponding values of g and f . Then, having obtained ˜ Q through o ff -line training, the one-step lookahead minimization in Eq. (1.37) must be performed on-line with the use of the simulator.

## 1.4.3 Understanding Approximation in Value Space

We will now discuss some of our aims as we try to get insight into the process of approximation in value space. Clearly, it makes sense to approximate J * with a function ˜ J that is as close as possible to J * . However, we should also try to understand quantitatively the relation between ˜ J and J ˜ θ , the cost function of the resulting one-step lookahead (or multistep lookahead) policy ˜ θ . Interesting questions in this regard are the following:

- (a) How is the quality of the lookahead policy ˜ θ a ff ected by the quality of the o ff -line training? A related question is how much should we care about improving ˜ J through a longer and more sophisticated training process, for a given approximation architecture? A fundamental fact that provides a lot of insight in this respect is that J ˜ θ is the result of a step of Newton's method that starts at ˜ J and is applied to the Bellman Eq. (1.33) . This will be the focus of our discussion in the next section, and has been a major point in the narrative of the author's books, [Ber20a] and [Ber22a].

A related fact is that in approximation in value space with multistep lookahead, J ˜ θ is the result of a step of Newton's method that starts at the function obtained by applying multiple value iterations to ˜ J .

- (b) How do simplifications in the multistep lookahead implementation affect J ˜ θ ? The Newton step interpretation of approximation in value space leads to an important insight into the special character of the initial step of the multistep lookahead. In particular, it is only the first step that acts as the Newton step, and needs to be implemented with precision . The subsequent steps are value iterations, which only serve to enhance the quality of the starting point of the Newton step, and hence their precise implementation is not critical .

This idea suggests that simplifications of the lookahead steps after the first can be implemented with relatively small (if any) performance loss for the multistep lookahead policy. Important examples of such simplifications are the use of certainty equivalence (Sections 1.6.9, 2.7.2, 2.8.3), and forms of pruning of the lookahead tree (Section 2.4). In practical terms, simplifications after the first step of the multistep lookahead can save a lot of on-line computation, which can be fruitfully invested in extending the length of the lookahead.

- (c) When is ˜ θ stable? The question of stability is very important in many control applications where the objective is to keep the state near some reference point or trajectory. Indeed, in such applications, stability is the dominant concern, and optimality is secondary by comparison. Among others, here we are interested to characterize the set of terminal cost approximations ˜ J that lead to a stable ˜ θ .
- (d) How does the length of lookahead minimization or the length of the truncated rollout a ff ect the stability and quality of the multistep lookahead policy ˜ θ ? While it is generally true that the length of lookahead has a beneficial e ff ect on quality, it turns out that it also has a beneficial e ff ect on the stability properties of the multistep lookahead policy, and we are interested in the mechanism by which this occurs.

In what follows we will be keeping in mind these questions. In particular, in the next section, we will discuss them in the context of the simple and convenient linear quadratic problem. Our conclusions, however, hold within a far more general context with the aid of the abstract DP formalism; see the author's books [Ber20a] and [Ber22a] for a broader presentation and analysis, which address these questions in greater detail and generality.

## 1.5 NEWTON'SMETHOD-LINEARQUADRATICPROBLEMS

We will now aim to understand the character of the Bellman equation, approximation in value space, and the VI and PI algorithms within the context of an important deterministic problem. This is the classical continuous-spaces problem where the system is linear, with no control constraints, and the cost function is nonnegative quadratic. While this prob-

lem can be solved analytically, it provides a uniquely insightful context for understanding visually the Bellman equation and its algorithmic solution, both exactly and approximately.

In its general form, the problem deals with the system

<!-- formula-not-decoded -->

where x k and u k are elements of the Euclidean spaces /Rfractur n and /Rfractur m , respectively, A is an n × n matrix, and B is an n × m matrix. It is assumed that there are no control constraints. The cost per stage is quadratic of the form

<!-- formula-not-decoded -->

where Q and R are positive definite symmetric matrices of dimensions n × n and m × m , respectively (all finite-dimensional vectors in this work are viewed as column vectors, and a prime denotes transposition). The analysis of this problem is well known and is given with proofs in several control theory texts, including the author's DP books [Ber17a] and [Ber12].

In what follows, we will focus for simplicity only on the one-dimensional version of the problem, where the system has the form

<!-- formula-not-decoded -->

/negationslash cf. Example 1.3.1. Here the state x k and the control u k are scalars, and the coe ffi cients a and b are also scalars, with b = 0. The cost function is undiscounted and has the form

<!-- formula-not-decoded -->

where q and r are positive scalars. The one-dimensional case allows a convenient and insightful analysis of the algorithmic issues that are central for our purposes. This analysis generalizes to multidimensional linear quadratic problems and beyond, but requires a more demanding mathematical treatment.

## The Riccati Equation and its Justification

The analytical results for our problem may be obtained by taking the limit in the results derived in the finite horizon Example 1.3.1, as the horizon length tends to infinity. In particular, we can show that the optimal cost function is expected to be quadratic of the form

<!-- formula-not-decoded -->

where the scalar K ∗ solves the equation

<!-- formula-not-decoded -->

with F defined by where

<!-- formula-not-decoded -->

This is the limiting form of Eq. (1.19).

Moreover, the optimal policy is linear of the form

<!-- formula-not-decoded -->

where L ∗ is the scalar given by

<!-- formula-not-decoded -->

To justify Eqs. (1.41)-(1.44), we show that J * as given by Eq. (1.40), satisfies the Bellman equation

<!-- formula-not-decoded -->

and that θ ∗ ( x ), as given by Eqs. (1.43)-(1.44), attains the minimum above for every x when J = J * . Indeed for any quadratic cost function J ( x ) = Kx 2 with K ≥ 0, the minimization in Bellman's equation (1.45) is written as

<!-- formula-not-decoded -->

Thus it involves minimization of a positive definite quadratic in u and can be done analytically. By setting to 0 the derivative with respect to u of the expression in braces in Eq. (1.46), we obtain

<!-- formula-not-decoded -->

so the minimizing control and corresponding policy are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By substituting this control, the minimized expression (1.46) takes the form

<!-- formula-not-decoded -->

After straightforward algebra, using Eq. (1.48) for L K , it can be verified that this expression is written as F ( K ) x 2 , with F given by Eq. (1.42). Thus when J ( x ) = Kx 2 , the Bellman equation (1.45) takes the form

<!-- formula-not-decoded -->

or equivalently K = F ( K ) [cf. Eq. (1.41)].

In conclusion, when restricted to quadratic functions J ( x ) = Kx 2 with K ≥ 0, the Bellman equation (1.45) is equivalent to the equation

<!-- formula-not-decoded -->

We refer to this equation as the Riccati equation and to the function F as the Riccati operator . ‡ Moreover, the policy corresponding to K ∗ , as per Eqs. (1.47)-(1.48), attains the minimum in Bellman's equation, and is given by Eqs. (1.43)-(1.44).

The Riccati equation can be visualized and solved graphically as illustrated in Fig. 1.5.1. As shown in the figure, the quadratic coe ffi cient K ∗ that corresponds to the optimal cost function J * [cf. Eq. (1.40)] is the unique solution of the Riccati equation K = F ( K ) within the nonnegative real line.

## The Riccati Equation for a Stable Linear Policy

We can also characterize the cost function of a policy θ that is linear of the form θ ( x ) = Lx , and is also stable, in the sense that the scalar L satisfies ♣ a + bL ♣ &lt; 1, so that the corresponding closed-loop system

<!-- formula-not-decoded -->

is stable (its state x k converges to 0 as k →∞ ). In particular, we can show that its cost function has the form

<!-- formula-not-decoded -->

This is an algebraic form of the Riccati di ff erential equation, which was invented in its one-dimensional form by count Jacopo Riccati in the 1700s, and has played an important role in control theory. It has been studied extensively in its di ff erential and di ff erence matrix versions; see the book by Lancaster and Rodman [LR95], and the paper collection by Bittanti, Laub, and Willems [BLW91], which also includes a historical account by Bittanti [Bit91] of Riccati's remarkable life and accomplishments.

‡ The Riccati operator is a special case of the Bellman operator , denoted by T , which transforms a function J into the right side of Bellman's equation:

<!-- formula-not-decoded -->

Thus the Bellman operator T transforms a function J of x into another function TJ also of x . Bellman operators allow a succinct abstract description of the problem's data, and are fundamental in the theory of abstract DP (see the author's monographs [Ber22a] and [Ber22b]). We may view the Riccati operator as the restriction of the Bellman operator to the subspace of quadratic functions of x .

arK

7+ 62K +

ar + 9

45° Line

Riccati Operator

Figure 1.5.1 Graphical construction of the solutions of the Riccati equation (1.41)-(1.42) for the linear quadratic problem. The optimal cost function is J ∗ ( x ) = K ∗ x 2 , where the scalar K ∗ solves the fixed point equation K = F ( K ) ↪ with F being the Riccati operator given by

<!-- image -->

<!-- formula-not-decoded -->

Note that F is concave and monotonically increasing in the interval ( -rglyph[triangleleft]b 2 ↪ ∞ ) and 'flattens out' as K → ∞ , as shown in the figure. The quadratic Riccati equation K = F ( K ) also has another solution, denoted by ¯ K , which is negative and therefore of no interest.

where K L solves the equation

<!-- formula-not-decoded -->

with F L defined by

<!-- formula-not-decoded -->

This equation is called the Riccati equation for the stable policy θ ( x ) = Lx . It is illustrated in Fig. 1.5.2, and it is linear, with linear coe ffi cient ( a + bL ) 2 that is strictly less than 1. Hence the line that represents the graph of F L intersects the 45-degree line at a unique point, which defines the quadratic cost coe ffi cient K L .

The Riccati equation (1.50)-(1.51) for θ ( x ) = Lx may be justified by verifying that it is in fact the Bellman equation for θ ,

<!-- formula-not-decoded -->

62

+9

Unstable L

a + 6L| &gt; 1

0

FI(K) for L Corresponding to an Unstable and a Stable Policy

Stable L

a +6L &lt;1

<!-- image -->

K

L

Figure 1.5.2 Illustration of the construction of the cost function of a linear policy θ ( x ) = Lx , which is stable, i.e., ♣ a + bL ♣ &lt; 1 glyph[triangleright] The cost function J θ ( x ) has the form

<!-- formula-not-decoded -->

with K L obtained as the unique solution of the linear equation K = F L ( K ) ↪ where

<!-- formula-not-decoded -->

/negationslash is the Riccati equation operator corresponding to θ ( x ) = Lx . If θ is not stable, i.e., ♣ a + bL ♣ ≥ 1 ↪ we have J θ ( x ) = ∞ for all x = 0, but the equation has K = F L ( K ) still has a solution that is of no interest within our context.

/negationslash

[cf. Eq. (1.34)], restricted to quadratic functions of the form J ( x ) = Kx 2 . We note, however, that J θ ( x ) = K L x 2 is the solution of the Riccati equation (1.50)-(1.51) only when θ ( x ) = Lx is stable. If θ is not stable, i.e., ♣ a + bL ♣ ≥ 1, then (since q &gt; 0 and r &gt; 0) we have J θ ( x ) = ∞ for all x = 0. Then, the Riccati equation (1.50)-(1.51) is still defined, but its solution is negative and is of no interest within our context.

## Value Iteration

The VI algorithm for our linear quadratic problem is given by

<!-- formula-not-decoded -->

When J k is quadratic of the form J k ( x ) = K k x 2 with K k ≥ 0, it can be seen that the VI iterate J k +1 is also quadratic of the form J k +1 ( x ) = K k +1 x 2 ,

Corresponding to an Unstable and a Stable Policy

a-rK

+ 62K +

Kk+1

0

K

Kk Kk+1

Riccati Operator

Figure 1.5.3 Graphical illustration of value iteration for the linear quadratic problem. It has the form K k +1 = F ( K k ), where F is the Riccati operator,

<!-- image -->

<!-- formula-not-decoded -->

The algorithm converges to K ∗ starting from any K 0 ≥ 0.

where

<!-- formula-not-decoded -->

with F being the Riccati operator of Eq. (1.49). The algorithm is illustrated in Fig. 1.5.3. As can be seen from the figure, when starting from any K 0 ≥ 0, the algorithm generates a sequence ¶ K k ♦ of nonnegative scalars that converges to K ∗ .

## 1.5.1 Visualizing Approximation in Value Space - Region of Stability

The use of Riccati equations allows insightful visualization of approximation in value space. This visualization, although specialized to linear quadratic problems, is consistent with related visualizations for more general infinite horizon problems; this is a recurring theme in what follows. In particular, in the books [Ber20a] and [Ber22a], Bellman operators, which define the Bellman equations, are used in place of Riccati operators, which define the Riccati equations.

In summary, we will aim to show that:

- (a) Approximation in value space with one-step lookahead can be viewed as a Newton step for solving the Bellman equation, and maps the

ON-LINE PLAY

OFF-LINE TRAINING

Figure 1.5.4 Illustration of the interpretation of approximation in value space with one-step lookahead as a Newton step that maps ˜ J to the cost function J ˜ θ of the one-step lookahead policy.

<!-- image -->

terminal cost function approximation ˜ J to the cost function J ˜ θ of the one-step lookahead policy; see Fig. 1.5.4.

- (b) Approximation in value space with multistep lookahead and truncated rollout can be viewed as a Newton step for solving the Bellman equation, and maps the result of multiple VI iterations starting with the terminal cost function approximation ˜ J to the cost function J ˜ θ of the multistep lookahead policy; see Fig. 1.5.5.

Our derivation will be given for the one-dimensional linear quadratic problem, but applies far more generally . The reason is that the Bellman equation is valid universally in DP, and the corresponding Bellman operator has a concavity property that is well-suited for the application of Newton's method; see the books [Ber20a] and [Ber22a], where the connection of approximation in value space with Newton's method was first developed in detail.

Let us consider one-step lookahead minimization with any terminal cost function approximation of the form ˜ J ( x ) = Kx 2 , where K ≥ 0. We have derived the one-step lookahead policy θ K ( x ) in Eqs. (1.47)-(1.48), by minimizing the right side of Bellman's equation when J ( x ) = Kx 2 :

<!-- formula-not-decoded -->

We can break this minimization into a sequence of two minimizations as

NEWTON STEP for Bellman Eq. 2-Step Lookahead Minimization

ON-LINE PLAY

ON-LINE PLAY Lookahead Tree States

<!-- image -->

OFF-LINE TRAINING

Figure 1.5.5 Illustration of the interpretation of approximation in value space with multistep lookahead and truncated rollout as a Newton step, which maps the result of multiple VI iterations starting with the terminal cost function approximation ˜ J to the cost function J ˜ θ of the multistep lookahead policy.

follows:

<!-- formula-not-decoded -->

From this equation, it follows that

<!-- formula-not-decoded -->

where the function F L ( K ) is defined by

<!-- formula-not-decoded -->

Figure 1.5.6 illustrates the relation (1.52)-(1.53), and shows how the graph of the Riccati operator F can be obtained as the lower envelope of the linear operators F L , as L ranges over the real numbers.

## One-Step Lookahead Minimization and Newton's Method

Let us now fix the terminal cost function approximation to some ˜ Kx 2 , where ˜ K ≥ 0, and consider the corresponding one-step lookahead policy,

abK

Tangent Riccati

Operator at K

abK

[=--

Tangent Riccati Operator at

FL (K) = (a+bL)2K+9+rI2

arK

Figure 1.5.6 Illustration of how the graph of the Riccati operator F can be obtained as the lower envelope of the linear operators

<!-- image -->

<!-- formula-not-decoded -->

as L ranges over the real numbers. We have

<!-- formula-not-decoded -->

cf. Eq. (1.52). Moreover, for any fixed ˜ K , the scalar ˜ L that attains the minimum is given by

<!-- formula-not-decoded -->

[cf. Eq. (1.48)], and is such that the line corresponding to the graph of F ˜ L is tangent to the graph of F at ˜ K , as shown in the figure.

which we will denote by ˜ θ . Figure 1.5.7 illustrates the corresponding linear function F ˜ L , and shows that its graph is a tangent line to the graph of F at the point ˜ K [cf. Fig. 1.5.6 and Eq. (1.53)].

Thus the function F ˜ L can be viewed as a linearization of F at the point ˜ K , and defines a linearized problem: to find a solution of the equation

<!-- formula-not-decoded -->

The important point now is that the solution of this equation, denoted K ˜ L , is the same as the one obtained from a single iteration of Newton's method for solving the Riccati equation, starting from the point ˜ K . This is illustrated in Fig. 1.5.7, and is also justified analytically in Exercise 1.7.

abK

дІ(Ук)

+=

0

Fi (K)

K

K

abK

r + 62 K

Figure 1.5.7 Illustration of approximation in value space with one-step lookahead for the linear quadratic problem. Given a terminal cost approximation ˜ J ( x ) = ˜ Kx 2 , we compute the corresponding linear policy ˜ θ ( x ) = ˜ Lx , where

<!-- image -->

<!-- formula-not-decoded -->

and the corresponding cost function K ˜ L x 2 , using the Newton step shown.

To explain this connection, we note that the classical form of Newton's method for solving a fixed point problem of the form y = T ( y ), where y is an n -dimensional vector, operates as follows: At the current iterate y k , we linearize T and find the solution y k +1 of the corresponding linear fixed point problem. Assuming T is di ff erentiable, the linearization is obtained by using a first order Taylor expansion:

<!-- formula-not-decoded -->

where ∂ T ( y k ) glyph[triangleleft] ∂ y is the n × n Jacobian matrix of T evaluated at the vector y k , as indicated in Fig. 1.5.7.

The most commonly given convergence rate property of Newton's method is quadratic convergence . It states that near the solution y ∗ , we have

<!-- formula-not-decoded -->

where ‖ · ‖ is the Euclidean norm, and holds assuming the Jacobian matrix exists and is Lipschitz continuous (see [Ber16], Section 1.4). There are extensions of Newton's method that are based on solving a linearized

T. :

=

abK1

7+ 62K1

K

Fi (K)

K1

Starting point enhancement osition Evaluator Engine Newton step Starting point enhancement

abK1

7 + 62K1

Figure 1.5.8 Illustration of approximation in value space with two-step lookahead for the linear quadratic problem. Starting with a terminal cost approximation ˜ J ( x ) = ˜ Kx 2 , we obtain K 1 using a single value iteration, thereby enhancing the starting point of the Newton step. We then compute the corresponding linear policy ˜ θ ( x ) = ˜ Lx , where

<!-- image -->

<!-- formula-not-decoded -->

and the corresponding cost function K ˜ L x 2 , using the Newton step shown. The figure shows that for any K ≥ 0, the corresponding /lscript -step lookahead policy will be stable for all /lscript larger than some threshold.

system at the current iterate, but relax the di ff erentiability requirement to piecewise di ff erentiability, and/or component concavity, while maintaining the either a quadratic or a similarly fast superlinear convergence property of the method; see the monograph [Ber22a] (Appendix A) and the paper [Ber22c], which also provide a convergence analysis.

Note also that if the one-step lookahead policy is stable, i.e., ♣ a + b ˜ L ♣ &lt; 1 ↪ then K ˜ L is the quadratic cost coe ffi cient of its cost function, i.e.,

<!-- formula-not-decoded -->

/negationslash

The reason is that J ˜ θ solves the Bellman equation for policy ˜ θ . On the other hand, if ˜ θ is not stable, then in view of the positive definite quadratic cost per stage, we have J ˜ θ ( x ) = ∞ for all x = 0.

## Multistep Lookahead

In the case of /lscript -step lookahead minimization, a similar Newton step inter-

[=

Riccati Equation Formulas for One-Dimensional Problems

Riccati equation for minimization [cf. Eqs. (1.41) and (1.42)]

<!-- formula-not-decoded -->

Riccati equation for a linear policy θ ( x ) = Lx

<!-- formula-not-decoded -->

Cost coe ffi cient K L of a stable linear policy θ ( x ) = Lx

<!-- formula-not-decoded -->

Linear coe ffi cient L K of the one-step lookahead linear policy θ K for K in the region of stability [cf. Eq. (1.48)]

<!-- formula-not-decoded -->

Quadratic cost coe ffi cient ˜ K of a one-step lookahead linear policy θ K for K in the region of stability

Obtained as the solution of the linearized Riccati equation

<!-- formula-not-decoded -->

or equivalently by a Newton iteration starting from K .

pretation is possible. Instead of linearizing F at ˜ K , we linearize at

<!-- formula-not-decoded -->

i.e., the result of /lscript -1 successive applications of F starting with ˜ K . Each application of F corresponds to a value iteration. Thus the e ff ective starting point for the Newton step is F /lscript -1 ( ˜ K ). Figure 1.5.8 depicts the case /lscript = 2.

## Region of Stability

It is also useful to define the region of stability as the set of K ≥ 0 such

Unstable Policy

Slope=1

Instability Region Stability Region Slope=1

also

Region of Convergence of

Newton's Method

Line Stable Policy Unstable Policy Region of stability

Also Region of Convergence of Newton's Method

Stable Policy

Figure 1.5.9 Illustration of the region of stability, i.e., the set of K ≥ 0 such that the one-step lookahead policy θ K is stable. This is also the set of initial conditions for which Newton's method converges to K ∗ asymptotically.

<!-- image -->

that

Line Stable Policy Unstable Policy Region of stability

<!-- formula-not-decoded -->

where L K is the linear coe ffi cient of the one-step lookahead policy corresponding to K ; cf. Eq. (1.48). The region of stability may also be viewed as the region of convergence of Newton's method . It is the set of starting points K for which Newton's method, applied to the Riccati equation F = F ( K ), converges to K ∗ asymptotically, and with a quadratic convergence rate (asymptotically as K → K ∗ ). Note that for our one-dimensional problem, the region of stability is the interval ( K S ↪ ∞ ) that is characterized by the single point K S where F has derivative equal to 1; see Fig. 1.5.9.

For multidimensional problems, the region of stability may not be characterized as easily. Still, however, it is generally true that the region of stability is enlarged as the length of the lookahead increases .

Indeed, with increased lookahead, the e ff ective starting point

<!-- formula-not-decoded -->

is pushed more and more within the region of stability. In particular, for any given K ≥ 0 , the corresponding /lscript -step lookahead policy will be stable for all /lscript larger than some threshold ; see Fig. 1.5.8. The book [Ber22a], Section 3.3, contains a broader discussion of the region of stability and the role of multistep lookahead in enhancing it; see also Exercises 1.8 and 1.9.

## Newton Step Interpretation of Approximation in Value Space in General Infinite Horizon Problems

The interpretation of approximation in value space as a Newton step, and related notions of stability that we have discussed in this section admit a broad generalization to the infinite horizon problems that we consider in this book and beyond. The key fact in this respect is that our DP problem formulation allows arbitrary state and control spaces, both discrete and continuous, and can be extended even further to general abstract models with a DP structure; see the abstract DP book [Ber22b].

Within this context, the Riccati operator is replaced by an abstract Bellman operator, and valuable insight can be obtained from graphical interpretations of the Bellman equation, the VI and PI algorithms, onestep and multistep approximation in value space, the region of stability, and exceptional behavior; see the book [Ber22a] for an extensive discussion. Naturally, the graphical interpretations and visualizations are limited to one dimension. However, they provide insight, and motivate conjectures and mathematical analysis, much of which is given in the book [Ber20a].

## The Importance of the First Step in Multistep Lookahead

The Newton step interpretation of approximation in value space leads to an important insight into the special character of the initial step in /lscript -step lookahead implementations. In particular, it is only the first step that acts as the Newton step , and needs to be implemented with precision; cf. Fig. 1.5.5. The subsequent /lscript -1 steps are a sequence of value iterations starting with ˜ J , and only serve to enhance the quality of the starting point of the Newton step. As a result, their precise implementation is not critical , a major point in the narrative of the author's book [Ber22a].

This idea suggests that we can simplify (within reason) the lookahead steps after the first with small (if any) performance loss for the multistep lookahead policy. An important example of such a simplification is the use of certainty equivalence, which will be discussed later in various contexts (Sections 1.6.9, 2.7.2, 2.8.3). Other possibilities include the 'pruning' of the lookahead tree after the first step; see Section 2.4. In practical terms, simplifications after the first step of the multistep lookahead can save a lot of on-line computation, which can be fruitfully invested in extending the length of the lookahead. This insight is supported by substantial computational experimentation, starting with the paper by Bertsekas and Casta˜ non [BeC99], which verified the beneficial e ff ect of using certainty equivalence after the first step.

## 1.5.2 Rollout and Policy Iteration

We will now consider the rollout algorithm for the linear quadratic problem, starting from a linear stable base policy θ . It generates the rollout policy

Figure 1.5.10 Illustration of the rollout algorithm for the linear quadratic problem. Starting from a linear stable base policy θ , it generates a stable rollout policy ˜ θ . The quadratic cost coe ffi cient of ˜ θ is obtained from the quadratic cost coe ffi cient of θ with a Newton step for solving the Riccati equation.

<!-- image -->

˜ θ by using a policy improvement operation, which by definition, yields the one-step lookahead policy that corresponds to terminal cost approximation ˜ J = J θ . Figure 1.5.10 illustrates the rollout algorithm. It can be seen from the figure that the rollout policy is in fact an improved policy, in the sense that J ˜ θ ( x ) ≤ J θ ( x ) for all x . Among others, this implies that the rollout policy is stable, since θ is assumed stable so that J θ ( x ) &lt; ∞ for all x .

Since the rollout policy is a one-step lookahead policy, it can also be described using the formulas that we developed earlier in this section. In particular, let the base policy have the form

<!-- formula-not-decoded -->

where L 0 is a scalar. We require that the base policy must be stable, i.e., ♣ a + bL 0 ♣ &lt; 1. From our earlier calculations, we have that the cost function of θ 0 is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Moreover, the rollout policy θ 1 has the form θ 1 ( x ) = L 1 x↪ where

<!-- formula-not-decoded -->

cf. Eqs. (1.47)-(1.48).

The PI algorithm is simply the repeated application of nontruncated rollout, and generates a sequence of stable linear policies ¶ θ k ♦ . By replicating our earlier calculations, we see that the policies have the form

<!-- formula-not-decoded -->

where L k is generated by the iteration

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with K k given by

<!-- formula-not-decoded -->

The corresponding cost function sequence is

<!-- formula-not-decoded -->

cf. Eq. (1.54). Part of the classical linear quadratic theory is that J θ k converges to the optimal cost function J * , while the generated sequence of linear policies ¶ θ k ♦ , where θ k ( x ) = L k x , converges to the optimal policy, assuming that the initial policy is linear and stable. The convergence rate of the sequence ¶ K k ♦ is quadratic, as indicated earlier. This result was proved by Kleinman [Kle68] for the continuous-time multidimensional version of the linear quadratic problem, and it was extended later to more general problems; see the references given in the books [Ber20a] and [Ber22a] (Kleinman gives credit to Bellman and Kalaba [BeK65] for the one-dimensional version of his results).

## Truncated Rollout

An m -step truncated rollout scheme with a stable linear base policy θ ( x ) = Lx , one-step lookahead minimization, and terminal cost approximation ˜ J ( x ) = ˜ Kx 2 is geometrically interpreted as in Fig. 1.5.11. The truncated rollout policy ˜ θ is obtained by starting at ˜ K , executing m VI steps using θ , followed by a Newton step for solving the Riccati equation.

We mentioned some interesting performance issues in our discussion of truncated rollout in Section 1.1. In particular we noted that:

- (a) Lookahead by rollout may be an economic substitute for lookahead by minimization: it may achieve a similar performance for the truncated rollout policy at a much reduced computational cost.

FI (K)

Optimal cost!

K*

Cost of Truncated

Rollout Policy й

Figure 1.5.11 Illustration of truncated rollout with one-step lookahead minimization, a stable base policy θ ( x ) = Lx , and terminal cost approximation ˜ K for the linear quadratic problem. In this figure the number of rollout steps is m = 4.

<!-- image -->

- (b) Lookahead by rollout with a stable policy has a beneficial e ff ect on the stability properties of the lookahead policy.

These statements are di ffi cult to establish analytically in some generality. However, they can be intuitively understood in the context with our onedimensional linear quadratic problem, using geometrical constructions like the one of Fig. 1.5.11. They are also consistent with the results of computational experimentation. We refer to the monograph [Ber22a] for further discussion.

## Double Newton Step - Rollout on Top of Approximation in Value Space

Given a cost function approximation ˜ K that defines the policy ˜ θ ( x ) = L ˜ K x , it is possible to consider rollout that uses ˜ θ as a base policy. This can be viewed as rollout built on top of approximation in value space, and is referred to as a double Newton step ; it is a Newton step that starts from the result of the Newton step that starts from ˜ K (see Fig. 1.5.12). The double Newton step is much more powerful than approximation in value space with two-step lookahead starting from ˜ K , which amounts to a value iteration followed by a Newton step. Moreover, the idea of a double Newton step extends to general infinite horizon problems.

Note that it is also possible to consider variants of rollout on top of approximation in value space, such as truncated, simplified, and multi-

Policy Improvement with Base Policy

Figure 1.5.12 Illustration of a double Newton step. Here, a base policy θ is obtained by one-step lookahead using cost function approximation ˜ J ( x ) = ˜ Kx 2 . The cost function of the corresponding rollout policy ˜ θ is obtained with two successive Newton steps starting from ˜ K .

<!-- image -->

step lookahead versions. An important example of the truncated version is the 1996 TD-Gammon architecture [TeG96], where the base policy is obtained through approximation in value space with a terminal cost function approximation that is computed o ff -line using a neural network.

## 1.5.3 Local and Global Error Bounds for Approximation in Value Space

In approximation in value space, an important analytical issue is to quantify the level of suboptimality of the one-step or multistep lookahead policy obtained. It is thus important to understand the character of the critical mapping between the approximation error ˜ J -J * and the performance error J ˜ θ -J * , where as earlier, J ˜ θ is the cost function of the lookahead policy ˜ θ and J * is the optimal cost function.

There is a classical one-step lookahead error bound for the case of an α -discounted problem with finite state space X , which has the form

<!-- formula-not-decoded -->

where ‖ · ‖ denotes the maximum norm,

<!-- formula-not-decoded -->

Figure 1.5.13 Illustration of the linear error bound (1.58) for /lscript -step lookahead approximation in value space. For /lscript = 1, we obtain the one-step bound (1.57).

<!-- image -->

see e.g., [Ber19a], Prop. 5.1.1. The bound (1.57) predicts a linear relation between the size of the approximation error ‖ ˜ J -J * ‖ and the performance error ‖ J ˜ θ -J * ‖ . For a generalization, we may view /lscript -step lookahead as onestep lookahead with a terminal cost function T /lscript -1 ˜ J , i.e., ˜ J transformed by /lscript -1 value iterations. We then obtain the /lscript -step bound

<!-- formula-not-decoded -->

The linear bounds (1.57)-(1.58) are illustrated in Fig. 1.5.13, and apply beyond the α -discounted case, to problems where the Bellman equation involves a contraction mapping over a subset of functions; see the RL book [Ber19a], Section 5.9.1, or the abstract DP book [Ber22b], Section 2.2.

Unfortunately, the linear error bounds are very conservative, and do not reflect practical reality, even qualitatively so . The main reason is that they are global error bounds, i.e., they hold for all ˜ J , even the worst possible. In practice, ˜ J is often chosen su ffi ciently close to J * , so that the error J ˜ θ -J * behaves consistently with the superlinear convergence rate of the Newton step that starts at ˜ J . In other words, for ˜ J relatively close to J * , we have the local estimate

<!-- formula-not-decoded -->

illustrated in Fig. 1.5.14.

A salient characteristic of this superlinear relation is that the performance error rises rapidly outside the region of superlinear convergence of Newton's method. Note that small improvements in the quality of ˜ J (e.g., better sampling methods, improved confidence intervals, and the like, without changing the approximation architecture) have little e ff ect, both inside and outside the region of convergence .

Figure 1.5.14 Schematic illustration of the correct superlinear error bound (1.59) for the case of /lscript -step lookahead approximation in value space scheme. The performance error rises rapidly outside the region of convergence of Newton's method [the illustration in the figure is not realistic; in fact the region of convergence is not bounded as it contains lines of the form γ e , where γ is a scalar and e is the unit vector (all components equal to 1)]. Note that this region expands as the size of lookahead /lscript increases. Furthermore, with long enough lookahead /lscript , the /lscript -step lookahead policy ˜ θ can be shown to be exactly optimal for many problems of interest; this is a theoretical result, which holds for α -discounted finite-state problems, among others, and has been known since the 60s-70s (Prop. 2.3.1 of [Ber22a] proves a general form of this result that applies beyond discounted problems).

<!-- image -->

In practical terms, there is often a huge di ff erence, both quantitative and qualitative, between the linear error bounds (1.57)-(1.58) and the superlinear error bound (1.59) . Moreover, the linear bounds, despite their popularity in academia, often misdirect academic research and confuse practitioners. Note that as we have mentioned earlier, the qualitative performance behavior predicted in Fig. 1.5.14 holds very broadly in ap-

A study by Laidlaw, Russell, and Dragan [LRD23] has assessed the practical performance of popular methods on a set of 155 problems, and found wide disparities relative to theoretical predictions. Quoting from this paper: 'we find that prior bounds do not correlate well with when deep RL succeeds vs. fails.'

The study goes on to assert the importance of long multistep lookahead (the size of /lscript ) in stabilizing the performance of approximation in value space schemes. It also confirms computationally a known theoretical result, namely that with long enough lookahead /lscript , the /lscript -step lookahead policy ˜ θ is exactly optimal (but the required length of /lscript depends on the approximation error ˜ J -J ∗ ). This fact has been known since the 60s-70s for α -discounted finite-state problems. A generalization of this result is given as Prop. 2.3.1 of the abstract DP book [Ber22b]; see also Section A.4 of the book [Ber22a], which discusses the convergence of Newton's method for systems of equations that involve nondi ff erentiable mappings (such as the Bellman operator).

used i with

Depen as S ir

of S, c at S

15

10 -

5

1

2

3

4

- Ки - K*

5

Figure 1.5.15 Illustration of the global error bound for the one-step lookahead error K ˜ θ -K ∗ as a function of ˜ K , compared with the true error obtained by one step of Newton's method starting from ˜ K ; cf. Example 1.5.1.

<!-- image -->

The problem data are a = 2, b = 2, q = 1, and r = 5. With these numerical values, we have K ∗ = 5 and the region of stability is ( S↪ ∞ ) with S = 1 glyph[triangleright] 25. The modulus of contraction α used in the figure is computed at S = S + 0 glyph[triangleright] 5. Depending on the chosen value of S , α can be arbitrarily close to 1, but decreases as S increases. Note that the error K ˜ θ -K ∗ is much smaller when ˜ K is larger than K ∗ than when it is lower, because the slope of F diminishes as K increases. This is not reflected by the global error bound.

proximation in value space, because it relies on notions of abstract DP that apply very generally, for arbitrary state spaces, control spaces, and other problem characteristics; see the abstract DP book [Ber22a].

We illustrate the failure of the linear error bound (1.57) to predict the performance of the one-step lookahead policy with an example given in Fig. 1.5.15.

## Example 1.5.1 (Global and Actual Error Bounds for a Linear Quadratic Problem)

Consider the one-dimensional linear quadratic problem, involving the system x k +1 = ax k + bu k ↪ and the cost per stage qx 2 k + ru 2 k glyph[triangleright] We will consider one-step lookahead, and a quadratic cost function approximation

<!-- formula-not-decoded -->

with ˜ K within the region of stability, which is some interval of the form ( S↪ ∞ ). The Riccati operator is

<!-- formula-not-decoded -->

and the one-step lookahead policy ˜ θ has cost function

<!-- formula-not-decoded -->

where K ˜ θ is obtained by applying one step of Newton's method for solving the Riccati equation K = F ( K ), starting at K = ˜ K .

Let S be the boundary of the region of stability, i.e., the value of K at which the derivative of F with respect to K is equal to 1:

<!-- formula-not-decoded -->

∣ Then the Riccati operator F is a contraction within any interval [ S↪ ∞ ) with S &gt; S , with a contraction modulus α that depends on S . In particular, α is given by

<!-- formula-not-decoded -->

The error bound (1.57) can be rederived for the case of quadratic functions and can be rewritten in terms of quadratic cost coe ffi cients as

∣ and satisfies 0 &lt; α &lt; 1 because S &gt; S , and the derivative of F is positive and monotonically decreasing to 0 as K increases to ∞ .

<!-- formula-not-decoded -->

where K ˜ θ is the quadratic cost coe ffi cient of the lookahead policy ˜ θ [and also the result of a Newton step for solving the fixed point Riccati equation F = F ( K ) starting from ˜ K ]. Aplot of ( K ˜ θ -K ∗ ) as a function of ˜ K , compared with the bound on the right side of this equation is shown in Fig. 1.5.15. It can be seen that ( K ˜ θ -K ∗ ) exhibits the qualitative behavior of Newton's method, which is very di ff erent than the bound (1.60). An interesting fact is that the bound (1.60) depends on α , which in turn depends on how close ˜ K is to the boundary S of the region of stability, while the local behavior of Newton's method is independent of S .

## 1.5.4 How Approximation in Value Space Can Fail and What to Do About It

Let us finally discuss the most common way that approximation in value space can fail. Consider the case where the terminal cost approximation ˜ J is obtained through training with data of an approximation architecture such as a neural network (e.g., as in AlphaZero and TD-Gammon). Then there are three components that determine the approximation error ˜ J -J * :

- (a) The power of the architecture , which roughly speaking is a measure of the representation error , i.e., the error that would be obtained even if infinite data were available and were used optimally to obtain ˜ J .
- (b) The error degradation due the limited availability of training data .

- (c) The additional error degradation due to imperfections in the training methodology .

Thus if the architecture is not powerful enough to bring ˜ J -J ∗ within the region of convergence of Newton's method, approximation in value space with one-step lookahead will likely fail, no matter how much data is collected and how e ff ective the associated training method is .

In this case, there are two potential practical remedies:

- (1) Use a more powerful architecture/neural network for representing ˜ J .
- (2) Extend the combined length of the lookahead minimization and truncated rollout in order to bring the e ff ective value of ˜ J within the region of convergence of Newton's method.

The first remedy typically requires a deep neural network or transformer, which uses more weights and requires more expensive training (see Chapter 3). The second remedy requires longer on-line computation and/or simulation, which may run up against some practical real-time implementation constraint. Parallel computation and sophisticated multistep lookahead implementation methods may help to mitigate these requirements (see Chapter 2).

## 1.6 EXAMPLES, REFORMULATIONS, AND SIMPLIFICATIONS

In this section we provide a few examples that illustrate problem formulation techniques, exact and approximate solution methods, and adaptations of the basic DP algorithm to various contexts. We refer to DP textbooks for extensive additional discussions of modeling and problem formulation techniques (see e.g., the many examples that can be found in the author's DP and RL textbooks [Ber12], [Ber17a], [Ber19a], [Ber20a], as well as in the neuro-dynamic programming book [BeT96]).

An important fact to keep in mind is that there are many ways to model a given practical problem in terms of DP, and that there is no unique choice for state and control variables. This will be brought out by the examples in this section, and is facilitated by the generality of DP: its basic algorithmic principles apply for arbitrary state, control, and disturbance spaces, and system and cost functions.

## 1.6.1 A Few Words About Modeling

In practice, optimization problems seldom come neatly packaged as mathematical problems that can be solved by DP/RL or some other methodology.

For a recent example of implementation of a grandmaster-level chess program with one-step lookahead and a huge-size (270M parameter) neural network position evaluator, see Ruoss et al. [RDM24].

Generally, a practical problem is a prime candidate for a DP formulation if it involves multiple sequential decisions, which are separated by feedback, i.e., by observations that can be used to enhance the e ff ectiveness of future decisions.

However, there are other types of problems that can be fruitfully formulated by DP. These include the entire class of deterministic problems, where there is no information to be collected: all the information needed in a deterministic problem is either known or can be predicted from the problem data that is available at time 0 (see, e.g., the traveling salesman Example 1.2.3). Moreover, for deterministic problems there is a plethora of non-DP methods, such as linear, nonlinear, and integer programming, random and nonrandom search, discrete optimization heuristics, etc. Still, however, the use of RL methods for deterministic optimization is a major subject in this book, which will be discussed in Chapter 2. We will argue there that rollout and its variations, when suitably applied, can improve substantially on the performance of other heuristic or suboptimal methods, however derived. Moreover, we will see that often for discrete optimization problems the DP sequential structure is introduced artificially, with the aim to facilitate the use of approximate DP/RL methods.

There are also problems that fit quite well into the sequential structure of DP, but can be fruitfully addressed by RL methods that do not have a fundamental connection with DP. An important case in point is policy gradient and policy search methods, which will be considered somewhat briefly in Chapter 3. Here the policy of the problem is parametrized by a set of parameters, so that the cost of the policy becomes a function of these parameters, and can be optimized by non-DP methods such as gradient or random search-based suboptimal approaches. This generally relates to the approximation in policy space approach, which we have discussed in Section 1.3.3 and we will discuss further in Section 3.4; see also Section 5.7 of the RL book [Ber19a].

## A Practical Guide

As a guide for formulating optimal control problems in a manner that is suitable for a DP solution the following two-stage process is suggested:

- (a) Identify the controls/decisions u k and the times k at which these controls are applied. Usually this step is fairly straightforward. However, in some cases there may be some choices to make. For example in deterministic problems, where the objective is to select an optimal sequence of controls ¶ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ , one may lump multiple controls to be chosen together, e.g., view the pair ( u 0 ↪ u 1 ) as a single choice. This is usually not possible in stochastic problems, where distinct decisions are di ff erentiated by the information/feedback available when making them.

- (b) Select the states x k . The basic guideline here is that x k should encompass all the information that is relevant for future optimization , i.e., the information that is known to the controller at time k and can be used with advantage in choosing u k . In e ff ect, at time k the state x k should separate the past from the future , in the sense that anything that has happened in the past (states, controls, and disturbances from stages prior to stage k ) is irrelevant to the choices of future controls as long we know x k . Sometimes this is described by saying that the state should have a 'Markov property' to express an analogy with states of Markov chains, where (by definition) the conditional probability distribution of future states depends on the past history of the chain only through the present state.

The control and state selection may also have to be refined or specialized in order to enhance the application of known results and algorithms. This includes the choice of a finite or an infinite horizon, and the availability of good base policies or heuristics in the context of rollout.

Note that there may be multiple possibilities for selecting the states, because information may be packaged in several di ff erent ways that are equally useful from the point of view of control. It may thus be worth considering alternative ways to choose the states; for example try to use states that minimize the dimensionality of the state space. For a trivial example that illustrates the point, if a quantity x k qualifies as state, then ( x k -1 ↪ x k ) also qualifies as state, since ( x k -1 ↪ x k ) contains all the information contained within x k that can be useful to the controller when selecting u k . However, using ( x k -1 ↪ x k ) in place of x k , gains nothing in terms of optimal cost while complicating the DP algorithm that would have to be executed over a larger space.

The concept of a su ffi cient statistic , which refers to a quantity that summarizes all the essential content of the information available to the controller, may be useful in providing alternative descriptions of the state space. An important paradigm is problems involving partial or imperfect state information, where x k evolves over time but is not fully accessible for measurement (for example, x k may be the position/velocity vector of a moving vehicle, but we may obtain measurements of just the position). If I k is the collection of all measurements and controls up to time k (the information vector ), it is correct to use I k as state in a reformulated DP problem that involves perfect state observation. However, a better alternative may be to use as state the conditional probability distribution

<!-- formula-not-decoded -->

called belief state , which (as it turns out) subsumes all the information that is useful for the purposes of choosing a control. On the other hand, the belief state P k ( x k ♣ I k ) is an infinite-dimensional object, whereas I k may be finite dimensional, so the best choice may be problem-dependent.

Still, in either case, the stochastic DP algorithm applies, with the su ffi cient statistic [whether I k or P k ( x k ♣ I k )] playing the role of the state.

## A Few Words about the Choice of an RL Method

An attractive aspect of the current RL methodology, inherited by the generality of our DP formulation, is that it can address a very broad range of challenging problems, deterministic as well as stochastic, discrete as well as continuous, etc. However, in the practical application of RL one has to contend with limited theoretical guarantees. In particular, several of the RL methods that have been successful in practice have less than solid performance properties, and may not work on a given problem, even one of the type for which they are designed.

This is a reflection of the state of the art in the field: there are no methods that are guaranteed to work for all or even most DP problems . However, there are enough methods to try on a given problem with a reasonable chance of success in the end (after some heuristic and problem specific tuning). For this reason, it is important to develop insight into the inner workings of various methods, as a means of selecting the proper type of methodology to try on a given problem.

A related consideration is the context within which a method is applied. In particular, is it a single problem that is being addressed, such as chess that has fixed rules and a fixed initial condition, or is it a family of related problems that must be periodically be solved with small variations in its data or its initial conditions? Also, are the problem data fixed or may they change over time as the system is being controlled?

Generally, convenient but relatively unreliable methods, which can be tuned to the problem at hand, may be tried with a reasonable chance of success if a single problem is addressed. Similarly, RL methods that require extensive tuning of parameters, including ones that involve approximation in policy space and the use of neural networks, may be well suited for a stable problem environment and a single problem solution. However, they are not well suited for problems with a variable environment and/or realtime changes of model parameters. For such problems, RL methods based on approximation in value space and on-line play, possibly involving on-line replanning, are much better suited.

Note also that even when on-line replanning is not needed, on-line play may improve substantially the performance of o ff -line trained policies, so we may wish to use it in conjunction with o ff -line training. This is

Aside from insight and intuition, it is also important to have a foundational understanding of the analytical principles of the field and of the mechanisms underlying the central computational methods. The role of the theory in this respect is to structure mathematically the methodology, guide the art, and delineate the sound from the flawed ideas.

due to the Newton step that is implicit in one-step or multistep lookahead minimization, cf. our discussion of the AlphaZero and TD-Gammon architectures in Section 1.1. Of course the computational requirements of an on-line play method may be substantial and have to be taken into account when assessing its suitability for a particular application. In this connection, deterministic problems are better suited than stochastic problems for on-line play. Moreover, methods that are well-suited for parallel computation, and/or involve the use of certainty equivalence approximations are generally better suited for a stochastic control environment.

## 1.6.2 Problems with a Termination State

Many DP problems of interest involve a termination state , i.e., a state t that is cost-free and absorbing in the sense that for all k ,

<!-- formula-not-decoded -->

Thus the control process essentially terminates upon reaching t , even if this happens before the end of the horizon. One may reach t by choice if a special stopping decision is available, or by means of a random transition from another state. Problems involving games, such as chess, Go, backgammon, and others include a termination state (the end of the game) and have played an important role in the development of the RL methodology.

Generally, when it is known that an optimal policy will reach the termination state with certainty within at most some given number of stages N , the DP problem can be formulated as an N -stage horizon problem, with a very large termination cost for the nontermination states. ‡ The reason is that even if the termination state t is reached at a time k &lt; N , we can extend our stay at t for an additional N -k stages at no additional cost, so the optimal policy will still be optimal, since it will not incur the large termination cost at the end of the horizon.

## Example 1.6.1 (Parking)

A driver is looking for inexpensive parking on the way to his destination. The parking area contains N spaces, numbered 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, and a garage following space N -1. The driver starts at space 0 and traverses the parking

Games often involve two players/decision makers, in which case they can be addressed by suitably modified exact or approximate DP algorithms. The DP algorithm that we have discussed in this chapter involves a single decision maker, but can be used to find an optimal policy for one player against a fixed and known policy of the other player.

‡ When an upper bound on the number of stages to termination is not known, the problem may be formulated as an infinite horizon problem of the stochastic shortest path problem.

have k beir

he dec

(F)|

F, с

es F a

**(F):

dF)

*(F):

c(0)

1

2

c(1)

k + 1•••

c(k)

N

c(k + 1)

c(N - 1)

Parking Spaces

Termination State tial State 15 1 5 18 4 19 9 21 25 8 12 13

Figure 1.6.1 Cost structure of the parking problem. The driver may park at space k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 at cost c ( k ), if the space is free, or continue to the next space k +1 at no cost. At space N (the garage) the driver must park at cost C .

<!-- image -->

spaces sequentially, i.e., from space k he goes next to space k +1, etc. Each parking space k costs c ( k ) and is free with probability p ( k ) independently of whether other parking spaces are free or not. If the driver reaches the last parking space N -1 and does not park there, he must park at the garage, which costs C . The driver can observe whether a parking space is free only when he reaches it, and then, if it is free, he makes a decision to park in that space or not to park and check the next space. The problem is to find the minimum expected cost parking policy.

We formulate the problem as a DP problem with N stages, corresponding to the parking spaces, and an artificial termination state t that corresponds to having parked; see Fig. 1.6.1. At each stage k = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, we have three states: the artificial termination state t , and the two states F and F , corresponding to space k being free or taken, respectively. At stage 0, we have only two states, F and F , and at the final stage there is only one state, the termination state t . The decision/control is to park or continue at state F [there is no choice at states F and state t ]. From location k , the termination state t is reached at cost c ( k ) when a parking decision is made (assuming location k is free). Otherwise, the driver continues to the next state at no cost. At stage N , the driver must park at cost C .

Let us now derive the form of the DP algorithm, denoting:

J ∗ k ( F ): The optimal cost-to-go upon arrival at a space k that is free.

J ∗ k ( F ): The optimal cost-to-go upon arrival at a space k that is taken.

J ∗ k ( t ): The cost-to-go of the 'parked'/termination state t .

The DP algorithm for k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 takes the form

<!-- formula-not-decoded -->

for the states other than the termination state t , while for t we have

<!-- formula-not-decoded -->

N

The minimization above corresponds to the two choices (park or not park) at the states F that correspond to a free parking space.

While this algorithm is easily executed, it can be written in a simpler and equivalent form. This can be done by introducing the scalars

<!-- formula-not-decoded -->

which can be viewed as the optimal expected cost-to-go upon arriving at space k but before verifying its free or taken status . Indeed, from the preceding DP algorithm, we have

ˆ J N -1 = p ( N -1) min [ c ( N -1) ↪ C ] + ( 1 -p ( N -1) ) C↪ ˆ J k = p ( k ) min [ c ( k ) ↪ ˆ J k +1 ] + ( 1 -p ( k ) ) ˆ J k +1 ↪ k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -2 glyph[triangleright] From this algorithm we can also obtain the optimal parking policy:

<!-- formula-not-decoded -->

This is an example of DP simplification that occurs when the state involves components that are not a ff ected by the choice of control, and will be addressed in the next section.

## 1.6.3 General Discrete Optimization Problems

Discrete deterministic optimization problems, including challenging combinatorial problems, can be typically formulated as DP problems by breaking down each feasible solution into a sequence of decisions/controls, similar to the preceding four queens example, the scheduling Example 1.2.1, and the traveling salesman Examples 1.2.2 and 1.2.3. This formulation often leads to an intractable exact DP computation because of an exponential explosion of the number of states as time progresses. However, a reformulation to a discrete optimal control problem brings to bear approximate DP methods, such as rollout and others, to be discussed later, which can deal with the exponentially increasing size of the state space.

Let us now extend the ideas of the examples just noted to the general discrete optimization problem:

<!-- formula-not-decoded -->

where U is a finite set of feasible solutions and G ( u ) is a cost function.

We assume that each solution u has N components; i.e., it has the form

<!-- formula-not-decoded -->

where N is a positive integer. We can then view the problem as a sequential decision problem, where the components u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 are selected one-ata-time. A k -tuple ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ) consisting of the first k components of a

., UN

le Uk+

..., UN.

k, Uk+1

Artificial

Initial State

Stage 1

•

U1

States

(no)

Stage 2

States

Stage 3

Stage N

) Approximate ..

<!-- image -->

) Approximate ..

) Approximate ..

) Approximate ..

) Approximate ..

) Approximate ..

Figure 1.6.2 Formulation of a discrete optimization problem as a DP problem with N stages. There is a cost G ( u ) only at the terminal stage on the arc connecting an N -solution u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ) upon reaching the terminal state. Note that there is only one incoming arc at each node.

solution is called a k -solution . We associate k -solutions with the k th stage of the finite horizon discrete optimal control problem shown in Fig. 1.6.2. In particular, for k = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N , we view as the states of the k th stage all the k -tuples ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ). For stage k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, we view u k as the control. The initial state is an artificial state denoted s . From this state, by applying u 0 , we may move to any 'state' ( u 0 ), with u 0 belonging to the set

<!-- formula-not-decoded -->

Thus U 0 is the set of choices of u 0 that are consistent with feasibility.

More generally, from a state ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ) ↪ we may move to any state of the form ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ↪ u k ) ↪ upon choosing a control u k that belongs to the set

<!-- formula-not-decoded -->

These are the choices of u k that are consistent with the preceding choices u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 , and are also consistent with feasibility [we do not exclude the possibility that the set (1.63) is empty]. The last stage corresponds to the N -solutions u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ), and the terminal cost is G ( u ); see Fig. 1.6.2. All other transitions in this DP problem formulation have cost 0.

Let J * k ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ) denote the optimal cost starting from the k -solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ), i.e., the optimal cost of the problem over solutions whose first k components are constrained to be equal to u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 . The

DP algorithm is described by the equation

<!-- formula-not-decoded -->

with the terminal condition

<!-- formula-not-decoded -->

This algorithm executes backwards in time: starting with the known function J * N = G , we compute J * N -1 , then J * N -2 , and so on up to computing J * 0 . An optimal solution ( u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ) is then constructed by going forward through the algorithm

<!-- formula-not-decoded -->

where U 0 is given by Eq. (1.62), and U k is given by Eq. (1.63): first compute u ∗ 0 , then u ∗ 1 , and so on up to u ∗ N -1 ; cf. Eq. (1.8).

Of course here the number of states typically grows exponentially with N , but we can use the DP minimization (1.64) as a starting point for approximation methods. For example we may try to use approximation in value space, whereby we replace J * k +1 with some suboptimal ˜ J k +1 in Eq. (1.64). One possibility is to use as

<!-- formula-not-decoded -->

the cost generated by a heuristic method that solves the problem suboptimally with the values of the first k + 1 decision components fixed at u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ k -1 ↪ u k . This is the rollout algorithm , which turns out to be a very simple and e ff ective approach for approximate combinatorial optimization.

Let us finally note that while we have used a general cost function G and constraint set U in our discrete optimization model of this section, in many problems G and/or U may have a special (e.g., additive) structure, which is consistent with a sequential decision making process and may be computationally exploited. The traveling salesman Example 1.2.2 is a case in point, where G consists of the sum of N components (the intercity travel costs), one per stage. Our next example deals with a problem of great current interest.

## Example 1.6.2 (A Large Language Model Based on N -Grams)

Let us consider an N -gram model, whereby a text string consisting of N words is transformed into another string of N words by adding a word at the front of the string and deleting the word at the back of the string. We view the text strings as states of a dynamic system a ff ected by the added word choice, which we view as the control. We denote by x k the string obtained at

indow Next Text Window Next Word Prompt

Figure 1.6.3 Schematic visualization of an LLM problem based on N -grams.

<!-- image -->

time k , and by u k the word added at time k . We assume that u k is chosen from a given set U ( x k ). Thus we have a controlled dynamic system, which is deterministic and is described by an equation of the form

<!-- formula-not-decoded -->

where f specifies the operation of adding u k at the front of x k and removing the word at the back of x k . The initial string x 0 is assumed given.

If we have a cost function G by which to evaluate a text string, we can pose a DP problem with either a finite or an infinite horizon. For example if the string evolution terminates after exactly N steps, we obtain the finite horizon problem of minimizing the function G ( x N ) of the final text string x N . In this case, x N is obtained after we have a chance to change successively all the words of the initial string x 0 , subject to the constraints u k ∈ U ( x k ).

Another possibility is to introduce a termination action, whereby addition/deletion of words is optionally stopped at some time and the final text string x is obtained with cost G ( x ). In such a problem formulation, we may also include an additive stage cost that depends on u . This is an infinite horizon formulation that involves an additional termination state t in the manner of Section 1.6.2.

Note that in both the finite and the infinite horizon formulation of the problem, the initial string x 0 may include a 'prompt,' which may be subject to optimization through some kind of 'prompt engineering.' Depending on the context, this may include the use of another optimization or heuristic algorithm, perhaps unrelated to DP, which searches for a favorable prompt from within a given set of choices.

Interesting policies for the preceding problem formulation may be provided by a neural network, such as a Generative Pre-trained Transformer (GPT). In our terms, the GPT can be viewed simply as a policy that generates next words. This policy may be either deterministic, i.e., u k = θ ( x k ) for some function θ , or it may be a 'randomized' policy, which generates u k according to a probability distribution that depends on x k . Our DP formulation can also form the basis for policy improvement algorithms such as rollout, which aim to improve the quality of the output generated by the GPT. Another, more ambitious, possibility is to consider an approximate, neural network-based, policy iteration/self-training scheme, such as the ones discussed earlier, based on the AlphaZero/TD-Gammon architecture. Such a scheme generates a sequence of GPTs, with each GPT trained with data provided by the preceding GPT, a form of self-learning in the spirit of the AlphaZero and TD-Gammon policy iteration algorithms, cf. Section 1.1.

It is also possible to provide an infinite horizon formulation of the general discrete optimization problem

<!-- formula-not-decoded -->

where U is a finite set of feasible solutions, G ( u ) is a cost function, and u consists of N components, u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ); cf. Eq. (1.61). To this end, we introduce a termination state t that the system enters after N steps. At step k , the component u k is selected subject to u k ∈ U k ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 , where the constraint set U k ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 is given by Eq. (1.63). This is a special case of a general finite to infinite horizon stochastic DP problem reformulation, which we describe in the next section.

## 1.6.4 General Finite to Infinite Horizon Reformulation

There is a conceptually important reformulation that transforms a finite horizon problem, possibly involving a nonstationary system and cost per stage, to an equivalent infinite horizon problem. It is based on introducing an expanded state space, which is the union of the state spaces of the finite horizon problem plus an artificial cost-free termination state that the system moves into at the end of the horizon. This reformulation is of great conceptual value, as it provides a mechanism to bring to bear ideas that can be most conveniently understood within an infinite horizon context. For example, it helps to understand the synergy of o ff -line training and on-line play based on Newton's method, and the related insights that explain the good performance of rollout algorithms in practice.

To define the reformulation, let us consider the N -stage horizon stochastic problem of Section 1.3.1, whose system has the form

<!-- formula-not-decoded -->

and let us denote by X k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N , and U k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, the corresponding state spaces and control spaces, respectively. We introduce an artificial termination state t , and we consider an infinite horizon problem with state and control spaces X and U given by see Fig. 1.6.4.

<!-- formula-not-decoded -->

The system equation and the control constraints of this problem are also reformulated so that states in X k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, are mapped to states in X k +1 , according to Eq. (1.66), while states x N ∈ X N are mapped to the termination state t at cost g N ( x N ). Upon reaching t , the state stays at t at no cost. Thus the policies of the infinite horizon problem map states

9N (XN)

JN -1 (XN-1)

MN - 1 (XN-1)

Jj (x2)

1º (x2)

Ji (x1) .

H] (х1)

J, (x0)

Но (хо)

Optimal cost and policy

XN

X2

Х1

XN

X2

X1

Xo

Optimal cost and policy

Optimal cost and policy

XN

N

Figure 1.6.4 Illustration of the infinite horizon equivalent of a finite horizon problem. The state space is X = ( ∪ N k =0 X k ) ∪ ¶ t ♦ ↪ and the control space is U = ∪ N -1 k =0 U k glyph[triangleright] Transitions from states x k ∈ X k lead to states in x k +1 ∈ X k +1 according to the system equation x k +1 = f k ( x k ↪ u k ↪ w k ), and they are stochastic when they involve the random disturbance w k . The transition from states x N ∈ X N lead deterministically to the termination state at cost g N ( x N ). The termination state t is cost-free and absorbing.

<!-- image -->

The infinite horizon optimal cost J ∗ ( x k ) and optimal policy θ ∗ ( x k ) at state x k ∈ X k of the infinite horizon problem are equal to optimal cost-to-go J ∗ k ( x k ) and optimal policy θ ∗ k ( x k ) of the finite horizon problem.

x k ∈ X k to controls in U k ( x k ) ⊂ U k , and consist of functions θ k ( x k ) that are policies of the finite horizon problem. Moreover, the Bellman equation for the infinite horizon problem is identical to the DP algorithm for the finite horizon problem.

It can be seen that the optimal cost and optimal control, J * ( x k ) and θ ∗ ( x k ), at a state x k ∈ X k in the infinite horizon problem are equal to the optimal cost-to-go J ∗ k ( x k ) and optimal control θ ∗ k ( x k ) of the original finite horizon problem, respectively; cf. Fig. 1.6.4. Moreover approximation in value space and rollout in the finite horizon problem translate to infinite horizon counterparts, and can be understood as Newton steps for solving the Bellman equation of the infinite horizon problem (or equivalently the DP algorithm of the finite horizon problem).

XN

XN

In summary, finite horizon problems can be viewed as infinite horizon problems with a special structure that involves a termination state t , and the state and control spaces of Eq. (1.67), as illustrated in Fig. 1.6.4. The Bellman equation of the infinite horizon problem coincides with the DP algorithm of the finite horizon problem. The PI algorithm for the infinite horizon problem can be translated directly to a PI algorithm for the finite horizon problem, involving repeated policy evaluations and policy improvements. Finally, the Newton step interpretations for approximation in value space and rollout schemes for the infinite horizon problem have straightforward analogs for finite horizon problems, and explain the powerful cost improvement mechanism that underlies the rollout algorithm and its variations.

## 1.6.5 State Augmentation, Time Delays, Forecasts, and Uncontrollable State Components

In practice, we are often faced with situations where some of the assumptions of our stochastic optimal control problem formulation are violated. For example, the disturbances may involve a complex probabilistic description that may create correlations that extend across stages, or the system equation may include dependences on controls applied in earlier stages, which a ff ect the state with some delay.

Generally, in such cases the problem can be reformulated into our DP problem format through a technique, which is called state augmentation because it typically involves the enlargement of the state space. The general intuitive guideline in state augmentation is to include in the enlarged state at time k all the information that is known to the controller at time k and can be used with advantage in selecting u k . State augmentation allows the treatment of time delays in the e ff ects of control on future states, correlated disturbances, forecasts of probability distributions of future disturbances, and many other complications. We note, however, that state augmentation often comes at a price: the reformulated problem may have a very complex state space. We provide some examples.

## Time Delays

In some applications the system state x k +1 depends not only on the preceding state x k and control u k , but also on earlier states and controls. Such situations can be handled by expanding the state to include an appropriate number of earlier states and controls.

As an example, assume that there is at most a single stage delay in the state and control; i.e., the system equation has the form

<!-- formula-not-decoded -->

If we introduce additional state variables y k and s k , and we make the identifications y k = x k -1 , z k = u k -1 , the system equation (1.68) yields

<!-- formula-not-decoded -->

By defining ˜ x k = ( x k ↪ y k ↪ z k ) as the new state, we have

<!-- formula-not-decoded -->

where the system function ˜ f k is defined from Eq. (1.69).

By using the preceding equation as the system equation and by expressing the cost function in terms of the new state, the problem is reduced to a problem without time delays. Naturally, the control u k should now depend on the new state ˜ x k , or equivalently a policy should consist of functions θ k of the current state x k , as well as the preceding state x k -1 and the preceding control u k -1 .

When the DP algorithm for the reformulated problem is translated in terms of the variables of the original problem, it takes the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar reformulations are possible when time delays appear in the cost or the control constraints; for example, in the case where the cost is

<!-- formula-not-decoded -->

The extreme case of time delays in the cost arises in the nonadditive form

<!-- formula-not-decoded -->

Then, the problem can be reduced to the standard problem format, by using as augmented state

<!-- formula-not-decoded -->

and E { g N (˜ x N ) } as reformulated cost. Policies consist of functions θ k of the present and past states x k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x 0 , the past controls u k -1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u 0 , and the past disturbances w k -1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w 0 . Naturally, we must assume that the past disturbances are known to the controller. Otherwise, we are faced with a problem where the state is imprecisely known to the controller, which will be discussed in the next section.

## Forecasts

Consider a situation where at time k the controller has access to a forecast y k that results in a reassessment of the probability distribution of the subsequent disturbance w k and, possibly, future disturbances. For example, y k may be an exact prediction of w k or an exact prediction that the probability distribution of w k is a specific one out of a finite collection of distributions. Forecasts of interest in practice are, for example, probabilistic predictions on the state of the weather, the interest rate for money, and the demand for inventory. Generally, forecasts can be handled by introducing additional state variables corresponding to the information that the forecasts provide. We will illustrate the process with a simple example.

Assume that at the beginning of each stage k , the controller receives an accurate prediction that the next disturbance w k will be selected according to a particular probability distribution out of a given collection of distributions ¶ P 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ P m ♦ ; i.e., if the forecast is i , then w k is selected according to P i . The a priori probability that the forecast will be i is denoted by p i and is given.

The forecasting process can be represented by means of the equation

<!-- formula-not-decoded -->

where y k +1 can take the values 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , corresponding to the m possible forecasts, and ξ k is a random variable taking the value i with probability p i . The interpretation here is that when ξ k takes the value i , then w k +1 will occur according to the distribution P i .

By combining the system equation with the forecast equation y k +1 = ξ k , we obtain an augmented system given by

<!-- formula-not-decoded -->

The new state and disturbance are

<!-- formula-not-decoded -->

The probability distribution of ˜ w k is determined by the distributions P i and the probabilities p i , and depends explicitly on ˜ x k (via y k ) but not on the prior disturbances.

Thus, by suitable reformulation of the cost, the problem can be cast as a stochastic DP problem. Note that the control applied depends on both the current state and the current forecast. The DP algorithm takes the form

<!-- formula-not-decoded -->

where y k may take the values 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , and the expectation over w k is taken with respect to the distribution P y k .

Note that the preceding formulation admits several extensions. One example is the case where forecasts can be influenced by the control action (e.g., pay extra for a more accurate forecast), and may involve several future disturbances. However, the price for these extensions is increased complexity of the corresponding DP algorithm.

## Problems with Uncontrollable State Components

In many problems of interest the natural state of the problem consists of several components, some of which cannot be a ff ected by the choice of control. In such cases the DP algorithm can be simplified considerably, and be executed over the controllable components of the state.

As an example, let the state of the system be a composite ( x k ↪ y k ) of two components x k and y k . The evolution of the main component, x k , is a ff ected by the control u k according to the equation

<!-- formula-not-decoded -->

where the distribution P k ( w k ♣ x k ↪ y k ↪ u k ) is given. The evolution of the other component, y k , is governed by a given conditional distribution P k ( y k ♣ x k ) and cannot be a ff ected by the control, except indirectly through x k . One is tempted to view y k as a disturbance, but there is a di ff erence: y k is observed by the controller before applying u k , while w k occurs after u k is applied, and indeed w k may probabilistically depend on u k .

It turns out that we can formulate a DP algorithm that is executed over the controllable component of the state, with the dependence on the uncontrollable component being 'averaged out' (see also the parking Example 1.6.1). In particular, let J * k ( x k ↪ y k ) denote the optimal cost-to-go at

stage k and state ( x k ↪ y k ), and define

<!-- formula-not-decoded -->

Note that the preceding expression can be interpreted as an 'average costto-go' at x k (averaged over the values of the uncontrollable component y k ). Then, similar to the parking Example 1.6.1, a DP algorithm that generates ˆ J k ( x k ) can be obtained, and has the following form:

<!-- formula-not-decoded -->

This is a consequence of the calculation

<!-- formula-not-decoded -->

Note that the minimization in the right-hand side of the preceding equation must still be performed for all values of the full state ( x k ↪ y k ) in order to yield an optimal control law as a function of ( x k ↪ y k ). Nonetheless, the equivalent DP algorithm (1.71) has the advantage that it is executed over a significantly reduced state space. Later, when we consider approximation in value space, we will find that it is often more convenient to approximate ˆ J k ( x k ) than to approximate J * k ( x k ↪ y k ); see the following discussions of forecasts and of the game of tetris.

As an example, consider the augmented state resulting from the incorporation of forecasts, as described earlier. Then, the forecast y k represents an uncontrolled state component, so that the DP algorithm can be simplified as in Eq. (1.71). In particular, assume that the forecast y k can take values i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m with probability p i . Then, by defining

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 1.6.5 Illustration of a tetris board.

<!-- image -->

which is executed over the space of x k rather than x k and y k . Note that this is a simpler algorithm to approximate than the one of Eq. (1.70).

Uncontrollable state components often occur in arrival systems, such as queueing, where action must be taken in response to a random event (such as a customer arrival) that cannot be influenced by the choice of control. Then the state of the arrival system must be augmented to include the random event, but the DP algorithm can be executed over a smaller space, as per Eq. (1.71). Here is an example of this type.

## Example 1.6.3 (Tetris)

Tetris is a popular video game played on a two-dimensional grid. Each square in the grid can be full or empty, making up a 'wall of bricks' with 'holes' and a 'jagged top' (see Fig. 1.6.5). The squares fill up as blocks of di ff erent shapes fall from the top of the grid and are added to the top of the wall. As a given block falls, the player can move horizontally and rotate the block in all possible ways, subject to the constraints imposed by the sides of the grid and the top of the wall. The falling blocks are generated independently according to some probability distribution, defined over a finite set of standard shapes. The game starts with an empty grid and ends when a square in the top row becomes full and the top of the wall reaches the top of the grid. When a row of full squares is created, this row is removed, the bricks lying above this row move one row downward, and the player scores a point. The player's objective is to maximize the score attained (total number of rows removed) up to termination of the game, whichever occurs first.

We can model the problem of finding an optimal tetris playing strategy as a finite horizon stochastic DP problem, with very long horizon. The state consists of two components:

- (1) The board position, i.e., a binary description of the full/empty status of each square, denoted by x .

## (2) The shape of the current falling block, denoted by y .

The control, denoted by u , is the horizontal positioning and rotation applied to the falling block. There is also an additional termination state which is cost-free. Once the state reaches the termination state, it stays there with no change in score. Moreover there is a very large amount added to the score when the end of the horizon is reached without the game having terminated.

The shape y is generated according to a probability distribution p ( y ), independently of the control, so it can be viewed as an uncontrollable state component. The DP algorithm (1.71) is executed over the space of board positions x and has the intuitive form

<!-- formula-not-decoded -->

where g ( x↪ y↪ u ) is the number of points scored (rows removed),

f ( x↪ y↪ u ) is the next board position (or termination state), when the state is ( x↪ y ) and control u is applied, respectively. The DP algorithm (1.72) assumes a finite horizon formulation of the problem.

Alternatively, we may consider an undiscounted infinite horizon formulation, involving a termination state (i.e., a stochastic shortest path problem). The 'reduced' form of Bellman's equation, which corresponds to the DP algorithm (1.72), has the form

<!-- formula-not-decoded -->

The value ˆ J ( x ) can be interpreted as an 'average score' at x (averaged over the values of the uncontrollable block shapes y ).

Finally, let us note that despite the simplification achieved by eliminating the uncontrollable portion of the state, the number of states x is still enormous, and the problem can only be addressed by suboptimal methods.

## 1.6.6 Partial State Information and Belief States

We have assumed so far that the controller has access to the exact value of the current state x k , so a policy consists of a sequence of functions of x k . However, in many practical settings, this assumption is unrealistic

Tetris is generally considered to be an interesting and challenging stochastic testbed for RL algorithms, and has received a lot of attention over a period spanning 20 years (1995-2015), starting with the paper [TsV96], the subsequent paper [BeI96] and the neuro-dynamic programming book [BeT96], and ending with the papers [GGS13], [SGG15], which contain many references to related works in the intervening years. All of these works are based on approximation in value space and various forms of approximate policy iteration.

because some components of the state may be inaccessible for observation, the sensors used for measuring them may be inaccurate, or the cost of measuring them more accurately may be prohibitive.

Often in such situations, the controller has access to only some of the components of the current state, and the corresponding observations may also be corrupted by stochastic uncertainty. For example in threedimensional motion problems, the state may consist of the six-tuple of position and velocity components, but the observations may consist of noisecorrupted radar measurements of the three position components. This gives rise to problems of partial or imperfect state information, which have received a lot of attention in the optimization and artificial intelligence literature (see e.g., [Ber17a], [RuN16]; these problems are also popularly referred to with the acronym POMDP for partially observed Markovian Decision problem ).

Generally, solving a POMDP exactly is typically intractable, even though there are DP algorithms for doing so. Thus in practice, POMDP are solved approximately, except under very special circumstances.

Despite their inherent computational di ffi culty, it turns out that conceptually, partial state information problems are no di ff erent than the perfect state information problems we have been addressing so far. In fact by various reformulations, we can reduce a partial state information problem to one with perfect state information, which involves a di ff erent and more complicated state, called a su ffi cient statistic . Once this is done, we can state an exact DP algorithm that is defined over the space of the su ffi cient statistic. Roughly speaking, a su ffi cient statistic is a quantity that summarizes the content of the information available up to k for the purposes of optimal control. This statement can be made more precise, but we will not elaborate further in this book; see e.g., the DP textbook [Ber17a].

A common su ffi cient statistic is the belief state , which we will denote by b k . It is the probability distribution of x k given all the observations that have been obtained by the controller and all the controls applied by the controller up to time k , and it can serve as 'state' in an appropriate DP algorithm. The belief state can in principle be computed and updated by a variety of methods that are based on Bayes' rule, such as Kalman filtering (see e.g., [AnM79], [KuV86], [Kri16], [ChC17]) and particle filtering (see e.g., [GSS93], [DoJ09], [Can16], [Kri16]).

## Example 1.6.4 (Bidirectional Parking)

Let us consider a more complex version of the parking problem of Example 1.6.1. As in that example, a driver is looking for inexpensive parking on the way to his destination, along a line of N parking spaces with a garage at the end. The di ff erence is that the driver can move in either direction, rather than just forward towards the garage. In particular, at space i , the driver can park at cost c ( i ) if i is free, can move to i -1 at a cost t -i or can move to i +1 at a cost t + i . Moreover, the driver records and remembers the free/taken

Initial State 15 1 5 18 4 19 9 21 25 8 12 13

Figure 1.6.6 Cost structure and transitions of the bidirectional parking problem. The driver may park at space k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 at cost c ( k ), if the space is free, can move to k -1 at cost t -k or can move to k +1 at cost t + k . At space N (the garage) the driver must park at cost C .

<!-- image -->

status of the spaces previously visited and may return to any of these spaces; see Fig. 1.6.6.

We assume that the probability p ( i ) of a space i being free changes over time, i.e., a space found free (or taken) at a given visit may get taken (or become free, respectively) by the time of the next visit. The initial probabilities p ( i ), before visiting any spaces, are known, and the mechanism by which these probabilities change over time is also known to the driver. As an example, we may assume that at each time stage, p ( i ) increases by a certain known factor with some probability ξ and decreases by another known factor with the complementary probability 1 -ξ .

Here the belief state is the vector of current probabilities

<!-- formula-not-decoded -->

and it can be updated with a simple algorithm at each time based on the new observation: the free/taken status of the space visited at that time.

We can use the belief state as the basis of an exact DP algorithm for computing an optimal policy. This algorithm is typically intractable computationally, but it is conceptually useful, and it can form the starting point for approximations. It has the form

<!-- formula-not-decoded -->

where:

J * k ( b k ) denotes the optimal cost-to-go starting from belief state b k at stage k .

U k is the control constraint set at time k (since the state x k is unknown at stage k , U k must be independent of x k ).

ˆ g k ( b k ↪ u k ) denotes the expected stage cost of stage k . It is calculated as the expected value of the stage cost g k ( x k ↪ u k ↪ w k ), with the joint

ective Terminal Cost Approximation Observation

Figure 1.6.7 Schematic illustration of the view of an imperfect state information problem as one of perfect state information, whose state is the belief state b k , i.e., the conditional probability distribution of x k given all the observations up to time k . The observation z k +1 plays the role of the stochastic disturbance. The function F k is a sequential estimator that updates the current belief state b k .

<!-- image -->

distribution of ( x k ↪ w k ) determined by the belief state b k and the distribution of w k .

F k ( b k ↪ u k ↪ z k +1 ) denotes the belief state at the next stage, given that the current belief state is b k , control u k is applied, and observation z k +1 is received following the application of u k :

<!-- formula-not-decoded -->

This is the system equation for a perfect state information problem with state b k , control u k , 'disturbance' z k +1 , and cost per stage ˆ g k ( b k ↪ u k ). The function F k is viewed as a sequential belief estimator , which updates the current belief state b k based on the new observation z k +1 . It is given by either an explicit formula or an algorithm (such as Kalman filtering or particle filtering) that is based on the probability distribution of z k and the use of Bayes' rule.

The expected value E z k +1 ¶· ∣ ∣ b k ↪ u k ♦ is taken with respect to the distribution of z k +1 , given b k and u k . Note that z k +1 is random, and its distribution depends on x k and u k , so the expected value

<!-- formula-not-decoded -->

in Eq. (1.73) is a function of b k and u k .

The algorithm (1.73) is just the ordinary DP algorithm for the perfect state information problem shown in Fig. 1.6.7. It involves the system/belief estimator (1.74) and the cost per stage ˆ g k ( b k ↪ u k ). Note that since b k takes

values in a continuous space, the algorithm (1.73) will typically require an approximate implementation, using approximation in value space methods.

We refer to the textbook [Ber17a], Chapter 4, for a detailed derivation of the DP algorithm (1.73), and to the monograph [BeS78] for a mathematical treatment that applies to infinite-dimensional state and disturbance spaces as well.

## An Alternative DP Algorithm for POMDP

The DP algorithm (1.73) is not the only one that can be used for POMDP. There is also an exact DP algorithm that operates in the space of information vectors I k , defined by

<!-- formula-not-decoded -->

where z k is the observation received at time k . This is another su ffi cient statistic, and hence an alternative to the belief state b k . In particular, we can view I k as a state of the POMDP, which evolves over time according to the equation

<!-- formula-not-decoded -->

Denoting by J ∗ k ( I k ) the optimal cost starting at information vector I k at time k , the DP algorithm takes the form

<!-- formula-not-decoded -->

A drawback of the preceding approach is that the information vector I k is growing in size over time, thereby leading to a nonstationary system even in the case of an infinite horizon problem with a stationary system and cost function. This di ffi culty can be remedied in an approximation scheme that uses a finite history of the system (a fixed number of most recent observations) as state, thereby working e ff ectively with a stationary finite-state system; see the paper by White and Scherrer [WhS94]. In particular, this approach is used in large language models such as ChatGPT and DeepSeek.

for k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, with J ∗ N ( I N ) = E { g N ( x N ) ♣ I N } ; see e.g., the DP textbook [Ber17a], Section 4.1.

Finite-memory approximations for POMDP can be viewed within the context of feature-based approximation architectures, as we will discuss in Chapter 3 (see Example 3.1.6). Moreover, the finite-history scheme can be generalized through the concept of a finite-state controller ; see the paper by Yu and Bertsekas [YuB08], which also addresses the issue of convergence of the approximation error to zero as the size of the finite-history or finitestate controller is increased.

## 1.6.7 Multiagent Problems and Multiagent Rollout

In this book, we will view a multiagent system as a collection of decision making entities, called agents , which aim to optimally achieve a common goal. The agents accomplish this by collecting and exchanging information, and otherwise interacting with each other. The agents can be software programs or physical entities such as robots, and they may have di ff erent capabilities.

Among the generic challenges of e ffi cient implementation of multiagent systems, one may note issues of limited communication and lack of fully shared information, due to factors such as limited bandwidth, noisy channels, and lack of synchronization. Another important generic issue is that as the number of agents increases, the size of the set of possible joint decisions of the agents increases exponentially, thereby complicating control selection by lookahead minimization. In this section, we will focus on ways to resolve this latter di ffi culty for problems where the agents fully share information, and in Section 2.9 we will address some of the challenges of problems where the agents may have some autonomy, and act without fully coordinating with each other.

For a mathematical formulation, let us consider the discounted infinite horizon problem and a special structure of the control space, whereby the control u consists of m components, u = ( u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ), with a separable control constraint structure u /lscript ∈ U /lscript ( x ), /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m . Thus the control constraint set is the Cartesian product

<!-- formula-not-decoded -->

where the sets U /lscript ( x ) are given. This structure arises in applications involving distributed decision making by multiple agents; see Fig. 1.6.8.

In particular, we will view each component u /lscript , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , as being chosen from within U /lscript ( x ) by a separate 'agent' (a decision making entity). For the sake of the following discussion, we assume that each set U /lscript ( x ) is finite. Then the one-step lookahead minimization of the standard rollout scheme with base policy θ is given by

<!-- formula-not-decoded -->

and involves as many as n m Q-factors, where n is the maximum number of elements of the sets U /lscript ( x ) [so that n m is an upper bound to the number of controls in U ( x ), in view of its Cartesian product structure (1.76)]. Thus the standard rollout algorithm requires an exponential [order O ( n m )] number of Q-factor computations per stage, which can be overwhelming even for moderate values of m .

In a more general version of a multiagent system, which is outside our scope, the agents may have di ff erent goals, and act in their own self-interest.

Info

Info

Agent 1

Info

Environment

Computing Cloud

Info

Info

Info

Info

Info

<!-- image -->

u3

Agent 2

Single policy Info

Figure 1.6.8 Schematic illustration of a multiagent problem. There are multiple 'agents,' and each agent /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m controls its own decision variable u /lscript . At each stage, agents exchange new information and also exchange information with the 'environment,' and then select their decision variables for the stage.

This potentially large computational overhead motivates a far more computationally e ffi cient rollout algorithm, whereby the one-step lookahead minimization (1.77) is replaced by a sequence of m successive minimizations, one-agent-at-a-time , with the results incorporated into the subsequent minimizations. In particular, given a base policy θ = ( θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m ), we perform at state x the sequence of minimizations

<!-- formula-not-decoded -->

Thus each agent component u /lscript is obtained by a minimization with the preceding agent components u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u /lscript -1 fixed at the previously computed values of the rollout policy, and the following agent components u /lscript +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m fixed at the values given by the base policy. This algorithm requires order

Agent 5

X

ul

х, и')

(2, 41,2?)

u3

ит-1

...

Stage

Figure 1.6.9 Equivalent formulation of the stochastic optimal control problem for the case where the control u consists of m components u 1 ↪ u 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m :

<!-- image -->

<!-- formula-not-decoded -->

The figure depicts the k th stage transitions. Starting from state x , we generate the intermediate states

<!-- formula-not-decoded -->

using the respective controls u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m -1 . The final control u m leads from ( x↪ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m -1 ) to ¯ x = f ( x↪ u↪ w ), and the random cost g ( x↪ u↪ w ) is incurred.

O ( nm ) Q-factor computations per stage, a potentially huge computational saving over the order O ( n m ) computations required by standard rollout.

A key idea here is that the computational requirements of the rollout one-step minimization (1.77) are proportional to the number of controls in the set U ( x k ) and are independent of the size of the state space. This motivates a reformulation of the problem, first suggested in the neurodynamic programming book [BeT96], Section 6.1.4, whereby control space complexity is traded o ff with state space complexity , by 'unfolding' the control u k into its m components, which are applied one agent-at-a-time rather than all-agents-at-once.

In particular, we can reformulate the problem by breaking down the collective decision u k into m individual component decisions, thereby reducing the complexity of the control space while increasing the complexity of the state space. The potential advantage is that the extra state space complexity does not a ff ect the computational requirements of some RL algorithms, including rollout .

To this end, we introduce a modified but equivalent problem, involving one-at-a-time agent control selection. At a state x , we break down the control u into the sequence of the m controls u 1 ↪ u 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m , and between x and the next state ¯ x = f ( x↪ u↪ w ), we introduce artificial intermediate 'states' ( x↪ u 1 ) ↪ ( x↪ u 1 ↪ u 2 ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ( x↪ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m -1 ), and corresponding transitions. The choice of the last control component u m at 'state' ( x↪ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m -1 ) marks the transition to the next state ¯ x = f ( x↪ u↪ w ), while incurring cost g ( x↪ u↪ w ); see Fig. 1.6.9.

It is evident that this reformulated problem is equivalent to the origi-

'• ••, um-1

(x, ul

Control um

Random Transition

* = f (x, u, w)

Random Cost

9 (х, и, г)

nal, since any control choice that is possible in one problem is also possible in the other problem, while the cost structure of the two problems is the same. In particular, every policy θ = ( θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m ) of the original problem, including a base policy in the context of rollout, is admissible for the reformulated problem, and has the same cost function for the original as well as the reformulated problem.

The motivation for the reformulated problem is that the control space is simplified at the expense of introducing m -1 additional layers of states, and the corresponding m -1 cost-to-go functions

<!-- formula-not-decoded -->

The increase in size of the state space does not adversely a ff ect the operation of rollout, since the Q-factor minimization (1.77) is performed for just one state at each stage.

The major fact that can be proved about multiagent rollout (see Section 2.9 and the end-of-chapter references) is that it achieves cost improvement :

<!-- formula-not-decoded -->

where J θ ( x ) is the cost function of the base policy θ = ( θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m ), and J ˜ θ ( x ) is the cost function of the rollout policy ˜ θ = (˜ θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ m ), starting from state x . Furthermore, this cost improvement property can be extended to multiagent PI schemes that involve one-agent-at-a-time policy improvement operations, and have sound convergence properties. Moreover, multiagent rollout becomes the starting point for related PI schemes that are well suited for distributed operation in contexts involving multiple autonomous decision makers; see Section 2.9, the book [Ber20a], the papers [Ber20b] and [BKB20], and the tutorial survey [Ber21a].

## Example 1.6.5 (Spiders and Flies)

This example is representative of a broad range of practical problems such as multirobot service systems involving delivery, maintenance and repair, search and rescue, firefighting, etc. Here there are m spiders and several flies moving on a 2-dimensional grid; cf. Fig. 1.6.10. The objective is for the spiders to catch all the flies as fast as possible.

During a stage, each fly moves to a some other position according to a given state-dependent probability distribution. Each spider learns the current state (the vector of spiders and fly locations) at the beginning of each stage, and either moves to a neighboring location or stays where it is. Thus each spider has as many as 5 choices at each stage. The control is u = ( u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ), where u /lscript is the choice of the /lscript th spider, so there are about 5 m possible values of u .

To apply multiagent rollout, we need a base policy. A simple possibility is to use the policy that directs each spider to move on the path of minimum distance to the closest fly position. According to the multiagent rollout formalism, the spiders choose their moves one-at-time in the order from 1 to m ,

1 6 10 Tluctration of a 2-dimencional cniderg\_and-fly nroblem

Figure 1.6.10 Illustration of a 2-dimensional spiders-and-fly problem with 20 spiders and 5 flies (cf. Example 1.6.5). The flies moves randomly, regardless of the position of the spiders. During a stage, each spider moves to a neighboring location or stays where it is, so there are 5 moves per spider (except for spiders at the edges of the grid). The total number of possible joint spiders moves is a little less than 5 20 .

<!-- image -->

taking into account the current positions of the flies and the earlier moves of other spiders, and assuming that future moves will be chosen according to the base policy, which is a tractable computation.

In particular, at the beginning at the typical stage, spider 1 selects its best move (out of the no more than 5 possible moves), assuming the other spiders 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m will move towards their closest surviving fly during the current stage, and all spiders will move towards their closest surviving fly during the following stages, up to the time where no surviving flies remain. Spider 1 then broadcasts its selected move to all other spiders. Then spider 2 selects its move taking into account the move already chosen by spider 1, and assuming that spiders 3 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m will move towards their closest surviving fly during the current stage, and all spiders will move towards their closest surviving fly during the following stages, up to the time where no surviving flies remain. Spider 2 then broadcasts its choice to all other spiders. This process of one-spider-at-a-time move selection is repeated for the remaining spiders 3 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , marking the end of the stage.

Note that while standard rollout computes and compares 5 m Q-factors (actually a little less to take into account edge e ff ects), multiagent rollout computes and compares ≤ 5 moves per spider, for a total of less than 5 m . Despite this tremendous computational economy, experiments with this type of spiders and flies problems have shown that multiagent rollout achieves a comparable performance to the one of standard rollout.

## 1.6.8 Problems with Unknown or Changing Model - Adaptive Control

Our discussion so far dealt with problems with a known mathematical model, i.e., one where the system equation, cost function, control constraints, and probability distributions of disturbances are perfectly known. The mathematical model may be available through explicit mathematical formulas and assumptions, or through computer software, such as a trained neural network, or a simulator that can emulate mathematical operations involved in the model or Monte Carlo simulation for the calculation of expected values.

From our point of view, it makes no di ff erence whether the mathematical model is available through closed form mathematical expressions or through a computer model, such as a simulator or a neural network : the methods that we discuss are valid in both cases, only their suitability for a given problem may be a ff ected by the type of model.

In practice, however, it is common that the system parameters are either not known exactly or can change over time, and this introduces potentially enormous complications. As an example consider our oversimplified cruise control system that we noted in Example 1.3.1 or its infinite horizon version. The state evolves according to

<!-- formula-not-decoded -->

where x k is the deviation v k -¯ v of the vehicle's velocity v k from the nominal ¯ v , u k is the force that propels the car forward, and w k is the disturbance. However, the coe ffi cient b and the distribution of w k change frequently, and cannot be modeled with any precision because they depend on unpredictable time-varying conditions, such as the slope and condition of the road, and the weight of the car (which is a ff ected by the number of passengers). Moreover, the nominal velocity ¯ v is set by the driver, and when it changes it may a ff ect the parameter b in the system equation, and other parameters. ‡

In this section, we will briefly review some of the most commonly used approaches for dealing with unknown models and/or parameters in optimal

The di ffi culties of decision and control within a changing environment are often underestimated. Among others, they complicate the balance between o ff -line training and on-line play, which we discussed in Section 1.1 in connection the AlphaZero. It is worth keeping in mind that as much as learning to play high quality chess is a great challenge, the rules of play are stable and do not change unpredictably in the middle of a game! Problems with changing system parameters can be far more challenging!

‡ Adaptive cruise control, which can also adapt the car's velocity based on its proximity to other cars, has been studied extensively and has been incorporated in several commercially sold car models.

control theory and practice. We should note also that unknown problem environments are an integral part of the artificial intelligence view of RL. In particular, to quote from the popular book by Sutton and Barto [SuB18], RL is viewed as 'a computational approach to learning from interaction,' and 'learning from interaction with the environment is a foundational idea underlying nearly all theories of learning and intelligence.'

The idea of learning from interaction with the environment is often connected with the idea of exploring the environment to identify its characteristics. In control theory this is often viewed as part of the system identification methodology, which aims to construct mathematical models of dynamic systems. The system identification process is often combined with the control process to deal with unknown or changing problem parameters, in a framework that is sometimes called dual control . This is one of the most challenging areas of stochastic optimal and suboptimal control, and has been studied intensively since the early 1960s.

Let us also note that modern advances in neural network and artificial intelligence technologies have expanded significantly the range of the system identification methodology. As an example, we refer to Section 2.3.8, which provides a brief discussion of world models . These can be used as a computational surrogate for an exact model to provide predictions of the system's trajectories beyond the first step of lookahead minimization, in contexts of approximation in value space and rollout.

## Robust and Adaptive Control

Given a controller design that has been obtained assuming a nominal DP problem model, one possibility is to simply ignore changes in problem parameters. We may then try to investigate the performance of the current design for a suitable range of problem parameter values, and ensure that it is adequate for the entire range. This is sometimes called a robust controller design . For example, consider the oversimplified cruise control system of Eq. (1.78) with a linear controller of the form θ ( x ) = Lx for some scalar L . Then we check the range of parameters b for which the current controller is stable (this is the interval of values b for which ♣ 1 + bL ♣ &lt; 1), and ensure that b remains within that range during the system's operation.

The more general class of methods where the controller is modified in response to problem parameter changes is part of a broad field known as adaptive control , i.e., control that adapts to changing parameters. This is a rich methodology that we will discuss only briefly in this book. For the moment, let us just mention a simple time-honored adaptive control approach for a broad class of continuous-state problems without a known mathematical or computer model, called PID (Proportional-Integral-Derivative) control , for which we refer to the control literature, including the books by Astr om and Hagglund [AsH95], [AsH06], and the end-of-chapter references on adaptive control (also Section 5.7 of the RL textbook [Ber19a]).

System Data Control Parameter Estimation

) System Data Control Parameter Estimation

System State Data Control Parameter Estimation

System State Data Control Parameter Estimation

System State Data Control Parameter Estimation

System State Data Control Parameter Estimation

Figure 1.6.11 Schematic illustration of concurrent parameter estimation and system control. The system parameters are estimated on-line and the estimates are periodically passed on to the controller.

<!-- image -->

In particular, PID control aims to maintain the output of a singleinput single-output dynamic system around a set point or to follow a given trajectory, as the system parameters change within a relatively broad range. In its simplest form, the PID controller is parametrized by three scalar parameters, which may be determined by a variety of methods, some of them manual/heuristic. PID control is used widely and with success, although its range of application is mainly restricted to single-input, single-output continuous-state control systems.

## Dealing with Unknown Parameters Through System Identification

In PID control, no attempt is made to maintain a mathematical model and to track unknown model parameters as they change. An alternative and apparently reasonable form of suboptimal control is to separate the control process into two phases, a system identification phase and a control phase . In the first phase the unknown parameters are estimated, while the control takes no account of the interim results of estimation. The final parameter estimates from the first phase are then used to implement an optimal or suboptimal policy in the second phase. This alternation of estimation and control phases may be repeated several times during any system run in order to take into account subsequent changes of the parameters. Moreover, it is not necessary to introduce a hard separation between the identification and the control phases. They may be going on simultaneously, with new parameter estimates being introduced into the control process, whenever this is thought to be desirable; see Fig. 1.6.11.

One drawback of this approach is that it is not always easy to determine when to terminate one phase and start the other. A second di ffi culty, of a more fundamental nature, is that the control process may make some

of the unknown parameters invisible to the estimation process. This is known as the problem of parameter identifiability , which is discussed in the context of optimal control in several sources, including [BoV79] and [Kum83]; see also [Ber17a], Section 6.7.

## Example 1.6.6 (Parameter Identifiability Under Closed-Loop Control)

For a simple example, consider the scalar system

<!-- formula-not-decoded -->

and the quadratic cost

<!-- formula-not-decoded -->

Assuming perfect state information, if the parameters a and b are known, it can be seen that the optimal control law is

<!-- formula-not-decoded -->

which sets all future states to 0. Assume now that the parameters a and b are unknown, and consider the two-phase method. During the first phase the control law

<!-- formula-not-decoded -->

is used ( γ is some scalar; for example, γ = -a b ↪ where a and b are some a priori estimates of a and b , respectively). At the end of the first phase, the control law is changed to

<!-- formula-not-decoded -->

where ˆ a and ˆ b are the estimates obtained from the estimation process. However, with the control law (1.79), the closed-loop system is

<!-- formula-not-decoded -->

so the estimation process can at best yield the value of ( a + b γ ) but not the values of both a and b . In other words, the estimation process cannot discriminate between pairs of values ( a 1 ↪ b 1 ) and ( a 2 ↪ b 2 ) such that

<!-- formula-not-decoded -->

Therefore, a and b are not identifiable when feedback control of the form (1.79) is applied.

On-line parameter estimation algorithms, which address among others the issue of identifiability, have been discussed extensively in the control

theory literature, but the corresponding methodology is complex and beyond our scope in this book. However, assuming that we can make the estimation phase work somehow, we are free to revise the controller using the newly estimated parameters in a variety of ways, in an on-line replanning process.

Unfortunately, there is still another di ffi culty with this type of online replanning: it may be hard to recompute an optimal or near-optimal policy on-line, using a newly identified system model. In particular, it may be impossible to use time-consuming methods that involve for example the training of a neural network or discrete/integer control constraints. A simpler possibility is to use rollout, which we discuss next.

## Adaptive Control by Rollout and On-Line Replanning

We will now consider an approach for dealing with unknown or changing parameters, which is based on on-line replanning. We have discussed this approach in the context of rollout and multiagent rollout, where we stressed the importance of fast on-line policy improvement.

Let us assume that some problem parameters change and the current controller becomes aware of the change 'instantly' (i.e., very quickly before the next stage begins). The method by which the problem parameters are recalculated or become known is immaterial for the purposes of the following discussion. It may involve a limited form of parameter estimation, whereby the unknown parameters are 'tracked' by data collection over a few time stages, with due attention paid to issues of parameter identifiability; or it may involve new features of the control environment, such as a changing number of servers and/or tasks in a service system (think of new spiders and/or flies appearing or disappearing unexpectedly in the spiders-and-flies Example 1.6.5).

We thus assume away/ignore issues of parameter estimation, and focus on revising the controller by on-line replanning based on the newly obtained parameters. This revision may be based on any suboptimal method, but rollout with the current policy used as the base policy is particularly attractive. Here the advantage of rollout is that it is simple and reliable. In particular, it does not require a complicated training procedure to re-

Another possibility is to deal with this di ffi culty by precomputation. In particular, assume that the set of problem parameters may take a known finite set of values (for example each set of parameter values may correspond to a distinct maneuver of a vehicle, motion of a robotic arm, flying regime of an aircraft, etc). Then we may precompute a separate controller for each of these values. Once the control scheme detects a change in problem parameters, it switches to the corresponding predesigned current controller. This is sometimes called a multiple model control design or gain scheduling , and has a long history of success in various settings over the years; see e.g., Athans et al., [ACD77].

Lookahead

Minimization

Xk

Possible States

Xk+ 1

Rollout with

Base Policy

Changing System,

<!-- image -->

Changing System, Cost, and Constraint Parameters

Changing System, Cost, and Constraint Parameters

Multiagent Q-factor minimization

Possible States

Figure 1.6.12 Schematic illustration of adaptive control by rollout. One-step lookahead is followed by simulation with the base policy, which stays fixed. The system, cost, and constraint parameters are changing over time, and the most recent values are incorporated into the lookahead minimization and rollout operations. For the discussion in this section, we may assume that all the changing parameter information is provided by some computation and sensor 'cloud' that is beyond our control. The base policy may also be revised based on various criteria.

vise the current policy, based for example on the use of neural networks or other approximation architectures, so no new policy is explicitly computed in response to the parameter changes . Instead the current policy is used as the base policy for rollout, and the available controls at the current state are compared by a one-step or mutistep minimization, with cost function approximation provided by the base policy (cf. Fig. 1.6.12).

Note that over time the base policy may also be revised (on the basis of an unspecified rationale), in which case the rollout policy will be revised both in response to the changed current policy and in response to the changing parameters. This is necessary in particular when the constraints of the problem change.

The principal requirement for using rollout in an adaptive control context is that the rollout control computation should be fast enough to be performed between stages. Note, however, that accelerated/truncated versions of rollout, as well as parallel computation, can be used to meet

this time constraint.

The following example considers on-line replanning with the use of rollout in the context of the simple one-dimensional linear quadratic problem that we discussed earlier in this chapter. The purpose of the example is to illustrate analytically how rollout with a policy that is optimal for a nominal set of problem parameters works well when the parameters change from their nominal values. This property is not practically useful in linear quadratic problems because when the parameter change, it is possible to calculate the new optimal policy in closed form, but it is indicative of the performance robustness of rollout in other contexts. Generally, adaptive control by rollout and on-line replanning makes sense in situations where the calculation of the rollout controls for a given set of problem parameters is faster and/or more convenient than the calculation of the optimal controls for the same set of parameter values. These problems include cases involving nonlinear systems and/or di ffi cult (e.g., integer) constraints.

## Example 1.6.7 (On-Line Replanning for Linear Quadratic Problems Based on Rollout)

Consider the deterministic undiscounted infinite horizon linear quadratic problem. It involves the linear system

<!-- formula-not-decoded -->

and the quadratic cost function

<!-- formula-not-decoded -->

The optimal cost function is given by

<!-- formula-not-decoded -->

where K ∗ is the unique positive solution of the Riccati equation

<!-- formula-not-decoded -->

The optimal policy has the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

As an example, consider the optimal policy that corresponds to the nominal problem parameters b = 2 and r = 0 glyph[triangleright] 5: this is the policy (1.81)(1.82), with K obtained as the positive solution of the quadratic Riccati Eq. (1.80) for b = 2 and r = 0 glyph[triangleright] 5. In particular, we can verify that

<!-- formula-not-decoded -->

From Eq. (1.82) we then obtain

<!-- formula-not-decoded -->

Wewill now consider changes of the values of b and r while keeping L constant, and we will compare the quadratic cost coe ffi cient of the following three cost functions as b and r vary:

- (a) The optimal cost function K ∗ x 2 , where K ∗ is given by the positive solution of the Riccati Eq. (1.80).
- (b) The cost function K L x 2 that corresponds to the base policy

<!-- formula-not-decoded -->

where L is given by Eq. (1.83). From our earlier discussion, we have

<!-- formula-not-decoded -->

- (c) The cost function ˜ K L x 2 that corresponds to the rollout policy

<!-- formula-not-decoded -->

obtained by using the policy θ L as base policy. Using the formulas given earlier, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Figure 1.6.13 shows the coe ffi cients K ∗ , K L , and ˜ K L for a range of values of r and b . We have

<!-- formula-not-decoded -->

The di ff erence K L -K ∗ is indicative of the robustness of the policy θ L , i.e., the performance loss incurred by ignoring the values of b and r , and continuing to use the policy θ L , which is optimal for the nominal values b = 2 and r = 0 glyph[triangleright] 5, but suboptimal for other values of b and r . The di ff erence ˜ K L -K ∗ is indicative of the performance loss due to using on-line replanning by rollout rather than using optimal replanning. Finally, the di ff erence K L -˜ K L is indicative of the performance improvement due to on-line replanning using rollout rather than keeping the policy θ L unchanged.

Note that Fig. 1.6.13 illustrates the behavior of the error ratio

<!-- formula-not-decoded -->

where for a given initial state, ˜ J is the rollout performance, J ∗ is the optimal performance, and J is the base policy performance. This ratio approaches 0 as J -J ∗ becomes smaller because of the quadratic convergence rate of Newton's method that underlies the rollout algorithm.

Exactly Reoptimized Policy

Approximately Reoptimized Rollout Policy

Approximately Reoptimized Rollout Policy

Figure 1.6.13 Illustration of adaptive control by rollout under changing problem parameters. The quadratic cost coe ffi cients K ∗ (optimal, denoted by solid line), K L (base policy, denoted by circles), and ˜ K L (rollout policy, denoted by asterisks) for the two cases where r = 0 glyph[triangleright] 5 and b varies, and b = 2 and r varies. The value of L is fixed at the value that is optimal for b = 2 and r = 0 glyph[triangleright] 5 [cf. Eq. (1.83)].

<!-- image -->

The rollout policy performance is very close to the one of the exactly reoptimized policy, while the base policy yields much worse performance. This is a consequence of the quadratic convergence rate of Newton's method that underlies rollout:

<!-- formula-not-decoded -->

where for a given initial state, ˜ J is the rollout performance, J ∗ is the optimal performance, and J is the base policy performance.

## Adaptive Control as POMDP

The preceding adaptive control formulation strictly separates the dual objective of estimation and control: first parameter identification and then controller reoptimization (either exact or rollout-based). In an alternative adaptive control formulation, the parameter estimation and the application of control are done simultaneously, and indeed part of the control e ff ort may be directed towards improving the quality of future estimation. This alternative (and more principled) approach is based on a view of adaptive control as a partially observed Markovian decision problem (POMDP) with a special structure. We will see in Section 2.11 that this approach is well-suited for approximation in value space schemes, including forms of rollout.

To describe briefly the adaptive control reformulation as POMDP, we introduce a system whose state consists of two components:

- (a) A perfectly observed component x k that evolves over time according to a discrete-time equation.
- (b) A component θ which is unobserved but stays constant, and is estimated through the perfect observations of the component x k .

We view θ as a parameter in the system equation that governs the evolution of x k . Thus we have

<!-- formula-not-decoded -->

where u k is the control at time k , selected from a set U k ( x k ), and w k is a random disturbance with given probability distribution that depends on ( x k ↪ θ ↪ u k ). For convenience, we will assume that θ can take one of m known values θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ m .

The a priori probability distribution of θ is given and is updated based on the observed values of the state components x k and the applied controls u k . In particular, the information vector

<!-- formula-not-decoded -->

is available at time k , and is used to compute the conditional probabilities

<!-- formula-not-decoded -->

These probabilities form a vector

<!-- formula-not-decoded -->

which together with the perfectly observed state x k , form the pair ( x k ↪ b k ), which is the belief state of the POMDP at time k . The overall control scheme takes the form illustrated in Fig. 1.6.14.

System State Data Control Parameter Estimation

k

) Belief Estimator

) Belief Estimator

Figure 1.6.14 Schematic illustration of simultaneous control and belief estimation for the unknown system parameter θ . The control applied is a function of the current belief state ( x k ↪ b k ), where b k is the conditional probability distribution of θ given the observations accumulated up to time k (the current and past states x k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x 0 , and the past controls u k -1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u 0 ).

<!-- image -->

As discussed in Section 1.6.6, an exact DP algorithm can be written for the equivalent POMDP, and this algorithm is suitable for the use of approximation in value space and rollout. We will describe this approach in some detail in Section 2.11. Related ideas will also be discussed in the context of Bayesian estimation and sequential estimation in Section 2.10.

Note that the case of a deterministic system

<!-- formula-not-decoded -->

is particularly interesting, because we can then typically expect that the true parameter θ ∗ will be identified in a finite number of stages. The reason is that at each stage k , we are receiving a noiseless observation relating to θ , namely the state x k . Once the true parameter θ ∗ is identified, the problem becomes one of perfect state information.

## 1.6.9 Model Predictive Control

In this section, we will provide a brief summary of the model predictive control (MPC) methodology for control system design, with a view towards its connection with DP/RL, approximation in value space, and rollout schemes. We will focus on the classical control problem: keeping the state of a deterministic system close to some reference point, taken here to be the origin of the state space (see Fig. 1.6.15). Another type of classical

An extensive overview of the connections of the conceptual framework of this book with model predictive and adaptive control is given in the author's paper [Ber24]. The corresponding video is a good supplement to the present section and can be found at https://www.youtube.com/watch?v=UeVs0Op-Ui4 and a related video can also be found at https://www.youtube.com/watch?v=ZBRouvMat2Q

m

REGULATION PROBLEM

Keep the state near some given point

## PATH PLANNING FOLLOW A GIVEN TRAJECTORY REGULATION PROBLEM

-Component Control

Figure 1.6.15 Illustration of a classical regulation problem, known as the 'cartpole problem' or 'inverse pendulum problem.' The state is the two-dimensional vector of angular position and angular velocity. We aim to keep the pole at the upright position (state equal to 0) by exerting horizontal force u on the cart.

<!-- image -->

control problem is to keep the system close to a given trajectory (see Fig. 1.6.16). It can also be treated by forms of MPC, but will not be addressed in this book.

We discussed earlier the linear quadratic approach, whereby the system is represented by a linear model, the cost is quadratic in the state and the control, and there are no state and control constraints. The linear quadratic and other approaches based on state variable system representations and optimal control became popular, starting in the late 50s and early 60s. Unfortunately, however, the analytically convenient linear quadratic problem formulations are often not satisfactory. There are two main reasons for this:

- (a) The system may be nonlinear, and it may be inappropriate to use for control purposes a model that is linearized around the desired point or trajectory. Moreover, some of the control variables may be naturally discrete, and this is incompatible with the linear system viewpoint.
- (b) There may be control and/or state constraints, which are not handled adequately through quadratic penalty terms in the cost function. For example, the motion of a car may be constrained by the presence of obstacles and hardware limitations (see Fig. 1.6.16). The solution obtained from a linear quadratic model may not be suitable for such a problem, because quadratic penalties treat constraints 'softly' and may produce trajectories that violate the constraints.

These inadequacies of the linear quadratic formulation have motivated

FOLLOW A

GIVEN TRAJECTORY

Moving Obstacle

Fixed Obstacles

PATH PLANNING FOLLOW A GIVEN TRAJECTORY States at the End of t

Best Score Fixed Obstacles

Velocity

Constraints

Acceleration

<!-- image -->

Must Deal with State and Control Constraints Linear-Quadratic F

Must Deal with State and Control Constraints Linear-Quadratic F

Figure 1.6.16 Illustration of constrained motion of a car from point A to point B. There are state (position/velocity) constraints, and control (acceleration) constraints. When there are mobile obstacles, the state constraints may change unpredictably, necessitating on-line replanning.

MPC, which combines elements of several ideas that we have discussed so far, such as multistep lookahead, rollout with a base policy, and certainty equivalence. Aside from dealing adequately with state and control constraints, MPC is well-suited for on-line replanning, like all approximation in value space methods.

Note that the ideas of MPC were developed independently of the approximate DP/RL methodology. However, the two fields are closely related, and there is much to be gained from understanding one field within the context of the other, as the subsequent development will aim to show. A major di ff erence between MPC and finite-state stochastic control problems that are popular in the RL/artificial intelligence literature is that in MPC the state and control spaces are continuous/infinite, such as for example in self-driving cars, the control of aircraft and drones, or the operation of chemical processes. At the same time, at a fundamental level, this di ff erence turns out to be inconsequential, because the key underlying framework for approximation in value space, which is based on Newton's method, is valid for both discrete and continuous state and control spaces.

In this section, we will primarily focus on the undiscounted infinite

horizon deterministic problem, which involves the system

<!-- formula-not-decoded -->

whose state x k and control u k are finite-dimensional vectors. The cost per stage is assumed nonnegative

<!-- formula-not-decoded -->

(e.g., a positive definite quadratic cost). There are control constraints u k ∈ U ( x k ) ↪ and to simplify the following discussion, we will initially consider no state constraints. We assume that the system can be kept at the origin at zero cost, i.e.,

<!-- formula-not-decoded -->

For a given initial state x 0 , we want to obtain a sequence ¶ u 0 ↪ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ that satisfies the control constraints, while minimizing the total cost.

This is a classical problem in control system design, known as the regulation problem , where the aim is to keep the state of the system near the origin (or more generally some desired set point), in the face of disturbances and/or parameter changes. In an important variant of the problem, there are additional state constraints of the form x k ∈ X , and there arises the issue of maintaining the state within X , not just at the present time but also in future times. We will address this issue later in this section.

## The Classical Form of MPC - View as a Rollout Algorithm

We will first focus on a classical form of the MPC algorithm, discussed in the form given here by Keerthi and Gilbert [KeG88], with a view towards stability analysis. In this algorithm, at each encountered state x k , we apply a control ˜ u k that is computed as follows; see Fig. 1.6.17:

- (a) We solve an /lscript -stage optimal control problem involving the same cost function and the requirement that the state after /lscript steps is driven to 0, i.e., x k + /lscript = 0. This is the problem

<!-- formula-not-decoded -->

subject to the system equation constraints

<!-- formula-not-decoded -->

the control constraints

<!-- formula-not-decoded -->

-Factors Current State

Current State

<!-- image -->

Sample Q-Factors Simulation Control 1 Control 2 Control 3

1)-Stages Base Heuristic Minimization

Figure 1.6.17 Illustration of the problem solved by a classical form of MPC at state x k . We minimize the cost function over the next /lscript stages while imposing the requirement that x k + /lscript = 0. We then apply the first control of the optimizing sequence. In the context of rollout, the minimization over u k is the one-step lookahead, while the minimization over u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k + /lscript -1 that drives x k + /lscript to 0 is the base heuristic.

and the terminal state constraint

<!-- formula-not-decoded -->

Here /lscript is an integer with /lscript &gt; 1, which is chosen in some largely empirical way.

- (b) If ¶ ˜ u k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u k + /lscript -1 ♦ is the optimal control sequence of this problem, we apply ˜ u k and we discard the other controls ˜ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u k + /lscript -1 .
- (c) At the next stage, we repeat this process, once the next state x k +1 is revealed.

To make the connection of the preceding MPC algorithm with rollout, we note that the one-step lookahead function ˜ J implicitly used by MPC [cf. Eq. (1.85)] is the cost function of a certain stable base policy . This is the policy that drives to 0 the state after /lscript -1 stages ( not /lscript stages ) and keeps the state at 0 thereafter, while observing the state and control constraints, and minimizing the associated ( /lscript -1)-stages cost. This rollout view of MPC was first discussed in the author's paper [Ber05]. It is useful for making a connection with the approximate DP/RL, rollout, and its interpretation in terms of Newton's method. In particular, an important consequence is that the MPC policy is stable , since rollout with a stable base policy can be shown to yield a stable policy under very general conditions, as we have noted earlier for the special case of linear quadratic problems in Section 1.5; cf. Fig. 1.5.10.

## Terminal Cost Approximation - Stability Issues

In a common variant of MPC, the requirement of driving the system state to 0 in /lscript steps in the /lscript -stage MPC problem (1.85), is replaced by a terminal cost G ( x k + /lscript ), which is positive everywhere except at 0. Thus at state x k , we solve the problem

<!-- formula-not-decoded -->

instead of problem (1.85) where we require that x k + /lscript = 0. This variant can be viewed as rollout with one-step lookahead, and a base policy, which at state x k +1 applies the first control ˜ u k +1 of the sequence ¶ ˜ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u k + /lscript -1 ♦ that minimizes

<!-- formula-not-decoded -->

On the other hand, this MPC variant can also be viewed as approximation in value space with /lscript -step lookahead minimization and terminal cost approximation given by G . It can be interpreted in terms of a Newton step, as illustrated in Fig. 1.5.7 for the case of one-step lookahead, and in Fig. 1.5.8 for the case of multistep lookahead.

An important question is how to choose the terminal cost approximation so that the resulting MPC controller is stable. Our discussion of Section 1.5 on the region of stability of approximation in value space schemes applies here. In particular, under the nonnegative cost assumption of this section, the MPC controller can be proved to be stable if a single value iteration (VI) starting from G produces a function that takes uniformly smaller values than G :

<!-- formula-not-decoded -->

This is also known as the Lyapunov condition in MPC. Figure 1.6.18 provides a graphical illustration, showing how this condition guarantees that successive iterates of value iteration, as implemented through multistep lookahead, lie within the region of stability, so that the policy produced by MPC is stable.

We expect that as the length /lscript of the lookahead minimization is increased, the stability properties of the MPC controller are improved. In particular, given G ≥ 0 , the resulting MPC controller is likely to be stable for /lscript su ffi ciently large , since the VI algorithm ordinarily converges to J * , which lies within the region of stability. Results of this type are known within the MPC framework under various conditions (see the papers by Mayne at al. [MRR00], Magni et al. [MDM01], the MPC book [RMD17],

Value Iterations

Bellman Operator

-

TJ.

Slope = 1

Optimal cost

J* = TJ*

Cost of

MPC Policy й

MPC Policy й

l = 3

Figure 1.6.18 Illustration of the Bellman operator, defined by

<!-- image -->

<!-- formula-not-decoded -->

The condition in (1.90) can be written compactly as ( TG )( x ) ≤ G ( x ) for all x . When satisfied by the terminal cost function G , it guarantees stability of the MPC policy ˜ θ with /lscript -step lookahead minimization. In this figure, /lscript = 3.

and the author's book [Ber20a], Section 3.1.2). Our discussion of stability in Section 1.5 is also relevant within this context; cf. Fig. 1.5.8.

In another variant of MPC, in addition to the terminal cost function approximation G , we use truncated rollout, which involves running some stable base policy θ for a number of steps m ; see Fig. 1.6.19. This is quite similar to standard truncated rollout, except that the computational solution of the lookahead minimization problem (1.89) may become complicated when the control space is infinite. As discussed in Section 1.5, increasing the length of the truncated rollout enlarges the region of stability of the MPC controller . The reason is that by increasing this length, we push the start of the Newton step towards of the cost function J θ of the stable policy, which lies within the region of stability. The base policy may also be used to address state constraints; see the papers by Rosolia and Borelli [RoB17], [RoB19], Li et al. [LJM21], and the discussions in the author's RL books [Ber20a], [Ber22a].

G

Base Policy

TuJ

TJ

Slope = 1

Defined by

Cost-to-go approximation Expected value approximation

Optimal cost

J* = TJ*

Stability Region l-Step

Lookahead

Minimization

Te

Cost of

MPC Policy й

m-Step Truncated

Rollout with

Stable Policy H

<!-- image -->

) Yields Truncated Rollout Policy ˜

Terminal Cost Approximation

Figure 1.6.19 An MPC scheme with /lscript -step lookahead minimization, m -step truncated rollout with a stable base policy θ , and a terminal cost function approximation G , together with its interpretation as a Newton step. In this figure, /lscript = 2 and m = 4. The truncated rollout with base policy θ consists of m value iterations, starting with the function G , and using the Bellman operator corresponding to θ , which is given by

<!-- formula-not-decoded -->

Thus, truncated rollout yields the function T θ m G . Then /lscript -1 value iterations are applied to this function through the ( /lscript -1)-step minimization, yielding the function

<!-- formula-not-decoded -->

Finally, the Newton step is applied to this function to yield the cost function of the MPC policy ˜ θ . As m increases, the starting point for the Newton step moves closer to J θ , which lies within the region of stability.

Finally, let us note that when faced with changing problem parameters, it is natural to consider on-line replanning as per our earlier adaptive control discussion. In this context, once new estimates of system and/or cost function parameters become available, MPC can readily adapt by introducing the new parameter estimates into the /lscript -stage optimization problem in (a) above. This is an important and often decisive advantage of

MPC over approximation in policy space for problems with changing environments.

## State Constraints and Invariant Sets

Our discussion so far has skirted a major issue in MPC, which is that there may be additional state constraints of the form x k ∈ X , for all k , where X is some subset of the true state space. Indeed much of the original work on MPC was motivated by control problems with state constraints, imposed by the physics of the problem, which could not be handled e ff ectively with the nice unconstrained framework of the linear quadratic problem that we discussed in Section 1.5.

To deal with additional state constraints of the form x k ∈ X , where X is some subset of the state space, the MPC problem to be solved at the k th stage [cf. Eq. (1.89)] must be modified. Assuming that the current state x k belongs to the constraint set X , the MPC problem should take the form

<!-- formula-not-decoded -->

subject to the control constraints

<!-- formula-not-decoded -->

and the state constraints

<!-- formula-not-decoded -->

The control ˜ u k thus obtained will generate a state

<!-- formula-not-decoded -->

that will belong to X , and similarly the entire state trajectory thus generated will satisfy the state constraint x t ∈ X for all t , assuming that the initial state does.

However, there is an important di ffi culty with the preceding MPC scheme, namely there is no guarantee that the problem (1.91)-(1.93) has a feasible solution for all initial states x k ∈ X . Here is a simple example.

## Example 1.6.8 (State Constraints in MPC)

Consider the scalar system with control constraint

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∈

Bellman Operator Value Iterations Largest Invariant Set

β

Figure 1.6.20 Illustration of invariance of a state constraint set X . Here the sets of the form X = { x k ♣ ♣ x k ♣ ≤ β } are invariant for β ≤ 1. For β = 1, we obtain the largest invariant set (the one that contains all other invariant sets). The figure shows some state trajectories produced by MPC. Note that starting with an initial condition x 0 with ♣ x 0 ♣ &gt; 1 (or ♣ x 0 ♣ &lt; 1) the closed-loop system obtained by MPC is unstable (or stable, respectively); cf. the red and green trajectories shown.

<!-- image -->

and state constraints of the form x k ∈ X , for all k , where

<!-- formula-not-decoded -->

Then if β &gt; 1, the state constraint cannot be satisfied for all initial states x 0 ∈ X . In particular, if we take x 0 = β , then 2 x 0 &gt; 2 and x 1 = 2 x 0 + u 0 will satisfy x 1 &gt; x 0 = β for any value of u 0 with ♣ u 0 ♣ ≤ 1. Similarly the entire sequence of states ¶ x k ♦ generated by any set of feasible controls will satisfy

<!-- formula-not-decoded -->

The state constraint can be satisfied only for initial states x 0 in the set ˆ X given by

<!-- formula-not-decoded -->

see Fig. 1.6.20, which also illustrates the trajectories generated by the MPC scheme of Eq. (1.89), which does not involve state constraints.

The preceding example illustrates a fundamental point in state-constrained MPC: the state constraint set X must be invariant in the sense that starting from any one of its points x k there must exist a control u k ∈ U ( x k ) for which the next state x k +1 = f ( x k ↪ u k ) must belong to X . Mathematically, X is invariant if for every x ∈ X , there exists u ∈ U ( x ) such that f ( x↪ u ) ∈ Xglyph[triangleright]

In particular, it can be seen that the set X of Eq. (1.94) is invariant if and only if β ≤ 1.

Given an MPC calculation of the form (1.91)-(1.93), we must make sure that the set X is invariant, or else it should be replaced by an invariant subset ˆ X ⊂ X . Then the MPC calculation (1.91)-(1.93) will be feasible provided the initial state x 0 belongs to ˆ X .

This brings up the question of how we compute an invariant subset of a given constraint set. This is typically an o ff -line calculation that cannot be performed during on-line play. It turns out that given X there exists a largest possible invariant subset of X , which can be computed in the limit with an algorithm that resembles value iteration. In particular, starting with X 0 = X , we obtain a nested sequence of subsets through the recursion

<!-- formula-not-decoded -->

Clearly, we have X k +1 ⊂ X k for all k , and under mild conditions it can be shown that the intersection set ˆ X = ∩ ∞ k =0 X k ↪ is the largest invariant subset of X ; see the author's PhD thesis [Ber71] and subsequent paper [Ber72a], which introduced the concept of invariance and its use in satisfying state constraints in control over a finite and an infinite horizon.

To illustrate, in the preceding Example 1.6.8, the sequence of value iterates (1.95) starting with the set X 0 = ¶ x ♣ ♣ x ♣ ≤ β ♦ , where β &gt; 1, is given by

<!-- formula-not-decoded -->

It can be seen that we have β k +1 &lt; β k for all k and β k ↓ 1, so the intersection ˆ X = ∩ ∞ k =0 X k yields the largest invariant set ˆ X = { x k ♣ ♣ x k ♣ ≤ 1 } glyph[triangleright]

## Suboptimal Invariant Subsets

Since computing the largest invariant subset of a constraint set X is often computationally intractable, one may consider using smaller invariant subsets of X . A relatively simple possibility is to compute an invariant subset ˆ X that corresponds to some nominal policy ˆ θ [i.e., starting from any point x ∈ ˆ X , the state f ( x↪ ˆ θ ( x ) ) belongs to ˆ X ]. Such an invariant subset may be obtained or approximated with some form of simulation using the policy ˆ θ . Moreover, ˆ θ can also be used for truncated rollout and also provide a terminal cost function approximation. An alternative for the case of a linear system driven by ellipsoid-bounded disturbances, is to construct an ellipsoidal invariant subset; the author's PhD thesis [Ber71] provided an algorithm for doing so.

For more broadly applicable possibilities, we refer to the MPC literature; see e.g., the book by Rawlings, Mayne, and Diehl [RMD17] (Chapter

The term used in [Ber71] and [Ber72a] is reachability of a target tube ¶ X↪X↪glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ , which is synonymous to invariance of X .

3), and the surveys by Mayne [May14], and Houska, Muller, and Villanueva [HMV24], which give additional references. An important point is that the computation of an invariant subset of the given constraint set X must be done o ff -line, thus becoming part of the o ff -line training phase, along with the terminal cost function G .

To deal with state constraints in the context of a partially unknown or changing system model, combinations of MPC with robust control ideas have been suggested [see [RMD17] (Chapter 3), where robust tube-based MPC ideas are discussed]. An alternative is to replace the state constraints with penalty or barrier functions as part of the cost per stage. This approach has received attention, using what is known as control barrier functions and control Lyapunov functions . We refer to the several survey papers in the literature, and to the book by Xiao, Cassandras, and Belta [XCB23], for accounts of this research direction.

## Stochastic MPC by Certainty Equivalence

We note that while we have focused on deterministic problems in this section, there are variants of MPC that include the treatment of uncertainty. The books and papers cited earlier contain several ideas along these lines; see e.g. the books by Kouvaritakis and Cannon [KoC16], Rawlings, Mayne, and Diehl [RMD17], and the survey by Mesbah [Mes16].

In this connection, it is also worth mentioning the certainty equivalence approach that we discussed briefly earlier. In particular, upon reaching state x k we may perform the MPC calculations after replacing the uncertain quantities w k +1 ↪ w k +2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] with deterministic quantities w k +1 ↪ w k +2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] , while allowing for the stochastic character of the disturbance w k of just the current stage k . Note that only the first step of this MPC calculation is stochastic. Thus the calculation needed per stage is not much more di ffi cult than the one for deterministic problems, while still implementing a Newton step for solving the associated Bellman equation; see our earlier discussion, and also Section 2.5.3 of the RL book [Ber19a], Section 3.2 of the book [Ber22a], and the MPC overview paper [Ber24].

## Data-Driven MPC

One of the principal limitations of the mainstream methodology of MPC is the need for a mathematical model. In many cases, such a model may not be available or may be di ffi cult to derive. To deal with this situation, data-driven versions of MPC have been suggested, where sampled triplets, consisting of state, control, and next state, are used for prediction of system trajectories.

In this approach, the system is implicitly represented by a dataset of observed trajectories, rather than an explicit mathematical model, an approach pioneered by J. C. Willems; see Markovsky and Dorfler [MaD21] for a recent review. The data consists of many sample triplets ( x k ↪ u k ↪ x k +1 ),

and the idea is to use this data to approximate the system dynamics and perform MPC. We can do this by fitting a parametric system equation model to the data, and predicting future system trajectories using this model. Among other approaches, we may use a neural network (possibly a transformer) or other type of system identification method for this purpose. Nonparametric models are also possible, based on statistical models, such as Bayesian process regression.

Once the data-approximated model has been constructed o ff -line, it can be used on-line to predict and optimize the future system trajectory over a finite prediction horizon, and to apply the first control in the optimal sequence at the current state. Naturally, the quality of the model has a major impact on the performance of the corresponding MPC policy. Some challenges here are the choice of state, and also the fact that the learned model may be inaccurate outside the training set. Moreover, major di ffi -culties arise in an adaptive control environment, where the model must be modified on-line by using data. For some representative relevant papers, see [CLD19], [GrZ19], [HWM20], [KRW21], [CLR21], [BGH22], [CWA22], [GrZ22], [MHD23], and [BeA24].

## MPC with Imperfect State Information

The MPC methodology, in the form we have described it here, requires full state feedback. When the exact state is not available, an estimate may be used instead. In classical control methodology, it is common to use a Kalman filtering algorithm or an observer to reconstruct the state. This is possible when the state is continuous and a system model is known. Also, in some noncontinuous state problems, it may be possible to use some inference method to obtain a state estimate, and use it in the MPC algorithm as if it were the exact state. Otherwise, it is necessary to view the problem as one of imperfect state information, introduce belief states in place of the unobservable states, and adopt the corresponding POMDP-like methodology. This is, however, much more demanding computationally, both for training a terminal cost function approximation and for on-line computation of the MPC controls.

The alternative for problems with imperfect state information is a non-MPC/approximation in policy space approach, which may also have some serious drawbacks:

- (a) The training of suitable policies can be complicated and unreliable due to local minima and other di ffi culties (see Chapter 3).
- (b) In an adaptive control/changing model setting, it may be necessary to retrain policies on-line, which can be very sample-ine ffi cient or practically impossible.
- (c) Restricting the structure of the policy to be a function of just part of the available information (e.g., the current system output) places a

limit on performance: we are not using the full information available.

- (d) Fundamentally, MPC/approximation in value space has a generic advantage over approximation in policy space: the MPC policy is the result of a Newton step for solving Bellman's equation. Thus the error from optimality J ˜ θ -J * of the MPC policy ˜ θ is governed by a superlinear relation, as we have discussed, and is negligible if the terminal cost approximation error ˜ J -J * is small.

## Model Mismatch and Disturbance Estimation in MPC

In practical implementations of MPC, an important source of performance degradation is model mismatch. Even when a reasonably accurate model is available to start with, unmodeled dynamics, parameter drift, or varying workloads, may cause the true evolution of the system to di ff er from the predictions used by the controller. If such mismatch is ignored, the closed-loop system may exhibit long-term state deviations from their desired targets or degraded constraint handling. For this reason, many practical MPC schemes for control system design incorporate some mechanism for estimating and compensating for the dominant model errors and other disturbances during real-time operation.

A viewpoint, emphasized in chemical process control (see the classical textbook by Stephanopoulos [Ste84] and its updated sequel [Ste25]), is that in many industrial settings, modeling errors can be significant, but can e ff ectively be viewed as a constant (or slowly varying) disturbance. If the controller can infer this disturbance from available measurements, it can suitably compensate for it; this is an idea that stems from the so called internal model principle , articulated by Francis and Wonham [FrW76]. The simplest instance of this is the integral term in a PID control scheme (cf. Section 1.6.8), which acts as a dynamic estimator of the unknown average disturbance: by accumulating the output error, it gradually identifies the control action needed to make the measured output coincide with the desired one.

A similar strategy is adopted in MPC formulations in a more general and systematic way. Rather than relying solely on integral action, one may introduce a small number of disturbance variables into the system model and allow an observer to update their values based on the mismatch between predicted and measured state evolution. Within our DP framework, this can be implemented through state augmentation (cf. Section 1.6.5). For a simple example, let the system x k +1 = f ( x k ↪ u k ) be replaced by

<!-- formula-not-decoded -->

where w k is a disturbance that represents a constant (or slowly varying) modeling error, so that

<!-- formula-not-decoded -->

We may then introduce w k as an additional state variable and an augmented system with state ˜ x k = ( x k ↪ w k ). If the state x k is observed perfectly, the same is true for the augmented state ˜ x k , since w k can be computed exactly as the di ff erence x k -f ( x k -1 ↪ u k -1 ).

A popular generalization of this approach, is to model the e ff ect of the disturbance through a system equation of the form

<!-- formula-not-decoded -->

where E is a known matrix, which models the e ff ects of the components of w k on di ff erent state variables/components of x k . When the state is not measured exactly, the MPC controller is naturally combined with a Kalman filter or related estimator. This estimator updates both the state and disturbance estimates at each stage, essentially producing a continually refined model of the system. The MPC controller then applies its optimization step as if the estimates of state and disturbance were exact.

For many systems of interest in chemical process control, the combination of disturbance modeling and state estimation, outlined above, captures the main benefits of more elaborate adaptive or robust MPC techniques; see the book by Stephanopoulos [Ste25] noted earlier. For larger and more rapidly varying uncertainties, more sophisticated identification or robust formulations may be necessary; we refer to the MPC literature for further discussion, particularly the textbooks by Rawlings, Mayne, and Diehl [RMD17], and by Borrelli, Bemporad, and Morari [BBM17].

## 1.7 REINFORCEMENTLEARNINGANDDECISION/CONTROL

The current state of RL has greatly benefited from the cross-fertilization of ideas from decision and control, and from artificial intelligence; see Fig. 1.7.1. The strong connections between these two fields are now widely recognized. Still, however, there are cultural di ff erences, including the traditional reliance on mathematical analysis for the decision and control field, and the emphasis on challenging problem implementations in the artificial intelligence (AI) field. Moreover, substantial di ff erences in language and emphasis remain between RL-based discussions (where AI-related terminology is used) and DP-based discussions (where optimal control-related terminology is used).

## 1.7.1 Di ff erences in Terminology

The terminology used in this book is standard in DP and optimal control, and in an e ff ort to forestall confusion of readers that are accustomed to either the AI or the decision and control terminology, we provide a list of terms commonly used in AI/RL, and their optimal control counterparts.

- (a) Environment = System.
- (b) Agent = Decision maker or controller.

Decision/

Control/DP

Principle of

Optimality

Markov Decision

Problems

POMDP

Policy Iteration

Value Iteration

Complementary

Late 80s-Early 90s

AIRL

Learning through

Data/Experience

Figure 1.7.1 A schematic illustration of the synergy of ideas between decision and control on one hand, and artificial intelligence on the other.

<!-- image -->

- (c) Action = Decision or control.
- (d) Reward of a stage = (Opposite of) Cost of a stage.
- (e) State value = (Opposite of) Cost starting from a state.
- (f) Value (or reward) function = (Opposite of) Cost function.
- (g) Maximizing the value function = Minimizing the cost function.
- (h) Action (or state-action) value = Q-factor (or Q-value) of a statecontrol pair. (Q-value is also used often in RL.)
- (i) Planning = Solving a DP problem with a known mathematical model. (Often related to MPC and approximation in value space.)
- (j) Learning = Solving a DP problem without using a known mathematical model. (This is the principal meaning of the term 'learning' in AI/RL. Other meanings are also common.)
- (k) Self-learning (or self-play in the context of games) = Solving a DP problem using some form of policy iteration.
- (l) Deep reinforcement learning = Approximate DP using value and/or policy approximation with deep neural networks.
- (m) Prediction = Policy evaluation.
- (n) Generalized policy iteration = Optimistic policy iteration.
- (o) State abstraction = State aggregation.
- (p) Temporal abstraction = Time aggregation.
- (q) Learning a model = System identification.

- (r) Episodic task or episode = Finite-step system trajectory.
- (s) Continuing task = Infinite-step system trajectory.
- (t) Experience replay = Reuse of samples in a simulation process.
- (u) Bellman operator = DP mapping or operator.
- (v) Backup = Applying the DP operator at some state.
- (w) Sweep = Applying the DP operator at all states.
- (x) Greedy policy with respect to a cost function J = Minimizing policy in the DP expression defined by J .
- (y) Afterstate = Post-decision state.
- (z) Ground truth = Empirical evidence or information provided by direct observation.

Some of the preceding terms will be introduced in future chapters; see also the RL textbook [Ber19a]. The reader may then wish to return to this section as an aid in connecting with the relevant RL literature.

## 1.7.2 Di ff erences in Notation

Unfortunately, the confusion caused by di ff ering terminology has been further compounded by the use of inconsistent notations across various sources. This book adheres to the 'standard' notation that emerged during the Bellman/Pontryagin optimal control era; see e.g., the books by Athans and Falb [AtF66], Bellman [Bel67], and Bryson and Ho [BrH75]. This notation is consistent with the author's other DP books and is the most appropriate for a unified treatment of the subject, which simultaneously addresses discrete and continuous spaces problems.

A summary of the most prominently used symbols in our notational system is as follows:

- (a) x : state (also i for finite-state systems).
- (b) u : control.
- (c) w : stochastic disturbance.
- (d) J : cost function.
- (e) f : system function. For deterministic systems,

<!-- formula-not-decoded -->

and for stochastic systems,

<!-- formula-not-decoded -->

Also f k in place of f for time-varying systems.

- (f) g : cost per stage [ g ( x↪ u ) for deterministic systems, and g ( x↪ u↪ w ) for stochastic systems; also g k in place of g for time-varying systems].
- (g) p xy ( u ): transition probability from state x to state y under control u in finite-state systems [also p ij ( u )].
- (h) α : discount factor in discounted problems.

The x -u -J notation is standard in deterministic optimal control textbooks (e.g., the classical books [AtF66] and [BrH75], noted earlier, as well as the more recent books by Stengel [Ste94], Kirk [Kir04], and Liberzon [Lib11]). The symbols f (system function) and g (cost per stage) are also widely used in both early and later optimal control literature (unfortunately the more natural symbol ' c ' has not been used much in place of ' g ' for the cost per stage).

The notations i (state) and p ij ( u ) (transition probability) are common in the discrete-state Markov decision process (MDP) and operations research literature. Sometimes the alternative notation p ( j ♣ i↪ u ) is used for the transition probabilities.

In the artificial intelligence literature, the focus is primarily on finitestate MDPs, particularly discounted and stochastic shortest path infinite horizon problems. The most commonly used notation is s for state, a for action, r ( s↪ a↪ s ′ ) for reward per stage, p ( s ′ ♣ s↪ a ) or p ( s↪ a↪ s ′ ) for transition probability from s to s ′ under action a , and γ for discount factor. While this notation is well-suited to finite-state problems, it is not ideal for continuous spaces models. The reason is that it requires the use of transition probability distributions defined over continuous spaces, and leads to more complex and less intuitive mathematics. Moreover, for deterministic problems, which lack a probabilistic component, the transition probability notation becomes cumbersome and unnecessary.

## 1.7.3 Relations Between DP and RL

When comparing the RL and DP methodologies, it is important to recognize that they are fundamentally connected by their shared focus on sequential decision making. Thus, any problem that can be addressed by DP can, in principle, also be addressed by RL, and vice versa.

One may argue that the RL algorithmic methodology is broader than that of DP. It includes the use of gradient descent and random search algorithms, simulation-based methods, statistical methods of sampling and performance evaluation, and neural network design and training ideas. However, methods of this type have also been considered in DP-related research and applications for many years, albeit less intensively.

In the artificial intelligence view of RL, a machine learns through trial and error by interacting with an environment. In practical terms,

Acommon description is that 'the machine learns sequentially how to make

this is more or less the same as what DP aims to do, but in RL there is often an emphasis on the presence of uncertainty and exploration of the environment. In the decision, control, and optimization community, there is a lot of interest in using RL methods to address intractable problems, including deterministic discrete/integer optimization, which need not involve data collection, interaction with the environment, uncertainty, and learning (adaptive control is the only decision and control problem type, where uncertainty and exploration arise in a significant way).

In terms of applications, DP was originally developed in the 1950s and 1960s as part of the then emerging methodologies of operations research and optimal control. These methodologies are now mature and provide important tools and perspectives, as well as a rich variety of applications, such as robotics, autonomous transportation, and aerospace, which can benefit from the use of RL. Moreover, DP has been used in a broad range of applications in industrial engineering, operations research, economics, and finance, so these applications can also benefit from the use of RL methods and perspectives.

At the same time, RL and machine learning have ushered opportunities for the application of DP techniques in new domains, such as machine translation, image recognition, knowledge representation, database organization, large language models, and automated planning, where they can have a significant practical impact. We may also add that RL has brought into the field of sequential decision making a fresh and ambitious spirit that has made possible the solution of problems thought to be well outside the capabilities of DP. Indeed, before the connections between RL and DP were recognized, large dimensional problems, like those involving a Euclidean state space of even moderate dimension, or POMDP problems, were considered totally intractable with the DP methodology.

## 1.7.4 Synergy Between Large Language Models and DP/RL

Can RL and large language models (LLMs) work synergistically? This is an important question, as these two AI paradigms operate with distinct methodologies, objectives, and capabilities. While RL focuses on sequential decision-making, LLMs specialize in natural language understanding and generation, including computer code.

In particular, RL is designed to optimize policies for sequential control tasks, excelling in applications such as robotics and resource allocation, where adaptive decision-making is essential. In contrast, LLMs process and generate human-like text, enabling them to perform tasks such as translation, summarization, sentiment analysis, and code generation. Moreover,

decisions that maximize a reward signal, based on the feedback received from the environment.'

by leveraging vast amounts of pre-trained knowledge, LLMs can generalize across diverse contexts.

Despite their di ff erences, the capabilities of RL and LLMs are complementary , making them powerful tools when used in combination. Let us now summarize ways in which the synergy between RL and LLMs can manifest itself in practice.

The advent of pre-trained transformers, such as ChatGPT, has revolutionized natural language processing. These transformers can undergo further refinement through o ff -line training to specialize in specific tasks or mitigate undesirable biases. Notably, RL methodologies, particularly policy-space approximation techniques, have played a crucial role in this refinement process, as discussed in Section 3.5 of Chapter 3. Thus, RL has been instrumental in enhancing the capabilities of LLMs through o ff -line optimization techniques.

Conversely, LLMs serve as catalysts for RL by injecting domain knowledge, improving interpretability, and enabling more human-aligned training. By leveraging natural language input, LLMs facilitate the design of RL policies that are more transparent and adaptable. Additionally, LLMs support RL applications by assisting with mathematical formulation, algorithm selection, and code generation. This interplay between the two fields continues to evolve, driving new innovations in AI.

In conclusion, the interaction of RL and LLMs is not merely additive, it is multiplicative: RL equips LLMs with the ability to learn from interaction and feedback, while LLMs equip RL with contextual awareness, explainability, accessibility, and code generation capability. Together, they pave the way for enhanced applications, which align with human intent, and can also communicate and explain their reasoning.

## 1.7.5 Machine Learning and Optimization

Machine learning and optimization are closely intertwined fields, sharing similar mathematical models and computational algorithms. However, they di ff er in their cultures and application contexts, so it is worth reflecting on their similarities and di ff erences.

Machine learning can be broadly categorized into three main types of methods, all of which involve the collection and use of data in some form:

- (a) Supervised learning : Here a dataset of many input-output pairs (also called labeled data) is collected. An optimization algorithm is used to create a parametrized function that fits well the data, as well as make accurate predictions on new, unseen data. Supervised learning problems are typically formulated as optimization problems, examples

Both fields are also closely connected to the field of statistical analysis. However, in this section, we will not focus on this connection, as it is less relevant to the content of this book.

of which we will see in Chapter 3. A common algorithmic approach is to use a gradient-type algorithm to minimize a loss function that measures the di ff erence between the actual outputs of the dataset and the predicted outputs of the parametrized model.

- (b) Unsupervised learning : Here the dataset is 'unlabeled' in the sense that the data are not separated into input and matching output pairs. Unsupervised learning algorithms aim to identify patterns or structures within the data, which is useful for tasks like clustering, dimensionality reduction, and density estimation. The objective is to extract meaningful insights from the data. Some unsupervised learning methods can be related to DP, but the connection is not strong. Generally speaking, unsupervised learning does not seem to align well with the types of sequential decision making applications of this book.
- (c) Reinforcement learning : RL di ff ers in an important way from supervised and unsupervised learning. It does not use a dataset as a starting point . Instead, it generates data on-line or o ff -line as dictated by the needs of the optimization algorithm it uses, be it multistep lookahead minimization, approximate policy iteration and rollout, or approximation in policy space. A further complication in RL is that the generated data depends on the policy that is used to control the system. Ideally, the data should be generated using an optimal or near-optimal policy, but such a policy is unlikely to be available. We are thus forced to collect data using a sequence of (hopefully) improving policies, which is the essence of the approximate policy iteration algorithm of DP. This is a primary reason why this algorithm and its variations will be a focal point for our discussions in this book.

Another type of machine learning approach, which relates to DP/RL methods, is semi-supervised learning . It involves training a model using a dataset containing both labeled and unlabeled data. Here, some initial labeled data are sequentially augmented with unlabeled data, with the aim of constructing an 'informative' data set that enhances machine learning tasks such as classification. This approach lies between supervised learning (which requires all data to be labeled) and unsupervised learning (which works with exclusively unlabeled data). Semi-supervised learning is related to the field of active learning , where DP-like methods are used to augment sequentially the labeled set; see e.g., the monograph by Zhu and Goldberg [ZhG22], the survey by Van Engelen and Hoos [VaH20], and the illustrative application papers by Marchesoni-Acland et al. [MMK23], and Bhusal, Miller, and Merkurjev [BMM24].

Optimization problems and algorithms on the other hand may or may not involve the collection and use of data. They involve data only in the

A variant of RL called o ffl ine RL or batch RL , starts from a historical dataset, and does not explore the environment to collect new data.

context of special applications, most of which are related to machine learning. In theoretical terms, optimization problems are categorized in terms of their mathematical structure, which is the primary determinant of the suitability of particular types of methods for their solution. In particular, it is common to distinguish between static optimization problems and dynamic optimization problems . The latter problems involve sequential decision making, with feedback between decisions, while the former problems involve a single decision.

Stochastic problems with perfect or imperfect state observations are dynamic (unless they involve open-loop decision making without the use of any feedback), and they require the use of DP for their optimal solution. Deterministic problems can be formulated as static, but they can also be formulated as dynamic for reasons of algorithmic expediency. In this case, the decision making process is (sometimes artificially) broken down into stages, as is often done in this book for discrete optimization and other contexts.

Another important categorization of optimization problems is based on whether their search space is discrete or is continuous . Discrete problems include deterministic problems such as integer and combinatorial optimization problems, and can be addressed by formal methods of integer programming as well as by DP. These problems tend to be challenging, so they are often addressed (suboptimally) with the use of heuristics.

Continuous problems are usually addressed with very di ff erent methods, which are based on calculus and convexity, such as Lagrange multiplier theory and duality, and the computational machinery of linear, nonlinear, and convex programming. Some discrete problems, particularly those that involve graphs (such as matching, transportation, and transshipment problems), can be addressed using continuous spaces network optimization methods that rely on linear programming and duality. Hybrid problems, which combine discrete and continuous variables, usually require discrete optimization techniques, but can also benefit from convex duality methods, which are fundamentally continuous.

The DP methodology, generally speaking, applies to just about any kind of optimization problem, deterministic or stochastic, static or dynamic, discrete or continuous, as long as it is formulated as a sequential decision problem , in the manner described in Sections 1.2-1.4. In terms of algorithmic structure, DP di ff ers significantly from other optimization techniques, particularly those based on calculus and convexity. Notably, DP can handle both discrete and continuous problems and is not concerned with local minima, focusing instead on finding global minima.

Notice a qualitative di ff erence between optimization and machine learning: the former is mostly organized around mathematical structures and the analysis of the foundational issues of the corresponding algorithms, while the latter is mostly organized around how data is collected, used, and analyzed, often with a strong emphasis on statistical issues . This is an im-

portant distinction, which a ff ects profoundly the perspectives of researchers in the two fields.

## 1.7.6 Mathematics in Machine Learning and Optimization

Let us now discuss some di ff erences between the research cultures of the optimization and machine learning fields, as they pertain to the use of mathematics. In optimization, the emphasis is often on general purpose methods that o ff er broad and mathematically rigorous performance guarantees, for a wide variety of problems. In particular, it is generally believed that a solid mathematical foundation for a given optimization methodology enhances its reliability and clarifies the boundaries of its applicability. Furthermore, it is recognized that formulating practical problems and matching them to the right algorithms is greatly enhanced by one's understanding of the mathematical structure of the underlying optimization methodology.

Machine learning research includes important lines of analysis with a strongly mathematical character, particularly relating to theoretical computer science, complexity theory, and statistical analysis. At the same time, in machine learning there are eminently useful algorithmic structures, such as neural networks, large language models, and image generative models, which are not well-understood mathematically and defy to a large extent mathematical analysis. This can add to a perception that focusing on rigorous mathematics, as opposed to practical implementation, may be a low payo ff investment in many practical machine learning contexts.

Moreover, the starting point in machine learning is often a specific dataset or a specialized type of training problem (e.g., language translation or image recognition). The priority is to find a method that works well for that specific dataset or problem, even if it is not generalizable to others. Thus specialized approximation architectures, implementation techniques, and heuristics, which perform well for the given problem and dataset type, may be perfectly acceptable in a machine learning context, even if they do not provide rigorous and generally applicable performance guarantees.

In conclusion, both optimization and machine learning involve mathematical models and rigorous analysis in important ways, and often overlap in the techniques and tools that they use, as well as in the practical applications that they address. However, depending on the problem at hand, there may be di ff erences in the emphasis and priority placed on mathe-

As an illustration, the paper by He et al., 'Deep Residual Learning for Image Recognition,' published in Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition, 2016, has been cited over 296,276 times as of November 2025, and contains only two equations. The famous neural network architecture paper by Vaswani et al., 'Attention is all you Need,' published in NIPS, 2017, which laid the foundation for GPT, has been cited over 205,945 times as of November 2025, and contains only six equations.

matical analysis, insight, and generality versus practical e ff ectiveness and problem-specific e ffi ciency. This can lead to some tension, as di ff erent fields may not fully appreciate each other's perspective.

## 1.8 NOTES, SOURCES, AND EXERCISES

We will now summarize this chapter and describe how it can be used flexibly as a foundation for a few di ff erent courses. We will also provide a selective overview of the DP and RL literature, and give a few exercises that have been used in ASU classes.

## Chapter Summary

In this chapter, we have aimed to provide an overview of the approximate DP/RL landscape, which can serve as the foundation for a deeper in-class development of other RL topics. In particular, we have described in varying levels of depth the following:

- (a) The algorithmic foundation of exact DP in all its major forms: deterministic and stochastic, discrete and continuous, finite and infinite horizon.
- (b) Approximation in value space with one-step and multistep lookahead, the workhorse of RL, which underlies its major success stories, including AlphaZero. We contrasted approximation in value space with approximation in policy space, and discussed how the two may be combined.
- (c) The important division between o ff -line training and on-line play in the context of approximation in value space. We highlighted how their synergy can be intuitively explained in terms of Newton's method.
- (d) The fundamental methods of policy iteration and rollout, the former being primarily an o ff -line method, and the latter being primarily a less ambitious on-line method. Both methods and their variants bear close relation to Newton's method and draw their e ff ectiveness from this relation.
- (e) Some major models with a broad range of applications, such as discrete optimization, POMDP, multiagent problems, adaptive control, and model predictive control. We delineated their principal characteristics and the major RL implementation issues within their contexts.
- (f) The use of function approximation, which has been a recurring theme in our presentation. We have touched upon some of the principal schemes for approximation, e.g., neural networks and feature-based architectures.

One of the principal aims of this chapter was to provide a foundational platform for a range of RL courses that explore at a deeper level various algorithmic methodologies, such as:

- (1) Rollout and policy iteration.
- (2) Neural networks and other approximation architectures for o ff -line training.
- (3) Aggregation, which can be used for cost function approximation in the context of approximation in value space.
- (4) A broader discussion of sequential decision making in contexts involving changing system parameters, sequential estimation, and simultaneous system identification and control.
- (5) Stochastic algorithms, such as temporal di ff erence methods and Qlearning, which can be used for o ff -line policy evaluation in the context of approximate policy iteration.
- (6) Sampling methods to collect data for o ff -line training in the context of cost and policy approximations.
- (7) Statistical estimates and e ffi ciency enhancements of various sampling methods used in simulation-based schemes. This includes confidence intervals and computational complexity estimates.
- (8) On-line methods for specially structured contexts, including problems of the multi-armed bandit type.
- (9) Simulation-based algorithms for approximation in policy space, including policy gradient and random search methods.
- (10) A deeper exploration of control system design methodologies such as model predictive control and adaptive control, and their applications in robotics and automated transportation.

In our course we have focused selectively on the methodologies (1)(4), with a limited coverage of (9) in Section 3.5. In a di ff erent course, other choices from the above list may be made, by building on the content of the present chapter.

## Notes and Sources for Individual Sections

In the literature review that follows, we will focus primarily on textbooks, research monographs, and broad surveys, which supplement our discussions, present related viewpoints, and collectively provide a guide to the literature. Inevitably, our selection reflects a certain cultural bias and an overemphasis on sources that are familiar to the author and aligned in style with this book (including the author's own works). We acknowledge in advance that this may lead to omissions of research references that fall outside

our own understanding and perspective on the field, and we apologize for any such exclusions.

Sections 1.1-1.4 : Our discussion of exact DP in this chapter has been brief since our focus in this book will be on approximate DP and RL. For a more comprehensive treatment of finite-horizon exact DP and its applications to both discrete and continuous space problems, the author's DP textbook [Ber17a] provides an extensive overview, using notation and style consistent with this book. The books by Puterman [Put94] (written from an operations research perspective) and by the author [Ber12] (written from a decision and control perspective) provide detailed (but substantially different) treatments of infinite horizon finite-state stochastic DP problems. The book [Ber12] also covers continuous/infinite state and control spaces problems, including the linear quadratic problems that we have discussed for one-dimensional problems in this chapter. Continuous spaces problems present special analytical and computational challenges, which are at the forefront of research of the RL methodology. The author's 1976 DP textbook [Ber76] was the first to develop discrete-time DP within a general framework that allows arbitrary state, control, and disturbance spaces.

Some of the more complex mathematical aspects of exact DP were addressed in the monograph by Bertsekas and Shreve [BeS78], particularly the probabilistic/measure-theoretic issues associated with stochastic optimal control, including partial state information problems. This monograph provides an extensive treatment of these issues. The followup work by Huizhen Yu and the author [YuB15] addresses the special measurability issues that relate to policy iteration, and provides further analysis relating to the convergence of value iteration. The second volume of the author's DP book [Ber12], Appendix A, includes an accessible summary introduction of the measure-theoretic framework of the book [BeS78]. In the RL literature, the mathematical di ffi culties around measurability are usually

The rigorous mathematical theory of stochastic optimal control, including the development of an appropriate Borel space measure-theoretic framework, originated in the 60s, with the work of Blackwell [Bla65], [Bla67]. It relies on the theory of analytic sets of descriptive set theory, introduced in 1917 by M. Suslin, a young Russian mathematician, and further developed by his mentor N. Luzin. It culminated in the Bertsekas and Shreve monograph [BeS78], which provides the now 'standard' framework, based on the formalism of Borel spaces, lower semianalytic functions, and universally measurable policies. This development involves daunting mathematical complications, which stem, among others, from the observation that when a Borel measurable function F ( x↪ u ), of the two variables x and u , is minimized with respect to u , the resulting function G ( x ) = min u F ( x↪ u ) need not be Borel measurable (it belongs to the broader class of lower semianalytic functions); this key fact was the starting point of Suslin's analysis. Moreover, even if the minimum is attained by several policies θ , i.e., G ( x ) = F ( x↪ θ ( x ) ) for all x , it is possible that none of these θ is Borel

neglected (as they are in this book), and this is fine because they do not play an important role in practical applications. Moreover, measurability issues do not arise for problems involving finite or countably infinite state and control spaces. We note, however, that there are quite a few published works in RL as well as exact DP, which purport to address measurability issues with a mathematical narrative that is either confusing or plain incorrect.

The third edition of the author's abstract DP monograph [Ber22b], expands on the original 2013 first edition, and aims at a unified development of the core theory and algorithms of total cost sequential decision problems. It addresses simultaneously stochastic, minimax, game, risksensitive, and other DP problems, through the use of abstract DP operators (or Bellman operators as we call them here). The idea is to gain insight through abstraction. In particular, the structure of a DP model is encoded in its abstract Bellman operator, which serves as the 'mathematical signature' of the model. Thus, characteristics of this operator (such as monotonicity and contraction) largely determine the analytical results and computational algorithms that can be applied to that model. Abstract DP ideas are also useful for visualizations and interpretations of RL methods using the Newton method formalism that we have discussed somewhat briefly in this book in the context of linear quadratic problems.

Approximation in value space, rollout, and policy iteration are the principal subjects of this book. These are very powerful and general techniques: they can be applied to deterministic and stochastic problems, finite and infinite horizon problems, discrete and continuous spaces problems, and mixtures thereof. Moreover, rollout is reliable, easy to implement, and can be used in conjunction with on-line replanning. It is also compatible measurable (however, there does exist a minimizing policy that belongs to the broader class of universally measurable policies). Thus, starting with a Borel measurability framework for cost functions and policies, we quickly get outside that framework when executing DP algorithms, such as value and policy iteration. The broader framework of universal measurability, introduced in [BeS78], is required to correct this deficiency, in the absence of additional (fairly strong) assumptions.

The name 'rollout' (also called 'policy rollout') was introduced by Tesauro and Galperin [TeG96] in the context of rolling the dice in the game of backgammon. In Tesauro's proposal, a given backgammon position is evaluated by 'rolling out' many games starting from that position to the end of the game. To quote from the paper [TeG96]: 'In backgammon parlance, the expected value of a position is known as the 'equity' of the position, and estimating the equity by Monte-Carlo sampling is known as performing a 'rollout.' This involves playing the position out to completion many times with di ff erent random dice sequences, using a fixed policy to make move decisions for both sides.'

with new and exciting technologies such as transformer networks and large language models (see Section 2.3.7).

As we have noted, rollout with a given base policy is simply the first iteration of the policy iteration algorithm starting from the base policy. Truncated rollout can be interpreted as an 'optimistic' form of a single policy iteration, whereby a policy is evaluated inexactly, by using a limited number of value iterations; see the books [Ber20a], [Ber22a].

Policy iteration, which can be seen as repeated rollout, is more ambitious and challenging than rollout. It requires o ff -line training, possibly in conjunction with the use of neural networks. Together with its neural network and distributed implementations, it will be discussed in more detail later. Note that rollout does not require any o ff -line training, once the base policy is available; this is its principal advantage over policy iteration.

Section 1.5: There is a vast literature on linear quadratic problems. The connection of policy iteration with Newton's method within this context was first derived by Kleinman [Kle68], as part of his doctoral research at MIT, under the supervision of M. Athans. Kleinman's work focused on continuous-time linear quadratic problems (see Hewer [Hew71] for the discrete-time case). For followup work, which relates to approximate policy iteration, see Feitzinger, Hylla, and Sachs [FHS09], and Hylla [Hyl11].

The general relation of approximation in value space with Newton's method, beyond policy iteration, and its connections with MPC and adaptive control was first presented in the author's book [Ber20a], the papers [Ber21b], [Ber22c], and in the book [Ber22a], which contains an extensive discussion. This relation provides the starting point for an in-depth understanding of the synergy between the o ff -line training and the on-line play components of the approximation in value space architecture, including the role of multistep lookahead in enhancing the starting point of the Newton

Truncated rollout was also proposed in the context of backgammon in the paper [TeG96]. To quote from this paper: 'Using large multi-layer networks to do full rollouts is not feasible for real-time move decisions, since the large networks are at least a factor of 100 slower than the linear evaluators described previously. We have therefore investigated an alternative Monte-Carlo algorithm, using so-called 'truncated rollouts.' In this technique trials are not played out to completion, but instead only a few steps in the simulation are taken, and the neural net's equity estimate of the final position reached is used instead of the actual outcome. The truncated rollout algorithm requires much less CPU time, due to two factors: First, there are potentially many fewer steps per trial. Second, there is much less variance per trial, since only a few random steps are taken and a real-valued estimate is recorded, rather than many random steps and an integer final outcome. These two factors combine to give at least an order of magnitude speed-up compared to full rollouts, while still giving a large error reduction relative to the base player.' Analysis and computational experience with truncated rollout since 1996 are consistent with the preceding assessment.

step. The monograph [Ber22a] (Appendix A) also provides analysis of variants of Newton's method for nondi ff erentiable fixed point problems, such as the ones arising in Bellman's equation (which involves a nondi ff erentiable right-hand side in finite-control space problems, among others).

Note that in approximation in value space, we are applying Newton's method to the solution of a system of equations (the Bellman equation). This context has no connection with the 'gradient descent' methods that are popular for the solution of special types of optimization problems in RL, arising for example in neural network training problems (see Chapter 3). In particular, there are no gradient descent methods that can be used for the solution of systems of equations such as the Bellman equation. There are, however, 'first order' deterministic algorithms such as the Gauss-Seidel and Jacobi methods (and stochastic asynchronous extensions) that can be applied to the solution of systems of equations with special structure, including Bellman equations. Such methods include various Q-learning algorithms, which are discussed in the neuro-dynamic programming book by Bertsekas and Tsitsiklis [BeT89], as well as the recent book by Meyn [Mey22]. While these methods can be useful, they are much slower than Newton's method and have limited utility in the context of on-line play.

Section 1.6: Many applications of DP are discussed in the 1st volume of the author's DP book [Ber17a]. This book also covers a broad variety of state augmentation and problem reformulation techniques, including the mathematics of how problems with imperfect state information can be transformed to perfect state information problems. In Section 1.6 we have aimed to provide an overview, with an emphasis on the use of approximations. In what follows we provide some related historical notes.

Multiagent problems : This subject has a long history (Marschak [Mar55], Radner [Rad62], Witsenhausen [Wit68], [Wit71a], [Wit71b]), and was researched extensively in the 70s; see the review paper by Ho [Ho80] and the references cited there. The names used at that time were team theory and decentralized control . For a sampling of subsequent works in team theory and multiagent optimization, we refer to the papers by Krainak, Speyer, and Marcus [KLM82a], [KLM82b], and de Waal and van Schuppen [WaS00]. For more recent works, see Nayyar, Mahajan, and Teneketzis [NMT13], Nayyar and Teneketzis [NaT19], Li et al. [LTZ19], Qu and Li [QuL19], Gupta [Gup20], the book by Zoppoli, Sanguineti, Gnecco, and Parisini [ZSG20], and the references quoted there. In addition to the aforementioned works, surveys of multiagent DP from an RL perspective were given by Busoniu, Babuska, and De Schutter [BBD08], [BBD10b].

The term 'multiagent' has been used with various meanings in the literature. Some authors emphasize scenarios where agents lack common information when making their decisions, leading to sequential decision problems with 'nonclassical information patterns.' These problems are particularly complex because they cannot be solved using exact DP techniques.

Other authors focus on situations where the agents are 'weakly' coupled through the system equation, the cost function, or the constraints. They consider methods that exploit the weak coupling to address the problem with (suboptimal) decoupled computations.

Agent-by-agent minimization in multiagent approximation in value space and rollout was proposed and theoretically justified in the author's paper [Ber19c], which also discusses extensions to infinite horizon policy iteration algorithms, and explores connections with the concept of personby-person optimality from team theory; see also the textbook [Ber20a], the papers [Ber19d], [Ber20b] for further elaboration. The papers by Bhattacharya et al. [BKB23], Di Gennaro at al. [DBF22], Di Gennaro at al. [DBP23], Garces et al. [GBG22], Macesker et al. [MPL23], and Weber at al. [WGP23] present computational studies with challenging problems, where several of the multiagent algorithmic ideas were adapted, tested, and validated. These papers consider large-scale multi-robot and vehicle routing problems, involving partial state information, and explore some of the attendant implementation issues, including autonomous multiagent rollout, through the use of policy neural networks and other precomputed signaling policies. They also compare the performance of these multiagent methods against alternative approaches, including those based on policy gradient techniques.

A di ff erent form of distributed computation and multiagent optimization, where each agent has a partial/local model of the system within part of the state space and relies on aggregate information from other agents for DP computations, is proposed in the author's DP book [Ber12], Section 6.5.4; see also Section 3.6.8 of the present book.

Adaptive control : The research on adaptive control has a long history and its literature is very extensive; see the books by Astr  om and Wittenmark [AsW94], Astr  om and Hagglund [AsH06], Bodson [Bod20], Goodwin and Sin [GoS84], Ioannou and Sun [IoS96], Jiang and Jiang [JiJ17], Krstic, Kanellakopoulos, and Kokotovic [KKK95], Kumar and Varaiya [KuV86], Liu, et al. [LWW17], Lavretsky and Wise [LaW13], Narendra and Annaswamy [NaA12], Sastry and Bodson [SaB11], Slotine and Li [SlL91], and Vrabie, Vamvoudakis, and Lewis [VVL13]. These books describe a vast array of methods spanning 60 years, and ranging from adaptive and PID model-free approaches, to simultaneous or sequential control and identification, to time series models, to extremum-seeking methods, to simulationbased RL techniques, etc.

The ideas of PID control have been applied widely to adaptive and robust control contexts, and have a long history; see the books by Astr  om and Hagglund [AsH95], [AsH06], which provide many references. According to Wikipedia, 'a formal control law for what we now call PID or three-term control was first developed using theoretical analysis, by Russian American engineer Nicolas Minorsky' in 1922 [Min22].

The DP framework for adaptive control was introduced in a series of papers by Feldbaum, starting in 1960 with [Fel60], under the name dual control theory . These papers emphasized the division of e ff ort between system estimation and control, now more commonly referred to as the exploration-exploitation tradeo ff . In the last paper of the series [Fel63], Feldbaum prophetically concluded as follows: 'At the present time, the most important problem for the immediate future is the development of approximate solution methods for dual control theory problems, the formulation of sub-optimal strategies, the determination of the numerical value of risk in quasi-optimal systems and its comparison with the value of risk in existing systems.'

The research on problems involving unknown models and using data for model identification simultaneously with control was rekindled with the advent of the artificial intelligence side of RL and its focus on the active exploration of the environment. Here there is emphasis on 'learning from interaction with the environment' [SuB18] through the use of (possibly hidden) Markov decision models, machine learning, and neural networks, in a wide array of methods that are under active development at present. This is more or less the same as the classical problems of dual and adaptive control that have been discussed in the control literature since the 60s.

The formulation of adaptive and dual control problems as POMDP (cf. Section 2.11) is classical. The use of rollout within this context was first suggested in the author's book [Ber22a], Section 6.7.

Model predictive control : The idea underlying MPC, on-line optimization with a truncated rolling horizon and a terminal cost function approximation, has arisen in several contexts, motivated by di ff erent types of applications. It has been part of the folklore of the control theory and operations research literature, dating to the 1960s and 1970s. Simultaneously, it was used in important chemical process control applications, where the name 'model predictive control' (or 'model-based predictive control') and the related name 'dynamic matrix control' were introduced. The term 'predictive' arises often in this path breaking literature, and generally refers to taking into account the system's future, while applying control in the present. Related ideas appeared independently in the computer science literature, in contexts of heuristic search, planning, and game playing. The history of the subject within the decision and control domain is recounted in the paper by Morari [Mor25].

The literature on the theory and applications of MPC is voluminous. Some early widely cited papers are Richalet et al. [RRT78], Cutler and Ramaker [CuR80], Rouhani and Mehra [RoM82], Clarke, Mohtadi, and Tu ff s [CMT87a], [CMT87b], Keerthi and Gilbert [KeG88], Mayne and Michalska [MaM88], Rawlings and Muske [RaM93], Mayne et al. [MRR00]. For early surveys, see Morari and Lee [MoL99], and Findeisen et al. [FIA03]. For a more recent review, which addresses issues of tube MPC and ro-

bustness among others, see Mayne [May14]. Textbooks on MPC include Maciejowski [Mac02], Goodwin, Seron, and De Dona [GSD06], Camacho and Bordons [CaB07], Kouvaritakis and Cannon [KoC16], Borrelli, Bemporad, and Morari [BBM17], Rawlings, Mayne, and Diehl [RMD17]. The textbook by Stephanopoulos [Ste25] focuses on applications of MPC and related algorithms in process and biological systems control. The connections between MPC and rollout were discussed in the author's paper [Ber05a]. For an extensive survey of the DP/Newton step framework of this chapter, as applied to MPC, see the author's paper [Ber24].

## Reinforcement Learning Sources

The first DP/RL books were written in the 1990s, setting the tone for subsequent developments in the field. One in 1996 by Bertsekas and Tsitsiklis [BeT96], which reflects a decision, control, and optimization viewpoint, and another in 1998 by Sutton and Barto, which is culturally di ff erent and reflects an artificial intelligence viewpoint (a 2nd edition, [SuB18], was published in 2018). We refer to the former book and also to the author's DP textbooks [Ber12], [Ber17a] for a broader discussion of some of the topics of this book, including algorithmic convergence issues and additional DP models, such as those based on average cost and semi-Markov problem optimization. Note that both of these books deal with finite-state Markovian decision models and use a transition probability notation, as they do not address continuous spaces problems, which are one of the major focal points of this book.

More recent books are by Gosavi [Gos15] (a much expanded 2nd edition of his 2003 monograph), which emphasizes simulation-based optimization and RL algorithms, Cao [Cao07], which focuses on a sensitivity approach to simulation-based methods, Chang, Fu, Hu, and Marcus [CFH13] (a 2nd edition of their 2007 monograph), which emphasizes finite-horizon/multistep lookahead schemes and adaptive sampling, Busoniu, Babuska, De Schutter, and Ernst [BBD10a], which focuses on function approximation methods for continuous space systems and includes a discussion of random search methods, Szepesvari [Sze10], which is a short monograph that selectively treats some of the major RL algorithms such as temporal di ff erences, armed bandit methods, and Q-learning, Powell [Pow11], which emphasizes resource allocation and operations research applications, Powell and Ryzhov [PoR12], which focuses on specialized topics in learning and Bayesian optimization, Vrabie, Vamvoudakis, and Lewis [VVL13], which discusses neural network-based methods and on-line adaptive control, Kochenderfer et al. [KAC15], which selectively discusses applications and approximations in DP and the treatment of uncertainty, Jiang and Jiang [JiJ17], which addresses adaptive control and robustness issues within an approximate DP framework, Liu, Wei, Wang, Yang, and Li [LWW17], which deals with forms of adaptive dynamic programming, and topics in both RL and optimal control, and Zoppoli, Sanguineti, Gnecco,

and Parisini [ZSG20], which addresses neural network approximations in optimal control as well as multiagent/team problems with nonclassical information patterns. The book by Meyn [Mey22] focuses on the connections of RL and optimal control, from a mathematical perspective, and treats stochastic problems and algorithms in more detail.

There are also several books that, while not exclusively focused on DP and/or RL, touch upon some of the topics of the present book. The book by Borkar [Bor08] is an advanced monograph that addresses rigorously many of the convergence issues of iterative stochastic algorithms in approximate DP, mainly using the so-called ODE approach. The book by Meyn [Mey07] is broader in its coverage, but discusses some of the popular approximate DP/RL algorithms. The book by Haykin [Hay08] discusses approximate DP in the broader context of neural network-related subjects. The book by Krishnamurthy [Kri16] focuses on partial state information problems, with a discussion of both exact DP, and approximate DP/RL methods. The textbooks by Kouvaritakis and Cannon [KoC16], Borrelli, Bemporad, and Morari [BBM17], and Rawlings, Mayne, and Diehl [RMD17] collectively provide a comprehensive view of the MPC methodology. The book by Lattimore and Szepesvari [LaS20] is focused on multiarmed bandit methods. The book by Brandimarte [Bra21] is a tutorial introduction to DP/RL that emphasizes operations research applications and includes MATLAB codes. The book by Hardt and Recht [HaR21] focuses on broader subjects of machine learning and covers selectively approximate DP and RL topics.

The present book is similar in style, terminology, and notation to the author's recent textbooks [Ber19a] (Reinforcement Learning and Optimal Control), [Ber20a] (Rollout and Policy Iteration), [Ber22a] (Lessons from AlphaZero), and the 3rd edition of the abstract DP monograph [Ber22b], which collectively provide a fairly comprehensive and more mathematical account of the subject. In particular, the book [Ber19a] includes a broader coverage of approximation in value space methods, including certainty equivalent control and aggregation methods. It also addresses approximation in policy space in greater detail than the present book. The book [Ber20a] focuses more closely on rollout, policy iteration, and multiagent problems, and introduced the connection of approximation in value space with Newton's method. The book [Ber22a] focuses primarily on this connection, relying on analysis first provided in the book [Ber20a] and the paper [Ber22c]. The abstract DP monograph [Ber22b] (a 3rd edition of the original 2013 1st edition) is an advanced treatment of exact DP, which provides the mathematical framework of Bellman operators that are central for some of the Newton method visualizations presented in the present book and in the books [Ber20a], [Ber22a].

In addition to textbooks, there are many surveys and short research monographs relating to our subject, which are rapidly multiplying in number. Influential early surveys were written, from an artificial intelligence viewpoint, by Barto, Bradtke, and Singh [BBS95] (which dealt with the

methodologies of real-time DP and its antecedent, real-time heuristic search [Kor90], and the use of asynchronous DP ideas [Ber82a], [Ber83], [BeT89] within their context), and by Kaelbling, Littman, and Moore [KLM96] (which focused on general principles of RL). The volume by White and Sofge [WhS92] also contains surveys describing early work in the field.

Several overview papers in the volume by Si, Barto, Powell, and Wunsch [SBP04] describe some approximation methods that we will not be covering in much detail in this book: linear programming approaches (De Farias [DeF04]), large-scale resource allocation methods (Powell and Van Roy [PoV04]), and deterministic optimal control approaches (Ferrari and Stengel [FeS04], and Si, Yang, and Liu [SYL04]). Updated accounts of these and other related topics are given in the survey collections by Lewis, Liu, and Lendaris [LLL08], and Lewis and Liu [LeL13].

Recent extended surveys and short monographs are Borkar [Bor09] (a methodological point of view that explores connections with other Monte Carlo schemes), Lewis and Vrabie [LeV09] (a control theory point of view), Szepesvari [Sze10] (which discusses approximation in value space from a RL point of view), Deisenroth, Neumann, and Peters [DNP11], and Grondman et al. [GBL12] (which focus on policy gradient methods), Browne et al. [BPW12] (which focuses on Monte Carlo Tree Search), Mausam and Kolobov [MaK12] (which deals with Markovian decision problems from an artificial intelligence viewpoint), Ge ff ner and Bonet [GeB13], and Moerland et al. [MBP23] (which deal with methods related to search and automated planning), Schmidhuber [Sch15], Arulkumaran et al. [ADB17], Li [Li17], Busoniu et al. [BDT18], and Caterini and Chang [CaC18] (which deal with reinforcement learning schemes that are based on the use of deep neural networks), Recht [Rec18a] (which discusses continuous spaces optimal control), and the author's [Ber05a] (which focuses on rollout algorithms and MPC), [Ber11a] (which focuses on approximate policy iteration), [Ber18a] (which focuses on aggregation methods), [Ber20b] (which focuses on multiagent problems), and [Ber24] (which focuses on the relations between RL and MPC).

Figure 1.8.1 Solution of parts (a), (b), and (c) of Exercise 1.1. A 5-city traveling salesman problem illustration of rollout with the nearest neighbor base heuristic.

<!-- image -->

## E X E R C I S E S

## 1.1 (Computational Exercise - Traveling Salesman Problem)

Consider a modified version of the four-city traveling salesman problem of Example 1.2.3, where there is a fifth city E. The intercity travel costs are shown in Fig. 1.8.1, which also gives the solutions to parts (a), (b), and (c).

- (a) Use exact DP with starting city A to verify that the optimal tour is ABDECA with cost 20.
- (b) Verify that the nearest neighbor heuristic starting with city A generates the tour ACDBEA with cost 48.
- (c) Apply rollout with one-step lookahead minimization, using as base heuristic the nearest neighbor heuristic. Show that it generates the tour AECDBA with cost 37.

Illustration of the algorithm : At city A, the nearest neighbor heuristic generates the tour ACDBEA with cost 48, as per part (b). At city A, the rollout algorithm considers the four options of moving to cities B, C, D, E, or equivalently to states AB, AC, AD, AE, and it computes the nearest neighbor-generated tours corresponding to each of these states. These tours are ABCDEA with cost 49, ACDBEA with cost 48, ADCEBA with cost

63, and AECDBA with cost 37. The tour AECDBA has the least cost, so the rollout algorithm moves to city E or equivalently to state AE.

At AE, the rollout algorithm considers the three options of moving to cities B, C, D, or equivalently to states AEB, AEC, AED, and it computes the nearest neighbor-generated tours corresponding to each of these states. These tours are AEBCDA with cost 42, AECDBA with cost 37, AEDCBA with cost 63. The tour AECDBA has the least cost, so the rollout algorithm moves to city C or equivalently to state AEC.

At AEC, the rollout algorithm considers the two options of moving to cities B, D, and compares the nearest neighbor-generated tours corresponding to each of these. These tours are AECBDA with cost 52 and AECDBA with cost 37. The tour AECDBA has the least cost, so the rollout algorithm moves to city D or equivalently to state AECD. Then the rollout algorithm has only one option and generates the tour AECDBA with cost 37.

- (d) Apply rollout with two-step lookahead minimization, using as base heuristic the nearest neighbor heuristic. This rollout algorithm operates as follows. For k = 1 ↪ 2 ↪ 3, it starts with a k -city partial tour, it generates every possible two-city addition to this tour, uses the nearest neighbor heuristic to complete the tour, and selects as next city to add to the k -city partial tour the city that corresponds to the best tour thus obtained (only one city is added to the current tour at each step of the algorithm, not two). Show that this algorithm generates the optimal tour.
- (e) Estimate roughly the complexity of the computations in parts (a), (b), (c), and (d), assuming a generic N -city traveling salesman problem. Answer : The exact DP algorithm requires O ( N N ) computation, since there are

<!-- formula-not-decoded -->

arcs in the DP graph to consider, and this number can be estimated as O ( N N ). The nearest neighbor heuristic that starts at city A performs O ( N ) comparisons at each of N stages, so it requires O ( N 2 ) computation. The rollout algorithm at stage k runs the nearest neighbor heuristic N -k times, so it must run the heuristic O ( N 2 ) times for a total computation of O ( N 4 ). Thus the rollout algorithm's complexity involves a low order polynomial increase over the complexity of the base heuristic, something that is generally true for practical discrete optimization problems. Note that even though this may represent a substantial increase in computation over the base heuristic, it is a potentially enormous improvement over the complexity of the exact DP algorithm.

## 1.2 (Computational Exercise - Linear Quadratic Problem)

In this problem we focus on the one-dimensional linear quadratic problem of Section 1.5 and the interpretation of approximation space as a Newton step for solving the Riccati equation. Consider the undiscounted linear quadratic problem with parameters a = 2, b = 1, q = 1, r = 5. For this problem:

- (a) Plot and solve graphically the Riccati equation as in Fig. 1.5.1.

- (b) Plot and solve graphically the Riccati equation corresponding to the linear policy θ ( x ) = -(3 glyph[triangleleft] 2) x .
- (c) Plot graphically the numerical solution of the Riccati equation by value iteration as in Fig. 1.5.3, using a starting point K 0 &lt; K ∗ and a starting point K 0 &gt; K ∗ .
- (d) Interpret graphically approximation in value space with one-step, two-step, and three-step lookahead as a Newton step in the manner of Figs. 1.5.7 and 1.5.8. Use cost function approximations ˜ Kx 2 with ˜ K &lt; K ∗ and ˜ K &gt; K ∗ . What is the region of stability, i.e., the set of ˜ K for which approximation in value space produces a stable policy under one-step, two-step, and threestep lookahead.
- (e) Plot the performance error ♣ K ˜ θ -K ∗ ♣ as a function of ♣ ˜ K -K ∗ ♣ for one-step, two-step, and three-step lookahead approximation in value space.
- (f) Plot graphical interpretations of rollout and truncated rollout in the manner of Figs. 1.5.10 and 1.5.11 using a stable starting linear policy of your choice.

## 1.3 (Computational Exercise - Spiders and Flies)

Consider the spiders and flies problem of Example 1.6.5 with two di ff erences: the five flies stay still (rather than moving randomly), and there are only two spiders, both of which start at the fourth square from the right at the top row of the grid of Fig. 1.6.10. The base policy is to move each spider one square towards its nearest fly, with distance measured by the Manhattan metric, and with preference given to a horizontal direction over a vertical direction in case of a tie. Apply the multiagent rollout algorithm of Section 1.6.7, and compare its performance with the one of the ordinary rollout algorithm, and with the one of the base policy. This problem is also discussed in Section 2.9.

## 1.4 (Computational Exercise - Exercising an Option)

This exercise deals with a computational comparison of the optimal policy, a heuristic policy, and on-line approximation in value space using the heuristic policy, in the context of a problem that involves the timing of the sale of a stock.

An investor has the option to sell a given amount of stock at any one of N time periods. The initial price of the stock is an integer x 0 . The price x k , if it is positive and it is less than a given positive integer value ¯ x , it evolves according to where p + and p -have known values with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If x k = 0, then x k +1 moves to 1 with probability p + , and stays unchanged at 0 with probability 1 -p + . If x k = ¯ x , then x k +1 moves to ¯ x -1 with probability p -, and stays unchanged at ¯ x with probability 1 -p -.

At each period k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 for which the stock has not yet been sold, the investor (with knowledge of the current price x k ), can either sell the stock at the current price x k or postpone the sale for a future period. If the stock has not been sold at any of the periods k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, it must be sold at period N at price x N . The investor wants to maximize the expected value of the sale. For the following computations, use reasonable values of your choice for N , p + , p -, ¯ x , and x 0 (you should choose x 0 between 0 and ¯ x ). You are encouraged to experiment with di ff erent sets of values. A set of values that you may try first is

<!-- formula-not-decoded -->

- (a) Formulate the problem as a finite horizon DP problem by identifying the state, control, and disturbance spaces, the system equation, the cost function, and the probability distribution of the disturbance. Write the corresponding exact DP algorithm, and use it to compute the optimal policy and the optimal cost as a function of x 0 .

Solution : The optimal reward-to-go is generated by the following DP algorithm:

<!-- formula-not-decoded -->

and for k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, if x k = 0, then

<!-- formula-not-decoded -->

if x k = ¯ x , then

<!-- formula-not-decoded -->

(since the price cannot go higher than ¯ x , once at ¯ x , but can go lower), and if 0 &lt; x k &lt; ¯ x , then

<!-- formula-not-decoded -->

The optimal policy is to sell at x k = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ¯ x -1, if x k attains the maximum in the above equation, and not to sell otherwise. When x k = 0, it is optimal not to sell, while when x k = ¯ x , it is optimal to sell.

The values of J ∗ k ( x k ) and the optimal policy are tabulated as shown in Fig. 1.8.2. For this figure, all the calculations are done for the following special case:

<!-- formula-not-decoded -->

These values are also used for parts (b) and (c). However, you are asked to solve the problem for di ff erent values as noted earlier. Note that for the problem to have an interesting solution, the problem data must be chosen so that the problem's policies are materially a ff ected by the presence of the upper and lower bounds on the price x k . As an example consider the case where

<!-- formula-not-decoded -->

12

* 10

11

3

NO

хо 2

2.303

Expected Rewards (Exact Dynamic Programming)

Expected Reward

5.001

5

4.015 4.008

3.088

2.255

1.614

1

5.8

4

3.044 3.026 3.013 3.004

3.064

2.208 2.162 2.118 2.078 2.043

1.54

1.464

1.169

2

1.385

4

3

2.016

1.141

3

4

6

Policy (Exact Dynamic Programming)

12

* 10

11

xo mNHO

D

1

D

2

3

D

D

4

S

S

S

5

Figure 1.8.2 Table of values of optimal reward-to-go, obtained by exact DP, and corresponding optimal policy [cf. the algorithm (1.96)-(1.99). Only the states x k that are reachable from x 0 at time k are considered (this is the state space for time k ).

<!-- image -->

Then the bounds 0 ≤ x k and x k ≤ ¯ x never become 'active,' and it can be verified that the optimal expected reward is J ∗ ( x 0 ) = x 0 , while all policies are optimal and attain this optimal expected reward.

- (b) Suppose the investor adopts a heuristic, referred to as base heuristic, whereby he/she sells the stock if its price is greater or equal to β x 0 , where β is some number with β &gt; 1. Write an exact DP algorithm to compute the expected value of the sale under this heuristic.

Solution : The reward-to-go for the base heuristic starting from state x k , denoted J x k k ( x k ), can be generated by the following (exact) DP algorithm. (Note here the use of superscript x k in the quantities J x k n ( x n ) computed by the algorithm. The reason is that the computed values J x k n ( x n ) depend on x k , which incidentally implies that base heuristic is not sequentially consistent, as defined later in Section 2.3.) The algorithm is given by

<!-- formula-not-decoded -->

and for n = k↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, if 0 &lt; x n &lt; β x k , then

<!-- formula-not-decoded -->

if x n = 0, then

S

10

4

3

1.062

2

4

3

2

1

N - 1

N = 10

x = 10

p+ = p = 0.25

Xo = 2

<!-- formula-not-decoded -->

12

11

× 10

3

хо 2

1

2.268

Expected Rewards (Exact DP Base Heuristic; B = 1.4)

Expected Reward

3

1.609

1

1.538

2.231

2.193

2.154

1.169

2

3

1.463

1.071

1.223

1.141

2. 113 2.07 2.043 2.020

0.966

1.385

1.305

0.854

0.73

0.594

4

6

7

Policy (Exact DP Base Heuristic; B = 1.4)

12

* 10

11

3

xo 2

1

D

0

D

1

D

D

3

1.062

2.0

0.438

3

1.0

2.0

0.25

N = 10

x = 10

x0 = 2

p+ = p = 0.25

Figure 1.8.3 Table of rewards-to-go for the base policy with β = 1 glyph[triangleright] 4, starting from x 0 [cf. the algorithm (1.100)-(1.103) for k = 0].

<!-- image -->

and if x n β x k

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The values of J x k k ( x k ) computed by this algorithm are shown in Fig. 1.8.3, together with the decisions applied by the base heuristic.

While the reward-to-go for the base heuristic starting from state x k is very simple to compute for our problem, in order to apply the rollout algorithm only the values J x k +1 k +1 ( x k +1), J x k k +1 ( x k ), and J x k -1 k +1 ( x k -1) need to be calculated for each state x k encountered during on-line operation. Moreover, the base heuristic's reward-to-go J x k k ( x k ) can also be computed on-line by Monte Carlo simulation for the relevant states x k . This would be the principal option in a more complicated problem where the exact DP algorithm is too time-consuming.

- (c) Apply approximation in value space with one-step lookahead minimization and with function approximation that is based on the heuristic of part (b). In particular, use ˜ J N ( x N ) = x N , and for k = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, use ˜ J k ( x k ) that is equal to the expected value of the sale when starting at x k and using the heuristic that sells the stock when its price exceeds β x k . Use exact DP as well as Monte Carlo simulation to compute/approximate on-line the needed values ˜ J k ( x k ). Compare the expected values of sale price computed with the optimal, heuristic, and approximation in value space methods. Solution : The rollout policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ is determined by the base heuristic, where for every possible state x k , and stage k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, the rollout decision ˜ θ k ( x k ) is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

otherwise. The sell or don't sell decision of the rollout algorithm is made on-line according to the preceding criterion, at each state x k encountered during on-line operation.

Figure 1.8.4 shows the rollout policy, which is computed by the preceding equations using the rewards-to-go of the base heuristic J x k k ( x k ), as given in Fig. 1.8.3. Once the rollout policy is computed, the corresponding reward function ˜ J k ( x k ) can be calculated similar to the case of the base heuristic. Of course, during on-line operation, the rollout decision need only be computed for the states x k encountered on-line.

The important observation when comparing Figs. 1.8.3 and 1.8.4 is that the rewards-to-go of the rollout policy are greater or equal to the ones for the base heuristic. In particular, starting from x 0 , the rollout policy attains reward 2.269, and the base heuristic attains reward 2.268. The optimal policy attains reward 2.4. The rollout policy reward is slightly closer to the optimal than the base heuristic reward.

The rollout reward-to-go values shown in Fig. 1.8.4 are 'exact,' and correspond to the favorable case where the heuristic rewards needed at x k , J x k +1 k +1 ( x k +1), J x k k +1 ( x k ), and J x k -1 k +1 ( x k -1), are computed exactly by DP or by infinite-sample Monte Carlo simulation.

When finite-sample Monte Carlo simulation is used to approximate the needed base heuristic rewards at state x k , i.e., J x k +1 k +1 ( x k +1), J x k k +1 ( x k ), and J x k -1 k +1 ( x k -1), the performance of the rollout algorithm will be degraded. In particular, by using a computer program to implement rollout with Monte Carlo simulation, it can be shown that when J x k +1 k +1 ( x k +1), J x k k +1 ( x k ), and J x k -1 k +1 ( x k -1) are approximated using a 20-sample Monte-Carlo simulation per reward value, the rollout algorithm achieves reward 2.264 starting from x 0 . This reward is evaluated by (almost exact) 400-sample Monte Carlo simulation of the rollout algorithm.

When J x k +1 k +1 ( x k + 1), J x k k +1 ( x k ), and J x k -1 k +1 ( x k -1) were approximated using a 200-sample Monte-Carlo simulation per reward value, the rollout algorithm achieves reward 2.273 [as evaluated by (almost exact) 400-sample Monte Carlo simulation of the rollout algorithm]. Thus with 20-sample simulation, the rollout algorithm performs worse than the base heuristic starting from x 0 . With the more accurate 200-sample simulation, the rollout algorithm performs better than the base heuristic starting from x 0 , and performs nearly as well as the optimal policy (but still somewhat worse than in the case where exact values of the needed base heuristic rewards are used (based on an 'infinite' number Monte Carlo samples).

It is worth noting here that the heuristic is not a legitimate policy because at any state x n is makes a decision that depends on the state x k where it started. Thus the heuristic's decision at x n depends not just on x n , but also on the starting state x k . However, the rollout algorithm is always an approximation in value space scheme with approximation reward ˜ J k ( x k ) defined by the heuristic, and it provides a legitimate policy.

12

2

Xo

HO

Expected Rewards (Rollout w/ Exact DP Base Heuristic; B = 1.4)

10

9

3

2

1

7

Expected Reward

5.0

6

3.084

3.062

1.286

4.015

2.231

1

3.043 3.026 3.013

1.208 1.165

6

4.004

2.193

0.355

1.248

2

9

4

3

2.236

2.016

4

3.004

2.154 2.115 2.077 2.043

0.323

0.34

3

6

4

5

7

Policy (Rollout w/ Exact DP Base Heuristic; B = 1.4)

12

11

* 10

9

5

4

2

1

10

9

1

N - 1

2.269

Xo

D

1

D

2

D

D

D

3

S

D

D

D

D

4

S

D

D

D

D

D

5

<!-- image -->

nan

S

S

D

D

D

D

6

Figure 1.8.4 Table of values of reward-to-go and decisions applied by the rollout policy that corresponds to the base heuristic with β = 1 glyph[triangleright] 4.

- (d) Repeat part (c) but with two-step instead of one-step lookahead minimiza-

Answer : The implementation is very similar to the one-step lookahead case. The main di ff erence is that at state x k , the rollout algorithm needs to calculate the base heuristic reward values J x k +2 k +2 ( x k +2), J x k +1 k +2 ( x k +1), J x k k +2 ( x k ), J x k -1 k +2 ( x k -1), and J x k -2 k +2 ( x k -2). Thus the on-line Monte Carlo simulation work is accordingly increased. Generally the simulation work per stage of the rollout algorithm is proportional to 2 /lscript + 1, when /lscript -stage lookahead minimization is used, since the number of leafs at the end of the

- tion. lookahead tree is 2 /lscript +1.

## 1.5 (Computational Exercise - Linear Quadratic Problem)

In a more realistic version of the cruise control system of Example 1.3.1, the system has the form

<!-- formula-not-decoded -->

where the coe ffi cient a satisfies 0 &lt; a ≤ 1, and the disturbance w k has zero mean and variance σ 2 . The cost function has the form

<!-- formula-not-decoded -->

where ¯ x 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ¯ x N are given nonpositive target values (a velocity profile) that serve to adjust the vehicle's velocity, in order to maintain a safe distance from

S

S

N = 10

* = 10

p+ = p = 0.25

x0 = 2

the vehicle ahead, etc. In a practical setting, the velocity profile is recalculated by using on-line radar measurements.

Design an experiment to compare the performance of a fixed linear policy π , derived for a fixed nominal velocity profile, and the performance of the algorithm that uses on-line replanning, whereby the optimal policy π ∗ is recalculated each time the velocity profile changes. Compare with the performance of the rollout policy ˜ π that uses π as the base policy and on-line replanning.

## 1.6 (Computational Exercise - Parking Problem)

In reference to Example 1.6.4, a driver aims to park at an inexpensive space on the way to his destination. There are L parking spaces available and a garage at the end. The driver can move in either direction. For example if he is in space i he can either move to i -1 with a cost t -i , or to i +1 with a cost t + i , or he can park at a cost c ( i ) (if the parking space i is free). The only exception is when he arrives at the garage (indicated by index N ) and he has to park there at a cost C . Moreover, after the driver visits a parking space he remembers its free/taken status and has an option to return to any parking space he has already visited. However, the driver must park within a given number of stages N , so that the problem has a finite horizon. The initial probability of space i being free is given, and the driver can only observe the free/taken status of a parking only after he/she visits the space. Moreover, the free/taken status of a parking visited so far does not change over time.

Write a program to calculate the optimal solution using exact dynamic programming over a state space that is as small as possible. Try to experiment with di ff erent problem data, and try to visualize the optimal cost/policy with suitable graphical plots. Comment on the run-time as you increase the number of parking spots L .

## 1.7 (Newton's Method for Solving the Riccati Equation)

The classical form of Newton's method applied to a scalar equation of the form H ( K ) = 0 takes the form

<!-- formula-not-decoded -->

where ∂ H ( K k ) ∂ K is the derivative of H , evaluated at the current iterate K k . This exercise shows algebraically (rather than graphically), within the context of linear quadratic problems, that in approximation in value space with quadratic cost approximation, the cost function of the corresponding one-step lookahead policy is the result of a Newton step for solving the Riccati equation. To this end, we will apply Newton's method to the solution of the Riccati Eq. (1.42), which we write in the form H ( K ) = 0 ↪ where

<!-- formula-not-decoded -->

- (a) Show that the operation that generates K L starting from K is a Newton iteration of the form (1.104). In other words, show that for all K that lead to a stable one-step lookahead policy, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where is the quadratic cost coe ffi cient of the one-step lookahead linear policy θ ( x ) = Lx corresponding to the cost function approximation J ( x ) = Kx 2 :

<!-- formula-not-decoded -->

Proof : Our approach for showing the Newton step formula (1.106) is to express each term in this formula in terms of L , and then show that the formula holds as an identity for all L . To this end, we first note from Eq. (1.108) that K can be expressed in terms of L as

<!-- formula-not-decoded -->

Furthermore, by using Eqs. (1.108) and (1.109), H ( K ) as given in Eq. (1.105), can be expressed in terms of L as follows:

<!-- formula-not-decoded -->

Moreover, by di ff erentiating the function H of Eq. (1.105), we obtain after a straightforward calculation

<!-- formula-not-decoded -->

where the second equation follows from Eq. (1.108). Having expressed all the terms in the Newton step formula (1.106) in terms of L through Eqs. (1.107), (1.109), (1.110), and (1.111), we can write this formula in terms of L only as

<!-- formula-not-decoded -->

or equivalently as

<!-- formula-not-decoded -->

Figure 1.8.5 Illustration of the performance errors of the one-step and two-step lookahead policies as a function of ˜ K ; cf. Exercise 1.8.

<!-- image -->

A straightforward calculation now shows that this equation holds as an identity for all L .

- (b) What happens when K lies outside the region of stability?
- (c) Show that in the case of /lscript -step lookahead, the analog of the quadratic convergence rate estimate has the form

<!-- formula-not-decoded -->

∣ ∣ where F /lscript -1 ( ˜ K ) is the result of the ( /lscript -1)-fold application of the mapping F to ˜ K . Thus a stronger bound for ♣ K ˜ L -K ∗ ♣ is obtained.

## 1.8 (Error Bounds and Region of Stability)

Consider a one-dimensional linear quadratic problem with problem data a = 2, b = 1, q = 1, r = 1.

- (a) Plot the regions of stability for one-step, two-step, and four-step lookahead (cf. Fig. 1.5.9 in Section 1.5).
- (b) Let ˜ θ 1 and ˜ θ 2 be the one-step and two-step lookahead policies for a given quadratic cost approximation coe ffi cient ˜ K . Verify that the performance errors ♣ K ˜ θ 1 -K ∗ ♣ and ♣ K ˜ θ 2 -K ∗ ♣ as a function of ˜ K are as shown in the plot of Fig. 1.8.5. Interpret the figure in terms of your results of part (a). Verify that longer lookahead expands the region of stability, and that the performance error increases sharply as ˜ K approaches the boundary of the region of stability.
- (c) Experiment with other problem data of your choice, including a range of values of a . Verify that for a system that is already stable ( ♣ a ♣ &lt; 1), the region of stability includes the entire nonnegative axis.

## 1.9 (Region of Stability and the Role of Multistep Lookahead)

In Section 1.5, we discussed the concept of the region of stability in the context of linear quadratic problems. The concept extends to far more general infinite horizon problems (see e.g., the book [Ber22a], Section 3.3). The idea is to call a stationary policy θ unstable if J θ ( x ) = ∞ for some states x , and call it stable otherwise. For /lscript ≥ 1, the /lscript -step region of stability is the set of cost function approximations ˜ J for which the corresponding /lscript -step lookahead policy is stable.

Generally, the /lscript -step region of stability expands as /lscript increases. Note also that in finite-state discounted problems all policies are stable, so all ˜ J belong to the region of stability. However, for SSP this is not so: there are policies, called improper , that do not terminate with positive probability for some initial states (this is very common, for example, in nonacyclic deterministic shortest path problems; see the books [Ber12] and [Ber22b] for extensive discussions). Such policies can be unstable. The following example illustrates di ffi cult problems where the region of instability includes functions that are very close to J ∗ , even with large /lscript . This example involves small stage costs, a class of problems that pose challenges for approximation in value space; see Section 2.6.

Consider a deterministic shortest path problem with a single state 1, plus the termination state t . At state 1 we can either stay at that state at cost /epsilon1 &gt; 0 or move to the state t at cost 1. Thus the optimal policy at state 1 is to move to t , the optimal cost J ∗ (1) = 1, and is the unique solution of Bellman's equation

<!-- formula-not-decoded -->

(In shortest path problems the optimal cost at t is 0 by assumption, and Bellman's equation involves only the costs of the states other than t .)

- (a) Show that the one-step region of stability is the set of all ˜ J (1) &gt; 1 -/epsilon1 . What happens in the case where ˜ J (1) = 1 -/epsilon1 ? Show also that the /lscript -step region of stability is the set of all ˜ J (1) &gt; 1 -/lscript/epsilon1 . Note : The /lscript -step region of stability becomes arbitrarily large for su ffi ciently large /lscript . However, the boundary of the /lscript -step region of stability is arbitrarily close to J ∗ (1) for su ffi ciently small /epsilon1 .
- (b) What happens in the case where there are additional states i = 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , and for each of these states i there is the option to stay at i at cost /epsilon1 or to move to i -1 at cost 0? Partial answer : The one-step region of stability consists of all ˜ J = ( ˜ J ( n ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ J (1) ) such that /epsilon1 + ˜ J ( i ) &gt; ˜ J ( i -1) for all i ≥ 2 and /epsilon1 + ˜ J (1) &gt; 1.