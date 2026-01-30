# 1.3: Stochastic Exact and Approximate DP

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 28-41
**Topics:** stochastic DP, finite horizon stochastic, approximation in value space, approximation in policy space, training cost function, policy approximation

---

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