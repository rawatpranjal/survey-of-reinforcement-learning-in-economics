MATHEMATICS OF OPERATIONS RESEARCH

Vol. 12, No 3, August 1987

Printed in U.S.A.

THE COMPLEXITY OF MARKOV DECISION PROCESSES*†

## We investigate the complexity of the classical problem of optimal policy computation in

## by dynamic programming (finite horizon problems), linear programming, or successive ap-

Macko decision procese A ve age casts of the problem freie horizon a faire ho too deterministic cases of all three problems can be solved very fast in parallel. The version with patially pornon ate than the bee protes act is even less like to be. it is not possible to have an efficient on-line implementation (involving polynomial time on-line computations and memory) of an optimal policy, even if an arbitrary amount of precomputation is allowed. Finally, the variant of the problem in which there are no observations is shown to be NP-complete. 1. Introduction. In the past, Complexity Theory has been applied with considerable success to separate optimization problems with respect to their computational difficulty (GJ, PS]. Such results are valuable in that they succeed in formalizing the

problems include, from the optimization point of view, minimum spanning tree problem, the shortest path problem, and others), whereas certain other problems (first of all linear programming, but also some special cases of it, such as maximum flow) do not appear to be susceptible to such massive parallelization. NC is the class of problems that have algorithms using a polynomial number of processors, and time delay which is polynomial in the logarithm of the input size. The question then arises, is *Received June 10, 1985; revised February 21, 1986. AMS 1980 subject classification. Primary: 90C7. Secondary: 68C25.

- intuitive feeling by researchers in the area about the impossibility of certain approaches in solving some hard problems, and in some cases give a clear separation between the easy and the hard versions (special cases or generalizations) of important problems. In most of this work, NP-completeness is the main notion of complexity used. The important distinction here is between problems that are in P, that is, can be solved in polynomial time (linear programming, max flow, minimum spanning tree) and those that are NP-complete, and thus presumably cannot be so solved (examples are integer programming and the traveling salesman problem). Recent research in parallel algorithms [Co] has succeeded in pointing out some important differences between problems that are in P, that is, can be solved in polynomial time. Some of these problems can be solved by algorithms that use cooperating processors so that the time requirements are reduced tremendously (such

IAOR 1973 subject classification. Main: Programming: Markov Decision.

OR/MS Index 1978 subject classification. Primary: 117 Dynamic programming/Markov. Secondary: 568

Probability/Markov processes.

tion.

Key words. Markov decision processes, dynamic programming, computational complexity, parallel computa-

*This research was supported by an IBM faculty development award, and ONR contract N0001485-C-

*Stanford University.

0731.

*Massachusetts Institute of Technology.

Copyright © 1987, The Institute of Management Sciences/Operations Research Society of America

441 0364765X/87/1202/501.25

442

CHRISTOS H. PAPADIMITRIOU &amp; JOHN N. ISITSIKLIS

NC = P? In other words, is it true that all problems that can be solved satisfactorily by computers? Most researchers believe that the answer is negative.

sequential computers can also take the maximum possible advantage of parallel

The notions of reductions and completeness offer some important evidence here: Linear programming, maximum flow, and some other problems have been shown to be P-complete [DLS, GSS]. This means, intuitively, that such problems are as hard as any problem in P; thus, they can be massively parallelized only if NC = P, that is, if all specialized "policy improvement" routines (Hol); this technique appears to be unpromising from the parallel algorithm point of view, since it involves linear programming, a problem that seems inherently sequential. Secondly, there are iterative techniques that are known to converge fast on the optimal policy; however, such techniques are sequential also by nature. (Let us note in passing that the problem of deriving a "clean" polynomial time algorithm for this problem, without using general linear programming or approximate techniques, is an important open question that has not been emphasized in the literature as much as it deserves.) An interesting question arises: Is the Markov decision problem in NC, or is it complete for P, and thus, in some sense, the use of linear programming for its solution is inherent? In this paper we show that computing the optimal policy of a Markov decision process is complete for P, and therefore most probably is inherently sequential. This is true also for the average cost case, as well as the finite horizon (nonstationary) case. However, we also show that the deterministic special cases of all three problems (that

problems in P can, and thus there are no inherently sequential problems. Since this event is considered unlikely, P-completeness is täken as evidence that a problem is not in NC, and thus cannot be satisfactorily parallelized. The design of optimal strategies for Markov decision processes (say, for the discounted infinite horizon case) has been a very central and well-solved optimization problem. There are basically two approaches to its solution. First, the problem can be expressed as a linear program and thus solved (either by generic techniques, or is, in which all possible stochastic matrices have zero-one entries) are in NC. Next, we address the question of the complexity of the partially observed generalization of this problem; this is an important problem arising in many applications in control theory and operations research [Be]. We show that the partially observed version of the finite horizon problem is PSPACE-complete. We do not consider infinite

horizon partially observed problems because these are not combinatorial problems and do not seem to be exactly solvable by finite algorithms. Notice that only recently has there been work relating combinatorial optimization problems [Or, Pal] to this notion of complexity, which is stronger than NP-competeness (see §2 for definitions). In fact, we show that, most likely, it is not possible to have an efficient on-line implementation (involving polynomial time on-line computations and memory) of an optimal policy, even if an arbitrary amount of precomputation is allowed. Finally, we show that the same problem with no observations is NP-complete. In the next section we introduce the necessary concepts from Markov Decision Processes, and Complexity Theory. In §3 we present our results concerning the possibility of designing massively parallel algorithms for these problems. Finally, in §4 we prove our results on partially observable Markov processes.

2. Definitions.

Complexity. For a general introduction to Computational Complexity see the

Turing machine, random access machine, etc. [AHU, LP]; all these models of computa- bibliography in [Pa2]. Our model of (sequential) computation can be any one of the

tion are known to be equivalent within a polynomial. We denote by P the class of all

COMPLEXITY OF MARKOV DECISION PROCESSES

443

problems that can be solved in polynomial time (in such a model of computation). As question. Optimization problems can be transformed into such problems by asking, not

usual, by "problem" we mean an infinite set of instances, each of which is a "yes-no"

for the optimal feasible solution, but simply whether the optimal cost is below a given bound. NP is the class of problems that can be solved nondeterministically in polynomial time; a nondeterministic machine essentially has the power of branching out and following both computation paths; this can be repeated in each of the paths, and so on, example, logarithmic). An important class is PSPACE, the class of all problems that can be solved in polynomial space. If a computation can be carried out in time (or nondeterministic time) T(n), then it requires only T(n) space; it follows that P and NP are subsets of PSPACE. If the space requirements are T(n), then the time requirement cannot exceed c(n), for some constant c. We say that a problem A is reducible to another B, if for any instance x of A, we can construct using space log(|*) an instance f(x) of B such that the answer for x, considered as an instance of A, is the same as the answer of f(x), considered as an instance of B (here x is a string encoding an instance of problem A, and |x/ is its

creating a potentially exponential set of computations. (Equivalently, NP is the class of "yes-no" problems that have the "succinct certificate property", see (PS].) Besides time complexity, we shall concern ourselves with the space complexity of algorithms. The space consumed by a computation is the number of memory cells necessary for the computation; notice that this does not count the space necessary for the input, so the space requirement of an algorithm may be less than linear (for length). Notice that we insist that the function f be computed in logarithmic space; this is stricter than the polynomial time usually required. In practice this is not a major restriction as most known polynomial reductions can in fact be accomplished in logarithmic space. Also, it can be shown that log-space reduction is transitive. Let A be a problem, and &amp; be a class of problems (such as P, NP, PSPACE). We say that A is complete for 8, or &amp; complete if (a) A is a member of 8, and (b) for every member B of 88, B is reducible to A. Although there are many NP-complete problems in the literature, the notion of PSPACE-completeness is somewhat less well-known. A fundamental PSPACE-complete problem is that of quantified satisfiability (QSAT) [SM]: We are given a quantified Boolean formula 3x, Vx, 3x3...Vx, F(x],..., X,), where F is an ordinary

polynomial hierarchy (SM]. The next level above NP in this hierarchy is the class &amp;4, which can be thought of as all problems that reduce to the problem of telling whether a formula with one string of existential quantifiers, followed by a string of universal quantifiers, is true. An example of such a formula is 3x,7x2. Ix,VyVY... In the interest of formalizing parallel computation, we can define a parallel random access machine (PRAM), that is, a set of RAMs operating in unison, and communicat-

Boolean formula in conjunctive normal form with three literals per clause. We are to determine whether this formula is true, that is, whether there exists a truth value for x, such that, for all truth values of x2, etc. for all truth values of x,, F comes out true. As mentioned above, PSPACE contains NP. In fact, NP can be thought of as the class of problems reducible to the special case of QSAT, called SAT, in which all quantifiers are existential (that is, there is no alternation of quantifiers). In fact, there is a number of intermediate classes' based on less restricted alternation; this is called the

"Of course, it should be borne in mind that, for all we know, PSPACE, NP, even P might all coincide, in which case so would these "intermediate" classes.

444

CHRISTOS H. PAPADIMITRIOU &amp; JOHN N. TSITSIKLIS

ing through a sequence of common registers, some of which initially hold the input

[Co]. The measures of complexity here are the delay until the final answer is produced, solved in a satisfactory way by such a supercomputer if the delay is polynomial in the logarithm of the length of the input, and the number of processors involved are polynomial in the length of the input. The class of problems solvable under these terms is called NC. Obviously (by multiplying delay times number of processors) NC is a subset of P. The great enigma for parallel computation, analogous to P = ?NP for sequential computation, is whether NC = P. That is, while P = NP asks whether there are (reasonable) problems that are inherently exponential, NC = P asks whether there are problems that are inherently sequential. The bets are that such problems do exist, but there is no proof. Now, if inherently sequential problems exist (equivalently, if NC # P) then the problems that are complete for P will certainly be among them. Thus, P-completeness plays vis-a-vis the NC = P problem the same role enjoyed by NP-completeness in relation to the P = NP problem. One basic P-complete problem is the circuit value problem (CVP). A circuit is a finite sequence of triples C = (a,, bi, ci), i = 1,..., k). For each i &lt;k, a, is one of the "operations" false, true, and, and or, and b,, c, are nonnegative integers smaller than i. If a, is either false or true then the triple is called an input, and b, = c, = 0. If a, is

i EDs, at time t, we incur a cost c(s, i,t), and the next state s' has probability distribution given by p(s, s', i, t). We say the process is stationary if c and p are independent of 1, in which case we write p as p(s, s', i). A policy &amp; is a mapping assigning to each / € (0,1,2,...) and state s E S a decision 8(s, t). A policy is stationary if 8(s, f) is independent of 1. We shall consider three variants of the problem. First, in the finite horizon problem, we are given a (nonstationary) Markov decision process and an integer T. We wish to find a policy that minimizes the expectation of the cost I,., c(s,, 8(s,, t), t). The other variants that we deal with have infinite horizon, and thus we shall only consider their stationary special cases (otherwise we need the description of an infinite object, namely the parameters c and p for all times). In the discounted (infinite horizon) problem we are given a Markov decision process and a number B € (0,1), and we wish to minimize the expectation of the discounted cost Lo C(s,, 8(5,, *))B'. Finally, in the average cost (infinite horizon) problem we are given a Markov decision process, and we wish to minimize the expectation of the limit lim,- {'-0 c(s, 8(5,,+))/T). It is well known that in the two latter cases (those for which the optimal policy is a potentially infinite object), there exist stationary optimal policies, and therefore optimal policies admit a finite description. For all three cases of the problem, there are well-known polynomial-time algorithms. 3. P-completeness and parallel algorithms. We show the following result: THEOREM 1. The Markou Decision Process problem is P-complete in all three cases (finite horizon, discounted, and average cost).

either and or or then the triple is called a gate and b, c, &gt; 0. The value of a triple is defined recursively as follows: First, the value of an input (true; 0, 0) is true, and that of (false, 0, 0) is false. The value of a gate (aj, bi, (;) is the Boolean operation denoted by a, applied to the values of the b,th and c,th triples. The value of the circuit C is the value of the last gate. Finally, the CVP is the following problem: Given a circuit C, is its value true? Markou Decision Processes. In a Markov decision process we are given a finite set S of states, of which one, so is the initial state. At each time t = 0, 1,... the current state is s, E S. For each state s € S we have a finite set D, of decisions. By making decision

COMPLEXITY OF MARKOV DECISION PROCESSES

445

PROOF. We shall reduce the CVP to the finite horizon version first. Given a circuit

C= ((aj, b,, c;), i = 1,..., k), we construct a stationary Markov process M = (S, c, p)

(aj, bi, C;) is an input, then the corresponding state i has a single decision 0, with p(i, q,0) = 1, and cost c(i, 0) = 1 if a; = false, and 0 otherwise. All other costs in this process are 0. From q we only have one decision, which has p(9, 9,0) = 1. If a, is an or gate, there are two decisions 0, 1 from state i, with zero cost, and p(i, b,, 0) = 1, p(i, c;, 1) = 1. That is, we decide whether the next state is going to be b, or c,. If a, is an and gate, then there is one choice 0 in Di, with p(i, bi, 0) = p(i, c,,0) = 1/2, that is, the next state can be either b, or c,. The initial state is k, the last gate, and the time horizon also equals k, the size of the circuit. We claim that the optimum expected cost is 0 or less iff the value of C was true. In proof, suppose that the expectation is indeed zero (it cannot be less). Then it follows that there are decisions so that the states with positive costs are impossible to reach. However, these states correspond to the false inputs, and thus these decisions are as follows: S has one state i for each triple (ai, bi, C,) of C, plus an extra state q. If

choices of a true gate among bi, c, for each or gate i of the circuit so that its overall value is true. Conversely, if the value is true, there must be a way to choose an input gate for each or gate so that the false inputs are not reachable, or, equivalently, the states with positive costs are not possible. Essentially the same construction works for the discounted case; for the average cost problem, a modification is necessary: We must first make sure that all paths from the inputs to the last gate of C have the same number of gates on them, and then have transitions from the states corresponding to the inputs not to a state q, but back to the initial state.

We note that the above proof shows that even the stationary special case of the finite horizon problem is P-hard. It is not known whether the stationary finite-horizon problem is in P, because of the following difficulty: We could be given a stationary process with n states, and a horizon T up to 2", and the input would still be of size tionary), discounted, and average cost problem are in NC. Our approach is to look at these problems as variants of the graph-theoretic shortest-path problem [La]. This is done as follows: Given a deterministic Markov decision process, we construct a directed graph G = (V, A), where the nodes are the states, and the arcs emanating from a node are the decisions available at this state. The node pointed to by an arc is the state to which the decision is certain to lead (recall that there are only 0 or 1 probabilities). There is a weight c(u,v) on each arc (u, v), equal to the cost of the corresponding decision. We denote by u, the node corresponding to the initial state So. The particular variants of the problem are then equivalent to certain variants of the shortest path problem. The nonstationary finite horizon problem. The parallel algorithms that we describe in this section employ a technique used in the past to yield fast parallel (or space efficient) algorithms known as "path doubling" [Sa, SV]. The idea is, once we have computed all

0(n). Still, the dynamic programming algorithm for this problem would take time proportional to nt, and thus exponential in the input. (In the nonstationary case, the input must specify c and p for each i &lt; T, and so the input size is at least T.) Deterministic problems. Consider now the special case of these problems, in which the only allowed values for p are 0 and 1; we call this the deterministic case. We shall show below that the deterministic cases of the finite horizon (stationary and nonsta- optimal policies between any two states, where each policy starts at time 4 and ends at time 12, and similarly between 12 and 4, to compute in one step all optimal policies between fy and 13. We can think of this as multiplying two |V| X |V| matrices A(41, 12)

446

CHRISTOS H. PAPADIMITRIOO &amp; JOHN N. TSITSIALIS

and A(42, *3), where the (u, o)th entry of (4,½2) is the cost of the optimal policy in which multiplication of reals is replaced by addition, and addition of reals by the

from state u to o between times &amp; and i. This "matrix multiplication" is the variant operation of taking the minimum. Using n' processors, we can "multiply" n X n matrices in log n parallel steps, by independently computing each of the n? inner products using n processors in log n steps. This latter can be done by computing the n terms of the inner product independently, and combining them in log n stages. This approach immediately suggests an NC parallel algorithm for the finite horizon nonstationary problem: We start from the matrices A(t, 7 + 1), 1 = 0,... T - 1; the (4, 0)th entry of A(t, t + 1) equals the cost of the decision leading at time t from state u to state u, if such a decision exists at time i, and is equal to o otherwise. Then to

technique, explained later.

compute the required A(0, T) we multiply in log T stages these matrices. The total number of processors is Tn', and the total parallel time log T log n; since the size of the input for the nonstationary problem is n'T, this is an NC algorithm. THEOREM 2. The finite-horizon, nonstationary deterministic problem is in NC. · Notice that this technique does not solve the stationary problem, whose input is of length.n? + logT, and which will have to be attacked by a more sophisticated

The infinite horizon undiscounted case. It is easy to see that the infinite horizon average cost problem is equivalent to finding the cycle in the graph corresponding to the process that is reachable from lo, and has the smallest average length of arcs. In proof of the equivalence, the limit of the average cost of an infinite path which starts at to, reaches this cycle, and then follows this cycle for ever, equals the optimum average cost, and this cannot be improved. To make sure that we do not consider solutions that are not reachable from uo, we first compute in log n parallel time [SV] the nodes reachable from ko, and in the sequel consider only these. The cycle with the shortest average cost can be found by

computing, in parallel, for each k = 1,..., n the shortest cycle of length k, and companying the results, each divided by k. To compute the shortest cycle of length k, we essentially have to compute the kth power of matrix A, whose (u, v)th entry is equal to the cost of the decision leading from state u to state u, if such a decision exists, and o otherwise. This is done with n'k processors in log k log n parallel steps. The total time is log'n + 2 log n, using n* processors. THEOREM 3. The infinite-horizon, average cost deterministic problem is in NC. The infinite horizon, discounted case. Define a sigma in a directed graph to be a discounted cost of a sigma P = (40,..., U4, V1,..., U,, U,) is defined to be с (Р) = U;+1) B' Bk+1 (1 - 8') c(Vi, Di+ 1 mod 1) B. j=1

path of the form (40,..., U4, 01,..., 01, U1), where all nodes indicated are distinct. In other words, a sigma is a path from u, until the first repetition of a node. The

<!-- formula-not-decoded -->

optimal policy is the optimum discounted cost of a sigma in the corresponding directed graph. We can compute the optimum sigma as follows; First, we compute the shortest discounted path of length j = 0, 1….., among any pair of nodes by "multiplying" the

COMPLEXITY OF MARKOV DECISION PROCESSES

447

matrices A, BA,..., Bi-'A, where A is the matrix defined in the paragraph before the resulting products; the (u, o)th entry of B, is the length of the shortest path with j

Theorem 3. This can be done in log'n steps by using n* processors. Let B,..., B, be arcs from u to v. Once this is done, for each node u and for each k, 1 = 0,1,..., n we con of We sick , her The he (ure a car of po the and processors, and login + 3 log n parallel steps. THEOREM 4. The infinite-horizon, discounted deterministic problem is in NC. = The finite horizon, stationary case. Given a stationary deterministic process, the

should run in a number of parallel steps that is polynomial to log log T; therefore the "path-doubling" technique would not give the desired result. In the sequel we assume that T &gt; n?, otherwise the previous technique applies. Without loss of generality, assume that the arc lengths are such that no ties in the lengths of paths are possible (this can be achieved by perturbing the lengths). Consider the shortest path out of u, with T arcs. Since T &gt; n?, there are many repetitions of stationary finite horizon problem with horizon T is equivalent to finding the shortest path with T arcs in the corresponding graph, starting from the node up. The algorithm

nodes on this path. Consider the first such repetition, that is, the first time the path forms a sigma, and remove the cycle from the path. Then consider the first repetition in the resulting sequence. Continuing this way, we can decompose the path into a simple path (i.e., without repetitions of nodes) out of lo, plus several simple cycles (notice that this decomposition may not be unique). We first need to show that we can assume that only one simple cycle is repeated more than n times, namely the one that has the shortest average length of its arcs. In proof, consider two cycles of length m, and m2, repeated n, and n, times, respectively. If n,, n,≥ n, then we can repeat the cycle with smaller average arc length, say the first, m, times less, and repeat the second m, times more, to obtain another path of smaller length. Thus, only one cycle is repeated more than n times. Furthermore, since we have no ties, only one cycle of each length is repeated. Therefore, the shortest path of length T has the following structure: It consists of a path with length / &lt; n' (this is the simple path, plus the at most n repetitions of one cycle from each length), plus a simple cycle repeated many times to fill the required number of arcs. Therefore, for each value of I &lt; n' and each node u and each possible number of arcs in the cycle k that divides T - 1 we do, in parallel, the following: We compute the shortest path of length / from u, through u, the shortest cycle of length k through u, and the cost of adding (T - 1)/k copies of the cycle to the path. Of the resulting combinations, we pick the cheapest. THEOREM S. The finite-horizon, stationary deterministic problem is in NC. 4. The partially observed problem. We can define an important generalization of

I = (21 .., 2p) of S, where the 2,'s are disjoint subsets of S that exhaust S, and at any time we only know the particular set in II to which the current state belongs (naturally, we also remember the previous such sets).? Each set z € II has a set D, of decisions associated with it, and each decision i ED, has a cost c(z, i) and a probability distribution p(s,s', i) of the next state s', given the current state s € z. *In the traditional formulation of the partially observed problem (Be], the observation at time ‹ is a random variable, for which we know its conditional distribution given the current state and time. Here we essentially deal with the special case in which the observation is a deterministic function of the current state.

Markov decision processes if we assume that our policy must arrive at decisions with only partial information on the current state. In other words, we have a partition

Clearly, the general case cannot be any easier.

448

CHRISTOS H. PAPADIMITRIOU &amp; JOHN N. TSITSIKLIS

The initial state is known as before to be So. A policy is now a mapping from sequences of observations 2,,..., z, to decisions in De;

problem by redefining the state to be the vector p whose ith component is the conditional probability that the state of the original problem equals i [Be]. In this formulation the state space is infinite. Nevertheless, it is known [SS] that the function assigning to each probability vector and time t the optimum future cost (cost-to-go) is of the form Jp, 1) = min, ek, [Ax + B«P], where A, is a scalar and B, is a vector of suitable dimension and K, is a finite index set. This allows the solution in finite time of the problem even though the state space is infinite. Nevertheless, the cardinality of the index set K, may be an exponential function of the time horizon and, therefore, this procedure is of limited practical use. The following result shows that most likely this is a generic property of the problem we are dealing with rather than a defect of the particular reformulation of the problem. Notice also that in the infinite horizon limit the index set K, will be infinite, in general, which makes fairly doubtful whether there exists a finite-algorithm for the infinite horizon problem. For this reason, we only consider the problem in which we are asked to minimize the expected undiscounted cost over a finite time horizon T. THEOREM 6. . The partially observed problem is PSPACE-hard, even if the process is stationary and T is restricted to be smaller than IS| (and it is in PSPACE in the latter

case).

PROOF. To show PSPACE-hardness, we shall reduce QSAT to this problem. Starting from any quantified formula 3x,Vx2...Vx, F(x,,..., x,), with n variables (existential and universal) and m clauses ,,..., Cm, we construct a partially observed form a set called A,, the states Tij, i = 1,..., m form a set called Ij, the states Tij, i = 1,..., m form a set called T' and so on, up to Ff. The states Ai,n+1 and Ai, n+1 are each on a different set. Next we shall describe the decisions, the probabilities, and the costs. All decisions except those out of the set An+ have zero cost. At so there is only one decision, leading to the states A/, i = 1,..., m with equal probability. If x, is an existential

stationary Markov process and a horizon T &lt; IS/, such that its optimum policy has cost 0 or less if and only if the formula is true. We first describe S. Besides so, the initial state, S contains six states Ajjo Aig Tig, Tij, Fujo Faj for each clause C,, and variable x.. There are also 2m states Ai, n+1, Ai,n+1. We next describe the partition II. The initial state so is in a set by itself. For each variable j, the states Aj, i = 1,..., m and the states Ali,, i = 1,..., m variable, there are two decisions out of the set A,, leading certainly from A,, to I,, and Fi, respectively; similarly there are two decisions out of the set A', leading with certainty from A', to Ty and Fij, respectively. If x; is a universal variable, there is one decision out of the set A, leading with equal probability from A,, to 1, and Fij similarly there is one decision out of the set A', leading with equal probability from A', to Tf and Ff. From the T,, F, Ty and F' sets there is only one decision, which leads with certainty from Tij, Fij, Ty, and Fl to (respectively) Ai, j+1, Aji+1, Al,j+1» to Aij+1· Finally, out of Ai,n+ 1 and Ai,n+1 there is one decision, leading to a new state with certainty. In the entire process, all decisions have cost zero, except for the decision out of Ai, n+ 1, which incurs a cost of 1. This completes the construction of the process; the horizon T' is defined to be 2m + 2 (just enough time for the process to reach one of

Ai, n+ 1 OF 14, n+1).

COMPLEXITY OF MARKOV DECISION PROCESSES

449

We claim that there exists a policy with expected cost zero iff the formula is true.

any state A' of A4; we think that the process "chooses" a clause C,. Once this has

Suppose such a policy exists. Recall that the transition from the initial state can be to happened, the process remains at states subscripted by i forever, without ever observing the real value of i. It follows that the policy has zero expected cost for all such initial choices. We next claim that the policy must guarantee that the process ends up in Ai, n+ 1- If not, that is, if for some choices of decisions for the universal variables the process ends up in Ain+1, then this contributes an expected cost of at least 2-"/m, which is absurd. It follows that the policy must, based on the previous observations on the transitions at the sets corresponding to previous universal variables, pick at the existential variables those decisions that correspond to a truth assignment which satisfies the clause. Since all clauses must be satisfied, the formula was true. Conversely, if the formula were true, there is a policy for setting the existential variables, based on the values of previous universal ones, so that all clauses are satisfied. This, however, can be translated into a policy for choosing the corresponding decisions at the sets corresponding to existential variables so that, for all choices of

possible outcomes of the process within the finite horizon T as a tree of depth T, which has internal nodes both for decisions and transitions. The leaves of this tree can be evaluated to determine the total contribution to the expected cost of this path. To determine whether the optimal policy has expected cost less than some number B, we need to traverse this tree, making nondeterministic decision nodes, iterating over all transitions and taking care to make the same decision every time the history of observations is identical. This latter can be achieved by organizing the search in such a way that nodes at the same level of the tree are divided into intervals from left to right, corresponding to distinct observation sequences: Decisions from nodes in the same interval must be the same. That the problem is in PSPACE follows now from the fact that PSPACE is robust under nondeterminism. · Theorem 6 says that, unless P = PSPACE (an event even less likely than P = NP), there is no polynomial-time algorithm for telling whether a cost can be achieved in a give parial obich a he protes by this ere there a mose racial probis i, the clauses, the state A,, n+1 is reached. A policy with zero cost is thus possible. The proof is complete. Finally, we sketch a proof that the special case of the problem examined here is in PSPACE. The construction is a bit tedious, but quite reminiscent of polynomial-space algorithms for other similar problems (see for example [Pall). We can think of the

could analyze such a process (perhaps by preprocessing it for an exponential time or worse), and, based on the analysis, design a practically realizable controller (polynomial-time algorithm) for driving the process on-line optimally. This algorithm would presumably consult come polynomially large data structure, constructed during the analysis phase. We point out below that even this is very unlikely: CoROLLARy 1. Unless PSPACE = E{, it is impossible to have an algorithm Al and a mapping i (of arbitrary high complexity) from a partially observed Markou process M to a string M(M) of length polynomial in the description of the process, such that the quantified Boolean formula is true) as follows: "There exists a string M(M) such that,

algorithm, with the string and the observations as input, computes the optimum decision for the process in polynomial time. Sketch. If such an e and y existed, then we could express the question of whether M has zero expected cost (and thus, by the previous proof, whether an arbitrary

450

CHRISTOS H. PAPADIMITRIOU &amp; JOHN N. TSITSIKLIS

for all possible transitions, the decisions of algorithm A lead to a zero cost." This last quantifiers. =

sentence, however, can be rewritten as a Boolean formula with two alternations of

Finally, considering the special case of the partially observed problem, in which the partition II = (S} (i.e., the case of unobserved processes or, equivalently, an open-loop control) we note that a simplification of the proof of Theorem 6 establishes that this problem is NP-complete:

COROLLARY 2. Deciding whether the optimal policy in an unobserved Markov decision process has expected cost (undiscounted, over a finite horizon) equal to zero is an NP-complete problem. ·

Acknowledgement. We wish to thank Professor M. Katehakis for a number of helpful discussions and insightful comments, and an anonymous referee for careful reading of the manuscript and a number of helpful suggestions. This research was supported by an IBM faculty development award, and ONR contract N0001485-C0731. References [AHU) Aho, A. V., Hopcrost, J. E. and Ullman, J. D. (1974). The Design and Analysis of Computer Algorithms. Addison-Wesley, Reading, MA.

(Be]

[Co]

Bertsekas, D. P. (1976). Dynamic Programming and Stochastic Control. Academic Press, New York.

Cook, S. A. (1981). Towards a Complexity Theory of Synchronous Parallel Computation. Enseign.

[DLR]

(GSS]

(GJ)

(Ho]

[La]

(LP)

(Or]

[Pa2]

[Pal]

(PS]

[Sa]

[SS]

(SM]

[SV)

- Math. 2, 27 99-124. Dobkin, D., Lipton, R. J. and Reiss, S., (1979). Linear Programming is Log-Space Hard for P.
- Goldschlager, L. M., Shaw, R. A. and Staples, J. (1982). The Maximum Flow Problem is Log Space Complete for P. Theoret. Comput. Sci. 21 105-111.
- Informat. Process Lett. 8 96-97.
- Garey, M. R. and Johnson, D. S. (1979), Computers and Intractability: A Guide to the Theory of NP-Completeness. Freeman, San Francisco.
- Howard, R. A. (1960). Dynamic Programming and Marko Processes. MIT Press, Cambridge. Lawler, E. L. (1977). Combinatorial Optimization: Networks and Matroids. Holt, Rinehart and
- Englewood Cliffs, NJ.
- Winston, New York. Lewis, H. R. and Papadimitriou, C. H. (1981). Elements of the Theory of Computation. Prentice-Hall,
- Orlin, J. (1981). The Complexity of Dynamic Languages and Dynamic Optimization Problems. Proc. 13th STOC, 218-227.
- (1985). Games Against Nature. J. Comput. Systems Sci. 31, 2 288-301. and Steiglitz, K. (1982) Combinatorial Optimization: Algorithms and Complexity. Prentice-
- Papadimitriou, C. H. (1985). Computational Complexity. In Combinatorial Optimization: Annotated Bibliographies. M. O'hEigeartaigh, J. K. Lenstra, A. H. G. Rinooy Kan (eds.), 39-51.
- Hall, Englewood Cliffs, NJ. Savitch, W. J. (1970). Relationships Between Nondeterministic and Deterministic Tape Complexi-
- Smallwood, R. D. and Sondik, E. J. (1973). The Optimal Control of Partially Observable Markov Processes over a Finite Horizon. Oper. Res. 11 1971-1088.
- ties. J. Compur. Systems Sci. 4 177-192.
- Stockmeyer, L. J. and Meyer, A. R. (1973). Word Problems Requiring Exponential Space. Proc. 5th
- Shiloach, Y. and Vishkin, U. (1982). An O (log n) Parallel Connectivity Algorithm. J. Algorithms 3

STOC.

- PAPADIMITRIOU: DEPARTMENTS OF COMPUTER SCIENCE AND OPERATIONS RE-
- SEARCH, STANFORD UNIVERSITY, STANFORD, CALIFORNIA 94305

TSITSIKLIS: DEPARTMENT OF ELECTRICAL ENGINEERING, MASSACHUSETTS INSTITUTE

OF TECHNOLOGY, CAMBRIDGE, MASSACHUSETTS 02139