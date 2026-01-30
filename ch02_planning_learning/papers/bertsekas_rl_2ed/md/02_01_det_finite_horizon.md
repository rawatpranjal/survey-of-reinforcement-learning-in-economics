# 2.1-2.2: Deterministic Finite Horizon & Approximation

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 166-176
**Topics:** deterministic finite horizon, approximation in value space

---

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