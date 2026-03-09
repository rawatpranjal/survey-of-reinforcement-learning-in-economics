<!-- image -->

## Dynamic Programming and Optimal Control Volume II

Dimitri P. Bertsekas

Massachusetts Institute of Technology

Athena Scientitic Post Office Box 391 Behnout, Mass. 02178-0998 U.S.A.

Email: athenase@workd.std.com

Cover Design: Arn Gallager

<!-- image -->

## Contents

| "ror                                                                                                                                                                                                                       | 1. Infinite Horizon - Discounted Problems                                                                                                                                                                                                                |                                |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| © 1995 Dimitri P. Bertsckas                                                                                                                                                                                                | 1.3. Finite-State Systems Computational Methods 1.3.1. Value Iteration and Error Bounds 1.3.2. Policy Iteration . . .                                                                                                                                    | . 1. 16. • p. 19.              |
| All rights reserved. No part of this book may be reproduced in any form                                                                                                                                                    |                                                                                                                                                                                                                                                          | . 1). 35                       |
| by any electronic or mechanical means (including photocopying, recording, or information storage and retrieval) without permission in writing from                                                                         | 1.3.3. Adaptive Aggregation .                                                                                                                                                                                                                            | 1. 11                          |
|                                                                                                                                                                                                                            | 1.3.1. Linear Programming                                                                                                                                                                                                                                | 1). 19                         |
| the publisher.                                                                                                                                                                                                             | / 1.1. The Rolo of Contraction Mappings                                                                                                                                                                                                                  | 1. 52                          |
|                                                                                                                                                                                                                            | 1.5. Scheduling and Multiarmed Bandit. Problems                                                                                                                                                                                                          | 1. 5.1                         |
|                                                                                                                                                                                                                            | 1.6. Notes, Sources, and Exercises                                                                                                                                                                                                                       |                                |
| Portions of this volume are adapted and reprinted from the author's Dy-                                                                                                                                                    | 2. Stochastic Shortest Path Problems                                                                                                                                                                                                                     | 1. 6.1                         |
| munic Programming: Deterministic and Stochastic Models. Prentice-Hall. 1987, by permission of Prontice-Hall, Ine. Publisher's Cataloging-in-Publication Data Bertsekas, Dimitri P. Dynamic Programming and Optimal Control | • 2.1. Main Results 2.2. Computational Methods 2.2.1. Value Iteration . 2.2.2. Policy Iteration 2.3. Simulation-Based Methods 2.3.1. Policy Evaluation by Monte-Carlo Simulation 2.3.2. Q-Learning. 3.3. Inventory Control . . 3.4. Optimal Stopping . . | p. 78 p. 87 D. 88 p. 91 1. 9-1 |
| QA 402.5.3165 1995 519.703 95-075941                                                                                                                                                                                       |                                                                                                                                                                                                                                                          | 1). 99 1. 101                  |
| 1. Mathomatical Optimization. 2. Dynamic Programming. I. Title.                                                                                                                                                            | 2.3.3. Approximations 2.3.4. Extensions to Discounted Problems 2.1. Notes, Souces, and Exercises 3. Undiscounted Problems                                                                                                                                | 1. 1.18 1). 120 p. 121         |
| Includes Bibliography and Index                                                                                                                                                                                            | 3.1. Unbounded Costs per Stage 3.2. Linear Systems and Quadratic Cost                                                                                                                                                                                    | D. 95                          |
| ISBN 1-886529-12-1 (Vol. I)                                                                                                                                                                                                |                                                                                                                                                                                                                                                          |                                |
|                                                                                                                                                                                                                            | 2.3.5. The Role of Parallel Computation                                                                                                                                                                                                                  |                                |
| ISBN 1-886529-13-2 (Vol. 11)                                                                                                                                                                                               |                                                                                                                                                                                                                                                          | D. 131                         |
| ISBN 1-886529-11-6 (Vol. 1 and 11)                                                                                                                                                                                         |                                                                                                                                                                                                                                                          | р. 150                         |
|                                                                                                                                                                                                                            |                                                                                                                                                                                                                                                          | р. 153                         |
|                                                                                                                                                                                                                            |                                                                                                                                                                                                                                                          | 0. 155                         |
|                                                                                                                                                                                                                            | 3.5. Optimal Gambling Strategies                                                                                                                                                                                                                         |                                |
|                                                                                                                                                                                                                            |                                                                                                                                                                                                                                                          | D. 160                         |

|                                                                                                                                                                                                                                                                                                                                                  | C'ontents   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| 3.6. Nonstationary and Periodic Problems 3.7. Notes, Sources, and Exercises 4. Average Cost per Stage Problems p. 167 1. 172                                                                                                                                                                                                                     |             |
| 4.1. Preliminary Analysis 4.2. Optimality Conditions 1.3. Computational Methods 1.3.1. Value Iteration 4.3.2. Policy Iteration 4.3.3. Linear Programmig 1.3.4. Simulation-Based Methods 4.1. Infinite State Space 4.5. Notes, Sources, and Exercises 5. Continuous-Time Problems p. 18•1 1. 191 0. 202 p. 202 1. 213 p. 221 p. 222 0. 226 p. 229 |             |
| 5.1. Uniformization 5.2. Quencing Applications                                                                                                                                                                                                                                                                                                   | 1. 2:12     |
|                                                                                                                                                                                                                                                                                                                                                  | 1. 250      |
|                                                                                                                                                                                                                                                                                                                                                  | p. 261      |
| 5.3. Somi-Markov Problems                                                                                                                                                                                                                                                                                                                        |             |
| 5.1. Notes, Sourcos, and Exercises . . ........                                                                                                                                                                                                                                                                                                  |             |
|                                                                                                                                                                                                                                                                                                                                                  | 1. 273      |

| CONTENTS OF VOLUME I                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | CONTENTS OF VOLUME I                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. The Dynamic Programming Algorithm 1.I. Introduction 1.2. The Basic Problem 1.3. The Dynamic Programming Algorithm 1.4. Stado Angmentation 1.5. Some Mathematical Issues 1.6. Notes. Sources, and Exercises 2. Deterministic Systems and the Shortest Path Problem 2.1. Finite-State Systems and Shortest Paths 2.2. Some Shortest Path Applications 2.2.1. Critical Path Analysis 2.2.2. Hidden Markov Models and the Viterbi Algorithm 2.3. Shortest Path Algorithms 2.3.1. Label Correcting Methods 2.3.2. Auction Algorithus | 1. The Dynamic Programming Algorithm 1.I. Introduction 1.2. The Basic Problem 1.3. The Dynamic Programming Algorithm 1.4. Stado Angmentation 1.5. Some Mathematical Issues 1.6. Notes. Sources, and Exercises 2. Deterministic Systems and the Shortest Path Problem 2.1. Finite-State Systems and Shortest Paths 2.2. Some Shortest Path Applications 2.2.1. Critical Path Analysis 2.2.2. Hidden Markov Models and the Viterbi Algorithm 2.3. Shortest Path Algorithms 2.3.1. Label Correcting Methods 2.3.2. Auction Algorithus |
| 3. Deterministic Continuous-Time Optimal Control 3.1. Continous-Time Optimal Control 3.2. The Hamilton Jacobi Bellman Equation                                                                                                                                                                                                                                                                                                                                                                                                     | 3. Deterministic Continuous-Time Optimal Control 3.1. Continous-Time Optimal Control 3.2. The Hamilton Jacobi Bellman Equation                                                                                                                                                                                                                                                                                                                                                                                                     |
| 1.1. Linear Systems and Quadratic Cost. 1.2. Inventory Control 1.3. Dynamic Portfolio Analysis                                                                                                                                                                                                                                                                                                                                                                                                                                     | 1.1. Linear Systems and Quadratic Cost. 1.2. Inventory Control 1.3. Dynamic Portfolio Analysis                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 4. Problems with Perfect State Information                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 4. Problems with Perfect State Information                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 3.5. Notes, Sources, and Excreises                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 3.5. Notes, Sources, and Excreises                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2.1. Notes. Sources, and loxercises                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 2.1. Notes. Sources, and loxercises                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 3.3. The Pontryagin Minimum Principle                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 3.3. The Pontryagin Minimum Principle                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 3.3.1. An Informal Derivation Using the HIB Equation 3.3.2. A Dorivation Based on Variational Ideas                                                                                                                                                                                                                                                                                                                                                                                                                                | 3.3.1. An Informal Derivation Using the HIB Equation 3.3.2. A Dorivation Based on Variational Ideas                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 3.3.3. The Minimum Principle for Discrete- Time Problems                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 3.3.3. The Minimum Principle for Discrete- Time Problems                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 3.1. Extensions of the Miniman Principle 3.1.1. Fixed Terminal State                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 3.1. Extensions of the Miniman Principle 3.1.1. Fixed Terminal State                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 3.1.2. Free Initial Stato                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 3.1.2. Free Initial Stato                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 3.1.3. Free Terminal Time                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 3.1.3. Free Terminal Time                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 3.4.1. Time-Varying System and Cost.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 3.4.1. Time-Varying System and Cost.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 3.1.5. Singular Problems                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 3.1.5. Singular Problems                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 1.5. Scheduling and the Interchange Argument.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 1.5. Scheduling and the Interchange Argument.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 1.6. Notes, Sources. and Exercises                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 1.6. Notes, Sources. and Exercises                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 1.4. Optimal Stopping Problems                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 1.4. Optimal Stopping Problems                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

- .1

## 5. Problems with Imperfect State Information

- 5.1. Reduction to the Perfect Information Case
- 5.2. Lincar Systems and Quadratic Cost
- 5.3. Minimum Variance Control of Linear Systems
- 5.4. Sufficient, Statistics and Finite-State Markov Chains
- 5.5. Sequontial Hypothesis Testing
- 5.6. Notes, Sources, and Exercises

## 6. Suboptimal and Adaptive Control

- 6.1. Certainty Equivalont and Adaptive Coutrol
- 6.1.1. Caution. Probing, and Dual Control
- 6.1.4. Solf-Tuning Regulators
- 613 CoPy Cite stin entailly
- 6.2. Open-Loop Feedback Control
- 6.3. Limited Lookahead Policies and Applications
- 6.3.1. Plexible Manofacturing
- 6.3.2. Computer Choss
- 6.1. Approximations in Dynamic Programming
- 6.1.1. Discretization of Optimal Control Problous
- 6.4.2. Cost-to-Go Approximation
- 6.1.3. Other Approximations
- 6.5. Notes, Sources, and Exercises

## 7. Introduction to Infinite Horizon Problems

- 7.1. An Overviow
- 7.2. Stochastic Shortest Path Problems
- 7.3. Discounted Problems
- 7.4. Averago Cost Problems
- 7.5. Notes, Sources, and Exercises

Appendix A: Mathematical Review

Appendix 3: On Optimization Theory

Appendix C: On Probability Theory

Appendix D: On Finite-State Markov Chains

Appendix E: Least-Squares Estimation and Kalman Filtering

Appendix F: Modeling of Stochastic Linear Systems

## ABOUT THE AUTHOR

Dimitri Bertsckas studied Mechanical and Electrical Engineering at. the National Technical University of Athens, Greece, and obtained his Ph.D. in system science from the Massachusetts Institute of Technologs: Ho has held faculty positions with the Engineering-Economic Systems Dept.., Stanford University and the Electrical Engineering Dept. of the University of Illinois, Urbana. He is currently Professor of Electrical Engineering and Computer Science at the Massachusetts Institute of Technology: He consults regularly with private industry and has hold editorial positions in several journals. He has been elected Fellow of the IDE.

Professor Bertsckas has done rescarch in a broad variety of subjects from control theory, optimization theory, parallel and distributed computation, data communication networks, and systems analysis. He has written numerous papers in each of these areas. This book is his fourth on dynamic programming and optimal control.

## Other books by the author:

- 1) Dynamic Programming and Stochastic Control, Academic Press, 1976.
- 3) Constrained Optimization and Lagrange Multiplier Methods, Academic Press, 1982 (translated in Russian).
- 2) Stochustic Optimal Control: The Discrete-Time Case, Academic Press. 1978 (with S. E. Shrove; translated in Russian).
- 4) Dynamic Programming: Deterministic and Stochastic Models. Prentice-Hall, 1987.
- 5) Data Networks. Prentico-Hall, 1987 (with R. G. Gallager; translated in Russian and Japanese): 2nd Edition 1992.
- 6) Parallel and Distributed Computation: Numerical Methods, PrenticeHall, 1089 (with J. N. Tsitsiklis).
- 7) Linear Network Optimization: Algorithms and Codes, M.I.T. Press

## Preface

This two-volume book is based on a first-yoar graduate course on dynamic programming and optimal control that. I have taught for over twenty vears at Stanford University, the University of Illinois, and the MasSachusotts Institute of Technology. The couse has been typically attended by students from engineering, operations research, economics, and applied mathematics. Accordingly, a principal objective of the book has been to provide a unified treatment of the subject, suitable for a broad audience. In particular, problems with a continuous character, such as stochastic control problems, popular in modern control theory, are simultancously treated with problems with a discrete character, such as Markovian decision problems, popular in operations research. Rathermore, many applications and examples, drawn from a broad variety of fields, are discussed.

The book may be viowed as a greatly expanded and pedagogically improved version of my 1987 book "Dynamic Programming: Deterministic and Stochastic Models." published by Prentice-Mall. I have included much now material on dotorministic and stochastic shortest path problems. as well as a new chapter on continuons-time optimal control problems and the Pontyagin Maximum Principle, developed from a dynamic programming viewpoint. I have also added a fairly extensive exposition of simulationbased approximation techniques for dynamic programming. These techniques, which are often referred to as "neuro-dynamic programming" "reinforcement. learning," represent a breakthrough in the practical application of dynamie programming to complex probloms that involve the dual curse of large dimension and lack of an accurate mathematical model. Other material was also augmented, substantially modified, and updated.

With the new material, however, the book grow so much in size that it. became necessars to divide it into too volumes: one on finite horizon, and the other on infinite horizon problems. This division was not only natural in terms of size, but also in terms of style and orientation. The first volume is more oriented towards modeling, and the second is more oriented towards mathematical analysis and computation. To make the first volume selfcontained for instructors who wish to cover a modest amont of infinite horizon material in a course that is primarily oriented towards modeling,

conceptualization, and finite horizon problems, I have added a final chapter that provides an introductory treatment of infinite horizon problems.

Many topics in the book are relativoly independent of the others. For example Chapter 2 of Vol. I on shortest path problems can be skipped without loss of continuity, and the same is true for Chapter 3 of Vol. 1, which deals with continuous-time optimal control. As a result, the book can be used to teach several ditferent types of courses.

- (a) A two-semester course that covers both volumes.
- (b) A one-semester course primarily focused on finite horizon problems that covers most, of the first volume.
- (c) A one-semester course focused on stochastie optimal control that covers Chapters 1, 4, 5, and 6 of Vol. 1, and Chapters 1, 2, and 4 of Vol.
- (c) A one-semester course that covers Chapter 1, about 50% of Chapters 2 through 6 of Vol. I, and about 70% of Chapters 1, 2, and 1 of Vol. 11. This is the course I usually teach at MIT.
- (0) A one-quarter mathematically oriented course focused on infinite horizon problems that covers Vol. II.
- (d) A one-quarter engineering course that, covers the first three chapters and parts of Chapters 4 through 6 of Vol. I.

The mathematical prerequisite for the text is knowledge of advanced calculus, introductory probability theory, and matrix-vector algebra. A summary of this material is provided in the appendixes. Naturally, prior exposure to dynamic system theory, control, optimization, or operations research will be helpful to the reader, but based on my experience, the material given here is reasonably self-contained.

Dynamic programming is a conceptually simple technique that, can be adequately explained using clementary analysis. Yet a mathematically rigorous treatment, of general dynamic programming requires the complicated machinery of measure-theoretic probability. My choice has been to bypass the complicated mathematies by developing the subject in generality, while claiming rigor only when the underlying probability spaces are countable. A mathematically rigorous treatment of the subject is carried out in my monograph "Stochastic Optimal Control: The Discrete Time Academic Press, 1978, coauthored by Steven Shreve. This monograph complemonts the present text and provides a solid foundation for the

The book contains a large number of exercises, and the serious reader will beuefit greatly by going through them. Solutions to all exercises are compiled in a manual that, is available to instructors from Athena Scientific or from the author. Many thanks are due to the several people who spent long hours contributing to this manual, particularly Steven Shreve, Eric Loiederman, Lakis Polymenakos, and Cynara Wi.

subjoets developed somewhat informally here.

Finally, I am thankful to a mber of individuals and institutions for their contributions do the book. My understanding of the subject was sharponed while I worked with Stoven Shrove on our 1978 monograph. My interaction and collaboration with John Tsitsiklis on stochastic shortest paths and approximate dynamic programming have been most valuable. Michael Caramanis, Emmanuel Fernandez-Gauchorand, Pierre Humblet, Lomart Ljung, and John Tsitsiklis taught from versions of the book. and contributed several substantive comments and homework problems. A number of colleagues offered valuable insights and information, particularl David Castanon, Eugene Feinberg, and Krishua Pattipati. NSF provided rescarch support. Prentice-Hall graciously allowed the use of material from my 1987 book. Teaching and interacting with the students at ANT' have kept up my interest and excitement for the subjeet.

Dimitri P. Bertsokas bertsokas@lids.mit.odu

## Infinite Horizon Discounted Problems

| Contents   |
|------------|

This volume focuses on stochastic optimal control problems with an infinite number of decision stages (an infinite horizon). An introduction to these problems was presented in Chapter 7 of Vol. I. Here, we provide a more comprehensive analysis. In particular, we do not assume a finite mumber of states and we also discuss the associated analytical and computational issues in much groater depth.

We rocall from Chapter 7 of Vol. I that there are four classes of infinite horizon problems of major interest.

- (a) Discounted problems with bounded cost, per stage.
- (b) Stochastic shortest path problems.
- (e) Discounted and undiscounted problems with unbounded cost por stage.
- (4) Average cost por stage problems.

Back one of the first four chapters of the present volume considers one of the above problem classes, while the final chapter extonds the analysis to contimous-time problems with a countable mumber of states. Throughout this volume we concentrate on the perfoct information case, where each docision is made with exact knowledgo of the current system state. Inperfect state information problems can be treated, as in Chapter 5 of Vol. 1. by rotormulation into perfect information problems involving a sufficient. statistic.

## 1.1 MINIMIZATION OF TOTAL COST - INTRODUCTION

We now formulate the total cost minimization problem, which is the subject of this chaptor and the next two. This is an infinite horizon, stationary version of the basie problem of Chapter 1 of Vol. I.

## Total Cost Infinite Horizon Problem

Consider the stationary discrete-time dynamic system

<!-- formula-not-decoded -->

where for all h, the state th is an element of a spaceS, the control in is an oloment. of is space Co, and the random disturbance of is an element. of a space D. We assume that D is a countable set. The control uk is constrained to take values in a given nonempty subset U(ak) of C, which depends on the current state te labe lilia), for all teC Sl. The radom disturbances na.k = 0, 1o.., have identical statistics and are characterized by probabilities PC. hold) defined on D, where Pluck-uk) is the probability of occurrence of wh, when the current state and control are th and th, respectively. The probability of an may depend explicitly on th and «k but. not. on values of prior disturbances us p...1%.

Given an initial state to, we want to find a policy a = 140,11,...3, where ph: S → C, 14(ad) E U(ch), for all ta E S, A = 0,1,.... that. minimizes the cost. function t

<!-- formula-not-decoded -->

subject to the system equation constraint. (1.1). The cost per stage y: S x CX D-→ " is given, and o is a positive scalar referred to as the discount factor.

We denote by ll the set of all admissible policies o, that is, the set of all sequences of functions a = 140,14,...f with ph: S'+→ C, 14(1) € U(ir) for all r € S, A = 0,1... The optimal cost finction fe is dotined l

A stationary policy is an admissible policy of the form o = 14, 1,...1. and its corresponding cost function is denoted by dy. for brevity, wo refer to fu.p....? as the stationary policy p. We say that ye is optimal if J, (2) = J*(e) for all states t.

Note that, while we allow arbitrary state and control spaces, wore quire that the disturbance space be countable. This is necessary to avoid the mathematical complications discussed in Section 1.5 of Vol. I. The countability assumption, however, is satisfied in many problems of intorest, notably for deterministic optimal control problems and problems with a finite or a countable number of states. For other problems, our main results can typically be proved (under additional technical conditions) by following the same line of argument as the one given here, but also by dealing with the mathematical complications of various measure theoretic frameworks;, see (BeS78).

finite horizon costs. These costs are well defined as discussed in Section

+ In what follows we will generally impose appropriate assumptions on the cost per stage g and the discount factor a that guarantee that the limit defining the total cost a(co) exists. I this limit is not known to exist. we use instead the definition

<!-- formula-not-decoded -->

- -1

1.5 of Vol. I. Another possibility would be to minimize over n the expected infinite horizon cost

<!-- formula-not-decoded -->

Such a cost. would require a far more comples mathematical formulation (a probability measuro on the space of all disturbance sequences; seo (BeS78]). Howeset, we mention that, under the assumptions that we will be using, the proceding expression is equal to the cost given by Ed. (1.2). This may be proved by using the monotone convergence theorem (see Section 3. 1) and other stochastic convergence theorems, which allow interchange of limit and expectation under appropriate conditions.

## The DP Algorithmn for the Finite-Horizon Version of the Problem

Consider any admissiblo policy n= 440. 81,..8, any positive intoger N. and any finction : Stodi. Suppose that we accumulate the costs of the first. N stages, and to them we add the terminal cost cod. (on), for a total expocted cost.

<!-- formula-not-decoded -->

The minimum of this cost over o ran be calenlated by starting with oN dc) and by carrying ont. N iterations of the corresponding DP algorithm of Section 1:3 of Vol. 1. This algorithm expresses the optimal (N - h)-stage cost starting from state t, denoted by d(r), as the minimum of the expected sum of the cost of stage N- d and the optimal (N -hi- 1)-stage cost starting from the next state. It is given by

<!-- formula-not-decoded -->

with the initial condition

For all initial states d, the optimal Nestage cost. is the finction lo) obtained from the last step of the DI algorithm.

Let as consider for all l ande, the functions Va given by

<!-- formula-not-decoded -->

Then Ta(e) is the optimal N-stage cost Jod), while the Dl' recursion (1.3) can be equivalently be written in torns of the functions fi as

<!-- formula-not-decoded -->

with the initial condition

<!-- formula-not-decoded -->

The above algorithm can be used to calculate all the optimal finite horizon cost functions with a segle DI recursion. In particular, supposo that we have computed the optimal (N - 1)-stage cost function la-1. Then, to calculate the optimal N-stage cost finction Vo, we do not aced to exocute the Nestage DI algorithm. Instead, we can calculate to using the one-stage iteration

More generally, starting with some terminal cost function, wo can consider applying repeatedly the DP iteration as above. With each application, we will be obtaining the optimal cost finetion of some finite horizon problem. The horizon of this problem will be longer by one stage over the horizon of the preceding problem. Note that this convonience is possible only because we are dealing with a stationary system and a common cost function o for all stages.

## Some Shorthand Notation

The preceding method of calculating finite horizon optimal costs motivates the introduction of two mappings that play an important theoretical role and provide a convenient shorthand notation in expressions that. would be too complicated to write otherwise.

For any function d: S-M. wo consider the function obtained bo applying the DP mapping to J, and we denote it by t

<!-- formula-not-decoded -->

Since (T.)(·) is itsell' a function defined on the state space 5, we vien T'as a mapping that transforms the function of on S into the function T. onS. Note that I. is the optimal cost function for the one-stage problem that has stage cost g and terminal cost a..

+ Whenever we use the mapping 7', we will impose sufficient assmuptions to guarantee that the expected value involved in by. (1.1) is well defined.

Similarly, for any function: SeD and any control function p: SeC, we denote

Again, Two may be viewed as the cost function associated with de for the one-stage problem that has stage cost g and torminal cost. a..

We will denote by Th the composition of the mapping T' with itself hi times; that is, for all hi we write

Thas th is the lection obtained by applying the mapping 'I' to the function Th-11. For convenience, we also write

<!-- formula-not-decoded -->

Similarly, Thy is defined by

and

<!-- formula-not-decoded -->

It can be verified by induction that (Ti)(x) is the optimal cost for the ki-stage, o-discounted problem with initial state x, cost per slage y, and terminal cost function old. Similarly, (Th./(e) is the cost of a policy {e, 1,...) for the same problem. To illustrate the case where hi = 2, note that.

<!-- formula-not-decoded -->

The last expression can be recognized as the DP algorithm for the 2-stage, o-discounted problem with initial stated, cost per stage ?, and terminal cost function a2.).

and represouts the cost of the policuo for the kestage. d-discounted problem with initial state d, cost per stage a, and terminal cost function of f

.......-.

.......

## Some Basic Properties

The following monotonicity properte plays a fundamental cole in the developments of thus volume.

Lemma 1.1: (Monotonicity Lemma) For any functions 1: 5→ I and J': S + R, such that

<!-- formula-not-decoded -->

and for any function u: S +- C with e(x) € U(d), for all a E S, we have

<!-- formula-not-decoded -->

Proof: The result follows by viewing (Th./)(x) and (Th.)(e) as li-stage problem costs, since as the terminal cost function increases uniformly so will the ki-stage costs. (One can also prove the lemma by using a straight forward induction argument.) Q.E.D.

For any two functions J: S'- D and D': 5+→ De, We write

<!-- formula-not-decoded -->

With this notation, Lemma 1.1 is stated as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us also denote bye: S' R the uit function that takes the value 1 identically on S:

<!-- formula-not-decoded -->

We have from the definitions (1.1) and (1.5) of T and I,, for any function 1: 51→ M and scalar i

<!-- formula-not-decoded -->

More generally, the following lemma can be verified by induction using the preceding two relations.

Lemma 1.2: For every ki, function o: 5- li, stationary policy 4, and scalar r, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A Preview of Infinite Horizon Results

It is worth at this point to speculate on the type of results that we will be aiming for.

- (a) Converyence of the DP Algorithm. Let Jo denote the zero function {Jo(e) = 0 for all de). Since the infinite horizon cost of a policy is, by dofinition, the limit of its hostage costs as li → oc, it is reasonable to speculate that the optimal infinite horizon cost is equal to the lint. of the optimal k-stage costs; that is,

<!-- formula-not-decoded -->

This means that if we start with the zero function Jo and iterate with the DI algorithin indefinitely, we will get in the limit the optimal cost. function J*. Also, for @ &lt; 1 and a bounded finction J, a terminal cost. of diminishes with ki, so it is reasonable to speculate that, if a &lt; 1,

<!-- formula-not-decoded -->

- (b) Bellman's Equation. Since by definition we have for all &amp; € S'

<!-- formula-not-decoded -->

it is reasonable to speculate that if link -x ThiJo = J* as in (a) above, then we must have by taking limit. as li → oo,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or, equivalently,

This is known as Bellman's coation and asserts that the optimal cost luction f* is a fixed point of the mapping T. We will see that

Bellman's equation holds for all the total cost minimization probloms that we will consider. although depending on our assumptions, its proof can be quite complex.

- (·) Churactorcation of Oplimal Statonary Policies. I wo vion Bollman's equation as the DP algorithm taken to its limit as koa. il is reasonable to speculato that if p(r) attains the minimum in the righthand side of Bellman's equation for all r, then the stationary policy « is optimal.

Most of the analysis of total cost infinito horizon problems revolves around the above throe issues and also around the issue of oflicient computation of J* and an optimal stationary policy. For the discounted cost problems with bounded cost per stage considered in this chapter, and for stochastic shortost path problems under our assumptions of Chapter 2. the preceding conjectures are correct. For problems with unbounded costs por stage and for stochastic shortest path problems where on assumptions of Chapter 2 de violated, there may be counterintuitive mathematical ple nomena that invalidate some of the proceding conjectures. This illustrates that infinite horizon problems should bo approached carefully and with mathematical precision.

## 1.2 DISCOUNTED PROBLEMS WITH BOUNDED COST PER. STAGE

We now discuss the simplest type of infinite horizon problem. He assume the following:

Assumption D (Discounted Cost - Bounded Cost per Stage): The cost per stage g satisfies

<!-- formula-not-decoded -->

where M is some scalar. Furthermore, 0 &lt; xx &lt; 1.

Boundedness of the cost per stage is not as restrictive as micht appear. It hods for problems where the spaces S, C, and D are finite sets. Even if these spaces are not finite, during the computational solution of the problem they will ordinarily be approximated by finite sets. Also, it is often possible to reformulate the problem so that it is defined over bounded regions of the state and control spaces over which the cost is bounded.

The following proposition shows that the DP algorithm converges to the optimal cost function f* for an arbitrary bounded starting function . This will follow as a consequence of Assumption D, which implies that the "tail" of the cost after stage N, that is,

diminishes to zero as N → x. Parthermore, when a torminal cost aN en) is added to the N-stage cost, its effect diminishes to zero as N → oo if J is bounded.

Proposition 2.1: (Convergence of the DP Algorithm) For any bounded function f : S. Ji, the optimal cost function satisfies vil

<!-- formula-not-decoded -->

Proof: For evory positive intoger A, initial state to CS, and policy o= 110,81,...J. we break down the cost da(co) into the portions incurred over the first. A stages and over the remaining stages

<!-- formula-not-decoded -->

Since by Assumption D we have le(chopk(a), 14)1 ≤ 1S, we also obtain

Using these relations, it follows that

<!-- formula-not-decoded -->

By taking the minimum over a, we obtain for all to and A,

<!-- formula-not-decoded -->

and by taking the limit as A → a, the result follows. Q.E.D.

Note that based on the preceding proposition, the DP algorithm may be used to compute at least an approximation to J*. This computational method together with some additional methods will be examined in the next section.

Given any stationary policy 4, we can consider a modified discounted problem, which is the same as the original except that the control constraint. set contains only one element for each state e, the control 4(2): that is, the control constraint set is Ü(x) = {u(x)} instead of U(2). Proposition 2.1 applies to this moditied problem and yields the following corollary:

Corollary 2.1.1: For every stationary policy y, the associated cost function satisfies

<!-- formula-not-decoded -->

The next proposition shows that. Jr is the unique solution of Bellman's equation.

Proposition 2.2: (Bellman's Equation) The optimal cost function J* satisfies

<!-- formula-not-decoded -->

or, equivalently,

<!-- formula-not-decoded -->

Furthermore, J* is the unique solution of this equation within the class of bounded functions.

Proof: From Ey. (2:3), we have for ill i € Sand N,

<!-- formula-not-decoded -->

where lo is the aero function Joc) = 0 for all &amp; € S). Applying the mapping T' to this relation and using the Monotonicity Loma I. as well as Lemma 1.2, we obtain for all &amp; E Sand N

<!-- formula-not-decoded -->

Since (T'N+10)(r) converges to A*(r) (of. Prop. 2.1), by taking the limit. as N → so in the preceding relation, we obtain J* = T.J*.

To show uniqueness, observe that if J is bounded and satisfies J = T.J, then A = limN-x TN/, so by Prop. 2.1, wo have J= J*. Q.E.D.

Based on the same reasoning we used to obtain Cor. 2.1.1 from Prop. 2.1, wo have:

Corollary 2.2.1: For every stationary policy y, the associated cost function satisfies

<!-- formula-not-decoded -->

(2.7)

or, equivalently,

Furthermore, J, is the unique solution of this equation within the class of bounded functions.

The next proposition characterises stationary optimal policies.

Proposition 2.3: (Necessary and Sufficient Condition for Optimality) A stationary policy de is optimal if and only if 4(2) attains the minimum in Bellman's equation (2.5) for each &amp; € S; that is,

<!-- formula-not-decoded -->

Proof: 11 TJ* = 0,0, then using Bolhnan's equation (J* = T.J*), we have J* = 1,0*, so by the uniqueness part of Cor: 2:2.1, we obtain f* = ,;

... ...re..

that is, de is optimal. Conversely, if the stationary policy ye is optimal, wo have J* = J,. which by Cor. 2.2.1, yiolds J* = T,/*. Combining this with Bollman's equation (J* = TJ*), we obtain 1.* = 4,1*. Q.E.D.

Note that Prop. 2.3 implies the existence of an optimal stationary policy when the minimum in the right-hand side of Bellman's cquation is attained for all r € S. In particular, when U(r) is finite for cache C.S, an optimal stationary policy is guaranteed to exist.

No finally show the following convergence rate estimate for any bounded function J:

<!-- formula-not-decoded -->

This rolation is obtained by combining Belhnan's equation and the following result:

Proposition 2.4: For any two bounded functions f: S' - li, S→ 92, and for all h = 0,1,..., there holds

Proof: Denote

Then we have

Applying Tl in this relation and using the Monotonicity Lemma I. as well as Lemma 1.2, we obtain

<!-- formula-not-decoded -->

It follows that

which proves the result. Q.E.D.

As carlier, we have:

Corollary 2.4.1: For any two bounded functions J : S → R, J' : S-N, and any stationary policy y, we have

## Example 2.1 (Machine Replacement)

Consider an infinite horizon discounted version of a problem we formulated in Section 1.1 of Vol, 1. Here, we want to operate elliciently a machine that. can be in any one of a states, denoted 1,2,...,%. State 1 corresponds to a machine in perfect condition. The transition probabilities pe are given. There is a cost. gi) for operating for one time period the machine when it is in state i. The options at the start of each period are to (a) let the machine operate ono more period in the state it currently is, or (b) replace the machine with a now machine (state 1) at a cost R. Once replaced, the machine is guaranteed lo stay in stato 1 for one period; in subsequent periods, it may deteriorato to states j 2 1 according to the transition probabilities pog. We assumo an indinite horizon and a discount factor or € (0, 1), so the theory of this section applies.

Bellman's equation (of. Prop. 22) takes the form

By Prop. 2.3, a stationary policy is optimal if it roplaces at states i where

and it. does not. replace al states i where

We can use the convorgence of the DP algorithm (ef. Prop. 2.1) 1o takes the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assume that gfi) is nondecreasing in i, and that the transition probabilities satisty

<!-- formula-not-decoded -->

for all functions Ji), which are monotonically nondocreasing in i. I can be shown that this assumption is satisfied if and only it, for every hi. E" ail, is monotonically nondecreasing in i (sco (Ros83b), p. 252). The assumption (2.10) is satisfied in particular if

<!-- formula-not-decoded -->

i.e., the machine cannot go to a better State with usage, and

<!-- formula-not-decoded -->

i.e.. there is greater chance of ending at a bad state jif we start at a vorso State i. Since gi) is nondecreasing in i. we have that (T.)(i) is nondecreasing in i, and in view of the assumption (2.10) on the transition probabilities, the sumo is true for (P%o)(i). Similarly, it is seen that, for all hi, (To)(1) is nondecreasing in i and so is its limit.

This it word clershe out cust the uld dear us the machine

is nondecreasing in i. Consider the set. of states

and let.

<!-- formula-not-decoded -->

Then, an optimal policy takes the form

<!-- formula-not-decoded -->

as shown in Fig. 1.2.1.

Figure 1.2.1 Determining the optimal policy in the machine replacemont exam-

<!-- image -->

## 1.3 FINITE-STATE SYSTEMS - COMPUTATIONAL METHODS

In this section we discuss several alternative approaches for numerically solving the discounted problem with bounded cost per stage. The first approach, value iteration, is essentially the DP algorithan and yioks in the limit. the optimal cost linction and in optimal policy, as discussed in the preceding section. We will describe some variations aimed at accelerating convergence. Two other approaches, policy iteration and linear programming, terminate in a finite number of iterations (assuming the number of states and controls are finite). However. when the number of states is large, these approaches are impractical because of large overhead per iteration. Another approach, ilaptive aggregation, bridges the gap between value iteration and poliey iteration, and in a sense combines the best features of both methods.

Iu Section 2.3 we will consider some additional mothods, which are woll-suited for dynamic systems that. are hard to model but relativoly casy to simulate. In particular, we will assume in Section 2.3 that the transition probabilities of the problem are unknown, but the system's dynmics and cost structure can be observed through simlation. We will then discuss ileration using. for example, neural networks.

We first translate some of our carlier analysis in a notation that is sure we Cake des, Did the state sper S coint of a

Throughout this section we assume a discounted problem (Assumption D holds). We farther assume that the state, control. and disturbince spaces underlying the problem are finite sets, so that. we are dealing in effect with the control of a finite-state Markov chain.

<!-- formula-not-decoded -->

We denote D p(a) the transition probabilities

<!-- formula-not-decoded -->

These transition probabilities may be given a priori or may be calentated from the system equation

<!-- formula-not-decoded -->

and the known probability distribution P(fe. a) of the input disturbance Wh. Indeed. we have

<!-- formula-not-decoded -->

where We,(u) is the (finite) set

To simplify notation. we assume that the cost per stage does not. depend on . This amounts to using expected cost per stage in all calentations, which makes no essential diflerence in the definitions of the mappings T and To of Egs. (1.1) and (1.5), and in the subsequent analysis. Thus. it gli, u.) is the cost of using a at state i and moving to state j, we use as cost per stage the expocted cost. gi. u) given by

<!-- formula-not-decoded -->

The mappings T' and 1, of Bys. (101) and (15) can be written as

<!-- formula-not-decoded -->

Any function of on S, is well as the functions T.J and To may be ropresented by the n-dimensional vectors

<!-- formula-not-decoded -->

matrix For a stationary policy ye, we denote by Pa the transition probability

<!-- formula-not-decoded -->

and by gy the cost. vector

<!-- formula-not-decoded -->

We can then write in vector notation

The cost timotion d, corresponding do a stationary policy ye is, by Cor. 2.2.1, the unique solution of the equation

This ecuation should be vowed as a system of a linear equations with « unknowas, the components 1, (i) of the a-climensional vector ,. The oquation can also be written as

or, equivalently,

<!-- formula-not-decoded -->

where / denotes the ox n identity matrix. The invertibility of the matrix 1-of, is assured since we have proved that the system of equations representing a = Too, has a unique solution for any vector go (ef. Cor. 2:2.1). For another way to see that 1-oP, is an invertible matrix, note that the cigenvales of any transition probability matrix lie within the unit. circle of the complex plane. 'Thas no eigenvalue of al, can be equal to 1, which is the nerossary and sullicient condition for 1- raP, to be invertible.

## 1.3.1 Value Iteration and Error Bounds

Here wo start with any n-dimensional rector and successively compute T.J, T2.J,.... By Prop. 2.1, we have for all i

Furthermore, by Prop. 2.1 (using ' = * in Ey. (29)l, the error sequence |(Tk J)(i) - J*(i) is bounded by a constant multiple of ah, for all i € S. This method is called value iteration or successive upprozimation. The method can be substantially improved thanks to certain monotonie error bounds, which are casily obtained as a byproduct of the computation.

The following argument is helpful in understanding the nature of those bounds. Let us first break down the cost of a stationary policy de into the first stage cost and the romainder:

It follows that

<!-- formula-not-decoded -->

where e is the unit vector. e = (1, 1..., 1)', and / and 3 are the minimum and maximum cost per stage:

Using the definition of is and 3. we can strengthen the bounds (32) as follows:

These bounds will now be applied in the context of the value iteration method.

Suppose that we have a rector of and no combute

By subtracting this cquation from the relation

we obtain

This equation can be vieved as a mariational form of the equation o, = Tode, and implies that do - is the cost vector associated with the stationany policy le and a cost por stage vector equal to Tud - J. Therefore, the bounds (3.3) apply with , replaced by da-and in replaced by T"d -J.

<!-- formula-not-decoded -->

where

Equivalently, for every vector J, ve have

where

<!-- formula-not-decoded -->

The following proposition is obtained by a more sophisticated application of the preceding argument.

Proposition 3.1: For every vector J, state i, and li, we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: Denote

We have

Applying 1' to both sides and using the monotonicity of 1', we have

and, combining this relation with By. (3.7), we obtain

<!-- formula-not-decoded -->

This process can be repeated, tirst applying T to obtain

<!-- formula-not-decoded -->

and then using Ey. (3.7) to write

<!-- formula-not-decoded -->

After li steps. this results in the inequalities

<!-- formula-not-decoded -->

Taking the limit as he do and using the coquality go = ag/(1-a). wr obtain

<!-- formula-not-decoded -->

where cy is defined by ky. (35). Replacing by the in this inequalits: wr have

which is the second inequality in liy. (3.1).

From Ey. (3.8). we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and consequently

<!-- formula-not-decoded -->

Using this relation in 1g. (3.9) violds

<!-- formula-not-decoded -->

and replacing a le Te 10. we have the first inequality in Bo. (3.1). An analogons argument shows the last two inequalities in la. (3.1). Q.E.D.

We note that the preceding proof does not rely on the finiteness of the state space, and indeed Prop. 3.1 can be proved for an infinite state space (soo also Exercise 1.9). The following example demonstrates the nature of the error bounds.

## Example 3.1 (Illustration of the Error Bounds)

C'onsider a problem where there are two states and two controls

<!-- formula-not-decoded -->

The transition probabilities coresponding to the controls d' aud n are as shown in Fig. 1.3.1: that is, the transition probability matrices are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The transition costs are

<!-- formula-not-decoded -->

and the discount factor is a = 0.9. The mapping 'I' is given for i = 1,2 by

<!-- formula-not-decoded -->

The scalars Co and ta of Eas. (3.5) and (3.0) are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The results of the valoe iteration mothod starting with the zero function Jo In(1) = 1(2) = 0) are shown in Fig. 1.3.2 and illustrate the power of the eror bomols.

Figure 1.3.1 Stato transition dingram for Example 3.1: (a) u = 1': (b) « = 1*.

<!-- image -->

Figure 1.3.2 Performance of the value iteration method with and withon the error bounds of Prop. 3.1 for the problem of Pixample :3.1.

|           | (Th. 10) (1)   | (7'1.10)(2) (Th Jo)(1)   | + Chi       | (Th 10) (1)   | (T'%: Jo) (2)   | (Th: 10)(2) titi   |
|-----------|----------------|--------------------------|-------------|---------------|-----------------|--------------------|
| 0 1 0.500 |                | 1.000                    | 5.000 9.500 | 5.500         |                 | 10.000             |
| 2 1.287   |                | 1.562                    | 6.350       | 8.375         | 6.625           | 8.650              |
| 3         | 1.811          | 2.220                    | 6.856       | 7.767         | 7.232           | 8.111              |
|           | 2.11.1         | 2.745                    | 7.121       | 7.510         | 7.160           | 7.870              |
| 5         | 2.896          | 3.217                    | 7.232       | 7.117         | 7.583           | 7.768              |
|           | 3.3.13         | 3.686                    | 7.287       | 7.371         | 7.629           | 7.712              |
| 7 3.710   |                | 4.086                    | 7.308       | 7.3.15        | 7.651           | 7.692              |
| 8 1.090   |                | 1.411                    | 7.319       | 7.3:36        | т.6633          | 7.680              |
| 9 1.122   | 1.767          |                          | 7.324       | 7.331         | 7.669           | 7.676              |
| 10        | 1.713          | 5.057                    | 7.326       | 7.329         | 7.671           | 7.67.1             |
| 11        | 1.971          | 5.319                    | 7.327       | 7.328         | 7.672           | 7.673              |
| 12 5.209  |                | 5.55.1                   | 7.327       | 7.328         | 7.672           | 7.6731             |
| 13        | 5.421          | 5.766                    | 7.327       | 7.328         | 7.672           | 7.07:              |
| 14        | 5.012          | 5.957                    | 7.32%       | 7.328         | 7.072           | 7.072              |
| 15        | 5.783          | 6.128                    | 7.328       | T.328         | 7.072           | 7.672              |

## Termination Issues - Optimality of the Obtained Policy

Let us now discuss how to use the error bounds to obtain an optimal

-...

or near-optimal policy in a finite number of value iterations. We first note that given any J, if we compute T./ and a policy d attaining the minimum in the calculation of To, io.. TadsTD, then we can obtain the following bound on the suboptimality of y

To soo this, apply Eq. (3.4) with k= 1 to obtain for all i

<!-- formula-not-decoded -->

and also apply 1a. (3.1) with li - land with Ta replacing T to obtain

Subtracting the abovo two cquations, we obtain the estimate (3.10).

In practico, one forminates the value iteration method when the dif forence (ca-g) of the error bounds becomes sufficiently small. One can then take as final estimate of J+ the "wedian"

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Both of these vectors lie in the region delineated by the error bounds. Then, the estimate (3.10) provides a bound on the suboptimality of the policy y attaining the minimom in the calculation of Th..

The bound (3.10) can also be used to show that after a sufliciently largo mober of value iterations, the stationary policy pi that attains the 7 &gt; 0 such that il a stationary policy ye satisties

theny is optimal. Now let &amp; be such that for all hi&gt; t we have

Then from Ey. (3.10) we see that for all l&gt; hi, the stationary policy that attains the miniman in the lith value iteration is optimal.

or the "averago"

## Rate of Convergence

To analyze the rade of convergence of value iteration with error bounds. assume that there is a stationary policy pt that attains the minimum over fe in the relation

<!-- formula-not-decoded -->

for all li sulliciontly large, so that eventually the method reduces to the linear iteration

<!-- formula-not-decoded -->

In view of our preceding discussion, this is true for example if go is a unique optimal stationary policy. Generally the rate of convorgence of linear iterations is governed by the maximun cigenvalue modulus of the matrix of the iteration (which is a in our case, since any transition probability matrix has a unit. rigonvalue with corresponding cigenvoctor e = (1,1,..., 1). while all other eigenvalues lie within the mit circle of the complex plane.

It turns out, however. that when error bounds are used, the rate at which the iterates he and th of Egs. (3.11) and (3.12) approach the optimal cost vector J* is governed by the modulus of the subdominant eigenvalue of the transition probability matris Pyo. that is. the eigenvalue with second largest modulus. The proof of this is outlined in lexerrise 1.8. For a sketch of the ideas involved, let deo... d, be the cigenvalues of P," ordered according to decreasing modulus; that is

<!-- formula-not-decoded -->

with d, equal to 1 and de being the subdominant cigenvalue. Assumo that there is a set of linearly indepondent eigenectors co.cao..r, corresponding to 11,12...., 1, With e=r= (1.1. ..., 1)'. Then the initial enor J- Jee can bo expressed as a linear combination of the cigenrectors

for some sendare Er. EnEm. Since Tred = He tale and lie s gue trol, dye, successise errors are rolated by

<!-- formula-not-decoded -->

Thus the error after l iterations can be written as

Using the error bounds of Prop. 3. 1 amounts to a translation of Tied along the vector c. Thus, at best. the error bounds are tight enough to eliminate the component alge of the error, but cannot aflect the romainine term adiE MEiei, which diminishes like ablo with do being the subdominant eigensalur.

## Problems where Convergence is Slow

In Example 3.1. the convorgence of value iteration with the error bounds is vory fast. For this example, it can be verified that 1*(1) = u2. 11* (2) = 1, and that.

<!-- formula-not-decoded -->

The eigenvalues of Par can be calculated to be d, = Land 12 = -2, which explains the fast convergence, since the modulus 1/2 of the subdominant. digenvalue de is considerably smaller than one. On the other hand, there are situations where convergence of the method even with the use of error bounds is very slow. For example, suppose that P, is block diagonal with two or more blocks, or more generally, that Pue corresponds to a system with more than one recurrent class of states (see Appendix D of Vol. 1). Then it can be shown that the subdominant digenvalue de is equal to 1. and convergence it typically slow when a is close to 1.

As an example. consider the following three simple deterministic problems. cach having a single policy and more than one recurent class of st.at.os:

Problem 1: 1=: 3. 1 = Merce-dimensional idontity, glin(i)) = 1.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 1.3.3 shows the mumber of iterations needed by the value itcration method with and withont the error bounds of Prop. 3.1 to find of, within an orror per coordinate of less than or equal to 10-6 max;J,(i)l. The starting finetion in all cases was taken to be coro. The performance is rather unsatisfactory but, nonetheless. is typical of situations where the subdominant. cigenvalue modulus of the optimal transition probabilily matrix is close to 1. One possible approach to improve the performance of valuo iteration for such problems is based on the adaptive aggrecation method to be discussed in Section 1.3.3.

Figure 1.3.3 Number of iterations for the value iteration method with and withe out error bounds. The problems are deterministic. Because the subdominant eigenvalue of the transition probability matrix is equal to 1, the error bounds are ineflective.

|              |   1'r. 1 |   Pr. 1 0=9/4=09 |   Pr. 2 |   Pr. '2 ÷ .99 |   Pr. 3 | Pr. 3   |
|--------------|----------|------------------|---------|----------------|---------|---------|
| W/out bounds |      131 |             1371 |     131 |           1371 |     132 | 1392    |
| With bounds  |      127 |             1333 |     129 |           1352 |     131 | 13TI    |

## Elimination of Nonoptimal Actions in Value Iteration

We know from Prop. 2.3 that, if i € U(i) is such that

<!-- formula-not-decoded -->

then i cannot be optimal at state i; that is, for evory optimal stationary policy a, we have ali) #i. Therelore, if we are sure that the above inequality holds, we can salely climinate a from the admissible sor U(i). While we cannot chock this incquality, since we do not. know the optimal cost function J*, we can guarantee that it holds if

<!-- formula-not-decoded -->

where T and 1 are upper and lover bounds satisfying

<!-- formula-not-decoded -->

The preceding observation is the basis for a useful application of the error bounds given carlier in Prop. 3.1. As these bounds ate computed in the course of the value iteration method, the inequality (3.13) can be simultancously checked and nonoptimal actions can be eliminated from the acinissible set with attendant savings in subsequent computations. Since the upper and lower bound functions J and &amp; converge to J*, it can be seen taking into account the finiteness of the constrain set. U(i)l that eventualls all nonoptimal i € U(i) will be climinated, thereby reducing the set fri) after a finite mumber of iterations to the set of controls that are optimal at ¿. In this manner the computational requirements of value iteration can be substantially reduced. However, the amount of computer memory required to maintain the set of controls not as yet climinated at cach i eS mas he increased.

## Gauss-Seidel Version of Value Iteration

In the value iteration method described carlier, the estimate of the cost function is iterated for all states simultancously. An alternative is to iterate one state at a time, while incorporating into the computation the interim results. This corresponds to using what is known as the GaussSeidel method for solving the nonlinear system of equations 1 - 1. (ser (BeT89a) or (OrR70)).

For a-dimensional voctors o, define the mapping F by

<!-- formula-not-decoded -->

and, for i = 2,...,11,

Furthermore, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In words, (F./)(i) is computed by the same equation as (T.J)(i) excopt that the previously caleulated values (F.D)(1),..., (FJ)(i - 1) are used in place of J(1),..,J; - 1). Note that the computation of FJ is as casy as the computation of T.J (unless a parallel computer is used, in which case the computation of T. may potentially be obtained much faster than EJ; see ('Tsis!), [Bot9ta| for a comparative analysis).

Consider now the value iteration method whereby we computes, FJ, 12J,.. The following propositions show that the method is valid and provide an indication of better performance over the carlier value iteration mothod.

Proposition 3.2: Let J, J' be two n-dimensional vectors. Then for any ki = 0, 1,...,

<!-- formula-not-decoded -->

Proof: It is sullicient to prove Ey. (3.16) lor A = 1. We have by the definition of Fand Prop. 2.4,

<!-- formula-not-decoded -->

Also, using this inequality:

<!-- formula-not-decoded -->

Proceeding similarly, we have, for every i and i &lt; i,

so Eq. (3.16) is proved for fi = 1. The cquation P.J* = * follows from the definition (3.14) and (3.15) of F, and Bellman's equation * = TJ". The convergence property (3.18) follows from Eas. (3.16) and (3.17). Q.E.D.

Proposition 3.3: If an a-dimensional vector of satisfies

<!-- formula-not-decoded -->

then

<!-- formula-not-decoded -->

Proof: The proof follows by using the definition (3.14) and (3.15) of F. and the monotonicity property of T (Lemma 1.1). Q.E.D.

The preceding proposition provides the main motivation for employing the mapping F in place of T in the value iteration method. The result indicates that the Crauss-Seidel version converges faster than the ordinary value iteration method. The faster convergence property can be substantiated by further analysis (soo og., (BeT89a)) and has been continned in practico through extensive experimentation. This comparison is somewhat misleading. however. because the ordinary mothod will normally be used in conjunction with the error bounds of Prop. 3.. One mar also emplos error bounds in the Gauss-Seidel version (see Exercise 1.9). However. there is no clear superiorits of one method over the other when bounds are jus troduced. Rathermore. the ordinary method is botter suited for parallel computation than the Ganss-Seidel version.

No note that there is a more flexible form of the Gauss-Seidel method, which selects states in arbitrary order to update their costs. This mothod maintains an approximation A to the optimal vector f*, and at cach iteration. il selects a state i and replaces di) by (TJ)(i). The remaining values .%, #i, are left unchanged. The choice of the state i at each iteration is arbitrarv. excopt for the restriction that. all states are solected infinitoly often. This method is an example of an asynchronous fired point iteration and can be shown to converge to * starting from any initial .. Analyses of this type of mothod are given in (Ber&amp;Zal, and in Chapter Gof (BoT89a): ser also Exercise 1.15.

## Generic Rank-One Corrections

We may view value iteration coupled with the error bounds of Prop. 3.1 as a mothod that makes a correction to the results of value iteration along the mit vector e. It is possible to generalize the idea of correction along a fixed vector so that it works for any type of convorgent lincar itoration.

Let us consider the case of a single stationary policy fe and an iteration of the form o := 1., where

<!-- formula-not-decoded -->

Here, Qu is a matris with eigenvalues strictly within the unit circle, and hi, is a vector such that

An example is the Gauss-Seidel iteration of Section 1.3.1, and some other examples are given in Exercises 1.4, 1.5, and 1.7, and in Section 5.3. Also, the value iteration method for stochastic shortest path problems and a single stationary policy, to be discussed in Section 2.2, is of the above form.

Consider in place of := J. an iteration of the form

<!-- formula-not-decoded -->

where is related to boy

<!-- formula-not-decoded -->

with ol a fixed vector and &amp; a scalar to be selected in some optimal manner. In particular, consider choosing by minimizing over y

<!-- formula-not-decoded -->

which, by denoting

can be writton as

<!-- formula-not-decoded -->

By setting to vero the derivative of this expression with respect to g. it is straightforward to vorily that the optimal solution is

<!-- formula-not-decoded -->

Thus the iteration A := 1J can bo written as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

We note that this iteration requires only slightly more computation than the iteration of := E.J. since the vectore is computed once and the computation of § is simple.

A key question of course is under what circumstances the iteration J:= M. converges faster than the iteration f := F.. and whother indeed it convorges at all tody. K is straightforward to verify that in the case where Q, = oP, and d =6, the iteration D := MJ can be written as

compare with Ed. (3.12)1. 'Thus in this case the iteration J:= 16) shifts the result TaJ of value iteration to a vector that lios somewhore in the middle of the error bound range given by Prop. 3.1. By the resul of this proposition it. follows that the iteration convorges to d,.

Generally, however. the iteration o:= A. need not conserge in the case where the direction vector d is chosen arbitrarily. If on the other hand d. is chosen to be an eigenvector of Q,, convergence can be proved. This is shown in Exercise 1.8, where it is also proved that if d is an rigendertor corresponding lo the dominant cagonalus of On The one with laryest modulus), the convergence rate of the iteration J:= M. is governed by the subdominant eigenvalur of Q, (the one with second largest modults). One possibility for finding approximately such an cigenvoctor is to apply Fa sufficiently large mumber of times to a voctor d. In particular, suppose that. the initial error of - 1, can be decomposed as

for some scalars go....En, where co...en are cigenertors of On. and 1...., du are corresponding cigenvalues. Suppose also that d is the

- di)er for largo hi and can be nsed to estimate the dominant cigenvoctor c. I order to decide whother ki has been chosen large enough, one can test to see if the angle betweon the successive dillerences dat1 - Al lund A - Die 1 is very small; if this is so, the components of Ph+1, - Fk. along the cigenvoctors ca....Co must also be very small. (For a more sophisticated version of this argument, see (Ber93), where the generic rank-one correction method is developed in more goneral form.)

Ne can thas consider a to-phase approach: in the first phase. we apply soveral times the regular iteration 1 := PJ both to improve on estimate of of and also to obtain an estimate d of an eigenvector corresponding to a dominant cigenvalue; in tho second phase we use the modified iteration J:= M. that involves extrapolation along d. It can be shown that the two-phase method converges to J, provided the orror in the estimation of d is small onough, that is, the cosine of the angle between d and Qnd as mcasured by the ratio

<!-- formula-not-decoded -->

is sufliciently close to one.

Note that the computation of the first phase is not wasted since it uses the iteration . := EJ that we are trying to accelerate. Furthermore, since the second phase involves the calculation of F.J at the current iterate J, any error bonods or termination criteria based on PJ can be used to terminate the algorithm. As a result, the same finite termination mechanism can be used for both iterations d := PJ and J := A.J.

One dilliculty of the correction method outlined above is that the appropriato vector d depends on Qu and therefore also on f. In the case of optimization over several policies, the mapping F is defined by

<!-- formula-not-decoded -->

One can then use the rank-one correction approach in two different ways:

- (1) Iterativoly compute the cost vectors of the policies generated by a policy iteration scheme of the type discussed in the next subsection.
- (2) Guess at an optimal policy within the first phase, switch to the second phase, and then return to the first phase if the policy changes "substantially" during the second phase. In particular in the fist phase, the iteration J:= E.J is used, where F is the nonlinear mapping of Ey. (3.20). Upon switching to the second phase, the vector &amp; is taken

to be equal to Quad, where pt is the policy that attains the minimum in 1g. (3.20) at the time of the switch. The second phase consists of the iteration

where f' is the nonlinear mapping of Ey. (3.20), and 6 is again given by

<!-- formula-not-decoded -->

To guard against subsequent changes in policy, which induce corresponding changes in the matrix Que, one should ensure that the method is working properly, for example, by recomputing d if the policy changes andor the error 1EJ - Al is not reduced at a satisfactory rate. This method is generally olfective because the value itoration method typically linds an optimal policy much before it finds the optimal cost vector.

It should be mentioned, however, that the rank-one correction method is ineffective if there is little or no separation between the dominant and the subdominant cigenvalue moduli, both because the convergence rate of the method for obtaining d is slow, and also because the convergence rate of the modified itoration J:= AID is not much faster than the one of the regular iteration := 1J. for such problems, one should try cortections over subspaces of dimension larger than one (see (Ber93), and the adaptive aggregation and mmltiple-rank correction methods given in Section 1.3.3).

## Infinite State Space - Approximate Value Iteration

The valne iteration method is valid nader the assumptions of Prop. 2.1, so it is guaranteed to converge to * for problems with infinite state and control spaces. However, for such problems, the method may be inplementable only through approximations. In particular, givon a function J, one may only be able to calculate a fiction such that

<!-- formula-not-decoded -->

where e is a given positive scalar. A similar situation may occur oven when the state space is finite but the mumber of states is very large. Then instead of caloulating (T.)(x) for all states e, one may do so only for some states and estimate (T./)(e) for the remaining states d by some forn of interpolation, or by a least-squares ortor fit of (T.)Ce) with a lanetion from a suitable parametrie class (compare with the discussion of Section 2.3). Then the finctionI thus obtained will satisfs a relation such as (3.21).

Ne are thus led to consider the approximate value iteration mothod that generates a sequence told satistring

<!-- formula-not-decoded -->

starting from an arbitrary bounded function Jo. Generally, such a sequence "converges" to fe to within an error of c/(1 -). To see this, note that. 14. (3.22) yiolds

By applying T' to this relation, wo obtain

so by using Eg. (3.22) to write

we have

Proccoding similarly, wo obtain for all l 2 1,

By taking the limit superior and the limit. inferior as li - oo, and by using the fact. limax T% Jo = J*, wo see that.

It is also possible to obtain versions of the error bounds of Prop. 3.1 for the approximato value iteration method. We have from that proposition

<!-- formula-not-decoded -->

By using Ey. (3.22) in the abovo relation, we obtain

<!-- formula-not-decoded -->

Those bonds hold even when the state space is infinite because the bounds of Prop. 3. 1 can bo shown for an infinite state space as well. However. for these bounds to be useful. one should know r.

## 1.3.2 Policy Iteration

The policy iteration algorithm generates a sequence of stationary policies, each with improved cost. over the proceding one. Given the stationary policy y, and the corresponding cost function oy. an improved policy {i, Fi,.. is computed by minimization in the Dl' equation corresponding to Ju. that is. Tad, = Toy. and the process is ropeated.

Tho algoritin is based on the following proposition.

Proposition 3.4: Let a and fi be stationary policies such that TaJ, = TJ,, or equivalently, for i = 1, ....

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

Furthernore, if ye is not optimal, strict incquality holds in the above cquation for at least one state i.

Proof: Since J, = T,J, (Cor. 2.2.1) and, by hypothesis. Til, = T.J,. wr have for every i.

<!-- formula-not-decoded -->

Applying repeatedly '14 on both sides of this inequality and using the monotonicity of 77 (Lemma 1.1) and Cor. 2.1.1, we obtain

proving Eq. (13.233).

Trid, wil in hey a her prey implying that J, = J* by Prop. 2.2. Thus a must be optimal. I follous that if ye is not optimal, then düli) &lt; o,(i) for some state i. Q.E.D.

## Policy Iteration Algorithm

Step 1: (Initialization) Guess an initial stationary policy yo.

Step 2: (Policy Evaluation) Given the stationary policy ah, compute the corresponding cost function J, from the linear system of equations

<!-- formula-not-decoded -->

Step 3: (Policy Improvement) Obtain a new stationary policy uh+s satisfying

<!-- formula-not-decoded -->

If at = I'J,k stop; else return to Stop 2 and repeat the process.

Since the collection of all stationary policies is finito (by the finiteness of Sand C) and an improved policy is generated at every iteration, it follows that the algorithm will find an optimal stationary policy in a finite number of iterations. This property is the main advantage of policy iteration over value iteration, which in general converges in in infinite number of iterations. On the other hand, finding the exact value of t in Step 2 of the algorithm requires solving the system of linear equations (I-al,) Jd = Ok. The dimension of this system is equal to the nunber of states, and thus when this number is very large, the method is not attractive.

Figure 1.3.4 provides a geometric interpretation of policy iteration and compares it. with value iteration.

We note that. in some cases, one can exploit the special structure of the problem at hand to accelerate policy iteration. For example, sometimes we can show that, if fe belongs to some restricted subset. Af of admissible control functions, then d, has a form guaranteeing that To will also belong to the subset. M. In this case, policy iteration will be confined within the subset Al, if the initial policy belongs to M. Ruthermore, the policy evaluation stop may be facilitated. For an example, see Exercise 1.14.

We now demonstrato policy iteration by means of the examble considered carlier in this section.

## Example 3.1 (continued)

Let us go through the calculations of the policy iteration method: Initialization: We solect the initial stationary policy

<!-- formula-not-decoded -->

Figure 1.3.4 Gcometric interpretation of policy iteration and value iteration. Each stationary policy pe delines the linear function au terPad of the voctor . and T.J is the piecewise linear function minglen told. The optimal rost fo satisfies J* = T.J". so it is obtained from the intersoction of the graph of T. and the 45 decree line shown. The value iteration secuence is indicated in the top figure by the staircase construction, which asymptotically leads to do. The polies iteration sequence terminates when the correct linear segmont of the graph of Til (i.e., the optimal stationary policy) is identified, as shown in the bottom ligne.

<!-- image -->

Policy Evaluation: We obtain do through the equation ," = I,nd,, or. equivalontly, the linear systom of equations

<!-- formula-not-decoded -->

Substituting the data of the problem, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Solving this system of linear equations for d,"(1) and d"(2), we obtain

<!-- formula-not-decoded -->

Policy Improvement: We now find p'(D) and @'(2) satistying 1,10, = T.,". We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The minimizing controls are

<!-- formula-not-decoded -->

Policy Evaluation: We obtain as through the equation = I, "1:

<!-- formula-not-decoded -->

Substitution of the data of the problem and solution of the system of equations yields

<!-- formula-not-decoded -->

Policy Improvement: We perform the minimization required to lind to,:

<!-- formula-not-decoded -->

Hence we have J"s = Ta"s, which implies that y' is optimal and on =d

<!-- formula-not-decoded -->

## Modified Policy Iteration

When the mumber of states is large, solving the linear systom (1 aP,) t=!, in the policy evaluation stop by direct methods such as Gaussian climination can be prohibitively time consuming. One war to get around this difficults is to solve the linear sestem iteratively by using value iteration. In fact, we may consider solving the system on approximately by exceuting a limited mumber of value iterations. This is called the modified policy deration algorithm.

To formalize this method, let do be an arbitrary a-dimensional vector. Let 10.11.... be positive integers, and let the vectors a.de.. and the stationary policies 40.4.... be defined by

Thus, a stationary policy pe is defined from th according to Talk = Thi and the cost. le is approximatoly evnated by me - 1 additional value iterations. violding the vector deed. which is used in turn to define pite. We have the following:

Proposition 3.5: Let foy and (k) be the sequences generated by the modified policy itoration algoritlon. Then toky converges to .f. Furthermore, there exists an integer ti such that for all ki ≥ Ti, pet is optimal.

Proof: Let r be a scalar such that the vector Jo, defined by To = Jo + re, satisties TOo ≤ Jo. Any scalar e such that, max, (TJo) (i) - Jo(i)] ≤ (1 0)e has this property: Doline for all li Dend TWO. Then, i can be seen by induction that for all hi and m =0, 1..., Mi, the vectors Th and The ditter by the multiple of the amit veetor ra't tie it'e. ll. follows that if do is replaced by To as the starting vector in the algorithm. the same sequence of policies feel will be obtained; that is, we have for all li

Now we will show that for all h we have Th STk7o. Indeed, we have TuTo = TJo &lt; Jo, from which we obtain

<!-- formula-not-decoded -->

so that.

This argument. can be contimed to show that for all hi, we have Th&lt; They, so that

On the other hand, since 10o ≤ Do, We have do ≤ Do, and it follows that application of any number of mappings of the form 1, to Jo produces fictions that are bonded from belor he fo. Thus.

<!-- formula-not-decoded -->

Bo taking the limit as l x. we obtain lime ex Dali add (i) for all i. md side la colad

Since the mber of stationars policies in finite. there exists an to 0 suct that il a stationare polier er saristies

then fe is optimal. Now lot l be such that for all to k we have

Then from Eg. (3.10) we see that for all hi &gt; he the stationary policy ph that satisties land = T. is optimal. Q.E.D.

Note that if mh = 1 for all k in the modified policy iteration algorithm. we obtain the value iteration method, while if ma = so we obtain the policy iteration mothed. where the policy evaluation slep is performed iteratively by means of value iteration. Analysis and computational experience suggest. that it is usually best to take me larger than I according to some hemistic scheme. A key idea here is that a value iteration involving a single policy (evaluating Tud for some fe and J) is much less expensive than an iteration involving all policies (evaluating T for some d), when the mumber of controls available at each state is large. Note that error bounds such is the ones of Prop. 3.1 can be used to improve the approximation process. Furthermore, Gauss-Seidol iterations can be used in place of the usual value iterations.

## Infinite State Space - Approximate Policy Iteration

The policy iteration method can be defined for problems with infinite state and control spaces by means of the rolation

<!-- formula-not-decoded -->

The proof of Prop. 3.4 can then be used to show that the generated so quence of policies fly is improving in the sease that Just Sand for all ki. However, for infinito state space problems, the policy evaluation slep and/or the policy improvement step of the method may be implementable only through approximations. A similar situation may occur even when the state space is finite but the mamber of states is very large.

We are thus led to consider an approximate policy iteration mothod that generates a sequence of stationary policies fold and a corresponding sequence of approximate cost functions te satisfying

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where d and e are some positive scalars. and yo is an arbitrary stationary poliev. We call this the approtimate policy iteration algorithm. The following proposition provides error bounds for this algorithm.

Proposition 3.6: The sequence deky generated by the approximate policy iteration algorithm satisfies

<!-- formula-not-decoded -->

Proof: From Egs. (3.24) and (3.25), we have for all hi

where a = (1,1,..., 1)' is the mit. vector, while from Ey. (3.24), wo have for all li

By combining these two rolations, we obtain for all li

From Ba. (3.27) and the equation Tah, = dad, We have

By subtracting from this relation the equation Titlet = its, we obtain

<!-- formula-not-decoded -->

which can bo written as where this the fimetion given by

Let.

Then we have Mo) &amp; Sa for all ecS. and lin. (3.28) yioks

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let.

From By. (3.27) and the relation

which follows from Prop. 2.1, we have

We also have

and by subtracting the last two relations, we obtain

From this rolation we soo that

<!-- formula-not-decoded -->

By taking the limit superior as &amp; → oc and by using Do. (3.29), we obtain

<!-- formula-not-decoded -->

This relation simplifies to which was to be proved. Q.E.D.

Proposition 3.6 suggests that the approximate policy iteration method makes steads progress up to a point and then the iterates our oscillate within a nrichborhood of the optimum oft. This behavior appears to be typical in practice. Note that for &amp; = 0 and 6 = 0, Prop. 3.6 shows that. the cost sequence {,) generated by the (exact) policy iteration algorithm converges to Do, excn when the stale space is infinile.

<!-- formula-not-decoded -->

## 1.3.3 Adaptive Aggregation

Lot us now consider ant alternative to vlue iteration for performing approximate evaluation of a stationary policy y, that is, for solving approximately the system

<!-- formula-not-decoded -->

This alternative is recommended for problems where convergence of value iteration, even with error bounds, is very slow. The idea here is to solve instead of the system on = Too,, another system of smaller dimension, which is obtained by lumping together the states of the original system into subsets Sy, S2,..., Sm, that can be viewed as aggregate states. These subsets are disjoint and cover the entire state space, that is,

<!-- formula-not-decoded -->

Consider the a X m matrix l' whose ith column has unit entries at coor‹linates corrosponding to states in S, and all other entries conal to 40ro. Consider also an mX " matrix @ such that the ith row of Q is a probability distribution (40,., 4m) with 4a = 0 ifs &amp; Si. The structure of Q implies two useful properties:

- (a) QI = 1.
- (b) The matrix

is ab m X m transition probability matrix. In particular, the ijth component of R is equal to

<!-- formula-not-decoded -->

transition probability matrix l definos a Markov chain, called the aggregate Markon chan, whose states are the m aggregate states. Figuro 1.3.5 illustrates the aggrogate Markov chain.

Aggregate Markor chains are most useful when their transition behavior captures the broad attributes of the behavior of the original chain. This is generally true if the states of each aggregate state are "similar" in somo sense. Let us describe one such situation. In partientar, suppose that. wo have an estimato of o, and that me postulate that over the slates s of coery aggregate slate Si the variation Ju(s) - Als) is constunt. This amounts to hypothesizing that for some m-dimensional vector y we have

<!-- formula-not-decoded -->

Figure 1.3.5 Illustration of the aggregate Markow chain. In this example, the aggregate statos are Si = {1,2,37, 52 = 44,53, and Sy = {6}. The matrix 11 as columns (1, 1, 1,0,0, 0%, (0,0,0, 1, 1, 0), and (0,0, 0, 0, 0, 1)'. In this example, the matrix @ is chosen so that each of its rows dofines a miform probability distribution over the states of the corresponding aggregate state. Thus the rows of Q are (1/3, 1/3, 1/3,0,0, 0), (0. 0, 0, 1/2, 1/2, 0), and (0, 0, 0,0, 0, 1). 'The aggrogate Markov chain has transition probabilities ria = §(pze + p2s), ri2 = 4(pra+pso). Ma = 0,121 = 4(012 + 803), 122 = 4015, 223 = 1p16. 091 = 0. 152 = Poo. and 733 = 0.

<!-- image -->

By combining the equations Tad = gu tal,d and gu = (1-al,)d. no have

<!-- formula-not-decoded -->

This is the variational form of the equation J, = 'Ted, discussed carlier in connection with error bounds in Section 1.3.1, and can be used equally well for evaluating J,. Let us multiply both sides with @ and use the cquation J, - J = 11y. We obtain

which, by using the equations QW = L and R = Ql,W. is written as

<!-- formula-not-decoded -->

This equation can be solved for y. since R. is a transition probability matrix and therefore the matrix / - oR is invertible. Also, by applying T, to both sides of the cquation J, = 1 + 1Vy, we obtain

We thus conclude that, if the variation of (s) - f(x) is roughly constant over the states s of cach aggregate state, then the vector T,ItoP, Ily is a good approximation for Jy. Starting with . this approximation is oblained as follows.

## Aggregation Iteration

Step 1: Compute T,J.

Stop 2: Dolineate the aggregate states (ie., define WV) and specify the matrix Q

Step 3: Solve for y the systom

<!-- formula-not-decoded -->

where R. = QP,W, and approximate J, using

<!-- formula-not-decoded -->

Note that the aggregation iteration (3.31) can be equivalently written as

<!-- formula-not-decoded -->

so it differs from a value iteration in that it operates with T, on f-+ ly rather than ..

Solving the system (3.30) in the aggrogation iteration has an interesting interprotation. I can bo seen thaty is the o-discounted cost, vector corresponding to the transition probability matrix R and the cost-per-stage vector Q(T,, -J). Thus, calculating y can be viewed as a policy evaluation stop for the aggregate Markor chain when the cost per stage for each aggrogato stato S, is equal to

<!-- formula-not-decoded -->

which is the average 7,.J - I over the aggregate state Si according to the distribution fools E Sil. A hoy attractivo aspoot of the aggregation iteration is that the dimension of the system (3.30) is m (the number of aggregato states), which can be mach smaller than a (the dimension of the system o, = Tudy arising in the policy evaluation step of policy iteration).

## Delineating the Aggregate States

A key issuo is how to identify the aggregate states Si,..., Sm in a. way that the error do -d is of similar magnitude in each one. One way to do this is to view Tid as an approximation to of and to group together statos i with comparable magnitudes of (T,J)(i) - Ji). Thus the interval

is divided into mn segments and membership of a state i in an aggregate state is determined by the sogment within which (V,.)(1) ·i) lies. Br this we mean that for each state i noset i E soil (Tad)(i) -di) =c. and we set

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

This choice is based on the conjecture that, at least near convergener. (TuX)(i) - Ji) will be of comparable magnitude for states i for which J,(i) - Ji) is of comparable maguitude. Analysis and experimentation given in (BeC89) has shown that the preceding scheme often works well with a small nubor of aggregates states m (say 3 to 6), although the properties of the method are yet fully understood.

Note that the aggregate states can change from one iteration to the next, so the aggregation scheme "adapts" do the progress of the computation. The criterion used to delineate the aggregate states does not. exploit any special problem structure. In some cases, however, it is possible to take advantage of existing special structure and modify accordingly the mothod used to form the aggregato states.

## Adaptive Aggregation Methods

It is possible to construct a number of mothods that calculate J, by using aggregation iterations. One possibility is simple to perform a se quence of aggregation iterations using the preceding method to partition the state space into a low, say 3 to 10, aggregato states. This method can be greatly improved by interleaving cach aggrogation iteration with multiple value iterations (applications of the mapping T, on the current iterate). This is recommended based on experimentation and analysis given in (BeC89), to which we refer for further discussion. An interesting cmpirically observed phenomenon is that the error between the itorate and Ju is often increased by au aggregation itoration, but thea musually large improvements are made during the nest few value iterations. This suggests that the munber of value iterations following an aggregation iteration should be based on algorithnic progress; that. is, in aggrogation iteration should be performed when the progress of the value iterations becomes rolatively small. Some experimentation may be needed with a given problem to determine an appropriate criterion for switching from the value iterations to an aggregation iteration.

There is no proof of convorgonce of the scheme just described. On the basis of computational experimentation, it appears celiable in practice. Its convergence nonetheless can be guaranteed by introducing a feature that

enforces some irreversiblo progress via the value iteration method following an aggregation iteration. In particular, one may calculate the error bounds of Prop. 3.1 at the valne iteration Step 1, and impose a requirement that the subsequent aggrogation iteration is skipped if these error bounds do not, improve by a certain factor over the bounds computed prior to the proceding aggregation iteration.

To illustrate the oflectiveness of the adaptive aggregation mothod, consider the three deterministic problems described carlier (of. Fig. 1.3.3), and the performance of the method with two, three, and four aggregale states, starting from the sero function. The results, given in Fig. 1.3.0, should be compared with those of Fig. 1.3.3.

It is intuitively clear that the performance of the aggregation method should improve as the mumber of aggregato states increases, and indood tho computational results bear this out. The two extreme cases where m. =n and m= 1 are of interest. When m a n, cach aggregate state has a single stato and we obtain the policy iteration algorithm. When m = 1, thore is only one aggregate state, W is equal to the mmit vector e = (1,. 1)', and a straightforward calculation shows that for tho choice @ = (1/1. ., 1/11), tho solation of the aggregate system (3.30) is

<!-- formula-not-decoded -->

From this equation (using also the fact P,o = c), we obtain the iteration

<!-- formula-not-decoded -->

which is the same as the rank-one correction formula (3.12) obtained in Soction 1.3.1 and amounts to shifting the result Tad of value itcration within the error bound range given by Prop. 3.1. Thus we may view the aggregation scheme as a contimmun of algorithmas with policy iteration and value iteration (coupled with the error bouds of Prop. 3.1) included as the two extrome special cases.

## Adaptive Multiple-Rank Corrections

One may observo that the aggregation iteration

<!-- formula-not-decoded -->

amounts to applying T, to a correction of of along the subspace spaned ho the columns of 0. Once the matrix If is computed based on the adaytive procedure discussed above, wo may consider choosing the vector y in alternative ways. An interesting possibility, which leads to a generalization

Figure 1.3.6 Number of iterations of adaptive aggregation methods with two, three, and four aggregate states to solve the probions of Fig. 1.8.. lach tow of Q was choson to define a uniform probability distribution over the states of tho corresponding aggregate state.

| No. of aggregate states   | Pr. 1 .9   | Pr. 1 * = 99   | Pr. 2 r=.9   | Pr. 2 a = .99   |   Pr. 3 r = .!) |   Pr. 3 |
|---------------------------|------------|----------------|--------------|-----------------|-----------------|---------|
| 2                         | 11         | 13             |              |                 |              83 |     505 |
| 3                         |            | 1              | 3            | 3               |              61 |     367 |
|                           |            |                | 3            | :3              |              26 |     351 |

of the rank-one correction method of the proceding subsoction. is to select. y so that

<!-- formula-not-decoded -->

is minimized. By sotting to zero the gradient with respect to y of the above expression, we can verify that the optimal vector is given by

where Z = (I - oP,)W. The corresponding iteration then bocomos

Much of our discussion regarding the rank-one correction method also applies to this generalized version. In particular, we can use a two-phase implementation, which allows a return from phaso too to phase one whenever the progress of phase to is unsatisfactory. Parthermore, a version of the method that works in the case of multiple policies is possible.

## 1.3.4 Linear Programming

Since limy ex TN. = J* for all f (ef. Prop. 2.1), we have

<!-- formula-not-decoded -->

Thus J* is the "lugest" f that satisfies the constraint o ETJ. This constraint can be written as a finite system of linear inequalities

<!-- formula-not-decoded -->

and dolineates a polyhedron in De". The optimal cost veetor fe is the "northeast" comer of this polyhedron. as illustrated in Fig. 1.3.7. In pirticular. *(1)....*(a) solve the following problem (in 1.....A,):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where 5 is amy nonempty subset of the state space S = (1,..,). This is a linear program with a variables and as many as 1 X 4 constraints, where « is the maximum number of clements of the sots U(i). As a increases. its solution becomes more complex. For vory large a and 4. the linear programming approach can be practical only with the use of special largescale lincar programming methods.

Figure 1.3.7 Linear programming problem associated with the discounted infinite horizon problem. The constraint set, is shaded and the obiective to maximize is 1(1) + 1(2).

<!-- image -->

## Example 3.1 (continued)

For the example considered carlier in this section, the linear programming problem takes the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Cost, Approximation Based on Lincar Programming

When the mumber of states is vory large or infinite. we mas consider finding an approximation to the optimal cost function. which can be used in turn to obtain a (suboptimal) police by minimization in Bolman's equation. One possibility is to approximate e(a) with the lincar form

<!-- formula-not-decoded -->

where * = (ro...,rm) is a vector of pammotors, and for each stato t, We(e) are some fixed and known scalars. This amonts to approximating the cost function J*(r) by a linear combination of m given functions uk(e). where l = 1,...,m. These functions play the role of a basis for the space of cost function approximations Jc.v) that can be gonerated with diflerent choices of r (soo also the discussion of approximations in Section 2.3.3).

It is then possible to determine e by using de,e) in place of in the preceding linear programming approach. In particular. we compute as the solution of the program

<!-- formula-not-decoded -->

where&amp; is either the state space Sor a suitably chosen finite subset of s. and Ü(r) is either U(e) or a suitably chosen finite subset of C(r). Because (2, 7) is linear in the parumeter voctor r, the above program is linear in the parameters M,..., "m. Thus if m. is small, the number of variables of the liver prooman is small. The number of constraints is as large ass.g, where's is the number of elements of S'and a is the maximmn number of olomouts of the sets O(e). However, linear programs with a small mumber of variables and a large mmber of constraints can often be solved relativoly quickly with the use of special large-scale linear programming methods known as cutting plane or column generation methods (see o.g. (DanG3), (Ber'95al). Thus, the preceding linear programming approach may be practical even lor problems with a very large mmber of states.

- -1

## Approximate Policy Evaluation Using Linear Programming

In the case of a very large or infinite state space, it is also possible to use linear programming to evaluate approximately the cost function ., of a stationary policy ye in the context of the approximate policy iteration scheme discussed in Section 1.3.2. Suppose that we wish to approximate J, by a timotion J,r) of a given form, which is paramoterized by the vector ·= (0,..,Im). The bound of Prop. 3.6 suggests that we should try to determine the parameter vector o so as to minimize

From the error bounds given just prior to Prop. 3.1, il can also be sech that we have

<!-- formula-not-decoded -->

This motivates choosing r by solving the problem

where &amp; is either the state space S'or a suitably chosen finite subset of S. The preceding problem is equivalent to minimize

<!-- formula-not-decoded -->

When Aer) has the linear form (3.34), this is a linear program in the variables a and no..,rm.

## 1.4 THE ROLE OF CONTRACTION MAPPINGS

Tie kes structural properties in Dl models are responsible for most. of the mathomatieal resulls one can prose abont them. The first is the monotonicil! property of the mappings l' and 1 (of. Lemma 1. I in Section 1.1). This property is fondamental for total cost intimito horizon problems. P'or example, it forns the basis for the results on positive and negative DP models to be shown in Chapter 3.

When the cost per stage is bounded and there is discounting, however, we have another proporty that strongthens the ollects of monotonicity: the mappings T and T, are contraction mappings. In this section, we explain the meaning and implications of this property. The material in this section is conceptually very important, since contraction mappings are present. in several additional DP models. However, the main result of this section (Prop. 4.1) will not be used explicitly in any of the proofs givon later in this book.

Let B(5) denote the sot of all bounded real-valued timotions on S. With every function of : S'+ IN that belongs to B(S), we associate the

<!-- formula-not-decoded -->

As an aid for the advanced roader, we mention that the function | -I may be shown to be a norm on the linear space B(S), and with this norm BS) becomes a complete normed linear space (Lue69] The following definition and proposition are specializations to B(S) of a more general notion and result, that apply to such a space (see, og., references (LiS61) and (Lac69)).

Definition 4.1: A mapping H : B(S) - B(S) is said to be a contraction mapping if there exists a scalar p &lt; 1 such that

where ||·| is the norm of Eq. (4.1). It is said to be an m-stage contraction mapping if there exists a positive integer m and some p &lt; 1 such that

<!-- formula-not-decoded -->

where Il'' denotes the composition of H with itself me times.

The main result concerning contraction mappings is the following. For a proof, see references (LiSG1] and (Luc(9).

Proposition 4.1: (Contraction Mapping Fixed-Point Theorem) If H : B(S) + B(S) is a contraction mapping or an m-stage contraction mapping, then there exists a mique lised poind of 1l; that is, there exists a unique function f* € B(S) such that

<!-- formula-not-decoded -->

Furthermore, if .f is any function in B(S) and Ilk is the composition of Il with itself ki times, then

Now consider the mappings T and Ta defined by Exs. (1.1) and (1.5). Proposition 2.1 and Cor. 2.4.1 show that 'T and T, are contraction mappings (p=0). As a result, the convergence of the value iteration mothod to the unique fixed point of 1' follows directly from the contraction mapping theorem. Note also that, by Prop. 3.2, the mapping l corresponding lo the Ganss-Seidel variant of the value iteration method is also a contraction mapping with p = a, and the convergence result of Prop. 3.2 is again a spocial case of the contraction mapping theorem.

## 1.5 STOCHASTIC SCHEDULING AND THE MULTIARMED

## BANDIT

In the problem of this section there are a projects (or activities) of which only one can be worked on at any time period. Each project i is characterized at time hi by its state c. If projoct i is worked on at time hi, one receives an expected reward ok Ri(irt), where a € (0. 1) is a discount factor; the state te then evolves according to the equation

<!-- formula-not-decoded -->

where w' is a random disturbance with probability distribution depending on 8 but not. on prior disturbances. The states of all idle projects are

<!-- formula-not-decoded -->

We assume perfect state information and that the reward functions Ri(.) are uniformly bounded above and below. so the problem comes under the discounted cost framework of Section 1.2.

We assume also that. at any time &amp; there is the option of permanently retiring from all projects, in which case a reward aid is received and no additional rewards are obtained in the future. The retirement reward M is given and provides a parameterization of the problem, which will prove very useful. Note that for A/ sufficiently small it is never optimal to retire, thereby allowing the possibility of modeling problems where retirement is not a real option.

mallected; that. is,

The key characteristic of the problem is the independence of the projects manifested in our three basic assumptious:

1. States of idlo projects remain lixed.
2. Rewards reccived depend only on the state of the project currently cugaged.
3. Only ouc project can be worked on at a time.

The rich structure implied by these assumptions makes possiblo a powerful methodology. It turns out that optimal policies have the form of an indez rule; that is, for cach project i, there is a funetion mi(e) such that an optimal policy at time l is to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus m'(r) may be viewed as an index of profitability of operating the ith project. while A/ represents profitability of retirement at time h. The optimal policy is to exercise the option of masimo profitability.

The problem of this section is known as a multiarmed bandit problem. Au analogy here is drawn between project schoduling and selecting a sequence of plays on a slot machine that has several arms corresponding to diflerent but unknown probability distributions of payofi. With cach play the distribution of the solected arm is better identilied, so the tradeotl here is between playing arms with high expected payoll and exploring the winning potential of other arms.

## Index of a Project

Let Ja. 11) denote the optimal reward attainable when the initial state is .&amp; = (r',...,d") and the rotirement reward is Al. From Section 1.2 we know that, for cach 1, (, N7) is the unique bounded solution of Bellman's equation

<!-- formula-not-decoded -->

where L' is defined by

The next proposition gives some useful properties of .

Proposition 5.1: Let B = max; max, R'(ci)|. For fixed i, the optimal reward finction fc, A) has the following properties its il function of Al:

- (a) Jit, N) is convex and monotonically nondecreasing.
- (b) J.c, M) is constant for M ≤ -B/(1 - a).
- (c) •/(e, N) = N/ for all A ≥ B/(1 - a).

Proof: Consider the value iteration method starting with the function

Successive iterates are generated by

<!-- formula-not-decoded -->

and we know from Prop. 2.1 of Section 1.2 that.

<!-- formula-not-decoded -->

We show inductively that dad, AT) has the properties (a) to (e) stated in the proposition and, by taking the limit as hi → do, we establish the same properties for J. Clearly, Joe, A/) satisfies properties (a) to (e). Assume that. D(v, A/) satisfies (a) to (c). Then from Eys. (5.5) and (5.6) it follows that Ja+ (x, Af) is convex and monotonically nondecreasing in M, since the expectation and maximization operations preserve these properties. Verification of (b) and (e) is straightforward, and is left for the reader. Q.E.D.

Consider now a problem where there is only one project that can be worked on, say project, i. The optimal reward fuction for this problem is denoted D(z, M) and has the properties indicated in Prop. 5.1. A typica, form for Jir, A/), viewed as a funetion of M for fixed it. is shown in Fig. 1.5.1. Clearly, there is a mininal value mi(ri) of M for which D'(1", 11) = AJ; that, is,

<!-- formula-not-decoded -->

The function m'(ri) is called the inder function for simply index) of projeet. i. It provides an indifference threshold at each state; that is, m'(e) is the retirement reward for which we are indifferent between retiring and operating the project when at state t

Our objective is to show the optimality of the index rule (5.3) for the index function defined by Eg. (5.8).

<!-- image -->

## Project-by-Project Retirement Policies

Consider fust a problem with a single project, say project i, and a fixed retirement reward M. Then by the definition (58) of the index. an optimal policy is to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In other words, the project is operated continously up to the time that. its state falls into the etiremont set.

<!-- formula-not-decoded -->

At that time the project is permanently retired.

Consider now the multiproject problem for fixed retiremont roward M. Suppose at some time we are at state d = (ol,.,r"). Let us ask two questions:

1. Does it make souse to retire (from all projects) when there is still a project i with state d such that mery &gt; Al? The ansver is negative. Retiring when m'(r) &gt; A cannot, be optimal, since if we operate project i exclusively up to the time that its stated falls within the retirement set. S' of Eg. (5.10) and then retire, we will gain

- -1

a higher expected reward. (This follows from the definition (5.8) of the index and the nature of the optimal policy (59) for the singl-projoc problem.

2. Does it ever make sense to work on a projoct i with state in the retirement set Si of Eg. (5.10)? Intuitively, the answer is negative; it soems unlikoly that, a projedd mattractive chough to be retired if il were the only choice woukd become attractive moroly because of the availability of other projects that are independent in the souse assumed here.

No are led therofore to the conjecture that there is an optimal projectby-project retirement (PPR) policy that permanently retires projects in the sale way as if they were the only projoct available. Thus at cach time a PPR policy, when at state a = (al,..,d"),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where s' is the ith project retirement set. of By. (5.11). Note that. a PPR. policy decides about retirement of projects but does not specify the project to be worked on out of those not. yot retired.

The following proposition substantiates our conjecture. The proof is lengthy but quite simple.

Proposition 5.2: There exists an optimal PPR policy.

Proof: In view of Eys. (5.1), (5.5), and (5.11), existence of a P'PR. policy is equivalent to having, for all i.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where l is given by and /(c, Al) is the optimal roward function corresponding to e and Al.

<!-- formula-not-decoded -->

The ith single-project optimal reward function fi clearly satisfies, for all r,

<!-- formula-not-decoded -->

since having the option of working at projects other than i cannot decrease the optimal reward. Farthermore, from the definition of the retirement, set S' lef. Ex. (5.10)),

<!-- formula-not-decoded -->

Using Ers. (5.13) to (5.15), we obtain Eg. (5.126).

11. will sullice to show 1g. (5.12a) for i I. Denot:

i -0.02. ...e"): The state of all projecs other than project 1.

Jc. N): The optimal reward function for the problem resulting after project 1 is permanontly retired.

Jd.2M): The optimal reward function for the problem involving all projects and corresponding to state f = (r).

We will show the following inequality for all t (al,d):

<!-- formula-not-decoded -->

lu words this expresses the intuitively clear fact that at state (dog) one would be happy to retire project 1 permanently if one gots in roturn the maximum reward that can be obtained from project. 1 in excess of the retiremont reward M. We claim that to show Eg. (5.12a) for i = 1. it will suffice to show Ey. (5.16). ludeed. when t e S', then d'(i. 11) = M1. so from Eg. (5.16) we obtain Jc,2, 11) = 2(2. 12), which is in turn equivalont to Eg. (5.12a) for i = 1.

No now turn to the proof of Ba. (5.16). As left side is evident. To show the right side, we proceed by induction on the value iteration recursions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The initial conditions for the recursions (5.17) are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We know that 1(2, 2) → 1(0. 2. M). E(2) → 12. 1). and 1(11) J'(2), A/), so to show by. (6.16) it will suttice to show that for all l and X = (x).2) we have

<!-- formula-not-decoded -->

where, for all i # 1 and 2 = (12....,"),

In view of the definitions (5.19), we see that Eq. (5.20) holds for ki = 0. Assmmo that it. hokas for some f. We will show that it holds for kI. From Pa. (5.17) and the induction hypothesis (5.20), we have

<!-- formula-not-decoded -->

Using the facts 1(2) ≥ M and D(0) 2 1 lef. By. (5.17)), and the preceding equation, we see that,

whore

Using Bas. (5.176), (5.176), and the preceding equations, we see that

It can be seen from Eys. (5.17) and (5.19) that (e) ≤ Jit (el) and k(2) ≤ 1+(2) for all hi,d, and &amp;, so from Eq. (5.21) we obtain that By. (5.20) hoks for ki + 1. The induction is complete. Q.E.D.

As a first stop towards showing optinality of the index rule, we use the preceding proposition to derive an expression for the partial derivative of de. M) with respoot of Al.

Lemma 5.1: For fixed it, let Kar denote the retirement time under an optimal policy when the retirement reward is M. Then for all M for which 0J.e, M1)/OM/ exists we have

<!-- formula-not-decoded -->

Proof: Fix &amp; and M. Let 7* be an optimal policy and let And be the retirement. tine under pe. If ae is used for a problem with retirement. roward A to. we receive

Breward prior to retirement) + (N +o)E{any = 1e. M) +eEfaling.

The optimal reward Jd, M + c) when the retirement reward is A +e is no less than the procoding expression, so

Similarly, wo obtain

For € &gt; 0, these two relations viold

<!-- formula-not-decoded -->

The result follows by taking &amp; → 0. Q.E.D.

spect to Lobesgue measure (Roc70). Furthermore, it can be shown that. OJa, 11)/ON/ exists for all A/ for which the optimal policy is anique.

For a given M, initial state i, and optimal PPR policy, let 1; be the retirement time of project. i if it were the only project available, let I' be the retirement time for the multiproject problem. Both T, and T take values that are either nonnegative or oo. The existence of an optimal PPR. policy implies that we must have

<!-- formula-not-decoded -->

and in addition T,, i = I,...,n, are independent random variables. Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Optimality of the Index Rule

We are now ready to show our main result.

Proposition 5.3: The index rule (5.3) is an optimal stationary policy.

Using Lemma 5.1, we obtain

Proof: Fix = (el..,r"). donoto

and let i be such that.

<!-- formula-not-decoded -->

If m(r) &lt; Al the optimality of the index rule (5.3a) at state &amp; follows from the existence of an optimal PPR policy. 161(r) &gt; M, wr note that

and then use this relation together with Eq. (5.22) to write

<!-- formula-not-decoded -->

and finally

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

(The interchange of diflorentiation and expectation can be justified for almost all Al; see (Ber73al.) By the existence of an optimal PPR policy, we also have

Therefore, the convox functions f(v, A/) and D'(r, 11, D) viewed as finetions of Af for fixed &amp; are egual for A = m(r) and have equal derivative for almost. all A/ &lt; m(r). 1 follows that for all Af &lt; m(r) we have

<!-- formula-not-decoded -->

This implies that the index rule (5.31) is optimal for all &amp; with m(r) ≥ M. Q.E.D.

## Deteriorating and Improving Cases

It is evident that great simplification results from the optimality of the index rule (5.3). since optimization of a multiproject problem has been reduced to n separate single-project optimization problems. Nonetheless. solution of each of these single-project problems can be complicated. Under certain circumstances. however, the situation simplifies.

Suppose that for all it. and n that can ocear with positive probability, we have either

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under Ey. (5.23) (or Eq. (5.21)] projects become more (less) profitable as they are worked on. We call these cases improving and deteriorating. respectively.

In the improving case the nature of the optimal policy is evident: either retire at the first period or else select a project with maximal index at the first period and continue engaging that project for all subsequent periods.

In the deteriorating case, note that Eq. (5.21) implies that if retirement is optimal when at state ri then it is also optimal at cach state fi(2", 21). Therefore, for all ' such that Af = m'(r) we have, for all wr

<!-- formula-not-decoded -->

From Bellman's equation

we obtain or

<!-- formula-not-decoded -->

Thus the optimal poliey in the deteriorating case is retire if M&gt; masK and otherwise engage the project i with retarine one-step reward Ri(r").

Or

<!-- formula-not-decoded -->

--=

## Example 5.1 (Treasure Hunting)

Consider a scarch problem involving N sites. Each site i may contain a treasure with expected value o. A search at site i costs o, and reveals the treasuro with probability 6, (assuming a trcasure is there). Let 1, be the probability that thero is a treasure at site i. We take P, as the state of the project corresponding to searching site i. Then the corresponding one-step reward is

<!-- formula-not-decoded -->

If a search at site i does not reveal the troasure, the probability P, drops to

<!-- formula-not-decoded -->

as can be vorified using Bayes' rule. If the search tinds the treasure, the probability P, drops to sero, since the treasure is romoved from the site. Based on this and the fact that I(P) is increasing with P, (ol. By. (5.26)J, it is seen that the deteriorating condition (5.24) holds. Therefore, it is optimal to scarch the site i for which the expression R'(P) of ls. (5.26) is maximal, provided max, R'(P) &gt; 0, and to rotire it K'(P.) ≤ 0 for all i.

## 1.6 NOTES, SOURCES, AND EXERCISES

Many authors have contributed to the analysis of the discounted problem with bounded cost per stage, most notably Shapley (Sha53), Bollto mcasurability concerns are analyzed extensivoly in (BeS78], (DyY79], and (Her89).

The error bounds given in Section 1.3 and Exercise 1.9 are improvements on results of (Mc@66) (seo [Por71), (Por75). (Ber76), and (Pol78]). The corresponding convergence rate was discussed in Mor71] and MoWV77). The Gauss-Seidel method for discounted problems was proposed in (Kis71) (sco also (Has68)). An extensive discussion of the convergence aspects of the method and related background is given in Section 2.0 of (BeT80a). The material on the generie rank-one correction, including the convergence rank correction method where the ollect of several cigenvalues is nullified. Value iteration is partientarly well-suited for parallel computation; see eg.

Policy iteration for discounted problems was proposed in [Bel57). The modified policy itoration algorithm was suggested and analyzed in (PuS78)

analysis of Exercise 1.8, is now; see (Ber03), which also describes a multiple(ANT93), (BeT8941.

and [Pu582). The approximate policy iteration analysis and the convergence proof of policy iteration for an infinite state space (Prop, 36) are new and were developed in collaboration with o. Isitsiklis. The colation between policy iteration and Nowton's method (Exercise 1.10) was pointed

The material on adaptive aggregation is due to (BoC89). In am alternative aggregation approach (SPK89), the aggrogato states are fixed. Changing adaptively the aggregate states from one iteration to the nest depending on the progress of the computation has a potentially signiticant effect on the efficiener of the computation for difficult problems where the out in (PoA69) and was further discussed in (Pul378}.

ordinary value iteration method is very slow.

A complexity analysis of finite-state infinite horizon problems is given in (Pa'187). Discretization methods that approximate infinite state space systems with finite-state Markov chains. are discussed in (Ber75), (Fox71), (HaL86], (Whi78), (Whi79), and (WhisOa). For related multigrid approximation methods and associated complexity analysis, see (ChT89] and [ChT91]. A different approach to deal with infinite state spaces, which is based on randomization. has been introduced in (Rus9.1); see also (Rus95]. Rather

The linear programming approach of Section 1.3.4 was proposed in (D'Ep60). 'There is a relation between policy iteration and the simples method applied to solving the linear program associated with the discounted problem. In particular, it. can be shown that the simplex method for linear programming with a block pivoting rule is mathematically equivalent to the policy iteration algorithm. There are also duality connectious that relate the linear programming approach with randomised policies, constraints, and multiple criteria; see c.g.. (Kal83), (Put94. Approximation methods using basis functions and linear programming were proposed in (ScS85].

material on computational methods may be found in (Put 78].

The role of contraction mappings in disconted problems was first recognized and exploited in (Sha53), which considers two-player dynamic games. Abstract DP models and the implications of monotonicity and contraction have been explored in detail in (Den67), (Ber77). (BeS78), (VoP&amp;1). and (VeP87).

The index rule solution of the multiarmed bandit problem is due to (Git79] and [GiJ74). Subsequent contributions include (Whi80b}. (Kel81], [Whi81], and (Whi82]. The proof given here is due to [Tsi86). Ahternative proofs and analysis are given in (VWB85), (NTW89), (Tso91), (Web02), (BeN93), [Tsi93b, [BIT94al, and [BPT94b. Much additional work on the subject, is described in (Km085) and (KuV86).

Finally, we note that even though our analysis in this chapter recures a countable disturbance space, it may still serve as the starting point of analysis of problems with uncountable disturbance space. This can be done by reducing such problems to deterministic problems with state space a set of probability measures. The basic idea of this reduction is demonstrted

in Exerciso 1.83. The advanced reader may consult (130578) (Section 9.2), and ser how such a reduction can be ellected for a very broad class of finite and infinite horizon problems.

## EXERCISES

1.1

Write a computer problem and compute iteratively the vector d, satisfying

<!-- formula-not-decoded -->

Do your computations for all combinations of a = 0.9 and c = 0.999, and c= 0.5 and 6 = 0.001. Try value iteration with and without error bounds, and also adaptive aggregation with two aggregate classes of states. Discuss your results.

1.2

The purpose of this problem is to show that shortest path problems with a discount factor make little sense. Suppose that we have a graph with a nonnegative longth ai, for each arc (inj). The cost of a path (io, is,.., im) is El "'raged, where a is a discount factor from (0, 1). Consider thie problem of finding a path of minimum cost that connects two given nodes. Show that this problem need not have a solution.

1.3

Consider a problem similar to that of Section 1.1 except that when we are at state de, there is a probability is, where 0 &lt; 1 &lt; 1, Cat the next state ditt will be determined according lo rat = f(ra, Ma, wk) and a probability (1-(3) that the system will move to a termination state, where it stays permanently thercafter at no cost. Show that even if a = 1, the problem can be put into the discounted cost. framework.

Consider a problem similar do that of Section 1.2 except that the discont factor a depends on the current state ta. the control an, and the disturbance wa: that is. the cost function has the form

<!-- formula-not-decoded -->

where

with a(r..w) a given function satisfying

<!-- formula-not-decoded -->

Argue that the results and algorithns of Sections 1:2 and 1.3 have direct counterparts for such problems.

## 1.5 (Column Reduction (Por75])

The purpose of this problom is to provide a transformation of a certain topo of discounted problem into another discounted problem with smaller discount. factor. Consider the nestate discounted problem under the assumptions of Section 1.3. The cost per stage is g(i, a), the discount factor is d, and the transition probabilities aro po («). Kor cach j = 1,..,n, Iot.

<!-- formula-not-decoded -->

assuming that S"

- (a) Show that p, (a) are transition probabilities.
- (b) Consider the discounted problem with cost per stage gi, a), discount factor all=1' =, m,). and transition probabilities p., (4). Show that this problem has the same optimal policies as the original, and that its optimal cost vector l' satisfies

<!-- formula-not-decoded -->

where J* is the optimal cost vector of the original problem and e is the unit vector.

For all i, j, and 4, let

1.6

Let J: 5+D be any bounded function onS and consider the value itoration method of Section 1.3 with a starting function 1: 5, li of the form

where r is some scaler. Show that the bounds (1"./)(c) +S and (1D)(e) +Ck of Prop. 3.1 are independent of the scalare for all &amp; E.S. Show also that if S consists of a singlo stato &amp; (ic., 5 = fol), then

<!-- formula-not-decoded -->

## 1.7 (Jacobi Version of Value Iteration)

Consider the problem of Section 1.3 and the version of the value iteration method that starts with an arbitrary function o: So St and generates recursively F.J, 230,..., where D' is the mapping givon by

<!-- formula-not-decoded -->

Show that (PAD)(1) → '(i) as hi → no and provide a rate of convergence estimato that is at least as favorable as the one for the ordinary method (ef. Prop. 2.3).

## 1.8 (Convergence Properties of Rank-One Correction (Ber93])

Consider the solution of the system o = FJ, where F: I" - D" is the mapping

<!-- formula-not-decoded -->

11 is a givon voctor in 10", and Q is an a X a matrix. Consider the generic rank-one coroction iteration o:= AJ. where A/: 30" -&gt; y" is the mapping

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

- (a) Show that any solution f' of the system o = 1. satislies f* = 11.1*.
- (6) Verife that the value iteration method that uses the error bounds in the manner of By. (3.12) is a special case of the iteration . := M. with d catal to the unit voctor.
- (e) Assume that d is an eigenvertor of Q, let A De The corresponding rigonvalue, and let A.., do be the remaining rigonsales. Show That M.J can be written as

<!-- formula-not-decoded -->

where he is some vector in 99" and

Show also that Rd. = 0 and that for all l and of,

<!-- formula-not-decoded -->

- (d) Lot d be as in part (6), and suppose that en..,cued are digonvectors corresponding to Ar,..., do-1. Suppose that a vector I can be written as

where of is a solution of the system. Show that, for all li&gt; 1,

<!-- formula-not-decoded -->

so that if 1 is a dominant eigonvalue and Ay.., do-d lie within the unit circle, Ak0 convorges to * at a rate governed by the subdominant. eigenvalue. Note: This result can be goneralized for the case where Q does not have a fall set of linearly independent eigenectors, and for the case where f is modified through multiple-rank corrections (Ber9:3).

## 1.9 (Generalized Error Bounds (Ber76])

Let S be a set. aud B(5) be the set of all bounded real-valued timotions on S. Let T: B(S) + B(S) be a mapping with the following two properties:

- (1) T. = T.l' for all do d' E B(5) with l sol'.
- (2) For every scalar * # 0 and all a € .S.

<!-- formula-not-decoded -->

where ay, a2 are two scalars with O &lt; 0, &lt;00 &lt; 1.

- (a) Show that 'T is a contraction mapping on B(S), and hence for every J € B(S) wo have

<!-- formula-not-decoded -->

where J* is the unique fixed point of T in B(S).

- (b) Show that for all / € B(S), &amp; € S, and hi = 1,2,.

where for all ki

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A geometrie interpretation of these relations for the case where S' consists of a single element. is provided in Fig. 1.6.1.

- (r) Considor the following algorithm:

where .lo is amy function in 1(S), ye is any scalar in tho range (Sa,Tk),

- (d) Let J € M and consider the equation J = TJ, where

<!-- formula-not-decoded -->

and the vector hE N" and the matris A are given. Let s, be the ith row sum of A/, that. is.

.s, =

and lot. an = mine so. org = max, s,. Show that if the elements mi, of Al are all nonnegative and an &lt;1. Then the conclusions of parts (a) and (1) hold.

- (0) [Por75) Consider the Gauss-Seidel method for solving the system J = 4 taPJ. where 0 &lt; 0 &lt; 1 and P' is a transition probability matrix. Use part (d) to obtain suitable error bounds.

Figure 1.6.1 Craphical interpretation of the error bounds of lixereise 1.9.

<!-- image -->

## 1.10 (Policy Itcration and Newton's Method)

The purpose of this problem is to demonstrate a relation betwoon policy iteration and Nowton's mothod for solving nonlinear equations. Consider an equation of the form F(X) = 0, where KiD" D". Givon a vector a ED" Newton's method determines the by solving the linear system of equations

<!-- formula-not-decoded -->

where F(Ja)/d. is the Jacobian matrix of l ovaluated al .

- (a) Consider the discounted finite-state problem of Section 1.3 and define

<!-- formula-not-decoded -->

Show that if there is a unique yo such that.

<!-- formula-not-decoded -->

then the Jacobian matrix of Fat is

<!-- formula-not-decoded -->

where I is thon x 1 identity.

- (6) Show that the policy iteration algorithon can be identified with Newton's mothod for solving 1(.%) = 0 (assuming it gives a anique policy at cach step).

## 1.11 (Minimax Problems)

Provide analogs of the results and algorithms of Sections 1.2 and 1.3 for the minimax problem where the cost is

<!-- formula-not-decoded -->

is bounded, ta is gonerated by cant = fra.p(ra). uk), and W(r.u.) is a given nonempty subset of D for each (ra) E SX C". (Compare with Exercise 1.5 in Chaptor 1 of Vol. I.)

## 1.12 (Data Transformations (Sch72])

A finite-state problom where the discount. factor at cach stage deponds on the stato can be transformed into a problem with state independent. discount factors. 'To see this, consider the following set of equations in the variables .J (i):

<!-- formula-not-decoded -->

where we assume that for all i. a Cli). and i. mo, (4) ≥ 0 and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and define. for all and;.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Show that, for all 1 and j.

<!-- formula-not-decoded -->

and that a solution (8(0) |6 - 1...., 0} of 1a. (0.3) is also a solution of the equations

<!-- formula-not-decoded -->

## 1.13 (Stochastic to Deterministic Problem Transformation)

Under the assmmptions and notation of Soction 1.3. consider the controlled system

where pe is a probability distribution over &amp; viewed as a row vector, and lik is the transition probability matrix corresponding to the control function pa. The state is pe and the control is fe. Consider also the cost fimotion

<!-- formula-not-decoded -->

Show that the optimal cost and an optimal policy for the deterministic problem involving the abore systom and cost function yield the optimal cost and an optimal policy for the discounted cost problem of Section 1.3.

## 1.14 (Threshold Policies and Policy Iteration)

- (a) Consider the machine replacement example of Section 1.2, and assume that, the condition (2.10) holds. Let us doline a threshold policy to be a stationary policy that replaces if and onk if the state is greater than of equal to some fixed state i. Suppose that we start the policy iteration algorithm using a threshold policy. Show that all the subsequontly gonerated policies will be threshold policies, so that the algorithm will terminate aftor at. most. n iterations.
- (b) Prove the result of part (a) for the assol solling example of Section 1.2. assuming that there is a finite munber of values that the offer or can Lako. Hero, a threshold policy is a stationary policy that solls the assel if the offer is higher than a certain lixed munbor.

## 1.15 (Distributed Asynchronous DP [Ber82a), (BeT89a])

The value itoration mothod is woll suited for distributed (or parallel) computation since the iteration

<!-- formula-not-decoded -->

can be executed in parallel for all states i. C'onsider the finite-state discounted problem of Section 1.3, and assume that the above iteration is executed asynchronously at. a difforont. processor i for each state i. By this we mean that. the ith processor holds de vector of and updates the ith component of that. voctor at arbotrary times with an iteration of the form

<!-- formula-not-decoded -->

and at arbitrary times transmits the results of the latest computation to other processors m. who then update of (i) according to

Assone that all processors never stop computing and transmitting the results of their computation to the other processors. Show that the estimates of of the optimal cost function available at cach processor i at time t converge to the optimal solution finction o* as 1 - 00. Hint: Let Jand J be two functions such that J&lt; TD and T. &lt;7. and suppose that for all initial estimates fi of the processors, we have 1S 0, 50. Show that the estimates f; of the processors at time A salisty de ETCorall 120, and TEST ST.J lord. sufficiently large.

## 1.16

Assume that wo have two gold mines, Anaconda and Bonanza, and a gokimining machine. Let to and des be the current amounts of gold in Anaconda and Bonanza, respoctively. When the machine is used in Anaconda for Bomanza), there is a probability po (or pe, respoctively) that rard (or ridi, respoctively) of the gold will be mined without damaging the machine, and a probability 1-po (or 1-pes, respectively) that the machine will be damaged bevond ropair and no gold will be mined. We assume that O &lt; ra &lt; 1 and 0.11.. 1.

- (a) Assumo Chat po = pe = p, Where O &lt; p &lt; 1. Find the mine selcetion polier that maximizes the expected amont of sold mined before the machine breaks down. Hunt: This problom can be viewed is a discounted multiarmed bandit problem with a discount. factor p.
- (b) Assumo That Pa &amp; 1 and Pes = 1. Argue that the optimal expected amont of gold mined has the form (rhes) = JA(ed) ten, where Introd is the optimal expected amount of gold mined if mining is restricted just to Anaconda. Show that there is no policy that attains the optimal amount, '(rid, rm).

## 1.17 (The Tax Problem (VWB85])

This problem is similar to the multiarmed bandit problem. The only dillerence is that, if we engage project. i at period hi, we pay a tax aff(e) for every other projects for a total of de Let Cle, instead of carming a rewind a'R'(r). The objetive is to find a project selection policy that minimizes the total tax paid. Show that the problem can be converted into a bandit problem with reward function for project. i cqual to

## 1.18 (The Restart Problem [KaV87])

The purpose of this problem is to show that the index of a project in the multiarmed bandit context can be calenlated by solving an associated infinite horizon disconned cost problem. In what follows we consider a single project with reward fimction R(r), a fixed initial state to. and the caleulation of the valuo of index m(to) for that state. Consider the problem where at state and time li there are two options: (1) Continue, which brings reward aR(ca) and moves the projoct to state ditt = flos.m), or (2) restart the project, which moves the state to do. brings reward of Roo), and moves the project to state tary = floo.@). Show that the optimal reward functions of this problom and of the bandit problem with A = m(co) are identical, and therefore the optimad coward for both problems when starting at do equals m(ro). Hint: Show that Bellman's oquation for both probloms takes the form

## Stochastic Shortest Path Problems

| Contents  2.1. Main Results  2.2. Computational Methods  2.2.1. Value Iteration  2.2.2. Policy Iteration  2.3. Simulation-Based Methods  2.3.1. Policy Evaluation by Monte-Carto Simulation  2.3.2. Q-Learning  2.3.3. Approximations .  2.3.4. Extensions to Discounted Problems  2.3.5. The Role of Parallel Computation  2.4. Notes, Sources, and Exerrises  1. 78  1). 87  P. 88  D. 91  D. 94  P. 95  • 1. 99  1). 101  1. 118  D). 120  p. 121   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

--1

In this chapter we consider a stochastic vorsion of the shortest path problem discussed in Chapter 2 of Vol. 1. An introductory analysis of this problem was given in Section 7.2 of Vol. I. The analysis of this chapter is more sophisticated and uses weaker assumptions. In particular, we make assumptions that, generalize those made for deterministic shortest path problems in Chapter 2 of Vol. I.

In this chapter we also discuss another major topic of this book. In particular, in Section 2.3 we develop simulation-based methods, possibly involving approximations, which are suitable for complex problems that involve a largo mumber of states and/or a lack of an explicit mathematical model. These methods are most economically developed in the contoxt of stochastic shortest path problems. They can then bo extended to discounted problems, and this is done in Section 2.3.1. Further extensions to average cost per stage problems are discussed in Soction 4.3.4.

## 2.1 MAIN RESULTS

Suppose that we have a graph with nodes 1,2,...,u,t, where t is a special state called the destination or the termination state. We can view the deterministic shortest path problem of Chapter 2 of Vol. I as follows: we want to choose for each node i #t, a successor node pi) so that (i. pi)) is an arc, and the path formed by a sequence of successor nodes starting at any nodej terminates at t and has the minimum sun of arc lengths over all paths that start at j and terminate at t.

The stochastie shortest path problem is a generalization whereby at each node i, we must select a probability distribution over all possible successor nodes j out of a given set of probability distributions pi (4) paramotorized by a control a E U(i). For a given selection of distributions and for a given origin node, the path traversed as well as its length are now random, but we wish that the path leads to the destination t with probability one and has minimum expected length. Note that if every feasiblo probability distribution assigas a probability of 1 to a single successor node, we obtain the deterministic shortest path problem.

Ve formulate this problem as the special case of the total cost infinite horizon problem where:

- (a) There is no discounting (r = 1). fairiend ie :
- (b) The state space is 5 = 11.2...1, 1} with transition probabilities denoted by

<!-- formula-not-decoded -->

Furthermore, the destination t is absorbing, that is, for all u € U(t),

<!-- formula-not-decoded -->

- (0) The control constraint.so 0(p) is a finito set for all i.
2. (d) A cost gi. a) is incurred when control a Eli) is selected. luthernore, the destination is cost-free; that is, g(1, 1) = 0 for all 4 E U(1).

Note that as in Section 1.3. we assume that the cost per stage does not depond on a. This amounts to using expected cost per stage in all calculations. Iu particular, if the cost of using a at state i and moving to slate jis gi.a.j), we use as cost per stage the expected cost

Wo are interested in problems where either reaching the destination is inevitablo or else there is an incentive to reach the destination in a finite expected maber of stages, so that the essence of the problem is to reach the destination with minimum expected cost. We will be more specific about this shortly.

Note that since the destination is a cost-free and absorbing state, the cost starting from &amp; is zero for every policy: Accordingly, for all cost. functions. we ignore the component that corresponds to t and define the mappings T and T, on functions J with components J (1)...., Jn). We will also view the functions Jas n-dimensional vectors. This

<!-- formula-not-decoded -->

As in Section 1.3, for any stationary policy 4, we use the compact notation

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

We can then write in vector notation

1 -1

In terms of this notation, the cost function of a policy o= from... can Do written as

where do demotes the sero vector. 'The cost finction of a stationary policy I can be writton as

<!-- formula-not-decoded -->

The stochastic shortest path problem was discussed in Soction 7.2 of Vol. I. under the assumption that. all policies load to the destination with probability 1, regardless of the initial state. In order to analyze the problem under weaker conditions. we introduce the notion of a proper policy.

Definition 1.1: A stationary policy ye is said to be proper if, when using this policy, there is positive probability that the destination will be reached after at most a stages, regardless of the initial state; that is, if

<!-- formula-not-decoded -->

A stationary policy that is not, proper is said to be improper.

With a little thought, it can bo seen that ye is proper if and only if in the Markov chain corresponding to p, each state i is connected to the destination with a path of positive probability transitions. Note from the definition (1.1) that.

<!-- formula-not-decoded -->

More generally for a proper police p. the probability of not reaching the destination alter l stages diminishes as ple/" regardless of the initial state: that is,

<!-- formula-not-decoded -->

Thus the destination will eventually be reached with probability one under a proper policy. Fathermore, the limit defining the associated total cost.

vector a will exist and be limite, sinee the expected cost incurred in the lith period is bounded in absolute valuo hy

<!-- formula-not-decoded -->

Note that under a proper polier: the cost structure is similar to the one for discounted problems, the main difference being that the effective discount. factor depends on the current state and stage. but builds up to at least py \_per " stages.

Throughout this section, we assume the following:

Assumption 1.1: There exists at least one proper policy.

Assumption 1.2: For every improper policy e, the correspouding cost. Ju(i) is do for at least one state i; that is, some component of the sum La-d Plig, diverges to d0 as N → 20.

In the case of a deterministic shortest path problem, Assumption 1.1 is satisfied if and only if every node is connected to the destination with a path. while Assumption 1.2 is satisfied if and only if each cycle that does not. contain the destination has positive longth. A simple condition that. implies Assumption 1.2 is that the cost gion) is strictly positive for all i # 1 and « € U(i). Another important case where Assumptions 1.1 and 1.2 are satisfied is when all policies are proper. that is. when tormination is inevitable under all stationary policies (this was assumed in Soction 7.2 of Vol. I). Actually: for this case, it is possible to show that mappings t' and T, are contraction mappings with respect to some norm not necessarily the maximum norm of Eg. (4.1) in Chapter 11: soo Section 1.3 of (BoTS9al. or (Ts090). As a result of this contraction property, the results shown for discounted problons can also be shown for stochastic shortest path problems where termination is inovitable under all stationary policies. T turns out, however, that similar results can be shown even when some improper policies exist: the results that we prove under Assumptions 1.1 and 1.2 are almost as strong as those for discounted problems with bounded cost per stage. La partiontar, rshow that:

- (a) The optimal cost vector is the unique solution of Bellman's cquation J* = T.J*.
- (1) The value iteration method converges to the optimed cot vertor l for an arbitrary starting vector.

- (0) A stationary policy a is optimal if and only il 7J* = T.1*.
2. (d) The policy iteration algorithm yields an optimal proper policy start.ing from an arbitrary proper policy.

The following proposition provides some basic proliminary results:

## Proposition 1.1:

- (a) For a proper policy ye, the associated cost vector J, satisfies

<!-- formula-not-decoded -->

for every vector J. Furthermore,

<!-- formula-not-decoded -->

and J, is the unique solution of this equation.

- (b) A stationary policy ye satisfying for some vector J,

<!-- formula-not-decoded -->

is proper.

Proof: (a) Using an induction argument, we have for all &amp; € 'i" and hi ≥ 1

<!-- formula-not-decoded -->

Equation (1.2) implies that for all I E D", we have

so that.

<!-- formula-not-decoded -->

where the limit above can be shown to exist using ly. (1.2).

Also we have by definition

and by taking the limit. as li → oo, we obtain

which is equivalent to , = 1joy.

Finally, to show uniqueness, note that if J = T,J, then we have

- (b) The hypothesis A ≥ TaD. the monotonicity of Ty, and ly. (1.5) imply

<!-- formula-not-decoded -->

Il' pe wore not proper. by Assmmption 1.2, some component. of the sumn in the right-hand side of the above relation would diverge to oo as li → ou, which is a contradiction. Q.E.D.

The following proposition is the main result of this section, and provides analogs to the main results for discounted cost problems (Props. 2.1-2.3 in Section 1.2).

## Proposition 1.2:

- (a) The optimal cost vector J* satisfics Bellman's equation

<!-- formula-not-decoded -->

Furthermore, J* is the unique solution of this equation.

- (b) We have

<!-- formula-not-decoded -->

for every vector ..

- (c) A stationary policy y is optimal if and ouly if

<!-- formula-not-decoded -->

Proof: (a). (b) No first show that. T has at most one fixed point. Indeed, if J and d'are two fixed points, then we select y and do such that = T. = T"J and .1 = T.V' = Tud'; this is possible becauso the control constraint. set is finite. By Prop. 11D), we have that fand dare proper. and Prop. 1.1(a) implies that J= J, and 1 = De. We have T = Tk1 STI for all hi ≥ 1, and by Prop. 11(a), we obtain A ≤ limk ex This = = 1'. Similarly, f'S J, showing that of = f' and that T has at most. one fixed point.

-.-1

We next show that. 'Y' has at least one fixed point. Let y be a proper police (there exists one by Assumption 1.1). Choose de such that.

Then we have J, = Tad, 2 Tody. By Prop. 1.1(b), d' is proper, and using the monotonicity of To and Prop. 1.1(a), we obtain

<!-- formula-not-decoded -->

Contiming in the same manner, we construct a sequence falf such that each ye is proper and

<!-- formula-not-decoded -->

Since the set. of proper policies is finite, some policy # must. be repeated within the sequence fak}, and by By. (1.7), we have

<!-- formula-not-decoded -->

Thus J, is a lixed point. of'T, and in view of the uniqueness property shown carlier, o, is the unique fixed point of T.

Next. we show that the unique fixed point of 1' is equal to the optimal cost voctor o", and that Thy → do for all o. The construction of the preceding paragraph provides a propor ye such that To, = J,. We will show that Glo → J, for all l and that , = J*. Lete = (1.1,. .. I). lot. · &gt; 0 be some scalar, and let. À be the voctor satisfying

<!-- formula-not-decoded -->

There is a mique such vector because the equation i = T, + de can be written as i =an toe + Pad, so i is the cost vector corresponding lo a for a replaced by am toe. Since &amp; is proper, by Prop. 11(a), A is unique. Furthermore, we have of So, which implies that

Using the monotonicity of T and the preceding rolation, we obtain

Hence. Th converges to some rector. and we have

The mapping T' can be scon to bo contimous, so we can interchange '1' with the limit in the preceding relation. thereby obtaining f I.T. By the uniqueness of the fixed point of T' shown carlier, we must have i . li is also seen that.

Thus, Tk(J, - de) is monotonically increasing and bounded above. As carlier, it follows that limina Ta(J, -de) = d,. For any D. we can find 8 &gt; 0) such that

By the monotonicity of T. we then have

and since ling x 14(0, --De) = link -x ThiD = do, it follows that.

To show that Ja=%. take auy policy a = 440,49,..f. Wo have

<!-- formula-not-decoded -->

where do is the zero vector. Taking the limsup of both sides as li - xi in the preceding inequality, we obtain

so ye is an optimal stationary policy aud , = J*.

(e) If ye is optimal, then o, = J* and, by Assumptions 1.1 and 1.2, y is proper, so by Prop. 11(4), T,J* = 7,0, = 1, = 1* = TJ*. Conversels, if J* = TJ* = T,J*, it follows from Prop. 1.1(b) that y is proper. and by using Prop. 1.1(a), we obtain f* = d,. Therefore, y is optimal. Q.E.D.

Tho results of Prop. 1.2 can also be proved (with minor changes) assuming. in place of Assumption 1.2, that gi.o) &gt; O lor all/ and u € U(i). and that there exists an optimal proper policy; see Exerciso 2.12.

## Compact Control Constraint Sets

It turns out that. the finitoness assumption on the control cons mint U(i) can be weakened. It is sufficient that, for each i, U(i) be a compact subset of a Endlidean space, and that pu(e) and gli,a) be continuous in a over Ci). for all sand j. Nader these compactness and continuits assumptions, and also Assumptions 1.1 and 1.2, Prop. 1.2 holds as stated. The proof is similar to the one given above. but is technically much more complex. It can be found in BeTolb.

--1

## Underlying Contractions

We mentioned in Section 1.1 that the strong results we derived for disconnted problems in Chapter 1 owe their validity to the contraction property of the mapping I. Despite the similarity of Prop. 1.2 with the corresponding discounted cost results of Section 1.2, under Assumptions 1.1 and 1.2, the mapping 'T of this section need not. be a contraction mapping with respect to any norm; see Exercise 2.13 for a counterexample. On the other hand there is an important special case where T' is a contraction mapping with respect to a weighted sup norm. In particular, it can be shown that if all stationary policies are proper, then there exist positive constants Money and some owith 059 &lt; 1, such that we have for all rectors and 2,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A proof of this fact is outlined in Exorciso 2.14.

## Pathologies of Stochastic Shortest Path Problems

We now give two examples that illustrate the sensitivity of our results to scemingly minor changes in our assumptions.

## Example 1.1 (The Blackmailer's Dilemma [Whi82])

This example shows that the assumption of a finite or compact control con- straint set cannot be easily relaxed. Here, there are two states, state I and the destination state 1. At stato 1, we can choose a control a with 0 &lt; 1 ≤ 1; wo then move to state fat no cost with probability di, and stay in state 1 at a cost -a with probabilite lan?. Note that every stationary policy is proper in the sense that it leads to the destination with probability one.

We may regard o as a demand made by a blackmailer, aud state 1 is the situation where the victim complies. State / is the situation where the victim refuses to yield to the blackmailer's demand. The problem then can be seen as one wheroby the blackmailer tries to maximize his total gain by balancing his desire for increased demands with keeping his victim compliant.

If controls were chosen from a finite subset of the interval (0, IJ, the The optimal cost would then be finite, and there would exist. an optimal stationary policy. It turns out, however, that without the finiteness restriction the optimal cost starting al sale lis -o and there criss no oplimal stationary policy. In- problem would come mader the framework of this section. deed, for and stationary policy ye with p(1) = a. wo have

<!-- formula-not-decoded -->

from which

<!-- formula-not-decoded -->

Therefore, min, J,(1) = -x and J"(1) = -x, but there is no stationary policy that achieves the optimal cost. Note also that this situation would not change if the constraint set were a € (0, 1 Cio.. 4= 0 were an allowable control), although in this case the stationary policy that applies d(1) = 0 is improper and its corresponding cost, voctor is coro. thas violating Assumption

An interesting fact about this problem is that there is an optimal nonstationary policy a for which Ja (1) = -x. This is the policy 7 = 11011...1 that applies pa (1) = 9/(k + 1) at time land state to where gis a scalar in the interval (0, 1/2). No leave the voritication of this tact, to the reader. What happons with the policy a is that the blackmailer requests diminishing However. the probability of the victim's refusal diminishes at a much faster rate over time, and as a result, the probability of the victim romaining compliant forever is strictly amounts over time. which nonetheless add to ar positive, leading to an infinite total expocted payoff to the blackmailer.

## Example 1.2 (Pure Stopping Problems)

This example illustrates why we need to assume that all improper policies have infinite cost for at. least some initial state (Assumption 1.2). Consider an optimal stopping problem where a state-dependent cost is incurred only when invoking a stopping action that drives the systom to the destination: all costs are zero prior to stopping. Eventual stopping is a requirement here. so to properly formulato such a stopping problem as a total cost infinito horizon problem, it is essential to make the stopping costs negative (by adding a negative constant. to all stopping costs if nocessary). providing an incentive to stop. We then come under the framework of this section but with Assumption 1.2 violated becauso the improper policy that never stops does not. yiold infinito cost, for any starting state. Unfortmatoly, this scomingly small relasation of our assumptions invalidates our results as shown by the example of Fig. 2.1.1. This example is in oflect a detormmistic shortest path problem involving a cycle with zoro longth. la particular, in the example there is a (nonoptimal) improper policy that violds finito cost. for all initial states (rather than infinito cost for somo initial state), and 7' has multiple fixed points.

## 2.2 COMPUTATIONAL METHODS

All the methods devoloped in comection with the discounted cost.

problem in Section 1.3. have stochastic shortest path analogs. For example, extension (ef. Section 1.3.4), since J* is the largest solution of the sostem of inoqualities d ≤ T.. In this section, we will discuss in more detail some

Transition diagram and costs under policy (и,н....)

<!-- image -->

Cost = 0

<!-- image -->

<!-- image -->

Cost = 0 Cost = 0

Transition diagram and costs under policy (i.p...)

Figure 2.1.1 lixamplo where Prop. 1.2 fails to hold when Assumption 1.2 is violated. There are two Stationary policies, f and p', with transition probabilities and costs as shown. The cquation d = 'T. is givon by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and is satisfied by any o of the form

<!-- formula-not-decoded -->

with a ≤ -1. lore the proper policy y is optimal and the corresponding optimal cost vector is

<!-- formula-not-decoded -->

The dillically is that the impropor poliey d' has tinite (soro) cost for all initial stalls.

of the major methods. and we will also focus on some stochastic shortest. path problems with special structure. To turns out that by exploiting this special structure, we can improve the convergence properties of some of the methods. For example, in deterministie shortest path problems, value iteration torminates finitoly (Section 2.1 of Vol. 1), whereas this does not. bappen for any significant class of discounted cost problems.

## 2.2.1 Value Itcration

As shoo Do Prop. 1:2(0), value iteration works for stochastic shortest path problems. Rathermore, several of the enhancements and variations of value iteration for disconted problems have stochastic shortest path analogs. In particular, there are error bonds similar to the ones of Prop. 3.1 in Section 1.3 (although not quite as powerful: see Section 7.2 of Vol. 1). It can also be shown that the Crauss-Seidel version of the method works and that its rate of convergonce is typically faster than that of the ordinary method (Exercise 2.6). Luthermore, the rank-one correction method describer in Scotion 1.3. 1 is straightforward and eflective, as long as there is some separation between the dominant and the subdominant rigonvalie moduli.

## Finite Termination of Value Iteration

Generally, the value iteration method requires an infinite mumber of iterations in stochastic shortest path problems. Hovover, under sporial circumstances. the method can terminato finitely. A prominent example is the case of a deterministie shortest path problem. but there are other more general circumstances where termination occurs. In particular, let us assume that the transition probabilly graph corresponding to some optimal statonary poleg pt is acycle. By this we mean that there are no cycles in the graph that has as nodes the states Mo.not, and has an are (i.j) for cach pair of states i and i such that poli)) &gt; 0. We assume in particular that there are no positivo solf-transition probabilities po (r* (i)) for i #1. but it turns out that under Assumptions 1.1 and 1.2, a stochas tic shortest path problem with such self-transitions can be converted into another stochastic shortest path problem where po(a) = O for all i # 1 and u € U(i). In particular, it can be shown (Exercise 28) that the modified stochastic shortest path problem that has costs

<!-- formula-not-decoded -->

in place of gi.@). and transition probabilities

<!-- formula-not-decoded -->

instead of p(a) is equivalent to the original in the sense that it has the same optimal costs and policies.

No claim that, under the preceding acyclicity assumption. the calur iteration method will girld J+ after at most n iterations when started from the vector I yiven by

<!-- formula-not-decoded -->

To show this, consider the sets of states So. Son. dotined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and let Se be the last of these sots that is nonemper. Then in vie of our acyclicity assumption. we have

<!-- formula-not-decoded -->

Let as show bor induction that, starting from the voctor of la. (2.1. the value iteration method will viok for had. lo...Ti.

<!-- formula-not-decoded -->

Indeed, this is so for hi = 0. Assume Cat (Y% 1)(1) = 1*(i) if i E UH=OS. Then, by the monotonicity of D', we have for all i,

<!-- formula-not-decoded -->

while we have by the induction hypothesis, the dolinition of the sels She and the optimality of fit,

<!-- formula-not-decoded -->

The last two rolations complete the induction.

Thus, we have shown that moder the aeyelicity assumption, it the

Aith itcration, the value iteration method, will set. to the optimal values the costs of states in the set. Sa. In particular, all optimal costs will bo found after k itorations.

## Consistently Improving Policies

Tho properties of value iteration can be further improved if there is an optimal policy d* nuder which from a givon state, we can only go to a state of lower cost; that is, for all i, we have

<!-- formula-not-decoded -->

Wo call such a policy consistently improving.

A caso whoro a consistently improving policy exists arises in deterministic shortest path problems when all the are lengths are positive. Another important, case arises in continnons-space shortest path problems; see ['T'sioBal and Excrriso 2.10.

The transition probability graph corresponding to a consistently imever, a stronger proporte can bo proved. As discussed in Chapter 2 of Vol.

proving policy is seen to be acyclic, so when such a policy exists, by the preceding discussion, the value iteration method torminates finitely. How1, for shortest path problems with positive are lengths, one can use Dijkstra's algorithm. This is the label correcting method, which removes from the OPEN list a node with minimon label at. each iteration and requires just one iteration per node. A similar property hods for stochastic shortest path problems if there is a consistently improving policy: il one removes from the OPEN list a state j with minimuo cost estimate JJ), the GaussSeidel version of the value iteration method requires just one iteration per state: soo Exerciso 2.11.

Por problems where a consistently improving policy exists, il is also appropriate to use straightforward adaptations of the label correcting shortest path methods discussed in Section 2.3.1 of Vol. I. In particular, one may approximate the policy of removing from the OPEN list. a minimum cost state by using the SLl and LLl strategies (sco (PBT95]).

## 2.2.2 Policy Iteration

The policy iteration algorithm is based on the construction used in the proof of Prop. 1.2 to show that. T has a fixed point. In the typical iteration. given a proper policy « and the corresponding cost voctor of one obtains a new proper policy T satisfying Tad, = Tol. It was shown in Eg. (1.6) that Jp Ed. A can be seen also that strict inequality dali) &lt; "Ci) holds for at least one state i, if y is nonoptimal; otherwise we would have J, = T., and by Prop. 1.2(c), ye would be optimal. Theretore, the new policy is strictly better if the current policy is nonoptimal. Since the mmber of proper policies is finite, the policy iteration algorithm terminates aftor a finite number of iterations with an optimal proper policy.

It is possible to execute approximatoly the policy evaluation stop of policy iteration, using a finite number of value iterations, is in the discounted case. Hero westart with some voctor do. Por all li, a stationary policy nk is defined from th according to Y' Mathi = loh, the cost. h is approximately evaluated by ma - 1 additional value iterations, yielding the vector Jat, which is used in turn to define akt. The proof of Prop. 3.5 in Section 1.3 can be essentially repeated to show that th → J*, assuming that the initial vector Jo satisfies Too S Jo. Unfortunately, the requiroment T.Jo ≤ Jo is essential for the convergence proof, unless all stationary policies are proper. in which case T' is a contraction mapping (of. Exercise 2.11).

As in Section 1.3.3, it, is possible to uso adaptivo aggregation in conjunction with approximate policy evaluation. However. it is important. that. the destination t forms by itself an aggregate state, which will play the role of the destination in the aggregate Markov chain.

## Approximate Policy Iteration

Let us consider an approximate policy iteration algorithm that generates a sequence of stationary policies fell and a corresponding sequence of approximate cost vectors l satistying

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where 8 and e are somo positive scalars, and go is some proper polier. One difficulty with such an algorithm is that. even if the current policy y is proper, the next policy pitt may not. be proper. lu this case, wo have Juk+ (i) = 00 for some i, and the method breaks down. Note, however, that for a sufficiently small r, Eq. (2.6) implies that Think = T... so by Prop. 1.1(b). akits will be proper. In any case, we will analyze the method

under the assumption that. all generated policies are proper. The following proposition parallels Prop. 3.6 in Soction 1.3. It provides an estimato of the difforence h - * in torms of the scalar

<!-- formula-not-decoded -->

Note that for every proper policy pe and state i, we have Plent 1110 = i,14 &lt; 1 by the delinition of a proper policy, and since the number of proper policios is finite, we have p &lt; 1.

Proposition 2.1: Assume that, the stationary policies yi generated by the approximate policy iteration algorithm are all proper. Then

<!-- formula-not-decoded -->

Proof: The proof is similar do the one of Prop. 3.6 in Section 1.3. We modify the argumonts in order to uso the relations T, (0 + ve) ≤ TO + ve and Par ≤ pe, which hold for all proper policies ye and positive scalars r. No uso leos. (2.5) and (2.6) to obtain for all li

<!-- formula-not-decoded -->

From By. (2.8) and the equation 'Tad, = one wo have

By subtracting from this relation the equation Tilt ht = Jht: We obtain

This relation can be written as

<!-- formula-not-decoded -->

where Pans is the transition probability matris corresponding to pitt. Lot.

Then Ey. (2.9) vields

By multiplying this relation with Pants and by ackling (e + 20)e. we obtain

<!-- formula-not-decoded -->

By repeating this process for a total of n- 1 times. we have

<!-- formula-not-decoded -->

Thus.

<!-- formula-not-decoded -->

Let ye be an optimal statiouary policy. From la. (2.8). we have

We also have

By subtracting the last two relations, and by using the definition of &amp; and Eq. (2.10). we obtain

<!-- formula-not-decoded -->

Let

Then Ey. (2.11) vields, for all li.

<!-- formula-not-decoded -->

By multipling this relation with fro and by adding (1-p+1)(&lt;+28)/(1p), we obtain

<!-- formula-not-decoded -->

By repeating this process for a total of a - 1 times, we have

By taking the limit superior as fi → 2o, we obtain

<!-- formula-not-decoded -->

which was to be proved. Q.E.D.

The error bound (2.7) uses the worst-case estimate of the number of stages required to reach &amp; with positive probability, which is 1. We can streugthen the error bound if we have a better estimate. In particular, for all m. &gt; 1, lot.

<!-- formula-not-decoded -->

and let i be the minimal m for which pu &lt; 1. Then the proof of Prop. 2.1 can be adapted to show that.

<!-- formula-not-decoded -->

## 2.3 SIMULATION-BASED METHODS

The computational methods described so far apply when there is a mathomatical model of the cost structure and the transition probabilities of the system. In many problems, howover, such a model is not available, but. instead, the system and cost structure can be simulated. By this we mean that the state space and the control space are known, and there is a computer program that simulates, for a given control e, the probabilistic transitions from any given state i to a successor state j according to the transition probabilities po (a), and also gonorates a corresponding transition cost gi, maj). A is then of courso possible to use repeated simulation to calculate (at least approximately) the transition probabilities of the system and the expected stage costs by averaging, and then to apply the mothods discussed earlier.

The methodology discussed in this section, however, is geared towards an alternative possibilits, which is much more attractive when one is faced with a large and complex system, and one contemplates approximations: rather than estimate explicitly the transition probabilities and costs, we

estimate the cost. function of a given policy by generating a number of simulated sestem trajectories and associated costs, and by using some form of "least-squares fit."

Within this context. there are a mumber of possible approximation techniques, which for the most part are patterned after the value and tho policy iteration methods. We focus first on exact methods where estimates of various cost motions aro maintained in a "look-up table" that contains one entry per stato. No later develop approximate mothods where cost. functions are maintained in a "compact" form; that is, they are represented by a fimction chosen from a parametric class. perhaps involving a feature extraction mapping or a neural network. We first. consider these methods for the stochastic shortest path problem, and wo later adapt them for the discounted cost problem in Section 2.3.4.

To make the notation botter suited for the simulation context. we make a slight change in the problem definition. In particular, instead of considering the expected cost gi.@) at state i under control a, we allow the costy to depend on the next state j. 'Thas our notation for the cost per stage is now g(i. n. j). All the results and the entire analysis of the proceding sections can be rowritten in terms of the new notation by replacing gi.)

## 2.3.1 Policy Evaluation by Monte-Carlo Simulation

Consider the stochastic shortest path problem of Soction 2.1. Suppose that we are given a fixed stationary policy de and we want to calculate by simulation the corresponding cost vector da. One possibility is of course to generate, starting from each i. many sample state trajectories and average the corresponding costs to obtain an approximation to ,(i). We can do this soparatoly for cach possible initial state, but a more officiont mothod is to use cach trajectory to obtain a cost sample for many states by considering the costs of the trajectory portions that start at these states. If a stale is encountered multiple times within the same trajectory, the corresponding cost samples can be treated as multiple independont samples for that state.t

To simplify notation. in what follows we do not show the dependence of various quantities on the given policy. Io particular. the transition probability from i to j, and the corresponding stage cost are denoted by de, and gi,j), in place of pu (ali)) and gin(a), d), respectively.

To formalize the process, suppose that. we perform an infinite mumber of simulation runs. each ending at the termination state 1. Assume also that within the total mber of nos, cach state is cocontered an infinite

+ The validity of doing so is not quite obvious because in the case of multiple visits to the same state within the same trajectory, the corresponding multiple cost samples are correlated. since portions of the corresponding cost sequence are

shared by these cost samples. For a justifying analysis, see Exercise 2.15.

- -1

mumber of times. Consider the mth time a given state i is encountered, and let (i, 1, 12o..,in,@) be the remainder of the corresponding trajectory. 10l di, m) be the coresponding cost. of reaching state 1.

<!-- formula-not-decoded -->

No assume that the simulations correctly average the desired quantities; that is, for all states i, we have

<!-- formula-not-decoded -->

Wo can iterativoly calendato the sums appearing in the abovo oquations by using the update formulas

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and the initial conditions are, for all i,

<!-- formula-not-decoded -->

The normal way to implement the preceding algorithm is to update the costs da(i) at the end of cad simdation run that generates the state trajectory (ir.i,..,in./), by using for cach ki = 1...., N. the formula

<!-- formula-not-decoded -->

where me is the number of visits thus far to state in and ime = 1/1h. There are also forms of the law of large mumbers, which allow the use of a different stopsize one in the above equation. It can be shown that for convorgence of iteration (3.2) to the correet cost value 1, (ia), it is sufticient that "me be diminishing at the rate of one over the number of visits to state

## Monte-Carlo Simulation Using Temporal Differences

An altornative (and essontially equivalent) method to implemont the Monte-Carlo simulation update (3.2), is to update A,(in) immediately after gir.de) and ie are generated, then update da(i) and J,(i2) immediatoly after gig, is) and is are gonerated, and so on. The method uses the quantitics

<!-- formula-not-decoded -->

with inty =t. which are called temporal differences. They represent the difference botween the current. estimate o, (in) of expected cost to go to the termination state and the predicted cost to go to the formination state.

<!-- formula-not-decoded -->

based on the simulated outcomo of the current stage. Given a sample state trajectory (ir.ke....Av,O). the cost update formula (32) can De rewritten in terms of the temporal differences de as follows to see this, just add the formulas below and use the fact of (in+1) = ,(1) = 0):

Following the stato transition (1.12). sot

Pollowing the state transition (12.13). sot.

Following the stato transition (in./), sol.

The stepsizes Ama A = 1.....N, are given by Yma = 1/mh. Where ms is the number of visits alreads made to state is. In the case where the sample trajectory involves at. most one visit to cach state, the preceding updates are equivalont to the update (3.2). If there are multiple visits to some state during a samplo trajectors: there is a ditlerence between the procoding updates and the updato (3.2), because the updates corresponding lo each visit, to the given state affect the updates corresponding to subsequent. visits to the same state. However, this is an eflect which is of second order in the stepsize e. so once a becomes small, the difference is negligible.

TD(1)

The procoding implomontation of the Monto Carlo simmlation method for evaluating the cost of a policy de is known as TD(1) (here TD stands for Tomporal Differences). A generalization of TD(1) is TD(1), where 1 is a paracter with

<!-- formula-not-decoded -->

Given a sample trajectory (ir,..., in, /) with a coresponding cost sequence (ir, 12)...gin,), TD(A) updates the cost. estimates da(a),..,J,(in)

using the tomporal differences

and the equations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

und more generally for k = 1...,N,

<!-- formula-not-decoded -->

The use of a value of 1 less than 1 tends to discount. the eflect of the tomporal diflerences of state transitions far into the fature on the cost estimate of the current state. In the case where 1 = 0, we obtain the TD(0) algorithm, which following the transition (in, its) updates J,(in) by

<!-- formula-not-decoded -->

This algorithm is a special case of an important stochastic iterative algo rithm known as the stochastic amrozimation (or Robbins-Monro) method (sco o.g., (BMP90): (BeT89a), (LjS83]) for solving Bellman's equatious

In this algorithm, the expected value above is approximated by using a

The stepsizes me need not be equal to 1/mk, where my is the number of visits thus far to state in, but they should diminish to zero with the mumber of visits to each state. For example one may use the same stepsize *m= 1/m for all states within the mth simulation trajectory. With such single samplo al, each iteration lof. 1y. (33.1).

a stepsize and under some technical conditions, chief of which is that cach state i = 1...., " is visited infinitely often in the course of the simulation, it can be shown that for all A e 10, 4, the cost estimates "Ci) generated

While TD(A) yields the correct values of J,(i) in the limit regardless of the value of 1, the choice of 1 may have a substantial eflect ou the rate of convergence. Some experience suggests that using 1 &lt; 1 (rather than 1 = 1); often reduces the number of sample trajectories needed to attain the same variance of error between (i) and its estimate. However, it.

by TD(1) converge to the correct. values with probability 1.

present there is no analysis relating to this phenomenon.

## Simulation-Based Policy Iteration

The policy evaluation procedures discussed above can be embedded within a simulation-based policy iteration approach. Let us introduce the notion of the Q-fuctor of a state-control pair (i, «) and a stationary policy M, defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is the expected cost corresponding to starting at state i, using control u at the first stage, and using the stationary policy ye at the second and subsequent stages.

The Q-factors can be evaluated by first evaluating of is above, and then using further simulation and averaging (if necessary) to compute the right-hand side of Eg. (3.5) for all pairs (i,u). Once this is done, one can execute a policy improvemont. step using the cquation

<!-- formula-not-decoded -->

We thus obtain a version of the policy iteration algorithm that combines policy evaluation using simulation, and policy improvement using Eg. (3.6) and further simulation, if necessary. In particular, givon a policy p and its associated cost vector Ju, the cost of the improved policy of is computed by simulation, with ji) determined using Eq. (3.6) on-line.

## 2.3.2 Q-Learning

We now introduce an alternative method for cases where there is no explicit model of the sistem and the cost structure. This method is analogous to value iteration aud has the advantage that it can be used directly in the case of multiple policies. Lastead of approximating the cost function of a particular policy, it updates directly the Q-factors associated with an optimal policy, thereby avoiding the multiple policy evaluation

stops of the policy iteration mothod. These Q-factors are defined, for all pairs (i, «) hy

From this dofinition and Bollman's equation, we see that the @-factors satisfy for all pairs (i,«),

<!-- formula-not-decoded -->

and it can be shown that the @-factors are the migne solution of the above system of cquations. The proof is essentially the same as the proof of existonce and uniqueness of solation of Bellman's equation: soo Prop. 1.2 of Section 2.1. In fact, by introducing a system whose states are the original states lo...,n./ together with all the pairs (i, «), the above system of equations can be seen to be a special caso of Bellman's cquation (sec Exorcise 2.17). Furthermore, the Q-actors can be obtained by the iteration

<!-- formula-not-decoded -->

whick is amalogons to value iteration. A more general version of this is

Where a is a stopsize parameter with a € (0. 1), that may change from one iteration to the next. The Q-learning method is an approximate version of this iteration, whereby the expected value is replaced by a single sample, i.c.,

Here jand gi. nog) are generated from the pair (i.a) by simulation, that is, according to the transition probabilities po, (a). Thus Q-learning can be viewed as a combination of value iteration and simdation.

Because Q-learning works using a single sample per iteration, it is well suited for a simulation context. By contrast, there is no single sample vorsion of the value iteration method, except in special cases [see Exercise 2.9(1)). The reason is that, while it is possible to use a single-sample approximation of a term of the torn lemin. such as the one appearing in the Q-factor equation (3.8), it is not possible to do so for a term of the form min|88-87, suck as the one appearing in Bellman's equation.

To guarantee the convergence of the Q-learning algorithm to the optimal Q-factors, all state-control pairs (i, a) must be visited infinitoly often, and the stopsire o should be chosen in some special way. to particular, it the iteration corresponds to the mth visit of the pair (i.«), one may use in the Q-learning iteration the stopsize th = o/m, where is a positive constant. We refer to (Tsi91) for a proof of convergence of Q-learning under very genoral conditions.

## 2.3.3 Approximations

Wo now consider approximation /suboptimal control schemes that ar suitable for problems with a large mumber of states. The discounted vorsions of these schemes, which are discussed in Section 2.3.4. can be adapted for the caso of an infinite state space. Generally there are two types of approximations to consider:

- (a) Approximation of tho optimal cost function of*. This is done by using a function that, given a stato i, produces an approximation fi.r) off*(i) where r is a parameter/weight vector that is typically dotermined by some form of optimization; for example, by using some type of least squares framework. Onee fior) is known, it can be used in rcal-time to gonerate a suboptimal control at any state i according to

<!-- formula-not-decoded -->

An alternative possibility, which does not require the real-time calculation of the expected value in the above formula, is to obtain approximations @(i.a.%) of the Q-factors Q(i. u), and then to generate a suboptimal control at any state i according to

<!-- formula-not-decoded -->

11 is also possible to use approximations in(i.") of the cost fimotions Je of policies ye in an approximate policy iteration scheme. Note that the cost approximation approach can be enhanced if we have additional information on the true fimetions d*(i), Q(i.@), or ,(i). For example, if we know that c*(i) &gt; O for all i, wo may first compute the approximation 16 a by us some mished, and de she approximation procedures of this section.

- (b) Approximation of a policy p. or the optimal policy 4*. Again this approximation will be done by some limotion parameterized by a

parameter/weight vector r, which given a state i, produces an approximation #(i,r) of f(i) or an approximation p* (i.r) of f*(i). The parameter/weight, vector o can be determined by some type of least squares optimization framcwork.

In this section we discuss several possibilities, emphasizing primarily the case of cost approximation. The choice of the structure of the approximating functions is vory significant. for the success of the approximation approach. One possibility is to use the linear form

<!-- formula-not-decoded -->

where r = (ri,n.., In) is the parameter vector and wni) are some fixed and known scalars. This amounts to approximating the cost. finction J* by a linear combination of m given basis functions (ne(1),..., 14(1)), where k: = 1,..., m.

## Example 3.1: (Polynomial Approximations)

An important example of linear cost. approximation is based on polynomial basis functions. Suppose that. the state consists of &amp; intoger compononts di,.., dig, cach taking values within some limited range of the nonnegative integers. For example, in a quencing system, &amp;n way represent the number of customers in the kith quene, where ki = 1,...,4. Suppose that we want to use an approximating function that is quadratic in the components te. Then we can deline a total of 1 + q+d basis functions that depend on the state

<!-- formula-not-decoded -->

An approximating function that is a lincar combination of these functions is given by

where the parameter vectore has components ro. ta. and this, with his = 1,..,g. In fact, amy kind of approximating function that is polynomial in the componouts too,d, can be constructed in this way.

## Example 3.2: (Feature Extraction)

Suppose that through intuition or analysis we can identify a number of characteristies of the state that affect the optimal cost function in a substantial was: We assume that these characteristies can be numerically quantified, and that they form a y-dimensional vector fi) = (fi(i),..., Ju(i)), called the feature vector of state i. For example, in computer chess (Section 6.3.2 of

Vol. 1) where the state is the current. board position, appropriato features are matorial balance, piece mobility, king safely, and other positional factors. Features, when well-choson. can capture the dommant nonlinearities of the optimal cost. function o", and can bo used to approximate f through the linear combination

<!-- formula-not-decoded -->

where ro.Mof, are appropriatoly chosen weights.

It, is also possible to combine foature extraction with more general polynomial approximations of the type discussed in lexcople 3.1. Por example, a feature extraction mapping f folloved by a quadratic polynomial mapping. yields an approximating tinction of the form

where the paramoter vector e has components ro. ta. and the. with has a 1.....4. This timotion can be vowed as a lincar cost. approximation that. uses the basis functions

<!-- formula-not-decoded -->

Note that more that, one state may map into the same feature vector. so that cach distinct value of feature voctor coresponds to a subset of states. 'This subset may be viewed as an aggregate state." The optimal cost fumetion J* is approximated by a function that is constant over cach aggregato state. We will discuss this viewpoint shortly.

It can be scen from the preceding examples that lincar approximating fictions of the form (3.9) are woll suited for a broad variets of situations. There are also interesting nonlinear approximating functions fi.r). including those dofined by neural networks, porhaps in combination with foature extraction mappings. Iu ow discussion, we will not address the choice of the structure of ji,r). but rather focus on various methods for obtaining a suitable parameter vector r. We will primarily discuss three approaches:

- (a) Feature-based aggregation. where o is determined as the cost vector of ant aggregato stochastic shortest path problem."
- (b) Minimizing the Bellman cquation cror. where y is determined so that the approximate cost function Ji.r) neady satisfies Bollman's equation.
- (c) Apporimate policy iteration, where the cost. linctions o, of the generated policies peace approximated by dadar), with e chosen according to a least-squares error criterion.

We note, however, that the methods described in this subsection are not. fully understood. We have chosen to present them because of their potential to deal with problems that are too comples to be handled in any other was.

- ..1

## Feature-Based Aggrogation

No mentioned carlier in Example 3.2 that. a feature extraction mayping divides the stato space into subsols. The states of cach subert are mapped into the same feature voctor, and are "similar" in that they "share thosamefeaturos?" With thiscontext. in mind. lot.thesot.of states fl....,n} of the given stochastic shortest path problom be partitioned in m disjoint subsets Sa, hi = I,..,m. We approximate the optimal cost J*i) by a function that, is constant over cach set She that is.

<!-- formula-not-decoded -->

whore e= (Pio...."'.)' is a voctor of paramotors and

<!-- formula-not-decoded -->

Equivalently: the approsimate cost function ((1.,)..... Jn.o))" is represented as Ir. where l is the ox m matrix whose enter in the ith row and lith column is exti). The ith row of l'may be viewed as the feature vector corresponding to state i (of. Examplo 3.2).

In the aggregation approach, the parameters te are obtained as the optimal costs of an aggregate stochastie shortest path problem" states are the subsets Sa. Thus re is chosen to be the optimal cost of the aggrogato state Sy in im aggregate problem, which is formulated similar to the aggregation mothod of Section 1.3.3. In particular, let. Q be an m X n matrix such that the kith row of @is a probability distribution (dil....,1,) with de, =0 it i &amp; 5w. As in Section 1.3.3, the structure of Q implies that for cach stationary policy ye, the matrix

<!-- formula-not-decoded -->

is an m X m transition probability matrix. The states of the aggregate stochastie shortest path problem are the sets Sio, Si, together with the termination stato /; the stationary policies select at aggregate state Sy a control u elli) for cach i e Sy and thas can be identilied with stationary policies of the original stochastic shortest path problom; finally the transition probability matrix coresponding to ye in the aggregate stochastic shortest path problem is Ra. Given a stationary policy pe, the state transition mochanism in the aggrogate stochastic shortest path problem can Do described as follows: al aggregate state Sa. we move to state i with probability the, then we move to state j with probability pe(i)), and finally, if j is not. the termination state 1, we move to the aggregate state S, corresponding lo i (if sp).

Suppose now that r= (moon' is the optimal cost fiction of the aggrogato stochastic shortest path problem. Then e solves the corresponding Bellman cquation, which has the form

<!-- formula-not-decoded -->

One way to obtain r is policy iteration based on Monte-Carlo simulation. as described in Section 2.3.1. An alternative, due to fTsl91, is to nsca sinulation-based form of value iteration for the aggregate problem. Here. at each iteration we choose a subset. Sh. we randomly select a state i &amp; Sh according to the probabilities tie, and we update ta according to

<!-- formula-not-decoded -->

whero a is a positivo stopsize that diminishos to sero as tho algorithm progresses. The following example illustrates the mothod. We refer to (TsV9.1 for experimental results relating to this example as well for convergonce analysis of the mothod.

## Example 3.3: (Tetris (TsV94])

Tetris is a popular video gamo played on a two dimonsional grid. Bach squaro in the grid can be tall of empts: making up a "wall of bricks" with "holes" and a "jagged top". The squares fill up as blocks of differont shapes fall at a constant rate from the top of the grid and are added to the top of the wall. As a given block falls, the player can move horizontally and rotato the block in all possible ways, subjoet to the constraints imposed by the sides of the grid and the top of the wall. There is a finite set of standard shapes for the falling blocks. The game starts with an empty grid and ends when a square in the top row becomes full and the top of the wall reaches the top of the grid. However, when a row of full squares is created, this row is removed, the bricks lying above this row move one row downward, and the player scores a point. The player's objective is to maximize the score attained (total number of rows removed) up to termination of the game.

Assuming that, for every policy, the gamo terminates with probability one (something that is not really known at prosent). we can model the problem of finding an optimal tetris playing strategy as a stochastic shortest path problem. The control. denoted by a, is the horizontal positioning and rotation applied to the falling block. The state consists of two components:

- (1) The board position, that is. a binary description of the full/empty status ol each square, denoted by t
- (2) 'The shapo of the coront falling block denoted by y.

The componenty is gonerated according to a probability distribution p(?), independently of the control. Exercise 2.9 shows that under these circumstances, it is possible to derive a reduced form of Bellman's Conation

involving a cost function of that deponds only on the component of the state (see also Exercise 1.22 of Vol. 1). This cquation has the intuitive form

<!-- formula-not-decoded -->

whero g(r.y,«) and f(x,!, «) are the number of points scored (rows removod), and the board position when the state is (ty) and control " is applied, respectively.

Unfortunately, the mmber of states is extremoly large. It is oqual to m2", where m is the manbor of diflerent shapes of falling blocks, and he and ' are the height and width of the grid, respectively. In partientar, for the rcasonable numbers m. = 7, l = 20, and w = 7 we have over 1012 states. Thus it is essential to use approximations.

An approximating function that involves feature extraction is particularly attractive here, since the quality of a given position can be described quite well by a few features that are casily recognizable by experienced playors. These features include the current height of the wall, and the prosence of "holes" and "glitches" (sevore irregularities) in the first fow rows. Suppose that, based on human experience and tial and orror, we obtain a mothed to may each board position d into a vector of features. Suppose that there is a finite number of possible feature vectors, say me and define if board position &amp; maps into the lith feature vector,

The approximating tinction Orr) is given by

<!-- formula-not-decoded -->

where r= (ri,..., Im) is the paramotor voctor. 'The simulation-based value ¡toration (3.10) takes the form

where the positive stepsize y diminishes with the maber of visits to position .:.

One way to implement the method is as follows: The game is simulated many times from start to finish, starting from a variety of "representative" board positions. At each iteration, we have the current board position 2 and we determine the foature doctor l to wlich &amp; maps. They we candomly gonerate a falling block y according to a known and lixed probabilistic mechanism, and we update te using the above iteration. Let a" be the choice of a that attains the maximum in the iteration.

<!-- formula-not-decoded -->

Then the board position subsequent to &amp; in the simulation is f(x,y, u."), and this position is used as the current state for the next iteration.

In the aggregate stochastic shortest path problem formulated above, policies consist. of a different control choice for each state. A somewhat different aggregate stochastic shortest path problem is obtained by requiring that, for each ki, the same control is used at all states of Sa. This control must be chosen from a suitable set U (k) of admissible controls for the states in Sk. The optimal cost function &amp; = (ri,....in) corresponding to this aggregate stochastic shortest path problem solves the following Bollman equation

<!-- formula-not-decoded -->

This equation can be solved by Q-learning, particularly when m is relativoly small and the mumber of controls in the sets Ü(k) is also small.

Noto also that the aggrogato problem need not. be solved exactly, but. can itself be solved approximatoly by any of the simulation-based methods to be discussed subsequently in this section. Iu this context, aggregation is used as a feature extraction mapping that maps cach state i to the corresponding feature vector a(i) = (w(i)....,un(i)). This feature vector becomes the input to some other approximating function (see Fig, 2.3.1).

Figure 2.3.1 View of a cost function approximation scheme that consists of a feature extraction mapping followed by an approximator. The scheme conceptually separates into an aggrogate system and a cost approximator for the aggregate system.

<!-- image -->

where each row of the n X m matrix I is a probability distribution. This

<!-- formula-not-decoded -->

where wn fi) is the (i, lth ontry of the matris I. and we have

<!-- formula-not-decoded -->

The transition probability matrix of the aggregato stochastic shortest path problem corresponding to ye is still Ra = QP"W, and we may use as parameter vector r the optimal cost vector of this aggregato problem.

## Approximation Based on Bellman's Equation

Another possibility for approximation of the optimal cost by a finction f(ir), where r is a vector of unknown parameters, is based on minimizing the error in Bellman's equation; for example by solving the problem

<!-- formula-not-decoded -->

whoreS is a suitably chosen subset of "representative" states. This minimization may be attempted by using somo type of gradient or GaussNewton method.

A gradient-like method that can be used to solve this problem is obtained by making a correction to &amp; that is proportional to the gradient of the squared error torm in Ey. (3.11). This method is given by

<!-- formula-not-decoded -->

where P denotes the gradient with respoct tor, D(ir) is the error in Bellman's equation, given by

<!-- formula-not-decoded -->

" is given by

<!-- formula-not-decoded -->

andy is a stopsize, which may change from one iteration to the next. The method should perform many such iterations at each of the representative states. 'Typically one should evole through the set of representative states § in some order, which may change (perhaps randomly) from one cycle 10 the nort.

by

Note that in iteration (3.12) we approximate the gradient of the term

<!-- formula-not-decoded -->

which can be shown to be correct only when the above minimum is attained at a unique i € U(i) (otherwise the function (3.13) is nondifferentiable with respect tor. Thus the convergence of iteration (3.12) should be analyzed using the theory of nondifferentiable optimization. One possibility to avoid this complication is to replace the nondifierentiable term (3.13) by a smooth approximation, which can be arbitrarily accurate (see (Ber82b), Ch. 3).

An interesting special case arises when we want to approximate the cost function of a given policy a by a function 1,(i.r), where e is a paramcler vector. The iteration (3.12) then takes the form

<!-- formula-not-decoded -->

where

and Eff li. denotes expected valne over using the transition probabilities p. (1(i)). There is a simpler version of iteration (3.10) that does not require averaging over the successor states j. In this version, the two expected values in iteration (3.14) are replaced by two independent single sample values. In particular, e is updated by

<!-- formula-not-decoded -->

where j and correspond to two independent transitions starting from i. It is necessary to nso two independently generated states jand 7 in oider that the expected value (overjand 3) of the product.

given i, is equal to the term

appearing in the right-hand side of by. (3.11).

There are also versions of the above iterations that update @-factor approximations rather than cost approximations. I particular. let us introduce an approximation @(i,a, 4) to the -factor Qi, a), where r is ill

unknown parameter voctor. Bellman's equation for the Q-factors is given by lef. Er. (3.7))

so in analogy with problem (3.11), we determine the parameter vector v by solving the least squares problom

where V is a suitably chosen subset of "representative" state-control pairs. The analog of the gradient-like methods (3.12) and (3.14) is given by

where de(i, ja) is given by

<!-- formula-not-decoded -->

it is obtained by

<!-- formula-not-decoded -->

and y is a stopsize parameter. In analogy with Ey. (3.15), the two-sample version of this iteration is given by

where jand are two states independently generated from i aceording to the transition probabilities corresponding to u, and

Note that there is no two-sample version of iteration (3.12), which is based on optimal cost approximation. This is the advantage of using Qfactor approximations rather than optimal cost approximations. The point is that it is possiblo do uso single-samplo or two-sample approximations in gradient-like methods for terms of the form E{minfl, such as the one appearing in Ey. (3.16), but not for terms of the form min Ef, such as the one appearing in Eg. (3.11). The following example illustrates the use of the two-sample approximation idea.

## Example 3.4: (Tetris Continued)

Consider the game of tetris described in Example 3.3. and suppose that an approximation of a given form Je,r) is desired, where the parameter vector * is obtained by solving the problem

<!-- formula-not-decoded -->

where § is a suitably chosen set of "representative" slates. Because this problem involves a term of the form E{maxl}, a two-sample gradient-like method is possible. It has the form

where y and I are two falling blocks that are randomly and independontly generated.

<!-- formula-not-decoded -->

and

Similar to Example 3.3, consider a feature-based approximating funetion Ha.r) given by

where r = (moo.rm) is tho parameter vector and

20x4) = 16 if board position maps into the kith feature vector, otherwise.

For this approximating function, the preceding two-sample gradient iteration takes the relativels simple form

Note that this iteration updates at most two parameters the ones corresponding to the feature voctors to which the board positions d and f(r.T.7) map. assuming that these feature vectors are dillerent. To implement the method, a sot § containing a large maber of states t is solected and at caches. two falling blocksy and y are independently generated. The controls a and n that are optimal for (ay) and (6,7), based on the current paramoter vector *, are calculated, and the parameters of the feature vectors associated with a and f(x, y,ü) are adjusted according to the preceding formula.

Figure 2.3.2 Block diagram of approximate policy iteration.

<!-- image -->

## Approximate Policy Iteration Using Moute-Carlo Simulation

No now discuss an approximate form of the policy iteration method, where we use approximations ),(ir) to the cost d, of stationary policies 1, and/or approximations @(i.a,z) to the corresponding Q-factors. The thooretical basis of the method was discussed in Section 2.2.2 (of. Prop. 2.1).

Similar to our carlier discussion on simulation, suppose that for a lised stationary police de, we have a subse of "represontativo" (perhaps chosen in the course of the simulation), and that for each i eS, wo have A/(i) samples of the cost. Jai). The mth such sample is donoted by cli. m). Then, we can introduce approximate costs i,(ir), where ris a paramotor/weight vector obtained by solving the following least-squares

<!-- formula-not-decoded -->

Once the optimal value of e has been determined, we can approximate the costs d"(i) of the policy a by dai,o). Then, we can evaluate approximate Q)-factors using the formula

<!-- formula-not-decoded -->

Figure 2.3.3 Structuro of approximate policy iteration algorithm.

<!-- image -->

and we can obtain an improved policy 7 using the formula

<!-- formula-not-decoded -->

We thus obtain an algorithm that alternates botween approximato policy evaluation steps and policy improvement stops, as illustrated in Fig. 2.3.2. The algorithm requires a single approximation per policy iteration, namoly the approximation ,(ir) associated with the current policy a. 'The parameter vector e determines the Q-factors via Ey. (3.17) and the next policy I via Er. (3.18).

For another vier of the approximate policy itoration algorithm, note that it consists of four modules (see Fig. 2.3.3):

- (a) Tho simulator, which given a state-decision pair (i, «), generates the next state j according to the correct transition probabilities.
- (b) The decision generator, which generates the decision pi) of the improved policy at the current state i led. Ey. (3.18)] for nse in the
- (c) The cost-lo-go approcimator, which is the function ,(ir) that is consulted by the decision generator for approximato cost-to-go values to use in the minimization of Eg. (3.18).
- (d) The oplimeer, which accopts as imput the sample trajectories produced by the simulator and solves the problem

<!-- formula-not-decoded -->

-...

to obtain the approximation Ap(iF) of the cost of fi.

Note that in very large problems, the policy Do canot be evaluated and stored in explicit form, and thus the optimization in By. (3.18) must be evaluated "on the fly" during the simulation. When this is the case, the parameter vector e associated with pe remains unchanged as we evaluate the cost of the improved policy to by genorating the simulation data and by solving the least squares problem (3.19).

One way to solve this latter problem is to use gradient-like methods. Given a sample state trajectory (is, iz.., iN, Y) generated using the policy Ti, which is defined by By. (3.18), the parameter vector 7 associated with IT is updated by

<!-- formula-not-decoded -->

where y is a stopsize. The summation in the right-hand sido above is a sample gradient corresponding to a term in the least squares summation of problem (3.19).

Nic linally mention two variants of the approximate policy iteration algorithm, both of which require additional approximations per poley iteration. In the first variant, instead of caleulating the approximate Q-factors via lie. (3.17), we form an approximation @Q,(i.n,g), where the parameter vector y is determined by solving the least squares problem

<!-- formula-not-decoded -->

where Vis a "reprosentative" set of state control pairs (i,u), and Q,(i, 1, r) is evaluated using Ex. (3.17) and either exact calculation or simulation. This variant is useful if it speeds up the calculations of the policy improvemont. stop lof. ly. (3.18)1.

In the second variant of the algorithm, we first perform the approximate policy evaluation stop to obtain (ir). Then we compute the improved policy Fi) by the formula (3.18) only for states i in a represoutative" subset. §. We then obtain an "improved" policy ali, d). which is defined over all states, by introducing a parameter vector o and by solving the least squares problem

<!-- formula-not-decoded -->

Here, we assume that the controls are elements of some Euclidean space and I I denotes the norm on that space. This approach accelerates the policy improvement step jef. Ed. (3.18)] at the expense of solving an additional least squares problem per policy iteration.

## Approximate Policy Iteration Using TD(1)

Just as there is a tomporal differences implementation of Monto Carlo simulation, there is also a temporal differences implementation of the gradient iteration (3.20). The temporal differences da are given by

and the iteration (3.20) can be alternatively written as follows just add the equations below using the temporal difference expression (3.23) to obtain the iteration (3.20)):

Following the stato transition (ir.12), set.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Following the state transition (in,t), set.

<!-- formula-not-decoded -->

The vector 7 may be updated at each transition. although the gradients V.itia,) are evaluated for the value of that previls at the time in is generated. Also, for convorgence, the stepsize y should diminish over time. A popular choice is to use during the mith trajectory y = c/m, where c is a constant.

A variant of this method that has been proposed under the name TD(1) uses a parameter 1 € (0, 1] in the formulas (3.23)-(3.26). It has the following form:

For A = 1..... N, following the state transition (in itt), set.

While this method has received wide attention, its validity has been questioned. Examples have been constructed (Ber95b| where the approximating function (ir) obtained in the limit ly TD(X) is an increasingl poor approximation to d,(1) as A decreases towards 0, and the approximation obtained by T'DO) is very poor. 16 is possible. however, to nse the two-sample gradient iteration (3.15) for a simdation-based, approximate evaluation of the cost finctions of varions policies in an approximate policy itcration scheme. This iteration resembles the TD(0) formula but aims at. minimizing the error in Bellman's equation.

Following the state transition (i2, is), sot.

## Optimistic Policy Iteration

In the approximate policy iteration approach discussed so far, the least squares problem that evaluates the cost of the improved policy fi must be solved completely for the vector t. An alternative is to solve this problem approximately and replace the policy o with the policy fi after a single or a few simulation runs. An extreme possibility is to replace y with I at the end of cach state transition, as in the next algorithm:

Following the state transition (in, in+1), set.

and goncrate the next transition (int 1, itz) by simulation using the control

<!-- formula-not-decoded -->

The theoretical convergence properties of this method have not been investigated so far, although its TD(A) version has been used with success in solving some challenging probloms (Tes)2).

## Variations Involving Multistage Lookahead

'To reduce the ofloct, of the approximation error

between the truo and approximate costs of a policy a, one can consider i lookahead of several stages in computing the improved polics 7. The method adopted carlier for generating the decisions pi) of the improved policy.

<!-- formula-not-decoded -->

corresponds to a single stage lookahead. At a given state i, it finds the optimal decision for a onc-stage problem with stage cost gi, ad) and terminal cost. (after the first stage) (j.r).

An m-stage lookahead version finds the optimal policy for an m-stage problem, whereby we start at the current state i. make the m subsequent decisions with perfect state information, inem the corresponding costs of the m stages, and pay a terminal cost i,(ja), where j is the state after m stages. This is a finite horizon stochastic optimal control problem that may be tractable, depending on the horizon mand the number of possible successor states from each state. If uni) is the first decision of the m-stage lookabcad optimal policy starting at state i, the improved policy is defined by

<!-- formula-not-decoded -->

Note that if luCir) is equal to the exact cost a(i) for all states i, that is, there is no approximation. the multistage version of policy iteration can be shown to terminate with an optimal policy under the same conditions as ordinary policy iteration (seo Exercise 2.16).

Multistage lookahead can also be used in the real-time caleulation of a suboptimal control policy, once an approximation Ji.r) of the optimal cost. has been obtained by any one of the methods of this subsection. An example is the computer chess programs discussed in Section 6.3 of Vol. 1. In that case, the approximation of the cost-to-go function (the position evaluator discussed in Section 6.3 of Vol. I) is relatively primitive. It is derived from the features of the position (material balance, piece mobility, king safoty, ote.). appropriatoly weighted with factors that me either houristically determined by trial and error, or (in the case of a champion program. IBNI's Doop Thought) by training on examples from grandmaster play. It is well-known that the quality of play of computer chess programs crucially depends on the size of the lookahead. This indicates that in many types of problems, the multistage lookahead vorsions of the mothods of this subsection should be much more eflective than their single stage lookahead counterparts. This improvement in performance must of course be weighted against the considerable increase in computation required to optimally solve the associated multistage problems.

## Approximation in Policy Space

We finally mention a conceptually different approximation possibility that aims at direct optimization over policies of a given type. Here we lypothesize a stationary policy of a certain structural form, say #(i,r), where r is a vector of unknown parameters/weights that is subject to optimization. We also assume that for a fixed r, the cost of starting at i and using the stationary policy #(or), call it Ji,*), can be evaluated by simulation. We may then minimize overr

<!-- formula-not-decoded -->

where the expectation is taken with respect to some probability distribution over the set of initial states. This minimization will trpically be carried out. by some method that does not require the use of gradients if the gradient. loes not require the of Ji,r) with respect to e camot be easily calculated. If the simulation can produce the value of the gradient P.fi.o) together with fir). thema gradient-based method can be used. Generally, the minimization of the cost

- -1

function (3:27) tends to be quite dillicult if the dimension of the parameter vector is large (say over 10). As a result, the method is most likely effective only when adequate optimal policy approximation is possible with very fow paramotors.

## 2.3.4 Extension to Discounted Problems

We now discuss adaptations of the simulation-based methods for the case of a discounted problem. Consider first the evaluation of policies by simulation. One difficulty here is that trajectories do not terminate, so we cannot, obtain sample costs corresponding to different states. One way to get around this dilliculty is to approximate a discounted cost. by a finite horizon cost. of sufficiently large horizon. Another possibility is to convert the e-discounted problem to an equivalent stochastic shortest path problem by introducing an artificial termination state t and a transition probability 1 - a from each state i # t to the termination state t. The remaining transition probabilities are scaled by multiplication with a (sce Vol. 1, Section 7.3). Bellman's equation for this stochastic shortest path problem is identical with Bellman's equation for the original a-discounted problem, so the optimal cost. functions and the optimal policies of the two problems are identical.

The preceding approaches may load to long simulation runs, involving many transitions. An alternative possibility that is useful in some cases is based on identifying a special state, called the reference state, that is as sumed to be reachable from all other states under the given policy. Suppose that such a state can be identified and for concreteness assume that it, is state 1. Thus, we assumo that the Markov chain corresponding to the given policy has a single recurrent class and state I belongs to that class (see Appendix D of Vol. 1). If there are multiple recurrent classes, the procedure described in what follows can be modified so that there is a reference state for cach class.

To simplity notation, we do not show the dopendence of various quantities on the given policy. In particular, the transition probability from i to j and the corresponding stage cost are denoted by po, and gli,j), in place of pr (n(i)) and gi,di),d), respectivels: For cach initial state i, let C(i) donote the average disconted cost incurred up to reaching the reference state 1. Let also me denote the first passage time from state i to state 1. that is, the number of transitions required to reach state 1 starting from state i. Note that in, is a random variable. We denote

<!-- formula-not-decoded -->

By dividing the cost. di) into the portion up to reaching state 1 and the remaining portion starting from state I. we have

<!-- formula-not-decoded -->

Applying this equation for i = 1, we have 1"(1) = C(1) + D(1).,(1), so that.

<!-- formula-not-decoded -->

Combining Egs. (3.28) and (3.29), we obtain

<!-- formula-not-decoded -->

Therofore, do calculate the cost, vector on. it is sullicient to cakedlate the costs C(i), and in addition to calculato the expected discount torms D(i). Both of these can be computed. similar to the stochastic shortest path problem, by generating many sample system trajectories, and averaging the corresponding sample costs and discount terms up to reaching the reference

Note here that because C(1) and D(1) crucially affect the caleulated values J,(i), it may be worth doing some extra simulations starting from the roference state I to onsuro that C(1) and D(1) are accurately calculated.

Once a simulation method is available to ovaluate (perhaps approximately) the cost of various policies. it cau be embedded within a (perhaps approximate) policy iteration algorithm along the lines discussed for the stochastic shortest path problem.

We note also that there is a straightforward extension of the (learning algoritlan to disconnted problems. The optimal Q-factors are the unique solution of the cqnation

This is again proved by introducing a system whose states aro pairs (i.«). so that the above system of equations becomes a special case of Bellman's

-. -- equation. With similar obsorvations, it follows that the vootor of Q-factors can be obtained by the value iteration

<!-- formula-not-decoded -->

The Q-learning mothod is an approximate version of this iteration, whereby the expected value is replaced by a singlo sample, ic..

Here j and gi, a.j) are generated from the pair (i.«) by simulation, that, is, according to the transition probabilities po, («).

We finally note that approsimation based on minimization of the error in Bellman's equation can also be used in the case of a discounted cost. One simply needs to introduce the discount factor at the appropriate places in the various iterations given above. For example, the variant of iteration (3.15) for evaluating the discounted cost of a policy po is

where a is the discount factor and

## 2.3.5 The Role of Parallel Computation

It is well-known that. Monte-Carlo simulation is vory well-suited for parallolization; one can simply carry out mmitiple simulation ruus in parallel and occasionally morgo the results. Also several DP-related methods are well-suited for parallelization; for example, each value iteration can be parallolized by executing the cost updates of different states in different parallel processors (soo c.g.. [AM'T93)). lu fact the parallel updates can be assnchronons. By this we mean that dittoront Drocessors mar execute cost updates as fast as they can, without waiting to acquire the most recent updates from other processors; these latter updates may be late in coming because some of the other processors may be slow or because some of the communication channels comocting the processors may be slow. Asyu« bronous parallel value iteration can be shown to have the same convergence properties as its synchronons couterpart, and is often substantially faster. We refer to Bor82al and BolSDal for an extensive discussion.

There are similar parallelization possibilities in approximate DP. Indeed. approximate police iteration mar he viowed as a combination of tin operations:

- (a) Simulation, which produces many pairs (ire)) of states i and sample coster(i) associated with the improved policy it.
- (1)) Training. which obtains the state-sample cost pairs produced by the simulator and uses them in the least squares optimization of the pirameter vector e of the approsimate cost function ,(.).

The simulation operation can be parallelized in the usual way by excruting multiple independent simulations in multiple processors. The training operation can also be parallelized to a great extent. For example. one may parallelize the gradient iteration

<!-- formula-not-decoded -->

that is used for training lef. Ey. (3.20)!. There are two possibilities here:

- (1) To assign differout components of 7 to different processors and to execute the component updates in parallel.
- (2) To parallelize the computation of the sample gradient.

in the gradient iteration. by assigning diflerent blocks of state-sample cost pairs to difterent processors.

There are several straight forward versions of these parallelization methods. and it is also valid to use asynchronous versions of them ((BoT89a). Ch. 7).

There is still another parallelization approach for the training process. It is possible to divido the state space S' into soveral subsets Sm,. m = 1.....M. and to calentato a different approximation in(i. Im) for each subset Su. In other words, the parameter vector to that is used to calculate the approximate cost lu(i Ta) depends on the subset, S., 1o which state i belongs. The parameters Fu, can be obtained by a parallel training process using the applicable simulation data, that is, the state sample cost pairs (ir(e)) with i eS. Note that the extreme case Where each set. Si, corresponds to a single state. coresponds to the case where there is no approximation.

## 2.4 NOTES, SOURCES, AND EXERCISES

The analesis of the stochastic shortest path problems of Section 2.1 is Lakon from BoTSDal and BeT91b. The latter reference proves the resulls

shown here under a more goneral compactness assumption on U(1) and continuity assumption on g(i, 4) and po (4). Stochastic shortest path probloms were first formulated and analyzed in [EaZ62) moder the assumption 4(i, 1) &gt; 0 for all i = 1,.., 1 and « € U(i). Finitely terminating value iteration algorithms have been developed for several types of stochastic shortest, path problems (see (Ng88), (PoT92), (PST93), (Tsi93al). The use of a Dijkstra-like algoritlon for continuons space shortest path problems involving a consistently improving policy was proposed in fTsi93al (see Exercise 2.10). A Dijkstra-like algorithm was also proposed for another class ol problems involving a consistently improving policy in (Ngly$). 'The algorithm of Exorcise 2.11 is new in the general form given here. The error bound on the performance of approximate policy iteration (Prop. 2.1), which was developed in collaboration with J. Tsitsiklis, is also new. Twoplayer dynamic game versions of the stochastic shortest path problem have been discussed in (PoA69) (see also the survey (RaF91]).

Soveral approximation methods that are not based on simulation were given in (Sc585). The interest in simulation-based methods is relatively recout. In the artificial intelligence community, these methods are collectively roferred to as conforcement learning. lu the enginecring community, these methods are also referred to as ncuro-dynamic programming. The method of tomporal differences was proposed in an influential paper by Sutton (Sut88). Q-learning was proposed by Watkins (Wat89). A convergence proof of Q-learning under fairly weak assumptions was given in (Tsiot): see also (J.JS93), which discusses the convergence of TD(X). For a nice survey of related mothods, which also includes historical references, see (BB593). A variant of Q-learning is the mothod of advantage updating developed two-sample simulation-based gradient method for minimizing the error in Bellman's equation was proposed in (Ber95b; see also (HBK94). The optimistic policy iteration method was used in an application to backgammon described in (Tost)2).

## EXERCISES

2.1

Sappose that you want to travel from a start pointS to a destination point

D) in minimum average time. 'There are two dotions:

(1) Use a dircet coute that roguires a time units.

2.2

A gambler engages in a game of successive coin lipping over an infinite hori- zon. Ile wins one dollar each time heads comes up, and loses m. &gt; 0 dollars each time two successive tails come up (so the sequence TTTT loses 3m dollars). The gambler at each time period either flips a fair coin or else cheats by lipping a two-headed coin. In the latter case, however, he gets caught with probability @ &gt; 0 before he flips the coin, the gamo terminates, and tho gambler keeps his carniugs thus far. The gambler wishes to maximize his

- (in) View this as a stochastic shortest path problem and identify all proper
- and all improper policies.
- (b) Identify a critical value M such that if m. &gt; mi, then all improper policies give an infinite cost, for some initial slate.
- (c) Assume that m &gt; m, and show that. it is then optimal to try to cheat if the last flip was tails and to play fair otherise.
- (d) Show that if m &lt; M it is optimal to always play fair.

2.3

Consider a stochastic shortest path problem where all stationary policies are proper. Show that for every policy o there exists an m &gt; 0 such that

- (2) 'Take a potential shortcut that requires o time units to go to an intermediate point 1. From I you can either go to the destination D in c timo units or return to the start. (this will take an additional o time units). You will find out the value of e once you reach the intermediate What you know a priori is that e has one of the in values 1. ·, Can I with corresponding probabilities pr...P,. Consider two cases: (i) The value of e is constant over time, and (i) The value of c changes each time you return to the start. indopondently of the valoo at. the previous time periods.
2. (a) Formulate tho problom as a stochastic shortost path problen. Write Belan's equation and charactories the optimal stationary policies as best as you can in terms of the given problem data. Solve the problem for the case a = 2, 6 = 1, co = 0, 62 = 5, 71 = 0.5, 72 = 0.5.
3. (b) Formulate as a stochastic shortest path problem the variation where once you reach the intermediate point I, you can wait there. Each i time units the value of e changes to one of the values cr,..., Cm with probabilities pi,..., Pm. independently of its earlier valnes. Each time the value of e changes, you have the option of waiting for an extra d units, returning to the shitt, or going to the destination. Characteries the optimal stationary policies as best is you can.

for all i = 1,..., 8. Abbreviated Proof: Assume the contrary; that is, there exists a nonstationary o = 10,01,.. and an initial state e such that P(rn = 1 | 20 = i,7) = 0 for all m. For each state j. let m(i) be the minimum integer m such that stato j is reachable from i with positive probability under policy o; that. is,

<!-- formula-not-decoded -->

where we adopt the convention that mfj) = d if jis not. reachable from i under n. i.d., P(r,, = j (ro = i,) = 0 for all m. In particular, we have m(i) = 0 and m(1) = do. Consider any stationary policy a such that /(i) = 1mos(j) for all j with m(j) &lt; 00. Argue that for any two states states j' with m(j') = o0 (including t) are not reachable under the stationary policy &amp; from states i with m(j) &lt; 00 (including i), thereby contradicting the hypothesis.

2.4

Consider the stochastic shortest path problom, and assume that gi, u) &lt; 0 for all i and a E U(i). Show that either the optimal cost is -oo for some initial state, or else, undor evory policy, the systom eventually enters with probability one a set. of cost-free states and nover leaves that sot thereafter.

2.5

Consider the stochastie shortest path problem, and assume that there crists at least one proper policy. Proposition 1.2 implies that. if, for each improper policy 1, we have J,(i) = ou for at least, one state i, then there is no improper policy de such that do (j) = -00 for at least one state j. Give an alternative proof of this fact that does not use Prop. 1.2. Hint: Suppose that there exists an improper policy d such that d(j) = -00 for at least one state i. Combine this polier with a proper polier to produce another improper policy "" for which o,"(i) &lt; ou for all i.

## 2.6 (Gauss-Seidel Method for Stochastic Shortest Paths)

Show that the Causs-Seidel version of the value iteration method for stochastie shortest paths convorges under the same assumptions as the ordinary method (Assumptions 1.1 and 1.2). Hint: Consider two finctions A and 5 that differ by a constant from " at all states except the destination, and are such that 1 &lt; To and TT &lt;7.

## 2.7 (Sequential Space Decomposition)

Consider the stochastic shortest path problem, and suppose that there is a finito sequence of subsets of states Si. 52,. So such that each of the states i = 1,. ... bolonge to one and only one of these subsols, and the following proporty holds:

For all me = 1... ..Il and states i € Sm, the successor state i is either the termination state t or else belongs to one of the subsets S.., S,, 1.., S, for all choices of the control n&amp; l'(i).

- (a) Show that the solution of this problem decomposes into the solution
- of 1/ stochastic shortest path problems, each involving the states in a subset. Si, plus a termination state.
- (b) Show also that a finite horizon problem with A stages can be viewed as a stochastic shortest path problem with the property given above.

2.8

Consider a stochastic shortest path problem under Assumptions 1.1 and 1.2. Assuming pu(a) &lt; 1 for all i £1 and n E li(6), consider another stochastic shortest path problem that has transition probabilities po, (a)/(1-p, («)) for all i # 1 and j#i, and costs

<!-- formula-not-decoded -->

- (a) Show that the two problems are equivalent in that they have the same How would you deal with the case where
- optimal costs and policies. P, (4) = 1 for some i # t and a € U(i)?
- (b) Interpret gi,«) as au average cost incurred betweon arrival to stale i
- and transition to a stale j# i.

## 2.9 (Simplifications for Uncontrollable State Components)

Consider a stochastic shortest path problem under Assmoptions 1.1 and 1:2. where the state is a composite (i.y) of too compononts i and y, and the croIntion of the main component. i can be directly allected by the control o, but the evolution of the other component o cannot (of. Soction 1.4 aud Exercise 1.22 of Vol. D). In particular, we assime that given the state (i.g) and the control o. the next state (ire) is determined as follows: first jis conorator necording to transition probabilities po (u.y), and then &amp; is generated according to conditional probabilitios pfe ij) that depend on the main component j of the now state. We also assume that the cost per stage is glin. nod) and does not depend on the second component e of the next state (i, a). For functions i (i), i = 1, , 1. consider the mapping

<!-- formula-not-decoded -->

Atti miss

- -1

and the corresponding mapping of a stationary policy 1,

<!-- formula-not-decoded -->

- (a) Show that D:? is a form of Bolman's equation and can be used to characterize the optimal stationary policies. Hint. Given Ji, y), define
- (1)) Show the validity of a modified value iteration algorithm that starts with an arbitrary function &amp; and sequentially produces ti, id, ...
- policy evaluation step, which computes the unique function . in that solves the linea systom of equations ink = Think. (2) The policy improvoment stop, which computes the improved policy ait i,g) from
- (d) Suppose that y is the only source of randomness in the problem; that is, for each (i,!, «), there is a stato j such that pr(u,y) = 1. Justify the use of the following single sample version of value iteration (cf. the @-learning algorithmn of Section 7.6.2)

Here, givon i, we gonerate y according to the probability distribution p(« | i), and j is the unique state corresponding to (i,.u).

## 2.10 (Discretized Shortest Path Problems (Tsi93a])

Suppose that the states are the grid points of a grid on the plane. The set of neighbors ol each grid point r is denoted (.) and includes between two and four grid points. At each grid point e, we have two options:

- (1) Choose two neighbors ator € ((r) and a probability p € (0, 1], Day a cost. g(r) Vp+ (T-p)", and move to at or tod with probability " or 1-p, rospectively. Here y is a function such that gle) &gt; 0 for all t.
- (2) Stop and par a cost 1(.).

Show that there exists a consistently improving optimal policy lot this problem. Note: This problem can be used to model discretized versions of deterministic continuous space 2-dimensional shortest path problems. (Compare also with Exercise 6.11 in Chapter G of Vol. I.)

## 2.11 (Dijkstra's Algoritlun and Consistently Improving Policies)

Considor the stochastic shortest path problom under Assumptions 1. and 1.2, and assume that, there exists a consistently improving optimal stationary policy.

- (a) Show that the transition probability graph of this policy is depolic
- (b) C'onsidor the following algorithm, which maintains too subsots of states P' and L, and a tumotion e dotinod on the stato space. ('lo relate the algorithm with Dijkstra's method of Section 2.3. 1 of Vol. 1. associate I with the node labels, I, with the OPEN list, and f' with the salset of nodes that have already exited the OPEN list.) Initially, 1 = 0. L = f. and

<!-- formula-not-decoded -->

Ar the typical iteration, select a state f" from &amp; such that

(Il 1, is ompty the algorithm terminates.) Remore i from I and place it. in P. In addition, for all i &amp; P such that there exists a nE life) with P,,* (1) &gt; 0. and

<!-- formula-not-decoded -->

dofine

<!-- formula-not-decoded -->

set

<!-- formula-not-decoded -->

and place i in Dif it is not. already there. Show that the algorithm is well defined in the sonse that Ci) is nonempty and the set L does not become ompty until all states are in P. Farthermore, each state jis removed from To once, and at the time it is remosed, we have

## 2.12 (Alternative Assumptions for Prop. 1.2)

Consider a variation of Assumption 1.2, whereby we assume that go. 4) ≥ 0 for all i and a € U(i), and that there exists an optimal proper police Prove the assertions of Prop. 1.2. except that, in part (a), miqueness of the solution of Bellman's equation should be shown within the set Sit =4010 &gt; 09 (rather than within D"), and the vector of in part (b) must belong to li'.

Int: Proposition 1.1 is not valid, so a somewhat different proof is needed. Complete the dotails of the following argument. The assumptions guarantee that D' is linite and ' E Dèt. (We have o* ≥ 0 because gin) ≥ 0, and I'(i) &lt; x because a proper polies exists. The idea now is to show that I"&gt; 'TO", and then do choose ye such that Tad" = I." and show that ye is optimal and propor. Let a = 100,89,..) be a policy. We havo for all i.

<!-- formula-not-decoded -->

where t is the policy lee Me...f. Since a, 20", we obtain

Taking the infimum over a in the proceding cquation, we obtain

<!-- formula-not-decoded -->

Let ye be such that Dod' = 1". From Kg. (4.1), we have de 2 1,1", and using the monotonicity of Ta, wo obtain

<!-- formula-not-decoded -->

By taking limit superior as toes we obtain deady. 'Therefore, y is in optimal proper policy. and J* = ,. Since f was selected so that T"J" = T.J", we obtain, using f* = o, and J, = Toon. that f* = T.J". lor the rest of the proof, use tho vector de similar to the proof of Prop. 1.2.

## 2.13 (A Contraction Counterexample)

Consider a stochastie shortest path problem with a single state 1, in addition to tho tormination stato 1. At stato 1 there are two controls a and 4'. Uider « the cost is land the system remains in state I for one more stage: under i the cost is 2 and the system moves to t. Show that Assumptions 1.1 and 1.2 aro satisfied, bu. 'Tis not. a contraction mapping with respect to any norm.

## 2.14 (Contraction Property - All Stationary Policies are Proper)

Assume that all stationary policies are proper. Show that the mappings T and 'T, are contraction mappings with respect to some weighted sup norm

<!-- formula-not-decoded -->

where e is a vector whose components li,e, "n ate positive. Abbroniall proof (from /807800). p. 305: sce also (Tso901): Partition the state spaco as follows Lot Sy = 114 and lor li == 2.13. .. deline sequentially

<!-- formula-not-decoded -->

Let. S,, be the last of these sots that is nonempty. We clain that the sets Sa covor the entire stato space, that is, U4 684 = 6. To see this, suppose that the set so = fili &amp; Utsay is nonomply: Then for each ee So, there exists some a, € U(i) such that pe(a,) = 0 for all j&amp; Sx. Take any oo such that /(i) = «, for all i € Sx. The stationary police A satisfies (2,91 = 0 for allie Sa.j&amp; Se. and .V, and thereforo camot bo propor. This contradicts the hypothesis.

We will choose a vector a &gt; 0 so that. T is a contraction mapping with respect to l-l.. We will take the ith component e, to be the same for states ¿ in the same sot Sa. In particular. we will choose the components o, of the voctor els

<!-- formula-not-decoded -->

where y. ...In are appropriatol chosen scalars satisfying

<!-- formula-not-decoded -->

Let.

<!-- formula-not-decoded -->

and note that OxeS 1. No will show that it is sullicion to choose ver ... ,!!n so that for some a &lt; 1. we have

<!-- formula-not-decoded -->

and then show that such a choice of yea... exists.

Indeed. for vectors find f' in 3", let y be such that TD = 1.. Then we have for all i.

<!-- formula-not-decoded -->

Let dCi) be such that bolongs to the set Saes. Then we have for any constant

<!-- formula-not-decoded -->

and Eg. (1.6) implics that for all i,

<!-- formula-not-decoded -->

where the second inequality follows from ly. (1.3), the third incquality uses Ey. (4.1) and the fact tho) - - Yw S0, and the last incquality follows from Eq. (1.5). Thas, wo have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and wo obtain

11 follows that 'T' is a contraction mapping with respoet to ll• l..

We: now show how to choose the scalars me layn, lu so that Ers. (1.3) and (1.5) hold, 60t 10 = 0, 10 = 1, and suppose that 02,.., th have been choson. HEr = 1, we choose tan = glit 1. le &lt; 1, we choose pate to be

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Using the fact

<!-- formula-not-decoded -->

il is soon by induction that for all li,

lu particular. we hare

<!-- formula-not-decoded -->

which implics lig. (11.5).

## 2.15 (Multiple State Visits in Monte Carlo Simulation)

Argue that the Monte-Carlo simulation formula

<!-- formula-not-decoded -->

Jef. Eg. (3.1) is valid even if a state may be revisited within the same sample trajoctory. Hint: Suppose the Al cost samples dre generated from N trajetories, and that the kith trajectory involves ne visits to state i and gonerates ne corresponding cost samplos. Denote ma = n t. + ni. Write

<!-- formula-not-decoded -->

and argue that

<!-- formula-not-decoded -->

(or see (Ros83)), Cor. 7.2.3 for a closoly related result.).

## 2.16 (Multistage Lookahead Policy Iteration)

- (a) Consider the stochastic shortest path problem under Assumptions 1.1 and 1.2. Let fe be a stationary policy, let. J be a function such that. TJ ≤ 1 ≤ 1, (J = 1, is one possibility), and let Kot,.., EN-il be an optimal policy for the N-stage problem with terminal cost function J, i.r.

- (a) Show that.

<!-- formula-not-decoded -->

Mint: First show that Phild &lt; 1%0 SO for all li, and theo show that.

- (b) Use part (a) to show the validity of the multistage policy iteration algorit ln discussed in Section 2.3.3.

en accedere caraccata cron ma lora cara

## 2.17 (Viewing Q-Factors as Optimal Costs)

Consider the stochastic shortest path problem under Assumptions 1.1 and 1.2. Show that the @-factors Q(i, a) can be viewed as state costs associated with a modified stochastic shortest path problem. Use this fact to show that the Q-factors Q(i, 1) are the unique solution of the system of equations

<!-- formula-not-decoded -->

Hint: Introduce a new state for each pair (i, 2), with transition probabilities po, («) to the states j = 1,..., 1, t.

## 2.18 (Advantage Updating)

Consider the optimal Q-factors @°(i, a) of the stochastic shortest path problom under Assumptions 1.1 and 1.2. Dofino the advantoge function by

<!-- formula-not-decoded -->

- (a) Show that. /*(i,«) together with the optimal costs *(i) solve uniquely the system of cquations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (0)) Introduce approximating fictions Ai, 4,2) and i,o), and derive a gradient mothod aimed at minimizing the sum of the squared errors of the Bellman-liko cquations of part (a) (of. Section 2.3.3).

3

## Undiscounted Problems

| 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        | 3.1. Unbounded Costs per Stage D. 13•4        |
|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|
| 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 | 3.2. Linear Systems and Quadratic Cost р. 150 |
| 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 | 3.3. Inventory Control р. 153                 |
| 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 | 3.4. Optimal Stopping 1›. 155                 |
| 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       | 3.5. Optimal Gambling Strategies p. 160       |
| 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      | 3.6. Nonstationary and Periodic Problems      |
| p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        | p. 167                                        |
| 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          | 3.7. Notes, Sources, and Exercises .          |
| p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        | p. 172                                        |

In this chapter we consider total cost infinite horizon problems where we allow costs per stage that are unbounded above or below. Also, the discount factor a does not have to be less than one. The complications resulting are substantial, and the analysis required is considerably more sophisticated than the one given thus far. We also consider applications of the theory to important classes of problems. The problem section touches on several related topics.

## 3.1 UNBOUNDED COSTS PER STATE

In this section we consider the total cost infinite horizon problem of Section I. I under one of the following two assumptions.

Assumption P: (Positivity) The cost per stage g satisfies

<!-- formula-not-decoded -->

Assumption N: (Negativity) The cost per stage g satisfies

<!-- formula-not-decoded -->

Problems corresponding to Assumption Pare sometimes referred to in the research literature as negative DI problems. This name was used in the original reference (Str66), where the problem of maximizing the infinite sil of negative rewards per stage was considered. Similarly, problems cortesponding to Assumption N are sometimes roferred to as positive DP problems (Bla65), (St266). Assumption N arises in problems where there is a nonnegative reward per stage and the total expected reward is to be marimized.

Note that when a &lt; 1 and o is either bonded above or below, we mav add a suitable scalar to g in order to satisfy Eg. (1.1) or Eg. (1.2), respectively. An optimal policy will not be alfected by this change since, in view of the presence of the discount factor, the addition of a constant r 10g merely adds (1 -a)-ly to the cost associated with every policy.

One complication arising from unbounded costs per stage is that, for some initial states do and some genuinely interesting admissible policies

= =110:141....7, the cost da(co) may be o (in the case of Assumption P) or -o (in the case of Assumption N). Here is an example:

## Example 1.1

Consider the scalar system and hence

while

<!-- formula-not-decoded -->

Note a peculiarity here: if 8 &gt; 1 the state da diverges to oo or to -oo, but if the discount factor is sufficiently small (a &lt; 1/13), the cost. Ja(ro) is finite.

It is also possible to verify that when 6 &gt; 1 and of &gt; 1 the optimal cost J'(ico) is equal to so for |col ≥ 1/(- 1) and is finite for |rol &lt; 1/(B- 1). The problem here is that when 8 &gt; 1 the system is unstable, and in view of the restriction fuel &lt; 1 on the control, it may not be possible to force the state near zero ouce it has reached sufficiently largo magnitude.

The preceding example shows that there is not much that, can be done about the possibility of the cost function being infinite for some policios. To cope with this situation, we conduct our analysis with the notational understanding that the costs Ja(co) and J*(co) may be do (or -00) under Assumption P (or N, respectively) for some initial states to and policies т. In other words, we consider An(.) and J*(.) to be extended real-valued functions. In fact, the entire subsequent analysis is valid even if the cost. 9(2,4,4)) is 00 or -00 for some (8, 4,1), as long as Assumption l' or Assumption N holds.

The line of analysis of this section is fundamentally different. from the one of the discounted problem of Section 1.2. For the latter problem. the analysis was based on ignoring the "tails" of the cost sequences. In

<!-- formula-not-decoded -->

where da € X and un € M. for all ki, and &amp; is a positive scalar. The control constraint is Jus| &lt; 1, and the cost is

<!-- formula-not-decoded -->

Consider the policy a = {i,a,...%, where 7(r) = 0 for all &amp; € M. Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

this section, the tails of the cost sequences may not be small, and for this reason, the control is much more focused on affecting the long-term behavior of the state. For example, lot a = 1, and assume that the stage cost at all statos is nomoro excopt. for a cost-free and absorbing termination state. 'Then, a primary task of control under Assumption P' (or Assumption N) is roughly to bring the state of the systom to the tommination state or to a region where the cost per stage is nearly coro as quickly as possible (as late as possiblo, respectively). Note the dillerence in control objective between Assumptions P and N. It accounts for some strikingly different results under the two assumptions.

## Main Results - Bellman's Equation

We now present results that characterize the optimal cost function J*, ils well as optimal stationary policies. We also give conditions under which value iteration converges to the optimal cost function J*. In the proofs we will often need to interchange expectation and limit in various rolations. This interchange is valid under the assumptions of the following theorem.

Monotone Convergence Theorem: Let P = (21,72,...) be a probability distribution over S = 11,2,..7. Let {hing be a sequence of extended real-valued functions on S such that for all i e S and N = 1,2,...,

<!-- formula-not-decoded -->

Let he: 5→ (0,00] be the limit function

Then

Proof: We have

By taking the limit, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so there remains to prove the reverso inequality. For every intoger 1 1, we have

<!-- formula-not-decoded -->

and by taking the limit as A - d the reverse incquality follows. Q.E.D.

Similar to all the infinite horizon probloms considered so far. tho optimal cost function satisfies Bellman's equation.

Proposition 1.1: (Bellman's Equation) Under either Assumption P or N the optimal cost function J* satisfies

<!-- formula-not-decoded -->

or, quivalently, where, for all i eS,

<!-- formula-not-decoded -->

Thus, W(e) is the cost from stage 1 to infinity using a when the initial state is a. We have clearly

<!-- formula-not-decoded -->

Hence, from Eg. (1.3),

<!-- formula-not-decoded -->

Taking the minimum over all admissible policies. we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: For any admissible policy a = 140.891,...%, consider the cost d(e) corresponding to o when the initial state is d. Wehave

<!-- formula-not-decoded -->

--1

Thus there remains to prove that the reverse inequality also holds. We provo this soparately for Assumption N and for Assumption P.

Assino P. The following proof of J* ≤ T.J" under this assumption would be considerably simplitied if we knew that there exists i fe such that. T,J* = T.J*. Since in general such a f need not exist, we introduce a positive sequence fred, and we choose an admissible policy a = 410,14,...3 such that

<!-- formula-not-decoded -->

Such a choice is possible because we know that, under P, we have -20 &lt; /*(e) for all r. By using the inequality T.J* &lt; J* shown carlier, we obtain

Applying They to both sides of this rolation, we have

<!-- formula-not-decoded -->

Continuing this process, we obtain

By taking the limit as hi → x and noting that

where do is the vero timotion, it follows that

<!-- formula-not-decoded -->

Since the sequence fond is arbitrary, we can take Exce, its small as desired, and we obtain d*(e) ≤ (T./*)(a) for all a € S. Combining this with the inequality 1*(e) ≥ (T.*)(e) shown carlier. the result follows (under Assimoption P).

Assumo Nand lot. de be the optimal cost function for the corresponding N-stage problem

<!-- formula-not-decoded -->

We first show that

<!-- formula-not-decoded -->

Indeed, in view of Assumption N, we have J* &lt; IN for all N, so

<!-- formula-not-decoded -->

Also, for all a = {no, Me,...%, we have

and by taking the limit as N - oo,

Taking the minimum over 7, we obtain f(x) ≥ limy-ou Jw(c), and combining this relation with Eg. (1.7), we obtain Eg. (1.6).

For every admissible y, we have

and by taking the limit as N → ∞, and using the monotone convergence theorem and Eq. (1.6), we obtain

Taking the minimum over y, we obtain TJ* ≥ J*, which combined with the inequality J* ≥ T.J* shown carlier, proves the result under Assumption N. Q.E.D.

Similar to Cor. 2.2.1 in Section 1.2, we have:

Corollary 1.1.1: Let fe be a stationary policy. Then under Assumption P or N, we have

<!-- formula-not-decoded -->

or, equivalently,

<!-- formula-not-decoded -->

Contrary to discounted problems with bounded cost per stage, the optimal cost function J* under Assumption P or N need not be the unique solution of Bellman's equation. Consider the following example.

## Example 1.2

Let 5 = (0, x) (or 5 = (-x,0)) and

Then for every 1, the function of given by

<!-- formula-not-decoded -->

is a solution of Bollman's equation, so T' has an infinite mumber of fixed points. Note, however, that, there is a unique fixed point within the class of bounded fictions, the zero function do(r) =0, which is the optimal cost finction for this problem. More generally, it can be shown by using the following Prop. 1.2 that if a &lt; 1 and there exists a bounded function that is a fixed point of T, then that function must be equal to the optimal cost. function f" (sec Exercise 3.5). When a = 1, Bollman's equation may have an infinity of solutions even within the class of bounded functions. This is because if a = 1 and fo) is any solution, thou for any scalar e, J-) + e is also a solution.

The optimal cost function f*, however, has the property that it is the smallest (under Assumption P) or largest (under Assumption N) lixed point of 'T in the souse described in the following proposition.

## Proposition 1.2:

- (a) Under Assumption P, if J : S' + (-00, oo| satisfies J &gt; TJ and either J is bounded below and a &lt; 1, or J &gt; 0, then . &gt; J*.
- (b) Under Assumption N, if j: S'+ 1-00, 20) satisfies I &lt; TJ and either J is bounded above and a &lt; 1, or J &lt; 0, then J &lt; J*.

Proof: (a) Under Assumption P, let r be a scalar such that J.) + , &gt; 0 for all a € Sand if a 2 1 let r= 0. For any sequence {hy with eh &gt; 0, let # = fioM,.. be an admissible policy such that, for evory &amp; E Sand ki,

<!-- formula-not-decoded -->

Such a polies exists since ('T.))(e) &gt; -00 for all &amp; € S. We have for any initial state to € S,

<!-- formula-not-decoded -->

Using Ey. (1.9) and the assumption &amp; ≥ 'P.), we obtain

<!-- formula-not-decoded -->

Combining these incqualities, we obtain

<!-- formula-not-decoded -->

Since the sequence fred is arbitrary (except for ce &gt; 0), wo may select. fend so that limnex Eide is arbitrarily close to zero, and the result

(b) Under Assumption N, let e be a scalar such that (r) + e ≤ 0 for all ¿ E S, and it a 2 1. let 6=0. We have for every initial state to € S.

<!-- formula-not-decoded -->

where the last incomality follows from the fact that for an sequence Shafg)! of functions of a parameter e we have

This incquality follows by writing

<!-- formula-not-decoded -->

and by subsequently taking the limsup of both sides and the minimum over &amp; of the left-hand side.

Now we have, by using the assumption J≤TJ,

<!-- formula-not-decoded -->

Using this rolation in By. (1.10), we obtain

Q.E.D.

As before, we have the following corollary:

Corollary 1.2.1: Let ye be an admissible stationary policy.

- (a) Under Assumption P, if J: S + (-00, ool satisfies J &gt; TuJ and either I is bounded below and a &lt; 1, or J ≥ 0, then J ≥ Ju.
- (b) Under Assumption N, if J : S+→ (-00, 00) satisfies J≤ T,J and cither J is bounded above and a &lt; 1, or I &lt; 0, then I ≤ Ju.

## Conditions for Optimality of a Stationary Policy

Under Assumption P, we have the same optimality condition as for discounted problons with bounded cost per stage.

Proposition 1.3: (Necessary and Sufficient Condition for Optimality under P) Let Assumption P hold. A stationary policy y is optimal if and only if

<!-- formula-not-decoded -->

Proof: 11T.* = TaJ*, Bolman's equation (* = 1.*) implies that 1' Tul*. From Cor. 1.2.1(a) we then obtain f* ≥ Ja, showing that y is optimal. Conversoly, if J* = Ju, we have using Cor. 1.1.1, T.J* = 1* = J, = T, 1, = T, 1*. Q.E.D.

Unfortunately, the sufficiency part of the above proposition need not. be true under Assumption N; that is, we may have T.J* = T,/* while y is not optimal. This is illustrated in the following example.

## Example 1.3

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all (2, u, w) € S X C X D. Then f*(d) = -00 for all a € S, and every stationary policy y satisties the condition of the preceding proposition. On the other hand, when 4(r) = O for all &amp; E S, we have J,(v) = 0 forall i CS, and honce ye is not optimal.

It is worth noting that. Prop. 1.3 implies the existonce of an optimal stationary policy under Assumption P when U(.) is a finite set for every x ES. This need not be true under Assmmption N (sco Example 4. in Section 3.1).

Under Assumption N, we have a different characterization of an optimal stationary policy.

Proposition 1.4: (Necessary and Sufficient Condition for Optimality under N) Let Assumption N hold. A stationary policy y is optimal if and only if

<!-- formula-not-decoded -->

Proof: If TJ, = TuDe, then from Cor. 1.1.1 we have op = Tid,, so that. J, is a fixed point of T. Then by Prop. 1.2, we have 1, &lt; J*, which implies that ye is optimal. Conversely, if J, = J*, then TaD, = 1, = 1*=T./* = TJu. Q.E.D.

The interpretation of the preceding optimality condition is that persistently using y is optimal il and only if this performs at least as well is using any to at the first stage and using a thereafter. Under Assumption P' this condition is not sallicient. to guarantee optimality of the stationary policy 1e, as the following example shows.

## Example 1.4

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all (6, 1, «i) € S X C X D. Let 4(2) = 1 for all r E S. Then J,(x) = ∞0 if c: # 0 and J,(0) = 0. Furthermore, we have d, = TaJ, = T.,, is the reader can casily verify. It can also be verified that f(x) = Jel, and hence the stationary policy yo is not. optimal.

## The Value Iteration Method

We now turn to the question whether the DP algorithm converges to the optimal cost function J*. Let do be the zero function on S,

<!-- formula-not-decoded -->

Then under Assumption P, we have

while under Assumption N, we have

In either case the limit function

<!-- formula-not-decoded -->

is well dofined, provided we allow the possibility that de can take the value x (under Assumption P) or -do (under Assumption N). The question is whether the value iteration method is valid in the souse

<!-- formula-not-decoded -->

This question is, of course, of computational interest, but it is also of analytical interest sinee, if one knows that. f* = linken ThiJo, one can infer properties of the unknown fuuction f* from properties of the li-stage optimal cost fimotions fedo, which are defined in a concrete algorithmie manner.

We will show that ee =o under Assumption N. Il Tarns out. however, that under Assumption P, we may have of of (sco lixorcise :3.1). Ne will later provide casily veritiable conditions that guaranter that. Jx = 1* under Assumption P. We have the following proposition.

## Proposition 1.5:

- (a) Let Assumption P hold and assume that

<!-- formula-not-decoded -->

Then if J: S → d is any bounded function and a &lt; 1, or otherwise if Ju ≤1 ≤ 0*, we have

<!-- formula-not-decoded -->

- (b) Let Assumption N hold. Then if J: S+ D is any bounded function and a &lt; 1, or otherwise if J* ≤ .J ≤ Jo, we have

<!-- formula-not-decoded -->

Proof: (a) Since under Assumption P, wo have

it follows that limax Th Jo = x SoJ*. Since De is also a fixed point of T' by assumption, wo obtain from Prop. 1.2(a) that. * ≤ ox. It. follows that

<!-- formula-not-decoded -->

and hence Eq. (1.14) is proved for the case f = Jo.

For the case where a &lt; 1 and of is bounded, let y be a scalar such that,

<!-- formula-not-decoded -->

Applying Th to this relation, we obtain

<!-- formula-not-decoded -->

Since 1'% Jo converges to J*. as shown carlier, this relation implies that Th converges also to o*.

Undiscounted Problems Chap. 3

In the case where do 5d S*, we have by applying Th

<!-- formula-not-decoded -->

Since Th do converges to *, so does Thy.

(0) It was shown carlier jef. Eg. (1.6)j that under Assumption N, we have

<!-- formula-not-decoded -->

The proof from this point is identical to that for part (a). Q.E.D.

We now derive conditions guaranteeing that Jx = T. hokds under Assumption P, which by Prop. 1.5 implies that x = J*. We prove two propositions. The first admits an easy proof but requires a finiteness assumption on the control constraint set. The second is harder to prove but requires a weaker compactness assumption.

Proposition 1.6: Let Assumption P hold and assume that the control constraint set is finite for every 2 € S. Then

<!-- formula-not-decoded -->

Proof: As shown in the proof of Prop. 1.5(4), we have for all hi, Th do ≤ Ix ≤1". Applying T to this relation, we obtain

<!-- formula-not-decoded -->

and by taking the limit. as ki → oo, it follows that.

Supposo that thore existod a state i e S such that

<!-- formula-not-decoded -->

Let th minimize in Ey. (1.22) when d =&amp;. Since U(x) is finite, there must exist some do (?) such that need for all le in some infinite subset K of the positive intogers. By By. (1.22) we have for all hi € 1

<!-- formula-not-decoded -->

Taking the limit is hi → ∞0, k E k, we obtain

<!-- formula-not-decoded -->

This contradicts Eq. (1.23), so we have 1x (E) = (T.Ja)(E). Q.E.D.

The following proposition strengthens Prop. 1.6 in that it requires a compactness rather than a finiteness assumption. We recall (see Appendix A of Vol. 1) that a subset X' of the 2-dimensional Euclidean space K" is said to be compact if every sequence {ak) with an € X contains a subsequence {zk)ken that converges to à point i: € X. Equivalently, X is compact if and only if it is closed and bounded. The empty set is (trivially) considered compact. Given any collection of compact sets, their intersection is a compact set (possibly empty). Given a sequence of noncmpty compact. sets X1, 12..., Xa,... such that

<!-- formula-not-decoded -->

their intersection nee, ta is both nonempty and compact. In view of this fact, it follows that if f: 1P" → (-00, 00) is a function such that the set.

<!-- formula-not-decoded -->

is compact. for every de R, then there exists a vector to minimizing f; that is, there exists an * C R" such that.

<!-- formula-not-decoded -->

Net for all k: If mine (e) &lt; so, th a mix and l →

<!-- formula-not-decoded -->

are nonemply and compact. Furthermore, Exe O Fary for all hi. and hence the interscction nee Fa, is also nonemply and compact. Lett be any vector in Ma, Fag. Then

<!-- formula-not-decoded -->

and taking the limit as l → do. we obtain f(r*) ≤ miner f(x), proving that c* minimizes f(x). The most common case where we can guarantee

that the set Ky of By. (1.25) is compact. for all A is when fis continuous and f(r) → xu is llell → 20

Proposition 1.7: Let Assumption I hold, and assume that the sets

<!-- formula-not-decoded -->

are compact subsets of a Euclidean space for every i ES, 1 € 9i, and for all ki greater than some integer hi. Then

<!-- formula-not-decoded -->

Furthermore, there exists a stationary optimal policy.

Proof: As in Prop. 1.6, wo have do ≤ T.Ja. Suppose that there existed a state # € S' such that.

<!-- formula-not-decoded -->

Clearly, we must have x (r) &lt; xo. For every hi ≥ hi, consider the set.

<!-- formula-not-decoded -->

Let also un be a point attaining the minimum in

<!-- formula-not-decoded -->

that. is, an is such that

<!-- formula-not-decoded -->

Such minimizing points me exist by our compactness assumption. For every A &gt; li, consider the sequence {ube. Since TaJo S'ptIJo &lt; ... ≤ 100, it follows that

Therefore {u,5,24 CUrld. do (di)). and since Uk(ix (2)) is compact. all the limit points of furth belong to Cald.da(d)) and at least. one such limit point exists. Hence the same is true of the limit points of the whole sequence fu,), 11 follows that. if a is a limit point of food, then

This implies by Eq. (1.29) that for all ki ≥ Ti

Taking the limit as li → x, wo obtain

Since the right-hand side is greater than or equal to (T.)x)(2), Ey. (1.31) is contradicted. Honce a = 7. and Eg. (1.30) is proved in riow of Prop. 1.5 (al).

To show that there exists an optimal stationary policy observe that. Ey. (1.30) and the last relation imply that &amp; attains the minimum in

<!-- formula-not-decoded -->

for a state i E S' with J*(r) &lt; x. Forstates i E S' such that J*(7) =2. every « € U(%) attains the preceding minimum. Hence by Prop. 1.3(a) an optimal stationary policy exists. Q.E.D.

The reader may verify by inspection of the preceding proof that if 12(2). k=0. 1.., attains the minimme in the relation

<!-- formula-not-decoded -->

then if p*(8) is a limit point of fre(2)%, for every i E S, the stationary policy /* is optimal. Furthermore, {re(2)} has at loast, one limit point for every i E S' for which J*(r) &lt; oo. Thus the voluc iteration method under the assumplions of either Prop. 1.6 or Prop. 1.7 violds in the limit not only the optimal cost funcion fo but also an optimal slationary policy.

## Other Computational Methods

Unfortunately, polier iteration is not a valid procedure under either l or N in the absence of further conditions. If y and a are stationare policies such that Id, = TO,, then it can be shown that under Assumption I we have

<!-- formula-not-decoded -->

To see this, note that Tal, = T.J, STudy = Ju from which we obtain LimN-oo TW Ju. ≤ Ju. Since J = limN -x TW Jo and Jo ≤ Ju, we obtain Jie ≤ Ju. However, du SAy by itsell is not sufficient to guarantee the validity of policy iteration. For example, it is not clear that strict incquality holds in Eq. (1.32) for al. least. one state &amp; € S when f is not optinal. The dilliculty here is that the equality on = T., does not. imply that. ye is optimal, and additional conditions are needed to guarantee the validity of policy iteration. However, for special cases such conditions can be verified (see for example Section 3.2 and Exercise 3.16).

It is possible to devise a computational method based on mathematical programming when S, C, and D are finite sets by making use of Prop. 1.2. Under N and a = 1, the corresponding (lincar) program is (compare with Section 1.3.4)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When a = 1 and Assumption P holds, the corresponding program takes the form

<!-- formula-not-decoded -->

but unfortunately this program is not linear or even convox.

## 3.2 LINEAR SYSTEMS AND QUADRATIC COST

Consider the case of the linear system

<!-- formula-not-decoded -->

where de EX", Un E D" for all ki, and the matrices A, B are known. As in Sertions 1.1 and 5.2 of Vol. I, we assume that the random disturbances wh are indopendent with zero mean and finite second moments. The cost. function is quadratic and has the form

<!-- formula-not-decoded -->

where @ is a positivo semidefinite symmetric a X n. matrix and Ris a positive definite symmetrie m. x m matrix. Clearly, Assumption P of Section 3.1

holds.

Our approach will be to use the DP algoritlum to obtain the functions

As in Section 4.1 of Vol. I, we have

TJo, T2 Jo..., as well as the poitwise limit fanction fa = link-a ThAo. Subsequently, we show that do satisfies 1o = Tols and hence, by Prop. 1.5(a) of Section 3.1, dx =*. The optimal policy is then obtained from the optimal cost function J* by minimizing in Bellman's equation (ef. Prop. 1.3 of Section 3.1).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the matrices Ao. Ki. Az.... are given recursively by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By defining A = R/a and A = Jad, the preceding equation mar be written as

and is of the form cousidered in Section 4. 1 of Vol. I. By using the result shown there, we have that the generated matrix sequence fael convorges to a positive definite symmetric matrix A,

provided the pairs (À, 13) and (Ä.C), where Q = C'C. are controllable and observable, respectively. Since A = Sad, controllability and observability

of (1, 3) or (A, C) are clearly equivalent to controllability and observability of (1.13) 08(1, C); respectively: The matrix A is the unique solation of tho equation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, in conclusion, if the pairs (A, B) and (A. C) are controllable and observable, respectively, the limit. of the functions Tilo is given by

<!-- formula-not-decoded -->

Using Egs. (2.1) to (2.3), it can bo verified by straightforward calculation that for allies

and honce, by Prop. 1.5(a) of Section 3.1. da =*. Another method for proving that x = Tole is to show that the assmuption of Prop. 1.7 of Section 3.1, is satisfied; that is, the sets

are compact. for all ki and scalars 1. This can bo verified using the fact that Tido is a positive semidefinite quadratic function and R is positive delinito. The optimal stationary policy pe, obtained by minimization in Py. (2.1), has the torm

This policy is attractive for practical implomentation since it is linear and stationary. A number of generalized versions of the problem of this section, including the case of imperfect state information, are treated in the exercises. Interestingly, the problem can be solved by policy iteration (see Exercise 3.16), even though, as discussed in Section 3.1, policy iteration is not valid in general under Assumption P.

Becauso A% = 16, it can also be seen that the limit.

<!-- formula-not-decoded -->

is woll dofined. and in fact.

## 3.3 INVENTORY CONTROL

Lot us consider a discounted. infinite horizon vorsion of the inventory control problem of Section 4.2 in Vol. I. Inventory stock evolves according to the equation

<!-- formula-not-decoded -->

We assume that the successive domands w% are independent and bounded, and have identical probabilite distributions. No also assume for simplicity that there is no fixed cost. The case of a nonzero fixed cost can be treated similarly. The cost function is

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The DI algorithm is givon by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first show that the optimal cost is finite for all initial states, that. is,

<!-- formula-not-decoded -->

Indeed, consider the policy a = f.",..%, where y is defined by

<!-- formula-not-decoded -->

Since un is nonnegative and bounded, it follows that the invontory stock ta when the policy a is used satisfies

<!-- formula-not-decoded -->

and is bounded. Hence pen) is also bounded. It follows that the cost per stage incurred when t is used is bounded, and in view of the presence of the discount factor we have

<!-- formula-not-decoded -->

Since 1* ≤ Ji, the finiteness of the optimal cost follows.

Next we observe that, under the assumption e &lt;p. the factions I% do are real-valued and convex. Indeed, we have

which implies that Tado is real-valued. Convoxity follows by induction as shown in Section 4.2 of Vol. 1.

Consider now the sets

<!-- formula-not-decoded -->

Those sets aro bounded since the expected value within the braces above lends to do as a → ∞0. Also, the sots Uk(r, d) are closed since the expected value in Eq. (3.1) is a continuous function of « recall that Thilo is a realvalued convox and hence contimous function]. Thus we may invoke Prop. 1.7 of Section 3.1 and assert that

11 follows from the convexity of the functions Th.J, that the limit, function J* is a real-valued conves function. Furthermore, an optimal stationary policy p* can be obtained by minimizing in the right-hand side of Bellman's cquation

We have

<!-- formula-not-decoded -->

where S* is a minimizing point of

<!-- formula-not-decoded -->

with

It can be seen that if o &gt; r, we have limy +x G*(y) = oo, so that such a minimizing point exists. Furthermore, by using the observation made near the end of Section 3.1, it follows that a minimizing point S* of G*(y) may be obtained as a limit. point. of a sequener (Sal, where for each &amp; the scalar Sa minimizes

and is obtained by means of the value iteration method.

It turns out, that the critical level S* has a simple charcterization. It can be shown that So minimizes over y the expression (1 - a)cy + Ly), and it can be essentially obtained in closed form (see Exercise 3.18, and (HcS84], Ch. 2).

In the case where there is a positive fixed cost (A &gt; 0), the same line of argument may be used. Similarly, we prove that. fe is a real-valued K-convex function. A separate argument is necessary to prove that J* is also continuous (this is intuitively clear and is left. for the reader). Once K-convexity and continuity of J* are established, the optimality of a stationary (s*, S*) policy follows from the equation

where C(u) = A + cu if a &gt; 0 and C(0) = 0.

## 3.4 OPTIMAL STOPPING

Consider an infinito horizon version of the stopping problems of Section 4.1 of Vol. I. At each state e, we must choose between two actions: pay a stopping cost s(e) and stop, or pay a cost c(x) and continuc the process according to the system equation

<!-- formula-not-decoded -->

The objective is to find the optimal stopping policy that minimizes the total expected cost over an infinite number of stages. It is assumed that the input disturbances an have the same probability distribution for all l, which depends only on the current state th

This problem may be viewed as a special case of the stochastic shortest path problem of Section 2.1, but hore we will not assume that the state space is finite and that only proper policies can be optimal. as vo did in Section 2.1. Instead we will rely on the general theory of unbounded cost. problems developed in Section 3.1.

To put the problem within the framework of the total cost infinite horizon problem, we introduce an additional state t (termination state) and we complete the system equation (4.1) as in Section 4.4 of Vol. I by letting

<!-- formula-not-decoded -->

Once the system reaches the termination state, it remains there permanently at no cost.

We first assume that

<!-- formula-not-decoded -->

thus coming under the framework of Assumption P of Section 3.1. The case corresponding to Assumption N, where se) ≤ 0 and co) ≤ 0 for all i ES will be considered later. Actually, whoever there exists an O such that c(r) = c for all&amp; ES, the results to be obtained under the assumption (4.2) apply also to the case where s(r) is bounded bolow by some scalar rather than bounded by coro. The reason is that, if c(r) is assumed to be greater than &gt; O for all re.S, any policy that will not stop within a finite expected number of stages results in infinite cost and can be excluded from consideration. As a result, if we reformulate the problem and add i constant e to s(r) so thatse) tr 2 0 for all &amp; ES, te optimal cost. J*(c) will increly be increased bye, while optimal policies will remain unaffected.

The mapping T' that defines the DP algorithm takes the form

<!-- formula-not-decoded -->

whore s(r) is the cost of the stopping action, and c(r) + ElAG(e,m))} is the cost. of the contimation action. Since the control space has only two dements. by Prop. 1.6 of Soction 3.1, we have

<!-- formula-not-decoded -->

where do is the zero function lo(e) = 0, for all &amp; € S. By Prop. 1.3 of Section 3.1. there exists a stationary optimal policy given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us denoto by S* the optimal stopping set. (which may be empty)

Consider also the sots

<!-- formula-not-decoded -->

that determine the optimal policy for finite horizon versions of the stopping problem. Since we have

it follows that

<!-- formula-not-decoded -->

and therefore Up Sa C.5*. Also, if i &amp; UPe, S, then we have

and by taking the limit, and using the monotone convergonce theorem and the fact ''do - Jo, we obtain

<!-- formula-not-decoded -->

from which a &amp; S*. Hence

<!-- formula-not-decoded -->

In other words, the optimal slopping sel. So for the infine horizon problem is equal to the union of all the finite horizon stopping sols Sh.

Consider now, as in Soction 4.1 of Vol. I, the one-step-to-go stopping set

<!-- formula-not-decoded -->

and assume that Sy is absorbing in the sense

<!-- formula-not-decoded -->

Then, as in Section 1.4 of Vol. I, it follows that the one-step lookahead policy

<!-- formula-not-decoded -->

is optimal. We now provide some examples.

## Example 4.1 (Asset Selling)

Consider the version of the asset, selling example of Sections 4.1 and 7.3 of Vol. I, where the rate of interest r is zero and there is instead a maintenance cost e &gt; 0 per period for which the house remains unsold. Furthermore, past oflers can be accepted at any future time. We have the following optimality equation:

In this case we consider maximization of total expected reward, the continuation cost, is strictly negative, and the stopping rewarde is positive. Hence the assumption (42) is not satisfied. If, however, we assume that e takes values in a bounded interval (0, A/, where if is an upper bound on the possible values of offers, our analysis is still applicable jef. the discussion following Ey. (4.2). Consider the one-step-to-go stopping set. givon by

After a caleulation similar to the one given in Section 4.1 of Vol. I. we see

<!-- formula-not-decoded -->

where a is the scalar satistying

Clearly, S, is absorbing in the sense of lie. (17) and therefore the one-step lookahead poliev that accepts the first offer greater than or equal to a is optimal.

.. \_1.

## Example 4.2 (Sequential Hypothesis Testing)

Consider the bypothesis testing problem of Section 5.5 of Vol. I for the case where the number of possible observations is unlimited. Here the states are "" and e' (true distribution of the observations is fo and f, respectively). The set S' is the interval 0, 1] and corresponds to the sufficient statistic

<!-- formula-not-decoded -->

To each p € (0, 1] we may assign the stopping cost

<!-- formula-not-decoded -->

that is, the cost associated with optimal choice between the distributions fo and fi. The mapping T of Ey. (4.3) takes the form

for all p € 10, 11, where the expectation over &amp; is taken with respect to the probability distribution

"The optimal cost. function * satisties Bellman's equation

and is obtained in the limit through the equation

<!-- formula-not-decoded -->

where Jo is the zoro function on [0, 1].

Now consider the timotions To Jo, k = 0, 1,... I, is clear that

Parthermore, in view of the analysis of Section 5.5 of Vol. I, we have that the function 4% Jo is concave on 10, 1) for all hi. Hence the pointwise limit function 1' is also concave on (0, 1). In addition, Bolkman's equation implies that

<!-- formula-not-decoded -->

Using the reasoning illustrated in Fig. 3.4.1 it follows that [provided c &lt; Land,/(Lu + Lo)) there exist two scalars d, 7 with 0 &lt; 7 ≤à &lt; 1, that determino an optimal stationary policy of the form accept. fo ilpen,

accept fil if pIT, continue the observations

<!-- formula-not-decoded -->

In viow of the optimality of the preceding stationary policy, the sequential probability ntio test described in Soction 5.5 of Vol. I is justified when the minber of possible observations is infinite.

Figure: 3.4.1 Derivation of the sequential probabilits ratio lost.

<!-- image -->

## The Case of Negative Transition Costs

We now consider the stopping problem under Assumption N, that is,

<!-- formula-not-decoded -->

Under these circumstances there is no ponalty for contiming operation of the system (although by not stopping at a giveu state, a favorable opportunity may be missed). The mapping T is given by

The optimal cost function * satisfies J*(e) ≤ s(e) for all e E S, and by using Props. 1.1 and 1.5(b) of Section 3.1, we have

<!-- formula-not-decoded -->

where Jo is the zero fimotion. At can also be seen that if the one-step-to-go stopping set Si is absorbing lef. Eg. (4.7)). a one-step lookahead policy is optimal.

## Example 4.3 (The Rational Burglar)

This example was considered at the ond of Section ol of Vol. I where it was shown that a one-stop lookahead policy is optimal for any finite horizon length. The optimality oquation is

The problem is equivalent to a minimization problem where

so Assumption N hokle. From the preceding analesis, we have that 1, → " and that a one stop lookahead policy is optimal if the one stop stopping set is absorbing fof. Egs. (1.6) and (1.7)]. It can be shown (see the analysis of Section 10d of Vol. 1) Chat this condition bokls, so the finite borizon optimal policy whereby the burglar retires when his accumulated carnings reach or exceed (1 - p)0/p is optimal for an infinite horizon as well.

## Example 4.4 (A Problem with no Optimal Policy)

This is a deterministic stopping problem where Assumption N holds, and an optimal policy does not exist, evon thongh only two controls are available at. each state (stop and contine). 'The states are the positive integers, and continuation from state i leads to state it 1 with certainty and no cost, that is, 5 = 11,2,..1, ‹(i) = 0, and f(i, M) = it 1 for all i E S' and a E D. The stopping cost is s(i) = - 1 + (1/i) for all i € S, so that there is an incentive to delay stopping at overy state. We have 7*(i) = - 1 for all i. and the optimal cost -1 can be approached arbitrarily closely by postponing the stopping action for a sulliciently long time. However, there does not exist an optimal policy that. attains the optimal cost.

## 3.5 OPTIMAL GAMBLING STRATEGIES

A gambles enters a certain game played as follows. The gambler may stake at any time li and amount un &gt; 0 that does not exceed his current fortune to (defined to be his initial capital plus his gain or minus his loss thus far). He wins his stake back and as much more with probability p and he loses his stake with probability (1 -p). Thus the gambler's fortune evolves according to the cquation

<!-- formula-not-decoded -->

where an = 1 with probability p and ax = —1 with probability (1 - p). Several games, such as playing red and black in roulette. fit this description.

Tho gambler enters the game with an initial capital to. and his goal is to increase his fortuno up to a level X. He continues gambling until he either reaches his goal or loses his ontire initial capital, at which point he leaves the gamo. The problem is to determino the optimal gambling strategy for maximizing the probability of reaching his goal. By a gambling stratogy, we mean a rule that specifies what the stake should be at time &amp; when the gambler's fortune is th, for evory ta with O &lt; ith &lt; X.

The problem may be cast. within the total cost, infinite horizon framework, where we consider maximization in place of minimization. bot ns assume for conconience that fortanes are normalized so that. X=1. Tho state spaco is the so 10, 1040%, where d is a termination State to which the system moves with certainty from both states 0 and 1 with corresponding rewards 0 aud 1. When de &amp; 0. # 1, the system evolves according to Eq. (5.1). The control constraint set is specified by

The reward per stage when th f 0 and taf lis sero. Under these circumstances the probability of reaching the goal is equal to the total expected reward. Assumption N holds since our problem is equivalent to a problem of minimizing expected total cost with nompositive costs per stage.

Tho mapping T defining the DI algorithin takes the form

<!-- formula-not-decoded -->

for any function J: (0.1] *, (0. ∞). Consider now the case where

that is, the game is unfair to the gambler. A discretized version of the caso where 1/2 ≤ p &lt; 1 is considered in Exercise 3.21. When 0 &lt; 1 &lt; 1/2, it is intuitively clear that. if the gambler follows a very conservative strategy and stakes a very small amount at each time. he is all but certain to loso his capital. For example, if the gambler adopts a strategy of betting 1/11 at each time, then it may be shown (see Exercise 3.21 or [Ash70), p. 182) that his probability of attaining the target fortune of 1 starting with an initial capital i/m. 0 &lt; i &lt; 1. is given by

--1

If 0 &lt; p &lt; 1/2, 2 louds to infinity, and i/a tends to a constant, the above probability tends to vero, thus indicating that placing consistently small bols is a bad strategy.

We are thus led to a policy that places largo bets and, in particular, the bold strategy whereby the gambler stakes at each time ki his entire fortune ta of just cnough to reach his goal, whichever is least. In other words, the bold strategy is the stationary policy d given by

<!-- formula-not-decoded -->

We will prove that the bold strategy is indeed an optimal policy. To this end it is suflicient to show that for evory initial fortune a € (0, 1) the value of the reward function J,+ (d:) corresponding to the bold strategy p* satisfies the sufficiency condition (ef. Prop. 1.4, Section 3. L)

or equivalently

<!-- formula-not-decoded -->

for all it € (0, 1) and a € (0, 2) 0 (0, 1 - ce).

By using the dofinition of the bold strategy, Bellman's equation

is writteu as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following lemma shows that dye is uniquely defined from these relations.

Lemma 5.1: For every p, with 0 &lt; p &lt; 1/2, there is only one bounded function on (0,1] satisfying Egs. (5.4) and (5.5), the function Ju*. Furthermore, Jy is continuous and strictly increasing on (0, 1).

Proof: Suppose that there existed two bounded functions J : 10, 1| - R and 12: 10, 1) → D such that J,(0) = 0, J, (1) = 1, i = 1,2, and

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let a be any real number with 0 ≤ = ≤ 1. Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for A = 1,2.... Then from Ers. (5.6) and (5.7) it follows (using y ≤ 1/2) that.

<!-- formula-not-decoded -->

Since 1(2) - Je(an) is bounded, it follows that 11(5) - 12(z) = 0, for otherwise the right side of the inequality would tend to oo. Since a € (0, 1] is arbitrary, we obtain d = Ja. Hence ye is the unique bounded function on (0, 1] satisfying Eqs. (5.4) and (5.5).

To show that. Je is strictly increasing and continuous, we consider the mapping Ta', which operates on functions d: 10,13 → 10.13 and is defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consider the functions Jo. TiJo,..., ThiJo,..., where do is the zero function /Jo(d) = 0 for all ¿ € [0, 1]]. We have

<!-- formula-not-decoded -->

Furthermore, the functions Th hi. Jo can be shown to be monotonically nondecreasing in the interval (0, 1). 'Hence, by Eq. (5.9), J,e is also monotonically nondecreasing.

Consider now for 1 = 0, 1,... the sets

<!-- formula-not-decoded -->

--1

It is straightforward to vorify that

<!-- formula-not-decoded -->

As a result of this equality and Ey. (5.9),

<!-- formula-not-decoded -->

A further fact that may be veridied by using indnction and Ers. (5.8) and (5.10) is that for any nomegative intogers li, 1 for which 0 ≤ h2-1 &lt; (hi + 1)2-' ≤ 1, we have

<!-- formula-not-decoded -->

Since any mumber in 0, 1] can be approximated arbitrarily closely from above and below by numbers of the form k2.", and since ye has been shown to be monotonically nondecreasing, it follows from Eg. (5.11) that J, is continuons and strictly increasing. Q.E.D.

Wo are now in a position to prove the following proposition.

Proposition 5.1: The bold strategy is an optimal stationary gambling policy.

Proof: We will prove the sullicioner condition

<!-- formula-not-decoded -->

In view of the continuity of dye established in the previons lemma, it it suflicient to establish Eg, (5.12) for all e € (0, 1| and « € 10, 4)n (0. 1 -x) that belong to the union UpenS, of the sets Sa defined by

<!-- formula-not-decoded -->

He will use induction. By using the fact that y-(0) = 0. J,+ (1/2) = p, and o, (1) = 1, we can show that By. (5.12) holds for all &amp; and a in So and Sy. Assume that Eq. (5.12) holds for all 1.4 € S,. We will show that it holds for all an ES'"+1.

For any don E Sunt with a € 10, 20 10, 1 -d, there are four possi bilities:

<!-- formula-not-decoded -->

2. 11&gt;1/2.

3. 1- 111151/250+11.

<!-- formula-not-decoded -->

We will prove Ba. (5.12) for each of these cases.

Case 1. Mone Sundo then 20C So, and 2011 Sno and by Tie induction hypothesis

<!-- formula-not-decoded -->

If a; + « ≤ 1/2. then by Eg. (5.5)

<!-- formula-not-decoded -->

and using By. (5.13), the desired relation By. (5.12) is proved for the case under consideration.

Case 8. 16.0.06. 5,11, Then (21 - 1) € 5, and 211 C So. and by 11e induction hypothosis

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and Ey. (5.12) follows from the proceding relations.

Case i. Using Ey. (5.5). we have

<!-- formula-not-decoded -->

Now we must have a ≥ f. for otherwise a &lt; band i +4 &lt; 1/2. Honce 2e ≥ 1/2 and the sequence of equalities can be contined as follows:

<!-- formula-not-decoded -->

Since pS(1-p), the last expression is greater than or equal to both

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Now for a, u € Si+1, and a. &gt; 1, we have (22 - 1/2) € Su. and (21 - 1/2) € 5, if (21 - 1/2) € [0, 1J, and (1/2 - 216) € S, if (1/2 - 2u) € (0, 1). By the induction hypothesis, the first or the second of the preceding expressions is nonnegative, depending on whether 21 + 24 - 1 ≥ 24 - 1/2 or 2.1 - 2u ≥ 21 - 1/2 610., «&gt; 4 or a &lt; 4). Hence By. (5.12) is proved for case 3.

Case 4. The proof resembles the one for case 3. Using y. (5.5), we have

<!-- formula-not-decoded -->

We must have ≤1 for otherwise a &lt; Gand t-a &gt; 4. Hence 0 ≤ 24 - 1 ≤ 1/2&lt;28-1/2 ≤ 1, and using Ey. (5.5) we have

<!-- formula-not-decoded -->

Using the preceding relations, we obtain

<!-- formula-not-decoded -->

These colations are contal to both

<!-- formula-not-decoded -->

aud

<!-- formula-not-decoded -->

Since 0 5 04(21 + 24-1) &lt; 1 and 0 &lt; 1,0 (26 - 2u) &lt; 1, Chese expressions are greater than or equal to both

<!-- formula-not-decoded -->

and

and the result follows as in caso 3. Q.E.D.

We note that the bold strategy is not the unique optimal stationary gambling stratogy. For a characterization of all optimal strategies. see (DuS65): p. 90. Several other gambling problems where strategies of the bold type are optimal are described in (DuS65), Chapters 5 and 6.

## 3.6 NONSTATIONARY AND PERIODIC PROBLEMS

The standing assumption so far in this chapter has been that the problem involves a stationary systom and a stationary cost per stage (except for tho prosence of tho discount factor). Problons with nonstationary srstom or cost per stage arise occasionally in practice or in theoretical studies aud are thus of some interest. It turns ouf that such problems can be converted to stationary ones by a simple reformulation. We can then obtain results analogons to those obtained carlier for stationary problems.

Consider a nonstationary system of the form

<!-- formula-not-decoded -->

and a cost. function of the form

<!-- formula-not-decoded -->

In these equations, for cach lied belongs to a space Sa. Mh bolongs to a space Co and satisfies us E Vers) for all thE Sa. and un belongs 10 a countable space Da. The soto Sa ChoCe(re). De may ditter from one stage to the next. The random disturbances my are characterised by probabilities Phl. buk). which depend on to and up as well as the time indox h. The set of admissiblo policies 11 is the set of all sequences a = leo Mend with 1h: Sh → Ca and pa(ra) E Unlich) for all ira E Sa and A = 0, 1.... The functions pa: Sa. X Ca X De. → D are given aud are assumed to satisfi one of the following three assumptions:

- -1

Assumption D': We have a &lt; 1. and the functions ga satisfy, for all hi = 0, 1,...,

<!-- formula-not-decoded -->

whero A/ is somo scalar.

Assumption P': The functions ge satisty, for all li = 0, 1....,

<!-- formula-not-decoded -->

Assumption N': The functions ge satisty, for all li = 0. 1,...,

<!-- formula-not-decoded -->

We will refer to the problem formulated as the nonstationary problem. (NST for short). We can got an idea on how the NSP can be converted to a stationary problem by considering the special case where the state space is tho same for each stage (ic., So. =S' lor all k). We consider an andmented stale

<!-- formula-not-decoded -->

where a ES, and l is the time index. The new state space is 5=5 x k, where A denotes the set of nonnegative integers. The angmented sestem evolves according to

Similarl, we can deline a cost por stage as

It is ovidont that the problem corresponding to the augmented system is stationary. If we restrict altention to initial states to € S × (0}, it can be soon that this stationars problem is canivalent to the NSP.

Let us now consider the more general case. To simplify notation, we will assmmo that the state spaces So, i = 0. Lo.... the control spaces Ci,

¿ = 0. 1...., and the disturbanco spaces D,. i. = 0, l..., are all mutually disjoint. This assumption does not involve a loss of generality sinee. it necessars. we may relabel the olomonts of 5,. Co. and D, without allecting the structure of the problem. Define now a new state spaceS. a now control space C, and a new (countable) disturbance space D) by

Introduce a now (stationary) system

<!-- formula-not-decoded -->

where th ES. Me € C, d% € D. and thosestom mction f: SxCxD-S is defined by

<!-- formula-not-decoded -->

For triplots (P.a. t). where forsome i = 0. I.., vohared € S,, but i dC, or @ &amp; D,. the dofinition of f is immaterial; any definition is adequate for our purposes in view of the control constraints to be introduced. The control constraint is taken to be й E Uf&amp;) for all i e 5, where U() is defined by

<!-- formula-not-decoded -->

The disturbance w is characterized by probabilities P(wlf, d) such that

<!-- formula-not-decoded -->

Furthermore for any 1, € Done, E S,. 4, E Ci, i = 0. Lo... we have

We also introduce a new cost finetion

<!-- formula-not-decoded -->

where the (stationary) cost per stage g: 5'x C' X De d is delined for all i = 0. 1...

For triplets (7, ü, ữ), where forsome i = 0. 1,..., we have a € S; but i &amp; Ci or a &amp; Di any definition of g is adequate provided luci.a.w &lt; 1 for all (20.w) when Assumption D' holds. Os gra, d) when D holds and

1(2. 0. m) ≤ 0 when N' bolds. The set, of admissible policies It for the new problem consists of all sequences d= dio Mood, where The St Cand MA(P) E U(P) for all i E Sand ki = 0, 1...

The construction given defines a problem that clearly tits the framework of the infinite horizon total cost problem. We will refer to this problem as the stationary problem (SP for short.).

It is important to understand the nature of the intimate connection betweon the NSP and the SP formulated here. Let 7= Ko.p... Do an admissible policy for the NSP. Also, let * = 1ão, M...Y be an admissible policy for the SP such that

<!-- formula-not-decoded -->

Let do € So Do Cho initial State for the NSP and consider the samo initial state for the SP (i.e.. To = do € So). Then the sequence of states fiel gonorated in the SP will satisty &amp;, E S,, i = 0.1.., with probability 1 (ir, the system will move from the set So to the sol So, Then to S2, etc:, just as in the NSP). Furthermore, the probabilistic law of generation of states and costs is identical in the NSP and the SP. As a result, it is casy to soo that for any admissible policies a and &amp; satisfying Ry. (6.4) and initial states do, to satistying do = do € So, the sequence of generated states in the NSP and the SP is the same (o, =7,, for all i) provided the generated disturbances n, and d, are also the same for all i (w; = w,, for all 6). Parthermore, if a and a satisfy Ey. (611), we have 1(10) = (70) if do=do € So. Let as also consider the optimal cost. functions for the NSP and the SP:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then it follows from the construction of the SP that.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

if.?o=d, ES,. Note that in this equation, the right-band side is defined in ters of the data of the NSP. As a special case of this equation, we obtain

<!-- formula-not-decoded -->

where. for all i = 0,1.....

Thus the optimal cost function J* of the NSP can be obtained from the optimal cost function fe of the SP. Furthermore, in - Mom... is in optimal policy for the SP, then the policy n* = 4g,d... defined by

<!-- formula-not-decoded -->

is an optimal policy for the NSP. Thus optimal policies for the SP yield optimal policies for the N.SP via Kg. (6.8). Another point to be noted is that if Assumption D' (P', N') is satisfied for the NSP, then Assumption D (P, N) introduced carlier in this chapter is satisfied for the SP.

These observations show that one may analyze the NSP by means of the SP. Every result given in the preceding sections when applied to the Sl' yields a corresponding result for the NSP. We will just provide the form of the optimality equation for the NSP in the following proposition.

Proposition 6.1: Under Assunption D' (P', N'), there holds

<!-- formula-not-decoded -->

where for all i = 0, 1,..., the functions J*(, i) map Si into R (10, 00), [-∞, 0]), are given by Eq. (6.6), and satisfy for all zi € Si and i =

<!-- formula-not-decoded -->

Under Assumption D' the functions J* (., ¿), ¿ = 0, 1,..., are the unique bounded solutions of the set of cquations By. (6.9). Furthermore, under Assumption D' or P', if / (2;) € Vi(di) attains the minimum in Eq. (6.9) for all xi E Si and i, then the policy 7* = {out,...} is optimal for the NSP.

## Periodic Problems

Assume within the framework of the NSP that there exists an integer * ≥ 2 (called the period) such that for all integers i and j with li- il = mp. m. = 1.2.... we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We assume that the spaces Si, C;, Di, i = 0, 1,...,p - I. are mutually disjoint. No define now state, control, and disturbance spaces by

<!-- formula-not-decoded -->

The optimality equation for the equivalent stationary problem reduces to the system of pequations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These equations may be used to obtain (under Assmoption D' or P) a .., whenover the 1. When all spaces involved are linite, an optimal policy may be found by means of the algoritons of Scotion 1.3. appropriately adapted to the corresponding SP.

## 3.7 NOTES, SOURCES, AND EXERCISES

cost per stago were first analyzed systematically in (DuS65), (Bla65), and (Str66). An extensivo treatment, which also resolves the associated measurability questions, is (Be578). Sufficient conditions for convergence of the value iteration method under Assumption P (ef. Props. 1.6 and 1.7) were derived independently in (Ber77) and (Sch75). The former reference also de rives necessary conditions for convorgence. Problems involving convexity

We have bypassed a maber of complex theoretical issues relating to stationars policies that historically have played an important role in the development of the subjeet of this chapter. The main question is to what extent is it possible to restrict altcution to stationary policies. Much theoretical work has been done on this question (Be579), (Bla65), (Bla70], (DuS65), (10178). (FeS83), [Feioal, (Pei026), (Om60), and some aspects are still open. Suppose, for example, that we are given an c &gt; 0. One issue is whother there exists an c-optimal stationary policy. that is, a stationary police de such that.

assumptions are analyzed in (Ber7:31.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The answer is positive under any one of the following conditions:

1. Assumption l holds and a &lt; 1 (soo lexerciso 3.8).
2. Assumption N holds, Sis a limitesoled LandfCe) se no for all " E.$ (sco Exercise 3.11 or (Bla65), (Bla70), and [Orn69)).
3. Assumption N holds. S is a countable sod. a l. and the problem is
4. doterministic (soo (130579)).

The answer can be negative under any one of the following conditions:

1. Assmoption P' holds and a - 1 (sco Exerciso 3.8).
2. Assimuption N holds and a &lt; 1 (see Exerciso 3.11 or (Bo579)).

The existence of an optimal statiouary policy for stochastic shortest path problems with a finito state space. but under somewhat diflerent assump-

Anothor issue is whether there exists an optimal stationary policy whenever there exists an optimal policy for each initial state. This is tric under Assumption P (soo Exercise 3.9). It is also true (but very hard to prove) under Assumption Nif f(e) &gt; -2 for all eE Son = 1. and the disturbance space D is countable (Bla70). (DuS65). (Or69). Simple 1ostate examples can be constructed showing that the result fails to hold it a = 1 and J*(p) = -a for some stato d (soo Exercise 3.10). Howevor. these examples rely on the presence of a stochastic clement. in the problem. If the problem is deterministic, stronger results are available: one can find an optimal stationary policy if there exists au optimal policy at each initial state and cither a = lora &lt; land (r) &gt; -x for all &amp; ES. These results also require a difficult proof (BeS79.

tions than the ones of Section 2.1 is established in (koi92b.

The gambling problem and its solution are taken from (Du565). In

Bilds) a surprising property of the optimal reward finetion fo for this woo, so is suit celes, cowhere silentall

## EXERCISES

- -1

respectively, let the system equation be

<!-- formula-not-decoded -->

where c is the discount factor, and let

be the cost per stage. Show that for this deterministic problem, Assumption P holds and that J'(x) = 00 for all &amp; E S, but (T% Jo) (0) = 0 for all hi [Jo is the zero function, Jo(x) = 0, for all a € S).

3.2

Let Assumption P hold and consider the finite-stato case S' = D = {1,2,.., 2}, rr = 1, d'h+1 = 1k. The mapping l' is represented as

<!-- formula-not-decoded -->

where po (e) denotes the transition probability that the next state will be j when the current state is i and control &amp; is applied. Assume that the sets 1(i) are compact subsets of 12" for all i, and that pi,(a) and gi, «) are continuous on U(i) for all i sund j. Show that lime-x (7% Jo)(1) = .*(i), where Joi) = 0 for all i = 1,...,n. Show also that there exists an optimal stationary policy.

3.3

Consider a deterministic problem involving a linear system

<!-- formula-not-decoded -->

where the pair (il, B3) is controllable and ta ED", Ma E". Assume no constraints on the control aud a cost per stage g satistving

Assume furthermore that g is continous in &amp; and a, and that yen, l,) → 00 if find is bounded and ju, l → oo.

- (a) Show that for a discount factor a &lt; 1, the optimal cost satisfies 0 ≤ I'(e) &lt; 20, for all e E W". Earthermore, there exists an optimal stationary policy and

<!-- formula-not-decoded -->

- (0) Show that the same is true, except perhaps for f(x) &lt; oc, when the system is of the form th+ = fUn uh), with f:%" X'→ l" being i continuons finction.
2. (e) Prove the same results assuming that the control is constrained to lie in a compact set UE D" Us) = U for all el in place of the assumption ain.,) tox if fond is bounded and luull → oo. Hint: Show that. Th do is real valued and contimons for evory hi. and use Prop. 1.7.

Sec. 3.7 Notes, Sources, and Exercises

3.4

Under Issumption P. let y bosuch that tor all rES. no) el'(r) and

where r is some positive scalar. Show that, if a &lt; 1.

lint: Show that (TU)(e) ≤ 0(0) + Elor. Alternatively, let d' = d+(0/(8-10)) 01 show-tbat 'Cud' Ko', und use Con. 7.1.1.

3.5

Under Assumption P or N, show that if or &lt; 1 and f': S→ l is a bounded. function satislying d'= 1J', then d'=". Hunt: Under P, let. o bo a scalar such that. J* +re D'. Argue that de ≥ l' and use Prop. 1.2(a).

3.6

He want to lind a scalar sequence fool,...y that sitisties SEll 5r «A≥ 0, for all li, and maximizes Logue), where e &gt; 0769) &gt; 6 for all « &gt; 0, 9(0) = 0. Assume that, g is monotonically nondecreasing on (0, x). Show that the optimal value of the problem is d'(e), where d" is a monotonically nondecreasing finction on 0, ∞0) satisfving J*(0) = 0 and

3.7

Let Assumption P hold and assume that a* = tani....? E ll satisties J= Thod" for all hi. Show that a" is optimal, ic. d"a = J".

3.8

Under Assumption P, show that givene &gt; 0, there exists a policy nell such that de(r) ≤ 0(e) to for allo ES, and that for a &lt; 1 the poliey no can bo takon stationary. Give an example where a =1 and for cach stationary policy a we have da(e) = do, while d'(r) = 0 for all 2. Hint: See the proof of Prop. 1.1.

3.9

linder Assumption P, show that if there exists an optimal policy (a policy x' € 1l such that Dae = 0*), then there exists an optimal stationary policy.

## 3.10

liso the following counteresample to show that the result of Exercise 3.9 may fail to hold under Assumption Nif '(6) = -x forsome € S. Lot s l 10.11.000.1.2) 1. 0(0.11.1) 103, 11 010-0.0= Handpa= 1|8=1.1) = 1. ShowThat l'(0) = -x. I'(1) == 0 and that the admissible nonstationary policy fragi.. with rolatod analysis).

3.11

Show that the result of Exercise 3. holds nader Assumption NilS is a finite sot, a = 1, and 1(8) &gt; -x for all &amp; ES. Construct a counteresamplo to show that the resoll can tail to hold it is contable and a &lt; 1 feron if d(r) &gt; -x for all &amp; € SJ. Hint: C'onsider an integor N such that the Nestage optimal cost. do satisfies dote) ≤ 0(0) te for all r. For a counterexample, see (830579).

## 3.12 (Deterministic Linear-Quadratic Problems)

C'onsider the doterministie linear-quadratic problem involving the system

and the cost

He assume that do is positive definite sommetrie. O is of the form C"C, and the pairs (.. 1). 6.0. (*) are controllable and obserrable, respectivoly. Nor the Theory of Sections Ail of Vol. Land sit to show that the stationary policy pe with

<!-- formula-not-decoded -->

is optimal. where A' is the unigue positive somidefinite symmotric solution of the algebraie Riccati equation (of. Section 1. 1 of Vol. 1):

Provide de similar result under an appropriate controllabilite assumption lor the case of a periodie deterministic linear system and a periodic quadratic cost. (of. Section 3.6).

3.13

Consider the linear-quadratic problem of Section 3.2 ith the oul dillorone that the disturbances de have zero mean, but their covariance matrices are nonstationary and uniformly bounded over k. Show that the optimal control law remains unchanged.

## 3.14 (Periodic Linear-Quadratic Problems)

Consider the linear system

<!-- formula-not-decoded -->

and the quadratic cost

<!-- formula-not-decoded -->

where the matrices havo appropriate dimensions. De and le are positive somidetinite aud positive definite symmetrie, respectivels for all he and 0&lt; a &lt; 1. Assumo that the sestom and cost are periodic with period p (ot. Sortion 3.6). that the controls are unconstrained. and that. the disturbances are independent, and have zero mean and tinito covariance. Assume father that the following (controllability) condition is in ollect.

For any state To, there exists a finite sequence of controls fay. To...it, Y such that Tree =0. where Frey is gonerated by

Show that there is an optimal periodie policy a of the form

where Modi,...do 1 are givon by

<!-- formula-not-decoded -->

and the matrices do. hi. Ago satisfy the coupled set of galgebir kiecati equations givon for i = 0, 1.....p - 1 by

with

## 3.15 (Lincar-Quadratic Problems - Timperfect State Information)

C'onsider the lincar-quadratic problem of Soction 3.2 with the difference that. the controller, instead of having portect state information, has access to measuroments of the forn

<!-- formula-not-decoded -->

As in Soction 5.2 of Vol. I, the disturbances on are independent and have identical statisties, soro mean, and finito covariance matris. Assume that for every admissiblo policy a tho matrices

<!-- formula-not-decoded -->

aro mitorly bonoded overk, where te is the information voctor dotined in Sortion 52 of Vol. I. Show that the stationary polies a given by

is optimal. Show also that the same is troo if we and te are nonstationary with soro mean and covariance matrices that are uniformly bounded over hi. Hul: Combmo the theory of Sections 5.2 of Vol. I and 3:2.

## 3.16 (Policy Iteration for Linear-Quadratic Problems [Kle68])

Consider the problom of Section 3.2 and let. Do be an m X o matrix such that. the matrix (4 + Blo) has cigonvalues strictly within the unit circle.

- (a) Show that the cost corresponding to the stationary policy do, where 110(0) = 10d is of the form

<!-- formula-not-decoded -->

whoro die is a positivo somidefinite semmetric matrix satisfying the (lincar) cquation

- (6) Lot p(e) attain the miniman for each in the expression

<!-- formula-not-decoded -->

Show that for all t we have

<!-- formula-not-decoded -->

whore Ai, is somo positive somidefinite symmetric matrix.

- (0) Show that the police iteration process described in parts (a) and (b) yields a sequenee: Sed such that.

where A is the optimal cost matrix of the problem.

## 3.17 (Periodic Inventory Control Problems)

In the inventory control problem of Section 3.3. consider the case where the statistics of the demands wn, the prices ca, and the holding and the shortage costs are periodic with period p. Show that there exists an optimal periodic policy of the form a* = 1a.... airlo,.... 1,,....%.

<!-- formula-not-decoded -->

where So..., See are appropriato scalars.

## 3.18 (HeS84]

Stron that the critical level S" for the inventory problem with zero fixed cost. of Section 3.3 minimizes (1 - a)cy + Ly) over y. Hint: Show that the cost can bo expressed as

<!-- formula-not-decoded -->

Where 14 = 8A t 114(11).

3.19

Consider a machine that may broak down and can be repaired. When it operates over a time unit, it costs -1 (that is, it produces a benofit of 1 unit.). and it may break down with probability 0.1. When it is in the breakdown mode, it may be repaired with an effort a. The probability of making it operative over one time unit is then a, and the cost is Cal. Determine the optimal repair effort over an infinite time horizon with discount factor a &lt; 1.

## 3.20

Let 2o, al,... Do a sequence of independent and identically distributed random variables taking values on a finite set Z. We know that the probability distribution of the ca's is one out of a distributions for..fo, and we are trying to decide which distribution is the correct one. At each time h after observing ero.,t, we may either stop the observations and accept one of the a distributions as correct, or take another observation at a cost r &gt; 0. The cost. for accepting f, given that f, is correct is Lo, i,j = 1..,n. We assumo Lo, &gt; O for i # 6, Lo, = 0, ¿ = 1...a. The a priori distribution of fi....A, is denoted

Show that the optimal cost d'(16) is a concare timotion of a. Characterize the optimal acceptance regions and show how they can be obtained in the limit by means of a value iteration method.

## 3.21 (Gambling Strategies for Favorable Games)

A gamblor plays a gamo such as the one of Section 3.5, but where to probability of winning a satisties 1/24 pe 1. His objective is to roach a final fortomo n, where a is an intogor with a 2 2. His initial fortune is an intoger i with O&lt; i. &lt; 11, and his stake at time hi can take only intoger values th satistying 05 1a 504. 05 11 2 1- de, Where de is his fortne at time ki. Show that, the strategy that always stakos one unit is optimal (ic., /"(r) = 1 for all integers &amp; with O&lt;re a is optimal). Hiul: Show that if p € (1/2. 1),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for soo (Ash70. p. 182, for a proof). Then use the sufliciones condition of Prop. 1.1 in Sootion 3.1.

## 3.22 (Sch&amp;1]

Considor a network of a queues whereby a customer at quone i upon completion of sorvice is ronted to quone with probability peg, and exite the network with probability 1-2, 8.,. For each quone i denote:

- n,: the external costomer arrival rate,

L: the average customer service time.

1,: the castomer departure rate,

- «,: the total customor arrival rate (sum of external rate and departure rates from upstream queues weighted by the corresponding probabilitios).

We have

<!-- formula-not-decoded -->

and wo assume that. any portion of the arrisal rate o, in excess of the service rato ye, is lost: so the departure rate al quono o satislies

<!-- formula-not-decoded -->

there is a cmone i with 1-9? ww.that/s @tor at least one/, and that forevery queue is with ≥ 0, preceding equations are mique and can be fond by value iteration or policy iteration. Hind: This problem does not quite fit our tramowork because we and it wad fe antil way i in persible to curry unt sin

## 3.23 (Infinitr Time Reachability (Ber71), (Ber72])

Consider the stationary sostem

<!-- formula-not-decoded -->

whore tho disturbance space D is an arbitrary (not necessarily contable) set. The disturbances we can take values in a subser Words) of l that may depond once and n. This problem deals with the following question: Given a nonempty subsol l of the stato space S, moder what conditions does there exist an admissible policy that koops the state of the (olosod-loop) system

<!-- formula-not-decoded -->

in those l' for all l and all possible valuos an E I Cok fe(ra)). that is.

<!-- formula-not-decoded -->

The set l' is said to be enfinitoly reachable if there exists an admissible police 19o.90...) and some initial state to eX' for which the above rolations aro satisfied. It is said to bo strongly reochable if there exists an admissiblo police feude...&amp; such that for all initial states do eX The abore relations are satisfied.

C'onsider the tunction k mapping any subsod % of the state spaceS into a subset R(%) of s dofined by

- (a) Show that tho sot l' is strongly reachable if and only it K(X) = .&amp;.
- (6) Given f. consider the set to defined as follows: do Cd" if and onl if co E d'and there exists an admissible policy feo pho. such that that Eys. (7.1) and (7.2) are satisfied when to is taken as the initial state of the sostem. Show that a set dis intimitely reachable if and only if it contains a nonempty strongly reachable set. Farthermore. the largest such set is X' in the sense that ' is strongly reachable whenever nonempty. and if&amp;ed is another strongly reachable set, then fe.
- (e) Show that if l' is infinitely reachable, there exists an admissible stationary policy o such that if the initial state do belongs to X*. then all subse neut states of the closed-loop system can = f(ad./(th).na) aro guaranteed to belong to f'*.
- (1) Given d. consider the sots Ri(d), l = 1.2,.. whore Ak(d) donotes the set obtained after l applications of the mapping RonX. Show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Show that, if there exists an indes de such that for all ef and li &gt; Ti cho set De(r) is a compact salsod of a lindisteam space, thon

## 3.24 (Infinite Time Reachability for Linear Systems)

Consider the linear stationary systom

where tk E", un EMi', and Wa ED", and the matrices A, B. and G are known and have appropriate dimensions. The matrix A is assumed invertible. The controls un and the disturbances we are restricted to take values in the ellipsoids U = full' Ru ≤ 1} and W = fw/w'Qw ≤ 1}, respectively, where R and Q are positive definito symmetric matrices of appropriate dimensions. Show that, in order for the ellipsoid X'= fola' K. &lt; 1%, where A is a positive dotinite symmotric matrix, to be strongly reachable (in the terminology of Exercise 3.23), it is suflicient that for some positive definite symmetric matrix Al and for some scalar 6 € (0, 1) we have

<!-- formula-not-decoded -->

Show also that if the above relations are satisfied, the linear stationary policy 11", where p" (v) = Lad and

<!-- formula-not-decoded -->

achieves reachability of the clipsoid d= felekeS 16. Partbormore, the matrix (4 + BL) has all its cigenvalues strictly within the mit circle. (For a proof together with a computational procedure for finding matrices k satisfying the above, soe (Ber71] and (Ber721).)

## 3.25 (The Blackmailer's Dilemma)

Consider Examplo 1.1 of Section 2.1. Here, there are two states, state I and a termination state 1. At state 1, we can choose a control a with 0 &lt; 4 &lt; 1; we then mose to state fal no cost with probability p(a), and stay in state 1 at a cost -a with probability 1-ya).

- (a) Lot pa) en. Nor this case it was shown in Example 1.1 of Section 2.1. that the optimal costs are d*(L) = - x and f*(0) = 0. Furthermore, it was shown that there is no optimal stationary policy, although there is an optimal constationary polics. Find the set of solations to Bollman's equation and vorily the result. of Prop. 1.2(b).
- (6) 10l pa) = 0. Find the set of solutions to Bollman's equation and uso Prop. 1.2(b) to show that the optimal costs are f'(1) = -l and J"(1) = 0. Show that there is no optimal policy (stationary or not).

4

## Average Cost per Stage Problems

| Conteuts   |
|------------|

The results of the preceding chapters apply mainly to problems where the optimal total expected cost is finite either because of disconting or because of a cost-free absorbing state that the system eventually enters. In many situations. however. discounting is inappropriate and there is no madurad cost-free absorbing state. Ta such situations it is often meaningful to optinase the average cost per stage. to be defined shortly. In this chapter, we discuss this type of optimization. with an emphasis on the case of a finite-state Markor chain.

An introductory analysis of the problem of this chapter was given in Soction 7. of Vol. I. That analysis was based on a comection between the average cost per stage and the stockastic shortest path problem. While this connection can be further extended to obtain more powerful results (sco Exercises 1.13-1.16). wo develop here an altorativo line of amalysis that is based on a relation with the disconted cost problem. This relation allows as to use discounted cost results, derived in Soctions 1.2 and 1.3, in order to conjecture and prove results for the average cost problem.

## 4.1 PRELIMINARY ANALYSIS

Lot as formulate the problem of this chapter for the case of finite state and control spaces. No adopt the Markor chain notation used in Section 1.3. lo particular, we denote the states by lo..n. To cach state i and control a there corresponds a set of transition probabilities pr, (a), j = I...,". Bach time the system is in state i and control a is applied. we incur in expected cost gi. a), and the system moves to state j with probability P. (4). The objective is to minimize over all policies a = 400.19....y with 14(1) E U(i), for all i and li, the avorage cost per slago t

<!-- formula-not-decoded -->

for any given initial state to.

1 When the limit defining the average cost is not known to exist. we use instead the definition

<!-- formula-not-decoded -->

We will show. however. as part of our sabsequent analysis that the limit exists al keast for those policies a that are of interest.

As in Section Is we use the following shorthand notation for a sli tionary policy y

<!-- formula-not-decoded -->

Since the (i. ith element. of the matrix Do (P, to the lith power) is the hi-stop trambition probability Pore rileo = 1) corresponding 1o p. it can be soon that

<!-- formula-not-decoded -->

An important result regarding transition probability matrices is that the limit in the proceding equation exists. We show this fact shortly in the context of a more goneral result, which establishes the connection between the average cost por stage problem and the discounted cost problem.

## Au Overview of Results

While the material of this chapter does not rely on the analesis of the average cost problem of Scction 7.1 in Vol. I, it is worth summarizing some of the salient features of that analysis (see also Exercises 1.13-1.16). 14 assuned there that there is a spocial state, by convontion stato n. which is recurrent in the Markor chain corresponding to each stationary policy. li we consider a sequonce of generated states. and divide it into cooles marked by successivo visits to the spocial state n. we see that each of the cycles can be viewed as a state trajectory of a corresponding stochastic shortest path problem with the tormination state being essentially n. More precisely: this stochastie shortest path problem has states 1.2,11. Dus an artificial termination state t to which we nove from state with transitiou probability pu(4). The transition probabilities from a state i to a stato ¿# n are the same as those of the original problem. while p(a) is soro. For any scalar d. we considered the stochastic shortest path problem with expected stage cost gfi. a) - A for cach state i = 1....,n. We then argued that il no fix the expocted stago cost incurred at state i to be

where d* is the optimal average cost por stage starting from the special state n. then the associated stochastic shortest path problem becomes ssentially equivalent. to the original average cost por stage problem. Tinthermore. Bellman's equation for the associated stochastic shortest path

.. .л.

problem can be viewed as Bellman's equation for the original average cost. per stage problem. Based on this line of analysis, we showed a number of results, which will be strengthened in the present chapter by using dillerent. methods. In summary, these results are the following:

- (a) The optimal average cost per stago is independent of the initial state. This property is a generic feature for almost all average cost problems of practical interest.
- (b) Bellman's equation takes the form

<!-- formula-not-decoded -->

where 1°(n) = 0, 1* is the optimal avorago cost per stage, and h*(i) has the interprotation of a relativo or diflorential cost for each state ¡ (it is the minimum of the dillerence betwoon the expected cost to reach o from i for the first time and the cost that. would be incurred if the cost per stage was the average 1*).

- (0) 'There are versions of the value itoration, policy itoration, adaptive aggregation, and lincar programming methods that can be used for computational solution under reasonable conditions.

No will now provide the fondation for the analysis of this chapter by developing the connection between average cost and discounted problems.

## Relation with the Discounted Cost Problem

Let us consider the cost. of a stationary polies ye for the corresponding a-discounted problem. It is given by

<!-- formula-not-decoded -->

To get a sonse of the relation with the average cost of ye, we note that this latter cost is writton is

<!-- formula-not-decoded -->

Asstuning that the order of the two limits in the right-hand side above can be interchanged, we obtain

<!-- formula-not-decoded -->

The formal proof of the above relation will follow as a corollary to the next, proposition.

Proposition 1.1: For any stochastic matrix P and a € (0,1), there holds

<!-- formula-not-decoded -->

where O(1 - al) is an a-dependent matrix such that

<!-- formula-not-decoded -->

and the matrices P* and H are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It will be shown as part of the proof that the limit in Ey. (1.4) and the inverse in Eg. (1.5) exist. Furthermore, p+ and II satisfy the following cquations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: From the matrix inversion formula that expresses each enter of the inverse as a ratio of two determinants. it is seen that the matrix

<!-- formula-not-decoded -->

can be expressed as a matrix with elements that are either coro or tractions whose munerator and denominator are polynomials in a with no common divisor. The donominator polynomials of the nonzero dements of 11(a) cannot have 1 as a root, since otherwise some dements of l/(a) would lond to indinity as de 1; this is not possible, because from lia. (1.1) for ¡my 14, we have (1 - 66) -' 11(00) 94 = (8-00P) -11 = dog and KonG) ≤ (1-0) ' mas, log(i)l. implying that the absolute values of the coordinates o A(0) de are bounded by max, lan(i)| for all a &lt;1. Therefore, the (i.juh demont of the matrix A/(a) is of the form

<!-- formula-not-decoded -->

where go Good = Ronade and G. i = Jon,%, are scalaus such that 5 f1 for i = 1.....g. Defino

<!-- formula-not-decoded -->

and det 1l be the matrix having as (i. ith clement the lat derivative of -m,(0) evaluated al. c = 1. By the Ast. order Taylor expansion of the demonts of mog(a) of A/(a), we have for all a in a neighborhood of a = 1

<!-- formula-not-decoded -->

where 0(1 - 02) is an a-dopondont matris such that.

<!-- formula-not-decoded -->

Multiplving Eg. (1.10) with (1-a) 1, we obtain the desired rolation (1.2) although, we have got to show that to and 1l are also given by bos. (1.1) and (1.5), respectivoly.

We will now show that De as defined by Ey. (1.9), satisties les. (1.6). (1.5), (1.7), (1.8). and (1.1). in that order.

We have and

<!-- formula-not-decoded -->

Subtracting theso two cquations and rearranging terms, we obtain

<!-- formula-not-decoded -->

By taking the linit as a → Land using the definition (1.9), it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Also. Do reversing the order of (1-al) and (1 -al). 1 in les. (1.11) and (1.12). it follows similarly that 17 = 0. From 00 = Po. we also obtain (1 taking the limit as a - 1 and by using Ed. (1.9), wo have Pr= fp. Thus Da. (16) has boon proved.

He have: using 14. (1.0). 08-1892-12 1%. and similarly

<!-- formula-not-decoded -->

Theretore.

<!-- formula-not-decoded -->

On the other hand, from Eg. (1.10), we have

<!-- formula-not-decoded -->

By combining the last two equations, wo obtain

<!-- formula-not-decoded -->

which is 1g. (1.5).

From Eg. (1.5), we obtain

or. using Ey. (1.6).

<!-- formula-not-decoded -->

Multiplying this relation by Pe and using y. (1.6), we obtain PH = 0. which is Ey. (1.7). Equation (1.8) then follows from dig. (1.13).

Multiplying By. (1.8) with Pl and using By. (1.6), we obtain

Adding this relation over dO..' - 1. we have

<!-- formula-not-decoded -->

Dividing by Nund taking the limit. is N → 20, we obtain Ey. (1.4). Q.E.D.

Note that the matrix P* of Ey. (1.1) can be used to express concisely the average cost vector &amp; of any Markov chain with transition probability matrix P and cost vector g as

<!-- formula-not-decoded -->

To interpret, this equation, note that, we may view the ith row of pr as a vector of steady-state occupancy probabilities corresponding to starting at state i; that is, the lith clement p, of Pr represents the long-term fraction of time that the Markov chain spends at state j given that it starts at state ¿. Thas the above cquation gives the averago cost. por stage fi), starting from state i, as the sum S*" = Pyaj of all the single-stage costs!, weighted by the corresponding occupancy probabilities.

From Ey. (1.1) and Prop. 1.1, we obtain the following relation between a-discounted and average cost corresponding to a stationary policy:

Proposition 1.2: For any stationary policy &amp; and a € (0, 1), we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is the average cost vector corresponding to f, and hu is a vector satisfying

<!-- formula-not-decoded -->

Proof: Equation (1.11) follows from lus. (1.1) and (1.2) with the idenlifications P = Pn, 1 = Po. and l, = Hqn. Equation (1.15) follows by multiplying Ry. (1.8) with go and by using the same identifications. Q.E.D.

In the next section we use the procoding results to establish Bellman's equation for the adorage cost per stage problem. As in the earlier chapters, this ernation insolves the mappings T and Tw. which take the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 4.2 OPTIMALITY CONDITIONS

Our first result introduces the analog of Bollnan's equation for the case of equal optimal cost for cach initial state. This is the case that normally appears in practice, as discussed in Section To of Vol. I. The proposition shows that all solutions of this equation can be identified with the optimal average cost and an associated differential cost. However. it provides no assurance that the cquation has a solution. For this we nerd further assumptions, which will be given in the sequel (soo Prop. 2.6).

Proposition 2.1: If a scalar 1 and an a-dimensional vector le satisfy

<!-- formula-not-decoded -->

or equivalently

<!-- formula-not-decoded -->

then 1 is the optimal average cost per stage J*(i) for all i.

<!-- formula-not-decoded -->

Furthermore, if ye* (i) attains the minimum in By. (2.1) for each i, the stationary policy * is optimal, that is, 1,-(i) = 1 for all i.

Proof: Let a = 100-11.... bo any admissible policy and let A boa positive integer. We have. from Ey. (2.2).

By applying Tire to both sides of this relation. and by using the monotonicity of Too and 18y. (2.2). we see that.

Contiming in the same manner. we finally obtain

<!-- formula-not-decoded -->

- -1

with equality il each pha di = 0. I....N= 1. attains the minimum in Eg. (2.1). As discussed in Section 11. (Tint,, ...Ton ,le)(i) is equal do the N-stago cost corresponding to initial state i. policy 110.14.. 1H0-18. and torminal cost function h: that. is,

Using this rolation in Ey. (2.1) and dividing by N, we obtain for all i

By taking the limit is N -ex, we sor that

<!-- formula-not-decoded -->

with equality il pati). li = 0. Do... attains the minimum in by. (2.1). Q.E.D.

Note that the proof of Prop. 2. 1 carries through even if the state space and control space are infinite as long as the tinetion &amp; is bounded and the minimum in the optimality equation (2.1) is attained for each i.

states i and j we have

<!-- formula-not-decoded -->

which by subtraction violds

for an i. (99)(1) is the optinal Nestage expected cost starting at i when the terminal cost. lection is k. Thus, according to the preceding equation, 4(i) - ACi) represents, for every N, the dillerence in optimal Nsage expected cost one to starting at state i rather that starting at state j. Based on this interpretation. we refer to has the differential or relative cost vector. (An alternative but similar interpretation is givon in Section

Now given a stationare polies ye, we mar consider. as in Section 1.2, a problem where the constraint set it is replaced by the set N(i) = le(i));

that is. ((;) contains a single element. the control yi). Since Then we would have oudy one admissible policy. Me police a. application of Prop. 2.1 sickle the following corollary:

<!-- formula-not-decoded -->

or equivalently then

<!-- formula-not-decoded -->

## Blackwell Optimal Policies

It turns out that the converse of Prop. 2.1 also holds: that is. if for some scalar A we have J*(i) = A for all i = looK. Then A togother with a vector he satisfies Bellman's equation (2.1). Moshow this be introducing the notion of a Blackwell optimal policy, wlich was first formulated in (81a62). together with the line of analysis of the prosent section.

Definition 1.1: A stationary policy a is said to be Blachwell optimul if it is simmitanconsly optimal for all the a-discounted problems with a in au interval (ar. 1), where a is some scalar with O &lt; T &lt; 1.

The following proposition providos a asofal characterization of Blacke well optimal policies, and essentially shows the converse of Prop. 2.1.

Proposition 2.2: The following hold true:

- (a) A Blackwell optimal policy is optimal for the averago cost problem within the class of all stationary policies.
- (b) There exists a Blackwell optimal policy:

Proof: (a) If 4* is Blackwell optimal, then for all stationary policies y

and do in an interval (o, 1) we have Joo Slow. Equivalently, using Ex. (1.1 1),

<!-- formula-not-decoded -->

By taking the limit as a - 1. we obtain d, &lt; o,.

(1)) From Eq. (1.1), we know that, for cach fe and state i, don(i) is a rational function of ee, that is, a ratio of two polynomials in a. Therefore, for any to policies pe aud de the graphs of dog i) and Jan(i) either coincide or cross only a finite nmber of times in the interval (0, 1). Since there are only a finite maber of policies, we conclude that for each state i there is a policy de and a scalar a, e (0. 1) such that yo is optimal for the adisconned problem for a &amp; (ar. 1) when the initial state is i. Consider the stationary policy defined for cach i by p(i) = y'(i). Then p*(i) attains the minimum in Bellman's equation for the a-discounted problem

<!-- formula-not-decoded -->

for all i and for all a in the interval (max, a,. 1). Therefore, p* is a stationary optimal police for the a-discounted problem for all a in (max; ai, 1), implying that at is Blackwell optimal. Q.E.D.

No noto that. the converso of Prop. 2.2(a) is not true; it is possible that a stationary averago cost optimal policy is not. Blackwell optimal (see Exerciso 4.6). We mention also that one can show a stronger result than Prop. 2.2(b). namolo that a Blackwell optimal polier is average cost optimal within the class of all policies (not just those that are stationary; see Exerciso .1.7).

The next proposition provides a useful charcterization of Blackwell optimal policies.

Proposition 2.3: If * is Blackwell optimal, then for all stationary policies ye we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, for all fe such that Pard,+ = Pulet, we have where h, is a vector corresponding to p* as in Prop. 1.2.

Proof: Since * is optimal for the a-discounted problem for all a in an intorval (a, 1), wo must have, for evory pand a f (7, 1),

<!-- formula-not-decoded -->

From Prop. 1.2, we have, for all a € (i, 1),

<!-- formula-not-decoded -->

Substituting this expression in lEg. (2.8), we obtain

or equivalently

<!-- formula-not-decoded -->

By taking the limit as a → 1, we obtain the desired relation P,-J,. ≤

Ily is such that. Pood, = Pad,", then from Eg. (2.9) we obtain

By taking the limit as a - 1 and by using also the rolation of + l, = 9,* + P, 0, 10f. Ed. (1.15)). we obtain the desired relation (2.7). Q.E.D.

As a consequence of the proceding proposition, we obtain a converse of Prop. 2.1.

Proposition 2.4: If the optimal average cost over the class of stationary policies is equal to 1 for all initial states, then there exists a vector 11 such that.

<!-- formula-not-decoded -->

or equivalently

<!-- formula-not-decoded -->

Proof: Let 4* be a Blackwell optimal policy: We thou have de(i) 1 for all i. For every a. each element of the vector Pad, is equal to d. so that ProJ,+ = PiD,*. Trom Ey. (2.7), we then obtain the desired relation (2.10) with h = l,*. Q.E.D.

## Bellman's Equation for a Unichain Policy

Wo recall from Appendix Dof Vol. I that in a finite state Markov chain. a recurrent. class is a set of states that commmicate in the souse that, from evory state of the set, there is a probability of 1 to eventually go to all other states of the set and a probability of O to ever go to any state outside the set. There are two kinds of states: those that bolong to some recurrent. class (these are the states that after they are visited once, they will be visited an infinite maber of times with probability 1). and those that ar transiont (these are the states that with probability I will be visited only a finite numbor of times rogardless of the initial state).

Stationary policies whose associated Markor chains havo a single recument class and a possibly cmply sot of transient states will play an important role in our development. Such policies are called unichain. The state trajectory of the Markor chain corresponding to a michain policy. is eventually (with probability 1) confined to the recurront class of states, so the average cost per stage corresponding to all initial states as well as the differential costs of the recurrent states are independent of the stage costs of the transiend states. The next proposition shons that for a michain polios p. the averago cost por stage is the same for all initial states, and that Bollman's equation de tAg = Dil, holds. Rathermore: we show that Bollman's equation has a mique solution, provided we fix the diflerential cost of some state at some arbitrary value (0. for examplo). This is necessary, since if do and by satisfy Bollman's equation (2.5), the same is true tor do and hot ge, where y is any scalar.

Proposition 2.5: Lot fe be a unichain policy. Theu:

- (a) There exists a constant do and a vector l, such that

<!-- formula-not-decoded -->

aull

<!-- formula-not-decoded -->

- (1) Lot 1 be a fixed state. The system of the a + 1 linear equations

<!-- formula-not-decoded -->

Let.

<!-- formula-not-decoded -->

Multiplying Eg. (2.16) by 1, and subtracting it from Eg. (2.15), we obtain

<!-- formula-not-decoded -->

By defining

<!-- formula-not-decoded -->

and by noting that from Ea. (2.17). we have

<!-- formula-not-decoded -->

we obtain

<!-- formula-not-decoded -->

which is leg. (2.12). Equation (2.11) follous from Ed. (2.12) and Cor. 2.1.1. (b) By part (a). for ano solution (1.1) of the sestom of cuations (9) 1) and (2.11). we have de du. as well as 4(0) =: 0. Suppose that 1 belongs to

<!-- formula-not-decoded -->

in the a + 1 unknowns 1. 4(1)....k(a) has a unique solution.

Proof: (a) Let the a recurrent state under a. for each sadde flet C, and it, bo the expected cost and the expected mumber of stages. to spectivoly: to reach t for the first time starting from i nder policy y. Let also Cy and di be the expected cost and expected number of stages. ter spectivoly: to return to 1 for the first time starting from t under policy 1. From Prop. 1.1 in Section 2.1, we hare that. C, and N, solve uniquely the systems of equations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the recurrent class of states of the Markov chain corresponding toy. 'Then, in viow of Er. (2.14), the system of cquations (2.13) can be writton as

<!-- formula-not-decoded -->

and is the same as Bellman's cquation for a corresponding stochastic shortest, path problem where t is the termination state, gin(i)) - A, is the expected stage cost at stato i, and hi) is the average cost. starting from i up to reaching 1. By Prop. 1.2 in Section 2.1, this system has a unique solution, so hi) is uniquoly dofined by Eq. (2.13) for all i # t..

Suppose now that &amp; is a transient state of the Markov chain corresponding top. Then we choose another state &amp; that belongs to the recurrent class and make the transformation of variables 7(1) = 4(i) - 48). The system of equations (2.13) and (2.11) cau be written in terms of the variables 1 and 7(1) is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so by the stochastic shortest path argument, given carlier, it has a unique solution, implying that the solution of the system of equations (2.13) and (2.14) is also unique. Q.E.D.

## Conditions for Equal Optimal Cost for All Initial States

We now turn to the caso of multiple policies, and we provide conditions under which Bellman's equation de + h. = 7h has a solution, and by Prop. 2.1, the optimal cost is independent of the initial state.

Proposition 2.6: Assume any one of the following three conditions:

- (1) Every policy that is optimal within the class of stationary policies is unichain.
- (2) For every two slater i sud, there exists a stationary policy i

<!-- formula-not-decoded -->

- (3) There exists a state t, and constants L &gt; 0 and a € (0, 1) such

<!-- formula-not-decoded -->

where Je is the a-discounted optimal cost vector.

Then the optimal average cost per stage has the same value d for all initial states i. Furthermore, 1 satisties

<!-- formula-not-decoded -->

and for any state t, the vector h given by

<!-- formula-not-decoded -->

satisfies Bellman's equation

<!-- formula-not-decoded -->

together with 1.

Proof: Assume condition (1). Proposition 2.2 assorts that a Blackwell optimal policy exists and is optimal within the class of stationary policies. Therefore, by condition (1). this policy is unchain, and by Prop. 2.5, the corresponding average cost is independent of the initial state. The result follows from Prop. 2.1.

Assume condition (2). Consider a Blackwell optimal policy p*. If it yields average cost that is independent of the initial state, we are done, as carlier. Assume the contrary; that is, both the set.

<!-- formula-not-decoded -->

and its complement AT are nonempts. The idea now is to use the hypothesis that overs pair of states communicates under some stationary polics: in order to show that the average cost of states in olf can be reduced by opening commuication to the states in MY, theroby creating a contradiction. Take any states i € A/ and je M, and a stationary policy ye such that, for somo h, Ple = j|do=in) &gt; 0. Then there must exist states m E M and in E AT such that there is a positive transition probability from m. to i under e; that is, (Palmm = Plenty = m | tk = my) &gt; 0. It can thus be seen that the mth component of Pad, is strictly less than

max, du(i), which is equal to the mth component of Joe. 'This contradicts the nocossary condition (2.6).

Finally, assume condition (3). Let. pt be a Blackwell optimal policy. By Pa. (1.11), we have for all states i and e in some interval (i, 1)

<!-- formula-not-decoded -->

Writing this oquation for state i and for state t, and subtracting, we obtain for all i #1,

<!-- formula-not-decoded -->

Taking the limit as a → 1 and using the hypothesis that Lo (i) - do (0)| ≤ L for all a € (0, 1), we obtain that Ano (i) = g+ (t) for all i. Thus the average cost of the Blackwell optimal policy is independent of the initial state, and

To show Egs. (2.20)-(2.22), we note that the rolation limed(1 -

«).. (i) = 1 for all i follows from Ey. (2:23) and the fact , (i) = 1 for all i. Also, from Kg. (2.23), wo bavo

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

Setting 4(i) = A,+(1) -Ano (t) for all i, and using the fact J,(i) = 1 for all i and By. (2.7). we see that the condition de + h. = Th is satisfied. Q.E.D.

The conditions of the preceding proposition are among the weakest guaranteeing that the optimal average cost per stage is independent of the initial state. In particular, it is clear that some sort of accessibility condition must be satisfied by the transition probability matrices corresponding to stationary policies or at least to optimal stationary policies. For if there existed two states neither of which could be reached from the other no matter which policy we use, then it can be only by accident that the same optimal cost per stage will correspond to each one. An extreme example is a problem where the state is forced to star the same regardless of the different for different states.

## Example 2.1: (Machine Replacement)

C'onsider a machine that can be in my one of a states, 1,2,...,a. There is a cost. gfi) for operating for one time period the machine when it. is in state i. The options at the start of each period are to (a) let the machine operate one more period in the state it currently is, or (b) ropair the machine

al, i positive cost R and bring it to state 1 (corresponding to a machine in perfoct condition). The transitions between different states over each time period are governed by given probabilities po. Once repaired, the machine is guaranteed to stay in stato 1 for one period, and in subsequent, periods, it. may deteriorato to states i 2 1 according to the transition probabilities Po,. The problem is to tind a policy that minimizes the average cost por stage. Note that we have analyzed the discounted cost, version of this problom in Example 2.1 of Section 1.2. As in that example, we will assume that go is nondecreasing in 2. and that the transition probabilities satisfy

<!-- formula-not-decoded -->

for all fictions Ji), which are monotonically nondecreasing in i.

classes, {1,2..., 0.- 17 and {a) (assuming that Pin = 0). It can also be soch that condition (2) of Prop. 2.0 is not guaranteed in the absence of further assumptions. This condition is satisfied if we assume in addition that for all i we have preen &gt; 0, because, by replacing, we can bring the system to state 1, from where, by not replacing, wo can reach every other state.]

Note that, not all policies are unichain here. For example, consider the stationary policy that replaces at every state excopt the worst state a (a poor but legitimate choice). The corresponding Markov chain has two recurrent

We can show, however, that condition (3) of Prop. 2.6 is satislied. Indeed. consider the corresponding discounted problem with a discount factor

<!-- formula-not-decoded -->

and in particular.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From the last two equations, by subtraction we obtain

<!-- formula-not-decoded -->

where the last inequality follows from the fact

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which holds since A.(1) - J.(1) is nondecreasing in i. as shown in lexamble 2. 1 of Section 1.2. The last two relations imply that condition (3) of Prop.

2.6 is satistied, and it follows that there exists a scalar d and a vector he, such that for all i,

<!-- formula-not-decoded -->

while the policy that chooses the minimizing action above is average cost optimal.

By Prop. 2.0, we can take k(i) = limo-d(Jai) - Ju(D)), and since J, (i) - J, (1) is nondocreasing in i, it follows that k(i) is also nondecreasing in ¿. Similar to Example 2.1 of Section 1.2, this implies that an optimal policy takes the form

<!-- formula-not-decoded -->

whore

<!-- formula-not-decoded -->

## 4.3 COMPUTATIONAL METHODS

All the computational methods developed for discounted and stochastie shortest path problems (ef. Sections 1.3 and 2.2) have average cost per stago counterparts, which we discuss in this section. However, the derivations of these methods are often intricate, and have no direct analogs in the discounted and stochastic shortest path context. In fact, the validity of these methods may depend on assumptions that relate to the structure of the underlying Markov chains, something that we have not encountered

## 4.3.1 Value Iteration

The natural version of the value iteration method for the average cost problem is simply to generate successively the finite borizon optimal costs ThiJu. hi = 1,2,..., starting with the zero function Jo. It is then natural to speculate that the k-stage average costs 74Jo/hi converge to the optimal average cost, vector as hi → 00 (this is in fact. proved under natural conditions in Section 7.4 of Vol. I). This method has two drawbacks. First, some of the components of Th Jo typically diverge to do or -00, so direct calculation of limax fk do/k is mumerically impractical. Second, this method will not provide as with a coresponding differential cost voctor hi.

Hi can bypass both difficulties by subtracting a multiple of the mit. vector e from Tk Jo, so that the difference, call it Ah, remains bounded. In particular, we consider methods of the fonn

<!-- formula-not-decoded -->

where &amp; is some scalar satisfying

such as for example the average of (Th Jo) (i)

<!-- formula-not-decoded -->

or

<!-- formula-not-decoded -->

where t is some fixed state. Then if the differences max, (Th Jo)(i) min, (T% Jo)(i) remain bounded as li → ∞ (this can be guaranteed under the assumptions of the subsequent. Prop. 3.1), the vectors At also remain bounded, and we will see that with a proper choice of the scalar si, the vectors he converge to a differential cost voctor.

Let us now restate the algorithm he = 7%Jo - Sir in a form that is suitable for iterative calculation. We have

<!-- formula-not-decoded -->

and since we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the case where gh is given by the average of (7% Jo)(i), we have

<!-- formula-not-decoded -->

so that the iteration (3.2) is written as

<!-- formula-not-decoded -->

Similarly, in the case where we fix a state / and no choose sh = (TiJo)(4), No have

and the iteration (3.2) is written is

<!-- formula-not-decoded -->

We will henceforth restrict attention to the case where di = (Th.16) L), sund we will call the corresponding algorithm (3.1) relative volur iteration, since the itorate he is equal to Tho - (ThJo)(0)e and may Do viewed as a li-stage optimal cost. voctor relater to state 1. The following results also apple to othor versions of the algoriton (soe Exercises 4.4 and -1.5). Note that relativo value iteration, which generates he, is not. really diflerent than ordinary value iteration, which generates Tado. The vectors gonorated by the too methods moroly ditter by a mmltiple of the unit vector, and the minimization probloms involved in the corresponding iterations of the two mothods are mathomatically equivalent.

It can be seen that if the relativo value iteration (3.1) converges to some vector hi, then

<!-- formula-not-decoded -->

which by Prop. 2.1, implies that (Th)(0) is the optimal average cost per stage for all initial states, and l is an associated diflerontial cost vector. Thus convergence can only be expected when the optimal average cost per stage is indepondent. of the initial state, indicating that at least one of the conditions of Prop. 2.6 is required. However. it turns out. that, a stronger hypothesis is needed for convergence. The following example illustrates the rOsOn.

## Example 3.1:

C'onsider the itoration

which is the relativo value iteration (3.1) for the case of a fixed f. Using the expressions Till = got Pold and (Tilt) 0) = cn+ Pull), where e, is the row vector having all coordinates equal to 0 excopt, for coordinate t which is equal to 1. this iteration can be written as

Ranivalent, we have

<!-- formula-not-decoded -->

where corresponding eigenvector i,

and in particular, for the cigenvalue y = 1 and the corresponding cigenvoctor 0 = r we obtain asing the fact. die = 1.

Therefore, we have

and it follows that each cigenvalue g of P, with corresponding cigenvoctor ". which is not a scalar multiple of e. is also an eigenvalue of l', with corresponding cigenvoctor (e-cro). Thus, il Pa has an cigonvalno o # 1 that. is on the unit circle, the iteration (3.5) is not convergent. This occurs when 1, has a periodic structure aud some of its nommity digonvalues are on the unit circlo. For example, suppose that.

which has vigonvalues I and -1. Then taking i = 1, the matrix D" of la. (3.6) is given by

and has digonsalues 0 aud - 1. As a result, iteration (3.5) does not. convorgo even though y is a unichain policy.

The following proposition shows convergence of the relatise value itcration (3. 1) under a technical condition that excludes situations such as the one of the preceding example. When there is only one control asailablo per state, that is, there is only one stationary polier ye. the condition of the following proposition requires that for some positive integed me. the matrix Ph has at least one columm all the components of which ine positive. As can be seen from the proceding example. This condition need not hold ily is unchain. However. we will later provide a variant of the relative value itcration (3. 1) that converges moder the weaker condition that all stationary policies are unichain (to Prop. 3.3).

<!-- formula-not-decoded -->

Proposition 3.1: Assume that there exists a positive integer m such that, for every admissible policy a = 440, 11,...7, there exists an e &gt; 0 and a state s such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where is denotes the element of the ith row and sth column of the corresponding matrix. Fix a state t and consider the relative value iteration algorithm

<!-- formula-not-decoded -->

where h°(i) are arbitrary scalars. Then the sequence {hk converges to a vector he satisfying (Th)(t)e + h. = Th, so that by Prop. 2.1, (Th) (t) is equal to the optimal average cost per stage for all initial states and he is an associated differential cost vector.

Proof: Donoto

<!-- formula-not-decoded -->

We will show that for all i and hi 2 me we have

<!-- formula-not-decoded -->

where m and are as stated in the hypothesis. From this relation we then obtain, for some 1 &gt; 0 and all hi.

Since d(1) = 0, it follows that, for all i,

Therefore, for every e &gt; land i we have

<!-- formula-not-decoded -->

so that. {hk(i)) is a Cauchy sequence and converges to a limit. k(i). From Ed. (3.9) wo see then that the cquation (Th)(1) + 4(i) = (Th)(i) holds for all i. I will thas be sullicient to prove Ey. (3.10).

To, prove Eq. (3.10), we denote by fe(i) the control that attains the minimum in the relation

<!-- formula-not-decoded -->

for every l and i. Denote

Then we have

where yet hunt seer. tron dhese relations, sing

Since this rolation holds for evory ki 2 1. by iterating we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

First, let us assumo that the spocial states coresponding to ph me.., fl as in Eas. (3.7) and (3.8) is the fixed state I used in iteration (3.9); that is.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The right-hand side of Ey. (3.13) yiekls

<!-- formula-not-decoded -->

so using Eq. (3.15) and the fact gh-(p) = 0, we obtain

- -1

implying that.

Similarly, from the left-hand side of Ey. (3.13) we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and by subtracting the last two relations, we obtain the desired By. (3.10).

When the special states corresponding to ph-mon.., this in Egs. (3.7) iud (3.8) is not equal to t, we define a related iterative process

<!-- formula-not-decoded -->

Thon, as carlier, we have

<!-- formula-not-decoded -->

1, is straightforward to verify, using Eys. (3.9) and (3.18), that, for all i and hi we have

<!-- formula-not-decoded -->

Therefore, the coordinates of both th and gi difter from the coordinates of Th and ih, respectivels by a constant. At follows that

<!-- formula-not-decoded -->

and from Ey. (3.19) we obtain the desired Ey. (3.10). Q.E.D.

As a by-product. of the preceding proof, we obtain a rate of convergence estimate. By taking the limit in By. (3.11) as y - do, we obtain

<!-- formula-not-decoded -->

so the bound on the error is reduced by (1 - c) /m at each iteration. A sharper rate of convergence result can be obtained if we asswoo that there exists a unique optimal stationary policy p*. Then. it is possible to show that. the minim in ly. (3.12) is attained by p(d) for all i and all k after a cortain index, so for such k. the relative value iteration takes the form At = T, li- (7, 1d)(0)e, aund is governed by the largest cigenvalue modulas of the matrix Pur given by 1. (3.6).

Note that contrary to the case of a discounted or a stochastic shortest path problem, the Gauss-Seidel version of the relative value iteration method need not. converge. Indeed, the reader can construet examples of such behavior involving two-state systems and a single policy.

Error Bounds

Similar to discounted problems. the relative value iteration method Can be strengthened by the calculation of monotonie error bounds.

Proposition 3.2: Under the assumption of Prop. 3.1, the iterates h of the relative value iteration mothod (3.9) satisfy

<!-- formula-not-decoded -->

where 1 is the optimal average cost per stage for all initial states, and

<!-- formula-not-decoded -->

Proof: Let 14(1) attain the minimum in

for each hi and i. We have, using Eg. (3.9),

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Subtracting the last two relations, we obtain

and it follows that

+ +

or equivalently

A similar argument shows that.

and

<!-- formula-not-decoded -->

The mapping T' has the form

Letting 1 = 1 be the reference state, the relative value iteration (3.9) takes the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The results of the computation starting with 1º(1) = 1° (2) = 0 are shown in the table of Pig. 1.3.1.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By l'rop. 3.1 we have ha(i) → 4(i) and (TI)(i) - 1(i) = 1 for all i, so that Ca → 1. Since fend is also nondecreasing, we must have gos A for all hi. Similarly, ta ≥ 1 for all ki. Q.E.D.

We now demonstrate the relative value iteration method and the error bounds (3.20) by means of an example.

## Example 3.2:

Consider an undiscounted version of the example of Section 1.3. We have

riste dat eatend exact gerated ly the relative vator iterations

| 1k: (1)   | I,ti (2)   | Thi           |
|-----------|------------|---------------|
|           | 0.500      | 0.625 0.875   |
|           | 0.250      | 0.687|0.812   |
| 0         | 0.375      | 0.719: 0.781  |
| 0         | 0.312      | 0.734| 0.765  |
| 0         | 0.34.4     | 0.712 | 0.758 |
| 0         | 0.328      | 0.746 | 0.754 |
| 0         | 0.336      | 0.748 0.752   |
|           | 0.3:32     | 0.749 | 0.751 |
|           | 0.331      | 0.749 0.750   |
| 0         | 0.3:3:31   | (0.750 0.750  |

We note an interesting application of the error bounds of Prop. 3.2. Suppose that for some voctor h, we calculate a fo such that.

<!-- formula-not-decoded -->

Then by applying Prop. 3.2 to the original problem aud also to the moditied problem where the only stationary policy is y, we obtain

where

We thus obtain a bound on the degree of suboptimality of f. This bound can be proved in a more general setting, where J*(;) is not necessarily independent of the initial state i (see Exercise 4.10).

## Other Versions of the Relative Value Iteration Method

As mentioned earlier, the condition for convergence of the relative value iteration method given in Prop. 3.1 is stronger than the conditions of Prop. 2.6 for the optimal average cost per stage to be independent of the initial state. We now show that we can bypass this dilficulty by modifying

the problem without allecting either the optimal cost. or the optimal policios and by applying the relative value iteration method to the modified problem.

Let t bo any scalar with

<!-- formula-not-decoded -->

and consider the problem that results when each transition matris fy corresponding to a stationary policy ye is replaced by

<!-- formula-not-decoded -->

where 1 is the identity matrix. Note that. Pa is i transition probability matrix with the property that, at evory state, a self-transition occurs with probability at. Jeast (1 - 7). 'This destroys any periodic character that Pu may have. For another view of the same point, note that each eigonvalue of P, is of the form ty + (1-7), where y is an eigenvalue of P. Therefore, all cigenvalues y # 1 of Pa that. lie on the mit circle are mapped into cigenvalues of Pa strietly inside the mit cirele.

Bellnan's cquation for the modified problem is

which can be written as

<!-- formula-not-decoded -->

Wo observe that this equation is the same as Bellman's conation for the original problem.

<!-- formula-not-decoded -->

with the identification

<!-- formula-not-decoded -->

It. follows from Cor. 2.1.1 that if the avorage cost per stage for the original problem is independent of i for every do then the same is true for the modified problem. Parthermore, the costs of all stationary policies, as well as the optimal cost. are equal for both the original and the modified problem.

Consider now the relative value iteration method (3.9) for the moditied problem. A straightforward calenlation shows that it takes the form

<!-- formula-not-decoded -->

where t, is some fixed state with 49(t) = 0. Note that this iteration is as casy to execute as the original version. It is convergent, howeser, under weaker conditions than those required in Prop. 3.1.

Propositiou 3.3: Assume that each stationary policy is unichain. Then, for D &lt; 7 &lt; 1, the sequences fk (i) I generated by the modified relative value iteration (3.22) satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where d is the optimal average cost per stage and h. is a differential cost vector.

Proof: 'The proof consists of showing that the conditions of Prop. 3.1 are satisfied for the modified problem involving the transition probability matrices P, of Ey. (3.21).

Indeed. let m &gt; now, where a is the munber of states and my is the munber of distinct stationary policies. Consider a set of control functions 10,11,....fm. Then at least one y is repeated a times within the subsot Ms,..fm-1. Lots be a state belonging to the recuront class of the Markos chain corresponding to a. 'Then the conditions

are satisfied for some e because, in view of By. (3.21), when there is a positive probability of reachings from i at somo stage, there is also a positive probability of reaching it at any subsequent stage. Q.E.D.

Note that, since the modified value iteration mothod is nothing but the ordinary method applied to a modified problem, the error bounds of Prop. 3.2 apply in appropriately modified form.

## 4.3.2 Policy Iteration

The poliey iteration algorithan for the average cost problem is similar to those described in Sections 1.3 and 2.2. Given a stationary policy, one obtains an improved policy by means of a minimization process until no further improvement, is possible. We will assume throughout this section

that coery stationary policy encountered in the course of the algorithm is unichain.

At the lith stop of the policy iteration algorithm, we have a stationary policy ali. No then perform a policy craluation stop; that is, wo obtain corresponding average and differential costs th and Mi(i) satisfying

or equivalently

<!-- formula-not-decoded -->

Note that di and he can be computed as the mique solution of the linear systom of equations (3.21) togother with the normalizing equation hi(t) = 0. whero / is any state (ef. Prop. 2.5). This system can be solved either directly or iterativoly using the relativo vale iteration method or by an adaptive aggregation method, as discussed later.

We subsenently perform a policy improvement stop; that is, we find a stationary policy pate, where for all i, pitt (i) is such that.

<!-- formula-not-decoded -->

or equivalontly

<!-- formula-not-decoded -->

Il gets =ph, the algorithm terminates; otherwise, the process is repeated with pitl replacing at.

There is an easy proof, given in Exercise ..l, that the policy iteration algorithm terminates tinitely if we assume that the Markov chain corresponding to each ph is irreducible (is unchain and has no transient states). To prove the result withont this assumption, we impose the following restriction in the way the algorith is operated: if k(i) attains the minimon. in Eq. (3.25), we choose it' (i) = ph(;) even if there ure other controls attaining the minimum in addition to p(i). We then have:

Proposition 3.4: If all the generated policies are nichain, the policy iteration algorithin terminates finitely with an optimal stationary policy.

It is convenient to state the main argument needed for the proof of Prop. 3.4 as a lemma:

Lemma 3.1: Let de be a michain stationary policy, and let 1 and l. be corresponding average and differential costs satisfying

<!-- formula-not-decoded -->

as well as the normalization condition

<!-- formula-not-decoded -->

(The above limit and the limit in the following Ed. (3.29) are shown to exist in Prop. 1.1.) Let {H, I,...} be the policy obtained from i via the policy iteration step described previously, and let 7 and ii be corresponding average and differential cost satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Then if fe # ye, we unst have either (1) 7 &lt; 1, or (2) 7 = 1 and T.(i) ≤ hi) for all i = 1,...,n, with strict inequality for at least one

We note that, once Lemma 3.1 is established, it can be shown that. the policy iteration algorithm will terminate finitely. The reason is that the vector le corresponding to y via 84. (3.26) and (3.27) is mique by Prop. 2.5(b), and therefore the conclusion of Lemma 3.1 guarantees that no policy will be encountered more than once in the course of the algorithm. Since the number of stationary policies is finite, the algorithm must terminate finitely: It the algorithm stops at the kith step with pitt pt, we see from Eqs. (3.24) and (3.25) that.

<!-- formula-not-decoded -->

which by Prop. 2.1 implies that pe is an optimal stationary policy. So lo prove P'rop. 3.1 Chero remains to prove Lomma 3.1.

Proof of Lemma 3.1: For notational convenience, denote

<!-- formula-not-decoded -->

Define the vector &amp; by

<!-- formula-not-decoded -->

We have, by assumption, Til = Th. STul. = de +1, or equivalently

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By combining Ey. (3:30) with the equation Tr + T = I + Thi, ve obtain

<!-- formula-not-decoded -->

Multiplying this relation with To and adding from O to N' - 1, we obtain

<!-- formula-not-decoded -->

Dividing by N and taking the limit as N → 2o, we obtain

<!-- formula-not-decoded -->

In vice of the fact 2= 0 lot. 1y. (3.32), we see that

<!-- formula-not-decoded -->

Il 1 &gt; 5, no are done. so assume that 1 = 7. A state i is called P. recurrent (P-transicat) if i belongs (does not belong, respectively) to the that are positive are those columns corresponding to P-recurrent states, we obtain

<!-- formula-not-decoded -->

It follows by construction of the algorithin that if is T-recurrent, then The ith rows of Pand T are identical since fli) = ad) for all i with §(i) = 0. Since I and 7 have a single recurrent class. it follows that this from which we obtain

Define also

<!-- formula-not-decoded -->

class is identical for both l' and P. From the normalization conditions (3.27) and (3.29), we then obtain 4(i) = Ai) for all i that are P-recurrent. Equivalontly:

<!-- formula-not-decoded -->

From Eg. (3.33) we obtain

In view of Lie, (3.36). the coordinates of 7" A corresponding to l-transient states tend to zero. Therefore, we have

<!-- formula-not-decoded -->

From Egs. (3.32) and (3.35) to (3.37), we see how that either d = 0. in which case pe = I, Or else 1 2 0 with strict inequality Ai) - 0 for at least. one P-transient state i. Q.E.D.

We now demonstrate tho policy iteration algorithm be means of the example of the previous section.

## Example 3.2: (continued)

Let.

We take 1 = 1 as a reference state and obtain 1,0. Han(D), and A,, (2) from the system of equations

<!-- formula-not-decoded -->

Substituting the data of the problem,

<!-- formula-not-decoded -->

from which

<!-- formula-not-decoded -->

No now lind 1'(D) and p'(2) by the minimization indicated in Ey. (3.25). No detormine

<!-- formula-not-decoded -->

an

<!-- formula-not-decoded -->

Tho minimization yiolds

<!-- formula-not-decoded -->

Ne obtain Aar. 4"9(1), and 4, (2) from the systom of equations

<!-- formula-not-decoded -->

By substituting the data of the problem. we obtain

<!-- formula-not-decoded -->

We find de(0) and d°(2) by determining the minimum in

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

The minimization riolds

and hence the preceding policy is optimal and the optimal average cost. is 1, 1 = 3/1.

The algoritan of this section shares some of the features of other types of policy iteration algorithms. In particular. it is possible to carry out policy evaluation approximatoly by using a fow relativo value iterations: see (Put94) for an analysis. Note also that in specially structured problems one may be ablo to contine policy iteration within a convenient subset of policies, for which policy evaluation is facilitated.

## Adaptive Aggregation

C'onsider now an extension to the average cost problem of the aggre gation method described in Section 1.3.3. For a given unichain stationary policy a, we want to calculate an approximation to the pair (1,,.l,) satisfying Bellman's equation do + Ag = Til, and 1g(11) = 0. where the state a is viewed as the reference state. By expressing d, as 1, = (T,A,)("). wo can climinate it. from this system of equations, and obtain If = T,V, - (T,h,)(1)e. This equation is written compactly its

where the mapping " is defined by

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

with en = (0.0...., 0, 1)'.

We partition the set of states into disjoint subsets Sy, Sz,..., S,,, That are viewed as aggregate states. We assume that one of the subsets, say Sm, consists of just the reference state n; that is, S,, = In%. As in Section 1.3.3, consider the n x m matrix W whose ith column has unit, entries at. coordinates corresponding to states in S, and all other entries coual to zero. Consider also an m'x a matrix Q such that the ith row of Q is a probability distribution (ail,...,Gin) with dis = 0 if s &amp;S,. Note that QWV = 1, and that the m x m. matrix k=- QIN is the transition probability matrix of the aggregate Markor chain, whose states are the m aggregate states.

Suppose now that we have an estimate l of hy and that we postulate that over the states s of cvery aggregate state S, the variation ha(s) - 1(5)

is constant. This amounts to hypothesizing that for some m-dimonsional vectory we have

<!-- formula-not-decoded -->

By combining the equations Theme Pol and Ag-1 + Polo, we have

By undtiplying both sides of this equation with Q. and by using the rolations An- 1 = My and QW' = 1, we obtain

<!-- formula-not-decoded -->

Assuming that the matrix 1 - QP, I' is invertible, this equation can be solved forg. Also, hy applying T' to both sides of the cquation An =-h+-Wy, no obtain

Thus the aggregation iteration for average cost problons is as follows:

## Aggregation Iteration

Step 1: Compute 7h, =g, + Pol, where

<!-- formula-not-decoded -->

Stop 2: Dolincate the aggregate states (ie., dofino WV) and spocify the matrix Q.

Step 3: Solve for y the systom

and approximate lip using

<!-- formula-not-decoded -->

For the iteration to Do valid, the matrix A-QP, I must be invertible. We will show that this is guaranteed under an aperiodicits assumption such as the one used to prose convergonce of the relative value iteration method (of. Prop. 3.1. lo particular. we assume that all the cigenvalues of the transition matrix 8 - 08,00 of the aggregate Markos chain. except for a single mite eigensalue. lie strictly within the mit circle. Let us denote by e the m-dimensional vector of all l's. and by em the m-dimensional vector with last coordinate 1, and all other coordinates 0. Then using the casily verified relations @e=d and dine -do. we see that.

From the analysis of Example 3.1 in Section 1.3. we have that @PI has m - 1 cigenvalues that are equal to the m - 1 nonmity eigenvalues of K and has 0 as its mth eigenvalue. Thus QP, IF must have all its cigenvalnes strictly within the unit circle, and it follows that the matrix 1 OPill is invertible.

In an adaptive aggregation mothod, a key issue is how to identify the aggregate states Sy..., Si, in a way that the error h, - he is of similar magnitudo in cach aggrogate state. Similar to Section 1.3.3, one way to do this is to view Th as an approximation to hy and to group together states i with comparable magnitudes of (TH)(i) - 4). As discussed in Section 1.3.3, this type of aggregation method can be greatly improved by interleaving cach aggregation iteration with multiple relative value itortions (applications of the mapping &amp; on the current iterate). We rofor to [BeC'89) for luther experimentation, analysis, and discussion.

## 4.3.3 Linear Programming

Let us now develop a linear programming-based solution method. assuming that any one of the conditions of Prop. 3.3 holds, so that the optimal average cost. A* is indepondent. of the initial state, and together with i associated differential cost vector hr, satisfies der the = 11*. Consider the following optimization problem in the variables 1 and Ai). i = 1.....n.

<!-- formula-not-decoded -->

which is equivalent to the linear program maximize 1

<!-- formula-not-decoded -->

(3.38)

A ncarly verbatim repetition of the proof of Prop. 2.1 shows that it (1,11) is a feasiblo solution, that is, do thE Th, Then 1: 1, which implies that (1*.h*) is an optimal solution of the linear program (3.38). Furthermore, in my optimal solution (T, 7) of the linear program (3.38), we have T = 1*.

There is a linear program. which is dual to the abore aud which adinits an interesting interpretation. In particular, the duality theory of

linear programming (see o.g.. (Dan63)) asserts that the following (dual) linear program

<!-- formula-not-decoded -->

has the samo optimal value as the (primal) program (3.38). Tho variables 4(1, 1), i = 1,..,n, « € U(i), of the dual program can be interpreted as the steady state probabilities that state i will be visited at the typical transition and that control a will then be applied. The constraints of the dual program are the constraints that yi, «) must satisfy in order to be feasible steady-state probabilities under some rundomized stationary policy, that is, a policy that chooses at state i the control a probabilistically, by sampling the constraint set. U(i) according to the probabilities qi, u), u E U(i). The cost function

<!-- formula-not-decoded -->

of the dual problem is the steady-state average cost per transition. Duality theory asserts that the minimal value of this cost is d*, thus implying that the optimal averago cost per stage that can be obtained using randomized stationary policies is no better than what can be achieved with ordinary (detorministio) stationary policies. Indeed, it. can be verified that if yet is an optimal (deterministic) stationary policy that is michain, and py is the steady-state probability of state i in the corresponding Markov chain, then

<!-- formula-not-decoded -->

is a optimal solution of the dual problem (3.39).

## 4.3.4 Simulation-Based Methods

We now describe briefly how the simulation-based methods of Section 2.3 can be adapted to work for average cost problems. We make a slight. change in the problem definition to make the notation better suited for the simmlation context. In particular, instead of considering the expected cost gi.u) at state i under control a. we allow the costy to depond on the next stated. Thus our notation for the cost por stage is now gli, a,j). as in the sumation-related material for stochastic shortost path and discounted problems (of. Section 213). All the results and the entire analysis of the preceding sections can be rewritten in torms of the new notation by replacing (i, a) with E"- 1P. (a)g(i,u.i).

## Policy Iteration

In order to implement. a simulation-based policy iteration algorithm like the one of Section 2.3.1, we noed to be able to carry out the policy evaluation stop for a given unichain policy y. This can be done by using the connection with the stochastic shortest path formulation described in Section 4.1. We fix a state d, and we evaluate the cost of the givon policy « for two stochastic shortest path problems whose tormination state is (essentially) 1. In particular. ve evaluate by Monto-Carlo simulation or TD(1) the expected cost C, from each state i up to reaching t of. Ed. (2.15)). This requires the generation of many trajectories terminating at state t and the corresponding sample costs. Simultaneously with the evaluation of the costs C,, we evaluate the expected number of transitions N, from each state i up to reaching / lef. Ey. (2.16)!. Then the average cost de of the policy is obtained as

<!-- formula-not-decoded -->

(of. Eq. 2.17)), and the associated ditferential costs are obtamed as

<!-- formula-not-decoded -->

lef. Eq. (2.18)}.

To implement. a simulation-based approximate policy iteration algorithm, a similar procedure can be used. In particular, one can obtain by Monte-Carlo simulation or TD(1) functions C(r) and Ni(r) that depend on a parameter vector and approximate the costs Ci and N, of the conesponding stochastic shortest path problems, as described in Section 2.3.3. Then, one can use

<!-- formula-not-decoded -->

as au approximation to the average cost per stage of the policy and also use

<!-- formula-not-decoded -->

as approximations to the corresponding differential costs lef. Eys. (3.40) and (3.41)).

Note here that becanse the approximations C(r) and Ne(r) play an important role in the calculations, it may be worth doing some extra simulations starting from the reference state t to onsure that these approximations are ncarly exact.

## Value Iteration and (-Learning

To derive the appropriate form of the Q-learning algorithin of Section 2.3.2. we form an auxiliary average cost problem by augmenting the original system with one auditional state for each possible pair (i.u) with a € U(i). The probabilistic transition mechanism from the original states is the same as for the original problem, while the probabilistic transition mechanism from an auxiliary state (i, a) is that we move only to states j of the original problem with corresponding probabilities pe(a) and costs gi, a,j). It can be seen that the auxiliary problem has the same optimal average cost per stago d as the original, and that the corresponding Bellman's equation is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Qi,u) is the dillerential cost corresponding to (i,«). Taking the minimum over « in Ey. (3.13) and substituting in Eg. (3.42), we obtain

<!-- formula-not-decoded -->

Substituting the above form of hi) in Eg. (3.13), we obtain Bellman's equation in a form that exclusively involves the O-factors:

<!-- formula-not-decoded -->

Let as now apply to the auxiliary problem the following variant of the rolative value itcration

(see Desercise 1.5 for the caso where o = 0, po = 1, and pj = O for i # t). We then obtain the iteration

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From those oquations, wo have that

(3.17)

<!-- formula-not-decoded -->

and he substituting the above form of ht in Ba. (3.47), we obtain the following relative value iteration for the Q-factors

<!-- formula-not-decoded -->

An incremental version of the preceding iteration that involves a positive stopsize y is given by

This iteration is analogous to the value iteration for O-factors in the stochastie shortest path context. The sequence of values min,ela) @k(1.a) is expected to convergo to the optimal average cost per stage and the sequences of values minetto) Qi. a) are expected to consorgo lo dilterential costs k(i).

<!-- formula-not-decoded -->

(comparo with Ey. (3.8) in Section 2.31. The natural form of the @-learning method for the averago cost problem is an approximate version of this iteration, whereby the expectod value is roplaced by a siugle sample, i.c..

<!-- formula-not-decoded -->

where and ali, a. are gonerated from the pair Coad by samulation.

## Minimization of the Bolman liquation Perror

There is a straightforward extension of the mothod of Soction 2.3.3 for obtaining an approximate represcutation of the average cost. A and issociated differential costs k). based on minimizing the squared orror in Bellman's equation. Here we approximate 1 by Ar) and hi) by hi.*).

where e is i vector of unknown parameters/weights. We impose a normalization constraint such as k(1) = 0, where t is a fixed state, and we minimize the error in Bellman's cquation by solving the problem

<!-- formula-not-decoded -->

where S is a suitably chosen subset of representative" states. This minimization may be attempted by using some gradient method of the type discussed in Scotion 2.3.3.

## 4.4 INFINITE STATE SPACE

The standing assumption in the preceding sections has been that the state space is finite. Without finiteness of the state space, many of the results presented in the past three sections no longer hokd. For example, whereas one could restrict attention to stationary policies for finite state systems. this is no longer true when the state space is infinite. The following example from (Ros83a) shows that. if the state space is countable, there may not exist an optimal policy.

## Example 4.1:

Let the State Space Do 11. 1', 2,2,3,36,..J. and let there Do too controls, 1' and d". The transition probabilities and costs por stage are

<!-- formula-not-decoded -->

lu words, at state o we may, at a cost. 0, cither move to state (et 1) or move to state i, where we stay thercalter at a cost -14 1/i per stage.

It can be seen that for ovory policy a and state i = 1.2,..., we have Ja(i) &gt; -1. However, for every stato i, we can obtain an average cost per stago -1 + 1/j, where j 2 i, by moving to state j' once we got to slato j. Hence, for every initial state i = 1,2,..., an average cost per stage of -1 can be approached arbitrarily closely, but cannot be attained by any policy.

Here is another example, from (Ros70), which shows that for a countable state space there may exist an optimal nonstationary policy, but not in optimal stationary policy.

## Example 1.2:

Let the state space be f1, 2.3...J. and lot there be two controls, n' and i? The transition probabilities and costs per stage are cit=

<!-- formula-not-decoded -->

<!-- image -->

In words, al stato e we may dither move to state (r+- 1) at a cost lor stay at i at a cost 1/i.

For any stationary policy y other than the policy for which pi) = n for all i. let n(p) be the smallest integer for which

<!-- formula-not-decoded -->

Then the corresponding average cost per stage satisties

<!-- formula-not-decoded -->

For the policy where eli)e n' for all i, we have dad ellorall i. Since the optimal cost por stage cannot. be less than sero, it is clear that

<!-- formula-not-decoded -->

However, the optimal cost is not. attained by any stationary policy, so no stationary policy is optimal. On the other hand, consider the nonstationary policy n that on entering state i chooses d for i consecutive times and then chooses i' If the starting state is i, the sequence of costs incurred is

<!-- formula-not-decoded -->

The average cost. corresponding to this policy is

<!-- formula-not-decoded -->

Hence the nonstationary policy n" is optimal while, as shown previously, no Stationary policy is optial.

Generally, the analysis of average cost problems with an infinite state space is diffical. although there has boon considerable progross (are the roferences). An important tool is Prop. 2.1, which admits a straightforward extension to the case where the state and control spaces are infimite. In particular, if we can find a scalar 1 and a bounded function he such that. Bellman's equation (2.1) holds, then by repeating the proof of Prop. 2.1. we can show that A must be the optimal average cost per stage for all initial states. Among other situations, this result is useful when we cal guess the right 1 and h, and verify that they satisfy Bellman's equation. Some important special cases can be satisfactorily analyzed in this way (see the references). We describe one such case, the average cost version of the linear-quadratie problem examined in Chapters 1, and 5 of Vol. 1.

--1

## Linear Systems with Quadratic Cost

Consider the linear-quadratic problem involving the system

<!-- formula-not-decoded -->

and the cost function

<!-- formula-not-decoded -->

Ne make the same assumptions as in Section S.l. That is. Q is positive semidetinito symmetrie, R is positive dotinito symmetric, and uk are independent, and have zero mean and finito socond momonts. We also as sumo that the pair (.1, 8) is controllable and that the pair (A.(*), where Q= C"C', is observable. Under these assumptions. it was shown in Sortion 1. 1 of Vol. I that the Riccati oquation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

vicks in the limit a matrix k.

which is the uniquo solution of the cquation within the class of positive semidefinite symmetrio matrices.

The optimal value of the Nestage costs

<!-- formula-not-decoded -->

has been derived earlier and was soon to be canal to

<!-- formula-not-decoded -->

Since A = linex As and

<!-- formula-not-decoded -->

we see that the optimal N-stage costs tend to

<!-- formula-not-decoded -->

as N→ x. In addition, the Nestage optimal policy in its initial stages tends to the stationary policy

<!-- formula-not-decoded -->

Furthermore, a simple caloulation shows that, by the definition of 1, A, and p*(x), we have

while the minimm in the right-hand side of this equation is attained at. u* = 1*(r) as given by Ey. (1.9).

By repeating the proof of Prop. 2.1, we obtain

<!-- formula-not-decoded -->

wit wait i der, Were lyi died in the preceding rolation.

with equality if m= far.po....?. Thus the linear stationary policy given by Ba. (1.9) is optimal over all policies o with Elected Leo, ay bounded uniformly over N.

## 4.5 NOTES, SOURCES, AND EXERCISES

Sovoral authors have contributed to the average cost problem (low60) (Bro65). (Ros70). [Sch68). (Vri66). (Voi69)), most notably Blackwell (Bla621). An alternative detailed treatment to outs is given in Ptol. An extensive survey containing many references is given in (A13/93).

The result of Prop. 2.6 under conditions (2) and (3) was shown in (Bat73) and (Ros70). respectivels: The relative value iteration method of Section 1.3 is due to (Whi63). and its modified version of Eg. (3.22) is due

to (Sch78. The erros bounds of Prop. 3.2 are due to Odoni ((Odo69)). The value itoration method has boon analyzed exhaustivoly in (Scholl, (ScF77), and 50078). Convorgence under slightly woaker conditions than those given here is shown in (Pla77). The error bounds of Exercise 4.10 are due to Varaiya (VarTSp), who ased them to construct a differential form of the value iteration mothod. Discrete Limo versions of Varaiya's method are given in (PBW 79). 'The value iteration nothod based on stochastic shortest paths of Exercise 1.15 is new (sec (Ber95c]).

The policy iteration algorithm can be generalized for probloms where the optimal avorage cost per stage is not the same for overs initial state (see (Bla62), (lu91), (Vei66), and (Der70)). The adaptive aggregation method is due to (BeC89).

algorithms of Section 1.3. and Exercise 4.16 are now. Alternative algorithmas of the Q-learning type are given in (Sch93) and (Singt).

The approximation procedures of Section 1.3.1, and tho Q-learning

For amalssis of infinite horizon corsions of inventory control prob(HoT71).

lems, such as the ones of Section 4.2 of Vol. I, soo (gl63al, (g1636), and lufinite state space modols are discussed in [Kus78), (Sen&amp;6), (Lass] (Bor89), (Cav89a), (Car896), (Her89), [Sen&amp;Dal. (Sonsob), (FAM90), LANSE, (Car9D), (MHLO8), (Sen91), (Ca5921, (KiS92), (AB/93), (SenD3a), [Sen936). and (Put91).

## EXERCISES

4.1

Solve the avorage cost version (a = 1) of the computer manufacturer's problom (Exercise 7.3, Vol. 1).

4.2

Consider a stationary inventory control problem of the type considered in Section 1.2 of Vol. I but. with the citterence that the stock te can only take integor values from O to some integer M. The order as can take integer values with 0 = 14, SM- rh, and the random demand ek can only take nonnegative integer values with Plus = 0) &gt; Dand P(uk = 1) &gt; 0. Unsatisfied demand is lost, so stock evolves according to the equation tito = maxO, rht Ud - Wk). The problem is to find an inventory policy that minimizes the average cost per stage. Show that there exists an optimal stationary policy and that the optimal cost is independent of the initial stock to.

## 1.3 (Lik.71]

Consider a person providing a certai tope of service to castones. The person receives at the beginming of each timo period with probability p, a proposal ly a customer of typee, where i- 1,2,...,8, Who offers an amount of money A, We assume that E"D SO. The person may reject the oller. in which case the customer leaves and the person remains idle during that period. or the person may accopt the oller in which case the person spends some time with that customer determined according to a Markor process with transition probabilities ga, where, for fi = 1.2,..,

Ba= probability that the lope i customer will leave after l periods, given that the customer has already stayed with the person for hi - 1 periods.

The problem is to determino an accoptance rejoction policy that maximizes

<!-- formula-not-decoded -->

C'onsider loo casos:

1. 184 = 18, € (0, 1) for all fi.
2. Har each, there exists fi, such that Ph, =1.
3. (a) Formulate the person's problem as an average cost per stage problem. and show that the optimal cost is independent of the initial state.
4. (1)) Show that there exists a scalar d° and in optimal policy that acropts
5. the offer of a type i customer if and only if

<!-- formula-not-decoded -->

where T, is the expected time spent with the type i customer given by

4.1

Let A" be an arbitrary vector in l", and dotine for all and he 1

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (a) Show that the sequences they, thy, and fliy are generated by the algorithmns

- (b) Show that the convergence result of Prop. 3.1 holds for the algorithms of

## 4.5 (Variants of Relative Value Iteration)

Consider the following two variants of the relative value iteration algorithm:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hore c is an arbitrary scalar and (Pi,..,P.) is an arbitrary probability distribution over the states of the system. Under the assumptions of Prop. 3.1, show that the sequence fall converges to a vector hand the sequence 114} converges to a scalar &amp; satisfying de tA= Th, so that by Prop. 2.1, 1 is equal to the optimal averago cost por stage for all initial states and he is an associated difforontial cost vector. Hint: Modify the problem by introducing an artificial state t from which the system moves al a coste to state with probability po, for all a. Apply Prop. 3.1.

4.6

Consider # deterministic sostom with too states Oand I. Upon entering state 0, the sestem stays there permanently at no cost. In state l there is a choice of staying there at no cost or moving to state O at cost 1. Show that every policy is averago cost optimal, but the only stationary policy that is Blackwell optimal is the one that. keeps the system in the state it currently is.

where

4.7

Show that a Blackwell optimal policy is optimal over all policies (not just those that are stationary). Hoet: Use the following result: If fond is a nomegative bounded sequence, then

<!-- formula-not-decoded -->

A proof of this reonlt can be found in (Put91, p. AlT.

## 4.8 (Reduction to the Discounted Case)

For the finite-state average cost problem suppose there is a state &amp; such that for some 1 &gt; 0 wo have po(a) 2 18 for all states i and controls a. Consider the (1 - 13)-discounted problem with the same state space, control space, and transition probabilities

<!-- formula-not-decoded -->

Show that 3J(0) and Ji) are optimal average and dilferential costs, respertively, where J is the optimal cost function of the (1 - (8)-discounted problem.

## 4.9 (Deterministic Finite-State Systems)

Consider a deterministic finite-state system. Suppose that the system is controllable in the sense that given any two states i and j, there exists a sequence. of admissible controls that drives the state of the system from i to j. Consider the problem of finding an admissible control sequence {uo, us,..} that. minimizos

<!-- formula-not-decoded -->

Show that the optimal cost is indepondent of the initial State, and that there exist optimal control sequences. which after a certain time index are periodic.

## 4.10 (Generalized Error Bounds)

Let h be any a-dimensional vector and fe be such that.

<!-- formula-not-decoded -->

Show that, for all i,

rogardless of whother f*(i) is independent of the initial state i. Hint: Complate the dotails of the following argumont. Lot.

<!-- formula-not-decoded -->

and lot &amp; be the voctor with coordinates d(i). We have

and, continuing in the same manner,

<!-- formula-not-decoded -->

Hence where

<!-- formula-not-decoded -->

proving the right-hand side of the desired rolation. Also, let 7 = fron...3 be any culmissible policy. We have

<!-- formula-not-decoded -->

from which we obtain

Thus. for all i.

and, taking the limit as N= o, we obtain

<!-- formula-not-decoded -->

Since n is arbitrary, we obtain the left-hand side of the desired relation.

<!-- formula-not-decoded -->

4.11

Use Prop. I. to show that in the policy iteration algorithm we have for all

where

<!-- formula-not-decoded -->

Use this fact to show that if the Markow chain corresponding to pal has no transient states and pit' is not optimal, then pots &lt; 1

## 4.12 (Policy Iteration for Lincar-Quadratic Problems)

The purpose of this problem is to show that policy iteration works for linearquadratic probloms (even though neither the state space nor the control space are (inite). C'onsider the problem of Section 4.1 under the usual controllability. observability, and positive (semi) definiteness assumptions. Lot. Lo be an mxn matrix such that tho matris (i + BLo) is stable.

- (a) Show that the average cost per stage corresponding to the stationary policy 4", where p"(r) = Lor, is of the form

where A. I a positive comdemote emotic matrice eatettne thr (lincar) equation

<!-- formula-not-decoded -->

- (b) Let 4'(r) = L= (8 + 1'103) '1'Bode Do the control Mietion attaining the minimum tor each in the contession

Show that.

where 2, is some positive semidefinite symmetric matrix.

- (e) Consider repeating the (policy iteration) process described in parts (a) and (b), thereby obtaining a sequence of positive semidefinite symmetric matrices fled. Show that.

<!-- formula-not-decoded -->

where A is the optimal cost. matrix of the problem.

## 4.13 (Alternative Analysis for the Unichain Case)

The purpose of this exercise is to show how to extend the average cost problem analysis based on the connection with the stochastic shortest path problem, which is given in Section 7.4 of Vol. I. In particular, here this connection is used to show that there exists a solution (A, h) to Bellman's equation deth = Th if every policy that is optimal within the class of stationary policies is unichain, without resorting to the use of Blackwell optimal policies (ct. Prop. 2.6). For this we will use the stochastic shortest path theory of Section 2.1, and from the present chapter, Prop. 2.1 and Prop. 2.5 (which is proved using a stochastic shortest path argument). Complete the details of the following proof:

For any stationary policy ye, let A, bo the average cost per stage as delined by lig. (2.17), let A = min, de, and let. N = f| A, = A%. Suppose that there is a stato s that is simultancously recurrent in the Markov chains corresponding to all ye. € A. Similar to Section 7.1 in Vol. I, consider an associated stochastic shortest path problem with states 1,2,...,n and an artificial termination state / to which we move from state i with transition probability pa(4). The stage costs in this problem aro gi,u) - 1 for i = 1,...,", and the transition probabilities from a state i to a state j*s are the same as those of the original problem, while pe(a) is zero. Show that in this stochastie shortest path problem, evory impropor policy has infinito cost for some initial state, and use this fact to conclude that if kei) is the optimal cost starting at state i = I,...,n, then A and he satisfy Ae +h = Th. If there is no state s that is simultancously recurrent for all ye € N, select a f. € M such that there is no ye € A/ whose recurrent class is a striet subset of the recurrent class of ji (it is sufliciont, that fi bas minimal mumber of recurrent states over all « € A/), change the stage cost of all states i that are not recurrent under ji to gi, u) +e, where &amp; &gt; 0, use as state s in the preceding irgument any state that is recurrent under jr, and take e → 0.

## 1.14 (Stochastic Shortest Path Solution Method)

The purpose of this exercise is to show how the average cost problem can be solved by solving a finite sequence of stochastic shortest path problems. As in Section 7.1 of Vol. 1, we assume that a special state, by convention state 1, is recurrent in the Markov chain corresponding to each stationary policy. For a stationary policy 1, let.

C*"(0): expocted cost starting from i up to the first visit ton,

N"(i) : expected number of stages starting from i up to the first visit to n.

The proof of Prop. 2.5 shows that d, = C,(n)/N,(e). Let 1°= ming de be the corresponding optimal cost.

Consider the stochastie shortest path problem obtained by leaving unchanged all transition probabilities po (a) fory ta. by sotting all trausition probabilities pin(e) to 0, and by introducing an artificial termination state t to which we move from cach state i with probability pin (a). The expected stage cost at state i is gi, u) - 1, where d is a scalar parameter. Let ha, (4) be the cost of stationary policy a for this stochastic shortest path problom, starting from state i. and let he(i) = min, ha.di) be the corresponding optimal cost.

- (a) Show that for all scalars A and X', wo have

<!-- formula-not-decoded -->

- (b) Define

<!-- formula-not-decoded -->

Show that, le(i) is concave, monotonically decreasing, and piecowise lincar as a function of 1, and that.

<!-- formula-not-decoded -->

Figure 4.5.1 illustrates those relations.

- (c) Consider a one-dimensional search procedure that. finds a zero of tho fiction he(n) of 1. This procedure brackets d' from above and below, and is illustrated in Fig. 1.5.2. Show that, this procedure solves the average cost problem by solving a finite vumber of stochastic shortest path problems.

Figure 4.5.1 Rotation of the costs of stationary policies in the accrage cost. problem and the associated stochastic shortest path problem.

<!-- image -->

Figure 4.5.2 One dimensional iterative search procedure do find 1 such that 12(1) = 0 fot. lixercise 1.14(0)). Rach value ha(a) is obtained by solving the associated stochastic shortest path problem with stage cost gli, a) - 1. Al. the start of the typical iteration, we have scalars o and &amp; such that a &lt; 1* &lt; 1, toother with the corresponding nonzero values han) and hg(n). We find a' such that

<!-- image -->

<!-- formula-not-decoded -->

and wo calculate lo(n). Lot 1' be such that

<!-- formula-not-decoded -->

We then roplace or by d', and if 18 &lt; 3, we also calculate l, (4) and we replace B Dy 1. We then portorm another iteration. The algorithm stops if either ha, (n) = 0 or 11(n) = 0.

## 4.15 (Stochastic Shortest Path-Based Value Iteration (Ber95c])

The purpose of this exercise is to provide a value iteration method for average cost problems, which is based on the connection with the stochastic shortest path problem. Lot the assumptions of Exercise 1.14 hold. Consider the algorithm

<!-- formula-not-decoded -->

whore n is the special state that is recurrent for all michain policies and sk is a positive stopsize.

- (a) Interprod do algorita as a vatuo itoration algoritlem for a slowly vary ing stochastic shortest path problem of the type cousidered in fixercise 1.11. Given that, for small s, the iteration of lo is laster that the itoration of 1, speculato on the convergence properties of the algorithm. constant. Another interesting possibility for which convergonce can be prosed is to solect do as a constant divided by 1 plus the maber of times that A (o) has changed sign.]
- (D) lise the error bounds of Prop. 3.2 to justify the iteration

<!-- formula-not-decoded -->

where let denotes the projection of a scalare on the interval

## 4.16 (Q-Learning Based on Stochastic Shortest Paths)

The purposo of this exercise is to provide a Q-learning method for average cost problems, which is based on the value iteration method of Exercise 1.15. Let the assumptions of Exercise 1.14 hold. Speculate on the convergence properties of the following Q-learning algorithm where

and jud gi, ad) are generated from the pair (e,«) by simulation. Here the stopsizes a and &amp; should he dimmishing, but &amp; should diminish "faster" than a for example 9 = c/k and 8 = cz/hilogh, where cy and ca are positive constants and t is the number of iterations performed on the corresponding pair (i. 1) or 1.

<!-- formula-not-decoded -->

## Continuous-Time Problems

| Contents   |
|------------|

We havo considered so far problems where the cost per stage does not depend on the time required for transition from one state to the noxt. Such problems have a natural discrote time copresentation. On the other hand, there are situations where controls are applied at discrete times but cost is continmously accumulated. Rathermore, the time between successive control choices is variable; it may be random or it may depend on the current. state and the choice of control. For example, in queucing systoms stato transitions correspond to arrivals or departures of customers, aud tho corresponding times of transition aro random. This chapter primarily discusses problems of this type. We restrict attention to contimons-time systems with a finite or countable munber of states. Many of the practical systems of this tope involvo the Poisson process. so for many of the examples discussod, wo assume that the roader is familiar with this process at the level of textbooks such as (Ros836) and (Gal95).

In Section 5.1, we concentrate on an important class of continuoustime optimization models of the discounted type, where the times between successive transitions have an exponential probabilty distribution. We show that by using a conversion process called uniformication. discounted versions of these models can be analyzed within the discrete-time framework discussed up to now.

In Section 5.2, we discuss applications of uniformization. We concentrate on quencing models arising in varions communications and scheduling contexts.

In Section 5.3, we discuss more general continuous-time models, called semi-Markoo problems, where the times between successive transitions need not have an exponential distribution.

## 5.1 UNIFORMIZATION

In this chapter, we restriet ourselves to contimons-time systems with a finite or a contable mober of states. Here state transitions and control selections take place at discrete times, bout the time from one transition to the next. is random. In this section, we assume that:

1. If the sostem is in state i and control a is applied. the next state will he j with probability pu(n).
2. The time interval o between the transition to state i and the transition to the next state is exponentially distributed with parameter 1(4);

Patransition time interval #Ti.a% 51-6-4(4)r. or equivalently, the probability density function of t is

<!-- formula-not-decoded -->

Furthermore, o is independent. of carlier transition times, states, and controls. The paramotors e(a) are amitorol bonded in the sense that for some e we have

<!-- formula-not-decoded -->

The parameter " (4) is referred to as the rate of transction associated with state i and control a. It can bo verified that the corresponding average transition fine is

<!-- formula-not-decoded -->

so Vi(a) can be interpreted as the avorage number of transitions per unit time.

The state and control at any time / are denoted by d(d) and 1(1). respectively. and stay constant between transitions. We use the following

I: The time of occurrence of the lith transition. By convontion, wo denote to == 0.

Th = 14 - 1a 1: The kith transition time interval.

<!-- formula-not-decoded -->

Wo consider a cost. function of the form

<!-- formula-not-decoded -->

where y is a given finction and 3 is a given positive discount parameter. Similar to discrete-time problems, an admissible policy is a sequenee o= 440.11,...f. where each pa is a function mapping states to controls with 1(i) € U(i) for all states i. Under o, the control applied in the interval (th, (h+1) is 16(24). Because states stay constant. between transitions, the cost fuction of a is given by

<!-- formula-not-decoded -->

He first consider the case where the rate of transition is the same for all states and controls; that. is,

<!-- formula-not-decoded -->

A little thought shows that the problem is then essentially the same as the one where transition times are fixed, because the control canot influence the cost. of a stage by aflocting the length of the next transition time interval.

Indeed, the cost (1.1) corresponding to a sequence {ch, 1)} can be expressed is

<!-- formula-not-decoded -->

We have (using the indopendence of the transition time intervals)

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The above expression for a yiekds (1 - 0) /18 = 1/(3 + v), so that from Eg. (1.3), we have

<!-- formula-not-decoded -->

From this equation together with Ey. (1.2) it follows that the cost of the problem can be expressed as

<!-- formula-not-decoded -->

Thus we are faced in eflect with an ordinary discrete-time problem where expected total cost is to be minimized. The effect of randomness of the transition times has been simply to appropriately scale the cost per stage.

To summarize, a contimous-time Markov chain problem with cost,

and rate of transition v that is independent of state and control is equivalent to a discrote-time Markov chain problem with discount factor

<!-- formula-not-decoded -->

and cost per stage given by or equivalently;

<!-- formula-not-decoded -->

In particular, Bolkman's equation takes the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

lu some problems, in addition to the cost (1.1), there is an extra expected stage cost gia) that is incurred at the time the control a is chosen at. state i, and is independent of the length of the transition interval. In that case the expected stage cost (1.5) should be changed to

<!-- formula-not-decoded -->

and Bellman's equation (1.6) becomes

## Example 1.1

A manufacturor of a specialty item processes ordors in batches. Orders arrive according to a Poisson process with rate o per mit time; that is, the successive interarrival intervals are independent and exponentially distributed with parameter v. For each order there is a positive coste per unit time that the order is unfilled. Costs are discounted with a discount parameter B &gt; 0. The setup cost for processing the orders is A. Upon arrival of a now order, the manufacturer must decide whether to process the current batch or to wait for the next. order.

Here the state is the number i of unfilled orders. If the decision to fill the orders at state i is made, the cost is A and the next transition will be lo state 1. Otherwise, there will be an average cost (ci) /(3 + ") up to the transition to the next state i + 1 Jef. By. (1.5)), as shown in Fig. 5.1.1. [Note that the setup cost. is incured immediately after a decision lo process the orders is made, so k' is not. discounted over the time interval up to the next

<!-- formula-not-decoded -->

transition: of. Pa. (1.9)] We are in offord faced with a disconned discretetime problem with positive but unbounded cost per stage. (le could also consider an alternative model where an upper bound is placed on the mmber of unfilled order. We would then have a discounted discrete limo problem with bounded cost por stage.)

Since Assumption P is satisfied (of. Section 3.1). Bollman's cquation holds and takes the form

<!-- formula-not-decoded -->

where a = v/(8 + «) is the ellective discount. factor lot. Eg. (1.1)]. Reasoning from first principles, wo see that fi) is a monotonically nondecreasing funstion of i, so from Bollman's cquation it follows that there exists a threshold i such that it is optimal to process the orders if and only if their mumbor excreds i.

Figure 5.1.1 Transition diagram for the continuous-time Markov chain of Exsunple 1.1. The transitions associated with the first contro! (do not fill the orders) are shown with solid lines, and the transitions associated with the second control (fill the ordors) are shown with broken lines.

<!-- image -->

## Nonuniform Transition Rates

We now argue that the more general case where the transition rate 1, (1) depends on the state and the control can be converted to the previous case of miform transition rate by using the trick of allowing fictitious transitions from a state to itself. Roughly, transitions that are slow on the average are speeded up with the understanding that sometimes after a transition the state may stay unchanged. To see how this works, let y be a new uniformn transition rate with e(a) So for all i and a, and define new transition probabilitios ly

<!-- formula-not-decoded -->

No refer to this process as the uniform version of the original (soo Fig. 5.1.2). No arguo now that leaving stato i at. a tate o,«) in the original process is statistically identical to leaving state i at the faster rate r. but returning back to i with probability 1- v,(a)/e in the now process. Equivalently, transitions are real (lead to a different state) with probability 4(1)/V&lt; 1. By statistical equivalence, we mean that, for any given policy r. initial state do. time f, and state i, the probability Ple(1) =110o.) is identical for the original process and its uniform version. We gire a proof of this fact in Exercise 5.1 for the caso of a finite mumber of states (soo also (Lip75), (Sor79), and (Ros$3b) for further discussion).

To summarize, wo can convert a continous-time Markor chain probkom with transition rates vi(a), transition probabilities pij(«), and cost

into a discrete-time Markos chain problem with discout factor

<!-- formula-not-decoded -->

where a is a uniform transition rate chosen so that.

<!-- formula-not-decoded -->

The transition probabilities are

<!-- formula-not-decoded -->

and the cost per stage is

<!-- formula-not-decoded -->

In particadar, Bellman's equation takes the form

<!-- image -->

Transition rates and probabilities for continuous-time chain

<!-- image -->

Transition probabilities for uniform version

Figure 5.1.2 'Transforming a contimous-time Markov chain into its uniform version through the use of fictitious self-transitions. The uniform version has a uniform transition rate e, which is an upper bound for all transition rates v. (a) of thie original, and transition probabilities pe(4) = (0, (4)/U)ma(2) for i # j, and ps, (4) = (w, (2)/0) p(4) +1-0, (1) /0 for j= i. In the example of the figure we have p, (a) = 0 for ill i and 1.

which, after some calculation using the proceding definitions, can be written

In the case where there is an extra expected stage cost ali.u) that is incurred at the time the control a is chose at state i, Bellman's equation becomes (of. Eg. (1.9)]

<!-- formula-not-decoded -->

## Undiscounted and Average Cost Problems

When the discount parameter B is zero in the preceding problem formulation, we obtain a continuous-time version of the undiscounted cost problem considered in Chapter 3. If in addition, the mmber of states is tie and her i soft did de in de, i stipe 2. However, when i = 0, it. is unnecessary to resort to uniformization. It can be seen that the problem is essentially the same as the discretetime problem with the same trusition probabilities but where the avorage transition cost at state i under a is the average cost per unit. time gi,u) amltiplied with the expected length 1/v, (a) of the transition interval. Thus Bellman's equation has the form

<!-- formula-not-decoded -->

After some caleulation. it can be seen that the above equation can also be obtained from Ed. (1.14) by setting i = 0.

In fact for undiscounted problems, the preceding argument does not depend on the character of the probability distributions of the transition Regardless of whether these distributions are exponential of not. one simply needs to multiply gi,u) with the average transition time corresponding to (i, 4) and then treat the problem as if it were a discrete-time

Thero is also a continuons-time version of the average cost. por stago problem of Chapter 4. The cost function has the tomo

We will consider this problem in Section 5.3 in a more general contest where the probability distributions of the transition times need not be exponential.

- -+

## 5.2 QUEUEING APPLICATIONS

We now illustrate the theory of the proceding section through some applications involving the control of queues.

## Example 2.1 (M/M/1 Queue with Controlled Service Rate)

C'onsider a single server quencing systom where customers arrive according to a Poisson process with rate A. The service time of a customer is exponontially distributed with parameter &amp; (called the service rate). Service times of customers aro indopendont and are also independont of customer interarrival times. The service rate pe can be solected from de closed subsce. A of an intorval 0,77 and can be changed at the times when a customer arrives or when i customer departs from the system. There is a cost. 4(a) per unit time for using rate ye and a waiting cost ci) per unit time when there are i customers in the system (waiting in quone of undergoing service). The idea is that one should be able to cat down on the customer waiting costs by choosing a faster service rate, which presumably costs more. The problem. roughly, is to solect the service rate so that. the sorvice cost is optimally traded off with the customer waiting cost.

No assume the following:

1. for some a E Al we have a &gt; A. (lo words, there is available a service rate that is fast enough to keep up with the arrival rate, thereby maintaining the queue longth bounded.)
2. The waiting cost function e is nonnegative, monotonically nondecreasing, and "convex" in the sonse

3. 'The service rate cost timotion y is nonnegative, and continous on (0. 7), with «(0) = 0.

The problem fits the tamework of this soction. The state is the minber of customers in the system, and the control is the choice of service rate following a customer arrival or doparture. The transition rate at state i is

<!-- formula-not-decoded -->

The transition probabilities of the Markos chain and its uniform vorsion for the choice

<!-- formula-not-decoded -->

are shown in Fig. 5.2.1

The effective discount factor is

<!-- image -->

Transition probabilities for continuous-time chain

<!-- image -->

Transition probabilities for the uniform version

Figure 5.2.1 Continuous-time Markor chain and uniform version for Example 2.1 when the service rate is equal to fe. The transition rates of the original Markor chain aro v,(4) = A 4 pe for states 1 2 1, and 4o60) = A for Stato 0. The transition rate for the uniform version is 1 = 1 + j.

and the cost per stage is

<!-- formula-not-decoded -->

The form of Bellman's equation is /of. 84. (1.11)]

<!-- formula-not-decoded -->

and for i = 1,2....

An optimal policy is to use at state i the service rate that minimizes the expression on the right. Thus it is optimal to use at state i the service rate

<!-- formula-not-decoded -->

where Ji) is the diflerential of the optimal cost.

<!-- formula-not-decoded -->

When the minimum in log. (2.2) is attained by more than one service rate de we choose by convention the smallest.! We will demonstrate shorts wut. A(i) is monotonically nondecreasiny. It will then follow from By. (2.2) (see

Fig. 5.2.2) that the optimal service rate y"(i) is monotonically nondecreasing. Thas, as the queue length increases, it is optimal to use a faster sorvice rate.

To show that. A(i) is monotonically nondecreasing, we use the value iteration method to generate a sequence of functions Ja from the starting function

<!-- formula-not-decoded -->

For t = 0, 1,.., Gef. Rig. (2.1)}, we have

<!-- formula-not-decoded -->

and for i = 1,2,...,

<!-- formula-not-decoded -->

For A = 0. I,... and i = 1,2,.., lot.

For completeness of notation, define also Ad(0) = 0. From the theory of Section 3.1 (see Prop. 1.7 of that section), we have (i) → Ji) as hi → oc. It. follows that wo have

<!-- formula-not-decoded -->

Therotore, it. will suflice to show that J.(i) is monotonically nondecreasing for evory hi. For this we nse induction. The assertion is trivially true for k = 0. Assuming that A,(e) is monotonically nondecreasing. we show that the same is true for Aan(i). Lot.

<!-- formula-not-decoded -->

From 8g. (2.3) we have, for all i = 0, 1,...,

<!-- formula-not-decoded -->

Similarly, we obtain. for i = 1,2,....

<!-- formula-not-decoded -->

Subtracting the last two inequalities, we obtain, for a = 1.2,...

<!-- formula-not-decoded -->

Using our convexity assumption on c(i), the fact 1 - 1--/(i + 1) = 11 #(i + 1) ≥ 0, and the induction hypothesis, we see that every term on the right-hand side of the preceding inequality is nonnegative. Therefore, Ahri(i + 1) 244+(1) for i = 1,2,.. From Eq. (2.1) we can also show that Alt (1) ≥ 0 = 41+(0), and the induction proof is complete.

<!-- image -->

Walk to rome is the pleader ate ) in ly li d

## Example 2.2 (M/M/1 Queue with Controlled Arrival Rate)

C'onsider the same queueing systom as in the previous example with the diflorenco that the service rato y is fixed, but the arrival rate A can be controlled. We assume that 1 is chosen from a closed subset. A of an interval fO, X, and there is a cost e(A) per unit time. All other assumptions of Example 2.1 are also in ollect. What we have here is a problem of How control, whereby wo want to trade off optimally the cost for Throttling the arrival process with the customer waiting cost.

This problem is vory similar to the one of Example 2.1. We choose as uniform transition rate

and construct the miform vorsion of the Markow chain. Bollman's ognation takes the form

<!-- formula-not-decoded -->

An optimal policy is to uso at stato ¿ the arrival rate

<!-- formula-not-decoded -->

whero, as before, A(i) is the differential of the optimal cost.

<!-- formula-not-decoded -->

As in Example 2.1, we can show that. A(i) is monotonically nondecreasing; so from Eg. (2.5) we see that the optimal arrival rate tends to decrease as the sustom becomes more crouded (i increases).

## Example 2.3 (Priority Assigmnent and the /c Rule)

Consider a quenes that share a single server. There is a positive cost ca per Imit time and por customer in each quone i. The service time of a customer of quone i is exponentially distributed with parameter pi, and all customer service times are independent. Assuming that we start with a given number of customers in each queue and no further arrivals occur, what is the optimal order for serving the customers? The cost. here is

<!-- formula-not-decoded -->

wherer. (0) is the miber of customers in the ith quene at time t, and Bis a positive discount parameter.

No lirst construct the uniform version of the problem. The construction is shown in Fig. 5.2.3. The discount factor is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where t is the mmber of customers in the ith quene after the kth transition (real or lictitions).

uniformn version is fe = max, 13.

<!-- image -->

We now cowrite the cost, in a way that is more convoniont for analysis. The idea is to transform the problem from one of minimizing waiting costs to one of maximizing savings in waiting costs through customer service. Tor k = 0.1..... defino i = lo if the be trustion to finds to a departure from queue i.

where and the corresponding cost is

<!-- formula-not-decoded -->

Denote also

<!-- formula-not-decoded -->

do: the initial number of customers in queue i.

Then the cost. (2.7) can also bo written as

<!-- formula-not-decoded -->

Therefore, instead of minimizing the cost (2.7), we can equivalently

<!-- formula-not-decoded -->

where ce can be viewed as the sammys in waiting cost rate obtained from the lite transition.

We now recognize problem (2.8) as a mulliarmed bandit problem. The « quenes can be viewed as separate projects. At each time, i nonempty queue, say i, is selected and served. Since a customer departure occurs with probability po/, and a lietitions transition that leaves the state unchanged occurs with probability 1--y,/a. the corresponding expected reward is

<!-- formula-not-decoded -->

It is evident that the problem falls in the deteriorating case examined at the end of Section 1.5. Therefore, after cach customer departure, il is optimal to serve the quone with maximum expected roward por stage (ile., engage the project with maximal index; ef. the end of Section 1.5). Equivalently 1ef. Eg. (29)l, il is optanal to serve the nonemply queue i for which per, as maximum. This policy is known as the de vale. A plays an important role in several other formulations of the priority assignment problem (see (BDM83), Har75al, and (Har75bl). We can view por as the ratio of the waiting cost rate o, by the average time 1/k, needed to serve a customer. Therefore, the de rule amounts to serving the quene for which the savings in waiting cost rate per unit average service time are maximized.

## Example 2.4 (Routing Policies for a Two-Station System)

Consider the system consisting of two quenes shown in Fig. 5.2.1. Customers arrive according to a Poisson process with rate and are routed upon arrival to one of the two queues. Service times are indepondent and exponentially distributed with parameter me, in the first quone and ee in the second queue. The cost is

where ,3, 61, and ca are given positive scalars, and mi (1) and de(1) denote the mumber of customers at time &amp; in queues land 2, respectively.

As carlior, we construct the uniform version of this problem with uniform rate

<!-- formula-not-decoded -->

and the transition probabilities shown in Fig. 5.2.5. We take as state space the set of pairs (i.j) of customers in queues 1 and 2. Bollman's equation takes

<!-- formula-not-decoded -->

where for any&amp; we donote

<!-- formula-not-decoded -->

From this equation we see that an optimal policy is to route an arriving customer to queue fif and only if the state (i,j) at the time of arrival belongs to the set

<!-- formula-not-decoded -->

Figure 5.2.4 Queueing system of Example 2.4. The problem is to toute each arriving customer to quebe for 2 so as to minimize the total average discounted waiting cost.

<!-- image -->

<!-- image -->

...

Transition probabilities for uniform version

Figure 5.2.5 Continuous-time Markov chain and uniform version for Example 2.1 when customers are conted to the feat quone. The states are the pairs of customer numbers in the two comes.

This optimal policy can bo chameterized better by some fother analysis. Intuitively, one expects that optimal conting can be achieved by sending i castomer to the quene that is "less crowded" in some sense. It is theretore natural to conjecture that, if it is optimal to route to the first queue when the state is (i.g), it mast be optimal to do the same when the first quone is even less crowded; that is, the state is (i-m.j) with m. 2 1. This is equivalent to saying that the set of states sy for which it is optimal to route to the first. quone is characterized by a monotonically nondecreasing threshold function F hy means of

<!-- formula-not-decoded -->

(see Fig. 5.2.6). Accordingly, we call the corresponding optimal policy a threshold policy.

Figure 5.2.6 'Typical threshold policy characterized by a threshold function Р..

<!-- image -->

We will demonstrate the existence of a threshold optimal policy by showing that the functions

<!-- formula-not-decoded -->

are monotonically nondecreasing ine for each fixed j, and in jfor each lixed i, respectively. We will show this property for A,: the proof for Ag is analogons. It will be suflicient to show that for all li= 0, Poo. The functions

<!-- formula-not-decoded -->

are monotonically nondecreasing in i for each fixed j, where a is gonerated by the value iteration method starting from the soro function; that is, Ja+ (i,) = (TOk)(i,j), where 'T is the DP mapping defining Bellman's equation (2.11) and Jo = 0. This is true because ali.j) → Ji.j) for all i. as A → x. (Prop, 1.6 in Section 3.1). To provo that. Affid has the desired

property, it is useful to first. vorify that, Infi,j) is monotonically nondecrensing in i (or i) for fixed j (or i). This is simple to show by induction or by arguing from first principles using the fact that Ji,j) has a f-stage optimal cost interpretation. Next, we uso Egs. (2.11) and (2.11) to write

<!-- formula-not-decoded -->

We now argue by induction. We have A}(i,j) = 0 for all (i,g). We assume that Af(i,j) is monotonically sondecreasing in i for fixed j, and show that the same is true for Af+'(i,j. This can be verified by showing that each of the torns in the right-hand side of Ey. (2.15) is monotonically nondecreasing in i tor fixcel j. Indeed, the first torm is constant, and the second and third torms are seen to be monotonically nondecreasing in i using the induction hypothesis for the case where i,g &gt; 0 and the carlier shown fact that ta(i, j) is monotonically nondecreasing in i for the case where i = 0 or j = 0. The last term on the right-hand side of Eq. (2.15) can be written as

<!-- formula-not-decoded -->

Since Af(i + 1,) and Af(i. j+ 1) are monotonically nondecreasing in i by the fore, each of the torms on the right-hand side of Eg. (2.15) is monotonically nondecreasing in i, and the induction proof is complete. Thus the existence induction hypothesis, the same is true for tho preceding expression. Thereof an optimal threshold policy is established.

Thore are a number of generalizations of the routing problem of this example that admit, ie similar analysis and for which there exist optimal poliries of the threshold type. For example, suppose that there are additional Poisson arrival processes with rates de and de at quones 1 and 2, respectively. The existonce of an optimal threshold policy can be shown by a nearly verbatim repotition of our analysis. A more substantive extension is obtained when there is additional sorvice capacite o that can be switched at the times of transition due to an arrival or service completion to serve a customer in quene for 2. Then we can similarks prove that it is optimal to route to queue 1 if and onl if (i.;) &amp; So and to switch the additional service capacity to quone 2if and only if (i + 1.j+ 1) E So. where Sy is given by Eg. (2.12) and is characterized by a threshokd function as in Eg. (2.13). For a proof of this and further extonsions. we refer to [laj81. which goneralizes and nailies several curlier results on the subioet.

## 5.3 SEMI-MARKOV PROBLEMS

Wo now consider a more gonoral version of the continons-time problem of Section 5.1. Westill have a finite or a constable mmber of states, but wo coplace transition probabilities with transition distributions Qu (T, «) that, for a given pair (i.«), specify the joint. distribution of the transition interval and the next state:

<!-- formula-not-decoded -->

We assume that for all states i and j, and controls a e li), Q(T, a) is known and that the average transition time is finite:

<!-- formula-not-decoded -->

Note that the transition distributions spocify the ordinary transition probabilities via

Continuous-time problems with goneral transition distributions as described above are called semi-Markon problems bocause, for any given policy, while at a transition time th the future of the systom statistically depends only on the curront state, at other times it may depend in addition on the time elapsed since the preceding transition. By contrast. when the transition distributions are exponential. The future of the system depends only on its curront state at all times. This is a consequence of the so called memoryless property of the exponential distribution. he our contest. this property implies that, for any time I between the transition times th and th+1, the additional time tits -I needed to elleet the next transition is independent of the time t- ta that the system has been in the current state

The difference from the model of Section 5.1 is that (,, (T. «) need not. be an exponential distribution.

(to see this, use the following generic calculation

<!-- formula-not-decoded -->

where r1 = 1 - 1h, 12 = tart - 1, and &amp; is the transition rate). 'Thus, when the transition distributions are exponential, the state evolves in continuous time as a Markor process, but this need not be true for a more general distribution.

## Discounted Problems

Let us first consider a cost function of the form

<!-- formula-not-decoded -->

where Is is the completion time of the Nob transition, and the foction g and the positive discount parameter (8 are given. The cost function of an admissible N-stage policy a = 440,41,.,N-1) is given by

Wo see that for all states i we have

<!-- formula-not-decoded -->

where / (i) is the (N- 1)-stage cost of the policy r1=64142....UN-13 that is used after the first stage, and G(i.«) is the expected single stage cost corresponding to (i, a). This latter cost is given by

or equisalently, since for-ol = (1-6-17)/13,

<!-- formula-not-decoded -->

If we donote

<!-- formula-not-decoded -->

wo see that Ta. (3.2) can be written in the form

<!-- formula-not-decoded -->

which is similar to the corresponding equation for discounted discrete-time problems we have me (a) in place of op, (a)l.

The expression (3.5) motivates the use of mappings l' and T, that are similar to those nsed in Chapter 1 for discounted problems. Let us define for a function o and a stationary policy a,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then by using By. (3.5), it can be scen that, the cost. function on of an infinite horizon policy a = 140.1.. can be expressed as

where do is the zero function (Jofi) = 0 for all ij. The cost of a stationary policy a can be expressed as

The discounted cost analysis of Soction 1.2 carries through in its entirets; provided we assume that:

- (a) gli. a) land hence also G(i. a)] is a bounded function of i and u.
- (1) The maximum over (i. 4) of the sum L, My (4) is less than one: that

<!-- formula-not-decoded -->

Under these circumstances, the mappings T and Ta can be shown to be contraction mappings with modulus of contraction o compare also with Prop. 2.1 in Section 1.2]. Using this feet, analogs of Props. 2.1-2.3 of Section 1.2 can be readily shown. In particular, the optimal cost function J* is the unique bonded solution of Bellman's cquation J = T. or

<!-- formula-not-decoded -->

In addition, there are analogs of soveral of the computational methods of Section 1.3, including policy iteration and linear programming.

We note that for the contraction property a &lt; 1 foR. Do. (3.8) to hold. it is sufficiont that there exist o&gt; Onde: O such Chat. the transition time is greater than 7 with probability greater than e &gt; 0; that is, we have for all i and « € U(i).

What is happening here is that essentially we have the equivalent of a discrete-time disconted problem where the discount factor depends on i and 1. In fact, in Exorcise 1.12 of Chapter 1, a data transformation is gison. which converts such a problem to an ordinary discrete-time discounted problem where the discount factor is the same for all i and a. With a little thought. it can be seen that this date transformation is very similar to the uniformization process we discussed in Section 5.1.

<!-- formula-not-decoded -->

In the case where the state space is countably infinite and the func11009(4.1) is nod. bounded, the mappings 'T and Ta are not contraction mappings, and a disconted cost, analysis that, parallels the one of Section 1.2 is not possible. levon in this case, however, amalogs of the results of Scotion 3.1 can often bo shown under appropriato conditions that parallel Assumptions Pand N of that section.

We finally note that in some problems, in addition to the cost. (3.1), there is an extra expected stage cost gi.@) that is incurred at the time the control a is choson at state i, and is independont of the length of the transition interval. In that case the mappings Tand T, should be changed

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Another problem variation arises when the cost per mit tine g depends on the next state j. In this problem formulation, once the system goes into state i. a control a € U(i) is selected, the next state is determined to be j with probability po (a), and the cost of the next transition is gli, u. j) rig(4) where to,«) is random with distribution Qig(t,)/Pidu). lu this case, C(i, «) should be defined by

<!-- formula-not-decoded -->

(of. 1g. (3.3)) and the preceding development goes through without modification.

## Example 3.1

Consider the manufacturer's problem of Example I.d, with the only differenco that the times between the arrivals of successive orders are uniformly distributed in a given interval (0, max) instead of being exponentially dis tributed. Let D and Nf denote the choices of filling and not filling the? orders, respectively. The transition distributions are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The olloctive cost por stage C of De. (3.3) is given by

<!-- formula-not-decoded -->

and where

<!-- formula-not-decoded -->

The scalars me, of big GiB. 1) Meat are nonzero ince

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Bellman's equation has the form

<!-- formula-not-decoded -->

As in Example 1.1, we can conclude that there exists a threshold i" such that it is optimal to fill the orders if and only it their number exceeds it

## Example 3.2 (Control of an M/D/1 Queue)

Consider a single server queue where customers arrive according to a Poisson process with rate A. The service timo of a customer is deterministic and is equal to 1/fe where ye is the service rate provided. The arrival and service rates 1 and de can be selected from given subsets A and Al, and can be changed only when a customer departs from the system. There are costs q(X) and ry) per unit time for using rates 1 and a, respectively, and there is a waiting cost c(i) per unit time when there are i customers in the system (waiting in quenc or undergoing service). We wish to find a rate-sotting policy that minimizes the total cost when there is a positive discount. Daramotor 3.

to use as state the number of customers in the system; if we allowed the arrival rate to also change when a customer arrives, the time already spent in service by the customer found in service by the arriving customer would have to be part of the state.

This problem bears similarity with Examples 2.1 and 2.2 of Section 5.2. Note, however, that while in those examples the rates can be changed both when a customer arrives and when a customer departs, here the rates can be changed only when a customer departs. Because the service time distribution is not exponential, it is necessary to make this restriction in ordor to be ablo

The transition distributions are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where pi, (A,) are the state transition probabilities. He can be seen that for i &gt; land j≥ i- 1, p, (d, 1) can be caleulated as the probability that j-4+1 arrivals will occur in an interval of length (0, 1/1. In particular, we have

<!-- formula-not-decoded -->

Using the above formulas and ligs. (3.3)-(3.4) and (3.6)-(3.7), one can write Bellman's equation and solve the problem as if it were essentially a discretetime disconted problem.

## Average Cost Problems

A natural cost fimotion for the contimous-time average cost problem would be

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where 1y is the complotion time of the eth transition. This cost function is also reasonable and turns out to be analytically convenient. We note, however, that the cost fnctions (3.11) and (3.12) are equivalent under the conditions of the subsequent analysis. although a rigorous justification of this is beyond our scope (see (Ros70), p. 52 and p. 160 for related analysis).

No assumo that there are a states, denoted lo.,n, and that the control constraint sot (s) is finite for each state i. For cach pair (i.«), we donote by Gi. a) the single stage expected cost corresponding to state i

<!-- formula-not-decoded -->

where 7,(«) is the expected value of the transition time corresponding to (i, «1):

<!-- formula-not-decoded -->

If the cost per mit time o depends on the next state i, the experted transition cost. G(i, a) should be defined by

<!-- formula-not-decoded -->

and the following analysis and results go through without modification.] He assume throughont the romainder of this section that.

<!-- formula-not-decoded -->

However, we will use instead the cost function

The cost function of an admissible policy a = 4040,14,.. is given by

<!-- formula-not-decoded -->

Our carlier analysis of the discrete-time average cost problem in Chapter 4 suggests that under assumptions similar to those of Section 1.2, the cost lu(i) of a stationary policy 4e, as well as the optimal average cost por stage 1*(i), are independent of the initial state i. Indeed, we will see that. the character of the solution of the problem is determined by the structure of the embedded Markoo chain, which is the controlled discrete-time Markor chain whose transition probabilities are

In particular, we will show that Jaft) and +(s) are independent of i if and only if the same is true for the embedded Markow chain problem. For example, we will show that Jai) and a*(i), are independent. of i if all stationary policies y are unichain; that is, the Markov chain with transition probabilities po (ni)) has a single recurent, class.

We will also show that. Bellman's cquation for average cost semiMarkor problems resembles the corresponding discrote-time equation, and takes the form

<!-- formula-not-decoded -->

As a special case, when 7(4) = 1 for all (i,2), we obtain the corresponding discrete-time equation of Chapter 4. We illustrate Bellman's equation (3.16) for the caso of a single michain policy with the stochastic shortest. path argument that we used to prove Prop. 2.5 in Section 1.2.

Consider a unichain policy a and withont loss of generality assume that state n is a recurrent state in the Markor chain corresponding to y. For cach state i #n let C, and 'T; be the expected cost and the expected time. respectively, up to reaching state a for the first time starting from i. Let also C, and Ta be the expocted cost. and the expected time, respectively. up to returning to a for the first time starting from n. We can view C, as the costs corresponding to ye in a stochastic shortest path problem where " is a termination state and the costs are Clin(i)). Since ye is a propor policy for this problem, from Prop. 1.1 in Section 2.1, we have that the scalars C; solve uniquely the system of eqnations

<!-- formula-not-decoded -->

Similarly, wo can viow 'T, as the costs coresponding to pe in a stochastic shortest path problem where o is a tormination state and the costs are Ti(P(i)), so that the T; solve uniquely the systom of equations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Donote

Multiplying By. (3.18) by do and subtracting it from By. (3.17), we obtain for all i = 1....,n,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By dolining and by noting that from Ey. (3.19) we have

<!-- formula-not-decoded -->

we obtain for all i = 1.....n,

<!-- formula-not-decoded -->

which is Bollman's equation (3.16) for the case of a single stationary policy 11.

We have not yet proved that the scalar 1, of Ey. (3.19) is the average cost per stage corresponding top. This fact will follow from the following proposition, which parallels Prop. 2.1 in Section 1.2 and shows that if Bollman's cquation (3.16) has a solution (A,/), then the optimal average cost is equal to d aud is indepondent of the initial state.

Proposition 3.1: Il a scalar 1 and an a-dimensional vector l satisfy

<!-- formula-not-decoded -->

then A is the optimal average cost per stage J*(i) for all i,

<!-- formula-not-decoded -->

Farthermore, if p* (i) attains the minimun in Ey. (3.22) for each i, the stationary policy * is optimal; that is, Ju(i) = 1 for all i.

Proof: For any ye consider the mapping Ta: l" to l" given by

<!-- formula-not-decoded -->

and the vector 7(a) and matrix P, given by

<!-- formula-not-decoded -->

Let a = 140.11,...) be any admissible policy and N be a positive integer. No have from Eq. (3.22):

By applying Tune to both sides of this relation, and by using the monotonicits of Tin-2 and Ed. (3.22), we see that

<!-- formula-not-decoded -->

Continuing in the same manner, we finally obtain where In(t) is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the ith component of the vector In(t) is Elen 1c0 = i.my. the expected value of the completion time of the Nth transition when the initial stato is i and a is used. Note also that. cquality holds in Eg. (3.21) if pa (i) attains the minimun in Eq. (3.22) for all hand i. I can bo seen that

Using this relation in lo 13.20) and disaling De lee Leo -101. 1e obtain for all i

<!-- formula-not-decoded -->

Taking the limit is N → do and using the fact limy-x EftN | 10 = i. 7} = 0u /ct. By (3.15)), we see that

<!-- formula-not-decoded -->

with equality if pa(i) attains the minimun in ly. (3.22) for all kiand i. Q.E.D.

By combining Prop. 3.1 with By. (3.21), we obtain the following:

Proposition 3.2: Lot. y be a michain policy. Then:

- (a) There exists a scalar de and a vector h, such that,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

- (b) Let t be a fixed state. The system of the n + 1 linear cquations

<!-- formula-not-decoded -->

in the a-t 1 unknowns 1, 4(1),..., k(a) has a mique solution.

Proof: Part (a) follows from Prop. 3.1 and Ed. (3.21). The proof of part (1)) is identical to the proof of Prop. 2.5(b) in Section 4.2. Q.E.D.

To establish conditions under which there exists a solution (1, l) to Bellman's equation (3.22), wo formulato a corresponding discrete time arcrage cost problem. Let g be any scalar such that.

<!-- formula-not-decoded -->

for all i and a E U(i) with pi(u) &lt; 1. Define also for all i and u € U(i).

<!-- formula-not-decoded -->

It can be seen that we have for all i and i

<!-- formula-not-decoded -->

We view pu(«) as the transition probabilities of the discrete-time average cost problem whose expected stage cost corresponding to (ia) is

<!-- formula-not-decoded -->

We call this the anliary descrete lome andrage cond problem. The follow ing proposition shows that this problom is essentially equivalent with our original semi-Markov average cost problem.

Proposition 3.3 If the scalar 1 and the vector i satisfy

<!-- formula-not-decoded -->

then 1 and the vector h with components

<!-- formula-not-decoded -->

satisfy Bellman's equation

<!-- formula-not-decoded -->

for the semi-Markov average cost. problem.

Proof: By substituting Eys. (3.29), (3.31), and (3.33) in Er. (3:32), wo obtain after a straightforward calculation

This implies that the minimum of the expression within brackets in the right-hand side above is zero, which is equivalent to Bellman's equation (3.31). Q.E.D.

- -1

Note that in viow of Ey. (3.30), the auxiliary discrete-time average cost problem and the semi-Markov average cost problem have the samo probabilistic structure. In particular, if all stationary policies are nichain for one problem, the same is true for the other. Thus, the results and algorithms of Sections 1.2 and lis, when applied to the auxiliary discretetime problem, yiold results and algorithms for the somi-Markov problem. For example, value iteration, policy iteration. and lincar programming can Do applied to the anxiliary probion in order to solve the somi-Markor problem. We state a partial analog of Prop. 2.6 from Soction 1.2.

Proposition 3.4: Consider the semi-Markor average cost problem, and assume either one of the following two conditions:

- (1) Every policy that is optimal within the class of stationary policies is unichain.
- (2) For every two states i and j, there exists a stationary policy n (depending on i and j) such that, for some k,

<!-- formula-not-decoded -->

Then the optimal average cost per stage has the same value 1 for all initial states i. Furthermore, 1 together with a vector h satisfies Bellman's cquation (3.34) for the semi-Markor average cost problem.

Proof: By Prop. 2.6 in Section 1.2, under either one of the conditions stated, Bellman's equation (3.32) for the auxiliary discrete-time average cost problem has a solution (A,), from which a solution to Bellman's equation (3.31) can be extracted according to Prop. 3.3. Q.E.D.

## Example 3.3:

Consider the average cost vorsion of the manufacturer's problem of Example 3.1. Here we have

<!-- formula-not-decoded -->

where and Nd donote the decisions to fill and not fill the orders, respectivels: Belhman's equation takes the form

We leave it as an exerciso for the reader to show that. there exists a threshold i such that it is optimal to fill the orders if and only ifs exceeds i

## Example 3.4: (LiR.71)

Consider a person providing a cortain type of service to customers. Polental customers arrive according to a Poisson process with rate e: that is the customer's interarrival times are independent and exponontially distributed with paramotor r. Each customer offers ono of a pairs (m, T.), i == 1....,11, where m, is the amount. of money offered for the service and T, is the average amount of time that will be required to perform the service. Successive offers are independent and offer (maT) occurs with probability De, where E", M, = 1. An offer may be rejected, in which case the customer leaves. or may be accepted in which caso all offers that arrive while the customer is being served are lost. The problem is to determine the acceptance-rejection policy that maximizes the service provider's average income per unit time.

Let us donoto by i the state corresponding to the offer (m,, 'Ti), and lot, and A denote the accept and reject decision, respectively. We have

<!-- formula-not-decoded -->

Bellman's equation is given by

<!-- formula-not-decoded -->

11 follow's that an optimal policy is to accopt offer (iT if

<!-- formula-not-decoded -->

where -1 is the optimal averago income por amit. time.

## 5.4 NOTES, SOURCES. AND EXERCISES

The idea of using uniformization to convert contimous-time stochastic control problems involving Markos chains into discrete timo problems gained wide attention following (lip75): see also (BeR87).

Control of quencing systoms has been rescarched extonsivoly. For additional material on the problom of control of arrival rate or sorvice rate (of. Examples 2.1 and 2.2 in Section 5.2), soo (BWN92), [CoR87), [CoV8 1), (RVW82), (Sob82], [St27d), and (Stids). For more on priority assignment and routing (ef. Examples 2.3, 2.4 in Section 5.2), sec (BDN83, BaD&amp;1]. (BeT896, (BhE91, (CoV84), (MarSal, (Mar75b, (Palot, (SuCot, and (yR91, [Cr091), (EVW80), (EpV89), (Maj81), (LiK84, (TSC92), (ViE88], respectively.

Semi-Markov decision models were introduced in Jew63) and are also discussed in (Ros70).

## EXERCISES

## 5.1 (Proof of Validity of Uniformnization)

Complete the details of the following argument, showing the validity of the of transition rates on the control. Let p(t) be the row vector with coordinates

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

whore p(0) is the row vector with ith coordinate equal to oue if to = i and zero otherwise, and the matrix has elements

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From this we obtain whore

C'onsider the transition probability matrix 13 of the miform version

<!-- formula-not-decoded -->

whore v &gt; 0,, i = 1,....n. Consider also the following equation:

Uso these relations to write

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Verify that, for 1 = 1,...,a. we have

<!-- formula-not-decoded -->

5.2

Consider the Al/Af/1 queneing problem with variable service rate (Example 2.1 in Section 52). Assame that no arrivals are allowed (A = (), and one can either serve a customer al rate de or refuse service (11 = {0, 4}). Lot the cost rates for customer waiting and servier he cfs) = ci and qud), respectivels: with 4(0) = 0.

(a) Show that an optimal policy is to always serve an available customer if

<!-- formula-not-decoded -->

and to always refuse service otherwise.

- (1)) Analyze the problem when the cost rate for waiting is instead c(i) = ci.

5.3

A person has an asset to sell for which she receives ollers that can take one of a values. The times between successive ollers are random, independent. and identically distributed with given distribution. Find the offer acceptance policy that maximizes lifosy, where T is the time of sales is the sale price, and « € (D, 1) is a discount. factor.

5.4

Analyzo the priority assignment problem of example 2.3 in Section 5.2 within times are indopendont but not exponontially distributed. Consider both the the semi-Markov context of Section 5.3, assuming that the customer service discounted and the average cost cases.

5.5

An unemployed worker receives job offers according to a Poisson process with rater. which she may accept or reject. The offered salary (per unit timo) takes one of « possiblo valnes me.,, with given probabilities, indopondently of preceding offors. A sho accepts an offer at salary w,, she koops the job for a random amount of time that has expocted value 1. If she rejects the offer, sho receives unemployment compensation e (per unit time) and is eligible to accept, future offers. Solve the problem of maximizing the worker's average income per nuit. time.

## 5.6

Consider a single server quencing sostem where the server may be either on or off. Cnstomers arrive according to a Poisson process with rate d, and their service times are indepondent, identically distributed with givon distribution. Each time a customer doparts, the server may switch from on to off at a fixed cost. C'o or from off to on at a fixed cost. Ca. 'There is a cost e per unit time and customer residing in the system. Analyze this problem as a semi-Markov problem for the disconted and the avorage cost cases. Iu the latter case, assame that the quone has limited storage, and that customers arriving when the quone is full are lost.

5.7

Consider a semi-Markov version of the machine replacement problem of Ex- ample 2.1 in Section 1.2. Here, the transition times are random, independent. and have givon distributions. Also gfi) is the cost per unit time of operat.ing the machine at state i. Assume that proto &gt; O lor all i &lt; n. Derive Bellman's cquation and analyze the problem.

## References

(ABF93) Arapostathis. A., Borkar, V., Fernandez-Gaucherand, B.. Ghosh. M., and Marcus, S., 1993. "Discrete-Time Controlled Markov Processes with Average Cost. Criterion: A Surves:" STANd. on Control iuld Opti- mization, Vol. 31, pp. 282-341.

(ANT9:3) Archibald, T. W., MoKimon, k. I. M., and 'Thomas. L. C., 1993. "Serial and Parallel Value Iteration Algorithms for Discounted Markov De-

(Ash70) Ash, R. B., 1970. Basic Probability Theory. Wiley. N. Y.

cision Processes," Enr. J. Operations Research, Vol. 67, py. 188-203.

[AyR91) Ayonn, S., and Rosberg, Z., 1991. "Optimal Routing to Two

Parallel Heterogeneons Servers with Resequencing," BEEE Trans. on Altomatic Control, Vol. 36. pp. 1136-1119.

(BB593) Barto, A. G., Bradtko, S. J., and Singh. S. P., 1993. "RealTime Learing and Control Using Asynchronons Dynamic Programming." Comp. Science Dept. Tech. Report 91-57, Uuiv. of Massachusetts, Artificial Intolligence, Vol. 72, 1905. pp. 81-138

[BDM83] Baras, J. S., Dorsey, A. J., and Makowski. A. M., 1983. "Two Competing Queues with Linear Costs: The pe-Rude is Often Optimal," Koport SKR 83-1, Department of Eloctrical Engineering, University of Marylund.

(B1P90) Benvenistc, A., Motivier, M., and Prourier, P'., 1990. Adaptise

(BIT91a) Bertsimas, D., Paschalidis, 1. Co, and Tsitsiklis, J. N.. 1991. "Optimization of Multiclass Quencing Networks: Polyhedral and Nonlinear Characterizations of Achievable Performance." Annals of Applied Probability. Vol. 4. pp. 13-75.

Algorithms and Stochastic Approximations, Springer-Vorlag, N. Y.

(BY'T916] Bertsimas, D., Paschalidis, T. Co, and Tsitsiklis, J. N., 1991. "Branching Bandits and Alimov's Problem: Achiovable Region and Side Constraints," Proc. of the 1991 IBEE Conference on Decision and Control, pp. 174 • 180, ICEE Trans. on Automatic Control, to appear.

[BWN92) Blanc, J. P. Co, de Waal, L. R., Nain, P., and Towsley, D., 1992. "Optimal Control of Admission to a Multiserver Queue with Two Arrival Stacans," TEEB Trans, on Automatic Control, Vol. 37, pp. 785-797.

Vol. AC-26, pp. 1106-1117.

[Bai93) Baird, L. C., 1993. "Advantage Updating." Report WL-TR-93-

1116, Wright. Patterson AFB, OH.

(Bai91) Baird, L. C., 1994. "Reinforcement. Learning in Continuous Time: Advantage Updating," International Conf. on Neural Networks, Orlando,

Ba195| Baird, L. C., 1995. "Residual Algorithms: Reinforcement. Learning with Function Approximation," Dept. of Computer Science Report, U.S. Air Force Academy, CO.

541-553.

(BeC89) Bertsekas, D. P., and Castanon, D. A., 1989. "Adaptive Aggregation Methods for lufinite Horizon Dynamic Programming." IECE Trans. on Automatic Control, Vol. AC-31, p. 589-598.

(BeN93) Bertsimas, D., and Nino-Mora, J., 1993. "Conservation Laws, Extended Polymatroids, and the Multiarmed Bandit, Problem: A Unified Polyhedral Approach," Mathematics of Operations Research, to appen.

[BeR87) Bentler, F. J., and Ross. K. W., 1987. "Uniformization for SomiMarkov Decision Processes Under Stationary Policies," J. Appl. Prob., Vol.

24, pp. 399-420.

(BeS78) Bertsekas, D. P., and Sluevo, S. B., 1978. Stochastic Optimal Con-

1rol: The Discrete Time Case, Academic Press, N. Y.

(B0579) Bertsckas, D. P., and Shreve, S. E., 1979. "Existence of Optimal Stationary Policies in Deterministic Optimal Control," J. Math. Anal. and

Appl., Vol. 69, po. 607-620.

[BeT89al Bertsokas. D. P., and Tsitsiklis, J. N., 1989. Parallel and Distributed Computation: Numerical Methods, Prentice Hall, Englewood Clifis, N. J.

(Bo'191a) Bertsekas, D. D., and Tsitsiklis, J. N., 1991. "A Survey of Some

Aspocts of Parallel and Distributed Iterative Algorithms," Automatica,

Vol. 27. pp. 3-21.

(BoT916) Bertsokas. D. P.. and Tsitsiklis, J. N., 1991. "An Analysis of Stochastic Shortest Path Problems," Math. Operations Research, Vol. 16. Рр. 580-595.

(B‹157) Bellman, K., 1957. Applied Dynamic Programming, Princeton University Press, Princeton, N. J.

(Bor71) Bertsokas, D. P., 1971. "Control of Uncertain Systems With it Set-Membership Description of the Uncertainty," Ph.D. Dissertation, Massachusetts Institute of Technology.

[Ber72) Bortsokas, D. P., 1972. "Infinite Timo Reachability of Stato Space Regions by Using Feedback Control," Hobb Trans. Automatic Control. Vol. АС-17. рр. 601-613.

(301732) Bertsokas, D. P., 1973. "Stochastic Optimization Problems with Nondifferentiable Cost Ructionals," J. Optimization Theory Appl., Vol. 12. py. 218-231.

(Bor736) Bertsckas. D. P., 1973. "Linear Convex Stochastic Control Prob lems Over an Intinite Tine Horizon." HEBE Trans. Automatic Control, Vol.

АС-18. рр. 31.1-315.

(Ber75) Bertsckas, D. P.. 1975. "Convergence of Discretization Procedures in Dynamic Programming. TEEE, Trans. Automatic Control, Vol. AC-20, pD. 415-119.

(Ber76) Bertsekas, D. P.. 1976. "On Error Bounds for Successivo Approximation Methods," IECE Trans. Automatic Control. Vol. AC-21. pp. 391-

Рр. 138-164.

(Ber82al Bertsckas. D. 1, 1982. "Distributed Dynamic Programming." LEEE Trans. Automatic Control, Vol. AC-27. pp. 610-616.

[Bor826 Bertsckas, D. P., 1982. Constrained Optimization and Lagrange

Multiplier Methods. Academic Press. N. Y.

[Ber931 Bertsokas. D. P., 1993. "A Gonerie Rank Ono Correction Algorithm for Markovian Decision Problems," Lab for Info. and Decision Systems Report LIDS-P-2212, Massachusets Institute of Techmology, Operations Research Lotters, lo appear.

[Ber95a) Bertsekas, D. P., 1995. Nonlinear Programming, Athena Scientific,

Belmont, MA, to appear.

(Ber951] Bertsekas, D. P., 1905. "A Counteresample to Temporal Ditorences Learning." Neural Computation, Vol. 7, pp. 270-279.

(Ber956) Bertsokas, D. P., 1995. "A Now Value Itoration Method for the Average Cost. Dynamic Programming Problon," Lab. for Info, and Decision Systems Report, Massachusetts Institute of 'Technology.

(BhE91) Bhattacharya. P. P., and Ephremides, A., 1991. "Optimal Allocations of a Server Between Two Queues with Due Times," ICEE Trans. on Automatic Control, Vol. 36. pp. 1117-1123.

[Bil8:3] Billingsley, P., 1983. "The Singular Function of Bokd Play," American Scientist, Vol. 71, pp. 392-397.

(Ba62) Blackwell, D., 1962. "Discrete Dynamic Programming," Ann. Math.

Statist., Vol. 33, pp. 719-726.

(31a65) Blackwell, D., 1965. "Discounted Dynamic Programming," Ann.

Math. Statist., Vol. 36. pp. 226-235.

[Bla70) Blackwell, D., 1970. "On Stationary Policies," J. Roy: Statist. Soc:

Ser. A, Vol. 133, pp. 3.3-38.

(Bor89) Borkar, V. S., 1989. "Control of Markov Chains with Long-Rum A vorago Cost, Criterion: The Dynamic Programming Equations," SIAM J. on Control and Optimization, Vol. 27. pp. 642-657.

(Bro65) Brown, B. W., 1965. "On the Iterativo Method of Dynamic Programming on a Pinite Space Discrete Markov Process." Ann. Math. Statist.. Vol. 36, pp. 1279-1286.

Research Letters, Vol. 11. pp. 33-37.

(Cav89al Cavazos-Cadena, R., 1989. "Necessary Conditions for the Oplimality Equations in Average-Reward Markov Decision Processes," Sys. Control Letters, Vol. 11, py. 65-71.

(Cav89b| Cavazos-Cadona, R., 1989. "Weak Conditions for the Existence of Optimal Stationary Policies in Average Markov Decisions Chains with Unbounded Costs," Hyberetika, Vol. 25, pp. 115-156.

(Car91| Cavazos-Cadena, R., 1991. "Recent Results on Conditions for the Existence of Averago Optimal Stationary Policies." Annals of Operations Rescarch, Vol. 28. 3p. 3-28.

(Ch189, Chow, C.-S., and Tsitsiklis, J. N., 1989. "The Complexity of

Dynamic Programming," Journal of Complexity, Vol. 5, pp. 106-488.

(CH'T91 Chow, C.-S., and Tsitsiklis, A. N., 1991. "An Optimal One Way Multigrid Algorithm for Discrete Time Stochastic Control," IEED Trans. on Automatic Control, Vol. AC-36, pp. 898-911.

(CoR87) Courcoubetis, C*. A., and Reinan, M. 1., 1987. "Optimal Control of a Queueing System with Simultancous Service Requirements," ISEE

Trans. on Automatic Control. Vol. AC-32, pp. 717-727.

(CoN'81) C'ourconbotis. Co, and Varaiga, D. P.. 1981. "The Sorvice Process with beast Thinking Time Maximizes Kesourer Utilization," IBE Thus. Automatic Control. Vol. AC-29, рр. 1005-1008.

[0/091) Cruz. R. D. and Clual. M. C. 1991. "A Alinimas Approach to a Simple Routing Problem, ABBE Trans. on Automatic Control. Vol. 136.

Фр. 1121-1435.

(D'Ep60) D'Epenous. F.. 1960. "Sur am Problemo de Production ot de Stockago Dans l'Alcatoire." Rev. Pancaise And. Aufor. Recherche Operarionnelle, Nol. 1d. (English Transl: Management Sci., Vol. 10. 1963. po.

(D:63) Dantzig. G. 13., 1963. Linear Programming and Extensions. Princeton Univ. Press. Princeton, N. J.

[Don67) Denardo. E. V., 1967. "Contraction Mappings in the Theory Underlying Dynamic Programming." SIAM Review, Vol. 9, pp. 165-177.

[Der'70] Derman, C. 1970. Finite State Markovian Decision Processes. Academic Press, N. Y.

[DuS65) Dubins. L., and Savage. L. M., 1965. How to Gamblo If You Must. McGraw-Hill. N. Y

Dy Y79) Dynkin. E. B., and Suskerich, A. A., 1979. Controlled Markos

Processes, Springer-Verlag. N. Y.

[EVW'80] Ephromides, A., Varaisa, P. P., and Walrand, J. C., 1980. "A Simple Dymamic Routing Problem," HEED Trans. Automatic Control. Vol.

АС-25, рр. 690-693.

(EaZ62) Eaton, J. H., and Zadch, L. A., 1962. "Optimal Pursuit Strategies in Discrete State Probabilistie Sistems." Trans. ASME Ser. D. J. Basic Eng.. Vol. 84. pp.23-29.

(EpV89) Epbremides, A., and Verdu, S.. 1989. "Control and Optimization Methods in Commmication Notwork Problems." IEED Trans. Automatic

Control, Vol. AC-34. 1p. 030-912.

[FAM90] Fernández-Gaucherand. E., Arapostathis, A., and Marcus, S. I., 1090. "Romarks on the Existence of Solations to the Averago Cost. Optimality Equation in Markov Decision Processes." Systems and Control

Letters, Vol. 15.00. 125-132.

(FAM191) Fernández-Ganchorod, E., Arapostathis, A., and Marcus, S. I., 1991. "On the Average Cost. Optimality Equation and the Structure of (p)timal Policies for Partially Observable Markov Decision Processes." Anals of Operations Research, Vol. 29, pp. 439-470.

10591 Feinberg. E. A.. and Shwarte, A., 1991. "Markor Decision Models

- -1

with Weighted Discounted Criteria," Mathomatics of Operations Research, Vol. 19. 1p. 1-17.

(Fei78) Feinberg, B. A., 1978. "The Existence of a Stationary c-Optimal Policy for a Finite-State Markor Chain," Theor. Prob. Appl., Vol. 23, pp. 297-313.

(Fri92a) Feinberg, E. A., 1992. "Stationary Strategies in Borel Dynamic Programming," Mathematics of Operations Research, Vol. 125, pp. 87-96.

(£0920 Feinberg, E. A., 1992. "A Markov Decision Model of a Search Process," Comtemporary Mathematics, Vol. 125, pp. 87-96.

[Fox71) Fox, B. L., 1971. "Finite State Approximations to Denunerable State Dynamic Progams," J. Math. Anal. Appl., Vol. 34, pp. 665-670.

(Gal95) Gallager, R. G., 1995. Discrote Stochastic Processes, Kluwer, N. Y. (Gho90) Ghosh, M. K., 1990. "Markov Decision Processes with Multiple Costs," Operations Research Letters, Vol. 9. pp. 257-260.

(GiJ74) Gittins, J. C., and Jones, D. M., 1974. "A Dynamic Allocation Index for the Sequential Design of Experiments," Progress in Statistics (J. Gani, «d.), North-Holland, Amsterdam, pp. 241-266.

(Ca179) Gittins, J. C., 1979. "Bandit. Processes and Dynamic Allocation

Indices," J. Roy: Statist. Soc., Vol. B, No. 11, pp. 1.18-164.

(MB/91) Harmon, NJ. E., Baird, L. C., and Klopf, A. H., 1994. "Advantage Updating Applied to a Differential Game," Presented at NIPS Conf., Denver, Colo.

(HIIL91) Hornandez-berma, O., Honnot, J. C., and Lasserre, J. B., 1991. "Average Cost Markov Decision Processes: Optimality Conditions," J.

Math. Anal. Appl., Vol. 158, pp. 396-106.

(Ma186) Maurio, A., and 1'Ecuyer, P., 1986. "Approximation and Bounds in Discrete Event. Dynamic Programming." BEEB Trans. Automatic Control, Vol. AC-31, pp. 227-2:35.

(Hajs-1 Hajek, B., 1981. "Optimal Control of Two Interacting Sorvice Stations," ABEE Trans. Automatic Control, Vol. AC-29, pp. 491-499.

(Iar75a) Harrison, J. M., 1975. A Priority Queue with Discounted Lincar Costs," Operations Research, Vol. 23, pp. 260-269.

(Har75b Harrison, J. M., 1975. "Dynamie Scheduling of a Multiclass Queue: Discount Optimality," Operations Rescarch, Vol. 2:3, pp. 270-282.

(Has08) Hastings, N. A. J., 1968. "Some Notes on Dynamic Programming and Replacement," Operational Research Quart., Vol. 19, pp. 453-464.

(11e584) Heyman, D. P., and Sobol, M. J., 1981. Stochastic Models in Opcrations Kesearch, Vol. 11, McGraw-Hill, N. Y.

Her89, Hernandez-Lerma, O., 1989. Adaptive Markov Control Processes, Springer-Verlag, N.

Ho'1'74] Hordijk, A., and Tijns, I., 1974. "Convergence Results and Ayproxinations for Optimal (s, S) Policies," Management, Sci., Vol. 20, pp.

1431-1438.

How60] Howard, R., 1960. Dynamic Programming and Markov Processes.

MIT Press, Cambridge, MA.

(Ig163a] Iglchart, D. I., 1963. "Optimality of (S,s) Policies in the Infinite Horizon Dynamic Inventory Problem," Management Sci., Vol. 9, pp. 259-

267.

(Ig163)] Iglehart, D. L., 1963. "Dynamic Programming and Stationary Analysis of liventory Problems," in Scarf, I., Gilford, D., and Shelly, N., (eds.), Multistage Inventory Models and Teeniques, Stanford University Press, Stanford, CA, 1963.

(JJS04) Jaakkola, T., Jordan, M. I., and Singh, S. P., 1994. "On the Convergence of Stochastic Iterative Dynamic Programming Algorithms,"

Neural Computation, Vol. 6, pp. 1185-1201.

(Jew63) Jewell, W., 1963. "Markov Renewal Programming Fand 11," Oyerations Research, Vol. 2, pp. 938-971.

[KaV87) Katchakis. M., and Veinott, A. F., 1987. "The Multi-Armed Bandit Problem: Decomposition and Computation, Math. of Operations Research, Vol. 12, pp. 2062-268.

(Kal83] Kallenberg, L. C. M., 1983. Linear Programming aud Finite Markov

Control Problems, Mathomatical Contro Report, Amsterdam.

(Kel81] Kelly, F. P., "Multi-Armed Bandits with Discount Factor Near One:

The Bernoulli Case," The Annals of Statisties, Vol. 1, po. 987-1001.

Kle68] Kleimman, D. L., 1968. "On an Iterative Technique for Riccati Equation Computatious." IEEE Trans. Automatic Control, Vol. AC-13,

DI). 114-115.

[KuV86] Kunar, P. R., and Varaiya, P. I., 1986. Stochastic Systems: Estimation, Identification, and Adaptive Control, Prentice-Hall, Englewood

Cliffs, N. J.

[Kum85] Kumar, P. R., 1985. "A Survey of Some Results in Stochastic Adaptive Control," SIAM . on Control and Optimization, Vol. 23. 1p. 329-380.

[Kus71] Kushner. I. J., 1971. Introduction to Stochastic Control. Holt. Rinchart and Winston, N. Y.

[Kus78] Kushner, H. J., 1978. Optimality Conditions for the Average Cost per Unit Time Problem with a Diflasion Model." SIAM 1. Control Opti-

mixation, Vol. 16, Pp. 330-316.

(Lasss) Lassore, J. B., 1988. Conditions for Existence of Average and Blackwell Optimal Stationary Policies in Denunerable Markov Decision

Processos," J. Math. Anal. Appl., Vol. 136, pp. 179-190.

(LiK&amp;4 Lim, W., and Kumar, P. R., 1984. "Optimal Control of a Queucing System with Two Heterogeneous Servers," IEEE Trans, Automatic Control,

Vol. AC-29. pD. 696-703.

[Lik71) Lippman, S. A., and Ross, S. M., 1971. "The Streetwalker's Dilemma: A Job-Shop Model," SIAM J. of Appl. Math., Vol. 20, pp. 336-342.

[LiS61] Liustornik, L., and Sobolev, V., 1961. Elements of Functional Anal- ysis, Vogar, N. Y.

(Lip75) Lippman, S. A., 1975. "Applying a New Device in the Optimization of Exponential Quening Systems," Operations Research, Vol. 23, py. 687710.

(LjS83) Ljung, Do, and Soderstrom, T., 1983. Theory and Practice of Recursive Identification, NT Press, Cambridge, MA.

[Lne69) Luenberger, D. G., 1969. Optimization by Vector Space Methods,

Wiler: N. Y.

Alc(66 MacQueen, J., 1966. "A Modified Dynamic Programming Method for Markovian Decision Problems," J. Math, Anal. Appl., Vol. 11, py. 38-

4.3.

MoW77) Morton, T. E., and Wecker, W., 1977. "Discounting, Ergodicity and Convergence for Markov Decision Processes," Management Sei.. Vol. 23, рр. 890-900.

(Mor71) Morton, T. E., 1971. "On the Asymptotic Convergence Rate of Cost. Differences for Markovian Decision Processes," Operations Research, Vol. 19, py. 244-218.

[N7W89] Nain, P., Tsoucas, P., and Walrand, J., 1989. "Interchango Arguments in Stochastic Scheduling," J. of Appl. Prob., Vol. 27, pp. 815-826.

(Ng/'86] Nguven, S., and Pallottino, S., 1986. "Hyperpaths and Shortest Hyperpaths," in Combinatorial Optimization by 13. Simeone (ed.), SpringerVerlag. N. Y. 0p. 258-271.

@do69! Odoni, A. K., 1969. "On Finding the Maximal Gain for Markov

Decision Processes," Operations Research, Vol. 17, pp. 857-860.

10/K70| Ortega, J. Al., and Kheinbold, W. C., 1970. Iterative Solution of

Nonlincar Equations in Several Variables, Academic Press, N. Y.

(Orn69) Ornstein, D., 1969. "On the Existence of Stationary Optimal Strate- gies," Proc. Amer. Math. Soc., Vol. 20, po. 563-569.

(PB'Т95) Polymenakos, L. C., Bertsekas, D. P., and Tsitsiklis, J. N.. 1995. "Elicient Algorithms for Continuous-Space Shortest Path Problems." Lab. for Info, and Decision Systems Report LIDS-P-2292, Massachusetts lustitute of Tochmology.

(PBW'79) Popyack, J. L. Brown, R. L.. and White. C. C.. HI. 1969. "Discrete Versions of an Algorithin due to Varaiya," 16EE Trans. Ant. Control.

Vol. 21, pp. 503-50.1.

(Palist] Pattipati, k. R., and Kleinman, D. L., 1981. "Priority Assignment Using Dynamic Programming for a Class of Queueing Systens." IBEE Trans. on Automatic Control, Vol. AC-26, py. 1095-1106.

(PaT87) Papadimitrion. C. H., and Tsitsiklis, J. N. 1987. "The Complexity of Markor Decision Processes," Math. Operations Research, Vol. 12, 1p. 441-150.

[Pla77) Platzman, L., 1977. "Improved Conditions for Convorgence in Undiscounted Markor Renewal Programming," Operations Kesearch, Vol. 25. pp. 529-533.

(Po169) Pollatschok, M., and Avi-ltzhak, B. 1969. "Algorithms for Stochastie Games with Geometrical Interpretation. Man. Sci., Vol. 15, pp. 399-

(PoT78) Porteus, E., and Totten, J., 1978. "Accelerated Computation of the Expected Discounted Return in a Markov Chain," Operations Research, Vol. 26. pp. 350-358.

(PoT92) Polychronopoulos. G. 11., ind Tsitsiklis. J. N., 1992. "Stochis tie Shortest Path Problems with Recourse." Lab. for Auto, and Decision Systems Report LADS-P-P-2183, Massachusetts Institute of Technology:

[Por71] Porteus, E., 1971. "Some Bounds for Discounted Sequential Deci- sion Processes," Man. Sci., Vol. 18, pp. 7-11.

(Por75) Portens, E., 1975. "Bounds and Transformations for Finite Markov

Decision Chains." Operations Rescarch, Vol. 23. pp. 761-781.

(Por81] Portens, E., 1981. "Improved Conditions for Convorgence in Undiscounted Markor Renewal Programming," Operations Research, Vol. 25. pp. 529-533.

(Ps193) Psaraftis. M. N.. and Tsitsiklis, d. N., 1993. "Dynamic Shortest Paths in Acyclie Networks with Markovian Arc Costs," Operations Kescarch. Vol. 11. pp. 91-101.

'uB78) luterman, M. I., and Bramelle, S. L., 1978. "The Aualytic Theory of Policy Aeration," in Dynamic Programming and Its Applications, M. L. Puterman (ed.), Academic Press. N. Y.

(°4578| Paterman, M. Do. and Shin, M. C. 197d. "Modified Polies Heration

iris -ted Nurkor Decision Probles, Maungenest Sri

(PuS82, Puterman, M. L., and Shin, M. C., 1982. "Action Elimination Procedures for Modifiod Policy Iteration Algorithus," Operations Research,

Vol. 30, pp. 301-318.

(Put78) Puterman, M. L. (ed.), 1978. Dynamic Programming and its Ap- plications, Academic Press, N. Y.

(Put.94 Puterman, M. L., 1994. "Markovian Decision Problems," J. Wiley,

N. Y.

[RVW82] Rosberg, Z., Varaiya, P. P., and Walrand, J. C., 1982. "Optimal

Control of Service in Tandem Queues," IEEE Trans. Automatic Control, Vol. AC-27, pp. 600-609.

(RaF91) Raghavan, T. E. S., and Filar. J. A., 1991. "Algoritlans for StochasLic Games A Survey," ZOR Methods and Models of Operations Re- search, Vol. 35, pp. 437-472.

|Ki$92| Rite, R. K., and Sonnot, L. I.. 1992. "Optimal Stationary Policies in Cieneral State Markov Decision Chains with Finite Action Set," Math.

Operations Research, Vol. 17, pp. 901-909.

[Ro:70) Rockafellar, K. T., 1970. Convex Analysis, Princeton University Press, Princeton, N. J.

Ros70) Ross, S. M., 1970. Applied Probability Models with Optimization

Applications, Holdon-Day, San Francisco, CA.

(Ros83a] Ross, S. M., 1983. Introduction to Stochastic Dynamic Programming, Academic Press, N. Y.

(Ros836) Ross, S. M., 1983. Stochastic Processes. Wiley: N. Y.

scarch, Vol. 137. pp. 474-477.

consin.

(Kus95) Rust, J., 1995. "Numerical Dynamic Programming in Economics,"

in Handbook of Computational Economies, H. Amman, D. Kendrick. and J. Rust (eds.).

(SP'1585) Schweitzer. D. D., Poterman, M. Lo, and Kindle, K. W., 1985.

"Iterativo Aggregation-Disaggregation Procedures for Solving Discounted Somi-Markovian Reward Processes," Operations Research, Vol. 33, pp. 580605.

ScF77) Schweitzer, P. J., and Federgruen, A., 1977. "The Asymptotic Behavior of Value Iteration in Markov Decision Problems," Math. Operations

Research, Vol. 2, pp. 360-381.

(Sc 178) Schweitzer, P. J., and Federgruen, A., 1978. "The Punctional liquations of Undiscounted Markov Renewal Programming." Math. Operations

Rescarch, Vol. 3, pp. 308-321.

(Sc585) Schweitzer, P. J.. and Seidman, A., 1985. "Generalized Polynomial Approximations in Markovian Decision Problems," J. Math. Anal.

and Appl., Vol. 110, pp. 568-582.

(Sch68) Schweitzer. P. J., 1968. "Perorbation Theory and Finite Markos Chains." J. Appl. Prob., Vol. 5, py. 101-113.

(Sch71] Schweitzer, P. J., 1971. "Iterativo Solution of the Rmctional Banations of Undiscounted Markov Renewal Programming," J. Math. Anal.

Appl., Vol. 34, pp. 195-501.

(Sch72) Schweitzer, P. J., 1972. "Data Transformations for Markov Renewal Programming," talk at National OKSA Mooting, Atlantic Cily: N. J.

(Sch75) Schal, M., 1975. Conditions for Optimality in Dynamic Program- ming and for the Limit of a-Stage Optimal Policies to be Optimal," ½. Wahuscheinlichkeitstheorie und Verw. Gebiete, Vol. 32, pp. 179-196.

(Schol] Schweitzer, 1. J., 1981. "Bottleneck Detormination in a Network of Quoues," Graduate School of Management, Working Paper No. 8107. University of Rochester, Rochester, N. Y.

(Sch93) Schwartz. A., 1993. "A Reinforcement. Learning Method for Maximizing Undiscounted Rewards," Proc. of the Tenth, Machine Learning Con- ference.

Operations Research Lott., Vol. 5. po. 17-23.

Sen89al Senott, L. J., 1989. "Average Cost Optimal Stationary Policies in Infinite State Markov Decision Processes with Unbounded Costs," Opcrations Research. Vol. 37, pp. 626-633.

[Sen89b) Sonnott, L. I., 1989. "Average Cost. Semi-Markov Decision Procosses and the Control of Queucing Systems," Prob. Eng. Info. Sci.. Vol. 3.

pD. 247-272.

(Sen91) Sonott, D. 1., 1091. "Value Heration in Comtable State Averaro Cost Markos Decision Processes with Unbounded Cost." Annals of Oper- ations Research, Vol. 28, pp. 261-272.

(Sen93a) Senott, L. I., 1993. "The Average Cost Optimality Equation and Critical Number Policies." Prob. Eng. Info. Sci., Vol. 7. pp. 47-67.

(Sen93b| Sonott, L. I., 1993. "Constrained Avorage Cost Markov Decision Chains," Prob. Eng. Info. Sci., Vol. 7, pp. 69-83.

Ser793 Serfozo, R., 1979. An Equivalenco Botween Discrete and Continnous 'Timo Markor Docision Processes," Operations Research, Vol. 27. py.

616-620.

(Sha5:3) Shapley, L. S., 1953. "Stochastic Games," Proc. Nat. Acad. Sei. U.S.A., Vol. 39.

(Sin94) Singh, S. P., 1994. "Reinforcomont Learning Algorithms for A veragePayoll Markovian Decision Procosses." Proc. of 12th National Conferenco on Artilicial Intelligenee, pp. 202-207.

(Sob82, Sobol, M. J., 1982. "Tho Optimality of Pall-Service Policies." Op- orations Research, Vol. 30, op. 636-619.

(St.271) Stidham, S., and Prablu, N. U., 1974. "Optimal Control of Quene- ing Systems," in Mathematical Methods in Quencing Theory (Lecture Notes in Economics and Math. Syst., Vol. 98), A. B. Clarke (Ed.), SpringerVorlag, N. Y., Dp. 263-291.

[Sti85) Stidham, S. S., 1985. "Optimal Control of Admission to a Quencing

System," DEBE Trans. Automatic Control, Vol. AC-30, рр. 705-713.

[Str66) Strauch, R., 1966. "Negative Dynamic Programming." Aun. Math.

Statist., Vol. 37, pp. 871-890.

Control, Vol. 30, pp. 1086-1091.

(Su1.88) Sutton, K. S., 1988. "Learning to Predict by the Methods of Tem- poral Dillorences," Machine Learning, Vol. 3, pp. 9-41.

(TSC92) Tousley, D., Sparaggis, P. D., and Cassandras, C. G., 1992. "Optimal Routing and Butter Allocation for a Class of Finite Capacity Queucing Systoms," IEEE 'Trans. on Automatic Control, Vol. 37, py. 1416-1451.

(Tes92] Tosauro, G.. 1992. "Practical Issues in Temporal Difference Learning." Machine Learning. Vol. 8, pp. 257-277.

[T'sV9 1) Tsitsiklis, J. N., and Van Roy, B., 1991. "Featuro-Based Mothods lems Koport. LIDS-P-2277, Massachusetts Institute of Technology, Machine for Large-Scade Dynamic Programming," Lab. for Info. and Decision SysLearning, to appear.

[T'se90| Tsong, 1., 1990. "Solving A-Horizon, Stationary Markor Decision

Problems in Time Proportional to log(1)." Operations Research Letters, Vol. 9. pp. 287-297.

(Tsi86] Tsitsiklis, J. N., 1986. "A Lemma on the Multiarned Bandit Problem," DEBE Trans. Automatic Control. Vol. AC-31. po. 576-577.

(Tri89 Isitsiklis, J. N.. 1989. "A Comparison of Jacobi nud Ganss-Scidel Parallel Iterations." Appliod Math. Lett.. Vol. 2. pp. 167-170.

(Tsi93a Tritsiklis, J. N., 199:3. "Ellicient Algorithms for Cloball Optimal Trajectories," Lab. for Info, and Decision Systems Report LIDS-P-2210. Massachusetts Institute of Technology: HERE Trans. on Automatic C'ontrol. to appear.

(Tsi931) Tsitsiklis, J. N.. 1993. "A Short Proof of the Gittins Index Thieorem." Lab. for Info. and Decision Systems Report LIDS-P-2171, Massachusetts Institute of Tochnology: also Annals of Applied Probability, Vol.

1. 1994. pp. 19.1-199.

(Tsi90 Tritsiklis, J. N.. 199.1. "Asynchronons Stochastic Approximation and ()-Learing." Machino Learning, Vol. 16, pp. 185-202.

[75091] Tsoukas. P., 1991. "The Region of Achievable Performance in a Model of Klimor." Research Report., 1.B.NI.

[VWB85) Varaiya, P. P., Walrand. J. C., and Buynkkoe. C.. 1985. "Extensions of the Multiarmed Bandit Problem: The Discounted Case," LEDE

(Var78) Varaiva, P. P., 1978. "Optimal and Suboptimal Stationary Controls of Markov Chains," KEEB Trans. Automatic Control. Vol. AC-23. pD. 388-

391.

[VoP81) Verd'n, S., and Poor, 11. V., 1984. "Backward, Forward. and Backward-Forward Dynmic Programming Models under Commutativits Conditions," Proc. 1984 DEBE Decision and Control Conference, Las Vegas.

NE, DD. 1081-1086.

(VeP87) Verdo, S.. and Poor. I. V., 1987. "Abstract, Dynamic Programming Models under Commutativity Conditions," SIAM J. on Control and

Optimization. Vol. 25, pp. 990-1006.

[Voi66] Veinott, A. F., Jr., 1966. "On Finding Optimal Policies in Discrete Dyunmic Programning with no Discounting." Ann. Math. Statist., Vol. 37.

pp. 1281-129.1.

(Ne69) Voinott, A. F.. Jr., 1969. "Discrete Dynamic Programming with Sensitivo Discount. Optimality Criteria," Ann. Math. Statist.. Vol. 10. pD.

1635-1660.

(ViEds) Viniotis, A.. and Ephremides. A.. 1988. "Extension of the Optimality of the Threshold Polire in Hoterograrons Multiser Ourind

Sustonis." HERE Trans, on Automatic Control, Vol. 38. Do. 101-109.

(Wats9] Watkins. C. J. C. H.. "Learning from Delaved Rowards." Ph.D.

Thesis. Cambridge Unir. England.

(Nch92) Weber, R., 1991. "On the Gittins Index for Multiarmed Bandits," preprint; Annals of Applied Probability, Vol. 3. 1993.

(Whk80) White, C. Co, and Kim, K., 1980. "Solution Procedures for Partially Observed Markov Decision Processes," J. Largo Scale Systems, Vol. 1, Рр. 129-1·10.

W163) White, D. J., 1963. "Dynamic Programming, Markov Chains, and the Method of Successive Approximations," . Math. Anal. and Appl., Vol. 6. рр. 373-376.

(Whi78) Whitt, W., 1978. "Approximations of Dynamic Programs I," Math. Operations Rescarch, Vol. 3, pp. 231-213.

(|179) Whitt, W., 1979. "Approximations of Dynamic Programs II," Math. Operations Research, Vol. 4, pD. 179-185.

(Whi80a) White. D. J., 1980. "Finite State Approximations for Denumerable State Iatinito Horizon Discounted Markov Decision Processes: The Method of Successive Approximations," in Keront Developments in Markov Decision Processes, Marley, R., 'Thomas, L. C., and White, D. J. (ods.), Academic Press, N. Y., Do. 57-72.

(Whisol) Whittle, 1., 1980. "Multi-Armed Bandits and the Gittins Index." J. Roy. Statist. Soc. Sor. 13, Vol. 42. pp. 143-149.

Whiod Wlande, P.. Moj. "Aimede gutsing Bunder The Aonate Proiability: Viol. 1), p. 281-2012.

(Whi82) Whittle, P., 1982. Optimization Over 'Time, Wiley, N. Y., Vol. 1. 1982, Vol. 2, 1983.

## INDEX

## A

Admissible policy, 3

Advantage updating, 122, 132

Approximation in policy space, 117 Asset, selling, 157, 275

Aggregation, 44, 104, 219

Asyachronous algorithms, 130, 74, 120

Average cost problem, 181, 219, 206

## B

Basis functions, 51, 65. 103 Bellman's cquation, 8. 11, 83, 108, 137, 186. 191. 196, 225, 247, 268 Blackwoll optimal policy: 193. 2333 Bold -tratoes: 162

Chess. 102. 117 Column reduction. 67 Contraction mappings. 52, 65, 86, 128 Consistontly improving policies, 90. 122. 127 Controllability, 151, 228 Cost approsimation, 51, 101. 225

## D

Data transformations, 72. 263, 271 Ditlerential cost, 186, 192 Dijkstra's algorithm. 90, 122 Discounted cost, 9. 186, 213, 262 Discretization. 65 Distributed computation, 74, 120 Duality; 65, 222

## E

c-optimal policy, 172 Error bounds, 19, 69, 209, 213, 231, 239

## F

Feature-based aggregation, 101 Feature extraction, 103 Feature vectors, 1033

## G

Gambling, 160, 17:3, 180 Gauss-Seidel method, 28. 88, 208

## I

Improper policy: so Index finction, 56 Index of a project. 55 Index rule, 55, 005 Inventore control. 153. 179 Mccarible Marker cham. 211

## J

Jacobi method, fis

## L

LLL stratego, 00 Label comecting method, 90 Liner programming, 19, 150, 221 Lincar quadratic problems. 150. 176178. 225. 235

## M

Measurability issues. 01, 172 Minimax problems. 72 Monotone convergence theorem, 136 Monte-Carlo simulation, 96. 112. 120, 131. 223 Multiarmed bandit problem. 51. 256 Multiple-rank corrections, 18. 61

## N

Negative DP model, 134 Nowady mate primming. 122

Nonstationary problems, 167

Observability, 151, 228

One-stop-lookahead rule, 157, 159.

160

Optimistic policy iteration, 116, 122

## P

Parallel computation, 61, 71, 120

Periodie problems, 167, 171, 177,

179

Polics, 3

Policy existence, 160, 172, 182, 220

Policy evaluation, 36, 214

Policy improvement, 36, 211

Policy iteration, 35, 71, 73, 91, 149,

186, 213, 223

Policy iteration, approximate, 41,

91, 112, 115

Policy iteration, moditiod, 39, 91

Polynomial approximations, 102

Positive Dl' model, 131

Priority assignment, 251

Proper policy, 80

## Q

Q-factor, 99, 132

(-learning, 16, 99, 122, 224, 230,

2:39

Quadratic cost, 150, 176-178, 228,

2:35

(Quonding control, 250, 265

## R.

Randomised policy: 222

Rank-one correction, 30. 68

Reachability, 181, 182

Reinforcement learning, 122

Relative cost, 186. 192

Replacement. problems, 14, 200, 276

Riccati equation, 151, 228

Robbins-Mono method. 98

Routing, 257

SLE strategy, 90

Scheduling problems, 51

Somi-Markov problems, 261

Sequential probability ratio, 158

Sequential hypothesis testing, 158

Sequential space decomposition, 125

Shortest, path problem, 78, 90, 126

Simulation-based methods, 16, 78,

91, 222

Stochastic approximation method, 98

Stationary policy, 3

1-1:3, 100, 172, 182, 227

Stationary policy, existence, 13, 83,

Stochastic shortest, paths, 78, 185,

236-239

Stopping problems, 87, 155

Successive approximation, 19

Tomporal differences. 16, 97, 115,

122, 223

Totris, 105, 111

Threshold policies, 73

## U

Unbounded costs per stago, 134

Uncontrollable state components,

105, 125

Uniformization, 212, 271

Undisconuted problems, 131, 249

Unichain policy, 196

Vilue iteration. 19, 88, 144, 186,

202. 211. 221, 238

Value iteration, approximate, 33

Vine iteration, relative, 204, 211, 220. 232

Vane iteration, tormination, 23, 89

Weighted sup norm, 86, 128