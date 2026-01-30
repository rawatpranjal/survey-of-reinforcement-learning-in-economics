## Approximate Dynamic Programming for Operations Research:

Solving the curses of dimensionality

Warren B. Powell

May 2, 2005

(c) Warren B. Powell, 2005 All rights reserved.

Department of Operations Research and Financial Engineering Princeton University, Princeton, NJ 08544

## Contents

| 1   | The challenges of dynamic programming   | The challenges of dynamic programming                           |   1 |
|-----|-----------------------------------------|-----------------------------------------------------------------|-----|
|     | 1.1                                     | A dynamic programming example: a shortest path problem          |   2 |
|     | 1.2                                     | The three curses of dimensionality . . . . . . . . . . . . . .  |   3 |
|     | 1.3                                     | Some real applications . . . . . . . . . . . . . . . . . . . .  |   5 |
|     | 1.4                                     | Problem classes in asset management . . . . . . . . . . . .     |   8 |
|     | 1.5                                     | What is new in this book? . . . . . . . . . . . . . . . . . .   |  10 |
|     | 1.6                                     | The many dialects of dynamic programming . . . . . . . .        |  12 |
|     | 1.7                                     | Bibliographic notes . . . . . . . . . . . . . . . . . . . . . . |  14 |
| 2   | Some illustrative applications          | Some illustrative applications                                  |  15 |
|     | 2.1                                     | Deterministic problems . . . . . . . . . . . . . . . . . . . .  |  15 |
|     |                                         | 2.1.1 The budgeting problem . . . . . . . . . . . . . . . .     |  15 |
|     |                                         | 2.1.2 The shortest path problem . . . . . . . . . . . . . .     |  17 |
|     | 2.2                                     | Stochastic problems . . . . . . . . . . . . . . . . . . . . . . |  19 |
|     |                                         | 2.2.1 The gambling problem . . . . . . . . . . . . . . . .      |  19 |
|     |                                         | 2.2.2 The batch replenishment problem . . . . . . . . . .       |  21 |
|     |                                         | 2.2.3 The secretary problem . . . . . . . . . . . . . . . .     |  22 |
|     |                                         | 2.2.4 Optimal stopping . . . . . . . . . . . . . . . . . . .    |  28 |
|     | 2.3                                     | Bibliographic notes . . . . . . . . . . . . . . . . . . . . . . |  29 |
| 3   | Modeling dynamic programs               | Modeling dynamic programs                                       |  32 |
|     | 3.1 Notational style .                  | . . . . . . . . . . . . . . . . . . . . . . .                   |  33 |

## CONTENTS

| 3.2   | Modeling time . . . . . . . . . . . . . . . . . . .   | Modeling time . . . . . . . . . . . . . . . . . . .   |   34 |
|-------|-------------------------------------------------------|-------------------------------------------------------|------|
| 3.3   | Modeling assets . . . . . . . .                       | . . . . . . . . .                                     |   38 |
| 3.4   | Illustration: the nomadic trucker . . .               | . . . .                                               |   41 |
|       | 3.4.1                                                 | A basic model . . . . . . . . . . . . . .             |   41 |
|       | 3.4.2                                                 | A more realistic model . . . . . . . . .              |   42 |
|       | 3.4.3                                                 | The state of the system . . . . . . . .               |   43 |
| 3.5   | The exogenous information process                     | . . . . . .                                           |   44 |
|       | 3.5.1                                                 | Basic notation for information processes              |   44 |
|       | 3.5.2                                                 | Models of information processes . . . .               |   45 |
| 3.6   | The states of our system . . . . . . .                | . . . . .                                             |   48 |
|       | 3.6.1                                                 | The three states of our system . . . . .              |   48 |
|       | 3.6.2                                                 | Pre- and post-decision state variables .              |   51 |
|       | 3.6.3                                                 | Partially observable states . . . . . . .             |   53 |
| 3.7   | Modeling decisions . .                                | . . . . . . . . . . . . .                             |   54 |
|       | 3.7.1                                                 | Decisions, actions, and controls . . . .              |   54 |
|       | 3.7.2                                                 | The nomadic trucker revisited . . . . .               |   56 |
|       | 3.7.3                                                 | Decision epochs . . . . . . . . . . . . .             |   56 |
|       | 3.7.4                                                 | Policies . . . . . . . . . . . . . . . . .            |   57 |
|       | 3.7.5                                                 | Randomized policies . . . . . . . . . .               |   58 |
| 3.8   | Information processes, revisited . . . .              | . . . .                                               |   59 |
|       | 3.8.1                                                 | Combining states and decisions . . . .                |   59 |
|       | 3.8.2                                                 | Supervisory processes . . . . . . . . . .             |   60 |
| 3.9   | Modeling system dynamics .                            | . . . . . . . . . .                                   |   60 |
|       | 3.9.1                                                 | A general model . . . . . . . . . . . . .             |   60 |
|       | 3.9.2                                                 | System dynamics for simple assets . . .               |   63 |
|       | 3.9.3                                                 | System dynamics for complex assets . .                |   64 |
| 3.10  | The contribution function                             | . . . . . . . . . . .                                 |   68 |

## CONTENTS

| 3.12           |                                                                                                  | Models for a single, discrete asset                                                              | 71    |
|----------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-------|
| 4 Introduction |                                                                                                  | to Markov decision                                                                               |       |
| 4.1            | The                                                                                              | optimality equations . . . .                                                                     | 83 84 |
| 3.14           | Bibliographic                                                                                    | notes . . . . . . .                                                                              | 77    |
| 4.2 4.3        |                                                                                                  |                                                                                                  | 87    |
|                | The optimality equations using the post-decision state variable .                                | The optimality equations using the post-decision state variable .                                | 88    |
|                | Finite horizon problems . . . . . . . . . . . . . . . . . . . . . .                              | Finite horizon problems . . . . . . . . . . . . . . . . . . . . . .                              |       |
|                | 4.3.1 The optimality equations . . . . . . . . . . . . . . . . . .                               | 4.3.1 The optimality equations . . . . . . . . . . . . . . . . . .                               | 88    |
|                | 4.3.2 Backward dynamic programming . . . . . . . . . . . .                                       | 4.3.2 Backward dynamic programming . . . . . . . . . . . .                                       | 90    |
| 4.4            | Infinite horizon problems . . . . . . . . . . . . . . . . .                                      | Infinite horizon problems . . . . . . . . . . . . . . . . .                                      | 91    |
|                | 4.4.1                                                                                            | Value iteration . . . . .                                                                        | 93    |
|                | 4.4.2 Policy iteration . . . . . . . . . . . .                                                   | 4.4.2 Policy iteration . . . . . . . . . . . .                                                   | 98    |
|                | 4.4.3 Hybrid value-policy iteration . . . . . . . . . The linear programming formulation . . . . | 4.4.3 Hybrid value-policy iteration . . . . . . . . . The linear programming formulation . . . . | 99    |
|                |                                                                                                  |                                                                                                  | 100   |
|                | 4.4.4 . . Why . . . . . . .                                                                      | 4.4.4 . . Why . . . . . . .                                                                      | 101   |
| 4.5            | does it work?** . . . . . . . . . . . . 4.5.1 The optimality equations . . . . . . .             | does it work?** . . . . . . . . . . . . 4.5.1 The optimality equations . . . . . . .             | 101   |
|                | 4.5.2 Proofs for value iteration . . . . . . . . . .                                             | 4.5.2 Proofs for value iteration . . . . . . . . . .                                             | 106   |
|                | . . 4.5.3 Optimality of Markovian policies . . . . . . .                                         | . . 4.5.3 Optimality of Markovian policies . . . . . . .                                         | 112   |
|                | 4.5.4 Optimality of deterministic policies . . . .                                               | 4.5.4 Optimality of deterministic policies . . . .                                               |       |
|                | notes . . . . . . . . . . . . . . . . .                                                          | notes . . . . . . . . . . . . . . . . .                                                          | 113   |
| 4.6            | Bibliographic                                                                                    | Bibliographic                                                                                    | 115   |
| 5              | Introduction to approximate dynamic programming                                                  | Introduction to approximate dynamic programming                                                  | 121   |
| 5.1            | The three curses of dimensionality (revisited) . . . . . . .                                     | The three curses of dimensionality (revisited) . . . . . . .                                     | 122   |
| 5.2            | Monte Carlo sampling and forward dynamic programming . . . . . . . . . . . .                     | Monte Carlo sampling and forward dynamic programming . . . . . . . . . . . .                     | 123   |
| 5.3            | Using the post-decision optimality equations . . . . . .                                         | Using the post-decision optimality equations . . . . . .                                         | 126   |
| 5.4            | Low-dimensional representations of value functions .                                             | Low-dimensional representations of value functions .                                             | 127   |

6

|                                  | 5.4.1                                                     | Aggregation . . . . . . . . . . . . . . . . . .           |   128 |
|----------------------------------|-----------------------------------------------------------|-----------------------------------------------------------|-------|
|                                  | 5.4.2                                                     | Continuous value function approximations .                |   129 |
|                                  | 5.4.3                                                     | Algorithmic issues . . . . . . . . . . . . . .            |   130 |
| 5.5                              | Complex resource allocation problems . . . . .            | . . .                                                     |   131 |
| 5.6                              | Experimental issues .                                     | . . . . . . . . . . . . . . . . .                         |   133 |
|                                  | 5.6.1                                                     | The initialization problem . . . . . . . . . .            |   135 |
|                                  | 5.6.2                                                     | Sampling strategies . . . . . . . . . . . . . .           |   135 |
|                                  | 5.6.3                                                     | Exploration vs. exploitation . . . . . . . . .            |   135 |
|                                  | 5.6.4                                                     | Evaluating policies . . . . . . . . . . . . . .           |   137 |
| 5.7                              | Dynamic programming with missing or incomplete models . . | Dynamic programming with missing or incomplete models . . |   139 |
| 5.8                              | Relationship to reinforcement learning . . . . . . .      | Relationship to reinforcement learning . . . . . . .      |   140 |
| 5.9                              | But does it work? . .                                     | . . . . . . . . . . . . . . . . .                         |   141 |
| 5.10                             | Bibliographic notes . . . . .                             | . . . . . . . . . . . . .                                 |   142 |
| Stochastic approximation methods | Stochastic approximation methods                          | Stochastic approximation methods                          |   146 |
| 6.1                              | A stochastic gradient algorithm . . . .                   | . . . . . . .                                             |   148 |
| 6.2                              | Sampling random variables . . .                           | . . . . . . . . . . .                                     |   150 |
| 6.3                              | Some stepsize recipes . . .                               | . . . . . . . . . . . . . .                               |   152 |
|                                  | 6.3.1                                                     | Properties for convergence . . . . . . . . . .            |   152 |
|                                  | 6.3.2                                                     | Deterministic stepsizes . . . . . . . . . . . .           |   154 |
|                                  | 6.3.3                                                     | Stochastic stepsizes . . . . . . . . . . . . . .          |   159 |
|                                  | 6.3.4                                                     | A note on counting visits . . . . . . . . . . .           |   165 |
| 6.4                              | Computing bias and variance                               | . . . . . . . . . . . .                                   |   165 |
| 6.5                              | Optimal stepsizes .                                       | . . . . . . . . . . . . . . . . . .                       |   167 |
|                                  | 6.5.1                                                     | Optimal stepsizes for stationary data . . . .             |   168 |
|                                  | 6.5.2                                                     | Optimal stepsizes for nonstationary data - I              |   171 |
|                                  | 6.5.3                                                     | Optimal stepsizes for nonstationary data - II             |   172 |
| 6.6                              | Some experimental comparisons of stepsize formulas .      | Some experimental comparisons of stepsize formulas .      |   174 |
| 6.7                              | Convergence .                                             | . . . . . . . . . . . . . . . . . . . . .                 |   179 |

7

| 6.8   | Why does it work?** . . . . . .                                    | . . . . . . . . . . . . . . .                                      |   180 |
|-------|--------------------------------------------------------------------|--------------------------------------------------------------------|-------|
|       | 6.8.1                                                              | Some probabilistic preliminaries . . . . . . . . . . .             |   181 |
|       | 6.8.2                                                              | An older proof . . . . . . . . . . . . . . . . . . . .             |   182 |
|       | 6.8.3                                                              | A more modern proof . . . . . . . . . . . . . . . . .              |   186 |
|       | 6.8.4                                                              | Proof of theorem 6.5.1 . . . . . . . . . . . . . . . .             |   190 |
| 6.9   | Bibliographic notes                                                | . . . . . . . . . . . . . . . . . . . . . .                        |   193 |
|       | 6.9.1                                                              | Stochastic approximation literature . . . . . . . . .              |   193 |
|       | 6.9.2                                                              | Stepsizes . . . . . . . . . . . . . . . . . . . . . . . .          |   193 |
|       | Discrete, finite horizon problems                                  | Discrete, finite horizon problems                                  |   198 |
| 7.1   | Applications . . . . . . . . . . . . . . . . . . . . . . . . . . . | Applications . . . . . . . . . . . . . . . . . . . . . . . . . . . |   199 |
| 7.2   | Sample models . . . . . . . . . . . . . . . . . . .                | . . . . .                                                          |   200 |
|       | 7.2.1                                                              | The shortest path problem . . . . . . . . . . . . . .              |   200 |
|       | 7.2.2                                                              | Getting through college . . . . . . . . . . . . . . . .            |   204 |
|       | 7.2.3                                                              | The taxi problem . . . . . . . . . . . . . . . . . . .             |   206 |
|       | 7.2.4                                                              | Selling an asset . . . . . . . . . . . . . . . . . . . .           |   207 |
| 7.3   | Strategies for finite horizon problems . . . . . . . . . . .       | .                                                                  |   208 |
|       | 7.3.1                                                              | Value iteration using a post-decision state variable .             |   208 |
|       | 7.3.2                                                              | Value iteration using a pre-decision state variable .              |   210 |
|       | 7.3.3                                                              | Q -learning . . . . . . . . . . . . . . . . . . . . . . .          |   211 |
| 7.4   | Temporal difference learning . . . .                               | . . . . . . . . . . . . .                                          |   216 |
|       | 7.4.1                                                              | The basic idea . . . . . . . . . . . . . . . . . . . . .           |   216 |
|       | 7.4.2                                                              | Variations . . . . . . . . . . . . . . . . . . . . . . .           |   218 |
| 7.5   | Monte Carlo value and policy iteration                             | . . . . . . . . . . .                                              |   218 |
| 7.6   | Policy iteration . .                                               | . . . . . . . . . . . . . . . . . . . . . .                        |   220 |
| 7.7   | State sampling strategies .                                        | . . . . . . . . . . . . . . . . . .                                |   221 |
|       | 7.7.1                                                              | Sampling all states . . . . . . . . . . . . . . . . . .            |   221 |
|       | 7.7.2                                                              | Tree search . . . . . . . . . . . . . . . . . . . . . .            |   222 |
|       | 7.7.3                                                              | Rollout heuristics . . . . . . . . . . . . . . . . . . .           |   224 |

| 7.8                           | A taxonomy of approximate dynamic programming strategies .              | A taxonomy of approximate dynamic programming strategies .              |   225 |
|-------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|-------|
| 7.9                           | But does it work? .                                                     | . . . . . . . . . . . . . . . . . .                                     |   227 |
|                               | 7.9.1                                                                   | Convergence of temporal difference learning                             |   227 |
|                               | 7.9.2                                                                   | Convergence of Q -learning . . . . . . . . . .                          |   227 |
| 7.10                          | Why does it work** .                                                    | . . . . . . . . . . . . . . . . .                                       |   227 |
| 7.11                          | Bibliographic notes .                                                   | . . . . . . . . . . . . . . . . .                                       |   227 |
| Infinite horizon problems     | Infinite horizon problems                                               | Infinite horizon problems                                               |   230 |
| 8.1                           | Approximate dynamic programming for infinite horizon problems .         | Approximate dynamic programming for infinite horizon problems .         |   231 |
| 8.2                           | Algorithmic strategies for discrete value functions . . . . . . . . . . | Algorithmic strategies for discrete value functions . . . . . . . . . . |   231 |
| 8.3                           | Value iteration . . . . . . . . . . . . . . . . . . . . . . . .         | Value iteration . . . . . . . . . . . . . . . . . . . . . . . .         |   232 |
| 8.4                           | Approximate policy iteration . . . . . . . . . . . . . . . . . . .      | Approximate policy iteration . . . . . . . . . . . . . . . . . . .      |   233 |
| 8.5                           | TD learning with discrete value functions . . . . . . . . . . . . .     | TD learning with discrete value functions . . . . . . . . . . . . .     |   235 |
| 8.6                           | Q-learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | Q-learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  |   237 |
| 8.7                           | Why does it work?** . . . . . . . . . . . . . . . .                     | .                                                                       |   237 |
| 8.8                           | Bibliographic notes . . . . . . . .                                     | . . . . . . . . . .                                                     |   238 |
| Value function approximations | Value function approximations                                           | Value function approximations                                           |   240 |
| 9.1                           | Simple aggregation . . . . . .                                          | . . . . . . . . . . . .                                                 |   241 |
| 9.2                           | The case of biased estimates . . . . . . . .                            | . . . . .                                                               |   245 |
| 9.3                           | Multiple levels of aggregation . . .                                    | . . . . . . . . .                                                       |   249 |
|                               | 9.3.1                                                                   | Combining multiple statistics . . . . . . . .                           |   250 |
|                               | 9.3.2                                                                   | The problem of correlated statistics . . . . .                          |   252 |
|                               | 9.3.3                                                                   | A special case: two levels of aggregation . .                           |   255 |
|                               | 9.3.4                                                                   | Experimenting with hierarchical aggregation                             |   256 |
| 9.4                           | General regression models                                               | . . . . . . . . . . . . . .                                             |   256 |
|                               | 9.4.1                                                                   | Pricing an American option . . . . . . . . .                            |   258 |
|                               | 9.4.2                                                                   | Playing 'lose tic-tac-toe' . . . . . . . . . . .                        |   261 |
| 9.5                           | Recursive methods for regression models                                 | . . . . . .                                                             |   262 |

|                                                          | 9.5.1                                                       | Parameter estimation using a stochastic gradient algorithm      |   263 |
|----------------------------------------------------------|-------------------------------------------------------------|-----------------------------------------------------------------|-------|
|                                                          | 9.5.2                                                       | Recursive formulas for statistical estimation . . . . . . . .   |   263 |
|                                                          | 9.5.3                                                       | Recursive time-series estimation . . . . . . . . . . . . . . .  |   266 |
|                                                          | 9.5.4                                                       | Estimation using multiple observations . . . . . . . . . . .    |   267 |
| 9.6                                                      | Why does it work?*                                          | . . . . . . . . . . . . . . . . . . . . . . . . . .             |   268 |
|                                                          | 9.6.1                                                       | Proof of Proposition 1 . . . . . . . . . . . . . . . . . . . .  |   268 |
|                                                          | 9.6.2                                                       | Proof of Proposition 2 . . . . . . . . . . . . . . . . . . . .  |   269 |
|                                                          | 9.6.3                                                       | Derivation of the recursive estimation equations . . . . . .    |   270 |
|                                                          | 9.6.4                                                       | The Sherman-Morrison updating formula . . . . . . . . . .       |   272 |
| 9.7                                                      | Bibliographic notes .                                       | . . . . . . . . . . . . . . . . . . . . . . . . .               |   273 |
| 10 The exploration vs. exploitation problem              | 10 The exploration vs. exploitation problem                 | 10 The exploration vs. exploitation problem                     |   275 |
| 10.1                                                     | A learning exercise: the nomadic trucker .                  | . . . . . . . . . . . . .                                       |   275 |
| 10.2                                                     | Learning strategies . . . . . .                             | . . . . . . . . . . . . . . . . . . . .                         |   277 |
|                                                          | 10.2.1                                                      | Pure exploration . . . . . . . . . . . . . . . . . . . . . .    |   279 |
|                                                          | 10.2.2                                                      | Pure exploitation . . . . . . . . . . . . . . . . . . . . . .   |   279 |
|                                                          | 10.2.3                                                      | Mixed exploration and exploitation . . . . . . . . . .          |   280 |
|                                                          | 10.2.4                                                      | Boltzman exploration . . . . . . . . . . . . . . . . . . .      |   280 |
|                                                          | 10.2.5                                                      | Remarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   281 |
| 10.3                                                     | A simple information acquisition problem . .                | . . . . . . . . . . . .                                         |   282 |
| 10.4                                                     | Gittins indices and the information acquisition problem . . | . . . .                                                         |   284 |
|                                                          | 10.4.1                                                      | Foundations . . . . . . . . . . . . . . . . . . . . . . . . . . |   284 |
|                                                          | 10.4.2                                                      | Basic theory of Gittins indices . . . . . . . . . . . . . . . . |   286 |
|                                                          | 10.4.3                                                      | Gittins indices for normally distributed rewards . . . . . .    |   288 |
|                                                          | 10.4.4                                                      | Gittins exploration . . . . . . . . . . . . . . . . . . . . . . |   290 |
| 10.5                                                     | Why does it work?** .                                       | . . . . . . . . . . . . . . . . . . . . . . . .                 |   291 |
| 10.6                                                     | Bibliographic notes .                                       | . . . . . . . . . . . . . . . . . . . . . . . . .               |   291 |
| 11 Value function approximations for resource allocation | 11 Value function approximations for resource allocation    | 11 Value function approximations for resource allocation        |   296 |

## CONTENTS

| 11.1                             | Value functions versus gradients .                           | . . . . . . . . . . . . . . . . . . . . . . . .                             | 297   |
|----------------------------------|--------------------------------------------------------------|-----------------------------------------------------------------------------|-------|
| 11.2                             | Linear approximations . . . . . .                            | . . . . . . . . . . . . . . . . . . . . . . . .                             | 298   |
| 11.3                             | Monotone function approximations                             | . . . . . . . . . . . . . . . . . . . . . . .                               | 300   |
| 11.4                             | The SHAPE algorithm for continuously differentiable problems | . . . . . . . .                                                             | 302   |
| 11.5                             | Regression methods . .                                       | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                 | 306   |
| 11.6                             | Why does it work?**                                          | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .               | 309   |
|                                  | 11.6.1                                                       | The projection operation . . . . . . . . . . . . . . . . . . . . . . . . .  | 309   |
|                                  | 11.6.2 Proof of convergence of the                           | learning version of the SPAR algorithm .                                    | 311   |
| 11.7                             | Bibliographic notes . . . . . . . .                          | . . . . . . . . . . . . . . . . . . . . . . . .                             | 320   |
| 12 The                           | asset acquisition problem                                    |                                                                             | 322   |
| 12.1                             | The single-period . . .                                      | problem . . . . . . . . . . . . . . . . . . . . . . . . .                   | 323   |
|                                  | 12.1.1                                                       | Properties and optimality conditions . . . . . . . . . . . . . . . . . .    | 325   |
|                                  | 12.1.2                                                       | A stochastic gradient algorithm . . . . . . . . . . . . . . . . . . . . .   | 327   |
|                                  | 12.1.3                                                       | Nonlinear approximations for continuous problems . . . . . . . . . . .      | 328   |
|                                  | 12.1.4                                                       | Piecewise linear approximations . . . . . . . . . . . . . . . . . . . . .   | 329   |
| 12.2                             | The multiperiod asset acquisition problem                    | . . . . . . . . . . . . . . . . . . .                                       | 334   |
|                                  | 12.2.1 . . . . . . . . .                                     | The model . . . . . . . . . . . . . . . . . . . . . . . .                   | 334   |
|                                  | 12.2.2                                                       | Computing gradients with a forward pass . . . . . . . . . . . . . . . .     | 336   |
|                                  | 12.2.3                                                       | Computing gradients with a backward pass . . . . . . . . . . . . . . .      | 336   |
| 12.3                             | Lagged .                                                     | information processes . . . . . . . . . . . . . . . . . . . . . . . . .     | 338   |
|                                  | 12.3.1                                                       | Modeling lagged information processes . . . . . . . . . . . . . . . . .     | 340   |
|                                  | 12.3.2                                                       | Algorithms and approximations for continuously differentiable problems342   |       |
|                                  |                                                              | 12.3.3 Algorithms and approximations for nondifferentiable problems . . . . | 344   |
| 12.4                             | . . . . . .                                                  | Why does it work?** . . . . . . . . . . . . . . . . . . . . . . . . .       | 346   |
|                                  |                                                              | 12.4.1 Proof of convergence of the optimizing version of the SPAR algorithm | 346   |
| 12.5                             | . . . . .                                                    | Bibliographic references . . . . . . . . . . . . . . . . . . . . . . . .    | 354   |
| 13 Batch replenishment processes | 13 Batch replenishment processes                             | 13 Batch replenishment processes                                            | 356   |

| 13.1   | A positive accumulation problem . . . . . . . . . . . . . . . . . . . . . .                        |   356 |
|--------|----------------------------------------------------------------------------------------------------|-------|
|        | 13.1.1 The model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   357 |
|        | 13.1.2 Properties of the value function . . . . . . . . . . . . . . . . . . .                      |   358 |
|        | 13.1.3 Approximating the value function . . . . . . . . . . . . . . . . . .                        |   359 |
|        | 13.1.4 Solving a multiclass problem using linear approximations . . . . .                          |   361 |
| 13.2   | Monotone policies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                    |   363 |
|        | 13.2.1 Submodularity and other stories . . . . . . . . . . . . . . . . . . .                       |   364 |
|        | 13.2.2 From submodularity to monotonicity . . . . . . . . . . . . . . . .                          |   366 |
| 13.3   | Why does it work?** . . . . . . . . . . . . . . . . . . .                                          |   368 |
|        | 13.3.1 Optimality of monotone policies . . . . . . . . . . . . . . . . . . .                       |   368 |
| 13.4   | Bibliographic notes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                    |   373 |
| 14     | Two-stage stochastic programming                                                                   |   376 |
| 14.1   | Two-stage stochastic programs with recourse . . . . . . . . . . . . . . . .                        |   377 |
| 14.2   | Stochastic projection algorithms for constrained optimization . . . . . . .                        |   382 |
| 14.3   | Proximal point algorithms . . . . . . . . . . . . . . . . . . . . . . . . . .                      |   385 |
| 14.4   | The SHAPE algorithm for differentiable functions . . . . . . . . . . . . .                         |   386 |
| 14.5   | Separable, piecewise-linear approximations for nondifferentiable problems                          |   389 |
| 14.6   | Benders decomposition . . . . . . . . . . . . . . . . . . . . . . . . . . . .                      |   392 |
|        | 14.6.1 The basic idea . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                    |   393 |
|        | 14.6.2 Variations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                    |   395 |
|        | 14.6.3 Experimental comparisons . . . . . . . . . . . . . . . . . . . . . .                        |   397 |
| 14.7   | Why does it work?** . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                      |   399 |
|        | 14.7.1 Proof of the SHAPE algorithm . . . . . . . . . . . . . . . . . . .                          |   399 |
| 14.8   | Bibliographic notes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                    |   405 |
| 15     | General asset management problems                                                                  |   407 |
| 15.1   | A basic model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                    |   407 |

## CONTENTS

| 15.3   | A myopic policy . . . . . . . . . . . . . . .               | . . . . . . . . . . . . .                                             | 414                                                                   |
|--------|-------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|
| 15.4   | An approximate dynamic programming strategy . . . . . . . . | . .                                                                   | 416                                                                   |
|        | 15.4.1                                                      | A linear approximation . . . . . . . . . . . . . . . . . . . .        | 416                                                                   |
|        | 15.4.2                                                      | A separable, piecewise linear approximation . . . . . . . .           | 417                                                                   |
|        | 15.4.3                                                      | Network structure, multicommodity problems and the Markov property418 | Network structure, multicommodity problems and the Markov property418 |
| 15.5   | Some numerical experiments                                  | . . . . . . . . . . . . . . . . . . . . .                             | 421                                                                   |
|        | 15.5.1                                                      | Experiments for single commodity flow problems . . . . . .            | 422                                                                   |
|        | 15.5.2                                                      | Experiments for multicommodity flow problems . . . . . .              | 424                                                                   |
| 15.6   | Bibliographic notes . . . .                                 | . . . . . . . . . . . . . . . . . . . . . .                           | 427                                                                   |

## Chapter 1

## The challenges of dynamic programming

The optimization of problems over time arises in many settings, ranging from the control of heating systems to managing entire economies. In between are examples including landing aircraft, purchasing new equipment, managing fleets of vehicles, selling assets, investing money in portfolios or just playing a game of tic-tac-toe or backgammon. As different fields encountered these problems, they tended to discover that certain fundamental equations described their behavior . Known generally as the 'optimality equations,' they have been rediscovered by different fields under names like dynamic programming and optimal control.

This book focuses on a broad range of topics that arise in operations research. Most of these can be categorized as some form of asset management , with the understanding that this term refers to both physical and financial assets. We make an important distinction between problems that involve a single asset and those that involve multiple assets or asset classes. Problems involving a single asset range from selling a bond to routing a single aircraft, but also include playing a game of tic-tac-toe or planning an academic schedule to maximize the chances of graduating from college. In principle, single-asset problems could also include the problem of landing an aircraft or controlling a robot, but we avoid these examples primarily because of their emphasis on low-dimensional controls of continuously differentiable function.

Although single-asset problems represent an important class of applications, the book continuously builds toward problems that involve the management of mulitple assets. Examples include allocating resources between competing projects or activities, managing fleets of containers in international commerce, scheduling pilots over a set of flights, assigning specialists to different tasks over time, upgrading technologies (information technologies, energy generating technologies) over time, and acquiring assets of different types (capital to run a company, jets for a charter airline, oil for national energy needs) to meet demands as they evolve over time.

All of these problems can be formulated as dynamic programs that represent a mathematical framework for modeling problems where information and decisions evolve over time.

Dynamic programming is a fairly mature branch of applied mathematics, but it has struggled with the transition from theory to computation. Most of the textbooks on dynamic programming focus on problems where all the quantities are discrete. A variety of algorithms exist for these problems, but they typically suffer from what is commonly referred to as the 'curse of dimensionality' which we illustrate in the next section.

Dynamic programming has its roots in several fields. Engineering and economics tend to focus on problems with continuous states and decisions (these communities refer to decisions as controls), while the fields of operations research and artificial intelligence work primarily with discrete states and decisions (or actions). Problems that are modeled with continuous states and decisions (and typically in continuous time) are typically addressed under the umbrella of 'control theory' whereas problems with discrete states and decisions, modelled in discrete time, are studied at length under the umbrella of 'Markov decision processes.' Both of these subfields set up recursive equations that depend on the use of a state variable to capture history in a compact way. The study of asset management problems (or more broadly, 'resource allocation') is dominated by the field of mathematical programming (or stochastic programming when we wish to explicitly capture uncertainty), which has evolved without depending on the construct of a state variable. Our treatment draws heavily from all three fields.

## 1.1 A dynamic programming example: a shortest path problem

Perhaps one of the best known applications of dynamic programming is that faced by a typical driver choosing a path in a transportation network. For simplicity (and this is a real simplification for this application), we assume that the driver has to decide at each node (or intersection) which link to traverse next (we are not going to get into the challenges of left turns versus right turns). Let I be the set of intersections. If the driver is at intersection i , he can go to a subset of intersections J + i at a cost c ij . He starts at the origin node s ∈ I and has to find his way to the destination node d ∈ I at the least cost.

The problem can be easily solved using dynamic programming. Let:

v i = The cost to get from intersection i ∈ I to the destination node d .

We assume that v d = 0. Initially, we do not know v i , and so we start by setting v i = M , where ' M ' is known as 'big M' and represents a large number. Let J -j be the set of intersections i such that there is a link from i to j . We can solve the problem by iteratively computing:

<!-- formula-not-decoded -->

Equation (1.1) has to be solved iteratively. We stop when none of the values v i change. It should be noted that this is not a very efficient way of solving a shortest path problem. For

example, in the early iterations, it may well be the case for a particular intersection j that v j = M . In this case, there is no point in executing the update. In efficient implementations, instead of looping over all j ∈ I , we maintain a list of intersections j that we have already reached out to (and in particular those where we just found a better path). The algorithm is guaranteed to stop in a finite number of iterations.

Shortest path problems arise in a variety of settings that have nothing to do with transportation or networks. Consider, for example, the challenge faced by a college freshman trying to plan her schedule to graduation. By graduation, she must take 32 courses overall, including eight departmentals, two math courses, one science course, and two language courses. We can describe the state of her academic program in terms of how many courses she has taken under each of these five categories. Let S tc be the number of courses she has taken by the end of semester t in category c = { Total courses , Departmentals , Math , Science , Language } , and let S t = ( S tc ) c be the state vector. Based on this state, she has to decide which courses to take in the next semester. To graduate, she has to reach the state S 8 = (32 , 8 , 2 , 1 , 2). We assume that she has a measurable desirability for each course she takes, and that she would like to maximize the total desirability of all her courses.

The problem can be viewed as a shortest path problem from the state S 0 = (0 , 0 , 0 , 0 , 0) to S 8 = (32 , 8 , 2 , 1 , 2). Let S t be her current state at the beginning of semester t and let x t represent the decisions she makes while determining what courses to take. We then assume we have access to a transition function f ( S t , x t ) which tells us that if she is in state S t and makes decision x t , she will land in state S t +1 , which we represent by simply using:

<!-- formula-not-decoded -->

In our transportation problem, we would have S t = i if we are at intersection i , and x t would be the decision to 'go to j ,' leaving us in the state S t +1 = j .

Finally, let C t ( S t , x t ) be the contribution or reward she generates from being in state S t and making the decision x t . The value of being in state S t is defined by the equation:

<!-- formula-not-decoded -->

where S t is the set of all possible (discrete) states that she can be in at the beginning of the year.

## 1.2 The three curses of dimensionality

All dynamic programs can be written in terms of a recursion that relates the value of being in a particular state at one point in time to the value of the states that we are carried into at the next point in time. For deterministic problems, this equation can be written:

<!-- formula-not-decoded -->

Equation (1.3) is known as Bellman's equation, or the Hamilton-Jacobi equation, or increasingly, the Hamilton-Jacobi-Bellman equation (HJB for short). Recursions of this sort are fundamental to many classes of dynamic decision-making problems. If we can solve this equation, we can solve our problem. In a simple example such as our shortest path problem, the algorithm is extremely easy, so a student might ask: ' And people spend entire lifetimes on this equation??? '

Bellman's equation is, in fact, very easy to solve if the state variable is something simple, like a street intersection, the amount of money in a retirement account, or the price of a stock. Mathematically, the problems can become quite rich and subtle when we introduce uncertainty. But computationally, the challenge arises when S t (and therefore x t ) is not a scalar, but a vector. For our college student, the state space is approximately 33 × 9 × 3 × 2 × 3 = 5 , 346 (not all states are reachable). If the school adds an additional requirement that the student take at least seven liberal arts courses (to make sure that students in the sciences have breadth in their course selection), the state space grows to 5 , 346 × 8 = 42 , 768. This is the curse of dimensionality at work. In other words, while difficult theoretical questions abound, the real challenge in dynamic programming is computation.

When we introduce uncertainty, we have to find the value of x t that maximizes the expected contribution:

<!-- formula-not-decoded -->

For scalar problems, equation (1.4) is also relatively easy to solve. There are, however, many real problems where the state variable, the random variable (over which we are taking an expectation) and the decision variable are all vectors. For example, state variable S t might be the amount of money we have in different investments, the number of aircraft of different types that a company owns or the location of containers distributed around the country. Random quantities can be the prices of different stocks, the demands for different types of products or the number of loads that need to be moved in containers between different regions. Finally, our decision vector x t can be how much to invest in different stocks, how many aircraft of different types to purchase, or the flows of containers between locations. For these problems, the dimensions of these vectors can range from dozens to tens of thousands to millions.

If we applied our method for solving Bellman's equation to our college student in section 1.1, we would have to solve (1.2) for every possible value of S t . This would require finding the expectation, which involves a summation (if we continue to assume that all quantities are discrete) over all the dimensions of our random quantities. Finally, to find the best value of x t , we have to enumerate over all possible decisions to find the best one. Each one of these steps involves enumerating all outcomes of a multidimensional vector. These problems can be computationally intractable if the number of dimensions is as small as 10 or 20.

7040

Figure 1.1: The major railroads in the United States have to manage complex assets such as boxcars, locomotives and the people who operate them.

<!-- image -->

## 1.3 Some real applications

Asset management problems arise in a broad range of applications. Assets can be physical objects including people, equipment such as aircraft, trucks or electric power transformers, commodities such as oil and food, and fixed facilities such as buildings and power generators. An example of a very complex asset allocation problem arises in railroads. In North America, there are six major railroads (known as 'Class I' railroads) which operate thousands of locomotives, many of which cost over $1 million. Decisions have to be made now to assign locomotives to trains, taking into account how the locomotives will be used at the destination. For example, a train may be going to a location that needs additional power. Or a locomotive may have to be routed to a maintenance facility, and the destination of a train may or may not offer good opportunities for getting the locomotive to the shop.

The balancing of immediate and future costs and rewards is common throughout applications involving freight transportation. In the military, the military airlift command has to make decisions about which aircraft to assign to move passengers or freight. These decisions have to balance what appears to be the best decision now (which aircraft is closest) and the downstream effects of a decision (the destination of a load of freight or passengers may not have good maintenance facilities for a particular aircraft type).

Similar issues arise in the truckload motor carrier industry, where drivers are assigned to move loads that arise in a highly dynamic environment. Large companies manage fleets of thousands of drivers, and the challenge at any moment in time is to find the best driver. There is much more to the problem than simply finding the closest driver; each driver is

11.0

1.0

11.0

1.0

1.0

11.0

11.01

11.0

10

1.0

1.0

1.0

11.0

1.0

10

11.0

1.0

11.0

11.0

dr 29812 Sys 6

dr 29137-Sys\_6

dr 29901. Sys. 6

dr 29985\_ Sys. 6

dr 30197 Sys. 6

dr 30158 Sys 6

dr 30293. Sys 6l dr 27387 Sys 6

27461 SyS

dr 27917\_Sys\_6

Ide 27970\_ Sys 6

dr 28466 Sys 6

|dr 28535 Sys\_6

dr 20875\_Sys\_6

dr\_29130\_ Sys\_6

dr 29220 Sys 6

dr 29383\_Sys 8

dr 34741 Sys\_7

dr 34643 Sys 7

dr 34696 Sys 7

0522

1.0

1.0

1.01

1.0

1.0

1.0

1.0

1.0

1.0

1.0

1.0

1.0

1101

1.0

1.0

0360918

0320349

0624671

0622613

0102029

0624671

0500451

0504475

0102029

0303311

0303311

0523526

0523526

04424321

0102029

0622613

R P

O WER

Figure 1.2: Airlift capacity can be a major bottleneck in military airlift operations

<!-- image -->

Figure 1.3: Large truckload motor carriers have to manage fleets of as many as 10,000 drivers in a highly dynamic environment where shippers place orders at the last minute.

<!-- image -->

characterized by attributes such as his or her home location and equipment type as well as his or her skill level and experience. As with the locomotives, there is a need to balance decisions that maximize profits now versus those that produce good long run behavior. A major challenge is getting drivers back home. In some segments of the industry, a driver may be away for two weeks or more. It is often necessary to look two or more moves into the future to find strategies for routing a driver toward his or her home.

The electric power industry faces the problem of designing and purchasing expensive equipment used to run the electric power grid. It can take a year or more to build one of these components, and each must be designed to a set of specifications. However, it is not always known exactly how the component will be used in the future, as it may be necessary to use the component to respond to an equipment failure. The auto industry also faces design decisions, but in this case the industry has to choose which cars to build and with what features. It is not possible to design the perfect car for each population segment, so

Figure 1.4: The electic power industry has to design, purchase and place expensive equipment that can be used when failures occur.

<!-- image -->

the challenge is to design a car and hope that people are willing to compromise and purchase a particular design. With design cycles that often exceed three years, there is a tremendous amount of uncertainty in these decisions.

A second important asset class is money which can take on the form of cash, money market certificates, bonds, stocks, futures, options and other financial instruments such as derivatives. Since physical objects and money are often interchangeable (money can be used to purchase a physical object; the physical object can be sold and turned into money), the financial community will talk about real assets as opposed to financial assets. .

A third 'asset' which is often overlooked is information. Consider the problem faced by the government which is interested in researching a new technology such as fuel cells or converting coal to hydrogen. There may be dozens of avenues to pursue, and the challenge is to determine which projects to invest in. The state of the system is the set of estimates of how well different components of the technology work. The government funds research to collect information. The result of the research may be the anticipated improvement, or the results may be disappointing. The government wants to plan a research program to maximize the likelihood that a successful technology is developed within a reasonable time frame (say, 20 years). Depending on time and budget constraints, the government may wish to fund competing technologies in the event that one does not work. Alternatively, it may be more effective to fund one promising technology, and to then switch to an alternative if the first does not work out.

## 1.4 Problem classes in asset management

The vast array of applications in asset management can be divided into some major problem classes. We use these problem classes throughout the book to motivate various algorithmic strategies.

The budgeting problem. Here we face the problem of allocating a fixed resource over a set of activities that returns a reward which is a function of how much we invest in the activity. For example, drug companies have to decide how much to invest in different research projects or how much to spend on advertising for different drugs. Oil exploration companies have to decide how much to spend exploring potential sources of oil. Political candidates have to decide how much time to spend campaigning in different states.

Asset acquisition with concave costs. A company can raise capital by issuing stock or floating a bond. There are costs associated with these financial instruments independent of how much money is being raised. Similarly, an oil company purchasing oil will be given quantity discounts (or it may face the fixed cost of purchasing a tankerload of oil). Retail outlets get a discount if they purchase a truckload of an item. All of these are instances of acquiring assets with a concave (or more generally, non-convex) cost function, which means there is an incentive for purchasing larger quantities.

Asset acquisition with lagged information processes. We can purchase commodity futures that allow us to purchase a product in the future at a lower cost. Alternatively, we may place an order for memory chips from a factory in southeast Asia with one to two week delivery times. A transportation company has to provide containers for a shipper who may make requests several days in advance or at the last minute. All of these are asset acquisition problems with lagged information processes .

Buying/selling an asset. In this problem class, the process stops when we either buy an asset when it looks sufficiently attractive or sell an asset when market conditions warrant. The game ends when the transaction is made. For these problems, we tend to focus on the price (the purchase price or the sales price), and our success depends on our ability to trade off current value with future price expectations.

General asset allocation problems. This class encompasses the problem of managing reusable and substitutable assets over time. Applications abound in transportation and logistics. Railroads have to move locomotives and boxcars to serve different activities (moving trains, moving freight) over time. An airline has to move aircraft and pilots in order to move passengers. Consumer goods have to move through warehouses to retailers to satisfy customer demands.

Demand management. There are many applications where we focus on managing the demands being placed on a process. Should a hospital admit a patient? Should a trucking company accept a request by a customer to move a load of freight?

Shortest paths. In this problem class, we typically focus on managing a single, discrete resource. The resource may be someone playing a game, a truck driver we are trying to

route to return him home, a driver who is trying to find the best path to his destination or a locomotive we are trying to route to its maintenance shop. Shortest path problems, however, also represent a general mathematical structure that applies to a broad range of dynamic programs that have nothing to do with routing a physical asset to a destination.

Dynamic assignment. Consider the problem of managing multiple resources, such as computer programmers, to perform different tasks over time (writing code or fixing bugs). Each resource and task is characterized by a set of attributes that determines the cost (or contribution) from assigning a particular resource to a particular task.

All of these problems focus on the problem of managing physical or financial assets. They provide an idea of the diversity of applications that can be studied. In each case, we have focused on the question of how to manage the asset. In addition, there are three other classes of questions that arise for each application:

Pricing. Often the question being asked is what price should be paid for an asset. The right price for an asset depends on how it is managed, so it should not be surprising that we often find asset prices as a byproduct.

Information collection. Since we are modeling sequential information and decision processes, we explicitly capture the information that is available when we make a decision, allowing us to undertake studies that change the information process. For example, the military uses unmanned aerial vehicles (UAV's) to collect information about targets in a military setting. Oil companies drill holes to collect information about underground geologic formations. Travelers try different routes to collect information about travel times. Pharmaceutical companies use test markets to experiment with different pricing and advertising strategies.

In addition, the algorithmic strategies that we pursue under the umbrella of approximate dynamic programming all involve the need to explore different regions of the state space to estimate the value of being in these states. These strategies require that we understand the tradeoff between the cost (time) required to visit different states and the benefits derived from improving the precision with which we can estimate the value of being in a state.

Technology switching. The last class of questions addresses the underlying technology that controls how the physical process evolves over time. For example, when should a power company upgrade a generating plant (e.g. to burn oil and natural gas)? Should an airline switch to aircraft that fly faster or more efficiently? How much should a communications company invest in a technology given the likelihood that better technology will be available in a few years?

Most of these problems arise in both discrete and continuous versions. Continuous models would be used for money, physical products such as oil, grain and coal, or discrete products that occur in large volume (most consumer products). In other settings, it is important to retain the integrality of the assets being managed (people, aircraft, locomotives, trucks, and expensive items that come in small quantities). For example, how do we position emergency response units around the country to respond to emergencies (bioterrorism, major oil spills,

failure of certain components in the electric power grid)?

What makes these problems hard? With enough assumptions, none of these problems are inherently difficult. But in real applications, a variety of issues emerge that can make all of them intractable. These include:

- Evolving information processes - We have to make decisions now before we know the information that will arrive later. This is the essence of stochastic models, and this property quickly turns the easiest problems into computational nightmares.
- High dimensional problems - Most problems are easy if they are small enough. In real applications, there can be many types of assets, producing decision vectors of tremendous size.
- Measurement problems - Normally, we assume that we look at the state of our system and from this determine what decision to make. In many problems, we cannot measure the state of our system precisely. The problem may be delayed information (stock prices), incorrectly reported information (the truck is in the wrong location), misreporting (a manager does not properly add up his total sales), theft (retail inventory), or deception (an equipment manager underreports his equipment so it will not be taken from him).
- Unknown models (information, system dynamics) - We can anticipate the future by being able to say something about what might happen (even if it is with uncertainty) or the effect of a decision (which requires a model of how the system evolves over time).
- Missing information - There may be costs that simply cannot be computed, and are instead ignored. The result is a consistent model bias (although we do not know when it arises).
- Comparing solutions - Primarily as a result of uncertainty, it can be difficult comparing two solutions to determine which is better. Should we be better on average, or are we interested in the best, worst solution? Do we have enough information to draw a firm conclusion?

## 1.5 What is new in this book?

As of this writing, dynamic programming has enjoyed a relatively long history, with many superb books. Within the operations research community, the original text by Bellman (Bellman (1957)) was followed by a sequence of books focusing on the theme of Markov decision processes. Of these, the current high-water mark is Markov Decision Processes by Puterman, which played an influential role in the writing of chapter 4. This field offers a powerful theoretical foundation, but the algorithms are limited to problems with very low dimensional state and action spaces.

This volume focuses on a field that is coming to be known as approximate dynamic programming which emphasizes modeling and computation for much harder classes of problems. The problems may be hard because they are large (for example, large state spaces), or because we lack a model of the underlying process which the field of Markov decision processes takes for granted. Two major references precede this volume. Neuro-Dynamic Programming by Bertsekas and Tsitsiklis was the first book to appear that summarized a vast array of strategies for approximating value functions for dynamic programming. Reinforcement Learning by Sutton and Barto presents the strategies of approximate dynamic programming in a very readable format, with an emphasis on the types of applications that are popular in the computer science/artificial intelligence community.

This volume focuses on models of problems that can be broadly described as 'asset management,' where we cover both physical and financial assets. Many of these applications involve very high dimensional decision vectors (referred to as controls or actions in other communities) which can only be solved using the techniques from the field of mathematical programming. As a result, we have adopted a notational style that makes the relationship to the field of math programming quite transparent. A major goal of this volume is to lay the foundation, starting as early as chapter 3, for solving these very large and complex problems.

There are several major differences between this volume and these major works which precede it.

- We focus much more heavily on the modeling of problems. Emphasis is placed throughout on the proper representation of exogenous information processes and system dynamics. Partly for this reason, we present finite-horizon models first since it requires more careful modeling of time than is needed for steady state models.
- Examples are drawn primarily from the classical problems of asset management that arise in operations research. We make a critical distinction between single asset problems (when to sell an asset, how to fly a plane from one location to another) and problems with multiple asset classes (how to manage a fleet of aircraft, purchasing different types of equipment, managing money in different forms of investments) by introducing specific notation for each.
- We bring together the power of approximate dynamic programming, which is itself represents a merger of Markov decision processes and stochastic approximation methods, with stochastic programming and math programming. The result is a new class of algorithms for solving (approximately) complex resource allocation problems which exhibit state and action (decision) vectors with thousands or even tens of thousands of dimensions. The notation is chosen to facilitate the link between dynamic programming and math programming.
- We explicitly identify the three curses of dimensionality that arise in asset management problems, and introduce an approximation strategy based on using the post-decision state variable which has received limited attention in the literature (and apparently no attention in other textbooks).

- The theoretical foundations of this material can be deep and rich, but our presentation is aimed at a undergraduate or masters level students with introductory courses in statistics, probability and, for the later chapters, linear programming. For more advanced students, proofs are provided in 'Why does it work' sections. The presentation is aimed primarily at students interested in taking real, complex problems, producing proper mathematical models and developing computationally tractable algorithms.

Our presentation integrates the contributions of several communities. Much of the foundational theory was contributed by the probability community in the study of Markov decision processes and, in a separate subcommunity, stochastic approximation methods. We also recognize the many contributions that have emerged from the control theory community, generally in the context of classical engineering problems. Finally, we integrate important contributions from stochastic programming for solving high dimensional decision problems under uncertainty. We think that this volume, by bringing these different fields together, will contribute to the thinking in all fields.

## 1.6 The many dialects of dynamic programming

Dynamic programming arises from the study of sequential decision processes. Not surprisingly, these arise in a wide range of applications. While we do not wish to take anything from Bellman's fundamental contribution, the optimality equations are, to be quite honest, somewhat obvious. As a result, they were discovered independently by the different communities in which these problems arise.

The problems arise in a variety of engineering problems, typically in continuous time with continuous control parameters. These applications gave rise to what is now referred to as control theory. While uncertainty is a major issue in these problems, the formulations tend to focus on deterministic problems (the uncertainty is typically in the estimation of the state or the parameters that govern the system). Economists adopted control theory for a variety of problems involving the control of activities from allocating single budgets or managing entire economies (admittedly at a very simplistic level). Operations research (through Bellman's work) did the most to advance the theory of controlling stochastic problems, thereby producing the very rich theory of Markov decision processes. Computer scientists, especially those working in the realm of artificial intelligence, found that dynamic programming was a useful framework for approaching certain classes of machine learning problems known as reinforcement learning.

It is not simply the case that different communities discovered the fundamentals of dynamic programming independently. They also discovered the computational challenges that arise in their solution (the 'curse of dimensionality'). Not surprisingly, different communities also independently discovered classes of solution algorithms.

As different communities discovered the same concepts and algorithms, they invented their own vocabularies to go with them. As a result, we can solve the Bellman equations, the

Hamiltonian, the Jacobian, the Hamilton-Jacobian, or the all-purpose Hamilton-JacobianBellman equations (typically referred to as the HJB equations). In our presentation, we prefer the term 'optimality equations.'

There is an even richer vocabulary for the types of algorithms that are the focal point of this book. Everyone has discovered that the backward recursions required to solve the optimality equations in section 2.1.1 do not work as the number of dimensions increases. A variety of authors have independently discovered that an alternative is to step forward through time, using iterative algorithms to help estimate the value function. This general strategy has been referred to as forward dynamic programming, iterative dynamic programming, adaptive dynamic programming, and neuro-dynamic programming. However, the term that appears to have been most widely adopted is approximate dynamic programming . In some cases, the authors would discover the algorithms and only later discover their relationship to classical dynamic programming.

The use of iterative algorithms that are the basis of most approximate dynamic programming procedures also have their roots in a field known as stochastic approximation methods. Again, authors tended to discover the technique and only later learn of its relationship to the field of stochastic approximation methods. Unfortunately, this relationship was sometimes discovered only after certain terms became well established.

Throughout the presentation, students need to appreciate that many of the techniques in the fields of approximate dynamic programming and stochastic approximation methods are fundamentally quite simple, and often obvious. The proofs of convergence and some of the algorithmic strategies can become quite difficult, but the basic strategies often represent what someone would do with no training in the field. As a result, the techniques frequently have a very natural feel to them, and the algorithmic challenges we face often parallel problems we encounter in every day life.

As of this writing, the relationship between control theory (engineering and economics), Markov decision processes (operations research), and reinforcement learning (computer science/artificial intelligence) are well understood by the research community. The relationship between iterative techniques (reviewed in chapter 5) and the field of stochastic approximations is also well established.

There is, however, a separate community that evolved from the field of deterministic math programming, which focuses on very high dimensional problems. As early as the 1950's, this community was trying to introduce uncertainty into mathematical programs. The resulting subcommunity is called stochastic programming which uses a vocabulary that is quite distinct from that of dynamic programming. The relationship between dynamic programming and stochastic programming has not been widely recognized, despite the fact that Markov decision processes are considered standard topics in graduate programs in operations research.

Our treatment will try to bring out the different dialects of dynamic programming, although we will tend toward a particular default vocabulary for important concepts. Students need to be prepared to read books and papers in this field that will introduce and develop

important concepts using a variety of dialects. The challenge is realizing when authors are using different words to say the same thing.

## 1.7 Bibliographic notes

Basic references: Bellman (1957),Howard (1971),Derman (1970), Dynkin (1979), Ross (1983), Heyman &amp; Sobel (1984),Puterman (1994) , Bertsekas &amp; Tsitsiklis (1996)

Sequential allocation: Taylor (1967)

## Chapter 2

## Some illustrative applications

Dynamic programming is one of those incredibly rich fields that has filled the careers of many. But it is also a deceptively easy idea to illustrate and use. This chapter presents a series of dynamic programming problems that lend themselves to simple (often analytical) solutions. The goal is to teach dynamic programming by example.

It is possible, after reading this chapter, to conclude that 'dynamic programming is easy' and to wonder 'why do I need the rest of this book?' The answer is: sometimes dynamic programming is easy and requires little more than the understanding gleaned from these simple problems. But there is a vast array of problems which are quite difficult to model, and where standard solution approaches are computationally intractable.

We divide our presentation between deterministic and stochastic problems. The careful reading will pick up on subtle modeling differences between these problems. If you do not notice these, chapter 3 brings these out more explicitly.

## 2.1 Deterministic problems

## 2.1.1 The budgeting problem

A problem with similar structure to the gambling problem is the budgeting problem. Here, we have to allocate a budget of size R to a series of tasks T . Let x t be the amount of money allocated to task t , and let C t ( x t ) be the contribution (or reward) that we receive from this allocation. We would like to maximize our total contribution:

<!-- formula-not-decoded -->

subject to the constraint on our available resources:

<!-- formula-not-decoded -->

In addition, we cannot allocate negative resources to any task, so we include:

<!-- formula-not-decoded -->

We refer to (2.1)-(2.3) as the budgeting problem (other authors refer to it as the 'resource allocation problem,' a term we find too general for such a simple problem). In this example, all data is deterministic. There are a number of algorithmic strategies for solving this problem that depend on the structure of the contribution function, but we are going to show how it can be solved without any assumptions.

We will approach this problem by first deciding how much to allocate to task 1, then to task 2, and so on until the last task, T . In the end, however, we want a solution that optimizes over all tasks. Let:

V t ( R t ) = The value of having R t resources remaining before we solve the problem of allocating for task t

Implicit in our definition of V t ( R t ) is that we are going to solve the problem of allocating R t over tasks t, t + 1 , . . . , T in an optimal way. Imagine that we somehow know the function V t +1 ( R t +1 ), where R t +1 = R t -x t . Then it seems apparent that the right solution for task t is to solve:

<!-- formula-not-decoded -->

Equation (2.4) forces us to balance between the contributions that we receive from task t and what we would receive from all the remaining tasks (which is captured in V t +1 ( R t -x t )). One way to solve (2.4) is to assume that x t is discrete. For example, if our budget is R = $10 million, we might require x t to be in units of $100,000 dollars. In this case, we would solve (2.4) simply by searching over all possible values of x t (since it is a scalar, this is not too hard). The problem is that we do not know what V t +1 ( R t +1 ) is.

The simplest strategy for solving our dynamic program in (2.4) is to start by using V T +1 ( R ) = 0 (for any value of R ). Then we would solve:

<!-- formula-not-decoded -->

for 0 ≤ R T ≤ R . Now we know V T ( R T ) for any value of R T that might actually happen. Next we can solve:

<!-- formula-not-decoded -->

Clearly, we can play this game recursively, solving:

<!-- formula-not-decoded -->

for t = T -1 , T -2 , . . . , 1. Once we have computed V t for t ∈ T , we can then start at t = 1 and step forward in time to determine our optimal allocations.

This strategy is simple, easy and optimal. It has the nice property that we do not need to make any assumptions about the shape of C t ( x t ), other than finiteness. We do not need concavity or even continuity; we just need the function to be defined for the discrete values of x t that we are examining.

## 2.1.2 The shortest path problem

Perhaps one of the most popular dynamic programming problems is known as the shortest path problem. Although it has a vast array of applications, it is easiest to describe in terms of the problem faced by every driver when finding a path from one location to the next over a road network. Let:

I = The set of nodes (intersections) in the network.

L = The set of links ( i, j ) in the network.

c ij = The cost (typically the time) to drive from node i to node j , i, j ∈ I , ( i, j ) ∈ L

N i = The set of nodes j for which there is a link ( i, j ) ∈ L .

We assume that a traveler at node i can choose to traverse any link ( i, j ) where j ∈ N i . Assume our traveler is starting at some node r and needs to get to a destination node s at the least cost. Let:

v j = The minimum cost required to get from node j to node s .

/negationslash

Initially, we do not know v j , but we do know that v s = 0. Let v n j be our estimate, at iteration n , of the cost to get from j to s . We can find the optimal costs, v j , by initially setting v 0 j to a large number for j = s , and then iteratively looping over all the intersections, finding the best link to traverse out of an intersection i by minimizing the sum of the outbound link cost c ij plus our current estimate of the downstream value v n -1 j . The complete algorithm is summarized in figure 2.1. This algorithm has been proven to converge to the optimal set of node values.

There is a substantial literature on solving shortest path problems. Because they arise in so many applications, there is tremendous value in solving them very quickly. Our basic algorithm is not very efficient because we are often solving equation (2.8) for an intersection i where v n -1 i = M , and where v n -1 j = M for all j ∈ N i . A more standard strategy is to

Step 0. Let

Step 0. Let

/negationslash

<!-- formula-not-decoded -->

where ' M ' is known as 'big-M' and represents a large number. Let n = 1.

Step 1. Solve for all i ∈ I :

<!-- formula-not-decoded -->

Step 2. If v n i &lt; v n -1 i for any i , let n := n +1 and return to step 1. Else stop.

Figure 2.1: Basic shortest path algorithm

/negationslash

<!-- formula-not-decoded -->

where ' M ' is known as 'big-M' and represents a large number. Let n = 1. Set the candidate list C = { r } .

Step 1. Choose node i ∈ C from the top of the candidate list.

Step 2. For all nodes j ∈ N i do:

Step 2a.

<!-- formula-not-decoded -->

Step 2b. If ˜ v n j &lt; v n -1 j and if j /negationslash∈ C , then set v n j = ˜ v n j and add j to the candidate list: C = C ∪ { j } ( j is assumed to be put at the bottom of the list).

Step 3. Drop node i from the candidate list. If the candidate list C is not empty, return to step 1.

Figure 2.2: An origin-based shortest path algorithm

maintain a candidate list of nodes C which consists of an ordered list i 1 , i 2 , . . . . Initially the list will consist only of the origin node r . As we reach out of a node i in the candidate list, we may find a better path to some node j which is then added to the candidate list (if it is not already there).

This is often referred to as Bellman's algorithm, although the algorithm in figure 2.1 is a purer form of Bellman's equation for dynamic programming. A very effective variation of the algorithm in 2.2 is to keep track of nodes which have already been in the candidate list. If a node is added to the candidate list which was previously in the candidate list, a very effective strategy is to add this node to the top of the list. This variant is known as Pape's algorithm (pronounced 'papa's algorithm'). Another powerful variation, called Dijkstra's algorithm (pronounced 'Diekstra') chooses the node from the candidate list with

the smallest value of v n i .

Almost any (deterministic) discrete dynamic program can be viewed as a shortest path problem. We can view each node i as representing a particular discrete state of the system. The origin node r is our starting state, and the ending state s might be any state at an ending time T . We can also have shortest path problems defined over infinite horizons, although we would typically include a discount factor.

We are often interested in problems where there is some source of uncertainty. For our shortest path problem, it is natural to view the cost on a link as random, reflecting the variability in the travel time over each link. There are two ways we can handle the uncertainty. The simplest is to assume that our driver has to make a decision before seeing the travel time over the link. In this case, our updating equation would look like:

<!-- formula-not-decoded -->

where W is some random variable describing the network. This problem is identical to our original problem; all we have to do is to let c ij = E { c ij ( W ) } be the expected cost on an arc.

<!-- formula-not-decoded -->

Here, the expectation is outside of the min operator which chooses the best decision, capturing the fact that now the decision itself is random.

If we go outside of our transportation example, there are many settings where the decision does not take us deterministically to a particular node j .

## 2.2 Stochastic problems

## 2.2.1 The gambling problem

A gambler has to determine how much of his capital he should bet on each round of a game, where he will play a total of N rounds. He will win a bet with probability p and lose with probability q = 1 -p (assume q &lt; p ). Let s n be his total capital after n plays, n = 1 , 2 , . . . , N , with s 0 being his initial capital. For this problem, we refer to s n as the state of the system. Let x n be the amount he bets in round n , where we require that x n ≤ s n -1 . He wants to maximize ln s N (this provides a strong penalty for ending up with a small amount of money at the end and a declining marginal value for higher amounts).

Let

<!-- formula-not-decoded -->

The system evolves according to:

<!-- formula-not-decoded -->

Let V n ( s n ) be the value of having s n dollars at the end of the n th game. The value of being in state s n at the end of the n th round can be written:

<!-- formula-not-decoded -->

Here, we claim that the value of being in state s n is found by choosing the decision that maximizes the expected value of being in state s n +1 given what we know at the end of the n th round.

We solve this by starting at the end of the N th trial, and assuming that we have finished with S N dollars. The value of this is:

<!-- formula-not-decoded -->

Now step back to n = N -1, where we may write:

<!-- formula-not-decoded -->

Let V N -1 ( s N -1 , x N ) be the value within the max operator. We can find x N by differentiating V N -1 ( s N -1 , x N ) with respect to x N , giving:

<!-- formula-not-decoded -->

Setting this equal to zero and solving for x N gives:

<!-- formula-not-decoded -->

The next step is to plug this back into (2.10) to find V N -1 ( s N -1 ):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where K is a constant with respect to s N -1 . Since the additive constant does not change our decision, we may ignore it and use V N -1 ( s N -1 ) = ln s N -1 as our value function for N -1, which is the same as our value function for N . Not surprisingly, we can keep applying this same logic backward in time and obtain:

<!-- formula-not-decoded -->

for all n , where again, K n is some constant that can be ignored. This means that for all n , our optimal solution is:

<!-- formula-not-decoded -->

The optimal strategy at each iteration is to bet a fraction β = (2 p -1) of our current money on hand. Of course, this requires that p &gt; . 5.

## 2.2.2 The batch replenishment problem

One of the classical problems in operations research is one that we refer to here as the batch replenishment problem. To illustrate the basic problem, assume that we have a single type of resource which is consumed over time. As the reserves of the resource run low, it is necessary to replenish the resources. In many problems, there are economies of scale in this process. It is cheaper (on an average cost basis) to increase the level of resources in one jump (see examples).

Example 2.1: A startup company has to maintain adequate reserves of operating capital to fund product development and marketing. As the cash is depleted, the finance officer has to go to the markets to raise additional capital. There are fixed costs of raising capital, so this tends to be done in batches.

Example 2.2: An oil company maintains an aggregate level of oil reserves. As these are depleted, it will undertake exploration expeditions to identify new oil fields, which will produce jumps in the total reserves under the company's control.

We address this problem in some depth in chapter 13. We use it here simply as an illustration of dynamic programming methods. Let:

ˆ D t = Demand for the resources during time interval t .

R t = Resource level at time t .

x t = Additional resources acquired at time t to be used during time interval t +1.

The transition function is given by:

<!-- formula-not-decoded -->

Our one period cost function (which we wish to minimize) is given by:

<!-- formula-not-decoded -->

c h = The unit holding cost.

For our purchases, C t ( x t ) could be any nonconvex function; this is a simple example of one. Since the cost function is nonconvex, it helps to order larger quantities at the same time.

Assume that we have a family of decision functions X π ( R t ) for determining x t . Our goal is to solve:

<!-- formula-not-decoded -->

This problem class often yields policies that take a form such as 'if the resource level goes below a certain amount, then order up to a fixed amount.'

The basic batch replenishment problem, where R t and x t are scalars, is quite easy (if we know things like the distribution of demand). But there are many real problems where these are vectors because there are different types of resources. The vectors may be small (different types of fuel, raising different types of funds) or extremely large (hiring different types of people for a consulting firm or the military; maintaining spare parts inventories). Even a small number of dimensions would produce a very large problem using a discrete representation.

## 2.2.3 The secretary problem

The so-called secretary problem (Cayley (1875)) is one of the classic problems of dynamic programming. The motivation of the problem is determining when to hire a candidate for a job (presumably a secretarial position), but it can also be applied to reviewing a series of offers for an asset (such as selling your house or car). This problem provides a nice illustration of a dynamic programming problem that can be solved very easily.

## Setup

Assume that we have N candidates for a secretarial position. Each candidate is interviewed in sequence and assigned a score that allows us to compare him or her to other candidates. While it may be reasonable to try to maximize the expected score that we would receive, in this case, we want to maximize the probability of hiring the best candidate out of the entire pool. We need to keep in mind that if we stop at candidate n , then we would not have even interviewed candidates n +1 , . . . , N .

Let:

ω n = Score of the n th candidate.

s n = { 1 If the score of the n th candidate is the best so far 0 Otherwise.

S = State space, given by (0 , 1 , ∆), where the states 0 and 1 mean that we are still searching, and ∆ means we have stopped the process.

X = { 0( continue ) , 1( quit ) } , where 'quit' means that we hire the last candidate interviewed.

Because the decision function uses the most recent piece of information, we define our history as:

<!-- formula-not-decoded -->

To describe the system dynamics, it is useful to define an indicator function:

<!-- formula-not-decoded -->

which tells us if the last observation is the best. Our system dynamics can now be given by:

<!-- formula-not-decoded -->

To compute the one-step transition matrix, we observe that the event the n +1 st applicant is the best has nothing to do with whether the n th was the best. As a result:

<!-- formula-not-decoded -->

This simplifies the problem of finding the one-step transition matrix. By definition we have:

<!-- formula-not-decoded -->

I n +1 ( h n +1 ) = 1 if the n +1 st candidate is the best out of the first n +1, which clearly occurs with probability 1 / ( n +1). So:

<!-- formula-not-decoded -->

Our goal is to maximize the probability of hiring the best candidate. So, if we do not hire the last candidate, then the probability that we hired the best candidate is zero. If we hire the n th candidate, and the n th candidate is the best so far, then our reward is the probability that this candidate is the best out of all N . This probability is simply the probability that the best candidate out of all N is one of the first n , which is n/N . So, the conditional reward function is:

<!-- formula-not-decoded -->

With this information, we can now set up the optimality equations:

<!-- formula-not-decoded -->

## Solution

The solution to the problem is quite elegant, but the technique is unique to this particular problem. Readers interested in the elegant answer but not the particular proof (which illustrates dynamic programming but otherwise does not generalize to other problem classes) can skip to the end of the section.

Let:

V n ( s ) = The probability of choosing the best candidate out of the entire population, given that we are in state s after interviewing the n th candidate.

Recall that implicit in the definition of our value function is that we are behaving optimally from time period t onward. The terminal reward is:

V N (1) = 1

<!-- formula-not-decoded -->

V N (∆) = 0

The optimality recursion for the problem is given by:

<!-- formula-not-decoded -->

Noting that:

<!-- formula-not-decoded -->

We get:

<!-- formula-not-decoded -->

Similarly, it is easy to show that:

<!-- formula-not-decoded -->

Comparing (2.12) and (2.11), we can rewrite (2.11) as:

<!-- formula-not-decoded -->

From this we obtain the inequality:

<!-- formula-not-decoded -->

which seems pretty intuitive (you are better off if the last candidate you interviewed was the best you have seen so far).

At this point, we are going to suggest a policy that seems to be optimal. We are going to interview the first ¯ n candidates, without hiring any of them. Then, we will stop and hire the first candidate who is the best we have seen so far. The decision rule can be written as

<!-- formula-not-decoded -->

To prove this, we are going to start by showing that if V m (1) &gt; m/N for some m (or alternatively if V m (1) = m/N = V m (0)), then V m ′ (1) &gt; m ′ /N for m ′ &lt; m . If V m (1) &gt; m/N , then it means that the optimal decision is to continue. We are going to show that if it was optimal to continue at set m , then it was optimal to continue for all steps m ′ &lt; m .

Assume that V m (1) &gt; m/N . This means, from equation (2.13), that it was better to continue, which means that V m (1) = V m (0) (or there might be a tie, implying that V m (1) = m/N = V m (0)). This allows us to write:

<!-- formula-not-decoded -->

Equation (2.15) is true because V m (1) = V m (0), and equation (2.16) is true because V m (1) ≥ m/N . Stepping back in time, we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (2.17) is true because V m -1 (0) ≥ m/N . We can keep repeating this for m -1 , m -2 , . . . , so it is optimal to continue for m ′ &lt; m .

Now we have to show that if N &gt; 2, then ¯ n ≥ 1. If this is not the case, then for all n , V n (1) = n/N (because we would never quit). This means that (from equation (2.12)):

<!-- formula-not-decoded -->

Using V N (0) = 0, we can solve (2.19) by backward induction:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In general, we get:

<!-- formula-not-decoded -->

We can easily see that V 1 (0) &gt; 1 N ; since we were always continuing, we had found that V 1 (1) = 1 N . Finally, equation (2.14) tells us that V 1 (1) ≥ V 1 (0), which means we have a contradiction.

This structure tells us that for m ≤ ¯ n :

<!-- formula-not-decoded -->

and for m&gt; ¯ n :

<!-- formula-not-decoded -->

It is optimal to continue as long as V m (0) &gt; m/N , so we want to find the largest value for m such that:

<!-- formula-not-decoded -->

or:

<!-- formula-not-decoded -->

If N = 5, then we can solve by enumeration:

<!-- formula-not-decoded -->

So for N = 5, we would use ¯ n = 2. This means interview (and discard) two candidates, and then take the first candidate that is the best to date.

For large N , we can find a neat approximation. We would like to find m such that:

<!-- formula-not-decoded -->

Solving for m means finding ln( N/m ) = 1 or N/m = e or m/N = e -1 = 0 . 368. So, for large N , we want to interview 37 percent of the candidates, and then choose the first candidate that is the best to date.

## 2.2.4 Optimal stopping

A particularly important problem in asset pricing is known as the optimal stopping problem . Imagine that you are holding an asset which you can sell at a price that fluctuates randomly. Let ˆ p t be the price that is revealed in period t , at which point you have to make a decision:

<!-- formula-not-decoded -->

Our system has two states:

<!-- formula-not-decoded -->

If have sold the asset, then there is nothing we can do. We want to maximize the price we receive when we sell our asset. Let the scalar V t be the value of holding the asset at time t . This can be written:

<!-- formula-not-decoded -->

So, we either get the price ˆ p t if we sell, or we get the discounted future value of the asset. Assuming the discount factor γ &lt; 1, we do not want to hold too long simply because the value in the future is worth less than the value now. In practice, we eventually will see a price ˆ p t that is greater than the future expected value, at which point we would stop the process and sell our asset.

The time at which we sell our asset is known as a stopping time . By definition, x τ = 1. It is common to think of τ as the decision variable, where we wish to solve:

<!-- formula-not-decoded -->

Equation (2.20) is a little tricky to interpret. Clearly, the choice of when to stop is a random variable since it depends on the price ˆ p t . We cannot optimally choose a random variable, so what is meant by (2.20) is that we wish to choose a function (or policy ) that determines when we are going to sell. For example, we would expect that we might use a rule that says:

<!-- formula-not-decoded -->

In this case, we have a function parameterized by ¯ p . In this case, we would write our problem in the form:

<!-- formula-not-decoded -->

This formulation raises two questions. First, while it seems very intuitive that our policy would take the form given in equation (2.21), there is the theoretical question of whether this in fact is the structure of an optimal policy. Questions of this type can be quite theoretical in nature. Chapter 13 demonstrate how these questions can be answered in the context of a class of batch replenishment problems. The second question is how to find the best policy within this class. For this problem, that means finding the parameter ¯ p . For problems where the probability distribution of the random process driving prices is (assumed) known, this is a rich and deep theoretical challenge. Alternatively, there is a class of algorithms from stochastic optimization that allows us to find 'good' values of the parameter in a fairly simple way.

## 2.3 Bibliographic notes

The examples provided in this chapter represent classic problems that can be found in a number of sources. The presentation of the secretary problem is based on Puterman (1994). Nice presentations of simple dynamic programs can be found in Ross (1983), Whittle (1982) and Bertsekas (2000).

## Exercises

- 2.1) Give an example of a sequential decision process from your own experience. Describe the decisions that have to be made, the exogenous information process, the state variable, and the cost or contribution function. Then describe the types of rules you might use to make a decision.
- 2.2) Repeat the gambling problem assuming that the value of ending up with S N dollars is √ S N .
- 2.3) Write out the steps of a shortest path algorithm, similar to that shown in figure 2.2 which starts at the destination and works backward to the origin.
- 2.4) Consider a budget problem where you have to allocate R advertising dollars (which can be treated as a continuous variable) among T = (1 , 2 , . . . , T ) products. Let x t be the total dollars allocated to product t , where we require that ∑ t ∈T x t ≤ R . Further assume that the increase in sales for product t is given by √ x t (this is the contribution we are trying to maximize).

- a) Set up the optimality equation (similar to equation (2.7)) for this problem, where the state variable R t is the total remaining funds that remain after allocating funds to products (1 , 2 , . . . , t -1).
- b) Assuming you have R T available to be allocated to the last product, what is the optimal allocation to the last product? Use this answer to write out an expression for V T ( R T ).
- c) Use your answer to part (b) to find the optimal allocation to product T -1 assuming we have R T -1 dollars available to be allocated over the last two products. Find the optimal allocation x T -1 and an expression for V T -1 .
- d) By now, you should see a pattern forming. Propose what appears to be the functional form for V t ( R t ) and use inductive reasoning to prove your conjecture by showing that it returns the same functional form for V t -1 ( R t -1 ).
- e) What is your final optimal allocation over all products?
6. 2.5) Repeat exercise (2.4) assuming that the reward for product t is c t √ x t .
7. 2.6) Repeat exercise (2.4) assuming that the reward (the increased sales) for product t is given by ln ( x ).
8. 2.7) Repeat exercise (2.4) one more time, but now assume that all you know is that the reward is continuously differentiable, monotonically increasing and concave.
9. 2.8) What happens to your answer to the budget allocation problem (for example, exercise 2.4) if the contribution is convex instead of concave?
10. 2.9) We are now going to do a budgeting problem where the reward function does not have any particular properties. It may have jumps, as well as being a mixture of convex and concave functions. But this time we will assume that R = $30 dollars and that the allocations x t must be in integers between 0 and 30. Assume that we have T = 5 products, with a contribution function C t ( x t ) = cf ( x t ) where c = ( c 1 , . . . , c 5 ) = (3 , 1 , 4 , 2 , 5) and where f ( x ) is given by:

<!-- formula-not-decoded -->

Find the optimal allocation of resources over the five products.

- 2.10) You suddenly realize towards the end of the semester that you have three courses that have assigned a term project instead of a final exam. You quickly estimate how much each one will take to get 100 points (equivalent to an A+) on the project. You then

guess that if you invest t hours in a project, which you estimated would need T hours to get 100 points, that for t &lt; T , your score will be:

<!-- formula-not-decoded -->

That is, there are declining marginal returns to putting more work into a project. So, if a project is projected to take 40 hours and you only invest 10, you estimate that your score will be 50 points (100 times the square root of 10 over 40). You decide that you cannot spend more than a total of 30 hours on the projects, and you want to choose a value of t for each project that is a multiple of 5 hours. You also feel that you need to spend at least 5 hours on each project (that is, you cannot completely ignore a project). The time you estimate to get full score on each of the four projects is given by:

|   Project 1 |   Completion time T 20 |
|-------------|------------------------|
|           2 |                     15 |
|           3 |                     10 |

You decide to solve the problem as a dynamic program.

- a) What is the state variable and decision epoch for this problem?
- b) What is your reward function?
- c) Write out the problem as an optimization problem.
- d) Set up the optimality equations.
- d) Solve the optimality equations to find the right time investment strategy.
6. 2.11) You have to send a set of questionnaires to each of N population segments. The size of each population segment is given by w i . You have a budget of B questionnaires to allocate among the population segments. If you send x i questionnaires to segment i , you will have a sampling error proportional to:

<!-- formula-not-decoded -->

You want to minimize the weighted sum of sampling errors, given by:

<!-- formula-not-decoded -->

You wish to find the allocation x that minimizes F ( x ) subject to the budget constraint ∑ N i =1 x i ≤ R . Set up the optimality equations to solve this problem as a dynamic program (needless to say, we are only interested in integer solutions).

- 2.12) Do the following:
- a) Set up the dynamic programming recursion for this problem. Define your state and decision spaces, and one period reward function.
- b) Show that the optimal betting strategy is to bet a fraction β of his fortune on each play. Find β .

## Chapter 3

## Modeling dynamic programs

Good modeling begins with good notation. Complex problems in asset management require considerable discipline in notation, because they combine the complexities of the original physical problem and to the challenge of modeling sequential information and decision processes. Students will typically find the modeling of time to be particularly subtle. In addition to a desire to model problems accurately, we also need to be able to understand and exploit the structure of the problem, which can become lost in a sea of complex notation.

It is common in textbooks on dynamic programming to quickly adopt a standard paradigm so that the presentation can focus on dynamic programming principles and properties. Our emphasis is on modeling and computation, and our goal is to solve large-scale, complex problems. For this reason, we devote far more attention to modeling than would be found in other dynamic programming texts.

The choice of notation has to balance historical patterns with the needs of a particular problem class. Notation is easier to learn if it is mnemonic (the letters look like what they mean) and compact (avoiding a profusion of symbols). Notation also helps to bridge communities. For example, it is common in dynamic programming to refer to actions using ' a ' (where a is discrete); in control theory a decision (control) is ' u ' (which may be continuous). For high dimensional problems, it is essential to draw on the field of mathematical programming, where decisions are typically written as ' x ' and resource constraints are written in the standard form Ax = b . In this text, many of our problems involve managing assets (resources) where we are trying to maximize or minimize an objective subject to constraints. For this reason, we adopt, as much as possible, the notation of math programming to help us bridge the fields of math programming and dynamic programming.

Proper notation is also essential for easing the transition from simple illustrative problems to the types of real world problems that can arise in practice. In operations research, it is common to refer to an asset class (in finance, it could be a money market fund, real estate or a stock; in the physical world, it could be a type of aircraft or locomotive) as a 'commodity' which might be indexed as k in a set of classes K . But as projects evolve, these asset classes may pick up new dimensions. A common asset management problem in railroads is the movement of boxcars, where there is a clear set of different boxcar types make up

our 'commodities.' Real boxcars, however, have other attributes such as who owns them (so-called 'pools'), the precise configuration of the boxcar (they vary in aspects such as the exact location of the door and locking mechanism, for example), their maintenance status, and their cleanliness. As these attributes are added to the problem, the number of boxcar types grows dramatically.

It is especially important that notation be clear and elegant. Simple, textbook problems are easy. The challenge is modeling complex, realistic problems. If the foundational notation is not properly designed, the modeling of a real problem will explode into a tortuous vocabulary.

## 3.1 Notational style

Notation is a language: the simpler the language, the easier it is to understand the problem. As a start, it is useful to adopt notational conventions to simplify the style of our presentation. For this reason, we adopt the following notational conventions:

- Variables - Variables are always a single letter. We would never use, for example, CH for 'holding cost.'
- Indexing vectors - Vectors are almost always indexed in the subscript, as in x ij . Since we use discrete time models throughout, an activity at time t can be viewed as an element of a vector. So x t would be an activity at time t , with the vector x = ( x 1 , x 2 , . . . , x t , . . . , x T ) giving us all the activities over time. When there are multiple indices, they should be ordered from outside in the general order over which they might be summed (think of the outermost index as the most detailed information). So, if x tij is the flow from i to j at time t with cost c tij , we might sum up the total cost using ∑ t ∑ i ∑ j c tij x tij . Dropping one or more indices creates a vector over the elements of the missing indices to the right. So, x t = ( x tij ) ∀ i, ∀ j is the vector of all flows occuring at time t . If we write x ti , this would be the vector of flows out of i at time t to all destinations j . Time, when present, is always the innermost index.
- Superscripts - These are used to represent different flavors of variables. So, c h (or c hold ) might be a holding cost while c o (or c order ) could be an order cost. Note that while variables must be a single letter, superscripts may be words (although this should be used sparingly). We think of a variable like ' c h ' as a single piece of notation.
- Iteration counters - We always place iteration counters in the superscript, and we primarily use n as our iteration counter. So, x n is our activity at iteration n . If we are using a descriptive superscript, we might write x h,n to represent x h at iteration n . Sometimes algorithms require inner and outer iterations. In this case, we use n to index the outer iteration and m for the inner iteration. While this will prove to be the most natural way to index iterations, students should be aware of the occasional potential for confusion where it may not be clear if the superscript n is an index (as we view it) or raising a variable to the n th power.

- Sets are represented using capital letters in a caligraphic font, such as X , F or I . We generally use the lower case roman letter as an element of a set, as in x ∈ X or i ∈ I .
- Exogenous information - Information that first becomes available (from outside the system) at time t is denoted using hats, for example, ˆ D t or ˆ p t . These are our basic random variables.
- Statistics - Statistics computed using exogenous information are generally indicated using bars, for example ¯ x t or ¯ V t . Since these are functions of random variables, they are also random.

Index variables - Throughout, i, j, k, l, m and n are always scalar indices.

Of course, there are exceptions to every rule. It is extremely common in the transportation literature to model the flow of a type of resource (called a commodity and indexed by k ) from i to j using x k ij . Following our convention, this should be written x kij . Authors need to strike a balance between a standard notational style and existing conventions.

## 3.2 Modeling time

A survey of the literature reveals different styles toward modeling time. When using discrete time, some authors start at one while others start at zero. When solving finite horizon problems, it is popular to index time by the number of time periods remaining, rather than elapsed time. Some authors index a variable, say S t , as being a function of information up through t -1, while others assume it includes information up through time t . t may be used to represent when a physical event actually happens, or when we first know about a physical event.

The confusion over modeling time arises in large part because there are two processes that we have to capture: the flow of information, and the flow of physical and financial assets. There are many applications of dynamic programming to deterministic problems where the flow of information does not exist. Similarly, there are many models where the arrival of the information about a physical asset, and when the information takes affect in the physical system, are the same. For example, the time at which a customer physically arrives to a queue is often modeled as being the same as when the information about the customer first arrives. Similarly, we often assume that we can sell an asset at a market price as soon as the price becomes known.

However, there is a rich collection of problems where the information process and physical process are different. A buyer may purchase an option now (an information event) to buy a commodity in the future (the physical event). Customers may call an airline (the information event) to fly on a future flight (the physical event). An electric power company has to purchase equipment now to be used one or two years in the future. All of these problems represent examples of lagged information processes and force us to explicitly model the informational and physical events.

<!-- image -->

3.1b: Physical processes

Figure 3.1: Relationship between discrete and continuous time for information processes (3.1a) and physical processes (3.1b).

Notation can easily become confused when an author starts by writing down a deterministic model of a physical process, and then adds uncertainty. The problem arises because the proper convention for modeling time for information processes is different than what should be used for physical processes.

We begin by establishing the relationship between discrete and continuous time. All of the models in this book are presented in discrete time, since this is the most natural framework for computation.

The relationship of our discrete time approximation to the real flow of information and physical assets is depicted in figure 3.1. When we are modeling information, time t = 0 is special; it represents 'here and now' with the information that is available at the moment. The discrete time t refers to the time interval from t -1 to t (illustrated in figure 3.1a). This means that the first new information arrives during time interval 1. This notational style means that any variable indexed by t , say S t or x t , is assumed to have access to the information that arrived up to time t , which means up through time interval t . This property will dramatically simplify our notation in the future. For example, assume that f t is our forecast of the demand for electricity. If ˆ D t is the observed demand during time interval t , we would write our updating equation for the forecast using:

<!-- formula-not-decoded -->

We refer to this form as the informational representation . Note that the forecast f t is written as a function of the information that became available during time interval t .

When we are modeling a physical process, it is more natural to adopt a different convention (illustrated in figure 3.1b): discrete time t refers to the time interval between t and t +1. This convention arises because it is most natural in deterministic models to use time to represent when something is happening or when an asset can be used. For example, let

R t be our cash on hand that we can use during day t (implicitly, this means that we are measuring it at the beginning of the day). Let D t be the demand for cash during the day, and let x t represent additional cash that we have decided to add to our balance (to be used during day t ). We can model our cash on hand using the simple equation:

<!-- formula-not-decoded -->

We refer to this form as the actionable representation . Note that the left hand side is indexed by t + 1, while all the quantities on the right hand side are indexed by t . This equation makes perfect sense when we interpret time t to represent when a quantity can be used. For example, many authors would write our forecasting equation (3.1) as:

<!-- formula-not-decoded -->

This equation is correct if we interpret f t as the forecast of the demand that will happen in time interval t .

A review of the literature quickly reveals that both modeling conventions are widely used. Students need to be aware of the two conventions and how to interpret them. We handle the modeling of informational and physical processes by using two time indices, a form that we refer to as the '( t, t ′ )' notation. For example:

ˆ D tt ′ = The demands that first become known during time interval t to be served during time interval t ′ .

f tt ′ = The forecast for activities during time interval t ′ made using the information available up through time t .

R tt ′ = The resources on hand at time t that cannot be used until time t ′ .

x tt ′ = The decision to purchase futures at time t to be exercised during time interval t ′ .

Each of these variables can be written as vectors, such as:

D t = ( D tt ′ ) t ′ ≥ t f t = ( f tt ′ ) t ′ ≥ t

x t = ( x tt ′ ) t ′ ≥ t

R t = ( R tt ′ ) t ′ ≥ t

Note that these vectors are now written in terms of the information content. For stochastic problems, this style is the easiest and most natural.

Each one of these quantities is computed at the end of time interval t (that is, with the information up through time interval t ) and represents a quantity that can be used at time t ′ in the future. We could adopt the convention that the first time index uses the indexing

system illustrated in figure 3.1a, while the second time index uses the system in figure 3.1b. While this convention would allow us to easily move from a natural deterministic model to a natural stochastic model, we suspect most people will struggle with an indexing system where time interval t in the information process refers to time interval t -1 in the physical process. Instead, we adopt the convention to model information in the most natural way, and live with the fact that product arriving at time t can only be used during time interval t +1.

Using this convention it is instructive to interpret the special case where t = t ′ . ˆ D tt is simply demands that arrive during time interval t , where we first learn of them when they arrive. f tt makes no sense, because we would never forecast activities during time interval t after we have this information. R tt represents assets that we know about during time interval t and which can be used during time interval t . Finally, x tt is a decision to purchase assets to be used during time interval t given the information that arrived during time interval t . In financial circles, this is referred to as purchasing on the spot market.

The most difficult notational decision arises when first starting to work on a problem. It is natural at this stage to simplify the problem (often, the problem appears simple) and then choose notation that seems simple and natural. If the problem is deterministic and you are quite sure that you will never solve a stochastic version of the problem, then the actionable representation (figure 3.1b and equation (3.2) is going to be the most natural. Otherwise, it is best to choose the informational format. If you do not have to deal with lagged information processes (e.g. ordering at time t to be used at some time t ′ in the future) you should be able to get by with a single time index, but you need to remember that x t may mean purchasing product to be used during time interval t +1.

As a final observation, consider what happens when we want to know the expected costs given that we make decision x t -1 . We would have to compute E { C t ( x t -1 , ˆ D t ) } , where the expectation is over the random variable ˆ D t . The function that results from taking the expectation is now a function of information up through time t -1. Thus, we might use the notation:

<!-- formula-not-decoded -->

This can take a little getting used to. The costs are incurred during time interval t , but now we are indexing the function with time t -1. The problem is that if we use a single time index, we are not capturing when the activity is actually happening. An alternative is to switch to a double time index, as in:

<!-- formula-not-decoded -->

where ¯ C t -1 ,t ( x t -1 ) is the expected costs that will be incurred during time interval t using the information known at time t -1.

## 3.3 Modeling assets

Many of the models in this book are of fairly complex problems, but we typically start with relatively simple problems. We need notation that allows us to evolve from simple to complex problems in a natural way.

The first issue we deal with is the challenge of modeling single and multiple assets. In engineering, a popular problem is to use dynamic programming to determine how to best land an aircraft, or control a power generating plant. In computer science, researchers in artificial intelligence might want to use a computer to play a game of backgammon or chess. We would refer to these applications as single asset problems. Our 'system' would be the aircraft, and the state variable would describe the position, velocity and acceleration of the aircraft. If we were to model the problem of flying a single aircraft as a dynamic program, we would have considerable difficulty extending this model to simultaneously manage multiple aircraft.

The distinction between modeling a single asset (such as an aircraft) and multiple assets (managing a fleet of aircraft) is important. For this reason, we adopt special notation when we are modeling a single asset. For example, it is quite common when modeling a dynamic program to use a variable such as S t to represent the 'state' of our system, where S t could be the attributes describing a single aircraft, or all the attributes of a fleet of aircraft. Unfortunately, using such general notation disguises the structure of the problem and significantly complicates the challenge of designing effective computational algorithms.

For this reason, if we are managing a single asset, we adopt special notation for the attributes of the asset. We let:

.

a t = Vector of attributes of the asset at time t A = Set of possible attributes. .

The attribute a t can be a single element or a vector, but we will always assume that the vector is not too big (no more than 10 or 20 elements). In the case of our shortest path problem, a t would be the node number corresponding to the intersection where a driver had to make a decision. If we are solving an asset selling problem, a t might capture whether the asset has been sold, and possibly how long we have held it. For a college student planning her course of study, a t would be a vector describing the number of courses she has taken to fulfill each requirement.

There is a vast array of problems that involving modeling what we would call a single asset. If there is no interest in extending the model to handle multiple assets, then it may be more natural to use S t as the state variable. Students need to realize, however, that this notational framework can be quite limiting, as we show over the course of this chapter.

If we are modeling multiple assets, we would capture the resource state of our system by defining the resource vector:

R ta = The number of assets with attribute a at time t .

<!-- formula-not-decoded -->

R t is a vector with |A| dimensions. If a is a vector (think of our college student planning her course work), then |A| may be quite large. The size of |A| will have a major impact on the algorithmic strategy.

We often have to model random arrivals of assets over time. For this purpose, we might define:

ˆ R t = Vector of new arrivals to the system during time period t .

ˆ R t may be the withdrawals from a mutual fund during time interval t (a single type of asset), or the number of requests for a particular car model (multiple asset classes), or the number of aircraft an airline has ordered where each aircraft is characterized by a vector of attributes a . When we are representing ˆ R t mathematically, we assume it takes a set of outcomes in a set that is always denoted Ω (don't ask), with elements ω ∈ Ω. Using this notation, ˆ R t is a random variable giving the number of new arrivals of each type of asset class, and ˆ R t ( ω ) is a specific sample realization.

New information may be more than just a new arrival to the system. An aircraft flying from one location to the next may arrive with some sort of maintenance failure. This can be modeled as a random change in an attribute of the aircraft. We can model this type of new information by defining:

ˆ R ta ( R t ) = The change in the number of assets with attribute a due to exogenous information.

ˆ R t ( R t ) = The information function capturing exogeneous changes to the resource vector.

Here, ˆ R t ( R t ) is a function, where a sample realization would be written ˆ R t ( R t , ω ).

There are many settings where the information about a new arrival comes before the new arrival itself as illustrated in the examples.

Example 3.1: An airline may order an aircraft at time t and expect the order to be filled at time t ′ .

Example 3.2: An orange juice products company may purchase futures for frozen concentrated orange juice at time t that can be exercised at time t ′ .

Example 3.3: A programmer may start working on a piece of coding at time t with the expectation that it will be finished at time t ′ .

This concept is important enough that we offer the following term:

Definition 3.3.1 The actionable time of an asset is the time at which a decision may be used to change its attributes (typically generating a cost or reward).

The actionable time is simply one attribute of an asset. For example, if at time t we own a set of futures purchased in the past with exercise dates of t +1 , t +2 , . . . , t ′ , then the exercise date would be an attribute of each futures contract. When writing out a mathematical model, it is sometimes useful to introduce an index just for the actionable time (rather than having it buried as an element of the attribute vector a ). Before, we let R ta be the number of resources that we know about at time t with attribute a . The attribute might capture that the resource is not actionable until time t ′ in the future. If we need to represent this explicitly, we might write:

R t,t ′ a = The number of resources that we know about at time t that will be actionable with attribute a at time t ′ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Perhaps the most powerful aspect of our notation is the attribute vector a , which allows us to represent a broad range of problems using a single variable. In fact, there are six major problem classes that can be represented using the same notation:

- Basic asset acquisition problems -a = {} . Here R t is a scalar, often representing how much money (or quantity of a single type of asset) that is on hand.
- Multiclass asset management -a = { k } where k ∈ K is a set of asset classes. The attribute a consists of a single, static element.
- Single commodity flow problems -a = { i } where i ∈ I is a set of states of an asset. Examples include managing money that can be invested in different stocks and a fleet of identical transport containers whose only attributesare their current location.
- Multicommodity flow -a = { k, i } where k ∈ K represents the asset class and i ∈ I is a set of locations or states.
- Heterogeneous resource allocation problem -a = ( a 1 , . . . , a n ). Here we have an n -dimensional attribute vector. These applications arise primarily in the management of people and complex equipment.
- Multilayered resource scheduling problem -a = { a c 1 | a c 2 | · · · | a c n } . Now the attribute vector is a concatenation of attribute vectors.

The only class that we do not specifically address in this volume is the last one.

## 3.4 Illustration: the nomadic trucker

The 'nomadic trucker' is a colorful illustration of a multiattribute resource which helps to illustrate some of the modeling conventions being introduced in this chapter. Later, we use this example to illustrate different issues that arise in approximate dynamic programming, leading up to the solution of large-scale asset management problems later in the book.

The problem of the nomadic trucker arises in what is known as the truckload trucking industry. In this industry, a truck driver works much like a taxicab. A shipper will call a truckload motor carrier and ask it to send over a truck. The driver arrives, loads up the shipper's freight and takes it to the destination where it is unloaded. The shipper pays for the entire truck, so the carrier is not allowed to consolidate the shipment with freight from other shippers. In this sense, the trucker works much like a taxicab for people. However, as we will soon see, our context of the trucking company adds an additional level of richness that offers some relevant lessons for dynamic programming.

Our trucker runs around the United States, where we assume that his location consists of which of the 48 contiguous states he is located in. When he arrives in a state, he sees the customer demands for loads to move from that state to other states. There may be none, one, or several. He may choose a load to move if one is available; alternatively, he has the option of doing nothing or moving empty to another state (even if a load is available). Once he moves out of a state, all other customer demands (in the form of loads to be moved) are assumed to be picked up by other truckers and are therefore lost. He is not able to see the availability of loads out of states other than where he is located.

Although truckload motor carriers can boast fleets of over 10,000 drivers, our model focuses on the decisions made by a single driver. There are, in fact, thousands of trucking 'companies' that consist of a single driver. It is also the case that a driver in a large fleet still has some flexibility over what loads he accepts and where he moves. The problem of dispatching drivers has often been described as a negotiation, implying that drivers retain some independence in how they are assigned. In chapter 15 we show that the concepts we develop here form the foundation for managing the largest and most complex versions of this problem. For now, our 'nomadic trucker' represents a particularly effective way of illustrating some important concepts in dynamic programming.

## 3.4.1 A basic model

The simplest model of our nomadic trucker assumes that his only attribute is his location, which we assume has to be one of the 48 contiguous states. We let:

I = The set of 'states' (locations) that the driver can be located at.

We use i and j to index elements of I . His attribute vector then consists of:

<!-- formula-not-decoded -->

In addition to the attributes of the driver, we also have to capture the attributes of the loads that are available to be moved. For our basic model, loads are characterized only by where they are going. Let:

b = The vector of characteristics of a load.

B = The space of possible load attributes.

For our basic problem, the load attribute vector consists of:

<!-- formula-not-decoded -->

The set B is the set of all pairs of origins and destinations.

## 3.4.2 A more realistic model

We need a richer set of attributes to capture some of the realism of the real life of a truck driver. A driver's behavior is determined in the United States by a set of rules set by the Department of Transportation ('DOT') that limit how much a driver can drive so he does not become too tired. There are three basic limits: the amount a driver can be behind the wheel in one shift, the amount of time a driver can be 'on duty' in one shift, and the amount of time that a driver can be on duty over any contiguous eight day period. These rules were revised effective in year 2004 to be as follows: a driver can drive at most 11 hours, he may be on duty for at most 14 continuous hours (there are exceptions to this rule), and the driver can work at most 70 hours in any eight day period. The last clock is reset if the driver is off-duty for 34 successive hours during any stretch (known as the '34 hour reset'). If we add these three variables, our attribute vector grows to:

<!-- formula-not-decoded -->

was on duty over each of the previous eight days.

We emphasize that element a 4 is actually a vector that holds the number of hours the driver was on duty during each calendar day over the last eight days.

The three attributes that capture the DOT rules affect our ability to assign drivers to certain loads. A load may have the attribute that it must be delivered by a certain time. If a driver is about to hit some limit, then he will not be able to drive as many hours as he would otherwise be able to do. We may assign a driver to a load even if we know he cannot make the delivery appointment in time, but we would need to assess a penalty.

















A particularly important attribute for a driver is the one that represents his home 'domicile.' This would be represented as a geographical location, although it may be represented at a different level of detail than the driver's location (stored in a 1 ). We also need to keep track of how many days that our driver has been away from home:

<!-- formula-not-decoded -->

            The location of the driver The number of hours a driver has been behind the wheel during his current shift The number of hours that a driver has been on-duty during his current shift. An eight element vector giving the number of hours the driver was on duty over each of the previous eight days. The geographical location giving the driver's home domicile. The number of days that the driver has been away from home.            

It is typically the case that drivers like to be home over a weekend. If we cannot get him home on one weekend, we might want to work on getting him home on the next weekend. This adds a very rich set of behaviors to the management of our driver. We might think that it is best to get our driver close to home, but it does not help to get him close to home in the middle of a week. In addition, there may be a location that is fairly far away, but that is known for generating a large number of loads that would take a driver near or to his home.

## 3.4.3 The state of the system

We can now use two different methods for describing the state of our driver. The first is his attribute vector a at time t . If our only interest was optimizing the behavior of a single driver, we would probably let S t be the state of the driver, although this state would be nothing more than his vector of attributes. Alternatively, we can use our resource vector notation, which allows us to scale to problems with multiple drivers:

<!-- formula-not-decoded -->

In the same way, we can represent the state of all the loads to be moved using:

R L = The number of loads with attribute b .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, our complete resource vector is given by:

<!-- formula-not-decoded -->

## 3.5 The exogenous information process

An important dimension of many of the problems that we address is the arrival of exogenous information, which changes the state of our system. While there are many important deterministic dynamic programs, exogenous information processes represent an important dimension in many problems in asset management.

## 3.5.1 Basic notation for information processes

We have already seen one example of an exogenous information process in the variable ˆ R t . We can use this notation to represent customer demands, new equipment arriving at the company or new drivers being hired (as long as these are not decisions that we are modeling). There are, however, other forms of exogenous information: interest rates, prices, travel times, costs, equipment breakdowns, people quitting, and so on. To write down a complete model, we would need to introduce notation for each class of exogenous information. It is standard to let:

- ω = A realization of all the information arriving over all time periods.
- = ( ω 1 , ω 2 , . . . , ω t , . . . ), where: ω t = The information arriving during time interval t . Ω = The set of all possible sample realizations (with ω ∈ Ω).

ω is an actual sample realization of information from the set Ω. ω is sometimes referred to as a sample path or a 'scenario.' It is important to recognize, however, that ω is not a random variable. It is not meaningful, for example, to take the expected value of a function f ( ω ), since ω is viewed as a single, deterministic realization of the information. The mathematical representation of information that is not yet known requires the introduction of a function. Surprisingly, while ω t is fairly standard notation for a sample realization, there is not standard notation for a generic random variable to represent unknown information. Different authors use I t , ξ t , ˆ ω t or, fairly commonly, ω t itself. Students need to understand the different subcommunities that work in this field. Probabilists will object to writing E [ f ( ω )] (the expectation of f ( ω )), while others will claim that this is perfectly clear.

Whenever we have a random variable (for example, ˆ D or ˆ p ) we refer to a sample realization of the random variable by ˆ D ( ω ) or ˆ p ( ω ). It is important to recognize that ˆ D and ˆ p are random variables, whereas ˆ D ( ω ) and ˆ p ( ω ) are numbers.

When we need a generic random variable, we suggest using:

W t = The exogenous information becoming available during interval t .

The choice of notation W t as a generic 'information function' is not standard, but it is mnemonic (it looks like ω t ). We would then write ω t = W t ( ω ) as a sample realization. This

notation adds a certain elegance when we need to write decision functions and information in the same equation.

Generic information variables, such as W t , should be used to simplify notation, which means they are useful when there are different forms of exogenous information. If there is only one form of exogenous information (say, ˆ R t ), then that variable should be used as the information variable.

We also need to refer to the history of our process, for which we define:

H t = The history of the process, consisting of all the information known through time t .

<!-- formula-not-decoded -->

H t = The set of all possible histories through time t .

= { H t ( ω ) | ω ∈ Ω } .

h t = A sample realization of a history,

= H t ( ω ).

We sometimes need to refer to the subset of Ω that corresponds to a particular history. The following is useful for this purpose:

<!-- formula-not-decoded -->

## 3.5.2 Models of information processes

Information processes come in varying degrees of complexity. Needless to say, the structure of the information process plays a major role in the models and algorithms used to solve the problem. Below, we describe information processes in increasing levels of complexity.

## Processes with independent increments

A large number of problems in asset management can be characterized by what are known as processes with independent increments. What this means is that the change in the process is independent of the history of the process, as illustrated in the examples.

The practical challenge we typically face in these applications is that we do not know the parameters of the system. In our price process, the price may be trending upward or downward, as determined by the parameter µ . In our customer arrival process, we need to know the rate λ (which can also be a function of time).

Example 3.4: A publicly traded index fund has a price process that can be described (in discrete time) as p t +1 = p t + σδ , where δ is normally distributed with mean µ , variance 1, and σ is the standard deviation of the change over the length of the time interval.

Example 3.5: Requests for credit card confirmations arrive according to a Poisson process with rate λ . This means that the number of arrivals during a period of length ∆ t is given by a Poisson distribution with mean λ ∆ t , which is independent of the history of the system.

## State-dependent information processes

The standard dynamic programming models require that the distribution of the process moving forward be a function of the state of the system. This is a more general model than one with independent increments. Interestingly, many models of Markov decision processes use information processes that do, in fact, exhibit independent increments. For example, we may have a queueing problem where the state of the system is the number of customers in the queue. The number of arrivals may be Poisson, and the number of customers served in an increment of time is determined primarily by the length of the queue. It is possible, however, that our arrival process is a function of the length of the queue itself (see the examples for illustrations).

Example 3.6: Customers arrive at an automated teller machine according to a Poisson process, but as the line grows longer, an increasing proportion decline to join the queue (a property known as balking in the queueing literature). The apparent arrival rate at the queue is a process that depends on the length of the queue.

Example 3.7: A market with limited information may respond to price changes. If the price drops over the course of a day, the market may interpret the change as a downward movement, increasing sales and putting further downward pressure on the price. Conversely, upward movement may be interpreted as a signal that people are buying the stock, encouraging more buying behavior.

State-dependent information processes are more difficult to model and introduce additional parameters that must be estimated. However, from the perspective of dynamic programming, they do not introduce any fundamental complexities. As long as the distribution of outcomes is dependent purely on the state of the system, we can apply our standard models.

It is also possible that the information arriving to the system depends on its state, as depicted in the next set of examples.

This is a different form of state-dependent information process. Normally, an outcome ω is assumed to represent all information available to the system. A probabilist would

Example 3.8: A driver is planning a path over a transportation network. When the driver arrives at intersection i of the network, he is able to determine the transit times of each of the segments ( i, j ) emanating from i . Thus, the transit times that arrive to the system depend on the path taken by the driver.

insist that this is still the case with our driver; the fact that the driver does not know the transit times on all the links is simply a matter of modeling the information the driver uses. However, most engineering students will find it more natural to think of the information as depending on the state.

## More complex information processes

Now consider the problem of modeling currency exchange rates. The change in the exchange rate between one pair of currencies is usually followed quickly by changes in others. If the Japanese yen rises relative to the U.S. dollar, it is likely that the Euro will also rise relative to it, although not necessarily proportionally. As a result, we have a vector of information processes that are correlated.

In addition to correlations between information processes, we can also have correlations over time. An upward push in the exchange rate between two currencies in one day is likely to be followed by similar changes for several days while the market responds to new information. Sometimes the changes reflect long term problems in a country's economy. Such processes may be modeled using advanced statistical models which capture correlations between processes as well as over time.

An information model can be thought of as a probability density function φ t ( ω t ) that gives the density (we would say the probability of ω if it were discrete) of an outcome ω t in time t . If the problem has independent increments, we would write the density simply as φ t ( ω t ). If the information process is Markovian (dependent on a state variable), then we would write it as φ t ( ω t | S t -1 ). If the state variable requires information from history (for example, our 'state variable' is the history ( W t -1 , W t -2 , . . . , W t -T )), then we have a 'history dependent' model.

In some cases with complex information models, it is possible to proceed without any model at all. Instead, we can use realizations drawn from history. For example, we may take samples of changes in exchange rates from different periods in history and assume that these are representative of changes that may happen in the future. The value of using samples from history is that they capture all of the properties of the real system. This is an example of planning a system without a model of an information process.

## 3.6 The states of our system

We have now established the notation we need to talk about the most important quantity in a dynamic program: its state . The state variable is perhaps the most critical piece of modeling that we will encounter when solving a dynamic program. Surprisingly, other treatments of dynamic programming spend little time defining a state variable. Bellman's seminal text [Bellman (1957), p. 81] says '... we have a physical system characterized at any stage by a small set of parameters, the state variables .' In a much more modern treatment, Puterman first introduces a state variable by saying [Puterman (1994), p. 18] 'At each decision epoch, the system occupies a state .' In both cases, the italics are in the original manuscript, indicating that the term 'state' is being introduced. In effect, both authors are saying that given a system, the state variable will be apparent from the context. In this section, we take this topic a bit further.

We open our discussion with a presentation on three perspectives of a state variable, after which we discuss where the state variable is measured.

## 3.6.1 The three states of our system

Interestingly, standard dynamic programming texts do not offer a formal definition of a state, presuming instead that it is readily apparent from an application. For many problems, this is true. Our interest, however, is in more complex problems (which require computational solutions), and in these settings, state variables become more subtle.

To set up our discussion, assume that we are interested in solving a relatively complex asset management problem, one that involves multiple (possibly many) different types of assets which can be modified in various ways (changing their attributes). For such a problem, it is necessary to work with three types of states:

- The state of a single resource As a resource evolves, the state of a resource is captured by its attribute vector a .
- The resource state vector This is the state of all the different types of resources at the same time, given by R t .
- The information state This captures what we know at time t , which includes R t along with estimates of parameters such as prices, times and costs, or the parameters of a function for forecasting demands (or prices or times).

We have already introduced the attribute vector a for the state of an asset. Consider the problem of routing a single asset (such as an aircraft, a locomotive, a pilot, or a truck driver) over time (possibly, but not necessarily, in the presence of uncertainty). We could let a t be the attribute vector of the asset at time t . In this setting, a t is the state of our asset.

If we have more than one asset, then R t becomes the joint state of all our assets at time t . The dimensionality of R t is potentially equal to the dimensionality of A . If our asset has

no attributes (for example, we are only interested with acquiring and selling a single type of asset), then |A| = 1. In some problems |A| can be quite large, which means that R t can be a very high dimensional vector.

It is common in some subcommunities to use S t as a 'state variable.' We suggest using S t as a generic state variable when it is not important to be specific, and in particular when we may wish to include other forms of information. Typically, the other information represents what we know about various parameters of the system (costs, speeds, times, prices). To represent this, let:

¯ θ t = A vector of estimates of different problem parameters at time t . ˆ θ t = New information about problem parameters that arrive during time interval t .

We can think of ¯ θ t as the state of our information about different problem parameters at time t . We can now write a more general form of our state variable as:

.

```
S t = Our information state at time t = ( R t , ¯ θ t ) .
```

Remark: In one of his earliest papers, Bellman struggled with the challenge of representing both the resource state and other types of information, which he did using the notation ( x, I ) where x was his resource state variable and I represented other types of information. The need to differentiate between the resource state and 'other information' indicates the equivalence in his mind (and those of many authors to follow) between the 'state of the system' and the resource state. The most notable exception to this view is the study of information acquisition problems. The best-known examples of this problem class are the bandit problems discussed in chapter 10, where the state is an estimate of a parameter.

It is important to have a formal definition of a state variable. For this purpose, we offer:

Definition 3.6.1 A state variable is the minimally dimensioned function of history that is necessary and sufficient to model all future dynamics of the system.

We use the term 'minimally dimensioned function' so that our state variable is as compact as possible. For example, we could argue that the history h t is the information we need to model future dynamics (if we know h t , then we have all the information that has come to us during the simulation). But this is not practical. As we start doing computational work, we are going to want S t to be as compact as possible. Furthermore, there are many problems where we simply do not need to know the entire history. It might be enough to know the status of all our assets at time t (the resource variable R t ). But there are examples where this is not enough.

Assume, for example, that we need to use our history to forecast the price of a stock. Our history of prices is given by (ˆ p 1 , ˆ p 2 , . . . , ˆ p t ). If we use a simple exponential smoothing model, our estimate of the mean price ¯ p t can be computed using:

<!-- formula-not-decoded -->

where α is a stepsize satisfying 0 ≤ α ≤ 1. With this forecasting mechanism, we do not need to retain the history of prices, but rather only the latest estimate ¯ p t . As a result, ¯ p t is called a sufficient statistic , which is a statistic that captures all relevant information needed to compute any additional statistics from new information. A state variable, according to our definition, is always a sufficient statistic.

Consider what happens when we switch from exponential smoothing to an N -period moving average. Our forecast of future prices is now given by:

<!-- formula-not-decoded -->

Now, we have to retain the N -period rolling set of prices (ˆ p t , ˆ p t -1 , . . . , ˆ p t -N +1 ) in order to compute the price estimate in the next time period. With exponential smoothing, we could write:

<!-- formula-not-decoded -->

If we use the moving average, our state variable would be:

<!-- formula-not-decoded -->

Students need to realize that many authors say that if we use the moving average model, we no longer have a proper state variable. Rather, we would have an example of a 'historydependent process' where the state variable needs to be augmented with history. Using our definition of a state variable, the concept of a history-dependent process has no meaning. The state variable is simply the minimal information required to capture what is needed to model future dynamics. State variables differ only in their dimensionality. Needless to say, having to explicitly retain history, as we did with the moving average model, produces a much larger state variable than the exponential smoothing model.

The state variable is critical to the success of dynamic programming as a practical, computational tool. The higher the dimensionality of S t , the more parameters we are going to have to estimate. One problem characteristic that can have a major impact on the design of a state variable is the probabilistic structure of the information process. The simplest process occurs when the information variables W 1 , W 2 , . . . , W t are independent (the process may be nonstationary, but independence is really nice) or is conditionally independent given the

state variable. For example, we might decide that the price of an asset fluctuates randomly up or down from the previous period's price. This means that the probability distribution of ˆ p t +1 depends only on ˆ p t . When our random information is customer demands, we often find that we can assume independence. When the information process is interest rates, on the other hand, it can easily be the case that the process is characterized by a fairly complex underlying structure. In this case, we may find that we get good results by assuming that the price in t +1 depends on only a few periods of history.

## 3.6.2 Pre- and post-decision state variables

We can view our system as evolving through sequences of new information followed by a decision followed by new information (and so on). Although we have not yet discussed decisions, for the moment let the decisions (which will often be vectors) be represented generically using x t (we discuss our choice of notation for a decision in the next section). In this case, a history of the process might be represented using:

<!-- formula-not-decoded -->

h t contains all the information we need to make a decision d t at time t . As we discussed before, h t is sufficient but not necessary. We expect our state variable to capture what is needed to make a decision, allowing us to represent the history as:

<!-- formula-not-decoded -->

The sequence in equation (3.6) defines our state variable as occurring after new information arrives and before a decision is made. For this reason, we call S t the pre-decision state variable . This is the most natural place to write a state variable because the point of capturing information from the past is to make a decision.

For most problem classes, we can design more effective computational strategies using the post-decision state variable. This is the state of the system after a decision x t . For this reason, we denote this state variable S x t , which produces the history:

<!-- formula-not-decoded -->

We again emphasize that our notation S x t means that this function has access to all the exogenous information up through time t , along with the decision x t (which also has access to the information up through time t ).

Interestingly, virtually every text on stochastic, dynamic programming assumes that the state variable is the pre-decision state variable. The optimality recursion relates the predecision state S t +1 to S t , requiring that we model both the decision x t that is made after observing S t as well as the information W t +1 that arrives during time interval t +1.

Figure 3.2: Decision tree showing decision nodes (pre-decision state variable) and outcome nodes (post-decision state variable).

<!-- image -->

By contrast, it has always been the tradition in the decision-theory literature to model both the pre- and post-decision states when representing decision trees. Figure 3.2 shows the decision tree for a classic problem from the decision-theory literature which addresses the problem of whether we should collect information about the weather to determine if we should hold a Little League game. Squares represent nodes where decisions have to be made: Should we check the weather report? Should we schedule the game? Circles represent outcome nodes: What does the weather report say? What will the weather actually be? Decision nodes represent the pre-decision state of the system. Outcome nodes are the state of the system just before new information arrives, which is the same as immediately after a decision is made.

Unless otherwise specified (not just in this volume, but throughout the dynamic programming/control communities), a state variable is the pre-decision state variable. There are specific problem classes in asset management that really need the post-decision state variable. In this case, it is more convenient notationally to simply define the state variable as the post-decision state variable, which allows us to drop the ' x ' superscript. The examples provide a series of illustrations.

As we progress, we will see that the choice of using pre- versus post-decision state variables is governed by problem complexity and our ability to accurately approximate the future without sacrificing computational tractability. The vast majority of the dynamic programming literature uses the pre-decision state variable. Most authors do not even distinguish

Example 3.9: If we are selling an asset, the pre-decision state variable can be written as S t = ( R t , p t ) where R t = 1 if we are holding the asset and 0 otherwise, while p t is the price that we can sell the asset at if we are still holding it. The post-decision variable S x t = R x t simply captures whether we are still holding the asset or not.

Example 3.10: The nomadic trucker revisited. Let R ta = 1 if the trucker has attribute a at time t and 0 otherwise. Now let L tb be the number of loads of type b available to be moved at time t . The pre-decision state variable for the trucker is S t = ( R t , L t ), which tells us the state of the trucker and the loads available to be moved. Assume that once the trucker makes a decision, all the loads in L t are lost, and new loads become available at time t + 1. The post-decision state variable is given by S x t = R x t where R x ta = 1 if the trucker has attribute a after a decision has been made.

Example 3.11: Imagine playing backgammon where R ti is the number of your pieces on the i th 'point' on the backgammon board (there are 24 points on a board). Let d be the decision to move from one point to another, where the set of potential decisions D t depends on the roll of the dice during the t th play. x td is the number of pieces the player moves from one point to another, where x t = ( x tad ) d ∈D t . The state of our board when we make a decision is given by S t = ( R t , D i ). The transition from S t to S t +1 depends on the player's decision x t , the play of the opposing player, and the next roll of the dice. The post-decision state variable is simply R x t which is the state of the board after a player moves.

between whether they are using a pre- or post-decision state variable. Students can easily identify which is being used: if the expectation is within the max or min operator, then the formulation is using the pre-decision state vector.

## 3.6.3 Partially observable states

There is a subfield of dynamic programming that is referred to as partially observable Markov decision processes where we cannot measure the state exactly, as illustrated in the examples.

Example 3.12: A retailer may have to order inventory without being able to measure the precise current inventory. It is possible to measure sales, but theft and breakage introduce errors.

Example 3.13: A transportation company needs to dispatch a fleet of trucks, but does not know the precise location or maintenance status of each truck.

Example 3.14: The military has to make decisions about sending out aircraft to remove important military targets that may have been damaged in previous raids. These decisions typically have to be made without knowing the precise state of the targets.

Markov decision processes with partially observable states provides a nice framework for modeling systems that can not be measured precisely. This is an important subfield of Markov decision processes, but is outside the scope of our presentation.

It is tempting to confuse a post-decision state variable as a pre-decision state variable that can only be measured imperfectly. This ignores the fact that we can measure the postdecision state variable perfectly, and we can formulate a version of the optimality equations that determine the value function. In addition, post-decision state variables are often simpler than pre-decision state variables.

## 3.7 Modeling decisions

Fundamental to dynamic programs is the characteristic that we are making decisions over time. For stochastic problems, we have to model the sequencing of decisions and information, but there are many uses of dynamic programming that address deterministic problems. In this case, we use dynamic programming because it offers specific structural advantages, such as our budgeting problem in chapter 1. But the concept of sequencing decisions over time is fundamental to a dynamic program.

It is important to model decisions properly so we can scale to high dimensional problems. This requires that we start with a good fundamental model of decisions for asset management problems. Our choices are nonstandard for the dynamic programming community, but very compatible with the math programming community.

## 3.7.1 Decisions, actions, and controls

A survey of the literature reveals a distressing variety of words used to mean 'decisions.' The classical literature on Markov decision process talks about choosing an action a ∈ A (or a ∈ A s where A s is the set of actions available when we are in state s ). The optimal control community works to choose a control u ∈ U x when the system is in state x . The math programming community wants to choose a decision represented by the vector x , while the Markov decision community wants to choose a policy and the simulation community wants to apply a rule.

Our interest is primarily in solving large-scale asset allocation problems, and for this purpose we must draw on the skills of the math programming community where decisions are typically vectors represented by x . Most of our examples focus on decisions that act on assets (buying them, selling them, or managing them within the system). For this reason,

we define:

- d = A type of decision that acts on an asset (or asset type) in some way (buying, selling, or managing).
- D a = The set of potential types of decisions that can be used to act on a resource with attribute a .
- x tad = The quantity of resources with attribute a acted on with decision d at time t .
- x t = ( x tad ) a ∈A ,d ∈D a .
- X t = Set of acceptable decisions given the information available at time t .

If d is a decision to purchase an asset, then x d is the quantity of assets being purchased. If we are moving transportation assets from one location i to another location j , then d would represent the decision to move from i to j , and x tad would be the flow of resources.

Earlier, we observed that the attribute of a single asset a might have as many as 10 or 20 dimensions, but we would never expect the attribute vector to have 100 or more dimensions (problems involving financial assets, for example, might require 0, 1 or 2 dimensions). Similarly, the set of decision types, D a , might be on the order of 100 or 1000 for the most complex problems, but we simply would never expect sets with, say, 10 10 types of decisions (that can be used to act on a single asset class). Note that there is a vast array of problems where the size of D a is less than 10.

Example 3.15: Assume that you are holding an asset, where you will receive a price ˆ p t if you sell at time t . Here there is only one type of decision (whether or not to sell). We can represent the decision as x t = 1 if we sell, and x t = 0 if we hold.

Example 3.16: A taxi waiting at location i can serve a customer if one arrives, sit and do nothing or reposition to another location (without a customer) where the chances of finding one seem better. We can let D M be the set of locations the cab can move to (without a customer), where the decision d = i represents the decision to hold (at location i ). Then let d s be the decision to serve a customer, although this decision can only be made if there is a customer to be served. The complete set of decisions is D = d s ∪ D M . x d = 1 if we choose to take decision d .

It is significant that we are representing a decision d as acting on a single asset or asset type. The field of Markov decision processes represents a decision as an action (typically denoted a ), but the concept of an action in this setting is equivalent to our vector x . Actions are typically represented as being discrete, whereas our decision vector x can be discrete or continuous. We do, however, restrict our attention to cases where the set D is discrete and finite.

In some problem classes, we manage a discrete asset which might be someone playing a game, the routing of a single car through traffic, or the control of a single elevator moving up

/negationslash and down in a building. In this case, at any point in time we face the problem of choosing a single decision d ∈ D . Using our x notation, we would represent this using x ta ˆ d = 1 if we choose decision ˆ d and x tad = 0 for d = ˆ d . Alternatively, we could simply drop our ' x ' notation and simply let d t be the decision we chose at time t . While recognizing that the ' d ' notation is perhaps more natural and elegant if we face simply the scalar problem of choosing a single decision, it greatly complicates our transition from simple, scalar problems to the complex, high-dimensional problems.

Our notation represents a difficult choice between the vocabularies of the math programming community (which dominates the field of high-dimensional asset management problems) and the control community (which dominates the field of approximate dynamic programming). Since some in the control community use a for action instead of u for control, primarily to exploit results in the field of Markov decision processes, it made more sense to choose notation that was natural and mnemonic (and did not conflict with our critical notation a for the attribute of an asset).

## 3.7.2 The nomadic trucker revisited

We return to our nomadic trucker example to review the decisions for this application. There are two classes of decisions the trucker may choose from:

- D l = The decision to move a load with attribute vector b ∈ B .
- b D l = ( D l b ) b ∈B D e i = The decision to move empty to location i ∈ I . D e = ( D e i ) i ∈I D = D l ∪ D e

The trucker may move 'empty' to the same state that he is located in, which represents the decision to do nothing. The set D l requires a little explanation. Recall that b is the attribute vector of a load to be moved. An element of D l represents the decision to move a type of load. Other decision classes that could be modeled including buying and selling trucks, repairing them, or reconfiguring them (for example, adding refrigeration units so a trailer can carry perishable commodities).

## 3.7.3 Decision epochs

Most of our presentation in this book adopts a discrete time format. Information arrives during time interval t (between t -1 and t ), and decisions are made at time t . We typically view the times t = (1 , 2 , . . . , ) as equally spaced points in time. But there are settings where decisions are determined by exogenous events. We may have to decide whether to admit a patient to a hospital for elective surgery. The decision has to be made when the patient calls

in. We may have to decide when to sell a thinly traded stock. Such a decision is naturally made when the stock changes price.

The points in time when a decision has to be made (even if the decision is to do nothing) are referred to as decision epochs. Decision epochs may occur at evenly spaced points in time or may be determined by exogenous information events. If they are determined by information events, we might define a set E with element e ∈ E . Now let t e be the time that information event e occurs (for example, the e th phone call). Instead of indexing time by t = (1 , 2 , . . . , ), we may index time by ( t 1 , t 2 , . . . , t e , ).

## 3.7.4 Policies

When we are solving deterministic problems, our interest is in finding a set of decisions x t over time. When we are solve stochastic problems (problems with dynamic information processes), the decision x t for t ≥ 1 is a random variable. This happens because we do not know (at time t = 0) the state of our system S t at time t .

How do we make a decision if we do not know the state of the system? The answer is that instead of finding the best decision, we are going to focus on finding the best rule for making a decision given the information available at the time. This rule is commonly known as a policy :

Definition 3.7.1 A policy is a rule that determines a decision given the available information.

This definition implies that our policy produces a decision deterministically; that is, given a state S t , it produces a single action x . There are, however, instances where S t does not contain all the information needed to make a decision (for example, our post-decision state variable S x t ). In addition, there are special situations (arising in the context of two-player games) where there is value in choosing a decision somewhat randomly. For our computational algorithms, there will be many instances when we want to choose what appears to be a non-optimal decision for the purpose of collecting more information.

Policies come in many forms (each with their own notation). Perhaps the most common form in introductory treatments of dynamic programming is to assume that the policy is of the 'table lookup' variety. That is, given a discrete state S t , our policy can be viewed as a simple rule of the form 'if we are in state S t we should make decision x t .' Although different authors use different notation, it is most common to represent such a rule as a policy π ∈ Π where Π is our set of possible policies (rules) from which we have to choose. In this version, the set Π is viewed as consisting of a set of discrete policies which are typically finite.

For high dimensional problems, we virtually never use policies of the table-lookup variety. Instead, these are functions that must be solved to produce a decision. For this reason, we use the notation:

X π t = A function returning a decision vector x , where ( X π t ) π ∈ Π is the family of functions from which we have to choose.

Often, a policy is determined by choosing a particular function for making a decision and then tuning the parameters of the function (which could easily be continuous variables). In this setting, the set of potential policies is infinite.

To illustrate, consider our budget problem from chapter 1. There, we were making decisions by solving problems of the form:

<!-- formula-not-decoded -->

where 'arg max' means 'find the value of x t (the argument) that maximizes the expression that follows.' Here, ¯ V π might be a particular value function (think of it as an approximation of the real value function). This type of approximation means that we have to estimate ¯ V t +1 ( R t +1 ) for each possible (discrete) value of R t +1 . If R t +1 is a vector (as would arise if we are managing different types of assets), this strategy means that we may have to estimate a very large number of parameters. We can simplify this problem by replacing our discrete value function with a linear approximation ¯ v t +1 R t +1 . Now we wish to solve:

<!-- formula-not-decoded -->

Our policy now consists of choosing a single parameter ¯ v t +1 .

This example illustrates two policies: one that requires us to specify the value of being in each state (at least approximately) and one that only requires us to come up with a single slope. These are two classes of policies, which we might denote by Π discrete and Π linear . Each class contains an infinite set of parameters over which we have to search.

## 3.7.5 Randomized policies

Assume you need to buy an asset at an auction, and you do not have the time to attend the auction yourself. Your problem is to decide which of your two assistants to send. Assistant A is young and aggressive, and is more likely to bid a higher price (but may also scare off other bidders). Assistant B is more tentative and conservative, and might drop out if he thinks the bidding is heading too high.

This is an example of a randomized policy. We are not directly making a decision of what to bid, but we are making a decisiont that will influence the probability distribution of whether a bid will be made. This is known as a randomized policy.

In section 4.5.4, we show that given a choice between a deterministic policy and a randomized policy, the deterministic policy will always be at least as good as a randomized policy. But there are situations where we may not have a choice. In addition, there are

situations involving two-player games where a deterministic policy allows the other player to predict your response and obtain a better result.

## 3.8 Information processes, revisited

The introduction of decisions and policies requires that we revisit our model of the information process. We are going to want to compute quantities such as expected profits, but we cannot find an expectation using only the probability of different outcomes of the exogenous information. We also have to know something about how decisions are generated.

## 3.8.1 Combining states and decisions

With our vocabulary for policies in hand, we need to take a fresh look at our information process. The sequence of information ( ω 1 , ω 2 , . . . , ω t ) is assumed to be driven by some sort of exogenous process. However, we are generally interested in quantities that are functions of both exogenous information as well as the decisions. It is useful to think of decisions as endogenous information . But where do the decisions come from? We now see that decisions come from policies. In fact, it is useful to represent our sequence of information and decisions as:

<!-- formula-not-decoded -->

Now our history is characterized by a family of functions: the information variables W t , the decision functions (policies) X π t , and the state variables S t . We see that to characterize a particular history h t , we have to specify both the sample outcome ω as well as the policy π . Thus, we might write a sample realization as:

<!-- formula-not-decoded -->

We can think of a complete history H π ∞ ( ω ) as an outcome in an expanded probability space (if we have a finite horizon, we would denote this by H π T ( ω )). Let:

<!-- formula-not-decoded -->

be an outcome in our expanded space, where ω π is determined by ω and the policy π . Let Ω π be the set of all outcomes of this expanded space. The probability of an outcome in Ω π obviously depends on the policy we are following. Thus, computing expectations (for example, expected costs or rewards) requires knowing the policy as well as the set of exogenous outcomes. For this reason, if we are interested, say, in the expected costs during time period t , some authors will write E π t { C t ( S t , x t ) } to express the dependence of the expectation on the policy. However, even if we do not explicitly index the policy, it is important to understand that we need to know how we are making decisions if we are going to compute expectations or other quantities.

## 3.8.2 Supervisory processes

In many instances, we are trying to control systems that are already controlled by some process, often a human. Now, we have two sets of decisions: X π t ( S t ) made by our mathematical model and the decisions that are made by human operators. The examples (below) provides an illustration.

Example 3.17: An asset management problem in the printing industry involves the assignment of printing jobs to printing machines. An optimization model may assign a print job to one printing plant, while a knowledgeable planner may insist that the job should be assigned to a different plant. The planner may know that this particular job requires skills that only exist at the other plant.

Example 3.18: A military planner may know that it is best to send a cargo aircraft on a longer path because this will take it near an airbase where tankers can fly up and refuel the plane. Without this information, it may be quite hard for an algorithm to discover this strategy.

Example 3.19: An expert chess player may know that a sequence of steps produces a powerful defensive position.

When we combine machines and people, we actually create two decision processes: what the machine recommends and what the human implements. Since these 'supervisory' decisions are exogenous (even though they have access to the machine-generated decision), we might let ˆ x t be the supervisory decisions (which we assume override those of the machine). One of the opportunities in machine learning is to use the sequence of decisions ˆ x t to derive patterns to guide the model.

## 3.9 Modeling system dynamics

We begin our discussion of system dynamics by introducing some general mathematical notation. While useful, this generic notation does not provide much guidance into how specific problems should be modeled. We then describe how to model the dynamics of some simple problems, followed by a more general model for complex assets.

## 3.9.1 A general model

The dynamics of our system is represented by a function that describes how the state evolves as new information arrives and decisions are made. The dynamics of a system can be

represented in different ways. The easiest is through a simple function that works as follows:

<!-- formula-not-decoded -->

The function S M ( · ) goes by different names such as 'plant model' (literally, the model of a physical production plant), 'plant equation,' 'law of motion,' 'transfer function,' 'system dynamics,' 'system model,' 'transition law,' and 'transition function.' We prefer 'transition function' because it is the most descriptive. We choose the notation S M ( · ) to reflect that this is the state transition function, which represents a model of the dynamics of the system. Below, we reinforce the ' M ' superscript with other modeling devices.

The arguments of the function follows standard notational conventions in the control literature (state, action, information), but students will find that different authors will follow one of two conventions for modeling time. While equation (3.11) is fairly common, many authors will write the recursion as:

<!-- formula-not-decoded -->

If we use the form in equation (3.12), we would say 'the state of the system at the beginning of time interval t + 1 is determined by the state at time t , plus the decision that is made at time t and the information that arrives during time interval t .' In this representation, t indexes when we are using the information. We refer to (3.12) as the actionable representation since it captures when we can act on the information. This representation is always used for deterministic models, and many authors adopt it for stochastic models as well. We prefer the form in equation (3.11) where time t indexes the information content of the variable or function. We refer to this style as the informational representation .

In equation (3.11), we have written the function assuming that the function does not depend on time (it does depend on data that depends on time). A common notational error is to write a function, say, f t ( S t , x t ) as if it depends on time, when in fact the function is stationary, but depends on data that depends on time. If the parameters (or structure) of the function depends on time, then we would use S t M ( S t , x t , W t +1 ) (or possibly S t M +1 ( S t , x t , W t +1 )). If not, the transition function should be written S M ( S t , x t , W t +1 ).

This is a very general way of representing the dynamics of a system. In many problems, the information W t +1 arriving during time interval t +1 depends on the state S t at the end of time interval t , but is conditionally independent of all prior history given S t . When this is the case, we say that we have a Markov information process. When the decisions also depend only on the state S t , then we have a Markov decision process. In this case, we can store the system dynamics in the form of a one-step transition matrix as follows:

p ( s ′ | s, x ) = The probability that S t +1 = s ′ given S t = s and X π t = x .

P ( x ) = Matrix of elements where p ( s ′ | s, x ) is the element in row s and column s ′ .

There is a simple relationship between the transition function and the one-step transition matrix. Let:

<!-- formula-not-decoded -->

The one-step transition matrix can be computed using

<!-- formula-not-decoded -->

It is common in the field of Markov decision processes to assume that the one-step transition is given as data. Often, it can be quickly derived (for simple problems) using assumptions about the underlying process. For example, consider an asset selling problem with state variable S t = ( R t , p t ) where:

<!-- formula-not-decoded -->

and where p t is the price at time t . We assume the price process is described by:

<!-- formula-not-decoded -->

where /epsilon1 t is a random variable with distribution:

<!-- formula-not-decoded -->

Assume the prices are integer and range from 1 to 100. We can number our states from 0 to 100 using:

<!-- formula-not-decoded -->

Now assume that we adopt a decision rule for selling of the form:

<!-- formula-not-decoded -->

Assume that ¯ p = 60. A portion of the one-step transition matrix for the rows and columns corresponding to the state (0 , -) and (1 , 58) , (1 , 59) , (1 , 60) , (1 , 61) , (1 , 62) looks like:

<!-- formula-not-decoded -->

This matrix plays a major role in the theory of Markov decision processes, although its value is more limited in practical applications. By representing the system dynamics as a one-step transition matrix, it is possible to exploit the rich theory surrounding matrices in general and Markov chains in particular.

In engineering problems, it is far more natural to develop the transition function first. Given this, it may be possible to compute the one-step transition matrix exactly or estimate it using simulation. The techniques in this book do not, in general, use the one-step transition matrix, but use instead the transition function directly. But formulations based on the transition matrix provide a powerful foundation for proving convergence of both exact and approximate algorithms.

## 3.9.2 System dynamics for simple assets

It is useful to get a feel of the system dynamics by considering some simple applications.

## Asset acquisition I - Purchasing assets for immediate use

Let R t be the quantity of a single asset class we have available at the end of a time period, but before we have acquired new assets (for the following time period). The asset may be money available to spend on an election campaign, or the amount of oil, coal, grain or other commodities available to satisfy a market. Let ˆ D t be the demand for the resource that occurs over time interval t , and let x t be the quantity of the resource that is acquired at time t to be used during time interval t +1. The transition function would be written:

<!-- formula-not-decoded -->

## Asset acquisition II: purchasing futures

Now assume that we are purchasing futures at time t to be exercised at time t ′ . At the end of time period t , we would let R tt ′ be the number of futures we are holding that can be

exercised during time period t ′ . Now assume that we purchase x tt ′ additional futures to be used during time period t ′ . Our system dynamics would look like:

<!-- formula-not-decoded -->

In many problems, we can purchase assets on the spot market, which means we are allowed to see the actual demand before we make the decision. This decision would be represented by x t +1 ,t +1 , which means the amount purchased using the information that arrived during time interval t +1 to be used during time interval t +1 (of course, these decisions are usually the most expensive). In this case, the dynamics would be written:

<!-- formula-not-decoded -->

## Planning a path through college

Consider a student trying to satisfy a set of course requirements (for example, number of science courses, language courses, departmentals, and so on). Let R tc be the number of courses taken that satisfy requirement c at the end of semester t . Let x tc be the number of courses the student enrolled in at the end of semester t for semester t + 1 to satisfy requirement c . Finally let ˆ F tc ( x t -1 ) be the number of courses in which the student received a failing grade during semester t given x t -1 . This information depends on x t -1 since a student cannot fail a course that she was not enrolled in. The system dynamics would look like:

<!-- formula-not-decoded -->

## 3.9.3 System dynamics for complex assets

We adopt special notation when we are modeling the dynamics for problems with multiple asset classes. This notation is especially useful for complex assets which are represented using the attribute vector a . Whereas above we modeled the dynamics using the system state variable, with complex assets it is more natural to model the dynamics at the level of individual asset classes.

Assume we have resources with attribute a , and we act on them with decision d . The result may be a resource with a modified set of attributes a ′ . In general, the decision will generate a contribution (or a cost) and will require some time to complete. This process is modeled using a device called the modify function :

<!-- formula-not-decoded -->

Here, we are acting on an asset with attribute vector a with decision d using the information available at time t , producing an asset with attribute vector a ′ , generating cost (or contribution) c , and requiring time τ to complete.

The modify function is basically our transition function at the level of an individual asset (or asset class) and a single decision acting on that asset. For many problems in the general area of asset management, this modeling strategy will seem more natural. However, it introduces a subtle discrepancy with the more classical transition function notation of equation (3.11) which includes an explicit dependence on the information W t +1 that arrives in the next time interval. As we progress through increasingly more complex models in this volume, we will need to model different assumptions about the information required by the modify function. A wide range of problems can be modeled as one of the three cases:

- Information known at time t -M ( t, a, d ). The result of a decision (e.g. the attribute vector a ′ ) is completely determined given the information at time t . For example, an aircraft with attribute a (which specifies its location among other things), sent to a particular city, will arrive with attribute a ′ which might be a deterministic function of a and d and the information available at time t when the decision is made.
- Information known at time t +1 M ( t, a, d, W t +1 ). The modify function depends on information that becomes available in the time period after decision d is made (which uses the information available at time t ). For example, a funding agency may invest in a new technology, where a characterizes what we know about the technology. After the research is funded, we learn the outcome of the research (say, in the following year) which is unknown at the time the decision is made to fund the research.
- Information known at time t + τ -M ( t, a, d, W t + τ ). The modify function depends on information that becomes available at the end of an action, at time t + τ , where τ itself may be a random variable. Returning to our aircraft, the attribute vector may include elements describing the maintenance status of the aircraft and the time of arrival . The flight time may be random, and we will not learn about the mechanical status of the aircraft until it lands (at time t + τ ).

In the latter two cases, the argument t refers to when the decision is made (and hence the information content of the decision), but the additional argument W t +1 or W t + τ tells us the information we need to compute the outcome of the decision.

It is sometimes convenient to refer to the attribute vector a ′ using a function, so we define:

<!-- formula-not-decoded -->

We use the superscript ' M ' to emphasize the relationship with the modify function (students may also think of this as the 'model' of the physical process). The argument t indicates the information content of the function, which is to say that we can compute the function using information that is available up through time interval t . Normally, when we make a

decision to act on an asset at time t , the transition function can use the information in the full state variable S t (whereas a t is the state of the asset we are acting on), so we could write a M ( S t , a, d ) (or a M ( S t , d )).

Example 3.20: The attributes of a taxicab can be described by its location, fuel level and how many hours the driver has been on duty. If the cab takes a customer to location j , it changes location, burns fuel and adds more hours to the time the driver has been on duty.

Example 3.21: A student progressing through college can be described by the course requirements she has completed. The decision d represents the courses she decides to take, where she may drop (or fail) a course. a ′ = a M ( t, a, d ) describes her academic progress at the end of the next semester.

Our modify function brings out a common property of many asset management problems: an action can take more than one time period to complete. If τ &gt; 1, then at time t +1, we know that there will be an asset available at time t ′ = t + τ in the future. This means that to capture the state of the system at time t + 1, we need to recognize that an important attribute is when the asset can be used.

For algebraic purposes, it is also useful to define the indicator function:

<!-- formula-not-decoded -->

In addition to the attributes of the modified resource, we sometimes have to capture the fact that we may gain or lose resources in the process of completing a decision. We define:

γ ( t, a, d ) = The multiplier giving the quantity of resources with attribute a available after being acted on with decision d at time t .

The multiplier may depend on the information available at time t , but is often random and depends on information that has not yet arrived. Illustrations of gains and losses are given in the next set of examples.

Using our modify function and gain, we can now provide a specific set of equations to capture the evolution of our resource vector. Remembering that R tt ′ represents the resources we know about at time t (now) that are actionable at time t ′ ≥ t , we assume that we can only act on resources that are actionable now. So, for t ′ &gt; t , the evolution of the resource vector is given by:

<!-- formula-not-decoded -->

Example 3.22: A corporation is holding money in an index fund with a 180 day holding period (money moved out of this fund within the period incurs a four percent load) and would like to transfer them into a high yield junk bond fund. The attribute of the asset would be a = (AssetType , Age). There is a transaction cost (the cost of executing the trade) and a gain γ , which is 1.0 for funds held more than 180 days, and 0.96 for funds held less than 180 days.

Example 3.23: Transportation of liquified natural gas - A company would like to purchase 500,000 tons of liquified natural gas in southeast Asia for consumption in North America. Although in liquified form, the gas evaporates at a rate of 0.2 percent per day, implying γ = . 998.

Equation (3.16) can be read as follows: the number of resources with attribute a ′ (that are actionable at time t ′ ) that we know about at time t + 1 is the sum of i) the number of resources with attribute a ′ with actionable time t ′ that we know about at time t , plus ii) the number of resources that are actionable now that will become actionable (due to our decisions) at time t ′ with attribute a ′ , plus iii) the number of resources with attribute a ′ that are actionable at time t ′ that we first learn about during time interval t +1.

A more compact form can be written if we view the actionable time t ′ as a part of the attribute a ′ . Assume that we 'act' on any resource that is not actionable by 'doing nothing.' In this case, we can write (3.16) as:

<!-- formula-not-decoded -->

Equation (3.16) can be written in matrix form:

<!-- formula-not-decoded -->

or more simply:

<!-- formula-not-decoded -->

It is often useful to have a compact functional representation for the resource dynamics. For this reason, we introduce the notation:

R

M

(

R

t

, x

t

, ω

)

=

∆

t

(

ω

)

x

t

+ ˆ

R

t

+1

(

ω

)

The superscript ' M ' indicates that this is really just the modify function in vector form. We are implicitly assuming that our decision x t is derived from a deterministic function of the state of the system, although this is not always the case. If the only source of randomness

is new arrivals, then it is going to be most common that R M ( R t , x t , ω ) will depend on information that arrives during time interval t + 1. However, there are many applications where the function ∆ depends on information that arrives in later time periods.

## 3.10 The contribution function

Next we need to specify the contribution (or cost if we are minimizing) produced by the decisions we make in each time period. If we use a pre-decision state variable, and we are at time t trying to make decision x t , we would represent our contribution function using:

ˆ C t +1 ( S t , x t , W t +1 ) = Contribution at time t from being in state S t , making decision x t and then receiving the information W t +1 .

When we make the decision x t , we do not know W t +1 , so it is common to use:

<!-- formula-not-decoded -->

The role that W t +1 plays is problem dependent, as illustrated in the examples below.

There are many asset allocation problems where the contribution of a decision can be written using:

c tad = The unit contribution of acting on an asset with attribute a with decision d .

This contribution is incurred in period t using information available in period t . In this case, our total contribution at time t could be written:

<!-- formula-not-decoded -->

In general, when we use a pre-decision state variable, it is best to think of C t ( S t , x t ) as an expectation of a function that may depend on future information. Students simply need to be aware that in some settings, the contribution function does not depend on future information.

It is surprisingly common for us to want to work with two contributions. The common view of a contribution function is that it contains revenues and costs that we want to maximize or minimize. In many operational problems, there can be a mixture of 'hard dollars' and 'soft dollars.' The hard dollars are our quantifiable revenues and costs. But there are often other issues that are important in an operational setting, but which cannot always be easily quantified. For example, if we cannot cover all of the demand, we may wish to assess a penalty for not satisfying it. We can then manipulate this penalty to reduce the amount

Example 3.24: In asset acquisition problems, we order x t in time period t to be used to satisfy demands ˆ D t +1 in the next time period. Our state variable is S t = R t = the product on hand after demands in period t have been satisfied. We pay a cost c p x t in period t and receive a revenue p min( R t + x t , ˆ D t +1 ) in period t + 1. Our total one-period contribution function is then:

<!-- formula-not-decoded -->

The expected contribution is:

<!-- formula-not-decoded -->

Example 3.25: Now consider the same asset acquisition problem, but this time we place our orders in period t to satisfy the known demand in period t . Our cost function contains both a fixed cost c f (which we pay for placing an order of any size) and a variable cost c p . The cost function would look like:

<!-- formula-not-decoded -->

Note that our contribution function no longer contains information from the next time period. If we did not incur a fixed cost c f , then we would simply look at the demand D t and order the quantity needed to cover demand (as a result, there would never be any product left over). However, since we incur a fixed cost c f with each order, there is a benefit to ordering enough to cover the demand now and future demands. This benefit is captured through the value function.

of unsatisfied demand. Examples of the use of soft-dollar bonuses and penalties abound in operational problems (see examples).

Given the presence of these so-called 'soft dollars,' it is useful to think of two contribution functions. We can let C t ( S t , x t ) be the hard dollars and C π t ( S t , x t ) be the contribution function with the soft dollars included. The notation captures the fact that a set of soft bonuses and penalties represents a form of policy. So we can think of our policy as making decisions that maximize C π t ( S t , x t ), but measure the value of the policy (in hard dollars), using C t ( S t , X π ( S t )).

## 3.11 The objective function

We are now ready to write out our objective function. Let X π t ( S t ) be a decision function (equivalent to a policy) that determines what decision we make given that we are in state S t .

Example 3.26: A trucking company has to pay the cost of a driver to move a load, but wants to avoid using inexperienced drivers for their high priority accounts (but has to accept the fact that it is sometimes necessary). An artificial penalty can be used to reduce the number of times this happens.

Example 3.27: A charter jet company requires that in order for a pilot to land at night, he/she has to have landed a plane at night three times in the last 60 days. If the third time a pilot landed at night is at least 50 days ago, the company wants to encourage assignments of these pilots to flights with night landings so that they can maintain their status. A bonus can be assigned to encourage these assignments.

Example 3.28: A student planning her schedule of courses has to face the possibility of failing a course, which may require taking either an extra course one semester or a summer course. She wants to plan out her course schedule as a dynamic program, but use a penalty to reduce the likelihood of having to take an additional course.

Example 3.29: An investment banker wants to plan a strategy to maximize the value of an asset and minimize the likelihood of a very poor return. She is willing to accept lower overall returns in order to achieve this goal and can do it by adding an additional penalty when the asset is sold at a significant loss.

Our optimization problem is to choose the best policy by choosing the best decision function from the family ( X π t ( S t )) π ∈ Π . We are going to measure the total return from a policy as the (discounted) total contribution over a finite (or infinite) horizon. This would be written as:

<!-- formula-not-decoded -->

where γ discounts the money into time t = 0 values. In some communities, it is common to use an interest rate r , in which case the discount factor is:

<!-- formula-not-decoded -->

Important variants of this objective function are the infinite horizon problem ( T = ∞ )and the finite horizon problem ( γ = 1). A separate problem class is the average reward, infinite horizon problem:

<!-- formula-not-decoded -->

Our optimization problem is to choose the best policy. In most practical applications,

we can write the optimization problem as one of choosing the best policy, or:

<!-- formula-not-decoded -->

Often (and this is generally the case in our discussions) a policy is characterized by a continuous parameter. It might be that the optimal policy corresponds to a value of the parameter equal to infinity. It is possible that F ∗ 0 exists, but that an optimal 'policy' does not exist (because it requires finding a parameter equal to infinity). While this is more of a mathematical curiosity, we handle these situations by writing the optimization problem as:

<!-- formula-not-decoded -->

where ' sup ' is the supremum operator, which finds the smallest number greater than or equal to F π 0 for any value of π . If we were minimizing, we would use 'inf,' which stands for 'infimum,' which is the largest value less than or equal to the value of any policy. It is common in more formal treatments to use 'sup' instead of 'max' or 'inf' instead of 'min' since these are more general. Our emphasis is on computation and approximation, where we consider only problems where a solution exists. For this reason, we use 'max' and 'min' throughout our presentation.

The expression (3.20) contains one important but subtle assumption that will prove to be critical later and which will limit the applicability of our techniques in some problem classes. Specifically, we assume the presence of what is known as linear, additive utility . That is, we have added up contributions for each time period. It does not matter if the contributions are discounted or if the contribution functions themselves are nonlinear. However, we will not be able to handle functions that look like:

<!-- formula-not-decoded -->

The assumption of linear, additive utility means that the total contribution is a separable function of the contributions in each time period. While this works for many problems, it certainly does not work for all of them, as depicted in the examples below.

In some cases these apparent instances of violations of linear, additive utility can be solved using a creatively defined state variable.

## 3.12 Models for a single, discrete asset

With our entire modeling framework behind us, it is useful to contrast two strategies for modeling a single (discrete) asset. The asset may be yourself (planning a path through

Example 3.30: We may value a policy of managing an asset using a nonlinear function of the number of times the price of an asset dropped below a certain amount.

Example 3.31: Assume we have to find the route through a network where the traveler is trying to arrive at a particular point in time. The value function is a nonlinear function of the total lateness, which means that the value function is not a separable function of the delay on each link.

Example 3.32: Consider a mutual fund manager who has to decide how much to allocate between aggressive stocks, conservative stocks, bonds, and money market instruments. Let the allocation of assets among these alternatives represent a policy π . The mutual fund manager wants to maximize long term return, but needs to be sensitive to short term swings (the risk). He can absorb occasional downswings, but wants to avoid sustained downswings over several time periods. Thus, his value function must consider not only his return in a given time period, but also how his return looks over one year, three year and five year periods.

college, or driving to a destination), a piece of equipment (an electric power generating plant, an aircraft or a locomotive), or a financial asset (where you have to decide to buy, sell or hold). There are two formulations that we can use to model a single asset problem. Each illustrates a different modeling strategy and leads to a different algorithmic strategy. We are particularly interested in developing a modeling strategy that allows us to naturally progress from managing a single asset to multiple assets.

## 3.12.1 A single asset formulation

Assume we have a single, discrete asset. Just before we act on an asset, new information arrives that determines the contribution we will receive. The problem can be modeled using:

a t = The attribute vector of the asset at time t .

D a = The set of decisions that can be used to act on the resource with attribute a .

- a M ( t, a t , d t ) = The terminal attribute function, which gives the attribute of a resource with attribute a t after being acted on with decision d t at time t .
- W t +1 = New information that arrives during time period t + 1 that is used to determine the contribution generated by the asset.

a

C

t

+1

(

t

, d

t

, W

t

+1

)

= The contribution returned by acting on a resource with attribute a t with decision d ∈ D a given the information that becomes available during time interval t +1.

When we manage a single asset, there are two ways to represent the decisions and the state of the system. The first is geared purely to the modeling of a single, discrete asset:

The resource state variable: a

t

.

The system state variable: S t = ( a t , W t ).

The decision variable: d ∈ D a .

The transition function: a t +1 = a M ( t, a, d ) (or a t +1 = a M ( t, a, d, W t +1 )

Recall that a M ( t, a, d ) is the terminal attribute function , which gives the attributes of a resource after it has been acted on. This is an output of the modify function (see equation (3.14) in section 3.9). Here, we are assuming that the outcome of the modify function is known in the next time period. There are many models where the modify function is deterministic given the information at time t . By contrast, there are many applications where the outcome is not known until time t + τ , where τ itself may be random.

This is the conventional formulation used in dynamic programming, although we would customarily let the attribute vector a t be the state S t of our system (the single resource), which can be acted on by a set of actions a ∈ A . The optimality equations for this problem are easily stated as:

<!-- formula-not-decoded -->

Given the value function V t +1 ( a ), equation (3.25) is solved by simply computing the total contribution for each d ∈ D and choosing the best decision.

There is a vast array of dynamic programming problems where the single asset is the system we are trying to optimize. For these problems, instead of talking about the attributes of the asset we would simply describe the state of the system.

In our discussion, we will occasionally use this model. The reader should keep in mind that in this formulation, we assume that the attribute vector is small enough that we can usually enumerate the complete attribute space A . Furthermore, we also assume that the decision set D is also small enough that we can enumerate all the decisions.

## 3.12.2 A multiple asset formulation for a single asset

The second formulation is mathematically equivalent, but uses the same notation that we use for more complex problems:

The resource state variable: R t .

The system state variable: S t = ( R t , W t ).

The decision variable: x t = ( x tad ) a ∈A ,d ∈D a .

The transition function: R t +1 ,a ′ = ∑ a ∈A ∑ d ∈D a δ a ′ ( t, a, d ) x tad

.

Since we have a single, discrete asset, ∑ a ∈A R ta = 1. Our optimality equation becomes:

<!-- formula-not-decoded -->

Our feasible region X t is given by the set:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we assume that the set D a includes a 'do nothing' option.

It might seem as though the optimality equation (3.26) is much harder to solve because we are now choosing a vector x t ∈ X rather than a scalar decision d ∈ D . Of course, because of equation (3.27), we are really facing an identical problem. Since ∑ a ∈A R t +1 ,a = 1, we can rewrite V t +1 ( R t +1 ) using:

<!-- formula-not-decoded -->

where v t +1 ,a ′ = V t +1 ( R t +1 ) if R t +1 ,a = 1. v t +1 ,a ′ is the value of having an asset with attribute a ′ at time t +1. Since:

<!-- formula-not-decoded -->

we can rewrite (3.29) as

<!-- formula-not-decoded -->

since by definition,

a

M

(

t, a, d

) =

∑

a

∈A

∑

∈D

d

We can also simplify our contribution function by taking advantage of the fact that we have to choose exactly one decision:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

δ

a

′

(

t, a, d

)

x

tad

Combining (3.26) with (3.31) gives:

<!-- formula-not-decoded -->

Equation (3.34) makes it apparent that we are doing the same thing that we did in (3.25). We have to compute the total contribution of each decision and choose the best.

The first is the most natural model for a single discrete asset, but extending it to multiple assets is extremely awkward. The second model is no harder to solve, but forms the basis for solving far larger problems (including those with multiple assets).

## 3.13 A measure-theoretic view of information**

For students interested in proving theorems or reading more theoretical research articles, it is useful to have a more fundamental understanding of information.

When we work with random information processes and uncertainty, it is standard in the probability community to define a probability space, which consists of three elements. The first is the set of outcomes Ω, which is generally assumed to represent all possible outcomes of the information process (actually, Ω can include outcomes that can never happen). If these outcomes are discrete, then all we would need is the probability of each outcome p ( ω ).

It is nice to have a terminology that allows for continuous quantities. We want to define the probabilities of our events, but if ω is continuous, we cannot talk about the probability of an outcome ω . However we can talk about a set of outcomes E that represent some specific event (if our information is a price, the event E could be all the prices that constitute the event that the price is greater than some number). In this case, we can define the probability of an outcome E by integrating the density function p ( ω ) over all ω in the event E .

Probabilists handle continuous outcomes by defining a set of events F , which is literally a 'set of sets' because each element in F is itself a set of outcomes in Ω. This is the reason we resort to the script font F as opposed to our calligraphic font for sets; students may find it easy to read E as 'calligraphic E' and F as 'script F.' The set F has the property that if an event E is in F , then its complement Ω \ E is in F , and the union of any two events E X ∪E Y in F is also in F . F is called a 'sigma-algebra' (which may be written σ -algebra). An understanding of sigma-algebras is not important for computational work, but can be useful in certain types of proofs, as we see in this volume. Sigma-algebras are without question one of the more arcane devices used by the probability community, but once they are mastered, they are a powerful theoretical tool.

Finally, it is required that we specify a probability measure denoted P , which gives the

probability (or density) of an outcome ω which can then be used to compute the probability of an event in F .

We can now define a formal probability space for our exogenous information process as (Ω , F , P ). If we wish to take an expectation of some quantity that depends on the information, say Ef ( W t ), then we would sum (or integrate) over the set ω multiplied by the probability (or density) P .

It is important to emphasize that ω represents all the information that will become available, over all time periods. As a rule, we are solving a problem at time t , which means we do not have the information that will become available after time t . To handle this, we let F t be the sigma-algebra representing events that can be created using only the information up to time t . To illustrate, consider an information process W t consisting of a single 0 or 1 in each time period. W t may be the information that a customer purchases a jet aircraft, or the event that an expensive component in an electrical network fails. If we look over three time periods, there are eight possible scenarios, as shown in table 3.1.

Table 3.1: Set of demand outcomes

| Outcome   |   Time |   Time |   period |
|-----------|--------|--------|----------|
| ω         |      1 |      2 |        3 |
| 1         |      0 |      0 |        0 |
| 2         |      0 |      0 |        1 |
| 3         |      0 |      1 |        0 |
| 4         |      0 |      1 |        1 |
| 5         |      1 |      0 |        0 |
| 6         |      1 |      0 |        1 |
| 7         |      1 |      1 |        0 |
| 8         |      1 |      1 |        1 |

Let E { W 1 } be the set of outcomes ω that satisfy some logical condition on W 1 . If we are at time t = 1, we only see W 1 . The event W 1 = 0 would be written

<!-- formula-not-decoded -->

The sigma-algebra F 1 would consist of the events

<!-- formula-not-decoded -->

Now assume that we are in time period 2 and have access to W 1 and W 2 . With this information, we are able to divide our outcomes Ω into finer subsets. Our history H 2 consists of the elementary events H 2 = { (0 , 0) , (0 , 1) , (1 , 0) , (1 , 1) } . Let h 2 = (0 , 1) be an element of H 2 . The event E { h 2 =(0 , 1) } = { 3 , 4 } . In time period 1, we could not tell the difference between outcomes 1, 2, 3 and 4; now that we are at time 2, we can differentiate between ω ∈ (1 , 2) and ω ∈ (3 , 4). The sigma-algebra F 2 consists of all the events E h 2 , h 2 ∈ H 2 , along with all possible unions and complements.

Another event in F 2 is the { ω | ( W 1 , W 2 ) = (0 , 0) } = { 1 , 2 } . A third event in F 2 is the union of these two events, which consists of ω = { 1 , 2 , 3 , 4 } which, of course, is one of the

events in F 1 . In fact, every event in F 1 is an event in F 2 , but not the other way around, the reason being that the additional information from the second time period allows us to divide Ω into a finer set of subsets. Since F 2 consists of all unions (and complements), we can always take the union of events, which is the same as ignoring a piece of information. By contrast, we cannot divide F 1 into a finer subsets. The extra information in F 2 allows us to filter Ω into a finer set of subsets than was possible when we only had the information through the first time period. If we are in time period 3, F will consist of each of the individual elements in Ω as well as all the unions needed to create the same events in F 2 and F 1 .

From this example, we see that more information (that is, the ability to see more elements of W 1 , W 2 , . . . ) allows us to divide Ω into finer-grained subsets. We see that F t -1 ⊆ F t . F t always consists of every event in F t -1 in addition to other finer events. As a result of this property, F t is termed a filtration . It is because of this interpretation that the sigma-algebras are typically represented using the letter F (which literally stands for filtration) rather the more natural letter H (which stands for history). The fancy font used to denote a sigmaalgebra is used to designate that it is a set of sets (rather than just a set).

It is always assumed that information processes satisfy F t -1 ⊆ F t . Interestingly, this is not always the case in practice. The property that information forms a filtration requires that we never 'forget' anything. In real applications, this is not always true. Assume, for example, that we are doing forecasting using a moving average. This means that our forecast f t might be written as f t = (1 /T ) ∑ T t ′ =1 ˆ D t -t ′ . Such a forecasting process 'forgets' information that is older than T time periods.

## 3.14 Bibliographic notes

Most textbooks on dynamic programming give very little emphasis on modeling. The multiattribute notation for multiple asset classes is based primarily on Powell et al. (2001). Figure 3.1 which describes the mapping from continuous to discrete time was outlined for me by Erhan Cinlar.

## Exercises

- 3.1) A college student must plan what courses she takes over each of eight semesters. To graduate, she needs 34 total courses, while taking no more than five and no less than three courses in any semester. She also needs two language courses, one science course, eight departmental courses in her major and two math courses.
- a) Formulate the state variable for this problem in the most compact way possible.
- 3.2) Assume that we have N discrete assets to manage, where R i is the number of assets of type a ∈ A and N = ∑ a ∈A R a . Let R be the set of possible values of

the vector R . Show that:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is the number of combinations of X items taken Y at a time.

- b) Give the transition function for our college student assuming that she successfully passes any course she takes. You will need to introduce variables representing her decisions.
- c) Now give the transition function for our college student, but now allow for the random outcome that she may not pass every course.
3. 3.3) A broker is working in thinly traded stocks. He must make sure that he does not buy or sell in quantities that would move the price and he feels that if he works in quantities that are no more than 10 percent of the average sales volume, he should be safe. He tracks the average sales volume of a particular stock over time. Let ˆ v t be the sales volume on day t , and assume that he estimates the average demand f t using f t = (1 -α ) f t -1 + α ˆ v t . He then uses f t as his estimate of the sales volume for the next day. Assuming he started tracking demands on day t = 1, what information would constitute his state variable?
4. 3.4) How would your previous answer change if our broker used a 10-day moving average to estimate his demand? That is, he would use f t = 0 . 10 ∑ 10 i =1 ˆ v t -i +1 as his estimate of the demand.
5. 3.5) The pharmaceutical industry spends millions managing a sales force to push the industry's latest and greatest drugs. Assume one of these salesmen must move between a set I of customers in his district. He decides which customer to visit next only after he completes a visit. For this exercise, assume that his decision does not depend on his prior history of visits (that is, he may return to a customer he has visited previously). Let S n be his state immediately after completing his n th visit that day.
- a) Assume that it takes exactly one time period to get from any customer to any other customer. Write out the definition of a state variable, and argue that his state is only his current location.
- b) Now assume that τ ij is the (deterministic and integer) time required to move from location i to location j . What is the state of our salesman at any time t ? Be sure to consider both the possibility that he is at a location (having just finished with a customer) or between locations.
- c) Finally assume that the travel time τ ij follows a discrete uniform distribution between a ij and b ij (where a ij and b ij are integers?

- 3.6) Consider a simple asset acquisition problem where x t is the quantity purchased at the end of time period t to be used during time interval t +1. Let D t be the demand for the assets during time interval t . Let R t be the pre-decision state variable (the amount on hand before you have ordered x t ) and R x t be the post-decision state variable.
- a) Write the transition function so that R t +1 is a function of R t , x t and D t +1 .
- b) Write the transition function so that R x t is a function of R x t -1 , D t and x t .
- c) Write R x t as a function of R t , and write R t +1 as a function of R x t .
- 3.7) As a buyer for an orange juice products company, you are responsible for buying futures for frozen concentrate. Let x tt ′ be the number of futures you purchase in year t that can be exercised during year t ′ .
- a) What is your state variable in year t ?
- b) Write out the transition function.
- 3.8) A classical inventory problem works as follows. Assume that our state variable R t is the amount of product on hand at the end of time period t and that D t is a random variable giving the demand during time interval ( t -1 , t ) with distribution p d = Prob ( D t = d ). The demand in time interval t must be satisfied with the product on hand at the beginning of the period. We can then order a quantity x t at the end of period t that can be used to replenish the inventory in period t + 1. Give the transition function that relates R t +1 to R t .
- 3.9) Many problems involve the movement of assets over networks. The definition of the state of the single asset, however, can be complicated by different assumptions for the probability distribution for the time required to traverse a link. For each example below, give the state of the asset:
- a) You have a deterministic, static network, and you want to find the shortest path from an origin node r to a destination node s. There is a known cost cij for traversing each link ( i, j ).
- b) Each day, you need to choose between one of two paths from home to work, but you do not know the travel time for each path because it is random (but the mean and variance of the distribution of travel times remains the same from one day to the next). Each time you follow a path, you get more information about the travel time over that path. You need to devise a strategy for determining which path to choose each day.
- c) A taxicab is moving people in a set of cities C . After dropping a passenger off at city i , the dispatcher may have to decide to reposition the cab from i to j , ( i, j ) ∈ C . The travel time from i to j is τ ij , which is a random variable with a discrete uniform distribution (that is, the probability that τ ij = t is 1 /T , for t = 1 , 2 , . . . , T ). Assume that the travel time is known before the trip starts.
- d) Same as (c), but now the travel times are random with a geometric distribution (that is, the probability that τ ij = t is (1 -θ ) θ t -1 , for t = 1 , 2 , 3 , . . . ).

- 3.10) In the figure below, a sailboat is making its way upwind from point A to point B. To do this, the sailboat must tack, whereby it sails generally at a 45 degree angle to the wind. The problem is that the angle of the wind tends to shift randomly over time. The boats skipper decides to check the angle of the wind each minute and must decide whether the boat should be on port or starboard tack. Note that the proper decision must consider the current location of the boat, which we may indicate by an (x,y) coordinate.
- 3.11) What is the difference between the history of a process, and the state of a process?
- 3.12) As the purchasing manager for a major citrus juice company, you have the responsibility of maintaining sufficient reserves of oranges for sale or conversion to orange juice products. Each week, you can purchase up to a quantity q ti at price p ti from supplier i ∈ I , where the price/quantity pairs ( p ti , q ti ) i ∈I fluctuate from week to week. Let x ti be the amount that you decide to purchase from supplier i in week t to be used in week t +1. Let s 0 be your total initial inventory, and let D t be the amount of product that the company needs for production during week t . If we are unable to meet demand, the company must purchase additional product on the spot market at a spot price p spot ti .
- a) What is the exogenous stochastic process for this system?
- b) What are the decisions you can make to influence the system?
- c) What would be the state variable for your problem?
- d) Write out the transition equations.
- e) What is the one-period contribution function?

<!-- image -->

- f) Propose a reasonable structure for a decision rule for this problem, and call it X π . Your decision rule should be in the form of a function that determines how much to produce in a given period.
- g) Carefully and precisely, write out the objective function for this problem in terms of the exogenous stochastic process. Clearly identify what you are optimizing over.
- h) For your decision rule, what do we mean by the space of policies?
4. 3.13) Customers call in to a service center according to a (nonstationary) Poisson process. Let E be the set of events representing phone calls, where t e , e ∈ E is the time that the call is made. Each customer makes a request that will require time τ e to complete and will pay a reward r e to the service center. The calls are initially handled by a receptionist who determines τ e and r e . The service center does not have to handle all calls and obviously favors calls with a high ratio of reward per time unit required ( r e /τ e ). For this reason, the company adopts a policy that the call will be refused if ( r e /τ e ) &lt; γ . If the call is accepted, it is placed in a queue to wait for one of the available service representatives. Assume that the probability law driving the process is known, where we would like to find the right value of γ .
- a) This process is driven by an underlying exogenous stochastic process with element ω ∈ Ω. What is an instance of ω ?
- b) What are the decision epochs?
- c) What is the state variable for this system? What is the transition function?
- d) What is the action space for this system?
- e) Give the one-period reward function.
- f) Give a full statement of the objective function that defines the Markov decision process. Clearly define the probability space over which the expectation is defined, and what you are optimizing over.
11. 3.14) A major oil company is looking to build up its storage tank reserves, anticipating a surge in prices. It can acquire 20 million barrels of oil, and it would like to purchase this quantity over the next 10 weeks (starting in week 1). At the beginning of the week, the company contacts its usual sources, and each source j ∈ J is willing to provide q tj million barrels at a price p tj . The price/quantity pairs ( p tj , q tj ) fluctuate from week to week. The company would like to purchase (in discrete units of millions of barrels) x tj million barrels (where x tj is discrete) from source j in week t ∈ (1 , 2 , . . . , 10). Your goal is to acquire 20 million barrels while spending the least amount possible.
- a) What is the exogenous stochastic process for this system?
- b) What would be the state variable for your problem? Give an equation(s) for the system dynamics.
- c) Propose a structure for a decision rule for this problem and call it X π .
- d) For your decision rule, what do we mean by the space of policies? Give examples of two different decision rules.

- e) Write out the objective function for this problem using an expectation over the exogenous stochastic process.
- f) You are given a budget of $300 million to purchase the oil, but you absolutely must end up with 20 million barrels at the end of the 10 weeks. If you exceed the initial budget of $300 million, you may get additional funds, but each additional $1 million will cost you $1.5 million. How does this affect your formulation of the problem?
3. 3.15) You own a mutual fund where at the end of each week t you must decide whether to sell the asset or hold it for an additional week. Let r t be the one-week return (e.g. r t = 1 . 05 means the asset gained five percent in the previous week), and let p t be the price of the asset if you were to sell it in week t (so p t +1 = p t r t +1 ). We assume that the returns r t are independent and identically distributed. You are investing this asset for eventual use in your college education, which will occur in 100 periods. If you sell the asset at the end of time period t , then it will earn a money market rate q for each time period until time period 100, at which point you need the cash to pay for college.
- a) What is the state space for our problem?
- b) What is the action space?
- c) What is the exogenous stochastic process that drives this system? Give a five time period example. What is the history of this process at time t?
- d) You adopt a policy that you will sell if the asset falls below a price ¯ p (which we are requiring to be independent of time). Given this policy, write out the objective function for the problem. Clearly identify exactly what you are optimizing over.

## Chapter 4

## Introduction to Markov decision processes

This chapter provides an introduction to what are classically known as Markov decision processes, or stochastic, dynamic programming. Throughout, we assume finite numbers of discrete states and decisions ('actions' in the parlance of Markov decision processes), and we assume we can compute a one-step transition matrix. Several well-known algorithms are presented, but these are exactly the types of algorithms that do not scale well to realisticallysized problems.

So why cover material that is widely acknowledged to work only on small or highly specialized problems? First, some problems have small state and action spaces and can be solved with these techniques. Second, the theory of Markov decision processes can be used to identify structural properties that can dramatically simplify computational algorithms. But far more importantly, this material provides the intellectual foundation for the types of algorithms that we present in later chapters. Using the framework in this chapter, we can prove very powerful results that will provide a guiding hand as we step into richer and more complex problems in many real-world settings. Furthermore, the behavior of these algorithms provide important insights that guide the behavior of algorithms for more general problems.

There is a rich and elegant theory behind Markov decision processes, and this chapter is aimed at bringing it out. However, the proofs are deferred to the 'Why does it work' section (section 4.5). The intent is to allow the presentation of results to flow more naturally, but serious students of dynamic programming are encouraged to delve into these proofs. This is partly to develop a deeper appreciation of the properties of the problem as well as to develop an understanding of the proof techniques that are used in this field.

## 4.1 The optimality equations

In the last chapter, we were able to formulate our problem as one of finding a policy that maximized the following optimization problem:

<!-- formula-not-decoded -->

For most problems, solving equation (4.1) is computationally intractable, but it provides the basis for identifying the properties of optimal solutions and finding and comparing 'good' solutions to determine which is better.

With a little thought, we realize that we do not have to solve this entire problem at once. Assume our problem is deterministic (as with our budgeting problem of chapter 1). If we are in state s t and make decision x t , our transition function will tell us that we are going to land in some state s ′ = S t +1 ( x ). What if we had a function V t +1 ( s ′ ) that told us the value of being in state s ′ ? We could evaluate each possible decision x and simply choose the one decision x that had the largest value of the one-period contribution, C t ( s t , x t ), plus the value of landing in state s ′ = S t +1 ( x ) which we represent using V t +1 ( S t +1 ( x t )). Since this value represents the money we receive one time period in the future, we might discount this by a factor γ . In other words, we have to solve:

<!-- formula-not-decoded -->

Furthermore, the value of being in state s t is the value of using the optimal decision x ∗ t ( s t ). That is:

<!-- formula-not-decoded -->

Equation (4.2) is known as either Bellman's equation, in honor of Richard Bellman, or 'the optimality equations' because they characterize the optimal solution. They are also known as the Hamilton-Jacobi equations, reflecting their discovery through the field of control theory, or the Hamilton-Jacobi-Bellman equations (in honor of everybody), or HJB for short.

When we are solving stochastic problems, we have to model the fact that new information becomes available after we make the decision x t and before we measure the state variable S t +1 . Our one period contribution function is given by:

<!-- formula-not-decoded -->

When we are making decision x t , we only know s t , which means that both ˆ C t +1 ( s t , x t , W t +1 ) and the next state S t +1 are random. If we are to choose the best decision, we need to

maximize the expected contribution:

<!-- formula-not-decoded -->

Let:

<!-- formula-not-decoded -->

Substituting this into (4.3) gives us what we call the expectation form of the optimality equations:

<!-- formula-not-decoded -->

This equation forms the basis for our algorithmic work in later chapters. Interestingly, this is not the usual way that the optimality equations are written in the dynamic programming community. We can write the expectation using:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where S is the set of potential states. The reader may wish to refer back to (3.13) to review the substitution of the one-step transition matrix p t ( s ′ | s t , x t ) into equation (4.5). Substituting (4.6) back into (4.4) gives us the standard form of the optimality equations:

<!-- formula-not-decoded -->

While the transition matrix can, in practice, be computationally intractable, equation (4.7) offers a particularly elegant mathematical structure that is the basis for much of the theory about the properties of Markov decision processes.

We can write (4.7) in a more compact form. Recall that a policy π is a rule that specifies the action x t given the state s t . The probability that we transition from state S t = s to S t +1 = s ′ can be written as:

<!-- formula-not-decoded -->

We would say that ' p ss ′ ( x ) is the probability that we end up in state s ′ if we start in state s at time t and take action x .' Since the action x is determined by our policy (or, decision function) X π , it is common to write this probability as:

<!-- formula-not-decoded -->

It is often useful to write this in matrix form:

P π t = The one-step transition matrix under policy π

where p π ss ′ is the element in row s and column s ′ .

Now let C t ( x t ) and v t +1 be column vectors with elements C ts ( x t ) and v s,t +1 respectively, where s ∈ S . Then (4.7) is equivalent to:

<!-- formula-not-decoded -->

where the max operator is applied to each element in the column vector. If the decision x t is a scalar (for example, whether to sell or hold an asset), then the solution to (4.8) is a vector, with a decision x ts for each state s . Note that this is equivalent to a policy - it is a rule specifying what to do in each state. If the decision x t is itself a vector, then the solution to (4.8) is a family of decision vectors x t ( s t ) for all s t ∈ S . For example, assume our problem is to assign individual programmers to different programming tasks, where our state s t captures the availability of programmers and the different tasks that need to be completed. Of course, computing a vector x t for each state s t which is itself a vector is much easier to write than to implement.

The vector form of Bellman's equation in (4.8) can be written even more compactly using operator notation. Let M be the 'max' (or 'min') operator in (4.8) that can be viewed as acting on the vector v t +1 to produce the vector v t . Let V be the space of value functions. Then, M is a mapping:

<!-- formula-not-decoded -->

defined by equation (4.8). We may also define the operator M π for a particular policy π , which is simply the linear operator:

<!-- formula-not-decoded -->

for some vector v ∈ V . We see later in the chapter that we can exploit the properties of this operator to derive some very elegant results for Markov decision processes. These proofs provide insights into the behavior of these systems, which can guide the design of algorithms. For this reason, it is relatively immaterial that the actual computation of these equations may be intractable for many problems; the insights still apply.

.

## 4.2 The optimality equations using the post-decision state variable

In section 3.6.2, we pointed out that it is possible to capture the state of a system immediately before or immediately after a decision is made. Virtually every textbook on dynamic programming uses what we will sometimes call the pre-decision state variable. This is most natural because it has all the information we need to make a decision. The complication that arises in computational work is that if we want to make a decision that takes into account the impact on the future (which is the whole point of dynamic programming), then we have to work with the value of a decision x that puts us in state S t +1 ( x ), which is a random variable. As a result, we are forced to compute (or, more commonly, approximate) the quantity E { V t +1 ( x ) | S t } . For some problem classes, this can cause real complications.

We can circumvent this problem by formulating the optimality equations around the post-decision state variable. Recall that we can write our history of information, decisions and states as:

<!-- formula-not-decoded -->

When we wrote our recursion around the pre-decision state variable S t , we obtained the optimality equations that are given in equations (4.4)-(4.8). If we write the same equations around the post-decision state variable, we obtain equations of the form:

<!-- formula-not-decoded -->

We have indexed both the state variables and the value functions with the superscript ' x ' to denote when a quantity is computed for the post-decision state variable. The reader needs to keep in mind while reading equation (4.10) that the time index t always refers to the information content of the variable or function. The biggest difference in the optimality recursions is that now the expectation is outside of the max operator. Since we are conditioning on s x t -1 , we need the information W t in order to compute x t .

There is a simple relationship between V t ( S t ) and V x t ( S t ) that is summarized as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that from equation (4.11), V t ( S t ) is a deterministic function of V x t ( S t ( x t )). That is, we do not have to compute an expectation, but we do have to solve a maximization problem. From equation (4.12), we see that V x t ( S x t ) is just the conditional expectation of V t +1 ( S t +1 ).

So now we pose the question: why would we ever want the value functions computed around a post-decision state vector? When we use the pre-decision state vector, we have one

maximization problem to solve (for a particular state). Now, we have to solve a maximization problem within an expectation. This certainly seems more complicated.

The value of the post-decision optimality equations, oddly enough, arises purely for computational reasons. Keep in mind that for more complex problems, it is impossible to compute the expectation exactly. As a result, using the pre-decision state variable requires that we first approximate the value function, and then approximate the expectation. Using the post-decision state variable produces the following decision function:

<!-- formula-not-decoded -->

Note that to determine the decision x t +1 , we just have to sample the information that would have been available, namely W t +1 . The basic strategy in approximate dynamic programming is to simulate forward in time for a sample realization ω . Thus, we would simply simulate the information and solve the decision problem for a single sample realization (as we would do in any practical application).

## 4.3 Finite horizon problems

Finite horizon problems tend to arise in two settings. First, some problems have a very specific horizon. For example, we might be interested in the value of an American option where we are allowed to sell an asset at any time t ≤ T where T is the exercise date. Another problem is to determine how many seats to sell at different prices for a particular flight departing at some point in the future. In the same class are problems that require reaching some goal (but not at a particular point in time). Examples include driving to a destination, selling a house, or winning a game.

Asecond class of problems are actually infinite horizon, but where the goal is to determine what to do right now given a particular state of the system. For example, a transportation company might want to know what drivers should be assigned to a particular set of loads right now. Of course, these decisions need to consider the downstream impact, so models have to extend into the future. For this reason, we might model the problem over a horizon T which, when solved, yields a decision of what to do right now.

## 4.3.1 The optimality equations

The foundation of dynamic programming is the property that the optimality equations give you the optimal solution. Section (4.5) provides the core proofs, but there are some important principles that should be understood by any student interested in using dynamic programming.

We begin by writing the expected profits using policy π from time t onward:

<!-- formula-not-decoded -->

F π t ( s t ) is the expected total contribution if we are in state s t in time t , and follow policy π from time t onward. If F π t ( s t ) were easy to calculate, we would probably not need dynamic programming. Instead, it seems much more natural to calculate V π t recursively using:

<!-- formula-not-decoded -->

Our first step is to establish the equivalence between F π t and V π t using:

<!-- formula-not-decoded -->

The proof, given in section 4.5.1, uses a proof by induction: assume it is true for V π t +1 , and then show that it is true for V π t . Not surprisingly, inductive proofs are very popular in dynamic programming.

Proposition 4.3.1 is one of those small results that is easy to overlook. It establishes the equivalence between the value function for a policy and the value of a policy.

With this result in hand, we can then establish the key theorem:

Theorem 4.3.1 Let V t ( s t ) be a solution to equation (4.3) (or (4.7)). Then

<!-- formula-not-decoded -->

Theorem 4.3.1 says that the value of following the optimal policy over the horizon is the same as the solution to the optimality equations, which establishes that if we solve the optimality equations, then we know the value of the optimal policy. We should also note, however, that while an optimal solution may exist, an optimal policy may not. While such issues are of tremendous importance to the theory of Markov decision policies, they are rarely an issue in practical applications.

Theorem 4.3.1 also expresses a fundamental property of dynamic programs that was first observed by Bellman. It says that the optimal policy is the same as taking the best decision given the state you are in and then following the optimal policy from then on.

## 4.3.2 Backward dynamic programming

When we encounter a finite horizon problem, we assume that we are given the function V T ( S T ) as data. Often, we simply use V T ( S T ) = 0 because we are primarily interested in what to do now, given by x 0 , or in projected activities over some horizon t = 0 , 1 , . . . , T ph where T ph is the length of a planning horizon. If we set T sufficiently larger than T ph , then we may be able to assume that the decisions x 0 , x 1 , . . . , x T ph are of sufficiently high quality to be useful.

Solving a finite horizon problem, in principle, is straightforward. As outlined in figure 4.1, we simply have to start at the last time period, compute the value function for each possible state s ∈ S , and then step back another time period. This way, at time period t we have already computed V t +1 ( S ).

<!-- formula-not-decoded -->

Figure 4.1: A backward dynamic programming algorithm

One of the most popular illustrations of dynamic programming is the discrete asset acquisition problem (popularly known in the operations research community as the inventory planning problem). Assume that you order a quantity x t at each time period to be used in the next time period to satisfy a demand ˆ D t +1 . Any unused product is held over to the following time period. For this, our 'state variable' S t is the quantity of inventory left over at the end of the period after demands are satisfied. The transition equation is given by S t +1 = [ S t + x t -ˆ D t +1 ] + where [ x ] + = max { x, 0 } . The cost function (which we seek to minimize) is given by ˆ C t +1 ( S t , x t , ˆ D t +1 ) = c h S t + c o I x t &gt; 0 where I X = 1 if X is true and 0 otherwise. Note that the cost function is nonconvex. This does not cause us a problem if we solve our minimization problem by searching over different (discrete) values of x t . Since all our quantities are scalar, there is no difficulty finding C t ( S t , x t ) = E { ˆ C t +1 ( S t , x t , ˆ D t +1 ) } . The one-step transition matrix is computed using:

<!-- formula-not-decoded -->

where Ω is the set of (discrete) outcomes of the demand ˆ D t +1 .

Another example is the shortest path problem with random arc costs. Assume that you are trying to get from origin node r to destination node s in the shortest time possible. As you reach each intermediate node i , you are able to observe the time required to traverse each arc out of node i . Let V j be the expected shortest path time from j to the destination node s . At node i , you see the arc time ˆ τ ij ( ω ) and then choose to traverse arc i, j ∗ ( ω ) where j ∗ ( ω ) solves min j ˆ τ ij { ( ω ) + V j } . We would then compute the value of being at node i using V i = E { min j ˆ τ ij ( ω ) + V j } .

## 4.4 Infinite horizon problems

Infinite horizon problems arise whenever we wish to study a stationary problem in steady state. More importantly, infinite horizon problems provide a number of insights into the properties of problems and algorithms, drawing off an elegant theory that has evolved around this problem class. Even students who wish to solve complex, nonstationary problems will benefit from an understanding of this problem class.

We begin with the optimality equations:

<!-- formula-not-decoded -->

We can think of a steady state problem as one without the time dimension. Letting V ( s ) = lim t →∞ V t ( s t ) (and assuming the limit exists), we obtain the steady state optimality equations:

<!-- formula-not-decoded -->

The functions V ( s ) can be shown (as we do later) to be equivalent to solving the infinite horizon problem:

<!-- formula-not-decoded -->

Now define:

<!-- formula-not-decoded -->

We further define P π, 0 to be the identity matrix. Now let:

<!-- formula-not-decoded -->

be the column vector of the expected cost of being in each state given that we choose the action x t described by policy π . The infinite horizon, discounted value of a policy π starting at time t is given by:

<!-- formula-not-decoded -->

Assume that after following policy π 0 we follow policy π 1 = π 2 = . . . = π . In this case, equation (4.18) can be now written as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (4.22) shows us that the value of a policy is the single period reward plus a discounted terminal reward that is the same as the value of a policy starting at time 1. If our decision rule is stationary, then π 0 = π 1 = . . . = π t = π , which allows us to rewrite (4.22) as:

<!-- formula-not-decoded -->

This allows us to solve for the stationary reward explicitly (as long as 0 ≤ γ &lt; 1):

<!-- formula-not-decoded -->

We can also write an infinite horizon version of the optimality equations as we did earlier. Letting M be the 'max' (or 'min') operator, the infinite horizon version of equation (4.9) would be written:

<!-- formula-not-decoded -->

There are several algorithmic strategies for solving infinite-horizon problems. The first, value iteration, is the most widely used method. It involves iteratively estimating the value function. At each iteration, the estimate of the value function determines which decisions we will make and as a result defines a policy. The second strategy is policy iteration. At every iteration, we define a policy (literally, the rule for determining decisions) and then determine

Step 0: Initialization:

Set v 0 ( s ) = 0 ∀ s ∈ S .

Set n = 1.

Fix a tolerance parameter /epsilon1 &gt; 0.

Step 1: For each s ∈ S compute:

<!-- formula-not-decoded -->

Let x n +1 be the decision vector that solves equation (4.25).

Step 2: If ‖ v n +1 -v n ‖ &lt; /epsilon1 (1 -γ ) / 2 γ , set x /epsilon1 = x n +1 , v /epsilon1 = v n +1 and stop; else set n = n +1 and go to step 1.

Figure 4.2: The value iteration algorithm for infinite horizon optimization

the value function for that policy. Careful examination of value and policy iteration reveals that these are closely related strategies that can be viewed as special cases of a general strategy that uses value and policy iteration. Finally, the third major algorithmic strategy exploits the observation that the value function can be viewed as the solution to a specially structured linear programming problem.

## 4.4.1 Value iteration

Value iteration is perhaps the most widely used algorithm in dynamic programming because it is the simplest to implement and, as a result, often tends to be the most natural way of solving many problems. It is virtually identical to backward dynamic programming for finite horizon problems. In addition, most of our work in approximate dynamic programming is based on value iteration.

## Basic value iteration

Value iteration comes in several flavors. The basic version of the value iteration algorithm is given in figure 4.2.

It is easy to see that the value iteration algorithm is similar to the backward dynamic programming algorithm. Rather than using a subscript t , which we decrement from T back to 0, we use an iteration counter n that starts at 0 and increases until we satisfy a convergence criterion.

A slight variant of the value iteration algorithm provides a somewhat faster rate of convergence. In this version (typically called the Gauss-Seidel variant), we take advantage of the fact that when we are computing the expectation of the value of the future, we have

to loop over all the states s ′ to compute ∑ s ′ p ( s ′ | s, x ) v n ( s ′ ). For a particular state s , we would have already computed v n +1 (ˆ s ) for ˆ s = 1 , 2 , . . . , s -1. By simply replacing v n (ˆ s ) with v n +1 (ˆ s ) for the states we have already visited, we obtain an algorithm that typically exhibits a noticeably faster rate of convergence.

## Relative value iteration

Another version of value iteration is called relative value iteration , which is useful in problems that do not have a discount factor or where the optimal policy converges much more quickly than the value function, which may grow steadily for many iterations. The relative value iteration algorithm is shown in 4.4.

In relative value iteration, we focus on the fact that we are more interested in the convergence of the difference | v ( s ) -v ( s ′ ) | than we are in the values of v ( s ) and v ( s ′ ). What often happens is that, especially toward the limit, all the values v ( s ) start increasing by the same rate. For this reason, we can pick any state (denoted s ∗ in the algorithm) and subtract its value from all the other states.

Replace Step 1 with:

Step 1': For each s ∈ S compute:

<!-- formula-not-decoded -->

Figure 4.3: The Gauss-Seidel variation of value iteration

Step 0: Initialization:

- Choose some v 0 ∈ V .
- Choose a base state s ∗ and a tolerance /epsilon1 .
- Let w 0 = v 0 -v 0 ( s ∗ ) e where e is a vector of ones.
- Set n = 0.

Step 1: Set

<!-- formula-not-decoded -->

Step 2: If sp ( v n +1 -v n ) &lt; (1 -γ ) /epsilon1/γ , go to step 3; otherwise, go to step 1.

Step 3: Set x /epsilon1 = arg max x ∈X { c ( x ) + γP π v n } .

Figure 4.4: Relative value iteration

To provide a bit of formalism for our algorithm, we define the span of a vector v as follows:

<!-- formula-not-decoded -->

Here and throughout this section, we define the norm of a vector as:

<!-- formula-not-decoded -->

Note that the span has the following six properties:

- 1) sp ( v ) ≥ 0.
- 2) sp ( u + v ) ≤ sp ( u ) + sp ( v ).
- 3) sp ( kv ) = | k | sp ( v ).
- 4) sp ( v + ke ) = sp ( v ).
- 5) sp ( v ) = sp ( -v ).
- 6) sp ( v ) ≤ 2 ‖ v ‖ .

Property (4) implies that sp ( v ) = 0 does not mean that v = 0 and therefore it does not satisfy the properties of a norm. For this reason, it is called a semi-norm .

The relative value iteration algorithm is simply subtracting a constant from the value vector at each iteration. Obviously, this does not change the optimal decision, but it does change the value itself. If we are only interested in the optimal policy, relative value iteration often offers much faster convergence, but it may not yield accurate estimates of the value of being in each state.

## Bounds and rates of convergence

One important property of value iteration algorithms is that if our initial estimate is too low, the algorithm will rise to the correct value from below. Similarly, if our initial estimate is too high, the algorithm will approach the correct value from above. This property is formalized in the following theorem:

## Theorem 4.4.1 For a vector v ∈ V :

- a) If v satisfies v ≥ M v , then v ≥ v ∗ .
- b) If v satisfies v ≤ M v , then v ≤ v ∗ .
- c) If v satisfies v = M v , then v is the unique solution to this system of equations and v = v ∗ .

The proof is given in section 4.5.2. The result is also true for finite horizon problems. It is a nice property because it provides some valuable information on the nature of the convergence path. In practice, we generally do not know the true value function, which makes it hard to know if we are starting from above or below (although some problems have natural bounds, such as nonnegativity).

The proof of the monotonicity property above also provides us with a nice corollary. If V ( s ) = M V ( s ) for all s , then V ( s ) is the unique solution to this system of equations, which must also be the optimal solution.

This result raises the question: what if some of our estimates of the value of being in some states are too high, while others are too low? In this case, we might cycle above and below the true estimate before settling in on the final solution.

Value iteration also provides a nice bound on the quality of the solution. Recall that when we use the value iteration algorithm, we stop when

<!-- formula-not-decoded -->

where γ is our discount factor and /epsilon1 is a specified error tolerance. It is possible that we have found the optimal policy when we stop, but it is very unlikely that we have found the optimal value functions. We can, however, provide a bound on the gap between the solution we terminate with, v n and the optimal values v ∗ by using the following theorem:

Theorem 4.4.2 If we apply the value iteration algorithm with stopping parameter /epsilon1 and the algorithm terminates at iteration n with value function v n +1 , then:

<!-- formula-not-decoded -->

Let x /epsilon1 be the optimal decision rule that we terminate with, and let v π /epsilon1 be the value of this policy. Then:

‖

v

π

/epsilon1

-

v

∗

‖ ≤

/epsilon1

The proof is given in section 4.5.2. While it is nice that we can bound the error, the bad news is that the bound can be quite poor. More important is what the bound teaches us about the role of the discount factor.

We can provide some additional insights into the bound, as well as the rate of convergence, by considering a trivial dynamic program. In this problem, we receive a constant reward c at every iteration. There are no decisions, and there is no randomness. The value of this 'game' is quickly seen to be:

<!-- formula-not-decoded -->

Consider what happens when we solve this problem using value iteration. Starting with v 0 = 0, we would use the iteration:

<!-- formula-not-decoded -->

After we have repeated this n times, we have:

<!-- formula-not-decoded -->

Comparing equations (4.28) and (4.29), we see that:

<!-- formula-not-decoded -->

Similarly, the change in the value from one iteration to the next is given by:

<!-- formula-not-decoded -->

If we stop at iteration n +1, then it means that:

<!-- formula-not-decoded -->

If we choose /epsilon1 so that (4.31) holds with equality, then our error bound (from 4.27) is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From (4.30), we know that the distance to the optimal solution is:

<!-- formula-not-decoded -->

Step 0: Initialization:

- Set n = 0.
- Select a policy π 0 .

Step 1: Let v n be the solution to:

<!-- formula-not-decoded -->

Step 2: Find a policy π n +1 that satisfies:

<!-- formula-not-decoded -->

which must be solved for each state s .

Step 3: If x n +1 ( s ) = x n ( s ) for all states s , then set x ∗ = x n +1 ; otherwise, set n = n +1 and go to step 1.

Figure 4.5: Policy iteration

which matches our bound.

This little exercise confirms that our bound on the error may be tight. It also shows that the error decreases geometrically at a rate determined by the discount factor. For this problem, the error arises because we are approximating an infinite sum with a finite one. For more realistic dynamic programs, we also have the effect of trying to find the optimal policy. When the values are close enough that we have, in fact, found the optimal policy, then we have only a Markov reward process (a Markov chain where we earn rewards for each transition). Once our Markov reward process has reached steady state, it will behave just like the simple problem we have just solved, where c is the expected reward from each transition.

## 4.4.2 Policy iteration

In policy iteration, we choose a policy and then find the infinite horizon, discounted value of the policy. This value is then used to choose a new policy. The general algorithm is described in figure 4.5.

It is useful to illustrate the policy iteration algorithm in different settings. In the first, consider a batch replenishment problem where we have to replenish resources (raising capital, exploring for oil to expand known reserves, hiring people) where there are economies from ordering larger quantities. We might use a simple policy where if our level of resources R t &lt; s for some lower limit s , we order a quantity x t = S -R t . This is known as an ( s, S ) policy and is written:

<!-- formula-not-decoded -->

The parameters ( s, S ) constitute a policy π . For a given policy π n = ( s n , S n ), we can compute a one-step transition matrix P π n . We then find the steady-state value of being in each state R given this policy by solving equation (4.34). Given these values, we can find a new set of actions x n +1 for each state s , which represents a new policy. It can be shown that this vector x n +1 will have the same structure as equation (4.36), so we can infer ( s n +1 , S n +1 ) from (4.35).

This example illustrates policy iteration when the policy can be represented as a table lookup function. Now, consider an asset allocation problem where we face the problem of placing expensive components of the electric power grid around the country in case of equipment failures. If there is a failure, we will use the closest available replacement. As new components become available, we will place the new component in the empty location that has the highest value. The set of 'values' for each location represents a policy. Assume that each time there is a failure, an order for a new component is placed. However, it takes a year to receive delivery. If ¯ v i is the value of placing a component at location i , then the vector ¯ v determines the rule for how components are placed around the country as they become available (as a result, ¯ v determines our policy). Let S ti = 1 if there is a spare component at location i after the t th arrival of a new component (which is when we have to make a decision). Given ¯ v and a model of how failures arise, we can compute the probability transition matrix P π and use (4.34) to determine the value of being in each state S t (which consists of the vector of elements S ti ). We may then use the steady state vector v n to infer a new set of values ¯ v n +1 i that would then determine a new policy.

The policy iteration algorithm is simple to implement and has fast convergence when measured in terms of the number of iterations. However, solving equation (4.34) is quite hard if the number of states is large. If the state space is small, we can use v π = ( I -γP π ) -1 c π , but the matrix inverse can be computationally expensive. For this reason, we may use a hybrid algorithm that combines the features of policy iteration and value iteration.

## 4.4.3 Hybrid value-policy iteration

Value iteration is basically an algorithm that updates the value at each iteration and then determines a new policy given the new estimate of the value function. At any iteration, the value function is not the true, steady-state value of the policy. By contrast, policy iteration picks a policy and then determines the true, steady-state value of being in each state given the policy. Given this value, a new policy is chosen.

It is perhaps not surprising that policy iteration converges faster in terms of the number of iterations because it is doing a lot more work in each iteration (determining the true, steady-state value of being in each state under a policy). Value iteration is much faster per iteration, but it is determining a policy given an approximation of a value function and then performing a very simple updating of the value function, which may be far from the true value function.

A hybrid strategy that combines features of both methods is to perform a somewhat more complete update of the value function before performing an update of the policy. Figure 4.6

Step 0: Initialization:

- Set n = 1.
- Select a tolerance parameter /epsilon1 and inner iteration limit M .
- Select some v 0 ∈ V .

Step 1: Find a decision x n ( s ) for each s that satisfies:

<!-- formula-not-decoded -->

which we represent as policy π n .

Step 2: Partial policy evaluation.

- a) Set m = 0 and let: u n (0) = c π + γP π n v n -1 .
- b) If ‖ u n (0) -v n -1 ‖ &lt; /epsilon1 (1 -γ ) / 2 γ , go to step 3. Else:
- c) While m&lt;M do the following:
- i) u n ( m +1) = c π n + γP π n u n ( m ) = M π u n ( m ).
5. ii) Set m = m +1 and repeat ( i ).
- d) Set v n = u n ( M ) , n = n +1 and return to step 1.

Step 3: Set x /epsilon1 = x n +1 and stop.

Figure 4.6: Hybrid value/policy iteration

outlines the procedure where the steady state evaluation of the value function in equation (4.34) is replaced with a much easier iterative procedure (Step 2 in figure 4.6). This step is run for M iterations where M is a user-controlled parameter that allows the exploration of the value of a better estimate of the value function. Not surprisingly, it will generally be the case that M should decline with the number of iterations as the overall process converges.

## 4.4.4 The linear programming formulation

Theorem 4.4.1 showed us that if v ≥ c + γPv , then v is an upper bound (actually, a vector of upper bounds) on the value of being in each state. This means that the optimal solution, which satisfies v ∗ = c + γPv ∗ , is the smallest value of v that satisfies this inequality. We can use this insight to formulate the problem of finding the optimal values as a linear program. Let β be a vector with element β s &gt; 0 , ∀ s ∈ S . The optimal value function can be found by solving the following linear program:

<!-- formula-not-decoded -->

subject to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The linear program has a |S| -dimensional decision vector (the value of being in each state), with |S| inequality constraints (equation (4.38).

This formulation has been viewed as primarily a theoretical result since it was first suggested in 1960. The linear program can only be solved for problems with relatively small numbers of states. High quality solutions can be obtained more simply using value or policy iteration. However, recent research has suggested new approximate dynamic programming algorithms based on the linear programming formulation.

## 4.5 Why does it work?**

The theory of Markov decision processes is especially elegant. While not needed for computational work, an understanding of why they work will provide a deeper appreciation of the properties of these problems.

## 4.5.1 The optimality equations

Until now, we have been presenting the optimality equations as though they were a fundamental law of some sort. To be sure, they can easily look as though they were intuitively obvious, but it is still important to establish the relationship between the original optimization problem and the optimality equations. Since these equations are the foundation of dynamic programming, it seems beholden on us to work through the steps of proving that they are actually true.

We start by remembering the original optimization problem that we are trying to solve:

<!-- formula-not-decoded -->

Since (4.40) is, in general, exceptionally difficult to solve, we resort to the optimality equations:

<!-- formula-not-decoded -->

Our challenge is to establish that these are the same. In order to establish this result, it is going to help if we first prove the following:

Lemma 4.5.1 Let S t be a state variable that captures the relevant history up to time t , and let F t ′ ( S t +1 ) be some function measured at time t ′ ≥ t +1 conditioned on the random variable S t +1 . Then:

<!-- formula-not-decoded -->

Proof: Assume, for simplicity, that F t ′ is a discrete, finite random variable that takes outcomes in F . We start by writing:

<!-- formula-not-decoded -->

Recognizing that S t +1 is a random variable, we may take the expectation of both sides of (4.43), conditioned on S t as follows:

<!-- formula-not-decoded -->

First, we observe that we may write P ( F t ′ = f | s t +1 , s t ) = P ( F t ′ = f | s t +1 ), because conditioning on S t +1 makes all prior history irrelevant. Next, we can reverse the summations on the right hand side of (4.44) (some technical conditions have to be satisfied to do this, but let us assume that all our functions are 'nice'). This means:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which proves our result. Note that the essential step in the proof occurs in equation (4.45) when we add S t to the conditioning. /square

We are now ready to show:

## Proposition 4.5.1 F π t ( s t ) = V π t ( s t ) .

Proof: To prove that (4.40) and (4.41) are equal, we use a standard trick in dynamic programming: proof by induction. Clearly, F π T ( s T ) = V π T ( s T ) = C t ( s T ). Next, assume that it holds for t +1 , t +2 , . . . , T . We want to show that it is true for t . This means that we can write:

<!-- formula-not-decoded -->

We then use lemma 4.5.1 to write E [ E { . . . | s t +1 } | s t ] = E [ . . . | s t ]. Hence,

<!-- formula-not-decoded -->

When we condition on s t , X π t ( s t ) (and therefore C t ( s t , X π t ( s t ))) is deterministic, so we can pull the expectation out to the front giving:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which proves our result.

/square

The expectation in equation (4.41) provides for a significant level of generality. For example, we might have a history dependent process where we would write:

<!-- formula-not-decoded -->

where h t +1 = ( h t , ω t ) (if we are using the exogneous stochastic process). When we have a Markovian process, we can express the conditional expectation in (4.41) using a one-step transition matrix:

<!-- formula-not-decoded -->

Using equation (4.41), we have a backward recursion for calculating V π t ( s t ) for a given policy π . Now that we can find the expected reward for a given π , we would like to find the best π . That is, we want to find:

<!-- formula-not-decoded -->

As before, if the set Π is infinite, we replace the 'max' with 'sup'. We solve this problem by solving the optimality equations. These are:

<!-- formula-not-decoded -->

We are claiming that if we find the set of V ′ s that solves (4.55), then we have found the policy that optimizes F π t . We state this claim formally as:

Theorem 4.5.1 Let V t ( s t ) be a solution to equation (4.55). Then

<!-- formula-not-decoded -->

Proof: The proof is in two parts. First, we show by induction that V t ( s t ) ≥ F ∗ t ( s t ) for all s t ∈ S and t = 0 , 1 , . . . , T -1. Then, we show that the reverse inequality is true, which gives us the result.

Part 1:

We resort again to our proof by induction. Since V T ( s T ) = C t ( s T ) = F π T ( s T ) for all s T and all π ∈ Π, we get that V T ( s T ) = F ∗ T ( s T ).

Assume that V t ( s t ) ≥ F ∗ t ( s t ) for t = n +1 , n +2 , . . . , T , and let π be an arbitary policy. For t = n , the optimality equation tells us:

<!-- formula-not-decoded -->

By the induction hypothesis, F ∗ n +1 ( s ) ≤ V n +1 ( s ), so we get:

<!-- formula-not-decoded -->

Of course, we have that F ∗ n +1 ( s ) ≥ F π n +1 ( s ) for an arbitrary π . Also let X π ( s n ) be the decision that would be chosen by policy π when in state s n . Then:

<!-- formula-not-decoded -->

This means:

<!-- formula-not-decoded -->

which proves part 1.

Part 2: Now we are going to prove the inequality from the other side. Specifically, we want to show that for any /epsilon1 &gt; 0 there exists a policy π that satisfies:

<!-- formula-not-decoded -->

To do this, we start with the definition:

<!-- formula-not-decoded -->

We may let x n ( s n ) be the decision rule that solves (4.57). This rule corresponds to the policy π . In general, the set X may be infinite, whereupon we have to replace the 'max' with a 'sup' and handle the case where an optimal decision may not exist. For this case, we know that we can design a decision rule x n ( s n ) that returns a decision x that satisfies:

<!-- formula-not-decoded -->

We can prove (4.56) by induction. Assume that it is true for t = n + 1 , n + 2 , . . . , T . We already know that

<!-- formula-not-decoded -->

We can use our induction hypothesis which says F π n +1 ( s ′ ) ≥ V n +1 ( s ′ ) -( T -( n +1)) /epsilon1 to get:

<!-- formula-not-decoded -->

Now, using equation (4.58), we replace the term in brackets with the smaller V n ( s n ) (equation (4.58)):

<!-- formula-not-decoded -->

which proves the induction hypothesis. We have shown that:

<!-- formula-not-decoded -->

This proves the result.

/square

Now we know that solving the optimality equations also gives us the optimal value function. This is our most powerful result because we can solve the optimality equations for many problems that cannot be solved any other way.

## 4.5.2 Proofs for value iteration

Infinite horizon dynamic programming provides a compact way to study the theoretical properties of these algorithms. The insights gained here are applicable to problems even when we cannot apply this model, or these algorithms, directly.

Our first result establishes a monotonicity property that can be exploited in the design of an algorithm:

Theorem 4.5.2 For a vector v ∈ V :

- a) If v satisfies v ≥ M v , then v ≥ v ∗ .
- b) If v satisfies v ≤ M v , then v ≤ v ∗ .
- c) If v satisfies v = M v , then v is the unique solution to this system of equations and v = v ∗ .

Proof: Part ( a ) requires that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (4.59) is true by assumption (part ( a ) of the theorem) and equation (4.60) is true because π 0 is some policy that is not necessarily optimal for the vector v . Using similar reasoning, equation (4.61) is true because π 1 is another policy which, again, is not necessarily optimal. Using P π, ( t ) = P π 0 P π 1 · · · P π t , we obtain by induction:

<!-- formula-not-decoded -->

Recall that:

<!-- formula-not-decoded -->

Breaking the sum in (4.63) into two parts allows us to rewrite the expansion in (4.62) as:

<!-- formula-not-decoded -->

Taking the limit of both sides of (4.64) as t →∞ gives us:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The limit in (4.65) exists as long as the reward function c π is bounded and γ &lt; 1. Because (4.66) is true for all π ∈ Π, it is also true for the optimal policy, which means that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which proves part (a) of the theorem. Part ( b ) can be proved in an analogous way. Parts (a) and (b) mean that v ≥ v ∗ and v ≤ v ∗ . If v = M v , then we satisfy the preconditions of both parts (a) and (b), which means they are both true and therefore we must have v = v ∗ . /square

This result means that if we start with a vector that is higher than the optimal vector, then we will decline monotonically to the optimal solution (almost - we have not quite proven that we actually get to the optimal). Alternatively, if we start below the optimal vector, we will rise to it. Note that it is not always easy to find a vector v that satisfies either condition ( a ) or ( b ) of the theorem. In problems where the rewards can be positive and negative, this can be tricky.

We now undertake the proof that the basic value function iteration converges to the optimal solution. This is not only an important result, it is also an elegant one that brings some powerful theorems into play. The proof is also quite short. However, we will need some mathematical preliminaries:

Definition 4.5.1 Let V be a set of (bounded, real-valued) functions and define the norm of v by:

<!-- formula-not-decoded -->

where we replace the ' sup ' with a ' max ' when the state space is finite. Since V is closed under addition and scalar multiplication and has a norm, it is a normed linear space .

Definition 4.5.2 T : V → V is a contraction mapping if there exists a γ , 0 ≤ γ &lt; 1 such that:

<!-- formula-not-decoded -->

Definition 4.5.3 A sequence v n ∈ V , n = 1 , 2 , . . . is said to be a Cauchy sequence if for all /epsilon1 &gt; 0 , there exists N such that for all n, m ≥ N :

<!-- formula-not-decoded -->

Definition 4.5.4 A normed linear space is complete if every Cauchy sequence contains a limit point in that space.

Definition 4.5.5 A Banach space is a complete normed linear space.

Definition 4.5.6 We define the norm of a matrix Q as:

<!-- formula-not-decoded -->

which is to say, the largest row sum of the matrix. If Q is a one-step transition matrix, then ‖ Q ‖ = 1 .

Definition 4.5.7 The triangle inequality , which is satisfied by the Euclidean norm as well as many others, means that given two vectors a, b ∈ /Rfractur n :

<!-- formula-not-decoded -->

The triangle inequality is commonly used in proofs because it helps us establish bounds between two solutions (and in particular, between a solution and the optimum).

We now state and prove one of the famous theorems in applied mathematics and then use it immediately to prove convergence of the value iteration algorithm.

Theorem 4.5.3 (Banach Fixed-Point Theorem) Let V be a Banach space, and let T : V → V be a contraction mapping. Then:

- a) There exists a unique v ∗ ∈ V such that Tv ∗ = v ∗ .
- b) For an arbitrary v 0 ∈ V , the sequence v n defined by: v n +1 = Tv n = T n +1 v 0 converges to v ∗ .

Proof: We start by showing that the distance between two vectors v n and v n + m goes to zero for sufficiently large n and by writing the difference v n + m -v n using the following:

<!-- formula-not-decoded -->

Taking norms of both sides and invoking the triangle inequality gives:

<!-- formula-not-decoded -->

Since γ &lt; 1 for sufficiently large n , the right hand side of (4.67) can be made arbitrarily small, which means that v n is a Cauchy sequence. Since V is complete , it must be that v n has a limit point v ∗ . From this we conclude:

<!-- formula-not-decoded -->

We now want to show that v ∗ is a fixed point of the mapping T . To show this, we observe:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (4.69) comes from the properties of a norm. We play our standard trick of adding and subtracting a quantity (in this case, v n ), which sets up the triangle inequality in (4.71). Using v n = Tv n -1 gives us (4.72). The inequality in (4.73) is based on the assumption of the theorem that T is a contraction mapping. From (4.68), we know that

<!-- formula-not-decoded -->

Combining (4.69), (4.73) and (4.74) gives:

<!-- formula-not-decoded -->

from which we conclude:

<!-- formula-not-decoded -->

∗ ∗

<!-- formula-not-decoded -->

We can now show that the value iteration algorithm converges to the optimal solution if we can establish that M is a contraction mapping. So we need to show:

Proposition 1 If 0 ≤ γ &lt; 1 , then M is a contraction mapping on V .

Proof: Let u ( s ) , v ( s ) ∈ V and assume that M v ( s ) ≥ M u ( s ) and let:

<!-- formula-not-decoded -->

where we assume that a solution exists. Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

≤

c

(

s, x

∗

s

(

v

)) +

γ

∑

s

′

∈S

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (4.75) is true by assumption while (4.76) holds by definition. The inequality in (4.77) holds because x ∗ s ( v ) is not optimal when the value function is u , giving a reduced value

p

(

s

′

|

s, x

∗

s

(

v

))

v

(

s

′

)

in the second set of parentheses. Equation (4.78) is a simple reduction of (4.77). Equation (4.79) forms an upper bound because the definition of ‖ v -u ‖ is to replace all the elements [ v ( s ) -u ( s )] with the largest element of this vector. Since this is now a vector of constants, we can pull it outside of the summation, giving us (4.80), which then easily reduces to (4.81) because the probabilities add up to one.

This result states that if M v ( s ) ≥ M u ( s ), then M v ( s ) - M u ( s ) ≤ γ ‖ v -u ‖ . If we start by assuming that M v ( s ) ≤ M u ( s ), then we obtain the same result with the inequality reversed. This gives us:

<!-- formula-not-decoded -->

Since (4.82) is true for all s , then, from our definition of a norm, we obtain:

<!-- formula-not-decoded -->

This means that M is a contraction mapping, which means that the sequence v n generated by v n +1 = M v n converges to a unique limit point v ∗ that satisfies the optimality equations. /square

We now wish to establish a bound on our error from value iteration, which will establish our stopping rule. We propose two bounds: one on the value function estimate that we terminate with and one for the long-run value of the decision rule that we terminate with. To define the latter, let x /epsilon1 be the decision that satisfies our stopping rule, and let π /epsilon1 = { x /epsilon1 , x /epsilon1 , . . . , x /epsilon1 , . . . } be an infinite horizon policy made up of decisions x /epsilon1 with value v π /epsilon1 .

Theorem 4.5.4 If we apply the value iteration algorithm with stopping parameter /epsilon1 and the algorithm terminates at iteration n with value function v n +1 , then:

<!-- formula-not-decoded -->

Let x /epsilon1 be the optimal decision rule that we terminate with, and let v π /epsilon1 be the value of this policy. Then:

<!-- formula-not-decoded -->

Proof: We start by writing:

<!-- formula-not-decoded -->

Recall that x /epsilon1 is the decision that solves M v n +1 , which means that M ( x /epsilon1 ) v n +1 = M v n +1 . This allows us to rewrite the first term on the right hand side of (4.85) as:

<!-- formula-not-decoded -->

Solving for ‖ v π /epsilon1 -v n +1 ‖ gives:

<!-- formula-not-decoded -->

We can use similar reasoning applied to the second term in equation (4.85) to show that:

<!-- formula-not-decoded -->

The value iteration algorithm stops when: ‖ v n +1 -v n ‖ ≤ /epsilon1 (1 -γ ) / 2 γ . Substituting this in (4.86) gives:

<!-- formula-not-decoded -->

Recognizing that the same bound applies to ‖ v π /epsilon1 -v n +1 ‖ and combining these with (4.85) gives us:

<!-- formula-not-decoded -->

which completes our proof.

Next, we turn our attention to two relatively minor results that tend to receive a lot of attention in treatments of Markov decision policies. The first is the optimality of Markovian policies, and the second is the optimality of deterministic policies.

## 4.5.3 Optimality of Markovian policies

Under the assumptions that we are using, we can show that we do not need to worry about randomized, history-dependent policies.

We start by showing that we only need to consider state dependent policies. The simple reason for this property is that we have already assumed (implicitly) that both our reward

function and our transition matrix are state dependent. So, the property that the value function is also state dependent should not be surprising.

We use the representation that:

<!-- formula-not-decoded -->

It is clear, then, that:

<!-- formula-not-decoded -->

Once again, we show our result by using induction. Clearly, V T ( h T ) = C t ( s T ), so the result is true for t = T . Now, assume it is true for n = t +1 , t +2 , . . . , T , and we will show it is true for n = t . We start with:

<!-- formula-not-decoded -->

Using the induction hypothesis gives:

<!-- formula-not-decoded -->

Since everything on the right hand side is only a function of s t , we can then write V t ( h t ) = V t ( s t ).

Note that we cannot always write functions purely as a function of the state. A reward may depend not only on how well you serve the customer now, but also on how well you have been doing over the last few time periods. For example, if you do not satisfy a customer, you may have to pay a penalty. This penalty may be small if you generally provide good service, but may be much higher if your recent performance has been poor.

More often, it is the transition matrix that is history dependent. Sequences of random variables often exhibit autocorrelation. Generally, when this happens, we cannot even compute a transition matrix. Rather we have to work explicitly with a relatively complex transition function. Even more frequently, we may have a series of observations from past history instead of an underlying mathematical model. .

## 4.5.4 Optimality of deterministic policies

A second result that is commonly used in Markov decision processes is the optimality of deterministic policies. Recall that a deterministic policy arises when you can reliably choose

an action. For example, you are the one deciding how many bags of potato chips to purchase. However, you may have to work through an intermediary. Perhaps you have a sales staff that goes out to stores and places the orders for you. Some of your sales staff are aggressive and tend to order in large quantities, partially because they are better at wrangling shelf space from the store owner. Other people, however, are more timid and place smaller orders because the retail store outlets do not want to give them as much shelf space. By choosing the right person, you are influencing the probability that a certain action will be taken.

When we can only influence the likelihood of an action, then we have an instance of a randomized MDP. Let:

q π t ( x | s t ) = The probability that decision x will be taken at time t given state s t and policy π (more precisely, decision rule X π ).

In this case, our optimality equations look like:

<!-- formula-not-decoded -->

Now let us consider the single best action that we could take. Call this x ∗ , given by:

<!-- formula-not-decoded -->

This means that:

<!-- formula-not-decoded -->

Substituting (4.89) back into (4.88) gives us:

<!-- formula-not-decoded -->

What this means is that if you have a choice between picking exactly the action you want versus picking a probability distribution over potentially optimal and nonoptimal actions, you would always prefer to pick exactly the best action. Clearly, this is not a surprising result.

## 4.6 Bibliographic notes

Discrete state MDP: Bellman (1957), Howard (1971), Derman (1970), Dreyfus &amp; Law (1977), Dynkin (1979), Ross (1983), Heyman &amp; Sobel (1984), Puterman (1994), Continuous DP: White (1969), DP and optimal control: Bellman (1971), Bertsekas (1976), Bertsekas &amp; Tsitsiklis (1996), Bertsekas &amp; Shreve (1978), DP in economics: Stokey &amp; R. E. Lucas (1989), Chow (1997), Stochastic shortest paths - Bertsekas &amp; Tsitsiklis (1991)

## Exercises

- 4.1) A classical inventory problem works as follows: Assume that our state variable R t is the amount of product on hand at the end of time period t and that D t is a random variable giving the demand during time interval ( t -1 , t ) with distribution p d = Prob ( D t = d ). The demand in time interval t must be satisfied with the product on hand at the beginning of the period. We can then order a quantity x t at the end of period t that can be used to replenish the inventory in period t +1.
- a) Give the transition function that relates R t +1 to R t if the order quantity is x t (where x t is fixed for all R t ).
- b) Give an algebraic version of the one-step transition matrix P π = { p π ij } where p π ij = Prob ( R t +1 = j | R t = i, X π = x t ).
- 4.2) Repeat the previous exercise, but now assume that we have adopted a policy π that says we should order a quantity x t = 0 if R t ≥ s and x t = Q -R t if R t &lt; s (we assume that R t ≤ Q ). Your expression for the transition matrix will now depend on our policy π (which describes both the structure of the policy and the control parameter s ).
- 4.3) An insurance company labels all drivers as either 'G' for good drivers or 'B' for bad drivers based on how many points they have on their license. Good drivers are charged $1200 per year in insurance, while bad drivers are charged $2000 per year. In any given year, a typical good driver will collect 0 points with probability 0.70, 2 points with probability 0.20, and 4 points with probability 0.10. A driver who collects 0 points is allowed to deduct 2 points from their total, although he/she can never go below zero. A driver who has been classified as bad, as a result of the higher insurance, will collect 0 points with probability 0.80, 2 points with probability 0.15 and 4 points with probability 0.05. The insurance company has found that drivers with 0 points on their license cost, on average, $500 per year; drivers with 2 points on their license cost $3000 per year. For drivers with more than 2 points, the insurance company has found that their insurance costs are reasonably explained by the function C ( S ) = 4000 + 500 S where S is the number of points (the state of the driver) for the driver. If the driver

accumulates 12 points, then the company drops the policy (often the driver also loses his/her license). If S t is the number of points for a driver at the end of a year, and if a driver is deemed a bad driver if S t ≥ π , then we would like to determine the optimal policy. Formulate and implement the value iteration algorithm to solve this problem. Use zero for the initial estimate of the value of being in each state, and plot the value of being in each state after each iteration. Also, plot the average value over the first five states ( S = 0 , 2 , 4 , 6 , 8) at each iteration.

- 4.4) We are going to use a very simple Markov decision process to illustrate how the initial estimate of the value function can affect convergence behavior. In fact, we are going to use a Markov reward process to illustrate the behavior because our process does not have any decisions. Assume we have a two-stage Markov chain with one-step transition matrix:

<!-- formula-not-decoded -->

The contribution of each transition is given by:

<!-- formula-not-decoded -->

Apply the value iteration algorithm for an infinite horizon problem (note that you are not choosing a decision so there is no maximization step). The calculation of the value of being in each state will depend on your previous estimate of the value of being in each state. The calculations can be easily implemented in a spreadsheet. Assume that you start in state 1 and that your discount factor is .8.

- a) Plot the value of being in state 1 as a function of the number of iterations if your initial estimate of the value of being in each state is 0. Show the graph for 50 iterations of the algorithm.
- b) Repeat this calculation using initial estimates of 100.
- c) Repeat the calculation using an initial estimate of the value of being in state 1 of 100, and use 0 for the value of being in state 2. Contrast the behavior with the first two starting points.
4. 4.5) Repeat exercise (4.4), but this time initialize the value function by doubling the final value function estimates obtained.
5. 4.6) Repeat exercise (4.4) using policy iteration. Plot the average value function (over the first four states) after each iteration against the value function found using value iteration after each iteration. Try to estimate the computation time for one iteration of value iteration and one iteration of policy iteration.
6. 4.7) Repeat exercise (4.4) using the hybrid value-policy iteration algorithm. Show the average value function after each major iteration (update of n ) with M = 1 , 2 , 3 , 5 , 10.

- 4.8) An oil company will order tankers to fill two large storage tanks. One full tanker is required to fill an entire storage tank. Orders are placed at the beginning of each four week accounting period but do not arrive until the end one. During this period, the company may be able to sell 0, 1 or 2 tanks of oil to one of the regional chemical companies (orders are conveniently made in units of storage tanks). The probability of a demand of 0, 1 or 2 is 0.40, 0.40 and 0.20, respectively.

A tank of oil costs $1 million (M) to purchase and sells for $2M. It costs $0.020M to store a tank of oil during each period (oil ordered in period t , which cannot be sold until period t +1, is not charged any holding cost in period t ). Storage is only charged on oil that is in the tank at the beginning of the period and remains unsold during the period. It is possible to order more oil than can be stored. For example, the company may have two full storage tanks, order three more, and then only sell one. This means that at the end of the period, they will have four tanks of oil (and can only store two). When this happens, the company must sell the oil directly from the ship for a price of $0.70 M (a loss of $0.3M). There is no penalty for unsatisfied demand.

An order placed in time period t must be paid for in time period t even though the order does not arrive until t +1. The company uses an interest rate of 20 percent per accounting period (that is, a discount factor of 0.80).

- a) Give an expression for the one-period reward function r ( s, d ) for being in state s and making decision d . Compute the reward function for all possible states (0, 1, 2) and all possible decisions (0, 1, 2).
- b) Find the one-step probability transition matrix when your action is to order one or two tanks of oil. The transition matrix when you order zero is given by:
- c) Write out the general form of the optimality equations and solve this problem in steady state.
- d) Solve the optimality equations using the value iteration algorithm, starting with V ( s ) = 0 for s = 0 , 1 and 2. You may use a programming environment, but the problem can be solved in a spreadsheet. Run the algorithm for 20 iterations. Plot V n ( s ) for s = 0 , 1 , 2 , and give the optimal action for each state at each iteration.
- e) Give a bound on the value function after each iteration.
6. 4.9) Every day, a salesman visits N customers in order to sell the R identical items he has in his van. Each customer is visited exactly once and each customer buys zero or one item. Upon arrival at a customer location, the salesman quotes one of the prices 0 &lt; p 1 ≤ p 2 ≤ . . . ≤ p m . Given that the quoted price is p i , a customer buys an item with probability r i . Naturally, r i is decreasing in i . The salesman is interested in maximizing the total expected revenue for the day. Show that if r i p i is increasing in i , then it is always optimal to quote the highest price p m .

```
From-To 0 1 2 0 1 0 0 1 0 . 6 0 . 4 0 2 0 . 2 0 . 4 0 . 4
```

- 4.10) Consider a finite-horizon MDP with the following properties:
- -S ∈ /Rfractur n , the action space X is a compact subset of /Rfractur n , X ( s ) = A for all s ∈ S .
- -C t ( s t , x t ) = c t s t + g t ( x t ), where g t ( · ) is a known scalar function, and C N ( s N ) = c N s N .
- -If action x t is chosen when the state is s t at time t , the next state is

<!-- formula-not-decoded -->

where f t ( · ) is a given function in /Rfractur n , and A t and ω t are respectively n × n and n × 1dimensional random variables whose distributions are independent of { s n : n ≤ t } , { x n : n ≤ t } , { A n : n ≤ t -1 } , { ω n : n ≤ t -1 } . Assume that the optimal expected reward is known to be finite.

- a) Show that the optimal value function is linear in the state variable.
- b) Show that there exists an optimal policy π ∗ = ( x ∗ 1 , . . . , x ∗ N -1 ) composed of constant decision functions. That is, X π ∗ t ( s ) = x ∗ t for all s ∈ S for some constant x ∗ t .
3. 4.11) Assume that you have invested R 0 dollars in the stock market which evolves according to the equation:

<!-- formula-not-decoded -->

where ε t is a discrete, positive random variable that is independent and identically distributed, and where 0 &lt; γ &lt; 1. If you sell the stock at the end of period t , it will earn a riskless return r until time T . You have to the stock, all on the same day, some time before T .

- a) Write a dynamic programming recursion to solve the problem.
- b) Show that there exists a point in time τ such that it is optimal to sell for t ≥ τ , and optimal to hold for t &lt; τ .
- c) How does your answer to (b) change if you are allowed to sell only a portion of the assets in a given period?
4. 4.12) You need to decide when to replace your car. If you own a car of age y years, then the cost of maintaining the car that year will be c ( y ). Purchasing a new car (in constant dollars) costs P dollars. If the car breaks down, which it will do with probability b ( y ) (the breakdown probability), it will cost you an additional K dollars to repair it, after which you immediately sell the car and purchase a new one. At the same time, you express your enjoyment with owning a new car as a negative cost -r ( y ) where r ( y ) is a declining function with age. At the beginning of each year, you may choose to purchase a new car ( z = 1) or to hold onto your old one ( z = 0). You anticipate that you will actively drive a car for another T years.
- a) Identify all the elements of a Markov decision process for this problem.

- b) Write out the objective function which will allow you to find an optimal decision rule.
- c) Write out the one-step transition matrix.
- d) Write out the optimality equations that will allow you to solve the problem.
4. 4.13) You are trying to find the best parking space to use that minimizes the time needed to get to your restaurant. There are 50 parking spaces, and you see spaces 1 , 2 , . . . , 50 in order. As you approach each parking space, you see whether it is full or empty. We assume, somewhat heroically, that the probability that each space is occupied follows an independent Bernoulli process, which is to say that each space will be occupied with probability p , free with probability 1 -p , and that each outcome is independent of the other.

It takes 2 seconds to drive past each parking space and it takes 8 seconds to walk past. That is, if we park in space n, it will require 8(50-n) seconds to walk to the restaurant. Furthermore, it would have taken you 2n seconds to get to this space. If you get to the last space without finding an opening, then you will have to drive into a special lot down the block, adding 30 seconds to your trip.

We want to find an optimal strategy for accepting or rejecting a parking space.

- a) Give the sets of state and action spaces and the set of decision epochs.
- b) Give the expected reward function for each time period and the expected terminal reward function.
- c) Give a formal statement of the objective function.
- d) Give the optimality equations for solving this problem.
- e) You have just looked at space 45, which was full. There are five more spaces remaining (46 through 50). What should you do? Using p = 0 . 6, find the optimal policy by solving your optimality equations for parking spaces 46 through 50.
- f) Give the optimal value of the objective function in part (e) corresponding to your optimal solution.
7. 4.14) We have a four state process (shown in the figure). In state 1, we will remain in the state with probability 0.7, and will make a transition to state 2 with probability 0.3. In states 2 and 3, we may choose between two policies: remain in the state waiting for an upward transition or make the decision to return to state 1 and receive the indicated

<!-- image -->

- reward. In state 4, we return to state 1 immediately and receive $20. We wish to find an optimal long run policy using a discount factor γ = . 8. Set up and solve the optimality equations for this problem.
- 4.15) Assume that you have been applying value iteration to a four-state Markov decision process, and that you have obtained the values over iterations 8 through 12 shown in the table below (assume a discount factor of 0.90). Assume you stop after iteration 12. Give the tightest possible (valid) bounds on the optimal value of being in each state.
- 4.16) In the proof of theorem 4.5.2 we showed that if v ≥ M v , then v ≥ v ∗ . Go through the steps of proving the converse, that if v ≤ M v , then v ≤ v ∗ .

|       |   Iteration |   Iteration |   Iteration |   Iteration |   Iteration |
|-------|-------------|-------------|-------------|-------------|-------------|
| State |        8    |        9    |       10    |       11    |       12    |
| 1     |        7.42 |        8.85 |        9.84 |       10.54 |       11.03 |
| 2     |        4.56 |        6.32 |        7.55 |        8.41 |        9.01 |
| 3     |       11.83 |       13.46 |       14.59 |       15.39 |       15.95 |
| 4     |        8.13 |        9.73 |       10.85 |       11.63 |       12.18 |

## Chapter 5

## Introduction to approximate dynamic programming

In chapter 4, we saw that we could solve intractably complex optimization problems of the form:

<!-- formula-not-decoded -->

by recursively computing the optimality equations:

<!-- formula-not-decoded -->

backward through time. Equation (5.1) can be computationally intractable even for very small problems. The optimality equations give us a mechanism for solving these stochastic optimization problems in a simple and elegant way. Unfortunately, in a vast array of applications, the optimality equations are themselves computationally intractable.

Approximate dynamic programming offers a powerful set of strategies for problems that are hard because they are large, but this is not the only application. Even small problems may be hard because we lack a formal model of the information process, or we may not know the transition function. For example, we may have observations of changes in prices in an asset, but we do not have a mathematical model that describes these changes. In this case, we are not able to compute the expectation. Alternatively, consider the problem of modeling how the economy of a small country responds to loans from the International Monetary Fund. If we do not now how the economy responds to the size of a loan, then this means that we do not know the transition function.

We begin our presentation by revisiting the 'curses of dimensionality.' The remainder of the chapter provides an overview of the basic principles and vocabulary of approximate dynamic programming, which helps to set the stage for the important material on stochastic approximation methods in chapter 6.

## 5.1 The three curses of dimensionality (revisited)

The concept of the backward recursion of dynamic programming is so powerful that we have to remind ourselves again why its usefulness can be so limited for many problems. We start by restating the optimality equations for ease of reference:

<!-- formula-not-decoded -->

Let us now reconsider our simple asset acquisition problem, but now assume that we have multiple asset classes. For example, we may wish to purchase stocks each month for our retirement account using the money that is invested in it each month. Our on-line brokerage charges $50 for each purchase order we place, so we have an incentive to purchase stocks in reasonably sized lots. Let ˆ p tk be the purchase price of asset type k ∈ K in period t , and let x tk be the number of shares we purchase in period t . The total cost of our order is:

<!-- formula-not-decoded -->

Each month, we have $2000 to invest in our retirement plan. We have to make sure that ∑ k ∈K x tk ≤ $2000.

Let S tk be the number of shares of stock k that we have on hand at the end of period t . Assume for the moment that we do not sell any shares, but the value of the porfolio reflects the purchase prices ˆ p t which are random. In addition, each stock returns a random dividend γ tk given in dollars per share which are reinvested. Thus, the information we receive in each time period is:

<!-- formula-not-decoded -->

If we were to reinvest our dividends, then our transition function would be:

<!-- formula-not-decoded -->

We now have a problem where our state S t and decision x t variables have |K| dimensions. Furthermore, our information process W t has 2 |K| dimensions. To illustrate how bad this gets, assume that we are restricting our attention to 50 different stocks, and that we always purchase shares in blocks of 100, but that we can own as many as 20,000 shares (or 200 blocks) of any one stock at a given time. This means that the size of our state space is |S| = 200 50 . Thus, we would have to loop over equation (5.3) 200 50 times to compute V t ( s t ) for all possible values of s t . This is the classical 'curse of dimensionality' that is widely cited as the Achilles heel of dynamic programming.

As we look at our problem, we see the situation is much worse. For each of the 200 50 states, we have to compute the expectation in (5.3). Since our random variables might not be independent (the prices generally will not be) we could conceive of finding a joint probability distribution and performing 2 × 50 nested summations to complete the expectation. It seems that this can be avoided if we work with the one-step transition matrix and use the form of the optimality equations expressed in (4.7). This form of the recursion, however, only hides the expectation (the one-step transition matrix is, itself, an expectation).

We are not finished. This expectation has to be computed for each action x . Assume we are willing to purchase up to 10 blocks of 100 shares of any one of the 50 stocks. Since we can also purchase 0 shares of a stock, we have upwards of 11 50 different combinations of x t that we might have to consider (the actual number is smaller because we have a budget constraint of $2000 to spend).

So, we see that we have three curses of dimensionality if we wish to work in discrete quantities. In other words, vectors really cause problems. In this chapter, we are going to review strategies for overcoming these 'curses of dimensionality.'

## 5.2 Monte Carlo sampling and forward dynamic programming

Without question, the single most powerful tool we have is to step forward through time. This means that we need to have at hand an approximation of the value function that we can use to make decisions. Needless to say, this approximation will have to be updated iteratively, so at iteration n , we let:

¯ V n t ( S t ) = Approximate value function for time t at iteration n .

We use the notational convention (consistent with how we index time) that the value function approximation is indexed by n -1, as in ¯ V n -1 t ( S t ), while we are solving problems in iteration n . We then use information gained in iteration n to produce ¯ V n t ( S t ).

To make a decision, we have to solve (5.2), which means finding the expectation. Frequently, this is computationally intractable, so we approximate it by randomly sampling a set of outcomes ˆ Ω t where ˆ p t ( ω t ) is the probability of ω t ∈ ˆ Ω. Thus, we make a decision by solving:

<!-- formula-not-decoded -->

Having made a decision, we determine the next state by drawing a new random sample ω t +1 and computing:

<!-- formula-not-decoded -->

Step 0. For all t , initialize an approximation for the value function ¯ V 0 t ( S t ) for all states S . Let n = 1 and set t = 0.

Step 1. For time period t do:

Step 1a. Solve:

<!-- formula-not-decoded -->

Step 1b. Choose ω t +1 .

Step 1c. Compute S t +1 = S M ( S t , X n t ( S t ) , ω t +1 ).

Step 1d. Set t = t +1. If t &lt; T go to step 1 a .

Step 2. Use the results to update the approximation ¯ V n -1 t ( S t ) (using any of a variety of algorithms).

Step 3. Let n := n +1. If n &lt; N , go to step 1.

Figure 5.1: Generic forward dynamic programming algorithm

where S M ( · ) is our state transition function. In some settings, it is notationally more compact to use the state variable S t +1 instead of the transition function.

An outline of the overall algorithm is provided in figure 5.1. The power of this simple algorithm is that it eliminates the need to search over an entire state space. Instead, it steps forward along a sample path ω and visits only the sequence of states that are produced by a particular set of decisions.

There is a broad range of methods for updating the value function. In the simplest case, we update the value of being in a particular state by first computing:

<!-- formula-not-decoded -->

which is an estimate of the value of being in state S t . Of course, this estimate depends on our value function approximation ¯ V n -1 t +1 ( S t +1 ). Since there is typically some noise in our estimate of ˆ v n t , we perform the following smoothing operation to update ¯ V n -1 t ( S t ), which is our current estimate of the value of being in state S t , to produce an updated estimate:

<!-- formula-not-decoded -->

where α n is known as a 'stepsize' (among other things), and takes on values between 0 and 1. Equation (5.6) is an operation that we see a lot of in approximate dynamic programming. As we discover over the remainder of this volume, there are a number of ways to represent a value function, and a similarly broad range of methods for updating the approximation.

Remark: It is important to emphasize that the idea of stepping forward through states and updating estimates of the value of being in only the states that you visit does not solve the

problem of high-dimensional state spaces. Instead, it replaces the computational burden of looping over every state with the statistical problem of estimating the value of being in each state. As a general rule, we do not need the value of being in every state, but we may not know which states we need to evaluate (if we did, we would have a much easier time of implementing backward dynamic programming). However, there are a number of strategies we can use if we exploit the structure of the problem, something that backward dynamic programming does not really allow us to do. As you gain experience with these methods, you will find that some understanding of the structure of the problem tends to be critical.

It is useful to think of the value function approximation ¯ V as a kind of policy. Thus, our policy is to solve the function in (5.4). Typically, we will have the option of choosing between different classes of functional approximations (for example, functions that are linear in the state variable, separable quadratic, nonseparable, and so on). We can think of each class of functions as being a class of policies. Choosing the best policy within a class is equivalent to finding the best set of parameters.

As we make the transition to high dimensional decision vectors, we will encounter applications where the myopic problem (that is, the problem of maximizing the one-period reward) is itself fairly hard. These situations typically arise in the management of physical assets where the myopic problem falls in one of the classes of hard optimization problems (see examples).

Example 5.1: An operator of business jets has to respond to requests for service with as little as four hours notice. The operator has to find a pilot and aircraft to satisfy the customer. The choice of pilot and aircraft has to consider not just the needs of the customer, but also the downstream impact of the decision. For example, the flight may require sending the aircraft to Europe, which might make it difficult to get certain aircraft back to the maintenance shop on time. In addition, the trip may require a landing at night, which can help maintain a pilot's nighttime rating.

Example 5.2: Package delivery companies respond to customer pickup requests during the day. The problem involves routing vehicles in real-time (which is itself a combinatorially difficult problem), but it also requires thinking about the effect of decisions now on the future (will the routing place too many or too few trucks in a particular part of the city?).

Example 5.3: A portfolio optimization problem, which involves maximizing return while minimizing risk, may be formulated as a quadratic assignment problem while also having to consider the downstream impact of a particular allocation strategy.

These are examples of problems where the myopic problem requires the use of optimization algorithms that depend on particular structural properties. If the approximation of the value function destroys these properties, then the problem may become intractable. The need to approximate the expectation of the value function may do nothing more than add additional computational requirements to the myopic problem, but it also may destroy the structure entirely.

## 5.3 Using the post-decision optimality equations

In many applications it is computationally more convenient to work with the post-decision state variable (introduced in section 4.2). In this case, the transition function is written:

<!-- formula-not-decoded -->

The optimality equations for the post-decision state variable are given by:

<!-- formula-not-decoded -->

Since it can be notationally cumbersome writing out S M,x ( S t -1 , ω t , x t ), we may use S x t instead with the understanding that S x t is a function of ( S t -1 , ω t , x t ). Thus, we may write the optimality equations equivalently using:

<!-- formula-not-decoded -->

Now, assume that we have found a suitable approximation ¯ V x t ( s ) for the value function. We can find a decision x t by sampling the information that would be available before we make our decision, denoted by ω t = W t ( ω ), and then choosing x t by solving:

<!-- formula-not-decoded -->

Since we are using the post-decision state variable, we choose a single sample realization (the information that we would know anyway before making decision x t ).

If we are using a discrete, table-lookup representation of a value function, we would update it by computing:

<!-- formula-not-decoded -->

and then smoothing this into the current value function approximation:

<!-- formula-not-decoded -->

It is useful to contrast updating the value function using a post-decision state variable to that with a pre-decision state variable. The biggest difference is that we do not have to approximate the expectation. Thus, ˆ v n t in equation (5.5) is an approximation of the expectation of the value of being in state S t +1 conditioned on being in state S t (it is indexed

Step 0. For all t , initialize an approximation for the value function ¯ V 0 t ( S t ) for all time periods t and states S . Let n = 1.

Step 1. Set t = 0 and choose ω n .

Step 1a. Let ω = W t ( ω

<!-- formula-not-decoded -->

Step 1b. Compute S x,n = S M ( S x,n , ω n , x n ).

<!-- formula-not-decoded -->

Step 1c. Set t = t +1. If t &lt; T go to step 1 a .

Step 2. Use the results to update the approximation ¯ V x,n -1 t -1 ( S x t -1 ) for all t (using any of a variety of algorithms).

Step 3. Let n = n +1. If n &lt; N , go to step 1.

Figure 5.2: Forward dynamic programming using the post-decision state variable

by t because it is a function of information up through time t ). With the post-decision state variable, ˆ v n t uses a sample realization of information from time t , and is used to update the value function ¯ V t -1 ( S t -1 ), where the smoothing operation has the effect of approximating the expectation.

Another benefit of the post-decision state variable is that it is often simpler, and sometimes dramatically so. Consider the problem of assigning resources to tasks, where tasks become known during time period t at which time we have to design which resources should serve each task. Further assume that any tasks that are not served are discarded. We can let R R ta be the number of resources of type a that are available at time t , while R T tb is the number of tasks of type b that have to be served. The pre-decision state variable is R t = ( R R t , R T t ), while the post decision state variable describes only the modified resources.

Forward dynamic programming using the post-decision state variable is more elegant because it avoids the need to approximate the expectation. It also gives us another powerful device. Since the decision function 'sees' ¯ V x t ( S x t ) directly (rather than indirectly through the approximation of the expectation), we are able to control the structure of ¯ V x t ( S x t ). This feature is especially useful when the myopic problem max x t ∈X t C t ( S, x ) is an integer program or a difficult linear or nonlinear program that requires special structure.

## 5.4 Low-dimensional representations of value functions

Classical dynamic programming assumes a discrete representation of the value function. This means that for every state s ∈ S , we have to estimate a parameter v s that gives the value of being in state s . Forward dynamic programming may eliminate the loop over all states that is required in backward dynamic programming, but it does not solve the classic 'curse of dimensionality' in the state space. Forward dynamic programming focuses attention on

the states that we actually visit, but it also requires that we have some idea of the value of being in a state that we might visit (we need this estimate to conclude that we should not visit the state).

Virtually every large scale problem in approximate dynamic programming will focus on determining how to approximate the value function with a smaller number of parameters. In backward discrete dynamic programming, we have one parameter per state, and we want to avoid searching over a large number of states. In forward dynamic programming, we depend on Monte Carlo sampling, and the major issue is statistical error. It is simply easier to estimate a function that is characterized by fewer parameters.

In practice, approximating value functions always requires understanding the structure of the problem. However, there are general strategies that emerge. Below we discuss two of the most popular.

## 5.4.1 Aggregation

In the early days of dynamic programming, aggregation was quickly viewed as a way to provide a good approximation with a smaller state space, allowing the classical tools described in chapter 4 to be used. In the area of approximate dynamic programming, aggregation provides a mechanism for reducing statistical error. Thus, while it may introduce structural error, it actually makes a model more accurate by improving statistical robustness.

Example 5.4: The financial markets always face the challenge of estimating the forward earnings of a company. The earnings of a company will, of course, reflect its own management, its business relationships, and contracts with customers. Consider a company that makes disk drives for personal computers. The fortunes of the company will also tend to move with the broader market for personal computers, which itself will reflect the fortunes of the broader 'high tech' market.

Example 5.5: A trucking company that moves loads over long distances has to consider the profitability of assigning a driver to a particular load. The costs and revenues from moving the load are well understood. The load requires that the driver move to a particular location. We can estimate the value of the driver at the level of the 5-digit zip code of the location, or the 3-digit level, or at the level of a region (companies typically represent the United States using about 100 regions, which is far more aggregate than a 3-digit zip). Regions might be further aggregated into areas. The value of a driver in a location may also depend on his home domicile, which can also be represented at several levels of aggregation.

Aggregation is particularly powerful when there are no other structural properties to exploit (see examples). Aggregation is particularly powerful when the attributes of an asset are categorical. When this is the case, we typically find that the attribute space A does not have any metric to provide a 'distance' between two attributes. For example, we have

an intuitive sense that a disk drive company and a company that makes screens for laptops both serve the personal computer industry. We would expect that valuations in these two segments would be more closely correlated than they would be with a textile manufacturer. But we do not have a formal metric that measures this relationship.

In chapter 10 we investigate the statistics of aggregation in far greater detail.

## 5.4.2 Continuous value function approximations

In many problems, the state variable does include continuous elements. Our most obvious example is our resource vector R t = ( R ta ) a ∈A . R ta is the number of resources with attribute vector a . R ta may be either continuous (how much money is invested in a particular asset class, and how long has it been invested there) or discrete (the number of people with a particular skill set), but it is always numerical. Elements of the attribute vector a may also be numerical (how long has the asset been held in a portfolio, how old is the piece of equipment).

Numerical attributes offer the opportunity to build continuous functional representations. If we attempt to model the problem of allocating discrete resources (or discretized quantities of money), the state space explodes very quickly. Assume that we have N = ∑ a ∈A R ta assets to manage. Let R be the set of possible values of the vector R t . The size of R can be shown to be:

<!-- formula-not-decoded -->

Now consider what happens when we replace the value function V t ( R t ) with a linear approximation of the form:

<!-- formula-not-decoded -->

Instead of having to estimate |R| parameters ( V t ( R t ) for each value of R t ), we have only to estimate ¯ v ta , one for each value of a ∈ A . For some problems, this reduces the size of the problem from 10 100 or greater to one with several hundred or several thousand parameters.

Not all problems lend themselves to linear-in-the-resource approximations, but other approximations may emerge. Early in the development of dynamic programming, Bellman realized that a value function could be represented using statistical models and estimated with regression techniques. A particularly rich theory has developed for statistical models that are linear in the parameters. In fact, most references to 'linear models' in the approximate dynamic programming literature refer to 'linear in the parameters.'

To illustrate let ¯ V ( R | θ ) be a statistical model where the elements of R t are used to create

the independent variables, and θ is the set of parameters. For example, we might specify:

<!-- formula-not-decoded -->

If a ∈ A represents an asset class, then we might decide that we will add value if we include a nonlinear term that is a function of aggregated asset classes. Let

<!-- formula-not-decoded -->

represent an aggregation on A to the more aggregated set of attributes A g . We could then create another function:

<!-- formula-not-decoded -->

The variables R a , R 2 a and R 2 a g are referred to as features . The choice of useful features is highly problem dependent and usually requires considerable insight into the underlying problem.

The formulation and estimation of continuous value function approximations is one of the most powerful tools in approximate dynamic programming. The fundamentals of this approach is presented in considerably more depth in chapters 9 and 11.

## 5.4.3 Algorithmic issues

The design of an approximation strategy involves two algorithmic challenges. First, we have to be sure that our value function approximation does not unnecessarily complicate the solution of the myopic problem. We have to assume that the myopic problem is solvable in a reasonable period of time. If our myopic problem is a linear or nonlinear program, it is usually impossible to consider a value function that is of the discrete, table-lookup variety. If our myopic problem is continuously differentiable and concave, we do not want to introduce a potentially nonconcave value function. By contrast, if our myopic problem is a discrete scheduling problem that is being solved with a search heuristic, then table-lookup value functions can work just fine.

Once we have decided on the structure of our functional approximation, we have to devise an updating strategy. Value functions are basically statistical models that are updated using classical statistical techniques. However, it is very convenient when our updating algorithm is in a recursive form. A strategy that fits a set of parameters by using a sequence of observations using standard regression techniques may be too expensive for many applications.

## 5.5 Complex resource allocation problems

There are many complex resource allocation problems that can benefit from approximate dynamic programming. These problems might involve managing fleets of vehicles (trucks, boxcars, cargo aircraft, locomotives), distributing products through supply chains, managing people in the military (how many recruits should be assigned to a particular type of training, how many people should be assigned to a particular base), and managing stockpiles of commodities (energy commodities such as oil or coal, agricultural commodities such as corn or wheat).

For these problems, we have to determine not only what to do with a resource (where should the truck be sent, what training should the military recruit receive, when should the generating plant be turned off or on) but also how much we should do. Should we move 10 or 20 trucks into a region? Should we train 20 or 200 recruits to repair helicopters? Should we purchase 10 million or 20 million barrels of oil?

All of these problems can be viewed as managing a resource vector R t = ( R ta ) a ∈A where R ta is the quantity of resources with attribute a at time t . For many problems, R t constitutes the state variable, and the techniques we have introduced in this chapter would focus on estimating the value V t ( R t ) of being in state R t . However, it is often the case that it is more useful to focus on the marginal value of V t ( R t ) with respect to R t .

To illustrate, consider the problem of moving commodities (agricultural, coal, oil) from the point of production through various intermediate storage facilities. Commodities may be sold to intermediate distributors or moved closer to the final market. In addition to our standard notation, we let D ta be the random demand for commodities of type a at time t , and p ta be the random price. At time t , we have R t -1 ,a units of product of type a resulting from decisions (and information) from time t -1 with which to satisfy our demand in period t (remember: t is indexing the information content, not the availability of the product). Given outcome ω t = ( p ta ( ω ) , D ta ( ω )) of prices and quantities, we have to solve the problem at time t :

<!-- formula-not-decoded -->

subject to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A few notes are in order about this formulation. First, we assume that holding product in a location is captured by a 'do nothing' decision in the set D . Next, we assume that we

first have to satisfy the demand for an asset class a if it is available. We may 'transform' assets from, say, type a to type a ′ to satisfy demands for type a ′ , but demands can only be satisfied with assets of exactly the right type.

The optimization problem given by (5.12) subject to constraints (5.13)-(5.16) is a linear program that is concave in the resource constraint R t -1 . A simple way to get an estimate of the slope of this function with respect to R t -1 is to use the dual variable for constraint (5.13). Let ˆ v n ta be this dual variable. We know from the theory of linear programs that ˆ v n t = (ˆ v n ta ) a ∈A satisfies:

<!-- formula-not-decoded -->

Thus, ˆ v n t is a stochastic subgradient of ˆ V n t ( R n t -1 , ω n ) and therefore can be used to estimate a linear approximation. Of course, we have to perform our usual smoothing, so we estimate the linear approximation using:

<!-- formula-not-decoded -->

Our linear approximation requires up to |A| parameters rather than a parameter for every possible value of R t . For a large number of real-world problems, |A| may number in the hundreds or thousands, which is quite manageable in today's computing environment. The reduction in the size of the problem, from a computational perspective, is dramatic. The question of course is: how good are the solutions? Not surprisingly, this will be highly problem dependent.

If we do not have access to dual variables, we can resort to finite differences. Let e ta ′ ( a ) be a vector with element:

<!-- formula-not-decoded -->

We can then estimate our derivatives using:

<!-- formula-not-decoded -->

If the optimization problem (5.12) is a linear program, then it is usually possible to find these finite differences very quickly once the base problem has been solved. Once we obtain our estimate ˆ v for a sample ω n , we typically have to smooth these using our standard stochastic gradient logic:

<!-- formula-not-decoded -->

There is a wide array of resource allocation problems that can be quickly solved with a linear value function approximation. If the original (myopic) problem can be solved (the problem

may have tens of thousands of variables), then adding a linear approximation will never complicate the problem. The challenge is determining when a linear approximation works well.

This algorithmic strategy is illustrated in figure 5.3, where a single large problem has been divided into a series of smaller subproblems divided over space and time (time moves left to right). In each subproblem are a series of colored boxes. The boxes in the left column represent different types of resources, and the boxes on the right represent different types of demands. At any point in time, we have to solve what is known in linear programming as a transportation problem involving the assignment of resources to demands. But decisions at time t have to account for the value of resources in the future. Instead of using the value of the state in the future, we use the marginal value of additional resources in the future.

Using approximate dynamic programming with a linear value function approximation involves assigning resources to demands at a given point in time, using a value function approximation of the future. Figure 5.3a illustrates making decisions and stepping forward in time, effectively simulating the process of making decisions and modeling the physics of the problem.

Figure 5.3b then illustrates a backward pass, where we find the marginal value of resources in the future, and step backward in time, computing the marginal value at each subproblem as we step forward. This 'backward pass' produces the marginal value ˆ v ta of resources of type a at each point in time. These are then smoothed to produced an updated approximation.

## 5.6 Experimental issues

Once we have developed an approximation strategy, we have the problem of testing the results. Two issues tend to dominate the experimental side: rate of convergence and solution quality. The time required to execute a forward pass can range between microseconds and hours, depending on the underlying application. You may be able to assume that you can run millions of iterations, or you may have to work with a few dozen. The answer depends on both your problem and the setting. Are you able to run for a week on a large parallel processor to get the very best answer? Or do you need real-time response on a purchase order or to run a 'what if' simulation with a user waiting at the terminal?

Within our time constraints, we usually want the highest quality of solution possible. Here, the primary interest tends to be in the rate of convergence. Monte Carlo methods are extremely flexible, but the price of this flexibility is a slow rate of convergence. If we are estimating a parameter from statistical observations drawn from a stationary process, the quality of the solution tends to improve at a rate proportional to the square root of the number of observations. Unfortunately, we typically do not enjoy the luxury of stationarity. We usually face the problem of estimating a value function which is steadily increasing or decreasing (it might well be increasing for some states and decreasing for others) as the learning process evolves.

1.0 SENI DSHST

1.0

11.0 | SEN OSHST

1.0

SEN XINITI

8EN XINIT

8EN XINITI

8BN XINTT

&amp;EN XINTT

1.0

SEN COALC

SEN XINIT

1.0

1.0

SEN COALC

11.0 | SEN COALC

11.0 BENI COALCI

10

1.0 8EN 6H30T

SEN XINTT

SBN XINIT

11.0 8EN XINTT

8BN\_ 4H40T

1.0 SEN XINIT

8BN 4H40T

&amp;EN 4H4OT

8BN 4H30T

SEN 4440T

8BN XINTT

8EN\_ 4H30T

110

SEN DSHST

1.0 | 8BN XINIT

1.0 8EN DSHST

SBN FOREU

1.0

11.0

SEN 4H40T

8BN FORET

&amp;EN FORET

&amp;EN 4H40T

101

8BN XINTT

8EN XINIT

&amp;EN FORET

11.0 SEN XINIT

1.0

BEN XINIT

WILLMAR\_MN

WILLMAR\_MN

10 \_MSUPKCK103A-2-76

11.0 MSUPKCK103A/2.7

10 MMLMNTW1104-1-7

11.0 XDOLERE108A-27

10 XTACSPESOSA. 1.7

11.0 XTACSPESOSA 17

110 INOULERETORA 27

BRECKENRI MN

BRECKENRI\_MN

MLAW/GFD1094-3-7

1.0 MLAWGFD1094.3.7

8BN 6H301

1.0 | BEN 6H301

SEN EHSUT

1.0 88N 6H30T

SEN 6HSOL

1.0

BEN EHSOT

8BN FORET

1.0

BEN FORET

10 8EN XINTT

1.0 38N XINTT

1.0

BEN XINITI

SBN XINIT

11.0 88N XINIT

1.0 | 88N XINITI

1.0 88N COALCI

SEN COALC

SBN COALC

1.0 | 3EN COALC]

8BN 6H30T

1.0 | BEN 6H30T

SEN XINIT

BEN XINTT

8BN 4H401

0 | BEN 4440T

SEN 4H4OT

BEN 4440T

SEN 4H30T

8BN XINTT

BEN 4H30T

SBN DSH9T

88N XINIT

88N OSHST

SIOCITY IA

SIOCITY IA

WILLMAR MN

WILLMAR\_MN

910 MSLPKCK109A 23/

1110 | XOULBRE1086 2-8

INDUL

NDNXE

5.3a Forward pass simulating a policy.

<!-- image -->

5.3b Backward pass communicating gradients.

<!-- image -->

Figure 5.3: Illustration of forward and backward pass in an asset allocation problem, using a linear approximation of the value of an asset in the future.

WILLMAR MN

WILLMAR MN

11.0 SEN OSHST

1.0 SEN DSHST

10 8BN 6H401

0 BEN SHAOT

BEN COALC

COALC

BEN COALC

6H30T

SEN XINI

PLSOSA. 9

## 5.6.1 The initialization problem

All of the strategies above depend on setting an initial estimate ¯ V 0 for the value function approximation. In chapter 4, the initialization of the value function did not affect our ability to find the optimal solution; it only affected the rate of convergence. In approximate dynamic programming, it is often the case that the value function approximation ¯ V n -1 used in iteration n affects the choice of the action taken and therefore the next state visited.

The problem is illustrated in figure 5.4 for our nomadic trucker. Here, the trucker depends on the estimate of the value of being in a 'state.' If we initialize the value function to a low value, we will quickly obtain higher estimates for the states that we visit, which encourages decisions that allow us to return to these states. If the initial estimates are too high, the process will favor decisions that take us to states that we have visited the least (since these will tend to be the highest). If the state space (attribute space) is not too large, we will obtain the best answer using a high initial estimate. But if the state space and action spaces are very large, we will be endlessly exploring states that we have never visited before.

## 5.6.2 Sampling strategies

There are two strategies for sampling states in approximate dynamic programming. Up to now, we have used asynchronous approximate dynamic programming, which implies that we only sample the states that we visit from the decision produced by using the approximate value function. An alternate strategy is synchronous approximate dynamic programming, where we sample all states at every iteration, as illustrated in figure 5.5.

We quickly recognize that we cannot loop over all the states for problems with large state spaces. If we could do this, we could use classical backward dynamic programming. There are, however, two uses of synchronous approximate dynamic programming. The first arises in problems where we do not have a mathematical model of our underlying information process, and depend, instead, on our ability to sample the forward trajectory from an exogenous source. For such problems, we cannot compute the expectation of the value function in the future and need to depend on samples from an exogenous source.

The second use is motivated by theoretical issues. Convergence proofs are available for table-lookup versions of synchronous algorithms, where we can exploit the assumption that all states are visited infinitely often. These requirements are not satisfied when we use asynchronous algorithms.

## 5.6.3 Exploration vs. exploitation

Closely related to the initialization problem is the classical issue of 'exploration' vs. 'exploitation.' Exploration implies visiting states in order to more accurately estimate the value of being in the state, regardless of whether that was a profitable decision to make. Exploitation means taking our best estimate of the contributions and values, and making

ER 5. INTRODUCTION TO APPROXIMATE DYNAMIC PROGRAN

5.4a: Low initial estimate of the value function.

5.4a: Low initial estimate of the value function.

<!-- image -->

5.4b: High initial estimate of the value function.

4: The effect of value function initialization on search process. Case

5.4b: High initial estimate of the value function.

<!-- image -->

Figure 5.4: The effect of value function initialization on search process. Case (a) uses a low initial estimate, and produces limited exploration; Case (b) uses a high initial estimate, which forces exploration of the entire state space.

decisions that seem to be the best given the information we have (we are 'exploiting' our estimates of the value function). Of course, exploiting our value function to visit a state is also a form of exploration, but the literature typically uses the term 'exploration' to mean that we are visiting a state without regard to the estimated contribution from visiting the state.

Determining the right balance between exploring states just to estimate their values, and using current value functions to visit the states that appear to be the most profitable, represents one of the great unsolved problems in approximate dynamic programming. If we only visit states that appear to be the most profitable given the current estimates (a pure exploitation strategy) we run the risk of landing in local optima unless the problem has special properties. There are strategies that help minimize the likelihood of being trapped

Step 0. For all t , initialize an approximation for the value function ¯ V 0 t ( S t ) for all states S . Let n = 1 and set t = 0.

Step 1. For time period t do:

Step 2. For each state s ∈ S , do:

Step 2a. Solve:

<!-- formula-not-decoded -->

Step 2b. Use the results to update the approximation ¯ V n -1 t ( s ).

Step 3. Let n := n +1. If n &lt; N , go to step 1.

Figure 5.5: Synchronous approximate dynamic programming

in a local optima but at a cost of very slow convergence. Finding good strategies with fairly fast convergence appears to depend on taking advantage of natural problem structure.

## 5.6.4 Evaluating policies

A common question is whether a policy X π 1 is better than another policy X π 2 . Assume we are facing a finite horizon problem that can be represented by the objective function:

<!-- formula-not-decoded -->

Since we cannot compute the expectation, we might choose a sample ˆ Ω ⊆ Ω and then calculate:

<!-- formula-not-decoded -->

where ˆ p ( ω ) is the probability of the outcome ω ∈ ˆ Ω. If we have chosen the outcomes in ˆ Ω at random from within Ω, we would use:

<!-- formula-not-decoded -->

Alternatively, we may choose ˆ Ω so that we control the types of outcomes in Ω that are represented in ˆ Ω. Such sampling strategies fall under names such as stratified sampling or

importance sampling. They require that we compute the sample probability distribution ˆ p to reflect the proper frequency of an outcome ω ∈ ˆ Ω within the larger sample space Ω.

The choice of the size of ˆ Ω should be based on a statistical analysis of ¯ F π . For a given policy π , it is possible to compute the variance of ¯ F π ( ˆ Ω) using:

<!-- formula-not-decoded -->

In most applications, it is reasonable to assume that (¯ σ π ) 2 is independent of the policy, allowing us to use a single policy to estimate the variance of our estimate. If we treat the estimates of ¯ F π 1 and ¯ F π 2 as independent random variables the variance of the difference is 2(¯ σ π ) 2 . We can then compute a confidence interval on the difference using:

<!-- formula-not-decoded -->

where z α/ 2 is the standard normal deviate for a confidence level α .

Typically, we can obtain a much tighter confidence interval by using the same sample ˆ Ω to test both policies. In this case, ¯ F π 1 and ¯ F π 2 will not be independent and may, in fact, be highly correlated (in a way we can use to our benefit). Instead of computing an estimate of the variance of the value of each policy, we should compute a sample realization of the difference:

<!-- formula-not-decoded -->

from which we can compute an estimate of the difference:

<!-- formula-not-decoded -->

When comparing two policies, it is very important to compute the variance of the estimate of the difference to see if it is statistically significant. If we evaluate each policy using a different set of random outcomes (say, ω 1 and ω 2 ), the variance of the difference would be given by:

<!-- formula-not-decoded -->

This is generally not the best way to estimate the variance of the difference between two policies. It is typically better to evaluate two policies using the same random sample for each

policy. In this case, ˆ F π 1 ( ω ) and ˆ F π 2 ( ω ) are usually correlated, which means the variance would be:

<!-- formula-not-decoded -->

The covariance is typically positive, so this estimate of the variance will be smaller (and possibly much smaller). One way to estimate the variance is to compute ˆ ∆ π 1 ,π 2 ( ω ) for each ω ∈ ˆ Ω and then compute:

<!-- formula-not-decoded -->

In general, ¯ σ π 1 ,π 2 will be much smaller than 2(¯ σ π ) 2 which we would obtain if we chose independent estimates.

For some large scale experiments, it will be necessary to perform comparisons using a single sample realization ω . In fact, this is the strategy that would typically be used if we were solving a steady state problem. However, the strategy can be used for any problem where the horizon is sufficiently long that the variance of an estimate of the objective function is not too large. We again emphasize that the variance of the difference in the estimates of the contribution of two different policies may be much smaller than would be expected if we used multiple sample realizations to compute the variance of an estimate of the value of a policy.

## 5.7 Dynamic programming with missing or incomplete models

Some physical systems are so complex that we cannot describe them with mathematical models, but we are able to observe behaviors directly. In the reinforcement learning community, these are sometimes referred to as direct models; solving problems without an explicit model is referred to as model-free dynamic programming.

There are three key mathematical models used in dynamic programming. The first is a model of the exogenous information process. If we are studying a process where the randomness represents the Poisson arrival of demands at a rate that is easy to measure, then it is not hard to construct a mathematical model of the demand process. If the problem is the modeling of interest rates of different countries, we face a problem with complex correlations between interest rate streams, and across time.

The second type of model is the transition function, which specifies which state we transition to given a state, action, and information. It is possible that even if we could measure the state where we know both the action and the information, we still might not know exactly what state we will land in (see example).

Example 5.6: A truck driver is assigned to move a load from New York to Chicago. He is limited to driving 10 hours a day. The trip requires 12 hours of driving. The driver could drive 10 hours, rest and then deliver the load the next day after driving two hours, leaving him with eight more hours on his 'clock.' But the driver may decide instead to split the trip, moving seven hours the first day and five the second, which changes what the driver can do the second day. Since we are not able to mathematically represent the driver's behavior, we do not have an explicit model of the transition function.

In some cases, we can approximate the transition function and view discrepancies between what we expected to happen and what actually happens as simply missing information on the state variable. But as our example illustrates, our basic model may not be accurate, which limits our ability to test policies such as increasing the number of hours a driver can move in a day.

The third model is the decision function itself. We think we are capturing all the important contributions and constraints in our objective function, but this is simply not always going to be the case. There are a number of examples where a human operator will make decisions based on information not available to the model. Such outside guidance is known as a supervisor and training an approximate dynamic program in the presence of such a source of information is known as supervisory learning .

The term 'model free' is often used to mean that we do not have a one-step transition matrix, but where we may have a transition function. This means that what we are missing is a model of the information process. For example, we may not know the set of outcomes, or the probability of an outcome. This is typical of data that is coming from a physical process. Such problems may have small state spaces, allowing us to use algorithms that require sampling all the states at every iteration. Solving problems this way is generally referred to as model-free approximate dynamic programming.

## 5.8 Relationship to reinforcement learning

A different path of development for dynamic programming has emerged from within the artificial intelligence community that has found the dynamic programming paradigm a useful framework for thinking about mechanisms for learning human behavior. Consider the problem of training a robot to execute a series of steps to insert a part in a machine or teaching a machine to play a game of backgammon. A rule (or policy) may specify that when the system is in state s , we should make decision d . A set of state/decision pairs constitutes a policy for completing the task. From an initial state, we can simulate the policy to see if the result is successful or not. We can let ¯ V π ( s ) be the probability that the policy will ultimately be successful. Over the iterations, we steadily learn which state/action pairs ultimately produce a successful game.

In this setting, a decision function that chooses a decision given the state is known as an actor . The process that determines the contribution (cost or reward) from a decision is known as the critic , from which we can compute a value function. The interaction of making decisions and updating the value function is referred to as an actor-critic framework. The slight change in vocabulary brings out the observation that the techniques of approximate dynamic programming closely mimic human behavior. This is especially true when we drop any notion of costs or contributions and simply work in terms of succeeding (or winning) and failing (or losing).

Research in reinforcement learning can be stated mathematically using the same notation we have used up to now, but the types of applications are often quite different. A popular class of applications are games where a number of steps have to be taken before learning if we won the game or not. We refer to these problems as applications with deeply nested strategies which is to say, we have to explore a very deep tree before learning what path is the best. The only reward received is 1 if we win and 0 otherwise, which means the value function v ( s ) can be interpreted as the probability of winning if we are in state s . Another application is training a system to fly a plane to a particular destination, or getting a robot to move its arm to a particular position.

It is particularly difficult training an approximate dynamic programming algorithm in these settings without some advance information. Without either a rough approximation of a value function, or guidance of what decisions to make in each state, the system has to wander aimlessly about the state/action space until it discovers (almost by accident) a good strategy. For example, when training a system to play chess, researchers used an extensive database of past games to guide the training of the system.

## 5.9 But does it work?

The technique of stepping forward through time using Monte Carlo sampling is a powerful strategy, but it effectively replaces the challenge of looping over all possible states with the problem of statistically estimating the value of being in 'important states.' Furthermore, it is not enough just to get reasonable estimates for the value of being in a state. We have to get reasonable estimates for the value of being in states we might want to visit.

As of this writing, formal proofs of convergence are limited to a small number of very specialized algorithms. Compounding this lack of proofs are experimental work that illustrate cases in which the methods simply do not work. What has emerged from the various laboratories doing experimental work are two themes:

- The functional form of an approximation has to reasonably capture the true value function.
- For large problems, it is essential to exploit the structure of the problem so that a visit to one state s ′ provides improved estimates of the value of visiting a large number of other states.

For example, discrete table-lookup functions will always capture the general shape of a (discrete) value function, but it does little to exploit what we have learned from visiting one state in terms of updated estimates of the value of visiting other states. As a result, it is quite easy to design an approximate dynamic programming strategy (using, for example, a table-lookup value function) that either does not work at all, or provides a suboptimal solution that is well below the optimal solution.

It is our belief that general purpose results in approximate dynamic programming will be few and far between. Instead, we believe that most results will involve taking advantage of the structure of a particular problem class. Identifying a value function approximation, along with a sampling and updating strategy, that produces high quality solution represents a major contribution to the field in which the problem arises. The best we can offer in a general textbook on the field is to provide guiding principles and general tools, allowing domain experts to devise the best possible solution. We suspect that an ADP strategy applied to a problem context is probably a patentable invention.

## 5.10 Bibliographic notes

```
Introduced TD Learning: Sutton (1988) Survey of reinforcement learning Leslie Pack Kaelbling & Moore (1996) Review articles: Tsitsiklis & Van Roy (1996), Van Roy (2001) Books: Bertsekas & Tsitsiklis (1996), Sutton & Barto (1998), White & Sofge (1992), Si et al. (2004), Pflug - stochastic optimization Pflug (1996) Functional approximation: Bellman & Dreyfus (1959) Werbos: Werbos (1987), Werbos (1990), Werbos (1992 b ), Werbos (1992 c ), Werbos (1992 a Partially observable MDP's: White (1991)
```

## Exercises

- 5.1) You are holding an asset which has an initial market price p 0 = 100. The price evolves each time period according to p t = p t -1 + /epsilon1 t where /epsilon1 t is the discrete uniform distribution taking values from -5 to 5 (each with probability 1 / 11). You have 10 opportunities to sell the asset, and you must sell it at the end of the 10 th time period. Use the approximate dynamic programming with a pre-decision state variable to estimate the optimal value function, and give your estimate of the optimal policy (that is, at each price, should we sell or hold). Use 1000 forward passes, and use 10 random samples to approximate the expectation.
- 5.2) Repeat exercise (5.1) assuming that /epsilon1 t has a uniform distribution from -6 to 4. After

)

training the value function for 1000 iterations, run 100 samples (holding the value function fixed) and determine when the model decided to sell the asset. Plot the distribution of times that the asset was held over these 100 realizations.

- 5.3) Here we are going to solve a variant of the asset selling problem using a post-decision state variable. We assume we are holding a real asset and we are responding to a series of offers. Let ˆ p t be the t th offer, which is uniformly distributed between 500 and 600 (all prices are in thousands of dollars). We also assume that each offer is independent of all prior offers. You are willing to consider up to 10 offers, and your goal is to get the highest possible price. If you have not accepted the first nine offers, you must accept the 10 th offer.
- a) Write out the decision function you would use in an approximate dynamic programming algorithm in terms of a Monte Carlo sample of the latest price and a current estimate of the value function approximation.
- b) Use the knowledge that p t is uniform between 500 and 600 and derive the exact value of holding the asset after each offer.
- c) Write out the updating equations (for the value function) you would use after solving the decision problem for the t th offer using Monte Carlo sampling.
- d) Implement an approximate dynamic programming algorithm using synchronous state sampling (you sample both states at every iteration). Using 100 iterations, write out your estimates of the value of being in each state immediately after each offer.
- e) From your value functions, infer a decision rule of the form 'sell if the price is greater than ¯ p t '.
- 5.4) We are going to use approximate dynamic programming to estimate:

<!-- formula-not-decoded -->

where R t is a random variable that is uniformly distributed between 0 and 100 and γ = . 7. We assume that R t is independent of prior history. We can think of this as a single state Markov decision process with no decisions.

- a) Using the fact that E R t = 50, give the exact value for F 20 .
- b) Let ˆ v n t = ∑ T t ′ = t γ t ′ R t ′ ( ω n ) where ω n represents the n th sample path, and R t ′ ( ω n ) is the random realization of the t th random variable R t for the n th sample path. Show that ˆ v n t = R n t + γ ˆ v n t +1 which means that ˆ v n 0 is a sample realization of R T .
- c) Propose an approximate dynamic programming algorithm to estimate F T . Give the value function updating equation, using a stepsize α n = 1 /n for iteration n .
- d) Perform 100 iterations of the approximate dynamic programming algorithm to produce an estimate of F 20 . How does this compare to the true value?

- e) Repeat part (d) 10 times, but now use a discount factor of 0.7. Average the results to obtain an averaged estimate. Now use these 10 calculations to produce an estimate of the standard deviation of your average.
- f) From your answer to part (e), estimate how many iterations would be required to obtain an estimate where 95 percent confidence bounds would be within two percent of the true number.
3. 5.5) Consider a batch replenishment problem where we satisfy demand in a period from the available inventory at the beginning of the period and then order more inventory at the end of the period. Define both the pre- and post-decision state variables, and write the pre-decision and post-decision forms of the transition equations.
4. 5.6) A mutual fund has to maintain a certain amount of cash on hand for redemptions. The cost of adding funds to the cash reserve is $250 plus $0.005 per dollar transferred into the reserve. The demand for cash redemptions is uniformly distributed between $5,000 and $20,000 per day. The mutual fund manager has been using a policy of transferring in $500,000 whenever the cash reserve goes below $250,000. He thinks he can lower his transaction costs by transferring in $650,000 whenever the cash reserve goes below $250,000. However, he pays an opportunity cost of $0.00005 per dollar per day for cash in the reserve account.
- a) Use a spreadsheet to set up a simulation over 1000 days to estimate total transaction costs plus opportunity costs for the policy of transferring in $500,000, assuming that you start with $500,000 in the account. Perform this simulation 10 times, and compute the sample average and variance of these observations. Then compute the variance and standard deviation of your sample average.
- b) Repeat (a) but now test the policy of transferring in $650,000. Repeat this 10 times and compute the sample average and the standard deviation of this average. Use a new set of observations of the demands for capital.
- c) Can you conclude which poicy is better at a 95 percent confidence level? Estimate the number of observations of each that you think you would need to conclude there is a difference at a 95 percent confidence level.
- d) How does your answer to (d) change if instead of using a fresh set of observations for each policy, you use the same set of random demands for each policy?
9. 5.7) Let M be the dynamic programming 'max' operator:

<!-- formula-not-decoded -->

Let:

<!-- formula-not-decoded -->

and define a policy π ( v ) using:

<!-- formula-not-decoded -->

Let v π be the value of this policy. Show that:

<!-- formula-not-decoded -->

where e is a vector of 1's.

- 5.8) As a broker, you sell shares of a company you think your customers will buy. Each day, you start with s t -1 shares left over from the previous day, and then buyers for the brokerage add r t new shares to the pool of shares you can sell that day ( r t fluctuates from day to day). Each day, customer i ∈ I calls in asking to purchase q ti units at price p ti , but you do not confirm orders until the end of the day. Your challenge is to determine x t which is the number of shares you wish to sell to the market on that day. You will collect all the orders. Then, at the end of the day, you must choose x t , which you will then allocate among the customers willing to pay the highest price. Any shares you do not sell on one day are available for sale the next day. Let p t = ( p ti ) i ∈I and q t = ( q ti ) i ∈I be the vectors of price/quantity offers from all the customers on a given day.
- a) What is the exogenous information process for this system? What is a history for the process?
- b) Give a general definition of a state variable. Is s t a valid state variable given your definition?
- c) Set up the optimality equations for this problem using the post-decision state variable. Be precise!
- d) Set up the optimality equations for this problem using the pre-decision state variable. Contrast the two formulations from a computational perspective.

## Chapter 6

## Stochastic approximation methods

Stochastic approximation methods are the foundation of most approximate dynamic programming algorithms. They represent a technique whose simplicity is matched only by the depth and richness of the theory that supports them. Particularly attractive from a practical perspective is the ease with which they provide us with the means for solving, at least approximately, problems with considerable complexity. There is a price for this generality, but for many complex problems, we gladly pay it in return for a strategy that can provide insights into problems that would otherwise seem completely intractable.

Our presentation of stochastic gradient methods is broken into two chapters. In this chapter, we focus on the fundamental problem of estimating the mean of a random variable. This is particularly important in dynamic programming because we use these methods to estimate the value of being in a state. The same techniques are also used to estimate the parameters of an exogenous random variable such as demands or prices.

The basic problem can be stated as one of optimizing the expectation of a function, which we can state as:

<!-- formula-not-decoded -->

Here, θ is a vector of parameters that we are trying to estimate. An element of θ may be θ s which is the value of being in state s . Alternatively, we may have an asset with attribute a , where θ a is the value of the asset. W may be a vector of random variables representing information that becomes known after we make the decision θ . For example, we may make a decision based on an estimate of the value of a state, and then obtain a random observation of the value of the state which helps us improve the estimate.

The general stochastic optimization problem in equation (6.1) also arises in many settings that have nothing to do with parameter estimation. For example, we may need to make a decision to allocate resources, denoted by x . After we allocate the resources, we then learn about the demands for these resources (captured by a random variable D ). Because the demands become known after we decide on our allocation, we face the problem of solving E { F ( x, D ) } .

We generally assume that we cannot compute E { F ( θ, W ) } , but we still have to optimize over θ . In practice, we may not even known the probability distribution of W , but we assume that we do have access to observations of W , either from a physical process or from the use of Monte Carlo methods (where we assume that we are randomly generating realizations from a mathematical model). This behavior is illustrated in the examples.

Example 6.1: An information technology group uses planning tools to estimate the man-hours of programming time required to complete a project. Based on this estimate, the group then allocates programming resources to complete the project. After the project, the group can compute how many man-hours were actually needed to complete it, which can be used to estimate the distribution of the error between the estimate of what was required and what was actually required.

Example 6.2: A venture capital firm has to estimate what it can expect for a return if it invests in an emerging technology company. Since these companies are privately held, the firm can only learn the return by agreeing to invest in the company.

Example 6.3: A project manager has to estimate how much money will be needed to design and build a new network router. The company then has to allocate funds for the project. If the funds allocated are insufficient to complete the project, it risks requiring additional time (and money) while additional funds are allocated or the cancellation of the project altogether (with a complete loss of the investment).

Example 6.4: An electric power company has to maintain a set of spare components in the event of failure. These components can require a year or two to build and are quite expensive, so it is necessary to have a reasonable number of spares with the right features and in the right location to respond to emergencies.

Example 6.5: A business jet company is trying to estimate the demand for its services. Each week, it uses the actual demands to update its estimate of the average demand. The goal is to minimize the deviation between the estimate of the demand and the actual.

This chapter introduces the elegant and powerful theory of stochastic approximation methods (also known as stochastic gradient algorithms). These are properly viewed as the algorithm of last resort in stochastic optimization, but for the vast majority of practical applications, they are the only option. As with other chapters in this volume, the core algorithms are separated from the proofs of convergence. However, students are encouraged to delve into these proofs to gain an appreciation of the supporting theory.

## 6.1 A stochastic gradient algorithm

Our basic problem is to optimize a function that also depends on a random variable. This can be stated as:

<!-- formula-not-decoded -->

where Θ is the set of allowable values for θ (many estimation problems are unconstrained). For example, assume that we are trying to find the mean µ of a random variable R . We wish to find a number θ that produces the smallest squared error between the estimate of the mean θ and that of a particular sample. This can be stated as:

<!-- formula-not-decoded -->

If we want to optimize (6.3), we could take the derivative and set it equal to zero (for a simple, unconstrained, continuously differentiable problem such as ours). However, let us assume that we cannot take the expectation easily. Instead, we can choose a sample observation of the random variable R that is represented by R ( ω ). A sample of our function is now:

<!-- formula-not-decoded -->

Let g ( θ ) be the gradient of F ( θ ) = E F ( θ, R ( ω )), and let g ( θ, ω ) be the sample gradient, taken when R = R ( ω ). In our example, clearly:

<!-- formula-not-decoded -->

We call g ( θ, ω ) a stochastic gradient because, obviously, it is a gradient and it is stochastic.

What can we do with our stochastic gradient? If we had an exact gradient, we could use the standard optimization sequence:

<!-- formula-not-decoded -->

where α n is the stepsize that is typically chosen so that θ = ¯ θ n minimizes the objective function (remember that we are using the negative gradient because we are minimizing). However, we are using a stochastic gradient, and because of this, we cannot use sophisticated logic such as finding the best stepsize. Part of the problem is that a stochastic gradient can even point away from the optimal solution such that any positive stepsize actually makes the solution worse. For example, consider the problem of estimating the mean of a random variable R where ¯ θ n is our estimate after n iterations. Assume the mean is 10 and our current

estimate of the mean is ¯ θ n -1 = 7. If we now observe R n = 3 with α n = . 1, our update would be:

<!-- formula-not-decoded -->

which means our estimate has moved even further from the true value. This is the world we live in when we depend on stochastic gradients.

Remark: Many authors will write equation (6.6) in the form:

<!-- formula-not-decoded -->

We use the form in (6.6) to be consistent with our time indexing. We index ¯ θ n on the left hand side of (6.6) because the right hand side has information from iteration n whereas equation (6.7) using an indexing style based on when it is used ( ¯ θ n +1 is computed at the end of iteration n but is used in iteration n +1). It is often the case that time t is also our iteration counter, and so it helps to be consistent with our time indexing notation.

If we put (6.5) and (6.6) together, we can write our updating algorithm in the following two forms:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (6.7) writes the updating in the standard form of a stochastic gradient algorithm. The form given in equation (6.8) is normally referred to as exponential smoothing , which is widely used in the demand forecasting literature. For problems where we are trying to estimate the mean of a random variable, we will typically require that 0 ≤ α n ≤ 1, because the units of the gradient and the units of the decision variable are the same.

Throughout our presentation, we use the index n to count the number of times we are updating our estimate of a parameter. In a number of settings, we do not obtain a new observation of every parameter

There are many applications where the units of the gradient, and the units of the decision variable, are different. For example, consider the problem of ordering a quantity of product x to satisfy a random demand D . We pay an overage cost c o for each unit that we order over the demand, and an underage cost c u for each unit of demand that is unsatisfied. The resulting problem requires that we solve:

<!-- formula-not-decoded -->

In this case, the gradient is either c o or -c u depending on whether x &gt; D or x &lt; D . The units of c o and c u are in dollars, while the units of x is a quantity of product. In this situation, the stepsize will generally not be between 0 and 1. In some cases, however, the units may be quite different. This situation would arise if we were minimizing a function F ( θ, W ) which represented, for example, the profits of an activity. In this case, the gradient will be in units of dollars, whereas the decision variable x might be in the units of the asset being managed. In such cases, it is necessary for the stepsize to be properly scaled.

Returning to our original problem of estimating the mean, we assume when running a stochastic gradient algorithm that ¯ θ 0 is an initial guess, and that R ( ω 1 ) is our first observation. If our stepsize sequence uses an initial stepsize α 1 = 1, then we do not need an initial guess. Smaller initial stepsizes would only make sense if we had access to an initial guess, and in this case, the stepsize should reflect the confidence in our original estimate (for example, we might be warmstarting an algorithm from a previous iteration).

We can evaluate our performance using a mean squared statistical measure. If we have an initial estimate ¯ θ 0 , we would use:

<!-- formula-not-decoded -->

However, it is often the case that the sequence of random variables R ( ω n ) is nonstationary. In this case, estimating the mean squared error is similar to our problem of estimating the mean of the random variable R , in which case we should use a standard stochastic gradient (smoothing) expression of the form:

<!-- formula-not-decoded -->

where β n is another stepsize sequence (which could be the same as α n ).

## 6.2 Sampling random variables

Our stochastic gradient algorithm depends on having access to a sequence of sample realizations of our random variable. How this is done depends on the setting. There are three ways that we can obtain random samples:

- 1) The real world - Random realizations may come from real physical processes. For example, we may be trying to estimate average demand using sequences of actual demands. We may also be trying to estimate prices, costs, travel times, or other system parameters from real observations.
- 2) Computer simulations - The realization may be a calculation from a computer simulation of a complex process. The simulation may be of a physical system such as a

supply chain or an asset allocation model. Some simulation models can require extensive calculations (a single sample realization could take hours or days on a computer).

- 3) Sampling from a known distribution - This is the easiest way to sample a random variable. We can use existing tools available in most software languages or spreadsheet packages to generate samples from standard probability distributions. These tools can be used to generate many thousands of random observations extremely quickly.

The ability of stochastic gradient algorithms to work with real data (or data coming from a complex computer simulation) means that we can solve problems without actually knowing the underlying probability distribution. We may have multiple random variables that exhibit complex correlations. For example, observations of interest rates or currency exchange rates can exhibit complex interdependencies that are difficult to estimate. As a result, it is possible that we cannot compute an expectation because we do not know the underlying probability distribution. That we can still solve such a problem (approximately) greatly expands the scope of problems that we can address.

When we do have a probability model describing our information process, we can use the power of computers to generate random observations from a distribution using a process that is generally referred to as Monte Carlo sampling. Although most software tools come with functions to generate observations from major distributions, it is often necessary to customize tools to handle more general distributions. When this is the case, we often find ourselves depending on sampling techniques that depend on functions for generating random variables that are uniformly distributed between 0 and 1, which we denote by U , or for generating random variables that are normally distributed with mean 0 and variance 1 (the standard normal distribution), which we denote by Z .

If we need a random variable X that is uniformly distributed between a and b , we use the internal random number generator to produce U and then calculate:

<!-- formula-not-decoded -->

Similarly, if we wish to randomly generate a random variable that is normally distributed with mean µ and variance σ 2 , then we use the internal random number generator to compute a standard normal deviate Z and then perform the transformation:

<!-- formula-not-decoded -->

If we need a random variable X with a cumulative distribution function F ( θ ) = Prob ( θ ≤ x ), then it is possible to show that F ( θ ) is a random variable with a uniform distribution between 0 and 1. This implies that F -1 ( U ) is a random variable that has the same distribution as X , which means that we can generate observations of X using:

<!-- formula-not-decoded -->

For example, consider the case of an exponential density function γe -γx with cumulative distribution function 1 -e -γx . Setting U = 1 -e -γx and solving for x gives:

<!-- formula-not-decoded -->

Since 1 -U is also uniformly distributed between 0 and 1, we can use simply:

<!-- formula-not-decoded -->

There is an extensive literature on generating Monte Carlo random variables that extends well beyond the scope of this book. This section provides only a brief introduction.

## 6.3 Some stepsize recipes

One of the challenges in Monte Carlo methods is finding the stepsize α n . A standard technique in deterministic problems (of the continuously differentiable variety) is to find the value of α n so that ¯ θ n gives the smallest possible objective function value (among all possible values of α ). For a deterministic problem, this is generally not too hard. For a stochastic problem, it means calculating the objective function, which involves computing an expectation. For most applications, expectations are computationally intractable which makes it impossible to find an optimal stepsize.

Below, we start with a general discussion of stepsize rules. Following this, we provide a number of examples of different stepsize rules, divided between deterministic rules (section 6.3.2) and adaptive (or stochastic) rules (section 6.3.3).

## 6.3.1 Properties for convergence

The theory for proving convergence of stochastic gradient algorithms was first developed in the early 1950's and has matured considerably since then. However, all the proofs require three basic conditions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (6.11) obviously requires that the stepsizes be nonnegative. The most important requirement is (6.12), which states that the infinite sum of stepsizes must be infinite. If this condition did not hold, the algorithm might stall prematurely. Finally, condition 6.13 requires that the infinite sum of the squares of the stepsizes be finite. This condition, in effect, requires that the stepsize sequence converge 'reasonably quickly.' A good intuitive justification for this condition is that it guarantees that the variance of our estimate of the optimal solution goes to zero in the limit. Sections 6.8.2 and 6.8.3 illustrate two proof techniques that both lead to these requirements on the stepsize. Fifty years of research in this area has not been able to relax them.

Conditions (6.12) and (6.13) effectively require that the stepsizes decline according to an arithmetic sequence such as:

<!-- formula-not-decoded -->

This rule has an interesting property. If we substitute this rule into the updating equation (6.6), we get:

<!-- formula-not-decoded -->

If we continue this exercise, we finally obtain:

<!-- formula-not-decoded -->

Of course, we have a nice name for equation (6.16): it is called a sample average. And we are all aware that in general (some modest technical conditions required) as n →∞ , ¯ θ n will converge (in some sense) to the mean of our random variable R . What is nice about equation (6.7) is that it is very easy to use (actually, much easier than equation (6.16)). Also, it lends itself nicely to adaptive estimation, where we may not know the sample size in advance.

The issue of the rate at which the stepsizes decrease is of considerable practical importance. Consider, for example, the stepsize sequence:

<!-- formula-not-decoded -->

which is a geometrically decreasing progression. Here, the stepsizes would decrease so quickly that we might never reach the final solution.

Figure 6.1: Illustration of poor convergence of 1 /n stepsize rule.

<!-- image -->

Surprisingly, the '1 /n ' stepsize formula tends not to work in practice because it drops to zero too quickly when applied to approximate dynamic programming applications. The reason is that we are usually updating the value function using biased estimates. For example, consider the updating expression we used for the post-decision state variable given in equation (5.10), which we repeat here for convenience:

<!-- formula-not-decoded -->

ˆ v n t is our sample observation of an estimate of the value of being in state S t -1 , which we then smooth into the current approximation ¯ V n -1 t -1 ( S t -1 ). If ˆ v n t were an unbiased estimate of the true value, then a stepsize of 1 /n would be the best we could do. However, ˆ v n t depends on ¯ V x,n -1 t ( S t ), which is an imperfect estimate of the value function for time t . What typically happens is that the value functions undergo a transient learning phase, as illustrated in figure 6.2. As a result of this learning phase, ˆ v n t is biased. When this happens, the 1 /n stepsize puts too much weight on observations in the beginning (when our estimate of ¯ V x,n -1 t ( S t ) is at its worst).

The remainder of this section discusses ways to overcome this basic problem.

## 6.3.2 Deterministic stepsizes

Deterministic stepsizes are purely a function of the n and do not depend on the data. Below are the most popular rules that we have seen.

Figure 6.2: Illustration of geometric convergence of value function from value iteration.

<!-- image -->

## Constant stepsizes

Constant stepsizes are popular when we are estimating not one but many parameters (these can easily number in the thousands or millions). In these cases, no single rule is going to be right for all of the parameters and there is enough noise that any reasonable stepsize rule will work well. Constant stepsizes are easy to code (no memory requirements) and, in particular, easy to tune (there is only one parameter). Perhaps the biggest point in their favor is that we simply may not know the rate of convergence, which means that we run the risk with a declining stepsize rule of allowing the stepsize to decline too quickly, producing a behavior we refer to as 'apparent convergence.'

For complex problems where near-optimality is very difficult to verify, it can be more important to ensure that we are close (which a constant stepsize will do) without stalling. Stalling arises when stepsizes become too small to make noticeable progress. We avoid this with a constant stepsize (assuming it is not too small), but constant stepsizes mean that we may bounce around the vicinity of the optimal solution. Often, near-optimal solutions are good enough for most practical situations, but there are other reasons why even a modest amount of bouncing around can be problematic. For example, it is sometimes important to compute the variance in the estimate of the value function. It is nice when running more iterations decreases this variance, something that will not happen with a constant stepsize.

Constant stepsizes are also useful if we are modeling a nonstationary problem. If our random realizations are coming from an exogenous source that may not be stationary, then

<!-- image -->

6.3b: High-noise

Figure 6.3: Illustration of the effects of smoothing using constant stepsizes. Case (a) represents a low-noise dataset, with an underlying nonstationary structure; Case (b) is a high-noise dataset from a stationary process.

we cannot work with a stepsize rule that approaches zero. Any data that deviates from the normal can be a combination of random fluctuation and structural movement in the underlying signal. If the stepsize is allowed to approach zero, the model will not be able to respond to fundamental changes in the underlying signal.

Constant stepsizes are popular in the demand forecasting community where demand for products will exhibit varying combinations of noise and structural variation. As illustrated in figure 6.3, a small stepsize works best when there is a lot of noise relative to the structural variation, while a large stepsize allows the system to adjust to changes in the mean when the level of noise is relatively low.

## Generalized harmonic stepsizes

<!-- formula-not-decoded -->

This is the fundamental class of deterministic stepsize rules that satisfy the conditions for convergence. Increasing a slows the rate at which the stepsize drops to one, as illustrated in figure 6.4. In practice, it seems that despite theoretical convergence proofs to the contrary, the stepsize 1 /n can drop to zero far too quickly, resulting in 'apparent convergence' when in fact the solution is far from the best that can be obtained.

20

80 -

60

40 -

20

00

001

4001

- a=10, b=0, beta=1

A a=20, b=0, beta=1

-* a=40, b=0, beta=1

Figure 6.4: Stepsizes for a/ ( a + n ) while varying a .

<!-- image -->

## Polynomial learning rates

An extension of the basic harmonic sequence is the stepsize:

<!-- formula-not-decoded -->

where β ∈ ( 1 2 , 1]. Smaller values of β slow the rate at which the stepsizes decline, which improves the responsiveness in the presence of initial transient conditions. The best value of β depends on the degree to which the initial data is transient, and as such is a parameter that needs to be tuned.

## McClain's formula

<!-- formula-not-decoded -->

Note that steps generated by this model satisfy the following properties

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Effect of "a" on stepsize

20

00 -*

80

40

00

- target=0

-target=0.05

- target=0.1

Figure 6.5: The McClain stepsize rule with varying targets.

<!-- image -->

McClain's rule combines the features of the '1 /n ' rule and the constant stepsize. If we set ¯ α = 0, then it is easy to verify that McClain's rule produces α n = 1 /n . In the limit, α n → ¯ α . The value of the rule is that the 1 /n averaging generally works quite well in the very first iterations (this is a major weakness of constant stepsize rules), but avoids going to zero. The rule can be effective when you are not sure how many iterations are required to start converging, and it can also work well in nonstationary environments.

## Search-then-converge learning rule

The following stepsize rule has been proposed in the machine learning literature.

<!-- formula-not-decoded -->

where α 0 , β and τ are parameters to be determined. A more compact and slightly more general version of this formula is the extended learning stepsize formula:

<!-- formula-not-decoded -->

If β = 1, then this formula is identical to the machine learning rule. In addition, if b = 0, then it is the same as the a/ ( a + n -1) rule. The addition of the term b/n to the numerator

McClain stepsize

20

00

80

40

20

AAA

Effect of  "b" on stepsize

• a=40, b=0, beta=1

•- a=40, b=10000, beta=1

• a=40, b=40000, beta=1

-* a=40, b=100000, beta=1

Figure 6.6: Stepsizes for ( b/n + a ) / ( b/n + a + n ) while varying b .

<!-- image -->

and the denominator can be viewed as a kind of a/ ( a + n ) rule where a is very large but declines with n . The effect of the b/n term, then, is to keep the stepsize larger for a longer period of time, as illustrated in figure 6.6. This can help algorithms that have to go through an extended learning phase when the values being estimated are relatively unstable. The relative magnitude of b depends on the number of iterations which are expected to be run, which can range from several dozen to several million.

This class of stepsize rules is termed 'search-then-converge' because they provide for a period of high stepsizes (while searching is taking place) after which the stepsize declines (to achieve convergence). The degree of delayed learning is controlled by the parameter b , which can be viewed as playing the same role as the parameter a but which declines as the algorithm progresses.

The exponent β in the denominator has the effect of increasing the stepsize in later iterations. With this parameter, it is possible to accelerate the reduction of the stepsize in the early iterations (by using a smaller a ) but then slow the descent in later iterations (to sustain the learning process). This may be useful for problems where there is an extended transient phase requiring a larger stepsize for a larger number of iterations.

## 6.3.3 Stochastic stepsizes

There is considerable appeal to the idea that the stepsize should depend on the actual trajectory of the algorithm. In this section, we first review the case for stochastic stepsizes, then present the revised theoretical conditions for convergence, and finally outline a series

20

00

80

60

40

20

00

ЛАЛІ

• a=20, b=0, beta=1

-- a=20, b=0, beta=0.9

• a=20, b=0, beta=0.8

-x a=20, b=0, beta=0.7

-*- a=160, b=0, beta=1

Figure 6.7: Stepsizes for ( b/n + a ) / ( b/n + a + n β ) while varying β .

<!-- image -->

of recipes that have been suggested in the literature (including some that have not).

## The case for stochastic stepsizes

Assume that our estimates are consistently under or consistently over the actual observations. This can easily happen during early iterations due to either a poor initial starting point or the use of biased estimates (which is common in dynamic programming) during the early iterations. For large problems, it is possible that we have to estimate thousands of parameters. It seems unlikely that all the parameters will approach their true value at the same rate. Also, if we are solving a dynamic program in an online, dynamic environment, the data may be nonstationary, requiring a system that responds to shifts in the underlying source of information.

Figure 6.8 shows an actual example of the path taken for estimating different parameters in an algorithm. In some cases, we start with reasonable initial observations and benefit from a smaller stepsize to stabilize the estimate of the mean. In other cases, we start with a poor initial estimate and also face the problem that we are sampling observations from a nonstationary distribution. This situation arises in dynamic programming because of the nature of value iteration (as we showed in section 4.4.1) where the values evolve over time as the system learns more about values farther into the future.

The solution to these problems is to devise a stepsize rule that depends on the specific pattern of errors for each parameter. These rules are known as stochastic (or adaptive) stepsize rules.

Effect of varying beta on stepsize

Figure 6.8: Convergence path for different parameters, using a 1/n stepsize rule.

<!-- image -->

## Convergence conditions

When the stepsize depends on the history of the process, the stepsize itself becomes a random variable. This change requires some subtle modifications to our requirements for convergence (equations (6.12) and (6.13)). For technical reasons, our convergence criteria change to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The condition 'almost surely' (universally abbreviated 'a.s.') means that equation (6.25) holds for every sample path ω (where the probability of this sample path is nonzero).

For the reasons behind these conditions, go to our 'Why does it work' section (section 6.8). It is important to emphasize, however, that these conditions are completely unverifiable and are purely for theoretical reasons. The real issue with stochastic stepsizes is whether they contribute to the rate of convergence.

## Recipes for stochastic stepsizes

To present our stochastic stepsize formulas, we need to define a few quantities. If we are trying to estimate the mean of a random variable (which we are often trying to do in

approximate dynamic programming), the error in our estimate is given by:

<!-- formula-not-decoded -->

We may wish to smooth the error in the estimate, which we designate by the function:

<!-- formula-not-decoded -->

Some formulas depend on tracking changes in the sign of the error. This can be done using the indicator function:

<!-- formula-not-decoded -->

Thus, I ε n ε n -1 &lt; 0 indicates if the sign of the error has changed in the last iteration.

Following is a series of formulas that adjust the stepsize based on the observed errors in the estimates.

## Kesten's rule

<!-- formula-not-decoded -->

where α 0 is the initial stepsize and a is a parameter to be calibrated. K n counts the number of times that the sign of the error has changed:

<!-- formula-not-decoded -->

Kesten's rule is particularly well suited to initialization problems. It slows the reduction in the stepsize as long as the error exhibits the same sign (and indication that the algorithm is still climbing into the correct region).

## Mirozahmedov's rule

Mirozahmedov &amp; Uryasev (1983) formulates an adaptive stepsize rule that increases or decreases the stepsize in response to whether the inner product of the successive errors is positive or negative, along similar lines as in Kesten's rule.

<!-- formula-not-decoded -->

where a and δ are some fixed constants. A variation of this rule where δ is zero is proposed by Ruszczy´ nski &amp; Syski (1986).

## Gaivoronski's rule

Gaivoronski (1988) proposes an adaptive stepsize rule according to which the stepsize is computed as a function of the ratio of the progress to the path of the algorithm. The progress is measured in terms of the difference in the values of the smoothed estimate between a certain number of iterations. The path is measured as the sum of absolute values of the differences between successive estimates for the same number of iterations.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where γ 1 and γ 2 are constants.

## Stochastic gradient adaptive stepsize rule

This class of rules uses stochastic gradient logic to update the stepsize. We first compute:

<!-- formula-not-decoded -->

The stepsize is then given by

<!-- formula-not-decoded -->

where α + and α -are, respectively, upper and lower limits on the stepsize. [ · ] α + α -represents a projection back into the interval [ α -, α + ], and ν is a scaling factor. ψ n ε n is a stochastic gradient that indicates how we should change the stepsize to improve the error. Since the stochastic gradient has units that are the square of the units of the error, while the stepsize is unitless, ν has to perform an important scaling function. The equation α n -1 + νψ n ε n can easily produce stepsizes that are bigger than 1 or smaller than 0, so it is customary to specify an allowable interval (which is generally smaller than (0,1)). This rule has provable convergence, but in practice, ν , α + and α -all have to be tuned.

The remaining formulas are drawn from the forecasting literature. These problems are characterized by nonstationary series which can exhibit shifts in the mean. For these problems, it can be useful to have a stepsize rule that moves upward when it detects what appears to be a structural change in the signal.

## Trigg's formula (Trigg &amp; Leach (1967)

<!-- formula-not-decoded -->

This is the first of the adaptive stepsize formulas that uses the ratio of absolute value of the smoothed error over the smoothed absolute values of errors. Although appealing, experiments with Trigg's formula indicated that it was too responsive to what were nothing more than random sequences of errors with the same sign. The overresponsiveness of Trigg's adaptive formula has produced variants that dampen this behavior.

## Damped Trigg formula

<!-- formula-not-decoded -->

where α trigg is as in (6.19) and γ is a constant in the interval ]0 , 1] to be calibrated. A variation on this model is the following:

<!-- formula-not-decoded -->

The damped version of Trigg's formula was designed to reduce the tendency of Trigg's formula to jump around.

Another variant is a hybrid of McClain's formula and Trigg:

## Adaptive McClain formula (Godfrey's rule)

<!-- formula-not-decoded -->

where γ n is the adaptive target step at iteration n given by Trigg's formula. Godfrey's rule (invented by Greg Godfrey) is like Trigg's formula with a shock absorber. McClain's formula moves toward the target with a rate comparable to an arithmetic sequence. By using Trigg's formula as the target, changes in the Trigg stepsize are damped by McClain's formula.

Adaptive 1 /n (Belgacem's rule)

In this formula, we use a 1 /n stepsize, but we reset the iteration counter when certain conditions are satisfied.

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Belgacem's rule (developed by Belgacem Bouzaiene-Ayari) uses a 1 /n rule but resets the counter when it detects what appears to be a change in the underlying signal by using Trigg's formula as a trigger.

A word of caution is offered when testing out stepsize rules. It is quite easy to test out these ideas in a controlled way in a simple spreadsheet, but it is one thing to develop a stepsize formula that works well on a specific class of datasets, and it is another thing altogether to produce a stepsize formula that works well in real applications.

## 6.3.4 A note on counting visits

Throughout our presentation, we represent the stepsize at iteration n using α n . For discrete, table-lookup representations of value functions (as we are doing here), the stepsize should reflect how many times we have visited a particular state. If n ( S ) is the number of times we have visited state S , then the stepsize for updating ¯ V ( S ) should be α n ( S ) . For notational simplicity, we suppress this capability, but it can have a significant impact on the empirical rate of convergence.

## 6.4 Computing bias and variance

In section 6.5 we present a stepsize rule that minimizes the expected error in an idealized setting where certain (unknowable) parameters are assumed known. We then use this expression to produce a practical stepsize formula that is quite easy to use and has some very desirable properties. But before we present this rule, it helps to present some results on the variance and bias of our estimators which are true for any stepsize rule, and which play an important role in our development of an optimal stepsize rule.

In a stochastic optimization problem min E F ( θ, W ), we might use updates of the form ¯ θ n = ¯ θ n -1 -α n ∇ θ F ( ¯ θ n -1 , W n ). When updating a value function, we might use updates of

the form ¯ v n = (1 -α n )¯ v n -1 + α n ˆ v n , where ¯ v n is an estimate of the value of being in a state. For this section, we are going to assume that we are estimating some parameter θ n , where ¯ θ n is our estimate after iteration n . Our updating equation is of the form

<!-- formula-not-decoded -->

where ˆ θ n is an unbiased observation (that is, E ˆ θ n = θ n ) that is assumed to be independent of ¯ θ n -1 . We are interested in estimating the variance of ¯ θ n and its bias, which is given by ¯ θ n -1 -θ n .

We start by computing V ar [ ¯ θ n ]. We assume that our observations of θ can be represented using

<!-- formula-not-decoded -->

where E ε n = 0 and V ar [ ε n ] = σ 2 . We propose that we can compute the variance of ¯ θ n using:

<!-- formula-not-decoded -->

where λ n can be computed from the simple recursion:

<!-- formula-not-decoded -->

To see this, we start with n = 1. For a given (deterministic) initial estimate ¯ θ 0 , we first observe that the variance of ¯ θ 1 is given by:

<!-- formula-not-decoded -->

For general ¯ θ n , we use a proof by induction. Assume that V ar [ ¯ θ n -1 ] = λ n -1 σ 2 . Then, since ¯ θ n -1 and ˆ θ n are independent, we find

<!-- formula-not-decoded -->

Equation (6.44) is true by assumption (in our induction proof), while equation 6.45 establishes the recursion in equation (6.42). This gives us the variance, assuming of course that σ 2 is known. Similarly, our bias is given by:

<!-- formula-not-decoded -->

which assumes that θ n is known. These two results for the variance and bias are called the parameters-known formulas.

In practice, we do not know σ 2 , and we certainly do not know θ n . As a result, we have to estimate both parameters from our data. Normally we would estimate the variance by simply smoothing on the squares of the errors. Let ¯ ν n be the estimate of the total variance, which we might compute using:

<!-- formula-not-decoded -->

where γ n is a deterministic stepsize rule such as McClain's (equation (6.19). The problem with this estimate of the variance is that it combines the noise of the observations with the bias. Let

<!-- formula-not-decoded -->

be the total variance (including noise and bias). It is possible to show that

<!-- formula-not-decoded -->

Using ¯ ν n as our estimate of the total variance, we can compute an estimate of σ 2 using

<!-- formula-not-decoded -->

Now we just need an estimate of the bias. We compute an estimate of the bias by simply smoothing on the error using

<!-- formula-not-decoded -->

This estimate of the bias can be quite rough (if we could estimate the bias accurately, then it means we know the true parameter accurately). As a general rule, we should pick a stepsize for γ n which produces larger stepsizes because we are more interested in tracking the true signal than producing an estimate with a low variance.

## 6.5 Optimal stepsizes

Given the variety of stepsize formulas we can choose from, it seems natural to ask whether there is an optimal stepsize rule. Before we can answer such a question, we have to define exactly what we mean by it. Assume that we are trying to estimate a parameter (such as a value of being in a state or the slope of a value function) that we denote by θ n that may

be changing over time. Let ˆ θ n be a random observation of the process in iteration n , and let ¯ θ n be our estimate of the parameter after n iterations that we update using a stochastic gradient algorithm. At iteration n , ¯ θ n is a random variable that depends on our stepsize rule. To express this dependence, let α represent a stepsize rule, and let ¯ θ n ( α ) be the estimate of the parameter θ after iteration n using stepsize rule α . We would like to choose a stepsize rule to minimize:

<!-- formula-not-decoded -->

Here, the expectation is over the entire history of the algorithm and requires (in principle) knowing the true value of the parameter being estimated. If we could solve this problem (which requires knowing certain parameters about the underlying distributions), we would obtain a deterministic stepsize rule. In practice, we do not generally know these parameters which need to be estimated from data, producing a stochastic stepsize rule.

There are other objective functions we could use. For example, instead of minimizing the distance to an unknown parameter sequence θ n , we could minimize:

<!-- formula-not-decoded -->

where we are trying to minimize the deviation between our prediction, obtained at iteration n , and the actual observation at n +1. Here, we are again proposing an unconditional expectation, which means that ¯ θ n ( α ) is a random variable within the expectation. Alternatively, we could condition on our history up to iteration n :

<!-- formula-not-decoded -->

where conditioning on F n means that we are conditioning on the history of our algorithm up through iteration n (for students with a measure-theoretic background, F n is the sigmaalgebra generated by the history of the process). In this formulation ¯ θ n ( α ) is now deterministic, but ˆ θ n +1 is random.

We begin our discussion of optimal stepsizes in section 6.5.1 by addressing the case of a constant parameter. Section 6.5.2 considers the case where we are estimating a parameter that is changing over time, but where the changes have mean zero. Finally, section 6.5.3 addresses the case where the mean may be drifting up or down with nonzero mean.

## 6.5.1 Optimal stepsizes for stationary data

Assume that we observe ˆ θ n at iteration n , and that the observations ˆ θ n can be described by the following model:

<!-- formula-not-decoded -->

where θ is an unknown constant and /epsilon1 n is a stationary sequence of independent and identically distributed random deviations with mean 0 and variance σ 2 . We can approach the problem of estimating θ from two perspectives: choosing the best stepsize and choosing the best linear combination of the estimates. That is, we may choose to write our estimate ¯ θ n after n observations in the form:

<!-- formula-not-decoded -->

For our discussion, we will fix n and work to determine the coefficients a m (recognizing that they can depend on the iteration). We would like our statistic to have two properties: it should be unbiased, and it should have minimum variance (that is, it should solve (6.47)). To be unbiased, it should satisfy:

<!-- formula-not-decoded -->

which implies that we must satisfy:

<!-- formula-not-decoded -->

The variance of our estimator is given by:

<!-- formula-not-decoded -->

We use our assumption that the random deviations are independent, which allows us to write:

<!-- formula-not-decoded -->

Now we face the problem of finding a 1 , . . . , a n to minimize (6.50) subject to the requirement that ∑ m a m = 1. This problem is easily solved using the Lagrange multiplier method. We start with the nonlinear programming problem:

<!-- formula-not-decoded -->

subject to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We relax constraint (6.51) and add it to the objective function:

n

∑

m

min

{

a

m

}

=1

n

(

∑

m

=1

subject to (6.52). We are now going to try to solve L ( a, λ ) (known as the 'Lagrangian') and hope that the coefficients a are all nonnegative. If this is true, we can take derivatives and set them equal to zero:

<!-- formula-not-decoded -->

The optimal solution ( a ∗ , λ ∗ ) would then satisfy:

<!-- formula-not-decoded -->

which means that at optimality:

<!-- formula-not-decoded -->

which tells us that the coefficients a m are all equal. Combining this result with the requirement that they sum to one gives the expected result:

<!-- formula-not-decoded -->

In other words, our best estimate is a sample average. From this (somewhat obvious) result, we can obtain the optimal stepsize, since we already know that α n = 1 /n is the same as using a sample average.

L

(

a, λ

) =

a

2

m

-

λ

a

m

-

1)

This result tells us that if the underlying data is stationary, and we have no prior information about the sample mean, then the best stepsize rule is the basic 1 /n rule. Using any other rule requires that there be some violation in our basic assumptions. In practice, the most common violation is that the observations are not stationary because they are derived from a process where we are searching for the best solution.

## 6.5.2 Optimal stepsizes for nonstationary data - I

Assume now that our parameter evolves over time (iterations) according to the process:

<!-- formula-not-decoded -->

where E ν n = 0 is a zero mean drift term with variance ( σ ν ) 2 . As before, we measure θ n with an error according to:

<!-- formula-not-decoded -->

We want to choose a stepsize so that we minimize the mean squared error. This problem can be solved using the Kalman filter. The Kalman filter is a powerful recursive regression technique, but we adapt it here for the problem of estimating a single parameter. Typical applications of the Kalman filter assume that the variance of ν n , given by ( σ ν ) 2 , and the variance of the measurement error, ε n , given by σ 2 , are known. In this case, the Kalman filter would compute a stepsize (generally referred to as the gain) using:

<!-- formula-not-decoded -->

where p n is computed recursively using:

<!-- formula-not-decoded -->

For our application, we will not know the variances so these have to be estimated from data. We first estimate the bias using:

<!-- formula-not-decoded -->

where γ n is a simple stepsize rule such as McClain's formula. We then estimate the total variation using:

<!-- formula-not-decoded -->

Finally, we estimate the variance of the error using:

<!-- formula-not-decoded -->

We then use (¯ σ n ) 2 and ( ¯ β n ) 2 as our estimates of σ 2 and ( σ δ ) 2 , respectively.

## 6.5.3 Optimal stepsizes for nonstationary data - II

Our challenge is to devise a stepsize that strikes a balance between minimizing error (which prefers a smaller stepsize) and responding to the nonstationary data (which works better with a large stepsize). For our model of the data, we use:

¯ θ n = The actual baseline at iteration n .

ε n = The random noise in the observation at iteration n .

ˆ θ n = The observation at iteration n .

<!-- formula-not-decoded -->

As before we assume that { ε n } n =1 , 2 ,... are independent and identically distributed with mean value of zero and variance, σ 2 . We perform the usual stochastic gradient update to obtain our estimates of the mean:

<!-- formula-not-decoded -->

α n denotes the smoothing stepsize at iteration n . We wish to find α n that solves,

<!-- formula-not-decoded -->

We use the fact that the observation at iteration n is unbiased, which is to say:

<!-- formula-not-decoded -->

But the smoothed estimate is biased because we are using simple smoothing on nonstationary data. We denote this bias as:

<!-- formula-not-decoded -->

We propose the following theorem for finding the optimal stepsizes.

Theorem 6.5.1 The optimal stepsizes ( α m ) n m =1 that minimize the objective function in equation (6.61) can be computed using the expression,

<!-- formula-not-decoded -->

where,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof is given in section 6.8.4. The somewhat cumbersome calculation in equation (6.65) can be replaced with the recursive expression:

<!-- formula-not-decoded -->

There are two interesting corollaries that we can easily establish: the optimal stepsize if there is no trend, and the optimal stepsize if there is no noise. The case where there is no trend is the same as assuming that the data is stationary. Earlier we proved this case by solving a linear regression problem, and noting that the best estimate was a sample average, which implied a particular stepsize. Here, we directly find the optimal stepsize as a corollary of our earlier result.

Corollary 6.5.1 For a sequence with a static mean, the optimal stepsizes are given by,

<!-- formula-not-decoded -->

Proof: In this case, the mean θ n = θ is a constant. Therefore, the estimates of the mean are unbiased, which means β n = 0 ∀ t = 2 , . . . , . This allows us to write the optimal stepsize as:

<!-- formula-not-decoded -->

Equation (6.69) is now given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the case where there is no noise ( σ 2 = 0), we have:

Corollary 6.5.2 For a sequence with zero noise, the optimal stepsizes are given by,

<!-- formula-not-decoded -->

The corollary is proved by simply setting σ 2 = 0 in equation (6.64).

The problem with using the optimal stepsize formula in equation (6.64) is that it assumes that the variance σ 2 and the bias ( β n ) 2 are known. This can be problematic in real problems, especially the assumption of knowing the bias, since computing this basically requires knowing the real function. If we have this information, we do not need this algorithm.

As an alternative, we can try to estimate these quantities from data. Let:

(¯ σ 2 ) n = Estimate of the variance of the error after iteration n .

¯ β n = Estimate of the bias after iteration n .

¯ ¯ β n = Estimate of the variance of the bias after iteration n .

To make these estimates, we need to smooth new observations with our current best estimate, something that requires the use of a stepsize formula. We could attempt to find an optimal stepsize for this purpose, but it is likely that a reasonably chosen deterministic formula will work fine. We propose to use McClain's formula (equation (6.19):

<!-- formula-not-decoded -->

A limit point such as ¯ γ ∈ (0 . 05 , 0 . 10) appears to work well across a broad range of functional behaviors.

If we introduce these steps to compute the variance and bias dynamically from data, the result is the algorithm shown in figure 6.9, which we refer to as the optimal stepsize algorithm (OSA).

## 6.6 Some experimental comparisons of stepsize formulas

George &amp; Powell (2004) reports on a series of experimental comparisons of stepsize formulas. The methods were tested on a series of functions shown in figure 6.10. Two classes of functions were used, but they all increased monotonically to a limiting value. The first class, which we denote by f I ( n ), increases at a geometrically decreasing rate right from the beginning. The argument n is referred to as the 'iteration' since we view these functions as reflecting the change in a value function over the iterations. The second class, denoted

Step 0. Initialization:

Step 0a. Set the baseline to its intial value, ¯ θ 0 .

Step 0b. Initialize the parameters -¯ β 0 , ¯ ¯ β 0 and ¯ λ 0 .

Step 0c. Choose initial and target values for the error stepsize γ 0 and ¯ γ .

Step 0d. Set the iteration counter, n = 1.

Step 1. Obtain the new observation, ˆ θ n .

Step 2. Update the following parameters:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 3. Evaluate the stepsizes for the current iteration.

<!-- formula-not-decoded -->

Step 3a. Compute the coefficient for the variance of the smoothed estimate of the baseline.

<!-- formula-not-decoded -->

Step 4. Smooth the baseline estimate.

<!-- formula-not-decoded -->

Step 5. If n &lt; N , then n = n +1 and go to Step 1, else stop.

Figure 6.9: The Optimal Stepsize Algorithm with parameters unknown (OSA-PU).

f II ( n ), remains constant initially, and then underwent a delayed increase. For each class of functions, there are five variations with different overall rates of increase.

Each function was measured with an error term with an assumed variance. Three levels of variance were used, depicted in figure 6.11 which shows the structural change in the expected value along with random observations drawn from distributions with each of the three noise levels.

The idea of an adaptive stepsize is most dramatically illustrated for the functions that start off constant, and then rise quickly at a later iteration. Standard stepsize formulas that decline monotonically over time run the risk of dropping to small values. When the function starts to increase, the stepsizes have dropped to such small values that they cannot respond quickly. Figure 6.12 compares the search-then-converge formula to the optimal stepsize method (the algorithm given in figure 6.9). This behavior can actually occur in

Figure 6.10: Ten functions divided into two groups. The first group has a steadily declining rate of convergence, while the second has delayed convergence.

<!-- image -->

Figure 6.11: Observations at different noise levels.

<!-- image -->

dynamic programs which require a number of steps before receiving a reward. Examples include most games (backgammon, checkers, tic-tac-toe) where it is necessary to play an entire game before we learn if we won (we receive a reward) or lost.

Many dynamic programs exhibit the behavior of our type I functions where the value function rises steadily and then levels off. The problem is that the rate of increase can vary widely. It helps to use a larger stepsize for parameters that are increasing quickly.

Table 6.1 shows the results of a series of experiments comparing different stepsize rules for

Figure 6.12: Comparison of stepsizes formulas

<!-- image -->

the class I functions ( f I ( n )). The experiments were run using three noise levels, and measured at three different points n along the curve. At n = 25, the curve is still rising quickly; n = 50 corresponds to the 'elbow' of the curve; and at n = 75 the curve is stabilizing. The stepsizes considered where 1 /n , 1 /n β (with β = . 85), STC (search-then-converge) using parameters optimized for this problem class, McClain (with target 0.10), Kesten's rule, the stochastic gradient adaptive stepsize rule (SGASR), and the optimal stepsize algorithm (OSA). The table gives the average mean squared error and the standard deviation of the average. Table 6.2 provides the same statistics for the second class of functions which exhibit delayed learning. For these functions n = 25 occurs when the function is in its initial stable period; n = 50 is the point where the curve is rising the most; and n = 75 corresponds to where the curve is again stable.

These results show that the adaptive stepsize formulas (in particular the optimal stepsize algorithm) works best with lower levels of noise. If the noise level is quite high, stochastic formulas can be thrown off. For these cases, the family of arithmetically declining stepsizes (1 /n, 1 /n β , STC and McClain) all work reasonably well. The risk of using 1 /n or even 1 /n β is that they may decline too quickly for functions which are still changing after dozens of iterations. McClain avoids this by imposing a target. For problems where we are running more than a few dozen iterations, McClain is similar to using a constant stepsize.

The real value of an adaptive stepsize rule in dynamic programming is its ability to respond to different rates of convergence for different parameters (such as the value of being in a particular states). Some parameters may converge much more slowly than others, and will benefit from a larger stepsize. Since we cannot tune a stepsize rule for each parameter, an adaptive stepsize rule may work best. Table 6.3 shows the percent error for a steady state nomadic trucker application (which can be solved optimally) using three values for the discount factor. Since updates of the value function depend on the value function approximation itself, convergence tends to be slow (especially for the higher discount factors). Here

Table 6.1: A comparison of stepsize rules for class I functions, which are concave and monotonically increasing (from George &amp; Powell (2004)).

| Variance   | n         |   1 /n |   1 /n β |    STC |   McClain |   Kesten |   SGASR |    OSA |
|------------|-----------|--------|----------|--------|-----------|----------|---------|--------|
| σ 2 = 1    | 25        |  5.721 |    3.004 |  0.494 |     1.76  |    0.368 |   0.855 |  0.365 |
| σ 2 = 1    | std. dev. |  0.031 |    0.024 |  0.014 |     0.021 |    0.014 |   0.025 |  0.015 |
| σ 2 = 1    | 50        |  5.697 |    2.374 |  0.322 |     0.502 |    0.206 |   0.688 |  0.173 |
| σ 2 = 1    | std. dev. |  0.021 |    0.015 |  0.008 |     0.01  |    0.008 |   0.021 |  0.008 |
| σ 2 = 1    | 75        |  4.993 |    1.713 |  0.2   |     0.169 |    0.135 |   0.578 |  0.126 |
| σ 2 = 1    | std. dev. |  0.016 |    0.011 |  0.005 |     0.005 |    0.006 |   0.018 |  0.006 |
| σ 2 = 10   | 25        |  6.216 |    3.527 |  1.631 |     2.404 |    2.766 |   3.031 |  1.942 |
| σ 2 = 10   | std. dev. |  0.102 |    0.08  |  0.068 |     0.073 |    0.107 |   0.129 |  0.078 |
| σ 2 = 10   | 50        |  5.911 |    2.599 |  0.871 |     0.968 |    1.571 |   1.61  |  1.118 |
| σ 2 = 10   | std. dev. |  0.07  |    0.05  |  0.037 |     0.038 |    0.064 |   0.075 |  0.047 |
| σ 2 = 10   | 75        |  5.127 |    1.87  |  0.588 |     0.655 |    1.146 |   1.436 |  1.039 |
| σ 2 = 10   | std. dev. |  0.053 |    0.035 |  0.025 |     0.028 |    0.049 |   0.07  |  0.045 |
| σ 2 = 100  | 25        | 10.088 |    7.905 | 12.484 |     8.066 |   25.958 |  72.842 | 13.42  |
| σ 2 = 100  | std. dev. |  0.358 |    0.317 |  0.53  |     0.341 |    1.003 |   2.49  |  0.596 |
| σ 2 = 100  | 50        |  8.049 |    5.078 |  6.675 |     5.971 |   15.548 |  73.83  |  9.563 |
| σ 2 = 100  | std. dev. |  0.239 |    0.194 |  0.289 |     0.258 |    0.655 |   2.523 |  0.427 |
| σ 2 = 100  | 75        |  6.569 |    3.51  |  4.277 |     5.241 |   10.69  |  71.213 |  9.625 |
| σ 2 = 100  | std. dev. |  0.182 |    0.137 |  0.193 |     0.236 |    0.46  |   2.477 |  0.421 |

Table 6.2: A comparison of stepsize rules for class II functions which undergo delayed learning (from George &amp; Powell (2004)).

| Variance   | n         |   1 /n |   1 /n β |    STC |   McClain |   Kesten |   SGASR |    OSA |
|------------|-----------|--------|----------|--------|-----------|----------|---------|--------|
| σ 2 = 1    | 25        |  0.416 |    0.3   |  0.227 |     0.231 |    0.285 |   0.83  |  0.209 |
| σ 2 = 1    | std. dev. |  0.008 |    0.007 |  0.009 |     0.007 |    0.011 |   0.024 |  0.008 |
| σ 2 = 1    | 50        | 13.437 |   10.711 |  3.718 |     6.232 |    2.685 |   1.274 |  1.813 |
| σ 2 = 1    | std. dev. |  0.033 |    0.032 |  0.04  |     0.036 |    0.042 |   0.045 |  0.061 |
| σ 2 = 1    | 75        | 30.403 |   15.818 |  0.466 |     1.186 |    0.252 |   0.605 |  0.23  |
| σ 2 = 1    | std. dev. |  0.041 |    0.033 |  0.011 |     0.016 |    0.009 |   0.019 |  0.009 |
| σ 2 = 10   | 25        |  0.784 |    0.715 |  1.92  |     0.77  |    2.542 |   2.655 |  1.319 |
| σ 2 = 10   | std. dev. |  0.031 |    0.03  |  0.076 |     0.033 |    0.097 |   0.123 |  0.057 |
| σ 2 = 10   | 50        | 13.655 |   10.962 |  4.689 |     6.754 |    4.109 |   4.87  |  4.561 |
| σ 2 = 10   | std. dev. |  0.104 |    0.102 |  0.13  |     0.116 |    0.14  |   0.164 |  0.155 |
| σ 2 = 10   | 75        | 30.559 |   15.989 |  1.133 |     1.677 |    1.273 |   1.917 |  1.354 |
| σ 2 = 10   | std. dev. |  0.127 |    0.103 |  0.047 |     0.055 |    0.055 |   0.089 |  0.056 |
| σ 2 = 100  | 25        |  4.5   |    5.002 | 19.399 |     6.359 |   25.887 |  71.974 | 12.529 |
| σ 2 = 100  | std. dev. |  0.199 |    0.218 |  0.765 |     0.271 |    0.991 |   2.484 |  0.567 |
| σ 2 = 100  | 50        | 15.682 |   13.347 | 14.21  |    11.607 |   18.093 |  72.855 | 13.726 |
| σ 2 = 100  | std. dev. |  0.346 |    0.345 |  0.6   |     0.438 |    0.751 |   2.493 |  0.561 |
| σ 2 = 100  | 75        | 32.069 |   17.709 |  7.602 |     6.407 |   11.119 |  73.286 | 10.615 |
| σ 2 = 100  | std. dev. |  0.409 |    0.338 |  0.333 |     0.278 |    0.481 |   2.504 |  0.461 |

the optimal stepsize algorithm works particularly well

There does not appear to be a universally accepted stepsize rule. Stochastic stepsize rules offer tremendous appeal. For example, if we are estimating the value of being in each

Table 6.3: Percentage error in the estimates from the optimal values, averaged over all the resource states, as a function of the average number of observations per state. The figures in italics denote the standard deviations of the values to the left (from George &amp; Powell (2004)).

|    γ |   n |   1 /n |   1 /n |   1 /n β |   1 /n β |   STC |   STC |   Benveniste |   Benveniste |   OSA |   OSA |
|------|-----|--------|--------|----------|----------|-------|-------|--------------|--------------|-------|-------|
| 0.8  |   2 |  43.55 |   0.14 |    52.33 |     0.12 | 41.93 |  0.14 |        50.63 |         0.14 | 41.38 |  0.16 |
| 0.8  |   5 |  17.39 |   0.12 |    25.53 |     0.11 | 12.04 |  0.13 |        19.05 |         0.14 | 11.76 |  0.15 |
| 0.8  |  10 |  10.54 |   0.08 |    14.91 |     0.08 |  4.29 |  0.04 |         7.7  |         0.07 |  4.86 |  0.05 |
| 0.9  |   2 |  51.27 |   0.14 |    64.23 |     0.09 | 48.39 |  0.13 |        60.78 |         0.11 | 46.15 |  0.12 |
| 0.9  |   5 |  29.22 |   0.12 |    42.12 |     0.11 | 19.2  |  0.12 |        27.61 |         0.12 | 15.26 |  0.14 |
| 0.9  |  10 |  22.72 |   0.11 |    31.36 |     0.08 |  8.55 |  0.07 |         9.84 |         0.1  |  7.45 |  0.09 |
| 0.95 |   2 |  62.64 |   0.19 |    76.92 |     0.08 | 58.75 |  0.14 |        72.76 |         0.12 | 54.93 |  0.15 |
| 0.95 |   5 |  45.26 |   0.21 |    60.95 |     0.09 | 33.21 |  0.15 |        41.64 |         0.15 | 24.38 |  0.14 |
| 0.95 |  10 |  39.39 |   0.2  |    51.83 |     0.08 | 21.46 |  0.11 |        18.78 |         0.11 | 14.93 |  0.1  |

state, it may be the case that each of these values moves according to its own process. Some may be relatively stationary, while others move quickly before stabilizing. The value of an adaptive stepsize rule appears to depend on the nature of the data and our ability to tune a deterministic rule. It may be the case that if we can properly tune a single, deterministic stepsize rule, then this will work the best. The challenge is that this tuning process can be time consuming. Furthermore, it is not always obvious when a particular stepsize rule is not working.

Despite the apparent benefits of adaptive stepsize rules, deterministic rules, such as variants of the STC rule, remain quite popular. One issue is that the adaptive rules not only introduce a series of additional computations, the logic introduces additional statistics that have to be computed and stored for each parameter. In some applications, there can be tens of thousands of such parameters, which introduces significant computational and memory overhead. Just as important, while adaptive formulas can do a better job of estimating a value function, this does not always translate into higher quality solutions. The reason for this seeming contradiction is that a higher quality value function does not always translate into a noticeably higher quality solution.

## 6.7 Convergence

A practical issue that arises with all stochastic approximation algorithms is that we simply do not have reliable, implementable stopping rules. Proofs of convergence in the limit are an important theoretical property, but they provide no guidelines or guarantees in practice. A good illustration of the issue is given in figure 6.13. Figure 6.13a shows the objective function for a dynamic program over 100 iterations (in this application, a single iteration required approximately 20 minutes of CPU time). The figure shows the objective function for several different algorithms, but the convergence appeared to be about the same. The same algorithm was then run over 1000 iterations, with the results shown in 6.13b. Note

6.13a: Objective function over 100 iterations. 6.13b: Objective function over 1000 iterations.

<!-- image -->

<!-- image -->

Figure 6.13: The objective function, plotted over 100 iterations (a), displays 'apparent convergence,' The same algorithm, continued over 1000 iterations (b), shows significant improvement.

that the best solution found over 100 iterations, shown as a horizontal bar in both figures, is nowhere near the optimal when the algorithm was run an additional 900 iterations.

We refer to this behavior as 'apparent convergence.' In figure 6.13a, the algorithm had appeared to convergence, showing a number of iterations with no apparent improvement. In any algorithm using a declining stepsize, it is possible to show a stabilizing objective function simply because the stepsize is decreasing. However, it is also possible for a dynamic program to go through periods of stability which are simply a precursor to breaking through to new plateaus.

The problem with convergence in dynamic programs arises primarily when using some variant of value iteration, where the updated estimate of the value of being in a state depends on an estimate of a future value. Since the estimate of the future value may be far from the true value, we may be making incorrect decisions, which further slows convergence to the right value.

## 6.8 Why does it work?**

Stochastic approximation methods have a rich history starting with the seminal paper Robbins &amp; Monro (1951) and followed by Blum (1954 a ) and Dvoretzky (1956). The serious reader should see Kushner &amp; Yin (1997) for a modern treatment of the subject. A separate line of investigation was undertaken by researchers in eastern European community focusing on constrained stochastic optimization problems (Ermoliev (1971), Gaivoronski (1988), Ermoliev (1983), Ermoliev (1988), Ruszczy´ nski (1980 a ), Ruszczy´ nski (1987)). This work is critical to our fundamental understanding of Monte Carlo-based stochastic learning methods.

The theory behind these proofs is fairly deep and requires some mathematical maturity.

For pedagogical reasons, we start in section 6.8.1 with some probabilistic preliminaries, after which section 6.8.2 presents one of the original proofs, which is relatively more accessible, and which allows a student to understand the basis for the universal requirements that stepsizes must satisfy for theoretical proofs. Section 6.8.3 provides a more modern proof based on the theory of martingales.

## 6.8.1 Some probabilistic preliminaries

The goal in this section is to prove that these algorithms work. But what does this mean? The solution ¯ θ n at iteration n is a random variable. Its value depends on the sequence of sample realizations of the random variables over iterations 1 to n . If ω = ( ω 1 , ω 2 , . . . , ω n , . . . ) represents the sample path that we are following, we can ask what is happening to the limit lim n →∞ ¯ θ n ( ω ). If the limit is ¯ θ ∗ , does ¯ θ ∗ depend on the sample path ω ?

In the proofs below, we show that the algorithms converge almost surely . What this means is that:

<!-- formula-not-decoded -->

for all ω ∈ Ω that can occur with positive probability (that is, p ( ω ) &gt; 0). Here, ¯ θ ∗ is a deterministic quantity that does not depend on the sample path. Because of the restriction p ( ω ) &gt; 0, we accept that in theory, there could exist a sample outcome that can never occur that would produce a path that converges to some other point. As a result, we say that the convergence is 'almost sure,' which is universally abbreviated as ' a.s. ' Almost sure convergence establishes the core theoretical property that the algorithm will eventually settle in on a single point. This is an important property for an algorithm, but it says nothing about the rate of convergence.

Let x ∈ R n . At each iteration n , we sample some random variables to compute the function (and its gradient). The sample realizations are denoted by ω n . We let ω = ( ω 1 , ω 2 , . . . , ) be a realization of all the random variables over all iterations. Let Ω be the set of all possible realizations of ω , and let F be the σ -algebra on Ω (that is to say, the set of all possible events that can be defined using Ω). We need the concept of the history up through iteration n . Let:

H n = A random variable giving the history of all random variables up through iteration n .

A sample realization of H n would be:

<!-- formula-not-decoded -->

/negationslash

We could then let Ω n be the set of all outcomes of the history (that is, h n ∈ H n ) and let H n be the σ -algebra on Ω n (which is the set of all events, including their complements and unions, defined using the outcomes in Ω n ). Although we could do this, this is not the convention followed in the probability community. Instead, we define a sequence of σ -algebras F 1 , F 2 , . . . , F n as the sequence of σ -algebras on Ω that can be generated as we have access to the information through the first 1 , 2 , . . . , n iterations, respectively. What does this mean? Consider two outcomes ω = ω ′ for which H n ( ω ) = H n ( ω ′ ). If this is the case, then any event in F n that includes ω must also include ω ′ . If we say that a function is F n -measurable, then this means that it must be defined in terms of the events in F n , which is in turn equivalent to saying that we cannot be using any information from iterations n +1 , n +2 , . . . .

We would say, then, that we have a standard probability space (Ω , F , P ) where ω ∈ Ω represents an elementary outcome, F is the σ -algebra on F and P is a probability measure on Ω. Since our information is revealed iteration by iteration, we would also then say that we have an increasing set of σ -algebras F 1 ⊆ F 2 ⊆ . . . ⊆ F n .

## 6.8.2 An older proof

Enough with probabilistic preliminaries. Let F ( θ, ω ) be a F -measurable function. We wish to solve the unconstrained problem:

<!-- formula-not-decoded -->

with ¯ θ ∗ being the optimal solution. Let g ( θ, ω ) be a stochastic ascent vector that satisfies:

<!-- formula-not-decoded -->

For many problems, the most natural ascent vector is the gradient itself:

<!-- formula-not-decoded -->

which clearly sastifies (6.73).

We assume that F ( θ ) = E { F ( θ, ω ) } is continuously differentiable and convex, with bounded first and second derivatives so that for finite M :

<!-- formula-not-decoded -->

A stochastic gradient algorithm (sometimes called a stochastic approximation method) is given by:

<!-- formula-not-decoded -->

We first prove our result using the proof technique of Blum (1954 a ) that generalized the original stochastic approximation procedure proposed by Robbins &amp; Monro (1951) to multidimensional problems. This approach does not depend on more advanced concepts such as Martingales, and as a result is accessible to a broader audience. This proof helps the reader understand the basis for the conditions ∑ ∞ n =0 α n = ∞ and ∑ ∞ n =0 ( α n ) 2 &lt; ∞ that are required of all stochastic approximation algorithms.

We make the following (standard) assumptions on stepsizes:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Wewant to show that under suitable assumptions, the sequence generated by (6.76) converges to an optimal solution. That is, we want to show that:

<!-- formula-not-decoded -->

We now use Taylor's theorem (remember Taylor's theorem from freshman calculus?), which says that for any continuously differentiable convex function F ( θ ), there exists a parameter 0 ≤ θ ≤ 1 that satisfies:

<!-- formula-not-decoded -->

This is the first order version of Taylor's theorem. The second order version says:

<!-- formula-not-decoded -->

We use the second order version. Replace ¯ θ 0 with ¯ θ n , and replace θ with ¯ θ n +1 . Also, we can simplify our notation by using:

<!-- formula-not-decoded -->

This means that:

<!-- formula-not-decoded -->

From our stochastic gradient algorithm (6.76), we may write:

<!-- formula-not-decoded -->

It is now time to use a standard mathematician's trick . We sum both sides of (6.84) to get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the terms F n ( ¯ θ n ) , n = 2 , 3 , . . . , N appear on both sides of (6.86). We can cancel these. We then use our lower bound on the quadratic term (6.75) to write:

<!-- formula-not-decoded -->

We now want to take the limit of both sides of (6.87) as N →∞ . In doing so, we want to show that everything must be bounded. We know that F ( ¯ θ N ) is bounded ( almost surely ) because we assumed that the original function was bounded. We next use the assumption (6.13) that the infinite sum of the squares of the stepsizes is also bounded to conclude that the rightmost term in (6.87) is bounded. Finally, we use (6.73) to claim that all the terms in the remaining summation ( ∑ N n =1 ∇ F ( ¯ θ n )( α n g n )) are positive. That means that this term is also bounded (from both above and below).

What do we get with all this boundedness? Well, if

<!-- formula-not-decoded -->

and (from (6.12))

<!-- formula-not-decoded -->

We can conclude that

<!-- formula-not-decoded -->

Since all the terms in (6.90) are positive, they must go to zero. (Remember, everything here is true almost surely ; after a while, it gets a little boring to keep saying almost surely every time. It is a little like reading Chinese fortune cookies and adding the automatic phrase 'under the sheets' at the end of every fortune.)

We are basically done except for some relatively difficult (albeit important if you are ever going to do your own proofs) technical points to really prove convergence. At this point, we would use technical conditions on the properties of our ascent vector g n to argue that if ∇ F ( ¯ θ n , ω n ) g n → 0 then ∇ F ( ¯ θ n , ω n ) → 0 (it is okay if g n goes to zero as F ( ¯ θ n , ω n ) goes to zero, but it cannot go to zero too quickly).

This proof was first proposed in the early 1950's by Robbins and Monro and became the basis of a large area of investigation under the heading of stochastic approximation methods. A separate community, growing out of the Soviet literature in the 1960's, addressed these problems under the name of stochastic gradient (or stochastic quasi-gradient) methods. More modern proofs are based on the use of Martingale processes, which do not start with Taylor's formula and do not (always) need the continuity conditions that this approach needs.

Our presentation does, however, help to present several key ideas that are present in most proofs of this type. First, concepts of almost sure convergence are virtually standard. Second, it is common to set up equations such as (6.84) and then take a finite sum as in (6.86) using the alternating terms in the sum to cancel all but the first and last elements of the sequence of some function (in our case, F ( ¯ θ n , ω n )). We then establish the boundedness of this expression as N →∞ , which will requirethe assumption that ∑ ∞ n =1 ( α n ) 2 &lt; ∞ . Then, the assumption ∑ ∞ n =1 α n = ∞ is used to show that if the remaining sum is bounded, then its terms must go to zero.

More modern proofs will use functions other than F ( ¯ θ n ). Popular is the introduction of so-called Lyapunov functions, which are somewhat artificial functions that provide a measure of optimality. These functions are constructed for the purpose of the proof and play no role in the algorithm itself. For example, we might let T n = || ¯ θ n -¯ θ ∗ || be the distance between our current solution ¯ θ n and the optimal solution. We will then try to show that T n is suitably reduced to prove convergence. Since we do not know ¯ θ ∗ , this is not a function we can actually measure, but it can be a useful device for proving that the algorithm actually converges.

It is important to realize that stochastic gradient algorithms of all forms do not guarantee an improvement in the objective function from one iteration to the next. First, a sample gradient g n may represent an appropriate ascent vector for a sample of the function F ( ¯ θ n , ω n ) but not for its expectation. In other words, randomness means that we may go in the wrong direction at any point in time. Second, our use of a nonoptimizing stepsize, such as α n = 1 /n , means that even with a good ascent vector, we may step too far, and actually end up with a lower value.

## 6.8.3 A more modern proof

Since the original work by Robbins and Monro, more powerful proof techniques have evolved. Below we illustrate a basic Martingale proof of convergence. The concepts are somewhat more advanced, but the proof is more elegant and requires milder conditions. A significant generalization is that we no longer require that our function be differentiable (which our first proof required).

First, just what is a Martingale? Let ω 1 , ω 2 , . . . , ω t be a set of exogenous random outcomes, and let h t = H t ( ω ) = ( ω 1 , ω 2 , . . . , ω t ) represent the history of the process up to time t . We also let F t be the σ -algebra on Ω generated by H t . Further, let U t be a function that depends on h t (we would say that U t is a F t -measurable function). This means that if we know h t , then we know U t deterministically (needless to say, if we only know h t , then U t +1 is still a random variable). We further assume that our function satisfies:

<!-- formula-not-decoded -->

If this is the case, then we say that U t is a martingale . Alternatively, if:

<!-- formula-not-decoded -->

then we say that U t is a supermartingale . If U t is a supermartingale, then it has the property that it drifts downward. Note that both sides of equation (6.91) are random variables. The way to understand this equation is to think about an outcome ω ∈ Ω; if we fix ω , then U t is fixed, as is F t (that is, we have chosen a particular history from the set F t ). The inequality in equation (6.91) is then interpreted as being true for each ω .

Finally, assume that U t ≥ 0. If this is the case, we have a sequence U t that is decreasing but which cannot go below zero. Not surprisingly, we obtain the following key result:

Theorem 6.8.1 Let U t be a positive supermartingale. Then, U t converges to a finite random variable U ∗ almost surely.

So what does this mean for us? In our immediate application, we are interested in studying the properties of an algorithm as it progresses from one iteration to the next. Following our convention of putting the iteration counter in the superscript (and calling it n ), we are going to study the properties of a (nonnegative) function U n where h n represents the history of our algorithm. We want to find a function U n and show that it is a supermartingale to establish convergence to some limit ( U ∗ ). Finally, we want to show that it must converge to a point that represents the optimal solution.

The first step is finding the function U n . It is important to remember that we are proving convergence, not designing an algorithm. So, it is not necessary that we be able to actually calculate U n . We want a function that measures our progress toward finding the optimum

solution. One convenient function is simply:

<!-- formula-not-decoded -->

where ¯ θ ∗ is the optimal solution. Of course, we don't know F ∗ , so we cannot actually compute U n , but that is not really a problem for us. Note that we can quickly verify that U n ≥ 0. If we can show that U n is a supermartingale, then we get the result that U n converges to a random variable U ∗ , and, of course, we would like to show that U ∗ = 0.

We assume that we are still solving a problem of the form:

<!-- formula-not-decoded -->

where we assume that F ( θ, ω ) is continuous and convex (but we do not require differentiability). We are solving this problem using a stochastic gradient algorithm:

<!-- formula-not-decoded -->

where g n is our stochastic gradient, given by:

<!-- formula-not-decoded -->

We next need to assume some properties of the stochastic gradient g n . Specifically, we need to assume that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (6.95) assumes that on average, the gradient g n points toward the optimal solution ¯ θ ∗ . This is easy to prove for deterministic, differentiable functions. While this may be harder to establish for stochastic problems or problems where F ( θ ) is nondifferentiable, we have not had to assume that F ( θ ) is differentiable.

To show that U n is a supermartingale, we start with:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking conditional expectations on both sides gives:

<!-- formula-not-decoded -->

We now apply assumption 1 in equation (6.95) to argue that the last term on the right hand side of (6.101) is nonnegative. Also, recognizing that E [ U n |F n ] = U n is deterministic (given F n ) allows us to rewrite (6.101) as:

<!-- formula-not-decoded -->

Because of the positive term on the right hand side of (6.102), we cannot directly get the result that U n is a supermartingale. But hope is not lost. We appeal to a neat little trick that works as follows. Let:

<!-- formula-not-decoded -->

We are going to show that W n is a supermartingale. From its definition, we obtain:

<!-- formula-not-decoded -->

Taking conditional expectations of both sides gives:

<!-- formula-not-decoded -->

which is the same as:

<!-- formula-not-decoded -->

We see from equation (6.102) that I ≥ 0. Removing this term, then, gives us the inequality:

<!-- formula-not-decoded -->

This means that W n is a supermartingale. It turns out that this is all we really need because lim n →∞ W n = lim n →∞ U n . This means that:

<!-- formula-not-decoded -->

Now that we have the basic convergence of our algorithm, we have to ask: but what is it converging to? For this result, we return to equation (6.100) and sum it over the values n = 0 up to some number N :

<!-- formula-not-decoded -->

The left hand side of (6.108) is an alternating sum (sometimes referred to as a telescoping sum), which means that every element cancels out except the first and the last. Using this, and taking expectations of both sides gives:

<!-- formula-not-decoded -->

Finally, taking the limit as N →∞ and using the limiting result from equation (6.107) gives:

<!-- formula-not-decoded -->

Because of our supermartingale property (equation 6.107) we can argue that the left hand side of (6.109) is bounded. Next, using assumption 2 (which ensures that g n is bounded) and the requirement that E ∑ ∞ n =1 ( α n ) 2 &lt; ∞ allows us to claim that the first term on the right hand side is also bounded. This means that the second term on the right hand side of (6.109) is also bounded. Let β n = g n +1 ( ¯ θ n -¯ θ ∗ ). We have just shown that:

<!-- formula-not-decoded -->

But, we have required that ∑ ∞ n =1 α n = ∞ . If ∑ ∞ n =0 α n +1 β n is bounded which implies that

<!-- formula-not-decoded -->

which means g n +1 ( ¯ θ n -¯ θ ∗ ) → 0. This can happen in two ways. Either g n +1 → 0 or ( ¯ θ n -¯ θ ∗ ) → 0 (or both). Either way, we win. If g n +1 → 0, then this can only happen because ¯ θ n → ¯ θ ∗ (under our convexity assumption, the gradient can only vanish at the minimum). In general, this will never be the case either because the problem is stochastic or F ( θ ) is not differentiable. In this case, we must have ¯ θ n → ¯ θ ∗ , which is what we were looking for in the first place.

## 6.8.4 Proof of theorem 6.5.1

Proof: Let J ( α t ) denote the objective function from the problem stated in (6.61).

<!-- formula-not-decoded -->

The expected value of the cross-product term, I , vanishes under the assumption of independence of the observations and the objective function reduces to the following form:

<!-- formula-not-decoded -->

In order to find the optimal stepsize, α ∗ t , that minimizes this function, we obtain the first order optimality condition by setting ∂J ( α t ) ∂α t = 0, which gives us,

<!-- formula-not-decoded -->

Solving this for α ∗ t gives us the following result,

<!-- formula-not-decoded -->

We observe that the least squares estimate of the stepsize suggests combining the previous estimate and the new observation in inverse proportion to their mean squared errors. It is well known that the mean-squared error of an estimate can be computed as the sum of its variance and the square of its bias from the true value (Hastie et al. (2001)). The biases in the estimates can be obtained using equations (6.62) and (6.63).

The variance of the observation, ˆ θ t is computed as follows:

<!-- formula-not-decoded -->

We can write the estimate from the previous iteration as:

<!-- formula-not-decoded -->

Assume that α 1 = 1. This means that when we extend the sequence back to the beginning, the first term vanishes, and we are left with:

<!-- formula-not-decoded -->

The expected value of this estimate can be expressed as:

<!-- formula-not-decoded -->

We can compute the variance of the smoothed estimate, ¯ θ n -1 , as follows:

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

We can compute λ t and the variance of the smoothed estimate recursively using:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using these results, we can express the mean squared errors as,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can obtain the desired result by putting together equations (6.115), (6.122) and (6.123).

/square

## 6.9 Bibliographic notes

## 6.9.1 Stochastic approximation literature

Kushner &amp; Clark (1978) , Robbins &amp; Monro (1951) , Kiefer &amp; Wolfowitz (1952) , Blum (1954 b ) , Wasan (1969), Dvoretzky (1956) Pflug (1988) gives an overview of deterministic and adaptive stepsize rules used in stochastic approximation methods.

```
Pflug (1996) Spall: Overview Spall (2003)
```

## 6.9.2 Stepsizes

In forecasting, exponential smoothing is widely used to predict future values of exogenous quantities such as random demands. The original work in this area was done by Brown and Holt. Forecasting models for data with a trend component have also been developed. In most of the literature, constant stepsizes are used (see Brown (1959), Holt et al. (1960)) and Winters (1960)), mainly because they are easy to implement in large forecasting systems. These may be tuned to work well for specific problem classes. However, it has been shown that models with fixed parameters will demonstrate a lag if there are fluctuations in the mean or trend components of the observed data. There are several methods that monitor the forecasting process using the observed value of the errors in the predictions with respect to the observations. For instance, Brown (1963), Giffin (1971), Trigg (1964) and Gardner (1983) develop tracking signals that are functions of the errors in the predictions. If the tracking signal falls outside of certain limits during the forecasting process, either the parameters of the existing forecasting model are reset or a more suitable model is used.

Similar updating techniques that involve stepsizes are used extensively in closely related fields such as reinforcement learning (Jaakkola et al. (1994), Bertsekas &amp; Tsitsiklis (1996)), where convergence of the learning algorithms is dependent on the stepsizes satisfying certain conditions. Darken &amp; Moody (1991) addresses the problem of having the stepsizes evolve at different rates, depending on whether the learning algorithm is required to search for the neigborhood of the right solution or to converge quickly to the true value. However, the rate at which the stepsize changes is a deterministic function.

Darken &amp; Moody (1991) introduce search then converge.

From stochastic programming: Kesten (1958) , Mirozahmedov &amp; Uryasev (1983) , Gaivoronski (1988) , Ruszczy´ nski &amp; Syski (1986)

Kmenta - Best linear unbiased estimators: Kmenta (1997)

```
Hastie et al - machine learning Hastie et al. (2001) Darken and Moody STC Darken et al. (1992)
```

Learning rates for Q-learning: Even-Dar &amp; Mansour (2004)

Jacobs (1988)

Kalman filter: Stengel (1994)

Stochastic gradient stepsizes: Mathews &amp; Xie (1993), Douglas &amp; Mathews (1995) Benveniste: Benveniste et al. (1990), Douglas &amp; Mathews (1995)

## Exercises

In all the exercises below, U is a random variable that is uniformly distributed between 0 and 1.

- 6.1) Let U be a random variable that is uniformly distributed between 0 and 1. Let R = -1 λ lnU . Show that Prob [ R ≤ x ] = 1 -e -λx , which shows that R has an exponential distribution.
- 6.2) Let R = U 1 + U 2 . Derive the probability density function for R .
- 6.3) Let X be an arbitrary random variable (it can be discrete or continuous). Let F ( θ ) = 1 -Prob [ X ≤ x ], and let F -1 ( y ) be its inverse (that is, if y = F ( θ ), then F -1 ( y ) = x ). Show that F -1 ( U ) has the same distribution as X .
- 6.4) Prove that the recursive expression for λ t in equation (6.67) is equivalent to the expression in equation (6.65).
- 6.5) Prove equation 6.46. Use the definition of ν :

<!-- formula-not-decoded -->

Expand the term in the expectation and reduce.

- 6.6) We are going to again try to use approximate dynamic programming to estimate a discounted sum of random variables (we first saw this in chapter 5):

<!-- formula-not-decoded -->

where R t is a random variable that is uniformly distributed between 0 and 100 (you can use this information to randomly generate outcomes, but otherwise you cannot use this information). This time we are going to use a discount factor of γ = . 95. We assume that R t is independent of prior history. We can think of this as a single state Markov decision process with no decisions.

- a) Using the fact that E R t = 50, give the exact value for F 100 .
- b) Propose an approximate dynamic programming algorithm to estimate F T . Give the value function updating equation, using a stepsize α t = 1 /t .
- c) Perform 100 iterations of the approximate dynamic programming algorithm to produce an estimate of F 100 . How does this compare to the true value?
- d) Compare the performance of the following stepsize rules: Kesten's rule, the stochastic gradient adaptive stepsize rule (use µ = . 001), 1 /n β with β = . 85, the Kalman filter rule, and the optimal stepsize rule. For each one, find both the estimate of the sum and the variance of the estimate.
5. 6.7) Consider a random variable given by R = 10 U (which would be uniformly distributed between 0 and 10). We wish to use a stochastic gradient algorithm to estimate the mean of R using the iteration ¯ θ n = ¯ θ n -1 -α n ( R n -¯ θ n -1 ), where R n is a Monte Carlo sample of R in the n th iteration. For each of the stepsize rules below, use equation (6.10) to measure the performance of the stepsize rule to determine which works best, and compute an estimate of the bias and variance at each iteration. If the stepsize rule requires choosing a parameter, justify the choice you make (you may have to perform some test runs).
- a) α n = 1 /n .
- b) Fixed stepsizes of α n = . 05 , . 10 and. 20.
- c) The stochastic gradient adaptive stepsize rule (equations 6.33)-(6.34)).
- d) The Kalman filter (equations (6.54)-(6.58)).
- e) The optimal stepsize rule (algorithm 6.9).
11. 6.8) Repeat (6.7) using

<!-- formula-not-decoded -->

- 6.9) Repeat (6.7) using

<!-- formula-not-decoded -->

- 6.10) We are going to solve a classic stochastic optimization problem known as the newsvendor problem. Assume we have to order x assets after which we try to satisfy a random demand D for these assets, where D is randomly distributed between 100 and 200. If x &gt; D , we have ordered too much and we pay 5( x -D ). If x &lt; D , we have an underage, and we have to pay 20( D -x ).
- a) Write down the objective function in the form min x E f ( x, D ).
- b) Derive the stochastic gradient for this function.
- c) Since the gradient is in units of dollars while x is in units of the quantity of the asset being ordered, we encounter a scaling problem. Let x 0 = 100 and choose as a stepsize α n = 5 /n . Estimate the optimal solution using 100 iterations.

- 6.11) A customer is required by her phone company to commit pay for a minimum number of minutes per month for her cell phone. She pays 12 cents per minute of guaranteed minutes, and 30 cents per minute that she goes over his minimum. Let x be the number of minutes she commits to each month, and let M be the random variable representing the number of minutes she uses each month.
- a) Write down the objective function in the form min x E f ( x, D ).
- b) Derive the stochastic gradient for this function.
- c) Let x 0 = 0 and choose as a stepsize α n = 10 /n . Use 100 iterations to determine the optimum number of minutes the customer should commit to each month.
- 6.12) An oil company covers the annual demand for oil using a combination of futures and oil purchased on the spot market. Orders are placed at the end of year t -1 for futures that can be exercised to cover demands in year t . If too little oil is purchased this way, the company can cover the remaining demand using the spot market. If too much oil is purchased with futures, then the excess is sold at 70 percent of the spot market price (it is not held to the following year - oil is too valuable and too expensive to store).

To write down the problem, model the exogenous information using:

ˆ D t = Demand for oil during year t .

ˆ p s t = Spot price paid for oil purchased in year t .

ˆ p f + t = Futures price paid in year t for oil to be used in year t +1.

The decision variables are given by:

¯ θ f t,t +1 = Number of futures to be purchased at the end of year t to be used in year t +1.

¯ θ s t = Spot purchases made in year t .

- a) Set up the objective function to minimize the expected total amount paid for oil to cover demand in a year t + 1 as a function of ¯ θ f t . List the variables in your expression that are not known when you have to make a decision at time t .
- b) Give an expression for the stochastic gradient of your objective function. That is, what is the derivative of your function for a particular sample realization of demands and prices (in year t +1)?
- c) Generate 100 years of random spot and futures prices as follows:

<!-- formula-not-decoded -->

where U f t and U s t are random variables uniformly distributed between 0 and 1. Run 100 iterations of a stochastic gradient algorithm to determine the number of

futures to be purchased at the end of each year. Use ¯ θ f 0 = 30 as your initial order quantity, and use as your stepsize α t = 20 /t . Compare your solution after 100 years to your solution after 10 years. Do you think you have a good solution after 10 years of iterating?

## Chapter 7

## Discrete, finite horizon problems

A rich array of techniques have evolved in the field of approximate dynamic programming that focuses on problems with discrete states and actions. We use this framework to describe a class of techniques that depend on our ability to enumerate states and actions and to estimate value functions of the 'table look-up variety' where there is an estimate of the value of being in each discrete state.

The techniques that we describe in this chapter are only interesting for problems where the state and action spaces are 'not too big.' While the so-called curse of dimensionality arises in a vast array of applications, there are many problems where this does not happen. A particular class of discrete problems that fall in this category are those that involve managing a single (discrete) asset such as equipment (locomotives, aircraft, trucks, printing presses), people (automobile drivers, equipment operators, a student planning an academic career), or a project (where a set of tasks have to be completed in order). Most game problems (chess, checkers, backgammon, tetris) also fall in this category. These are important problems, but typically have the quality that the state and action spaces are of reasonable size.

There are numerous problems involving the management of a single asset that are important problems by themselves. In addition, techniques for solving more complex, multiasset problems are often solved by decomposing them into single asset problems. As a result, this is the proper foundation for addressing these more complex problems. For example, the most widely used strategy for scheduling airline crews involves solving dynamic programs to schedule each individual crew, and then using an optimization package to determine which set of schedules should be used to produce the best overall solution for the airline.

This chapter focuses purely on finite horizon problems. Since most treatments of this material have been done in the context of infinite horizon problems, a word of explanation is in order. Our justification for starting with a finite horizon framework is based on pedagogical, practical, and theoretical reasons. Pedagogically, finite horizon problems require a more careful modeling of the dynamics of the problem; stationary models allow us to simplify the modeling by ignoring time indices, but this hides the modeling of the dynamics of decisions and information. By starting with a finite horizon model, the student is forced to clearly write down the dynamics of the problem, which is a good foundation for building infinite

horizon models.

More practically, finite horizon models are the natural framework for a vast array of operational problems where the data is nonstationary and/or where the important behavior falls in the initial transient phase of the problem. Even when a problem is stationary, the decision of what to do now depends on value function approximations that often depend on the initial starting state. If we were able to compute an exact value function, we would be able to use this value function for any starting state. For more complex problems where we have to depend on approximations, the best approximation may quite easily depend on the initial state.

The theoretical justification is that certain algorithms depend on our ability to obtain unbiased sample estimates of the value of being in a state by following a path into the future. With finite horizon models, we only have to follow the path to the end of the horizon. With infinite horizon problems, authors either assume the path is infinite or depend on the presence of zero-cost, absorbing states.

## 7.1 Applications

There are a wide variety of problems that can be viewed as managing a single, discrete asset. Some examples include:

- The shortest path problem - Consider the problem faced by a traveler trying to get from home to work over a network with uncertain travel times. Let I be the set of intersections, and let τ ij be a random variable representing the travel time from intersection i to intersection j .
- The traveling salesman problem - Here, a person or piece of equipment has to make a series of stops to do work at each stop (making sales calls, performing repairs, picking up cash from retail stores, delivering gasoline to gas stations). The attributes of our asset would include its location and the time it arrives at a location, but might also include total elapsed time and how full or empty the vehicle is if it is picking up or delivering goods.
- Planning a college academic program - A student has to plan eight semesters of courses that will lead to a degree. The student has five sets of requirements to satisfy (two language courses, one science course, one writing course, four courses in a major field of study, and two courses chosen from four groups of courses to provide breadth). In addition, there is a requirement to finish a minimum number of courses for graduation. The attribute of a student can be described as the number of courses completed in each of the six dimensions (or the number of courses remaining in each of the six dimensions).
- Sailing a sailboat - Sailboats moving upwind have to sail at roughly a 45 degree angle to the wind. Periodically, the boat has to change tack , which requires turning the boat by

90 degrees so the wind will come over the opposite side. Tacks take time, but they are necessary if the wind changes course so the boat can move as much as possible towards the next marker. If the wind moved in a constant direction, it would be easy to plan a triangle that required a single tack. But such a path leads the boat vulnerable to a wind shift. A safer strategy is to plan a path where the boat does not veer too far from the line between the boat and the next mark.

- The unit commitment problem - This is a class of problems that arises in the electric power industry. A unit might be a piece of power generation equipment that can be turned on or off to meet changing power needs. There is a cost to turning the unit on or off, so we must to decide when future power demands justify switching a unit on.
- The fuel commitment problem - A variation on the unit commitment problem is the decision of what fuel to burn in plants that can switch between coal, oil, and natural gas. Here, the issue is not only the switching costs but potential price fluctuations.
- Selling an asset - A very special instance of a discrete asset problem is determining when to sell an asset. We want to sell the asset at the highest price given assumptions about how prices are evolving and the conditions under which it can be sold.

These problems all assume that the only pieces of information are the attributes of the asset itself. Even single asset problems can be hard when there is other information available to make a decision. In our traveling salesman problem, the attribute of the salesman may be her location, but other information can be the status of other requests for a visit.

As we show later, the real complication is not how much information is available to make a decision, but rather how much we capture as an attribute for the purpose of estimating a value function. In the world of approximate dynamic programming, we are not looking for optimal solutions, but for solutions that are better than we would obtain without these techniques. We do not have to put all the dimensions of the state variable when we are estimating a value function; the goal is to choose the elements that contribute the most to explaining the value of being in a state.

## 7.2 Sample models

In this section, we present a series of real problems that involve the management of a single asset. Our goal here is to provide a basic formulation that highlights a class of applications where approximate dynamic programming can be useful.

## 7.2.1 The shortest path problem

An important special case of the single asset problem is the shortest path problem. Actually, any discrete dynamic program can be formulated as a shortest path problem by introducing

a 'super state' that the system has to end up in and by viewing the problem in terms of the evolution of the entire system. However, given the array of real problems that fit the general framework of shortest paths, it seems more natural to work in the context of problems that are naturally formulated this way.

Shortest path problems can usually be formulated in terms of getting an asset from one state to a terminal state in either the least amount of time or at the lowest cost. These are most naturally viewed as finite horizon problems, but we have to handle the possibility that the asset may not make it to the terminal state within the horizon. This situation is easily handled, at least conceptually, by including an end-of-horizon transition from any state into the terminal state with a cost.

Shortest path problems come in a variety of flavors, especially when we introduce uncertainty. Below we review some of the major classes of shortest path problems.

## A deterministic shortest path problem

The best example of a shortest path problem is that faced by a driver making her way through a network. The network consists of a set of intersections I , and at each intersection, she has to decide which link ( i, j ) to progress down. Let q be the node her trip originates at, and let r be her intended destination. Let:

I i = The set of nodes that can be reached directly from intersection

I j = The set of nodes that can reach directly out to intersection j .

```
→ i . ←
```

The forward and backward reachable sets identify the set of links out of, and into, a node (assuming that there is only one link connecting two nodes).

For our presentation, we assume that we are minimizing the cost of getting to the destination, recognizing that the cost may be time. Let:

c ij = The cost of traversing link ( i, j ).

x ij = { 1 If we traverse link ( i, j ) given that we are at intersection i 0 Otherwise

Let v i be the value of being at node i . Here, the value is the cost of getting from node i to the destination node r . If we knew all these values, we could find our optimal path to the destination by simply solving:

<!-- formula-not-decoded -->

Clearly, the value of being at node i should now satisfy:

<!-- formula-not-decoded -->

which says that the cost of getting from i to r should equal the cost of making the best decision out of i considering both the cost of the first link out of i plus the cost of getting from the destination of the link to the final destination of the trip.

This means that our values ( v i ) i ∈I should satisfy the equation:

<!-- formula-not-decoded -->

Equation (7.1) has been widely studied. Since these algorithms are applied to networks with tens or even hundreds of thousands of links, researchers have spent decades looking for algorithms that solve these equations very quickly. One algorithm that is especially poor is a straightforward application of backward dynamic programming, where we iteratively compute:

<!-- formula-not-decoded -->

We initialize v 0 i = M for all i ∈ I except for v 0 r . It turns out that this algorithm will converge in N iterations, where N is the length of the path with the longest number of links from q to r .

A more effective algorithm uses a backward recursion. Starting at the destination node r (where we know that v r = 0), we put node r in a candidate list . We then take the node at the top of the candidate list and look backward from that node. Any time we find a node i where v n i &lt; v n -1 i (that is, where we found a better value to node i ) is added to the bottom of the candidate list (assuming it is not already in the list). After the node at the top of the list is updated, we drop it from the list. The algorithm ends when the list is empty.

## Random costs

There are three flavors of problems with random arc costs:

- Case I - All costs are known in advance: Here, we assume that we have a wonderful realtime network tracking system that allows us to see all the costs before we start our trip.

Case II - Costs are learned as the trip progresses: In this case, we assume that we see the actual arc costs for links out of node i when we arrive at node i .

Case III - Costs are learned after the fact: In this setting, we only learn the cost on each link after the trip is finished.

Let C ij be a random variable representing the cost with expectation ¯ c ij . Let C ij ( ω ) be a sample realization. For each of the three cases, we can solve the problem using different versions of the same dynamic programming recursion:

Case I - All costs known in advance:

Since we know everything in advance, it is as if we are solving the problem using deterministic costs C ij ( ω ). For each sample realization, we have a set of node values that are therefore random variables V i . These are computed using:

<!-- formula-not-decoded -->

We would have to compute v i ( ω ) for each ω ∈ Ω (or a sample ˆ Ω). On average, the expected cost to the destination r would be given by:

<!-- formula-not-decoded -->

Case II - Link costs become known when we arrive at the intersection:

For this case, the node values are expectations, but the decisions are random variables. The node values satisfy the recursion:

<!-- formula-not-decoded -->

Case III - Link costs become known only after the trip:

Now we are assuming that we do not get any new information about link costs as we traverse the network. As a result, the best we can do is to use expected link costs ¯ C ij :

<!-- formula-not-decoded -->

which is the same as our original deterministic problem.

## Random arc availabilities

A second source of uncertainty arises when links may not be available at all. We can handle this using a random upper bound:

<!-- formula-not-decoded -->

The case of random arc capabilities can also be modeled by using random arc costs with C ij = M if U ij = 0, and so the problems are effectively equivalent. The practical difference arises because if a link is not available, we face the possibility that the problem is simply infeasible.

## 7.2.2 Getting through college

The challenge of progressing through four years of college requires taking a series of courses that satisfy various requirements. For our example, we will assume that there are five sets of requirements: two courses in mathematics, two in language, eight departmentals (that is, courses from the department a student is majoring in), four humanities courses (from other departments), and one science course (chemistry, physics, geology, biology). The total number of courses has to add up to 32.

Each semester, a student may take three, four or five courses. Of these courses, a student will normally select courses that also satisfy the various requirements that have to be satisfied prior to graduation.

We can describe the state of our student at the beginning of each semester in terms of the following vector:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We assume that R t is a pre-decision state variable, so that it indicates the information available for making a decision. The student has to make decisions about which courses to

take at the end of each semester. For this problem, we would have eight decision epochs indexed t = (0 , 1 , . . . 7) representing decisions to be made before each semester begins. x 0 represents decisions made at the beginning of her first year. We can represent our decisions using:

D =

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We also have to keep track of which courses were completed 'satisfactorily.' During semester t , the information W t = ( W td ) d ∈D that comes in is whether the student passed or failed the course, where:

<!-- formula-not-decoded -->

We can keep track of the courses that have been completed satisfactorily using:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The decisions have to satisfy certain constraints:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Constraints (7.6) and (7.7) ensure that the student takes between three and five courses each semester. Constraint (7.8) prevents the student from taking a course that has already been completed.

In addition to completing the necessary number of courses while meeting all requirements, the student may prefer some courses over others. These preferences are captured in the contribution function C t ( x t , R t ), which combines a numerical preference for each course. In addition, there is a terminal reward C 8 ( R 8 ) that captures the value of being in state R 8 at the end of eight semesters. It is here that we include penalties for not taking enough courses, or not satisfying one of the requirements.

The state of our system is given by S t = ( R t , P t ) which describes the number of requirements the student has completed, and the courses she has taken. Technically, R t can be computed from P t , but it is useful to represent both in the state variable since P t tells us which courses she is allowed to take (she cannot take the same course twice), while R t indicates the degree to which she is progressing toward graduation.

## 7.2.3 The taxi problem

In our version of the taxi problem, a taxi picks up and drops off customers and tries to maximize his total profit over time (we can view this as a steady state, infinite horizon problem or as a time-dependent, finite horizon problem). The process starts when a customer gets in the cab and tells him where he wants to go. After the cab drops off the customer, he faces the decision of where to go next to wait for a customer (sit and wait? go to a nearby hotel? head back to the airport?).

We let:

```
a t = The attributes of the taxi at the end of a trip. R ta = { 1 If the taxi has attribute a at time t 0 Otherwise R t = ( R ta ) a ∈A
```

We assume that space is discretized into a set of locations, represented by:

I = The set of locations that an empty taxi can move to.

This allows us to represent decisions using:

```
D = The decision to move to a location in I .
```

An element d ∈ D represents a decision to move to a location i d ∈ I . After the cab makes a decision, he sits and waits for a customer to arrive. At a location, a cab will sit if no customers arrive or move to the destination requested by the first customer who arrives and requests service (our cabbie is not allowed to turn down customers). From the perspective of a dispatcher, the cab will call in letting her know that he has a passenger and where he is going to. We can model the information that arrives when the customer boards as:

```
W d t = { 0 No customer arrived and the cabbie remained in his location i The destination of the trip W f t = The fare paid for the trip.
```

Thus, W d t &gt; 0 means that a customer arrived, in which case W d t is the destination requested by the customer. If W d t = 0, then we would assume that W f t = 0.

In this problem, decisions are made at the end of a trip. If the taxi decides to change locations, we wait until the cab arrives at his new location, and then information arrives that determines whether he moves to a new location and how much money he makes. At the end of any time period, the taxi can choose to either sit and wait another time period or move to another location.

We can model this problem in discrete time or at specific events ('decision epochs'). For example, we might assume that once a cab has decided to wait for a customer, he has to wait until a customer actually arrives. In this case, the only decision points are at the end of a trip (which need not occur at discrete time points). If the taxi is at location i , then he will wait a random amount of time and then serve a customer who will take him to location j with probability p ij . If the cab decides to move to location k , he will then move from k to j with probability p kj . In the language of Markov decision processes, we would say that the probability that a cab at i goes to location j is given by p ij ( x ) where x = ( x d ) d ∈D captures the decision to stay or move to an alternate location. If the cab could not reposition to another location, we would have a classic example of a Markov chain (or, to be more precise, a semi-Markov process, because the time during which the cab waits may be an arbitrary probability distribution). Since we can make decisions that effectively change the probability of a transition, we have a Markov decision process.

## 7.2.4 Selling an asset

An important problem class is determining when to sell an asset. Let:

R t = { 1 If we are holding the asset at time t 0 Otherwise x t = { 1 If we sell the asset at time t 0 Otherwise p t = The price that we obtain for the asset if we sell at time t . c t = The contribution from holding the asset during time interval t . ˆ p t = The change in price that arrives during time interval t . T = Time by which the asset must be sold.

The state of our system is given by:

<!-- formula-not-decoded -->

which evolves according to the equation:

<!-- formula-not-decoded -->

In our modeling strategy, the price process (ˆ p t ) can be quite general. There is an extensive literature on the asset selling process that assumes the sequence (ˆ p t ) is independent where the variance is either constant ('constant volatility') or time varying ('dynamic volatility'). We have a true state variable, and therefore a Markovian system, even if ˆ p t depends on p t -1 . If this assumption is violated (the price changes might depend on prices before time t -1), then S t is not a complete state variable. But even this does not prevent us from developing good approximate algorithms. For example, we may obtain price information from a real system (where we may not even have a mathematical model of the information process). We may still use our simpler state variable as a basis for building a value function approximation.

Our one-period contribution function is given by:

<!-- formula-not-decoded -->

Given a family of decision rules ( X π ( S t )) π ∈ Π , our problem is to solve:

<!-- formula-not-decoded -->

## 7.3 Strategies for finite horizon problems

In this section, we sketch the basic strategies for using forward dynamic programming methods to approximate policies and value functions. The techniques are most easily illustrated using a post-decision state variable, after which we describe the modifications needed to handle a pre-decision state variable. We then describe a completely different strategy, known as Q -learning , to overcome the problem of approximating the expectation imbedded in a pre-decision framework.

## 7.3.1 Value iteration using a post-decision state variable

There are two simple strategies for estimating the value function for finite horizon problems. The first uses a single pass procedure. Here, we step forward in time using an approximation of the value function from the previous iteration. After solving a timet problem, we update the value function for time t . The procedure is illustrated in figure 7.1.

<!-- formula-not-decoded -->

Figure 7.1: Single pass version of the approximate dynamic programming algorithm.

A few notes are in order. We show in step 1 that we can choose the entire sample path before we start stepping forward through time. In many applications, random variables are correlated across time, or are dependent on the state. Frequently, the relevant information is very state dependent. For example, in our stochastic shortest path problem, the only information we want to see when we are at node i are the costs on links emanating from node i (so there is no point in generating costs on all the other links). For these applications, it will be much more natural to generate information as we progress.

After solving the problem at time t , we obtain an updated estimate of the value of being in state R t , which we call ˆ v t . In step 2c, we then smooth over these to obtain an updated estimate of ¯ V t -1 . The change in the time index can be confusing at first. ˆ v t is indexed by t because it is a random variable that depends on information from time interval t . The smoothing in step 2c has the effect of approximating the expectation over this information, producing an estimate of a function that depends only on information up through time t -1.

Figure 7.1 is known as a single pass procedure because all the calculations are finished at the end of each forward pass. The updates of the value function take place as the algorithm progresses forward in time. The algorithm is fairly easy to implement, but may not provide the fastest convergence. As an alternative, we can use a double pass procedure, which is illustrated in figure 7.2. In this version, we step forward through time creating a trajectory of states, actions and outcomes. We then step backwards through time updating the value of being in a state using information from the same trajectory in the future.

```
Step 0. Initialize ¯ V 0 t , t ∈ T . Step 0a. Set n = 1. Step 0b. Initialize R 1 0 . Step 1: Choose a sample path ω n . Step 2: Do for t = 1 , 2 , . . . , T : Step 2a: Choose ω n t = W t ( ω n ). Step 2b: Find: x n t = arg max x t ∈X t C t ( R n t -1 , ω n t , x t ) + γ ¯ V n -1 t ( R M ( R n t -1 , ω n t , x t Step 2c: Compute R n t = R M ( R n t -1 , ω n t , x n t ). Step 3: Do for t = T, T -1 , . . . , 1: Step 3a: Compute ˆ v n t using the decision x n t from the forward pass: ˆ v n t = C t ( R n t -1 , ω n t , x n t ) + γ ˆ v n t +1 Step 3b: Update the value function approximations: ¯ V n t -1 ( R n t -1 ) = (1 -α n ) ¯ V n -1 t -1 ( R n t -1 ) + α n ˆ v n t Step 4. Increment n . If n ≤ N go to Step 1. Step 5: Return the value functions ( ¯ V n t ) T t =1 .
```

Figure 7.2: Double pass version of the approximate dynamic programming algorithm

```
))
```

The result of our backward pass is ˆ v n t , which is the contribution from the sample path ω n and a particular policy. Our policy is, quite literally, the set of decisions produced by the value function approximation ¯ V n . In the double pass algorithm, if we repeated step 4 over and over (for a particular initial state R 0 ), ¯ V n 0 ( R n 0 ) would eventually converge to the correct estimate of the value of being in state R 0 and following the policy produced by the approximation ¯ V n -1 t . As a result, although ˆ v n t is a valid, unbiased estimate of the value of being in state R n t at time t and following the policy produced by ¯ V n , we cannot say that ¯ V n t ( R n t ) is an unbiased estimate of the value of being in state R n t . Rather, we can only talk about the properties of ¯ V n t in the limit.

## 7.3.2 Value iteration using a pre-decision state variable

Our presentation up to now has focused on using a post-decision state variable, which gives us a much simpler process of finding decisions. It is much more common in the literature to formulate problems using the pre-decision state variable. The concepts we have described up to now all apply directly when we use a pre-decision state variable. For this section, we let R t be the pre-decision state variable and represent the dynamics using the equation

<!-- formula-not-decoded -->

We remind the reader that we change the order of the arguments in the function R M ( · ) when we use a pre-decision state variable. The arguments reflect the order in which events happen: we see the state R n , we make a decision x n , and then we see new information W n +1 . Since we always use only one form of the transition function in any particular application, we do not introduce a different functional name for reasons of notational simplicity. It is important for the reader to keep in mind whether a model is being formulated using the pre or post-decision state variable.

A sketch of the general algorithm is given in figure 7.3. Although the algorithm closely parallels that for the post-decision state variable, there are important differences. The first one is how decisions are made. We can assume that we have a policy π that determines how we make a decision given the state R , but in the field of approximate dynamic programming, we generally have to resort to solving approximations of:

<!-- formula-not-decoded -->

In practice, we typically replace the value function V () with an approximation ¯ V (), and we approximate the expectation by taking a sample of outcomes and solving:

<!-- formula-not-decoded -->

A second difference is that ˆ v n is now an approximation of an expectation. We can still do smoothing to update ¯ V n , but the choice of stepsize should reflect the size of the sample ˆ Ω.

Students should pay particular attention to the indexing over time and iterations. In equation (7.11), we smoothed in ˆ v n t to update our estimate of ¯ V n t -1 . In equation (7.14), we smoothed in ˆ v n t to update our estimate of ¯ V n t . The reason is that in equation (7.14), ˆ v n t is actually an approximation of the expectation of ¯ V t (rather than just a sample realization).

## 7.3.3 Q -learning

Return for the moment to the classical way of making decisions using dynamic programming. Normally we would look to solve:

<!-- formula-not-decoded -->

Solving equation (7.15) can be problematic for two different reasons. The first is that we may not know the underlying distribution of the exogenous information process. If we do not know the probability of an outcome, then we cannot compute the expectation. These are problems where we do not have a model of the information process. The second reason is that

## Step 0:

Step 1: Do while n ≤ N

Step 3: V n t ) T t =1

<!-- formula-not-decoded -->

Figure 7.3: Approximate dynamic programming using a pre-decision state variable.

while we may know the probability distribution, the expectation may be computationally intractable. This typically arises when the information process is characterized by a vector of random variables.

We can circumvent this problem by replacing the expectation with a (not-too-large) random sample of possible realizations, which we represent by ˆ Ω. We may construct ˆ Ω so that each outcome ω ∈ ˆ Ω occurs with equal probability (that is, 1 / | ˆ Ω | ), or each may have its own probability p ( ω ). Using this approach, we would make a decision using:

<!-- formula-not-decoded -->

Solving equation (7.16) can be computationally difficult for some applications. For example, if x t is a vector, then solving the myopic problem (the value function is zero) may be a linear or integer program of significant size (that is solvable, but not easily). Solving it over a set of scenarios ˆ Ω makes the problem dramatically larger.

One thought is to solve the problem for a single sample realization:

<!-- formula-not-decoded -->

The problem is that this means we are choosing x t for a particular realization of the future information W t +1 ( ω ). This problem is probably solvable, but it is not likely to be a reasonable

approximation (we can always do much better if we make a decision now knowing what is going to happen in the future). But what if we choose the decision x t first (for example, at random), and then compute the cost? Let the resulting cost be represented using:

<!-- formula-not-decoded -->

We could now smooth these values to obtain:

<!-- formula-not-decoded -->

We use ¯ Q n t ( R t , x t ) as an approximation of:

<!-- formula-not-decoded -->

The functions Q t ( R t , x t ) are known as Q -factors and they capture the value of being in a state, and taking a particular action. We can now choose an action by solving:

<!-- formula-not-decoded -->

This strategy is known as Q -learning. The complete algorithm is summarized in figure 7.4. A major advantage of the strategy is that we do not have to compute the expectation in equation (7.15), or even solve approximations of the form in equation (7.16). The price of this convenience is that we have significantly enlarged the statistical problem. If we let R be the state space and X be the action space, this implies that we have to learn |R| × |X| parameters. For multidimensional problems, this can be a daunting task, and unlikely to be of practical value.

But there is another application. Assume, for example, that we do not know the probability distribution of the exogenous information, such as might arise with a control problem running in real time. If we choose an action x t (given a state R t ), the physical process tells us the contribution C t as well as the next state R t +1 . Implicit in the generation of the state R t +1 is both the exogenous information as well as the transition function. For problems where the state and action spaces are not too large, but where we do not have a model of the information process or transition function, Q -learning can be a valuable tool.

The challenge that we face with Q -learning is that it is replacing the problem of finding a value function V ( R ) with that of finding a function Q ( R,x ) of both the state and the action. If we are working on a problem of a single resource with a relatively small attribute vector a (recall that with a single resource, the resource state space is the same as the attribute space) and not too many decision types d , this technique should work fine. Of course, if the state and action spaces are small, we can use standard backward dynamic programming techniques, but this assumes that we have a one-step transition matrix. It is for this reason that some authors describe Q -learning as a technique for problems where we are missing the transition matrix (random outcomes come from an exogenous source).

Step 0: Initialization:

Step 0a. Initialize an approximation for the value function ¯ Q 0 t ( R t , x t ) for all states R t and decisions x t ∈ X t , t ∈ T .

Step 0b. Set n = 1.

Step 0c. Initialize R 1 0 .

Step 1: Choose a sample path ω n .

Step 2: Do for t = 1 , 2 , . . . , T :

<!-- formula-not-decoded -->

Step 2b.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2d: Find the next state:

<!-- formula-not-decoded -->

Step 3. Increment n . If n ≤ N go to Step 1.

Step 4: Return the Q-factors ( ¯ Q n t ) T t =1 .

Figure 7.4: A Q -learning algorithm.

Q -learning has been applied to a variety of problems. One famous illustration involved the management of a set of elevators. However, there are many problems (in particular those involving the management of multiple assets simultaneously) where the state space is already large, and the action space can be many times larger. For these problems, estimating a function Q ( R,x ) even when R and x have small dimensions would be completely intractable.

There is a cosmetic similarity between Q -learning and approximate dynamic programming using a post-decision state variable. The post-decision state variable requires finding ¯ V x,n -1 t ( R x t ) and then finding an action by solving:

<!-- formula-not-decoded -->

Since we can write R t as a function of R x t -1 and W t , we can replace ( R x t -1 , W t ( ω n )) with the pre-decision state R t ( ω n ), giving us:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is a form of Q -factor computed using the post-decision value function. Although it is not the same as the original Q -factor, it is still a function that estimates the value of being in a particular state and taking an action. Thus, estimating a Q -factor can be viewed as the same as learning a post-decision value function.

Viewed this way, Q -learning is easy to confuse with learning a post-decision value function. Computationally, however, they are quite different. In Q -learning, we face the challenge of estimating |R| × |X| parameters. If R x is the state space for the post-decision state variable, we face the problem of estimating only |R x | parameters if we use the post-decision state variable. In many applications, |R x | &lt;&lt; |R| . What appears to be the same on paper can be quite different computationally.

As an illustration, consider a simple problem of selling an asset at the highest price. The pre-decision state variable is S t = ( R t , p t ) where R t = 1 if we are holding the asset, and 0 otherwise. p t is the price offered for the asset at time t , which we can accept or reject. Assume that the price p t is independent of all prior history, and that we can represent the set of possible prices by values from 1 to 100. When we include the state that we have sold the asset, we have 101 states. With Q -learning, we would have to learn Q ( S, x ), which gives us two actions for each state with R = 1, for a total of 201 state-action pairs.

Now consider what happens when we use a post-decision state variable. The post-decision state S x t = ( R x t ), where R x t = 1 means we are still holding the asset after the decision is made. Now we only need to learn V x t ( R x t ) for R x t ∈ (0 , 1).

A similar example arises with our nomadic trucker problem. Here, S t = ( R t , L t ) where R ta = 1 if the trucker has attribute a , and L t is a vector where L tb is the number of loads with attribute b ∈ B . Thus, the dimensionality of R t is |A| and the dimensionality of L t is |B| , which means that |S| = |A| × |B| . |A| and |B| may be on the order of hundreds to tens of thousands (or more). If we use Q -learning, we have to estimate Q ( S, x ) for all x ∈ X , which means we have to estimate |A| × |B| × |X| parameters. With a post-decision state vector, we only have to estimate V x t ( R x t ), where the dimension of R x t is |A| .

There is an important theoretical distinction as well. Because we compute Q ( R,x ) for each x (in Q -learning), we control the choice of decision which produces powerful convergence results for Q -learning that do not exist when we use a post-decision state variable. The limitation of algorithms which exogenously choose a decision is that they are not going to be effective on problems with large action spaces. For example, in high-dimensional asset allocation problems, x t may be a vector with thousands (even tens of thousands) of dimensions. This is an example where a provably convergent algorithm is unlikely to be useful in practice.

## 7.4 Temporal difference learning

The previous section introduced two variations on a simple forward dynamic programming procedure. These variants can be viewed as special cases of a general class of methods known as temporal difference learning (often abbreviated as 'TD learning'). Originally introduced formally in Sutton (1988), the method is widely used in approximate dynamic programming and a rich theory has evolved around it.

## 7.4.1 The basic idea

This approach is most easily described using the problem of managing a single asset that initially has attribute a t , then information W t +1 becomes available and finally we make decision d t +1 (using policy π ), which produces an asset with attribute a t +1 = a M ( a t , W t +1 , d t +1 ). The contribution of this transition is given by C t +1 ( a t , W t +1 , d t +1 ). Imagine, now, that we continue this until the end of our horizon T . For simplicity, we are going to drop discounting. In this case, the contribution along this path would be:

<!-- formula-not-decoded -->

This is the contribution of following the path produced from a combination of the information process produced by outcome ω (this determines W t +1 , W t +2 , . . . , W T ) and policy π . This is an estimate of the value of a resource that starts with attribute a t , for which we follow the decisions produced by policy π until the end of our horizon.

Reverting back to our vector notation (where we use state R t and decision x t ), our path cost is given by:

<!-- formula-not-decoded -->

ˆ v n is an unbiased sample estimate of the value of being in state R t and following policy π . We can use our standard stochastic gradient algorithm to estimate the value of being in state R t using:

<!-- formula-not-decoded -->

We can obtain a richer class of algorithms by breaking down our path cost in (7.22) by using:

<!-- formula-not-decoded -->

where ¯ V n -1 T ( R T ) = 0 (this is where our finite horizon model is useful). Substituting (7.25) into (7.23) gives:

<!-- formula-not-decoded -->

Let:

<!-- formula-not-decoded -->

The terms D τ are called temporal differences . These are, of course, stochastic gradients. The name reflects the historical development within the field of approximate dynamic programming. We can think of each term in (7.26) as a correction to the estimate of the value function. It makes sense that updates farther along the path should not be given as much weight as those earlier in the path. As a result, it is common to introduce an artificial discount factor λ , producing updates of the form:

<!-- formula-not-decoded -->

Updates of the form given in equation (7.27) produce an algorithm that is known as TD( λ ) (or, temporal difference learning with discount λ ). We could let λ = γ , but in practice, it is common to use λ &lt; γ to produce a form of heuristic discounting that accounts for the fact that we are typically following a suboptimal policy. If λ = 0, then we get the one-period look-ahead update that is illustrated in figure 7.1. If λ = 1, then we are directly using the entire path (for a specified number of steps).

The updating formula in equation (7.27) requires that we step all the way to the horizon before updating our estimates of the value. There is, however, another way of implementing the updates. The temporal differences D τ are computed as the algorithm steps forward in time. As a result, our updating formula can be implemented recursively. Assume we are at time t ′ in our simulation. We would simply execute:

<!-- formula-not-decoded -->

When we reach time t ′ = T , our value functions would have undergone a complete update. We note that at time t ′ , we need to update the value function for every t ≤ t ′ .

## 7.4.2 Variations

There are a number of variations of temporal difference learning. We have presented the algorithm in the context of a finite horizon problem where we update the value of being in a state by always following policy π until we reach the end of our horizon. This is equivalent to our double pass procedure in figure 7.2. The only difference is that the double pass algorithm implicitly assumes that our policy is determined by our approximate value function (which is generally the case) whereas it could, in fact, be some rule (our policy) that simply specifies the action given the state.

Another strategy is to look out to a time period T = t + τ that is not necessarily the end of our horizon. We would then normally use an approximation of the value function V T ( R T ) in equations (7.25) and (7.26). If we use this idea but only look one period into the future, we are effectively using our single pass algorithm (figure 7.1).

## 7.5 Monte Carlo value and policy iteration

In the discussion above, we were a little vague as to how decisions were being made as we moved forward in time. We were effectively assuming that decisions were determined by some 'policy' π that we normally represent as a decision function X π . When we compute ∑ T -1 τ = t λ τ -t D t + τ , we effectively update our estimate of the value function when we follow policy π . But, we make decisions by solving approximations of the form given in equation (7.30). As a result, we mix the estimation of the value of being in a state and the process of choosing decisions based on the value.

It might be a bit cleaner if we wrote our decision problem using:

<!-- formula-not-decoded -->

for a given function ¯ V π ( R ). Fixing ¯ V π determines the decision we are making given our state and information, so fixing ¯ V π is equivalent to fixing the decision function (or equivalently,

```
Step 0: Initialization: Step 0a. Initialize ¯ V π, 0 t , t ∈ T . Step 0b. Set n = 1. Step 0c. Initialize R 1 0 . Step 1. Do while n ≤ N : Step 2. Do while m ≤ M : Step 3: Choose a sample path ω m . Step 4: Do for t = 1 , 2 , . . . , T : Step 4a: Choose ω t m = W t ( ω m ). Step 4b: Solve: ˆ v m t = max x t ∈X t C t ( R m t -1 , ω m t , x t ) + γ ¯ V π,n -1 t ( R M ( R m t -1 , ω m t , x t )) and let x t m be the value of x t that solves (7.30). Step 4c: Compute: R n t = R M ( R n t -1 , ω m t , x m t ) . Step 3d: Update the value function: ¯ V t m -1 ( R m t -1 ) = (1 -α m ) ¯ V m -1 t -1 ( R m t -1 ) + α m ˆ v m t Step 5. Update the policy: ¯ V π,n t ( R t ) = ¯ V t M ( R t ) ∀ R t Step 6: Return the value functions ( ¯ V π,n t ) T t =1 .
```

<!-- formula-not-decoded -->

Figure 7.5: Hybrid value/policy iteration in a Monte Carlo setting

the policy we are following). We can then use any of the family of temporal differencing methods to estimate the value of being in a state while following policy π . After a while, our estimates of the value of being in a state will start to converge, although the quality of our decisions is not changing (because we are not changing ¯ V π ). We can reasonably assume that we will make better decisions if the functions ¯ V π are close to our estimates of the value of being in a state, given by our best estimates ¯ V . So, we could periodically set ¯ V π = ¯ V and then start the process all over again.

This strategy represents a process of estimating value functions (value iteration) and then updating the rules that we use to make decisions (policy iteration). The result is the Monte Carlo version of the hybrid value-policy iteration, which we presented in chapter 4. The overall algorithm is summarized in figure 7.5. Here, we use n to index our policy-update iterations, and we use m to index our value-update iterations. The optimal value of the inner iteration limit M is an open question. The figure assumes that we are using a TD(0) algorithm for finding updates (steps 3 c and 3 d ), but this can easily be replaced with a general TD( λ ) procedure.

```
Step 0: Initialization: Step 0a. Initialize ¯ V π, 0 t , t ∈ T . Step 0b. Set n = 1. Step 0c. Initialize R 1 0 . Step 1. Do while n ≤ N : Step 2. Do while m ≤ M : Step 2a: Choose a sample path ω m . Step 2b: Initialize ˆ v m = 0 Step 3: Do for t = 1 , 2 , . . . , T : Step 3a: Choose ω t m = W t ( ω m ). Step 3b: Solve: x m t = arg max x t ∈X t C t ( R m t -1 , ω m t , x t ) + γ ¯ V π,n -1 t ( R M ( R m t -1 , ω m t , x t )) (7.31) Step 3c: Compute: R n,m t = R M ( R n,m t -1 , ω m t , x m t ) . Step 4: Do for t = T, T -1 , . . . , 1: Step 4a: Accumulate the path cost: ˆ v m t := C t ( R n,m t -1 , ω , m x m t ) + γ ˆ v m t +1 Step 4b. Update approximate value of the policy starting at time t : ¯ V n,m t ( R n,m t ) = (1 -α m ) ¯ V n,m -1 t ( R n,m t ) + α m ˆ v m t where we typically use α m = 1 /m . Step 5: Update the policy: ¯ V π,n t ( R t ) = ¯ V n,M t ( R t ) ∀ t = 1 , . . . , T Step 6: Return the value functions ( ¯ V π,N t ) T t =1 .
```

Figure 7.6: An approximate dynamic version of policy iteration

## 7.6 Policy iteration

We can create an approximate dynamic programming version of policy iteration, as shown in figure 7.6. In it, we sweep forward in time using a set of value functions ¯ V π,n -1 t to make decisions. We then add up the contributions along this path until the end of the horizon, which is stored as ˆ v m . We repeat this exercise using the same value functions ¯ V π,n -1 t but using a different sample realization ω m for m = 1 , 2 , . . . , M . The cost of each trajectory, ˆ v m , is then smoothed (typically averaged) into an estimate ¯ V n,m t ( R n t ). After M repetitions of this process, we obtain ¯ V n,M t ( R n t ) which we then use to produce a new policy.

As M → ∞ , this algorithm starts to look like real policy iteration. The problem is that we are not controlling what states we are visiting. As a result, we cannot obtain any guarantees that the value functions are converging to the optimal solution.

## 7.7 State sampling strategies

Dynamic programming is at its best when decisions can be made at one point in time based on a value function approximation of the future. In the early iterations, we generally have to depend on fairly poor approximations of the value function. The problem we typically encounter is that the states we visit depend on our value function approximations, which in turn depend on the states we visit. If we depend purely on our value function approximations to determine which state we visit next, it is very easy to get caught in a local optimum, where poor approximations of the value of being in some states prevent us from revisiting those states.

In this section, we review some strategies for overcoming this classic chicken-and-egg problem. In later chapters, we partially overcome these problems by using different methods for approximating the value function and by exploiting problem structure.

## 7.7.1 Sampling all states

If we are managing a single asset, our state space is given by the size of the attribute space |A| . While this can be extremely large for more complex assets, for many problems, it may be small enough that we can enumerate and sample at every iteration. If this is the case, it may be possible for us to use the classical backward dynamic programming techniques described in chapter 4. However, we may still encounter our 'second curse of dimensionality': computing the expectation (when managing a single asset, the action space is typically not too large).

For example, consider our nomadic trucker example (section 3.4). Here, the attribute vector might consist of just the location (a typical discretization of the continental United States is 100 regions), but it may include other attributes such as a fleet type (for this discussion, assume there are five fleet types), producing an attribute space of 500, which is not large at all. Each time the trucker enters a region, he faces the random realization of market demands out of that region to all 100 regions (a demand can originate and terminate in the same region). Taking expectations over this much larger vector of market demands is where the problem can become quite hard.

Since the attribute space is not that large, it is not unreasonable to think of looking over all the states of our asset and using Monte Carlo methods to sample the value of being in each state at every iteration. An illustration of such an algorithm for approximating the value functions is given in figure 7.7. On the 3 Ghz chips available at the time of this writing, this technique can be used effectively when the attribute space ranges up to 50,000 or more, but it would generally be impractical if the attribute space were in the millions.

Algorithms which loop over all states are referred to as synchronous because we are synchronizing the updates across all the states.

If we are dealing with more than one asset at a time, we have to work in terms of the resource state space rather than the attribute space. Multiple asset problems typically exhibit extremely large state spaces (even 'small' problems can have state spaces that are

Figure 7.7: A single-asset learning algorithm sampling the entire attribute space

```
Step 0. Initialize ¯ V 0 t , t ∈ T . Step 1. Do while n ≤ N : Step 2. Do for t = 1 , . . . , T : Step 3. Choose ω t . Step 4. Do for all a ∈ A : Step 4a. Set R t = 0, R ta = 1. Step 4b. Solve: ˆ v n t = max x t ∈X t C t ( R n t -1 , ω n t , x t ) + γ ¯ V n -1 t ( R M ( R n t -1 , ω n t , x t )) Step 4c. Update the value function: ¯ V n t -1 ( R n t -1 ) = (1 -α n ) ¯ V n -1 t -1 ( R n t -1 ) + α n ˆ v n t
```

Figure 7.7: A single-asset learning algorithm sampling the entire attribute space.

Figure 7.8: Illustration of a tree search, branching forward in time along multiple trajectories.

<!-- image -->

in the 10 10 -10 100 range). In chapter 15, we describe techniques for solving multiple asset problems that involve successively approximating single asset problems.

## 7.7.2 Tree search

It may not be practical to search the entire attribute space simply because it is far too large or because enumerating all combinations of the attribute vector produces attributes that would simply not happen in practice. If we are not able to sample the entire attribute space, we face the problem that we may have to make a decision about what to do with an asset based on a very poor estimate of the value of being in a particular state in the future.

A strategy for overcoming this is to use a tree search strategy, which is illustrated using our dynamic travelling salesman in figure 7.8. Assume we have an asset with initial attribute

## Procedure EvaluateV(t,a)

Step 0. Initialize: Given initialize attribute a , increment tree depth m := m +1. Step 1. For all d ∈ D a : If m&lt;M : Do: Step 3a. Set a ′ = a M ( t, a, d ) and t ′ = t + τ M ( t, a, d ). Step 3b. If ¯ V n ( t ′ , a ′ ) = null: % Use tree search to evaluate the value of the asset at the destination. ¯ V n ( t ′ , a ′ ) = EvaluateV ( t ′ , a ′ ) Step 3c. Find the value of the decision: c d = C ( a, d ) + γ ¯ V n ( t ′ , a ′ ) Else: % If we have reached the end of our tree search, just use the immediate contribution. c d = C ( a, d ) Endif Step 4. Find the value of the best decision using:

<!-- formula-not-decoded -->

Figure 7.9: A tree search algorithm.

a n, 0 = a n at location A, and we are contemplating a decision d 0 that would produce an asset with attribute a n, 1 d 0 at location B, whose value is approximated as ¯ V n ( a n, 1 d 0 ). Assume now that we have no observations of the attribute a n, 1 d 0 , or we have decided that our estimate is not good enough (we revisit this topic in much greater depth in chapter 10). We can quickly get an estimate of a resource being in a particular state by searching outward from a n, 1 d 0 , and evaluating each decision d 1 ∈ D a n, 1 d 0 (location C in our figure). To evaluate decision d 1 , we will need an estimate of the value of the resource with attribute a n, 2 d 1 produced by each decision d 1 .

An outline of a tree search algorithm is shown in figure 7.9. These algorithms are most naturally implemented as recursive procedures. Thus, as we enter the procedure EvaluateV ( t, a ) with an attribute vector a , we enumerate the set of potential decisions D a . If the depth of our tree search is less than M , we use our terminal attribute function a M ( t, a, d ) to determine the attribute of the asset after being acted on by decision d and call the function over again.

Tree search is, of course, a brute force strategy. It will work well for small problems such as asset selling, where the decision sets are very small (but specialized problems such as asset selling also yield to other analysis strategies). It can also work well for problems where reasonable approximations can be obtained with very shallow trees (our nomadic trucker problem would be a good example of this problem class). Deeper trees are needed if we need a number of steps before learning whether a decision is good or bad. For most problems, tree search has to be significantly truncated to keep run times reasonable, which means that it is an approximation of last resort.

## Procedure MyopicRollOut(t,a)

Step 0. Initialize: Given initialize attribute a , increment tree depth m := m +1. Step 1. Find the best myopic decision using: ¯ d = max d ∈D a C ( a, d ) Step 2. If m&lt;M : Do: Step 3a. Set a ′ = a M ( t, a, ¯ d ) and t ′ = t + τ M ( t, a, ¯ d ). Step 3b. If ¯ V n ( t ′ , a ′ ) = null: ¯ V n ( t ′ , a ′ ) = MyopicRollOut ( t ′ , a ′ ) Step 3c. Find the value of the decision: c ¯ d = C ( a, ¯ d ) + γ ¯ V n ( t ′ , a ′ ) Endif; Else: c ¯ d = C ( a, ¯ d ) Endif Step 4. Approximate the value of the state using: MyopicRollOut = c ¯ d

Figure 7.10: Approximating the value of being a state using a myopic policy.

## 7.7.3 Rollout heuristics

For problems where the decision sets are of moderate to large sizes and where a very shallow tree does not work well, another strategy is to use an approximate policy to guide the choice of decisions. The easiest way to illustrate the idea is by using a myopic policy, which is depicted in figure 7.10.

The advantage of this strategy is that it is simple and fast. It provides a better approximation than using a pure myopic policy since in this case, we are using a myopic policy to provide a rough estimate of the value of being in a state. This can work quite well, and it can work very poorly. The performance is very situation dependent.

It is also be possible to choose a decision using the current value function approximation, as in:

<!-- formula-not-decoded -->

where t ′ and a ′ are computed as in Step 3a. We note that we are most dependent on this logic in the early iterations when the estimates ¯ V n may be quite poor, but some estimate may be better than nothing.

## 7.8 Ataxonomy of approximate dynamic programming strategies

There are a wide variety of approximate dynamic programming algorithms. The algorithms reviewed in this chapter provide a sample of some of the variations. It is easier to get a sense of the diversity of strategies by creating a taxonomy based on the different choices that must be made in the design of an algorithm. Below is a summary of the dimensions of an algorithmic strategy

## 1) Choice of state variable:

- a) Use traditional pre-decision state variable.
- b) Use state-action combination (estimate the value of a state-action pair using Q -learning).
- c) Use the post-decision state variable.

## 2) State sampling:

- a) Asynchronous (choose specific states). Asynchronous state sampling has a number of variants:
- i) Pure exploitation - Choose the states based on a greedy policy.
3. ii) Decision-based exploration - From a given state, choose a decision at random which produces a downstream state.
4. iii) Uniform state-based sampling - Randomly choose a state from a uniform distribution.
5. iv) Nonuniform state-based sampling - Choose a state based on an exogenous probability distribution.
- b) Synchronous (loop over all states).
- 3) Average vs. marginal values - Depending on the application we may:
- a) Estimate the value of being in a state (may be used for single or multiple asset management problems).
- b) Estimate the marginal value of an additional asset with a particular attribute (multiple asset management problems).
- 4) Representation of the value function:
- a) Use a discrete table-lookup representation. Variations include:
- i) The value function is represented using the original state variable.
13. ii) The value function is represented using an aggregated state variable.
14. iii) The value function is represented using a weighted combination of a family of aggregations.

- b) Use a continuous approximation parameterized by a lower-dimensional parameter vector θ . For asset allocation problems, these approximations may be:
- i) Linear in R ta .
3. ii) Nonlinear (or piecewise linear) and separable in R ta .
4. iii) More general functions.
- c) For more general problems, we may represent the state variable using a smaller set of basis functions ( φ b ( S t )) b ∈B . These can be divided into two broad classes:
- i) The approximation is linear in the parameters, which is to say that ¯ V ( S t ) = ∑ b ∈B θ b φ b ( S t ).
7. ii) The approximation is nonlinear in the parameters.
- 5) Obtaining a Monte Carlo update of a value function from a state s (ˆ v ( s )):
- a) Perform a single step update. If we use a post-decision state variable, this would be computed using ˆ v t -1 ( s t -1 ) = max x ( C ( s t -1 , ω, x ) + γ ¯ V t ( S t ( s t -1 , ω, x )) ) . This is TD(0), which we have represented as our single-pass algorithm.
- b) Simulate a policy over an infinite horizon (conceptually) or until the end of the planning horizon (time period T ), computing temporal differences D τ multiplied by λ (this is TD( λ )).
- c) Simulate a policy for up to T ph time periods into the future, computing temporal differences factored by λ (finite horizon TD( λ )). The last temporal difference includes a value function approximation. Variations:
- i) Use the trajectory to update only the starting state.
13. ii) Use all the intermediate partial trajectories to update all the states visited along the trajectory.
- 6) Number of samples - We have to decide how well to estimate the value of the policy produced by the current value function approximation.
- a) Update the value function after a single realization (single step, T steps or until the end of the horizon).
- b) Average (or smooth) N forward trajectories before updating ¯ v .
- c) Average (or smooth) an infinite number of forward trajectories.
- 7) Updating the value function - we compute a sample observation ˆ v which is then used to update the current estimate ¯ v . We may update ¯ v using two strategies:
- a) Stochastic gradient updates using a single sample realization. There are two variations here:
- i) If we are using a table-lookup representation, this involves smoothing the new estimate ¯ v ( s ) with the current approximation ¯ v ( s ).
21. ii) If we are using a parameterized approximation ¯ V ( s | θ ), our stochastic gradient algorithm would update θ .

- b) Parameter estimation using (recursive) regression methods.
- c) Use multiple realizations of ˆ v to find an updated parameter vector (typically θ ) by solving an optimization problem that produces the best parameter vector over the set of observations (ˆ v ) i .

## 7.9 But does it work?

## 7.9.1 Convergence of temporal difference learning

Temporal difference learning was first proposed in Sutton (1988), with a proof of convergence for TD(0). This proof was extended informally for general TD( λ ) in Dayan (1992). These proofs were done without establishing the relationship to the elegant theory of stochastic approximation methods. This relationship was exploited in Jaakkola et al. (1994) and Tsitsiklis (1994) (the latter treatment is also presented in Bertsekas &amp; Tsitsiklis (1996) (chapter 4)). These proofs applied to table-lookup approximations of the value function.

## 7.9.2 Convergence of Q -learning

Q -learning was first introduced in Watkins' Ph.D. dissertation (Watkins (1989)), and later introduced in a journal publication in Watkins &amp; Dayan (1992). Tsitsiklis (1994) provides a more formal proof of convergence using the theory of stochastic approximation methods.

## 7.10 Why does it work**

## 7.11 Bibliographic notes

```
Roll-out heuristics Bertsekas & Castanon (1999) TD lambda: Dayan (1992), Tsitsiklis (1994)
```

```
Elevator group control: Crites & Barto (1994) & Tsitsiklis (1991), Andreatta &
```

Stochastic shortest paths: Frank (1969), Bertsekas Romeo (1988), Frieze &amp; Grimmet (1985), Psaraftis &amp; Tsitsiklis (1990), Sigal et al. (1980),

Factored MDP's: Guestrin et al. (2003)

## Exercises

- 7.1) Compute the size of the state space (the attribute space) of our college student in section 7.2.2.
- 7.2) Section 7.2.1 introduced three information models for the shortest path problem. Let v I i be the expected cost to get from node i to the destination for Case I (equation (7.3)). Similarly, let v II i and v III i be the expected costs for cases II and III (equations (7.4) and (7.5), respectively). Show that v I i ≤ v I i ≤ v III i .
- 7.3) The network above has an origin at node 1 and a destination at node 4. Each link has two possible costs with a probability of each outcome.
- a) Write out the dynamic programming recursion to solve the shortest path problem from 1 to 4. Assuming that the driver does not see the cost on a link until he arrives at the link (that is, he will not see the cost on link (2,4) until he arrives to node 2). Solve the dynamic program and give the expected cost of getting from 1 to 4.
- b) Set up and solve the dynamic program to find the expected shortest path from 1 to 4 assuming the driver sees all the link costs before he starts the trip.
- c) Set up and solve the dynamic program to find the expected shortest path assuming the driver does not see the cost on a link until after he traverses the link.
- d) Give a set of inequalities relating the results from parts (a), (b) and (c) and provide a coherent argument to support your relationships.
- 7.4) Here we are going to again solve a variant of the asset selling problem using a postdecision state variable, but this time we are going to use asynchronous state sampling (in chapter 5 we used synchronous approximate dynamic programming). We assume we are holding a real asset and we are responding to a series of offers. Let ˆ p t be the t th offer, which is uniformly distributed between 500 and 600 (all prices are in thousands of dollars). We also assume that each offer is independent of all prior offers. You are willing to consider up to 10 offers, and your goal is to get the highest possible price. If you have not accepted the first nine offers, you must accept the 10 th offer.

<!-- image -->

a grid (below), and at each cell, it can go up/down or left/right.

ach it to find the exit as quickly as possible. Write an approx-

- a) Implement an approximate dynamic programming algorithm using asynchronous state sampling, initializing all value functions to zero. Using 100 iterations, write out your estimates of the value of being in each state immediately after each offer. Use a stepsize rule of 5 / (5+ n -1). Summarize your estimate of the value of each state after each offer.
- b) Compare your results against the estimates you obtain using synchronous sampling. Which produces better results?

## Computer exercises

the following initialization strategies:

- 7.5) The taxicab problem is a famous learning problem in the artificial intelligence literature. The cab enters a grid (below), and at each cell, it can go up/down or left/right. The challenge is to teach it to find the exit as quickly as possible. Write an approx-

i = 1 is the bottom row.

lue function using the formula (10 - i) + j. This estimate pushes ok down and to the right.

<!-- image -->

imate dynamic programming algorithm to learn the best path to the exit where at every iteration you are allowed to loop over every cell and update the value of being in that cell.

- 7.6) Repeat the taxicab exercise, but now assume you can only update the value of the cell that you are in, using the following initialization strategies:
- a) Initialize your value function with 0 everywhere.
- b) Initialize your value function where the value of being in cell ( i, j ) is 10 -i where i is the row, and i = 1 is the bottom row.
- c) Initialize your value function using the formula (10 -i ) + j . This estimate pushes the system to look down and to the right.

## Chapter 8

## Infinite horizon problems

Infinite horizon problems have traditionally been the most popular setting for analyzing approximate methods. They tend to be simpler and more elegant, which makes them easier to analyze. By contrast, our presentation used finite horizon problems to provide the initial presentation of approximate dynamic programming methods, specifically because it provided the framework for describing the sequencing of decisions and information that depend on an explicit temporal model (which is required for finite horizon problems). This presentation, however, was restricted to discrete representations of the value function. In this chapter, we take advantage of our presentation of continuous approximations to provide a combined presentation of discrete and continuous approximation methods in the context of infinite horizon problems.

Infinite horizon problems arise in any setting where the processes do not vary over time. These models are particularly useful in applications where the primary interest is in understanding the properties of the problem, rather than in operational solutions. They tend to be computationally much easier because we do not have to estimate value functions at each point in time.

Typically, adapting the algorithms that we presented in chapter 7 to infinite horizon problems is fairly straightforward. Often, all that is required is dropping the time index. This is one reason our presentation started with the time-dependent (finite horizon) version of the problem. It is much easier to convert a time-indexed version of the algorithm to steady state (where there is no time indexing) than the other way around. There are, however, some subtle issues that are introduced in the transition. For this reason, we begin our presentation by briefly summarizing some of the same algorithms that use discrete table lookup forms of the value function that were originally presented in the context of finite horizon problems.

## 8.1 Approximate dynamic programming for infinite horizon problems

Our algorithmic strategy is going to closely mimic the algorithms used for finite horizon problems. When we solved a finite horizon problem, we solved recursions of the form (using a post-decision state variable):

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

is the resource vector at time t given the resource vector at time t -1, the information that arrives during time interval t given by W t , and the decision x t made at the end of time interval t . Replacing the value function with our approximation, our decision function would look like:

<!-- formula-not-decoded -->

When we solve a steady state problem, we want to solve recursions of the form:

<!-- formula-not-decoded -->

Substituting an approximate value function, our decision function would look like:

<!-- formula-not-decoded -->

A sketch of a basic approximate dynamic programming algorithm for infinite horizon problems is given in figure 8.1. This is the architecture shared by virtually all approximate algorithms for infinite horizon problems. Algorithms primarily vary in terms of how the value function is updated in step 3 (and implicitly, the choice of the structure of the approximate value function), as well as the choice of the next state to visit (sometimes it is useful to choose states other than the one indicated by the decision x n ).

## 8.2 Algorithmic strategies for discrete value functions

The solution of infinite horizon problems using discrete value function representations is primarily an adaptation of the techniques for finite horizon problems (chapter 7). Often,

Step 1. Initialize an approximation for the value function ¯ V 0 ( S ) for all states s ∈ S . Initialize s 0 . Let n = 1.

Step 2. Choose ω n and solve:

<!-- formula-not-decoded -->

Step 3. Use the results to update the approximation ¯ V n -1 ( s n -1 ) (using any of a variety of algorithms).

- Step 4. Compute s n = S M ( s n -1 , W ( ω n ) , x n ).

Step 5. Let n := n +1. If n &lt; N , go to step 2.

Figure 8.1: A basic approximate dynamic programming algorithm for infinite horizon problems.

steady state models look the same as finite horizon models, but without the time index. In some cases, however, the transition to infinite horizon introduces some subtleties.

In this section, we assume that if we are in state s n , then we only update the value of being in that state, given by ¯ V n -1 ( s n ). Discrete representations of value functions (often referred to as 'table lookup' versions of the value function) also require discrete actions where we choose the best action by searching over all possible actions. As explained in the beginning of chapter 7, these formulations are particularly applicable for problems involving the management of a single asset (selling an asset, determining when to turn an electric power generator off or on, determining the optimal flying trajectory of a single aircraft).

## 8.3 Value iteration

The basic algorithm in figure 8.1 can be viewed as a value iteration algorithm if the update step is done as shown in figure 8.2. The only difference is that we are using a Monte Carlo sample of the value of being in a state, rather than an expectation. As a result, we lose all the fundamental properties that we were able to establish when we used expectations. For example, in theorem 4.5.2, we showed that if v ≥ M v , then v ≥ v ∗ , and if v ≤ M v , then v ≤ v ∗ . Not only do we lose this result, but the algorithm will not converge to the optimal solution even in the limit. In fact, it is quite easy to construct problems where the algorithm will converge to a very poor policy. The problem is that we are only updating the value of states that we visit, and we only visit states that are the result of actions from the approximate value function. If our estimate of the value of being in a state is very low, we may avoid decisions that would put us back in that state, which means the error does not get corrected.

Proofs of convergence of approximate value iteration for discrete state and discrete action problems generally require assuming that every state will be visited infinitely often. We can accomplish this by looping over all the states in every iteration and sampling their value. Needless to say, this only works for problems with small state spaces, and typically, we

STEP 3a: Let:

<!-- formula-not-decoded -->

STEP 3b: Update the value function:

<!-- formula-not-decoded -->

Figure 8.2: The value iteration step for approximate, infinite horizon dynamic programming.

would try to use exact methods for these problems. It is possible that we cannot use exact techniques if the 'second curse of dimensionality', the exogenous information process, makes the expectation computationally intractable, or if we are missing a probability model of the information process. Instead, authors will introduce strategies that insure that every state is visited with a strictly positive probability (which may be small). While these ideas will provide a proof of convergence, they may not be computationally effective since the rate of convergence can be extremely slow.

The secret to approximate value iteration, for both finite and infinite horizon problems, lies in our ability to use the structure of the value function so that updates to one state allow us to make broader updates to the entire function.

## 8.4 Approximate policy iteration

In section 7.6, we presented a form of a policy iteration algorithm for finite horizon problems. This algorithm featured the ability to sample a trajectory until the end of the horizon multiple times, building up an estimate of the value of a policy before updating the policy. The problem with the algorithm we presented (in figure 7.5) was that we did not estimate the value of being in every state. While this is more practical for large problems, it left us without the ability to say anything about whether the algorithm actually worked.

The limitation (from a theoretical perspective) of our algorithm for finite horizon problems is that we would have to loop over all possible states for all possible time periods. In figure 8.3 we describe a version of an approximate dynamic programming algorithm for policy iteration for an infinite horizon problem. It is unlikely that anyone would actually implement this algorithm, but it helps to illustrate the choices that can be made when designing a policy iteration algorithm in an approximate setting. The algorithm features four nested loops. The innermost loops step forward and backward in time from an initial state R n, 0 . The next outer loop repeats this process M times, estimating the value of a policy starting from a particular starting state. The third loop considers all possible initial states R n, 0 and the fourth and most outer loop iteratively updates the policy (more precisely, the value function we are using to make decisions).

The algorithm helps us illustrate different variables. First, if we let T →∞ , we are eval-

```
Step 0: Initialization: Step 0a. Initialize ¯ V π, 0 t , t ∈ T . Step 0b. Set n = 1. Step 1. Do while n ≤ N : Step 2. Loop over all possible states R n, 0 : Step 3. Do while m ≤ M : Step 3a. Choose a sample path ω m : Step 3b. Do for t = 1 , 2 , . . . , T : x n,m t = arg max x ∈X C ( R n,m t -1 , W t ( ω m ) , x ) + γ ¯ V π,n -1 ( R M ( R n,m t -1 , W t ( ω m ) , x )) R n,m t = R M ( R n,m t -1 , ω m t , x n,m t ) . Step 3c. Initialize ˆ v n,m T +1 = 0 Step 3d: Do for t = T, T -1 , . . . , 1: ˆ v n,m t -1 := C ( R n,m t , ω m t , x n,m t ) + γ ˆ v n,m t Step 3e. Update approximate value of the policy: ¯ v n,m = (1 -α m )¯ v n,m -1 + α m ˆ v n,m 0 where typically α m = 1 /m (the observations ˆ v n,m are drawn from the same distribution). Step 4. Update the value function at R n, 0 : ¯ V n ( R n, 0 ) = (1 -α n ) ¯ V n -1 ( R n, 0 ) + α n ¯ v n,M Step 5: Update the policy: ¯ V π,n ( R n, 0 ) = ¯ V n ( R n, 0 ) ∀ R n, 0 . Step 6: Return the value functions ( ¯ V π,N ).
```

Figure 8.3: A policy iteration algorithm for steady state problems

uating a true infinite horizon policy. If we simultaneously let M →∞ , then ¯ v n approaches the exact, infinite horizon value of the policy π determined by ¯ V π,n . Thus, for M = T = ∞ , we have an Monte Carlo-based version of policy iteration.

Of course, this is impractical. We can choose a finite value of T that produces values ˆ v n,m that are close to the infinite horizon results. We can also choose finite values of M , including M = 1. When we use finite values of M , this means that we are updating the policy before we have fully evaluated the policy. This variant is known in the literature as optimistic policy iteration because rather than wait until we have a true estimate of the value of the policy, we update the policy after each sample (presumably, although not necessarily, producing a better policy). Students may also think of this as a form of partial policy evaluation, not unlike the hybrid value/policy iteration described in section 4.4.3. Optimistic policy iteration ( M = 1) is one of the few variations of approximate dynamic programming which produces a provably convergent algorithm. But it does require synchronous state sampling (where we loop over all possible states). Nice, but not very practical.

## 8.5 TD learning with discrete value functions

As we learned with finite horizon problems, there is a close relationship between value iteration and a particular case of temporal difference learning. To see this, we start by rewriting equation (8.3) as:

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

is our temporal difference (see equations (7.26) and (7.27). Thus, our basic value iteration algorithm is the same as TD(0).

We can perform updates using a general TD( λ ) strategy as we did for finite horizon problems. However there are some subtle differences. With finite horizon problems, it is common to assume that we are estimating a different function ¯ V t for each time period t . As we step through time, we obtain information that can be used for a value function at a specific point in time. With stationary problems, information from each transition produces information that can be used to update the value function, which is then used in all future updates. By contrast, if we update ¯ V t for a finite horizon problem, then this update is not used until the next forward pass through the states.

In chapter 7, we found that we could write our updating logic as:

<!-- formula-not-decoded -->

We then defined the temporal differences:

<!-- formula-not-decoded -->

which allowed us to rewrite the updating equation (8.4) using:

<!-- formula-not-decoded -->

When we move to infinite horizon problems, we drop the indexing by t . Instead of stepping forward in time, we step through iterations, where at each iteration we generate a temporal

difference:

<!-- formula-not-decoded -->

To do a proper update of the value function at each state, we would normally use an infinite series of the form:

<!-- formula-not-decoded -->

where we can use any initial starting state s 0 = s . Of course, we would use the same update for each state s n that we visit, so we might write:

<!-- formula-not-decoded -->

Equations (8.5) and (8.6) both imply stepping forward in time (presumably a 'large' number of iterations) and computing temporal differences before performing an update. A more natural way to run the algorithm is to do the updates incrementally. After we compute D n , we can update the value function at each of the previous states we visited. So, at iteration n , we would execute:

<!-- formula-not-decoded -->

We can now use the temporal difference D n to update the estimate of the value function for every state we have visited up to iteration n .

Figure 8.4 outlines the basic structure of a TD( λ ) algorithm for an infinite horizon problem. Step 1 makes a single step forward. After computing the temporal-difference for Step 2, we traverse previous states we have visited in Step 3 to update their value functions. In step 3, we update all the states ( s m ) n m =1 that we have visited up to now. Thus, at iteration n , we would have simulated the partial update:

<!-- formula-not-decoded -->

This means that at any iteration n , we have updated our values using biased sample observations (as is generally the case in value iteration). We avoided this problem for finite horizon problems by extending out to the end of the horizon. We can obtain unbiased updates for infinite horizon problems by assuming that all policies eventually put the system into an absorbing state. For example, if we are modeling the process of holding or selling an asset, we might be able to guarantee that we eventually sell the asset.

Step 0. Initialize an approximation for the value function ¯ V 0 ( s t ) for all states s ∈ S . Initialize S 0 . Let n = 1.

Step 1. Choose ω n and solve:

<!-- formula-not-decoded -->

Step 2. Compute the temporal difference for this step:

<!-- formula-not-decoded -->

Step 3. Update ¯ V for m = 1 , 2 , . . . , n :

<!-- formula-not-decoded -->

Step 4. Compute s n = S M ( s n -1 , W ( ω n ) , x n ).

Step 5. Let n := n +1. If n &lt; N , go to step 1.

Figure 8.4: A TD( λ ) algorithm for infinite horizon problems.

One subtle difference between temporal difference learning for finite horizon and infinite horizon problems is that in the infinite horizon case, we may be visiting the same state two or more times on the same sample path. For the finite horizon case, the states and value functions are all indexed by the time that we visit them. Since we step forward through time, we can never visit the same state at the same point in time twice in the same sample path. By contrast, it is quite easy in a steady state horizon to revisit the same state over and over again. For example, we could trace the path of our nomadic trucker, who might go back and forth between the same pair of locations in the same sample path. As a result, we are using the value function to determine what state to visit, but at the same time we are updating the value of being in these states.

## 8.6 Q-learning

Q-learning in steady state is exactly analogous to Q-learning for time-dependent problems (section 7.3.3). A generic Q-learning algorithm is given in figure 8.5.

## 8.7 Why does it work?**

van Roy and Tsitsiklis Tsitsiklis &amp; Van Roy (1997)

Step 0.

```
Step 0a Initialize an approximation for the value function ¯ Q 0 ( a, d ) for all states a and decisions d . Step 0b Initialize ¯ V 0 = max d ∈D ¯ Q 0 ( a, d ). Step 0c Let n = 1. Step 1. Determine d n = arg max d ∈D ¯ Q n -1 ( a n -1 , d ) Step 2. Choose ω n and find: ˆ Q n ( a n -1 , d n ) = C ( a n -1 , d n , ω n ) + γ ¯ V n -1 ( a M ( a n -1 , d n , ω n )) Step 3. Update ¯ Q n -1 and ¯ V n -1 using: ¯ Q n ( a n -1 , d n ) = (1 -α n ) ¯ Q n -1 ( a n , d n ) + α n ˆ Q n ( a n , d n ) ¯ V n ( a n -1 ) = (1 -α n ) ¯ V n -1 ( a n ) + α n ˆ Q n ( a n , d n ) Step 4. Sample ω ′ ,n . Step 5. Compute a n = a M ( a n -1 , d n , ω ′ ,n ). Step 6. Let n := n +1. If n < N , go to step 1.
```

Figure 8.5: A Q -learning algorithm for a steady-state problem

## 8.8 Bibliographic notes

Tsitsiklis &amp; Van Roy (1997), Bertsekas &amp; Tsitsiklis (1996), A.Nedi¸ c &amp; D.P.Bertsekas (2003), Bertsekas et al. (2004), Boyan (2002)

## Exercises

- 8.1) You have two two-state Markov chains, with transition matrices:

| Action 1   |   Action 1 |   Action 1 |
|------------|------------|------------|
| From-to    |        1   |        2   |
| 1          |        0.2 |        0.8 |
| 2          |        0.7 |        0.3 |

The reward for each state-action pair is given by:

Action 2

From-to

1

1

2

0.5

0.6

2

0.5

0.4

|       | Action   |
|-------|----------|
| State | 1 2      |
| 1     | 5 10     |
| 2     | 10 15    |

- a) Solve the problem exactly using value iteration. Plot the value of being in each state as a function of the number of iterations, and give the optimal policy (the best action for each state) at optimality.
- b) Use the approximate dynamic programming algorithm in figure 8.1 to estimate the value of being in each state and estimate the optimal policy. Use 1000 Monte Carlo samples, and plot the estimate of the value function after each iteration.

## Chapter 9

## Value function approximations

The algorithms in chapter 7 assumed a standard 'table-lookup' representation for a value function. That is to say, if we visited state s we would then estimate the value of being in state s . While we have eliminated the need to loop over all states (as was required in chapter 4), we now have the problem of estimating the value of being in a state. However, we need more than just an estimate of the states we have visited. We need an estimate of the value of being in each state that we might visit, since this is the mechanism by which we choose from a set of decisions. While this can be much smaller than the entire state space, it is still much harder than looking at only the states that we actually visit.

In this chapter we introduce some general methods for approximating the value of being in a state using fewer parameters than the number of states. Our presentation is done using the vocabulary of managing a single asset, which means that our state space is the attribute space of the asset. We defer until chapter 11 the special challenge of approximating value functions when we are managing multiple assets. We make minimal assumptions about the attributes or the structure of the value function over the attribute space.

The starting point of our presentation is that we are trying to estimate the value of an asset with attribute vector a (we can also think of this as estimating the value of being in state s ). We need to be able to make a decision by solving a problem of the form:

<!-- formula-not-decoded -->

where, as before, a M ( a, d ) is the attribute of the asset after being acted on by decision d . We assume that the attribute vector consists of I dimensions, where I is not too large (large problems may have a few dozen or a hundred attributes, but we do not anticipate problems with thousands of attributes). In the classical language of dynamic programming, our challenge is to estimate the value v ( a ) of being in state a . However, we assume that the attribute space |A| is too big to enumerate, which means we are going to have a difficult time estimating the value of being in any particular state.

This chapter introduces two types of general purpose techniques designed for working with complex attribute spaces. The first depends on the use of classical aggregation methods,

and assumes that we have access to one or more functions which can aggregate the attribute space into smaller sets. The second is general regression methods where we assume that analytical functions, with a relatively small number of parameters, can be formed to capture the important properties of the attribute vector.

## 9.1 Simple aggregation

For many years, researchers addressed the 'curse of dimensionality' by using aggregation to reduce the size of the state space. The idea was to aggregate the original problem, solve it optimally, and then disaggregate it back to obtain an approximate solution to the original problem. Of course, aggregation introduced errors, and much of the research focused on bounding these errors.

This approach assumes that we are converting the problem to a smaller problem that is then solved entirely using the aggregated state space. In approximate dynamic programming, we do not simplify how we represent the asset being managed; instead, we approximate the value function itself by estimating the value of an asset over an aggregated state space. For example, in our nomadic trucker problem it is necessary to capture location, domicile, fleet type, equipment type, number of hours he has driven that day, how many hours he has driven on each of the past seven days, and the number of days he has been driving since he was last at home. All of this information is needed to simulate the driver forward in time. But we might estimate the value function using only the location of the driver, his domicile and fleet type.

The presence of an aggregation function is highly context specific, and typically reflects an understanding of the structure of a problem. An attribute may be continuous (the price of a stock, the amount of fuel carried by a jet) and the aggregation may simply divide the continuous interval into a set of discrete intervals. Alternatively, the attribute may be categorical, with an aggregation function simply producing coarser categories. For example, the value of a stock such as DuPont Chemical may be aggregated into one for the chemical industry, or part of the S&amp;P 500, or a domestic stock, or a growth stock.

It is often the case that we have to decide on the right level of aggregation. For this reason, we define a family of aggregation functions

<!-- formula-not-decoded -->

A ( g ) represents the g th level of aggregation of the attribute space A . Let a ( g ) = G g ( a ), the g th level aggregation of the attribute vector a .

G = The set of indices corresponding to the levels of aggregation.

In this section, we consider only two levels of aggregation: g = 0 represents the disaggregate level (which means that A (0) = A ), while g = 1 represents the aggregated level.

To begin our study of aggregation, we first need to characterize our sampling process. For this discussion, we can assume we have two exogenous processes: at iteration n , the first process chooses an attribute to sample (which we denote by ˆ a n ), and the second produces an observation of the value of being in state ˆ a n ∈ A , which we denote by ˆ v n (or ˆ v n a ). Let:

ν ( g ) a = The true value of being in state a at aggregation level g .

Here, we view a as a disaggregated attribute vector. ν (0) a is the true (expected) value of being in this state, while ν (1) a is the expected value of being in aggregated state G ( a ).

At iteration n , we sample a (disaggregated) state ˆ a n with noise, producing:

<!-- formula-not-decoded -->

We represent the sets of outcomes of the state sampling process ˆ a n and the measurement process ε n using:

Ω a = The set of outcomes of observations of attribute vectors.

Ω ε = The set of outcomes of observations of the errors in the values.

Ω = The overall set of outcomes.

= Ω a × Ω ε .

<!-- formula-not-decoded -->

= An element of the outcome space.

We next define

<!-- formula-not-decoded -->

We can now rewrite ˆ v n in terms of the bias and the noise:

<!-- formula-not-decoded -->

We are going to present two ways of estimating the bias and variance. In the first, we are going to assume we use simple averages over observations. This presentation brings out the role of sample sizes, and the difference in the sample sizes at different levels of aggregation. Afterwards, we present recursive formulas for estimating these quantities.

To bring out the role of sample sizes we need to define:

N ( g,n ) a = The set of indices that correspond to observations of the attribute vector a at the g th level of aggregation after n iterations. = { n | G g (ˆ a n ) = G g ( a ) } N ( g,n ) a = ∣ ∣ N ( g,n ) a ∣ ∣ ¯ v ( g,n ) a = The estimate of the value associated with the attribute vector a at the g th level of aggregation, given the sample N after n observations. ˆ ε ( g,n ) a = The error in the estimate ¯ v ( g,n ) a from the observed value associated with attribute vector a = ˆ a n . = ˆ v n -¯ v ( g,n -1) a

¯ v ( g,n ) a can be computed using a simple average:

<!-- formula-not-decoded -->

The differerenc between the disaggregate value ν a and the disaggregate estimate ¯ ν (0 ,n ) is simply noise and is captured by the error. The bias is estimated using We can get an estimate of the bias and average error using:

<!-- formula-not-decoded -->

The average error is estimated using

<!-- formula-not-decoded -->

).

¯ ε ( g,n ) a estimates the variance of the noise around the estimate of the value of being in state a at the g th level of aggregation. This error can be broken down between observations of

Figure 9.1: Illustration of a disaggregate function, an aggregated approximation and a set of samples. For a particular attribute a , we show the estimate and the bias.

<!-- image -->

attribute a , and observations of other attributes a ′ that aggregate up to the same aggregated attribute:

<!-- formula-not-decoded -->

The relationships are illustrated in figure 9.1 which shows a simple function defined over a single, continuous attribute (for example, the price of an asset). If we selected a particular attribute a , we find we have only two observations for that attribute, versus seven for that section of the function. If we use an aggregate approximation, we would produce a single number over that range of the function, creating a bias between the true function and the aggregated estimate. As the illustration shows, the size of the bias depends on the shape of the function in that region.

Equation (9.3) allows us to see the effects of sample size, and the samples themselves, at the two levels of aggregation. However, this is not the equation we would actually use to estimate the values. Instead, we would use the recursive form:

<!-- formula-not-decoded -->

When computing the stepsize, it is important to use the stepsize α N ( g,n ) a ; we write α n only for notational simplicity. In addition to estimating the value function, it is also important to estimate the variance. If we are doing true averaging, we can get a more precise estimate of the variance by using the recursive form of the small sample formula for the variance:

<!-- formula-not-decoded -->

(ˆ σ 2 ) g,n is an estimate of the variance of ε a , which may be especially useful if we believe that the variance of ε a is independent of a . Typically we are more interested in the variance of our estimate ¯ v ( g,n ) a , which would be computed using:

<!-- formula-not-decoded -->

If we do not believe that the variance is the same over the attribute space, we have to compute variances for each (perhaps aggregated) attribute. Let a = ˆ a n . Our estimate of the variance of ˆ v a after n iterations is given by:

<!-- formula-not-decoded -->

The variance of ¯ v ( g,n ) a is then given by:

<!-- formula-not-decoded -->

An overall measure of the quality of an approximation can now be computed using:

<!-- formula-not-decoded -->

using the fact that ∑ a ∈A N ( g,n ) a = n .

## 9.2 The case of biased estimates

If we use the variance measures given by equations (9.4) or (9.5) around either the aggregate or disaggregate levels, we are not getting a true measure of the distance of observations from disaggregate attribute a to the aggregate function ¯ v ( g ) a . The problem is illustrated in figure 9.2. Figure 9.2a shows errors from measuring disaggregate attribute a around the true function for a . Figure 9.2b shows the errors when we include all the observations for

the aggregated attribute G ( a ). To determine the proper weight on the disaggregate level for attribute a , we need an estimate of the distance from the observations for attribute a at the disaggregate level, and the aggregate function for attribute a . This distance requires measuring the distance between observations of a at the disaggregate level, plus the bias between the disaggregate and aggregate functions at a , as shown in figure 9.2c.

To fix this, we have to measure the errors around the disaggregate function (more precisely, our estimate of the disaggregate function) in addition to estimating the bias. The true variances is given by:

<!-- formula-not-decoded -->

If we let

<!-- formula-not-decoded -->

then it is possible to show that:

<!-- formula-not-decoded -->

The variance ( σ 2 a ) ( g ) is estimated by (¯ σ 2 a ) ( g,n ) in equation (9.5). The bias is estimated by ¯ µ ( g,n ) in equation (9.4). Finally, the weights are given by:

<!-- formula-not-decoded -->

The effect of capturing the correlations in the estimates at different levels of aggregation is best illustrated using a simple scalar function such as that shown in figure 9.3a. At the disaggregate level, the function is defined for 10 discrete values. This range is then divided into three larger intervals, and an aggregated function is created by estimating the function over each of these three larger ranges.

Figure 9.3b shows the weight given to the disaggregate level at each of the 10 points. The weights are computed both with and without the independence assumption. We note that under both assumptions, the weights are highest when the function has the greatest slope, which means the highest errors when we try to aggregate. This is consistent with our analysis of the two-level problem. When we compute the optimal weights (which captures the correlation), the weight on the disaggregate level for the portion of the curve that is flat is zero, as we would expect. Note that when we assume independence, the weight on the disaggregate level (when the slope is zero) is no longer zero. Clearly a weight of zero is best because it means that we are aggregating all the points over the interval into a single estimate, which is going to be better than trying to produce three individual estimates.

<!-- image -->

9.2a: Errors measured around disaggregate function.

9.2b: Errors measured around aggregate function.

<!-- image -->

9.2c: Errors masured around disaggregate function plus the bias.

<!-- image -->

Figure 9.2: Illustration of different forms of variance measure. (a) measures errors around the disaggregate function, (b) measures errors around the aggregate function, and (c) shows errors around the disaggregate function plus the bias.

Figure 9.3: The weight given to the disaggregate level for a two-level problem at each of 10 points, with and without the independence assumption (from George et al. (2005)). The graph illustrates that the highest weights are put on the disaggregate level when there is the greatest slope in the function.

<!-- image -->

One would expect that using the optimal weights, which captures the correlations between estimates at different levels of aggregation, would also produce better estimates of the function itself. We compared the errors between the estimated function and the actual function using both methods for computing weights, using three levels of noise around the function. The results are shown in figure 9.4, which indicates that there is virtually no difference in the accuracy of the estimates produced by the two methods. This observation has held up under a number of experiments.

Performance measure (e'

0.9 -

0.8 -

0.7 -

0.6

0.5 -

T

0.4

0.3 -

0.2 -

0.1

Low noise assumption

40

50

60

Number of observations

Moderate noise with independence assumption

High noise

Figure 9.4: The effect of ignoring the correlation between estimates at different levels of aggregation

<!-- image -->

A little thought tells us why the independence assumption is not that critical. Notice from figure 9.3b that the difference in the weights is smallest when there is the biggest difference between the aggregate and disaggregate functions. The difference is greatest when the difference is the least. This turns out to be a general result. This behavior means that assuming independence introduces negligible errors when it is most important to put the right weight on the disaggregate level. At the same time, the difference is greatest when it does not really matter. The conclusion appears to be that we can safely assume independence, which allows us to use the simpler expression for the weights, with minimal loss in accuracy.

## 9.3 Multiple levels of aggregation

It is typically the case that we can choose between multiple levels of aggregation. It is natural, then, to want to know which level of aggregation produces the best results. As the level of aggregation increases, we obtain higher statistical validity, but encounter more structural noise in addition to sampling error. Furthermore, the level of aggregation that produces the lowest error will change as we gather more observations. In addition, since we are often sampling states in a nonuniform manner, the best level of aggregation will depend on the state a .

In section 9.1, we focused on two levels of aggregation. The disaggregate state space A and the aggregated state space A ( g ) . In this section, we are going to assume that we have a

Table 9.1: Examples of aggregations on the attribute space for the nomadic trucker problem. '-' indicates that the particular attribute is ignored

|   Aggregation level | Location   | Fleet type   | Domicile   | Size of state space       |
|---------------------|------------|--------------|------------|---------------------------|
|                   0 | Sub-region | Fleet        | Region     | 400 × 5 × 100 = 200 , 000 |
|                   1 | Region     | Fleet        | Region     | 100 × 5 × 10 = 50 , 000   |
|                   2 | Region     | Fleet        | Zone       | 100 × 5 × 10 = 5 , 000    |
|                   3 | Region     | Fleet        | -          | 100 × 5 × 1 = 500         |
|                   4 | Zone       | -            | -          | 10 × 1 × 1 = 10           |

family of aggregations which we model using:

G = Set of different levels of aggregation.

<!-- formula-not-decoded -->

Now, g becomes a numerical index rather than simply a designation for 'aggregate.' We adopt the convention that g = 0 corresponds to the most disaggregate level (no aggregation), with increasing g corresponding to more aggregate levels.

There are many applications where aggregation is naturally hierarchical. For example, in our nomadic trucker problem we might want to estimate the value of a truck based on three attributes: location, home domicile and fleet type. The first two represent geographical locations, which can be represented (for this example) at three levels of aggregation: 400 sub-regions, 100 regions and 10 zones. Table 9.1 illustrates five levels of aggregation that might be used. In this example, each higher level can be represented as an aggregation of the previous level.

Hierarchical aggregation is often the simplest to work with, but in most cases there is no reason to assume that the structure is hierarchical. In fact, we may even use overlapping aggregations (sometimes known as 'soft' aggregation), where the same attribute a aggregates into multiple elements in A g . For example, assume that a represents an ( x, y ) coordinate in a continuous space. Assume that our aggregated space consists of a smaller set of coordinates ( x i , y i ) i ∈I . Further assume that we have a distance metric from any point ( x, y ) to every aggregated point ( x i , y i ) , i ∈ I . Let ρ (( x, y ) , ( x i , y i )) be our distance measure. We might use an observation at the point ( x, y ) to update estimates at each ( x i , y i ) with a weight that declines with ρ (( x, y ) , ( x i , y i )).

## 9.3.1 Combining multiple statistics

When we have access to multiple levels of aggregation (which is typical for more complex problems) we have several options:

- a) Pick a single level of aggregation that appears to work best.

- b) Pick a single level of aggregation that produces the lowest error at iteration n .
- c) Pick the best level of aggregation for each attribute vector a by choosing the level of aggregation with the lowest error.
- d) Use a weighted combination of the estimated produced by each level of aggregation for a particular attribute vector a .

In this section we focus on the last option, since the first three are straightforward to apply once we compute the error measures required for option d.

Using a weighted combination means that our estimate of being in state a is given by:

<!-- formula-not-decoded -->

Here, the attribute a is interpreted as the disaggregate state, whereas ¯ v ( g ) a is the value of being in state G g ( a ), computed using the methods of section 9.1.

We can view the estimates (¯ v ( g ) ) g ∈G as different ways of estimating the same quantity. There is an extensive statistics literature on this problem. For example, it is well known that the weights that minimize the variance of ¯ v n a are given by:

<!-- formula-not-decoded -->

Since the weights should sum to one, we obtain:

<!-- formula-not-decoded -->

Using equation (9.8) allows us to state that ¯ v n a has the lowest possible variance, under two assumptions: a) that the variances (¯ σ 2 a ) ( g,n ) are exact rather than approximate and b) that the statistics ¯ v ( g ) a are independent. Of course, the variances are never known, and as a result we always have to use our best estimate of the variances. The standard game in statistics is to claim that we have the best estimate assuming that the exact variances are given by our current estimates. We address the issue of correlated statistics in section 9.3.2.

The biggest practical challenge in using variances to weight different estimates is that our estimates of the variances themselves can be extremely poor unless we have at least 10 or so observations. In many applications, obtaining even five observations of most states is unachievable. Furthermore, even when this is the case we still face the problem of estimating variances in the early iterations. A simple method that has been found to work quite well in the early iterations is to use weights that are equal to the number of observations. The weights are stable, and are biased toward the most aggregate level which can work best when there is a lot of noise and very few observations. In the limit, this can work quite poorly because the highest weight is being put on the most aggregate level, so it is necessary to switch to another weighting formula when there are enough observations.

## 9.3.2 The problem of correlated statistics

It is possible to derive the optimal weights for the case where the statistics ¯ v ( g ) a are not independent. In general, if we are using a hierarchical strategy and have g ′ &gt; g (which means that aggregation g ′ is more aggregate than g ), then the statistic ¯ v ( g ′ ,n ) a is computed using observations ˆ v n a that are also used to compute ¯ v ( g,n ) a . We can see this relationship clearly by writing ¯ v ( g ′ ) a using:

<!-- formula-not-decoded -->

This relationship shows us that we can write the error term at the higher level of aggregation g ′ as a sum of a term involving the errors at the lower level of aggregation g (for the same state a ) and a term involving errors from other states a ′′ where G g ′ ( a ′′ ) = G g ′ ( a ), given by

<!-- formula-not-decoded -->

We can overcome this problem by rederiving the expression for the optimal weights. For a given (disaggregate) attribute a , the problem of finding the optimal weights ( w ( g ) a ) g ∈G is stated as follows:

<!-- formula-not-decoded -->

subject to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let:

¯ δ ( g ) a = The error in the estimate, ¯ v ( g ) a , from the true value associated with attribute vector a . = ¯ v ( g ) a -ν a

The optimal solution is given in the following theorem:

Theorem 1 For a given attribute vector, a , the optimal weights, w ( g ) a , g ∈ G , where the individual estimates are correlated by way of a tree structure, are given by solving the following system of linear equations in ( w, λ ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: The proof is not too difficult and it illustrates how we obtain the optimal weights. We start by formulating the Lagrangian for the problem formulated in (9.10)-(9.12), which gives us:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first order optimality conditions are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To simplify equation (9.16), we note that,

<!-- formula-not-decoded -->

Combining equations (9.16) and (9.18) gives us equation (9.13) which completes the proof. /square

Finding the optimal weights that handles the correlations between the statistics at different levels of aggregation requires finding E [ ¯ δ ( g ) a ¯ δ ( g ′ ) a ] . We are going to compute this expectation by conditioning on the set of attributes ˆ a n that are sampled. This means that our expectation is defined over the outcome space Ω ε . The expectation is computed using:

Proposition 2 The coefficients of the weights in equation (9.14) can be expressed as follows:

<!-- formula-not-decoded -->

The proof is given in section 9.6.2.

Now consider what happens when we make the assumption that the measurement error ε n is independent of the attribute being sampled, ˆ a n . We do this by assuming that the variance of the measurement error is a constant given by σ ε 2 . This gives us the following result

Corollary 1 For the special case where the statistical noise in the measurement of the values is independent of the attribute vector sampled, equation (9.19) reduces to,

<!-- formula-not-decoded -->

The proof is given in section 9.6.1.

For the case where g = 0 (the most disaggregate level), we assume that µ (0) a = 0 which gives us

<!-- formula-not-decoded -->

This allows us to further simplify (9.20) to obtain:

<!-- formula-not-decoded -->

## 9.3.3 A special case: two levels of aggregation

It is useful to consider the case of two levels of aggregation, since this allows us to compute the weights analytically.

For the case of two levels of aggregation, the system of linear equations given by (9.10)(9.12) reduces to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Solving for the weight on the disaggregate level produces

<!-- formula-not-decoded -->

By contrast, if we assume the estimates at the different levels of aggregation are independent we can use E [ ¯ δ (0) a ¯ δ (1) a ] = 0 which gives us:

<!-- formula-not-decoded -->

To see the relationship between the two formulas, we use the following results from section 9.3.2:

<!-- formula-not-decoded -->

Substituting (9.27) into (9.25) gives:

<!-- formula-not-decoded -->

From (9.28) we see that as the bias goes to zero, the weight on the disaggregate level goes to zero. Similarly, as the bias grows, the weight on the disaggregate level increases.

## 9.3.4 Experimenting with hierarchical aggregation

Using a value function approximation based on a weighted, multiple aggregation formula can produce a significant increase in the quality of the solution. Figure 9.5 shows the results of estimating the value of our nomadic trucker with three attributes. We are using a pure exploration strategy so that we are focused purely on the problem of estimating the value of the attribute. Also shown are the estimates when we use a single level of aggregation (we tried each of the four levels of aggregation in isolation). The weighted estimate works the best at almost all iterations (the level three aggregation works slightly better in the very beginning). In the limit, only the pure disaggregate estimate matches the weighted combination in the limit. As the number of observations increases, all of the more aggregate estimates level off with a higher error (as we would expect). These results will, of course, vary between applications. It is not uncommon for a higher level of aggregation to work best at first. However, it appears that using a weighted combination across the levels of aggregation can be quite robust.

## 9.4 General regression models

There are many problems where we can exploit structure in the state variable, allowing us to propose functions characterized by a small number of parameters which have to be estimated statistically. Section 9.1 represented one version where we had a parameter for each (possibly aggregated) state. The only structure we assumed was implicit in the ability to specify a series of one or more aggregation functions.

In this section, we are going to allow ourselves to create a much wider range of approximations by viewing the value function as nothing more than a complex quantity that we want to predict using standard statistical techniques. Using conventional statistical notation, imagine that we have a series of explanatory (independent) variables ( X i ) i ∈I , and a single dependent variable Y that we would like to predict given knowledge of each X i . Further assume that we have n observations of each, and let x in be the n th observation of x i , and let

Multi-attribute nomadic trucker - myopic runs

Figure 9.5: Using a mixture of estimates based on different levels of performance can give more accurate estimates than using a single, disaggregate estimate.

<!-- image -->

y n be the corresponding dependent variable. For simplicity, let's now assume that we think that a reasonable model can be formed to explain Y using:

<!-- formula-not-decoded -->

This is the classic linear-in-the-parameters model that is often the subject of introductory regression courses. Since we have I +1 parameters, as long as we have at least n ≥ I +1 observations (and some other conditions) we can find a vector θ that minimizes the deviations between the predicted values of Y and the observed values. This is the science of regression. The art of regression is determining an appropriate set of explanatory variables.

This is exactly the approach we are going to take to approximate a value function. Our observed dependent variables are the updates to the value function that we have represented as ˆ v n in the past. For each observed ˆ v n is a corresponding state, which in this chapter we have represented as our attribute vector ˆ a n . Now assume that we have, through knowledge and insight, decided that we can capture what is important through a series of functions which we are going to represent as ( φ b ( a )) b ∈B . These functions are often referred to as features , since they are expected to capture the important aspects of a state variable. The number of these functions, given by the size of the set B , is generally constructed so that this is not too large (and certainly nowhere near the size of the state space A ). For historical reasons, these functions are known in the approximate dynamic programming literature as basis functions (hence the choice of notation for indexing the functions).

## 9.4.1 Pricing an American option

Consider the problem of determining the value of an American put option which gives us the right to sell at $1.20 at any of four time periods. We assume a discount factor of 0.95, representing a five percent rate of return (compounded at each time period rather than continuously). If we wait until time period 4, we must exercise the option, receiving zero if the price is over $1.20. At intermediate periods, however, we may choose to hold the option even if the price is below $1.20 (of course, exercising it if the price is above $1.20 does not make sense). Our problem is to determine whether to hold or exercise the option at the intermediate points.

From history, we have found 10 samples of price trajectories which are shown in table 9.2. If we wait until time period 4, our payoff is shown in table 9.3, which is zero if the price

Table 9.2: Ten sample realizations of prices over four time periods.

| Stock prices   | Stock prices   | Stock prices   | Stock prices   | Stock prices   |
|----------------|----------------|----------------|----------------|----------------|
|                | Time period    | Time period    | Time period    | Time period    |
| Outcome        | 1              | 2              | 3              | 4              |
| 1              | 1.21           | 1.08           | 1.17           | 1.15           |
| 2              | 1.09           | 1.12           | 1.17           | 1.13           |
| 3              | 1.15           | 1.08           | 1.22           | 1.35           |
| 4              | 1.17           | 1.12           | 1.18           | 1.15           |
| 5              | 1.08           | 1.15           | 1.10           | 1.27           |
| 6              | 1.12           | 1.22           | 1.23           | 1.17           |
| 7              | 1.16           | 1.14           | 1.13           | 1.19           |
| 8              | 1.22           | 1.18           | 1.21           | 1.28           |
| 9              | 1.08           | 1.11           | 1.09           | 1.10           |
| 10             | 1.15           | 1.14           | 1.18           | 1.22           |

is above 1.20, and 1 . 20 -p 4 for prices below $1.20.

Table 9.3: The payout at time 4 if we are still holding the option.

| Option   | value       | at          | t =         | 4           |
|----------|-------------|-------------|-------------|-------------|
|          | Time period | Time period | Time period | Time period |
| Outcome  | 1           | 2           | 3           | 4           |
| 1        | -           | -           | -           | 0.05        |
| 2        | -           | -           | -           | 0.07        |
| 3        | -           | -           | -           | 0.00        |
| 4        | -           | -           | -           | 0.05        |
| 5        | -           | -           | -           | 0.00        |
| 6        | -           | -           | -           | 0.03        |
| 7        | -           | -           | -           | 0.01        |
| 8        | -           | -           | -           | 0.00        |
| 9        | -           | -           | -           | 0.10        |
| 10       | -           | -           | -           | 0.00        |

At time t = 3, we have access to the price history ( p 1 , p 2 , p 3 ). Since we may not be able

to assume that the prices are independent or even Markovian (where p 3 depends only on p 2 ), the entire price history represents our state variable. We wish to predict the value of holding the option at time t = 4. Let V 4 ( a 4 ) be the value of the option if we are holding it at time 4, given the state (which includes the price p 4 ) at time 4. Now let the conditional expectation at time 3 be:

<!-- formula-not-decoded -->

Our goal is to approximate ¯ V 3 ( a 3 ) using information we know at time 3. For our basis function, we propose a linear regression of the form:

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

Keep in mind that it is important that our explanatory variables X i must be a function of information we have at time t = 3, whereas we are trying to predict what will happen at time t = 4 (the payoff). We would then set up the data matrix given in table 9.4.

Table 9.4: The data table for our regression at time 3.

| Regression data   | Regression data   | Regression data   | Regression data   | Regression data    |
|-------------------|-------------------|-------------------|-------------------|--------------------|
|                   | Independent       | Independent       | variables         | Dependent variable |
| Outcome           | X 1               | X 2               | X 3               | Y                  |
| 1                 | 1.08              | 1.17              | 1.3689            | 0.05               |
| 2                 | 1.12              | 1.17              | 1.3689            | 0.07               |
| 3                 | 1.08              | 1.22              | 1.4884            | 0.00               |
| 4                 | 1.12              | 1.18              | 1.3924            | 0.05               |
| 5                 | 1.15              | 1.10              | 1.2100            | 0.00               |
| 6                 | 1.22              | 1.23              | 1.5129            | 0.03               |
| 7                 | 1.44              | 1.13              | 1.2769            | 0.01               |
| 8                 | 1.18              | 1.21              | 1.4641            | 0.00               |
| 9                 | 1.11              | 1.09              | 1.1881            | 0.10               |
| 10                | 1.14              | 1.18              | 1.3924            | 0.00               |

We may now run a regression on this data to determine the parameters ( θ i ) 4 i =0 . It makes sense to consider only the paths which produce a positive value in the fourth time period. The linear regression is only an approximation, and it is best to fit the approximation in the region of prices which are the most interesting (we could use the same reasoning to

include some 'near misses'). For our illustration, however, we use all 10 observations, which produces the equation:

<!-- formula-not-decoded -->

¯ V 3 is (an approximation of) the expected value of the price we would receive if we hold the option until time period 4. We can now use this approximation to help us decide what to do at time t = 3. Table 9.5 compares the value of exercising the option at time 3 against holding the option until time 4, computed as γ ¯ V 3 ( a 3 ). Taking the larger of the two payouts, we find, for example, that we would hold the option given samples 1-4, but would sell given samples 5 and 6.

Table 9.5: The payout if we exercise at time 3, and the expected value of holding based on our approximation.

| Rewards   | Rewards   | Rewards   | Rewards   | Rewards   |
|-----------|-----------|-----------|-----------|-----------|
|           | Decision  | Decision  | Decision  | Decision  |
| Outcome   | Exercise  | Hold      | Hold      | Hold      |
| 1         | 0.03      | 0.04155   | × . 95 =  | 0.03947   |
| 2         | 0.03      | 0.03662   | × . 95 =  | 0.03479   |
| 3         | 0.00      | 0.02397   | × . 95 =  | 0.02372   |
| 4         | 0.02      | 0.03346   | × . 95 =  | 0.03178   |
| 5         | 0.10      | 0.05285   | × . 95 =  | 0.05021   |
| 6         | 0.00      | 0.00414   | × . 95 =  | 0.00394   |
| 7         | 0.07      | 0.00899   | × . 95 =  | 0.00854   |
| 8         | 0.00      | 0.01610   | × . 95 =  | 0.01530   |
| 9         | 0.11      | 0.06032   | × . 95 =  | 0.05731   |
| 10        | 0.02      | 0.03099   | × . 95 =  | 0.02944   |

We can repeat the exercise to estimate ¯ V 2 ( a t ). This time, our independent variables ' Y ' can be calculated two different ways. The simplest is to take the larger of the two columns from table 9.5. So, for sample path 1, we would have Y 1 = max { . 03 , . 04155 } = . 04155. This means that our observed value is actually based on our approximate value function ¯ V 3 ( a 3 ). This represents an implementation of our single-pass algorithm described in figure 7.1.

An alternative way of computing the observed value of holding the option in time 3 is to use the approximate value function to determine the decision, but then use the actual price we receive when we eventually exercise the option. Using this method, we receive 0.05 for the first sample path because we decide to hold the asset at time 3 (based on our approximate value function) after which the price of the option turns out to be worth 0.05. Discounted, this is worth 0.0475. For sample path 2, the option proves to be worth 0.07 which discounts back to 0.0665 (we decided to hold at time 3, and the option was worth 0.07 at time 4). For sample path 5 the option is worth 0.10 because we decided to exercise at time 3. This is exactly the double pass algorithm given in figure 7.2.

Regardless of which way we compute the value of the problem at time 3, the remainder of the procedure is the same. We have to construct the independent variables ' Y ' and regress

|   1 |   2 |   3 |
|-----|-----|-----|
|   4 |   5 |   6 |
|   7 |   8 |   9 |

9.6a

<!-- image -->

9.6b

Figure 9.6: Some tic-tac-toe boards. 9.6a) gives our indexing scheme, and 9.6b) is a sample board.

them against our observations of the value of the option at time 3 using the price history ( p 1 , p 2 ). Our only change in methodology would occur at time 1 where we would have to use a different model (because we do not have a price at time 0).

## 9.4.2 Playing 'lose tic-tac-toe'

The game of 'lose tic-tac-toe' is the same as the familiar game of tic-tac-toe, with the exception that now you are trying to make the other person get three in a row. This nice twist on the popular children's game provides the setting for our next use of regression methods in approximate dynamic programming.

Unlike our exercise in pricing options, representing a tic-tac-toe board requires capturing a discrete state. Assume the cells in the board are numbered left to right, top to bottom as shown in figure 9.6a. Now consider the board in figure 9.6b. We can represent the state of the board after the t th play using:

<!-- formula-not-decoded -->

We see that this simple problem has up to 3 9 = 19 , 683 states. While many of these states will never be visited, the number of possibilities is still quite large, and seems to overstate the complexity of the game (the state space is the same if we play the original version of tic-tac-toe).

We quickly realize that what is important about a game board is not the status of every cell as we have represented it. For example, rotating the board does not change a thing, but

it does represent a different state. Also, we tend to focus on strategies (early in the game when it is more interesting) such as winning the center of the board or a corner. We might start defining variables (basis functions) such as:

φ 1 ( a t ) = 1 if there is an 'X' in the center of the board, 0 otherwise.

φ 2 ( a t ) = The number of corner cells with an 'X'.

φ 3 ( a t ) = The number of instances of adjacent cells with an 'X' (horizontally, vertically or diagonally).

There are, of course, numerous such functions we can devise, but it is unlikely that we could come up with more than a few dozen (if that) which appeared to be useful. It is important to realize that we do not need a value function to tell us to make obvious moves such as blocking your opponent after he'she gets two in a row.

Once we form our basis functions, our value function approximation is given by:

<!-- formula-not-decoded -->

We note that we have indexed the parameters by time t (the number of plays) since this is likely to play a role in determining the value of the feature being measured by a basis function. We estimate the parameters θ by playing the game (and following some policy) after which we see if we won or lost. We let Y n = 1 if we won the n th game, 0 otherwise. This also means that the value function is trying to approximate the probability of winning if we are in a particular state.

We may play the game by using our value functions to help determine a policy. Another strategy, however, is simply to allow two people (ideally, experts) to play the game and use this to collect observations of states and game outcomes. This is an example of supervisory learning. If we lack a 'supervisor' then we have to depend on simple strategies combined with the use of slowly learned value function approximations. In this case, we also have to recognize that in the early iterations, we are not going to have enough information to reliably estimate the coefficients for a large number of basis functions.

## 9.5 Recursive methods for regression models

Estimating regression models to estimate a value function involves all the same tools and statistical issues that students would encounter in any course on regression. The only difference in dynamic programming is that our data is usually generated internal to the algorithm, which means that we have the opportunity to update our regression model after every iteration. This is both an opportunity and a challenge. Traditional methods for estimating the parameters of a regression model either involve solving a system of linear equations or solving a nonlinear programming problem to find the best parameters. Both methods are

generally too slow in the context of dynamic programming. The remainder of this section describes some simple updating methods that have been used in the context of approximate dynamic programming.

## 9.5.1 Parameter estimation using a stochastic gradient algorithm

In our original representation, we effectively had a basis function for each state a t and the parameters were the value of being in each state, given by ¯ v t ( R t ). Our updating step was given by:

<!-- formula-not-decoded -->

This update is a step in the algorithm required to solve:

<!-- formula-not-decoded -->

where ˆ v is a sample estimate of V ( a ). When we parameterize the value function, we create a function that we can represent using ¯ V a ( θ ). We want to find θ that solves:

<!-- formula-not-decoded -->

Applying our standard stochastic gradient algorithm, we obtain the updating step:

<!-- formula-not-decoded -->

where φ ( a ) is a |B| -element vector.

## 9.5.2 Recursive formulas for statistical estimation

In this section, we provide a primer on recursive estimation. We assume that we are using a linear-in-the-parameters model of the form:

<!-- formula-not-decoded -->

Let y n be the n th observation of our dependent variable (what we are trying to predict) based on the observation ( x n 1 , x n 2 , . . . , x n I ) of our dependent variables (the x i are equivalent to the

basis functions we used earlier). If we define x 0 = 1, we can define

<!-- formula-not-decoded -->

to be an I + 1-dimensional column vector of observations. Throughout this section, and unlike the rest of the book, we use traditional vector operations, where x T x is an inner product (producing a scalar) while xx T is an outer product, producing a matrix of cross products.

Letting θ be the column vector of parameters, we can write our model as:

<!-- formula-not-decoded -->

We assume that the errors ( ε 1 , . . . , ε n ) are independent and identically distributed. We do not know the parameter vector θ , so we replace it with an estimate ¯ θ n which gives us the predictive formula:

<!-- formula-not-decoded -->

where ¯ y n is our predictor of x n . Our prediction error is:

<!-- formula-not-decoded -->

Our goal is to choose θ to minimize the mean squared error:

<!-- formula-not-decoded -->

It is well known that this can be solved very simply. Let X n be the n by I +1 matrix:

<!-- formula-not-decoded -->

Next, denote the vector of observations of the dependent variable as:

<!-- formula-not-decoded -->

The optimal parameter vector θ n (after n observations) is given by:

<!-- formula-not-decoded -->

Equation (9.30) is far too expensive to be useful in dynamic programming applications. Even for a relatively small number of parameters (which may not be that small), the matrix inverse is going to be too slow for most applications. Fortunately, it is possible to compute these formulas recursively. The updating equation for θ is:

<!-- formula-not-decoded -->

where H n is a column vector computing using:

<!-- formula-not-decoded -->

where γ n is a scalar and B n -1 is an I +1 by I +1 matrix. γ n is computed using:

<!-- formula-not-decoded -->

while the matrix B n is updated recursively using:

<!-- formula-not-decoded -->

The derivation of equations (9.31)-(9.34) is given in section. 9.6.3. Equation (9.31) has the feel of a stochastic gradient algorithm, but it has one significant difference. Instead of using a typical stepsize, we have the vector H n .

In our dynamic programming applications, the observations y n will represent estimates of the value of being in a state, and our independent variables will be either the states of our system (if we are estimating the value of being in each state) or the basis functions, in which case we are estimating the coefficients of the basis functions. The equations assume implicitly that the estimates come from a stationary series.

There are many problems where the number of basis functions can be extremely large. In these cases, even the efficient recursive expressions in this section cannot avoid the fact that we are still updating a matrix where the number of rows and columns is the number of states (or basis functions). If we are only estimating a few dozen or a few hundred parameters, this can be fine. If the number of parameters extends into the thousands, even this strategy would probably bog down. It is very important for students to work out the approximate dimensionality of the matrices before using these methods.

## 9.5.3 Recursive time-series estimation

We can use our recursive formulas to estimate a more general time-series model. At iteration n , let the elements of our basis function be given by:

<!-- formula-not-decoded -->

which is the observed value of each function given the observed state vector R n . If we wished to include a constant term, we would define a basis function φ = 1. Let n b = |B| be the number of basis functions used to explain the value function, so φ n is an n b -dimensional column vector. In our most general representation, we may feel that the value function should be explained over several iterations of inputs. Let

<!-- formula-not-decoded -->

be the history of observations of φ over the iterations. In addition to this information, we also have the history of observations of the updated values, which we represent using:

<!-- formula-not-decoded -->

where each ˆ v n is a scalar. Taken together, φ ( n ) and ˆ v ( n ) is our population of potential explanatory variables that we can use to help us predict ˆ v n +1 .

The standard way of stating a model that uses information from previous observations is with a backshift operator. Let q -1 be an operator that accesses information from the previous time period, as in:

<!-- formula-not-decoded -->

Now, define two series of backshift operators:

<!-- formula-not-decoded -->

where n o and n i are parameters that specify how many previous iterations of output and input vectors (respectively) will be used to predict the next iteration. We only use the previous n o observations of the value function observations ˆ v n and the previous n i observations of the basis functions φ . Using the backshift operators, we can write our model as:

<!-- formula-not-decoded -->

or, equivalently:

<!-- formula-not-decoded -->

where ε n is a random noise term. Since the elements ( φ m ) n -1 m = n -n i are |B| -dimensional vectors, the coefficients ( ˆ b m ) n i m =1 are also each |B| -dimensional vectors of coefficients, each of which has to be estimated. We assume that ˆ b m φ n -m represents an inner product between the two vectors.

Stated differently, we are trying to predict ˆ v n using:

<!-- formula-not-decoded -->

Let:

<!-- formula-not-decoded -->

be our vector of parameters, and:

<!-- formula-not-decoded -->

Our prediction error is computed using:

<!-- formula-not-decoded -->

We could take the history of observations and find θ using standard algorithms for minimizing the variance. A more efficient strategy is to use the recursive equations given in section 9.5.2.

In practice, it is not clear how much history should be used when specifying a model. This will be problem dependent, and obviously, the computational complexity will rise quickly as more history is added. It is likely that we would want to only use the current observations of the basis functions (that is, n i = 1), but perhaps several observations from the past history of the actual values would capture biases and trends. The appeal of the ability to incorporate a history of past estimates of value functions is that it can be a mechanism for adjusting for the bias, which minimizes the need to tune a stepsize.

## 9.5.4 Estimation using multiple observations

The previous methods assume that we get one observation and use it to update the parameters. Another strategy is to sample several paths and solve a classical least-squares problem for estimating the parameters. In the simplest implementation, we would choose a set of realizations ˆ Ω n (rather than a single sample ω n ) and follow all of them, producing a set of estimates (ˆ v n ( ω )) ω ∈ ˆ Ω n that we can use to update the value function.

If we have a set of observations, we then face the classical problem of finding a vector of parameters ˆ θ n that best match all of these value function estimates. Thus, we want to solve:

<!-- formula-not-decoded -->

This is the standard parameter estimation problem faced in the statistical estimation community. If ¯ V ( θ ) is linear in θ , then we can use the usual formulas for linear regression. If the function is more general, we would typically resort to nonlinear programming algorithms to solve the problem. In either case, ˆ θ n is still an update that needs to be smoothed in with the previous estimate θ n -1 , which we would do using:

<!-- formula-not-decoded -->

One advantage of this strategy is that in contrast with the updates that depend on the gradient of the value function, updates of the form given in equation (9.35) do not encounter a scaling problem, and therefore, we return to our more familiar territory where 0 &lt; α n ≤ 1. Of course, as the sample size ˆ Ω increases, the stepsize should also be increased because there is more information in ˆ θ n .

The usefulness of this particular strategy will be very problem dependent. In many applications, the computational burden of producing multiple estimates ˆ v n ( ω ) , ω ∈ ˆ Ω n before producing a parameter update will simply be too costly.

## 9.6 Why does it work?*

## 9.6.1 Proof of Proposition 1

Proof: The second term on the right hand side of equation (9.41) can be further simplified using,

<!-- formula-not-decoded -->

Combining equations (9.19), (9.41) and (9.36) gives us the result in equation (9.20). /square

## 9.6.2 Proof of Proposition 2

We start by defining

<!-- formula-not-decoded -->

Equation (9.37) gives us:

<!-- formula-not-decoded -->

We note that

<!-- formula-not-decoded -->

Similarly

<!-- formula-not-decoded -->

This allows us to write equation (9.38) as,

<!-- formula-not-decoded -->

We start with the second term on the right hand side of equation (9.39). Using equation (9.3), this term can be written as

<!-- formula-not-decoded -->

The term I can be rewritten using

<!-- formula-not-decoded -->

which means

<!-- formula-not-decoded -->

Combining (9.39) and (9.41) proves the proposition.

/square

## 9.6.3 Derivation of the recursive estimation equations

Here we derive the recursive estimation equations given by equations (9.31)-(9.34). To begin, we note that the matrix ( X n ) T X n is an I +1 by I +1 matrix where the element for row i , column j is given by:

<!-- formula-not-decoded -->

This term can be computed recursively using:

<!-- formula-not-decoded -->

In matrix form, this can be written:

<!-- formula-not-decoded -->

Keeping in mind that x n is a column vector, x n ( x n ) T is an I +1 by I +1 matrix formed by the cross products of the elements of x n . We now use the Sherman-Morrison formula (see section 9.6.4 for a derivation) for updating the inverse of a matrix

<!-- formula-not-decoded -->

where A is an invertible n × n matrix, and u is an n -dimensional column vector. Applying this formula to our problem, we obtain:

<!-- formula-not-decoded -->

The term ( X n ) T Y n can also be updated recursively using:

<!-- formula-not-decoded -->

To simplify the notation let:

<!-- formula-not-decoded -->

This simplifies our inverse updating equation (9.43) to:

<!-- formula-not-decoded -->

Combining (9.30) with (9.43) and (9.44) gives:

<!-- formula-not-decoded -->

We can start to simplify by using θ n -1 = B n -1 ( X n -1 ) T Y n -1 . We are also going to bring the term x n B n -1 inside the square brackets. Finally, we are going to bring the last term B n -1 x n y n inside the brackets by taking the coefficient B n -1 x n outside the brackets and multiplying the remaining y n by the scalar γ n = 1 + ( x n ) T B n -1 x n :

<!-- formula-not-decoded -->

Again, we use θ n -1 = B n -1 ( X n -1 ) T Y n -1 and observe that there are two terms x n B n -1 ( x n ) T y n that cancel, leaving:

<!-- formula-not-decoded -->

We note that θ n -1 ( x n ) T is our prediction of y n using the parameter vector from iteration n -1 and the explanatory variables x n . y n is, of course, the actual observation, so the difference is our error, ˆ ε n . Let:

<!-- formula-not-decoded -->

We can now write our updating equation using:

<!-- formula-not-decoded -->

## 9.6.4 The Sherman-Morrison updating formula

The Sherman-Morrison matrix updating formula (see Golub &amp; Loan (1996)) assumes that we have a matrix A , and that we are going to update it with the outer product of the column vector u to produce the matrix B :

<!-- formula-not-decoded -->

Pre-multiply by B -1 and post-multiply by A -1 , giving:

<!-- formula-not-decoded -->

Post-multiply by u :

<!-- formula-not-decoded -->

Note that u T A -1 u is a scalar. Divide through by ( 1 + u T A -1 u ) :

<!-- formula-not-decoded -->

Now post-multiply by u T A -1 :

<!-- formula-not-decoded -->

Equation (9.47) gives us:

<!-- formula-not-decoded -->

Substituting (9.49) into (9.48) gives:

<!-- formula-not-decoded -->

Solving for B -1 gives us:

<!-- formula-not-decoded -->

which is the desired formula.

## 9.7 Bibliographic notes

Strategies range from picking a fixed level of aggregation (Whitt (1978), Bean et al. (1987)), or using adaptive techniques that change the level of aggregation as the sampling process progresses (Bertsekas &amp; Castanon (1989), Mendelssohn (1982),Bertsekas &amp; Tsitsiklis (1996)), but which still use a fixed level of aggregation at any given time.

Bounds on state/row aggregation: Zipkin (1980 b ), Zipkin (1980 a )

LeBlanc &amp; Tibshirani (1996) outlines a general framework for how to combine a collection of general regression/classification fit vectors in order to obtain a better predictive model. The weights on the estimates from the individual predictors are computed by least squares minimization, stacked regression, generalized cross-validation and bootstrapping. Adaptive regression by mixing (Yang (1999)) assigns weights on candidate models that are combined after proper assessment of performance of the estimators, with the aim of reducing instability. The weights for combining the models are obtained as functions of the distributions of the error estimates and the variance of the random errors. '

```
Bayesian reference: Bernardo & Smith (1994) George et al. (2003)
```

## Exercises

- 9.1) In a spreadsheet, create a 4 × 4 grid where the cells are numbered 1, 2, . . . , 16 starting with the upper left hand corner and moving left to right, as shown below. We are going

|   1 |   2 |   3 |   4 |
|-----|-----|-----|-----|
|   5 |   6 |   7 |   8 |
|   9 |  10 |  11 |  12 |
|  13 |  14 |  15 |  16 |

to treat each number in the cell as the mean of the observations drawn from that cell.

Now assume that if we observe a cell, we observe the mean plus a random variable that is uniformly distributed between -1 and +1. Next define a series of aggregation where aggregation 0 is the disaggregate level, aggregation 1 divides the grid into four 2 × 2 cells, and aggregation 2 aggregates everything into a single cell. After n iterations, let ¯ v ( g,n ) a be the estimate of cell ' a ' at the n th level of aggregation, and let

<!-- formula-not-decoded -->

be your best estimate of cell a using a weighted aggregation scheme. Compute an overall error measure using:

<!-- formula-not-decoded -->

where ν a is the true value (taken from your grid) of being in cell a . Also let w ( g,n ) be the average weight after n iterations given to the aggregation level g when averaged over all cells at that level of aggregation (for example, there is only one cell for w (2 ,n ) ). Perform 1000 iterations where at each iteration you randomly sample a cell and measure it with noise. Update your estimates at each level of aggregation, and compute the variance of your estimate with and without the bias correction.

- a) Plot w ( g,n ) for each of the three levels of aggregation at each iteration. Do the weights behave as you would expect? Explain.
- b) For each level of aggregation, set the weight given to that level equal to one (in other words, we are using a single level of aggregation) and plot the overall error as a function of the number of iterations.
- c) Add to your plot the average error when you use a weighted average, where the weights are determined by equation (9.5) without the bias correction.
- d) Finally add to your plot the average error when you used a weighted average, but now determine the weights by equation (9.7) which uses the bias correction.
- e) Repeat the above assuming that the noise is uniformly distributed between -5 and +5.
6. 9.2) Prove equation 9.6.
7. 9.3) Show that the vector H n in the recursive updating formula from equation (9.45)

<!-- formula-not-decoded -->

reduces to H n = 1 /n for the case of a single parameter.

## Chapter 10

## The exploration vs. exploitation problem

A fundamental challenge with approximate dynamic programming is that our ability to estimate a value function may require that we visit states just to estimate the value of being in the state. Should we make a decision because we think it is the best decision (based on our current estimate of the values of states the decision may take us to), or do we make a decision just to try something new? This is a decision we face in day-to-day life, so it is not surprising that we face this problem in our algorithms.

This choice is known in the approximate dynamic programming literature as the 'exploration vs. exploitation' problem. Do we make a decision to explore a state? Or do we 'exploit' our current estimates of downstream values to make what we think is the best possible decision? It can cost time and money to visit a state, so we have to consider the future value of action in terms of improving future decisions.

Intertwined with this question is the challenge of learning. When we visit a state, what did we learn? In some problems, we obtain nothing more than another observation of the value of being in the state. But in many applications, we can use our experience of visiting one state to improve what we know about other states. When this ability is included, it can change our strategy.

## 10.1 A learning exercise: the nomadic trucker

A nice illustration of the explore vs. exploit problem is provided by our nomadic trucker example. Assume that the only attribute of our nomadic trucker is his location. Thus,

a = { i } , where i ∈ I is a location. At any location, we have two types of choices:

e

D a = The set of locations a driver with attribute a can move empty to.

D l a = The customer orders that are available to a driver with attribute a . D = D e ∪ D l

a a a

The set D l a is random. As the driver arrives to location i , he sees a set of customer orders that are drawn from a probability distribution. The driver may choose to serve one of these orders, thereby earning a positive revenue, or he may choose to move empty to another location (that is, he may choose a decision d ∈ D e a ). Included in the set D e a , where a = i , is location i , representing a decision to stay in the same location for another time period. The set D l a may be empty.

Each decision earns a contribution c ad which is positive if d ∈ D l and negative or zero if d ∈ D e . If the driver has attribute a n at iteration n , he observes a sample realization of the orders D l ( ω n ) and then makes his next decision d n by solving:

<!-- formula-not-decoded -->

Here, a M ( t, a, d ) tells us the destination that results from making a decision. ¯ V n -1 is our estimate of the value of being in this state. After making a decision, we compute:

<!-- formula-not-decoded -->

and then update our value function using:

<!-- formula-not-decoded -->

We start by initializing the value of being in each location to zero, and use a pure exploitation strategy. If we simulate 500 iterations of this process, we produce the pattern shown in figure 10.1. Here, the circles at each location are proportional to the value ¯ V 500 ( a ) of being in that location. The small circles indicate places where the trucker never visited. Out of 50 cities, our trucker has ended up visiting nine.

An alternative strategy is to initialize ¯ V 0 ( a ) to a large number. For our illustration, where rewards tends to average several hundred dollars per iteration (we are using a discount factor of 0.80), we might initialize the value function to $2000 which is higher than we would expect the optimal solution to be. Using the same strategy, visiting a state general produces a reduction in the estimate of the value of being in the state. Not surprisingly, the logic tends to favor visiting locations we have never visited before (or have visited the least). The

CHAPTER 10. THE EXPLORATION VS. EXPLOITATION PROBLEM

277

<!-- image -->

becomes stuck in a local solution, visiting only a handful of cities.

Figure 10.1: Using a pure exploitation strategy and low initial estimates the nomadic trucker becomes stuck in a local solution, visiting only a handful of cities.

resulting behavior is shown in figure 10.2. Here, the pattern of lines shows that after 500 iterations, we have visited almost every city.

How do these strategies compare? We also ran an experiment where we estimated the value functions by using a pure exploration strategy, where we ran five iterations of sampling every single location. Then, for all three methods of estimating the value function, we simulated the policy produced by these value functions for 200 iterations. The results are shown in figure 10.3. The results show that for this example, pure exploitation with a high initial estimate for the value function works better than when we use a low initial estimate, but estimating the value functions using a pure exploration strategy works best of all. Furthermore, the differences are fairly substantial.

## 10.2 Learning strategies

Much of the challenge of estimating a value function is identical to that facing any statistician trying to fit a model to data. The biggest difference is that in dynamic programming, we may choose what data to collect by controlling what states to visit. Further complicating the problem is that it takes time (and may cost money) to visit these states to collect the information. Do we take the time to visit the state, and better learn the value of being in the state? Or do we live with what we know?

CHAPTER 10. THE EXPLORATION VS. EXPLOITATION PROBLEM

"ow initial

Hiah initial

Evnlora alll tries to visit everywhere.

278

Figure 10.2: Using a pure exploitation strategy and high initial estimates the nomadic trucker tries to visit everywhere.

<!-- image -->

Figure 10.3: Expected value of policies from pure exploitation with low initial value functions, pure exploitation with high initial value function, and pure exploration.

<!-- image -->

Below we review several simple strategies, any of which can be effective for specific problem classes.

## 10.2.1 Pure exploration

Here, we use an exogenous process (such as random selection) to choose either a state to visit, or an action (which leads to a state). Once in a state, we sample information and obtain an estimate of the value of being in the state which is then used to update our estimate.

In a pure exploration strategy, we can guarantee that we visit every state, or at least have a chance of visiting every state. We need to remember that some problems have 10 100 states or more, so even if we run a million iterations, we may sample only a fraction of the complete state space. But at least we sample a broad range of the state space.

The amount of exploration we undertake depends in large part on the cost of collecting the information (how much time does it take to run each iteration) and the acceptable level of errors. The problem with a pure exploration strategy is that we may only be interested in a very tiny fraction of a large state space.

## 10.2.2 Pure exploitation

A pure exploitation strategy assumes that we have to make decisions by solving:

<!-- formula-not-decoded -->

Some authors refer to this as a greedy strategy, since we are doing the best that we think we can given what we know.

A pure exploitation may be needed for practical reasons. For example, consider a large resource allocation problem where we are acting on a resource vector R t = ( R ta ) a ∈A which we act on with a decision vector x t = ( x tad ) a ∈A ,d ∈D . For some applications in transportation, the dimensionality of R t may be in the thousands, while x t may be in the tens of thousands. For problems of this size, randomly choosing an action, or even a state, even if we run millions of iterations (very unlikely for problems of this size), means that we are sampling a tiny fraction of the state or action space. For such problems, exploration can be pointless. Furthermore, an exploitation policy avoids visiting states that are unreachable or truly suboptimal.

The problem with pure exploitation is that it is quite easy to become stuck in a local solution simply because we have poor estimates of the value of being in some states. While it is easy to construct small problems where this problem is serious, the errors can be substantial on virtually any problem that lacks specific structure that can be exploited to ensure convergence. As a rule, optimal solutions are not available for large problems, so we have to be satisfied with doing the best we can do. But just because your algorithm appears

to have converged, do not fool yourself into believing that you have reached an optimal, or even near-optimal, solution.

## 10.2.3 Mixed exploration and exploitation

A common strategy is to mix exploration and exploitation. We might specify an exploration rate ρ where ρ is the fraction of iterations where decisions should be chosen at random (exploration). The intuitive appeal of this approach is that we maintain a certain degree of forced exploration, while the exploitation steps focuses attention on the states that appear to be the most valuable.

This strategy is particularly popular for proofs of convergence because it guarantees that in the limit, all (reachable) states will be visited infinitely often. This property is then used to prove that estimates will reach their true value.

In practice, using a mix of exploration steps only adds value for problems with relatively small state or action spaces. The only exception arises when the problem lends itself to an approximation which is characterized by a relatively small number of parameters.

## 10.2.4 Boltzman exploration

The problem with exploration steps is that you are choosing a decision d ∈ D at random. Sometimes this means that you are choosing really poor decisions where you are learning nothing of value. An alternative is Boltzman exploration where from state S , a decision d is chosen with probability with a probability proportional to the estimated value of a decision. For example, let Q ( S, d ) = R ( S, d ) + ¯ V n ( S, d ) be the value of choosing decision d when we are in state S . Using Boltzman exploration, we would choose decision d with probability:

<!-- formula-not-decoded -->

T is known as the temperature, since in physical systems, electrons at high temperatures are more likely to bounce from one state to another. As the parameter T increases, the probability of choosing different decisions becomes more uniform. As T → 0, the probability of choosing the best decision approaches 1.0. It makes sense to start with T relatively large and steadily decrease it as the algorithm progresses.

Boltzman exploration provides for a more elegant choice of decision. Those which appear to be of lower value are selected with a lower probability. We focus our energy on the decisions that appear to be the most beneficial, but provide for intelligent exploration.

5 .

5

0

Pure exploitation

Projection algorithm

Figure 10.4: Pure exploration outperforms pure exploitation initially, but slows as the iterations progress.

<!-- image -->

## 10.2.5 Remarks

The tradeoff between exploration and exploitation is nicely illustrated in figure 10.4 where we are estimating the value of being in each state for a small problem with a few dozen states. For this problem, we are able to compute the exact value function, which allows us to compute the value of a policy using the approximate value function as a percentage of the optimal. This graph nicely shows that pure exploration has a much faster initial rate of convergence, whereas the pure exploitation policy works better as the function becomes more accurate.

This behavior, however, is very problem dependent. The value of any exploration strategy drops as the number of parameters increases. If a mixed strategy is used, the best fraction of exploration iterations is problem dependent, and may be difficult to ascertain without access to an optimal solution. Tests on smaller, computationally tractable problems (where exploration is more useful) will not tell us the right balance for larger problems.

Consider, for example, the problem of allocating hundreds (or thousands) of different types of assets, which can be described by a resource state vector R t with hundreds (or thousands) of dimensions. There may be 10 to 100 different types of decisions that we can act on each asset class, producing a decision vector x t with thousands or even tens of thousands of dimensions. The state space may be 10 10 or more, with an even larger action space. Choosing actions (or states) at random for exploration purposes, in an algorithm where we are running thousands (or tens of thousands) of iterations means we are sampling at most a tiny sample of states (and these are only being sampled once or twice).

For such large problems, an exploration strategy is going to be of little value unless we can exploit a significant amount of structure. At the same time, a pure exploitation strategy is very likely to become stuck in a local solution that may be of poor quality.

## 10.3 A simple information acquisition problem

Consider the situation of a company selling a product at a price p during time period t . Assume that production costs are negligible, and that the company wants to sell the product at a price that maximizes revenue. Let p ∗ be this price, which is unknown. Further assume that the lost revenue (per unit sold) is approximated by β ( p t -p ∗ ) 2 which, of course, can only be computed if we actually knew the optimal price.

In any given time period (e.g. a month) the company may conduct market research at a cost per unit sold of c (assume the company continues to sell the product during this time). When the company conducts a market research study, it obtains an imperfect estimate of the optimal price which we denote ˆ p t = p ∗ + /epsilon1 t where E /epsilon1 = 0 and V ar ( /epsilon1 ) = σ 2 . Let x t = 1 if the company conducts a market research study during time period t , and 0 otherwise. We assume that our ability to estimate the correct price is independent of our pricing policy. For this reason, the market research strategy, captured by x = ( x t ) t , is independent of the actual observations (and is therefore deterministic). Our goal is to minimize expected costs (lost revenue plus marketing costs) per unit over a finite horizon t = 1 , 2 , . . . , T .

Since each market research study gives us an unbiased estimate of the true optimal price, it makes sense for us to set our price to be the average over all the market research studies. Let:

<!-- formula-not-decoded -->

be the number of market research studies we have performed up to (and including) time t . Thus:

<!-- formula-not-decoded -->

p t is an unbiased estimate of p ∗ with variance

<!-- formula-not-decoded -->

where we assume for simplicity that σ 2 is known. We note that our lost revenue function was conveniently chosen so that:

<!-- formula-not-decoded -->

Since our decisions x t are independent of the state of our system, we can formulate the optimization problem for choosing x as follows:

<!-- formula-not-decoded -->

We use the intuitive result (which the reader is expected to prove in the exercises) that we should perform market research for µ time periods and then stop. This means that x t = 1 , t = 1 , 2 , . . . , µ with x t = 0 , t &gt; µ , which also implies that n t = t for t ≤ µ . Using this behavior, we may simplify F ( x ) to be:

<!-- formula-not-decoded -->

We can solve this easily if we treat time as continuous, which allows us to write F ( x, µ ) as:

<!-- formula-not-decoded -->

Differentiating with respect to µ and setting the result equal to zero gives:

<!-- formula-not-decoded -->

Finding the optimal point µ ∗ to stop collecting information requires solving:

<!-- formula-not-decoded -->

Applying the familiar solution to quadratic equations, and recognizing that we are interested in a positive solution, gives:

<!-- formula-not-decoded -->

We see from this expression that the amount of time we should be collecting information increases with σ 2 , β and T , and decreases with c , as we would expect. If there is no noise ( σ 2 = 0), then we should not collect any information. Most importantly, it highlights the concept that there is an optimal strategy for collecting information, and that we should collect more information when our level of uncertainty is higher. The next section extends this basic idea to a more general (but still restrictive) class of problems.

## 10.4 Gittins indices and the information acquisition problem

For the most part, the balance of exploration and exploitation is ad hoc, problem dependent and highly experimental. There is, however, one body of theory that offers some very important insights into how to best make the tradeoff between exploring and exploiting. This theory is often referred to as multiarmed bandits which is the name given to the underlying mathematical model, or Gittins indices which refers to the elegant method for solving the problem.

## 10.4.1 Foundations

Consider the problem faced by a gambler playing a set of slot machines ('one-armed bandits') in a casino. Now pretend that the probability of winning is different for each slot machine, but we do not know what these probabilities are. We can, however, obtain information about the probabilities by playing a machine and watching the outcomes. Because our observations are random, the best we can do is obtain statistical estimates of the probabilities, but as we play a machine more, the quality of our estimates improves.

Since we are looking at a set of slot machines, the problem is referred to as the multiarmed bandit problem. This is a pure exercise in information acquisition, since after every round, our player is faced with the same set of choices. Contrast this situation with most dynamic programs which involve allocating an asset where making a decision changes the attribute (state) of the asset. In the multiarmed bandit problem, after every round the player faces the same decisions with the same rewards. All that has changed is what she knows about the system.

This problem, which is extremely important in approximate dynamic programming, provides a nice illustration of what might be called the knowledge state (or information state ). The difference between the state of the resource (in this case, the player) and the state of what we know has confused authors since Bellman first encountered the issue. The vast majority of papers in dynamic programming implicitly assume that the state variable is the state of the resource. This is precisely the reason that our presentation in chapter 3 adopted the term 'resource state' to be clear about what we were referring to.

In our multiarmed bandit problem, let W i be the random variable that gives the amount

that we win if we play the i th bandit. Most of our presentation assumes that W i is normally distributed. Let θ i be the true mean of W i (which is unknown) and variance σ 2 i (which we may assume is known or unknown). Now let ( ¯ θ n i , ¯ σ 2 i ) be our estimate of the mean and variance of W i after n iterations. Under our assumption of normality, the mean and variance completely determine the distribution.

We next need to specify our transition equations. When we were managing physical assets, we used equations such as R t +1 = [ R t + x t -D t +1 ] + to capture the quantity of assets available. In our bandit problem, we have to show how the estimates of the parameters of the distribution evolve over time. Now let x n i = 1 if we play the i th slot machine during the n th round, and let W n i be the amount that we win in this round. Also let:

<!-- formula-not-decoded -->

be the total number of times we sampled the i th machine. Since the observations ( W n ′ i ) n n ′ =1 come from the same distribution, the best estimate of the mean is a simple average, which can be computed recursively using:

<!-- formula-not-decoded -->

Similarly, we would estimate the variance of W i using:

<!-- formula-not-decoded -->

We are more interested in the variance of ¯ θ n i which is given by:

<!-- formula-not-decoded -->

The apparent discrepancy in the stepsizes between (10.3) and (10.2) arises because of the small sample adjustment for variances when the mean is unknown.

One challenge in using (10.3) to estimate the variance, especially for larger problems, is that the number of observations N n i may be quite small, and often zero or 1. A reasonable approximation may be to assume (at least initially) that the variance is the same across the slot machines. In this case, we could estimate a single population variance using:

<!-- formula-not-decoded -->

which is updated after every play. The variance of ¯ θ n i would then be given by:

<!-- formula-not-decoded -->

Even if significant differences are suspected between different choices, it is probably a good idea to use a single population variance unless N n i is at least 10.

Under the assumption of normality, S n = ( ¯ θ n , ˆ σ n , N n i ) is our state variable, where equations (10.2) and (10.3) represent our transition function. We do not have a resource state variable because our 'resource' (the player) is always able to play the same machines after every round, without affecting the reward structure. Some authors (including Bellman) refer to ( ¯ θ n , ¯ σ n ) as the hyperstate , but given our definitions (see section 3.6), this is a classic state variable since it captures everything we need to know to model the future evolution of our system.

Given this model, it would appear that we have a classic dynamic program. We have a 2 |I| -dimensional state variable, which also happens to be continuous. Even if we could model ¯ θ n and ¯ σ n as discrete, we have a multidimensional state variable with all the computational challenges this entails.

In a landmark paper (Gittins &amp; Jones (1974)), it was shown that this problem could be solved as a series of one-dimensional problems using an index policy. That is, it is possible to compute a number ν i for each bandit i , using information about only this bandit. It is then optimal to choose which bandit to play next by simply finding the largest ν i for all i ∈ I . This is known as an index policy, and the values ν i are widely known as Gittins indices .

## 10.4.2 Basic theory of Gittins indices

Assume we face the choice of playing a single slot machine, or stopping and converting to a process that pays a reward ρ in each time period until infinity. If we choose to stop sampling and accept the fixed reward, the total future reward is ρ/ (1 -γ ). Alternatively, if we play the slot machine, we not only win a random amount W , we also learn something about the parameter θ that characterizes the distribution of W (for our presentation, E W = θ , but θ could be a vector of parameters that characterizes the distribution of W ). ¯ θ n represents our state variable, and the optimality equations are

<!-- formula-not-decoded -->

where we have written the value function to express the dependence on ρ . C ( ¯ θ n ) = E W is our expected reward given our estimate ¯ θ n .

Since we have an infinite horizon problem, the value function must satisfy the optimality equations:

<!-- formula-not-decoded -->

where ¯ θ ′ is defined by equation (10.2). It can be shown that if we choose to stop sampling in iteration n and accept the fixed payment ρ , then that is the optimal strategy for all future rounds. This means that starting at iteration n , our optimal future payoff (once we have decided to accept the fixed payment) is:

<!-- formula-not-decoded -->

which means that we can write our optimality recursion in the form:

<!-- formula-not-decoded -->

Now for the magic of Gittins indices. Let ν be the value of ρ which makes the two terms in the brackets in (10.5) equal. That is,

<!-- formula-not-decoded -->

ν depends on our current estimate of the mean, ¯ θ , the estimate of the variance ¯ σ 2 , and the number of observations n we have made of the process. We express this dependence by writing the index as ν ( ¯ θ, ¯ σ 2 , n ).

Now assume that we have a family of slot machines I , and let ν i ( ¯ θ i , ¯ σ 2 i , N n i ) be the value of ν ( ¯ θ i , ¯ σ 2 i , N n i ) that we compute for each slot machine i ∈ I , where N n i is the number of times we have played slot machine i by iteration n . An optimal policy for selecting slot machines is to choose the slot machine with the highest value for ν i ( ¯ θ i , ¯ σ 2 i , N n i ). Such policies are known as index policies , and for this problem, the parameters ν i ( ¯ θ i , ¯ σ 2 i , N n i ) are widely known as Gittins indices.

The computation of Gittins indices highlights a subtle issue when computing expectations for information-collection problems. The proper computation of the expectation required to solve the optimality equations requires, in theory, knowledge of exactly the distribution that we are trying to compute. To illustrate, the expected winnings are given by C ( ¯ θ n ) = E W = θ , but θ is unknown. Instead, we adopt a Bayesian approach that our expectation is computed with respect to the distribution we believe to be true. Thus, at iteration n we believe that our winnings are normally distributed with mean ¯ θ n , so we would use C ( ¯ θ n ) = ¯ θ n . The term E { V ( ¯ θ n +1 | ρ ) ∣ ∣ ¯ θ n } captures what we believe the effect of observing W n +1 will have on our estimate ¯ θ n +1 , but this belief is based on what we think the distribution of W n +1 is, rather than the true distribution.

The beauty of Gittins indices (or any index policy) is that it reduces N -dimensional problems into a series of one dimensional problems. The problem is that solving equation (10.5) (or equivalently, (10.6)) offers its own challenges. Finding ν ( ¯ θ, ¯ σ 2 , n ) requires solving the optimality equation in (10.5) for different values of ρ until (10.6) is satisfied. In addition, this has to be done for different values of ¯ θ and n . Although algorithmic procedures have been designed for this, they are not simple.

## 10.4.3 Gittins indices for normally distributed rewards

The calculation of Gittins indices is simplified for special classes of distributions. In this section, we consider the case where the observations of rewards W are normally distributed. Students learn in their first statistics course that normally distributed random variables satisfy a nice property. If Z is normally distributed with mean 0 and variance 1, and if:

<!-- formula-not-decoded -->

then X is normally distributed with mean µ and variance σ 2 . This property simplifies what are otherwise difficult calculations about probabilities of events. For example, computing Prob [ X ≥ x ] is difficult because the normal density function cannot be integrated analytically. Instead, we have to resort to numerical procedures. But because of the above translationary and scaling properties of normally distributed random variables, we can perform the difficult computations for the random variable Z (the 'standard normal deviate'), and use this to answer questions about any random variable X . For example, we can write:

<!-- formula-not-decoded -->

Thus, the ability to answer probability questions about Z allows us to answer the same questions about any normally distributed random variable.

The same property applies to Gittins indices. Although the proof requires some development, it is possible to show that:

<!-- formula-not-decoded -->

Thus, we have only to compute a 'standard normal Gittins index' for problems with mean 0 and variance 1, and n observations.

Unfortunately, as of this writing, there do not exist easy-to-use software utilities for computing standard Gittins indices. The situation is similar to doing statistics before computers when students had to look up the cumulative distribution for the standard normal deviate in the back of a statistics book. Table 10.1 is exactly such a table for Gittins indices. The table gives indices for both the parameters-known and parameters unknown cases. In the parameters known case, we assume that σ 2 is given, which allows us to estimate the variance of the estimate for a particular slot machine just by dividing by the number of observations.

Given access to a table of values, applying Gittins indices becomes quite simple. Instead of choosing the option with the highest ¯ θ n i (which we would do if we were ignoring the value

Table 10.1: Gittins indices for the case of observations that are normally distributed with mean 0, variance 1, from Gittins (1989).

|              | Discount factor   | Discount factor   | Discount factor   | Discount factor   |
|--------------|-------------------|-------------------|-------------------|-------------------|
|              | Known variance    | Known variance    | Unknown variance  | Unknown variance  |
| Observations | 0.95              | 0.99              | 0.95              | 0.99              |
| 1            | 0.9956            | 1.5758            |                   |                   |
| 2            | 0.6343            | 1.0415            | 10.1410           | 39.3343           |
| 3            | 0.4781            | 0.8061            | 1.1656            | 3.1020            |
| 4            | 0.3878            | 0.6677            | 0.6193            | 1.3428            |
| 5            | 0.3281            | 0.5747            | 0.4478            | 0.9052            |
| 6            | 0.2853            | 0.5072            | 0.3590            | 0.7054            |
| 7            | 0.2528            | 0.4554            | 0.3035            | 0.5901            |
| 8            | 0.2274            | 0.4144            | 0.2645            | 0.5123            |
| 9            | 0.2069            | 0.3808            | 0.2353            | 0.4556            |
| 10           | 0.1899            | 0.3528            | 0.2123            | 0.4119            |
| 20           | 0.1058            | 0.2094            | 0.1109            | 0.2230            |
| 30           | 0.0739            | 0.1520            | 0.0761            | 0.1579            |
| 40           | 0.0570            | 0.1202            | 0.0582            | 0.1235            |
| 50           | 0.0464            | 0.0998            | 0.0472            | 0.1019            |
| 60           | 0.0392            | 0.0855            | 0.0397            | 0.0870            |
| 70           | 0.0339            | 0.0749            | 0.0343            | 0.0760            |
| 80           | 0.0299            | 0.0667            | 0.0302            | 0.0675            |
| 90           | 0.0267            | 0.0602            | 0.0269            | 0.0608            |
| 100          | 0.0242            | 0.0549            | 0.0244            | 0.0554            |

of collecting information), we choose the option with the highest value of:

<!-- formula-not-decoded -->

This strategy is attractive because it is simple to apply and does not require using the device of a pure exploration step. As we have pointed out, for large state and action spaces, exploration steps are of little value when the number of iterations being run is much smaller than the state space. Using Gittins indices allows us to use a modified exploitation strategy, where the choice of decision requires adding the term (¯ σ 2 i ) n ν (0 , 1 , N n i ) to the value of being in a state. Since the indices ν (0 , 1 , N n i ) decline naturally to zero (along with the standard deviation ¯ σ ), in the limit we have a pure exploitation strategy.

Perhaps the most useful insight from the multiarmed bandit problem is that it illustrates the role that uncertainty plays in the exploration process. We have to strike a balance between choosing what appears to be the best option, and what might be the best option. If an option has a somewhat lower estimated value, but where the variance is so high that the upper tail exceeds the upper tail of another option, then this is something we should explore. How far we go out on the upper tail depends on the number of observations and the discount factor. As the discount factor approaches 1.00, the value of exploring goes up.

## 10.4.4 Gittins exploration

We have to remind ourselves that Gittins indices work only for multiarmed bandit problems. This is a very special problem class, since at each iteration we face exactly the same set of choices. In addition, while our understanding of the value of each choice changes, the actual flows of rewards is the same from one iteration to the next. Not surprisingly, this is a very small set of dynamic programs.

Consider now a more general problem where an asset with attribute a t , after making decision d ∈ D a , becomes an asset with attribute a ′ t where we face options d ′ ∈ D a ′ . Consider the challenge of deciding which of two decisions d 1 , d 2 ∈ D a is better. Decision d 1 produces an asset with attribute a M ( a, d 1 ) = a ′ 1 . Let ¯ V n ( a ′ 1 ) be our estimate of the value of the asset at this time. Similarly, decision d 2 produces an asset with attribute a M ( a, d 2 ) = a ′ 2 and estimated value ¯ V n ( a ′ 2 ). Finally, let c ad be the immediate contribution of the decision.

If we were using a pure exploitation policy, we should choose the decision that minimizes c ad + ¯ V n ( a M ( a, d )). The development of Gittins indices suggests that we could apply the idea heuristically as follows:

<!-- formula-not-decoded -->

where ¯ σ n ( a M ( a, d )) is our estimate of the standard deviation of ¯ V ( a M ( a, d )), and ν ( n ) tells us how many standard deviations away from the mean we should consider (which is a function of the number of observations n ). For this more general problem, we do not have any theory that tells us what ν ( n ) should be. Since strategies for balancing exploration and exploitation are largely heuristic in nature, it seems reasonable to simply adopt a heuristic rule for ν ( n ). An analysis of the exact Gittins indices suggests that we might use

<!-- formula-not-decoded -->

where ρ G is a parameter we have to choose using calibration experiments. The presence of √ n in the denominator reflects the observation that the Gittins indices drop approximately with the square root of the number of observations. We note that we use (10.7) only to decide which state to visit next. To update the value of being in state a , we still use

<!-- formula-not-decoded -->

We refer to the policy of using equation (10.7) to decide what state should be visited next as a Gittins exploration strategy . The attraction of a Gittins exploration policy is that it does not depend on randomly sampling states or actions. This means we may be able to use it even when our decision is a vector x t acting on a resource vector R t , both of which may have hundreds or thousands of dimensions. A Gittins exploration strategy encourages us to visit states which might be attractive.

Experiments with this idea using the nomadic trucker problem quickly demonstrated that the biggest challenge is estimating ¯ σ n ( a M ( a, d )). Simply estimating the variance of the estimate of the value of being in a state does not accurately estimate the spread of the errors between the true value function (which we are able to compute) and the approximate one. It was not unusual to find that the exact value function for a state was 10 to 30 standard deviations away from our estimate. The problem is that there is a compounding effect of errors in the value function, especially when we use a pure exploitation strategy. For this reason, we found it necessary to try values of ρ G that produces values of ν ( n ) that were much larger than what would be found in the Gittins tables.

Figure 10.5 shows the percent error in the value function obtained using a Gittins exploration strategy for both a single and multiattribute nomadic trucker problem. For the single attribute problem (the attribute is simply the location of the truck, of which there are only 50), a pure exploration strategy produced the most accurate value functions (figure 10.5a). This outperformed a Gittins exploration policy even with ρ G = 500. By contrast, a Gittins exploration strategy on a multiattribute problem (with three attributes and an attribute space over 1,000), using ρ G ∈ (5 , 10 , 20 , 50) significantly outperformed both pure exploration and exploitation policies (figure 10.5b). For this specific example, ρ G produced the best results, with values of ρ G of 5 and 20 producing nearly equivalent results. However, the results for the single attribute problem suggest that this is not a generalizable result.

In practice, we cannot count on having access to the optimal value function. Instead, we may have to take an approximate value function and simulate decisions using this approximation. Using this approach, we are getting an estimate of the value of a policy. Figure 10.6 shows the value of the policies produced by the value function approximations found using the exploration strategies given in figure 10.5. Figure 10.6a shows that for the single attribute problem, the high quality value function produced by the exploration strategy also produced the best policies. For the multiattribute problem, the better value functions produced by the Gittins exploration policies also translated into better policies, but the differences were less noticeable. Pure exploration produces the worst policies initially, but eventually catches up. Pure exploitation starts well, but tails off in later iterations. All the Gittins exploration policies perform reasonably well throughout the range tested.

## 10.5 Why does it work?**

An optimal weighting strategy for hierarchical aggregation.

Tsitsklis 'short proof' of Gittins indices?

## 10.6 Bibliographic notes

Bandit processes: Weber (1992), Whittle (1982)

10.5a Effect of Gittins exploration on a single attribute nomadic trucker problem

<!-- image -->

10.5b Effect of Gittins exploration on a multiattribute nomadic trucker problem

<!-- image -->

Figure 10.5: The heuristic application of Gittins to a multiattribute asset management problem produces more accurate value functions than either pure exploitation or pure exploration policies.

Q-learning for bandit processes Duff &amp; Barto (2003), Duff (1995), Berry &amp; Fristedt (1985)

10.6a The value of the policies for the single attribute nomadic trucker

<!-- image -->

10.6b The value of the policies for the multiattribute nomadic trucker

<!-- image -->

Figure 10.6: The value of the policy produced by the approximate value functions created using different exploration policies.

Gittins indices: Gittins (1979), Gittins (1981), Lai &amp; Robbins (1985), Gittins &amp; Jones (1974), Gittins (1989)

## Exercises

- 10.1) Joe Torre, manager of the Yankees (the greatest baseball team in the country), has to struggle with the constant game of guessing who his best hitters are. The problem is that he can only observe a hitter if he puts him in the order. He has four batters that he is looking at. The table below shows their actual batting averages (that is to say, batter 1 will produce hits 30 percent of the time, batter 2 will get hits 32 percent of the time, and so on). Unfortunately, Joe doesnt know these numbers. As far as he is concerned, these are all .300 hitters.

For each at bat, Joe has to pick one of these hitters to hit. The table shows what would have happened if each batter were given a chance to hit (1 = hit, 0 = out). Again, Joe does not get to see all these numbers. He only gets to observe the outcome of the hitter who gets to hit.

Assume that Joe always lets the batter hit with the best batting average. Assume that he uses an initial batting average of .300 for each hitter (in case of a tie, use batter 1 over batter 2 over batter 3 over batter 4). Use .300 as your initial estimate of each batters average. Whenever a batter gets to hit, calculate a new batting average by putting an 80 percent weight on your previous estimate of his average plus a 20 percent weight on how he did for his at bat. So, according to this logic, you would choose Batter 1 first. Since he does not get a hit, his updated average would d be 0.80(.200) + .20(0)=.240. For the next at bat, you would choose Batter 2 because your estimate of his average is still .300, while your estimate for Batter 1 is now .240.

After 10 at bats, who would you conclude is your best batter? Comment on the limitations of this way of choosing the best batter. Do you have a better idea? (It would be nice if it were practical.)

|     | Actual batting average   | Actual batting average   | Actual batting average   | Actual batting average   |
|-----|--------------------------|--------------------------|--------------------------|--------------------------|
|     | 0.300                    | 0.320                    | 0.280                    | 0.260                    |
| Day | Batter                   | Batter                   | Batter                   | Batter                   |
|     | A                        | B                        | C                        | D                        |
| 1   | 0                        | 1                        | 1                        | 1                        |
| 2   | 1                        | 0                        | 0                        | 0                        |
| 3   | 0                        | 0                        | 0                        | 0                        |
| 4   | 1                        | 1                        | 1                        | 1                        |
| 5   | 1                        | 1                        | 0                        | 0                        |
| 6   | 0                        | 0                        | 0                        | 0                        |
| 7   | 0                        | 0                        | 1                        | 0                        |
| 8   | 1                        | 0                        | 0                        | 0                        |
| 9   | 0                        | 1                        | 0                        | 0                        |
| 10  | 0                        | 1                        | 0                        | 1                        |

- 10.2) There are four paths you can take to get to your new job. On the map, they all seem reasonable, and as far as you can tell, they all take 20 minutes, but the actual times vary quite a bit. The value of taking a path is your current estimate of the travel time on that path. In the table below, we show the travel time on each path if you had travelled that path. Start with an initial estimate of each value function of 20 minutes with your tie-breaking rule to use the lowest numbered path. At each iteration, take the path with the best estimated value, and update your estimate of the value of the path based on your experience. After 10 iterations, compare your estimates of each path to the estimate you obtain by averaging the 'observations' for each path over all 10 days. How well did you do?
- 10.3) We are going to try again to solve our asset selling problem, We assume we are holding a real asset and we are responding to a series of offers. Let ˆ p t be the t th offer, which is uniformly distributed between 500 and 600 (all prices are in thousands of dollars). We also assume that each offer is independent of all prior offers. You are willing to consider up to 10 offers, and your goal is to get the highest possible price. If you have not accepted the first nine offers, you must accept the 10 th offer.
- a) Write out the decision function you would use in an approximate dynamic programming algorithm in terms of a Monte Carlo sample of the latest price and a current estimate of the value function approximation.
- b) Write out the updating equations (for the value function) you would use after solving the decision problem for the t th offer.
- c) Implement an approximate dynamic programming algorithm using synchronous state sampling. Using 100 iterations, write out your estimates of the value of being in each state immediately after each offer.
- d) From your value functions, infer a decision rule of the form 'sell if the price is greater than ¯ p t .'

|     |   Paths |   Paths |   Paths |   Paths |
|-----|---------|---------|---------|---------|
| Day |       1 |       2 |       3 |       4 |
| 1   |      37 |      29 |      17 |      23 |
| 2   |      32 |      32 |      23 |      17 |
| 3   |      35 |      26 |      28 |      17 |
| 4   |      30 |      35 |      19 |      32 |
| 5   |      28 |      25 |      21 |      26 |
| 6   |      24 |      19 |      25 |      31 |
| 7   |      26 |      37 |      33 |      30 |
| 8   |      28 |      22 |      28 |      27 |
| 9   |      24 |      28 |      31 |      30 |
| 10  |      33 |      29 |      17 |      29 |

## Chapter 11

## Value function approximations for resource allocation

In chapter 9, we focused on estimating the value of being in a discrete state, a problem that we posed in terms of managing a single asset. In this chapter, we turn our attention to the challenge of estimating the value of being in a state when we are managing multiple assets or asset classes.

Weassume throughout this chapter that we have a resource vector R t where the number of dimensions is 'not too large.' Practically speaking, R t may have hundreds or even thousands of dimensions, but problems with more than 10,000 dimensions tend to be computationally very difficult using the readily available hardware available as of this writing. If R t is discrete, we may still be facing a state space of staggering size, but we are going to treat R t as continuous and focus on separable approximations or those where the number of parameters is a manageable size.

We consider a series of approximation strategies of increasing sophistication:

- Linear approximations - These are typically the simplest nontrivial approximations that work well when the functions are approximately linear over the range of interest. It is important to realize that we mean 'linear in the resource state' as opposed to the more classical 'linear in the parameters' model that we considered earlier.
- Separable, piecewise linear, concave (convex if minimizing) - These functions are especially useful when we are interested in integer solutions. Separable functions are relatively easy to estimate and offer special structural properties when solving the optimality equations.
- Auxiliary functions - This is a special class of algorithms that fixes an initial approximation and uses stochastic gradients to tilt the function.
- General nonlinear regression equations - Here, we bring the full range of tools available from the field of regression. These techniques can be used for more general problems than just approximating V ( R ), but we use this setting to illustrate them.

Ultimately, the challenge of estimating value functions can draw on the entire field of statistical estimation. Approximate dynamic programming introduces some unique challenges to the problem of statistically estimating value functions, but in the end, it all boils down to statistical estimation.

## 11.1 Value functions versus gradients

It is common in dynamic programming to talk about the problem of estimating the value of being in a state. In the arena of asset management, it is often the case that we are more interested in estimating the derivative of the function rather than the function itself.

In principal, the challenge of estimating the slope of a function is the same as that of estimating the function itself (the slope is simply a different function). However, there can be important, practical advantages to estimating slopes. First, if the function is approximately linear, it may be possible to replace estimates of the parameter at each state (or set of states) with a single parameter which is the estimate of the slope of the function. Estimating constant terms is typically unnecessary.

A second and equally important difference is that if we estimate the value of being in a state, we get one estimate of the value of being in a state when we visit that state. When we estimate a gradient, we get an estimate of a gradient for each parameter. For example, if R t = ( R ta ) a ∈A is our asset vector and V t ( R t ) is our value function, then the gradient of the value function with respect to R t would look like:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

There may be additional work required to obtain each element of the gradient, but the incremental work can be far less than the work required to get the value function itself. This is particularly true when the optimization problem naturally returns these gradients (for example, dual variables from a linear program), but this can even be true when we have to resort to numerical derivatives. Once we have all the calculations to solve a problem once, solving small perturbations can be very inexpensive.

There is one important problem class where finding the value of being in a state, and finding the derivative, is equivalent. That is the case of managing a single asset (see section 3.12). In this case, the state of our system (the asset) is the attribute vector a , and we are

interested in estimating the value V ( a ) of our asset being in state a . Alternatively, we can represent the state of our system using the vector R t , where R ta = 1 indicates that our asset has attribute a (we assume that ∑ a ∈A R ta = 1). In this case, the value function can be written:

<!-- formula-not-decoded -->

Here, the coefficient v ta is the derivative of V t ( R t ) with respect to R ta .

In a typical implementation of an approximate dynamic programming algorithm, we would only estimate the value of an asset when it is in a particular state (given by the vector a ). This is equivalent to finding the derivative ˆ v a only for the value of a where R ta = 1. By contrast, computing the gradient ∇ R t V t ( R t ) implicitly assumes that we are computing ˆ v a for each a ∈ A . There are some algorithmic strategies (see, for example, section 14.6) where this assumption is implicit in the algorithm. Computing ˆ v a for all a ∈ A is reasonable if the attribute state space is not too large (for example, if a is a physical location among a set of several hundred locations). If a is a vector, then enumerating the attribute space can be prohibitive (it is, in effect, the 'curse of dimensionality' revisited).

Given these issues, it is critical to first determine whether it is necessary to estimate the slope of the value function, or the value function itself. The result can have a significant impact on the algorithmic strategy.

## 11.2 Linear approximations

There are a number of problems where we are allocating assets of different types. As in the past, we let a be the attributes of an asset and R ta be the quantity of assets with attribute a in our system at time t with R t = ( R ta ) a ∈A . R t may describe our investments in different asset classes (growth stocks, value stocks, index funds, international mutual funds, domestic stock funds, bond funds). Or R t might be the amount of oil we have in different reserves or the number of people in a management consulting firm with particular skill sets. We want to make decisions to acquire or drop assets of each type, and we want to capture the impact of decisions now on the future through a value function V t ( R t ).

Rather than attempt to estimate V t ( R t ) for each value of R t , it may make more sense to estimate a linear approximation of the value function with respect to the resource vector. Linear approximations can work well when the single-period contribution function is continuous and increases or decreases monotonically over the range we are interested in (the function may or may not be differentiable). They can also work well in settings where the value function increases or decreases monotonically, even if the value function is neither convex nor concave, nor even continuous.

To illustrate, consider the problem of purchasing a commodity. Let

D t = The random demand during time interval t .

R t = The commodities on hand at time t to be used during time interval t +1.

x t = The quantity ordered to be used during time interval t +1.

p t = The market price for selling commodities during time interval t .

c t = The purchase cost for commodities purchased at time t .

At time t , we know the price p t and demand D t for time interval t , but we have to choose how much to order for the next time interval. We can do this by solving:

<!-- formula-not-decoded -->

where

R

t

+1

= [

R

t

+

x

t

-

D

t

+1

]

+

Now, assume that we introduce a linear value function approximation:

<!-- formula-not-decoded -->

The resulting approximation can be written:

<!-- formula-not-decoded -->

We assume that we can compute, or at least approximate, the expectation in equation (11.3). If this is the case, we may approximate the gradient at iteration n using a numerical derivative, as in:

<!-- formula-not-decoded -->

We may either use ˆ v t as the slope of the function (that is, ¯ v t = ˆ v t ), or we may perform smoothing on ˆ v :

<!-- formula-not-decoded -->

Linear approximations are especially useful in the context of more complex problems (for example, those involving multiple asset types). The quality of the approximation depends on how much the slope changes as a function of R t .

## 11.3 Monotone function approximations

There are a number of settings in asset management where we can prove that a value function is increasing or decreasing in the state variable. These are referred to as monotone functions . If the function is increasing in the state variable, we might say that it is 'monotonically increasing,' or that it is isotone (although the latter term is not widely used).

Assume we have such a function, which means that while we do not know the value function exactly, we know that V ( R +1) ≤ V ( R ) (for scalar R ). Next, assume our current approximation ¯ V n -1 ( R ) satisfies this property, and that we have a new estimate ˆ v n for R = R n . If we use our standard updating algorithm, we would write:

<!-- formula-not-decoded -->

After the update, it is quite possible that our updated approximation no longer satisfies our monotonicity property. One way to maintain monotonicity is through the use of a leveling algorithm, which works as follows

<!-- formula-not-decoded -->

where x ∧ y = max { x, y } , and x ∨ y = min { x, y } . Equation (11.4) starts by updating the slope ¯ V n ( r ) for r = R n . We then want to make sure that the slopes are declining. So, if we find a slope to the right that is larger, we simply bring it down to our estimated slope for r = R n . Similarly, a slope to the left that is smaller, we simply raise it to the slope for r = R n . The steps are illustrated in figure 11.1.

The leveling algorithm is easy to visualize, but it is unlikely to be the best way to maintain monotonicity. For example, we may update a value at r = R n for which there are very few observations. But because it produces an unusually high or low estimate, we find ourselves simply forcing other slopes higher or lower just to maintain monotonicity.

A more elegant strategy is the SPAR algorithm which works as follows. Assume that we start with our original set of values ( ¯ V n -1 ( r )) r ≥ 0 , and that we sample r = R n and obtain an estimate of the slope ˆ v n . After the update, we obtain the set of values (which we store temporarily in the function ¯ y n ( r )):

<!-- formula-not-decoded -->

If ¯ y n ( r ) ≥ ¯ y n ( r + 1) for all r , then we are in good shape. If not, then either ¯ y n ( R n ) &lt; ¯ y n ( R n + 1) or ¯ y n ( R n -1) &lt; ¯ y n ( R n ). We can fix the problem by solving the projection

<!-- image -->

11.1a:

Initial monotone function.

11.1b: After update of a single segment.

<!-- image -->

11.1c: After leveling operation.

<!-- image -->

Figure 11.1: Steps of the the leveling algorithm. Figure 11.1a shows the initial monotone function, with the observed R and observed value of the function ˆ v . Figure 11.1b shows the function after updating the single segment, producing a non-monotone function. Figure 11.1c shows the function after monotonicity restored by leveling the function.

problem:

<!-- formula-not-decoded -->

subject to:

<!-- formula-not-decoded -->

Solving this projection is especially easy. Imagine that after our update, we have a violation to the left. The projection is achieved by averaging the updated cell with all the cells to the left that create a monotonicity violation. This means that we want to find the largest i ≤ R n such that:

<!-- formula-not-decoded -->

In other words, we can start by averaging the values for R n and R n -1 and checking to see if we now have a concave function. If not, we keep lowering the left end of the range until we either restore monotonicity or reach r = 0. If our monotonicity violation is to the right, then we repeat the process to the right.

The process is illustrated in figure 11.2. We start with a monotone set of values (a), then update one of the values to produce a monotonicity violation (b), and finally average the violating values together to restore monotonicity (c).

## 11.4 The SHAPE algorithm for continuously differentiable problems

A particularly simple algorithm for approximating value functions of continuous resources starts with an initial approximation and then 'tilts' this function to improve the approximation. The concept is most effective if it is possible to build an initial approximation, perhaps using some simplifications, that produces a 'pretty good' solution. The idea works as follows: Assume that we have access to some sort of initial approximation that we will call ¯ V 0 t ( R ), which we assume is continuously differentiable. Since we can choose this approximation, we can further assume that the derivatives are fairly easy to compute (for example, it might be a low order polynomial). We also assume that we have access to a stochastic gradient ˆ v t = ∇ V ( R n -1 t -1 , ω n ). So, we have an exact gradient of our approximate function and a stochastic gradient of the real function.

We now update our approximate function using:

<!-- formula-not-decoded -->

<!-- image -->

11.2a: Initial monotone function.

<!-- image -->

it

11.2b: After update of a single segment.

<!-- image -->

11.2c: After projection.

Figure 11.2: Steps of the SPAR algorithm. Figure 11.2a shows the initial monotone function, with the observed R and observed value of the function ˆ v . Figure 11.2b shows the function after updating the single segment, producing a non-monotone function. Figure 11.2c shows the function after the projection operation.

Step 0 Initialize ¯ V 0 and set n = 1.

Step 1 Sample R n .

Step 2 Observe a sample of the value function ˆ v n .

Step 3 Calculate the vector y n as follows

<!-- formula-not-decoded -->

Step 4 Project the updated estimate onto the space of monotone functions:

<!-- formula-not-decoded -->

by solving (11.6)-(11.7). Increase n by one and go to Step 1.

Figure 11.3: The learning form of the separable, projective approximation routine (SPAR).

The first term on the right hand side is our current functional approximation. The second term is a linear adjustment (note that the term in parentheses is a constant) that adds to the current approximation the difference between the stochastic gradient of the real function and the exact gradient of the current approximation. This linear adjustment has the effect of tilting the original approximation. As a result, this algorithm does not change the shape of the original approximation, but does help to fix errors in the slope of the approximation.

The steps of the SHAPE algorithm are illustrated in figure 11.4. The algorithm is provably convergent if V ( x, W ) and ¯ V 0 ( x ) are continuously differentiable (see section 14.7.1), but it can be used as an approximation even when these conditions are not satisfied.

We can illustrate the SHAPE algorithm using a simple numerical example:

<!-- formula-not-decoded -->

where W represents random measurement error, which is normally distributed with mean 0 and variance 4. Now, assume that we start with a convex approximation ˆ f 0 ( s ).

<!-- formula-not-decoded -->

We begin by obtaining the initial solution x 0 :

<!-- formula-not-decoded -->

Note that our solution to the approximate problem may be unbounded, requiring us to impose artificial limits. Since our approximation is concave, we can set the derivative equal

<!-- image -->

11.4a: True function and initial approximation.

11.4b: Difference between stochastic gradient of true function and actual gradient of approximation.

<!-- image -->

11.4c: Updated approximation.

<!-- image -->

Figure 11.4: Illustration of the steps of the SHAPE algorithm.

to zero to find:

<!-- formula-not-decoded -->

which gives us x 0 = 2 . 25. Since x 0 ≥ 0, it is optimal. To find the stochastic gradient, we have to sample the random variable W . Assume that W ( ω 1 ) = 1 . 75. Our stochastic gradient is then:

<!-- formula-not-decoded -->

Thus, while we have found the optimal solution to the approximate problem (which produces a zero slope), our estimate of the slope of the true function is positive, so we update with the adjustment:

<!-- formula-not-decoded -->

This algorithm is provably convergent for two-stage problems even if the original approximation is something simple such as a separable polynomial. For example, we could use something as simple as:

<!-- formula-not-decoded -->

where ¯ R a is a centering term.

The SHAPE algorithm is incredibly simple, but has seen little numerical work. It is likely to be more stable than a simple linear approximation, but the best results will be obtained when information about the problem can be used to develop an initial approximation that captures the structure of the real problem. Arbitrary approximations (such as R 2 ) are unlikely to add much value because they contain no information about the problem.

## 11.5 Regression methods

As in chapter 9 we can create regression models where the basis functions are manipulations of the number of resources of each type. For example, we might use:

<!-- formula-not-decoded -->

where θ = ( θ 0 , ( θ 1 a ) a ∈A , ( θ 2 a ) a ∈A ) is a vector of parameters that are to be determined. The choice of explanatory terms in our approximation will generally reflect an understanding of the properties of our problem. For example, equation (11.10) assumes that we can use a mixture of linear and separable quadratic terms. A more general representation is to assume that we have developed a family B of basis functions ( φ b ( R )) b ∈B . Examples of a basis function are

<!-- formula-not-decoded -->

A common strategy is to capture the number of resources at some level of aggregation. For example, if we are purchasing emergency equipment, we may care about how many pieces we have in each region of the country, and we may also care about how many pieces of a type of equipment we have (regardless of location). These issues can be captured using a family of aggregation functions G b , b ∈ B , where G b ( a ) aggregates an attribute vector a into a space A ( b ) where for every basis function b there is an element a b ∈ A ( b ) . Our basis function would then be expressed using:

<!-- formula-not-decoded -->

As we originally introduced in section 9.4, the explanatory variables used in the examples above, which are generally referred to as independent variables in the regression literature, are typically referred to as basis functions in the approximate dynamic programming literature. A basis function can be linear, nonlinear separable, nonlinear nonseparable, and even nondifferentiable, although the nondifferentiable case will introduce additional technical issues. The challenge, of course, is that it is the responsibility of the modeler to devise these functions for each application. We have written our basis functions purely in terms of the resource vector, but it is possible for them to be written in terms of other parameters in a more complex state vector, such as asset prices. Given a set of basis functions, we can write our value function approximation as:

<!-- formula-not-decoded -->

It is important to keep in mind that ¯ V ( R | θ ) (or more generally, ¯ V ( S | θ )), is any functional form that approximates the value function as a function of the state vector parameterized by θ . Equation (11.11) is a classic linear-in-the-parameters function. We are not constrained to this form, but it is the simplest and offers some algorithmic shortcuts.

The issues that we encounter in formulating and estimating ¯ V ( θ, R ) are the same that any student of statistical regression would face when modeling a complex problem. The major difference is that our data arrives over time (iterations), and we have to update our formulas recursively. Also, it is typically the case that our observations are nonstationary. This is particularly true when an update of a value function depends on an approximation of the value function in the future (as occurs with value iteration or any of the TD( λ ) classes of algorithms). When we are estimating parameters from nonstationary data, we do not want to equally weight all observations.

The problem of finding θ can be posed in terms of solving the following stochastic optimization problem:

<!-- formula-not-decoded -->

We can solve this using a stochastic gradient algorithm, which produces updates of the form:

<!-- formula-not-decoded -->

If our value function is linear in R t , we would write:

<!-- formula-not-decoded -->

In this case, our number of parameters has shrunk from the number of possible realizations of the entire vector R t to the size of the attribute space (which, for some problems, can still be large, but nowhere near as large as the original state space). For this problem, φ ( R n ) = R n .

It is not necessarily the case that we will always want to use a linear-in-the-parameters model. We may consider a model where the value increases with the number of resources, but at a declining rate that we do not know. Such a model could be captured with the representation:

<!-- formula-not-decoded -->

where we expect θ 2 &lt; 1 to produce a concave function. Now, our updating formula will look like:

<!-- formula-not-decoded -->

where we assume the exponentiation operator in R θ n 2 is performed componentwise.

We can put this updating strategy in terms of temporal differencing. As before, the temporal difference is given by:

<!-- formula-not-decoded -->

The original parameter updating formula (equation 7.27) when we had one parameter per state now becomes:

<!-- formula-not-decoded -->

It is important to note that in contrast with most of our other applications of stochastic gradients, updating the parameter vector using gradients of the objective function requires mixing the units of θ with the units of the value function. In these applications, the stepsize α n has to also perform a scaling role.

## 11.6 Why does it work?**

## 11.6.1 The projection operation

under construction

This section is taken verbatim from Powell et al. (2004).

Let:

<!-- formula-not-decoded -->

Let us now describe the way the projection v = Π V ( z ) can be calculated. Clearly, v is the solution to the quadratic programming problem

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where, for uniformity, we denote v 0 = B , v M +1 = -B . Associating with (11.14) Lagrange multipliers λ s ≥ 0, s = 0 , 1 , . . . , M , we obtain the necessary and sufficient optimality conditions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If i 1 , . . . , i 2 is a sequence of coordinates such that

<!-- formula-not-decoded -->

then, adding the equations (11.15) from i 1 to i 2 yields

<!-- formula-not-decoded -->

If i 1 = 1, then c is the minimum of the above average and B , and for i 2 = M the maximum of -B and this average has to be taken.

The second useful observation is that v n ∈ V and z n computed by (11.12) differs from v n in just one coordinate. If z n /negationslash∈ V one of two cases must occur: either z n s n -1 &lt; z n s n , or z n s n +1 &gt; z n s n .

If z n s n -1 &lt; z n s n , then we search for the largest 1 &lt; i ≤ s n for which

<!-- formula-not-decoded -->

If such i cannot be found we set i = 1. Then we calculate

<!-- formula-not-decoded -->

and set

<!-- formula-not-decoded -->

We have λ 0 = max(0 , c -B ), and

<!-- formula-not-decoded -->

It is straightforward to verify that the solution found and the above Lagrange multipliers satisfy conditions (11.15)-(11.16).

The procedure in the case when z n s n &lt; z n s n +1 is symmetrical: it is the same procedure applied to the graph of z rotated by π .

## 11.6.2 Proof of convergence of the learning version of the SPAR algorithm

This section provides a detailed proof of the learning version of the SPAR algorithm. The goal of the presentation is not just to prove convergence, but to also demonstrate the proof techniques that are required.

We start from the description and analysis of the basic learning algorithm for a concave piecewise linear function of one variable f : [0 , M ] → I R . We assume that f is linear on the intervals [ s -1 , s ], s = 1 , . . . , M . Let

<!-- formula-not-decoded -->

Let us note that the knowledge of the vector v allows us to reconstruct f ( x ), x ∈ [0 , M ], except for the constant term f (0):

<!-- formula-not-decoded -->

where l is such that l ≤ x &lt; l +1.

The main idea of the algorithm is to recursively update a vector ¯ v n ∈ I R M , n = 0 , 1 , . . . , in order to achieve convergence of ¯ v n to v (in some stochastic sense).

Let us note that by the concavity of f the vector v has decreasing components:

<!-- formula-not-decoded -->

We shall at first assume that there exists a constant B such that

<!-- formula-not-decoded -->

Clearly, the set V of vectors satisfying (11.21)-(11.22) is convex and closed. We shall therefore ensure that all our approximate slopes ¯ v n are elements of V as well. To this end we shall employ the operation of orthogonal projection on V

<!-- formula-not-decoded -->

Let ( Ω, H , I P ) be the probability space under consideration. Let s n be a random variable taking values in { 1 , . . . , M } . Denote by F 0 the σ -algebra generated by ¯ v 0 and, for k = 1 , 2 , . . . , let F n denote the σ -algebra generated by ¯ v 0 , · · · , ¯ v n , s 0 , . . . , s n -1 .

Now, for n = 0 , 1 , · · · , define F s,n = σ (¯ v 0 , · · · , ¯ v n , s 0 , · · · , s n ). Note that F n ⊂ F s,n and that s n is not measurable with respect to F n . We can not avoid this ugly notation in this version of the algorithm, although it will be a clear notation for the optimizing version of the algorithm, as in this last version, s n will be a deterministic function of ¯ v 0 , . . . , ¯ v n , s 0 , . . . , s n -1 .

The SPAR-Exploration algorithm is given in figure 11.5.

STEP 0 Set ¯ v 0 ∈ V, n = 0.

STEP 1 Sample s n ∈ { 1 , . . . , M } .

STEP 2 Observe a real-valued random variable ˆ v n +1 such that

<!-- formula-not-decoded -->

STEP 3 Calculate the vector z n +1 ∈ I R M as follows:

<!-- formula-not-decoded -->

where α n ∈ (0 , 1] and α n is F n -measurable.

## STEP 4 Calculate

<!-- formula-not-decoded -->

increase n by one and go to step 1.

Figure 11.5: Separable, Projective Approximation Routine - Exploration version

We will need a few more definitions and assumptions before we prove the main result of this section.

We denote

<!-- formula-not-decoded -->

Also, let

<!-- formula-not-decoded -->

Then, for n = 0 , 1 , . . . ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

Equality (11.28) is due to the Tower property, while equality (11.29) follows from definition (11.26). Furthermore, (11.30) is given by the F s k measurability of 1 { s n = s } and ¯ v n s . Finally, equalities (11.31), (11.32) and (11.33) are due to assumption (11.24), F n measurability of ¯ v n s and to definition (11.25), respectively.

Thus

<!-- formula-not-decoded -->

In addition to (11.24), we assume that there exists a constant C such that for all n

<!-- formula-not-decoded -->

We also assume that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

.

We say that a sequence of random variables { M n : 0 ≤ n &lt; ∞} is a martingale (submartingale) with respect to the filtration F provided that the sequence { M n } has three basic properties:

- (i) M n is F n -adapted, n = 0 , 1 . . .
- (ii) I E { M n |F n -1 } = ( ≥ ) M n -1 , n = 1 , 2 , . . .
- (iii) I E {| M n |} &lt; ∞ , n = 0 , 1 . . .

Theorem 2 Assume (11.24) and (11.35) -(11.38) . Then Algorithm SPAR-Exploration generates a sequence { ¯ v n } such that ¯ v n → v a.s.

To prove the theorem, we need to use two lemmas. The Euclidean norm is the norm under consideration.

Lemma 11.6.1 Let S 0 = 0 and S m = ∑ m -1 n =0 ( α n ) 2 ‖ g n +1 ‖ 2 , m = 1 , 2 , . . . . Then { S m } is a F -submartingale that converges almost surely to a finite random variable S ∞ .

Proof: The first submartingale property is clearly satisfied. In order to show the second one, note that { S m } is positive and increasing. Thus,

<!-- formula-not-decoded -->

The third property is obtained recalling that S m -S m -1 = ( α m -1 ) 2 ‖ g m ‖ 2 . Hence,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Also, note that

<!-- formula-not-decoded -->

where the equality is obtained from the definition of g m and since ¯ v ∈ V , it is bounded by B , then the first inequality holds. This last displayed inequality together with the fact that α m -1 is F m -1 -measurable yields

<!-- formula-not-decoded -->

We also have that

<!-- formula-not-decoded -->

where the first equality follows from Tower property and the inequality is due to assumption (11.35). Furthermore,

<!-- formula-not-decoded -->

The first two equalities and the inequality follow, respectively, from the Tower property, assumption (11.24) and boundedness of V .

Therefore, there exists a constant C 1 such that

<!-- formula-not-decoded -->

The last inequality together with equation (11.39) yields

<!-- formula-not-decoded -->

Thus, taking the expected valued we obtain

<!-- formula-not-decoded -->

since I E { S 0 } = 0 and by (11.37).

Therefore, as S m is positive, we have checked all three submartingale properties and { S m } is a F -submartingale. Also, since sup m I E { S m } ≤ C 2 &lt; ∞ , by the Submartingale Convergence Theorem (Shiryaev, 1996, page 508), S m a.s. - - → S ∞ , where S ∞ is finite.

Lemma 11.6.2 Let U 0 = 0 and U m = ∑ m -1 n =0 α n 〈 ¯ v n -v, g n +1 -P n (¯ v n -v ) 〉 , m = 1 , 2 , . . . . Then { U m } is a F -martingale that converges almost surely to a finite random variable U ∞ .

Proof: The first property is clearly satisfied. In order to show the second one, note that U m -U m -1 = α m -1 〈 ¯ v m -1 -v, g m -P m -1 (¯ v m -1 -v ) 〉 . Then,

<!-- formula-not-decoded -->

where the second equality is due to the definition of 〈 a, b 〉 and the last one follows from (11.34).

To obtain the third property, recall that

<!-- formula-not-decoded -->

Thus, taking expectations yields

<!-- formula-not-decoded -->

As the second term of the sum is equal to I E { U m -U m -1 |F m -1 } , we know it is zero. Now, let's focus on the last term. We have

<!-- formula-not-decoded -->

where (11.43) is due to the boundedness of V . Hence,

<!-- formula-not-decoded -->

by (11.40) and (11.41), where C 3 is a constant. The previous inequality together with equation (11.42) yields

<!-- formula-not-decoded -->

Thus, taking the expected valued we obtain

<!-- formula-not-decoded -->

since I E { U 2 0 } = 0 and by (11.37).

Therefore, { U m } is bounded in L 2 , and thus bounded in L 1 . This means we have checked all three conditions and { U m } is a F -martingale. Also, the L 2 -Bounded Martingale Convergence Theorem (Shiryaev, 1996, page 510) tells us that U m a.s. - - → U ∞ , where U ∞ &lt; ∞ .

## Proof: [Proof of theorem 2]

Since V is a closed and convex set of I R M , the Projection Theorem (Bertsekas et al., 2003, page 88) tells us that Π V : I R M → V is continuous and nonexpansive, i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as ‖ a -b ‖ 2 = ‖ a ‖ 2 -2 〈 a, b 〉 + ‖ b ‖ 2 . Now, if we add and subtract 2 α n 〈 ¯ v n -v, P n (¯ v n -v ) 〉 we get

<!-- formula-not-decoded -->

as 〈 a, b 〉 + 〈 a, c 〉 = 〈 a, b + c 〉 .

Now, we will prove that {‖ ¯ v n -v ‖} converges almost surely. Let, for n = 0 , 1 , . . . ,

<!-- formula-not-decoded -->

where { S k } and { U k } are defined as in lemmas 11.6.1 and 11.6.2. Note that A n +1 is the last two terms of (11.44). Also consider

<!-- formula-not-decoded -->

Thus,

Clearly, ∑ ∞ m =0 A m +1 = B ∞ = S ∞ -U ∞ &lt; ∞ a.s., as both S ∞ and U ∞ are finite almost surely from lemmas 11.6.1 and 11.6.2.

Hence, it is valid to write

<!-- formula-not-decoded -->

Therefore, inequality (11.44) can be rewritten as

<!-- formula-not-decoded -->

Thus, from the positiveness of the previous inner product term,

<!-- formula-not-decoded -->

We can infer from this inequality, that the sequence defined by

<!-- formula-not-decoded -->

is decreasing. This sequence is also bounded, as V is bounded and ∑ ∞ m =0 A m +1 is finite almost surely. From these two facts (decreasing and bounded), we can conclude that the sequence { D k } converges.

Moreover, as ∑ ∞ m = n A m +1 → 0 as n → ∞ , we can also conclude that the sequence {‖ ¯ v n -v ‖ 2 } or, equivalently, {‖ ¯ v n -v ‖} converges almost surely, since the sum of the limits is the limit of the sums.

We are finally ready to finish our proof. Recall that inequality (11.44) holds for all n . Then,

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Passing to the limits we obtain

<!-- formula-not-decoded -->

as the last sum is finite. Therefore, since {‖ ¯ v n +1 -v ‖ 2 } is convergent, the last inequality tells us

<!-- formula-not-decoded -->

But all the terms of this sum are positive, and from (11.38), at least one diagonal element of P m is strictly positive. Therefore, there must exist a subsequence { ¯ v n k } of { ¯ v n } that converges to v almost surely.

Moreover, as {‖ ¯ v n +1 -v ‖ 2 } is convergent a.s., all its subsequences converge and have the same limit. Thus, since the subsequence {‖ ¯ v n k -v ‖ 2 } converges to zero a.s., as { ¯ v n k } converges to v a.s., the whole sequence {‖ ¯ v n +1 -v ‖ 2 } converges to zero and thus { ¯ v n } converges to v a.s.

It is also possible that at a given point s n ∈ { 1 , . . . , M -1 } we can observe two random variables: ˆ v n +1 satisfying (11.24) and (11.35), and ˆ v n +1 + such that, for all n ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

The algorithm SPAR-Exploration can be easily adapted to this case, too. The only difference is Step 3, where we use both random observations, whenever they are available:

<!-- formula-not-decoded -->

The analysis of this version of the method is similar to the basic case. We define

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

as in the basic case, and for s = 2 , . . . , M ,

<!-- formula-not-decoded -->

where the last equality was obtained following the same reasoning as the basic case. It follows that

<!-- formula-not-decoded -->

Therefore, after replacing the coefficients p n s by

<!-- formula-not-decoded -->

we can reduce this version of the method to the basic case analyzed earlier.

If we have a separable multidimensional problem, the algorithm SPAR-Exploration can be applied component-wise. All the analysis remains true, provided the observation points s n = ( s n 1 , . . . , s n n ) are sampled at random, and for each coordinate i , assumption (11.38) is satisfied.

## 11.7 Bibliographic notes

The proofs of the SPAR algorithm, for both the learning (section 11.6.2) and optimizing (section 12.4.1) versions, were adapted from the original proofs Powell et al. (2004). Our presentation is more complete, and is designed to demonstrate the proof technique. The presentation was prepared by Juliana Nascimento.

Recursive estimation: Ljung &amp; Soderstrom (1983), Young (1984),

Recursive least squares in dynamic programming: A.Nedi¸ c &amp; D.P.Bertsekas (2003), Bradtke &amp; Barto (1996)

SPAR: Powell et al. (2004)

SHAPE: Cheung &amp; Powell (2000)

Leveling algorithm Topaloglu &amp; Powell (2003) CAVE: Godfrey &amp; Powell (2001)

Matrix methods: Golub &amp; Loan (1996)

Convergence: Tsitsiklis and van Roy: Tsitsiklis &amp; Van Roy (1997), Jaakkola et al. (1994)

Lp method and approximate DP: de Farias &amp; Van Roy ((to appear),

Bertazzi et al Bertazzi et al. (2000)

Aggregation:

Rogers et al. (1991), Wright (1994), Bertsekas &amp; Castanon (1989), Schweitzer et al. (1985), Tsitsiklis &amp; Van Roy (1996), Bean et al. (1987), Mendelssohn (1982), Zipkin (1980 b ), Zipkin (1980 a ),

Soft state aggregation Singh et al. (1995)

Iterative methods for continuous problems: Luus (2000)

The SPAR learning proof was adapted from the proof given in Powell et al. (2004), but the presentation is designed to carefully demonstrate mathematical style and proof technique. The presentation was prepared by Juliana Nascimento.

## Exercises

- 11.1) Show that the scaling matrix H n in the recursive updating formula from equation (9.45)

<!-- formula-not-decoded -->

reduces to H n = 1 /n for the case of a single parameter.

## Chapter 12

## The asset acquisition problem

One of the fundamental problems in asset management is the asset acquisition problem, which is where we face the challenge of acquiring assets that are then consumed in a random way (see examples).

Example 12.1: The chief financial officer for a startup company has to raise capital to finance research, development, and business startup expenses to get the company through to its initial public offering. The CFO has to decide how much capital it has to raise initially as well as plan the addition of new capital as needs arise. The consumption of capital depends on the actual flow of expenses, which are partially offset by growing revenues from the new product line.

Example 12.2: A business jet company sells fractional ownership in its jets. It has to order new jets, which can take a year or more to arrive, in order to meet demands as they arise. Requests for jets fluctuate due to a variety of factors, including random noise (the behavior of individuals making decisions) and the stock market.

Example 12.3: A management consulting firm has to decide how many new college graduates to hire annually to staff its needs over the course of the year.

Example 12.4: An energy company has to sign contracts for supplies of coal, oil and natural gas to serve its customers. These decisions have to be made in advance in the presence of fluctuating market prices and the aggregate demand for energy from its customers.

There are two issues that make these problems interesting. One is the presence of nonconvex cost functions, where it costs less, on average, to acquire larger quantities at the same time. This property, which arises in a large number of operational settings, creates an incentive to acquire assets now that can be held in inventory and used to satisfy demands in the future. We will defer this problem class to chapter 13 under the category of batch replenishment processes.

The second complicating issue is the uncertainty in demand for assets and fluctuations in both the cost of purchasing as well as selling assets. This is the problem class we address in this chapter. We restrict our attention to problems where there is only one type of asset. This eliminates, for the moment, the complications that arise with substitution between asset classes (this problem is addressed in chapters 14 and 15). Instead, our focus is on modeling various types of information processes. We start with the basic problem of making the decision to acquire additional assets in one time period to meet demand in the next time period (where the assets are consumed or lost). This simple problem allows us to explore specific approximation strategies that exploit the natural concavity of the value function. The remainder of the chapter extends this core strategy to multiperiod problems where assets can be held, and to problems where orders can be made more cheaply for time periods farther in the future.

This chapter is the first time that we try to solve a problem purely using continuous state and decision variables with continuous value function approximations. In addition, we will create these approximations using gradients rather than estimates of the functions themselves. This work will lay the foundation for much larger and more complex asset management problems.

## 12.1 The single-period problem

The core problem in asset acquisition is one that is famously known as the newsvendor problem (in the old days, this was called the 'newsboy problem'). The story goes something like this: At the beginning of the day, our newsvendor has to decide how many newspapers to put in the newstand at the side of the road. Needless to say, this decision has to be made before she knows the demand for newspapers. At the end of the day, the newspapers are worthless, and remaining papers have to be discarded. It is possible that all the papers are sold out, at which point we face the problem of losing an unknown amount of demand.

Of course, our interest in this problem is much broader than newspapers. For example, a growing business jet company has to sign contracts at the beginning of one year for jets that will arrive at the beginning of the next year. The company hopes that these jets will all be purchased (often in fractions of 1 / 8) before the end of the year. If not, the company faces an overage situation, which means the company is paying for assets that are not generating revenue. The alternative is that demand has exceeded supply, which may be turned away or satisfied by leasing aircraft at a cost that is higher than what the company receives.

In our notation, we would define:

x 0 = The quantity the company orders initially which arrives during time interval 1 (the time between 0 and 1) that can be used to satisfy demands during this time period.

D 1 = The demand for assets that arise during time interval 1.

c p = The unit purchase cost of assets.

p = The price charged to each customer for the asset.

Our contribution function is given by:

<!-- formula-not-decoded -->

This is termed the profit maximizing form of the newsvendor problem. The problem is sometimes stated in a cost-minimizing version. In this form, we define:

c o = The unit cost of ordering too much.

c u = The unit cost of ordering too little.

We would now define the cost function (to be minimized):

<!-- formula-not-decoded -->

where [ x ] + = max { 0 , x } . We are going to use the profit maximizing version in equation (12.1), but our presentation will generally work for a more general model.

We assume (as occurs in the real newsvendor problem) that unused assets have no value. This is the reason it is called the one-period model. Each time period is a new problem. As we show later, solving the one-period model sets the foundation for solving multiperiod problems.

At this point we are not concerned with whether the demands are discrete or continuous, but this will affect our choice of algorithmic strategies. Our starting point for analyzing and solving this problem is the stochastic optimization problem that we first posed in chapter 6, where we addressed the problem of solving:

<!-- formula-not-decoded -->

Clearly, the single-period newsvendor problem fits this basic framework where we face the problem of choosing x = x 0 before we have access to the information W = D 1 . This opens the door to algorithms we have already presented in chapter 6, but we can also exploit the properties of this special problem.

Figure 12.1: The newsvendor profit function for a sample realization (a) and an average over several samples (b).

<!-- image -->

## 12.1.1 Properties and optimality conditions

The first important property of the newsvendor problem is that it is continuous in x (in chapter 13, we looked at problems that are not continuous). If demands are continuous, the objective function will be continuously differentiable. If demands are discrete (but we allow the order quantity to be continuous), the objective function will be piecewise linear.

The second property that we are going to exploit in our solution of the profit maximizing version of the newsvendor problem (equation (12.1)) is concavity. It is pretty easy to see the concavity of the function for a sample realization D 1 ( ω ). Figure 12.1 shows the function (a) for a single realization, and (b) averaged over 10 realizations. For the sample in figure 12.1(a), we see the clear effect of the min operator on the revenue, whereas the costs are linear. When averaged over multiple samples, the plot takes on a smoother shape.

If demands are continuous, then the profit function will be a continuously differentiable function. If demands are discrete and integer (but if we let x 0 be continuous), then the profit function would be piecewise linear with kinks for integer values of x . Later, we exploit this property when we develop approximations for the value function.

The basic newsvendor problem is simple enough that we can derive a nice optimality condition. It is easy to see that the gradient of the function, given a sample realization ω , is given by:

<!-- formula-not-decoded -->

Since the function is not differentiable at x 0 = D 1 ( ω ), we have to arbitrarily break a tie (we could legitimately use a value for the gradient in between -c p and p -c p ). Since our function is unconstrained, we expect the gradient at the optimal solution to equal zero. If our function is piecewise linear (because demands are discrete), we would say that there

Figure 12.2: At optimality, the gradient at the optimal solution will equal zero if demands are continuous (a) or the set of subgradients at the optimal solution will include the zero gradient (b).

<!-- image -->

exists a gradient of the function that is zero at the optimal solution. The two cases are illustrated in figure 12.2.

When we look at the expression for the gradient in equation (12.3), we quickly see that the gradient is never equal to zero. What we want is for the expected value of the gradient to be zero. We can write this easily as:

<!-- formula-not-decoded -->

Solving gives us:

<!-- formula-not-decoded -->

Equation (12.4) is well known in the operations management community as the critical ratio . It tells us that at optimality, the fraction of time we want to satisfy demand is given by ( p -c p ) /p . Obviously we require c p ≤ p . As c p approaches the price, our marginal profit from satisfying demand drops to zero, and we want to cover less and less demand (dropping to zero when c p = p ).

This is an elegant result, although not one that we can actually put into practice. But it helps to see that at optimality, our sample gradient is, in general, never equal to zero. Instead, we have to judge optimality by looking at the average value of the gradient.

## 12.1.2 A stochastic gradient algorithm

In chapter 6, we saw that we could solve stochastic optimization problems of the form:

<!-- formula-not-decoded -->

using iterations of the form:

<!-- formula-not-decoded -->

This is true for unconstrained problems. At a minimum, we will have nonnegativity constraints, and in addition, we may have limits on how much we can order. Constraints like these are fairly easy to handle using a simple projection algorithm. Let Π X be a projection operator that takes a vector from outside of the feasible region to the point in X closest to the original vector. Our algorithm would then look like:

<!-- formula-not-decoded -->

For nonnegativity constraints, the projection operator simply takes any negative elements and makes them zero. Similarly, if we require x ≤ u , then any elements greater than u are simply set equal to u .

Interestingly, this algorithm is convergent even if F ( x, W ) is nondifferentiable. It turns out that our basic conditions on the stepsize required to ensure convergence for stochastic problems are also the conditions needed for nondifferentiable problems. Solving stochastic, nondifferentiable problems (which is what we often encounter with newsvendor problems) does not introduce additional complexities.

Stochastic gradient algorithms can be notoriously unstable. In addition to the use of a declining stepsize, another strategy is to perform smoothing on the gradient itself. That is, we could compute:

<!-- formula-not-decoded -->

where β is a stepsize for smoothing the gradient. We would then update our solution using:

<!-- formula-not-decoded -->

with a projection operation inserted as necessary.

The smoothed gradient ¯ g n is effectively creating a linear approximation of the value function. Since maximizing (or minimizing) a linear approximation can produce extreme results, the stepsize α n plays a critical role in stabilizing the algorithm.

Consider now a minor reformulation of the newsvendor problem:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, we are maximizing a one-period contribution function C 0 ( x 0 ), which captures the cost of purchasing the assets plus the expected value of the future given that we are in R 0 = x 0 . Now, replace the value function with a linear approximation:

<!-- formula-not-decoded -->

For a given value of x n -1 0 , we can find the gradient of the value function given a sample realization of the demand. Let:

<!-- formula-not-decoded -->

Given the sample gradient, we would perform smoothing to approximate an expectation:

<!-- formula-not-decoded -->

Of course, optimizing a linear approximation can produce extreme solutions (in fact, the problem does not even make sense unless we bound x 0 from above and below), but if we take the gradient of (12.9), we can move in the direction of this gradient. Of course, the gradient of (12.9) is the same as our stochastic gradient.

Linear approximations can work well in special situations. Later, we introduce variations to the newsvendor where we already have a supply of assets on hand, but may need to order more. The amount we can order may be small compared to the demand and the amount that we already have on hand. As a result, x 0 may only influence our 'value function' over a small range where a linear approximation is appropriate.

## 12.1.3 Nonlinear approximations for continuous problems

In general, nonlinear approximations are going to work better than linear ones. To produce a nonlinear approximation, we could use any of the techniques for statistically estimating

value function approximations that were introduced in chapter 9. A particularly simple algorithm for problems with continuous demands (and therefore continuously differentiable value functions) is the SHAPE algorithm (section 11.4).

To illustrate, start with the dynamic programming form of the newsvendor problem given in equations (12.5)-(12.6). Now, assume we have an idea of what the value function might look like, although we may not have an exact expression. It is clear, looking at equation (12.8), that the function will initially rise somewhat linearly, but level off at some point. It would be reasonable to guess at a function such as:

<!-- formula-not-decoded -->

Right now, we are going to assume that we have an estimate of the parameter vector θ which would be based on an understanding of the problem. We can then use the SHAPE algorithm to refine our estimate. Using the sample gradient ˆ v n in equation (12.10), we would use the updating equation:

<!-- formula-not-decoded -->

At iteration n , our value function will look like the original value function plus a linear correction term:

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

The SHAPE algorithm will work best when it is possible to exploit the problem structure to produce a 'pretty good' initial approximation. The SHAPE algorithm will then tune this initial approximation, producing a convergent algorithm in certain problem settings (such as the simple newsvendor problem).

An alternative is to use an arbitrary polynomial (for example, -x 2 ). In this case, it is probably more effective to use the more general regression methods described in section 9.4.

## 12.1.4 Piecewise linear approximations

There are several reasons why we may be interested in a piecewise linear approximation. First, our demands may be discrete, in which case the value function will actually be piecewise linear in the resources (if we are allowed to cover a fraction of a demand). Second, we may wish to solve our newsvendor problem using a linear programming package (this is

relevant in more general, multidimensional settings). Third (and often the most important), our underlying demand distribution may not be known, and we may have to allow for the possibility that it has an unusual shape. For example, we may face demands that are usually zero, but may be large. Or, the demands may be bimodal. In the presence of such behaviors, it may be difficult to propose reasonable polynomial approximations that will work well across all the different types of demand behaviors.

We can handle such problems using piecewise linear approximations where we represent our value function in the form:

<!-- formula-not-decoded -->

where /floorleft R /floorright is the largest integer less than or equal to R , and /ceilingleft R /ceilingright is the smallest integer greater than or equal to R .

If our resource vector R is small and discrete, we would have a slope for every integer value of R , which is equivalent to estimating a discrete version of the value function. The form in equation (12.12) has two advantages. First, the range of R may be quite large, and we may prefer to approximate the function using only a few slopes. Second, it shifts the emphasis from approximating the value of having R assets on hand to estimating the marginal value of additional assets. This will prove to be a critical step as we move on to vector-valued problems (multiple asset types).

We assume that at iteration n , we make a decision x n 0 that we obtain by solving:

<!-- formula-not-decoded -->

given x n 0 , we find the sample gradient ˆ v n (equation (12.10)) using R n 0 = x n 0 . For simplicity, we are going to assume that we have an estimate of a slope ¯ v n -1 ( r ) for every value of r (more general applications of this idea have to handle the possibility that the function is piecewise linear over ranges of values of r ). In this case, we would update our slope using:

<!-- formula-not-decoded -->

With this update, we may no longer have the property that ¯ v n ( R n -1) ≥ ¯ v n ( R n ) ≥ ¯ v n ( R n +1), which means that our value function approximation ¯ V n ( R ) is no longer concave. There are two problems with this state of affairs. Theoretically it is unattractive because we know the function is concave. More practically, it makes equation (12.13) much harder to solve (when we make the step to multiple asset types, the problem becomes virtually impossible).

Instead, we can use the same methods described in section 11.3. Instead of estimating a monotone function, we estimate a monotone set of slopes. We would first update our slopes

using:

<!-- formula-not-decoded -->

We then check to see if we have a concavity violation (equivalently, a violation of the monotonicity of the slopes). If y n ( r ) ≥ y n ( r +1) for all r , then our function is still concave, and we would set:

<!-- formula-not-decoded -->

Otherwise, we have a violation, which means that either y n ( r n ) &lt; y n ( r n +1) or y n ( r n -1) &lt; y n ( r n ). We would solve the problem by solving the projection problem:

<!-- formula-not-decoded -->

subject to:

<!-- formula-not-decoded -->

We solve this problem as we did in section 11.3. Assume that our violation is of the form y n ( R n ) &lt; y n ( R n + 1), which means that our updated slope raised the value of y n ( R n ) too much. We want to find the smallest value i &gt; R n such that

<!-- formula-not-decoded -->

We are using gradients to estimate a set of slopes just as we estimated the value function directly in section 11.3.

So far, it seems as if we are doing the same thing here using slopes that we did in section 11.3 using direct estimates of the function. However, there is one critical difference. When we first presented the SPAR algorithm, we presented the learning version, which means we assumed there was some exogenous process that was choosing the states for us. We also assumed that we were going to sample each state with a positive probability, which meant that in the limit, each state would be sampled infinitely often.

For our newsvendor problem, we no longer have this property. Our decisions are driven by equation (12.13), which means our decision x n depends on our value function approximation. Since the states we sample are determined by the solution of an optimization problem, we refer to this as the optimizing version of the algorithm, which is summarized in figure 12.3. In fact, this is the way all of our approximate dynamic programming algorithms work that we have presented up to now. The problem with these algorithms is that in general, not only

Step 0 Initialize ¯ V 0 and set n = 1.

Step 1 Solve:

<!-- formula-not-decoded -->

and set R 0 = x 0 .

Step 2 Sample D 1 ( ω n ) and determine the gradient ˆ v n using (12.10).

Step 3 Calculate the vector y n as follows:

<!-- formula-not-decoded -->

Step 4 Project the updated estimate onto the space of monotone functions:

<!-- formula-not-decoded -->

by solving (11.6)-(11.7). Set n = n +1 and go to Step 1.

Figure 12.3: The optimizing form of the separable, projective approximation routine (SPAR).

is there no proof of optimality, but it is easy to create examples where they work extremely poorly. The problem is that if you get a poor estimate of being in a state, you may avoid decisions that put you in it.

In our case (assuming discrete demands), the algorithm is optimal! The formal proof is given in section 12.4.1. A natural question is: why does it work? In the learning version, we sample each state infinitely often. Since we get unbiased estimates of each state, the only nontrivial aspect of the proof is the introduction of the monotonicity preserving step. But no one would be surprised that the algorithm is convergent.

When we depend on our current value function approximation to tell us what point to sample next, it would seem that a poor approximation would encourage us to sample the wrong points. For example, assume we were estimating a discrete value function, and that at each iteration, we looked at all values of x 0 to choose the best decision. Then, assume that we only update the value that we observe without introducing an intermediate step such as maintaining monotonicity or concavity. It would be easy, due to statistical noise, to get a low estimate of the value of being in the optimal state. We would never visit this state again, and therefore we would never fix our error.

The reason the algorithm works here is because of the concavity property. Assume we can order between 0 and 10 units of our asset, and that our initial approximation produces an optimal solution of 0 when the true optimum is 5. As we repeatedly sample the solution corresponding to x 0 = 0, we eventually learn that the slope is larger than we thought (which means that the optimal solution is to the right). This moves the optimal solution to a higher value. This behavior is illustrated in figure 12.4. The function does not converge to the true function everywhere. It is entirely possible that we may have inaccurate estimates of the function as we move away from the region of optimality. However, concavity bounds these

Figure 12.4: Illustration of the steps of the SPAR update logic of a concave function.

<!-- image -->

errors. We do not need to get all the slopes exactly. We will, however, get the slopes at the optimal solution exactly.

The use of a piecewise linear approximation will, in general, require the estimation of more parameters than would be needed if we characterized the value function using a polynomial that could be fitted using regression methods. The additional parameters are the reason the method will work well even for odd demand distributions, but the importance of this feature will depend on specific applications.

This problem illustrates how we can estimate value functions using a small number of parameters. Furthermore, if we can exploit a property such as concavity, we may even be able to get optimal solutions.

## 12.2 The multiperiod asset acquisition problem

While the newsvendor problem can be important in its own right, there is a vast array of problems that involve acquiring assets over time and where assets that are not used in one time period remain in the next time period. In financial applications, there can also be gains and losses from one time period to the next reflecting changes in prices due to market fluctuations. In this section, we show how the multiperiod asset acquisition problem can be solved using techniques that we have already described.

We present the basic model in section 12.2.1. In order to update the value function approximation, we have to compute approximations of gradients of the value functions. This can be done in two ways. Section 12.2.2 describes the simpler single-pass method. Then, section 12.2.3 describes the two-pass method which, while somewhat more difficult to implement, generally has faster rates of convergence.

## 12.2.1 The model

We start with a basic multiperiod model:

x t = The quantity of assets acquired at time t to be used during time

R t = Assets on hand at the end of time interval t that can be used during time interval t +1.

D t = The demand for assets during time interval t . interval t +1. C t ( R t -1 , x t , D t ) = Contribution earned during time interval t . = p t min { R t -1 , D t } -c p x t

We note that the revenue term, p t min { R t -1 , D t } is a constant at time t whereas in our single-period problem, this term did not even appear (see equation (12.6)).

For our presentation, it is going to be useful to use both the pre- and post- decision state

variables, which produces the following transition equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Writing the optimality equation using the post-decision state variable gives

<!-- formula-not-decoded -->

An effective solution technique is to combine the approximate dynamic programming methods described in chapter 5, the functional approximation methods of section 11 and the ideas suggested in section 12.1. Assume we have designed a functional approximation ¯ V t ( R t ). At iteration n , we would make decisions by solving:

<!-- formula-not-decoded -->

Recall that solving (12.16) represents a form of decision function that we represent by X π t , where the policy is to solve (12.16) using the value function approximation ¯ V n -1 t ( R x t ). Assume, for example, that our approximation is of the form

<!-- formula-not-decoded -->

In this case, (12.16) looks like:

<!-- formula-not-decoded -->

We can find x n t by differentiating (12.17) with respect to x t and setting the result equal to zero:

<!-- formula-not-decoded -->

where we used dR x t /dx t = 1 and R x t = [ R x t -1 -D t ( ω n )] + + x t . Solving for x t given that R x t -1 = R x,n t -1 gives:

<!-- formula-not-decoded -->

If we used a piecewise linear approximation for the value function, we would have to find the slope of the sum of the one-period contribution plus the value function for each value of x t until we found a point where the slope changed from negative to positive.

As we described in chapter 7, there are two methods for updating the value function: a single-pass method where updates are computed as we step forward in time (see figure 7.1), and a double-pass method (outlined in figure 7.2). These discussions were presented in the context of computing the value of being in a state. We now illustrate these calculations in terms of computing the derivative of the function, which is more useful for asset acquisition problems.

## 12.2.2 Computing gradients with a forward pass

The simplest way to compute a gradient is using a pure forward pass implementation. At time t , we have to solve:

<!-- formula-not-decoded -->

Here, ∼ V n t ( R t -1 , D t ( ω n )) is just a placeholder. We next compute a derivative using the finite difference:

<!-- formula-not-decoded -->

Note that when we solve the perturbed value ∼ V n t ( R t -1 +1 , D t ( ω n )) that we have to reoptimize x t . For example, it is entirely possible that if we increase R t -1 by one then x n t will decrease by one. As is always the case with a forward pass algorithm, ˆ v n t depends on ¯ V n t ( R t ), and as a result will typically be biased.

Once we have a sample estimate of the gradient ˆ v n t , we next have to update the value function. We can represent this updating process generically using:

<!-- formula-not-decoded -->

The actual updating process depends on whether we are using a piecewise linear approximation, a general regression equation, or the SHAPE algorithm.

The complete algorithm is outlined in figure 12.5, which is an adaptation of our original single-pass algorithm. A simple but critical conceptual difference is that we are now explicitly assuming that we are using a continuous functional approximation.

## 12.2.3 Computing gradients with a backward pass

Computing the gradient using a backward pass means that we are finding the gradient of the total contribution when we are following a specific policy determined by our current value

Step 0. For all t choose a function form and initial parameters for the value function ¯ V 0 t ( S t ) for all time periods t . Let n = 1.

- Step 1. Choose ω n .
- Step 2: Do, for t = 1 , 2 , . . . , T :

Step 2a. Solve:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2b. Compute:

<!-- formula-not-decoded -->

Step 2c. Update the value function:

<!-- formula-not-decoded -->

Step 2d. Compute R n t = [ R n t -1 -D t ( ω n )] + + x t .

Step 3. Let n = n +1. If n &lt; N , go to step 1.

Figure 12.5: An approximate dynamic programming algorithm for the multiperiod newsvendor problem function approximation. We start by writing the expected total costs from time t onward, given the information up through time t (that is, we condition on the pre-decision state variable R t ), as

<!-- formula-not-decoded -->

We would now like to compute the gradient of F π t ( R t ) with respect to R x t -1 for a specific sample realization ω n . We first write (12.18) for a sample realization ω as follows:

<!-- formula-not-decoded -->

Computing the finite difference for a unit change in R x t -1 , and using

<!-- formula-not-decoded -->

gives

<!-- formula-not-decoded -->

This can be computed recursively. Let:

<!-- formula-not-decoded -->

be the change in contribution if we have one more unit available. Note that this can produce a change in the amount ordered which we represent using

<!-- formula-not-decoded -->

We can now determine the change in R x t due to a unit change in R x t -1 :

<!-- formula-not-decoded -->

This allows us to write

<!-- formula-not-decoded -->

This can be computed recursively starting with t = T (using ˆ v n T +1 = 0) and stepping backward in time. A complete statement of the algorithm is given in figure 12.6.

## 12.3 Lagged information processes

There is a much richer class of problems when we allow ourselves to make orders now for different times into the future. The difference between when we order an asset and when we can use it is called a lag , and problems that exhibit lags between when we know about them and when we can use them are referred to as lagged information processes. Time lags arise in a variety of settings. The examples illustrate this process.

These are all examples of using contracts to make a commitment now (a statement of the information content) for assets that can be used in the future (when they become actionable). The 'contracts' become implicit in other settings. It is possible to purchase components from a supplier in southeast Asia that may require as much as 10 weeks to be built and shipped to North America. A shipload of oil departing from the Middle East may require three weeks to arrive at a port on the Atlantic coast of the United States. A truck departing from Atlanta, Georiga will arrive in California four days later. A plane departing from New York will arrive in London in six hours.

We first provide a basic model of a lagged asset acquisition process. After this, we describe two classes of value function approximations and algorithms: continuously differentiable approximations and piecewise linear, nondifferentiable approximations.

- Step 0. For all t choose a function form and initial parameters for the value function ¯ V 0 t ( S t ) for all time periods t . Let n = 1.

Step 1. Choose ω n .

- Step 2: Do, for t = 1 , 2 , . . . , T :

Step 2a. Solve:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2b. Compute R n t = [ R n t -1 -D t ( ω n )] + + x t .

Step 3: Do, for t = T, T -1 , . . . , 1

Step 3a. Compute:

<!-- formula-not-decoded -->

Step 3b. Compute:

<!-- formula-not-decoded -->

Step 3c. Update the value function:

<!-- formula-not-decoded -->

Step 4. Let n = n +1. If n &lt; N , go to step 1.

Figure 12.6: A two-pass algorithm for the multiperiod newsvendor problem

Example 12.5: An orange juice products company can purchase futures contracts that allow it to buy product in later years at lower prices.

Example 12.6: As a result of limited shipping capacity, shippers have to purchase space on container ships months or years in the future.

Example 12.7: Airlines have to sign contracts to purchase aircraft from manufacturers in the future as a result of manufacturing bottlenecks.

Example 12.8: Electric power companies have to purchase expensive equipment such as transformers for delivery 18 months in the future, but may pay a higher rate to get delivery 12 months in the future.

Example 12.9: Large energy consumers will purchase energy (electricity, oil and gas) in the future to lock in lower prices.

## 12.3.1 Modeling lagged information processes

We begin by writing the decisions we have to make using x tt ′ = The quantity of assets ordered at time t (more specifically, with the information up through time t ), to be used in time period t ′ .

The double time indices have specific roles. The first ' t ' represents the information content (the knowable time ) while the second ' t ′ ' represents when the quantity in the variable can be used (the actionable time ). Since the decision is made with the information that became available during time interval t , the decision has to be made at the end of the time interval. By contrast, if the product arrives during time interval t ′ , we treat is as if it can be used during time interval t ′ , which is as if it arrives at the beginning of the time period. The reader may wish to review the discussion of modeling time in section 3.2.

The single period and multiperiod newsvendor problems which we have considered up to now used a variable x t that represented the decision made at time t (literally, with the information up through time t ) for assets that could be used in time interval t + 1. We would now represent this variable as x t,t +1 . If we always order for the next time period, the double time index is a bit clumsy. With our more general model, we face a much richer set of decisions.

We adopt the convention that:

<!-- formula-not-decoded -->

which means that x t is now a vector. We note that our notation provides for the element x tt which is, literally, the assets ordered after we see the demand. This is known as purchasing on the spot market. By contrast, we would term x tt ′ for t ′ &gt; t as purchasing futures . Depending on the setting, we might pay for 'futures' when either we purchase them or when we receive them.

Once we introduce lags, it is useful to apply this concept throughout the problem. For example, if we are able to order assets that can be used in the future, then at any time t , there will be assets that we can use now and others we know about which we cannot use until the future. We represent this state of affairs using:

R tt ′ = The assets that we know about at time t that can be used in time interval t ′ .

<!-- formula-not-decoded -->

Time lags can also apply to costs and prices. For example, we could define:

c tt ′ = The cost of assets purchased at time t that can be used at time t ′ . c t = ( c tt ′ ) t ′ ≥ t

We might even know about future demands. For this we would define:

D tt ′ = Demands that become known during time interval t that need to be met during time interval t ′ .

<!-- formula-not-decoded -->

But for now, we are going to assume that demands have to be served as they arise, but that we can make decisions now for assets that can be used at a deterministic time in the future.

Noting that R t is now a vector, our transition function R M ( R t -1 , W t , x t ) must be a similarly dimensioned vector-valued function. We would write:

<!-- formula-not-decoded -->

We distinguish two important cases: τ = 1, and τ &gt; 1. These are given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can now write our transition equation as we did before (using R t = R M ( R t -1 , D t , x t )), recognizing that both sides are vectors.

Assume we pay for assets when they are first purchased (rather than when they are actionable). In this case, our single period contribution function becomes:

<!-- formula-not-decoded -->

Our decision function:

<!-- formula-not-decoded -->

is identical to what we used for the multiperiod case without lags (equation (12.16)) with the exception that x t and R t are now vectors. This is a nice example of how notation fails to communicate notational complexity. When we have a vector, it is important to estimate how many dimensions it will have. For the problem of lagged information processes, R t would be a vector with anywhere from a half dozen (signing futures contracts up to six years into the future) to several dozen (the number of days in advance of a shipment from one part of the world to another) or more. But it is unlikely that this issue would produce vectors with 100 or more dimensions. So, we have multiple dimensions, but not an extremely large number of dimensions (this comes later). However, as we have learned, even 10 or 20 dimensions can really explode a problem.

## 12.3.2 Algorithms and approximations for continuously differentiable problems

We first consider algorithms that are designed for continuously differentiable, concave functions. These are the 'nicest' class of functions and enjoy the vast array of algorithms from nonlinear programming. These algorithms can be divided into two broad classes: those that depend on linear approximations of the objective function (gradient algorithms) and those that depend on exact or approximate estimates of the Hessian (matrix of second derivatives) of the objective function (known as Newton algorithms).

Let:

<!-- formula-not-decoded -->

be the function we are trying to maximize. C t is linear in x t . We assume that ¯ V t ( R t ) is continuously differentiable in R t . Then, F t ( x t ) is continuously differentiable in x t as long as R M ( R t -1 , D t , x t ) is continuously differentiable in x t . As they are written, the transition equations are not differentiable with respect to the spot purchases x tt since we may purchase more than we need which allows the purchases to spill over to the next time period (see equation (12.20)). The point at which too many spot purchases spill to the next time period is what creates the nondifferentiability. This problem is handled by splitting the spot purchases into two components:

x now tt = The amount of the spot purchases that must be used now.

x future tt = The amount of the spot purchases that must be used in the future.

<!-- formula-not-decoded -->

We can now rewrite equation (12.20) as:

<!-- formula-not-decoded -->

where x now tt is constrained by:

<!-- formula-not-decoded -->

where s now tt is a slack variable. In practice, spot prices are typically the most expensive, and we would never want to overbuy. For all practical purposes, we can treat x tt = x now tt and assume x future tt = 0. We also observe that [ R t -1 ,t + x now tt -D t ] + = 0 which means that we do not even have to worry about finding the optimal value of x tt .

Because ¯ V t ( R M ( R t -1 , x t )) = ¯ V t ( R t ) is concave and continuously differentiable in R t , and since R t is a linear function of x t , then F t ( x t ) is concave and continuously differentiable in

x t . We typically have to solve these problems subject to a set of constraints. These may be nothing more than nonnegativity constraints ( x t ≥ 0), or we may also have to deal with limits on how much we are able to purchase in a time period ( ∑ t ′ ≥ t x tt ′ ≤ u t or ∑ t ′′ ≤ t x t ′′ t ≤ u t ). We may have to recognize limits on total purchases. This gives us a generalized upper bounding constraint (also called GUB constraints) of the form ∑ t ′ ≥ t x tt ′ ≤ U t . We can represent these constraints generally as x t ∈ X t , but it is important to recognize that constraints of this form are especially easy to handle. Thus, our problem is to solve:

<!-- formula-not-decoded -->

This might be quite easy to solve. Assume, for example, that our value function approximation is separable in x tt ′ . We might have chosen to use:

<!-- formula-not-decoded -->

Substituting (12.25) and (12.22) into (12.24) gives

<!-- formula-not-decoded -->

(12.26) is separable in x tt ′ , make it possible to simply take the derivative and set it equal to zero:

<!-- formula-not-decoded -->

which gives

<!-- formula-not-decoded -->

From equation (12.21) we have

<!-- formula-not-decoded -->

Since x tt ′ ≥ 0, our solution is

<!-- formula-not-decoded -->

Separable approximations are especially nice to work with, although the errors introduced need to be quantified. Chapter 15 demonstrates that these can work quite well on much more complex asset allocation problems. Separable approximations are also especially easy to estimate. If we use more complex, nonseparable approximations, it will be necessary to design algorithms that handle the structure of the approximation used.

## 12.3.3 Algorithms and approximations for nondifferentiable problems

It is possible that the value function is nondifferentiable, as would happen if our demands were discrete. In chapter 14, we consider a very elegant set of strategies known as Benders decomposition for handling this problem class. These algorithms are beyond the scope of our discussion here, so instead, we might suggest using a separable, piecewise linear approximation for the value function. We already developed a piecewise linear approximation in section 12.1.4. We can use this same approximation here by further approximating the value function as separable:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

¯ V n -1 tt ′ ( R tt ′ ) is a piecewise linear function, where ¯ v n -1 tt ′ ( r ) is the slope between r and r +1. We would estimate this function just as we did in section 12.1.4. If ¯ V n -1 t ( R t ) is piecewise linear, then it is nondifferentiable and we have to move to a different class of functions. We can still use our gradient projection algorithm, but we have to use the same types of stepsize rules that we used when we were solving problems using stochastic gradients. These rules require the conditions ∑ ∞ n =1 α n = ∞ and ∑ ∞ n =1 ( α n ) 2 &lt; ∞ , which pushes us into rules of the general class α n = a/ ( b + n ) (see chapter 6).

Amuch better approach for this problem class is to solve our problem as a linear program. To do this, we need to use a common device for representing concave, piecewise linear functions. Instead of writing the nonlinear function ¯ V ( R ), we introduce variables y ( r ) where R = ∑ R max r =0 y ( r ) and R max is the largest possible value of R . We require that 0 ≤ y ( r ) ≤ 1. We can then write our function using:

<!-- formula-not-decoded -->

When we solve our linear program, it would be very odd if y (0) = 1 , y (1) = 0 , y (2) = 1 , y (3) = 1 , y (4) = 0. We could try to write a constraint to make sure that if y ( r +1) &gt; 0 then y ( r ) = 1, but we do not have to. We will assume that ¯ v (0) ≥ ¯ v (1) ≥ . . . ≥ ¯ v ( r ). This means that we always want to maximize y ( r ) before allowing y ( r +1) to increase. Our only problem arises when ¯ v ( r ) = ¯ v ( r + 1). We could handle this by defining intervals that are more than one unit (in practice, we do not use intervals of unit length anyway). But even if we did not, if ¯ v ( r ) = ¯ v ( r +1), then it does not matter if y ( r +1) &gt; y ( r ).

Using this device, we write our linear program as:

<!-- formula-not-decoded -->

subject to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equations (12.30) and (12.31) constrain the y ( r ) variables to be between 0 and 1. Equation (12.32) restricts the amount of spot purchases to be made in time period t so that there is only enough to cover the demand for time t , where s tt is a slack variable. Equation (12.33) defines the actionable assets during time interval t ′ , which we know about at time t , to be what we purchase at time t that is actionable during time interval t ′ plus what we purchased for time interval t ′ , which we knew about at time t -1. Equation (12.34) sets up the relationship between the y tt ′ ( r ) variables and the total flow R tt ′ . Equation (12.35) is the usual nonnegativity constraint, and equations (12.36) and (12.37) introduce possible upper bounds on the amount we can order. Note that if we did not have (12.37), the problem would decompose in the variables ( x tt ′ ) t ′ ≥ t .

This formulation uses a well-known device for modeling piecewise linear, concave functions. We also get something for free from our LP solver. Equations (12.32) and (12.33) represent the equations that capture the impact of the assets from the previous time period, R t -1 , on this problem. These constraints yield the dual variables ˆ v tt ′ , t ′ ≥ t . Just as we

have used them earlier, the duals ˆ v tt ′ represent gradients of our optimization problem with respect to R t -1 ,t ′ . However, since our problem is not differentiable, these are subgradients, which is to say they satisfy:

<!-- formula-not-decoded -->

Instead of a subgradient (which we get for free from the LP package), we could use numerical derivatives to obtain, say, the true right derivative:

<!-- formula-not-decoded -->

where e tt ′ is a suitably dimensioned vector of zeroes with a 1 in the element corresponding to ( tt ′ ). Finding numerical derivatives in this setting is not very expensive. The big expense is solving the linear program (although for these small problems it is quite easy - we consider much larger problems later). Once we have solved a linear program once, perturbing the right hand side and resolving is very fast, but obviously not as fast as using a dual variable (which is free). However it can, for some problems, improve the rate of convergence.

## 12.4 Why does it work?**

## 12.4.1 Proof of convergence of the optimizing version of the SPAR algorithm

As with the proof of the learning version of the SPAR algorithm (section 11.6.2), this section is designed partly to prove the convergence of the optimizing version of the SPAR algorithm, and partly to demonstrate the mathematical techniques required by the proof. The challenge in the optimizing version of the algorithm is that we no longer assume that we are going to visit each point along the curve infinitely often. The points that we visit are determined by the approximation, so we have to show that we end up visiting the optimal points infinitely often. For this reason, the steps of the proof are provided in a much greater level of detail than would normally be used in most publications.

In the optimizing version of SPAR, the observation points s n , n = 0 , 1 , . . . are generated by solving the approximate problem

<!-- formula-not-decoded -->

where each ¯ f n i , i = 1 , . . . , n is a concave, piecewise linear function, defined as (11.20) and X is a convex and closed set such that X ⊂ { x ∈ I R n : 0 ≤ x i ≤ M i , i = 1 , . . . , n } . Thus, ¯ F n is a concave, piecewise linear and separable function, x n i ∈ { 1 , . . . , M i } and we can set s n = ( x n 1 , . . . , x n n ).

Note that s n i is now measurable with respect to F n . Also note that assumption (11.38) applied component-wise may not be satisfied. Even though, since (11.48) holds for each coordinate i , inequality (11.44) is true for each coordinate i :

<!-- formula-not-decoded -->

The matrix P n i , which is F n -measurable, is a positive diagonal matrix with entries s strictly positive if and only if the s coordinate of ¯ v i has a chance of being updated in the current iteration of the algorithm. Proceeding exactly as in the proof of Theorem 2, we conclude that the series ∑ ∞ n =0 A i,n +1 is convergent a.s. Furthermore, the sequence {‖ ¯ v n i -v i ‖} is convergent a.s., for every i = 1 , . . . , n .

Thus, the SPAR-Optimization and the SPAR-Exploration version only differ by the way s n are obtained. Our aim is to prove that even without assumption (11.38), the sequence { s n } generated by SPAR-Optimization converges to an optimal solution of

<!-- formula-not-decoded -->

provided a certain stability condition is satisfied.

Before we state our theorem, we should recall some definitions and remarks.

1. The subdifferential of a function h at x , denoted by ∂h ( x ), is the set of all subgradients of g at x .
2. The normal cone to a convex set C ⊂ I R n at x , denoted by N C ( x ), is the set N C ( x ) = { d ∈ I R n : d T ( y -x ) ≤ 0 , y ∈ C } .
3. (Bertsekas et al., 2003, page 257) Let h : I R n → I R be a concave function. A vector x ∗ maximizes h over a convex set C ⊂ I R n if and only if

<!-- formula-not-decoded -->

4. An optimal point is called stable if it satisfies

<!-- formula-not-decoded -->

5. A stable point is also a solution to a perturbed problem

<!-- formula-not-decoded -->

provided that dist( ∂h ( x ∗ ) , ∂ ˜ h ( x ∗ )) &lt; /epsilon1 and /epsilon1 is a sufficiently small positive number.

6. Every closed and bounded set is compact. Every sequence in a compact set has a subsequence that converges to some point of the set. This point is called an accumulation point .

Applying these concepts to our problem, we have that the subdifferential of ¯ F n at an integer point x n is given by

<!-- formula-not-decoded -->

Furthermore, as V n × X is closed and bounded, hence compact, the sequence (¯ v n , x n ) ∈ V n × X generated by the algorithm SPAR-Optimization has accumulation points (¯ v ∗ , x ∗ ). Also, as ¯ F n is concave and X is convex, the solution x n of (12.38), as it is optimal, satisfies 0 ∈ ∂ ¯ F n ( x n ) -N X ( x n ). Then, by passing to the limit, we can conclude that each accumulation point (¯ v ∗ , x ∗ ) of the sequence { (¯ v n , x n ) } satisfies the condition

<!-- formula-not-decoded -->

We will also need the following lemma:

Lemma 12.4.1 Assume that for each i = 1 , . . . , n the conditions (11.24), (11.35)-(11.37) and (11.45)-(11.46) are satisfied. Define R 0 = 0 and R m = ∑ m -1 j =0 α j ( ‖ g j +1 i ‖-I E {‖ g j +1 i ‖|F j } ) , m = 1 , 2 , . . . . Then { R m } is a F -martingale that converges to a finite random variable R ∞ almost surely.

Proof: The first martingale property is easily verified. In order to get the second one, note that R m -R m -1 = α m -1 ( ‖ g m i ‖ -I E {‖ g m i ‖|F m -1 } ). Then,

<!-- formula-not-decoded -->

To obtain the third property, recall that

<!-- formula-not-decoded -->

Thus, taking expectations yields

<!-- formula-not-decoded -->

Also, note that

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

and since (11.24),(11.35),(11.45) and (11.46) hold, following the same steps as in (11.40) and (11.41) for the last two expectations, we know that there exists a constant C 5 such that

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Finally, taking the expected value we obtain

<!-- formula-not-decoded -->

since I E { R 2 0 } &lt; ∞ and by (11.37).

<!-- formula-not-decoded -->

Therefore, { R m } is bounded in L 2 , and thus bounded in L 1 . This means we have checked all three martingale properties and { R m } is a F -martingale. Also, the L 2 -Bounded Martingale Convergence Theorem tells us that R m a.s. - - → R ∞ , where R ∞ &lt; ∞ .

Now we are ready to state and prove the main result of this section.

Theorem 3 Assume that for each i = 1 , . . . , n the conditions (11.24), (11.35)-(11.37) and (11.45)-(11.46) are satisfied. If an accumulation point (¯ v ∗ , x ∗ ) of the sequence { (¯ v n , x n ) } generated by the algorithm, satisfies the stability condition:

<!-- formula-not-decoded -->

then, with probability one x ∗ , is an optimal solution of (12.40) .

Proof: Let us fix ω ∈ Ω and consider a convergent subsequence { (¯ v n ( ω ) , x n ( ω )) } , along N ( ω ) ⊆ I N . Let us denote by (¯ v ∗ , x ∗ ) the limit of this subsequence. This limit depends on ω too, but we shall omit the argument ω to simplify notation.

From the stability condition, there exists /epsilon1 &gt; 0 such that for all iterations n for which

<!-- formula-not-decoded -->

the solution x n of the approximate problem (12.38) is equal to x ∗ , by remark 5.

Then, the coefficients p n i,s are equal to 1 for s = x ∗ i and s = x ∗ i +1, and are zero otherwise, for each i , as x ∗ i and x ∗ i + 1 are the coordinates of ¯ v i that will be updated in the current iteration. Thus, for a fixed i , if we focus our attention on the points s = x ∗ i :

<!-- formula-not-decoded -->

Thus, the last inequality together with (12.39) yields

<!-- formula-not-decoded -->

Let n ∈ N ( ω ) be large enough so that | ¯ v n i,x ∗ i ( ω ) -¯ v ∗ i,x ∗ i | &lt; /epsilon1/ 2. This n exists as ¯ v n converges to ¯ v ∗ . Consider j ≥ n such that

<!-- formula-not-decoded -->

Let us suppose that the x ∗ i th coordinate of the accumulation point is not optimal, i.e.,

/negationslash

<!-- formula-not-decoded -->

We shall prove that it leads to a contradiction.

We will again divide the rest of the proof in several parts. Part 1 shows that the set of consecutive j ≥ n for which condition (12.47) holds is finite.

Part 1: From assumption (12.48), we can always choose a sufficiently small /epsilon1 &gt; 0 such that | ¯ v ∗ i,x ∗ i -v i,x ∗ i | &gt; 2 /epsilon1 . Then, for the iterations j satisfying (12.47), we have

<!-- formula-not-decoded -->

Combining the previous inequality with the fact that inequality (12.46) holds true yields

<!-- formula-not-decoded -->

If the set of consecutive j ≥ n for which condition (12.47) holds was infinite, then the previous inequality holds for all j ≥ n and we can write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is a contradiction. Therefore, for all n ∈ N ( ω ) and all sufficiently small /epsilon1 &gt; 0, the random index

<!-- formula-not-decoded -->

is finite.

We shall prove that the sum of stepsizes between n ∈ N ( ω ) and l ( n, /epsilon1,ω ) -1 is at least of order /epsilon1 , if n is large enough. We will accomplish this in the following part.

Part 2: By the definition of l ( n, /epsilon1,ω ) we have, | ¯ v l ( n,/epsilon1,ω ) i,x ∗ i ( ω ) -¯ v ∗ i,x ∗ i | &gt; /epsilon1 , for some i . Thus,

<!-- formula-not-decoded -->

Since ¯ v n ( ω ) → ¯ v ∗ , n ∈ N ( ω ), for all sufficiently large n ∈ N ( ω ), we also have | ¯ v n i,x ∗ i ( ω ) -¯ v ∗ i,x ∗ i | &lt; /epsilon1/ 2. Hence,

<!-- formula-not-decoded -->

Moreover, from the projection theorem, we obtain

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

From lemma 12.4.1, unless ω is in a certain null set, the sequence { R m ( ω ) } is convergent, and hence for sufficiently large n ∈ N ( ω ), from Cauchy Criterion,

<!-- formula-not-decoded -->

This together with (12.51) yields

<!-- formula-not-decoded -->

Therefore /epsilon1/ 3 ≤ ∑ l ( n,/epsilon1,ω ) -1 j = n α j ( ω ) I E {‖ g j +1 i ‖|F j } ( ω ).

From assumption (11.35) it follows that there exists a constant C such that I E {‖ g j +1 i ‖|F j } ≤ C for all i and j . Using this fact in the last displayed inequality we obtain

<!-- formula-not-decoded -->

for all sufficiently small /epsilon1 &gt; 0 and all sufficiently large n ∈ N ( ω ).

In the next part, we will finally obtain that (12.48) leads to a contradiction.

Part 3: Inequality (12.49) holds for all j such that l ( n, /epsilon1,ω ) &gt; j ≥ n . Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Letting η = 1 / 3 C and combining (12.52) with the previous inequality we obtain

<!-- formula-not-decoded -->

Let ∆( ω ) be the limit of the the entire sequence {‖ ¯ v j i ( ω ) -v i ‖ 2 } , as j → ∞ , whose existence has been established in the begining of this subsection. Passing the previous inequality to the limit, with n →∞ , n ∈ N ( ω ), and using the fact that ∑ ∞ j = n A ij ( ω ) → 0, as n →∞ , we obtain

<!-- formula-not-decoded -->

a contradiction. Therefore our assumption (12.48) must be false, and we have

<!-- formula-not-decoded -->

The proof can now be finished by observing that the inequality (12.46) is also true with x n i replaced by x n i + 1 (if x n i &lt; M i ). We can thus apply the same argument as before to prove

<!-- formula-not-decoded -->

For x ∗ i = M i we take the convention that ¯ v ∗ i,x ∗ i +1 = v i,x ∗ i +1 = -∞ . Consequently,

<!-- formula-not-decoded -->

and the point x ∗ is optimal for (12.40).

## 12.5 Bibliographic references

## Exercises

- 12.1) In a flexible spending account (FSA), a family is allowed to allocate x pretax dollars to an escrow account maintained by the employer. These funds can be used for medical expenses in the following year. Funds remaining in the account at the end of the following year revert back to the employer (dont you just love that phrase?). Assume that you are in a 40 percent tax bracket (sounds nice, and the arithmetic is a bit easier).

Let S be the random variable representing total medical expenses in the upcoming year, and let F ( s ) = Prob [ S ≤ s ] be the cumulative distribution function of the random variable S .

- a) Write out the objective function that we would want to solve in order to find x to minimize the total cost (in pretax dollars) of covering your medical expenses next year.
- b) If x ∗ is the optimal solution and g ( x ) is the gradient of your objective function if you allocate x to the FSA, use the property that g ( x ∗ ) = 0 to derive (you must show the derivation) the critical ratio that gives the relationship between and the cumulative distribution function .
- c) Given your 40 percent tax bracket, what percentage of the time should you have funds left over at the end of the year?

- 12.2) Consider an asset acquisition problem where you purchase x tt ′ at time t to arrive at time t ′ . x tt represents purchases on the spot market. You pay for the assets at the time of acquisition an amount c tt ′ . Let R tt ′ be the post-decision state variable giving the assets that you know about at time t that can be used during time interval t ′ for t ′ ≥ t . The problem is defined over a finite horizon t = 1 , 2 , . . . , T with t &lt; t ′ ≤ T . You are going to choose x t = ( x tt ′ ) t ′ &gt;t by solving the approximation:

<!-- formula-not-decoded -->

where ¯ V n -1 t ( R t ) is an approximation of the value function. Recalling that ¯ V n -1 t defines a policy that we designate by π n , let

<!-- formula-not-decoded -->

be the value of the policy for sample path ω n given R t -1 and D t ( ω n ).

- a) Show how to compute a gradient ˆ v n tt ′ of F π t ( R t -1 , D t ( ω n ) , ω n ) with respect to R t -1 ,t ′ , given D t ( ω n ). Your method should not depend on any functional form for ¯ V n -1 t ( R t ).
- b) Assume now that ¯ V t ( R t ) = ∑ t ′ &gt;t ¯ v tt ′ R tt ′ . Show that the gradient ˆ v n tt ′ = ˆ v n t ′ , which is to say that the gradient depends only on the actionable time t ′ .

## Chapter 13

## Batch replenishment processes

We often encounter processes in which the supply of an asset is consumed over time (usually in random increments) and followed by replenishments to restore the supply. The need to periodically replenish is driven by a bit of economics unique to this problem class - we assume that the cost of acquiring additional assets is concave with respect to the quantity. In this chapter, we focus on problems where the concavity arises because we have to pay a fixed cost to place an order, after which we pay a fixed unit cost. However, most of the work in this chapter will apply to general concave functions which might reflect discounts for ordering larger amounts.

Replenishment processes come in two flavors: negative drift (assets are consumed by an exogenous demand) with positive replenishment and positive drift (assets arrive according to an exogenous process) followed by batch depletion (see examples).

We use the term 'replenishment' for problems where there is a negative drift (an exogenous process that consumes assets) that requires periodic replenishments, but the same basic dynamics arise when there is a positive drift (an exogenous process governing the accumulation of assets) that requires a periodic clearing of assets. The problems are not equivalent (that is, one is not simply the negative of the other), but, from a computational perspective, the issues are quite similar.

## 13.1 A positive accumulation problem

Positive accumulation problems arise when resources arrive to a system as a result of an exogenous process. Examples include customers returning cars to a car rental agency who then wait for a bus to return to the airport; money generated from dividend income in a stock fund that has to be reinvested; shipments arriving at a loading dock waiting for a truck to move them to another location; and complex equipment (jet engines, electric power transformers) that accumulates particles in the oil indicating that the component is degenerating and may require maintenance or repair.

Example 13.1: A software company continually updates its software product. From time to time, it ships a new release to its customer base. Most of the costs of shipping an update, which include preparing announcements, posting software on the internet, printing new CD's, and preparing manuals summarizing the new features, are relatively independent of how many changes have been made to the software.

Example 13.2: E-Z Pass is an automated toll collection system. Users provide a credit card, and the system deducts $25 to provide an initial account balance. This balance is reduced each time the traveler passes through one of the automated toll booths. When the balance goes below a minimum level, another $25 is charged to the credit card to restore the balance.

Example 13.3: Shipments accumulate at a freight dock where they are loaded onto an outbound truck. Periodically, the truck fills and it is dispatched, but sometimes it is dispatched before it is full to avoid excessive service penalties for holding the shipments.

Example 13.4: An oil company monitors its total oil reserves, which are constantly drawn down. Periodically, it acquires new reserves either through exploration or by purchasing known reserves owned by another company.

We use the context of customers (people, freight) arriving to a queue that have to be moved by a vehicle of finite capacity.

## 13.1.1 The model

Our model uses the following parameters:

c r = The fixed cost of a replenishment.

c h = Penalty per time period for holding a unit of freight.

K = Maximum size of a batch.

Our exogenous information process consists of

A t = Quantity of new arrivals during time interval t .

Our (post-decision) state variable is

R t = Assets remaining at the end of time period t .

There are two decisions we have to make. The first is whether to dispatch a vehicle, and the second is how many customers to put on the vehicle. For this problem, once we make the decision to dispatch a vehicle, we are going to put as many customers as we can onto the

vehicle, so the 'decision' of how many customers to put onto the vehicle seems unnecessary. It becomes more important when we consider multiple customer types later in the chapter. For consistency with this section, we define:

<!-- formula-not-decoded -->

x t = The number of customers to put on the truck.

In theory, we might be able to put a large number of customers on the vehicle, but we may face a nonlinear cost that makes this suboptimal. For the moment, we are going to assume that we always want to put as many as we can, so we set:

<!-- formula-not-decoded -->

The transition function is described using

<!-- formula-not-decoded -->

The objective function is modeled using:

<!-- formula-not-decoded -->

Our problem is to find the policy X π t ( R t ) that solves:

<!-- formula-not-decoded -->

If we are managing a single asset class, then R t and x t are scalars and the problem can be solved using standard backward dynamic programming techniques of the sort that were presented in chapter 4 (assuming that we have a probability model for the demand). In practice, many problems involve multiple asset classes, which makes standard techniques impractical. But we can use this simple problem to study the structure of the problem.

## 13.1.2 Properties of the value function

Recognizing that R t is a post-decision state variable, the value function V t ( R t ) is given by the standard equation:

<!-- formula-not-decoded -->

<!-- image -->

Number of customers at time 0

Figure 13.1: Shape of the value function for the positive-drift batch replenishment problem, from Papadaki &amp; Powell (2003).

The value function is illustrated in figure 13.1. It turns out it has a few nice properties which can be proven. The first is that it increases monotonically (rather, it never decreases). The second is that it is concave over the range R ∈ ( nK, ( n +1)( K ) -1) for n = 0 , 1 , . . . . The third is that the function is K -convex, which means that it satisfies:

<!-- formula-not-decoded -->

For example, if we have a vehicle that can take 20 customers, K -convexity means that V (25) -V (5) ≤ V (35) -V (15). The value function is convex when measured on a lattice K units apart.

We can exploit these properties in the design of an approximation strategy.

## 13.1.3 Approximating the value function

While finding an optimal policy is nice, we are more interested in obtaining good quality solutions using methods that are scalable to more complex problems. The most important property of the function is that it rises monotonically, suggesting that a linear approximation is likely to work reasonably well. This means that our value function approximation will look like:

<!-- formula-not-decoded -->

If we replace the expectation in (13.4) and use the linear approximation we obtain:

<!-- formula-not-decoded -->

Solving this is quite easy, since we only have to try two values of z t . We can get an estimate of the slope of the function using:

<!-- formula-not-decoded -->

which we then smooth to obtain our estimate of ¯ v n t -1 :

<!-- formula-not-decoded -->

Deterministic

Another strategy is to recognize that the most important part of the curve corresponds to values 0 ≤ R ≤ K , over which the function is concave. We can use the techniques of chapter 11 to produce a concave approximation which would more accurately capture the function in this region. We can compare these approximations to the optimal solution for the scalar case, since this is one of the few problems where we can obtain an optimal solution. The results are shown in figure 13.2 for both the linear and nonlinear (concave) approximations. Note that the linear case works better with fewer iterations. It is easier to estimate a single slope rather than an entire piecewise linear function. As we run more iterations, the nonlinear function works better. For some large scale problems, it may be impossible to run hundreds of iterations (or even dozens). For such applications, a linear approximation is best. 0 50 100 150 200 250 0 0.05 0.1 0.15 0.2 iterations percentage error from optimal concave linear

Stochastic

300

Figure 13.2: Percent error produced by linear and nonlinear approximations as a function of the training iterations, from Papadaki &amp; Powell (2003).

<!-- image -->

## 13.1.4 Solving a multiclass problem using linear approximations

Now assume that we have different types of assets arriving to our queue. Using our standard notation, let R ta be the quantity of assets of type a at time t (we continue to use a postdecision state variable). We are going to assume that our attribute is not too big (dozens, hundreds, but not thousands).

For this problem, we still have a scalar decision variable z t that indicates whether we are dispatching the vehicle or not. However now we have a nontrivial problem of determining how many customers of each type to put on the vehicle. Let:

R ta = The number of customers of type a at the end of period t .

A ta = The number of arrivals of customeres of type a during time in- terval t .

x ta = The number of customers of type a that we can put on the vehicle.

We let R t , A t and x t be the corresponding vectors over all the customer types. We require that ∑ a ∈A x ta ≤ K . The transition equation is now given by

<!-- formula-not-decoded -->

For this problem, nonlinear value functions become computationally much more difficult. Our linear value function now looks like:

<!-- formula-not-decoded -->

which means that we are solving, at each point in time:

<!-- formula-not-decoded -->

This is characterized by the parameters (¯ v ta ) a ∈A , which means we have to estimate one parameter per customer type. We can estimate this parameter by computing a numerical derivative for each customer type. Let e ta be a |A| -dimensional vector of zeroes with a 1 in the a th element. Then compute:

<!-- formula-not-decoded -->

Equation 13.9 is a sample estimate of the slope, which we then smooth to obtain an updated estimate of the slope:

<!-- formula-not-decoded -->

Computing ∼ V t -1 ( R t -1 + e ta , A t ( ω n )) for each product type a can be computationally burdensome in applications with large numbers of product types. It can be approximated by assuming that the decision variable z t does not change when we increment the resource vector.

Solving equation (13.8) requires that we determine both z t and x t . We only have two values of z t to evaluate, so this is not too hard, but how do we determine the vector x t ? We need to know something about how the customers differ. Assume that our customers are differentiated by their 'value' where the holding cost is a function of value. Given this, and given the linear value function approximation, it is not surprising that it is optimal to put as many of the most valuable customers into the vehicle as we can, and then move to the second most valuable, and so on, until we fill up the vehicle. This is how we determine the vector x t .

Given this method of determining x t , finding the best value of z t is not very difficult, since we simply have to compute the cost for z t = 0 and z t = 1. The next question is: how well does it work? We saw that it worked quite well for the case of a single customer type. With multiple customer types, we are no longer able to find the optimal solution. The classic 'curse of dimensionality' catches up with us, since we are not able to enumerate all possibilities of the resource vector R t .

As an alternative, we can compare against a sensible dispatch policy. Assume that we are going to dispatch the vehicle whenever it is full, but we will never hold it more than a maximum time τ . Further assume that we are going to test a number of values of τ and find the one that minimizes total costs. We call this an optimized 'dispatch-when-full' (DWF) policy (a bit misleading, since the limit on the holding time means that we may be dispatching the vehicle before it is full).

When testing the policy, it is important to control the relationship of the holding cost c h to the average dispatch cost per customer, c r /K . For c h &lt; c r /K , the best strategy tends to be to hold the vehicle until it is full. If c h &gt; c r /K , the best strategy will often be to limit how long the vehicle is held. When c h /similarequal c r /K , the strategy gets more complicated.

A series of simulations (reported in Papadaki &amp; Powell (2003)) were run on datasets with two types of customer arrival patterns: periodic, where the arrivals varied according to a fixed cycle (a period of high arrival rates followed by a period of low arrival rates). The results are given in table 13.1 for both the single and multiproduct problems, where the results are expressed as a fraction of the costs returned by our optimized DWF policy. The results show that the linear approximation always outperforms the DWF policy, even for the case c h &lt; c r /K where DWF should be nearly optimal. The multiproduct results are very close to the single product results. The computational complexity of the value function approximation for each forward pass is almost the same as DWF. If it is possible to estimate the value functions off-line, then there is very little additional burden for using an ADP policy.

Table 13.1: Costs returned by the value function approximation as a fraction of the costs returned by an dispatch-when-full policy.

|                          | Method     | linear   | linear   | linear   | linear   | linear   | linear   | linear   | linear   |
|--------------------------|------------|----------|----------|----------|----------|----------|----------|----------|----------|
|                          |            | scalar   | scalar   | scalar   | scalar   | mult.    | mult.    | mult.    | mult.    |
|                          | Iterations | (25)     | (50)     | (100)    | (200)    | (25)     | (50)     | (100)    | (200)    |
| c h > c r /K             | periodic   | 0.602    | 0.597    | 0.591    | 0.592    | 0.633    | 0.626    | 0.619    | 0.619    |
|                          | aperiodic  | 0.655    | 0.642    | 0.639    | 0.635    | 0.668    | 0.660    | 0.654    | 0.650    |
| c h /similarequal c r /K | periodic   | 0.822    | 0.815    | 0.809    | 0.809    | 0.850    | 0.839    | 0.835    | 0.835    |
|                          | aperiodic  | 0.891    | 0.873    | 0.863    | 0.863    | 0.909    | 0.893    | 0.883    | 0.881    |
| c h < c r /K             | periodic   | 0.966    | 0.962    | 0.957    | 0.956    | 0.977    | 0.968    | 0.965    | 0.964    |
|                          | aperiodic  | 0.976    | 0.964    | 0.960    | 0.959    | 0.985    | 0.976    | 0.971    | 0.969    |
| Average                  |            | 0.819    | 0.809    | 0.803    | 0.802    | 0.837    | 0.827    | 0.821    | 0.820    |

## 13.2 Monotone policies

One of the most dramatic success stories from the study of Markov decision processes has been the identification of the structure of optimal policies. An example is what are known as monotone policies . Simply stated, a monotone policy is one where the decision gets bigger as the state gets bigger, or the decision gets smaller as the state gets bigger (see examples).

Example 13.5: A software company must decide when to ship the next release of its operating system. Let S t be the total investment in the current version of the software. Let x t = 1 denote the decision to ship the release in time period t while x t = 0 means to keep investing in the system. The company adopts the rule that x t = 1 if S t ≥ ¯ S . Thus, as S t gets bigger, x t gets bigger (this is true even though x t is equal to zero or one).

Example 13.6: An oil company maintains stocks of oil reserves to supply its refineries for making gasoline. A supertanker comes from the Middle East each month, and the company can purchase different quantities from this shipment. Let R t be the current inventory. The policy of the company is to order x t = Q -S t if S t &lt; R . R is the reorder point, and Q is the 'order up to' limit. The bigger S t is, the less the company orders.

Example 13.7: A mutual fund has to decide when to sell its holding in a company. Its policy is to sell the stock when the price ˆ p t is greater than a particular limit ¯ p .

In each example, the decision of what to do in each state is replaced by a function that determines the decision. The function depends on the choice of one or two parameters. So, instead of determining the right action for each possible state, we only have to determine the parameters that characterize the function. Interestingly, we do not need dynamic programming for this. Instead, we use dynamic programming to determine the structure of the

optimal policy. This is a purely theoretical question, so the computational limitations of (discrete) dynamic programming are not relevant.

The study of monotone policies is included partly because it is an important part of the field of dynamic programming. It is also useful in the study of approximate dynamic programming because it yields properties of the value function. For example, in the process of showing that a policy is monotone, we also need to show that the value function itself is monotone (that is, increases or decreases with the resource state).

To demonstrate the analysis of a monotone policy, we consider a classic batch replenishment policy that arises when there is a random accumulation that is then released in batches. Examples include dispatching elevators or trucks, moving oil inventories away from producing fields in tankers, and moving trainloads of grain from grain elevators.

## 13.2.1 Submodularity and other stories

In the realm of optimization problems over a continuous set, it is important to know a variety of properties about the objective function (such as convexity/concavity, continuity and boundedness). Similarly, discrete problems require an understanding of the nature of the functions we are maximizing, but there is a different set of conditions that we need to establish.

One of the most important properties that we will need is supermodularity. Interestingly, different authors define supermodularity differently (although not inconsistently). We assume we are studying a function g ( u ) , u ∈ U where U ⊆ /Rfractur n is an n -dimensional space. Consider two vectors u 1 , u 2 ∈ U where there is no particular relationship between u 1 and u 2 . Now define:

```
u 1 ∧ u 2 = min { u 1 , u 2 } u 1 ∨ u 2 = max { u 1 , u 2 }
```

where the min and max are defined elementwise. Let u + = u 1 ∧ u 2 and u -= u 1 ∨ u 2 . We first have to ask the question of whether u + , u -∈ U , since this is not guaranteed. For this purpose, we define:

Definition 13.2.1 The space U is a lattice if for each u 1 , u 2 ∈ U , then u + = u 1 ∧ u 2 ∈ U and u -= u 1 ∨ u 2 ∈ U .

The term 'lattice' for these sets arises if we think of u 1 and u 2 as the northwest and southeast corners of a rectangle. In that case, these corners are u + and u -. If all four corners fall in the set (for any pair ( u 1 , u 2 )), then the set can be viewed as containing many 'squares,' similar to a lattice.

For our purposes, we assume that U is a lattice (if it is not, then we have to use a more general definition of the operators ' ∨ ' and ' ∧ '). If U is a lattice, then a general definition of supermodularity is given by:

Definition 13.2.2 A function g ( u ) , u ∈ U is supermodular if it satisfies:

<!-- formula-not-decoded -->

A function is submodular if the inequality in equation (13.11) is reversed. There is an alternative definition of supermodular when the function is defined on sets. Let U 1 and U 2 be two sets of elements, and let g be a function defined on these sets. Then we have:

Definition 13.2.3 A function g : U ↦→ /Rfractur 1 is supermodular if it satisfies:

<!-- formula-not-decoded -->

We may refer to definition 13.2.2 as the vector definition of supermodularity, while definition 13.2.3 as the set definition. We give both definitions for completeness, but our work uses only the vector definition.

In dynamic programming, we are interested in functions of two variables, as in f ( s, x ) where s is a state variable and x is a decision variable. We want to characterize the behavior of f ( s, x ) as we change s and x . If we let u = ( s, x ), then we can put this in the context of our definition above. Assume we have two states s + ≥ s -(again, the inequality is applied elementwise) and two decisions x + ≥ x -. Now, form two vectors u 1 = ( s + , x -) and u 2 = ( s -, x + ). With this definition, we find that u 1 ∧ u 2 = ( s + , x + ) and u 1 ∨ u 2 = ( s -, x -). This gives us the following:

Proposition 13.2.1 A function g ( s, x ) is supermodular if for s + ≥ s -and x + ≥ x -, then:

<!-- formula-not-decoded -->

For our purposes, equation (13.13) will be the version we will use.

A common variation on the statement of a supermodular function is the equivalent condition:

<!-- formula-not-decoded -->

In this expression, we are saying that the incremental change in s for larger values of x is greater than for smaller values of x . Similarly, we may write the condition as:

<!-- formula-not-decoded -->

which states that an incremental change in x increases with s .

Some examples of supermodular functions include:

- a) If g ( s, x ) = g 1 ( s )+ g 2 ( x ), meaning that it is separable, then (13.13) holds with equality.
- b) g ( s, x ) = h ( s + x ) where h ( · ) is convex and increasing.
- c) g ( s, x ) = sx, s, x ∈ /Rfractur 1 .

A concept that is related to supermodularity is superadditivity :

Definition 13.2.4 A superadditive function f : /Rfractur n →/Rfractur 1 satisfies:

<!-- formula-not-decoded -->

Some authors use superadditivity and supermodularity interchangeably, but the concepts are not really equivalent, and we need to use both of them.

## 13.2.2 From submodularity to monotonicity

It seems intuitively obvious that a dispatch rule would follow a monotone structure. Specifically, there would be a limit ¯ r t whereby if s t ≥ ¯ s t , then we dispatch a batch. Otherwise we hold it. This is known as a control limit structure . A question arises: when is an optimal policy monotone? The following theorem establishes sufficient conditions for an optimal policy to be monotone.

Theorem 13.2.1 Assume that:

- a) C t ( R,x ) is supermodular on R×X .
- b) ∑ R ′ ∈R p ( R ′ | R,x ) v t +1 ( R ′ ) is supermodular on R×X .

Then there exists a decision rule X π ( R ) that is nondecreasing on R .

In the presentation that follows, we need to show submodularity (instead of supermodularity) because we are minimizing costs rather than maximizing rewards.

It is obvious that C t ( R,x ) is nondecreasing in R . So it remains to show that C t ( R,x ) satisfies:

<!-- formula-not-decoded -->

Substituting equation (13.2) into (13.17), we must show that:

<!-- formula-not-decoded -->

This simplifies to:

<!-- formula-not-decoded -->

Since R + ≥ R -, ( R + -K ) + = 0 ⇒ ( R --K ) + = 0. This implies there are three possible cases for equation (13.18): Case 1: ( R + -K ) + &gt; 0 and ( R --K ) + &gt; 0.

In this case, (13.18) reduces to R + -R -= R + -R -. Case 2: ( R + -K ) + &gt; 0 and ( R --K ) + = 0.

Here, (13.18) reduces to R -≤ K , which follows since ( R --K ) + = 0 implies that R -≤ K . Case 3: ( R + -K ) + = 0 and ( R --K ) + = 0.

Now, (13.18) reduces to R -≤ R + .

Now we have to show submodularity of ∑ ∞ R ′ =0 p ( R ′ | s, x ) V ( R ′ ). We will do this for the special case that the vehicle capacity is so large that we never exceed it. (A proof is available for the finite capacity case, but it is much more difficult).

Submodularity requires that for R -≤ R + :

<!-- formula-not-decoded -->

For the case that R -, R + ≤ K :

<!-- formula-not-decoded -->

which simplifies to:

<!-- formula-not-decoded -->

Since V is nondecreasing: V ( R ′ + R + ) ≥ V ( R ′ + R -). Thus, (13.19) is satisfied.

## 13.3 Why does it work?**

## 13.3.1 Optimality of monotone policies

The foundational result that we use is the following technical lemma:

Lemma 13.3.1 If a function g ( s, x ) is supermodular, then:

<!-- formula-not-decoded -->

is monotone and nondecreasing in s .

If the function g ( s, x ) has a unique, optimal x ∗ ( s ) for each value of s , then we can replace (13.20) with:

<!-- formula-not-decoded -->

Discussion: The lemma is saying that if g ( s, x ) is supermodular, then as s grows larger, the optimal value of x given s will grow larger. When we use the version of supermodularity given in equation (13.15), we see that the condition implies that as the state becomes larger, the value of increasing the decision also grows. As a result, it is not surprising that the condition produces a decision rule that is monotone in the state vector.

Proof of the lemma: Assume that s + ≥ s -, and choose x ≤ x ∗ ( s -). Since x ∗ ( s ) is, by definition, the best value of x given s , we have:

<!-- formula-not-decoded -->

The inequality arises because x ∗ ( s -) is the best value of x given s -. Supermodularity requires that:

<!-- formula-not-decoded -->

Rearranging (13.23) gives us:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We obtain equation (13.25) because the term in brackets in (13.24) is nonnegative (from (13.22)).

Clearly:

<!-- formula-not-decoded -->

because x ∗ ( s + ) optimizes g ( s + , x ). This means that x ∗ ( s + ) ≥ x ∗ ( s -) since otherwise, we would simply have chosen x = x ∗ ( s + ).

Just as the sum of concave functions are concave, we have:

## Proposition 13.3.1 The sum of supermodular functions is supermodular.

The proof follows immediately from the definition of supermodularity, so we leave it as one of those proverbial exercises for the student.

The main theorem regarding monotonicity is relatively easy to state and prove, so we will do it right away. The conditions required are what make it a little more difficult.

## Theorem 13.3.1 Assume that:

- a) C t ( s, x ) is supermodular on S × X .
- b) ∑ s ′ ∈S p ( s ′ | s, x ) v t +1 ( s ′ ) is supermodular on S × X .

Then there exists a decision rule x ( s ) that is nondecreasing on S .

## Proof: Let:

<!-- formula-not-decoded -->

The two terms on the right hand side of (13.26) are assumed to be supermodular, and we know that the sum of two supermodular functions is supermodular, which tells us that w ( s, x ) is supermodular. Let:

<!-- formula-not-decoded -->

From Lemma 13.3.1, we obtain the result that the decision x ∗ ( s ) increases monotonically over S , which proves our result.

The proof that the one-period reward function C t ( s, x ) is supermodular must be based on the properties of the function for a specific problem. Of greater concern is establishing the conditions required to prove condition ( b ) of the theorem because it involves the property of the value function, which is not part of the basic data of the problem.

In practice, it is sometimes possible to establish condition ( b ) directly based on the nature of the problem. These conditions usually require conditions on the monotonicity of the reward function (and hence the value function) along with properties of the one-step transition matrix. For this reason, we will start by showing that if the one-period reward function is nondecreasing (or nonincreasing), then the value functions are nondecreasing (or nonincreasing). We will first need the following technical lemma:

Lemma 13.3.2 Let p j , p ′ j , j ∈ J be probability mass functions defined over J that satisfy:

<!-- formula-not-decoded -->

and let v j , j ∈ J be a nondecreasing sequence of numbers. Then:

<!-- formula-not-decoded -->

Remark: We would say that the distribution represented by { p j } j ∈J stochastically dominates the distribution { p ′ j } j ∈J . If we think of p j as representing the probability a random variable V = v j , then equation (13.28) is saying that E p V ≥ E p ′ V . Although this is well known, a more algebraic proof is as follows:

Proof: Let v - 1 = 0 and write:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In equation (13.29), we replace v j with an alternating sequence that sums to v j . Equation (13.30) involves one of those painful change of variable tricks with summations. Equation (13.31) is simply getting rid of the term that involves v -1 . In equation (13.32), we replace the

cumulative distributions for p j with the distributions for p ′ j , which gives us the inequality. Finally, we simply reverse the logic to get back to the expectation in (13.33). /square

We stated that lemma 13.3.2 is true when the sequences { p j } and { p ′ j } are probability mass functions because it provides an elegant interpretation as expectations. For example, we may use v j = j , in which case equation (13.28) gives us the familiar result that when one probability distribution stochastically dominates another, it has a larger mean. If we use an increasing sequence v j instead of j , then this can be viewed as nothing more than the same result on a transformed axis.

In our presentation, however, we need a more general statement of the lemma, which follows:

Lemma 13.3.3 Lemma 13.3.2 holds for any real valued, nonnegative sequences { p j } and { p ′ j } .

The proof involves little more than realizing that the proof of lemma 13.3.2 never required that the sequences { p j } and { p ′ j } be probability mass functions.

## Proposition 13.3.2 Suppose that:

- a) C t ( s, x ) is nondecreasing (nonincreasing) in s for all x ∈ X and t ∈ T .
- b) C t ( s ) is nondecreasing (nonincreasing) in s .
- c) q t (¯ s | s, x ) = ∑ s ′ ≥ ¯ s p ( s ′ | s, x ) , the reverse cumulative distribution function for the transition matrix, is nondecreasing in s for all s ∈ S , x ∈ X and t ∈ T .

Then, v t ( s ) is nondecreasing (nonincreasing) in s for t ∈ T .

Proof: As always, we use a proof by induction. We will prove the result for the nondecreasing case. Since v T ( s ) = C t ( s ), we obtain the result by assumption for t = T . Now, assume the result is true for v t ′ ( s ) for t ′ = t +1 , t +2 , . . . , T . Let x ∗ t ( s ) be the decision that solves:

<!-- formula-not-decoded -->

Let ˆ s ≥ s . Condition ( c ) of the proposition implies that:

<!-- formula-not-decoded -->

Lemma 13.3.3 tells us that when (13.35) holds, and if v t +1 ( s ′ ) is nondecreasing (the induction hypothesis), then:

<!-- formula-not-decoded -->

Combining equation (13.36) with condition (a) of proposition 13.3.2 into equation (13.34) gives us:

<!-- formula-not-decoded -->

which proves the proposition.

With this result, we can establish condition (b) of theorem 13.3.1:

## Proposition 13.3.3 If:

<!-- formula-not-decoded -->

b) v ( s ) is nondecreasing in s , then ∑ s ′ ∈S p ( s ′ | s, x ) v ( s ′ ) is supermodular on S × X .

Proof: Supermodularity of the reverse cumulative distribution means:

<!-- formula-not-decoded -->

We can apply Lemma 13.3.3 using p ¯ s = ∑ s ′ ≥ ¯ s p ( s ′ | s + , x + ) + ∑ s ′ ≥ ¯ s p ( s ′ | s -, x -) and p ′ ¯ s = ∑ s ′ ≥ ¯ s p ( s ′ | s + , x -) + ∑ s ′ ≥ ¯ s p ( s ′ | s -, x + ), which gives:

<!-- formula-not-decoded -->

which implies that ∑ s ′ ∈S p ( s ′ | s, x ) v ( s ′ ) is supermodular.

/square

Remark: Supermodularity of the reverse cumulative distribution ∑ s ′ ∈S p ( s ′ | s, x ) may seem like a bizarre condition at first, but a little thought suggests that it is often satisfied in practice. As stated, the condition means that:

<!-- formula-not-decoded -->

/square

Assume that the state s is the water level in a dam, and the decision x controls the release of water from the dam. Because of random rainfalls, the amount of water behind the dam in the next time period, given by s ′ , is random. The reverse cumulative distribution gives us the probability that the amount of water is greater than s + (or s -). Our supermodularity condition can now be stated as: 'If the amount of water behind the dam is higher one month ( s + ), then the effect of the decision of how much water to release ( x ) has a greater impact than when the amount of water is initially at a lower level ( s -).' This condition is often satisfied because a control frequently has more of an impact when a state is at a higher level than a lower level.

Remark: For another example of supermodularity of the reverse cumulative distribution, assume that the state represents a person's total wealth, and the control is the level of taxation. The effect of higher or lower taxes is going to have a bigger impact on wealthier people than on those who are not as fortunate (but not always: think about other forms of taxation that affect less affluent people more than the wealthy, and use this example to create an instance of a problem where a monotone policy may not apply).

We now have the result that if the reward function C t ( s, x ) is nondecreasing in s for all x ∈ X and the reverse cumulative distribution ∑ s ′ ∈S p ( s ′ | s, x ) is supermodular, then ∑ s ′ ∈S p ( s ′ | s, x ) v ( s ′ ) is supermodular on S × X . Combine this with the supermodularity of the one-period reward function, and we obtain the optimality of a nondecreasing decision function.

## 13.4 Bibliographic notes

Topkins (1978), Papadaki &amp; Powell (2003), Puterman (1994), Papadaki &amp; Powell (2002), Bulk queues: Neuts (1967), Powell &amp; Humblet (1986), Aalto (2000), Berg et al. (1998), Control of batch queues: Deb (1978 a ), Deb (1978 b ), Deb (1984), Deb &amp; Schmidt (1987), Deb &amp; Serfozo (1973) Approximate DP and batch replenishment: Adelman (2004)

## Exercises

- 1) An airline has to decide when to bring an aircraft in for a major engine overhaul. Let s t represent the state of the engine in terms of engine wear, and let d t be a nonnegative amount by which the engine deteriorates during period t . At the beginning of period t , the airline may decide to continue operating the aircraft ( z t = 0) or to repair the aircraft ( z t = 1) at a cost of c R , which has the effect of returning the aircraft to s t +1 = 0 . If the airline does not repair the aircraft, the cost of operation is c o ( s t ), which is a nondecreasing, convex function in s t .

- a) Define what is meant by a control limit policy in dynamic programming, and show that this is an instance of a monotone policy.
- b) Formulate the one-period period reward function r t ( s t , z t ), and show that it is submodular.
- c) Show that the decision rule is monotone in s t . (Outline the steps in the proof, and then fill in the details).
- d) Assume that a control limit policy exists for this problem, and let γ be the control limit. Now, we may write r t ( s t , z t ) as a function of one variable: the state s . Using our conrol limit structure, we can write the decision z t as the decision rule z π ( s t ). Illustrate the shape of r t ( s t , z π ( s )) by plotting it over the range 0 ≤ s ≤ 3 γ (in theory, we may be given an aircraft with s &gt; γ initially).
- 2) A dispatcher controls a finite capacity shuttle that works as follows: In each time period, a random number A t arrives. After the arrivals occur, the dispatcher must decide whether to call the shuttle to remove up to M customers. The cost of dispatching the shuttle is c , which is independent of the number of customers on the shuttle. Each time period that a customer waits costs h . If we let z = 1 if the shuttle departs and 0 otherwise, then our one period reward function is given by:

<!-- formula-not-decoded -->

where M is the capacity of the shuttle. Show that r t ( s, a ) is submodular where we would like to minimize r . Note that we are representing the state of the system after the customers arrive.

- 3) Assume that a control limit policy exists for our shuttle problem in exercise 2 that allows us to write the optimal dispatch rule as a function of s , as in z π ( s ). We may write r ( s, z ) as a function of one variable, the state s .
- a) Illustrate the shape of r ( s, z ( s )) by plotting it over the range 0 &lt; s &lt; 3 M (since we are allowing there to be more customers than can fill one vehicle, assume that we are allowed to send z = 0 , 1 , 2 , . . . vehicles in a single time period).
- b) Let c = 10, h = 2, and M = 5, and assume that A t = 1 with probability 0.6 and is 0 with probability 0.4. Set up and solve a system of linear equations for the optimal value function for this problem in steady state.
- 4) A general aging and replenishment problem arises as follows: Let s t be the 'age' of our process at time t . At time t , we may choose between a decision d = C to continue the process, incurring a cost g ( s t ) or a decision d = R to replenish the process, which incurs a cost K + g (0). The state of the system evolves according to:

<!-- formula-not-decoded -->

where D t is a nonnegative random variable giving the degree of deterioration from one epoch to another (also called the 'drift').

- a) Prove that the structure of this policy is monotone. Clearly state the conditions necessary for your proof.
- b) How does your answer to part (1) change if the random variable D t is allowed to take on negative outcomes? Give the weakest possible conditions on the distribution of required to ensure the existence of a monotone policy.
- c) Now assume that the action is to reduce the state variable by an amount q ≤ s t at a cost of cq (instead of K ). Further assume that g ( s ) = as 2 . Show that this policy is also monotone. Say as much as you can about the specific structure of this policy.

## Chapter 14

## Two-stage stochastic programming

The next problem class that we would like to consider is the management of multiple asset classes. We may have problems with dozens, hundreds or tens of thousands of asset classes. An asset could be money held in a particular type of investment, products moving through a supply chain, equipment such as electric power transformers, and transportation assets such as trucks, trains and planes. For these problems, we are going to need the tools in the field of math programming.

Arecurring theme in approximate dynamic programming for resource allocation problems is that solving n -period problems is reduced to solving sequences of what are known as 'twostage' problems. A two-stage problem involves making a decision, seeing information, and then possibly making one more decision, at which point the problem stops. These problems are solved by making the first decision with some sort of approximation of its downstream effects. Needless to say, this sounds like what we do in dynamic programming, where the downstream effects of a decision are captured through the value function. The stochastic programming uses the term recourse function .

There are some other more substantive differences between the perspectives of the dynamic programming and the stochastic programming communities. Dynamic programs are often formulated using discrete states and actions, although it is becoming increasingly common to approximate value functions using a continuous representation of a state variable. Problems with dozens of dimensions are often viewed as being 'high dimensional.' Stochastic programming traces its roots to the efficient management of resources. Transportation applications provided some of the earliest motivating applications, but many problems in the management of inventories and 'supply chains' provide fresh applications. These problems are often characterized by much higher dimensionalities. A decision vector can easily have ten thousand dimensions. State vectors are measured in terms of the number of dimensions (which can easily be in the thousands) rather than in the number of potential states. However, states and decisions are typically modeled as continuous variables (although we may insist that the final solution be integer). Furthermore, these problems have special qualities that can be leveraged, producing computationally efficient algorithms for problems that the dynamic programming community would view as 'ultra large.'

An extensive literature for solving two-stage problems has evolved from within the stochastic programming community. It does not use the vocabulary of dynamic programming, but some of the techniques are directly applicable to multistage stochastic resource allocation problems (or, as they are called in this book, 'asset management problems'). However, the techniques are directly applicable to solving the same types of multistage problems that we have been solving using dynamic programming.

Throughout this chapter, we view x as a continuous, possibly high-dimensional vector (even if we are interested in integer solutions). An important characteristic of all these problems is that they have to be solved subject to a set of (typically linear) constraints. If we did not have to worry about the effect of decisions now on the future, we would simply apply standard math programming algorithms. The need to use standard math programming algorithms limits somewhat the classes of approximations we can use for a value function.

## 14.1 Two-stage stochastic programs with recourse

One of the fundamental problems that arise in asset allocation arises when we have to make decisions about acquiring and managing assets before we know exactly how they will be used. As a result, we have to make a decision before seeing what the real needs are. When we do, we may be able to make another set of decisions. The second set of decisions, the ones still available for us after we make our first stage decisions and after we see the new information, represent our recourse.

An important class of recourse strategies for asset allocation problems is called network recourse . In this class, our recourse after seeing new information is to solve a network problem. The following examples provide illustrations of problems that use network recourse.

In all of these problems, we make an initial allocation decision. This could be the decision to invest in (purchase) a particular type of equipment, build a particular model of car, or stock inventories. Once the initial decision is made, we see information about the demand for the asset as well as the prices/costs derived from satisfying a demand (which can also be random). After this information is revealed, we may make new decisions. The goal is to make the best initial decisions given the potential downstream decisions that might be made.

We can write an allocation problem using the following notation:

- R 0 a = The number of resources with attribute a initially in the system. x 0 ad = The number of resources that initially have attribute a that we act on with decision d . D 0 = Set of decision types.
- c 0 ad = Contribution earned (negative if it is a cost) from using decision d acting on resources with attribute a .

Example 14.1: An electric power utility needs to purchase expensive components that cost millions of dollars and require a year or more to order. The industry needs to maintain a supply of these in case of a failure. The problem is to determine how many units to purchase, when to purchase them, what features they should have, and where they should be stored. When a failure occurs, the company will find the closest unit that has the features required for a particular situation.

Example 14.2: An investment bank needs to allocate funds to various investments (long-term, high risk investments, real estate, stocks, index funds, bonds, money markets, CD's). As opportunities arise, the bank will move money from one investment to another, but these transactions can take time to execute and cost money (for example, it is easiest and fastest to move money out of a money market fund).

Example 14.3: An online bookseller prides itself in fast delivery, but this requires holding books in inventory. If orders arrive when there is no inventory, the seller may have to delay filling the order (and risk losing it) or purchase the books at a higher cost from the publisher. If the inventory is too high, the company has to choose between holding the books in inventory (tying up space and capital), discounting the book to increase sales, or selling inventory to another distributor (at a substantial discount).

Example 14.4: An automotive manufacturer has to decide what models to design and build, and with what features. Given a three year design and build cycle, they have to create cars that will respond to an uncertain marketplace in the future. Once the models are built, customers have to adjust and purchase models that are closest to their wishes.

The contribution function for the first time period might be:

<!-- formula-not-decoded -->

We capture the result of a particular decision type d using our indicator function:

<!-- formula-not-decoded -->

The first stage decisions produce a second stage (resource) state vector as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 14.1: A two-stage allocation problem. The first stage decisions have to be made before the second stage information becomes known. When this information is revealed, it is possible to re-allocate resources.

<!-- image -->

In the second stage, we might see three types of information: the costs, the result of a decision, and most importantly, the demands for resources. We typically assume that we can apply a new set of decisions (which we denote D 1 ) which, when applied to a resource with attribute a ′ , will produce a resource with attribute a ′′ . Let:

<!-- formula-not-decoded -->

We note that δ a ′ (1 , a, d ) is initially random, but is a function of the information that becomes available in time interval 1. Finally we assume that the resource vector R x 1 can be used to satisfy demands that first become known during time interval 1. Our decisions in the second stage can be 'do nothing' types of decisions, which are not constrained, and decisions to satisfy a demand that first became known during time interval 1. For simplicity, we assume that there is a contribution (which can be negative) for a resource of type a ′′ and an upper bound on the number of resources that can produce this contribution. Through the attribute vector, we capture resources that are being used to satisfy a particular type of demand. We assume that all resources can be acted on in a way that is not constrained so that we are sure of a feasible solution. Thus, our second stage contribution function is given by:

<!-- formula-not-decoded -->

Remember that every quantity with a time index t = 1 is allowed to see information that becomes available during time interval 1, which means that it is random at time 0. If we want to make the best initial allocation decisions, we have to solve problems of the form:

<!-- formula-not-decoded -->

This problem has to be solved subject to the first stage contraints:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The second stage decisions have to be made subject to constraints for each outcome ω :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (14.6) is the constraint that defines the vector of resources available after all the initial decisions are made (we assume that the outcome of decisions now are deterministic). This becomes the resource constraint in the second stage (equation (14.7)). Equation (14.9) tells us what happens as a result of decisions in this stage (which can be random), and for those decisions that are used to satisfy demands (that only become known in the second stage), we encounter a limit on the number of resources that can be used for this purpose (equation (14.10)). We assume that for resources that are simply being held in inventory, u 1 a ′′ = ∞ .

Equations (14.3)-(14.10) describe a fairly general model. It is useful to see the model formulated at a detailed level, but it can also be useful to express this model in a more compact form. The constraints can also be written:

## First stage constraints:

<!-- formula-not-decoded -->

Second stage constraints:

<!-- formula-not-decoded -->

The second stage contribution function depends on the first stage allocation, R x 0 ( x 0 ), so we can write it as:

<!-- formula-not-decoded -->

which allows us to rewrite (14.3) as:

<!-- formula-not-decoded -->

This shows that our two-stage problem consists of a one-period contribution function (using the information we know now) and a value function that captures the expected contribution, which depends on the decisions that we made in the first stage.

We pause to note a major stylistic departure between stochastic and dynamic programming. In writing equation (14.11), it is clear that we have written our expected value function in terms of the resource state variable R x 0 . Of course, R x 0 is a function of x 0 , which means we could write (14.11) as:

<!-- formula-not-decoded -->

This is the style favored by the stochastic programming community. Mathematically, they are equivalent, but computationally, they can be quite different. If |A| is the size of our resource attribute space, the dimensionality of x 0 is typically on the order of |A| × |A| whereas the dimensionality of R x 0 will be on the order of |A| . As a rule, it is computationally much easier to approximate V 0 ( R x 0 ) than V 0 ( x 0 ) simply because lower dimensional functions are easier to approximate than higher dimensional ones. In some of our discussions in this chapter, we adopt the convention of writing V 0 as a function of x 0 in order to clarify the relationship with stochastic programming.

This problem class arises in a broad range of resource allocation problems. It is also a nice prototypical example of a constrained stochastic optimization problem. If we replace the time index 0 with time index t , then we can easily see how this could be imbedded within a multistage framework.

## 14.2 Stochastic projection algorithms for constrained optimization

In chapter 6, we saw the power and flexibility of the basic stochastic gradient algorithm:

<!-- formula-not-decoded -->

In this chapter, we want to solve problems subject to a requirement that x ∈ X , where X captures constraints such as nonnegativity and conservation of flow. It is quite likely that our basic stochastic gradient update will produce a value for x n that is no longer feasible (for example, the gradient might tell us that we want more of everything!). One strategy for solving this problem is to perform our update in two steps:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Π X [ x ] is the nearest point projection of x onto the (presumably convex) feasible region X . In this algorithm, equation (14.13) first performs the basic update, possibly producing a solution ˜ x n that is outside the feasible region. Then, equation (14.14) projects ˜ x n back onto the feasible region. In general, this projection is performed by solving:

<!-- formula-not-decoded -->

This step looks daunting, but for several important classes of problems it is, in fact, extremely simple. For example, nonnegativity constraints are handled simply by letting:

<!-- formula-not-decoded -->

where the max operator is performed elementwise (see figure 14.2(a)). Upper bound constraints, such as x ≤ U , are similarly handled using:

<!-- formula-not-decoded -->

Figure 14.2(b) illustrates the handling of two upper bound violations. We first project x 2 onto the region x 2 ≤ u 2 and then project x 1 onto x 1 ≤ u 1 .

For resource allocation problems, we often face flow conservation constraints of the form:

<!-- formula-not-decoded -->

This is more difficult than a simple upper or lower bound, but it turns out to remain quite easy. To see why, consider the situation depicted in figure 14.2(c) where we start with a

<!-- image -->

14.2a: Projection onto the constraint x 1 ≥ 0.

<!-- image -->

14.2c: Projection onto the constraint x 1 + x 2 = R .

<!-- image -->

14.2b: Projection first onto the constraint x 2 ≤ u 2 and then onto the constraint x 1 ≤ u 1 .

<!-- image -->

14.2d:

Projection onto the constraint

x

2

=

R

x

1

+

then onto the constraint

x

2

≥

,

0, and then back onto the con- straint

x

1

+

x

2

=

R

.

Figure 14.2: Illustration of nearest point projections. (a) Shows the projection back onto the constraint x 1 ≥ 0. (b) Shows how we can enforce two upper bound violations by first projecting onto the region x 2 ≤ u 2 , and then onto the region x 1 ≤ u 1 . (c) Shows a projection onto the feasible region x 1 + x 2 = R . (d) Illustrates the steps when the projection onto x 1 + x 2 = R produces a nonnegativity violation. After projecting onto x 1 + x 2 = R , we project onto x 2 ≥ 0 and then set x 2 = 0 for further calculations. The next projection onto x 1 + x 2 = R only involves x 1 .

feasible point x n -1 and move to a point ˜ x n , which is out of the feasible region. In our twodimensional example, the line from ˜ x n to the nearest point in the feasible region x n is at a 45 degree angle, which means we can write:

<!-- formula-not-decoded -->

where e is a vector of 1's and β is a scalar. So, our problem is simply to find β . We want x n to satisfy equation (14.16). In our figure, ˜ x n is too big, so we have to make all the elements smaller. The element is too large by the amount:

<!-- formula-not-decoded -->

So, if | e | = |D| is the number of elements in the vector x , we simply let β equal ∆ R divided by the number of dimensions in x :

<!-- formula-not-decoded -->

Now, we are assured that x n satisfies (14.16). The only problem is that we usually have nonnegativity constraints as well, and the update (14.17) may easily produce negative elements of x n (see figure 14.2(d)). Let e d be the element of e corresponding to d ∈ D . We handle this by setting:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all d ∈ D . The vector e is now an indicator vector. If an element is zero, it means that we had to make it zero to avoid a negative allocation. However, if ∑ d x n d = R before applying equation (14.19), then if we set any of the elements e d to zero because x n d &lt; 0, we would now have ∑ d x n d &lt; R . If this is the case, we recompute ∆ R :

<!-- formula-not-decoded -->

followed by:

<!-- formula-not-decoded -->

Note that this time around, the vector e consists of zeroes and ones. This means that once we have set an element of x n to zero, it stays zero. At this point, we have ∑ d ∈D x n d = R ,

but we again have to check to see if any of the elements are negative. If not, we are done. If so, we repeat the process again. The steps are illustrated graphically in figure 14.2(d).

The algorithm can be applied to our two-stage stochastic program:

<!-- formula-not-decoded -->

by viewing F ( x ) = C 0 ( x 0 ) + V 0 ( R x 0 ( x 0 )) = C 0 ( x 0 ) + E V 1 ( R x 0 ( x 0 ) , W 1 ). We compute our stochastic gradient by dropping the expectation and finding the gradient of V 1 ( R x 0 ( x 0 ) , W 1 ( ω )) for a sample realization ω .

This algorithmic strategy also works even if our original function f ( x, W ) is nondifferentiable (although we do require continuity). This property will prove useful in certain classes of asset allocation problems where the function being minimized is the optimal solution of a linear program (which is continuous but nondifferentiable). The rate of convergence, however, remains an open question.

Stochastic gradient algorithms are problematic if we are looking for integer solutions. This is not a problem if we are allocating money or continuous quantities such as energy or agricultural resources, but it would be a problem if we are trying to determine whether a power plant should be burning oil or natural gas, or how to allocate discrete assets such as jets or people. Stochastic gradient algorithms also do little to take advantage of the structure of the problem. It might be the case that a stochastic gradient would want us to increase our allocations everywhere (since more might always be better). We are then depending on the projection operator to allocate finite assets among competing resources.

## 14.3 Proximal point algorithms

Proximal point algorithms were initially introduced by Rockafellar (1976) to stabilize algorithms for nondifferentiable problems. In this problem class, nondifferentiable problems can introduce instabilities as the algorithm bounces around solutions that produce identical or nearly identical objective function values. Later, the concept was extended to stochastic optimization problems(Ruszczy´ nski (1980 a )). The basic idea is to minimize an approximation of the problem plus a term that penalizes deviations from a previous solution. An iteration of the algorithm looks like:

<!-- formula-not-decoded -->

Here, θ n is a scaling parameter, and ‖ x -x n -1 ‖ measures the difference between the current solution x and the previous solution x n -1 . Typically, we would use a Euclidean norm, which gives us:

<!-- formula-not-decoded -->

Larger values of θ n penalize deviations from the previous solution, which stabilizes the algorithm.

To obtain a convergent algorithm, it is necessary to create a sequence of weights θ n that increase with the iterations. The idea is that as the algorithm progresses, we want to put more weight on previous solutions and less on the current stochastic gradient. Consider what happens when we use:

<!-- formula-not-decoded -->

where α n is our standard stepsize which goes to zero over time. Since (14.22) is unconstrained, we can find x n by differentiating and setting the result equal to zero:

<!-- formula-not-decoded -->

Solving for x gives us:

<!-- formula-not-decoded -->

Thus, a proximal point algorithm using a Euclidean distance metric and a weight factor equal to 1 / 2 α n gives us our standard stochastic gradient algorithm.

The real power of the proximal point algorithm is that it yields feasible solutions at every iteration. This avoids the need to solve a nearest point projection, which can be computationally difficult for general linear programs.

## 14.4 The SHAPE algorithm for differentiable functions

The SHAPE algorithm, which we first saw in section 11.4, can be used in the context of twostage stochastic approximation problems by providing a convenient way of approximating the second stage. We start by writing our problem in the form:

<!-- formula-not-decoded -->

where our second stage allocation problem is:

<!-- formula-not-decoded -->

Our second stage constraints might look like:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where R x 0 a is the number of assets with attribute a that we have as a result of first-stage decisions and ˆ R 1 a ( ω ) represents the amount of new assets with attribute a that randomly become available in the second time period. We can treat some decisions d as a decision to satisfy a certain type of demand. For these decisions, the upper bounds u 1 d capture the amount of these demands (these would be upper bounds on how much we can sell at a particular price). For this section, we assume that all variables are continuous. In practical applications, the allocation vectors x 0 and x 1 can have any where from dozens to tens of thousands of elements. Not surprisingly, computing E ( R x 0 , W 1 ) is computationally intractable.

We can solve this problem using the SHAPE algorithm of section 11.4. We start by picking a convenient analytical approximation of E ( R x 0 , W 1 ) that captures the properties of the problem as much as possible. For example, we know the function is concave. We might also know roughly where the 'elbow' of the function, the point where we are roughly satisfying demand, occurs. If we have this information, let ¯ R a be our rough estimate of how many assets with attribute a that we think we need. We might then create an approximate value function:

<!-- formula-not-decoded -->

where θ 0 a and θ 1 a are parameters that have to be specified. (A notational note: the superscript 0 in ¯ V 0 0 ( R x 0 ) refers to the initial estimate of the function; the subscript 0 indicates that we are approximating the expectation, and therefore can be measured at time 0.)

The unconstrained optimum value of ¯ V 0 0 ( R x 0 ) is zero. There are two reasons why this solution may not be optimal. First, we may not have the resources to supply our target quantities ¯ R a . Second, and more relevant to our presentation here, is that our targets θ 1 a are just rough approximations. The SHAPE algorithm would start by solving:

<!-- formula-not-decoded -->

This is a nonlinear programming problem, and we would have to use any of a broad variety of nonlinear programming algorithms. We note that we have chosen an approximation for our value function that is separable. Although we did not have to do this, it may introduce additional simplifications in the solution of the problem.

Step 0. Initialize ¯ V 0 0 and set n = 1. Set n = 0 and choose an initial approximation ¯ V 0 0 ( x );

Step 1. Obtain x n by

<!-- formula-not-decoded -->

Step 2. Obtain g n = ∇ V ( R x,n -1 , ω n ) and update ¯ V n -1 ( R x

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 3. Check for convergence (in practice, a limit on the number of iterations). If we do not satisfy this test, set n = n +1 and go to step 1.

Figure 14.3: The SHAPE algorithm for two-stage, constrained stochastic optimization problems.

We then take our solution x 1 0 , choose a Monte Carlo sample ω 1 , and solve the deterministic second stage problem subject to the constraints (14.25)-(14.27), where:

<!-- formula-not-decoded -->

is the initial second stage supplies resulting from our first stage decisions.

In our example, this is a linear program that typically has the structure of a network. Let:

<!-- formula-not-decoded -->

The vector of dual variables ˆ v 1 1 represents a stochastic subgradient of V 1 ( R x 0 , W 1 ( ω )). It is stochastic because we obtained the dual variables for a single realization ω , and it is a subgradient because dual variables are always subgradients. We can then use this information to update our approximation using the SHAPE algorithm:

<!-- formula-not-decoded -->

We emphasize that ˆ v 1 1 and ∇ ¯ V 0 0 ( R x, 1 0 ) are vectors of constants whereas R x 0 is a variable. Thus, we are simply adding a linear term to our approximation.

The SHAPE algorithm is described in figure 14.3. As with our other stochastic approximation methods, practical convergence criteria do not exist. Instead, we have to investigate the stability of the algorithm with different stepsize rules to determine the number of iterations needed to provide the best possible solution.

The SHAPE algorithm allows us to work with approximations where we have considerable control over the structure of the approximation. This allows us to design algorithms that

exploit this structure. For example, separable, quadratic approximations can be used with specialized algorithms for this problem class.

The SHAPE algorithm will, in the limit, produce an optimal solution. For a proof, and a summary of the technical conditions that have to be satisfied, are given in section 14.7.1. At this point, it is more useful to discuss why it works. The theorem states that if the original problem (and the approximation) is continuously differentiable, then we can use a sequence of approximations (which may even be separable) and get a convergent algorithm just by tilting the function. The reason for this is simply that all we need is an approximation that gives us the correct slope at the point of optimality. We do not need the function to converge to the true value function; it only has to be locally accurate.

This observation also hints at a limitation of SHAPE. Eventually, we will want to apply these methods to multistage problems. Multistage problems can be viewed as sequence of two-stage problems, but for future time periods, the initial resource state vector ( R 0 in our two-stage formulation), will be random. If the initial state were random, then we would need an approximation method that would try to approximate a larger portion of the value function. The SHAPE algorithm will not do this.

It is important to note that SHAPE is an optimizing algorithm, which means that the point that we visit for the second stage ( R x 0 ) depends on the approximation we are using. The theorem tells us that no matter how bad our initial approximation is, we will eventually converge to the right answer (we will converge much faster if we start with a good approximation). This is another example of how concavity helps steer us in the right direction.

## 14.5 Separable, piecewise-linear approximations for nondifferentiable problems

Return for a moment to our two-stage resource allocation problem with network recourse shown in figure 14.1. Here, we act on a set of resources (this may represent buying them, making them, moving them, repairing them) to produce a new set of resources which, if we have done our job in the first stage, can handle a broad range of demands that might come up in the second stage. Having the right amount and right type of resources to handle various forms of random demands means we have a robust resource allocation.

Assume for the moment that our demands are discrete (or discretized) and that the demands for resources are also discrete. We would expect that our allocation decisions are also discrete, producing discrete flows of resources into the second stage. The steps in our procedure are depicted in figure 14.4.

We begin with our basic stochastic program which is depicted in figure 14.4a. We assume that we have to make first-stage decisions, after which new information might arrive about demands, prices and costs in the second stage. The mathematical problem is the same as that given in equations (14.3)-(14.10).

<!-- image -->

14.4a: The two-stage problem with stochastic second stage data.

14.4b: Solving the first stage using a separable, piecewise linear approximation of the second stage.

<!-- image -->

14.4c: Solving a Monte Carlo realization of the second stage and obtaining dual variables.

<!-- image -->

Figure 14.4: Steps in estimating separable, piecewise-linear approximations for two-stage stochastic programs.

Wepropose to solve the problem by replacing the second stage using a separable, piecewiselinear approximations of the value function (see figure 14.4b). The technique is the same as what we used in section 12.1.4, but it is useful to briefly review how the approximation is developed in our two-stage setting. We start by assuming that we have an initial value function approximation of the form:

<!-- formula-not-decoded -->

where R x is the post-decision state variable given by:

<!-- formula-not-decoded -->

We might, for example, simply use ¯ V 0 ( R x ) = 0 (the convergence of the algorithm does not depend on the initial starting solution).

Now assume that we are at iteration n and we have an approximation ¯ V n -1 ( R x ) from the previous iteration. Solve the first stage:

<!-- formula-not-decoded -->

For our simple problem, the feasible region X may consist of nothing more than:

<!-- formula-not-decoded -->

Problem (14.34)-(14.35), along with the definitional constraint (14.33), is a simple linear program, typically of modest size. Solving it gives us x n 0 , from which we obtain R x,n which gives us the flows into the second stage.

With R x,n 0 now given, we solve the second stage for a single, Monte Carlo sample, subject to constraints (14.7)-(14.10), as depicted in figure 14.4c). This again is a linear program, typically of modest size (even for fairly large problems). This problem is solved subject to the flow conservation constraint (equation (14.7)) which limits the flow out of a second stage node a to the flow R x,n a into the node from the first stage. From the linear program, we obtain dual variables for these flow conservation constraints. Let ˆ v n a be the dual variable for the resource constraint R x 0 a . We now update our piecewise linear approximation just as we did in section 12.1.4 (see the algorithm in figure 12.3 and figure 12.4).

For many problems, solving the first stage problem using a separable, piecewise linear approximation for the value function is equivalent to solving a network problem. We can

14.5a: The two-stage problem with stochastic second stage data.

<!-- image -->

14.5b: Solving the first stage using a separable, piecewise linear approximation of the second stage.

<!-- image -->

Figure 14.5: Formulating the first-stage problem using a second stage separable, piecewise linear approximation into a network.

use a standard trick from linear programming where piecewise linear functions are converted into a series of parallel links, each representing one of the linear portions of our piecewise linear function, as illustrated in 14.5a. As a result, the first stage problem with second stage value function approximation looks like the network shown in 14.5b.

## 14.6 Benders decomposition

Benders decomposition is perhaps the most widely studied method for solving stochastic linear programs. It is based on the idea that we can approximate a linear program using a series of cuts created by solving the dual of the second stage. The strategy is not typically

presented in the context of approximate dynamic programming, but it represents a way of approximating value functions for dynamic programs that can be formulated as multistage stochastic linear programs.

## 14.6.1 The basic idea

To illustrate, we start with our original two-stage problem

<!-- formula-not-decoded -->

subject to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the value function, given the sample outcome ω ∈ Ω, is given by:

<!-- formula-not-decoded -->

subject to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For our discussion, we will assume that we have a discrete set of sample outcomes Ω, where the probability of outcome ω is given by p ( ω ). Let ˆ v 1 ( ω ) be the dual variable for constraints (14.39). The dual of the second stage problem takes the form:

<!-- formula-not-decoded -->

subject to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The dual is also a linear program, and the optimal ˆ v ∗ 1 ( ω ) must occur at one of a set of vertices, which we denote by V 1 . We immediately obtain the nice property that because we have assumed that our only source of randomness is in the right hand side constraints (14.39)-(14.40), it only shows up in the linear coefficients of the dual. This means the set of vertices V 1 is independent of the outcome ω .

Now, let z ( ω ) be the optimal solution of the dual, which is to say:

<!-- formula-not-decoded -->

where ˆ v ∗ 1 ( ω ) solves (14.41)-(14.43). Since this is the optimal solution, also observe that if

<!-- formula-not-decoded -->

then clearly:

<!-- formula-not-decoded -->

since z ∗ ( ω ) is the best we can do. Furthermore, (14.44) is true for all ˆ v 1 ∈ V 1 , and all outcomes ω . We know from the theory of linear programming that our primal must always be less than or equal to our dual, which means that:

V

0

(

x

0

, ω

)

≤

=

This means that:

<!-- formula-not-decoded -->

where the inequality (14.45) is tight for ˆ v 1 = ˆ v ∗ 1 ( ω ). Now let:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which allows us to write 14.45 in the more compact form:

<!-- formula-not-decoded -->

The right hand side of (14.48) is called a cut since it is a plane (actually, an n -dimensional hyperplane) that represents an upper bound on the value function. Using these cuts, we can replace (14.35) with:

<!-- formula-not-decoded -->

z

z

(ˆ

v

1

(ˆ

v

∗

1

, ω

(

ω

)

)

∀

, ω

v

)

ˆ

1

∈ V

∀

ω

1

∈

,

Ω

ω

∈

Ω

subject to (14.36)-(14.37) plus:

<!-- formula-not-decoded -->

Unfortunately, equation (14.50) can be computationally problematic. The set V 1 may be extremely large, so enforcing this constraint for each vertex is prohibitive. Furthermore, even if Ω is finite, it may be quite large, making summations over the elements in Ω expensive for some applications.

A strategy that overcomes the size of the set V works as follows: We start with an empty set, solve (14.49), and obtain a first stage solution x 1 0 . Using this solution, we solve (14.41) to obtain ˆ v 1 0 ( ω ) for each ω ∈ Ω, from which we compute α 1 1 and β 1 1 . We then set V 1 1 = ( α 1 1 , β 1 1 ). Using this cut, we return to the first stage problem and obtain a new solution, x 2 1 , and return to the second stage to find a new set of duals (ˆ v 1 0 ( ω )) ω ∈ Ω and coefficients ( α 2 1 , β 2 1 ). With each iteration, the set V n is incremented by one element. Thus, instead of having to solve (14.49) subject to the entire set of constraints (14.50), we use only the cuts:

<!-- formula-not-decoded -->

This algorithm is known as the 'L-shaped' algorithm. For a finite Ω, it is convergent. The problem is the requirement that we calculate ˆ v ∗ 1 ( ω ) for all ω ∈ Ω, which means we must solve a linear program for each outcome at each iteration. For most problems, this is computationally pretty demanding.

## 14.6.2 Variations

Two variations of this algorithm that avoid the computational burden of computing ˆ v ∗ 1 ( ω ) for each ω have been proposed. These algorithms vary only in how the cuts are computed and updated. The first is known as stochastic decomposition . At iteration n , let V n be the set of dual vertices that have been computed so far. We solve

<!-- formula-not-decoded -->

for a single outcome ω n 1 . The cut is computed using:

<!-- formula-not-decoded -->

Then, all the previous cuts are updated using:

<!-- formula-not-decoded -->

Step 1. Solve the following master problem :

<!-- formula-not-decoded -->

Step 2. Sample ω n ∈ Ω and solve the following subproblem:

<!-- formula-not-decoded -->

to obtain the optimal dual solution:

<!-- formula-not-decoded -->

Augment the set of dual vertices by:

<!-- formula-not-decoded -->

Step 3. Construct the coefficients α n n and β n n of the n th cut to be added to the master problem by computing

<!-- formula-not-decoded -->

Step 4. Update the previously generated cuts by:

<!-- formula-not-decoded -->

Step 5. Check for convergence; if not, set n := n +1 and return to Step 1.

Figure 14.6: Sketch of the stochastic decomposition algorithm.

The complete algorithm is given in figure 14.6.

Asecond variation, called the CUPPS algorithm (for 'cutting plane and partial sampling' algorithm), updates the cuts using:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Comparing (14.51) to (14.56), it is easy to see that the CUPPS algorithm requires more work per iteration. While the calculation in (14.56) is trivial, it has to be computed for each scenario. For problems with a large number of scenarios, this will require more work. The steps of this algorithm are given in figure 14.7.

L-shaped decomposition, stochastic decomposition, and the CUPPS algorithm offer three contrasting strategies for generating cuts, which are illustrated in figure 14.8. L-shaped decomposition is computationally the most demanding, but it produces tight cuts and will,

Step 1. Solve the following master problem :

<!-- formula-not-decoded -->

Step 2. Sample ω n ∈ Ω, and solve the following dual subproblem :

<!-- formula-not-decoded -->

Augment the set of dual vertices by:

<!-- formula-not-decoded -->

Step 3. Set:

<!-- formula-not-decoded -->

Step 4. Construct the coefficients of the n th cut to be added to the master problem by:

<!-- formula-not-decoded -->

Figure 14.7: Sketch of the CUPPS algorithm.

in general, produce the fastest convergence (measured in terms of the number of iterations). The cuts produced by stochastic decomposition are neither tight nor valid, but steadily converge to the correction function. The cuts generated by CUPPS are valid but not tight.

## 14.6.3 Experimental comparisons

The relative performance of L-shaped decomposition, stochastic decomposition, and CUPPS can be illustrated using a classical two-stage resource allocation problem with network resource. The problem can be viewed as a classical distribution problem where product is shipped to a series of warehouses from which they can be used to serve customers (who can be served by more than one warehouse). If our problem is not too large, we can obtain exact solutions using L-shaped decomposition. For this reason, the problems used only 100 scenarios with 10 to 100 warehouse locations.

The results are shown in table 14.1 which shows the percentage distance from the optimal solution for all three algorithms as a function of the number of iterations. Also shown is the CPU time per iteration. The results show that L-shaped decomposition generally has the fastest rate of convergence, although with substantially higher CPU times, especially in the earlier iterations, where the other techniques loop over small sets of vertices. Stochastic decomposition and CUPPS exhibit similar rates of convergence at first, but CUPPS shows better behavior in the limit. All the methods show progressively worse results as the number of locations increase, suggesting that Benders may have a problem with higher dimensional problems.

<!-- image -->

14.8a: Cuts generated by L-shaped decomposition.

<!-- image -->

14.8b: Cuts generated by the stochastic decomposition algorithm.

<!-- image -->

14.8c: Cuts generated by the CUPPS algorithm.

Figure 14.8: Illustration of the cuts generated by L-shaped decomposition (a), stochastic decomposition (b) and CUPPS (c).

| Locations   | Method   |   Number of independent demand observations |   Number of independent demand observations |   Number of independent demand observations | Number of independent demand observations   | Number of independent demand observations   | Number of independent demand observations   | Number of independent demand observations   | Number of independent demand observations   |   CPU time (sec.) per iteration | CPU time (sec.) per iteration   | CPU time (sec.) per iteration   | CPU time (sec.) per iteration   |
|-------------|----------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| Locations   | Method   |                                       25    |                                       50    |                                      100    | 250                                         | 500                                         | 1000                                        | 2500                                        | 5000                                        |                         100     | 500                             | 1000                            | 5000                            |
| 10          | LSD      |                                        3.3  |                                        0.19 |                                        0    |                                             |                                             |                                             |                                             |                                             |                           0.06  |                                 |                                 |                                 |
|             | CUPPS    |                                        5.84 |                                        4.19 |                                        0.5  | 0.29                                        | 0.00                                        |                                             |                                             |                                             |                           0.008 | 0.032                           | 0.071                           | 2.260                           |
|             | SD       |                                       45.45 |                                        9.18 |                                       11.35 | 2.35                                        | 2.30                                        | 1.12                                        | 0.30                                        | 0.26                                        |                           0.009 | 0.044                           | 0.107                           | 0.798                           |
| 25          | LSD      |                                       19.88 |                                       15.98 |                                        2.14 | 0.11                                        | 0.00                                        |                                             |                                             |                                             |                           0.297 | 0.423                           | 0.269                           | 0.082                           |
|             | CUPPS    |                                        8.27 |                                       13.4  |                                        4.33 | 4.02                                        | 1.47                                        | 0.16                                        | 0.00                                        |                                             |                           0.027 | 0.135                           | 0.317                           | 1.232                           |
|             | SD       |                                       40.55 |                                       29.79 |                                       22.22 | 12.80                                       | 4.24                                        | 4.80                                        | 1.06                                        | 0.95                                        |                           0.02  | 0.094                           | 0.225                           | 1.760                           |
| 50          | LSD      |                                       42.56 |                                       20.3  |                                        6.07 | 1.49                                        | 0.52                                        | 0.04                                        | 0.00                                        |                                             |                           0.95  | 1.980                           | 2.510                           |                                 |
|             | CUPPS    |                                       34.93 |                                        9.91 |                                       19.3  | 11.71                                       | 5.09                                        | 1.38                                        | 0.32                                        | 0.00                                        |                           0.057 | 0.178                           | 0.377                           | 9.382                           |
|             | SD       |                                       43.18 |                                       29.81 |                                       17.94 | 8.09                                        | 5.91                                        | 6.25                                        | 2.73                                        | 1.02                                        |                           0.056 | 0.207                           | 0.466                           | 3.112                           |
| 100         | LSD      |                                       74.52 |                                       29.79 |                                       26.21 | 7.30                                        | 2.32                                        | 0.85                                        | 0.11                                        | 0.02                                        |                           3.906 | 8.012                           | 10.147                          | 12.978                          |
|             | CUPPS    |                                       54.59 |                                       35.54 |                                       23.99 | 17.58                                       | 14.68                                       | 14.13                                       | 5.36                                        | 0.91                                        |                           0.223 | 0.675                           | 1.422                           | 5.227                           |
|             | SD       |                                       62.63 |                                       34.82 |                                       40.73 | 12.14                                       | 15.22                                       | 17.43                                       | 17.49                                       | 9.42                                        |                           0.203 | 0.580                           | 1.305                           | 8.692                           |

Table 14.1: Percent error over optimal (from L-shaped decomposition) with different numbers of locations, based on research reported in Powell et al. (2004).

## 14.7 Why does it work?**

## 14.7.1 Proof of the SHAPE algorithm

The convergence proof of the SHAPE algorithm is a nice illustration of a martingale proof. We start with the Martingale convergence theorem (Doob (1953), Neveu (1975) and Taylor (1990)), which has been the basis of convergence proofs for stochastic subgradient methods (as illustrated in Gladyshev (1965)). We then list some properties of the value function approximation ˆ V ( · ) and produce a bound on the difference between x n and x n +1 . Finally, we prove the theorem. This section is based on Cheung &amp; Powell (2000).

Let ω n be the information that we sample in iteration n , and let ω = ( ω 1 , ω 2 , . . . ) be an infinite sequence of observations of sample information where ω ∈ Ω. Let h n = ω 1 , ω 2 , . . . , ω n be the history up to (and including) iteration n . If F is the the σ -algebra on Ω, and let F n ⊆ F n +1 be the sequence of increasing subsigma-algebras on Ω representing the information we know up through iteration n .

We assume the following:

- (A.1) X is convex and compact.
- (A.2) E V ( x, W ) is convex, finite and continuous on X .
- (A.3) g n is bounded such that ‖ g n ‖ ≤ c 1 .
- (A.4) ˆ V n ( x ) is strongly convex, meaning that

<!-- formula-not-decoded -->

where b is a positive number that is a constant throughout the optimization process. The term b ‖ x -y ‖ 2 is used to ensure that the slope ˆ v n ( x ) is a monotone function of x .

When we interchange y and x in (14.59) and add the resulting inequality to (14.59), we can write

<!-- formula-not-decoded -->

- (A.5) The stepsizes α n are F n measurable and satisfy

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

- (A.6) ˆ V 0 ( x ) is bounded and continuous, and ˆ v 0 ( x ) (the derivative of ˆ V 0 ( x )) is bounded for x ∈ X .

Note that we require that the expected sum of squares be bounded whereas we must impose the almost sure condition that the sum of stepsizes be infinite. We now state our primary theorem.

Theorem 14.7.1 If (A.1) - (A.6) are satisfied, then the sequence x n , produced by algorithm SHAPE, converges almost surely to an optimal solution x ∗ ∈ X ∗ of problem (14.25).

The proof illustrates several proof techniques that are commonly used for these problems. Students interested in doing more fundamental research in this area may be able to use some of the devices in their own work.

To prove the theorem, we need to use the Martingale convergence theorem and two lemmas.

## Martingale convergence theorem

Asequence of random variables { W n } that are H n -measurable is said to be a supermartingale if the sequence of conditional expectations E { W n +1 | H n } exists and satisfies:

<!-- formula-not-decoded -->

Theorem 14.7.2 (p.26, Neveu (1975)) Let W n be a positive supermartingale. Then, W n converges to a finite random variable a.s.

From the definition, W n is essentially the stochastic analog of a decreasing sequence.

## Property of Approximations

In addition to equations (14.59) - (14.60) of assumption (A.4), the optimal solution for problem (14.31) at iteration n can be characterized by the variational inequality:

<!-- formula-not-decoded -->

Furthermore, at iteration k +1,

<!-- formula-not-decoded -->

The first lemma below provides a bound on the difference between two consecutive solutions. The second lemma establishes that V n ( x ) is bounded.

Lemma 14.7.1 The solutions x n produced by algorithm SHAPE satisfy:

<!-- formula-not-decoded -->

where b satisfies equation (14.24).

Proof: Substituting x by x n in (14.62), we have:

<!-- formula-not-decoded -->

Rearranging the terms, we obtain:

<!-- formula-not-decoded -->

Combining (14.60), (14.61), and 0 &lt; α n &lt; 1 gives us:

<!-- formula-not-decoded -->

Applying Schwarz's inequality, we have that

<!-- formula-not-decoded -->

Dividing both sides by ‖ x n -x n +1 ‖ , it follows that ‖ x n -x n +1 ‖ ≤ α n 2 b ‖ g n ‖ .

Lemma 14.7.2 The approximation function ˆ V n ( x ) in iteration n can be written as

<!-- formula-not-decoded -->

where r n is a finite vector.

Proof: The algorithm proceeds by adding linear terms to the original approximation. Thus, at iteration n , the approximation is the original approximations plus the lienar term:

<!-- formula-not-decoded -->

where r n is the sum of the linear correction terms. We just have to show that r n is finite.

When taking the first derivative of ˆ V n ( x ) in equation (14.63), we have

<!-- formula-not-decoded -->

With that, we can write ˆ V n +1 ( x ) in terms of ˆ V 0 ( x ):

<!-- formula-not-decoded -->

Therefore, r n +1 and r n are related as follows:

<!-- formula-not-decoded -->

So, the total change in our initial approximation is a weighted sum of g n -ˆ v 0 ( x n ) and the current cumulative change. Since both g n and ˆ v 0 ( x n ) are finite, there exists a finite, positive vector such that

<!-- formula-not-decoded -->

We can now use induction to show that r n ≤ ˆ d for all n . For n = 1, we have r 1 = a 0 ( g 0 -ˆ v 0 0 ) ≤ a 0 ˆ d . Since a 0 &lt; 1 and is positive, we have r 1 ≤ ˆ d . Assuming that r n ≤ ˆ d , we want to show r n +1 ≤ ˆ d . By using this assumption and the definition of ˆ d , equation (14.64) implies that

<!-- formula-not-decoded -->

We now return to our main result.

## Proof of theorem 14.7.1

Our algorithm proceeds in steps. First we establish a supermartingale that provides a basic convergence result. Then, we show that there is a convergent subsequence. Finally, we show that the entire sequence is convergent. For simplicity, we write ˆ v n = ˆ v n ( x n )

Step 1: Establish a supermartingale for theorem 14.7.2 Let T n = ˆ V n ( x ∗ ) -ˆ V n ( x n ), and consider the difference of T n +1 and T n

:

<!-- formula-not-decoded -->

If we write x ∗ -x n +1 as x ∗ -x n + x n -x n +1 , we get

<!-- formula-not-decoded -->

Consider each part individually. First, by the convexity of ˆ V n ( x ), it follows that

<!-- formula-not-decoded -->

From equation (14.61) and 0 &lt; α n &lt; 1, we know that (I) ≤ 0. Again, from equation (14.61) and 0 &lt; α n &lt; 1, we show that (II) ≥ 0.

For (III), by the definition that g n ∈ ∂V ( x n , ω n +1 ),

<!-- formula-not-decoded -->

where V ( x, ω n +1 ) is the recourse function given outcome ω n +1 . For (IV), lemma 14.7.1 implies that

<!-- formula-not-decoded -->

Therefore, the difference T n +1 -T n becomes

<!-- formula-not-decoded -->

Taking conditional expectation with respect to H n on both sides, it follows that

<!-- formula-not-decoded -->

where T n , α n and x k on the right hand side are deterministic given the conditioning on H n . We replace V ( x, ω n +1 ) (for x = x n and x = x ∗ ) with its expectation ¯ V ( x ) since conditioning on H n tells us nothing about ω n +1 . Since ¯ V ( x n ) -¯ V ( x ∗ ) ≥ 0, the sequence:

<!-- formula-not-decoded -->

is a positive supermartingale. Theorem 14.7.2 implies the almost sure convergence of W n . Thus,

<!-- formula-not-decoded -->

Step 2: Show that there exists a subsequence n j of n such that x n j → x ∗ ∈ X ∗ a.s.

Summing equation (14.66) over n up to n and cancelling the alternating terms of T n gives:

<!-- formula-not-decoded -->

Take expectations of both sides. For the first term on the right hand side, we take the conditional expectation first conditioned on H n and then over all H n :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking the limit as K →∞ and using the finiteness of T n and ∑ ∞ i =0 E { a 2 i } , we have

<!-- formula-not-decoded -->

Since ¯ V ( x n ) -¯ V ( x ∗ ) ≥ 0 and ∑ ∞ n =0 α n = ∞ (a.s.), there exists a subsequence n j of n such that

<!-- formula-not-decoded -->

By continuity of ¯ V , this sequence converges. That is,

<!-- formula-not-decoded -->

Step 3: Show that x n → x ∗ ∈ X ∗ a.s.

Consider the convergent subsequence x n j in Step 2. By using the expression of ˆ V n in lemma 14.7.2, we can write T n j as

<!-- formula-not-decoded -->

where ˆ d is the positive, finite vector defined in (14.65). When x n j → x ∗ , both the terms ˆ d | x n j -x ∗ | and ˆ V 0 ( x n j ) -ˆ V 0 ( x ∗ ) (by continuity of ˆ V 0 ) go to 0. Since T n j is positive, we obtain that T n j → 0 a.s. Combining this result and the result in Step 1 (that T n converges to a unique nonnegative T ∗ a.s.), we have T n → T ∗ = 0 a.s. Finally, we know from the strong convexity of ˆ V n ( · ) that

<!-- formula-not-decoded -->

Therefore, x n → x ∗ a.s.

## 14.8 Bibliographic notes

General stochastic programming

Dantzig &amp; Ferguson (1956), Birge &amp; Louveaux (1997), Infanger (1994), Kall &amp; Wallace (1994), Sen &amp; Higle (1999), Wets (1989)

Ermoliev (1988)

Scenario methods: Rockafellar &amp; Wets (1991), Mulvey &amp; Ruszczy´ nski (1995),

Benders: Van Slyke &amp; Wets (1969), Higle &amp; Sen (1991), Chen &amp; Powell (1999), Infanger &amp; Morton (1996)

Network recourse: Wallace (1987), Wallace (1986 a ), Wallace (1986 b )

Bounds based on separable approximations: Birge &amp; Wets (1989), Birge &amp; Wallace (1988)

Ruszczy´ nski (1980 b ), Ruszczy´ nski (1987),

## Exercises

- 14.1) Consider the stochastic projection algorithm represented by equations (14.13)-(14.14). Assume that an element of ˜ x n is negative and we require x ≥ 0 (with no other con-

- straints). Show that the nearest point projection back to the feasible region requires nothing more than setting negative elements of ˜ x n to zero.
- 14.2) Assume that our only constraint is the requirement that ∑ d ∈D x d = R (we are not requiring nonnegativity) and equation (14.13) produces a vector ˜ x n that no longer satisfies this constraint. Show that one iteration of equations (14.18)-(14.21) produces a nearest point projection onto the feasible region.

## Chapter 15

## General asset management problems

We now have the infrastructure to handle what would appear to be a much harder class of dynamic programs: the management of multiple asset classes over time. This problem class covers a vast array of applications. We allow for a variety of asset classes, but we assume that this number is not too big (possibly hundreds or thousands but not millions). Assets may be consumed after satisfying a demand (as with electric power or consumer products) or they may be reusable (money, equipment, people). Assets may be modified (bought, sold, built, moved or changed) to satisfy future demands, and these changes may require time to complete. The examples provide some illustrations.

For this problem class, state variables, exogenous information processes, and decision variables are all vectors. The techniques for discrete state and action problems simply do not work here. Fortunately, we have all the basic machinery we need to solve this problem class.

## 15.1 A basic model

We assume that we have a set of assets with each characterized by:

- a = The attribute vector of an asset.

A = The space of possible attribute vectors.

Assets can be used to satisfy demands, which are characterized by:

b = The attribute vector of a demand.

B = The space of possible demand attribute vectors.

In this chapter, a and b can be vectors, but we assume that the attribute spaces A and B are discrete and 'not too big.' At the time of this writing, this means that they may be

Example 15.1: A fleet of containers must be managed to meet current and future customer demands. There are a variety of container types with some ability for substitution between demands. These are moved around the world, primarily by ocean and rail, and it can take weeks to move between some pairs of locations. As a result, the company has to move containers now to meet demands that may arise in the future.

Example 15.2: The general manager of a baseball team has to decide whether to sign a player to a multiyear contract and for what duration. The player could emerge as a star, or he may perform below expectations (or be injured). If the team signs the player to a short contract, they face the prospect of dealing with a free agent in a short period of time. If they commit to a long contract, they may lock themselves into paying a player who does not work out. Finally, decisions about one player have to consider the salaries, contracts, and substitution opportunities for all the positions at the same time.

- Example 15.3: An electronics manufacturer has to determine how much product to ship at each stage of the supply chain. The decisions have to consider random future demands, shipping delays, and potential production problems.

Example 15.4: The unit commitment problem for the electric power industry poses the problem of determining which power units to turn off or on and, if they are on, at what level of production. The decisions have to balance the cost of turning units on and off right now against the value of having them be in a particular state in the future.

in the thousands. An asset attribute vector might contain a geographical location, one or more descriptors of the attributes of the asset itself, and the time at which it can actually be used (the actionable time ). A demand attribute vector might specify the attributes of the best asset for satisfying the demand as well as a list of other asset types that are reasonable substitutions.

We write our asset and demand state vectors using:

R A ta = The number of assets with attribute a that we know about at time t after we make a decision.

<!-- formula-not-decoded -->

R D tb = The number of demands with attribute b that we know about at time t after we make a decision.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that we are starting with the assumption that we are using a post-decision state variable. The resource state vector R t captures the status of not only the assets we are managing, but also the demands that we are serving. Although we require that the sets A and B be discrete, the resource vector R t may be either discrete or continuous. Over time, assets and demands may arrive to or depart from the system as a result of an exogenous process. These

are modeled using:

ˆ R A ta ( R t -1 , ω ) = The number of assets of type a that arrive to or depart from the system during time interval t .

ˆ R D tb ( R t -1 , ω ) = The number of demands of type b that arrive to or depart from the system during time interval t .

<!-- formula-not-decoded -->

ˆ R A and ˆ R D are functions that depend on both the current state R t -1 and the sample realization ω . If we did not allow random departures from the system, we could drop the dependence on the resource vector R t -1 and treat ˆ R A and ˆ R D as purely exogenous information processes. In many applications, the only source of randomness is the exogenous arrivals of new demands.

We assume that we can act on assets in order to change their attributes. For simplicity, we assume that demands can be served (at which point they leave the system). If they are not served, we may assume that they either leave the system or are held to the next time period. In actual applications, it may be possible to act on a demand by itself. Typically, this refers to outsourcing. While these features help make our model more realistic, they complicate the dynamic programming principles that we are trying to describe.

We act on our assets with decisions that are modeled as follows:

D A = The set of decisions that act directly to modify the attributes of an asset without serving a demand.

D D = The set of decisions to act on demands, where each element of D D corresponds to a decision to act on a particular type of demand in the set B .

d φ = The decision to do nothing with an asset.

D = D A ∪ D D ∪ d φ

An element d ∈ D represents a type of decision. For our baseball players, the decision d ∈ D A could represent the length of a contract for a player, while a decision in D D is the decision to give a player a particular starting position (e.g. center field). If we are managing containers, a decision in D A would represent moving the container empty, while an element of D D would represent the decision to serve a type of traffic moving between a particular origin-destination pair.

The quantity of assets that are acted on is given by:

x tad = The number of assets with attribute a that are acted on with decision d using the information available at time t .

x t = ( x tad ) a ∈A ,d ∈D

Decisions have to reflect flow conservation constraints on assets:

<!-- formula-not-decoded -->

In addition, we cannot act on more assets than we have. We assume that we need one asset per demand. This means that we also have to satisfy the constraint:

<!-- formula-not-decoded -->

wbere b d represents the attributes of the demand that decision d ∈ D D are acting on. Finally, we have the usual nonnegativity constraints:

<!-- formula-not-decoded -->

Equations (15.1)-(15.3) constitute our feasible region, which we denote by X t . For the moment, we assume that we have some function for making decisions:

<!-- formula-not-decoded -->

We assume that we have a well-defined class of decision functions ( X π ) π ∈ Π where, as before, we assume that the set Π is 'compact' (mathematically, this means that it is closed and bounded). Keep in mind that a value function represents a form of policy, so the set of all possible value functions constitutes a set of policies.

We already introduced the idea that an element of an attribute vector could be when the asset (or demand) is actionable. If d ∈ D D , then the decision d means to serve a demand of type b d . But what if the asset or the demand type is not actionable right now? We can introduce the idea of decisions that are made now (with the information available now) to be implemented in the future. We assume that these decisions are locked into place, producing modified assets (and possibly demands) that are acted on after they become actionable.

The effect of a decision on an asset is captured using the indicator function:

<!-- formula-not-decoded -->

Keeping in mind that the attribute vector includes the actionable time, we may write our transition equation using:

<!-- formula-not-decoded -->

If we assume that our demands leave the system if they are not immediately served, then the post-decision state variable would never have any demands, which means that R D t = 0. If unserved demands are held until the next time period, then we would write:

<!-- formula-not-decoded -->

Recall that for each d ∈ D D there is a demand type with attribute vector b , so choosing decision d ∈ D D is the same as choosing to use an asset to serve a demand with attribute vector b d . It is also important to note that from the perspective of solving this problem as a dynamic program, making the assumption that unserved demands are lost significantly simplifies the problem.

Equations (15.4)-(15.5) define our transition function, which we write generally as:

<!-- formula-not-decoded -->

We assume that each decision produces a contribution:

c tad = The contribution generated by acting on a resource with attribute a with decision d using the information available at time t .

<!-- formula-not-decoded -->

Our optimization problem, then, is to solve:

<!-- formula-not-decoded -->

At this stage, we have a general optimization problem which, by now, should look quite familiar. What is different is that everything is a vector, possibly with hundreds or even thousands of dimensions.

## 15.2 Sample applications

There is a vast array of problems that can be modeled as resource allocation problems. In this section we illustrate a few by presenting the attribute vector and discussing the transition function.

Figure 15.1: Oil companies have to plan both storage capacity as well as oil supplies.

<!-- image -->

## Managing oil reserves

Oil companies have to plan contracts and inventories for years in the future. This requires anticipating the range of potential demands for oil and the evolution of prices for years into the future. Taking these into account, the company has to sign contracts guaranteeing reasonable prices to cover demand, with sufficient inventory to handle the possibility of having more oil on hand than they are able to sell.

From a high level planning perspective, the attributes of oil might be represented by:

<!-- formula-not-decoded -->

In addition to managing the oil itself, the oil company also has to maintain adequate storage capacity. Oil cannot be stored without the proper tanks and supporting infrastructure (pipelines, catch basins in case of spills, support personnel). The storage tanks themselves are a type of asset, which might be described using

<!-- formula-not-decoded -->

## The container management problem

Almost all global trade takes place in metal boxes called containers which carry freight on ships, trains and trucks. Containers come in a variety of shapes and sizes, although they are

Figure 15.2: Containers in a Hong Kong port.

<!-- image -->

always designed so they can be stacked on container ships. Most are 20 feet long, but some are 40 feet. They may have refrigeration units, and there are a variety of differences in how they are configured.

<!-- formula-not-decoded -->

The first two attributes are static (they do not change over time), while the last three are dynamic. For example, after delivering an empty container to a shipper, the company may be told that there is a problem with the container (it may be dirty, or a hinge may not allow a door to close properly). This information often only becomes available after a customer inspects the equipment.

## Electric power transformers

Electric power companies have to use massive transformers to convert high voltage lines (500,000 and 750,000 volts) to lower voltages for distribution to customers. These transformers are expensive (one or two million dollars each), heavy (over 200 tons for the largest), may require over a year to build and are difficult to transport. The decisions a company faces is when to purchase a transformer (and implicitly, with what features), when to move a transformer from one location to another, when to inspect a transformer (transformers go through nonuniform deterioration as a result of power surges) and whether or not to

Figure 15.3: Electric power companies have to puchase expensive transformers to convert high voltage lines to lower voltage lines.

<!-- image -->

rehabilitate a unit. Attributes of a transformer include

<!-- formula-not-decoded -->

## 15.3 A myopic policy

While it is nice to find an optimal solution, it is useful to take a look at a simple myopic heuristic, especially since this is how many of these problems are solved in practice. In a myopic heuristic, we use only what we know at time t . At time t , this involves solving:

<!-- formula-not-decoded -->

subject to (15.1)-(15.3). The problem is depicted in figure 15.4. We can either assign assets to demands or act on them with a decision in the set D A (for example, move a container empty to another location, repair a piece of equipment). Myopic models work best when the only decisions involve using assets to perform tasks which generate some sort of contribution. We also capture 'do nothing' decisions using arcs into a supersink. A demand of type b is modeled as an arc into the supersink with an upper bound of R b .

Figure 15.4: Graphical illustration of a myopic asset allocation problem.

<!-- image -->

As long as our number of asset types and demand types is not too large, our myopic problem can generally be solved as a linear program using a commercial solver. In practice, the major computational burden tends to be computing the contributions c tad rather than actually solving the linear program itself. The linear program will return dual variables ˆ v ta for each asset constraint (15.1).

Can a myopic model work well? Possibly. In fleet management problems, we may assign containers to known orders, which may require moving a container empty to the origin of the order. A number of engineering systems work this way. A clear limitation of this model is that it would never recommend that a container should be moved to another location because an order might arise at a location (humans routinely make such judgments). Thus, a myopic solution would work reasonably well but would certainly not be optimal.

A myopic model works poorly when decisions about assets have to be made based purely on what might happen in the future. An electric power company has to decide to build power plants to meet future demands. A decision d ∈ D A represents a decision to purchase a very specific type of power plant, whereas decisions in D D represent decisions to satisfy demands that are known right now. It is not possible to build a power plant to meet demands right away, so it would result in a solution where demands are going unsatisfied for a year or more. The problem even arises in short-range decisions. Our same power utility faces power demands that change hourly. While the demands follow a pattern, there are fluctuations due to weather changes as well as occasional equipment failures. It can take several hours to turn a unit on, so the utility cannot wait for power surges before turning a unit on.

## 15.4 An approximate dynamic programming strategy

In this section we describe some approximation strategies that can provide very high quality solutions for these more complex resource allocation problems. We start by writing the optimality equations around the post-decision state variable:

<!-- formula-not-decoded -->

We define an approximate policy by introducing a value function approximation, producing the decision function:

<!-- formula-not-decoded -->

which is solved subject to constraints (15.1)-(15.3). Our challenge now is finding a good value function approximation. We have already observed that our myopic policy can be solved as a linear program. This imposes a constraint on the types of value function approximations that we can consider. If the myopic policy can be solved as a linear program, it is very desirable to create value function approximations that allow the resulting problem to also be solved as a linear program. Fortunately, there are several options that have this property.

## 15.4.1 A linear approximation

The simplest approximation strategy outside of a myopic policy (equivalent to ¯ V t ( R t ) = 0), is one that is linear in the resource state:

<!-- formula-not-decoded -->

Using equation 15.4 to write R ta ′ as a function of x t , we obtain our decision function:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall that a M ( t, a, d ) is the terminal attribute function (see (3.15)) which gives the attributes We quickly see that including a linear value function approximation produces a problem with the same structure as our myopic problem. The only difference is that instead of a

coefficient of c tad , we now use c tad + γ ¯ v t,a M ( t,a,d ) which includes a term that captures the marginal, downstream value of an asset after it has been acted on.

Solving X π ( R t -1 , ˆ R t ( ω )) is a linear program, and therefore returns a dual variable ˆ v ta for each flow conservation constraint. We can use these dual variables to update our linear approximation using:

<!-- formula-not-decoded -->

Linear (in the resource state) approximations are especially easy to develop and use, but they do not always provide good results. They are particularly useful when managing complex assets where R ta tends to be small (say, 0 or 1). This tends to happen when managing relatively complex assets such as people, locomotives or aircraft. In general, a linear approximation will work well if the size of the attribute space is much larger than the number of discrete resources being managed, although even these problems can have pockets where there are a lot of resources of the same type. For example, a major trucking company keeps drivers at a home location (their domicile). These drivers are then dispatched to move goods in a region before returning home. The drivers at home tend to all have the same attribute vector, even though the entire attribute space may be extremely large.

Linear approximations can also work well when the value function is approximately linear. For example, purchasing oil or agricultural commodities on the open market, or selling stocks in the financial markets, can represent problems where the value of additional resources is approximately linear. If the quantities are large enough (for example, a major mutual fund trying to sell 50 million shares of a stock on a single day), the nonlinear behavior of the problem becomes more apparent, but a linear approximation may provide good results.

## 15.4.2 A separable, piecewise linear approximation

For many problems, we are trying to move enough resources to satisfy forecasted demands. If we provide too many resources, the marginal value of the additional resources can be zero or negative. If we provide too few, we have unsatisfied demand and the value of additional resources can be high. For these problems, we need to capture nonlinear behavior.

One approximation strategy that has proven very effective is the use of separable, piecewise linear approximations which were described in section 14.5 for two stage problems. This strategy used the same piecewise linear approximation first introduced in section 12.1.4 for the asset acquisition problem. We can use this strategy for the much larger and more difficult problems in this chapter, adapted for multistage applications. Figure 15.5 outlines the steps of the procedure, which is just a merger of our basic single-pass forward dynamic programming algorithm (see figure 7.1) and our approximation strategy for two-stage problems (described in section 14.5).

The algorithm is depicted in figure 15.6. It requires solving a linear program to allocate resources at each time period, using separable, piecewise linear approximations of the value

```
Step 0: Initialization: Step 0a. Initialize ¯ V 0 t , t ∈ T . Step 0b. Set n = 1. Step 0c. Initialize R 1 0 . Step 1: Choose a sample path ω n . Step 2: Do for t = 1 , 2 , . . . , T : Step 2a: Let: ω n t = W t ( ω n ). Step 2b: Solve: x n t = arg max x t ∈X t ( ω ) { c t x t + ∑ a ′ ∈A ¯ V n -1 ta ′ ( R ta ′ ( x t )) } and let ˆ v n ta be the dual variable for the resource constraint for all a ∈ A . Step 2c: Use the dual variable ˆ v n ta to update the value function ¯ V n -1 t -1 ,a using the techniques of section 11.3. Step 2d: Compute: R n t = R M ( R n t -1 , ω n t , x n t ) . Step 3. Increment n . If n ≤ N go to Step 1. Step 4: Return the value functions ( ¯ V n t ) T t =1 .
```

Figure 15.5: The SPAR algorithm for general multistage resource allocation problems.

of resources in the future. The process steps forward in time, using value function approximations for assets in the future. The dual variables from each linear program are used to update the value function for the previous time period.

A critical assumption is that we obtain ˆ v ta for all a ∈ A . If we were managing a single asset, this would be equivalent to sampling the value of being in all possible states at every iteration. In chapter 5, we referred to this strategy as synchronous approximate dynamic programming (see section 5.6.2).

With relatively little work, we are now solving large scale resource allocation problems. This method has been applied to large freight transportation companies managing hundreds or thousands of drivers (trucking), containers (railroads) and aircraft (military airlift).

## 15.4.3 Network structure, multicommodity problems and the Markov property

In section 14.5 we showed that if we use a separable, piecewise linear approximation for the value function that our problem at each time t reduces to a pure network as shown in figure 14.5. A pure network is any linear program where the constraints consist only of flow conservation (flow into a node has to equal flow out of a node), nonnegativity and

N: V:

<!-- image -->

15.6a: The approximate linear program for time t .

<!-- image -->

15.6b: The approximate linear program for time t +1.

<!-- image -->

15.6c: The approximate linear program for time t +2.

Figure 15.6: The SPAR algorithm for general, multistage resource allocation problems.

upper bounds on links. One useful property of pure networks is that if the data is integer (specifically the flows supplies, demands and upper bounds) then the optimal solution (more precisely, the solution returned by a linear programming package) will be integer.

It is useful to take a closer look at this property and the conditions under which it holds. Recall that our problem at time t requires solving decision problems of the form

<!-- formula-not-decoded -->

If our only constraints are the flow conservation constraint (equation (15.1)), nonnegativity and possibly upper bounds on individual flows ( x tad ≤ u tad ), then equation (15.9) is a pure network. What may destroy this structure are equations of the form

<!-- formula-not-decoded -->

Equations such as this are known as bundle constraints since they bundle different decisions together. These constraints tell the problem that we can only serve as many tasks as are there to be served. Often, different types of assets can be used to serve a task, and after serving the task, they are still different, and their value to the system is different. The situation is depicted in figure 15.7a.

If you do not care about integer solutions, it does not matter if the problem is a pure network or not. If you do care about integrality, the property that solving pure networks (with integer data) always produces integer solutions can be quite handy. One way to restore the pure network property is to aggregate the downstream value function so that all the flow serving the task lands has the same aggregated attribute (the aggregation is performed purely for the value function approximation). This produces the network shown in figure 15.7b which is a pure network.

Experimental work has demonstrated that even if we lose the pure network property, solving a continuous linear program for these problems still produces integer solutions the vast majority of the time. For this reason, aggregation should be used primarily for statistical reasons rather than the ability to obtain integer solutions.

Another way to retain a pure network structure is to use value function approximations that are linear in the resource state. This means that we are solving problems with the structure shown in figure 15.8a, which can be easily transformed to the equivalent network shown in figure 15.8b where all we have done is to take the slope of the linear value function for each downstream resource type (for example, the slope v 1 for resources that have attribute a ′ 1 in the future) and add this value to the arc assigning the resource to the demand. So, if the cost of assigning resources with attribute a 1 to the demand has a cost of c 1 , then we would use a modified cost of c 1 + v 1 .

15.7a: A multicommodity flow subproblem with disaggregate, piecewise linear value functions.

<!-- image -->

15.7b: A multicommodity flow subproblem with an aggregate piecewise linear value functions.

<!-- image -->

Figure 15.7: When resources with different attributes can serve the same task, the result is a linear program which is not a network (15.7a). If resources are aggregated after the task is finished, the pure network structure is restored (15.7b).

## 15.5 Some numerical experiments

In this section we summarize a body of research conducted in the context of managing fleets of vehicles. These are problems that can be modeled deterministically or stochastically. The deterministic problems can be solved using a linear programming code, giving us a benchmark against which we can compare our approximation. We also consider two important problem classes: single commodity and multicommodity. With single commodity formulations, we assume there is only one type of asset being managed (for example, a type of trailer or container). Assets at a physical location i can only serve customer requests to move freight from location i to some other location. With multicommodity problems, there may be different types of equipment as well as different types of customer requests where a customer might prefer one type of equipment but is willing to accept other types of equipment (there is typically a cost associated with substitution). In addition, a particular customer might be willing to accept only a subset of other equipment types.

15.8a: A multicommodity flow subproblem with linear value functions value functions.

<!-- image -->

,

15.8b: Conversion to an equivalent problem which is a pure network.

<!-- image -->

Figure 15.8: If linear vlaue functions are used (15.8a) the problem can be converted into an equivalent pure network (15.8b).

## 15.5.1 Experiments for single commodity flow problems

Our first experiments are for the simplest type of resource allocation problem, known as a single commodity flow problem. Imagine that we are flowing fleets of identical vehicles which may move loaded or empty between points in space and time. Customer requests are represented as capacitated arcs with a positive contribution, while empty movements have negative contributions and no upper bound. The resulting linear program is illustrated in figure 15.9. This problem is easily solved using a linear programming code. Since the resulting linear program has the structure of what is known as a pure network , it is also the case that if the flows (and upper bounds) are integer, then the resulting solution is integer.

We first consider a problem where the demands are deterministic. By this we mean that each time we take a sample of the random demand, we obtain the same number. We can solve this problem using our piecewise linear approximations. Although the problem is deterministic, if we are solving the problem at time t , we model all customer requests that start at times t ′ &gt; t as if they were unknown. It just means that each time we take a random sample of new information (step 2a in figure 15.5) it means we always sample the same information. We can also put the entire problem (over all time periods) into a linear

Figure 15.9: Illustration of a pure network for time-staged, single commodity flow problems.

<!-- image -->

Table 15.1: Percentage of the optimal solution.

|           | Simulation Horizon   | Simulation Horizon   | Simulation Horizon   |
|-----------|----------------------|----------------------|----------------------|
| Locations | 15                   | 30                   | 60                   |
| 20        | 100.00%              | 100.00%              | 100.00%              |
| 40        | 100.00%              | 99.99%               | 100.00%              |
| 80        | 99.99%               | 100.00%              | 99.99%               |

programming package to obtain the optimal solution.

Experiments (reported in Godfrey &amp; Powell (2002 a )) were run on problems with 20, 40 and 80 locations, and 15, 30 and 60 time periods. The results are reported in table 15.1 as percentages of the optimal solution produced by the linear programming algorithm. It is not hard to see that the results are very near optimal. We know that separable, piecewise linear approximations do not produce provably optimal solutions (even in the limit) for this problem class, but it appears that the error is extremely small. We have to note, however, that a good commercial linear programming code is much faster than iteratively estimating value function approximations.

We now consider what happens when the demands are truly stochastic, which is to say that we obtain different numbers each time we sample information. For this problem, we do not have an optimal solution. Although this problem is relatively small (compared to true industrial applications), formulated as a Markov decision process produces state spaces that are far larger than anything we could hope to solve. We can use Benders decomposition (section 14.6) for multistage problems, but experiments have shown that the rate of convergence is so slow that it does not produce reasonable results. Instead, experimental research has found that the best results are obtained using a rolling horizon procedure where at each time t , we combine the demands that are known at time t with expectations of any random demands for future time periods. A deterministic problem is then solved over a planning horizon of length T ph which typically has to be chosen experimentally.

Figure 15.10: Percentage of posterior bound produced by a rolling horizon procedure using a point forecast of the future versus an approximate dynamic programming approximation.

<!-- image -->

For a specific sample realization, we can still find an optimal solution using an linear programming solver, but this solution 'cheats' by being able to use information about what is happening in the future. This solution is known as the posterior bound since it uses information that only becomes known after the fact.

The results of these comparisons are shown in figure 15.10. The experiments were run on problems with 20, 40 and 80 locations, and with 100, 200 and 400 vehicles in our fleet. Problems with 100 vehicles were able to cover roughly half of the demands, while a fleet of 200 vehicles could cover over 90 percent. The fleet of 400 vehicles was much larger than would have been necessary. The ADP approximation produced better results across all the problems, although the difference was most noticeable for problems where the fleet size was not large enough to cover all the demands. Not surprisingly, this was also the problem class where the posterior solution was relatively the best.

## 15.5.2 Experiments for multicommodity flow problems

Experiments for multicommodity flow problems (reported in Topaloglu &amp; Powell (to appear)) assumed that it was possible to substitute different types of equipment for different types of demands at a single location as illustrated in figure 15.11. This means that we cannot use a vehicle at location i to serve a customer out of location j , but we can use a vehicle of type k ∈ K to serve a demand of type /lscript ∈ K (we assume the demand types are the same as vehicle types). Different substitution rules can be considered. For example, a customer type might allow a specific subset of vehicle types (with no penalty). Alternatively, we can assume that if we can provide a vehicle of type k to serve a customer of type k , there is no

Table 15.2: Different substitution patterns used in the experiments.

| 1 0       | .8   | .5   | .3   | 0   | 1   | 0   | 0   | 0   | 1   | .5   | 0   | 0   | 0   | 1   |    | 1 1   | 1   | 1   |
|-----------|------|------|------|-----|-----|-----|-----|-----|-----|------|-----|-----|-----|-----|----|-------|-----|-----|
| .7 0      | 1    | .8   | .3   | 0   | 1   | 1   | 0   | 0   | .5  | 1    | .5  | 0   | 0   | 1   | 1  | 1     | 1   | 1   |
| .6 0      | .6   | 1    | .5   | .5  | 1   | 1   | 1   | 0   | 0   | .5   | 1   | .5  | 0   | 1   | 1  | 1     | 1   | 1   |
| 0 0       | .4   | .7   | 1    | .5  | 1   | 1   | 1   | 1   | 0   | 0    | .5  | 1   | .5  | 1   | 1  | 1     | 1   | 1   |
| 0         | .4   | .6   | .6   | 1   |     | 1   | 1 1 | 1 1 | 0   | 0    | 0   | .5  | 1   | 1   | 1  | 1     | 1   | 1   |
| II III IV |      | I    |      |     |     |     |     |     |     |      |     |     |     |     |    |       |     |     |

/negationslash substitution penalty, but there is a penalty (in the form of receiving only a fraction of the reward for moving a load of freight) if we use a vehicle of type k to serve a customer of type /lscript ( k = /lscript ).

Four substitution matrices were used (all datasets used five equipment types and five demand types). These are given in table 15.2. Matrix S I allows all forms of substitution, but discounts the revenue received from a mismatch of equipment type and demand type. Matrix S II would arise in settings where you can always 'trade up'; for example, if we do not have equipment type 3, you are willing to accept 4 or 5. Matrix S III limits substitution to trading up or down by one type. Matrix S IV allows all types of substitution without penalty, effectively producing a single commodity problem. This problem captures the effect of using an irrelevant attribute when representing a resource class.

Figure 15.11: Illustration of substitution possibilities for the multicommodity flow formulation.

<!-- image -->

Table 15.3: Characteristics of the test problems.

| Problem   |   T |   |I| |   |K| |   |D| |   F |   Demands |   c |   r | S   |
|-----------|-----|-------|-------|-------|-----|-----------|-----|-----|-----|
| Base      |  60 |    20 |     5 |     5 | 200 |      4000 | 4   |   5 | I   |
| T 30      |  30 |    20 |     5 |     5 | 200 |      2000 | 4   |   5 | I   |
| T 90      |  90 |    20 |     5 |     5 | 200 |      6000 | 4   |   5 | I   |
| I 10      |  60 |    10 |     5 |     5 | 200 |      4000 | 4   |   5 | I   |
| I 40      |  60 |    40 |     5 |     5 | 200 |      4000 | 4   |   5 | I   |
| C II      |  60 |    20 |     5 |     5 | 200 |      4000 | 4   |   5 | II  |
| C III     |  60 |    20 |     5 |     5 | 200 |      4000 | 4   |   5 | III |
| C IV      |  60 |    20 |     5 |     5 | 200 |      4000 | 4   |   5 | IV  |
| R 1       |  60 |    20 |     5 |     5 |   1 |      4000 | 4   |   5 | I   |
| R 5       |  60 |    20 |     5 |     5 |   5 |      4000 | 4   |   5 | I   |
| R 400     |  60 |    20 |     5 |     5 | 400 |      4000 | 4   |   5 | I   |
| R 800     |  60 |    20 |     5 |     5 | 200 |      4000 | 4   |   5 | I   |
| c 1.6     |  60 |    20 |     5 |     5 | 200 |      4000 | 1.6 |   5 | I   |
| c 8       |  60 |    20 |     5 |     5 | 200 |      4000 | 8   |   5 | I   |

Each dataset was characterized by the following parameters:

T = Number of time periods. I = Set of locations K = Set of vehicle types D = Set of demand types F = Fleet size D = Total number of demands to be served over the horizon c = Cost per mile for moving a vehicle empty r = Contribution per mile for moving a load

- S = Choice of substitution matrix: I, II, III, IV

A series of datasets were created by choosing a single base problem and then modifying one attribute at a time (e.g. the length of the horizon) to test the effect of that parameter. Table 15.3 summarizes all the datasets that were created.

The results of the experiments are shown in table 15.4 which gives the max (highest objective function as a percent of the optimal), min, mean, standard deviation, and the CPU time (and iterations) to reach solutions that are 85 th , 90 th , 95 th , 97 . 5 th percent of the optimal solution. It is important to note that the approximate dynamic programming solutions were always integer, whereas the linear programming optimal solution was allowed to produce fractional solutions. If we required integer solutions, the resulting integer program would be quite large and hard to solve. As a result, some of the gap between the ADP approximation and the optimal solution can be attributed to the relaxation of the integrality requirement. Recognizing that all experimental tests depend to a degree on the structure of the problem and the choice of problem parameters, the results seem to suggest that the use of separable, piecewise linear approximations are giving high quality results.

Table 15.4: Performance of ADP approximation on deterministic, multicommodity flow datasets expressed as a percent of the linear programming solution (from Topaloglu &amp; Powell (to appear)).

| Problem   | Max.   | Min.   | Mean   | Std. dev.   |   Time (sec.) |   Time (sec.) |   to reach | to reach   |   No. iterations for |   No. iterations for |   No. iterations for | No. iterations for   | Time (s) per iter.   |
|-----------|--------|--------|--------|-------------|---------------|---------------|------------|------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Problem   | Max.   | Min.   | Mean   | Std. dev.   |            85 |            90 |         95 | 97.5       |                   85 |                   90 |                   95 | 97.5                 | Time (s) per iter.   |
| Base      | 98.90  | 98.65  | 98.76  | 0.062       |            37 |           101 |        248 | 506        |                    2 |                    4 |                    8 | 14                   | 46.6                 |
| T 30      | 98.58  | 98.12  | 98.37  | 0.130       |            19 |            35 |        165 | 276        |                    2 |                    3 |                   10 | 15                   | 24.9                 |
| T 90      | 98.89  | 98.64  | 98.75  | 0.055       |            56 |           106 |        276 | 721        |                    2 |                    3 |                    6 | 13                   | 74.9                 |
| I 10      | 99.75  | 99.55  | 99.65  | 0.039       |            11 |            11 |         30 | 63         |                    2 |                    2 |                    4 | 7                    | 14.5                 |
| I 40      | 98.90  | 98.49  | 98.52  | 0.211       |           194 |           530 |        992 | 2276       |                    3 |                    6 |                    9 | 17                   | 154.2                |
| S I       | 98.90  | 98.65  | 98.76  | 0.062       |            37 |           101 |        248 | 506        |                    2 |                    4 |                    8 | 14                   | 46.6                 |
| S II      | 99.51  | 99.17  | 99.34  | 0.083       |            59 |            59 |        433 | 991        |                    2 |                    2 |                    7 | 13                   | 75.5                 |
| S III     | 98.61  | 98.23  | 98.41  | 0.092       |            33 |            88 |        374 | 505        |                    2 |                    4 |                   12 | 15                   | 44.8                 |
| S IV      | 99.86  | 99.73  | 99.75  | 0.032       |           235 |           287 |        479 | 938        |                    4 |                    5 |                    9 | 14                   | 82.4                 |
| R 100     | 96.87  | 96.16  | 96.48  | 0.189       |            66 |           384 |        475 |            |                    3 |                   12 |                   14 |                      | 50.2                 |
| R 400     | 99.52  | 99.33  | 99.43  | 0.045       |            40 |            40 |        140 | 419        |                    2 |                    2 |                    5 | 12                   | 48.8                 |
| c 1.6     | 99.13  | 98.72  | 90.01  | 0.009       |            33 |            33 |         54 | 602        |                    2 |                    2 |                    3 | 13                   | 47.2                 |
| c 8       | 98.55  | 98.11  | 98.36  | 0.092       |            95 |           209 |        431 | 1274       |                    4 |                    7 |                   12 | 30                   | 43.7                 |

The table also shows the number of iterations and the CPU time per iteration. Caution is required when comparing CPU times against those of competing algorithms. There are many applications where it is possible to use trained value functions. In these settings, a solution may be obtained using a single forward pass through a new dataset. Real-time (sometimes known as 'on-line') applications may run continuously, where each forward pass uses updated data.

The same experiments were run using stochastic demand data. As with the single commodity dataset, we used two forms of comparison. First, we compared the results obtained using the approximate dynamic programming approach to those obtained using a point forecast of the future over a planning horizon. Then, for each sample path, we computed the optimal solution using all the information. All results are reported as a percentage of the posterior optimal solution.

The results are shown in figure 15.12. Again we see consistently better solutions across all the problem variations.

## 15.6 Bibliographic notes

Powell et al. (2004), Topaloglu &amp; Powell (to appear), Godfrey &amp; Powell (2002 a ), Godfrey &amp; Powell (2002 b )

Powell et al. (2002)

Spivey &amp; Powell (2004)

Figure 15.12: Percentage of posterior bound produced by a rolling horizon procedure versus an approximate dynamic programming approximation when using stochastic demand data.

<!-- image -->

## Bibliography

- Aalto, S. (2000), 'Optimal control of batch service queues with finite service capacity and linear holding costs', Mathematical Methods of Operations Research 51 , 263-285. 373
- Adelman, D. (2004), Price directed control of a closed logistics queueing network, Tech. rep., University of Chicago. 373
- Andreatta, G. &amp; Romeo, L. (1988), 'Stochastic shortest paths with recourse', Networks 18 , 193-204. 227
- A.Nedi¸ c &amp; D.P.Bertsekas (2003), 'Least squares policy evaluation algorithms with linear function approximation', Journal of Discrete Event Systems 13 , 79-110. 238, 320
- Bean, J., Birge, J. &amp; Smith, R. (1987), 'Aggregation in dynamic programming', Operations Research 35 , 215-220. 273, 321
- Bellman, R. (1957), Dynamic Programming , Princeton University Press, Princeton. 10, 14, 48, 115
- Bellman, R. (1971), Introduction to the Mathematical Theory of Control Processes, Vol. II , Academic Press, New York. 115
- Bellman, R. &amp; Dreyfus, S. (1959), 'Functional approximations and dynamic programming', Mathematical Tables and Other Aids to Computation 13 , 247-251. 142
- Benveniste, A., Metivier, M. &amp; Priouret, P. (1990), Adaptive Algorithms and Stochastic Approximations , Springer-Verlag, New York. 194
- Berg, M., Schouten, F. &amp; Jansen, J. (1998), 'Optimal batch provisioning to customers subject to a delay-limit', Management Science 44 (5), 684-697. 373
- Bernardo, J. M. &amp; Smith, A. F. M. (1994), Bayesian Theory , John Wiley and Sons, New York. 273
- Berry, D. A. &amp; Fristedt, B. (1985), Bandit Problems , Chapman and Hall, London. 292
- Bertazzi, L., Bertsekas, D. &amp; Speranza, M. G. (2000), Optimal and neuro-dynamic programming solutions for a stochastic inventory trasportation problem, Unpublished technical report, Universita Degli Studi Di Brescia. 321
- Bertsekas, D. (1976), Dynamic Programming and Stochastic Control , Academic Press, New York. 115
- Bertsekas, D. (2000), Dynamic Programming and Optimal Control , Athena Scientific, Belmont, Massachusetts. 29

- Bertsekas, D. &amp; Castanon, D. (1989), 'Adaptive aggregation methods for infinite horizon dynamic programming', IEEE Transactions on Automatic Control 34 (6), 589-598. 273, 321
- Bertsekas, D. &amp; Castanon, D. (1999), 'Rollout algorithms for stochastic scheduling problems', J. Heuristics 5 , 89-108. 227
- Bertsekas, D. &amp; Shreve, S. (1978), Stochastic Optimal Control: The Discrete Time Case , Academic Press, New York. 115
- Bertsekas, D. &amp; Tsitsiklis, J. (1991), 'An analysis of stochastic shortest path problems', Mathematics of Operations Research 16 , 580-595. 115, 227
- Bertsekas, D. &amp; Tsitsiklis, J. (1996), Neuro-Dynamic Programming , Athena Scientific, Belmont, MA. 14, 115, 142, 193, 227, 238, 273
- Bertsekas, D., Nedic, A. &amp; Ozdaglar, E. (2003), Convex Analysis and Optimization , Athena Scientific, Belmont, Massachusetts. 317, 347
- Bertsekas, D. P., Borkar, V. S. &amp; Nedich, A. (2004), Improved temporal difference methods with linear function approximation, in J. Si, A. G. Barto, W. B. Powell &amp; D. W. II, eds, 'Handbook of Learning and Approximate Dynamic Programming', IEEE Press. 238
- Birge, J. &amp; Louveaux, F. (1997), Introduction to Stochastic Programming , Springer-Verlag, New York. 405
- Birge, J. &amp; Wallace, S. W. (1988), 'A separable piecewise linear upper bound for stochastic linear programs', SIAM J. Control and Optimization 26 (3), 1-14. 405
- Birge, J. &amp; Wets, R. (1989), 'Sublinear upper bounds for stochastic programs with recourse', Mathematical Programming 43 , 131-149. 405
- Blum, J. (1954 a ), 'Approximation methods which converge with probability one', Annals of Mathematical Statistics 25 , 382-386. 180, 183
- Blum, J. (1954 b ), 'Multidimensional stochastic approximation methods', Annals of Mathematical Statistics 25 , 737-744. 193
- Boyan, J. (2002), 'Technical update: Least-squares temporal difference learning', Machine Learning 49 , 1-15. 238
- Bradtke, S. J. &amp; Barto, A. G. (1996), 'Linear least-squares algorithms for temporal difference learning', Machine Learning 22 , 33-57. 320
- Brown, R. (1959), Statistical Forecasting for Inventory Control , McGraw-Hill, New York. 193
- Brown, R. (1963), Smoothing, Forecasting and Prediction of Discrete Time Series , PrenticeHall, Englewood Cliffs, N.J. 193
- Cayley, A. (1875), 'Mathematical questions with their solutions, no. 4528', Educational Times . 22
- Chen, Z.-L. &amp; Powell, W. (1999), 'A convergent cutting-plane and partial-sampling algorithm for multistage linear programs with recourse', Journal of Optimization Theory and Applications 103 (3), 497-524. 405

- Cheung, R. K.-M. &amp; Powell, W. B. (2000), 'SHAPE: A stochastic hybrid approximation procedure for two-stage stochastic programs', Operations Research 48 (1), 73-79. 320, 399
- Chow, G. (1997), Dynamic Economics , Oxford University Press, New York. 115
- Crites, R. &amp; Barto, A. (1994), 'Elevator group control using multiple reinforcement learning agents', Machine Learning 33 , 235-262. 227
- Dantzig, G. &amp; Ferguson, A. (1956), 'The allocation of aircrafts to routes: An example of linear programming under uncertain demand', Management Science 3 , 45-73. 405
- Darken, C. &amp; Moody, J. (1991), Note on learning rate schedules for stochastic optimization, in Lippmann, Moody &amp; Touretzky, eds, 'Advances in Neural Information Processing Systems 3', pp. 1009-1016. 193
- Darken, C., Chang, J. &amp; Moody, J. (1992), 'Learning rate schedules for faster stochastic gradient search', Neural Networks for Signal Processing 2 - Proceedings of the 1992 IEEE Workshop . 193
- Dayan, P. (1992), 'The convergence of td( λ ) for general λ ', Machine Learning 8 , 341-362. 227
- de Farias, D. &amp; Van Roy, B. ((to appear)), 'The linear programming approach to approximate dynamic programming', Operations Research 00 , 000-000. 320
- Deb, R. (1978 a ), 'Optimal control of batch service queues with switching costs', Advances in Applied Probability 8 , 177-194. 373
- Deb, R. (1978 b ), 'Optimal dispatching of a finite capacity shuttle', Management Science 24 , 1362-1372. 373
- Deb, R. (1984), 'Optimal control of bulk queues with compound poisson arrivals and batch service', Operations Research 21 , 227-245. 373
- Deb, R. &amp; Schmidt, C. (1987), 'Optimal average cost policies for the two-terminal shuttle', Management Science 33 , 662-669. 373
- Deb, R. &amp; Serfozo, R. (1973), 'Optimal control of batch service queues', Advances in Applied Probability 5 , 340-361. 373
- Derman, C. (1970), Finite State Markovian Decision Processes , Academic Press, New York. 14, 115
- Doob, J. L. (1953), Stochastic Processes , John Wiley &amp; Sons. 399
- Douglas, S. &amp; Mathews, V. (1995), 'Stochastic gradient adaptive step size algorithms for adaptive filtering', Proc. International Conference on Digital Signal Processing, Limassol, Cyprus 1 , 142-147. 194
- Dreyfus, S. &amp; Law, A. M. (1977), The art and theory of dynamic programming , Academic Press, New York. 115
- Duff, M. O. (1995), Q-learning for bandit problems, Technical report, Department of Computer Science, University of Massachusetts, Amherst, MA. 292
- Duff, M. O. &amp; Barto, A. G. (2003), Local bandit approximation for optimal learning problems, Technical report, Department of Computer Science, University of Massachusetts, Amherst, MA. 292

- Dvoretzky, A. (1956), On stochastic approximation, in J. Neyman, ed., 'Proc. 3 rd Berkeley Sym. on Math. Stat. and Prob.', Berkeley: University of California Press, pp. 39-55. 180, 193
- Dynkin, E. B. (1979), Controlled Markov Processes , Springer-Verlag, New York. 14, 115
- Ermoliev, Y. (1971), 'The general problem in stochastic programming', Kibernetika . 180
- Ermoliev, Y. (1983), 'Stochastic quasigradient methods and their application to system optimization', Stochastics 9 , 1-36. 180
- Ermoliev, Y. (1988), Stochastic quasigradient methods, in Y. Ermoliev &amp; R. Wets, eds, ' Numerical Techniques for Stochastic Optimization ', Springer-Verlag, Berlin. 180, 405
- Even-Dar, E. &amp; Mansour, Y. (2004), 'Learning rates for q-learning', Journal of Machine Learning Research 5 , 1-25. 194
- Frank, H. (1969), 'Shortest paths in probabilistic graphs', Operations Research 17 , 583-599. 227
- Frieze, A. &amp; Grimmet, G. (1985), 'The shortest path problem for graphs with random arc lengths', Discrete Applied Mathematics 10 , 57-77. 227
- Gaivoronski, A. (1988), Stochastic quasigradient methods and their implementation, in Y. Ermoliev &amp; R. Wets, eds, 'Numerical Techniques for Stochastic Optimization', SpringerVerlag, Berlin. 163, 180, 193
- Gardner, E. S. (1983), 'Automatic monitoring of forecast errors', Journal of Forecasting 2 , 1-21. 193
- George, A. &amp; Powell, W. B. (2004), Adaptive stepsizes for recursive estimation in dynamic programming, Technical report, Department of Operations Research and Financial Engineering, Princeton University. 174, 178, 179
- George, A., Powell, W. B. &amp; Kulkarni, S. (2003), The statistics of hierarchical aggregation for multiattribute resource management, Technical report, Department of Operations Research and Financial Engineering, Princeton University. 273
- George, A., Powell, W. B. &amp; Kulkarni, S. (2005), The statistics of hierarchical aggregation for multiattribute resource management, Technical report, Department of Operations Research and Financial Engineering, Princeton University. 248
- Giffin, W. (1971), Introduction to Operations Engineering , R. D. Irwin, Inc., Homewood, Illinois. 193
- Gittins, J. (1979), 'Bandit processes and dynamic allocation indices', Journal of the Royal Statistical Society, Series B 14 , 148-177. 294
- Gittins, J. (1981), 'Multiserver scheduling of jobs with increasing completion times', Journal of Applied Probability 16 , 321-324. 294
- Gittins, J. (1989), Multi-Armed Bandit Allocation Indices , John Wiley and Sons, New York. 289, 294
- Gittins, J. C. &amp; Jones, D. M. (1974), A dynamic allocation index for the sequential design of experiments, in J. Gani, ed., 'Progress in Statistics', pp. 241-266. 286, 294

- Gladyshev, E. G. (1965), 'On stochastic approximation', Theory of Prob. and its Appl. 10 , 275-278. 399
- Godfrey, G. &amp; Powell, W. B. (2002 a ), 'An adaptive, dynamic programming algorithm for stochastic resource allocation problems I: Single period travel times', Transportation Science 36 (1), 21-39. 423, 427
- Godfrey, G. &amp; Powell, W. B. (2002 b ), 'An adaptive, dynamic programming algorithm for stochastic resource allocation problems II: Multi-period travel times', Transportation Science 36 (1), 40-54. 427
- Godfrey, G. A. &amp; Powell, W. B. (2001), 'An adaptive, distribution-free approximation for the newsvendor problem with censored demands, with applications to inventory and distribution problems', Management Science 47 (8), 1101-1112. 320
- Golub, G. &amp; Loan, C. V. (1996), Matrix Computations , John Hopkins University Press, Baltimore, Maryland. 272, 320
- Guestrin, C., Koller, D. &amp; Parr, R. (2003), 'Efficient solution algorithms for factored MDPs', Journal of Artificial Intelligence Research 19 , 399-468. 227
- Hastie, T., Tibshirani, R. &amp; Friedman, J. (2001), The Elements of Statistical Learning , Springer series in Statistics, New York, NY. 190, 193
- Heyman, D. &amp; Sobel, M. (1984), Stochastic Models in Operations Research, Volume II: Stochastic Optimization , McGraw Hill, New York. 14, 115
- Higle, J. &amp; Sen, S. (1991), 'Stochastic decomposition: An algorithm for two stage linear programs with recourse', Mathematics of Operations Research 16 (3), 650-669. 405
- Holt, C., Modigliani, F., Muth, J. &amp; Simon, H. (1960), Planning, Production, Inventories and Work Force , Prentice-Hall, Englewood Cliffs, NJ. 193
- Howard, R. (1971), Dynamic Probabilistic Systems, Volume II: Semimarkov and Decision Processes , John Wiley and Sons, New York. 14, 115
- Infanger, G. (1994), Planning under Uncertainty: Solving Large-scale Stochastic Linear Programs , The Scientific Press Series, Boyd &amp; Fraser, New York. 405
- Infanger, G. &amp; Morton, D. (1996), 'Cut sharing for multistage stochastic linear programs with interstate dependency', Mathematical Programming 75 , 241-256. 405
- Jaakkola, T., Jordan, M. I. &amp; Singh, S. P. (1994), Convergence of stochastic iterative dynamic programming algorithms, in J. D. Cowan, G. Tesauro &amp; J. Alspector, eds, 'Advances in Neural Information Processing Systems', Vol. 6, Morgan Kaufmann Publishers, Inc., pp. 703-710. 193, 227, 320
- Jacobs, R. A. (1988), 'Increased rate of convergence through learning rate adaptation', Neural Networks 1 , 295 - 307. 194
- Kall, P. &amp; Wallace, S. (1994), Stochastic Programming , John Wiley and Sons, New York. 405
- Kesten, H. (1958), 'Accelerated stochastic approximation', The Annals of Mathematical Statistics 29 (4), 41-59. 193

- Kiefer, J. &amp; Wolfowitz, J. (1952), 'Stochastic estimation of the maximum of a regression function', Annals Math. Stat. 23 , 462-466. 193
- Kmenta, J. (1997), Elements of Econometrics , second edn, University of Michigan Press, Ann Arbor, Michigan. 193
- Kushner, H. J. &amp; Clark, S. (1978), Stochastic Approximation Methods for Constrained and Unconstrained Systems , Springer-Verlag, New York. 193
- Kushner, H. J. &amp; Yin, G. G. (1997), Stochastic Approximation Algorithms and Applications , Springer-Verlag, New York. 180
- Lai, T. L. &amp; Robbins, H. (1985), 'Asymptotically efficient adaptive allocation rules', Advances in Applied Mathematics 6 , 4-22. 294
- LeBlanc, M. &amp; Tibshirani, R. (1996), 'Combining estimates in regression and classification', Journal of the American Statistical Association 91 , 1641-1650. 273
- Leslie Pack Kaelbling, M. L. L. &amp; Moore, A. W. (1996), 'Reinforcement learning: A survey', Journal of Artifcial Intelligence Research 4 , 237-285. 142
- Ljung, l. &amp; Soderstrom, T. (1983), Theory and Practice of Recursive Identification , MIT Press, Cambridge, MA. 320
- Luus, R. (2000), Iterative Dynamic Programming , Chapman &amp; Hall/CRC, New York. 321
- Mathews, V. J. &amp; Xie, Z. (1993), 'A stochastic gradient adaptive filter with gradient adaptive step size', IEEE Transactions on Signal Processing 41 , 2075-2087. 194
- Mendelssohn, R. (1982), 'An iterative aggregation procedure for Markov decision processes', Operations Research 30 (1), 62-73. 273, 321
- Mirozahmedov, F. &amp; Uryasev, S. P. (1983), 'Adaptive stepsize regulation for stochastic optimization algorithm', Zurnal vicisl. mat. i. mat. fiz. 23 6 , 1314-1325. 162, 193
- Mulvey, J. M. &amp; Ruszczy´ nski, A. J. (1995), 'A new scenario decomposition method for large-scale stochastic optimization', Operations Research 43 (3), 477-490. 405
- Neuts, M. (1967), 'A general class of bulk queues with Poisson input', Ann. Math. Stat. 38 , 759-770. 373
- Neveu, J. (1975), Discrete Parameter Martingales , North Holland, Amsterdam. 399, 400
- Papadaki, K. &amp; Powell, W. B. (2002), 'A monotone adaptive dynamic programming algorithm for a stochastic batch service problem', European Journal of Operational Research 142 (1), 108-127. 373
- Papadaki, K. &amp; Powell, W. B. (2003), 'An adaptive dynamic programming algorithm for a stochastic multiproduct batch dispatch problem', Naval Research Logistics 50 (7), 742-769. 359, 360, 362, 373
- Pflug, G. C. (1988), Stepsize rules, stopping times and their implementation in stochastic quasi-gradient algorithms, in 'Numerical Techniques for Stochastic Optimization', Springer-Verlag, pp. 353-372. 193

- Pflug, G. C. (1996), Optimization of Stochastic Models: The Interface Between Simulation and Optimization , Kluwer International Series in Engineering and Computer Science: Discrete Event Dynamic Systems, Kluwer Academic Publishers, Boston. 142, 193
- Powell, W. B. &amp; Humblet, P. (1986), 'The bulk service queue with a general control strategy: Theoretical analysis and a new computational procedure', Operations Research 34 (2), 267275. 373
- Powell, W. B., Ruszczy´ nski, A. &amp; Topaloglu, H. (2004), 'Learning algorithms for separable approximations of stochastic optimization problems', Mathematics of Operations Research 29 (4), 814-836. 309, 320, 321, 399, 427
- Powell, W. B., Shapiro, J. A. &amp; Sim˜ ao, H. P. (2001), A representational paradigm for dynamic resource transformation problems, in R. F. C. Coullard &amp; J. H. Owens, eds, 'Annals of Operations Research', J.C. Baltzer AG, pp. 231-279. 77
- Powell, W. B., Shapiro, J. A. &amp; Sim˜ ao, H. P. (2002), 'An adaptive dynamic programming algorithm for the heterogeneous resource allocation problem', Transportation Science 36 (2), 231-249. 427
- Psaraftis, H. &amp; Tsitsiklis, J. (1990), Dynamic shortest paths with Markovian arc costs, Preprint. 227
- Puterman, M. L. (1994), Markov Decision Processes , John Wiley and Sons, Inc., New York. 14, 29, 48, 115, 373
- Robbins, H. &amp; Monro, S. (1951), 'A stochastic approximation method', Annals of Math. Stat. 22 , 400-407. 180, 183, 193
- Rockafellar, R. &amp; Wets, R. (1991), 'Scenarios and policy aggregation in optimization under uncertainty', Mathematics of Operations Research 16 (1), 119-147. 405
- Rockafellar, R. T. (1976), 'Augmented Lagrangians and applications of the proximal point algorithm in convex programming', Math. of Operations Research 1 , 97-116. 385
- Rogers, D., Plante, R., Wong, R. &amp; Evans, J. (1991), 'Aggregation and disaggregation techniques and methodology in optimization', Operations Research 39 (4), 553-582. 321
- Ross, S. (1983), Introduction to Stochastic Dynamic Programming , Academic Press, New York. 14, 29, 115
- Ruszczy´ nski, A. (1980 a ), 'Feasible direction methods for stochastic programming problems', Math. Programming 19 , 220-229. 180, 385
- Ruszczy´ nski, A. (1980 b ), 'Feasible direction methods for stochastic programming problems', Mathematical Programming 19 , 220-229. 405
- Ruszczy´ nski, A. (1987), 'A linearization method for nonsmooth stochastic programming problems', Mathematics of Operations Research 12 (1), 32-49. 180, 405
- Ruszczy´ nski, A. &amp; Syski, W. (1986), 'A method of aggregate stochastic subgradients with on-line stepsize rules for convex stochastic programming problems', Mathematical Programming Study 28 , 113-131. 162, 193
- Schweitzer, P., Puterman, M. &amp; Kindle, K. (1985), 'Iterative aggregation-disaggregation procedures for discounted semi-Markov reward processes', Operations Research 33 (3), 589605. 321

- Sen, S. &amp; Higle, J. (1999), 'An introductory tutorial on stochastic linear programming models', Interfaces 29 (2), 33-61. 405
- Shiryaev, A. (1996), Probability Theory , Vol. 95 of Graduate Texts in Mathematics , SpringerVerlag, New York. 315, 317
- Si, J., Barto, A. G., Powell, W. B. &amp; D. Wunsch II, e. (2004), Handbook of Learning and Approximate Dynamic Programming , IEEE Press, New York. 142
- Sigal, C., Pritsker, A. &amp; Solberg, J. (1980), 'The stochastic shortest route problem', Operations Research 28 (5), 1122-1129. 227
- Singh, S., Jaakkola, T. &amp; Jordan, M. I. (1995), Reinforcement learning with soft state aggregation, in G. Tesauro, D. Touretzky &amp; T. K. Leen, eds, 'Advances in Neural Information Processing Systems 7', MIT Press. 321
- Spall, J. C. (2003), Introduction to stochastic search and optimization: estimation, simulation and control , John Wiley and Sons, Inc., Hoboken, NJ. 193
- Spivey, M. &amp; Powell, W. B. (2004), 'The dynamic assignment problem', Transportation Science 38 (4), 399-419. 427
- Stengel, R. (1994), Optimal Control and Estimation , Dover Publications, New York, NY. 194
- Stokey, N. L. &amp; R. E. Lucas, J. (1989), Recursive Methods in Dynamic Economics , Harvard University Press, Cambridge. 115
- Sutton, R. (1988), 'Learning to predict by the methods of temporal differences', Machine Learning 3 , 9-44. 142, 216, 227
- Sutton, R. &amp; Barto, A. (1998), Reinforcement Learning , The MIT Press, Cambridge, Massachusetts. 142
- Taylor, H. (1967), 'Evaluating a call option and optimal timing strategy in the stock market', Management Science 12 , 111-120. 14
- Taylor, H. M. (1990), Martingales and Random Walks , Vol. 2, Elsevier Science Publishers B.V.,, chapter 3. 399
- Topaloglu, H. &amp; Powell, W. B. (2003), 'An algorithm for approximating piecewise linear concave functions from sample gradients', Operations Research Letters 31 (1), 66-76. 320
- Topaloglu, H. &amp; Powell, W. B. (to appear), 'Dynamic programming approximations for stochastic, time-staged integer multicommodity flow problems', Informs Journal on Computing . 424, 427
- Topkins, D. M. (1978), 'Minimizing a submodular function on a lattice', Operations Research 26 , 305-321. 373
- Trigg, D. (1964), 'Monitoring a forecasting system', Operations Research Quarterly 15 (3), 271-274. 193
- Trigg, D. &amp; Leach, A. (1967), 'Exponential smoothing with an adaptive response rate', Operations Research Quarterly 18 (1), 53-59. 164

- Tsitsiklis, J. &amp; Van Roy, B. (1997), 'An analysis of temporal-difference learning with function approximation', IEEE Transactions on Automatic Control 42 , 674-690. 237, 238, 320
- Tsitsiklis, J. N. (1994), 'Asynchronous stochastic approximation and q-learning', Machine Learning 16 , 185-202. 227
- Tsitsiklis, J. N. &amp; Van Roy, B. (1996), 'Feature-based methods for large scale dynamic programming', Machine Learning 22 , 59-94. 142, 321
- Van Roy, B. (2001), Neuro-dynamic programming: Overview and recent trends, in E. Feinberg &amp; A. Shwartz, eds, 'Handbook of Markov Decision Processes: Methods and Applications', Kluwer, Boston. 142
- Van Slyke, R. &amp; Wets, R. (1969), 'L-shaped linear programs with applications to optimal control and stochastic programming', SIAM Journal of Applied Mathematics 17 (4), 638663. 405
- Wallace, S. (1986 a ), 'Decomposing the requirement space of a transportation problem', Math. Prog. Study 28 , 29-47. 405
- Wallace, S. (1986 b ), 'Solving stochastic programs with network recourse', Networks 16 , 295317. 405
- Wallace, S. W. (1987), 'A piecewise linear upper bound on the network recourse function', Mathematical Programming 38 , 133-146. 405
- Wasan, M. (1969), Stochastic approximations, in J. T. J.F.C. Kingman, F. Smithies &amp; T. Wall, eds, 'Cambridge Transactions in Math. and Math. Phys. 58', Cambridge University Press, Cambridge. 193
- Watkins, C. (1989), Learning from delayed rewards, Ph.d. thesis, Cambridge University, Cambridge, UK. 227
- Watkins, C. &amp; Dayan, P. (1992), 'Q-learning', Machine Learning 8 , 279-292. 227
- Weber, R. (1992), 'On the gittins index for multiarmed bandits', The Annals of Applied Probability 2 (4), 1024-1033. 291
- Werbos, P. (1990), A menu of designs for reinforcement learning over time, in R. S. W.T. Miller &amp; P. Werbos, eds, 'Neural Networks for Control', MIT PRess, Cambridge, MA, pp. 67-96. 142
- Werbos, P. (1992 a ), Neurocontrol and supervised learning: an overview and evaluation, in D. A. White &amp; D. A. Sofge, eds, 'Handbook of Intelligent Control', Von Nostrand Reinhold, New York, NY, pp. 65-86. 142
- Werbos, P. J. (1987), 'Building and understanding adaptive systems: A statistical/numerical approach to factory automation and brain research', IEEE Transactions on Systems, Man and Cybernetics . 142
- Werbos, P. J. (1992 b ), Approximate dynamic programming for real-time control and neural modelling, in D. J. White &amp; D. A. Sofge, eds, 'Handbook of Intelligent Control: Neural, Fuzzy, and Adaptive Approaches'. 142
- Werbos, P. J. (1992 c ), Neurocontrol and supervised learning: an overview and valuation, in D. A. White &amp; D. A. Sofge, eds, 'Handbook of Intelligent Control: Neural, Fuzzy, and Adaptive Approaches'. 142

- Wets, R. (1989), Stochastic programming, in 'Handbooks in Operations Research and Management Science: Optimization', Vol. 1, Elsevier Science Publishers B.V., Amsterdam, pp. Volume 1, Chapter 8. 405
- White, C. C. (1991), 'A survey of solution techniques for the partially observable Markov decision process', Annals of operations research 32 , 215-230. 142
- White, D. A. &amp; Sofge, D. A. (1992), Handbook of Intelligent Control , Von Nostrand Reinhold, New York, NY. 142
- White, D. J. (1969), Dynamic Programming , Holden-Day, San Francisco. 115
- Whitt, W. (1978), 'Approximations of dynamic programs I', Mathematics of Operations Research 3 , 231-243. 273
- Whittle, P. (1982), Optimization over time: Dynamic programming and stochastic control Volume I , John Wiley and Sons, New York. 29, 291
- Winters, P. R. (1960), 'Forecasting sales by exponentially weighted moving averages', Management Science 6 , 324-342. 193
- Wright, S. E. (1994), 'Primal-dual aggregation and disaggregation for stochastic linear programs', Mathematics of Operations Research 19 , 893-908. 321
- Yang, Y. (1999), 'Adaptive regression by mixing', Journal of the American Statistical Association . 273
- Young, P. (1984), Recursive Estimation and Time-Series Analysis , Springer-Verlag, Berlin, Heidelberg. 320
- Zipkin, P. (1980 a ), 'Bounds for row-aggregation in linear programming', Operations Research 28 , 903-916. 273, 321
- Zipkin, P. (1980 b ), 'Bounds on the effect of aggregating variables in linear programming', Operations Research 28 , 403-418. 273, 321